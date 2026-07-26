from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scene.two_outfit_episode_scheduler import OutfitBalancedEpisodeScheduler
from utils.full_training_checkpoint_utils import (
    load_full_training_checkpoint,
    save_full_training_checkpoint,
)


TWO_OUTFIT_CHECKPOINT_CONTRACT_VERSION = 1


def config_fingerprint(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def required_two_outfit_checkpoint_fields() -> dict[str, list[str]]:
    return {
        "root": [
            "model_state",
            "optimizer_state",
            "scheduler_state",
            "amp_state",
            "training_state",
            "random_state",
            "data_state",
            "method_state",
        ],
        "training_state": [
            "global_step",
            "balanced_episode_scheduler",
            "per_outfit_update_counts",
            "per_view_update_counts",
            "current_cycle_position",
            "cycle_count",
        ],
        "data_state": ["manifest_sha256", "base_fingerprint"],
        "method_state": [
            "two_outfit_checkpoint_contract_version",
            "config",
            "config_sha256",
            "trainable_fingerprint",
        ],
    }


def build_two_outfit_checkpoint_states(
    *,
    balanced_scheduler: OutfitBalancedEpisodeScheduler,
    global_step: int,
    config: Mapping[str, Any],
    manifest_sha256: str,
    base_fingerprint: str,
    trainable_fingerprint: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if int(global_step) != balanced_scheduler.index:
        raise ValueError("global_step must equal the next balanced scheduler index")
    scheduler_state = balanced_scheduler.state_dict()
    training_state = {
        "global_step": int(global_step),
        "balanced_episode_scheduler": scheduler_state,
        "per_outfit_update_counts": dict(scheduler_state["outfit_update_counts"]),
        "per_view_update_counts": dict(scheduler_state["per_view_update_counts"]),
        "current_cycle_position": int(scheduler_state["current_cycle_position"]),
        "cycle_count": int(scheduler_state["epoch_or_cycle_count"]),
    }
    data_state = {
        "manifest_sha256": str(manifest_sha256),
        "base_fingerprint": str(base_fingerprint),
    }
    method_state = {
        "two_outfit_checkpoint_contract_version": TWO_OUTFIT_CHECKPOINT_CONTRACT_VERSION,
        "config": json.loads(json.dumps(config, default=str)),
        "config_sha256": config_fingerprint(config),
        "trainable_fingerprint": str(trainable_fingerprint),
    }
    return training_state, data_state, method_state


def save_two_outfit_checkpoint(
    path: str | Path,
    *,
    model: Any,
    optimizer: Any,
    optimizer_group_names: Sequence[str],
    lr_scheduler: Any,
    scaler: Any,
    balanced_scheduler: OutfitBalancedEpisodeScheduler,
    global_step: int,
    config: Mapping[str, Any],
    manifest_sha256: str,
    base_fingerprint: str,
    trainable_fingerprint: str,
) -> dict[str, Any]:
    """Save a future formal-run checkpoint; this helper never constructs an optimizer."""

    training_state, data_state, method_state = build_two_outfit_checkpoint_states(
        balanced_scheduler=balanced_scheduler,
        global_step=global_step,
        config=config,
        manifest_sha256=manifest_sha256,
        base_fingerprint=base_fingerprint,
        trainable_fingerprint=trainable_fingerprint,
    )
    return save_full_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        optimizer_group_names=optimizer_group_names,
        scheduler=lr_scheduler,
        scaler=scaler,
        training_state=training_state,
        data_state=data_state,
        method_state=method_state,
    )


def load_two_outfit_checkpoint(
    path: str | Path,
    *,
    model: Any,
    optimizer: Any,
    optimizer_group_names: Sequence[str],
    lr_scheduler: Any,
    scaler: Any,
    balanced_scheduler: OutfitBalancedEpisodeScheduler,
    config: Mapping[str, Any],
    manifest_sha256: str,
    base_fingerprint: str,
    trainable_fingerprint: str,
) -> dict[str, Any]:
    expected_data = {
        "manifest_sha256": str(manifest_sha256),
        "base_fingerprint": str(base_fingerprint),
    }
    expected_method = {
        "two_outfit_checkpoint_contract_version": TWO_OUTFIT_CHECKPOINT_CONTRACT_VERSION,
        "config_sha256": config_fingerprint(config),
        "trainable_fingerprint": str(trainable_fingerprint),
    }
    checkpoint = load_full_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        optimizer_group_names=optimizer_group_names,
        scheduler=lr_scheduler,
        scaler=scaler,
        expected_method_state=expected_method,
        expected_data_state=expected_data,
    )
    training_state = checkpoint["training_state"]
    balanced_scheduler.load_state_dict(training_state["balanced_episode_scheduler"])
    if balanced_scheduler.index != int(training_state["global_step"]):
        raise ValueError("restored scheduler index differs from global_step")
    if dict(balanced_scheduler.outfit_update_counts) != training_state["per_outfit_update_counts"]:
        raise ValueError("restored per-outfit counts differ from checkpoint")
    if dict(balanced_scheduler.view_update_counts) != training_state["per_view_update_counts"]:
        raise ValueError("restored per-view counts differ from checkpoint")
    return checkpoint
