from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

TRAINING_CHECKPOINT_VERSION = 1


def capture_random_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_random_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])


def random_state_fingerprint(state: dict[str, Any]) -> str:
    payload = repr(state["python"]).encode() + repr(state["numpy"][:2]).encode()
    payload += state["numpy"][1].tobytes() + state["torch_cpu"].cpu().numpy().tobytes()
    for value in state["torch_cuda_all"]: payload += value.cpu().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def optimizer_group_contract(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    group_names: Sequence[str],
) -> list[dict[str, Any]]:
    if len(group_names) != len(optimizer.param_groups):
        raise ValueError("optimizer group_names must match param_groups")
    names_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    result = []
    for group_name, group in zip(group_names, optimizer.param_groups):
        names, shapes, trainable = [], [], []
        for parameter in group["params"]:
            name = names_by_id.get(id(parameter))
            if name is None: raise ValueError("optimizer contains parameter outside model")
            names.append(name); shapes.append(list(parameter.shape)); trainable.append(bool(parameter.requires_grad))
        fingerprint = hashlib.sha256(json.dumps(
            {"group_name": group_name, "parameter_names": names, "parameter_shapes": shapes},
            sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()
        result.append({
            "group_name": group_name, "parameter_names": names, "parameter_shapes": shapes,
            "requires_grad": trainable, "learning_rate": float(group["lr"]),
            "weight_decay": float(group.get("weight_decay", 0.0)),
            "parameter_count": sum(int(parameter.numel()) for parameter in group["params"]),
            "fingerprint": fingerprint,
            "hyperparameters": {key: value for key, value in group.items() if key != "params"},
        })
    return result


def save_full_training_checkpoint(
    path: str | Path, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    optimizer_group_names: Sequence[str], scheduler: Any, scaler: Any,
    training_state: dict[str, Any], data_state: dict[str, Any], method_state: dict[str, Any],
) -> dict[str, Any]:
    random_state = capture_random_state()
    contract = optimizer_group_contract(model, optimizer, optimizer_group_names)
    payload = {
        "training_checkpoint_version": TRAINING_CHECKPOINT_VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": {
            "class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
            "state_dict": optimizer.state_dict(), "parameter_groups": contract,
        },
        "scheduler_state": {
            "scheduler_enabled": scheduler is not None,
            "class": None if scheduler is None else f"{type(scheduler).__module__}.{type(scheduler).__qualname__}",
            "state_dict": None if scheduler is None else scheduler.state_dict(),
        },
        "amp_state": {
            "scaler_enabled": scaler is not None,
            "state_dict": None if scaler is None else scaler.state_dict(),
        },
        "training_state": training_state, "random_state": random_state,
        "random_state_fingerprint": random_state_fingerprint(random_state),
        "data_state": data_state, "method_state": method_state,
    }
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); torch.save(payload, path)
    return payload


def validate_full_training_checkpoint(
    checkpoint: dict[str, Any], *, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    optimizer_group_names: Sequence[str], expected_method_state: dict[str, Any],
    expected_data_state: dict[str, Any],
) -> None:
    if checkpoint.get("training_checkpoint_version") != TRAINING_CHECKPOINT_VERSION:
        raise ValueError("unsupported training checkpoint version")
    current = optimizer_group_contract(model, optimizer, optimizer_group_names)
    saved = checkpoint["optimizer_state"]["parameter_groups"]
    if current != saved: raise ValueError("optimizer parameter-group contract mismatch")
    for key, value in expected_method_state.items():
        if checkpoint["method_state"].get(key) != value: raise ValueError(f"method state mismatch: {key}")
    for key, value in expected_data_state.items():
        if checkpoint["data_state"].get(key) != value: raise ValueError(f"data state mismatch: {key}")


def load_full_training_checkpoint(
    path: str | Path, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    optimizer_group_names: Sequence[str], scheduler: Any, scaler: Any,
    expected_method_state: dict[str, Any], expected_data_state: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    validate_full_training_checkpoint(
        checkpoint, model=model, optimizer=optimizer, optimizer_group_names=optimizer_group_names,
        expected_method_state=expected_method_state, expected_data_state=expected_data_state,
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"]["state_dict"])
    scheduler_state = checkpoint["scheduler_state"]
    if scheduler_state["scheduler_enabled"] != (scheduler is not None): raise ValueError("scheduler enabled state mismatch")
    if scheduler is not None: scheduler.load_state_dict(scheduler_state["state_dict"])
    amp_state = checkpoint["amp_state"]
    if amp_state["scaler_enabled"] != (scaler is not None): raise ValueError("GradScaler enabled state mismatch")
    if scaler is not None: scaler.load_state_dict(amp_state["state_dict"])
    restore_random_state(checkpoint["random_state"])
    return checkpoint


def mark_legacy_model_only_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return {
        "resume_capability": "model_weights_only",
        "available_fields": sorted(checkpoint),
        "missing_full_training_fields": [
            "optimizer_state", "scheduler_state", "amp_state", "training_state", "random_state", "data_state",
        ],
    }
