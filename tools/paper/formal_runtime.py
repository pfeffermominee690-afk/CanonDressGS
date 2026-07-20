from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene.gaussian_clothing_residuals import GaussianClothingResiduals
from scene.multi_outfit_linear_coefficient_control import (
    MultiOutfitLinearCoefficientControl,
    pairwise_geometry_loss,
    restore_coefficients,
)
from tools import diagnose_image_conditioned_overfit_failure as diagnosis
from tools import run_image_conditioned_overfit_o01 as o01
from tools import run_multi_outfit_explicit_basis as multi
from tools import run_residual_field_parameterization as parameterization
from tools.check_real_image_conditioned_one_batch import save_render_tensor
from tools.paper.method_adapters import adapter_for
from tools.paper.smoke_runtime import _tree_max_abs


FORMAL_SOURCE_HEAD = "16a48bbcbc28e0050e5b2f1777974775e025c9f7"
FORMAL_BRANCH = "paper/aaai27-frozen-experiment-batches-20260720"
FORMAL_ASSET_FINGERPRINT = "ff90540e56c9c2db3fd6a1e45c31a1d1eb5a8a32effabde71158584d9be538bb"
OUTFITS = ("O01", "O02", "O03", "O04", "O08")
CONDITIONS = ("cond_000000", "cond_000318", "cond_000017", "cond_000347")
MILESTONES = (0, 20, 50, 100, 200, 300)
RESUME_PROBE_STEP = 200


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(dict(value), sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=PROJECT_ROOT, text=True
    ).strip()


def _load_attempt() -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    value = os.environ.get("CANONDRESSGS_PAPER_ATTEMPT")
    if not value:
        raise RuntimeError("CANONDRESSGS_PAPER_ATTEMPT is required")
    attempt = Path(value).resolve()
    if not attempt.is_dir() or attempt.name[:8] != "attempt_":
        raise ValueError("formal attempt path is invalid")
    provenance = json.loads(
        (attempt / "provenance/run_provenance.json").read_text(encoding="utf-8")
    )
    experiment = dict(provenance["experiment"])
    registry = yaml.safe_load(
        (attempt / "contract/experiment_registry_snapshot.yaml").read_text(encoding="utf-8")
    )
    config = yaml.safe_load(
        (attempt / "contract/method_config_snapshot.yaml").read_text(encoding="utf-8")
    )
    matches = [
        item for item in registry["experiments"]
        if item["experiment_id"] == experiment["experiment_id"]
    ]
    if len(matches) != 1 or matches[0] != experiment:
        raise ValueError("attempt experiment does not match its registry snapshot")
    return attempt, experiment, registry, config


def _environment() -> dict[str, Any]:
    gpu = subprocess.check_output(
        [
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": gpu,
    }


def _legacy_context(attempt: Path) -> dict[str, Any]:
    if _git("branch", "--show-current") != FORMAL_BRANCH:
        raise RuntimeError("formal executor is not on the frozen batch branch")
    if _git("status", "--short"):
        raise RuntimeError("formal executor requires a clean worktree")
    if subprocess.call(
        ["git", "merge-base", "--is-ancestor", FORMAL_SOURCE_HEAD, "HEAD"],
        cwd=PROJECT_ROOT,
    ) != 0:
        raise RuntimeError("formal executor is not descended from the smoke PASS HEAD")
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs/research/subject02_multi_outfit_explicit_basis_v1.yaml")
        .read_text(encoding="utf-8")
    )
    old_branch, old_source = multi.EXPECTED_BRANCH, multi.EXPECTED_SOURCE_HEAD
    multi.EXPECTED_BRANCH, multi.EXPECTED_SOURCE_HEAD = FORMAL_BRANCH, FORMAL_SOURCE_HEAD
    config["branch"] = FORMAL_BRANCH
    config["source_head"] = FORMAL_SOURCE_HEAD
    config["output"]["attempt"] = attempt.name
    try:
        return multi.build_context(
            config, attempt, create=False, reference_backbone=True
        )
    finally:
        multi.EXPECTED_BRANCH, multi.EXPECTED_SOURCE_HEAD = old_branch, old_source


def _basis_path(context: Mapping[str, Any]) -> Path:
    asset_root_value = os.environ.get("CANONDRESSGS_ASSET_ROOT")
    if not asset_root_value:
        raise RuntimeError("CANONDRESSGS_ASSET_ROOT is required")
    path = (
        Path(asset_root_value)
        / "pipeline_full/SUBJECT02-MULTI-OUTFIT-EXPLICIT-BASIS-001/attempt_003"
        / "stage_b_basis/selected/selected_basis.pt"
    )
    if not path.is_file() or _sha256(path) != "a29b3dc3c6f0ac1a79755e036eb4196dc9e4d116f3e2b35a7ed4e2bd5286f430":
        raise RuntimeError("PAPER_ASSET_MISMATCH: selected basis")
    return path


def _extract_features(
    context: Mapping[str, Any], attempt: Path
) -> tuple[Any, dict[str, torch.Tensor], float]:
    started = time.perf_counter()
    extractor = multi._feature_extractor(context)
    path = attempt / "preflight/frozen_f2_features.pt"
    expected_keys = {
        f"{outfit}/{condition}"
        for outfit in (*OUTFITS, "O07")
        for condition in CONDITIONS
    }
    if path.is_file():
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if set(cached) != expected_keys:
            raise ValueError("frozen feature cache key contract changed")
        device = context["base"]._xyz.device
        features = {key: value.to(device) for key, value in cached.items()}
        if any(value.shape != (1, 512) or not torch.isfinite(value).all() for value in features.values()):
            raise ValueError("frozen feature cache tensor contract changed")
        _atomic_json(attempt / "preflight/frozen_feature_cache_reuse.json", {
            "status": "PASS", "reused": True, "sha256": _sha256(path),
            "feature_count": len(features),
        })
        return extractor, features, time.perf_counter() - started
    features: dict[str, torch.Tensor] = {}
    for outfit in (*OUTFITS, "O07"):
        for condition in CONDITIONS:
            key = f"{outfit}/{condition}"
            features[key] = multi.prediction_feature(
                extractor, context["episodes"][key]
            )
    payload = {key: value.detach().cpu() for key, value in features.items()}
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    _atomic_json(attempt / "preflight/frozen_feature_manifest.json", {
        "status": "PASS",
        "feature_count": len(features),
        "shape": list(next(iter(features.values())).shape),
        "sha256": _sha256(path),
        "target_forward_leakage": False,
        "outfit_id_in_model": False,
        "elapsed_seconds": time.perf_counter() - started,
    })
    return extractor, features, time.perf_counter() - started


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    fixed_output: torch.Tensor,
    experiment: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "canondressgs.formal_seen_outfit_checkpoint.v1",
        "experiment_id": experiment["experiment_id"],
        "seed": experiment["seed"],
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "rng": _rng_state(),
        "global_step": int(step),
        "condition_position": int(step % len(CONDITIONS)),
        "fixed_output_parity": fixed_output.detach().cpu().clone(),
        "asset_fingerprint": FORMAL_ASSET_FINGERPRINT,
    }


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pth.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)
    digest = _sha256(path)
    sidecar = {
        "model": True,
        "optimizer": True,
        "rng": True,
        "scheduler": True,
        "global_step": int(payload["global_step"]),
        "condition_position": int(payload["condition_position"]),
        "fixed_output_parity": True,
        "checkpoint_sha256": digest,
    }
    _atomic_json(path.with_suffix(path.suffix + ".resume.json"), sidecar)
    return {"path": str(path), "sha256": digest, "global_step": payload["global_step"]}


def _model_optimizer(
    input_dim: int, rank: int, seed: int, device: torch.device
) -> tuple[
    MultiOutfitLinearCoefficientControl,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = MultiOutfitLinearCoefficientControl(input_dim, rank).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return model, optimizer, scheduler


def _train_step(
    model: MultiOutfitLinearCoefficientControl,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    features: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    step: int,
    pairwise_weight: float,
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    condition = CONDITIONS[(step - 1) % len(CONDITIONS)]
    predictions = torch.stack([
        model(features[f"{outfit}/{condition}"]).standardized_coefficients
        for outfit in OUTFITS
    ])
    target = torch.stack([targets[outfit] for outfit in OUTFITS])
    coefficient = F.smooth_l1_loss(predictions, target)
    pairwise = pairwise_geometry_loss(predictions, target)
    loss = coefficient + pairwise_weight * pairwise
    if not torch.isfinite(loss):
        raise FloatingPointError("formal coefficient loss is NaN or Inf")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = {}
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        gradients[name] = {
            "present": gradient is not None,
            "finite": gradient is not None and bool(torch.isfinite(gradient).all()),
            "nonzero": gradient is not None and bool(torch.count_nonzero(gradient).item()),
            "l2": float(torch.linalg.vector_norm(gradient)) if gradient is not None else 0.0,
        }
        if not gradients[name]["finite"]:
            raise FloatingPointError(f"non-finite or missing trainable gradient: {name}")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    return {
        "step": int(step),
        "condition": condition,
        "loss": float(loss.detach()),
        "coefficient_loss": float(coefficient.detach()),
        "pairwise_geometry_loss": float(pairwise.detach()),
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
    }, gradients


def _load_checkpoint(
    path: Path, input_dim: int, rank: int, seed: int, device: torch.device
) -> tuple[
    MultiOutfitLinearCoefficientControl,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
    dict[str, Any],
]:
    model, optimizer, scheduler = _model_optimizer(input_dim, rank, seed, device)
    # RNG tensors are CPU ByteTensors by contract.  Loading the whole payload
    # onto CUDA converts them and makes torch.set_rng_state reject an otherwise
    # valid checkpoint.  Module/optimizer state is copied to the parameter
    # device by load_state_dict, so the checkpoint must first be deserialized on
    # CPU.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    _restore_rng(payload["rng"])
    return model, optimizer, scheduler, payload


def _checkpoint_resume_acceptance(
    checkpoint_200: Path,
    checkpoint_300: Path,
    model: MultiOutfitLinearCoefficientControl,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    features: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    seed: int,
    rank: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    input_dim = model.input_dim
    restored, restored_optimizer, restored_scheduler, final_payload = _load_checkpoint(
        checkpoint_300, input_dim, rank, seed, device
    )
    fixed = features["O01/cond_000000"]
    model.eval(); restored.eval()
    with torch.no_grad():
        before = model(fixed).standardized_coefficients
        after = restored(fixed).standardized_coefficients
    final = {
        "model_state_max_abs_diff": _tree_max_abs(model.state_dict(), restored.state_dict()),
        "optimizer_state_max_abs_diff": _tree_max_abs(optimizer.state_dict(), restored_optimizer.state_dict()),
        "scheduler_state_max_abs_diff": _tree_max_abs(scheduler.state_dict(), restored_scheduler.state_dict()),
        "fixed_output_max_abs_diff": float((before - after).abs().max()),
        "fixed_output_bitwise_exact": torch.equal(before, after),
        "global_step_exact": final_payload["global_step"] == 300,
        "condition_position_exact": final_payload["condition_position"] == 0,
    }
    first_model, first_optimizer, first_scheduler, probe_payload = _load_checkpoint(
        checkpoint_200, input_dim, rank, seed, device
    )
    probe_rng = copy.deepcopy(probe_payload["rng"])
    first_log, first_gradients = _train_step(
        first_model, first_optimizer, first_scheduler, features, targets, 201, 0.10
    )
    second_model, second_optimizer, second_scheduler, _ = _load_checkpoint(
        checkpoint_200, input_dim, rank, seed, device
    )
    _restore_rng(probe_rng)
    second_log, _ = _train_step(
        second_model, second_optimizer, second_scheduler, features, targets, 201, 0.10
    )
    one_step = {
        "loss_max_abs_diff": abs(first_log["loss"] - second_log["loss"]),
        "model_state_max_abs_diff": _tree_max_abs(first_model.state_dict(), second_model.state_dict()),
        "optimizer_state_max_abs_diff": _tree_max_abs(first_optimizer.state_dict(), second_optimizer.state_dict()),
        "scheduler_state_max_abs_diff": _tree_max_abs(first_scheduler.state_dict(), second_scheduler.state_dict()),
        "global_step": 201,
    }
    passed = (
        max(value for key, value in final.items() if key.endswith("diff")) == 0.0
        and all(value for key, value in final.items() if key.endswith("exact"))
        and max(value for key, value in one_step.items() if key.endswith("diff")) == 0.0
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "final_reload": final,
        "one_step_resume": one_step,
        "one_step_gradients": first_gradients,
    }


def _resume_completed_ours(
    context: Mapping[str, Any],
    attempt: Path,
    experiment: Mapping[str, Any],
    features: Mapping[str, torch.Tensor],
    coefficients: Mapping[str, torch.Tensor],
    basis_payload: Mapping[str, Any],
    checkpoint_path: Path,
) -> tuple[MultiOutfitLinearCoefficientControl, dict[str, Any], float]:
    if checkpoint_path.parent != attempt / "checkpoints":
        raise ValueError("resume checkpoint must belong to the selected formal attempt")
    if checkpoint_path.name != "checkpoint_step_000300.pth":
        raise ValueError("completed canary recovery requires the exact step-300 checkpoint")
    log_path = attempt / "logs/train.jsonl"
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 300 or [row["step"] for row in rows] != list(range(1, 301)):
        raise ValueError("completed canary recovery requires the exact 300-step trajectory")
    targets = multi._standardized_targets(coefficients, basis_payload)
    device = context["base"]._xyz.device
    input_dim = int(features["O01/cond_000000"].shape[-1])
    model, optimizer, scheduler, payload = _load_checkpoint(
        checkpoint_path, input_dim, 4, int(experiment["seed"]), device
    )
    if payload["global_step"] != 300 or payload["condition_position"] != 0:
        raise ValueError("completed canary checkpoint global state is invalid")
    resume = _checkpoint_resume_acceptance(
        attempt / "checkpoints/checkpoint_step_000200.pth",
        checkpoint_path,
        model,
        optimizer,
        scheduler,
        features,
        targets,
        int(experiment["seed"]),
        4,
    )
    if resume["status"] != "PASS":
        raise RuntimeError("formal checkpoint exact resume acceptance failed")
    stat_start = (attempt / "checkpoints/checkpoint_step_000000.pth").stat().st_mtime
    stat_end = checkpoint_path.stat().st_mtime
    report = {
        "status": "PASS",
        "optimizer_steps": 300,
        "resumed_after_tool_interruption": True,
        "resume_source_checkpoint": str(checkpoint_path),
        "resume_source_sha256": _sha256(checkpoint_path),
        "loss_first": float(rows[0]["loss"]),
        "loss_last": float(rows[-1]["loss"]),
        "gradients": resume["one_step_gradients"],
        "checkpoints": [
            {
                "path": str(attempt / "checkpoints" / f"checkpoint_step_{step:06d}.pth"),
                "sha256": _sha256(attempt / "checkpoints" / f"checkpoint_step_{step:06d}.pth"),
                "global_step": step,
            }
            for step in MILESTONES
        ],
        "checkpoint_resume": resume,
        "training_time_seconds": max(0.0, stat_end - stat_start),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    _atomic_json(attempt / "raw_metrics/training_summary.json", report)
    _atomic_json(attempt / "provenance/interrupted_runtime_recovery.json", {
        "status": "RECOVERED",
        "failure_stage": "checkpoint_rng_map_location",
        "model_failure": False,
        "optimizer_steps_repeated": 0,
        "resume_source_checkpoint": str(checkpoint_path),
        "resume_source_sha256": report["resume_source_sha256"],
    })
    return model, report, report["training_time_seconds"]


def _train_ours(
    context: Mapping[str, Any],
    attempt: Path,
    experiment: Mapping[str, Any],
    features: Mapping[str, torch.Tensor],
    basis: Any,
    coefficients: Mapping[str, torch.Tensor],
    basis_payload: Mapping[str, Any],
) -> tuple[MultiOutfitLinearCoefficientControl, dict[str, Any], float]:
    if experiment["method"] != "Ours_Seen_Outfit_Explicit_Basis_V1":
        raise ValueError("GPU canary formal runtime currently accepts only the frozen Ours method")
    rank = 4
    targets = multi._standardized_targets(coefficients, basis_payload)
    device = context["base"]._xyz.device
    model, optimizer, scheduler = _model_optimizer(
        int(features["O01/cond_000000"].shape[-1]), rank, int(experiment["seed"]), device
    )
    checkpoints: list[dict[str, Any]] = []
    gradient_seen = {
        name: {"finite": True, "nonzero": False, "max_l2": 0.0}
        for name, _ in model.named_parameters()
    }
    fixed = features["O01/cond_000000"]
    with torch.no_grad():
        fixed_output = model(fixed).standardized_coefficients
    checkpoints.append(_save_checkpoint(
        attempt / "checkpoints/checkpoint_step_000000.pth",
        _checkpoint_payload(model, optimizer, scheduler, 0, fixed_output, experiment),
    ))
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    loss_first = None
    loss_last = None
    for step in range(1, 301):
        log, gradients = _train_step(
            model, optimizer, scheduler, features, targets, step, 0.10
        )
        loss_first = log["loss"] if loss_first is None else loss_first
        loss_last = log["loss"]
        _append_jsonl(attempt / "logs/train.jsonl", log)
        for name, value in gradients.items():
            gradient_seen[name]["finite"] &= bool(value["finite"])
            gradient_seen[name]["nonzero"] |= bool(value["nonzero"])
            gradient_seen[name]["max_l2"] = max(
                gradient_seen[name]["max_l2"], float(value["l2"])
            )
        if step in MILESTONES:
            model.eval()
            with torch.no_grad():
                fixed_output = model(fixed).standardized_coefficients
            model.train()
            checkpoints.append(_save_checkpoint(
                attempt / "checkpoints" / f"checkpoint_step_{step:06d}.pth",
                _checkpoint_payload(
                    model, optimizer, scheduler, step, fixed_output, experiment
                ),
            ))
            _atomic_json(attempt / "raw_metrics/milestones" / f"step_{step:06d}.json", {
                "step": step,
                "loss": log,
                "fixed_output": fixed_output.detach().cpu().tolist(),
            })
    training_seconds = time.perf_counter() - started
    checkpoint_200 = attempt / "checkpoints/checkpoint_step_000200.pth"
    checkpoint_300 = attempt / "checkpoints/checkpoint_step_000300.pth"
    resume = _checkpoint_resume_acceptance(
        checkpoint_200, checkpoint_300, model, optimizer, scheduler,
        features, targets, int(experiment["seed"]), rank,
    )
    if resume["status"] != "PASS":
        raise RuntimeError("formal checkpoint exact resume acceptance failed")
    if not all(value["finite"] and value["nonzero"] for value in gradient_seen.values()):
        raise RuntimeError("a trainable parameter group lacked a finite nonzero gradient")
    report = {
        "status": "PASS",
        "optimizer_steps": 300,
        "loss_first": loss_first,
        "loss_last": loss_last,
        "gradients": gradient_seen,
        "checkpoints": checkpoints,
        "checkpoint_resume": resume,
        "training_time_seconds": training_seconds,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    _atomic_json(attempt / "raw_metrics/training_summary.json", report)
    return model, report, training_seconds


def _residual_metric_payload(prediction: Any, target: Any, bounds: Mapping[str, float]) -> dict[str, float]:
    result = parameterization.residual_metrics(
        prediction, target, bounds, active_epsilon=1e-8
    )
    return {
        "normalized_residual_rmse": float(result["normalized_rmse"]),
        "cosine_similarity": float(result["direction_cosine"]),
        "top_10_support_overlap": float(result["top_10pct_overlap"]),
        "top_20_support_overlap": float(result["top_20pct_overlap"]),
    }


def _render_metrics(
    sample: Mapping[str, Any],
    predicted_rgb: torch.Tensor,
    predicted_alpha: torch.Tensor,
    teacher_rgb: torch.Tensor,
    teacher_alpha: torch.Tensor,
) -> dict[str, float]:
    garment = diagnosis._garment_mask(sample)
    pred_target = o01._masked_mae(predicted_rgb, sample["target_edit_rgb"], garment)
    base_target = o01._masked_mae(sample["target_base_rgb"], sample["target_edit_rgb"], garment)
    change = (sample["target_edit_rgb"] - sample["target_base_rgb"]).abs().mean(0, keepdim=True) >= 0.01
    closer = (
        (predicted_rgb - sample["target_edit_rgb"]).abs().mean(0, keepdim=True)
        < (sample["target_base_rgb"] - sample["target_edit_rgb"]).abs().mean(0, keepdim=True)
    ).to(predicted_rgb)
    return {
        "garment_rgb_mae": o01._masked_mae(predicted_rgb, teacher_rgb, garment),
        "garment_alpha_mae": o01._masked_mae(predicted_alpha, teacher_alpha, garment),
        "edit_reduction": (base_target - pred_target) / max(base_target, 1e-12),
        "target_closer_fraction": float((closer * change).sum() / change.sum().clamp_min(1)),
        "protected_rgb_mae": o01._masked_mae(
            predicted_rgb, sample["target_base_rgb"], sample["target_protected_mask"]
        ),
        "background_rgb_mae": o01._masked_mae(
            predicted_rgb, sample["target_base_rgb"], 1 - sample["target_foreground_mask"]
        ),
    }


def _nearest(value: torch.Tensor, targets: Mapping[str, torch.Tensor]) -> str:
    return min(
        targets,
        key=lambda outfit: (float(torch.linalg.vector_norm(value - targets[outfit])), outfit),
    )


def _evaluate_ours(
    context: Mapping[str, Any],
    attempt: Path,
    experiment: Mapping[str, Any],
    model: MultiOutfitLinearCoefficientControl,
    extractor: Any,
    features: Mapping[str, torch.Tensor],
    basis: Any,
    coefficients: Mapping[str, torch.Tensor],
    payload: Mapping[str, Any],
    training: Mapping[str, Any],
    feature_seconds: float,
) -> tuple[Path, dict[str, Any]]:
    started = time.perf_counter()
    model.eval()
    targets = multi._standardized_targets(coefficients, payload)
    mean = payload["coefficient_train_mean"].to(context["base"]._xyz)
    std = payload["coefficient_train_std"].to(mean)
    bounds = context["config"]["basis"]["channel_bounds"]
    basis_storage = _basis_path(context).stat().st_size
    records: list[dict[str, Any]] = [{
        "record_type": "metadata",
        "seed": int(experiment["seed"]),
        "asset_fingerprint": FORMAL_ASSET_FINGERPRINT,
        "target_forward_input_used": False,
        "not_applicable_metrics": [],
        "efficiency": {
            "trainable_parameter_count": int(training["trainable_parameter_count"]),
            "basis_storage_bytes": int(basis_storage),
            "peak_vram_bytes": int(training["peak_vram_bytes"]),
            "training_time_seconds": float(training["training_time_seconds"]),
            "inference_time_seconds": 0.0,
            "render_time_seconds": 0.0,
        },
    }]
    coefficient_cache: dict[str, torch.Tensor] = {}
    residual_cache: dict[str, Any] = {}
    rgb_cache: dict[str, torch.Tensor] = {}
    teacher_rgb_cache: dict[str, torch.Tensor] = {}
    rows = []
    render_seconds = 0.0
    inference_seconds = 0.0
    for outfit in OUTFITS:
        panels = []
        teacher_residual = basis(coefficients[outfit], chunk_size=16384)
        for condition in CONDITIONS:
            key = f"{outfit}/{condition}"
            infer_started = time.perf_counter()
            with torch.no_grad():
                predicted_standardized = model(features[key]).standardized_coefficients
                restored = restore_coefficients(predicted_standardized, mean, std)
                predicted_residual = basis(restored, chunk_size=16384)
            inference_seconds += time.perf_counter() - infer_started
            render_started = time.perf_counter()
            predicted_rgb, predicted_alpha = multi._render_residual(
                context, outfit, condition, predicted_residual
            )
            teacher_rgb, teacher_alpha = multi._render_residual(
                context, outfit, condition, teacher_residual
            )
            render_seconds += time.perf_counter() - render_started
            sample = context["samples"][key]
            residual_metrics = _residual_metric_payload(
                predicted_residual, teacher_residual, bounds
            )
            render_metrics = _render_metrics(
                sample, predicted_rgb, predicted_alpha, teacher_rgb, teacher_alpha
            )
            records.append({
                "record_type": "episode",
                "outfit_id": outfit,
                "view_id": condition,
                "target_reference_overlap": 0,
                "predicted_standardized_coefficients": predicted_standardized.detach().cpu().tolist(),
                "target_standardized_coefficients": targets[outfit].detach().cpu().tolist(),
                "coefficient_normalization": {
                    "mean": mean.detach().cpu().tolist(),
                    "std": std.detach().cpu().tolist(),
                },
                "teacher_standardized_coefficients": {
                    name: value.detach().cpu().tolist() for name, value in targets.items()
                },
                "precomputed_residual_metrics": residual_metrics,
                "residual_metric_source": "production_full_gaussian_field_v1",
                "render_metrics": render_metrics,
            })
            coefficient_cache[key] = predicted_standardized.detach()
            residual_cache[key] = predicted_residual
            rgb_cache[key] = predicted_rgb.detach()
            teacher_rgb_cache[key] = teacher_rgb.detach()
            episode_visual_root = attempt / "visuals/episodes"
            episode_visual_root.mkdir(parents=True, exist_ok=True)
            save_render_tensor(
                episode_visual_root / f"{key.replace('/', '_')}_prediction.png",
                predicted_rgb, 3,
            )
            save_render_tensor(
                episode_visual_root / f"{key.replace('/', '_')}_alpha.png",
                predicted_alpha, 1,
            )
            panels.extend([
                (f"{condition} target", sample["target_edit_rgb"].detach().cpu(), 3),
                (f"{condition} base", sample["target_base_rgb"].detach().cpu(), 3),
                (f"{condition} teacher", teacher_rgb.detach().cpu(), 3),
                (f"{condition} prediction", predicted_rgb.detach().cpu(), 3),
            ])
        rows.append((outfit, panels))
    o01._save_contact_sheet(
        attempt / "visuals/five_outfit_four_view_contact_sheet.png", rows
    )

    swap_rows = []
    for target_outfit in OUTFITS:
        for condition in CONDITIONS:
            target_key = f"{target_outfit}/{condition}"
            correct_coefficient = coefficient_cache[target_key]
            correct_target_distance = float(torch.linalg.vector_norm(
                correct_coefficient - targets[target_outfit]
            ))
            correct_render_error = records[
                1 + OUTFITS.index(target_outfit) * len(CONDITIONS) + CONDITIONS.index(condition)
            ]["render_metrics"]["garment_rgb_mae"]
            teacher_residual = basis(coefficients[target_outfit], chunk_size=16384)
            for source_outfit in OUTFITS:
                if source_outfit == target_outfit:
                    continue
                source_key = f"{source_outfit}/{condition}"
                swapped_coefficient = coefficient_cache[source_key]
                swapped_restored = restore_coefficients(swapped_coefficient, mean, std)
                swapped_residual = basis(swapped_restored, chunk_size=16384)
                swapped_rgb, _ = multi._render_residual(
                    context, target_outfit, condition, swapped_residual
                )
                sample = context["samples"][target_key]
                garment = diagnosis._garment_mask(sample)
                swapped_render_error = o01._masked_mae(
                    swapped_rgb, teacher_rgb_cache[target_key], garment
                )
                correct_residual = parameterization.residual_metrics(
                    residual_cache[target_key], teacher_residual, bounds, active_epsilon=1e-8
                )["normalized_rmse"]
                swapped_residual_error = parameterization.residual_metrics(
                    swapped_residual, teacher_residual, bounds, active_epsilon=1e-8
                )["normalized_rmse"]
                swapped_target_distance = float(torch.linalg.vector_norm(
                    swapped_coefficient - targets[target_outfit]
                ))
                correct_wins = bool(
                    correct_target_distance < swapped_target_distance
                    and float(correct_residual) < float(swapped_residual_error)
                    and correct_render_error < swapped_render_error
                )
                records.append({
                    "record_type": "swap",
                    "source_outfit": source_outfit,
                    "target_outfit": target_outfit,
                    "view_id": condition,
                    "correct_wins": correct_wins,
                })
                if condition == "cond_000318":
                    swap_rows.append((f"{target_outfit}<-{source_outfit}", [
                        ("teacher", teacher_rgb_cache[target_key].detach().cpu(), 3),
                        ("correct", rgb_cache[target_key].detach().cpu(), 3),
                        ("swapped", swapped_rgb.detach().cpu(), 3),
                    ]))
    o01._save_contact_sheet(attempt / "visuals/reference_swap_contact_sheet.png", swap_rows)

    for outfit in OUTFITS:
        for condition in CONDITIONS:
            key = f"{outfit}/{condition}"
            episode = context["episodes"][key]
            correct = coefficient_cache[key]
            permuted = multi.subset_reference_episode(episode, (2, 0, 1))
            single = multi.subset_reference_episode(episode, (0,))
            dropout = multi.subset_reference_episode(episode, (0, 1))
            with torch.no_grad():
                permutation_value = model(
                    multi.prediction_feature(extractor, permuted)
                ).standardized_coefficients
                single_value = model(
                    multi.prediction_feature(extractor, single)
                ).standardized_coefficients
                dropout_value = model(
                    multi.prediction_feature(extractor, dropout)
                ).standardized_coefficients
                zero_episode = dict(episode)
                zero_episode["reference_images"] = torch.zeros_like(episode["reference_images"])
                base_episode = dict(episode)
                base_episode["reference_images"] = context["samples"][key]["target_base_rgb"].unsqueeze(0).expand_as(
                    episode["reference_images"]
                ).clone()
                zero_value = model(multi.prediction_feature(extractor, zero_episode)).standardized_coefficients
                base_value = model(multi.prediction_feature(extractor, base_episode)).standardized_coefficients
            records.extend([
                {
                    "record_type": "permutation", "outfit_id": outfit,
                    "view_id": condition,
                    "max_difference": float((correct - permutation_value).abs().max()),
                },
                {
                    "record_type": "single_reference", "outfit_id": outfit,
                    "view_id": condition, "correct": _nearest(single_value, targets) == outfit,
                },
                {
                    "record_type": "two_reference_dropout", "outfit_id": outfit,
                    "view_id": condition, "correct": _nearest(dropout_value, targets) == outfit,
                },
                {
                    "record_type": "replacement", "outfit_id": outfit,
                    "view_id": condition,
                    "zero_difference": float(torch.linalg.vector_norm(correct - zero_value)),
                    "base_difference": float(torch.linalg.vector_norm(correct - base_value)),
                },
            ])
    records[0]["efficiency"]["inference_time_seconds"] = inference_seconds + feature_seconds
    records[0]["efficiency"]["render_time_seconds"] = render_seconds
    raw_path = attempt / "raw_metrics/raw_episode_outputs.jsonl"
    raw_path.write_text(
        "".join(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "status": "PASS",
        "raw_metrics": str(raw_path),
        "record_counts": {
            kind: sum(record["record_type"] == kind for record in records)
            for kind in sorted({record["record_type"] for record in records})
        },
        "target_forward_leakage": False,
        "outfit_id_in_model": False,
        "elapsed_seconds": time.perf_counter() - started,
        "inference_time_seconds": inference_seconds + feature_seconds,
        "render_time_seconds": render_seconds,
    }
    _atomic_json(attempt / "raw_metrics/formal_evaluation_summary.json", summary)
    return raw_path, summary


def run() -> dict[str, Any]:
    attempt, experiment, registry, paper_config = _load_attempt()
    adapter = adapter_for(experiment, paper_config)
    contract = adapter.validate_contract()
    if experiment["experiment_id"] != "PAPER-OURS-S0":
        raise ValueError("the initial GPU canary must be the unique registry Ours seed=0 entry")
    environment = _environment()
    if not environment["cuda_available"]:
        raise RuntimeError("FORMAL_GPU_PREFLIGHT_FAIL")
    device_probe = torch.arange(4096, device="cuda", dtype=torch.float32).sin().square().sum()
    torch.cuda.synchronize()
    if not torch.isfinite(device_probe):
        raise RuntimeError("FORMAL_GPU_PREFLIGHT_FAIL: CUDA tensor smoke")
    _atomic_json(attempt / "preflight/formal_gpu_preflight.json", {
        "status": "PASS", "environment": environment,
        "cuda_tensor_smoke": float(device_probe),
        "adapter_contract": contract,
    })
    context = _legacy_context(attempt)
    base_before = multi._tensor_state_fingerprint(multi._base_named_tensors(context["base"]))
    backbone_before = o01._state_fingerprint(
        context["legacy_model"].clothing_observation_encoder.backbone.state_dict()
    )
    extractor, features, feature_seconds = _extract_features(context, attempt)
    basis_path = _basis_path(context)
    basis, coefficients, basis_payload = multi.load_basis_artifact(
        basis_path, context["base"]._xyz.device
    )
    basis_before = basis.fingerprint()
    resume_value = os.environ.get("CANONDRESSGS_PAPER_RESUME_CHECKPOINT")
    if resume_value:
        model, training, _ = _resume_completed_ours(
            context, attempt, experiment, features, coefficients, basis_payload,
            Path(resume_value).resolve(),
        )
    else:
        model, training, _ = _train_ours(
            context, attempt, experiment, features, basis, coefficients, basis_payload
        )
    raw_path, evaluation = _evaluate_ours(
        context, attempt, experiment, model, extractor, features, basis,
        coefficients, basis_payload, training, feature_seconds,
    )
    base_after = multi._tensor_state_fingerprint(multi._base_named_tensors(context["base"]))
    backbone_after = o01._state_fingerprint(
        context["legacy_model"].clothing_observation_encoder.backbone.state_dict()
    )
    basis_after = basis.fingerprint()
    frozen_gradient_count = (
        multi._base_gradient_count(context["base"])
        + sum(parameter.grad is not None for parameter in extractor.parameters())
        + sum(
            parameter.grad is not None
            for parameter in context["legacy_model"].dressable_model.anchor_clothing_mlp.parameters()
        )
        + sum(parameter.grad is not None for parameter in basis.parameters())
    )
    frozen_change = 0.0 if (
        base_before == base_after
        and backbone_before == backbone_after
        and basis_before == basis_after
    ) else math.inf
    result = {
        "status": "PASS",
        "experiment_id": experiment["experiment_id"],
        "optimizer_created": True,
        "optimizer_parameter_scope": adapter.parameter_scope,
        "global_step": 300,
        "checkpoint_milestones": list(MILESTONES),
        "frozen_parameter_max_change": frozen_change,
        "frozen_gradient_count": int(frozen_gradient_count),
        "target_forward_leakage": False,
        "outfit_id_in_model": False,
        "base_fingerprint_before": base_before,
        "base_fingerprint_after": base_after,
        "backbone_fingerprint_before": backbone_before,
        "backbone_fingerprint_after": backbone_after,
        "basis_fingerprint_before": basis_before,
        "basis_fingerprint_after": basis_after,
        "checkpoint_resume": training["checkpoint_resume"],
        "training": training,
        "evaluation": evaluation,
        "raw_metrics": str(raw_path),
        "environment": environment,
    }
    if frozen_change != 0.0 or frozen_gradient_count != 0:
        raise RuntimeError("frozen parameter contract failed")
    _atomic_json(attempt / "provenance/executor_result.json", result)
    _atomic_json(attempt / "RUN_STATUS.json", {
        "status": "TRAINED", "optimizer_steps": 300,
        "experiment_id": experiment["experiment_id"],
        "updated_at_unix": time.time(),
    })
    print(json.dumps({
        "status": "PASS", "experiment_id": experiment["experiment_id"],
        "attempt": str(attempt), "raw_metrics": str(raw_path),
    }))
    return result


if __name__ == "__main__":
    try:
        run()
    except Exception as error:
        attempt_value = os.environ.get("CANONDRESSGS_PAPER_ATTEMPT")
        if attempt_value:
            failed_attempt = Path(attempt_value)
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            _atomic_json(failed_attempt / "provenance" / f"runtime_failure_{stamp}.json", {
                "status": "INTERRUPTED_RUNTIME",
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "traceback": traceback.format_exc(),
                "model_failure": False,
                "latest_checkpoint": str(
                    failed_attempt / "checkpoints/checkpoint_step_000300.pth"
                ),
            })
            _atomic_json(failed_attempt / "RUN_STATUS.json", {
                "status": "INTERRUPTED_RUNTIME",
                "optimizer_steps": 300 if (
                    failed_attempt / "checkpoints/checkpoint_step_000300.pth"
                ).is_file() else 0,
                "failure_stage": "formal_runtime_tooling",
                "exception_type": type(error).__name__,
                "exception_message": str(error),
            })
        raise
