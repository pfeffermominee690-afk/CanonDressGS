from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

if __package__ in (None, ""):
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from scene.frozen_f2_linear_coefficient_control import FrozenF2ReferenceFeatureExtractor
from scene.gaussian_clothing_residuals import GaussianClothingResiduals
from scene.multi_outfit_linear_coefficient_control import (
    MultiOutfitLinearCoefficientControl,
    pairwise_geometry_loss,
)
from scene.reference_basis_coefficient_fusion import (
    MaskAwareReferenceTokenEncoderV1,
    ReferenceSetCoefficientFusionV1,
)
from tools import diagnose_image_conditioned_overfit_failure as diagnosis
from tools import run_image_conditioned_overfit_o01 as o01
from tools import run_multi_outfit_explicit_basis as multi
from tools import run_residual_field_parameterization as parameterization
from tools.check_real_image_conditioned_one_batch import save_render_tensor
from tools.paper import formal_runtime as core


OUTFITS = core.OUTFITS
CONDITIONS = core.CONDITIONS
MILESTONES = core.MILESTONES
ASSET_FINGERPRINT = core.FORMAL_ASSET_FINGERPRINT


@dataclass(frozen=True)
class PredictionOutput:
    standardized_coefficients: torch.Tensor


class BoundedVectorControl(nn.Module):
    """LayerNorm -> Linear(K) with the frozen legacy tanh endpoint."""

    def __init__(self, input_dim: int, rank: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.rank = int(rank)
        self.normalization = nn.LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, rank)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, value: torch.Tensor) -> PredictionOutput:
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.shape != (1, self.input_dim):
            raise ValueError("bounded predictor input shape mismatch")
        coefficient = torch.tanh(self.linear(self.normalization(value))).reshape(self.rank)
        if not torch.isfinite(coefficient).all():
            raise FloatingPointError("bounded predictor produced NaN or Inf")
        return PredictionOutput(coefficient)


class LegacyRFFRank4(nn.Module):
    """Four frozen-contract RF-F scalar heads sharing one token adapter.

    The frozen spatial backbone is evaluated into raw reference tokens before
    optimization.  This module is exactly the trainable RF-F portion: the
    historical token adapter followed by four scalar reference-set fusion
    heads, one per frozen basis coefficient.
    """

    def __init__(self, raw_dim: int, rank: int = 4) -> None:
        super().__init__()
        self.raw_dim = int(raw_dim)
        self.rank = int(rank)
        self.token_adapter = nn.Sequential(
            nn.LayerNorm(raw_dim), nn.Linear(raw_dim, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.LayerNorm(128),
        )
        self.fusions = nn.ModuleList(
            [ReferenceSetCoefficientFusionV1(token_dim=128, hidden_dim=128) for _ in range(rank)]
        )

    def forward(self, packed: torch.Tensor) -> PredictionOutput:
        if packed.ndim == 2 and packed.shape[0] == 1:
            packed = packed[0]
        expected = 3 * self.raw_dim + 3
        if packed.ndim != 1 or packed.numel() != expected:
            raise ValueError("RF-F packed reference-token shape mismatch")
        raw = packed[: 3 * self.raw_dim].reshape(3, self.raw_dim)
        valid = packed[3 * self.raw_dim :].reshape(3, 1)
        tokens = self.token_adapter(raw) * valid
        values = [head(tokens, valid).coefficient.reshape(()) for head in self.fusions]
        result = torch.stack(values)
        if not torch.isfinite(result).all():
            raise FloatingPointError("RF-F predictor produced NaN or Inf")
        return PredictionOutput(result)


def _atomic_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _episode_rows(
    extractor: FrozenF2ReferenceFeatureExtractor,
    episode: Mapping[str, Any],
    images: torch.Tensor,
) -> dict[str, torch.Tensor]:
    cloth = episode["reference_cloth_masks"].to(images)
    foreground = episode["reference_foreground_masks"].to(images)
    valid = episode["reference_valid_mask"].reshape(-1, 1).to(images)
    with torch.no_grad():
        maps = extractor.spatial_backbone(images)
        cloth_small = F.interpolate(cloth, maps.shape[-2:], mode="area").clamp(0, 1)
        foreground_small = F.interpolate(foreground, maps.shape[-2:], mode="area").clamp(0, 1)
        clothing_mean = MaskAwareReferenceTokenEncoderV1._weighted_mean(maps, cloth_small)
        clothing_max = MaskAwareReferenceTokenEncoderV1._masked_max(maps, cloth_small)
        foreground_mean = MaskAwareReferenceTokenEncoderV1._weighted_mean(maps, foreground_small)
        global_mean = maps.mean(dim=(2, 3))
        area, aspect, centroid = MaskAwareReferenceTokenEncoderV1._mask_geometry(cloth)
        direction = MaskAwareReferenceTokenEncoderV1._view_direction(
            episode["reference_w2c"].to(images)
        )
        raw_rff = torch.cat(
            (
                clothing_mean, clothing_max, foreground_mean,
                clothing_mean - foreground_mean, area, aspect, centroid, direction,
            ),
            dim=-1,
        )
    values = {
        "f2": torch.cat((clothing_mean, clothing_max), dim=-1),
        "mean": clothing_mean,
        "global": global_mean,
        "rff": raw_rff,
        "valid": valid,
    }
    if any(not torch.isfinite(item).all() for item in values.values()):
        raise FloatingPointError("shared frozen feature cache contains NaN or Inf")
    return {name: value.detach().cpu() for name, value in values.items()}


def build_shared_feature_cache(
    context: Mapping[str, Any], formal_root: Path,
) -> tuple[dict[str, Any], Path, str, float]:
    path = formal_root / "shared_preflight/frozen_reference_feature_rows_v1.pt"
    started = time.perf_counter()
    expected = {
        f"{outfit}/{condition}"
        for outfit in (*OUTFITS, "O07") for condition in CONDITIONS
    }
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != "canondressgs.paper_frozen_reference_rows.v1":
            raise ValueError("shared feature cache schema mismatch")
        if set(payload["episodes"]) != expected:
            raise ValueError("shared feature cache key mismatch")
        return payload, path, core._sha256(path), time.perf_counter() - started
    extractor = multi._feature_extractor(context)
    rows: dict[str, Any] = {}
    for outfit in (*OUTFITS, "O07"):
        for condition in CONDITIONS:
            key = f"{outfit}/{condition}"
            episode = context["episodes"][key]
            sample = context["samples"][key]
            images = episode["reference_images"]
            base_images = sample["target_base_rgb"].unsqueeze(0).expand_as(images).clone()
            rows[key] = {
                "normal": _episode_rows(extractor, episode, images),
                "zero": _episode_rows(extractor, episode, torch.zeros_like(images)),
                "base": _episode_rows(extractor, episode, base_images),
            }
    payload = {
        "schema_version": "canondressgs.paper_frozen_reference_rows.v1",
        "asset_fingerprint": ASSET_FINGERPRINT,
        "backbone_fingerprint": o01._state_fingerprint(
            context["legacy_model"].clothing_observation_encoder.backbone.state_dict()
        ),
        "target_forward_leakage": False,
        "outfit_id_in_model": False,
        "episodes": rows,
    }
    _atomic_torch(path, payload)
    return payload, path, core._sha256(path), time.perf_counter() - started


def _selected_rows(
    cache: Mapping[str, Any], key: str, kind: str, *, variant: str = "normal",
    indices: Sequence[int] = (0, 1, 2),
) -> tuple[torch.Tensor, torch.Tensor]:
    row = cache["episodes"][key][variant]
    index = torch.tensor(tuple(indices), dtype=torch.long)
    return row[kind].index_select(0, index), row["valid"].index_select(0, index)


def feature_value(
    cache: Mapping[str, Any], key: str, kind: str, device: torch.device,
    *, variant: str = "normal", indices: Sequence[int] = (0, 1, 2),
) -> torch.Tensor:
    rows, valid = _selected_rows(cache, key, kind, variant=variant, indices=indices)
    rows, valid = rows.to(device), valid.to(device)
    if valid.sum().item() <= 0:
        raise ValueError("reference feature set is empty")
    if kind == "f2":
        mean = (rows * valid).sum(0, keepdim=True) / valid.sum().clamp_min(1e-8)
        lowest = torch.finfo(rows.dtype).min
        maximum = rows.masked_fill(valid <= 0, lowest).amax(0, keepdim=True)
        return torch.cat((mean, maximum), dim=-1)
    if kind in {"mean", "global"}:
        return (rows * valid).sum(0, keepdim=True) / valid.sum().clamp_min(1e-8)
    if kind == "rff":
        padded = rows.new_zeros((3, rows.shape[-1]))
        padded_valid = valid.new_zeros((3, 1))
        padded[: rows.shape[0]] = rows
        padded_valid[: valid.shape[0]] = valid
        return torch.cat((padded.reshape(-1), padded_valid.reshape(-1)))
    raise KeyError(kind)


def _rank_for(method: str) -> int:
    if method.startswith("A1_Basis_Rank_"):
        return int(method.rsplit("_", 1)[1])
    return 4


def _reference_count(method: str) -> int:
    if method.startswith("A7_Reference_Count_"):
        return int(method.rsplit("_", 1)[1])
    return 3


def _feature_kind(method: str) -> str:
    if method.startswith("B3_") or method.startswith("A2_"):
        return "global"
    if method.startswith("B4_") or method.startswith("A3_"):
        return "mean"
    if method.startswith("B5_"):
        return "rff"
    return "f2"


def _basis_artifact(
    context: Mapping[str, Any], method: str,
) -> tuple[Any, dict[str, torch.Tensor], dict[str, Any], Path]:
    rank = _rank_for(method)
    if rank == 4 and not method.startswith("A1_"):
        path = core._basis_path(context)
    else:
        root = Path(os.environ["CANONDRESSGS_ASSET_ROOT"])
        path = (
            root / "pipeline_full/SUBJECT02-MULTI-OUTFIT-EXPLICIT-BASIS-001/attempt_003"
            / f"stage_b_basis/ranks/rank_{rank}/basis.pt"
        )
        if not path.is_file():
            raise RuntimeError(f"PAPER_ASSET_MISMATCH: rank-{rank} basis")
    basis, coefficients, payload = multi.load_basis_artifact(
        path, context["base"]._xyz.device
    )
    if int(payload["rank"]) != rank:
        raise ValueError("basis rank artifact contract mismatch")
    return basis, coefficients, payload, path


def _targets(
    method: str, coefficients: Mapping[str, torch.Tensor], payload: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    if method.startswith("A4_"):
        return {name: value.detach() for name, value in coefficients.items() if name in OUTFITS}
    return multi._standardized_targets(coefficients, payload)


def _to_standardized(
    method: str, value: torch.Tensor, payload: Mapping[str, Any]
) -> torch.Tensor:
    if method.startswith("A4_"):
        mean = payload["coefficient_train_mean"].to(value)
        std = payload["coefficient_train_std"].to(value)
        return (value - mean) / std
    return value


def _model_factory(
    method: str, input_dim: int, rank: int, seed: int, device: torch.device,
) -> nn.Module:
    _seed(seed)
    if method.startswith("B5_"):
        model: nn.Module = LegacyRFFRank4(raw_dim=(input_dim - 3) // 3, rank=rank)
    elif method.startswith("A5_"):
        model = BoundedVectorControl(input_dim, rank)
    else:
        model = MultiOutfitLinearCoefficientControl(input_dim, rank)
    return model.to(device)


def _optimizer(model: nn.Module) -> tuple[torch.optim.Optimizer, Any]:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return optimizer, scheduler


def _training_loss(
    method: str, prediction: torch.Tensor, target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    coefficient = F.smooth_l1_loss(prediction, target)
    geometry = pairwise_geometry_loss(prediction, target)
    if method.startswith("A5_"):
        sign = F.relu(0.8 - torch.sign(target) * prediction).mean()
        pair_values = []
        for left in range(prediction.shape[0]):
            for right in range(left + 1, prediction.shape[0]):
                pair_values.append(
                    F.relu(1.5 - torch.abs(prediction[left] - prediction[right])).mean()
                )
        absolute_pair = torch.stack(pair_values).mean()
        total = coefficient + 0.25 * sign + 0.10 * absolute_pair
        return total, {
            "coefficient_loss": coefficient, "sign_loss": sign,
            "absolute_pair_loss": absolute_pair, "pairwise_geometry_loss": geometry,
        }
    weight = 0.0 if method.startswith("A6_") else 0.10
    return coefficient + weight * geometry, {
        "coefficient_loss": coefficient,
        "pairwise_geometry_loss": geometry,
    }


def _training_features(
    cache: Mapping[str, Any], method: str, condition: str, device: torch.device,
) -> list[torch.Tensor]:
    kind = _feature_kind(method)
    count = _reference_count(method)
    indices = tuple(range(count))
    return [
        feature_value(cache, f"{outfit}/{condition}", kind, device, indices=indices)
        for outfit in OUTFITS
    ]


def _train_step(
    model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any,
    cache: Mapping[str, Any], method: str, targets: Mapping[str, torch.Tensor],
    step: int, device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    condition = CONDITIONS[(step - 1) % len(CONDITIONS)]
    predictions = torch.stack([
        model(value).standardized_coefficients
        for value in _training_features(cache, method, condition, device)
    ])
    target = torch.stack([targets[outfit] for outfit in OUTFITS])
    loss, parts = _training_loss(method, predictions, target)
    if not torch.isfinite(loss):
        raise FloatingPointError("formal training loss is NaN or Inf")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients: dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        gradients[name] = {
            "present": gradient is not None,
            "finite": gradient is not None and bool(torch.isfinite(gradient).all()),
            "nonzero": gradient is not None and bool(torch.count_nonzero(gradient)),
            "l2": 0.0 if gradient is None else float(torch.linalg.vector_norm(gradient)),
        }
        if not gradients[name]["finite"]:
            raise FloatingPointError(f"trainable gradient invalid: {name}")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    row = {
        "step": step, "condition": condition, "loss": float(loss.detach()),
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
    }
    row.update({name: float(value.detach()) for name, value in parts.items()})
    return row, gradients


def _checkpoint_payload(
    model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any,
    step: int, fixed: torch.Tensor, experiment: Mapping[str, Any], method: str,
) -> dict[str, Any]:
    return {
        "schema_version": "canondressgs.formal_seen_outfit_checkpoint.v2",
        "experiment_id": experiment["experiment_id"], "method": method,
        "seed": experiment["seed"], "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()), "rng": core._rng_state(),
        "global_step": int(step), "condition_position": int(step % len(CONDITIONS)),
        "fixed_output_parity": fixed.detach().cpu().clone(),
        "asset_fingerprint": ASSET_FINGERPRINT,
    }


def _load_checkpoint(
    path: Path, factory: Callable[[], nn.Module], device: torch.device,
) -> tuple[nn.Module, torch.optim.Optimizer, Any, dict[str, Any]]:
    model = factory()
    optimizer, scheduler = _optimizer(model)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    core._restore_rng(payload["rng"])
    return model.to(device), optimizer, scheduler, payload


def _checkpoint_resume(
    attempt: Path, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any,
    factory: Callable[[], nn.Module], cache: Mapping[str, Any], method: str,
    targets: Mapping[str, torch.Tensor], device: torch.device,
) -> dict[str, Any]:
    path200 = attempt / "checkpoints/checkpoint_step_000200.pth"
    path300 = attempt / "checkpoints/checkpoint_step_000300.pth"
    restored, restored_optimizer, restored_scheduler, payload = _load_checkpoint(
        path300, factory, device
    )
    fixed_feature = _training_features(cache, method, CONDITIONS[0], device)[0]
    model.eval(); restored.eval()
    with torch.no_grad():
        before = model(fixed_feature).standardized_coefficients
        after = restored(fixed_feature).standardized_coefficients
    final = {
        "model_state_max_abs_diff": core._tree_max_abs(model.state_dict(), restored.state_dict()),
        "optimizer_state_max_abs_diff": core._tree_max_abs(optimizer.state_dict(), restored_optimizer.state_dict()),
        "scheduler_state_max_abs_diff": core._tree_max_abs(scheduler.state_dict(), restored_scheduler.state_dict()),
        "fixed_output_max_abs_diff": float((before - after).abs().max()),
        "fixed_output_bitwise_exact": torch.equal(before, after),
        "global_step_exact": payload["global_step"] == 300,
        "condition_position_exact": payload["condition_position"] == 0,
    }
    first, first_optimizer, first_scheduler, payload200 = _load_checkpoint(path200, factory, device)
    saved_rng = copy.deepcopy(payload200["rng"])
    first_log, first_gradients = _train_step(
        first, first_optimizer, first_scheduler, cache, method, targets, 201, device
    )
    second, second_optimizer, second_scheduler, _ = _load_checkpoint(path200, factory, device)
    core._restore_rng(saved_rng)
    second_log, _ = _train_step(
        second, second_optimizer, second_scheduler, cache, method, targets, 201, device
    )
    one_step = {
        "loss_max_abs_diff": abs(first_log["loss"] - second_log["loss"]),
        "model_state_max_abs_diff": core._tree_max_abs(first.state_dict(), second.state_dict()),
        "optimizer_state_max_abs_diff": core._tree_max_abs(first_optimizer.state_dict(), second_optimizer.state_dict()),
        "scheduler_state_max_abs_diff": core._tree_max_abs(first_scheduler.state_dict(), second_scheduler.state_dict()),
        "global_step": 201,
    }
    passed = (
        max(value for key, value in final.items() if key.endswith("diff")) == 0.0
        and all(value for key, value in final.items() if key.endswith("exact"))
        and max(value for key, value in one_step.items() if key.endswith("diff")) == 0.0
    )
    return {
        "status": "PASS" if passed else "FAIL", "final_reload": final,
        "one_step_resume": one_step, "one_step_gradients": first_gradients,
    }


def train_model(
    attempt: Path, experiment: Mapping[str, Any], cache: Mapping[str, Any],
    coefficients: Mapping[str, torch.Tensor], payload: Mapping[str, Any],
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    method = experiment["method"]
    rank = _rank_for(method)
    kind = _feature_kind(method)
    sample_feature = feature_value(
        cache, f"O01/{CONDITIONS[0]}", kind, device,
        indices=tuple(range(_reference_count(method))),
    )
    input_dim = int(sample_feature.numel())
    seed = int(experiment["seed"])
    factory = lambda: _model_factory(method, input_dim, rank, seed, device)
    model = factory()
    optimizer, scheduler = _optimizer(model)
    targets = _targets(method, coefficients, payload)
    checkpoints = []
    with torch.no_grad():
        fixed = model(sample_feature).standardized_coefficients
    checkpoints.append(core._save_checkpoint(
        attempt / "checkpoints/checkpoint_step_000000.pth",
        _checkpoint_payload(model, optimizer, scheduler, 0, fixed, experiment, method),
    ))
    def group_name(name: str) -> str:
        if not method.startswith("B5_"):
            return name
        if name.startswith("token_adapter."):
            return "token_adapter"
        parts = name.split(".")
        return ".".join(parts[:3])

    gradient_seen: dict[str, dict[str, Any]] = {}
    for name, _ in model.named_parameters():
        gradient_seen.setdefault(
            group_name(name), {"finite": True, "nonzero": False, "max_l2": 0.0}
        )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    first_loss = None
    last_loss = None
    for step in range(1, 301):
        row, gradients = _train_step(
            model, optimizer, scheduler, cache, method, targets, step, device
        )
        first_loss = row["loss"] if first_loss is None else first_loss
        last_loss = row["loss"]
        core._append_jsonl(attempt / "logs/train.jsonl", row)
        for name, value in gradients.items():
            group = group_name(name)
            gradient_seen[group]["finite"] &= bool(value["finite"])
            gradient_seen[group]["nonzero"] |= bool(value["nonzero"])
            gradient_seen[group]["max_l2"] = max(
                gradient_seen[group]["max_l2"], float(value["l2"])
            )
        if step in MILESTONES:
            model.eval()
            with torch.no_grad():
                fixed = model(sample_feature).standardized_coefficients
            model.train()
            checkpoints.append(core._save_checkpoint(
                attempt / "checkpoints" / f"checkpoint_step_{step:06d}.pth",
                _checkpoint_payload(
                    model, optimizer, scheduler, step, fixed, experiment, method
                ),
            ))
            core._atomic_json(
                attempt / "raw_metrics/milestones" / f"step_{step:06d}.json",
                {"step": step, "loss": row, "fixed_output": fixed.detach().cpu().tolist()},
            )
    training_seconds = time.perf_counter() - started
    resume = _checkpoint_resume(
        attempt, model, optimizer, scheduler, factory, cache, method, targets, device
    )
    if resume["status"] != "PASS":
        raise RuntimeError("formal checkpoint exact resume acceptance failed")
    if not all(value["finite"] and value["nonzero"] for value in gradient_seen.values()):
        raise RuntimeError("a trainable parameter lacked a finite nonzero gradient")
    report = {
        "status": "PASS", "optimizer_steps": 300,
        "loss_first": first_loss, "loss_last": last_loss,
        "gradients": gradient_seen, "checkpoints": checkpoints,
        "checkpoint_resume": resume, "training_time_seconds": training_seconds,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    core._atomic_json(attempt / "raw_metrics/training_summary.json", report)
    return model, report


def _predict_coefficients(
    method: str, model: nn.Module | None, cache: Mapping[str, Any], key: str,
    payload: Mapping[str, Any], device: torch.device, *, variant: str = "normal",
    indices: Sequence[int] | None = None,
) -> torch.Tensor:
    if model is None:
        raise ValueError("fixed methods do not have a reference predictor")
    count = _reference_count(method)
    selected = tuple(range(count)) if indices is None else tuple(indices)
    value = feature_value(
        cache, key, _feature_kind(method), device, variant=variant, indices=selected
    )
    with torch.no_grad():
        raw = model(value).standardized_coefficients
    return _to_standardized(method, raw, payload)


def _fixed_coefficient(
    method: str, outfit: str, coefficients: Mapping[str, torch.Tensor],
    payload: Mapping[str, Any], device: torch.device,
) -> torch.Tensor:
    mean = payload["coefficient_train_mean"].to(device)
    std = payload["coefficient_train_std"].to(device)
    if method.startswith("B0_"):
        raw = torch.zeros_like(mean)
        return (raw - mean) / std
    return (coefficients[outfit].to(device) - mean) / std


def evaluate_model(
    context: Mapping[str, Any], attempt: Path, experiment: Mapping[str, Any],
    model: nn.Module | None, cache: Mapping[str, Any], basis: Any,
    coefficients: Mapping[str, torch.Tensor], payload: Mapping[str, Any],
    training: Mapping[str, Any], cache_seconds: float, basis_path: Path,
    teacher_residuals: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    started = time.perf_counter()
    method = experiment["method"]
    device = context["base"]._xyz.device
    rank = _rank_for(method)
    mean = payload["coefficient_train_mean"].to(device)
    std = payload["coefficient_train_std"].to(device)
    targets = multi._standardized_targets(coefficients, payload)
    fixed = model is None
    if model is not None:
        model.eval()
    not_applicable = []
    if fixed:
        not_applicable = [
            "correct_vs_swapped_wins", "permutation_max_difference",
            "single_reference_accuracy", "two_reference_dropout_accuracy",
            "zero_replacement_sensitivity", "base_replacement_sensitivity",
        ]
    metadata_seed = 0 if experiment.get("seed") is None else int(experiment["seed"])
    records: list[dict[str, Any]] = [{
        "record_type": "metadata", "seed": metadata_seed,
        "asset_fingerprint": ASSET_FINGERPRINT,
        "target_forward_input_used": False,
        "not_applicable_metrics": not_applicable,
        "efficiency": {
            "trainable_parameter_count": int(training["trainable_parameter_count"]),
            "basis_storage_bytes": int(basis_path.stat().st_size),
            "peak_vram_bytes": int(training["peak_vram_bytes"]),
            "training_time_seconds": float(training["training_time_seconds"]),
            "inference_time_seconds": 0.0, "render_time_seconds": 0.0,
        },
    }]
    coefficient_cache: dict[str, torch.Tensor] = {}
    residual_cache: dict[str, Any] = {}
    rgb_cache: dict[str, torch.Tensor] = {}
    teacher_rgb_cache: dict[str, torch.Tensor] = {}
    render_seconds = 0.0
    inference_seconds = 0.0
    visual_rows = []
    for outfit in OUTFITS:
        panels = []
        basis_teacher = basis(coefficients[outfit], chunk_size=16384)
        teacher_residual = (
            teacher_residuals[outfit] if teacher_residuals is not None else basis_teacher
        )
        for condition in CONDITIONS:
            key = f"{outfit}/{condition}"
            infer_started = time.perf_counter()
            if fixed:
                standardized = _fixed_coefficient(method, outfit, coefficients, payload, device)
                if method.startswith("B0_"):
                    predicted_residual = GaussianClothingResiduals.zeros(context["base"])
                elif method.startswith("B1_") and teacher_residuals is not None:
                    predicted_residual = teacher_residuals[outfit]
                else:
                    predicted_residual = basis_teacher
            else:
                standardized = _predict_coefficients(
                    method, model, cache, key, payload, device
                )
                restored = standardized * std + mean
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
            residual_metrics = core._residual_metric_payload(
                predicted_residual, teacher_residual, context["config"]["basis"]["channel_bounds"]
            )
            render_metrics = core._render_metrics(
                sample, predicted_rgb, predicted_alpha, teacher_rgb, teacher_alpha
            )
            records.append({
                "record_type": "episode", "outfit_id": outfit, "view_id": condition,
                "target_reference_overlap": 0,
                "predicted_standardized_coefficients": standardized.detach().cpu().tolist(),
                "target_standardized_coefficients": targets[outfit].detach().cpu().tolist(),
                "coefficient_normalization": {
                    "mean": mean.detach().cpu().tolist(), "std": std.detach().cpu().tolist(),
                },
                "teacher_standardized_coefficients": {
                    name: value.detach().cpu().tolist() for name, value in targets.items()
                },
                "precomputed_residual_metrics": residual_metrics,
                "residual_metric_source": "production_full_gaussian_field_v1",
                "render_metrics": render_metrics,
            })
            coefficient_cache[key] = standardized.detach()
            residual_cache[key] = predicted_residual
            rgb_cache[key] = predicted_rgb.detach()
            teacher_rgb_cache[key] = teacher_rgb.detach()
            episode_root = attempt / "visuals/episodes"
            episode_root.mkdir(parents=True, exist_ok=True)
            save_render_tensor(
                episode_root / f"{key.replace('/', '_')}_prediction.png", predicted_rgb, 3
            )
            save_render_tensor(
                episode_root / f"{key.replace('/', '_')}_alpha.png", predicted_alpha, 1
            )
            panels.extend([
                (f"{condition} target", sample["target_edit_rgb"].detach().cpu(), 3),
                (f"{condition} base", sample["target_base_rgb"].detach().cpu(), 3),
                (f"{condition} teacher", teacher_rgb.detach().cpu(), 3),
                (f"{condition} prediction", predicted_rgb.detach().cpu(), 3),
            ])
        visual_rows.append((outfit, panels))
    o01._save_contact_sheet(
        attempt / "visuals/five_outfit_four_view_contact_sheet.png", visual_rows
    )

    swap_rows = []
    bounds = context["config"]["basis"]["channel_bounds"]
    for target_outfit in OUTFITS:
        for condition in CONDITIONS:
            target_key = f"{target_outfit}/{condition}"
            correct = coefficient_cache[target_key]
            correct_distance = float(torch.linalg.vector_norm(correct - targets[target_outfit]))
            correct_render = records[
                1 + OUTFITS.index(target_outfit) * len(CONDITIONS) + CONDITIONS.index(condition)
            ]["render_metrics"]["garment_rgb_mae"]
            target_teacher = (
                teacher_residuals[target_outfit]
                if teacher_residuals is not None
                else basis(coefficients[target_outfit], chunk_size=16384)
            )
            for source_outfit in OUTFITS:
                if source_outfit == target_outfit:
                    continue
                source_key = f"{source_outfit}/{condition}"
                if fixed:
                    swapped = correct
                    swapped_residual = residual_cache[target_key]
                    swapped_rgb = rgb_cache[target_key]
                else:
                    swapped = coefficient_cache[source_key]
                    swapped_residual = basis(swapped * std + mean, chunk_size=16384)
                    swapped_rgb, _ = multi._render_residual(
                        context, target_outfit, condition, swapped_residual
                    )
                garment = diagnosis._garment_mask(context["samples"][target_key])
                swapped_render = o01._masked_mae(
                    swapped_rgb, teacher_rgb_cache[target_key], garment
                )
                correct_residual_error = parameterization.residual_metrics(
                    residual_cache[target_key], target_teacher, bounds, active_epsilon=1e-8
                )["normalized_rmse"]
                swapped_residual_error = parameterization.residual_metrics(
                    swapped_residual, target_teacher, bounds, active_epsilon=1e-8
                )["normalized_rmse"]
                swapped_distance = float(torch.linalg.vector_norm(swapped - targets[target_outfit]))
                records.append({
                    "record_type": "swap", "source_outfit": source_outfit,
                    "target_outfit": target_outfit, "view_id": condition,
                    "correct_wins": bool(
                        correct_distance < swapped_distance
                        and float(correct_residual_error) < float(swapped_residual_error)
                        and correct_render < swapped_render
                    ),
                })
                if condition == "cond_000318":
                    swap_rows.append((f"{target_outfit}<-{source_outfit}", [
                        ("teacher", teacher_rgb_cache[target_key].detach().cpu(), 3),
                        ("correct", rgb_cache[target_key].detach().cpu(), 3),
                        ("swapped", swapped_rgb.detach().cpu(), 3),
                    ]))
    o01._save_contact_sheet(
        attempt / "visuals/reference_swap_contact_sheet.png", swap_rows
    )

    for outfit in OUTFITS:
        for condition in CONDITIONS:
            key = f"{outfit}/{condition}"
            correct = coefficient_cache[key]
            if fixed:
                permutation = single = dropout = zero = base = correct
            else:
                reference_count = _reference_count(method)
                permutation_indices = {
                    1: (0,), 2: (1, 0), 3: (2, 0, 1),
                }[reference_count]
                dropout_indices = tuple(range(max(1, reference_count - 1)))
                permutation = _predict_coefficients(
                    method, model, cache, key, payload, device,
                    indices=permutation_indices,
                )
                single = _predict_coefficients(
                    method, model, cache, key, payload, device, indices=(0,)
                )
                dropout = _predict_coefficients(
                    method, model, cache, key, payload, device,
                    indices=dropout_indices,
                )
                normal_indices = tuple(range(_reference_count(method)))
                zero = _predict_coefficients(
                    method, model, cache, key, payload, device,
                    variant="zero", indices=normal_indices,
                )
                base = _predict_coefficients(
                    method, model, cache, key, payload, device,
                    variant="base", indices=normal_indices,
                )
            records.extend([
                {
                    "record_type": "permutation", "outfit_id": outfit,
                    "view_id": condition,
                    "max_difference": float((correct - permutation).abs().max()),
                },
                {
                    "record_type": "single_reference", "outfit_id": outfit,
                    "view_id": condition, "correct": core._nearest(single, targets) == outfit,
                },
                {
                    "record_type": "two_reference_dropout", "outfit_id": outfit,
                    "view_id": condition, "correct": core._nearest(dropout, targets) == outfit,
                },
                {
                    "record_type": "replacement", "outfit_id": outfit,
                    "view_id": condition,
                    "zero_difference": float(torch.linalg.vector_norm(correct - zero)),
                    "base_difference": float(torch.linalg.vector_norm(correct - base)),
                },
            ])
    records[0]["efficiency"]["inference_time_seconds"] = inference_seconds + cache_seconds
    records[0]["efficiency"]["render_time_seconds"] = render_seconds
    raw = attempt / "raw_metrics/raw_episode_outputs.jsonl"
    raw.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    summary = {
        "status": "PASS", "raw_metrics": str(raw),
        "record_counts": {
            kind: sum(row["record_type"] == kind for row in records)
            for kind in sorted({row["record_type"] for row in records})
        },
        "target_forward_leakage": False, "outfit_id_in_model": False,
        "elapsed_seconds": time.perf_counter() - started,
        "inference_time_seconds": inference_seconds + cache_seconds,
        "render_time_seconds": render_seconds,
    }
    core._atomic_json(attempt / "raw_metrics/formal_evaluation_summary.json", summary)
    return raw, summary


def run_experiment(
    context: Mapping[str, Any], cache: Mapping[str, Any], cache_path: Path,
    cache_sha256: str, cache_seconds: float, attempt: Path,
    experiment: Mapping[str, Any], adapter_contract: Mapping[str, Any],
    training_plan: Mapping[str, Any],
) -> dict[str, Any]:
    environment = core._environment()
    if not environment["cuda_available"]:
        raise RuntimeError("FORMAL_GPU_PREFLIGHT_FAIL")
    probe = torch.arange(4096, device="cuda", dtype=torch.float32).sin().square().sum()
    torch.cuda.synchronize()
    if not torch.isfinite(probe):
        raise RuntimeError("FORMAL_GPU_PREFLIGHT_FAIL: CUDA tensor smoke")
    core._atomic_json(attempt / "preflight/formal_gpu_preflight.json", {
        "status": "PASS", "environment": environment,
        "cuda_tensor_smoke": float(probe), "adapter_contract": dict(adapter_contract),
    })
    core._atomic_json(attempt / "preflight/shared_feature_cache.json", {
        "status": "PASS", "path": str(cache_path), "sha256": cache_sha256,
        "asset_fingerprint": ASSET_FINGERPRINT,
        "target_forward_leakage": False, "outfit_id_in_model": False,
    })
    method = experiment["method"]
    basis, coefficients, payload, basis_path = _basis_artifact(context, method)
    base_before = multi._tensor_state_fingerprint(multi._base_named_tensors(context["base"]))
    backbone_before = o01._state_fingerprint(
        context["legacy_model"].clothing_observation_encoder.backbone.state_dict()
    )
    basis_before = basis.fingerprint()
    trainable = bool(adapter_contract["trainable"])
    if trainable:
        model, training = train_model(
            attempt, experiment, cache, coefficients, payload,
            context["base"]._xyz.device,
        )
        teacher_residuals = None
    else:
        model = None
        training = {
            "status": "PASS", "optimizer_steps": 0, "loss_first": None,
            "loss_last": None, "gradients": {}, "checkpoints": [],
            "checkpoint_resume": {"status": "NOT_APPLICABLE"},
            "training_time_seconds": 0.0,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
            "trainable_parameter_count": 0,
        }
        core._atomic_json(attempt / "raw_metrics/training_summary.json", training)
        teacher_residuals = None
        if method.startswith("B1_"):
            teacher_residuals = multi.load_teacher_residuals(context, OUTFITS)[0]
    raw, evaluation = evaluate_model(
        context, attempt, experiment, model, cache, basis, coefficients, payload,
        training, cache_seconds, basis_path, teacher_residuals,
    )
    base_after = multi._tensor_state_fingerprint(multi._base_named_tensors(context["base"]))
    backbone_after = o01._state_fingerprint(
        context["legacy_model"].clothing_observation_encoder.backbone.state_dict()
    )
    basis_after = basis.fingerprint()
    frozen_gradient_count = (
        multi._base_gradient_count(context["base"])
        + sum(
            parameter.grad is not None
            for parameter in context["legacy_model"].clothing_observation_encoder.backbone.parameters()
        )
        + sum(parameter.grad is not None for parameter in basis.parameters())
    )
    frozen_change = 0.0 if (
        base_before == base_after
        and backbone_before == backbone_after
        and basis_before == basis_after
    ) else math.inf
    result = {
        "status": "PASS", "experiment_id": experiment["experiment_id"],
        "optimizer_created": trainable,
        "optimizer_parameter_scope": training_plan["optimizer_parameter_scope"],
        "global_step": 300 if trainable else 0,
        "checkpoint_milestones": list(MILESTONES if trainable else (0,)),
        "frozen_parameter_max_change": frozen_change,
        "frozen_gradient_count": int(frozen_gradient_count),
        "target_forward_leakage": False, "outfit_id_in_model": False,
        "base_fingerprint_before": base_before, "base_fingerprint_after": base_after,
        "backbone_fingerprint_before": backbone_before,
        "backbone_fingerprint_after": backbone_after,
        "basis_fingerprint_before": basis_before, "basis_fingerprint_after": basis_after,
        "basis_path": str(basis_path), "basis_sha256": core._sha256(basis_path),
        "training": training, "evaluation": evaluation,
        "raw_metrics": str(raw), "environment": environment,
    }
    if frozen_change != 0.0 or frozen_gradient_count != 0:
        raise RuntimeError("frozen parameter contract failed")
    core._atomic_json(attempt / "provenance/executor_result.json", result)
    core._atomic_json(attempt / "RUN_STATUS.json", {
        "status": "TRAINED" if trainable else "EVALUATION_READY",
        "optimizer_steps": 300 if trainable else 0,
        "experiment_id": experiment["experiment_id"],
        "updated_at_unix": time.time(),
    })
    del model, basis
    torch.cuda.empty_cache()
    return result
