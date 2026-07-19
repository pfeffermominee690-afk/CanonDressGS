from __future__ import annotations

from dataclasses import fields
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from scene.gaussian_clothing_residuals import CHANNELS, GaussianClothingResiduals


GEOMETRY_CHANNELS = frozenset({"delta_xyz", "delta_log_scaling", "delta_rotvec"})
APPEARANCE_CHANNELS = frozenset({"delta_opacity_logit", "delta_sh0", "delta_shN"})
COUNTERFACTUAL_CHANNELS = {
    "H0_full": frozenset(CHANNELS),
    "H1_xyz": frozenset({"delta_xyz"}),
    "H2_geometry": GEOMETRY_CHANNELS,
    "H3_appearance": APPEARANCE_CHANNELS,
    "H4_xyz_opacity_sh0": frozenset({"delta_xyz", "delta_opacity_logit", "delta_sh0"}),
    "H5_zero_geometry": APPEARANCE_CHANNELS,
    "H6_zero_appearance": GEOMETRY_CHANNELS,
}
TARGET_FORWARD_FIELDS = frozenset({
    "target_rgb", "target_edit_rgb", "target_base_rgb", "target_foreground_mask",
    "target_clothing_mask", "target_edit_mask", "target_edit_core_mask",
    "target_preserve_mask", "target_transition_mask", "target_protected_mask",
    "teacher", "teacher_residual", "teacher_active_mask", "oracle_residual",
    "outfit_id", "cloth_id",
})


class DiagnosticOutfitLatents(nn.Module):
    """Two diagnostic codes used only by Probe D, never by formal inference."""

    def __init__(self, outfit_ids: Sequence[str], dimension: int, seed: int) -> None:
        super().__init__()
        if len(outfit_ids) < 2 or len(set(outfit_ids)) != len(outfit_ids):
            raise ValueError("Probe D requires at least two unique outfit IDs")
        if dimension <= 0:
            raise ValueError("latent dimension must be positive")
        self.outfit_ids = tuple(outfit_ids)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        initial = torch.randn(len(outfit_ids), dimension, generator=generator) * 0.02
        self.codes = nn.Parameter(initial)

    def forward(self, outfit_id: str) -> torch.Tensor:
        if outfit_id not in self.outfit_ids:
            raise KeyError(f"unknown diagnostic outfit ID: {outfit_id}")
        return self.codes[self.outfit_ids.index(outfit_id)].reshape(1, -1)


def assert_forward_boundary(payload: Mapping[str, Any]) -> None:
    leaked = sorted(TARGET_FORWARD_FIELDS.intersection(payload))
    if leaked:
        raise RuntimeError(f"forbidden diagnostic forward fields: {leaked}")


def reference_variant(
    episode: Mapping[str, Any],
    geometry: Mapping[str, Any],
    mode: str,
    *,
    substitute_images: torch.Tensor | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create a reference-only intervention while preserving target pose/camera."""

    reference_count = int(episode["reference_images"].shape[0])
    if reference_count < 1:
        raise ValueError("reference episode is empty")
    device = episode["reference_images"].device
    if mode == "correct":
        indices = torch.arange(reference_count, device=device)
    elif mode == "permuted":
        indices = torch.arange(reference_count - 1, -1, -1, device=device)
    elif mode == "same_reference_tripled":
        indices = torch.zeros(reference_count, dtype=torch.long, device=device)
    elif mode == "single_reference":
        indices = torch.zeros(1, dtype=torch.long, device=device)
    elif mode in {"zero_rgb_masks_kept", "base_rgb_substitution", "color_perturbation"}:
        indices = torch.arange(reference_count, device=device)
    else:
        raise ValueError(f"unknown reference intervention: {mode}")

    result: dict[str, Any] = {}
    tensor_fields = (
        "reference_images", "reference_cloth_masks", "reference_foreground_masks",
        "reference_poses", "reference_valid_mask",
    )
    for name in tensor_fields:
        result[name] = episode[name].index_select(0, indices)
    cameras = list(episode["reference_cameras"])
    result["reference_cameras"] = [cameras[int(index)] for index in indices.detach().cpu()]
    ids = list(episode.get("reference_condition_ids", range(reference_count)))
    result["reference_condition_ids"] = [ids[int(index)] for index in indices.detach().cpu()]

    geometry_result = dict(geometry)
    for name in ("surface_depth_maps", "surface_alpha_maps"):
        geometry_result[name] = geometry[name].index_select(0, indices)

    if mode == "zero_rgb_masks_kept":
        result["reference_images"] = torch.zeros_like(result["reference_images"])
    elif mode == "base_rgb_substitution":
        if substitute_images is None or substitute_images.shape != episode["reference_images"].shape:
            raise ValueError("base RGB substitution must match the complete reference image tensor")
        result["reference_images"] = substitute_images.index_select(0, indices).to(
            device=result["reference_images"].device,
            dtype=result["reference_images"].dtype,
        )
    elif mode == "color_perturbation":
        channels = result["reference_images"].shape[1]
        if channels != 3:
            raise ValueError("deterministic color perturbation requires RGB references")
        shift = torch.tensor([0.025, -0.015, 0.010], device=device, dtype=result["reference_images"].dtype)
        result["reference_images"] = (result["reference_images"] + shift.reshape(1, 3, 1, 1)).clamp(0, 1)

    return result, geometry_result


def select_residual_channels(
    residuals: GaussianClothingResiduals,
    selected: Sequence[str] | set[str] | frozenset[str],
) -> GaussianClothingResiduals:
    selected_set = frozenset(selected)
    unknown = selected_set.difference(CHANNELS)
    if unknown:
        raise ValueError(f"unknown residual channels: {sorted(unknown)}")
    values: dict[str, torch.Tensor | None] = {}
    for item in fields(residuals):
        value = getattr(residuals, item.name)
        values[item.name] = value if value is None or item.name in selected_set else torch.zeros_like(value)
    return GaussianClothingResiduals(**values)


def normalized_residual_regression_loss(
    prediction: GaussianClothingResiduals,
    target: GaussianClothingResiduals,
    bounds: Mapping[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    names = {
        "delta_xyz": "xyz", "delta_log_scaling": "log_scaling",
        "delta_rotvec": "rotation", "delta_opacity_logit": "opacity_logit",
        "delta_sh0": "sh0", "delta_shN": "shN",
    }
    parts: dict[str, torch.Tensor] = {}
    for name, bound_name in names.items():
        first, second = getattr(prediction, name), getattr(target, name)
        if first is None or second is None or first.shape != second.shape:
            raise ValueError(f"residual regression shape mismatch for {name}")
        bound = float(bounds[bound_name])
        if bound <= 0:
            raise ValueError(f"non-positive residual bound: {bound_name}")
        parts[name] = F.smooth_l1_loss(first / bound, second.to(first) / bound, beta=0.1)
    return sum(parts.values()) / len(parts), parts


def channel_comparison(
    prediction: GaussianClothingResiduals,
    target: GaussianClothingResiduals,
    bounds: Mapping[str, float],
    *,
    active_epsilon: float = 1e-8,
    top_fraction: float = 0.10,
) -> dict[str, Any]:
    names = {
        "delta_xyz": "xyz", "delta_log_scaling": "log_scaling",
        "delta_rotvec": "rotation", "delta_opacity_logit": "opacity_logit",
        "delta_sh0": "sh0", "delta_shN": "shN",
    }
    result: dict[str, Any] = {}
    for name, bound_name in names.items():
        first = getattr(prediction, name).detach().float().reshape(getattr(prediction, name).shape[0], -1)
        second = getattr(target, name).detach().float().reshape(getattr(target, name).shape[0], -1)
        if first.shape != second.shape or not torch.isfinite(first).all() or not torch.isfinite(second).all():
            raise ValueError(f"invalid residual comparison channel: {name}")
        delta = first - second
        flat_delta = delta.abs().reshape(-1)
        first_flat, second_flat = first.reshape(-1), second.reshape(-1)
        cosine = F.cosine_similarity(first_flat[None], second_flat[None], eps=1e-12).item()
        first_norm, second_norm = first.norm(dim=1), second.norm(dim=1)
        count = max(1, int(round(first.shape[0] * float(top_fraction))))
        top_first = torch.topk(first_norm, count).indices
        top_second = torch.topk(second_norm, count).indices
        top_overlap = len(set(top_first.cpu().tolist()).intersection(top_second.cpu().tolist())) / count
        union = torch.logical_or(first_norm > active_epsilon, second_norm > active_epsilon)
        intersection = torch.logical_and(first_norm > active_epsilon, second_norm > active_epsilon)
        bound = float(bounds[bound_name])
        result[name] = {
            "mae": float(flat_delta.mean()),
            "rmse": float(delta.square().mean().sqrt()),
            "bound_normalized_mae": float(flat_delta.mean() / bound),
            "bound_normalized_rmse": float(delta.square().mean().sqrt() / bound),
            "cosine_similarity": float(cosine),
            "absolute_error_p50": float(torch.quantile(flat_delta, 0.50)),
            "absolute_error_p90": float(torch.quantile(flat_delta, 0.90)),
            "absolute_error_p99": float(torch.quantile(flat_delta, 0.99)),
            "prediction_bound_hit_ratio": float((first.abs().amax(dim=1) >= 0.99 * bound).float().mean()),
            "target_bound_exceed_ratio": float((second.abs().amax(dim=1) > bound).float().mean()),
            "prediction_nonzero_ratio": float((first_norm > active_epsilon).float().mean()),
            "target_nonzero_ratio": float((second_norm > active_epsilon).float().mean()),
            "active_support_iou": 1.0 if not union.any() else float(intersection.sum() / union.sum()),
            "top_10pct_row_overlap": float(top_overlap),
        }
    return result


def residual_distance_scalar(comparison: Mapping[str, Mapping[str, float]]) -> float:
    return sum(float(comparison[name]["bound_normalized_rmse"]) for name in CHANNELS) / len(CHANNELS)


def decision_case(
    *,
    probe_d_pass: bool,
    probe_c_ran: bool,
    probe_c_pass: bool,
    probe_r_ran: bool,
    probe_r_degraded: bool,
    original_zero_initialization_failed: bool = True,
) -> dict[str, str]:
    if not probe_d_pass:
        return {"case": "ND", "state": "DECODER_RESIDUAL_CAPACITY_FAILURE", "next_task": "REDESIGN_RESIDUAL_DECODER_CAPACITY"}
    if probe_c_ran and not probe_c_pass:
        return {"case": "NC", "state": "REFERENCE_CONDITIONING_FAILURE", "next_task": "FIX_REFERENCE_CONDITIONING_AND_FEATURE_FUSION"}
    if probe_r_ran and probe_r_degraded:
        return {"case": "NO", "state": "RENDER_OBJECTIVE_OPTIMIZATION_FAILURE", "next_task": "REDESIGN_IMAGE_SPACE_OBJECTIVE_AND_GATE_REGULARIZATION"}
    if probe_r_ran and not probe_r_degraded and original_zero_initialization_failed:
        return {"case": "NS", "state": "INITIALIZATION_OR_SCHEDULE_FAILURE", "next_task": "CALIBRATE_IMAGE_CONDITIONED_INITIALIZATION_AND_SCHEDULE"}
    return {"case": "NM", "state": "MULTIPLE_OR_UNRESOLVED_FAILURES", "next_task": "ISOLATE_REMAINING_IMAGE_CONDITIONED_FAILURES"}
