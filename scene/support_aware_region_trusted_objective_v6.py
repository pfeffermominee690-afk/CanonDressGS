from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from scene.gaussian_clothing_residuals import GaussianClothingResiduals
from scene.representation_capacity_oracle import garment_trainable_mask
from utils.rendering_loss_utils import boundary_aware_transition_alpha_target


V6_OBJECTIVE_NAME = "SUPPORT_AWARE_REGION_TRUSTED_OBJECTIVE_V6"
TARGET_PROGRESS_NAME = "TARGET_PROGRESS_MARGIN_LOSS"
REGION_NAMES = (
    "protected_identity",
    "background",
    "target_garment",
    "old_garment_removal",
    "new_silhouette",
    "transition",
    "neutral_preserve",
)


def _nchw(value: torch.Tensor, channels: int, name: str) -> torch.Tensor:
    if value.ndim == 3 and value.shape[0] == channels:
        result = value.unsqueeze(0)
    elif value.ndim == 4 and value.shape[1] == channels:
        result = value
    else:
        raise ValueError(f"{name} must have shape [C,H,W] or [N,C,H,W]")
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return result


def _mask(value: torch.Tensor, reference: torch.Tensor, name: str) -> torch.Tensor:
    result = _nchw(value, 1, name).to(reference)
    if result.shape[0] == 1 and reference.shape[0] > 1:
        result = result.expand(reference.shape[0], -1, -1, -1)
    if result.shape[0] != reference.shape[0] or result.shape[-2:] != reference.shape[-2:]:
        raise ValueError(f"{name} does not match prediction batch/spatial shape")
    if torch.any(result < 0) or torch.any(result > 1):
        raise ValueError(f"{name} values must be in [0,1]")
    return result.clamp(0, 1)


def _binary(value: torch.Tensor) -> torch.Tensor:
    return (value >= 0.5).to(value)


def build_support_aware_regions(sample: Mapping[str, torch.Tensor], reference: torch.Tensor) -> dict[str, torch.Tensor]:
    """Build one deterministic priority-disjoint V6 supervision partition."""

    protected = _binary(_mask(sample["target_protected_mask"], reference, "target_protected_mask"))
    target_fg = _binary(_mask(sample["target_foreground_mask"], reference, "target_foreground_mask"))
    base_fg = _binary(_mask(sample["target_base_foreground_mask"], reference, "target_base_foreground_mask"))
    clothing = _binary(_mask(sample["target_clothing_mask"], reference, "target_clothing_mask"))
    old_clothing = _binary(_mask(sample["target_old_clothing_mask"], reference, "target_old_clothing_mask"))
    edit = _binary(_mask(sample["target_edit_mask"], reference, "target_edit_mask"))
    transition_source = _binary(_mask(sample["target_transition_mask"], reference, "target_transition_mask"))
    preserve_source = _binary(_mask(sample["target_preserve_mask"], reference, "target_preserve_mask"))

    new_silhouette = target_fg * (1 - base_fg) * (1 - protected)
    target_garment = clothing * (1 - protected) * (1 - new_silhouette)
    old_removal = old_clothing * edit * (1 - clothing) * (1 - protected) * (1 - new_silhouette)
    occupied = torch.maximum(new_silhouette, torch.maximum(target_garment, torch.maximum(old_removal, protected)))
    transition = transition_source * (1 - occupied)
    occupied = torch.maximum(occupied, transition)
    neutral = preserve_source * base_fg * (1 - old_clothing) * (1 - edit) * (1 - occupied)
    occupied = torch.maximum(occupied, neutral)
    background = (1 - torch.maximum(target_fg, base_fg)) * (1 - occupied)

    regions = {
        "protected_identity": protected,
        "background": background,
        "target_garment": target_garment,
        "old_garment_removal": old_removal,
        "new_silhouette": new_silhouette,
        "transition": transition,
        "neutral_preserve": neutral,
    }
    overlap = torch.zeros_like(protected)
    for value in regions.values():
        overlap = overlap + _binary(value)
    if torch.any(overlap > 1):
        raise AssertionError("V6 supervision regions must be priority-disjoint")
    if torch.any((target_garment + old_removal + new_silhouette) * protected > 0):
        raise AssertionError("protected identity entered garment supervision")
    if torch.any(old_removal * neutral > 0):
        raise AssertionError("old garment removal entered neutral preserve")
    return regions


def _normalized(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.expand(-1, value.shape[1], -1, -1)
    return (value * expanded).sum() / expanded.sum().clamp_min(1)


def multiscale_masked_charbonnier(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    scales: tuple[int, ...] = (1, 2, 4),
    epsilon: float = 1e-3,
) -> torch.Tensor:
    if epsilon <= 0 or not scales:
        raise ValueError("Charbonnier epsilon/scales are invalid")
    values = []
    for scale in scales:
        if scale <= 0:
            raise ValueError("Charbonnier scales must be positive")
        if scale == 1:
            pred, truth, weight = prediction, target, mask
        else:
            pred = F.avg_pool2d(prediction, scale, stride=scale)
            truth = F.avg_pool2d(target, scale, stride=scale)
            weight = F.avg_pool2d(mask, scale, stride=scale)
        values.append(_normalized(torch.sqrt((pred - truth).square() + epsilon * epsilon), weight))
    return torch.stack(values).mean()


def target_progress_margin_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    base: torch.Tensor,
    trusted_edit_mask: torch.Tensor,
    *,
    margin: float,
    change_epsilon: float,
) -> torch.Tensor:
    """Generic target-vs-base relative margin; contains no outfit/color rule."""

    if margin < 0 or change_epsilon <= 0:
        raise ValueError("target-progress margin/epsilon are invalid")
    d_target = (prediction - target).abs().mean(1, keepdim=True)
    d_base = (prediction - base).abs().mean(1, keepdim=True)
    changed = ((target - base).abs().mean(1, keepdim=True) > change_epsilon).to(prediction)
    active = trusted_edit_mask * changed
    return _normalized(F.relu(d_target - d_base + margin), active)


def _alpha_bce_dice(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred = prediction.clamp(1e-6, 1 - 1e-6)
    bce = _normalized(F.binary_cross_entropy(pred, target, reduction="none"), mask)
    expanded = mask
    intersection = (prediction * target * expanded).sum()
    denominator = ((prediction + target) * expanded).sum()
    dice = 1 - (2 * intersection + 1e-6) / (denominator + 1e-6)
    return bce + dice


@dataclass(frozen=True)
class V6LossOutput:
    total: torch.Tensor
    parts: dict[str, torch.Tensor]
    regions: dict[str, torch.Tensor]
    transition_target: torch.Tensor


def support_aware_region_trusted_objective_v6(
    pred_rgb: torch.Tensor,
    pred_alpha: torch.Tensor,
    sample: Mapping[str, torch.Tensor],
    *,
    weights: Mapping[str, float],
    residual_loss: torch.Tensor,
    stability_loss: torch.Tensor,
    progress_margin: float,
    change_epsilon: float,
    transition_alpha_target: torch.Tensor | None = None,
) -> V6LossOutput:
    prediction = _nchw(pred_rgb, 3, "pred_rgb")
    alpha = _nchw(pred_alpha, 1, "pred_alpha")
    target = _nchw(sample["target_edit_rgb"], 3, "target_edit_rgb").to(prediction)
    base = _nchw(sample["target_base_rgb"], 3, "target_base_rgb").to(prediction)
    target_alpha = _nchw(sample["target_foreground_mask"], 1, "target_foreground_mask").to(alpha)
    base_alpha = _nchw(sample["target_base_foreground_mask"], 1, "target_base_foreground_mask").to(alpha)
    if prediction.shape != target.shape or prediction.shape != base.shape or alpha.shape != target_alpha.shape:
        raise ValueError("V6 prediction/target shapes differ")
    regions = build_support_aware_regions(sample, prediction)
    trusted_edit = torch.maximum(regions["target_garment"], torch.maximum(regions["old_garment_removal"], regions["new_silhouette"]))
    if transition_alpha_target is None:
        transition_target = boundary_aware_transition_alpha_target(
            target_alpha,
            base_alpha,
            trusted_edit,
            torch.maximum(regions["neutral_preserve"], regions["background"]),
            regions["protected_identity"],
        )["target"]
    else:
        transition_target = _nchw(transition_alpha_target, 1, "transition_alpha_target").to(alpha)
        if transition_target.shape != alpha.shape:
            raise ValueError("cached transition alpha target shape differs from prediction")
        if torch.any(transition_target < 0) or torch.any(transition_target > 1):
            raise ValueError("cached transition alpha target values must be in [0,1]")
        if torch.any((transition_target - base_alpha).abs() * regions["protected_identity"] > 1e-6):
            raise ValueError("cached protected transition target must equal base alpha")
    parts = {
        "edit_rgb": multiscale_masked_charbonnier(prediction, target, trusted_edit),
        "target_progress": target_progress_margin_loss(
            prediction, target, base, trusted_edit, margin=progress_margin, change_epsilon=change_epsilon,
        ),
        "new_silhouette": _alpha_bce_dice(alpha, target_alpha, regions["new_silhouette"]),
        "transition_alpha": _normalized(F.smooth_l1_loss(alpha, transition_target, reduction="none", beta=1.0), regions["transition"]),
        "identity": multiscale_masked_charbonnier(prediction, base, regions["protected_identity"], scales=(1,))
        + _normalized((alpha - base_alpha).abs(), regions["protected_identity"]),
        "background": multiscale_masked_charbonnier(prediction, base, regions["background"], scales=(1,))
        + _normalized((alpha - base_alpha).abs(), regions["background"]),
        "neutral_preserve": multiscale_masked_charbonnier(prediction, base, regions["neutral_preserve"], scales=(1,))
        + _normalized((alpha - base_alpha).abs(), regions["neutral_preserve"]),
        "residual": residual_loss,
        "stability": stability_loss,
    }
    expected = set(parts)
    if set(weights) != expected:
        raise ValueError(f"V6 loss weights differ from frozen groups: {sorted(set(weights) ^ expected)}")
    total = sum(float(weights[name]) * value for name, value in parts.items())
    if not torch.isfinite(total):
        raise FloatingPointError("V6 objective is non-finite")
    return V6LossOutput(total=total, parts=parts, regions=regions, transition_target=transition_target)


def bound_normalized_regularization(
    residuals: GaussianClothingResiduals,
    base_model: Any,
    bounds: Mapping[str, float],
    *,
    garment_weight: float = 0.25,
    protected_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not 0 <= garment_weight <= protected_weight:
        raise ValueError("region regularization weights are invalid")
    garment = garment_trainable_mask(base_model).to(base_model._xyz)
    region_weight = garment * garment_weight + (1 - garment) * protected_weight
    mapping = {
        "xyz": residuals.delta_xyz,
        "log_scaling": residuals.delta_log_scaling,
        "rotation": residuals.delta_rotvec,
        "opacity_logit": residuals.delta_opacity_logit,
        "sh0": residuals.delta_sh0,
    }
    parts = {}
    for name, value in mapping.items():
        bound = float(bounds[name])
        if bound <= 0:
            raise ValueError("residual bounds must be positive")
        weight = region_weight[:, 0] if value.ndim == 1 else region_weight
        while weight.ndim < value.ndim:
            weight = weight.unsqueeze(-1)
        parts[name] = ((value / bound).square() * weight).mean()
    return torch.stack(tuple(parts.values())).mean(), parts


def residual_stability_loss(
    residuals: GaussianClothingResiduals,
    base_model: Any,
    bounds: Mapping[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    opacity = torch.sigmoid(base_model._opacity + residuals.delta_opacity_logit)
    scale_ratio = torch.exp(residuals.delta_log_scaling.abs()).amax(1)
    displacement = torch.linalg.vector_norm(residuals.delta_xyz, dim=1)
    hit_terms = []
    for name, value in (
        ("xyz", residuals.delta_xyz),
        ("log_scaling", residuals.delta_log_scaling),
        ("rotation", residuals.delta_rotvec),
        ("opacity_logit", residuals.delta_opacity_logit),
        ("sh0", residuals.delta_sh0),
    ):
        hit_terms.append(F.relu(value.abs() / float(bounds[name]) - 0.95).square().mean())
    parts = {
        "bound_edge": torch.stack(hit_terms).mean(),
        "opacity_saturation": (F.relu(0.005 - opacity) + F.relu(opacity - 0.995)).mean(),
        "scale_abnormal": F.relu(scale_ratio - 8.0).square().mean(),
        "floating_splat": F.relu(displacement - 0.25).square().mean(),
    }
    return torch.stack(tuple(parts.values())).sum(), parts
