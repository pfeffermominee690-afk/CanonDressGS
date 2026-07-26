from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from scene.support_aware_region_trusted_objective_v6 import (
    _nchw,
    _normalized,
    multiscale_masked_charbonnier,
    target_progress_margin_loss,
)
from scene.trusted_silhouette_semantics_v6_1 import (
    V6_1_OBJECTIVE_NAME,
    active_normalized_asymmetric_removal,
    active_normalized_asymmetric_underfill,
    build_trusted_silhouette_regions,
)
from utils.rendering_loss_utils import boundary_aware_transition_alpha_target


@dataclass(frozen=True)
class V6_1LossOutput:
    total: torch.Tensor
    parts: dict[str, torch.Tensor]
    regions: dict[str, torch.Tensor]
    transition_target: torch.Tensor


def support_aware_region_trusted_objective_v6_1(
    pred_rgb: torch.Tensor,
    pred_alpha: torch.Tensor,
    sample: Mapping[str, torch.Tensor],
    *,
    weights: Mapping[str, float],
    residual_loss: torch.Tensor,
    stability_loss: torch.Tensor,
    progress_margin: float,
    change_epsilon: float,
    support_diagonal_ratio: float,
    minimum_radius_pixels: int,
    transition_alpha_target: torch.Tensor | None = None,
) -> V6_1LossOutput:
    prediction = _nchw(pred_rgb, 3, "pred_rgb")
    alpha = _nchw(pred_alpha, 1, "pred_alpha")
    target = _nchw(sample["target_edit_rgb"], 3, "target_edit_rgb").to(prediction)
    base = _nchw(sample["target_base_rgb"], 3, "target_base_rgb").to(prediction)
    target_alpha = _nchw(sample["target_foreground_mask"], 1, "target_foreground_mask").to(alpha)
    base_alpha = _nchw(sample["target_base_foreground_mask"], 1, "target_base_foreground_mask").to(alpha)
    if prediction.shape != target.shape or prediction.shape != base.shape or alpha.shape != target_alpha.shape:
        raise ValueError("V6.1 prediction/target shapes differ")
    regions = build_trusted_silhouette_regions(
        sample,
        prediction,
        support_diagonal_ratio=support_diagonal_ratio,
        minimum_radius_pixels=minimum_radius_pixels,
    )
    trusted_edit = torch.maximum(
        regions["target_garment"],
        torch.maximum(
            regions["old_garment_removal"],
            torch.maximum(regions["trusted_expansion"], regions["trusted_removal"]),
        ),
    )
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
            prediction,
            target,
            base,
            trusted_edit,
            margin=progress_margin,
            change_epsilon=change_epsilon,
        ),
        "trusted_underfill": active_normalized_asymmetric_underfill(alpha, target_alpha, regions["trusted_expansion"]),
        "trusted_removal": active_normalized_asymmetric_removal(alpha, target_alpha, regions["trusted_removal"]),
        "transition_alpha": _normalized(
            F.smooth_l1_loss(alpha, transition_target, reduction="none", beta=1.0), regions["transition"],
        ),
        "identity": multiscale_masked_charbonnier(prediction, base, regions["protected_identity"], scales=(1,))
        + _normalized((alpha - base_alpha).abs(), regions["protected_identity"]),
        "background": multiscale_masked_charbonnier(prediction, base, regions["background"], scales=(1,))
        + _normalized((alpha - base_alpha).abs(), regions["background"]),
        "neutral_preserve": multiscale_masked_charbonnier(prediction, base, regions["neutral_preserve"], scales=(1,))
        + _normalized((alpha - base_alpha).abs(), regions["neutral_preserve"]),
        "residual": residual_loss,
        "stability": stability_loss,
    }
    if set(weights) != set(parts):
        raise ValueError(f"V6.1 loss weights differ from frozen groups: {sorted(set(weights) ^ set(parts))}")
    total = sum(float(weights[name]) * value for name, value in parts.items())
    if not torch.isfinite(total):
        raise FloatingPointError("V6.1 objective is non-finite")
    return V6_1LossOutput(total=total, parts=parts, regions=regions, transition_target=transition_target)
