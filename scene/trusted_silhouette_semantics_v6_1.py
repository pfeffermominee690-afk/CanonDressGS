from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn.functional as F

from scene.support_aware_region_trusted_objective_v6 import _binary, _mask


V6_1_OBJECTIVE_NAME = "SUPPORT_AWARE_REGION_TRUSTED_OBJECTIVE_V6_1"
V6_1_REGION_NAMES = (
    "protected_identity",
    "background",
    "target_garment",
    "old_garment_removal",
    "trusted_expansion",
    "trusted_removal",
    "silhouette_uncertain",
    "transition",
    "neutral_preserve",
)


def _disk_dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return _binary(mask)
    coordinates = torch.arange(-radius, radius + 1, device=mask.device)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    kernel = ((xx.square() + yy.square()) <= radius * radius).to(mask).reshape(1, 1, 2 * radius + 1, 2 * radius + 1)
    batches, channels, _, _ = mask.shape
    value = F.conv2d(_binary(mask).reshape(batches * channels, 1, *mask.shape[-2:]), kernel, padding=radius)
    return (value > 0).to(mask).reshape_as(mask)


def _bbox_diagonal(mask: torch.Tensor) -> float:
    indices = torch.nonzero(mask >= 0.5, as_tuple=False)
    if not indices.numel():
        return 0.0
    y = indices[:, -2]
    x = indices[:, -1]
    return math.hypot(float(x.max() - x.min() + 1), float(y.max() - y.min() + 1))


def support_radius_pixels(base_foreground: torch.Tensor, diagonal_ratio: float, minimum: int = 1) -> int:
    if diagonal_ratio <= 0 or minimum < 0:
        raise ValueError("support-band radius contract is invalid")
    return max(int(minimum), int(math.ceil(_bbox_diagonal(base_foreground) * float(diagonal_ratio))))


def build_trusted_silhouette_regions(
    sample: Mapping[str, torch.Tensor],
    reference: torch.Tensor,
    *,
    support_diagonal_ratio: float,
    minimum_radius_pixels: int = 1,
) -> dict[str, torch.Tensor]:
    """Build the deterministic V6.1 supervision partition.

    Every input is an existing segmentation/region artifact. No RGB, outfit ID,
    color threshold, teacher signal, or forward gate is consulted.
    """

    protected = _binary(_mask(sample["target_protected_mask"], reference, "target_protected_mask"))
    target_fg = _binary(_mask(sample["target_foreground_mask"], reference, "target_foreground_mask"))
    base_fg = _binary(_mask(sample["target_base_foreground_mask"], reference, "target_base_foreground_mask"))
    safe_clothing = _binary(_mask(sample["target_clothing_mask"], reference, "target_clothing_mask"))
    raw_clothing = _binary(_mask(sample.get("target_clothing_mask_raw", sample["target_clothing_mask"]), reference, "target_clothing_mask_raw"))
    old_clothing = _binary(_mask(sample["target_old_clothing_mask"], reference, "target_old_clothing_mask"))
    edit = _binary(_mask(sample["target_edit_mask"], reference, "target_edit_mask"))
    core = _binary(_mask(sample["target_edit_core_mask"], reference, "target_edit_core_mask"))
    transition_source = _binary(_mask(sample["target_transition_mask"], reference, "target_transition_mask"))
    preserve_source = _binary(_mask(sample["target_preserve_mask"], reference, "target_preserve_mask"))
    artifact_source = sample.get("target_background_artifact_mask", torch.zeros_like(target_fg))
    artifact = _binary(_mask(artifact_source, reference, "target_background_artifact_mask"))

    radii = [support_radius_pixels(base_fg[index:index + 1], support_diagonal_ratio, minimum_radius_pixels) for index in range(base_fg.shape[0])]
    support_rows = []
    for index, radius in enumerate(radii):
        evidence = torch.maximum(core[index:index + 1], torch.maximum(safe_clothing[index:index + 1], old_clothing[index:index + 1]))
        base_neighborhood = base_fg[index:index + 1] * _disk_dilate(evidence, radius)
        support_rows.append(_disk_dilate(torch.maximum(evidence, base_neighborhood), radius))
    support = torch.cat(support_rows, dim=0)

    expand_raw = target_fg * (1 - base_fg)
    remove_raw = base_fg * (1 - target_fg)
    trusted_expansion = expand_raw * safe_clothing * (1 - protected) * (1 - artifact) * support
    trusted_removal = remove_raw * old_clothing * (1 - protected)
    uncertain = torch.maximum(expand_raw, remove_raw) * (1 - torch.maximum(trusted_expansion, trusted_removal)) * (1 - protected)

    target_garment = safe_clothing * (1 - protected) * (1 - trusted_expansion) * (1 - trusted_removal)
    old_removal = old_clothing * edit * (1 - safe_clothing) * (1 - protected)
    old_removal = old_removal * (1 - trusted_expansion) * (1 - trusted_removal) * (1 - target_garment)
    occupied = torch.maximum(protected, torch.maximum(trusted_expansion, torch.maximum(trusted_removal, torch.maximum(target_garment, old_removal))))
    transition = torch.maximum(transition_source, uncertain * raw_clothing) * (1 - occupied)
    occupied = torch.maximum(occupied, transition)
    neutral = preserve_source * base_fg * (1 - old_clothing) * (1 - edit) * (1 - occupied)
    occupied = torch.maximum(occupied, neutral)
    artifact_neighborhood = _disk_dilate(artifact, max(radii)) if radii else artifact
    background = (1 - torch.maximum(target_fg, base_fg)) * (1 - transition) * (1 - artifact_neighborhood) * (1 - occupied)

    regions = {
        "protected_identity": protected,
        "background": background,
        "target_garment": target_garment,
        "old_garment_removal": old_removal,
        "trusted_expansion": trusted_expansion,
        "trusted_removal": trusted_removal,
        "silhouette_uncertain": uncertain,
        "transition": transition,
        "neutral_preserve": neutral,
        "garment_support_band": support,
        "background_artifact": artifact,
        "raw_expansion": expand_raw,
        "raw_removal": remove_raw,
    }
    hard_names = V6_1_REGION_NAMES
    overlap = torch.zeros_like(protected)
    for name in hard_names:
        overlap = overlap + _binary(regions[name])
    if torch.any(overlap > 1):
        raise AssertionError("V6.1 hard supervision regions must be priority-disjoint")
    if torch.any(trusted_expansion * protected > 0) or torch.any(trusted_removal * protected > 0):
        raise AssertionError("protected identity entered trusted silhouette supervision")
    if torch.any(trusted_expansion * (1 - safe_clothing) > 0):
        raise AssertionError("trusted expansion lacks safe garment evidence")
    return regions


def active_normalized_asymmetric_underfill(
    prediction_alpha: torch.Tensor,
    target_alpha: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float = 1.0,
) -> torch.Tensor:
    active = _binary(mask).to(prediction_alpha)
    underfill = F.relu(target_alpha - prediction_alpha)
    values = F.smooth_l1_loss(underfill, torch.zeros_like(underfill), reduction="none", beta=beta)
    return (values * active).sum() / active.sum().clamp_min(1)


def active_normalized_asymmetric_removal(
    prediction_alpha: torch.Tensor,
    target_alpha: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float = 1.0,
) -> torch.Tensor:
    active = _binary(mask).to(prediction_alpha)
    overfill = F.relu(prediction_alpha - target_alpha)
    values = F.smooth_l1_loss(overfill, torch.zeros_like(overfill), reduction="none", beta=beta)
    return (values * active).sum() / active.sum().clamp_min(1)
