from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from scene.reference_basis_coefficient_fusion import (
    MaskAwareReferenceTokenEncoderV1,
    _normalize_masks,
    _normalize_valid_mask,
    _require_finite,
)


@dataclass(frozen=True)
class FrozenF2FeatureOutput:
    per_reference_f2: torch.Tensor
    set_mean: torch.Tensor
    set_max: torch.Tensor
    set_feature: torch.Tensor
    clothing_mean: torch.Tensor
    clothing_max: torch.Tensor
    resized_clothing_mask: torch.Tensor
    pooling_denominator: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class LinearCoefficientOutput:
    raw_logit: torch.Tensor
    coefficient: torch.Tensor


class FrozenF2ReferenceFeatureExtractor(nn.Module):
    """Exact frozen F2 extraction followed by deterministic set mean/max."""

    def __init__(self, spatial_backbone: nn.Module, feature_dim: int) -> None:
        super().__init__()
        if not isinstance(feature_dim, int) or isinstance(feature_dim, bool) or feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer")
        self.spatial_backbone = spatial_backbone
        self.feature_dim = feature_dim
        self.per_reference_dim = 2 * feature_dim
        self.set_feature_dim = 2 * self.per_reference_dim
        for parameter in self.spatial_backbone.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        reference_images: torch.Tensor,
        reference_clothing_masks: torch.Tensor,
        reference_valid_mask: torch.Tensor | None = None,
    ) -> FrozenF2FeatureOutput:
        if reference_images.ndim != 4 or reference_images.shape[1] != 3:
            raise ValueError("reference_images must have shape [K,3,H,W]")
        if reference_images.shape[0] < 1 or reference_images.shape[0] > 3:
            raise ValueError("frozen F2 control supports K in {1,2,3}")
        if not torch.is_floating_point(reference_images):
            raise TypeError("reference_images must be floating point")
        _require_finite("reference_images", reference_images)
        if torch.any(reference_images < 0) or torch.any(reference_images > 1):
            raise ValueError("reference_images values must be in [0,1]")
        masks = _normalize_masks(
            "reference_clothing_masks", reference_clothing_masks, reference_images
        )
        valid = _normalize_valid_mask(
            reference_valid_mask, reference_images.shape[0], reference_images
        )
        if valid.sum().item() <= 0:
            raise ValueError("reference set must contain at least one valid view")
        feature_maps = self.spatial_backbone(reference_images)
        expected = (reference_images.shape[0], self.feature_dim)
        if feature_maps.ndim != 4 or feature_maps.shape[:2] != expected:
            raise ValueError(
                f"spatial backbone returned {tuple(feature_maps.shape)}, expected [K,{self.feature_dim},h,w]"
            )
        _require_finite("spatial feature maps", feature_maps)
        resized = F.interpolate(masks, feature_maps.shape[-2:], mode="area").clamp(0, 1)
        denominator = resized.sum(dim=(2, 3))
        clothing_mean = MaskAwareReferenceTokenEncoderV1._weighted_mean(feature_maps, resized)
        clothing_max = MaskAwareReferenceTokenEncoderV1._masked_max(feature_maps, resized)
        per_reference = torch.cat((clothing_mean, clothing_max), dim=-1) * valid
        valid_count = valid.sum(dim=0, keepdim=True).clamp_min(1e-8)
        set_mean = (per_reference * valid).sum(dim=0, keepdim=True) / valid_count
        lowest = torch.finfo(per_reference.dtype).min
        set_max = per_reference.masked_fill(valid <= 0, lowest).amax(dim=0, keepdim=True)
        set_feature = torch.cat((set_mean, set_max), dim=-1)
        return FrozenF2FeatureOutput(
            per_reference_f2=per_reference,
            set_mean=set_mean,
            set_max=set_max,
            set_feature=set_feature,
            clothing_mean=clothing_mean,
            clothing_max=clothing_max,
            resized_clothing_mask=resized,
            pooling_denominator=denominator,
            valid_mask=valid,
        )


class FrozenF2LinearLogitControl(nn.Module):
    """The pre-registered LayerNorm -> Linear scalar coefficient control."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        if not isinstance(input_dim, int) or isinstance(input_dim, bool) or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        self.input_dim = input_dim
        self.normalization = nn.LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, frozen_set_feature: torch.Tensor) -> LinearCoefficientOutput:
        if frozen_set_feature.ndim == 1:
            value = frozen_set_feature.unsqueeze(0)
        elif frozen_set_feature.ndim == 2 and frozen_set_feature.shape[0] == 1:
            value = frozen_set_feature
        else:
            raise ValueError("frozen_set_feature must have shape [D] or [1,D]")
        if value.shape[1] != self.input_dim:
            raise ValueError(
                f"frozen_set_feature dimension must be {self.input_dim}, got {value.shape[1]}"
            )
        _require_finite("frozen_set_feature", value)
        raw_logit = self.linear(self.normalization(value)).reshape(1)
        coefficient = torch.tanh(raw_logit)
        _require_finite("raw_logit", raw_logit)
        _require_finite("coefficient", coefficient)
        return LinearCoefficientOutput(raw_logit=raw_logit, coefficient=coefficient)
