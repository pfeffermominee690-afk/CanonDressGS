from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ReferenceTokenOutput:
    tokens: torch.Tensor
    raw_tokens: torch.Tensor
    cloth_mean: torch.Tensor
    cloth_max: torch.Tensor
    foreground_mean: torch.Tensor
    cloth_foreground_difference: torch.Tensor
    mask_area: torch.Tensor
    bbox_aspect: torch.Tensor
    centroid: torch.Tensor
    view_direction: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class ReferenceSetCoefficientOutput:
    coefficient: torch.Tensor
    set_feature: torch.Tensor
    per_reference_features: torch.Tensor
    attention_weights: torch.Tensor


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")


def _normalize_valid_mask(
    value: torch.Tensor | None,
    count: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    if value is None:
        result = reference.new_ones(count, 1)
    elif value.shape == (count,):
        result = value.reshape(count, 1).to(reference)
    elif value.shape == (count, 1):
        result = value.to(reference)
    else:
        raise ValueError(
            "reference_valid_mask must have shape [K] or [K,1], got "
            f"{tuple(value.shape)}"
        )
    _require_finite("reference_valid_mask", result)
    if torch.any(result < 0) or torch.any(result > 1):
        raise ValueError("reference_valid_mask values must be in [0,1]")
    return result


def _normalize_masks(name: str, value: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    expected = (images.shape[0], 1, images.shape[2], images.shape[3])
    if value.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")
    result = value.to(images)
    _require_finite(name, result)
    if torch.any(result < 0):
        raise ValueError(f"{name} cannot contain negative values")
    maximum = result.max()
    if maximum.item() > 1:
        if maximum.item() > 255:
            raise ValueError(f"{name} values must be in [0,1] or [0,255]")
        result = result / 255.0
    return result.clamp(0, 1)


class MaskAwareReferenceTokenEncoderV1(nn.Module):
    """Pool frozen spatial image features into one token per reference view.

    The forward signature intentionally contains no target-view, outfit-id, or
    teacher field.  ``spatial_backbone`` may be a frozen submodule shared with
    the existing observation encoder; this module never changes its parameters.
    """

    metadata_dim = 7  # area, bbox aspect, centroid x/y, view direction x/y/z

    def __init__(
        self,
        spatial_backbone: nn.Module,
        feature_dim: int,
        token_dim: int = 128,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or token_dim <= 0 or hidden_dim <= 0:
            raise ValueError("feature_dim, token_dim, and hidden_dim must be positive")
        self.spatial_backbone = spatial_backbone
        self.feature_dim = int(feature_dim)
        self.token_dim = int(token_dim)
        for parameter in self.spatial_backbone.parameters():
            parameter.requires_grad_(False)
        raw_dim = 4 * self.feature_dim + self.metadata_dim
        self.token_adapter = nn.Sequential(
            nn.LayerNorm(raw_dim),
            nn.Linear(raw_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, token_dim),
            nn.LayerNorm(token_dim),
        )

    @staticmethod
    def _validate_inputs(
        reference_images: torch.Tensor,
        reference_poses: torch.Tensor,
        reference_w2c: torch.Tensor,
    ) -> None:
        if reference_images.ndim != 4 or reference_images.shape[1] != 3:
            raise ValueError("reference_images must have shape [K,3,H,W]")
        if not torch.is_floating_point(reference_images):
            raise TypeError("reference_images must be floating point")
        if reference_images.shape[0] < 1 or reference_images.shape[0] > 3:
            raise ValueError("mask-aware coefficient fusion supports K in {1,2,3}")
        _require_finite("reference_images", reference_images)
        if torch.any(reference_images < 0) or torch.any(reference_images > 1):
            raise ValueError("reference_images values must be in [0,1]")
        count = reference_images.shape[0]
        if reference_poses.ndim != 2 or reference_poses.shape[0] != count:
            raise ValueError("reference_poses must have shape [K,P]")
        if reference_w2c.shape != (count, 4, 4):
            raise ValueError("reference_w2c must have shape [K,4,4]")
        _require_finite("reference_poses", reference_poses)
        _require_finite("reference_w2c", reference_w2c)

    @staticmethod
    def _weighted_mean(features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        denominator = masks.sum(dim=(2, 3)).clamp_min(1e-8)
        return (features * masks).sum(dim=(2, 3)) / denominator

    @staticmethod
    def _masked_max(features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        selected = masks > 0
        lowest = torch.finfo(features.dtype).min
        pooled = features.masked_fill(~selected, lowest).amax(dim=(2, 3))
        empty = ~selected.flatten(2).any(dim=2)
        return torch.where(empty, torch.zeros_like(pooled), pooled)

    @staticmethod
    def _mask_geometry(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        count, _, height, width = mask.shape
        area = mask.mean(dim=(2, 3))
        binary = mask[:, 0] > 0
        aspect_values, centroids = [], []
        for index in range(count):
            rows, columns = torch.where(binary[index])
            if rows.numel() == 0:
                aspect = mask.new_zeros(1)
                centroid = mask.new_zeros(2)
            else:
                box_height = (rows.max() - rows.min() + 1).to(mask.dtype)
                box_width = (columns.max() - columns.min() + 1).to(mask.dtype)
                aspect = (box_width / box_height.clamp_min(1)).reshape(1)
                centroid = torch.stack((
                    columns.to(mask.dtype).mean() / max(width - 1, 1),
                    rows.to(mask.dtype).mean() / max(height - 1, 1),
                ))
            aspect_values.append(aspect)
            centroids.append(centroid)
        return area, torch.stack(aspect_values), torch.stack(centroids)

    @staticmethod
    def _view_direction(reference_w2c: torch.Tensor) -> torch.Tensor:
        camera_forward = reference_w2c.new_tensor([0.0, 0.0, 1.0]).expand(
            reference_w2c.shape[0], 3
        )
        direction = torch.bmm(reference_w2c[:, :3, :3].transpose(1, 2), camera_forward.unsqueeze(-1)).squeeze(-1)
        return F.normalize(direction, dim=-1, eps=1e-8)

    def forward(
        self,
        reference_images: torch.Tensor,
        reference_clothing_masks: torch.Tensor,
        reference_foreground_masks: torch.Tensor,
        reference_poses: torch.Tensor,
        reference_w2c: torch.Tensor,
        reference_valid_mask: torch.Tensor | None = None,
    ) -> ReferenceTokenOutput:
        self._validate_inputs(reference_images, reference_poses, reference_w2c)
        cloth = _normalize_masks("reference_clothing_masks", reference_clothing_masks, reference_images)
        foreground = _normalize_masks("reference_foreground_masks", reference_foreground_masks, reference_images)
        valid = _normalize_valid_mask(reference_valid_mask, reference_images.shape[0], reference_images)
        if torch.any(cloth > foreground + 1e-4):
            raise ValueError("reference clothing masks must be a subset of foreground masks")

        # The backbone is frozen, but gradients must still flow from the scalar
        # coefficient through its outputs to the trainable token adapter.
        feature_maps = self.spatial_backbone(reference_images)
        if feature_maps.ndim != 4 or feature_maps.shape[:2] != (
            reference_images.shape[0], self.feature_dim
        ):
            raise ValueError(
                "spatial backbone returned an unexpected shape: "
                f"{tuple(feature_maps.shape)}"
            )
        _require_finite("spatial feature maps", feature_maps)
        cloth_features = F.interpolate(cloth, feature_maps.shape[-2:], mode="area").clamp(0, 1)
        foreground_features = F.interpolate(foreground, feature_maps.shape[-2:], mode="area").clamp(0, 1)
        cloth_mean = self._weighted_mean(feature_maps, cloth_features)
        cloth_max = self._masked_max(feature_maps, cloth_features)
        foreground_mean = self._weighted_mean(feature_maps, foreground_features)
        difference = cloth_mean - foreground_mean
        area, aspect, centroid = self._mask_geometry(cloth)
        view_direction = self._view_direction(reference_w2c.to(reference_images))
        raw = torch.cat(
            (cloth_mean, cloth_max, foreground_mean, difference, area, aspect, centroid, view_direction),
            dim=-1,
        )
        tokens = self.token_adapter(raw) * valid
        return ReferenceTokenOutput(
            tokens=tokens,
            raw_tokens=raw,
            cloth_mean=cloth_mean,
            cloth_max=cloth_max,
            foreground_mean=foreground_mean,
            cloth_foreground_difference=difference,
            mask_area=area,
            bbox_aspect=aspect,
            centroid=centroid,
            view_direction=view_direction,
            valid_mask=valid,
        )


class ReferenceSetCoefficientFusionV1(nn.Module):
    """Permutation-invariant K-reference fusion with one bounded coefficient."""

    def __init__(self, token_dim: int = 128, hidden_dim: int = 128) -> None:
        super().__init__()
        if token_dim <= 0 or hidden_dim <= 0:
            raise ValueError("token_dim and hidden_dim must be positive")
        self.reference_mlp = nn.Sequential(
            nn.Linear(token_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)
        )
        self.attention_score = nn.Linear(hidden_dim, 1)
        self.coefficient_head = nn.Sequential(
            nn.LayerNorm(3 * hidden_dim),
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        reference_tokens: torch.Tensor,
        reference_valid_mask: torch.Tensor | None = None,
    ) -> ReferenceSetCoefficientOutput:
        if reference_tokens.ndim != 2:
            raise ValueError("reference_tokens must have shape [K,D]")
        if reference_tokens.shape[0] < 1 or reference_tokens.shape[0] > 3:
            raise ValueError("reference-set fusion supports K in {1,2,3}")
        _require_finite("reference_tokens", reference_tokens)
        valid = _normalize_valid_mask(
            reference_valid_mask, reference_tokens.shape[0], reference_tokens
        )
        if valid.sum().item() <= 0:
            raise ValueError("reference set must contain at least one valid view")
        features = self.reference_mlp(reference_tokens)
        binary_valid = valid > 0
        denominator = valid.sum(dim=0, keepdim=True).clamp_min(1e-8)
        mean_feature = (features * valid).sum(dim=0, keepdim=True) / denominator
        lowest = torch.finfo(features.dtype).min
        max_feature = features.masked_fill(~binary_valid, lowest).amax(dim=0, keepdim=True)
        logits = self.attention_score(features).masked_fill(~binary_valid, lowest)
        attention = torch.softmax(logits, dim=0)
        attention_feature = (features * attention).sum(dim=0, keepdim=True)
        set_feature = torch.cat((mean_feature, max_feature, attention_feature), dim=-1)
        coefficient = torch.tanh(self.coefficient_head(set_feature)).reshape(1)
        _require_finite("coefficient", coefficient)
        return ReferenceSetCoefficientOutput(
            coefficient=coefficient,
            set_feature=set_feature,
            per_reference_features=features,
            attention_weights=attention,
        )


def build_reference_coefficient_fusion(
    config: Mapping[str, Any],
    *,
    spatial_backbone: nn.Module,
    feature_dim: int,
) -> tuple[MaskAwareReferenceTokenEncoderV1, ReferenceSetCoefficientFusionV1] | None:
    fusion = config.get("coefficient_fusion", config)
    fusion_type = fusion.get("type", "legacy")
    if fusion_type == "legacy":
        return None
    if fusion_type != "mask_aware_reference_set_v1":
        raise ValueError(f"unknown coefficient_fusion.type: {fusion_type!r}")
    token_dim = int(fusion.get("token_dim", 128))
    hidden_dim = int(fusion.get("hidden_dim", 128))
    encoder = MaskAwareReferenceTokenEncoderV1(
        spatial_backbone,
        feature_dim=feature_dim,
        token_dim=token_dim,
        hidden_dim=int(fusion.get("token_hidden_dim", 256)),
    )
    predictor = ReferenceSetCoefficientFusionV1(token_dim=token_dim, hidden_dim=hidden_dim)
    return encoder, predictor
