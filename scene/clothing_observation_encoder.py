from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ClothingObservationEncoder(nn.Module):
    """Encode masked multi-view clothing observations into one global embedding."""

    def __init__(
        self,
        embedding_dim: int = 64,
        backbone_name: str = "small_cnn",
        feature_dim: int = 128,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        for name, value in {
            "embedding_dim": embedding_dim,
            "feature_dim": feature_dim,
        }.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int, got {type(value)!r}")
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if backbone_name != "small_cnn":
            raise ValueError(
                "offline ClothingObservationEncoder currently supports only "
                "backbone_name='small_cnn'"
            )
        if not isinstance(freeze_backbone, bool):
            raise TypeError("freeze_backbone must be a bool")

        middle_dim = min(64, feature_dim)
        first_dim = min(32, middle_dim)
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        self.backbone_name = backbone_name
        self.freeze_backbone = freeze_backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, first_dim, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(first_dim, middle_dim, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(middle_dim, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection_head = nn.Linear(feature_dim, embedding_dim)
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)

    def forward(
        self,
        reference_images: torch.Tensor,
        reference_cloth_masks: torch.Tensor,
        reference_valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return per-view features and a permutation-invariant global embedding."""

        self._validate_images(reference_images)
        masks = self._normalize_cloth_masks(reference_cloth_masks, reference_images)
        valid = self._normalize_valid_mask(
            reference_valid_mask,
            reference_images.shape[0],
            reference_images,
        )
        masked_images = reference_images * masks
        feature_maps = self.backbone[:-1](masked_images)
        per_view_features = self.backbone[-1](feature_maps).flatten(1)
        feature_cloth_masks = F.interpolate(
            masks,
            size=feature_maps.shape[-2:],
            mode="area",
        ).clamp(0, 1)
        view_weights = masks.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(-1) * valid
        if not torch.isfinite(view_weights).all():
            raise ValueError("computed clothing view weights must be finite")
        weight_sum = view_weights.sum()
        if weight_sum.item() <= 0:
            raise ValueError(
                "all reference views are invalid or have zero clothing-mask area"
            )
        pooled_feature = (
            per_view_features * view_weights
        ).sum(dim=0, keepdim=True) / weight_sum
        global_embedding = self.embedding_norm(self.projection_head(pooled_feature))
        return {
            "global_clothing_embedding": global_embedding,
            "per_view_features": per_view_features,
            "feature_maps": feature_maps,
            "feature_cloth_masks": feature_cloth_masks,
            "feature_height": torch.tensor(
                feature_maps.shape[-2], device=feature_maps.device, dtype=torch.long
            ),
            "feature_width": torch.tensor(
                feature_maps.shape[-1], device=feature_maps.device, dtype=torch.long
            ),
            "view_weights": view_weights,
        }

    @staticmethod
    def _validate_images(reference_images: torch.Tensor) -> None:
        if not isinstance(reference_images, torch.Tensor):
            raise TypeError("reference_images must be a torch.Tensor")
        if reference_images.ndim != 4 or reference_images.shape[1] != 3:
            raise ValueError(
                "reference_images must have shape [K,3,H,W], got "
                f"{tuple(reference_images.shape)}"
            )
        if reference_images.shape[0] <= 0:
            raise ValueError("reference_images must contain at least one view")
        if reference_images.shape[2] <= 0 or reference_images.shape[3] <= 0:
            raise ValueError("reference image height and width must be positive")
        if not torch.is_floating_point(reference_images):
            raise TypeError("reference_images must have a floating-point dtype")
        if not torch.isfinite(reference_images).all():
            raise ValueError("reference_images contains NaN or Inf")
        if torch.any(reference_images < 0) or torch.any(reference_images > 1):
            raise ValueError("reference_images values must be in [0,1]")

    @staticmethod
    def _normalize_cloth_masks(
        reference_cloth_masks: torch.Tensor,
        reference_images: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(reference_cloth_masks, torch.Tensor):
            raise TypeError("reference_cloth_masks must be a torch.Tensor")
        expected_shape = (
            reference_images.shape[0],
            1,
            reference_images.shape[2],
            reference_images.shape[3],
        )
        if tuple(reference_cloth_masks.shape) != expected_shape:
            raise ValueError(
                "reference_cloth_masks must have shape "
                f"{expected_shape}, got {tuple(reference_cloth_masks.shape)}"
            )
        masks = reference_cloth_masks.to(
            device=reference_images.device,
            dtype=reference_images.dtype,
        )
        if not torch.isfinite(masks).all():
            raise ValueError("reference_cloth_masks contains NaN or Inf")
        if torch.any(masks < 0):
            raise ValueError("reference_cloth_masks cannot contain negative values")
        maximum = masks.max()
        if maximum.item() > 1:
            if maximum.item() > 255:
                raise ValueError("reference_cloth_masks values must be in [0,1] or [0,255]")
            masks = masks / 255
        return masks.clamp(0, 1)

    @staticmethod
    def _normalize_valid_mask(
        reference_valid_mask: torch.Tensor | None,
        num_views: int,
        reference_images: torch.Tensor,
    ) -> torch.Tensor:
        if reference_valid_mask is None:
            return reference_images.new_ones(num_views, 1)
        if not isinstance(reference_valid_mask, torch.Tensor):
            raise TypeError("reference_valid_mask must be a torch.Tensor or None")
        if reference_valid_mask.shape == (num_views,):
            valid = reference_valid_mask.unsqueeze(-1)
        elif reference_valid_mask.shape == (num_views, 1):
            valid = reference_valid_mask
        else:
            raise ValueError(
                "reference_valid_mask must have shape [K] or [K,1], got "
                f"{tuple(reference_valid_mask.shape)}"
            )
        valid = valid.to(device=reference_images.device, dtype=reference_images.dtype)
        if not torch.isfinite(valid).all() or torch.any(valid < 0) or torch.any(valid > 1):
            raise ValueError("reference_valid_mask values must be finite and in [0,1]")
        return valid
