from __future__ import annotations

import torch
from torch import nn


class MultiViewClothingAggregator(nn.Module):
    """Aggregate per-anchor observations with shared visibility-aware weights."""

    def __init__(self, input_dim: int, output_dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        for name, value in {"input_dim": input_dim, "output_dim": output_dim}.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not isinstance(eps, (int, float)) or isinstance(eps, bool) or eps <= 0:
            raise ValueError("eps must be a positive number")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.eps = float(eps)
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )
        self.unknown_anchor_feature = nn.Parameter(torch.zeros(1, output_dim))

    def forward(
        self,
        sampled_features: torch.Tensor,
        visibility_confidence: torch.Tensor,
        global_view_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return a weighted mean and a learned feature for unseen anchors."""

        if not isinstance(sampled_features, torch.Tensor) or sampled_features.ndim != 3:
            raise ValueError("sampled_features must be a Tensor[K,A,C]")
        num_views, num_anchors, channels = sampled_features.shape
        if channels != self.input_dim or num_views <= 0 or num_anchors <= 0:
            raise ValueError(
                f"sampled_features must have non-empty shape [K,A,{self.input_dim}]"
            )
        if not torch.is_floating_point(sampled_features) or not torch.isfinite(
            sampled_features
        ).all():
            raise ValueError("sampled_features must be finite and floating point")
        expected_visibility = (num_views, num_anchors, 1)
        if not isinstance(visibility_confidence, torch.Tensor) or tuple(
            visibility_confidence.shape
        ) != expected_visibility:
            raise ValueError(
                f"visibility_confidence must have shape {expected_visibility}"
            )
        visibility = visibility_confidence.to(
            device=sampled_features.device, dtype=sampled_features.dtype
        )
        if (
            not torch.isfinite(visibility).all()
            or torch.any(visibility < 0)
            or torch.any(visibility > 1)
        ):
            raise ValueError("visibility_confidence values must be finite and in [0,1]")
        if global_view_weights is None:
            global_weights = sampled_features.new_ones(num_views, 1)
        else:
            if not isinstance(global_view_weights, torch.Tensor):
                raise TypeError("global_view_weights must be a torch.Tensor or None")
            if global_view_weights.shape == (num_views,):
                global_view_weights = global_view_weights.unsqueeze(-1)
            if global_view_weights.shape != (num_views, 1):
                raise ValueError("global_view_weights must have shape [K] or [K,1]")
            global_weights = global_view_weights.to(
                device=sampled_features.device, dtype=sampled_features.dtype
            )
            if (
                not torch.isfinite(global_weights).all()
                or torch.any(global_weights < 0)
                or torch.any(global_weights > 1)
            ):
                raise ValueError("global_view_weights values must be finite and in [0,1]")

        weights = visibility * global_weights[:, None, :]
        weight_sums = weights.sum(dim=0)
        normalized = weights / (weight_sums.unsqueeze(0) + self.eps)
        projected = self.feature_projection(sampled_features)
        aggregated = (normalized * projected).sum(dim=0)
        invisible = weight_sums <= self.eps
        aggregated = torch.where(
            invisible.expand(-1, self.output_dim),
            self.unknown_anchor_feature.expand(num_anchors, -1),
            aggregated,
        )
        return {
            "anchor_clothing_features": aggregated,
            "anchor_visibility": weight_sums.clamp(0, 1),
            "normalized_view_weights": normalized,
        }
