from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from scene.reference_basis_coefficient_fusion import _require_finite


@dataclass(frozen=True)
class MultiOutfitCoefficientOutput:
    standardized_coefficients: torch.Tensor


class MultiOutfitLinearCoefficientControl(nn.Module):
    """Pre-registered LayerNorm -> Linear(K) control with no bounded endpoint."""

    def __init__(self, input_dim: int, rank: int) -> None:
        super().__init__()
        if min(int(input_dim), int(rank)) <= 0:
            raise ValueError("input_dim and rank must be positive")
        self.input_dim = int(input_dim)
        self.rank = int(rank)
        self.normalization = nn.LayerNorm(self.input_dim)
        self.linear = nn.Linear(self.input_dim, self.rank)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, frozen_set_feature: torch.Tensor) -> MultiOutfitCoefficientOutput:
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
        output = self.linear(self.normalization(value)).reshape(self.rank)
        _require_finite("standardized_coefficients", output)
        return MultiOutfitCoefficientOutput(standardized_coefficients=output)


def restore_coefficients(
    standardized: torch.Tensor, train_mean: torch.Tensor, train_std: torch.Tensor,
) -> torch.Tensor:
    if standardized.shape != train_mean.shape or train_mean.shape != train_std.shape:
        raise ValueError("standardized coefficient and train statistics shapes differ")
    if torch.any(train_std <= 0):
        raise ValueError("coefficient train_std must be positive")
    for name, value in (
        ("standardized", standardized), ("train_mean", train_mean), ("train_std", train_std)
    ):
        _require_finite(name, value)
    return standardized * train_std + train_mean


def pairwise_geometry_loss(
    predicted: torch.Tensor, target: torch.Tensor,
) -> torch.Tensor:
    if predicted.ndim != 2 or predicted.shape != target.shape or predicted.shape[0] < 2:
        raise ValueError("paired coefficient matrices must share shape [M,K] with M >= 2")
    _require_finite("predicted coefficients", predicted)
    _require_finite("target coefficients", target)
    predicted_distances = torch.pdist(predicted, p=2)
    target_distances = torch.pdist(target, p=2)
    return (predicted_distances - target_distances).abs().mean()
