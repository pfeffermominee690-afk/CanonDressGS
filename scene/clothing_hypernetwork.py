from __future__ import annotations

import torch
from torch import nn


class ClothingFiLMGenerator(nn.Module):
    """Generate lightweight FiLM modulation from one clothing embedding."""

    def __init__(
        self,
        clothing_dim: int,
        hidden_dims: list[int],
        hyper_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        for name, value in {
            "clothing_dim": clothing_dim,
            "hyper_hidden_dim": hyper_hidden_dim,
        }.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int, got {type(value)!r}")
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not isinstance(hidden_dims, list) or not hidden_dims:
            raise ValueError("hidden_dims must be a non-empty list of layer widths")
        for index, width in enumerate(hidden_dims):
            if not isinstance(width, int) or isinstance(width, bool):
                raise TypeError(f"hidden_dims[{index}] must be an int, got {type(width)!r}")
            if width <= 0:
                raise ValueError(f"hidden_dims[{index}] must be positive, got {width}")

        self.clothing_dim = clothing_dim
        self.hidden_dims = list(hidden_dims)
        self.hyper_hidden_dim = hyper_hidden_dim
        self.trunk = nn.Sequential(
            nn.Linear(clothing_dim, hyper_hidden_dim),
            nn.SiLU(),
        )
        self.gamma_heads = nn.ModuleList(
            nn.Linear(hyper_hidden_dim, width) for width in hidden_dims
        )
        self.beta_heads = nn.ModuleList(
            nn.Linear(hyper_hidden_dim, width) for width in hidden_dims
        )
        for head in (*self.gamma_heads, *self.beta_heads):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, clothing_embedding: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        """Return one gamma and beta tensor for every modulated hidden layer."""

        if clothing_embedding.shape != (1, self.clothing_dim):
            raise ValueError(
                "clothing_embedding must have shape "
                f"(1, {self.clothing_dim}), got {tuple(clothing_embedding.shape)}"
            )
        features = self.trunk(clothing_embedding)
        return {
            "film_gamma": [head(features) for head in self.gamma_heads],
            "film_beta": [head(features) for head in self.beta_heads],
        }
