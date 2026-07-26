from __future__ import annotations

import torch
from torch import nn


class ClothingEmbedding(nn.Module):
    """Map one clothing id to a single learned conditioning vector."""

    def __init__(self, num_clothes: int, embedding_dim: int = 64) -> None:
        super().__init__()
        if not isinstance(num_clothes, int) or isinstance(num_clothes, bool):
            raise TypeError(f"num_clothes must be an int, got {type(num_clothes)!r}")
        if num_clothes <= 0:
            raise ValueError(f"num_clothes must be positive, got {num_clothes}")
        if not isinstance(embedding_dim, int) or isinstance(embedding_dim, bool):
            raise TypeError(f"embedding_dim must be an int, got {type(embedding_dim)!r}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")

        self.num_clothes = num_clothes
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_clothes, embedding_dim)

    def forward(self, cloth_id: torch.Tensor | int) -> torch.Tensor:
        """Return a ``[1, embedding_dim]`` embedding for one clothing id."""

        index = self._normalize_cloth_id(cloth_id)
        return self.embedding(index)

    def _normalize_cloth_id(self, cloth_id: torch.Tensor | int) -> torch.Tensor:
        if isinstance(cloth_id, bool):
            raise TypeError("cloth_id must be a Python int or scalar integer tensor, not bool")

        if isinstance(cloth_id, int):
            resolved = cloth_id
        elif isinstance(cloth_id, torch.Tensor):
            if cloth_id.ndim != 0:
                raise NotImplementedError(
                    "batched cloth_id tensors are not supported; pass one scalar clothing id"
                )
            if cloth_id.dtype == torch.bool or torch.is_floating_point(cloth_id):
                raise TypeError("scalar cloth_id tensor must have an integer dtype")
            resolved = int(cloth_id.item())
        else:
            raise TypeError(
                f"cloth_id must be a Python int or scalar torch.Tensor, got {type(cloth_id)!r}"
            )

        if not 0 <= resolved < self.num_clothes:
            raise IndexError(
                f"cloth_id {resolved} is out of range for {self.num_clothes} clothes"
            )
        return torch.tensor(
            [resolved],
            dtype=torch.long,
            device=self.embedding.weight.device,
        )
