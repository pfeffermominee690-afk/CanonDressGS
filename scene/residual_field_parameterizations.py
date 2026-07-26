from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from scene.gaussian_clothing_residuals import GaussianClothingResiduals, interpolate_anchor_field
from scene.support_conditioned_dual_branch_residual_decoder_v7 import ResidualMLPBlock


FIXED_ONE_GATE_MODE = "fixed_one_capacity_probe"
P1_FIELD_TYPE = "per_gaussian_support_token"
P2_FIELD_TYPE = "per_anchor_support_token"

CHANNEL_TO_BOUND = {
    "delta_xyz": "xyz",
    "delta_log_scaling": "log_scaling",
    "delta_rotvec": "rotation",
    "delta_opacity_logit": "opacity_logit",
    "delta_sh0": "sh0",
    "delta_shN": "shN",
}


@dataclass(frozen=True)
class ResidualFieldOutput:
    raw_gaussian_residuals: GaussianClothingResiduals
    bounded_gaussian_residuals: GaussianClothingResiduals
    gated_gaussian_residuals: GaussianClothingResiduals
    geometry_gate: torch.Tensor
    appearance_gate: torch.Tensor
    gate_mode: str


def residual_output_shapes(base_model: Any) -> dict[str, tuple[int, ...]]:
    return {
        "delta_xyz": tuple(base_model._xyz.shape),
        "delta_log_scaling": tuple(base_model._scaling.shape),
        "delta_rotvec": (int(base_model._rotation.shape[0]), 3),
        "delta_opacity_logit": tuple(base_model._opacity.shape),
        "delta_sh0": tuple(base_model._sh0.shape),
        "delta_shN": tuple(base_model._shN.shape),
    }


def _validate_shapes_and_bounds(
    output_shapes: Mapping[str, tuple[int, ...]],
    channel_bounds: Mapping[str, float],
) -> tuple[int, dict[str, tuple[int, ...]], dict[str, float]]:
    if set(output_shapes) != set(CHANNEL_TO_BOUND):
        raise ValueError("output_shapes must contain the exact six residual channels")
    if set(channel_bounds) != set(CHANNEL_TO_BOUND.values()):
        raise ValueError("channel_bounds must contain the exact six residual bounds")
    shapes = {name: tuple(int(value) for value in shape) for name, shape in output_shapes.items()}
    counts = {shape[0] for shape in shapes.values() if shape}
    if len(counts) != 1 or any(not shape or any(value <= 0 for value in shape) for shape in shapes.values()):
        raise ValueError("all residual shapes must be positive and share one Gaussian count")
    bounds = {name: float(value) for name, value in channel_bounds.items()}
    if any(not torch.isfinite(torch.tensor(value)) or value <= 0 for value in bounds.values()):
        raise ValueError("all residual bounds must be finite and positive")
    return counts.pop(), shapes, bounds


class DirectPerGaussianResidualTableControl(nn.Module):
    """P0: one directly optimized bound-normalized residual table for one garment."""

    def __init__(
        self,
        output_shapes: Mapping[str, tuple[int, ...]],
        channel_bounds: Mapping[str, float],
    ) -> None:
        super().__init__()
        gaussian_count, shapes, bounds = _validate_shapes_and_bounds(output_shapes, channel_bounds)
        self.output_shapes = shapes
        self.channel_bounds = bounds
        self.normalized_residual_tables = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(gaussian_count, self._flat_dim(shape)))
            for name, shape in shapes.items()
        })

    @staticmethod
    def _flat_dim(shape: tuple[int, ...]) -> int:
        return int(torch.tensor(shape[1:]).prod().item())

    @property
    def gaussian_count(self) -> int:
        return next(iter(self.output_shapes.values()))[0]

    def forward(self, gaussian_indices: torch.Tensor | None = None) -> GaussianClothingResiduals:
        reference = next(iter(self.normalized_residual_tables.values()))
        if gaussian_indices is None:
            selected = torch.arange(self.gaussian_count, device=reference.device)
        else:
            if gaussian_indices.ndim != 1 or gaussian_indices.dtype != torch.long:
                raise ValueError("gaussian_indices must be a one-dimensional torch.long tensor")
            if gaussian_indices.device != reference.device:
                raise ValueError("gaussian_indices must share the table device")
            if gaussian_indices.numel() and (
                gaussian_indices.min() < 0 or gaussian_indices.max() >= self.gaussian_count
            ):
                raise IndexError("gaussian_indices are out of range")
            selected = gaussian_indices
        values = {}
        for name, table in self.normalized_residual_tables.items():
            selected_table = table.index_select(0, selected)
            values[name] = (
                selected_table * self.channel_bounds[CHANNEL_TO_BOUND[name]]
            ).reshape(selected.shape[0], *self.output_shapes[name][1:])
        return GaussianClothingResiduals(**values)


class _SupportTokenResidualFieldBase(nn.Module):
    field_type: str

    def __init__(
        self,
        *,
        static_support_descriptor: torch.Tensor,
        gaussian_anchor_indices: torch.Tensor,
        gaussian_anchor_weights: torch.Tensor,
        garment_embedding_dim: int,
        token_dim: int,
        hidden_dim: int,
        num_blocks: int,
        output_shapes: Mapping[str, tuple[int, ...]],
        channel_bounds: Mapping[str, float],
        default_chunk_size: int,
        token_count: int,
    ) -> None:
        super().__init__()
        gaussian_count, shapes, bounds = _validate_shapes_and_bounds(output_shapes, channel_bounds)
        for name, value in {
            "garment_embedding_dim": garment_embedding_dim,
            "token_dim": token_dim,
            "hidden_dim": hidden_dim,
            "num_blocks": num_blocks,
            "default_chunk_size": default_chunk_size,
            "token_count": token_count,
        }.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if num_blocks not in {2, 3}:
            raise ValueError("support-token fields require two or three residual blocks")
        if static_support_descriptor.ndim != 2 or static_support_descriptor.shape[0] != gaussian_count:
            raise ValueError("static_support_descriptor must have shape [N,D]")
        if not torch.is_floating_point(static_support_descriptor) or not torch.isfinite(static_support_descriptor).all():
            raise ValueError("static support descriptor must be finite floating point")
        if gaussian_anchor_indices.ndim != 2 or gaussian_anchor_weights.shape != gaussian_anchor_indices.shape:
            raise ValueError("anchor interpolation must have matching [N,K] tensors")
        if gaussian_anchor_indices.shape[0] != gaussian_count or gaussian_anchor_indices.dtype != torch.long:
            raise ValueError("anchor interpolation must provide long indices for every Gaussian")
        if gaussian_anchor_indices.device != static_support_descriptor.device:
            raise ValueError("anchor indices and support descriptor must share a device")
        if gaussian_anchor_weights.device != static_support_descriptor.device or gaussian_anchor_weights.dtype != static_support_descriptor.dtype:
            raise ValueError("anchor weights and support descriptor must share device and dtype")
        if gaussian_anchor_indices.numel() and int(gaussian_anchor_indices.max()) >= token_count:
            raise ValueError("token_count is smaller than the frozen anchor assignment")

        self.output_shapes = shapes
        self.channel_bounds = bounds
        self.garment_embedding_dim = garment_embedding_dim
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.default_chunk_size = default_chunk_size
        self.register_buffer("static_support_descriptor", static_support_descriptor.detach().clone(), persistent=False)
        self.register_buffer("gaussian_anchor_indices", gaussian_anchor_indices.detach().clone(), persistent=False)
        self.register_buffer("gaussian_anchor_weights", gaussian_anchor_weights.detach().clone(), persistent=False)
        self.support_tokens = nn.Parameter(torch.empty(token_count, token_dim))
        nn.init.normal_(self.support_tokens, mean=0.0, std=0.02)

        self.garment_to_token = nn.Linear(garment_embedding_dim, token_dim, bias=False)
        fused_dim = token_dim + int(static_support_descriptor.shape[1]) + garment_embedding_dim + token_dim
        reinject_dim = garment_embedding_dim + token_dim
        self.geometry_input = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
        )
        self.appearance_input = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
        )
        self.geometry_blocks = nn.ModuleList(ResidualMLPBlock(hidden_dim) for _ in range(num_blocks))
        self.appearance_blocks = nn.ModuleList(ResidualMLPBlock(hidden_dim) for _ in range(num_blocks))
        self.geometry_condition_reinject = nn.Linear(reinject_dim, hidden_dim)
        self.appearance_condition_reinject = nn.Linear(reinject_dim, hidden_dim)

        flat_dims = {name: self._flat_dim(shape) for name, shape in shapes.items()}
        self.xyz_head = self._zero_head(flat_dims["delta_xyz"])
        self.scaling_head = self._zero_head(flat_dims["delta_log_scaling"])
        self.rotation_head = self._zero_head(flat_dims["delta_rotvec"])
        self.opacity_head = self._zero_head(flat_dims["delta_opacity_logit"])
        self.sh0_head = self._zero_head(flat_dims["delta_sh0"])
        self.shN_head = self._zero_head(flat_dims["delta_shN"])

    @staticmethod
    def _flat_dim(shape: tuple[int, ...]) -> int:
        return int(torch.tensor(shape[1:]).prod().item())

    def _zero_head(self, output_dim: int) -> nn.Linear:
        head = nn.Linear(self.hidden_dim, output_dim)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        return head

    @property
    def gaussian_count(self) -> int:
        return int(self.static_support_descriptor.shape[0])

    @property
    def token_count(self) -> int:
        return int(self.support_tokens.shape[0])

    def token_rows(self, gaussian_indices: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        garment_embedding: torch.Tensor,
        *,
        gaussian_indices: torch.Tensor | None = None,
        chunk_size: int | None = None,
    ) -> ResidualFieldOutput:
        if garment_embedding.ndim == 1:
            garment_embedding = garment_embedding.unsqueeze(0)
        if tuple(garment_embedding.shape) != (1, self.garment_embedding_dim):
            raise ValueError(f"garment_embedding must have shape [1,{self.garment_embedding_dim}]")
        if garment_embedding.device != self.static_support_descriptor.device or garment_embedding.dtype != self.static_support_descriptor.dtype:
            raise ValueError("garment_embedding must share support device and dtype")
        if not torch.isfinite(garment_embedding).all():
            raise ValueError("garment_embedding contains NaN or Inf")
        if gaussian_indices is None:
            selected = torch.arange(self.gaussian_count, device=self.static_support_descriptor.device)
        else:
            if gaussian_indices.ndim != 1 or gaussian_indices.dtype != torch.long:
                raise ValueError("gaussian_indices must be a one-dimensional torch.long tensor")
            if gaussian_indices.device != self.static_support_descriptor.device:
                raise ValueError("gaussian_indices must share the support device")
            if gaussian_indices.numel() and (
                gaussian_indices.min() < 0 or gaussian_indices.max() >= self.gaussian_count
            ):
                raise IndexError("gaussian_indices are out of range")
            selected = gaussian_indices
        actual_chunk = self.default_chunk_size if chunk_size is None else int(chunk_size)
        if actual_chunk <= 0:
            actual_chunk = max(1, int(selected.numel()))

        raw_parts: dict[str, list[torch.Tensor]] = {name: [] for name in self.output_shapes}
        bounded_parts: dict[str, list[torch.Tensor]] = {name: [] for name in self.output_shapes}
        for start in range(0, int(selected.numel()), actual_chunk):
            index = selected[start:start + actual_chunk]
            raw, bounded = self._forward_chunk(index, garment_embedding)
            for name in raw:
                raw_parts[name].append(raw[name])
                bounded_parts[name].append(bounded[name])
        raw_values = self._assemble(raw_parts, selected.shape[0])
        bounded_values = self._assemble(bounded_parts, selected.shape[0])
        residual_raw = GaussianClothingResiduals(**raw_values)
        residual_bounded = GaussianClothingResiduals(**bounded_values)
        ones = torch.ones(selected.shape[0], 1, device=garment_embedding.device, dtype=garment_embedding.dtype)
        return ResidualFieldOutput(
            raw_gaussian_residuals=residual_raw,
            bounded_gaussian_residuals=residual_bounded,
            gated_gaussian_residuals=residual_bounded,
            geometry_gate=ones,
            appearance_gate=ones.clone(),
            gate_mode=FIXED_ONE_GATE_MODE,
        )

    def _forward_chunk(
        self,
        gaussian_indices: torch.Tensor,
        garment_embedding: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        token = self.token_rows(gaussian_indices)
        descriptor = self.static_support_descriptor.index_select(0, gaussian_indices)
        garment_rows = garment_embedding.expand(gaussian_indices.shape[0], -1)
        projected_garment = self.garment_to_token(garment_embedding).expand_as(token)
        multiplicative_interaction = token * projected_garment
        fused = torch.cat((token, descriptor, garment_rows, multiplicative_interaction), dim=1)
        condition = torch.cat((garment_rows, multiplicative_interaction), dim=1)
        geometry = self.geometry_input(fused)
        appearance = self.appearance_input(fused)
        reinject_at = max(1, self.num_blocks // 2)
        for index, block in enumerate(self.geometry_blocks):
            if index == reinject_at:
                geometry = geometry + self.geometry_condition_reinject(condition)
            geometry = block(geometry)
        for index, block in enumerate(self.appearance_blocks):
            if index == reinject_at:
                appearance = appearance + self.appearance_condition_reinject(condition)
            appearance = block(appearance)
        raw = {
            "delta_xyz": self.xyz_head(geometry),
            "delta_log_scaling": self.scaling_head(geometry),
            "delta_rotvec": self.rotation_head(geometry),
            "delta_opacity_logit": self.opacity_head(appearance),
            "delta_sh0": self.sh0_head(appearance),
            "delta_shN": self.shN_head(appearance),
        }
        bounded = {
            "delta_xyz": self.channel_bounds["xyz"] * torch.tanh(raw["delta_xyz"]),
            "delta_log_scaling": self.channel_bounds["log_scaling"] * torch.tanh(raw["delta_log_scaling"]),
            "delta_opacity_logit": self.channel_bounds["opacity_logit"] * torch.tanh(raw["delta_opacity_logit"]),
            "delta_sh0": self.channel_bounds["sh0"] * torch.tanh(raw["delta_sh0"]),
            "delta_shN": self.channel_bounds["shN"] * torch.tanh(raw["delta_shN"]),
        }
        rotation_norm = torch.linalg.vector_norm(raw["delta_rotvec"], dim=-1, keepdim=True)
        rotation_scale = torch.where(
            rotation_norm > 1e-6,
            torch.tanh(rotation_norm) / rotation_norm.clamp_min(1e-6),
            torch.ones_like(rotation_norm),
        )
        bounded["delta_rotvec"] = raw["delta_rotvec"] * self.channel_bounds["rotation"] * rotation_scale
        return raw, bounded

    def _assemble(
        self,
        parts: Mapping[str, list[torch.Tensor]],
        selected_count: int,
    ) -> dict[str, torch.Tensor]:
        values = {}
        for name, tensors in parts.items():
            flat_dim = self._flat_dim(self.output_shapes[name])
            flat = torch.cat(tensors, dim=0) if tensors else self.static_support_descriptor.new_empty((0, flat_dim))
            values[name] = flat.reshape(selected_count, *self.output_shapes[name][1:])
        return values


class GaussianSupportTokenResidualField(_SupportTokenResidualFieldBase):
    """P1: one outfit-independent learnable token per canonical Gaussian."""

    field_type = P1_FIELD_TYPE

    def __init__(self, **kwargs: Any) -> None:
        descriptor = kwargs["static_support_descriptor"]
        super().__init__(token_count=int(descriptor.shape[0]), **kwargs)

    def token_rows(self, gaussian_indices: torch.Tensor) -> torch.Tensor:
        return self.support_tokens.index_select(0, gaussian_indices)


class AnchorSupportTokenResidualField(_SupportTokenResidualFieldBase):
    """P2: frozen anchor interpolation of one learnable token per canonical anchor."""

    field_type = P2_FIELD_TYPE

    def __init__(self, *, anchor_count: int, **kwargs: Any) -> None:
        super().__init__(token_count=int(anchor_count), **kwargs)

    def token_rows(self, gaussian_indices: torch.Tensor) -> torch.Tensor:
        indices = self.gaussian_anchor_indices.index_select(0, gaussian_indices)
        weights = self.gaussian_anchor_weights.index_select(0, gaussian_indices)
        return interpolate_anchor_field(self.support_tokens, indices, weights)
