from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from scene.gaussian_clothing_residuals import GaussianClothingResiduals, interpolate_anchor_field


V7_DECODER_TYPE = "support_conditioned_dual_branch_v7"
V7_CAPACITY_GATE_MODE = "fixed_one_capacity_probe"
V7_EXTERNAL_GATE_MODE = "external"


@dataclass(frozen=True)
class V7DecoderOutput:
    raw_gaussian_residuals: GaussianClothingResiduals
    bounded_gaussian_residuals: GaussianClothingResiduals
    gated_gaussian_residuals: GaussianClothingResiduals
    geometry_gate: torch.Tensor
    appearance_gate: torch.Tensor
    gate_mode: str


class ResidualMLPBlock(nn.Module):
    """A per-row residual block whose normalization is chunk invariant."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.SiLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.fc2(self.activation(self.fc1(self.norm(value))))
        return value + hidden


def _flatten_base_attribute(value: torch.Tensor, gaussian_count: int, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.shape[0] != gaussian_count:
        raise ValueError(f"base {name} must have one row per Gaussian")
    if not torch.is_floating_point(value) or not torch.isfinite(value).all():
        raise ValueError(f"base {name} must be finite floating point")
    return value.detach().reshape(gaussian_count, -1)


def build_static_gaussian_support_descriptor(
    base_model: Any,
    canonical_anchors: torch.Tensor,
    gaussian_anchor_indices: torch.Tensor,
    gaussian_anchor_weights: torch.Tensor,
    *,
    fourier_frequencies: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build an outfit-independent descriptor from frozen canonical support only."""

    if not isinstance(fourier_frequencies, int) or isinstance(fourier_frequencies, bool):
        raise TypeError("fourier_frequencies must be an int")
    if fourier_frequencies < 0:
        raise ValueError("fourier_frequencies must be non-negative")
    xyz = _flatten_base_attribute(base_model._xyz, int(base_model._xyz.shape[0]), "xyz")
    if xyz.shape[1] != 3:
        raise ValueError("base xyz must have shape [N,3]")
    gaussian_count = xyz.shape[0]
    scaling = _flatten_base_attribute(base_model._scaling, gaussian_count, "log_scaling")
    opacity = _flatten_base_attribute(base_model._opacity, gaussian_count, "opacity_logit")
    sh0 = _flatten_base_attribute(base_model._sh0, gaussian_count, "sh0")
    if canonical_anchors.ndim != 2 or canonical_anchors.shape[1] != 3:
        raise ValueError("canonical_anchors must have shape [A,3]")
    if canonical_anchors.device != xyz.device or canonical_anchors.dtype != xyz.dtype:
        raise ValueError("canonical anchors must match base xyz device and dtype")
    if gaussian_anchor_indices.shape[0] != gaussian_count:
        raise ValueError("anchor assignment must have one row per Gaussian")
    if gaussian_anchor_indices.device != xyz.device or gaussian_anchor_weights.device != xyz.device:
        raise ValueError("anchor assignment tensors must share the base device")
    if gaussian_anchor_weights.dtype != xyz.dtype:
        raise ValueError("anchor weights must share the base dtype")

    center = xyz.mean(dim=0, keepdim=True)
    scale = (xyz - center).abs().amax().clamp_min(torch.finfo(xyz.dtype).eps)
    normalized_xyz = (xyz - center) / scale
    normalized_anchors = (canonical_anchors.detach() - center) / scale
    interpolated_anchor_xyz = interpolate_anchor_field(
        normalized_anchors, gaussian_anchor_indices, gaussian_anchor_weights,
    )
    anchor_delta = normalized_xyz - interpolated_anchor_xyz
    weight_max = gaussian_anchor_weights.amax(dim=1, keepdim=True)
    weight_entropy = -(
        gaussian_anchor_weights.clamp_min(torch.finfo(xyz.dtype).tiny)
        * gaussian_anchor_weights.clamp_min(torch.finfo(xyz.dtype).tiny).log()
    ).sum(dim=1, keepdim=True)
    weight_l2 = gaussian_anchor_weights.square().sum(dim=1, keepdim=True).sqrt()

    pieces = [
        normalized_xyz,
        scaling,
        opacity,
        sh0,
        weight_max,
        weight_entropy,
        weight_l2,
        interpolated_anchor_xyz,
        anchor_delta,
        anchor_delta.norm(dim=1, keepdim=True),
    ]
    names = [
        "normalized_canonical_xyz",
        "base_log_scaling",
        "base_opacity_logit",
        "base_sh0",
        "anchor_weight_max",
        "anchor_weight_entropy",
        "anchor_weight_l2",
        "interpolated_normalized_anchor_xyz",
        "gaussian_minus_anchor_xyz",
        "gaussian_anchor_distance",
    ]
    if fourier_frequencies:
        frequency = (2.0 ** torch.arange(
            fourier_frequencies, device=xyz.device, dtype=xyz.dtype,
        )).reshape(1, 1, -1)
        phase = torch.pi * normalized_xyz.unsqueeze(-1) * frequency
        pieces.extend((torch.sin(phase).flatten(1), torch.cos(phase).flatten(1)))
        names.extend(("canonical_xyz_fourier_sin", "canonical_xyz_fourier_cos"))
    descriptor = torch.cat(pieces, dim=1).detach().contiguous()
    if not torch.isfinite(descriptor).all():
        raise ValueError("static support descriptor contains NaN or Inf")
    return descriptor, {
        "outfit_independent": True,
        "target_signal_used": False,
        "gaussian_count": gaussian_count,
        "dimension": int(descriptor.shape[1]),
        "fourier_frequencies": fourier_frequencies,
        "fields": names,
        "normalization_center": center.detach().cpu().reshape(-1).tolist(),
        "normalization_scale": float(scale.detach().cpu()),
    }


class SupportConditionedDualBranchResidualDecoderV7(nn.Module):
    """Direct per-Gaussian six-channel decoder with separate residual trunks."""

    decoder_type = V7_DECODER_TYPE

    def __init__(
        self,
        *,
        static_support_descriptor: torch.Tensor,
        gaussian_anchor_indices: torch.Tensor,
        gaussian_anchor_weights: torch.Tensor,
        global_feature_dim: int,
        local_feature_dim: int,
        hidden_dim: int,
        num_blocks: int,
        channel_bounds: Mapping[str, float],
        output_shapes: Mapping[str, tuple[int, ...]],
        default_chunk_size: int = 16384,
        gate_mode: str = V7_CAPACITY_GATE_MODE,
        descriptor_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        dimensions = {
            "global_feature_dim": global_feature_dim,
            "local_feature_dim": local_feature_dim,
            "hidden_dim": hidden_dim,
            "num_blocks": num_blocks,
            "default_chunk_size": default_chunk_size,
        }
        for name, value in dimensions.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if num_blocks < 3 or num_blocks > 4:
            raise ValueError("V7 requires three or four residual blocks per branch")
        if static_support_descriptor.ndim != 2 or static_support_descriptor.shape[1] <= 0:
            raise ValueError("static_support_descriptor must have shape [N,D]")
        if not torch.is_floating_point(static_support_descriptor) or not torch.isfinite(static_support_descriptor).all():
            raise ValueError("static support descriptor must be finite floating point")
        if gaussian_anchor_indices.ndim != 2 or gaussian_anchor_weights.shape != gaussian_anchor_indices.shape:
            raise ValueError("Gaussian anchor assignment must have matching [N,K] shapes")
        if gaussian_anchor_indices.shape[0] != static_support_descriptor.shape[0]:
            raise ValueError("support descriptor and anchor assignment Gaussian counts differ")
        if gaussian_anchor_indices.dtype != torch.long:
            raise TypeError("gaussian_anchor_indices must use torch.long")
        if gate_mode not in {V7_CAPACITY_GATE_MODE, V7_EXTERNAL_GATE_MODE}:
            raise ValueError("unsupported V7 gate mode")

        required_bounds = {"xyz", "log_scaling", "rotation", "opacity_logit", "sh0", "shN"}
        if set(channel_bounds) != required_bounds:
            raise ValueError("V7 bounds must contain the exact legacy six-channel keys")
        self.channel_bounds = {name: float(value) for name, value in channel_bounds.items()}
        if any(not torch.isfinite(torch.tensor(value)) or value <= 0 for value in self.channel_bounds.values()):
            raise ValueError("V7 channel bounds must be positive and finite")

        expected_shape_keys = {
            "delta_xyz", "delta_log_scaling", "delta_rotvec",
            "delta_opacity_logit", "delta_sh0", "delta_shN",
        }
        if set(output_shapes) != expected_shape_keys:
            raise ValueError("output_shapes must contain the exact six residual channels")
        gaussian_count = int(static_support_descriptor.shape[0])
        self.output_shapes = {}
        for name, shape in output_shapes.items():
            shape = tuple(int(value) for value in shape)
            if not shape or shape[0] != gaussian_count:
                raise ValueError(f"{name} output shape must begin with the Gaussian count")
            self.output_shapes[name] = shape

        self.global_feature_dim = global_feature_dim
        self.local_feature_dim = local_feature_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.default_chunk_size = default_chunk_size
        self.gate_mode = gate_mode
        self.descriptor_metadata = dict(descriptor_metadata or {})
        self.register_buffer("static_support_descriptor", static_support_descriptor.detach().clone(), persistent=False)
        self.register_buffer("gaussian_anchor_indices", gaussian_anchor_indices.detach().clone(), persistent=False)
        self.register_buffer("gaussian_anchor_weights", gaussian_anchor_weights.detach().clone(), persistent=False)

        descriptor_dim = int(static_support_descriptor.shape[1])
        condition_dim = local_feature_dim + global_feature_dim
        self.support_encoder = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            ResidualMLPBlock(hidden_dim),
        )
        fused_dim = hidden_dim + condition_dim
        self.geometry_input = nn.Sequential(nn.Linear(fused_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.appearance_input = nn.Sequential(nn.Linear(fused_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.geometry_blocks = nn.ModuleList(ResidualMLPBlock(hidden_dim) for _ in range(num_blocks))
        self.appearance_blocks = nn.ModuleList(ResidualMLPBlock(hidden_dim) for _ in range(num_blocks))
        self.geometry_condition_reinject = nn.Linear(condition_dim, hidden_dim)
        self.appearance_condition_reinject = nn.Linear(condition_dim, hidden_dim)

        flat_dims = {name: int(torch.tensor(shape[1:]).prod().item()) for name, shape in self.output_shapes.items()}
        self.xyz_head = self._zero_head(flat_dims["delta_xyz"])
        self.scaling_head = self._zero_head(flat_dims["delta_log_scaling"])
        self.rotation_head = self._zero_head(flat_dims["delta_rotvec"])
        self.opacity_head = self._zero_head(flat_dims["delta_opacity_logit"])
        self.sh0_head = self._zero_head(flat_dims["delta_sh0"])
        self.shN_head = self._zero_head(flat_dims["delta_shN"])

    @classmethod
    def from_frozen_support(
        cls,
        base_model: Any,
        canonical_anchors: torch.Tensor,
        gaussian_anchor_indices: torch.Tensor,
        gaussian_anchor_weights: torch.Tensor,
        config: Mapping[str, Any],
        channel_bounds: Mapping[str, float],
    ) -> "SupportConditionedDualBranchResidualDecoderV7":
        descriptor, metadata = build_static_gaussian_support_descriptor(
            base_model,
            canonical_anchors,
            gaussian_anchor_indices,
            gaussian_anchor_weights,
            fourier_frequencies=int(config.get("support_fourier_frequencies", 8)),
        )
        shapes = {
            "delta_xyz": tuple(base_model._xyz.shape),
            "delta_log_scaling": tuple(base_model._scaling.shape),
            "delta_rotvec": (int(base_model._rotation.shape[0]), 3),
            "delta_opacity_logit": tuple(base_model._opacity.shape),
            "delta_sh0": tuple(base_model._sh0.shape),
            "delta_shN": tuple(base_model._shN.shape),
        }
        return cls(
            static_support_descriptor=descriptor,
            gaussian_anchor_indices=gaussian_anchor_indices,
            gaussian_anchor_weights=gaussian_anchor_weights,
            global_feature_dim=int(config["global_feature_dim"]),
            local_feature_dim=int(config["local_feature_dim"]),
            hidden_dim=int(config.get("hidden_dim", 128)),
            num_blocks=int(config.get("num_blocks", 4)),
            channel_bounds=channel_bounds,
            output_shapes=shapes,
            default_chunk_size=int(config.get("chunk_size", 16384)),
            gate_mode=str(config.get("gate_mode", V7_CAPACITY_GATE_MODE)),
            descriptor_metadata=metadata,
        ).to(device=base_model._xyz.device, dtype=base_model._xyz.dtype)

    def _zero_head(self, output_dim: int) -> nn.Linear:
        head = nn.Linear(self.hidden_dim, output_dim)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        return head

    @property
    def gaussian_count(self) -> int:
        return int(self.static_support_descriptor.shape[0])

    def forward(
        self,
        global_clothing_feature: torch.Tensor,
        completed_local_clothing_feature: torch.Tensor,
        *,
        geometry_gate: torch.Tensor | None = None,
        appearance_gate: torch.Tensor | None = None,
        gaussian_indices: torch.Tensor | None = None,
        chunk_size: int | None = None,
        gate_mode: str | None = None,
    ) -> V7DecoderOutput:
        if global_clothing_feature.ndim == 1:
            global_clothing_feature = global_clothing_feature.unsqueeze(0)
        if tuple(global_clothing_feature.shape) != (1, self.global_feature_dim):
            raise ValueError(f"global clothing feature must have shape [1,{self.global_feature_dim}]")
        if completed_local_clothing_feature.ndim != 2 or completed_local_clothing_feature.shape[1] != self.local_feature_dim:
            raise ValueError(f"completed local clothing feature must have shape [A,{self.local_feature_dim}]")
        reference = self.static_support_descriptor
        for name, value in (
            ("global clothing feature", global_clothing_feature),
            ("completed local clothing feature", completed_local_clothing_feature),
        ):
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError(f"{name} must match V7 device and dtype")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or Inf")
        if self.gaussian_anchor_indices.numel() and self.gaussian_anchor_indices.max() >= completed_local_clothing_feature.shape[0]:
            raise IndexError("local feature anchor count is smaller than the interpolation topology")

        if gaussian_indices is None:
            selected = torch.arange(self.gaussian_count, device=reference.device)
        else:
            if gaussian_indices.dtype != torch.long or gaussian_indices.ndim != 1:
                raise ValueError("gaussian_indices must be a one-dimensional torch.long tensor")
            if gaussian_indices.device != reference.device:
                raise ValueError("gaussian_indices must share the V7 device")
            if gaussian_indices.numel() and (gaussian_indices.min() < 0 or gaussian_indices.max() >= self.gaussian_count):
                raise IndexError("gaussian_indices are out of range")
            selected = gaussian_indices
        selected_count = int(selected.numel())
        active_gate_mode = self.gate_mode if gate_mode is None else gate_mode
        resolved_geometry_gate, resolved_appearance_gate = self._resolve_gates(
            selected, geometry_gate, appearance_gate, active_gate_mode,
        )
        actual_chunk = self.default_chunk_size if chunk_size is None else int(chunk_size)
        if actual_chunk <= 0:
            actual_chunk = max(selected_count, 1)

        raw_parts: dict[str, list[torch.Tensor]] = {name: [] for name in self.output_shapes}
        bounded_parts: dict[str, list[torch.Tensor]] = {name: [] for name in self.output_shapes}
        gated_parts: dict[str, list[torch.Tensor]] = {name: [] for name in self.output_shapes}
        for start in range(0, selected_count, actual_chunk):
            stop = min(start + actual_chunk, selected_count)
            index = selected[start:stop]
            raw, bounded = self._forward_chunk(
                index, global_clothing_feature, completed_local_clothing_feature,
            )
            geometry = resolved_geometry_gate[start:stop]
            appearance = resolved_appearance_gate[start:stop]
            for name, value in raw.items():
                raw_parts[name].append(value)
                bounded_parts[name].append(bounded[name])
                gate = geometry if name in {"delta_xyz", "delta_log_scaling", "delta_rotvec"} else appearance
                while gate.ndim < bounded[name].ndim:
                    gate = gate.unsqueeze(-1)
                gated_parts[name].append(bounded[name] * gate)

        empty_reference = reference.new_empty((0,))
        raw_values = {
            name: self._reshape_output(name, torch.cat(parts, dim=0) if parts else empty_reference)
            for name, parts in raw_parts.items()
        }
        bounded_values = {
            name: self._reshape_output(name, torch.cat(parts, dim=0) if parts else empty_reference)
            for name, parts in bounded_parts.items()
        }
        gated_values = {
            name: self._reshape_output(name, torch.cat(parts, dim=0) if parts else empty_reference)
            for name, parts in gated_parts.items()
        }
        return V7DecoderOutput(
            raw_gaussian_residuals=GaussianClothingResiduals(**raw_values),
            bounded_gaussian_residuals=GaussianClothingResiduals(**bounded_values),
            gated_gaussian_residuals=GaussianClothingResiduals(**gated_values),
            geometry_gate=resolved_geometry_gate,
            appearance_gate=resolved_appearance_gate,
            gate_mode=active_gate_mode,
        )

    def _forward_chunk(
        self,
        gaussian_indices: torch.Tensor,
        global_feature: torch.Tensor,
        local_anchor_feature: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        descriptor = self.static_support_descriptor.index_select(0, gaussian_indices)
        anchor_indices = self.gaussian_anchor_indices.index_select(0, gaussian_indices)
        anchor_weights = self.gaussian_anchor_weights.index_select(0, gaussian_indices)
        local = interpolate_anchor_field(local_anchor_feature, anchor_indices, anchor_weights)
        global_rows = global_feature.expand(descriptor.shape[0], -1)
        condition = torch.cat((local, global_rows), dim=1)
        support = self.support_encoder(descriptor)
        fused = torch.cat((support, condition), dim=1)
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
        bounded["delta_rotvec"] = raw["delta_rotvec"] * (
            self.channel_bounds["rotation"] * rotation_scale
        )
        return raw, bounded

    def _resolve_gates(
        self,
        selected: torch.Tensor,
        geometry_gate: torch.Tensor | None,
        appearance_gate: torch.Tensor | None,
        gate_mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reference = self.static_support_descriptor
        if gate_mode == V7_CAPACITY_GATE_MODE:
            if geometry_gate is not None or appearance_gate is not None:
                raise ValueError("capacity probe gate is fixed one and rejects external gates")
            ones = torch.ones(selected.shape[0], 1, device=reference.device, dtype=reference.dtype)
            return ones, ones.clone()
        if gate_mode != V7_EXTERNAL_GATE_MODE:
            raise ValueError("unsupported V7 gate mode")
        result = []
        for name, gate in (("geometry", geometry_gate), ("appearance", appearance_gate)):
            if gate is None:
                raise ValueError(f"external V7 gate mode requires a {name} gate")
            if gate.ndim == 1:
                gate = gate.unsqueeze(1)
            if gate.shape == (self.gaussian_count, 1):
                gate = gate.index_select(0, selected)
            elif gate.shape != (selected.shape[0], 1):
                raise ValueError(f"{name} gate must have shape [N,1]")
            if gate.device != reference.device or gate.dtype != reference.dtype:
                raise ValueError(f"{name} gate must match V7 device and dtype")
            if not torch.isfinite(gate).all() or torch.any((gate < 0) | (gate > 1)):
                raise ValueError(f"{name} gate must be finite in [0,1]")
            result.append(gate)
        return result[0], result[1]

    def _reshape_output(self, name: str, value: torch.Tensor) -> torch.Tensor:
        suffix = self.output_shapes[name][1:]
        if value.numel() == 0:
            return self.static_support_descriptor.new_empty((0, *suffix))
        return value.reshape(value.shape[0], *suffix)
