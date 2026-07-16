from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import nn

from scene.gaussian_clothing_residuals import (
    AnchorClothingResiduals,
    CanonicalGaussianOverrides,
    GaussianClothingResiduals,
    compose_canonical_gaussian_overrides,
    interpolate_anchor_clothing_residuals,
    interpolate_anchor_field,
)


ORACLE_TYPE = "representation_capacity_upper_bound"
DEFAULT_BOUNDS = {
    "xyz": 0.05,
    "log_scaling": 0.35,
    "rotation": 0.2617993878,
    "opacity_logit": 2.0,
    "sh0": 0.25,
    "shN": 0.10,
}
STAGE_TRAINABLE = {
    0: frozenset(),
    1: frozenset({"raw_opacity", "raw_sh0", "appearance_gate_logits"}),
    2: frozenset({
        "raw_xyz", "raw_log_scaling", "raw_rotvec", "raw_opacity", "raw_sh0",
        "geometry_gate_logits", "appearance_gate_logits",
    }),
    3: frozenset({
        "raw_xyz", "raw_log_scaling", "raw_rotvec", "raw_opacity", "raw_sh0",
        "geometry_gate_logits", "appearance_gate_logits",
    }),
    4: frozenset({
        "raw_xyz", "raw_log_scaling", "raw_rotvec", "raw_opacity", "raw_sh0", "raw_shN",
        "geometry_gate_logits", "appearance_gate_logits",
    }),
}


@dataclass(frozen=True)
class OracleRegularizationOutput:
    residual_magnitude: dict[str, torch.Tensor]
    geometry_gate_sparsity: torch.Tensor
    appearance_gate_sparsity: torch.Tensor
    gate_binary: torch.Tensor
    graph_gate_smoothness: torch.Tensor
    graph_residual_smoothness: torch.Tensor


@dataclass(frozen=True)
class OracleForwardOutput:
    raw_residuals: AnchorClothingResiduals | GaussianClothingResiduals
    gated_residuals: AnchorClothingResiduals | GaussianClothingResiduals
    gaussian_residuals: GaussianClothingResiduals
    canonical_overrides: CanonicalGaussianOverrides
    geometry_gate: torch.Tensor
    appearance_gate: torch.Tensor
    gaussian_geometry_gate: torch.Tensor
    gaussian_appearance_gate: torch.Tensor
    regularization: OracleRegularizationOutput
    oracle_type: str = ORACLE_TYPE


def _positive_bound_config(bounds: Mapping[str, float] | None) -> dict[str, float]:
    result = dict(DEFAULT_BOUNDS)
    if bounds:
        unknown = set(bounds).difference(result)
        if unknown:
            raise ValueError(f"unknown oracle residual bounds: {sorted(unknown)}")
        result.update({key: float(value) for key, value in bounds.items()})
    for key, value in result.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"bound {key} must be positive and finite")
    return result


def _bounded_rotvec(raw: torch.Tensor, max_angle: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
    scale = torch.where(
        norm > 1e-6,
        torch.tanh(norm) / norm.clamp_min(1e-6),
        torch.ones_like(norm),
    )
    return raw * (max_angle * scale)


def _binary_loss(gate: torch.Tensor) -> torch.Tensor:
    return (gate * (1.0 - gate)).mean()


def _edge_smoothness(value: torch.Tensor, indices: torch.Tensor | None) -> torch.Tensor:
    if indices is None or indices.numel() == 0:
        return value.sum() * 0
    if indices.ndim != 2 or indices.shape[1] != 2 or indices.dtype != torch.long:
        raise ValueError("graph_edges must be a long tensor with shape [E,2]")
    if indices.min() < 0 or indices.max() >= value.shape[0]:
        raise IndexError("graph edge index is out of bounds")
    return (value[indices[:, 0]] - value[indices[:, 1]]).abs().mean()


class _ResidualOracleBase(nn.Module):
    oracle_type = ORACLE_TYPE

    def __init__(
        self,
        element_count: int,
        *,
        bounds: Mapping[str, float] | None = None,
        initial_gate_probability: float = 0.05,
        enable_shn: bool = False,
        graph_edges: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if element_count <= 0:
            raise ValueError("element_count must be positive")
        if not 0 < initial_gate_probability < 1:
            raise ValueError("initial_gate_probability must be in (0,1)")
        self.element_count = int(element_count)
        self.bounds = _positive_bound_config(bounds)
        self.enable_shn = bool(enable_shn)
        self.effective_sh_degree = 1 if self.enable_shn else 0
        gate_logit = math.log(initial_gate_probability / (1 - initial_gate_probability))
        self.geometry_gate_logits = nn.Parameter(torch.full((element_count, 1), gate_logit))
        self.appearance_gate_logits = nn.Parameter(torch.full((element_count, 1), gate_logit))
        self.register_buffer(
            "graph_edges",
            torch.empty((0, 2), dtype=torch.long) if graph_edges is None else graph_edges.long(),
            persistent=True,
        )
        self.stage = 0

    def configure_stage(self, stage: int) -> None:
        if stage not in STAGE_TRAINABLE:
            raise ValueError("oracle stage must be in [0,4]")
        if stage == 4 and not self.enable_shn:
            raise ValueError("stage 4 requires enable_shn=True")
        enabled = STAGE_TRAINABLE[stage]
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name in enabled)
        self.stage = stage

    def parameter_groups(self, learning_rates: Mapping[str, float]) -> tuple[list[dict[str, Any]], list[str]]:
        groups = {
            "geometry_residuals": ["raw_xyz", "raw_log_scaling", "raw_rotvec"],
            "appearance_residuals": ["raw_opacity", "raw_sh0", "raw_shN"],
            "geometry_gate": ["geometry_gate_logits"],
            "appearance_gate": ["appearance_gate_logits"],
        }
        named = dict(self.named_parameters())
        output, names = [], []
        for group_name, parameter_names in groups.items():
            parameters = [named[name] for name in parameter_names if name in named]
            output.append({"params": parameters, "lr": float(learning_rates[group_name])})
            names.append(group_name)
        return output, names

    def _bounded_values(self) -> dict[str, torch.Tensor]:
        values = {
            "delta_xyz": self.bounds["xyz"] * torch.tanh(self.raw_xyz),
            "delta_log_scaling": self.bounds["log_scaling"] * torch.tanh(self.raw_log_scaling),
            "delta_rotvec": _bounded_rotvec(self.raw_rotvec, self.bounds["rotation"]),
            "delta_opacity_logit": self.bounds["opacity_logit"] * torch.tanh(self.raw_opacity),
            "delta_sh0": self.bounds["sh0"] * torch.tanh(self.raw_sh0),
            "delta_shN": self.bounds["shN"] * torch.tanh(self.raw_shN),
        }
        if not self.enable_shn:
            values["delta_shN"] = torch.zeros_like(values["delta_shN"])
        return values

    def _regularization(
        self,
        residuals: AnchorClothingResiduals | GaussianClothingResiduals,
        geometry_gate: torch.Tensor,
        appearance_gate: torch.Tensor,
    ) -> OracleRegularizationOutput:
        magnitudes = {
            name: getattr(residuals, name).abs().mean()
            for name in (
                "delta_xyz", "delta_log_scaling", "delta_rotvec",
                "delta_opacity_logit", "delta_sh0", "delta_shN",
            )
        }
        opacity_gate = torch.maximum(geometry_gate, appearance_gate)
        geometry_fields = torch.cat((
            residuals.delta_xyz * geometry_gate,
            residuals.delta_log_scaling * geometry_gate,
            residuals.delta_rotvec * geometry_gate,
        ), dim=1)
        appearance_fields = torch.cat((
            residuals.delta_opacity_logit.reshape(self.element_count, -1) * opacity_gate,
            residuals.delta_sh0.reshape(self.element_count, -1) * appearance_gate,
            residuals.delta_shN.reshape(self.element_count, -1) * appearance_gate,
        ), dim=1)
        edges = self.graph_edges if self.graph_edges.numel() else None
        return OracleRegularizationOutput(
            residual_magnitude=magnitudes,
            geometry_gate_sparsity=geometry_gate.mean(),
            appearance_gate_sparsity=appearance_gate.mean(),
            gate_binary=0.5 * (_binary_loss(geometry_gate) + _binary_loss(appearance_gate)),
            graph_gate_smoothness=0.5 * (
                _edge_smoothness(geometry_gate, edges) + _edge_smoothness(appearance_gate, edges)
            ),
            graph_residual_smoothness=0.5 * (
                _edge_smoothness(geometry_fields, edges) + _edge_smoothness(appearance_fields, edges)
            ),
        )


class GaussianResidualOracle(_ResidualOracleBase):
    def __init__(
        self,
        base_model: Any,
        *,
        bounds: Mapping[str, float] | None = None,
        initial_gate_probability: float = 0.05,
        enable_shn: bool = False,
        graph_edges: torch.Tensor | None = None,
    ) -> None:
        count = int(base_model._xyz.shape[0])
        super().__init__(
            count, bounds=bounds, initial_gate_probability=initial_gate_probability,
            enable_shn=enable_shn, graph_edges=graph_edges,
        )
        reference = base_model._xyz
        self.raw_xyz = nn.Parameter(torch.zeros_like(base_model._xyz))
        self.raw_log_scaling = nn.Parameter(torch.zeros_like(base_model._scaling))
        self.raw_rotvec = nn.Parameter(torch.zeros(count, 3, device=reference.device, dtype=reference.dtype))
        self.raw_opacity = nn.Parameter(torch.zeros_like(base_model._opacity))
        self.raw_sh0 = nn.Parameter(torch.zeros_like(base_model._sh0))
        self.raw_shN = nn.Parameter(torch.zeros_like(base_model._shN))
        self.configure_stage(0)

    def forward(self, base_model: Any) -> OracleForwardOutput:
        values = self._bounded_values()
        raw = GaussianClothingResiduals(**values).validate(base_model)
        geometry_gate = torch.sigmoid(self.geometry_gate_logits)
        appearance_gate = torch.sigmoid(self.appearance_gate_logits)
        opacity_gate = torch.maximum(geometry_gate, appearance_gate)
        gated = GaussianClothingResiduals(
            delta_xyz=raw.delta_xyz * geometry_gate,
            delta_log_scaling=raw.delta_log_scaling * geometry_gate,
            delta_rotvec=raw.delta_rotvec * geometry_gate,
            delta_opacity_logit=raw.delta_opacity_logit * opacity_gate,
            delta_sh0=raw.delta_sh0 * appearance_gate.reshape((-1,) + (1,) * (raw.delta_sh0.ndim - 1)),
            delta_shN=raw.delta_shN * appearance_gate.reshape((-1,) + (1,) * (raw.delta_shN.ndim - 1)),
        ).validate(base_model)
        return OracleForwardOutput(
            raw_residuals=raw,
            gated_residuals=gated,
            gaussian_residuals=gated,
            canonical_overrides=compose_canonical_gaussian_overrides(base_model, gated),
            geometry_gate=geometry_gate,
            appearance_gate=appearance_gate,
            gaussian_geometry_gate=geometry_gate,
            gaussian_appearance_gate=appearance_gate,
            regularization=self._regularization(raw, geometry_gate, appearance_gate),
        )


class AnchorResidualOracle(_ResidualOracleBase):
    def __init__(
        self,
        base_model: Any,
        anchor_count: int,
        gaussian_anchor_indices: torch.Tensor,
        gaussian_anchor_weights: torch.Tensor,
        *,
        bounds: Mapping[str, float] | None = None,
        initial_gate_probability: float = 0.05,
        enable_shn: bool = False,
        graph_edges: torch.Tensor | None = None,
    ) -> None:
        super().__init__(
            anchor_count, bounds=bounds, initial_gate_probability=initial_gate_probability,
            enable_shn=enable_shn, graph_edges=graph_edges,
        )
        reference = base_model._xyz
        self.raw_xyz = nn.Parameter(torch.zeros(anchor_count, 3, device=reference.device, dtype=reference.dtype))
        self.raw_log_scaling = nn.Parameter(torch.zeros(anchor_count, 3, device=reference.device, dtype=reference.dtype))
        self.raw_rotvec = nn.Parameter(torch.zeros(anchor_count, 3, device=reference.device, dtype=reference.dtype))
        self.raw_opacity = nn.Parameter(torch.zeros(anchor_count, 1, device=reference.device, dtype=reference.dtype))
        self.raw_sh0 = nn.Parameter(torch.zeros(anchor_count, int(base_model._sh0[0].numel()), device=reference.device, dtype=reference.dtype))
        self.raw_shN = nn.Parameter(torch.zeros(anchor_count, int(base_model._shN[0].numel()), device=reference.device, dtype=reference.dtype))
        self.register_buffer("gaussian_anchor_indices", gaussian_anchor_indices.long(), persistent=True)
        self.register_buffer("gaussian_anchor_weights", gaussian_anchor_weights.to(reference), persistent=True)
        self.configure_stage(0)

    def forward(self, base_model: Any) -> OracleForwardOutput:
        raw = AnchorClothingResiduals(**self._bounded_values()).validate(self.element_count, base_model)
        geometry_gate = torch.sigmoid(self.geometry_gate_logits)
        appearance_gate = torch.sigmoid(self.appearance_gate_logits)
        opacity_gate = torch.maximum(geometry_gate, appearance_gate)
        gated = AnchorClothingResiduals(
            delta_xyz=raw.delta_xyz * geometry_gate,
            delta_log_scaling=raw.delta_log_scaling * geometry_gate,
            delta_rotvec=raw.delta_rotvec * geometry_gate,
            delta_opacity_logit=raw.delta_opacity_logit * opacity_gate,
            delta_sh0=raw.delta_sh0 * appearance_gate,
            delta_shN=raw.delta_shN * appearance_gate,
        ).validate(self.element_count, base_model)
        gaussian = interpolate_anchor_clothing_residuals(
            gated, self.gaussian_anchor_indices, self.gaussian_anchor_weights, base_model,
        )
        gaussian_geometry = interpolate_anchor_field(
            geometry_gate, self.gaussian_anchor_indices, self.gaussian_anchor_weights,
        )
        gaussian_appearance = interpolate_anchor_field(
            appearance_gate, self.gaussian_anchor_indices, self.gaussian_anchor_weights,
        )
        return OracleForwardOutput(
            raw_residuals=raw,
            gated_residuals=gated,
            gaussian_residuals=gaussian,
            canonical_overrides=compose_canonical_gaussian_overrides(base_model, gaussian),
            geometry_gate=geometry_gate,
            appearance_gate=appearance_gate,
            gaussian_geometry_gate=gaussian_geometry,
            gaussian_appearance_gate=gaussian_appearance,
            regularization=self._regularization(raw, geometry_gate, appearance_gate),
        )
