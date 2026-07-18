from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn

from scene.full_attribute_oracle import OracleForwardOutput, OracleRegularizationOutput
from scene.gaussian_clothing_residuals import GaussianClothingResiduals, compose_canonical_gaussian_overrides


FIXED_OPEN_ORACLE_TYPE = "o00_fixed_open_gaussian_representation_capacity"
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
    1: frozenset({"raw_xyz", "raw_log_scaling", "raw_rotvec", "raw_opacity"}),
    2: frozenset({"raw_opacity", "raw_sh0"}),
    3: frozenset({"raw_xyz", "raw_log_scaling", "raw_rotvec", "raw_opacity", "raw_sh0"}),
}


def fixed_open_stage_for_step(step: int) -> int:
    if not 0 <= step <= 480:
        raise ValueError("O00 fixed-open step must be in [0,480]")
    if step == 0:
        return 0
    if step <= 80:
        return 1
    if step <= 160:
        return 2
    return 3


def _positive_bounds(bounds: Mapping[str, float] | None) -> dict[str, float]:
    result = dict(DEFAULT_BOUNDS)
    if bounds:
        unknown = set(bounds).difference(result)
        if unknown:
            raise ValueError(f"unknown fixed-open residual bounds: {sorted(unknown)}")
        result.update({name: float(value) for name, value in bounds.items()})
    if any(not math.isfinite(value) or value <= 0 for value in result.values()):
        raise ValueError("fixed-open residual bounds must be positive and finite")
    return result


def _bounded_rotvec(raw: torch.Tensor, maximum: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
    scale = torch.where(norm > 1e-6, torch.tanh(norm) / norm.clamp_min(1e-6), torch.ones_like(norm))
    return raw * (maximum * scale)


class FixedOpenGaussianOracle(nn.Module):
    """One shared O00 Gaussian residual with fixed-open gates and frozen SHN."""

    oracle_type = FIXED_OPEN_ORACLE_TYPE

    def __init__(self, base_model: Any, *, bounds: Mapping[str, float] | None = None) -> None:
        super().__init__()
        count = int(base_model._xyz.shape[0])
        if count <= 0:
            raise ValueError("base Gaussian count must be positive")
        self.element_count = count
        self.bounds = _positive_bounds(bounds)
        self.enable_shn = False
        self.effective_sh_degree = 0
        self.raw_xyz = nn.Parameter(torch.zeros_like(base_model._xyz))
        self.raw_log_scaling = nn.Parameter(torch.zeros_like(base_model._scaling))
        self.raw_rotvec = nn.Parameter(torch.zeros(count, 3, device=base_model._xyz.device, dtype=base_model._xyz.dtype))
        self.raw_opacity = nn.Parameter(torch.zeros_like(base_model._opacity))
        self.raw_sh0 = nn.Parameter(torch.zeros_like(base_model._sh0))
        self.raw_shN = nn.Parameter(torch.zeros_like(base_model._shN), requires_grad=False)
        self.register_buffer(
            "fixed_geometry_gate", torch.ones(count, 1, device=base_model._xyz.device, dtype=base_model._xyz.dtype),
            persistent=True,
        )
        self.register_buffer(
            "fixed_appearance_gate", torch.ones(count, 1, device=base_model._xyz.device, dtype=base_model._xyz.dtype),
            persistent=True,
        )
        self.stage = 0
        self.configure_stage(0)

    def configure_stage(self, stage: int) -> None:
        if stage not in STAGE_TRAINABLE:
            raise ValueError("fixed-open stage must be in [0,3]")
        trainable = STAGE_TRAINABLE[stage]
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name in trainable)
        self.stage = int(stage)

    def parameter_groups(self, learning_rates: Mapping[str, float]) -> tuple[list[dict[str, Any]], list[str]]:
        expected = {"geometry_residuals", "appearance_residuals"}
        if set(learning_rates) != expected:
            raise ValueError("fixed-open optimizer must contain only geometry/appearance residual groups")
        named = dict(self.named_parameters())
        groups = [
            {
                "params": [named[name] for name in ("raw_xyz", "raw_log_scaling", "raw_rotvec")],
                "lr": float(learning_rates["geometry_residuals"]),
            },
            {
                "params": [named[name] for name in ("raw_opacity", "raw_sh0", "raw_shN")],
                "lr": float(learning_rates["appearance_residuals"]),
            },
        ]
        return groups, ["geometry_residuals", "appearance_residuals"]

    def _raw_residuals(self, base_model: Any) -> GaussianClothingResiduals:
        values = GaussianClothingResiduals(
            delta_xyz=self.bounds["xyz"] * torch.tanh(self.raw_xyz),
            delta_log_scaling=self.bounds["log_scaling"] * torch.tanh(self.raw_log_scaling),
            delta_rotvec=_bounded_rotvec(self.raw_rotvec, self.bounds["rotation"]),
            delta_opacity_logit=self.bounds["opacity_logit"] * torch.tanh(self.raw_opacity),
            delta_sh0=self.bounds["sh0"] * torch.tanh(self.raw_sh0),
            delta_shN=torch.zeros_like(self.raw_shN),
        )
        return values.validate(base_model)

    def forward(self, base_model: Any) -> OracleForwardOutput:
        residuals = self._raw_residuals(base_model)
        zero = residuals.delta_xyz.sum() * 0
        regularization = OracleRegularizationOutput(
            residual_magnitude={
                name: getattr(residuals, name).abs().mean()
                for name in (
                    "delta_xyz", "delta_log_scaling", "delta_rotvec",
                    "delta_opacity_logit", "delta_sh0", "delta_shN",
                )
            },
            geometry_gate_sparsity=zero,
            appearance_gate_sparsity=zero,
            gate_binary=zero,
            graph_gate_smoothness=zero,
            graph_residual_smoothness=zero,
        )
        return OracleForwardOutput(
            raw_residuals=residuals,
            gated_residuals=residuals,
            gaussian_residuals=residuals,
            canonical_overrides=compose_canonical_gaussian_overrides(base_model, residuals),
            geometry_gate=self.fixed_geometry_gate,
            appearance_gate=self.fixed_appearance_gate,
            gaussian_geometry_gate=self.fixed_geometry_gate,
            gaussian_appearance_gate=self.fixed_appearance_gate,
            regularization=regularization,
            oracle_type=self.oracle_type,
        )


def fixed_open_oracle_contract(oracle: FixedOpenGaussianOracle) -> dict[str, Any]:
    parameter_names = [name for name, _ in oracle.named_parameters()]
    gate_parameters = [name for name in parameter_names if "gate" in name.lower()]
    return {
        "oracle_type": oracle.oracle_type,
        "shared_canonical_residual": True,
        "condition_specific_residual": False,
        "parameter_names": parameter_names,
        "gate_parameter_names": gate_parameters,
        "geometry_gate_fixed_value": 1.0,
        "appearance_gate_fixed_value": 1.0,
        "opacity_gate_fixed_value": 1.0,
        "SHN_enabled": False,
        "rotation_path": "R2 local wxyz q_base * q_delta via compose_canonical_gaussian_overrides",
        "pass": not gate_parameters and not any(token in name for name in parameter_names for token in ("condition", "view", "camera")),
    }
