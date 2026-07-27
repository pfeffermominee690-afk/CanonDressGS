from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import torch

from scene.full_attribute_oracle import (
    GaussianResidualOracle,
    OracleForwardOutput,
)
from scene.gaussian_clothing_residuals import (
    GaussianClothingResiduals,
    compose_canonical_gaussian_overrides,
)


def gate_residuals_once(
    residuals: GaussianClothingResiduals,
    geometry_gate: torch.Tensor,
    appearance_gate: torch.Tensor,
) -> tuple[GaussianClothingResiduals, torch.Tensor]:
    """Apply the formal dual gate exactly once and expose the opacity gate."""

    if geometry_gate.shape != appearance_gate.shape or geometry_gate.ndim != 2:
        raise ValueError("geometry and appearance gates must share shape [N,1]")
    if geometry_gate.shape[1] != 1:
        raise ValueError("gate tensors must have one channel")
    opacity_gate = torch.maximum(geometry_gate, appearance_gate)

    def expand(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        if value.shape[0] != gate.shape[0]:
            raise ValueError("gate and residual leading dimensions differ")
        if value.ndim == 1:
            return gate[:, 0]
        return gate.reshape((-1,) + (1,) * (value.ndim - 1))

    return GaussianClothingResiduals(
        delta_xyz=residuals.delta_xyz * expand(geometry_gate, residuals.delta_xyz),
        delta_log_scaling=residuals.delta_log_scaling * expand(geometry_gate, residuals.delta_log_scaling),
        delta_rotvec=residuals.delta_rotvec * expand(geometry_gate, residuals.delta_rotvec),
        delta_opacity_logit=residuals.delta_opacity_logit * expand(opacity_gate, residuals.delta_opacity_logit),
        delta_sh0=residuals.delta_sh0 * expand(appearance_gate, residuals.delta_sh0),
        delta_shN=residuals.delta_shN * expand(appearance_gate, residuals.delta_shN),
    ), opacity_gate


class FixedOpenGaussianOracle(GaussianResidualOracle):
    """Diagnostic Gaussian oracle whose gates are fixed to one.

    This class is deliberately separate from the formal predictor and is only
    valid for the Module 4B-R capacity diagnosis.
    """

    def configure_stage(self, stage: int) -> None:
        super().configure_stage(stage)
        self.geometry_gate_logits.requires_grad_(False)
        self.appearance_gate_logits.requires_grad_(False)

    def parameter_groups(self, learning_rates: Mapping[str, float]):
        groups, names = super().parameter_groups(learning_rates)
        output_groups, output_names = [], []
        for group, name in zip(groups, names):
            if name not in {"geometry_gate", "appearance_gate"}:
                output_groups.append(group)
                output_names.append(name)
        return output_groups, output_names

    def forward(self, base_model: Any) -> OracleForwardOutput:
        raw = GaussianClothingResiduals(**self._bounded_values()).validate(base_model)
        ones = torch.ones(
            self.element_count,
            1,
            device=base_model._xyz.device,
            dtype=base_model._xyz.dtype,
        )
        return OracleForwardOutput(
            raw_residuals=raw,
            gated_residuals=raw,
            gaussian_residuals=raw,
            canonical_overrides=compose_canonical_gaussian_overrides(base_model, raw),
            geometry_gate=ones,
            appearance_gate=ones,
            gaussian_geometry_gate=ones,
            gaussian_appearance_gate=ones,
            regularization=self._regularization(raw, ones, ones),
        )


def root_cause_decision_matrix(
    *,
    double_gate: bool,
    rotation_path_broken: bool,
    gate_bottleneck: bool | None,
    base_support: str,
    objective_conflict: bool,
    residual_bound_limit: bool,
) -> list[str]:
    cases: list[str] = []
    if rotation_path_broken:
        cases.append("R2")
    if gate_bottleneck is True and base_support in {"BASE_SUPPORT_PRESENT", "BASE_SUPPORT_AMBIGUOUS"}:
        cases.append("R1")
    if base_support == "BASE_SUPPORT_MISSING":
        cases.append("R3")
    if objective_conflict:
        cases.append("R4")
    if residual_bound_limit:
        cases.append("R5")
    if double_gate:
        cases.append("DOUBLE_GATE_COMPOSITION_BUG")
    return cases or ["UNRESOLVED"]


def tensor_fingerprint(values: Mapping[str, torch.Tensor]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        tensor = value.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()
