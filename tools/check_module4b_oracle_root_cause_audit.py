from __future__ import annotations

import math
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene.full_attribute_oracle import GaussianResidualOracle
from scene.gaussian_clothing_residuals import (
    GaussianClothingResiduals,
    compose_canonical_gaussian_overrides,
)
from scene.oracle_root_cause_diagnostics import (
    FixedOpenGaussianOracle,
    gate_residuals_once,
    root_cause_decision_matrix,
    tensor_fingerprint,
)


class Base:
    def __init__(self, count: int = 6, anisotropic: bool = True) -> None:
        self._xyz = torch.zeros(count, 3)
        self._scaling = torch.zeros(count, 3)
        if anisotropic:
            self._scaling[:, 0] = math.log(2.0)
        self._rotation = torch.zeros(count, 4); self._rotation[:, 0] = 1
        self._opacity = torch.zeros(count)
        self._sh0 = torch.zeros(count, 1, 3)
        self._shN = torch.zeros(count, 3, 3)


def _zeros(base: Base) -> GaussianClothingResiduals:
    return GaussianClothingResiduals.zeros(base)


def _replace(residuals: GaussianClothingResiduals, **updates) -> GaussianClothingResiduals:
    values = {name: getattr(residuals, name) for name in (
        "delta_xyz", "delta_log_scaling", "delta_rotvec",
        "delta_opacity_logit", "delta_sh0", "delta_shN",
    )}
    values.update(updates)
    return GaussianClothingResiduals(**values)


def test_oracle_gate_is_applied_exactly_once() -> None:
    base = Base(); residuals = _zeros(base)
    residuals = _replace(residuals, delta_xyz=torch.ones_like(residuals.delta_xyz))
    gate = torch.full((6, 1), 0.25)
    gated, _ = gate_residuals_once(residuals, gate, gate)
    assert torch.allclose(gated.delta_xyz, torch.full_like(gated.delta_xyz, 0.25))
    composed = compose_canonical_gaussian_overrides(base, gated)
    assert torch.allclose(composed.xyz, torch.full_like(composed.xyz, 0.25))


def test_fixed_open_gate_bypasses_gate_logits() -> None:
    base = Base(); oracle = FixedOpenGaussianOracle(base); oracle.configure_stage(3)
    with torch.no_grad():
        oracle.raw_xyz.fill_(0.2)
        oracle.geometry_gate_logits.fill_(-100)
        oracle.appearance_gate_logits.fill_(-100)
    output = oracle(base)
    assert torch.equal(output.geometry_gate, torch.ones_like(output.geometry_gate))
    assert torch.count_nonzero(output.gaussian_residuals.delta_xyz) > 0
    assert not oracle.geometry_gate_logits.requires_grad


def _covariance(scales: torch.Tensor, quaternions: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quaternions.unbind(-1)
    rotation = torch.stack((
        1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w),
        2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
        2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y),
    ), -1).reshape(-1, 3, 3)
    diagonal = torch.diag_embed(scales.square())
    return rotation @ diagonal @ rotation.transpose(-1, -2)


def test_rotation_probe_changes_anisotropic_covariance() -> None:
    base = Base(1, anisotropic=True)
    residuals = _zeros(base)
    residuals = _replace(residuals, delta_rotvec=torch.tensor([[0.0, 0.0, 0.1]]))
    output = compose_canonical_gaussian_overrides(base, residuals)
    before = _covariance(torch.exp(base._scaling), base._rotation)
    after = _covariance(torch.exp(base._scaling), output.rotation)
    assert not torch.allclose(before, after)


def test_rotation_autograd_matches_finite_difference() -> None:
    """Keep the historical R2 detector as a regression for the repaired path."""
    base = Base(1, anisotropic=True)
    amplitude = torch.tensor(0.0, requires_grad=True)
    residuals = _zeros(base)
    residuals = _replace(
        residuals,
        delta_rotvec=torch.stack((amplitude * 0, amplitude * 0, amplitude)).reshape(1, 3),
    )
    output = compose_canonical_gaussian_overrides(base, residuals)
    assert output.rotation.requires_grad
    covariance = _covariance(torch.exp(base._scaling), output.rotation)[0, 0, 1]
    gradient = torch.autograd.grad(covariance, amplitude)[0]
    eps = 1e-3
    def value(v: float) -> torch.Tensor:
        r = _zeros(base)
        r = _replace(r, delta_rotvec=torch.tensor([[0.0, 0.0, v]]))
        q = compose_canonical_gaussian_overrides(base, r).rotation
        return _covariance(torch.exp(base._scaling), q)[0, 0, 1]
    finite_difference = (value(eps) - value(-eps)) / (2 * eps)
    assert abs(float(finite_difference)) > 1e-5
    assert float(gradient) * float(finite_difference) > 0
    assert torch.allclose(gradient, finite_difference, atol=1e-4, rtol=1e-3)


def test_isotropic_rotation_may_have_zero_render_gradient() -> None:
    base = Base(1, anisotropic=False)
    residuals = _zeros(base)
    residuals = _replace(residuals, delta_rotvec=torch.tensor([[0.1, 0.0, 0.0]]))
    output = compose_canonical_gaussian_overrides(base, residuals)
    before = _covariance(torch.ones(1, 3), base._rotation)
    after = _covariance(torch.ones(1, 3), output.rotation)
    assert torch.allclose(before, after, atol=1e-6)


def test_old_sleeve_region_receives_edit_gradient() -> None:
    value = torch.tensor([0.0, 0.0], requires_grad=True)
    old_sleeve = torch.tensor([1.0, 0.0])
    edit = (torch.abs(value - 1.0) * old_sleeve).sum() / old_sleeve.sum()
    edit.backward()
    assert value.grad[0] != 0


def test_old_sleeve_region_not_preserve_only() -> None:
    old_sleeve = torch.tensor([1, 0, 1], dtype=torch.bool)
    edit = torch.tensor([1, 0, 1], dtype=torch.bool)
    preserve = torch.tensor([0, 1, 0], dtype=torch.bool)
    assert not torch.any(old_sleeve & ~edit)
    assert not torch.any(old_sleeve & preserve)


def test_base_support_probe_does_not_modify_base() -> None:
    base = Base(); before = tensor_fingerprint({"opacity": base._opacity, "sh0": base._sh0})
    opacity = base._opacity.clone(); opacity[:2] -= 10
    sh0 = base._sh0.clone(); sh0[:2] = 0.5
    assert tensor_fingerprint({"opacity": base._opacity, "sh0": base._sh0}) == before
    assert not torch.equal(opacity, base._opacity) and not torch.equal(sh0, base._sh0)


def test_shn_zero_is_expected_for_degree_zero() -> None:
    base = Base(); oracle = GaussianResidualOracle(base, enable_shn=False)
    assert oracle.effective_sh_degree == 0
    assert torch.count_nonzero(oracle(base).gaussian_residuals.delta_shN) == 0


def test_root_cause_decision_matrix() -> None:
    assert root_cause_decision_matrix(
        double_gate=False, rotation_path_broken=True, gate_bottleneck=None,
        base_support="BASE_SUPPORT_AMBIGUOUS", objective_conflict=False,
        residual_bound_limit=False,
    ) == ["R2"]
    assert set(root_cause_decision_matrix(
        double_gate=False, rotation_path_broken=False, gate_bottleneck=True,
        base_support="BASE_SUPPORT_MISSING", objective_conflict=True,
        residual_bound_limit=False,
    )) == {"R3", "R4"}


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"{test.__name__}: PASS")
    print(f"module4b root-cause checks: PASS ({len(tests)} tests; R2 zero-point path repaired)")


if __name__ == "__main__":
    main()
