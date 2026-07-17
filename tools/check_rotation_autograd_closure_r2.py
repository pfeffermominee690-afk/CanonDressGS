from __future__ import annotations

import math
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene.anchor_clothing_mlp import AnchorClothingMLP
from scene.full_attribute_oracle import AnchorResidualOracle, GaussianResidualOracle
from scene.gaussian_clothing_residuals import (
    CHANNELS,
    GaussianClothingResiduals,
    axis_angle_to_quaternion_wxyz,
    compose_canonical_gaussian_overrides,
    interpolate_anchor_clothing_residuals,
)

CUDA_FINITE_TESTED = False


class Base:
    def __init__(self, count: int = 4, *, dtype: torch.dtype = torch.float32, device: str = "cpu") -> None:
        self._xyz = torch.zeros(count, 3, dtype=dtype, device=device)
        self._scaling = torch.tensor([[math.log(2.0), 0.0, math.log(0.5)]], dtype=dtype, device=device).repeat(count, 1)
        self._rotation = torch.zeros(count, 4, dtype=dtype, device=device)
        self._rotation[:, 0] = 1
        self._opacity = torch.zeros(count, 1, dtype=dtype, device=device)
        self._sh0 = torch.zeros(count, 1, 3, dtype=dtype, device=device)
        self._shN = torch.zeros(count, 3, 3, dtype=dtype, device=device)


def _replace(residuals: GaussianClothingResiduals, **updates) -> GaussianClothingResiduals:
    values = {name: getattr(residuals, name) for name in CHANNELS}
    values.update(updates)
    return GaussianClothingResiduals(**values)


def _quaternion_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quaternions.unbind(-1)
    return torch.stack((
        1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w),
        2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
        2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y),
    ), dim=-1).reshape(-1, 3, 3)


def _covariance(base: Base, quaternion: torch.Tensor) -> torch.Tensor:
    rotation = _quaternion_matrix(quaternion)
    scales = torch.exp(base._scaling)
    return rotation @ torch.diag_embed(scales.square()) @ rotation.transpose(-1, -2)


def _zero_composition(base: Base, rotvec: torch.Tensor):
    residuals = _replace(GaussianClothingResiduals.zeros(base), delta_rotvec=rotvec)
    return compose_canonical_gaussian_overrides(base, residuals)


def test_zero_rotvec_output_matches_base_quaternion() -> None:
    torch.manual_seed(7)
    base = Base(16, dtype=torch.float64)
    base._rotation = F.normalize(torch.randn_like(base._rotation), dim=-1)
    rotvec = torch.zeros(16, 3, dtype=torch.float64, requires_grad=True)
    composed = _zero_composition(base, rotvec).rotation
    assert torch.allclose(composed, base._rotation, atol=1e-12, rtol=1e-12)


def test_zero_rotvec_keeps_autograd_dependency() -> None:
    base = Base(2, dtype=torch.float64)
    rotvec = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
    composed = _zero_composition(base, rotvec).rotation
    assert composed.requires_grad and composed.grad_fn is not None
    weights = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64)
    (composed[:, 1:] * weights).sum().backward()
    assert rotvec.grad is not None and torch.isfinite(rotvec.grad).all()
    assert torch.count_nonzero(rotvec.grad) > 0


def test_small_rotvec_matches_first_order_behavior() -> None:
    rotvec = torch.tensor([[1e-8, -1e-6, 1e-4]], dtype=torch.float64)
    delta = axis_angle_to_quaternion_wxyz(rotvec)
    assert torch.allclose(delta[:, 1:], 0.5 * rotvec, atol=1e-13, rtol=1e-9)


def test_rotvec_to_quaternion_is_finite_at_zero() -> None:
    global CUDA_FINITE_TESTED
    devices = ["cpu"]
    if torch.cuda.is_available():
        try:
            torch.sinc(torch.zeros(1, device="cuda"))
            torch.cuda.synchronize()
            devices.append("cuda")
            CUDA_FINITE_TESTED = True
        except RuntimeError:
            if os.environ.get("CANONDRESSGS_REQUIRE_CUDA_TEST") == "1":
                raise
            print("CUDA sinc runtime unavailable locally; CUDA coverage deferred to cloud")
    elif os.environ.get("CANONDRESSGS_REQUIRE_CUDA_TEST") == "1":
        raise AssertionError("CUDA coverage is required but CUDA is unavailable")
    for device in devices:
        rotvec = torch.tensor(
            [[0.0, 0.0, 0.0], [1e-8, 0.0, 0.0], [0.0, 1e-6, 0.0], [0.0, 0.0, 1e-4]],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        quaternion = axis_angle_to_quaternion_wxyz(rotvec)
        assert quaternion.shape == (4, 4) and torch.isfinite(quaternion).all()
        quaternion[:, 1:].sum().backward()
        assert rotvec.grad is not None and torch.isfinite(rotvec.grad).all()


def test_zero_rotvec_changes_anisotropic_covariance_gradient() -> None:
    base = Base(1, dtype=torch.float64)
    rotvec = torch.zeros(1, 3, dtype=torch.float64, requires_grad=True)
    covariance = _covariance(base, _zero_composition(base, rotvec).rotation)
    covariance[0, 0, 1].backward()
    assert rotvec.grad is not None and torch.isfinite(rotvec.grad).all()
    assert torch.count_nonzero(rotvec.grad) > 0


def test_rotation_autograd_matches_finite_difference_at_zero() -> None:
    base = Base(1, dtype=torch.float64)
    amplitude = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    rotvec = torch.stack((amplitude * 0, amplitude * 0, amplitude)).reshape(1, 3)
    value = _covariance(base, _zero_composition(base, rotvec).rotation)[0, 0, 1]
    gradient = torch.autograd.grad(value, amplitude)[0]
    epsilon = 1e-6

    def evaluate(number: float) -> torch.Tensor:
        rv = torch.tensor([[0.0, 0.0, number]], dtype=torch.float64)
        return _covariance(base, _zero_composition(base, rv).rotation)[0, 0, 1]

    finite_difference = (evaluate(epsilon) - evaluate(-epsilon)) / (2 * epsilon)
    relative_error = (gradient - finite_difference).abs() / finite_difference.abs().clamp_min(1e-12)
    assert gradient * finite_difference > 0
    assert relative_error < 1e-6


def test_isotropic_rotation_may_have_zero_render_gradient() -> None:
    base = Base(1, dtype=torch.float64)
    base._scaling.zero_()
    rotvec = torch.zeros(1, 3, dtype=torch.float64, requires_grad=True)
    covariance = _covariance(base, _zero_composition(base, rotvec).rotation)
    covariance[0, 0, 1].backward()
    assert rotvec.grad is not None and torch.isfinite(rotvec.grad).all()
    assert torch.allclose(rotvec.grad, torch.zeros_like(rotvec.grad), atol=1e-12, rtol=0)


def test_gaussian_oracle_zero_rotation_is_connected() -> None:
    base = Base(2, dtype=torch.float64)
    oracle = GaussianResidualOracle(base).to(dtype=torch.float64)
    oracle.configure_stage(2)
    output = oracle(base)
    assert output.raw_residuals.delta_rotvec.requires_grad
    value = _covariance(base, output.canonical_overrides.rotation)[:, 0, 1].sum()
    value.backward()
    assert oracle.raw_rotvec.grad is not None and torch.isfinite(oracle.raw_rotvec.grad).all()
    assert torch.count_nonzero(oracle.raw_rotvec.grad) > 0


def test_anchor_oracle_zero_rotation_is_connected() -> None:
    base = Base(4, dtype=torch.float64)
    indices = torch.tensor([[0, 1], [1, 2], [2, 0], [0, 2]], dtype=torch.long)
    weights = torch.full((4, 2), 0.5, dtype=torch.float64)
    oracle = AnchorResidualOracle(base, 3, indices, weights).to(dtype=torch.float64)
    oracle.configure_stage(2)
    output = oracle(base)
    assert output.gaussian_residuals.delta_rotvec.grad_fn is not None
    value = _covariance(base, output.canonical_overrides.rotation)[:, 0, 1].sum()
    value.backward()
    assert oracle.raw_rotvec.grad is not None and torch.isfinite(oracle.raw_rotvec.grad).all()
    assert torch.count_nonzero(oracle.raw_rotvec.grad) > 0


def test_image_conditioned_rotation_head_zero_output_is_connected() -> None:
    torch.manual_seed(9)
    base = Base(4)
    mlp = AnchorClothingMLP(anchor_feature_dim=5, clothing_dim=4, hidden_dim=8, num_layers=2)
    mlp.configure_six_channel_decoder({
        "xyz": {"enabled": True, "max_abs": 0.05},
        "log_scaling": {"enabled": True, "max_abs": 0.35},
        "rotation": {"enabled": True, "max_angle_rad": 0.2617993878},
        "opacity_logit": {"enabled": True, "max_abs": 2.0},
        "sh0": {"enabled": True, "max_abs": 0.25},
        "shN": {"enabled": False, "max_abs": 0.10},
    }, shN_flat_dim=9)
    anchor_features = torch.randn(3, 5)
    gamma = [torch.zeros(1, 8)]
    beta = [torch.zeros(1, 8)]
    raw, bounded = mlp.forward_film_six_channel_outputs(anchor_features, gamma, beta)
    assert torch.equal(raw.delta_rotvec, torch.zeros_like(raw.delta_rotvec))
    assert bounded.delta_rotvec.grad_fn is not None
    indices = torch.tensor([[0, 1], [1, 2], [2, 0], [0, 2]], dtype=torch.long)
    weights = torch.full((4, 2), 0.5)
    gaussian = interpolate_anchor_clothing_residuals(bounded, indices, weights, base)
    rotation = compose_canonical_gaussian_overrides(base, gaussian).rotation
    assert rotation.grad_fn is not None
    _covariance(base, rotation)[:, 0, 1].sum().backward()
    assert mlp.rotation_head.weight.grad is not None
    assert mlp.rotation_head.bias.grad is not None
    assert torch.isfinite(mlp.rotation_head.weight.grad).all()
    assert torch.count_nonzero(mlp.rotation_head.weight.grad) > 0


def test_zero_rotation_forward_regression() -> None:
    torch.manual_seed(11)
    base = Base(8, dtype=torch.float64)
    base._rotation = F.normalize(torch.randn_like(base._rotation), dim=-1)
    rotvec = torch.zeros(8, 3, dtype=torch.float64, requires_grad=True)
    output = _zero_composition(base, rotvec)
    assert torch.allclose(output.rotation, base._rotation, atol=1e-12, rtol=1e-12)
    assert torch.allclose(_covariance(base, output.rotation), _covariance(base, base._rotation), atol=1e-12, rtol=1e-12)


def test_other_five_residual_channels_unchanged() -> None:
    torch.manual_seed(13)
    base = Base(4, dtype=torch.float64)
    residuals = GaussianClothingResiduals(
        delta_xyz=torch.randn_like(base._xyz) * 0.01,
        delta_log_scaling=torch.randn_like(base._scaling) * 0.01,
        delta_rotvec=torch.zeros(4, 3, dtype=torch.float64, requires_grad=True),
        delta_opacity_logit=torch.randn_like(base._opacity) * 0.01,
        delta_sh0=torch.randn_like(base._sh0) * 0.01,
        delta_shN=torch.randn_like(base._shN) * 0.01,
    )
    output = compose_canonical_gaussian_overrides(base, residuals)
    assert torch.equal(output.xyz, base._xyz + residuals.delta_xyz)
    assert torch.equal(output.scaling, base._scaling + residuals.delta_log_scaling)
    assert torch.equal(output.opacity, base._opacity + residuals.delta_opacity_logit)
    assert torch.equal(output.sh0, base._sh0 + residuals.delta_sh0)
    assert torch.equal(output.shN, base._shN + residuals.delta_shN)


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"{test.__name__}: PASS")
    print(f"rotation autograd closure R2 checks: PASS ({len(tests)} tests; cuda_finite_tested={CUDA_FINITE_TESTED})")


if __name__ == "__main__":
    main()
