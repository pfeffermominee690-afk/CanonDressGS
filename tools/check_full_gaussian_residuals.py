from __future__ import annotations

import math

import torch

from scene.gaussian_clothing_residuals import (
    CHANNELS,
    AnchorClothingResiduals,
    GaussianClothingResiduals,
    axis_angle_to_quaternion_wxyz,
    compose_canonical_gaussian_overrides,
    interpolate_anchor_clothing_residuals,
    quaternion_multiply_wxyz,
)


class MockBase:
    def __init__(self, opacity_shape=(7,)):
        self._xyz = torch.randn(7, 3)
        self._scaling = torch.randn(7, 3)
        self._rotation = torch.tensor([[1.0, 0, 0, 0]]).repeat(7, 1)
        self._opacity = torch.randn(opacity_shape)
        self._sh0 = torch.randn(7, 1, 3)
        self._shN = torch.randn(7, 3, 3)


def expect_error(function, error_type):
    try:
        function()
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def main() -> None:
    base = MockBase()
    full = GaussianClothingResiduals.zeros(base).validate(base)
    assert full.enabled_channels() == list(CHANNELS)
    overrides = compose_canonical_gaussian_overrides(base, full, CHANNELS)
    for name, source in (("xyz", "_xyz"), ("scaling", "_scaling"), ("rotation", "_rotation"), ("opacity", "_opacity"), ("sh0", "_sh0"), ("shN", "_shN")):
        assert torch.equal(getattr(overrides, name), getattr(base, source))

    disabled = GaussianClothingResiduals.zeros(base, ["delta_xyz"])
    assert disabled.enabled_channels() == ["delta_xyz"]
    compose_canonical_gaussian_overrides(base, disabled, ["delta_xyz"])
    bad = GaussianClothingResiduals(delta_log_scaling=torch.ones_like(base._scaling))
    expect_error(lambda: compose_canonical_gaussian_overrides(base, bad, []), ValueError)
    expect_error(lambda: GaussianClothingResiduals(delta_xyz=torch.zeros(7, 4)).validate(base), ValueError)
    expect_error(lambda: GaussianClothingResiduals(delta_xyz=torch.full((7, 3), float("nan"))).validate(base), ValueError)

    converted = full.to(dtype=torch.float64)
    assert all(value is None or value.dtype == torch.float64 for value in converted.as_dict().values())
    for opacity_shape in ((7,), (7, 1)):
        shaped = MockBase(opacity_shape)
        assert GaussianClothingResiduals.zeros(shaped).delta_opacity_logit.shape == shaped._opacity.shape
    assert full.delta_sh0.shape == base._sh0.shape and full.delta_shN.shape == base._shN.shape

    identity = torch.tensor([[1.0, 0, 0, 0]])
    z180 = axis_angle_to_quaternion_wxyz(torch.tensor([[0.0, 0, math.pi]]))
    assert torch.allclose(z180, torch.tensor([[0.0, 0, 0, 1.0]]), atol=1e-6)
    assert torch.allclose(quaternion_multiply_wxyz(identity, z180), z180)
    x90 = axis_angle_to_quaternion_wxyz(torch.tensor([[math.pi / 2, 0, 0.0]]))
    assert x90[0, 0] > 0 and x90[0, 1] > 0 and x90[0, 3] == 0

    anchors = AnchorClothingResiduals.zeros(4, base, ["delta_xyz", "delta_shN"])
    indices = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 3], [1, 3], [0, 2], [2, 3]])
    weights = torch.full((7, 2), 0.5)
    gaussian = interpolate_anchor_clothing_residuals(anchors, indices, weights, base)
    assert gaussian.delta_xyz.shape == base._xyz.shape and gaussian.delta_shN.shape == base._shN.shape
    print("Full Gaussian residual contract tests PASS")


if __name__ == "__main__":
    main()
