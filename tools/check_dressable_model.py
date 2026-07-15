from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene.dressable_gaussian_model import DressableGaussianModel


class MinimalGaussianModel(nn.Module):
    """Minimal raw-parameter mock matching the fields used by GaussianModel."""

    def __init__(self, num_gaussians: int = 16) -> None:
        super().__init__()
        n = num_gaussians

        self._xyz = nn.Parameter(torch.arange(n * 3, dtype=torch.float32).reshape(n, 3) / 100.0)
        self._scaling = nn.Parameter(torch.full((n, 3), -4.5, dtype=torch.float32))
        self._opacity = nn.Parameter(torch.linspace(-1.0, 1.0, n, dtype=torch.float32))
        rotation = torch.zeros(n, 4, dtype=torch.float32)
        rotation[:, 0] = 1.0
        self._rotation = nn.Parameter(rotation)
        self._sh0 = nn.Parameter(torch.full((n, 1, 3), 0.25, dtype=torch.float32))
        self._shN = nn.Parameter(torch.zeros(n, 3, 3, dtype=torch.float32))


def _clone_base_params(base_model: MinimalGaussianModel) -> dict[str, torch.Tensor]:
    return {
        "_xyz": base_model._xyz.detach().clone(),
        "_scaling": base_model._scaling.detach().clone(),
        "_opacity": base_model._opacity.detach().clone(),
        "_rotation": base_model._rotation.detach().clone(),
        "_sh0": base_model._sh0.detach().clone(),
        "_shN": base_model._shN.detach().clone(),
    }


def _assert_keys(output: dict[str, torch.Tensor]) -> None:
    expected = {"xyz", "scaling", "opacity", "rotation", "sh0", "shN"}
    actual = set(output)
    if actual != expected:
        raise AssertionError(f"Unexpected output keys: {sorted(actual)}, expected {sorted(expected)}")


def _assert_shapes(output: dict[str, torch.Tensor], num_gaussians: int) -> None:
    expected_shapes = {
        "xyz": (num_gaussians, 3),
        "scaling": (num_gaussians, 3),
        "opacity": (num_gaussians,),
        "rotation": (num_gaussians, 4),
        "sh0": (num_gaussians, 1, 3),
        "shN": (num_gaussians, 3, 3),
    }

    for key, expected_shape in expected_shapes.items():
        actual_shape = tuple(output[key].shape)
        if actual_shape != expected_shape:
            raise AssertionError(f"{key} shape mismatch: got {actual_shape}, expected {expected_shape}")


def _assert_zero_offset_identity(
    output: dict[str, torch.Tensor],
    base_model: MinimalGaussianModel,
) -> None:
    comparisons = {
        "xyz": base_model._xyz,
        "scaling": base_model._scaling,
        "opacity": base_model._opacity,
        "rotation": base_model._rotation,
        "sh0": base_model._sh0,
        "shN": base_model._shN,
    }
    for key, base_value in comparisons.items():
        if not torch.equal(output[key], base_value):
            raise AssertionError(f"{key} differs from base parameter under zero offsets")


def _assert_base_unchanged(
    base_model: MinimalGaussianModel,
    before: dict[str, torch.Tensor],
) -> None:
    for name, before_value in before.items():
        current_value = getattr(base_model, name).detach()
        if not torch.equal(current_value, before_value):
            raise AssertionError(f"base parameter {name} was modified in-place")


def main() -> None:
    torch.manual_seed(0)

    num_gaussians = 16
    base_model = MinimalGaussianModel(num_gaussians)
    before = _clone_base_params(base_model)

    dressable_model = DressableGaussianModel()
    dressable_model.set_base_model(base_model)

    offsets = dressable_model.compute_clothing_offsets(cloth_id=0)
    expected_offset_shapes = {
        "delta_xyz": (num_gaussians, 3),
        "delta_scaling": (num_gaussians, 3),
        "delta_opacity": (num_gaussians,),
    }
    for key, expected_shape in expected_offset_shapes.items():
        actual_shape = tuple(offsets[key].shape)
        if actual_shape != expected_shape:
            raise AssertionError(f"{key} shape mismatch: got {actual_shape}, expected {expected_shape}")
        if not torch.equal(offsets[key], torch.zeros_like(offsets[key])):
            raise AssertionError(f"{key} is not zero in zero-offset implementation")

    dressed = dressable_model(cloth_id=0)
    _assert_keys(dressed)
    _assert_shapes(dressed, num_gaussians)
    _assert_zero_offset_identity(dressed, base_model)
    _assert_base_unchanged(base_model, before)
    print("DressableGaussianModel zero-offset test: PASS")

    mask = torch.zeros(num_gaussians, dtype=torch.bool)
    mask[4:12] = True
    mask_2d = mask[:, None]

    xyz_offset = torch.tensor([0.01, 0.0, 0.0], dtype=torch.float32)
    scaling_offset = 0.02
    opacity_offset = 0.1

    dressable_model.set_manual_clothing_offset(
        gaussian_mask=mask_2d,
        xyz_offset=xyz_offset,
        scaling_offset=scaling_offset,
        opacity_offset=opacity_offset,
    )
    dressed = dressable_model(cloth_id=0)

    expected_xyz = base_model._xyz + mask_2d.to(base_model._xyz.dtype) * xyz_offset.reshape(1, 3)
    if not torch.allclose(dressed["xyz"], expected_xyz):
        raise AssertionError("manual xyz offsets were not applied correctly")
    print("manual xyz offset test: PASS")

    expected_scaling = base_model._scaling + mask_2d.to(base_model._scaling.dtype) * scaling_offset
    if not torch.allclose(dressed["scaling"], expected_scaling):
        raise AssertionError("manual scaling offsets were not applied correctly")
    print("manual scaling offset test: PASS")

    expected_opacity = base_model._opacity + mask.to(base_model._opacity.dtype) * opacity_offset
    if not torch.allclose(dressed["opacity"], expected_opacity):
        raise AssertionError("manual opacity offsets were not applied correctly")
    print("manual opacity offset test: PASS")

    unmasked = ~mask
    if not torch.equal(dressed["xyz"][unmasked], base_model._xyz[unmasked]):
        raise AssertionError("unmasked xyz parameters changed")
    if not torch.equal(dressed["scaling"][unmasked], base_model._scaling[unmasked]):
        raise AssertionError("unmasked scaling parameters changed")
    if not torch.equal(dressed["opacity"][unmasked], base_model._opacity[unmasked]):
        raise AssertionError("unmasked opacity parameters changed")
    if not torch.equal(dressed["rotation"], base_model._rotation):
        raise AssertionError("rotation changed in manual offset stage")
    if not torch.equal(dressed["sh0"], base_model._sh0):
        raise AssertionError("sh0 changed in manual offset stage")
    if not torch.equal(dressed["shN"], base_model._shN):
        raise AssertionError("shN changed in manual offset stage")
    print("unmasked parameters unchanged: PASS")

    _assert_base_unchanged(base_model, before)

    dressable_model.clear_manual_clothing_offset()
    dressed = dressable_model(cloth_id=0)
    _assert_zero_offset_identity(dressed, base_model)
    print("clear manual offset test: PASS")

    if base_model._xyz.grad is not None:
        base_model._xyz.grad.zero_()
    loss = dressed["xyz"].sum()
    loss.backward()
    if base_model._xyz.grad is None:
        raise AssertionError("base xyz did not receive gradients")
    if not torch.equal(base_model._xyz.grad, torch.ones_like(base_model._xyz)):
        raise AssertionError("base xyz gradient is not the expected all-ones tensor")
    print("gradient test: PASS")

    invalid_shape_checks = [
        lambda: dressable_model.set_manual_clothing_offset(torch.ones(num_gaussians + 1)),
        lambda: dressable_model.set_manual_clothing_offset(mask, xyz_offset=torch.ones(2)),
        lambda: dressable_model.set_manual_clothing_offset(mask, scaling_offset=torch.ones(num_gaussians, 2)),
        lambda: dressable_model.set_manual_clothing_offset(mask, opacity_offset=torch.ones(num_gaussians, 1)),
    ]
    for check in invalid_shape_checks:
        try:
            check()
        except (TypeError, ValueError):
            continue
        raise AssertionError("invalid shape did not raise TypeError or ValueError")
    print("invalid shape test: PASS")

    _assert_base_unchanged(base_model, before)
    print("base parameters unchanged: PASS")


if __name__ == "__main__":
    main()
