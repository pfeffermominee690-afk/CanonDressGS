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

    loss = dressed["xyz"].sum()
    loss.backward()
    if base_model._xyz.grad is None:
        raise AssertionError("base xyz did not receive gradients")
    if not torch.equal(base_model._xyz.grad, torch.ones_like(base_model._xyz)):
        raise AssertionError("base xyz gradient is not the expected all-ones tensor")
    print("gradient test: PASS")

    _assert_base_unchanged(base_model, before)
    print("base parameters unchanged: PASS")


if __name__ == "__main__":
    main()
