from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from scene.gaussian_model import GaussianModel


class DressableGaussianModel(nn.Module):
    """A minimal wrapper for clothing-conditioned canonical Gaussian offsets.

    This first-stage implementation intentionally does not modify or render with
    the wrapped MMLPHuman ``GaussianModel``. It only exposes a zero-offset
    canonical composition interface that later modules can replace with learned
    clothing offsets.
    """

    def __init__(self, base_model: GaussianModel | None = None) -> None:
        super().__init__()
        self.base_model: GaussianModel | None = None
        if base_model is not None:
            self.set_base_model(base_model)

    def set_base_model(self, base_model: GaussianModel) -> None:
        """Attach an existing MMLPHuman GaussianModel without taking ownership."""

        self._validate_base_model(base_model)
        self.base_model = base_model

    def compute_clothing_offsets(
        self,
        cloth_id: torch.Tensor | int,
    ) -> dict[str, torch.Tensor]:
        """Return zero canonical clothing offsets for the selected clothing id.

        The current version ignores ``cloth_id`` and dynamically matches the
        wrapped base model's Gaussian count, dtype, and device.
        """

        del cloth_id
        base_model = self._require_base_model()

        return {
            "delta_xyz": torch.zeros_like(base_model._xyz),
            "delta_scaling": torch.zeros_like(base_model._scaling),
            "delta_opacity": torch.zeros_like(base_model._opacity),
        }

    def get_dressed_canonical_params(
        self,
        cloth_id: torch.Tensor | int,
    ) -> dict[str, torch.Tensor]:
        """Compose raw canonical Gaussian parameters with clothing offsets.

        Returned ``xyz``, ``scaling``, and ``opacity`` stay in the raw canonical
        parameter space. Activation functions, pose-dependent basis offsets, LBS
        deformation, and rasterization remain the responsibility of the original
        MMLPHuman ``GaussianModel`` path.
        """

        base_model = self._require_base_model()
        offsets = self.compute_clothing_offsets(cloth_id)

        return {
            "xyz": base_model._xyz + offsets["delta_xyz"],
            "scaling": base_model._scaling + offsets["delta_scaling"],
            "opacity": base_model._opacity + offsets["delta_opacity"],
            "rotation": base_model._rotation,
            "sh0": base_model._sh0,
            "shN": base_model._shN,
        }

    def forward(self, cloth_id: torch.Tensor | int) -> dict[str, torch.Tensor]:
        """Return zero-offset dressed canonical parameters."""

        return self.get_dressed_canonical_params(cloth_id)

    def _require_base_model(self) -> Any:
        if self.base_model is None:
            raise RuntimeError("DressableGaussianModel requires set_base_model() before use.")
        return self.base_model

    @staticmethod
    def _validate_base_model(base_model: Any) -> None:
        required_attrs = ("_xyz", "_scaling", "_opacity", "_rotation", "_sh0", "_shN")
        missing = [name for name in required_attrs if not hasattr(base_model, name)]
        if missing:
            raise AttributeError(f"base_model is missing required Gaussian parameters: {missing}")

        for name in required_attrs:
            value = getattr(base_model, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"base_model.{name} must be a torch.Tensor, got {type(value)!r}")

        num_gaussians = base_model._xyz.shape[0]
        expected_shapes = {
            "_xyz": (num_gaussians, 3),
            "_scaling": (num_gaussians, 3),
            "_opacity": (num_gaussians,),
            "_rotation": (num_gaussians, 4),
            "_sh0": (num_gaussians, 1, 3),
            "_shN": (num_gaussians, 3, 3),
        }

        for name, expected_shape in expected_shapes.items():
            actual_shape = tuple(getattr(base_model, name).shape)
            if actual_shape != expected_shape:
                raise ValueError(
                    f"base_model.{name} has shape {actual_shape}, expected {expected_shape}"
                )
