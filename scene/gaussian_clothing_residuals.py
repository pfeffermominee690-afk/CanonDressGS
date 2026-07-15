from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F


RESIDUAL_CONTRACT_VERSION = 1
CHANNELS = (
    "delta_xyz",
    "delta_log_scaling",
    "delta_rotvec",
    "delta_opacity_logit",
    "delta_sh0",
    "delta_shN",
)
QUATERNION_CONVENTION = "wxyz"
ROTATION_COMPOSITION = "local: q_dressed = normalize(q_base * q_delta)"


def _validate_tensor(value: torch.Tensor | None, name: str, shape: tuple[int, ...]) -> None:
    if value is None:
        return
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor or None")
    if tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {tuple(shape)}")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must have a floating-point dtype")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")


@dataclass(frozen=True)
class AnchorClothingResiduals:
    delta_xyz: torch.Tensor | None = None
    delta_log_scaling: torch.Tensor | None = None
    delta_rotvec: torch.Tensor | None = None
    delta_opacity_logit: torch.Tensor | None = None
    delta_sh0: torch.Tensor | None = None
    delta_shN: torch.Tensor | None = None

    @classmethod
    def zeros(cls, anchor_count: int, base_model: Any, enabled_channels: Iterable[str] = CHANNELS):
        enabled = set(enabled_channels)
        unknown = enabled.difference(CHANNELS)
        if unknown:
            raise ValueError(f"unknown residual channels: {sorted(unknown)}")
        reference = base_model._xyz
        shapes = {
            "delta_xyz": (anchor_count, 3),
            "delta_log_scaling": (anchor_count, base_model._scaling.shape[-1]),
            "delta_rotvec": (anchor_count, 3),
            "delta_opacity_logit": (anchor_count, int(torch.tensor(base_model._opacity.shape[1:]).prod().item()) if base_model._opacity.ndim > 1 else 1),
            "delta_sh0": (anchor_count, int(torch.tensor(base_model._sh0.shape[1:]).prod().item())),
            "delta_shN": (anchor_count, int(torch.tensor(base_model._shN.shape[1:]).prod().item())),
        }
        return cls(**{
            name: torch.zeros(shapes[name], device=reference.device, dtype=reference.dtype)
            if name in enabled else None for name in CHANNELS
        })

    def validate(self, anchor_count: int, base_model: Any) -> "AnchorClothingResiduals":
        expected = AnchorClothingResiduals.zeros(anchor_count, base_model)
        for field in fields(self):
            _validate_tensor(getattr(self, field.name), field.name, getattr(expected, field.name).shape)
        return self

    def to(self, *args, **kwargs) -> "AnchorClothingResiduals":
        return type(self)(**{field.name: None if getattr(self, field.name) is None else getattr(self, field.name).to(*args, **kwargs) for field in fields(self)})

    def enabled_channels(self) -> list[str]:
        return [field.name for field in fields(self) if getattr(self, field.name) is not None]

    def as_dict(self, cpu: bool = False) -> dict[str, torch.Tensor | None]:
        return {field.name: None if getattr(self, field.name) is None else getattr(self, field.name).detach().cpu() if cpu else getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True)
class GaussianClothingResiduals:
    delta_xyz: torch.Tensor | None = None
    delta_log_scaling: torch.Tensor | None = None
    delta_rotvec: torch.Tensor | None = None
    delta_opacity_logit: torch.Tensor | None = None
    delta_sh0: torch.Tensor | None = None
    delta_shN: torch.Tensor | None = None

    @classmethod
    def zeros(cls, base_model: Any, enabled_channels: Iterable[str] = CHANNELS):
        enabled = set(enabled_channels)
        unknown = enabled.difference(CHANNELS)
        if unknown:
            raise ValueError(f"unknown residual channels: {sorted(unknown)}")
        shapes = {
            "delta_xyz": base_model._xyz.shape,
            "delta_log_scaling": base_model._scaling.shape,
            "delta_rotvec": (base_model._rotation.shape[0], 3),
            "delta_opacity_logit": base_model._opacity.shape,
            "delta_sh0": base_model._sh0.shape,
            "delta_shN": base_model._shN.shape,
        }
        return cls(**{
            name: torch.zeros(shapes[name], device=base_model._xyz.device, dtype=base_model._xyz.dtype)
            if name in enabled else None for name in CHANNELS
        })

    def validate(self, base_model: Any) -> "GaussianClothingResiduals":
        expected = GaussianClothingResiduals.zeros(base_model)
        for field in fields(self):
            _validate_tensor(getattr(self, field.name), field.name, getattr(expected, field.name).shape)
        return self

    def to(self, *args, **kwargs) -> "GaussianClothingResiduals":
        return type(self)(**{field.name: None if getattr(self, field.name) is None else getattr(self, field.name).to(*args, **kwargs) for field in fields(self)})

    def enabled_channels(self) -> list[str]:
        return [field.name for field in fields(self) if getattr(self, field.name) is not None]

    def as_dict(self, cpu: bool = False) -> dict[str, torch.Tensor | None]:
        return {field.name: None if getattr(self, field.name) is None else getattr(self, field.name).detach().cpu() if cpu else getattr(self, field.name) for field in fields(self)}

    @classmethod
    def from_dict(cls, values: Mapping[str, torch.Tensor | None]) -> "GaussianClothingResiduals":
        unknown = set(values).difference(CHANNELS)
        if unknown:
            raise ValueError(f"unknown residual fields: {sorted(unknown)}")
        return cls(**{name: values.get(name) for name in CHANNELS})


@dataclass(frozen=True)
class CanonicalGaussianOverrides:
    xyz: torch.Tensor
    scaling: torch.Tensor
    rotation: torch.Tensor
    opacity: torch.Tensor
    sh0: torch.Tensor
    shN: torch.Tensor

    def validate(self, base_model: Any) -> "CanonicalGaussianOverrides":
        expected = {
            "xyz": base_model._xyz.shape, "scaling": base_model._scaling.shape,
            "rotation": base_model._rotation.shape, "opacity": base_model._opacity.shape,
            "sh0": base_model._sh0.shape, "shN": base_model._shN.shape,
        }
        for field in fields(self):
            _validate_tensor(getattr(self, field.name), field.name, expected[field.name])
        return self

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


def axis_angle_to_quaternion_wxyz(rotvec: torch.Tensor) -> torch.Tensor:
    if rotvec.shape[-1] != 3:
        raise ValueError("rotvec must end in dimension 3")
    angle = torch.linalg.vector_norm(rotvec, dim=-1, keepdim=True)
    half = angle * 0.5
    scale = torch.where(angle > 1e-8, torch.sin(half) / angle, 0.5 - angle.square() / 48)
    return torch.cat([torch.cos(half), rotvec * scale], dim=-1)


def quaternion_multiply_wxyz(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape or left.shape[-1] != 4:
        raise ValueError("quaternion operands must have matching [...,4] shapes")
    lw, lx, ly, lz = left.unbind(-1); rw, rx, ry, rz = right.unbind(-1)
    return torch.stack((
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ), dim=-1)


def compose_canonical_gaussian_overrides(
    base_model: Any,
    gaussian_residuals: GaussianClothingResiduals,
    enabled_channels: Iterable[str] | None = None,
) -> CanonicalGaussianOverrides:
    gaussian_residuals.validate(base_model)
    enabled = set(gaussian_residuals.enabled_channels() if enabled_channels is None else enabled_channels)
    unknown = enabled.difference(CHANNELS)
    if unknown:
        raise ValueError(f"unknown enabled channels: {sorted(unknown)}")
    for name in CHANNELS:
        value = getattr(gaussian_residuals, name)
        if name in enabled and value is None:
            raise ValueError(f"enabled channel {name} has no tensor")
        if name not in enabled and value is not None and torch.count_nonzero(value).item() != 0:
            raise ValueError(f"disabled channel {name} must be absent or strictly zero")

    def residual(name: str, fallback: torch.Tensor) -> torch.Tensor:
        value = getattr(gaussian_residuals, name)
        return torch.zeros_like(fallback) if value is None or name not in enabled else value

    delta_quaternion = axis_angle_to_quaternion_wxyz(
        residual("delta_rotvec", torch.zeros(base_model._rotation.shape[0], 3, device=base_model._rotation.device, dtype=base_model._rotation.dtype))
    )
    rotation = F.normalize(quaternion_multiply_wxyz(F.normalize(base_model._rotation, dim=-1), delta_quaternion), dim=-1)
    overrides = CanonicalGaussianOverrides(
        xyz=base_model._xyz + residual("delta_xyz", base_model._xyz),
        scaling=base_model._scaling + residual("delta_log_scaling", base_model._scaling),
        rotation=rotation,
        opacity=base_model._opacity + residual("delta_opacity_logit", base_model._opacity),
        sh0=base_model._sh0 + residual("delta_sh0", base_model._sh0),
        shN=base_model._shN + residual("delta_shN", base_model._shN),
    )
    return overrides.validate(base_model)


def interpolate_anchor_clothing_residuals(
    anchor_residuals: AnchorClothingResiduals,
    gaussian_anchor_indices: torch.Tensor,
    gaussian_anchor_weights: torch.Tensor,
    base_model: Any,
) -> GaussianClothingResiduals:
    present = [getattr(anchor_residuals, name) for name in CHANNELS if getattr(anchor_residuals, name) is not None]
    if not present:
        raise ValueError("at least one anchor residual channel must be present")
    anchor_count = present[0].shape[0]
    anchor_residuals.validate(anchor_count, base_model)
    if gaussian_anchor_indices.ndim != 2 or gaussian_anchor_weights.shape != gaussian_anchor_indices.shape:
        raise ValueError("interpolation indices/weights must have matching [N,K] shapes")
    weights = gaussian_anchor_weights.unsqueeze(-1)
    values: dict[str, torch.Tensor | None] = {}
    for name in CHANNELS:
        value = getattr(anchor_residuals, name)
        if value is None:
            values[name] = None
            continue
        interpolated = (value[gaussian_anchor_indices] * weights).sum(dim=1)
        target_shape = getattr(GaussianClothingResiduals.zeros(base_model), name).shape
        values[name] = interpolated.reshape(target_shape)
    return GaussianClothingResiduals(**values).validate(base_model)


def residual_contract_metadata(base_model: Any, enabled_channels: Iterable[str]) -> dict[str, Any]:
    return {
        "residual_contract_version": RESIDUAL_CONTRACT_VERSION,
        "enabled_channels": list(enabled_channels),
        "quaternion_convention": QUATERNION_CONVENTION,
        "rotation_composition": ROTATION_COMPOSITION,
        "base_attribute_shapes": {
            "xyz": list(base_model._xyz.shape), "scaling": list(base_model._scaling.shape),
            "rotation": list(base_model._rotation.shape), "opacity": list(base_model._opacity.shape),
            "sh0": list(base_model._sh0.shape), "shN": list(base_model._shN.shape),
        },
    }
