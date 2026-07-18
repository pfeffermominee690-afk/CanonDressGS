from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from scene.gaussian_clothing_residuals import (
    CanonicalGaussianOverrides,
    GaussianClothingResiduals,
    axis_angle_to_quaternion_wxyz,
    compose_canonical_gaussian_overrides,
    quaternion_multiply_wxyz,
)


CAPACITY_LOSS_NAME = "CAPACITY_ORACLE_LOSS_V1"
GARMENT_LAYER_POINT_COUNT = 30_000
GARMENT_INITIALIZATION = "fixed_seed_base_canonical_garment_envelope_outward_12mm"
FORMAL_LBS_JOINT_COUNT = 55

# SMPL-X body joints used by the frozen subject02 base. Head, wrists/hands and
# feet/shoes are deliberately excluded from trainable garment support.
GARMENT_BODY_PARTS = frozenset({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 16, 17, 18, 19})
PROTECTED_BODY_PARTS = frozenset({10, 11, 15, 20, 21, *range(22, 55)})


def _body_part(base_model: Any) -> torch.Tensor:
    weights = base_model.get_weights.detach()
    if weights.ndim != 2 or weights.shape[0] != base_model._xyz.shape[0] or weights.shape[1] < FORMAL_LBS_JOINT_COUNT:
        raise ValueError("formal base LBS weights must have shape [N,>=55]")
    weights = weights[:, :FORMAL_LBS_JOINT_COUNT]
    if not torch.isfinite(weights).all():
        raise FloatingPointError("base LBS weights contain NaN/Inf")
    if not torch.allclose(weights.sum(1), torch.ones(weights.shape[0], device=weights.device), atol=1e-5, rtol=0):
        raise ValueError("formal base LBS weight rows must sum to one")
    return weights.argmax(1)


def garment_trainable_mask(base_model: Any) -> torch.Tensor:
    part = _body_part(base_model)
    mask = torch.zeros_like(part, dtype=torch.bool)
    for index in GARMENT_BODY_PARTS:
        mask |= part == index
    if int(mask.sum()) < GARMENT_LAYER_POINT_COUNT:
        raise ValueError("canonical garment envelope contains fewer than 30k points")
    return mask[:, None]


def _masked(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask[:, 0] if value.ndim == 1 and mask.ndim == 2 else mask
    while expanded.ndim < value.ndim:
        expanded = expanded.unsqueeze(-1)
    return value * expanded.to(device=value.device, dtype=value.dtype)


class UnboundedGaussianDeltaField(nn.Module):
    """Target-independent direct canonical delta field on the original support."""

    oracle_type = "unbounded_original_200k_gaussian_delta"

    def __init__(self, base_model: Any) -> None:
        super().__init__()
        count = int(base_model._xyz.shape[0])
        if count != 200_000:
            raise ValueError("representation triage requires the original 200k support")
        self.element_count = count
        self.register_buffer("trainable_support", garment_trainable_mask(base_model), persistent=True)
        self.raw_xyz = nn.Parameter(torch.zeros_like(base_model._xyz))
        self.raw_log_scaling = nn.Parameter(torch.zeros_like(base_model._scaling))
        self.raw_rotvec = nn.Parameter(torch.zeros(count, 3, device=base_model._xyz.device, dtype=base_model._xyz.dtype))
        self.raw_opacity = nn.Parameter(torch.zeros_like(base_model._opacity))
        self.raw_sh0 = nn.Parameter(torch.zeros_like(base_model._sh0))
        self.register_buffer("raw_shN", torch.zeros_like(base_model._shN), persistent=True)

    def residuals(self, base_model: Any) -> GaussianClothingResiduals:
        return GaussianClothingResiduals(
            delta_xyz=_masked(self.raw_xyz, self.trainable_support),
            delta_log_scaling=_masked(self.raw_log_scaling, self.trainable_support),
            delta_rotvec=_masked(self.raw_rotvec, self.trainable_support),
            delta_opacity_logit=_masked(self.raw_opacity, self.trainable_support),
            delta_sh0=_masked(self.raw_sh0, self.trainable_support),
            delta_shN=torch.zeros_like(base_model._shN),
        ).validate(base_model)

    def forward(self, base_model: Any) -> CanonicalGaussianOverrides:
        return compose_canonical_gaussian_overrides(base_model, self.residuals(base_model))

    def parameter_groups(self, geometry_lr: float, appearance_lr: float) -> tuple[list[dict[str, Any]], list[str]]:
        groups = [
            {"params": [self.raw_xyz, self.raw_log_scaling, self.raw_rotvec], "lr": float(geometry_lr)},
            {"params": [self.raw_opacity, self.raw_sh0], "lr": float(appearance_lr)},
        ]
        return groups, ["geometry", "appearance"]


@dataclass(frozen=True)
class PosedGarmentGaussians:
    xyz: torch.Tensor
    covariance: torch.Tensor
    opacity: torch.Tensor
    color: torch.Tensor


class TemporaryGarmentGaussianLayer(nn.Module):
    """Capacity-only 30k canonical garment layer with formal subject02 LBS."""

    oracle_type = "temporary_fixed_30k_formal_lbs_garment_layer"

    def __init__(self, base_model: Any, *, seed: int, point_count: int = GARMENT_LAYER_POINT_COUNT) -> None:
        super().__init__()
        if int(point_count) != GARMENT_LAYER_POINT_COUNT:
            raise ValueError("garment layer point count is frozen at 30,000")
        mask = garment_trainable_mask(base_model)[:, 0]
        candidates = torch.nonzero(mask, as_tuple=False)[:, 0].detach().cpu()
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        selected = candidates[torch.randperm(candidates.numel(), generator=generator)[:point_count]].to(base_model._xyz.device)
        xyz = base_model._xyz.detach()[selected].clone()
        center = base_model._xyz.detach().mean(0)
        outward = xyz - center
        outward[:, 1] = 0
        fallback = torch.zeros_like(outward); fallback[:, 2] = 1
        outward = torch.where(
            (torch.linalg.vector_norm(outward, dim=1, keepdim=True) > 1e-8),
            F.normalize(outward, dim=1, eps=1e-8), fallback,
        )
        xyz = xyz + outward * 0.012
        weights = base_model.get_weights.detach()[selected, :FORMAL_LBS_JOINT_COUNT].clone()
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
        self.register_buffer("source_base_indices", selected, persistent=True)
        self.register_buffer("initial_xyz", xyz, persistent=True)
        self.register_buffer("initial_log_scaling", base_model._scaling.detach()[selected].clone(), persistent=True)
        self.register_buffer("initial_rotation", F.normalize(base_model._rotation.detach()[selected].clone(), dim=-1), persistent=True)
        self.register_buffer("formal_lbs_weights", weights, persistent=True)
        self.register_buffer("initial_sh0", base_model._sh0.detach()[selected].clone(), persistent=True)
        self.xyz = nn.Parameter(xyz.clone())
        self.log_scaling = nn.Parameter(self.initial_log_scaling.clone())
        self.raw_rotvec = nn.Parameter(torch.zeros(point_count, 3, device=xyz.device, dtype=xyz.dtype))
        initial_opacity = torch.full_like(base_model._opacity.detach()[selected], -2.9444389791664403)
        self.opacity_logit = nn.Parameter(initial_opacity)
        self.sh0 = nn.Parameter(self.initial_sh0.clone())
        self.point_count = int(point_count)
        self.seed = int(seed)

    @property
    def rotation(self) -> torch.Tensor:
        delta = axis_angle_to_quaternion_wxyz(self.raw_rotvec)
        return F.normalize(quaternion_multiply_wxyz(self.initial_rotation, delta), dim=-1)

    def canonical_covariance(self) -> torch.Tensor:
        from gsplat import quat_scale_to_covar_preci

        return quat_scale_to_covar_preci(
            quats=self.rotation,
            scales=torch.exp(self.log_scaling),
            compute_preci=False,
        )[0]

    def canonical_color(self) -> torch.Tensor:
        from gsplat import spherical_harmonics

        directions = torch.ones_like(self.xyz)
        return torch.clamp_min(spherical_harmonics(0, directions, self.sh0) + 0.5, 0)

    def deform(
        self,
        rigid_transforms: torch.Tensor,
        Rh: torch.Tensor,
        Th: torch.Tensor,
    ) -> PosedGarmentGaussians:
        if rigid_transforms.shape != (FORMAL_LBS_JOINT_COUNT, 4, 4):
            raise ValueError("formal garment LBS requires [55,4,4] rigid transforms")
        blended = torch.einsum("nj,jab->nab", self.formal_lbs_weights, rigid_transforms)
        posed_body = torch.einsum("nij,nj->ni", blended, F.pad(self.xyz, (0, 1), value=1))[:, :3]
        xyz = torch.einsum("ij,nj->ni", Rh, posed_body) + Th
        linear = torch.einsum("ij,njk->nik", Rh, blended[:, :3, :3])
        covariance = linear @ self.canonical_covariance() @ linear.transpose(1, 2)
        values = PosedGarmentGaussians(
            xyz=xyz,
            covariance=covariance,
            opacity=torch.sigmoid(self.opacity_logit).reshape(-1),
            color=self.canonical_color(),
        )
        if not all(torch.isfinite(value).all() for value in (values.xyz, values.covariance, values.opacity, values.color)):
            raise FloatingPointError("posed garment layer contains NaN/Inf")
        return values


class AugmentedGarmentCapacityOracle(nn.Module):
    """Frozen base geometry plus limited old-garment appearance and a 30k layer."""

    oracle_type = "augmented_original_200k_plus_fixed_30k_garment_layer"

    def __init__(self, base_model: Any, *, seed: int, point_count: int = GARMENT_LAYER_POINT_COUNT) -> None:
        super().__init__()
        self.register_buffer("base_garment_mask", garment_trainable_mask(base_model), persistent=True)
        self.base_opacity_delta_raw = nn.Parameter(torch.zeros_like(base_model._opacity))
        self.base_sh0_delta_raw = nn.Parameter(torch.zeros_like(base_model._sh0))
        self.garment = TemporaryGarmentGaussianLayer(base_model, seed=seed, point_count=point_count)

    def base_overrides(self, base_model: Any) -> CanonicalGaussianOverrides:
        residuals = GaussianClothingResiduals(
            delta_xyz=torch.zeros_like(base_model._xyz),
            delta_log_scaling=torch.zeros_like(base_model._scaling),
            delta_rotvec=torch.zeros(base_model._rotation.shape[0], 3, device=base_model._xyz.device, dtype=base_model._xyz.dtype),
            delta_opacity_logit=_masked(4.0 * torch.tanh(self.base_opacity_delta_raw), self.base_garment_mask),
            delta_sh0=_masked(0.5 * torch.tanh(self.base_sh0_delta_raw), self.base_garment_mask),
            delta_shN=torch.zeros_like(base_model._shN),
        )
        return compose_canonical_gaussian_overrides(base_model, residuals)

    def parameter_groups(self, geometry_lr: float, appearance_lr: float) -> tuple[list[dict[str, Any]], list[str]]:
        groups = [
            {"params": [self.garment.xyz, self.garment.log_scaling, self.garment.raw_rotvec], "lr": float(geometry_lr)},
            {"params": [self.garment.opacity_logit, self.garment.sh0, self.base_opacity_delta_raw, self.base_sh0_delta_raw], "lr": float(appearance_lr)},
        ]
        return groups, ["garment_geometry", "garment_and_base_appearance"]


def _mask_like(mask: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=value.device, dtype=value.dtype)
    if mask.ndim == 2:
        mask = mask[None]
    if mask.ndim != 3 or mask.shape[0] != 1:
        raise ValueError("capacity mask must have shape [1,H,W]")
    return mask


def _masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = _mask_like(mask, prediction).expand_as(prediction)
    return ((prediction - target.to(prediction)).abs() * selected).sum() / selected.sum().clamp_min(1)


def capacity_oracle_loss_v1(
    prediction_rgb: torch.Tensor,
    prediction_alpha: torch.Tensor,
    *,
    target_rgb: torch.Tensor,
    base_rgb: torch.Tensor,
    target_foreground: torch.Tensor,
    base_foreground: torch.Tensor,
    edit_mask: torch.Tensor,
    clothing_mask: torch.Tensor,
    old_clothing_mask: torch.Tensor,
    protected_mask: torch.Tensor,
    transition_mask: torch.Tensor,
    weights: Mapping[str, float],
    stability: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Capacity loss that never preserves old garment pixels as base supervision."""

    if prediction_rgb.shape != target_rgb.shape or prediction_rgb.shape != base_rgb.shape:
        raise ValueError("capacity RGB shapes differ")
    if prediction_alpha.shape != target_foreground.shape or prediction_alpha.shape != base_foreground.shape:
        raise ValueError("capacity alpha shapes differ")
    if not all(torch.isfinite(value).all() for value in (prediction_rgb, prediction_alpha, target_rgb, base_rgb)):
        raise FloatingPointError("capacity loss input contains NaN/Inf")
    protected = _mask_like(protected_mask, prediction_rgb)
    garment = torch.maximum(torch.maximum(_mask_like(edit_mask, prediction_rgb), _mask_like(clothing_mask, prediction_rgb)), _mask_like(old_clothing_mask, prediction_rgb))
    garment = garment * (1 - protected)
    foreground = _mask_like(target_foreground, prediction_alpha)
    base_fg = _mask_like(base_foreground, prediction_alpha)
    new_silhouette = foreground * (1 - base_fg)
    boundary = _mask_like(transition_mask, prediction_alpha) * (1 - protected)
    garment_rgb = _masked_l1(prediction_rgb, target_rgb, garment)
    alpha_foreground = _masked_l1(prediction_alpha, foreground, torch.ones_like(foreground))
    new_silhouette_alpha = _masked_l1(prediction_alpha, foreground, new_silhouette)
    boundary_rgb = _masked_l1(prediction_rgb, target_rgb, boundary)
    protected_rgb = _masked_l1(prediction_rgb, base_rgb, protected)
    protected_alpha = _masked_l1(prediction_alpha, base_fg, protected)
    total = (
        float(weights["garment_rgb"]) * garment_rgb
        + float(weights["alpha_foreground"]) * alpha_foreground
        + float(weights["new_silhouette_alpha"]) * new_silhouette_alpha
        + float(weights["boundary_rgb"]) * boundary_rgb
        + float(weights["protected_rgb"]) * protected_rgb
        + float(weights["protected_alpha"]) * protected_alpha
        + float(weights["stability"]) * stability
    )
    return {
        "total": total,
        "garment_rgb": garment_rgb,
        "alpha_foreground": alpha_foreground,
        "new_silhouette_alpha": new_silhouette_alpha,
        "boundary_rgb": boundary_rgb,
        "protected_rgb": protected_rgb,
        "protected_alpha": protected_alpha,
        "stability": stability,
    }


def direct_stability(field: UnboundedGaussianDeltaField) -> torch.Tensor:
    values = (field.raw_xyz, field.raw_log_scaling, field.raw_rotvec, field.raw_opacity, field.raw_sh0)
    return sum(_masked(value, field.trainable_support).square().mean() for value in values)


def garment_stability(oracle: AugmentedGarmentCapacityOracle) -> torch.Tensor:
    garment = oracle.garment
    return (
        (garment.xyz - garment.initial_xyz).square().mean()
        + 0.1 * (garment.log_scaling - garment.initial_log_scaling).square().mean()
        + 0.1 * garment.raw_rotvec.square().mean()
        + 0.01 * oracle.base_opacity_delta_raw.square().mean()
        + 0.01 * oracle.base_sh0_delta_raw.square().mean()
    )


def decide_representation_case(
    rung1_pass_or_warn_views: int,
    rung2_final_status: str | None,
    rung3_final_status: str | None,
) -> str:
    """Deterministic frozen Case A–E decision matrix for one outfit."""

    rung1_pass = int(rung1_pass_or_warn_views) >= 3
    rung2_pass = bool(rung2_final_status and rung2_final_status.endswith("_PASS"))
    rung3_pass = bool(rung3_final_status and rung3_final_status.endswith("_PASS"))
    if rung1_pass and rung2_pass:
        return "A"
    if rung1_pass and not rung2_pass and rung3_pass:
        return "B"
    if not rung1_pass and rung3_pass:
        return "C"
    if rung1_pass and not rung2_pass and not rung3_pass:
        return "D"
    return "E"
