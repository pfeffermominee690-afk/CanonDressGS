"""Target-independent protected-residual attribution and static guards.

This module deliberately contains no optimizer, renderer, target-mask, cloud-mask,
or outfit-specific logic.  It only operates on a frozen base-derived membership
mask and immutable Gaussian residual value objects.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from scene.gaussian_clothing_residuals import CHANNELS, GaussianClothingResiduals


DIAGNOSTIC_VARIANTS = frozenset({"D0", "D1", "D2", "D3", "D4"})
FORMAL_VARIANTS = frozenset({"F0", "F1", "F2", "F3", "F4", "F5"})
FORMAL_ZERO_CHANNELS = {
    "F0": (),
    "F1": ("delta_xyz",),
    "F2": ("delta_xyz", "delta_log_scaling", "delta_rotvec"),
    "F3": ("delta_xyz", "delta_log_scaling", "delta_rotvec", "delta_opacity_logit"),
    "F4": CHANNELS,
}
DIAGNOSTIC_ZERO_CHANNELS = {
    "D0": (),
    "D1": ("delta_xyz",),
    "D2": ("delta_xyz", "delta_log_scaling", "delta_rotvec"),
    "D3": CHANNELS,
    "D4": CHANNELS,
}
OWNERSHIP_CLASSES = (
    "DIRECT_GAUSSIAN_RESIDUAL",
    "SHARED_ANCHOR_INTERPOLATION",
    "SHARED_GLOBAL_MODULATION",
    "POSE_DEFORMATION_AMPLIFICATION",
    "COVARIANCE_ONLY_CLOUD",
    "MIXED",
    "UNRESOLVED",
)


@dataclass(frozen=True)
class GuardResult:
    residuals: GaussianClothingResiduals
    variant: str
    membership_source: str
    changed_channels: tuple[str, ...]
    diagnostic_only: bool


def membership_sha256(mask: torch.Tensor) -> str:
    value = _membership(mask).detach().cpu().to(torch.uint8).numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def _membership(mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor) or mask.ndim != 1 or mask.dtype != torch.bool:
        raise TypeError("protected membership must be a 1D bool tensor")
    return mask


def _zero_selected(value: torch.Tensor | None, mask: torch.Tensor) -> torch.Tensor | None:
    if value is None:
        return None
    if value.shape[0] != mask.numel():
        raise ValueError("membership and residual row counts differ")
    broadcast = mask
    while broadcast.ndim < value.ndim:
        broadcast = broadcast.unsqueeze(-1)
    return torch.where(broadcast, torch.zeros_like(value), value)


def apply_formal_guard(
    residuals: GaussianClothingResiduals,
    stable_protected: torch.Tensor,
    variant: str,
    *,
    membership_source: str = "base_derived_stable_protected",
) -> GuardResult:
    """Return a new residual value object; never mutate residuals or the base.

    Formal membership is intentionally constrained to the frozen base-derived
    stable-protected mask.  No target, outfit, cloud, or rendered result enters
    this function.
    """

    if variant not in FORMAL_ZERO_CHANNELS:
        raise ValueError(f"unsupported formal guard: {variant}")
    if membership_source != "base_derived_stable_protected":
        raise ValueError("formal guards require base-derived stable-protected membership")
    mask = _membership(stable_protected)
    channels = tuple(FORMAL_ZERO_CHANNELS[variant])
    values = {
        name: _zero_selected(getattr(residuals, name), mask)
        if name in channels else getattr(residuals, name)
        for name in CHANNELS
    }
    return GuardResult(
        residuals=GaussianClothingResiduals(**values),
        variant=variant,
        membership_source=membership_source,
        changed_channels=channels,
        diagnostic_only=False,
    )


def apply_diagnostic_counterfactual(
    residuals: GaussianClothingResiduals,
    cloud_indices: torch.Tensor,
    variant: str,
) -> GuardResult:
    if variant not in DIAGNOSTIC_ZERO_CHANNELS:
        raise ValueError(f"unsupported diagnostic counterfactual: {variant}")
    if cloud_indices.ndim != 1 or cloud_indices.dtype != torch.long:
        raise TypeError("cloud indices must be a 1D int64 tensor")
    present = next((getattr(residuals, name) for name in CHANNELS if getattr(residuals, name) is not None), None)
    if present is None:
        raise ValueError("at least one residual channel is required")
    mask = torch.zeros(present.shape[0], dtype=torch.bool, device=present.device)
    if cloud_indices.numel():
        if cloud_indices.min() < 0 or cloud_indices.max() >= present.shape[0]:
            raise IndexError("cloud index out of range")
        mask[cloud_indices.to(present.device)] = True
    channels = tuple(DIAGNOSTIC_ZERO_CHANNELS[variant])
    values = {
        name: _zero_selected(getattr(residuals, name), mask)
        if name in channels else getattr(residuals, name)
        for name in CHANNELS
    }
    return GuardResult(
        residuals=GaussianClothingResiduals(**values),
        variant=variant,
        membership_source="diagnostic_cloud_indices_only_not_a_formal_candidate",
        changed_channels=channels,
        diagnostic_only=True,
    )


def fixed_open_ownership_record(
    gaussian_index: int,
    *,
    production_contribution_count: int,
    production_alpha_mass: float,
    dominant_lbs_joint: int,
) -> dict[str, Any]:
    if gaussian_index < 0:
        raise ValueError("Gaussian index must be nonnegative")
    return {
        "base_gaussian_index": int(gaussian_index),
        "ownership_class": "DIRECT_GAUSSIAN_RESIDUAL",
        "raw_parameter_sources": {
            "delta_xyz": f"raw_xyz[{gaussian_index}]",
            "delta_log_scaling": f"raw_log_scaling[{gaussian_index}]",
            "delta_rotvec": f"raw_rotvec[{gaussian_index}]",
            "delta_opacity_logit": f"raw_opacity[{gaussian_index}]",
            "delta_sh0": f"raw_sh0[{gaussian_index}]",
            "delta_shN": "frozen_zero",
        },
        "independent_residual_parameter": True,
        "anchor_or_control_point": None,
        "top_k_source_anchors": [],
        "interpolation_weights": [],
        "shared_anchor": False,
        "shared_global_modulation": False,
        "condition_view_camera_parameter": False,
        "gate": {"geometry": 1.0, "appearance": 1.0, "trainable": False},
        "bounded_residual": "channel_bound * tanh(raw); rotvec uses bounded norm",
        "canonical_composition": "per-Gaussian six-channel residual composition",
        "pose_deformation": f"frozen MMLP-Human LBS; dominant_joint={int(dominant_lbs_joint)}",
        "production_contribution_count": int(production_contribution_count),
        "production_alpha_mass": float(production_alpha_mass),
        "protected_membership_effect": "loss-region safety only in P3; not residual composition",
    }


def classify_mechanism(
    *,
    independent_parameter: bool,
    shared_anchor: bool,
    shared_global: bool,
    canonical_displacement: float,
    posed_amplification_ratio: float,
    center_enters_cloud: bool,
    covariance_nonzero: bool,
) -> str:
    mechanisms = []
    if independent_parameter and canonical_displacement > 0:
        mechanisms.append("DIRECT_GAUSSIAN_RESIDUAL")
    if shared_anchor:
        mechanisms.append("SHARED_ANCHOR_INTERPOLATION")
    if shared_global:
        mechanisms.append("SHARED_GLOBAL_MODULATION")
    if canonical_displacement <= 1e-6 and posed_amplification_ratio >= 4:
        mechanisms.append("POSE_DEFORMATION_AMPLIFICATION")
    if not center_enters_cloud and covariance_nonzero:
        mechanisms.append("COVARIANCE_ONLY_CLOUD")
    if not mechanisms:
        return "UNRESOLVED"
    if len(mechanisms) == 1:
        return mechanisms[0]
    return "MIXED"


def gradient_provenance(
    final_gradient: torch.Tensor | None,
    raw_gradient: torch.Tensor | None,
    protected_mask: torch.Tensor,
    *,
    shared_parameter_gradient_norm: float = 0.0,
) -> dict[str, float | int | bool | str]:
    mask = _membership(protected_mask)

    def summarize(value: torch.Tensor | None) -> tuple[float, int]:
        if value is None:
            return 0.0, 0
        if value.shape[0] != mask.numel():
            raise ValueError("gradient row count differs from protected membership")
        rows = value[mask]
        return float(torch.linalg.vector_norm(rows.reshape(rows.shape[0], -1), dim=1).sum()) if rows.numel() else 0.0, int(torch.count_nonzero(rows).item())

    final_norm, final_nonzero = summarize(final_gradient)
    raw_norm, raw_nonzero = summarize(raw_gradient)
    indirect = final_norm == 0.0 and float(shared_parameter_gradient_norm) > 0.0
    return {
        "direct_protected_gradient_norm": final_norm,
        "direct_protected_gradient_nonzero_values": final_nonzero,
        "raw_protected_gradient_norm": raw_norm,
        "raw_protected_gradient_nonzero_values": raw_nonzero,
        "shared_parameter_gradient_norm": float(shared_parameter_gradient_norm),
        "indirect_shared_parameter_leakage": indirect,
        "classification": "INDIRECT_SHARED_PARAMETER_LEAKAGE" if indirect else "DIRECT_PROTECTED_GRADIENT",
    }


def relative_decrease(baseline: float, candidate: float) -> float:
    if baseline < 0 or candidate < 0:
        raise ValueError("nonnegative metrics required")
    return (baseline - candidate) / max(baseline, 1e-12)


def qualify_formal_guard(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    seam_pass: bool,
    o01_pass: bool,
    membership_target_independent: bool,
) -> dict[str, Any]:
    gates = {
        "cloud_alpha_mass_decrease": relative_decrease(baseline["cloud_alpha_mass"], candidate["cloud_alpha_mass"]) >= .80,
        "cloud_active_decrease": relative_decrease(baseline["cloud_active"], candidate["cloud_active"]) >= .80,
        "center_entered_decrease": relative_decrease(baseline["center_entered"], candidate["center_entered"]) >= .80,
        "trusted_expansion_npre": candidate["trusted_expansion_npre"] >= baseline["trusted_expansion_npre"] * .98,
        "trusted_expansion_recall": candidate["trusted_expansion_recall"] >= baseline["trusted_expansion_recall"] - .02,
        "trusted_removal_recall": candidate["trusted_removal_recall"] >= baseline["trusted_removal_recall"] - .03,
        "target_closer": candidate["target_closer"] >= baseline["target_closer"] - .01,
        "edit_reduction": candidate["edit_reduction"] >= baseline["edit_reduction"] - .02,
        "protected_mae": candidate["protected_mae"] <= .01,
        "background_leakage": candidate["background_leakage"] <= .03,
        "o01_regression": bool(o01_pass),
        "protected_seam": bool(seam_pass),
        "membership_target_independent": bool(membership_target_independent),
    }
    return {"status": "PASS" if all(gates.values()) else "FAIL", "gates": gates}


def choose_case(qualifications: Mapping[str, Mapping[str, Any]], *, f5_ran: bool = False) -> tuple[str, str]:
    if qualifications.get("F1", {}).get("status") == "PASS":
        return "PX", "IMPLEMENT_PROTECTED_XYZ_RESIDUAL_GUARD"
    if qualifications.get("F2", {}).get("status") == "PASS":
        return "PG", "IMPLEMENT_PROTECTED_GEOMETRY_RESIDUAL_GUARD"
    if qualifications.get("F3", {}).get("status") == "PASS":
        return "PO", "IMPLEMENT_PROTECTED_GEOMETRY_OPACITY_GUARD"
    if qualifications.get("F4", {}).get("status") == "PASS":
        return "PF", "DECOUPLE_PROTECTED_AND_GARMENT_RESIDUAL_FIELDS"
    if f5_ran and qualifications.get("F5", {}).get("status") == "PASS":
        return "PT", "IMPLEMENT_PROTECTED_BOUNDARY_TAPER"
    return "PU", "TRACE_PROTECTED_RESIDUAL_JACOBIAN_AT_ANCHOR_LEVEL"


def weighted_classification_summary(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for name in OWNERSHIP_CLASSES:
        rows = [row for row in records if row["ownership_class"] == name]
        result.append({
            "ownership_class": name,
            "gaussian_count": len(rows),
            "npre_occurrences": sum(int(row.get("npre_occurrences", 0)) for row in rows),
            "active_contributions": sum(int(row.get("active_contributions", 0)) for row in rows),
            "alpha_mass": sum(float(row.get("alpha_mass", 0.0)) for row in rows),
        })
    return result
