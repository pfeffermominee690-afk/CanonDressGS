"""Reference-only controller for the sealed dual-support renderer contract.

The prediction branch ends at :class:`SupportSelection`.  Endpoint paths are
resolved only afterwards by :func:`construct_dual_support_runtime`, so teacher
endpoint identity cannot enter the controller forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from scene.p0_candidate_adapters import pool_frozen_f2_reference_set
from scene.p0_candidate_initialization_protocol import (
    RANDOM_SEEDED_INITIALIZATION,
    seed_all,
)


OUTFIT_ORDER = ("O01", "O02", "O03", "O04", "O08")
SOFTMAX_TEMPERATURE = 1.0
SECONDARY_WEIGHT_THRESHOLD = 0.10
TOP2_MASS_THRESHOLD = 0.90
FORBIDDEN_FORWARD_INPUTS = (
    "target_rgb",
    "target_mask",
    "target_pose",
    "target_camera",
    "garment_id",
    "outfit_id",
    "teacher_endpoint_id",
    "teacher_residual",
    "ground_truth_mixture_weight",
    "target_render",
    "loss_target",
)


@dataclass(frozen=True)
class ControllerDistribution:
    logits: torch.Tensor
    probabilities: torch.Tensor


@dataclass(frozen=True)
class SupportSelection:
    top1_outfit: str
    top2_outfit: str
    top1_probability: float
    top2_probability: float
    top2_mass: float
    normalized_top2_weight_1: float
    normalized_top2_weight_2: float
    weight_1: float
    weight_2: float
    mode: str
    fallback_reason: str | None
    stable_rank_indices: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "top1_outfit": self.top1_outfit,
            "top2_outfit": self.top2_outfit,
            "top1_probability": self.top1_probability,
            "top2_probability": self.top2_probability,
            "top2_mass": self.top2_mass,
            "normalized_top2_weight_1": self.normalized_top2_weight_1,
            "normalized_top2_weight_2": self.normalized_top2_weight_2,
            "weight_1": self.weight_1,
            "weight_2": self.weight_2,
            "mode": self.mode,
            "fallback_reason": self.fallback_reason,
            "stable_rank_indices": list(self.stable_rank_indices),
            "tie_handling": "FROZEN_OUTFIT_ORDER",
            "target_forward_leakage": 0,
        }


@dataclass(frozen=True)
class ImmutableEndpointBranch:
    branch_index: int
    outfit_id: str
    endpoint_path: str
    opacity_weight: float
    geometry_policy: str = "IMMUTABLE_ENDPOINT_NO_INTERPOLATION"


@dataclass(frozen=True)
class DualSupportRuntimeRequest:
    mode: str
    branches: tuple[ImmutableEndpointBranch, ...]
    selected_endpoint_paths: tuple[str, ...]
    fallback_reason: str | None
    renderer_contract: str = "SEALED_DUAL_SUPPORT_OPACITY_GATE"
    geometry_interpolation: bool = False
    geometry_averaging: bool = False
    basis_coefficient_interpolation: bool = False
    target_forward_leakage: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "branches": [branch.__dict__ for branch in self.branches],
            "selected_endpoint_paths": list(self.selected_endpoint_paths),
            "fallback_reason": self.fallback_reason,
            "renderer_contract": self.renderer_contract,
            "geometry_interpolation": self.geometry_interpolation,
            "geometry_averaging": self.geometry_averaging,
            "basis_coefficient_interpolation": self.basis_coefficient_interpolation,
            "target_forward_leakage": self.target_forward_leakage,
        }


def spatial_f2_rows(
    reference_feature_maps: torch.Tensor,
    reference_clothing_masks: torch.Tensor,
    reference_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the frozen F2 mask mean/max contract to spatial feature maps.

    The frozen backbone itself is deliberately external to this module.  This
    function accepts only its spatial output and never target-view inputs.
    """
    if reference_feature_maps.ndim != 4:
        raise ValueError("reference_feature_maps must have shape [R,C,H,W]")
    count, _, height, width = reference_feature_maps.shape
    if reference_clothing_masks.shape != (count, 1, height, width):
        raise ValueError("reference_clothing_masks must match frozen F2 spatial size")
    if reference_valid.shape != (count, 1):
        raise ValueError("reference_valid must have shape [R,1]")
    if not torch.isfinite(reference_feature_maps).all():
        raise FloatingPointError("reference F2 maps contain NaN or Inf")
    if not torch.isfinite(reference_clothing_masks).all():
        raise FloatingPointError("reference clothing masks contain NaN or Inf")
    masks = reference_clothing_masks.to(reference_feature_maps).clamp(0, 1)
    mass = masks.sum(dim=(2, 3)).clamp_min(1.0e-8)
    weighted_mean = (reference_feature_maps * masks).sum(dim=(2, 3)) / mass
    lowest = torch.finfo(reference_feature_maps.dtype).min
    masked_max = reference_feature_maps.masked_fill(masks <= 0, lowest).amax(dim=(2, 3))
    if torch.any(masks.sum(dim=(2, 3)) <= 0):
        raise ValueError("every valid reference clothing mask must be non-empty")
    rows = torch.cat((weighted_mean, masked_max), dim=-1)
    return rows, reference_valid.to(rows)


class ReferenceConditionedDualSupportController(nn.Module):
    """LayerNorm + five-logit head over legal frozen reference features."""

    initialization_policy = RANDOM_SEEDED_INITIALIZATION
    outfit_order = OUTFIT_ORDER
    temperature = SOFTMAX_TEMPERATURE
    outfit_id_in_prediction_forward = False
    target_pose_camera_in_prediction_forward = False
    loads_b6_checkpoint = False
    loads_ours_v2_checkpoint = False
    loads_teacher_classifier_weights = False

    def __init__(self, *, seed: int, input_dim: int = 512) -> None:
        super().__init__()
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        seed_all(seed)
        self.seed = seed
        self.input_dim = int(input_dim)
        self.normalization = nn.LayerNorm(self.input_dim)
        self.linear = nn.Linear(self.input_dim, len(OUTFIT_ORDER))

    def forward(
        self, reference_f2: torch.Tensor, reference_valid: torch.Tensor
    ) -> ControllerDistribution:
        pooled = pool_frozen_f2_reference_set(reference_f2, reference_valid)
        if pooled.shape != (1, self.input_dim):
            raise ValueError("controller pooled feature dimension mismatch")
        logits = self.linear(self.normalization(pooled)).reshape(len(OUTFIT_ORDER))
        probabilities = torch.softmax(logits / SOFTMAX_TEMPERATURE, dim=0)
        if not torch.isfinite(logits).all() or not torch.isfinite(probabilities).all():
            raise FloatingPointError("controller distribution contains NaN or Inf")
        return ControllerDistribution(logits=logits, probabilities=probabilities)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def soft_target_cross_entropy(
    logits: torch.Tensor, target_distribution: torch.Tensor
) -> torch.Tensor:
    """Soft-label CE; mixed labels are never collapsed to hard labels."""
    if logits.shape != target_distribution.shape or logits.shape[-1] != len(OUTFIT_ORDER):
        raise ValueError("logits and target_distribution must have matching [...,5] shape")
    if not torch.isfinite(logits).all() or not torch.isfinite(target_distribution).all():
        raise FloatingPointError("soft-target loss inputs contain NaN or Inf")
    if torch.any(target_distribution < 0):
        raise ValueError("soft target cannot contain negative mass")
    sums = target_distribution.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1.0e-7, rtol=0.0):
        raise ValueError("soft target must sum to one")
    return -(target_distribution * F.log_softmax(logits / SOFTMAX_TEMPERATURE, dim=-1)).sum(dim=-1).mean()


def stable_top2_selection(probabilities: torch.Tensor) -> SupportSelection:
    """Select support with deterministic ties resolved by frozen outfit order."""
    if probabilities.shape != (len(OUTFIT_ORDER),):
        raise ValueError("probabilities must have shape [5]")
    if not torch.isfinite(probabilities).all() or torch.any(probabilities < 0):
        raise ValueError("probabilities must be finite and non-negative")
    if not torch.allclose(probabilities.sum(), probabilities.new_tensor(1.0), atol=1e-6, rtol=0):
        raise ValueError("probabilities must sum to one")
    values = [float(value) for value in probabilities.detach().cpu()]
    ranking = tuple(sorted(range(len(values)), key=lambda index: (-values[index], index)))
    first, second = ranking[:2]
    p1, p2 = values[first], values[second]
    mass = p1 + p2
    if mass <= 0:
        raise ValueError("top-2 probability mass must be positive")
    normalized_second = p2 / mass
    normalized_first = 1.0 - normalized_second
    if mass < TOP2_MASS_THRESHOLD:
        mode, reason, runtime_w1, runtime_w2 = "SINGLE_ENDPOINT", "LOW_TOP2_MASS", 1.0, 0.0
    elif normalized_second < SECONDARY_WEIGHT_THRESHOLD:
        mode, reason, runtime_w1, runtime_w2 = "SINGLE_ENDPOINT", "LOW_SECONDARY_WEIGHT", 1.0, 0.0
    else:
        mode, reason = "DUAL_SUPPORT", None
        runtime_w1, runtime_w2 = normalized_first, normalized_second
    return SupportSelection(
        top1_outfit=OUTFIT_ORDER[first],
        top2_outfit=OUTFIT_ORDER[second],
        top1_probability=p1,
        top2_probability=p2,
        top2_mass=mass,
        normalized_top2_weight_1=normalized_first,
        normalized_top2_weight_2=normalized_second,
        weight_1=runtime_w1,
        weight_2=runtime_w2,
        mode=mode,
        fallback_reason=reason,
        stable_rank_indices=ranking,
    )


def construct_dual_support_runtime(
    selection: SupportSelection, endpoint_bank: Mapping[str, str]
) -> DualSupportRuntimeRequest:
    """Resolve immutable endpoint paths after reference-only prediction."""
    if tuple(endpoint_bank.keys()) != OUTFIT_ORDER:
        raise ValueError("endpoint_bank must preserve the frozen five-outfit order")
    if any(not isinstance(path, str) or not path for path in endpoint_bank.values()):
        raise ValueError("endpoint paths must be non-empty strings")
    first = ImmutableEndpointBranch(
        branch_index=1,
        outfit_id=selection.top1_outfit,
        endpoint_path=endpoint_bank[selection.top1_outfit],
        opacity_weight=selection.weight_1,
    )
    branches: tuple[ImmutableEndpointBranch, ...]
    if selection.mode == "SINGLE_ENDPOINT":
        branches = (first,)
    elif selection.mode == "DUAL_SUPPORT":
        second = ImmutableEndpointBranch(
            branch_index=2,
            outfit_id=selection.top2_outfit,
            endpoint_path=endpoint_bank[selection.top2_outfit],
            opacity_weight=selection.weight_2,
        )
        branches = (first, second)
    else:
        raise ValueError("unsupported controller mode")
    if abs(sum(branch.opacity_weight for branch in branches) - 1.0) > 1.0e-7:
        raise RuntimeError("runtime branch opacity weights are not normalized")
    return DualSupportRuntimeRequest(
        mode=selection.mode,
        branches=branches,
        selected_endpoint_paths=tuple(branch.endpoint_path for branch in branches),
        fallback_reason=selection.fallback_reason,
    )


def inference_result_schema(
    distribution: ControllerDistribution,
    selection: SupportSelection,
    runtime: DualSupportRuntimeRequest,
) -> dict[str, Any]:
    """Serializable audit schema; tensors are copied to plain CPU values."""
    return {
        "logits": [float(value) for value in distribution.logits.detach().cpu()],
        "probabilities": [float(value) for value in distribution.probabilities.detach().cpu()],
        **selection.as_dict(),
        "selected_endpoint_paths": list(runtime.selected_endpoint_paths),
        "branches": [branch.__dict__ for branch in runtime.branches],
        "renderer_contract": runtime.renderer_contract,
        "geometry_interpolation": False,
        "geometry_averaging": False,
        "basis_coefficient_interpolation": False,
        "target_forward_leakage": 0,
    }


def controller_forward_argument_names() -> tuple[str, ...]:
    """Explicit boundary record used by governance tests and audits."""
    return ("reference_f2", "reference_valid")


def assert_no_forbidden_forward_names(names: Sequence[str]) -> None:
    overlap = sorted(set(names).intersection(FORBIDDEN_FORWARD_INPUTS))
    if overlap:
        raise RuntimeError(f"CONTROLLER-FORWARD-LEAKAGE: {overlap}")
