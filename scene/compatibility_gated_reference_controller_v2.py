"""Compatibility-gated, tri-mode reference controller V2 design adapter.

This module is a no-training design artifact.  It separates garment identity,
binary mixedness, and pair-conditioned mixture weight while keeping all target
and ground-truth query information outside the prediction boundary.
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
from scene.reference_conditioned_dual_support_controller import (
    OUTFIT_ORDER,
    soft_target_cross_entropy,
    spatial_f2_rows,
)


PAIR_ORDER = tuple(
    f"{first}_{second}"
    for index, first in enumerate(OUTFIT_ORDER)
    for second in OUTFIT_ORDER[index + 1 :]
)
PAIR_TO_INDEX = {pair: index for index, pair in enumerate(PAIR_ORDER)}
MODES = ("SINGLE_ENDPOINT", "DUAL_SUPPORT", "HARD_GEOMETRY_SOFT_VA")
PAIR_CONFIDENCE_DEFINITION = "TOP2_VS_TOP3_PROBABILITY_MARGIN"
INFORMATION_SUFFICIENT_MINIMUM = 2
FORBIDDEN_FORWARD_INPUTS = (
    "ground_truth_garment_id",
    "ground_truth_pair",
    "ground_truth_composition",
    "ground_truth_alpha",
    "target_rgb",
    "target_mask",
    "target_pose",
    "target_camera",
    "teacher_residual",
    "target_render",
    "visual_artifact_label",
)


@dataclass(frozen=True)
class ControllerV2Output:
    reference_feature: torch.Tensor
    garment_logits: torch.Tensor
    garment_probabilities: torch.Tensor
    mixedness_logit: torch.Tensor
    mixedness_probability: torch.Tensor
    pair_weight_logits: torch.Tensor
    all_pair_weights: torch.Tensor
    valid_reference_count: int


@dataclass(frozen=True)
class StablePairPrediction:
    predicted_top1: str
    predicted_top2: str
    predicted_pair: str
    top1_probability: float
    top2_probability: float
    top3_probability: float
    pair_confidence: float
    stable_rank_indices: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "predicted_top1": self.predicted_top1,
            "predicted_top2": self.predicted_top2,
            "predicted_pair": self.predicted_pair,
            "top1_probability": self.top1_probability,
            "top2_probability": self.top2_probability,
            "top3_probability": self.top3_probability,
            "pair_confidence": self.pair_confidence,
            "pair_confidence_definition": PAIR_CONFIDENCE_DEFINITION,
            "stable_rank_indices": list(self.stable_rank_indices),
            "tie_handling": "FROZEN_OUTFIT_ORDER",
        }


@dataclass(frozen=True)
class RoutingThresholds:
    mixedness: float
    pair_confidence: float
    provenance: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.mixedness <= 1.0:
            raise ValueError("mixedness threshold must be in [0,1]")
        if not 0.0 <= self.pair_confidence <= 1.0:
            raise ValueError("pair-confidence threshold must be in [0,1]")
        if not self.provenance:
            raise ValueError("threshold provenance is required")


@dataclass(frozen=True)
class CompatibilityEntry:
    pair_id: str
    compatibility_score: float
    compatibility_label: str
    manifest_sha256: str
    calibration_condition: str

    def __post_init__(self) -> None:
        if self.pair_id not in PAIR_TO_INDEX:
            raise ValueError(f"unknown pair: {self.pair_id}")
        if self.compatibility_label not in {"COMPATIBLE", "INCOMPATIBLE"}:
            raise ValueError("compatibility label must be rule generated")
        if not 0.0 <= self.compatibility_score <= 1.0:
            raise ValueError("compatibility score must be in [0,1]")


class CompatibilityPrior:
    """Closed-wardrobe prior with complete fixed-order pair coverage."""

    explicit_pair_blacklist = False
    lookup_key = "PREDICTED_PAIR"
    query_ground_truth_used = False

    def __init__(self, entries: Sequence[CompatibilityEntry]) -> None:
        if tuple(entry.pair_id for entry in entries) != PAIR_ORDER:
            raise ValueError("compatibility entries must cover all ten pairs in frozen order")
        self._entries = {entry.pair_id: entry for entry in entries}

    def lookup(self, predicted_pair: str) -> CompatibilityEntry:
        if predicted_pair not in self._entries:
            raise KeyError(predicted_pair)
        return self._entries[predicted_pair]

    def as_dict(self) -> dict[str, Any]:
        return {
            "pair_order": list(PAIR_ORDER),
            "lookup_key": self.lookup_key,
            "query_ground_truth_used": self.query_ground_truth_used,
            "explicit_pair_blacklist": self.explicit_pair_blacklist,
            "entries": [self._entries[pair].__dict__ for pair in PAIR_ORDER],
        }


@dataclass(frozen=True)
class ControllerV2Decision:
    prediction: StablePairPrediction
    mixedness_probability: float
    all_pair_weights: tuple[float, ...]
    selected_pair_weight_a: float
    selected_pair_weight_b: float
    compatibility_score: float
    compatibility_label: str
    mode: str
    fallback_reason: str | None
    dominant_outfit: str
    valid_reference_count: int
    target_forward_leakage: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.prediction.as_dict(),
            "mixedness_probability": self.mixedness_probability,
            "all_pair_weights": list(self.all_pair_weights),
            "selected_pair_weight_a": self.selected_pair_weight_a,
            "selected_pair_weight_b": self.selected_pair_weight_b,
            "compatibility_score": self.compatibility_score,
            "compatibility_label": self.compatibility_label,
            "mode": self.mode,
            "fallback_reason": self.fallback_reason,
            "dominant_outfit": self.dominant_outfit,
            "valid_reference_count": self.valid_reference_count,
            "target_forward_leakage": self.target_forward_leakage,
        }


@dataclass(frozen=True)
class GeometrySupport:
    support_index: int
    outfit_id: str
    endpoint_path: str
    opacity_weight: float
    geometry_policy: str = "IMMUTABLE_ENDPOINT_NO_INTERPOLATION"


@dataclass(frozen=True)
class VisibilityAppearanceSource:
    source_index: int
    outfit_id: str
    endpoint_path: str
    mixture_weight: float


@dataclass(frozen=True)
class TriModeRuntimeRequest:
    mode: str
    geometry_supports: tuple[GeometrySupport, ...]
    visibility_appearance_sources: tuple[VisibilityAppearanceSource, ...]
    selected_endpoint_paths: tuple[str, ...]
    fallback_reason: str | None
    renderer_contract: str
    geometry_interpolation: bool = False
    geometry_averaging: bool = False
    basis_coefficient_interpolation: bool = False
    target_forward_leakage: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "geometry_supports": [support.__dict__ for support in self.geometry_supports],
            "visibility_appearance_sources": [
                source.__dict__ for source in self.visibility_appearance_sources
            ],
            "selected_endpoint_paths": list(self.selected_endpoint_paths),
            "fallback_reason": self.fallback_reason,
            "renderer_contract": self.renderer_contract,
            "geometry_interpolation": self.geometry_interpolation,
            "geometry_averaging": self.geometry_averaging,
            "basis_coefficient_interpolation": self.basis_coefficient_interpolation,
            "target_forward_leakage": self.target_forward_leakage,
        }


class CompatibilityGatedReferenceControllerV2(nn.Module):
    """Three independent linear heads over one normalized reference feature."""

    initialization_policy = RANDOM_SEEDED_INITIALIZATION
    outfit_order = OUTFIT_ORDER
    pair_order = PAIR_ORDER
    loads_v1_checkpoint = False
    loads_b6_checkpoint = False
    loads_ours_v2_checkpoint = False
    target_pose_camera_in_prediction_forward = False
    ground_truth_pair_in_prediction_forward = False

    def __init__(self, *, seed: int, input_dim: int = 512) -> None:
        super().__init__()
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        seed_all(seed)
        self.seed = seed
        self.input_dim = int(input_dim)
        self.reference_normalization = nn.LayerNorm(self.input_dim)
        self.garment_head = nn.Linear(self.input_dim, len(OUTFIT_ORDER))
        self.mixedness_head = nn.Linear(self.input_dim, 1)
        self.pair_weight_head = nn.Linear(self.input_dim, len(PAIR_ORDER))

    def aggregate_reference_representation(
        self, reference_f2: torch.Tensor, reference_valid: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        if reference_f2.ndim != 2 or reference_f2.shape[1] != self.input_dim:
            raise ValueError("reference_f2 must have shape [R,input_dim]")
        if reference_valid.shape != (reference_f2.shape[0], 1):
            raise ValueError("reference_valid must have shape [R,1]")
        if not torch.isfinite(reference_f2).all() or not torch.isfinite(reference_valid).all():
            raise FloatingPointError("reference inputs contain NaN or Inf")
        valid_count = int((reference_valid.reshape(-1) > 0).sum().item())
        if valid_count == 0:
            pooled = reference_f2.new_zeros((1, self.input_dim))
        else:
            pooled = pool_frozen_f2_reference_set(reference_f2, reference_valid)
        normalized = self.reference_normalization(pooled)
        feature = F.normalize(normalized, p=2, dim=-1, eps=1.0e-8)
        return feature, valid_count

    def forward(
        self, reference_f2: torch.Tensor, reference_valid: torch.Tensor
    ) -> ControllerV2Output:
        feature, valid_count = self.aggregate_reference_representation(
            reference_f2, reference_valid
        )
        garment_logits = self.garment_head(feature).reshape(len(OUTFIT_ORDER))
        mixedness_logit = self.mixedness_head(feature).reshape(())
        pair_weight_logits = self.pair_weight_head(feature).reshape(len(PAIR_ORDER))
        garment_probabilities = torch.softmax(garment_logits, dim=0)
        mixedness_probability = torch.sigmoid(mixedness_logit)
        all_pair_weights = torch.sigmoid(pair_weight_logits)
        values = (
            garment_logits,
            mixedness_logit.reshape(1),
            pair_weight_logits,
            garment_probabilities,
            mixedness_probability.reshape(1),
            all_pair_weights,
        )
        if not all(torch.isfinite(value).all() for value in values):
            raise FloatingPointError("Controller V2 output contains NaN or Inf")
        return ControllerV2Output(
            reference_feature=feature.reshape(self.input_dim),
            garment_logits=garment_logits,
            garment_probabilities=garment_probabilities,
            mixedness_logit=mixedness_logit,
            mixedness_probability=mixedness_probability,
            pair_weight_logits=pair_weight_logits,
            all_pair_weights=all_pair_weights,
            valid_reference_count=valid_count,
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def head_parameter_counts(self) -> dict[str, int]:
        return {
            "reference_normalization": sum(
                parameter.numel() for parameter in self.reference_normalization.parameters()
            ),
            "garment_head": sum(
                parameter.numel() for parameter in self.garment_head.parameters()
            ),
            "mixedness_head": sum(
                parameter.numel() for parameter in self.mixedness_head.parameters()
            ),
            "pair_weight_head": sum(
                parameter.numel() for parameter in self.pair_weight_head.parameters()
            ),
        }


def stable_pair_prediction(probabilities: torch.Tensor) -> StablePairPrediction:
    if probabilities.shape != (len(OUTFIT_ORDER),):
        raise ValueError("garment probabilities must have shape [5]")
    if not torch.isfinite(probabilities).all() or torch.any(probabilities < 0):
        raise ValueError("garment probabilities must be finite and non-negative")
    if not torch.allclose(
        probabilities.sum(), probabilities.new_tensor(1.0), atol=1.0e-6, rtol=0
    ):
        raise ValueError("garment probabilities must sum to one")
    values = [float(value) for value in probabilities.detach().cpu()]
    ranking = tuple(sorted(range(len(values)), key=lambda index: (-values[index], index)))
    first, second, third = ranking[:3]
    pair_indices = tuple(sorted((first, second)))
    pair = f"{OUTFIT_ORDER[pair_indices[0]]}_{OUTFIT_ORDER[pair_indices[1]]}"
    return StablePairPrediction(
        predicted_top1=OUTFIT_ORDER[first],
        predicted_top2=OUTFIT_ORDER[second],
        predicted_pair=pair,
        top1_probability=values[first],
        top2_probability=values[second],
        top3_probability=values[third],
        pair_confidence=values[second] - values[third],
        stable_rank_indices=ranking,
    )


def route_controller_v2(
    output: ControllerV2Output,
    compatibility_prior: CompatibilityPrior,
    thresholds: RoutingThresholds,
) -> ControllerV2Decision:
    prediction = stable_pair_prediction(output.garment_probabilities)
    entry = compatibility_prior.lookup(prediction.predicted_pair)
    pair_weight_a = float(
        output.all_pair_weights[PAIR_TO_INDEX[prediction.predicted_pair]].detach().cpu()
    )
    pair_weight_b = 1.0 - pair_weight_a
    mixedness = float(output.mixedness_probability.detach().cpu())
    if output.valid_reference_count < INFORMATION_SUFFICIENT_MINIMUM:
        mode = "SINGLE_ENDPOINT"
        reason = "REFERENCE_INFORMATION_INSUFFICIENT"
    elif mixedness < thresholds.mixedness:
        mode = "SINGLE_ENDPOINT"
        reason = "LOW_MIXEDNESS"
    elif prediction.pair_confidence < thresholds.pair_confidence:
        mode = "SINGLE_ENDPOINT"
        reason = "LOW_PAIR_CONFIDENCE"
    elif entry.compatibility_label == "COMPATIBLE":
        mode = "DUAL_SUPPORT"
        reason = None
    else:
        mode = "HARD_GEOMETRY_SOFT_VA"
        reason = "PAIR_INCOMPATIBLE"
    return ControllerV2Decision(
        prediction=prediction,
        mixedness_probability=mixedness,
        all_pair_weights=tuple(
            float(value) for value in output.all_pair_weights.detach().cpu()
        ),
        selected_pair_weight_a=pair_weight_a,
        selected_pair_weight_b=pair_weight_b,
        compatibility_score=entry.compatibility_score,
        compatibility_label=entry.compatibility_label,
        mode=mode,
        fallback_reason=reason,
        dominant_outfit=prediction.predicted_top1,
        valid_reference_count=output.valid_reference_count,
    )


def construct_tri_mode_runtime(
    decision: ControllerV2Decision, endpoint_bank: Mapping[str, str]
) -> TriModeRuntimeRequest:
    if tuple(endpoint_bank.keys()) != OUTFIT_ORDER:
        raise ValueError("endpoint bank must preserve frozen outfit order")
    if any(not isinstance(path, str) or not path for path in endpoint_bank.values()):
        raise ValueError("endpoint paths must be non-empty strings")
    first, second = decision.prediction.predicted_pair.split("_")
    pair_weights = (
        (first, decision.selected_pair_weight_a),
        (second, decision.selected_pair_weight_b),
    )
    appearance = tuple(
        VisibilityAppearanceSource(
            source_index=index,
            outfit_id=outfit,
            endpoint_path=endpoint_bank[outfit],
            mixture_weight=weight,
        )
        for index, (outfit, weight) in enumerate(pair_weights, 1)
    )
    if decision.mode == "SINGLE_ENDPOINT":
        outfit = decision.prediction.predicted_top1
        geometry = (
            GeometrySupport(
                support_index=1,
                outfit_id=outfit,
                endpoint_path=endpoint_bank[outfit],
                opacity_weight=1.0,
            ),
        )
        appearance = (
            VisibilityAppearanceSource(
                source_index=1,
                outfit_id=outfit,
                endpoint_path=endpoint_bank[outfit],
                mixture_weight=1.0,
            ),
        )
        renderer = "SEALED_SINGLE_ENDPOINT"
    elif decision.mode == "DUAL_SUPPORT":
        geometry = tuple(
            GeometrySupport(
                support_index=index,
                outfit_id=outfit,
                endpoint_path=endpoint_bank[outfit],
                opacity_weight=weight,
            )
            for index, (outfit, weight) in enumerate(pair_weights, 1)
        )
        renderer = "SEALED_DUAL_SUPPORT_OPACITY_GATE"
    elif decision.mode == "HARD_GEOMETRY_SOFT_VA":
        outfit = decision.dominant_outfit
        geometry = (
            GeometrySupport(
                support_index=1,
                outfit_id=outfit,
                endpoint_path=endpoint_bank[outfit],
                opacity_weight=1.0,
            ),
        )
        renderer = "SEALED_HARD_GEOMETRY_SOFT_VA"
    else:
        raise ValueError(f"unsupported Controller V2 mode: {decision.mode}")
    if abs(sum(item.opacity_weight for item in geometry) - 1.0) > 1.0e-7:
        raise RuntimeError("geometry opacity weights are not normalized")
    if abs(sum(item.mixture_weight for item in appearance) - 1.0) > 1.0e-7:
        raise RuntimeError("visibility/appearance weights are not normalized")
    selected = tuple(dict.fromkeys(item.endpoint_path for item in geometry + appearance))
    return TriModeRuntimeRequest(
        mode=decision.mode,
        geometry_supports=geometry,
        visibility_appearance_sources=appearance,
        selected_endpoint_paths=selected,
        fallback_reason=decision.fallback_reason,
        renderer_contract=renderer,
    )


def pair_weight_smooth_l1(
    pair_weight_logits: torch.Tensor,
    *,
    ground_truth_pair_index: int,
    target_weight_a: torch.Tensor,
) -> torch.Tensor:
    """Training-only loss: supervise one GT pair scalar after all ten exist."""
    if pair_weight_logits.shape[-1] != len(PAIR_ORDER):
        raise ValueError("pair-weight logits must end in dimension 10")
    if not 0 <= ground_truth_pair_index < len(PAIR_ORDER):
        raise ValueError("ground-truth pair index is outside frozen pair order")
    predicted = torch.sigmoid(pair_weight_logits[..., ground_truth_pair_index])
    return F.smooth_l1_loss(predicted, target_weight_a)


def mixedness_binary_cross_entropy(
    mixedness_logit: torch.Tensor, mixedness_target: torch.Tensor
) -> torch.Tensor:
    if mixedness_logit.shape != mixedness_target.shape:
        raise ValueError("mixedness logit and target shapes must match")
    return F.binary_cross_entropy_with_logits(mixedness_logit, mixedness_target)


def controller_v2_loss_schema() -> dict[str, Any]:
    return {
        "total": "L_garment + lambda_mix*L_mixedness + lambda_weight*L_weight + lambda_cons*L_consistency",
        "garment": {
            "loss": soft_target_cross_entropy.__name__,
            "pure_target": "ONE_HOT",
            "mixed_target": "TWO_THIRDS_PLUS_ONE_THIRD",
        },
        "mixedness": {
            "loss": mixedness_binary_cross_entropy.__name__,
            "pure_target": 0,
            "mixed_target": 1,
        },
        "weight": {
            "loss": pair_weight_smooth_l1.__name__,
            "records": "MIXED_ONLY",
            "supervised_scalar": "GROUND_TRUTH_PAIR_TRAINING_ONLY",
            "forward_output_count": len(PAIR_ORDER),
        },
        "consistency": {
            "eligible": [
                "mild_blur",
                "mild_mask_erosion",
                "mild_mask_dilation",
                "assignment_permutation",
            ],
            "excluded_information_ablations": [
                "complete_reference_dropout",
                "single_reference",
            ],
        },
    }


def controller_v2_forward_argument_names() -> tuple[str, ...]:
    return ("reference_f2", "reference_valid")


def assert_no_forbidden_forward_names(names: Sequence[str]) -> None:
    overlap = sorted(set(names).intersection(FORBIDDEN_FORWARD_INPUTS))
    if overlap:
        raise RuntimeError(f"CONTROLLER-V2-FORWARD-LEAKAGE: {overlap}")


def inference_result_schema(
    output: ControllerV2Output,
    decision: ControllerV2Decision,
    runtime: TriModeRuntimeRequest,
) -> dict[str, Any]:
    return {
        "garment_logits": [
            float(value) for value in output.garment_logits.detach().cpu()
        ],
        "garment_probabilities": [
            float(value) for value in output.garment_probabilities.detach().cpu()
        ],
        "mixedness_logit": float(output.mixedness_logit.detach().cpu()),
        "mixedness_probability": float(
            output.mixedness_probability.detach().cpu()
        ),
        "pair_weight_logits": [
            float(value) for value in output.pair_weight_logits.detach().cpu()
        ],
        **decision.as_dict(),
        "selected_endpoint_paths": list(runtime.selected_endpoint_paths),
        "runtime": runtime.as_dict(),
        "target_forward_leakage": 0,
    }


__all__ = [
    "CompatibilityEntry",
    "CompatibilityGatedReferenceControllerV2",
    "CompatibilityPrior",
    "ControllerV2Decision",
    "ControllerV2Output",
    "FORBIDDEN_FORWARD_INPUTS",
    "GeometrySupport",
    "INFORMATION_SUFFICIENT_MINIMUM",
    "MODES",
    "OUTFIT_ORDER",
    "PAIR_CONFIDENCE_DEFINITION",
    "PAIR_ORDER",
    "PAIR_TO_INDEX",
    "RoutingThresholds",
    "StablePairPrediction",
    "TriModeRuntimeRequest",
    "VisibilityAppearanceSource",
    "assert_no_forbidden_forward_names",
    "construct_tri_mode_runtime",
    "controller_v2_forward_argument_names",
    "controller_v2_loss_schema",
    "inference_result_schema",
    "mixedness_binary_cross_entropy",
    "pair_weight_smooth_l1",
    "route_controller_v2",
    "soft_target_cross_entropy",
    "spatial_f2_rows",
    "stable_pair_prediction",
]
