from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from scene.p0_candidate_initialization_protocol import (
    DETERMINISTIC_ZERO_INITIALIZATION,
    PAIRED_RANDOM_TRUNK_WITH_ZERO_OUTPUT_HEAD,
    RANDOM_SEEDED_INITIALIZATION,
    CandidateOutput,
    OursV2DeterministicZeroCandidate,
    ProtocolComplexCandidate,
    build_m3_m4_paired_candidates,
    seed_all,
    tensor_mapping_sha256,
)


SEEN_OUTFITS = ("O01", "O02", "O03", "O04", "O08")
FIXED_VIEWS = ("cond_000000", "cond_000318", "cond_000017", "cond_000347")
PREDICTION_FORBIDDEN_INPUTS = (
    "target_rgb",
    "target_mask",
    "target_pose",
    "target_camera",
    "ground_truth_coefficient",
    "teacher_residual",
    "outfit_id",
    "target_loss",
    "predicted_target_render",
)


@dataclass(frozen=True)
class HardLookupOutput:
    logits: torch.Tensor
    predicted_class: int
    predicted_outfit: str


@dataclass(frozen=True)
class NearestCentroidOutput:
    query_feature: torch.Tensor
    standardized_query: torch.Tensor
    squared_distances: Mapping[str, float]
    predicted_outfit: str


def pool_frozen_f2_reference_set(
    reference_f2: torch.Tensor, reference_valid: torch.Tensor
) -> torch.Tensor:
    """Deterministic reference-set mean/max over already frozen F2 rows."""
    if reference_f2.ndim != 2 or reference_f2.shape[1] <= 0:
        raise ValueError("reference_f2 must have shape [R,D]")
    if reference_valid.shape != (reference_f2.shape[0], 1):
        raise ValueError("reference_valid must have shape [R,1]")
    if not torch.isfinite(reference_f2).all() or not torch.isfinite(reference_valid).all():
        raise FloatingPointError("reference feature inputs contain NaN or Inf")
    valid = reference_valid.to(dtype=reference_f2.dtype)
    if valid.sum().item() <= 0:
        raise ValueError("reference set is empty")
    mean = (reference_f2 * valid).sum(0, keepdim=True) / valid.sum().clamp_min(1e-8)
    maximum = reference_f2.masked_fill(
        valid <= 0, torch.finfo(reference_f2.dtype).min
    ).amax(0, keepdim=True)
    return torch.cat((mean, maximum), dim=-1)


def pack_complex_reference_set(
    reference_tokens: torch.Tensor, reference_valid: torch.Tensor
) -> torch.Tensor:
    if reference_tokens.ndim != 2 or reference_tokens.shape[0] != 3:
        raise ValueError("complex reference tokens must have shape [3,D]")
    if reference_valid.shape != (3, 1):
        raise ValueError("complex reference validity must have shape [3,1]")
    if reference_valid.sum().item() <= 0:
        raise ValueError("complex reference set is empty")
    if not torch.isfinite(reference_tokens).all() or not torch.isfinite(reference_valid).all():
        raise FloatingPointError("complex reference inputs contain NaN or Inf")
    return torch.cat((reference_tokens.reshape(-1), reference_valid.reshape(-1)))


class OursV2CandidateAdapter(nn.Module):
    identity = "Ours-v2"
    initialization_policy = DETERMINISTIC_ZERO_INITIALIZATION
    loss_contract = "SMOOTHL1_STANDARDIZED_COEFFICIENT_ONLY"
    initial_output_semantics = "MEAN-GARMENT INITIAL PREDICTION"
    loads_a6_checkpoint = False
    copies_a6_state_dict = False
    pairwise_geometry_weight = 0.0
    outfit_id_in_prediction_forward = False

    def __init__(self, *, input_dim: int = 512) -> None:
        super().__init__()
        self.predictor = OursV2DeterministicZeroCandidate(input_dim=input_dim, rank=4)

    def forward(
        self, reference_f2: torch.Tensor, reference_valid: torch.Tensor
    ) -> CandidateOutput:
        return self.predictor(pool_frozen_f2_reference_set(reference_f2, reference_valid))

    @staticmethod
    def training_loss(
        predicted_standardized: torch.Tensor,
        teacher_standardized: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        coefficient = F.smooth_l1_loss(predicted_standardized, teacher_standardized)
        return {"total": coefficient, "coefficient_smooth_l1": coefficient}


class B6ReferenceClassifierHardLookupAdapter(nn.Module):
    identity = "B6"
    initialization_policy = RANDOM_SEEDED_INITIALIZATION
    loss_contract = "CROSS_ENTROPY_ONLY"
    outfit_label_role = "LOSS_TARGET_ONLY"
    lookup_key = "PREDICTED_CLASS_ONLY"
    outfit_id_in_prediction_forward = False

    def __init__(self, *, seed: int, input_dim: int = 512) -> None:
        super().__init__()
        seed_all(seed)
        self.input_dim = int(input_dim)
        self.normalization = nn.LayerNorm(self.input_dim)
        self.linear = nn.Linear(self.input_dim, len(SEEN_OUTFITS))

    def forward(
        self, reference_f2: torch.Tensor, reference_valid: torch.Tensor
    ) -> HardLookupOutput:
        feature = pool_frozen_f2_reference_set(reference_f2, reference_valid)
        if feature.shape != (1, self.input_dim):
            raise ValueError("B6 pooled feature dimension mismatch")
        logits = self.linear(self.normalization(feature)).reshape(len(SEEN_OUTFITS))
        if not torch.isfinite(logits).all():
            raise FloatingPointError("B6 logits contain NaN or Inf")
        predicted_class = int(torch.argmax(logits).item())
        return HardLookupOutput(
            logits=logits,
            predicted_class=predicted_class,
            predicted_outfit=self.class_index_to_outfit(predicted_class),
        )

    @staticmethod
    def class_index_to_outfit(class_index: int) -> str:
        if isinstance(class_index, bool) or not isinstance(class_index, int):
            raise TypeError("class index must be an integer")
        if class_index < 0 or class_index >= len(SEEN_OUTFITS):
            raise IndexError("predicted class is outside the frozen five-class mapping")
        return SEEN_OUTFITS[class_index]

    @staticmethod
    def training_loss(logits: torch.Tensor, outfit_label: torch.Tensor) -> dict[str, torch.Tensor]:
        if logits.ndim != 2 or logits.shape[1] != len(SEEN_OUTFITS):
            raise ValueError("B6 loss requires logits with shape [N,5]")
        if outfit_label.ndim != 1 or outfit_label.shape[0] != logits.shape[0]:
            raise ValueError("B6 labels must have shape [N]")
        cross_entropy = F.cross_entropy(logits, outfit_label)
        return {"total": cross_entropy, "cross_entropy": cross_entropy}

    @staticmethod
    def lookup_predicted_teacher(
        prediction: HardLookupOutput, teacher_residual_bank: Mapping[str, Any]
    ) -> Any:
        if set(teacher_residual_bank) != set(SEEN_OUTFITS):
            raise ValueError("teacher residual bank must contain exactly five seen outfits")
        return teacher_residual_bank[prediction.predicted_outfit]

    @staticmethod
    def confusion_matrix_schema() -> dict[str, Any]:
        return {
            "schema_version": "canondressgs.paper.p0_b6_confusion_matrix.v1",
            "row_role": "ground_truth_loss_label",
            "column_role": "predicted_class",
            "class_order": list(SEEN_OUTFITS),
            "counts_shape": [5, 5],
        }


class B7F2NearestCentroidHardLookupAdapter(nn.Module):
    identity = "B7"
    trainable = False
    initialization_policy = "FIXED_NON_TRAINABLE"
    optimizer_policy = "NONE"
    distance_metric = "FEATURE_STANDARDIZATION_THEN_SQUARED_L2"
    tie_handling = "REGISTERED_SEEN_OUTFIT_ORDER"
    outfit_id_in_prediction_forward = False

    def __init__(
        self,
        *,
        fold: str,
        target_condition: str,
        reference_condition_ids: Sequence[str],
        feature_mean: torch.Tensor,
        feature_scale: torch.Tensor,
        standardized_centroids: Mapping[str, torch.Tensor],
        construction_manifest: Mapping[str, Any],
    ) -> None:
        super().__init__()
        if fold != target_condition or target_condition not in FIXED_VIEWS:
            raise ValueError("B7 fold/target condition mismatch")
        if tuple(reference_condition_ids) != tuple(
            view for view in FIXED_VIEWS if view != target_condition
        ):
            raise ValueError("B7 fold must use exactly the three non-target views")
        if target_condition in reference_condition_ids:
            raise ValueError("B7-CENTROID-LEAKAGE")
        if set(standardized_centroids) != set(SEEN_OUTFITS):
            raise ValueError("B7 requires five seen-outfit centroids")
        self.fold = fold
        self.target_condition = target_condition
        self.reference_condition_ids = tuple(reference_condition_ids)
        self.register_buffer("feature_mean", feature_mean.detach().clone())
        self.register_buffer("feature_scale", feature_scale.detach().clone())
        for index, outfit in enumerate(SEEN_OUTFITS):
            self.register_buffer(
                f"centroid_{index}", standardized_centroids[outfit].detach().clone()
            )
        self.construction_manifest = dict(construction_manifest)

    def forward(
        self, reference_f2: torch.Tensor, reference_valid: torch.Tensor
    ) -> NearestCentroidOutput:
        query = pool_frozen_f2_reference_set(reference_f2, reference_valid).reshape(-1)
        if query.shape != self.feature_mean.shape:
            raise ValueError("B7 query feature dimension mismatch")
        standardized = (query - self.feature_mean) / self.feature_scale
        distances = {
            outfit: float(
                torch.square(standardized - getattr(self, f"centroid_{index}")).sum()
            )
            for index, outfit in enumerate(SEEN_OUTFITS)
        }
        nearest = min(SEEN_OUTFITS, key=lambda outfit: (distances[outfit], SEEN_OUTFITS.index(outfit)))
        return NearestCentroidOutput(query, standardized, distances, nearest)

    @staticmethod
    def lookup_nearest_teacher(
        prediction: NearestCentroidOutput, teacher_residual_bank: Mapping[str, Any]
    ) -> Any:
        if set(teacher_residual_bank) != set(SEEN_OUTFITS):
            raise ValueError("teacher residual bank must contain exactly five seen outfits")
        return teacher_residual_bank[prediction.predicted_outfit]


def build_b7_fold_adapter(
    *,
    target_condition: str,
    fold_rows: Mapping[str, torch.Tensor],
    fold_validity: Mapping[str, torch.Tensor],
    reference_file_hashes: Mapping[str, Sequence[str]],
) -> tuple[B7F2NearestCentroidHardLookupAdapter, dict[str, Any]]:
    if target_condition not in FIXED_VIEWS:
        raise ValueError("unknown target-view fold")
    if set(fold_rows) != set(SEEN_OUTFITS) or set(fold_validity) != set(SEEN_OUTFITS):
        raise ValueError("B7 centroid inputs must contain exactly five seen outfits")
    if set(reference_file_hashes) != set(SEEN_OUTFITS):
        raise ValueError("B7 reference hash manifest must contain five outfits")
    reference_conditions = tuple(view for view in FIXED_VIEWS if view != target_condition)
    pooled = {
        outfit: pool_frozen_f2_reference_set(
            fold_rows[outfit], fold_validity[outfit]
        ).reshape(-1)
        for outfit in SEEN_OUTFITS
    }
    matrix = torch.stack([pooled[outfit] for outfit in SEEN_OUTFITS])
    feature_mean = matrix.mean(0)
    raw_scale = matrix.std(0, unbiased=False)
    feature_scale = torch.where(raw_scale > 1e-12, raw_scale, torch.ones_like(raw_scale))
    centroids = {
        outfit: (pooled[outfit] - feature_mean) / feature_scale
        for outfit in SEEN_OUTFITS
    }
    manifest = {
        "schema_version": "canondressgs.paper.p0_b7_centroid_construction.v1",
        "fold": target_condition,
        "target_condition": target_condition,
        "reference_condition_ids": list(reference_conditions),
        "target_condition_excluded": target_condition not in reference_conditions,
        "target_rgb_used": False,
        "target_mask_used": False,
        "outfit_label_role": "OFFLINE_CENTROID_GROUPING_ONLY",
        "distance_metric": B7F2NearestCentroidHardLookupAdapter.distance_metric,
        "normalization": {
            "feature_mean_sha256": tensor_mapping_sha256({"mean": feature_mean}),
            "feature_scale_sha256": tensor_mapping_sha256({"scale": feature_scale}),
            "zero_variance_dimensions_use_scale_one": True,
        },
        "outfits": {
            outfit: {
                "reference_condition_ids": list(reference_conditions),
                "reference_file_hashes": list(reference_file_hashes[outfit]),
                "reference_feature_hashes": [
                    tensor_mapping_sha256({"feature": row})
                    for row in fold_rows[outfit]
                ],
                "pooled_feature_sha256": tensor_mapping_sha256({"feature": pooled[outfit]}),
                "centroid_sha256": tensor_mapping_sha256({"centroid": centroids[outfit]}),
            }
            for outfit in SEEN_OUTFITS
        },
        "tie_handling": B7F2NearestCentroidHardLookupAdapter.tie_handling,
    }
    adapter = B7F2NearestCentroidHardLookupAdapter(
        fold=target_condition,
        target_condition=target_condition,
        reference_condition_ids=reference_conditions,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        standardized_centroids=centroids,
        construction_manifest=manifest,
    )
    return adapter, manifest


class P0ComplexCandidateAdapter(nn.Module):
    initialization_policy = PAIRED_RANDOM_TRUNK_WITH_ZERO_OUTPUT_HEAD
    initial_output_semantics = "MEAN-GARMENT INITIAL PREDICTION"
    outfit_id_in_prediction_forward = False

    def __init__(self, candidate: ProtocolComplexCandidate) -> None:
        super().__init__()
        self.identity = candidate.identity
        self.candidate = candidate
        self.loss_contract = (
            "SMOOTHL1_STANDARDIZED_COEFFICIENT_ONLY"
            if self.identity == "M3"
            else "A5_LEGACY_ENDPOINT_SUPERVISION"
        )

    @property
    def trunk(self) -> nn.Module:
        return self.candidate.trunk

    @property
    def output_head(self) -> nn.Linear:
        return self.candidate.output_head

    def forward(
        self, reference_tokens: torch.Tensor, reference_valid: torch.Tensor
    ) -> CandidateOutput:
        return self.candidate(pack_complex_reference_set(reference_tokens, reference_valid))

    def training_loss(
        self,
        predicted_standardized: torch.Tensor,
        teacher_standardized: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.identity == "M3":
            coefficient = F.smooth_l1_loss(predicted_standardized, teacher_standardized)
            return {"total": coefficient, "coefficient_smooth_l1": coefficient}
        return a5_legacy_endpoint_supervision_loss(
            predicted_standardized, teacher_standardized
        )


def build_m3_m4_candidate_adapters(
    *, raw_dim: int, seed: int, device: torch.device | str = "cpu"
) -> tuple[P0ComplexCandidateAdapter, P0ComplexCandidateAdapter]:
    m3, m4 = build_m3_m4_paired_candidates(raw_dim=raw_dim, seed=seed, device=device)
    return P0ComplexCandidateAdapter(m3.module), P0ComplexCandidateAdapter(m4.module)


def a5_legacy_endpoint_supervision_loss(
    prediction: torch.Tensor, target: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Field-for-field parity with formal_batch_runtime._training_loss(A5_*)."""
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("A5 legacy supervision requires matching [N,4] tensors")
    if prediction.shape[0] < 2:
        raise ValueError("A5 absolute-pair supervision requires at least two outfits")
    coefficient = F.smooth_l1_loss(prediction, target)
    sign = F.relu(0.8 - torch.sign(target) * prediction).mean()
    pair_values = []
    for left in range(prediction.shape[0]):
        for right in range(left + 1, prediction.shape[0]):
            pair_values.append(
                F.relu(1.5 - torch.abs(prediction[left] - prediction[right])).mean()
            )
    absolute_pair = torch.stack(pair_values).mean()
    total = coefficient + 0.25 * sign + 0.10 * absolute_pair
    return {
        "total": total,
        "coefficient_loss": coefficient,
        "sign_loss": sign,
        "absolute_pair_loss": absolute_pair,
    }


def candidate_parameter_manifest(module: nn.Module) -> dict[str, Any]:
    names = [name for name, parameter in module.named_parameters() if parameter.requires_grad]
    return {
        "parameter_names": names,
        "parameter_count": sum(
            parameter.numel() for parameter in module.parameters() if parameter.requires_grad
        ),
        "state_sha256": tensor_mapping_sha256(module.state_dict()),
    }


def build_candidate_optimizer(
    module: nn.Module,
) -> tuple[torch.optim.Optimizer | None, dict[str, Any]]:
    parameters = [(name, value) for name, value in module.named_parameters() if value.requires_grad]
    if not parameters:
        return None, {
            "created": False,
            "class": None,
            "groups": [],
            "parameter_count": 0,
            "parameter_names": [],
            "zero_grad_count": 0,
            "step_count": 0,
            "state_saved": False,
        }
    optimizer = torch.optim.Adam(
        [value for _, value in parameters], lr=0.02, weight_decay=0.0
    )
    return optimizer, {
        "created": True,
        "class": "torch.optim.adam.Adam",
        "groups": [
            {
                "name": "candidate_trainable_parameters",
                "learning_rate": 0.02,
                "weight_decay": 0.0,
                "parameter_count": sum(value.numel() for _, value in parameters),
            }
        ],
        "parameter_count": sum(value.numel() for _, value in parameters),
        "parameter_names": [name for name, _ in parameters],
        "zero_grad_count": 0,
        "step_count": 0,
        "state_saved": False,
    }
