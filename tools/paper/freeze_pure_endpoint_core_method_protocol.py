from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
RISK = ROOT / "paper_protocol/reviewer_risk"
DOCS = ROOT / "docs/PAPER"
HANDOFF = ROOT / "project_control_handoff"

TASK_ID = "AAAI27-PURE-ENDPOINT-CORE-METHOD-PROTOCOL-001"
SOURCE_BRANCH = "research/controller-v2-crossfit-micro-pilot-from-repaired-contract-20260723"
SOURCE_HEAD = "a8557b461f301c19a0f24eb17d924ddd9159a580"
RUN_BRANCH = "research/pure-endpoint-core-method-protocol-20260724"
CLASSIFICATION = "PURE_ENDPOINT_CROSSFOLD_PROTOCOL_READY"
NEXT_TASK = "TRAIN_AND_EVALUATE_PURE_ENDPOINT_CORE_METHOD_CROSSFIT"

OUTFITS = ("O01", "O02", "O03", "O04", "O08")
CONDITIONS = ("cond_000000", "cond_000318", "cond_000017", "cond_000347")
FOLDS = (
    ((0, 1), 2, 3),
    ((1, 2), 3, 0),
    ((2, 3), 0, 1),
    ((3, 0), 1, 2),
)
PURE_TYPES = {"AAA", "BBB"}

SOURCE_MANIFEST = RISK / "dual_support_controller_training_manifest.json"
SOURCE_ROTATIONS = RISK / "controller_v2_micro_pilot_rotation_manifests.json"
ASSET_MANIFEST = ROOT / "paper_protocol/frozen_asset_manifest.json"

OUTPUTS = {
    "protocol_doc": DOCS / "AAAI27_PURE_ENDPOINT_CORE_METHOD_PROTOCOL_20260724.md",
    "baseline_doc": DOCS / "AAAI27_PURE_ENDPOINT_BASELINE_REGISTRY_20260724.md",
    "claim_doc": DOCS / "AAAI27_PURE_ENDPOINT_CLAIM_BOUNDARY_20260724.md",
    "protocol": RISK / "pure_endpoint_crossfit_protocol.yaml",
    "rotations": RISK / "pure_endpoint_rotation_manifests.json",
    "model": RISK / "pure_endpoint_model_contract.json",
    "baselines": RISK / "pure_endpoint_baseline_registry.json",
    "evaluator": RISK / "pure_endpoint_evaluator_contract.json",
    "gates": RISK / "pure_endpoint_success_gates.json",
    "summary": RISK / "pure_endpoint_protocol_final_summary.json",
    "handoff": HANDOFF / "pure_endpoint_protocol_handoff.json",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha(path: Path, *, lf: bool = False) -> str:
    data = path.read_bytes()
    if lf:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def canonical_sha(value: Any) -> str:
    data = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value, allow_nan=False)
    return json.dumps(str(value), ensure_ascii=False)


def yaml_lines(value: Any, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, Mapping):
        lines: list[str] = []
        for key, child in value.items():
            rendered_key = str(key)
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", rendered_key):
                rendered_key = json.dumps(rendered_key, ensure_ascii=False)
            if isinstance(child, (Mapping, list, tuple)):
                lines.append(f"{prefix}{rendered_key}:")
                lines.extend(yaml_lines(child, indent + 2))
            else:
                lines.append(f"{prefix}{rendered_key}: {yaml_scalar(child)}")
        return lines
    if isinstance(value, (list, tuple)):
        lines = []
        for child in value:
            if isinstance(child, Mapping):
                items = list(child.items())
                if not items:
                    lines.append(f"{prefix}- {{}}")
                    continue
                key, item = items[0]
                if isinstance(item, (Mapping, list, tuple)):
                    lines.append(f"{prefix}- {key}:")
                    lines.extend(yaml_lines(item, indent + 4))
                else:
                    lines.append(f"{prefix}- {key}: {yaml_scalar(item)}")
                for key, item in items[1:]:
                    if isinstance(item, (Mapping, list, tuple)):
                        lines.append(f"{prefix}  {key}:")
                        lines.extend(yaml_lines(item, indent + 4))
                    else:
                        lines.append(f"{prefix}  {key}: {yaml_scalar(item)}")
            else:
                lines.append(f"{prefix}- {yaml_scalar(child)}")
        return lines
    return [f"{prefix}{yaml_scalar(value)}"]


def write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(yaml_lines(value)) + "\n", encoding="utf-8", newline="\n"
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8", newline="\n")


def evidence(relative: str) -> dict[str, str]:
    return {"path": relative, "sha256_lf": file_sha(ROOT / relative, lf=True)}


def asset_index() -> dict[str, Any]:
    return {row["asset_id"]: row for row in read_json(ASSET_MANIFEST)["assets"]}


def overlap(left: Iterable[str], right: Iterable[str]) -> int:
    return len(set(left) & set(right))


def partition(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    logical_counts = Counter(row["logical_input_sha256"] for row in records)
    multiplicity = Counter(logical_counts.values())
    garment_counts = Counter(row["garment_labels"][0] for row in records)
    assignment_counts = Counter(row["assignment_type"] for row in records)
    images = sorted({item for row in records for item in row["source_image_hashes"]})
    masks = sorted({item for row in records for item in row["source_mask_hashes"]})
    return {
        "record_count": len(records),
        "record_ids": [row["record_id"] for row in records],
        "garment_counts": {name: garment_counts[name] for name in OUTFITS},
        "assignment_type_counts": {
            name: assignment_counts[name] for name in ("AAA", "BBB")
        },
        "assignment_position_counts": {
            "PURE_ENDPOINT_NOT_APPLICABLE": len(records)
        },
        "duplicate_counts": {
            "marked_duplicate_of_formal_pure": sum(
                row["duplicate_of"] is not None for row in records
            ),
            "within_partition_excess_records": len(records) - len(logical_counts),
            "all_duplicate_targets_consistent": all(
                row["duplicate_target_consistent"] for row in records
            ),
        },
        "unique_query_count": len(logical_counts),
        "logical_query_multiplicity_histogram": {
            str(key): value for key, value in sorted(multiplicity.items())
        },
        "logical_input_sha256": sorted(logical_counts),
        "reference_assets": {
            "unique_rgb_sha256_count": len(images),
            "unique_mask_sha256_count": len(masks),
            "unique_combined_sha256_count": len(set(images) | set(masks)),
            "rgb_sha256": images,
            "mask_sha256": masks,
        },
    }


def build_rotations() -> dict[str, Any]:
    source = read_json(SOURCE_MANIFEST)
    inherited = read_json(SOURCE_ROTATIONS)
    by_id = {row["record_id"]: row for row in source["query_sets"]}
    pure_ids = {
        row["record_id"]
        for row in source["query_sets"]
        if row["assignment_type"] in PURE_TYPES
    }
    rotations = []
    for index, (train_indices, cal_index, test_index) in enumerate(FOLDS):
        inherited_rotation = inherited["rotations"][index]
        train_folds = [CONDITIONS[value] for value in train_indices]
        if inherited_rotation["train_folds"] != train_folds:
            raise RuntimeError(f"rotation {index} train folds changed")
        if inherited_rotation["calibration_fold_index"] != cal_index:
            raise RuntimeError(f"rotation {index} calibration fold changed")
        if inherited_rotation["test_fold_index"] != test_index:
            raise RuntimeError(f"rotation {index} test fold changed")

        records: dict[str, list[Mapping[str, Any]]] = {}
        partitions: dict[str, Any] = {}
        for name in ("train", "calibration", "test"):
            ids = [
                record_id
                for record_id in inherited_rotation["partitions"][name]["record_ids"]
                if record_id in pure_ids
            ]
            records[name] = [by_id[record_id] for record_id in ids]
            if any(row["assignment_type"] not in PURE_TYPES for row in records[name]):
                raise RuntimeError("mixed record entered pure protocol")
            partitions[name] = partition(records[name])

        intersections = {}
        for left, right, label in (
            ("train", "calibration", "train_calibration"),
            ("train", "test", "train_test"),
            ("calibration", "test", "calibration_test"),
        ):
            left_images = [
                item for row in records[left] for item in row["source_image_hashes"]
            ]
            right_images = [
                item for row in records[right] for item in row["source_image_hashes"]
            ]
            left_masks = [
                item for row in records[left] for item in row["source_mask_hashes"]
            ]
            right_masks = [
                item for row in records[right] for item in row["source_mask_hashes"]
            ]
            rgb_count = overlap(left_images, right_images)
            mask_count = overlap(left_masks, right_masks)
            intersections[label] = {
                "record_id": overlap(
                    [row["record_id"] for row in records[left]],
                    [row["record_id"] for row in records[right]],
                ),
                "logical_input_sha256": overlap(
                    [row["logical_input_sha256"] for row in records[left]],
                    [row["logical_input_sha256"] for row in records[right]],
                ),
                "reference_rgb_sha256": rgb_count,
                "reference_mask_sha256": mask_count,
                "reference_combined_sha256": rgb_count + mask_count,
            }

        value = {
            "rotation": index,
            "train_fold_indices": list(train_indices),
            "train_folds": train_folds,
            "calibration_fold_index": cal_index,
            "calibration_fold": CONDITIONS[cal_index],
            "test_fold_index": test_index,
            "test_fold": CONDITIONS[test_index],
            "partitions": partitions,
            "intersections": intersections,
            "logical_disjointness_pass": all(
                row["record_id"] == 0 and row["logical_input_sha256"] == 0
                for row in intersections.values()
            ),
            "asset_overlap_interpretation": (
                "EXPECTED_CLOSED_WARDROBE_REFERENCE_ASSET_OVERLAP; "
                "condition-fold records are logically disjoint while target-excluded "
                "reference sets reuse closed-wardrobe assets"
            ),
        }
        value["rotation_manifest_sha256"] = canonical_sha(value)
        rotations.append(value)

    formal = source["formal_pure_endpoint_episodes"]
    result = {
        "schema_version": "canondressgs.paper.pure_endpoint_rotation_manifests.v1",
        "task_id": TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "source_manifest": str(SOURCE_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "source_manifest_sha256_lf": file_sha(SOURCE_MANIFEST, lf=True),
        "source_rotation_manifest": str(SOURCE_ROTATIONS.relative_to(ROOT)).replace(
            "\\", "/"
        ),
        "source_rotation_manifest_sha256_lf": file_sha(SOURCE_ROTATIONS, lf=True),
        "identity": "subject02",
        "outfit_order": list(OUTFITS),
        "condition_order": list(CONDITIONS),
        "record_policy": {
            "allowed_assignment_types": ["AAA", "BBB"],
            "forbidden_assignment_types": ["AAB", "ABB"],
            "pair_labels_used": False,
            "mixture_alpha_used": False,
            "compatibility_labels_used": False,
            "protocol_records_preserved": True,
            "duplicate_records_receive_extra_training_weight": False,
            "training_batches_use_one_unique_query_per_garment": True,
        },
        "rotations": rotations,
        "formal_pure_secondary_safety_set": {
            "record_count": len(formal),
            "unique_query_count": len(
                {row["logical_input_sha256"] for row in formal}
            ),
            "record_ids": [row["record_id"] for row in formal],
            "training_use": 0,
            "calibration_use": 0,
            "primary_crossfit_denominator_use": 0,
            "role": "SEPARATE_SECONDARY_SAFETY_AUDIT_ONLY",
        },
        "paper_final": False,
        "paper_final_count": 0,
    }
    result["archive_content_sha256"] = canonical_sha(result)
    return result


def build_model() -> dict[str, Any]:
    assets = asset_index()
    teachers = {
        outfit: {
            "path": assets[f"teacher_checkpoint_{outfit}"]["path"],
            "sha256": assets[f"teacher_checkpoint_{outfit}"]["fingerprint"],
        }
        for outfit in OUTFITS
    }
    return {
        "schema_version": "canondressgs.paper.pure_endpoint_model_contract.v1",
        "task_id": TASK_ID,
        "status": "UNAMBIGUOUS",
        "classification_on_missing_field": "PURE_ENDPOINT_PROTOCOL_BLOCKED_MODEL_CONTRACT",
        "paper_name": "CanonDressGS-Endpoint",
        "historical_internal_name": "Ours-v2",
        "identity": "subject02",
        "pipeline": [
            "reference RGB and clothing mask",
            "frozen F2",
            "per-reference clothing-weighted mean and clothing-masked max",
            "deterministic validity-aware reference-set mean and max",
            "LayerNorm(512)",
            "Linear(512,4)",
            "coefficient de-normalization",
            "nearest frozen endpoint in standardized coefficient space",
            "rank-4 explicit canonical residual basis",
            "frozen MMLP-Human deformation and renderer",
        ],
        "prediction_boundary": {
            "allowed_inputs": [
                "reference_rgb",
                "reference_clothing_mask",
                "reference_foreground_mask",
                "reference_validity",
            ],
            "forbidden_inputs": [
                "target_rgb",
                "target_mask",
                "target_pose",
                "target_camera",
                "ground_truth_outfit_id",
                "ground_truth_coefficient",
                "teacher_residual",
                "pair_label",
                "mixture_alpha",
                "compatibility_label",
                "AAB",
                "ABB",
            ],
            "target_pose_camera_downstream_renderer_only": True,
            "empty_reference_guard": (
                "valid count zero emits ABSTAIN_EMPTY_REFERENCE and Base Avatar; "
                "no garment endpoint is selected"
            ),
        },
        "frozen_f2": {
            "per_reference_output_dimension": 256,
            "reference_count_normal": 3,
            "per_reference_contract": (
                "clothing-weighted mean concatenated with clothing-masked max"
            ),
            "set_aggregation": (
                "validity-aware mean concatenated with validity-aware max"
            ),
            "predictor_input_dimension": 512,
            "backbone_checkpoint": assets["backbone_checkpoint"]["path"],
            "backbone_checkpoint_sha256": assets["backbone_checkpoint"]["fingerprint"],
            "backbone_state_fingerprint": assets["backbone_state"]["fingerprint"],
            "trainable": False,
        },
        "predictor": {
            "input_dimension": 512,
            "output_dimension": 4,
            "architecture": (
                "LayerNorm(512) followed by Linear(512,4); no output activation"
            ),
            "parameter_count": 3076,
            "parameter_breakdown": {
                "layer_norm_weight_bias": 1024,
                "linear_weight_bias": 2052,
            },
            "initialization": {
                "layer_norm_weight": 1.0,
                "layer_norm_bias": 0.0,
                "linear_weight": 0.0,
                "linear_bias": 0.0,
                "semantics": "mean-garment standardized coefficient at step zero",
            },
        },
        "coefficient_target": {
            "raw_target": (
                "teacher_coefficients[outfit] in selected_basis.pt; one rank-4 "
                "canonical endpoint coefficient per closed-wardrobe garment"
            ),
            "standardized_target": "y_g = (c_g - mean) / std",
            "mean": [
                -1.220703143189894e-05,
                -3.051757857974735e-06,
                2.441406286379788e-05,
                -6.103515625e-05,
            ],
            "std": [
                412.4079895019531,
                364.3363037109375,
                313.2566223144531,
                297.9817810058594,
            ],
            "normalization_asset": assets["coefficient_normalization"]["path"],
            "normalization_asset_sha256": assets["coefficient_normalization"][
                "fingerprint"
            ],
            "normalization_scope": (
                "five closed-wardrobe endpoint coefficients; every rotation train "
                "partition contains all five garments"
            ),
            "test_condition_statistics_used": False,
        },
        "loss": {
            "name": "coefficient SmoothL1 only",
            "domain": "standardized rank-4 coefficients",
            "definition": (
                "mean smooth_l1(predicted-target,beta=1); 0.5*d^2 for |d|<1, "
                "|d|-0.5 otherwise"
            ),
            "reduction": "mean over batch and four dimensions",
            "pairwise_geometry_weight": 0.0,
            "classification_loss": 0.0,
            "render_loss": 0.0,
        },
        "endpoint_selection": {
            "candidate_order": list(OUTFITS),
            "distance": "squared L2 in standardized rank-4 coefficient space",
            "rule": (
                "argmin_g ||predicted_standardized-target_standardized[g]||_2^2"
            ),
            "tie_break": "first garment in frozen candidate_order",
            "selected_residual": (
                "basis(selected raw teacher coefficient); continuous prediction is "
                "not rendered by CanonDressGS-Endpoint"
            ),
            "linear_coefficient_predictor_baseline": (
                "renders the de-normalized continuous prediction before snapping"
            ),
        },
        "basis": {
            "schema": "canondressgs.multi_outfit_explicit_basis_artifact.v1",
            "rank": 4,
            "method": "deterministic centered SVD",
            "asset": assets["rank4_explicit_basis"]["path"],
            "sha256": assets["rank4_explicit_basis"]["fingerprint"],
            "explained_variance": 0.9999999999978457,
            "trainable": False,
            "channel_bounds": {
                "xyz": 0.05,
                "log_scaling": 0.35,
                "rotation": 0.2617993878,
                "opacity_logit": 2.0,
                "sh0": 0.25,
                "shN": 0.10,
            },
        },
        "training_plan": {
            "runs": "4 rotations x 3 deterministic protocol replicates",
            "replicate_indices": [0, 1, 2],
            "seeds": [0, 1, 2],
            "initialization_is_seed_independent_by_contract": True,
            "optimizer": {
                "class": "torch.optim.Adam",
                "learning_rate": 0.02,
                "weight_decay": 0.0,
                "betas": [0.9, 0.999],
                "epsilon": 1e-08,
                "amsgrad": False,
            },
            "scheduler": {
                "class": "torch.optim.lr_scheduler.LambdaLR",
                "multiplier": 1.0,
            },
            "steps": 300,
            "batch_size": 5,
            "batch_rule": (
                "one unique pure query per garment in frozen garment order; "
                "pair-container duplicates add no weight"
            ),
            "garment_order": list(OUTFITS),
            "condition_order_per_rotation": (
                "registered two train folds, round-robin as "
                "train_folds[(step-1) mod 2]"
            ),
            "data_order_identical_across_replicates": True,
            "gradient_clip_norm": 5.0,
            "early_stopping": False,
            "best_seed_selection": False,
            "threshold_selection": False,
        },
        "checkpoint_rule": {
            "milestone_steps": [0, 20, 50, 100, 200, 300],
            "writes_per_run": 6,
            "evaluate_checkpoint": 300,
            "best_checkpoint_selection": False,
            "payload": [
                "model state",
                "optimizer state",
                "scheduler state",
                "RNG state",
                "global step",
                "condition position",
                "frozen asset fingerprints",
            ],
            "exact_resume_required": True,
        },
        "frozen_runtime": {
            "teacher_endpoint_bank": teachers,
            "mmlp_human_checkpoint": assets["mmlphuman_checkpoint"]["path"],
            "mmlp_human_checkpoint_sha256": assets["mmlphuman_checkpoint"][
                "fingerprint"
            ],
            "base_gaussian_state_fingerprint": assets["base_gaussian_state"][
                "fingerprint"
            ],
            "renderer_source_bundle_sha256": assets["renderer_source_bundle"][
                "fingerprint"
            ],
        },
        "provenance": [
            evidence("scene/p0_candidate_initialization_protocol.py"),
            evidence("scene/p0_candidate_adapters.py"),
            evidence("tools/paper/formal_batch_runtime.py"),
            evidence(
                "paper_protocol/reviewer_risk/p0_formal_candidate_run_registry.yaml"
            ),
            evidence(
                "paper_protocol/reviewer_risk/p0_formal_candidate_final_summary.json"
            ),
            evidence("paper_protocol/frozen_asset_manifest.json"),
            evidence("tools/run_multi_outfit_explicit_basis.py"),
        ],
        "execution_in_this_task": {
            "training": 0,
            "forward_training_batches": 0,
            "backward": 0,
            "optimizer_created": 0,
            "checkpoint_writes": 0,
            "formal_renderer_runs": 0,
        },
        "paper_final": False,
        "paper_final_count": 0,
    }


def build_baselines(model: Mapping[str, Any]) -> dict[str, Any]:
    assets = asset_index()
    methods = [
        {
            "index": 1,
            "paper_name": "Base Avatar",
            "role": "no-garment-residual lower reference",
            "deployable": True,
            "input": "none",
            "output": "frozen base Gaussian state without garment residual",
            "trainable": False,
            "provenance": "paper registry B0 and frozen base_gaussian_state",
        },
        {
            "index": 2,
            "paper_name": "Teacher Upper Bound",
            "role": "non-deployable full teacher endpoint upper bound",
            "deployable": False,
            "input": "ground-truth outfit ID for evaluation-only retrieval",
            "output": "full frozen teacher residual, not rank-4 reconstruction",
            "trainable": False,
            "provenance": "teacher_checkpoint_O01/O02/O03/O04/O08",
        },
        {
            "index": 3,
            "paper_name": "Outfit-ID Oracle",
            "role": "non-deployable rank-4 endpoint oracle",
            "deployable": False,
            "input": "ground-truth outfit ID",
            "output": "rank-4 endpoint coefficient lookup and frozen basis",
            "trainable": False,
            "provenance": "selected_basis.pt teacher_coefficients mapping",
        },
        {
            "index": 4,
            "paper_name": "Reference Classifier Lookup",
            "role": "reference-controlled hard lookup baseline",
            "deployable": True,
            "input": "same pooled frozen-F2 512-vector",
            "output": "five logits; argmax selects rank-4 endpoint",
            "architecture": "LayerNorm(512) followed by Linear(512,5)",
            "parameter_count": 3589,
            "loss": "cross entropy only",
            "training": "same train folds, 300 steps, Adam lr=0.02",
            "provenance": "formal P0 B6 adapter and results",
        },
        {
            "index": 5,
            "paper_name": "Nearest-Centroid Lookup",
            "role": "fixed non-trainable reference lookup baseline",
            "deployable": True,
            "input": "same pooled frozen-F2 512-vector",
            "output": "nearest train-only standardized garment centroid endpoint",
            "distance": "squared L2",
            "tie_break": "frozen garment order",
            "trainable": False,
            "crossfit_rule": (
                "feature mean, scale, and five centroids use only rotation train "
                "unique queries"
            ),
            "provenance": "formal P0 B7 adapter and centroid construction",
        },
        {
            "index": 6,
            "paper_name": "Direct Residual Decoder",
            "role": "historical direct canonical-residual decoder comparator",
            "deployable": True,
            "input": "reference RGB/mask only",
            "output": "canonical Gaussian residual without endpoint lookup",
            "historical_source": (
                "scene/support_conditioned_dual_branch_residual_decoder_v7.py"
            ),
            "historical_source_sha256_lf": file_sha(
                ROOT
                / "scene/support_conditioned_dual_branch_residual_decoder_v7.py",
                lf=True,
            ),
            "historical_registry_id": "PAPER-A8-V7",
            "crossfit_status": "MATCHED_CROSSFIT_RERUN_REQUIRED",
            "historical_metrics_reusable": False,
            "provenance": "A8 Residual Decoder V7 historical diagnostic",
        },
        {
            "index": 7,
            "paper_name": "Linear Coefficient Predictor",
            "role": "continuous explicit-basis coefficient baseline",
            "deployable": True,
            "input": "reference RGB/mask only",
            "output": "de-normalized predicted rank-4 coefficient rendered directly",
            "architecture": model["predictor"]["architecture"],
            "parameter_count": model["predictor"]["parameter_count"],
            "training": "same model and training contract as primary method",
            "endpoint_snapping": False,
            "provenance": "formal Ours-v2 continuous prediction path",
        },
        {
            "index": 8,
            "paper_name": "CanonDressGS-Endpoint",
            "role": "primary pure endpoint method",
            "deployable": True,
            "input": "reference RGB/mask only",
            "output": "discrete nearest rank-4 garment endpoint",
            "architecture": model["predictor"]["architecture"],
            "parameter_count": model["predictor"]["parameter_count"],
            "endpoint_snapping": True,
            "provenance": (
                "formal Ours-v2 predictor plus frozen nearest-endpoint rule"
            ),
        },
    ]
    return {
        "schema_version": "canondressgs.paper.pure_endpoint_baseline_registry.v1",
        "task_id": TASK_ID,
        "status": "COMPLETE",
        "classification_on_missing_provenance": (
            "PURE_ENDPOINT_PROTOCOL_BLOCKED_BASELINE_PROVENANCE"
        ),
        "shared_contract": {
            "identity": "subject02",
            "garments": list(OUTFITS),
            "primary_split": "same pure endpoint condition-fold cross-fit",
            "historical_metrics_may_replace_crossfit_results": False,
        },
        "methods": methods,
        "teacher_assets": {
            outfit: assets[f"teacher_checkpoint_{outfit}"]["fingerprint"]
            for outfit in OUTFITS
        },
        "paper_naming_guards": {
            "forbidden_names": ["Ours-v2", "B6", "B7", "M3", "M4"],
            "teacher_upper_bound_is_deployable": False,
            "outfit_id_oracle_is_reference_controlled": False,
            "canondressgs_endpoint_may_use_outfit_id": False,
        },
        "provenance": [
            evidence("paper_protocol/experiment_registry.yaml"),
            evidence("configs/paper/aaai27_seen_outfit_explicit_basis_v1.yaml"),
            evidence("scene/p0_candidate_adapters.py"),
            evidence(
                "paper_protocol/reviewer_risk/p0_formal_candidate_final_summary.json"
            ),
        ],
        "paper_final": False,
        "paper_final_count": 0,
    }


def build_evaluator() -> dict[str, Any]:
    return {
        "schema_version": "canondressgs.paper.pure_endpoint_evaluator_contract.v1",
        "task_id": TASK_ID,
        "status": "FROZEN_BEFORE_TRAINING",
        "primary_denominator": {
            "records": "test-partition pure AAA/BBB protocol records only",
            "per_rotation_protocol_weighted_count": 20,
            "per_rotation_unique_query_count": 5,
            "global_protocol_weighted_count": 80,
            "global_unique_query_count": 20,
            "formal_pure_20_merged": False,
            "calibration_merged": False,
            "duplicate_policy": (
                "report protocol-weighted and unique-query metrics; duplicates never "
                "add training weight"
            ),
        },
        "identification": {
            "class_order": list(OUTFITS),
            "top1": "selected endpoint equals true garment",
            "per_rotation": True,
            "per_garment_recall": "TP_g divided by true unique test queries for g",
            "confusion_matrix": "rows true garment, columns selected endpoint",
            "macro_5way_top1": "unweighted mean of five per-garment recalls",
            "protocol_weighted": "mean over 20 protocol records per rotation",
            "unique_query": (
                "mean after grouping identical logical_input_sha256 values"
            ),
        },
        "coefficient": {
            "coefficient_mae": (
                "continuous raw rank-4 prediction MAE versus true endpoint coefficient"
            ),
            "coefficient_rmse": (
                "continuous raw rank-4 prediction RMSE versus true endpoint coefficient"
            ),
            "endpoint_coefficient_error": (
                "L2 selected raw endpoint versus true raw endpoint"
            ),
            "basis_reconstruction_error": (
                "bound-normalized six-channel residual RMSE versus full Teacher "
                "Upper Bound residual"
            ),
            "also_report_standardized": True,
        },
        "render": {
            "reference": "Teacher Upper Bound at identical pose and camera",
            "rgb_mae": "mean absolute RGB error inside target garment mask",
            "lpips": {
                "network": "VGG LPIPS v0.1",
                "crop": (
                    "garment bbox plus 5 percent, aspect-preserving resize and zero "
                    "pad to 256x256"
                ),
                "mask_threshold": 0.5,
            },
            "silhouette_iou": (
                "binary alpha>=0.5 versus target garment mask>=0.5"
            ),
            "boundary_fscore": {
                "boundary": "3x3 binary erosion XOR mask",
                "tolerance_pixels": (
                    "max(1,floor(0.005*min(height,width)+0.5))"
                ),
            },
            "protected_lpips": (
                "LPIPS inside target_protected_mask against Base Avatar render"
            ),
            "identity_metric": (
                "protected-mask mean absolute RGB difference against Base Avatar"
            ),
            "endpoint_parity": (
                "ground-truth class replacement must make CanonDressGS-Endpoint and "
                "Outfit-ID Oracle select identical coefficients and produce exact "
                "residual/render parity"
            ),
            "severe_artifact_grade": {
                "scale": [0, 1, 2, 3],
                "severe": 3,
                "categories": [
                    "wrong_outfit_endpoint",
                    "cloud_or_mottle",
                    "edge_scatter",
                    "silhouette_discontinuity",
                    "identity_contamination",
                    "component_contamination",
                    "empty_render",
                ],
                "review_policy": (
                    "review all unique primary test queries without cherry-picking"
                ),
            },
        },
        "safety": {
            "identity_contamination": (
                "unique test query count with identity grade greater than zero"
            ),
            "wrong_outfit_endpoint": "selected endpoint differs from true garment",
            "severe_wrong_outfit_endpoint": (
                "wrong endpoint and wrong-outfit grade equals 3"
            ),
            "empty_render": (
                "no foreground alpha>=0.5 or empty-render grade greater than zero"
            ),
            "component_contamination": (
                "component-contamination grade greater than zero"
            ),
        },
        "efficiency": {
            "predictor_parameters": "all trainable predictor parameters",
            "forward_time": (
                "batch-one aggregation plus predictor and endpoint selection"
            ),
            "active_gaussian_count": (
                "Gaussians passing the frozen renderer active test"
            ),
            "render_time": "frozen deformation plus renderer after selection",
            "peak_vram": "torch.cuda.max_memory_allocated after reset",
            "timing_protocol": (
                "20 warm-up then 100 synchronized iterations; median and IQR"
            ),
        },
        "perturbations": {
            "mild_blur": {
                "kernel": 11,
                "sigma": 3.0,
                "padding": "reflect",
                "scope": "RGB inside clothing mask; restore outside pixels",
            },
            "mask_erosion": {
                "operation": "scipy.ndimage.binary_erosion",
                "iterations": 3,
                "structure": "SciPy default cross connectivity 1",
            },
            "mask_dilation": {
                "operation": "scipy.ndimage.binary_dilation",
                "iterations": 3,
                "structure": "SciPy default cross connectivity 1",
            },
            "assignment_permutation": {
                "normal_slots": [0, 1, 2],
                "permuted_slots": [2, 0, 1],
            },
            "single_reference": {
                "retained_slots": [0],
                "report": (
                    "full identification, coefficient, render, and safety metrics"
                ),
            },
            "reference_dropout": {
                "drop_one_retained_slots": [0, 1],
                "complete_dropout": (
                    "all validity zero; require ABSTAIN_EMPTY_REFERENCE plus Base "
                    "Avatar, not garment recovery"
                ),
            },
            "required_reports": [
                "garment prediction stability",
                "endpoint selection stability",
                "coefficient drift",
                "render drift",
                "safe behavior",
            ],
            "mixed_controller_routing_metrics_used": False,
        },
        "aggregation": {
            "report_every_rotation": True,
            "report_every_replicate": True,
            "deterministic_replicates_are_not_independent_seed_samples": True,
            "best_seed_selection": False,
            "threshold_selection": False,
            "test_fold_used_for_calibration": False,
        },
        "provenance": [
            evidence(
                "tools/paper/run_p0_color_spatial_soft_control_evaluations.py"
            ),
            evidence("tools/paper/p0_evaluation_protocol.py"),
            evidence("tools/paper/formal_batch_runtime.py"),
            evidence(
                "tools/paper/run_reference_conditioned_dual_support_controller_formal_evaluation.py"
            ),
        ],
        "formal_rendering_in_this_task": 0,
        "paper_final": False,
        "paper_final_count": 0,
    }


def build_gates() -> dict[str, Any]:
    return {
        "schema_version": "canondressgs.paper.pure_endpoint_success_gates.v1",
        "task_id": TASK_ID,
        "status": "PREREGISTERED_BEFORE_TRAINING",
        "primary_method": "CanonDressGS-Endpoint",
        "primary_gate": {
            "macro_5way_top1_min": 0.90,
            "every_rotation_top1_min": 0.80,
            "every_garment_recall_min": 0.75,
            "endpoint_parity": "PASS",
            "identity_contamination_count": 0,
            "severe_wrong_outfit_endpoint_rate_max": 0.05,
            "all_clauses_required": True,
        },
        "information_ablation": {
            "single_reference_full_report_required": True,
            "single_reference_gate_required": False,
            "complete_dropout_required_behavior": "SAFE_ABSTAIN_AND_BASE_AVATAR",
            "complete_dropout_garment_recovery_required": False,
        },
        "decision_rules": {
            "PURE_ENDPOINT_CORE_METHOD_SUPPORTED": (
                "all primary gate clauses pass on frozen crossfit results"
            ),
            "REFERENCE_CONTROL_NOT_SUPPORTED": (
                "reference identification fails while Outfit-ID Oracle endpoint "
                "parity and rendering remain valid"
            ),
            "PURE_ENDPOINT_CORE_METHOD_PARTIAL": (
                "at least one but not all rotation clauses pass"
            ),
        },
        "post_result_gate_changes_allowed": False,
        "threshold_selection_allowed": False,
        "paper_final": False,
        "paper_final_count": 0,
    }


def build_protocol() -> dict[str, Any]:
    return {
        "schema_version": "canondressgs.paper.pure_endpoint_crossfit_protocol.v1",
        "task_id": TASK_ID,
        "status": CLASSIFICATION,
        "source": {"branch": SOURCE_BRANCH, "head": SOURCE_HEAD},
        "run": {
            "branch": RUN_BRANCH,
            "windows_worktree": (
                "E:/model_train/canondressgs_pure_endpoint_core_method_protocol"
            ),
            "cloud_worktree": (
                "/root/autodl-tmp/canondressgs_work/worktrees/"
                "canondressgs_pure_endpoint_core_method_protocol"
            ),
        },
        "scientific_name": "PURE-ENDPOINT CONDITION-FOLD CROSS-FIT",
        "paper_method_name": "CanonDressGS-Endpoint",
        "identity": "subject02",
        "closed_wardrobe": list(OUTFITS),
        "input": "pure garment reference RGB/mask set",
        "output": (
            "discrete canonical garment endpoint and frozen deformation/render"
        ),
        "excluded": [
            "AAB",
            "ABB",
            "garment pair prediction",
            "continuous mixture weight",
            "Dual-Support",
            "HARD_GEOMETRY_SOFT_VA",
            "compatibility routing",
        ],
        "rotations": [
            {
                "rotation": index,
                "train": list(train),
                "calibration": calibration,
                "test": test,
            }
            for index, (train, calibration, test) in enumerate(FOLDS)
        ],
        "contracts": {
            "rotation_manifests": (
                "paper_protocol/reviewer_risk/pure_endpoint_rotation_manifests.json"
            ),
            "model": (
                "paper_protocol/reviewer_risk/pure_endpoint_model_contract.json"
            ),
            "baselines": (
                "paper_protocol/reviewer_risk/pure_endpoint_baseline_registry.json"
            ),
            "evaluator": (
                "paper_protocol/reviewer_risk/pure_endpoint_evaluator_contract.json"
            ),
            "success_gates": (
                "paper_protocol/reviewer_risk/pure_endpoint_success_gates.json"
            ),
        },
        "execution_boundary": {
            "training": 0,
            "forward_training_batches": 0,
            "backward": 0,
            "optimizer_created": 0,
            "checkpoint_writes": 0,
            "formal_renderer_runs": 0,
            "threshold_selection": 0,
            "formal_upstream_asset_mutation": 0,
            "paper_final": False,
            "paper_final_count": 0,
        },
        "next_task": NEXT_TASK,
        "next_task_started": False,
    }


def docs(
    rotations: Mapping[str, Any],
    baselines: Mapping[str, Any],
) -> dict[str, str]:
    rotation_rows = []
    for row in rotations["rotations"]:
        p = row["partitions"]
        o = row["intersections"]
        rotation_rows.append(
            "| {r} | {tr} | {cal} | {test} | {rc} | {uq} | {ao} |".format(
                r=row["rotation"],
                tr=",".join(str(value) for value in row["train_fold_indices"]),
                cal=row["calibration_fold_index"],
                test=row["test_fold_index"],
                rc="/".join(
                    str(p[name]["record_count"])
                    for name in ("train", "calibration", "test")
                ),
                uq="/".join(
                    str(p[name]["unique_query_count"])
                    for name in ("train", "calibration", "test")
                ),
                ao="/".join(
                    str(o[name]["reference_rgb_sha256"])
                    for name in (
                        "train_calibration",
                        "train_test",
                        "calibration_test",
                    )
                ),
            )
        )
    baseline_rows = [
        "| {index} | {paper_name} | {role} | {deploy} | {provenance} |".format(
            deploy="yes" if row["deployable"] else "no", **row
        )
        for row in baselines["methods"]
    ]
    protocol = f"""# AAAI-27 Pure Endpoint Core Method Protocol

Task: {TASK_ID}
Scientific protocol: PURE-ENDPOINT CONDITION-FOLD CROSS-FIT
Paper method: CanonDressGS-Endpoint
Classification: {CLASSIFICATION}

## Scope

This protocol fixes subject02 and the closed wardrobe O01/O02/O03/O04/O08. A
query contains only pure-garment reference RGB and clothing masks. The method
selects a discrete canonical garment endpoint, then uses frozen MMLP-Human
deformation and rendering. AAB, ABB, pair labels, mixture alpha, compatibility
labels, Dual-Support, HARD_GEOMETRY_SOFT_VA, and compatibility routing are out
of scope.

This is condition-fold cross-fit. It is not unseen-garment, unseen-identity,
strict novel-view, or unseen-reference generalization. Exact reference assets
overlap because target-excluded sets reuse the same closed wardrobe.

## Frozen Source

- Branch: {SOURCE_BRANCH}
- HEAD: {SOURCE_HEAD}
- Source records: paper_protocol/reviewer_risk/dual_support_controller_training_manifest.json
- Source rotations: paper_protocol/reviewer_risk/controller_v2_micro_pilot_rotation_manifests.json
- Historical Controller failures remain immutable.

## Cross-Fit Records

| Rotation | Train folds | Calibration | Test | Records T/C/V | Unique T/C/V | RGB overlap T-C/T-V/C-V |
|---:|---|---:|---:|---:|---:|---:|
{chr(10).join(rotation_rows)}

Exact record IDs are in pure_endpoint_rotation_manifests.json. Each condition
has five unique pure queries, one per garment. Historical pair containers
repeat each query four times. Protocol-weighted and unique-query metrics are
both reported; training uses each logical query once. Record IDs and logical
hashes are disjoint. RGB/mask overlap is disclosed, not hidden.

The retained formal_pure 20-record set is a separate safety audit. It never
enters training, calibration, the primary denominator, or the global macro.

## Model Contract

The recovered contract is:

reference RGB/mask -> frozen F2 -> deterministic mean/max -> LayerNorm(512)
-> Linear(4) -> nearest frozen endpoint -> rank-4 basis -> frozen renderer.

- Input/output dimensions: 512 to 4.
- Parameters: 3076.
- Initialization: LayerNorm 1/0; Linear 0/0.
- Target: standardized rank-4 teacher endpoint coefficient.
- Loss: mean SmoothL1 beta 1, coefficient only.
- Optimizer plan: Adam, lr 0.02, no weight decay, fixed LR.
- Schedule: 300 steps, batch 5, fixed garment order.
- Data order: two train folds in registered order, round-robin.
- Checkpoints: 0/20/50/100/200/300; evaluate step 300 only.
- No early stopping, checkpoint selection, seed selection, or thresholds.

Linear Coefficient Predictor renders the continuous prediction.
CanonDressGS-Endpoint snaps to a frozen endpoint before rendering.

## Gates

- macro five-way top-1 >= 0.90;
- every rotation >= 0.80;
- every garment recall >= 0.75;
- endpoint parity PASS;
- identity contamination count = 0;
- severe wrong-outfit endpoint rate <= 0.05.

Single-reference behavior is fully reported. Complete reference dropout must
safely abstain and use Base Avatar; garment recovery is not required.

## Execution Boundary

This freeze performs no training, training forward batch, backward, optimizer
creation, checkpoint write, formal render, threshold selection, or upstream
asset mutation. PAPER_FINAL=false. The next task is {NEXT_TASK} and was not
started.
"""
    baseline = f"""# AAAI-27 Pure Endpoint Baseline Registry

Task: {TASK_ID}

| # | Formal name | Role | Deployable | Frozen provenance |
|---:|---|---|:---:|---|
{chr(10).join(baseline_rows)}

## Separation Rules

Teacher Upper Bound retrieves the full teacher residual and is not deployable.
Outfit-ID Oracle uses ground-truth outfit ID to retrieve the rank-4 endpoint
and is not reference-controlled. Reference Classifier Lookup,
Nearest-Centroid Lookup, Linear Coefficient Predictor, and
CanonDressGS-Endpoint receive reference RGB/masks only.

Direct Residual Decoder is bound to historical V7 source, but historical
metrics remain diagnostic. It requires a matched cross-fit rerun. No
historical denominator may replace this protocol.

Ours-v2, B6, B7, M3, and M4 are internal provenance labels and are forbidden
as formal paper method names.
"""
    claim = f"""# AAAI-27 Pure Endpoint Claim Boundary

Task: {TASK_ID}

## Allowed Claim

Only after every preregistered gate passes may the paper claim
reference-controlled selection among five closed-wardrobe pure garment
endpoints for fixed subject02 under condition-fold cross-fit. The formal
method name is CanonDressGS-Endpoint.

## Forbidden Claims

- unseen garment or arbitrary garment generation;
- unseen identity or cross-identity generalization;
- strict novel-view or novel-pose generalization;
- unseen-reference generalization without a separate asset-level audit;
- mixed AAB/ABB control, pair prediction, mixture weights, Dual-Support, or
  compatibility routing;
- merging the formal-pure safety set into the primary denominator;
- treating deterministic replicates as independent random seeds;
- changing thresholds after results.

Reference assets overlap by design. The defensible term is PURE-ENDPOINT
CONDITION-FOLD CROSS-FIT.

## Future Decision

- All primary clauses pass: PURE_ENDPOINT_CORE_METHOD_SUPPORTED.
- Identification fails while Outfit-ID Oracle remains valid:
  REFERENCE_CONTROL_NOT_SUPPORTED.
- Only some rotations pass: PURE_ENDPOINT_CORE_METHOD_PARTIAL.

This freeze makes no performance claim. PAPER_FINAL=false, historical
Controller failures remain unchanged, and no training was started.
"""
    return {"protocol_doc": protocol, "baseline_doc": baseline, "claim_doc": claim}


def validate(
    rotations: Mapping[str, Any],
    model: Mapping[str, Any],
    baselines: Mapping[str, Any],
    evaluator: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> list[str]:
    if len(rotations["rotations"]) != 4:
        raise RuntimeError("rotation count changed")
    expected = {
        "train": (40, 10, 8),
        "calibration": (20, 5, 4),
        "test": (20, 5, 4),
    }
    for rotation in rotations["rotations"]:
        for name, (record_count, unique_count, garment_count) in expected.items():
            value = rotation["partitions"][name]
            if value["record_count"] != record_count:
                raise RuntimeError(f"{name} record count changed")
            if value["unique_query_count"] != unique_count:
                raise RuntimeError(f"{name} unique count changed")
            if set(value["assignment_type_counts"]) != PURE_TYPES:
                raise RuntimeError("mixed assignment entered protocol")
            if any(value["garment_counts"][g] != garment_count for g in OUTFITS):
                raise RuntimeError(f"{name} garment balance changed")
        if not rotation["logical_disjointness_pass"]:
            raise RuntimeError("logical split overlap")
        expected_assets = {
            "train_calibration": 15,
            "train_test": 15,
            "calibration_test": 10,
        }
        for name, count in expected_assets.items():
            value = rotation["intersections"][name]
            if value["reference_rgb_sha256"] != count:
                raise RuntimeError("RGB overlap changed")
            if value["reference_mask_sha256"] != count:
                raise RuntimeError("mask overlap changed")
    if (
        model["predictor"]["input_dimension"],
        model["predictor"]["output_dimension"],
        model["predictor"]["parameter_count"],
    ) != (512, 4, 3076):
        raise RuntimeError("predictor contract changed")
    expected_names = [
        "Base Avatar",
        "Teacher Upper Bound",
        "Outfit-ID Oracle",
        "Reference Classifier Lookup",
        "Nearest-Centroid Lookup",
        "Direct Residual Decoder",
        "Linear Coefficient Predictor",
        "CanonDressGS-Endpoint",
    ]
    if [row["paper_name"] for row in baselines["methods"]] != expected_names:
        raise RuntimeError("baseline order changed")
    if evaluator["primary_denominator"]["formal_pure_20_merged"]:
        raise RuntimeError("formal pure set entered primary denominator")
    if gates["post_result_gate_changes_allowed"]:
        raise RuntimeError("success gates are mutable")
    return [
        "four rotations",
        "pure counts and garment balance",
        "record and logical disjointness",
        "reference asset overlap",
        "model contract complete",
        "baseline names and provenance",
        "evaluator denominator separation",
        "success gates frozen",
    ]


def main() -> None:
    rotations = build_rotations()
    model = build_model()
    baselines = build_baselines(model)
    evaluator = build_evaluator()
    gates = build_gates()
    checks = validate(rotations, model, baselines, evaluator, gates)
    protocol = build_protocol()
    rendered_docs = docs(rotations, baselines)

    execution = {
        "training": 0,
        "forward_training_batches": 0,
        "backward": 0,
        "optimizer_created": 0,
        "checkpoint_writes": 0,
        "formal_renderer_runs": 0,
        "threshold_selection": 0,
    }
    frozen_sources = {
        relative: file_sha(ROOT / relative, lf=True)
        for relative in (
            "paper_protocol/reviewer_risk/dual_support_controller_training_manifest.json",
            "paper_protocol/reviewer_risk/controller_v2_micro_pilot_rotation_manifests.json",
            "paper_protocol/frozen_asset_manifest.json",
            "scene/p0_candidate_initialization_protocol.py",
            "scene/p0_candidate_adapters.py",
            "tools/paper/formal_batch_runtime.py",
            "tools/run_multi_outfit_explicit_basis.py",
            "paper_protocol/reviewer_risk/p0_formal_candidate_final_summary.json",
        )
    }
    summary = {
        "schema_version": "canondressgs.paper.pure_endpoint_protocol_final_summary.v1",
        "task_id": TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "run_branch": RUN_BRANCH,
        "classification": CLASSIFICATION,
        "classification_reason": (
            "pure manifests, formal Ours-v2 model fields, baseline provenance, "
            "evaluator denominators, perturbations, and success gates are complete"
        ),
        "pure_task": {
            "identity": "subject02",
            "garments": list(OUTFITS),
            "scientific_name": "PURE-ENDPOINT CONDITION-FOLD CROSS-FIT",
            "paper_method": "CanonDressGS-Endpoint",
        },
        "per_rotation_counts": {
            str(row["rotation"]): {
                name: {
                    "records": row["partitions"][name]["record_count"],
                    "unique_queries": row["partitions"][name][
                        "unique_query_count"
                    ],
                }
                for name in ("train", "calibration", "test")
            }
            for row in rotations["rotations"]
        },
        "logical_overlap_all_zero": True,
        "reference_asset_overlap_per_rotation": {
            "rgb_train_calibration": 15,
            "rgb_train_test": 15,
            "rgb_calibration_test": 10,
            "mask_train_calibration": 15,
            "mask_train_test": 15,
            "mask_calibration_test": 10,
        },
        "model_contract_status": model["status"],
        "baseline_registry_status": baselines["status"],
        "evaluator_status": evaluator["status"],
        "success_gate_status": gates["status"],
        "execution_counts": execution,
        "frozen_upstream_mutation_count": 0,
        "frozen_upstream_sha256_lf": frozen_sources,
        "generation_checks": {"status": "PASS", "checks": checks},
        "reports": {
            key: str(path.relative_to(ROOT)).replace("\\", "/")
            for key, path in OUTPUTS.items()
        },
        "paper_final": False,
        "paper_final_count": 0,
        "next_task": NEXT_TASK,
        "next_task_started": False,
    }
    handoff = {
        "schema_version": "canondressgs.project_control.pure_endpoint_protocol_handoff.v1",
        "task_id": TASK_ID,
        "status": "COMPLETE",
        "classification": CLASSIFICATION,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "run_branch": RUN_BRANCH,
        "paper_method": "CanonDressGS-Endpoint",
        "protocol": (
            "paper_protocol/reviewer_risk/pure_endpoint_crossfit_protocol.yaml"
        ),
        "final_summary": (
            "paper_protocol/reviewer_risk/pure_endpoint_protocol_final_summary.json"
        ),
        "execution_counts": execution,
        "frozen_upstream_mutation_count": 0,
        "paper_final": False,
        "paper_final_count": 0,
        "next_task": NEXT_TASK,
        "next_task_started": False,
        "stop_rule": (
            "Do not train automatically; NEXT_TASK requires explicit authorization"
        ),
    }

    write_json(OUTPUTS["rotations"], rotations)
    write_json(OUTPUTS["model"], model)
    write_json(OUTPUTS["baselines"], baselines)
    write_json(OUTPUTS["evaluator"], evaluator)
    write_json(OUTPUTS["gates"], gates)
    write_yaml(OUTPUTS["protocol"], protocol)
    for key, text in rendered_docs.items():
        write_text(OUTPUTS[key], text)
    write_json(OUTPUTS["summary"], summary)
    write_json(OUTPUTS["handoff"], handoff)
    print(json.dumps({"status": "PASS", "outputs": len(OUTPUTS), "checks": checks}))


if __name__ == "__main__":
    main()
