"""No-training design smoke for compatibility-gated Controller V2.

This program only parses frozen archives, constructs cross-fit compatibility
manifests, initializes a fresh random controller, and exercises tensor/routing
interfaces.  It never builds an optimizer, runs a renderer, evaluates images,
or writes a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene.compatibility_gated_reference_controller_v2 import (  # noqa: E402
    MODES,
    OUTFIT_ORDER,
    PAIR_CONFIDENCE_DEFINITION,
    PAIR_ORDER,
    CompatibilityEntry,
    CompatibilityGatedReferenceControllerV2,
    CompatibilityPrior,
    ControllerV2Output,
    RoutingThresholds,
    assert_no_forbidden_forward_names,
    construct_tri_mode_runtime,
    controller_v2_forward_argument_names,
    controller_v2_loss_schema,
    inference_result_schema,
    route_controller_v2,
    stable_pair_prediction,
)


TASK_ID = "AAAI27-COMPATIBILITY-GATED-CONTROLLER-V2-DESIGN-001"
SOURCE_HEAD = "25318857b4cc4ee11011cd852c5cfc237a8b3cee"
SOURCE_BRANCH = "research/controller-calibration-compatibility-diagnostic-20260723"
RUN_BRANCH = "research/compatibility-gated-controller-v2-design-20260723"
CONDITION_ORDER = (
    "cond_000000",
    "cond_000318",
    "cond_000017",
    "cond_000347",
)
ROTATIONS = (
    {"rotation": 0, "train_folds": [0, 1], "calibration_fold": 2, "test_fold": 3},
    {"rotation": 1, "train_folds": [1, 2], "calibration_fold": 3, "test_fold": 0},
    {"rotation": 2, "train_folds": [2, 3], "calibration_fold": 0, "test_fold": 1},
    {"rotation": 3, "train_folds": [3, 0], "calibration_fold": 1, "test_fold": 2},
)
MIXEDNESS_THRESHOLD_GRID = tuple(round(index / 10, 1) for index in range(1, 10))
PAIR_CONFIDENCE_THRESHOLD_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
CORE_ARTIFACTS = (
    "patch",
    "cloud",
    "mottle",
    "edge_scatter",
    "silhouette_discontinuity",
    "full_body_contamination",
)
SEVERE_ZERO_ARTIFACTS = (
    "patch",
    "cloud",
    "mottle",
    "full_body_contamination",
)
FROZEN_REPOSITORY_INPUTS = (
    "paper_protocol/reviewer_risk/dual_support_all_pair_visual_review.json",
    "paper_protocol/reviewer_risk/dual_support_all_pair_results.json",
    "paper_protocol/reviewer_risk/controller_pair_compatibility_analysis.json",
    "paper_protocol/reviewer_risk/controller_calibration_diagnostic_final_summary.json",
    "paper_protocol/reviewer_risk/dual_support_controller_final_summary.json",
    "paper_protocol/reviewer_risk/controller_v2_design_protocol.yaml",
    "paper_protocol/reviewer_risk/controller_v2_crossfit_splits.json",
)
ZERO_EXECUTION_COUNTS = {
    "training_steps": 0,
    "training_forward_batches": 0,
    "backward_calls": 0,
    "optimizer_creations": 0,
    "optimizer_steps": 0,
    "scheduler_steps": 0,
    "checkpoint_loads": 0,
    "checkpoint_writes": 0,
    "inference_runs": 0,
    "evaluation_runs": 0,
    "renderer_calls": 0,
    "metric_aggregations": 0,
    "formal_renders": 0,
    "formal_metrics": 0,
    "formal_visual_reviews": 0,
    "paper_final": 0,
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path: Path, value: Any) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"append-only collision: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def mean(values: Iterable[float]) -> float:
    rows = list(values)
    if not rows:
        raise ValueError("cannot average an empty descriptor set")
    return float(sum(rows) / len(rows))


def frozen_fingerprint(repo_root: Path) -> dict[str, Any]:
    records = []
    for relative in FROZEN_REPOSITORY_INPUTS:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return {
        "record_count": len(records),
        "records": records,
        "aggregate_sha256": canonical_sha256(records),
    }


def parameter_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            digest.update(name.encode("utf-8"))
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_init_probe(args: argparse.Namespace) -> None:
    if args.seed not in (0, 1, 2):
        raise ValueError("seed must be one of 0, 1, 2")
    if not args.probe_id:
        raise ValueError("--probe-id is required for init-probe")
    torch.set_num_threads(1)
    model = CompatibilityGatedReferenceControllerV2(seed=args.seed)
    if model.parameter_count != 9232:
        raise RuntimeError(f"unexpected parameter count: {model.parameter_count}")
    result = {
        "schema_version": "canondressgs.research.controller_v2_init_probe.v1",
        "task_id": TASK_ID,
        "status": "PASS",
        "fresh_process_pid": os.getpid(),
        "seed": args.seed,
        "probe_id": args.probe_id,
        "initialization": "RANDOM_SEEDED_FRESH_PROCESS",
        "parameter_count": model.parameter_count,
        "head_parameter_counts": model.head_parameter_counts,
        "parameter_sha256": parameter_sha256(model),
        "loads_v1_checkpoint": False,
        "loads_b6_checkpoint": False,
        "loads_ours_v2_checkpoint": False,
        "frozen_input_fingerprint": frozen_fingerprint(args.repo_root),
        "execution_counts": ZERO_EXECUTION_COUNTS,
    }
    write_once(
        args.output_root / "audits" / f"init_seed_{args.seed}_{args.probe_id}.json",
        result,
    )


def endpoint_parity_by_pair(results: Mapping[str, Any]) -> dict[str, bool]:
    values = {pair: True for pair in PAIR_ORDER}
    seen = {pair: 0 for pair in PAIR_ORDER}
    for row in results["endpoint_parity"]["records"]:
        if row["variant"] != "DUAL_SUPPORT_GEOMETRY_BLEND":
            continue
        pair = row["pair_id"]
        values[pair] = values[pair] and bool(row["pass"])
        seen[pair] += 1
    if any(seen[pair] == 0 for pair in PAIR_ORDER):
        raise RuntimeError("endpoint-parity archive is incomplete")
    return values


def calibration_records(
    results: Mapping[str, Any], pair: str, condition: str
) -> list[Mapping[str, Any]]:
    rows = [
        row
        for row in results["execution"]["records"]
        if row["pair_id"] == pair
        and row["condition"] == condition
        and row["variant"] == "DUAL_SUPPORT_GEOMETRY_BLEND"
    ]
    if len(rows) != 6:
        raise RuntimeError(
            f"expected six calibration Oracle records for {pair}/{condition}, got {len(rows)}"
        )
    return rows


def calibration_visual_grades(
    visual: Mapping[str, Any], pair: str, condition: str
) -> list[Mapping[str, int]]:
    grades = [
        grade_row["grades"]
        for item in visual["items"]
        if item.get("kind") == "MAIN_SHEET" and item.get("pair_id") == pair
        for grade_row in item["grades_by_variant_alpha_view"]
        if grade_row["variant"] == "DUAL_SUPPORT_GEOMETRY_BLEND"
        and grade_row["condition"] == condition
    ]
    if len(grades) != 6:
        raise RuntimeError(
            f"expected six calibration visual records for {pair}/{condition}, got {len(grades)}"
        )
    return grades


def build_manifest(
    rotation: Mapping[str, Any],
    *,
    results: Mapping[str, Any],
    visual: Mapping[str, Any],
    endpoint_parity: Mapping[str, bool],
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    calibration_fold = int(rotation["calibration_fold"])
    test_fold = int(rotation["test_fold"])
    calibration_condition = CONDITION_ORDER[calibration_fold]
    test_condition = CONDITION_ORDER[test_fold]
    entries = []
    for pair in PAIR_ORDER:
        metric_rows = calibration_records(results, pair, calibration_condition)
        grade_rows = calibration_visual_grades(visual, pair, calibration_condition)
        maximum_core_grade = max(
            int(grades[field]) for grades in grade_rows for field in CORE_ARTIFACTS
        )
        severe_core_count = sum(
            int(grades[field]) == 3
            for grades in grade_rows
            for field in SEVERE_ZERO_ARTIFACTS
        )
        identity_maximum_grade = max(
            int(grades["identity_contamination"]) for grades in grade_rows
        )
        grade3_ghosting_count = sum(
            int(grades["ghosting"]) == 3 for grades in grade_rows
        )
        parity = bool(endpoint_parity[pair])
        rule_components = {
            "maximum_core_artifact_grade_lte_2": maximum_core_grade <= 2,
            "severe_patch_cloud_mottle_full_body_count_eq_0": severe_core_count == 0,
            "identity_contamination_maximum_grade_eq_0": identity_maximum_grade == 0,
            "grade3_ghosting_count_eq_0": grade3_ghosting_count == 0,
            "endpoint_parity_pass": parity,
        }
        label = "COMPATIBLE" if all(rule_components.values()) else "INCOMPATIBLE"
        score = mean(float(value) for value in rule_components.values())
        active_coverage = mean(
            float(row["active_gaussian_count"]) / float(row["total_gaussian_count"])
            for row in metric_rows
        )
        entry = {
            "pair_id": pair,
            "pair_order_index": PAIR_ORDER.index(pair),
            "calibration_fold": calibration_fold,
            "calibration_condition": calibration_condition,
            "test_fold": test_fold,
            "test_condition": test_condition,
            "test_fold_excluded_from_manifest_construction": True,
            "query_ground_truth_used": False,
            "target_or_test_artifact_label_used": False,
            "descriptor_provenance": "CALIBRATION_FOLD_FROZEN_ORACLE_ARCHIVE_ONLY",
            "intrinsic_and_calibration_descriptors": {
                "symmetric_nearest_support_distance_proxy_mean_displacement_rms": mean(
                    float(row["displacement_rms"]) for row in metric_rows
                ),
                "canonical_support_overlap_proxy_mean_silhouette_iou": mean(
                    float(row["silhouette_iou"]) for row in metric_rows
                ),
                "opacity_weighted_support_overlap_proxy_one_minus_outside_opacity": (
                    1.0
                    - mean(float(row["outside_garment_opacity"]) for row in metric_rows)
                ),
                "garment_support_coverage_ratio_active_over_total": active_coverage,
                "boundary_consistency_mean_fscore": mean(
                    float(row["boundary_fscore"]) for row in metric_rows
                ),
                "canonical_bbox_ratio": None,
                "canonical_bbox_ratio_status": (
                    "UNAVAILABLE_IN_FROZEN_ARCHIVE_NOT_FABRICATED"
                ),
            },
            "rule_evidence": {
                "maximum_core_artifact_grade": maximum_core_grade,
                "severe_patch_cloud_mottle_full_body_count": severe_core_count,
                "identity_contamination_maximum_grade": identity_maximum_grade,
                "grade3_ghosting_count": grade3_ghosting_count,
                "endpoint_parity_pass": parity,
                "visual_record_count": len(grade_rows),
                "oracle_metric_record_count": len(metric_rows),
            },
            "uniform_rule_components": rule_components,
            "compatibility_score": score,
            "compatibility_label": label,
            "expected_mode_when_mixed_and_pair_confident": (
                "DUAL_SUPPORT" if label == "COMPATIBLE" else "HARD_GEOMETRY_SOFT_VA"
            ),
        }
        entry["entry_content_sha256"] = canonical_sha256(entry)
        entries.append(entry)
    manifest = {
        "schema_version": "canondressgs.research.controller_v2_compatibility_manifest.v1",
        "task_id": TASK_ID,
        "status": "PASS",
        "rotation": int(rotation["rotation"]),
        "train_folds": list(rotation["train_folds"]),
        "calibration_fold": calibration_fold,
        "calibration_condition": calibration_condition,
        "test_fold": test_fold,
        "test_condition": test_condition,
        "condition_order": list(CONDITION_ORDER),
        "pair_order": list(PAIR_ORDER),
        "pair_count": len(entries),
        "uniform_rule": {
            "compatible_iff_all": [
                "maximum_core_artifact_grade <= 2",
                "severe_patch_cloud_mottle_full_body_count == 0",
                "identity_contamination_maximum_grade == 0",
                "grade3_ghosting_count == 0",
                "endpoint_parity == PASS",
            ],
            "explicit_pair_blacklist": False,
        },
        "provenance": {
            "allowed": [
                "canonical_endpoint_intrinsic_descriptors",
                "calibration_fold_frozen_oracle_evaluations",
            ],
            "forbidden_and_unused": [
                "target_query_ground_truth",
                "test_fold",
                "manual_pair_blacklist",
            ],
            "source_sha256": dict(source_hashes),
        },
        "entries": entries,
    }
    manifest["manifest_content_sha256"] = canonical_sha256(manifest)
    return manifest


def prior_from_manifest(manifest: Mapping[str, Any]) -> CompatibilityPrior:
    manifest_sha256 = str(manifest["manifest_content_sha256"])
    return CompatibilityPrior(
        [
            CompatibilityEntry(
                pair_id=entry["pair_id"],
                compatibility_score=float(entry["compatibility_score"]),
                compatibility_label=str(entry["compatibility_label"]),
                manifest_sha256=manifest_sha256,
                calibration_condition=str(entry["calibration_condition"]),
            )
            for entry in manifest["entries"]
        ]
    )


def synthetic_output(
    probabilities: list[float],
    *,
    mixedness: float,
    valid_count: int,
    pair_weights: list[float] | None = None,
) -> ControllerV2Output:
    probs = torch.tensor(probabilities, dtype=torch.float32)
    weights = torch.tensor(pair_weights or [0.60] * len(PAIR_ORDER), dtype=torch.float32)
    return ControllerV2Output(
        reference_feature=torch.zeros(512),
        garment_logits=torch.log(probs),
        garment_probabilities=probs,
        mixedness_logit=torch.logit(torch.tensor(mixedness)),
        mixedness_probability=torch.tensor(mixedness),
        pair_weight_logits=torch.logit(weights),
        all_pair_weights=weights,
        valid_reference_count=valid_count,
    )


def boundary_cases(prior: CompatibilityPrior) -> list[dict[str, Any]]:
    thresholds = RoutingThresholds(
        mixedness=0.50,
        pair_confidence=0.05,
        provenance="DESIGN_INTERFACE_SMOKE_ONLY_NOT_SELECTED_FOR_FORMAL_USE",
    )
    cases = (
        (
            "compatible_pair",
            synthetic_output([0.48, 0.34, 0.08, 0.06, 0.04], mixedness=0.90, valid_count=3),
        ),
        (
            "incompatible_pair",
            synthetic_output([0.48, 0.08, 0.34, 0.06, 0.04], mixedness=0.90, valid_count=3),
        ),
        (
            "low_mixedness",
            synthetic_output([0.48, 0.34, 0.08, 0.06, 0.04], mixedness=0.20, valid_count=3),
        ),
        (
            "low_pair_confidence",
            synthetic_output([0.47, 0.20, 0.19, 0.08, 0.06], mixedness=0.90, valid_count=3),
        ),
        (
            "complete_reference_dropout",
            synthetic_output([0.48, 0.34, 0.08, 0.06, 0.04], mixedness=0.90, valid_count=0),
        ),
        (
            "single_reference",
            synthetic_output([0.48, 0.34, 0.08, 0.06, 0.04], mixedness=0.90, valid_count=1),
        ),
        (
            "exact_probability_tie",
            synthetic_output([0.30, 0.30, 0.20, 0.10, 0.10], mixedness=0.90, valid_count=3),
        ),
    )
    endpoint_bank = {outfit: f"/frozen_endpoint_bank/{outfit}.pt" for outfit in OUTFIT_ORDER}
    records = []
    for case_id, output in cases:
        decision = route_controller_v2(output, prior, thresholds)
        runtime = construct_tri_mode_runtime(decision, endpoint_bank)
        records.append(
            {
                "case_id": case_id,
                "result": inference_result_schema(output, decision, runtime),
            }
        )
    return records


def random_forward_cases(model: CompatibilityGatedReferenceControllerV2) -> list[dict[str, Any]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260723)
    specifications = [
        *(f"pure_{outfit}" for outfit in OUTFIT_ORDER),
        "AAB_case_0",
        "AAB_case_1",
        "AAB_case_2",
        "ABB_case_0",
        "ABB_case_1",
        "ABB_case_2",
    ]
    records = []
    with torch.no_grad():
        for index, case_id in enumerate(specifications):
            reference_count = 3
            features = torch.randn(
                reference_count,
                model.reference_row_dim,
                generator=generator,
                dtype=torch.float32,
            )
            output = model(features, torch.ones(reference_count, 1))
            records.append(
                {
                    "case_id": case_id,
                    "case_class": (
                        "PURE"
                        if case_id.startswith("pure_")
                        else "AAB"
                        if case_id.startswith("AAB")
                        else "ABB"
                    ),
                    "reference_shape": list(features.shape),
                    "garment_logit_count": int(output.garment_logits.numel()),
                    "mixedness_logit_count": int(output.mixedness_logit.numel()),
                    "pair_weight_logit_count": int(output.pair_weight_logits.numel()),
                    "all_finite": all(
                        bool(torch.isfinite(value).all())
                        for value in (
                            output.garment_logits,
                            output.mixedness_logit.reshape(1),
                            output.pair_weight_logits,
                        )
                    ),
                    "valid_reference_count": output.valid_reference_count,
                }
            )
    return records


def init_probe_audit(output_root: Path) -> dict[str, Any]:
    paths = {
        (0, "a"): output_root / "audits/init_seed_0_a.json",
        (0, "b"): output_root / "audits/init_seed_0_b.json",
        (1, "a"): output_root / "audits/init_seed_1_a.json",
        (2, "a"): output_root / "audits/init_seed_2_a.json",
    }
    records = {key: read_json(path) for key, path in paths.items()}
    same_seed = records[(0, "a")]["parameter_sha256"] == records[(0, "b")]["parameter_sha256"]
    cross_seed_hashes = {
        records[(seed, "a")]["parameter_sha256"] for seed in (0, 1, 2)
    }
    different_seeds = len(cross_seed_hashes) == 3
    distinct_pids = len({row["fresh_process_pid"] for row in records.values()}) == 4
    fingerprints = {
        row["frozen_input_fingerprint"]["aggregate_sha256"] for row in records.values()
    }
    audit = {
        "required_probe_count": 4,
        "actual_probe_count": len(records),
        "same_seed_reproducible": same_seed,
        "different_seeds_distinct": different_seeds,
        "fresh_process_pids_distinct": distinct_pids,
        "frozen_input_fingerprint_consistent": len(fingerprints) == 1,
        "seed_parameter_sha256": {
            f"seed_{seed}_{probe}": row["parameter_sha256"]
            for (seed, probe), row in records.items()
        },
    }
    if not all(
        (
            same_seed,
            different_seeds,
            distinct_pids,
            len(fingerprints) == 1,
        )
    ):
        raise RuntimeError(f"initialization audit failed: {audit}")
    return audit


def run_design_dry_run(args: argparse.Namespace) -> None:
    assert_no_forbidden_forward_names(controller_v2_forward_argument_names())
    before = frozen_fingerprint(args.repo_root)
    probes = init_probe_audit(args.output_root)
    risk_root = args.repo_root / "paper_protocol/reviewer_risk"
    visual_path = risk_root / "dual_support_all_pair_visual_review.json"
    results_path = risk_root / "dual_support_all_pair_results.json"
    diagnosis_path = risk_root / "controller_pair_compatibility_analysis.json"
    visual = read_json(visual_path)
    results = read_json(results_path)
    diagnosis = read_json(diagnosis_path)
    if visual["actual_opened_count"] != visual["expected_count"]:
        raise RuntimeError("frozen all-pair visual review is incomplete")
    if diagnosis["final_diagnostic_classification"] != "MULTIPLE_FACTORS":
        raise RuntimeError("inherited diagnosis changed")
    parity = endpoint_parity_by_pair(results)
    source_hashes = {
        "dual_support_all_pair_visual_review.json": file_sha256(visual_path),
        "dual_support_all_pair_results.json": file_sha256(results_path),
        "controller_pair_compatibility_analysis.json": file_sha256(diagnosis_path),
    }
    manifests = [
        build_manifest(
            rotation,
            results=results,
            visual=visual,
            endpoint_parity=parity,
            source_hashes=source_hashes,
        )
        for rotation in ROTATIONS
    ]
    for manifest in manifests:
        write_once(
            args.output_root
            / f"rot_{manifest['rotation']}"
            / "compatibility_manifest.json",
            manifest,
        )
    manifest_archive = {
        "schema_version": "canondressgs.research.controller_v2_compatibility_manifests.v1",
        "task_id": TASK_ID,
        "status": "PASS",
        "rotation_count": len(manifests),
        "pair_count_per_rotation": len(PAIR_ORDER),
        "total_pair_records": len(manifests) * len(PAIR_ORDER),
        "rotation_specific_calibration_conditions": [
            manifest["calibration_condition"] for manifest in manifests
        ],
        "rotation_differences_preserved": True,
        "labels_are_rule_generated_not_blacklisted": True,
        "manifests": manifests,
        "execution_counts": ZERO_EXECUTION_COUNTS,
    }
    write_once(
        args.output_root / "audits/controller_v2_compatibility_manifests.json",
        manifest_archive,
    )

    model = CompatibilityGatedReferenceControllerV2(seed=0)
    forward_records = random_forward_cases(model)
    boundary_records = boundary_cases(prior_from_manifest(manifests[0]))
    reached_modes = sorted(
        {row["result"]["mode"] for row in boundary_records}, key=MODES.index
    )
    if tuple(reached_modes) != MODES:
        raise RuntimeError(f"tri-mode reachability failed: {reached_modes}")
    dry_run = {
        "schema_version": "canondressgs.research.controller_v2_dry_run_summary.v1",
        "task_id": TASK_ID,
        "status": "PASS",
        "random_initialization_only": True,
        "model_parameter_count": model.parameter_count,
        "initialization_audit": probes,
        "forward_cases": forward_records,
        "forward_case_counts": {"PURE": 5, "AAB": 3, "ABB": 3, "total": 11},
        "boundary_cases": boundary_records,
        "boundary_case_count": len(boundary_records),
        "reached_modes": reached_modes,
        "formal_rendering_performed": False,
        "execution_counts": ZERO_EXECUTION_COUNTS,
    }
    write_once(
        args.output_root / "audits/controller_v2_dry_run_summary.json", dry_run
    )

    forward_boundary = {
        "schema_version": "canondressgs.research.controller_v2_forward_boundary.v1",
        "task_id": TASK_ID,
        "status": "PASS",
        "prediction_forward_arguments": list(controller_v2_forward_argument_names()),
        "allowed_inputs": ["reference_rgb", "reference_mask", "frozen_F2_spatial_features"],
        "target_forward_leakage": 0,
        "ground_truth_pair_in_prediction_forward": False,
        "ground_truth_weight_in_prediction_forward": False,
        "pair_selection_source": "PREDICTED_UNORDERED_TOP2_GARMENTS",
        "weight_selection_source": "PREDICTED_PAIR_INDEX_IN_ALL_TEN_LOGITS",
        "v1_probability_ratio_as_final_weight": "FORBIDDEN",
        "target_pose_camera_boundary": "RENDERER_ONLY_AFTER_CONTROLLER_DECISION",
        "pair_confidence_definition": PAIR_CONFIDENCE_DEFINITION,
        "tie_handling": "FROZEN_OUTFIT_ORDER",
    }
    write_once(
        args.output_root / "audits/controller_v2_forward_boundary.json",
        forward_boundary,
    )

    optimizer_plan = {
        "schema_version": "canondressgs.research.controller_v2_optimizer_training_plan.v1",
        "task_id": TASK_ID,
        "status": "DESIGN_ONLY_NO_OPTIMIZER_CREATED",
        "fresh_initialization_seeds": [0, 1, 2],
        "fresh_process_per_rotation_and_seed": True,
        "initialization_source": "RANDOM_SEEDED",
        "v1_or_b6_initialization": "FORBIDDEN",
        "frozen_F2": True,
        "optimizer_object_created_in_this_task": False,
        "loss_schema": controller_v2_loss_schema(),
        "mixedness_threshold_grid": list(MIXEDNESS_THRESHOLD_GRID),
        "pair_confidence_threshold_grid": list(PAIR_CONFIDENCE_THRESHOLD_GRID),
        "threshold_selection_source": "CALIBRATION_FOLD_ONLY_FUTURE_PILOT",
        "best_seed_or_rotation_selection": "FORBIDDEN",
        "execution_counts": ZERO_EXECUTION_COUNTS,
    }
    write_once(
        args.output_root / "audits/controller_v2_optimizer_training_plan.json",
        optimizer_plan,
    )

    evaluator_contract = {
        "schema_version": "canondressgs.research.controller_v2_evaluator_contract.v1",
        "task_id": TASK_ID,
        "status": "DESIGN_ONLY",
        "rotations": [dict(rotation) for rotation in ROTATIONS],
        "condition_order": list(CONDITION_ORDER),
        "seeds": [0, 1, 2],
        "report_every_rotation_and_seed": True,
        "selection_or_cherry_picking": "FORBIDDEN",
        "threshold_calibration_uses_test_fold": False,
        "compatibility_manifest_uses_test_fold": False,
        "modes": list(MODES),
        "formal_rendering_in_this_task": False,
        "paper_final": 0,
    }
    write_once(
        args.output_root / "audits/controller_v2_evaluator_contract.json",
        evaluator_contract,
    )

    after = frozen_fingerprint(args.repo_root)
    frozen_audit = {
        "schema_version": "canondressgs.research.controller_v2_frozen_asset_audit.v1",
        "task_id": TASK_ID,
        "status": "PASS" if before == after else "FAIL",
        "before": before,
        "after": after,
        "unchanged": before == after,
        "formal_v1_archive_mutation_count": 0,
        "diagnostic_archive_mutation_count": 0,
    }
    if before != after:
        raise RuntimeError("frozen repository inputs changed during design smoke")
    write_once(
        args.output_root / "audits/frozen_assets_before_after.json", frozen_audit
    )

    final = {
        "schema_version": "canondressgs.research.controller_v2_design_smoke_summary.v1",
        "task_id": TASK_ID,
        "status": "READY",
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "run_branch": RUN_BRANCH,
        "diagnosis": {
            "final": diagnosis["final_diagnostic_classification"],
            "primary": diagnosis["primary_residual_source"],
            "secondary": [
                "WEIGHT_CALIBRATION",
                "FALLBACK_CALIBRATION",
                "REFERENCE_ROBUSTNESS",
            ],
        },
        "architecture": {
            "parameter_count": model.parameter_count,
            "garment_logit_count": 5,
            "mixedness_logit_count": 1,
            "pair_weight_logit_count": 10,
            "pair_order": list(PAIR_ORDER),
            "shared_reference_row_dim": model.reference_row_dim,
            "aggregated_reference_dim": model.input_dim,
        },
        "crossfit_rotation_count": len(ROTATIONS),
        "compatibility_manifest_pair_records": len(manifests) * len(PAIR_ORDER),
        "compatibility_label_counts_per_rotation": [
            {
                "rotation": manifest["rotation"],
                "compatible": sum(
                    row["compatibility_label"] == "COMPATIBLE"
                    for row in manifest["entries"]
                ),
                "incompatible": sum(
                    row["compatibility_label"] == "INCOMPATIBLE"
                    for row in manifest["entries"]
                ),
            }
            for manifest in manifests
        ],
        "dry_run": {
            "forward_case_count": len(forward_records),
            "boundary_case_count": len(boundary_records),
            "reached_modes": reached_modes,
        },
        "initialization_audit": probes,
        "frozen_assets_unchanged": True,
        "execution_counts": ZERO_EXECUTION_COUNTS,
        "paper_final": 0,
        "next_task": (
            "TRAIN_AND_EVALUATE_COMPATIBILITY_GATED_CONTROLLER_V2_CROSSFIT_MICRO_PILOT"
        ),
    }
    write_once(args.output_root / "audits/design_smoke_summary.json", final)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", required=True, choices=("init-probe", "design-dry-run")
    )
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--probe-id")
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.output_root = args.output_root.resolve()
    return args


def main() -> None:
    args = parse_args()
    if args.phase == "init-probe":
        if args.seed is None:
            raise ValueError("--seed is required for init-probe")
        run_init_probe(args)
    else:
        run_design_dry_run(args)


if __name__ == "__main__":
    main()
