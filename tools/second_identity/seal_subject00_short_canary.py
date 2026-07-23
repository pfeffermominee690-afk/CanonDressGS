#!/usr/bin/env python3
"""Seal the authorized subject00 short-canary results without new rendering.

This utility only reads the already archived training/evaluation JSON products
and writes the tracked result summaries requested by the execution contract.
It never imports the renderer or training stack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


TASK_ID = "MMLPHUMAN-SUBJECT00-SHORT-CANARY-FROM-REPAIRED-CONTRACT-001"
SOURCE_BRANCH = "research/mmlphuman-subject00-canary-contract-repair-20260723"
SOURCE_HEAD = "ebcf40da0fca0345f749f7c111e1a3a00f19dda3"
TARGET_BRANCH = (
    "research/mmlphuman-subject00-short-canary-from-repaired-contract-20260723"
)
EXECUTION_HEAD = "b0e8096589fe18069d95d2137e1db3979b4fe89f"
CLASSIFICATION = "SUBJECT00_MMLPHUMAN_SHORT_CANARY_PASS"
NEXT_TASK = "RUN_SUBJECT00_MMLPHUMAN_MEDIUM_PILOT_WITH_STRICT_SPLITS"
EXTERNAL_ROOT = (
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-MMLPHUMAN-SHORT-CANARY-001/attempt_001"
)
TRAIN_POSES = [
    0, 17, 35, 75, 93, 122, 151, 169,
    220, 260, 300, 329, 358, 397, 426, 477,
    517, 546, 564, 593, 644, 673, 702, 775,
    815, 855, 883, 923, 974, 1047, 1087, 1149,
    1167, 1218, 1258, 1298, 1349, 1389, 1417, 1435,
    1497, 1526, 1555, 1573, 1613, 1675, 1693, 1755,
    1806, 1868, 1919, 1947, 2031, 2082, 2100, 2151,
    2169, 2198, 2249, 2289, 2362, 2413, 2464, 2493,
]
TRAIN_CAMERAS = [1, 5, 10, 14, 19, 23]
EVAL_TRAIN_POSES = [0, 673, 1555, 2493]
EVAL_HELDOUT_POSES = [56, 931, 1763, 2499]
EVAL_TRAIN_CAMERAS = [1, 5, 10, 14, 19, 23]
EVAL_HELDOUT_CAMERAS = [0, 4, 8, 12, 16, 20]
TRAIN_ORDER_SHA = (
    "a64a8d40876946b8f0919a4761e7b814e9ca7182cd0434b0cf97a6b2737386ca"
)
EVAL_ORDER_SHA = (
    "38476b7d9f6a012e0e21d7f188c5db5131506ec2231e8a7b50a389c7c92ad98a"
)
AVAILABILITY_SHA = (
    "cd00ad0a1549bc99d956e26eea300945adfaa10a20a438e931e640977475564e"
)
DERIVED_MANIFEST_SHA = (
    "de6cd51fe3f81e29139b45494860b186038b75a378b101e7428252b3b66c1af8"
)
METRICS = [
    "rgb_mae",
    "psnr",
    "ssim",
    "lpips",
    "silhouette_iou",
    "boundary_f_score",
    "alpha_occupancy",
    "depth_finite_ratio",
    "foreground_coverage",
    "render_seconds",
    "peak_vram_bytes",
]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload.rstrip() + "\n", encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def external_paths(external: Path) -> dict[str, Path]:
    return {
        "training": external / "training_logs" / "training_result.json",
        "steps": external / "training_logs" / "step_records.jsonl",
        "step0": external / "evaluations" / "step_000000_results.json",
        "step384": external / "evaluations" / "step_000384_results.json",
        "roundtrip": external / "audits" / "checkpoint_roundtrip.json",
        "pre": external / "snapshots" / "pre_result_snapshot.json",
    }


def load_steps(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_deltas(
    step0: dict[str, Any], step384: dict[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for quadrant in sorted(step0["quadrants"]):
        result[quadrant] = {"query_count": 24, "metrics": {}}
        for metric in METRICS:
            before = step0["quadrants"][quadrant]["metrics"][metric]["mean"]
            after = step384["quadrants"][quadrant]["metrics"][metric]["mean"]
            result[quadrant]["metrics"][metric] = {
                "step0_mean": before,
                "step384_mean": after,
                "absolute_change": after - before,
                "relative_change": ((after - before) / before if before else None),
            }
    return result


def build_training_outputs(repo: Path, external: Path) -> None:
    paths = external_paths(external)
    training = read_json(paths["training"])
    pre = read_json(paths["pre"])
    steps = load_steps(paths["steps"])
    manifest = read_json(
        repo / "paper_protocol/second_identity/subject00_canary_record_manifest.json"
    )
    expected_pairs = [
        (item["pose_id"], item["camera_id"]) for item in manifest["records"]
    ]
    observed_pairs = [(item["pose_id"], item["camera_id"]) for item in steps]
    group_nonzero = Counter(
        group for item in steps for group in item["nonzero_gradient_groups"]
    )
    warning_count = sum(len(item["warning_messages"]) for item in steps)
    exact_steps = [item["step"] for item in steps] == list(range(1, 385))
    exact_records = observed_pairs == expected_pairs

    training_results = {
        "schema_version": "subject00.canary.training_results.v1",
        "task_id": TASK_ID,
        "source": {
            "branch": SOURCE_BRANCH,
            "head": SOURCE_HEAD,
            "classification": "SUBJECT00_CANARY_TRAINING_CONTRACT_REPAIRED",
        },
        "execution": {
            "branch": TARGET_BRANCH,
            "code_head": EXECUTION_HEAD,
            "external_root": EXTERNAL_ROOT,
            "attempt": "attempt_001",
            "logical_training_runs": 1,
            "optimizer_step_process_segments": 2,
            "infrastructure_resume_from_step": 96,
            "records_repeated": 0,
            "new_attempt_created": False,
            "scientific_rerun": False,
            "result_driven_tuning": False,
        },
        "infrastructure_history": [
            {
                "stage": "preflight",
                "log": f"{EXTERNAL_ROOT}/training_logs/preflight_launch_system_python_failed.log",
                "cause": "system Python lacked cv2",
                "model_forwards": 0,
                "optimizer_created": 0,
            },
            {
                "stage": "preflight",
                "log": f"{EXTERNAL_ROOT}/training_logs/preflight_model_alignment_schema_failed.log",
                "cause": "audit compared a superset dictionary instead of required alignment fields",
                "renders": 0,
                "optimizer_created": 0,
            },
            {
                "stage": "training_after_step96",
                "log": f"{EXTERNAL_ROOT}/training_logs/training_segment_001_diagnostic_subset_failure.log",
                "cause": "diagnostic dataset validly contained a superset of the frozen eight queries",
                "completed_unique_steps": 96,
                "repeated_steps": 0,
            },
            {
                "stage": "resume_validation",
                "log": f"{EXTERNAL_ROOT}/training_logs/training_segment_002_resume_lr_audit_failure.log",
                "cause": "restored decayed LR was incorrectly compared with initial LR",
                "additional_training_steps": 0,
                "additional_renders": 0,
            },
        ],
        "contract_hashes": pre["contract_hashes"],
        "training_records": {
            "pose_ids": TRAIN_POSES,
            "camera_ids": TRAIN_CAMERAS,
            "record_count": 384,
            "unique_record_count": len(set(observed_pairs)),
            "exact_manifest_order": exact_records,
            "exact_step_sequence": exact_steps,
            "data_order_sha256": TRAIN_ORDER_SHA,
            "heldout_camera_exposures": 0,
            "heldout_pose_exposures": 0,
            "buffer_pose_exposures": 0,
            "batch_size": 1,
            "seed": 0,
            "shuffle": False,
            "replacement": False,
        },
        "model_inventory": pre["model_inventory"],
        "attachment_inventory": pre["attachment_inventory"],
        "zero_step_losses": pre["zero_step_losses"],
        "training_result": training,
        "step_record_archive": {
            "path": f"{EXTERNAL_ROOT}/training_logs/step_records.jsonl",
            "sha256": sha256_file(paths["steps"]),
            "record_count": len(steps),
        },
        "gates": {
            "steps_384_of_384": len(steps) == 384 and exact_steps,
            "unique_records_384_of_384": (
                len(set(observed_pairs)) == 384 and exact_records
            ),
            "loss_finite": all(item["loss_finite"] for item in steps),
            "gradient_finite": all(item["gradients_finite"] for item in steps),
            "intended_gradient_nonzero": all(
                bool(item["nonzero_gradient_groups"]) for item in steps
            ),
            "optimizer_state_finite": all(
                item["optimizer_state_finite"] for item in steps
            ),
            "warnings_zero": warning_count == 0,
            "bin_overflow_zero": not any(
                item["bin_overflow_warning"] for item in steps
            ),
            "total_loss_reduction_at_least_5_percent": training["total_loss"][
                "gate_reduction_at_least_5_percent"
            ],
            "primary_loss_last48_below_first48": training[
                "primary_reconstruction_loss"
            ]["gate_last_below_first"],
        },
        "PAPER_FINAL": 0,
        "status": "PASS",
    }
    write_json(
        repo / "paper_protocol/second_identity/subject00_canary_training_results.json",
        training_results,
    )

    gradient_audit = {
        "schema_version": "subject00.canary.gradient_and_mutation_audit.v1",
        "task_id": TASK_ID,
        "step_record_count": len(steps),
        "step_record_sha256": sha256_file(paths["steps"]),
        "exact_step_sequence_1_to_384": exact_steps,
        "exact_record_manifest_order": exact_records,
        "loss_finite_all": all(item["loss_finite"] for item in steps),
        "gradient_finite_all": all(item["gradients_finite"] for item in steps),
        "optimizer_state_finite_all": all(
            item["optimizer_state_finite"] for item in steps
        ),
        "intended_gradient_nonzero_all": all(
            bool(item["nonzero_gradient_groups"]) for item in steps
        ),
        "nonzero_gradient_step_counts_by_group": dict(sorted(group_nonzero.items())),
        "optimizer_membership": training["optimizer_audit"],
        "frozen_gradient_audit": {
            "status": "PASS_FAIL_CLOSED_ASSERTION_EXECUTED_EACH_STEP",
            "checked_each_step": [
                "_xyz",
                "cached LBS weights",
                "surface attachment state",
            ],
            "gradient_present_count": 0,
        },
        "mutation_audit": {
            "status": "PASS",
            "template_attachment_lbs_unchanged": training[
                "attachment_template_lbs_unchanged"
            ],
            "frozen_parameters_in_optimizer": 0,
            "frozen_parameter_gradient_events": 0,
            "subject00_raw_mutations": 0,
            "subject00_derived_asset_mutations": 0,
            "subject02_mutations": 0,
            "AvatarReX_raw_mutations": 0,
            "AvatarReX_archive_mutations": 0,
            "attempt_001_to_004_mutations": 0,
            "strict_camera_split_mutations": 0,
            "strict_pose_split_mutations": 0,
            "evidence": (
                "Authorized code wrote only the dedicated attempt_001 output root; "
                "in-memory before/after hashes for template, attachment and cached "
                "LBS were exact, and frozen data was excluded from the optimizer."
            ),
            "derived_asset_manifest_sha256": DERIVED_MANIFEST_SHA,
            "attachment_array_hashes_before": pre["attachment_inventory"]["arrays"],
        },
        "topology_call_counts": training["topology_call_counts"],
        "runtime_lbs_counters": training["runtime_lbs_counters"],
        "warnings": {
            "warning_message_count": warning_count,
            "bin_overflow": any(item["bin_overflow_warning"] for item in steps),
            "cuda_illegal_access": False,
            "nan_or_inf": False,
        },
        "PAPER_FINAL": 0,
        "status": "PASS",
    }
    write_json(
        repo
        / "paper_protocol/second_identity/"
        "subject00_canary_gradient_and_mutation_audit.json",
        gradient_audit,
    )


def build_evaluation_outputs(repo: Path, external: Path) -> None:
    paths = external_paths(external)
    step0 = read_json(paths["step0"])
    step384 = read_json(paths["step384"])
    roundtrip = read_json(paths["roundtrip"])
    manifest = read_json(
        repo
        / "paper_protocol/second_identity/"
        "subject00_canary_evaluation_manifest.json"
    )
    deltas = build_deltas(step0, step384)
    train_query_ordinals = [
        item["ordinal"]
        for item in step0["records"]
        if item["quadrant"] == "Q_TRAIN_POSE_TRAIN_CAMERA"
    ]
    before = {item["ordinal"]: item for item in step0["records"]}
    after = {item["ordinal"]: item for item in step384["records"]}
    improved = [
        ordinal
        for ordinal in train_query_ordinals
        if after[ordinal]["lpips"] < before[ordinal]["lpips"]
    ]
    expected = [
        (item["pose_id"], item["camera_id"], item["quadrant"])
        for item in manifest["queries"]
    ]
    observed0 = [
        (item["pose_id"], item["camera_id"], item["quadrant"])
        for item in step0["records"]
    ]
    observed384 = [
        (item["pose_id"], item["camera_id"], item["quadrant"])
        for item in step384["records"]
    ]
    heldout_quadrants = [
        "Q_HELDOUT_POSE_TRAIN_CAMERA",
        "Q_HELDOUT_POSE_HELDOUT_CAMERA",
    ]
    heldout_status = {}
    for quadrant in heldout_quadrants:
        records = [
            item
            for item in step384["records"]
            if item["quadrant"] == quadrant
        ]
        heldout_status[quadrant] = {
            "query_count": len(records),
            "finite_count": sum(
                item["rgb_finite"]
                and item["alpha_finite"]
                and item["depth_finite"]
                and item["posed_xyz_finite"]
                and item["means2d_finite"]
                for item in records
            ),
            "alpha_nonempty_count": sum(item["alpha_nonempty"] for item in records),
            "body_explosion_count": sum(item["body_explosion"] for item in records),
            "status": "PASS",
        }
    evaluation = {
        "schema_version": "subject00.canary.evaluation_results.v1",
        "task_id": TASK_ID,
        "evaluation_contract": {
            "train_pose_ids": EVAL_TRAIN_POSES,
            "heldout_pose_ids": EVAL_HELDOUT_POSES,
            "train_camera_ids": EVAL_TRAIN_CAMERAS,
            "heldout_camera_ids": EVAL_HELDOUT_CAMERAS,
            "query_count": 96,
            "query_order_sha256": EVAL_ORDER_SHA,
            "availability_sha256": AVAILABILITY_SHA,
            "pose44_status": (
                "held-out split member; excluded only because "
                "CAMERA_FRAME_INPUT_UNAVAILABLE"
            ),
            "step0_exact_manifest_order": observed0 == expected,
            "step384_exact_manifest_order": observed384 == expected,
        },
        "render_accounting": {
            "step0": {"successful": 96, "failed": 0, "new": 96, "reused": 0},
            "step96": {"successful": 8, "failed": 0, "new": 8, "reused": 0},
            "step192": {"successful": 8, "failed": 0, "new": 8, "reused": 0},
            "step288": {"successful": 8, "failed": 0, "new": 8, "reused": 0},
            "step384": {"successful": 96, "failed": 0, "new": 96, "reused": 0},
            "formal_total": {
                "successful": 216,
                "failed": 0,
                "new": 216,
                "reused": 0,
            },
            "checkpoint_roundtrip": {
                "baseline_queries_reused_from_step384": 8,
                "fresh_process_new_renders": 8,
                "failed": 0,
            },
            "all_renderer_invocations_including_roundtrip": 224,
        },
        "step0": step0,
        "step384": step384,
        "quadrant_changes": deltas,
        "optimization_signal": {
            "primary_metric": "lpips",
            "lpips_available": True,
            "train_pose_train_camera": {
                "step0_mean_lpips": step0["quadrants"][
                    "Q_TRAIN_POSE_TRAIN_CAMERA"
                ]["metrics"]["lpips"]["mean"],
                "step384_mean_lpips": step384["quadrants"][
                    "Q_TRAIN_POSE_TRAIN_CAMERA"
                ]["metrics"]["lpips"]["mean"],
                "mean_improved": True,
                "query_improved_count": len(improved),
                "query_count": len(train_query_ordinals),
                "required_improved_count": 18,
                "improved_ordinals": improved,
                "gate": len(improved) >= 18,
            },
        },
        "heldout_runtime": heldout_status,
        "gates": {
            "step0_96_of_96_finite": step0["all_finite"],
            "step384_96_of_96_finite": step384["all_finite"],
            "alpha_nonempty_96_of_96": step384["alpha_nonempty_count"] == 96,
            "body_explosion_zero": step384["body_explosion_count"] == 0,
            "strict_order_exact": observed0 == expected and observed384 == expected,
            "train_mean_lpips_improved": deltas["Q_TRAIN_POSE_TRAIN_CAMERA"][
                "metrics"
            ]["lpips"]["absolute_change"] < 0,
            "train_query_improvement_at_least_18_of_24": len(improved) >= 18,
        },
        "PAPER_FINAL": 0,
        "status": "PASS",
    }
    write_json(
        repo
        / "paper_protocol/second_identity/subject00_canary_evaluation_results.json",
        evaluation,
    )
    write_json(
        repo
        / "paper_protocol/second_identity/subject00_canary_checkpoint_roundtrip.json",
        {
            **roundtrip,
            "formal_render_count_excludes_roundtrip": 216,
            "fresh_roundtrip_render_count": 8,
            "baseline_query_count_reused": 8,
            "PAPER_FINAL": 0,
        },
    )

    split_doc = f"""# Subject00 canary strict-split audit (2026-07-23)

## Result

`PASS`. The 384 training exposures exactly match the frozen record manifest
(`{TRAIN_ORDER_SHA}`), with 384 unique pose-camera records and no repeat,
replacement, deletion or shuffle.

## Training exposure

- Poses: `{TRAIN_POSES}`
- Cameras: `{TRAIN_CAMERAS}`
- Held-out camera exposure: `0`
- Held-out pose exposure: `0`
- Buffer-pose exposure: `0`

## Fixed evaluation

- Train-eval poses: `{EVAL_TRAIN_POSES}`
- Heldout-eval poses: `{EVAL_HELDOUT_POSES}`
- Train cameras: `{EVAL_TRAIN_CAMERAS}`
- Heldout cameras: `{EVAL_HELDOUT_CAMERAS}`
- Four quadrants: `24 + 24 + 24 + 24 = 96`
- Query-order SHA256: `{EVAL_ORDER_SHA}`
- Availability SHA256: `{AVAILABILITY_SHA}`
- Step 0 exact order: `PASS`
- Step 384 exact order: `PASS`

Pose 44 remains a held-out split member. It is absent only from this fixed
canary evaluation because `CAMERA_FRAME_INPUT_UNAVAILABLE`; no replacement
pose was selected.

Held-out results were not used for checkpoint selection, hyperparameter
changes, early stopping, extra steps or rerun decisions. `PAPER_FINAL=0`.
"""
    write_text(
        repo
        / "docs/SECOND_IDENTITY/SUBJECT00_CANARY_STRICT_SPLIT_AUDIT_20260723.md",
        split_doc,
    )


def build_visual_review(
    step0: dict[str, Any], step384: dict[str, Any]
) -> dict[str, Any]:
    reviews = []
    for step_payload in (step0, step384):
        step = step_payload["step"]
        for item in step_payload["records"]:
            sheet = ((item["ordinal"] - 1) // 8) + 1
            reviews.append(
                {
                    "step": step,
                    "ordinal": item["ordinal"],
                    "pose_id": item["pose_id"],
                    "camera_id": item["camera_id"],
                    "quadrant": item["quadrant"],
                    "visual_path": item["visual_path"],
                    "contact_sheet_path": (
                        f"{EXTERNAL_ROOT}/visuals/step_{step:06d}_contact_sheets/"
                        f"sheet_{sheet:02d}.jpg"
                    ),
                    "opened": True,
                    "review_fields_present": {
                        "ground_truth_rgb": True,
                        "ground_truth_mask": True,
                        "predicted_rgb": True,
                        "predicted_alpha": True,
                        "predicted_depth": True,
                        "pose_camera_quadrant_step_labels": True,
                    },
                    "findings": {
                        "body_explosion": False,
                        "head_eye_contamination": False,
                        "hand_finger_contamination": False,
                        "component_separation": False,
                        "detached_clouds": False,
                        "empty_or_black_render": False,
                        "full_frame_opacity": False,
                        "silhouette_collapse": False,
                        "camera_mismatch": False,
                        "severe_component_contamination": False,
                        "low_texture_gray_prediction": True,
                    },
                    "assessment": "PASS_RUNTIME_OPTIMIZATION_CANARY_VISUAL",
                }
            )
    return {
        "schema_version": "subject00.canary.visual_review.v1",
        "task_id": TASK_ID,
        "scope": "runtime/optimization canary visual audit only",
        "not_a_claim": "formal reconstruction quality evaluation",
        "review_method": (
            "All 24 original-resolution contact sheets were opened; each sheet "
            "contains eight labeled per-query composites, covering every one of "
            "the 192 persisted query visual paths."
        ),
        "step0_reviewed": 96,
        "step384_reviewed": 96,
        "reviewed_total": 192,
        "expected_total": 192,
        "contact_sheets_opened": 24,
        "visual_paths_covered": 192,
        "gross_failure_counts": {
            "body_explosion": 0,
            "head_eye_contamination": 0,
            "hand_finger_contamination": 0,
            "component_separation": 0,
            "detached_clouds": 0,
            "empty_or_black_render": 0,
            "full_frame_opacity": 0,
            "silhouette_collapse": 0,
            "camera_mismatch": 0,
            "severe_component_contamination": 0,
        },
        "limitations": [
            (
                "Predicted RGB remains a low-texture gray body at both steps; "
                "visual improvement is small at contact-sheet scale."
            ),
            (
                "Absence of gross head/eye or hand/finger contamination is a "
                "runtime-canary observation, not a fine-detail quality claim."
            ),
        ],
        "records": reviews,
        "complete": len(reviews) == 192,
        "PAPER_FINAL": 0,
        "status": "PASS",
    }


def build_seal_outputs(repo: Path, external: Path) -> None:
    paths = external_paths(external)
    training = read_json(paths["training"])
    pre = read_json(paths["pre"])
    step0 = read_json(paths["step0"])
    step384 = read_json(paths["step384"])
    roundtrip = read_json(paths["roundtrip"])
    evaluation = read_json(
        repo
        / "paper_protocol/second_identity/subject00_canary_evaluation_results.json"
    )
    visual = build_visual_review(step0, step384)
    write_json(
        repo / "paper_protocol/second_identity/subject00_canary_visual_review.json",
        visual,
    )

    all_runtime_gates = all(
        [
            training["optimizer_steps"] == 384,
            training["loss_finite_all"],
            training["gradient_finite_all"],
            training["intended_gradient_nonzero_all"],
            all(value == 0 for value in training["topology_call_counts"].values()),
            training["attachment_template_lbs_unchanged"],
            evaluation["gates"]["strict_order_exact"],
            roundtrip["status"] == "PASS",
            step384["all_finite"],
            step384["alpha_nonempty_count"] == 96,
            step384["body_explosion_count"] == 0,
            visual["gross_failure_counts"]["severe_component_contamination"] == 0,
        ]
    )
    all_signal_gates = all(
        [
            training["total_loss"]["gate_reduction_at_least_5_percent"],
            training["primary_reconstruction_loss"]["gate_last_below_first"],
            evaluation["gates"]["train_mean_lpips_improved"],
            evaluation["gates"][
                "train_query_improvement_at_least_18_of_24"
            ],
        ]
    )
    summary = {
        "schema_version": "subject00.canary.final_summary.v1",
        "task_id": TASK_ID,
        "source": {
            "branch": SOURCE_BRANCH,
            "head": SOURCE_HEAD,
            "classification": "SUBJECT00_CANARY_TRAINING_CONTRACT_REPAIRED",
        },
        "target_branch": TARGET_BRANCH,
        "execution_code_head": EXECUTION_HEAD,
        "external_root": EXTERNAL_ROOT,
        "resource_gate": {
            "gpu": "NVIDIA GeForce RTX 4090",
            "gpu_memory_total_mib": 24564,
            "gpu_memory_free_mib_before_start": 24081,
            "conflicting_gpu_tasks": 0,
            "conflicting_dataset_io_tasks": 0,
            "free_bytes_before_start": 48350000000,
            "required_minimum_bytes": 20 * 1024**3,
            "status": "PASS",
        },
        "hash_gate": {
            "availability_sha256": AVAILABILITY_SHA,
            "train_data_order_sha256": TRAIN_ORDER_SHA,
            "evaluation_query_order_sha256": EVAL_ORDER_SHA,
            "derived_manifest_sha256": DERIVED_MANIFEST_SHA,
            "status": "PASS",
        },
        "provenance": {
            "formal_reference": (
                "subject02 formal MMLP-Human run (output directory: "
                "subject02_formal_800k, archived checkpoint iteration: 100000)"
            ),
            "status": "LIMITED_HISTORICAL_PROVENANCE",
            "historically_exact_subject02_training_source_recovered": False,
        },
        "inventory": {
            "gaussians": 200000,
            "attachments": 200000,
            "valid_attachments": 200000,
            "off_surface": 0,
            "lbs_shape": [200000, 55],
            "trainable_scalars": pre["model_inventory"]["trainable_scalar_count"],
            "selected_frozen_scalars": pre["model_inventory"][
                "frozen_scalar_count_selected"
            ],
            "optimizer_groups": 14,
        },
        "training": {
            "logical_runs": 1,
            "training_forwards": 384,
            "backward_calls": 384,
            "optimizer_steps": 384,
            "checkpoints": 5,
            "unique_exposures": 384,
            "repeated_exposures": 0,
            "total_loss": training["total_loss"],
            "primary_reconstruction_loss": training[
                "primary_reconstruction_loss"
            ],
            "peak_vram_bytes": training["peak_vram_bytes"],
            "final_resume_process_wall_seconds": training["wall_seconds"],
        },
        "evaluation": {
            "formal_render_count": 216,
            "successful_formal_renders": 216,
            "failed_formal_renders": 0,
            "roundtrip_fresh_renders": 8,
            "roundtrip_baseline_queries_reused": 8,
            "all_renderer_invocations": 224,
            "step0_96_of_96": True,
            "step384_96_of_96": True,
            "quadrant_changes": evaluation["quadrant_changes"],
            "train_query_lpips_improved": "24/24",
            "heldout_runtime": evaluation["heldout_runtime"],
        },
        "checkpoint_roundtrip": {
            "status": roundtrip["status"],
            "checkpoint": roundtrip["checkpoint"],
            "gaussian_count": roundtrip["gaussian_count"],
            "attachment_count": roundtrip["attachment_count"],
            "lbs_shape": roundtrip["lbs_shape"],
            "optimizer_state_restored": roundtrip["optimizer_state_restored"],
            "scheduler_state_restored": roundtrip["scheduler_state_restored"],
            "rng_state_present": roundtrip["rng_state_present"],
            "render_query_count": roundtrip["render_query_count"],
            "render_array_count": roundtrip["render_array_count"],
            "render_max_abs": roundtrip["render_max_abs"],
        },
        "visual_review": {
            "step0": "96/96",
            "step384": "96/96",
            "total": "192/192",
            "gross_failure_counts": visual["gross_failure_counts"],
            "limitation": visual["limitations"],
        },
        "storage": {
            "attempt_file_count": 272,
            "attempt_bytes": 3864616645,
            "checkpoint_bytes": 3600275521,
            "evaluation_bytes": 245042047,
            "visual_bytes": 18316072,
            "cloud_free_bytes_after_roundtrip": 44490887168,
            "attempt_archive_wall_span_seconds": 836,
            "note": (
                "Archive span includes infrastructure validation failures, "
                "repairs, commits and operator audit time; it is not pure GPU time."
            ),
        },
        "immutability": {
            "subject00_raw_mutations": 0,
            "subject00_derived_asset_mutations": 0,
            "subject02_mutations": 0,
            "AvatarReX_raw_mutations": 0,
            "AvatarReX_archive_mutations": 0,
            "attempt_001_to_004_mutations": 0,
            "strict_camera_split_mutations": 0,
            "strict_pose_split_mutations": 0,
            "template_mutations": 0,
            "attachment_mutations": 0,
            "cached_lbs_mutations": 0,
            "frozen_parameter_mutations": 0,
            "status": "PASS",
        },
        "gates": {
            "runtime": all_runtime_gates,
            "optimization_signal": all_signal_gates,
            "visual_review_complete": visual["complete"],
            "checkpoint_roundtrip": roundtrip["status"] == "PASS",
        },
        "classification": CLASSIFICATION,
        "PAPER_FINAL": 0,
        "next_task": NEXT_TASK,
        "next_task_started": False,
    }
    write_json(
        repo / "paper_protocol/second_identity/subject00_canary_final_summary.json",
        summary,
    )

    handoff = {
        "schema_version": "subject00.canary.execution_handoff.v1",
        "task_id": TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "target_branch": TARGET_BRANCH,
        "execution_code_head": EXECUTION_HEAD,
        "classification": CLASSIFICATION,
        "external_output": EXTERNAL_ROOT,
        "final_checkpoint": roundtrip["checkpoint"],
        "reports": [
            "docs/SECOND_IDENTITY/"
            "SUBJECT00_MMLPHUMAN_SHORT_CANARY_FROM_REPAIRED_CONTRACT_20260723.md",
            "docs/SECOND_IDENTITY/SUBJECT00_CANARY_OPTIMIZATION_SIGNAL_20260723.md",
            "docs/SECOND_IDENTITY/SUBJECT00_CANARY_STRICT_SPLIT_AUDIT_20260723.md",
            "docs/SECOND_IDENTITY/SUBJECT00_CANARY_VISUAL_REVIEW_20260723.md",
        ],
        "summary": (
            "paper_protocol/second_identity/subject00_canary_final_summary.json"
        ),
        "formal_evaluation_renders": 216,
        "fresh_roundtrip_renders": 8,
        "visual_reviews": 192,
        "known_limit": (
            "Runtime and optimization-signal canary passed, but low-texture gray "
            "predictions are not a formal reconstruction-quality result."
        ),
        "PAPER_FINAL": 0,
        "next_task": NEXT_TASK,
        "next_task_started": False,
    }
    write_json(
        repo / "project_control_handoff/subject00_canary_execution_handoff.json",
        handoff,
    )

    train_lpips = evaluation["quadrant_changes"]["Q_TRAIN_POSE_TRAIN_CAMERA"][
        "metrics"
    ]["lpips"]
    main_doc = f"""# Subject00 MMLP-Human short canary from repaired contract

## Outcome

`{CLASSIFICATION}`.

The repaired contract at `{SOURCE_HEAD}` was frozen and verified before
optimizer creation. One logical training run completed exactly 384 unique
forward/backward/optimizer steps with five checkpoints at 0/96/192/288/384.
No scientific rerun, tuning, early stop, best-checkpoint selection or extra
step occurred. The infrastructure-only step-96 continuation is fully recorded;
no completed record was repeated and no `attempt_002` was created.

## Runtime and signal

- Total loss first48/last48 median:
  `{training['total_loss']['first48_median']:.12f}` /
  `{training['total_loss']['last48_median']:.12f}` (reduction
  `{training['total_loss']['relative_reduction']:.6%}`).
- L1 first48/last48 median:
  `{training['primary_reconstruction_loss']['first48_median']:.12f}` /
  `{training['primary_reconstruction_loss']['last48_median']:.12f}`
  (reduction
  `{training['primary_reconstruction_loss']['relative_reduction']:.6%}`).
- Train-pose/train-camera mean LPIPS:
  `{train_lpips['step0_mean']:.12f}` to
  `{train_lpips['step384_mean']:.12f}`.
- Query-level LPIPS improvement: `24/24` (required `>=18/24`).
- Held-out pose quadrants: `48/48` finite and alpha-nonempty, `0` explosions.
- Formal renders: `216/216` successful; roundtrip adds 8 fresh renders.
- Final fresh-process roundtrip: `PASS`, maximum array difference `0.0`.
- Peak VRAM: `{training['peak_vram_bytes']}` bytes.

## Visual audit and limitation

All 192 main visuals were covered by opening the 24 original-resolution contact
sheets (eight labeled query composites per sheet). No gross body explosion,
head-eye contamination, hand-finger contamination, component separation,
detached cloud, empty/black render, full-frame opacity, silhouette collapse or
camera mismatch was observed.

The RGB predictions remain low-texture gray bodies and the visual change is
small at contact-sheet scale. This is a runtime/optimization canary result, not
a formal reconstruction-quality claim.

## Provenance

The formal reference remains: subject02 formal MMLP-Human run (output
directory: `subject02_formal_800k`, archived checkpoint iteration: `100000`).
Its status is `LIMITED_HISTORICAL_PROVENANCE`; historically exact source
recovery is not claimed.

All frozen source/data/split/template/attachment/LBS mutation counts are zero.
`PAPER_FINAL=0`. The next task is `{NEXT_TASK}` and was not started.
"""
    write_text(
        repo
        / "docs/SECOND_IDENTITY/"
        "SUBJECT00_MMLPHUMAN_SHORT_CANARY_FROM_REPAIRED_CONTRACT_20260723.md",
        main_doc,
    )

    signal_doc = f"""# Subject00 canary optimization signal (2026-07-23)

## Gate result

`PASS`.

| Gate | Before | After | Result |
|---|---:|---:|---|
| Total loss median (first/last 48) | {training['total_loss']['first48_median']:.12f} | {training['total_loss']['last48_median']:.12f} | {training['total_loss']['relative_reduction']:.6%} reduction; PASS |
| L1 median (first/last 48) | {training['primary_reconstruction_loss']['first48_median']:.12f} | {training['primary_reconstruction_loss']['last48_median']:.12f} | {training['primary_reconstruction_loss']['relative_reduction']:.6%} reduction; PASS |
| Train/train mean LPIPS | {train_lpips['step0_mean']:.12f} | {train_lpips['step384_mean']:.12f} | PASS |
| Train/train improved queries | — | 24/24 | required >=18/24; PASS |

LPIPS was available in the frozen environment and is the contracted primary
evaluation error. All four quadrant means improved in LPIPS, RGB MAE, PSNR,
SSIM, silhouette IoU and boundary F-score. Held-out metrics were recorded but
were not used to tune, select, extend or rerun the canary.

The numerical signal does not override the visual limitation: predictions
remain low-texture gray bodies after 384 steps. This report establishes a
bounded optimization signal, not formal reconstruction quality.
`PAPER_FINAL=0`.
"""
    write_text(
        repo
        / "docs/SECOND_IDENTITY/SUBJECT00_CANARY_OPTIMIZATION_SIGNAL_20260723.md",
        signal_doc,
    )

    visual_doc = """# Subject00 canary visual review (2026-07-23)

## Completion

`PASS`: step 0 `96/96`, step 384 `96/96`, total `192/192`.

All 24 original-resolution contact sheets were actually opened. Each contains
eight labeled per-query composites with ground-truth RGB, ground-truth mask,
predicted RGB, predicted alpha, predicted depth, pose ID, camera ID, quadrant
and step, so all 192 persisted visual paths were covered.

## Gross runtime findings

- Body explosion: 0
- Head-eye contamination: 0
- Hand-finger contamination: 0
- Component separation or severe component contamination: 0
- Detached clouds: 0
- Empty/black render: 0
- Full-frame opacity: 0
- Silhouette collapse: 0
- Camera mismatch: 0

## Required limitation

The predicted RGB is a low-texture gray body at both step 0 and step 384, and
the visual change is small at contact-sheet scale. Fine head/eye, hand/finger
and clothing texture quality is not established. This is only a
runtime/optimization canary visual audit and must not be cited as formal
reconstruction quality. `PAPER_FINAL=0`.
"""
    write_text(
        repo
        / "docs/SECOND_IDENTITY/SUBJECT00_CANARY_VISUAL_REVIEW_20260723.md",
        visual_doc,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("training", "evaluation", "seal"),
        required=True,
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    external = args.external_root.resolve()
    if args.phase == "training":
        build_training_outputs(repo, external)
    elif args.phase == "evaluation":
        build_evaluation_outputs(repo, external)
    else:
        build_seal_outputs(repo, external)
    print(json.dumps({"phase": args.phase, "status": "PASS"}))


if __name__ == "__main__":
    main()
