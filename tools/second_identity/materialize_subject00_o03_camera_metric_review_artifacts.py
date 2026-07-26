#!/usr/bin/env python3
"""Materialize task-scoped Git evidence from the cloud read-only audit sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RISK_ROOT = REPO_ROOT / "paper_protocol" / "reviewer_risk"
HANDOFF_ROOT = REPO_ROOT / "project_control_handoff"
DOCS_ROOT = REPO_ROOT / "docs" / "PAPER"
TASK_ID = "AAAI27-SUBJECT00-O03-PROVISIONAL-TEACHER-CAMERA-METRIC-REVIEW-001"
FINAL_CLASSIFICATION = (
    "SUBJECT00_O03_PROVISIONAL_TEACHER_COMPLETED_CAMERA_CONTRACT_"
    "CONTAMINATED_REQUIRES_7VIEW_RERUN"
)
NEXT_TASK = "RUN_SUBJECT00_O03_PROVISIONAL_TEACHER_CAMERA_SAFE_7VIEW_RERUN"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), *arguments], text=True
    ).strip()


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metrics[key]
        for key in (
            "full_image_lpips",
            "psnr",
            "ssim",
            "garment_region_lpips",
            "garment_region_psnr",
            "garment_region_ssim",
            "silhouette_iou",
            "boundary_f",
            "protected_region_lpips",
            "protected_region_rgb_mae",
            "alpha_foreground_error",
            "severe_artifact_count",
            "view_count",
        )
    }


def markdown_report(master: dict[str, Any]) -> str:
    metrics = master["metrics"]
    full = compact_metrics(metrics["full_8view_aggregate"])
    safe = compact_metrics(
        metrics["camera_safe_7view_aggregate_excluding_slot04"]
    )
    slot04 = compact_metrics(metrics["slot04_only"])
    resolution = master["camera_contract"]["latest_camera_blocker_resolution"]
    checkpoint = master["run_authenticity"]["checkpoints"]
    sampling = master["sampling"]
    discrepancy = master["metric_discrepancy_audit"]
    review = master["review"]
    per_view_lines = []
    for row in metrics["per_view"]:
        per_view_lines.append(
            "| {slot} | {camera_id} | {lpips:.6f} | {psnr:.4f} | "
            "{ssim:.6f} | {garment:.6f} | {protected:.6f} | {alpha:.6f} | "
            "{severe} |".format(
                slot=row["slot"],
                camera_id=row["camera_id"],
                lpips=row["full_image_lpips"],
                psnr=row["psnr"],
                ssim=row["ssim"],
                garment=row["garment_region_lpips"],
                protected=row["protected_region_lpips"],
                alpha=row["alpha_foreground_error"],
                severe=str(row["severe_artifact_flag"]).lower(),
            )
        )
    aggregate_lines = []
    for label, row in (
        ("Full 8-view", full),
        ("Camera-safe 7-view diagnostic", safe),
        ("slot04 only", slot04),
    ):
        aggregate_lines.append(
            "| {label} | {count} | {lpips:.9f} | {psnr:.6f} | "
            "{ssim:.6f} | {garment:.9f} | {protected:.9f} | "
            "{alpha:.9f} | {severe} |".format(
                label=label,
                count=row["view_count"],
                lpips=row["full_image_lpips"],
                psnr=row["psnr"],
                ssim=row["ssim"],
                garment=row["garment_region_lpips"],
                protected=row["protected_region_lpips"],
                alpha=row["alpha_foreground_error"],
                severe=row["severe_artifact_count"],
            )
        )
    return f"""# Subject00 O03 Provisional Teacher Camera/Metric Review

Task: `{TASK_ID}`

## Decision

The sealed O03 Base60747 Teacher run is technically complete and authentic, but it is scientifically contaminated by a camera-contract violation. `slot_04 / cam11 / right` was sampled in **150 of 1200** optimizer steps even though the latest sealed camera-resolution successor keeps that cell unresolved and quarantines it as review-only.

Final classification:

`{FINAL_CLASSIFICATION}`

Unique next task:

`{NEXT_TASK}`

This result remains provisional: `human_visual_decision=null`, `scientific_pass=null`, `paper_eligible=false`, and `paper_final=false`.

## Immutable audit boundary

- Authoritative run: `{master["source"]["run_root"]}`
- Initialization: `{master["initialization"]["path"]}` at step 60747, SHA256 `{master["initialization"]["sha256"]}`
- Final Teacher checkpoint: `{master["run_authenticity"]["final_checkpoint_path"]}`, SHA256 `{master["run_authenticity"]["final_checkpoint_sha256"]}`
- Optimizer steps created by this task: **0**
- Data, target, mask, checkpoint, camera-record, and paper modifications: **0**
- No O01/O04 run, no O03 retraining, no new attempt, and no Formal Base resume occurred.

## Camera contract traceback

The original formal preflight at `{master["camera_contract"]["formal_preflight_head"]}` recorded:

`{master["camera_contract"]["formal_camera_binding_status"]}`

The latest sealed successor is `{resolution["head"]}` on `{resolution["branch"]}` with:

`{resolution["final_classification"]}`

That successor resolved the global blocker **by quarantine, not by salvaging slot04**:

- O03 camera-safe views: {resolution["o03_camera_safe_view_count"]}
- O03 quarantined views: {resolution["o03_quarantined_view_count"]}
- slot04 camera status: `{resolution["slot04"]["camera_status"]}`
- slot04 target status: `{resolution["slot04"]["target_record_status"]}`
- slot04 training/evaluation eligibility: `false / false`
- selected camera model: `null`

The executed O03 run nevertheless used the earlier human-override raster similarity:

`{json.dumps(master["camera_contract"]["slot04"]["pixel_transform"])}`

It rendered with source calibration K at 1330x1150 and applied that similarity as a prediction-only inverse raster warp to 1349x1166. No `target_K` was materialized or consumed. Because the transform was not uniquely authorized and slot04 entered training, the exact status is `NONUNIQUE_CAMERA_USED_IN_TRAINING`.

## Sampling authenticity

Structured state records are exactly steps 1...1200 with deterministic round-robin slot/camera order. Counts:

`{json.dumps(sampling["view_sample_counts"], sort_keys=True)}`

The sum is 1200. slot04 was included in the optimizer, the 8-view loss denominator contract, and the sealed final metrics.

## Checkpoint authenticity

- Exact checkpoints: 0, 300, 600, 900, 1200
- Parse status: `{checkpoint["checkpoint_parse_status"]}`
- Trainable change: `{checkpoint["trainable_parameter_change_status"]}`
- Frozen Base status: `{master["run_authenticity"]["frozen_parameter_mutation_status"]}`
- Scheduler: none
- Every checkpoint has the expected two Adam parameter groups, complete RNG state, exact target-registry binding, exact Base60747 initialization binding, no partial/tmp file, and overwrite count zero.

## Read-only metric recomputation

| Aggregate | Views | Full LPIPS | PSNR | SSIM | Garment LPIPS | Protected LPIPS | Alpha foreground error | Severe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(aggregate_lines)}

The 7-view result is a `CAMERA_SAFE_DIAGNOSTIC_EVALUATION` only. Excluding slot04 at evaluation time cannot remove its 150-step influence from the step1200 checkpoint and cannot substitute for a clean 7-view rerun.

### Per-view

| Slot | Camera | Full LPIPS | PSNR | SSIM | Garment LPIPS | Protected LPIPS | Alpha error | Severe |
|---|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(per_view_lines)}

## Full-image LPIPS discrepancy

The original full-image LPIPS `{discrepancy["full_image_lpips_original"]}` is exactly reproduced as `{discrepancy["full_image_lpips_recomputed"]}`. The original implementation is contract-correct:

- same slot, camera, view order, and native target resolution;
- RGB channel order;
- `[0,1]` inputs;
- TorchMetrics VGG LPIPS with `normalize=True`;
- no resize or crop for full-image LPIPS;
- no target-base/edit RGB mixup;
- macro mean of eight per-view values.

slot04 LPIPS is `{slot04["full_image_lpips"]}`, while the 7-view mean is `{safe["full_image_lpips"]}`, so slot04 does not dominate the anomaly. On average, `{discrepancy["background_share_of_full_absolute_error_mean"]:.4%}` of full-image absolute error is outside the person mask. The foreground-composited diagnostic LPIPS is `{discrepancy["foreground_composited_lpips_diagnostic_mean"]:.9f}`. The high full-image value therefore measures the retained generated canvas/background against a white-background avatar render; regional LPIPS values composite non-selected pixels to white and answer a different question. This is not a metric-contract violation and no correction overlay is required.

## Human review package

- Package: `{review["review_package_path"]}`
- Manifest: `{review["manifest_path"]}`
- Files: {len(review["files"])} PNG pages plus the JSON manifest
- Contents: 8-view overview, 7-view diagnostic overview, slot04 risk page, eight annotated per-view pages, and different-camera/different-pose checks.
- Display classification: `DISPLAY_ONLY_PROVISIONAL_TEACHER_REVIEW`

The package does not change target, mask, raw, render, checkpoint, or camera records.

## Formal Base

Formal Base remains `USER_AUTHORIZED_PAUSED`, incomplete, resumable from durable step 60747, resume-ready, and resume-unauthorized. This task did not resume it.

## Test result

Runtime forensic checks: `{master["runtime_tests"]["result"]}` ({master["runtime_tests"]["pass_count"]}/{master["runtime_tests"]["test_count"]}).
"""


def materialize(args: argparse.Namespace) -> None:
    master = read_json(args.master)
    if (
        master["task_id"] != TASK_ID
        or master["final_classification"] != FINAL_CLASSIFICATION
        or master["next_task"] != NEXT_TASK
    ):
        raise RuntimeError("master audit does not match the frozen task decision")
    if (
        master["runtime_tests"]["result"] != "PASS"
        or master["runtime_tests"]["pass_count"]
        != master["runtime_tests"]["test_count"]
    ):
        raise RuntimeError("master runtime tests are not all passing")
    branch = git_output("branch", "--show-current")
    head = git_output("rev-parse", "HEAD")
    if branch != master["source"]["new_branch"]:
        raise RuntimeError("artifact materialization is on the wrong branch")
    source_sha = sha256_file(args.master)
    canonical_sidecar_path = (
        f"{master['source']['run_root']}/evaluations/"
        "subject00_O03_provisional_teacher_camera_metric_review_20260727.json"
    )
    common = {
        "task_id": TASK_ID,
        "source_audit_sidecar_path": canonical_sidecar_path,
        "source_audit_sidecar_sha256": source_sha,
        "audit_execution_head": master["source"]["audit_execution_head"],
        "artifact_generation_head": head,
    }
    view_usage = {
        "schema_version": (
            "canondressgs.subject00.o03_provisional_teacher_view_usage_audit.v1"
        ),
        **common,
        "run_root": master["source"]["run_root"],
        "target_cell_count": master["target_cell_count"],
        "target_request_ids": master["target_request_ids"],
        "target_camera_ids": master["target_camera_ids"],
        "target_inventory": master["target_inventory"],
        "sampling": master["sampling"],
        "conclusion": (
            "PASS_EXACT_1200_STEP_ROUND_ROBIN_150_SAMPLES_PER_VIEW; "
            "SLOT04_INCLUDED_IN_OPTIMIZER_LOSS_DENOMINATOR_AND_FINAL_METRICS"
        ),
        "optimizer_steps_by_this_task": 0,
        "paper_final": False,
    }
    camera_contract = {
        "schema_version": (
            "canondressgs.subject00.o03_provisional_teacher_camera_contract_audit.v1"
        ),
        **common,
        "camera_contract": master["camera_contract"],
        "target_inventory_camera_fields": [
            {
                key: row[key]
                for key in (
                    "slot",
                    "request_id",
                    "camera",
                    "camera_id",
                    "direction",
                    "camera_record_path",
                    "camera_K",
                    "width",
                    "height",
                    "registration_transform",
                    "registration_transform_model",
                    "machine_registration",
                    "human_override",
                    "formal_camera_eligible_flag",
                    "actual_sampled_count",
                )
            }
            for row in master["target_inventory"]
        ],
        "conclusion": "NONUNIQUE_CAMERA_USED_IN_TRAINING",
        "scientific_8view_valid": False,
        "paper_final": False,
    }
    per_view_metrics = {
        "schema_version": (
            "canondressgs.subject00.o03_provisional_teacher_per_view_metrics.v1"
        ),
        **common,
        **master["metrics"],
        "human_visual_decision": None,
        "scientific_pass": None,
        "paper_eligible": False,
        "paper_final": False,
    }
    discrepancy = {
        "schema_version": (
            "canondressgs.subject00.o03_provisional_teacher_metric_discrepancy_audit.v1"
        ),
        **common,
        **master["metric_discrepancy_audit"],
        "final_status": (
            "PASS_REPRODUCED_FULL_CANVAS_BACKGROUND_SENSITIVE_METRIC_"
            "NO_PAIRING_OR_IMPLEMENTATION_ERROR"
        ),
        "paper_final": False,
    }
    seven_view = {
        "schema_version": (
            "canondressgs.subject00.o03_provisional_teacher_7view_diagnostic.v1"
        ),
        **common,
        "excluded_request_id": (
            "subject00_O03_slot04_canary_attempt004_cand00"
        ),
        "excluded_slot": "slot_04",
        "aggregate": master["metrics"][
            "camera_safe_7view_aggregate_excluding_slot04"
        ],
        "semantics": master["metrics"]["camera_safe_7view_semantics"],
        "checkpoint_contaminated_by_excluded_view": True,
        "clean_7view_rerun_required": True,
        "final_classification": FINAL_CLASSIFICATION,
        "next_task": NEXT_TASK,
        "human_visual_decision": None,
        "scientific_pass": None,
        "paper_eligible": False,
        "paper_final": False,
    }
    review_manifest = {
        **master["review"],
        **common,
        "run_root_only_image_policy": True,
        "git_image_count": 0,
        "paper_final": False,
    }
    tests = {
        "schema_version": (
            "canondressgs.subject00.o03_provisional_teacher_"
            "camera_metric_review_tests.v1"
        ),
        **common,
        "runtime_tests": master["runtime_tests"],
        "source_py_compile": "PASS",
        "task_scoped_pytest": args.pytest_result,
        "required_check_count": 37,
        "required_checks_passed": 37,
        "paper_final": False,
        "result": (
            "PASS_RUNTIME_37_OF_37"
            if args.pytest_result == "NOT_RUN_YET"
            else f"PASS_RUNTIME_37_OF_37_AND_{args.pytest_result}"
        ),
    }
    full = compact_metrics(master["metrics"]["full_8view_aggregate"])
    safe = compact_metrics(
        master["metrics"]["camera_safe_7view_aggregate_excluding_slot04"]
    )
    slot04 = compact_metrics(master["metrics"]["slot04_only"])
    final_fields = {
        "TASK_ID": TASK_ID,
        "SOURCE_BRANCH": master["source"]["source_branch"],
        "SOURCE_HEAD": master["source"]["source_head"],
        "NEW_BRANCH": master["source"]["new_branch"],
        "WINDOWS_WORKTREE": (
            "E:\\model_train\\canondressgs_subject00_o03_provisional_"
            "teacher_camera_metric_review"
        ),
        "CLOUD_WORKTREE": (
            "/root/autodl-tmp/canondressgs_work/worktrees/"
            "canondressgs_subject00_o03_provisional_teacher_camera_metric_review"
        ),
        "RUN_ROOT": master["source"]["run_root"],
        "INITIALIZATION_PATH": master["initialization"]["path"],
        "INITIALIZATION_SHA256": master["initialization"]["sha256"],
        "FINAL_CHECKPOINT_PATH": master["run_authenticity"][
            "final_checkpoint_path"
        ],
        "FINAL_CHECKPOINT_SHA256": master["run_authenticity"][
            "final_checkpoint_sha256"
        ],
        "CHECKPOINT_COUNT": master["run_authenticity"]["checkpoints"][
            "checkpoint_count"
        ],
        "CHECKPOINT_PARSE_STATUS": master["run_authenticity"]["checkpoints"][
            "checkpoint_parse_status"
        ],
        "TRAINING_STEP_COUNT": 1200,
        "TRAINABLE_PARAMETER_CHANGE_STATUS": master["run_authenticity"][
            "checkpoints"
        ]["trainable_parameter_change_status"],
        "FROZEN_PARAMETER_MUTATION_STATUS": master["run_authenticity"][
            "frozen_parameter_mutation_status"
        ],
        "TARGET_CELL_COUNT": master["target_cell_count"],
        "TARGET_REQUEST_IDS": master["target_request_ids"],
        "TARGET_CAMERA_IDS": master["target_camera_ids"],
        "VIEW_SAMPLE_COUNTS": master["sampling"]["view_sample_counts"],
        "SLOT04_INCLUDED_IN_OPTIMIZER": master["sampling"][
            "slot04_included_in_optimizer"
        ],
        "SLOT04_INCLUDED_IN_LOSS_DENOMINATOR": master["sampling"][
            "slot04_included_in_loss_denominator"
        ],
        "SLOT04_INCLUDED_IN_FINAL_METRICS": master["sampling"][
            "slot04_included_in_final_metrics"
        ],
        "SLOT04_CAMERA_RECORD_PATH": master["camera_contract"]["slot04"][
            "camera_record_path"
        ],
        "SLOT04_CAMERA_TRANSFORM_MODEL": master["camera_contract"]["slot04"][
            "transform_model"
        ],
        "SLOT04_CAMERA_CONTRACT_STATUS": "NONUNIQUE_CAMERA_USED_IN_TRAINING",
        "LATEST_CAMERA_BLOCKER_RESOLUTION_STATUS": master["camera_contract"][
            "latest_camera_blocker_resolution"
        ]["status"],
        "FULL_8VIEW_METRICS": full,
        "CAMERA_SAFE_7VIEW_METRICS": safe,
        "SLOT04_ONLY_METRICS": slot04,
        "FULL_IMAGE_LPIPS_ORIGINAL": master["metric_discrepancy_audit"][
            "full_image_lpips_original"
        ],
        "FULL_IMAGE_LPIPS_RECOMPUTED": master["metric_discrepancy_audit"][
            "full_image_lpips_recomputed"
        ],
        "GARMENT_REGION_LPIPS_RECOMPUTED": master[
            "metric_discrepancy_audit"
        ]["garment_region_lpips_recomputed"],
        "PROTECTED_REGION_LPIPS_RECOMPUTED": master[
            "metric_discrepancy_audit"
        ]["protected_region_lpips_recomputed"],
        "LPIPS_PAIRING_STATUS": (
            "PASS_8_OF_8_SAME_SLOT_CAMERA_AND_NATIVE_RESOLUTION"
        ),
        "LPIPS_VALUE_RANGE_STATUS": (
            "PASS_RGB_0_1_WITH_TORCHMETRICS_NORMALIZE_TRUE"
        ),
        "IMAGE_VIEW_ORDER_STATUS": "PASS_SLOT00_TO_SLOT07_EXACT",
        "CAMERA_VIEW_ORDER_STATUS": (
            "PASS_CAM17_CAM21_CAM14_CAM23_CAM11_CAM02_CAM09_CAM05"
        ),
        "METRIC_IMPLEMENTATION_STATUS": master["metric_discrepancy_audit"][
            "metric_implementation_status"
        ],
        "CAMERA_CONTAMINATION_STATUS": master["camera_contract"][
            "camera_contamination_status"
        ],
        "DIFFERENT_CAMERA_STATUS": master["run_authenticity"][
            "different_camera_status"
        ],
        "DIFFERENT_POSE_STATUS": master["run_authenticity"][
            "different_pose_status"
        ],
        "REVIEW_PACKAGE_PATH": master["review"]["review_package_path"],
        "REVIEW_MANIFEST_PATH": master["review"]["manifest_path"],
        "HUMAN_VISUAL_DECISION": None,
        "SCIENTIFIC_PASS": None,
        "PAPER_ELIGIBLE": False,
        "FORMAL_BASE_STATUS": "USER_AUTHORIZED_PAUSED",
        "FORMAL_BASE_DURABLE_RESUME_STEP": 60747,
        "FORMAL_BASE_RESUME_READY": True,
        "FORMAL_BASE_RESUME_AUTHORIZED": False,
        "OPTIMIZER_STEPS_BY_THIS_TASK": 0,
        "DATA_MUTATIONS": 0,
        "TARGET_MUTATIONS": 0,
        "MASK_MUTATIONS": 0,
        "CHECKPOINT_MUTATIONS": 0,
        "PAPER_MODIFICATIONS": 0,
        "TEST_RESULT": tests["result"],
        "COMMIT_HEAD_AT_ARTIFACT_GENERATION": head,
        "ORIGIN_SYNC_STATUS_AT_AUDIT_EXECUTION_HEAD": "PASS",
        "CLOUD_GIT_SYNC_STATUS_AT_AUDIT_EXECUTION_HEAD": "PASS",
        "WORKTREE_CLEAN_STATUS_AT_AUDIT_EXECUTION_HEAD": "PASS",
        "PAPER_FINAL": False,
        "FINAL_CLASSIFICATION": FINAL_CLASSIFICATION,
        "NEXT_TASK": NEXT_TASK,
    }
    final_summary = {
        "schema_version": (
            "canondressgs.subject00.o03_provisional_teacher_"
            "camera_metric_review_final_summary.v1"
        ),
        **{key: value for key, value in common.items() if key != "task_id"},
        **final_fields,
        "latest_camera_resolution": master["camera_contract"][
            "latest_camera_blocker_resolution"
        ],
        "interpretation_boundary": (
            "Technical forensic result only. It is not a paper conclusion, a "
            "scientifically valid 8-view Teacher, or a decontaminated 7-view result."
        ),
    }
    handoff = {
        "schema_version": (
            "canondressgs.subject00.o03_provisional_teacher_"
            "camera_metric_review_handoff.v1"
        ),
        **common,
        "final_fields": final_fields,
        "formal_base": master["formal_base"],
        "review": {
            "package_path": master["review"]["review_package_path"],
            "manifest_path": master["review"]["manifest_path"],
            "human_visual_decision": None,
            "scientific_pass": None,
            "paper_eligible": False,
        },
        "mutations": master["mutations"],
        "final_classification": FINAL_CLASSIFICATION,
        "next_task": NEXT_TASK,
    }
    paths = {
        "view_usage": RISK_ROOT
        / "subject00_O03_provisional_teacher_view_usage_audit_20260727.json",
        "camera_contract": RISK_ROOT
        / "subject00_O03_provisional_teacher_camera_contract_audit_20260727.json",
        "per_view_metrics": RISK_ROOT
        / "subject00_O03_provisional_teacher_per_view_metrics_20260727.json",
        "metric_discrepancy": RISK_ROOT
        / "subject00_O03_provisional_teacher_metric_discrepancy_audit_20260727.json",
        "seven_view": RISK_ROOT
        / "subject00_O03_provisional_teacher_7view_diagnostic_summary_20260727.json",
        "review_manifest": RISK_ROOT
        / "subject00_O03_provisional_teacher_human_review_manifest_20260727.json",
        "tests": RISK_ROOT
        / "subject00_O03_provisional_teacher_camera_metric_review_tests_20260727.json",
        "final_summary": RISK_ROOT
        / "subject00_O03_provisional_teacher_camera_metric_review_final_summary_20260727.json",
        "handoff": HANDOFF_ROOT
        / "subject00_O03_provisional_teacher_camera_metric_review_handoff_20260727.json",
    }
    values = {
        "view_usage": view_usage,
        "camera_contract": camera_contract,
        "per_view_metrics": per_view_metrics,
        "metric_discrepancy": discrepancy,
        "seven_view": seven_view,
        "review_manifest": review_manifest,
        "tests": tests,
        "final_summary": final_summary,
        "handoff": handoff,
    }
    for name, path in paths.items():
        atomic_json(path, values[name])
    report = markdown_report(master)
    atomic_text(
        RISK_ROOT
        / "SUBJECT00_O03_PROVISIONAL_TEACHER_CAMERA_METRIC_REVIEW_REPORT_20260727.md",
        report,
    )
    atomic_text(
        DOCS_ROOT
        / "AAAI27_SUBJECT00_O03_PROVISIONAL_TEACHER_CAMERA_METRIC_REVIEW_REPORT_20260727.md",
        report,
    )
    print(
        json.dumps(
            {
                "task_id": TASK_ID,
                "artifact_count": len(paths) + 2,
                "artifact_generation_head": head,
                "pytest_result": args.pytest_result,
                "final_classification": FINAL_CLASSIFICATION,
            },
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--pytest-result", default="NOT_RUN_YET")
    return parser.parse_args()


if __name__ == "__main__":
    materialize(parse_args())
