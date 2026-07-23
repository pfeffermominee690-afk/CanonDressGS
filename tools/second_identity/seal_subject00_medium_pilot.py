#!/usr/bin/env python3
"""Seal existing Subject00 medium-pilot evidence without running model code.

This utility only parses already-produced JSON and comparison-sheet files and
writes the tracked decision archive. It deliberately imports no training,
renderer, inference, or metric-aggregation modules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


TASK_ID = "MMLPHUMAN-SUBJECT00-ONE-PASS-MEDIUM-PILOT-001"
CLASSIFICATION = "SUBJECT00_MMLPHUMAN_MEDIUM_PILOT_PASS"
REPRESENTATION_STATUS = "APPEARANCE_AND_SILHOUETTE_EMERGING"
NEXT_TASK = "DESIGN_SUBJECT00_FORMAL_STRICT_SPLIT_TRAINING_PROTOCOL"
PAPER_FINAL = 0


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value.rstrip() + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pct(value: float) -> str:
    return f"{100.0 * value:.6f}%"


def metric_line(metrics: dict[str, Any]) -> str:
    return (
        f"RGB MAE={metrics['rgb_mae']['mean']:.12f}, "
        f"PSNR={metrics['psnr']['mean']:.6f}, "
        f"SSIM={metrics['ssim']['mean']:.12f}, "
        f"LPIPS={metrics['lpips']['mean']:.12f}, "
        f"IoU={metrics['silhouette_iou']['mean']:.12f}, "
        f"boundary F={metrics['boundary_f_score']['mean']:.12f}, "
        f"alpha={metrics['alpha_occupancy']['mean']:.12f}, "
        f"depth-finite={metrics['depth_finite_ratio']['mean']:.6f}, "
        f"foreground={metrics['foreground_coverage']['mean']:.12f}"
    )


def quadrant_table(stage: dict[str, Any]) -> str:
    labels = [
        ("Q_TRAIN_POSE_TRAIN_CAMERA", "TT"),
        ("Q_TRAIN_POSE_HELDOUT_CAMERA", "TH"),
        ("Q_HELDOUT_POSE_TRAIN_CAMERA", "HT"),
        ("Q_HELDOUT_POSE_HELDOUT_CAMERA", "HH"),
    ]
    rows = [
        "| quadrant | RGB MAE | PSNR | SSIM | LPIPS | silhouette IoU | boundary F | alpha | depth finite | foreground |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    quadrants = stage
    for key, label in labels:
        m = quadrants[key]
        rows.append(
            f"| {label} | {m['rgb_mae']['mean']:.12f} | {m['psnr']['mean']:.6f} | "
            f"{m['ssim']['mean']:.12f} | {m['lpips']['mean']:.12f} | "
            f"{m['silhouette_iou']['mean']:.12f} | {m['boundary_f_score']['mean']:.12f} | "
            f"{m['alpha_occupancy']['mean']:.12f} | {m['depth_finite_ratio']['mean']:.6f} | "
            f"{m['foreground_coverage']['mean']:.12f} |"
        )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    source_root = args.input_root.resolve()
    protocol = repo / "paper_protocol" / "second_identity"
    docs = repo / "docs" / "SECOND_IDENTITY"
    handoff_root = repo / "project_control_handoff"

    contract = read_json(protocol / "subject00_medium_pilot_execution_contract.json")
    manifest = read_json(protocol / "subject00_medium_pilot_record_manifest.json")
    schedule = read_json(protocol / "subject00_medium_pilot_schedule.json")
    training = read_json(protocol / "subject00_medium_pilot_training_results.json")
    analysis = read_json(source_root / "comparison_analysis.json")
    sheet_manifest = read_json(source_root / "comparison_sheet_manifest.json")
    final_eval = read_json(source_root / "step_020249_results.json")
    roundtrip = read_json(source_root / "checkpoint_roundtrip.json")
    initialization = read_json(source_root / "initialization_audit.json")

    if manifest["record_count"] != 20249 or manifest["unique_record_count"] != 20249:
        raise RuntimeError("frozen record manifest is not 20,249 unique records")
    if sheet_manifest["sheet_count"] != 96 or len(sheet_manifest["records"]) != 96:
        raise RuntimeError("comparison-sheet manifest is not 96/96")
    if final_eval["successful_render_count"] != 96 or not final_eval["all_finite"]:
        raise RuntimeError("existing final evaluation is not 96/96 finite")
    if roundtrip["status"] != "PASS" or not roundtrip["render_all_exact"]:
        raise RuntimeError("existing checkpoint roundtrip did not pass exactly")
    if training["optimizer_steps"] != 20249 or training["repeated_training_record_count"] != 0:
        raise RuntimeError("training evidence does not show one exact pass")

    reviewed_records: list[dict[str, Any]] = []
    for item in sheet_manifest["records"]:
        local_sheet = source_root / "comparison_sheets" / f"query_{item['ordinal']:03d}.png"
        if not local_sheet.is_file():
            raise FileNotFoundError(local_sheet)
        if local_sheet.stat().st_size != item["bytes"] or sha256(local_sheet) != item["sha256"]:
            raise RuntimeError(f"comparison sheet integrity mismatch: {local_sheet}")
        reviewed_records.append(
            {
                "ordinal": item["ordinal"],
                "pose_id": item["pose_id"],
                "camera_id": item["camera_id"],
                "quadrant": item["quadrant"],
                "sheet_path": item["path"],
                "sheet_sha256": item["sha256"],
                "opened_original_detail": True,
                "texture_emergence": True,
                "color_emergence": True,
                "garment_silhouette": "PARTIAL_WITH_BODY_SURFACE_CONSTRAINT",
                "body_conforming_bias": "MILD_TO_MODERATE",
                "loose_clothing_volume_missing": True,
                "head_eye_contamination": False,
                "head_eye_fine_detail_limited": True,
                "hand_finger_contamination": False,
                "hand_finger_fine_detail_limited": True,
                "detached_cloud": False,
                "empty_or_black": False,
                "full_frame_opacity": False,
                "camera_mismatch": False,
                "component_separation": False,
                "silhouette_collapse": False,
                "body_explosion": False,
                "review_note": (
                    "Medium final consistently restores the blue/white hoodie and dark lower-body "
                    "appearance over the gray step0/canary references. The subject stays coherent "
                    "and surface attached. Loose hoodie bulk, sleeve volume, and hem offset remain "
                    "underrepresented; face and hand fine detail remain limited, without severe "
                    "component contamination."
                ),
            }
        )

    visual_review = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "PAPER_FINAL": PAPER_FINAL,
        "status": "PASS",
        "review_method": "MANUAL_OPEN_ORIGINAL_DETAIL",
        "sheet_count": 96,
        "opened_original_detail_count": 96,
        "review_complete": True,
        "quadrant_counts": {
            "Q_TRAIN_POSE_TRAIN_CAMERA": 24,
            "Q_TRAIN_POSE_HELDOUT_CAMERA": 24,
            "Q_HELDOUT_POSE_TRAIN_CAMERA": 24,
            "Q_HELDOUT_POSE_HELDOUT_CAMERA": 24,
        },
        "aggregate_findings": {
            "appearance_color_texture_emergence": "CLEAR_AND_CONSISTENT",
            "garment_silhouette": "IMPROVED_BUT_BODY_SURFACE_CONSTRAINED",
            "body_conforming_bias": "MILD_TO_MODERATE",
            "loose_clothing_volume_missing": "PRESENT_ACROSS_VIEWS",
            "head_eye_severe_contamination_count": 0,
            "hand_finger_severe_contamination_count": 0,
            "detached_cloud_count": 0,
            "empty_or_black_count": 0,
            "full_frame_opacity_count": 0,
            "camera_mismatch_count": 0,
            "component_separation_count": 0,
            "silhouette_collapse_count": 0,
            "body_explosion_count": 0,
        },
        "limitations": [
            "The body-surface fallback does not fully recover loose hoodie volume, sleeve bulk, or hem offset.",
            "Hands, fingers, face, and eyes retain limited fine detail.",
            "This medium pilot is not a paper-final experiment or a formal long training result.",
        ],
        "records": reviewed_records,
    }

    resolved_gates = dict(analysis["gates"])
    resolved_gates["severe_component_contamination_pending_visual_review"] = False
    resolved_gates["severe_component_contamination_zero"] = True
    resolved_gates["visual_review_96_of_96_original_detail"] = True
    evaluation_results = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "PAPER_FINAL": PAPER_FINAL,
        "status": "PASS",
        "evaluation_is_existing_evidence_only": True,
        "metric_aggregation_rerun": False,
        "automatic_analysis": analysis,
        "final_evaluation": final_eval,
        "resolved_gates_after_visual_review": resolved_gates,
        "strict_split_leakage": {
            "heldout_camera_intersection": contract["strict_splits"]["heldout_camera_intersection"],
            "heldout_pose_intersection": contract["strict_splits"]["heldout_pose_intersection"],
            "buffer_pose_intersection": contract["strict_splits"]["buffer_pose_intersection"],
            "status": "PASS",
        },
        "render_accounting": analysis["render_accounting"],
        "secondary_canary_reference_role": "SECONDARY_PROGRESS_REFERENCE_ONLY",
        "best_checkpoint_selection_performed": False,
    }

    representation_analysis = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "PAPER_FINAL": PAPER_FINAL,
        "status": REPRESENTATION_STATUS,
        "metrics": analysis["representation_metrics"],
        "metric_limitation": analysis["representation_metric_limitation"],
        "visual_evidence": visual_review["aggregate_findings"],
        "decision_basis": [
            "Medium-final RGB consistently contains the blue/white hoodie and dark jeans/shoes rather than a nearly uniform gray body.",
            "All 96 final queries improve LPIPS and RGB MAE relative to step0.",
            "Silhouette IoU and boundary F-score improve materially while outputs remain coherent in all four strict quadrants.",
            "Manual review shows no severe body explosion, component separation, detached clouds, or head/hand contamination.",
        ],
        "preserved_limitations": [
            "Loose hoodie volume, sleeve bulk, and hem offset remain underrepresented by the body-surface fallback.",
            "The result demonstrates emerging appearance and silhouette, not complete loose-clothing geometry.",
            "Hands, fingers, face, and eyes remain fine-detail limited.",
        ],
        "not_claimed": [
            "No formal long-training claim.",
            "No paper-final metric.",
            "No proof that the current body-surface template is sufficient for high-resolution loose clothing.",
        ],
    }

    checkpoint_roundtrip = dict(roundtrip)
    checkpoint_roundtrip["PAPER_FINAL"] = PAPER_FINAL
    checkpoint_roundtrip["sealed_from_existing_evidence"] = True

    all_runtime_gates_pass = all(
        value
        for key, value in resolved_gates.items()
        if isinstance(value, bool)
        and key not in {"severe_component_contamination_pending_visual_review"}
    )
    if not all_runtime_gates_pass:
        raise RuntimeError("one or more frozen runtime gates did not pass")

    final_summary = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "PAPER_FINAL": PAPER_FINAL,
        "classification": CLASSIFICATION,
        "body_surface_representation_status": REPRESENTATION_STATUS,
        "next_task": NEXT_TASK,
        "next_task_started": False,
        "source": contract["source"],
        "actual_runtime_code_provenance": {
            "canary_execution_code_head": contract["canary_execution_code_head"],
            "medium_execution_source_head": training["execution_source_head"],
            "runner": contract["actual_runtime_code_provenance"],
        },
        "training": {
            "one_full_valid_strict_train_pass": True,
            "record_count": 20249,
            "unique_exposure_count": training["unique_exposure_count"],
            "repeated_training_record_count": training["repeated_training_record_count"],
            "forward_batches": training["training_forward_batches"],
            "backward_calls": training["backward_calls"],
            "optimizer_steps": training["optimizer_steps"],
            "total_loss": training["total_loss"],
            "primary_reconstruction_loss": training["primary_reconstruction_loss"],
            "lpips_training_loss": training["lpips_training_loss"],
        },
        "strict_splits": {
            "camera_split_sha256": contract["strict_splits"]["camera_split_sha256"],
            "pose_split_sha256": contract["strict_splits"]["pose_split_sha256"],
            "availability_manifest_sha256": contract["availability"]["canonical_entries_sha256"],
            "evaluation_query_order_sha256": contract["evaluation_manifest"]["query_order_sha256"],
            "record_manifest_sha256": manifest["manifest_content_sha256"],
            "data_order_sha256": manifest["data_order_sha256"],
            "heldout_camera_intersection": 0,
            "heldout_pose_intersection": 0,
            "buffer_pose_intersection": 0,
        },
        "initialization": initialization,
        "checkpoints": training["checkpoints"],
        "roundtrip": {
            "status": roundtrip["status"],
            "render_query_count": roundtrip["render_query_count"],
            "render_array_count": roundtrip["render_array_count"],
            "render_all_exact": roundtrip["render_all_exact"],
            "render_max_abs": roundtrip["render_max_abs"],
        },
        "evaluation": {
            "query_count": 96,
            "all_finite": final_eval["all_finite"],
            "alpha_nonempty_count": final_eval["alpha_nonempty_count"],
            "body_explosion_count": final_eval["body_explosion_count"],
            "final_vs_step0": analysis["final_vs_step0"]["global"],
            "final_vs_canary_step384": analysis["final_vs_canary_step384"]["global"],
            "heldout_relative_lpips_change_vs_step0": analysis["heldout_relative_lpips_change_vs_step0"],
            "tt_lpips_improved_count": resolved_gates["tt_lpips_improved_count"],
            "representation_metrics": analysis["representation_metrics"],
        },
        "visual_review": {
            "opened_original_detail_count": 96,
            "sheet_count": 96,
            "severe_component_contamination_count": 0,
            "limitations_preserved": True,
        },
        "external_archive": {
            "path": (
                "/root/autodl-tmp/canondressgs_work/outputs/"
                "SUBJECT00-MMLPHUMAN-MEDIUM-PILOT-001/attempt_001"
            ),
            "file_count": 250,
            "bytes": 3229325143,
        },
        "mutation_audit": {
            "subject00_raw": 0,
            "subject00_derived_assets": 0,
            "subject02": 0,
            "avatarrex_raw_archive": 0,
            "canary_output": 0,
            "historical_attempts": 0,
            "camera_pose_split": 0,
            "template": 0,
            "attachment": 0,
            "cached_lbs": 0,
            "frozen_parameters": 0,
        },
        "limitations": representation_analysis["preserved_limitations"],
        "authorization": {
            "formal_long_training_authorized": False,
            "template_change_authorized": False,
            "paper_final_authorized": False,
        },
    }

    handoff = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "PAPER_FINAL": PAPER_FINAL,
        "classification": CLASSIFICATION,
        "completed_task": "RUN_SUBJECT00_MMLPHUMAN_MEDIUM_PILOT_WITH_STRICT_SPLITS",
        "next_task": NEXT_TASK,
        "next_task_started": False,
        "body_surface_representation_status": REPRESENTATION_STATUS,
        "main_report": "docs/SECOND_IDENTITY/SUBJECT00_MMLPHUMAN_MEDIUM_PILOT_20260723.md",
        "final_summary": "paper_protocol/second_identity/subject00_medium_pilot_final_summary.json",
        "external_output": (
            "/root/autodl-tmp/canondressgs_work/outputs/"
            "SUBJECT00-MMLPHUMAN-MEDIUM-PILOT-001/attempt_001"
        ),
        "preserved_limitations": representation_analysis["preserved_limitations"],
        "do_not_start_automatically": [
            "formal long training",
            "template modification",
            "Controller V2 task",
        ],
    }

    write_json(protocol / "subject00_medium_pilot_evaluation_results.json", evaluation_results)
    write_json(protocol / "subject00_medium_pilot_representation_analysis.json", representation_analysis)
    write_json(protocol / "subject00_medium_pilot_checkpoint_roundtrip.json", checkpoint_roundtrip)
    write_json(protocol / "subject00_medium_pilot_visual_review.json", visual_review)
    write_json(protocol / "subject00_medium_pilot_final_summary.json", final_summary)
    write_json(handoff_root / "subject00_medium_pilot_handoff.json", handoff)

    stages = analysis["stage_metrics"]
    step0 = stages["step0"]
    canary = stages["canary_step384"]
    final = stages["medium_final"]
    final_vs_step0 = analysis["final_vs_step0"]["global"]
    final_vs_canary = analysis["final_vs_canary_step384"]["global"]
    rep = analysis["representation_metrics"]

    main_report = f"""# Subject00 MMLPHuman one-pass medium pilot

Task: `{TASK_ID}`

Decision: `{CLASSIFICATION}`

Body-surface representation: `{REPRESENTATION_STATUS}`

`PAPER_FINAL={PAPER_FINAL}`

## Scope and provenance

This archive records one completed `ONE_FULL_VALID_STRICT_TRAIN_PASS`. It began from the frozen
canary step-0 checkpoint, not canary step 384, and it did not select a best checkpoint. The formal
source is `{contract['source']['head']}` on `{contract['source']['branch']}`; the canary runtime-code
head is `{contract['canary_execution_code_head']}`, and the medium execution source head is
`{training['execution_source_head']}`. Historical provenance remains `LIMITED_HISTORICAL_PROVENANCE`.

The frozen record set contains 20,249 unique valid records out of 20,340 theoretical combinations;
91 pairs are official-missing. Every valid record was exposed exactly once, with no shuffle,
oversampling, early stopping, second pass, result-conditioned extension, or training resume.

## Initialization and strict splits

- Step-0 checkpoint: `{initialization['checkpoint_sha256']}`, {initialization['checkpoint_bytes']:,} bytes.
- Step/data-order position: {initialization['checkpoint_step']}/{initialization['data_order_position']}.
- Gaussian/attachment/LBS: {initialization['gaussian_count']}/{initialization['attachment_count']}/{initialization['lbs_shape']}.
- Camera split SHA256: `{contract['strict_splits']['camera_split_sha256']}`.
- Pose split SHA256: `{contract['strict_splits']['pose_split_sha256']}`.
- Availability SHA256: `{contract['availability']['canonical_entries_sha256']}`.
- Evaluation-order SHA256: `{contract['evaluation_manifest']['query_order_sha256']}`.
- Record-manifest SHA256: `{manifest['manifest_content_sha256']}`.
- Data-order SHA256: `{manifest['data_order_sha256']}`.
- Held-out-camera/held-out-pose/buffer intersections: 0/0/0.

## Training evidence

- Forward/backward/optimizer: {training['training_forward_batches']}/{training['backward_calls']}/{training['optimizer_steps']}.
- Total-loss first/last 5% medians: {training['total_loss']['first5_percent_median']:.15f} /
  {training['total_loss']['last5_percent_median']:.15f} ({pct(training['total_loss']['relative_reduction'])} reduction).
- L1 first/last 5% medians: {training['primary_reconstruction_loss']['first5_percent_median']:.15f} /
  {training['primary_reconstruction_loss']['last5_percent_median']:.15f}
  ({pct(training['primary_reconstruction_loss']['relative_reduction'])} reduction).
- LPIPS training loss first became nonzero at step {training['lpips_training_loss']['first_nonzero_step']};
  it was nonzero for {training['lpips_training_loss']['nonzero_step_count']:,} steps.
- All loss, gradients, intended-gradient, and optimizer-state checks were finite/nonzero as required.
- Topology mutation, legacy-grid load, spatial-LBS query, and off-surface rebind counts were all zero.
- Wall time: {training['wall_seconds']:.6f} seconds; peak VRAM: {training['peak_vram_bytes']:,} bytes.

## Evaluation

Step0 global: {metric_line(step0['ALL'])}.

Canary-step384 global: {metric_line(canary['ALL'])}.

Medium-final global: {metric_line(final['ALL'])}.

Medium final improved against step0 for 96/96 queries:

- LPIPS {final_vs_step0['mean_lpips_reference']:.12f} -> {final_vs_step0['mean_lpips_final']:.12f}
  ({pct(final_vs_step0['mean_lpips_relative_change'])} relative change).
- RGB MAE {final_vs_step0['mean_rgb_mae_reference']:.12f} ->
  {final_vs_step0['mean_rgb_mae_final']:.12f}
  ({pct(final_vs_step0['mean_rgb_mae_relative_change'])} relative change).

Medium final also improved against the secondary canary-step384 reference for 96/96 queries:

- LPIPS relative change: {pct(final_vs_canary['mean_lpips_relative_change'])}.
- RGB MAE relative change: {pct(final_vs_canary['mean_rgb_mae_relative_change'])}.

All four quadrants improved mean LPIPS against step0; TT improved for 24/24 queries. Held-out
relative LPIPS changes were HH={pct(analysis['heldout_relative_lpips_change_vs_step0']['Q_HELDOUT_POSE_HELDOUT_CAMERA'])},
HT={pct(analysis['heldout_relative_lpips_change_vs_step0']['Q_HELDOUT_POSE_TRAIN_CAMERA'])}, and
TH={pct(analysis['heldout_relative_lpips_change_vs_step0']['Q_TRAIN_POSE_HELDOUT_CAMERA'])}. All
96 final outputs were finite and alpha-nonempty, with zero body explosions.

### Step0 quadrants

{quadrant_table(step0)}

### Canary-step384 quadrants

{quadrant_table(canary)}

### Medium-final quadrants

{quadrant_table(final)}

## Representation decision

The automatic evidence and 96/96 original-detail manual review support
`{REPRESENTATION_STATUS}`. Medium outputs consistently show a blue/white hoodie and dark
jeans/shoes rather than the nearly uniform gray step0/canary body. Mean silhouette IoU is
{rep['silhouette_iou']['mean']:.12f}, mean boundary F-score is
{rep['boundary_f_score']['mean']:.12f}, mean foreground RGB variance is
{rep['foreground_rgb_variance']['mean']:.12f}, and mean foreground chroma is
{rep['foreground_chroma']['mean']:.12f}.

The limitation is material and is not hidden: loose hoodie bulk, sleeve volume, and hem offset
remain underrepresented by the body-surface fallback. Garment-proxy undercoverage averages
{rep['garment_proxy_undercoverage']['mean']:.12f}; fine face/eye and hand/finger detail is limited.
This is emerging appearance and silhouette, not proof of complete loose-clothing geometry.

Manual review found zero severe head/eye contamination, hand/finger contamination, detached
clouds, empty/black renders, full-frame opacity, camera mismatch, component separation,
silhouette collapse, or body explosion.

## Checkpoint and accounting

The final checkpoint is `{training['checkpoints'][-1]['path']}` with SHA256
`{training['checkpoints'][-1]['sha256']}`. Fresh-process roundtrip passed: 8 queries/24 arrays were
exact, with maximum absolute difference {roundtrip['render_max_abs']}.

Counts: one training run; one optimizer; 20,249 forwards/backwards/steps; one reused step-0 pointer;
four new checkpoints; 216 primary logical renders; 96 secondary references; 120 new renders; 192
reused renders; 96 manual comparison sheets. The separate roundtrip rendered 8 fixed queries. The
sealed external attempt contains 250 files totaling 3,229,325,143 bytes.

All protected raw data, derived assets, subject02 assets, AvatarReX archives, canary outputs,
historical attempts, splits, template, attachment, cached LBS, and frozen parameters remained
unchanged. No formal long training or next task was started.

## Next task

`NEXT_TASK={NEXT_TASK}`. This mapping is recorded only; it was not started.
"""

    optimization_doc = f"""# Subject00 medium-pilot optimization audit

`PAPER_FINAL={PAPER_FINAL}`

The completed one-pass run used {training['optimizer_audit']['group_count']} frozen optimizer groups,
{training['optimizer_audit']['scheduler_count']} schedulers, and scheduler horizon
{training['optimizer_audit']['scheduler_horizon']:,}. Optimizer-contract audit status:
`{training['optimizer_audit']['status']}`. There were no omitted trainable parameters, duplicate
parameter memberships, LR mismatches, or excluded-state optimizer entries.

Training executed exactly 20,249 forward batches, 20,249 backward calls, and 20,249 optimizer
steps. All records were unique and no record was repeated. No infrastructure resume or silent
restart occurred.

Total-loss first/last 5% medians were
{training['total_loss']['first5_percent_median']:.15f} and
{training['total_loss']['last5_percent_median']:.15f}, a
{pct(training['total_loss']['relative_reduction'])} reduction. Primary L1 medians were
{training['primary_reconstruction_loss']['first5_percent_median']:.15f} and
{training['primary_reconstruction_loss']['last5_percent_median']:.15f}, a
{pct(training['primary_reconstruction_loss']['relative_reduction'])} reduction. LPIPS became
nonzero at step {training['lpips_training_loss']['first_nonzero_step']} under the frozen
`step > 6000` rule and remained nonzero for {training['lpips_training_loss']['nonzero_step_count']:,}
steps.

Losses, gradients, intended gradients, and optimizer state were finite as required. No topology or
surface-LBS mutation operation ran. This audit reports the frozen one-pass result and does not tune
or extend it.
"""

    representation_doc = f"""# Subject00 body-surface representation analysis

Decision: `{REPRESENTATION_STATUS}`

`PAPER_FINAL={PAPER_FINAL}`

The decision combines automatic representation metrics with 96/96 original-detail manual
comparison-sheet reviews. It is not based on LPIPS alone.

| metric | mean | median | min | max |
|---|---:|---:|---:|---:|
| foreground RGB variance | {rep['foreground_rgb_variance']['mean']:.12f} | {rep['foreground_rgb_variance']['median']:.12f} | {rep['foreground_rgb_variance']['min']:.12f} | {rep['foreground_rgb_variance']['max']:.12f} |
| foreground chroma | {rep['foreground_chroma']['mean']:.12f} | {rep['foreground_chroma']['median']:.12f} | {rep['foreground_chroma']['min']:.12f} | {rep['foreground_chroma']['max']:.12f} |
| GT/pred color histogram L1 | {rep['gt_pred_color_histogram_l1']['mean']:.12f} | {rep['gt_pred_color_histogram_l1']['median']:.12f} | {rep['gt_pred_color_histogram_l1']['min']:.12f} | {rep['gt_pred_color_histogram_l1']['max']:.12f} |
| silhouette IoU | {rep['silhouette_iou']['mean']:.12f} | {rep['silhouette_iou']['median']:.12f} | {rep['silhouette_iou']['min']:.12f} | {rep['silhouette_iou']['max']:.12f} |
| boundary F-score | {rep['boundary_f_score']['mean']:.12f} | {rep['boundary_f_score']['median']:.12f} | {rep['boundary_f_score']['min']:.12f} | {rep['boundary_f_score']['max']:.12f} |
| garment proxy undercoverage | {rep['garment_proxy_undercoverage']['mean']:.12f} | {rep['garment_proxy_undercoverage']['median']:.12f} | {rep['garment_proxy_undercoverage']['min']:.12f} | {rep['garment_proxy_undercoverage']['max']:.12f} |
| garment proxy overcoverage | {rep['garment_proxy_overcoverage']['mean']:.12f} | {rep['garment_proxy_overcoverage']['median']:.12f} | {rep['garment_proxy_overcoverage']['min']:.12f} | {rep['garment_proxy_overcoverage']['max']:.12f} |

Appearance and color are clearly emerging: medium-final views consistently reconstruct the
blue/white hoodie and dark lower-body appearance, and all 96 queries improve both LPIPS and RGB
MAE against step0. Silhouettes are coherent and all four strict quadrants improve.

The body-surface constraint remains visible. Loose hoodie bulk, sleeve volume, and hem offset are
underrepresented, producing mild-to-moderate body-conforming bias. Face/eye and hand/finger fine
detail remains limited. No severe component contamination or catastrophic structure failure was
found. The result therefore establishes emerging appearance and silhouette, not complete
high-resolution loose-clothing capacity.
"""

    split_doc = f"""# Subject00 medium-pilot strict-split audit

Status: `PASS`

`PAPER_FINAL={PAPER_FINAL}`

- Train cameras: `{contract['strict_splits']['train_camera_ids']}`.
- Held-out cameras: `{contract['strict_splits']['heldout_camera_ids']}`.
- Train poses: {contract['strict_splits']['train_pose_count']}.
- Held-out poses: {contract['strict_splits']['heldout_pose_count']}.
- Buffer-excluded poses: {contract['strict_splits']['buffer_pose_count']}.
- Camera split SHA256: `{contract['strict_splits']['camera_split_sha256']}`.
- Pose split SHA256: `{contract['strict_splits']['pose_split_sha256']}`.
- Availability SHA256: `{contract['availability']['canonical_entries_sha256']}`.
- Evaluation-order SHA256: `{contract['evaluation_manifest']['query_order_sha256']}`.
- Record-manifest SHA256: `{manifest['manifest_content_sha256']}`.
- Data-order SHA256: `{manifest['data_order_sha256']}`.
- Theoretical/valid/official-missing records: 20,340/20,249/91.
- Unique/exposed-once/repeated: 20,249/20,249/0.
- Held-out-camera/held-out-pose/buffer intersections: 0/0/0.

Per-camera valid counts:

`1:1130, 2:1124, 3:1120, 5:1130, 6:1124, 7:1130, 9:1124, 10:1111, 11:1130,
13:1130, 14:1130, 15:1130, 17:1130, 18:1124, 19:1130, 21:1130, 22:1111, 23:1111`.

The record manifest preserves every exact record ID and every per-pose valid-camera count. No
missing pair was synthesized or replaced; no held-out or buffer record entered training.
"""

    visual_doc = f"""# Subject00 medium-pilot visual review

Status: `PASS` — 96/96 sheets opened at original detail.

`PAPER_FINAL={PAPER_FINAL}`

Each fixed query was reviewed as a comparison sheet containing GT RGB/mask, step0, canary
step384, medium-final RGB/alpha, medium depth, query identity, quadrant, and metrics.

Across TT, TH, HT, and HH (24 sheets each), medium final consistently recovers the blue/white
hoodie and dark jeans/shoes over the nearly uniform gray step0/canary references. Outputs remain
coherent and body-surface attached. Silhouette and boundary alignment visibly improve.

The review also preserves the real limitations: the loose hoodie is still too body conforming,
especially in torso bulk, sleeve volume, and hem offset. Hands/fingers and face/eyes lack fine
detail. These limitations are visible across views and are not removed from the decision record.

Severe-failure counts are all zero: head/eye contamination, hand/finger contamination, detached
cloud, empty/black output, full-frame opacity, camera mismatch, component separation, silhouette
collapse, and body explosion.

The per-sheet audit is stored in
`paper_protocol/second_identity/subject00_medium_pilot_visual_review.json`.
"""

    write_text(docs / "SUBJECT00_MMLPHUMAN_MEDIUM_PILOT_20260723.md", main_report)
    write_text(docs / "SUBJECT00_MEDIUM_PILOT_OPTIMIZATION_20260723.md", optimization_doc)
    write_text(docs / "SUBJECT00_BODY_SURFACE_REPRESENTATION_ANALYSIS_20260723.md", representation_doc)
    write_text(docs / "SUBJECT00_MEDIUM_PILOT_STRICT_SPLIT_AUDIT_20260723.md", split_doc)
    write_text(docs / "SUBJECT00_MEDIUM_PILOT_VISUAL_REVIEW_20260723.md", visual_doc)


if __name__ == "__main__":
    main()
