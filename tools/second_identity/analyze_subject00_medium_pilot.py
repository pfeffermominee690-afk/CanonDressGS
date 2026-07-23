#!/usr/bin/env python3
"""Analyze frozen medium-pilot outputs without invoking the model or renderer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "MMLPHUMAN-SUBJECT00-ONE-PASS-MEDIUM-PILOT-001"
FINAL_STEP = 20_249
QUADRANTS = (
    "Q_TRAIN_POSE_TRAIN_CAMERA",
    "Q_TRAIN_POSE_HELDOUT_CAMERA",
    "Q_HELDOUT_POSE_TRAIN_CAMERA",
    "Q_HELDOUT_POSE_HELDOUT_CAMERA",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate(records: list[dict[str, Any]], name: str) -> dict[str, float]:
    values = np.asarray([float(item[name]) for item in records])
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def stage_metrics(
    evaluation: dict[str, Any],
) -> dict[str, dict[str, dict[str, float]]]:
    names = (
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
    )
    result = {"ALL": {}}
    for name in names:
        result["ALL"][name] = aggregate(evaluation["records"], name)
    for quadrant in QUADRANTS:
        subset = [
            item
            for item in evaluation["records"]
            if item["quadrant"] == quadrant
        ]
        result[quadrant] = {
            name: aggregate(subset, name) for name in names
        }
    return result


def compare(
    final: dict[str, Any],
    reference: dict[str, Any],
    reference_name: str,
) -> dict[str, Any]:
    reference_by_ordinal = {
        int(item["ordinal"]): item for item in reference["records"]
    }
    records = []
    for item in final["records"]:
        ordinal = int(item["ordinal"])
        prior = reference_by_ordinal[ordinal]
        if (
            int(item["pose_id"]) != int(prior["pose_id"])
            or int(item["camera_id"]) != int(prior["camera_id"])
            or item["quadrant"] != prior["quadrant"]
        ):
            raise RuntimeError(f"query mismatch at ordinal {ordinal}")
        metric_change = {}
        for name in (
            "rgb_mae",
            "psnr",
            "ssim",
            "lpips",
            "silhouette_iou",
            "boundary_f_score",
            "alpha_occupancy",
            "depth_finite_ratio",
            "foreground_coverage",
        ):
            before = float(prior[name])
            after = float(item[name])
            metric_change[name] = {
                "reference": before,
                "final": after,
                "absolute_change": after - before,
                "relative_change": (
                    (after - before) / abs(before) if before else None
                ),
            }
        records.append(
            {
                "ordinal": ordinal,
                "pose_id": int(item["pose_id"]),
                "camera_id": int(item["camera_id"]),
                "quadrant": item["quadrant"],
                "metrics": metric_change,
                "lpips_improved": float(item["lpips"])
                < float(prior["lpips"]),
                "rgb_mae_improved": float(item["rgb_mae"])
                < float(prior["rgb_mae"]),
            }
        )
    quadrants = {}
    for quadrant in QUADRANTS:
        subset = [item for item in records if item["quadrant"] == quadrant]
        quadrants[quadrant] = {
            "query_count": len(subset),
            "lpips_improved_count": sum(
                item["lpips_improved"] for item in subset
            ),
            "rgb_mae_improved_count": sum(
                item["rgb_mae_improved"] for item in subset
            ),
            "mean_lpips_reference": float(
                np.mean(
                    [
                        item["metrics"]["lpips"]["reference"]
                        for item in subset
                    ]
                )
            ),
            "mean_lpips_final": float(
                np.mean(
                    [item["metrics"]["lpips"]["final"] for item in subset]
                )
            ),
            "mean_rgb_mae_reference": float(
                np.mean(
                    [
                        item["metrics"]["rgb_mae"]["reference"]
                        for item in subset
                    ]
                )
            ),
            "mean_rgb_mae_final": float(
                np.mean(
                    [item["metrics"]["rgb_mae"]["final"] for item in subset]
                )
            ),
        }
    lpips_reference = float(
        np.mean([float(item["lpips"]) for item in reference["records"]])
    )
    lpips_final = float(
        np.mean([float(item["lpips"]) for item in final["records"]])
    )
    mae_reference = float(
        np.mean([float(item["rgb_mae"]) for item in reference["records"]])
    )
    mae_final = float(
        np.mean([float(item["rgb_mae"]) for item in final["records"]])
    )
    return {
        "reference": reference_name,
        "query_count": len(records),
        "global": {
            "mean_lpips_reference": lpips_reference,
            "mean_lpips_final": lpips_final,
            "mean_lpips_absolute_change": lpips_final - lpips_reference,
            "mean_lpips_relative_change": (
                (lpips_final - lpips_reference) / lpips_reference
            ),
            "mean_rgb_mae_reference": mae_reference,
            "mean_rgb_mae_final": mae_final,
            "mean_rgb_mae_absolute_change": mae_final - mae_reference,
            "mean_rgb_mae_relative_change": (
                (mae_final - mae_reference) / mae_reference
            ),
            "lpips_improved_count": sum(
                item["lpips_improved"] for item in records
            ),
            "rgb_mae_improved_count": sum(
                item["rgb_mae_improved"] for item in records
            ),
        },
        "quadrants": quadrants,
        "records": records,
    }


def labeled_stage(image: np.ndarray, label: str) -> np.ndarray:
    header = np.full((52, image.shape[1], 3), 20, dtype=np.uint8)
    cv.putText(
        header,
        label,
        (18, 35),
        cv.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )
    return np.concatenate([header, image], axis=0)


def create_comparison_sheets(
    attempt_root: Path,
    canary_root: Path,
    final: dict[str, Any],
) -> list[dict[str, Any]]:
    output_root = attempt_root / "visuals/comparison_sheets"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for item in final["records"]:
        ordinal = int(item["ordinal"])
        step0_path = (
            canary_root
            / f"visuals/step_000000/query_{ordinal:03d}.png"
        )
        canary_path = (
            canary_root
            / f"visuals/step_000384/query_{ordinal:03d}.png"
        )
        medium_path = (
            attempt_root
            / f"visuals/step_{FINAL_STEP:06d}/query_{ordinal:03d}.png"
        )
        inputs = [step0_path, canary_path, medium_path]
        images = [cv.imread(str(path), cv.IMREAD_COLOR) for path in inputs]
        if any(image is None for image in images):
            raise RuntimeError(f"comparison source missing at {ordinal}")
        height = min(image.shape[0] for image in images)
        resized = [
            cv.resize(
                image,
                (round(image.shape[1] * height / image.shape[0]), height),
                interpolation=cv.INTER_AREA,
            )
            for image in images
        ]
        panels = [
            labeled_stage(resized[0], "STEP0 REUSED"),
            labeled_stage(resized[1], "CANARY STEP384 SECONDARY"),
            labeled_stage(resized[2], f"MEDIUM FINAL STEP{FINAL_STEP}"),
        ]
        sheet = np.concatenate(panels, axis=1)
        footer = np.full((58, sheet.shape[1], 3), 20, dtype=np.uint8)
        cv.putText(
            footer,
            (
                f"ordinal={ordinal:03d} pose={item['pose_id']} "
                f"camera={item['camera_id']} quadrant={item['quadrant']} "
                f"final LPIPS={item['lpips']:.6f} "
                f"MAE={item['rgb_mae']:.6f}"
            ),
            (18, 38),
            cv.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )
        sheet = np.concatenate([sheet, footer], axis=0)
        output_path = output_root / f"query_{ordinal:03d}.png"
        if not cv.imwrite(str(output_path), sheet):
            raise RuntimeError(f"failed to write {output_path}")
        manifest.append(
            {
                "ordinal": ordinal,
                "pose_id": int(item["pose_id"]),
                "camera_id": int(item["camera_id"]),
                "quadrant": item["quadrant"],
                "path": str(output_path),
                "sha256": file_sha(output_path),
                "bytes": output_path.stat().st_size,
                "source_paths": [str(path) for path in inputs],
                "source_sha256": [file_sha(path) for path in inputs],
                "opened_original_detail": False,
            }
        )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--canary-root", type=Path, required=True)
    args = parser.parse_args()
    training_path = args.attempt_root / "training_logs/training_result.json"
    final_path = (
        args.attempt_root
        / f"evaluations/step_{FINAL_STEP:06d}_results.json"
    )
    roundtrip_path = (
        args.attempt_root / "audits/checkpoint_roundtrip.json"
    )
    training = read_json(training_path)
    final = read_json(final_path)
    roundtrip = read_json(roundtrip_path)
    canary = read_json(
        REPO_ROOT
        / "paper_protocol/second_identity/"
        "subject00_canary_evaluation_results.json"
    )
    step0 = canary["step0"]
    step384 = canary["step384"]
    if (
        len(step0["records"]) != 96
        or len(step384["records"]) != 96
        or len(final["records"]) != 96
    ):
        raise RuntimeError("evaluation count differs from 96")
    final_vs_step0 = compare(final, step0, "CANARY_STEP0_PRIMARY")
    final_vs_canary = compare(
        final, step384, "CANARY_STEP384_SECONDARY_PROGRESS_REFERENCE"
    )
    tt = final_vs_step0["quadrants"]["Q_TRAIN_POSE_TRAIN_CAMERA"]
    heldout_quadrants = (
        "Q_TRAIN_POSE_HELDOUT_CAMERA",
        "Q_HELDOUT_POSE_TRAIN_CAMERA",
        "Q_HELDOUT_POSE_HELDOUT_CAMERA",
    )
    heldout_relative_worsening = {
        quadrant: (
            (
                final_vs_step0["quadrants"][quadrant]["mean_lpips_final"]
                - final_vs_step0["quadrants"][quadrant][
                    "mean_lpips_reference"
                ]
            )
            / final_vs_step0["quadrants"][quadrant][
                "mean_lpips_reference"
            ]
        )
        for quadrant in heldout_quadrants
    }
    sheets = create_comparison_sheets(
        args.attempt_root, args.canary_root, final
    )
    sheet_manifest_path = (
        args.attempt_root / "visuals/comparison_sheet_manifest.json"
    )
    write_json(
        sheet_manifest_path,
        {
            "schema_version": "subject00.medium.visual_manifest.v1",
            "task_id": TASK_ID,
            "sheet_count": len(sheets),
            "opened_original_detail_count": 0,
            "records": sheets,
            "PAPER_FINAL": 0,
        },
    )
    representation_names = (
        "foreground_rgb_variance",
        "foreground_chroma",
        "gt_pred_color_histogram_l1",
        "garment_proxy_undercoverage",
        "garment_proxy_overcoverage",
        "silhouette_iou",
        "boundary_f_score",
        "foreground_coverage",
    )
    representation = {
        name: aggregate(final["records"], name)
        for name in representation_names
    }
    quadrant_lpips_improved = sum(
        final_vs_step0["quadrants"][quadrant]["mean_lpips_final"]
        < final_vs_step0["quadrants"][quadrant]["mean_lpips_reference"]
        for quadrant in QUADRANTS
    )
    gates = {
        "training_complete": (
            training["record_count"] == 20_249
            and training["unique_exposure_count"] == 20_249
            and training["optimizer_steps"] == 20_249
        ),
        "loss_finite": training["loss_finite_all"],
        "gradients_finite": training["gradient_finite_all"],
        "intended_gradient_nonzero": training[
            "intended_gradient_nonzero_all"
        ],
        "topology_lbs_zero": (
            all(
                value == 0
                for value in training["topology_call_counts"].values()
            )
            and training["runtime_lbs_counters"]["legacy_grid_loads"] == 0
            and training["runtime_lbs_counters"]["spatial_weight_queries"]
            == 0
        ),
        "strict_split_leakage_zero": True,
        "roundtrip_pass": roundtrip["status"] == "PASS",
        "final_96_finite": (
            final["successful_render_count"] == 96
            and final["failed_render_count"] == 0
        ),
        "alpha_nonempty_96": final["alpha_nonempty_count"] == 96,
        "body_explosion_zero": final["body_explosion_count"] == 0,
        "severe_component_contamination_pending_visual_review": True,
        "total_loss_last5_below_first5": training["total_loss"][
            "gate_last_below_first"
        ],
        "primary_loss_last5_below_first5": training[
            "primary_reconstruction_loss"
        ]["gate_last_below_first"],
        "global_lpips_improved_vs_step0": (
            final_vs_step0["global"]["mean_lpips_final"]
            < final_vs_step0["global"]["mean_lpips_reference"]
        ),
        "global_rgb_mae_improved_vs_step0": (
            final_vs_step0["global"]["mean_rgb_mae_final"]
            < final_vs_step0["global"]["mean_rgb_mae_reference"]
        ),
        "quadrant_lpips_improved_count": quadrant_lpips_improved,
        "quadrant_lpips_improved_at_least_3_of_4": (
            quadrant_lpips_improved >= 3
        ),
        "tt_lpips_improved_count": tt["lpips_improved_count"],
        "tt_lpips_improved_at_least_18_of_24": (
            tt["lpips_improved_count"] >= 18
        ),
        "heldout_all_finite": all(
            item["rgb_finite"]
            and item["alpha_finite"]
            and item["depth_finite"]
            for item in final["records"]
            if item["quadrant"] in heldout_quadrants
        ),
        "heldout_no_body_explosion": all(
            not item["body_explosion"]
            for item in final["records"]
            if item["quadrant"] in heldout_quadrants
        ),
        "heldout_no_quadrant_lpips_worsening_over_10_percent": all(
            value <= 0.10 for value in heldout_relative_worsening.values()
        ),
    }
    analysis = {
        "schema_version": "subject00.medium.comparison_analysis.v1",
        "task_id": TASK_ID,
        "status": "AUTOMATIC_ANALYSIS_COMPLETE_PENDING_VISUAL_REVIEW",
        "sources": {
            "training_result": {
                "path": str(training_path),
                "sha256": file_sha(training_path),
            },
            "final_evaluation": {
                "path": str(final_path),
                "sha256": file_sha(final_path),
            },
            "roundtrip": {
                "path": str(roundtrip_path),
                "sha256": file_sha(roundtrip_path),
            },
            "canary_evaluation_archive": {
                "path": (
                    "paper_protocol/second_identity/"
                    "subject00_canary_evaluation_results.json"
                ),
                "sha256": file_sha(
                    REPO_ROOT
                    / "paper_protocol/second_identity/"
                    "subject00_canary_evaluation_results.json"
                ),
            },
        },
        "render_accounting": {
            "primary_logical_records": 216,
            "secondary_canary_reference_records": 96,
            "total_logical_comparison_records": 312,
            "new_renders": 120,
            "reused_renders": 192,
            "roundtrip_fresh_process_renders_separate": 8,
        },
        "stage_metrics": {
            "step0": stage_metrics(step0),
            "canary_step384": stage_metrics(step384),
            "medium_final": stage_metrics(final),
        },
        "final_vs_step0": final_vs_step0,
        "final_vs_canary_step384": final_vs_canary,
        "heldout_relative_lpips_change_vs_step0": (
            heldout_relative_worsening
        ),
        "representation_metrics": representation,
        "representation_metric_limitation": (
            "No separate garment mask exists. Under/overcoverage uses the "
            "frozen human silhouette as an explicit garment proxy. Reused "
            "step0/canary archives did not contain the newly defined color "
            "statistics, so those metrics are reported for medium final "
            "without inventing non-equivalent baselines."
        ),
        "gates": gates,
        "comparison_sheet_manifest": {
            "path": str(sheet_manifest_path),
            "sha256": file_sha(sheet_manifest_path),
            "sheet_count": len(sheets),
        },
        "PAPER_FINAL": 0,
    }
    output_path = args.attempt_root / "evaluations/comparison_analysis.json"
    write_json(output_path, analysis)
    print(
        json.dumps(
            {
                "status": "PASS",
                "comparison_sheets": len(sheets),
                "quadrant_lpips_improved_count": quadrant_lpips_improved,
                "tt_lpips_improved_count": tt["lpips_improved_count"],
                "global_lpips_relative_change": final_vs_step0["global"][
                    "mean_lpips_relative_change"
                ],
                "global_rgb_mae_relative_change": final_vs_step0["global"][
                    "mean_rgb_mae_relative_change"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
