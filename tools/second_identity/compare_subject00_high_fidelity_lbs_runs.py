#!/usr/bin/env python3
"""Compare two fresh high-fidelity LBS prototype runs and apply frozen gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def fidelity_pass(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["mae"] <= 1.0e-8
        and metrics["max_abs"] <= 1.0e-7
        and metrics["dominant_agreement"] == 1.0
        and metrics["weight_sum_max_abs_error"] <= 1.0e-7
        and metrics["finite"]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--external-output", type=Path, required=True)
    parser.add_argument("--surface-output", type=Path, required=True)
    parser.add_argument("--narrow-output", type=Path, required=True)
    parser.add_argument("--densification-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    manifest_a_path = args.run_a / "deterministic_manifest.json"
    manifest_b_path = args.run_b / "deterministic_manifest.json"
    manifest_a_bytes = manifest_a_path.read_bytes()
    manifest_b_bytes = manifest_b_path.read_bytes()
    manifest_a = json.loads(manifest_a_bytes)
    manifest_b = json.loads(manifest_b_bytes)
    attachment_a_path = args.run_a / "surface_attachments.npz"
    attachment_b_path = args.run_b / "surface_attachments.npz"
    checkpoint_a_path = args.run_a / "densification_checkpoint.npz"
    checkpoint_b_path = args.run_b / "densification_checkpoint.npz"

    arrays = {}
    with np.load(attachment_a_path, allow_pickle=False) as left, np.load(attachment_b_path, allow_pickle=False) as right:
        if left.files != right.files:
            raise RuntimeError(f"Attachment key order differs: {left.files} != {right.files}")
        for key in left.files:
            a = np.asarray(left[key])
            b = np.asarray(right[key])
            exact = bool(np.array_equal(a, b))
            maximum = 0.0 if exact else float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
            arrays[key] = {
                "shape": list(a.shape),
                "dtype": str(a.dtype),
                "exact": exact,
                "max_abs": maximum,
            }
    checkpoint_arrays_exact = True
    checkpoint_array_records = {}
    with np.load(checkpoint_a_path, allow_pickle=False) as left, np.load(checkpoint_b_path, allow_pickle=False) as right:
        if left.files != right.files:
            checkpoint_arrays_exact = False
        for key in sorted(set(left.files) & set(right.files)):
            exact = bool(np.array_equal(left[key], right[key]))
            checkpoint_array_records[key] = exact
            checkpoint_arrays_exact &= exact

    determinism = {
        "manifest_bytes_exact": manifest_a_bytes == manifest_b_bytes,
        "manifest_sha256_a": sha256_file(manifest_a_path),
        "manifest_sha256_b": sha256_file(manifest_b_path),
        "attachment_npz_bytes_exact": attachment_a_path.read_bytes() == attachment_b_path.read_bytes(),
        "attachment_npz_sha256_a": sha256_file(attachment_a_path),
        "attachment_npz_sha256_b": sha256_file(attachment_b_path),
        "checkpoint_npz_bytes_exact": checkpoint_a_path.read_bytes() == checkpoint_b_path.read_bytes(),
        "checkpoint_npz_sha256_a": sha256_file(checkpoint_a_path),
        "checkpoint_npz_sha256_b": sha256_file(checkpoint_b_path),
        "arrays": arrays,
        "checkpoint_arrays_exact": checkpoint_arrays_exact,
        "checkpoint_array_records": checkpoint_array_records,
    }
    determinism["pass"] = bool(
        determinism["manifest_bytes_exact"]
        and determinism["attachment_npz_bytes_exact"]
        and determinism["checkpoint_npz_bytes_exact"]
        and all(record["exact"] for record in arrays.values())
        and checkpoint_arrays_exact
    )

    vertex_metrics = manifest_a["vertices"]["surface_attachment"]
    surface_metrics = manifest_a["surface_samples"]["surface_attachment"]
    leakage = manifest_a["surface_samples"]["leakage"]
    vertex_gate = fidelity_pass(vertex_metrics) and manifest_a["vertices"]["vertex_weight_bitwise_exact"]
    surface_gate = fidelity_pass(surface_metrics)
    surface_leakage_gate = bool(
        leakage["head_eye_cross_component_or_region"] == 0
        and leakage["hand_finger_cross_region"] == 0
    )

    narrow = manifest_a["narrow_band"]
    total_head_eye_switch = sum(int(record["head_eye_component_switch_count"]) for record in narrow)
    total_component_switch = sum(int(record["component_switch_count"]) for record in narrow)
    total_hand_region_switch = sum(int(record["hand_region_switch_count"]) for record in narrow)
    narrow_gate = total_head_eye_switch == 0
    tie_gate = bool(manifest_a["closest_triangle_tie_test"]["face_ids_exact_to_min_incident_face"])

    densification = manifest_a["densification"]
    densification_gate = bool(
        densification["clone"]["face_ids_exact"]
        and densification["clone"]["barycentric_exact"]
        and densification["clone"]["components_exact"]
        and densification["clone"]["regions_exact"]
        and densification["clone"]["weights_exact"]
        and densification["split_small"]["parent_face_exact"]
        and densification["split_small"]["component_switch_count"] == 0
        and densification["split_small"]["formula_max_abs"] <= 1.0e-8
        and densification["prune"]["metadata_index_parity"]
        and densification["checkpoint_roundtrip_arrays_exact"]
        and checkpoint_arrays_exact
    )

    pose_records = manifest_a["poses"]
    pose_gate_records = []
    for record in pose_records:
        metrics = record["surface_vs_official_runtime_lbs"]
        passed = bool(
            metrics["vertex_mae"] <= 1.0e-8
            and metrics["vertex_max_abs"] <= 1.0e-6
            and metrics["finite"]
            and metrics["flipped_faces"] == 0
        )
        pose_gate_records.append({"label": record["label"], "pass": passed, "metrics": metrics})
    pose_gate = all(record["pass"] for record in pose_gate_records) and len(pose_gate_records) == 9

    gaussian_200k = manifest_a["gaussian_200k"]
    gaussian_200k_gate = bool(
        gaussian_200k["count"] == 200000
        and gaussian_200k["attachment_coverage"] == 1.0
        and gaussian_200k["off_surface_fallback"] == 0
    )
    runtime_compatible = bool(
        manifest_a["runtime_gate"]["current_create_from_pcd_accepts_fixed_weights"]
        and manifest_a["runtime_gate"]["current_runtime_persists_face_barycentric_semantic_metadata"]
    )

    if not (vertex_gate and surface_gate and surface_leakage_gate and pose_gate):
        classification = "BODY_SURFACE_FALLBACK_INSUFFICIENT"
        next_task = "AUDIT_ANIMATABLE_GAUSSIANS_HIGH_RESOLUTION_TEMPLATE_FOR_SUBJECT00"
    elif not (narrow_gate and tie_gate and densification_gate):
        classification = "CLOSEST_TRIANGLE_FALLBACK_INSUFFICIENT"
        next_task = "DESIGN_REGION_GATED_OFF_SURFACE_LBS_FALLBACK"
    elif not runtime_compatible:
        classification = "PER_GAUSSIAN_ATTACHMENT_RUNTIME_REPAIR_REQUIRED"
        next_task = "IMPLEMENT_MMLPHUMAN_SURFACE_ATTACHED_LBS_RUNTIME"
    elif determinism["pass"] and gaussian_200k_gate:
        classification = "HIGH_FIDELITY_LBS_CONTRACT_READY"
        next_task = "IMPLEMENT_AND_BUILD_SUBJECT00_HIGH_FIDELITY_LBS_ASSETS"
    else:
        classification = "HIGH_FIDELITY_LBS_CONTRACT_INCONCLUSIVE"
        next_task = "MANUAL_REVIEW_MMLPHUMAN_LBS_RUNTIME_CONSUMERS"

    surface_result = {
        "schema_version": "subject00.mmlphuman.surface_attachment_results.v1",
        "status": "PASS" if vertex_gate and surface_gate and surface_leakage_gate else "FAIL",
        "reference": manifest_a["reference"],
        "semantic": manifest_a["semantic"],
        "determinism": determinism,
        "vertices": manifest_a["vertices"],
        "surface_samples": manifest_a["surface_samples"],
        "poses": pose_records,
        "pose_gate_records": pose_gate_records,
        "gaussian_200k": gaussian_200k,
        "legacy_vs_surface": {
            "vertex_legacy": manifest_a["vertices"]["legacy_grid"],
            "vertex_surface": vertex_metrics,
            "sample_legacy": manifest_a["surface_samples"]["legacy_grid"],
            "sample_surface": surface_metrics,
            "legacy_head_eye_dominant_region_mismatch": leakage["legacy_head_eye_dominant_region_mismatch"],
            "surface_head_eye_leakage": leakage["head_eye_cross_component_or_region"],
            "legacy_hand_dominant_region_mismatch": leakage["legacy_hand_dominant_region_mismatch"],
            "surface_hand_leakage": leakage["hand_finger_cross_region"],
        },
        "gates": {
            "vertex_fidelity": vertex_gate,
            "surface_sample_fidelity": surface_gate,
            "surface_leakage": surface_leakage_gate,
            "pose_parity": pose_gate,
            "gaussian_200k_attachment": gaussian_200k_gate,
        },
    }
    narrow_result = {
        "schema_version": "subject00.mmlphuman.narrow_band_lbs_results.v1",
        "status": "PASS" if narrow_gate and tie_gate else "FAIL",
        "distances": narrow,
        "closest_triangle_tie_test": manifest_a["closest_triangle_tie_test"],
        "aggregate": {
            "component_switch_count": total_component_switch,
            "head_eye_component_switch_count": total_head_eye_switch,
            "hand_region_switch_count": total_hand_region_switch,
        },
        "gates": {
            "head_eye_cross_component_zero": narrow_gate,
            "tie_rule": tie_gate,
            "two_run_determinism": determinism["pass"],
        },
    }
    densification_result = {
        "schema_version": "subject00.mmlphuman.densification_lbs_results.v1",
        "status": "PASS" if densification_gate else "FAIL",
        "results": densification,
        "two_run_checkpoint_arrays_exact": checkpoint_arrays_exact,
        "gate": densification_gate,
    }
    summary = {
        "task_id": "MMLPHUMAN-SUBJECT00-HIGH-FIDELITY-LBS-DESIGN-001",
        "status": "COMPLETE_NO_TRAINING_NO_PUBLICATION",
        "source_head": "f88cc6a798747ad93ab9e6fe26f92c3e23229c9c",
        "candidate_results": {
            "LEGACY_POINTINTERPOLANT_GRID": "FAILURE_BASELINE_RETAINED",
            "SURFACE_FACE_ATTACHMENT": surface_result["status"],
            "DETERMINISTIC_CLOSEST_TRIANGLE": narrow_result["status"],
            "HYBRID_SURFACE_ATTACHED_LBS": classification,
        },
        "gates": {
            "vertex_fidelity": vertex_gate,
            "surface_sample_fidelity": surface_gate,
            "surface_leakage": surface_leakage_gate,
            "determinism": determinism["pass"],
            "narrow_band": narrow_gate,
            "closest_triangle_tie": tie_gate,
            "densification": densification_gate,
            "pose_parity": pose_gate,
            "gaussian_200k_attachment": gaussian_200k_gate,
            "runtime_consumer_compatible_without_repair": runtime_compatible,
            "renderer_smoke": False,
        },
        "renderer_smoke": "NOT_RUN_RUNTIME_INTERFACE_PRECONDITION_FAILED",
        "publication": "NOT_RUN_DESIGN_TASK_ONLY",
        "formal_target_exists": False,
        "counters": {
            "training_steps": 0,
            "training_forward_batches": 0,
            "backward_calls": 0,
            "optimizer_created": 0,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "training_checkpoint_writes": 0,
            "PAPER_FINAL": 0,
        },
        "classification": classification,
        "next_task": next_task,
    }
    comparison = {
        "run_a": str(args.run_a),
        "run_b": str(args.run_b),
        "determinism": determinism,
        "surface_result": surface_result,
        "narrow_result": narrow_result,
        "densification_result": densification_result,
        "summary": summary,
    }
    write_json(args.external_output, comparison)
    write_json(args.surface_output, surface_result)
    write_json(args.narrow_output, narrow_result)
    write_json(args.densification_output, densification_result)
    write_json(args.summary_output, summary)
    print(json.dumps({"classification": classification, "next_task": next_task, "gates": summary["gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
