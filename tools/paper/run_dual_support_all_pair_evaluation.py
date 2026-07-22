"""Inference-only all-pair evaluation of the frozen dual-support geometry blend."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from tools.paper import run_continuous_control_artifact_root_cause as provenance_tools
from tools.paper import run_geometry_dual_support_micro_pilot as micro
from tools.paper.verify_seen_outfit_paper_assets import verify_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "AAAI27-DUAL-SUPPORT-ALL-PAIR-EVALUATION-001"
SOURCE_HEAD = "b216acd02323c822a7aca12f424efed6ba2e0a81"
RUN_BRANCH = "research/dual-support-all-pair-evaluation-20260722"
LABEL = "RESEARCH EVALUATION — NOT PAPER FINAL"
PROTOCOL_PATH = PROJECT_ROOT / "paper_protocol/reviewer_risk/dual_support_all_pair_protocol.yaml"
PROTOCOL_SHA256 = "34f7e7cd45f3a9d5cb49323f9981936104e11e2b23b1a9f88461524d9b0a8d2e"
MICRO_REPORT = PROJECT_ROOT / "docs/PAPER/AAAI27_GEOMETRY_DUAL_SUPPORT_MICRO_PILOT_20260722.md"
MICRO_PROTOCOL = PROJECT_ROOT / "paper_protocol/reviewer_risk/geometry_dual_support_micro_pilot_protocol.yaml"
MICRO_RESULTS = PROJECT_ROOT / "paper_protocol/reviewer_risk/geometry_dual_support_micro_pilot_results.json"
MICRO_REVIEW = PROJECT_ROOT / "paper_protocol/reviewer_risk/geometry_dual_support_micro_pilot_visual_review.json"
MICRO_SUMMARY = PROJECT_ROOT / "paper_protocol/reviewer_risk/geometry_dual_support_micro_pilot_final_summary.json"
MICRO_OUTPUT = Path("/root/autodl-tmp/canondressgs_work/outputs/GEOMETRY-DUAL-SUPPORT-MICRO-PILOT-001")
PAIRS = (
    ("O01", "O02"), ("O01", "O03"), ("O01", "O04"), ("O01", "O08"),
    ("O02", "O03"), ("O02", "O04"), ("O02", "O08"),
    ("O03", "O04"), ("O03", "O08"), ("O04", "O08"),
)
PAIR_IDS = tuple(f"{left}_{right}" for left, right in PAIRS)
VISUAL_CATEGORIES = (
    "patch", "cloud", "mottle", "edge_scatter", "silhouette_discontinuity",
    "full_body_contamination", "identity_contamination", "double_outline", "ghosting",
)
CORE_CATEGORIES = VISUAL_CATEGORIES[:6]
EXPECTED_LOGICAL = len(PAIRS) * len(micro.DIRECTIONS) * len(micro.ALPHAS) * len(micro.CONDITIONS)
EXPECTED_MAIN_SHEETS = len(PAIRS) * len(micro.DIRECTIONS)
EXPECTED_VISUALS = EXPECTED_MAIN_SHEETS * 4 + 2


BASE_ARCHIVE_FINGERPRINTS = micro.archive_fingerprints
BASE_FROZEN_TREES = micro.frozen_trees
BASE_VALIDATE_SOURCE = micro.validate_source


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()


def sha256(path: Path, *, lf: bool = False) -> str:
    data = path.read_bytes()
    if lf:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def configure_micro() -> None:
    micro.TASK_ID = TASK_ID
    micro.SOURCE_HEAD = SOURCE_HEAD
    micro.RUN_BRANCH = RUN_BRANCH
    micro.PROTOCOL_PATH = PROTOCOL_PATH
    micro.PROTOCOL_SHA256 = PROTOCOL_SHA256
    micro.SELECTED_PAIRS = PAIRS
    micro.STABLE_PAIRS = {"O01_O02", "O01_O04", "O03_O04"}
    micro.EVALUATION_LABEL = LABEL
    micro.previous.TASK_ID = TASK_ID
    micro.previous.RUN_BRANCH = RUN_BRANCH
    micro.previous.SOURCE_HEAD = SOURCE_HEAD


configure_micro()


def protocol() -> dict[str, Any]:
    if sha256(PROTOCOL_PATH, lf=True) != PROTOCOL_SHA256:
        raise RuntimeError("ALL-PAIR-PROTOCOL-MISMATCH: fingerprint")
    value = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if tuple(value["selection"]["pairs"]) != PAIR_IDS:
        raise RuntimeError("ALL-PAIR-PROTOCOL-MISMATCH: pairs")
    if tuple(value["design"]["directions"]) != micro.DIRECTIONS:
        raise RuntimeError("ALL-PAIR-PROTOCOL-MISMATCH: directions")
    if tuple(float(item) for item in value["design"]["alpha_values"]) != micro.ALPHAS:
        raise RuntimeError("ALL-PAIR-PROTOCOL-MISMATCH: alpha")
    if tuple(value["design"]["target_conditions"]) != micro.CONDITIONS:
        raise RuntimeError("ALL-PAIR-PROTOCOL-MISMATCH: conditions")
    thresholds = value["success_thresholds"]
    frozen = {
        "endpoint_parity_pair_count": 10,
        "pair_count_required_for_core_grade_drop_at_alpha_0_5": 7,
        "minimum_core_grade_drop": 1,
        "pair_count_required_for_severe_count_drop": 7,
        "minimum_severe_count_reduction_fraction": 0.5,
        "maximum_grade_3_double_outline_or_ghosting_pairs": 1,
    }
    for key, expected in frozen.items():
        if float(thresholds[key]) != float(expected):
            raise RuntimeError(f"ALL-PAIR-PROTOCOL-MISMATCH: {key}")
    return value


def archive_fingerprints() -> dict[str, Any]:
    value = BASE_ARCHIVE_FINGERPRINTS()
    value.update({
        "micro_report_sha256": sha256(MICRO_REPORT),
        "micro_protocol_sha256": sha256(MICRO_PROTOCOL),
        "micro_results_sha256": sha256(MICRO_RESULTS),
        "micro_review_sha256": sha256(MICRO_REVIEW),
        "micro_summary_sha256": sha256(MICRO_SUMMARY),
        "micro_output_tree": provenance_tools.tree_manifest(MICRO_OUTPUT),
    })
    return value


def frozen_trees(asset_root: Path) -> dict[str, Any]:
    value = BASE_FROZEN_TREES(asset_root)
    value["micro_pilot"] = provenance_tools.tree_manifest(MICRO_OUTPUT)
    return value


def validate_source(*, require_clean: bool = True) -> dict[str, Any]:
    value = BASE_VALIDATE_SOURCE(require_clean=require_clean)
    frozen = protocol()
    summary = read_json(MICRO_SUMMARY)
    if summary["decision"]["classification"] != "DUAL_SUPPORT_MICRO_PILOT_PASS":
        raise RuntimeError("ALL-PAIR-GOVERNANCE-MISMATCH: micro-pilot classification")
    if summary["paper_final_count"] != 0 or not summary["frozen_unchanged"]:
        raise RuntimeError("ALL-PAIR-GOVERNANCE-MISMATCH: micro-pilot archive")
    value.update({"protocol": frozen, "micro_summary": summary})
    return value


micro.archive_fingerprints = archive_fingerprints
micro.frozen_trees = frozen_trees
micro.validate_source = validate_source


def run_preflight(output_root: Path, asset_root: Path) -> dict[str, Any]:
    attempt = output_root / "attempt_001"
    result_path = attempt / "audits/preflight.json"
    if result_path.is_file():
        return read_json(result_path)
    if output_root.exists():
        raise RuntimeError("append-only output root exists without a completed preflight")
    governance = validate_source()
    frozen_assets = verify_manifest(read_json(micro.FROZEN_MANIFEST), PROJECT_ROOT, asset_root, verify_external=True)
    if frozen_assets["status"] != "PASS":
        raise RuntimeError("ALL-PAIR-ASSET-MISMATCH: frozen assets")
    selected_full = {
        (pair_id, condition, alpha)
        for pair_id in PAIR_IDS for condition in micro.CONDITIONS for alpha in micro.ALPHAS
    }
    if len(selected_full) != 120 or any(key not in governance["full"] for key in selected_full):
        raise RuntimeError("ALL-PAIR-ASSET-MISMATCH: sealed FULL selection")
    smoke = torch.tensor([17.0, 19.0], device="cuda").sum()
    torch.cuda.synchronize()
    if float(smoke) != 36.0:
        raise RuntimeError("ALL-PAIR-ASSET-MISMATCH: CUDA smoke")
    for name in ("renders", "metrics", "visuals", "audits", "aggregates"):
        (attempt / name).mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": "canondressgs.research.dual_support_all_pair_preflight.v1",
        "status": "PASS",
        "label": LABEL,
        "task_id": TASK_ID,
        "source_head": SOURCE_HEAD,
        "execution_head": git("rev-parse", "HEAD"),
        "run_branch": RUN_BRANCH,
        "protocol_sha256_lf": PROTOCOL_SHA256,
        "pairs": list(PAIR_IDS),
        "pair_count": 10,
        "direction_count": 20,
        "full_reuse_logical_entries": EXPECTED_LOGICAL,
        "full_reuse_unique_files": 120,
        "full_regenerated_files": 0,
        "expected_hard_renders": EXPECTED_LOGICAL,
        "expected_dual_renders": EXPECTED_LOGICAL,
        "archive_fingerprints_before": archive_fingerprints(),
        "frozen_assets_before": frozen_assets,
        "frozen_trees_before": frozen_trees(asset_root),
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "counts": {"training": 0, "backward": 0, "optimizer_step": 0, "checkpoint_write": 0},
        "paper_final": False,
    }
    micro.atomic_json(result_path, result)
    return result


def aggregate_metrics(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "garment_rgb_mae", "garment_lpips", "silhouette_iou", "boundary_fscore",
        "protected_lpips", "identity_metric", "target_closer_fraction",
        "outside_garment_opacity", "opacity_mass", "active_gaussian_count",
        "render_time_seconds", "peak_vram_bytes", "displacement_rms", "displacement_p95",
    )
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(row["pair_id"], row["variant"])].append(row)
    return [
        {
            "pair_id": pair_id,
            "variant": variant,
            "record_count": len(rows),
            "means": {field: float(statistics.fmean(float(row[field]) for row in rows)) for field in fields},
            "medians": {field: float(statistics.median(float(row[field]) for row in rows)) for field in fields},
            "maxima": {field: max(float(row[field]) for row in rows) for field in fields},
        }
        for (pair_id, variant), rows in sorted(grouped.items())
    ]


def macro_metrics(aggregates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, float]]] = defaultdict(list)
    for row in aggregates:
        grouped[row["variant"]].append(row["means"])
    return [
        {
            "variant": variant,
            "pair_count": len(rows),
            "means": {
                field: float(statistics.fmean(float(row[field]) for row in rows))
                for field in rows[0]
            },
        }
        for variant, rows in sorted(grouped.items())
    ]


def run_analyze(output_root: Path) -> dict[str, Any]:
    attempt = output_root / "attempt_001"
    result_path = attempt / "aggregates/dual_support_all_pair_analysis.json"
    if result_path.is_file():
        return read_json(result_path)
    validate_source()
    execution = read_json(attempt / "aggregates/geometry_dual_support_execution.json")
    records = execution["records"]
    index = {
        (row["pair_id"], row["direction"], float(row["alpha"]), row["condition"], row["variant"]): row
        for row in records
    }
    if len(index) != EXPECTED_LOGICAL * len(micro.VARIANTS):
        raise RuntimeError("all-pair analysis record index mismatch")
    main_sheets, opacity_sheets, silhouette_sheets = [], [], []
    main_index: dict[tuple[str, str], str] = {}
    for left, right in PAIRS:
        pair_id = f"{left}_{right}"
        for direction in micro.DIRECTIONS:
            main_rows, opacity_rows, silhouette_rows = [], [], []
            for condition in micro.CONDITIONS:
                first = index[(pair_id, direction, micro.ALPHAS[0], condition, "FULL_LINEAR_BASELINE")]
                main_panels: list[tuple[str, Path]] = [("Teacher source", Path(first["source_rgb_path"]))]
                opacity_panels: list[tuple[str, Path]] = [("Teacher source", Path(first["source_alpha_path"]))]
                silhouette_panels: list[tuple[str, Path]] = []
                mask_path = attempt / "metrics/frozen_garment_masks" / left / f"{condition}.png"
                for alpha in micro.ALPHAS:
                    for variant in micro.VARIANTS:
                        row = index[(pair_id, direction, alpha, condition, variant)]
                        label = f"{alpha:.2f} {variant.replace('_GEOMETRY_', '_G_').replace('_BASELINE', '')}"
                        main_panels.append((label, Path(row["rgb_path"])))
                        opacity_panels.append((label, Path(row["alpha_path"])))
                        component = (
                            attempt / "visuals/silhouette_components" / pair_id / direction / condition
                            / f"{micro.alpha_id(alpha)}_{variant}.png"
                        )
                        micro.silhouette_overlay(Path(row["rgb_path"]), Path(row["alpha_path"]), mask_path, component)
                        silhouette_panels.append((label, component))
                main_panels.append(("Teacher target", Path(first["target_rgb_path"])))
                opacity_panels.append(("Teacher target", Path(first["target_alpha_path"])))
                main_rows.append((condition, main_panels))
                opacity_rows.append((condition, opacity_panels))
                silhouette_rows.append((condition, silhouette_panels))
            main_path = attempt / "visuals/main" / pair_id / f"{direction}.png"
            opacity_path = attempt / "visuals/opacity" / pair_id / f"{direction}.png"
            silhouette_path = attempt / "visuals/silhouette" / pair_id / f"{direction}.png"
            micro.contact_sheet(main_path, main_rows)
            micro.contact_sheet(opacity_path, opacity_rows)
            micro.contact_sheet(silhouette_path, silhouette_rows)
            main_sheets.append(str(main_path))
            opacity_sheets.append(str(opacity_path))
            silhouette_sheets.append(str(silhouette_path))
            main_index[(pair_id, direction)] = str(main_path)
    support_sheets = execution["geometry_support_overlays"]
    aggregates = aggregate_metrics(records)
    macros = macro_metrics(aggregates)
    aggregate_index = {(row["pair_id"], row["variant"]): row for row in aggregates}
    comparisons = []
    for pair_id in PAIR_IDS:
        baseline = aggregate_index[(pair_id, "FULL_LINEAR_BASELINE")]["means"]
        for variant in micro.NEW_VARIANTS:
            candidate = aggregate_index[(pair_id, variant)]["means"]
            comparisons.append({
                "pair_id": pair_id,
                "variant": variant,
                "garment_lpips_delta": candidate["garment_lpips"] - baseline["garment_lpips"],
                "silhouette_iou_delta": candidate["silhouette_iou"] - baseline["silhouette_iou"],
                "boundary_fscore_delta": candidate["boundary_fscore"] - baseline["boundary_fscore"],
                "protected_lpips_delta": candidate["protected_lpips"] - baseline["protected_lpips"],
                "baseline": baseline,
                "candidate": candidate,
            })
    dual_comparisons = [row for row in comparisons if row["variant"] == "DUAL_SUPPORT_GEOMETRY_BLEND"]
    best = min(dual_comparisons, key=lambda row: (row["garment_lpips_delta"], row["pair_id"]))
    worst = max(dual_comparisons, key=lambda row: (row["garment_lpips_delta"], row["pair_id"]))
    ranked_sheets = []
    for rank, row in (("best", best), ("worst", worst)):
        ranked_path = attempt / "visuals/ranked" / f"{rank}_pair_{row['pair_id']}.png"
        micro.contact_sheet(
            ranked_path,
            [(f"{rank}: {row['pair_id']} lpips_delta={row['garment_lpips_delta']:.6f}", [
                ("A_TO_B", Path(main_index[(row["pair_id"], "A_TO_B")])),
                ("B_TO_A", Path(main_index[(row["pair_id"], "B_TO_A")])),
            ])],
        )
        ranked_sheets.append(str(ranked_path))
    collapse_records = [
        {
            "pair_id": row["pair_id"], "direction": row["direction"], "alpha": row["alpha"],
            "condition": row["condition"],
            "source_endpoint_rgb_max_abs": row["source_endpoint_rgb_max_abs"],
            "target_endpoint_rgb_max_abs": row["target_endpoint_rgb_max_abs"],
        }
        for row in records
        if row["variant"] == "DUAL_SUPPORT_GEOMETRY_BLEND"
        and min(row["source_endpoint_rgb_max_abs"], row["target_endpoint_rgb_max_abs"]) <= 1.0e-6
    ]
    manifest = {
        "schema_version": "canondressgs.research.dual_support_all_pair_visual_manifest.v1",
        "status": "READY_FOR_MANUAL_REVIEW",
        "main_sheets": main_sheets,
        "opacity_diagnostic_sheets": opacity_sheets,
        "silhouette_overlay_sheets": silhouette_sheets,
        "geometry_support_overlay_sheets": support_sheets,
        "ranked_display_sheets": ranked_sheets,
        "best_pair": best["pair_id"],
        "worst_pair": worst["pair_id"],
        "ranking_metric": "DUAL minus FULL per-pair mean garment LPIPS",
        "expected_actual_open_count": EXPECTED_VISUALS,
        "paper_final": False,
    }
    visual_paths = main_sheets + opacity_sheets + silhouette_sheets + support_sheets + ranked_sheets
    if len(visual_paths) != EXPECTED_VISUALS or len(set(visual_paths)) != EXPECTED_VISUALS:
        raise RuntimeError("all-pair visual manifest accounting mismatch")
    if any(not Path(path).is_file() for path in visual_paths):
        raise RuntimeError("all-pair visual path missing")
    micro.atomic_json(attempt / "audits/visual_manifest.json", manifest)
    result = {
        "schema_version": "canondressgs.research.dual_support_all_pair_analysis.v1",
        "status": "AUTOMATIC_ANALYSIS_COMPLETE_MANUAL_REVIEW_REQUIRED",
        "label": LABEL,
        "task_id": TASK_ID,
        "record_count": len(records),
        "aggregates": aggregates,
        "all_pair_macro": macros,
        "comparisons": comparisons,
        "endpoint_collapse_records": collapse_records,
        "visual_manifest": manifest,
        "paper_final": False,
    }
    micro.atomic_json(result_path, result)
    return result


def expected_visual_paths(manifest: Mapping[str, Any]) -> list[str]:
    return (
        manifest["main_sheets"] + manifest["opacity_diagnostic_sheets"]
        + manifest["silhouette_overlay_sheets"] + manifest["geometry_support_overlay_sheets"]
        + manifest["ranked_display_sheets"]
    )


def validate_manual_review(review: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    expected = expected_visual_paths(manifest)
    if review["status"] != "COMPLETE" or review["expected_count"] != EXPECTED_VISUALS:
        raise RuntimeError("manual review status mismatch")
    if review["actual_opened_count"] != EXPECTED_VISUALS or len(review["items"]) != EXPECTED_VISUALS:
        raise RuntimeError("manual review count mismatch")
    if [item["source_path"] for item in review["items"]] != expected:
        raise RuntimeError("manual review path order mismatch")
    if not all(item["actual_opened"] for item in review["items"]):
        raise RuntimeError("manual review contains unopened paths")
    main_items = [item for item in review["items"] if item["kind"] == "MAIN_SHEET"]
    if len(main_items) != EXPECTED_MAIN_SHEETS:
        raise RuntimeError("manual main-sheet count mismatch")
    expected_coordinates = {
        (variant, alpha, condition)
        for variant in micro.VARIANTS for alpha in micro.ALPHAS for condition in micro.CONDITIONS
    }
    for item in main_items:
        rows = item["grades_by_variant_alpha_view"]
        if len(rows) != 36:
            raise RuntimeError("manual grade matrix is incomplete")
        if {(row["variant"], float(row["alpha"]), row["condition"]) for row in rows} != expected_coordinates:
            raise RuntimeError("manual grade coordinates mismatch")
        for row in rows:
            if set(row["grades"]) != set(VISUAL_CATEGORIES):
                raise RuntimeError("manual grade categories mismatch")
            if any(not isinstance(value, int) or not 0 <= value <= 3 for value in row["grades"].values()):
                raise RuntimeError("manual grade outside frozen scale")


def visual_grade_index(review: Mapping[str, Any]) -> dict[tuple[str, str, str, float, str], dict[str, int]]:
    result = {}
    for item in review["items"]:
        if item["kind"] != "MAIN_SHEET":
            continue
        for row in item["grades_by_variant_alpha_view"]:
            key = (item["pair_id"], item["direction"], row["variant"], float(row["alpha"]), row["condition"])
            result[key] = {name: int(value) for name, value in row["grades"].items()}
    if len(result) != EXPECTED_MAIN_SHEETS * 36:
        raise RuntimeError("visual grade index mismatch")
    return result


def visual_aggregate(review: Mapping[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for item in review["items"]:
        if item["kind"] != "MAIN_SHEET":
            continue
        for row in item["grades_by_variant_alpha_view"]:
            for category, grade in row["grades"].items():
                grouped[(row["variant"], category)].append(int(grade))
    return [
        {
            "variant": variant, "category": category, "observation_count": len(values),
            "mean_grade": float(statistics.fmean(values)), "maximum_grade": max(values),
            "severe_count": sum(value == 3 for value in values),
        }
        for (variant, category), values in sorted(grouped.items())
    ]


def classify(
    review: Mapping[str, Any], analysis: Mapping[str, Any], parity: Mapping[str, Any], frozen_unchanged: bool,
) -> dict[str, Any]:
    grades = visual_grade_index(review)
    pair_rows = []
    grade_pass_count = severe_pass_count = 0
    severe_ghosting_pairs = []
    for pair_id in PAIR_IDS:
        maximums, severe = {}, {}
        for variant in ("FULL_LINEAR_BASELINE", "DUAL_SUPPORT_GEOMETRY_BLEND"):
            maximums[variant] = max(
                grades[(pair_id, direction, variant, 0.5, condition)][category]
                for direction in micro.DIRECTIONS for condition in micro.CONDITIONS for category in CORE_CATEGORIES
            )
            severe[variant] = sum(
                grades[(pair_id, direction, variant, alpha, condition)][category] == 3
                for direction in micro.DIRECTIONS for alpha in micro.ALPHAS
                for condition in micro.CONDITIONS for category in CORE_CATEGORIES
            )
        drop = maximums["FULL_LINEAR_BASELINE"] - maximums["DUAL_SUPPORT_GEOMETRY_BLEND"]
        reduction = (severe["FULL_LINEAR_BASELINE"] - severe["DUAL_SUPPORT_GEOMETRY_BLEND"]) / max(severe["FULL_LINEAR_BASELINE"], 1)
        grade_pass = drop >= 1
        severe_pass = reduction >= 0.5
        grade_pass_count += int(grade_pass)
        severe_pass_count += int(severe_pass)
        pair_ghosting_max = max(
            grades[(pair_id, direction, "DUAL_SUPPORT_GEOMETRY_BLEND", alpha, condition)][category]
            for direction in micro.DIRECTIONS for alpha in micro.ALPHAS for condition in micro.CONDITIONS
            for category in ("double_outline", "ghosting")
        )
        if pair_ghosting_max == 3:
            severe_ghosting_pairs.append(pair_id)
        pair_rows.append({
            "pair_id": pair_id,
            "full_alpha_0_5_max_core_grade": maximums["FULL_LINEAR_BASELINE"],
            "dual_alpha_0_5_max_core_grade": maximums["DUAL_SUPPORT_GEOMETRY_BLEND"],
            "core_grade_drop": drop,
            "core_grade_drop_pass": grade_pass,
            "full_severe_core_count": severe["FULL_LINEAR_BASELINE"],
            "dual_severe_core_count": severe["DUAL_SUPPORT_GEOMETRY_BLEND"],
            "severe_reduction_fraction": reduction,
            "severe_reduction_pass": severe_pass,
            "dual_double_outline_or_ghosting_max_grade": pair_ghosting_max,
        })
    identity_max = max(
        grades[(pair_id, direction, "DUAL_SUPPORT_GEOMETRY_BLEND", alpha, condition)]["identity_contamination"]
        for pair_id in PAIR_IDS for direction in micro.DIRECTIONS for alpha in micro.ALPHAS for condition in micro.CONDITIONS
    )
    macro = {row["variant"]: row["means"] for row in analysis["all_pair_macro"]}
    full_macro, dual_macro = macro["FULL_LINEAR_BASELINE"], macro["DUAL_SUPPORT_GEOMETRY_BLEND"]
    parity_pairs = {row["pair_id"] for row in parity["records"] if row["pass"]}
    gates = {
        "endpoint_parity_10_of_10": parity["status"] == "PASS" and len(parity_pairs) == 10,
        "alpha_0_5_core_grade_reduction_7_of_10": grade_pass_count >= 7,
        "severe_count_reduction_7_of_10": severe_pass_count >= 7,
        "all_pair_macro_garment_lpips_improves": dual_macro["garment_lpips"] < full_macro["garment_lpips"],
        "all_pair_macro_silhouette_iou_non_decrease": dual_macro["silhouette_iou"] >= full_macro["silhouette_iou"],
        "identity_contamination_zero": identity_max == 0,
        "grade_3_double_outline_or_ghosting_pairs_at_most_one": len(severe_ghosting_pairs) <= 1,
        "no_new_endpoint_collapse": not analysis["endpoint_collapse_records"],
        "no_direction_or_view_selection": True,
        "frozen_unchanged": frozen_unchanged,
    }
    safety_keys = (
        "endpoint_parity_10_of_10", "identity_contamination_zero",
        "grade_3_double_outline_or_ghosting_pairs_at_most_one", "no_new_endpoint_collapse",
        "no_direction_or_view_selection", "frozen_unchanged",
    )
    safety = all(gates[key] for key in safety_keys)
    efficacy = (
        gates["alpha_0_5_core_grade_reduction_7_of_10"],
        gates["severe_count_reduction_7_of_10"],
    )
    if all(gates.values()):
        classification = "DUAL_SUPPORT_ALL_PAIR_PASS"
    elif safety and any(efficacy):
        classification = "DUAL_SUPPORT_ALL_PAIR_PARTIAL"
    else:
        classification = "DUAL_SUPPORT_ALL_PAIR_FAIL"
    next_task = {
        "DUAL_SUPPORT_ALL_PAIR_PASS": "DESIGN_REFERENCE_CONDITIONED_DUAL_SUPPORT_CONTROLLER",
        "DUAL_SUPPORT_ALL_PAIR_PARTIAL": "DIAGNOSE_DUAL_SUPPORT_GHOSTING_AND_OVERLAP",
        "DUAL_SUPPORT_ALL_PAIR_FAIL": "DESIGN_GEOMETRY_CORRESPONDENCE_WARP_DIAGNOSTIC",
    }[classification]
    return {
        "classification": classification,
        "gates": gates,
        "pair_artifact_comparisons": pair_rows,
        "grade_reduction_pair_count": grade_pass_count,
        "severe_reduction_pair_count": severe_pass_count,
        "identity_contamination_maximum_grade": identity_max,
        "grade_3_double_outline_or_ghosting_pairs": severe_ghosting_pairs,
        "grade_3_double_outline_or_ghosting_pair_count": len(severe_ghosting_pairs),
        "endpoint_collapse_count": len(analysis["endpoint_collapse_records"]),
        "all_pair_visual_aggregate": visual_aggregate(review),
        "next_task": next_task,
        "next_task_started": False,
    }


def tree_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def asset_storage(asset_root: Path) -> dict[str, Any]:
    manifest = read_json(micro.FROZEN_MANIFEST)
    selected = {"O01", "O02", "O03", "O04", "O08"}
    teacher_paths = []
    basis_paths = []
    base_paths = []
    for asset in manifest["assets"]:
        raw = asset.get("path") or asset.get("source_path")
        if not raw:
            continue
        path = Path(raw.replace("${CANONDRESSGS_OUTPUT_ROOT}", str(asset_root)).replace("${CANONDRESSGS_REPO_ROOT}", str(PROJECT_ROOT)))
        if asset["asset_id"].startswith("teacher_checkpoint_") and asset["asset_id"].rsplit("_", 1)[-1] in selected:
            teacher_paths.append(path)
        elif asset["asset_id"] == "rank4_explicit_basis":
            basis_paths.append(path)
        elif asset["asset_id"] == "mmlphuman_checkpoint":
            base_paths.append(path)
    return {
        "teacher_endpoint_checkpoint_count": len(teacher_paths),
        "teacher_endpoint_storage_bytes": sum(path.stat().st_size for path in teacher_paths),
        "basis_storage_bytes": sum(path.stat().st_size for path in basis_paths),
        "base_avatar_checkpoint_storage_bytes": sum(path.stat().st_size for path in base_paths),
        "checkpoint_copies_created": 0,
    }


def efficiency(analysis: Mapping[str, Any], output_root: Path, asset_root: Path) -> dict[str, Any]:
    macro = {row["variant"]: row["means"] for row in analysis["all_pair_macro"]}
    hard, dual = macro["HARD_GEOMETRY_SOFT_VA"], macro["DUAL_SUPPORT_GEOMETRY_BLEND"]
    result = {
        "hard_macro": hard,
        "dual_macro": dual,
        "active_gaussian_ratio_dual_over_hard": dual["active_gaussian_count"] / hard["active_gaussian_count"],
        "render_time_ratio_dual_over_hard": dual["render_time_seconds"] / hard["render_time_seconds"],
        "peak_vram_ratio_dual_over_hard": dual["peak_vram_bytes"] / hard["peak_vram_bytes"],
        "peak_vram_increase_bytes": dual["peak_vram_bytes"] - hard["peak_vram_bytes"],
        "evaluation_output_storage_bytes_before_finalize": tree_size(output_root),
        "runtime_duplicate_support": True,
        "checkpoint_copy_required": False,
        "onboarding_requirement": "controller must provide top-1/top-2 garment endpoints, normalized weights, confidence, and a fallback decision",
    }
    result.update(asset_storage(asset_root))
    return result


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def run_finalize(output_root: Path, asset_root: Path, review_path: Path) -> dict[str, Any]:
    attempt = output_root / "attempt_001"
    validate_source()
    preflight = read_json(attempt / "audits/preflight.json")
    execution = read_json(attempt / "aggregates/geometry_dual_support_execution.json")
    analysis = read_json(attempt / "aggregates/dual_support_all_pair_analysis.json")
    parity = read_json(attempt / "audits/endpoint_parity.json")
    manifest = read_json(attempt / "audits/visual_manifest.json")
    review = read_json(review_path)
    validate_manual_review(review, manifest)
    assets_after = verify_manifest(read_json(micro.FROZEN_MANIFEST), PROJECT_ROOT, asset_root, verify_external=True)
    trees_after = frozen_trees(asset_root)
    archives_after = archive_fingerprints()
    frozen_unchanged = (
        assets_after == preflight["frozen_assets_before"]
        and trees_after == preflight["frozen_trees_before"]
        and archives_after == preflight["archive_fingerprints_before"]
    )
    if not frozen_unchanged:
        raise RuntimeError("ALL-PAIR-FROZEN-MUTATION")
    provenance = provenance_tools.aggregate_optimizer_provenance(attempt)
    decision = classify(review, analysis, parity, frozen_unchanged)
    efficiency_result = efficiency(analysis, output_root, asset_root)
    counts = {
        "pair_count": 10,
        "pair_direction_count": 20,
        "full_reuse_logical_entries": execution["full_reuse_logical_entries"],
        "full_reuse_unique_files": execution["full_reuse_unique_files"],
        "full_regenerated_files": execution["full_regenerated_files"],
        "hard_renders": execution["hard_render_count"],
        "dual_renders": execution["dual_render_count"],
        "endpoint_parity_checks": parity["logical_check_count"],
        "visual_review": review["actual_opened_count"],
        "training_steps": 0,
        "backward_calls": provenance["backward_count"],
        "diagnostic_optimizer_created": int(provenance["diagnostic_optimizer"]["created"]),
        "diagnostic_optimizer_steps": provenance["diagnostic_optimizer"]["step_count"],
        "legacy_optimizer_creation_count": provenance["legacy_context_optimizer"]["creation_count"],
        "legacy_optimizer_zero_grad": provenance["legacy_context_optimizer"]["zero_grad_count"],
        "legacy_optimizer_steps": provenance["legacy_context_optimizer"]["step_count"],
        "scheduler_steps": provenance["legacy_context_optimizer"]["scheduler_step_count"],
        "checkpoint_writes": provenance["checkpoint_write_count"],
        "teacher_mutation": 0,
        "basis_mutation": 0,
        "formal_output_mutation": 0,
        "paper_final": 0,
    }
    results = {
        "schema_version": "canondressgs.research.dual_support_all_pair_results.v1",
        "status": "COMPLETE",
        "label": LABEL,
        "task_id": TASK_ID,
        "protocol_sha256_lf": PROTOCOL_SHA256,
        "execution": execution,
        "analysis": analysis,
        "endpoint_parity": parity,
        "decision": decision,
        "efficiency": efficiency_result,
        "optimizer_provenance": provenance,
        "paper_final": False,
    }
    summary = {
        "schema_version": "canondressgs.research.dual_support_all_pair_final_summary.v1",
        "status": "COMPLETE",
        "label": LABEL,
        "task_id": TASK_ID,
        "source_head": SOURCE_HEAD,
        "execution_head": git("rev-parse", "HEAD"),
        "run_branch": RUN_BRANCH,
        "protocol_sha256_lf": PROTOCOL_SHA256,
        "pairs": list(PAIR_IDS),
        "counts": counts,
        "endpoint_parity": {
            "status": parity["status"],
            "pair_count": 10,
            "maximum_render_max_abs": parity["maximum_render_max_abs"],
            "tolerance": parity["tolerance"],
        },
        "decision": decision,
        "efficiency": efficiency_result,
        "archive_fingerprints_before": preflight["archive_fingerprints_before"],
        "archive_fingerprints_after": archives_after,
        "frozen_assets_before": preflight["frozen_assets_before"],
        "frozen_assets_after": assets_after,
        "frozen_trees_before": preflight["frozen_trees_before"],
        "frozen_trees_after": trees_after,
        "frozen_unchanged": frozen_unchanged,
        "visual_review_path": "paper_protocol/reviewer_risk/dual_support_all_pair_visual_review.json",
        "controller_interface_report": "docs/PAPER/AAAI27_REFERENCE_CONDITIONED_DUAL_SUPPORT_INTERFACE_20260722.md",
        "next_task": decision["next_task"],
        "next_task_started": False,
        "paper_final": False,
        "paper_final_count": 0,
    }
    micro.atomic_json(attempt / "aggregates/dual_support_all_pair_results.json", results)
    micro.atomic_json(attempt / "aggregates/dual_support_all_pair_visual_review.json", review)
    micro.atomic_json(attempt / "aggregates/dual_support_all_pair_final_summary.json", summary)
    micro.atomic_json(PROJECT_ROOT / "paper_protocol/reviewer_risk/dual_support_all_pair_results.json", results)
    micro.atomic_json(PROJECT_ROOT / "paper_protocol/reviewer_risk/dual_support_all_pair_visual_review.json", review)
    micro.atomic_json(PROJECT_ROOT / "paper_protocol/reviewer_risk/dual_support_all_pair_final_summary.json", summary)
    pair_rows = decision["pair_artifact_comparisons"]
    macro = {row["variant"]: row["means"] for row in analysis["all_pair_macro"]}
    report = [
        "# AAAI27 Dual-Support All-Pair Evaluation",
        "", f"**{LABEL}**", "",
        f"- Source HEAD: `{SOURCE_HEAD}`.",
        f"- Protocol SHA-256: `{PROTOCOL_SHA256}`.",
        "- Scope: all 10 frozen seen-garment unordered pairs, both directions, three alpha values, and four frozen views.",
        f"- FULL reused: {counts['full_reuse_logical_entries']} logical / {counts['full_reuse_unique_files']} unique / 0 regenerated.",
        f"- HARD renders: {counts['hard_renders']}; DUAL renders: {counts['dual_renders']}.",
        f"- Endpoint parity: `{parity['status']}`; maximum render max_abs `{parity['maximum_render_max_abs']}`.",
        f"- Manual visual review: {review['actual_opened_count']}/{EXPECTED_VISUALS}.",
        f"- Classification: **{decision['classification']}**.",
        f"- NEXT_TASK: **{decision['next_task']}** (not started).", "",
        "## Per-pair artifact change", "",
        markdown_table(
            ["pair", "FULL max", "DUAL max", "drop", "FULL severe", "DUAL severe", "severe reduction", "ghost max"],
            [[row["pair_id"], row["full_alpha_0_5_max_core_grade"], row["dual_alpha_0_5_max_core_grade"], row["core_grade_drop"], row["full_severe_core_count"], row["dual_severe_core_count"], f"{row['severe_reduction_fraction']:.3f}", row["dual_double_outline_or_ghosting_max_grade"]] for row in pair_rows],
        ), "", "## All-pair macro metrics", "",
        markdown_table(
            ["variant", "LPIPS", "silhouette IoU", "boundary F-score", "protected LPIPS", "target-closer fraction"],
            [[variant, f"{values['garment_lpips']:.6f}", f"{values['silhouette_iou']:.6f}", f"{values['boundary_fscore']:.6f}", f"{values['protected_lpips']:.6f}", f"{values['target_closer_fraction']:.6f}"] for variant, values in sorted(macro.items())],
        ), "", "## Scientific boundary", "",
        "The result concerns only the closed five-garment seen wardrobe. It does not establish a trained reference-conditioned controller, arbitrary or unseen garments, novel poses/views, or cross-identity generalization.",
        "No pair, direction, or view was excluded or selected for aggregation. Quantitative best/worst sheets are display-only.",
        "No training, backward, optimizer step, scheduler step, checkpoint write, teacher mutation, basis mutation, historical-output mutation, or threshold adjustment occurred. PAPER_FINAL=0.",
    ]
    micro.write_new_text(PROJECT_ROOT / "docs/PAPER/AAAI27_DUAL_SUPPORT_ALL_PAIR_EVALUATION_20260722.md", "\n".join(report))
    eff = efficiency_result
    efficiency_report = [
        "# AAAI27 Dual-Support Efficiency Analysis", "", f"**{LABEL}**", "",
        "Dual support retains both immutable endpoint Gaussian branches at runtime; it does not copy or rewrite checkpoints.", "",
        markdown_table(
            ["measure", "value"],
            [
                ["HARD active Gaussians", f"{eff['hard_macro']['active_gaussian_count']:.3f}"],
                ["DUAL active Gaussians", f"{eff['dual_macro']['active_gaussian_count']:.3f}"],
                ["active ratio DUAL/HARD", f"{eff['active_gaussian_ratio_dual_over_hard']:.6f}"],
                ["render-time ratio DUAL/HARD", f"{eff['render_time_ratio_dual_over_hard']:.6f}"],
                ["peak-VRAM ratio DUAL/HARD", f"{eff['peak_vram_ratio_dual_over_hard']:.6f}"],
                ["peak-VRAM increase bytes", f"{eff['peak_vram_increase_bytes']:.0f}"],
                ["five endpoint checkpoints bytes", eff["teacher_endpoint_storage_bytes"]],
                ["basis bytes", eff["basis_storage_bytes"]],
                ["base-avatar checkpoint bytes", eff["base_avatar_checkpoint_storage_bytes"]],
                ["evaluation output bytes before finalize", eff["evaluation_output_storage_bytes_before_finalize"]],
                ["checkpoint copies created", 0],
            ],
        ), "",
        "Onboarding requires endpoint identities, normalized top-2 weights, confidence, compatibility/entropy handling, and a single-endpoint fallback. Runtime cost cannot be omitted from method selection.",
    ]
    micro.write_new_text(PROJECT_ROOT / "docs/PAPER/AAAI27_DUAL_SUPPORT_EFFICIENCY_ANALYSIS_20260722.md", "\n".join(efficiency_report))
    interface_report = [
        "# AAAI27 Reference-Conditioned Dual-Support Interface", "", f"**{LABEL}**", "",
        "This is an interface design only. No controller is trained or declared final.", "",
        "## Required controller outputs", "",
        "- A softmax garment-endpoint distribution over the frozen garment bank.",
        "- Top-1 and top-2 garment IDs with normalized mixture weights.",
        "- Confidence plus entropy and endpoint-compatibility diagnostics.",
        "- A confidence-based single-endpoint fallback that cannot silently delete a support after rendering.", "",
        "## Integration boundary", "",
        "The current Ours-v2 predictor emits four-dimensional basis coefficients; it does not directly select or weight dual-support endpoint branches. A future controller must map reference evidence to endpoint selection before the frozen opacity-gating renderer contract is applied.", "",
        "Candidate mechanisms are: softmax garment distribution, top-2 endpoint mixture, confidence-based single-endpoint fallback, and an entropy/compatibility gate. Their thresholds must be preregistered before any controller evaluation.", "",
        "This document does not claim arbitrary garments, unseen garments, novel poses/views, cross-identity generalization, or a final method.",
    ]
    micro.write_new_text(PROJECT_ROOT / "docs/PAPER/AAAI27_REFERENCE_CONDITIONED_DUAL_SUPPORT_INTERFACE_20260722.md", "\n".join(interface_report))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("preflight", "execute", "analyze", "finalize", "all"), default="all")
    parser.add_argument("--visual-review", type=Path)
    args = parser.parse_args()
    output_root, asset_root = args.output_root.resolve(), args.asset_root.resolve()
    if args.phase in {"preflight", "all"}:
        value = run_preflight(output_root, asset_root)
        print(json.dumps({"phase": "preflight", "status": value["status"]}, sort_keys=True))
        if args.phase == "preflight":
            return
    if args.phase in {"execute", "all"}:
        value = micro.run_execute(output_root, asset_root)
        print(json.dumps({
            "phase": "execute", "endpoint_parity": value["parity"]["status"],
            "hard_renders": value["execution"]["hard_render_count"],
            "dual_renders": value["execution"]["dual_render_count"],
        }, sort_keys=True))
        if args.phase == "execute":
            return
    if args.phase in {"analyze", "all"}:
        value = run_analyze(output_root)
        print(json.dumps({"phase": "analyze", "status": value["status"], "visual_count": EXPECTED_VISUALS}, sort_keys=True))
        if args.phase == "analyze":
            return
    if args.phase in {"finalize", "all"}:
        if args.visual_review is None:
            if args.phase == "all":
                print(json.dumps({"phase": "finalize", "status": "MANUAL_REVIEW_REQUIRED"}, sort_keys=True))
                return
            raise ValueError("--visual-review is required for finalize")
        value = run_finalize(output_root, asset_root, args.visual_review.resolve())
        print(json.dumps({
            "phase": "finalize", "status": value["status"],
            "classification": value["decision"]["classification"],
            "next_task": value["next_task"],
        }, sort_keys=True))


if __name__ == "__main__":
    main()
