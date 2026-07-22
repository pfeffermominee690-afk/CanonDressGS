"""No-training calibration, compatibility, and counterfactual diagnostic.

The formal controller archive is read-only.  This runner never constructs an
optimizer: the legacy helper that normally builds and immediately discards an
optimizer is bypassed, while every actual ``Optimizer.__init__`` call is a hard
error.  Existing formal Controller and Oracle renders are always reused.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene.reference_conditioned_dual_support_controller import OUTFIT_ORDER  # noqa: E402
from scene.p0_candidate_initialization_protocol import seed_all  # noqa: E402
from tools import diagnose_image_conditioned_overfit_failure as diagnosis  # noqa: E402
from tools.paper import run_geometry_dual_support_micro_pilot as geometry  # noqa: E402
from tools.paper import run_p0_color_spatial_soft_control_evaluations as sealed  # noqa: E402
from tools.paper import run_reference_conditioned_dual_support_controller_formal as formal  # noqa: E402
from tools.paper import run_reference_conditioned_dual_support_controller_formal_evaluation as evaluator  # noqa: E402


TASK_ID = "AAAI27-CONTROLLER-CALIBRATION-COMPATIBILITY-DIAGNOSTIC-001"
SOURCE_HEAD = "f45a518f330fb407942055756373e83f65717853"
RUN_BRANCH = "research/controller-calibration-compatibility-diagnostic-20260723"
SEEDS = (0, 1, 2)
TOP2_GRID = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95)
SECONDARY_GRID = (0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
CONTAMINATION = ("patch", "cloud", "mottle", "edge_scatter", "silhouette_discontinuity", "full_body_contamination")
METRIC_FIELDS = (
    "garment_rgb_mae", "garment_lpips", "silhouette_iou", "boundary_fscore",
    "protected_lpips", "identity_metric", "outside_garment_opacity",
)
VARIANTS = (
    "ACTUAL_CONTROLLER",
    "FORCE_DUAL_PRED_PAIR_PRED_WEIGHT",
    "ORACLE_PAIR_PREDICTED_PROBABILITY_WEIGHT",
    "CORRECT_PAIR_SUBSET_PRED_PAIR_ORACLE_WEIGHT",
    "ORACLE_PAIR_ORACLE_WEIGHT",
    "TOP1_SINGLE",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


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
    return statistics.fmean(rows) if rows else float("nan")


def composition(row: Mapping[str, Any]) -> str:
    value = str(row["assignment_type"])
    return "AAB" if value.startswith("AAB") else "ABB" if value.startswith("ABB") else value


def scalar_values(row: Mapping[str, Any]) -> dict[str, float]:
    ranked = sorted((float(value) for value in row["probabilities"]), reverse=True)
    return {
        "top1_probability": ranked[0],
        "secondary_normalized_weight": ranked[1] / max(ranked[0] + ranked[1], 1e-12),
        "top2_mass": ranked[0] + ranked[1],
        "probability_entropy": -sum(value * math.log(max(value, 1e-12)) for value in ranked),
        "top1_top2_margin": ranked[0] - ranked[1],
        "top2_top3_margin": ranked[1] - ranked[2],
    }


def quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)), "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)), "median": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)), "q95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)), "mean": float(np.mean(array)), "std": float(np.std(array)),
    }


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and scores[order[end]] == scores[order[cursor]]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = average_rank
        cursor = end
    positives = sum(labels)
    negatives = len(labels) - positives
    return (sum(rank for rank, label in zip(ranks, labels) if label) - positives * (positives + 1) / 2) / (positives * negatives)


def average_precision(scores: Sequence[float], labels: Sequence[int]) -> float:
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    positives = sum(labels)
    hit = 0
    total = 0.0
    for rank, index in enumerate(order, 1):
        if labels[index]:
            hit += 1
            total += hit / rank
    return total / positives


def overlap_coefficient(first: Sequence[float], second: Sequence[float]) -> float:
    low = min(min(first), min(second))
    high = max(max(first), max(second))
    if high <= low:
        return 1.0
    first_hist, edges = np.histogram(first, bins=64, range=(low, high), density=True)
    second_hist, _ = np.histogram(second, bins=64, range=(low, high), density=True)
    return float(np.minimum(first_hist, second_hist).sum() * (edges[1] - edges[0]))


def grouped_summary(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in fields:
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row[field])].append(row)
        result[field] = {
            key: {
                "count": len(group),
                "dual_support_count": sum(item["mode"] == "DUAL_SUPPORT" for item in group),
                "dual_support_rate": mean(float(item["mode"] == "DUAL_SUPPORT") for item in group),
                "pair_accuracy": mean(float(item["pair_correct"]) for item in group),
                "ordering_accuracy": mean(float(item["ordering_correct"]) for item in group),
                "weight_mae": mean(float(item["weight_absolute_error"]) for item in group),
            }
            for key, group in sorted(groups.items())
        }
    return result


def formal_paths(formal_root: Path, seed: int) -> dict[str, Path]:
    return {
        "mixed": formal_root / f"attempt_001/seed_{seed}/mixed/mixed_classification.json",
        "pure": formal_root / f"attempt_001/seed_{seed}/pure/pure_classification.json",
        "render": formal_root / f"attempt_004/seed_{seed}/metrics/render_evaluation.json",
        "perturb": formal_root / f"attempt_004/seed_{seed}/perturbations/perturbation_evaluation.json",
        "checkpoint": formal_root / f"attempt_001/seed_{seed}/checkpoints/final_step_300.pt",
    }


def load_formal(formal_root: Path) -> dict[str, Any]:
    data: dict[str, Any] = {"mixed": {}, "pure": {}, "render": {}, "perturb": {}, "paths": {}}
    for seed in SEEDS:
        paths = formal_paths(formal_root, seed)
        if not all(path.is_file() for path in paths.values()):
            raise FileNotFoundError(paths)
        data["paths"][seed] = paths
        for key in ("mixed", "pure", "render", "perturb"):
            data[key][seed] = read_json(paths[key])
    return data


def tree_content_fingerprint(root: Path) -> dict[str, Any]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        relative = str(path.relative_to(root)).replace("\\", "/")
        size = path.stat().st_size
        total_bytes += size
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(str(size).encode("ascii") + b"\0")
        digest.update(sha256(path).encode("ascii") + b"\n")
    return {"file_count": len(files), "total_bytes": total_bytes, "sha256": digest.hexdigest()}


def run_preflight(formal_root: Path, diagnostic_root: Path, asset_root: Path, repository: Path) -> dict[str, Any]:
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repository, text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=repository, text=True).strip()
    ancestor = subprocess.call(["git", "merge-base", "--is-ancestor", SOURCE_HEAD, "HEAD"], cwd=repository) == 0
    if branch != RUN_BRANCH or dirty or not ancestor:
        raise RuntimeError(f"diagnostic governance mismatch branch={branch} head={head} dirty={bool(dirty)} ancestor={ancestor}")
    if diagnostic_root.exists():
        raise FileExistsError(f"append-only diagnostic output root exists before preflight: {diagnostic_root}")
    data = load_formal(formal_root)
    protocol = repository / "paper_protocol/reviewer_risk/controller_calibration_diagnostic_protocol.yaml"
    result = {
        "schema_version": "canondressgs.research.controller_calibration_diagnostic_preflight.v1",
        "status": "PASS", "task_id": TASK_ID, "source_head": SOURCE_HEAD,
        "execution_head": head, "run_branch": branch,
        "formal_output_before": tree_content_fingerprint(formal_root),
        "frozen_asset_snapshots_before": formal.tree_snapshots(asset_root),
        "checkpoint_sha256": {str(seed): sha256(data["paths"][seed]["checkpoint"]) for seed in SEEDS},
        "protocol_sha256_lf": hashlib.sha256(protocol.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")).hexdigest(),
        "counts": {
            "training_steps": 0, "training_forward_batches": 0, "backward_calls": 0,
            "optimizer_created": 0, "optimizer_steps": 0, "scheduler_steps": 0, "checkpoint_writes": 0,
        },
        "formal_output_write_count": 0, "paper_final": False, "paper_final_count": 0,
    }
    write_once(diagnostic_root / "attempt_001/audits/preflight.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "frozen_asset_snapshots_before"}, indent=2, sort_keys=True))
    return result


def reaggregate(data: Mapping[str, Any], repository: Path) -> dict[str, Any]:
    summary = read_json(repository / "paper_protocol/reviewer_risk/dual_support_controller_final_summary.json")
    visual = read_json(repository / "paper_protocol/reviewer_risk/dual_support_controller_visual_review.json")
    by_seed = []
    all_mixed: list[dict[str, Any]] = []
    all_pure: list[dict[str, Any]] = []
    activation_counts = []
    for seed in SEEDS:
        mixed_payload = data["mixed"][seed]
        pure_payload = data["pure"][seed]
        protocol_rows = mixed_payload["records"]
        mixed_rows = [dict(row, seed=seed, composition=composition(row)) for row in protocol_rows if row["pair_correct"] is not None]
        pure_rows = [dict(row, seed=seed) for row in pure_payload["records"]]
        if len(protocol_rows) != 320 or len(mixed_rows) != 240 or len(pure_rows) != 20:
            raise RuntimeError("FORMAL_CONTROLLER_SUMMARY_REAGGREGATION_MISMATCH: record counts")
        if {row["assignment_position"] for row in mixed_rows} != {0, 1, 2}:
            raise RuntimeError("FORMAL_CONTROLLER_SUMMARY_REAGGREGATION_MISMATCH: assignments")
        if len({row["target_view_fold"] for row in mixed_rows}) != 4 or len({row["pair_id"] for row in mixed_rows}) != 10:
            raise RuntimeError("FORMAL_CONTROLLER_SUMMARY_REAGGREGATION_MISMATCH: folds/pairs")
        current = evaluator.stats(protocol_rows)
        expected = mixed_payload["protocol_weighted"]
        for key, value in expected.items():
            if isinstance(value, (int, float)) and not math.isclose(float(current[key]), float(value), rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(f"FORMAL_CONTROLLER_SUMMARY_REAGGREGATION_MISMATCH: seed={seed} key={key}")
        active = sum(row["mode"] == "DUAL_SUPPORT" for row in mixed_rows)
        activation_counts.append(active)
        by_seed.append({"seed": seed, "protocol_weighted": current, "mixed_count": len(mixed_rows), "pure_count": len(pure_rows), "dual_support_count": active})
        all_mixed.extend(mixed_rows)
        all_pure.extend(pure_rows)
    macro = {key: mean(float(row["protocol_weighted"][key]) for row in by_seed) for key in by_seed[0]["protocol_weighted"]}
    for key, value in summary["mixed_protocol_weighted_macro"].items():
        if isinstance(value, (int, float)) and not math.isclose(float(macro[key]), float(value), rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"FORMAL_CONTROLLER_SUMMARY_REAGGREGATION_MISMATCH: macro {key}")
    primary = [row for row in visual["records"] if row.get("included_in_primary_aggregation")]
    categories = sorted(primary[0]["grades"])
    visual_recomputed = {}
    for category in categories:
        selected = [int(row["grades"][category]) for row in primary]
        visual_recomputed[category] = {
            "maximum_grade": max(selected), "mean_grade": mean(selected), "severe_sheet_count": sum(value == 3 for value in selected),
            "affected_pair_count": len({row.get("pair_id") for row in primary if row.get("pair_id") and int(row["grades"][category]) > 0}),
        }
        expected = visual["category_summary"][category]
        for key in ("maximum_grade", "mean_grade", "severe_sheet_count", "affected_pair_count"):
            if not math.isclose(float(visual_recomputed[category][key]), float(expected[key]), rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(f"FORMAL_CONTROLLER_SUMMARY_REAGGREGATION_MISMATCH: visual {category}/{key}")
    return {
        "schema_version": "canondressgs.research.formal_controller_reaggregation.v1",
        "status": "PASS", "task_id": TASK_ID, "source_head": SOURCE_HEAD,
        "by_seed": by_seed, "macro": macro, "activation_counts": activation_counts,
        "mixed_seed_record_query_count": len(all_mixed), "pure_formal_seed_record_count": len(all_pure),
        "visual_primary_record_count": len(primary), "visual_category_summary": visual_recomputed,
        "checkpoint_sha256": {str(seed): sha256(data["paths"][seed]["checkpoint"]) for seed in SEEDS},
        "paper_final": False,
    }


def fallback_analysis(data: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for raw in data["mixed"][seed]["records"]:
            if raw["pair_correct"] is None:
                continue
            row = dict(raw)
            ranked = sorted((float(value) for value in row["probabilities"]), reverse=True)
            row.update({
                "seed": seed, "composition": composition(row), "top1_probability": ranked[0],
                "top2_probability": ranked[1], "top1_top2_margin": ranked[0] - ranked[1],
                "top2_top3_margin": ranked[1] - ranked[2],
                "both_conditions": float(row["top2_mass"]) < 0.90 and float(row["normalized_secondary_weight"]) < 0.10,
            })
            rows.append(row)
    reasons = Counter(str(row["fallback_reason"] or "DUAL_SUPPORT") for row in rows)
    special = {
        "correct_pair_fallback": sum(row["pair_correct"] and row["mode"] == "SINGLE_ENDPOINT" for row in rows),
        "correct_pair_correct_ordering_fallback": sum(row["pair_correct"] and row["ordering_correct"] and row["mode"] == "SINGLE_ENDPOINT" for row in rows),
        "wrong_pair_dual_active": sum(not row["pair_correct"] and row["mode"] == "DUAL_SUPPORT" for row in rows),
        "wrong_ordering_dual_active": sum(not row["ordering_correct"] and row["mode"] == "DUAL_SUPPORT" for row in rows),
    }
    return {
        "schema_version": "canondressgs.research.controller_fallback_reason_decomposition.v1",
        "status": "COMPLETE", "task_id": TASK_ID, "record_count": len(rows),
        "counts": {
            "LOW_TOP2_MASS": reasons["LOW_TOP2_MASS"],
            "LOW_SECONDARY_WEIGHT": reasons["LOW_SECONDARY_WEIGHT"],
            "DUAL_SUPPORT": reasons["DUAL_SUPPORT"],
            "both_conditions_logical_hit": sum(row["both_conditions"] for row in rows),
        },
        "rates": {key: value / len(rows) for key, value in reasons.items()},
        "special_counts": special,
        "by_dimension": grouped_summary(rows, ("seed", "pair_id", "composition", "assignment_position", "target_view_fold", "pair_correct", "ordering_correct")),
        "records": rows, "threshold_change": 0, "training": 0, "paper_final": False,
    }


def separability_analysis(data: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for seed in SEEDS:
        for source, label in ((data["pure"][seed]["records"], 0), (data["mixed"][seed]["records"], 1)):
            for row in source:
                if label and row["pair_correct"] is None:
                    continue
                values = scalar_values(row)
                rows.append({"seed": seed, "label": "MIXED" if label else "PURE", "binary_label": label,
                             "pair_id": row.get("pair_id") or row["top1_outfit"], "assignment_position": row["assignment_position"],
                             "target_view_fold": row["target_view_fold"], **values})
    orientations = {
        "top1_probability": -1.0, "secondary_normalized_weight": 1.0, "top2_mass": -1.0,
        "probability_entropy": 1.0, "top1_top2_margin": -1.0, "top2_top3_margin": 1.0,
    }
    metrics = {}
    labels = [int(row["binary_label"]) for row in rows]
    for name, orientation in orientations.items():
        raw = [float(row[name]) for row in rows]
        scores = [orientation * value for value in raw]
        pure = [float(row[name]) for row in rows if row["label"] == "PURE"]
        mixed = [float(row[name]) for row in rows if row["label"] == "MIXED"]
        metrics[name] = {
            "mixed_score_orientation": "HIGH" if orientation > 0 else "LOW",
            "auroc": auroc(scores, labels), "auprc": average_precision(scores, labels),
            "overlap_coefficient_histogram64": overlap_coefficient(pure, mixed),
            "pure_distribution": quantiles(pure), "mixed_distribution": quantiles(mixed),
        }
        for group_name in ("seed", "pair_id", "assignment_position", "target_view_fold"):
            groups: dict[str, list[float]] = defaultdict(list)
            for row in rows:
                if row["label"] == "MIXED":
                    groups[str(row[group_name])].append(float(row[name]))
            group_means = [mean(group) for group in groups.values()]
            metrics[name][f"{group_name}_mean_variance"] = float(np.var(group_means))
    return {
        "schema_version": "canondressgs.research.controller_pure_mixed_separability.v1",
        "status": "COMPLETE", "task_id": TASK_ID, "pure_count": sum(not row["binary_label"] for row in rows),
        "mixed_count": sum(row["binary_label"] for row in rows), "metrics": metrics,
        "diagnostic_labels_entered_controller": False, "model_fit": False, "paper_final": False,
    }


def threshold_analysis(data: Mapping[str, Any], repository: Path) -> dict[str, Any]:
    visual = read_json(repository / "paper_protocol/reviewer_risk/dual_support_controller_visual_review.json")
    grade_index = {
        (int(row["seed"]), str(row["pair_id"]), str(row["composition"])): max(int(row["grades"][key]) for key in CONTAMINATION)
        for row in visual["records"] if row.get("kind") == "MIXED_MAIN"
    }
    pure = [(seed, row) for seed in SEEDS for row in data["pure"][seed]["records"]]
    mixed = [(seed, row) for seed in SEEDS for row in data["mixed"][seed]["records"] if row["pair_correct"] is not None]
    rows = []
    for mass_threshold in TOP2_GRID:
        for secondary_threshold in SECONDARY_GRID:
            pure_dual = [(seed, row) for seed, row in pure if float(row["top2_mass"]) >= mass_threshold and float(row["normalized_secondary_weight"]) >= secondary_threshold]
            mixed_dual = [(seed, row) for seed, row in mixed if float(row["top2_mass"]) >= mass_threshold and float(row["normalized_secondary_weight"]) >= secondary_threshold]
            wrong = [(seed, row) for seed, row in mixed if not row["pair_correct"]]
            wrong_dual = [(seed, row) for seed, row in mixed_dual if not row["pair_correct"]]
            correct = [(seed, row) for seed, row in mixed if row["pair_correct"]]
            correct_dual = [(seed, row) for seed, row in mixed_dual if row["pair_correct"]]
            severe = sum(grade_index[(seed, str(row["pair_id"]), composition(row))] == 3 for seed, row in mixed_dual)
            combined_dual_rate = (len(pure_dual) + len(mixed_dual)) / (len(pure) + len(mixed))
            pure_seed = {str(seed): mean(not (float(row["top2_mass"]) >= mass_threshold and float(row["normalized_secondary_weight"]) >= secondary_threshold) for current, row in pure if current == seed) for seed in SEEDS}
            mixed_seed = {str(seed): mean(float(row["top2_mass"]) >= mass_threshold and float(row["normalized_secondary_weight"]) >= secondary_threshold for current, row in mixed if current == seed) for seed in SEEDS}
            row = {
                "top2_mass_threshold": mass_threshold, "secondary_weight_threshold": secondary_threshold,
                "pure_single_rate": 1.0 - len(pure_dual) / len(pure), "mixed_dual_rate": len(mixed_dual) / len(mixed),
                "pure_false_dual_rate": len(pure_dual) / len(pure), "mixed_false_single_rate": 1.0 - len(mixed_dual) / len(mixed),
                "correct_pair_dual_rate": len(correct_dual) / len(correct),
                "wrong_pair_dual_rate": len(wrong_dual) / len(wrong), "wrong_pair_dual_count": len(wrong_dual), "wrong_pair_count": len(wrong),
                "severe_artifact_exposure_estimate_count": severe,
                "severe_artifact_exposure_estimate_rate_among_mixed": severe / len(mixed),
                "active_gaussian_expectation": 199999.26 + combined_dual_rate * (399949.56 - 199999.26),
                "render_time_expectation_seconds": 0.00860150 + combined_dual_rate * (0.01354002 - 0.00860150),
                "pure_single_rate_by_seed": pure_seed, "mixed_dual_rate_by_seed": mixed_seed,
            }
            row["simple_gate_criteria_met"] = (
                row["pure_single_rate"] >= 0.95 and row["mixed_dual_rate"] >= 0.80 and row["wrong_pair_dual_rate"] <= 0.10
                and min(pure_seed.values()) >= 0.90 and min(mixed_seed.values()) >= 0.70
            )
            rows.append(row)
    frontier = []
    for candidate in rows:
        dominated = any(
            other is not candidate
            and other["pure_single_rate"] >= candidate["pure_single_rate"]
            and other["mixed_dual_rate"] >= candidate["mixed_dual_rate"]
            and other["wrong_pair_dual_rate"] <= candidate["wrong_pair_dual_rate"]
            and other["severe_artifact_exposure_estimate_rate_among_mixed"] <= candidate["severe_artifact_exposure_estimate_rate_among_mixed"]
            and (other["pure_single_rate"], other["mixed_dual_rate"], -other["wrong_pair_dual_rate"], -other["severe_artifact_exposure_estimate_rate_among_mixed"])
            != (candidate["pure_single_rate"], candidate["mixed_dual_rate"], -candidate["wrong_pair_dual_rate"], -candidate["severe_artifact_exposure_estimate_rate_among_mixed"])
            for other in rows
        )
        if not dominated:
            frontier.append(candidate)
    candidates = [row for row in rows if row["simple_gate_criteria_met"]]
    return {
        "schema_version": "canondressgs.research.controller_threshold_sweep_diagnostic.v1",
        "status": "COMPLETE", "task_id": TASK_ID, "grid_count": len(rows), "grid": rows,
        "pareto_frontier": frontier, "simple_gate_status": "SIMPLE_GATE_SEPARABLE" if candidates else "SIMPLE_GATE_NOT_SEPARABLE",
        "separable_combination_count": len(candidates), "separable_combinations": candidates,
        "formal_threshold_unchanged": {"top2_mass": 0.90, "secondary_weight": 0.10},
        "threshold_selected": False, "new_pass_conclusion": False, "paper_final": False,
    }


def weight_analysis(data: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    render_lpips = {
        (seed, row["record_id"]): float(row["metrics"]["garment_lpips"])
        for seed in SEEDS for row in data["render"][seed]["records"] if row["role"] == "mixed"
    }
    for seed in SEEDS:
        for row in data["mixed"][seed]["records"]:
            if row["pair_correct"] is None:
                continue
            target = [float(value) for value in row["target_distribution"]]
            probabilities = [float(value) for value in row["probabilities"]]
            pair_indices = [index for index, value in enumerate(target) if value > 0]
            dominant = max(pair_indices, key=lambda index: target[index])
            pair_mass = sum(probabilities[index] for index in pair_indices)
            predicted_dominant = probabilities[dominant] / max(pair_mass, 1e-12)
            signed = predicted_dominant - 2.0 / 3.0
            rows.append({
                "seed": seed, "record_id": row["record_id"], "pair_id": row["pair_id"], "composition": composition(row),
                "assignment_position": row["assignment_position"], "target_view_fold": row["target_view_fold"],
                "pair_correct": bool(row["pair_correct"]), "ordering_correct": bool(row["ordering_correct"]),
                "mode": row["mode"], "predicted_dominant_weight": predicted_dominant,
                "predicted_secondary_weight": 1.0 - predicted_dominant, "signed_error": signed,
                "absolute_error": abs(signed), "under_mixing": signed < 0, "over_mixing": signed > 0,
                "controller_vs_oracle_lpips": render_lpips[(seed, row["record_id"])],
            })
    def summarize(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not selected:
            return {"count": 0}
        return {
            "count": len(selected), "dominant_weight_mean": mean(float(row["predicted_dominant_weight"]) for row in selected),
            "secondary_weight_mean": mean(float(row["predicted_secondary_weight"]) for row in selected),
            "signed_error_mean": mean(float(row["signed_error"]) for row in selected),
            "mae": mean(float(row["absolute_error"]) for row in selected),
            "under_mixing_rate": mean(float(row["under_mixing"]) for row in selected),
            "over_mixing_rate": mean(float(row["over_mixing"]) for row in selected),
            "lpips_mean": mean(float(row["controller_vs_oracle_lpips"]) for row in selected),
        }
    correct = [row for row in rows if row["pair_correct"]]
    correlation = float(np.corrcoef([row["absolute_error"] for row in correct], [row["controller_vs_oracle_lpips"] for row in correct])[0, 1])
    breakdown = {
        "pair_correct_order_correct": summarize([row for row in rows if row["pair_correct"] and row["ordering_correct"]]),
        "pair_correct_order_wrong": summarize([row for row in rows if row["pair_correct"] and not row["ordering_correct"]]),
        "pair_wrong": summarize([row for row in rows if not row["pair_correct"]]),
        "dual_active": summarize([row for row in rows if row["mode"] == "DUAL_SUPPORT"]),
        "single_fallback": summarize([row for row in rows if row["mode"] == "SINGLE_ENDPOINT"]),
    }
    group_breakdown = {}
    for field in ("seed", "pair_id", "assignment_position", "target_view_fold"):
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in correct:
            groups[str(row[field])].append(row)
        group_breakdown[field] = {key: summarize(value) for key, value in sorted(groups.items())}
    return {
        "schema_version": "canondressgs.research.controller_weight_calibration_analysis.v1",
        "status": "COMPLETE", "task_id": TASK_ID, "record_count": len(rows), "correct_pair_count": len(correct),
        "target_dominant_weight": 2.0 / 3.0, "target_secondary_weight": 1.0 / 3.0,
        "breakdown": breakdown, "correct_pair_weight_error_lpips_pearson": correlation,
        "by_dimension": group_breakdown, "records": rows,
        "temperature_fit": False, "calibration_model_fit": False, "posthoc_weight_replacement": False, "paper_final": False,
    }


def run_offline(formal_root: Path, diagnostic_root: Path, repository: Path) -> dict[str, Any]:
    data = load_formal(formal_root)
    attempt = diagnostic_root / "attempt_001"
    for name in ("reused", "counterfactuals", "metrics", "visuals", "perturbations", "audits", "aggregates"):
        (attempt / name).mkdir(parents=True, exist_ok=True)
    reaggregation = reaggregate(data, repository)
    fallback = fallback_analysis(data)
    separability = separability_analysis(data)
    threshold = threshold_analysis(data, repository)
    weights = weight_analysis(data)
    write_once(attempt / "audits/formal_summary_reaggregation.json", reaggregation)
    write_once(attempt / "aggregates/controller_fallback_reason_decomposition.json", fallback)
    write_once(attempt / "aggregates/controller_pure_mixed_separability.json", separability)
    write_once(attempt / "aggregates/controller_threshold_sweep_diagnostic.json", threshold)
    write_once(attempt / "aggregates/controller_weight_calibration_analysis.json", weights)
    result = {
        "status": "PASS", "formal_summary_reaggregation": "PASS",
        "mixed_count": fallback["record_count"], "fallback_counts": fallback["counts"],
        "simple_gate_status": threshold["simple_gate_status"],
        "checkpoint_sha256": reaggregation["checkpoint_sha256"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


class NoTrainingGate(contextlib.AbstractContextManager["NoTrainingGate"]):
    """Make optimizer construction, backward, scheduler steps, and torch saves impossible."""

    def __init__(self) -> None:
        self.builder_bypass_count = 0
        self.optimizer_init_count = 0
        self.backward_calls = 0
        self.scheduler_steps = 0
        self.checkpoint_writes = 0
        self.in_memory_torch_save_count = 0
        self._originals: dict[str, Any] = {}

    def __enter__(self) -> "NoTrainingGate":
        import train_dressable

        self._training = train_dressable
        self._originals = {
            "builder": train_dressable.build_image_conditioned_optimizer,
            "optimizer_init": torch.optim.Optimizer.__init__,
            "tensor_backward": torch.Tensor.backward,
            "autograd_backward": torch.autograd.backward,
            "scheduler_step": torch.optim.lr_scheduler.LRScheduler.step,
            "torch_save": torch.save,
        }
        gate = self

        def bypass_builder(model: torch.nn.Module, optimizer_config: Mapping[str, Any]) -> None:
            del model, optimizer_config
            gate.builder_bypass_count += 1
            return None

        def forbidden_optimizer_init(instance: Any, *args: Any, **kwargs: Any) -> None:
            del instance, args, kwargs
            gate.optimizer_init_count += 1
            raise RuntimeError("DIAGNOSTIC_NO_TRAINING_GATE: optimizer construction")

        def forbidden_backward(*args: Any, **kwargs: Any) -> None:
            del args, kwargs
            gate.backward_calls += 1
            raise RuntimeError("DIAGNOSTIC_NO_TRAINING_GATE: backward")

        def forbidden_scheduler(instance: Any, *args: Any, **kwargs: Any) -> None:
            del instance, args, kwargs
            gate.scheduler_steps += 1
            raise RuntimeError("DIAGNOSTIC_NO_TRAINING_GATE: scheduler step")

        def guarded_save(*args: Any, **kwargs: Any) -> None:
            destination = args[1] if len(args) > 1 else kwargs.get("f")
            if isinstance(destination, (str, bytes, os.PathLike, Path)):
                gate.checkpoint_writes += 1
                raise RuntimeError("DIAGNOSTIC_NO_TRAINING_GATE: checkpoint/path torch.save")
            gate.in_memory_torch_save_count += 1
            return gate._originals["torch_save"](*args, **kwargs)

        train_dressable.build_image_conditioned_optimizer = bypass_builder
        torch.optim.Optimizer.__init__ = forbidden_optimizer_init
        torch.Tensor.backward = forbidden_backward
        torch.autograd.backward = forbidden_backward
        torch.optim.lr_scheduler.LRScheduler.step = forbidden_scheduler
        torch.save = guarded_save
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        self._training.build_image_conditioned_optimizer = self._originals["builder"]
        torch.optim.Optimizer.__init__ = self._originals["optimizer_init"]
        torch.Tensor.backward = self._originals["tensor_backward"]
        torch.autograd.backward = self._originals["autograd_backward"]
        torch.optim.lr_scheduler.LRScheduler.step = self._originals["scheduler_step"]
        torch.save = self._originals["torch_save"]
        return False

    def result(self) -> dict[str, Any]:
        return {
            "training_steps": 0, "training_forward_batches": 0,
            "backward_calls": self.backward_calls, "optimizer_created": self.optimizer_init_count,
            "optimizer_steps": 0, "scheduler_steps": self.scheduler_steps,
            "checkpoint_writes": self.checkpoint_writes,
            "legacy_discarded_optimizer_builder_bypassed": self.builder_bypass_count,
            "in_memory_torch_save_count": self.in_memory_torch_save_count,
            "status": "PASS" if not any((self.backward_calls, self.optimizer_init_count, self.scheduler_steps, self.checkpoint_writes)) else "FAIL",
        }


def variant_branches(
    variant: str, record: Mapping[str, Any], prediction: Mapping[str, Any], endpoints: Mapping[str, Any]
) -> tuple[tuple[str, Any, float], ...] | None:
    probabilities = {outfit: float(value) for outfit, value in zip(OUTFIT_ORDER, prediction["probabilities"])}
    if variant == "FORCE_DUAL_PRED_PAIR_PRED_WEIGHT":
        first, second = str(prediction["top1_outfit"]), str(prediction["top2_outfit"])
        secondary = float(prediction["normalized_secondary_weight"])
        return ((first, endpoints[first], 1.0 - secondary), (second, endpoints[second], secondary))
    if variant == "ORACLE_PAIR_PREDICTED_PROBABILITY_WEIGHT":
        outfits = [OUTFIT_ORDER[index] for index, value in enumerate(record["target_distribution"]) if float(value) > 0]
        mass = sum(probabilities[outfit] for outfit in outfits)
        return tuple((outfit, endpoints[outfit], probabilities[outfit] / max(mass, 1e-12)) for outfit in outfits)
    if variant == "CORRECT_PAIR_SUBSET_PRED_PAIR_ORACLE_WEIGHT":
        if not prediction["pair_correct"]:
            return None
        target = {OUTFIT_ORDER[index]: float(value) for index, value in enumerate(record["target_distribution"]) if float(value) > 0}
        outfits = (str(prediction["top1_outfit"]), str(prediction["top2_outfit"]))
        return tuple((outfit, endpoints[outfit], target[outfit]) for outfit in outfits)
    if variant == "TOP1_SINGLE":
        outfit = str(prediction["top1_outfit"])
        return ((outfit, endpoints[outfit], 1.0),)
    raise KeyError(variant)


def load_rgb(path: Path, device: torch.device) -> torch.Tensor:
    return sealed.image_tensor(path, 3).to(device)


def load_alpha(path: Path, device: torch.device) -> torch.Tensor:
    return sealed.image_tensor(path, 1).to(device)


def existing_baseline_paths(formal_render: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Path]:
    safe_id = str(record["record_id"]).replace("/", "__")
    left, right = str(record["pair_id"]).split("_")
    root = Path(str(formal_render["oracle_rgb_path"])).parent
    return {
        "oracle_rgb": Path(str(formal_render["oracle_rgb_path"])),
        "oracle_alpha": root / f"{safe_id}__oracle_alpha.png",
        "source_rgb": root / f"{safe_id}__source_{left}_rgb.png",
        "target_rgb": root / f"{safe_id}__target_{right}_rgb.png",
    }


def run_render_seed(
    seed: int, formal_root: Path, diagnostic_root: Path, asset_root: Path, repository: Path
) -> dict[str, Any]:
    data = load_formal(formal_root)
    attempt = diagnostic_root / "attempt_001"
    result_path = attempt / f"metrics/counterfactual_seed_{seed}.json"
    if result_path.exists():
        result = read_json(result_path)
        print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2, sort_keys=True))
        return result
    manifest, _, _ = formal.contract()
    query_index = {row["record_id"]: row for row in manifest["query_sets"]}
    all_predictions = {row["record_id"]: row for row in data["mixed"][seed]["records"]}
    predictions = {record_id: row for record_id, row in all_predictions.items() if row["pair_correct"] is not None}
    formal_render = {
        row["record_id"]: row for row in data["render"][seed]["records"] if row["role"] == "mixed"
    }
    os.environ["CANONDRESSGS_ASSET_ROOT"] = str(asset_root)
    formal.configure_determinism(strict=False)
    sealed.RUN_BRANCH = RUN_BRANCH
    sealed.SOURCE_HEAD = SOURCE_HEAD
    seed_all(0)
    gate = NoTrainingGate()
    records = []
    created = Counter()
    reused = Counter({"ACTUAL_CONTROLLER": 0, "ORACLE_PAIR_ORACLE_WEIGHT": 0})
    with gate:
        runtime = sealed.EvaluationRuntime(attempt / f"audits/runtime_seed_{seed}_no_write", asset_root, {})
        endpoints = geometry.endpoint_residuals(runtime)
        device = runtime.device
        # The formal paired-render process instantiates the seeded Controller
        # after the immutable runtime and before the first render.  Although
        # checkpoint loading replaces its initialization, construction advances
        # the process RNG used by the legacy renderer.  Reproduce that read-only
        # ordering exactly; the model is not trained and receives no GT fields.
        _controller_rng_parity_model = evaluator.load_model(formal_root, seed, device)
        parity_max_abs = 0.0
        replay_count = 0
        for record_id, record in query_index.items():
            prediction = all_predictions[record_id]
            formal_row = formal_render[record_id]
            baseline_paths = existing_baseline_paths(formal_row, record)
            if not all(path.is_file() for path in baseline_paths.values()):
                raise FileNotFoundError(baseline_paths)
            left = str(record["pair_id"]).split("_")[0]
            condition = str(record["target_view_fold"])
            oracle_rgb = load_rgb(baseline_paths["oracle_rgb"], device)
            source_rgb = load_rgb(baseline_paths["source_rgb"], device)
            target_rgb = load_rgb(baseline_paths["target_rgb"], device)
            formal_candidate_rgb = load_rgb(Path(formal_row["rgb_path"]), device)
            # Reproduce the exact append-only FORMAL-002 per-record order in
            # memory.  This both advances any legacy runtime state identically
            # and proves that new variants share the same paired context.
            with torch.inference_mode():
                replay_candidate, _, _ = geometry.render_branches(
                    runtime, left, condition, evaluator.controller_branches(prediction, endpoints)
                )
                replay_oracle, _, _ = geometry.render_branches(
                    runtime, left, condition, evaluator.oracle_branches(record, endpoints)
                )
                replay_source, _, _ = geometry.render_branches(runtime, left, condition, ((left, endpoints[left], 1.0),))
                right = str(record["pair_id"]).split("_")[1]
                replay_target, _, _ = geometry.render_branches(runtime, left, condition, ((right, endpoints[right], 1.0),))
            current_parity = max(
                float((replay_candidate - formal_candidate_rgb).abs().max()),
                float((replay_oracle - oracle_rgb).abs().max()),
                float((replay_source - source_rgb).abs().max()),
                float((replay_target - target_rgb).abs().max()),
            )
            parity_max_abs = max(parity_max_abs, current_parity)
            replay_count += 4
            if current_parity > 1.0 / 255.0 + 1e-6:
                raise RuntimeError(f"counterfactual runtime parity failed record={record_id}: {current_parity}")
            if record_id not in predictions:
                continue
            actual = {
                "variant": "ACTUAL_CONTROLLER", "defined": True, "reused": True,
                "rgb_path": formal_row["rgb_path"], "alpha_path": formal_row["alpha_path"],
                "metrics": formal_row["metrics"], "render_diagnostics": formal_row["render_diagnostics"],
                "selected_endpoints": formal_row["selected_endpoints"], "weights": formal_row["predicted_weights"],
            }
            oracle_alpha = load_alpha(baseline_paths["oracle_alpha"], device)
            oracle_metrics = evaluator.render_metrics(runtime, left, condition, oracle_rgb, oracle_alpha, oracle_rgb, source_rgb, target_rgb)
            oracle = {
                "variant": "ORACLE_PAIR_ORACLE_WEIGHT", "defined": True, "reused": True,
                "rgb_path": str(baseline_paths["oracle_rgb"]), "alpha_path": str(baseline_paths["oracle_alpha"]),
                "metrics": oracle_metrics, "render_diagnostics": formal_row["oracle_render_diagnostics"],
                "selected_endpoints": [OUTFIT_ORDER[index] for index, value in enumerate(record["target_distribution"]) if float(value) > 0],
                "weights": [float(value) for value in record["target_distribution"] if float(value) > 0],
            }
            variants: dict[str, Any] = {"ACTUAL_CONTROLLER": actual, "ORACLE_PAIR_ORACLE_WEIGHT": oracle}
            reused["ACTUAL_CONTROLLER"] += 1
            reused["ORACLE_PAIR_ORACLE_WEIGHT"] += 1
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all()
            for variant in ("FORCE_DUAL_PRED_PAIR_PRED_WEIGHT", "ORACLE_PAIR_PREDICTED_PROBABILITY_WEIGHT", "CORRECT_PAIR_SUBSET_PRED_PAIR_ORACLE_WEIGHT", "TOP1_SINGLE"):
                branches = variant_branches(variant, record, prediction, endpoints)
                if branches is None:
                    variants[variant] = {"variant": variant, "defined": False, "reason": "WRONG_PAIR_QUERY"}
                    continue
                safe_id = str(record_id).replace("/", "__")
                path = attempt / f"counterfactuals/seed_{seed}/{variant}/{safe_id}"
                rgb, alpha, diagnostics, was_created = evaluator.render_or_load(path, runtime, left, condition, branches)
                created[variant] += int(was_created)
                reused[variant] += int(not was_created)
                metrics = evaluator.render_metrics(runtime, left, condition, rgb, alpha, oracle_rgb, source_rgb, target_rgb)
                variants[variant] = {
                    "variant": variant, "defined": True, "reused": not was_created,
                    "rgb_path": str(path.with_name(path.name + "_rgb.png")),
                    "alpha_path": str(path.with_name(path.name + "_alpha.png")),
                    "metrics": metrics, "render_diagnostics": diagnostics,
                    "selected_endpoints": [branch[0] for branch in branches], "weights": [float(branch[2]) for branch in branches],
                }
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state_all(cuda_rng)
            records.append({
                "seed": seed, "record_id": record_id, "pair_id": record["pair_id"],
                "composition": composition(record), "assignment_position": record["assignment_position"],
                "target_view_fold": condition, "pair_correct": bool(prediction["pair_correct"]),
                "ordering_correct": bool(prediction["ordering_correct"]), "formal_mode": prediction["mode"],
                "formal_fallback_reason": prediction["fallback_reason"], "probabilities": prediction["probabilities"],
                "variants": variants, "target_forward_leakage": 0,
                "ground_truth_used_in_actual_controller_forward": 0,
                "ground_truth_used_only_for_diagnostic_variants_and_metrics": True,
            })
    gate_result = gate.result()
    if gate_result["status"] != "PASS":
        raise RuntimeError(gate_result)
    result = {
        "schema_version": "canondressgs.research.controller_counterfactual_render_seed.v1",
        "status": "COMPLETE", "task_id": TASK_ID, "seed": seed, "record_count": len(records),
        "variant_definitions": list(VARIANTS), "records": records,
        "reused_count": dict(reused), "newly_rendered_count": dict(created),
        "runtime_parity_max_abs_vs_formal_png": parity_max_abs, "formal_sequence_replay_render_count": replay_count,
        "no_training_gate": gate_result, "formal_output_write_count": 0,
        "threshold_change": 0, "controller_change": 0, "paper_final": False,
    }
    write_once(result_path, result)
    write_once(attempt / f"audits/no_training_gate_render_seed_{seed}.json", gate_result)
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2, sort_keys=True))
    return result


def pooled_feature(rows: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid.to(dtype=torch.bool)
    if mask.ndim:
        selected = rows[mask]
    else:
        selected = rows
    if selected.numel() == 0:
        return torch.zeros(rows.shape[-1], dtype=rows.dtype)
    return selected.float().mean(dim=0)


def live_schema_and_logits(
    model: torch.nn.Module, rows: torch.Tensor, valid: torch.Tensor, device: torch.device
) -> tuple[dict[str, Any], torch.Tensor]:
    with torch.inference_mode():
        distribution = model(rows.to(device), valid.to(device))
    schema = evaluator.infer(model, rows, valid, device)
    return schema, distribution.logits.detach().float().cpu()


def perturbation_rows(
    variant: str, normal_rows: torch.Tensor, normal_valid: torch.Tensor,
    runtime: Any, base_images: Sequence[Any], base_masks: Sequence[Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    if variant == "reference_dropout":
        return normal_rows[:2], normal_valid[:2]
    if variant == "single_reference":
        return normal_rows[:1], normal_valid[:1]
    if variant == "assignment_permutation":
        return normal_rows[[2, 0, 1]], normal_valid[[2, 0, 1]]
    images = [image.copy() for image in base_images]
    masks = [mask.copy() for mask in base_masks]
    if variant == "grayscale":
        images = [sealed.frozen.fixed_grayscale(image) for image in images]
    elif variant == "hue":
        images = [sealed.frozen.hue_shift(image, 120.0) for image in images]
    elif variant == "blur":
        images = [sealed.frozen.c5_gaussian_blur(image, mask) for image, mask in zip(images, masks)]
    elif variant == "mask_erosion":
        masks = [sealed.ndimage.binary_erosion(mask, iterations=3).astype(bool) for mask in masks]
    elif variant == "mask_dilation":
        masks = [sealed.ndimage.binary_dilation(mask, iterations=3).astype(bool) for mask in masks]
    else:
        raise KeyError(variant)
    rows, valid = runtime.f2_rows(images, masks)
    return rows.detach().cpu(), valid.detach().cpu()


def run_perturbation_chain_seed(
    seed: int, formal_root: Path, diagnostic_root: Path, asset_root: Path, feature_cache: Path
) -> dict[str, Any]:
    data = load_formal(formal_root)
    attempt = diagnostic_root / "attempt_001"
    result_path = attempt / f"perturbations/perturbation_chain_seed_{seed}.json"
    if result_path.exists():
        result = read_json(result_path)
        print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2, sort_keys=True))
        return result
    manifest, _, _ = formal.contract()
    query_index = {row["record_id"]: row for row in manifest["query_sets"]}
    predictions = {row["record_id"]: row for row in data["mixed"][seed]["records"]}
    formal_perturb = data["perturb"][seed]
    perturb_index = {(row["record_id"], row["variant"]): row for row in formal_perturb["records"]}
    representatives = sorted({record_id for record_id, _ in perturb_index})
    if len(representatives) != 20:
        raise RuntimeError("perturbation representative count mismatch")
    cache = torch.load(feature_cache, map_location="cpu", weights_only=False)
    os.environ["CANONDRESSGS_ASSET_ROOT"] = str(asset_root)
    formal.configure_determinism(strict=False)
    sealed.RUN_BRANCH = RUN_BRANCH
    sealed.SOURCE_HEAD = SOURCE_HEAD
    seed_all(0)
    gate = NoTrainingGate()
    records = []
    inference_count = 0
    with gate:
        runtime = sealed.EvaluationRuntime(attempt / f"audits/perturb_runtime_seed_{seed}_no_write", asset_root, {})
        model = evaluator.load_model(formal_root, seed, torch.device("cuda"))
        device = torch.device("cuda")
        for record_id in representatives:
            record = query_index[record_id]
            baseline_prediction = predictions[record_id]
            normal_rows, normal_valid = formal.case_rows(cache, record)
            baseline_schema, baseline_logits = live_schema_and_logits(model, normal_rows, normal_valid, device)
            inference_count += 2
            maximum = max(abs(float(a) - float(b)) for a, b in zip(baseline_schema["probabilities"], baseline_prediction["probabilities"]))
            if maximum > 1e-7:
                raise RuntimeError("persisted/live perturbation baseline mismatch")
            condition = record["target_view_fold"]
            observations = [runtime.observation(outfit, view) for outfit, view in zip(record["garment_labels"], record["reference_condition_ids"])]
            base_images = [item[0] for item in observations]
            base_masks = [item[1] for item in observations]
            baseline_feature = pooled_feature(normal_rows, normal_valid)
            baseline_probabilities = torch.tensor(baseline_schema["probabilities"], dtype=torch.float64)
            for variant in formal_perturb["aggregates"]:
                rows, valid = perturbation_rows(variant, normal_rows, normal_valid, runtime, base_images, base_masks)
                schema, logits = live_schema_and_logits(model, rows, valid, device)
                inference_count += 2
                prediction = evaluator.prediction_row(record, schema)
                feature = pooled_feature(rows, valid)
                cosine = float(torch.nn.functional.cosine_similarity(baseline_feature[None], feature[None]).item())
                l2 = float(torch.linalg.vector_norm(feature - baseline_feature).item())
                logit_l2 = float(torch.linalg.vector_norm(logits - baseline_logits).item())
                probabilities = torch.tensor(schema["probabilities"], dtype=torch.float64)
                probability_kl = float(torch.sum(baseline_probabilities * torch.log((baseline_probabilities + 1e-12) / (probabilities + 1e-12))).item())
                archived = perturb_index[(record_id, variant)]
                archived_pair = sorted(archived["perturbed_pair"])
                live_pair = sorted((prediction["top1_outfit"], prediction["top2_outfit"]))
                if archived_pair != live_pair or archived["perturbed_mode"] != prediction["mode"]:
                    raise RuntimeError("perturbation live/archive mismatch")
                records.append({
                    "seed": seed, "record_id": record_id, "pair_id": record["pair_id"], "composition": composition(record),
                    "variant": variant, "feature_cosine_drift": 1.0 - cosine, "feature_l2_drift": l2,
                    "logit_l2_drift": logit_l2, "probability_kl": probability_kl,
                    "top1_flip": baseline_prediction["top1_outfit"] != prediction["top1_outfit"],
                    "top2_pair_flip": set((baseline_prediction["top1_outfit"], baseline_prediction["top2_outfit"])) != set((prediction["top1_outfit"], prediction["top2_outfit"])),
                    "ordering_flip": bool(baseline_prediction["ordering_correct"]) != bool(prediction["ordering_correct"]),
                    "secondary_weight_drift": float(prediction["normalized_secondary_weight"]) - float(baseline_prediction["normalized_secondary_weight"]),
                    "secondary_weight_absolute_drift": abs(float(prediction["normalized_secondary_weight"]) - float(baseline_prediction["normalized_secondary_weight"])),
                    "top2_mass_drift": float(prediction["top2_mass"]) - float(baseline_prediction["top2_mass"]),
                    "mode_flip": baseline_prediction["mode"] != prediction["mode"],
                    "fallback_reason_flip": baseline_prediction["fallback_reason"] != prediction["fallback_reason"],
                    "render_lpips_drift": float(archived["lpips_change"]),
                    "silhouette_iou_drift": float(archived["silhouette_iou_change"]),
                    "artifact_grade_drift": None,
                    "artifact_grade_drift_status": "MANUAL_DIAGNOSTIC_SHEET_REQUIRED",
                    "perturbed_rgb_path": archived["rgb_path"], "perturbed_alpha_path": archived["alpha_path"],
                })
    gate_result = gate.result()
    if gate_result["status"] != "PASS":
        raise RuntimeError(gate_result)
    aggregates = {}
    for variant in sorted(formal_perturb["aggregates"]):
        selected = [row for row in records if row["variant"] == variant]
        values = {
            "record_count": len(selected),
            "feature_cosine_drift_mean": mean(float(row["feature_cosine_drift"]) for row in selected),
            "feature_l2_drift_mean": mean(float(row["feature_l2_drift"]) for row in selected),
            "logit_l2_drift_mean": mean(float(row["logit_l2_drift"]) for row in selected),
            "probability_kl_mean": mean(float(row["probability_kl"]) for row in selected),
            "top1_flip_rate": mean(float(row["top1_flip"]) for row in selected),
            "top2_pair_flip_rate": mean(float(row["top2_pair_flip"]) for row in selected),
            "ordering_flip_rate": mean(float(row["ordering_flip"]) for row in selected),
            "secondary_weight_absolute_drift_mean": mean(float(row["secondary_weight_absolute_drift"]) for row in selected),
            "top2_mass_absolute_drift_mean": mean(abs(float(row["top2_mass_drift"])) for row in selected),
            "mode_flip_rate": mean(float(row["mode_flip"]) for row in selected),
            "fallback_reason_flip_rate": mean(float(row["fallback_reason_flip"]) for row in selected),
            "render_lpips_drift_mean": mean(float(row["render_lpips_drift"]) for row in selected),
        }
        if values["top2_pair_flip_rate"] >= 0.20 and values["mode_flip_rate"] >= 0.20:
            stage = "MULTIPLE_STAGES"
        elif values["top2_pair_flip_rate"] >= 0.20:
            stage = "PAIR_SELECTION"
        elif values["mode_flip_rate"] >= 0.20 or values["fallback_reason_flip_rate"] >= 0.20:
            stage = "FALLBACK_GATE"
        elif values["secondary_weight_absolute_drift_mean"] >= 0.05:
            stage = "WEIGHT_CALIBRATION"
        elif values["feature_cosine_drift_mean"] >= 0.01:
            stage = "FEATURE_EXTRACTION"
        elif abs(values["render_lpips_drift_mean"]) >= 0.01:
            stage = "RENDER_ONLY"
        else:
            stage = "FEATURE_EXTRACTION"
        values["primary_affected_stage"] = stage
        aggregates[variant] = values
    result = {
        "schema_version": "canondressgs.research.controller_perturbation_failure_chain_seed.v1",
        "status": "COMPLETE", "task_id": TASK_ID, "seed": seed, "record_count": len(records),
        "controller_inference_forward_count": inference_count, "records": records, "aggregates": aggregates,
        "no_training_gate": gate_result, "formal_perturbation_render_reuse_count": len(records),
        "new_perturbation_render_count": 0, "paper_final": False,
    }
    write_once(result_path, result)
    write_once(attempt / f"audits/no_training_gate_perturbation_seed_{seed}.json", gate_result)
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2, sort_keys=True))
    return result


def improvement(base: Mapping[str, float], fixed: Mapping[str, float], field: str) -> float:
    if field in {"silhouette_iou", "boundary_fscore"}:
        return float(fixed[field]) - float(base[field])
    return float(base[field]) - float(fixed[field])


def aggregate_counterfactual(diagnostic_root: Path, formal_root: Path) -> dict[str, Any]:
    attempt = diagnostic_root / "attempt_001"
    seeds = [read_json(attempt / f"metrics/counterfactual_seed_{seed}.json") for seed in SEEDS]
    records = [row for payload in seeds for row in payload["records"]]
    if len(records) != 720:
        raise RuntimeError("counterfactual logical count mismatch")
    variant_summary = {}
    for variant in VARIANTS:
        selected = [row["variants"][variant] for row in records if row["variants"][variant]["defined"]]
        variant_summary[variant] = {
            "logical_count": len(selected),
            "reused_count": sum(item["reused"] for item in selected),
            "newly_rendered_count": sum(not item["reused"] for item in selected),
            "metric_means": {field: mean(float(item["metrics"][field]) for item in selected) for field in METRIC_FIELDS},
            "opacity_mass_mean": mean(float(item["render_diagnostics"]["opacity_mass"]) for item in selected),
            "active_gaussian_mean": mean(float(item["render_diagnostics"]["active_gaussian_count"]) for item in selected),
            "render_time_seconds_mean": mean(float(item["render_diagnostics"]["render_time_seconds"]) for item in selected),
            "peak_vram_bytes_max": max(float(item["render_diagnostics"]["peak_vram_bytes"]) for item in selected),
        }
    comparisons = {
        "PAIR_FIX_GAIN": ("FORCE_DUAL_PRED_PAIR_PRED_WEIGHT", "ORACLE_PAIR_PREDICTED_PROBABILITY_WEIGHT"),
        "WEIGHT_FIX_GAIN": ("FORCE_DUAL_PRED_PAIR_PRED_WEIGHT", "CORRECT_PAIR_SUBSET_PRED_PAIR_ORACLE_WEIGHT"),
        "FALLBACK_REMOVAL_GAIN": ("ACTUAL_CONTROLLER", "FORCE_DUAL_PRED_PAIR_PRED_WEIGHT"),
        "FULL_ORACLE_GAIN": ("ACTUAL_CONTROLLER", "ORACLE_PAIR_ORACLE_WEIGHT"),
    }
    attribution = {}
    for name, (base_name, fixed_name) in comparisons.items():
        selected = [row for row in records if row["variants"][base_name]["defined"] and row["variants"][fixed_name]["defined"]]
        attribution[name] = {
            "record_count": len(selected),
            "metric_improvement_means": {
                field: mean(improvement(row["variants"][base_name]["metrics"], row["variants"][fixed_name]["metrics"], field) for row in selected)
                for field in METRIC_FIELDS
            },
            "lpips_improved_fraction": mean(
                float(row["variants"][fixed_name]["metrics"]["garment_lpips"] < row["variants"][base_name]["metrics"]["garment_lpips"])
                for row in selected
            ),
        }
    source_files = [
        formal_root / f"attempt_001/seed_{seed}/mixed/mixed_classification.json" for seed in SEEDS
    ] + [formal_root / f"attempt_004/seed_{seed}/metrics/render_evaluation.json" for seed in SEEDS]
    result = {
        "schema_version": "canondressgs.research.controller_counterfactual_render_results.v1",
        "status": "COMPLETE", "task_id": TASK_ID, "record_count": len(records),
        "variant_summary": variant_summary, "numerical_attribution": attribution, "records": records,
        "reuse": {
            "actual_controller_regenerated": 0, "oracle_pair_oracle_weight_regenerated": 0,
            "source_paths": [str(path) for path in source_files],
            "source_hashes": {str(path): sha256(path) for path in source_files},
        },
        "training_steps": 0, "optimizer_created": 0, "checkpoint_writes": 0,
        "ground_truth_used_in_actual_controller_forward": 0,
        "ground_truth_used_only_for_diagnostic_counterfactuals": True,
        "paper_final": False,
    }
    write_once(attempt / "aggregates/controller_counterfactual_render_results.json", result)
    return result


def aggregate_perturbation(diagnostic_root: Path) -> dict[str, Any]:
    attempt = diagnostic_root / "attempt_001"
    seeds = [read_json(attempt / f"perturbations/perturbation_chain_seed_{seed}.json") for seed in SEEDS]
    records = [row for payload in seeds for row in payload["records"]]
    if len(records) != 480:
        raise RuntimeError("perturbation chain count mismatch")
    aggregates = {}
    for variant in sorted({row["variant"] for row in records}):
        selected = [row for row in records if row["variant"] == variant]
        aggregate = {
            "record_count": len(selected),
            "feature_cosine_drift_mean": mean(float(row["feature_cosine_drift"]) for row in selected),
            "feature_l2_drift_mean": mean(float(row["feature_l2_drift"]) for row in selected),
            "logit_l2_drift_mean": mean(float(row["logit_l2_drift"]) for row in selected),
            "probability_kl_mean": mean(float(row["probability_kl"]) for row in selected),
            "top1_flip_rate": mean(float(row["top1_flip"]) for row in selected),
            "top2_pair_flip_rate": mean(float(row["top2_pair_flip"]) for row in selected),
            "ordering_flip_rate": mean(float(row["ordering_flip"]) for row in selected),
            "secondary_weight_absolute_drift_mean": mean(float(row["secondary_weight_absolute_drift"]) for row in selected),
            "top2_mass_absolute_drift_mean": mean(abs(float(row["top2_mass_drift"])) for row in selected),
            "mode_flip_rate": mean(float(row["mode_flip"]) for row in selected),
            "fallback_reason_flip_rate": mean(float(row["fallback_reason_flip"]) for row in selected),
            "render_lpips_drift_mean": mean(float(row["render_lpips_drift"]) for row in selected),
        }
        seed_stages = [payload["aggregates"][variant]["primary_affected_stage"] for payload in seeds]
        aggregate["primary_affected_stage"] = Counter(seed_stages).most_common(1)[0][0]
        aggregate["seed_stage_votes"] = dict(Counter(seed_stages))
        aggregates[variant] = aggregate
    result = {
        "schema_version": "canondressgs.research.controller_perturbation_failure_chain.v1",
        "status": "COMPLETE", "task_id": TASK_ID, "record_count": len(records),
        "aggregates": aggregates, "records": records,
        "formal_perturbation_render_reuse_count": len(records), "new_render_count": 0,
        "artifact_grade_drift": "MANUAL_DIAGNOSTIC_SUBSET_RECORDED_IN_VISUAL_REVIEW",
        "training_steps": 0, "optimizer_created": 0, "paper_final": False,
    }
    write_once(attempt / "aggregates/controller_perturbation_failure_chain.json", result)
    return result


def pair_compatibility(
    counterfactual: Mapping[str, Any], repository: Path, diagnostic_root: Path
) -> dict[str, Any]:
    all_pair = read_json(repository / "paper_protocol/reviewer_risk/dual_support_all_pair_results.json")
    all_pair_visual = read_json(repository / "paper_protocol/reviewer_risk/dual_support_all_pair_visual_review.json")
    formal_visual = read_json(repository / "paper_protocol/reviewer_risk/dual_support_controller_visual_review.json")
    oracle_metrics = {
        row["pair_id"]: row["means"]
        for row in all_pair["analysis"]["aggregates"] if row["variant"] == "DUAL_SUPPORT_GEOMETRY_BLEND"
    }
    historical_grades: dict[str, list[int]] = defaultdict(list)
    for item in all_pair_visual["items"]:
        if item.get("kind") != "MAIN_SHEET":
            continue
        for row in item["grades_by_variant_alpha_view"]:
            if row["variant"] == "DUAL_SUPPORT_GEOMETRY_BLEND":
                historical_grades[str(item["pair_id"])].append(max(int(row["grades"][key]) for key in CONTAMINATION))
    actual_grades: dict[str, list[int]] = defaultdict(list)
    for row in formal_visual["records"]:
        if row.get("kind") == "MIXED_MAIN":
            actual_grades[str(row["pair_id"])].append(max(int(row["grades"][key]) for key in CONTAMINATION))
    rows = counterfactual["records"]
    pair_rows = []
    for pair in sorted(oracle_metrics):
        selected = [row for row in rows if row["pair_id"] == pair]
        predictions = [row for row in selected]
        weight_errors = []
        for row in predictions:
            target_weights = row["variants"]["ORACLE_PAIR_ORACLE_WEIGHT"]["weights"]
            predicted = row["variants"]["ORACLE_PAIR_PREDICTED_PROBABILITY_WEIGHT"]["weights"]
            weight_errors.append(abs(float(predicted[0]) - float(target_weights[0])))
        pair_rows.append({
            "pair_id": pair,
            "historical_oracle_lpips": oracle_metrics[pair]["garment_lpips"],
            "historical_oracle_iou": oracle_metrics[pair]["silhouette_iou"],
            "historical_oracle_boundary": oracle_metrics[pair]["boundary_fscore"],
            "historical_oracle_severe_count": sum(value == 3 for value in historical_grades[pair]),
            "historical_oracle_grade_max": max(historical_grades[pair]),
            "actual_controller_activation_rate": mean(float(row["formal_mode"] == "DUAL_SUPPORT") for row in selected),
            "pair_prediction_accuracy": mean(float(row["pair_correct"]) for row in selected),
            "weight_mae": mean(weight_errors),
            "wrong_pair_exposure_count": sum(not row["pair_correct"] and row["formal_mode"] == "DUAL_SUPPORT" for row in selected),
            "formal_actual_severe_sheet_count": sum(value == 3 for value in actual_grades[pair]),
            "formal_actual_grade_max": max(actual_grades[pair]),
            "exact_oracle_lpips_mean": mean(float(row["variants"]["ORACLE_PAIR_ORACLE_WEIGHT"]["metrics"]["garment_lpips"]) for row in selected),
            "force_dual_lpips_mean": mean(float(row["variants"]["FORCE_DUAL_PRED_PAIR_PRED_WEIGHT"]["metrics"]["garment_lpips"]) for row in selected),
        })
    result = {
        "schema_version": "canondressgs.research.controller_pair_compatibility_analysis.v1",
        "status": "COMPLETE", "task_id": TASK_ID, "pair_count": len(pair_rows), "pairs": pair_rows,
        "focus_pairs": [row for row in pair_rows if row["pair_id"] in {"O01_O03", "O02_O03"}],
        "pair_blacklist_implemented": False, "compatibility_gate_implemented": False,
        "interpretation_pending_manual_exact_oracle_review": True, "paper_final": False,
    }
    write_once(diagnostic_root / "attempt_001/aggregates/controller_pair_compatibility_analysis.json", result)
    return result


def font(size: int) -> ImageFont.ImageFont:
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def panel(path: str | Path | None, label: str, size: tuple[int, int] = (190, 240)) -> Image.Image:
    width, height = size
    canvas = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(canvas)
    if path is not None and Path(path).is_file():
        image = Image.open(path).convert("RGB")
        image.thumbnail((width - 8, height - 52), Image.Resampling.LANCZOS)
        canvas.paste(image, ((width - image.width) // 2, 42 + (height - 48 - image.height) // 2))
    else:
        draw.rectangle((4, 42, width - 4, height - 4), outline="gray", width=2)
        draw.text((12, 100), "NOT DEFINED", fill="gray", font=font(14))
    draw.multiline_text((5, 4), label, fill="black", font=font(12), spacing=2)
    return canvas


def save_sheet(path: Path, title: str, rows: Sequence[Sequence[Image.Image]]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = max(len(row) for row in rows)
    cell_w, cell_h = rows[0][0].size
    canvas = Image.new("RGB", (columns * cell_w, 54 + len(rows) * cell_h), "white")
    ImageDraw.Draw(canvas).text((10, 10), title, fill="black", font=font(20))
    for y, row in enumerate(rows):
        for x, image in enumerate(row):
            canvas.paste(image, (x * cell_w, 54 + y * cell_h))
    temporary = path.with_name(path.name + ".tmp.png")
    canvas.save(temporary)
    temporary.replace(path)


def record_endpoints(row: Mapping[str, Any]) -> tuple[Path, Path]:
    safe = str(row["record_id"]).replace("/", "__")
    left, right = str(row["pair_id"]).split("_")
    oracle = Path(row["variants"]["ORACLE_PAIR_ORACLE_WEIGHT"]["rgb_path"])
    root = oracle.parent
    return root / f"{safe}__source_{left}_rgb.png", root / f"{safe}__target_{right}_rgb.png"


def create_causal_visuals(
    counterfactual: Mapping[str, Any], perturbation: Mapping[str, Any], diagnostic_root: Path
) -> dict[str, Any]:
    root = diagnostic_root / "attempt_001/visuals"
    records = counterfactual["records"]
    main_paths = []
    pairs = sorted({str(row["pair_id"]) for row in records})
    for seed in SEEDS:
        for pair in pairs:
            for comp in ("AAB", "ABB"):
                selected = sorted(
                    [row for row in records if row["seed"] == seed and row["pair_id"] == pair and row["composition"] == comp and row["assignment_position"] == 0],
                    key=lambda row: row["target_view_fold"],
                )
                if len(selected) != 4:
                    raise RuntimeError(f"causal sheet fold count mismatch {seed}/{pair}/{comp}")
                sheet_rows = []
                for row in selected:
                    source, target = record_endpoints(row)
                    variants = row["variants"]
                    sheet_rows.append([
                        panel(source, f"SOURCE | {row['target_view_fold']}"),
                        panel(variants["ACTUAL_CONTROLLER"]["rgb_path"], f"ACTUAL | {row['formal_mode']}"),
                        panel(variants["FORCE_DUAL_PRED_PAIR_PRED_WEIGHT"]["rgb_path"], "FORCE DUAL pred/pred"),
                        panel(variants["ORACLE_PAIR_PREDICTED_PROBABILITY_WEIGHT"]["rgb_path"], "ORACLE pair / pred weight"),
                        panel(variants["CORRECT_PAIR_SUBSET_PRED_PAIR_ORACLE_WEIGHT"].get("rgb_path"), "PRED pair / oracle weight"),
                        panel(variants["ORACLE_PAIR_ORACLE_WEIGHT"]["rgb_path"], "ORACLE pair / oracle weight"),
                        panel(variants["TOP1_SINGLE"]["rgb_path"], "TOP1 SINGLE"),
                        panel(target, "TARGET ENDPOINT"),
                    ])
                path = root / f"causal_main/seed_{seed}_{pair}_{comp}.png"
                save_sheet(path, f"Seed {seed} | {pair} | {comp} | assignment 0 | four folds", sheet_rows)
                main_paths.append(str(path))

    def diagnostic(name: str, title: str, selected: Sequence[Mapping[str, Any]]) -> str:
        rows = []
        for row in selected[:12]:
            variants = row["variants"]
            rows.append([
                panel(variants["ACTUAL_CONTROLLER"]["rgb_path"], f"ACTUAL s{row['seed']} {row['pair_id']} {row['composition']}"),
                panel(variants["FORCE_DUAL_PRED_PAIR_PRED_WEIGHT"]["rgb_path"], "FORCE DUAL"),
                panel(variants["ORACLE_PAIR_PREDICTED_PROBABILITY_WEIGHT"]["rgb_path"], "PAIR FIX"),
                panel(variants["CORRECT_PAIR_SUBSET_PRED_PAIR_ORACLE_WEIGHT"].get("rgb_path"), "WEIGHT FIX"),
                panel(variants["ORACLE_PAIR_ORACLE_WEIGHT"]["rgb_path"], "FULL ORACLE"),
                panel(variants["TOP1_SINGLE"]["rgb_path"], "TOP1 SINGLE"),
            ])
        path = root / f"diagnostics/{name}.png"
        save_sheet(path, title, rows)
        return str(path)

    diagnostics = []
    false_negative = [row for row in records if row["pair_correct"] and row["ordering_correct"] and row["formal_mode"] == "SINGLE_ENDPOINT"]
    false_negative.sort(key=lambda row: row["variants"]["ACTUAL_CONTROLLER"]["metrics"]["garment_lpips"], reverse=True)
    diagnostics.append(diagnostic("fallback_false_negatives", "Correct pair/order but frozen gate falls back", false_negative))
    wrong_pair = [row for row in records if not row["pair_correct"]]
    wrong_pair.sort(key=lambda row: row["variants"]["FORCE_DUAL_PRED_PAIR_PRED_WEIGHT"]["metrics"]["garment_lpips"], reverse=True)
    diagnostics.append(diagnostic("wrong_pair_cases", "Wrong predicted-pair counterfactuals", wrong_pair))
    weight_cases = [row for row in records if row["pair_correct"]]
    weight_cases.sort(
        key=lambda row: row["variants"]["FORCE_DUAL_PRED_PAIR_PRED_WEIGHT"]["metrics"]["garment_lpips"] - row["variants"]["CORRECT_PAIR_SUBSET_PRED_PAIR_ORACLE_WEIGHT"]["metrics"]["garment_lpips"],
        reverse=True,
    )
    diagnostics.append(diagnostic("correct_pair_wrong_weight", "Largest oracle-weight gains on correct pairs", weight_cases))
    oracle_cases = sorted(records, key=lambda row: row["variants"]["ORACLE_PAIR_ORACLE_WEIGHT"]["metrics"]["outside_garment_opacity"], reverse=True)
    diagnostics.append(diagnostic("oracle_residual_failures", "Highest outside-garment opacity under exact Oracle", oracle_cases))
    diagnostics.append(diagnostic("O01_O03_focus", "O01_O03 exact causal decomposition", [row for row in records if row["pair_id"] == "O01_O03"]))
    diagnostics.append(diagnostic("O02_O03_focus", "O02_O03 exact causal decomposition", [row for row in records if row["pair_id"] == "O02_O03"]))

    perturb_records = perturbation["records"]
    for field, name, title in (
        ("top2_pair_flip", "perturbation_pair_flips", "Perturbation top-2 pair flips"),
        ("mode_flip", "perturbation_mode_flips", "Perturbation mode flips"),
    ):
        selected = [row for row in perturb_records if row[field]]
        selected.sort(key=lambda row: abs(float(row["render_lpips_drift"])), reverse=True)
        sheet_rows = []
        counter_index = {(row["seed"], row["record_id"]): row for row in records}
        for row in selected[:16]:
            baseline = counter_index[(row["seed"], row["record_id"])]["variants"]["ACTUAL_CONTROLLER"]["rgb_path"]
            sheet_rows.append([
                panel(baseline, f"BASE s{row['seed']} {row['pair_id']}"),
                panel(row["perturbed_rgb_path"], f"{row['variant']} | LPIPS drift {row['render_lpips_drift']:.3f}"),
            ])
        path = root / f"diagnostics/{name}.png"
        save_sheet(path, title, sheet_rows)
        diagnostics.append(str(path))

    manifest = {
        "schema_version": "canondressgs.research.controller_calibration_visual_manifest.v1",
        "status": "GENERATED_MANUAL_OPENING_REQUIRED", "task_id": TASK_ID,
        "causal_main_count": len(main_paths), "causal_main": main_paths,
        "diagnostic_count": len(diagnostics), "diagnostics": diagnostics,
        "all_paths_exist": all(Path(path).is_file() for path in main_paths + diagnostics),
        "manual_open_count": 0, "paper_final": False,
    }
    write_once(root / "visual_manifest.json", manifest)
    return manifest


def diagnostic_plots(diagnostic_root: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    attempt = diagnostic_root / "attempt_001"
    output = attempt / "visuals/diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    separability = read_json(attempt / "aggregates/controller_pure_mixed_separability.json")
    threshold = read_json(attempt / "aggregates/controller_threshold_sweep_diagnostic.json")
    weights = read_json(attempt / "aggregates/controller_weight_calibration_analysis.json")
    paths = []

    path = output / "pure_mixed_separability.png"
    if not path.exists():
        names = list(separability["metrics"])
        fig, axis = plt.subplots(figsize=(10, 5))
        axis.bar(np.arange(len(names)) - 0.18, [separability["metrics"][name]["auroc"] for name in names], 0.36, label="AUROC")
        axis.bar(np.arange(len(names)) + 0.18, [separability["metrics"][name]["auprc"] for name in names], 0.36, label="AUPRC")
        axis.set_xticks(range(len(names)), names, rotation=25, ha="right"); axis.set_ylim(0, 1); axis.legend(); axis.grid(axis="y", alpha=.2)
        fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    paths.append(str(path))

    path = output / "threshold_grid_pareto.png"
    if not path.exists():
        fig, axis = plt.subplots(figsize=(7, 6))
        for row in threshold["grid"]:
            axis.scatter(row["pure_single_rate"], row["mixed_dual_rate"], c=row["wrong_pair_dual_rate"], vmin=0, vmax=1, cmap="viridis", s=40)
        axis.set_xlabel("Pure SINGLE rate"); axis.set_ylabel("Mixed DUAL rate"); axis.set_xlim(0, 1.02); axis.set_ylim(0, 1.02); axis.grid(alpha=.2)
        axis.set_title(f"Frozen grid only | {threshold['simple_gate_status']}")
        fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    paths.append(str(path))

    path = output / "weight_calibration_diagnostics.png"
    if not path.exists():
        records = [row for row in weights["records"] if row["pair_correct"]]
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes[0, 0].scatter([2/3] * len(records), [row["predicted_dominant_weight"] for row in records], s=8, alpha=.3)
        axes[0, 0].plot([.5, .8], [.5, .8], "k--"); axes[0, 0].set(xlabel="Target dominant weight", ylabel="Predicted")
        pair_names = sorted({row["pair_id"] for row in records})
        axes[0, 1].boxplot([[row["predicted_dominant_weight"] for row in records if row["pair_id"] == pair] for pair in pair_names], labels=pair_names)
        axes[0, 1].tick_params(axis="x", rotation=35); axes[0, 1].set_title("Per-pair")
        seed_names = [str(seed) for seed in SEEDS]
        axes[1, 0].bar(seed_names, [mean(row["absolute_error"] for row in records if row["seed"] == seed) for seed in SEEDS]); axes[1, 0].set_title("MAE by seed")
        fold_names = sorted({row["target_view_fold"] for row in records})
        axes[1, 1].bar(fold_names, [mean(row["absolute_error"] for row in records if row["target_view_fold"] == fold) for fold in fold_names]); axes[1, 1].tick_params(axis="x", rotation=25); axes[1, 1].set_title("MAE by fold")
        fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    paths.append(str(path))
    return paths


def run_finalize(formal_root: Path, diagnostic_root: Path, repository: Path) -> dict[str, Any]:
    counterfactual = aggregate_counterfactual(diagnostic_root, formal_root)
    perturbation = aggregate_perturbation(diagnostic_root)
    compatibility = pair_compatibility(counterfactual, repository, diagnostic_root)
    manifest = create_causal_visuals(counterfactual, perturbation, diagnostic_root)
    plot_paths = diagnostic_plots(diagnostic_root)
    manifest["diagnostics"].extend(plot_paths)
    manifest["diagnostic_count"] = len(manifest["diagnostics"])
    manifest["all_paths_exist"] = all(Path(path).is_file() for path in manifest["causal_main"] + manifest["diagnostics"])
    manifest_path = diagnostic_root / "attempt_001/visuals/visual_manifest_with_plots.json"
    write_once(manifest_path, manifest)
    result = {
        "status": "COMPLETE_MANUAL_REVIEW_REQUIRED", "counterfactual_records": counterfactual["record_count"],
        "perturbation_records": perturbation["record_count"], "pair_count": compatibility["pair_count"],
        "causal_main_count": manifest["causal_main_count"], "diagnostic_count": manifest["diagnostic_count"],
        "visual_manifest": str(manifest_path),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "offline", "render-seed", "perturbation-seed", "finalize"))
    parser.add_argument("--formal-output", type=Path, required=True)
    parser.add_argument("--diagnostic-output", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--repository", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    if args.phase == "preflight":
        if args.asset_root is None:
            parser.error("preflight requires --asset-root")
        run_preflight(args.formal_output.resolve(), args.diagnostic_output.resolve(), args.asset_root.resolve(), args.repository.resolve())
    elif args.phase == "offline":
        run_offline(args.formal_output.resolve(), args.diagnostic_output.resolve(), args.repository.resolve())
    elif args.phase == "render-seed":
        if args.seed is None or args.asset_root is None:
            parser.error("render-seed requires --seed and --asset-root")
        run_render_seed(args.seed, args.formal_output.resolve(), args.diagnostic_output.resolve(), args.asset_root.resolve(), args.repository.resolve())
    elif args.phase == "perturbation-seed":
        if args.seed is None or args.asset_root is None or args.feature_cache is None:
            parser.error("perturbation-seed requires --seed, --asset-root, and --feature-cache")
        run_perturbation_chain_seed(args.seed, args.formal_output.resolve(), args.diagnostic_output.resolve(), args.asset_root.resolve(), args.feature_cache.resolve())
    else:
        run_finalize(args.formal_output.resolve(), args.diagnostic_output.resolve(), args.repository.resolve())


if __name__ == "__main__":
    main()
