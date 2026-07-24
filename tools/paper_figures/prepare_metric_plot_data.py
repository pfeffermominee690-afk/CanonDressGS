#!/usr/bin/env python3
"""Normalize sealed structured metrics into deterministic plotting records."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--dual-summary", type=Path, required=True)
    parser.add_argument("--controller-trajectory", type=Path, required=True)
    parser.add_argument("--controller-budget-summary", type=Path, required=True)
    parser.add_argument("--subject-step0", type=Path, required=True)
    parser.add_argument("--subject-step384", type=Path, required=True)
    parser.add_argument("--subject-medium", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def stable_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def mean_error(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        raise ValueError("empty metric denominator")
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def historical_metric(root: Path, experiment_glob: str, metric: str) -> Tuple[List[float], List[str]]:
    paths = sorted(root.glob(f"{experiment_glob}/seed_*/attempt_*/evaluated_metrics/evaluated_metrics.json"))
    values: List[float] = []
    used: List[str] = []
    seen_seeds = set()
    for path in reversed(paths):
        data = load(path)
        seed = data.get("seed")
        if data.get("status") != "PASS" or seed in seen_seeds or metric not in data.get("metrics", {}):
            continue
        seen_seeds.add(seed)
        values.append(float(data["metrics"][metric]))
        used.append(str(path))
    return values, sorted(used)


def historical_series(
    root: Path,
    entries: Iterable[Tuple[float, str]],
    metric: str,
) -> Tuple[List[Dict[str, float]], List[str]]:
    points: List[Dict[str, float]] = []
    source_files: List[str] = []
    for x, pattern in entries:
        values, paths = historical_metric(root, pattern, metric)
        mean, error = mean_error(values)
        points.append({"x": x, "y": mean, "yerr": error, "seed_count": len(values)})
        source_files.extend(paths)
    return points, sorted(set(source_files))


def plot_historical_ablation(root: Path) -> Dict[str, Any]:
    entries = [
        (0, "PAPER-OURS-S*"), (1, "PAPER-A2-S*"), (2, "PAPER-A3-S*"),
        (3, "PAPER-A4-S*"), (4, "PAPER-A5-S*"), (5, "PAPER-A6-S*"),
    ]
    values, sources = historical_series(root, entries, "edit_reduction")
    return {
        "plot_id": "historical_ablation_edit_reduction",
        "kind": "bar",
        "title": "Historical representation/objective ablations",
        "x_label": "Frozen historical condition",
        "y_label": "Edit reduction",
        "x_ticks": [
            {"value": 0, "label": "Ours"}, {"value": 1, "label": "A2 mask"},
            {"value": 2, "label": "A3 mean"}, {"value": 3, "label": "A4 std"},
            {"value": 4, "label": "A5 legacy"}, {"value": 5, "label": "A6 no pair"},
        ],
        "series": [{"label": "mean +/- seed SD", "values": values}],
        "source_files": sources,
        "claim_status": "HISTORICAL_TRANSDUCTIVE_ABLATION | REQUIRES_MANUAL_ADJUDICATION",
    }


def plot_rank(root: Path) -> Dict[str, Any]:
    values, sources = historical_series(
        root,
        [(rank, f"PAPER-A1-K{rank}-S*") for rank in (1, 2, 3, 4)],
        "edit_reduction",
    )
    return {
        "plot_id": "rank_1_4_edit_reduction",
        "kind": "line",
        "title": "Historical explicit-basis rank ablation",
        "x_label": "Basis rank",
        "y_label": "Edit reduction",
        "x_ticks": [{"value": rank, "label": str(rank)} for rank in (1, 2, 3, 4)],
        "series": [{"label": "mean +/- seed SD", "values": values}],
        "source_files": sources,
        "claim_status": "HISTORICAL_TRANSDUCTIVE_ABLATION | READY_FROM_HISTORICAL_EVIDENCE",
    }


def plot_reference_count(root: Path) -> Dict[str, Any]:
    values, sources = historical_series(
        root,
        [(count, f"PAPER-A7-KREF{count}-S*") for count in (1, 2, 3)],
        "edit_reduction",
    )
    return {
        "plot_id": "reference_count_1_3_edit_reduction",
        "kind": "line",
        "title": "Historical reference-count ablation",
        "x_label": "Reference count",
        "y_label": "Edit reduction",
        "x_ticks": [{"value": count, "label": str(count)} for count in (1, 2, 3)],
        "series": [{"label": "mean +/- seed SD", "values": values}],
        "source_files": sources,
        "claim_status": "HISTORICAL_TRANSDUCTIVE_ABLATION | READY_FROM_HISTORICAL_EVIDENCE",
    }


def plot_dual(summary_path: Path) -> Dict[str, Any]:
    data = load(summary_path)
    aggregates = data["analysis"]["aggregates"]
    pairs = sorted({row["pair_id"] for row in aggregates})
    variants = {
        "FULL_LINEAR_BASELINE": "Single-support FULL",
        "DUAL_SUPPORT_GEOMETRY_BLEND": "Dual-Support",
    }
    series = []
    for variant, label in variants.items():
        by_pair = {row["pair_id"]: row for row in aggregates if row["variant"] == variant}
        series.append({
            "label": label,
            "values": [
                {"x": index, "y": float(by_pair[pair]["means"]["garment_lpips"])}
                for index, pair in enumerate(pairs)
            ],
        })
    return {
        "plot_id": "dual_support_all_pair_garment_lpips",
        "kind": "line",
        "title": "Dual-Support all-pair garment LPIPS",
        "x_label": "Frozen garment pair",
        "y_label": "Garment LPIPS (lower is better)",
        "x_ticks": [{"value": index, "label": pair} for index, pair in enumerate(pairs)],
        "series": series,
        "source_files": [str(summary_path)],
        "claim_status": "SUPPLEMENTARY_EXTENSION_CANDIDATE | CLOSED_WARDROBE_ONLY",
    }


def plot_controller_trajectory(path: Path) -> Dict[str, Any]:
    data = load(path)["aggregate_by_step"]
    series = []
    for family, label in (("MATCHED_V1", "Matched V1"), ("V2", "V2")):
        series.append({
            "label": label,
            "values": [{"x": row["step"], "y": row["test_pair_accuracy"]} for row in data[family]],
        })
    return {
        "plot_id": "controller_v1_v2_pair_accuracy_trajectory",
        "kind": "line",
        "title": "Controller checkpoint pair-identification trajectory",
        "x_label": "Training step",
        "y_label": "Mixed-only test pair accuracy",
        "series": series,
        "source_files": [str(path)],
        "claim_status": "SUPPLEMENTARY_DIAGNOSTIC_ONLY",
    }


def plot_difficult_pairs(path: Path) -> Dict[str, Any]:
    data = load(path)
    grouped: Dict[Tuple[str, int, str], List[float]] = defaultdict(list)
    for row in data["evaluations"]:
        if row["model_family"] != "V2":
            continue
        step = int(row["step"])
        recalls = row["splits"]["test"]["per_pair_recall"]
        for pair in ("O01_O03", "O02_O03"):
            grouped[("V2", step, pair)].append(float(recalls[pair]))
    steps = sorted({key[1] for key in grouped})
    series = []
    for pair in ("O01_O03", "O02_O03"):
        series.append({
            "label": pair,
            "values": [
                {
                    "x": step,
                    "y": mean_error(grouped[("V2", step, pair)])[0],
                    "yerr": mean_error(grouped[("V2", step, pair)])[1],
                }
                for step in steps
            ],
        })
    return {
        "plot_id": "controller_difficult_pair_recall_trajectory",
        "kind": "line",
        "title": "Controller V2 difficult-pair recall",
        "x_label": "Training step",
        "y_label": "Test pair recall",
        "series": series,
        "source_files": [str(path)],
        "claim_status": "SUPPLEMENTARY_DIAGNOSTIC_ONLY | ALL_ROTATIONS_AND_SEEDS",
    }


def plot_controller_budget(path: Path) -> Dict[str, Any]:
    data = load(path)["selected_test_trajectories"]
    families = (("MATCHED_V1_CONTINUED", "Matched V1"), ("FULL_V2_CONTINUED", "Full V2"))
    series = []
    for family, label in families:
        points = data[family]
        series.append({
            "label": label,
            "values": [
                {"x": int(step), "y": float(record["mixed_pair_protocol_weighted_macro"])}
                for step, record in sorted(points.items(), key=lambda item: int(item[0]))
            ],
        })
    return {
        "plot_id": "controller_budget_continuation_trajectory",
        "kind": "line",
        "title": "Controller continuation-budget trajectory",
        "x_label": "Training step",
        "y_label": "Mixed-only pair accuracy",
        "series": series,
        "source_files": [str(path)],
        "claim_status": "SUPPLEMENTARY_DIAGNOSTIC_ONLY | POST_FAILURE_OPTIMIZATION_DIAGNOSTIC",
    }


def plot_subject00(step0: Path, step384: Path, medium: Path) -> Dict[str, Any]:
    inputs = [(0, step0), (384, step384), (20249, medium)]
    values = [
        {"x": step, "y": float(load(path)["metrics_all"]["lpips"]["mean"])}
        for step, path in inputs
    ]
    return {
        "plot_id": "subject00_step0_to_medium_lpips",
        "kind": "line",
        "title": "Subject00 base-avatar progression",
        "x_label": "Training step",
        "y_label": "Mean LPIPS (lower is better)",
        "series": [{"label": "All 96 fixed queries", "values": values}],
        "source_files": [str(path) for _, path in inputs],
        "claim_status": "SUPPLEMENTARY_PORTABILITY_CANDIDATE | BASE_AVATAR_ONLY",
    }


def main() -> int:
    args = parse_args()
    result = {
        "schema_version": "paper_metric_plot_source_data.v1",
        "plots": [
            plot_historical_ablation(args.historical_root),
            plot_rank(args.historical_root),
            plot_reference_count(args.historical_root),
            plot_dual(args.dual_summary),
            plot_controller_trajectory(args.controller_trajectory),
            plot_controller_budget(args.controller_budget_summary),
            plot_difficult_pairs(args.controller_trajectory),
            plot_subject00(args.subject_step0, args.subject_step384, args.subject_medium),
        ],
        "prohibited_pending_result_plots": ["PURE_ENDPOINT", "COEFFICIENT_HEADROOM", "LOO_ADAPTATION"],
    }
    if args.dry_run:
        print("\n".join(plot["plot_id"] for plot in result["plots"]))
        return 0
    stable_write(args.output, result)
    print(json.dumps({"normalized_plots": len(result["plots"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
