from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence


SEEN_OUTFITS = ("O01", "O02", "O03", "O04", "O08")
FIXED_VIEWS = ("cond_000000", "cond_000318", "cond_000017", "cond_000347")
EPISODE_METRICS = (
    "standardized_coefficient_rmse", "restored_coefficient_rmse",
    "nearest_teacher_accuracy", "correct_outfit_rank", "pairwise_coefficient_margin",
    "normalized_residual_rmse", "cosine_similarity", "top_10_support_overlap",
    "top_20_support_overlap", "garment_rgb_mae", "garment_alpha_mae",
    "edit_reduction", "target_closer_fraction", "protected_rgb_mae",
    "background_rgb_mae",
)
REFERENCE_METRICS = (
    "correct_vs_swapped_wins", "permutation_max_difference",
    "single_reference_accuracy", "two_reference_dropout_accuracy",
    "zero_replacement_sensitivity", "base_replacement_sensitivity",
)
EFFICIENCY_METRICS = (
    "trainable_parameter_count", "basis_storage_bytes", "peak_vram_bytes",
    "training_time_seconds", "inference_time_seconds", "render_time_seconds",
)
ALL_METRICS = EPISODE_METRICS + REFERENCE_METRICS + EFFICIENCY_METRICS


def _finite_vector(values: Sequence[float], name: str) -> list[float]:
    result = [float(value) for value in values]
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must be non-empty and finite")
    return result


def _rmse(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("RMSE inputs must have equal non-zero lengths")
    return math.sqrt(fmean((a - b) ** 2 for a, b in zip(left, right)))


def _l2(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    denominator = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))
    return numerator / denominator if denominator > 0 else 1.0 if left == right else 0.0


def _top_overlap(left: Sequence[float], right: Sequence[float], percent: int) -> float:
    count = max(1, math.ceil(len(left) * percent / 100.0))
    lset = set(sorted(range(len(left)), key=lambda index: abs(left[index]), reverse=True)[:count])
    rset = set(sorted(range(len(right)), key=lambda index: abs(right[index]), reverse=True)[:count])
    return len(lset & rset) / count


def _optional_mean(values: Iterable[float | None]) -> float | None:
    applicable = [float(value) for value in values if value is not None]
    return fmean(applicable) if applicable else None


def _episode_metrics(
    record: Mapping[str, Any], *, not_applicable: set[str] | None = None,
) -> dict[str, float | None]:
    predicted = _finite_vector(record["predicted_standardized_coefficients"], "predicted coefficients")
    target = _finite_vector(record["target_standardized_coefficients"], "target coefficients")
    if len(predicted) != len(target):
        raise ValueError("coefficient shape mismatch")
    mean = _finite_vector(record["coefficient_normalization"]["mean"], "coefficient mean")
    std = _finite_vector(record["coefficient_normalization"]["std"], "coefficient std")
    restored_predicted = [value * scale + center for value, scale, center in zip(predicted, std, mean)]
    restored_target = [value * scale + center for value, scale, center in zip(target, std, mean)]
    teachers = {
        outfit: _finite_vector(values, f"teacher {outfit}")
        for outfit, values in record["teacher_standardized_coefficients"].items()
    }
    if set(teachers) != set(SEEN_OUTFITS):
        raise ValueError("teacher coefficient bank must contain five seen outfits")
    distances = {outfit: _l2(predicted, values) for outfit, values in teachers.items()}
    ordered = sorted(distances, key=lambda outfit: (distances[outfit], outfit))
    outfit = record["outfit_id"]
    correct_distance = distances[outfit]
    nearest_other = min(value for key, value in distances.items() if key != outfit)
    pred_residual = _finite_vector(record["predicted_normalized_residual"], "predicted residual")
    target_residual = _finite_vector(record["target_normalized_residual"], "target residual")
    render = record["render_metrics"]
    result = {
        "standardized_coefficient_rmse": _rmse(predicted, target),
        "restored_coefficient_rmse": _rmse(restored_predicted, restored_target),
        "nearest_teacher_accuracy": float(ordered[0] == outfit),
        "correct_outfit_rank": float(ordered.index(outfit) + 1),
        "pairwise_coefficient_margin": nearest_other - correct_distance,
        "normalized_residual_rmse": _rmse(pred_residual, target_residual),
        "cosine_similarity": _cosine(pred_residual, target_residual),
        "top_10_support_overlap": _top_overlap(pred_residual, target_residual, 10),
        "top_20_support_overlap": _top_overlap(pred_residual, target_residual, 20),
    }
    for metric in (
        "garment_rgb_mae", "garment_alpha_mae", "edit_reduction",
        "target_closer_fraction", "protected_rgb_mae", "background_rgb_mae",
    ):
        value = float(render[metric])
        if not math.isfinite(value):
            raise ValueError(f"non-finite render metric: {metric}")
        result[metric] = value
    for metric in not_applicable or set():
        if metric in result:
            result[metric] = None
    return result


def load_raw_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL line {line_number}: {error}") from error
    return records


def evaluate_records(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_asset_fingerprint: str,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["record_type"], []).append(record)
    metadata = grouped.get("metadata", [])
    if len(metadata) != 1:
        raise ValueError("exactly one metadata record is required")
    meta = metadata[0]
    if meta["asset_fingerprint"] != expected_asset_fingerprint:
        raise ValueError("PAPER_ASSET_MISMATCH")
    if meta.get("target_forward_input_used") is not False:
        raise ValueError("target-forward boundary violation")
    not_applicable = set(meta.get("not_applicable_metrics", []))
    if not not_applicable.issubset(ALL_METRICS):
        raise ValueError("unknown not-applicable metric")

    episodes = grouped.get("episode", [])
    expected_keys = {(outfit, view) for outfit in SEEN_OUTFITS for view in FIXED_VIEWS}
    actual_keys = [(item["outfit_id"], item["view_id"]) for item in episodes]
    if len(episodes) != 20 or set(actual_keys) != expected_keys or len(set(actual_keys)) != 20:
        raise ValueError("evaluator requires exactly 20 unique seen episodes")
    if any(item.get("target_reference_overlap", 1) != 0 for item in episodes):
        raise ValueError("reference-target overlap must be zero")

    swaps = grouped.get("swap", [])
    if len(swaps) != 80:
        raise ValueError("evaluator requires exactly 80 cross-outfit swaps")
    if any(item["source_outfit"] == item["target_outfit"] for item in swaps):
        raise ValueError("swap records must be cross-outfit")
    permutation = grouped.get("permutation", [])
    single = grouped.get("single_reference", [])
    dropout = grouped.get("two_reference_dropout", [])
    replacement = grouped.get("replacement", [])
    if not permutation or not single or not dropout or not replacement:
        raise ValueError("reference robustness records are incomplete")

    per_episode = []
    for record in episodes:
        per_episode.append({
            "seed": int(meta["seed"]), "outfit_id": record["outfit_id"],
            "view_id": record["view_id"],
            **_episode_metrics(record, not_applicable=not_applicable),
        })
    reference_metrics = {
        "correct_vs_swapped_wins": float(sum(bool(item["correct_wins"]) for item in swaps)),
        "permutation_max_difference": max(float(item["max_difference"]) for item in permutation),
        "single_reference_accuracy": fmean(float(bool(item["correct"])) for item in single),
        "two_reference_dropout_accuracy": fmean(float(bool(item["correct"])) for item in dropout),
        "zero_replacement_sensitivity": fmean(float(item["zero_difference"]) for item in replacement),
        "base_replacement_sensitivity": fmean(float(item["base_difference"]) for item in replacement),
    }
    reference_metrics = {
        key: None if key in not_applicable else value
        for key, value in reference_metrics.items()
    }
    efficiency = {key: float(meta["efficiency"][key]) for key in EFFICIENCY_METRICS}
    efficiency = {
        key: None if key in not_applicable else value
        for key, value in efficiency.items()
    }
    metrics = {
        metric: _optional_mean(item[metric] for item in per_episode)
        for metric in EPISODE_METRICS
    }
    metrics.update(reference_metrics)
    metrics.update(efficiency)
    held_out = grouped.get("held_out", [])
    if any(item.get("outfit_id") != "O07" for item in held_out):
        raise ValueError("held-out records may contain only O07")
    correct_count = sum(
        item["nearest_teacher_accuracy"] == 1.0
        and item["correct_outfit_rank"] == 1.0
        for item in per_episode
    )
    return {
        "status": "PASS",
        "seed": int(meta["seed"]),
        "asset_fingerprint": meta["asset_fingerprint"],
        "metric_count": len(ALL_METRICS),
        "metrics": metrics,
        "per_episode_metrics": per_episode,
        "validation": {
            "seen_episode_count": len(episodes), "correct_seen_episode_count": correct_count,
            "swap_count": len(swaps), "fixed_outfit_order": list(SEEN_OUTFITS),
            "fixed_view_order": list(FIXED_VIEWS), "target_forward_boundary": True,
            "outfit_macro_required": True, "best_seed_selection": False,
            "not_applicable_metrics": sorted(not_applicable),
        },
        "held_out_diagnostic": {
            "status": "Held-out Diagnostic — FAIL", "records": [dict(item) for item in held_out],
            "included_in_seen_aggregation": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate raw seen-outfit paper episode outputs")
    parser.add_argument("--raw-metrics", type=Path, required=True)
    parser.add_argument("--asset-fingerprint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_records(load_raw_records(args.raw_metrics), expected_asset_fingerprint=args.asset_fingerprint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "metric_count": result["metric_count"]}))


if __name__ == "__main__":
    main()
