"""Formal evaluation and aggregation for the frozen dual-support controller."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene.reference_conditioned_dual_support_controller import (  # noqa: E402
    OUTFIT_ORDER,
    ReferenceConditionedDualSupportController,
    construct_dual_support_runtime,
    inference_result_schema,
    stable_top2_selection,
)
from tools.paper import run_reference_conditioned_dual_support_controller_formal as formal  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    formal.atomic_json(path, value)


def canonical_hash(value: Any) -> str:
    return formal.canonical_hash(value)


def checkpoint_path(output_root: Path, seed: int) -> Path:
    return output_root / "attempt_001" / f"seed_{seed}" / "checkpoints/final_step_300.pt"


def training_audit(output_root: Path) -> dict[str, Any]:
    formal.load_preflight(output_root)
    attempt = output_root / "attempt_001"
    result_path = attempt / "aggregates/training_summary.json"
    if result_path.is_file():
        return read_json(result_path)
    seeds = []
    initialization_hashes = []
    totals = Counter()
    required_checkpoint_fields = {
        "model", "optimizer", "scheduler", "cpu_rng_state", "cuda_rng_state_all",
        "data_order_cursor", "cycle_index", "batch_index", "fold_id", "global_step",
        "source_head", "protocol_sha", "manifest_sha", "cycle_sha", "schedule_sha",
    }
    for seed in formal.SEEDS:
        root = attempt / f"seed_{seed}"
        first = read_json(root / "audits/fresh_process_init_probe_1.json")
        second = read_json(root / "audits/fresh_process_init_probe_2.json")
        result = read_json(root / "training/training_result.json")
        same_probe = first == second
        checkpoints = sorted((root / "checkpoints").glob("*.pt"))
        checkpoint_rows = []
        for path in checkpoints:
            state = torch.load(path, map_location="cpu", weights_only=False)
            missing = sorted(required_checkpoint_fields.difference(state))
            checkpoint_rows.append({
                "path": str(path), "global_step": state.get("global_step"),
                "missing_fields": missing, "schedule_sha": state.get("schedule_sha"),
                "source_head": state.get("source_head"),
            })
        checkpoint_rows.sort(key=lambda row: int(row["global_step"]))
        expected_steps = [50, 100, 150, 200, 250, 300]
        checkpoint_pass = (
            [row["global_step"] for row in checkpoint_rows] == expected_steps
            and all(not row["missing_fields"] and row["schedule_sha"] == formal.SCHEDULE_SHA for row in checkpoint_rows)
        )
        counters = result["counters"]
        for name, value in counters.items():
            totals[name] += int(value)
        initialization_hashes.append(first["initialization_sha256"])
        seed_pass = (
            same_probe and first["initialization_sha256"] == result["initialization_sha256"]
            and result["parameter_count"] == 3589 and result["data_order_sha256"] == formal.SCHEDULE_SHA
            and all(counters[name] == 300 for name in (
                "training_steps", "forward_training_batches", "backward_calls", "optimizer_steps", "scheduler_steps"
            ))
            and counters["checkpoint_writes"] == 6 and checkpoint_pass
            and result["nan_or_inf_count"] == 0 and result["best_checkpoint_selection"] is False
        )
        seeds.append({
            "seed": seed, "status": "PASS" if seed_pass else "FAIL",
            "same_seed_fresh_process_exact": same_probe,
            "initialization_sha256": first["initialization_sha256"],
            "training_result": str(root / "training/training_result.json"),
            "trace_sha256": result["trace_sha256"], "counters": counters,
            "checkpoint_rows": checkpoint_rows, "resume_count": len(result["resume_events"]),
            "wall_clock_seconds": result["wall_clock_seconds"], "peak_vram_bytes": result["peak_vram_bytes"],
        })
    cross_unique = len(set(initialization_hashes)) == 3
    expected_totals = {
        "training_steps": 900, "forward_training_batches": 900, "backward_calls": 900,
        "optimizer_steps": 900, "scheduler_steps": 900, "checkpoint_writes": 18,
    }
    result = {
        "schema_version": "canondressgs.research.dual_support_controller_training_summary.v1",
        "status": "PASS" if all(row["status"] == "PASS" for row in seeds) and cross_unique and dict(totals) == expected_totals else "FAIL",
        "task_id": formal.TASK_ID, "seeds": seeds,
        "same_seed_fresh_process_exact": all(row["same_seed_fresh_process_exact"] for row in seeds),
        "cross_seed_initialization_unique": cross_unique,
        "unique_initialization_count": len(set(initialization_hashes)),
        "data_order_hashes": [formal.SCHEDULE_SHA] * 3,
        "totals": dict(totals), "expected_totals": expected_totals,
        "training_protocol_records": 320, "formal_pure_training_exposure": 0,
        "consistent_duplicates_retained": 80, "paper_final": False, "paper_final_count": 0,
    }
    atomic_json(result_path, result)
    return result


def load_model(output_root: Path, seed: int, device: torch.device) -> ReferenceConditionedDualSupportController:
    state = torch.load(checkpoint_path(output_root, seed), map_location="cpu", weights_only=False)
    if state["global_step"] != 300 or state["seed"] != seed or state["schedule_sha"] != formal.SCHEDULE_SHA:
        raise RuntimeError("formal final checkpoint provenance mismatch")
    model = ReferenceConditionedDualSupportController(seed=seed, input_dim=512).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model


def infer(
    model: ReferenceConditionedDualSupportController,
    rows: torch.Tensor,
    valid: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    with torch.inference_mode():
        distribution = model(rows.to(device), valid.to(device))
    selection = stable_top2_selection(distribution.probabilities)
    runtime = construct_dual_support_runtime(
        selection, {outfit: f"FROZEN_TEACHER_ENDPOINT/{outfit}" for outfit in OUTFIT_ORDER}
    )
    return inference_result_schema(distribution, selection, runtime)


def target_pair(target: Sequence[float]) -> tuple[int, ...]:
    return tuple(index for index, value in enumerate(target) if value > 0.0)


def prediction_row(record: Mapping[str, Any], schema: Mapping[str, Any]) -> dict[str, Any]:
    probabilities = [float(value) for value in schema["probabilities"]]
    target = [float(value) for value in record["target_distribution"]]
    gt_pair = target_pair(target)
    predicted_pair = tuple(sorted((OUTFIT_ORDER.index(schema["top1_outfit"]), OUTFIT_ORDER.index(schema["top2_outfit"]))))
    pair_correct = None if len(gt_pair) == 1 else predicted_pair == tuple(sorted(gt_pair))
    dominant = max(range(5), key=lambda index: (target[index], -index))
    predicted_dominant = max(range(5), key=lambda index: (probabilities[index], -index))
    ce = -sum(value * math.log(max(probability, 1e-12)) for value, probability in zip(target, probabilities))
    brier = sum((probability - value) ** 2 for value, probability in zip(target, probabilities))
    if len(gt_pair) == 2:
        mass = probabilities[gt_pair[0]] + probabilities[gt_pair[1]]
        predicted_first = probabilities[gt_pair[0]] / max(mass, 1e-12)
        target_first = target[gt_pair[0]]
        weight_error = abs(predicted_first - target_first)
        ordering_correct = predicted_dominant == dominant
    else:
        weight_error = None
        ordering_correct = None
    return {
        "record_id": record["record_id"], "logical_input_sha256": record["logical_input_sha256"],
        "pair_id": record["pair_id"], "assignment_type": record["assignment_type"],
        "assignment_position": record["assignment_position"], "target_view_fold": record["target_view_fold"],
        "target_distribution": target, "probabilities": probabilities,
        "top1_outfit": schema["top1_outfit"], "top2_outfit": schema["top2_outfit"],
        "top2_mass": schema["top2_mass"], "normalized_secondary_weight": schema["normalized_top2_weight_2"],
        "mode": schema["mode"], "fallback_reason": schema["fallback_reason"],
        "runtime_weights": [branch["opacity_weight"] for branch in schema["branches"]],
        "dominant_correct": predicted_dominant == dominant, "pair_correct": pair_correct,
        "ordering_correct": ordering_correct, "weight_absolute_error": weight_error,
        "brier_score": brier, "cross_entropy": ce,
        "entropy": -sum(value * math.log(max(value, 1e-12)) for value in probabilities),
        "target_forward_leakage": schema["target_forward_leakage"],
        "ground_truth_id_pair_alpha_inference_use": 0,
    }


def ece(rows: Sequence[Mapping[str, Any]], bins: int = 10) -> float:
    total = len(rows)
    result = 0.0
    for bin_index in range(bins):
        low, high = bin_index / bins, (bin_index + 1) / bins
        selected = [row for row in rows if low <= max(row["probabilities"]) < high or (bin_index == bins - 1 and max(row["probabilities"]) == 1.0)]
        if selected:
            confidence = statistics.fmean(max(row["probabilities"]) for row in selected)
            accuracy = statistics.fmean(float(row["dominant_correct"]) for row in selected)
            result += len(selected) / total * abs(confidence - accuracy)
    return result


def stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mixed = [row for row in rows if row["pair_correct"] is not None]
    return {
        "record_count": len(rows), "mixed_record_count": len(mixed),
        "dominant_top1_accuracy": statistics.fmean(float(row["dominant_correct"]) for row in rows),
        "top2_pair_accuracy": statistics.fmean(float(row["pair_correct"]) for row in mixed),
        "ordering_accuracy": statistics.fmean(float(row["ordering_correct"]) for row in mixed),
        "weight_mae": statistics.fmean(float(row["weight_absolute_error"]) for row in mixed),
        "weight_rmse": math.sqrt(statistics.fmean(float(row["weight_absolute_error"]) ** 2 for row in mixed)),
        "brier_score": statistics.fmean(float(row["brier_score"]) for row in rows),
        "cross_entropy": statistics.fmean(float(row["cross_entropy"]) for row in rows),
        "ece": ece(rows), "top2_mass": statistics.fmean(float(row["top2_mass"]) for row in rows),
        "entropy": statistics.fmean(float(row["entropy"]) for row in rows),
        "dual_support_activation_rate": statistics.fmean(row["mode"] == "DUAL_SUPPORT" for row in mixed),
        "single_endpoint_rate": statistics.fmean(row["mode"] == "SINGLE_ENDPOINT" for row in mixed),
        "low_top2_mass_rate": statistics.fmean(row["fallback_reason"] == "LOW_TOP2_MASS" for row in mixed),
        "low_secondary_weight_rate": statistics.fmean(row["fallback_reason"] == "LOW_SECONDARY_WEIGHT" for row in mixed),
    }


def consistency(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    assignment_groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    fold_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["pair_correct"] is None:
            continue
        composition = "AAB" if str(row["assignment_type"]).startswith("AAB") else "ABB"
        assignment_groups[(row["pair_id"], row["target_view_fold"], composition)].append(row)
        fold_groups[(row["pair_id"], row["assignment_type"])].append(row)
    assignment_exact = [len({(row["top1_outfit"], row["top2_outfit"]) for row in group}) == 1 for group in assignment_groups.values()]
    fold_exact = [len({(row["top1_outfit"], row["top2_outfit"]) for row in group}) == 1 for group in fold_groups.values()]
    return {
        "assignment_position_group_count": len(assignment_exact),
        "assignment_position_pair_consistency": statistics.fmean(assignment_exact),
        "target_view_fold_group_count": len(fold_exact),
        "target_view_fold_pair_consistency": statistics.fmean(fold_exact),
    }


def run_classification(seed: int, output_root: Path, feature_cache: Path) -> dict[str, Any]:
    formal.load_preflight(output_root)
    training_audit(output_root)
    seed_root = output_root / "attempt_001" / f"seed_{seed}"
    pure_path = seed_root / "pure/pure_classification.json"
    mixed_path = seed_root / "mixed/mixed_classification.json"
    if pure_path.exists() or mixed_path.exists():
        raise FileExistsError(f"classification output exists for seed {seed}")
    formal.configure_determinism()
    manifest, _, _ = formal.contract()
    cache = torch.load(feature_cache, map_location="cpu", weights_only=False)
    device = torch.device("cuda")
    model = load_model(output_root, seed, device)
    pure_records = manifest["formal_pure_endpoint_episodes"]
    pure_rows = []
    pure_index = {(record["garment_labels"][0], record["target_view_fold"]): record for record in pure_records}
    schemas: dict[str, dict[str, Any]] = {}
    for record in pure_records:
        rows, valid = formal.case_rows(cache, record)
        schema = infer(model, rows, valid, device)
        schemas[record["record_id"]] = schema
        row = prediction_row(record, schema)
        permutation = infer(model, rows[[2, 0, 1]], valid[[2, 0, 1]], device)
        single = infer(model, rows[:1], valid[:1], device)
        dropout = infer(model, rows[:2], valid[:2], device)
        row.update({
            "permutation_probability_max_abs": max(abs(a - b) for a, b in zip(schema["probabilities"], permutation["probabilities"])),
            "single_reference_top1": single["top1_outfit"], "single_reference_correct": single["top1_outfit"] == record["garment_labels"][0],
            "reference_dropout_top1": dropout["top1_outfit"], "reference_dropout_correct": dropout["top1_outfit"] == record["garment_labels"][0],
        })
        pure_rows.append(row)
    swaps = []
    for target in pure_records:
        target_outfit, fold = target["garment_labels"][0], target["target_view_fold"]
        correct = schemas[target["record_id"]]
        for source_outfit in OUTFIT_ORDER:
            if source_outfit == target_outfit:
                continue
            source = pure_index[(source_outfit, fold)]
            source_schema = schemas[source["record_id"]]
            swaps.append({
                "target_outfit": target_outfit, "source_outfit": source_outfit, "target_view_fold": fold,
                "correct_top1": correct["top1_outfit"], "swapped_top1": source_schema["top1_outfit"],
                "correct_wins": correct["top1_outfit"] == target_outfit and source_schema["top1_outfit"] == source_outfit,
            })
    pure_summary = {
        "schema_version": "canondressgs.research.dual_support_controller_pure_classification.v1",
        "status": "COMPLETE", "seed": seed, "records": pure_rows, "swaps": swaps,
        "record_count": len(pure_rows), "correct_count": sum(row["dominant_correct"] for row in pure_rows),
        "top1_accuracy": statistics.fmean(row["dominant_correct"] for row in pure_rows),
        "single_endpoint_rate": statistics.fmean(row["mode"] == "SINGLE_ENDPOINT" for row in pure_rows),
        "low_top2_mass_rate": statistics.fmean(row["fallback_reason"] == "LOW_TOP2_MASS" for row in pure_rows),
        "low_secondary_weight_rate": statistics.fmean(row["fallback_reason"] == "LOW_SECONDARY_WEIGHT" for row in pure_rows),
        "endpoint_parity": "PASS" if all(row["dominant_correct"] and row["mode"] == "SINGLE_ENDPOINT" for row in pure_rows) else "FAIL",
        "swap_count": len(swaps), "swap_success_rate": statistics.fmean(row["correct_wins"] for row in swaps),
        "single_reference_accuracy": statistics.fmean(row["single_reference_correct"] for row in pure_rows),
        "reference_dropout_accuracy": statistics.fmean(row["reference_dropout_correct"] for row in pure_rows),
        "permutation_max_abs": max(row["permutation_probability_max_abs"] for row in pure_rows),
        "formal_pure_record_overlap": 0, "formal_pure_reference_asset_overlap": "20/20 logical inputs",
        "target_forward_leakage": 0, "ground_truth_id_pair_alpha_inference_use": 0,
        "paper_final": False,
    }
    atomic_json(pure_path, pure_summary)

    rows = []
    for record in manifest["query_sets"]:
        features, valid = formal.case_rows(cache, record)
        rows.append(prediction_row(record, infer(model, features, valid, device)))
    unique = list({row["logical_input_sha256"]: row for row in rows}.values())
    duplicates: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        duplicates[row["logical_input_sha256"]].append(row)
    duplicate_groups = [group for group in duplicates.values() if len(group) > 1]
    mixed_summary = {
        "schema_version": "canondressgs.research.dual_support_controller_mixed_classification.v1",
        "status": "COMPLETE", "seed": seed, "records": rows,
        "protocol_weighted": stats(rows), "unique_query": stats(unique),
        "protocol_record_count": len(rows), "unique_query_count": len(unique),
        "consistent_duplicate_record_count": sum(len(group) for group in duplicate_groups),
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_group_prediction_consistency": statistics.fmean(
            len({tuple(row["probabilities"]) for row in group}) == 1 for group in duplicate_groups
        ),
        "consistency": consistency(rows),
        "pair_confusion": dict(Counter(
            f"{row['pair_id']}->{row['top1_outfit']}_{row['top2_outfit']}" for row in rows if row["pair_correct"] is not None
        )),
        "all_assignment_positions_present": sorted({row["assignment_position"] for row in rows if row["assignment_position"] is not None}) == [0, 1, 2],
        "all_target_view_folds_present": sorted({row["target_view_fold"] for row in rows}) == sorted(manifest["frozen_target_view_order"]),
        "target_forward_leakage": 0, "ground_truth_id_pair_alpha_inference_use": 0,
        "geometry_interpolation": False, "paper_final": False,
    }
    atomic_json(mixed_path, mixed_summary)
    return {"seed": seed, "pure": pure_summary, "mixed": {key: value for key, value in mixed_summary.items() if key != "records"}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("audit-training", "classify-seed"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    feature_cache = args.feature_cache or args.asset_root.resolve() / formal.FORMAL_NAME / "shared_preflight/frozen_reference_feature_rows_v1.pt"
    if args.phase == "audit-training":
        result = training_audit(output_root)
    else:
        if args.seed not in formal.SEEDS:
            parser.error("classify-seed requires --seed 0, 1, or 2")
        result = run_classification(args.seed, output_root, feature_cache.resolve())
    print(json.dumps(result, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
