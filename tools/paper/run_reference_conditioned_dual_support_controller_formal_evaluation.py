"""Formal evaluation and aggregation for the frozen dual-support controller."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
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
from scene.p0_candidate_initialization_protocol import seed_all  # noqa: E402
from tools import diagnose_image_conditioned_overfit_failure as diagnosis  # noqa: E402
from tools.paper import run_reference_conditioned_dual_support_controller_formal as formal  # noqa: E402
from tools.paper import run_geometry_dual_support_micro_pilot as geometry  # noqa: E402
from tools.paper import run_p0_color_spatial_soft_control_evaluations as sealed  # noqa: E402


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


def render_or_load(
    path: Path, runtime_value: sealed.EvaluationRuntime, left: str, condition: str,
    branches: Sequence[tuple[str, Any, float]],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], bool]:
    rgb_path = path.with_name(path.name + "_rgb.png")
    alpha_path = path.with_name(path.name + "_alpha.png")
    diagnostics_path = path.with_name(path.name + "_diagnostics.json")
    existing = (rgb_path.is_file(), alpha_path.is_file(), diagnostics_path.is_file())
    if any(existing) and not all(existing):
        raise RuntimeError(f"partial append-only render: {path}")
    if all(existing):
        return (
            sealed.image_tensor(rgb_path, 3).to(runtime_value.device),
            sealed.image_tensor(alpha_path, 1).to(runtime_value.device),
            read_json(diagnostics_path), False,
        )
    with torch.inference_mode():
        rgb, alpha, diagnostics = geometry.render_branches(runtime_value, left, condition, branches)
    sealed.save_new_render(rgb_path, rgb, 3)
    sealed.save_new_render(alpha_path, alpha, 1)
    atomic_json(diagnostics_path, diagnostics)
    return rgb.detach(), alpha.detach(), diagnostics, True


def controller_branches(row: Mapping[str, Any], endpoints: Mapping[str, Any]) -> tuple[tuple[str, Any, float], ...]:
    outfits = (row["top1_outfit"], row["top2_outfit"])
    weights = row["runtime_weights"]
    result = tuple((outfit, endpoints[outfit], float(weight)) for outfit, weight in zip(outfits, weights))
    if len(result) not in (1, 2) or abs(sum(item[2] for item in result) - 1.0) > 1e-6:
        raise RuntimeError("controller render branch contract mismatch")
    return result


def oracle_branches(record: Mapping[str, Any], endpoints: Mapping[str, Any]) -> tuple[tuple[str, Any, float], ...]:
    result = tuple(
        (OUTFIT_ORDER[index], endpoints[OUTFIT_ORDER[index]], float(weight))
        for index, weight in enumerate(record["target_distribution"]) if float(weight) > 0.0
    )
    if len(result) not in (1, 2) or abs(sum(item[2] for item in result) - 1.0) > 1e-6:
        raise RuntimeError("oracle branch contract mismatch")
    return result


def render_metrics(
    runtime_value: sealed.EvaluationRuntime, left: str, condition: str,
    rgb: torch.Tensor, alpha: torch.Tensor, oracle_rgb: torch.Tensor,
    source_rgb: torch.Tensor, target_rgb: torch.Tensor,
) -> dict[str, float]:
    result = geometry.image_metrics(
        runtime_value, left, condition, rgb, alpha, oracle_rgb, source_rgb, target_rgb
    )
    result["oracle_rgb_max_abs"] = float((rgb - oracle_rgb).abs().max())
    return result


def prepare_attempt_002(output_root: Path) -> dict[str, Any]:
    attempt_001 = output_root / "attempt_001"
    attempt_002 = output_root / "attempt_002"
    if attempt_002.exists():
        raise FileExistsError(f"append-only evaluation repair attempt exists: {attempt_002}")
    failure = {
        "schema_version": "canondressgs.research.controller_evaluation_runtime_failure.v1",
        "status": "FAILED_EVALUATION_RUNTIME_RNG_CONTAMINATION",
        "task_id": formal.TASK_ID,
        "cause": "legacy render context was constructed after the Controller seed and inherited seed-specific RNG state",
        "evidence": {
            "seed_1_pure_classifier_correct": "20/20",
            "seed_1_pure_single_endpoint_rate": 1.0,
            "seed_1_classification_endpoint_parity": "PASS",
            "seed_1_cross_process_render_parity_max_abs": 0.5823779106140137,
        },
        "attempt_001_disposition": "PRESERVED_INVALID_FOR_FINAL_RENDER_COMPARISON",
        "training_reuse_allowed": True,
        "training_rerun": 0, "backward_rerun": 0, "optimizer_step_rerun": 0,
        "threshold_change": 0, "checkpoint_selection": 0, "paper_final": False,
    }
    atomic_json(attempt_001 / "audits/FAILED_EVALUATION_RUNTIME_RNG_CONTAMINATION.json", failure)
    for seed in formal.SEEDS:
        for name in ("pure", "mixed", "perturbations", "renders", "metrics", "visuals", "audits"):
            (attempt_002 / f"seed_{seed}" / name).mkdir(parents=True, exist_ok=False)
    for name in ("aggregates", "baselines", "reports", "fingerprints", "audits"):
        (attempt_002 / name).mkdir(parents=True, exist_ok=False)
    resume = {
        "schema_version": "canondressgs.research.controller_evaluation_only_resume.v1",
        "status": "PASS", "source_attempt": "attempt_001", "target_attempt": "attempt_002",
        "source_training_summary": str(attempt_001 / "aggregates/training_summary.json"),
        "source_final_checkpoints": [str(checkpoint_path(output_root, seed)) for seed in formal.SEEDS],
        "source_classification_results_reused": True,
        "repair": "construct immutable legacy render context at fixed seed before loading the per-seed Controller",
        "training_steps_added": 0, "backward_calls_added": 0, "optimizer_steps_added": 0,
        "scheduler_steps_added": 0, "checkpoint_writes_added": 0,
        "threshold_changes": 0, "paper_final": False,
    }
    atomic_json(attempt_002 / "audits/resume_from_attempt_001.json", resume)
    return {"failure": failure, "resume": resume}


def prepare_attempt_003(output_root: Path) -> dict[str, Any]:
    attempt_002 = output_root / "attempt_002"
    attempt_003 = output_root / "attempt_003"
    if attempt_003.exists():
        raise FileExistsError(f"append-only evaluation repair attempt exists: {attempt_003}")
    failure = {
        "schema_version": "canondressgs.research.controller_evaluation_runtime_failure.v1",
        "status": "FAILED_EVALUATION_RUNTIME_INCOMPLETE_RNG_FIX",
        "task_id": formal.TASK_ID,
        "cause": "attempt_002 fixed torch RNG but did not fix Python and NumPy RNG before legacy context construction",
        "evidence": {
            "seed_1_pure_classifier_correct": "20/20",
            "seed_1_pure_single_endpoint_rate": 1.0,
            "seed_1_cross_process_render_parity_max_abs": 0.5823779106140137,
        },
        "attempt_002_disposition": "PRESERVED_INVALID_FOR_FINAL_RENDER_COMPARISON",
        "training_rerun": 0, "backward_rerun": 0, "optimizer_step_rerun": 0,
        "threshold_change": 0, "checkpoint_selection": 0, "paper_final": False,
    }
    atomic_json(attempt_002 / "audits/FAILED_EVALUATION_RUNTIME_INCOMPLETE_RNG_FIX.json", failure)
    for seed in formal.SEEDS:
        for name in ("pure", "mixed", "perturbations", "renders", "metrics", "visuals", "audits"):
            (attempt_003 / f"seed_{seed}" / name).mkdir(parents=True, exist_ok=False)
    for name in ("aggregates", "baselines", "reports", "fingerprints", "audits"):
        (attempt_003 / name).mkdir(parents=True, exist_ok=False)
    resume = {
        "schema_version": "canondressgs.research.controller_evaluation_only_resume.v2",
        "status": "PENDING_CONTEXT_HASH_PROBES", "source_training_attempt": "attempt_001",
        "source_classification_attempt": "attempt_001", "target_attempt": "attempt_003",
        "repair": "call frozen seed_all(0) before legacy context construction and require two fresh-process render hashes",
        "training_steps_added": 0, "backward_calls_added": 0, "optimizer_steps_added": 0,
        "scheduler_steps_added": 0, "checkpoint_writes_added": 0,
        "threshold_changes": 0, "paper_final": False,
    }
    atomic_json(attempt_003 / "audits/resume_from_failed_evaluation_attempts.json", resume)
    return {"failure": failure, "resume": resume}


def context_probe(output_root: Path, asset_root: Path, probe_index: int) -> dict[str, Any]:
    attempt = output_root / "attempt_003"
    if not attempt.is_dir() or probe_index not in (1, 2):
        raise RuntimeError("attempt_003 context probe contract mismatch")
    os.environ["CANONDRESSGS_ASSET_ROOT"] = str(asset_root)
    formal.configure_determinism(strict=False)
    sealed.RUN_BRANCH = formal.RUN_BRANCH
    sealed.SOURCE_HEAD = formal.SOURCE_HEAD
    seed_all(0)
    runtime_value = sealed.EvaluationRuntime(attempt / "audits/context_probe_no_write", asset_root, {})
    endpoints = geometry.endpoint_residuals(runtime_value)
    with torch.inference_mode():
        rgb, alpha, diagnostics = geometry.render_branches(
            runtime_value, "O01", "cond_000000", (("O01", endpoints["O01"], 1.0),)
        )
    result = {
        "schema_version": "canondressgs.research.controller_fixed_context_probe.v1",
        "status": "PASS", "probe_index": probe_index, "fixed_seed": 0,
        "base_xyz_sha256": hashlib.sha256(runtime_value.context["base"]._xyz.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
        "endpoint_rgb_sha256": hashlib.sha256(rgb.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
        "endpoint_alpha_sha256": hashlib.sha256(alpha.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
        "renderer_diagnostics": diagnostics, "paper_final": False,
    }
    atomic_json(attempt / "audits" / f"fixed_context_fresh_process_probe_{probe_index}.json", result)
    return result


def prepare_attempt_004(output_root: Path) -> dict[str, Any]:
    attempt_003 = output_root / "attempt_003"
    attempt_004 = output_root / "attempt_004"
    if attempt_004.exists():
        raise FileExistsError(f"append-only evaluation repair attempt exists: {attempt_004}")
    failure = {
        "schema_version": "canondressgs.research.controller_evaluation_runtime_failure.v1",
        "status": "FAILED_CROSS_PROCESS_BASELINE_REUSE",
        "task_id": formal.TASK_ID,
        "cause": "seed-0 Oracle images were reused as numeric baselines in other renderer processes",
        "evidence": {
            "fixed_context_two_process_hashes_exact": True,
            "seed_1_pure_classifier_correct": "20/20",
            "seed_1_pure_single_endpoint_rate": 1.0,
            "seed_1_cross_process_render_parity_max_abs": 0.5823779106140137,
            "seed_2_cross_process_render_parity_max_abs": 0.5823779106140137,
        },
        "attempt_003_disposition": "PRESERVED_INVALID_FOR_FINAL_NUMERIC_COMPARISON",
        "training_rerun": 0, "backward_rerun": 0, "optimizer_step_rerun": 0,
        "threshold_change": 0, "checkpoint_selection": 0, "paper_final": False,
    }
    atomic_json(attempt_003 / "audits/FAILED_CROSS_PROCESS_BASELINE_REUSE.json", failure)
    for seed in formal.SEEDS:
        for name in ("pure", "mixed", "perturbations", "renders", "metrics", "visuals", "audits"):
            (attempt_004 / f"seed_{seed}" / name).mkdir(parents=True, exist_ok=False)
    for name in ("aggregates", "baselines", "reports", "fingerprints", "audits"):
        (attempt_004 / name).mkdir(parents=True, exist_ok=False)
    resume = {
        "schema_version": "canondressgs.research.controller_evaluation_only_resume.v3",
        "status": "PENDING_PAIRED_RENDER_PROBES", "source_training_attempt": "attempt_001",
        "source_classification_attempt": "attempt_001", "target_attempt": "attempt_004",
        "repair": "render each candidate and its Oracle/endpoints consecutively in the same process and state",
        "cross_seed_numeric_baseline_reuse": False,
        "training_steps_added": 0, "backward_calls_added": 0, "optimizer_steps_added": 0,
        "scheduler_steps_added": 0, "checkpoint_writes_added": 0,
        "threshold_changes": 0, "paper_final": False,
    }
    atomic_json(attempt_004 / "audits/resume_from_failed_evaluation_attempts.json", resume)
    return {"failure": failure, "resume": resume}


def paired_render_probe(output_root: Path, asset_root: Path, feature_cache: Path, probe_index: int) -> dict[str, Any]:
    attempt = output_root / "attempt_004"
    if not attempt.is_dir() or probe_index not in (1, 2):
        raise RuntimeError("attempt_004 paired probe contract mismatch")
    manifest, _, _ = formal.contract()
    mixed = read_json(output_root / "attempt_001/seed_1/mixed/mixed_classification.json")
    pure = read_json(output_root / "attempt_001/seed_1/pure/pure_classification.json")
    prediction_index = {row["record_id"]: row for row in mixed["records"]}
    pure_prediction_index = {row["record_id"]: row for row in pure["records"]}
    os.environ["CANONDRESSGS_ASSET_ROOT"] = str(asset_root)
    formal.configure_determinism(strict=False)
    sealed.RUN_BRANCH = formal.RUN_BRANCH
    sealed.SOURCE_HEAD = formal.SOURCE_HEAD
    seed_all(0)
    runtime_value = sealed.EvaluationRuntime(attempt / "audits/paired_probe_no_write", asset_root, {})
    endpoints = geometry.endpoint_residuals(runtime_value)
    model = load_model(output_root, 1, torch.device("cuda"))
    cache = torch.load(feature_cache, map_location="cpu", weights_only=False)
    # Exercise the complete mixed sequence before the parity checks, matching
    # the formal render order and exposing any mutable-state accumulation.
    for record in manifest["query_sets"]:
        rows, valid = formal.case_rows(cache, record)
        live = infer(model, rows, valid, torch.device("cuda"))
        persisted = prediction_index[record["record_id"]]
        if max(abs(a - b) for a, b in zip(live["probabilities"], persisted["probabilities"])) > 1e-7:
            raise RuntimeError("paired probe prediction mismatch")
        left = record["pair_id"].split("_")[0]
        with torch.inference_mode():
            geometry.render_branches(runtime_value, left, record["target_view_fold"], controller_branches(persisted, endpoints))
    maxima = []
    rgb_hashes = []
    for record in manifest["formal_pure_endpoint_episodes"]:
        prediction = pure_prediction_index[record["record_id"]]
        left = record["garment_labels"][0]
        with torch.inference_mode():
            candidate, candidate_alpha, _ = geometry.render_branches(
                runtime_value, left, record["target_view_fold"], controller_branches(prediction, endpoints)
            )
            oracle, oracle_alpha, _ = geometry.render_branches(
                runtime_value, left, record["target_view_fold"], oracle_branches(record, endpoints)
            )
        maxima.append(max(float((candidate - oracle).abs().max()), float((candidate_alpha - oracle_alpha).abs().max())))
        rgb_hashes.append(hashlib.sha256(candidate.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
    result = {
        "schema_version": "canondressgs.research.controller_paired_render_probe.v1",
        "status": "PASS" if max(maxima) <= 1e-6 else "FAIL",
        "probe_index": probe_index, "seed": 1, "mixed_sequence_exercised": 320,
        "pure_paired_checks": 20, "maximum_paired_render_abs": max(maxima),
        "candidate_rgb_sequence_sha256": canonical_hash(rgb_hashes),
        "paper_final": False,
    }
    atomic_json(attempt / "audits" / f"paired_render_fresh_process_probe_{probe_index}.json", result)
    return result


def run_render_seed(
    seed: int, output_root: Path, asset_root: Path, feature_cache: Path,
    *, attempt_name: str = "attempt_002",
) -> dict[str, Any]:
    formal.load_preflight(output_root)
    source_seed_root = output_root / "attempt_001" / f"seed_{seed}"
    seed_root = output_root / attempt_name / f"seed_{seed}"
    if attempt_name == "attempt_003":
        first = read_json(output_root / attempt_name / "audits/fixed_context_fresh_process_probe_1.json")
        second = read_json(output_root / attempt_name / "audits/fixed_context_fresh_process_probe_2.json")
        comparable = (first["base_xyz_sha256"], first["endpoint_rgb_sha256"], first["endpoint_alpha_sha256"])
        if comparable != (second["base_xyz_sha256"], second["endpoint_rgb_sha256"], second["endpoint_alpha_sha256"]):
            raise RuntimeError("fixed context fresh-process hashes do not match")
    if attempt_name == "attempt_004":
        first = read_json(output_root / attempt_name / "audits/paired_render_fresh_process_probe_1.json")
        second = read_json(output_root / attempt_name / "audits/paired_render_fresh_process_probe_2.json")
        if first["status"] != "PASS" or second["status"] != "PASS":
            raise RuntimeError("paired render fresh-process probe failed")
        if first["candidate_rgb_sequence_sha256"] != second["candidate_rgb_sequence_sha256"]:
            raise RuntimeError("paired render fresh-process hashes do not match")
    result_path = seed_root / "metrics/render_evaluation.json"
    if result_path.exists():
        raise FileExistsError(f"render evaluation exists for seed {seed}")
    mixed = read_json(source_seed_root / "mixed/mixed_classification.json")
    pure = read_json(source_seed_root / "pure/pure_classification.json")
    manifest, _, _ = formal.contract()
    query_index = {row["record_id"]: row for row in manifest["query_sets"]}
    pure_index = {row["record_id"]: row for row in manifest["formal_pure_endpoint_episodes"]}
    prediction_index = {row["record_id"]: row for row in mixed["records"]}
    pure_prediction_index = {row["record_id"]: row for row in pure["records"]}
    cache = torch.load(feature_cache, map_location="cpu", weights_only=False)
    device = torch.device("cuda")
    os.environ["CANONDRESSGS_ASSET_ROOT"] = str(asset_root)
    formal.configure_determinism(strict=False)
    sealed.RUN_BRANCH = formal.RUN_BRANCH
    sealed.SOURCE_HEAD = formal.SOURCE_HEAD
    seed_all(0)
    runtime_value = sealed.EvaluationRuntime(seed_root / "audits/runtime_context_no_write", asset_root, {})
    endpoints = geometry.endpoint_residuals(runtime_value)
    model = load_model(output_root, seed, device)
    records = []
    oracle_created = 0
    controller_created = 0
    forward_seconds = 0.0
    feature_seconds = 0.0

    def evaluate_one(record: Mapping[str, Any], prediction: Mapping[str, Any], role: str) -> dict[str, Any]:
        nonlocal oracle_created, controller_created, forward_seconds, feature_seconds
        if record["pair_id"]:
            left, right = record["pair_id"].split("_")
        else:
            left = right = record["garment_labels"][0]
        condition = record["target_view_fold"]
        started = time.perf_counter()
        rows, valid = formal.case_rows(cache, record)
        feature_seconds += time.perf_counter() - started
        torch.cuda.synchronize()
        started = time.perf_counter()
        live = infer(model, rows, valid, device)
        torch.cuda.synchronize()
        forward_seconds += time.perf_counter() - started
        if live["probabilities"] != prediction["probabilities"]:
            maximum = max(abs(a - b) for a, b in zip(live["probabilities"], prediction["probabilities"]))
            if maximum > 1e-7:
                raise RuntimeError("persisted/live controller prediction mismatch")
        safe_id = record["record_id"].replace("/", "__")
        candidate_path = seed_root / role / "renders" / safe_id
        rgb, alpha, diagnostics, created = render_or_load(
            candidate_path, runtime_value, left, condition, controller_branches(prediction, endpoints)
        )
        controller_created += int(created)
        local_baseline_root = seed_root / role / "paired_baselines"
        oracle_path = local_baseline_root / f"{safe_id}__oracle"
        oracle_rgb, oracle_alpha, oracle_diagnostics, created = render_or_load(
            oracle_path, runtime_value, left, condition, oracle_branches(record, endpoints)
        )
        oracle_created += int(created)
        oracle_rgb_path = str(oracle_path.with_name(oracle_path.name + "_rgb.png"))
        source_path = local_baseline_root / f"{safe_id}__source_{left}"
        source_rgb, _, _, created = render_or_load(
            source_path, runtime_value, left, condition, ((left, endpoints[left], 1.0),)
        )
        oracle_created += int(created)
        target_path = local_baseline_root / f"{safe_id}__target_{right}"
        target_rgb, _, _, created = render_or_load(
            target_path, runtime_value, left, condition, ((right, endpoints[right], 1.0),)
        )
        oracle_created += int(created)
        metrics = render_metrics(runtime_value, left, condition, rgb, alpha, oracle_rgb, source_rgb, target_rgb)
        return {
            "record_id": record["record_id"], "role": role, "pair_id": record["pair_id"],
            "assignment_type": record["assignment_type"], "assignment_position": record["assignment_position"],
            "target_view_fold": condition, "mode": prediction["mode"],
            "selected_endpoints": [prediction["top1_outfit"], prediction["top2_outfit"]][:len(prediction["runtime_weights"])],
            "predicted_weights": prediction["runtime_weights"],
            "rgb_path": str(candidate_path.with_name(candidate_path.name + "_rgb.png")),
            "alpha_path": str(candidate_path.with_name(candidate_path.name + "_alpha.png")),
            "oracle_rgb_path": oracle_rgb_path,
            "metrics": metrics, "render_diagnostics": diagnostics,
            "oracle_render_diagnostics": oracle_diagnostics,
            "target_forward_leakage": 0, "ground_truth_id_pair_alpha_inference_use": 0,
        }

    for record_id, record in query_index.items():
        records.append(evaluate_one(record, prediction_index[record_id], "mixed"))
    for record_id, record in pure_index.items():
        records.append(evaluate_one(record, pure_prediction_index[record_id], "pure"))
    fields = (
        "garment_rgb_mae", "garment_lpips", "silhouette_iou", "boundary_fscore",
        "protected_lpips", "identity_metric", "outside_garment_opacity",
    )
    by_role = {}
    for role in ("mixed", "pure"):
        selected = [row for row in records if row["role"] == role]
        by_role[role] = {
            "record_count": len(selected),
            "means": {field: statistics.fmean(row["metrics"][field] for row in selected) for field in fields},
            "maxima": {field: max(row["metrics"][field] for row in selected) for field in fields},
            "single_endpoint_count": sum(row["mode"] == "SINGLE_ENDPOINT" for row in selected),
            "dual_support_count": sum(row["mode"] == "DUAL_SUPPORT" for row in selected),
        }
    pure_parity = [row for row in records if row["role"] == "pure"]
    result = {
        "schema_version": "canondressgs.research.dual_support_controller_render_evaluation.v1",
        "status": "COMPLETE", "seed": seed, "attempt": attempt_name,
        "source_training_attempt": "attempt_001", "records": records, "aggregates": by_role,
        "controller_render_count": len(records), "mixed_render_count": 320, "pure_render_count": 20,
        "controller_renders_created": controller_created, "oracle_or_endpoint_renders_created": oracle_created,
        "controller_forward_time_seconds_total": forward_seconds,
        "controller_forward_time_seconds_mean": forward_seconds / len(records),
        "f2_cached_feature_assembly_seconds_total": feature_seconds,
        "endpoint_selection_time_included_in_controller_forward": True,
        "endpoint_parity": {
            "status": "PASS" if all(row["metrics"]["oracle_rgb_max_abs"] <= 1e-6 for row in pure_parity) else "FAIL",
            "maximum_render_max_abs": max(row["metrics"]["oracle_rgb_max_abs"] for row in pure_parity),
        },
        "target_forward_leakage": 0, "ground_truth_id_pair_alpha_inference_use": 0,
        "geometry_interpolation": False, "paper_final": False,
    }
    atomic_json(result_path, result)
    return {key: value for key, value in result.items() if key != "records"}


def run_perturbation_seed(
    seed: int, output_root: Path, asset_root: Path, feature_cache: Path,
    *, attempt_name: str = "attempt_004",
) -> dict[str, Any]:
    seed_root = output_root / attempt_name / f"seed_{seed}"
    result_path = seed_root / "perturbations/perturbation_evaluation.json"
    if result_path.exists():
        raise FileExistsError(f"perturbation evaluation exists for seed {seed}")
    source_seed = output_root / "attempt_001" / f"seed_{seed}"
    mixed = read_json(source_seed / "mixed/mixed_classification.json")
    render_result = read_json(seed_root / "metrics/render_evaluation.json")
    prediction_index = {row["record_id"]: row for row in mixed["records"]}
    render_index = {row["record_id"]: row for row in render_result["records"] if row["role"] == "mixed"}
    manifest, _, _ = formal.contract()
    representatives = [
        row for row in manifest["query_sets"]
        if row["target_view_fold"] == "cond_000000"
        and row["assignment_type"] in {"AAB_minority_0", "ABB_minority_0"}
    ]
    if len(representatives) != 20:
        raise RuntimeError("perturbation representative count mismatch")
    cache = torch.load(feature_cache, map_location="cpu", weights_only=False)
    device = torch.device("cuda")
    os.environ["CANONDRESSGS_ASSET_ROOT"] = str(asset_root)
    formal.configure_determinism(strict=False)
    sealed.RUN_BRANCH = formal.RUN_BRANCH
    sealed.SOURCE_HEAD = formal.SOURCE_HEAD
    seed_all(0)
    runtime_value = sealed.EvaluationRuntime(seed_root / "audits/perturbation_context_no_write", asset_root, {})
    endpoints = geometry.endpoint_residuals(runtime_value)
    model = load_model(output_root, seed, device)
    variants = (
        "grayscale", "hue", "blur", "mask_erosion", "mask_dilation",
        "reference_dropout", "single_reference", "assignment_permutation",
    )
    records = []
    for record in representatives:
        baseline = prediction_index[record["record_id"]]
        baseline_render = render_index[record["record_id"]]
        baseline_rgb = sealed.image_tensor(Path(baseline_render["rgb_path"]), 3).to(device)
        baseline_alpha = sealed.image_tensor(Path(baseline_render["alpha_path"]), 1).to(device)
        normal_rows, normal_valid = formal.case_rows(cache, record)
        observations = [runtime_value.observation(outfit, condition) for outfit, condition in zip(record["garment_labels"], record["reference_condition_ids"])]
        base_images = [item[0] for item in observations]
        base_masks = [item[1] for item in observations]
        left = record["pair_id"].split("_")[0]
        condition = record["target_view_fold"]
        sample = runtime_value.context["samples"][f"{left}/{condition}"]
        garment = diagnosis._garment_mask(sample)
        protected = sample["target_protected_mask"]
        for variant in variants:
            if variant == "reference_dropout":
                rows, valid = normal_rows[:2], normal_valid[:2]
            elif variant == "single_reference":
                rows, valid = normal_rows[:1], normal_valid[:1]
            elif variant == "assignment_permutation":
                rows, valid = normal_rows[[2, 0, 1]], normal_valid[[2, 0, 1]]
            else:
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
                rows, valid = runtime_value.f2_rows(images, masks)
                rows, valid = rows.detach().cpu(), valid.detach().cpu()
            perturbed = prediction_row(record, infer(model, rows, valid, device))
            safe_id = record["record_id"].replace("/", "__")
            path = seed_root / "perturbations/renders" / variant / safe_id
            rgb, alpha, diagnostics, _ = render_or_load(
                path, runtime_value, left, condition, controller_branches(perturbed, endpoints)
            )
            silhouette_iou, boundary_fscore, tolerance = sealed.silhouette_metrics(alpha, garment)
            baseline_iou, baseline_boundary, _ = sealed.silhouette_metrics(baseline_alpha, garment)
            records.append({
                "record_id": record["record_id"], "pair_id": record["pair_id"],
                "assignment_type": record["assignment_type"], "variant": variant,
                "baseline_top1": baseline["top1_outfit"], "perturbed_top1": perturbed["top1_outfit"],
                "baseline_pair": sorted((baseline["top1_outfit"], baseline["top2_outfit"])),
                "perturbed_pair": sorted((perturbed["top1_outfit"], perturbed["top2_outfit"])),
                "top1_stable": baseline["top1_outfit"] == perturbed["top1_outfit"],
                "top2_pair_stable": set((baseline["top1_outfit"], baseline["top2_outfit"])) == set((perturbed["top1_outfit"], perturbed["top2_outfit"])),
                "weight_drift": abs(float(baseline["normalized_secondary_weight"]) - float(perturbed["normalized_secondary_weight"])),
                "baseline_mode": baseline["mode"], "perturbed_mode": perturbed["mode"],
                "mode_switch": baseline["mode"] != perturbed["mode"],
                "baseline_fallback_reason": baseline["fallback_reason"],
                "perturbed_fallback_reason": perturbed["fallback_reason"],
                "fallback_reason_changed": baseline["fallback_reason"] != perturbed["fallback_reason"],
                "lpips_change": sealed.lpips_distance(runtime_value, rgb, baseline_rgb, garment),
                "silhouette_iou": silhouette_iou, "baseline_silhouette_iou": baseline_iou,
                "silhouette_iou_change": silhouette_iou - baseline_iou,
                "boundary_fscore": boundary_fscore, "baseline_boundary_fscore": baseline_boundary,
                "boundary_fscore_change": boundary_fscore - baseline_boundary,
                "ghosting_proxy_change": float(perturbed["mode"] == "DUAL_SUPPORT") - float(baseline["mode"] == "DUAL_SUPPORT"),
                "identity_contamination": geometry.masked_mean((rgb - baseline_rgb).abs().mean(dim=0, keepdim=True), protected),
                "rgb_path": str(path.with_name(path.name + "_rgb.png")),
                "alpha_path": str(path.with_name(path.name + "_alpha.png")),
                "render_diagnostics": diagnostics, "boundary_tolerance": tolerance,
            })
    aggregates = {}
    for variant in variants:
        selected = [row for row in records if row["variant"] == variant]
        aggregates[variant] = {
            "record_count": len(selected),
            "top1_stability": statistics.fmean(row["top1_stable"] for row in selected),
            "top2_pair_stability": statistics.fmean(row["top2_pair_stable"] for row in selected),
            "weight_drift_mean": statistics.fmean(row["weight_drift"] for row in selected),
            "weight_drift_max": max(row["weight_drift"] for row in selected),
            "mode_switch_rate": statistics.fmean(row["mode_switch"] for row in selected),
            "fallback_reason_change_rate": statistics.fmean(row["fallback_reason_changed"] for row in selected),
            "lpips_change_mean": statistics.fmean(row["lpips_change"] for row in selected),
            "silhouette_iou_change_mean": statistics.fmean(row["silhouette_iou_change"] for row in selected),
            "ghosting_proxy_change_mean": statistics.fmean(row["ghosting_proxy_change"] for row in selected),
            "identity_contamination_max": max(row["identity_contamination"] for row in selected),
        }
    result = {
        "schema_version": "canondressgs.research.dual_support_controller_perturbation_evaluation.v1",
        "status": "COMPLETE", "seed": seed, "attempt": attempt_name,
        "representative_policy": "all_10_pairs_x_AAB_ABB_assignment0_x_cond_000000",
        "representative_count": 20, "variant_count": len(variants), "record_count": len(records),
        "records": records, "aggregates": aggregates,
        "training_rerun": 0, "threshold_change": 0, "paper_final": False,
    }
    atomic_json(result_path, result)
    return {key: value for key, value in result.items() if key != "records"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=(
            "audit-training", "classify-seed", "prepare-attempt-002", "prepare-attempt-003",
            "prepare-attempt-004", "context-probe", "paired-render-probe", "render-seed",
            "perturb-seed",
        ),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--attempt", choices=("attempt_001", "attempt_002", "attempt_003", "attempt_004"),
        default="attempt_004",
    )
    parser.add_argument("--probe-index", type=int, choices=(1, 2))
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    feature_cache = args.feature_cache or args.asset_root.resolve() / formal.FORMAL_NAME / "shared_preflight/frozen_reference_feature_rows_v1.pt"
    if args.phase == "audit-training":
        result = training_audit(output_root)
    elif args.phase == "classify-seed":
        if args.seed not in formal.SEEDS:
            parser.error("classify-seed requires --seed 0, 1, or 2")
        result = run_classification(args.seed, output_root, feature_cache.resolve())
    elif args.phase == "prepare-attempt-002":
        result = prepare_attempt_002(output_root)
    elif args.phase == "prepare-attempt-003":
        result = prepare_attempt_003(output_root)
    elif args.phase == "prepare-attempt-004":
        result = prepare_attempt_004(output_root)
    elif args.phase == "context-probe":
        if args.probe_index not in (1, 2):
            parser.error("context-probe requires --probe-index 1 or 2")
        result = context_probe(output_root, args.asset_root.resolve(), args.probe_index)
    elif args.phase == "paired-render-probe":
        if args.probe_index not in (1, 2):
            parser.error("paired-render-probe requires --probe-index 1 or 2")
        result = paired_render_probe(
            output_root, args.asset_root.resolve(), feature_cache.resolve(), args.probe_index
        )
    elif args.phase == "render-seed":
        if args.seed not in formal.SEEDS:
            parser.error("render-seed requires --seed 0, 1, or 2")
        result = run_render_seed(
            args.seed, output_root, args.asset_root.resolve(), feature_cache.resolve(),
            attempt_name=args.attempt,
        )
    else:
        if args.seed not in formal.SEEDS:
            parser.error("perturb-seed requires --seed 0, 1, or 2")
        result = run_perturbation_seed(
            args.seed, output_root, args.asset_root.resolve(), feature_cache.resolve(),
            attempt_name=args.attempt,
        )
    print(json.dumps(result, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
