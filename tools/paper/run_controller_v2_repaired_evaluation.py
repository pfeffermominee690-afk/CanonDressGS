"""Calibrate and evaluate repaired Controller V2 and matched V1 checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scene.compatibility_gated_reference_controller_v2 import (  # noqa: E402
    OUTFIT_ORDER,
    PAIR_ORDER,
    PAIR_TO_INDEX,
    CompatibilityEntry,
    CompatibilityGatedReferenceControllerV2,
    CompatibilityPrior,
    RoutingThresholds,
    route_controller_v2,
    stable_pair_prediction,
)
from scene.reference_conditioned_dual_support_controller import (  # noqa: E402
    ReferenceConditionedDualSupportController,
    stable_top2_selection,
)
from tools.paper import run_controller_v2_repaired_training as training  # noqa: E402
from tools.paper import run_reference_conditioned_dual_support_controller_formal as formal  # noqa: E402


TASK_ID = training.TASK_ID
SOURCE_HEAD = training.SOURCE_HEAD
ATTEMPT = training.ATTEMPT
SEEDS = training.SEEDS
FAMILIES = training.FAMILIES
RISK = PROJECT_ROOT / "paper_protocol/reviewer_risk"
CALIBRATION = RISK / "controller_v2_micro_pilot_calibration_contract.json"
COMPATIBILITY = RISK / "controller_v2_compatibility_manifests.json"
PERTURBATION = (
    RISK / "controller_v2_micro_pilot_perturbation_representatives.json"
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            value, handle, indent=2, sort_keys=True, ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")
    os.replace(temporary, path)


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def configure() -> None:
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)


def root(output_root: Path) -> Path:
    return output_root / ATTEMPT


def load_model(
    output_root: Path, family: str, rotation: int, seed: int,
    device: torch.device,
) -> torch.nn.Module:
    if family == "V2":
        model: torch.nn.Module = CompatibilityGatedReferenceControllerV2(
            seed=seed
        )
    else:
        model = ReferenceConditionedDualSupportController(seed=seed)
    checkpoint = (
        training.training_root(output_root, family, rotation, seed)
        / "checkpoints/final_step_150.pt"
    )
    state = torch.load(checkpoint, map_location="cpu")
    if (
        state["global_step"] != 150
        or state["family"] != family
        or state["rotation"] != rotation
        or state["seed"] != seed
        or state["source_head"] != SOURCE_HEAD
    ):
        raise RuntimeError("FINAL-CHECKPOINT-PROVENANCE-MISMATCH")
    model.load_state_dict(state["model_state_dict"], strict=True)
    return model.to(device).eval()


def compatibility_prior(rotation: int) -> tuple[CompatibilityPrior, dict[str, str]]:
    archive = read_json(COMPATIBILITY)
    manifest = next(
        row for row in archive["manifests"]
        if int(row["rotation"]) == rotation
    )
    entries = [
        CompatibilityEntry(
            pair_id=row["pair_id"],
            compatibility_score=float(row["compatibility_score"]),
            compatibility_label=row["compatibility_label"],
            manifest_sha256=manifest["manifest_content_sha256"],
            calibration_condition=row["calibration_condition"],
        )
        for row in manifest["entries"]
    ]
    labels = {
        row["pair_id"]: row["compatibility_label"]
        for row in manifest["entries"]
    }
    return CompatibilityPrior(entries), labels


def infer_raw(
    model: torch.nn.Module, family: str, rows: torch.Tensor,
    valid: torch.Tensor, device: torch.device,
) -> tuple[Any, float]:
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(rows.to(device), valid.to(device))
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return output, elapsed


def v2_raw_row(
    record: Mapping[str, Any], output: Any, elapsed: float,
) -> dict[str, Any]:
    prediction = stable_pair_prediction(output.garment_probabilities)
    return {
        "record_id": record["record_id"],
        "logical_input_sha256": record["logical_input_sha256"],
        "pair_id": record.get("pair_id"),
        "assignment_type": record["assignment_type"],
        "assignment_position": record.get("assignment_position"),
        "target_view_fold": record["target_view_fold"],
        "target_distribution": record["target_distribution"],
        "garment_labels": record["garment_labels"],
        "garment_probabilities": [
            float(value) for value in output.garment_probabilities.detach().cpu()
        ],
        "predicted_top1": prediction.predicted_top1,
        "predicted_top2": prediction.predicted_top2,
        "predicted_pair": prediction.predicted_pair,
        "pair_confidence": prediction.pair_confidence,
        "mixedness_probability":
            float(output.mixedness_probability.detach().cpu()),
        "all_pair_weights": [
            float(value) for value in output.all_pair_weights.detach().cpu()
        ],
        "valid_reference_count": output.valid_reference_count,
        "controller_seconds": elapsed,
        "target_forward_leakage": 0,
    }


def v1_raw_row(
    record: Mapping[str, Any], output: Any, elapsed: float,
) -> dict[str, Any]:
    selection = stable_top2_selection(output.probabilities)
    predicted_pair = "_".join(sorted(
        (selection.top1_outfit, selection.top2_outfit),
        key=OUTFIT_ORDER.index,
    ))
    earlier = predicted_pair.split("_")[0]
    earlier_weight = (
        selection.normalized_top2_weight_1
        if selection.top1_outfit == earlier
        else selection.normalized_top2_weight_2
    )
    return {
        "record_id": record["record_id"],
        "logical_input_sha256": record["logical_input_sha256"],
        "pair_id": record.get("pair_id"),
        "assignment_type": record["assignment_type"],
        "assignment_position": record.get("assignment_position"),
        "target_view_fold": record["target_view_fold"],
        "target_distribution": record["target_distribution"],
        "garment_labels": record["garment_labels"],
        "garment_probabilities": [
            float(value) for value in output.probabilities.detach().cpu()
        ],
        "predicted_top1": selection.top1_outfit,
        "predicted_top2": selection.top2_outfit,
        "predicted_pair": predicted_pair,
        "top2_mass": selection.top2_mass,
        "normalized_secondary_probability":
            selection.normalized_top2_weight_2,
        "predicted_earlier_pair_weight": earlier_weight,
        "mode": selection.mode,
        "fallback_reason": selection.fallback_reason,
        "controller_seconds": elapsed,
        "target_forward_leakage": 0,
    }


def is_pure(row: Mapping[str, Any]) -> bool:
    return row["assignment_type"] in {
        "AAA", "BBB", "FORMAL_PURE_ENDPOINT"
    }


def apply_v2_route(
    raw: Mapping[str, Any], output: Any, prior: CompatibilityPrior,
    tau_mix: float, tau_pair: float,
) -> dict[str, Any]:
    decision = route_controller_v2(
        output, prior,
        RoutingThresholds(
            mixedness=tau_mix,
            pair_confidence=tau_pair,
            provenance="ROTATION_SEED_CALIBRATION_FOLD_ONLY",
        ),
    )
    pair = decision.prediction.predicted_pair
    return {
        **raw,
        "thresholds": {
            "tau_mix": tau_mix,
            "tau_pair": tau_pair,
            "provenance": "ROTATION_SEED_CALIBRATION_FOLD_ONLY",
        },
        "selected_pair_weight_a": decision.selected_pair_weight_a,
        "selected_pair_weight_b": decision.selected_pair_weight_b,
        "predicted_earlier_pair_weight": decision.selected_pair_weight_a,
        "compatibility_label": decision.compatibility_label,
        "compatibility_score": decision.compatibility_score,
        "compatibility_lookup_pair": pair,
        "mode": decision.mode,
        "fallback_reason": decision.fallback_reason,
        "dominant_outfit": decision.dominant_outfit,
        "geometry_interpolation": False,
        "ground_truth_pair_compatibility_lookup": False,
    }


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def calibration_objective(
    rows: Sequence[Mapping[str, Any]], labels: Mapping[str, str]
) -> dict[str, Any]:
    pure = [row for row in rows if is_pure(row)]
    mixed = [row for row in rows if not is_pure(row)]
    wrong = [
        row for row in mixed if row["predicted_pair"] != row["pair_id"]
    ]
    correct_compatible = [
        row for row in mixed
        if row["predicted_pair"] == row["pair_id"]
        and labels[row["pair_id"]] == "COMPATIBLE"
    ]
    correct_incompatible = [
        row for row in mixed
        if row["predicted_pair"] == row["pair_id"]
        and labels[row["pair_id"]] == "INCOMPATIBLE"
    ]
    values = {
        "pure_false_mixed_rate": rate(
            sum(
                row["mixedness_probability"]
                >= row["thresholds"]["tau_mix"]
                for row in pure
            ),
            len(pure),
        ),
        "wrong_pair_dual_rate": rate(
            sum(row["mode"] == "DUAL_SUPPORT" for row in wrong),
            len(wrong),
        ),
        "incompatible_pair_dual_rate": rate(
            sum(row["mode"] == "DUAL_SUPPORT" for row in correct_incompatible),
            len(correct_incompatible),
        ),
        "correct_compatible_dual_rate": rate(
            sum(row["mode"] == "DUAL_SUPPORT" for row in correct_compatible),
            len(correct_compatible),
        ),
        "correct_incompatible_hard_rate": rate(
            sum(
                row["mode"] == "HARD_GEOMETRY_SOFT_VA"
                for row in correct_incompatible
            ),
            len(correct_incompatible),
        ),
        "mixed_false_single_rate": rate(
            sum(row["mode"] == "SINGLE_ENDPOINT" for row in mixed),
            len(mixed),
        ),
    }
    values["denominators"] = {
        "pure": len(pure),
        "mixed": len(mixed),
        "wrong_pair_mixed": len(wrong),
        "correct_compatible_mixed": len(correct_compatible),
        "correct_incompatible_mixed": len(correct_incompatible),
    }
    return values


def objective_sort_key(row: Mapping[str, Any]) -> tuple[float, ...]:
    objective = row["objective"]
    return (
        objective["pure_false_mixed_rate"],
        objective["wrong_pair_dual_rate"],
        objective["incompatible_pair_dual_rate"],
        -objective["correct_compatible_dual_rate"],
        -objective["correct_incompatible_hard_rate"],
        objective["mixed_false_single_rate"],
        -row["tau_pair"],
        -row["tau_mix"],
    )


def calibrate_one(
    model: torch.nn.Module, rotation: int, seed: int,
    records: Sequence[Mapping[str, Any]], clean_cache: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], int, float]:
    prior, labels = compatibility_prior(rotation)
    raw_rows = []
    output_rows = []
    elapsed = 0.0
    for record in records:
        rows, valid = formal.case_rows(clean_cache, record)
        output, seconds = infer_raw(model, "V2", rows, valid, device)
        raw_rows.append(v2_raw_row(record, output, seconds))
        output_rows.append(output)
        elapsed += seconds
    contract = read_json(CALIBRATION)
    candidates = []
    for candidate in contract["all_candidates"]:
        tau_mix = float(candidate["tau_mix"])
        tau_pair = float(candidate["tau_pair"])
        routed = [
            apply_v2_route(raw, output, prior, tau_mix, tau_pair)
            for raw, output in zip(raw_rows, output_rows)
        ]
        candidates.append({
            "tau_mix": tau_mix,
            "tau_pair": tau_pair,
            "objective": calibration_objective(routed, labels),
        })
    ordered = sorted(candidates, key=objective_sort_key)
    selected = ordered[0]
    scientific_key = objective_sort_key(selected)[:6]
    scientific_ties = [
        row for row in candidates
        if objective_sort_key(row)[:6] == scientific_key
    ]
    result = {
        "rotation": rotation,
        "seed": seed,
        "calibration_fold": records[0]["target_view_fold"],
        "calibration_record_count": len(records),
        "test_fold_used": False,
        "information_ablation_used": False,
        "candidate_count": len(candidates),
        "all_candidates": candidates,
        "selected_threshold": {
            "tau_mix": selected["tau_mix"],
            "tau_pair": selected["tau_pair"],
        },
        "selected_objective": selected["objective"],
        "scientific_tie_count_before_safety_tie_break":
            len(scientific_ties),
        "scientific_tie_candidates": sorted(
            [
                {"tau_mix": row["tau_mix"], "tau_pair": row["tau_pair"]}
                for row in scientific_ties
            ],
            key=lambda row: (row["tau_pair"], row["tau_mix"]),
            reverse=True,
        ),
        "selection_trace": {
            "minimization_sort_key":
                "(pure_false_mixed, wrong_pair_dual, "
                "incompatible_pair_dual, -correct_compatible_dual, "
                "-correct_incompatible_hard, mixed_false_single, "
                "-tau_pair, -tau_mix)",
            "selected_full_sort_key": list(objective_sort_key(selected)),
            "larger_tau_pair_tie_break": True,
            "larger_tau_mix_tie_break": True,
            "remaining_non_uniqueness": 0,
        },
        "compatibility_labels": labels,
        "raw_prediction_sha256": canonical_hash(raw_rows),
    }
    return result, len(records), elapsed


def target_dominant(record: Mapping[str, Any]) -> str:
    values = record["target_distribution"]
    return OUTFIT_ORDER[max(range(len(values)), key=lambda index: values[index])]


def earlier_target_weight(record: Mapping[str, Any]) -> float:
    earlier = record["pair_id"].split("_")[0]
    return float(record["target_distribution"][OUTFIT_ORDER.index(earlier)])


def row_correctness(row: Mapping[str, Any]) -> dict[str, Any]:
    pair_correct = row.get("pair_id") is not None and (
        row["predicted_pair"] == row["pair_id"]
    )
    dominant = target_dominant(row)
    result = {
        "top1_correct": row["predicted_top1"] == dominant,
        "dominant_order_correct": row["predicted_top1"] == dominant,
        "unordered_top2_pair_correct": pair_correct,
    }
    if row.get("pair_id") is not None and not is_pure(row):
        result["target_earlier_pair_weight"] = earlier_target_weight(row)
        if pair_correct:
            error = (
                row["predicted_earlier_pair_weight"]
                - result["target_earlier_pair_weight"]
            )
            result["correct_pair_weight_error"] = error
    return result


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.0
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    start = 0
    while start < len(pairs):
        end = start + 1
        while end < len(pairs) and pairs[end][0] == pairs[start][0]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        rank_sum += average_rank * sum(label for _, label in pairs[start:end])
        start = end
    return (
        rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def auprc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        return 0.0
    ordered = sorted(
        zip(scores, labels), key=lambda item: item[0], reverse=True
    )
    true_positive = 0
    total = 0
    precision_sum = 0.0
    for _, label in ordered:
        total += 1
        true_positive += label
        if label:
            precision_sum += true_positive / total
    return precision_sum / positives


def ece(labels: Sequence[int], scores: Sequence[float], bins: int = 10) -> float:
    result = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = [
            (label, score) for label, score in zip(labels, scores)
            if lower <= score < upper or (index == bins - 1 and score == 1.0)
        ]
        if selected:
            accuracy = sum(label for label, _ in selected) / len(selected)
            confidence = sum(score for _, score in selected) / len(selected)
            result += len(selected) / len(labels) * abs(accuracy - confidence)
    return result


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def rmse(values: Iterable[float]) -> float:
    values = list(values)
    return math.sqrt(mean(value * value for value in values))


def metrics(rows: Sequence[Mapping[str, Any]], family: str) -> dict[str, Any]:
    labels = [0 if is_pure(row) else 1 for row in rows]
    if family == "V2":
        scores = [row["mixedness_probability"] for row in rows]
        false_mixed = [
            row["mixedness_probability"] >= row["thresholds"]["tau_mix"]
            for row in rows if is_pure(row)
        ]
    else:
        scores = [row["normalized_secondary_probability"] for row in rows]
        false_mixed = [
            row["mode"] == "DUAL_SUPPORT" for row in rows if is_pure(row)
        ]
    pair_correct = [
        row["unordered_top2_pair_correct"] for row in rows
    ]
    mixed = [row for row in rows if not is_pure(row)]
    pure = [row for row in rows if is_pure(row)]
    correct_mixed = [
        row for row in mixed if row["unordered_top2_pair_correct"]
    ]
    compatible = [
        row for row in correct_mixed
        if row["ground_truth_compatibility_label"] == "COMPATIBLE"
    ]
    incompatible = [
        row for row in correct_mixed
        if row["ground_truth_compatibility_label"] == "INCOMPATIBLE"
    ]
    wrong = [
        row for row in mixed if not row["unordered_top2_pair_correct"]
    ]
    errors = [
        row["correct_pair_weight_error"] for row in correct_mixed
        if "correct_pair_weight_error" in row
    ]
    clipped = [
        min(max(score, 1.0e-8), 1.0 - 1.0e-8) for score in scores
    ]
    confusion = {
        outfit: {
            predicted: sum(
                target_dominant(row) == outfit
                and row["predicted_top1"] == predicted
                for row in rows
            )
            for predicted in OUTFIT_ORDER
        }
        for outfit in OUTFIT_ORDER
    }
    return {
        "record_count": len(rows),
        "pair": {
            "top1_accuracy": mean(row["top1_correct"] for row in rows),
            "unordered_top2_accuracy": mean(pair_correct),
            "dominant_order_accuracy":
                mean(row["dominant_order_correct"] for row in rows),
            "confusion_matrix": confusion,
        },
        "mixedness": {
            "auroc": auroc(labels, scores),
            "auprc": auprc(labels, scores),
            "brier": mean(
                (score - label) ** 2 for score, label in zip(scores, labels)
            ),
            "cross_entropy": -mean(
                label * math.log(score) + (1 - label) * math.log(1 - score)
                for score, label in zip(clipped, labels)
            ),
            "ece_10_bin": ece(labels, scores),
            "pure_false_mixed_rate": mean(false_mixed),
            "mixed_false_single_rate":
                mean(row["mode"] == "SINGLE_ENDPOINT" for row in mixed),
        },
        "weight": {
            "correct_pair_count": len(errors),
            "mae": mean(abs(value) for value in errors),
            "rmse": rmse(errors),
        },
        "routing": {
            "single_rate": mean(
                row["mode"] == "SINGLE_ENDPOINT" for row in rows
            ),
            "dual_rate": mean(
                row["mode"] == "DUAL_SUPPORT" for row in rows
            ),
            "hard_rate": mean(
                row["mode"] == "HARD_GEOMETRY_SOFT_VA" for row in rows
            ),
            "pure_single_rate": mean(
                row["mode"] == "SINGLE_ENDPOINT" for row in pure
            ),
            "compatible_dual_rate": mean(
                row["mode"] == "DUAL_SUPPORT" for row in compatible
            ),
            "incompatible_hard_rate": mean(
                row["mode"] == "HARD_GEOMETRY_SOFT_VA"
                for row in incompatible
            ),
            "incompatible_dual_rate": mean(
                row["mode"] == "DUAL_SUPPORT" for row in incompatible
            ),
            "wrong_pair_dual_rate": mean(
                row["mode"] == "DUAL_SUPPORT" for row in wrong
            ),
            "denominators": {
                "all": len(rows), "pure": len(pure), "mixed": len(mixed),
                "correct_compatible": len(compatible),
                "correct_incompatible": len(incompatible),
                "wrong_pair_mixed": len(wrong),
            },
        },
    }


def unique_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    result = {}
    for row in rows:
        result.setdefault(row["logical_input_sha256"], row)
    return list(result.values())


def enrich_primary(
    row: dict[str, Any], labels: Mapping[str, str],
) -> dict[str, Any]:
    result = {**row, **row_correctness(row)}
    result["ground_truth_compatibility_label"] = labels[row["pair_id"]]
    return result


def evaluate_clean(
    output_root: Path, asset_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    configure()
    contract = training.validate_contract()
    manifest = contract["manifest"]
    rotations = contract["rotations"]["rotations"]
    clean_cache = torch.load(
        asset_root / training.FEATURE_CACHE_RELATIVE, map_location="cpu"
    )
    device = torch.device("cuda")
    calibration_results = []
    thresholds: dict[tuple[int, int], tuple[float, float]] = {}
    predictions = []
    formal_pure = []
    inference_count = 0
    controller_seconds = 0.0
    index = {row["record_id"]: row for row in manifest["query_sets"]}
    for rotation_row in rotations:
        rotation = int(rotation_row["rotation"])
        calibration_records = [
            index[record_id]
            for record_id in rotation_row["partitions"]["calibration"]["record_ids"]
        ]
        test_records = [
            index[record_id]
            for record_id in rotation_row["partitions"]["test"]["record_ids"]
        ]
        prior, labels = compatibility_prior(rotation)
        for seed in SEEDS:
            v2_model = load_model(output_root, "V2", rotation, seed, device)
            calibration, count, seconds = calibrate_one(
                v2_model, rotation, seed, calibration_records,
                clean_cache, device,
            )
            calibration_results.append(calibration)
            inference_count += count
            controller_seconds += seconds
            tau_mix = calibration["selected_threshold"]["tau_mix"]
            tau_pair = calibration["selected_threshold"]["tau_pair"]
            thresholds[(rotation, seed)] = (tau_mix, tau_pair)
            for family in FAMILIES:
                model = (
                    v2_model if family == "V2"
                    else load_model(output_root, family, rotation, seed, device)
                )
                for record in test_records:
                    rows, valid = formal.case_rows(clean_cache, record)
                    output, seconds = infer_raw(
                        model, family, rows, valid, device
                    )
                    controller_seconds += seconds
                    inference_count += 1
                    if family == "V2":
                        raw = v2_raw_row(record, output, seconds)
                        row = apply_v2_route(
                            raw, output, prior, tau_mix, tau_pair
                        )
                    else:
                        row = v1_raw_row(record, output, seconds)
                        row["compatibility_label"] = labels[
                            row["predicted_pair"]
                        ]
                        row["compatibility_lookup_pair"] = (
                            row["predicted_pair"]
                        )
                        row["ground_truth_pair_compatibility_lookup"] = False
                        row["geometry_interpolation"] = False
                    row.update({
                        "family": family, "rotation": rotation, "seed": seed,
                        "split": "PRIMARY_TEST",
                    })
                    predictions.append(enrich_primary(row, labels))
                for record in manifest["formal_pure_endpoint_episodes"]:
                    rows, valid = formal.case_rows(clean_cache, record)
                    output, seconds = infer_raw(
                        model, family, rows, valid, device
                    )
                    controller_seconds += seconds
                    inference_count += 1
                    if family == "V2":
                        raw = v2_raw_row(record, output, seconds)
                        row = apply_v2_route(
                            raw, output, prior, tau_mix, tau_pair
                        )
                    else:
                        row = v1_raw_row(record, output, seconds)
                        row["compatibility_label"] = labels[
                            row["predicted_pair"]
                        ]
                    row.update({
                        "family": family, "rotation": rotation, "seed": seed,
                        "split": "FORMAL_PURE_SECONDARY",
                        "target_outfit": target_dominant(record),
                        "top1_correct":
                            row["predicted_top1"] == target_dominant(record),
                    })
                    formal_pure.append(row)
            del v2_model
            torch.cuda.empty_cache()
    if inference_count != 3360:
        raise RuntimeError(
            f"CLEAN-INFERENCE-COUNT-MISMATCH: {inference_count}"
        )
    group_metrics = []
    for family in FAMILIES:
        for rotation in range(4):
            for seed in SEEDS:
                rows = [
                    row for row in predictions
                    if row["family"] == family
                    and row["rotation"] == rotation and row["seed"] == seed
                ]
                group_metrics.append({
                    "family": family, "rotation": rotation, "seed": seed,
                    "protocol_weighted": metrics(rows, family),
                    "unique_query": metrics(unique_rows(rows), family),
                })
    calibration_archive = {
        "schema_version":
            "canondressgs.research.controller_v2_repaired_calibration.v1",
        "status": "PASS",
        "task_id": TASK_ID,
        "attempt": ATTEMPT,
        "source_head": SOURCE_HEAD,
        "selection_count": len(calibration_results),
        "candidate_count_per_selection": 81,
        "calibration_inference_count": 960,
        "test_fold_used": False,
        "test_render_used": False,
        "cross_seed_threshold_sharing": False,
        "pair_specific_thresholds": False,
        "results": calibration_results,
        "paper_final": False,
        "paper_final_count": 0,
    }
    predictions_archive = {
        "schema_version":
            "canondressgs.research.controller_v2_repaired_predictions.v1",
        "status": "PASS",
        "task_id": TASK_ID,
        "attempt": ATTEMPT,
        "source_head": SOURCE_HEAD,
        "counts": {
            "calibration": 960,
            "primary_test": len(predictions),
            "formal_pure_secondary": len(formal_pure),
            "clean_total": inference_count,
            "failed": 0,
        },
        "primary_test": predictions,
        "formal_pure_secondary": formal_pure,
        "target_forward_leakage": 0,
        "ground_truth_pair_in_inference": 0,
        "paper_final": False,
        "paper_final_count": 0,
    }
    routing_archive = {
        "schema_version":
            "canondressgs.research.controller_v2_repaired_routing.v1",
        "status": "PASS",
        "task_id": TASK_ID,
        "attempt": ATTEMPT,
        "group_metrics": group_metrics,
        "formal_pure_secondary": {
            family: {
                "record_count": sum(
                    row["family"] == family for row in formal_pure
                ),
                "top1_accuracy": mean(
                    row["top1_correct"] for row in formal_pure
                    if row["family"] == family
                ),
                "single_rate": mean(
                    row["mode"] == "SINGLE_ENDPOINT" for row in formal_pure
                    if row["family"] == family
                ),
            }
            for family in FAMILIES
        },
        "controller_seconds_total_clean": controller_seconds,
        "paper_final": False,
        "paper_final_count": 0,
    }
    atomic_json(root(output_root) / "calibration/results.json", calibration_archive)
    atomic_json(root(output_root) / "predictions/test_predictions.json", predictions_archive)
    atomic_json(root(output_root) / "routing/routing_results.json", routing_archive)
    return calibration_archive, predictions_archive, routing_archive


def perturbation_rows(
    output_root: Path, asset_root: Path,
    calibration: Mapping[str, Any],
    predictions: Mapping[str, Any],
) -> dict[str, Any]:
    configure()
    contract = training.validate_contract()
    manifest = contract["manifest"]
    record_index = {
        row["record_id"]: row for row in manifest["query_sets"]
    }
    representatives = read_json(PERTURBATION)
    clean_cache = torch.load(
        asset_root / training.FEATURE_CACHE_RELATIVE, map_location="cpu"
    )
    nuisance_cache = torch.load(
        training.augmented_cache_path(output_root), map_location="cpu"
    )
    thresholds = {
        (row["rotation"], row["seed"]): (
            row["selected_threshold"]["tau_mix"],
            row["selected_threshold"]["tau_pair"],
        )
        for row in calibration["results"]
    }
    baseline_index = {
        (row["family"], row["rotation"], row["seed"], row["record_id"]): row
        for row in predictions["primary_test"]
    }
    device = torch.device("cuda")
    results = []
    controller_seconds = 0.0
    for rotation_row in representatives["rotations"]:
        rotation = int(rotation_row["rotation"])
        prior, labels = compatibility_prior(rotation)
        records = [
            record_index[row["record_id"]]
            for row in rotation_row["representatives"]
        ]
        for seed in SEEDS:
            tau_mix, tau_pair = thresholds[(rotation, seed)]
            for family in FAMILIES:
                model = load_model(
                    output_root, family, rotation, seed, device
                )
                for record in records:
                    clean_rows, clean_valid = formal.case_rows(
                        clean_cache, record
                    )
                    for variant in representatives["variants"]:
                        if variant == "reference_dropout":
                            rows = torch.zeros_like(clean_rows)
                            valid = torch.zeros_like(clean_valid)
                        elif variant == "single_reference":
                            rows, valid = clean_rows[:1], clean_valid[:1]
                        else:
                            rows, valid = training.case_rows_variant(
                                clean_cache, nuisance_cache, record, variant
                            )
                        information_valid_count = int(
                            (valid.reshape(-1) > 0).sum().item()
                        )
                        forward_rows, forward_valid = rows, valid
                        empty_reference_forward_surrogate = None
                        if family == "MATCHED_V1" and information_valid_count == 0:
                            # The frozen V1 pool rejects an empty set.  A zero
                            # row provides no reference information but keeps
                            # the required network-forward accounting; the
                            # information boundary below then forces SINGLE.
                            forward_rows = rows[:1]
                            forward_valid = torch.ones_like(valid[:1])
                            empty_reference_forward_surrogate = (
                                "ONE_ZERO_FEATURE_ROW_NUMERIC_FORWARD_ONLY"
                            )
                        output, seconds = infer_raw(
                            model, family, forward_rows, forward_valid, device
                        )
                        controller_seconds += seconds
                        if family == "V2":
                            raw = v2_raw_row(record, output, seconds)
                            row = apply_v2_route(
                                raw, output, prior, tau_mix, tau_pair
                            )
                        else:
                            row = v1_raw_row(record, output, seconds)
                            row["compatibility_label"] = labels[
                                row["predicted_pair"]
                            ]
                            row["network_mode_before_information_boundary"] = (
                                row["mode"]
                            )
                            if information_valid_count < 2:
                                row["mode"] = "SINGLE_ENDPOINT"
                                row["fallback_reason"] = (
                                    "REFERENCE_INFORMATION_INSUFFICIENT"
                                )
                        row["information_valid_reference_count"] = (
                            information_valid_count
                        )
                        row["empty_reference_forward_surrogate"] = (
                            empty_reference_forward_surrogate
                        )
                        baseline = baseline_index[
                            (family, rotation, seed, record["record_id"])
                        ]
                        result = {
                            **row,
                            "family": family,
                            "rotation": rotation,
                            "seed": seed,
                            "variant": variant,
                            "baseline_predicted_pair":
                                baseline["predicted_pair"],
                            "baseline_mode": baseline["mode"],
                            "pair_flip":
                                row["predicted_pair"]
                                != baseline["predicted_pair"],
                            "mode_flip": row["mode"] != baseline["mode"],
                            "safe_single_endpoint":
                                row["mode"] == "SINGLE_ENDPOINT",
                            "wrong_dual_exposure": (
                                row["mode"] == "DUAL_SUPPORT"
                                and row["predicted_pair"] != row["pair_id"]
                            ),
                        }
                        if family == "V2":
                            result["mixedness_drift"] = (
                                row["mixedness_probability"]
                                - baseline["mixedness_probability"]
                            )
                            if row["predicted_pair"] == baseline["predicted_pair"]:
                                result["predicted_pair_weight_drift"] = (
                                    row["predicted_earlier_pair_weight"]
                                    - baseline[
                                        "predicted_earlier_pair_weight"
                                    ]
                                )
                        results.append(result)
                del model
                torch.cuda.empty_cache()
    if len(results) != 2880:
        raise RuntimeError(
            f"PERTURBATION-INFERENCE-COUNT-MISMATCH: {len(results)}"
        )
    by_family_variant = []
    for family in FAMILIES:
        for variant in representatives["variants"]:
            rows = [
                row for row in results
                if row["family"] == family and row["variant"] == variant
            ]
            by_family_variant.append({
                "family": family,
                "variant": variant,
                "record_count": len(rows),
                "pair_flip_rate": mean(row["pair_flip"] for row in rows),
                "mode_flip_rate": mean(row["mode_flip"] for row in rows),
                "safe_single_endpoint_rate":
                    mean(row["safe_single_endpoint"] for row in rows),
                "wrong_dual_exposure_rate":
                    mean(row["wrong_dual_exposure"] for row in rows),
                "mixedness_abs_drift": (
                    mean(abs(row["mixedness_drift"]) for row in rows)
                    if family == "V2" else None
                ),
            })
    archive = {
        "schema_version":
            "canondressgs.research.controller_v2_repaired_perturbations.v1",
        "status": "PASS",
        "task_id": TASK_ID,
        "attempt": ATTEMPT,
        "source_head": SOURCE_HEAD,
        "representative_archive_sha256":
            representatives["archive_content_sha256"],
        "representatives_per_rotation": 20,
        "variants": representatives["variants"],
        "counts": {
            "logical_inference": len(results),
            "unique_forward": len(results),
            "reused_feature_rows": len(results),
            "failed": 0,
        },
        "aggregates": by_family_variant,
        "records": results,
        "information_ablation_training_exposure": 0,
        "paper_final": False,
        "paper_final_count": 0,
    }
    atomic_json(root(output_root) / "perturbations/results.json", archive)
    return archive


def aggregate_summary(
    output_root: Path,
    calibration: Mapping[str, Any],
    predictions: Mapping[str, Any],
    routing: Mapping[str, Any],
    perturbations: Mapping[str, Any],
) -> dict[str, Any]:
    value = {
        "schema_version":
            "canondressgs.research.controller_v2_repaired_machine_evaluation.v1",
        "status": "PASS",
        "task_id": TASK_ID,
        "attempt": ATTEMPT,
        "counts": {
            "calibration_inference": 960,
            "primary_test_inference": 1920,
            "formal_pure_inference": 480,
            "perturbation_inference": 2880,
            "logical_inference": 6240,
            "unique_forward": 6240,
            "reused_feature_rows": 6240,
            "failed_inference": 0,
        },
        "calibration_sha256": canonical_hash(calibration),
        "predictions_sha256": canonical_hash(predictions),
        "routing_sha256": canonical_hash(routing),
        "perturbations_sha256": canonical_hash(perturbations),
        "paper_final": False,
        "paper_final_count": 0,
    }
    atomic_json(root(output_root) / "aggregates/machine_evaluation.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("clean", "perturb", "all"), required=True
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    arguments = parser.parse_args()
    output_root = arguments.output_root.resolve()
    asset_root = arguments.asset_root.resolve()
    if ATTEMPT != "attempt_004":
        raise RuntimeError("successful repaired evaluation is bound to attempt_004")
    if arguments.phase in {"clean", "all"}:
        calibration, predictions, routing = evaluate_clean(
            output_root, asset_root
        )
    else:
        calibration = read_json(root(output_root) / "calibration/results.json")
        predictions = read_json(root(output_root) / "predictions/test_predictions.json")
        routing = read_json(root(output_root) / "routing/routing_results.json")
    if arguments.phase in {"perturb", "all"}:
        perturbations = perturbation_rows(
            output_root, asset_root, calibration, predictions
        )
    elif arguments.phase == "all":
        perturbations = read_json(root(output_root) / "perturbations/results.json")
    else:
        result = {
            "status": "PASS",
            "phase": "clean",
            "clean_inference_count": predictions["counts"]["clean_total"],
            "perturbation_started": False,
        }
        print(json.dumps(result, sort_keys=True))
        return
    result = aggregate_summary(
        output_root, calibration, predictions, routing, perturbations
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
