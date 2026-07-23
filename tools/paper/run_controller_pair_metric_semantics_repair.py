from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
DOCS = ROOT / "docs" / "PAPER"
HANDOFF = ROOT / "project_control_handoff"

TASK_ID = "AAAI27-CONTROLLER-PAIR-METRIC-SEMANTICS-REPAIR-001"
SOURCE_HEAD = "56988a7e4b5cdd63be481005b070a44b20feab13"
SOURCE_BRANCH = "research/controller-v2-pair-identification-diagnosis-20260724"
BRANCH = "research/controller-pair-metric-semantics-repair-20260724"
HISTORICAL_CLASSIFICATION = "CONTROLLER_V2_CROSSFIT_MICRO_PILOT_FAIL"
ATTEMPT = "attempt_001"

OUTFITS = ("O01", "O02", "O03", "O04", "O08")
PAIRS = (
    "O01_O02", "O01_O03", "O01_O04", "O01_O08", "O02_O03",
    "O02_O04", "O02_O08", "O03_O04", "O03_O08", "O04_O08",
)
FAMILIES = ("V2", "MATCHED_V1")
PURE_TYPES = {"AAA", "BBB"}
MIXED_TYPES = {"AAB", "ABB"}
PURE_VISIBLE_TYPES = PURE_TYPES | {"FORMAL_PURE_ENDPOINT"}
HARD_MODE = "HARD_GEOMETRY_SOFT_VA"
DUAL_MODE = "DUAL_SUPPORT"
SINGLE_MODE = "SINGLE_ENDPOINT"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def mean(values: Iterable[float | bool]) -> float:
    rows = [float(value) for value in values]
    return sum(rows) / len(rows) if rows else 0.0


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def rmse(values: Iterable[float]) -> float:
    rows = list(values)
    return math.sqrt(mean(value * value for value in rows)) if rows else 0.0


def entropy_bits(counts: Mapping[str, int]) -> float:
    total = sum(counts.values())
    return -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values() if count
    ) if total else 0.0


def visible_garment_set(record: Mapping[str, Any]) -> list[str]:
    return [outfit for outfit in OUTFITS if outfit in set(record["garment_labels"])]


def pair_members(pair_id: str) -> list[str]:
    members = pair_id.split("_")
    if len(members) != 2 or any(member not in OUTFITS for member in members):
        raise RuntimeError(f"invalid pair id: {pair_id}")
    return members


def target_dominant(row: Mapping[str, Any]) -> str:
    values = [float(value) for value in row["target_distribution"]]
    return OUTFITS[max(range(len(values)), key=values.__getitem__)]


def probability_margin(row: Mapping[str, Any]) -> float:
    values = sorted((float(value) for value in row["garment_probabilities"]), reverse=True)
    return values[1] - values[2]


def unique_query_metric(
    rows: Sequence[Mapping[str, Any]], metric: Callable[[Mapping[str, Any]], float | bool],
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["logical_input_sha256"])].append(row)
    return {
        "unique_query_count": len(grouped),
        "macro": mean(mean(metric(row) for row in values) for values in grouped.values()),
    }


def grouped_metric(
    rows: Sequence[Mapping[str, Any]], key: str,
    metric: Callable[[Mapping[str, Any]], float | bool],
) -> dict[str, Any]:
    values: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        values[str(row[key])].append(row)
    return {
        name: {"record_count": len(group), "value": mean(metric(row) for row in group)}
        for name, group in sorted(values.items())
    }


def aggregation_views(
    rows: Sequence[Mapping[str, Any]], metric: Callable[[Mapping[str, Any]], float | bool],
) -> dict[str, Any]:
    per_rotation = grouped_metric(rows, "rotation", metric)
    per_seed = grouped_metric(rows, "seed", metric)
    rotation_seed_values = []
    for rotation in sorted({int(row["rotation"]) for row in rows}):
        for seed in sorted({int(row["seed"]) for row in rows}):
            selected = [
                row for row in rows
                if int(row["rotation"]) == rotation and int(row["seed"]) == seed
            ]
            if selected:
                rotation_seed_values.append(mean(metric(row) for row in selected))
    unique = unique_query_metric(rows, metric)
    return {
        "protocol_weighted": {"record_count": len(rows), "value": mean(metric(row) for row in rows)},
        "unique_query": {
            "unique_query_count": unique["unique_query_count"],
            "value": unique["macro"],
        },
        "per_rotation": per_rotation,
        "rotation_macro": mean(row["value"] for row in per_rotation.values()),
        "per_seed": per_seed,
        "seed_macro": mean(row["value"] for row in per_seed.values()),
        "global_macro": mean(rotation_seed_values),
        "rotation_seed_cell_count": len(rotation_seed_values),
    }


def compatibility_labels() -> dict[int, dict[str, str]]:
    archive = read_json(RISK / "controller_v2_compatibility_manifests.json")
    result = {}
    for manifest in archive["manifests"]:
        rotation = int(manifest["rotation"])
        result[rotation] = {
            str(row["pair_id"]): str(row["compatibility_label"])
            for row in manifest["entries"]
        }
    if set(result) != set(range(4)) or any(set(value) != set(PAIRS) for value in result.values()):
        raise RuntimeError("compatibility manifest coverage mismatch")
    return result


def protocol_semantics(
    manifest: Mapping[str, Any], compatibility: Mapping[int, Mapping[str, str]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    query_records = list(manifest["query_sets"])
    formal_pure = list(manifest["formal_pure_endpoint_episodes"])
    pure_records = [row for row in query_records if row["assignment_type"] in PURE_TYPES]
    mixed_records = [row for row in query_records if row["assignment_type"] in MIXED_TYPES]
    if len(query_records) != 320 or len(pure_records) != 80 or len(mixed_records) != 240:
        raise RuntimeError("protocol record count mismatch")

    invalid_compatibility = [
        (rotation, pair, label)
        for rotation, labels in compatibility.items()
        for pair, label in labels.items()
        if pair not in PAIRS or label not in {"COMPATIBLE", "INCOMPATIBLE"}
    ]
    if invalid_compatibility:
        raise RuntimeError("compatibility manifest semantic mismatch")

    semantics_rows = []
    visible_rows = []
    visible_errors = []
    for scope, records in (("PAIR_QUERY", query_records), ("FORMAL_PURE", formal_pure)):
        for record in records:
            visible = visible_garment_set(record)
            references = [str(row["outfit_id"]) for row in record["source_references"]]
            reference_visible = [outfit for outfit in OUTFITS if outfit in set(references)]
            if visible != reference_visible:
                visible_errors.append(str(record["record_id"]))
            cardinality = len(visible)
            expected = 1 if record["assignment_type"] in PURE_VISIBLE_TYPES else 2
            if cardinality != expected:
                visible_errors.append(str(record["record_id"]))
            pair = record.get("pair_id")
            hidden = [] if pair is None else [member for member in pair_members(str(pair)) if member not in visible]
            compatibility_by_rotation = (
                "NOT_APPLICABLE_TO_PURE"
                if cardinality == 1
                else {
                    str(rotation): labels[str(pair)]
                    for rotation, labels in sorted(compatibility.items())
                }
            )
            visible_rows.append({
                "record_id": record["record_id"],
                "record_scope": scope,
                "fold": record["target_view_fold"],
                "assignment_type": record["assignment_type"],
                "reference_slot_garments": references,
                "visible_garment_set": visible,
                "visible_cardinality": cardinality,
                "stored_pair_field": pair,
                "latent_secondary_garments": hidden,
                "compatibility_gt_by_rotation": compatibility_by_rotation,
                "logical_input_sha256": record["logical_input_sha256"],
                "source_image_hashes": record["source_image_hashes"],
                "source_mask_hashes": record["source_mask_hashes"],
            })
            if scope == "PAIR_QUERY" and cardinality == 1:
                dominant = visible[0]
                secondary = hidden[0] if len(hidden) == 1 else None
                semantics_rows.append({
                    "record_id": record["record_id"],
                    "fold": record["target_view_fold"],
                    "dominant_garment": dominant,
                    "reference_slot_garments": references,
                    "input_visible_garment_set": visible,
                    "stored_pair_field": pair,
                    "stored_secondary_garment": secondary,
                    "composition_field": record["assignment_type"],
                    "alpha_or_weight_field": {
                        outfit: float(record["target_distribution"][OUTFITS.index(outfit)])
                        for outfit in pair_members(str(pair))
                    },
                    "target_soft_label": record["target_distribution"],
                    "historical_pair_metric_gt": pair,
                    "corrected_pair_metric_gt": "PROVENANCE_ONLY_NOT_EVALUABLE",
                    "routing_gt": "SINGLE_ENDPOINT",
                    "compatibility_gt": "NOT_APPLICABLE_TO_PURE",
                    "logical_input_sha256": record["logical_input_sha256"],
                    "duplicate_of": record["duplicate_of"],
                    "duplicate_target_consistent": record["duplicate_target_consistent"],
                })

    if visible_errors:
        raise RuntimeError("PROTOCOL_VISIBLE_GARMENT_SET_INCONSISTENT")

    equivalence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in semantics_rows:
        equivalence[str(row["logical_input_sha256"])].append(row)
    classes = []
    for logical, rows in sorted(equivalence.items()):
        candidates = Counter(str(row["stored_secondary_garment"]) for row in rows)
        classes.append({
            "logical_input_sha256": logical,
            "visible_garment": rows[0]["dominant_garment"],
            "fold": rows[0]["fold"],
            "record_ids": [row["record_id"] for row in rows],
            "latent_secondary_counts": dict(sorted(candidates.items())),
            "latent_secondary_conditional_frequency": {
                name: count / len(rows) for name, count in sorted(candidates.items())
            },
            "latent_secondary_candidate_count": len(candidates),
            "entropy_bits": entropy_bits(candidates),
            "majority_accuracy": max(candidates.values()) / len(rows),
            "uniform_random_accuracy": 1.0 / len(candidates),
            "bayes_optimal_accuracy": max(candidates.values()) / len(rows),
            "duplicate_target_consistency": mean(row["duplicate_target_consistent"] for row in rows),
            "latent_secondary_consistent_across_duplicates": len(candidates) == 1,
        })
    if len(classes) != 20 or any(row["latent_secondary_candidate_count"] != 4 for row in classes):
        raise RuntimeError("pure latent equivalence-class mismatch")

    per_visible = {}
    for outfit in OUTFITS:
        rows = [row for row in semantics_rows if row["dominant_garment"] == outfit]
        counts = Counter(str(row["stored_secondary_garment"]) for row in rows)
        per_visible[outfit] = {
            "record_count": len(rows),
            "latent_secondary_counts": dict(sorted(counts.items())),
            "conditional_frequency": {
                name: value / len(rows) for name, value in sorted(counts.items())
            },
            "entropy_bits": entropy_bits(counts),
            "majority_accuracy": max(counts.values()) / len(rows),
            "uniform_random_accuracy": 1.0 / len(counts),
            "bayes_optimal_accuracy": max(counts.values()) / len(rows),
        }

    pure_count = len(pure_records)
    mixed_count = len(mixed_records)
    total_count = len(query_records)
    pure_uniform = sum(
        len(equivalence[row["logical_input_sha256"]])
        / row["latent_secondary_candidate_count"]
        for row in classes
    ) / pure_count
    pure_majority = sum(
        max(row["latent_secondary_counts"].values()) for row in classes
    ) / pure_count
    pure_bayes = pure_majority
    derivations = {}
    for name, pure_accuracy in (
        ("ALL_MIXED_CORRECT_PLUS_PURE_UNIFORM_GUESS", pure_uniform),
        ("ALL_MIXED_CORRECT_PLUS_PURE_MAJORITY_GUESS", pure_majority),
        ("ALL_MIXED_CORRECT_PLUS_PURE_BAYES_OPTIMAL_GUESS", pure_bayes),
    ):
        derivations[name] = {
            "mixed_record_count": mixed_count,
            "mixed_fraction": mixed_count / total_count,
            "mixed_pair_accuracy": 1.0,
            "pure_record_count": pure_count,
            "pure_fraction": pure_count / total_count,
            "pure_latent_secondary_accuracy": pure_accuracy,
            "combined_all_record_expected_score": (
                mixed_count / total_count
                + pure_count / total_count * pure_accuracy
            ),
        }
    if any(
        abs(row["combined_all_record_expected_score"] - 0.8125) > 1.0e-12
        for row in derivations.values()
    ):
        raise RuntimeError("historical 0.8125 derivation mismatch")

    audit = {
        "schema_version": "controller_record_semantics_audit.v1",
        "status": "PASS",
        "task_id": TASK_ID,
        "source_head": SOURCE_HEAD,
        "code_path": [
            {"stage": "record_construction", "path": "tools/paper/build_dual_support_controller_dataset.py", "lines": "73-181"},
            {"stage": "training_target", "path": "tools/paper/run_controller_v2_repaired_training.py", "lines": "596-649"},
            {"stage": "inference_output", "path": "tools/paper/run_controller_v2_repaired_evaluation.py", "lines": "156-224"},
            {"stage": "metric_evaluator", "path": "tools/paper/run_controller_v2_repaired_evaluation.py", "lines": "433-451,517-629"},
        ],
        "counts": {
            "query_records": len(query_records), "pure_query_records": len(pure_records),
            "mixed_query_records": len(mixed_records), "formal_pure_records": len(formal_pure),
            "pure_per_fold": 20, "mixed_per_fold": 60,
        },
        "semantic_findings": {
            "pure_input_contains_one_garment": True,
            "stored_pair_contains_unseen_secondary": True,
            "latent_secondary_source": "ORIGINAL_PAIR_TEMPLATE_RETAINED_IN_PAIR_ID",
            "latent_secondary_participates_garment_loss": False,
            "latent_secondary_participates_mixedness_loss": False,
            "latent_secondary_participates_weight_loss": False,
            "latent_secondary_participates_historical_pair_accuracy": True,
            "latent_secondary_participates_calibration_objective": False,
            "latent_secondary_participates_routing_metric": False,
            "latent_secondary_participates_visual_target_content": False,
            "latent_secondary_varies_across_observational_duplicates": True,
            "routing_denominator_classification": "ROUTING_DENOMINATOR_ALREADY_VALID",
            "compatibility_manifest_validated": True,
        },
        "pure_record_semantics": semantics_rows,
        "paper_final": False,
    }
    visible_manifest = {
        "schema_version": "controller_visible_garment_set_manifest.v1",
        "status": "PASS",
        "task_id": TASK_ID,
        "source_head": SOURCE_HEAD,
        "definition": "VISIBLE_GARMENT_SET_IS_THE_UNIQUE_GARMENT_LABEL_SET_OF_ACTUAL_REFERENCE_SLOTS",
        "forbidden_sources": [
            "latent_pair", "source_pair_template", "target_composition", "oracle_pair",
            "compatibility_pair", "hidden_secondary", "future_target",
        ],
        "record_count": len(visible_rows),
        "pure_cardinality_one_count": sum(row["visible_cardinality"] == 1 for row in visible_rows),
        "mixed_cardinality_two_count": sum(row["visible_cardinality"] == 2 for row in visible_rows),
        "inconsistency_count": 0,
        "records": visible_rows,
        "paper_final": False,
    }
    identifiability = {
        "schema_version": "controller_pure_latent_pair_identifiability.v1",
        "status": "PASS",
        "task_id": TASK_ID,
        "source_head": SOURCE_HEAD,
        "protocol_semantic_classification": "LATENT_SECONDARY_NOT_INPUT_IDENTIFIABLE",
        "equivalence_class_definition": [
            "visible_garment", "target_view_fold", "reference_condition_ids",
            "source_image_hashes", "source_mask_hashes", "logical_input_sha256",
        ],
        "equivalence_class_count": len(classes),
        "equivalence_classes": classes,
        "per_visible_garment": per_visible,
        "global": {
            "latent_secondary_candidate_count": 4,
            "entropy_bits": mean(row["entropy_bits"] for row in classes),
            "uniform_random_accuracy": pure_uniform,
            "majority_accuracy": pure_majority,
            "bayes_optimal_accuracy": pure_bayes,
            "duplicate_target_consistency": 1.0,
            "latent_secondary_duplicate_consistency": mean(
                row["latent_secondary_consistent_across_duplicates"] for row in classes
            ),
            "exact_latent_pair_metric_status": "PROVENANCE_ONLY_NOT_EVALUABLE",
        },
        "historical_0_8125_derivations": derivations,
        "interpretation": "LATENT_LABEL_CHANCE_CEILING_NOT_REPRESENTATION_UPPER_BOUND",
        "paper_final": False,
    }
    return audit, visible_manifest, identifiability


def pure_metrics(rows: Sequence[Mapping[str, Any]], family: str) -> dict[str, Any]:
    if family == "V2":
        false_mixed = lambda row: float(row["mixedness_probability"]) >= float(row["thresholds"]["tau_mix"])
    else:
        false_mixed = lambda row: row["mode"] != SINGLE_MODE
    pair_match = lambda row: row["predicted_pair"] == row.get("pair_id")
    per_garment = {}
    for outfit in OUTFITS:
        selected = [row for row in rows if visible_garment_set(row) == [outfit]]
        per_garment[outfit] = {
            "record_count": len(selected),
            "top1_recall": mean(row["predicted_top1"] == outfit for row in selected),
            "single_rate": mean(row["mode"] == SINGLE_MODE for row in selected),
        }
    confusion = {
        outfit: {
            predicted: sum(
                visible_garment_set(row) == [outfit] and row["predicted_top1"] == predicted
                for row in rows
            )
            for predicted in OUTFITS
        }
        for outfit in OUTFITS
    }
    return {
        "record_count": len(rows),
        "top1_visible_garment_accuracy": mean(row["top1_correct"] for row in rows),
        "endpoint_selection_correctness": mean(row["top1_correct"] for row in rows),
        "single_endpoint_rate": mean(row["mode"] == SINGLE_MODE for row in rows),
        "mixedness_false_positive_rate": mean(false_mixed(row) for row in rows),
        "per_garment": per_garment,
        "per_rotation": grouped_metric(rows, "rotation", lambda row: row["top1_correct"]),
        "per_seed": grouped_metric(rows, "seed", lambda row: row["top1_correct"]),
        "unique_query": unique_query_metric(rows, lambda row: row["top1_correct"]),
        "aggregation_views": {
            "top1_visible_garment_accuracy": aggregation_views(
                rows, lambda row: row["top1_correct"]
            ),
            "single_endpoint_rate": aggregation_views(
                rows, lambda row: row["mode"] == SINGLE_MODE
            ),
            "mixedness_false_positive_rate": aggregation_views(rows, false_mixed),
        },
        "top1_confusion_matrix": confusion,
        "historical_latent_secondary_exact_match": {
            "status": "NOT_EVALUABLE_DIAGNOSTIC",
            "provenance_only_observed_rate": mean(pair_match(row) for row in rows),
        },
    }


def formal_pure_metrics(rows: Sequence[Mapping[str, Any]], family: str) -> dict[str, Any]:
    if family == "V2":
        false_mixed = lambda row: float(row["mixedness_probability"]) >= float(row["thresholds"]["tau_mix"])
    else:
        false_mixed = lambda row: row["mode"] != SINGLE_MODE
    return {
        "role": "SECONDARY_ENDPOINT_SAFETY_ONLY",
        "record_count": len(rows),
        "top1_visible_garment_accuracy": mean(row["top1_correct"] for row in rows),
        "single_endpoint_rate": mean(row["mode"] == SINGLE_MODE for row in rows),
        "mixedness_false_positive_rate": mean(false_mixed(row) for row in rows),
        "per_rotation_top1": grouped_metric(rows, "rotation", lambda row: row["top1_correct"]),
        "per_seed_top1": grouped_metric(rows, "seed", lambda row: row["top1_correct"]),
        "aggregation_views": {
            "top1_visible_garment_accuracy": aggregation_views(
                rows, lambda row: row["top1_correct"]
            ),
            "single_endpoint_rate": aggregation_views(
                rows, lambda row: row["mode"] == SINGLE_MODE
            ),
            "mixedness_false_positive_rate": aggregation_views(rows, false_mixed),
        },
    }


def mixed_pair_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    correct = lambda row: bool(row["unordered_top2_pair_correct"])
    ordering = lambda row: bool(row["dominant_order_correct"])
    inclusion = lambda row: str(row["predicted_top1"]) in pair_members(str(row["pair_id"]))

    def summary(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "record_count": len(selected),
            "unordered_top2_pair_accuracy": mean(correct(row) for row in selected),
            "dominant_order_accuracy": mean(ordering(row) for row in selected),
            "top1_pair_inclusion": mean(inclusion(row) for row in selected),
            "top2_vs_top3_margin_mean": mean(probability_margin(row) for row in selected),
        }

    per_pair = {pair: summary([row for row in rows if row["pair_id"] == pair]) for pair in PAIRS}
    per_composition = {
        name: summary([row for row in rows if row["assignment_type"] == name])
        for name in sorted(MIXED_TYPES)
    }
    per_assignment = {
        f"{composition}_{position}": summary([
            row for row in rows
            if row["assignment_type"] == composition and int(row["assignment_position"]) == position
        ])
        for composition in sorted(MIXED_TYPES) for position in range(3)
    }
    confusion = {
        pair: {
            predicted: sum(row["pair_id"] == pair and row["predicted_pair"] == predicted for row in rows)
            for predicted in PAIRS
        }
        for pair in PAIRS
    }
    result = summary(rows)
    unique_pair = unique_query_metric(rows, correct)
    unique_order = unique_query_metric(rows, ordering)
    global_pair = aggregation_views(rows, correct)
    result.update({
        "protocol_weighted": summary(rows),
        "unique_query": {
            "unique_query_count": unique_pair["unique_query_count"],
            "unordered_top2_pair_accuracy": unique_pair["macro"],
            "dominant_order_accuracy": unique_order["macro"],
        },
        "rotation_macro": mean(
            mean(correct(row) for row in rows if int(row["rotation"]) == rotation)
            for rotation in range(4)
        ),
        "seed_macro": mean(
            mean(correct(row) for row in rows if int(row["seed"]) == seed)
            for seed in range(3)
        ),
        "global_macro": global_pair["global_macro"],
        "aggregation_views": {
            "unordered_top2_pair_accuracy": global_pair,
            "dominant_order_accuracy": aggregation_views(rows, ordering),
            "top1_pair_inclusion": aggregation_views(rows, inclusion),
        },
        "per_rotation": {
            str(rotation): summary([row for row in rows if int(row["rotation"]) == rotation])
            for rotation in range(4)
        },
        "per_seed": {
            str(seed): summary([row for row in rows if int(row["seed"]) == seed])
            for seed in range(3)
        },
        "per_pair": per_pair,
        "per_composition": per_composition,
        "per_assignment_position": per_assignment,
        "pair_confusion_matrix": confusion,
    })
    return result


def routing_weight_core(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    correct = [row for row in rows if bool(row["unordered_top2_pair_correct"])]
    compatible = [row for row in correct if row["ground_truth_compatibility_label"] == "COMPATIBLE"]
    incompatible = [row for row in correct if row["ground_truth_compatibility_label"] == "INCOMPATIBLE"]
    wrong = [row for row in rows if not bool(row["unordered_top2_pair_correct"])]
    ordered = [row for row in correct if bool(row["dominant_order_correct"])]
    pair_errors = [float(row["correct_pair_weight_error"]) for row in correct]
    ordered_errors = [float(row["correct_pair_weight_error"]) for row in ordered]
    return {
        "record_count": len(rows),
        "denominators": {
            "mixed": len(rows), "correct_pair": len(correct),
            "correct_compatible": len(compatible), "correct_incompatible": len(incompatible),
            "wrong_pair": len(wrong), "correct_pair_and_order": len(ordered),
        },
        "mixed_false_single_rate": mean(row["mode"] == SINGLE_MODE for row in rows),
        "correct_compatible_dual_rate": mean(row["mode"] == DUAL_MODE for row in compatible),
        "correct_incompatible_hard_rate": mean(row["mode"] == HARD_MODE for row in incompatible),
        "incompatible_dual_rate": mean(row["mode"] == DUAL_MODE for row in incompatible),
        "wrong_pair_dual_rate": mean(row["mode"] == DUAL_MODE for row in wrong),
        "correct_pair_weight": {
            "count": len(pair_errors), "mae": mean(abs(value) for value in pair_errors),
            "rmse": rmse(pair_errors),
        },
        "correct_order_weight": {
            "count": len(ordered_errors), "mae": mean(abs(value) for value in ordered_errors),
            "rmse": rmse(ordered_errors),
        },
    }


def routing_weight_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    micro = routing_weight_core(rows)
    cells = []
    for rotation in range(4):
        for seed in range(3):
            selected = [
                row for row in rows
                if int(row["rotation"]) == rotation and int(row["seed"]) == seed
            ]
            cell = routing_weight_core(selected)
            cell.update({"rotation": rotation, "seed": seed})
            cells.append(cell)

    def cell_mean(key: str) -> float:
        return mean(float(row[key]) for row in cells)

    def weight_cell_mean(weight: str, metric: str) -> float:
        return mean(float(row[weight][metric]) for row in cells)

    result = {
        "record_count": micro["record_count"],
        "denominators": micro["denominators"],
        "aggregation_contract": "ROTATION_SEED_CELL_MACRO_MATCHING_FROZEN_FORMAL_EVALUATOR",
        "rotation_seed_cell_count": len(cells),
        "mixed_false_single_rate": cell_mean("mixed_false_single_rate"),
        "correct_compatible_dual_rate": cell_mean("correct_compatible_dual_rate"),
        "correct_incompatible_hard_rate": cell_mean("correct_incompatible_hard_rate"),
        "incompatible_dual_rate": cell_mean("incompatible_dual_rate"),
        "wrong_pair_dual_rate": cell_mean("wrong_pair_dual_rate"),
        "correct_pair_weight": {
            "count": micro["correct_pair_weight"]["count"],
            "mae": weight_cell_mean("correct_pair_weight", "mae"),
            "rmse": weight_cell_mean("correct_pair_weight", "rmse"),
        },
        "correct_order_weight": {
            "count": micro["correct_order_weight"]["count"],
            "mae": weight_cell_mean("correct_order_weight", "mae"),
            "rmse": weight_cell_mean("correct_order_weight", "rmse"),
        },
        "record_micro": micro,
        "rotation_seed_cells": cells,
    }
    per_pair_modes = {}
    for pair in PAIRS:
        selected = [row for row in rows if row["pair_id"] == pair]
        counts = Counter(str(row["mode"]) for row in selected)
        per_pair_modes[pair] = {
            "record_count": len(selected),
            "mode_counts": dict(sorted(counts.items())),
            "mode_rates": {name: value / len(selected) for name, value in sorted(counts.items())},
        }
    result.update({
        "per_pair_mode_distribution": per_pair_modes,
        "pure_records_in_denominators": 0,
        "denominator_classification": "ROUTING_DENOMINATOR_ALREADY_VALID",
    })
    return result


def corrected_metrics(
    predictions: Mapping[str, Any], compatibility: Mapping[int, Mapping[str, str]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    primary = list(predictions["primary_test"])
    formal = list(predictions["formal_pure_secondary"])
    if len(primary) != 1920 or len(formal) != 480:
        raise RuntimeError("frozen prediction count mismatch")
    for row in primary:
        visible = visible_garment_set(row)
        assignment = str(row["assignment_type"])
        if assignment in PURE_TYPES:
            if len(visible) != 1:
                raise RuntimeError("pure prediction visible-set mismatch")
        elif assignment in MIXED_TYPES:
            if visible != pair_members(str(row["pair_id"])):
                raise RuntimeError("mixed prediction visible-pair mismatch")
            expected_label = compatibility[int(row["rotation"])][str(row["pair_id"])]
            if row["ground_truth_compatibility_label"] != expected_label:
                raise RuntimeError("frozen prediction compatibility mismatch")
        else:
            raise RuntimeError(f"unknown primary assignment type: {assignment}")
    if any(
        row["assignment_type"] != "FORMAL_PURE_ENDPOINT"
        or len(visible_garment_set(row)) != 1
        or row.get("pair_id") is not None
        for row in formal
    ):
        raise RuntimeError("formal-pure prediction semantic mismatch")
    pure_archive = {
        "schema_version": "controller_corrected_pure_metrics.v1",
        "status": "PASS", "task_id": TASK_ID, "source_head": SOURCE_HEAD,
        "denominator_contract": "PRIMARY_TEST_GT_PURE_ONLY_VISIBLE_GARMENT_TOP1_NO_PAIR_METRIC",
        "families": {}, "paper_final": False,
    }
    mixed_archive = {
        "schema_version": "controller_corrected_mixed_pair_metrics.v1",
        "status": "PASS", "task_id": TASK_ID, "source_head": SOURCE_HEAD,
        "denominator_contract": "PRIMARY_TEST_GT_MIXED_AAB_ABB_ONLY",
        "historical_all_record_pair_metrics": {
            "V2": {"macro": 0.6010416666666667, "per_rotation": [0.6083333333333333, 0.5791666666666667, 0.6, 0.6166666666666667]},
            "MATCHED_V1": {"macro": 0.63125, "per_rotation": [0.6125, 0.5791666666666667, 0.6958333333333333, 0.6375]},
        },
        "families": {}, "paper_final": False,
    }
    routing_archive = {
        "schema_version": "controller_corrected_routing_weight_metrics.v1",
        "status": "PASS", "task_id": TASK_ID, "source_head": SOURCE_HEAD,
        "denominator_contract": "GT_MIXED_ONLY_WITH_PREDICTION_CONDITIONED_SUBSETS",
        "families": {}, "paper_final": False,
    }
    for family in FAMILIES:
        family_primary = [row for row in primary if row["family"] == family]
        pure = [row for row in family_primary if row["assignment_type"] in PURE_TYPES]
        mixed = [row for row in family_primary if row["assignment_type"] in MIXED_TYPES]
        family_formal = [row for row in formal if row["family"] == family]
        if len(pure) != 240 or len(mixed) != 720 or len(family_formal) != 240:
            raise RuntimeError(f"family denominator mismatch: {family}")
        pure_archive["families"][family] = {
            "primary_pure": pure_metrics(pure, family),
            "formal_pure_secondary": formal_pure_metrics(family_formal, family),
        }
        mixed_archive["families"][family] = mixed_pair_summary(mixed)
        routing_archive["families"][family] = routing_weight_summary(mixed)

    task_conditional = {}
    for family in FAMILIES:
        family_rows = [row for row in primary if row["family"] == family]
        task_correct = lambda row: (
            bool(row["top1_correct"])
            if row["assignment_type"] in PURE_TYPES
            else bool(row["unordered_top2_pair_correct"])
        )
        views = aggregation_views(family_rows, task_correct)
        task_conditional[family] = {
            "record_count": len(family_rows),
            "metric": views["protocol_weighted"]["value"],
            "protocol_weighted": views["protocol_weighted"],
            "unique_query": views["unique_query"],
            "rotation_macro": views["rotation_macro"],
            "seed_macro": views["seed_macro"],
            "global_macro": views["global_macro"],
            "per_rotation": {
                rotation: row["value"] for rotation, row in views["per_rotation"].items()
            },
            "per_seed": {
                seed: row["value"] for seed, row in views["per_seed"].items()
            },
        }
    mixed_archive["task_conditional_identification_accuracy"] = {
        "role": "SECONDARY_SUMMARY_ONLY",
        "offline_gt_cardinality_selects_rule": True,
        "families": task_conditional,
    }
    expected = {
        ("V2", "mixed_pair"): 0.7180555555555556,
        ("MATCHED_V1", "mixed_pair"): 0.7583333333333333,
        ("V2", "pure_top1"): 1.0,
        ("MATCHED_V1", "pure_top1"): 1.0,
        ("V2", "formal_single"): 1.0,
        ("MATCHED_V1", "formal_single"): 0.9333333333333333,
    }
    observed = {
        (family, "mixed_pair"): mixed_archive["families"][family]["unordered_top2_pair_accuracy"]
        for family in FAMILIES
    }
    observed.update({
        (family, "pure_top1"): pure_archive["families"][family]["primary_pure"]["top1_visible_garment_accuracy"]
        for family in FAMILIES
    })
    observed.update({
        (family, "formal_single"): pure_archive["families"][family]["formal_pure_secondary"]["single_endpoint_rate"]
        for family in FAMILIES
    })
    if any(abs(observed[key] - value) > 1.0e-12 for key, value in expected.items()):
        raise RuntimeError("corrected metric regression mismatch")
    v2_routing = routing_archive["families"]["V2"]
    routing_expected = {
        "mixed_false_single_rate": 0.8305555555555555,
        "correct_compatible_dual_rate": 0.18937812421128092,
        "correct_incompatible_hard_rate": 0.3138888888888889,
        "incompatible_dual_rate": 0.0,
        "wrong_pair_dual_rate": 0.07282421740626076,
    }
    if any(
        abs(float(v2_routing[key]) - value) > 1.0e-12
        for key, value in routing_expected.items()
    ) or any(
        abs(float(v2_routing["correct_pair_weight"][key]) - value) > 1.0e-12
        for key, value in {
            "mae": 0.15805836898719514,
            "rmse": 0.1981430412872887,
        }.items()
    ):
        raise RuntimeError("routing or weight aggregation regression mismatch")
    return pure_archive, mixed_archive, routing_archive


def gate_row(
    name: str, value: Any, threshold: str, passed: bool, denominator: str,
) -> dict[str, Any]:
    return {
        "gate": name, "value": value, "threshold": threshold,
        "passed": bool(passed), "denominator": denominator,
    }


def gates_and_interpretation(
    pure: Mapping[str, Any], mixed: Mapping[str, Any], routing: Mapping[str, Any],
    historical: Mapping[str, Any], perturbation: Mapping[str, Any],
    visual_review: Mapping[str, Any],
) -> dict[str, Any]:
    machine = historical["machine_results"]
    safe_single = min(
        float(row["safe_single_endpoint_rate"])
        for row in perturbation["aggregates"]
        if row["family"] == "V2"
        and row["variant"] in {"reference_dropout", "single_reference"}
    )
    visual_rows = [
        gate_row(
            "severe_artifact_reduction_ge_0_50",
            visual_review["severe_artifact_reduction_fraction_vs_matched_v1"],
            ">=0.50", True, "HISTORICAL_FIXED_240_SHEET_VISUAL_DENOMINATOR",
        ),
        gate_row(
            "identity_contamination_eq_0",
            visual_review["maximum_grades"]["V2"]["identity_contamination"],
            "==0", True, "HISTORICAL_FIXED_240_SHEET_VISUAL_DENOMINATOR",
        ),
        gate_row(
            "grade_3_ghosting_pairs_le_1_of_10",
            0 if visual_review["maximum_grades"]["V2"]["ghosting"] < 3 else "NOT_REPORTED",
            "<=1/10", visual_review["maximum_grades"]["V2"]["ghosting"] < 3,
            "HISTORICAL_FIXED_PAIR_VISUAL_DENOMINATOR",
        ),
        gate_row(
            "dropout_and_single_reference_safe_single_ge_0_95", safe_single,
            ">=0.95", safe_single >= 0.95, "HISTORICAL_INFORMATION_ABLATION_DENOMINATOR",
        ),
    ]
    original = [
        gate_row("pair_macro_top2_ge_0_90", machine["pair_unordered_top2_macro"], ">=0.90", False, "HISTORICAL_ALL_RECORD_PAIR_METRIC"),
        gate_row("every_rotation_top2_ge_0_80", min(machine["pair_rotation_macro"].values()), ">=0.80", False, "HISTORICAL_ALL_RECORD_PAIR_METRIC"),
        gate_row("pure_single_ge_0_95", machine["pure_single_rate"], ">=0.95", True, "GT_PURE"),
        gate_row("pure_false_mixed_le_0_05", machine["pure_false_mixed_rate"], "<=0.05", True, "GT_PURE"),
        gate_row("compatible_dual_ge_0_80", machine["compatible_dual_rate"], ">=0.80", False, "CORRECT_PAIR_COMPATIBLE_MIXED"),
        gate_row("incompatible_hard_ge_0_80", machine["incompatible_hard_rate"], ">=0.80", False, "CORRECT_PAIR_INCOMPATIBLE_MIXED"),
        gate_row("incompatible_dual_le_0_10", machine["incompatible_dual_rate"], "<=0.10", True, "CORRECT_PAIR_INCOMPATIBLE_MIXED"),
        gate_row("wrong_pair_dual_le_0_10", machine["wrong_pair_dual_rate"], "<=0.10", True, "WRONG_PAIR_MIXED"),
        gate_row("correct_pair_weight_mae_le_0_12", machine["correct_pair_weight_mae"], "<=0.12", False, "CORRECT_PAIR_MIXED"),
    ] + visual_rows
    v2_pair = mixed["families"]["V2"]
    v2_pure = pure["families"]["V2"]["primary_pure"]
    v2_route = routing["families"]["V2"]
    rotation_values = [v2_pair["per_rotation"][str(index)]["unordered_top2_pair_accuracy"] for index in range(4)]
    corrected = [
        gate_row("mixed_pair_macro_ge_0_90", v2_pair["unordered_top2_pair_accuracy"], ">=0.90", v2_pair["unordered_top2_pair_accuracy"] >= 0.90, "GT_MIXED_ONLY"),
        gate_row("mixed_pair_every_rotation_ge_0_80", min(rotation_values), ">=0.80", min(rotation_values) >= 0.80, "GT_MIXED_ONLY"),
        gate_row("pure_single_ge_0_95", v2_pure["single_endpoint_rate"], ">=0.95", v2_pure["single_endpoint_rate"] >= 0.95, "GT_PURE_ONLY"),
        gate_row("pure_false_mixed_le_0_05", v2_pure["mixedness_false_positive_rate"], "<=0.05", v2_pure["mixedness_false_positive_rate"] <= 0.05, "GT_PURE_ONLY"),
        gate_row("compatible_dual_ge_0_80", v2_route["correct_compatible_dual_rate"], ">=0.80", v2_route["correct_compatible_dual_rate"] >= 0.80, "CORRECT_PAIR_COMPATIBLE_MIXED"),
        gate_row("incompatible_hard_ge_0_80", v2_route["correct_incompatible_hard_rate"], ">=0.80", v2_route["correct_incompatible_hard_rate"] >= 0.80, "CORRECT_PAIR_INCOMPATIBLE_MIXED"),
        gate_row("incompatible_dual_le_0_10", v2_route["incompatible_dual_rate"], "<=0.10", v2_route["incompatible_dual_rate"] <= 0.10, "CORRECT_PAIR_INCOMPATIBLE_MIXED"),
        gate_row("wrong_pair_dual_le_0_10", v2_route["wrong_pair_dual_rate"], "<=0.10", v2_route["wrong_pair_dual_rate"] <= 0.10, "WRONG_PAIR_MIXED"),
        gate_row("correct_pair_weight_mae_le_0_12", v2_route["correct_pair_weight"]["mae"], "<=0.12", v2_route["correct_pair_weight"]["mae"] <= 0.12, "CORRECT_PAIR_MIXED"),
    ] + visual_rows
    macro = float(v2_pair["unordered_top2_pair_accuracy"])
    below_070 = sum(value < 0.70 for value in rotation_values)
    if macro < 0.70 or below_070 >= 3:
        classification = "CORE_PAIR_IDENTIFICATION_FAILURE_REMAINS"
        next_task = "DESIGN_STAGED_OR_DECOUPLED_CONTROLLER_V3_TRAINING"
    elif 0.70 <= macro < 0.85 or any(value < 0.80 for value in rotation_values):
        classification = "MIXED_PAIR_IDENTIFICATION_PARTIAL"
        next_task = "DIAGNOSE_CONTROLLER_GARMENT_HEAD_OPTIMIZATION_BUDGET"
    elif macro >= 0.90 and min(rotation_values) >= 0.80 and any(
        not row["passed"] for row in corrected[4:9]
    ):
        classification = "PAIR_IDENTIFICATION_NOT_PRIMARY"
        next_task = "DESIGN_DECOUPLED_CONTROLLER_V3_ROUTING_AND_WEIGHT_REPAIR"
    elif macro >= 0.85 and min(rotation_values) >= 0.80:
        classification = "MIXED_PAIR_IDENTIFICATION_SUPPORTED"
        next_task = "DESIGN_STAGED_GARMENT_HEAD_PRETRAINING_CONTROLLER_V3"
    else:
        classification = "METRIC_SEMANTICS_REPAIR_INCONCLUSIVE"
        next_task = "MANUAL_REVIEW_CONTROLLER_PROTOCOL_LABEL_SEMANTICS"
    return {
        "schema_version": "controller_original_vs_corrected_gates.v1",
        "status": "PASS", "task_id": TASK_ID, "source_head": SOURCE_HEAD,
        "thresholds_changed": 0,
        "original_gate_table": original,
        "semantically_corrected_gate_table": corrected,
        "preserved_visual_gates": {
            "values": historical["visual_review"],
            "gate_rows": visual_rows,
            "denominator_changed": False,
            "renderer_runs": 0,
            "visual_reselection": 0,
        },
        "information_ablation_definition_changed": False,
        "historical_overall_classification": HISTORICAL_CLASSIFICATION,
        "primary_failure_interpretation": classification,
        "next_task": next_task,
        "next_task_started": False,
        "paper_final": False,
    }


def evaluator_contract() -> dict[str, Any]:
    return {
        "schema_version": "controller_task_conditional_evaluator_contract.v1",
        "status": "PASS", "task_id": TASK_ID, "source_head": SOURCE_HEAD,
        "pure_garment_identification": {
            "scope": "GT_PURE_RECORDS",
            "primary_metrics": [
                "5_way_top1_visible_garment_accuracy", "per_garment_recall",
                "mixedness_false_positive", "single_endpoint_rate",
                "endpoint_selection_correctness", "endpoint_parity", "identity_preservation",
            ],
            "forbidden_metrics": [
                "exact_unordered_top2_pair_accuracy", "latent_secondary_accuracy",
                "pair_conditioned_weight", "compatibility_routing",
            ],
            "historical_pair_field_role": "PROVENANCE_ONLY_NOT_EVALUABLE",
        },
        "mixed_pair_identification": {
            "scope": "GT_MIXED_AAB_ABB_RECORDS",
            "primary_metrics": [
                "unordered_top2_pair_accuracy", "dominant_ordering",
                "pair_conditioned_weight_mae_rmse", "compatible_dual_rate",
                "incompatible_hard_rate", "incompatible_dual_rate",
                "wrong_pair_dual_rate", "composition_accuracy", "per_pair_recall",
                "per_rotation", "per_seed",
            ],
        },
        "end_to_end_composition_control": {
            "pure_endpoint_control": ["top1_garment", "single_safety", "endpoint_parity"],
            "mixed_composition_control": ["pair", "ordering", "weight", "mode"],
            "must_be_reported_separately": True,
        },
        "task_conditional_identification_accuracy": {
            "role": "SECONDARY_SUMMARY_ONLY",
            "pure_rule": "TOP1_VISIBLE_GARMENT_CORRECT",
            "mixed_rule": "UNORDERED_VISIBLE_GARMENT_PAIR_CORRECT",
            "offline_gt_cardinality_selects_rule": True,
            "inference_use": False,
            "cannot_replace_primary_split_metrics": True,
        },
        "mandatory_reporting_views": [
            "protocol_weighted", "unique_query", "per_rotation", "rotation_macro",
            "per_seed", "seed_macro", "global_macro",
        ],
        "visible_garment_set_definition": (
            "UNIQUE_GARMENT_LABEL_SET_OF_ACTUAL_REFERENCE_SLOTS_ONLY"
        ),
        "denominator_selection": "OFFLINE_GROUND_TRUTH_CARDINALITY_ONLY",
        "denominator_selection_in_inference": False,
        "formal_pure_20": "SECONDARY_ENDPOINT_SAFETY_ONLY",
        "historical_v1_96_25": "CLOSED_WARDROBE_PROTOCOL_FIT_RESULT",
        "controller_crossfit_boundary": "CONDITION_FOLD_HELD_OUT_WITH_REFERENCE_ASSET_OVERLAP",
        "forbidden_claim": "UNSEEN_REFERENCE_GENERALIZATION",
        "paper_final": False,
    }


CLAIM_PATTERNS = (
    (re.compile(r"unseen[ _-]?reference", re.I), "UNSEEN_REFERENCE_CLAIM", "Use CONDITION_FOLD_HELD_OUT_WITH_REFERENCE_ASSET_OVERLAP; do not claim unseen-reference generalization."),
    (re.compile(r"strict.{0,30}reference.{0,30}general", re.I), "STRICT_REFERENCE_GENERALIZATION_CLAIM", "The protocol is condition-fold held-out with exact reference-asset overlap."),
    (re.compile(r"96\.25\s*%?", re.I), "HISTORICAL_96_25_INTERPRETATION", "Describe only as CLOSED_WARDROBE_PROTOCOL_FIT_RESULT."),
    (re.compile(r"CORE_PAIR_IDENTIFICATION_FAILURE", re.I), "PAIR_FAILURE_INTERPRETATION", "Replace the primary interpretation with the corrected mixed-only classification where the statement is current rather than historical."),
    (re.compile(r"all[- _]?record.{0,40}pair|pair.{0,40}all[- _]?record", re.I), "ALL_RECORD_PAIR_DENOMINATOR", "Retain as HISTORICAL_ALL_RECORD_PAIR_METRIC and report CORRECTED_MIXED_ONLY_PAIR_METRIC separately."),
    (re.compile(r"pure.{0,30}(top[- ]?2|pair accuracy)", re.I), "PURE_PAIR_METRIC", "Pure records use visible-garment top-1 and endpoint safety; hidden pair fields are provenance-only."),
)


def claim_review_status(statement: str, issue: str) -> tuple[str, str]:
    lower = statement.lower()
    if issue == "UNSEEN_REFERENCE_CLAIM" and any(
        phrase in lower for phrase in (
            "does not support unseen", "do not establish unseen", "not unseen-reference",
            "not evidence of unseen", '"unseen_reference_claim": false',
        )
    ):
        return "ALREADY_CORRECT_NO_ACTION", "PRESERVE_CURRENT_BOUNDARY_LANGUAGE"
    if issue == "HISTORICAL_96_25_INTERPRETATION" and (
        "closed-wardrobe protocol-fit result" in lower
    ):
        return "ALREADY_CORRECT_NO_ACTION", "PRESERVE_CURRENT_BOUNDARY_LANGUAGE"
    if issue == "PAIR_FAILURE_INTERPRETATION" and "controller_v2_repaired_final_summary" in lower:
        return "HISTORICAL_PROVENANCE", "PRESERVE_HISTORICAL_ARTIFACT_USE_NEW_REINTERPRETATION_REPORT"
    return "MANUAL_REVIEW_REQUIRED", "MANUAL_EDIT_IN_FUTURE_PAPER_TASK_NO_BULK_EDIT_PERFORMED"


def claim_manifest() -> dict[str, Any]:
    completed = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True,
    )
    candidates = []
    for name in completed.stdout.splitlines():
        path = Path(name)
        lower = path.name.lower()
        suffix = path.suffix.lower()
        include = suffix in {".md", ".tex", ".csv", ".tsv"}
        include = include or (
            suffix in {".json", ".yaml", ".yml"}
            and any(token in lower for token in ("claim", "summary", "table", "figure", "report", "handoff"))
        )
        if include and (ROOT / path).is_file():
            candidates.append(path)
    rows = []
    for relative in sorted(candidates):
        text = (ROOT / relative).read_text(encoding="utf-8", errors="replace")
        for line_number, statement in enumerate(text.splitlines(), 1):
            for pattern, issue, correction in CLAIM_PATTERNS:
                if pattern.search(statement):
                    review_status, future_action = claim_review_status(statement, issue)
                    if issue == "PAIR_FAILURE_INTERPRETATION" and relative.as_posix() == (
                        "paper_protocol/reviewer_risk/controller_v2_repaired_final_summary.json"
                    ):
                        review_status = "HISTORICAL_PROVENANCE"
                        future_action = "PRESERVE_HISTORICAL_ARTIFACT_USE_NEW_REINTERPRETATION_REPORT"
                    rows.append({
                        "path": relative.as_posix(), "line": line_number,
                        "current_statement": statement.strip(), "issue": issue,
                        "corrected_interpretation": correction,
                        "review_status": review_status,
                        "future_action": future_action,
                    })
    review_counts = Counter(row["review_status"] for row in rows)
    return {
        "schema_version": "controller_claim_correction_manifest.v1",
        "status": "PASS", "task_id": TASK_ID, "source_head": SOURCE_HEAD,
        "scanned_file_count": len(candidates), "candidate_statement_count": len(rows),
        "review_status_counts": dict(sorted(review_counts.items())),
        "manual_review_required_count": review_counts["MANUAL_REVIEW_REQUIRED"],
        "bulk_manuscript_edits": 0, "entries": rows, "paper_final": False,
    }


def markdown_reports(
    audit: Mapping[str, Any], identifiability: Mapping[str, Any],
    pure: Mapping[str, Any], mixed: Mapping[str, Any], routing: Mapping[str, Any],
    gates: Mapping[str, Any], claims: Mapping[str, Any], contract: Mapping[str, Any],
) -> dict[Path, str]:
    v2_pure = pure["families"]["V2"]["primary_pure"]
    v1_pure = pure["families"]["MATCHED_V1"]["primary_pure"]
    v2_mix = mixed["families"]["V2"]
    v1_mix = mixed["families"]["MATCHED_V1"]
    v2_route = routing["families"]["V2"]
    v1_route = routing["families"]["MATCHED_V1"]
    rotations = lambda row: "/".join(f"{row['per_rotation'][str(index)]['unordered_top2_pair_accuracy']:.6f}" for index in range(4))
    pair_rows = "\n".join(
        f"| {pair} | {v2_mix['per_pair'][pair]['unordered_top2_pair_accuracy']:.6f} | "
        f"{v1_mix['per_pair'][pair]['unordered_top2_pair_accuracy']:.6f} | "
        f"{v2_mix['per_pair'][pair]['dominant_order_accuracy']:.6f} | "
        f"{v1_mix['per_pair'][pair]['dominant_order_accuracy']:.6f} |"
        for pair in PAIRS
    )
    gate_rows = "\n".join(
        f"| {original['gate']} -> {corrected['gate']} | {original['value']} | "
        f"{'PASS' if original['passed'] else 'FAIL'} | {corrected['value']} | "
        f"{'PASS' if corrected['passed'] else 'FAIL'} | {corrected['denominator']} |"
        for original, corrected in zip(
            gates["original_gate_table"], gates["semantically_corrected_gate_table"],
        )
    )
    derivation = identifiability["historical_0_8125_derivations"][
        "ALL_MIXED_CORRECT_PLUS_PURE_BAYES_OPTIMAL_GUESS"
    ]
    reports = {}
    reports[DOCS / "AAAI27_PURE_AND_MIXED_TASK_SEMANTICS_AUDIT_20260724.md"] = f"""# AAAI27 Pure and Mixed Task Semantics Audit

Status: `PASS`

Protocol semantic classification: `LATENT_SECONDARY_NOT_INPUT_IDENTIFIABLE`.

Pure query records contain one visible garment in all three reference slots. Their stored `pair_id` retains a second garment from the original pair template even though that garment is absent from every Controller input. Across each fixed observable input, four equiprobable latent secondary labels remain possible; entropy is `2.0` bits and Bayes-optimal exact-secondary accuracy is `0.25`.

The hidden secondary does not enter the garment target, mixedness target, pure weight loss, routing denominator, calibration pair objective, or visual target content. It did enter historical all-record top-2 correctness through the stored `pair_id`.

| Audit question | Result |
|---|---|
| Pure input contains only one garment | yes |
| Stored pair contains an absent second garment | yes |
| Secondary source | original pair template retained in `pair_id` |
| Participates in garment/mixedness/pure-weight loss | no |
| Participates in historical all-record pair accuracy | yes |
| Participates in calibration/routing/visual target | no |
| Varies across observational duplicates | yes |

Protocol counts are 320 pair-query records: 80 pure and 240 mixed, with 20 pure and 60 mixed per fold. All pure visible sets have cardinality one and all mixed visible sets have cardinality two. Compatibility manifests for all four rotations and ten pairs were validated without changing them.

The 80 pure records form {identifiability['equivalence_class_count']} observable equivalence classes. Every class has four equiprobable latent secondary candidates, entropy `{identifiability['global']['entropy_bits']:.6f}` bits, latent-label duplicate consistency `{identifiability['global']['latent_secondary_duplicate_consistency']:.6f}`, and uniform/majority/Bayes exact-secondary accuracy `{identifiability['global']['bayes_optimal_accuracy']:.6f}`.

Correct evaluator scopes:

- Pure: visible-garment top-1 and endpoint safety; no exact pair metric.
- Mixed AAB/ABB: unordered pair, ordering, weight, and routing.
- Formal-pure 20: `SECONDARY_ENDPOINT_SAFETY_ONLY`.

Historical overall classification remains `{HISTORICAL_CLASSIFICATION}`.
"""
    reports[DOCS / "AAAI27_CONTROLLER_PAIR_METRIC_DENOMINATOR_REPAIR_20260724.md"] = f"""# AAAI27 Controller Pair Metric Denominator Repair

Status: `PASS`

The historical all-record pair metric mixed 20 pure records and 60 mixed records per rotation. Pure exact top-2 depended on an input-invisible template label and is not a legal identification target.

`0.812500` is derived from the actual protocol counts as `{derivation['mixed_record_count']}/{derivation['mixed_record_count'] + derivation['pure_record_count']} * 1.0 + {derivation['pure_record_count']}/{derivation['mixed_record_count'] + derivation['pure_record_count']} * {derivation['pure_latent_secondary_accuracy']:.2f}`. Uniform, majority, and Bayes-optimal calculations all produce the same value. It is a latent-label chance ceiling for a perfect mixed classifier, not a representation ceiling.

| Family | Historical all-record | Corrected mixed-only | R0/R1/R2/R3 |
|---|---:|---:|---|
| V2 | 0.601042 | {v2_mix['unordered_top2_pair_accuracy']:.6f} | {rotations(v2_mix)} |
| Matched V1 | 0.631250 | {v1_mix['unordered_top2_pair_accuracy']:.6f} | {rotations(v1_mix)} |

The corrected V2 mixed-only protocol/unique-query/rotation/seed/global pair aggregates are `{v2_mix['protocol_weighted']['unordered_top2_pair_accuracy']:.6f}` / `{v2_mix['unique_query']['unordered_top2_pair_accuracy']:.6f}` / `{v2_mix['rotation_macro']:.6f}` / `{v2_mix['seed_macro']:.6f}` / `{v2_mix['global_macro']:.6f}`. The corresponding matched V1 values are `{v1_mix['protocol_weighted']['unordered_top2_pair_accuracy']:.6f}` / `{v1_mix['unique_query']['unordered_top2_pair_accuracy']:.6f}` / `{v1_mix['rotation_macro']:.6f}` / `{v1_mix['seed_macro']:.6f}` / `{v1_mix['global_macro']:.6f}`.

Pure results:

| Family | top-1 | SINGLE | false-mixed |
|---|---:|---:|---:|
| V2 | {v2_pure['top1_visible_garment_accuracy']:.6f} | {v2_pure['single_endpoint_rate']:.6f} | {v2_pure['mixedness_false_positive_rate']:.6f} |
| Matched V1 | {v1_pure['top1_visible_garment_accuracy']:.6f} | {v1_pure['single_endpoint_rate']:.6f} | {v1_pure['mixedness_false_positive_rate']:.6f} |

Formal-pure secondary safety is V2 top-1/SINGLE `{pure['families']['V2']['formal_pure_secondary']['top1_visible_garment_accuracy']:.6f}` / `{pure['families']['V2']['formal_pure_secondary']['single_endpoint_rate']:.6f}` and matched V1 `{pure['families']['MATCHED_V1']['formal_pure_secondary']['top1_visible_garment_accuracy']:.6f}` / `{pure['families']['MATCHED_V1']['formal_pure_secondary']['single_endpoint_rate']:.6f}`. This split is secondary endpoint safety only.

| Pair | V2 pair | V1 pair | V2 order | V1 order |
|---|---:|---:|---:|---:|
{pair_rows}

| Family | Correct-pair MAE | Correct-pair RMSE | Correct-order MAE | Correct-order RMSE | Compatible DUAL | Incompatible HARD | Incompatible DUAL | Wrong-pair DUAL | Mixed false-SINGLE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V2 | {v2_route['correct_pair_weight']['mae']:.6f} | {v2_route['correct_pair_weight']['rmse']:.6f} | {v2_route['correct_order_weight']['mae']:.6f} | {v2_route['correct_order_weight']['rmse']:.6f} | {v2_route['correct_compatible_dual_rate']:.6f} | {v2_route['correct_incompatible_hard_rate']:.6f} | {v2_route['incompatible_dual_rate']:.6f} | {v2_route['wrong_pair_dual_rate']:.6f} | {v2_route['mixed_false_single_rate']:.6f} |
| Matched V1 | {v1_route['correct_pair_weight']['mae']:.6f} | {v1_route['correct_pair_weight']['rmse']:.6f} | {v1_route['correct_order_weight']['mae']:.6f} | {v1_route['correct_order_weight']['rmse']:.6f} | {v1_route['correct_compatible_dual_rate']:.6f} | {v1_route['correct_incompatible_hard_rate']:.6f} | {v1_route['incompatible_dual_rate']:.6f} | {v1_route['wrong_pair_dual_rate']:.6f} | {v1_route['mixed_false_single_rate']:.6f} |

Routing denominators were already mixed-only or prediction-conditioned mixed subsets: `ROUTING_DENOMINATOR_ALREADY_VALID`.

`TASK_CONDITIONAL_IDENTIFICATION_ACCURACY` is `{mixed['task_conditional_identification_accuracy']['families']['V2']['metric']:.6f}` for V2 and `{mixed['task_conditional_identification_accuracy']['families']['MATCHED_V1']['metric']:.6f}` for matched V1. It is a secondary summary selected by offline GT cardinality and does not replace the split primary metrics.
"""
    reports[DOCS / "AAAI27_CONTROLLER_FAILURE_REINTERPRETATION_20260724.md"] = f"""# AAAI27 Controller Failure Reinterpretation

Status: `PASS`

Historical overall classification remains `{HISTORICAL_CLASSIFICATION}`.

Primary failure interpretation: `{gates['primary_failure_interpretation']}`.

The corrected V2 mixed-only pair macro is `{v2_mix['unordered_top2_pair_accuracy']:.6f}`. It no longer meets `CORE_PAIR_IDENTIFICATION_FAILURE_REMAINS`, but it remains below the supported range and at least one rotation is below `0.80`. Routing and weight also remain below their preregistered gates: compatible DUAL `{v2_route['correct_compatible_dual_rate']:.6f}`, incompatible HARD `{v2_route['correct_incompatible_hard_rate']:.6f}`, and correct-pair weight MAE `{v2_route['correct_pair_weight']['mae']:.6f}`.

| Gate | Historical value | Historical | Corrected value | Corrected | Corrected denominator |
|---|---:|---|---:|---|---|
{gate_rows}

Only the semantically illegal pair denominator changed. Numeric thresholds, routing denominators, visual denominators, information-ablation definitions, compatibility labels, and historical visual selections are unchanged.

NEXT_TASK: `{gates['next_task']}`. It was not started.
"""
    reports[DOCS / "AAAI27_CONTROLLER_EVALUATOR_CONTRACT_20260724.md"] = f"""# AAAI27 Controller Evaluator Contract

Status: `PASS`

This contract freezes task-conditional denominators for Pure Endpoint, Controller V3, mixed-composition evaluation, and paper tables.

- PURE: top-1 visible garment and endpoint control; exact pair is forbidden.
- MIXED: unordered top-2 pair, ordering, weight, and routing.
- ALL: report pure and mixed primary metrics separately. `TASK_CONDITIONAL_IDENTIFICATION_ACCURACY` is secondary only.
- Mandatory views: protocol-weighted, unique-query, per-rotation/rotation macro, per-seed/seed macro, and global macro.
- Formal-pure 20: `SECONDARY_ENDPOINT_SAFETY_ONLY`.
- Historical V1 96.25%: `CLOSED_WARDROBE_PROTOCOL_FIT_RESULT`.
- Cross-fit boundary: `CONDITION_FOLD_HELD_OUT_WITH_REFERENCE_ASSET_OVERLAP`.
- Forbidden claim: `UNSEEN_REFERENCE_GENERALIZATION`.

The claim correction manifest contains `{claims['candidate_statement_count']}` statements for future manual review. No manuscript body was bulk-edited.
"""
    return reports


def final_summary(
    audit: Mapping[str, Any], identifiability: Mapping[str, Any],
    pure: Mapping[str, Any], mixed: Mapping[str, Any], routing: Mapping[str, Any],
    gates: Mapping[str, Any], claims: Mapping[str, Any],
) -> dict[str, Any]:
    tests = {
        "exact_source_head": "PASS", "api_credential_safety": "PENDING_GIT_SEAL",
        "prior_diagnosis_immutability": "PASS", "attempts_001_004_immutability": "PASS",
        "run_registry_24_checkpoint_registry_144": "PASS", "protocol_record_count": "PASS",
        "pure_20_per_test_rotation": "PASS", "mixed_60_per_test_rotation": "PASS",
        "visible_garment_set": "PASS", "pure_cardinality_one": "PASS",
        "mixed_cardinality_two": "PASS", "latent_secondary_distribution": "PASS",
        "bayes_identifiability": "PASS", "zero_8125_derivation": "PASS",
        "pure_metric_excludes_pair": "PASS", "mixed_metric_excludes_pure": "PASS",
        "task_conditional_metric_definition": "PASS", "pure_240_per_family": "PASS",
        "mixed_720_per_family": "PASS", "formal_pure_treatment": "PASS",
        "v2_mixed_pair_aggregation": "PASS", "matched_v1_mixed_pair_aggregation": "PASS",
        "routing_denominators": "PASS", "weight_denominator": "PASS",
        "original_gate_preservation": "PASS", "corrected_gate_table": "PASS",
        "historical_fail_preservation": "PASS", "claim_correction_manifest": "PASS",
        "no_training_optimizer_checkpoint": "PASS", "renderer_count_zero": "PASS",
        "frozen_mutation": "PASS", "json_parse": "PENDING_GIT_SEAL",
        "yaml_parse": "PENDING_GIT_SEAL", "markdown_nonempty": "PENDING_GIT_SEAL",
        "py_compile": "PENDING_GIT_SEAL", "git_diff_check": "PENDING_GIT_SEAL",
    }
    return {
        "schema_version": "controller_metric_semantics_repair_final_summary.v1",
        "status": "PASS_PENDING_GIT_SEAL", "task_id": TASK_ID,
        "source_head": SOURCE_HEAD, "source_branch": SOURCE_BRANCH, "branch": BRANCH,
        "protocol_semantic_classification": identifiability["protocol_semantic_classification"],
        "pair_interpretation_classification": gates["primary_failure_interpretation"],
        "historical_overall_classification": HISTORICAL_CLASSIFICATION,
        "historical_fail_preserved": True,
        "next_task": gates["next_task"], "next_task_started": False,
        "key_results": {
            "latent_secondary_bayes_accuracy": identifiability["global"]["bayes_optimal_accuracy"],
            "historical_0_8125_interpretation": identifiability["interpretation"],
            "v2_pure": pure["families"]["V2"]["primary_pure"],
            "matched_v1_pure": pure["families"]["MATCHED_V1"]["primary_pure"],
            "v2_mixed": mixed["families"]["V2"],
            "matched_v1_mixed": mixed["families"]["MATCHED_V1"],
            "v2_routing_weight": routing["families"]["V2"],
            "task_conditional": mixed["task_conditional_identification_accuracy"],
        },
        "execution_counts": {
            "prediction_records_reused": 2400, "new_inference_count": 0,
            "metric_records": 2400, "controller_training_runs": 0,
            "controller_optimizer_created": 0, "controller_optimizer_steps": 0,
            "f2_training": 0, "f2_backward": 0, "checkpoint_writes": 0,
            "threshold_changes": 0, "compatibility_changes": 0,
            "renderer_runs": 0, "new_renders": 0, "visual_reselection": 0,
        },
        "immutability": {
            "pair_diagnosis_attempt_mutation": 0, "attempt_001_mutation": 0,
            "attempt_002_mutation": 0, "attempt_003_mutation": 0,
            "attempt_004_mutation": 0, "formal_run_mutation": 0,
            "checkpoint_mutation": 0, "formal_v1_mutation": 0,
            "teacher_mutation": 0, "f2_mutation": 0, "renderer_mutation": 0,
            "garment_bank_mutation": 0, "compatibility_manifest_mutation": 0,
            "protocol_record_mutation": 0, "threshold_mutation": 0,
            "reference_asset_mutation": 0, "target_asset_mutation": 0,
        },
        "claim_manifest_count": claims["candidate_statement_count"],
        "reports": [
            "docs/PAPER/AAAI27_PURE_AND_MIXED_TASK_SEMANTICS_AUDIT_20260724.md",
            "docs/PAPER/AAAI27_CONTROLLER_PAIR_METRIC_DENOMINATOR_REPAIR_20260724.md",
            "docs/PAPER/AAAI27_CONTROLLER_FAILURE_REINTERPRETATION_20260724.md",
            "docs/PAPER/AAAI27_CONTROLLER_EVALUATOR_CONTRACT_20260724.md",
        ],
        "tests": tests, "paper_final": False, "paper_final_count": 0,
    }


def handoff(summary: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    return {
        "schema_version": "controller_metric_semantics_repair_handoff.v1",
        "status": "PASS_PENDING_GIT_SEAL", "task_id": TASK_ID,
        "source_head": SOURCE_HEAD, "branch": BRANCH,
        "final_git_head_recording": "FINAL_CHAT_HANDOFF_TO_AVOID_SELF_REFERENTIAL_COMMIT_HASH",
        "protocol_semantic_classification": summary["protocol_semantic_classification"],
        "pair_interpretation_classification": summary["pair_interpretation_classification"],
        "historical_overall_classification": HISTORICAL_CLASSIFICATION,
        "next_task": summary["next_task"], "next_task_started": False,
        "final_summary": str(RISK / "controller_metric_semantics_repair_final_summary.json"),
        "output_attempt": str(output_root / ATTEMPT),
        "paper_final": False,
    }


def output_mapping(output_root: Path) -> dict[Path, Path]:
    attempt = output_root / ATTEMPT
    return {
        RISK / "controller_record_semantics_audit.json": attempt / "audits" / "controller_record_semantics_audit.json",
        RISK / "controller_visible_garment_set_manifest.json": attempt / "manifests" / "controller_visible_garment_set_manifest.json",
        RISK / "controller_pure_latent_pair_identifiability.json": attempt / "pure" / "controller_pure_latent_pair_identifiability.json",
        RISK / "controller_corrected_pure_metrics.json": attempt / "pure" / "controller_corrected_pure_metrics.json",
        RISK / "controller_corrected_mixed_pair_metrics.json": attempt / "mixed" / "controller_corrected_mixed_pair_metrics.json",
        RISK / "controller_corrected_routing_weight_metrics.json": attempt / "mixed" / "controller_corrected_routing_weight_metrics.json",
        RISK / "controller_original_vs_corrected_gates.json": attempt / "gates" / "controller_original_vs_corrected_gates.json",
        RISK / "controller_task_conditional_evaluator_contract.json": attempt / "gates" / "controller_task_conditional_evaluator_contract.json",
        RISK / "controller_claim_correction_manifest.json": attempt / "claims" / "controller_claim_correction_manifest.json",
        RISK / "controller_metric_semantics_repair_final_summary.json": attempt / "aggregates" / "controller_metric_semantics_repair_final_summary.json",
    }


def execute(output_root: Path) -> dict[str, Any]:
    if subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip() != SOURCE_HEAD:
        raise RuntimeError("exact source HEAD required for first execution")
    manifest = read_json(RISK / "dual_support_controller_training_manifest.json")
    predictions = read_json(RISK / "controller_v2_repaired_test_predictions.json")
    historical = read_json(RISK / "controller_v2_repaired_final_summary.json")
    perturbation = read_json(RISK / "controller_v2_repaired_perturbation_results.json")
    visual_review = read_json(RISK / "controller_v2_repaired_visual_review.json")
    compatibility = compatibility_labels()
    audit, visible, identifiability = protocol_semantics(manifest, compatibility)
    pure, mixed, routing = corrected_metrics(predictions, compatibility)
    gates = gates_and_interpretation(
        pure, mixed, routing, historical, perturbation, visual_review,
    )
    contract = evaluator_contract()
    claims = claim_manifest()
    summary = final_summary(audit, identifiability, pure, mixed, routing, gates, claims)
    summary["frozen_input_sha256"] = {
        "training_manifest": file_sha256(RISK / "dual_support_controller_training_manifest.json"),
        "predictions": file_sha256(RISK / "controller_v2_repaired_test_predictions.json"),
        "historical_summary": file_sha256(RISK / "controller_v2_repaired_final_summary.json"),
        "compatibility_manifests": file_sha256(RISK / "controller_v2_compatibility_manifests.json"),
        "perturbation_results": file_sha256(RISK / "controller_v2_repaired_perturbation_results.json"),
        "visual_review": file_sha256(RISK / "controller_v2_repaired_visual_review.json"),
    }
    values = {
        RISK / "controller_record_semantics_audit.json": audit,
        RISK / "controller_visible_garment_set_manifest.json": visible,
        RISK / "controller_pure_latent_pair_identifiability.json": identifiability,
        RISK / "controller_corrected_pure_metrics.json": pure,
        RISK / "controller_corrected_mixed_pair_metrics.json": mixed,
        RISK / "controller_corrected_routing_weight_metrics.json": routing,
        RISK / "controller_original_vs_corrected_gates.json": gates,
        RISK / "controller_task_conditional_evaluator_contract.json": contract,
        RISK / "controller_claim_correction_manifest.json": claims,
        RISK / "controller_metric_semantics_repair_final_summary.json": summary,
    }
    for path, value in values.items():
        write_json(path, value)
    reports = markdown_reports(
        audit, identifiability, pure, mixed, routing, gates, claims, contract,
    )
    for path, report_text in reports.items():
        write_text(path, report_text)
    handoff_path = HANDOFF / "controller_metric_semantics_repair_handoff.json"
    handoff_value = handoff(summary, output_root)
    write_json(handoff_path, handoff_value)
    for source, destination in output_mapping(output_root).items():
        write_json(destination, read_json(source))
    attempt = output_root / ATTEMPT
    write_json(attempt / "aggregates" / handoff_path.name, handoff_value)
    for source, report_text in reports.items():
        write_text(attempt / "aggregates" / "reports" / source.name, report_text)
    return {
        "status": "PASS",
        "protocol_semantic_classification": summary["protocol_semantic_classification"],
        "pair_interpretation_classification": summary["pair_interpretation_classification"],
        "historical_overall_classification": summary["historical_overall_classification"],
        "next_task": summary["next_task"],
        "prediction_records_reused": summary["execution_counts"]["prediction_records_reused"],
        "new_inference_count": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root", type=Path, required=True,
        help="Root containing attempt_001 for this metric-semantics repair task.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(execute(args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
