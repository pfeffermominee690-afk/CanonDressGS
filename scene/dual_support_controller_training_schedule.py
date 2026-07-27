"""Pure-data construction of the frozen controller training schedule.

This module intentionally imports only the Python standard library.  It does
not construct a controller, optimizer, renderer, or training batch tensor.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


OUTFIT_ORDER = ("O01", "O02", "O03", "O04", "O08")
PAIR_ORDER = (
    "O01_O02", "O01_O03", "O01_O04", "O01_O08", "O02_O03",
    "O02_O04", "O02_O08", "O03_O04", "O03_O08", "O04_O08",
)
FOLD_ORDER = ("cond_000000", "cond_000318", "cond_000017", "cond_000347")
ASSIGNMENT_POSITION_ORDER = (0, 1, 2)
CYCLE_BATCH_COUNT = 64
TOTAL_STEPS = 300


@dataclass(frozen=True)
class ScheduleBundle:
    scientific_cycle: dict[str, Any]
    scientific_schedule: dict[str, Any]
    cycle_sha256: str
    schedule_sha256: str
    summary: dict[str, Any]


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical UTF-8 JSON plus one LF; absolute paths are caller-excluded."""
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _pair_members(pair_id: str) -> tuple[str, str]:
    if pair_id not in PAIR_ORDER:
        raise ValueError(f"unknown frozen pair: {pair_id}")
    left, right = pair_id.split("_", 1)
    return left, right


def _dominant_outfit(target: Sequence[float]) -> str:
    if len(target) != len(OUTFIT_ORDER):
        raise ValueError("TRAINING_MANIFEST_DOMINANT_LABEL_INVALID: target length")
    values = [float(value) for value in target]
    if any(value < 0.0 for value in values) or abs(sum(values) - 1.0) > 1.0e-8:
        raise ValueError("TRAINING_MANIFEST_DOMINANT_LABEL_INVALID: target distribution")
    maximum = max(values)
    winners = [index for index, value in enumerate(values) if value == maximum]
    if len(winners) != 1:
        raise ValueError("TRAINING_MANIFEST_DOMINANT_LABEL_INVALID: dominant tie")
    if maximum not in (1.0, 2.0 / 3.0):
        raise ValueError("TRAINING_MANIFEST_DOMINANT_LABEL_INVALID: dominant mass")
    return OUTFIT_ORDER[winners[0]]


def _composition_key(record: Mapping[str, Any], dominant: str) -> tuple[int, str]:
    left, right = _pair_members(str(record["pair_id"]))
    kind = str(record["assignment_type"])
    position = record["assignment_position"]
    if dominant == left:
        if kind == "AAA" and position is None:
            return 0, "PURE_DOMINANT"
        if kind == "AAB" and position in ASSIGNMENT_POSITION_ORDER:
            return int(position) + 1, f"MIXED_ASSIGNMENT_POSITION_{position}"
    elif dominant == right:
        if kind == "BBB" and position is None:
            return 0, "PURE_DOMINANT"
        if kind == "ABB" and position in ASSIGNMENT_POSITION_ORDER:
            return int(position) + 1, f"MIXED_ASSIGNMENT_POSITION_{position}"
    raise ValueError(
        f"record does not match dominant composition contract: {record.get('record_id')}"
    )


def _counterpart(pair_id: str, dominant: str) -> str:
    left, right = _pair_members(pair_id)
    if dominant == left:
        return right
    if dominant == right:
        return left
    raise ValueError("dominant outfit is absent from pair")


def _scientific_record(
    record: Mapping[str, Any], *, batch_position: int, dominant: str,
    composition: str,
) -> dict[str, Any]:
    return {
        "batch_position": batch_position,
        "dominant_outfit": dominant,
        "record_id": str(record["record_id"]),
        "pair_id": str(record["pair_id"]),
        "composition": composition,
        "assignment_position": record["assignment_position"],
        "target_distribution": [float(value) for value in record["target_distribution"]],
    }


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    counts = manifest.get("counts", {})
    if counts.get("pair_fold_query_sets") != 320:
        raise ValueError("training scope must contain exactly 320 protocol records")
    if counts.get("retained_formal_pure_endpoint_episodes") != 20:
        raise ValueError("formal pure scope must contain exactly 20 records")
    if tuple(manifest.get("frozen_outfit_order", ())) != OUTFIT_ORDER:
        raise ValueError("frozen outfit order mismatch")
    if tuple(manifest.get("frozen_target_view_order", ())) != FOLD_ORDER:
        raise ValueError("frozen target-view fold order mismatch")
    if tuple(manifest.get("pairs", ())) != PAIR_ORDER:
        raise ValueError("frozen pair order mismatch")
    queries = manifest.get("query_sets", [])
    formal = manifest.get("formal_pure_endpoint_episodes", [])
    if len(queries) != 320 or len(formal) != 20:
        raise ValueError("manifest record lists do not match frozen counts")
    ids = [row.get("record_id") for row in queries + formal]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest record IDs are not unique")
    if not manifest.get("duplicate_detection", {}).get("pass"):
        raise ValueError("manifest duplicate-label consistency failed")
    if manifest.get("duplicate_detection", {}).get("duplicate_records") != 80:
        raise ValueError("consistent duplicate multiplicity mismatch")


def _build_queues(
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, list[tuple[Mapping[str, Any], str]]]]:
    buckets: dict[tuple[str, str, str], list[tuple[int, str, Mapping[str, Any]]]] = defaultdict(list)
    for record in manifest["query_sets"]:
        dominant = _dominant_outfit(record["target_distribution"])
        fold = str(record["target_view_fold"])
        if fold not in FOLD_ORDER:
            raise ValueError("query record uses an unknown fold")
        pair_id = str(record["pair_id"])
        counterpart = _counterpart(pair_id, dominant)
        order_index, composition = _composition_key(record, dominant)
        buckets[(dominant, fold, counterpart)].append((order_index, composition, record))

    queues: dict[str, dict[str, list[tuple[Mapping[str, Any], str]]]] = {}
    for dominant in OUTFIT_ORDER:
        counterparts = tuple(outfit for outfit in OUTFIT_ORDER if outfit != dominant)
        queues[dominant] = {}
        for fold in FOLD_ORDER:
            queue: list[tuple[Mapping[str, Any], str]] = []
            for counterpart in counterparts:
                rows = sorted(buckets[(dominant, fold, counterpart)], key=lambda row: row[0])
                if [row[0] for row in rows] != [0, 1, 2, 3]:
                    raise ValueError(
                        f"dominant queue composition incomplete: {dominant}/{fold}/{counterpart}"
                    )
                queue.extend((record, composition) for _, composition, record in rows)
            if len(queue) != 16:
                raise ValueError("dominant/fold queue must contain exactly 16 records")
            queues[dominant][fold] = queue
    if any(
        sum(len(queues[outfit][fold]) for fold in FOLD_ORDER) != 64
        for outfit in OUTFIT_ORDER
    ):
        raise AssertionError("dominant queue must contain exactly 64 records")
    return queues


def build_schedule(manifest: Mapping[str, Any]) -> ScheduleBundle:
    validate_manifest(manifest)
    queues = _build_queues(manifest)
    batches = []
    cycle_record_ids = []
    for slot in range(16):
        for fold_index, fold in enumerate(FOLD_ORDER):
            batch_index = 4 * slot + fold_index
            records = []
            for batch_position, dominant in enumerate(OUTFIT_ORDER):
                record, composition = queues[dominant][fold][slot]
                records.append(
                    _scientific_record(
                        record,
                        batch_position=batch_position,
                        dominant=dominant,
                        composition=composition,
                    )
                )
                cycle_record_ids.append(str(record["record_id"]))
            batches.append(
                {
                    "batch_index": batch_index,
                    "slot": slot,
                    "target_view_fold_index": fold_index,
                    "target_view_fold": fold,
                    "records": records,
                }
            )
    if len(batches) != CYCLE_BATCH_COUNT:
        raise RuntimeError("cycle must contain exactly 64 batches")
    query_ids = {str(row["record_id"]) for row in manifest["query_sets"]}
    if len(cycle_record_ids) != 320 or set(cycle_record_ids) != query_ids:
        raise RuntimeError("cycle does not cover the 320 protocol records exactly once")
    if len(cycle_record_ids) != len(set(cycle_record_ids)):
        raise RuntimeError("cycle repeats a protocol record")

    scientific_cycle = {
        "schema_version": "canondressgs.research.dual_support_controller_training_cycle_scientific.v1",
        "outfit_order": list(OUTFIT_ORDER),
        "pair_order": list(PAIR_ORDER),
        "target_view_fold_order": list(FOLD_ORDER),
        "batch_size": 5,
        "batch_count": CYCLE_BATCH_COUNT,
        "batches": batches,
    }
    cycle_sha = canonical_sha256(scientific_cycle)
    steps = []
    exposure = Counter()
    for global_step in range(1, TOTAL_STEPS + 1):
        cycle_index = (global_step - 1) // CYCLE_BATCH_COUNT
        batch_index = (global_step - 1) % CYCLE_BATCH_COUNT
        batch = batches[batch_index]
        records = [dict(record) for record in batch["records"]]
        exposure.update(record["record_id"] for record in records)
        steps.append(
            {
                "global_step": global_step,
                "cycle_index": cycle_index,
                "batch_index": batch_index,
                "target_view_fold_index": batch["target_view_fold_index"],
                "target_view_fold": batch["target_view_fold"],
                "records": records,
            }
        )
    scientific_schedule = {
        "schema_version": "canondressgs.research.dual_support_controller_training_schedule_scientific.v1",
        "training_cycle_sha256": cycle_sha,
        "total_steps": TOTAL_STEPS,
        "steps": steps,
    }
    schedule_sha = canonical_sha256(scientific_schedule)

    record_index = {str(row["record_id"]): row for row in manifest["query_sets"]}
    unique_exposure: Counter[str] = Counter()
    duplicate_groups: dict[str, list[str]] = defaultdict(list)
    for record_id, count in exposure.items():
        logical = str(record_index[record_id]["logical_input_sha256"])
        unique_exposure[logical] += count
        duplicate_groups[logical].append(record_id)
    duplicate_group_exposure = {
        logical: {
            "record_ids": sorted(record_ids),
            "protocol_record_count": len(record_ids),
            "total_exposure": sum(exposure[record_id] for record_id in record_ids),
            "per_record_exposure": {
                record_id: exposure[record_id] for record_id in sorted(record_ids)
            },
        }
        for logical, record_ids in sorted(duplicate_groups.items())
        if len(record_ids) > 1
    }
    formal_ids = {str(row["record_id"]) for row in manifest["formal_pure_endpoint_episodes"]}
    formal_logical = {
        str(row["logical_input_sha256"])
        for row in manifest["formal_pure_endpoint_episodes"]
    }
    training_logical = {
        str(row["logical_input_sha256"]) for row in manifest["query_sets"]
    }
    summary = {
        "schema_version": "canondressgs.research.dual_support_controller_training_schedule_summary.v1",
        "training_cycle_sha256": cycle_sha,
        "training_300_step_data_order_sha256": schedule_sha,
        "counts": {
            "training_protocol_records": 320,
            "evaluation_only_formal_pure_records": 20,
            "dominant_queues": 5,
            "records_per_dominant_queue": 64,
            "folds": 4,
            "records_per_dominant_fold": 16,
            "batches_per_cycle": 64,
            "records_per_batch": 5,
            "records_per_cycle": 320,
            "training_steps": 300,
            "sample_exposures": sum(exposure.values()),
            "formal_pure_training_exposures": sum(exposure[record_id] for record_id in formal_ids),
            "consistent_duplicate_protocol_records": 80,
            "protocol_unique_logical_queries": len(unique_exposure),
            "duplicate_logical_groups": len(duplicate_group_exposure),
            "label_conflicts": 0,
        },
        "wrap_rule": {
            "complete_cycles": 4,
            "partial_cycle_batch_count": 44,
            "batch_indices_with_five_exposures": [0, 43],
            "batch_indices_with_four_exposures": [44, 63],
            "protocol_records_with_five_exposures": sum(count == 5 for count in exposure.values()),
            "protocol_records_with_four_exposures": sum(count == 4 for count in exposure.values()),
        },
        "seed_data_order": {
            "seed_0": schedule_sha,
            "seed_1": schedule_sha,
            "seed_2": schedule_sha,
            "unique_hash_count": 1,
        },
        "formal_pure_isolation": {
            "role": "FORMAL_PURE_EVAL_ONLY",
            "record_level_evaluation_isolation": formal_ids.isdisjoint(exposure),
            "training_exposure": 0,
            "optimizer_target_use": 0,
            "image_reference_asset_overlap": "PRESENT_CLOSED_WARDROBE",
            "formal_logical_inputs_overlapping_protocol_training": len(formal_logical & training_logical),
            "claim_of_image_level_independence": False,
        },
        "protocol_record_exposure": dict(sorted(exposure.items())),
        "unique_query_exposure": dict(sorted(unique_exposure.items())),
        "duplicate_group_exposure": duplicate_group_exposure,
        "execution_counts": {
            "training_steps_executed": 0,
            "forward_training_batches": 0,
            "backward_calls": 0,
            "optimizer_created": 0,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "checkpoint_writes": 0,
            "formal_renders": 0,
            "formal_metrics": 0,
            "formal_visual_reviews": 0,
        },
        "paper_final": False,
        "paper_final_count": 0,
    }
    if summary["counts"]["sample_exposures"] != 1500:
        raise RuntimeError("300-step schedule exposure mismatch")
    if summary["counts"]["formal_pure_training_exposures"] != 0:
        raise RuntimeError("formal-pure record leaked into training schedule")
    if summary["wrap_rule"]["protocol_records_with_five_exposures"] != 220:
        raise RuntimeError("partial-cycle five-exposure count mismatch")
    if summary["wrap_rule"]["protocol_records_with_four_exposures"] != 100:
        raise RuntimeError("partial-cycle four-exposure count mismatch")
    return ScheduleBundle(
        scientific_cycle=scientific_cycle,
        scientific_schedule=scientific_schedule,
        cycle_sha256=cycle_sha,
        schedule_sha256=schedule_sha,
        summary=summary,
    )


def step_lookup(bundle: ScheduleBundle, global_step: int) -> Mapping[str, Any]:
    if isinstance(global_step, bool) or not isinstance(global_step, int):
        raise TypeError("global_step must be an integer")
    if global_step < 1 or global_step > TOTAL_STEPS:
        raise IndexError("global_step is outside 1..300")
    return bundle.scientific_schedule["steps"][global_step - 1]
