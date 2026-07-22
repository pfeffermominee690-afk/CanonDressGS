"""Build the frozen 320-query soft-target controller training manifest."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


OUTFITS = ("O01", "O02", "O03", "O04", "O08")
CONDITIONS = ("cond_000000", "cond_000318", "cond_000017", "cond_000347")
PAIRS = tuple(itertools.combinations(OUTFITS, 2))
LABEL = "RESEARCH METHOD CANDIDATE — NOT PAPER FINAL"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_index(source: Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    index: dict[str, dict[str, dict[str, Any]]] = {}
    for outfit in source.get("outfits", []):
        name = outfit.get("outfit_id")
        if name not in OUTFITS:
            continue
        observations = {}
        for row in outfit.get("observations", []):
            condition = row.get("condition_id")
            if condition not in CONDITIONS:
                continue
            checksums = row.get("checksums", {})
            rgb_hash = checksums.get("target_edit_rgb")
            mask_hash = checksums.get("target_clothing_mask")
            rgb_path = row.get("target_edit_rgb", row.get("rgb"))
            mask_path = row.get("target_clothing_mask", row.get("clothing_mask"))
            if not all(isinstance(value, str) and value for value in (rgb_hash, mask_hash, rgb_path, mask_path)):
                raise ValueError(f"source observation is incomplete: {name}/{condition}")
            observations[condition] = {
                "outfit_id": name,
                "condition_id": condition,
                "image_path": rgb_path,
                "image_sha256": rgb_hash,
                "clothing_mask_path": mask_path,
                "clothing_mask_sha256": mask_hash,
            }
        index[name] = observations
    if tuple(name for name in OUTFITS if name in index) != OUTFITS:
        raise ValueError("source manifest does not contain the frozen five outfits")
    if any(tuple(condition for condition in CONDITIONS if condition in index[name]) != CONDITIONS for name in OUTFITS):
        raise ValueError("source manifest does not contain all frozen target-view conditions")
    return index


def target_distribution(labels: Sequence[str]) -> list[float]:
    if len(labels) != 3 or any(label not in OUTFITS for label in labels):
        raise ValueError("reference garment labels must have length three")
    return [labels.count(outfit) / 3.0 for outfit in OUTFITS]


def assignments(left: str, right: str) -> list[tuple[str, int | None, tuple[str, str, str]]]:
    result: list[tuple[str, int | None, tuple[str, str, str]]] = [
        ("AAA", None, (left, left, left))
    ]
    for minority in range(3):
        labels = [left, left, left]
        labels[minority] = right
        result.append(("AAB", minority, tuple(labels)))
    for minority in range(3):
        labels = [right, right, right]
        labels[minority] = left
        result.append(("ABB", minority, tuple(labels)))
    result.append(("BBB", None, (right, right, right)))
    return result


def make_record(
    *, record_id: str, target_view: str, labels: Sequence[str], assignment_type: str,
    assignment_position: int | None, index: Mapping[str, Mapping[str, Mapping[str, Any]]],
    pair_id: str | None, record_role: str,
) -> dict[str, Any]:
    reference_conditions = tuple(condition for condition in CONDITIONS if condition != target_view)
    if len(reference_conditions) != 3:
        raise RuntimeError("target-view fold does not have exactly three legal references")
    sources = [dict(index[label][condition]) for label, condition in zip(labels, reference_conditions)]
    if any(source["condition_id"] == target_view for source in sources):
        raise RuntimeError("CONTROLLER-DATASET-TARGET-LEAKAGE")
    distribution = target_distribution(labels)
    logical_payload = {
        "target_view_fold": target_view,
        "reference_condition_ids": list(reference_conditions),
        "garment_labels": list(labels),
        "image_sha256": [source["image_sha256"] for source in sources],
        "clothing_mask_sha256": [source["clothing_mask_sha256"] for source in sources],
    }
    return {
        "record_id": record_id,
        "record_role": record_role,
        "pair_id": pair_id,
        "target_view_fold": target_view,
        "reference_condition_ids": list(reference_conditions),
        "garment_labels": list(labels),
        "target_distribution": distribution,
        "target_distribution_by_outfit": dict(zip(OUTFITS, distribution)),
        "assignment_type": assignment_type,
        "assignment_position": assignment_position,
        "source_references": sources,
        "source_image_hashes": [source["image_sha256"] for source in sources],
        "source_mask_hashes": [source["clothing_mask_sha256"] for source in sources],
        "target_excluded": True,
        "target_image_used": False,
        "logical_input_sha256": canonical_sha256(logical_payload),
        "duplicate_of": None,
        "duplicate_target_consistent": True,
    }


def build_training_manifest(
    source: Mapping[str, Any], source_path: Path, source_display_path: str | None = None
) -> dict[str, Any]:
    index = source_index(source)
    formal_pure = []
    canonical: dict[str, tuple[str, list[float]]] = {}
    for outfit in OUTFITS:
        for target_view in CONDITIONS:
            record = make_record(
                record_id=f"formal_pure/{outfit}/{target_view}",
                target_view=target_view,
                labels=(outfit, outfit, outfit),
                assignment_type="FORMAL_PURE_ENDPOINT",
                assignment_position=None,
                index=index,
                pair_id=None,
                record_role="RETAINED_FORMAL_PURE_ENDPOINT",
            )
            formal_pure.append(record)
            canonical[record["logical_input_sha256"]] = (
                record["record_id"], record["target_distribution"]
            )

    query_sets = []
    duplicate_count = 0
    inconsistent = []
    for left, right in PAIRS:
        pair_id = f"{left}_{right}"
        for target_view in CONDITIONS:
            for order_index, (kind, position, labels) in enumerate(assignments(left, right)):
                suffix = kind if position is None else f"{kind}_minority_{position}"
                record = make_record(
                    record_id=f"query/{pair_id}/{target_view}/{order_index:02d}_{suffix}",
                    target_view=target_view,
                    labels=labels,
                    assignment_type=kind,
                    assignment_position=position,
                    index=index,
                    pair_id=pair_id,
                    record_role="PAIR_FOLD_QUERY_SET",
                )
                logical = record["logical_input_sha256"]
                if logical in canonical:
                    duplicate_count += 1
                    duplicate_id, duplicate_target = canonical[logical]
                    record["duplicate_of"] = duplicate_id
                    record["duplicate_target_consistent"] = record["target_distribution"] == duplicate_target
                    if not record["duplicate_target_consistent"]:
                        inconsistent.append([duplicate_id, record["record_id"]])
                else:
                    canonical[logical] = (record["record_id"], record["target_distribution"])
                query_sets.append(record)

    types = {kind: sum(row["assignment_type"] == kind for row in query_sets) for kind in ("AAA", "AAB", "ABB", "BBB")}
    positions = {
        kind: {str(position): sum(row["assignment_type"] == kind and row["assignment_position"] == position for row in query_sets) for position in range(3)}
        for kind in ("AAB", "ABB")
    }
    target_exclusion_failures = [
        row["record_id"] for row in formal_pure + query_sets if not row["target_excluded"] or row["target_image_used"]
    ]
    manifest = {
        "schema_version": "canondressgs.research.dual_support_controller_training_manifest.v1",
        "label": LABEL,
        "task_id": "AAAI27-REFERENCE-CONDITIONED-DUAL-SUPPORT-CONTROLLER-001",
        "source_manifest": source_display_path or str(source_path),
        "source_manifest_sha256": sha256(source_path),
        "frozen_outfit_order": list(OUTFITS),
        "frozen_target_view_order": list(CONDITIONS),
        "pairs": [f"{left}_{right}" for left, right in PAIRS],
        "soft_target_contract": {
            "loss": "-sum_g y_g log(p_g)",
            "temperature": 1.0,
            "mixed_hard_label_forbidden": True,
        },
        "counts": {
            "pair_count": len(PAIRS),
            "target_view_folds": len(CONDITIONS),
            "sets_per_pair_fold": 8,
            "pair_fold_query_sets": len(query_sets),
            "pair_fold_pure_query_sets": types["AAA"] + types["BBB"],
            "pair_fold_mixed_query_sets": types["AAB"] + types["ABB"],
            "retained_formal_pure_endpoint_episodes": len(formal_pure),
            "total_manifest_records": len(query_sets) + len(formal_pure),
            "unique_logical_inputs": len(canonical),
            "duplicate_records": duplicate_count,
        },
        "assignment_type_counts": types,
        "assignment_position_counts": positions,
        "duplicate_detection": {
            "policy": "LOGICAL_INPUT_SHA256_WITH_SOFT_TARGET_CONSISTENCY",
            "duplicate_records": duplicate_count,
            "inconsistent_target_count": len(inconsistent),
            "inconsistent_target_pairs": inconsistent,
            "pass": len(inconsistent) == 0,
        },
        "target_exclusion": {
            "target_view_never_in_reference_conditions": not target_exclusion_failures,
            "target_image_never_used": not target_exclusion_failures,
            "failure_record_ids": target_exclusion_failures,
            "pass": not target_exclusion_failures,
        },
        "all_assignment_positions_present": all(value == 40 for group in positions.values() for value in group.values()),
        "formal_pure_endpoint_episodes": formal_pure,
        "query_sets": query_sets,
        "paper_final": False,
        "paper_final_count": 0,
    }
    if len(query_sets) != 320 or len(formal_pure) != 20:
        raise RuntimeError("CONTROLLER-DATASET-COUNT-MISMATCH")
    if types != {"AAA": 40, "AAB": 120, "ABB": 120, "BBB": 40}:
        raise RuntimeError("CONTROLLER-DATASET-ASSIGNMENT-MISMATCH")
    if not manifest["all_assignment_positions_present"] or target_exclusion_failures or inconsistent:
        raise RuntimeError("CONTROLLER-DATASET-GOVERNANCE-FAIL")
    return manifest


def atomic_json(path: Path, value: Any, *, replace: bool) -> None:
    if path.exists() and not replace:
        raise FileExistsError(f"refusing to overwrite existing manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-display-path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    source_path = args.source_manifest.resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    manifest = build_training_manifest(source, source_path, args.source_display_path)
    atomic_json(args.output.resolve(), manifest, replace=args.replace)
    print(json.dumps({"status": "PASS", **manifest["counts"], "output": str(args.output.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
