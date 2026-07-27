#!/usr/bin/env python3
"""Prepare deterministic AAAI-27 data/sprint manifests from frozen local assets.

This tool is intentionally read-only with respect to source assets. It performs no
network requests, image generation, rendering, model import, or training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


SCHEMA_VERSION = "aaai27_sprint_preflight_v1"
CREATED_AT = "2026-07-18T00:00:00+08:00"
EXPECTED_LONG_TERM_HEAD = "9fc88033b407136073f7bddc6ffca6dd4dd1e0f5"
CANDIDATE_OUTFITS = ("O01", "O02", "O03", "O04", "O06", "O07", "O08")
PROVISIONAL_BENCHMARK_OUTFITS = ("O01", "O02", "O03", "O04", "O06", "O08")
PROVISIONAL_TRAIN_OUTFITS = ("O01", "O02", "O04", "O06")
PROVISIONAL_UNSEEN_OUTFITS = ("O03", "O08")
CANONICAL_CONDITIONS = {
    "front": "cond_000000",
    "back": "cond_000318",
    "left": "cond_000017",
    "right": "cond_000347",
}
VIEW_ORDER = ("front", "back", "left", "right")
DIFFICULTY_COUNTS = {"easy": 3, "medium": 3, "hard": 2}

OUTFIT_RISK = {
    "O01": {
        "exposed_skin_required": False,
        "large_exterior_silhouette_required": False,
        "support_risk": "moderate",
        "reason": "Long sleeves and trousers cover the known missing arm support; hoodie volume still needs four-view capacity evidence.",
    },
    "O02": {
        "exposed_skin_required": False,
        "large_exterior_silhouette_required": False,
        "support_risk": "low_to_moderate",
        "reason": "Shirt and slacks are close to existing body support but cuffs/collar remain a local geometry test.",
    },
    "O03": {
        "exposed_skin_required": False,
        "large_exterior_silhouette_required": False,
        "support_risk": "moderate",
        "reason": "The suit is structured and may stress lapel/shoulder silhouette without requiring exposed skin.",
    },
    "O04": {
        "exposed_skin_required": False,
        "large_exterior_silhouette_required": False,
        "support_risk": "moderate",
        "reason": "Jacket volume is support-compatible by design but needs a capacity check around the torso and sleeves.",
    },
    "O06": {
        "exposed_skin_required": False,
        "large_exterior_silhouette_required": False,
        "support_risk": "low_to_moderate",
        "reason": "Sportswear is close-fitting and fully covering, with limited expected topology change.",
    },
    "O07": {
        "exposed_skin_required": False,
        "large_exterior_silhouette_required": True,
        "support_risk": "high",
        "reason": "The down jacket is the highest-risk candidate for puffed exterior silhouette beyond current support.",
    },
    "O08": {
        "exposed_skin_required": False,
        "large_exterior_silhouette_required": False,
        "support_risk": "low_to_moderate",
        "reason": "Sweater and pants are fully covering and close to current support.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-pre-root", type=Path, default=Path(r"E:\data_pre"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/aaai27_sprint"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--expected-long-term-head", default=EXPECTED_LONG_TERM_HEAD)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv_atomic(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def git_value(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo_root, text=True, encoding="utf-8"
    ).strip()


def load_screening_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["cond_id"]: row for row in csv.DictReader(handle)}


def condition_number(condition_id: str) -> str:
    prefix, value = condition_id.split("_", 1)
    if prefix != "cond" or not value.isdigit():
        raise ValueError(f"Invalid condition id: {condition_id}")
    return value


def resolve_condition_assets(record: dict[str, Any]) -> dict[str, Path]:
    condition_id = record["condition_id"]
    number = condition_number(condition_id)
    image = Path(record["source_condition"])
    root = image.parent.parent
    return {
        "image": image,
        "mask": root / "masks" / f"{condition_id}.png",
        "pose": root / "poses" / f"pose_{number}.npz",
        "camera": root / "cameras" / f"camera_{number}.json",
    }


def pose_energy(path: Path) -> float:
    with np.load(path, allow_pickle=False) as payload:
        parts = []
        for key in ("body_pose", "left_hand_pose", "right_hand_pose"):
            if key in payload:
                value = np.asarray(payload[key], dtype=np.float64).reshape(-1)
                if not np.isfinite(value).all():
                    raise ValueError(f"Non-finite pose values in {path}:{key}")
                parts.append(value)
        if not parts:
            raise ValueError(f"No pose vectors found in {path}")
        merged = np.concatenate(parts)
        return float(np.sqrt(np.mean(np.square(merged))))


def audit_condition_record(
    record: dict[str, Any], screening: dict[str, dict[str, str]]
) -> dict[str, Any]:
    assets = resolve_condition_assets(record)
    missing = [name for name, path in assets.items() if not path.is_file()]
    if missing:
        return {
            **record,
            "assets": {name: str(path) for name, path in assets.items()},
            "asset_status": "FAIL_MISSING",
            "missing_assets": missing,
        }

    source_sha = sha256_file(assets["image"])
    with Image.open(assets["image"]) as image_handle:
        image_handle.load()
        image_size = list(image_handle.size)
        image_mode = image_handle.mode
    with Image.open(assets["mask"]) as mask_handle:
        mask_handle.load()
        mask_size = list(mask_handle.size)
        mask_array = np.asarray(mask_handle)
    if mask_array.ndim == 3:
        mask_array = mask_array.max(axis=2)
    foreground = mask_array > 0
    if not foreground.any():
        raise ValueError(f"Empty foreground mask: {assets['mask']}")
    yy, xx = np.nonzero(foreground)
    bbox = [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())]
    width, height = image_size
    uncropped = bool(
        bbox[0] > 0 and bbox[1] > 0 and bbox[2] < width - 1 and bbox[3] < height - 1
    )
    if image_size != mask_size:
        raise ValueError(f"Image/mask size mismatch for {record['condition_id']}")

    camera_payload = read_json(assets["camera"])
    camera_text = json.dumps(camera_payload, allow_nan=False)
    if any(token in camera_text for token in ("NaN", "Infinity", "-Infinity")):
        raise ValueError(f"Non-finite camera values in {assets['camera']}")

    screening_row = screening.get(record["condition_id"])
    if screening_row is not None:
        hand_score = float(screening_row.get("hand_total_score") or 0.0)
        separation_score = float(screening_row.get("hand_separation_score") or 0.0)
        hand_status = "PASS_SCREENED" if hand_score >= 12 and separation_score >= 3 else "WARN_SCREENED"
        screening_source = "selected_conditions_300_hand_strict.csv"
    else:
        hand_score = None
        separation_score = None
        hand_status = "PASS_ACCEPTED_UNION_VISUAL_REVIEW"
        screening_source = "jay_production_union_manifest_v2.json"

    return {
        **record,
        "assets": {name: str(path) for name, path in assets.items()},
        "asset_sha256": {
            "image": source_sha,
            "mask": sha256_file(assets["mask"]),
            "pose": sha256_file(assets["pose"]),
            "camera": sha256_file(assets["camera"]),
        },
        "source_condition_hash_matches_union": source_sha
        == record["source_condition_sha256"],
        "asset_status": "PASS",
        "image_size": image_size,
        "image_mode": image_mode,
        "foreground_bbox_xyxy": bbox,
        "full_body_uncropped": uncropped,
        "hand_screening_status": hand_status,
        "hand_total_score": hand_score,
        "hand_separation_score": separation_score,
        "hand_screening_source": screening_source,
        "pose_energy": pose_energy(assets["pose"]),
    }


def spread_select(
    records: list[dict[str, Any]], count: int, mandatory_id: str | None
) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda item: (item["pose_energy"], item["condition_id"]))
    chosen: list[dict[str, Any]] = []
    if mandatory_id:
        matches = [item for item in ordered if item["condition_id"] == mandatory_id]
        if len(matches) != 1:
            raise ValueError(f"Mandatory condition {mandatory_id} not unique in difficulty bin")
        chosen.append(matches[0])
    while len(chosen) < count:
        remaining = [item for item in ordered if item not in chosen]
        if not remaining:
            raise ValueError("Insufficient records for deterministic spread selection")
        if not chosen:
            selected = remaining[len(remaining) // 2]
        else:
            scale = max(ordered[-1]["pose_energy"] - ordered[0]["pose_energy"], 1e-12)
            selected = max(
                remaining,
                key=lambda item: (
                    min(
                        abs(item["pose_energy"] - old["pose_energy"]) / scale
                        for old in chosen
                    ),
                    -abs(
                        item["pose_energy"]
                        - 0.5 * (ordered[0]["pose_energy"] + ordered[-1]["pose_energy"])
                    ),
                    item["condition_id"],
                ),
            )
        chosen.append(selected)
    return sorted(chosen, key=lambda item: (item["pose_energy"], item["condition_id"]))


def select_conditions(
    audited_records: list[dict[str, Any]], union_manifest_path: Path
) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    for view in VIEW_ORDER:
        pool = [
            item
            for item in audited_records
            if item["view"] == view
            and item["asset_status"] == "PASS"
            and item["source_condition_hash_matches_union"]
            and item["full_body_uncropped"]
            and item["hand_screening_status"].startswith("PASS")
        ]
        if len(pool) < 8:
            raise ValueError(f"Only {len(pool)} eligible conditions for {view}")
        pool.sort(key=lambda item: (item["pose_energy"], item["condition_id"]))
        n = len(pool)
        bins = {
            "easy": pool[: math.ceil(n / 3)],
            "medium": pool[math.ceil(n / 3) : math.ceil(2 * n / 3)],
            "hard": pool[math.ceil(2 * n / 3) :],
        }
        canonical_id = CANONICAL_CONDITIONS[view]
        canonical_bins = [name for name, values in bins.items() if any(x["condition_id"] == canonical_id for x in values)]
        if len(canonical_bins) != 1:
            raise ValueError(f"Canonical condition {canonical_id} is not uniquely eligible")
        for difficulty, count in DIFFICULTY_COUNTS.items():
            mandatory = canonical_id if canonical_bins[0] == difficulty else None
            for item in spread_select(bins[difficulty], count, mandatory):
                compact = {
                    key: item[key]
                    for key in (
                        "condition_id",
                        "pose_id",
                        "view",
                        "source_frame",
                        "source_camera",
                        "pose_index",
                        "pose_fingerprint",
                        "source_run",
                        "source_condition",
                        "source_condition_sha256",
                        "sheet_01_path",
                        "sheet_02_path",
                        "output_hash",
                        "pose_camera_traceable",
                        "assets",
                        "asset_sha256",
                        "image_size",
                        "image_mode",
                        "foreground_bbox_xyxy",
                        "full_body_uncropped",
                        "hand_screening_status",
                        "hand_screening_source",
                        "pose_energy",
                    )
                }
                compact["difficulty"] = difficulty
                compact["canonical_gate_view"] = item["condition_id"] == canonical_id
                selected.append(compact)

    ids = [item["condition_id"] for item in selected]
    if len(ids) != 32 or len(set(ids)) != 32:
        raise ValueError("Condition selection must contain 32 unique conditions")

    for view in VIEW_ORDER:
        view_records = [item for item in selected if item["view"] == view]
        medium = [item for item in view_records if item["difficulty"] == "medium"]
        hard = [item for item in view_records if item["difficulty"] == "hard"]
        val_record = next((item for item in reversed(medium) if not item["canonical_gate_view"]), medium[-1])
        test_record = next((item for item in reversed(hard) if not item["canonical_gate_view"]), hard[-1])
        for item in view_records:
            if item is val_record:
                item["condition_split"] = "validation"
            elif item is test_record:
                item["condition_split"] = "novel_pose_test"
            else:
                item["condition_split"] = "train"

    selected.sort(key=lambda item: (VIEW_ORDER.index(item["view"]), item["pose_energy"], item["condition_id"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "condition_selection_32",
        "created_at": CREATED_AT,
        "status": "PASS_PREFLIGHT_SELECTION",
        "selection_policy": {
            "source": str(union_manifest_path),
            "accepted_union_only": True,
            "source_hash_verified": True,
            "full_body_uncropped_required": True,
            "hand_screening_required": True,
            "views": {view: 8 for view in VIEW_ORDER},
            "difficulty_per_view": DIFFICULTY_COUNTS,
            "condition_split_per_view": {"train": 6, "validation": 1, "novel_pose_test": 1},
            "canonical_conditions_forced": CANONICAL_CONDITIONS,
            "difficulty_proxy": "within-view terciles of RMS body+left-hand+right-hand axis-angle pose energy",
        },
        "counts": {
            "total": 32,
            "views": dict(Counter(item["view"] for item in selected)),
            "difficulty": dict(Counter(item["difficulty"] for item in selected)),
            "condition_split": dict(Counter(item["condition_split"] for item in selected)),
        },
        "records": selected,
    }


def build_candidate_audit(
    mapping: dict[str, Any],
    union: dict[str, Any],
    target_manifest: dict[str, Any],
    source_paths: dict[str, Path],
) -> dict[str, Any]:
    mapping_by_id = {item["canonical_outfit_id"]: item for item in mapping["entries"]}
    target_counts = Counter(
        item["outfit_id"]
        for item in target_manifest["records"]
        if item.get("status") == "GENERATED"
    )
    hash_cache: dict[Path, str] = {}
    records = []
    for outfit_id in CANDIDATE_OUTFITS:
        mapping_entry = mapping_by_id[outfit_id]
        sheet_key = f"{mapping_entry['sheet_id']}_path"
        output_key = mapping_entry["sheet_id"]
        source_files = []
        hash_matches = 0
        for reference in union["records"]:
            source_path = Path(reference[sheet_key])
            if source_path not in hash_cache:
                hash_cache[source_path] = sha256_file(source_path)
            actual_hash = hash_cache[source_path]
            expected_hash = reference["output_hash"][output_key]
            matches = actual_hash == expected_hash
            hash_matches += int(matches)
            source_files.append(
                {
                    "condition_id": reference["condition_id"],
                    "view": reference["view"],
                    "path": str(source_path),
                    "sha256": actual_hash,
                    "manifest_sha256": expected_hash,
                    "hash_matches": matches,
                }
            )
        risk = OUTFIT_RISK[outfit_id]
        records.append(
            {
                "outfit_id": outfit_id,
                "definition_en": mapping_entry["description_en"],
                "definition_zh": mapping_entry["description_zh"],
                "generator_outfit_id": mapping_entry["generator_outfit_id"],
                "sheet_id": mapping_entry["sheet_id"],
                "crop_box_xyxy": mapping_entry["crop_box_xyxy"],
                "mapping_status": "PASS_MAPPING_V2",
                "mapping_visual_consensus": mapping_entry["visual_consensus_ratio"],
                "reference_image_count": union["counts"]["accepted_complete"],
                "reference_view_coverage": union["counts"]["view_distribution"],
                "source_file_hash_matches": hash_matches,
                "source_file_count": len(source_files),
                "source_file_records": source_files,
                **risk,
                "donor_identity_contamination_audit": "NO_SYSTEMATIC_CONTAMINATION_RECORDED_IN_ACCEPTED_UNION",
                "existing_subject02_target_count": target_counts[outfit_id],
                "existing_subject02_target_status": "PARTIAL_EXISTING_FIXTURE"
                if target_counts[outfit_id]
                else "NONE",
                "gate_status": "REQUIRES_28_IMAGE_DATA_AND_CAPACITY_GATE",
                "provisional_benchmark_member": outfit_id in PROVISIONAL_BENCHMARK_OUTFITS,
                "provisional_outfit_split": (
                    "train"
                    if outfit_id in PROVISIONAL_TRAIN_OUTFITS
                    else "unseen_test"
                    if outfit_id in PROVISIONAL_UNSEEN_OUTFITS
                    else "gate_reserve"
                ),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "candidate_outfit_audit",
        "created_at": CREATED_AT,
        "status": "PASS_PREFLIGHT_AUDIT",
        "decision": "ALL_SEVEN_ENTER_28_IMAGE_GATE; SIX-OUTFIT SET REMAINS PROVISIONAL",
        "source_manifests": {name: str(path) for name, path in source_paths.items()},
        "mapping_contract_status": mapping["status"],
        "production_union_status": union["status"],
        "records": records,
    }


def reference_for_outfit(
    outfit: dict[str, Any], condition: dict[str, Any]
) -> dict[str, Any]:
    sheet_id = outfit["sheet_id"]
    return {
        "donor_identity": "Jay",
        "condition_id": condition["condition_id"],
        "sheet_id": sheet_id,
        "sheet_path": condition[f"{sheet_id}_path"],
        "sheet_sha256": condition["output_hash"][sheet_id],
        "outfit_crop_box_xyxy": outfit["crop_box_xyxy"],
        "mapping_contract": "outfit_mapping_v2",
    }


def generation_record(
    outfit: dict[str, Any], condition: dict[str, Any], phase: str, split: str
) -> dict[str, Any]:
    condition_id = condition["condition_id"]
    outfit_id = outfit["outfit_id"]
    return {
        "record_id": f"{condition_id}_{outfit_id}",
        "phase": phase,
        "outfit_id": outfit_id,
        "condition_id": condition_id,
        "view": condition["view"],
        "difficulty": condition.get("difficulty", "canonical_gate"),
        "split": split,
        "subject02_base_condition": {
            "path": condition["source_condition"],
            "sha256": condition["source_condition_sha256"],
            "role": "target_identity_pose_camera_source_not_dressed_target",
        },
        "donor_references": [reference_for_outfit(outfit, condition)],
        "model_conditioning_fields": ["donor_references", "target_pose", "target_camera"],
        "target_dependency": False,
        "generated_target_used_as_input": False,
        "teacher_used": False,
        "target_rgb_mask_inference_input": False,
        "planned_output_path": str(
            Path(r"E:\data_pre\outputs\aaai27_subject02_targets")
            / phase
            / outfit_id
            / f"{condition_id}_{outfit_id}.png"
        ),
        "status": "PLANNED_NOT_GENERATED",
    }


def build_generation_manifests(
    candidate_audit: dict[str, Any], selection: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    outfits = {item["outfit_id"]: item for item in candidate_audit["records"]}
    conditions = {item["condition_id"]: item for item in selection["records"]}
    gate_records = []
    for outfit_id in CANDIDATE_OUTFITS:
        for view in VIEW_ORDER:
            condition = conditions[CANONICAL_CONDITIONS[view]]
            gate_records.append(generation_record(outfits[outfit_id], condition, "gate_28", "capacity_gate"))
    full_records = []
    for outfit_id in PROVISIONAL_BENCHMARK_OUTFITS:
        for condition in selection["records"]:
            if outfit_id in PROVISIONAL_UNSEEN_OUTFITS:
                split = "unseen_outfit_evaluation_only"
            else:
                split = condition["condition_split"]
            full_records.append(generation_record(outfits[outfit_id], condition, "full_192", split))
    common = {
        "schema_version": SCHEMA_VERSION,
        "created_at": CREATED_AT,
        "status": "PLANNED_NOT_GENERATED",
        "network_requests": 0,
        "images_generated": 0,
        "gate_required_before_execution": True,
    }
    gate = {
        **common,
        "artifact_type": "target_generation_gate_28_manifest",
        "candidate_outfits": list(CANDIDATE_OUTFITS),
        "canonical_conditions": CANONICAL_CONDITIONS,
        "record_count": len(gate_records),
        "records": gate_records,
    }
    full = {
        **common,
        "artifact_type": "target_generation_full_192_manifest",
        "provisional_outfits": list(PROVISIONAL_BENCHMARK_OUTFITS),
        "train_outfits": list(PROVISIONAL_TRAIN_OUTFITS),
        "unseen_test_outfits": list(PROVISIONAL_UNSEEN_OUTFITS),
        "condition_count": 32,
        "record_count": len(full_records),
        "records": full_records,
    }
    return gate, full


def experiment_registry(long_term_head: str) -> dict[str, Any]:
    entries = [
        ("M0", "Base Avatar", "main_baseline", "Zero canonical residual."),
        ("M1", "Per-Outfit Gaussian Optimization", "main_baseline", "Independent canonical residual optimization per outfit; no amortized inference."),
        ("M2", "Outfit-ID Conditioning", "main_baseline", "Learned outfit embedding replaces image references."),
        ("M3", "Global Reference", "main_baseline_or_ablation", "Globally pooled image feature broadcast to anchors."),
        ("M4", "Projection Only", "main_baseline", "Camera-aware projection/aggregation without graph completion."),
        ("M5", "Full CanonDressGS", "main_method", "Projection, aggregation, graph completion, dual gates, six residual heads."),
        ("A1", "No Graph Completion", "core_ablation", "Observed/projected anchor evidence only."),
        ("A2", "Single Gate", "core_ablation", "One gate controls geometry and appearance."),
        ("A3", "Appearance Only", "core_ablation", "Disable xyz/scaling/rotation residuals."),
        ("A4", "Single-Target Supervision", "core_ablation", "Raw edit target without protected base target."),
        ("A5", "V5.2 Alpha Objective", "core_ablation", "Replace frozen V5.3 boundary-aware alpha objective."),
        ("A6", "1/2/4 References", "core_ablation", "Matched reference-count study."),
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "experiment_registry",
        "created_at": CREATED_AT,
        "status": "PREFLIGHT_ONLY",
        "frozen_long_term_head": long_term_head,
        "method_ids": [item[0] for item in entries],
        "entries": [
            {
                "experiment_id": experiment_id,
                "name": name,
                "category": category,
                "definition": definition,
                "status": "PLANNED_NOT_RUN",
                "output_dir": None,
                "checkpoint": None,
                "optimizer_created_by_preflight": False,
            }
            for experiment_id, name, category, definition in entries
        ],
        "second_target_asset_audit": {
            "directly_usable_target_avatar_found": False,
            "status": "NO_VERIFIED_SECOND_TARGET_CHECKPOINT_PLUS_DATA_CONTRACT",
            "note": "Local/cloud config names and historical actor directories are not directly usable assets. Optional replication is deferred to post-AAAI P9.",
        },
    }


def paper_registries() -> tuple[dict[str, Any], dict[str, Any]]:
    figures = [
        ("F1", "Full CanonDressGS pipeline"),
        ("F2", "Seen-outfit novel-pose results"),
        ("F3", "Unseen-outfit results"),
        ("F4", "Base / Outfit-ID / Projection-Only / Full"),
        ("F5", "Observed/completed anchors, dual gates, residual magnitude"),
        ("F6", "Single/dual target and V5.2/V5.3 protected crops"),
        ("F7", "One/two/four reference images"),
        ("F8", "O00/O05 applicability boundaries"),
    ]
    tables = [
        ("T1", "Main seen/unseen outfit quantitative results", ["M0", "M1", "M2", "M4", "M5"]),
        ("T2", "Core ablations", ["A1", "A2", "A3", "A4", "A5", "A6"]),
        ("T3", "Efficiency and parameter counts", ["M1", "M2", "M4", "M5"]),
        ("T4", "Four-view data/capacity gate", list(CANDIDATE_OUTFITS)),
    ]
    return (
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "paper_figure_registry",
            "created_at": CREATED_AT,
            "status": "PLANNED",
            "entries": [
                {"figure_id": figure_id, "title": title, "status": "PLANNED_NOT_RENDERED", "artifact_paths": []}
                for figure_id, title in figures
            ],
        },
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "paper_table_registry",
            "created_at": CREATED_AT,
            "status": "PLANNED",
            "entries": [
                {"table_id": table_id, "title": title, "methods": methods, "status": "PLANNED_NO_RESULTS"}
                for table_id, title, methods in tables
            ],
        },
    )


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    data_root = args.data_pre_root.resolve()
    source_paths = {
        "mapping_v2": data_root / "audit_subject02_outfit_conversion_pilot_v1" / "outfit_cell_mapping_v2_normalized.json",
        "production_union_v2": data_root / "audit_jay_coverage_supplement_v1" / "jay_production_union_manifest_v2.json",
        "condition_screening": data_root / "conditions_300_hand_strict_candidates" / "selected_conditions_300_hand_strict.csv",
        "existing_targets_v4": data_root / "audit_subject02_direct_edit_v4" / "final_manifest_v4.json",
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required preflight sources missing: {missing}")

    long_term_head = git_value(repo_root, "rev-parse", "refs/heads/pipeline/full-dressable-20260715")
    if long_term_head != args.expected_long_term_head:
        raise RuntimeError(
            f"Long-term branch changed: expected {args.expected_long_term_head}, found {long_term_head}"
        )

    mapping = read_json(source_paths["mapping_v2"])
    union = read_json(source_paths["production_union_v2"])
    target_manifest = read_json(source_paths["existing_targets_v4"])
    screening = load_screening_rows(source_paths["condition_screening"])
    audited_conditions = [audit_condition_record(item, screening) for item in union["records"]]
    failed_conditions = [item["condition_id"] for item in audited_conditions if item["asset_status"] != "PASS"]
    if failed_conditions:
        raise RuntimeError(f"Accepted union contains missing source assets: {failed_conditions}")

    selection = select_conditions(audited_conditions, source_paths["production_union_v2"])
    candidate_audit = build_candidate_audit(mapping, union, target_manifest, source_paths)
    gate_manifest, full_manifest = build_generation_manifests(candidate_audit, selection)
    experiments = experiment_registry(long_term_head)
    figures, tables = paper_registries()

    write_json_atomic(output_dir / "candidate_outfit_audit.json", candidate_audit)
    write_csv_atomic(
        output_dir / "candidate_outfit_audit.csv",
        [
            {
                "outfit_id": item["outfit_id"],
                "definition_en": item["definition_en"],
                "reference_image_count": item["reference_image_count"],
                "front": item["reference_view_coverage"]["front"],
                "back": item["reference_view_coverage"]["back"],
                "left": item["reference_view_coverage"]["left"],
                "right": item["reference_view_coverage"]["right"],
                "mapping_status": item["mapping_status"],
                "source_file_hash_matches": item["source_file_hash_matches"],
                "exposed_skin_required": item["exposed_skin_required"],
                "large_exterior_silhouette_required": item["large_exterior_silhouette_required"],
                "support_risk": item["support_risk"],
                "existing_subject02_target_count": item["existing_subject02_target_count"],
                "provisional_outfit_split": item["provisional_outfit_split"],
                "gate_status": item["gate_status"],
            }
            for item in candidate_audit["records"]
        ],
        [
            "outfit_id",
            "definition_en",
            "reference_image_count",
            "front",
            "back",
            "left",
            "right",
            "mapping_status",
            "source_file_hash_matches",
            "exposed_skin_required",
            "large_exterior_silhouette_required",
            "support_risk",
            "existing_subject02_target_count",
            "provisional_outfit_split",
            "gate_status",
        ],
    )
    write_json_atomic(output_dir / "condition_selection_32.json", selection)
    write_csv_atomic(
        output_dir / "condition_selection_32.csv",
        selection["records"],
        [
            "condition_id",
            "view",
            "difficulty",
            "condition_split",
            "source_frame",
            "source_camera",
            "pose_index",
            "pose_energy",
            "source_condition",
            "source_condition_sha256",
            "full_body_uncropped",
            "hand_screening_status",
            "canonical_gate_view",
        ],
    )
    write_json_atomic(output_dir / "target_generation_gate_28_manifest.json", gate_manifest)
    write_json_atomic(output_dir / "target_generation_full_192_manifest.json", full_manifest)
    write_json_atomic(output_dir / "experiment_registry.json", experiments)
    write_csv_atomic(
        output_dir / "experiment_registry.csv",
        experiments["entries"],
        ["experiment_id", "name", "category", "definition", "status", "output_dir", "checkpoint"],
    )
    write_json_atomic(output_dir / "paper_figure_registry.json", figures)
    write_json_atomic(output_dir / "paper_table_registry.json", tables)

    print(
        json.dumps(
            {
                "status": "PASS",
                "output_dir": str(output_dir),
                "candidate_outfits": len(candidate_audit["records"]),
                "selected_conditions": len(selection["records"]),
                "gate_records": gate_manifest["record_count"],
                "full_records": full_manifest["record_count"],
                "network_requests": 0,
                "images_generated": 0,
                "optimizer_created": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
