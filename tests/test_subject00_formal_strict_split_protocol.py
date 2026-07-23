#!/usr/bin/env python3
"""Build and statically validate the frozen subject00 formal protocol.

This module never imports torch, constructs an optimizer, loads a renderer, or
creates a formal output directory.  ``--build`` only materializes the
pre-result JSON/YAML/Markdown protocol artifacts authorized by the task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "MMLPHUMAN-SUBJECT00-FORMAL-STRICT-SPLIT-PROTOCOL-001"
SOURCE_BRANCH = "research/mmlphuman-subject00-one-pass-medium-pilot-20260723"
SOURCE_HEAD = "2d0913eb1b163a79e81c1804c216f91b0a737b44"
TARGET_BRANCH = (
    "research/mmlphuman-subject00-formal-strict-split-protocol-20260723"
)
CANARY_RUNTIME_HEAD = "b0e8096589fe18069d95d2137e1db3979b4fe89f"
MEDIUM_RUNNER_COMMIT = "36c43d9dd29abfe546a5352b46378244f3bd9f1b"
CANARY_STEP0_PATH = (
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-MMLPHUMAN-SHORT-CANARY-001/attempt_001/"
    "checkpoints/step_000000.pth"
)
CANARY_STEP0_SHA = (
    "29d0edb11474f4384b6d409141d17f5da91c98824bd070196d50804cd7cb480a"
)
CANARY_STEP0_BYTES = 701_938_720
MEDIUM_STEP20249_SHA = (
    "480805878f1dba3d0748ab0b31d1e18c68fdd3f42267d92cf4763f97c0d1f443"
)
MEDIUM_STEP20249_BYTES = 724_584_644
SOURCE_RECORD_MANIFEST_SHA = (
    "865118c2f216046008925ef14b049db6e1d2921922117f6e3c3e3e2fdacd3537"
)
SINGLE_PASS_ORDER_SHA = (
    "0f0e7463d9f9066f54e19ec31f85f722afd58e58aba123af144918fdc97639f8"
)
CAMERA_SPLIT_SHA = (
    "8827cdc05a08cb7068e1f5e89ab7ba5384bf83b89686e8c9f80024b78d49cf44"
)
POSE_SPLIT_SHA = (
    "c12e5eec87c737d3740b6e6199d657c859df7ce1ec9d371ea8d4a987ef62bf06"
)
AVAILABILITY_SHA = (
    "cd00ad0a1549bc99d956e26eea300945adfaa10a20a438e931e640977475564e"
)
AVAILABILITY_RAW_SHA = (
    "da9cbce12b6d0fa9a1fddced662eefa9d2f4f331011fbbf3a9944fc878011c7a"
)
FIXED96_SHA = (
    "38476b7d9f6a012e0e21d7f188c5db5131506ec2231e8a7b50a389c7c92ad98a"
)
FORMAL_OUTPUT_ROOT = (
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-MMLPHUMAN-FORMAL-STRICT-SPLIT-001"
)
TRAIN_RECORDS = 20_249
PASS_COUNT = 5
FINAL_STEP = 101_245
MATCHED_REFERENCE_STEP = 100_000
CHECKPOINT_STEPS = [0, 20_249, 40_498, 60_747, 80_996, 100_000, 101_245]
TRAIN_CAMERAS = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23]
HELDOUT_CAMERAS = [0, 4, 8, 12, 16, 20]
PROTOCOL_DIR = REPO_ROOT / "paper_protocol/second_identity"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def write_json(path: Path, value: object) -> None:
    write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_sha(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seal(value: dict[str, Any], key: str) -> dict[str, Any]:
    if key in value:
        raise AssertionError(f"{key} already exists")
    value[key] = canonical_sha(value)
    return value


def build_quadrant(
    name: str,
    pose_semantics: str,
    camera_semantics: str,
    pose_ids: list[int],
    camera_ids: list[int],
    all_lookup: dict[tuple[int, int], dict[str, Any]],
    valid_lookup: set[tuple[int, int]],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for pose_id in pose_ids:
        for camera_id in camera_ids:
            pair = (pose_id, camera_id)
            record_id = f"subject00:pose={pose_id}:camera={camera_id}"
            if pair in valid_lookup:
                records.append(
                    {
                        "ordinal": len(records) + 1,
                        "record_id": record_id,
                        "pose_id": pose_id,
                        "camera_id": camera_id,
                        "quadrant": name,
                        "pose_semantics": pose_semantics,
                        "camera_semantics": camera_semantics,
                    }
                )
            else:
                source = all_lookup.get(pair)
                missing.append(
                    {
                        "record_id": record_id,
                        "pose_id": pose_id,
                        "camera_id": camera_id,
                        "reason": (
                            source.get("reason")
                            if source is not None
                            else "PAIR_ABSENT_FROM_FROZEN_AVAILABILITY_MANIFEST"
                        ),
                        "official_missing_image": (
                            bool(source.get("official_missing_image"))
                            if source is not None
                            else None
                        ),
                        "official_missing_mask": (
                            bool(source.get("official_missing_mask"))
                            if source is not None
                            else None
                        ),
                    }
                )
    camera_counts = Counter(item["camera_id"] for item in records)
    pose_counts = Counter(item["pose_id"] for item in records)
    result: dict[str, Any] = {
        "name": name,
        "pose_semantics": pose_semantics,
        "camera_semantics": camera_semantics,
        "pose_ids": pose_ids,
        "camera_ids": camera_ids,
        "pose_count": len(pose_ids),
        "camera_count": len(camera_ids),
        "theoretical_count": len(pose_ids) * len(camera_ids),
        "valid_count": len(records),
        "missing_count": len(missing),
        "record_order": "POSE_ID_ASCENDING_THEN_FROZEN_CAMERA_ORDER",
        "availability_filter": (
            "image_available && mask_available && valid_pair in the frozen "
            "availability manifest"
        ),
        "synthetic_replacement_count": 0,
        "buffer_pose_count": 0,
        "per_camera_valid_count": {
            str(camera_id): camera_counts[camera_id]
            for camera_id in camera_ids
        },
        "per_pose_valid_count": {
            str(pose_id): pose_counts[pose_id] for pose_id in pose_ids
        },
        "missing_records": missing,
        "records": records,
        "query_order_sha256": canonical_sha(records),
    }
    return seal(result, "manifest_content_sha256")


def build(availability_path: Path) -> None:
    if file_sha(availability_path) != AVAILABILITY_RAW_SHA:
        raise AssertionError("availability raw SHA changed")
    availability = read_json(availability_path)
    if canonical_sha(availability["entries"]) != AVAILABILITY_SHA:
        raise AssertionError("availability canonical SHA changed")

    medium_contract = read_json(
        PROTOCOL_DIR / "subject00_medium_pilot_execution_contract.json"
    )
    source_manifest = read_json(
        PROTOCOL_DIR / "subject00_medium_pilot_record_manifest.json"
    )
    pose_split = read_json(
        PROTOCOL_DIR / "subject00_novel_pose_split_v2.json"
    )
    camera_split = read_json(
        PROTOCOL_DIR / "subject00_novel_view_split_v2.json"
    )
    fixed96 = read_json(
        PROTOCOL_DIR / "subject00_canary_evaluation_manifest.json"
    )
    if source_manifest["manifest_content_sha256"] != SOURCE_RECORD_MANIFEST_SHA:
        raise AssertionError("source record-manifest SHA changed")
    if source_manifest["data_order_sha256"] != SINGLE_PASS_ORDER_SHA:
        raise AssertionError("source data-order SHA changed")
    if canonical_sha(source_manifest["records"]) != SINGLE_PASS_ORDER_SHA:
        raise AssertionError("source records no longer match order SHA")
    if pose_split["split_sha256"] != POSE_SPLIT_SHA:
        raise AssertionError("pose split SHA changed")
    if camera_split["split_sha256"] != CAMERA_SPLIT_SHA:
        raise AssertionError("camera split SHA changed")
    if fixed96["query_order_sha256"] != FIXED96_SHA:
        raise AssertionError("fixed96 query order changed")

    train_pose_ids = sorted(int(v) for v in pose_split["train_frame_ids"])
    heldout_pose_ids = sorted(
        int(v) for v in pose_split["heldout_frame_ids"]
    )
    buffer_pose_ids = sorted(
        int(v) for v in pose_split["buffer_excluded_frame_ids"]
    )
    if camera_split["train_camera_ids"] != TRAIN_CAMERAS:
        raise AssertionError("train camera order changed")
    if camera_split["heldout_camera_ids"] != HELDOUT_CAMERAS:
        raise AssertionError("held-out camera order changed")

    all_lookup = {
        (int(item["frame_id"]), int(item["camera_id"])): item
        for item in availability["entries"]
    }
    valid_lookup = {
        pair
        for pair, item in all_lookup.items()
        if item["image_available"]
        and item["mask_available"]
        and item["valid_pair"]
    }

    formal_manifest: dict[str, Any] = {
        "schema_version": "subject00.formal.train_record_manifest.v1",
        "task_id": TASK_ID,
        "subject_id": "subject00",
        "source_manifest_path": (
            "paper_protocol/second_identity/"
            "subject00_medium_pilot_record_manifest.json"
        ),
        "source_manifest_content_sha256": SOURCE_RECORD_MANIFEST_SHA,
        "source_single_pass_data_order_sha256": SINGLE_PASS_ORDER_SHA,
        "source_manifest_unchanged": True,
        "construction": source_manifest["construction"],
        "record_order": source_manifest["record_order"],
        "pose_selection": source_manifest["pose_selection"],
        "camera_ids": source_manifest["camera_ids"],
        "theoretical_record_count": source_manifest[
            "theoretical_record_count"
        ],
        "record_count": source_manifest["record_count"],
        "unique_record_count": source_manifest["unique_record_count"],
        "invalid_official_missing_count": source_manifest[
            "invalid_official_missing_count"
        ],
        "invalid_official_missing_records": source_manifest[
            "invalid_official_missing_records"
        ],
        "per_camera_valid_count": source_manifest[
            "per_camera_valid_count"
        ],
        "per_pose_valid_camera_count": source_manifest[
            "per_pose_valid_camera_count"
        ],
        "batch_size": 1,
        "shuffle": False,
        "replacement": False,
        "oversampling": False,
        "skip_count": 0,
        "pass_count": PASS_COUNT,
        "exposures_per_record": PASS_COUNT,
        "total_record_exposures": FINAL_STEP,
        "heldout_camera_exposure_count": 0,
        "heldout_pose_exposure_count": 0,
        "buffer_pose_exposure_count": 0,
        "single_pass_data_order_sha256": SINGLE_PASS_ORDER_SHA,
        "records": source_manifest["records"],
    }
    seal(formal_manifest, "manifest_content_sha256")
    formal_manifest_path = (
        PROTOCOL_DIR / "subject00_formal_train_record_manifest.json"
    )
    write_json(formal_manifest_path, formal_manifest)

    source_records = source_manifest["records"]
    schedule_steps: list[dict[str, Any]] = []
    pass_boundaries: list[dict[str, Any]] = []
    per_pass_sha: list[str] = []
    for pass_index in range(PASS_COUNT):
        first_step = pass_index * TRAIN_RECORDS + 1
        last_step = (pass_index + 1) * TRAIN_RECORDS
        pass_boundaries.append(
            {
                "pass_index": pass_index,
                "first_step": first_step,
                "last_step": last_step,
                "first_record": source_records[0],
                "last_record": source_records[-1],
            }
        )
        per_pass_sha.append(canonical_sha(source_records))
        for item in source_records:
            schedule_steps.append(
                {
                    "step": pass_index * TRAIN_RECORDS + item["ordinal"],
                    "pass_index": pass_index,
                    "pass_record_ordinal": item["ordinal"],
                    "record_ordinal": item["ordinal"],
                    "pose_id": item["pose_id"],
                    "camera_id": item["camera_id"],
                }
            )
    step100000 = schedule_steps[MATCHED_REFERENCE_STEP - 1]
    step101245 = schedule_steps[FINAL_STEP - 1]
    exposure_counts = Counter(
        (item["pose_id"], item["camera_id"]) for item in schedule_steps
    )
    schedule: dict[str, Any] = {
        "schema_version": "subject00.formal.training_schedule.v1",
        "task_id": TASK_ID,
        "classification": "FIVE_FULL_VALID_STRICT_TRAIN_PASSES",
        "seed": 0,
        "batch_size": 1,
        "pass_count": PASS_COUNT,
        "optimizer_steps_per_pass": TRAIN_RECORDS,
        "optimizer_steps": FINAL_STEP,
        "training_forward_batches": FINAL_STEP,
        "backward_calls": FINAL_STEP,
        "total_record_exposures": FINAL_STEP,
        "unique_record_count": TRAIN_RECORDS,
        "record_exposure_distribution": {
            "minimum": min(exposure_counts.values()),
            "maximum": max(exposure_counts.values()),
            "records_with_exactly_five_exposures": sum(
                value == PASS_COUNT for value in exposure_counts.values()
            ),
            "records_with_other_exposure_count": sum(
                value != PASS_COUNT for value in exposure_counts.values()
            ),
        },
        "shuffle": False,
        "replacement": False,
        "oversampling": False,
        "skip_count": 0,
        "repeat_within_pass_count": 0,
        "early_stopping": False,
        "result_driven_extension_authorized": False,
        "source_single_pass_data_order_sha256": SINGLE_PASS_ORDER_SHA,
        "per_pass_data_order_sha256": per_pass_sha,
        "pass_boundaries": pass_boundaries,
        "global_data_order_sha256": canonical_sha(schedule_steps),
        "matched_iteration_reference": {
            "step": MATCHED_REFERENCE_STEP,
            "role": "SECONDARY_SUBJECT02_MATCHED_ITERATION_REFERENCE",
            "schedule_position": step100000,
            "checkpoint_selection_authorized": False,
        },
        "formal_final": {
            "step": FINAL_STEP,
            "role": "ONLY_FORMAL_FINAL_CHECKPOINT",
            "schedule_position": step101245,
            "checkpoint_selection_authorized": False,
        },
        "start_checkpoint": {
            "path": CANARY_STEP0_PATH,
            "sha256": CANARY_STEP0_SHA,
            "bytes": CANARY_STEP0_BYTES,
            "step": 0,
            "data_order_position": 0,
            "reused_pointer_only": True,
        },
        "checkpoint_steps": CHECKPOINT_STEPS,
        "new_checkpoint_steps": CHECKPOINT_STEPS[1:],
        "steps": schedule_steps,
    }
    seal(schedule, "schedule_content_sha256")
    schedule_path = (
        PROTOCOL_DIR / "subject00_formal_training_schedule.json"
    )
    write_json(schedule_path, schedule)

    inherited = medium_contract["subject02_formal_training_contract"]
    optimizer_runtime = {
        "provenance_status": inherited["provenance_status"],
        "historically_exact_training_source": inherited[
            "historically_exact_training_source"
        ],
        "source_closure": inherited["source_closure"],
        "source_files": inherited["source_files"],
        "trainable_modules": inherited["trainable_modules"],
        "frozen_or_non_optimizer_state": inherited[
            "frozen_or_non_optimizer_state"
        ],
        "optimizer": inherited["optimizer"],
        "scheduler": inherited["scheduler"],
        "loss": inherited["loss"],
        "renderer": inherited["renderer"],
        "runtime": {
            **inherited["formal_config"],
            "dataloader_shuffle": False,
            "data_order_source": (
                "subject00_formal_training_schedule.json; direct stable "
                "schedule indexing overrides the historical shuffled loader"
            ),
        },
    }
    training_contract: dict[str, Any] = {
        "schema_version": "subject00.formal.training_contract.v1",
        "task_id": TASK_ID,
        "classification": "PRE_RESULT_FORMAL_TRAINING_CONTRACT_FROZEN",
        "source": {
            "branch": SOURCE_BRANCH,
            "head": SOURCE_HEAD,
            "classification": "SUBJECT00_MMLPHUMAN_MEDIUM_PILOT_PASS",
        },
        "target_branch": TARGET_BRANCH,
        "execution_provenance": {
            "canary_runtime_code_head": CANARY_RUNTIME_HEAD,
            "medium_runner_source_commit": MEDIUM_RUNNER_COMMIT,
            "medium_runner_path": (
                "tools/second_identity/run_subject00_medium_pilot.py"
            ),
        },
        "formal_output_root": FORMAL_OUTPUT_ROOT,
        "formal_output_root_created_during_protocol_task": False,
        "initialization": {
            "only_authorized_checkpoint": {
                "path": CANARY_STEP0_PATH,
                "sha256": CANARY_STEP0_SHA,
                "bytes": CANARY_STEP0_BYTES,
                "step": 0,
            },
            "forbidden_initializations": [
                "canary step384",
                "medium step5062",
                "medium step10124",
                "medium step15186",
                "medium step20249",
                "any subject02 checkpoint",
                "new random initialization",
            ],
            "medium_final_checkpoint_reference_only": {
                "sha256": MEDIUM_STEP20249_SHA,
                "bytes": MEDIUM_STEP20249_BYTES,
                "training_initialization_authorized": False,
            },
        },
        "training_records": {
            "path": (
                "paper_protocol/second_identity/"
                "subject00_formal_train_record_manifest.json"
            ),
            "source_manifest_content_sha256": SOURCE_RECORD_MANIFEST_SHA,
            "formal_manifest_content_sha256": formal_manifest[
                "manifest_content_sha256"
            ],
            "theoretical": 20_340,
            "valid": TRAIN_RECORDS,
            "official_missing": 91,
            "single_pass_data_order_sha256": SINGLE_PASS_ORDER_SHA,
        },
        "training_schedule": {
            "path": (
                "paper_protocol/second_identity/"
                "subject00_formal_training_schedule.json"
            ),
            "schedule_content_sha256": schedule[
                "schedule_content_sha256"
            ],
            "pass_count": PASS_COUNT,
            "final_step": FINAL_STEP,
            "matched_reference_step": MATCHED_REFERENCE_STEP,
            "checkpoint_steps": CHECKPOINT_STEPS,
        },
        "optimizer_loss_scheduler_renderer": optimizer_runtime,
        "checkpoint_contract": {
            "step0_pointer_only": True,
            "new_checkpoint_write_count": 6,
            "new_checkpoint_steps": CHECKPOINT_STEPS[1:],
            "preserve_every_checkpoint": True,
            "best_checkpoint_selection": False,
            "only_final_step": FINAL_STEP,
            "schema": inherited["checkpoint_schema"],
        },
        "resume_contract": {
            "infrastructure_resume_allowed": True,
            "resume_from": "NEAREST_COMPLETE_CHECKPOINT",
            "complete_checkpoint_requires": [
                "checkpoint payload exists",
                "checkpoint sidecar is atomically sealed",
                "payload SHA256 matches sidecar",
                "training_step and data_order_position are internally consistent",
            ],
            "restore_exactly": [
                "model state",
                "optimizer states",
                "scheduler states",
                "Python RNG",
                "NumPy RNG",
                "Torch RNG",
                "CUDA RNG states",
                "data-order position",
            ],
            "same_attempt_required": True,
            "duplicate_records_allowed": False,
            "skipped_records_allowed": False,
            "interruption_reason_and_timestamps_required": True,
            "scientific_restart_allowed": False,
            "result_driven_step_extension_allowed": False,
            "contract_change_after_results_allowed": False,
            "better_attempt_creation_allowed": False,
        },
        "topology_lbs_safety": {
            "gaussian_count": 200_000,
            "attachment_count": 200_000,
            "lbs_shape": [200_000, 55],
            "off_surface_count": 0,
            "required_zero_call_counts": {
                "clone": 0,
                "split": 0,
                "densification": 0,
                "topology_changing_prune": 0,
                "off_surface_rebind": 0,
                "legacy_grid_load": 0,
                "spatial_lbs_query": 0,
            },
        },
        "mutation_contract": {
            "subject00_raw": 0,
            "subject00_derived_assets": 0,
            "template": 0,
            "attachment": 0,
            "cached_lbs": 0,
            "strict_splits": 0,
            "subject02": 0,
            "avatarrex": 0,
            "canary_outputs": 0,
            "medium_outputs": 0,
        },
        "claim_boundary": {
            "conditionally_supported_after_future_frozen_run": [
                "subject00 avatar reconstruction portability",
                "surface-attached LBS on a second identity",
                "strict held-out camera evaluation",
                "strict held-out pose evaluation",
                "strict combined held-out pose/view evaluation",
            ],
            "not_supported": [
                "cross-identity garment editing",
                "unseen garment",
                "multi-garment subject00 benchmark",
                "loose-clothing template equivalence",
                "arbitrary garment synthesis",
                "complete second-identity CanonDressGS dressing result",
            ],
            "initialization_distinction": (
                "subject00 uses a body-surface fallback; subject02 uses a "
                "loose-clothing template; the contracts are not equivalent"
            ),
        },
        "authorization_counts_during_protocol_task": {
            "training_runs": 0,
            "training_steps": 0,
            "training_forwards": 0,
            "backward_calls": 0,
            "optimizer_created": 0,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "checkpoint_writes": 0,
            "renderer_runs": 0,
            "formal_metrics": 0,
            "visual_sheets": 0,
            "threshold_selection": 0,
        },
        "tuning_or_result_selection_authorized": False,
        "PAPER_FINAL": 0,
    }
    seal(training_contract, "contract_content_sha256")
    training_contract_path = (
        PROTOCOL_DIR / "subject00_formal_training_contract.json"
    )
    write_json(training_contract_path, training_contract)

    quadrant_specs = [
        (
            "TRAIN_FIT",
            "train",
            "train",
            train_pose_ids,
            TRAIN_CAMERAS,
        ),
        (
            "STRICT_NOVEL_VIEW",
            "train",
            "heldout",
            train_pose_ids,
            HELDOUT_CAMERAS,
        ),
        (
            "STRICT_NOVEL_POSE",
            "heldout",
            "train",
            heldout_pose_ids,
            TRAIN_CAMERAS,
        ),
        (
            "STRICT_NOVEL_POSE_AND_VIEW",
            "heldout",
            "heldout",
            heldout_pose_ids,
            HELDOUT_CAMERAS,
        ),
    ]
    quadrants = {
        name: build_quadrant(
            name,
            pose_semantics,
            camera_semantics,
            poses,
            cameras,
            all_lookup,
            valid_lookup,
        )
        for name, pose_semantics, camera_semantics, poses, cameras
        in quadrant_specs
    }
    evaluation: dict[str, Any] = {
        "schema_version": "subject00.formal.evaluation_manifests.v1",
        "task_id": TASK_ID,
        "subject_id": "subject00",
        "camera_split_sha256": CAMERA_SPLIT_SHA,
        "pose_split_sha256": POSE_SPLIT_SHA,
        "availability_entries_sha256": AVAILABILITY_SHA,
        "availability_raw_file_sha256": AVAILABILITY_RAW_SHA,
        "buffer_pose_ids": buffer_pose_ids,
        "buffer_pose_query_count": 0,
        "missing_pair_replacement_count": 0,
        "result_based_filter_count": 0,
        "quadrant_order": [item[0] for item in quadrant_specs],
        "quadrants": quadrants,
        "primary_strict_result_quadrants": [
            "STRICT_NOVEL_VIEW",
            "STRICT_NOVEL_POSE",
            "STRICT_NOVEL_POSE_AND_VIEW",
        ],
        "training_fit_role": "SEPARATE_TRAINING_FIT_AUDIT_ONLY",
        "full_evaluation": {
            "only_checkpoint_step": FINAL_STEP,
            "checkpoint_selection_allowed": False,
            "storage_mode": "STREAMING_METRIC_EVALUATION",
            "default_png_persistence": False,
            "full_evaluation_png_count": 0,
            "per_query_sequence": [
                "render",
                "calculate metrics",
                "append lightweight query record",
                "release render tensors",
            ],
            "persist": [
                "metrics JSONL or Parquet",
                "aggregate JSON",
                "failed-query evidence",
                "frozen visual-review subset",
                "preregistered diagnostic cases",
            ],
        },
        "matched_iteration_step100000": {
            "evaluation_set": "FROZEN_FIXED96_ONLY",
            "full_four_quadrant_evaluation": False,
            "resource_reason": (
                "The full availability-filtered set contains 29,954 queries; "
                "the matched-iteration point is secondary and fixed96 already "
                "covers all four semantics. This choice is frozen before "
                "formal results and reserves full streaming evaluation for "
                "the only formal final at step101245."
            ),
            "query_order_sha256": FIXED96_SHA,
            "query_count": 96,
            "result_driven_change_allowed": False,
        },
        "training_trajectory_fixed96": {
            "steps": CHECKPOINT_STEPS,
            "query_count_per_step": 96,
            "query_order_sha256": FIXED96_SHA,
            "step0_reuse": "REUSE_CANARY_IF_HASHES_MATCH",
            "step20249_reuse": (
                "REUSE_MEDIUM_ONLY_IF_MODEL_STATE, RENDERER, ASSET, AND "
                "QUERY HASH PARITY ALL PASS; OTHERWISE RENDER FIXED96"
            ),
            "new_render_steps": [40_498, 60_747, 80_996, 100_000, 101_245],
            "heldout_results_used_for_training_decisions": False,
        },
        "metric_contract": {
            "metrics": [
                "rgb_mae",
                "psnr",
                "ssim",
                "lpips",
                "silhouette_iou",
                "boundary_f_score",
                "alpha_occupancy",
                "depth_finite_ratio",
                "foreground_coverage",
                "render_seconds",
                "peak_vram_bytes",
            ],
            "aggregations": [
                "mean",
                "median",
                "standard_deviation",
                "percentiles_p05_p25_p75_p95",
                "per_camera",
                "per_pose",
                "query_level_records",
                "invalid_and_failed_counts",
            ],
            "train_fit_must_not_be_pooled_with_primary_strict_results": True,
        },
        "representation_capacity_metric_contract": {
            "numeric_fields": [
                "foreground_rgb_variance",
                "foreground_chroma",
                "gt_pred_color_histogram_l1",
                "silhouette_iou",
                "boundary_f_score",
                "garment_proxy_undercoverage",
                "garment_proxy_overcoverage",
                "body_conforming_bias_score",
                "sleeve_bulk_roi_absolute_coverage_error",
                "hem_boundary_vertical_error_px",
                "face_hand_roi_lpips",
                "face_hand_roi_rgb_mae",
            ],
            "qualitative_fields": [
                "loose_clothing_volume_missing",
                "sleeve_bulk",
                "hem_offset",
                "head_eye_detail",
                "hand_finger_detail",
            ],
            "proxy_limitation": (
                "No separate garment mask exists. Undercoverage and "
                "overcoverage use the frozen human silhouette as an explicit "
                "garment proxy and cannot by themselves establish garment "
                "geometry."
            ),
            "classification_must_use_all_fields": True,
            "lpips_only_classification_forbidden": True,
        },
    }
    seal(evaluation, "manifests_content_sha256")
    evaluation_path = (
        PROTOCOL_DIR / "subject00_formal_evaluation_manifests.json"
    )
    write_json(evaluation_path, evaluation)

    sheet_specs = [
        {
            "ordinal": item["ordinal"],
            "pose_id": item["pose_id"],
            "camera_id": item["camera_id"],
            "quadrant": item["quadrant"],
            "planned_relative_path": (
                "visual_review/fixed96/"
                f"{item['ordinal']:03d}_pose{item['pose_id']:04d}_"
                f"cam{item['camera_id']:02d}.png"
            ),
            "panels": [
                "ground_truth",
                "step0",
                "medium_step20249",
                "formal_step100000",
                "formal_step101245",
                "alpha",
                "depth",
                "metrics_and_identifiers",
            ],
        }
        for item in fixed96["queries"]
    ]
    visual: dict[str, Any] = {
        "schema_version": "subject00.formal.visual_review_manifest.v1",
        "task_id": TASK_ID,
        "source_fixed96_manifest_path": (
            "paper_protocol/second_identity/"
            "subject00_canary_evaluation_manifest.json"
        ),
        "query_order_sha256": FIXED96_SHA,
        "query_count": 96,
        "queries": fixed96["queries"],
        "planned_sheets": sheet_specs,
        "required_original_detail_open_count": 96,
        "review_completion_rule": (
            "Every one of 96 sheets must be opened at original detail and "
            "receive a persisted human record; contact-sheet-only review fails."
        ),
        "extra_case_selection": {
            "selection_time": "AFTER_METRIC_COMPUTATION_USING_ONLY_FROZEN_RULES",
            "human_posthoc_selection_allowed": False,
            "deduplication": (
                "Within each category keep the first unique query in the "
                "frozen numeric ordering; cross-category duplicates remain "
                "visible and are labeled."
            ),
            "categories": {
                "silhouette_worst": {
                    "count": 12,
                    "sort": (
                        "silhouette_iou ASC, boundary_f_score ASC, ordinal ASC"
                    ),
                },
                "lpips_worst": {
                    "count": 12,
                    "sort": "lpips DESC, rgb_mae DESC, ordinal ASC",
                },
                "body_conforming_bias": {
                    "count": 12,
                    "sort": (
                        "body_conforming_bias_score DESC, "
                        "silhouette_iou ASC, ordinal ASC"
                    ),
                    "score": (
                        "abs(predicted_foreground_coverage - "
                        "ground_truth_foreground_coverage) + "
                        "garment_proxy_undercoverage + "
                        "garment_proxy_overcoverage"
                    ),
                },
                "hoodie_sleeve_or_bulk": {
                    "count": 12,
                    "sort": (
                        "sleeve_bulk_roi_absolute_coverage_error DESC, "
                        "ordinal ASC"
                    ),
                    "roi": (
                        "frozen GT-mask shoulder-to-wrist lateral thirds"
                    ),
                },
                "hem_offset": {
                    "count": 12,
                    "sort": "hem_boundary_vertical_error_px DESC, ordinal ASC",
                    "measurement": (
                        "absolute vertical distance between the lowest "
                        "predicted and GT foreground boundary in the frozen "
                        "lower-torso central half ROI"
                    ),
                },
                "face_or_hand": {
                    "count": 12,
                    "sort": (
                        "face_hand_roi_lpips DESC, face_hand_roi_rgb_mae DESC, "
                        "ordinal ASC"
                    ),
                    "roi": (
                        "frozen subject00 face and bilateral hand boxes from "
                        "the dataset keypoints; missing ROI sorts last"
                    ),
                },
            },
        },
        "review_fields": [
            "body_conforming_bias",
            "loose_clothing_volume_missing",
            "sleeve_bulk",
            "hem_offset",
            "head_eye_detail",
            "hand_finger_detail",
            "silhouette_discontinuity",
            "failed_or_invalid",
            "notes",
        ],
        "visual_sheets_created_during_protocol_task": 0,
    }
    seal(visual, "manifest_content_sha256")
    visual_path = (
        PROTOCOL_DIR / "subject00_formal_visual_review_manifest.json"
    )
    write_json(visual_path, visual)

    empirical_training_seconds = 4_338.664056
    expected_training_seconds = empirical_training_seconds * PASS_COUNT
    empirical_optimizer_tensor_bytes = 22_640_024
    all_group_adam_moments_upper_bytes = 1_248_776_000
    conservative_checkpoint_bytes = (
        MEDIUM_STEP20249_BYTES
        + all_group_adam_moments_upper_bytes
        - empirical_optimizer_tensor_bytes
    )
    checkpoint_central = MEDIUM_STEP20249_BYTES * 6
    checkpoint_upper = conservative_checkpoint_bytes * 6
    metrics_bytes = 29_954 * 4_096 + 64 * 1024 * 1024
    visual_bytes = 96 * 1024 * 1024
    logs_bytes = round(43_751_486 * PASS_COUNT)
    resource: dict[str, Any] = {
        "schema_version": "subject00.formal.resource_budget.v1",
        "task_id": TASK_ID,
        "basis": {
            "medium_optimizer_steps": TRAIN_RECORDS,
            "medium_wall_seconds": empirical_training_seconds,
            "medium_peak_vram_bytes": 1_710_931_456,
            "medium_external_archive_reported_bytes": 3_229_325_143,
            "medium_current_tree_observed_bytes": 3_229_291_588,
            "medium_current_tree_file_count": 250,
            "medium_empirical_checkpoint_bytes": MEDIUM_STEP20249_BYTES,
            "medium_empirical_optimizer_tensor_bytes_in_checkpoint": (
                empirical_optimizer_tensor_bytes
            ),
            "all_14_group_adam_moment_upper_estimate_bytes_per_checkpoint": (
                all_group_adam_moments_upper_bytes
            ),
        },
        "output_immutability_baseline": {
            "policy": (
                "SHA256(canonical JSON list sorted by POSIX relative path; "
                "each row contains path,size,mtime_ns; UTF-8, sort_keys=true, "
                "separators=(',',':'), ensure_ascii=false)"
            ),
            "canary_attempt_001": {
                "file_count": 272,
                "bytes": 3_864_571_005,
                "metadata_sha256": (
                    "0f2fd6a3f341050d4ee5c2e1fc3b8dcb2c7fa4105f3e95113f1de3c13ec6dd40"
                ),
            },
            "medium_attempt_001": {
                "file_count": 250,
                "bytes": 3_229_291_588,
                "metadata_sha256": (
                    "c4092946c92d1553f4d693073e6c16428dd31874803b61c7872e4c009f5da213"
                ),
            },
            "required_before_after_change_count": 0,
        },
        "formal": {
            "optimizer_steps": FINAL_STEP,
            "training_wall_seconds_linear_estimate": expected_training_seconds,
            "training_wall_hours_linear_estimate": (
                expected_training_seconds / 3600
            ),
            "training_wall_seconds_expected_interval": [
                expected_training_seconds * 0.8,
                expected_training_seconds * 1.2,
            ],
            "full_final_evaluation_query_count": 29_954,
            "evaluation_wall_seconds_budget_interval": [1_800, 7_200],
            "end_to_end_wall_seconds_expected_interval": [
                expected_training_seconds * 0.8 + 1_800,
                expected_training_seconds * 1.2 + 7_200,
            ],
            "peak_vram_bytes_expected": 1_710_931_456,
            "peak_vram_bytes_gate": 4_294_967_296,
            "new_checkpoint_count": 6,
            "checkpoint_bytes_central_per_file": MEDIUM_STEP20249_BYTES,
            "checkpoint_bytes_conservative_per_file": (
                conservative_checkpoint_bytes
            ),
            "checkpoint_storage_bytes_central": checkpoint_central,
            "checkpoint_storage_bytes_conservative": checkpoint_upper,
            "optimizer_state_storage_bytes_central_total_included": (
                empirical_optimizer_tensor_bytes * 6
            ),
            "optimizer_state_storage_bytes_upper_total_included": (
                all_group_adam_moments_upper_bytes * 6
            ),
            "metrics_storage_bytes_budget": metrics_bytes,
            "visual_storage_bytes_budget": visual_bytes,
            "training_log_storage_bytes_budget": logs_bytes,
            "other_protocol_and_audit_bytes_budget": 67_108_864,
            "total_incremental_storage_bytes_central": (
                checkpoint_central
                + metrics_bytes
                + visual_bytes
                + logs_bytes
                + 67_108_864
            ),
            "total_incremental_storage_bytes_conservative": (
                checkpoint_upper
                + metrics_bytes
                + visual_bytes
                + logs_bytes
                + 67_108_864
            ),
        },
        "resource_gate": {
            "observed_cloud_free_bytes": 41_215_741_952,
            "minimum_free_bytes": 32_212_254_720,
            "observed_cloud_free_inodes": 702_696_143,
            "gpu_total_mib": 24_564,
            "gpu_free_mib_at_protocol_audit": 24_081,
            "gpu_utilization_percent_at_protocol_audit": 0,
            "status": "PASS",
            "must_be_rechecked_immediately_before_future_training": True,
            "insufficient_resource_may_block_but_may_not_change_steps": True,
        },
        "step100000_choice": {
            "choice": "FROZEN_FIXED96_ONLY",
            "scientific_result_known_at_choice_time": False,
            "resource_basis": (
                "avoid a redundant 29,954-query full metric archive at a "
                "secondary contextual checkpoint; preserve full evaluation "
                "for the fixed step101245 final"
            ),
        },
    }
    seal(resource, "budget_content_sha256")
    resource_path = (
        PROTOCOL_DIR / "subject00_formal_resource_budget.json"
    )
    write_json(resource_path, resource)

    config_text = f"""# Frozen pre-result subject00 formal strict-split protocol.
schema_version: subject00.formal.strict_split.config.v1
task_id: {TASK_ID}
protocol_only: true
formal_output_root: {FORMAL_OUTPUT_ROOT}
initialization:
  checkpoint_path: {CANARY_STEP0_PATH}
  checkpoint_sha256: {CANARY_STEP0_SHA}
  checkpoint_bytes: {CANARY_STEP0_BYTES}
  checkpoint_step: 0
  random_initialization: false
data:
  record_manifest: paper_protocol/second_identity/subject00_formal_train_record_manifest.json
  source_manifest_content_sha256: {SOURCE_RECORD_MANIFEST_SHA}
  single_pass_data_order_sha256: {SINGLE_PASS_ORDER_SHA}
  valid_record_count: {TRAIN_RECORDS}
  pass_count: {PASS_COUNT}
  shuffle: false
  replacement: false
  oversampling: false
  skip_missing: false
  synthesize_missing: false
training:
  seed: 0
  batch_size: 1
  optimizer_steps: {FINAL_STEP}
  final_step: {FINAL_STEP}
  early_stopping: false
  gradient_accumulation_steps: 1
  mixed_precision: false
  gradient_clipping: NONE
optimizer:
  group_count: 14
  common_adam:
    betas: [0.9, 0.999]
    eps: 1.0e-15
    weight_decay: 0.0
  groups:
    dxyz: {{type: Adam, parameter: dxyz_vt, lr: "0.00016 * scene_scale"}}
    scales: {{type: Adam, parameter: _scaling, lr: 0.0005}}
    quats: {{type: Adam, parameter: _rotation, lr: 0.0005}}
    opacities: {{type: Adam, parameter: _opacity, lr: 0.0005}}
    sh0: {{type: Adam, parameter: _sh0, lr: 0.0005}}
    shN: {{type: Adam, parameter: _shN, lr: 0.000025}}
    dxyz_bs: {{type: Adam, parameter: dxyz_bs, lr: "0.00016 * scene_scale / 10"}}
    dscales_bs: {{type: Adam, parameter: scaling_bs, lr: 0.0001}}
    dquats_bs: {{type: Adam, parameter: rotation_bs, lr: 0.0001}}
    dopacities_bs: {{type: Adam, parameter: opacity_bs, lr: 0.0001}}
    dsh0_bs: {{type: Adam, parameter: sh0_bs, lr: 0.0001}}
    dshN_bs: {{type: Adam, parameter: shN_bs, lr: 0.0000025}}
    encoder_feat_params: {{type: AdamW, parameter: "encoder_feat_params.values()", lr: 0.0005, weight_decay: 0.001}}
    xyz_offset: {{type: Adam, parameter: xyz_offset, lr: 0.001}}
scheduler:
  type: ExponentialLR
  horizon_steps: 800000
  step_timing: after_every_optimizer_step
  groups:
    dxyz: "gamma = 0.01 ** (1 / 800000)"
    dxyz_bs: "gamma = 0.1 ** (1 / 800000)"
    encoder_feat_params: "gamma = 0.1 ** (1 / 800000)"
    opacities: "gamma = 0.1 ** (1 / 800000)"
    quats: "gamma = 0.1 ** (1 / 800000)"
    scales: "gamma = 0.1 ** (1 / 800000)"
    sh0: "gamma = 0.1 ** (1 / 800000)"
    shN: "gamma = 0.1 ** (1 / 800000)"
    xyz_offset: "gamma = 0.1 ** (1 / 800000)"
  unscheduled_groups: [dscales_bs, dquats_bs, dopacities_bs, dsh0_bs, dshN_bs]
loss:
  l1: {{weight: 1.0, domain: masked_background_replaced_full_image}}
  lpips: {{weight: 0.1, activation: "step > 6000", crop_size: 512, random_patch_activation: "step >= 300000"}}
  dxyz_smooth: {{weight: 0.1}}
  gaussian_scaling: {{weight: 1.0, threshold: 0.01}}
renderer:
  backend: gsplat.rasterization
  entrypoint: scene.gaussian_model.GaussianModel.render
  outputs: [RGB, alpha, depth]
  rgb_clamp: [0.0, 1.0]
  scaling_modifier: 1.0
background:
  base_rgb: [1.0, 1.0, 1.0]
  random_background: true
  sampling: "torch.rand(3, device='cuda') once per training batch"
mask:
  ground_truth_boundary: replace_with_sampled_background
  ground_truth_outside: replace_with_sampled_background
  prediction_boundary: replace_with_sampled_background
  loss_domain: full_image_after_replacements
checkpoint:
  steps: [0, 20249, 40498, 60747, 80996, 100000, 101245]
  step0_pointer_only: true
  new_write_count: 6
  preserve_all: true
  final_step: 101245
  best_selection: false
evaluation:
  fixed96_query_order_sha256: {FIXED96_SHA}
  trajectory_steps: [0, 20249, 40498, 60747, 80996, 100000, 101245]
  step100000: FIXED96_ONLY
  step101245: FULL_FOUR_QUADRANT_STREAMING
  full_evaluation_png_count: 0
safety:
  gaussian_count: 200000
  attachment_count: 200000
  lbs_shape: [200000, 55]
  off_surface_count: 0
  allow_clone: false
  allow_split: false
  allow_densification: false
  allow_topology_changing_prune: false
  allow_off_surface_rebind: false
  allow_legacy_grid_load: false
  allow_spatial_lbs_query: false
authorization:
  optimizer_created_during_protocol_task: 0
  training_steps_during_protocol_task: 0
  checkpoint_writes_during_protocol_task: 0
  renderer_runs_during_protocol_task: 0
  formal_metrics_during_protocol_task: 0
  PAPER_FINAL: 0
"""
    config_path = (
        REPO_ROOT / "config/subject00_surface_lbs_formal_strict_split.yaml"
    )
    write_text(config_path, config_text)

    counts = {
        name: {
            "theoretical": item["theoretical_count"],
            "valid": item["valid_count"],
            "missing": item["missing_count"],
        }
        for name, item in quadrants.items()
    }
    training_doc = f"""# Subject00 formal strict-split training protocol

Task `{TASK_ID}` freezes a pre-result protocol only. It authorizes no optimizer construction, training, checkpoint write, render, metric computation, or result selection during this task.

## Provenance and initialization

The sole source is `{SOURCE_BRANCH}` at `{SOURCE_HEAD}` with classification `SUBJECT00_MMLPHUMAN_MEDIUM_PILOT_PASS`. Runtime provenance is canary `{CANARY_RUNTIME_HEAD}` plus medium runner commit `{MEDIUM_RUNNER_COMMIT}`. The only authorized initialization is `{CANARY_STEP0_PATH}` (`{CANARY_STEP0_SHA}`, {CANARY_STEP0_BYTES} bytes). Medium and canary nonzero checkpoints, subject02 checkpoints, and random initialization are forbidden.

## Frozen training schedule

The unchanged source manifest has 20,340 theoretical, 20,249 valid, and 91 official-missing records; content SHA is `{SOURCE_RECORD_MANIFEST_SHA}` and stable single-pass order SHA is `{SINGLE_PASS_ORDER_SHA}`. Formal training is exactly five complete repetitions of that order: 101,245 exposures and optimizer steps, every record exactly five times, no shuffle, replacement, oversampling, skip, within-pass repeat, early stop, extension, or result-driven decision. Final is step 101,245; step 100,000 is a secondary subject02 matched-iteration reference and can never compete with final.

The full schedule SHA is `{schedule['schedule_content_sha256']}`. Checkpoints are `{CHECKPOINT_STEPS}`: step0 is a pointer and six future checkpoints must be written and retained. There is no best-checkpoint selection.

## Runtime and recovery

The 14 optimizer groups, Adam/AdamW settings, learning rates, nine ExponentialLR schedules with horizon 800,000, five unscheduled BS groups, batch size 1, seed 0, disabled AMP, accumulation 1, no gradient clipping, renderer, random background, mask rules, L1, LPIPS 0.1 after step 6000, dxyz smooth 0.1, and scaling regularizer 1.0 at threshold 0.01 are copied without tuning. `LIMITED_HISTORICAL_PROVENANCE` remains explicit.

Infrastructure recovery uses the nearest complete hash-verified checkpoint in the same attempt and restores model, optimizer, scheduler, all RNG states, and exact data-order position. It may neither duplicate nor skip a record. Scientific failure cannot trigger a restart, extension, contract change, or replacement attempt.

## Safety

The frozen inventory is 200,000 Gaussians, 200,000 surface attachments, LBS `[200000,55]`, and zero off-surface elements. Clone, split, densification, topology-changing prune, off-surface rebind, legacy-grid load, and spatial-LBS query counts must remain zero. No frozen dataset, derived asset, template, attachment, LBS, split, subject02, AvatarReX, canary, or medium artifact may mutate.
"""
    evaluation_doc = f"""# Subject00 formal strict evaluation protocol

This protocol freezes all evaluation choices before formal training results exist. Split hashes are camera `{CAMERA_SPLIT_SHA}`, pose `{POSE_SPLIT_SHA}`, and availability `{AVAILABILITY_SHA}`.

## Complete availability-filtered sets

| Quadrant | Theoretical | Valid | Missing | Role |
| --- | ---: | ---: | ---: | --- |
| TRAIN_FIT | {counts['TRAIN_FIT']['theoretical']} | {counts['TRAIN_FIT']['valid']} | {counts['TRAIN_FIT']['missing']} | separate fit audit |
| STRICT_NOVEL_VIEW | {counts['STRICT_NOVEL_VIEW']['theoretical']} | {counts['STRICT_NOVEL_VIEW']['valid']} | {counts['STRICT_NOVEL_VIEW']['missing']} | primary strict result |
| STRICT_NOVEL_POSE | {counts['STRICT_NOVEL_POSE']['theoretical']} | {counts['STRICT_NOVEL_POSE']['valid']} | {counts['STRICT_NOVEL_POSE']['missing']} | primary strict result |
| STRICT_NOVEL_POSE_AND_VIEW | {counts['STRICT_NOVEL_POSE_AND_VIEW']['theoretical']} | {counts['STRICT_NOVEL_POSE_AND_VIEW']['valid']} | {counts['STRICT_NOVEL_POSE_AND_VIEW']['missing']} | primary strict result |

Every exact ID, missing record, per-camera count, per-pose count, query-order SHA, and manifest SHA is in `subject00_formal_evaluation_manifests.json`. Buffer-pose and replacement counts are zero. TRAIN_FIT is never pooled into the three primary strict averages.

Full four-quadrant streaming evaluation occurs only at fixed final step 101,245. The secondary step100,000 evaluation is uniquely frozen to fixed96 because the complete set contains 29,954 queries and full-final evidence has priority; this resource choice cannot change after results. The fixed96 trajectory covers steps `{CHECKPOINT_STEPS}` with source order SHA `{FIXED96_SHA}`. Reuse of step0 is hash-gated; medium-step reuse is permitted only under exact model-state, renderer, asset, and query parity.

For each full query, render, compute RGB MAE, PSNR, SSIM, LPIPS, silhouette IoU, boundary F, alpha occupancy, depth finite ratio, foreground coverage, render time, and peak VRAM, append a lightweight record, then release tensors. Report mean, median, standard deviation, p05/p25/p75/p95, per-camera, per-pose, query records, and invalid/failed counts. Full-evaluation PNG count is zero except the separately frozen visual subset.

All 96 original-detail review sheets must later be opened and persisted. Each compares GT, step0, medium20249, formal100000, formal101245, alpha, depth, metrics, and identifiers. Extra cases use only preregistered numeric sorts for silhouette, LPIPS, body-conforming bias, sleeve/bulk, hem, and face/hand; post-hoc human cherry-picking is forbidden.
"""
    representation_doc = """# Subject00 formal representation-capacity protocol

`BODY_SURFACE_REPRESENTATION_STATUS` is a preregistered capacity diagnosis, not a garment-editing claim. After the fixed final evaluation and 96/96 visual review, exactly one of these labels may be assigned:

- `APPEARANCE_AND_SILHOUETTE_ESTABLISHED`
- `APPEARANCE_EMERGING_GEOMETRY_LIMITED`
- `LOW_TEXTURE_BODY_BOUND`
- `REPRESENTATION_INCONCLUSIVE`

The decision must jointly use foreground RGB variance, chroma, GT/pred histogram distance, silhouette IoU, boundary F, undercoverage, overcoverage, body-conforming bias, missing loose-clothing volume, sleeve bulk, hem offset, head/eye detail, and hand/finger detail. LPIPS alone is insufficient. Numeric aggregates must be reported by strict quadrant and reconciled with the complete visual records; failures and ambiguity remain visible.

If the future frozen run completes, supported scope is limited to subject00 avatar-reconstruction portability, surface-attached LBS on a second identity, and strict held-out camera, pose, and combined pose/view evaluation. It does not establish cross-identity garment editing, unseen garments, a multi-garment benchmark, loose-clothing-template equivalence, arbitrary garment synthesis, or a complete second-identity CanonDressGS dressing result.

Subject00 uses a body-surface fallback initialization. Subject02 uses a loose-clothing template. Their initialization contracts are materially different and must never be described as equivalent.
"""
    docs_dir = REPO_ROOT / "docs/SECOND_IDENTITY"
    write_text(
        docs_dir
        / "SUBJECT00_FORMAL_STRICT_SPLIT_TRAINING_PROTOCOL_20260723.md",
        training_doc,
    )
    write_text(
        docs_dir / "SUBJECT00_FORMAL_EVALUATION_PROTOCOL_20260723.md",
        evaluation_doc,
    )
    write_text(
        docs_dir
        / "SUBJECT00_FORMAL_REPRESENTATION_CAPACITY_PROTOCOL_20260723.md",
        representation_doc,
    )

    summary: dict[str, Any] = {
        "schema_version": "subject00.formal.protocol_final_summary.v1",
        "task_id": TASK_ID,
        "classification": "SUBJECT00_FORMAL_STRICT_SPLIT_PROTOCOL_READY",
        "source": {"branch": SOURCE_BRANCH, "head": SOURCE_HEAD},
        "target_branch": TARGET_BRANCH,
        "execution_provenance": {
            "canary_runtime_code_head": CANARY_RUNTIME_HEAD,
            "medium_runner_source_commit": MEDIUM_RUNNER_COMMIT,
        },
        "initialization_checkpoint": {
            "path": CANARY_STEP0_PATH,
            "sha256": CANARY_STEP0_SHA,
            "bytes": CANARY_STEP0_BYTES,
        },
        "strict_split_hashes": {
            "camera": CAMERA_SPLIT_SHA,
            "pose": POSE_SPLIT_SHA,
            "availability": AVAILABILITY_SHA,
            "fixed96": FIXED96_SHA,
        },
        "training": {
            "record_manifest_source_sha256": SOURCE_RECORD_MANIFEST_SHA,
            "record_count": TRAIN_RECORDS,
            "pass_count": PASS_COUNT,
            "final_step": FINAL_STEP,
            "schedule_content_sha256": schedule[
                "schedule_content_sha256"
            ],
            "exposures_per_record": 5,
            "checkpoint_steps": CHECKPOINT_STEPS,
        },
        "evaluation": {
            "counts": counts,
            "manifest_content_sha256": evaluation[
                "manifests_content_sha256"
            ],
            "step100000": "FROZEN_FIXED96_ONLY",
            "step101245": "FULL_FOUR_QUADRANT_STREAMING_ONLY_FINAL",
            "full_evaluation_png_count": 0,
        },
        "representation": {
            "status_to_be_assigned_after_future_evidence": True,
            "candidate_labels": [
                "APPEARANCE_AND_SILHOUETTE_ESTABLISHED",
                "APPEARANCE_EMERGING_GEOMETRY_LIMITED",
                "LOW_TEXTURE_BODY_BOUND",
                "REPRESENTATION_INCONCLUSIVE",
            ],
            "lpips_only_classification_forbidden": True,
        },
        "claim_boundary": training_contract["claim_boundary"],
        "authorization_counts": training_contract[
            "authorization_counts_during_protocol_task"
        ],
        "mutation_audit": training_contract["mutation_contract"],
        "PAPER_FINAL": 0,
        "next_task": (
            "RUN_SUBJECT00_FORMAL_STRICT_SPLIT_TRAINING_FROM_FROZEN_PROTOCOL"
        ),
        "next_task_started": False,
        "seal_commit_resolution": "git rev-parse HEAD after final seal commit",
        "reports": [
            "docs/SECOND_IDENTITY/"
            "SUBJECT00_FORMAL_STRICT_SPLIT_TRAINING_PROTOCOL_20260723.md",
            "docs/SECOND_IDENTITY/"
            "SUBJECT00_FORMAL_EVALUATION_PROTOCOL_20260723.md",
            "docs/SECOND_IDENTITY/"
            "SUBJECT00_FORMAL_REPRESENTATION_CAPACITY_PROTOCOL_20260723.md",
        ],
        "config": (
            "config/subject00_surface_lbs_formal_strict_split.yaml"
        ),
    }
    seal(summary, "summary_content_sha256")
    summary_path = (
        PROTOCOL_DIR / "subject00_formal_protocol_final_summary.json"
    )
    write_json(summary_path, summary)

    handoff: dict[str, Any] = {
        "schema_version": "subject00.formal.protocol_handoff.v1",
        "task_id": TASK_ID,
        "status": "READY_AFTER_LOCAL_ORIGIN_CLOUD_HEAD_PARITY",
        "source_head": SOURCE_HEAD,
        "target_branch": TARGET_BRANCH,
        "protocol_summary": (
            "paper_protocol/second_identity/"
            "subject00_formal_protocol_final_summary.json"
        ),
        "formal_output_root": FORMAL_OUTPUT_ROOT,
        "formal_output_root_created": False,
        "entry_gate": [
            "local, origin, and cloud HEAD are identical",
            "local and cloud worktrees are clean",
            "free disk is at least 32212254720 bytes",
            "step0 checkpoint SHA and bytes match",
            "all frozen protocol artifact hashes pass",
            "canary and medium output fingerprints remain unchanged",
        ],
        "next_task": (
            "RUN_SUBJECT00_FORMAL_STRICT_SPLIT_TRAINING_FROM_FROZEN_PROTOCOL"
        ),
        "next_task_started": False,
        "automatic_start_authorized": False,
        "PAPER_FINAL": 0,
    }
    seal(handoff, "handoff_content_sha256")
    write_json(
        REPO_ROOT
        / "project_control_handoff/subject00_formal_protocol_handoff.json",
        handoff,
    )


def validate(availability_path: Path | None = None) -> None:
    branch = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "branch", "--show-current"], text=True
    ).strip()
    if branch != TARGET_BRANCH:
        raise AssertionError(f"wrong branch: {branch}")
    subprocess.run(
        ["git", "-C", str(REPO_ROOT), "cat-file", "-e", f"{SOURCE_HEAD}^{{commit}}"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", SOURCE_HEAD, "HEAD"],
        check=True,
    )
    if availability_path is not None:
        if file_sha(availability_path) != AVAILABILITY_RAW_SHA:
            raise AssertionError("availability raw SHA mismatch")
        availability = read_json(availability_path)
        if canonical_sha(availability["entries"]) != AVAILABILITY_SHA:
            raise AssertionError("availability canonical SHA mismatch")

    paths = [
        PROTOCOL_DIR / "subject00_formal_training_contract.json",
        PROTOCOL_DIR / "subject00_formal_training_schedule.json",
        PROTOCOL_DIR / "subject00_formal_train_record_manifest.json",
        PROTOCOL_DIR / "subject00_formal_evaluation_manifests.json",
        PROTOCOL_DIR / "subject00_formal_visual_review_manifest.json",
        PROTOCOL_DIR / "subject00_formal_resource_budget.json",
        PROTOCOL_DIR / "subject00_formal_protocol_final_summary.json",
        REPO_ROOT
        / "project_control_handoff/subject00_formal_protocol_handoff.json",
    ]
    if any(not path.is_file() for path in paths):
        raise AssertionError("one or more required JSON files are absent")
    (
        contract,
        schedule,
        train_manifest,
        evaluation,
        visual,
        resource,
        summary,
        handoff,
    ) = [read_json(path) for path in paths]

    seals = [
        (contract, "contract_content_sha256"),
        (schedule, "schedule_content_sha256"),
        (train_manifest, "manifest_content_sha256"),
        (evaluation, "manifests_content_sha256"),
        (visual, "manifest_content_sha256"),
        (resource, "budget_content_sha256"),
        (summary, "summary_content_sha256"),
        (handoff, "handoff_content_sha256"),
    ]
    for value, key in seals:
        claimed = value[key]
        unsigned = dict(value)
        del unsigned[key]
        if canonical_sha(unsigned) != claimed:
            raise AssertionError(f"content seal failed: {key}")

    if contract["source"]["head"] != SOURCE_HEAD:
        raise AssertionError("source HEAD mismatch")
    if contract["initialization"]["only_authorized_checkpoint"] != {
        "path": CANARY_STEP0_PATH,
        "sha256": CANARY_STEP0_SHA,
        "bytes": CANARY_STEP0_BYTES,
        "step": 0,
    }:
        raise AssertionError("formal initialization mismatch")
    if train_manifest["source_manifest_content_sha256"] != SOURCE_RECORD_MANIFEST_SHA:
        raise AssertionError("training record source hash mismatch")
    if canonical_sha(train_manifest["records"]) != SINGLE_PASS_ORDER_SHA:
        raise AssertionError("single-pass record order mismatch")
    if schedule["pass_count"] != 5 or schedule["optimizer_steps"] != FINAL_STEP:
        raise AssertionError("five-pass or final-step mismatch")
    if len(schedule["steps"]) != FINAL_STEP:
        raise AssertionError("full schedule is incomplete")
    if schedule["checkpoint_steps"] != CHECKPOINT_STEPS:
        raise AssertionError("checkpoint schedule mismatch")
    if schedule["matched_iteration_reference"]["step"] != MATCHED_REFERENCE_STEP:
        raise AssertionError("step100000 reference mismatch")
    if schedule["formal_final"]["step"] != FINAL_STEP:
        raise AssertionError("step101245 final mismatch")
    exposure_counts = Counter(
        (item["pose_id"], item["camera_id"]) for item in schedule["steps"]
    )
    if len(exposure_counts) != TRAIN_RECORDS or set(exposure_counts.values()) != {5}:
        raise AssertionError("record exposures are not exactly five")
    for pass_index in range(PASS_COUNT):
        start = pass_index * TRAIN_RECORDS
        end = start + TRAIN_RECORDS
        projected = [
            {
                "ordinal": item["record_ordinal"],
                "pose_id": item["pose_id"],
                "camera_id": item["camera_id"],
            }
            for item in schedule["steps"][start:end]
        ]
        if canonical_sha(projected) != SINGLE_PASS_ORDER_SHA:
            raise AssertionError(f"pass {pass_index} order mismatch")

    runtime = contract["optimizer_loss_scheduler_renderer"]
    if len(runtime["optimizer"]["groups"]) != 14:
        raise AssertionError("optimizer group count mismatch")
    if len(runtime["scheduler"]["groups"]) != 9:
        raise AssertionError("scheduler group count mismatch")
    if runtime["runtime"]["dataloader_shuffle"] is not False:
        raise AssertionError("formal schedule must not shuffle")
    if runtime["runtime"]["mixed_precision"] != "DISABLED_NO_AUTOCAST_OR_GRADSCALER":
        raise AssertionError("mixed-precision contract mismatch")
    if contract["resume_contract"]["restore_exactly"][-1] != "data-order position":
        raise AssertionError("resume contract lacks data-order position")

    expected_counts = {
        "TRAIN_FIT": (20_340, 20_249, 91),
        "STRICT_NOVEL_VIEW": (6_780, 6_749, 31),
        "STRICT_NOVEL_POSE": (2_250, 2_217, 33),
        "STRICT_NOVEL_POSE_AND_VIEW": (750, 739, 11),
    }
    pose_split = read_json(
        PROTOCOL_DIR / "subject00_novel_pose_split_v2.json"
    )
    buffer_ids = set(int(v) for v in pose_split["buffer_excluded_frame_ids"])
    for name, expected in expected_counts.items():
        item = evaluation["quadrants"][name]
        actual = (
            item["theoretical_count"],
            item["valid_count"],
            item["missing_count"],
        )
        if actual != expected:
            raise AssertionError(f"{name} counts mismatch: {actual}")
        if item["valid_count"] + item["missing_count"] != item["theoretical_count"]:
            raise AssertionError(f"{name} availability accounting mismatch")
        if item["synthetic_replacement_count"] != 0:
            raise AssertionError(f"{name} replaces missing pairs")
        if any(record["pose_id"] in buffer_ids for record in item["records"]):
            raise AssertionError(f"{name} contains a buffer pose")
        if canonical_sha(item["records"]) != item["query_order_sha256"]:
            raise AssertionError(f"{name} query-order SHA mismatch")
    if evaluation["camera_split_sha256"] != CAMERA_SPLIT_SHA:
        raise AssertionError("camera split hash mismatch")
    if evaluation["pose_split_sha256"] != POSE_SPLIT_SHA:
        raise AssertionError("pose split hash mismatch")
    if evaluation["availability_entries_sha256"] != AVAILABILITY_SHA:
        raise AssertionError("availability hash mismatch")
    if evaluation["full_evaluation"]["only_checkpoint_step"] != FINAL_STEP:
        raise AssertionError("full evaluation is not final-only")
    if evaluation["full_evaluation"]["full_evaluation_png_count"] != 0:
        raise AssertionError("full-evaluation PNG count is not zero")
    if evaluation["matched_iteration_step100000"]["evaluation_set"] != "FROZEN_FIXED96_ONLY":
        raise AssertionError("step100000 choice is not uniquely frozen")
    if visual["query_order_sha256"] != FIXED96_SHA or len(visual["queries"]) != 96:
        raise AssertionError("fixed96 visual manifest mismatch")
    if len(visual["planned_sheets"]) != 96:
        raise AssertionError("visual sheet plan is incomplete")

    zero_counts = contract["authorization_counts_during_protocol_task"]
    if set(zero_counts.values()) != {0}:
        raise AssertionError("protocol task contains an execution count")
    if contract["PAPER_FINAL"] != 0 or summary["PAPER_FINAL"] != 0:
        raise AssertionError("PAPER_FINAL must remain zero")
    safety = contract["topology_lbs_safety"]
    if (
        safety["gaussian_count"] != 200_000
        or safety["attachment_count"] != 200_000
        or safety["lbs_shape"] != [200_000, 55]
        or safety["off_surface_count"] != 0
        or set(safety["required_zero_call_counts"].values()) != {0}
    ):
        raise AssertionError("topology/LBS safety contract mismatch")
    if resource["resource_gate"]["status"] != "PASS":
        raise AssertionError("resource budget gate failed")
    if (
        resource["resource_gate"]["observed_cloud_free_bytes"]
        < resource["resource_gate"]["minimum_free_bytes"]
    ):
        raise AssertionError("recorded cloud disk gate is inconsistent")
    if summary["classification"] != "SUBJECT00_FORMAL_STRICT_SPLIT_PROTOCOL_READY":
        raise AssertionError("final classification mismatch")

    config_path = (
        REPO_ROOT / "config/subject00_surface_lbs_formal_strict_split.yaml"
    )
    if not config_path.is_file() or not config_path.read_text(encoding="utf-8").strip():
        raise AssertionError("formal config missing or empty")
    docs = [
        REPO_ROOT
        / "docs/SECOND_IDENTITY/"
        "SUBJECT00_FORMAL_STRICT_SPLIT_TRAINING_PROTOCOL_20260723.md",
        REPO_ROOT
        / "docs/SECOND_IDENTITY/"
        "SUBJECT00_FORMAL_EVALUATION_PROTOCOL_20260723.md",
        REPO_ROOT
        / "docs/SECOND_IDENTITY/"
        "SUBJECT00_FORMAL_REPRESENTATION_CAPACITY_PROTOCOL_20260723.md",
    ]
    if any(not path.is_file() or not path.read_text(encoding="utf-8").strip() for path in docs):
        raise AssertionError("one or more Markdown reports are empty")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--availability-manifest", type=Path)
    args = parser.parse_args()
    if args.build:
        if args.availability_manifest is None:
            parser.error("--build requires --availability-manifest")
        build(args.availability_manifest)
    validate(args.availability_manifest)
    print(
        json.dumps(
            {
                "status": "PASS",
                "task_id": TASK_ID,
                "training_records": TRAIN_RECORDS,
                "pass_count": PASS_COUNT,
                "final_step": FINAL_STEP,
                "fixed96": 96,
                "PAPER_FINAL": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
