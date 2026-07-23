#!/usr/bin/env python3
"""Build the immutable pre-result contract for the subject00 medium pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "MMLPHUMAN-SUBJECT00-ONE-PASS-MEDIUM-PILOT-001"
SOURCE_BRANCH = (
    "research/mmlphuman-subject00-short-canary-from-repaired-contract-20260723"
)
SOURCE_HEAD = "8c69cdce0139532a33b1d4f839da7467784270e6"
TARGET_BRANCH = (
    "research/mmlphuman-subject00-one-pass-medium-pilot-20260723"
)
CANARY_EXECUTION_HEAD = "b0e8096589fe18069d95d2137e1db3979b4fe89f"
CANARY_CLASSIFICATION = "SUBJECT00_MMLPHUMAN_SHORT_CANARY_PASS"
AVAILABILITY_SHA = (
    "cd00ad0a1549bc99d956e26eea300945adfaa10a20a438e931e640977475564e"
)
AVAILABILITY_RAW_SHA = (
    "da9cbce12b6d0fa9a1fddced662eefa9d2f4f331011fbbf3a9944fc878011c7a"
)
CAMERA_SPLIT_SHA = (
    "8827cdc05a08cb7068e1f5e89ab7ba5384bf83b89686e8c9f80024b78d49cf44"
)
POSE_SPLIT_SHA = (
    "c12e5eec87c737d3740b6e6199d657c859df7ce1ec9d371ea8d4a987ef62bf06"
)
EVAL_ORDER_SHA = (
    "38476b7d9f6a012e0e21d7f188c5db5131506ec2231e8a7b50a389c7c92ad98a"
)
CANARY_STEP0_PATH = (
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-MMLPHUMAN-SHORT-CANARY-001/attempt_001/"
    "checkpoints/step_000000.pth"
)
CANARY_STEP0_SHA = (
    "29d0edb11474f4384b6d409141d17f5da91c98824bd070196d50804cd7cb480a"
)
CANARY_STEP0_BYTES = 701_938_720
CANARY_STEP0_PAYLOAD_SOURCE_HEAD = (
    "55cb5a28b8ff0d6a3373582759b8704df2267331"
)
TRAIN_CAMERAS = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23]
HELDOUT_CAMERAS = [0, 4, 8, 12, 16, 20]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def canonical_sha(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), *arguments], text=True
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--availability-manifest", type=Path, required=True)
    args = parser.parse_args()
    if git_value("rev-parse", "HEAD") != SOURCE_HEAD:
        raise RuntimeError("contract must be built from the exact source HEAD")
    if git_value("branch", "--show-current") != TARGET_BRANCH:
        raise RuntimeError("contract is being built on the wrong branch")
    availability = read_json(args.availability_manifest)
    if file_sha(args.availability_manifest) != AVAILABILITY_RAW_SHA:
        raise RuntimeError("availability raw-file SHA changed")
    if canonical_sha(availability["entries"]) != AVAILABILITY_SHA:
        raise RuntimeError("availability canonical SHA changed")
    pose_path = (
        REPO_ROOT
        / "paper_protocol/second_identity/subject00_novel_pose_split_v2.json"
    )
    view_path = (
        REPO_ROOT
        / "paper_protocol/second_identity/subject00_novel_view_split_v2.json"
    )
    evaluation_path = (
        REPO_ROOT
        / "paper_protocol/second_identity/"
        "subject00_canary_evaluation_manifest.json"
    )
    repaired_contract_path = (
        REPO_ROOT
        / "paper_protocol/second_identity/"
        "subject00_canary_training_contract_repaired.json"
    )
    canary_training_path = (
        REPO_ROOT
        / "paper_protocol/second_identity/subject00_canary_training_results.json"
    )
    canary_summary_path = (
        REPO_ROOT
        / "paper_protocol/second_identity/subject00_canary_final_summary.json"
    )
    runner_path = (
        REPO_ROOT
        / "tools/second_identity/run_subject00_medium_pilot.py"
    )
    pose = read_json(pose_path)
    view = read_json(view_path)
    evaluation = read_json(evaluation_path)
    repaired_contract = read_json(repaired_contract_path)
    canary_training = read_json(canary_training_path)
    canary_summary = read_json(canary_summary_path)
    if pose["split_sha256"] != POSE_SPLIT_SHA:
        raise RuntimeError("pose split SHA changed")
    if view["split_sha256"] != CAMERA_SPLIT_SHA:
        raise RuntimeError("camera split SHA changed")
    if view["train_camera_ids"] != TRAIN_CAMERAS:
        raise RuntimeError("train camera order changed")
    if view["heldout_camera_ids"] != HELDOUT_CAMERAS:
        raise RuntimeError("held-out cameras changed")
    if evaluation["query_order_sha256"] != EVAL_ORDER_SHA:
        raise RuntimeError("evaluation query-order SHA changed")
    if len(evaluation["queries"]) != 96:
        raise RuntimeError("evaluation query count changed")
    if canary_summary["classification"] != CANARY_CLASSIFICATION:
        raise RuntimeError("canary classification changed")
    valid_lookup = {
        (int(item["frame_id"]), int(item["camera_id"])): item
        for item in availability["entries"]
        if item["image_available"]
        and item["mask_available"]
        and item["valid_pair"]
    }
    all_lookup = {
        (int(item["frame_id"]), int(item["camera_id"])): item
        for item in availability["entries"]
    }
    train_pose_ids = sorted(int(value) for value in pose["train_frame_ids"])
    heldout_pose_ids = sorted(
        int(value) for value in pose["heldout_frame_ids"]
    )
    buffer_pose_ids = sorted(
        int(value) for value in pose["buffer_excluded_frame_ids"]
    )
    records = []
    invalid = []
    for pose_id in train_pose_ids:
        for camera_id in TRAIN_CAMERAS:
            pair = (pose_id, camera_id)
            if pair in valid_lookup:
                records.append(
                    {
                        "ordinal": len(records) + 1,
                        "pose_id": pose_id,
                        "camera_id": camera_id,
                    }
                )
            else:
                source = all_lookup.get(pair)
                invalid.append(
                    {
                        "pose_id": pose_id,
                        "camera_id": camera_id,
                        "reason": (
                            source.get("reason")
                            if source is not None
                            else "PAIR_ABSENT_FROM_FROZEN_MANIFEST"
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
    theoretical = len(train_pose_ids) * len(TRAIN_CAMERAS)
    n_valid = len(records)
    if theoretical != 20_340 or n_valid != 20_249 or len(invalid) != 91:
        raise RuntimeError(
            f"unexpected record counts: {theoretical}/{n_valid}/{len(invalid)}"
        )
    data_order_sha = canonical_sha(records)
    expected_data_order_sha = (
        "0f0e7463d9f9066f54e19ec31f85f722afd58e58aba123af144918fdc97639f8"
    )
    if data_order_sha != expected_data_order_sha:
        raise RuntimeError("medium data-order SHA changed")
    camera_counts = Counter(item["camera_id"] for item in records)
    pose_counts = Counter(item["pose_id"] for item in records)
    record_manifest = {
        "schema_version": "subject00.medium.record_manifest.v1",
        "task_id": TASK_ID,
        "subject_id": "subject00",
        "construction": (
            "for pose_id in sorted(strict_train_pose_ids), then camera_id "
            "in frozen train camera order; append iff frozen availability "
            "image_available && mask_available && valid_pair"
        ),
        "pose_selection": {
            "policy": "ALL_STRICT_TRAIN_POSES_SORTED",
            "count": len(train_pose_ids),
            "pose_ids": train_pose_ids,
        },
        "camera_ids": TRAIN_CAMERAS,
        "record_order": "POSE_MAJOR_FIXED_CAMERA_MINOR",
        "batch_size": 1,
        "shuffle": False,
        "replacement": False,
        "oversampling": False,
        "exposures_per_record": 1,
        "theoretical_record_count": theoretical,
        "record_count": n_valid,
        "unique_record_count": len(
            {(item["pose_id"], item["camera_id"]) for item in records}
        ),
        "invalid_official_missing_count": len(invalid),
        "invalid_official_missing_records": invalid,
        "per_camera_valid_count": {
            str(key): camera_counts[key] for key in TRAIN_CAMERAS
        },
        "per_pose_valid_camera_count": {
            str(key): pose_counts[key] for key in train_pose_ids
        },
        "heldout_camera_exposure_count": 0,
        "heldout_pose_exposure_count": 0,
        "buffer_pose_exposure_count": 0,
        "data_order_sha256": data_order_sha,
        "data_order_sha256_policy": (
            "SHA256(canonical JSON of records; UTF-8, sort_keys=true, "
            "separators=(',',':'), ensure_ascii=false)"
        ),
        "records": records,
    }
    record_manifest["manifest_content_sha256"] = canonical_sha(
        record_manifest
    )
    protocol_dir = REPO_ROOT / "paper_protocol/second_identity"
    record_path = (
        protocol_dir / "subject00_medium_pilot_record_manifest.json"
    )
    write_json(record_path, record_manifest)
    checkpoint_steps = [
        0,
        n_valid // 4,
        n_valid // 2,
        (3 * n_valid) // 4,
        n_valid,
    ]
    schedule_steps = [
        {
            "step": item["ordinal"],
            "record_ordinal": item["ordinal"],
            "pose_id": item["pose_id"],
            "camera_id": item["camera_id"],
        }
        for item in records
    ]
    schedule = {
        "schema_version": "subject00.medium.schedule.v1",
        "task_id": TASK_ID,
        "classification": "ONE_FULL_VALID_STRICT_TRAIN_PASS",
        "seed": 0,
        "batch_size": 1,
        "optimizer_steps": n_valid,
        "training_forward_batches": n_valid,
        "backward_calls": n_valid,
        "start_checkpoint": {
            "path": CANARY_STEP0_PATH,
            "sha256": CANARY_STEP0_SHA,
            "bytes": CANARY_STEP0_BYTES,
            "step": 0,
            "data_order_position": 0,
            "reused_pointer_only": True,
        },
        "checkpoint_steps": checkpoint_steps,
        "new_checkpoint_steps": checkpoint_steps[1:],
        "diagnostic_evaluation_steps": checkpoint_steps[1:-1],
        "final_evaluation_step": checkpoint_steps[-1],
        "early_stopping": False,
        "second_pass_authorized": False,
        "result_driven_extension_authorized": False,
        "data_order_sha256": data_order_sha,
        "steps": schedule_steps,
    }
    schedule["schedule_content_sha256"] = canonical_sha(schedule)
    schedule_path = protocol_dir / "subject00_medium_pilot_schedule.json"
    write_json(schedule_path, schedule)
    canary_step0_record = next(
        (
            item
            for item in canary_training["training_result"]["checkpoints"]
            if int(item["step"]) == 0
        ),
        None,
    )
    if canary_step0_record is None:
        raise RuntimeError("canary training results lack step0 checkpoint")
    if (
        canary_step0_record["path"] != CANARY_STEP0_PATH
        or canary_step0_record["sha256"] != CANARY_STEP0_SHA
        or int(canary_step0_record["bytes"]) != CANARY_STEP0_BYTES
    ):
        raise RuntimeError("canary step0 checkpoint provenance changed")
    execution_contract = {
        "schema_version": "subject00.medium.execution_contract.v1",
        "task_id": TASK_ID,
        "classification": "PRE_RESULT_EXECUTION_CONTRACT_FROZEN",
        "source": {"branch": SOURCE_BRANCH, "head": SOURCE_HEAD},
        "target_branch": TARGET_BRANCH,
        "canary_classification": CANARY_CLASSIFICATION,
        "canary_execution_code_head": CANARY_EXECUTION_HEAD,
        "actual_runtime_code_provenance": {
            "path": "tools/second_identity/run_subject00_medium_pilot.py",
            "file_sha256": file_sha(runner_path),
            "freeze_commit_head_resolution": (
                "git rev-parse HEAD after this pre-result contract is committed"
            ),
        },
        "canary_step0_checkpoint": {
            "path": CANARY_STEP0_PATH,
            "sha256": CANARY_STEP0_SHA,
            "bytes": CANARY_STEP0_BYTES,
            "step": 0,
            "data_order_position": 0,
            "checkpoint_payload_source_head": (
                CANARY_STEP0_PAYLOAD_SOURCE_HEAD
            ),
            "only_authorized_initialization": True,
            "source_record": (
                "paper_protocol/second_identity/"
                "subject00_canary_training_results.json"
            ),
            "source_record_file_sha256": file_sha(canary_training_path),
            "final_summary_provenance_gap": (
                "subject00_canary_final_summary.json does not embed the "
                "step0 path/SHA; the exact pointer is resolved from the "
                "same frozen archive's training-results JSON. Any mismatch "
                "is a contract hash mismatch."
            ),
        },
        "strict_splits": {
            "camera_split_sha256": CAMERA_SPLIT_SHA,
            "pose_split_sha256": POSE_SPLIT_SHA,
            "train_camera_ids": TRAIN_CAMERAS,
            "heldout_camera_ids": HELDOUT_CAMERAS,
            "train_pose_ids": train_pose_ids,
            "heldout_pose_ids": heldout_pose_ids,
            "buffer_pose_ids": buffer_pose_ids,
            "train_pose_count": len(train_pose_ids),
            "heldout_pose_count": len(heldout_pose_ids),
            "buffer_pose_count": len(buffer_pose_ids),
            "heldout_camera_intersection": 0,
            "heldout_pose_intersection": 0,
            "buffer_pose_intersection": 0,
        },
        "availability": {
            "external_path": str(args.availability_manifest),
            "raw_file_sha256": AVAILABILITY_RAW_SHA,
            "canonical_entries_sha256": AVAILABILITY_SHA,
            "total_pair_count": len(availability["entries"]),
        },
        "training_records": {
            "manifest_path": (
                "paper_protocol/second_identity/"
                "subject00_medium_pilot_record_manifest.json"
            ),
            "manifest_file_sha256": file_sha(record_path),
            "manifest_content_sha256": record_manifest[
                "manifest_content_sha256"
            ],
            "theoretical_count": theoretical,
            "valid_count": n_valid,
            "invalid_official_missing_count": len(invalid),
            "data_order_sha256": data_order_sha,
        },
        "schedule": {
            "path": (
                "paper_protocol/second_identity/"
                "subject00_medium_pilot_schedule.json"
            ),
            "file_sha256": file_sha(schedule_path),
            "content_sha256": schedule["schedule_content_sha256"],
            "checkpoint_steps": checkpoint_steps,
        },
        "evaluation_manifest": {
            "path": (
                "paper_protocol/second_identity/"
                "subject00_canary_evaluation_manifest.json"
            ),
            "file_sha256": file_sha(evaluation_path),
            "query_order_sha256": EVAL_ORDER_SHA,
            "query_count": 96,
            "quadrant_counts": evaluation["quadrant_counts"],
            "diagnostic_query_ordinals": [1, 24, 25, 48, 49, 72, 73, 96],
        },
        "evaluation_schedule": {
            "primary_logical": {
                "step0_reused": 96,
                "quarter_new": 8,
                "half_new": 8,
                "three_quarter_new": 8,
                "final_new": 96,
                "total": 216,
            },
            "secondary_canary_step384_reused": 96,
            "new_render_count": 120,
            "reused_render_count": 192,
            "total_logical_comparison_records": 312,
            "main_visual_sheet_count": 96,
        },
        "subject02_formal_training_contract": repaired_contract[
            "subject02_formal_training_contract"
        ],
        "fixed_runtime_policies": {
            "seed": 0,
            "batch_size": 1,
            "training_background": (
                "torch.rand(3, device='cuda') once per training batch"
            ),
            "evaluation_background": [1.0, 1.0, 1.0],
            "mixed_precision": False,
            "gradient_clipping": "NONE (exact subject02 contract)",
            "lpips_activation": "step > 6000",
            "lpips_random_patch_activation": "step >= 300000",
            "renderer": "unchanged repaired-canary renderer",
            "mask_contract": "unchanged repaired-canary mask boundary handling",
            "formal_training_enabled": False,
            "one_pass_only": True,
        },
        "initial_asset_hashes": {
            "surface_face_ids": (
                "2f63d9795d3c6e98cbc9cfeb4a1b87c5589ac5e669542e011c30b343f69195a4"
            ),
            "surface_barycentric": (
                "8c5a51f95107abc16b6a403e243501bf81cd0eff2e2e58a9e2e4b1913f82aa23"
            ),
            "cached_lbs": (
                "5176b159e64f55ed956c1e83ed67ca7bcf5a006a00b67568cc13a6a81aa71899"
            ),
            "template_ply": (
                "f10a3b516e2b3a2ad38dc4924a3692b2f3e72a6cc9e66f3c0063c4e9cd210031"
            ),
            "template_faces": (
                "c69d2c49de6c68870019569924956670466224040713adef8013989be4d57a1d"
            ),
            "template_vertices": (
                "7d54bbb1b47cd83472915d576294df666518fa8bf13dd33b56b4612ceea1936c"
            ),
            "derived_manifest": (
                "de6cd51fe3f81e29139b45494860b186038b75a378b101e7428252b3b66c1af8"
            ),
            "sampler": (
                "98af26a3f578d1e0239f6ea414e3664dae10a4b60f47aef594d0978f61b9ee82"
            ),
            "inventory": {
                "gaussian_count": 200_000,
                "attachment_count": 200_000,
                "lbs_shape": [200_000, 55],
            },
        },
        "resource_gate": {
            "status": "PASS",
            "gpu": "NVIDIA GeForce RTX 4090",
            "gpu_memory_total_mib": 24_564,
            "gpu_memory_free_mib": 24_081,
            "gpu_utilization_percent": 0,
            "conflicting_gpu_process_count": 0,
            "conflicting_heavy_io_process_count": 0,
            "controller_v2_active": False,
            "free_bytes": 44_489_945_088,
            "minimum_free_bytes": 32_212_254_720,
            "free_inodes": 693_959_860,
            "checked_before_branch_worktree_output": True,
        },
        "expected_counts": {
            "training_runs": 1,
            "optimizer_created": 1,
            "training_forwards": n_valid,
            "backward_calls": n_valid,
            "optimizer_steps": n_valid,
            "step0_reused_pointer": 1,
            "new_checkpoint_writes": 4,
            "primary_logical_renders": 216,
            "secondary_reference_renders": 96,
            "new_renders": 120,
            "reused_renders": 192,
            "main_visual_sheets": 96,
        },
        "mutation_snapshot": {
            "frozen_before_after_required": True,
            "subject00_raw_mutation_expected": 0,
            "subject00_derived_asset_mutation_expected": 0,
            "subject02_mutation_expected": 0,
            "avatarrex_raw_archive_mutation_expected": 0,
            "canary_output_mutation_expected": 0,
            "historical_attempt_mutation_expected": 0,
            "split_mutation_expected": 0,
            "template_attachment_lbs_mutation_expected": 0,
            "frozen_parameter_mutation_expected": 0,
        },
        "authorization_boundary": {
            "output_root_created": 0,
            "optimizer_created": 0,
            "training_forwards": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "checkpoint_writes": 0,
            "render_calls": 0,
            "training_authorized_before_contract_commit_push_cloud_sync": False,
        },
        "limited_historical_provenance": True,
        "tuning_or_result_selection_authorized": False,
        "formal_long_training_authorized": False,
        "PAPER_FINAL": 0,
    }
    execution_path = (
        protocol_dir / "subject00_medium_pilot_execution_contract.json"
    )
    write_json(execution_path, execution_contract)
    print(
        json.dumps(
            {
                "status": "PASS",
                "theoretical_records": theoretical,
                "valid_records": n_valid,
                "invalid_official_missing_records": len(invalid),
                "data_order_sha256": data_order_sha,
                "checkpoint_steps": checkpoint_steps,
                "record_manifest_file_sha256": file_sha(record_path),
                "schedule_file_sha256": file_sha(schedule_path),
                "execution_contract_file_sha256": file_sha(execution_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
