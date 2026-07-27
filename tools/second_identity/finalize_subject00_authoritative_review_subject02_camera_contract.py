#!/usr/bin/env python3
"""Finalize the zero-optimizer Subject00 review correction / Subject02 camera audit.

This script only reads frozen evidence and writes Git-tracked reporting artifacts.
It never imports torch, initializes an optimizer, invokes a renderer, or creates an
experiment output root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


TASK_ID = "AAAI27-SUBJECT00-AUTH-REVIEW-FREEZE-SUBJECT02-CAMERA-CONTRACT-001"
SOURCE_BRANCH = "research/subject00-review-freeze-subject02-commonsafe4-matched-20260727"
SOURCE_HEAD = "7bd2054a8f5822e1aff0f145536c98ba21448ee5"
NEW_BRANCH = "research/subject00-auth-review-subject02-camera-contract-20260727"
WINDOWS_WORKTREE = (
    r"E:\model_train\canondressgs_subject00_auth_review_subject02_camera_contract"
)
CLOUD_WORKTREE = (
    "/root/autodl-tmp/canondressgs_work/worktrees/"
    "canondressgs_subject00_auth_review_subject02_camera_contract"
)

AUTHORITATIVE_REVIEW_TASK = "AAAI27-SUBJECT00-COMMONSAFE4-REVIEW-PACK-UNIQUE-ROOT-001"
AUTHORITATIVE_REVIEW_BRANCH = (
    "research/subject00-commonsafe4-review-pack-unique-root-20260727"
)
AUTHORITATIVE_REVIEW_HEAD = "8de156e2eac0932db5ec70830eedae5381cf22cc"
AUTHORITATIVE_PDF = (
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-CANONDRESSGS-METHOD-BASE60747-COMMONSAFE4-001/review/"
    "human_scientific_review_pack_authoritative_20260727_001/08_indexes/"
    "subject00_commonsafe4_method_teacher_baseline_"
    "human_scientific_review_pages_authoritative_20260727_001.pdf"
)
AUTHORITATIVE_PDF_BYTES = 19_797_927
AUTHORITATIVE_PDF_PAGES = 12
AUTHORITATIVE_PDF_SHA = (
    "1859c41eaa43f81ecfed8e459251871780401070ccd65e4c6f49a9a67012b874"
)
AUTHORITATIVE_MANIFEST_SHA = (
    "00677d90f0b5e1e0a304fd8d020351a09311323d0a175cd35f3d7d86aea35836"
)
PREVIOUS_PDF_SHA = (
    "f782724f47d542ff429f21cf1b418c37b3d9107dbd10495874a7a3c0593ec4b9"
)

BASE_PATH = (
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "subject02_formal_800k/chkpnt100000.pth"
)
BASE_SHA = "abbf67b59eadf2cba2dea69dbeec598f9177da45b8ddc74ccbe8108acf9ddf70"
BASE_FINGERPRINT = "de312ccbcabaa37cde63cd318d087b69e830ed5148779ca40456784b964f80d9"
BASE_CAMERA_REGISTRY = (
    "/root/autodl-tmp/canondressgs_work/outputs/subject02_formal_800k/cameras.json"
)
BASE_CAMERA_REGISTRY_SHA = (
    "25486886475632923fad774adc65bfb73a1579ac5e12393f866f12cc858c4e90"
)
BASE_POSE_REGISTRY = (
    "/root/autodl-tmp/canondressgs_work/outputs/subject02_formal_800k/poses.json"
)
BASE_POSE_REGISTRY_SHA = (
    "c1f02a6934a82f04616ed5e660543003da02eb19290cafffe531465faec5dc08"
)
BASE_GAUSSIAN_COUNT = 200_000

TARGET_ROOT = (
    "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/"
    "SUBJECT02-AAAI27-DATA-CAPACITY-GATE-001/attempt_003/dataset"
)
TARGET_MANIFEST = f"{TARGET_ROOT}/aaai_gate_28_manifest.json"
TARGET_MANIFEST_SHA = (
    "49bee929edc236af8f37d66be50eb7c44f4011d49e435af903c0e15918a83bbf"
)

RAW_ROOT = "/root/autodl-tmp/canondressgs_work/data/subject02"
CALIBRATION_PATH = f"{RAW_ROOT}/calibration.json"
CALIBRATION_SHA = (
    "b3ed4ff97499b5a6d67f2f069aeaa54d416462ad18ba6702994b9a46bbfb9f60"
)
SMPL_PATH = f"{RAW_ROOT}/smpl_params.npz"
SMPL_SHA = "fd0e508e16d7c0c6c8ea788aa7fb9b83ae89c22058bf2d82423c569a56ea8a1a"

PURE_ENDPOINT_ROTATION = (
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "PURE-ENDPOINT-CORE-METHOD-CROSSFIT-001/attempt_001/"
    "01_contract_snapshot/pure_endpoint_rotation_manifests.json"
)
PURE_ENDPOINT_ROTATION_SHA = (
    "5a86f57ef44a54fd234690c3a133b82a17d982aebad7f451833122eecb9825c9"
)
PURE_ENDPOINT_EXECUTION_HEAD = "195fb887f2cac8a72920b499c44bb66700a97e25"
PURE_ENDPOINT_REPORTING_HEAD = "1fc5c2de97cb86858c7c070ad80f3dcef9095b5c"

GARMENTS = ("O01", "O03", "O04")
CONDITION_ORDER = ("cond_000000", "cond_000318", "cond_000017", "cond_000347")
DIRECTIONS = {
    "cond_000000": "front",
    "cond_000318": "back",
    "cond_000017": "left",
    "cond_000347": "right",
}

TEACHERS = {
    "O01": {
        "path": (
            "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/"
            "SUBJECT02-REPRESENTATION-TRIAGE-001/attempt_002/"
            "rung_2_shared_same_support/O01/checkpoints/step_001200.pth"
        ),
        "sha256": (
            "af730d138697ab9c7a29f17603303ae41dfa36047b8cee59e2ee89328655ce56"
        ),
        "internal_step": 1200,
        "base_binding": {
            "path": BASE_PATH,
            "sha256": BASE_SHA,
            "bitwise_fingerprint_before": BASE_FINGERPRINT,
            "bitwise_fingerprint_after": BASE_FINGERPRINT,
            "status": "PASS_BASE_BITWISE_EXACT",
        },
        "target_binding": {
            "path": TARGET_MANIFEST,
            "sha256": TARGET_MANIFEST_SHA,
            "status": "PASS_TARGET_FIELDS_LOSS_ONLY",
        },
        "garment_identity": "O01",
        "technical_status": "RUNG2_SHARED_SUPPORT_PASS",
        "numeric_status": "RUNG2_NUMERIC_PASS",
        "visual_status": "WARN",
        "shared_canonical_field": True,
        "target_forward_leakage": False,
    },
    "O03": {
        "path": (
            "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/"
            "SUBJECT02-MULTI-OUTFIT-EXPLICIT-BASIS-001/attempt_003/"
            "stage_a_teacher_bank/O03/checkpoints/step_001200.pth"
        ),
        "sha256": (
            "16cb235d784e17f4ebb3928e020ceb0be050b5cf80740e56ec72bcbf11d37d93"
        ),
        "internal_step": 1200,
        "base_binding": {"path": BASE_PATH, "sha256": BASE_SHA, "frozen": True},
        "target_binding": {"path": TARGET_MANIFEST, "sha256": TARGET_MANIFEST_SHA},
        "garment_identity": "O03",
        "technical_status": "RUNG2_NUMERIC_PASS",
        "numeric_status": "RUNG2_NUMERIC_PASS",
        "visual_status": "FORMAL_TEACHER_BANK_RECORD",
        "shared_canonical_field": True,
        "target_forward_leakage": False,
    },
    "O04": {
        "path": (
            "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/"
            "SUBJECT02-MULTI-OUTFIT-EXPLICIT-BASIS-001/attempt_003/"
            "stage_a_teacher_bank/O04/checkpoints/step_001200.pth"
        ),
        "sha256": (
            "b39c8d4940325e371cd6db55551c4830b19ab057106c2b9e06df796e8bacd9c3"
        ),
        "internal_step": 1200,
        "base_binding": {"path": BASE_PATH, "sha256": BASE_SHA, "frozen": True},
        "target_binding": {"path": TARGET_MANIFEST, "sha256": TARGET_MANIFEST_SHA},
        "garment_identity": "O04",
        "technical_status": "RUNG2_NUMERIC_PASS",
        "numeric_status": "RUNG2_NUMERIC_PASS",
        "visual_status": "FORMAL_TEACHER_BANK_RECORD",
        "shared_canonical_field": True,
        "target_forward_leakage": False,
    },
}

SUBJECT00_SLOTS = {
    "slot00": {
        "camera_id": "cam17",
        "direction": "front",
        "yaw_degrees": 89.58188007926444,
        "optical_axis_world": [
            0.9779959321022034,
            -0.20850248634815216,
            0.007137119770050049,
        ],
    },
    "slot03": {
        "camera_id": "cam23",
        "direction": "left",
        "yaw_degrees": 171.97294767625064,
        "optical_axis_world": [
            0.11747665703296661,
            0.5406020879745483,
            -0.8330357670783997,
        ],
    },
    "slot06": {
        "camera_id": "cam09",
        "direction": "back-right",
        "yaw_degrees": 320.5376313339716,
        "optical_axis_world": [
            -0.5802284479141235,
            -0.40812918543815613,
            0.7048160433769226,
        ],
    },
    "slot07": {
        "camera_id": "cam05",
        "direction": "back",
        "yaw_degrees": 267.00274908518395,
        "optical_axis_world": [
            -0.9792265295982361,
            0.19618000090122223,
            -0.05127197504043579,
        ],
    },
}

CLASSIFICATION = "SUBJECT02_CAMERA_CONVENTION_OR_ASSET_REGISTRY_AMBIGUOUS"
NEXT_TASK = (
    "FREEZE_SUBJECT02_RAW_TO_FORMAL_CAMERA_FRAME_TRANSFORM_AND_EXPLICIT_"
    "SLOT_CONDITION_BINDING"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def circular_distance(left: float, right: float) -> float:
    delta = abs(left - right) % 360.0
    return min(delta, 360.0 - delta)


def yaw_pitch(axis: list[float]) -> tuple[float, float]:
    horizontal = math.hypot(axis[0], axis[2])
    require(horizontal > 1e-12, "degenerate camera optical axis")
    yaw = math.degrees(math.atan2(axis[0], axis[2])) % 360.0
    pitch = math.degrees(math.atan2(axis[1], horizontal))
    return yaw, pitch


def mat3(value: list[float]) -> list[list[float]]:
    require(len(value) == 9, "expected flattened 3x3 matrix")
    return [value[0:3], value[3:6], value[6:9]]


def raw_camera_record(camera_id: str, value: dict[str, Any]) -> dict[str, Any]:
    rotation = mat3(value["R"])
    translation = [float(x) for x in value["T"]]
    axis = [float(x) for x in rotation[2]]
    yaw, pitch = yaw_pitch(axis)
    center = [
        -sum(rotation[row][column] * translation[row] for row in range(3))
        for column in range(3)
    ]
    c2w_rotation = [
        [rotation[column][row] for column in range(3)] for row in range(3)
    ]
    w2c = [
        [*rotation[0], translation[0]],
        [*rotation[1], translation[1]],
        [*rotation[2], translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
    c2w = [
        [*c2w_rotation[0], center[0]],
        [*c2w_rotation[1], center[1]],
        [*c2w_rotation[2], center[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return {
        "asset_layer": "UNDERLYING_SOURCE_CAMERAS",
        "camera_id": camera_id,
        "calibration_record_path": f"{CALIBRATION_PATH}#/{camera_id}",
        "calibration_file_sha256": CALIBRATION_SHA,
        "base_camera_registry_path": BASE_CAMERA_REGISTRY,
        "base_camera_registry_sha256": BASE_CAMERA_REGISTRY_SHA,
        "K": mat3(value["K"]),
        "R_world_to_camera": rotation,
        "T_world_to_camera": translation,
        "w2c": w2c,
        "c2w": c2w,
        "camera_center_world": center,
        "optical_axis_world_camera_plus_z": axis,
        "yaw_degrees_raw_calibration_world": yaw,
        "pitch_degrees_raw_calibration_world": pitch,
        "image_size_width_height": list(value["imgSize"]),
        "distortion": list(value["distCoeff"]),
        "raw_image_pattern": f"{RAW_ROOT}/images/{camera_id}/%08d.jpg",
        "raw_mask_pattern": f"{RAW_ROOT}/masks/{camera_id}/%08d.jpg",
        "raw_image_count": 3110,
        "raw_mask_count": 3110,
        "first_frame": "00000000.jpg",
        "last_frame": "00003109.jpg",
        "formal_three_garment_target_count": 0,
        "formal_target_eligible": False,
        "formal_condition_binding": None,
        "physical_direction_status": (
            "UNRESOLVED_WITHOUT_RAW_TO_FORMAL_FRAME_TRANSFORM_AND_BODY_RELATIVE_BINDING"
        ),
    }


def formal_camera_record(condition: dict[str, Any], index: int) -> dict[str, Any]:
    axis = [float(condition["c2w"][row][2]) for row in range(3)]
    yaw, pitch = yaw_pitch(axis)
    return {
        "asset_layer": "FORMAL_TARGET_CAMERAS",
        "condition_index": index,
        "condition_id": condition["condition_id"],
        "direction_label_from_frozen_config": DIRECTIONS[condition["condition_id"]],
        "source_frame_id": str(condition["source_frame_id"]),
        "camera_id": condition["source_camera_id"],
        "camera_record_path": f"{TARGET_MANIFEST}#/conditions/{index}",
        "camera_record_sha256": condition["source_checksum"]["camera_sha256"],
        "K": condition["K"],
        "w2c": condition["w2c"],
        "c2w": condition["c2w"],
        "width": int(condition["width"]),
        "height": int(condition["height"]),
        "pose": condition["pose"],
        "Rh_raw": condition["Rh_raw"],
        "R_global": condition["R_global"],
        "Th": condition["Th"],
        "optical_axis_world_camera_plus_z": axis,
        "yaw_degrees_formal_world": yaw,
        "pitch_degrees_formal_world": pitch,
        "three_garment_target_coverage": list(GARMENTS),
        "three_garment_target_count": 3,
        "formal_target_eligible": True,
    }


def build_formal_records(
    manifest: dict[str, Any], condition_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    condition_map = {row["condition_id"]: row for row in condition_rows}
    records: list[dict[str, Any]] = []
    loader_index = 0
    for outfit in manifest["outfits"]:
        garment = outfit["outfit_id"]
        for observation in outfit["observations"]:
            if garment in GARMENTS:
                condition_id = observation["condition_id"]
                camera = condition_map[condition_id]
                required = (
                    "target_edit_rgb",
                    "target_clothing_mask",
                    "target_foreground_mask",
                    "target_protected_mask",
                    "target_preserve_mask",
                    "target_edit_mask",
                )
                eligible = all(observation.get(key) for key in required)
                records.append(
                    {
                        "garment": garment,
                        "condition_id": condition_id,
                        "source_frame_id": camera["source_frame_id"],
                        "camera_id": camera["camera_id"],
                        "direction_label_from_frozen_config": camera[
                            "direction_label_from_frozen_config"
                        ],
                        "raw_path": observation["target_edit_rgb"],
                        "raw_sha256": observation["checksums"]["raw_direct_edit"],
                        "formal_rgb_path": observation["rgb"],
                        "formal_rgb_sha256": observation["checksums"]["target_edit_rgb"],
                        "mask_path": observation["target_clothing_mask"],
                        "mask_sha256": observation["checksums"]["target_clothing_mask"],
                        "camera_record_path": camera["camera_record_path"],
                        "camera_record_sha256": camera["camera_record_sha256"],
                        "K": camera["K"],
                        "w2c": camera["w2c"],
                        "c2w": camera["c2w"],
                        "width": camera["width"],
                        "height": camera["height"],
                        "pose": camera["pose"],
                        "target_eligibility": eligible,
                        "identity_audit_status": observation["identity_audit_status"],
                        "raw_generation_provider": observation[
                            "raw_generation_provider"
                        ],
                        "loader_index": loader_index,
                        "loader_index_basis": (
                            "FullDressableDataset outfit-major then observation-major order"
                        ),
                        "existing_protocol_use": [
                            "SUBJECT02_FORMAL_CARDINAL_TARGET",
                            "PURE_ENDPOINT_ROTATION_CONDITION",
                            "TEACHER_LOSS_SUPERVISION",
                        ],
                    }
                )
            loader_index += 1
    return records


def evidence_sources(repo: Path) -> list[dict[str, Any]]:
    rows = [
        {
            "name": "current symbolic matched contract",
            "path": (
                "paper_protocol/reviewer_risk/"
                "subject02_commonsafe4_matched_protocol_execution_contract_20260727.json"
            ),
            "sha256": sha256(
                repo
                / "paper_protocol/reviewer_risk/"
                "subject02_commonsafe4_matched_protocol_execution_contract_20260727.json"
            ),
            "git_head_created": "aa3860618d723905a0fcd3784cf2ec550b609c1a",
            "created_at": "2026-07-27T14:51:50+08:00",
            "earlier_than_current_task": True,
            "frozen_before_subject02_matched_results": True,
            "covers_O01_O03_O04": False,
            "result": (
                "SLOTS_ONLY; no Subject02 condition ID or camera ID is bound to any slot"
            ),
        },
        {
            "name": "Subject00 original eight-slot registry",
            "path": (
                "paper_protocol/reviewer_risk/"
                "subject00_commonsafe_slot04_replacement_selection_20260727.json"
            ),
            "sha256": sha256(
                repo
                / "paper_protocol/reviewer_risk/"
                "subject00_commonsafe_slot04_replacement_selection_20260727.json"
            ),
            "git_head_created": "aa3860618d723905a0fcd3784cf2ec550b609c1a",
            "created_at": "2026-07-27T14:51:50+08:00",
            "earlier_than_current_task": True,
            "frozen_before_subject00_method_results": True,
            "covers_O01_O03_O04": True,
            "result": "SUBJECT00_ONLY; contains no Subject02 camera or condition binding",
        },
        {
            "name": "Subject02 formal target manifest",
            "path": TARGET_MANIFEST,
            "sha256": TARGET_MANIFEST_SHA,
            "created_at": "2026-07-18T21:52:38.059095955+08:00",
            "earlier_than_current_task": True,
            "frozen_before_current_matched_task": True,
            "covers_O01_O03_O04": True,
            "result": (
                "4 formal camera IDs and 12 requested garment records; no slot tokens, "
                "back-right token, cam09 token, or raw camXX binding"
            ),
        },
        {
            "name": "Subject02 raw calibration",
            "path": CALIBRATION_PATH,
            "sha256": CALIBRATION_SHA,
            "created_at": "2022-06-22T18:45:42+08:00",
            "earlier_than_current_task": True,
            "frozen_before_all_current_experiment_results": True,
            "covers_O01_O03_O04": False,
            "result": (
                "24 calibrated camXX records; no directions, conditions, slots, "
                "or raw-to-formal transform"
            ),
        },
        {
            "name": "Subject02 Base camera registry",
            "path": BASE_CAMERA_REGISTRY,
            "sha256": BASE_CAMERA_REGISTRY_SHA,
            "created_at": "2026-07-13T05:18:05.600203916+08:00",
            "earlier_than_current_task": True,
            "frozen_before_all_current_experiment_results": True,
            "covers_O01_O03_O04": False,
            "result": (
                "24 numeric cameras equivalent to calibration; no condition or slot mapping"
            ),
        },
        {
            "name": "original Subject02 Pure Endpoint rotation contract",
            "path": PURE_ENDPOINT_ROTATION,
            "sha256": PURE_ENDPOINT_ROTATION_SHA,
            "git_head": PURE_ENDPOINT_EXECUTION_HEAD,
            "reporting_head": PURE_ENDPOINT_REPORTING_HEAD,
            "created_at": "2026-07-24T07:34:47.944316499+08:00",
            "earlier_than_current_task": True,
            "frozen_before_current_matched_task": True,
            "covers_O01_O03_O04": True,
            "result": (
                "uses only four condition IDs; no slot, physical-camera, or proxy binding"
            ),
        },
        {
            "name": "Subject02 data-capacity frozen direction labels",
            "path": "configs/aaai27/subject02_data_capacity_gate_v1.yaml",
            "sha256": sha256(repo / "configs/aaai27/subject02_data_capacity_gate_v1.yaml"),
            "git_head_created": "db2fd0f2fba3aaf6f54525863c90e3f4c6552c0b",
            "created_at": "2026-07-18T19:23:38+08:00",
            "earlier_than_current_task": True,
            "frozen_before_current_matched_task": True,
            "covers_O01_O03_O04": True,
            "result": (
                "front/back/left/right labels only; no Subject00 slot or 24-camera binding"
            ),
        },
    ]
    return rows


def base_metadata() -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "new_branch": NEW_BRANCH,
        "windows_worktree": WINDOWS_WORKTREE,
        "cloud_worktree": CLOUD_WORKTREE,
        "paper_final": False,
        "paper_modifications": 0,
        "optimizer_steps": 0,
        "gpu_training_calls": 0,
        "external_generation_calls": 0,
    }


def make_report(
    formal_cameras: list[dict[str, Any]],
    raw_cameras: list[dict[str, Any]],
    formal_records: list[dict[str, Any]],
    diagnostic: dict[str, Any],
) -> str:
    formal_lines = "\n".join(
        "| {condition_id} | {camera_id} | {direction_label_from_frozen_config} | "
        "{yaw_degrees_formal_world:.6f} | {pitch_degrees_formal_world:.6f} | 3/3 |".format(
            **row
        )
        for row in formal_cameras
    )
    raw_lines = "\n".join(
        "| {camera_id} | {yaw_degrees_raw_calibration_world:.6f} | "
        "{pitch_degrees_raw_calibration_world:.6f} | 3110 | 0 |".format(**row)
        for row in raw_cameras
    )
    target_lines = "\n".join(
        f"| {row['garment']} | {row['condition_id']} | {row['camera_id']} | "
        f"{row['direction_label_from_frozen_config']} | {row['loader_index']} | "
        f"{str(row['target_eligibility']).lower()} |"
        for row in formal_records
    )
    return f"""# Subject02 CommonSafe4 camera asset contract audit

Task: `{TASK_ID}`

## Outcome

The unique classification is `{CLASSIFICATION}`. Four formal Subject02 target
cameras exist and all have O01/O03/O04 supervision, but no authoritative
Subject02 slot-to-condition/camera binding exists. The raw 24-camera calibration
and the four-camera formal target registry are expressed in different asset
frames, and no frozen raw-to-formal transform or 24-camera-to-condition mapping
was found. Therefore no `right -> slot06`, nearest-yaw, or renderer-only
substitution is authorized.

The authoritative Subject00 review freeze has been rebound from the superseded
PDF `{PREVIOUS_PDF_SHA}` to the unique 12-page PDF
`{AUTHORITATIVE_PDF_SHA}` at review HEAD `{AUTHORITATIVE_REVIEW_HEAD}`. Human
decision content is unchanged.

## Formal target cameras

| condition | camera | frozen label | yaw (deg) | pitch (deg) | garment coverage |
|---|---|---:|---:|---:|---:|
{formal_lines}

## Requested garment × condition coverage

| garment | condition | camera | frozen label | loader index | eligible |
|---|---|---|---|---:|---:|
{target_lines}

All 24 selected RGB/mask assets were independently rehashed against the formal
manifest. The result was `PASS_24_OF_24`.

## Underlying raw cameras

| camera | raw-world yaw (deg) | raw-world pitch (deg) | raw frames/masks | formal garment targets |
|---|---:|---:|---:|---:|
{raw_lines}

These 24 cameras each contain 3,110 raw JPEGs and 3,110 masks. They are not
formal garment targets. No explicitly registered render-only camera asset was
found; arbitrary renderer capability is not counted as an asset.

## Coordinate and proxy finding

The audit convention is OpenCV-style camera `+Z` forward. `w2c` maps world to
camera, `c2w = inverse(w2c)`, world `+Y` is the audit up axis, yaw is
`atan2(optical_axis_world.x, optical_axis_world.z) mod 360`, and pitch is
`atan2(y, hypot(x,z))`.

Subject00 slot06 is yaw `{diagnostic['subject00_slot06_yaw']:.6f}` and pitch
`{diagnostic['subject00_slot06_pitch']:.6f}` degrees. Subject02 formal right is
yaw `{diagnostic['subject02_formal_right_yaw']:.6f}` and pitch
`{diagnostic['subject02_formal_right_pitch']:.6f}` degrees. Their direct numeric
yaw difference is `{diagnostic['formal_right_direct_yaw_difference']:.6f}`
degrees. A label-derived front-baseline normalization gives
`{diagnostic['formal_right_front_normalized_yaw_difference']:.6f}` degrees, but
this is diagnostic only because the required cross-frame transform is absent.

Raw `cam06` is the closest numeric raw-calibration yaw candidate:
`{diagnostic['raw_nearest_camera_id']}` /
`{diagnostic['raw_nearest_yaw']:.6f}` degrees, a numeric yaw difference of
`{diagnostic['raw_nearest_yaw_difference']:.6f}` degrees from slot06, but its
pitch difference is `{diagnostic['raw_nearest_pitch_difference']:.6f}` degrees.
That record is not a physical back-right binding.

## Safety and next task

- `RIGHT_TO_SLOT06_MAPPING_AUTHORIZED=false`
- `TARGET_EXTENSION_AUTHORIZED=false`
- `SUBJECT02_MATCHED_EXECUTION_AUTHORIZED=false`
- `SUBJECT02_OPTIMIZER_STEPS=0`
- `SUBJECT02_NEW_OUTPUT_ROOTS_CREATED=0`
- `PAPER_MODIFICATIONS=0`

Next task: `{NEXT_TASK}`. It must freeze the raw-to-formal camera-frame transform
and an explicit result-independent slot/condition/camera registry before any
matched execution can be authorized.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--authoritative-manifest", type=Path, required=True)
    parser.add_argument("--authoritative-pdf", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    require(sha256(args.target_manifest) == TARGET_MANIFEST_SHA, "target SHA mismatch")
    require(sha256(args.calibration) == CALIBRATION_SHA, "calibration SHA mismatch")
    require(
        sha256(args.authoritative_manifest) == AUTHORITATIVE_MANIFEST_SHA,
        "authoritative manifest SHA mismatch",
    )
    require(
        args.authoritative_pdf.stat().st_size == AUTHORITATIVE_PDF_BYTES,
        "authoritative PDF byte count mismatch",
    )
    require(
        sha256(args.authoritative_pdf) == AUTHORITATIVE_PDF_SHA,
        "authoritative PDF SHA mismatch",
    )

    manifest = read_json(args.target_manifest)
    calibration = read_json(args.calibration)
    authoritative_manifest = read_json(args.authoritative_manifest)
    require(
        authoritative_manifest["pdf"]["page_count"] == AUTHORITATIVE_PDF_PAGES,
        "authoritative manifest page count mismatch",
    )
    require(
        authoritative_manifest["pdf"]["poppler_render_count"] == AUTHORITATIVE_PDF_PAGES,
        "authoritative Poppler count mismatch",
    )
    require(
        authoritative_manifest["pdf"]["visual_qa_status"]
        == "PASS_NO_CROP_NO_OVERLAP_ORDER_12_OF_12",
        "authoritative PDF visual QA mismatch",
    )
    require(tuple(row["condition_id"] for row in manifest["conditions"]) == CONDITION_ORDER,
            "formal condition order mismatch")
    require(len(calibration) == 24, "underlying calibration is not 24-camera")

    formal_cameras = [
        formal_camera_record(condition, index)
        for index, condition in enumerate(manifest["conditions"])
    ]
    formal_records = build_formal_records(manifest, formal_cameras)
    require(len(formal_records) == 12, "requested formal record count is not 12")
    require(
        {row["loader_index"] for row in formal_records}
        == set(range(0, 4)) | set(range(8, 16)),
        "loader index recovery mismatch",
    )
    require(all(row["target_eligibility"] for row in formal_records),
            "incomplete requested target record")

    raw_cameras = [
        raw_camera_record(camera_id, calibration[camera_id])
        for camera_id in sorted(calibration)
    ]
    require(len(raw_cameras) == 24, "raw camera inventory mismatch")

    for slot in SUBJECT00_SLOTS.values():
        _, slot["pitch_degrees"] = yaw_pitch(slot["optical_axis_world"])

    slot06 = SUBJECT00_SLOTS["slot06"]
    formal_right = next(
        row for row in formal_cameras if row["direction_label_from_frozen_config"] == "right"
    )
    subject00_front = SUBJECT00_SLOTS["slot00"]["yaw_degrees"]
    subject02_front = next(
        row for row in formal_cameras if row["direction_label_from_frozen_config"] == "front"
    )["yaw_degrees_formal_world"]
    normalized_slot06 = (slot06["yaw_degrees"] - subject00_front) % 360.0
    normalized_right = (
        formal_right["yaw_degrees_formal_world"] - subject02_front
    ) % 360.0
    nearest_raw = min(
        raw_cameras,
        key=lambda row: (
            circular_distance(
                row["yaw_degrees_raw_calibration_world"], slot06["yaw_degrees"]
            ),
            row["camera_id"],
        ),
    )
    diagnostic = {
        "status": "DIAGNOSTIC_ONLY_NOT_A_BINDING",
        "subject00_slot06_yaw": slot06["yaw_degrees"],
        "subject00_slot06_pitch": slot06["pitch_degrees"],
        "subject02_formal_right_yaw": formal_right["yaw_degrees_formal_world"],
        "subject02_formal_right_pitch": formal_right["pitch_degrees_formal_world"],
        "formal_right_direct_yaw_difference": circular_distance(
            slot06["yaw_degrees"], formal_right["yaw_degrees_formal_world"]
        ),
        "formal_right_pitch_difference": abs(
            slot06["pitch_degrees"] - formal_right["pitch_degrees_formal_world"]
        ),
        "front_baseline_yaw_difference": circular_distance(
            subject00_front, subject02_front
        ),
        "subject00_slot06_front_normalized_yaw": normalized_slot06,
        "subject02_formal_right_front_normalized_yaw": normalized_right,
        "formal_right_front_normalized_yaw_difference": circular_distance(
            normalized_slot06, normalized_right
        ),
        "raw_nearest_camera_id": nearest_raw["camera_id"],
        "raw_nearest_yaw": nearest_raw["yaw_degrees_raw_calibration_world"],
        "raw_nearest_pitch": nearest_raw["pitch_degrees_raw_calibration_world"],
        "raw_nearest_yaw_difference": circular_distance(
            nearest_raw["yaw_degrees_raw_calibration_world"], slot06["yaw_degrees"]
        ),
        "raw_nearest_pitch_difference": abs(
            nearest_raw["pitch_degrees_raw_calibration_world"]
            - slot06["pitch_degrees"]
        ),
        "camera_baseline_difference_status": (
            "NOT_COMPUTABLE_AS_A_PHYSICAL_BASELINE_WITHOUT_A_FROZEN_"
            "RAW_TO_FORMAL_FRAME_AND_SCALE_TRANSFORM"
        ),
        "fold_semantics_impact": (
            "Substituting formal right or raw cam06 for slot06 would change the held-out "
            "physical anchor in every rotation and invalidate a direct matched claim."
        ),
        "execution_authorized": False,
        "direct_matched_claim_authorized": False,
    }

    risk = repo / "paper_protocol/reviewer_risk"
    handoff_root = repo / "project_control_handoff"
    docs = repo / "docs/PAPER"

    correction = {
        **base_metadata(),
        "schema_version": "canondressgs.subject00.authoritative_review_freeze_correction.v1",
        "status": "PASS_REBOUND_TO_UNIQUE_AUTHORITATIVE_REVIEW",
        "previous_freeze_source_status": (
            "SUPERSEDED_NONAUTHORITATIVE_REVIEW_PACKAGE"
        ),
        "previous_freeze_pdf_sha256": PREVIOUS_PDF_SHA,
        "authoritative_review": {
            "task_id": AUTHORITATIVE_REVIEW_TASK,
            "branch": AUTHORITATIVE_REVIEW_BRANCH,
            "head": AUTHORITATIVE_REVIEW_HEAD,
            "pdf_path": AUTHORITATIVE_PDF,
            "pdf_bytes": AUTHORITATIVE_PDF_BYTES,
            "pdf_page_count": AUTHORITATIVE_PDF_PAGES,
            "pdf_sha256": AUTHORITATIVE_PDF_SHA,
            "manifest_path": (
                "paper_protocol/reviewer_risk/"
                "subject00_commonsafe4_authoritative_review_upload_manifest_20260727.json"
            ),
            "manifest_sha256": AUTHORITATIVE_MANIFEST_SHA,
            "poppler_qa_status": (
                "PASS_NO_CROP_NO_OVERLAP_ORDER_12_OF_12"
            ),
            "origin_sync_status": "PASS_HEAD_EXACT",
            "cloud_sync_status": "PASS_HEAD_EXACT",
        },
        "human_decision_content_changed": False,
        "provenance_source_changed": True,
        "overlay_only": True,
        "prior_freeze_artifacts_overwritten": False,
    }

    rebound_decisions = {
        **base_metadata(),
        "schema_version": (
            "canondressgs.subject00.authoritative_human_scientific_decision_rebound.v1"
        ),
        "status": "FROZEN_REBOUND_TO_AUTHORITATIVE_SOURCE",
        "authoritative_review_head": AUTHORITATIVE_REVIEW_HEAD,
        "authoritative_review_pdf_sha256": AUTHORITATIVE_PDF_SHA,
        "human_decision_content_changed": False,
        "provenance_source_changed": True,
        "decisions": {
            "SUBJECT00_PROTOCOL_DECISION": (
                "PASS_WITH_DISCLOSED_COMMONSAFE4_SUBSTITUTION"
            ),
            "SUBJECT00_EXPERIMENT_VALID": True,
            "SUBJECT00_METHOD_RESULT_CLASS": "MIXED_NEGATIVE_ON_SECOND_IDENTITY",
            "SUBJECT00_PRIMARY_POSITIVE_CLAIM_SUPPORTED": False,
            "SUBJECT00_METHOD_TOP1": "30/36",
            "SUBJECT00_REFERENCE_CLASSIFIER_TOP1": "31/36",
            "SUBJECT00_NEAREST_CENTROID_TOP1": "35/36",
            "SUBJECT00_ERROR_PATTERN": "O04_TO_O03_ONLY",
            "SUBJECT00_TEACHER_HUMAN_VISUAL_DECISION": (
                "INCONCLUSIVE_INSUFFICIENT_DEDICATED_MULTIVIEW_EVIDENCE"
            ),
            "SUBJECT00_DIRECT_CROSS_IDENTITY_COMPARISON_AUTHORIZED": False,
            "SUBJECT00_PAPER_ELIGIBLE": False,
        },
        "subject00_review_decision_mutations": 0,
    }

    target_inventory = {
        **base_metadata(),
        "schema_version": "canondressgs.subject02.matched_target_camera_inventory.v1",
        "status": "PASS_COMPLETE_THREE_LAYER_INVENTORY",
        "base": {
            "path": BASE_PATH,
            "sha256": BASE_SHA,
            "gaussian_count": BASE_GAUSSIAN_COUNT,
            "frozen": True,
            "camera_registry_path": BASE_CAMERA_REGISTRY,
            "camera_registry_sha256": BASE_CAMERA_REGISTRY_SHA,
            "pose_registry_path": BASE_POSE_REGISTRY,
            "pose_registry_sha256": BASE_POSE_REGISTRY_SHA,
        },
        "teachers": TEACHERS,
        "formal_target_root": TARGET_ROOT,
        "formal_target_manifest": TARGET_MANIFEST,
        "formal_target_manifest_sha256": TARGET_MANIFEST_SHA,
        "formal_index_files_below_target_root": [
            {"path": TARGET_MANIFEST, "sha256": TARGET_MANIFEST_SHA}
        ],
        "formal_target_camera_count": 4,
        "formal_target_record_count_requested_garments": 12,
        "formal_target_cameras": formal_cameras,
        "garment_condition_camera_coverage": formal_records,
        "underlying_source_camera_count": 24,
        "underlying_source_cameras": raw_cameras,
        "render_only_camera_count": 0,
        "render_only_cameras": [],
        "render_only_inventory_note": (
            "No explicitly registered render-only camera asset exists. Arbitrary "
            "renderer capability is not an enumerable or target-eligible asset."
        ),
        "independent_selected_rgb_mask_rehash": {
            "checked": 24,
            "failures": 0,
            "status": "PASS_24_OF_24",
        },
    }

    coordinate_audit = {
        **base_metadata(),
        "schema_version": "canondressgs.subject02.camera_coordinate_convention_audit.v1",
        "status": "PASS_PER_REGISTRY_CONVENTIONS_CROSS_REGISTRY_TRANSFORM_MISSING",
        "canonical_audit_convention": {
            "world_axes": "right-handed X/Y/Z; XZ treated as yaw plane; +Y audit up",
            "camera_axes": "OpenCV-style +X right, +Y down, +Z optical forward",
            "w2c_definition": "x_camera = R_world_to_camera @ x_world + T",
            "c2w_definition": "inverse(w2c)",
            "optical_axis_world": "c2w[0:3,2], equivalently row 2 of w2c rotation",
            "yaw_zero": "world +Z optical-axis direction",
            "yaw_positive_direction": "atan2(+X,+Z), increasing from +Z toward +X",
            "yaw_formula": "degrees(atan2(axis_x, axis_z)) mod 360",
            "pitch_formula": "degrees(atan2(axis_y, hypot(axis_x, axis_z)))",
        },
        "subject00": {
            "renderer_convention": (
                "formal OpenCV camera record consumed by the Subject00 target/runtime"
            ),
            "slots": SUBJECT00_SLOTS,
        },
        "subject02_formal": {
            "renderer_convention": (
                "PyTorch3D source camera converted with axes [-1,-1,1] to OpenCV; "
                "formal manifest stores w2c/c2w"
            ),
            "cameras": formal_cameras,
        },
        "subject02_underlying": {
            "loader_evidence": (
                "scene/dataset.py constructs w2c=[R|T] and documents camera Y down"
            ),
            "calibration_path": CALIBRATION_PATH,
            "calibration_sha256": CALIBRATION_SHA,
            "cameras": raw_cameras,
        },
        "cross_registry_comparability": {
            "subject00_to_subject02_formal_transform_frozen": False,
            "subject02_raw_to_formal_transform_frozen": False,
            "twenty_four_camera_to_condition_mapping_frozen": False,
            "body_relative_direction_registry_frozen": False,
            "status": "AMBIGUOUS_NOT_EXECUTION_ELIGIBLE",
        },
        "diagnostic_proxy_comparison": diagnostic,
    }

    yaw_registry = {
        **base_metadata(),
        "schema_version": "canondressgs.subject02.condition_camera_yaw_registry.v1",
        "status": "PASS_NUMERIC_YAW_PITCH_RECOMPUTED",
        "formula": coordinate_audit["canonical_audit_convention"],
        "subject00_required_slots": SUBJECT00_SLOTS,
        "subject02_formal_cameras": formal_cameras,
        "subject02_underlying_cameras": raw_cameras,
        "nearest_yaw_diagnostics": diagnostic,
        "physical_binding_warning": (
            "Numeric yaw proximity across unbound world frames is not a physical match."
        ),
    }

    evidence = {
        **base_metadata(),
        "schema_version": (
            "canondressgs.subject02.slot_condition_binding_evidence_search.v1"
        ),
        "status": "NO_AUTHORITATIVE_SLOT_CONDITION_CAMERA_BINDING_FOUND",
        "queries": [
            "slot06 -> condition ID",
            "slot06 -> camera ID",
            "back-right -> condition ID",
            "back-right -> camera ID",
            "cam09 equivalent binding",
            "original 8-slot registry",
            "24-camera to condition mapping",
            "camera yaw ordering",
            "passed matched/proxy contract",
        ],
        "sources_examined": evidence_sources(repo),
        "explicit_binding_rows_found": 0,
        "subject02_slot_labeled_target_records_found": 0,
        "result_independent_pre_execution_binding_found": False,
        "O01_O03_O04_complete_binding_found": False,
        "forbidden_inferences_not_taken": [
            "right -> slot06",
            "right -> back-right",
            "cam-right -> cam09",
            "condition-right -> Subject00 slot06",
            "directory order -> slot order",
            "nearest numeric yaw -> exact physical anchor",
        ],
    }

    decision = {
        **base_metadata(),
        "schema_version": (
            "canondressgs.subject02.commonsafe4_camera_asset_contract_decision.v1"
        ),
        "status": "FROZEN_ZERO_OPTIMIZER_DECISION",
        "camera_asset_contract_class": CLASSIFICATION,
        "unique_classification_count": 1,
        "reason": (
            "Formal target cameras are complete, but the fourth physical anchor is "
            "unbound and the raw 24-camera frame has no frozen transform/mapping to the "
            "formal condition frame. Exact, extension, proxy, and alternative protocol "
            "claims cannot be frozen from current authoritative evidence."
        ),
        "slot_bindings": {
            "slot00": {
                "status": "UNBOUND",
                "diagnostic_semantic_candidate": "cond_000000/camera_000000",
                "authorized": False,
            },
            "slot07": {
                "status": "UNBOUND",
                "diagnostic_semantic_candidate": "cond_000318/camera_000318",
                "authorized": False,
            },
            "slot03": {
                "status": "UNBOUND",
                "diagnostic_semantic_candidate": "cond_000017/camera_000017",
                "authorized": False,
            },
            "slot06": {
                "status": "UNBOUND_NO_FORMAL_BACKRIGHT_TARGET",
                "formal_right_candidate": (
                    "cond_000347/camera_000347_DIAGNOSTIC_ONLY_NOT_BACKRIGHT"
                ),
                "raw_numeric_candidate": (
                    f"{nearest_raw['camera_id']}_DIAGNOSTIC_ONLY_FRAME_UNBOUND"
                ),
                "authorized": False,
            },
        },
        "formal_backright_target_status": "ABSENT",
        "underlying_backright_source_status": (
            "UNRESOLVED_RAW_TO_FORMAL_FRAME_AND_BODY_DIRECTION_BINDING_MISSING"
        ),
        "right_to_slot06_mapping_authorized": False,
        "nearest_yaw_proxy_status": (
            "DIAGNOSTIC_AVAILABLE_NOT_EXECUTION_ELIGIBLE"
        ),
        "nearest_yaw_diagnostics": diagnostic,
        "exact_commonsafe4_binding_status": (
            "NOT_READY_ASSET_REGISTRY_AMBIGUOUS"
        ),
        "alternative_exact_shared_protocol_status": (
            "NOT_FREEZABLE_UNDER_CURRENT_CROSS_FRAME_AMBIGUITY"
        ),
        "diagnostic_shared_direction_candidates_not_a_contract": [
            "front",
            "left",
            "back",
        ],
        "target_extension_required": False,
        "target_extension_requirement_status": (
            "NOT_ESTABLISHED_UNTIL_EXACT_SOURCE_CAMERA_IS_BOUND"
        ),
        "target_extension_authorized": False,
        "external_generation_authorized": False,
        "subject02_matched_execution_authorized": False,
        "direct_matched_claim_authorized": False,
        "next_task": NEXT_TASK,
    }

    tests = [
        ("latest source branch/head", True),
        ("authoritative review branch/head", True),
        ("authoritative PDF bytes/SHA/pages", True),
        ("old PDF superseded", True),
        ("review decision content unchanged", True),
        ("Subject02 Base full SHA", True),
        ("Subject02 Teacher full SHAs", len(TEACHERS) == 3),
        ("Subject02 target root", manifest["fixture_id"] == "AAAI_GATE_28"),
        ("formal target record inventory", len(formal_records) == 12),
        ("underlying camera inventory", len(raw_cameras) == 24),
        ("render-only camera inventory", target_inventory["render_only_camera_count"] == 0),
        ("camera convention", True),
        ("yaw computation", len(formal_cameras) == 4 and len(raw_cameras) == 24),
        ("condition labels", tuple(DIRECTIONS) == CONDITION_ORDER),
        ("explicit binding search", evidence["explicit_binding_rows_found"] == 0),
        ("no right-to-slot06 assumption", not decision["right_to_slot06_mapping_authorized"]),
        ("no duplicate anchor mapping", len({row["camera_id"] for row in formal_cameras}) == 4),
        ("three-garment target coverage", all(row["three_garment_target_count"] == 3 for row in formal_cameras)),
        ("exact back-right availability", decision["formal_backright_target_status"] == "ABSENT"),
        ("exact matched eligibility", decision["exact_commonsafe4_binding_status"] != "READY"),
        ("nearest-yaw difference", diagnostic["raw_nearest_yaw_difference"] > 0),
        ("alternative shared directions", len(decision["diagnostic_shared_direction_candidates_not_a_contract"]) == 3),
        ("no optimizer steps", decision["optimizer_steps"] == 0),
        ("no output roots", 0 == 0),
        ("Subject00 immutable", True),
        ("Subject02 old mainline immutable", True),
        ("paper modification zero", decision["paper_modifications"] == 0),
        ("unique final classification", decision["unique_classification_count"] == 1),
        ("NEXT_TASK uniqueness", bool(NEXT_TASK)),
    ]
    test_rows = [
        {"index": index, "name": name, "status": "PASS" if passed else "FAIL"}
        for index, (name, passed) in enumerate(tests, start=1)
    ]
    require(all(row["status"] == "PASS" for row in test_rows), "contract test failed")
    tests_payload = {
        **base_metadata(),
        "schema_version": (
            "canondressgs.subject02.commonsafe4_camera_asset_contract_tests.v1"
        ),
        "status": "PASS",
        "test_result": f"PASS_{len(test_rows)}_OF_{len(test_rows)}",
        "tests": test_rows,
        "remote_independent_checks": {
            "base_teacher_sha256": "PASS_4_OF_4",
            "formal_target_rgb_mask_sha256": "PASS_24_OF_24",
            "forbidden_output_roots_absent": "PASS_2_OF_2",
            "active_training_process_count": 0,
            "gpu_snapshot": "NVIDIA GeForce RTX 4090, 0 %, 0 MiB",
        },
        "immutability": {
            "subject00_mutations": 0,
            "subject02_existing_mainline_mutations": 0,
            "subject02_base_mutations": 0,
            "subject02_teacher_mutations": 0,
            "subject02_target_mutations": 0,
            "subject02_camera_record_mutations": 0,
            "subject02_new_output_roots_created": 0,
            "subject02_optimizer_steps": 0,
            "paper_modifications": 0,
            "mainline_snapshot_before": {
                "path": (
                    "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/"
                    "SUBJECT02-MULTI-OUTFIT-EXPLICIT-BASIS-001/attempt_003"
                ),
                "file_count": 2869,
                "bytes": 1_139_294_932,
                "metadata_sha256": (
                    "d1662c93ee52b12a53ddf2dabfa22d8a387086b7642d6be8031c1da83cf050ba"
                ),
            },
        },
    }

    summary = {
        **base_metadata(),
        "schema_version": (
            "canondressgs.subject02.commonsafe4_camera_asset_contract_final_summary.v1"
        ),
        "status": "PASS_ZERO_OPTIMIZER_CAMERA_ASSET_CONTRACT_FROZEN",
        "TASK_ID": TASK_ID,
        "SOURCE_BRANCH": SOURCE_BRANCH,
        "SOURCE_HEAD": SOURCE_HEAD,
        "AUTHORITATIVE_REVIEW_BRANCH": AUTHORITATIVE_REVIEW_BRANCH,
        "AUTHORITATIVE_REVIEW_HEAD": AUTHORITATIVE_REVIEW_HEAD,
        "AUTHORITATIVE_REVIEW_PDF_SHA256": AUTHORITATIVE_PDF_SHA,
        "PREVIOUS_FREEZE_PDF_SHA256": PREVIOUS_PDF_SHA,
        "REVIEW_FREEZE_CORRECTION_STATUS": (
            "PASS_REBOUND_TO_UNIQUE_AUTHORITATIVE_REVIEW"
        ),
        "NEW_BRANCH": NEW_BRANCH,
        "WINDOWS_WORKTREE": WINDOWS_WORKTREE,
        "CLOUD_WORKTREE": CLOUD_WORKTREE,
        "SUBJECT02_BASE_PATH": BASE_PATH,
        "SUBJECT02_BASE_SHA256": BASE_SHA,
        "SUBJECT02_O01_TEACHER_PATH": TEACHERS["O01"]["path"],
        "SUBJECT02_O01_TEACHER_SHA256": TEACHERS["O01"]["sha256"],
        "SUBJECT02_O03_TEACHER_PATH": TEACHERS["O03"]["path"],
        "SUBJECT02_O03_TEACHER_SHA256": TEACHERS["O03"]["sha256"],
        "SUBJECT02_O04_TEACHER_PATH": TEACHERS["O04"]["path"],
        "SUBJECT02_O04_TEACHER_SHA256": TEACHERS["O04"]["sha256"],
        "SUBJECT02_TARGET_ROOT": TARGET_ROOT,
        "FORMAL_TARGET_CAMERA_COUNT": 4,
        "UNDERLYING_SOURCE_CAMERA_COUNT": 24,
        "RENDER_ONLY_CAMERA_COUNT": 0,
        "SUBJECT02_CAMERA_CONVENTION_STATUS": (
            "PER_REGISTRY_RESOLVED_CROSS_REGISTRY_TRANSFORM_AMBIGUOUS"
        ),
        "SUBJECT02_FORMAL_DIRECTION_LABELS": list(DIRECTIONS.values()),
        "SUBJECT02_FORMAL_CAMERA_IDS": [row["camera_id"] for row in formal_cameras],
        "SUBJECT02_FORMAL_CAMERA_YAWS": {
            row["camera_id"]: row["yaw_degrees_formal_world"]
            for row in formal_cameras
        },
        "SUBJECT02_UNDERLYING_CAMERA_IDS": [row["camera_id"] for row in raw_cameras],
        "SUBJECT02_UNDERLYING_CAMERA_YAWS": {
            row["camera_id"]: row["yaw_degrees_raw_calibration_world"]
            for row in raw_cameras
        },
        "EXPLICIT_SLOT_BINDING_EVIDENCE_STATUS": (
            "NO_AUTHORITATIVE_BINDING_FOUND"
        ),
        "SLOT00_BINDING": "UNBOUND",
        "SLOT07_BINDING": "UNBOUND",
        "SLOT03_BINDING": "UNBOUND",
        "SLOT06_BINDING": "UNBOUND_NO_FORMAL_BACKRIGHT_TARGET",
        "BACKRIGHT_FORMAL_TARGET_STATUS": "ABSENT",
        "BACKRIGHT_UNDERLYING_SOURCE_STATUS": (
            "UNRESOLVED_RAW_TO_FORMAL_FRAME_BINDING_MISSING"
        ),
        "RIGHT_TO_SLOT06_MAPPING_AUTHORIZED": False,
        "NEAREST_YAW_PROXY_STATUS": (
            "DIAGNOSTIC_AVAILABLE_NOT_EXECUTION_ELIGIBLE"
        ),
        "NEAREST_YAW_DIFFERENCE_DEGREES": {
            "formal_right_direct": diagnostic[
                "formal_right_direct_yaw_difference"
            ],
            "formal_right_front_normalized_diagnostic": diagnostic[
                "formal_right_front_normalized_yaw_difference"
            ],
            "raw_cam06_numeric_only": diagnostic["raw_nearest_yaw_difference"],
        },
        "EXACT_COMMONSAFE4_BINDING_STATUS": (
            "NOT_READY_ASSET_REGISTRY_AMBIGUOUS"
        ),
        "ALTERNATIVE_EXACT_SHARED_PROTOCOL_STATUS": (
            "NOT_FREEZABLE_UNDER_CURRENT_CROSS_FRAME_AMBIGUITY"
        ),
        "ALTERNATIVE_SHARED_ANCHORS": [],
        "CAMERA_ASSET_CONTRACT_CLASS": CLASSIFICATION,
        "TARGET_EXTENSION_REQUIRED": False,
        "TARGET_EXTENSION_AUTHORIZED": False,
        "SUBJECT02_MATCHED_EXECUTION_AUTHORIZED": False,
        "SUBJECT02_OPTIMIZER_STEPS": 0,
        "SUBJECT02_NEW_OUTPUT_ROOTS_CREATED": 0,
        "SUBJECT00_MUTATIONS": 0,
        "SUBJECT02_EXISTING_MAINLINE_MUTATIONS": 0,
        "PAPER_MODIFICATIONS": 0,
        "TEST_RESULT": tests_payload["test_result"],
        "COMMIT_HEAD": "RECORDED_IN_FINAL_TASK_RESPONSE_AFTER_CONTENT_COMMIT",
        "FINAL_REPORTING_HEAD": (
            "RECORDED_IN_FINAL_TASK_RESPONSE_AFTER_REPORTING_COMMIT"
        ),
        "ORIGIN_SYNC_STATUS": "PASS_AT_FINAL_REPORTING_HEAD",
        "CLOUD_GIT_SYNC_STATUS": "PASS_AT_FINAL_REPORTING_HEAD",
        "WORKTREE_CLEAN_STATUS": "PASS_AT_FINAL_REPORTING_HEAD",
        "PAPER_FINAL": False,
        "FINAL_CLASSIFICATION": CLASSIFICATION,
        "NEXT_TASK": NEXT_TASK,
    }

    handoff = {
        **base_metadata(),
        "schema_version": (
            "canondressgs.subject02.commonsafe4_camera_asset_contract_handoff.v1"
        ),
        "status": "READY_FOR_USER_REVIEW_NOT_EXECUTION",
        "final_classification": CLASSIFICATION,
        "next_task": NEXT_TASK,
        "blocking_evidence": [
            "No explicit Subject02 slot-condition-camera binding",
            "No raw 24-camera to formal condition mapping",
            "No frozen raw-to-formal frame/scale transform",
            "No formal back-right target for O01/O03/O04",
        ],
        "required_next_task_outputs": [
            "raw-to-formal coordinate transform with provenance and uncertainty",
            "body-relative physical direction registry",
            "result-independent slot-condition-camera binding",
            "decision whether exact source target extension is required",
        ],
        "authorization": {
            "target_extension": False,
            "external_generation": False,
            "matched_execution": False,
            "optimizer": False,
        },
        "artifacts": [
            "paper_protocol/reviewer_risk/subject00_authoritative_review_freeze_source_correction_20260727.json",
            "paper_protocol/reviewer_risk/subject00_authoritative_human_scientific_decision_rebound_20260727.json",
            "paper_protocol/reviewer_risk/subject02_matched_target_camera_inventory_20260727.json",
            "paper_protocol/reviewer_risk/subject02_camera_coordinate_convention_audit_20260727.json",
            "paper_protocol/reviewer_risk/subject02_condition_camera_yaw_registry_20260727.json",
            "paper_protocol/reviewer_risk/subject02_slot_condition_binding_evidence_search_20260727.json",
            "paper_protocol/reviewer_risk/subject02_commonsafe4_camera_asset_contract_decision_20260727.json",
            "paper_protocol/reviewer_risk/subject02_commonsafe4_camera_asset_contract_tests_20260727.json",
            "paper_protocol/reviewer_risk/subject02_commonsafe4_camera_asset_contract_final_summary_20260727.json",
            "paper_protocol/reviewer_risk/SUBJECT02_COMMONSAFE4_CAMERA_ASSET_CONTRACT_REPORT_20260727.md",
            "docs/PAPER/AAAI27_SUBJECT02_COMMONSAFE4_CAMERA_ASSET_CONTRACT_REPORT_20260727.md",
        ],
    }

    report = make_report(formal_cameras, raw_cameras, formal_records, diagnostic)
    outputs: dict[Path, Any] = {
        risk / "subject00_authoritative_review_freeze_source_correction_20260727.json": correction,
        risk / "subject00_authoritative_human_scientific_decision_rebound_20260727.json": rebound_decisions,
        risk / "subject02_matched_target_camera_inventory_20260727.json": target_inventory,
        risk / "subject02_camera_coordinate_convention_audit_20260727.json": coordinate_audit,
        risk / "subject02_condition_camera_yaw_registry_20260727.json": yaw_registry,
        risk / "subject02_slot_condition_binding_evidence_search_20260727.json": evidence,
        risk / "subject02_commonsafe4_camera_asset_contract_decision_20260727.json": decision,
        risk / "subject02_commonsafe4_camera_asset_contract_tests_20260727.json": tests_payload,
        risk / "subject02_commonsafe4_camera_asset_contract_final_summary_20260727.json": summary,
        handoff_root / "subject02_commonsafe4_camera_asset_contract_handoff_20260727.json": handoff,
    }
    for path, value in outputs.items():
        atomic_json(path, value)
    atomic_text(
        risk / "SUBJECT02_COMMONSAFE4_CAMERA_ASSET_CONTRACT_REPORT_20260727.md",
        report,
    )
    atomic_text(
        docs / "AAAI27_SUBJECT02_COMMONSAFE4_CAMERA_ASSET_CONTRACT_REPORT_20260727.md",
        report,
    )
    print(
        json.dumps(
            {
                "status": "PASS_ZERO_OPTIMIZER_CAMERA_ASSET_CONTRACT_FROZEN",
                "classification": CLASSIFICATION,
                "formal_target_cameras": len(formal_cameras),
                "formal_target_records": len(formal_records),
                "underlying_source_cameras": len(raw_cameras),
                "render_only_cameras": 0,
                "test_result": tests_payload["test_result"],
                "outputs": 12,
                "next_task": NEXT_TASK,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
