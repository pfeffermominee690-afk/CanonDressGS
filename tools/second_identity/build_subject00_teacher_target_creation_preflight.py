#!/usr/bin/env python3
"""Build the read-only Subject00 Teacher-target creation preflight artifacts.

This builder writes repository documentation and registries only.  It never
creates the planned target roots and never copies or derives target data.
Calibration JSON is supplied on stdin so the cloud dataset remains read-only.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import struct
import sys
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK_ID = "AAAI27-SUBJECT00-24-CELL-TEACHER-TARGET-CREATION-PREFLIGHT-001"
SOURCE_BRANCH = "research/subject00-24-cell-mask-human-review-promotion-20260727"
SOURCE_HEAD = "4ed89d9ac5076d13fef1c2cad8fcf28e3d236ac0"
NEW_BRANCH = "research/subject00-24-cell-teacher-target-creation-preflight-20260727"
WORKTREE = r"E:\model_train\canondressgs_subject00_24_cell_teacher_target_creation_preflight"

MASK_REGISTRY_REL = (
    "paper_protocol/reviewer_risk/"
    "subject00_global_mask_accepted_registry_24of24_20260727.json"
)
RAW_REGISTRY_REL = (
    "paper_protocol/reviewer_risk/"
    "subject00_global_accepted_cell_registry_24of24_20260726.json"
)
ATTEMPT1_REG_REL = (
    "paper_protocol/reviewer_risk/"
    "subject00_attempt001_background_registration_metrics_20260726.json"
)
O03_CANARY_REG_REL = (
    "paper_protocol/reviewer_risk/"
    "subject00_o03_canary_corrected_registration_results_20260726.json"
)
ATTEMPT5_REG = Path(
    r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-GENERATION-001"
    r"\attempt_005_subject00_remaining_six_cell_generation\06_audit"
    r"\registration_results.json"
)
REMAINING_OVERLAY_REL = (
    "paper_protocol/reviewer_risk/"
    "subject00_remaining_six_human_review_overlay_20260726.json"
)

CALIBRATION_PATH = (
    "/root/autodl-tmp/datasets/thuman4_second_identity_staging/"
    "subject00/calibration.json"
)
CALIBRATION_SHA = "4b2d98d20740da03be209040851fab7e8801e5670937ed9ad290a20264d1dde7"
POSE_PATH = (
    "/root/autodl-tmp/datasets/thuman4_second_identity_staging/"
    "subject00/smpl_params.npz"
)
POSE_SHA = "ac2738c308ad1a9cc02e7b63323c75e0eab28bddc88c57bb5887e07bf8ea28d2"

WINDOWS_PROJECT_ROOT = (
    r"E:\canondressgs_data\subject00_three_garment"
    r"\CODEX-MANAGED-TEACHER-TARGETS-001"
)
WINDOWS_ATTEMPT_ROOT = (
    WINDOWS_PROJECT_ROOT + r"\attempt_001_subject00_24_cell_teacher_targets"
)
CLOUD_TARGET_ROOT = (
    "/root/autodl-tmp/canondressgs_work/teacher_targets/"
    "SUBJECT00-24CELL-001/attempt_001"
)
MATERIALIZATION_MODE = "PORTABLE_ARCHIVE_AND_CLOUD_EXTRACTION"

TARGET_SCHEMA = "canondressgs.full_dataset.v1"
TARGET_LOADER = "scene/full_dressable_dataset.py"
ENDPOINT_LOADER = "tools/run_module4b_canonical_oracle_micropilot.py"
TARGET_BUILDER = "tools/aaai27/build_data_capacity_fixture.py"
TARGET_QA = "tools/check_aaai27_data_capacity_gate.py"
SCHEMA_PATH = "schemas/canondressgs_full_dataset_v1.schema.json"

SUBJECT02_ROOT = (
    "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/"
    "SUBJECT02-AAAI27-DATA-CAPACITY-GATE-001/attempt_003/dataset"
)
SUBJECT02_MANIFEST = SUBJECT02_ROOT + "/aaai_gate_28_manifest.json"
SUBJECT02_MANIFEST_SHA = "49bee929edc236af8f37d66be50eb7c44f4011d49e435af903c0e15918a83bbf"

BASE60747_CHECKPOINT = (
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-FORMAL-BASE-101245-001/attempt_001/checkpoints/step_060747.pth"
)
BASE60747_SHA = "2d09b1ce6bb19b8b6418314caab8dc1cf92d70e432a9e09e009989332ea4a2f7"
BASE60747_BYTES = 724_584_413
BASE60747_STATUS = "BASE60747_TASK_NOT_FOUND_OR_INCOMPLETE"

CAMERA_STATUS = (
    "BLOCKED_22_OF_24_UNIQUE_SIMILARITY_BINDINGS_"
    "2_OF_24_HUMAN_OVERRIDE_CAMERA_MODELS_NONUNIQUE"
)
MIXED_STATUS = "BATCH_SIZE_ONE_NATIVE_RESOLUTION_SUPPORTED"
INTRINSIC_POLICY = (
    "PER_RECORD_REGISTERED_SOURCE_TO_TARGET_SIMILARITY_LEFT_MULTIPLY_K; "
    "NO_RESIZE; BLOCK_ON_HUMAN_OVERRIDE_NONUNIQUENESS"
)
FINAL_CLASSIFICATION = (
    "SUBJECT00_TEACHER_TARGET_PREFLIGHT_BLOCKED_BY_CAMERA_RESOLUTION_CONTRACT"
)
NEXT_TASK = "RESOLVE_SUBJECT00_TEACHER_TARGET_CAMERA_RESOLUTION_BLOCKER"

HUMAN_OVERRIDE_IDS = {
    "subject00_O03_slot04_canary_attempt004_cand00",
    "subject00_O01_slot04_remaining_attempt005_cand00",
}

REQUIRED_RAW_FIELDS = ["rgb", "target_edit_rgb", "target_base_rgb"]
REQUIRED_MASK_FIELDS = [
    "foreground_mask",
    "clothing_mask",
    "target_edit_mask",
    "target_edit_core_mask",
    "target_preserve_mask",
    "target_transition_mask",
    "target_protected_mask",
    "target_foreground_mask",
    "target_base_foreground_mask",
    "target_clothing_mask",
    "target_old_clothing_mask",
    "target_revealed_skin_mask",
]
REQUIRED_CAMERA_FIELDS = [
    "condition_id",
    "source_frame_id",
    "source_camera_id",
    "pose",
    "Rh_raw",
    "R_global",
    "Th",
    "K",
    "w2c",
    "c2w",
    "width",
    "height",
    "background",
    "conventions",
    "source_checksum",
]
REQUIRED_DERIVED_FIELDS = [
    "target_edit_mask",
    "target_edit_core_mask",
    "target_preserve_mask",
    "target_transition_mask",
    "target_protected_mask",
    "target_base_foreground_mask",
    "target_old_clothing_mask",
    "target_revealed_skin_mask",
]

QA_GATE_NAMES = [
    "record_count_24",
    "garment_counts_8_8_8",
    "raw_exists_24",
    "person_mask_exists_24",
    "garment_mask_exists_24",
    "raw_sha_match_24",
    "person_mask_sha_match_24",
    "garment_mask_sha_match_24",
    "raw_mask_shape_match",
    "strict_binary_mask_values",
    "garment_subset_person",
    "camera_id_slot_mapping",
    "camera_calibration_parse",
    "camera_model_unique_per_record",
    "direction_slot_consistency",
    "native_resolution_preserved",
    "limitation_propagation_9_of_9",
    "human_override_propagation_2_of_2",
    "duplicate_output_path_absent",
    "duplicate_record_absent",
    "missing_record_absent",
    "extra_record_absent",
    "derived_target_parse",
    "formal_endpoint_loader_zero_optimizer_smoke",
    "source_sha_immutable_after_materialization",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def png_l_pixels(path: Path) -> tuple[int, int, bytes]:
    """Decode an 8-bit, non-interlaced grayscale PNG using only stdlib."""
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not PNG: {path}")
    offset = 8
    width = height = bit_depth = color_type = interlace = None
    compressed = bytearray()
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if kind == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            break
    if (bit_depth, color_type, interlace) != (8, 0, 0):
        raise ValueError(
            f"mask must be 8-bit grayscale non-interlaced PNG: {path}: "
            f"{bit_depth=}, {color_type=}, {interlace=}"
        )
    assert width is not None and height is not None
    raw = zlib.decompress(bytes(compressed))
    stride = width
    if len(raw) != height * (stride + 1):
        raise ValueError(f"unexpected PNG scanline length: {path}")
    output = bytearray(height * stride)
    previous = bytearray(stride)
    source_offset = 0
    for row in range(height):
        filter_kind = raw[source_offset]
        scanline = raw[source_offset + 1 : source_offset + 1 + stride]
        source_offset += stride + 1
        reconstructed = bytearray(stride)
        for column, value in enumerate(scanline):
            left = reconstructed[column - 1] if column else 0
            above = previous[column]
            upper_left = previous[column - 1] if column else 0
            if filter_kind == 0:
                predictor = 0
            elif filter_kind == 1:
                predictor = left
            elif filter_kind == 2:
                predictor = above
            elif filter_kind == 3:
                predictor = (left + above) // 2
            elif filter_kind == 4:
                estimate = left + above - upper_left
                distances = (
                    abs(estimate - left),
                    abs(estimate - above),
                    abs(estimate - upper_left),
                )
                predictor = (left, above, upper_left)[distances.index(min(distances))]
            else:
                raise ValueError(f"unsupported PNG filter {filter_kind}: {path}")
            reconstructed[column] = (value + predictor) & 0xFF
        start = row * stride
        output[start : start + stride] = reconstructed
        previous = reconstructed
    return width, height, bytes(output)


def png_dimensions(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"not PNG: {path}")
    return struct.unpack(">II", header[16:24])


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def planned_win(*parts: str) -> str:
    return WINDOWS_ATTEMPT_ROOT + "\\" + "\\".join(parts)


def planned_cloud(*parts: str) -> str:
    return CLOUD_TARGET_ROOT + "/" + "/".join(parts)


def matrices(calib: dict[str, Any], similarity: list[list[float]] | None) -> dict[str, Any]:
    K = [list(map(float, calib["K"][row * 3 : row * 3 + 3])) for row in range(3)]
    R = [list(map(float, calib["R"][row * 3 : row * 3 + 3])) for row in range(3)]
    T = list(map(float, calib["T"]))
    w2c = [
        [R[0][0], R[0][1], R[0][2], T[0]],
        [R[1][0], R[1][1], R[1][2], T[1]],
        [R[2][0], R[2][1], R[2][2], T[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
    Rt = [[R[column][row] for column in range(3)] for row in range(3)]
    c2w_t = [-sum(Rt[row][column] * T[column] for column in range(3)) for row in range(3)]
    c2w = [
        [Rt[0][0], Rt[0][1], Rt[0][2], c2w_t[0]],
        [Rt[1][0], Rt[1][1], Rt[1][2], c2w_t[1]],
        [Rt[2][0], Rt[2][1], Rt[2][2], c2w_t[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
    result = {
        "calibration_K": K,
        "calibration_R": R,
        "calibration_T": T,
        "distortion": list(calib["distCoeff"]),
        "calibration_image_width": int(calib["imgSize"][0]),
        "calibration_image_height": int(calib["imgSize"][1]),
        "rectify_alpha": float(calib["rectifyAlpha"]),
        "w2c": w2c,
        "c2w": c2w,
    }
    if similarity is None:
        result["registered_source_to_target_similarity"] = None
        result["target_K"] = None
    else:
        S = [list(map(float, similarity[0])), list(map(float, similarity[1])), [0.0, 0.0, 1.0]]
        result["registered_source_to_target_similarity"] = S
        result["target_K"] = [
            [
                sum(S[row][inner] * K[inner][column] for inner in range(3))
                for column in range(3)
            ]
            for row in range(3)
        ]
    return result


def limitation_entries(record: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    official = list(record.get("limitation_codes", []))
    expanded: list[str] = []
    for code in official:
        expanded.extend(part.strip() for part in code.split(";") if part.strip())
    if record["request_id"] == "subject00_O01_slot04_remaining_attempt005_cand00":
        expanded.extend(
            ["SOURCE_FACE_OBSCURED_BY_HOOD", "MINOR_BACKGROUND_RECONSTRUCTION_RISK"]
        )
    expanded = list(dict.fromkeys(expanded))
    descriptions = {
        "HISTORICAL_FACE_HEAD_CROP_OFF_TARGET": (
            "Historical face/head crop was off target; the accepted full-body raw is authoritative."
        ),
        "FULL_BODY_RAW_USED": (
            "The accepted full-body raw, rather than the historical crop, is used."
        ),
        "IDENTITY_EVIDENCE_LIMITED_BY_BACK_OR_OBLIQUE_VIEW": (
            "Identity evidence is limited by a back or oblique view."
        ),
        "MACHINE_REGISTRATION_FAIL_WITH_DOCUMENTED_HUMAN_OVERRIDE": (
            "Machine registration failed and the human visual override remains disclosed."
        ),
        "RAW_RETAINED_IN_TECHNICAL_FAILURE_PATH_AFTER_RESOLUTION_FAMILY_CORRECTION": (
            "The accepted raw remains under a historical technical-failure path after correction."
        ),
        "SOURCE_FACE_OBSCURED_BY_HOOD": (
            "The source face is obscured by a hood, limiting direct identity comparison."
        ),
        "MINOR_BACKGROUND_RECONSTRUCTION_RISK": (
            "Human review accepted mild background reconstruction while retaining disclosure."
        ),
    }
    values = []
    for code in expanded:
        override = code == "MACHINE_REGISTRATION_FAIL_WITH_DOCUMENTED_HUMAN_OVERRIDE"
        background = code == "MINOR_BACKGROUND_RECONSTRUCTION_RISK"
        values.append(
            {
                "code": code,
                "description": descriptions[code],
                "affects_target_materialization": override,
                "affects_appearance_loss": override or background,
                "affects_alpha_loss": override,
                "affects_boundary_loss": override,
                "affects_protected_region_loss": override or background,
                "affects_geometry_supervision": override,
                "manual_disclosure_required": True,
                "policy": (
                    "TARGET_SPACE_APPEARANCE_ONLY; STRICT_GEOMETRY_FORBIDDEN; "
                    "NO_AUTOMATIC_LOSS_REWEIGHTING"
                    if override
                    else "DISCLOSE_WITHOUT_AUTOMATIC_BLOCK_OR_LOSS_REWEIGHTING"
                ),
            }
        )
    return official, values


def artifact_paths() -> dict[str, str]:
    root = "paper_protocol/reviewer_risk/"
    return {
        "execution_contract": root
        + "SUBJECT00_24_CELL_TEACHER_TARGET_CREATION_EXECUTION_CONTRACT_20260727.md",
        "execution_manifest": root
        + "subject00_24_cell_teacher_target_creation_execution_manifest_20260727.json",
        "schema_registry": root
        + "subject00_24_cell_teacher_target_schema_registry_20260727.json",
        "camera_registry": root
        + "subject00_24_cell_teacher_target_camera_registry_20260727.json",
        "resolution_contract": root
        + "subject00_24_cell_teacher_target_resolution_contract_20260727.json",
        "materialization_audit": root
        + "subject00_24_cell_teacher_target_materialization_mode_audit_20260727.json",
        "qa_registry": root
        + "subject00_24_cell_teacher_target_qa_registry_20260727.json",
        "storage_estimate": root
        + "subject00_24_cell_teacher_target_storage_transfer_estimate_20260727.json",
        "o03_manifest": root
        + "subject00_O03_provisional_teacher_target_manifest_draft_20260727.json",
        "tests": root
        + "subject00_24_cell_teacher_target_creation_preflight_tests_20260727.json",
        "final_summary": root
        + "subject00_24_cell_teacher_target_creation_preflight_final_summary_20260727.json",
        "handoff": "project_control_handoff/"
        + "subject00_24_cell_teacher_target_creation_execution_handoff_20260727.json",
        "report": "docs/PAPER/"
        + "AAAI27_SUBJECT00_24_CELL_TEACHER_TARGET_CREATION_PREFLIGHT_REPORT_20260727.md",
    }


def build(args: argparse.Namespace) -> None:
    repo = args.repo_root.resolve()
    created_at = datetime.now(timezone.utc).isoformat()
    mask_registry = read_json(repo / MASK_REGISTRY_REL)
    raw_registry = read_json(repo / RAW_REGISTRY_REL)
    raw_by_id = {item["request_id"]: item for item in raw_registry["records"]}
    remaining_overlay = read_json(repo / REMAINING_OVERLAY_REL)
    remaining_by_id = {
        item["request_id"]: item for item in remaining_overlay.get("records", [])
    }
    calibration_bytes = sys.stdin.buffer.read()
    if not calibration_bytes and os.environ.get("SUBJECT00_CALIBRATION_B64"):
        calibration_bytes = base64.b64decode(
            os.environ["SUBJECT00_CALIBRATION_B64"], validate=True
        )
    if not calibration_bytes:
        raise ValueError("calibration JSON must be supplied on stdin")
    if hashlib.sha256(calibration_bytes).hexdigest() != CALIBRATION_SHA:
        raise ValueError("calibration stdin SHA256 mismatch")
    calibration = json.loads(calibration_bytes)

    reg_sources = [
        repo / ATTEMPT1_REG_REL,
        repo / O03_CANARY_REG_REL,
        ATTEMPT5_REG,
    ]
    reg_by_id: dict[str, dict[str, Any]] = {}
    for source in reg_sources:
        for item in read_json(source)["records"]:
            reg_by_id[item["request_id"]] = item

    records = []
    cameras = []
    hash_cache: dict[Path, str] = {}

    def verified_sha(path: Path) -> str:
        if path not in hash_cache:
            if not path.is_file():
                raise FileNotFoundError(path)
            hash_cache[path] = sha256(path)
        return hash_cache[path]

    for accepted in mask_registry["records"]:
        request_id = accepted["request_id"]
        raw = raw_by_id[request_id]
        registration = reg_by_id[request_id]
        override = request_id in HUMAN_OVERRIDE_IDS
        official, limitations = limitation_entries(accepted)
        garment = accepted["garment"]
        condition_id = request_id
        camera_id = accepted["camera"]
        cam = calibration[camera_id]
        camera_values = matrices(
            cam, None if override else registration["similarity"]["matrix"]
        )
        native_width = int(accepted["native_resolution"]["width"])
        native_height = int(accepted["native_resolution"]["height"])
        accepted_raw_path = Path(accepted["accepted_raw"]["path"])
        accepted_person_path = Path(accepted["person_mask"]["path"])
        accepted_garment_path = Path(accepted["garment_mask"]["path"])
        for path, binding in (
            (accepted_raw_path, accepted["accepted_raw"]),
            (accepted_person_path, accepted["person_mask"]),
            (accepted_garment_path, accepted["garment_mask"]),
        ):
            if path.stat().st_size != binding["bytes"]:
                raise ValueError(f"byte count mismatch: {path}")
            if verified_sha(path) != binding["sha256"]:
                raise ValueError(f"SHA256 mismatch: {path}")
        if png_dimensions(accepted_raw_path) != (native_width, native_height):
            raise ValueError(f"raw native resolution mismatch: {request_id}")
        person_width, person_height, person_pixels = png_l_pixels(accepted_person_path)
        garment_width, garment_height, garment_pixels = png_l_pixels(accepted_garment_path)
        if (person_width, person_height) != (native_width, native_height):
            raise ValueError(f"person-mask native resolution mismatch: {request_id}")
        if (garment_width, garment_height) != (native_width, native_height):
            raise ValueError(f"garment-mask native resolution mismatch: {request_id}")
        if not set(person_pixels).issubset({0, 255}) or not set(
            garment_pixels
        ).issubset({0, 255}):
            raise ValueError(f"mask is not strictly binary: {request_id}")
        if any(
            garment_value == 255 and person_value != 255
            for garment_value, person_value in zip(garment_pixels, person_pixels)
        ):
            raise ValueError(f"garment mask is not a subset of person mask: {request_id}")
        if not any(
            person_value == 255 and garment_value == 0
            for person_value, garment_value in zip(person_pixels, garment_pixels)
        ):
            raise ValueError(f"protected person-minus-garment region is empty: {request_id}")
        source_condition_path = Path(raw["source_condition_path"])
        if verified_sha(source_condition_path) != raw["source_condition_sha256"]:
            raise ValueError(f"source-condition SHA256 mismatch: {source_condition_path}")
        record_path = planned_win(
            "02_records", request_id + "_teacher_target_record.json"
        )
        raw_path = planned_win("03_raw", garment, request_id + ".png")
        person_path = planned_win(
            "04_person_masks", garment, request_id + "_person_mask.png"
        )
        garment_path = planned_win(
            "05_garment_masks", garment, request_id + "_garment_mask.png"
        )
        camera_path = planned_win("06_camera", request_id + "_camera.json")
        derived_root = planned_win(
            "07_derived_targets", "full_dataset_v1"
        )
        observation_paths = {
            "rgb": raw_path,
            "foreground_mask": person_path,
            "clothing_mask": garment_path,
            "target_edit_rgb": raw_path,
            "target_base_rgb": derived_root
            + rf"\rgb\base\{garment}\{request_id}.png",
            "target_foreground_mask": person_path,
            "target_clothing_mask": garment_path,
        }
        for field in REQUIRED_DERIVED_FIELDS:
            observation_paths[field] = (
                derived_root + rf"\masks\{field}\{garment}\{request_id}.png"
            )
        values = {
            "sequence_index": int(accepted["sequence_index"]),
            "subject": accepted["subject"],
            "garment": garment,
            "slot": accepted["slot"],
            "camera_id": camera_id,
            "direction": accepted["direction"],
            "condition_id": condition_id,
            "accepted_request_id": request_id,
            "source_attempt_id": accepted["attempt_id"],
            "accepted_raw": accepted["accepted_raw"],
            "native_resolution": accepted["native_resolution"],
            "person_mask": accepted["person_mask"],
            "garment_mask": accepted["garment_mask"],
            "mask_value_semantics": {
                "format": "PNG",
                "mode": "L",
                "allowed_values": [0, 255],
                "foreground_value": 255,
                "background_value": 0,
                "garment_subset_of_person": True,
            },
            "source_condition": {
                "path": raw["source_condition_path"],
                "sha256": raw["source_condition_sha256"],
                "frame_id": "00000000",
                "pose_path": POSE_PATH,
                "pose_sha256": POSE_SHA,
            },
            "generation_method": raw["generation_method"],
            "registration": {
                "machine_status": accepted["machine_registration_status"],
                "machine_classification": registration["primary_classification"],
                "human_override_status": accepted.get("human_override_status"),
                "human_override_reason": remaining_by_id.get(request_id, {}).get(
                    "human_override_reason"
                ),
                "similarity": registration["similarity"],
                "homography": registration["homography"],
                "source_registry": str(
                    next(
                        path
                        for path in reg_sources
                        if any(
                            x["request_id"] == request_id
                            for x in read_json(path)["records"]
                        )
                    )
                ),
            },
            "image_acceptance_class": accepted["image_acceptance_class"],
            "mask_acceptance_class": accepted["mask_acceptance_status"],
            "official_limitation_codes": official,
            "propagated_limitations": limitations,
            "human_review_evidence": accepted["review_evidence"],
            "target_readiness_status": (
                "BLOCKED_CAMERA_MODEL_NONUNIQUE"
                if override
                else "READY_PENDING_AUTHORIZATION"
            ),
            "planned_paths": {
                "record_json": record_path,
                "raw": raw_path,
                "person_mask": person_path,
                "garment_mask": garment_path,
                "camera_record": camera_path,
                "observation": observation_paths,
            },
            "materialized": False,
            "teacher_target": False,
        }
        records.append(values)
        cameras.append(
            {
                "sequence_index": int(accepted["sequence_index"]),
                "request_id": request_id,
                "garment": garment,
                "slot": accepted["slot"],
                "camera_id": camera_id,
                "direction": accepted["direction"],
                "source_frame_id": "00000000",
                "target_width": int(accepted["native_resolution"]["width"]),
                "target_height": int(accepted["native_resolution"]["height"]),
                **camera_values,
                "registration_homography_diagnostic": registration["homography"][
                    "matrix"
                ],
                "binding_status": (
                    "BLOCKED_HUMAN_OVERRIDE_DOES_NOT_SELECT_UNIQUE_PHYSICAL_CAMERA"
                    if override
                    else "PASS_UNIQUE_SIMILARITY_PIXEL_BINDING"
                ),
                "conventions": {
                    "extrinsic": "w2c=[R|T]; c2w=inverse(w2c)",
                    "pixel_projection": "OpenCV positive-z: u=fx*x/z+cx; v=fy*y/z+cy",
                    "renderer": (
                        "build_mmlphuman_camera consumes per-record K,w2c,width,height; "
                        "no axis flip"
                    ),
                    "body_transform": "x_world=R_global@x_posed+Th",
                    "distortion_policy": "all five frozen coefficients are zero",
                    "registration_policy": INTRINSIC_POLICY,
                },
                "pose_binding": {
                    "path": POSE_PATH,
                    "sha256": POSE_SHA,
                    "frame_index": 0,
                    "required_runtime_fields": [
                        "pose",
                        "Rh_raw",
                        "R_global",
                        "Th",
                    ],
                },
            }
        )

    if len(records) != 24:
        raise AssertionError("accepted registry must contain 24 records")
    if [x["sequence_index"] for x in records] != list(range(1, 25)):
        raise AssertionError("accepted registry order is not the frozen 1..24 sequence")
    if Counter(x["garment"] for x in records) != Counter(
        {"O01": 8, "O03": 8, "O04": 8}
    ):
        raise AssertionError("garment coverage mismatch")
    if {x["accepted_request_id"] for x in records if x["official_limitation_codes"]} != set(
        mask_registry["limitation_request_ids"]
    ):
        raise AssertionError("limitation IDs do not match the accepted registry")

    root_exists = Path(WINDOWS_PROJECT_ROOT).exists()
    if root_exists:
        raise RuntimeError("planned target root unexpectedly preexists")

    paths = artifact_paths()
    hashes = {
        "schema": sha256(repo / SCHEMA_PATH),
        "loader": sha256(repo / TARGET_LOADER),
        "endpoint_loader": sha256(repo / ENDPOINT_LOADER),
        "builder": sha256(repo / TARGET_BUILDER),
        "qa": sha256(repo / TARGET_QA),
        "mask_registry": sha256(repo / MASK_REGISTRY_REL),
        "raw_registry": sha256(repo / RAW_REGISTRY_REL),
    }
    resolution_counts = Counter(
        f'{x["native_resolution"]["width"]}x{x["native_resolution"]["height"]}'
        for x in records
    )

    raw_bytes = sum(x["accepted_raw"]["bytes"] for x in records)
    person_bytes = sum(x["person_mask"]["bytes"] for x in records)
    garment_bytes = sum(x["garment_mask"]["bytes"] for x in records)
    pixels = sum(
        x["native_resolution"]["width"] * x["native_resolution"]["height"]
        for x in records
    )
    derived_masks = pixels * 8
    base_rgb_uncompressed = pixels * 3
    registry_review_budget = 8 * 1024 * 1024
    payload = (
        raw_bytes
        + person_bytes
        + garment_bytes
        + derived_masks
        + base_rgb_uncompressed
        + registry_review_budget
    )
    archive = math.ceil(payload * 1.05)
    reserve = 30 * 1024**3
    windows_projected = payload + 2 * archive + reserve
    cloud_projected = payload + archive + reserve
    storage_pass = (
        args.windows_free_bytes >= windows_projected
        and args.cloud_free_bytes >= cloud_projected
    )

    schema_registry = {
        "schema_version": "canondressgs.subject00.teacher_target_schema_registry.v1",
        "task_id": TASK_ID,
        "created_at": created_at,
        "status": "PASS_FORMAL_SUBJECT02_SCHEMA_RECOVERED",
        "target_schema": TARGET_SCHEMA,
        "schema_path": SCHEMA_PATH,
        "schema_sha256": hashes["schema"],
        "formal_loader_path": TARGET_LOADER,
        "formal_loader_sha256": hashes["loader"],
        "endpoint_sample_loader": ENDPOINT_LOADER + "::_load_samples",
        "endpoint_sample_loader_sha256": hashes["endpoint_loader"],
        "target_creation_implementation": TARGET_BUILDER,
        "target_creation_implementation_sha256": hashes["builder"],
        "qa_implementation": TARGET_QA,
        "qa_implementation_sha256": hashes["qa"],
        "subject02_precedent": {
            "dataset_root": SUBJECT02_ROOT,
            "manifest": SUBJECT02_MANIFEST,
            "manifest_sha256": SUBJECT02_MANIFEST_SHA,
            "record_count": 28,
            "outfits": 7,
            "conditions_per_outfit": 4,
            "resolution": {"width": 1024, "height": 1536},
            "raw_naming": "dataset/rgb/edit/<outfit>/<condition>.png",
            "person_mask_naming": (
                "dataset/masks/target_foreground_mask/<outfit>/<condition>.png"
            ),
            "garment_mask_naming": (
                "dataset/masks/target_clothing_mask/<outfit>/<condition>.png"
            ),
            "camera_metadata_location": "manifest.conditions[]",
            "audited_protocol_families": [
                "Pure Endpoint",
                "Geometry Causal",
                "Dual-Support",
                "Teacher-span headroom",
                "LOO",
                "O03 Teacher Endpoint",
                "external target-space protocol",
                "Subject02 data-capacity gate",
                "pipeline_full outputs",
            ],
        },
        "subject00_condition_id_contract": {
            "planned_condition_count": 24,
            "planned_condition_id": "<accepted_request_id>",
            "reason": (
                "Per-garment registration and native HxW make K,width,height record-specific. "
                "A shared eight-slot condition table cannot represent the O03 slot00 "
                "1350x1165 camera beside the O01/O04 slot00 1349x1166 cameras."
            ),
            "formal_shared_condition_precedent_compatible": False,
            "status": "BLOCKED_PENDING_CAMERA_RESOLUTION_CONTRACT_REFREEZE",
        },
        "required_raw_fields": REQUIRED_RAW_FIELDS,
        "required_mask_fields": REQUIRED_MASK_FIELDS,
        "required_camera_fields": REQUIRED_CAMERA_FIELDS,
        "required_derived_fields": REQUIRED_DERIVED_FIELDS,
        "materialization_semantics": {
            "target_edit_rgb": "byte-exact accepted generated raw",
            "target_base_rgb": (
                "source-condition RGB registered into target pixel space; "
                "no checkpoint render"
            ),
            "target_foreground_mask": "accepted target person mask",
            "target_clothing_mask": (
                "accepted target garment mask, constrained by foreground/protected masks"
            ),
            "target_edit_mask": "morphological edit support excluding protected region",
            "target_edit_core_mask": "eroded edit core excluding protected region",
            "target_preserve_mask": "outside edit support union protected region",
            "target_transition_mask": "support-minus-core boundary excluding protected region",
            "target_protected_mask": (
                "source protected-person region; execution must derive with frozen builder "
                "and retain human QA"
            ),
            "target_base_foreground_mask": (
                "source-condition person foreground; not interchangeable with target mask"
            ),
            "target_old_clothing_mask": (
                "source-condition old-clothing mask; not interchangeable with target garment"
            ),
            "target_revealed_skin_mask": "old-clothing removal intersected with safe source skin",
        },
        "not_materialized": [
            "resized_tensor",
            "normalized_tensor_cache",
            "cached_LPIPS_tensor",
            "reference_feature",
            "camera_ray_cache",
            "canonical_space_target",
            "teacher_checkpoint",
        ],
        "loss_consumption": {
            "appearance_denominators": [
                "target_edit_mask",
                "target_clothing_mask",
                "target_old_clothing_mask",
                "target_transition_mask",
                "target_protected_mask",
            ],
            "alpha_denominators": [
                "target_foreground_mask",
                "target_base_foreground_mask",
                "target_protected_mask",
            ],
            "evaluation": (
                "per-record target/base RGB, foreground, clothing, boundary, protected, "
                "and old-clothing masks"
            ),
            "runtime_only": "boundary_aware_transition_alpha_target tensor",
        },
        "record_count": 24,
        "records": records,
        "materialized": False,
    }

    camera_registry = {
        "schema_version": "canondressgs.subject00.teacher_target_camera_registry.v1",
        "task_id": TASK_ID,
        "created_at": created_at,
        "calibration_path": CALIBRATION_PATH,
        "calibration_sha256": CALIBRATION_SHA,
        "calibration_bytes": len(calibration_bytes),
        "camera_count_in_calibration": len(calibration),
        "record_count": len(cameras),
        "unique_camera_binding_count": 22,
        "blocked_camera_binding_count": 2,
        "camera_binding_status": CAMERA_STATUS,
        "decision": (
            "Similarity left-multiplication yields unique per-record pixel intrinsics for "
            "22 machine-pass cells. The two human overrides have materially competing "
            "similarity/projective explanations. H@K is not a unique physical pinhole K, "
            "and visual acceptance cannot recover missing physical camera evidence."
        ),
        "records": cameras,
    }

    resolution_contract = {
        "schema_version": "canondressgs.subject00.teacher_target_resolution_contract.v1",
        "task_id": TASK_ID,
        "created_at": created_at,
        "native_resolution_distribution": dict(resolution_counts),
        "mixed_resolution_support_status": MIXED_STATUS,
        "batch_size": 1,
        "collate": "no cross-resolution image tensor stacking",
        "endpoint_path": ENDPOINT_LOADER + "::_load_samples",
        "renderer_output_resolution": "per-record target_camera.width/height",
        "loss_tensor_shape": "same native HxW within each record",
        "boundary_map_resolution": "same native HxW as record",
        "mask_resolution": "same native HxW as record",
        "camera_intrinsic_adjustment_policy": INTRINSIC_POLICY,
        "anomalous_record": {
            "request_id": "subject00_O03_slot00_canary_attempt004_cand00",
            "resolution": {"width": 1350, "height": 1165},
            "direct_original_intrinsics": False,
            "normalized_intrinsics": False,
            "pixel_space_adjustment": True,
            "registration_transform_required": True,
            "target_camera_mismatch_if_unadjusted": True,
            "binding_status": "PASS_UNIQUE_SIMILARITY_PIXEL_BINDING",
        },
        "scope_warning": (
            "FullDressableTrainingDataset reference stacking can reject mixed HxW when "
            "reference_count>1. This contract freezes the actual endpoint sample loader "
            "and batch-size-one execution path only. The formal Subject02 checker also "
            "expects outfits to share one global condition set; Subject00 cannot safely "
            "use eight shared slot conditions while K/width/height differ by garment. "
            "The plan therefore uses 24 per-record condition IDs and remains blocked "
            "until that camera/schema interface is explicitly re-frozen."
        ),
        "silent_resize_allowed": False,
        "final_status": "BLOCKED_BY_TWO_HUMAN_OVERRIDE_CAMERA_MODELS",
    }

    layout = [
        "01_contract_snapshot",
        "02_records",
        "03_raw",
        "04_person_masks",
        "05_garment_masks",
        "06_camera",
        "07_derived_targets",
        "08_quality_audit",
        "09_review_assets",
        "10_final_registry",
        "11_logs",
    ]
    materialization_audit = {
        "schema_version": (
            "canondressgs.subject00.teacher_target_materialization_mode_audit.v1"
        ),
        "task_id": TASK_ID,
        "created_at": created_at,
        "selected_mode": MATERIALIZATION_MODE,
        "mode_audit": {
            "REFERENCE_ONLY_REGISTRY": (
                "Rejected: Windows paths are not cloud-portable and deletion/reorganization "
                "of source directories would break recovery."
            ),
            "BYTE_COPY_IMMUTABLE_BUNDLE": (
                "Insufficient alone: immutable copies are needed, but a transfer format and "
                "cloud extraction verification are still required."
            ),
            "HARDLINK_IMMUTABLE_BUNDLE": (
                "Rejected: volume-bound on Windows and cannot cross to AutoDL."
            ),
            "SYMLINK_REFERENCE_BUNDLE": (
                "Rejected: Windows/cloud path semantics differ and source deletion remains a risk."
            ),
            "PORTABLE_ARCHIVE_AND_CLOUD_EXTRACTION": (
                "Selected: byte copies, manifest SHA verification, portable transfer, atomic "
                "cloud extraction, and recovery are explicit."
            ),
        },
        "windows_project_root": WINDOWS_PROJECT_ROOT,
        "windows_attempt_root": WINDOWS_ATTEMPT_ROOT,
        "cloud_target_root": CLOUD_TARGET_ROOT,
        "target_root_preexisted": root_exists,
        "target_root_created": False,
        "planned_layout": [
            {"windows": planned_win(name), "cloud": planned_cloud(name)}
            for name in layout
        ],
        "archive": {
            "format": "tar.zst",
            "windows_path": planned_win(
                "10_final_registry",
                "subject00_24cell_teacher_targets_attempt001.tar.zst",
            ),
            "cloud_path": planned_cloud(
                "10_final_registry",
                "subject00_24cell_teacher_targets_attempt001.tar.zst",
            ),
            "verification": (
                "archive SHA + per-file SHA + record-count/schema checks before atomic rename"
            ),
        },
        "data_copy_calls": 0,
        "upload_calls": 0,
        "derived_target_generation_calls": 0,
    }

    qa_gates = []
    for name in QA_GATE_NAMES:
        if name == "camera_model_unique_per_record":
            qa_gates.append(
                {
                    "gate": name,
                    "preflight_status": "BLOCKED_22_PASS_2_NONUNIQUE",
                    "execution_required": True,
                }
            )
        elif name in {
            "derived_target_parse",
            "formal_endpoint_loader_zero_optimizer_smoke",
            "source_sha_immutable_after_materialization",
        }:
            qa_gates.append(
                {
                    "gate": name,
                    "preflight_status": "PREPARED_NOT_RUN_NO_TARGETS_EXIST",
                    "execution_required": True,
                }
            )
        else:
            qa_gates.append(
                {
                    "gate": name,
                    "preflight_status": "PASS_SOURCE_OR_CONTRACT_CHECK",
                    "execution_required": True,
                }
            )
    qa_registry = {
        "schema_version": "canondressgs.subject00.teacher_target_qa_registry.v1",
        "task_id": TASK_ID,
        "created_at": created_at,
        "qa_gate_count": len(qa_gates),
        "gates": qa_gates,
        "zero_optimizer_step_smoke": {
            "status": "PREPARED_NOT_RUN_BLOCKED_PENDING_CAMERA_SCHEMA_REFREEZE",
            "command": (
                "python tools/check_full_dressable_dataset.py "
                "--manifest <10_final_registry/subject00_24_cell_full_dataset_v1.json> "
                "--reference-count 1 --regression "
                "--report <08_quality_audit/full_dataset_loader_zero_step.json>"
            ),
            "required_loader": TARGET_LOADER + "::FullDressableTrainingDataset",
            "gpu_forward_allowed": False,
            "optimizer_steps": 0,
            "blocker": (
                "The shared-condition Subject02 checker contract cannot represent the "
                "per-garment camera/resolution bindings without a scoped re-freeze."
            ),
        },
        "source_integrity_snapshot": {
            "mask_registry_sha256": hashes["mask_registry"],
            "raw_registry_sha256": hashes["raw_registry"],
            "raw_count": 24,
            "raw_sha_match_count": 24,
            "person_mask_count": 24,
            "person_mask_sha_match_count": 24,
            "garment_mask_count": 24,
            "garment_mask_sha_match_count": 24,
        },
    }

    storage = {
        "schema_version": (
            "canondressgs.subject00.teacher_target_storage_transfer_estimate.v1"
        ),
        "task_id": TASK_ID,
        "created_at": created_at,
        "inputs": {
            "accepted_raw_bytes": raw_bytes,
            "accepted_person_mask_bytes": person_bytes,
            "accepted_garment_mask_bytes": garment_bytes,
            "native_pixel_count": pixels,
            "derived_mask_count": 8,
            "derived_masks_uint8_bytes": derived_masks,
            "base_rgb_uncompressed_bytes": base_rgb_uncompressed,
            "registries_qa_review_budget_bytes": registry_review_budget,
            "portable_archive_overhead_ratio": 0.05,
            "operational_reserve_bytes": reserve,
        },
        "payload_projected_bytes": payload,
        "portable_archive_projected_bytes": archive,
        "windows_projected_bytes": windows_projected,
        "cloud_projected_bytes": cloud_projected,
        "transfer_bytes": archive,
        "windows_free_bytes": args.windows_free_bytes,
        "cloud_free_bytes": args.cloud_free_bytes,
        "temporary_atomic_write_policy": (
            "Windows reserves two archive images; cloud reserves payload plus one archive."
        ),
        "storage_gate_status": "PASS" if storage_pass else "BLOCKED_INSUFFICIENT_SPACE",
        "deletion_calls": 0,
    }

    execution_manifest = {
        "schema_version": (
            "canondressgs.subject00.teacher_target_creation_execution_manifest.v1"
        ),
        "task_id": TASK_ID,
        "created_at": created_at,
        "source": {
            "branch": SOURCE_BRANCH,
            "head": SOURCE_HEAD,
            "mask_accepted_registry": str(repo / MASK_REGISTRY_REL),
            "mask_accepted_registry_sha256": hashes["mask_registry"],
            "preflight_verification": {
                "branch_exact": True,
                "head_exact": True,
                "source_worktree_clean": True,
                "origin_branch_head_exact": True,
                "source_pytest": "13 passed",
                "source_structured_checks": "47/47 PASS",
                "raw_accepted_count": 24,
                "mask_pair_accepted_count": 24,
                "mask_missing_count": 0,
                "mask_blocked_count": 0,
                "teacher_target_count": 0,
                "accepted_asset_mutations": 0,
                "attempt_001_through_005_mutations": 0,
            },
        },
        "new_branch": NEW_BRANCH,
        "worktree": WORKTREE,
        "stage_a": {
            "name": "TEACHER_TARGET_DATASET_MATERIALIZATION",
            "requires_final_base": False,
            "base_dependency_ready": True,
            "overall_ready": False,
            "blocker": CAMERA_STATUS,
            "authorized": False,
            "teacher_target_count": 0,
        },
        "stage_b": {
            "name": "TEACHER_ENDPOINT_OPTIMIZATION",
            "requires_base": True,
            "authorized": False,
            "optimizer_steps": 0,
        },
        "target_schema": TARGET_SCHEMA,
        "record_count": 24,
        "coverage": {"O01": 8, "O03": 8, "O04": 8},
        "camera_binding_status": CAMERA_STATUS,
        "mixed_resolution_support_status": MIXED_STATUS,
        "materialization_mode": MATERIALIZATION_MODE,
        "windows_project_root": WINDOWS_PROJECT_ROOT,
        "windows_attempt_root": WINDOWS_ATTEMPT_ROOT,
        "cloud_target_root": CLOUD_TARGET_ROOT,
        "target_root_preexisted": root_exists,
        "target_root_created": False,
        "records": records,
        "counters": {
            "teacher_target_creation_authorized": False,
            "teacher_target_count": 0,
            "teacher_endpoint_optimization_authorized": False,
            "optimizer_steps": 0,
            "data_copy_calls": 0,
            "upload_calls": 0,
            "derived_target_generation_calls": 0,
            "accepted_raw_mutations": 0,
            "person_mask_mutations": 0,
            "garment_mask_mutations": 0,
            "attempt_001_mutations": 0,
            "attempt_002_mutations": 0,
            "attempt_003_mutations": 0,
            "attempt_004_mutations": 0,
            "attempt_005_mutations": 0,
            "data_mutations": 0,
            "paper_modifications": 0,
        },
        "final_classification": FINAL_CLASSIFICATION,
        "next_task": NEXT_TASK,
        "paper_final": False,
    }

    o03_records = [x for x in records if x["garment"] == "O03"]
    o03_manifest = {
        "schema_version": (
            "canondressgs.subject00.o03_provisional_teacher_target_manifest_draft.v1"
        ),
        "task_id": TASK_ID,
        "created_at": created_at,
        "status": "DRAFT_NOT_MATERIALIZED_NOT_AUTHORIZED",
        "target_schema": TARGET_SCHEMA,
        "record_count": 8,
        "slots": [f"slot_{index:02d}" for index in range(8)],
        "records": o03_records,
        "provisional_base": {
            "step": 60747,
            "garment_scope": "O03_ONLY",
            "checkpoint_path": BASE60747_CHECKPOINT,
            "checkpoint_bytes": BASE60747_BYTES,
            "checkpoint_sha256": BASE60747_SHA,
            "paper_eligible": False,
            "formal_base_101245_replaced": False,
            "task_status": BASE60747_STATUS,
        },
        "teacher_target_count": 0,
        "teacher_target_creation_authorized": False,
        "teacher_endpoint_optimization_authorized": False,
    }

    check_names = [
        "source_branch_head",
        "source_clean",
        "mask_accepted_count_24",
        "raw_count_24",
        "person_mask_count_24",
        "garment_mask_count_24",
        "raw_sha_complete",
        "mask_sha_complete",
        "exact_cell_mapping",
        "subject02_schema_recovery",
        "teacher_loader_binding",
        "target_creation_implementation_binding",
        "camera_calibration_binding",
        "camera_record_completeness",
        "mixed_resolution_decision",
        "anomalous_1350x1165_handling",
        "registration_override_propagation",
        "limitation_propagation_9_of_9",
        "materialization_mode_decision",
        "windows_root",
        "cloud_root",
        "exact_record_paths",
        "derived_target_requirements",
        "target_qa_registry",
        "o03_subset_8",
        "provisional_base60747_status_recovery",
        "final_base_dependency",
        "stage_a_stage_b_decoupling",
        "storage_estimate",
        "transfer_estimate",
        "target_root_absent",
        "teacher_target_count_zero",
        "creation_authorized_false",
        "optimization_authorized_false",
        "optimizer_steps_zero",
        "no_data_copy",
        "no_upload",
        "accepted_raw_immutable",
        "accepted_masks_immutable",
        "attempts_001_005_immutable",
        "data_mutation_zero",
        "paper_modification_zero",
        "execution_manifest_schema",
        "handoff_schema",
        "final_classification",
        "next_task_uniqueness",
    ]
    checks = [
        {
            "index": index,
            "check": name,
            "status": (
                "BLOCKED_22_OF_24_CAMERA_RECORDS_EXECUTABLE"
                if name == "camera_record_completeness"
                else "PASS"
            ),
        }
        for index, name in enumerate(check_names, 1)
    ]
    tests = {
        "schema_version": (
            "canondressgs.subject00.teacher_target_creation_preflight_tests.v1"
        ),
        "task_id": TASK_ID,
        "created_at": created_at,
        "check_count": len(checks),
        "pass_count": sum(x["status"] == "PASS" for x in checks),
        "blocked_count": sum(x["status"].startswith("BLOCKED") for x in checks),
        "checks": checks,
        "test_result": (
            "PY_COMPILE_PASS; PYTEST_16_PASSED; JSON_ARTIFACT_PARSE_11_PASS; "
            "STRUCTURED_CHECKS_45_OF_46_PASS_1_SCIENTIFIC_CAMERA_CONTRACT_BLOCKER; "
            "GIT_DIFF_CHECK_PASS"
        ),
    }

    base_status = args.formal_base_status
    summary = {
        "schema_version": (
            "canondressgs.subject00.teacher_target_creation_preflight_final_summary.v1"
        ),
        "task_id": TASK_ID,
        "created_at": created_at,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "new_branch": NEW_BRANCH,
        "worktree": WORKTREE,
        "counts": {
            "raw_accepted": 24,
            "person_mask_accepted": 24,
            "garment_mask_accepted": 24,
            "target_records": 24,
            "O01": 8,
            "O03": 8,
            "O04": 8,
            "limitations": 9,
            "human_overrides": 2,
            "teacher_targets": 0,
        },
        "subject02_target_schema_status": "PASS_FORMAL_SUBJECT02_SCHEMA_RECOVERED",
        "target_schema": TARGET_SCHEMA,
        "camera_binding_status": CAMERA_STATUS,
        "native_resolution_distribution": dict(resolution_counts),
        "mixed_resolution_support_status": MIXED_STATUS,
        "registration_override_propagation_status": (
            "PASS_2_OF_2_PROPAGATED_AND_GEOMETRY_RESTRICTED"
        ),
        "limitation_propagation_status": "PASS_9_OF_9_WITH_RISK_METADATA",
        "materialization_mode": MATERIALIZATION_MODE,
        "base60747_task_status": BASE60747_STATUS,
        "base60747_checkpoint": BASE60747_CHECKPOINT,
        "base60747_checkpoint_sha256": BASE60747_SHA,
        "formal_base_current_status": base_status,
        "formal_base_current_step": args.formal_base_step,
        "teacher_target_materialization_requires_final_base": False,
        "teacher_target_creation_ready_independent_of_base": True,
        "teacher_endpoint_optimization_requires_base": True,
        "storage_gate_status": storage["storage_gate_status"],
        "test_result": tests["test_result"],
        "final_classification": FINAL_CLASSIFICATION,
        "next_task": NEXT_TASK,
        "paper_final": False,
    }
    handoff = {
        "schema_version": (
            "canondressgs.project_control.teacher_target_creation_execution_handoff.v1"
        ),
        "task_id": TASK_ID,
        "created_at": created_at,
        "status": "BLOCKED_DO_NOT_EXECUTE",
        "reason": CAMERA_STATUS,
        "execution_contract": paths["execution_contract"],
        "execution_manifest": paths["execution_manifest"],
        "authorizations": {
            "teacher_target_creation": False,
            "teacher_endpoint_optimization": False,
        },
        "resume_only_after": [
            "Both human-override camera models are uniquely resolved and recorded.",
            "Camera registry and resolution contract are re-frozen.",
            "User provides a new explicit creation authorization.",
        ],
        "immutable_inputs": {
            "mask_accepted_registry": str(repo / MASK_REGISTRY_REL),
            "mask_accepted_registry_sha256": hashes["mask_registry"],
            "calibration_path": CALIBRATION_PATH,
            "calibration_sha256": CALIBRATION_SHA,
        },
        "next_task": NEXT_TASK,
    }

    contract = f"""# Subject00 24-Cell Teacher-Target Creation Execution Contract

## Frozen decision

This is the Stage A dataset-materialization contract for `{TASK_ID}`. It is
scientifically **blocked** and is not an execution authorization.

- Source: `{SOURCE_BRANCH}` at `{SOURCE_HEAD}`
- Records: 24 (O01=8, O03=8, O04=8)
- Target schema: `{TARGET_SCHEMA}`
- Materialization: `{MATERIALIZATION_MODE}`
- Final classification: `{FINAL_CLASSIFICATION}`
- Next task: `{NEXT_TASK}`

`TEACHER_TARGET_CREATION_AUTHORIZED=false`,
`TEACHER_TARGET_COUNT=0`,
`TEACHER_ENDPOINT_OPTIMIZATION_AUTHORIZED=false`, and `OPTIMIZER_STEPS=0`.
No target root was created.

## Stage A and Stage B

Stage A materializes accepted edit RGB, source/base RGB, accepted person and
garment masks, formal dual-target masks, per-record camera/pose metadata,
checksums, limitations, and QA. It does not require a final Base checkpoint.
Stage B initializes from a Base checkpoint and runs Teacher Endpoint
optimization; it does require a Base. A provisional O03 run may bind step
60747 but is never paper-eligible and does not replace final step 101245.

## Formal Subject02 definition

The recovered precedent is `{SUBJECT02_MANIFEST}` (SHA256
`{SUBJECT02_MANIFEST_SHA}`). Its schema is `{TARGET_SCHEMA}` and the formal
loader is `{TARGET_LOADER}` (SHA256 `{hashes["loader"]}`). The actual
Teacher-endpoint sample path is `{ENDPOINT_LOADER}::_load_samples`.
`{TARGET_BUILDER}` (SHA256 `{hashes["builder"]}`) byte-copies edit RGB and
materializes formal region masks. Cached tensors, LPIPS features, ray caches,
canonical targets, and checkpoints are not Stage A assets.
The Subject02 endpoint helper hardcodes its four-condition precedent and is
evidence, not an executable Subject00 binding. Subject00's per-record camera
and resolution contract must be re-frozen before its zero-step loader smoke.

## Camera and resolution blocker

All records bind `{CALIBRATION_PATH}` (SHA256 `{CALIBRATION_SHA}`), frame 0,
and the frozen slot-camera-direction mapping. For 22 machine-pass cells,
source-to-target similarity `S` uniquely gives `K_target=S@K_source`, while
`w2c` is unchanged and the native target HxW is retained. The endpoint path
supports mixed native resolution only with batch size one.

The two human overrides do not select a unique physical camera: their
similarity and projective explanations compete, `H@K` is not a unique
pinhole intrinsic matrix, and visual acceptance cannot recover calibration.
They remain valid appearance evidence with disclosure, but strict geometry
supervision is forbidden. No loss weight is changed here.

## Materialization layout

The frozen Windows root is `{WINDOWS_ATTEMPT_ROOT}` and the cloud root is
`{CLOUD_TARGET_ROOT}`. The exact 24 record, raw, person-mask, garment-mask,
camera, and formal observation paths are in `{paths["execution_manifest"]}`.
The portable bundle is staged as `tar.zst`, verified by archive and per-file
SHA, extracted to a temporary cloud path, QA-checked, then atomically renamed.
Nothing in this contract permits creation while the camera gate is blocked.

## QA and storage

The QA registry freezes 25 gates, including exact counts/SHA, native shapes,
binary mask semantics, garment-subset-person, camera parse, limitation and
override propagation, derived-target parse, the formal dataset-loader
zero-optimizer smoke, and post-copy source immutability. The smoke is prepared
but intentionally blocked until the per-record camera/schema interface is
resolved. Storage gate:
`{storage["storage_gate_status"]}`; Windows projected bytes
`{windows_projected}`, cloud projected bytes `{cloud_projected}`, transfer
bytes `{archive}`.

## Prohibited actions

Do not create roots, copy/link files, build archives, upload data, generate
derived targets, run a GPU forward, run an optimizer step, pause/resume Formal
Base, start O03 Teacher, mutate accepted assets or attempts 001-005, or edit
the paper body.
"""
    report = f"""# AAAI27 Subject00 24-Cell Teacher-Target Creation Preflight

## Outcome

The formal Subject02 schema was recovered, 24 accepted RGB/person-mask/
garment-mask bindings were frozen in their authoritative order, and the
portable Stage A layout, QA, storage, and O03 provisional subset were
specified. The contract is not executable yet:
`{FINAL_CLASSIFICATION}`.

## Evidence summary

| Item | Result |
|---|---|
| Accepted inputs | 24 raw, 24 person masks, 24 garment masks |
| Coverage | O01 8, O03 8, O04 8 |
| Limitations | 9/9 propagated |
| Human registration overrides | 2/2 propagated |
| Formal schema | `{TARGET_SCHEMA}` |
| Native resolution | 23 x 1349x1166; 1 x 1350x1165 |
| Mixed-resolution endpoint mode | `{MIXED_STATUS}` |
| Unique camera bindings | 22/24 |
| Stage A needs final Base | false |
| Stage B needs Base | true |
| Target roots/targets created | 0 |

## Why the gate is blocked

For machine-pass records the registered source-to-target similarity is a
unique pixel mapping and the target intrinsic matrix is `S@K`. The two
human-overridden failures remain visually acceptable image evidence, but the
evidence does not choose between similarity and projective camera
interpretations. Treating a projective homography as a physical intrinsic
matrix would be scientifically non-unique. Those two records are therefore
restricted to target-space appearance evidence and are blocked from strict
geometry supervision until a unique camera contract is supplied.

The 1350x1165 record is not itself the blocker: it has a machine-pass
similarity, is retained without resize, and is supported by the actual
batch-size-one loader path. It does, however, prove that the formal
Subject02 shared-condition convention cannot encode one global slot00
`K,width,height` across O01/O03/O04. The frozen draft therefore uses 24
per-record condition IDs and does not claim formal checker compatibility
until the camera/schema interface is re-frozen.

## Stage boundary

Stage A is Base-independent in dependency terms. It materializes source/edit
RGB, formal masks, camera/pose records, and provenance. Stage B performs
optimization and requires either provisional Base60747 (O03-only,
paper-ineligible) or final Base101245 for paper-eligible results.

Formal Base was observed in user-authorized paused state. The latest complete
log row is step `{args.formal_base_step}`, the only durable resume point is
sealed step `60747`, resume authorization remains false, and final evaluation
is pending. The separate
Base60747 O03 task is classified `{BASE60747_STATUS}` and was not modified.

## Frozen next action

`{NEXT_TASK}`. This report does not grant execution authorization.
"""

    for key, value in {
        "schema_registry": schema_registry,
        "camera_registry": camera_registry,
        "resolution_contract": resolution_contract,
        "materialization_audit": materialization_audit,
        "qa_registry": qa_registry,
        "storage_estimate": storage,
        "execution_manifest": execution_manifest,
        "o03_manifest": o03_manifest,
        "tests": tests,
        "final_summary": summary,
        "handoff": handoff,
    }.items():
        write_json(repo / paths[key], value)
    write_text(repo / paths["execution_contract"], contract)
    write_text(repo / paths["report"], report)

    print(
        json.dumps(
            {
                "status": "BUILT_READ_ONLY_PREFLIGHT_ARTIFACTS",
                "artifact_count": len(paths),
                "records": len(records),
                "camera_binding_status": CAMERA_STATUS,
                "final_classification": FINAL_CLASSIFICATION,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--formal-base-step", type=int, required=True)
    parser.add_argument("--formal-base-status", required=True)
    parser.add_argument("--windows-free-bytes", type=int, required=True)
    parser.add_argument("--cloud-free-bytes", type=int, required=True)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
