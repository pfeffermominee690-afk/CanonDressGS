#!/usr/bin/env python3
"""Build the frozen Subject00 generation-preparation contract artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "AAAI27-SUBJECT00-DATA-PREPARATION-GENERATION-READY-001"
SOURCE_BRANCH = "research/subject00-minimal-second-identity-dataset-contract-20260725"
SOURCE_HEAD = "9da04d923c7c6c03c6a6f7c732d8d9c58d59c88d"
BRANCH = "research/subject00-data-preparation-generation-ready-20260725"
DATA_PREP_RESULT_HEAD = "44413b5665bce7749ec1d2e6a1c4726fa850b7ba"
WINDOWS_WORKTREE = "E:/model_train/canondressgs_subject00_data_preparation_generation_ready"
CLOUD_WORKTREE = (
    "/root/autodl-tmp/canondressgs_work/worktrees/"
    "canondressgs_subject00_data_preparation_generation_ready"
)
RAW_ROOT = "/root/autodl-tmp/datasets/thuman4_second_identity_staging/subject00"
DERIVED_ROOT = "/root/autodl-tmp/datasets/thuman4_second_identity_staging/derived_assets/subject00"
DATASET_ROOT = "/root/autodl-tmp/canondressgs_work/datasets/subject00_three_garment"
WINDOWS_DATASET_MIRROR = "E:/model_train/canondressgs_work/datasets/subject00_three_garment"
FORMAL_OUTPUT_ROOT = (
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-MMLPHUMAN-FORMAL-STRICT-SPLIT-001"
)
FORMAL_PROTOCOL_BRANCH = "research/mmlphuman-subject00-formal-strict-split-protocol-20260723"
FORMAL_PROTOCOL_HEAD = "4993f5c865ec19895f35811fa399fc4a1834c6a7"
STORAGE_BRANCH = "research/mmlphuman-subject00-storage-migration-adjudication-20260725"
STORAGE_HEAD = "34e91445ef45c001cebcd789a4a6a88cad7b9ad8"
GENERATION_WORKFLOW_BRANCH = "research/multi-identity-generation-backend-review-workflow-20260724"
GENERATION_WORKFLOW_HEAD = "e8a09525f745f373e38418c43040766d69a50d10"
FOUNDATIONS_BRANCH = "research/multi-identity-garment-benchmark-foundations-20260724"
FOUNDATIONS_HEAD = "9e46e85cd02527b52e2f5bf0af5571de4c8988d2"
ORIGINAL_PROTOCOL_BRANCH = "research/subject00-three-garment-canary-protocol-20260724"
ORIGINAL_PROTOCOL_HEAD = "b666e8168ec76d67a9e32f26fe9d2e778609cc98"
FORMAL_DEPENDENCY = "PENDING"
BACKEND_STATUS = "BACKEND_SELECTION_REQUIRED"
CLASSIFICATION = "SUBJECT00_DATA_PREPARATION_READY_PENDING_BACKEND"
NEXT_TASK = "USER_ADJUDICATE_SUBJECT00_GENERATION_BACKEND"
GARMENTS = ["O01", "O03", "O04"]
TRAIN_CAMERAS = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23]
HELDOUT_CAMERAS = [0, 4, 8, 12, 16, 20]
POSE_SPLIT_SHA = "c12e5eec87c737d3740b6e6199d657c859df7ce1ec9d371ea8d4a987ef62bf06"
CAMERA_SPLIT_SHA = "8827cdc05a08cb7068e1f5e89ab7ba5384bf83b89686e8c9f80024b78d49cf44"
RAW_FINGERPRINT = "2c0f894f70d944fa8d6ebd48ab1188b78cb19b673f0b7c006f1922a592b1ea7b"
IDENTITY_SET_ID = "subject00_pose00000000_strict_train_multiview_v1"


SLOTS = [
    {
        "slot_id": "slot_00",
        "semantic_pose_slot": "front",
        "orientation_degrees": 0,
        "camera_id": 17,
        "camera_azimuth_degrees": -90.62241378961899,
        "calibrated_view_orientation_degrees": 359.377586210381,
        "orientation_error_degrees": 0.622413789619,
        "reference_role": True,
    },
    {
        "slot_id": "slot_01",
        "semantic_pose_slot": "front_left_three_quarter",
        "orientation_degrees": 45,
        "camera_id": 21,
        "camera_azimuth_degrees": -39.48140215655296,
        "calibrated_view_orientation_degrees": 50.51859784344704,
        "orientation_error_degrees": 5.51859784344704,
        "reference_role": False,
    },
    {
        "slot_id": "slot_02",
        "semantic_pose_slot": "front_right_three_quarter",
        "orientation_degrees": 315,
        "camera_id": 14,
        "camera_azimuth_degrees": -135.82565234877947,
        "calibrated_view_orientation_degrees": 314.17434765122053,
        "orientation_error_degrees": 0.82565234877947,
        "reference_role": False,
    },
    {
        "slot_id": "slot_03",
        "semantic_pose_slot": "left",
        "orientation_degrees": 90,
        "camera_id": 23,
        "camera_azimuth_degrees": -8.560985248097323,
        "calibrated_view_orientation_degrees": 81.43901475190268,
        "orientation_error_degrees": 8.56098524809732,
        "reference_role": True,
    },
    {
        "slot_id": "slot_04",
        "semantic_pose_slot": "right",
        "orientation_degrees": 270,
        "camera_id": 11,
        "camera_azimuth_degrees": 169.8061218676584,
        "calibrated_view_orientation_degrees": 259.8061218676584,
        "orientation_error_degrees": 10.1938781323416,
        "reference_role": True,
    },
    {
        "slot_id": "slot_05",
        "semantic_pose_slot": "back_left_three_quarter",
        "orientation_degrees": 135,
        "camera_id": 2,
        "camera_azimuth_degrees": 46.46093245809582,
        "calibrated_view_orientation_degrees": 136.46093245809582,
        "orientation_error_degrees": 1.46093245809582,
        "reference_role": False,
    },
    {
        "slot_id": "slot_06",
        "semantic_pose_slot": "back_right_three_quarter",
        "orientation_degrees": 225,
        "camera_id": 9,
        "camera_azimuth_degrees": 140.91054645252837,
        "calibrated_view_orientation_degrees": 230.91054645252837,
        "orientation_error_degrees": 5.91054645252837,
        "reference_role": False,
    },
    {
        "slot_id": "slot_07",
        "semantic_pose_slot": "back",
        "orientation_degrees": 180,
        "camera_id": 5,
        "camera_azimuth_degrees": 88.20538437815966,
        "calibrated_view_orientation_degrees": 178.20538437815966,
        "orientation_error_degrees": 1.79461562184034,
        "reference_role": True,
    },
]


RAW_ASSETS = {
    17: {
        "rgb_bytes": 402181,
        "rgb_sha256": "b5e790c6e978fa0ecbf7bcad3fe64aa64382f67f22ff3f1b0bb3d0d33764abdf",
        "mask_bytes": 36065,
        "mask_sha256": "c54df74796fb9fdb500f264ebf6c9d534200a9051e8ce80eb2f2c985c153e234",
    },
    21: {
        "rgb_bytes": 447548,
        "rgb_sha256": "24123ce6df43b4b00fe6d06aae74e69fc8eb08552e86945a864340a431cf52ed",
        "mask_bytes": 35772,
        "mask_sha256": "1f876bcdd16dbb0b473639f3cd287f436a58ea6281c3fcaf636fb71bc7481857",
    },
    14: {
        "rgb_bytes": 418739,
        "rgb_sha256": "575d04d79c3b89e1f94171ab42c80591fa843429ee815fef823bdd13c24fba33",
        "mask_bytes": 34027,
        "mask_sha256": "7acb6949e6743ce978fa520f3ec76d9495224789d64e9f75e7ef9fb209fdb71e",
    },
    23: {
        "rgb_bytes": 416737,
        "rgb_sha256": "a0c34bd8bee414ef2aa06b8457ef661790ff9956aa9b1b16079e15dc6fb1a23b",
        "mask_bytes": 29649,
        "mask_sha256": "65d115eb03bdf718ac37159aa47919343fbca202d94071a4308cd6943ca3ef1d",
    },
    11: {
        "rgb_bytes": 396795,
        "rgb_sha256": "7b990d29fc29d2b96756db898fd01dc35f20f7995523ef2c9a56b885d32568b8",
        "mask_bytes": 29987,
        "mask_sha256": "47078237433af5e8a291f54fb8cdf7dc8abeec09ad9caa039fe8cc4c0f9e7069",
    },
    2: {
        "rgb_bytes": 392062,
        "rgb_sha256": "6b86a5c2961ff3c7e87c680d950708929fc823a39c62bac39111f6f6a30f62e9",
        "mask_bytes": 33751,
        "mask_sha256": "0e579c3fad7366c00438a429629d6aaf36497e880d113936d1c9d61ad5298150",
    },
    9: {
        "rgb_bytes": 373465,
        "rgb_sha256": "67c2616c073c57165efe54982bf2cd18a47d0be18cd1d67576ad4020af70c373",
        "mask_bytes": 35302,
        "mask_sha256": "cd48ea97b973f503caaf71f266c3216740b510cac6943e281e09d63d3b10a2ec",
    },
    5: {
        "rgb_bytes": 407615,
        "rgb_sha256": "28882ff5e36f5d740d12fff16ec9e892702e7e0c8db0151b8ca0ec72b378c294",
        "mask_bytes": 38130,
        "mask_sha256": "1c6f0cb0edd14096cf862e861ba3156905aabc1d6f4a13ce729c060475efb809",
    },
}


SHARED_RAW_ASSETS = [
    {
        "asset_id": "subject00_calibration",
        "category": "calibration_and_camera_poses",
        "absolute_path": f"{RAW_ROOT}/calibration.json",
        "logical_path": "subject00/calibration.json",
        "bytes": 22835,
        "sha256": "4b2d98d20740da03be209040851fab7e8801e5670937ed9ad290a20264d1dde7",
        "resolution": None,
        "camera_id": "ALL_24_CAMERAS",
        "frame_pose_id": None,
        "role": "CAMERA_CALIBRATION_AND_POSE_SOURCE",
        "generation_input_eligible": True,
        "formal_split_authorization": "SHARED_METADATA; SELECTED_RECORDS_STRICT_TRAIN_ONLY",
    },
    {
        "asset_id": "subject00_smpl_params",
        "category": "smplx_parameters",
        "absolute_path": f"{RAW_ROOT}/smpl_params.npz",
        "logical_path": "subject00/smpl_params.npz",
        "bytes": 1722078,
        "sha256": "ac2738c308ad1a9cc02e7b63323c75e0eab28bddc88c57bb5887e07bf8ea28d2",
        "resolution": None,
        "camera_id": None,
        "frame_pose_id": "ALL_2500_POSES; SELECTED_POSE_0",
        "role": "POSE_SMPLX_SOURCE",
        "generation_input_eligible": True,
        "formal_split_authorization": "POSE_0_STRICT_TRAIN",
    },
]


DERIVED_ASSETS = [
    ("surface_lbs/attachment_valid.npy", 200128, "3d689841628e1923264d5f23489e12ea2001b0bf1643c30b57fe76e547644a9e"),
    ("surface_lbs/barycentric.npy", 4800128, "66855bf9badaedcca51b153eb4bd69d4caff1822d51a3295ddc52abd24e45e8e"),
    ("surface_lbs/canonical_gaussian_positions.npy", 4800128, "2d810ac3de878f644772d91acdf5a3cad643e134a6b033ece7be3c25951c7b86"),
    ("surface_lbs/component_ids.npy", 400128, "bfb151aa95932fdd8177555e4493feb457fd60f87769148e6beabfd60c780639"),
    ("surface_lbs/face_ids.npy", 1600128, "1d072c3614cd2e6d3a409bfb9c0f93db455b45825f9d567155739a1e7c5efc28"),
    ("surface_lbs/gaussian_indices.npy", 1600128, "2f9c5487d165ba041155e217b24ecb23d19ac76f55378beea6717c21a0971f62"),
    ("surface_lbs/lbs_weights.npy", 44000128, "56aa68a9d4baade67621fa2bfac462ac88074eeaf7c9bfdbe86f2360b91261a1"),
    ("surface_lbs/semantic_region_ids.npy", 400128, "4e6832aa913a7cfc24b044655d1bf3e95905a3e2f1ce0cc2498f3cf5c70da1fd"),
    ("surface_lbs/source_type_ids.npy", 200128, "3d689841628e1923264d5f23489e12ea2001b0bf1643c30b57fe76e547644a9e"),
    ("surface_lbs/surface_attachment_manifest.json", 7581, "de6cd51fe3f81e29139b45494860b186038b75a378b101e7428252b3b66c1af8"),
    ("surface_lbs/surface_distance.npy", 1600128, "c21228fec8257ca1672af07a20e1397b54da5a0cbb71be41d426c88a30cbbbd9"),
    ("template/template_faces.npy", 501920, "c69d15f4631b49585df55b4581c6b47ba731af5bc7ff04461c6ba0e25c557a1d"),
    ("template/template_generation_report.json", 5839, "473450061dc8ac4655d9dba3bae805990357c9ffb0e0bc44c667f3ce3246151e"),
    ("template/template_smplx_body_surface.ply", 758242, "f10a3b516e2b3a2ad38dc4924a3692b2f3e72a6cc9e66f3c0063c4e9cd210031"),
    ("template/template_vertices.npy", 125828, "7d5450e493c6d6d211bd051dadbb83eaf362b8652d849f859acd11513b46936c"),
]


VISUAL_ASSETS = [
    (
        "canary_step0_query001",
        "white_model_clay_render_composite",
        "/root/autodl-tmp/canondressgs_work/outputs/SUBJECT00-MMLPHUMAN-SHORT-CANARY-001/attempt_001/visuals/step_000000/query_001.png",
        42262,
        "2b5010c1a58ae6b9c30219dd59231a4762ee29701ca6fd82aebcbca6bbc58b8f",
        [1850, 400],
        1,
        0,
    ),
    (
        "canary_step384_query001",
        "previous_canary_visual",
        "/root/autodl-tmp/canondressgs_work/outputs/SUBJECT00-MMLPHUMAN-SHORT-CANARY-001/attempt_001/visuals/step_000384/query_001.png",
        46168,
        "956520c314c8b124210e645703639e8b67901184ddb675f1ed02e2ee6f76688f",
        [1850, 400],
        1,
        0,
    ),
    (
        "canary_step0_sheet01",
        "existing_thumbnail_contact_sheet",
        "/root/autodl-tmp/canondressgs_work/outputs/SUBJECT00-MMLPHUMAN-SHORT-CANARY-001/attempt_001/visuals/step_000000_contact_sheets/sheet_01.jpg",
        293065,
        "fd9543d5da3428155dfe3dcd96d8a0f0f8107a9d51d49c3a8c410a5088a095e0",
        [3200, 1384],
        "MULTIPLE",
        "MULTIPLE",
    ),
    (
        "canary_step384_sheet01",
        "existing_thumbnail_contact_sheet",
        "/root/autodl-tmp/canondressgs_work/outputs/SUBJECT00-MMLPHUMAN-SHORT-CANARY-001/attempt_001/visuals/step_000384_contact_sheets/sheet_01.jpg",
        297835,
        "99dbe4708deadec5c4ef5e8efc85b737ac3c2582e417e203364bee41cb9ab79f",
        [3200, 1384],
        "MULTIPLE",
        "MULTIPLE",
    ),
    (
        "medium_step20249_query013",
        "medium_pilot_visual",
        "/root/autodl-tmp/canondressgs_work/outputs/SUBJECT00-MMLPHUMAN-MEDIUM-PILOT-001/attempt_001/visuals/step_020249/query_013.png",
        49610,
        "3490b0f37e66795f56133f8bd7366241c44ee61aab1e85d75c774370bdcbb4ac",
        [1850, 400],
        17,
        0,
    ),
    (
        "medium_comparison_query013",
        "medium_pilot_comparison_thumbnail",
        "/root/autodl-tmp/canondressgs_work/outputs/SUBJECT00-MMLPHUMAN-MEDIUM-PILOT-001/attempt_001/visuals/comparison_sheets/query_013.png",
        264742,
        "65ca20a12df606c2b232d8a58a2f9ec9eea0c91afdbfb99ee600279f016d6031",
        [5550, 510],
        17,
        0,
    ),
]


GARMENT_SEMANTICS = {
    "O01": {
        "description": "Light heather-gray pullover hoodie with black full-length straight trousers.",
        "reference_prompt": "A light heather-gray pullover hoodie with hood, drawstrings, long sleeves, ribbed cuffs and waist hem, paired with black full-length straight trousers.",
        "forbidden": ["zip-front hoodie", "short sleeves", "blue trousers", "double hood", "retained old cuffs"],
    },
    "O03": {
        "description": "Dark navy single-breasted business suit with matching trousers, white dress shirt, and dark navy tie.",
        "reference_prompt": "A dark navy tailored single-breasted two-button business suit with matching full-length trousers, a white dress shirt, and a dark navy tie.",
        "forbidden": ["casual jacket", "missing tie", "mismatched trousers", "open bare chest", "short sleeves"],
    },
    "O04": {
        "description": "Black waist-length zip-front bomber jacket over a light-gray crew-neck shirt with medium-blue denim jeans.",
        "reference_prompt": "A black waist-length zip-front bomber jacket with long sleeves over a light-gray crew-neck shirt, paired with medium-blue full-length straight denim jeans.",
        "forbidden": ["hood", "long coat", "black trousers", "formal blazer", "distressed or torn jeans"],
    },
}


DATASET_DIRS = [
    "00_contract",
    "01_identity_sources",
    "02_garment_references",
    "03_generation_requests",
    "04_generation_responses",
    "05_human_review",
    "06_accepted_rgb",
    "07_accepted_masks",
    "08_camera_pose",
    "09_teacher_targets",
    "10_controller_splits",
    "11_provenance",
    "12_final_verification",
]


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def seal(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["content_sha256"] = canonical_sha256(payload)
    return result


def write_json(relative: str, payload: dict[str, Any]) -> None:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_text(relative: str, value: str) -> None:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.strip() + "\n", encoding="utf-8", newline="\n")


def raw_paths(camera_id: int) -> dict[str, str]:
    camera = f"cam{camera_id:02d}"
    return {
        "rgb": f"{RAW_ROOT}/images/{camera}/00000000.jpg",
        "mask": f"{RAW_ROOT}/masks/{camera}/00000000.jpg",
        "camera": f"{RAW_ROOT}/calibration.json",
        "pose_smplx": f"{RAW_ROOT}/smpl_params.npz",
    }


def source_record(slot: dict[str, Any]) -> dict[str, Any]:
    camera_id = slot["camera_id"]
    assets = RAW_ASSETS[camera_id]
    paths = raw_paths(camera_id)
    return {
        "identity_source_id": f"subject00_pose00000000_cam{camera_id:02d}",
        "camera_id": camera_id,
        "pose_frame_id": 0,
        "semantic_pose_slot": slot["semantic_pose_slot"],
        "slot_id": slot["slot_id"],
        "orientation_degrees": slot["orientation_degrees"],
        "camera_azimuth_degrees": slot["camera_azimuth_degrees"],
        "calibrated_view_orientation_degrees": slot["calibrated_view_orientation_degrees"],
        "orientation_error_degrees": slot["orientation_error_degrees"],
        "source_rgb_path": paths["rgb"],
        "source_rgb_sha256": assets["rgb_sha256"],
        "source_rgb_bytes": assets["rgb_bytes"],
        "source_mask_path": paths["mask"],
        "source_mask_sha256": assets["mask_sha256"],
        "source_mask_bytes": assets["mask_bytes"],
        "camera_file_path": paths["camera"],
        "camera_file_sha256": SHARED_RAW_ASSETS[0]["sha256"],
        "pose_smplx_file_path": paths["pose_smplx"],
        "pose_smplx_file_sha256": SHARED_RAW_ASSETS[1]["sha256"],
        "resolution": {"width": 1330, "height": 1150},
        "strict_split_role": "STRICT_TRAIN",
        "formal_split_authorized": True,
        "heldout_camera": False,
        "heldout_pose": False,
        "buffer_only_pose": False,
        "existence_verified": True,
    }


def build_recovery_audit() -> dict[str, Any]:
    return seal(
        {
            "schema_version": "canondressgs.subject00.data_preparation_recovery_audit.v1",
            "task_id": TASK_ID,
            "source": {"branch": SOURCE_BRANCH, "head": SOURCE_HEAD},
            "new_branch": BRANCH,
            "worktrees": {"windows": WINDOWS_WORKTREE, "cloud": CLOUD_WORKTREE},
            "three_layer_consistency": {
                "status": "PASS",
                "git_head": SOURCE_HEAD,
                "final_summary_classification": "SUBJECT00_CONDITION_RESOURCE_GAP",
                "handoff_classification": "SUBJECT00_CONDITION_RESOURCE_GAP",
                "summary_handoff_counts_match": True,
            },
            "recovered_task_b": {
                "classification": "SUBJECT00_CONDITION_RESOURCE_GAP",
                "scientific_semantics": "SECOND_IDENTITY_REPLICATION",
                "formal_base_dependency": "PENDING",
                "garments": GARMENTS,
                "slots_per_garment": 8,
                "planned_references": 12,
                "planned_teacher_targets": 24,
                "planned_candidates": 48,
                "backend_status": "BACKEND_SELECTION_REQUIRED",
                "api_calls": 0,
                "materialized_images": 0,
                "endpoint_runs": 0,
                "controller_runs": 0,
                "paper_final": False,
            },
            "formal_base": {
                "branch_read": STORAGE_BRANCH,
                "head_read": STORAGE_HEAD,
                "classification": "SUBJECT00_FORMAL_OUTPUT_ROOT_REPAIRED_EXECUTION_INVALID",
                "formal_attempt_exists": False,
                "execution_head": None,
                "sealed_formal_checkpoint": None,
                "sealed_downstream_manifest": None,
                "required_manifest": "SEALED_SUBJECT00_FORMAL_BASE_FOR_THREE_GARMENT_ENDPOINTS_MANIFEST",
                "required_manifest_exists": False,
                "dependency": FORMAL_DEPENDENCY,
            },
            "storage": {
                "classification": "SUBJECT00_STORAGE_MIGRATION_PLAN_READY",
                "current_blocker": "STORAGE_CAPACITY",
                "current_free_bytes_at_adjudication": 17690701824,
                "live_free_bytes_at_recovery_audit": 17332740096,
                "contract_required_bytes": 32212254720,
                "status": "BLOCKED",
                "recommended_plan": "PLAN_C",
                "next_task_authorized": False,
            },
            "authoritative_sources": [
                {"branch": ORIGINAL_PROTOCOL_BRANCH, "head": ORIGINAL_PROTOCOL_HEAD},
                {"branch": GENERATION_WORKFLOW_BRANCH, "head": GENERATION_WORKFLOW_HEAD},
                {"branch": FOUNDATIONS_BRANCH, "head": FOUNDATIONS_HEAD},
                {"branch": FORMAL_PROTOCOL_BRANCH, "head": FORMAL_PROTOCOL_HEAD},
                {"branch": STORAGE_BRANCH, "head": STORAGE_HEAD},
            ],
            "immutability_baseline": {
                "task_b_source_head": SOURCE_HEAD,
                "formal_protocol_head": FORMAL_PROTOCOL_HEAD,
                "storage_adjudication_head": STORAGE_HEAD,
                "subject00_raw_fingerprint": RAW_FINGERPRINT,
                "subject00_raw_files": 119412,
                "subject00_raw_bytes": 26459647641,
            },
            "execution_counts": {
                "image_generation_api": 0,
                "vlm_api": 0,
                "renderer": 0,
                "training": 0,
                "teacher_endpoints": 0,
                "controller_runs": 0,
                "materialized_images": 0,
                "raw_mutations": 0,
                "formal_base_mutations": 0,
                "task_b_source_mutations": 0,
            },
            "paper_final": False,
        }
    )


def build_identity_registry() -> dict[str, Any]:
    records = [source_record(slot) for slot in SLOTS]
    return seal(
        {
            "schema_version": "canondressgs.subject00.identity_source_registry.v1",
            "task_id": TASK_ID,
            "status": "IDENTITY_SOURCE_SET_SELECTED",
            "identity_id": "subject00",
            "identity_source_set_id": IDENTITY_SET_ID,
            "identity_count": 1,
            "pose_frame_id": 0,
            "view_count": 8,
            "selection_rule": {
                "hard_filters": [
                    "real traceable THuman4.0 Subject00 only",
                    "strict-train pose and camera only",
                    "full body visible",
                    "both hands and both feet present",
                    "valid mask available",
                    "single shared pose across all views",
                    "no generated face or body substitution",
                ],
                "deterministic_tie_break": "lowest strict-train frame ID already covered by canary and medium-pilot original-detail review",
                "camera_rule": "nearest strict-train camera after frozen orientation=(camera_azimuth+90) mod 360 calibration; camera_id tie-break",
                "future_generation_quality_used": False,
            },
            "assessment": {
                "face_clarity": "PASS_FRONT_AND_FRONT_THREE_QUARTER_VIEWS",
                "hair_visibility": "PARTIAL_RAW_HOOD_OCCLUSION; VISIBLE_HAIRLINE_PRESERVED; NO_HAIRSTYLE_INFERENCE",
                "full_body_completeness": "PASS_ALL_8",
                "body_proportion": "PASS_REAL_CAPTURE_UNMODIFIED",
                "hands_feet": "PASS_ALL_8",
                "neutral_lighting": "PASS_FIXED_CAPTURE_LIGHTING",
                "background": "PASS_FIXED_CAPTURE_LAB; NOT_SYNTHETIC_NEUTRAL",
                "camera_consistency": "PASS_SHARED_CALIBRATION",
                "mask_quality": "PASS_VALID_OFFICIAL_MASKS",
                "pose_compatibility": "PASS_STRICT_TRAIN_POSE_0",
                "known_risk": "hood limits full hairstyle evidence; generated candidates must preserve visible face/hairline and may not invent a hairstyle",
            },
            "source_branch": FORMAL_PROTOCOL_BRANCH,
            "source_head": FORMAL_PROTOCOL_HEAD,
            "raw_fingerprint": RAW_FINGERPRINT,
            "records": records,
            "subject02_identity_pixels_used": False,
            "generated_identity_pixels_used": False,
            "paper_final": False,
        }
    )


def inventory_record(
    asset_id: str,
    category: str,
    absolute_path: str,
    logical_path: str,
    byte_count: int,
    sha256: str,
    resolution: list[int] | None,
    camera_id: Any,
    frame_pose_id: Any,
    role: str,
    source: str,
    generation_input_eligible: bool,
    formal_split_authorization: str,
) -> dict[str, Any]:
    return {
        "asset_id": asset_id,
        "category": category,
        "absolute_path": absolute_path,
        "logical_path": logical_path,
        "bytes": byte_count,
        "sha256": sha256,
        "resolution": resolution,
        "camera_id": camera_id,
        "frame_pose_id": frame_pose_id,
        "role": role,
        "source_branch_or_output": source,
        "generation_input_eligible": generation_input_eligible,
        "formal_split_authorization": formal_split_authorization,
        "existence_verified": True,
    }


def build_asset_inventory() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for slot in SLOTS:
        camera_id = slot["camera_id"]
        camera = f"cam{camera_id:02d}"
        assets = RAW_ASSETS[camera_id]
        records.append(
            inventory_record(
                f"subject00_rgb_pose00000000_{camera}",
                "identity_images_and_condition_rgb",
                f"{RAW_ROOT}/images/{camera}/00000000.jpg",
                f"subject00/images/{camera}/00000000.jpg",
                assets["rgb_bytes"],
                assets["rgb_sha256"],
                [1330, 1150],
                camera_id,
                0,
                f"IDENTITY_SOURCE_AND_{slot['slot_id'].upper()}_SOURCE_RGB",
                f"{FORMAL_PROTOCOL_BRANCH}@{FORMAL_PROTOCOL_HEAD}",
                True,
                "STRICT_TRAIN",
            )
        )
        records.append(
            inventory_record(
                f"subject00_mask_pose00000000_{camera}",
                "masks",
                f"{RAW_ROOT}/masks/{camera}/00000000.jpg",
                f"subject00/masks/{camera}/00000000.jpg",
                assets["mask_bytes"],
                assets["mask_sha256"],
                [1330, 1150],
                camera_id,
                0,
                f"{slot['slot_id'].upper()}_SOURCE_MASK",
                f"{FORMAL_PROTOCOL_BRANCH}@{FORMAL_PROTOCOL_HEAD}",
                True,
                "STRICT_TRAIN",
            )
        )
    for asset in SHARED_RAW_ASSETS:
        records.append(
            inventory_record(
                asset["asset_id"],
                asset["category"],
                asset["absolute_path"],
                asset["logical_path"],
                asset["bytes"],
                asset["sha256"],
                asset["resolution"],
                asset["camera_id"],
                asset["frame_pose_id"],
                asset["role"],
                f"{FORMAL_PROTOCOL_BRANCH}@{FORMAL_PROTOCOL_HEAD}",
                asset["generation_input_eligible"],
                asset["formal_split_authorization"],
            )
        )
    for relative, byte_count, sha256 in DERIVED_ASSETS:
        category = "surface_lbs" if relative.startswith("surface_lbs/") else "template"
        records.append(
            inventory_record(
                "subject00_" + relative.replace("/", "_").replace(".", "_"),
                category,
                f"{DERIVED_ROOT}/{relative}",
                f"derived_assets/subject00/{relative}",
                byte_count,
                sha256,
                None,
                None,
                None,
                "FORMAL_BASE_SUPPORT_ASSET",
                "research/mmlphuman-subject00-surface-lbs-runtime-20260723",
                False,
                "SUPPORT_ASSET_NOT_A_CONDITION_IMAGE",
            )
        )
    for asset_id, category, path, byte_count, sha256, resolution, camera_id, pose_id in VISUAL_ASSETS:
        records.append(
            inventory_record(
                asset_id,
                category,
                path,
                path.split("/outputs/", 1)[-1],
                byte_count,
                sha256,
                resolution,
                camera_id,
                pose_id,
                "HISTORICAL_DIAGNOSTIC_VISUAL_ONLY",
                path.split("/attempt_001/", 1)[0],
                False,
                "PREVIEW_ONLY_NOT_FORMAL_GENERATION_INPUT",
            )
        )
    return seal(
        {
            "schema_version": "canondressgs.subject00.condition_asset_inventory.v1",
            "task_id": TASK_ID,
            "inventory_scope": {
                "policy": "exhaustive per-file registration for all selected generation inputs and published template/surface-LBS support files; root-level registry for large raw and historical visual banks",
                "no_large_assets_copied_to_git": True,
                "raw_authoritative_full_manifest": "/root/autodl-tmp/datasets/thuman4_second_identity_staging/reports/SUBJECT00_VALID_FRAME_CAMERA_MANIFEST.json",
                "raw_full_manifest_sha256": "cd00ad0a1549bc99d956e26eea300945adfaa10a20a438e931e640977475564e",
            },
            "roots": {
                "raw": {"path": RAW_ROOT, "files": 119412, "bytes": 26459647641, "fingerprint": RAW_FINGERPRINT, "status": "AVAILABLE_READ_ONLY"},
                "derived": {"path": DERIVED_ROOT, "status": "PUBLISHED_READ_ONLY"},
                "condition_renders": {"path": FORMAL_OUTPUT_ROOT, "status": "ABSENT_FORMAL_BASE_PENDING"},
                "white_model_clay_renders": {"path": "/root/autodl-tmp/canondressgs_work/outputs/SUBJECT00-MMLPHUMAN-SHORT-CANARY-001/attempt_001/visuals/step_000000", "status": "AVAILABLE_PREVIEW_ONLY"},
                "canary_visuals": {"path": "/root/autodl-tmp/canondressgs_work/outputs/SUBJECT00-MMLPHUMAN-SHORT-CANARY-001/attempt_001/visuals", "status": "AVAILABLE_PREVIEW_ONLY"},
                "medium_pilot_visuals": {"path": "/root/autodl-tmp/canondressgs_work/outputs/SUBJECT00-MMLPHUMAN-MEDIUM-PILOT-001/attempt_001/visuals", "status": "AVAILABLE_PREVIEW_ONLY"},
            },
            "category_status": {
                "raw_data": "RECOVERED_ROOT_AND_AUTHORITATIVE_MANIFEST",
                "derived_assets": "RECOVERED_15_PUBLISHED_FILES",
                "identity_images": "RECOVERED_8_SELECTED_FILES",
                "masks": "RECOVERED_8_SELECTED_FILES",
                "calibration": "RECOVERED_SHARED_FILE",
                "smplx_parameters": "RECOVERED_SHARED_FILE",
                "camera_poses": "RECOVERED_IN_CALIBRATION_JSON",
                "template": "RECOVERED_4_FILES",
                "surface_lbs": "RECOVERED_11_FILES",
                "condition_renders": "ABSENT_FORMAL_BASE_PENDING",
                "white_model_clay_renders": "RECOVERED_PREVIEW_ONLY",
                "thumbnails": "RECOVERED_REPRESENTATIVE_CONTACT_SHEETS",
                "previous_canary_visuals": "RECOVERED_REPRESENTATIVE_RECORDS",
                "medium_pilot_visuals": "RECOVERED_REPRESENTATIVE_RECORDS",
            },
            "record_count": len(records),
            "records": records,
            "generation_eligible_asset_ids": [row["asset_id"] for row in records if row["generation_input_eligible"]],
            "formal_condition_image_count": 0,
            "paper_final": False,
        }
    )


def build_slot_binding() -> dict[str, Any]:
    bindings = []
    for garment in GARMENTS:
        for slot in SLOTS:
            source = source_record(slot)
            bindings.append(
                {
                    "condition_id": f"subject00/{garment}/{slot['semantic_pose_slot']}",
                    "identity_id": "subject00",
                    "garment_id": garment,
                    "slot_id": slot["slot_id"],
                    "semantic_pose_slot": slot["semantic_pose_slot"],
                    "orientation_degrees": slot["orientation_degrees"],
                    "roles": ["TEACHER_TARGET"] + (["GARMENT_REFERENCE"] if slot["reference_role"] else []),
                    "camera_id": source["camera_id"],
                    "camera_azimuth_degrees": source["camera_azimuth_degrees"],
                    "calibrated_view_orientation_degrees": source["calibrated_view_orientation_degrees"],
                    "orientation_error_degrees": source["orientation_error_degrees"],
                    "pose_frame_id": source["pose_frame_id"],
                    "source_rgb": {"path": source["source_rgb_path"], "sha256": source["source_rgb_sha256"], "bytes": source["source_rgb_bytes"]},
                    "mask": {"path": source["source_mask_path"], "sha256": source["source_mask_sha256"], "bytes": source["source_mask_bytes"]},
                    "camera_file": {"path": source["camera_file_path"], "sha256": source["camera_file_sha256"]},
                    "pose_smplx_file": {"path": source["pose_smplx_file_path"], "sha256": source["pose_smplx_file_sha256"]},
                    "strict_split_role": "STRICT_TRAIN",
                    "resolution": source["resolution"],
                    "binding_status": "COMPLETE_SOURCE_BINDING",
                    "target_materialization_status": "MATERIALIZATION_PENDING",
                    "accepted_teacher_target_path": f"{DATASET_ROOT}/09_teacher_targets/{garment}/{slot['slot_id']}.png",
                    "accepted_teacher_target_sha256": None,
                    "formal_base_dependency": FORMAL_DEPENDENCY,
                }
            )
    return seal(
        {
            "schema_version": "canondressgs.subject00.condition_slot_binding.v1",
            "task_id": TASK_ID,
            "status": "24_SOURCE_BINDINGS_COMPLETE_TARGETS_NOT_MATERIALIZED",
            "identity_source_set_id": IDENTITY_SET_ID,
            "front_axis_calibration": {
                "formula": "calibrated_view_orientation_degrees=(camera_azimuth_degrees+90) mod 360",
                "visual_cross_check": "pose 0 multiview images opened; cam17 is front and cam05 is back",
                "yaw_tolerance_degrees": 22.5,
                "all_selected_within_tolerance": True,
            },
            "counts": {
                "garments": 3,
                "slots_per_garment": 8,
                "source_bindings_complete": 24,
                "reference_source_bindings_complete": 12,
                "planned_teacher_targets": 24,
                "materialized_teacher_targets": 0,
                "materialized_garment_references": 0,
            },
            "strict_split": {
                "train_camera_ids": TRAIN_CAMERAS,
                "heldout_camera_ids": HELDOUT_CAMERAS,
                "train_pose_count": 1130,
                "selected_pose_id": 0,
                "pose_split_sha256": POSE_SPLIT_SHA,
                "camera_split_sha256": CAMERA_SPLIT_SHA,
                "heldout_leakage_count": 0,
                "buffer_leakage_count": 0,
            },
            "bindings": bindings,
            "remaining_gaps": [
                "generation backend provider/model/revision is not selected",
                "no generation request is authorized or executed",
                "24 accepted Teacher target pixels and SHA-256 values do not exist",
                "12 accepted cardinal garment-reference pixels and SHA-256 values do not exist",
                "Formal Base downstream manifest is absent",
            ],
            "paper_final": False,
        }
    )


def build_controller_split() -> dict[str, Any]:
    folds = {
        "fold_0": ["slot_00", "slot_04"],
        "fold_1": ["slot_01", "slot_05"],
        "fold_2": ["slot_02", "slot_06"],
        "fold_3": ["slot_03", "slot_07"],
    }
    rotations = []
    rotation_spec = [
        ([0, 1], 2, 3),
        ([1, 2], 3, 0),
        ([2, 3], 0, 1),
        ([3, 0], 1, 2),
    ]
    for index, (train_folds, calibration_fold, test_fold) in enumerate(rotation_spec):
        def logical_ids(fold_ids: list[int]) -> list[str]:
            return [
                f"subject00/{garment}/{slot_id}/accepted_teacher_target"
                for garment in GARMENTS
                for fold_id in fold_ids
                for slot_id in folds[f"fold_{fold_id}"]
            ]

        train_ids = logical_ids(train_folds)
        calibration_ids = logical_ids([calibration_fold])
        test_ids = logical_ids([test_fold])
        visual_ids = [f"subject00/{garment}/rotation_{index}/visual_only_planned" for garment in GARMENTS]
        rotations.append(
            {
                "rotation": index,
                "train_folds": train_folds,
                "calibration_fold": calibration_fold,
                "test_fold": test_fold,
                "training_references": {"status": "MATERIALIZATION_PENDING", "logical_ids": train_ids},
                "calibration_references": {"status": "MATERIALIZATION_PENDING", "logical_ids": calibration_ids},
                "test_references": {"status": "MATERIALIZATION_PENDING", "logical_ids": test_ids},
                "visual_only_references": {
                    "status": "MATERIALIZATION_PENDING_NOT_IN_48_CANDIDATE_BUDGET",
                    "logical_ids": visual_ids,
                    "materialization_authorized": False,
                },
                "logical_overlap": {
                    "train_test": sorted(set(train_ids) & set(test_ids)),
                    "calibration_test": sorted(set(calibration_ids) & set(test_ids)),
                    "train_calibration": sorted(set(train_ids) & set(calibration_ids)),
                },
            }
        )
    return seal(
        {
            "schema_version": "canondressgs.subject00.controller_split_blueprint.v1",
            "task_id": TASK_ID,
            "status": "FROZEN_BLUEPRINT_MATERIALIZATION_PENDING",
            "unit": "ACCEPTED_TEACHER_TARGET_LOGICAL_ID",
            "fold_assignment_rule": "slot ordinal modulo 4 with the frozen paired-slot table",
            "folds": folds,
            "rotations": rotations,
            "invariants": {
                "same_image_sha_across_train_test_allowed": False,
                "same_image_sha_across_calibration_test_allowed": False,
                "pixel_identical_derived_files_are_same_source": True,
                "quality_based_post_hoc_movement_allowed": False,
                "same_rule_across_garments": True,
                "teacher_test_metrics_for_controller_selection_allowed": False,
                "planned_ids_claimed_as_materialized": False,
            },
            "materialized_sha_overlap_audit": "PENDING_GENERATION_AND_ACCEPTANCE",
            "logical_overlap_count": 0,
            "paper_final": False,
        }
    )


def build_occlusion_allowlist() -> dict[str, Any]:
    return seal(
        {
            "schema_version": "canondressgs.subject00.garment_occlusion_allowlist.v1",
            "task_id": TASK_ID,
            "status": "FROZEN_STRICT_ALLOWLIST",
            "shared_allowed": [
                {"code": "NATURAL_GARMENT_BODY_OCCLUSION", "scope": "garment may cover the body regions it is designed to cover; body silhouette must remain anatomically coherent"},
                {"code": "NORMAL_ARM_TORSO_SELF_OCCLUSION", "scope": "an arm may naturally occlude a small torso or garment region in the frozen pose"},
                {"code": "VIEW_DEPENDENT_SELF_OCCLUSION", "scope": "far-side limbs and garment surfaces may be self-occluded in side/back views"},
                {"code": "LOWER_GARMENT_LEG_OCCLUSION", "scope": "trousers or jeans may cover legs to the ankles without deleting feet"},
            ],
            "garment_specific": {
                "O01": ["hood may cover the rear and sides of the head but not the face; sleeves and trousers may cover arms and legs normally"],
                "O03": ["suit jacket may cover shirt torso while collar, tie, jacket lapels, cuffs, and trouser separation remain legible"],
                "O04": ["bomber jacket may cover the inner shirt except at the intended opening; jeans may cover legs to ankle level"],
            },
            "never_allowed": [
                "face covered by garment or artifact",
                "missing hands or feet",
                "severely missing body contour",
                "garment fused into body",
                "garment region replaced by background",
                "camera crop removes any body part",
            ],
            "allowlist_may_override_hard_reject": False,
            "paper_final": False,
        }
    )


def build_reject_taxonomy() -> dict[str, Any]:
    hard = [
        ("IDENTITY_FACE_MISMATCH", "face or identity differs from Subject00"),
        ("HAIR_OR_HAIRLINE_DRIFT", "visible hairline or hair appearance changes or is invented"),
        ("SKIN_TONE_DRIFT", "skin tone changes beyond source-lighting variation"),
        ("BODY_PROPORTION_DRIFT", "height, build, or body proportions change"),
        ("POSE_MISMATCH", "frozen pose or limb configuration changes"),
        ("CAMERA_OR_VIEW_MISMATCH", "camera, framing, orientation, or perspective changes"),
        ("FACE_OCCLUDED", "face is covered by garment or artifact"),
        ("HANDS_OR_FEET_MISSING", "one or more hands or feet disappear or are cropped"),
        ("ANATOMY_FAILURE", "extra, duplicated, fused, or malformed body parts"),
        ("GARMENT_SEMANTIC_MISMATCH", "wrong category, color, cut, sleeve length, or upper/lower structure"),
        ("GARMENT_BODY_FUSION", "garment merges into body without a coherent boundary"),
        ("GARMENT_BACKGROUND_ERASURE", "background replaces a garment region"),
        ("BACKGROUND_OR_LIGHTING_DRIFT", "source background or lighting is materially altered"),
        ("EXTRA_PERSON_OR_OBJECT", "extra person or unrequested object appears"),
        ("TEXT_NUMBER_LOGO_WATERMARK", "new text, numbering, logo, watermark, border, or collage appears"),
        ("FULL_BODY_CROP_FAILURE", "camera crop breaks full-body completeness"),
        ("SEVERE_EDGE_OR_MASK_FAILURE", "edge/mask defects make the asset unusable"),
    ]
    return seal(
        {
            "schema_version": "canondressgs.subject00.garment_reject_taxonomy.v1",
            "task_id": TASK_ID,
            "status": "FROZEN",
            "hard_rejects": [{"code": code, "definition": definition, "decision": "REJECT"} for code, definition in hard],
            "quality_grades": {
                "A": "no visible artifact at inspection scale",
                "B": "minor non-scientific artifact outside identity and garment boundaries",
                "C": "material artifact requiring adjudication",
                "D": "hard failure and reject",
            },
            "quality_retry_policy": "scientific quality rejection is retained and is not an automatic retry reason",
            "vlm_final_decision_allowed": False,
            "paper_final": False,
        }
    )


def build_backend_selection() -> dict[str, Any]:
    candidates = [
        {
            "candidate": "CODEX_PLATFORM_MANAGED_DIRECT_IMAGE_EDIT",
            "local_evidence": "historical Subject02 success through image_gen.imagegen",
            "model": None,
            "model_status": "NOT_EXPOSED_BY_PLATFORM",
            "image_conditioning": "HISTORICALLY_USED",
            "identity_preservation": "HISTORICAL_SUCCESS_ONLY_NOT_CURRENTLY_VERIFIED",
            "mask_pose_support": "UNVERIFIED",
            "input_count": "UNVERIFIED",
            "output_resolution": "historical 1024x1536 portrait",
            "batch": "UNVERIFIED",
            "response_retention": "platform result asset only; raw provider response unavailable",
            "verified_cost": False,
            "content_policy_risk": "UNVERIFIED_FOR_THIS_DATASET",
            "retry": "technical retry contract reusable after backend binding",
            "reproducibility": "request provenance only; seed/model revision unavailable",
            "selection_result": "NOT_SELECTED",
        },
        {
            "candidate": "HISTORICAL_OPENAI_COMPATIBLE_PROXY_SUBLYX",
            "local_evidence": "E:/data_pre/scripts/run_gpt_image2_pose_outfit_sheets.py",
            "evidence_sha256": "2a431927e591e60a31ff796d8a5a774b415b58b0f31c0b8b3096c4c26ea6ae04",
            "historical_endpoint_identifier": "api.sublyx.org/v1/images/edits",
            "model": "historical alias gpt-image-2; revision unpinned",
            "image_conditioning": "IMPLEMENTED_WITH_IDENTITY_AND_CONDITION_IMAGES",
            "identity_preservation": "NOT_CURRENTLY_VERIFIED",
            "mask_pose_support": "condition image only; mask not used",
            "input_count": "multiple images historically accepted; current limit unverified",
            "output_resolution": "1536x1024 wire size",
            "batch": "n=1 concurrency=1",
            "response_retention": "status/request-id and optional JSON historically available; unsafe dumps forbidden",
            "verified_cost": False,
            "content_policy_risk": "provider-side only and not frozen",
            "retry": "2 historical retries; prospective 3/9 second technical backoff",
            "reproducibility": "request provenance only; seed absent",
            "selection_result": "NOT_SELECTED",
        },
        {
            "candidate": "HISTORICAL_OPENAI_COMPATIBLE_PROXY_78CODE",
            "local_evidence": "E:/data_pre/scripts/run_jay_coverage_supplement_v1.py",
            "evidence_sha256": "f3b493f9da325e38683260fdedb6315921c6f642bab1fbe8cb192698f2036e8b",
            "historical_endpoint_identifier": "www.78code.cc/v1/images/edits",
            "model": "CLI required; historically documented as gpt-image-2 without revision",
            "image_conditioning": "HISTORICALLY_IMPLEMENTED",
            "identity_preservation": "NOT_CURRENTLY_VERIFIED",
            "mask_pose_support": "UNVERIFIED",
            "input_count": "UNVERIFIED",
            "output_resolution": "1536x1024 wire size",
            "batch": "n=1; retries=2 historically",
            "response_retention": "NOT_VERIFIED_PROSPECTIVELY",
            "verified_cost": False,
            "content_policy_risk": "UNVERIFIED",
            "retry": "seed explicitly unsupported in historical fingerprint",
            "reproducibility": "request provenance only",
            "selection_result": "NOT_SELECTED",
        },
    ]
    reusable = {
        "REUSABLE_WITHOUT_CHANGE": [
            "append-only request, failure, retry, review, and provenance events",
            "two independent initial human reviewers plus human adjudication",
            "2 ACCEPT accepts; any REJECT rejects; all other pairs adjudicate",
            "VLM final tie-break forbidden",
            "technical-only retry taxonomy with 3/9 second backoff and maximum two retries",
            "credential-value, length, prefix, fingerprint, and authorization-header persistence forbidden",
            "PNG 1024x1536 portrait normalized output contract",
        ],
        "REUSABLE_WITH_SUBJECT00_BINDING": [
            "request schema and deterministic naming",
            "identity/condition image binding",
            "garment prompt templates",
            "review queue and localhost review server",
            "provider request/response field mapping after user selection",
        ],
        "NOT_REUSABLE": [
            "Subject02 identity pixels, paths, condition IDs, frame IDs, or accepted target pixels",
            "historical unpinned provider/model labels",
            "historical credential length or SHA-prefix logging",
            "historical unsafe raw debug-response dumps",
            "historical 160-candidate budget",
        ],
    }
    return seal(
        {
            "schema_version": "canondressgs.subject00.generation_backend_selection.v1",
            "task_id": TASK_ID,
            "status": BACKEND_STATUS,
            "selection": None,
            "network_or_connectivity_probe_performed": False,
            "paid_call_performed": False,
            "comparison_basis": "local scripts, manifests, historical logs, and frozen workflow contracts only",
            "historical_execution_counts": {"Jay_records": 915, "Jay_success": 494, "Rose_records": 1989, "Rose_success": 293},
            "candidates": candidates,
            "subject02_workflow_reuse": reusable,
            "selection_blockers": [
                "user has not selected a provider and exact model/revision or alias policy",
                "current provider capability, image count, mask support, seed behavior, price, rate limits, data processing, content policy, and paper-use license are unverified",
                "no connectivity or paid request is authorized",
                "platform-managed route does not expose an exact backend model",
            ],
            "credential_policy": {
                "prospective_env_name": "MULTI_IDENTITY_GENERATION_API_KEY",
                "credential_read_in_this_task": False,
                "persist_value_length_prefix_fingerprint_or_header": False,
                "dry_run_reads_credential": False,
            },
            "request_and_review_workflow_source": {
                "branch": GENERATION_WORKFLOW_BRANCH,
                "head": GENERATION_WORKFLOW_HEAD,
                "workflow_sha256": "135df99fbdef71ffffcdc8a54822fd25ed5a562c4153a4b26016bd327c4161aa",
                "review_server_sha256": "5f4861f58bc47f85f62a80c19ca83450af888faa7fef00ab9e5870ed71f7e360",
            },
            "image_generation_api_calls": 0,
            "external_api_calls": 0,
            "paper_final": False,
        }
    )


def build_prompt_registry() -> dict[str, Any]:
    records = []
    shared_positive = (
        "Use the Subject00 source RGB as the only identity and pose source. Preserve the same person, face, visible hairline, skin tone, body proportions, full-body pose, camera, framing, background, and lighting. "
        "Edit clothing only. Keep both hands and both feet complete and visible. "
    )
    shared_negative = (
        "different identity, changed face, invented or changed hairstyle, changed skin tone, changed body proportions, changed pose, changed camera, changed framing, cropped body, close-up, missing hand, missing foot, extra limb, fused anatomy, extra person, added scene, changed background, changed lighting, accessory drift, text, number, logo, watermark, border, collage, garment redesign, exaggerated fashion styling"
    )
    for garment in GARMENTS:
        semantics = GARMENT_SEMANTICS[garment]
        positive = (
            shared_positive
            + "Replace the current outfit with exactly this frozen garment: "
            + semantics["reference_prompt"]
            + " Match the frozen color, cut, sleeve length, upper/lower structure, fit, and hem. Do not redesign the garment."
        )
        negative = shared_negative + ", " + ", ".join(semantics["forbidden"])
        records.append(
            {
                "prompt_id": f"subject00_{garment}_positive_v1",
                "negative_constraint_id": f"subject00_{garment}_negative_v1",
                "garment_id": garment,
                "semantic_description": semantics["description"],
                "positive_prompt": positive,
                "positive_prompt_sha256": text_sha256(positive),
                "negative_constraints": negative,
                "negative_constraints_sha256": text_sha256(negative),
                "backend_specific_mapping": {
                    "status": "PENDING_BACKEND_SELECTION",
                    "source_image_role": "identity_and_pose_condition",
                    "mask_role": "provide only if selected backend supports and contract pins mask semantics",
                    "garment_semantics_role": "positive prompt; donor pixels are not yet frozen",
                },
            }
        )
    return seal(
        {
            "schema_version": "canondressgs.subject00.generation_prompt_registry.v1",
            "task_id": TASK_ID,
            "status": "PROMPTS_FROZEN_BACKEND_MAPPING_PENDING",
            "records": records,
            "prompt_count": 3,
            "identity_change_allowed": False,
            "pose_or_camera_change_allowed": False,
            "garment_redesign_allowed": False,
            "test_result_tuning_used": False,
            "paper_final": False,
        }
    )


def build_request_manifest(prompt_registry: dict[str, Any]) -> dict[str, Any]:
    prompt_by_garment = {row["garment_id"]: row for row in prompt_registry["records"]}
    requests = []
    for garment in GARMENTS:
        prompt = prompt_by_garment[garment]
        for slot in SLOTS:
            source = source_record(slot)
            for candidate_index in range(2):
                request_id = f"subject00_{garment}_{slot['slot_id'].replace('_', '')}_cand{candidate_index:02d}"
                requests.append(
                    {
                        "request_id": request_id,
                        "identity_id": "subject00",
                        "identity_source_set_id": IDENTITY_SET_ID,
                        "garment_id": garment,
                        "slot_id": slot["slot_id"],
                        "semantic_pose_slot": slot["semantic_pose_slot"],
                        "candidate_index": candidate_index,
                        "attempt_index": 0,
                        "condition_id": f"subject00/{garment}/{slot['semantic_pose_slot']}",
                        "expected_orientation_degrees": slot["orientation_degrees"],
                        "camera_id": slot["camera_id"],
                        "pose_frame_id": 0,
                        "input_file_paths": {
                            "identity_condition_rgb": source["source_rgb_path"],
                            "identity_condition_mask": source["source_mask_path"],
                            "camera": source["camera_file_path"],
                            "pose_smplx": source["pose_smplx_file_path"],
                            "garment_reference": None,
                        },
                        "input_sha256": {
                            "identity_condition_rgb": source["source_rgb_sha256"],
                            "identity_condition_mask": source["source_mask_sha256"],
                            "camera": source["camera_file_sha256"],
                            "pose_smplx": source["pose_smplx_file_sha256"],
                            "garment_reference": None,
                        },
                        "source_condition_sha256": source["source_rgb_sha256"],
                        "prompt_id": prompt["prompt_id"],
                        "prompt_sha256": prompt["positive_prompt_sha256"],
                        "negative_constraint_id": prompt["negative_constraint_id"],
                        "negative_constraint_sha256": prompt["negative_constraints_sha256"],
                        "backend": None,
                        "model": None,
                        "backend_selection_status": BACKEND_STATUS,
                        "seed": None,
                        "seed_status": "PENDING_BACKEND_CAPABILITY_ADJUDICATION",
                        "output_count": 1,
                        "output_format": "PNG",
                        "target_resolution": {"width": 1024, "height": 1536, "orientation": "portrait"},
                        "retry_policy": {
                            "maximum_retries_after_initial_attempt": 2,
                            "backoff_seconds": [3, 9],
                            "technical_errors_only": True,
                            "quality_rejection_retry_allowed": False,
                        },
                        "raw_response_path": f"{DATASET_ROOT}/04_generation_responses/{request_id}/raw_response_metadata.json",
                        "accepted_output_path": f"{DATASET_ROOT}/06_accepted_rgb/{garment}/{slot['slot_id']}.png",
                        "provenance_path": f"{DATASET_ROOT}/11_provenance/{request_id}.jsonl",
                        "credential_env_name": "MULTI_IDENTITY_GENERATION_API_KEY",
                        "credential_value_persisted": False,
                        "authorized": False,
                        "executed": False,
                        "response_count": 0,
                        "materialization_status": "MATERIALIZATION_PENDING_BACKEND_SELECTION_AND_AUTHORIZATION",
                    }
                )
    request_set_payload = [{k: v for k, v in row.items() if k != "request_manifest_sha256"} for row in requests]
    request_set_sha256 = canonical_sha256(request_set_payload)
    for row in requests:
        row["request_manifest_sha256"] = request_set_sha256
    return seal(
        {
            "schema_version": "canondressgs.subject00.generation_request_manifest.v1",
            "task_id": TASK_ID,
            "status": "48_REQUESTS_PREPARED_BACKEND_SELECTION_REQUIRED",
            "request_manifest_sha256_semantics": "canonical SHA-256 of the ordered request entries before adding the shared request_manifest_sha256 field",
            "request_set_sha256": request_set_sha256,
            "counts": {"garments": 3, "slots_per_garment": 8, "candidates_per_slot": 2, "requests": 48},
            "requests": requests,
            "authorized_count": 0,
            "executed_count": 0,
            "response_count": 0,
            "image_generation_api_calls": 0,
            "paper_final": False,
        }
    )


def build_review_manifest(request_manifest: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    for request in request_manifest["requests"]:
        candidates.append(
            {
                "candidate_id": request["request_id"],
                "request_id": request["request_id"],
                "candidate_path": None,
                "candidate_sha256": None,
                "identity_id": "subject00",
                "garment_id": request["garment_id"],
                "condition_id": request["condition_id"],
                "slot_id": request["slot_id"],
                "review_status": "PENDING_MATERIALIZATION",
                "initial_reviews": [],
                "adjudication_reviews": [],
                "final_decision": None,
                "reject_reasons": [],
                "previous_event_sha256": None,
            }
        )
    return seal(
        {
            "schema_version": "canondressgs.subject00.human_review_execution_manifest.v1",
            "task_id": TASK_ID,
            "status": "EMPTY_48_ENTRY_REVIEW_QUEUE",
            "decision_policy": {
                "minimum_independent_initial_reviewers": 2,
                "initial_roles": ["REVIEWER_A", "REVIEWER_B"],
                "both_accept": "ACCEPT",
                "any_reject": "REJECT",
                "all_other_complete_pairs": "HUMAN_ADJUDICATION_REQUIRED",
                "vlm_final_tie_break_allowed": False,
                "append_only_sha_hash_chain": True,
                "reviewer_initial_decisions_blinded": True,
            },
            "required_fields_per_review_event": [
                "identity_match",
                "face_match",
                "hair_match",
                "skin_tone_match",
                "body_shape_match",
                "pose_match",
                "camera_match",
                "garment_match",
                "garment_slot_match",
                "mask_quality",
                "edge_quality",
                "hands_feet_complete",
                "background_match",
                "lighting_match",
                "artifact_grade",
                "decision",
                "reject_reason",
            ],
            "candidate_count": 48,
            "review_event_count": 0,
            "accepted_count": 0,
            "rejected_count": 0,
            "candidates": candidates,
            "paper_final": False,
        }
    )


def build_directory_binding() -> dict[str, Any]:
    return seal(
        {
            "schema_version": "canondressgs.subject00.dataset_directory_binding.v1",
            "task_id": TASK_ID,
            "cloud_root": DATASET_ROOT + "/",
            "cloud_status": "CREATED_EMPTY_METADATA_ONLY",
            "windows_mirror_planned": WINDOWS_DATASET_MIRROR + "/",
            "windows_mirror_status": "PLANNED_NOT_COPIED",
            "directories": [
                {
                    "name": name,
                    "cloud_path": f"{DATASET_ROOT}/{name}/",
                    "status": "CREATED_EMPTY_METADATA_ONLY",
                    "large_assets_present": False,
                }
                for name in DATASET_DIRS
            ],
            "allowed_in_this_task": ["empty directories", "small README", "small metadata and schema files"],
            "forbidden_in_this_task": ["large raw images", "generated candidates", "accepted images", "checkpoints"],
            "large_files_copied": 0,
            "materialized_images": 0,
            "paper_final": False,
        }
    )


def build_test_report_template() -> dict[str, Any]:
    return {
        "schema_version": "canondressgs.subject00.generation_readiness_tests.v1",
        "task_id": TASK_ID,
        "status": "NOT_RUN",
        "checks": [],
        "summary": {"total": 0, "passed": 0, "failed": 0},
        "execution_boundary_verified": False,
        "paper_final": False,
    }


def build_final_summary() -> dict[str, Any]:
    return seal(
        {
            "schema_version": "canondressgs.subject00.generation_readiness_final_summary.v1",
            "task_id": TASK_ID,
            "branch": BRANCH,
            "base_head": SOURCE_HEAD,
            "data_prep_result_head": DATA_PREP_RESULT_HEAD,
            "data_prep_result_head_resolution": "git rev-parse HEAD after the data binding commit",
            "final_reporting_head_resolution": "git rev-parse HEAD after the final audit seal commit",
            "scientific_semantics": "SECOND_IDENTITY_REPLICATION",
            "identity_source_status": "IDENTITY_SOURCE_SET_SELECTED",
            "identity_source_set_id": IDENTITY_SET_ID,
            "garments": GARMENTS,
            "counts": {
                "garments": 3,
                "slots_per_garment": 8,
                "source_slot_bindings_complete": 24,
                "planned_references": 12,
                "planned_teacher_targets": 24,
                "planned_candidates": 48,
                "request_entries": 48,
                "review_queue_entries": 48,
                "materialized_images": 0,
                "accepted_teacher_targets": 0,
                "endpoint_runs": 0,
                "controller_runs": 0,
            },
            "backend_status": BACKEND_STATUS,
            "backend_selection": None,
            "formal_base_dependency": FORMAL_DEPENDENCY,
            "formal_base_required_manifest_exists": False,
            "storage_status": "BLOCKED_CAPACITY",
            "generation_readiness_gates": {
                "identity_source_selected": True,
                "garment_semantics_frozen": True,
                "eight_slots_per_garment_frozen": True,
                "twenty_four_source_slot_bindings_complete": True,
                "forty_eight_request_entries_complete": True,
                "backend_selected": False,
                "prompts_frozen": True,
                "identity_review_frozen": True,
                "provenance_schema_frozen": True,
                "directory_structure_ready": True,
                "secret_exposure": False,
                "split_leakage": False,
                "authorization_false": True,
                "api_calls_zero": True,
            },
            "generation_may_start": False,
            "generation_stop_rule": "GENERATION_MAY_NOT_START_UNTIL_EXPLICIT_AUTHORIZATION_AND_FORMAL_BASE_ADJUDICATION",
            "blocking_reason": "backend provider/model/revision and required capability/risk evidence are not selected",
            "tests": "PENDING_CHECK_SCRIPT",
            "execution_counts": {
                "image_generation_api": 0,
                "external_api": 0,
                "vlm_api": 0,
                "renderer": 0,
                "training": 0,
                "new_dataset_images": 0,
                "review_events": 0,
                "teacher_endpoints": 0,
                "controller_runs": 0,
            },
            "paper_final": False,
            "classification": CLASSIFICATION,
            "next_task": NEXT_TASK,
            "next_task_authorized": False,
            "next_task_started": False,
        }
    )


def build_handoff() -> dict[str, Any]:
    return seal(
        {
            "schema_version": "canondressgs.subject00.data_preparation_generation_ready_handoff.v1",
            "task_id": TASK_ID,
            "source": {"branch": SOURCE_BRANCH, "head": SOURCE_HEAD, "classification": "SUBJECT00_CONDITION_RESOURCE_GAP"},
            "branch": BRANCH,
            "windows_worktree": WINDOWS_WORKTREE,
            "cloud_worktree": CLOUD_WORKTREE,
            "data_prep_result_head": DATA_PREP_RESULT_HEAD,
            "data_prep_result_head_resolution": "git rev-parse HEAD after the data binding commit",
            "final_reporting_head_resolution": "git rev-parse HEAD after the final audit seal commit",
            "classification": CLASSIFICATION,
            "identity_source_set": IDENTITY_SET_ID,
            "slot_binding_status": "24_OF_24_SOURCE_BINDINGS_COMPLETE",
            "target_materialization_status": "0_OF_24_ACCEPTED_TEACHER_TARGETS",
            "backend_status": BACKEND_STATUS,
            "formal_base_dependency": FORMAL_DEPENDENCY,
            "storage_status": "BLOCKED_CAPACITY",
            "request_manifest": "paper_protocol/reviewer_risk/subject00_generation_request_manifest.json",
            "request_count": 48,
            "authorized_requests": 0,
            "executed_requests": 0,
            "api_calls": 0,
            "materialized_images": 0,
            "dataset_root": DATASET_ROOT + "/",
            "dataset_root_status": "CREATED_EMPTY_METADATA_ONLY",
            "tests": "PENDING_CHECK_SCRIPT",
            "paper_final": False,
            "next_task": NEXT_TASK,
            "next_task_authorized": False,
            "next_task_started": False,
        }
    )


def recovery_doc() -> str:
    return f"""
# Subject00 Data Preparation Recovery Audit

Task: `{TASK_ID}`

## Recovery gate

The exact source is `{SOURCE_BRANCH}` at `{SOURCE_HEAD}`. The Git commit, Task B final summary, and Task B handoff agree on `SUBJECT00_CONDITION_RESOURCE_GAP`, `SECOND_IDENTITY_REPLICATION`, three garments, eight slots per garment, 12 planned references, 24 planned Teacher targets, 48 planned candidates, zero API calls, zero materialized images, zero endpoint/controller runs, and `PAPER_FINAL=false`.

## Current dependencies

The Storage branch `{STORAGE_BRANCH}` at `{STORAGE_HEAD}` reports `SUBJECT00_STORAGE_MIGRATION_PLAN_READY` with the live storage gate still blocked. The actual Formal Base artifacts contain no attempt, no `EXECUTION_HEAD`, no sealed checkpoint, and no sealed downstream manifest. `FORMAL_BASE_DEPENDENCY=PENDING` remains mandatory.

## Recovered assets

The cloud raw root contains the frozen 24-camera/2500-frame Subject00 capture, official masks, `calibration.json`, and `smpl_params.npz`. The published Subject00 template and surface-LBS support files are present. Historical canary and medium-pilot visuals are preview-only and are forbidden as formal generation inputs. No Formal Base condition-render root exists.

The selected identity source is pose 0 across eight strict-train cameras. The source images were opened during this audit and are also covered by prior canary/medium-pilot original-detail review. Full-body completeness, hands, feet, pose, camera consistency, and official masks pass. The raw hood partially occludes hairstyle evidence; the contract preserves only the visible hairline and forbids hairstyle invention.

## Mutation boundary

No Task B, Formal Base, Storage, raw, derived, or historical output asset was modified. No renderer, training, image/VLM API, or endpoint/controller execution occurred. Large assets remain outside Git.

`PAPER_FINAL=false`.
"""


def slot_doc() -> str:
    rows = "\n".join(
        f"| {slot['slot_id']} | {slot['semantic_pose_slot']} | {slot['orientation_degrees']} | {slot['camera_id']} | 0 | {slot['orientation_error_degrees']:.3f} | STRICT_TRAIN |"
        for slot in SLOTS
    )
    return f"""
# Subject00 Condition Slot Binding

Task: `{TASK_ID}`

The front-axis calibration is `view_orientation=(camera_azimuth+90 degrees) mod 360`. Pose 0 is frozen for every garment and every slot. Each selected camera is strict-train, has a valid RGB/mask pair, uses the frozen shared calibration/SMPL-X files, and is within the 22.5-degree yaw tolerance.

| Slot | Semantic view | Target degrees | Camera | Pose | Error degrees | Split |
|---|---|---:|---:|---:|---:|---|
{rows}

The same eight source conditions are reused for O01, O03, and O04, yielding 24 complete source bindings. Cardinal slots 00/03/04/07 are both `GARMENT_REFERENCE` and `TEACHER_TARGET`, for 12 reference bindings. This does not claim generated pixels: accepted Teacher targets and references remain `MATERIALIZATION_PENDING`, with actual count zero.

Held-out cameras `[0,4,8,12,16,20]`, held-out poses, buffer-only poses, and ambiguous assets are forbidden. Current split leakage is zero.
"""


def backend_doc() -> str:
    return f"""
# Subject00 Generation Backend Selection

Task: `{TASK_ID}`

## Decision

`{BACKEND_STATUS}`. No provider was selected and no connectivity or paid request was made.

The local evidence covers three historical surfaces: the platform-managed Codex image-edit route, the Sublyx OpenAI-compatible proxy, and the 78Code proxy. Historical executions establish that image-edit workflows existed, but they do not establish one current provider/model/revision contract. The platform route does not expose its exact model. The proxy routes lack current capability, image-count, mask, seed, price, rate-limit, data-processing, content-policy, and paper-use verification. Provider selection also carries user payment and credential-risk decisions.

## Reuse

Append-only provenance, deterministic naming, technical-only retry policy, output normalization, two blinded initial human reviews, human adjudication, and the no-secret-persistence policy are reusable. Subject00 paths/prompts must be rebound. Subject02 identity pixels, condition IDs, accepted targets, unpinned backend labels, and historical credential fingerprint logging are not reusable.

The only persisted credential field is the prospective environment-variable name `MULTI_IDENTITY_GENERATION_API_KEY`. This task did not read, print, hash, fingerprint, or store any credential value.
"""


def request_doc(request_manifest: dict[str, Any]) -> str:
    return f"""
# Subject00 Generation Request Plan

Task: `{TASK_ID}`

The frozen budget is exactly `3 garments x 8 slots x 2 candidates = 48` requests. Stable IDs range from `subject00_O01_slot00_cand00` through `subject00_O04_slot07_cand01`. The canonical request-set SHA-256 is `{request_manifest['request_set_sha256']}`.

Every entry binds one real Subject00 RGB/mask condition, shared calibration and SMPL-X provenance, expected view, garment-specific prompt/negative hashes, one PNG output, and a 1024x1536 portrait normalization target. Backend, model, seed, and garment donor image remain null pending backend adjudication. Each record includes planned raw-response, accepted-output, and append-only provenance paths.

All 48 entries have `authorized=false`, `executed=false`, and `response_count=0`. No request may start before explicit authorization and Formal Base adjudication. Technical retries are limited to two with 3/9-second backoff; quality rejection is never an automatic retry reason.
"""


def review_doc() -> str:
    return f"""
# Subject00 Human Review Execution Plan

Task: `{TASK_ID}`

The manifest contains 48 empty candidate queue entries and zero review events. It does not invent review results.

Each candidate requires two blinded, independent initial reviewers. Two `ACCEPT` decisions accept; any `REJECT` rejects; every other complete pair requires a human adjudicator. A VLM may not make the final decision or break a tie. Review events are append-only and SHA-chained, and candidate pixels are read-only.

Required review dimensions cover identity, face, hair, skin tone, body shape, pose, camera, garment semantics, slot semantics, mask/edge quality, hands/feet, background, lighting, artifact grade, decision, and reject reason. The strict occlusion allowlist cannot override a hard reject.
"""


def main() -> None:
    recovery = build_recovery_audit()
    identity = build_identity_registry()
    inventory = build_asset_inventory()
    binding = build_slot_binding()
    split = build_controller_split()
    allowlist = build_occlusion_allowlist()
    taxonomy = build_reject_taxonomy()
    backend = build_backend_selection()
    prompts = build_prompt_registry()
    requests = build_request_manifest(prompts)
    reviews = build_review_manifest(requests)
    directory = build_directory_binding()
    summary = build_final_summary()
    handoff = build_handoff()

    outputs = {
        "paper_protocol/reviewer_risk/subject00_data_preparation_recovery_audit.json": recovery,
        "paper_protocol/reviewer_risk/subject00_identity_source_registry.json": identity,
        "paper_protocol/reviewer_risk/subject00_condition_asset_inventory.json": inventory,
        "paper_protocol/reviewer_risk/subject00_condition_slot_binding.json": binding,
        "paper_protocol/reviewer_risk/subject00_controller_split_blueprint.json": split,
        "paper_protocol/reviewer_risk/subject00_garment_occlusion_allowlist.json": allowlist,
        "paper_protocol/reviewer_risk/subject00_garment_reject_taxonomy.json": taxonomy,
        "paper_protocol/reviewer_risk/subject00_generation_backend_selection.json": backend,
        "paper_protocol/reviewer_risk/subject00_generation_prompt_registry.json": prompts,
        "paper_protocol/reviewer_risk/subject00_generation_request_manifest.json": requests,
        "paper_protocol/reviewer_risk/subject00_human_review_execution_manifest.json": reviews,
        "paper_protocol/reviewer_risk/subject00_dataset_directory_binding.json": directory,
        "paper_protocol/reviewer_risk/subject00_generation_readiness_tests.json": build_test_report_template(),
        "paper_protocol/reviewer_risk/subject00_generation_readiness_final_summary.json": summary,
        "project_control_handoff/subject00_data_preparation_generation_ready_handoff.json": handoff,
    }
    for relative, payload in outputs.items():
        write_json(relative, payload)

    write_text("docs/PAPER/AAAI27_SUBJECT00_DATA_PREPARATION_RECOVERY_20260725.md", recovery_doc())
    write_text("docs/PAPER/AAAI27_SUBJECT00_CONDITION_SLOT_BINDING_20260725.md", slot_doc())
    write_text("docs/PAPER/AAAI27_SUBJECT00_GENERATION_BACKEND_SELECTION_20260725.md", backend_doc())
    write_text("docs/PAPER/AAAI27_SUBJECT00_GENERATION_REQUEST_PLAN_20260725.md", request_doc(requests))
    write_text("docs/PAPER/AAAI27_SUBJECT00_HUMAN_REVIEW_EXECUTION_PLAN_20260725.md", review_doc())

    print(
        json.dumps(
            {
                "status": "BUILT",
                "classification": CLASSIFICATION,
                "json_outputs": len(outputs),
                "docs": 5,
                "slot_bindings": len(binding["bindings"]),
                "requests": len(requests["requests"]),
                "api_calls": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
