#!/usr/bin/env python3
"""Build AvatarReX license and base-avatar preflight contracts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


TASK_ID = "AAAI27-AVATARREX-BASE-AVATAR-PREFLIGHT-001"
SOURCE_BRANCH = "research/multi-identity-garment-benchmark-foundations-20260724"
SOURCE_HEAD = "9e46e85cd02527b52e2f5bf0af5571de4c8988d2"
BRANCH = "research/avatarrex-base-avatar-preflight-20260724"
CLASSIFICATION = "AVATARREX_BASE_AVATAR_PREFLIGHT_READY_FOR_DERIVED_ASSETS"
NEXT_TASK = "PREPARE_AVATARREX_LBN1_BASE_AVATAR_DERIVED_ASSETS_WITHOUT_TRAINING"
RAW_ROOT = "/root/autodl-tmp/datasets/avatarrex_second_dataset_staging/avatarrex_lbn1"
ARCHIVE_PATH = "/root/autodl-tmp/datasets/avatarrex_second_dataset_staging/downloads/avatarrex_lbn1.7z"
ARCHIVE_SHA256 = "531bd1c71ad9b35f6ae0e2595ee531aa7ba1f83c242f18f2d0505b2dcd5fbcc1"
RAW_FINGERPRINT = "00482b7c98f6f46773fd13a3f33ebe278b9353e09fdb51fa9ed72583f7b27b15"
CALIBRATION_SHA256 = "793281a00b808976122c0d33b4dd22d0ed0a48518577345c709db6cbb3d7f315"
SMPL_SHA256 = "6ed3b3877d3895999e2636990bd417783328d695412b45d0679773e34b54613b"


def canonical_sha256(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def current_head(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, encoding="utf-8"
    ).strip()


def build_license_contract() -> dict[str, Any]:
    official_commit = "2b0f6e3b4c5af823414eb6d5f0b2e1a59954d114"
    return {
        "schema_version": "avatarrex.license_and_citation_contract.v1",
        "task_id": TASK_ID,
        "dataset": "AvatarReX",
        "subject": "avatarrex_lbn1",
        "data_provider": "Tsinghua University research team led by Yebin Liu",
        "status": "CLOSED_FOR_PRIVATE_INTERNAL_NONCOMMERCIAL_SINGLE_SITE_DERIVED_ASSET_PREPARATION",
        "official_repository": {
            "url": "https://github.com/lizhe00/AnimatableGaussians",
            "owner": "lizhe00",
            "default_branch": "master",
            "commit_sha": official_commit,
            "commit_url": f"https://github.com/lizhe00/AnimatableGaussians/commit/{official_commit}",
            "commit_date_utc": "2024-11-16T13:23:11Z",
            "retrieved_at_utc": "2026-07-23T22:09:12Z",
        },
        "official_evidence": [
            {
                "path": "AVATARREX_DATASET.md",
                "url": f"https://github.com/lizhe00/AnimatableGaussians/blob/{official_commit}/AVATARREX_DATASET.md",
                "git_blob_sha": "960a113a131aead59cd3957ac91f6a7931b25eb0",
                "sha256": "14913f3414cf178af0fd6e00fc74145294c5a6c7bf748bf603d0ae7afe96ac60",
                "role": "PRIMARY_DATASET_LICENSE_AND_CITATION_EVIDENCE",
            },
            {
                "path": "LICENSE",
                "url": f"https://github.com/lizhe00/AnimatableGaussians/blob/{official_commit}/LICENSE",
                "git_blob_sha": "7742db7dc49835632f14c437758e7cc28f7ff3c6",
                "sha256": "e1d4453994f8ca902395fadbe6e922280a86acbbe04edf008f0966147cca3922",
                "role": "CODE_LICENSE_ONLY_NOT_A_DATASET_LICENSE_SUBSTITUTE",
            },
            {
                "path": "README.md",
                "url": f"https://github.com/lizhe00/AnimatableGaussians/blob/{official_commit}/README.md",
                "git_blob_sha": "d170c9a0453094cd5b4c46d147eaf52ce0bf51de",
                "sha256": "d7adaebc161d3d6ac70aa2077b42b2f3508c00094e957903023b2898bbbf915b",
                "role": "OFFICIAL_CODE_DATASET_AND_CITATION_CONTEXT",
            },
            {
                "path": "PREPROCESSED_DATASET.md",
                "url": f"https://github.com/lizhe00/AnimatableGaussians/blob/{official_commit}/PREPROCESSED_DATASET.md",
                "git_blob_sha": "8df3879be0010a28b0d1f459a44fab87d7c0d418",
                "sha256": "6e01922045bfae4178ea6881765f0dec8e6b8280fe8a193a2ca591124ca90620",
                "role": "OFFICIAL_PREPROCESSED_ASSET_INDEX",
            },
            {
                "path": "gen_data/GEN_DATA.md",
                "url": f"https://github.com/lizhe00/AnimatableGaussians/blob/{official_commit}/gen_data/GEN_DATA.md",
                "git_blob_sha": "766cf7950adce5fb36722f768839f854803fd037",
                "sha256": "347584c0a3986a5fd634361371c793be5f0146c8518d5f6b855efc4515ecd489",
                "role": "OFFICIAL_RECONSTRUCTION_PIPELINE_EVIDENCE",
            },
        ],
        "web_page_evidence": {
            "animatable_gaussians_project_page": {
                "url": "https://animatable-gaussians.github.io/",
                "status_code": 200,
                "title": "Projectpage of Animatable Gaussians",
                "sha256_utf8_content": "79143899e7e2bf0c893659ddf8898c607daf8ddc7b20beda9a45719f7ab70cf2",
                "retrieved_at_utc": "2026-07-23T22:09:10Z",
                "role": "OFFICIAL_PROJECT_PAGE",
            },
            "avatarrex_historical_project_domain": {
                "url": "https://liuyebin.com/AvatarRex/",
                "status_code": 200,
                "observed_title": "Hokiwin - Integrasi Fitur Canggih Tanpa Hambatan Akses",
                "sha256_utf8_content": "31537531a999a4e1fae47b8f02ff6ece55b0115db1e6796b4daa341d7616f4f5",
                "retrieved_at_utc": "2026-07-23T22:09:10Z",
                "status": "DOMAIN_CONTENT_DRIFT_EXCLUDED_FROM_LICENSE_EVIDENCE",
            },
            "official_lbn1_preprocessed_drive_page": {
                "url": "https://drive.google.com/file/d/1RDM3v5P4XF6Sp88EusDvokw-yHg6Je0C/view?usp=sharing",
                "status_code": 200,
                "file_id": "1RDM3v5P4XF6Sp88EusDvokw-yHg6Je0C",
                "title": "avatarrex_lbn1.7z",
                "mime_type": "application/x-7z-compressed",
                "size_bytes": 2271523272,
                "owner_metadata": "zerong1995@gmail.com",
                "sha256_utf8_content": "a8625713b8592e4b68ace86d954a763fec530603a80139c8c6044161bd424be6",
                "retrieved_at_utc": "2026-07-23T22:09:10Z",
                "page_hash_stability": "VOLATILE_GOOGLE_DRIVE_VIEWER_PAGE_NOT_AN_ASSET_HASH",
                "download_performed": False,
            },
        },
        "license_text_decision": {
            "allowed_purpose": "non-commercial research only",
            "commercial_use": "PROHIBITED_WITHOUT_COMMERCIAL_LICENSE",
            "pornographic_use": "PROHIBITED",
            "third_party_availability": "PROHIBITED_WITHOUT_PRIOR_WRITTEN_TSINGHUA_PERMISSION",
            "raw_dataset_copy_publish_distribute": "PROHIBITED_EXCEPT_INTERNAL_COPIES_AT_ONE_SITE_IN_THE_SAME_ORGANIZATION",
            "images_and_derived_data_third_party_transfer": "PROHIBITED_WITHOUT_PRIOR_WRITTEN_TSINGHUA_PERMISSION",
            "internal_single_site_derived_asset_preparation": "PERMITTED_FOR_THIS_NONCOMMERCIAL_RESEARCH_PROJECT",
            "public_raw_data": "PROHIBITED",
            "public_derived_assets": "PROHIBITED_WITHOUT_PRIOR_WRITTEN_TSINGHUA_PERMISSION",
            "public_data_derived_checkpoints": "TREATED_AS_DERIVED_DATA_AND_PROHIBITED_WITHOUT_PRIOR_WRITTEN_TSINGHUA_PERMISSION",
            "private_internal_templates_lbs_and_checkpoints": "PERMITTED_AT_THE_SINGLE_AUTHORIZED_ORGANIZATIONAL_SITE",
            "commercial_contact": "liuyebin@mail.tsinghua.edu.cn",
            "manual_confirmation_required_for_this_private_internal_preparation": False,
            "written_permission_required_before_public_or_third_party_release": True,
        },
        "citation_contract": {
            "publication_must_cite": ["AvatarReX", "Animatable Gaussians"],
            "avatarrex": {
                "title": "AvatarReX: Real-time Expressive Full-body Avatars",
                "authors": ["Zerong Zheng", "Xiaochen Zhao", "Hongwen Zhang", "Boning Liu", "Yebin Liu"],
                "venue": "ACM Transactions on Graphics",
                "volume": "42",
                "issue": "4",
                "year": 2023,
                "pages": "1-19",
                "doi": "10.1145/3592101",
                "url": "https://doi.org/10.1145/3592101",
            },
            "animatable_gaussians": {
                "title": "Animatable Gaussians: Learning Pose-dependent Gaussian Maps for High-fidelity Human Avatar Modeling",
                "authors": ["Zhe Li", "Zerong Zheng", "Lizhen Wang", "Yebin Liu"],
                "venue": "IEEE/CVF Conference on Computer Vision and Pattern Recognition",
                "year": 2024,
                "official_project_url": "https://animatable-gaussians.github.io/",
                "paper_url": "https://arxiv.org/pdf/2311.16096v3",
            },
        },
        "governance": {
            "code_license_equals_dataset_license": False,
            "third_party_blog_evidence_used": False,
            "legal_advice_claimed": False,
            "raw_or_derived_redistribution_authorized_by_this_contract": False,
            "paper_final": 0,
        },
    }


def build_runtime_adapter(old_adapter: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "avatarrex.zero_copy_runtime_adapter.v1",
        "task_id": TASK_ID,
        "identity_id": "avatarrex_lbn1",
        "status": "PASS",
        "standardization_mode": "READ_ONLY_ZERO_COPY_LOADER_ADAPTER",
        "source_root": RAW_ROOT,
        "record_key": ["identity_id", "canonical_camera_id", "frame_id"],
        "camera_mapping": old_adapter["camera_mapping"],
        "rgb_resolver": old_adapter["rgb_resolver"],
        "mask_resolver": old_adapter["mask_resolver"],
        "frame_naming": "zero-padded eight-digit frame_id with source suffix",
        "availability": {
            "camera_count": 16,
            "frame_count_per_camera": 1901,
            "logical_rgb_mask_pairs": 30416,
            "all_pairs_available": True,
            "source": "avatarrex_camera_pose_split.json:availability_manifest",
        },
        "calibration_adapter": {
            "mode": "READ_ONLY_IN_MEMORY",
            "contract": "avatarrex_calibration_runtime_contract.json",
            "source_sha256": CALIBRATION_SHA256,
        },
        "smpl_x_adapter": {
            "mode": "READ_ONLY_IN_MEMORY",
            "contract": "avatarrex_smplx_runtime_contract.json",
            "source_sha256": SMPL_SHA256,
        },
        "strict_split": {
            "contract": "avatarrex_camera_pose_split.json",
            "audit": "avatarrex_base_avatar_runtime_audit.json:strict_split_audit",
            "status": runtime["strict_split_audit"]["status"],
        },
        "license_guard": {
            "status": "PRIVATE_INTERNAL_NONCOMMERCIAL_SINGLE_SITE_ONLY",
            "raw_redistribution_authorized": False,
            "derived_asset_publication_authorized": False,
            "private_internal_derived_generation_after_preflight": True,
        },
        "loader_smoke": {
            "status": runtime["loader_smoke"]["status"],
            "dataset_class": runtime["loader_smoke"]["dataset_class"],
            "decoded_record_count": runtime["loader_smoke"]["decoded_record_count"],
            "camera_indices": runtime["loader_smoke"]["selected_camera_indices"],
            "frame_ids": runtime["loader_smoke"]["selected_frame_ids"],
            "runtime_audit_sha256": None,
        },
        "immutability": {
            "raw_tree_metadata_unchanged": runtime["mutation_audit"]["raw_tree_metadata_unchanged"],
            "raw_file_count": runtime["mutation_audit"]["raw_tree_metadata_after"]["file_count"],
            "raw_apparent_bytes": runtime["mutation_audit"]["raw_tree_metadata_after"]["apparent_bytes"],
            "full_rgb_copy_count": 0,
            "full_mask_copy_count": 0,
            "calibration_rewrites": 0,
            "smpl_npz_rewrites": 0,
            "raw_mutation": 0,
        },
        "paper_final": 0,
    }


def build_calibration_contract(runtime: dict[str, Any]) -> dict[str, Any]:
    audit = runtime["camera_audit"]
    return {
        "schema_version": "avatarrex.calibration_runtime_contract.v1",
        "task_id": TASK_ID,
        "status": audit["status"],
        "source": f"{RAW_ROOT}/calibration_full.json",
        "source_sha256": CALIBRATION_SHA256,
        "camera_count": audit["camera_count"],
        "source_fields": ["K", "R", "T", "distCoeff", "imgSize", "rectifyAlpha"],
        "raw_extrinsic_convention": audit["raw_extrinsic_convention"],
        "runtime_extrinsic_convention": audit["runtime_extrinsic_convention"],
        "camera_center_formula": audit["camera_center_formula"],
        "rotation_shape": audit["rotation_shape"],
        "translation_shape": audit["translation_shape"],
        "imgSize_order": audit["imgSize_order"],
        "decoded_array_order": audit["decoded_array_order"],
        "units": audit["units"],
        "distortion": audit["distortion_policy"],
        "validation_thresholds": {
            "abs_rotation_determinant_error_max": 1e-5,
            "rotation_orthogonality_max_abs_error": 1e-5,
            "camera_center_max_abs_residual": 1e-5,
            "known_3d_projection_max_abs_pixel_error": 1e-3,
            "pixel_unproject_reproject_max_abs_error": 1e-3,
        },
        "observed_maxima": {
            "abs_rotation_determinant_error": max(abs(row["rotation_determinant"] - 1.0) for row in audit["rows"]),
            "rotation_orthogonality_max_abs_error": max(
                row["rotation_orthogonality_max_abs_error"] for row in audit["rows"]
            ),
            "camera_center_max_abs_residual": max(row["camera_center_max_abs_residual"] for row in audit["rows"]),
            "known_3d_projection_max_abs_pixel_error": max(
                row["known_3d_projection_max_abs_pixel_error"] for row in audit["rows"]
            ),
            "pixel_unproject_reproject_max_abs_error": max(
                row["pixel_unproject_reproject_max_abs_error"] for row in audit["rows"]
            ),
        },
        "camera_rows": [
            {
                "canonical_camera_id": row["canonical_camera_id"],
                "raw_camera_name": row["raw_camera_name"],
                "K": row["K"],
                "R_shape": row["R_shape"],
                "T_shape": row["T_shape"],
                "distCoeff": row["distortion_coefficients"],
                "imgSize": row["imgSize_width_height"],
                "rectifyAlpha": row["rectifyAlpha"],
                "camera_center": row["camera_center"],
                "status": row["status"],
            }
            for row in audit["rows"]
        ],
        "mmlp_interface_notes": {
            "dataset_loader_passes_w2c_to_gsplat": True,
            "dataset_loader_uses_in_memory_undistortion": True,
            "config_image_scaling": 1,
            "non_unit_image_scaling_admitted": False,
            "scene_scale_issue": "scene.dataset.get_scene_scale currently uses w2c translation columns as camera locations; future zero-step runtime must use C=-R^T T or prove parity before canary.",
            "training_results_used_to_choose_convention": False,
        },
        "paper_final": 0,
    }


def build_smpl_contract(runtime: dict[str, Any]) -> dict[str, Any]:
    schema = runtime["smpl_x_schema_audit"]
    forward = runtime["smpl_x_neutral_forward"]
    return {
        "schema_version": "avatarrex.smplx_runtime_contract.v1",
        "task_id": TASK_ID,
        "status": "PASS" if schema["status"] == "PASS" and forward["status"] == "PASS" else "FAIL",
        "source": schema["source"],
        "source_sha256": schema["source_sha256"],
        "load_mode": "READ_ONLY_IN_MEMORY_NPZ_ALLOW_PICKLE_FALSE",
        "fields": schema["fields"],
        "frame_index_correspondence": schema["frame_index_correspondence"],
        "parameter_dtype": "float32",
        "axis_angle_convention": schema["parameter_convention"],
        "translation": {
            "runtime_contract": schema["translation_runtime_contract"],
            "explicit_unit_metadata_in_npz": schema["translation_unit_explicit_in_npz"],
        },
        "gender": schema["gender"],
        "gender_reason": schema["gender_reason"],
        "neutral_model_compatibility": {
            "status": forward["status"],
            "claim": forward["compatibility_claim"],
            "model_path": forward["model_path"],
            "model_sha256": forward["model_sha256"],
            "model_bytes": forward["model_bytes"],
            "vertex_count": forward["model_vertex_count"],
            "face_count": forward["model_face_count"],
            "frame_ids": [row["frame_id"] for row in forward["rows"]],
            "all_outputs_finite": all(
                row["vertices_finite"] and row["joints_finite"] for row in forward["rows"]
            ),
            "does_not_establish_gender": True,
        },
        "mmlp_mapping": {
            "pose_165": "global_orient(3) + body_pose(63) + zero jaw slot(3) + zero eye slots(6) + left_hand_pose(45) + right_hand_pose(45)",
            "Th": "transl[frame_id]",
            "Rh": "identity matrix; global orientation remains in pose[0:3]",
            "beta": "betas[0]",
            "jaw_pose_consumed_by_current_mmlp_body_deformation": False,
            "expression_consumed_by_current_mmlp_body_deformation": False,
        },
        "subject00_comparison": {
            "same": [
                "Both use AVRexDataset.load_pose_data and the same 165D pose assembly.",
                "Both pass transl as Th, identity as Rh, and betas[0] as beta.",
            ],
            "different": [
                "AvatarReX uses calibration_full.json and camera-root RGB/mask paths.",
                "subject00 uses calibration.json and images/{camera}/ plus masks/{camera}/ paths.",
                "Identity shape values, frame counts, camera names, and capture calibration differ.",
            ],
        },
        "paper_final": 0,
    }


def build_template_audit(license_contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "avatarrex.template_route_audit.v1",
        "task_id": TASK_ID,
        "status": "ROUTE_SELECTED_ASSET_NOT_ACQUIRED",
        "license_policy": "PRIVATE_INTERNAL_NONCOMMERCIAL_SINGLE_SITE_ONLY",
        "candidates": [
            {
                "id": "A_OFFICIAL_PREPROCESSED_LOOSE_CLOTHING_TEMPLATE",
                "availability": "OFFICIALLY_INDEXED_FOR_AVATARREX_LBN1",
                "source": "AnimatableGaussians PREPROCESSED_DATASET.md at pinned commit",
                "drive_file_id": "1RDM3v5P4XF6Sp88EusDvokw-yHg6Je0C",
                "drive_title": "avatarrex_lbn1.7z",
                "drive_page_size_bytes": 2271523272,
                "downloaded_in_this_task": False,
                "archive_sha256": None,
                "archive_contents": "UNKNOWN_UNTIL_CONTROLLED_ACQUISITION",
                "template_topology": "UNKNOWN_UNTIL_CONTROLLED_ACQUISITION",
                "vertex_count": None,
                "face_count": None,
                "raw_sequence_correspondence": "MUST_BE_VALIDATED_AGAINST_LBN1_BETAS_AND_CANONICAL_FRAME",
                "position_maps_in_archive": "UNKNOWN_UNTIL_ARCHIVE_LISTING",
                "canonical_lbs_volume_in_archive": "UNKNOWN_UNTIL_ARCHIVE_LISTING",
                "mmlp_direct_compatibility": "PENDING_SCHEMA_AND_TOPOLOGY_VALIDATION",
                "license": "Dataset license applies; private internal use only and no third-party sharing.",
                "rank": 1,
            },
            {
                "id": "B_OFFICIAL_ANIMATABLE_GAUSSIANS_TEMPLATE_RECONSTRUCTION",
                "availability": "SOURCE_AND_LBN1_CONFIG_PUBLIC_AT_PINNED_COMMIT",
                "inputs": ["raw lbn1 RGB/masks/calibration/SMPL-X", "neutral SMPL-X model", "canonical LBS volume"],
                "official_paths": [
                    "configs/avatarrex_lbn1/template.yaml",
                    "gen_data/gen_weight_volume.py",
                    "main_template.py",
                    "network/template.py",
                ],
                "official_git_blobs": {
                    "configs/avatarrex_lbn1/template.yaml": "bc94c1a24b1b9abaa15cf01797bf41611a86a113",
                    "gen_data/gen_weight_volume.py": "7b08114c52f48aceca45c7177ad97a70391979e3",
                    "main_template.py": "38367402ae8363874a1b6020ca2bcf78f24d53e6",
                    "network/template.py": "070177275c72534baa886d88872c468c0dcf2169",
                },
                "pretrained_model_required": False,
                "training_contract": {
                    "template_iterations_in_official_main": 150000,
                    "lbn1_config_frame_range": [1033, 1034, 1],
                    "lbn1_config_camera_count": 14,
                    "seed": 31359,
                    "optimizer": "Adam",
                },
                "external_components": [
                    "PointInterpolant compiled from PoissonRecon/AdaptiveSolvers",
                    "custom pose-vocabulary and root-finding operations",
                    "SMPL-X model under its separate license",
                ],
                "gpu_cpu_budget": "NOT_BENCHMARKED_IN_THIS_NO_GENERATION_TASK",
                "output_schema": "data_root/template.ply followed by position-map generation",
                "determinism": "SEEDED_BUT_BITWISE_DETERMINISM_NOT_ESTABLISHED_FOR_CUDA_TRAINING",
                "license": "AnimatableGaussians code license plus AvatarReX dataset license and SMPL-X terms",
                "rank": 2,
            },
            {
                "id": "C_SMPLX_BODY_SURFACE_FALLBACK",
                "availability": "EXISTING_NEUTRAL_SMPLX_MODEL_FORWARD_PASS",
                "generation": "Deterministic body mesh for fixed model/betas/pose; MMLP Scene writes fallback template when missing.",
                "garment_volume_loss_risk": "HIGH_FOR_LBN1_BECAUSE_MMLP_README_EXPLICITLY_RECOMMENDS_A_LOOSE_CLOTHING_TEMPLATE",
                "equivalent_to_loose_clothing_template": False,
                "admitted_use": "CANARY_ONLY_AFTER_EXPLICIT_FALLBACK_APPROVAL",
                "formal_base_avatar_route": False,
                "rank": 3,
            },
            {
                "id": "D_OTHER_OFFICIAL_ASSETS",
                "availability": "NO_ADDITIONAL_TOPOLOGY_KNOWN_LBN1_TEMPLATE_FOUND_AT_PINNED_OFFICIAL_COMMIT",
                "actorshq_template_archive_substitution_allowed": False,
                "rank": 4,
            },
        ],
        "recommended_route": {
            "candidate": "A_OFFICIAL_PREPROCESSED_LOOSE_CLOTHING_TEMPLATE",
            "reason": "It is the official lbn1 asset recommended by MMLP-Human for loose clothing and avoids reconstructing a learned template if its contents validate.",
            "fallback": "B_OFFICIAL_ANIMATABLE_GAUSSIANS_TEMPLATE_RECONSTRUCTION",
            "canary_only_fallback": "C_SMPLX_BODY_SURFACE_FALLBACK",
        },
        "asset_acquisition_plan": [
            "Run only in the next authorized derived-assets task at the same private organizational site.",
            "Download the official Drive file once into a private staging directory; record byte count and SHA256 before extraction.",
            "List the 7z archive before extraction; freeze every path, size, and archive-member CRC/hash available.",
            "Reject path traversal, unexpected RGB/mask duplication, executable payloads, and subject mismatches.",
            "Extract only required preprocessed assets into private staging and hash each file.",
            "Inspect template PLY schema, vertex/face counts, finiteness, manifold diagnostics, bounds, canonical pose, and lbn1 correspondence.",
            "Confirm whether canonical LBS and position maps are present; do not infer from the archive title.",
            "Materialize the accepted mesh at {DATASET_DIR}/gaussian/template.ply and freeze the exact source-to-destination provenance.",
        ],
        "expected_hash_path_contract": {
            "official_archive_expected_path": "PRIVATE_STAGING/avatarrex_lbn1_preprocessed/avatarrex_lbn1.7z",
            "official_archive_expected_bytes_from_drive_page": 2271523272,
            "official_archive_expected_sha256": None,
            "expected_sha_policy": "MUST_BE_OBSERVED_AND_FROZEN_BEFORE_EXTRACTION; NO GUESSED HASH",
            "mmlp_template_path": f"{RAW_ROOT}/gaussian/template.ply",
            "template_sha256": None,
            "topology": None,
            "status": "PENDING_CONTROLLED_ACQUISITION",
        },
        "counts": {"large_downloads": 0, "template_generation": 0, "raw_mutation": 0},
        "license_contract_status": license_contract["status"],
        "paper_final": 0,
    }


def build_lbs_audit() -> dict[str, Any]:
    return {
        "schema_version": "avatarrex.lbs_route_audit.v1",
        "task_id": TASK_ID,
        "status": "ROUTE_SELECTED_ASSET_NOT_GENERATED",
        "mmlp_consumer_contract": {
            "path": f"{RAW_ROOT}/gaussian/lbs_weights_grid.npz",
            "required_keys": ["grid", "bbox_min", "bbox_max", "grid_dims", "grid_resolution"],
            "runtime_query": "torch.nn.functional.grid_sample with xyz-to-zyx permutation, border padding, align_corners=True",
            "query_consumer": "GaussianModel.get_weights",
            "checkpoint_behavior": "GaussianModel.capture stores materialized per-Gaussian _weights; restore consumes stored _weights.",
        },
        "candidates": [
            {
                "id": "OFFICIAL_CANONICAL_LBS_VOLUME",
                "status": "PENDING_OFFICIAL_ARCHIVE_CONTENT_INSPECTION",
                "expected_official_schema": [
                    "diff_weight_volume",
                    "ori_weight_volume",
                    "sdf_volume",
                    "volume_bounds",
                    "smpl_bounds",
                    "center",
                ],
                "direct_mmlp_compatibility": False,
                "reason": "The official AnimatableGaussians schema differs from the MMLP-Human NPZ consumer schema.",
            },
            {
                "id": "ANIMATABLE_GAUSSIANS_LBS_ASSET",
                "status": "SAME_SCHEMA_RISK_AS_OFFICIAL_CANONICAL_VOLUME",
                "direct_mmlp_compatibility": False,
            },
            {
                "id": "SURFACE_ATTACHED_LBS",
                "status": "FEASIBLE_BUT_REQUIRES_A_NEW_RUNTIME_AND_CHECKPOINT_CONTRACT",
                "topology_leakage_risk": "Weights bind to a specific template or Gaussian point set unless recomputed after topology changes.",
                "direct_mmlp_compatibility": False,
            },
            {
                "id": "DETERMINISTIC_CLOSEST_SURFACE_FALLBACK",
                "status": "CANARY_FALLBACK_ONLY",
                "method": "Closest SMPL-X triangle plus barycentric interpolation of official SMPL-X vertex LBS weights.",
                "direct_mmlp_compatibility": False,
            },
            {
                "id": "MMLP_POINTINTERPOLANT_VOLUME",
                "status": "RECOMMENDED",
                "generator": "script/gen_weight_volume.py",
                "inputs": ["smpl_params.npz betas[0]", "neutral SMPL-X model", "PointInterpolant"],
                "output": f"{RAW_ROOT}/gaussian/lbs_weights_grid.npz",
                "schema_matches_current_consumer": True,
                "topology_independence": "The continuous grid is generated from fixed SMPL-X topology and queried at the actual initial Gaussian positions.",
                "determinism_gate": "Generate twice in isolated staging and require equal schema, arrays, bounds, and canonical content hash.",
            },
        ],
        "recommended_route": {
            "candidate": "MMLP_POINTINTERPOLANT_VOLUME",
            "reason": "It produces the exact current consumer schema, avoids binding weights to one loose-clothing mesh topology, and queries the actual Gaussian initialization positions.",
            "official_volume_role": "Cross-check only unless an explicit deterministic schema adapter is separately reviewed.",
        },
        "gaussian_lifecycle": {
            "initialization_source": "Poisson-disk sampling of the accepted canonical template mesh",
            "clone_split_densification_in_current_train_path": "ABSENT; the audited train.py and GaussianModel have a fixed initialized point set.",
            "weight_materialization": "First get_weights query interpolates the grid at _xyz and caches the result.",
            "checkpoint_consumption": "capture serializes _weights; restore loads _weights directly.",
            "future_topology_change_rule": "Any clone/split/densification or point replacement must explicitly rebind weights and version the checkpoint schema.",
        },
        "generation_gate_for_next_task": [
            "Pin the exact neutral SMPL-X hash and PointInterpolant source/build hash.",
            "Generate only in private staging, never in raw root until validation passes.",
            "Require finite nonnegative weights, row sums near one, expected joint count, valid bbox, and exact required NPZ keys.",
            "Query template vertices and representative interior/exterior points; reject discontinuities or NaN/Inf.",
            "Run isolated duplicate generation or a stronger deterministic equivalence test before promotion.",
            "Promote once to gaussian/lbs_weights_grid.npz with append-only provenance and no raw RGB/mask writes.",
        ],
        "counts": {"lbs_generation": 0, "template_generation": 0, "training": 0, "raw_mutation": 0},
        "paper_final": 0,
    }


def build_split_audit(split: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    audit = runtime["strict_split_audit"]
    return {
        "schema_version": "avatarrex.strict_split_audit.v1",
        "task_id": TASK_ID,
        "status": audit["status"],
        "source_contract": "paper_protocol/datasets/avatarrex_camera_pose_split.json",
        "source_contract_sha256": None,
        "split_content_sha256": split["split_content_sha256"],
        "regenerated_split_sha256": audit["regenerated_split_sha256"],
        "deterministic_regeneration": {
            "camera": audit["checks"]["camera_regeneration_deterministic"],
            "pose": audit["checks"]["pose_regeneration_deterministic"],
            "matches_frozen_camera": audit["checks"]["camera_matches_frozen"],
            "matches_frozen_pose": audit["checks"]["pose_matches_frozen"],
            "matches_frozen_availability": audit["checks"]["availability_matches_frozen"],
        },
        "camera_split": {
            "train_count": split["camera_split"]["train_count"],
            "heldout_count": split["camera_split"]["heldout_count"],
            "train_camera_ids": split["camera_split"]["train_camera_ids"],
            "heldout_camera_ids": split["camera_split"]["heldout_camera_ids"],
            "overlap": split["camera_split"]["overlap"],
            "split_sha256": split["camera_split"]["split_sha256"],
            "azimuth_order_camera_ids": split["camera_split"]["azimuth_order_camera_ids"],
            "coverage": audit["camera_coverage"],
        },
        "pose_split": {
            "train_count": split["pose_split"]["train_count"],
            "heldout_count": split["pose_split"]["heldout_count"],
            "buffer_excluded_count": split["pose_split"]["buffer_excluded_count"],
            "minimum_heldout_temporal_distance": split["pose_split"]["heldout_pair_min_temporal_distance"],
            "train_heldout_overlap": split["pose_split"]["train_heldout_overlap"],
            "train_buffer_overlap": split["pose_split"]["train_buffer_overlap"],
            "temporal_leakage": split["pose_split"]["temporal_leakage"],
            "split_sha256": split["pose_split"]["split_sha256"],
        },
        "availability": {
            "per_camera_valid_counts": audit["per_camera_valid_counts"],
            "per_pose_valid_camera_counts": audit["per_pose_valid_camera_counts"],
            "pair_counts": audit["available_pair_counts"],
            "all_records_available": audit["checks"]["all_records_available"],
        },
        "selection_governance": {
            "training_results_used_to_change_heldout_cameras": False,
            "future_result_driven_split_changes": "FORBIDDEN",
            "buffer_frames_forbidden": True,
        },
        "paper_final": 0,
    }


def build_preflight_contract(
    runtime: dict[str, Any], template: dict[str, Any], lbs: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": "avatarrex.base_avatar_preflight_contract.v1",
        "task_id": TASK_ID,
        "status": "READY_TO_PREPARE_PRIVATE_DERIVED_ASSETS_WITHOUT_TRAINING",
        "classification": CLASSIFICATION,
        "environment": {
            "python": runtime["environment"]["python"],
            "platform": runtime["environment"]["platform"],
            "torch": runtime["environment"]["torch"],
            "smplx": runtime["environment"]["smplx"],
            "neutral_model_sha256": runtime["smpl_x_neutral_forward"]["model_sha256"],
            "required_future_runtime": ["CUDA-capable PyTorch", "SMPL-X", "Open3D", "PyTorch3D", "gsplat", "PointInterpolant"],
        },
        "frozen_inputs": {
            "raw_root": RAW_ROOT,
            "archive_path": ARCHIVE_PATH,
            "archive_bytes": 12569755256,
            "archive_sha256": ARCHIVE_SHA256,
            "raw_file_count": 60834,
            "raw_apparent_bytes": 19135049684,
            "raw_full_content_fingerprint": RAW_FINGERPRINT,
            "calibration_sha256": CALIBRATION_SHA256,
            "smpl_params_sha256": SMPL_SHA256,
            "adapter_sha256": None,
            "split_sha256": None,
        },
        "gates": {
            "license": "PASS_PRIVATE_INTERNAL_NONCOMMERCIAL_SINGLE_SITE_ONLY",
            "zero_copy_adapter": runtime["loader_smoke"]["status"],
            "camera_calibration": runtime["camera_audit"]["status"],
            "smpl_x_schema": runtime["smpl_x_schema_audit"]["status"],
            "neutral_smpl_x_forward": runtime["smpl_x_neutral_forward"]["status"],
            "strict_split": runtime["strict_split_audit"]["status"],
            "template": template["status"],
            "lbs": lbs["status"],
            "image_mask_decode": runtime["loader_smoke"]["status"],
            "model_config_parse": runtime["static_interface_audit"]["status"],
            "model_construction": "PENDING_TEMPLATE_AND_LBS_ASSETS",
            "zero_step_render": "PENDING_TEMPLATE_AND_LBS_ASSETS",
            "checkpoint_schema": "STATIC_CAPTURE_RESTORE_PARSE_ONLY_PENDING_RUNTIME_INSTANCE",
            "training_contract_recovery": "PENDING_DERIVED_ASSET_AND_RUNTIME_CLOSURE",
            "mutation_audit": "PASS",
        },
        "required_future_preflight_sequence": [
            "Verify environment and all frozen hashes before access.",
            "Verify private-license guard and output-root access control.",
            "Acquire and validate the official loose-clothing template under the frozen acquisition contract.",
            "Generate and validate the MMLP PointInterpolant LBS grid in private staging.",
            "Promote template and LBS assets once with immutable provenance.",
            "Re-run zero-copy loader, calibration, SMPL-X, and strict-split audits.",
            "Construct the model with writes redirected to an isolated derived-output root.",
            "Run one zero-step render with no optimizer or backward and validate finite RGB/alpha and camera parity.",
            "Freeze actual checkpoint capture/restore schema and bitwise zero-step restoration.",
            "Recover and approve the short-canary step budget, checkpoint cadence, and optimization signal thresholds.",
            "Audit raw/frozen mutations and require every forbidden count to remain zero before canary admission.",
        ],
        "admission": {
            "derived_asset_preparation": True,
            "short_canary": False,
            "formal_training": False,
            "short_canary_blocker": "TEMPLATE_AND_LBS_ASSETS_REQUIRED",
            "formal_training_from_this_task": "FORBIDDEN",
        },
        "execution_counts": runtime["mutation_audit"],
        "paper_final": 0,
    }


def evenly_spaced(values: list[Any], count: int) -> list[Any]:
    if count == 1:
        return [values[0]]
    indices = [round(index * (len(values) - 1) / (count - 1)) for index in range(count)]
    return [values[index] for index in indices]


def build_canary_draft(split: dict[str, Any]) -> dict[str, Any]:
    train_frames = split["pose_split"]["train_frame_ids"]
    heldout_frames = split["pose_split"]["heldout_frame_ids"]
    train_camera_set = set(split["camera_split"]["train_camera_ids"])
    azimuth_train = [
        camera
        for camera in split["camera_split"]["azimuth_order_camera_ids"]
        if camera in train_camera_set
    ]
    canary_frames = evenly_spaced(train_frames, 16)
    canary_cameras = evenly_spaced(azimuth_train, 4)
    records = [
        {"identity_id": "avatarrex_lbn1", "canonical_camera_id": camera, "frame_id": frame}
        for frame in canary_frames
        for camera in canary_cameras
    ]
    record_manifest = {
        "selection": "16 evenly spaced ranks over frozen sorted train_frame_ids x 4 evenly spaced ranks over frozen train-camera azimuth order",
        "pose_frame_ids": canary_frames,
        "camera_ids": canary_cameras,
        "records": records,
    }
    train_eval_frame = canary_frames[len(canary_frames) // 2]
    heldout_eval_frame = heldout_frames[len(heldout_frames) // 2]
    train_eval_camera = canary_cameras[0]
    heldout_eval_camera = split["camera_split"]["heldout_camera_ids"][0]
    return {
        "schema_version": "avatarrex.canary_training_contract_draft.v1",
        "task_id": TASK_ID,
        "status": "PENDING_DERIVED_ASSET_AND_RUNTIME_CLOSURE",
        "execution_authorized": False,
        "prerequisite": "All base-avatar preflight gates including template, LBS, model construction, zero-step render, checkpoint schema, and mutation audit must pass.",
        "deterministic_training_subset": {
            **record_manifest,
            "record_count": len(records),
            "unique_record_count": len({(row["canonical_camera_id"], row["frame_id"]) for row in records}),
            "manifest_sha256": canonical_sha256(record_manifest),
            "strict_split_leakage": 0,
            "buffer_frame_count": 0,
        },
        "batch_size": 1,
        "optimizer_source": {
            "code": "scene.gaussian_model.GaussianModel.training_setup",
            "learning_rates": "config/avrex_lbn1.yaml",
            "created_in_this_task": False,
            "final_canary_parameter_groups": "PENDING_DERIVED_ASSET_AND_RUNTIME_CLOSURE",
        },
        "step_budget": {
            "value": None,
            "status": "PENDING_DERIVED_ASSET_AND_RUNTIME_CLOSURE",
            "reason": "No unique short-canary budget is established before zero-step runtime and checkpoint closure; no default is substituted.",
        },
        "checkpoint_cadence": {
            "value": None,
            "status": "PENDING_DERIVED_ASSET_AND_RUNTIME_CLOSURE",
            "reason": "Must be frozen together with the actual checkpoint capture/restore schema.",
        },
        "fixed_four_quadrant_evaluation": [
            {"quadrant": "TRAIN_POSE_TRAIN_VIEW", "camera_id": train_eval_camera, "frame_id": train_eval_frame},
            {"quadrant": "TRAIN_POSE_HELDOUT_VIEW", "camera_id": heldout_eval_camera, "frame_id": train_eval_frame},
            {"quadrant": "HELDOUT_POSE_TRAIN_VIEW", "camera_id": train_eval_camera, "frame_id": heldout_eval_frame},
            {"quadrant": "HELDOUT_POSE_HELDOUT_VIEW", "camera_id": heldout_eval_camera, "frame_id": heldout_eval_frame},
        ],
        "optimization_signal_thresholds": {
            "values": None,
            "status": "PENDING_DERIVED_ASSET_AND_RUNTIME_CLOSURE",
            "required_fields": [
                "finite loss and gradient fraction",
                "nonzero admitted trainable gradient count",
                "base/frozen parameter fingerprint invariance",
                "Gaussian finite-state fraction",
                "fixed four-quadrant zero-step and final metrics",
            ],
        },
        "rerun_policy": {
            "result_driven_reruns": "FORBIDDEN",
            "infrastructure_failure_rerun": "Allowed only before scientific signals are consumed, with append-only failure provenance.",
            "split_or_subset_changes_after_results": "FORBIDDEN",
        },
        "claim_boundary": {
            "formal_claim": False,
            "paper_final": 0,
            "canary_success_meaning": "Runtime and optimization-signal feasibility only; not paper evidence or formal AvatarReX performance.",
        },
        "execution_counts": {"training": 0, "optimizer_created": 0, "backward": 0, "checkpoint_writes": 0},
    }


def build_final_summary(runtime: dict[str, Any], artifact_parent_head: str) -> dict[str, Any]:
    return {
        "schema_version": "avatarrex.base_avatar_preflight_final_summary.v1",
        "task_id": TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "branch": BRANCH,
        "artifact_build_parent_head": artifact_parent_head,
        "classification": CLASSIFICATION,
        "classification_scope": "License closure, zero-copy runtime adapter, strict split, and routes for private derived-asset preparation only.",
        "license_status": "PRIVATE_INTERNAL_NONCOMMERCIAL_SINGLE_SITE_DERIVED_ASSET_PREPARATION_ALLOWED",
        "runtime_status": runtime["status"],
        "runtime_blocker": runtime["blocker"],
        "template_status": "ROUTE_SELECTED_ASSET_NOT_ACQUIRED",
        "lbs_status": "ROUTE_SELECTED_ASSET_NOT_GENERATED",
        "missing_assets": [
            f"{RAW_ROOT}/gaussian/template.ply",
            f"{RAW_ROOT}/gaussian/lbs_weights_grid.npz",
            f"{RAW_ROOT}/gaussian/init_body_points.ply",
        ],
        "model_render_claim": "No complete model/render PASS is claimed; only DATASET_AND_PARAMETER_ADAPTER_SMOKE_PASS.",
        "forbidden_counts": runtime["mutation_audit"],
        "reports": [
            "docs/DATASET/AVATARREX_LICENSE_CLOSURE_20260724.md",
            "docs/SECOND_IDENTITY/AVATARREX_BASE_AVATAR_PREFLIGHT_20260724.md",
            "docs/SECOND_IDENTITY/AVATARREX_TEMPLATE_AND_LBS_ROUTE_20260724.md",
            "docs/SECOND_IDENTITY/AVATARREX_STRICT_SPLIT_AUDIT_20260724.md",
        ],
        "contracts": [
            "paper_protocol/datasets/avatarrex_license_and_citation_contract.json",
            "paper_protocol/datasets/avatarrex_zero_copy_runtime_adapter.json",
            "paper_protocol/datasets/avatarrex_calibration_runtime_contract.json",
            "paper_protocol/datasets/avatarrex_smplx_runtime_contract.json",
            "paper_protocol/datasets/avatarrex_template_route_audit.json",
            "paper_protocol/datasets/avatarrex_lbs_route_audit.json",
            "paper_protocol/datasets/avatarrex_base_avatar_preflight_contract.json",
            "paper_protocol/datasets/avatarrex_canary_training_contract_draft.json",
            "paper_protocol/datasets/avatarrex_base_avatar_runtime_audit.json",
        ],
        "paper_final": 0,
        "next_task": NEXT_TASK,
        "next_task_automatic_execution": False,
    }


def build_handoff(summary: dict[str, Any], artifacts: dict[str, str]) -> dict[str, Any]:
    return {
        "schema_version": "avatarrex.base_avatar_preflight_handoff.v1",
        "task_id": TASK_ID,
        "branch": BRANCH,
        "source_head": SOURCE_HEAD,
        "classification": CLASSIFICATION,
        "summary": "paper_protocol/datasets/avatarrex_base_avatar_preflight_final_summary.json",
        "artifact_sha256": artifacts,
        "frozen_inputs": {
            "raw_root": RAW_ROOT,
            "archive_path": ARCHIVE_PATH,
            "archive_sha256": ARCHIVE_SHA256,
            "raw_full_content_fingerprint": RAW_FINGERPRINT,
            "calibration_sha256": CALIBRATION_SHA256,
            "smpl_params_sha256": SMPL_SHA256,
        },
        "private_output_policy": "All raw data, derived assets, templates, LBS volumes, position maps, and data-derived checkpoints remain private at the same organizational site.",
        "runtime_status": "DATASET_AND_PARAMETER_ADAPTER_SMOKE_PASS",
        "blocker": "TEMPLATE_AND_LBS_ASSETS_REQUIRED",
        "next_task": NEXT_TASK,
        "next_task_automatic_execution": False,
        "next_task_entry_gates": [
            "Reconfirm private single-site noncommercial license guard.",
            "Acquire official lbn1 preprocessed archive without raw RGB/mask duplication.",
            "Freeze archive/member hashes before extraction and validate template topology/schema.",
            "Generate and validate MMLP PointInterpolant LBS grid in isolated private staging.",
            "Do not train; stop after derived-asset validation and handoff.",
        ],
        "paper_final": 0,
    }


def build_reports(
    license_contract: dict[str, Any],
    calibration: dict[str, Any],
    smpl: dict[str, Any],
    template: dict[str, Any],
    lbs: dict[str, Any],
    split_audit: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, str]:
    license_report = f"""# AvatarReX License Closure

Task: `{TASK_ID}`

## Decision

The primary license evidence is `AVATARREX_DATASET.md` at official AnimatableGaussians commit `2b0f6e3b4c5af823414eb6d5f0b2e1a59954d114` (SHA256 `14913f3414cf178af0fd6e00fc74145294c5a6c7bf748bf603d0ae7afe96ac60`). The repository `LICENSE` is code-license evidence only and is not substituted for the dataset agreement.

Private, internal, single-site preparation of lbn1 templates, LBS volumes, and checkpoints is admitted for this non-commercial research project. Raw data, images, derived data, derived assets, and data-derived checkpoints must not be copied, published, distributed, sold, or made available to a third party without prior written Tsinghua University permission. Commercial and pornographic uses are prohibited. Commercial licensing contact: `liuyebin@mail.tsinghua.edu.cn`.

This is a project governance interpretation of the official text, not legal advice. Public release remains blocked until written permission is obtained.

## Sources

- Official repository: <https://github.com/lizhe00/AnimatableGaussians>
- Dataset agreement: <https://github.com/lizhe00/AnimatableGaussians/blob/2b0f6e3b4c5af823414eb6d5f0b2e1a59954d114/AVATARREX_DATASET.md>
- Code license: <https://github.com/lizhe00/AnimatableGaussians/blob/2b0f6e3b4c5af823414eb6d5f0b2e1a59954d114/LICENSE>
- Official preprocessed index: <https://github.com/lizhe00/AnimatableGaussians/blob/2b0f6e3b4c5af823414eb6d5f0b2e1a59954d114/PREPROCESSED_DATASET.md>
- Animatable Gaussians project: <https://animatable-gaussians.github.io/>

The historical AvatarReX project URL currently resolves to unrelated gambling content and was excluded from license evidence. The official pinned repository retains the authoritative dataset agreement and citation block.

## Citation

Any publication using lbn1 must cite both `AvatarReX: Real-time Expressive Full-body Avatars` (ACM TOG 42(4), 2023, DOI `10.1145/3592101`) and `Animatable Gaussians: Learning Pose-dependent Gaussian Maps for High-fidelity Human Avatar Modeling` (CVPR 2024).

Contract: `paper_protocol/datasets/avatarrex_license_and_citation_contract.json`.
"""
    preflight_report = f"""# AvatarReX Base-Avatar Preflight

Task: `{TASK_ID}`

## Result

Runtime status: `{runtime['status']}`. The 16-camera calibration audit, 3-camera x 3-frame loader smoke, SMPL-X schema, neutral SMPL-X forward, strict split regeneration, config parse, and model/renderer interface parse passed. Raw tree metadata was unchanged before and after the smoke.

This task does not claim a complete model or render pass. `{RAW_ROOT}/gaussian/template.ply` and `{RAW_ROOT}/gaussian/lbs_weights_grid.npz` are absent, so the explicit blocker is `TEMPLATE_AND_LBS_ASSETS_REQUIRED`. Calling `Scene` was intentionally forbidden because the missing-template fallback writes derived files.

## Calibration And Pose

- Raw/runtime extrinsics: `x_camera = R @ x_world + T`; camera center `C=-R^T T`.
- `imgSize=[1500,2048]` means width then height; decoded arrays are `2048 x 1500`.
- All distortion coefficients are zero, so the loader's in-memory undistortion branch is a no-op for this capture.
- SMPL-X fields exactly match the eight expected float32 arrays. Gender remains `UNKNOWN`.
- Neutral compatibility passed for frames `0/950/1900` with `10475` vertices and `20908` faces; this does not establish gender metadata.
- `scene.dataset.get_scene_scale` must be corrected or overridden to use camera centers before canary admission.

## Runtime Boundary

The current MMLP path accepts `AVRexDataset` items and statically exposes the expected gsplat `w2c/K/width/height` interface. Model construction, zero-step render, checkpoint runtime restoration, optimizer creation, backward, and training remain pending until private derived assets pass their own task.

Final classification: `{CLASSIFICATION}`. This admits the next private derived-assets task only; it does not admit canary or formal training.
"""
    route_report = f"""# AvatarReX Template And LBS Route

Task: `{TASK_ID}`

## Template

The primary route is the official lbn1 preprocessed archive indexed by `PREPROCESSED_DATASET.md` (Drive file `1RDM3v5P4XF6Sp88EusDvokw-yHg6Je0C`, viewer metadata title `avatarrex_lbn1.7z`, `2271523272` bytes). No download occurred. Archive SHA, members, template topology, vertex/face counts, position maps, and canonical LBS presence remain unknown and must be frozen during controlled acquisition.

If that asset fails validation, the secondary route is the official AnimatableGaussians reconstruction pipeline: PointInterpolant/canonical LBS preparation, the lbn1 template config, `main_template.py`, and position-map generation. It is a 150000-iteration learned reconstruction and was not run. The deterministic SMPL-X body surface is canary-only because MMLP explicitly recommends a loose-clothing template for `avatarrex_lbn1`; it is not equivalent evidence.

Expected accepted MMLP path: `{RAW_ROOT}/gaussian/template.ply`.

## LBS

The recommended route is MMLP-Human's own `script/gen_weight_volume.py` with the pinned neutral SMPL-X model and pinned PointInterpolant build. It produces the exact runtime schema `grid/bbox_min/bbox_max/grid_dims/grid_resolution` at `{RAW_ROOT}/gaussian/lbs_weights_grid.npz` and queries weights at the actual initialized Gaussian positions.

The official AnimatableGaussians `cano_weight_volume.npz` schema is not directly interchangeable. Surface-attached or closest-surface weights require a new runtime/checkpoint contract and are not the selected primary route. Current MMLP training has a fixed initialized Gaussian point set; no clone/split/densification path was found. Checkpoints serialize materialized per-Gaussian weights, so any future topology change requires explicit rebinding and schema versioning.

No template or LBS asset was generated in this task.
"""
    split_report = f"""# AvatarReX Strict Split Audit

Task: `{TASK_ID}`

## Camera Split

The frozen camera split regenerated exactly. Train cameras: `{', '.join(split_audit['camera_split']['train_camera_ids'])}`. Held-out cameras: `{', '.join(split_audit['camera_split']['heldout_camera_ids'])}`. Counts are `12/4`; overlap is zero. The camera split SHA256 is `{split_audit['camera_split']['split_sha256']}`.

Held-out cameras remain the fixed azimuth ranks `0/4/8/12`: `camera_01, camera_02, camera_09, camera_11`. No training result was used to choose or modify them.

## Pose Split

The frozen pose split regenerated exactly: `856` train, `95` held-out, and `950` buffer-excluded frames. Train/held-out and train/buffer overlaps are zero; temporal leakage is zero; minimum held-out temporal distance is `11`. The pose split SHA256 is `{split_audit['pose_split']['split_sha256']}`.

Every camera has `1901` valid RGB/mask pairs and every admitted pose has `16` valid cameras. Available strict quadrant counts are `10272` train-view/train-pose, `3424` held-out-view/train-pose, `1140` train-view/held-out-pose, and `380` held-out-view/held-out-pose.

Combined frozen split content SHA256: `{split_audit['split_content_sha256']}`. Future result-driven split changes are forbidden.
"""
    return {
        "docs/DATASET/AVATARREX_LICENSE_CLOSURE_20260724.md": license_report,
        "docs/SECOND_IDENTITY/AVATARREX_BASE_AVATAR_PREFLIGHT_20260724.md": preflight_report,
        "docs/SECOND_IDENTITY/AVATARREX_TEMPLATE_AND_LBS_ROUTE_20260724.md": route_report,
        "docs/SECOND_IDENTITY/AVATARREX_STRICT_SPLIT_AUDIT_20260724.md": split_report,
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    dataset_dir = repo_root / "paper_protocol" / "datasets"
    runtime_path = dataset_dir / "avatarrex_base_avatar_runtime_audit.json"
    old_adapter_path = dataset_dir / "avatarrex_zero_copy_adapter_contract.json"
    split_path = dataset_dir / "avatarrex_camera_pose_split.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    old_adapter = json.loads(old_adapter_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    artifact_parent_head = current_head(repo_root)

    license_contract = build_license_contract()
    adapter = build_runtime_adapter(old_adapter, runtime)
    calibration = build_calibration_contract(runtime)
    smpl = build_smpl_contract(runtime)
    template = build_template_audit(license_contract)
    lbs = build_lbs_audit()
    split_audit = build_split_audit(split, runtime)
    preflight = build_preflight_contract(runtime, template, lbs)
    canary = build_canary_draft(split)

    adapter["loader_smoke"]["runtime_audit_sha256"] = file_sha256(runtime_path)
    split_audit["source_contract_sha256"] = file_sha256(split_path)
    preflight["frozen_inputs"]["adapter_sha256"] = file_sha256(old_adapter_path)
    preflight["frozen_inputs"]["split_sha256"] = file_sha256(split_path)

    contracts = {
        "avatarrex_license_and_citation_contract.json": license_contract,
        "avatarrex_zero_copy_runtime_adapter.json": adapter,
        "avatarrex_calibration_runtime_contract.json": calibration,
        "avatarrex_smplx_runtime_contract.json": smpl,
        "avatarrex_template_route_audit.json": template,
        "avatarrex_lbs_route_audit.json": lbs,
        "avatarrex_base_avatar_preflight_contract.json": preflight,
        "avatarrex_canary_training_contract_draft.json": canary,
    }
    for name, payload in contracts.items():
        write_json(dataset_dir / name, payload)

    reports = build_reports(license_contract, calibration, smpl, template, lbs, split_audit, runtime)
    for relative, text in reports.items():
        write_text(repo_root / relative, text)

    artifact_hashes = {
        f"paper_protocol/datasets/{name}": file_sha256(dataset_dir / name) for name in contracts
    }
    artifact_hashes["paper_protocol/datasets/avatarrex_base_avatar_runtime_audit.json"] = file_sha256(runtime_path)
    for relative in reports:
        artifact_hashes[relative] = file_sha256(repo_root / relative)

    summary = build_final_summary(runtime, artifact_parent_head)
    summary["artifact_sha256"] = artifact_hashes
    summary_path = dataset_dir / "avatarrex_base_avatar_preflight_final_summary.json"
    write_json(summary_path, summary)
    artifact_hashes["paper_protocol/datasets/avatarrex_base_avatar_preflight_final_summary.json"] = file_sha256(
        summary_path
    )

    handoff = build_handoff(summary, artifact_hashes)
    handoff_path = repo_root / "project_control_handoff" / "avatarrex_base_avatar_preflight_handoff.json"
    write_json(handoff_path, handoff)
    print(
        json.dumps(
            {
                "classification": CLASSIFICATION,
                "contracts": len(contracts) + 2,
                "reports": len(reports),
                "handoff": str(handoff_path.relative_to(repo_root)),
                "next_task": NEXT_TASK,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
