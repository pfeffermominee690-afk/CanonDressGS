#!/usr/bin/env python3
"""Build the static Subject00 three-garment canary data protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "AAAI27-SUBJECT00-THREE-GARMENT-CANARY-PROTOCOL-001"
SOURCE_HEAD = "9e46e85cd02527b52e2f5bf0af5571de4c8988d2"
FORMAL_HEAD = "4993f5c865ec19895f35811fa399fc4a1834c6a7"
MEDIUM_HEAD = "2d0913eb1b163a79e81c1804c216f91b0a737b44"
BRANCH = "research/subject00-three-garment-canary-protocol-20260724"
CLASSIFICATION = "SUBJECT00_THREE_GARMENT_PROTOCOL_READY_PENDING_FORMAL_BASE"
NEXT_TASK = "RUN_SUBJECT00_FORMAL_BASE_THEN_FREEZE_THREE_GARMENT_CONDITIONS"
GARMENT_IDS = ["O01", "O03", "O04"]
SLOTS = [
    "front",
    "front_left_three_quarter",
    "front_right_three_quarter",
    "left",
    "right",
    "back_left_three_quarter",
    "back_right_three_quarter",
    "back",
]
REFERENCE_SLOTS = ["front", "left", "back", "right"]
STRICT_TRAIN_CAMERA_IDS = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23]
STRICT_HELDOUT_CAMERA_IDS = [0, 4, 8, 12, 16, 20]
PENDING = "PENDING_FORMAL_BASE"
BACKEND_PENDING = "GENERATION_BACKEND_DEPENDENCY_PENDING"

SOURCE_OBJECTS = {
    "shared_five_garment_wardrobe": {
        "commit": SOURCE_HEAD,
        "path": "paper_protocol/datasets/shared_five_garment_wardrobe.json",
        "git_blob": "c826eef0fd28065bc44008bcbac740a69a9e2c84",
        "content_sha256": "6da35b17759e95cdb515e7a1df95a170a0d7040067a0f1fa892fce76e73d21c0",
    },
    "shared_condition_pose_view_protocol": {
        "commit": SOURCE_HEAD,
        "path": "paper_protocol/datasets/shared_condition_pose_view_protocol.json",
        "git_blob": "5cee5d0b5b08b71089ef62deb64521f5c471b249",
        "content_sha256": "78a61b520958e2fd67a54d0f45298d5a718905e7a8464472515a24f86320569e",
    },
    "formal_pose_split": {
        "commit": FORMAL_HEAD,
        "path": "paper_protocol/second_identity/subject00_novel_pose_split_v2.json",
        "git_blob": "e2a5472a887bf1f5b31be808ea68ef36db10df5e",
        "scientific_sha256": "c12e5eec87c737d3740b6e6199d657c859df7ce1ec9d371ea8d4a987ef62bf06",
    },
    "formal_camera_split": {
        "commit": FORMAL_HEAD,
        "path": "paper_protocol/second_identity/subject00_novel_view_split_v2.json",
        "git_blob": "2fd37f4fa77385b79c8c1d7e157e75a2cb0c374d",
        "scientific_sha256": "8827cdc05a08cb7068e1f5e89ab7ba5384bf83b89686e8c9f80024b78d49cf44",
    },
    "formal_availability": {
        "commit": FORMAL_HEAD,
        "path": "paper_protocol/second_identity/subject00_formal_evaluation_manifests.json",
        "git_blob": "f9a40af69fb13723d8c1ff0c67ccec355bd66a91",
        "entries_sha256": "cd00ad0a1549bc99d956e26eea300945adfaa10a20a438e931e640977475564e",
        "raw_file_sha256": "da9cbce12b6d0fa9a1fddced662eefa9d2f4f331011fbbf3a9944fc878011c7a",
    },
    "formal_training_contract": {
        "commit": FORMAL_HEAD,
        "path": "paper_protocol/second_identity/subject00_formal_training_contract.json",
        "git_blob": "20773064786f434c08879189381cd79f930677ee",
        "final_step": 101245,
    },
    "medium_final_summary": {
        "commit": MEDIUM_HEAD,
        "path": "paper_protocol/second_identity/subject00_medium_pilot_final_summary.json",
        "git_blob": "5a1eb194b67c0da829fc1932eae68e1ebb4794ae",
        "use": "PREVIEW_ONLY",
        "formal_promotion_allowed": False,
    },
}


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def bytes_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_text(relative: str, value: str) -> None:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.strip() + "\n", encoding="utf-8", newline="\n")


def source_payloads() -> tuple[dict[str, Any], dict[str, Any]]:
    datasets = ROOT / "paper_protocol/datasets"
    wardrobe_path = datasets / "shared_five_garment_wardrobe.json"
    slots_path = datasets / "shared_condition_pose_view_protocol.json"
    wardrobe = json.loads(wardrobe_path.read_text(encoding="utf-8"))
    slots = json.loads(slots_path.read_text(encoding="utf-8"))
    if wardrobe["content_sha256"] != SOURCE_OBJECTS["shared_five_garment_wardrobe"]["content_sha256"]:
        raise RuntimeError("shared wardrobe content hash changed")
    if slots["content_sha256"] != SOURCE_OBJECTS["shared_condition_pose_view_protocol"]["content_sha256"]:
        raise RuntimeError("shared semantic slot content hash changed")
    if wardrobe["garment_ids"] != ["O01", "O02", "O03", "O04", "O08"]:
        raise RuntimeError("shared wardrobe IDs changed")
    return wardrobe, slots


def selected_garments(wardrobe: dict[str, Any]) -> list[dict[str, Any]]:
    index = {row["garment_id"]: row for row in wardrobe["garments"]}
    return [index[garment_id] for garment_id in GARMENT_IDS]


def pose_camera_mapping(slots: dict[str, Any]) -> dict[str, Any]:
    shared_by_name = {row["semantic_pose_slot"]: row for row in slots["slots"]}
    mappings = []
    for slot in SLOTS:
        shared = shared_by_name[slot]
        mappings.append(
            {
                "semantic_pose_slot": slot,
                "target_body_yaw_degrees": shared["target_body_yaw_degrees"],
                "yaw_tolerance_degrees": shared["yaw_tolerance_degrees"],
                "pose_candidate_set": {
                    "source": "formal_pose_split.train_frame_ids",
                    "source_commit": FORMAL_HEAD,
                    "source_split_sha256": SOURCE_OBJECTS["formal_pose_split"]["scientific_sha256"],
                    "candidate_count_before_availability": 1130,
                    "heldout_pose_candidates_allowed": False,
                    "buffer_pose_candidates_allowed": False,
                    "selected_pose_id": PENDING,
                },
                "camera_candidate_set": {
                    "source": "formal_camera_split.train_camera_ids",
                    "source_commit": FORMAL_HEAD,
                    "source_split_sha256": SOURCE_OBJECTS["formal_camera_split"]["scientific_sha256"],
                    "candidate_camera_ids": STRICT_TRAIN_CAMERA_IDS,
                    "heldout_camera_ids_forbidden": STRICT_HELDOUT_CAMERA_IDS,
                    "selected_camera_id": PENDING,
                },
                "selection_status": PENDING,
            }
        )
    return {
        "mapping_status": PENDING,
        "slot_count": 8,
        "slot_order": SLOTS,
        "reference_slots": REFERENCE_SLOTS,
        "strict_split_boundary": {
            "condition_and_teacher_target_pose_semantics": "STRICT_TRAIN_ONLY",
            "condition_and_teacher_target_camera_semantics": "STRICT_TRAIN_ONLY",
            "strict_heldout_pose_or_camera_as_teacher_target": False,
            "future_strict_evaluation_image_reuse": False,
            "endpoint_parity_on_training_endpoints": "TRAIN_FIT_DIAGNOSTIC_ONLY",
        },
        "global_pose_selection_rule": {
            "candidate_source": "all strict-train frame IDs",
            "hard_filters_in_order": [
                "availability entry exists for every selected slot camera",
                "valid source frame",
                "full body is complete",
                "both feet are complete and visible",
                "both hands are visible",
                "not a temporal buffer frame",
                "not a held-out pose",
            ],
            "score_order": [
                "body pose semantic distance to the frozen neutral target",
                "maximum per-slot hand visibility penalty",
                "maximum per-slot foot visibility penalty",
                "maximum per-slot crop-margin penalty",
                "frame_id",
            ],
            "tie_break": "lexicographic score order",
            "selected_once_and_reused_for_all_garments": True,
            "future_generation_quality_used": False,
        },
        "per_slot_camera_selection_rule": {
            "hard_filters": [
                "strict-train camera only",
                "availability for globally selected pose",
                "full body, hands, and feet pass",
            ],
            "front_axis_calibration": (
                "derive one immutable body-front azimuth offset from the formal base coordinate "
                "contract before selection; record its value and SHA"
            ),
            "score_order": [
                "absolute wrapped camera azimuth error after front-axis calibration",
                "absolute camera elevation error to the frozen identity-specific target elevation",
                "hand visibility penalty",
                "foot visibility penalty",
                "crop-margin penalty",
                "camera_id",
            ],
            "yaw_tolerance_degrees": 22.5,
            "future_generation_quality_used": False,
            "manual_best_looking_frame_selection": False,
        },
        "availability_source": SOURCE_OBJECTS["formal_availability"],
        "mappings": mappings,
    }


def condition_manifest(garments: list[dict[str, Any]], slots: dict[str, Any]) -> dict[str, Any]:
    mapping = pose_camera_mapping(slots)
    records = []
    for garment in garments:
        for ordinal, slot in enumerate(SLOTS, 1):
            records.append(
                {
                    "condition_id": f"subject00/{garment['garment_id']}/{slot}",
                    "identity_id": "subject00",
                    "garment_id": garment["garment_id"],
                    "semantic_pose_slot": slot,
                    "slot_ordinal": ordinal,
                    "endpoint_role": True,
                    "reference_role": slot in REFERENCE_SLOTS,
                    "materialization_status": PENDING,
                    "formal_checkpoint_step": 101245,
                    "formal_checkpoint_path": PENDING,
                    "formal_checkpoint_sha256": PENDING,
                    "selected_pose_id": PENDING,
                    "selected_camera_id": PENDING,
                    "pose_camera_selection_record_sha256": PENDING,
                    "source_rgb_path": PENDING,
                    "source_rgb_sha256": PENDING,
                    "source_alpha_path": PENDING,
                    "source_alpha_sha256": PENDING,
                    "source_depth_path": PENDING,
                    "source_depth_sha256": PENDING,
                    "source_pose_path": PENDING,
                    "source_pose_sha256": PENDING,
                    "source_camera_path": PENDING,
                    "source_camera_sha256": PENDING,
                    "renderer_config_sha256": PENDING,
                    "renderer_code_sha256": PENDING,
                    "future_generated_output_sha256": PENDING,
                }
            )
    return seal(
        {
            "schema_version": "subject00.three_garment.condition_manifest.v1",
            "task_id": TASK_ID,
            "status": PENDING,
            "formal_base_checkpoint_available": False,
            "planned_condition_record_count": len(records),
            "materialized_condition_record_count": 0,
            "reference_record_count_planned": 12,
            "endpoint_record_count_planned": 24,
            "reference_is_endpoint_subset": True,
            "reference_and_endpoint_asset_overlap_disclosure": (
                "The four cardinal references per garment are the same immutable generated "
                "endpoint assets used by the endpoint bank."
            ),
            "strict_evaluation_asset_overlap": 0,
            "pose_camera_mapping": mapping,
            "records": records,
            "condition_renders_in_this_task": 0,
            "paper_final": 0,
        }
    )


def formal_render_contract(slots: dict[str, Any]) -> dict[str, Any]:
    image = slots["image_contract"]
    return {
        "authorization": "FUTURE_TASK_ONLY_AFTER_FORMAL_BASE_EXISTS",
        "only_checkpoint_step": 101245,
        "checkpoint_path": PENDING,
        "checkpoint_sha256": PENDING,
        "formal_protocol_head": FORMAL_HEAD,
        "renderer": {
            "backend": "gsplat.rasterization",
            "entrypoint": "scene.gaussian_model.GaussianModel.render",
            "code_sha256": PENDING,
            "config_path": "config/subject00_surface_lbs_formal_strict_split.yaml",
            "config_sha256": PENDING,
            "scaling_modifier": 1.0,
            "image_scaling": 1,
            "rgb_clamp": [0.0, 1.0],
        },
        "camera_and_pose": {
            "selection_contract": "condition_manifest.pose_camera_mapping",
            "strict_train_only": True,
            "pose_path_and_sha_required": True,
            "camera_path_and_sha_required": True,
        },
        "image_contract": {
            "resolution_width": image["resolution_width"],
            "resolution_height": image["resolution_height"],
            "background": image["background"],
            "lighting": image["lighting"],
            "framing": image["camera_framing"],
            "full_body": True,
            "feet_complete": True,
            "hands_visible": True,
            "numbering_forbidden": True,
            "close_up_forbidden": True,
        },
        "required_outputs": [
            "RGB", "alpha", "depth", "source_pose", "source_camera",
            "checkpoint_sha256", "renderer_sha256", "output_sha256",
        ],
        "append_only": True,
        "overwrite_allowed": False,
        "render_count_in_this_task": 0,
    }


def generation_contract(garments: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_records = []
    for garment in garments:
        prompt_records.append(
            {
                "garment_id": garment["garment_id"],
                "reference_prompt": garment["reference_prompt"],
                "reference_prompt_sha256": text_sha256(garment["reference_prompt"]),
                "negative_prompt": garment["negative_prompt"],
                "negative_prompt_sha256": text_sha256(garment["negative_prompt"]),
                "semantic_fields_copied_without_change": [
                    "semantic_description", "colors", "material_appearance", "fit",
                    "sleeve_length", "hem", "forbidden_attributes", "reference_prompt",
                    "negative_prompt",
                ],
            }
        )
    return seal(
        {
            "schema_version": "subject00.three_garment.generation_contract.v1",
            "task_id": TASK_ID,
            "status": BACKEND_PENDING,
            "backend_dependency": {
                "status": BACKEND_PENDING,
                "provider": BACKEND_PENDING,
                "model_id_and_revision": BACKEND_PENDING,
                "base_url_identifier": BACKEND_PENDING,
                "candidate_generation_count_per_garment": BACKEND_PENDING,
                "minimum_acceptable_count_per_garment": BACKEND_PENDING,
                "quality_and_image_parameters": BACKEND_PENDING,
                "must_be_frozen_before_first_request": True,
            },
            "request_required_fields": [
                "identity_id", "garment_id", "semantic_pose_slot", "condition_id",
                "source_rgb_sha256", "source_alpha_sha256", "prompt_sha256",
                "negative_prompt_sha256", "provider", "model_id",
                "base_url_identifier", "seed", "image_size", "quality",
                "response_metadata", "output_sha256", "moderation_state",
                "retry_parent_record_id",
            ],
            "identity_id": "subject00",
            "garment_ids": GARMENT_IDS,
            "semantic_pose_slots": SLOTS,
            "prompt_records": prompt_records,
            "seed_policy": (
                "uint32 big-endian first four bytes of "
                "sha256(identity_id|garment_id|semantic_pose_slot|attempt_index)"
            ),
            "record_policy": {
                "append_only": True,
                "failed_outputs_retained": True,
                "moderated_outputs_retained": True,
                "retry_creates_new_record": True,
                "retry_parent_required": True,
                "successful_output_overwrite_allowed": False,
            },
            "medium_preview_as_formal_allowed": False,
            "adapter": {
                "path": "tools/datasets/subject00_three_garment_generation_record_adapter.py",
                "submits_requests": False,
                "mode": "VALIDATE_AND_NORMALIZE_ONLY",
            },
            "api_calls_in_this_task": 0,
            "generated_images_in_this_task": 0,
            "paper_final": 0,
        }
    )


def geometry_metrics() -> dict[str, list[str]]:
    return {
        "O01": [
            "torso_bulk", "hood_presence", "sleeve_looseness", "hem_offset",
            "body_conforming_bias",
        ],
        "O03": [
            "lapel_and_upper_torso_structure", "shoulder_line", "jacket_hem",
            "trouser_silhouette", "structural_symmetry",
        ],
        "O04": [
            "outer_layer_thickness", "collar_and_jacket_opening", "sleeve_thickness",
            "waist_and_hem_offset", "jeans_silhouette",
        ],
    }


def screening_contract() -> dict[str, Any]:
    shared = [
        "identity_preservation", "garment_semantic_match", "full_body", "hands",
        "feet", "anatomy", "background", "blur", "text_or_watermark",
        "view_consistency", "cross_view_garment_consistency", "body_conforming_bias",
        "garment_volume", "silhouette",
    ]
    rejection = [
        "IDENTITY_DRIFT", "GARMENT_SEMANTIC_MISMATCH", "BODY_CROPPED",
        "HAND_MISSING_OR_INVALID", "FOOT_MISSING_OR_INVALID", "ANATOMY_FAILURE",
        "BACKGROUND_FAILURE", "BLUR", "TEXT_OR_WATERMARK", "VIEW_INCONSISTENT",
        "CROSS_VIEW_GARMENT_INCONSISTENT", "BODY_CONFORMING_COLLAPSE",
        "GARMENT_VOLUME_MISSING", "SILHOUETTE_FAILURE", "GARMENT_SPECIFIC_GEOMETRY_FAILURE",
    ]
    return seal(
        {
            "schema_version": "subject00.three_garment.screening_contract.v1",
            "task_id": TASK_ID,
            "status": "FROZEN_PENDING_CANDIDATES",
            "dimensions": [
                {"name": name, "required": True, "hard_reject_on_fail": True}
                for name in shared
            ],
            "garment_specific_geometry_metrics": geometry_metrics(),
            "geometry_metrics_may_change_prompt": False,
            "decision_policy": {
                "automated_checks_must_pass": True,
                "minimum_independent_human_reviewers": 2,
                "reviewers_work_independently": True,
                "vlm_only_acceptance_allowed": False,
                "automatic_acceptance_allowed": False,
                "two_accept_votes_required": True,
                "disagreement_record_required": True,
                "disagreement_resolution": "third independent human reviewer; majority with reasons retained",
            },
            "generation_budget_dependency": {
                "candidate_generation_count_per_garment": BACKEND_PENDING,
                "minimum_acceptable_count_per_garment": BACKEND_PENDING,
            },
            "rejection_reason_taxonomy": rejection,
            "replacement_rule": (
                "A rejected or missing slot is replaced only by a new append-only attempt for the "
                "same garment and slot under the same frozen backend contract."
            ),
            "record_policy": {
                "candidate_records_append_only": True,
                "review_records_append_only": True,
                "failed_and_rejected_records_retained": True,
                "rejected_candidate_as_teacher_input_allowed": False,
            },
            "screened_candidates_in_this_task": 0,
            "accepted_endpoints_in_this_task": 0,
            "paper_final": 0,
        }
    )


def teacher_contract() -> dict[str, Any]:
    return seal(
        {
            "schema_version": "subject00.three_garment.teacher_contract.v1",
            "task_id": TASK_ID,
            "status": "FROZEN_NOT_AUTHORIZED_FOR_TRAINING",
            "garment_ids": GARMENT_IDS,
            "runs": 3,
            "input_contract": {
                "accepted_generated_rgb_required": True,
                "rejected_generated_rgb_forbidden": True,
                "source_alpha_or_mask_required": True,
                "source_pose_and_camera_required": True,
                "garment_id_required": True,
                "endpoint_slot_required": True,
                "identity_protected_regions_required": True,
                "accepted_endpoint_count_per_garment": 8,
                "strict_heldout_pose_or_camera_target_forbidden": True,
            },
            "base_contract": {
                "checkpoint_step": 101245,
                "checkpoint_sha256": PENDING,
                "checkpoint_status": PENDING,
                "template_lbs_mode": "surface_attachment_cached",
                "surface_attachment_source": "subject00 formal base checkpoint provenance",
            },
            "target_residual_schema": {
                "space": "shared_canonical_per_gaussian_residual",
                "channels": [
                    "dxyz", "dsh0", "dshN", "dscale", "drotation", "dopacity",
                ],
                "support": "same frozen 200000 Gaussian support as formal base",
                "identity_protected_regions": "zero residual and protected loss",
            },
            "initialization": {
                "all_residual_channels": "EXACT_ZERO",
                "seed": 20260718,
                "pretrained_subject02_teacher_load": False,
                "medium_checkpoint_load": False,
            },
            "optimizer": {
                "class": "Adam",
                "geometry_learning_rate": 0.001,
                "appearance_learning_rate": 0.002,
                "gradient_clip_norm": 1.0,
                "optimizer_steps": 1200,
                "sweep_allowed": False,
            },
            "checkpoint_cadence": [0, 40, 80, 160, 240, 400, 600, 800, 1000, 1200],
            "evaluation": {
                "endpoint_parity_required": True,
                "numerical_stability_required": True,
                "identity_contamination_required": 0,
                "component_contamination_required": 0,
                "per_garment_reporting_required": True,
                "average_may_hide_single_garment_failure": False,
            },
            "topology_and_lbs_invariants": {
                "clone_calls": 0,
                "split_calls": 0,
                "densification_calls": 0,
                "off_surface_rebind_calls": 0,
                "topology_changing_prune_calls": 0,
                "cached_surface_attachment_mutation": 0,
            },
            "teacher_training_runs_in_this_task": 0,
            "optimizer_created_in_this_task": 0,
            "checkpoint_writes_in_this_task": 0,
            "paper_final": 0,
        }
    )


def success_gates() -> dict[str, Any]:
    per_garment = {
        garment_id: {
            "endpoint_slots_valid": 8,
            "reference_slots_valid": 4,
            "teacher_numerically_stable": True,
            "endpoint_parity_pass": True,
            "identity_contamination": 0,
            "component_contamination": 0,
            "severe_body_conforming_collapse": 0,
            "severe_missing_garment_volume": 0,
            "severe_silhouette_failure": 0,
        }
        for garment_id in GARMENT_IDS
    }
    return seal(
        {
            "schema_version": "subject00.three_garment.success_gates.v1",
            "task_id": TASK_ID,
            "status": "FROZEN_FUTURE_GATES",
            "data_gates": {
                "garments_complete": "3/3",
                "endpoint_slots_valid": "24/24",
                "references_valid": "12/12",
                "identity_preservation": "PASS",
                "cross_view_consistency": "PASS",
            },
            "teacher_gates": {
                "stable_runs": "3/3",
                "endpoint_parity": "PASS",
                "identity_contamination": 0,
                "component_contamination": 0,
            },
            "per_garment_gates": per_garment,
            "single_garment_failure_may_be_hidden_by_average": False,
            "capacity_failure_classification": "SUBJECT00_THREE_GARMENT_CANARY_REPRESENTATION_LIMITED",
            "paper_final": 0,
        }
    )


def protocol_yaml(slots: dict[str, Any]) -> dict[str, Any]:
    return seal(
        {
            "schema_version": "subject00.three_garment.canary_protocol.v1",
            "task_id": TASK_ID,
            "source": {
                "branch": "research/multi-identity-garment-benchmark-foundations-20260724",
                "head": SOURCE_HEAD,
                "formal_protocol_head": FORMAL_HEAD,
                "medium_head": MEDIUM_HEAD,
            },
            "branch": BRANCH,
            "classification": CLASSIFICATION,
            "formal_base": {
                "required_step": 101245,
                "checkpoint_available": False,
                "status": PENDING,
                "medium_use": "PREVIEW_ONLY",
                "medium_may_be_promoted_to_formal": False,
            },
            "selected_garments": GARMENT_IDS,
            "semantic_slots": SLOTS,
            "reference_slots": REFERENCE_SLOTS,
            "reference_count": 12,
            "endpoint_condition_count": 24,
            "formal_render_contract": formal_render_contract(slots),
            "source_objects": SOURCE_OBJECTS,
            "claim_boundary": {
                "allowed": "identity-conditioned synthetic closed-wardrobe garment endpoint benchmark",
                "canary_pass_supports_only": "subject00 can carry synthetic closed-wardrobe endpoints",
                "forbidden": [
                    "captured multi-outfit subject00",
                    "real garment-transfer ground truth",
                    "real cross-identity garment data",
                    "original THuman4.0 multi-garment sequence",
                ],
            },
            "execution_counts": zero_counts(),
            "paper_final": 0,
        }
    )


def zero_counts() -> dict[str, int]:
    return {
        "formal_base_checkpoint_available": 0,
        "preview_renders": 0,
        "formal_condition_renders": 0,
        "image_api_calls": 0,
        "generated_images": 0,
        "candidate_files": 0,
        "accepted_endpoints": 0,
        "teacher_training_runs": 0,
        "optimizer_created": 0,
        "checkpoint_writes": 0,
        "raw_mutation": 0,
        "formal_base_mutation": 0,
        "paper_final": 0,
    }


def final_summary(
    protocol: dict[str, Any],
    manifest: dict[str, Any],
    generation: dict[str, Any],
    screening: dict[str, Any],
    teacher: dict[str, Any],
    gates: dict[str, Any],
    wardrobe: dict[str, Any],
) -> dict[str, Any]:
    wardrobe_path = ROOT / SOURCE_OBJECTS["shared_five_garment_wardrobe"]["path"]
    slot_path = ROOT / SOURCE_OBJECTS["shared_condition_pose_view_protocol"]["path"]
    return seal(
        {
            "schema_version": "subject00.three_garment.protocol_final_summary.v1",
            "task_id": TASK_ID,
            "status": "PASS",
            "classification": CLASSIFICATION,
            "source_head": SOURCE_HEAD,
            "formal_protocol_head": FORMAL_HEAD,
            "medium_head": MEDIUM_HEAD,
            "selected_garments": GARMENT_IDS,
            "selected_garment_semantics": selected_garments(wardrobe),
            "shared_wardrobe_hashes": {
                "content_sha256": wardrobe["content_sha256"],
                "file_sha256": bytes_sha256(wardrobe_path),
                "git_blob": SOURCE_OBJECTS["shared_five_garment_wardrobe"]["git_blob"],
            },
            "shared_slot_hashes": {
                "content_sha256": SOURCE_OBJECTS["shared_condition_pose_view_protocol"]["content_sha256"],
                "file_sha256": bytes_sha256(slot_path),
                "git_blob": SOURCE_OBJECTS["shared_condition_pose_view_protocol"]["git_blob"],
            },
            "formal_base_dependency": protocol["formal_base"],
            "condition_mapping_status": manifest["pose_camera_mapping"]["mapping_status"],
            "generation_dependency": generation["status"],
            "screening_status": screening["status"],
            "teacher_status": teacher["status"],
            "success_gate_status": gates["status"],
            "counts": zero_counts(),
            "immutability": {
                "shared_wardrobe_mutation": 0,
                "shared_slot_protocol_mutation": 0,
                "subject00_raw_mutation": 0,
                "formal_base_mutation": 0,
                "formal_split_mutation": 0,
                "medium_output_mutation": 0,
            },
            "claim_boundary": protocol["claim_boundary"],
            "paper_final": 0,
            "next_task": NEXT_TASK,
            "next_task_started": False,
        }
    )


def reviewer_form() -> dict[str, Any]:
    return {
        "schema_version": "subject00.three_garment.reviewer_form.v1",
        "candidate_record_id": "${FUTURE_CANDIDATE_RECORD_ID}",
        "reviewer_id": "${INDEPENDENT_REVIEWER_ID}",
        "review_round": 1,
        "garment_id": "${O01_OR_O03_OR_O04}",
        "semantic_pose_slot": "${FROZEN_SLOT}",
        "opened_original_detail": False,
        "automated_checks_pass": False,
        "dimension_decisions": {
            name: "UNREVIEWED"
            for name in [
                "identity_preservation", "garment_semantic_match", "full_body", "hands",
                "feet", "anatomy", "background", "blur", "text_or_watermark",
                "view_consistency", "cross_view_garment_consistency",
                "body_conforming_bias", "garment_volume", "silhouette",
            ]
        },
        "garment_specific_geometry": "UNREVIEWED",
        "decision": "UNREVIEWED",
        "rejection_reasons": [],
        "disagreement_record_id": "NOT_APPLICABLE_UNLESS_DISAGREEMENT",
        "notes": "",
    }


def generation_record_template() -> dict[str, Any]:
    return {
        "schema_version": "subject00.three_garment.generation_event.v1",
        "record_id": "${APPEND_ONLY_RECORD_ID}",
        "event_type": "REQUEST_PREPARED",
        "attempt_index": 0,
        "identity_id": "subject00",
        "garment_id": "${O01_OR_O03_OR_O04}",
        "condition_id": "${FROZEN_CONDITION_ID}",
        "semantic_pose_slot": "${FROZEN_SLOT}",
        "source_rgb_sha256": "${FUTURE_LOWERCASE_SHA256}",
        "source_alpha_sha256": "${FUTURE_LOWERCASE_SHA256}",
        "prompt_sha256": "${FROZEN_PROMPT_SHA256}",
        "negative_prompt_sha256": "${FROZEN_NEGATIVE_PROMPT_SHA256}",
        "provider": "${FROZEN_BACKEND_PROVIDER}",
        "model_id": "${FROZEN_MODEL_ID_AND_REVISION}",
        "base_url_identifier": "${FROZEN_NON_SECRET_IDENTIFIER}",
        "seed": "${DERIVED_UINT32}",
        "image_size": {"width": 1024, "height": 1536},
        "quality": "${FROZEN_BACKEND_QUALITY}",
        "request_timestamp": "${FUTURE_TIMESTAMP}",
        "response_metadata": {},
        "output_sha256": "${FUTURE_LOWERCASE_SHA256}",
        "moderation_state": "${FUTURE_STATE}",
        "retry_parent_record_id": "NONE_FOR_FIRST_ATTEMPT",
    }


def build_reports(summary: dict[str, Any]) -> None:
    reports = {
        "docs/DATASET/SUBJECT00_THREE_GARMENT_CANARY_PROTOCOL_20260724.md": f"""
# Subject00 Three-Garment Canary Protocol

Task: `{TASK_ID}`

The canary is fixed to `O01/O03/O04`, twelve references, and twenty-four endpoint conditions. The four cardinal references (`front/left/back/right`) are an explicitly disclosed subset of the eight endpoint slots. Garment semantics, prompt text, negative prompts, color, material, fit, sleeve, hem, and forbidden attributes are copied byte-for-byte at field level from the shared five-garment wardrobe; that five-garment contract is not modified.

Formal condition sources may be rendered only from the future Subject00 final checkpoint at step `101245` under formal protocol `{FORMAL_HEAD}`. The checkpoint is unavailable, so pose IDs, camera IDs, paths, and output hashes remain `PENDING_FORMAL_BASE`. The medium checkpoint is `PREVIEW_ONLY`, no preview was rendered, and promotion of a medium preview to formal data is forbidden.

All endpoint and teacher-target conditions use strict-train poses and cameras. Held-out poses/cameras are reserved for future strict evaluation and cannot be teacher targets. Selection is deterministic from availability, full-body completeness, hand/foot visibility, camera azimuth/elevation error, pose semantic distance, and stable IDs. Future generation quality and manual best-looking selection are forbidden.

Classification: `{CLASSIFICATION}`.
""",
        "docs/DATASET/SUBJECT00_THREE_GARMENT_GEOMETRY_CHALLENGES_20260724.md": """
# Subject00 Three-Garment Geometry Challenges

The geometry audit is per garment; averages cannot conceal a single-garment capacity failure. O01 evaluates torso bulk, hood presence, sleeve looseness, hem offset, and body-conforming bias. O03 evaluates lapel and upper-torso structure, shoulder line, jacket hem, trouser silhouette, and structural symmetry. O04 evaluates outer-layer thickness, collar and jacket opening, sleeve thickness, waist/hem offset, and jeans silhouette.

These dimensions are future screening and teacher-evaluation measurements only. They cannot be used to revise the already frozen shared wardrobe prompt. Each garment must independently avoid severe body-conforming collapse, missing garment volume, and silhouette failure. Any explicit capacity failure yields `SUBJECT00_THREE_GARMENT_CANARY_REPRESENTATION_LIMITED`.
""",
        "docs/DATASET/SUBJECT00_THREE_GARMENT_SCREENING_PROTOCOL_20260724.md": """
# Subject00 Three-Garment Screening Protocol

Every candidate must pass automated checks and independent review by at least two humans. VLM-only and automatic acceptance are forbidden. Both reviewers must accept identity preservation, garment semantics, full body, hands, feet, anatomy, background, blur, text/watermark, view consistency, cross-view garment consistency, body-conforming bias, garment volume, silhouette, and garment-specific geometry. A disagreement creates an append-only record and a third independent review; reasons and rejected candidates remain preserved.

Candidate and minimum acceptable counts inherit the still-pending unified generation backend contract. Replacements are append-only retries for the same garment and slot under an unchanged backend contract. Rejected candidates never become teacher inputs.
""",
        "docs/DATASET/SUBJECT00_THREE_GARMENT_TEACHER_CONTRACT_20260724.md": """
# Subject00 Three-Garment Teacher Contract

Future teachers consume only accepted generated RGB, source alpha/mask, frozen pose/camera, garment ID, endpoint slot, and identity-protected regions. Each of O01/O03/O04 starts from exact-zero shared-canonical residuals on the future formal step-101245 base and runs a fixed 1200-step Adam contract with no sweep. Checkpoints are fixed at 0/40/80/160/240/400/600/800/1000/1200.

Subject00 uses `surface_attachment_cached`. Clone, split, densification, topology-changing prune, spatial/off-surface rebind, and cached-attachment mutation are all forbidden. Each garment must separately pass finite numerics, endpoint parity, identity contamination equal to zero, component contamination equal to zero, and the geometry challenge gates. No teacher, optimizer, or checkpoint was created by this protocol task.
""",
    }
    for path, value in reports.items():
        write_text(path, value)


def update_skeleton(
    protocol: dict[str, Any], manifest: dict[str, Any], generation: dict[str, Any], screening: dict[str, Any]
) -> None:
    base = "multi_garment_benchmark/subject00"
    write_text(
        f"{base}/README.md",
        f"""
# subject00 Three-Garment Canary Protocol Skeleton

This tracked tree contains protocol-only artifacts for `{TASK_ID}`. Formal base checkpoint status is `PENDING_FORMAL_BASE`; generated candidates, accepted endpoints, teacher inputs, endpoint banks, renders, and checkpoints are absent. O01/O03/O04 use the shared wardrobe without semantic changes. Medium outputs are preview-only and can never be promoted to formal data.
""",
    )
    write_json(f"{base}/identity_manifest/subject00_three_garment_canary_protocol.json", protocol)
    write_json(f"{base}/condition_manifests/subject00_three_garment_condition_manifest.json", manifest)
    write_json(f"{base}/generation_contract/subject00_three_garment_generation_contract.json", generation)
    write_json(f"{base}/generation_contract/generation_record_template.json", generation_record_template())
    write_json(f"{base}/screening/subject00_three_garment_screening_contract.json", screening)
    write_json(f"{base}/screening/reviewer_form.json", reviewer_form())


def handoff(summary: dict[str, Any]) -> dict[str, Any]:
    return seal(
        {
            "schema_version": "subject00.three_garment.protocol_handoff.v1",
            "task_id": TASK_ID,
            "status": "PASS",
            "classification": CLASSIFICATION,
            "source_head": SOURCE_HEAD,
            "branch": BRANCH,
            "formal_base_dependency": PENDING,
            "generation_backend_dependency": BACKEND_PENDING,
            "summary": "paper_protocol/datasets/subject00_three_garment_protocol_final_summary.json",
            "reports": [
                "docs/DATASET/SUBJECT00_THREE_GARMENT_CANARY_PROTOCOL_20260724.md",
                "docs/DATASET/SUBJECT00_THREE_GARMENT_GEOMETRY_CHALLENGES_20260724.md",
                "docs/DATASET/SUBJECT00_THREE_GARMENT_SCREENING_PROTOCOL_20260724.md",
                "docs/DATASET/SUBJECT00_THREE_GARMENT_TEACHER_CONTRACT_20260724.md",
            ],
            "counts": summary["counts"],
            "next_task": NEXT_TASK,
            "next_task_started": False,
            "paper_final": 0,
        }
    )


def main() -> int:
    wardrobe, slots = source_payloads()
    garments = selected_garments(wardrobe)
    protocol = protocol_yaml(slots)
    manifest = condition_manifest(garments, slots)
    generation = generation_contract(garments)
    screening = screening_contract()
    teacher = teacher_contract()
    gates = success_gates()
    summary = final_summary(
        protocol, manifest, generation, screening, teacher, gates, wardrobe
    )

    outputs = {
        "paper_protocol/datasets/subject00_three_garment_canary_protocol.yaml": protocol,
        "paper_protocol/datasets/subject00_three_garment_condition_manifest.json": manifest,
        "paper_protocol/datasets/subject00_three_garment_generation_contract.json": generation,
        "paper_protocol/datasets/subject00_three_garment_screening_contract.json": screening,
        "paper_protocol/datasets/subject00_three_garment_teacher_contract.json": teacher,
        "paper_protocol/datasets/subject00_three_garment_success_gates.json": gates,
        "paper_protocol/datasets/subject00_three_garment_protocol_final_summary.json": summary,
        "project_control_handoff/subject00_three_garment_protocol_handoff.json": handoff(summary),
    }
    for path, payload in outputs.items():
        write_json(path, payload)
    update_skeleton(protocol, manifest, generation, screening)
    build_reports(summary)
    print(
        json.dumps(
            {
                "status": "PASS",
                "classification": CLASSIFICATION,
                "formal_base_checkpoint_available": False,
                "condition_renders": 0,
                "image_api_calls": 0,
                "generated_images": 0,
                "teacher_training_runs": 0,
                "optimizer_created": 0,
                "checkpoint_writes": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
