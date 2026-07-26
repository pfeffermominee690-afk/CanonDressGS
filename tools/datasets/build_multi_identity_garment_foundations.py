#!/usr/bin/env python3
"""Build static multi-identity garment benchmark foundation artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "AAAI27-MULTI-IDENTITY-GARMENT-DATA-PREPARATION-001"
SOURCE_HEAD = "a8557b461f301c19a0f24eb17d924ddd9159a580"
SUBJECT00_FORMAL_HEAD = "4993f5c865ec19895f35811fa399fc4a1834c6a7"
SUBJECT00_MEDIUM_HEAD = "2d0913eb1b163a79e81c1804c216f91b0a737b44"
GARMENT_IDS = ["O01", "O02", "O03", "O04", "O08"]
SLOTS = [
    ("front", 0),
    ("front_left_three_quarter", 45),
    ("left", 90),
    ("back_left_three_quarter", 135),
    ("back", 180),
    ("back_right_three_quarter", 225),
    ("right", 270),
    ("front_right_three_quarter", 315),
]
CARDINAL_REFERENCE_SLOTS = ["front", "left", "back", "right"]
SKELETON_DIRS = [
    "raw_references",
    "identity_manifest",
    "condition_manifests",
    "condition_previews",
    "generation_contract",
    "generated_candidates",
    "screening",
    "accepted_endpoints",
    "teacher_inputs",
    "endpoint_bank",
    "reports",
    "provenance",
]


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def write_text(relative: str, text: str) -> None:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8", newline="\n")


def historical_generation() -> dict[str, Any]:
    return {
        "generation_source": "CODEX_IMAGE_GENERATION_SKILL",
        "generation_mode": "CODEX_PLATFORM_MANAGED_DIRECT_IMAGE_EDIT",
        "model_id": None,
        "model_status": "NOT_EXPOSED_BY_PLATFORM",
        "seed": None,
        "seed_status": "NOT_RECORDED",
        "generation_parameters": {
            "known": {"expected_width": 1024, "expected_height": 1536},
            "unknown": [
                "backend_model_revision",
                "sampler_or_scheduler",
                "steps",
                "guidance",
                "provider_side_image_edit_parameters",
            ],
        },
        "provenance_completeness": "LIMITED_GARMENT_GENERATION_PROVENANCE",
    }


def wardrobe() -> dict[str, Any]:
    shared_negative = (
        "different identity, copied donor face, copied donor body shape, changed pose, changed camera, "
        "changed hands, changed feet, cropped body, close-up, extra person, duplicate limb, missing limb, "
        "text, number, logo, watermark, collage, border, non-neutral background, inconsistent garment across views"
    )
    records = [
        {
            "garment_id": "O01",
            "semantic_description": "Light heather-gray pullover hoodie with black full-length straight trousers.",
            "upper_garment_type": "pullover hooded sweatshirt",
            "lower_garment_type": "full-length straight trousers",
            "colors": {"upper": "light heather gray", "lower": "black"},
            "material_appearance": {"upper": "sweatshirt knit or fleece appearance", "lower": "matte woven appearance"},
            "fit": {"upper": "regular-to-relaxed", "lower": "regular straight"},
            "sleeve_length": "long",
            "hem": "ribbed waist-length hoodie hem; full-length trouser hems",
            "forbidden_attributes": ["zip-front hoodie", "short sleeves", "blue trousers", "double hood", "retained old cuffs"],
            "reference_prompt": "A light heather-gray pullover hoodie with hood, drawstrings, long sleeves, ribbed cuffs and waist hem, paired with black full-length straight trousers.",
        },
        {
            "garment_id": "O02",
            "semantic_description": "White long-sleeve button-front dress shirt tucked into dark navy tailored slacks.",
            "upper_garment_type": "button-front dress shirt",
            "lower_garment_type": "tailored slacks",
            "colors": {"upper": "white", "lower": "dark navy"},
            "material_appearance": {"upper": "smooth poplin-like woven appearance", "lower": "matte suiting appearance"},
            "fit": {"upper": "regular tailored and tucked", "lower": "tailored straight"},
            "sleeve_length": "long",
            "hem": "shirt hem tucked into waistband; full-length slacks",
            "forbidden_attributes": ["short sleeves", "untucked shirt", "hood", "jeans", "open shirt front"],
            "reference_prompt": "A crisp white long-sleeve button-front dress shirt with collar and button cuffs, tucked into dark navy full-length tailored slacks.",
        },
        {
            "garment_id": "O03",
            "semantic_description": "Dark navy single-breasted business suit with matching trousers, white dress shirt, and dark navy tie.",
            "upper_garment_type": "single-breasted two-button suit jacket over dress shirt and tie",
            "lower_garment_type": "matching tailored suit trousers",
            "colors": {"jacket_and_trousers": "dark navy", "shirt": "white", "tie": "dark navy"},
            "material_appearance": {"jacket_and_trousers": "smooth suiting appearance", "shirt": "smooth woven appearance"},
            "fit": {"upper": "tailored", "lower": "tailored straight"},
            "sleeve_length": "long",
            "hem": "hip-length suit jacket; full-length matching trousers",
            "forbidden_attributes": ["casual jacket", "missing tie", "mismatched trousers", "open bare chest", "short sleeves"],
            "reference_prompt": "A dark navy tailored single-breasted two-button business suit with matching full-length trousers, a white dress shirt, and a dark navy tie.",
        },
        {
            "garment_id": "O04",
            "semantic_description": "Black waist-length zip-front bomber jacket over a light-gray crew-neck shirt with medium-blue denim jeans.",
            "upper_garment_type": "zip-front bomber jacket over crew-neck shirt",
            "lower_garment_type": "full-length denim jeans",
            "colors": {"jacket": "black", "inner_shirt": "light gray", "lower": "medium blue"},
            "material_appearance": {"jacket": "matte lightweight woven appearance", "lower": "denim appearance"},
            "fit": {"upper": "regular waist-length", "lower": "regular straight"},
            "sleeve_length": "long jacket sleeves",
            "hem": "waist-length jacket hem; full-length jean hems",
            "forbidden_attributes": ["hood", "long coat", "black trousers", "formal blazer", "distressed or torn jeans"],
            "reference_prompt": "A black waist-length zip-front bomber jacket with long sleeves over a light-gray crew-neck shirt, paired with medium-blue full-length straight denim jeans.",
        },
        {
            "garment_id": "O08",
            "semantic_description": "Medium-gray rib-knit crew-neck sweater with olive-gray full-length straight trousers.",
            "upper_garment_type": "crew-neck knit sweater",
            "lower_garment_type": "full-length straight trousers",
            "colors": {"upper": "medium gray", "lower": "olive gray"},
            "material_appearance": {"upper": "fine rib-knit appearance", "lower": "matte woven appearance"},
            "fit": {"upper": "regular", "lower": "regular straight"},
            "sleeve_length": "long",
            "hem": "ribbed waist-length sweater hem; full-length trouser hems",
            "forbidden_attributes": ["hood", "cardigan opening", "short sleeves", "blue jeans", "cargo pockets"],
            "reference_prompt": "A medium-gray fine rib-knit crew-neck sweater with long sleeves and ribbed cuffs and hem, paired with olive-gray full-length straight trousers.",
        },
    ]
    for record in records:
        record["negative_prompt"] = shared_negative + ", " + ", ".join(record["forbidden_attributes"])
        record["attribute_evidence"] = "VISUAL_APPEARANCE_AUDIT_ONLY; FIBER_CONTENT_NOT_VERIFIED"
        record["historical_generation"] = historical_generation()
        record["prospective_generation"] = {
            "prompt_template": "identity-preserving full-body clothing edit using reference_prompt and the shared condition protocol",
            "provider_model_and_revision": "MUST_BE_PINNED_BEFORE_FIRST_REQUEST",
            "seed_policy": "derive uint32 from sha256(identity_id|garment_id|semantic_pose_slot|attempt_index)",
            "parameters": "MUST_BE_COMPLETE_AND_IMMUTABLE_BEFORE_FIRST_REQUEST",
        }
        record["screening_rule_set"] = "multi_identity_screening_contract.v1"
    return seal(
        {
            "schema_version": "multi_identity.shared_five_garment_wardrobe.v1",
            "task_id": TASK_ID,
            "status": "LIMITED_GARMENT_GENERATION_PROVENANCE",
            "garment_count": len(records),
            "garment_ids": GARMENT_IDS,
            "source_evidence": {
                "mapping": "E:/data_pre/audit_subject02_outfit_conversion_pilot_v1/outfit_cell_mapping_v2_normalized.json",
                "visual_reference_front_sheet_01": "E:/data_pre/outputs/gpt_image2_pose_outfit_sheets/Jay/sheet_01/pose_000000_sheet01.png",
                "visual_reference_front_sheet_02": "E:/data_pre/outputs/gpt_image2_pose_outfit_sheets/Jay/sheet_02/pose_000000_sheet02.png",
                "historical_generation_contract": "artifacts/aaai27_sprint/codex_direct_generation/generation_contract.json",
                "historical_prompt_template": "artifacts/aaai27_sprint/codex_direct_generation/production_prompt_template.txt",
            },
            "sharing_boundary": {
                "shared": ["garment semantics", "prompt template", "view slots", "screening thresholds", "endpoint count"],
                "never_shared": ["face", "identity pixels", "target images", "body shape", "raw frame IDs"],
            },
            "garments": records,
            "generated_images_in_this_task": 0,
            "paper_final": 0,
        }
    )


def condition_protocol() -> dict[str, Any]:
    return seal(
        {
            "schema_version": "multi_identity.shared_condition_pose_view_protocol.v1",
            "task_id": TASK_ID,
            "status": "FROZEN_PROSPECTIVE_PROTOCOL",
            "unit": "SEMANTIC_POSE_SLOT",
            "condition_slot_count": len(SLOTS),
            "reference_slot_count_per_garment": len(CARDINAL_REFERENCE_SLOTS),
            "endpoint_count_per_garment": len(SLOTS),
            "reference_slots": CARDINAL_REFERENCE_SLOTS,
            "slots": [
                {
                    "semantic_pose_slot": name,
                    "target_body_yaw_degrees": yaw,
                    "yaw_tolerance_degrees": 22.5,
                    "required": True,
                }
                for name, yaw in SLOTS
            ],
            "image_contract": {
                "full_body": True,
                "feet_complete": True,
                "hands_visible": True,
                "close_up_forbidden": True,
                "numbers_forbidden": True,
                "background": ["light gray", "white"],
                "lighting": "fixed per identity and reused for all garments and slots",
                "resolution_width": 1024,
                "resolution_height": 1536,
                "camera_framing": "fixed full-body portrait with identical identity-specific camera intrinsics and subject scale",
            },
            "deterministic_identity_matching": {
                "candidate_gate": ["valid source frame", "full body", "feet complete", "hands visible", "no temporal buffer frame"],
                "score_order": [
                    "absolute wrapped yaw error",
                    "hand visibility penalty",
                    "foot visibility penalty",
                    "crop-margin penalty",
                    "pose-neutrality distance",
                    "frame_id",
                    "canonical_camera_id",
                ],
                "tie_break": "lexicographic score order",
                "same_raw_frame_id_across_identities_required": False,
            },
            "identity_assignments": {
                "subject00": "PENDING_FORMAL_FINAL_CHECKPOINT",
                "avatarrex_lbn1": "PENDING_BASE_AVATAR_AND_CONDITION_PROTOCOL_PASS",
            },
            "paper_final": 0,
        }
    )


def subject00_manifest() -> dict[str, Any]:
    return seal(
        {
            "schema_version": "multi_identity.subject00_garment_preparation_manifest.v1",
            "task_id": TASK_ID,
            "identity_id": "subject00",
            "readiness": "SUBJECT00_GARMENT_BANK_PREPARATION_READY_PENDING_FORMAL_BASE",
            "evidence": {
                "surface_attached_lbs_runtime": "PASS",
                "short_canary": "PASS",
                "medium_pilot": "SUBJECT00_MMLPHUMAN_MEDIUM_PILOT_PASS",
                "medium_source_head": SUBJECT00_MEDIUM_HEAD,
                "formal_protocol": "SUBJECT00_FORMAL_STRICT_SPLIT_PROTOCOL_READY",
                "formal_source_head": SUBJECT00_FORMAL_HEAD,
                "formal_final_step": 101245,
                "formal_training_run_count": 0,
                "formal_final_checkpoint_exists": False,
            },
            "condition_source_contract": {
                "formal": "subject00 formal final checkpoint at step 101245 only",
                "medium": "PREVIEW_ONLY",
                "medium_preview_may_be_promoted_to_formal": False,
                "identity_reference_assignment": "PENDING_FORMAL_FINAL_CHECKPOINT",
                "semantic_protocol": "paper_protocol/datasets/shared_condition_pose_view_protocol.json",
            },
            "wardrobe": {
                "manifest": "paper_protocol/datasets/shared_five_garment_wardrobe.json",
                "garment_ids": GARMENT_IDS,
                "prompt_count": len(GARMENT_IDS),
            },
            "generation_adapter": {
                "path": "tools/datasets/identity_garment_generation_record_adapter.py",
                "mode": "RECORD_PREPARATION_AND_VALIDATION_ONLY",
                "submits_requests": False,
                "preview_command_template_not_executed": "python tools/datasets/identity_garment_generation_record_adapter.py --record <future_preview_record.json> --output <validated_record.json>",
            },
            "teacher_input_contract": {
                "accepted_endpoints_required": 40,
                "garment_teacher_optimizer_steps_per_garment": 1200,
                "training_authorized_by_this_manifest": False,
            },
            "counts": {
                "preview_images": 0,
                "formal_condition_images": 0,
                "generated_candidates": 0,
                "accepted_endpoints": 0,
                "avatar_training_runs": 0,
                "teacher_training_runs": 0,
                "renderer_formal_runs": 0,
            },
            "paper_final": 0,
        }
    )


def provenance_schema() -> dict[str, Any]:
    required = [
        "schema_version",
        "record_id",
        "event_type",
        "attempt_index",
        "identity_id",
        "garment_id",
        "condition_id",
        "semantic_pose_slot",
        "source_image_sha256",
        "source_mask_sha256",
        "prompt_sha256",
        "negative_prompt_sha256",
        "provider",
        "model_id",
        "api_base",
        "seed",
        "image_parameters",
        "request_timestamp",
        "response_metadata",
        "output_sha256",
        "retry_reason",
        "moderation_or_error_state",
    ]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "canondressgs.multi_identity_generation_provenance.v1",
        "title": "Append-only multi-identity garment generation event",
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": {
            "schema_version": {"const": "multi_identity.generation_event.v1"},
            "record_id": {"type": "string", "minLength": 1},
            "event_type": {"enum": ["REQUEST_PREPARED", "REQUEST_SUCCEEDED", "REQUEST_FAILED", "REQUEST_MODERATED"]},
            "attempt_index": {"type": "integer", "minimum": 0},
            "identity_id": {"enum": ["subject00", "avatarrex_lbn1"]},
            "garment_id": {"enum": GARMENT_IDS},
            "condition_id": {"type": "string", "minLength": 1},
            "semantic_pose_slot": {"enum": [name for name, _ in SLOTS]},
            "source_image_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "source_mask_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "prompt_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "negative_prompt_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "provider": {"type": "string", "minLength": 1},
            "model_id": {"type": "string", "minLength": 1},
            "api_base": {"type": "string", "minLength": 1},
            "seed": {"type": "integer", "minimum": 0, "maximum": 4294967295},
            "image_parameters": {"type": "object", "minProperties": 1},
            "request_timestamp": {"type": ["string", "null"]},
            "response_metadata": {"type": ["object", "null"]},
            "output_sha256": {"type": ["string", "null"], "pattern": "^[0-9a-f]{64}$"},
            "retry_reason": {"type": ["string", "null"]},
            "moderation_or_error_state": {"type": ["object", "null"]},
        },
        "x_append_only_policy": {
            "failure_records_must_be_retained": True,
            "records_may_be_updated_in_place": False,
            "retry_creates_new_attempt_index": True,
            "successful_output_is_immutable": True,
        },
        "x_execution_count_in_this_task": 0,
        "x_paper_final": 0,
    }


def screening_contract() -> dict[str, Any]:
    dimensions = [
        "identity_preservation",
        "garment_semantic_match",
        "full_body_completeness",
        "hands_and_feet_completeness",
        "background_compliance",
        "blur",
        "anatomy",
        "multi_person",
        "text_or_watermark",
        "view_consistency",
        "cross_view_garment_consistency",
    ]
    return seal(
        {
            "schema_version": "multi_identity.screening_contract.v1",
            "task_id": TASK_ID,
            "status": "FROZEN_PROSPECTIVE_PROTOCOL",
            "dimensions": [{"name": name, "required": True, "hard_reject_on_fail": True} for name in dimensions],
            "decision_policy": {
                "automatic_hard_rejection_allowed": True,
                "automatic_acceptance_allowed": False,
                "vlm_only_automatic_acceptance": False,
                "single_score_automatic_acceptance": False,
                "minimum_independent_human_reviewers": 2,
                "acceptance_requires": [
                    "all hard gates pass",
                    "both human reviewers accept",
                    "identity and garment semantics pass per view",
                    "cross-view consistency passes for all eight slots",
                ],
                "disagreement_resolution": "third human reviewer; majority decision with reasons retained",
            },
            "record_policy": {
                "candidate_records_append_only": True,
                "failed_screening_records_retained": True,
                "review_history_append_only": True,
                "rejected_output_may_become_teacher_input": False,
            },
            "accepted_endpoint_budget": {
                "per_garment": 8,
                "per_identity": 40,
                "garment_count": 5,
                "semantic_pose_slot_count": 8,
            },
            "screened_candidates_in_this_task": 0,
            "paper_final": 0,
        }
    )


def benchmark_summary(adapter: dict[str, Any], split: dict[str, Any]) -> dict[str, Any]:
    fairness = {
        "identities": ["subject02", "subject00", "avatarrex_lbn1"],
        "shared": {
            "garment_ids": GARMENT_IDS,
            "condition_slot_count": 8,
            "reference_count_per_garment": 4,
            "endpoint_count_per_garment": 8,
            "endpoint_teacher_optimizer_steps_per_garment": 1200,
            "screening_contract": "multi_identity.screening_contract.v1",
            "metrics": [
                "rgb_mae",
                "lpips",
                "ssim",
                "silhouette_iou",
                "boundary_f_score",
                "protected_region_mae",
                "garment_semantic_adjudication",
                "cross_view_consistency",
            ],
            "baseline_names": [
                "M0_BASE_AVATAR",
                "M1_PER_GARMENT_GAUSSIAN_OPTIMIZATION",
                "M2_GARMENT_ID_CONDITIONING",
                "M3_GLOBAL_REFERENCE",
                "M4_PROJECTION_ONLY",
                "M5_FULL_CANONDRESSGS",
            ],
        },
        "must_report_as_different": [
            "body topology",
            "template mode",
            "raw dataset",
            "camera count",
            "base avatar quality",
            "garment representation capacity",
        ],
        "subject00_non_equivalence": "body-surface fallback is not equivalent to subject02 loose-clothing template",
    }
    return seal(
        {
            "schema_version": "multi_identity.benchmark_foundations_final_summary.v1",
            "task_id": TASK_ID,
            "source": {
                "branch": "research/multi-identity-garment-benchmark-foundations-20260724",
                "source_head": SOURCE_HEAD,
                "subject00_formal_head": SUBJECT00_FORMAL_HEAD,
                "subject00_medium_head": SUBJECT00_MEDIUM_HEAD,
            },
            "subject02_garment_provenance_status": "LIMITED_GARMENT_GENERATION_PROVENANCE",
            "wardrobe": {"garment_ids": GARMENT_IDS, "garment_count": 5},
            "condition_protocol": {"semantic_pose_slot_count": 8, "reference_count_per_garment": 4},
            "avatarrex": {
                "adapter_status": adapter["status"],
                "loader_smoke_status": adapter["loader_smoke"]["status"],
                "camera_count": split["camera_split"]["camera_count"],
                "train_camera_count": split["camera_split"]["train_count"],
                "heldout_camera_count": split["camera_split"]["heldout_count"],
                "heldout_camera_ids": split["camera_split"]["heldout_camera_ids"],
                "train_pose_count": split["pose_split"]["train_count"],
                "heldout_pose_count": split["pose_split"]["heldout_count"],
                "buffer_pose_count": split["pose_split"]["buffer_excluded_count"],
                "split_content_sha256": split["split_content_sha256"],
                "raw_integrity_verification": {
                    "verification_timing": "REHASHED_AFTER_ZERO_COPY_LOADER_SMOKE",
                    "archive_bytes": 12569755256,
                    "archive_sha256": "531bd1c71ad9b35f6ae0e2595ee531aa7ba1f83c242f18f2d0505b2dcd5fbcc1",
                    "raw_file_count": 60834,
                    "raw_apparent_bytes": 19135049684,
                    "raw_full_content_fingerprint": "00482b7c98f6f46773fd13a3f33ebe278b9353e09fdb51fa9ed72583f7b27b15",
                    "calibration_sha256": "793281a00b808976122c0d33b4dd22d0ed0a48518577345c709db6cbb3d7f315",
                    "smpl_params_sha256": "6ed3b3877d3895999e2636990bd417783328d695412b45d0679773e34b54613b",
                    "all_frozen_values_match": True,
                    "raw_data_mutations": 0,
                },
            },
            "fairness_contract": fairness,
            "claim_boundary": {
                "allowed": "identity-conditioned synthetic closed-wardrobe garment endpoint bank foundations",
                "forbidden": [
                    "real captured multi-outfit dataset",
                    "captured multi-outfit dataset",
                    "true cross-identity garment ground truth",
                    "AVATARREX MULTI-GARMENT DATASET READY",
                ],
            },
            "per_identity_readiness": {
                "subject00": ["SUBJECT00_GARMENT_BANK_PREPARATION_READY_PENDING_FORMAL_BASE"],
                "avatarrex_lbn1": [
                    "AVATARREX_ZERO_COPY_STANDARDIZATION_READY",
                    "AVATARREX_BASE_AVATAR_PREPARATION_REQUIRED",
                ],
            },
            "overall_classification": "MULTI_IDENTITY_GARMENT_BENCHMARK_FOUNDATIONS_READY",
            "execution_counts": {
                "image_generation_api_calls": 0,
                "generated_images": 0,
                "avatar_training_runs": 0,
                "garment_teacher_training_runs": 0,
                "renderer_formal_runs": 0,
                "full_rgb_copies": 0,
                "full_mask_copies": 0,
                "raw_data_mutations": 0,
                "paper_final": 0,
            },
            "paper_final": 0,
        }
    )


def handoff(summary: dict[str, Any]) -> dict[str, Any]:
    return seal(
        {
            "schema_version": "multi_identity.garment_foundations_handoff.v1",
            "task_id": TASK_ID,
            "status": summary["overall_classification"],
            "source_head": SOURCE_HEAD,
            "summary": "paper_protocol/datasets/multi_identity_benchmark_final_summary.json",
            "control_center_modified": False,
            "next_tasks_started": False,
            "next_tasks": [
                "RUN_SUBJECT00_FORMAL_STRICT_SPLIT_TRAINING_FROM_FROZEN_PROTOCOL",
                "CONFIRM_AVATARREX_UPSTREAM_LICENSE_FOR_DERIVED_GENERATION",
                "RUN_AVATARREX_BASE_AVATAR_PREFLIGHT_CANARY_AND_FORMAL_PREPARATION",
                "FREEZE_IDENTITY_SPECIFIC_CONDITION_IMAGES_AND_IDENTITY_AUDITS",
                "PIN_GENERATION_PROVIDER_MODEL_REVISION_AND_ALL_PARAMETERS",
                "GENERATE_APPEND_ONLY_MULTI_GARMENT_CANDIDATE_BANKS",
                "RUN_TWO_REVIEWER_SCREENING_AND_CROSS_VIEW_AUDIT",
                "TRAIN_MATCHED_1200_STEP_GARMENT_TEACHERS_AFTER_ACCEPTANCE",
            ],
            "blocking_dependencies": {
                "subject00": "formal final checkpoint at step 101245 does not exist",
                "avatarrex_lbn1": "license confirmation and base avatar preparation are required",
            },
            "execution_counts": summary["execution_counts"],
            "raw_integrity_verification": summary["avatarrex"]["raw_integrity_verification"],
            "paper_final": 0,
        }
    )


def build_reports(adapter: dict[str, Any], split: dict[str, Any]) -> None:
    heldout = ", ".join(split["camera_split"]["heldout_camera_ids"])
    reports = {
        "docs/DATASET/MULTI_IDENTITY_GARMENT_BENCHMARK_PLAN_20260724.md": f"""
# Multi-Identity Garment Benchmark Foundation Plan

Task: `{TASK_ID}`

## Scope

This task freezes foundations for identity-conditioned synthetic closed-wardrobe garment endpoint banks for THuman4.0 `subject00` and AvatarReX `avatarrex_lbn1`. It does not create a captured multi-outfit dataset, cross-identity garment ground truth, generated images, avatars, garment teachers, or `PAPER_FINAL` evidence.

The shared wardrobe is exactly `O01/O02/O03/O04/O08`. Each identity uses eight semantic pose/view slots, four cardinal garment references, eight accepted endpoints per garment, and a matched 1200-step teacher budget per garment. Raw frame IDs are identity-specific deterministic assignments and are never forced to match across identities.

## Gates

Subject00 is blocked on its formal step-101245 final checkpoint. AvatarReX is blocked on upstream license confirmation and base-avatar preparation after zero-copy standardization. Generation requires a pinned provider, model revision, complete parameters, deterministic seed, append-only provenance, and two-reviewer screening.

Overall classification: `MULTI_IDENTITY_GARMENT_BENCHMARK_FOUNDATIONS_READY`. This classification covers protocol and adapter readiness only.
""",
        "docs/DATASET/SUBJECT00_GARMENT_BANK_PREPARATION_20260724.md": f"""
# Subject00 Garment Bank Preparation

Subject00 has surface-attached LBS runtime PASS, short canary PASS, medium pilot `SUBJECT00_MMLPHUMAN_MEDIUM_PILOT_PASS`, and formal protocol `SUBJECT00_FORMAL_STRICT_SPLIT_PROTOCOL_READY` at `{SUBJECT00_FORMAL_HEAD}`.

The formal 101245-step run has not started and its final checkpoint does not exist. Formal condition images must be rendered only from that future final checkpoint. The medium checkpoint is `PREVIEW_ONLY`; preview generation count remains zero and a preview can never be promoted directly to formal data.

Readiness: `SUBJECT00_GARMENT_BANK_PREPARATION_READY_PENDING_FORMAL_BASE`.
""",
        "docs/DATASET/AVATARREX_ZERO_COPY_STANDARDIZATION_20260724.md": f"""
# AvatarReX Zero-Copy Standardization

The adapter uses `READ_ONLY_ZERO_COPY_LOADER_ADAPTER` over `avatarrex_lbn1`. It resolves 16 camera directories, 1901 RGB frames and 1901 `mask/pha/*.jpg` masks per camera, reads calibration and SMPL-X arrays in memory, and decoded only nine smoke records. Loader smoke status is `{adapter['loader_smoke']['status']}`.

Camera centers use `C=-R^T T`; cameras are centered on the rig mean and sorted by `(azimuth,camera_index)`. Held-out ranks 0/4/8/12 yield `{heldout}`. The pose split uses seed 20260723, standardized rotation-6D descriptors, 95 farthest-point held-out frames, at least 11-frame held-out spacing, and radius-5 buffers. Counts are {split['pose_split']['train_count']} train, {split['pose_split']['heldout_count']} held-out, and {split['pose_split']['buffer_excluded_count']} excluded buffer frames with zero overlap.

No RGB, masks, calibration, or SMPL-X data were copied or rewritten. Upstream license confirmation is required before redistribution or derived generation.

After the loader smoke, all 60834 raw files (19135049684 bytes) were rehashed. The full-content tree fingerprint remained `00482b7c98f6f46773fd13a3f33ebe278b9353e09fdb51fa9ed72583f7b27b15`; the retained archive and both metadata files also matched their frozen SHA256 values.

Readiness: `AVATARREX_ZERO_COPY_STANDARDIZATION_READY`; base-avatar status: `AVATARREX_BASE_AVATAR_PREPARATION_REQUIRED`.
""",
        "docs/DATASET/CROSS_IDENTITY_WARDROBE_FAIRNESS_20260724.md": """
# Cross-Identity Wardrobe Fairness

Subject02, subject00, and AvatarReX share exactly five garment semantics, eight condition slots, four references per garment, eight endpoints per garment, identical screening dimensions, a 1200-step teacher budget per garment, metrics, and baseline names.

They do not share faces, identity pixels, body shape, target images, or raw frame IDs. Every result must report body topology, template mode, raw dataset, camera count, base-avatar quality, and garment representation capacity separately. Subject00's body-surface fallback is not equivalent to subject02's loose-clothing template.
""",
        "docs/DATASET/SYNTHETIC_GARMENT_DATA_CLAIM_BOUNDARY_20260724.md": """
# Synthetic Garment Data Claim Boundary

The permitted object is an **identity-conditioned synthetic closed-wardrobe garment endpoint bank**. These foundations do not establish a real captured multi-outfit dataset, a captured multi-outfit benchmark, true cross-identity garment ground truth, arbitrary garment synthesis, or cross-identity generalization.

Subject02 pixels and targets are not transferable ground truth for another identity. Historical Subject02 generation is `LIMITED_GARMENT_GENERATION_PROVENANCE`; its known platform route does not expose the backend model, seed, or complete parameters. New identities therefore require a prospective reproducible contract and append-only success, failure, moderation, retry, and screening records.

`MULTI_IDENTITY_GARMENT_BENCHMARK_FOUNDATIONS_READY` means only that protocols and the zero-copy adapter are ready. It does not mean garment images, AvatarReX multi-garment data, trained avatars, teachers, or `PAPER_FINAL` evidence exist.
""",
    }
    for path, content in reports.items():
        write_text(path, content)


def build_skeletons() -> None:
    for identity in ("subject00", "avatarrex_lbn1"):
        root = ROOT / "multi_garment_benchmark" / identity
        root.mkdir(parents=True, exist_ok=True)
        write_text(
            f"multi_garment_benchmark/{identity}/README.md",
            f"""
# {identity} Multi-Garment Benchmark Skeleton

This tracked tree contains contracts and placeholders only. It contains no generated candidate, accepted endpoint, copied RGB/mask, trained checkpoint, or formal renderer output. Large identity data remains external and is referenced by immutable pointers and hashes.
""",
        )
        for name in SKELETON_DIRS:
            directory = root / name
            directory.mkdir(parents=True, exist_ok=True)
            (directory / ".gitkeep").write_text("", encoding="ascii")


def main() -> int:
    dataset_root = ROOT / "paper_protocol" / "datasets"
    adapter = json.loads((dataset_root / "avatarrex_zero_copy_adapter_contract.json").read_text(encoding="utf-8"))
    split = json.loads((dataset_root / "avatarrex_camera_pose_split.json").read_text(encoding="utf-8"))
    if adapter["status"] != "PASS" or split["status"] != "PASS":
        raise RuntimeError("AvatarReX adapter and split must pass before foundation artifacts are built")

    wardrobe_payload = wardrobe()
    condition_payload = condition_protocol()
    subject00_payload = subject00_manifest()
    provenance_payload = provenance_schema()
    screening_payload = screening_contract()
    summary_payload = benchmark_summary(adapter, split)

    write_json("paper_protocol/datasets/shared_five_garment_wardrobe.json", wardrobe_payload)
    write_json("paper_protocol/datasets/shared_condition_pose_view_protocol.json", condition_payload)
    write_json("paper_protocol/datasets/subject00_garment_preparation_manifest.json", subject00_payload)
    write_json("paper_protocol/datasets/multi_identity_generation_provenance_schema.json", provenance_payload)
    write_json("paper_protocol/datasets/multi_identity_screening_contract.json", screening_payload)
    write_json("paper_protocol/datasets/multi_identity_benchmark_final_summary.json", summary_payload)
    write_json("project_control_handoff/multi_identity_garment_foundations_handoff.json", handoff(summary_payload))
    build_reports(adapter, split)
    build_skeletons()
    print(json.dumps({"status": summary_payload["overall_classification"], "generated_images": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
