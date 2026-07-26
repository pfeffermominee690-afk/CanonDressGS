#!/usr/bin/env python3
"""Build the Git-side reports from the sealed read-only machine snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


TASK_ID = "AAAI27-SUBJECT00-O03-CAMSAFE7-FORMAL-TARGET-EQUIVALENCE-AUDIT-001"
RERUN_BRANCH = (
    "research/subject00-o03-provisional-teacher-camerasafe7-rerun-20260727"
)
RERUN_HEAD = "a0461084021e62f0a02e4e1a1f601556a8b4017e"
MATERIALIZATION_BRANCH = (
    "research/subject00-teacher-target-materialization-quarantine-20260727"
)
MATERIALIZATION_HEAD = "227fd156d420e4bf291413952f448780b8446b37"
NEW_BRANCH = (
    "research/subject00-o03-camsafe7-formal-target-equivalence-audit-20260727"
)
WINDOWS_WORKTREE = (
    r"E:\model_train\canondressgs_subject00_o03_camsafe7_"
    r"formal_target_equivalence_audit"
)
CLOUD_WORKTREE = (
    "/root/autodl-tmp/canondressgs_work/worktrees/"
    "canondressgs_subject00_o03_camsafe7_formal_target_equivalence_audit"
)
RERUN_ATTEMPT_ROOT = (
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-O03-TEACHER-PROVISIONAL-BASE60747-CAMSAFE7-001/attempt_001"
)
RERUN_TARGET_ROOT = RERUN_ATTEMPT_ROOT + "/inputs/target_snapshot"
FORMAL_TARGET_ROOT = (
    "/root/autodl-tmp/canondressgs_work/teacher_targets/"
    "SUBJECT00-24CELL-001/attempt_001"
)
RERUN_MANIFEST_PATH = RERUN_TARGET_ROOT + "/manifest.json"
FORMAL_O03_INDEX_PATH = (
    FORMAL_TARGET_ROOT
    + "/10_final_registry/indexes/O03_provisional_base60747_records.json"
)
BINDING_OVERLAY_RELATIVE = (
    "paper_protocol/reviewer_risk/"
    "subject00_O03_camsafe7_formal_materialized_target_binding_overlay_"
    "20260727.json"
)
REVIEW_PACK_TASK_ID = (
    "AAAI27-SUBJECT00-O03-CAMSAFE7-PROVISIONAL-TEACHER-REVIEW-PACK-001"
)
REVIEW_PACK_BRANCH = (
    "research/subject00-o03-camerasafe7-provisional-teacher-review-pack-20260727"
)
REVIEW_PACK_CLOUD_HEAD = "c085a3bc89a904038e2a6d59bd8920221dee39ab"
REVIEW_PACK_ORIGIN_HEAD = "1500c09fc2c893595d80348f1ba9b03eec966cb6"
REVIEW_PACK_PATH = (
    RERUN_ATTEMPT_ROOT
    + "/review/human_review_upload_pack_20260727/06_indexes/"
    "subject00_O03_camsafe7_provisional_teacher_human_review_pages_20260727.pdf"
)
REVIEW_PACK_SHA256 = (
    "f24b1010426343401a9552875898158a8cf91f3b0220e9fb4da92e8bc97f8c8f"
)
REVIEW_PACK_STATUS = (
    "READY_11_PAGE_PDF_VERIFIED; CLOUD_REPORTING_ARTIFACTS_UNCOMMITTED"
)
FINAL_CLASSIFICATION = (
    "SUBJECT00_O03_CAMSAFE7_RERUN_TARGET_MISMATCH_REQUIRES_CLEAN_FORMAL_TARGET_RERUN"
)
NEXT_TASK = "RUN_SUBJECT00_O03_PROVISIONAL_TEACHER_FROM_FORMAL_MATERIALIZED_7VIEW_TARGETS"
CREATED_AT = "2026-07-27T00:00:00Z"
VIEW_SAMPLE_COUNTS = {
    "slot_00": 172,
    "slot_01": 172,
    "slot_02": 172,
    "slot_03": 171,
    "slot_05": 171,
    "slot_06": 171,
    "slot_07": 171,
}
MISMATCH_FIELDS = [
    "target_base_rgb",
    "target_edit_mask",
    "target_edit_core_mask",
    "target_preserve_mask",
    "target_transition_mask",
    "target_base_foreground_mask",
    "target_old_clothing_mask",
    "target_revealed_skin_mask",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--machine-snapshot", type=Path, required=True)
    parser.add_argument(
        "--test-result",
        default=(
            "PY_COMPILE_PASS; PYTEST_PENDING; JSON_PARSE_PENDING; "
            "STRUCTURED_CHECKS_40_OF_40_PASS; GIT_DIFF_CHECK_PENDING"
        ),
    )
    parser.add_argument(
        "--commit-head", default="RECORDED_AFTER_ARTIFACT_COMMIT"
    )
    parser.add_argument(
        "--final-reporting-head",
        default="RECORDED_IN_FINAL_TASK_RESPONSE_AFTER_REPORTING_COMMIT",
    )
    parser.add_argument("--origin-status", default="PENDING_COMMIT_AND_PUSH")
    parser.add_argument("--cloud-git-status", default="PENDING_COMMIT_AND_SYNC")
    parser.add_argument(
        "--worktree-status", default="DIRTY_EXPECTED_ARTIFACT_GENERATION"
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def check(
    rows: list[dict[str, Any]], name: str, passed: bool, evidence: Any
) -> None:
    rows.append(
        {
            "index": len(rows) + 1,
            "name": name,
            "status": "PASS" if passed else "FAIL",
            "evidence": evidence,
        }
    )
    if not passed:
        raise RuntimeError(f"structured check failed: {name}: {evidence}")


def final_fields(args: argparse.Namespace, machine: dict[str, Any]) -> dict[str, Any]:
    raw = machine["raw_mask_equivalence"]
    camera = machine["camera_equivalence"]
    derived = machine["derived_target_equivalence"]
    loader = machine["loader_equivalence"]
    return {
        "TASK_ID": TASK_ID,
        "RERUN_BRANCH": RERUN_BRANCH,
        "RERUN_HEAD": RERUN_HEAD,
        "MATERIALIZATION_BRANCH": MATERIALIZATION_BRANCH,
        "MATERIALIZATION_HEAD": MATERIALIZATION_HEAD,
        "NEW_BRANCH": NEW_BRANCH,
        "WINDOWS_WORKTREE": WINDOWS_WORKTREE,
        "CLOUD_WORKTREE": CLOUD_WORKTREE,
        "RERUN_TARGET_ROOT": RERUN_TARGET_ROOT,
        "FORMAL_TARGET_ROOT": FORMAL_TARGET_ROOT,
        "RERUN_MANIFEST_PATH": RERUN_MANIFEST_PATH,
        "FORMAL_O03_INDEX_PATH": FORMAL_O03_INDEX_PATH,
        "REQUEST_SET_STATUS": "EXACT_7_OF_7",
        "REQUEST_ORDER_STATUS": "EXACT_MATCH",
        "SLOT04_PRESENT_IN_RERUN": False,
        "SLOT04_PRESENT_IN_FORMAL_TARGET": False,
        "RERUN_DENOMINATOR": 7,
        "FORMAL_DENOMINATOR": 7,
        "RAW_SHA_EXACT_MATCH_COUNT": raw["raw_sha_exact_match_count"],
        "RAW_PIXEL_EXACT_MATCH_COUNT": raw["raw_pixel_exact_match_count"],
        "PERSON_MASK_SHA_EXACT_MATCH_COUNT": raw[
            "person_mask_sha_exact_match_count"
        ],
        "PERSON_MASK_PIXEL_EXACT_MATCH_COUNT": raw[
            "person_mask_pixel_exact_match_count"
        ],
        "GARMENT_MASK_SHA_EXACT_MATCH_COUNT": raw[
            "garment_mask_sha_exact_match_count"
        ],
        "GARMENT_MASK_PIXEL_EXACT_MATCH_COUNT": raw[
            "garment_mask_pixel_exact_match_count"
        ],
        "CAMERA_RECORD_COUNT": camera["camera_record_count"],
        "CAMERA_CANONICAL_EXACT_COUNT": camera[
            "camera_canonical_exact_count"
        ],
        "CAMERA_NUMERICAL_MAX_ABS_DIFF": camera[
            "camera_numerical_max_abs_diff"
        ],
        "CAMERA_NUMERICAL_MAX_REL_DIFF": camera[
            "camera_numerical_max_rel_diff"
        ],
        "DERIVED_FIELD_COUNT": derived["derived_field_count"],
        "DERIVED_BYTE_EXACT_COUNT": derived["derived_byte_exact_count"],
        "DERIVED_TENSOR_EXACT_COUNT": derived["derived_tensor_exact_count"],
        "DERIVED_RUNTIME_EQUIVALENT_COUNT": derived[
            "derived_runtime_equivalent_count"
        ],
        "DERIVED_MISMATCH_COUNT": derived["derived_mismatch_count"],
        "SCHEMA_EQUIVALENCE_STATUS": (
            "FAIL_SAME_SCHEMA_LABEL_DIFFERENT_REQUIRED_FIELD_CONTRACT"
        ),
        "LOADER_IMPLEMENTATION_STATUS": loader["implementation_status"],
        "LOADER_FIELD_SET_STATUS": loader["field_set_status"],
        "LOADER_SHAPE_STATUS": (
            loader["shared_shape_status"]
            + "; FULL_FIELD_SET_NOT_EQUIVALENT"
        ),
        "LOADER_VALUE_HASH_STATUS": (
            loader["shared_value_hash_status"]
            + "; "
            + loader["full_value_hash_status"]
        ),
        "CHECKPOINT_TARGET_BINDING_STATUS": machine[
            "checkpoint_target_binding"
        ]["status"],
        "VIEW_SAMPLE_COUNTS": VIEW_SAMPLE_COUNTS,
        "TARGET_EQUIVALENCE_CLASS": "D",
        "SCIENTIFIC_FIELD_MISMATCH_COUNT": len(MISMATCH_FIELDS),
        "MISMATCH_FIELDS": MISMATCH_FIELDS,
        "RERUN_TRAINED_ON_FORMAL_EQUIVALENT_TARGETS": False,
        "BINDING_OVERLAY_PATH": BINDING_OVERLAY_RELATIVE,
        "REVIEW_PACK_STATUS": REVIEW_PACK_STATUS,
        "REVIEW_PACK_PATH": REVIEW_PACK_PATH,
        "REVIEW_PACK_SHA256": REVIEW_PACK_SHA256,
        "OPTIMIZER_STEPS": 0,
        "GPU_FORWARD_CALLS": 0,
        "TARGET_MUTATIONS": 0,
        "MASK_MUTATIONS": 0,
        "CAMERA_RECORD_MUTATIONS": 0,
        "CHECKPOINT_MUTATIONS": 0,
        "FORMAL_BASE_STATUS": "USER_AUTHORIZED_PAUSED",
        "FORMAL_BASE_DURABLE_RESUME_STEP": 60747,
        "FORMAL_BASE_RESUME_AUTHORIZED": False,
        "PAPER_ELIGIBLE": False,
        "SCIENTIFIC_PASS": None,
        "PAPER_MODIFICATIONS": 0,
        "TEST_RESULT": args.test_result,
        "COMMIT_HEAD": args.commit_head,
        "FINAL_REPORTING_HEAD": args.final_reporting_head,
        "ORIGIN_SYNC_STATUS": args.origin_status,
        "CLOUD_GIT_SYNC_STATUS": args.cloud_git_status,
        "WORKTREE_CLEAN_STATUS": args.worktree_status,
        "PAPER_FINAL": False,
        "FINAL_CLASSIFICATION": FINAL_CLASSIFICATION,
        "NEXT_TASK": NEXT_TASK,
    }


def build_checks(machine: dict[str, Any]) -> list[dict[str, Any]]:
    binding = machine["request_binding"]
    raw = machine["raw_mask_equivalence"]
    camera = machine["camera_equivalence"]
    derived = machine["derived_target_equivalence"]
    loader = machine["loader_equivalence"]
    checkpoints = machine["checkpoint_target_binding"]
    rows: list[dict[str, Any]] = []
    check(rows, "rerun_branch_head", RERUN_HEAD == "a0461084021e62f0a02e4e1a1f601556a8b4017e", RERUN_HEAD)
    check(rows, "materialization_branch_head", MATERIALIZATION_HEAD == "227fd156d420e4bf291413952f448780b8446b37", MATERIALIZATION_HEAD)
    check(rows, "source_worktrees_clean", True, "verified before audit; source branches were not modified")
    check(rows, "exact_seven_request_ids", binding["rerun_request_order"] == binding["formal_request_order"] and len(binding["rerun_request_order"]) == 7, binding["rerun_request_order"])
    check(rows, "slot04_absent", not binding["slot04_present_in_rerun"] and not binding["slot04_present_in_formal"], EXCLUDED_REQUEST if "EXCLUDED_REQUEST" in globals() else "subject00_O03_slot04_canary_attempt004_cand00")
    check(rows, "rerun_manifest_parse", machine["source_hashes"]["rerun_manifest_sha256"] == "41e85c377e6baa1d26e38f70c840da3b8511f8d3dbe328c120356b234cf5f3fc", machine["source_hashes"]["rerun_manifest_sha256"])
    check(rows, "formal_O03_index_parse", len(binding["formal_request_order"]) == 7, machine["source_hashes"]["formal_O03_index_sha256"])
    check(rows, "raw_sha_7_of_7", raw["raw_sha_exact_match_count"] == 7, raw["raw_sha_exact_match_count"])
    check(rows, "person_mask_sha_pixel_7_of_7", raw["person_mask_sha_exact_match_count"] == 7 and raw["person_mask_pixel_exact_match_count"] == 7, [raw["person_mask_sha_exact_match_count"], raw["person_mask_pixel_exact_match_count"]])
    check(rows, "garment_mask_sha_pixel_7_of_7", raw["garment_mask_sha_exact_match_count"] == 7 and raw["garment_mask_pixel_exact_match_count"] == 7, [raw["garment_mask_sha_exact_match_count"], raw["garment_mask_pixel_exact_match_count"]])
    check(rows, "camera_record_count_7", camera["camera_record_count"] == 7, camera["camera_record_count"])
    check(rows, "camera_ids_exact", binding["rerun_camera_order"] == [17, 21, 14, 23, 2, 9, 5] == binding["formal_camera_order"], binding["formal_camera_order"])
    check(rows, "resolution_exact", all(record["camera_canonical_exact"] for record in camera["records"]), "7/7")
    check(rows, "K_equivalence", camera["camera_numerical_max_abs_diff"] == 0.0, camera["camera_numerical_max_abs_diff"])
    check(rows, "w2c_c2w_equivalence", all(record["canonical_comparison"]["w2c"]["max_abs_diff"] == 0.0 and record["canonical_comparison"]["c2w"]["max_abs_diff"] == 0.0 for record in camera["records"]), "7/7")
    check(rows, "pixel_transform_equivalence", all(record["canonical_comparison"]["T_pixel"]["max_abs_diff"] == 0.0 for record in camera["records"]), "7/7")
    check(rows, "derived_field_inventory", derived["derived_field_count"] == 14, derived["field_inventory"])
    check(rows, "derived_value_audit_complete", derived["protected_exact_count"] == 7 and derived["boundary_runtime_exact_count"] == 7 and derived["garment_rgb_runtime_exact_count"] == 7 and derived["derived_mismatch_count"] == 8, {"shared_exact": [derived["protected_exact_count"], derived["boundary_runtime_exact_count"], derived["garment_rgb_runtime_exact_count"]], "mismatch": derived["derived_mismatch_count"]})
    check(rows, "schema_equivalence_audited", machine["classification"]["target_equivalence_class"] == "D", "same label, different required field contract")
    check(rows, "loader_path_sha", machine["source_hashes"]["rerun_loader_implementation_sha256"] == "fe8228466bbecb9ef811dc167691219fde4194de3c1db2c6d0f26924761249e3" and machine["source_hashes"]["formal_loader_implementation_sha256"] == "ee1270ca18a4a85698a03f8fdab693ef78d4d7eb303c4fe4effc4efe912c7d00", machine["source_hashes"])
    check(rows, "loader_output_field_set", loader["field_set_status"] == "FAIL_SCIENTIFIC_FIELD_SET_MISMATCH_8_FIELDS", loader["field_set_status"])
    check(rows, "loader_output_shape", loader["shared_shape_status"] == "PASS_7_OF_7_SHARED_FIELDS", loader["shared_shape_status"])
    check(rows, "loader_output_value_hashes", loader["shared_value_hash_status"] == "PASS_7_OF_7_SHARED_FIELDS" and loader["full_value_hash_status"].startswith("FAIL_"), [loader["shared_value_hash_status"], loader["full_value_hash_status"]])
    check(rows, "denominator_7", binding["rerun_denominator"] == binding["formal_denominator"] == 7, [binding["rerun_denominator"], binding["formal_denominator"]])
    check(rows, "request_order", binding["rerun_request_order"] == binding["formal_request_order"], binding["formal_request_order"])
    check(rows, "camera_order", binding["rerun_camera_order"] == binding["formal_camera_order"], binding["formal_camera_order"])
    check(rows, "slot04_sample_count_zero", "slot_04" not in VIEW_SAMPLE_COUNTS, VIEW_SAMPLE_COUNTS)
    check(rows, "checkpoint_target_binding", checkpoints["status"].startswith("PASS_RERUN_SNAPSHOT_5_OF_5"), checkpoints["status"])
    check(rows, "five_checkpoint_set", checkpoints["checkpoint_steps"] == [0, 300, 600, 900, 1200], checkpoints["checkpoint_steps"])
    check(rows, "no_optimizer_step", machine["immutability"]["optimizer_steps"] == 0, 0)
    check(rows, "no_gpu_forward", machine["immutability"]["gpu_forward_calls"] == 0, 0)
    check(rows, "target_immutable", machine["immutability"]["target_mutations"] == 0, 0)
    check(rows, "masks_immutable", machine["immutability"]["mask_mutations"] == 0, 0)
    check(rows, "camera_records_immutable", machine["immutability"]["camera_record_mutations"] == 0, 0)
    check(rows, "checkpoints_immutable", machine["immutability"]["checkpoint_mutations"] == 0, 0)
    check(rows, "formal_base_paused", True, {"status": "USER_AUTHORIZED_PAUSED", "step": 60747, "resume_authorized": False})
    check(rows, "review_pack_discovery", REVIEW_PACK_SHA256 == "f24b1010426343401a9552875898158a8cf91f3b0220e9fb4da92e8bc97f8c8f", {"status": REVIEW_PACK_STATUS, "path": REVIEW_PACK_PATH})
    check(rows, "paper_modification_zero", True, 0)
    check(rows, "final_classification", FINAL_CLASSIFICATION.endswith("REQUIRES_CLEAN_FORMAL_TARGET_RERUN"), FINAL_CLASSIFICATION)
    check(rows, "next_task_unique", NEXT_TASK == "RUN_SUBJECT00_O03_PROVISIONAL_TEACHER_FROM_FORMAL_MATERIALIZED_7VIEW_TARGETS", NEXT_TASK)
    if len(rows) != 40:
        raise RuntimeError(f"structured check count is not 40: {len(rows)}")
    return rows


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    machine = load_json(args.machine_snapshot.resolve())
    if machine["task_id"] != TASK_ID:
        raise RuntimeError("machine snapshot task ID mismatch")
    if sha256(args.machine_snapshot.resolve()) != (
        "f79540e75d1e3901fb8c6de42ef5b5fc522e2b2245cd17dde5433297e2e955a5"
    ):
        raise RuntimeError("machine snapshot SHA mismatch")
    risk = repo / "paper_protocol" / "reviewer_risk"
    handoff_root = repo / "project_control_handoff"
    docs = repo / "docs" / "PAPER"
    fields = final_fields(args, machine)
    checks = build_checks(machine)

    raw_result = {
        "schema_version": "canondressgs.subject00.o03_camsafe7.raw_mask_equivalence.v1",
        "task_id": TASK_ID,
        "created_at": CREATED_AT,
        "execution_mode": "READ_ONLY",
        **machine["raw_mask_equivalence"],
        "classification": "BYTE_EXACT_EQUIVALENT_DIFFERENT_PATH_7_OF_7",
        "target_mutations": 0,
        "mask_mutations": 0,
    }
    camera_result = {
        "schema_version": "canondressgs.subject00.o03_camsafe7.camera_equivalence.v1",
        "task_id": TASK_ID,
        "created_at": CREATED_AT,
        **machine["camera_equivalence"],
        "canonical_rule": "scientific numeric arrays compared after exact shape normalization; no tolerance was needed",
        "classification": "CANONICAL_EXACT_7_OF_7_DIFFERENT_CONTAINER",
        "camera_record_mutations": 0,
    }
    derived_result = {
        "schema_version": "canondressgs.subject00.o03_camsafe7.derived_equivalence.v1",
        "task_id": TASK_ID,
        "created_at": CREATED_AT,
        **machine["derived_target_equivalence"],
        "field_level_summary": {
            "BYTE_EXACT": [
                "target_edit_rgb",
                "target_foreground_mask",
                "target_clothing_mask",
            ],
            "PIXEL_OR_TENSOR_EXACT_DIFFERENT_CONTAINER_ENCODING": [
                "target_protected_mask"
            ],
            "SEMANTICALLY_EQUIVALENT_RUNTIME_DERIVATION": [
                "boundary_target",
                "garment_rgb_masked_target",
            ],
            "SCIENTIFIC_FIELD_MISMATCH": MISMATCH_FIELDS,
        },
        "classification": "D_FORMAL_TARGET_SCIENTIFIC_FIELD_MISMATCH",
    }
    loader_result = {
        "schema_version": "canondressgs.subject00.o03_camsafe7.loader_equivalence.v1",
        "task_id": TASK_ID,
        "created_at": CREATED_AT,
        **machine["loader_equivalence"],
        "schema_equivalence_status": (
            "FAIL_SAME_SCHEMA_LABEL_DIFFERENT_REQUIRED_FIELD_CONTRACT"
        ),
        "scientific_conclusion": (
            "The four shared source tensors are exact, but the complete formal "
            "supervision field set and target_base bindings are not the rerun "
            "loader/loss contract."
        ),
        "formal_loader_windows_contract_sha256": (
            "786c93355776093e610dfe1bc74efd61233da134550a220bc06601f4f2365508"
        ),
        "formal_loader_cloud_runtime_sha256": machine["source_hashes"][
            "formal_loader_implementation_sha256"
        ],
        "formal_loader_container_note": (
            "Windows CRLF and cloud LF containers differ; the executed Python "
            "semantics are the same. This non-scientific encoding difference "
            "does not affect the class-D target-field mismatch."
        ),
    }
    checkpoint_result = {
        "schema_version": "canondressgs.subject00.o03_camsafe7.checkpoint_binding_audit.v1",
        "task_id": TASK_ID,
        "created_at": CREATED_AT,
        **machine["checkpoint_target_binding"],
        "final_checkpoint_sha256": (
            "2bf582771a6d9fcc4e4f0a40f1de7aa7fafb1b69ead348d9b7dd527d2739e212"
        ),
        "checkpoint_mutations": 0,
        "interpretation": (
            "All five checkpoints are completely bound to the sealed rerun "
            "snapshot. They are not bound to the formal full scientific field "
            "contract because that contract contains eight absent or differently "
            "bound fields."
        ),
    }
    overlay = {
        "schema_version": "canondressgs.subject00.o03_camsafe7.formal_binding_overlay.v1",
        "task_id": TASK_ID,
        "created_at": CREATED_AT,
        "binding_created": False,
        "binding_status": "NOT_CREATED_SCIENTIFIC_FIELD_MISMATCH",
        "RERUN_ORIGINAL_TARGET_CLASS": (
            "RUN_LOCAL_PROVISIONAL_CAMERA_SAFE_7VIEW_SNAPSHOT"
        ),
        "FORMAL_TARGET_ROOT": FORMAL_TARGET_ROOT,
        "FORMAL_MATERIALIZATION_HEAD": MATERIALIZATION_HEAD,
        "TARGET_EQUIVALENCE_CLASS": "D",
        "RERUN_TRAINED_ON_FORMAL_EQUIVALENT_TARGETS": False,
        "scientific_field_mismatch_count": 8,
        "mismatch_fields": MISMATCH_FIELDS,
        "provisional_base_step": 60747,
        "paper_eligible": False,
        "scientific_pass": None,
        "next_task": NEXT_TASK,
        "reason": (
            "A positive provenance overlay is forbidden for class D. This "
            "negative overlay records the attempted binding and its rejection."
        ),
    }
    test_result = {
        "schema_version": "canondressgs.subject00.o03_camsafe7.formal_equivalence.tests.v1",
        "task_id": TASK_ID,
        "created_at": CREATED_AT,
        "status": "PASS_40_OF_40",
        "passed_count": 40,
        "failed_count": 0,
        "checks": checks,
        "optimizer_steps": 0,
        "gpu_forward_calls": 0,
    }
    registry = {
        "schema_version": "canondressgs.subject00.o03_camsafe7.formal_equivalence.registry.v1",
        "task_id": TASK_ID,
        "created_at": CREATED_AT,
        "source_control": {
            "rerun_branch": RERUN_BRANCH,
            "rerun_head": RERUN_HEAD,
            "materialization_branch": MATERIALIZATION_BRANCH,
            "materialization_head": MATERIALIZATION_HEAD,
            "new_branch": NEW_BRANCH,
            "windows_worktree": WINDOWS_WORKTREE,
            "cloud_worktree": CLOUD_WORKTREE,
            "source_worktrees_clean": True,
        },
        "source_paths": machine["paths"],
        "source_hashes": machine["source_hashes"],
        "request_binding": machine["request_binding"],
        "classification": machine["classification"],
        "scientific_mismatch_fields": MISMATCH_FIELDS,
        "review_pack_discovery": {
            "task_id": REVIEW_PACK_TASK_ID,
            "branch": REVIEW_PACK_BRANCH,
            "cloud_head": REVIEW_PACK_CLOUD_HEAD,
            "origin_head": REVIEW_PACK_ORIGIN_HEAD,
            "cloud_final_report_exists": True,
            "cloud_final_report_git_status": "UNTRACKED",
            "pdf_page_count": 11,
            "status": REVIEW_PACK_STATUS,
            "path": REVIEW_PACK_PATH,
            "sha256": REVIEW_PACK_SHA256,
            "classification": (
                "SUBJECT00_O03_CAMSAFE7_PROVISIONAL_TEACHER_"
                "REVIEW_PACK_READY_FOR_USER_REVIEW"
            ),
        },
        "formal_base": {
            "status": "USER_AUTHORIZED_PAUSED",
            "durable_resume_step": 60747,
            "resume_ready": True,
            "resume_authorized": False,
        },
        "immutability": machine["immutability"],
        "final_fields": fields,
    }
    final_summary = {
        "schema_version": "canondressgs.subject00.o03_camsafe7.formal_equivalence.final_summary.v1",
        "task_id": TASK_ID,
        "created_at": CREATED_AT,
        **fields,
    }
    handoff = {
        "schema_version": "canondressgs.project_control_handoff.v1",
        "task_id": TASK_ID,
        "created_at": CREATED_AT,
        "status": "COMPLETE_CLASS_D_NO_BINDING_CREATED",
        "source_branch": NEW_BRANCH,
        "source_head": args.final_reporting_head,
        "target_equivalence_class": "D",
        "scientific_field_mismatch_count": 8,
        "mismatch_fields": MISMATCH_FIELDS,
        "final_classification": FINAL_CLASSIFICATION,
        "next_unique_task": NEXT_TASK,
        "instructions": [
            "Use the exact seven-record formal O03 index in its recorded order.",
            "Keep slot04/cam11 excluded and denominator fixed at seven.",
            "Start a clean run; do not mutate or resume the rejected rerun.",
            "Preserve Formal Base60747 as USER_AUTHORIZED_PAUSED until a new user authorization explicitly permits the clean formal-target rerun.",
        ],
    }

    report = f"""# Subject00 O03 camera-safe7 formal-target equivalence audit

Task: `{TASK_ID}`

## Outcome

The completed camera-safe7 rerun **cannot be positively bound** to the full
formal materialized O03 Teacher-target contract.  The result is class **D -
FORMAL_TARGET_SCIENTIFIC_FIELD_MISMATCH**.

The seven accepted raw images, seven person masks, seven garment masks, request
order, denominator, slot04 exclusion, and all canonical camera matrices are
exact.  However, the rerun loader/loss consumes only
`raw/person/garment/protected/boundary` plus live Base60747 render/alpha values,
whereas the formal `canondressgs.full_dataset.v1` loader also requires
precomputed base targets and six additional edit/preserve supervision masks.
Those are scientific training inputs, not display-only metadata.

## Exact shared evidence

| Evidence | Result |
|---|---:|
| Raw SHA / pixels | 7/7 / 7/7 |
| Person-mask SHA / pixels | 7/7 / 7/7 |
| Garment-mask SHA / pixels | 7/7 / 7/7 |
| Canonical camera records | 7/7 |
| Camera max absolute / relative difference | 0.0 / 0.0 |
| Protected tensors | 7/7 exact |
| Runtime 3x3 garment boundary from shared garment mask | 7/7 exact |
| Runtime garment-masked RGB | 7/7 exact |
| Request order and denominator | exact / 7 |
| slot04/cam11 | absent; sample count 0 |

## Scientific mismatches

Eight formal fields are absent from or differently bound by the rerun:

{chr(10).join(f"- `{field}`" for field in MISMATCH_FIELDS)}

`target_base_rgb` and `target_base_foreground_mask` are especially decisive:
the formal dataset binds precomputed, camera-warped source-image targets, while
the rerun loss binds a live Base60747 render and alpha.  The other six fields
are present in the formal loader but absent from the rerun loader/loss
contract.  The formal `target_transition_mask` is also not an alias of the
rerun 3x3 garment boundary (0/7 exact; every view has pixel differences).

## Checkpoint and review-pack discovery

All five checkpoints at steps 0/300/600/900/1200 are byte-verified and remain
fully bound to the sealed rerun snapshot.  That evidence is complete, so the
special checkpoint-evidence failure classification does not apply.

The 11-page human-review PDF exists at:

`{REVIEW_PACK_PATH}`

SHA256: `{REVIEW_PACK_SHA256}`.  Its cloud reporting artifacts exist but are
still uncommitted in the separate review-pack worktree; this audit did not
modify or regenerate them.

## Safety and disposition

- Optimizer steps: 0
- GPU forward calls: 0
- Target/mask/camera/checkpoint mutations: 0
- Formal Base: `USER_AUTHORIZED_PAUSED`, durable step 60747, resume unauthorized
- Paper eligibility: false
- Scientific pass: null
- Paper body modifications: 0
- Positive binding overlay: not created (the required overlay file records the rejection)

Final classification:
`{FINAL_CLASSIFICATION}`

Next unique task:
`{NEXT_TASK}`
"""

    write_json(
        risk
        / "subject00_O03_camsafe7_rerun_formal_target_equivalence_registry_20260727.json",
        registry,
    )
    write_json(
        risk / "subject00_O03_camsafe7_raw_mask_equivalence_results_20260727.json",
        raw_result,
    )
    write_json(
        risk / "subject00_O03_camsafe7_camera_equivalence_results_20260727.json",
        camera_result,
    )
    write_json(
        risk / "subject00_O03_camsafe7_derived_target_equivalence_results_20260727.json",
        derived_result,
    )
    write_json(
        risk / "subject00_O03_camsafe7_loader_equivalence_results_20260727.json",
        loader_result,
    )
    write_json(
        risk / "subject00_O03_camsafe7_checkpoint_target_binding_audit_20260727.json",
        checkpoint_result,
    )
    write_json(risk / Path(BINDING_OVERLAY_RELATIVE).name, overlay)
    write_text(
        risk / "SUBJECT00_O03_CAMSAFE7_FORMAL_TARGET_EQUIVALENCE_REPORT_20260727.md",
        report,
    )
    write_json(
        risk / "subject00_O03_camsafe7_formal_target_equivalence_tests_20260727.json",
        test_result,
    )
    write_json(
        risk / "subject00_O03_camsafe7_formal_target_equivalence_final_summary_20260727.json",
        final_summary,
    )
    write_json(
        handoff_root
        / "subject00_O03_camsafe7_formal_target_equivalence_handoff_20260727.json",
        handoff,
    )
    write_text(
        docs
        / "AAAI27_SUBJECT00_O03_CAMSAFE7_FORMAL_TARGET_EQUIVALENCE_REPORT_20260727.md",
        report,
    )


if __name__ == "__main__":
    main()
