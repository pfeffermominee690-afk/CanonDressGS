from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
HANDOFF = ROOT / "project_control_handoff"
TASK_ID = "AAAI27-SUBJECT00-O03-CAMSAFE7-FORMAL-TARGET-EQUIVALENCE-AUDIT-001"
FINAL_CLASSIFICATION = (
    "SUBJECT00_O03_CAMSAFE7_RERUN_TARGET_MISMATCH_REQUIRES_CLEAN_FORMAL_TARGET_RERUN"
)
NEXT_TASK = "RUN_SUBJECT00_O03_PROVISIONAL_TEACHER_FROM_FORMAL_MATERIALIZED_7VIEW_TARGETS"
JSON_PATHS = [
    RISK / "subject00_O03_camsafe7_rerun_formal_target_equivalence_registry_20260727.json",
    RISK / "subject00_O03_camsafe7_raw_mask_equivalence_results_20260727.json",
    RISK / "subject00_O03_camsafe7_camera_equivalence_results_20260727.json",
    RISK / "subject00_O03_camsafe7_derived_target_equivalence_results_20260727.json",
    RISK / "subject00_O03_camsafe7_loader_equivalence_results_20260727.json",
    RISK / "subject00_O03_camsafe7_checkpoint_target_binding_audit_20260727.json",
    RISK / "subject00_O03_camsafe7_formal_materialized_target_binding_overlay_20260727.json",
    RISK / "subject00_O03_camsafe7_formal_target_equivalence_tests_20260727.json",
    RISK / "subject00_O03_camsafe7_formal_target_equivalence_final_summary_20260727.json",
    HANDOFF / "subject00_O03_camsafe7_formal_target_equivalence_handoff_20260727.json",
]
REQUIRED_FINAL_FIELDS = [
    "TASK_ID",
    "RERUN_BRANCH",
    "RERUN_HEAD",
    "MATERIALIZATION_BRANCH",
    "MATERIALIZATION_HEAD",
    "NEW_BRANCH",
    "WINDOWS_WORKTREE",
    "CLOUD_WORKTREE",
    "RERUN_TARGET_ROOT",
    "FORMAL_TARGET_ROOT",
    "RERUN_MANIFEST_PATH",
    "FORMAL_O03_INDEX_PATH",
    "REQUEST_SET_STATUS",
    "REQUEST_ORDER_STATUS",
    "SLOT04_PRESENT_IN_RERUN",
    "SLOT04_PRESENT_IN_FORMAL_TARGET",
    "RERUN_DENOMINATOR",
    "FORMAL_DENOMINATOR",
    "RAW_SHA_EXACT_MATCH_COUNT",
    "RAW_PIXEL_EXACT_MATCH_COUNT",
    "PERSON_MASK_SHA_EXACT_MATCH_COUNT",
    "PERSON_MASK_PIXEL_EXACT_MATCH_COUNT",
    "GARMENT_MASK_SHA_EXACT_MATCH_COUNT",
    "GARMENT_MASK_PIXEL_EXACT_MATCH_COUNT",
    "CAMERA_RECORD_COUNT",
    "CAMERA_CANONICAL_EXACT_COUNT",
    "CAMERA_NUMERICAL_MAX_ABS_DIFF",
    "CAMERA_NUMERICAL_MAX_REL_DIFF",
    "DERIVED_FIELD_COUNT",
    "DERIVED_BYTE_EXACT_COUNT",
    "DERIVED_TENSOR_EXACT_COUNT",
    "DERIVED_RUNTIME_EQUIVALENT_COUNT",
    "DERIVED_MISMATCH_COUNT",
    "SCHEMA_EQUIVALENCE_STATUS",
    "LOADER_IMPLEMENTATION_STATUS",
    "LOADER_FIELD_SET_STATUS",
    "LOADER_SHAPE_STATUS",
    "LOADER_VALUE_HASH_STATUS",
    "CHECKPOINT_TARGET_BINDING_STATUS",
    "VIEW_SAMPLE_COUNTS",
    "TARGET_EQUIVALENCE_CLASS",
    "SCIENTIFIC_FIELD_MISMATCH_COUNT",
    "MISMATCH_FIELDS",
    "RERUN_TRAINED_ON_FORMAL_EQUIVALENT_TARGETS",
    "BINDING_OVERLAY_PATH",
    "REVIEW_PACK_STATUS",
    "REVIEW_PACK_PATH",
    "REVIEW_PACK_SHA256",
    "OPTIMIZER_STEPS",
    "GPU_FORWARD_CALLS",
    "TARGET_MUTATIONS",
    "MASK_MUTATIONS",
    "CAMERA_RECORD_MUTATIONS",
    "CHECKPOINT_MUTATIONS",
    "FORMAL_BASE_STATUS",
    "FORMAL_BASE_DURABLE_RESUME_STEP",
    "FORMAL_BASE_RESUME_AUTHORIZED",
    "PAPER_ELIGIBLE",
    "SCIENTIFIC_PASS",
    "PAPER_MODIFICATIONS",
    "TEST_RESULT",
    "COMMIT_HEAD",
    "FINAL_REPORTING_HEAD",
    "ORIGIN_SYNC_STATUS",
    "CLOUD_GIT_SYNC_STATUS",
    "WORKTREE_CLEAN_STATUS",
    "PAPER_FINAL",
    "FINAL_CLASSIFICATION",
    "NEXT_TASK",
]


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value.get("task_id", value.get("TASK_ID")) == TASK_ID
    return value


def test_required_artifacts_parse_and_final_fields_are_complete():
    assert len(JSON_PATHS) == 10
    values = {path.name: load(path) for path in JSON_PATHS}
    summary = values[
        "subject00_O03_camsafe7_formal_target_equivalence_final_summary_20260727.json"
    ]
    actual_fields = [
        key
        for key in summary
        if key not in {"schema_version", "task_id", "created_at"}
    ]
    assert actual_fields == REQUIRED_FINAL_FIELDS
    assert len(REQUIRED_FINAL_FIELDS) == 69
    assert summary["FINAL_CLASSIFICATION"] == FINAL_CLASSIFICATION
    assert summary["NEXT_TASK"] == NEXT_TASK
    assert summary["SCIENTIFIC_PASS"] is None
    assert summary["PAPER_ELIGIBLE"] is False
    assert summary["PAPER_FINAL"] is False


def test_class_d_is_supported_by_exact_shared_assets_and_full_field_mismatch():
    raw = load(JSON_PATHS[1])
    camera = load(JSON_PATHS[2])
    derived = load(JSON_PATHS[3])
    loader = load(JSON_PATHS[4])
    checkpoint = load(JSON_PATHS[5])
    overlay = load(JSON_PATHS[6])
    tests = load(JSON_PATHS[7])

    assert (
        raw["raw_sha_exact_match_count"],
        raw["raw_pixel_exact_match_count"],
        raw["person_mask_sha_exact_match_count"],
        raw["person_mask_pixel_exact_match_count"],
        raw["garment_mask_sha_exact_match_count"],
        raw["garment_mask_pixel_exact_match_count"],
    ) == (7, 7, 7, 7, 7, 7)
    assert camera["camera_record_count"] == 7
    assert camera["camera_canonical_exact_count"] == 7
    assert camera["camera_numerical_max_abs_diff"] == 0.0
    assert camera["camera_numerical_max_rel_diff"] == 0.0
    assert derived["derived_field_count"] == 14
    assert derived["derived_mismatch_count"] == 8
    assert derived["formal_transition_equals_rerun_boundary_count"] == 0
    assert len(derived["scientific_mismatch_fields"]) == 8
    assert loader["shared_value_hash_status"] == "PASS_7_OF_7_SHARED_FIELDS"
    assert loader["full_value_hash_status"].startswith("FAIL_")
    assert checkpoint["checkpoint_count"] == 5
    assert checkpoint["checkpoint_steps"] == [0, 300, 600, 900, 1200]
    assert overlay["binding_created"] is False
    assert overlay["TARGET_EQUIVALENCE_CLASS"] == "D"
    assert overlay["RERUN_TRAINED_ON_FORMAL_EQUIVALENT_TARGETS"] is False
    assert tests["status"] == "PASS_40_OF_40"
    assert tests["passed_count"] == 40
    assert tests["failed_count"] == 0
    assert all(row["status"] == "PASS" for row in tests["checks"])


def test_reports_match_and_no_payload_is_added():
    risk_report = (
        RISK / "SUBJECT00_O03_CAMSAFE7_FORMAL_TARGET_EQUIVALENCE_REPORT_20260727.md"
    )
    docs_report = (
        ROOT
        / "docs"
        / "PAPER"
        / "AAAI27_SUBJECT00_O03_CAMSAFE7_FORMAL_TARGET_EQUIVALENCE_REPORT_20260727.md"
    )
    assert risk_report.read_bytes() == docs_report.read_bytes()
    report = risk_report.read_text(encoding="utf-8")
    assert "FORMAL_TARGET_SCIENTIFIC_FIELD_MISMATCH" in report
    assert FINAL_CLASSIFICATION in report
    assert NEXT_TASK in report

    status = subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain=v1"],
        text=True,
        encoding="utf-8",
    )
    changed = [
        line[3:].strip().strip('"').replace("\\", "/")
        for line in status.splitlines()
        if line.strip()
    ]
    forbidden = (".png", ".jpg", ".jpeg", ".pdf", ".pth", ".pt", ".ckpt")
    assert not [path for path in changed if path.lower().endswith(forbidden)]
