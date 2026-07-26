from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
TARGET_ROOT = Path(
    r"E:\canondressgs_data\subject00_three_garment"
    r"\CODEX-MANAGED-TEACHER-TARGETS-001"
)


def load(name: str):
    return json.loads((RISK / name).read_text(encoding="utf-8"))


def test_execution_manifest_exact_counts_order_and_zero_mutation():
    value = load(
        "subject00_24_cell_teacher_target_creation_execution_manifest_20260727.json"
    )
    records = value["records"]
    assert len(records) == 24
    assert [x["sequence_index"] for x in records] == list(range(1, 25))
    assert [x["garment"] for x in records] == ["O01"] * 8 + ["O03"] * 8 + ["O04"] * 8
    assert len({x["accepted_request_id"] for x in records}) == 24
    assert not value["stage_a"]["authorized"]
    assert not value["stage_b"]["authorized"]
    assert not value["target_root_created"]
    assert value["stage_a"]["teacher_target_count"] == 0
    assert all(not x["materialized"] and not x["teacher_target"] for x in records)
    assert all(item == 0 for item in value["counters"].values() if isinstance(item, int))
    assert not TARGET_ROOT.exists()


def test_schema_camera_resolution_and_risk_contracts():
    schema = load("subject00_24_cell_teacher_target_schema_registry_20260727.json")
    camera = load("subject00_24_cell_teacher_target_camera_registry_20260727.json")
    resolution = load(
        "subject00_24_cell_teacher_target_resolution_contract_20260727.json"
    )
    assert schema["target_schema"] == "canondressgs.full_dataset.v1"
    assert schema["record_count"] == 24
    assert camera["record_count"] == 24
    assert camera["unique_camera_binding_count"] == 22
    assert camera["blocked_camera_binding_count"] == 2
    assert sum(x["target_K"] is None for x in camera["records"]) == 2
    assert resolution["native_resolution_distribution"] == {
        "1349x1166": 23,
        "1350x1165": 1,
    }
    assert (
        resolution["mixed_resolution_support_status"]
        == "BATCH_SIZE_ONE_NATIVE_RESOLUTION_SUPPORTED"
    )
    assert resolution["silent_resize_allowed"] is False
    limited = [
        x for x in schema["records"] if x["official_limitation_codes"]
    ]
    overrides = [
        x
        for x in schema["records"]
        if x["target_readiness_status"] == "BLOCKED_CAMERA_MODEL_NONUNIQUE"
    ]
    assert len(limited) == 9
    assert len(overrides) == 2
    assert all(
        any(
            item["code"] == "MACHINE_REGISTRATION_FAIL_WITH_DOCUMENTED_HUMAN_OVERRIDE"
            and item["affects_geometry_supervision"]
            and item["manual_disclosure_required"]
            for item in record["propagated_limitations"]
        )
        for record in overrides
    )


def test_materialization_qa_o03_storage_and_final_blocker():
    audit = load(
        "subject00_24_cell_teacher_target_materialization_mode_audit_20260727.json"
    )
    qa = load("subject00_24_cell_teacher_target_qa_registry_20260727.json")
    o03 = load(
        "subject00_O03_provisional_teacher_target_manifest_draft_20260727.json"
    )
    storage = load(
        "subject00_24_cell_teacher_target_storage_transfer_estimate_20260727.json"
    )
    tests = load(
        "subject00_24_cell_teacher_target_creation_preflight_tests_20260727.json"
    )
    summary = load(
        "subject00_24_cell_teacher_target_creation_preflight_final_summary_20260727.json"
    )
    assert audit["selected_mode"] == "PORTABLE_ARCHIVE_AND_CLOUD_EXTRACTION"
    assert audit["target_root_preexisted"] is False
    assert audit["target_root_created"] is False
    assert audit["data_copy_calls"] == audit["upload_calls"] == 0
    assert qa["qa_gate_count"] >= 24
    assert qa["zero_optimizer_step_smoke"]["status"].startswith("PREPARED_NOT_RUN")
    assert o03["record_count"] == 8
    assert [x["slot"] for x in o03["records"]] == [
        f"slot_{index:02d}" for index in range(8)
    ]
    assert o03["provisional_base"]["paper_eligible"] is False
    assert storage["storage_gate_status"] == "PASS"
    assert tests["check_count"] == 46
    assert tests["pass_count"] == 45
    assert tests["blocked_count"] == 1
    assert summary["teacher_target_materialization_requires_final_base"] is False
    assert summary["teacher_target_creation_ready_independent_of_base"] is True
    assert summary["teacher_endpoint_optimization_requires_base"] is True
    assert summary["final_classification"] == (
        "SUBJECT00_TEACHER_TARGET_PREFLIGHT_BLOCKED_BY_CAMERA_RESOLUTION_CONTRACT"
    )
    assert summary["next_task"] == (
        "RESOLVE_SUBJECT00_TEACHER_TARGET_CAMERA_RESOLUTION_BLOCKER"
    )
    assert summary["paper_final"] is False
