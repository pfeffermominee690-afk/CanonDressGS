from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / "paper_protocol" / "reviewer_risk"


def read(name: str) -> dict:
    return json.loads((RISK / name).read_text(encoding="utf-8"))


def test_materialization_summary_contract() -> None:
    summary = read("subject00_teacher_target_materialization_final_summary_20260727.json")
    assert summary["total_provenance_record_count"] == 24
    assert summary["training_eligible_record_count"] == 22
    assert summary["review_only_quarantined_record_count"] == 2
    assert summary["O01_training_record_count"] == 7
    assert summary["O03_training_record_count"] == 7
    assert summary["O04_training_record_count"] == 8
    assert summary["quarantine_derived_target_count"] == 0
    assert summary["optimizer_steps"] == 0
    assert summary["paper_modifications"] == 0
    assert (
        summary["final_classification"]
        == "SUBJECT00_TEACHER_TARGET_MATERIALIZATION_PASS_22_TRAINING_2_QUARANTINED"
    )


def test_quarantine_is_exact_and_camera_null() -> None:
    value = read(
        "subject00_teacher_target_quarantine_materialization_registry_20260727.json"
    )
    assert set(value["request_ids"]) == {
        "subject00_O03_slot04_canary_attempt004_cand00",
        "subject00_O01_slot04_remaining_attempt005_cand00",
    }
    assert value["camera_model"] is None
    assert value["K_target"] is None
    assert value["derived_camera_targets_authorized"] is False


def test_o03_camera_safe_seven_view_contract() -> None:
    value = read("subject00_O03_camera_safe_7view_teacher_target_registry_20260727.json")
    assert value["record_count"] == 7
    assert value["slots"] == ["00", "01", "02", "03", "05", "06", "07"]
    assert value["disclosure"] == "PROVISIONAL_O03_USES_CAMERA_SAFE_7_OF_8_VIEWS"
    assert value["paper_eligible"] is False


def test_structured_checks_are_real_and_complete() -> None:
    value = read("subject00_teacher_target_materialization_execution_tests_20260727.json")
    assert value["structured_check_count"] == 46
    assert value["structured_pass_count"] == 46
    assert len(value["checks"]) == 46
    assert {item["status"] for item in value["checks"]} == {"PASS"}


def test_cloud_loader_smoke_is_zero_optimizer() -> None:
    value = read("subject00_teacher_target_loader_smoke_results_20260727.json")
    assert value["status"] == "PASS_ZERO_OPTIMIZER_CPU_LOADER_SMOKE"
    assert value["loader_training_count"] == 22
    assert value["loader_quarantine_count"] == 2
    assert value["optimizer_steps"] == 0
    assert value["gpu_calls"] == 0
