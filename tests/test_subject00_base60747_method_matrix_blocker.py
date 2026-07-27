from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / "paper_protocol/reviewer_risk"
TASK_ID = "AAAI27-SUBJECT00-BASE60747-REMAINING-METHOD-MATRIX-FAIR-BASELINES-001"


def load(name: str) -> dict:
    return json.loads((RISK / name).read_text(encoding="utf-8"))


def test_runtime_preflight_passes_all_critical_checks() -> None:
    value = load("subject00_base60747_method_matrix_runtime_preflight_20260727.json")
    assert value["task_id"] == TASK_ID
    assert len(value["critical_checks"]) == 30
    assert all(value["critical_checks"].values())
    assert value["targets"]["training_count"] == 22
    assert value["targets"]["checksum_binding_count"] == 330
    assert value["targets"]["checksum_mismatch_count"] == 0


def test_matrix_contract_detects_exact_quarantine_fold_conflict() -> None:
    value = load("subject00_base60747_method_12run_matrix_contract_20260727.json")
    assert value["method_contract"] == "PURE_ENDPOINT"
    assert value["dual_support_enabled"] is False
    assert value["expected_run_count"] == 12
    assert value["expected_total_optimizer_steps"] == 3600
    assert value["seeds"] == [0, 1, 2]
    assert len(value["execution_gate"]["blockers"]) == 4
    assert value["execution_gate"]["status"] == "FAIL_FROZEN_FOLD_COVERAGE"


def test_completed_training_is_preserved_without_new_optimizer_steps() -> None:
    execution = load("subject00_base60747_method_12run_execution_registry_20260727.json")
    checkpoints = load("subject00_base60747_method_checkpoint_registry_20260727.json")
    metrics = load("subject00_base60747_method_per_run_metrics_20260727.json")
    assert execution["completed_run_count_before"] == 1
    assert execution["completed_run_count_after"] == 1
    assert execution["executed_new_run_ids"] == []
    assert execution["actual_total_optimizer_steps"] == 300
    assert execution["new_method_optimizer_steps"] == 0
    assert execution["new_baseline_optimizer_steps"] == 0
    assert checkpoints["runs"]["METHOD-R0-S0"]["status"] == "PASS_6_OF_6"
    measured = metrics["runs"]["METHOD-R0-S0"]
    assert measured["optimizer_steps"] == 300
    assert measured["training_fold_nearest_endpoint_correct"] == 6
    assert measured["formal_test_fold_result"] is None


def test_execution_lock_is_closed_with_no_optimizer_construction() -> None:
    value = load("subject00_base60747_method_matrix_execution_lock_snapshot_20260727.json")
    assert value["task_id"] == TASK_ID
    assert value["status"] == "CLOSED_BLOCKED_PREFLIGHT"
    assert value["active_optimizer_process_count_before"] == 0
    assert value["new_method_optimizer_steps"] == 0
    assert value["new_baseline_optimizer_steps"] == 0
    assert len(value["expected_pending_run_ids"]) == 11


def test_baselines_are_not_executed_after_matrix_gate_failure() -> None:
    contract = load("subject00_base60747_fair_baseline_contract_20260727.json")
    execution = load("subject00_base60747_fair_baseline_execution_registry_20260727.json")
    assert contract["paper_facing_names"] == [
        "Reference Classifier Lookup",
        "Nearest-Centroid Lookup",
        "Outfit-ID Oracle",
        "Teacher Endpoint",
    ]
    assert contract["name_set_unique"] is True
    assert contract["full_execution_contract_unique"] is False
    assert contract["execution_allowed_after_gates"] is False
    assert execution["executed_baselines"] == []
    assert execution["run_count"] == 0
    assert execution["optimizer_steps"] == 0


def test_54_nonempty_contract_checks_are_recorded() -> None:
    value = load("subject00_base60747_method_matrix_baseline_tests_20260727.json")
    assert len(value["contract_checks"]) == 54
    assert [row["index"] for row in value["contract_checks"]] == list(range(1, 55))
    assert all(row["name"] and row["status"] for row in value["contract_checks"])
    assert value["unexpected_failure_count"] == 0
    assert value["optimizer_constructed"] is False
    assert value["new_optimizer_steps"] == 0


def test_final_summary_preserves_all_immutability_requirements() -> None:
    value = load("subject00_base60747_method_matrix_baseline_final_summary_20260727.json")
    mutation_fields = [
        "BASE60747_CHECKPOINT_MUTATIONS",
        "O01_TEACHER_CHECKPOINT_MUTATIONS",
        "O03_TEACHER_CHECKPOINT_MUTATIONS",
        "O04_TEACHER_CHECKPOINT_MUTATIONS",
        "FORMAL_TARGET_MUTATIONS",
        "RAW_MUTATIONS",
        "MASK_MUTATIONS",
        "CAMERA_RECORD_MUTATIONS",
        "COMPLETED_ROTATION0_SEED0_MUTATIONS",
        "FORMAL_BASE_RUN_MUTATIONS",
        "PAPER_MODIFICATIONS",
    ]
    assert all(value[field] == 0 for field in mutation_fields)
    assert value["FORMAL_BASE_STATUS"] == "USER_AUTHORIZED_PAUSED"
    assert value["FORMAL_BASE_RESUME_AUTHORIZED"] is False
    assert value["PAPER_FINAL"] is False
    assert value["FINAL_CLASSIFICATION"] == "SUBJECT00_BASE60747_METHOD_MATRIX_ENGINEERING_FAIL"
    assert value["NEXT_TASK"] == "USER_REVIEW_SUBJECT00_BASE60747_METHOD_MATRIX_ENGINEERING_FAILURE"
