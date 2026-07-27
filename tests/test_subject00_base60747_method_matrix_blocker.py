from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / "paper_protocol/reviewer_risk"


def load(name: str) -> dict:
    return json.loads((RISK / name).read_text(encoding="utf-8"))


def test_matrix_contract_detects_exact_quarantine_fold_conflict() -> None:
    value = load("subject00_base60747_method_12run_matrix_contract_20260727.json")
    assert value["method"]["contract"] == "PURE_ENDPOINT"
    assert value["method"]["dual_support_enabled"] is False
    assert value["matrix"]["expected_run_count"] == 12
    assert value["matrix"]["expected_total_optimizer_steps"] == 3600
    assert value["matrix"]["seeds"] == [0, 1, 2]
    assert value["formal_fold_coverage"]["slot04"] == {"O01": 0, "O03": 0, "O04": 1}
    assert len(value["blocking_conflicts"]) == 4
    assert value["adjudication"]["new_method_runs_authorized"] is False


def test_completed_training_is_preserved_but_not_promoted_to_formal_crossfit() -> None:
    execution = load("subject00_base60747_method_12run_execution_registry_20260727.json")
    checkpoints = load("subject00_base60747_method_checkpoint_registry_20260727.json")
    metrics = load("subject00_base60747_method_per_run_metrics_20260727.json")
    assert execution["completed_run_count_before"] == 1
    assert execution["completed_training_authentic_count"] == 1
    assert execution["completed_formal_matrix_cell_count"] == 0
    assert execution["executed_new_run_count"] == 0
    assert execution["actual_total_optimizer_steps"] == 300
    assert [row["step"] for row in checkpoints["checkpoints"]] == [0, 20, 50, 100, 200, 300]
    assert checkpoints["completed_rotation0_seed0_mutations"] == 0
    measured = metrics["runs"][0]
    assert measured["training_status"] == "TECHNICAL_PASS"
    assert measured["formal_matrix_status"] == "FAIL_MISSING_FORMAL_TEST_FOLD"
    assert measured["formal_test_endpoint_top1"] is None


def test_baseline_set_is_unique_but_execution_is_correctly_blocked() -> None:
    contract = load("subject00_base60747_fair_baseline_contract_20260727.json")
    execution = load("subject00_base60747_fair_baseline_execution_registry_20260727.json")
    assert [row["name"] for row in contract["paper_facing_internal_fair_baselines"]] == [
        "Reference Classifier Lookup",
        "Nearest-Centroid Lookup",
        "Outfit-ID Oracle",
        "Teacher Endpoint",
    ]
    assert contract["execution_gate"]["execution_authorized"] is False
    assert execution["executed_run_count"] == 0
    assert execution["optimizer_steps"] == 0
    assert execution["attempt_002_created_count"] == 0


def test_52_nonempty_checks_and_required_failure_are_recorded() -> None:
    value = load("subject00_base60747_method_matrix_baseline_tests_20260727.json")
    assert value["test_count"] == 52
    assert len(value["tests"]) == 52
    assert [row["id"] for row in value["tests"]] == list(range(1, 53))
    assert all(row["name"] and row["evidence"] for row in value["tests"])
    assert value["fail_count"] == 1
    total = value["pass_count"] + value["blocked_count"] + value["fail_count"]
    assert total == 52
    assert value["tests"][24]["name"] == "total matrix steps 3600"
    assert value["tests"][24]["status"] == "FAIL"
