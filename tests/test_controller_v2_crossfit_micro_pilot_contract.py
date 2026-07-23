from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.paper import seal_controller_v2_crossfit_contract_incomplete as seal


ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / "paper_protocol/reviewer_risk"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_exact_source_head_is_frozen() -> None:
    assert seal.SOURCE_HEAD == "4f8c94405e68932e791f69c59ae129c863b1280d"


def test_all_ten_design_artifact_hashes_match() -> None:
    assert len(seal.DESIGN_HASHES) == 10
    for relative, expected in seal.DESIGN_HASHES.items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected


def test_resource_and_io_gate_passed() -> None:
    gate = seal.RESOURCE_GATE
    assert gate["status"] == "PASS"
    assert gate["storage"]["free_bytes"] >= gate["storage"]["minimum_required_bytes"]
    assert gate["gpu"]["compute_process_count"] == 0
    assert gate["activity"]["avatarex_extraction_active"] is False
    assert gate["activity"]["avatarex_full_jpeg_audit_active"] is False
    assert gate["activity"]["subject00_runtime_active"] is False
    assert gate["activity"]["other_formal_render_active"] is False


def test_contract_audits_exactly_thirty_required_fields() -> None:
    contract = read_json(
        RISK / "controller_v2_micro_pilot_training_contract.json"
    )
    assert contract["contract_check_count"] == 30
    assert [row["id"] for row in contract["contract_checks"]] == list(range(1, 31))


def test_contract_is_incomplete_before_training() -> None:
    contract = read_json(
        RISK / "controller_v2_micro_pilot_training_contract.json"
    )
    assert contract["status"] == "CONTRACT_INCOMPLETE"
    assert (
        contract["classification"]
        == "CONTROLLER_V2_MICRO_PILOT_CONTRACT_INCOMPLETE"
    )
    assert contract["contract_pass_count"] == 8
    assert contract["contract_missing_count"] == 22
    assert contract["training_authorized"] is False


def test_missing_optimizer_schedule_is_not_defaulted() -> None:
    contract = read_json(
        RISK / "controller_v2_micro_pilot_training_contract.json"
    )
    missing = set(contract["missing_fields"])
    assert {
        "batch_size",
        "data_order_algorithm",
        "optimizer_type",
        "learning_rate",
        "weight_decay",
        "total_optimizer_steps",
        "checkpoint_cadence",
        "final_checkpoint_rule",
        "lambda_mix",
        "lambda_weight",
        "lambda_cons",
    }.issubset(missing)


def test_crossfit_fold_membership_remains_frozen() -> None:
    split = read_json(RISK / "controller_v2_crossfit_splits.json")
    assert len(split["rotations"]) == 4
    for rotation in split["rotations"]:
        folds = (
            rotation["train_fold_indices"]
            + [rotation["calibration_fold_index"], rotation["test_fold_index"]]
        )
        assert sorted(folds) == [0, 1, 2, 3]


def test_exact_record_counts_are_unresolved_not_invented() -> None:
    contract = read_json(
        RISK / "controller_v2_micro_pilot_training_contract.json"
    )
    for rotation in contract["rotation_contracts"]:
        assert rotation["train_record_count"] is None
        assert rotation["calibration_record_count"] is None
        assert rotation["test_record_count"] is None
        assert rotation["exact_record_ids"] is None
        assert rotation["record_manifest_sha256"] is None


def test_matched_v1_is_not_substituted_by_historical_formal_v1() -> None:
    contract = read_json(
        RISK / "controller_v2_micro_pilot_training_contract.json"
    )
    comparability = contract["baseline_comparability"]
    assert comparability["status"] == "INCOMPLETE"
    assert comparability["historical_formal_v1_may_substitute"] is False
    assert (
        contract["model_families"]["MATCHED_V1"]["authorized_run_count"] == 0
    )


def test_training_optimizer_checkpoint_and_render_counts_are_zero() -> None:
    contract = read_json(
        RISK / "controller_v2_micro_pilot_training_contract.json"
    )
    assert all(value == 0 for value in contract["execution_counts"].values())
    assert contract["execution_counts"]["training_runs"] == 0
    assert contract["execution_counts"]["optimizer_creations"] == 0
    assert contract["execution_counts"]["checkpoint_writes"] == 0
    assert contract["execution_counts"]["renderer_calls"] == 0


def test_downstream_machine_artifacts_are_explicitly_not_run() -> None:
    paths = [
        "controller_v2_micro_pilot_training_results.json",
        "controller_v1_matched_crossfit_results.json",
        "controller_v2_calibration_results.json",
        "controller_v2_test_predictions.json",
        "controller_v2_routing_results.json",
        "controller_v2_visual_results.json",
        "controller_v2_perturbation_results.json",
        "controller_v2_efficiency_results.json",
        "controller_v2_visual_review.json",
    ]
    for name in paths:
        value = read_json(RISK / name)
        assert value["status"] == "NOT_RUN_CONTRACT_INCOMPLETE"
        assert value["records"] == []
        assert value["denominator"] == 0


def test_visual_review_is_not_falsely_marked_complete() -> None:
    review = read_json(RISK / "controller_v2_visual_review.json")
    assert review["expected_main_sheet_count_if_authorized"] == 240
    assert review["generated_main_sheet_count"] == 0
    assert review["actual_opened_count"] == 0
    assert review["reviewer"] is None
    assert review["grades"] == []


def test_no_gt_or_target_forward_execution_occurred() -> None:
    summary = read_json(RISK / "controller_v2_micro_pilot_final_summary.json")
    assert summary["information_boundary"]["gt_inference_use"] == 0
    assert summary["information_boundary"]["target_forward_use"] == 0
    assert summary["execution_counts"]["inference_runs"] == 0


def test_frozen_upstream_mutation_count_is_zero() -> None:
    summary = read_json(RISK / "controller_v2_micro_pilot_final_summary.json")
    assert summary["frozen_upstream_mutation_count"] == 0


def test_paper_final_is_zero() -> None:
    summary = read_json(RISK / "controller_v2_micro_pilot_final_summary.json")
    handoff = read_json(
        ROOT
        / "project_control_handoff/controller_v2_crossfit_micro_pilot_handoff.json"
    )
    assert summary["paper_final"] == 0
    assert handoff["paper_final"] == 0


def test_all_required_reports_are_nonempty() -> None:
    for relative in seal.REPORT_PATHS:
        path = ROOT / relative
        assert path.is_file()
        assert path.stat().st_size > 0


def test_all_required_machine_paths_exist() -> None:
    for relative in seal.MACHINE_PATHS:
        assert (ROOT / relative).is_file()
    assert (ROOT / seal.HANDOFF_PATH).is_file()


def test_sealer_contains_no_training_runtime_calls() -> None:
    source = (
        ROOT / "tools/paper/seal_controller_v2_crossfit_contract_incomplete.py"
    ).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "optimizer.step(" not in source
    assert "torch.save(" not in source
    assert "gaussian_renderer" not in source


def test_final_classification_and_next_task_are_frozen() -> None:
    summary = read_json(RISK / "controller_v2_micro_pilot_final_summary.json")
    assert summary["final_classification"] == seal.CLASSIFICATION
    assert summary["next_task"] == seal.NEXT_TASK
    assert summary["next_task_started"] is False
