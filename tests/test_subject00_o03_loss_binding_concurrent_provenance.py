import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
TASK_ID = "AAAI27-SUBJECT00-O03-LOSS-BINDING-CONCURRENT-RUN-PROVENANCE-001"
FINAL_CLASSIFICATION = (
    "SUBJECT00_O03_CONCURRENT_FORMAL_TEACHER_VALID_METHOD_STEP300_VALID_"
    "MATRIX_READY_TO_CONTINUE"
)
NEXT_TASK = (
    "CONTINUE_SUBJECT00_BASE60747_REMAINING_11_METHOD_RUNS_AND_FAIR_BASELINES"
)
O03_SHA = "054b9efe1086d89b18310b9831d3ed7aea59e286e31200509b58756545fc3920"


def load(name: str) -> dict:
    return json.loads((RISK / name).read_text(encoding="utf-8"))


def test_final_summary_is_unique_and_zero_step() -> None:
    value = load(
        "subject00_O03_loss_binding_concurrent_provenance_final_summary_"
        "20260727.json"
    )
    assert value["TASK_ID"] == TASK_ID
    assert value["FINAL_CLASSIFICATION"] == FINAL_CLASSIFICATION
    assert value["NEXT_TASK"] == NEXT_TASK
    assert value["NEW_TEACHER_OPTIMIZER_STEPS"] == 0
    assert value["NEW_METHOD_OPTIMIZER_STEPS"] == 0
    assert value["PAPER_MODIFICATIONS"] == 0
    assert value["PAPER_FINAL"] is False


def test_actual_loss_and_evaluation_fields_are_frozen() -> None:
    value = load(
        "subject00_capacity_oracle_loss_v1_active_field_registry_20260727.json"
    )
    assert value["status"] == "PASS_ALL_LOADER_FIELDS_CLASSIFIED"
    assert value["loader_field_count"] == 35
    assert set(value["loss_active_field_set"]) == {
        "target_edit_rgb",
        "target_base_rgb",
        "target_foreground_mask",
        "target_base_foreground_mask",
        "target_edit_mask",
        "target_clothing_mask",
        "target_old_clothing_mask",
        "target_protected_mask",
        "target_transition_mask",
    }
    assert set(value["schema_only_scientific_field_set"]) == {
        "target_edit_core_mask",
        "target_preserve_mask",
        "target_revealed_skin_mask",
    }
    assert not value["dynamic_access_unresolved_field_set"]


def test_runtime_trace_is_zero_optimizer_and_matches_static_graph() -> None:
    value = load(
        "subject00_capacity_oracle_loss_v1_runtime_field_access_trace_"
        "20260727.json"
    )
    active = load(
        "subject00_capacity_oracle_loss_v1_active_field_registry_20260727.json"
    )
    assert value["status"] == "PASS_ZERO_OPTIMIZER_RUNTIME_ACCESS_TRACE"
    assert value["optimizer_constructed"] is False
    assert value["optimizer_steps"] == 0
    assert value["backward_completed"] is True
    assert value["gradients_finite"] is True
    assert set(value["access_by_stage"]["loss"]) == set(
        active["loss_active_field_set"]
    )
    assert set(value["access_by_stage"]["evaluation"]) == set(
        active["evaluation_active_field_set"]
    )


def test_old_safe7_reassessment_uses_active_mismatches_only() -> None:
    value = load(
        "subject00_O03_active_input_equivalence_reassessment_20260727.json"
    )
    assert value["classification"] == "D_ACTIVE_LOSS_INPUT_SCIENTIFIC_MISMATCH"
    assert value["active_field_mismatch_count"] == 5
    assert set(value["active_field_mismatch_fields"]) == {
        "target_base_rgb",
        "target_edit_mask",
        "target_transition_mask",
        "target_base_foreground_mask",
        "target_old_clothing_mask",
    }
    assert value["inactive_schema_fields_used_to_maintain_class_d"] is False


def test_candidate_checkpoint_authenticity_is_complete() -> None:
    value = load(
        "subject00_O03_candidate_checkpoint_authenticity_audit_20260727.json"
    )
    assert value["expected_steps"] == [0, 300, 600, 900, 1200]
    assert value["checkpoint_count"] == value["parse_count"] == 5
    assert value["formal_target"]["denominator"] == 7
    assert value["formal_target"]["slot04_present"] is False
    assert value["structured_training"]["record_count"] == 1200
    assert value["candidate_final_checkpoint_sha256"] == O03_SHA
    assert value["authenticity_status"].startswith("PASS")


def test_writer_is_unique_without_cross_task_interleaving() -> None:
    value = load(
        "subject00_O03_concurrent_output_writer_timeline_20260727.json"
    )
    assert value["single_writer_status"].startswith("PASS")
    assert value["concurrent_writer_interleaving_status"].startswith("PASS")
    assert value["blocked_task_marker_count"] == 0
    assert value["blocked_task"]["optimizer_steps_completed"] == 0
    assert value["blocked_task"]["checkpoint_count"] == 0
    assert value["blocked_task"]["candidate_run_root_file_writes"] == 0


def test_method_step300_binds_selected_o03_teacher() -> None:
    value = load(
        "subject00_method_step300_teacher_dependency_audit_20260727.json"
    )
    assert value["checkpoint"]["global_step"] == 300
    assert value["checkpoint"]["optimizer_step"] == 300
    assert value["checkpoint"]["teacher_checkpoint_sha256"]["O03"] == O03_SHA
    assert value["o03_dependency_status"].startswith("PASS")
    assert value["method_rotation0_seed0_valid"] is True


def test_o01_o04_and_registry_overlay_are_valid() -> None:
    teachers = load(
        "subject00_O01_O04_teacher_lightweight_provenance_audit_20260727.json"
    )
    correction = load(
        "subject00_base60747_three_garment_teacher_registry_correction_overlay_"
        "20260727.json"
    )
    assert teachers["teachers"]["O01"]["validity"] == "VALID_FORMAL_TARGET_TEACHER"
    assert teachers["teachers"]["O04"]["validity"] == "VALID_FORMAL_TARGET_TEACHER"
    assert correction["original_registry_overwritten"] is False
    assert correction["corrections"]["O03"]["checkpoint_sha256"] == O03_SHA
    assert correction["method_step300_valid"] is True


def test_machine_checks_and_provenance_seal_are_complete() -> None:
    tests = load(
        "subject00_O03_loss_binding_concurrent_provenance_tests_20260727.json"
    )
    seal = load(
        "subject00_O03_concurrent_formal_teacher_provenance_seal_20260727.json"
    )
    assert tests["status"] == "PASS_45_OF_45"
    assert tests["passed_count"] == len(tests["checks"]) == 45
    assert tests["failed_count"] == 0
    assert all(row["status"] == "PASS" for row in tests["checks"])
    assert seal["validity_status"] == "VALID_FORMAL_TARGET_TEACHER"
    assert seal["checkpoint_sha256"]["1200"] == O03_SHA


def test_required_reports_and_handoff_exist() -> None:
    assert (
        RISK
        / "SUBJECT00_O03_LOSS_BINDING_CONCURRENT_PROVENANCE_REPORT_20260727.md"
    ).is_file()
    assert (
        ROOT
        / "docs/PAPER/"
        "AAAI27_SUBJECT00_O03_LOSS_BINDING_CONCURRENT_PROVENANCE_REPORT_"
        "20260727.md"
    ).is_file()
    handoff = json.loads(
        (
            ROOT
            / "project_control_handoff/"
            "subject00_O03_loss_binding_concurrent_provenance_handoff_"
            "20260727.json"
        ).read_text(encoding="utf-8")
    )
    assert handoff["outcome"]["final_classification"] == FINAL_CLASSIFICATION
    assert handoff["outcome"]["next_task"] == NEXT_TASK
