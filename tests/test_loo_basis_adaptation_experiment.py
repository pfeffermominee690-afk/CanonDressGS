from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.paper import loo_basis_output as output
from tools.paper import run_loo_basis_adaptation_experiment as runner


def test_repaired_source_and_historical_artifacts_are_exact() -> None:
    source = runner.source_artifact_audit()
    historical = runner.historical_immutability_audit()
    assert source["status"] == "PASS"
    assert source["artifact_count"] == len(runner.REPAIRED_FILES) + len(runner.INHERITED_FILES)
    assert historical["status"] == "PASS"
    assert historical["artifact_count"] == 14
    assert historical["mutation_count"] == 0
    assert historical["historical_classification"] == "LOO_PROTOCOL_INCOMPLETE"
    assert historical["historical_valid_tasks"] == 15
    assert historical["historical_blocked_tasks"] == 45


def test_protocol_audit_freezes_40_ready_k1_k2_tasks() -> None:
    audit = runner.protocol_audit()
    assert audit["status"] == "PASS"
    assert all(audit["checks"].values())
    assert audit["budget_counts"] == {1: 20, 2: 20}
    assert runner.expected_counts()["total_optimizer_runs"] == 120
    assert runner.expected_counts()["optimizer_steps"] == 36_000
    assert runner.expected_counts()["checkpoint_writes"] == 720
    assert runner.expected_counts()["evaluation_inference"] == 960
    assert runner.expected_counts()["total_logical_renders"] == 54_960
    assert runner.expected_counts()["unique_physical_renders"] == 54_840


def test_five_loo_banks_exclude_held_out_and_rank_contract() -> None:
    splits = runner.split_rows()
    assert len(splits) == 5
    for split in splits:
        held_out = split["held_out_garment"]
        assert len(split["basis_garments"]) == 4
        assert held_out not in split["basis_garments"]
        contract = split["basis_contract"]
        assert contract["rank_rule"] == "min(numerical_rank,3)"
        assert contract["affine_rank_upper_bound"] == 3
        assert contract["full_five_garment_basis_reused"] is False
        assert contract["forbidden_full_basis_sha256"] == runner.FULL_BASIS_FORBIDDEN_SHA


def test_task_mappings_subset_and_information_boundary() -> None:
    tasks = runner.tasks()
    k1 = {"R0": "cond_000000", "R1": "cond_000017", "R2": "cond_000347", "R3": "cond_000000"}
    for row in tasks:
        adaptation = row["selected_adaptation_conditions"]
        calibration = row["calibration_conditions"]
        test = row["test_conditions"]
        assert not set(adaptation) & set(calibration)
        assert not set(adaptation) & set(test)
        assert not set(calibration) & set(test)
        assert len(adaptation) == row["K"]
        assert len(set(adaptation)) == row["K"]
        assert row["held_out_garment"] not in row["basis_garments"]
        assert all(value == 0 for name, value in row["held_out_information_use"].items() if name != "teacher_offline_oracle_only")
        if row["K"] == 1:
            assert adaptation == [k1[row["rotation"]]]
    for outfit in runner.OUTFITS:
        for rotation in k1:
            pair = [row for row in tasks if row["held_out_garment"] == outfit and row["rotation"] == rotation]
            one = next(row for row in pair if row["K"] == 1)
            two = next(row for row in pair if row["K"] == 2)
            assert set(one["selected_adaptation_conditions"]) <= set(two["selected_adaptation_conditions"])


def test_optimizer_and_full_residual_schema_are_exact() -> None:
    values = runner.contracts()
    budget = values["execution"]["optimizer_budget"]
    assert budget["optimizer"] == "torch.optim.Adam"
    assert budget["learning_rate"] == 0.02
    assert budget["betas"] == [0.9, 0.999]
    assert budget["epsilon"] == 1e-8
    assert budget["weight_decay"] == 0
    assert budget["steps"] == 300
    assert budget["checkpoints"] == [0, 20, 50, 100, 200, 300]
    full = values["optimizer"]["methods"]["FEW_VIEW_FULL_RESIDUAL_OPTIMIZATION"]
    assert full["gaussian_count"] == 200_000
    assert full["scalars_per_gaussian"] == 22
    assert full["trainable_scalar_count"] == 4_400_000


def test_regularization_requires_basis_garment_only_runtime_calibration() -> None:
    selection = runner.regularization_selection()
    grids = selection["candidate_grids"]
    assert selection["selection_status"] == "RUNTIME_CALIBRATION_REQUIRED_BEFORE_FORMAL_OPTIMIZATION"
    assert len(grids["low_dimensional"]["coefficient_anchor_lambda"]) == 4
    assert len(grids["low_dimensional"]["trust_region_radius_standardized_l2"]) == 3
    assert len(grids["full_residual"]["residual_delta_lambda"]) == 4
    assert len(grids["full_residual"]["trust_region_radius_bound_normalized_l2_per_gaussian"]) == 3
    assert selection["held_out_garment_information_used"] == 0
    assert selection["test_used"] is False


def test_regularization_candidate_selection_uses_frozen_tie_break() -> None:
    rows = [
        {"candidate_id": "a", "mean_calibration_common_rendering_loss": 0.2, "trust_region_radius": 1.0, "coefficient_anchor_lambda": 0.1, "identity_contamination_count": 0},
        {"candidate_id": "b", "mean_calibration_common_rendering_loss": 0.1, "trust_region_radius": 4.0, "coefficient_anchor_lambda": 0.0, "identity_contamination_count": 0},
        {"candidate_id": "c", "mean_calibration_common_rendering_loss": 0.1, "trust_region_radius": 2.0, "coefficient_anchor_lambda": 0.01, "identity_contamination_count": 0},
        {"candidate_id": "d", "mean_calibration_common_rendering_loss": 0.1, "trust_region_radius": 2.0, "coefficient_anchor_lambda": 0.1, "identity_contamination_count": 0},
        {"candidate_id": "e", "mean_calibration_common_rendering_loss": 0.01, "trust_region_radius": 1.0, "coefficient_anchor_lambda": 0.1, "identity_contamination_count": 1},
    ]
    selected = runner.select_regularization_candidate(rows, "coefficient_anchor_lambda")
    assert selected["candidate_id"] == "d"


def test_recovery_ratio_preserves_nonpositive_denominator_as_null() -> None:
    zero = runner.recovery_ratio(0.1, 0.08, 0.1)
    negative = runner.recovery_ratio(0.1, 0.08, 0.11)
    positive = runner.recovery_ratio(0.1, 0.07, 0.06)
    assert zero["value"] is None
    assert negative["value"] is None
    assert zero["reason"] == "HARD_LOOKUP_ERROR_MINUS_FULL_RESIDUAL_ERROR_NONPOSITIVE"
    assert positive["value"] == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"capacity_limited": True, "hard_lookup_gate_pass": False, "recovery_pass": False, "scaling_pass": False, "efficiency_pass": False, "safety_pass": False}, "LOO_BASIS_CAPACITY_LIMITED"),
        ({"capacity_limited": False, "hard_lookup_gate_pass": False, "recovery_pass": False, "scaling_pass": True, "efficiency_pass": True, "safety_pass": True}, "LOO_HARD_LOOKUP_NOT_OUTPERFORMED"),
        ({"capacity_limited": False, "hard_lookup_gate_pass": True, "recovery_pass": False, "scaling_pass": True, "efficiency_pass": True, "safety_pass": True}, "LOO_OPTIMIZATION_LIMITED"),
        ({"capacity_limited": False, "hard_lookup_gate_pass": True, "recovery_pass": True, "scaling_pass": True, "efficiency_pass": True, "safety_pass": True}, "LOO_BASIS_ADAPTATION_SUPPORTED"),
        ({"capacity_limited": False, "hard_lookup_gate_pass": True, "recovery_pass": True, "scaling_pass": False, "efficiency_pass": True, "safety_pass": True}, "LOO_BASIS_ADAPTATION_PARTIAL"),
    ],
)
def test_classification_priority(kwargs: dict[str, bool], expected: str) -> None:
    assert runner.classify_results(**kwargs) == expected


def test_output_path_plan_is_collision_free() -> None:
    audit = output.audit_relative_paths(runner.planned_paths())
    assert audit["status"] == "PASS"
    assert audit["collision_count"] == 0
    assert audit["path_count"] == 120 * 2 + 720 + 40 + len(runner.PHASES) + 26


def test_wide_result_table_frozen_denominator_is_65() -> None:
    counts = runner.expected_counts()
    assert counts["primary_task_table_rows"] == 40
    assert counts["K2_minus_K1_scaling_rows"] == 20
    assert counts["oracle_capacity_rows"] == 5
    assert counts["wide_result_table_rows"] == 65


def test_atomic_output_rejects_overwrite_and_escape(tmp_path: Path) -> None:
    attempt = tmp_path / runner.OUTPUT_NAME / runner.ATTEMPT_NAME
    attempt.mkdir(parents=True)
    target = attempt / "00_preflight/test.json"
    output.atomic_write_json(attempt, target, {"status": "PASS"})
    with pytest.raises(FileExistsError):
        output.atomic_write_json(attempt, target, {"status": "DIFFERENT"})
    with pytest.raises(output.LOOOutputError):
        output.atomic_write_text(attempt, tmp_path / "escape.txt", "forbidden")
    assert output.temporary_file_leaks(attempt) == []


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"x": 1, "x": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        runner.read_json(path)


def test_credential_scan_reports_only_redacted_hits(tmp_path: Path) -> None:
    path = tmp_path / "leak.txt"
    path.write_text("Bearer " + "a" * 30, encoding="utf-8")
    result = runner.credential_scan((tmp_path,))
    assert result["status"] == "FAIL"
    assert result["credential_value_hits"] == 1
    assert result["hits"][0]["value"] == "REDACTED"
    assert "a" * 30 not in json.dumps(result)


def test_credential_scan_audits_but_ignores_explicit_fake_fixture(tmp_path: Path) -> None:
    path = tmp_path / "fixture.txt"
    path.write_text("Authorization: Bearer deliberately_fake_secret_value_123", encoding="utf-8")
    result = runner.credential_scan((tmp_path,))
    assert result["status"] == "PASS"
    assert result["credential_value_hits"] == 0
    assert result["ignored_explicit_fixture_hits"] == 1
    assert "deliberately_fake_secret_value_123" not in json.dumps(result)
