from __future__ import annotations

from tools.paper import run_controller_garment_budget_diagnosis as diagnosis


def test_six_family_and_execution_counts_are_frozen() -> None:
    assert len(diagnosis.FAMILIES) == 6
    assert diagnosis.EXPECTED_COUNTS["run_trajectories"] == 72
    assert diagnosis.EXPECTED_COUNTS["optimizer_steps"] == 39600
    assert diagnosis.EXPECTED_COUNTS["new_checkpoint_writes"] == 504
    assert diagnosis.EXPECTED_COUNTS["checkpoint_inference_count"] == 207360
    assert diagnosis.EXPECTED_COUNTS["renderer_runs"] == 0


def test_600_step_schedule_repeats_the_frozen_cycle() -> None:
    contract = diagnosis.contract_inputs()
    archive = diagnosis.build_schedules(contract)
    assert archive["batches_per_cycle"] == 32
    assert archive["steps"] == 600
    assert archive["full_cycles"] == 18
    assert archive["partial_cycle_batches"] == 24
    for row in archive["rotations"]:
        assert len(row["schedule"]["steps"]) == 600
        assert row["exposure_distribution"]["histogram"] == {"18": 40, "19": 120}
        assert row["exposure_distribution"]["clean_record_exposures"] == 3000


def test_parameter_memberships_match_unmodified_model_graph() -> None:
    paths = diagnosis.v2_static_paths()
    assert set(paths["garment"]) & set(paths["mixedness"]) == {
        "reference_normalization.bias",
        "reference_normalization.weight",
    }
    assert set(paths["garment"]) & set(paths["pair_weight"]) == {
        "reference_normalization.bias",
        "reference_normalization.weight",
    }
    assert diagnosis.LOSS_DEFINITIONS["GARMENT_ONLY"] == "L_garment"
    assert "L_garment_consistency" not in diagnosis.LOSS_DEFINITIONS[
        "FULL_NO_GARMENT_CONSISTENCY"
    ]


def test_primary_classifications_and_next_tasks_are_total() -> None:
    assert len(diagnosis.PRIMARY_CLASSIFICATIONS) == 7
    assert all(diagnosis.next_task(value) for value in diagnosis.PRIMARY_CLASSIFICATIONS)
    assert diagnosis.HISTORICAL_CLASSIFICATION == "CONTROLLER_V2_CROSSFIT_MICRO_PILOT_FAIL"
    assert diagnosis.SEMANTIC_CLASSIFICATION == "LATENT_SECONDARY_NOT_INPUT_IDENTIFIABLE"
