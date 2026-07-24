from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.paper import run_coefficient_headroom_experiment as runner


class CoefficientHeadroomExperimentTests(unittest.TestCase):
    def test_expected_execution_counts_are_exact(self) -> None:
        counts = runner.expected_counts()
        self.assertEqual(counts["optimization_runs"], 120)
        self.assertEqual(counts["optimizer_steps"], 36_000)
        self.assertEqual(counts["checkpoint_writes"], 960)
        self.assertEqual(counts["renderer_calls"], 36_662)
        self.assertEqual(counts["evaluation_physical_renders"], 660)
        self.assertEqual(counts["visual_sheets"], 20)
        self.assertTrue(all(value is not None for value in counts.values()))

    def test_frozen_lambda_and_milestone_contract(self) -> None:
        self.assertEqual(runner.POSITIVE_LAMBDAS, (1e-4, 1e-3, 1e-2, 1e-1))
        self.assertEqual(runner.ALL_LAMBDAS, (0.0, 1e-4, 1e-3, 1e-2, 1e-1))
        self.assertEqual(runner.MILESTONES, (0, 20, 50, 100, 150, 200, 250, 300))

    def test_rotations_have_no_partition_leakage(self) -> None:
        for rotation in runner.rotations():
            partitions = rotation["partitions"]
            optimize = set(partitions["optimize"]["observation_ids"])
            calibration = set(partitions["calibration"]["observation_ids"])
            test = set(partitions["test"]["observation_ids"])
            self.assertFalse(optimize & calibration)
            self.assertFalse(optimize & test)
            self.assertFalse(calibration & test)
            self.assertEqual(
                set(rotation["optimize_folds"])
                | {rotation["calibration_fold"], rotation["test_fold"]},
                set(runner.CONDITIONS),
            )

    def test_model_scalar_contracts(self) -> None:
        optimizer = runner.optimizer_contract()
        full = runner.full_contract()
        self.assertEqual(optimizer["coefficient_parameterization"]["degrees_of_freedom"], 4)
        self.assertEqual(full["implementation"]["allocated_trainable_scalars"], 2_600_000)
        self.assertEqual(full["implementation"]["effective_masked_trainable_scalars"], 2_217_111)
        self.assertEqual(full["implementation"]["other_trainable_scalars"], 0)

    def test_lambda_tie_break_prefers_stronger_regularization(self) -> None:
        selected, audit = runner.select_lambda({
            1e-4: 1.0,
            1e-3: 1.00009,
            1e-2: 1.2,
            1e-1: 1.4,
        })
        self.assertEqual(selected, 1e-3)
        self.assertEqual(audit["tie_tolerance_absolute"], 1e-4)
        self.assertIn("stronger", audit["tie_break"])

    def test_equal_wall_time_selects_largest_fitting_checkpoint(self) -> None:
        times = {0: 0.0, 20: 2.0, 50: 5.0, 100: 10.0, 150: 15.0, 200: 20.0, 250: 25.0, 300: 30.0}
        selected = runner.select_equal_wall_time_checkpoint(18.0, times)
        self.assertEqual(selected["selected_step"], 150)
        self.assertEqual(selected["time_delta_seconds"], 3.0)
        self.assertFalse(selected["interpolation_used"])
        self.assertFalse(selected["test_metric_used"])

    def test_nonpositive_span_denominator_is_null_with_reason(self) -> None:
        zero = runner.span_recovery_ratio(0.2, 0.19, 0.2)
        negative = runner.span_recovery_ratio(0.2, 0.19, 0.21)
        for result in (zero, negative):
            self.assertIsNone(result["value"])
            self.assertEqual(result["reason"], "Teacher_error-FullResidual_error<=0")
        valid = runner.span_recovery_ratio(0.2, 0.18, 0.1)
        self.assertAlmostEqual(valid["value"], 0.2)

    def test_classification_decision_order(self) -> None:
        useful = {
            "complete": True,
            "identity_contamination_count": 0,
            "component_contamination_count": 0,
            "improved_garments": 4,
            "macro_lpips_improvement_absolute": 0.006,
            "macro_lpips_relative_reduction": 0.06,
            "companion_gate": True,
            "valid_per_garment_span_ratios": 3,
            "macro_span_recovery_ratio": 0.3,
            "held_out_test_gate": True,
            "full_residual_improved_garments": 5,
            "full_residual_macro_lpips_improvement_absolute": 0.02,
            "full_residual_macro_lpips_relative_reduction": 0.2,
        }
        self.assertEqual(
            runner.classify_headroom(useful),
            "TEACHER_SPAN_HAS_USEFUL_RENDER_HEADROOM",
        )
        incomplete = dict(useful, complete=False)
        self.assertEqual(
            runner.classify_headroom(incomplete),
            "COEFFICIENT_HEADROOM_PROTOCOL_INCOMPLETE",
        )
        capacity = dict(
            useful,
            improved_garments=2,
            macro_lpips_improvement_absolute=-0.001,
            macro_lpips_relative_reduction=-0.01,
        )
        self.assertEqual(
            runner.classify_headroom(capacity),
            "TEACHER_SPAN_CAPACITY_LIMITED",
        )

    def test_pure_prediction_sha_is_frozen(self) -> None:
        self.assertEqual(
            runner.PURE_PREDICTION_SHA,
            "13e685087686d0ad597fd67daaad328f07910bceba33f46c6630623ba6e21290",
        )
        self.assertEqual(runner.PURE_HEAD, "ce110887a942cf8db082ba688c8d36d2433bfdbe")

    def test_paper_final_remains_false(self) -> None:
        self.assertFalse(runner.success_contract()["paper_final"])
        self.assertEqual(runner.expected_counts()["paper_final_count"], 0)

    def test_credential_scan_only_ignores_explicit_fake_fixture(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = Path(directory) / "fixture.txt"
            fixture.write_text(
                "Authorization: " + "Bearer deliberately_fake_secret_value_123\n",
                encoding="utf-8",
            )
            self.assertEqual(runner.credential_scan([fixture])["status"], "PASS")
            fixture.write_text(
                "Authorization: " + "Bearer plausible_nonfixture_value_123456\n",
                encoding="utf-8",
            )
            scan = runner.credential_scan([fixture])
            self.assertEqual(scan["status"], "FAIL")
            self.assertEqual(scan["finding_count"], 1)


if __name__ == "__main__":
    unittest.main()
