from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
SOURCE_HEAD = "ff56ebfaf7b733adbb41799e248d01d0e7801ea8"
OUTFITS = ["O01", "O02", "O03", "O04", "O08"]
CONDITIONS = ["cond_000000", "cond_000318", "cond_000017", "cond_000347"]
ZERO_FIELDS = {
    "optimization_runs",
    "training_runs",
    "optimizer_creations",
    "optimizer_steps",
    "forward_calls",
    "backward_calls",
    "checkpoint_writes",
    "renderer_runs",
    "new_renders",
}


def read(name: str) -> dict:
    return json.loads((RISK / name).read_text(encoding="utf-8"))


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


class CoefficientHeadroomProtocolTests(unittest.TestCase):
    def test_required_artifacts_exist(self) -> None:
        required = [
            "docs/PAPER/AAAI27_RENDER_REFINED_COEFFICIENT_HEADROOM_PROTOCOL_20260724.md",
            "docs/PAPER/AAAI27_TEACHER_ENDPOINT_HEADROOM_ANALYSIS_PLAN_20260724.md",
            "docs/PAPER/AAAI27_FULL_RESIDUAL_FAIR_COMPARISON_PROTOCOL_20260724.md",
            "paper_protocol/reviewer_risk/coefficient_headroom_protocol.yaml",
            "paper_protocol/reviewer_risk/coefficient_headroom_rotation_manifests.json",
            "paper_protocol/reviewer_risk/coefficient_headroom_loss_contract.json",
            "paper_protocol/reviewer_risk/coefficient_headroom_optimizer_contract.json",
            "paper_protocol/reviewer_risk/coefficient_headroom_full_residual_contract.json",
            "paper_protocol/reviewer_risk/coefficient_headroom_evaluator_contract.json",
            "paper_protocol/reviewer_risk/coefficient_headroom_success_gates.json",
            "paper_protocol/reviewer_risk/coefficient_headroom_protocol_final_summary.json",
            "project_control_handoff/coefficient_headroom_protocol_handoff.json",
        ]
        self.assertTrue(all((ROOT / path).is_file() for path in required))
        summary = read("coefficient_headroom_protocol_final_summary.json")
        for relative, expected in summary["artifact_sha256_lf"].items():
            actual = sha256_lf(ROOT / relative)
            self.assertEqual(actual, expected, relative)
        handoff = json.loads(
            (ROOT / "project_control_handoff/coefficient_headroom_protocol_handoff.json").read_text(encoding="utf-8")
        )
        summary_path = ROOT / handoff["summary"]["path"]
        self.assertEqual(sha256_lf(summary_path), handoff["summary"]["sha256_lf"])

    def test_exact_source_and_closed_scope(self) -> None:
        manifest = read("coefficient_headroom_rotation_manifests.json")
        self.assertEqual(manifest["source_head"], SOURCE_HEAD)
        self.assertEqual(manifest["identity"], "subject02")
        self.assertEqual(manifest["outfit_order"], OUTFITS)
        self.assertEqual(manifest["condition_order"], CONDITIONS)
        self.assertEqual(len(manifest["target_observations"]), 20)

    def test_rotations_are_disjoint_and_complete(self) -> None:
        manifest = read("coefficient_headroom_rotation_manifests.json")
        self.assertEqual(len(manifest["rotations"]), 4)
        observed_tests = []
        for rotation in manifest["rotations"]:
            partitions = rotation["partitions"]
            optimize = set(partitions["optimize"]["observation_ids"])
            calibration = set(partitions["calibration"]["observation_ids"])
            test = set(partitions["test"]["observation_ids"])
            self.assertEqual((len(optimize), len(calibration), len(test)), (10, 5, 5))
            self.assertFalse(optimize & calibration)
            self.assertFalse(optimize & test)
            self.assertFalse(calibration & test)
            observed_tests.extend(partitions["test"]["condition_ids"])
            for outfit in OUTFITS:
                query = rotation["per_garment_query_order"][outfit]
                sequence = [
                    f"subject02/{outfit}/{rotation['optimize_folds'][step % 2]}"
                    for step in range(300)
                ]
                digest = hashlib.sha256("\n".join(sequence).encode("utf-8")).hexdigest()
                self.assertEqual(query["sha256"], digest)
        self.assertEqual(observed_tests, ["cond_000347", "cond_000000", "cond_000318", "cond_000017"])

    def test_loss_is_historical_and_lpips_is_not_added(self) -> None:
        contract = read("coefficient_headroom_loss_contract.json")
        weights = {name: row["weight"] for name, row in contract["render_terms"].items()}
        self.assertEqual(
            weights,
            {
                "alpha_foreground": 0.5,
                "boundary_rgb": 0.25,
                "garment_rgb": 1.0,
                "new_silhouette_alpha": 1.0,
                "protected_alpha": 5.0,
                "protected_rgb": 10.0,
            },
        )
        self.assertEqual(contract["audited_but_disabled_terms"]["lpips"]["weight"], 0.0)
        self.assertEqual(contract["primary_coefficient_regularization"]["strategy"], "L2_ANCHOR_ONLY")
        self.assertTrue(all(value > 0 for value in contract["primary_coefficient_regularization"]["lambda_grid"]))
        self.assertFalse(contract["primary_coefficient_regularization"]["test_used"])
        self.assertFalse(contract["unregularized_diagnostic"]["primary"])

    def test_budget_and_selection_are_fixed(self) -> None:
        contract = read("coefficient_headroom_optimizer_contract.json")
        coefficient = contract["coefficient_optimizer"]
        self.assertEqual(coefficient["steps"], 300)
        self.assertEqual(coefficient["checkpoint_steps"], [0, 20, 50, 100, 150, 200, 250, 300])
        self.assertEqual(coefficient["final_rule"], "step_300_only")
        self.assertFalse(contract["data_schedule"]["early_stopping"])
        self.assertFalse(contract["data_schedule"]["best_checkpoint_selection"])
        self.assertFalse(contract["calibration_selection"]["test_used"])
        self.assertEqual(contract["planned_execution"]["total_optimization_runs"], 120)
        self.assertEqual(contract["coefficient_parameterization"]["garment_order"], OUTFITS)
        self.assertEqual(contract["pre_optimizer_initialization_gates"]["basis_rank"], 4)
        self.assertEqual(contract["pre_optimizer_initialization_gates"]["per_garment_bound_normalized_residual_rmse_max"], 1e-5)
        self.assertFalse(contract["checkpoint_schema"]["target_tensors_stored"])

    def test_full_residual_fairness_and_schema(self) -> None:
        contract = read("coefficient_headroom_full_residual_contract.json")
        implementation = contract["implementation"]
        self.assertEqual(implementation["allocated_trainable_scalars"], 2600000)
        self.assertEqual(implementation["effective_masked_trainable_scalars"], 2217111)
        self.assertEqual(implementation["other_trainable_scalars"], 0)
        self.assertEqual(implementation["frozen_schema_tensor"]["value"], "zero")
        self.assertTrue(all(contract["fairness"][name] for name in (
            "same_optimize_views", "same_calibration_views", "same_test_views",
            "same_six_render_terms", "same_identity_protection",
            "same_background_camera_pose_renderer",
        )))
        self.assertFalse(contract["fairness"]["extra_teacher_supervision"])
        self.assertFalse(contract["fairness"]["test_views_in_updates"])
        self.assertFalse(contract["fairness"]["test_views_in_selection"])

    def test_evaluator_and_ratio_denominator(self) -> None:
        evaluator = read("coefficient_headroom_evaluator_contract.json")
        self.assertEqual(evaluator["primary_denominator"]["global"], 20)
        self.assertIn("undefined", evaluator["headroom_definitions"]["span_recovery_denominator_rule"])
        self.assertFalse(evaluator["reporting"]["test_used_for_selection"])
        self.assertEqual(
            evaluator["headroom_definitions"]["coefficient_headroom_gain"],
            "raw metric delta: Render-Refined Coefficient - SVD Endpoint",
        )
        for metric in ("rgb_mae", "psnr", "ssim", "lpips", "silhouette_iou", "boundary_f", "protected_lpips", "identity_metric"):
            self.assertIn(metric, evaluator["render_quality"])

    def test_success_gates_are_numerical_and_frozen(self) -> None:
        gates = read("coefficient_headroom_success_gates.json")
        self.assertEqual(gates["thresholds"]["improved_garments_min"], 4)
        self.assertEqual(gates["thresholds"]["macro_lpips_improvement_absolute_min"], 0.005)
        self.assertEqual(gates["thresholds"]["macro_span_recovery_ratio_min"], 0.25)
        self.assertFalse(gates["thresholds_may_change_after_results"])
        classifications = [row["classification"] for row in gates["decision_order"]]
        self.assertEqual(classifications[0], "COEFFICIENT_HEADROOM_PROTOCOL_INCOMPLETE")
        self.assertEqual(
            set(classifications),
            {
                "TEACHER_SPAN_HAS_USEFUL_RENDER_HEADROOM",
                "TEACHER_SPAN_HEADROOM_SMALL",
                "TEACHER_SPAN_AT_LOCAL_OPTIMUM",
                "TEACHER_SPAN_CAPACITY_LIMITED",
                "COEFFICIENT_HEADROOM_PROTOCOL_INCOMPLETE",
            },
        )

    def test_protocol_task_executed_nothing(self) -> None:
        summary = read("coefficient_headroom_protocol_final_summary.json")
        self.assertEqual(summary["classification"], "COEFFICIENT_HEADROOM_PROTOCOL_READY")
        self.assertEqual(set(summary["actual_execution"]), ZERO_FIELDS)
        self.assertTrue(all(value == 0 for value in summary["actual_execution"].values()))
        self.assertFalse(summary["paper_final"])
        self.assertEqual(summary["paper_final_count"], 0)
        self.assertFalse(summary["next_task_started"])
        self.assertIn("refined_hard_lookup", summary["future_interfaces"])
        optimizer = read("coefficient_headroom_optimizer_contract.json")
        self.assertTrue(all(value == 0 for value in optimizer["actual_execution"].values()))

    def test_no_deprecated_teacher_name_in_new_artifacts(self) -> None:
        paths = [
            *ROOT.glob("docs/PAPER/*COEFFICIENT_HEADROOM*20260724.md"),
            ROOT / "docs/PAPER/AAAI27_TEACHER_ENDPOINT_HEADROOM_ANALYSIS_PLAN_20260724.md",
            ROOT / "docs/PAPER/AAAI27_FULL_RESIDUAL_FAIR_COMPARISON_PROTOCOL_20260724.md",
            *RISK.glob("coefficient_headroom_*"),
            ROOT / "project_control_handoff/coefficient_headroom_protocol_handoff.json",
        ]
        for path in paths:
            self.assertNotIn("Teacher Upper Bound", path.read_text(encoding="utf-8"), path)


if __name__ == "__main__":
    unittest.main()
