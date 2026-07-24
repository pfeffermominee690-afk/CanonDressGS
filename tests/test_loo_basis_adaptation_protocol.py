from __future__ import annotations

import json
import hashlib
from pathlib import Path
import re
import subprocess
import unittest

try:
    import yaml
except ModuleNotFoundError:  # Windows control Python omits PyYAML; cloud runtime parses fully.
    yaml = None


ROOT = Path(__file__).resolve().parents[1]
SOURCE_HEAD = "ff56ebfaf7b733adbb41799e248d01d0e7801ea8"
OUTFITS = ["O01", "O02", "O03", "O04", "O08"]
JSON_PATHS = [
    "paper_protocol/reviewer_risk/loo_basis_manifests.json",
    "paper_protocol/reviewer_risk/loo_few_view_manifests.json",
    "paper_protocol/reviewer_risk/loo_adaptation_loss_contract.json",
    "paper_protocol/reviewer_risk/loo_optimizer_contract.json",
    "paper_protocol/reviewer_risk/loo_baseline_registry.json",
    "paper_protocol/reviewer_risk/loo_evaluator_contract.json",
    "paper_protocol/reviewer_risk/loo_success_gates.json",
    "paper_protocol/reviewer_risk/loo_protocol_final_summary.json",
    "project_control_handoff/loo_basis_adaptation_protocol_handoff.json",
]
MARKDOWN_PATHS = [
    "docs/PAPER/AAAI27_LEAVE_ONE_GARMENT_OUT_BASIS_PROTOCOL_20260724.md",
    "docs/PAPER/AAAI27_FEW_VIEW_GARMENT_ADAPTATION_PROTOCOL_20260724.md",
    "docs/PAPER/AAAI27_HARD_LOOKUP_FAIR_COMPARISON_PROTOCOL_20260724.md",
    "docs/PAPER/AAAI27_LOO_BASIS_CAPACITY_ANALYSIS_PLAN_20260724.md",
]
ALLOWED_PATHS = set(JSON_PATHS + MARKDOWN_PATHS + [
    "paper_protocol/reviewer_risk/loo_basis_adaptation_protocol.yaml",
    "tests/test_loo_basis_adaptation_protocol.py",
])
ALLOWED_PATHS.update({
    "docs/PAPER/AAAI27_LOO_FEW_VIEW_FOLD_REPAIR_20260724.md",
    "docs/PAPER/AAAI27_LOO_K1_K2_PRIMARY_CROSSFIT_PROTOCOL_20260724.md",
    "docs/PAPER/AAAI27_LOO_K4_DEFERRED_DIAGNOSTIC_BOUNDARY_20260724.md",
    "paper_protocol/reviewer_risk/loo_few_view_fold_repair.json",
    "paper_protocol/reviewer_risk/loo_few_view_manifests_repaired.json",
    "paper_protocol/reviewer_risk/loo_basis_adaptation_protocol_amended.yaml",
    "paper_protocol/reviewer_risk/loo_shared_render_adaptation_loss_reference.json",
    "paper_protocol/reviewer_risk/loo_baseline_registry_amended.json",
    "paper_protocol/reviewer_risk/loo_evaluator_contract_amended.json",
    "paper_protocol/reviewer_risk/loo_success_gates_amended.json",
    "paper_protocol/reviewer_risk/loo_expected_counts_amended.json",
    "paper_protocol/reviewer_risk/loo_execution_contract_amended.json",
    "paper_protocol/reviewer_risk/loo_protocol_repair_tests.json",
    "paper_protocol/reviewer_risk/loo_protocol_repair_final_summary.json",
    "project_control_handoff/loo_protocol_repair_handoff.json",
    "tests/test_loo_few_view_fold_manifest_repair.py",
})
ALLOWED_PATHS.update({
    "tools/paper/loo_basis_output.py",
    "tools/paper/run_loo_basis_adaptation_experiment.py",
    "tests/test_loo_basis_adaptation_experiment.py",
    "paper_protocol/reviewer_risk/loo_execution_binding.json",
    "paper_protocol/reviewer_risk/loo_execution_expected_counts.json",
    "paper_protocol/reviewer_risk/loo_pre_result_tests.json",
    "paper_protocol/reviewer_risk/loo_basis_execution_registry.json",
    "paper_protocol/reviewer_risk/loo_adaptation_run_registry.json",
    "paper_protocol/reviewer_risk/loo_checkpoint_registry.json",
    "paper_protocol/reviewer_risk/loo_prediction_registry.json",
    "paper_protocol/reviewer_risk/loo_metric_summary.json",
    "paper_protocol/reviewer_risk/loo_oracle_capacity_analysis.json",
    "paper_protocol/reviewer_risk/loo_hard_lookup_analysis.json",
    "paper_protocol/reviewer_risk/loo_view_budget_scaling.json",
    "paper_protocol/reviewer_risk/loo_full_residual_summary.json",
    "paper_protocol/reviewer_risk/loo_visual_review_summary.json",
    "paper_protocol/reviewer_risk/loo_execution_count_verification.json",
    "paper_protocol/reviewer_risk/loo_tests.json",
    "paper_protocol/reviewer_risk/loo_final_summary.json",
    "paper_protocol/reviewer_risk/sealed_loo_figure_refresh_manifest.json",
    "docs/PAPER/AAAI27_LOO_BASIS_ADAPTATION_RESULTS_20260724.md",
    "docs/PAPER/AAAI27_LOO_HARD_LOOKUP_COMPARISON_20260724.md",
    "docs/PAPER/AAAI27_LOO_BASIS_CAPACITY_ANALYSIS_20260724.md",
    "docs/PAPER/AAAI27_LOO_FEW_VIEW_SCALING_RESULTS_20260724.md",
    "docs/PAPER/AAAI27_LOO_FULL_RESIDUAL_COMPARISON_20260724.md",
    "docs/PAPER/AAAI27_LOO_VISUAL_REVIEW_20260724.md",
    "docs/PAPER/AAAI27_LOO_FAILURE_ANALYSIS_20260724.md",
    "project_control_handoff/loo_basis_adaptation_experiment_handoff.json",
})


def load_json(relative: str):
    with (ROOT / relative).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def assert_no_nulls(testcase: unittest.TestCase, value, path: str = "$"):
    if value is None:
        testcase.fail(f"null value at {path}")
    if isinstance(value, dict):
        for key, child in value.items():
            assert_no_nulls(testcase, child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_no_nulls(testcase, child, f"{path}[{index}]")


class TestLooBasisAdaptationProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.basis = load_json(JSON_PATHS[0])
        cls.views = load_json(JSON_PATHS[1])
        cls.loss = load_json(JSON_PATHS[2])
        cls.optimizer = load_json(JSON_PATHS[3])
        cls.baselines = load_json(JSON_PATHS[4])
        cls.evaluator = load_json(JSON_PATHS[5])
        cls.gates = load_json(JSON_PATHS[6])
        cls.summary = load_json(JSON_PATHS[7])
        cls.handoff = load_json(JSON_PATHS[8])

    def test_exact_source_head_and_ancestry(self):
        self.assertEqual(self.summary["source"]["head"], SOURCE_HEAD)
        self.assertEqual(self.basis["source"]["head"], SOURCE_HEAD)
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", SOURCE_HEAD, "HEAD"],
            cwd=ROOT, check=True,
        )

    def test_five_loo_splits_and_exclusion(self):
        self.assertEqual(len(self.basis["splits"]), 5)
        self.assertEqual(
            [split_["held_out_garment"] for split_ in self.basis["splits"]],
            OUTFITS,
        )
        for split_ in self.basis["splits"]:
            held_out = split_["held_out_garment"]
            basis_garments = split_["basis_garments"]
            self.assertEqual(len(basis_garments), 4)
            self.assertNotIn(held_out, basis_garments)
            self.assertEqual(set(basis_garments), set(OUTFITS) - {held_out})
            self.assertNotIn(held_out, split_["basis_teacher_assets"])
            self.assertTrue(all(split_["held_out_exclusion"].values()))
            self.assertEqual(
                split_["held_out_teacher_asset"]["deployable_information_use_count"], 0
            )

    def test_basis_mean_rank_normalization_and_no_full_basis_reuse(self):
        for split_ in self.basis["splits"]:
            contract = split_["basis_contract"]
            self.assertEqual(contract["rank_rule"], "min(numerical_rank,3)")
            self.assertEqual(contract["affine_rank_upper_bound"], 3)
            self.assertFalse(contract["full_five_garment_basis_reused"])
            self.assertFalse(contract["runtime_basis_artifact_created_in_this_task"])
            self.assertFalse(split_["coefficient_contract"]["held_out_in_normalization"])
            self.assertEqual(split_["static_audit"]["svd_execution_runs"], 0)
            self.assertEqual(split_["static_audit"]["binary_basis_writes"], 0)
            self.assertRegex(contract["basis_hash"], r"^[0-9a-f]{64}$")
        self.assertEqual(self.basis["counts"]["held_out_teacher_in_basis"], 0)

    def test_sixty_task_records_and_feasibility_audit(self):
        tasks = self.views["tasks"]
        self.assertEqual(len(tasks), 60)
        self.assertEqual(len({task["task_id"] for task in tasks}), 60)
        valid = [task for task in tasks if task["status"] == "MANIFEST_VALID_PROTOCOL_ONLY"]
        blocked = [task for task in tasks if task["status"] == "BLOCKED_VIEW_FOLD_CONFLICT"]
        self.assertEqual(len(valid), 15)
        self.assertEqual(len(blocked), 45)
        self.assertEqual(self.views["counts"]["planned_tasks"], 60)
        self.assertEqual(self.views["counts"]["manifest_valid_tasks"], 15)
        self.assertEqual(self.views["counts"]["blocked_tasks"], 45)

    def test_k_semantic_views_are_exact(self):
        budgets = self.views["budgets"]
        self.assertEqual(budgets["1"]["exact_view_ids"], ["cond_000000"])
        self.assertEqual(
            budgets["2"]["exact_view_ids"], ["cond_000000", "cond_000318"]
        )
        self.assertEqual(
            budgets["4"]["exact_view_ids"],
            ["cond_000000", "cond_000017", "cond_000318", "cond_000347"],
        )
        self.assertEqual(
            self.views["semantic_slot_manifest"]["mapping"],
            {
                "front": "cond_000000",
                "back": "cond_000318",
                "left": "cond_000017",
                "right": "cond_000347",
            },
        )

    def test_adaptation_test_disjointness_is_enforced_not_fabricated(self):
        for task in self.views["tasks"]:
            self.assertRegex(task["basis_hash"], r"^[0-9a-f]{64}$")
            self.assertRegex(task["query_order_sha256"], r"^[0-9a-f]{64}$")
            encoded = json.dumps(
                task["query_order"], ensure_ascii=True, separators=(",", ":")
            ).encode("ascii")
            self.assertEqual(task["query_order_sha256"], hashlib.sha256(encoded).hexdigest())
            if task["status"] == "MANIFEST_VALID_PROTOCOL_ONLY":
                self.assertTrue(task["all_adaptation_views_in_train"])
                self.assertTrue(task["adaptation_test_disjoint"])
                self.assertTrue(task["adaptation_calibration_disjoint"])
                self.assertEqual(task["blockers"], [])
            else:
                self.assertTrue(task["blockers"])
                self.assertFalse(
                    task["all_adaptation_views_in_train"],
                    msg=f"blocked task lacks train-fold conflict: {task['task_id']}",
                )

    def test_no_held_out_teacher_in_deployable_adaptation(self):
        for task in self.views["tasks"]:
            use = task["held_out_teacher_information_use"]
            self.assertEqual(use["deployable_adaptation"], 0)
        self.assertEqual(self.loss["oracle_separation"]["held_out_teacher_forward_count"], 0)
        self.assertEqual(self.loss["oracle_separation"]["held_out_teacher_loss_count"], 0)
        self.assertFalse(self.loss["oracle_separation"]["oracle_projection_in_loss"])

    def test_hard_lookup_has_four_garments_and_oracles_are_offline(self):
        self.assertEqual(self.baselines["shared_contract"]["basis_bank_size"], 4)
        self.assertFalse(self.baselines["shared_contract"]["held_out_garment_in_bank"])
        methods = {method["id"]: method for method in self.baselines["methods"]}
        hard = methods["REFERENCE_NEAREST_HARD_LOOKUP"]
        self.assertFalse(hard["can_create_new_coefficient"])
        self.assertTrue(hard["deployable"])
        for name in ["RESIDUAL_NEAREST_ORACLE", "ORACLE_PROJECTION_LOO_BASIS"]:
            self.assertFalse(methods[name]["deployable"])
        self.assertEqual(methods["REFERENCE_PREDICTOR_TRANSFER"]["status"], "NOT_APPLICABLE")

    def test_low_dimensional_count_and_full_residual_fairness(self):
        low = self.optimizer["methods"]["FEW_VIEW_LOW_DIMENSIONAL_ADAPTATION"]
        full = self.optimizer["methods"]["FEW_VIEW_FULL_RESIDUAL_OPTIMIZATION"]
        self.assertLessEqual(low["trainable_scalar_count_max"], 3)
        self.assertEqual(full["trainable_scalar_count"], 4_400_000)
        self.assertTrue(full["same_steps"])
        self.assertFalse(full["extra_views"])
        self.assertFalse(full["held_out_teacher_used"])
        self.assertEqual(
            low["loss_contract"], "LOW_DIMENSIONAL_RENDER_ADAPTATION_LOSS_CONTRACT"
        )
        self.assertEqual(low["loss_contract"], full["loss_contract"])
        budget = self.optimizer["shared_budget"]
        self.assertEqual(budget["steps"], 300)
        self.assertEqual(budget["checkpoints"], [0, 20, 50, 100, 200, 300])
        self.assertEqual(budget["retry_count"], 0)
        self.assertEqual(budget["final_rule"], "step 300 only")

    def test_shared_loss_and_calibration_only_regularization(self):
        self.assertEqual(
            self.loss["formal_name"], "LOW_DIMENSIONAL_RENDER_ADAPTATION_LOSS_CONTRACT"
        )
        expected = {
            "garment_rgb_l1", "lpips", "mask_l1", "boundary_rgb_l1",
            "identity_rgb_l1", "identity_alpha_l1", "regularization",
        }
        self.assertEqual(set(self.loss["components"]), expected)
        selection = self.loss["regularization_selection"]
        self.assertEqual(selection["held_out_garment_information_used"], 0)
        self.assertFalse(selection["test_used"])
        self.assertFalse(self.loss["sharing"]["per_loo_task_tuning"])

    def test_evaluator_and_numeric_success_gates(self):
        render = self.evaluator["render_metrics"]
        self.assertEqual(
            set(render),
            {
                "rgb_mae", "psnr", "ssim", "lpips", "silhouette_iou",
                "boundary_f", "protected_lpips", "identity",
            },
        )
        self.assertEqual(self.evaluator["denominator"]["planned_adaptation_tasks"], 60)
        hard = self.gates["hard_lookup_outperformance"]
        self.assertEqual(hard["K"], 4)
        self.assertEqual(hard["minimum_successful_garments"], 4)
        self.assertEqual(hard["successful_rotations_per_garment_min"], 3)
        self.assertGreater(hard["per_rotation_lpips_absolute_reduction_min"], 0)
        self.assertGreater(hard["per_rotation_rgb_mae_absolute_reduction_min"], 0)
        self.assertEqual(self.gates["efficiency"]["low_dimensional_trainable_scalars_max"], 3)
        self.assertEqual(self.gates["full_residual_comparison"]["low_dimensional_gain_recovery_ratio_min"], 0.75)
        self.assertTrue(self.gates["frozen_before_result"])

    def test_execution_counts_are_zero_and_paper_final_absent(self):
        expected_keys = {
            "basis_execution_runs", "adaptation_runs", "optimizer_created",
            "forward", "backward", "checkpoint_writes", "renderer_runs",
            "new_renders", "paper_final",
        }
        counts = self.summary["execution_counts"]
        self.assertEqual(set(counts), expected_keys)
        self.assertTrue(all(value == 0 for value in counts.values()))
        self.assertFalse(self.summary["paper_final"])
        self.assertEqual(self.summary["paper_final_count"], 0)
        self.assertFalse(self.optimizer["execution_authorized"])

    def test_final_classification_and_next_task(self):
        self.assertEqual(self.summary["final_classification"], "LOO_PROTOCOL_INCOMPLETE")
        self.assertEqual(self.handoff["classification"], "LOO_PROTOCOL_INCOMPLETE")
        self.assertEqual(self.summary["next_task"], "REPAIR_LOO_FEW_VIEW_FOLD_MANIFEST")
        self.assertFalse(self.summary["run_experiment_next_task_started"])
        self.assertFalse(self.handoff["auto_execute_next_task"])

    def test_json_yaml_parse_and_no_nulls(self):
        for path in JSON_PATHS:
            value = load_json(path)
            assert_no_nulls(self, value, path)
        yaml_path = ROOT / "paper_protocol/reviewer_risk/loo_basis_adaptation_protocol.yaml"
        yaml_text = yaml_path.read_text(encoding="utf-8")
        self.assertNotIn("\t", yaml_text)
        if yaml is not None:
            protocol = yaml.safe_load(yaml_text)
            self.assertEqual(protocol["final_classification"], "LOO_PROTOCOL_INCOMPLETE")
            self.assertEqual(protocol["execution_boundary"]["optimizer_created"], 0)
        else:
            self.assertIn('final_classification: "LOO_PROTOCOL_INCOMPLETE"', yaml_text)
            self.assertIn("  optimizer_created: 0", yaml_text)

    def test_markdown_reports_are_nonempty(self):
        for relative in MARKDOWN_PATHS:
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertGreater(len(text.strip()), 500)
            self.assertNotIn("PAPER_FINAL", Path(relative).name)

    def test_frozen_mutation_scope(self):
        changed = subprocess.run(
            ["git", "diff", "--name-only", SOURCE_HEAD, "--"],
            cwd=ROOT, check=True, text=True, capture_output=True,
        ).stdout.splitlines()
        unexpected = sorted(set(changed) - ALLOWED_PATHS)
        self.assertEqual(unexpected, [], f"unexpected frozen mutation paths: {unexpected}")


if __name__ == "__main__":
    unittest.main()
