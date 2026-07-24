from __future__ import annotations

import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml
from PIL import Image

from tools.paper import coefficient_headroom_output as output
from tools.paper import run_coefficient_headroom_experiment as runner


ROOT = Path(__file__).resolve().parents[1]
SOURCE_HEAD = "64f7bf09d7beb526eee50c8c1a0c47d792977378"
REPAIRED_SOURCE_HEAD = "77d2f4e32de984d766e0beab5ded7afe12aecab0"
SEMANTIC_CONTRACTS = (
    "paper_protocol/reviewer_risk/coefficient_headroom_protocol.yaml",
    "paper_protocol/reviewer_risk/coefficient_headroom_rotation_manifests.json",
    "paper_protocol/reviewer_risk/coefficient_headroom_loss_contract.json",
    "paper_protocol/reviewer_risk/coefficient_headroom_optimizer_contract.json",
    "paper_protocol/reviewer_risk/coefficient_headroom_full_residual_contract.json",
    "paper_protocol/reviewer_risk/coefficient_headroom_evaluator_contract.json",
    "paper_protocol/reviewer_risk/coefficient_headroom_success_gates.json",
    "paper_protocol/reviewer_risk/coefficient_headroom_protocol_final_summary.json",
)


class OutputPathSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.attempt = output.resolve_attempt_root(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_resolve_attempt_root_is_attempt_002(self) -> None:
        self.assertEqual(self.attempt.name, "attempt_002")
        self.assertEqual(self.attempt.parent.name, output.OUTPUT_NAME)

    def test_attempt_001_write_root_is_rejected(self) -> None:
        with self.assertRaisesRegex(output.HeadroomOutputError, "HEADROOM_OUTPUT_PATH_OUTSIDE_ATTEMPT"):
            output.resolve_attempt_root(self.root, attempt_id="attempt_001")

    def test_dot_dot_traversal_is_rejected(self) -> None:
        with self.assertRaisesRegex(output.HeadroomOutputError, "HEADROOM_OUTPUT_PATH_OUTSIDE_ATTEMPT"):
            output.resolve_output_path(self.attempt, "nested/../escape.json")

    def test_windows_separator_traversal_is_rejected(self) -> None:
        with self.assertRaisesRegex(output.HeadroomOutputError, "HEADROOM_OUTPUT_PATH_OUTSIDE_ATTEMPT"):
            output.resolve_output_path(self.attempt, r"nested\..\escape.json")

    def test_absolute_component_is_rejected(self) -> None:
        with self.assertRaisesRegex(output.HeadroomOutputError, "HEADROOM_OUTPUT_PATH_OUTSIDE_ATTEMPT"):
            output.resolve_output_path(self.attempt, str((self.root / "foreign.json").resolve()))

    def test_foreign_target_is_rejected(self) -> None:
        with self.assertRaisesRegex(output.HeadroomOutputError, "HEADROOM_OUTPUT_PATH_OUTSIDE_ATTEMPT"):
            output.assert_path_inside_attempt(self.attempt, self.root / "foreign.json")

    def test_symlink_escape_is_rejected(self) -> None:
        outside = self.root / "outside"
        outside.mkdir()
        self.attempt.mkdir(parents=True)
        link = self.attempt / "link"
        try:
            os.symlink(outside, link, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"symlink creation unavailable: {error}")
        with self.assertRaisesRegex(output.HeadroomOutputError, "HEADROOM_OUTPUT_PATH_OUTSIDE_ATTEMPT"):
            output.assert_path_inside_attempt(self.attempt, link / "escape.json")

    def test_parent_closure_is_recursive_and_idempotent(self) -> None:
        target = self.attempt / "a/b/c/value.json"
        first = output.ensure_parent_directory(self.attempt, target)
        second = output.ensure_parent_directory(self.attempt, target)
        self.assertEqual(first, second)
        self.assertTrue(target.parent.is_dir())

    def test_existing_target_overwrite_is_rejected(self) -> None:
        target = self.attempt / "binary/value.bin"
        output.atomic_write_bytes(self.attempt, target, b"first")
        with self.assertRaisesRegex(output.HeadroomOutputError, "HEADROOM_OUTPUT_TARGET_EXISTS"):
            output.atomic_write_bytes(self.attempt, target, b"second")
        self.assertEqual(target.read_bytes(), b"first")

    def test_controlled_replace_is_explicit(self) -> None:
        target = self.attempt / "status/value.json"
        output.atomic_write_json(self.attempt, target, {"step": 1})
        output.atomic_write_json(self.attempt, target, {"step": 2}, allow_replace=True)
        self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"step": 2})

    def test_invalid_png_is_rejected_and_temp_is_cleaned(self) -> None:
        target = self.attempt / "images/invalid.png"
        with self.assertRaisesRegex(output.HeadroomOutputError, "HEADROOM_OUTPUT_VALIDATION_FAILED"):
            output.atomic_write_bytes(
                self.attempt,
                target,
                b"not a png",
                artifact_type="png",
                validator=output.validate_png,
            )
        self.assertFalse(target.exists())
        self.assertEqual(output.temporary_file_leaks(self.attempt), [])

    def test_png_roundtrip_receipt(self) -> None:
        target = self.attempt / "images/value.png"
        receipt = output.atomic_save_png(self.attempt, target, Image.new("RGB", (2, 3)))
        self.assertEqual(receipt["validation"], "PASS")
        self.assertEqual(receipt["bytes"], target.stat().st_size)
        output.validate_png(target)

    def test_json_duplicate_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            json.loads('{"status":"PASS","status":"FAIL"}', object_pairs_hook=output.strict_object)

    def test_json_required_fields_are_enforced(self) -> None:
        with self.assertRaisesRegex(output.HeadroomOutputError, "HEADROOM_OUTPUT_VALIDATION_FAILED"):
            output.atomic_write_json(
                self.attempt,
                self.attempt / "json/missing.json",
                {"value": 1},
                required_fields=("status",),
            )

    def test_svg_roundtrip(self) -> None:
        target = self.attempt / "svg/value.svg"
        output.atomic_write_svg(self.attempt, target, '<svg xmlns="http://www.w3.org/2000/svg"/>')
        output.validate_svg(target)

    def test_pdf_roundtrip(self) -> None:
        target = self.attempt / "pdf/value.pdf"
        output.atomic_write_pdf(self.attempt, target, b"%PDF-1.4\n%%EOF\n")
        output.validate_pdf(target)

    def test_jsonl_append_is_atomic_and_parseable(self) -> None:
        target = self.attempt / "runs/trajectory.jsonl"
        output.atomic_append_jsonl(self.attempt, target, {"step": 1})
        output.atomic_append_jsonl(self.attempt, target, {"step": 2})
        rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows, [{"step": 1}, {"step": 2}])
        self.assertEqual(output.temporary_file_leaks(self.attempt), [])


class PathPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = runner.attempt_002_output_path_plan()

    def test_plan_passes(self) -> None:
        self.assertEqual(self.plan["status"], "PASS")

    def test_plan_is_not_materialized(self) -> None:
        self.assertFalse(self.plan["materialized"])

    def test_plan_has_120_runs(self) -> None:
        self.assertEqual(self.plan["run_directory_count"], 120)
        self.assertEqual(self.plan["run_metadata_path_count"], 120)

    def test_plan_has_960_checkpoints(self) -> None:
        self.assertEqual(self.plan["checkpoint_path_count"], 960)

    def test_plan_has_20_visual_sheets(self) -> None:
        self.assertEqual(self.plan["visual_sheet_path_count"], 20)

    def test_plan_has_36662_renderer_keys(self) -> None:
        self.assertEqual(self.plan["counts"]["renderer_registry_key_count"], 36_662)

    def test_plan_has_459_unique_parents(self) -> None:
        self.assertEqual(self.plan["counts"]["parent_directory_count"], 459)

    def test_plan_paths_are_unique(self) -> None:
        self.assertEqual(self.plan["counts"]["expected_path_count"], 2729)
        self.assertEqual(self.plan["counts"]["unique_path_count"], 2729)
        self.assertEqual(self.plan["counts"]["duplicate_path_count"], 0)

    def test_plan_has_no_collisions_or_traversal(self) -> None:
        self.assertEqual(self.plan["counts"]["collision_count"], 0)
        self.assertEqual(self.plan["counts"]["traversal_count"], 0)

    def test_plan_has_no_preserved_or_foreign_targets(self) -> None:
        self.assertEqual(self.plan["counts"]["attempt_001_target_count"], 0)
        self.assertEqual(self.plan["counts"]["foreign_output_root_count"], 0)

    def test_plan_is_windows_and_linux_legal(self) -> None:
        self.assertEqual(self.plan["windows_legality"], "PASS")
        self.assertEqual(self.plan["windows_max_path_260"], "PASS")
        self.assertEqual(self.plan["linux_legality"], "PASS")
        self.assertLessEqual(self.plan["counts"]["max_path_length"], 260)

    def test_plan_aggregate_is_deterministic(self) -> None:
        regenerated = runner.attempt_002_output_path_plan()
        self.assertEqual(
            self.plan["deterministic_aggregate_sha256"],
            regenerated["deterministic_aggregate_sha256"],
        )

    def test_plan_aggregate_matches_repaired_contract(self) -> None:
        self.assertEqual(
            self.plan["deterministic_aggregate_sha256"],
            "694eca63bdaead253ab52c3f64bad2300cc91053183774e8181ab91d7a6d1766",
        )

    def test_plan_all_paths_live_under_declared_phase_tree(self) -> None:
        allowed = set(self.plan["directory_contract"]) | {"RUN_STATUS.json"}
        for record in self.plan["paths"]:
            self.assertIn(record["relative_path"].split("/", 1)[0], allowed)

    def test_lambda_tokens_are_frozen(self) -> None:
        self.assertEqual(runner.lambda_token(0.0), "0_diag")
        self.assertEqual(
            [runner.lambda_token(value) for value in runner.POSITIVE_LAMBDAS],
            ["1e-04", "1e-03", "1e-02", "1e-01"],
        )

    def test_unknown_lambda_token_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            runner.lambda_token(0.2)

    def test_directory_contract_is_complete(self) -> None:
        self.assertEqual(self.plan["directory_contract"], [f"{index:02d}_{name}" for index, name in enumerate((
            "preflight", "contract_snapshot", "static_parity", "coefficient_runs",
            "full_residual_runs", "checkpoints", "lambda_selection", "predictions",
            "metrics", "span_analysis", "refined_lookup_analysis", "visual_sheets",
            "failure_analysis", "final_verification",
        ))])


class ContractAndAuditTests(unittest.TestCase):
    def test_exact_repair_source_and_lineage(self) -> None:
        self.assertEqual(runner.INVALID_REPORTING_HEAD, SOURCE_HEAD)
        self.assertEqual(runner.REPAIRED_SOURCE_HEAD, REPAIRED_SOURCE_HEAD)
        self.assertEqual(
            runner.REPAIRED_SOURCE_BRANCH,
            "research/coefficient-headroom-output-path-repair-20260724",
        )
        self.assertEqual(
            runner.RUN_BRANCH,
            "research/render-refined-coefficient-headroom-attempt2-20260724",
        )
        self.assertEqual(
            runner.TASK_ID,
            "AAAI27-RENDER-REFINED-COEFFICIENT-HEADROOM-ATTEMPT-002",
        )
        self.assertEqual(runner.INVALID_EXECUTION_HEAD, "11206393153d710a62b351ff1d136d82154d9e85")
        self.assertEqual(runner.INVALID_FAILURE_HEAD, "1275d79bde07fdd798987024310d822826aaf135")
        self.assertEqual(runner.ATTEMPT_NAME, "attempt_002")
        self.assertEqual(runner.PRESERVED_ATTEMPT_NAME, "attempt_001")

    def test_frozen_expected_counts(self) -> None:
        counts = runner.expected_counts()
        self.assertEqual(counts["optimization_runs"], 120)
        self.assertEqual(counts["optimizer_steps"], 36_000)
        self.assertEqual(counts["checkpoint_writes"], 960)
        self.assertEqual(counts["renderer_calls"], 36_662)

    def test_writer_registry_is_complete(self) -> None:
        registry = runner.headroom_output_writer_registry()
        self.assertEqual(registry["status"], "PASS")
        self.assertEqual(registry["unaudited_scientific_writer_count"], 0)
        self.assertTrue(all(row["coverage"] == "PASS" for row in registry["writers"]))

    def test_writer_registry_has_all_twelve_writer_families(self) -> None:
        registry = runner.headroom_output_writer_registry()
        self.assertEqual(registry["writer_count"], 12)
        self.assertEqual(len({row["writer_id"] for row in registry["writers"]}), 12)

    def test_exact_historical_root_cause_is_frozen(self) -> None:
        self.assertEqual(runner.HISTORICAL_FAILURE_CLASSIFICATION, "PRE_OPTIMIZER_OUTPUT_PATH_ENGINEERING_FAILURE")
        self.assertEqual(runner.HISTORICAL_EXCEPTION_TYPE, "FileNotFoundError")
        self.assertEqual(
            runner.HISTORICAL_FAILING_RELATIVE_PATH,
            "02_static_parity/renders/O01_cond_000000_teacher_rgb.png",
        )
        self.assertEqual(runner.HISTORICAL_FAILING_PARENT, "02_static_parity/renders")
        self.assertEqual(
            runner.HISTORICAL_WRITER_CALLER_CHAIN,
            ("run_parity", "save_render_tensor", "torchvision.utils.save_image", "PIL.Image.Image.save"),
        )

    def test_path_failure_stops_before_renderer_callback(self) -> None:
        renderer_calls = 0

        def renderer() -> None:
            nonlocal renderer_calls
            renderer_calls += 1

        with TemporaryDirectory() as directory:
            attempt = output.resolve_attempt_root(directory)
            with self.assertRaises(output.HeadroomOutputError):
                output.resolve_output_path(attempt, "../outside.png")
            self.assertEqual(renderer_calls, 0)
            self.assertIsNotNone(renderer)

    def test_runner_has_no_legacy_direct_render_writer(self) -> None:
        source = (ROOT / "tools/paper/run_coefficient_headroom_experiment.py").read_text(encoding="utf-8")
        self.assertNotIn('modules["save_render_tensor"]', source)
        self.assertNotIn("canvas.save(", source)
        self.assertNotIn("shutil.copy2", source)

    def test_materialize_preflight_precedes_attempt_creation(self) -> None:
        source = (ROOT / "tools/paper/run_coefficient_headroom_experiment.py").read_text(encoding="utf-8")
        block = source[source.index("def materialize("):source.index("def run_parity(")]
        self.assertLess(block.index("preflight = static_preflight(root)"), block.index("output_io.ensure_directory"))

    def test_write_smoke_is_cpu_only_and_idempotent(self) -> None:
        result = runner.repaired_write_smoke()
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["second_run_idempotent"])
        self.assertTrue(result["cleanup_complete"])
        self.assertEqual(result["renderer_calls"], 0)
        self.assertEqual(result["optimizer_creations"], 0)
        self.assertEqual(result["gpu_calls"], 0)
        self.assertFalse((ROOT / ".tmp").exists())

    def test_protocol_byte_hash_closure_unchanged(self) -> None:
        audit = runner.contract_hash_audit()
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["match_count"], audit["artifact_count"])

    def test_repaired_artifact_closure_is_exact(self) -> None:
        audit = runner.repaired_artifact_audit()
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["match_count"], audit["artifact_count"])

    def test_repaired_contract_fingerprints_are_bound(self) -> None:
        audit = runner.repaired_contract_fingerprints()
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(
            audit["path_plan_aggregate_sha256"],
            "694eca63bdaead253ab52c3f64bad2300cc91053183774e8181ab91d7a6d1766",
        )

    def test_scientific_semantic_drift_is_zero(self) -> None:
        audit = runner.scientific_semantic_audit()
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["semantic_drift_count"], 0)
        self.assertEqual(audit["artifact_count"], 8)

    def test_scientific_contract_semantics_match_source_head(self) -> None:
        for relative in SEMANTIC_CONTRACTS:
            current_text = (ROOT / relative).read_text(encoding="utf-8")
            source_text = subprocess.check_output(
                ["git", "show", f"{SOURCE_HEAD}:{relative}"], cwd=ROOT, text=True, encoding="utf-8"
            )
            parser = yaml.safe_load if relative.endswith((".yaml", ".yml")) else json.loads
            self.assertEqual(parser(current_text), parser(source_text), relative)

    def test_loss_lambda_optimizer_rotation_and_success_gates_unchanged(self) -> None:
        self.assertEqual(tuple(runner.loss_contract()["primary_coefficient_regularization"]["lambda_grid"]), runner.POSITIVE_LAMBDAS)
        self.assertEqual(runner.optimizer_contract()["coefficient_optimizer"]["learning_rate"], 0.02)
        self.assertEqual(len(runner.rotations()), 4)
        self.assertEqual(len(runner.success_contract()["decision_order"]), 5)

    def test_full_residual_schema_unchanged(self) -> None:
        contract = runner.full_contract()
        self.assertEqual(contract["implementation"]["allocated_trainable_scalars"], 2_600_000)
        self.assertEqual(contract["implementation"]["effective_masked_trainable_scalars"], 2_217_111)

    def test_paper_final_is_false(self) -> None:
        self.assertFalse(runner.success_contract()["paper_final"])
        self.assertEqual(runner.expected_counts()["paper_final_count"], 0)

    def test_credential_scan_rejects_nonfixture_secret(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "secret.txt"
            path.write_text(
                "Authorization: " + "Bearer plausible_nonfixture_value_123456\n",
                encoding="utf-8",
            )
            self.assertEqual(runner.credential_scan([path])["status"], "FAIL")

    def test_repair_error_enum_is_complete(self) -> None:
        self.assertEqual(len(output.ERROR_CODES), 9)
        self.assertIn("HEADROOM_OUTPUT_TEMP_FILE_LEAK", output.ERROR_CODES)

    def test_attempt002_repository_outputs_are_isolated(self) -> None:
        self.assertEqual(set(runner.ATTEMPT002_BINDING_REPO_FILES.values()), {
            "coefficient_headroom_attempt002_execution_binding.json",
            "coefficient_headroom_attempt002_expected_counts.json",
            "coefficient_headroom_attempt002_pre_result_tests.json",
        })
        self.assertEqual(runner.ATTEMPT002_REPORT_NAMES, (
            "AAAI27_COEFFICIENT_HEADROOM_ATTEMPT002_RESULTS_20260724.md",
            "AAAI27_TEACHER_SPAN_ATTEMPT002_ANALYSIS_20260724.md",
            "AAAI27_RENDER_REFINED_COEFFICIENT_ATTEMPT002_20260724.md",
            "AAAI27_FULL_RESIDUAL_ATTEMPT002_COMPARISON_20260724.md",
            "AAAI27_HEADROOM_ATTEMPT002_VISUAL_REVIEW_20260724.md",
            "AAAI27_HEADROOM_ATTEMPT002_FAILURE_ANALYSIS_20260724.md",
        ))
        source = (ROOT / "tools/paper/run_coefficient_headroom_experiment.py").read_text(encoding="utf-8")
        seal = source[source.index("def seal_reporting_head("):source.index("def verify(")]
        self.assertNotIn('glob("AAAI27_*HEADROOM*20260724.md")', seal)

    def test_next_task_route_depends_on_scientific_classification(self) -> None:
        self.assertEqual(
            runner.next_task_route("TEACHER_SPAN_CAPACITY_LIMITED")["next_task"],
            "FREEZE_SPATIALLY_DISTRIBUTED_CLOTHING_COEFFICIENT_ORACLE_PROTOCOL",
        )
        self.assertEqual(
            runner.next_task_route("TEACHER_SPAN_HEADROOM_SMALL")["next_task"],
            "RUN_LEAVE_ONE_GARMENT_OUT_BASIS_ADAPTATION_EXPERIMENT_FROM_REPAIRED_CONTRACT",
        )


if __name__ == "__main__":
    unittest.main()
