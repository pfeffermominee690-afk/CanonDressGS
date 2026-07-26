#!/usr/bin/env python3
"""Contract tests for the immutable multi-source evidence bundle."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "paper" / "freeze_external_baseline_multi_source_evidence_bundle.py"
SPEC = importlib.util.spec_from_file_location("freeze_bundle", MODULE_PATH)
assert SPEC and SPEC.loader
freeze = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(freeze)

MANIFEST_PATH = (
    ROOT
    / "paper_protocol"
    / "external_baselines"
    / "source_bundle"
    / "canondressgs_external_baseline_multi_source_evidence_bundle.json"
)
PROVENANCE_PATH = MANIFEST_PATH.with_name(
    "canondressgs_external_baseline_multi_source_provenance_registry.json"
)
CONFLICT_PATH = MANIFEST_PATH.with_name("paper_vs_frozen_evidence_conflict_registry.json")
CONTRACT_PATH = MANIFEST_PATH.with_name("EXTERNAL_BASELINE_AUDIT_SOURCE_CONTRACT_20260726.md")
TEST_REPORT_PATH = MANIFEST_PATH.with_name(
    "canondressgs_external_baseline_multi_source_evidence_bundle_tests.json"
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def hash_lines(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def git_lines(cwd: Path, *args: str) -> list[str]:
    proc = subprocess.run(
        ["git", "-c", "core.quotepath=false", *args],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    text = proc.stdout.decode("utf-8").rstrip("\r\n")
    return [] if not text else text.splitlines()


class EvidenceBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load_json(MANIFEST_PATH)
        cls.provenance = load_json(PROVENANCE_PATH)
        cls.conflicts = load_json(CONFLICT_PATH)
        cls.contract = CONTRACT_PATH.read_text(encoding="utf-8")
        cls.entries = {row["evidence_id"]: row for row in cls.manifest["evidence_entries"]}

    def test_01_dirty_paper_worktree_preservation(self):
        dirty = freeze.DIRTY_WORKTREE
        self.assertEqual(git_lines(dirty, "branch", "--show-current"), [freeze.DIRTY_SNAPSHOT["branch"]])
        self.assertEqual(git_lines(dirty, "rev-parse", "HEAD"), [freeze.DIRTY_SNAPSHOT["head"]])
        checks = {
            "porcelain_v2": git_lines(dirty, "status", "--porcelain=v2"),
            "tracked": git_lines(dirty, "diff", "--name-only"),
            "untracked": git_lines(dirty, "ls-files", "--others", "--exclude-standard"),
            "staged": git_lines(dirty, "diff", "--cached", "--name-only"),
            "conflict": git_lines(dirty, "diff", "--name-only", "--diff-filter=U"),
        }
        for name, lines in checks.items():
            self.assertEqual(len(lines), freeze.DIRTY_SNAPSHOT[f"{name}_count"])
            self.assertEqual(hash_lines(lines), freeze.DIRTY_SNAPSHOT[f"{name}_sha256"])

    def assert_local_commit(self, evidence_id: str):
        commit = self.entries[evidence_id]["resolved_full_sha"]
        self.assertEqual(freeze.git_text("cat-file", "-t", commit), "commit")

    def test_02_paper_source_commit_exists(self): self.assert_local_commit("CURRENT_PAPER_SOURCE")
    def test_03_pure_endpoint_commit_exists(self): self.assert_local_commit("PURE_ENDPOINT")
    def test_04_geometry_causal_commit_exists(self): self.assert_local_commit("GEOMETRY_CAUSAL")
    def test_05_dual_support_commit_exists(self): self.assert_local_commit("DUAL_SUPPORT")
    def test_06_headroom_commit_exists(self): self.assert_local_commit("HEADROOM")
    def test_07_loo_commit_exists(self): self.assert_local_commit("LOO")
    def test_08_paper_restructure_commit_exists(self): self.assert_local_commit("PAPER_RESTRUCTURE")
    def test_09_figure_closure_commit_exists(self): self.assert_local_commit("FIGURE_CLOSURE")

    def test_10_local_origin_cloud_availability_recorded(self):
        self.assertTrue(self.manifest["all_commits_exist_local"])
        self.assertTrue(self.manifest["all_commits_exist_origin"])
        cloud_values = [row["commit_exists_cloud"] for row in self.entries.values()]
        self.assertEqual(self.manifest["all_commits_exist_cloud"], all(cloud_values))
        self.assertEqual(self.manifest["cloud_live_query_status"], "UNAVAILABLE_DNS_CANONDRESS_CLOUD")
        self.assertTrue(all("cloud_availability_basis" in row for row in self.entries.values()))

    def test_11_formal_summary_uniqueness(self):
        self.assertEqual(self.manifest["formal_summary_uniqueness"], "PASS")
        self.assertTrue(all(row["formal_summary_candidate_count_for_task"] == 1 for row in self.entries.values()))

    def test_12_git_object_only_extraction(self):
        for row in self.provenance["exports"]:
            expected = freeze.object_bytes(row["source_commit"], row["source_path"])
            actual = (ROOT / row["export_path"]).read_bytes()
            self.assertEqual(actual, expected, row["export_path"])

    def test_13_no_dirty_file_extraction(self):
        self.assertTrue(self.manifest["dirty_worktree_excluded"])
        self.assertEqual(self.provenance["dirty_file_export_count"], 0)
        valid_commits = {spec["sha"] for spec in freeze.EVIDENCE.values()}
        self.assertTrue(all(row["source_commit"] in valid_commits for row in self.provenance["exports"]))

    def test_14_classification_match(self):
        self.assertTrue(self.manifest["all_classifications_match"])
        for row in self.entries.values():
            self.assertEqual(row["expected_classification"], row["observed_classification"])

    def test_15_pure_endpoint_numeric_checks(self):
        row = self.entries["PURE_ENDPOINT"]
        self.assertTrue(row["numeric_check_pass"])
        values = row["observed_numeric_values"]
        self.assertEqual((values["training_runs"], values["optimizer_steps"], values["checkpoints"]), (24, 7200, 144))
        self.assertAlmostEqual(values["linear_lpips"], 0.067913006991148)
        self.assertEqual((values["mild_blur_top1"], values["single_reference_top1"]), (0.35, 0.95))

    def test_16_geometry_causal_numeric_checks(self):
        values = self.entries["GEOMETRY_CAUSAL"]["observed_numeric_values"]
        self.assertAlmostEqual(values["geometry_main_effect"], 0.9959405426885113)
        self.assertEqual((values["geometry_sufficient"], values["geometry_necessary"]), (10, 10))
        self.assertEqual(sum(values[key] for key in ["visibility_sufficient", "visibility_necessary", "appearance_sufficient", "appearance_necessary"]), 0)

    def test_17_dual_support_numeric_checks(self):
        values = self.entries["DUAL_SUPPORT"]["observed_numeric_values"]
        self.assertEqual(values["unordered_pairs"], 10)
        self.assertAlmostEqual(values["dual_lpips"], 0.05698481313108156)
        self.assertAlmostEqual(values["dual_iou"], 0.7252897968345946)
        self.assertAlmostEqual(values["dual_boundary_f"], 0.3791033717520791)
        self.assertEqual(values["severe_artifact_improved_pairs"], 10)

    def test_18_headroom_numeric_checks(self):
        values = self.entries["HEADROOM"]["observed_numeric_values"]
        self.assertAlmostEqual(values["teacher_lpips"], 0.03876773160882294)
        self.assertAlmostEqual(values["svd_endpoint_lpips"], 0.038765603490173814)
        self.assertAlmostEqual(values["refined_coefficient_lpips"], 0.03898213766515255)
        self.assertEqual((values["improved_garments"], values["garment_count"]), (0, 5))

    def test_19_loo_numeric_checks(self):
        values = self.entries["LOO"]["observed_numeric_values"]
        self.assertEqual((values["folds"], values["capacity_pass"], values["successful_garments"]), (5, 0, 0))
        self.assertGreaterEqual(values["projection_ratio_min"], 0.028)
        self.assertLessEqual(values["projection_ratio_max"], 0.074)
        self.assertAlmostEqual(values["hard_lookup_lpips"], 0.11343667805194854)
        self.assertAlmostEqual(values["low_dimensional_lpips"], 0.11355112642049789)

    def test_20_claim_boundaries(self):
        self.assertTrue(self.manifest["all_claim_boundaries_pass"])
        self.assertIn("fixed identity", self.contract)
        self.assertIn("closed", self.contract.lower())

    def test_21_hard_lookup_equivalence_preserved(self):
        self.assertIn("hard-lookup functional equivalence", self.contract)
        self.assertIn("HARD_LOOKUP_RELATION_DESCRIPTIVE_ONLY", self.entries["PURE_ENDPOINT"]["observed_numeric_values"]["hard_lookup_relation"])

    def test_22_automatic_controller_excluded(self):
        self.assertFalse(self.entries["DUAL_SUPPORT"]["observed_numeric_values"]["automatic_controller_claim"])
        self.assertIn("No automatic pair/weight controller", self.contract)

    def test_23_subject00_positive_claim_zero(self):
        self.assertEqual(self.conflicts["checks"]["subject00_positive_result_claim"], 0)
        self.assertIn("Subject00", self.contract)

    def test_24_actorshq_positive_claim_zero(self):
        self.assertIn("ActorsHQ currently provide no positive paper evidence", self.contract)

    def test_25_paper_vs_evidence_conflict(self):
        self.assertEqual(self.conflicts["conflict_count"], 0)
        self.assertEqual(self.conflicts["status"], "PASS_NO_PAPER_EVIDENCE_CONFLICT")

    def test_26_bundle_schema(self):
        required = {"task_id", "bundle_version", "paper_source_head", "evidence_entries", "final_bundle_classification", "next_task"}
        self.assertTrue(required.issubset(self.manifest))
        self.assertEqual(len(self.entries), 8)

    def test_27_export_sha(self):
        for row in self.provenance["exports"]:
            data = (ROOT / row["export_path"]).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), row["export_sha256"])
            self.assertEqual(freeze.blob_sha(row["source_commit"], row["source_path"]), row["source_blob_sha"])

    def test_28_source_contract_completeness(self):
        markers = ["3d50363dee63dded80c1361cb66b1c83c14b4ada", "dirty paper worktree", "do not need to be ancestors", "Endpoint LPIPS", "PAPER_FINAL=false"]
        for marker in markers:
            self.assertIn(marker, self.contract)

    def test_29_no_merge(self): self.assertEqual(self.provenance["operations"]["merge"], 0)
    def test_30_no_cherry_pick(self): self.assertEqual(self.provenance["operations"]["cherry_pick"], 0)
    def test_31_no_paper_modification(self): self.assertEqual(self.provenance["operations"]["paper_modifications"], 0)
    def test_32_no_training(self): self.assertEqual(self.provenance["operations"]["training_steps"], 0)
    def test_33_no_renderer_inference(self): self.assertEqual(self.provenance["operations"]["renderer_inferences"], 0)

    def test_34_final_classification(self):
        self.assertEqual(self.manifest["final_bundle_classification"], freeze.FINAL_CLASSIFICATION)

    def test_35_next_task_uniqueness(self):
        self.assertEqual(self.manifest["next_task"], freeze.NEXT_TASK)
        self.assertEqual(self.contract.count(freeze.NEXT_TASK), 1)

    def test_36_paper_main_blob(self):
        self.assertEqual(self.manifest["paper_main_tex_blob_sha"], "b3bde953e11a4d9803c39fb9a2e0bba1c141b433")

    def test_37_no_binary_exports(self):
        allowed = {".json", ".md", ".tex", ".bib"}
        self.assertTrue(all(Path(row["export_path"]).suffix.lower() in allowed for row in self.provenance["exports"]))

    def test_38_export_names_include_evidence_id(self):
        for row in self.provenance["exports"]:
            self.assertTrue(Path(row["export_path"]).name.startswith(row["evidence_id"] + "__"))

    def test_39_primary_files_are_unique(self):
        pairs = [(row["resolved_full_sha"], row["primary_summary_path"]) for row in self.entries.values()]
        self.assertEqual(len(pairs), len(set(pairs)))

    def test_40_cross_commit_ancestry_not_required(self):
        self.assertEqual(self.manifest["ancestry_requirement"], "NOT_REQUIRED_ACROSS_SCIENTIFIC_EVIDENCE_COMMITS")

    def test_41_loo_cross_source_provenance_explicit(self):
        note = self.entries["LOO"]["observed_numeric_values"]["absolute_lpips_provenance_note"]
        self.assertIn("cross-source", note)
        self.assertIn(freeze.EVIDENCE["PAPER_RESTRUCTURE"]["sha"], self.contract)

    def test_42_dirty_worktree_excluded(self): self.assertTrue(self.manifest["dirty_worktree_excluded"])
    def test_43_no_data_mutation(self): self.assertEqual(self.provenance["operations"]["data_mutations"], 0)
    def test_44_paper_final_false(self): self.assertFalse(self.manifest["paper_final"])


class ReportingResult(unittest.TextTestResult):
    pass


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(EvidenceBundleTests)
    test_ids = [case.id() for case in suite]
    runner = unittest.TextTestRunner(verbosity=2, resultclass=ReportingResult)
    result = runner.run(suite)
    failed = {case.id() for case, _ in result.failures + result.errors}
    tests = [
        {"name": test_id.split(".")[-1], "status": "FAIL" if test_id in failed else "PASS"}
        for test_id in test_ids
    ]
    report = {
        "schema_version": "canondressgs.external_baselines.multi_source_evidence_bundle_tests.v1",
        "task_id": freeze.TASK_ID,
        "test_count": result.testsRun,
        "pass_count": result.testsRun - len(failed),
        "failure_count": len(result.failures),
        "error_count": len(result.errors),
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests": tests,
        "paper_final": False,
    }
    TEST_REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    sys.exit(0 if result.wasSuccessful() else 1)
