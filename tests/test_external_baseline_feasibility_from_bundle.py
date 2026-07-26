#!/usr/bin/env python3
"""Contract tests for the frozen-bundle external-baseline feasibility audit."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "paper" / "run_external_baseline_feasibility_audit_from_bundle.py"
SPEC = importlib.util.spec_from_file_location("baseline_audit", MODULE_PATH)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)

OUT = ROOT / "paper_protocol" / "external_baselines"
SUMMARY_PATH = OUT / "external_baseline_feasibility_final_summary.json"
SOURCES_PATH = OUT / "external_baseline_official_source_registry.json"
PINS_PATH = OUT / "external_baseline_repository_pin_registry.json"
LICENSE_PATH = OUT / "external_baseline_license_audit.json"
COMPAT_PATH = OUT / "external_baseline_task_compatibility_matrix.json"
DATA_PATH = OUT / "external_baseline_data_dependency_matrix.json"
METRIC_PATH = OUT / "external_target_space_metric_protocol.md"
FULL_PATH = OUT / "full_avatar_finetuning_contract.md"
GSVTON_PATH = OUT / "gsvton_canary_contract.md"
TABLE_PATH = OUT / "external_baseline_table_plan.md"
PRIORITY_PATH = OUT / "external_baseline_execution_priority.md"
TEST_REPORT_PATH = OUT / "external_baseline_feasibility_tests.json"
REPORT_PATH = OUT / "EXTERNAL_BASELINE_FEASIBILITY_AUDIT_REPORT_20260726.md"
HANDOFF_PATH = ROOT / "project_control_handoff" / "external_baseline_feasibility_from_bundle_handoff.json"
SOURCE_WORKTREE = Path(r"E:\model_train\canondressgs_external_baseline_multi_source_evidence_freeze")
EXTERNAL_ROOT = Path(r"E:\model_train\_external_baseline_source_audit_20260726")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def git(cwd: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(["git", *args], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if check and proc.returncode:
        raise RuntimeError(proc.stderr.decode(errors="replace"))
    return proc.stdout.decode().strip()


class FeasibilityAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = load(SUMMARY_PATH)
        cls.sources = load(SOURCES_PATH)
        cls.pins = load(PINS_PATH)
        cls.licenses = load(LICENSE_PATH)
        cls.compat = load(COMPAT_PATH)
        cls.data = load(DATA_PATH)
        cls.metric = METRIC_PATH.read_text(encoding="utf-8")
        cls.full = FULL_PATH.read_text(encoding="utf-8")
        cls.gsvton = GSVTON_PATH.read_text(encoding="utf-8")
        cls.tables = TABLE_PATH.read_text(encoding="utf-8")
        cls.priority = PRIORITY_PATH.read_text(encoding="utf-8")

    def test_01_source_branch_head(self):
        self.assertEqual(self.summary["source_branch"], audit.SOURCE_BRANCH)
        self.assertEqual(self.summary["source_head"], audit.SOURCE_HEAD)
        self.assertEqual(git(ROOT, "cat-file", "-t", audit.SOURCE_HEAD), "commit")
        self.assertEqual(subprocess.run(["git", "merge-base", "--is-ancestor", audit.SOURCE_HEAD, "HEAD"], cwd=ROOT).returncode, 0)

    def test_02_source_worktree_clean(self):
        self.assertEqual(git(SOURCE_WORKTREE, "branch", "--show-current"), audit.SOURCE_BRANCH)
        self.assertEqual(git(SOURCE_WORKTREE, "rev-parse", "HEAD"), audit.SOURCE_HEAD)
        self.assertEqual(git(SOURCE_WORKTREE, "status", "--short"), "")

    def test_03_bundle_exists(self): self.assertTrue(audit.BUNDLE.exists())
    def test_04_bundle_sha(self): self.assertEqual(hashlib.sha256(audit.BUNDLE.read_bytes()).hexdigest(), audit.BUNDLE_SHA256)
    def test_05_source_contract(self): self.assertTrue((audit.BUNDLE.parent / "EXTERNAL_BASELINE_AUDIT_SOURCE_CONTRACT_20260726.md").exists())

    def test_06_bundle_classification_numeric_claim_freeze(self):
        bundle = load(audit.BUNDLE)
        self.assertEqual(bundle["final_bundle_classification"], "CANONDRESSGS_MULTI_SOURCE_EVIDENCE_BUNDLE_FROZEN_FOR_EXTERNAL_BASELINE_AUDIT")
        self.assertTrue(bundle["all_classifications_match"] and bundle["all_numeric_checks_pass"] and bundle["all_claim_boundaries_pass"])
        self.assertEqual(bundle["paper_evidence_conflict_count"], 0)

    def test_07_no_dirty_paper_access(self):
        serialized = json.dumps(self.sources) + json.dumps(self.summary)
        self.assertNotIn("canondressgs_paper_figure_p0_closure_prep", serialized)
        self.assertTrue(self.sources["official_source_only"])

    def test_08_official_source_only(self):
        allowed = {"official arXiv", "official publisher page", "author project page", "author repository", "official submission page", "official 3DV publication page", "official lab repository", "author supplement", "ACM DOI", "frozen local contract"}
        for method in self.sources["methods"]:
            for row in method["official_source_evidence"]:
                self.assertIn(row["kind"], allowed)

    def test_09_publication_status_evidence(self):
        rows = {m["method_id"]: m for m in self.sources["methods"]}
        self.assertEqual(rows["GS_VTON"]["current_publication_status"], "PUBLISHED")
        self.assertEqual(rows["GS_VTON"]["venue"], "International Journal of Computer Vision 2026, volume 134, article 215")
        self.assertEqual(rows["GS_VTON"]["official_paper_url"], "https://link.springer.com/article/10.1007/s11263-026-02805-3")
        self.assertEqual(rows["GAUSSIANVTON"]["current_publication_status"], "PREPRINT")
        self.assertEqual(rows["GAUSSIAN_WARDROBE"]["venue"], "3DV 2026 Poster")
        self.assertEqual(rows["SEMANTICGARMENT"]["venue"], "ACM Multimedia 2025")

    def test_10_repository_pinning(self):
        real = [p for p in self.pins["pins"] if p["repository"]]
        self.assertEqual(len(real), 4)
        self.assertTrue(all(len(p["commit"]) == 40 and p["default_branch"] == "main" for p in real))

    def test_11_license_completeness(self):
        rows = {x["method_id"]: x for x in self.licenses["entries"]}
        self.assertEqual(set(rows), {"GS_VTON", "GAUSSIANVTON", "GAUSSIAN_WARDROBE", "LAYGA", "DAMA", "SEMANTICGARMENT"})
        self.assertEqual(rows["GS_VTON"]["license"], "NO_ROOT_LICENSE_DETECTED")
        self.assertEqual(rows["GAUSSIAN_WARDROBE"]["license"], "MIT")
        self.assertEqual(rows["DAMA"]["license"], "MIT")

    def test_12_code_existence(self):
        rows = {m["method_id"]: m for m in self.sources["methods"]}
        self.assertEqual(rows["GS_VTON"]["official_code_status"], "OFFICIAL_CODE_AVAILABLE")
        self.assertEqual(rows["GAUSSIANVTON"]["official_code_status"], "PARTIAL_CODE_ONLY")
        self.assertEqual(rows["LAYGA"]["official_code_status"], "OFFICIAL_CODE_NOT_FOUND")

    def test_13_compatibility_matrix_completeness(self):
        self.assertEqual(len(self.compat["methods"]), 7)
        for row in self.compat["methods"]:
            self.assertEqual(set(row["answers"]), set(audit.QUESTION_KEYS))

    def test_14_data_dependency_completeness(self):
        self.assertEqual(len(self.data["methods"]), 7)
        for row in self.data["methods"]:
            self.assertTrue({"method_id", "available", "missing", "decision"}.issubset(row))

    def test_15_full_avatar_contract(self):
        markers = ["O01/O02/O03/O04/O08", "Equal-step", "Equal-wall-time", "peak VRAM", "optimizer-state bytes", "resume", "catastrophic drift"]
        for marker in markers: self.assertIn(marker, self.full)

    def test_16_equal_reference_protocol(self):
        self.assertIn("one-reference top-1 `0.95`", self.gsvton)
        self.assertIn("one garment image", self.gsvton)
        self.assertNotIn("multi-reference clean 1.0", self.gsvton.lower())

    def test_17_native_input_protocol(self):
        self.assertEqual(self.summary["native_input_protocol_status"], "NATIVE_INPUT_DIFFERENT_ASSUMPTIONS_COMPARISON")
        self.assertIn("native-input, different-assumptions", self.gsvton)

    def test_18_target_space_metric_protocol(self):
        for marker in ["full-image LPIPS", "garment-region LPIPS", "silhouette IoU", "Boundary F", "protected-region LPIPS", "background", "target camera", "target pose", "lpips==0.1.4", "structural_similarity"]:
            self.assertIn(marker, self.metric)

    def test_19_no_endpoint_lpips_external_table(self):
        table_a = self.tables.split("## Table B")[0]
        self.assertIn("No Endpoint LPIPS", table_a)
        self.assertNotIn("Teacher snapping parity", table_a)

    def test_20_face_id_gate(self):
        self.assertIn("FACE_ID_PROTOCOL_STATUS=NOT_USED_DUE_TO_UNVALIDATED_FACE_ID_PROTOCOL", self.metric)
        self.assertEqual(self.summary["face_id_protocol_status"], "NOT_USED_DUE_TO_UNVALIDATED_FACE_ID_PROTOCOL")

    def test_21_direct_dense_scope(self): self.assertEqual(self.summary["direct_dense_predictor_status"], "HISTORICAL_DIAGNOSTIC_ONLY")

    def test_22_mixedness_claim(self):
        self.assertIn("TARGET_MIXTURE_WEIGHT_ERROR", self.tables)
        self.assertIn("never true garment-mixture reconstruction accuracy", self.tables)

    def test_23_table_separation(self):
        self.assertEqual(self.tables.count("## Table"), 3)
        self.assertIn("must remain separate", self.tables)

    def test_24_maximum_two_candidates(self):
        self.assertLessEqual(len(self.summary["immediate_execution_candidates"]), 2)
        self.assertEqual(self.priority.count("micro-pilot`") + self.priority.count("micro-canary`"), 2)

    def test_25_no_checkpoint_download(self): self.assertEqual(self.summary["operations"]["checkpoint_download_bytes"], 0)
    def test_26_no_dataset_download(self): self.assertEqual(self.summary["operations"]["dataset_download_bytes"], 0)
    def test_27_training_zero(self): self.assertEqual(self.summary["operations"]["training_steps"], 0)
    def test_28_inference_zero(self): self.assertEqual(self.summary["operations"]["renderer_inferences"], 0)
    def test_29_data_mutation_zero(self): self.assertEqual(self.summary["operations"]["data_mutations"], 0)

    def test_30_paper_modification_zero(self):
        self.assertEqual(self.summary["operations"]["paper_modifications"], 0)
        self.assertEqual(git(ROOT, "diff", "--name-only", audit.SOURCE_HEAD, "--", "paper_draft"), "")

    def test_31_bundle_mutation_zero(self): self.assertEqual(self.summary["operations"]["bundle_mutations"], 0)
    def test_32_cloud_nonblocking(self): self.assertEqual(self.summary["cloud_sync_recheck"], "FAILED_NON_BLOCKING")
    def test_33_final_classification(self): self.assertEqual(self.summary["final_classification"], audit.FINAL_CLASSIFICATION)

    def test_34_next_task_uniqueness(self):
        self.assertEqual(self.summary["next_task"], audit.NEXT_TASK)
        self.assertEqual(list(self.summary).count("next_task"), 1)

    def test_35_internal_pure_numbers(self):
        row = self.summary["internal_evidence"]["pure_endpoint"]
        self.assertEqual((row["training_runs"], row["optimizer_steps"], row["checkpoints"]), (24, 7200, 144))
        self.assertAlmostEqual(row["linear_lpips"], 0.067913006991148)
        self.assertEqual(row["canon_one_reference_top1"], 0.95)

    def test_36_internal_geometry_numbers(self):
        row = self.summary["internal_evidence"]["geometry"]
        self.assertAlmostEqual(row["main_effect"], 0.9959405426885113)
        self.assertEqual((row["sufficient"], row["necessary"]), ("10/10", "10/10"))

    def test_37_internal_dual_numbers(self):
        row = self.summary["internal_evidence"]["dual_support"]
        self.assertEqual(row["pair_weight_source"], "EXTERNALLY_SPECIFIED")
        self.assertAlmostEqual(row["dual"]["boundary_f"], 0.379103)
        self.assertFalse(row["automatic_controller"])

    def test_38_negative_evidence_preserved(self):
        self.assertEqual(self.summary["internal_evidence"]["headroom"]["improved_garments"], "0/5")
        self.assertEqual(self.summary["internal_evidence"]["loo"]["capacity_pass"], "0/5")
        self.assertFalse(self.summary["internal_evidence"]["loo"]["unseen_garment_adaptation"])

    def test_39_gsvton_commit(self): self.assertEqual(self.summary["gs_vton"]["pinned_commit"], "96964b0a6528089123cc27a3ff3e3eb46505cf6e")
    def test_40_gsvton_canary_one_case(self):
        for marker in ["Exactly one garment", "one target condition", "one attempt", "No sweep", "max_steps=10000"]: self.assertIn(marker, self.gsvton)

    def test_41_gaussianvton_blocked(self): self.assertEqual(self.summary["gaussianvton"]["code_status"], "PARTIAL_CODE_ONLY")

    def test_42_blocked_method_decisions(self):
        self.assertEqual(self.summary["blocked_methods"], {"GAUSSIANVTON": "BLOCKED_BY_CODE", "GAUSSIAN_WARDROBE": "BLOCKED_BY_DATA", "LAYGA": "BLOCKED_BY_CODE", "DAMA": "BLOCKED_BY_DATA", "SEMANTICGARMENT": "RELATED_WORK_ONLY"})

    def test_43_external_repositories_clean(self):
        for name in ["GS-VTON", "GaussianVTON", "GaussianWardrobe", "DAMA-code"]:
            self.assertEqual(git(EXTERNAL_ROOT / name, "status", "--short"), "")

    def test_44_required_outputs(self):
        paths = [SOURCES_PATH, PINS_PATH, LICENSE_PATH, COMPAT_PATH, DATA_PATH, METRIC_PATH, FULL_PATH, GSVTON_PATH, TABLE_PATH, PRIORITY_PATH, SUMMARY_PATH, REPORT_PATH, HANDOFF_PATH]
        self.assertTrue(all(path.exists() and path.stat().st_size > 0 for path in paths))

    def test_45_paper_final_false(self): self.assertFalse(self.summary["paper_final"])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FeasibilityAuditTests)
    test_ids = [case.id() for case in suite]
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    failures = {case.id() for case, _ in result.failures + result.errors}
    tests = [{"name": test_id.split(".")[-1], "status": "FAIL" if test_id in failures else "PASS"} for test_id in test_ids]
    report = {
        "schema_version": "canondressgs.external_baseline.feasibility_tests.v1",
        "task_id": audit.TASK_ID,
        "test_count": result.testsRun,
        "pass_count": result.testsRun - len(failures),
        "failure_count": len(result.failures),
        "error_count": len(result.errors),
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests": tests,
        "paper_final": False,
    }
    TEST_REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    sys.exit(0 if result.wasSuccessful() else 1)
