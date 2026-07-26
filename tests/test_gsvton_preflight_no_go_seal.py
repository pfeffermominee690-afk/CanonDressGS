#!/usr/bin/env python3
"""Contract tests for the GS-VTON NO-GO decision seal."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "AAAI27-GSVTON-PREFLIGHT-NO-GO-SEAL-001"
SOURCE_BRANCH = "research/gsvton-license-data-conversion-preflight-20260726"
SOURCE_HEAD = "257a4930254950942cf3621cfb6ac90021d65c79"
NEXT_TASK = "CONTINUE_FULL_AVATAR_O03_PREFLIGHT_RESOLUTION"

BASE = ROOT / "paper_protocol" / "external_baselines"
DECISION_MD = BASE / "GSVTON_EXECUTION_DECISION_20260726.md"
DECISION_JSON = BASE / "gsvton_execution_decision_20260726.json"
TEST_REPORT = BASE / "gsvton_no_go_seal_test_results.json"

PREFLIGHT = ROOT / "artifacts" / "gsvton_preflight" / "AAAI27-GSVTON-LICENSE-DATA-CONVERSION-PREFLIGHT-001"
LICENSE_AUDIT = PREFLIGHT / "gsvton_license_audit.json"
DEPENDENCY_MATRIX = PREFLIGHT / "gsvton_dependency_matrix.json"
WEIGHTS = PREFLIGHT / "gsvton_required_weights_manifest.json"
CONVERSION = PREFLIGHT / "gsvton_subject02_conversion_manifest_draft.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def git(cwd: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and proc.returncode:
        raise RuntimeError(proc.stderr.decode(errors="replace"))
    return proc.stdout.decode(errors="replace").strip()


class GSVTONNoGoSealTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.decision = load(DECISION_JSON)
        cls.decision_md = DECISION_MD.read_text(encoding="utf-8")
        cls.license = load(LICENSE_AUDIT)
        cls.deps = load(DEPENDENCY_MATRIX)
        cls.weights = load(WEIGHTS)
        cls.conversion = load(CONVERSION)

    def test_01_source_head(self):
        self.assertEqual(self.decision["source_branch"], SOURCE_BRANCH)
        self.assertEqual(self.decision["source_head"], SOURCE_HEAD)
        self.assertEqual(git(ROOT, "branch", "--show-current"), SOURCE_BRANCH)
        self.assertEqual(git(ROOT, "cat-file", "-t", SOURCE_HEAD), "commit")
        self.assertEqual(
            subprocess.run(["git", "merge-base", "--is-ancestor", SOURCE_HEAD, "HEAD"], cwd=ROOT).returncode,
            0,
        )

    def test_02_preflight_facts_unchanged(self):
        facts = self.decision["fixed_preflight_facts"]
        self.assertEqual(self.license["official_code_license_decision"]["license_status"], facts["license_status"])
        self.assertEqual(facts["license_status"], "LICENSE_NOT_SPECIFIED")
        self.assertFalse(facts["execution_authorized_by_license"])
        self.assertGreaterEqual(self.weights["known_public_lower_bound_bytes"], facts["required_weight_bytes_lower_bound"])
        self.assertEqual(self.deps["environment_estimated_bytes"], facts["environment_estimated_bytes"])
        self.assertGreaterEqual(self.weights["storage_estimate"]["estimated_total_storage_bytes"], facts["estimated_total_storage_lower_bound"])
        self.assertEqual(
            self.conversion["static_3dgs_initialization_requirement"]["mmlphuman_initialization_compatibility"],
            "RETRAIN_STATIC_3DGS_REQUIRED",
        )
        self.assertEqual(self.conversion["status"], "DRAFT_FIELD_MAPPING_COMPLETE_NO_DATA_MUTATION")
        self.assertEqual(self.conversion["mask_semantics_status"], "SPECIFIED_FOREGROUND_NOT_HUMAN_PARSE_LABELS")
        self.assertEqual(self.conversion["final_classification"], "GSVTON_MICRO_CANARY_PREFLIGHT_MULTIPLE_BLOCKERS")
        self.assertEqual(git(ROOT, "diff", "--name-only", SOURCE_HEAD, "--", "artifacts/gsvton_preflight"), "")

    def test_03_execution_authorized_false(self):
        decision = self.decision["decision"]
        self.assertFalse(decision["gs_vton_execution_authorized"])
        self.assertIn("GS_VTON_EXECUTION_AUTHORIZED = false", self.decision_md)
        self.assertTrue(decision["not_negative_method_quality_judgment"])
        self.assertFalse(decision["canon_dress_gs_numerical_superiority_claim_over_gsvton_allowed"])

    def test_04_numeric_baseline_not_executed(self):
        decision = self.decision["decision"]
        self.assertEqual(decision["gs_vton_micro_canary"], "DEFERRED")
        self.assertEqual(decision["gs_vton_numeric_baseline"], "NOT_EXECUTED")
        self.assertEqual(decision["gs_vton_paper_role"], "RELATED_WORK_AND_DIFFERENT_ASSUMPTIONS_DISCUSSION_ONLY")
        self.assertFalse(decision["forged_or_paper_reported_gsvton_values_allowed"])

    def test_05_checkpoint_download_zero(self):
        for artifact in [self.decision, self.license, self.weights, self.conversion]:
            self.assertEqual(artifact["operations"]["checkpoint_download_bytes"], 0)
        self.assertEqual(self.weights["checkpoint_download_bytes"], 0)

    def test_06_training_zero(self):
        for artifact in [self.decision, self.license, self.weights, self.conversion]:
            self.assertEqual(artifact["operations"]["training_steps"], 0)
            self.assertEqual(artifact["operations"]["renderer_inferences"], 0)

    def test_07_paper_modification_zero(self):
        self.assertEqual(self.decision["operations"]["paper_modifications"], 0)
        self.assertFalse(self.decision["operations"]["paper_final"])
        self.assertFalse(self.decision["paper_final"])
        self.assertEqual(git(ROOT, "diff", "--name-only", SOURCE_HEAD, "--", "paper_draft"), "")

    def test_08_full_avatar_selected(self):
        baseline = self.decision["current_baseline_decision"]
        self.assertEqual(baseline["immediate_baseline_execution_selection"], "FULL_AVATAR_FINETUNING_ONLY")
        self.assertEqual(baseline["full_avatar_finetuning_type"], "SAME_BACKBONE_STRONG_CONTROL")
        self.assertEqual(baseline["external_numeric_baseline_currently_executed"], "NONE")
        self.assertEqual(baseline["table_a_current_candidates"], ["Base Avatar", "Full Avatar Fine-tuning", "CanonDressGS"])
        self.assertFalse(baseline["gs_vton_enters_numeric_table"])

    def test_09_next_task_uniqueness(self):
        self.assertEqual(self.decision["next_task"], NEXT_TASK)
        self.assertNotIn("next_tasks", self.decision)
        self.assertEqual(json.dumps(self.decision, sort_keys=True).count(NEXT_TASK), 1)
        self.assertEqual(self.decision["final_classification"], "GSVTON_PREFLIGHT_NO_GO_DECISION_SEALED_FULL_AVATAR_SELECTED")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(GSVTONNoGoSealTests)
    test_ids = [case.id() for case in suite]
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    failures = {case.id() for case, _ in result.failures + result.errors}
    report = {
        "schema_version": "canondressgs.gsvton_no_go_seal_tests.v1",
        "task_id": TASK_ID,
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "test_count": result.testsRun,
        "pass_count": result.testsRun - len(failures),
        "failure_count": len(result.failures),
        "error_count": len(result.errors),
        "contract_checks": [
            {"name": test_id.split(".")[-1], "status": "FAIL" if test_id in failures else "PASS"}
            for test_id in test_ids
        ],
        "operations": {
            "checkpoint_download_bytes": 0,
            "dataset_download_bytes": 0,
            "environment_creation": 0,
            "training_steps": 0,
            "renderer_inferences": 0,
            "data_mutations": 0,
            "paper_modifications": 0,
            "paper_final": False,
        },
        "final_classification": "GSVTON_PREFLIGHT_NO_GO_DECISION_SEALED_FULL_AVATAR_SELECTED",
        "next_task": NEXT_TASK,
        "paper_final": False,
    }
    TEST_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    sys.exit(0 if result.wasSuccessful() else 1)
