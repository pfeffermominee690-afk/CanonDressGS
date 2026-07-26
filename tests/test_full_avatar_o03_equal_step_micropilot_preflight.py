#!/usr/bin/env python3
"""Tests for the blocked Full Avatar O03 equal-step preflight."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "tools" / "paper" / "run_full_avatar_o03_equal_step_micropilot_preflight.py"
SPEC = importlib.util.spec_from_file_location("full_avatar_preflight", MODULE)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
OUT = ROOT / "paper_protocol" / "external_baselines" / "full_avatar_o03_equal_step_micropilot"
PREFLIGHT = OUT / "full_avatar_o03_equal_step_micropilot_preflight.json"
SUMMARY = OUT / "full_avatar_o03_equal_step_micropilot_final_summary.json"
TEST_REPORT = OUT / "full_avatar_o03_equal_step_micropilot_tests.json"
SOURCE_TESTS = ROOT / "paper_protocol" / "external_baselines" / "external_baseline_feasibility_tests.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.decode().strip()


class FullAvatarPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.preflight = load(PREFLIGHT)
        cls.summary = load(SUMMARY)

    def test_01_source_head_is_ancestor(self):
        self.assertEqual(subprocess.run(["git", "merge-base", "--is-ancestor", audit.SOURCE_HEAD, "HEAD"], cwd=ROOT).returncode, 0)

    def test_02_new_branch(self): self.assertEqual(git("branch", "--show-current"), audit.NEW_BRANCH)
    def test_03_bundle_sha(self): self.assertEqual(hashlib.sha256(audit.BUNDLE.read_bytes()).hexdigest(), audit.BUNDLE_SHA256)

    def test_04_source_feasibility_tests(self):
        row = load(SOURCE_TESTS)
        self.assertEqual((row["status"], row["test_count"], row["pass_count"]), ("PASS", 45, 45))

    def test_05_base_checkpoint_binding(self):
        row = self.preflight["base_avatar"]
        self.assertEqual((row["sha256"], row["step"], row["canonical_gaussian_count"]), (audit.BASE_SHA256, 100000, 200000))

    def test_06_teacher_budget_binding(self):
        row = self.preflight["equal_step"]
        self.assertEqual((row["teacher_checkpoint_sha256"], row["n_teacher_per_garment"], row["full_avatar_finetune_steps"]), (audit.TEACHER_SHA256, 1200, 1200))

    def test_07_target_manifest_binding(self):
        self.assertEqual(self.preflight["target_contract"]["manifest_sha256"], audit.TARGET_MANIFEST_SHA256)

    def test_08_target_counts(self):
        row = self.preflight["target_contract"]
        self.assertEqual((row["rgb_count"], row["primary_mask_count"], row["camera_count"]), (4, 8, 4))

    def test_09_target_registries_complete(self):
        registry = self.preflight["target_contract"]["registry"]
        for key in ["rgb_sha256", "foreground_mask_sha256", "garment_mask_sha256", "camera_sha256", "pose_sha256"]:
            self.assertEqual(len(registry[key]), 4)
            self.assertTrue(all(len(value) == 64 for value in registry[key].values()))

    def test_10_four_rotations_are_disjoint(self):
        rows = self.preflight["target_contract"]["candidate_rotations"]
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["logical_disjointness_pass"] for row in rows))

    def test_11_split_is_not_selected(self):
        self.assertIsNone(self.preflight["target_contract"]["selected_rotation"])
        self.assertIn("NOT_EVALUABLE", self.preflight["target_contract"]["target_leakage_status"])

    def test_12_parameter_contract_not_authorized(self):
        row = self.preflight["parameter_provenance"]
        self.assertEqual(row["candidate_trainable_parameter_count"], 156097000)
        self.assertIsNone(row["approved_trainable_parameter_count"])

    def test_13_required_blockers(self):
        self.assertEqual({row["id"] for row in self.preflight["blockers"]}, {
            "SPLIT_ROTATION_NOT_SELECTED", "LOSS_CONTRACT_NOT_SELECTED",
            "OPTIMIZER_INITIALIZATION_NOT_SELECTED", "SEED_NOT_UNIQUE",
        })

    def test_14_storage_gate(self):
        row = self.preflight["storage"]
        self.assertEqual(row["status"], "PASS")
        self.assertGreaterEqual(row["estimated_remaining_bytes"], row["minimum_remaining_bytes"])

    def test_15_gpu_gate(self):
        row = self.preflight["gpu"]
        self.assertEqual((row["count"], row["status"]), (1, "PASS_SINGLE_GPU"))
        self.assertIn("IMPORT_PASS", row["renderer"])

    def test_16_output_not_created(self): self.assertFalse(self.preflight["execution"]["output_root_created"])

    def test_17_zero_training_operations(self):
        row = self.preflight["operations"]
        for key in ["optimizer_steps_completed", "training_runs", "backward_calls", "checkpoint_count", "render_inferences", "generated_images"]:
            self.assertEqual(row[key], 0)

    def test_18_no_data_or_paper_mutation(self):
        row = self.preflight["operations"]
        self.assertEqual((row["data_mutations"], row["paper_modifications"]), (0, 0))
        self.assertEqual(git("diff", "--name-only", audit.SOURCE_HEAD, "--", "paper_draft"), "")

    def test_19_no_checkpoints_or_metrics(self):
        self.assertEqual((self.summary["checkpoint_count"], self.summary["optimizer_steps_completed"]), (0, 0))
        self.assertEqual(self.summary["metrics_path"], "NOT_RUN_PREFLIGHT_BLOCKED")

    def test_20_face_and_visual_metrics_not_fabricated(self):
        for key in ["full_image_lpips", "garment_region_lpips", "silhouette_iou", "boundary_f", "protected_region_lpips"]:
            self.assertIsNone(self.summary[key])
        self.assertEqual(self.summary["identity_contamination"], "NOT_RUN_PREFLIGHT_BLOCKED")

    def test_21_final_classification(self): self.assertEqual(self.summary["final_classification"], audit.FINAL_CLASSIFICATION)
    def test_22_next_task(self): self.assertEqual(self.summary["next_task"], audit.NEXT_TASK)
    def test_23_paper_final_false(self): self.assertFalse(self.summary["paper_final"])
    def test_24_required_artifacts(self):
        for path in [PREFLIGHT, SUMMARY, audit.REPORT_PATH, audit.HANDOFF_PATH, OUT / "BLOCKER_RESOLUTION_CONTRACT.md"]:
            self.assertTrue(path.is_file() and path.stat().st_size > 0)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FullAvatarPreflightTests)
    test_ids = [case.id() for case in suite]
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    failures = {case.id() for case, _ in result.failures + result.errors}
    report = {
        "schema_version": "canondressgs.full_avatar_o03_equal_step.tests.v1",
        "task_id": audit.TASK_ID, "test_count": result.testsRun,
        "pass_count": result.testsRun - len(failures), "failure_count": len(result.failures),
        "error_count": len(result.errors), "status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests": [{"name": test_id.split(".")[-1], "status": "FAIL" if test_id in failures else "PASS"} for test_id in test_ids],
        "optimizer_steps": 0, "paper_final": False,
    }
    TEST_REPORT.parent.mkdir(parents=True, exist_ok=True)
    TEST_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    sys.exit(0 if result.wasSuccessful() else 1)
