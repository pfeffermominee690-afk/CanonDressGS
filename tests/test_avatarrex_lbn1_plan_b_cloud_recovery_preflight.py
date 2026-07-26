#!/usr/bin/env python3
"""Pre-execution tests for recovered-cloud Plan B extraction."""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

MODULE_PATH = TOOLS / "execute_avatarrex_lbn1_plan_b_cloud_recovery.py"
SPEC = importlib.util.spec_from_file_location("avatarrex_cloud_recovery", MODULE_PATH)
assert SPEC and SPEC.loader
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)
base = recovery.base


class AvatarReXCloudRecoveryPreflightTests(unittest.TestCase):
    def test_01_task_id(self):
        self.assertEqual(
            recovery.TASK_ID,
            "AAAI27-AVATARREX-LBN1-PLAN-B-CLOUD-ACCESS-RECOVERY-AND-EXECUTION-001",
        )

    def test_02_source_branch_and_head(self):
        self.assertEqual(recovery.SOURCE_BRANCH, "research/avatarrex-lbn1-targeted-extraction-plan-b-20260726")
        self.assertEqual(recovery.SOURCE_HEAD, "95d84853ac1ba8d9928613f66310d23b6fd08c0c")

    def test_03_new_branch(self):
        self.assertEqual(
            recovery.NEW_BRANCH,
            "research/avatarrex-lbn1-plan-b-cloud-recovery-execution-20260726",
        )

    def test_04_worktree_paths(self):
        self.assertEqual(
            recovery.WINDOWS_WORKTREE,
            r"E:\model_train\canondressgs_avatarrex_lbn1_plan_b_cloud_recovery_execution",
        )
        self.assertEqual(
            recovery.CLOUD_WORKTREE,
            "/root/autodl-tmp/canondressgs_work/worktrees/"
            "canondressgs_avatarrex_lbn1_plan_b_cloud_recovery_execution",
        )

    def test_05_fixed_storage_threshold(self):
        self.assertEqual(recovery.MIN_PROJECTED_FREE_BYTES, 42_949_672_960)

    def test_06_frozen_allowlist_contract(self):
        members = base.load_members()
        self.assertEqual(len(members), 1602)
        self.assertEqual(sum(member.bytes for member in members), 501_198_133)

    def test_07_archive_contract(self):
        self.assertEqual(str(base.ARCHIVE_PATH), "/root/autodl-tmp/avatarrex_lbn1.7z")
        self.assertEqual(base.ARCHIVE_BYTES, 12_569_755_256)
        self.assertEqual(
            base.ARCHIVE_SHA256,
            "531bd1c71ad9b35f6ae0e2595ee531aa7ba1f83c242f18f2d0505b2dcd5fbcc1",
        )

    def test_08_loader_check_defaults_to_preserved_behavior(self):
        parameter = inspect.signature(base.verify_extraction).parameters["run_loader_check"]
        self.assertIs(parameter.default, True)

    def test_09_recovery_explicitly_skips_loader(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn("run_loader_check=False", source)
        self.assertNotIn("run_loader_smoke(", source)

    def test_10_exactly_one_extraction_call_site(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertEqual(source.count("base.run_extraction("), 1)

    def test_11_no_forbidden_downstream_calls(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        for forbidden in ("train(", "resize(", "outpaint(", "generate_garment("):
            self.assertNotIn(forbidden, source)

    def test_12_correction_overlay(self):
        path = (
            ROOT
            / "paper_protocol"
            / "avatarrex_lbn1_plan_b"
            / "avatarrex_plan_b_previous_classification_correction_overlay_20260726.json"
        )
        value = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(value["previous_classification"], "AVATARREX_PLAN_B_EXTRACTION_CONTRACT_FAIL")
        self.assertEqual(value["corrected_interpretation"], "PREFLIGHT_BLOCKED_BY_CLOUD_SOURCE_UNREACHABLE")
        self.assertFalse(value["previous_report_modified"])
        self.assertFalse(value["paper_final"])

    def test_13_success_boundary(self):
        self.assertEqual(
            recovery.FINAL_CLASSIFICATION,
            "AVATARREX_LBN1_PLAN_B_TARGETED_EXTRACTION_PASS_PENDING_LOADER_REVIEW",
        )
        self.assertEqual(
            recovery.NEXT_TASK,
            "USER_REVIEW_AVATARREX_PLAN_B_EXTRACTION_AND_AUTHORIZE_LOADER_CANARY",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
