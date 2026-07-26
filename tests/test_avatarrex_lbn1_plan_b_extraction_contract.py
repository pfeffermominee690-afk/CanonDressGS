#!/usr/bin/env python3
"""Contract tests for the AvatarReX LBN1 Plan B extraction harness."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "execute_avatarrex_lbn1_plan_b.py"
SPEC = importlib.util.spec_from_file_location("avatarrex_plan_b", MODULE_PATH)
assert SPEC and SPEC.loader
plan_b = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = plan_b
SPEC.loader.exec_module(plan_b)

PROTOCOL_ROOT = ROOT / "paper_protocol" / "avatarrex_lbn1_plan_b"
ALLOWLIST_TXT = PROTOCOL_ROOT / "manifests" / "avatarrex_lbn1_plan_b_allowlist.txt"
ALLOWLIST_JSON = PROTOCOL_ROOT / "manifests" / "avatarrex_lbn1_plan_b_allowlist.json"
MANIFEST_CHECK = PROTOCOL_ROOT / "avatarrex_lbn1_plan_b_manifest_check.json"
BLOCKED_SUMMARY = PROTOCOL_ROOT / "avatarrex_lbn1_plan_b_blocked_summary.json"
REPORT = PROTOCOL_ROOT / "AVATARREX_LBN1_PLAN_B_TARGETED_EXTRACTION_REPORT_20260726.md"
HANDOFF = ROOT / "project_control_handoff" / "avatarrex_lbn1_plan_b_handoff.json"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AvatarReXPlanBExtractionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.members = plan_b.load_members()
        cls.manifest_check = load_json(MANIFEST_CHECK)
        cls.blocked = load_json(BLOCKED_SUMMARY)
        cls.handoff = load_json(HANDOFF)

    def test_01_source_branch_head(self):
        self.assertEqual(plan_b.SOURCE_BRANCH, "research/external-baseline-feasibility-from-bundle-rerun-20260726")
        self.assertEqual(plan_b.SOURCE_HEAD, "572637f08aef21fde35dcd7f8879ff74549c2afc")
        self.assertEqual(plan_b.NEW_BRANCH, "research/avatarrex-lbn1-targeted-extraction-plan-b-20260726")

    def test_02_archive_contract_constants(self):
        self.assertEqual(str(plan_b.ARCHIVE_PATH), "/root/autodl-tmp/avatarrex_lbn1.7z")
        self.assertEqual(plan_b.ARCHIVE_BYTES, 12569755256)
        self.assertEqual(plan_b.ARCHIVE_SHA256, "531bd1c71ad9b35f6ae0e2595ee531aa7ba1f83c242f18f2d0505b2dcd5fbcc1")

    def test_03_allowlist_hashes(self):
        self.assertEqual(sha256(ALLOWLIST_TXT), plan_b.ALLOWLIST_TXT_SHA256)
        self.assertEqual(sha256(ALLOWLIST_JSON), plan_b.ALLOWLIST_JSON_SHA256)

    def test_04_allowlist_count_and_bytes(self):
        self.assertEqual(len(self.members), 1602)
        self.assertEqual(sum(member.bytes for member in self.members), 501198133)
        self.assertEqual(self.manifest_check["allowlist_count"], 1602)
        self.assertEqual(self.manifest_check["expected_uncompressed_bytes"], 501198133)

    def test_05_allowlist_asset_counts(self):
        counts = {}
        for member in self.members:
            counts[member.asset_type] = counts.get(member.asset_type, 0) + 1
        self.assertEqual(counts, {"rgb": 800, "pha": 800, "calibration": 1, "smpl_params": 1})

    def test_06_camera_set_exact(self):
        cameras = {member.camera for member in self.members if member.camera}
        self.assertEqual(cameras, set(plan_b.CAMERA_IDS))
        self.assertEqual(len(plan_b.CAMERA_IDS), 8)

    def test_07_frame_set_exact(self):
        frames = sorted({member.frame for member in self.members if member.frame})
        self.assertEqual(tuple(frames), plan_b.FRAME_IDS)
        self.assertEqual(len(plan_b.FRAME_IDS), 100)

    def test_08_rgb_pha_pairing_exact(self):
        rgb = {(member.camera, member.frame) for member in self.members if member.asset_type == "rgb"}
        pha = {(member.camera, member.frame) for member in self.members if member.asset_type == "pha"}
        self.assertEqual(rgb, pha)
        self.assertEqual(len(rgb), 800)

    def test_09_no_glob_or_full_archive_member(self):
        for member in self.members:
            self.assertNotIn("*", member.path)
            self.assertNotIn("?", member.path)
            self.assertNotIn("..", Path(member.path).parts)
            self.assertFalse(Path(member.path).is_absolute())
        self.assertNotIn("avatarrex_lbn1", {Path(member.path).parts[0] for member in self.members})

    def test_10_staging_not_formal_root(self):
        self.assertEqual(str(plan_b.STAGING_ROOT), "/root/autodl-tmp/datasets/avatarrex_lbn1_staging/PLAN_B_CANARY_001/attempt_001")
        self.assertEqual(str(plan_b.FORMAL_DATA_ROOT), "/root/autodl-tmp/datasets/avatarrex_lbn1")
        self.assertNotEqual(plan_b.STAGING_ROOT, plan_b.FORMAL_DATA_ROOT)

    def test_11_stricter_storage_gate(self):
        expected_required = plan_b.SAFETY_FLOOR_BYTES + plan_b.EXPECTED_UNCOMPRESSED_BYTES + plan_b.PROJECT_AUDIT_RESERVE_BYTES
        self.assertGreater(expected_required, plan_b.SAFETY_FLOOR_BYTES + plan_b.EXPECTED_UNCOMPRESSED_BYTES + plan_b.CONTRACT_OVERHEAD_BYTES)

    def test_12_blocked_summary_is_source_unreachable(self):
        self.assertEqual(self.blocked["final_classification"], plan_b.FINAL_CLASSIFICATION_CONTRACT_FAIL)
        self.assertEqual(self.blocked["failure"]["kind"], "ARCHIVE_SOURCE_UNREACHABLE")
        self.assertEqual(self.blocked["extraction_calls"], 0)

    def test_13_no_forbidden_steps(self):
        for key in ["preprocessing_steps", "training_steps", "generation_calls", "paper_modifications", "archive_mutations"]:
            self.assertEqual(self.blocked[key], 0)
        self.assertFalse(self.blocked["paper_final"])
        self.assertEqual(self.blocked["formal_data_root_status"], "NONE")

    def test_14_no_fake_inventory_or_sha_registry(self):
        self.assertEqual(self.blocked["inventory_path"], "NOT_CREATED_EXTRACTION_NOT_STARTED")
        self.assertEqual(self.blocked["sha_registry_path"], "NOT_CREATED_EXTRACTION_NOT_STARTED")

    def test_15_required_reports_exist(self):
        self.assertTrue(MANIFEST_CHECK.exists())
        self.assertTrue(REPORT.exists())
        self.assertTrue(HANDOFF.exists())
        self.assertIn(plan_b.FINAL_CLASSIFICATION_CONTRACT_FAIL, REPORT.read_text(encoding="utf-8"))

    def test_16_handoff_next_task_unique(self):
        self.assertEqual(self.handoff["next_task"], "USER_RESTORE_CLOUD_ACCESS_AND_RERUN_AVATARREX_PLAN_B_PREFLIGHT")
        self.assertEqual(list(self.handoff).count("next_task"), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
