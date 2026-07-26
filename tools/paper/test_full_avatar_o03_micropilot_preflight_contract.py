#!/usr/bin/env python3
"""Validate the frozen Full Avatar O03 preflight-resolution artifacts."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = ROOT / "paper_protocol/external_baselines"
OLD_PREFLIGHT = EXTERNAL / "full_avatar_o03_equal_step_micropilot/full_avatar_o03_equal_step_micropilot_preflight.json"
OVERLAY = EXTERNAL / "full_avatar_o03_preflight_correction_overlay_20260726.json"
MANIFEST = EXTERNAL / "full_avatar_o03_equalstep_execution_manifest_draft_20260726.json"
AUDIT = EXTERNAL / "full_avatar_o03_preflight_resolution_audit_20260726.json"
SUMMARY = EXTERNAL / "full_avatar_o03_preflight_resolution_final_summary_20260726.json"
RESULT = EXTERNAL / "full_avatar_o03_preflight_resolution_tests_20260726.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.overlay = load(OVERLAY)
        cls.manifest = load(MANIFEST)
        cls.audit = load(AUDIT)
        cls.summary = load(SUMMARY)

    def test_original_preflight_is_byte_exact(self) -> None:
        self.assertEqual(sha256(OLD_PREFLIGHT), "4788e9503999e30faf9c4e325068d02d38001e8bcfefd0fa558c57df0bf1bdc4")

    def test_overlay_is_append_only(self) -> None:
        self.assertTrue(self.overlay["original_report_preserved"])
        self.assertEqual(self.overlay["correction_kind"], "APPEND_ONLY_OVERLAY_NO_ORIGINAL_MUTATION")

    def test_old_storage_arithmetic(self) -> None:
        self.assertEqual(self.overlay["cloud_free_bytes_before"] - self.overlay["projected_final_free_bytes"], 10_680_535_158)

    def test_old_storage_shortfall(self) -> None:
        self.assertEqual(self.overlay["formal_safety_gate_bytes"] - self.overlay["projected_final_free_bytes"], 5_032_351_862)

    def test_storage_classification_corrected(self) -> None:
        self.assertEqual(self.overlay["original_storage_classification"], "PASS")
        self.assertEqual(self.overlay["corrected_storage_classification"], "FAIL_BELOW_SAFETY_GATE")

    def test_view_protocol(self) -> None:
        view = self.manifest["view_protocol"]
        self.assertEqual(view["name"], "VIEW_TRANSDUCTIVE_MATCHED_ADAPTATION")
        self.assertEqual((view["optimization_view_count"], view["target_space_evaluation_view_count"]), (4, 4))
        self.assertEqual(view["optimization_evaluation_view_overlap"], "4/4")

    def test_target_leakage_is_disclosed(self) -> None:
        self.assertEqual(self.manifest["view_protocol"]["target_leakage_disclosure"], "INTENTIONAL_PRIVILEGED_ADAPTATION_ON_REGISTERED_TARGET_VIEWS_DISCLOSED")

    def test_control_role_is_privileged(self) -> None:
        self.assertEqual(self.manifest["view_protocol"]["method_role"], "PRIVILEGED_PER_GARMENT_ADAPTATION_CONTROL")

    def test_target_condition_order(self) -> None:
        self.assertEqual(list(self.manifest["target"]["conditions"]), ["cond_000000", "cond_000318", "cond_000017", "cond_000347"])

    def test_each_target_has_camera_pose_and_assets(self) -> None:
        for condition in self.manifest["target"]["conditions"].values():
            self.assertIn("camera", condition)
            self.assertIn("pose", condition)
            self.assertEqual(len(condition["assets"]), 13)

    def test_loss_contract(self) -> None:
        self.assertEqual(self.manifest["loss"]["contract"], "EXACT_O03_TEACHER_OBJECTIVE_MATCH")
        self.assertEqual(self.manifest["loss"]["name"], "CAPACITY_ORACLE_LOSS_V1")

    def test_loss_weights(self) -> None:
        self.assertEqual(self.manifest["loss"]["weights"], {
            "garment_rgb": 1.0,
            "alpha_foreground": 0.5,
            "new_silhouette_alpha": 1.0,
            "boundary_rgb": 0.25,
            "protected_rgb": 10.0,
            "protected_alpha": 5.0,
            "stability": 0.0001,
        })

    def test_no_unfounded_ssim_or_perceptual_loss(self) -> None:
        self.assertTrue(self.manifest["loss"]["ssim_component"].startswith("ABSENT"))
        self.assertTrue(self.manifest["loss"]["perceptual_component"].startswith("ABSENT"))

    def test_loss_implementation_is_pinned(self) -> None:
        self.assertEqual(len(self.manifest["loss"]["implementation_sha256"]), 2)
        self.assertEqual(self.manifest["loss"]["resolved_teacher_config_sha256"], "f0c4c8d3c63b935575c43a00c44ca9a1bd472fbed87329c019ff0d87a8d2c60f")

    def test_trainable_policy(self) -> None:
        self.assertEqual(self.manifest["parameters"]["policy"], "ALL_FORMAL_BASE_OPTIMIZER_GROUPS")

    def test_trainable_registry(self) -> None:
        params = self.manifest["parameters"]
        self.assertEqual(len(params["trainable_groups"]), 14)
        self.assertEqual(params["trainable_tensor_count"], 23)
        self.assertEqual(sum(params["trainable_groups"].values()), 156_097_000)

    def test_frozen_parameter_semantics(self) -> None:
        params = self.manifest["parameters"]
        self.assertEqual(params["frozen_parameter_count_within_adaptation_model"], 0)
        self.assertEqual(params["frozen_checkpoint_state_value_count"], 14_166_176)

    def test_forbidden_initialization(self) -> None:
        forbidden = self.manifest["parameters"]["forbidden_initialization"]
        self.assertIn("O03 Teacher checkpoint", forbidden)
        self.assertIn("CanonDressGS endpoint coefficients", forbidden)

    def test_optimizer_group_count(self) -> None:
        self.assertEqual(len(self.manifest["optimizer"]["groups"]), 14)

    def test_optimizer_classes(self) -> None:
        groups = self.manifest["optimizer"]["groups"]
        self.assertEqual(sum(v["class"] == "torch.optim.Adam" for v in groups.values()), 13)
        self.assertEqual(sum(v["class"] == "torch.optim.AdamW" for v in groups.values()), 1)

    def test_optimizer_state_reset(self) -> None:
        self.assertEqual(self.manifest["optimizer"]["state_policy"], "FRESH_RESET_NO_BASE_MOMENTUM_RESTORE")

    def test_optimizer_constants(self) -> None:
        for name, group in self.manifest["optimizer"]["groups"].items():
            self.assertEqual(group["betas"], [0.9, 0.999], name)
            self.assertEqual(group["eps"], 1e-15, name)

    def test_optimizer_source_is_pinned(self) -> None:
        self.assertEqual(self.manifest["optimizer"]["source_sha256"], "131eb92881a483b71d8e4f27efb585c0bd794615f4c7a0cf39af6bf6f09538b5")

    def test_scheduler_policy(self) -> None:
        self.assertEqual(self.manifest["scheduler"]["policy"], "FRESH_LOCAL_1200_STEP_SCHEDULE_ANCHORED_AT_BASE_STEP100000_EFFECTIVE_LR")

    def test_scheduler_group_count(self) -> None:
        groups = self.manifest["scheduler"]["groups"]
        self.assertEqual(len(groups), 14)
        self.assertEqual(sum(v["family"].endswith("ExponentialLR") for v in groups.values()), 9)
        self.assertEqual(sum(v["family"].startswith("constant") for v in groups.values()), 5)

    def test_scheduler_state_is_fresh(self) -> None:
        self.assertFalse(self.manifest["scheduler"]["base_moments_restored"])
        self.assertFalse(self.manifest["scheduler"]["base_scheduler_state_restored"])

    def test_seed_and_coverage(self) -> None:
        determinism = self.manifest["determinism"]
        self.assertEqual(determinism["seed"], 0)
        self.assertEqual(len(determinism["covered"]), 6)

    def test_fixed_steps_and_milestones(self) -> None:
        execution = self.manifest["execution"]
        self.assertEqual(execution["steps"], 1200)
        self.assertEqual(execution["checkpoint_milestones"], [0, 300, 600, 900, 1200])

    def test_execution_is_unauthorized(self) -> None:
        execution = self.manifest["execution"]
        self.assertFalse(execution["generation_authorized"])
        self.assertFalse(execution["training_authorized"])

    def test_storage_steady_state(self) -> None:
        storage = self.manifest["storage"]
        self.assertEqual(storage["steady_state_storage_bytes"], 10_680_535_158)

    def test_storage_peak_includes_atomic_write(self) -> None:
        storage = self.manifest["storage"]
        self.assertEqual(storage["peak_write_storage_bytes"], storage["steady_state_storage_bytes"] + storage["atomic_temporary_checkpoint_bytes"])
        self.assertEqual(storage["peak_write_storage_bytes"], 12_387_145_460)

    def test_current_storage_projection(self) -> None:
        storage = self.manifest["storage"]
        self.assertEqual(storage["projected_final_free_bytes"], storage["live_free_bytes"] - storage["steady_state_storage_bytes"])
        self.assertEqual(storage["projected_minimum_free_bytes"], storage["live_free_bytes"] - storage["peak_write_storage_bytes"])

    def test_current_storage_shortfall(self) -> None:
        storage = self.manifest["storage"]
        self.assertEqual(storage["formal_safety_shortfall_bytes"], storage["formal_safety_gate_bytes"] - storage["projected_minimum_free_bytes"])
        self.assertGreater(storage["formal_safety_shortfall_bytes"], 0)

    def test_recommended_margin(self) -> None:
        storage = self.manifest["storage"]
        self.assertEqual(storage["recommended_operational_margin_bytes"], 2_147_483_648)
        self.assertEqual(storage["recommended_final_free_bytes"], 42_949_672_960)

    def test_avatarrex_windows_copy(self) -> None:
        archive = self.audit["avatarrex"]
        self.assertEqual(archive["windows_archive_status"], "PASS_BYTE_AND_SHA_MATCH")
        self.assertEqual(archive["windows_archive_bytes"], 12_569_755_256)

    def test_avatarrex_cloud_copy(self) -> None:
        archive = self.audit["avatarrex"]
        self.assertEqual(archive["cloud_archive_status"], "PASS_BYTE_AND_SHA_MATCH_NOT_DELETED")
        self.assertEqual(archive["cloud_archive_sha256"], "531bd1c71ad9b35f6ae0e2595ee531aa7ba1f83c242f18f2d0505b2dcd5fbcc1")

    def test_avatarrex_plan_b_blocks_deletion(self) -> None:
        archive = self.audit["avatarrex"]
        self.assertTrue(archive["cloud_plan_b_staging_status"].startswith("INCOMPLETE"))
        self.assertEqual(archive["recommended_plan"], "PLAN_A_PENDING_AVATARREX_PLAN_B_COMPLETION")

    def test_release_projection(self) -> None:
        archive = self.audit["avatarrex"]
        self.assertEqual(archive["expected_release_bytes"], 12_569_755_256)
        self.assertEqual(archive["projected_final_free_after_release_and_training"], 48_339_416_578)
        self.assertEqual(archive["projected_minimum_free_after_release_and_atomic_write"], 46_632_806_276)

    def test_summary_uses_current_storage_capture(self) -> None:
        storage = self.manifest["storage"]
        self.assertEqual(self.summary["cloud_free_bytes"], storage["live_free_bytes"])
        self.assertEqual(self.summary["projected_final_free_bytes"], storage["projected_final_free_bytes"])
        self.assertEqual(self.summary["safety_shortfall_bytes"], storage["formal_safety_shortfall_bytes"])

    def test_zero_optimizer_steps(self) -> None:
        self.assertEqual(self.summary["optimizer_steps"], 0)

    def test_zero_paper_modifications(self) -> None:
        self.assertEqual(self.summary["paper_modifications"], 0)
        self.assertFalse(self.summary["paper_final"])

    def test_final_classification(self) -> None:
        self.assertEqual(self.summary["final_classification"], "FULL_AVATAR_O03_EXECUTION_CONTRACT_FROZEN_STORAGE_RESOLUTION_PENDING")

    def test_next_task(self) -> None:
        self.assertEqual(self.summary["next_task"], "USER_AUTHORIZE_AVATARREX_CLOUD_DUPLICATE_DELETION_FOR_FULL_AVATAR_MICROPILOT")

    def test_audit_and_manifest_contract_match(self) -> None:
        scientific = self.audit["scientific_contract"]
        self.assertEqual(scientific["view"], self.manifest["view_protocol"])
        self.assertEqual(scientific["loss"], self.manifest["loss"])
        self.assertEqual(scientific["parameters"], self.manifest["parameters"])
        self.assertEqual(scientific["optimizer"], self.manifest["optimizer"])
        self.assertEqual(scientific["scheduler"], self.manifest["scheduler"])
        self.assertEqual(scientific["seed"], self.manifest["determinism"]["seed"])
        self.assertEqual(scientific["steps"], self.manifest["execution"]["steps"])


class RecordingResult(unittest.TextTestResult):
    pass


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests)
    runner = unittest.TextTestRunner(verbosity=2, resultclass=RecordingResult)
    result = runner.run(suite)
    payload = {
        "schema_version": "canondressgs.full_avatar_o03.preflight_resolution_tests.v1",
        "task_id": "AAAI27-FULL-AVATAR-O03-MICROPILOT-PREFLIGHT-RESOLUTION-001",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "result": "PASS" if result.wasSuccessful() else "FAIL",
        "training_authorized": False,
        "optimizer_steps": 0,
        "paper_modifications": 0,
    }
    RESULT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
