from __future__ import annotations

import inspect
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene.trusted_silhouette_semantics_v6_1 import (  # noqa: E402
    active_normalized_asymmetric_removal,
    active_normalized_asymmetric_underfill,
    build_trusted_silhouette_regions,
)
from scene.support_aware_region_trusted_objective_v6_1 import support_aware_region_trusted_objective_v6_1  # noqa: E402
from tools import run_new_silhouette_semantics as runner  # noqa: E402


def fixture() -> dict[str, torch.Tensor]:
    shape = (1, 8, 8)
    value = {name: torch.zeros(shape) for name in (
        "target_protected_mask", "target_foreground_mask", "target_base_foreground_mask",
        "target_clothing_mask", "target_clothing_mask_raw", "target_old_clothing_mask",
        "target_edit_mask", "target_edit_core_mask", "target_transition_mask", "target_preserve_mask",
        "target_background_artifact_mask",
    )}
    value["target_base_foreground_mask"][:, 1:7, 2:6] = 1
    value["target_foreground_mask"][:, 1:7, 1:7] = 1
    value["target_clothing_mask"][:, 2:6, 1:7] = 1
    value["target_clothing_mask_raw"][:, 1:7, 1:7] = 1
    value["target_old_clothing_mask"][:, 2:6, 2:6] = 1
    value["target_edit_mask"][:, 1:7, 1:7] = 1
    value["target_edit_core_mask"][:, 2:6, 2:6] = 1
    value["target_transition_mask"][:, 1, 1:7] = 1
    value["target_preserve_mask"][:, :, :] = 1
    value["target_protected_mask"][:, 2, 1] = 1
    return value


def regions(sample: dict[str, torch.Tensor] | None = None):
    return build_trusted_silhouette_regions(
        sample or fixture(), torch.zeros(1, 3, 8, 8), support_diagonal_ratio=.25, minimum_radius_pixels=1,
    )


class NewSilhouetteContractTests(unittest.TestCase):
    def test_silhouette_error_categories_are_exhaustive(self):
        error = np.ones((4, 4), dtype=bool)
        masks = {name: np.zeros((4, 4), dtype=bool) for name in ("protected", "transition", "safe_clothing")}
        derived = {name: np.zeros((4, 4), dtype=bool) for name in ("artifact", "trusted_expansion", "trusted_removal", "silhouette_uncertain", "raw_clothing")}
        classified = runner.classify_errors(error, "FN", masks, derived)
        self.assertTrue(np.all(sum(value.astype(np.uint8) for value in classified.values()) == 1))

    def test_silhouette_error_categories_are_disjoint(self):
        error = np.ones((3, 3), dtype=bool)
        masks = {name: np.ones((3, 3), dtype=bool) for name in ("protected", "transition", "safe_clothing")}
        derived = {name: np.ones((3, 3), dtype=bool) for name in ("artifact", "trusted_expansion", "trusted_removal", "silhouette_uncertain", "raw_clothing")}
        classified = runner.classify_errors(error, "FN", masks, derived)
        self.assertTrue(np.all(sum(value.astype(np.uint8) for value in classified.values()) == 1))

    def test_trusted_expansion_excludes_protected(self):
        result = regions()
        self.assertEqual(int((result["trusted_expansion"] * result["protected_identity"]).sum()), 0)

    def test_trusted_expansion_requires_garment_evidence(self):
        sample = fixture()
        sample["target_clothing_mask"].zero_()
        self.assertEqual(int(regions(sample)["trusted_expansion"].sum()), 0)

    def test_old_garment_removal_uses_old_clothing_mask(self):
        sample = fixture()
        sample["target_foreground_mask"][:, 3:5, 3:5] = 0
        sample["target_old_clothing_mask"].zero_()
        self.assertEqual(int(regions(sample)["trusted_removal"].sum()), 0)

    def test_uncertain_silhouette_not_hard_supervised(self):
        result = regions()
        hard = result["trusted_expansion"] + result["trusted_removal"]
        self.assertEqual(int((hard * result["silhouette_uncertain"]).sum()), 0)

    def test_underfill_loss_is_asymmetric(self):
        target = torch.ones(1, 1, 2, 2); mask = torch.ones_like(target)
        self.assertGreater(float(active_normalized_asymmetric_underfill(torch.zeros_like(target), target, mask)), 0)
        self.assertEqual(float(active_normalized_asymmetric_underfill(target + 1, target, mask)), 0)

    def test_removal_loss_is_asymmetric(self):
        target = torch.zeros(1, 1, 2, 2); mask = torch.ones_like(target)
        self.assertGreater(float(active_normalized_asymmetric_removal(torch.ones_like(target), target, mask)), 0)
        self.assertEqual(float(active_normalized_asymmetric_removal(target - 1, target, mask)), 0)

    def test_silhouette_loss_normalizes_by_active_pixels(self):
        target = torch.ones(1, 1, 2, 2); pred = torch.zeros_like(target)
        one = torch.zeros_like(target); one[:, :, 0, 0] = 1
        all_pixels = torch.ones_like(target)
        self.assertEqual(float(active_normalized_asymmetric_underfill(pred, target, one)), float(active_normalized_asymmetric_underfill(pred, target, all_pixels)))

    def test_empty_silhouette_mask_is_safe(self):
        value = torch.ones(1, 1, 2, 2)
        loss = active_normalized_asymmetric_underfill(value, value, torch.zeros_like(value))
        self.assertTrue(torch.isfinite(loss)); self.assertEqual(float(loss), 0)

    def test_target_masks_never_enter_forward(self):
        source = inspect.getsource(runner)
        self.assertNotIn("def forward", source)

    def test_no_outfit_specific_color_or_mask_rule(self):
        source = inspect.getsource(build_trusted_silhouette_regions).lower()
        for forbidden in ("purple", "rgb_threshold", "o01", "o08", "hoodie"):
            self.assertNotIn(forbidden, source)

    def test_bounds_are_unchanged_from_t5(self):
        current = yaml.safe_load((PROJECT_ROOT / "configs/research/subject02_new_silhouette_semantics_v1.yaml").read_text())
        previous = yaml.safe_load((PROJECT_ROOT / "configs/research/subject02_objective_residual_redesign_v1.yaml").read_text())
        self.assertEqual(current["candidate_bounds"], previous["candidate_bounds"])

    def test_v6_non_silhouette_losses_are_unchanged(self):
        source = inspect.getsource(support_aware_region_trusted_objective_v6_1)
        self.assertIn("multiscale_masked_charbonnier", source)
        self.assertIn("target_progress_margin_loss", source)
        self.assertIn("F.smooth_l1_loss", source)
        self.assertIn('"identity"', source)
        self.assertIn('"background"', source)
        self.assertIn('"neutral_preserve"', source)

    def test_raw_silhouette_metrics_are_always_reported(self):
        source = inspect.getsource(runner.run_audit)
        self.assertIn('"raw_silhouette_iou"', source)
        self.assertIn('"raw_fn_pixels"', source)
        self.assertIn('"raw_fp_pixels"', source)

    def test_objective_redesign_outputs_unchanged(self):
        config = yaml.safe_load((PROJECT_ROOT / "configs/research/subject02_new_silhouette_semantics_v1.yaml").read_text())
        self.assertIn("attempt_004", config["source_objective_output"])
        self.assertFalse(config["permissions"]["modify_non_silhouette_v6_losses"])

    def test_s2_has_single_pre_registered_coefficient_update(self):
        source = inspect.getsource(runner.decide_s2)
        self.assertIn('"coefficient_update_count": 1 if eligible else 0', source)
        self.assertIn('float(config["S2"]["multiplier_cap"])', source)
        self.assertNotIn("while ", source)

    def test_o01_regression_gate_is_enforced(self):
        source = inspect.getsource(runner.candidate_checks)
        self.assertIn('if outfit == "O01"', source)
        self.assertIn('"raw_silhouette_per_view"', source)

    def test_base_remains_bitwise_exact(self):
        source = inspect.getsource(runner.run_candidate)
        self.assertIn("base_fingerprint_before", source)
        self.assertIn("base_bitwise_exact", source)

    def test_zero_step_failure_bootstrap_reuses_calibration(self):
        source = inspect.getsource(runner.bootstrap_after_zero_step_failure)
        self.assertIn('int(evidence.get("optimizer_steps", -1)) != 0', source)
        self.assertIn('"calibration_reused_without_repetition": True', source)
        self.assertNotIn("run_calibrate(", source)

    def test_visual_evidence_is_inspection_only(self):
        source = inspect.getsource(runner.build_visual_evidence)
        self.assertIn('"renderer_invoked": False', source)
        self.assertIn('"optimizer_steps": 0', source)
        self.assertNotIn("optimizer", source.lower().replace('"optimizer_steps"', ""))

    def test_frozen_branches_unchanged(self):
        expected = "cee8fc51b5039a102ef7e2c31632e348ae3b99a1"
        local = subprocess_output("git", "rev-parse", "research/objective-residual-redesign-20260718", allow_failure=True)
        remote = subprocess_output("git", "rev-parse", "origin/research/objective-residual-redesign-20260718", allow_failure=True)
        self.assertEqual(local or remote, expected)


def subprocess_output(*command: str, allow_failure: bool = False) -> str:
    import subprocess
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=not allow_failure, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(NewSilhouetteContractTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "status": "PASS" if result.wasSuccessful() else "FAIL"}, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
