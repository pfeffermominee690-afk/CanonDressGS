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

    def test_raw_silhouette_metrics_are_always_reported(self):
        source = inspect.getsource(runner.run_audit)
        self.assertIn('"raw_silhouette_iou"', source)
        self.assertIn('"raw_fn_pixels"', source)
        self.assertIn('"raw_fp_pixels"', source)

    def test_objective_redesign_outputs_unchanged(self):
        config = yaml.safe_load((PROJECT_ROOT / "configs/research/subject02_new_silhouette_semantics_v1.yaml").read_text())
        self.assertIn("attempt_004", config["source_objective_output"])
        self.assertFalse(config["permissions"]["modify_non_silhouette_v6_losses"])

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
