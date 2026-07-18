from __future__ import annotations

import inspect
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene.gaussian_clothing_residuals import GaussianClothingResiduals  # noqa: E402
from scene.support_aware_region_trusted_objective_v6 import (  # noqa: E402
    bound_normalized_regularization,
    build_support_aware_regions,
    support_aware_region_trusted_objective_v6,
    target_progress_margin_loss,
)
import tools.run_objective_residual_redesign as runner  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs/research/subject02_objective_residual_redesign_v1.yaml"


def config():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def sample(height: int = 8, width: int = 8):
    mask = lambda: torch.zeros(1, height, width)
    target_fg = mask(); target_fg[:, 1:7, 1:7] = 1
    base_fg = mask(); base_fg[:, 2:6, 2:6] = 1
    clothing = mask(); clothing[:, 3:6, 3:6] = 1
    old = mask(); old[:, 2:4, 2:4] = 1; old[:, 3, 3] = 0
    edit = torch.maximum(clothing, old)
    protected = mask(); protected[:, 4, 4] = 1
    clothing *= 1 - protected; edit *= 1 - protected
    transition = mask(); transition[:, 1, 1:7] = 1
    preserve = base_fg * (1 - old) * (1 - edit); preserve[:, 4, 4] = 1
    base = torch.zeros(3, height, width)
    target = torch.ones(3, height, width) * edit
    return {
        "target_edit_rgb": target,
        "target_base_rgb": base,
        "target_foreground_mask": target_fg,
        "target_base_foreground_mask": base_fg,
        "target_clothing_mask": clothing,
        "target_old_clothing_mask": old,
        "target_edit_mask": edit,
        "target_edit_core_mask": edit,
        "target_transition_mask": transition,
        "target_preserve_mask": preserve,
        "target_protected_mask": protected,
    }


def objective_inputs(value=None):
    fixture = sample()
    pred = torch.zeros(3, 8, 8) if value is None else value
    alpha = fixture["target_base_foreground_mask"].clone()
    weights = {
        "edit_rgb": 1.0, "target_progress": 1.0, "new_silhouette": .5,
        "transition_alpha": .25, "identity": 2.0, "background": 1.0,
        "neutral_preserve": .5, "residual": 1e-4, "stability": 1e-4,
    }
    return support_aware_region_trusted_objective_v6(
        pred, alpha, fixture, weights=weights, residual_loss=pred.sum() * 0,
        stability_loss=pred.sum() * 0, progress_margin=.02, change_epsilon=.01,
    )


class ObjectiveResidualContractTests(unittest.TestCase):
    def test_v6_old_garment_removal_not_in_preserve(self):
        regions = build_support_aware_regions(sample(), torch.zeros(1, 3, 8, 8))
        self.assertEqual(int((regions["old_garment_removal"] * regions["neutral_preserve"]).sum()), 0)

    def test_v6_protected_identity_uses_base_target(self):
        first = sample(); second = sample(); second["target_edit_rgb"] = torch.rand_like(second["target_edit_rgb"])
        args = dict(weights={"edit_rgb": 1., "target_progress": 1., "new_silhouette": .5, "transition_alpha": .25,
                            "identity": 2., "background": 1., "neutral_preserve": .5, "residual": 1e-4, "stability": 1e-4},
                    residual_loss=torch.tensor(0.), stability_loss=torch.tensor(0.), progress_margin=.02, change_epsilon=.01)
        prediction = torch.full((3, 8, 8), .25); alpha = first["target_base_foreground_mask"]
        a = support_aware_region_trusted_objective_v6(prediction, alpha, first, **args)
        b = support_aware_region_trusted_objective_v6(prediction, alpha, second, **args)
        self.assertEqual(float(a.parts["identity"]), float(b.parts["identity"]))

    def test_v6_safe_clothing_excludes_protected(self):
        regions = build_support_aware_regions(sample(), torch.zeros(1, 3, 8, 8))
        garment = torch.maximum(regions["target_garment"], torch.maximum(regions["old_garment_removal"], regions["new_silhouette"]))
        self.assertEqual(int((garment * regions["protected_identity"]).sum()), 0)

    def test_v6_new_silhouette_uses_target_alpha(self):
        fixture = sample(); correct = objective_inputs().parts["new_silhouette"]
        fixture["target_foreground_mask"] = fixture["target_base_foreground_mask"].clone()
        altered = support_aware_region_trusted_objective_v6(
            torch.zeros(3, 8, 8), fixture["target_base_foreground_mask"], fixture,
            weights={"edit_rgb": 1., "target_progress": 1., "new_silhouette": .5, "transition_alpha": .25,
                     "identity": 2., "background": 1., "neutral_preserve": .5, "residual": 1e-4, "stability": 1e-4},
            residual_loss=torch.tensor(0.), stability_loss=torch.tensor(0.), progress_margin=.02, change_epsilon=.01,
        ).parts["new_silhouette"]
        self.assertGreater(float(correct), float(altered))

    def test_v6_transition_uses_soft_smoothl1(self):
        source = inspect.getsource(support_aware_region_trusted_objective_v6)
        self.assertIn("F.smooth_l1_loss", source)
        self.assertIn("transition_target", source)

    def test_target_progress_prefers_target_over_base(self):
        target = torch.ones(1, 3, 2, 2); base = torch.zeros_like(target); mask = torch.ones(1, 1, 2, 2)
        near_target = target_progress_margin_loss(target, target, base, mask, margin=.02, change_epsilon=.01)
        near_base = target_progress_margin_loss(base, target, base, mask, margin=.02, change_epsilon=.01)
        self.assertLess(float(near_target), float(near_base))

    def test_target_progress_ignores_unchanged_pixels(self):
        same = torch.rand(1, 3, 2, 2)
        value = target_progress_margin_loss(same + .1, same, same, torch.ones(1, 1, 2, 2), margin=.02, change_epsilon=.01)
        self.assertEqual(float(value), 0.0)

    def test_target_progress_has_no_outfit_specific_color(self):
        source = inspect.getsource(target_progress_margin_loss).lower()
        for forbidden in ("purple", "hoodie", "o01", "o08", "rgb_threshold"):
            self.assertNotIn(forbidden, source)

    def test_v6_target_never_enters_forward(self):
        source = inspect.getsource(runner.target_free_state)
        self.assertNotIn("target_edit_rgb", source)
        self.assertNotIn("target_foreground_mask", source)

    def test_bounds_are_global_across_outfits(self):
        cfg = config()
        self.assertIsInstance(cfg["candidate_bounds"], dict)
        self.assertNotIn("O01", json.dumps(cfg["candidate_bounds"]))

    def test_bounds_are_derived_from_pooled_rung2_statistics(self):
        cfg = config()
        for name, p995 in cfg["bound_calibration"]["required_p99_5"].items():
            expected = max(cfg["current_bounds"][name], 1.25 * p995)
            self.assertAlmostEqual(cfg["candidate_bounds"][name], expected, places=7)

    def test_regularization_is_bound_normalized(self):
        source = inspect.getsource(bound_normalized_regularization)
        self.assertIn("value / bound", source)
        self.assertIn("garment_weight", source)

    def test_gradient_calibration_caps_alpha_group(self):
        cfg = config(); source = inspect.getsource(runner.run_calibrate)
        self.assertEqual(cfg["v6"]["gradient_calibration"]["alpha_to_edit_cap"], .5)
        self.assertIn("alpha_scale = min(1.0", source)

    def test_gradient_calibration_caps_regularization_group(self):
        cfg = config(); source = inspect.getsource(runner.run_calibrate)
        self.assertEqual(cfg["v6"]["gradient_calibration"]["regularization_to_edit_cap"], .25)
        self.assertIn("regularization_scale = min(1.0", source)

    def test_t1_changes_only_objective(self):
        cfg = config(); t0, t1 = cfg["tests"]["T0"], cfg["tests"]["T1"]
        self.assertIn("CURRENT_FORMAL_BOUNDS", t0["definition"])
        self.assertEqual((t1["parameterization"], t1["bounds"], t1["regularization"]), ("formal", "current", "current_raw"))

    def test_t2_changes_only_parameterization(self):
        cfg = config(); t2 = cfg["tests"]["T2"]
        self.assertEqual(t2["objective"], "V5.3_OBJECTIVE")
        self.assertEqual(t2["parameterization"], "unbounded_direct")

    def test_t5_uses_formal_residual_composition(self):
        cfg = config(); self.assertEqual(cfg["tests"]["T5"]["parameterization"], "formal")
        self.assertIn("FixedOpenGaussianOracle", inspect.getsource(runner.run_test))

    def test_formal_base_remains_bitwise_exact(self):
        source = inspect.getsource(runner.run_test)
        self.assertIn("base_bitwise_exact", source)
        self.assertIn("_tensor_state_fingerprint", source)

    def test_numeric_checks_are_builtin_booleans(self):
        source = inspect.getsource(runner._numeric_status)
        self.assertIn("bool(value)", source)

    def test_aaai_no_go_remains_unchanged(self):
        cfg = config(); self.assertIn("attempt_003", cfg["source_aaai_output"])
        self.assertFalse(cfg["permissions"]["generate_targets"])

    def test_representation_triage_outputs_unchanged(self):
        cfg = config(); self.assertEqual(cfg["source_representation_triage_commit"], "45539af725dcff6188a326ef732614acfc09cab5")
        self.assertIn("attempt_002", cfg["source_representation_triage_output"])

    def test_long_term_branch_unchanged(self):
        cfg = config(); self.assertEqual(cfg["frozen_branches"]["pipeline/full-dressable-20260715"], "9fc88033b407136073f7bddc6ffca6dd4dd1e0f5")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ObjectiveResidualContractTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    output = {
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "status": "PASS" if result.wasSuccessful() else "FAIL",
    }
    print(json.dumps(output, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
