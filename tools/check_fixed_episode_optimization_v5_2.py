from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.run_fixed_episode_optimization_v5_2 import (  # noqa: E402
    compute_static_balance,
    evaluate_history,
)


def _history(edit_drop: float, clothing_drop: float, slope_sign: float = -1.0) -> list[dict[str, float]]:
    rows = []
    for step in range(81):
        fraction = step / 80
        edit = 1.0 - edit_drop * fraction
        clothing = 1.0 - clothing_drop * fraction
        if step >= 61:
            edit += slope_sign * 1e-5 * (step - 61)
            clothing += slope_sign * 1e-5 * (step - 61)
        rows.append({
            "step": step,
            "edit": edit,
            "clothing": clothing,
            "protected": 0.003,
            "preserve": 0.001,
            "alpha_base": 0.01,
            "objective": 2.0 - 0.01 * fraction,
            "gradient_norm": 1.0,
            "parameter_norm": 1.0,
        })
    return rows


class FixedEpisodeOptimizationTests(unittest.TestCase):
    def test_predeclared_acceptance_passes_stable_reduction(self) -> None:
        report = evaluate_history(
            _history(0.02, 0.02),
            base_unchanged=True,
            backbone_unchanged=True,
            shoe_closer_to_base=True,
        )
        self.assertTrue(report["pass"])
        self.assertGreaterEqual(report["means"]["edit_reduction_fraction"], 0.005)
        self.assertLess(report["slopes"]["clothing_last20"], 0)

    def test_minor_last_step_only_drop_cannot_manufacture_pass(self) -> None:
        rows = _history(0.0, 0.0, slope_sign=1.0)
        rows[-1]["edit"] = 0.98
        rows[-1]["clothing"] = 0.98
        report = evaluate_history(
            rows,
            base_unchanged=True,
            backbone_unchanged=True,
            shoe_closer_to_base=True,
        )
        self.assertFalse(report["pass"])

    def test_static_balance_is_clipped_and_identity_safe(self) -> None:
        report = compute_static_balance({
            "edit_rgb": 0.01,
            "identity": 10.0,
            "alpha": 1.0,
            "regularization": 1e-9,
        })
        scales = report["final_frozen_scales"]
        self.assertTrue(all(0.25 <= value <= 4.0 for value in scales.values()))
        self.assertGreaterEqual(scales["identity"], 1.0)
        self.assertGreaterEqual(scales["alpha"], 1.0)
        self.assertTrue(all(math.isfinite(value) for value in scales.values()))


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(FixedEpisodeOptimizationTests)
    )
    print(json.dumps({
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
    }))
    raise SystemExit(0 if result.wasSuccessful() else 1)
