from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "repaired_evaluation",
    ROOT / "tools/paper/run_controller_v2_repaired_evaluation.py",
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_inference_accounting_is_exact() -> None:
    assert 80 * 4 * 3 == 960
    assert 80 * 4 * 3 * 2 == 1920
    assert 20 * 4 * 3 * 2 == 480
    assert 20 * 6 * 4 * 3 * 2 == 2880
    assert 960 + 1920 + 480 + 2880 == 6240


def test_auc_ties_are_rank_averaged() -> None:
    assert module.auroc([0, 1], [0.5, 0.5]) == 0.5


def test_calibration_sort_prefers_larger_safety_thresholds() -> None:
    objective = {
        "pure_false_mixed_rate": 0.0,
        "wrong_pair_dual_rate": 0.0,
        "incompatible_pair_dual_rate": 0.0,
        "correct_compatible_dual_rate": 0.0,
        "correct_incompatible_hard_rate": 0.0,
        "mixed_false_single_rate": 1.0,
    }
    low = {"tau_mix": 0.1, "tau_pair": 0.0, "objective": objective}
    high = {"tau_mix": 0.9, "tau_pair": 0.5, "objective": objective}
    assert module.objective_sort_key(high) < module.objective_sort_key(low)
