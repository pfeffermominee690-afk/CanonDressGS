"""Deterministic contract tests for the dual-support micro-pilot."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Callable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene.gaussian_clothing_residuals import CHANNELS, GaussianClothingResiduals
from tools.paper import run_geometry_dual_support_micro_pilot as runner


def residual(value: float) -> GaussianClothingResiduals:
    shapes = {
        "delta_xyz": (4, 3),
        "delta_log_scaling": (4, 3),
        "delta_rotvec": (4, 3),
        "delta_opacity_logit": (4,),
        "delta_sh0": (4, 1, 3),
        "delta_shN": (4, 3, 3),
    }
    return GaussianClothingResiduals.from_dict({
        name: torch.full(shape, value, dtype=torch.float32) for name, shape in shapes.items()
    })


def test_pair_selection_is_deterministic() -> None:
    order = tuple(itertools.combinations(runner.OUTFITS, 2))
    stable = {"O01_O02", "O01_O04", "O03_O04"}
    stable_rows = [pair for pair in order if "_".join(pair) in stable]
    unstable_rows = [pair for pair in order if "_".join(pair) not in stable]
    assert runner.SELECTED_PAIRS == (stable_rows[0], unstable_rows[0], unstable_rows[1])


def test_full_linear_is_reused() -> None:
    value = runner.protocol()
    assert value["variants"]["FULL_LINEAR_BASELINE"]["new_renders"] == 0
    assert len(runner.SELECTED_PAIRS) * 2 * 3 * 4 == 72


def test_hard_geometry_never_interpolates_geometry() -> None:
    source, target, earlier = residual(1.0), residual(2.0), residual(3.0)
    for alpha, expected in ((0.2, source), (0.5, earlier), (0.8, target)):
        actual = runner.hard_geometry_residual(source, target, earlier, alpha)
        for name in runner.GEOMETRY_CHANNELS:
            assert torch.equal(getattr(actual, name), getattr(expected, name))
        for name in runner.VA_CHANNELS:
            wanted = (1.0 - alpha) * getattr(source, name) + alpha * getattr(target, name)
            assert torch.equal(getattr(actual, name), wanted)


def test_dual_support_preserves_endpoint_geometry() -> None:
    source, target = residual(1.0), residual(2.0)
    before = {name: (getattr(source, name).clone(), getattr(target, name).clone()) for name in CHANNELS}
    runner.duplicate_support_statistics(source, target)
    for name, (source_before, target_before) in before.items():
        assert torch.equal(getattr(source, name), source_before)
        assert torch.equal(getattr(target, name), target_before)


def test_dual_support_opacity_scaling_is_correct() -> None:
    endpoint_logit = torch.tensor([-2.0, 0.0, 2.0])
    for weight in (0.2, 0.5, 0.8):
        expected = weight * torch.sigmoid(endpoint_logit)
        recovered = torch.sigmoid(runner.stable_logit(expected))
        assert torch.allclose(recovered, expected, atol=1.0e-7, rtol=1.0e-6)


def test_dual_support_endpoint_parity() -> None:
    source, target = residual(1.0), residual(2.0)
    assert torch.equal(runner.hard_geometry_residual(source, target, source, 0.0).delta_xyz, source.delta_xyz)
    assert torch.equal(runner.hard_geometry_residual(source, target, source, 1.0).delta_xyz, target.delta_xyz)
    assert runner.PARITY_TOLERANCE == 1.0e-6


def test_dual_support_does_not_modify_teachers() -> None:
    source, target = residual(1.0), residual(2.0)
    hashes_before = {
        name: (getattr(source, name).numpy().tobytes(), getattr(target, name).numpy().tobytes())
        for name in CHANNELS
    }
    runner.hard_geometry_residual(source, target, source, 0.5)
    for name in CHANNELS:
        assert getattr(source, name).numpy().tobytes() == hashes_before[name][0]
        assert getattr(target, name).numpy().tobytes() == hashes_before[name][1]


def test_no_training_optimizer_or_checkpoint() -> None:
    gate = runner.protocol()["no_training_gate"]
    assert gate["training_steps"] == gate["backward_calls"] == gate["optimizer_steps"] == 0
    assert gate["diagnostic_optimizer_created"] is False
    assert gate["scheduler_steps"] == gate["checkpoint_writes"] == 0


def test_micro_pilot_thresholds_are_frozen() -> None:
    thresholds = runner.protocol()["success_thresholds"]
    assert thresholds["pair_count_required_for_core_grade_drop_at_alpha_0_5"] == 2
    assert thresholds["minimum_core_grade_drop"] == 1
    assert thresholds["pair_count_required_for_severe_count_drop"] == 2
    assert thresholds["minimum_severe_count_reduction_fraction"] == 0.5
    assert thresholds["maximum_silhouette_iou_relative_degradation"] == 0.05
    assert thresholds["maximum_garment_lpips_relative_degradation"] == 0.10


def test_paper_final_remains_zero() -> None:
    assert runner.protocol()["paper_final"] is False


def main() -> None:
    tests: list[tuple[str, Callable[[], None]]] = [
        (name, globals()[name])
        for name in (
            "test_pair_selection_is_deterministic",
            "test_full_linear_is_reused",
            "test_hard_geometry_never_interpolates_geometry",
            "test_dual_support_preserves_endpoint_geometry",
            "test_dual_support_opacity_scaling_is_correct",
            "test_dual_support_endpoint_parity",
            "test_dual_support_does_not_modify_teachers",
            "test_no_training_optimizer_or_checkpoint",
            "test_micro_pilot_thresholds_are_frozen",
            "test_paper_final_remains_zero",
        )
    ]
    rows = []
    for name, function in tests:
        function()
        rows.append({"name": name, "status": "PASS"})
    print(json.dumps({"status": "PASS", "test_count": len(rows), "tests": rows}, sort_keys=True))


if __name__ == "__main__":
    main()
