from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable

import torch
import yaml

from scene.gaussian_clothing_residuals import CHANNELS, GaussianClothingResiduals
from tools.paper import run_continuous_control_causal_attribution as runner


ROOT = Path(__file__).resolve().parents[1]


def test_factor_groups_are_disjoint_and_complete() -> None:
    groups = [set(value) for value in runner.FACTORS.values()]
    assert all(not first.intersection(second) for index, first in enumerate(groups) for second in groups[index + 1:])
    assert set().union(*groups) == set(CHANNELS)


def synthetic_endpoints() -> tuple[GaussianClothingResiduals, GaussianClothingResiduals]:
    left = GaussianClothingResiduals.from_dict({
        name: torch.zeros((3, 2), dtype=torch.float64) for name in CHANNELS
    })
    right = GaussianClothingResiduals.from_dict({
        name: torch.full((3, 2), 2.0, dtype=torch.float64) for name in CHANNELS
    })
    return left, right


def test_unselected_channels_equal_source_endpoint() -> None:
    left, right = synthetic_endpoints()
    result = runner.inject_residual(left, right, "100", 0.2)
    for name in set(CHANNELS) - set(runner.FACTORS["G"]):
        assert torch.equal(getattr(result, name), getattr(left, name))


def test_selected_channels_follow_endpoint_injection() -> None:
    left, right = synthetic_endpoints()
    result = runner.inject_residual(left, right, "110", 0.8)
    for name in runner.subset_channels("110"):
        assert torch.equal(getattr(result, name), getattr(left, name) + 0.8 * (getattr(right, name) - getattr(left, name)))


def test_empty_subset_equals_source_teacher() -> None:
    left, right = synthetic_endpoints()
    assert runner.residual_bitwise_equal(runner.inject_residual(left, right, "000", 0.5), left)


def test_full_subset_equals_existing_full_interpolation() -> None:
    left, right = synthetic_endpoints()
    result = runner.inject_residual(left, right, "111", 0.5)
    assert all(torch.equal(getattr(result, name), torch.ones_like(getattr(result, name))) for name in CHANNELS)


def test_reverse_direction_maps_to_one_minus_alpha() -> None:
    assert runner.stored_alpha("B_TO_A", 0.2) == 0.8
    assert runner.stored_alpha("B_TO_A", 0.5) == 0.5
    assert runner.stored_alpha("B_TO_A", 0.8) == 0.2


def test_new_render_count_is_1440() -> None:
    assert len(runner.NEW_SUBSETS) * len(runner.PAIRS) * len(runner.DIRECTIONS) * len(runner.ALPHAS) * len(runner.CONDITIONS) == 1440


def test_endpoint_and_full_are_reused() -> None:
    protocol = runner.protocol()
    assert protocol["factor_contract"]["endpoint_reuse_count"] == 80
    assert protocol["factor_contract"]["full_reuse_count"] == 240
    assert protocol["factor_contract"]["full_regeneration_forbidden"] is True
    assert protocol["factor_contract"]["endpoint_regeneration_forbidden"] is True


def test_factorial_contrasts_are_deterministic() -> None:
    values = {subset: float(index) for index, subset in enumerate(runner.SUBSET_ORDER)}
    first = runner.factorial_contrasts(values)
    second = runner.factorial_contrasts(values)
    assert first == second
    assert set(first) == set(runner.EFFECTS)


def test_previous_root_cause_archive_is_unchanged() -> None:
    protocol = runner.protocol()
    gates = protocol["previous_archive_gate"]
    for path, expected in (
        (runner.PREVIOUS_SUMMARY, gates["final_summary"]["sha256_lf"]),
        (runner.PREVIOUS_REVIEW, gates["visual_review"]["sha256_lf"]),
        (runner.PREVIOUS_PROTOCOL, gates["previous_protocol"]["sha256_lf"]),
    ):
        assert runner.sha256(path, lf=True) == expected


def test_no_diagnostic_optimizer_or_training() -> None:
    gate = runner.protocol()["no_training_gate"]
    assert gate["training_steps"] == 0
    assert gate["backward_calls"] == 0
    assert gate["diagnostic_optimizer_created"] is False
    assert gate["diagnostic_optimizer_steps"] == 0


def test_legacy_optimizer_has_zero_steps() -> None:
    gate = runner.protocol()["no_training_gate"]
    assert gate["legacy_context_optimizer_zero_grad"] == 0
    assert gate["legacy_context_optimizer_steps"] == 0
    assert gate["scheduler_steps"] == 0


def test_frozen_assets_are_immutable() -> None:
    protocol = runner.protocol()
    assert protocol["governance"]["teacher_mutation_forbidden"] is True
    assert protocol["governance"]["basis_mutation_forbidden"] is True
    assert protocol["governance"]["formal_output_mutation_forbidden"] is True


def test_paper_final_remains_zero() -> None:
    protocol = runner.protocol()
    assert protocol["paper_final"] is False
    assert protocol["no_training_gate"]["paper_final_count"] == 0


def test_causal_alpha_grid_is_020_050_080() -> None:
    assert runner.ALPHAS == (0.2, 0.5, 0.8)
    assert runner.protocol()["design"]["alpha"] == [0.2, 0.5, 0.8]


def test_all_causal_alphas_exist_in_sealed_full_manifest() -> None:
    manifest = json.loads(runner.SEALED_INTERPOLATION.read_text(encoding="utf-8"))
    alphas = {round(float(row["alpha"]), 2) for row in manifest["records"]}
    assert all(alpha in alphas for alpha in runner.ALPHAS)
    assert sum(round(float(row["alpha"]), 2) in runner.ALPHAS for row in manifest["records"]) == 120


def test_reverse_direction_020_maps_to_080() -> None:
    assert runner.stored_alpha("B_TO_A", 0.2) == 0.8


def test_reverse_direction_050_maps_to_050() -> None:
    assert runner.stored_alpha("B_TO_A", 0.5) == 0.5


def test_reverse_direction_080_maps_to_020() -> None:
    assert runner.stored_alpha("B_TO_A", 0.8) == 0.2


def test_full_logical_reuse_count_is_240() -> None:
    assert len(runner.PAIRS) * len(runner.DIRECTIONS) * len(runner.ALPHAS) * len(runner.CONDITIONS) == 240


def test_full_unique_reuse_count_is_120() -> None:
    assert len(runner.PAIRS) * len(runner.ALPHAS) * len(runner.CONDITIONS) == 120


def test_full_regenerated_count_is_zero() -> None:
    correction = json.loads(runner.CORRECTION_PATH.read_text(encoding="utf-8"))
    assert correction["full_reuse_contract"]["regenerated_files"] == 0


def test_new_factor_render_count_is_1440() -> None:
    test_new_render_count_is_1440()


def test_previous_failed_attempt_is_preserved(output_root: Path) -> None:
    path = output_root / "attempt_001/audits/FAILED_PRE_RESULT_ALPHA_GRID_ASSET_MISMATCH.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["status"] == "FAILED_PRE_RESULT_ALPHA_GRID_ASSET_MISMATCH"
    assert value["preserved"] is True
    assert all(int(count) == 0 for count in value["counts"].values())


def test_previous_archives_are_immutable() -> None:
    test_previous_root_cause_archive_is_unchanged()


def test_no_training_or_checkpoint_write() -> None:
    gate = runner.protocol()["no_training_gate"]
    assert gate["training_steps"] == gate["backward_calls"] == gate["checkpoint_writes"] == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/root/autodl-tmp/canondressgs_work/outputs/CONTINUOUS-CONTROL-CAUSAL-ATTRIBUTION-001"),
    )
    args = parser.parse_args()
    tests: list[tuple[str, Callable[[], None]]] = [
        ("test_factor_groups_are_disjoint_and_complete", test_factor_groups_are_disjoint_and_complete),
        ("test_unselected_channels_equal_source_endpoint", test_unselected_channels_equal_source_endpoint),
        ("test_selected_channels_follow_endpoint_injection", test_selected_channels_follow_endpoint_injection),
        ("test_empty_subset_equals_source_teacher", test_empty_subset_equals_source_teacher),
        ("test_full_subset_equals_existing_full_interpolation", test_full_subset_equals_existing_full_interpolation),
        ("test_reverse_direction_maps_to_one_minus_alpha", test_reverse_direction_maps_to_one_minus_alpha),
        ("test_new_render_count_is_1440", test_new_render_count_is_1440),
        ("test_endpoint_and_full_are_reused", test_endpoint_and_full_are_reused),
        ("test_factorial_contrasts_are_deterministic", test_factorial_contrasts_are_deterministic),
        ("test_previous_root_cause_archive_is_unchanged", test_previous_root_cause_archive_is_unchanged),
        ("test_no_diagnostic_optimizer_or_training", test_no_diagnostic_optimizer_or_training),
        ("test_legacy_optimizer_has_zero_steps", test_legacy_optimizer_has_zero_steps),
        ("test_frozen_assets_are_immutable", test_frozen_assets_are_immutable),
        ("test_paper_final_remains_zero", test_paper_final_remains_zero),
        ("test_causal_alpha_grid_is_020_050_080", test_causal_alpha_grid_is_020_050_080),
        ("test_all_causal_alphas_exist_in_sealed_full_manifest", test_all_causal_alphas_exist_in_sealed_full_manifest),
        ("test_reverse_direction_020_maps_to_080", test_reverse_direction_020_maps_to_080),
        ("test_reverse_direction_050_maps_to_050", test_reverse_direction_050_maps_to_050),
        ("test_reverse_direction_080_maps_to_020", test_reverse_direction_080_maps_to_020),
        ("test_full_logical_reuse_count_is_240", test_full_logical_reuse_count_is_240),
        ("test_full_unique_reuse_count_is_120", test_full_unique_reuse_count_is_120),
        ("test_full_regenerated_count_is_zero", test_full_regenerated_count_is_zero),
        ("test_new_factor_render_count_is_1440", test_new_factor_render_count_is_1440),
        ("test_previous_failed_attempt_is_preserved", lambda: test_previous_failed_attempt_is_preserved(args.output_root)),
        ("test_previous_archives_are_immutable", test_previous_archives_are_immutable),
        ("test_no_training_or_checkpoint_write", test_no_training_or_checkpoint_write),
    ]
    results = []
    for name, function in tests:
        function()
        results.append({"name": name, "status": "PASS"})
    print(json.dumps({"status": "PASS", "test_count": len(results), "tests": results}, sort_keys=True))


if __name__ == "__main__":
    main()
