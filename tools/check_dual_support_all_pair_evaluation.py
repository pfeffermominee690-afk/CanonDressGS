"""Deterministic contract tests for the dual-support all-pair evaluation."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.paper import run_dual_support_all_pair_evaluation as runner


def test_all_10_pairs_are_present() -> None:
    expected = tuple(itertools.combinations(runner.micro.OUTFITS, 2))
    assert runner.PAIRS == expected
    assert len(runner.PAIR_IDS) == 10


def test_both_directions_are_present() -> None:
    assert runner.micro.DIRECTIONS == ("A_TO_B", "B_TO_A")
    assert len(runner.PAIRS) * len(runner.micro.DIRECTIONS) == 20


def test_full_linear_is_reused() -> None:
    value = runner.protocol()
    assert value["variants"]["FULL_LINEAR_BASELINE"]["new_renders"] == 0
    assert value["design"]["full_unique_files"] == 120


def test_dual_support_definition_matches_micro_pilot() -> None:
    current = runner.protocol()["variants"]["DUAL_SUPPORT_GEOMETRY_BLEND"]
    previous = runner.micro.yaml.safe_load(runner.MICRO_PROTOCOL.read_text(encoding="utf-8"))["variants"]["DUAL_SUPPORT_GEOMETRY_BLEND"]
    assert current == previous


def test_hard_geometry_definition_matches_micro_pilot() -> None:
    current = runner.protocol()["variants"]["HARD_GEOMETRY_SOFT_VA"]
    previous = runner.micro.yaml.safe_load(runner.MICRO_PROTOCOL.read_text(encoding="utf-8"))["variants"]["HARD_GEOMETRY_SOFT_VA"]
    assert current == previous


def test_all_pair_endpoint_parity() -> None:
    parity = runner.protocol()["endpoint_parity"]
    assert parity["scope"] == "all 10 pairs, both directions, four target conditions"
    assert parity["render_max_abs_tolerance"] == 1.0e-6
    assert parity["relative_tolerance"] == 0.0


def test_expected_render_counts() -> None:
    assert runner.EXPECTED_LOGICAL == 240
    assert runner.protocol()["design"]["logical_entries_per_variant"] == 240


def synthetic_review_and_manifest() -> tuple[dict, dict]:
    manifest = {
        "main_sheets": [], "opacity_diagnostic_sheets": [], "silhouette_overlay_sheets": [],
        "geometry_support_overlay_sheets": [], "ranked_display_sheets": [],
    }
    items = []
    for pair_id in runner.PAIR_IDS:
        for direction in runner.micro.DIRECTIONS:
            path = f"/main/{pair_id}/{direction}.png"
            manifest["main_sheets"].append(path)
            items.append({
                "kind": "MAIN_SHEET", "source_path": path, "actual_opened": True,
                "pair_id": pair_id, "direction": direction,
                "grades_by_variant_alpha_view": [
                    {
                        "variant": variant, "alpha": alpha, "condition": condition,
                        "grades": dict.fromkeys(runner.VISUAL_CATEGORIES, 0),
                    }
                    for variant in runner.micro.VARIANTS for alpha in runner.micro.ALPHAS
                    for condition in runner.micro.CONDITIONS
                ],
            })
    for key, kind in (
        ("opacity_diagnostic_sheets", "OPACITY_DIAGNOSTIC_SHEET"),
        ("silhouette_overlay_sheets", "SILHOUETTE_OVERLAY_SHEET"),
        ("geometry_support_overlay_sheets", "GEOMETRY_SUPPORT_OVERLAY_SHEET"),
    ):
        for pair_id in runner.PAIR_IDS:
            for direction in runner.micro.DIRECTIONS:
                path = f"/{key}/{pair_id}/{direction}.png"
                manifest[key].append(path)
                items.append({"kind": kind, "source_path": path, "actual_opened": True})
    for rank in ("best", "worst"):
        path = f"/ranked/{rank}.png"
        manifest["ranked_display_sheets"].append(path)
        items.append({"kind": "RANKED_DISPLAY_SHEET", "source_path": path, "actual_opened": True})
    review = {"status": "COMPLETE", "expected_count": 82, "actual_opened_count": 82, "items": items}
    return review, manifest


def test_visual_review_covers_all_pair_directions() -> None:
    review, manifest = synthetic_review_and_manifest()
    runner.validate_manual_review(review, manifest)
    assert len([item for item in review["items"] if item["kind"] == "MAIN_SHEET"]) == 20


def test_efficiency_metrics_are_complete() -> None:
    required = {
        "active_gaussian_count", "render_time_seconds", "peak_vram_bytes", "opacity_mass",
    }
    assert required <= set(runner.protocol()["metrics"])
    assert runner.protocol()["efficiency_accounting"]["checkpoint_copy_allowed"] is False


def test_thresholds_are_frozen() -> None:
    thresholds = runner.protocol()["success_thresholds"]
    assert thresholds["pair_count_required_for_core_grade_drop_at_alpha_0_5"] == 7
    assert thresholds["pair_count_required_for_severe_count_drop"] == 7
    assert thresholds["minimum_severe_count_reduction_fraction"] == 0.5
    assert thresholds["maximum_grade_3_double_outline_or_ghosting_pairs"] == 1


def test_no_training_optimizer_or_checkpoint() -> None:
    gate = runner.protocol()["no_training_gate"]
    assert gate["training_steps"] == gate["backward_calls"] == gate["optimizer_steps"] == 0
    assert gate["diagnostic_optimizer_created"] is False
    assert gate["scheduler_steps"] == gate["checkpoint_writes"] == 0


def test_previous_micro_pilot_is_immutable() -> None:
    for path in (runner.MICRO_REPORT, runner.MICRO_PROTOCOL, runner.MICRO_RESULTS, runner.MICRO_REVIEW, runner.MICRO_SUMMARY):
        assert path.is_file()
    value = runner.archive_fingerprints()
    assert "micro_output_tree" in value and value["micro_summary_sha256"] == runner.sha256(runner.MICRO_SUMMARY)


def test_paper_final_remains_zero() -> None:
    assert runner.protocol()["paper_final"] is False


def main() -> None:
    names = (
        "test_all_10_pairs_are_present", "test_both_directions_are_present",
        "test_full_linear_is_reused", "test_dual_support_definition_matches_micro_pilot",
        "test_hard_geometry_definition_matches_micro_pilot", "test_all_pair_endpoint_parity",
        "test_expected_render_counts", "test_visual_review_covers_all_pair_directions",
        "test_efficiency_metrics_are_complete", "test_thresholds_are_frozen",
        "test_no_training_optimizer_or_checkpoint", "test_previous_micro_pilot_is_immutable",
        "test_paper_final_remains_zero",
    )
    tests: list[tuple[str, Callable[[], None]]] = [(name, globals()[name]) for name in names]
    rows = []
    for name, function in tests:
        function()
        rows.append({"name": name, "status": "PASS"})
    print(json.dumps({"status": "PASS", "test_count": len(rows), "tests": rows}, sort_keys=True))


if __name__ == "__main__":
    main()
