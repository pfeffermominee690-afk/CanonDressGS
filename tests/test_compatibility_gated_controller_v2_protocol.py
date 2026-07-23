from __future__ import annotations

import json
from pathlib import Path

import yaml

from scene.compatibility_gated_reference_controller_v2 import (
    MODES,
    OUTFIT_ORDER,
    PAIR_CONFIDENCE_DEFINITION,
    PAIR_ORDER,
)
from tools.paper import run_compatibility_gated_controller_v2_design as design


ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / "paper_protocol/reviewer_risk"


def _protocol() -> dict:
    return yaml.safe_load(
        (RISK / "controller_v2_design_protocol.yaml").read_text(encoding="utf-8")
    )


def _splits() -> dict:
    return json.loads(
        (RISK / "controller_v2_crossfit_splits.json").read_text(encoding="utf-8")
    )


def _frozen() -> tuple[dict, dict]:
    results = json.loads(
        (RISK / "dual_support_all_pair_results.json").read_text(encoding="utf-8")
    )
    visual = json.loads(
        (RISK / "dual_support_all_pair_visual_review.json").read_text(encoding="utf-8")
    )
    return results, visual


def test_v2_pair_order_is_all_ten_unordered_pairs() -> None:
    expected = tuple(
        f"{first}_{second}"
        for index, first in enumerate(OUTFIT_ORDER)
        for second in OUTFIT_ORDER[index + 1 :]
    )
    assert PAIR_ORDER == expected
    assert len(PAIR_ORDER) == 10


def test_v2_protocol_freezes_single_pair_confidence_definition() -> None:
    protocol = _protocol()
    assert PAIR_CONFIDENCE_DEFINITION == "TOP2_VS_TOP3_PROBABILITY_MARGIN"
    assert protocol["routing"]["pair_confidence_definition"] == PAIR_CONFIDENCE_DEFINITION


def test_v2_mixedness_threshold_grid_is_exact() -> None:
    assert design.MIXEDNESS_THRESHOLD_GRID == (
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    )


def test_v2_pair_confidence_threshold_grid_is_exact() -> None:
    assert design.PAIR_CONFIDENCE_THRESHOLD_GRID == (
        0.0,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.40,
        0.50,
    )


def test_v2_crossfit_has_four_rotations() -> None:
    assert len(design.ROTATIONS) == 4
    assert _splits()["rotation_count"] == 4


def test_v2_crossfit_rotation_zero_is_exact() -> None:
    assert design.ROTATIONS[0] == {
        "rotation": 0,
        "train_folds": [0, 1],
        "calibration_fold": 2,
        "test_fold": 3,
    }


def test_v2_crossfit_rotation_one_is_exact() -> None:
    assert design.ROTATIONS[1] == {
        "rotation": 1,
        "train_folds": [1, 2],
        "calibration_fold": 3,
        "test_fold": 0,
    }


def test_v2_crossfit_rotation_two_is_exact() -> None:
    assert design.ROTATIONS[2] == {
        "rotation": 2,
        "train_folds": [2, 3],
        "calibration_fold": 0,
        "test_fold": 1,
    }


def test_v2_crossfit_rotation_three_is_exact() -> None:
    assert design.ROTATIONS[3] == {
        "rotation": 3,
        "train_folds": [3, 0],
        "calibration_fold": 1,
        "test_fold": 2,
    }


def test_v2_condition_order_is_frozen() -> None:
    assert design.CONDITION_ORDER == (
        "cond_000000",
        "cond_000318",
        "cond_000017",
        "cond_000347",
    )


def test_v2_every_rotation_partitions_all_folds() -> None:
    for rotation in design.ROTATIONS:
        folds = (
            list(rotation["train_folds"])
            + [rotation["calibration_fold"], rotation["test_fold"]]
        )
        assert sorted(folds) == [0, 1, 2, 3]


def test_v2_compatibility_rule_uses_calibration_fold_only() -> None:
    results, visual = _frozen()
    source_hashes = {"test": "0" * 64}
    manifest = design.build_manifest(
        design.ROTATIONS[0],
        results=results,
        visual=visual,
        endpoint_parity=design.endpoint_parity_by_pair(results),
        source_hashes=source_hashes,
    )
    assert manifest["calibration_condition"] == "cond_000017"
    assert all(
        entry["calibration_condition"] == "cond_000017"
        and entry["test_condition"] == "cond_000347"
        and entry["test_fold_excluded_from_manifest_construction"]
        for entry in manifest["entries"]
    )


def test_v2_compatibility_manifest_has_all_ten_pairs_in_order() -> None:
    results, visual = _frozen()
    manifest = design.build_manifest(
        design.ROTATIONS[1],
        results=results,
        visual=visual,
        endpoint_parity=design.endpoint_parity_by_pair(results),
        source_hashes={"test": "0" * 64},
    )
    assert tuple(entry["pair_id"] for entry in manifest["entries"]) == PAIR_ORDER


def test_v2_compatibility_manifest_is_rule_generated_without_blacklist() -> None:
    results, visual = _frozen()
    manifest = design.build_manifest(
        design.ROTATIONS[2],
        results=results,
        visual=visual,
        endpoint_parity=design.endpoint_parity_by_pair(results),
        source_hashes={"test": "0" * 64},
    )
    assert manifest["uniform_rule"]["explicit_pair_blacklist"] is False
    assert {entry["compatibility_label"] for entry in manifest["entries"]} == {
        "COMPATIBLE",
        "INCOMPATIBLE",
    }


def test_v2_incompatible_pairs_preserve_frozen_severe_failures() -> None:
    results, visual = _frozen()
    manifest = design.build_manifest(
        design.ROTATIONS[3],
        results=results,
        visual=visual,
        endpoint_parity=design.endpoint_parity_by_pair(results),
        source_hashes={"test": "0" * 64},
    )
    incompatible = {
        entry["pair_id"]
        for entry in manifest["entries"]
        if entry["compatibility_label"] == "INCOMPATIBLE"
    }
    assert incompatible == {"O01_O03", "O02_O03"}


def test_v2_compatibility_manifest_sha_is_canonical() -> None:
    results, visual = _frozen()
    manifest = design.build_manifest(
        design.ROTATIONS[0],
        results=results,
        visual=visual,
        endpoint_parity=design.endpoint_parity_by_pair(results),
        source_hashes={"test": "0" * 64},
    )
    claimed = manifest.pop("manifest_content_sha256")
    assert claimed == design.canonical_sha256(manifest)


def test_v2_manifest_does_not_fabricate_missing_bbox_ratio() -> None:
    results, visual = _frozen()
    manifest = design.build_manifest(
        design.ROTATIONS[0],
        results=results,
        visual=visual,
        endpoint_parity=design.endpoint_parity_by_pair(results),
        source_hashes={"test": "0" * 64},
    )
    for entry in manifest["entries"]:
        descriptors = entry["intrinsic_and_calibration_descriptors"]
        assert descriptors["canonical_bbox_ratio"] is None
        assert "NOT_FABRICATED" in descriptors["canonical_bbox_ratio_status"]


def test_v2_design_runner_has_no_optimizer_construction() -> None:
    source = (
        ROOT / "tools/paper/run_compatibility_gated_controller_v2_design.py"
    ).read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert ".step(" not in source


def test_v2_design_runner_has_no_renderer_import() -> None:
    source = (
        ROOT / "tools/paper/run_compatibility_gated_controller_v2_design.py"
    ).read_text(encoding="utf-8")
    assert "gaussian_renderer" not in source
    assert "render_set" not in source


def test_v2_zero_execution_contract_covers_every_forbidden_action() -> None:
    assert all(value == 0 for value in design.ZERO_EXECUTION_COUNTS.values())
    assert design.ZERO_EXECUTION_COUNTS["paper_final"] == 0
    assert design.ZERO_EXECUTION_COUNTS["formal_renders"] == 0
    assert design.ZERO_EXECUTION_COUNTS["formal_visual_reviews"] == 0


def test_v2_loss_contract_excludes_information_ablation_consistency() -> None:
    consistency = _protocol()["loss"]["consistency"]
    assert "complete_reference_dropout" in consistency["information_ablations_excluded"]
    assert "single_reference" in consistency["information_ablations_excluded"]


def test_v2_protocol_requires_fresh_random_init_for_each_rotation_seed() -> None:
    initialization = _protocol()["future_training"]["initialization"]
    assert initialization == "FRESH_RANDOM"
    assert _protocol()["future_training"]["seeds"] == [0, 1, 2]
    assert (
        _protocol()["future_training"]["process_policy"]
        == "INDEPENDENT_FRESH_PROCESS_PER_ROTATION_AND_SEED"
    )
    assert _protocol()["future_training"]["load_v1_checkpoint"] is False
    assert _protocol()["future_training"]["load_b6_or_ours_v2"] is False


def test_v2_all_three_runtime_modes_are_frozen() -> None:
    assert MODES == (
        "SINGLE_ENDPOINT",
        "DUAL_SUPPORT",
        "HARD_GEOMETRY_SOFT_VA",
    )


def test_v2_protocol_does_not_authorize_paper_final() -> None:
    protocol = _protocol()
    assert protocol["paper_final"] is False
    assert protocol["paper_final_count"] == 0


def test_v2_source_head_is_exact() -> None:
    assert design.SOURCE_HEAD == "25318857b4cc4ee11011cd852c5cfc237a8b3cee"
