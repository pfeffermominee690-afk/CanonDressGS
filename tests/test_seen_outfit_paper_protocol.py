from __future__ import annotations

import inspect
import json
import os
import re
from pathlib import Path

import yaml

from scene.multi_outfit_linear_coefficient_control import MultiOutfitLinearCoefficientControl
from tools.paper.verify_seen_outfit_paper_assets import verify_manifest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/paper/aaai27_seen_outfit_explicit_basis_v1.yaml"
MANIFEST_PATH = ROOT / "paper_protocol/frozen_asset_manifest.json"
REGISTRY_PATH = ROOT / "paper_protocol/experiment_registry.yaml"
TABLE_FIGURE_PATH = ROOT / "docs/PAPER/AAAI27_PAPER_TABLE_AND_FIGURE_PLAN_20260720.md"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
REGISTRY = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))


def test_paper_split_is_frozen():
    assert CONFIG["data"]["seen_outfits"] == ["O01", "O02", "O03", "O04", "O08"]
    assert CONFIG["data"]["episode_count"] == 20


def test_o07_is_held_out_only():
    assert CONFIG["data"]["held_out_diagnostic"] == "O07"
    assert CONFIG["permissions"]["train_on_o07"] is False


def test_o06_is_unused_reserve():
    assert CONFIG["data"]["unused_reserve"] == "O06"
    assert CONFIG["permissions"]["substitute_o06_for_o07"] is False


def test_paper_basis_rank_is_four():
    assert CONFIG["basis"]["rank"] == 4 and CONFIG["basis"]["frozen"] is True


def test_paper_assets_match_sha256():
    assert len(MANIFEST["assets"]) == 19
    assert all(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", item["fingerprint"])
               for item in MANIFEST["assets"])
    output_root = os.environ.get("CANONDRESSGS_OUTPUT_ROOT")
    report = verify_manifest(
        MANIFEST, ROOT, Path(output_root) if output_root else None,
        verify_external=output_root is not None,
    )
    assert report["status"] == "PASS"


def test_paper_predictor_has_no_outfit_id():
    parameters = inspect.signature(MultiOutfitLinearCoefficientControl.forward).parameters
    assert "outfit_id" not in parameters and "cloth_id" not in parameters
    assert CONFIG["permissions"]["outfit_id_in_model"] is False


def test_paper_predictor_has_no_target_input():
    parameters = inspect.signature(MultiOutfitLinearCoefficientControl.forward).parameters
    assert not any(name.startswith("target") for name in parameters)
    assert CONFIG["permissions"]["target_forward_input"] is False


def test_paper_metrics_use_outfit_macro_average():
    aggregation = CONFIG["metrics"]["aggregation"]
    assert aggregation == ["episode", "outfit_macro", "seed_mean_std"]
    assert CONFIG["metrics"]["pixel_weighted_outfit_micro_average"] is False


def test_paper_seeds_are_exactly_0_1_2():
    assert CONFIG["training"]["seeds"] == [0, 1, 2]


def test_paper_registry_starts_not_run():
    assert len(REGISTRY["experiments"]) == 55
    executable = [item for item in REGISTRY["experiments"] if item["executable"]]
    historical = [item for item in REGISTRY["experiments"] if not item["executable"]]
    assert len(executable) == 51
    assert all(item["status"] == "NOT_RUN" for item in executable)
    assert all(item["status"] == "HISTORICAL_EVIDENCE" for item in historical)


def test_historical_results_are_not_paper_final():
    historical = [item for item in REGISTRY["experiments"]
                  if item.get("evidence_class") == "HISTORICAL_EVIDENCE"]
    assert len(historical) == 4
    assert all(item["status"] != "PAPER_FINAL" for item in REGISTRY["experiments"])


def test_b1_is_marked_optimization_upper_bound():
    assert CONFIG["baselines"]["B1"]["role"] == "optimization_upper_bound"
    assert CONFIG["baselines"]["B1"]["inference_method"] is False


def test_b2_is_marked_seen_only():
    assert CONFIG["baselines"]["B2"]["seen_only"] is True
    assert CONFIG["baselines"]["B2"]["allowed_in_held_out"] is False


def test_o07_table_is_marked_failure():
    text = TABLE_FIGURE_PATH.read_text(encoding="utf-8")
    assert "Held-out diagnostic — FAIL" in text
    assert CONFIG["tables"]["table_4"]["status_label"] == "Held-out diagnostic — FAIL"


def test_paper_visuals_use_fixed_views():
    assert CONFIG["data"]["target_views"] == [
        "cond_000000", "cond_000318", "cond_000017", "cond_000347"
    ]
    assert all(figure["fixed_views"] for figure in CONFIG["figures"].values())


def test_no_cherry_pick_visual_export():
    export = CONFIG["visual_export"]
    assert export["allowed_operations"] == ["concatenate", "aspect_preserving_crop", "label", "uniform_crop"]
    assert set(export["forbidden_operations"]) == {
        "retouch", "artifact_removal", "color_change", "successful_region_selection"
    }


def test_historical_outputs_immutable():
    asset = next(item for item in MANIFEST["assets"]
                 if item["asset_id"] == "historical_formal_output_tree")
    assert asset["verification"] == "tree_metadata_sha256"
    assert asset["fingerprint"] == "108c8f2a7e327cd76a508c16e1797cd220d169cbff44091fc9cfefbd2911c74c"
    assert CONFIG["permissions"]["modify_historical_outputs"] is False
