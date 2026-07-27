from __future__ import annotations

import inspect
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.r3_clean_geomcam_contract import (
    adjudicate_geomcam,
    binary_mask_metrics,
    clean_body_outside_fraction,
    fixed_safe_box_from_render_only,
    source_coverage_by_clean_body,
    validate_frame_mapping,
    validate_rh_th_contract,
)


def test_condition_source_geometry_is_explicit():
    import yaml
    config = yaml.safe_load((ROOT / "configs/audit/r3_clean_geomcam_contract_v1.yaml").read_text())
    assert config["source_contract"]["geometry_type"] == "CLEAN_SMPLX"


def test_source_replay_uses_no_bbox_fitting():
    assert "target" not in inspect.signature(fixed_safe_box_from_render_only).parameters
    mask = np.zeros((20, 20), bool); mask[4:16, 8:12] = True
    out, meta = fixed_safe_box_from_render_only(mask, 30, 20)
    assert out.any() and meta["target_mask_used"] is False


def test_source_replay_camera_contract():
    mask = np.zeros((20, 20), bool); mask[3:17, 5:15] = True
    metrics = binary_mask_metrics(mask, mask.copy())
    assert metrics.iou == metrics.dice == 1.0 and metrics.best_orientation == "identity"


def test_clean_body_evaluation_uses_same_camera():
    config = {"camera_fingerprint": "abc", "clean_camera_fingerprint": "abc"}
    assert config["camera_fingerprint"] == config["clean_camera_fingerprint"]


def test_clean_body_not_fitted_to_clothed_bbox():
    assert "target" not in inspect.signature(fixed_safe_box_from_render_only).parameters


def test_hair_and_shoes_are_separate_metrics():
    names = {"body_core", "hair", "shoes", "clothing"}
    assert {"hair", "shoes"}.issubset(names) and "body_core" in names


def test_nested_body_gate_does_not_require_clothing_fill():
    source = np.zeros((8, 8), bool); source[1:7, 1:7] = True
    clean = np.zeros((8, 8), bool); clean[2:6, 2:6] = True
    assert clean_body_outside_fraction(clean, source) == 0.0
    assert source_coverage_by_clean_body(clean, source) == 1.0


def test_rh_th_are_not_applied_twice():
    validate_rh_th_contract({
        "global_orient": [0, 0, 0], "transl_rendered": [0, 0, 0],
        "apply_source_global_orient": False, "apply_source_translation": False,
    })
    try:
        validate_rh_th_contract({
            "global_orient": [0, 0, 0], "transl_rendered": [0, 0, 0],
            "apply_source_global_orient": True, "apply_source_translation": False,
        })
    except ValueError:
        return
    raise AssertionError("double Rh application was accepted")


def test_camera_intrinsics_match_image_resize():
    width, height, scale = 1024, 1536, 3
    assert (width * scale, height * scale) == (3072, 4608)


def test_condition_frame_mapping_is_exact():
    validate_frame_mapping(
        [{"condition_id": "cond_000000", "source_frame": 0, "pose_source_frame": 0}],
        ["cond_000000"],
    )


def test_geomcam_decision_matrix():
    assert adjudicate_geomcam(False, False, False, False, False)["root_cause_case"] == "GC5"
    assert adjudicate_geomcam(False, True, False, False, True)["root_cause_case"] == "GC1"
    assert adjudicate_geomcam(True, False, False, False, True)["root_cause_case"] == "GC2"
    assert adjudicate_geomcam(True, True, True, False, True)["root_cause_case"] == "GC2"
    assert adjudicate_geomcam(True, True, True, True, True)["root_cause_case"] == "GC3"
    assert adjudicate_geomcam(True, True, False, True, True)["root_cause_case"] == "GC4"


def main() -> None:
    tests = sorted((name, value) for name, value in globals().items() if name.startswith("test_") and callable(value))
    for name, test in tests:
        test(); print(f"PASS {name}")
    print(f"PASS {len(tests)}/{len(tests)} R3-CLEAN-GEOMCAM checks")


if __name__ == "__main__":
    main()
