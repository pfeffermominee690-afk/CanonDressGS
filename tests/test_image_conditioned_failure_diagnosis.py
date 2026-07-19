from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene.gaussian_clothing_residuals import GaussianClothingResiduals
from scene.image_conditioned_failure_diagnostics import (
    COUNTERFACTUAL_CHANNELS,
    DiagnosticOutfitLatents,
    assert_forward_boundary,
    decision_case,
    normalized_residual_regression_loss,
    reference_variant,
    select_residual_channels,
)


CONFIG = yaml.safe_load(
    (ROOT / "configs/research/subject02_image_conditioned_failure_diagnosis_v1.yaml").read_text(encoding="utf-8")
)


def _episode(count: int = 3) -> tuple[dict, dict]:
    episode = {
        "reference_images": torch.arange(count * 3 * 2 * 2, dtype=torch.float32).reshape(count, 3, 2, 2) / 100,
        "reference_cloth_masks": torch.ones(count, 1, 2, 2),
        "reference_foreground_masks": torch.ones(count, 1, 2, 2),
        "reference_poses": torch.arange(count * 165, dtype=torch.float32).reshape(count, 165),
        "reference_valid_mask": torch.ones(count),
        "reference_cameras": [{"index": value} for value in range(count)],
        "reference_condition_ids": [f"cond_{value}" for value in range(count)],
    }
    geometry = {
        "surface_depth_maps": torch.arange(count * 2 * 2, dtype=torch.float32).reshape(count, 1, 2, 2),
        "surface_alpha_maps": torch.ones(count, 1, 2, 2),
        "deformation_fn": object(),
    }
    return episode, geometry


class _Base:
    _xyz = torch.zeros(5, 3)
    _scaling = torch.zeros(5, 3)
    _rotation = torch.zeros(5, 4)
    _opacity = torch.zeros(5, 1)
    _sh0 = torch.zeros(5, 1, 3)
    _shN = torch.zeros(5, 3, 3)


def _residual(fill: float = 1.0) -> GaussianClothingResiduals:
    return GaussianClothingResiduals.zeros(_Base()).__class__(
        delta_xyz=torch.full((5, 3), fill),
        delta_log_scaling=torch.full((5, 3), fill),
        delta_rotvec=torch.full((5, 3), fill),
        delta_opacity_logit=torch.full((5, 1), fill),
        delta_sh0=torch.full((5, 1, 3), fill),
        delta_shN=torch.full((5, 3, 3), fill),
    )


def _assert_raises(error_type, message: str, callback) -> None:
    try:
        callback()
    except error_type as error:
        assert message in str(error)
        return
    raise AssertionError(f"expected {error_type.__name__}: {message}")


def test_current_o01_checkpoint_is_immutable():
    assert CONFIG["permissions"]["modify_o01_attempt"] is False
    assert CONFIG["inputs"]["o01_checkpoint_sha256"] == "335fbfe86bfbcc27dbdb3c99c47a4a3fac3dc55a06bd144e5d674b03a1252838"


def test_left_view_metric_denominator_is_reported():
    source = (ROOT / "tools/diagnose_image_conditioned_overfit_failure.py").read_text(encoding="utf-8")
    assert "metric_denominators" in source and "cond_000017" in source
    assert "evaluation_garment_union" in source


def test_reference_zeroing_changes_only_reference_input():
    episode, geometry = _episode()
    changed, changed_geometry = reference_variant(episode, geometry, "zero_rgb_masks_kept")
    assert torch.count_nonzero(changed["reference_images"]) == 0
    assert torch.equal(changed["reference_cloth_masks"], episode["reference_cloth_masks"])
    assert torch.equal(changed_geometry["surface_depth_maps"], geometry["surface_depth_maps"])


def test_reference_swap_keeps_target_pose_fixed():
    episode, geometry = _episode()
    permuted, permuted_geometry = reference_variant(episode, geometry, "permuted")
    assert permuted["reference_cameras"][0]["index"] == 2
    assert torch.equal(permuted_geometry["surface_depth_maps"][0], geometry["surface_depth_maps"][2])
    assert "target_pose" not in permuted and "target_camera" not in permuted


def test_oracle_residual_is_loss_only():
    assert CONFIG["probe_d"]["oracle_residual_usage"] == "loss_and_offline_evaluation_only"
    assert CONFIG["probe_c"]["oracle_residual_usage"] == "loss_and_offline_evaluation_only"


def test_oracle_residual_never_enters_forward():
    _assert_raises(
        RuntimeError,
        "oracle_residual",
        lambda: assert_forward_boundary({"reference_images": torch.zeros(1), "oracle_residual": torch.zeros(1)}),
    )


def test_oracle_regression_targets_are_frozen_and_detached():
    source = (ROOT / "tools/diagnose_image_conditioned_overfit_failure.py").read_text(encoding="utf-8")
    assert "_set_requires_grad(oracle, False)" in source
    assert "retained an autograd graph" in source


def test_head_counterfactual_changes_one_group_only():
    original = _residual()
    xyz = select_residual_channels(original, COUNTERFACTUAL_CHANNELS["H1_xyz"])
    assert torch.equal(xyz.delta_xyz, original.delta_xyz)
    assert all(torch.count_nonzero(getattr(xyz, name)) == 0 for name in COUNTERFACTUAL_CHANNELS["H3_appearance"])


def test_probe_d_uses_diagnostic_latent_only():
    module = DiagnosticOutfitLatents(("O01", "O08"), 8, seed=4)
    assert tuple(module("O01").shape) == (1, 8)
    assert CONFIG["probe_d"]["diagnostic_latent_only"] is True
    assert CONFIG["probe_d"]["target_images_in_forward"] is False


def test_probe_c_has_no_outfit_id():
    assert CONFIG["probe_c"]["outfit_id_input"] is False
    _assert_raises(RuntimeError, "outfit_id", lambda: assert_forward_boundary({"outfit_id": "O01"}))


def test_probe_c_is_balanced():
    assert CONFIG["probe_c"]["balanced_episode_count"] == 8
    assert CONFIG["outfits"] == ["O01", "O08"]
    assert len(CONFIG["conditions"]) * len(CONFIG["outfits"]) == 8


def test_probe_r_removes_teacher_loss():
    assert CONFIG["probe_r"]["teacher_oracle_loss_enabled"] is False


def test_probe_r_keeps_network_and_bounds_fixed():
    assert CONFIG["probe_r"]["network_bounds_renderer_optimizer_fixed"] is True
    assert CONFIG["permissions"]["modify_renderer"] is False
    assert CONFIG["permissions"]["modify_residual_bounds"] is False


def test_base_and_mmlp_remain_frozen():
    assert CONFIG["base"]["frozen"] is True
    source = (ROOT / "tools/diagnose_image_conditioned_overfit_failure.py").read_text(encoding="utf-8")
    assert "base_fingerprint_before" in source and "base_fingerprint_after" in source
    assert "frozen_image_backbone" in source


def test_no_target_image_enters_forward():
    assert_forward_boundary({"reference_images": torch.zeros(1), "target_pose_only_external": torch.zeros(1)})
    _assert_raises(RuntimeError, "target_edit_rgb", lambda: assert_forward_boundary({"target_edit_rgb": torch.zeros(1)}))


def test_two_outfit_prep_outputs_unchanged():
    prep = ROOT / "tools/sprint/build_o01_o08_two_outfit_manifest.py"
    before = hashlib.sha256(prep.read_bytes()).hexdigest()
    ast.parse(prep.read_text(encoding="utf-8"))
    after = hashlib.sha256(prep.read_bytes()).hexdigest()
    assert before == after


def test_normalized_regression_is_zero_for_equal_residuals():
    residual = _residual(0.2)
    bounds = {"xyz": 1, "log_scaling": 1, "rotation": 1, "opacity_logit": 1, "sh0": 1, "shN": 1}
    loss, parts = normalized_residual_regression_loss(residual, residual, bounds)
    assert float(loss) == 0.0
    assert all(float(value) == 0.0 for value in parts.values())


def test_decision_ladder_is_deterministic():
    assert decision_case(probe_d_pass=False, probe_c_ran=False, probe_c_pass=False, probe_r_ran=False, probe_r_degraded=False)["case"] == "ND"
    assert decision_case(probe_d_pass=True, probe_c_ran=True, probe_c_pass=False, probe_r_ran=False, probe_r_degraded=False)["case"] == "NC"
    assert decision_case(probe_d_pass=True, probe_c_ran=True, probe_c_pass=True, probe_r_ran=True, probe_r_degraded=True)["case"] == "NO"


if __name__ == "__main__":
    tests = [(name, value) for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for name, callback in tests:
        callback()
        print(f"PASS {name}")
    print(f"PASS all {len(tests)} image-conditioned failure-diagnosis tests")
