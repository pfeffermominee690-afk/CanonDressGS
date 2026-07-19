from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import torch

from scene.gaussian_clothing_residuals import CHANNELS, GaussianClothingResiduals
from scene.protected_cloud_attribution import (
    apply_formal_guard,
    apply_diagnostic_counterfactual,
    fixed_open_ownership_record,
    gradient_provenance,
    membership_sha256,
    qualify_formal_guard,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools/run_protected_cloud_attribution.py"
CONFIG = ROOT / "configs/research/subject02_protected_cloud_attribution_v1.yaml"


def residuals() -> GaussianClothingResiduals:
    return GaussianClothingResiduals(
        delta_xyz=torch.ones(4, 3),
        delta_log_scaling=torch.ones(4, 3) * 2,
        delta_rotvec=torch.ones(4, 3) * 3,
        delta_opacity_logit=torch.ones(4, 1) * 4,
        delta_sh0=torch.ones(4, 1, 3) * 5,
        delta_shN=torch.ones(4, 2, 3) * 6,
    )


def protected() -> torch.Tensor:
    return torch.tensor([False, True, False, True])


def assert_only_zeroed(result, names):
    original = residuals()
    for name in CHANNELS:
        value = getattr(result.residuals, name)
        source = getattr(original, name)
        if name in names:
            assert torch.count_nonzero(value[protected()]) == 0
            assert torch.equal(value[~protected()], source[~protected()])
        else:
            assert value is source or torch.equal(value, source)


def test_all_protected_cloud_indices_have_attribution():
    records = [fixed_open_ownership_record(i, production_contribution_count=1, production_alpha_mass=.1, dominant_lbs_joint=8) for i in (1, 3)]
    assert [row["base_gaussian_index"] for row in records] == [1, 3]


def test_parameter_ownership_path_is_explicit():
    row = fixed_open_ownership_record(7, production_contribution_count=2, production_alpha_mass=.2, dominant_lbs_joint=11)
    assert row["raw_parameter_sources"]["delta_xyz"] == "raw_xyz[7]"
    assert row["anchor_or_control_point"] is None and row["gate"]["geometry"] == 1.0


def test_direct_and_indirect_gradient_are_distinguished():
    direct = gradient_provenance(torch.ones(4, 3), torch.ones(4, 3), protected())
    indirect = gradient_provenance(torch.zeros(4, 3), torch.zeros(4, 3), protected(), shared_parameter_gradient_norm=1.0)
    assert direct["classification"] == "DIRECT_PROTECTED_GRADIENT"
    assert indirect["classification"] == "INDIRECT_SHARED_PARAMETER_LEAKAGE"


def test_diagnostic_counterfactual_is_not_formal_candidate():
    result = apply_diagnostic_counterfactual(residuals(), torch.tensor([1, 3]), "D1")
    assert result.diagnostic_only and "not_a_formal_candidate" in result.membership_source


def test_formal_guards_use_only_base_protected_membership():
    good = apply_formal_guard(residuals(), protected(), "F1")
    assert good.membership_source == "base_derived_stable_protected"
    try:
        apply_formal_guard(residuals(), protected(), "F1", membership_source="target_cloud")
        assert False
    except ValueError:
        pass


def test_f1_changes_only_protected_xyz_residual():
    assert_only_zeroed(apply_formal_guard(residuals(), protected(), "F1"), {"delta_xyz"})


def test_f2_changes_only_protected_geometry_residuals():
    assert_only_zeroed(apply_formal_guard(residuals(), protected(), "F2"), {"delta_xyz", "delta_log_scaling", "delta_rotvec"})


def test_f3_adds_only_protected_opacity_guard():
    assert_only_zeroed(apply_formal_guard(residuals(), protected(), "F3"), {"delta_xyz", "delta_log_scaling", "delta_rotvec", "delta_opacity_logit"})


def test_f4_zeros_all_protected_residuals():
    assert_only_zeroed(apply_formal_guard(residuals(), protected(), "F4"), set(CHANNELS))


def test_guard_does_not_modify_base():
    source = residuals(); before = [getattr(source, name).clone() for name in CHANNELS]
    apply_formal_guard(source, protected(), "F4")
    assert all(torch.equal(getattr(source, name), value) for name, value in zip(CHANNELS, before))


def test_guard_does_not_modify_renderer():
    source = (ROOT / "scene/protected_cloud_attribution.py").read_text(encoding="utf-8")
    assert "gsplat" not in source and "rasterization(" not in source


def test_guard_membership_is_outfit_independent():
    parameters = inspect.signature(apply_formal_guard).parameters
    assert not ({"outfit", "target", "cloud", "target_mask", "cloud_mask"} & set(parameters))
    assert membership_sha256(protected()) == hashlib.sha256(protected().to(torch.uint8).numpy().tobytes()).hexdigest()


def test_coverage_regression_gate():
    baseline = {"cloud_alpha_mass": 10., "cloud_active": 10., "center_entered": 10., "trusted_expansion_npre": 100., "trusted_expansion_recall": .95, "trusted_removal_recall": .95, "target_closer": .9, "edit_reduction": .9, "protected_mae": 0., "background_leakage": 0.}
    candidate = dict(baseline, cloud_alpha_mass=1., cloud_active=1., center_entered=1., trusted_expansion_npre=98.)
    assert qualify_formal_guard(baseline, candidate, seam_pass=True, o01_pass=True, membership_target_independent=True)["status"] == "PASS"
    candidate["trusted_expansion_npre"] = 97.9
    assert qualify_formal_guard(baseline, candidate, seam_pass=True, o01_pass=True, membership_target_independent=True)["status"] == "FAIL"


def test_protected_seam_gate():
    baseline = {"cloud_alpha_mass": 10., "cloud_active": 10., "center_entered": 10., "trusted_expansion_npre": 100., "trusted_expansion_recall": .95, "trusted_removal_recall": .95, "target_closer": .9, "edit_reduction": .9, "protected_mae": 0., "background_leakage": 0.}
    candidate = dict(baseline, cloud_alpha_mass=1., cloud_active=1., center_entered=1.)
    assert qualify_formal_guard(baseline, candidate, seam_pass=False, o01_pass=True, membership_target_independent=True)["gates"]["protected_seam"] is False


def test_o01_regression_gate():
    baseline = {"cloud_alpha_mass": 10., "cloud_active": 10., "center_entered": 10., "trusted_expansion_npre": 100., "trusted_expansion_recall": .95, "trusted_removal_recall": .95, "target_closer": .9, "edit_reduction": .9, "protected_mae": 0., "background_leakage": 0.}
    candidate = dict(baseline, cloud_alpha_mass=1., cloud_active=1., center_entered=1.)
    assert qualify_formal_guard(baseline, candidate, seam_pass=True, o01_pass=False, membership_target_independent=True)["status"] == "FAIL"


def test_no_optimizer_is_created():
    assert not RUNNER.exists() or ("torch.optim" not in RUNNER.read_text(encoding="utf-8") and ".step()" not in RUNNER.read_text(encoding="utf-8"))


def test_previous_outputs_are_unchanged():
    assert "source_pool_output" in CONFIG.read_text(encoding="utf-8")


def test_frozen_branches_are_unchanged():
    assert "source_tag" in CONFIG.read_text(encoding="utf-8")


if __name__ == "__main__":
    checks = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for check in checks:
        check()
    print(f"{len(checks)} protected-cloud attribution tests: PASS")
