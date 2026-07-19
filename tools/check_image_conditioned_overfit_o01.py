from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene.gaussian_clothing_residuals import (  # noqa: E402
    CHANNELS,
    GaussianClothingResiduals,
    apply_protected_full_residual_guard,
)
from scene.image_conditioned_dressable_model import ImageConditionedDressableModel  # noqa: E402
from scene.trusted_silhouette_semantics_v6_1 import build_trusted_silhouette_regions  # noqa: E402
from tools.run_image_conditioned_overfit_o01 import (  # noqa: E402
    CONDITIONS,
    FORBIDDEN_FORWARD_FIELDS,
    _forward_inputs,
    leave_one_out_reference_ids,
    load_contract,
)


def _residuals(requires_grad: bool = False) -> GaussianClothingResiduals:
    values = {
        "delta_xyz": torch.arange(15, dtype=torch.float32).reshape(5, 3),
        "delta_log_scaling": torch.arange(15, dtype=torch.float32).reshape(5, 3) + 1,
        "delta_rotvec": torch.arange(15, dtype=torch.float32).reshape(5, 3) + 2,
        "delta_opacity_logit": torch.arange(5, dtype=torch.float32).reshape(5, 1) + 3,
        "delta_sh0": torch.arange(15, dtype=torch.float32).reshape(5, 1, 3) + 4,
        "delta_shN": torch.arange(45, dtype=torch.float32).reshape(5, 3, 3) + 5,
    }
    if requires_grad:
        values = {name: value.requires_grad_() for name, value in values.items()}
    return GaussianClothingResiduals(**values)


def test_guard_all_channels_values_and_gradients() -> None:
    values = _residuals(requires_grad=True)
    before = {name: value.detach().clone() for name, value in values.as_dict().items()}
    mask = torch.tensor([True, False, True, False, False])
    guarded = apply_protected_full_residual_guard(values, mask)
    total = sum(value.sum() for value in guarded.as_dict().values())
    total.backward()
    for name in CHANNELS:
        output = getattr(guarded, name)
        source = getattr(values, name)
        assert torch.count_nonzero(output[mask]).item() == 0
        assert torch.equal(output[~mask], before[name][~mask])
        assert torch.equal(source.detach(), before[name]), f"guard mutated {name}"
        assert torch.count_nonzero(source.grad[mask]).item() == 0
        assert torch.equal(source.grad[~mask], torch.ones_like(source.grad[~mask]))


def test_guard_rejects_ambiguous_membership() -> None:
    values = _residuals()
    for invalid in (
        torch.zeros(5),
        torch.zeros(4, dtype=torch.bool),
        torch.zeros(5, 2, dtype=torch.bool),
    ):
        try:
            apply_protected_full_residual_guard(values, invalid)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f"invalid protected membership was accepted: {invalid.shape}")


def test_guard_is_after_interpolation_before_composition() -> None:
    source = inspect.getsource(ImageConditionedDressableModel.compute_online_six_channel_residuals)
    interpolation = source.index("interpolate_anchor_clothing_residuals")
    guard = source.index("apply_protected_full_residual_guard")
    assert interpolation < guard < source.index("return")


def test_leave_one_out_protocol() -> None:
    for target in CONDITIONS:
        references = leave_one_out_reference_ids(target)
        assert len(references) == 3
        assert target not in references
        assert set(references) == set(CONDITIONS).difference({target})


def test_forward_boundary_rejects_target_or_oracle() -> None:
    reference = torch.zeros(3, 1)
    episode = {
        "reference_images": reference,
        "reference_cloth_masks": reference,
        "reference_foreground_masks": reference,
        "reference_poses": reference,
        "reference_cameras": [{}, {}, {}],
        "reference_valid_mask": torch.ones(3),
    }
    geometry = {
        "deformation_fn": lambda *args: None,
        "surface_depth_maps": reference,
        "surface_alpha_maps": reference,
    }
    values = _forward_inputs(episode, geometry)
    assert not FORBIDDEN_FORWARD_FIELDS.intersection(values)
    assert set(values) == {
        "reference_images", "reference_cloth_masks", "reference_foreground_masks",
        "reference_poses", "reference_cameras", "reference_valid_mask",
        "deformation_fn", "surface_depth_maps", "surface_alpha_maps",
        "require_depth_visibility",
    }


def test_protected_priority_excludes_spill_hard_regions() -> None:
    shape = (1, 1, 8, 8)
    zeros = torch.zeros(shape)
    ones = torch.ones(shape)
    sample = {
        "target_protected_mask": zeros.clone(),
        "target_foreground_mask": zeros.clone(),
        "target_base_foreground_mask": zeros.clone(),
        "target_clothing_mask": zeros.clone(),
        "target_old_clothing_mask": zeros.clone(),
        "target_edit_mask": zeros.clone(),
        "target_edit_core_mask": zeros.clone(),
        "target_transition_mask": zeros.clone(),
        "target_preserve_mask": zeros.clone(),
    }
    sample["target_protected_mask"][..., 3, 3] = 1
    sample["target_base_foreground_mask"][..., 3, 3] = 1
    sample["target_clothing_mask"][..., 3, 3] = 1
    sample["target_edit_mask"][..., 3, 3] = 1
    regions = build_trusted_silhouette_regions(
        sample, ones.expand(1, 3, 8, 8), support_diagonal_ratio=0.002708497051881002,
        minimum_radius_pixels=1,
    )
    assert regions["protected_identity"][..., 3, 3].item() == 1
    for name in ("background", "target_garment", "old_garment_removal", "trusted_expansion", "trusted_removal"):
        assert regions[name][..., 3, 3].item() == 0, name


def test_frozen_config_contract() -> None:
    config = load_contract(PROJECT_ROOT / "configs/sprint/subject02_image_conditioned_overfit_o01_v1.yaml")
    assert config["training"]["total_steps"] == 2000
    assert config["training"]["smoke_steps"] == 20
    assert config["outfit_id"] == "O01"
    assert config["guard"]["target_independent"] is True
    assert config["guard"]["outfit_independent"] is True
    assert config["guard"]["cloud_region_independent"] is True
    assert config["permissions"]["screen_space_proxy"] is False


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS all {len(tests)} O01 overfit contract tests")


if __name__ == "__main__":
    main()
