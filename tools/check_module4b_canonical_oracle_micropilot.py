from __future__ import annotations

import inspect
import tempfile
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene.full_attribute_oracle import AnchorResidualOracle, GaussianResidualOracle
from scene.gaussian_clothing_residuals import (
    GaussianClothingResiduals,
    axis_angle_to_quaternion_wxyz,
    compose_canonical_gaussian_overrides,
    interpolate_anchor_field,
    quaternion_multiply_wxyz,
)
from tools.run_module4b_canonical_oracle_micropilot import (
    _loss,
    _render,
    module4b_capacity_decision,
)
from utils.full_training_checkpoint_utils import (
    load_full_training_checkpoint,
    save_full_training_checkpoint,
)


class Base(torch.nn.Module):
    def __init__(self, count: int = 12) -> None:
        super().__init__()
        self._xyz = torch.nn.Parameter(torch.zeros(count, 3), requires_grad=False)
        self._scaling = torch.nn.Parameter(torch.zeros(count, 3), requires_grad=False)
        rotation = torch.zeros(count, 4); rotation[:, 0] = 1
        self._rotation = torch.nn.Parameter(rotation, requires_grad=False)
        self._opacity = torch.nn.Parameter(torch.zeros(count, 1), requires_grad=False)
        self._sh0 = torch.nn.Parameter(torch.zeros(count, 1, 3), requires_grad=False)
        self._shN = torch.nn.Parameter(torch.zeros(count, 3, 3), requires_grad=False)


def _anchor(base: Base, anchor_count: int = 4) -> AnchorResidualOracle:
    indices = torch.arange(base._xyz.shape[0]).reshape(-1, 1) % anchor_count
    weights = torch.ones_like(indices, dtype=torch.float32)
    return AnchorResidualOracle(base, anchor_count, indices, weights)


def test_gaussian_oracle_uses_one_shared_canonical_field() -> None:
    base = Base()
    oracle = GaussianResidualOracle(base)
    assert oracle.raw_xyz.shape == (12, 3)
    assert not any("condition" in name for name, _ in oracle.named_parameters())
    assert torch.equal(oracle(base).gaussian_residuals.delta_xyz, oracle(base).gaussian_residuals.delta_xyz)


def test_anchor_oracle_uses_one_shared_anchor_field() -> None:
    base = Base()
    oracle = _anchor(base)
    assert oracle.raw_xyz.shape == (4, 3)
    assert oracle(base).gaussian_residuals.delta_xyz.shape == (12, 3)


def test_oracle_has_no_condition_specific_residual() -> None:
    for oracle in (GaussianResidualOracle(Base()), _anchor(Base())):
        assert all(parameter.ndim >= 1 for _, parameter in oracle.named_parameters())
        assert not any(token in name for name, _ in oracle.named_parameters() for token in ("condition", "view", "frame", "camera"))


def test_gaussian_oracle_bypasses_image_conditioning() -> None:
    signature = inspect.signature(GaussianResidualOracle.forward)
    assert tuple(signature.parameters) == ("self", "base_model")
    source = inspect.getsource(GaussianResidualOracle.forward)
    assert not any(token in source for token in ("image", "reference", "teacher", "target"))


def test_anchor_oracle_uses_formal_interpolation() -> None:
    field = torch.tensor([[1.0], [3.0], [9.0]])
    indices = torch.tensor([[0, 1], [1, 2]])
    weights = torch.tensor([[0.25, 0.75], [0.5, 0.5]])
    assert torch.equal(interpolate_anchor_field(field, indices, weights), torch.tensor([[2.5], [6.0]]))
    assert "interpolate_anchor_clothing_residuals" in inspect.getsource(AnchorResidualOracle.forward)


def test_oracle_uses_v5_3_dual_target_loss() -> None:
    source = inspect.getsource(_loss)
    assert "region_aware_dual_target_loss" in source
    assert "transition_alpha_target" in source
    assert "target_edit_rgb" in source and "target_base_rgb" in source


def test_oracle_residual_bounds_match_formal_model() -> None:
    oracle = GaussianResidualOracle(Base())
    oracle.configure_stage(3)
    with torch.no_grad():
        for name, parameter in oracle.named_parameters():
            if name.startswith("raw_"):
                parameter.fill_(20)
    output = oracle(Base())
    assert output.raw_residuals.delta_xyz.abs().max() <= 0.05 + 1e-6
    assert output.raw_residuals.delta_log_scaling.abs().max() <= 0.35 + 1e-6
    assert torch.linalg.vector_norm(output.raw_residuals.delta_rotvec, dim=-1).max() <= 0.2617993878 + 1e-6
    assert output.raw_residuals.delta_opacity_logit.abs().max() <= 2.0 + 1e-6
    assert output.raw_residuals.delta_sh0.abs().max() <= 0.25 + 1e-6
    assert torch.count_nonzero(output.raw_residuals.delta_shN) == 0


def test_oracle_quaternion_composition_matches_model() -> None:
    base = Base(2)
    rotvec = torch.tensor([[0.1, -0.2, 0.3], [0.0, 0.2, 0.0]])
    zeros = GaussianClothingResiduals.zeros(base)
    residuals = GaussianClothingResiduals(
        delta_xyz=zeros.delta_xyz,
        delta_log_scaling=zeros.delta_log_scaling,
        delta_rotvec=rotvec,
        delta_opacity_logit=zeros.delta_opacity_logit,
        delta_sh0=zeros.delta_sh0,
        delta_shN=zeros.delta_shN,
    )
    composed = compose_canonical_gaussian_overrides(base, residuals).rotation
    expected = torch.nn.functional.normalize(
        quaternion_multiply_wxyz(base._rotation, axis_angle_to_quaternion_wxyz(rotvec)), dim=-1,
    )
    assert torch.allclose(composed, expected, atol=1e-7, rtol=0)


def test_oracle_frozen_base_and_pose_module() -> None:
    base = Base()
    oracle = GaussianResidualOracle(base); oracle.configure_stage(3)
    oracle(base).canonical_overrides.xyz.square().sum().backward()
    assert all(parameter.grad is None for parameter in base.parameters())
    assert all(not parameter.requires_grad for parameter in base.parameters())


def test_oracle_checkpoint_resume_exact() -> None:
    base = Base(); oracle = GaussianResidualOracle(base); oracle.configure_stage(3)
    groups, names = oracle.parameter_groups({
        "geometry_residuals": 1e-3, "appearance_residuals": 2e-3,
        "geometry_gate": 1e-3, "appearance_gate": 1e-3,
    })
    optimizer = torch.optim.Adam(groups)
    oracle(base).canonical_overrides.xyz.sum().backward(); optimizer.step()
    model_before = {name: value.detach().clone() for name, value in oracle.state_dict().items()}
    optimizer_before = optimizer.state_dict()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.pth"
        save_full_training_checkpoint(
            path, model=oracle, optimizer=optimizer, optimizer_group_names=names,
            scheduler=None, scaler=None, training_state={"global_step": 40},
            data_state={"fixture": "module4b"}, method_state={"kind": "gaussian"},
        )
        replacement = GaussianResidualOracle(base); replacement.configure_stage(3)
        replacement_groups, replacement_names = replacement.parameter_groups({
            "geometry_residuals": 1e-3, "appearance_residuals": 2e-3,
            "geometry_gate": 1e-3, "appearance_gate": 1e-3,
        })
        replacement_optimizer = torch.optim.Adam(replacement_groups)
        payload = load_full_training_checkpoint(
            path, model=replacement, optimizer=replacement_optimizer,
            optimizer_group_names=replacement_names, scheduler=None, scaler=None,
            expected_data_state={"fixture": "module4b"}, expected_method_state={"kind": "gaussian"},
        )
        assert payload["training_state"]["global_step"] == 40
        assert all(torch.equal(model_before[name], value) for name, value in replacement.state_dict().items())
        assert optimizer_before.keys() == replacement_optimizer.state_dict().keys()


def test_oracle_target_fields_are_loss_only() -> None:
    render_source = inspect.getsource(_render)
    for field in ("target_edit_rgb", "target_base_rgb", "target_foreground_mask", "target_clothing_mask", "target_protected_mask"):
        assert field not in render_source
    loss_source = inspect.getsource(_loss)
    assert all(field in loss_source for field in ("target_edit_rgb", "target_base_rgb", "target_foreground_mask", "target_clothing_mask", "target_protected_mask"))


def test_module4b_capacity_decision_matrix() -> None:
    strong = {outfit: "STRONG_FIT" for outfit in ("O00", "O01", "O05")}
    weak = dict(strong); weak["O05"] = "NO_FIT"
    assert module4b_capacity_decision(strong, strong, True) == "A"
    assert module4b_capacity_decision(strong, weak, False) == "B"
    assert module4b_capacity_decision(strong, strong, False) == "C"
    assert module4b_capacity_decision(weak, weak, False) == "D"


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"{test.__name__}: PASS")
    print(f"module4b oracle checks: PASS ({len(tests)} tests)")


if __name__ == "__main__":
    main()
