from __future__ import annotations

from functools import lru_cache
import inspect
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene.gaussian_clothing_residuals import (  # noqa: E402
    GaussianClothingResiduals,
    axis_angle_to_quaternion_wxyz,
    compose_canonical_gaussian_overrides,
    quaternion_multiply_wxyz,
)
from scene.o00_arm_support import (  # noqa: E402
    ARM_SKIN_REGION_NAMES,
    Subject02ArmSkinField,
    adjudicate_o00_support,
    build_o00_arm_support,
    classify_arm_skin_regions,
    classify_sampled_arm_skin_regions,
    deform_o00_arm_support,
    support_tensor_fingerprint,
)
from scene.o00_fixed_open_oracle import FixedOpenGaussianOracle, fixed_open_oracle_contract  # noqa: E402
from scene.r3_body_support_probe import ARM_PARTS, HAND_AND_FINGER_PARTS, covered_support_visibility_fraction  # noqa: E402
from tools.run_module4b_canonical_oracle_micropilot import _loss  # noqa: E402


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


def _surface_fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Subject02ArmSkinField]:
    vertices = []; faces = []; weights = []
    for part in ARM_PARTS:
        side = -1.0 if part in (16, 18) else 1.0
        proximal = 0.4 if part in (16, 17) else 0.9
        start = len(vertices)
        vertices.extend([
            [side * proximal, -0.1, 0.0],
            [side * (proximal + 0.25), 0.1, 0.0],
            [side * (proximal + 0.12), 0.0, 0.25],
        ])
        faces.append([start, start + 1, start + 2])
        for _ in range(3):
            row = [0.0] * 55; row[part] = 1.0; weights.append(row)
    hand_start = len(vertices)
    vertices.extend([[1.8, 0.0, 0.0], [1.9, 0.1, 0.0], [1.9, 0.0, 0.1]])
    faces.append([hand_start, hand_start + 1, hand_start + 2])
    for _ in range(3):
        row = [0.0] * 55; row[20] = 1.0; weights.append(row)
    xyz = torch.tensor(vertices, dtype=torch.float32)
    triangle = torch.tensor(faces, dtype=torch.long)
    lbs = torch.tensor(weights, dtype=torch.float32)
    region = classify_arm_skin_regions(xyz, lbs)
    color = torch.zeros(len(xyz), 3)
    palette = torch.tensor([
        [0.42, 0.31, 0.25], [0.45, 0.34, 0.27], [0.47, 0.36, 0.29],
        [0.50, 0.39, 0.31], [0.44, 0.33, 0.26], [0.48, 0.37, 0.30],
    ])
    arm = region >= 0; color[arm] = palette[region[arm]]; color[~arm] = palette.mean(0)
    field = Subject02ArmSkinField(
        rgb=color, confidence=torch.ones(len(xyz)), provenance=torch.zeros(len(xyz), dtype=torch.long),
        region=region, body_part=lbs.argmax(1), source_vertex=torch.arange(len(xyz)),
        metadata={
            "identity": "subject02", "observed_pixel_count": 63_445,
            "external_identity_used": False, "regions": {name: {} for name in ARM_SKIN_REGION_NAMES},
        },
    ).validate()
    return xyz, triangle, lbs, field


@lru_cache(maxsize=1)
def _support():
    vertices, faces, lbs, field = _surface_fixture()
    return build_o00_arm_support(
        vertices, faces, lbs, field, count=12_000, seed=7,
        inward_offset=0.007, opacity_scale=0.25,
    )


def test_o00_support_uses_subject02_skin_only() -> None:
    support = _support()
    assert support.metadata["appearance_identity"] == "subject02"
    assert support.metadata["skin_observed_pixel_count"] == 63_445
    assert support.metadata["external_identity_or_generated_texture_used"] is False
    assert set(support.skin_region.tolist()) == set(range(6))
    assert torch.unique(support.rgb, dim=0).shape[0] > 1
    relabeled = classify_sampled_arm_skin_regions(support.canonical_xyz, support.body_part)
    assert torch.equal(relabeled, support.skin_region)


def test_o00_support_uses_formal_lbs() -> None:
    support = _support()
    rigid = torch.eye(4).repeat(55, 1, 1)
    rigid[:, 0, 3] = torch.arange(55).float() / 100
    posed = deform_o00_arm_support(support, rigid, torch.eye(3), torch.tensor([0.0, 1.0, 0.0]))
    expected_x = support.canonical_xyz[:, 0] + (support.lbs_weights * torch.arange(55).float()[None] / 100).sum(1)
    assert torch.allclose(posed["xyz"][:, 0], expected_x, atol=1e-6, rtol=0)
    assert torch.allclose(posed["xyz"][:, 1], support.canonical_xyz[:, 1] + 1.0, atol=1e-6, rtol=0)


def test_o00_support_excludes_hands() -> None:
    support = _support()
    assert set(support.body_part.tolist()).issubset(set(ARM_PARTS))
    assert not set(support.body_part.tolist()).intersection(HAND_AND_FINGER_PARTS)
    assert int(support.face_indices.max()) < 4


def test_o00_support_is_frozen() -> None:
    support = _support()
    assert not isinstance(support, torch.nn.Module)
    assert support.count == 12_000
    for value in support.tensor_dict().values():
        assert not value.requires_grad and not isinstance(value, torch.nn.Parameter)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    moved = support.to(device)
    assert moved.lbs_weights.device.type == device
    assert moved.quaternion_wxyz.device.type == device


def test_o00_support_does_not_modify_base() -> None:
    base = Base(); before = {name: value.detach().clone() for name, value in base.state_dict().items()}
    first = support_tensor_fingerprint(_support()); second = support_tensor_fingerprint(_support())
    assert first == second
    assert all(torch.equal(before[name], value) for name, value in base.state_dict().items())


def test_o00_covered_visibility_threshold() -> None:
    rgb = torch.rand(3, 10, 10); alpha = torch.ones(1, 10, 10)
    assert float(covered_support_visibility_fraction(rgb, alpha, rgb.clone(), alpha.clone())) == 0
    changed = rgb.clone(); changed[:, 0, 0] += 0.02
    assert float(covered_support_visibility_fraction(rgb, alpha, changed, alpha)) <= 0.01


def test_o00_revealed_hole_repair_threshold() -> None:
    rows = [{
        "hole_repair_recall": value, "background_leakage": 1 - value,
        "covered_support_visibility": 0.005, "finite": True, "pose_explosion": False,
    } for value in (0.99, 0.98, 0.97, 0.96)]
    decision = adjudicate_o00_support(
        rows, outside_fraction=0.0, base_bitwise_exact=True,
        shoulder_visual="WARN", wrist_visual="WARN",
    )
    assert decision["status"] == "ARM_SUPPORT_PASS" and decision["oracle_allowed"]


def test_o00_fixed_open_oracle_has_no_gate_parameters() -> None:
    oracle = FixedOpenGaussianOracle(Base())
    contract = fixed_open_oracle_contract(oracle)
    assert contract["pass"] and contract["gate_parameter_names"] == []
    output = oracle(Base())
    assert torch.equal(output.geometry_gate, torch.ones_like(output.geometry_gate))
    assert torch.equal(output.appearance_gate, torch.ones_like(output.appearance_gate))


def test_o00_oracle_uses_shared_canonical_residual() -> None:
    oracle = FixedOpenGaussianOracle(Base())
    names = [name for name, _ in oracle.named_parameters()]
    assert oracle.raw_xyz.shape == (12, 3)
    assert not any(token in name for name in names for token in ("condition", "view", "camera"))
    assert inspect.signature(oracle.forward).parameters.keys() == {"base_model"}


def test_o00_oracle_uses_r2_rotation_path() -> None:
    base = Base(2)
    zeros = GaussianClothingResiduals.zeros(base)
    rotvec = torch.tensor([[0.1, -0.2, 0.3], [0.0, 0.2, 0.0]])
    residual = GaussianClothingResiduals(
        delta_xyz=zeros.delta_xyz, delta_log_scaling=zeros.delta_log_scaling,
        delta_rotvec=rotvec, delta_opacity_logit=zeros.delta_opacity_logit,
        delta_sh0=zeros.delta_sh0, delta_shN=zeros.delta_shN,
    )
    actual = compose_canonical_gaussian_overrides(base, residual).rotation
    expected = torch.nn.functional.normalize(
        quaternion_multiply_wxyz(base._rotation, axis_angle_to_quaternion_wxyz(rotvec)), dim=-1,
    )
    assert torch.allclose(actual, expected, atol=1e-7, rtol=0)
    assert "compose_canonical_gaussian_overrides" in inspect.getsource(FixedOpenGaussianOracle.forward)


def test_o00_oracle_uses_v5_3_loss() -> None:
    source = inspect.getsource(_loss)
    assert "region_aware_dual_target_loss" in source
    assert "transition_alpha_target" in source
    assert "target_edit_rgb" in source and "target_base_rgb" in source


def test_o00_support_decision_matrix() -> None:
    passing = [{
        "hole_repair_recall": 0.98, "background_leakage": 0.02,
        "covered_support_visibility": 0.005, "finite": True, "pose_explosion": False,
    }] * 4
    failing = [dict(passing[0], covered_support_visibility=0.011)] * 4
    assert adjudicate_o00_support(
        passing, outside_fraction=0.0, base_bitwise_exact=True,
        shoulder_visual="WARN", wrist_visual="PASS",
    )["status"] == "ARM_SUPPORT_PASS"
    assert adjudicate_o00_support(
        failing, outside_fraction=0.0, base_bitwise_exact=True,
        shoulder_visual="PASS", wrist_visual="PASS",
    )["status"] == "ARM_SUPPORT_FAIL"


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test(); print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)}/{len(tests)} O00 arm-support closure checks")


if __name__ == "__main__":
    main()
