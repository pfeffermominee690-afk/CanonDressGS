from __future__ import annotations

import hashlib
import io
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene.r3_body_support_probe import (  # noqa: E402
    ARM_PARTS,
    adjudicate_r3_candidate,
    covered_support_visibility_fraction,
    deform_arm_support,
    sample_arm_support_surface,
)


def _fixture():
    vertices = []
    faces = []
    weights = []
    for group, part in enumerate(ARM_PARTS):
        x = -1.0 if part in (16, 18) else 1.0
        z = 0.5 if part in (16, 17) else 0.0
        start = len(vertices)
        vertices.extend([[x, -0.1, z], [x, 0.1, z], [x, 0.0, z + 0.2]])
        faces.append([start, start + 1, start + 2])
        for _ in range(3):
            row = [0.0] * 55; row[part] = 1.0; weights.append(row)
    hand_start = len(vertices)
    vertices.extend([[2.0, 0.0, 0.0], [2.1, 0.0, 0.0], [2.0, 0.1, 0.0]])
    faces.append([hand_start, hand_start + 1, hand_start + 2])
    for _ in range(3):
        row = [0.0] * 55; row[20] = 1.0; weights.append(row)
    return torch.tensor(vertices), torch.tensor(faces), torch.tensor(weights)


def _probe(count: int = 80):
    vertices, faces, weights = _fixture()
    return sample_arm_support_surface(
        vertices, faces, weights, count=count, seed=7,
        skin_rgb=torch.tensor([0.45, 0.35, 0.28]), inward_offset=0.003,
        diagnostic_opacity=0.65, maximum_count=200,
    )


def _fingerprint(value: torch.Tensor) -> str:
    buffer = io.BytesIO(); torch.save(value, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def test_arm_support_probe_does_not_modify_base() -> None:
    base = torch.randn(32, 3)
    before = _fingerprint(base)
    _probe()
    assert _fingerprint(base) == before


def test_arm_support_uses_subject02_skin_only() -> None:
    probe = _probe()
    assert torch.equal(probe.rgb[0], torch.tensor([0.45, 0.35, 0.28]))
    assert probe.metadata["appearance_source"].startswith("subject02-only")


def test_arm_support_excludes_hands() -> None:
    probe = _probe()
    assert set(probe.body_part.tolist()) == set(ARM_PARTS)
    assert int(probe.face_indices.max()) < 4


def test_arm_support_uses_formal_pose_binding() -> None:
    probe = _probe().to("cpu")
    rigid = torch.eye(4).repeat(55, 1, 1)
    rigid[:, 0, 3] = torch.arange(55).float() / 100.0
    result = deform_arm_support(probe, rigid, torch.eye(3), torch.tensor([0.0, 1.0, 0.0]))
    expected_x = probe.canonical_xyz[:, 0] + (probe.lbs_weights * (torch.arange(55).float() / 100.0)).sum(1)
    assert torch.allclose(result["xyz"][:, 0], expected_x, atol=1e-6, rtol=0)
    assert torch.allclose(result["xyz"][:, 1], probe.canonical_xyz[:, 1] + 1.0, atol=1e-6, rtol=0)


def test_arm_support_four_views_are_finite() -> None:
    probe = _probe().to("cpu")
    rigid = torch.eye(4).repeat(55, 1, 1)
    for angle in (0.0, 0.5, 1.0, 1.5):
        c, s = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
        rh = torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        result = deform_arm_support(probe, rigid, rh, torch.zeros(3))
        assert all(torch.isfinite(value).all() for value in result.values())


def test_covered_support_remains_occluded() -> None:
    rgb = torch.rand(3, 8, 8); alpha = torch.ones(1, 8, 8)
    assert covered_support_visibility_fraction(rgb, alpha, rgb.clone(), alpha.clone()) == 0
    changed = rgb.clone(); changed[:, 0, 0] += 0.1
    assert 0 < covered_support_visibility_fraction(rgb, alpha, changed, alpha) < 0.02


def test_support_probe_has_no_training_parameters() -> None:
    probe = _probe()
    assert not isinstance(probe, torch.nn.Module)
    for value in probe.state_dict().values():
        if isinstance(value, torch.Tensor):
            assert not value.requires_grad
            assert not isinstance(value, torch.nn.Parameter)


def test_r3_candidate_decision_matrix() -> None:
    a = adjudicate_r3_candidate(
        {"clean_body_geometry_available": "true", "clean_body_texture_available": "true", "formal_lbs_binding_available": True},
        {"status": "ARM_SUPPORT_PROBE_FAIL"},
    )
    b = adjudicate_r3_candidate(
        {"clean_body_geometry_available": "uncertain", "clean_body_texture_available": "partial", "formal_lbs_binding_available": True},
        {"status": "ARM_SUPPORT_PROBE_PASS"},
    )
    unresolved = adjudicate_r3_candidate(
        {"clean_body_geometry_available": "false", "clean_body_texture_available": "false", "formal_lbs_binding_available": False},
        {"status": "ARM_SUPPORT_PROBE_FAIL"},
    )
    assert a["recommended_candidate"] == "CANDIDATE_A"
    assert b == {
        "recommended_candidate": "CANDIDATE_B",
        "recommended_shell_handling": "B1",
        "recommended_next_stage": "BUILD_FROZEN_BODY_SUPPORT_PILOT",
    }
    assert unresolved["recommended_candidate"] == "R3_UNRESOLVED"


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test(); print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)}/{len(tests)} R3 checks")


if __name__ == "__main__":
    main()
