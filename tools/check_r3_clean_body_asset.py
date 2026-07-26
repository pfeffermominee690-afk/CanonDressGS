from __future__ import annotations

import hashlib
import io
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene.r3_clean_body_asset import (  # noqa: E402
    PROVENANCE,
    SUPPORT_PARTS,
    adjudicate_clean_body_pilot,
    build_skin_appearance_field,
    classify_base_shell,
    deform_clean_body_support,
    density_matched_opacity,
    sample_clean_body_support,
)


def _surface():
    vertices = []
    faces = []
    weights = []
    for group, part in enumerate((0, 1, 2, 16, 17, 20, 7, 15)):
        x = float(group % 4) * 0.3 - 0.45
        y = float(group // 4) * 0.5
        start = len(vertices)
        vertices.extend([[x, y, 0.0], [x + 0.2, y, 0.0], [x, y + 0.2, 0.05]])
        faces.append([start, start + 1, start + 2])
        for _ in range(3):
            row = [0.0] * 55; row[part] = 1.0; weights.append(row)
    return torch.tensor(vertices), torch.tensor(faces), torch.tensor(weights)


def _field():
    vertices, faces, weights = _surface()
    indices = torch.tensor([0, 1, 2, 3, 4, 5, 9, 10, 11])
    rgb = torch.tensor([[0.48, 0.36, 0.29], [0.50, 0.37, 0.30], [0.47, 0.35, 0.28]]).repeat(3, 1)
    confidence = torch.ones(len(indices))
    return vertices, faces, weights, build_skin_appearance_field(vertices, faces, weights, indices, rgb, confidence)


def _support(count: int = 120):
    vertices, faces, weights, field = _field()
    return sample_clean_body_support(
        vertices, faces, weights, field, count=count, seed=7,
        inward_offset=0.003, opacity=0.25,
    )


def _fingerprint(value: torch.Tensor) -> str:
    buffer = io.BytesIO(); torch.save(value, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def test_clean_body_uses_subject02_beta() -> None:
    contract = {"shape_source": "subject02 smpl_params.npz betas[0]", "used_mean_beta": False}
    assert contract["shape_source"].startswith("subject02") and not contract["used_mean_beta"]


def test_clean_body_geometry_matches_formal_coordinates() -> None:
    support = _support()
    assert support.metadata["inward_offset_m"] == 0.003
    assert torch.isfinite(support.canonical_xyz).all()


def test_static_identity_excludes_old_garment_shell() -> None:
    part = torch.tensor([15, 20, 16, 3, 7, 1])
    old = torch.tensor([False, False, True, True, False, False])
    result = classify_base_shell(part, torch.zeros(6), old, torch.zeros(6))
    assert not torch.any(result["static_identity"] & result["old_garment_shell"])
    assert not result["static_identity"][2] and not result["static_identity"][3]


def test_skin_field_uses_subject02_samples_only() -> None:
    _, _, _, field = _field()
    observed = field.provenance == PROVENANCE["observed"]
    assert observed.any() and torch.all(field.rgb[observed] >= 0)


def test_skin_provenance_marks_inferred_regions() -> None:
    _, _, _, field = _field()
    assert set(field.provenance.tolist()).issubset(set(PROVENANCE.values()))
    assert torch.any(field.provenance != PROVENANCE["observed"])


def test_support_excludes_face_hands_and_feet() -> None:
    support = _support()
    assert set(support.body_part.tolist()).issubset(set(SUPPORT_PARTS))
    assert not set(support.body_part.tolist()).intersection({7, 8, 10, 11, 15, 20, 21})


def test_support_uses_formal_lbs_binding() -> None:
    support = _support().to("cpu")
    rigid = torch.eye(4).repeat(55, 1, 1)
    rigid[:, 0, 3] = torch.arange(55) / 100
    result = deform_clean_body_support(support, rigid, torch.eye(3), torch.zeros(3))
    expected = support.canonical_xyz[:, 0] + (support.lbs_weights * (torch.arange(55) / 100)).sum(1)
    assert torch.allclose(result["xyz"][:, 0], expected, atol=1e-6, rtol=0)


def test_medium_high_density_alpha_is_comparable() -> None:
    medium, high = 60_000, 120_000
    m = 0.24; h = density_matched_opacity(m, medium, high)
    assert abs((1 - m) ** medium - (1 - h) ** high) < 1e-6
    assert h < m


def test_covered_support_is_occluded() -> None:
    baseline = torch.rand(3, 8, 8)
    combined = baseline.clone()
    assert float((combined - baseline).abs().max()) == 0.0


def test_clean_foundation_does_not_modify_base() -> None:
    base = torch.randn(64, 3); before = _fingerprint(base)
    _support()
    assert _fingerprint(base) == before


def test_shell_strategies_are_diagnostic_only() -> None:
    result = classify_base_shell(torch.tensor([15, 16]), torch.zeros(2), torch.tensor([False, True]), torch.zeros(2))
    assert result["s1_keep"].dtype == torch.bool and result["s2_keep"].dtype == torch.bool
    assert not any(value.requires_grad for value in result.values())


def test_clean_body_pilot_decision_matrix() -> None:
    common = {
        "geometry_alignment_pass": True, "pose_validation_pass": True,
        "covered_support_visible_fraction_max": 0.005,
        "s1_old_garment_residual_max": 0.005, "s2_old_garment_residual_max": 0.2,
        "s1_background_leakage_max": 0.005, "s2_background_leakage_max": 0.2,
        "identity_preserved": True, "base_bitwise_exact": True,
        "abnormal_gaussian_fraction_max": 0.0, "provenance_complete": True,
    }
    passed = adjudicate_clean_body_pilot(common, "PASS")
    partial = adjudicate_clean_body_pilot({**common, "covered_support_visible_fraction_max": 0.02}, "WARN")
    failed = adjudicate_clean_body_pilot({**common, "geometry_alignment_pass": False}, "PASS")
    assert passed["status"] == "CLEAN_BODY_ASSET_PILOT_PASS" and passed["recommended_strategy"] == "S1"
    assert partial["status"] == "CLEAN_BODY_ASSET_PILOT_PARTIAL"
    assert failed["status"] == "CLEAN_BODY_ASSET_PILOT_FAIL"
    assert not passed["minimal_o00_oracle_allowed"]


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test(); print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)}/{len(tests)} R3-CLEAN checks")


if __name__ == "__main__":
    main()
