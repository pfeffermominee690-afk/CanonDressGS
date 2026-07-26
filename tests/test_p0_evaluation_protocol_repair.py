from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from tools.paper import p0_evaluation_protocol as protocol


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "paper_protocol/reviewer_risk/p0_color_spatial_soft_control_protocol.yaml"
CORRECTION_PATH = ROOT / "paper_protocol/reviewer_risk/p0_evaluation_protocol_correction.json"


@pytest.fixture(scope="module")
def contract() -> dict:
    return yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def correction() -> dict:
    return json.loads(CORRECTION_PATH.read_text(encoding="utf-8"))


def _rgb_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = np.linspace(0.0, 1.0, 16 * 3, dtype=np.float32).reshape(4, 4, 3)
    donor = np.flip(source, axis=0).copy()
    mask = np.zeros((4, 4), dtype=np.bool_)
    mask[1:3, 1:3] = True
    return source, donor, mask


def test_c3_uses_256_linear_quantiles(contract: dict) -> None:
    c3 = contract["color_transforms"]["C3"]
    assert c3["parameter_values"]["quantile_count"] == 256
    assert c3["parameter_values"]["quantile_method"] == "linear"
    assert protocol.C3_QUANTILES.dtype == np.float64
    assert len(protocol.C3_QUANTILES) == 256
    assert protocol.C3_QUANTILES[0] == 0.0
    assert protocol.C3_QUANTILES[-1] == 1.0


def test_c3_uses_half_up_8bit_rounding() -> None:
    values = np.array([0.5 / 255.0, 1.5 / 255.0, 254.5 / 255.0], dtype=np.float64)
    actual = protocol.round_half_up_8bit(values)
    assert np.array_equal(actual, np.array([1, 2, 255], dtype=np.float32) / 255.0)
    assert actual[0] != np.float32(np.round(0.5) / 255.0)


def test_c3_changes_masked_rgb_only() -> None:
    source, donor, mask = _rgb_fixture()
    before = source.copy()
    mask_before = mask.copy()
    output = protocol.c3_histogram_match(source, donor, mask)
    assert output.dtype == np.float32
    assert np.array_equal(output[~mask], before[~mask])
    assert np.array_equal(source, before)
    assert np.array_equal(mask, mask_before)


def test_c4_uses_epsilon_1e_6(contract: dict) -> None:
    assert protocol.C4_EPSILON == 1.0e-6
    assert contract["color_transforms"]["C4"]["parameter_values"]["epsilon"] == 1.0e-6


def test_c4_handles_degenerate_std() -> None:
    source = np.zeros((3, 3, 3), dtype=np.float32)
    source[1:, 1:] = np.array([0.25, 0.5, 0.75], dtype=np.float32)
    mask = np.zeros((3, 3), dtype=np.bool_)
    mask[1:, 1:] = True
    target_mean = np.array([0.2, 0.4, 0.6], dtype=np.float64)
    output = protocol.c4_brightness_contrast_normalize(
        source, mask, target_mean, [0.1, 0.1, 0.1]
    )
    assert np.allclose(output[mask], target_mean.astype(np.float32), atol=0.0, rtol=0.0)
    assert np.array_equal(output[~mask], source[~mask])


def test_c5_uses_kernel_11_sigma_3(contract: dict) -> None:
    c5 = contract["color_transforms"]["C5"]
    assert c5["parameter_values"]["kernel_size"] == protocol.C5_KERNEL_SIZE == 11
    assert c5["parameter_values"]["sigma_pixels"] == protocol.C5_SIGMA == 3.0
    kernel = protocol.gaussian_kernel_1d(11, 3.0)
    assert kernel.dtype == np.float32
    assert float(kernel.sum(dtype=np.float64)) == pytest.approx(1.0, abs=1e-7)
    assert np.array_equal(kernel, kernel[::-1])


def test_c5_blurs_after_resize_before_normalization(contract: dict) -> None:
    c5 = contract["color_transforms"]["C5"]
    assert c5["resize_position"] == (
        "transform is after resize at frozen F2 input resolution and before normalization"
    )
    assert c5["operation_order"].split(" -> ") == [
        "resize",
        "separable vertical/horizontal blur",
        "clip",
        "restore exterior",
        "F2 normalization",
    ]


def test_c5_preserves_mask_bitwise() -> None:
    rgb = np.zeros((15, 15, 3), dtype=np.float32)
    rgb[7, 7] = 1.0
    mask = np.zeros((15, 15), dtype=np.bool_)
    mask[3:12, 3:12] = True
    before = mask.tobytes()
    output = protocol.c5_gaussian_blur(rgb, mask)
    assert mask.tobytes() == before
    assert np.array_equal(output[~mask], rgb[~mask])
    assert output[7, 7, 0] < 1.0


def test_mixed_reference_enumerates_all_assignments() -> None:
    assignments = protocol.mixed_reference_assignments()
    assert len(assignments) == 8
    values = [value for _, value in assignments]
    assert values.count(("A", "A", "A")) == 1
    assert values.count(("B", "B", "B")) == 1
    assert {value.index("B") for value in values if value.count("B") == 1} == {0, 1, 2}
    assert {value.index("A") for value in values if value.count("A") == 1} == {0, 1, 2}


def test_mixed_reference_query_count_is_320_per_method(contract: dict) -> None:
    assert protocol.mixed_reference_query_count_per_method() == 320
    assert contract["mixed_reference"]["query_sets_per_method"] == 320
    assert contract["mixed_reference"]["total_method_queries"] == 960


def test_grayscale_ladder_is_fixed(contract: dict) -> None:
    assert protocol.GRAYSCALE_LADDER == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert tuple(contract["perturbation_ladders"]["grayscale"]["values"]) == protocol.GRAYSCALE_LADDER


def test_hue_ladder_is_fixed(contract: dict) -> None:
    assert protocol.HUE_LADDER_DEGREES == (0, 15, 30, 45, 60)
    assert tuple(contract["perturbation_ladders"]["hue"]["values_degrees"]) == protocol.HUE_LADDER_DEGREES


def test_blur_ladder_is_fixed(contract: dict) -> None:
    assert protocol.BLUR_LADDER_SIGMA == (0, 1, 2, 3, 4)
    assert [protocol.blur_ladder_kernel_size(x) for x in protocol.BLUR_LADDER_SIGMA] == [1, 7, 13, 19, 25]
    endpoint = contract["perturbation_ladders"]["blur"]["c5_endpoint"]
    assert endpoint == (
        "sigma=3 intensity; C5 keeps its independently fixed kernel_size=11 while the "
        "ladder formula yields 19"
    )


def test_mask_morphology_ladder_is_fixed(contract: dict) -> None:
    assert protocol.MASK_MORPHOLOGY_RADIUS == (-8, -4, 0, 4, 8)
    assert tuple(contract["perturbation_ladders"]["mask_morphology"]["values_radius_pixels"]) == protocol.MASK_MORPHOLOGY_RADIUS


def test_interpolation_uses_raw_coefficients(contract: dict) -> None:
    interpolation = contract["direct_interpolation"]
    assert interpolation["coefficient_space"] == "raw frozen teacher coefficient space; not standardized"
    assert interpolation["coefficient_source"] == "selected_basis.pt::teacher_coefficients"
    assert interpolation["reference_predictor_used"] is False


def test_interpolation_grid_is_440(contract: dict) -> None:
    assert protocol.interpolation_grid_size() == 440
    assert contract["direct_interpolation"]["grid_size"] == 440
    assert len(protocol.PAIR_ORDER) == 10
    assert len(protocol.INTERPOLATION_ALPHAS) == 11


def test_spatial_thresholds_are_contract_derived(contract: dict) -> None:
    half, one = protocol.trust_radii(0.05)
    assert (half, one) == (0.025, 0.05)
    spatial = contract["spatial_metrics"]
    assert spatial["trust_radius"]["b_xyz"] == 0.05
    assert spatial["normal_contract"]["audited_exact_min_tie_count"] == 0
    assert spatial["garment_mapping"]["all_five_seen_teachers_bitwise_equal"] is True


def test_boundary_tolerance_is_fixed(contract: dict) -> None:
    assert protocol.boundary_tolerance(1536, 1024) == 5
    assert protocol.boundary_tolerance(100, 100) == 1
    formula = contract["extended_metrics"]["boundary_fscore"]["tolerance_formula"]
    assert formula == "max(1,floor(0.005*min(H,W)+0.5))"


def test_no_evaluation_is_run(correction: dict) -> None:
    assert correction["original_ambiguity"]["evaluation_started"] is False
    assert correction["execution_guards"]["evaluation"] == 0


def test_no_render_or_metric_is_created(correction: dict) -> None:
    guards = correction["execution_guards"]
    assert guards["render_write"] == guards["metric_write"] == 0
    forbidden = [
        ROOT / "paper_protocol/reviewer_risk/p0_color_counterfactual_results.json",
        ROOT / "paper_protocol/reviewer_risk/p0_extended_metric_results.json",
        ROOT / "paper_protocol/reviewer_risk/p0_spatial_artifact_results.json",
        ROOT / "paper_protocol/reviewer_risk/p0_basis_interpolation_results.json",
        ROOT / "paper_protocol/reviewer_risk/p0_mixed_reference_results.json",
        ROOT / "paper_protocol/reviewer_risk/p0_perturbation_continuity_results.json",
    ]
    assert not any(path.exists() for path in forbidden)


def test_no_training_or_checkpoint_is_created(correction: dict) -> None:
    guards = correction["execution_guards"]
    assert guards["training"] == 0
    assert guards["backward"] == 0
    assert guards["optimizer"] == 0
    assert guards["checkpoint_write"] == 0
    changed = subprocess.run(
        ["git", "diff", "--name-only", protocol.SOURCE_HEAD, "--"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert not any(Path(path).suffix.lower() in {".pt", ".pth", ".ckpt"} for path in changed)


def test_formal_assets_are_immutable(correction: dict) -> None:
    assert correction["execution_guards"]["formal_asset_change"] == 0
    changed = set(
        subprocess.run(
            ["git", "diff", "--name-only", protocol.SOURCE_HEAD, "--"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    )
    protected = {
        "paper_protocol/frozen_asset_manifest.json",
        "paper_protocol/experiment_registry.yaml",
        "paper_protocol/reviewer_risk/p0_formal_candidate_run_registry.yaml",
        "paper_protocol/reviewer_risk/p0_formal_candidate_final_summary.json",
        "configs/paper/aaai27_seen_outfit_explicit_basis_v1.yaml",
    }
    assert changed.isdisjoint(protected)
    expected = correction["final_protocol_fingerprint"]["sha256"]
    normalized = PROTOCOL_PATH.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(normalized).hexdigest() == expected


def test_no_paper_final_is_created(contract: dict, correction: dict) -> None:
    assert contract["paper_final"] is False
    assert contract["no_evaluation_gate"]["paper_final_count"] == 0
    assert correction["execution_guards"]["paper_final_count"] == 0
    assert correction["final_paper_method_adjudication_written"] is False
