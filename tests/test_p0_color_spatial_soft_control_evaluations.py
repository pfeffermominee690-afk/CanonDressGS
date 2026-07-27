from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np

from tools.paper import p0_evaluation_protocol as protocol


ROOT = Path(__file__).resolve().parents[1]
FROZEN_PROTOCOL = ROOT / "paper_protocol/reviewer_risk/p0_color_spatial_soft_control_protocol.yaml"
EVALUATOR = ROOT / "tools/paper/run_p0_color_spatial_soft_control_evaluations.py"


def test_frozen_protocol_fingerprint() -> None:
    digest = hashlib.sha256(FROZEN_PROTOCOL.read_bytes().replace(b"\r\n", b"\n"))
    assert digest.hexdigest() == "44320d575a7012a00396e6093dd5a0ee06cdd5679c878b3c7b254ceb215a0bba"


def test_grayscale_and_hue_contracts() -> None:
    image = np.array([[[1.0, 0.0, 0.0], [0.2, 0.4, 0.8]]], dtype=np.float32)
    gray = protocol.fixed_grayscale(image)
    assert gray.dtype == np.float32
    assert np.array_equal(gray[0, 0], np.array([0.299, 0.299, 0.299], dtype=np.float32))
    shifted = protocol.hue_shift(image, 120.0)
    assert np.allclose(shifted[0, 0], [0.0, 1.0, 0.0], atol=1e-7)


def test_c6_preserves_boundary_and_exterior() -> None:
    image = np.arange(36, dtype=np.float32).reshape(3, 4, 3) / 35.0
    mask = np.zeros((3, 4), dtype=np.bool_)
    mask[1:, 1:3] = True
    result = protocol.c6_average_garment_color(image, mask)
    assert np.array_equal(result[~mask], image[~mask])
    assert np.allclose(result[mask], image[mask].astype(np.float64).mean(0))


def test_ladder_endpoints_and_blur_kernel() -> None:
    rng = np.random.default_rng(4)
    image = rng.random((24, 20, 3), dtype=np.float32)
    assert np.array_equal(protocol.grayscale_ladder(image, 0.0), image)
    assert np.array_equal(protocol.grayscale_ladder(image, 1.0), protocol.fixed_grayscale(image))
    assert np.array_equal(protocol.blur_ladder(image, 0.0), image)
    assert protocol.blur_ladder_kernel_size(3.0) == 19


def test_disk_morphology_uses_integer_disk() -> None:
    mask = np.zeros((25, 25), dtype=np.bool_)
    mask[12, 12] = True
    dilated = protocol.disk_morphology(mask, 4)
    yy, xx = np.mgrid[:25, :25]
    assert np.array_equal(dilated, (xx - 12) ** 2 + (yy - 12) ** 2 <= 16)
    eroded = protocol.disk_morphology(dilated, -4)
    assert eroded[12, 12]
    assert int(eroded.sum()) == 1


def test_fixed_enumerations_and_endpoint_grid() -> None:
    assert len(protocol.PAIR_ORDER) == 10
    assert protocol.interpolation_grid_size() == 440
    assert protocol.mixed_reference_query_count_per_method() == 320
    assert len(protocol.mixed_reference_assignments()) == 8
    assert protocol.INTERPOLATION_ALPHAS[0] == 0.0
    assert protocol.INTERPOLATION_ALPHAS[-1] == 1.0


def test_evaluator_has_no_prohibited_execution_calls() -> None:
    tree = ast.parse(EVALUATOR.read_text(encoding="utf-8"))
    prohibited_attributes = {"backward", "step", "zero_grad"}
    prohibited_names = {"train_candidate", "train_model", "execute_run"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Attribute):
            assert function.attr not in prohibited_attributes
        elif isinstance(function, ast.Name):
            assert function.id not in prohibited_names


def test_evaluator_never_writes_checkpoint_suffixes() -> None:
    source = EVALUATOR.read_text(encoding="utf-8")
    assert ".pth.tmp" not in source
    assert "torch.save(" not in source
    assert "checkpoint_latest" not in source
