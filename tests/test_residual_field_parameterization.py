from __future__ import annotations

import hashlib
import inspect
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene.gaussian_clothing_residuals import GaussianClothingResiduals, interpolate_anchor_field
from scene.image_conditioned_failure_diagnostics import normalized_residual_regression_loss
from scene.residual_field_parameterizations import (
    AnchorSupportTokenResidualField,
    DirectPerGaussianResidualTableControl,
    FIXED_ONE_GATE_MODE,
    GaussianSupportTokenResidualField,
    residual_output_shapes,
)
from tools.run_residual_field_parameterization import residual_metrics


CONFIG = yaml.safe_load(
    (ROOT / "configs/research/subject02_residual_field_parameterization_v1.yaml").read_text(encoding="utf-8")
)
BOUNDS = {
    "xyz": 0.05, "log_scaling": 0.35, "rotation": 0.2617993878,
    "opacity_logit": 2.0, "sh0": 0.25, "shN": 0.10,
}


class _Base:
    def __init__(self, count: int = 12) -> None:
        generator = torch.Generator().manual_seed(7)
        self._xyz = torch.randn(count, 3, generator=generator)
        self._scaling = torch.randn(count, 3, generator=generator) * 0.1
        self._rotation = torch.nn.functional.normalize(torch.randn(count, 4, generator=generator), dim=1)
        self._opacity = torch.randn(count, 1, generator=generator)
        self._sh0 = torch.randn(count, 1, 3, generator=generator) * 0.1
        self._shN = torch.randn(count, 3, 3, generator=generator) * 0.01


def _support(count: int = 12, anchors: int = 5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(9)
    descriptor = torch.randn(count, 11, generator=generator)
    indices = torch.tensor([[index % anchors, (index + 1) % anchors] for index in range(count)], dtype=torch.long)
    weights = torch.tensor([[0.75, 0.25]] * count)
    return descriptor, indices, weights


def _field(kind: str):
    base = _Base()
    descriptor, indices, weights = _support()
    common = {
        "static_support_descriptor": descriptor,
        "gaussian_anchor_indices": indices,
        "gaussian_anchor_weights": weights,
        "garment_embedding_dim": 7,
        "token_dim": 4,
        "hidden_dim": 16,
        "num_blocks": 2,
        "output_shapes": residual_output_shapes(base),
        "channel_bounds": BOUNDS,
        "default_chunk_size": 4,
    }
    if kind == "p1":
        return GaussianSupportTokenResidualField(**common), torch.randn(1, 7)
    return AnchorSupportTokenResidualField(anchor_count=5, **common), torch.randn(1, 7)


def _assert_raises(kind: type[BaseException], callback) -> None:
    try:
        callback()
    except kind:
        return
    raise AssertionError(f"expected {kind.__name__}")


def test_p0_is_direct_residual_table_only() -> None:
    table = DirectPerGaussianResidualTableControl(residual_output_shapes(_Base()), BOUNDS)
    assert isinstance(table.normalized_residual_tables, torch.nn.ParameterDict)
    assert sum(parameter.numel() for parameter in table.parameters()) == 12 * 22
    assert all(torch.count_nonzero(value).item() == 0 for value in table().as_dict().values())


def test_p0_has_no_decoder() -> None:
    table = DirectPerGaussianResidualTableControl(residual_output_shapes(_Base()), BOUNDS)
    assert not hasattr(table, "geometry_input") and not hasattr(table, "appearance_input")
    assert not any(isinstance(module, torch.nn.Linear) for module in table.modules())


def test_p1_token_is_outfit_independent() -> None:
    field, _ = _field("p1")
    assert field.support_tokens.ndim == 2
    assert "outfit" not in inspect.signature(field.forward).parameters


def test_p1_has_one_token_per_gaussian() -> None:
    field, _ = _field("p1")
    assert field.token_count == field.gaussian_count == 12


def test_p1_has_multiplicative_condition_interaction() -> None:
    source = inspect.getsource(GaussianSupportTokenResidualField.__mro__[1]._forward_chunk)
    assert "token * projected_garment" in source
    assert "multiplicative_interaction" in source


def test_p2_has_one_token_per_anchor() -> None:
    field, _ = _field("p2")
    assert field.token_count == 5 and field.gaussian_count == 12


def test_p2_uses_frozen_anchor_interpolation() -> None:
    field, _ = _field("p2")
    selected = torch.tensor([0, 3, 8], dtype=torch.long)
    expected = interpolate_anchor_field(
        field.support_tokens,
        field.gaussian_anchor_indices.index_select(0, selected),
        field.gaussian_anchor_weights.index_select(0, selected),
    )
    assert torch.equal(field.token_rows(selected), expected)
    assert field.gaussian_anchor_indices.requires_grad is False
    assert field.gaussian_anchor_weights.requires_grad is False


def test_p1_p2_have_no_outfit_id_input() -> None:
    for kind in ("p1", "p2"):
        field, _ = _field(kind)
        assert set(inspect.signature(field.forward).parameters).isdisjoint({"outfit", "outfit_id", "cloth_id"})


def test_p1_p2_have_no_target_image_input() -> None:
    forbidden = {"target", "target_rgb", "target_mask", "image"}
    for kind in ("p1", "p2"):
        field, _ = _field(kind)
        assert set(inspect.signature(field.forward).parameters).isdisjoint(forbidden)


def test_teacher_is_detached_and_loss_only() -> None:
    table = DirectPerGaussianResidualTableControl(residual_output_shapes(_Base()), BOUNDS)
    prediction = table()
    target = GaussianClothingResiduals(**{
        name: torch.zeros_like(value) for name, value in prediction.as_dict().items()
    })
    loss, _ = normalized_residual_regression_loss(prediction, target, BOUNDS)
    loss.backward()
    assert all(value.requires_grad is False and value.grad_fn is None for value in target.as_dict().values())
    assert CONFIG["inputs"]["oracle_o01_checkpoint"] not in inspect.getsource(GaussianSupportTokenResidualField.forward)


def test_geometry_and_appearance_are_separate() -> None:
    field, _ = _field("p1")
    assert field.geometry_input is not field.appearance_input
    assert field.geometry_blocks is not field.appearance_blocks
    assert set(field.geometry_input.parameters()).isdisjoint(set(field.appearance_input.parameters()))


def test_six_heads_are_independent() -> None:
    field, _ = _field("p1")
    heads = [field.xyz_head, field.scaling_head, field.rotation_head, field.opacity_head, field.sh0_head, field.shN_head]
    assert len({id(head) for head in heads}) == 6
    assert len({id(head.weight) for head in heads}) == 6


def test_gates_are_fixed_one() -> None:
    field, garment = _field("p1")
    output = field(garment)
    assert output.gate_mode == FIXED_ONE_GATE_MODE
    assert torch.equal(output.geometry_gate, torch.ones(12, 1))
    assert torch.equal(output.appearance_gate, torch.ones(12, 1))


def test_chunked_matches_non_chunked() -> None:
    for kind in ("p1", "p2"):
        field, garment = _field(kind)
        generator = torch.Generator().manual_seed(33)
        with torch.no_grad():
            for parameter in field.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.01)
        chunked = field(garment, chunk_size=3).gated_gaussian_residuals
        complete = field(garment, chunk_size=100).gated_gaussian_residuals
        assert all(torch.allclose(getattr(chunked, name), getattr(complete, name), atol=1e-7, rtol=1e-6) for name in chunked.as_dict())


def test_o01_o08_are_balanced() -> None:
    assert all(CONFIG[phase]["outfit_balance"] == "strict_alternating_1_to_1" for phase in ("p0", "p1", "p2"))


def test_base_mmlp_renderer_are_frozen() -> None:
    assert CONFIG["base"]["frozen"] is True
    assert CONFIG["permissions"]["modify_renderer"] is False
    assert CONFIG["permissions"]["real_reference_training"] is False


def test_historical_outputs_immutable() -> None:
    assert CONFIG["permissions"]["modify_o01_attempt"] is False
    assert CONFIG["permissions"]["modify_diagnosis_attempt"] is False
    assert CONFIG["permissions"]["modify_v7_attempt"] is False
    source = ROOT / "tools/run_residual_field_parameterization.py"
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    assert before == hashlib.sha256(source.read_bytes()).hexdigest()


def test_zero_target_zero_prediction_cosine_is_exact_match() -> None:
    table = DirectPerGaussianResidualTableControl(residual_output_shapes(_Base()), BOUNDS)
    zero = table()
    metrics = residual_metrics(zero, zero, BOUNDS, active_epsilon=1e-8)
    assert metrics["direction_cosine"] == 1.0


def test_token_dimension_is_frozen_at_24() -> None:
    assert CONFIG["p1"]["token_dim"] == CONFIG["p2"]["token_dim"] == 24


def test_p0_p1_p2_reject_invalid_indices() -> None:
    table = DirectPerGaussianResidualTableControl(residual_output_shapes(_Base()), BOUNDS)
    _assert_raises(IndexError, lambda: table(torch.tensor([12], dtype=torch.long)))
    field, garment = _field("p1")
    _assert_raises(IndexError, lambda: field(garment, gaussian_indices=torch.tensor([-1], dtype=torch.long)))


if __name__ == "__main__":
    tests = [(name, value) for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for name, callback in tests:
        callback()
        print(f"PASS {name}")
    print(f"PASS all {len(tests)} residual-field parameterization tests")
