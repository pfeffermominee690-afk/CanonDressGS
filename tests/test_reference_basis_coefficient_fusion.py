from __future__ import annotations

import inspect
from pathlib import Path

import torch
from torch import nn
import yaml

from scene.reference_basis_coefficient_fusion import (
    MaskAwareReferenceTokenEncoderV1,
    ReferenceSetCoefficientFusionV1,
    build_reference_coefficient_fusion,
)
from tools import run_reference_basis_coefficient_fusion as runner


ROOT = Path(__file__).resolve().parents[1]


def _inputs(count: int = 3, feature_dim: int = 4):
    torch.manual_seed(7)
    images = torch.rand(count, 3, 16, 12)
    foreground = torch.zeros(count, 1, 16, 12)
    foreground[:, :, 1:15, 1:11] = 1
    clothing = torch.zeros_like(foreground)
    clothing[:, :, 3:13, 2:10] = 1
    poses = torch.zeros(count, 165)
    w2c = torch.eye(4).repeat(count, 1, 1)
    w2c[:, 0, 0] = torch.tensor([1.0, 0.0, -1.0])[:count]
    w2c[:, 0, 2] = torch.tensor([0.0, 1.0, 0.0])[:count]
    w2c[:, 2, 0] = torch.tensor([0.0, -1.0, 0.0])[:count]
    w2c[:, 2, 2] = torch.tensor([1.0, 0.0, -1.0])[:count]
    backbone = nn.Sequential(nn.Conv2d(3, feature_dim, 3, padding=1), nn.SiLU())
    encoder = MaskAwareReferenceTokenEncoderV1(backbone, feature_dim, token_dim=8, hidden_dim=16)
    return encoder, images, clothing, foreground, poses, w2c


def _encode(count: int = 3):
    encoder, images, clothing, foreground, poses, w2c = _inputs(count)
    output = encoder(images, clothing, foreground, poses, w2c)
    return encoder, output, (images, clothing, foreground, poses, w2c)


def test_mask_aware_token_contains_required_statistics() -> None:
    _, output, _ = _encode()
    assert output.tokens.shape == (3, 8)
    assert output.cloth_mean.shape == output.cloth_max.shape == (3, 4)
    assert output.foreground_mean.shape == output.cloth_foreground_difference.shape == (3, 4)
    assert output.mask_area.shape == output.bbox_aspect.shape == (3, 1)
    assert output.centroid.shape == (3, 2)
    assert output.view_direction.shape == (3, 3)
    assert torch.allclose(output.cloth_foreground_difference, output.cloth_mean - output.foreground_mean)


def test_mask_aware_pooling_rejects_empty_and_full_semantic_errors() -> None:
    encoder, images, clothing, foreground, poses, w2c = _inputs()
    bad = torch.ones_like(clothing)
    try:
        encoder(images, bad, foreground, poses, w2c)
    except ValueError as error:
        assert "subset" in str(error)
    else:
        raise AssertionError("non-subset clothing mask was accepted")


def test_different_outfit_images_change_tokens() -> None:
    encoder, output, inputs = _encode()
    images, clothing, foreground, poses, w2c = inputs
    changed = encoder(1 - images, clothing, foreground, poses, w2c)
    assert not torch.equal(output.raw_tokens, changed.raw_tokens)
    assert not torch.equal(output.tokens, changed.tokens)


def test_same_outfit_permutation_is_exact_for_coefficient() -> None:
    _, output, _ = _encode()
    fusion = ReferenceSetCoefficientFusionV1(8, 12).eval()
    first = fusion(output.tokens).coefficient
    order = torch.tensor([2, 0, 1])
    second = fusion(output.tokens.index_select(0, order)).coefficient
    assert float((first - second).abs().max()) <= 1.0e-5


def test_permutation_preserves_token_set() -> None:
    encoder, output, inputs = _encode()
    images, clothing, foreground, poses, w2c = inputs
    order = torch.tensor([2, 0, 1])
    permuted = encoder(
        images[order], clothing[order], foreground[order], poses[order], w2c[order]
    )
    assert torch.allclose(output.tokens[order], permuted.tokens, atol=1.0e-6, rtol=1.0e-6)
    assert torch.allclose(output.raw_tokens[order], permuted.raw_tokens, atol=1.0e-6, rtol=1.0e-6)


def test_k1_k2_k3_are_supported() -> None:
    fusion = ReferenceSetCoefficientFusionV1(8, 12)
    for count in (1, 2, 3):
        _, output, _ = _encode(count)
        value = fusion(output.tokens)
        assert value.coefficient.shape == (1,)
        assert -1 <= float(value.coefficient) <= 1


def test_padding_does_not_leak() -> None:
    _, output, _ = _encode(3)
    fusion = ReferenceSetCoefficientFusionV1(8, 12).eval()
    valid = torch.tensor([1.0, 1.0, 0.0])
    first = fusion(output.tokens, valid).coefficient
    changed = output.tokens.clone()
    changed[2] = 1e6
    second = fusion(changed, valid).coefficient
    assert torch.equal(first, second)


def test_zero_and_base_substitution_change_raw_tokens() -> None:
    encoder, output, inputs = _encode()
    images, clothing, foreground, poses, w2c = inputs
    zero = encoder(torch.zeros_like(images), clothing, foreground, poses, w2c)
    base = encoder(images.mean(0, keepdim=True).expand_as(images), clothing, foreground, poses, w2c)
    assert not torch.equal(output.raw_tokens, zero.raw_tokens)
    assert not torch.equal(output.raw_tokens, base.raw_tokens)


def test_coefficient_is_bounded_and_finite() -> None:
    fusion = ReferenceSetCoefficientFusionV1(8, 12)
    value = fusion(torch.randn(3, 8)).coefficient
    assert torch.isfinite(value).all()
    assert torch.all(value >= -1) and torch.all(value <= 1)


def test_coefficient_backward_reaches_token_adapter_and_fusion() -> None:
    encoder, output, _ = _encode()
    fusion = ReferenceSetCoefficientFusionV1(8, 12)
    fusion(output.tokens).coefficient.sum().backward()
    adapter_grads = [p.grad for p in encoder.token_adapter.parameters()]
    fusion_grads = [p.grad for p in fusion.parameters()]
    assert all(value is not None and torch.isfinite(value).all() for value in adapter_grads)
    assert all(value is not None and torch.isfinite(value).all() for value in fusion_grads)
    assert any(float(value.abs().sum()) > 0 for value in adapter_grads)
    assert any(float(value.abs().sum()) > 0 for value in fusion_grads)


def test_spatial_backbone_is_strictly_frozen() -> None:
    encoder, output, _ = _encode()
    fusion = ReferenceSetCoefficientFusionV1(8, 12)
    fusion(output.tokens).coefficient.sum().backward()
    assert all(not parameter.requires_grad for parameter in encoder.spatial_backbone.parameters())
    assert all(parameter.grad is None for parameter in encoder.spatial_backbone.parameters())


def test_nan_inf_and_shape_mismatch_are_rejected() -> None:
    encoder, images, clothing, foreground, poses, w2c = _inputs()
    images[0, 0, 0, 0] = float("nan")
    for bad in (images,):
        try:
            encoder(bad, clothing, foreground, poses, w2c)
        except ValueError:
            pass
        else:
            raise AssertionError("NaN reference was accepted")
    fusion = ReferenceSetCoefficientFusionV1(8, 12)
    for bad in (torch.randn(1, 1, 8), torch.full((2, 8), float("inf"))):
        try:
            fusion(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid reference token was accepted")


def test_forward_signature_has_no_forbidden_prediction_fields() -> None:
    parameters = set(inspect.signature(MaskAwareReferenceTokenEncoderV1.forward).parameters)
    forbidden = {"target_rgb", "target_mask", "target_pose", "target_camera", "outfit_id", "cloth_id", "teacher_coefficient"}
    assert not parameters.intersection(forbidden)


def test_factory_preserves_legacy_configuration_path() -> None:
    backbone = nn.Conv2d(3, 4, 1)
    assert build_reference_coefficient_fusion(
        {"coefficient_fusion": {"type": "legacy"}}, spatial_backbone=backbone, feature_dim=4
    ) is None


def test_preregistered_contract_freezes_basis_base_and_renderer() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/research/subject02_reference_basis_coefficient_fusion_v1.yaml").read_text(encoding="utf-8")
    )
    assert config["coefficient_fusion"]["type"] == "mask_aware_reference_set_v1"
    assert config["training"]["max_steps"] == 500
    assert config["training"]["paired_outfits_per_step"] == ["O01", "O08"]
    assert config["permissions"]["modify_basis"] is False
    assert config["permissions"]["modify_base"] is False
    assert config["permissions"]["modify_renderer"] is False
    assert config["permissions"]["target_image_in_prediction_forward"] is False


def test_loss_contract_is_exact() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/research/subject02_reference_basis_coefficient_fusion_v1.yaml").read_text(encoding="utf-8")
    )
    assert config["training"]["loss"] == {
        "coefficient_weight": 1.0,
        "sign_weight": 0.25,
        "pair_weight": 0.10,
        "sign_margin": 0.8,
        "pair_margin": 1.5,
    }


def test_acceptance_thresholds_are_preregistered() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/research/subject02_reference_basis_coefficient_fusion_v1.yaml").read_text(encoding="utf-8")
    )
    assert config["acceptance"] == {
        "coefficient_mae_max": 0.10,
        "o01_mean_max": -0.80,
        "o08_mean_min": 0.80,
        "separation_min": 1.60,
        "correct_sign_count_min": 8,
        "swapped_coefficient_margin_min": 1.20,
        "correct_episode_wins_min": 8,
        "reference_replacement_change_min": 0.20,
        "permutation_max_abs_diff_max": 1.0e-5,
    }


def _config():
    return yaml.safe_load(
        (ROOT / "configs/research/subject02_reference_basis_coefficient_fusion_v1.yaml").read_text(encoding="utf-8")
    )


def test_reference_files_are_distinct() -> None:
    source = inspect.getsource(runner.audit_inputs)
    assert "different_outfit_rgb_hashes" in source
    assert "each_episode_uses_distinct_reference_rgb" in source


def test_reference_masks_are_not_empty_or_full() -> None:
    source = inspect.getsource(runner.audit_inputs)
    assert "value <= 0 or value >= 1" in source


def test_mask_pooling_uses_reference_mask_only() -> None:
    parameters = set(inspect.signature(MaskAwareReferenceTokenEncoderV1.forward).parameters)
    assert "reference_clothing_masks" in parameters
    assert not {"target_mask", "target_clothing_mask"}.intersection(parameters)


def test_reference_token_has_no_target_input() -> None:
    parameters = set(inspect.signature(MaskAwareReferenceTokenEncoderV1.forward).parameters)
    assert not {"target_rgb", "target_mask", "target_image_feature"}.intersection(parameters)


def test_coefficient_branch_has_no_target_pose_input() -> None:
    parameters = set(inspect.signature(runner.coefficient_forward).parameters)
    assert "target_pose" not in parameters and "target_camera" not in parameters


def test_reference_set_is_permutation_invariant() -> None:
    _, output, _ = _encode()
    fusion = ReferenceSetCoefficientFusionV1(8, 12).eval()
    difference = (fusion(output.tokens).coefficient - fusion(output.tokens[[2, 0, 1]]).coefficient).abs()
    assert float(difference.max()) <= 1.0e-5


def test_reference_set_supports_one_two_three_refs() -> None:
    fusion = ReferenceSetCoefficientFusionV1(8, 12)
    assert all(fusion(torch.randn(count, 8)).coefficient.shape == (1,) for count in (1, 2, 3))


def test_paired_batch_contains_o01_and_o08() -> None:
    source = inspect.getsource(runner.run_training)
    assert 'f"O01/{condition}"' in source and 'f"O08/{condition}"' in source
    assert _config()["training"]["paired_outfits_per_step"] == ["O01", "O08"]


def test_coefficient_output_is_scalar() -> None:
    assert ReferenceSetCoefficientFusionV1(8, 12)(torch.randn(3, 8)).coefficient.shape == (1,)


def test_coefficient_output_is_bounded() -> None:
    value = ReferenceSetCoefficientFusionV1(8, 12)(torch.randn(3, 8)).coefficient
    assert bool(torch.all(value >= -1) and torch.all(value <= 1))


def test_basis_is_frozen() -> None:
    assert _config()["basis"]["frozen"] is True
    assert _config()["permissions"]["modify_basis"] is False


def test_no_per_gaussian_parameter_is_trained() -> None:
    assert _config()["permissions"]["per_gaussian_trainable_parameters"] is False
    source = inspect.getsource(runner._trainable_groups)
    assert "basis" not in source and "gaussian" not in source


def test_teacher_coefficient_is_loss_only() -> None:
    assert "TEACHER" not in inspect.getsource(runner.coefficient_forward)
    assert "targets" in inspect.getsource(runner._loss_pair)


def test_reference_swap_keeps_target_pose_fixed() -> None:
    source = inspect.getsource(runner._variant_evaluation)
    assert '"target_pose_camera_fixed": True' in source
    assert "sample = context" in source


def test_zero_rgb_changes_reference_only() -> None:
    source = inspect.getsource(runner.reference_variants)
    assert 'zero["reference_images"] = torch.zeros_like' in source
    assert "target" not in source.split('zero["reference_images"]', 1)[1].split("base =", 1)[0]


def test_base_rgb_replacement_changes_reference_only() -> None:
    source = inspect.getsource(runner.reference_variants)
    assert 'base["reference_images"] = _base_reference_images' in source


def test_base_mmlp_renderer_are_frozen() -> None:
    permissions = _config()["permissions"]
    assert permissions["modify_base"] is False
    assert permissions["modify_mmlp"] is False
    assert permissions["modify_renderer"] is False


def test_historical_outputs_immutable() -> None:
    permissions = _config()["permissions"]
    assert permissions["modify_explicit_basis_attempt"] is False
    assert "immutable_tree_metadata_fingerprint" in inspect.getsource(runner.finalize)
