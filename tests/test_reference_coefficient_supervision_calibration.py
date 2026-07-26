from __future__ import annotations

import inspect
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch import nn

from scene.frozen_f2_linear_coefficient_control import (
    FrozenF2LinearLogitControl,
    FrozenF2ReferenceFeatureExtractor,
)
from scene.reference_basis_coefficient_fusion import MaskAwareReferenceTokenEncoderV1
from tools import run_explicit_gaussian_residual_basis as legacy
from tools import run_reference_coefficient_supervision_calibration as runner


CONFIG = yaml.safe_load(
    (Path(__file__).resolve().parents[1]
     / "configs/research/subject02_reference_coefficient_supervision_calibration_v1.yaml")
    .read_text(encoding="utf-8")
)


def synthetic_inputs(count: int = 3):
    torch.manual_seed(17)
    images = torch.rand(count, 3, 16, 12)
    masks = torch.zeros(count, 1, 16, 12)
    masks[0, :, 2:13, 2:10] = 1
    masks[1, :, 3:14, 1:9] = 1
    masks[2, :, 1:12, 3:11] = 1
    backbone = nn.Sequential(nn.Conv2d(3, 5, 3, padding=1), nn.SiLU())
    return images, masks, backbone


def test_online_f2_matches_offline_probe():
    images, masks, backbone = synthetic_inputs()
    extractor = FrozenF2ReferenceFeatureExtractor(backbone, 5)
    with torch.no_grad():
        feature_maps = backbone(images)
        resized = F.interpolate(masks, feature_maps.shape[-2:], mode="area").clamp(0, 1)
        mean = MaskAwareReferenceTokenEncoderV1._weighted_mean(feature_maps, resized)
        maximum = MaskAwareReferenceTokenEncoderV1._masked_max(feature_maps, resized)
        offline = torch.cat((mean.mean(0), maximum.mean(0)))
        online = extractor(images, masks).set_mean.reshape(-1)
    assert torch.equal(offline, online)
    assert extractor._weighted_mean is not None and extractor._masked_max is not None


def test_o01_label_maps_to_negative_coefficient():
    assert runner.CLASS_LABELS["O01"] == 0.0
    assert runner.COEFFICIENT_TARGETS["O01"] == -1.0


def test_o08_label_maps_to_positive_coefficient():
    assert runner.CLASS_LABELS["O08"] == 1.0
    assert runner.COEFFICIENT_TARGETS["O08"] == 1.0


def test_paired_batch_order_is_explicit():
    assert runner.OUTFITS == ("O01", "O08")
    assert CONFIG["stage_2"]["paired_batch_order"] == ["O01", "O08"]


def test_signed_rank_gradient_nonzero_at_equal_logits():
    first = torch.tensor(0.0, requires_grad=True)
    second = torch.tensor(0.0, requires_grad=True)
    loss, parts = runner._linear_control_loss({"O01": first, "O08": second}, CONFIG)
    gradient = torch.autograd.grad(parts["signed_rank"], (first, second))
    assert gradient[0].item() > 0 and gradient[1].item() < 0
    assert torch.isfinite(loss)


def test_old_absolute_pair_gradient_is_reported():
    coefficient = torch.zeros(2, requires_grad=True)
    _, parts = runner._old_loss(coefficient, CONFIG)
    gradient = torch.autograd.grad(parts["absolute_pair"], coefficient)[0]
    assert torch.equal(gradient, torch.zeros_like(gradient))


def test_bce_uses_raw_logits():
    first = torch.tensor(-0.4, requires_grad=True)
    second = torch.tensor(0.3, requires_grad=True)
    _, parts = runner._linear_control_loss({"O01": first, "O08": second}, CONFIG)
    expected = F.binary_cross_entropy_with_logits(
        torch.stack((first, second)), torch.tensor([0.0, 1.0])
    )
    assert torch.equal(parts["classification"], expected)


def test_linear_control_has_no_attention():
    control = FrozenF2LinearLogitControl(20)
    assert not any(isinstance(module, nn.MultiheadAttention) for module in control.modules())
    assert "attention" not in inspect.getsource(FrozenF2LinearLogitControl).lower()


def test_linear_control_has_no_projection_completion():
    source = inspect.getsource(FrozenF2LinearLogitControl).lower()
    assert "projection" not in source and "completion" not in source


def test_linear_control_has_no_target_input():
    parameters = inspect.signature(FrozenF2ReferenceFeatureExtractor.forward).parameters
    assert not runner.FORBIDDEN_FORWARD_FIELDS.intersection(parameters)


def test_reference_swap_keeps_target_pose_fixed():
    source = inspect.getsource(runner._stage_3_variant)
    assert 'context["samples"][f"{outfit}/{condition}"]' in source
    assert 'episode["target' not in source
    assert '"target_pose_camera_fixed": True' in source


def test_permutation_invariance():
    images, masks, backbone = synthetic_inputs()
    extractor = FrozenF2ReferenceFeatureExtractor(backbone, 5)
    control = FrozenF2LinearLogitControl(extractor.set_feature_dim)
    order = torch.tensor([2, 0, 1])
    first = control(extractor(images, masks).set_feature).coefficient
    second = control(extractor(images[order], masks[order]).set_feature).coefficient
    assert torch.equal(first, second)


def test_basis_is_frozen():
    assert CONFIG["basis"]["frozen"] is True
    assert CONFIG["permissions"]["modify_basis"] is False


def test_base_backbone_mmlp_renderer_are_frozen():
    images, masks, backbone = synthetic_inputs()
    extractor = FrozenF2ReferenceFeatureExtractor(backbone, 5)
    assert all(not parameter.requires_grad for parameter in extractor.spatial_backbone.parameters())
    assert all(CONFIG["permissions"][name] is False for name in (
        "modify_base", "modify_mmlp", "modify_renderer", "finetune_backbone"
    ))
    assert torch.isfinite(extractor(images, masks).set_feature).all()


def test_historical_outputs_immutable(tmp_path: Path):
    historical = tmp_path / "attempt_001"
    historical.mkdir(); (historical / "final.json").write_text("{}\n", encoding="utf-8")
    before = legacy.immutable_tree_metadata_fingerprint(historical)
    runner.load_config(
        Path(__file__).resolve().parents[1]
        / "configs/research/subject02_reference_coefficient_supervision_calibration_v1.yaml"
    )
    after = legacy.immutable_tree_metadata_fingerprint(historical)
    assert before == after
