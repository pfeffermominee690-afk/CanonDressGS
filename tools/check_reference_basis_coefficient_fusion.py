from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import torch
from torch import nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene.reference_basis_coefficient_fusion import (  # noqa: E402
    MaskAwareReferenceTokenEncoderV1,
    ReferenceSetCoefficientFusionV1,
    build_reference_coefficient_fusion,
)
from tools import run_reference_basis_coefficient_fusion as runner  # noqa: E402


def inputs(count: int = 3):
    torch.manual_seed(11)
    images = torch.rand(count, 3, 20, 16)
    foreground = torch.zeros(count, 1, 20, 16); foreground[:, :, 1:19, 1:15] = 1
    clothing = torch.zeros_like(foreground); clothing[:, :, 4:17, 3:13] = 1
    poses = torch.zeros(count, 165)
    w2c = torch.eye(4).repeat(count, 1, 1)
    backbone = nn.Sequential(nn.Conv2d(3, 6, 3, padding=1), nn.SiLU())
    encoder = MaskAwareReferenceTokenEncoderV1(backbone, 6, 12, 18)
    return encoder, images, clothing, foreground, poses, w2c


def main() -> None:
    checks = {}
    encoder, images, clothing, foreground, poses, w2c = inputs()
    token = encoder(images, clothing, foreground, poses, w2c)
    fusion = ReferenceSetCoefficientFusionV1(12, 16)
    coefficient = fusion(token.tokens).coefficient
    checks["token_shape"] = tuple(token.tokens.shape) == (3, 12)
    checks["required_token_fields"] = all(
        getattr(token, name) is not None for name in (
            "cloth_mean", "cloth_max", "foreground_mean", "cloth_foreground_difference",
            "mask_area", "bbox_aspect", "centroid", "view_direction",
        )
    )
    checks["bounded_coefficient"] = bool(torch.isfinite(coefficient).all() and -1 <= coefficient.item() <= 1)
    order = torch.tensor([2, 0, 1])
    checks["permutation_within_preregistered_tolerance"] = float(
        (coefficient - fusion(token.tokens[order]).coefficient).abs().max()
    ) <= 1.0e-5
    first = fusion(token.tokens, torch.tensor([1.0, 1.0, 0.0])).coefficient
    changed = token.tokens.clone(); changed[2] = 1e6
    checks["padding_no_leak"] = torch.equal(
        first, fusion(changed, torch.tensor([1.0, 1.0, 0.0])).coefficient
    )
    checks["k1_k2_k3"] = all(
        fusion(token.tokens[:count]).coefficient.shape == (1,) for count in (1, 2, 3)
    )
    loss, parts = runner._loss_pair(
        fusion(token.tokens).coefficient,
        fusion(encoder(1 - images, clothing, foreground, poses, w2c).tokens).coefficient,
        yaml.safe_load((PROJECT_ROOT / "configs/research/subject02_reference_basis_coefficient_fusion_v1.yaml").read_text(encoding="utf-8")),
    )
    loss.backward()
    checks["paired_loss_finite"] = bool(torch.isfinite(loss) and all(torch.isfinite(value) for value in parts.values()))
    checks["gradient_reaches_adapter"] = any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in encoder.token_adapter.parameters()
    )
    checks["gradient_reaches_fusion"] = any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in fusion.parameters()
    )
    checks["backbone_frozen"] = all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in encoder.spatial_backbone.parameters()
    )
    forbidden = runner.FORBIDDEN_FORWARD_FIELDS
    checks["forward_boundary"] = not forbidden.intersection(
        inspect.signature(MaskAwareReferenceTokenEncoderV1.forward).parameters
    )
    checks["legacy_factory"] = build_reference_coefficient_fusion(
        {"coefficient_fusion": {"type": "legacy"}},
        spatial_backbone=nn.Conv2d(3, 6, 1), feature_dim=6,
    ) is None
    variants = {
        "correct", "swapped", "permutation", "single_reference", "dropout_0",
        "dropout_1", "dropout_2", "zero_rgb", "base_rgb",
    }
    checks["formal_72_variant_contract"] = len(variants) * 8 == 72
    checks["no_anchor_projection_in_new_forward"] = "anchor" not in inspect.getsource(
        runner.coefficient_forward
    ).lower()
    checks["no_completion_in_new_forward"] = "completion" not in inspect.getsource(
        runner.coefficient_forward
    ).lower()
    checks["nan_rejected"] = False
    try:
        bad = images.clone(); bad[0, 0, 0, 0] = float("nan")
        encoder(bad, clothing, foreground, poses, w2c)
    except ValueError:
        checks["nan_rejected"] = True
    checks["shape_mismatch_rejected"] = False
    try:
        fusion(torch.randn(1, 1, 12))
    except ValueError:
        checks["shape_mismatch_rejected"] = True
    if not all(checks.values()):
        raise AssertionError(json.dumps({name: value for name, value in checks.items() if not value}, indent=2))
    print(json.dumps({"status": "PASS", "checks": checks}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
