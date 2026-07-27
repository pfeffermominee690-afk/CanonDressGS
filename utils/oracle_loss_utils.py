from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from scene.full_attribute_oracle import OracleForwardOutput
from utils.rendering_loss_utils import LPIPSLoss, alpha_mask_loss, rgb_reconstruction_loss


@dataclass(frozen=True)
class OracleLossWeights:
    rgb: float = 1.0
    ssim: float = 0.2
    lpips: float = 0.0
    alpha: float = 0.5
    clothing_rgb: float = 1.0
    clothing_alpha: float = 0.5
    non_clothing_rgb: float = 0.25
    non_clothing_alpha: float = 0.25
    xyz_magnitude: float = 1e-3
    scaling_magnitude: float = 1e-3
    rotation_magnitude: float = 1e-3
    opacity_magnitude: float = 1e-4
    sh0_magnitude: float = 1e-4
    shN_magnitude: float = 1e-4
    geometry_gate_sparsity: float = 1e-4
    appearance_gate_sparsity: float = 1e-4
    gate_binary: float = 1e-5
    graph_gate_smoothness: float = 1e-4
    graph_residual_smoothness: float = 1e-4

    @classmethod
    def from_mapping(cls, values: Mapping[str, float] | None) -> "OracleLossWeights":
        if values is None:
            return cls()
        unknown = set(values).difference(asdict(cls()))
        if unknown:
            raise ValueError(f"unknown oracle loss weights: {sorted(unknown)}")
        result = cls(**{key: float(value) for key, value in values.items()})
        if any(value < 0 or not torch.isfinite(torch.tensor(value)) for value in asdict(result).values()):
            raise ValueError("oracle loss weights must be finite and nonnegative")
        return result


def dilate_binary_mask(mask: torch.Tensor, radius: int = 3) -> torch.Tensor:
    value = _nchw(mask, 1, "mask")
    if radius < 0:
        raise ValueError("dilation radius must be nonnegative")
    if radius == 0:
        return value
    size = 2 * radius + 1
    return F.max_pool2d(value, kernel_size=size, stride=1, padding=radius)


def oracle_rendering_loss(
    *,
    prediction_rgb: torch.Tensor,
    prediction_alpha: torch.Tensor,
    target_rgb: torch.Tensor,
    target_foreground_mask: torch.Tensor,
    target_clothing_mask: torch.Tensor,
    base_rgb: torch.Tensor,
    base_alpha: torch.Tensor,
    oracle_output: OracleForwardOutput,
    weights: OracleLossWeights,
    lpips_loss: LPIPSLoss | None = None,
    non_clothing_dilation_radius: int = 3,
) -> dict[str, torch.Tensor]:
    pred_rgb = _nchw(prediction_rgb, 3, "prediction_rgb")
    pred_alpha = _nchw(prediction_alpha, 1, "prediction_alpha")
    target = _nchw(target_rgb, 3, "target_rgb").to(pred_rgb)
    foreground = _nchw(target_foreground_mask, 1, "target_foreground_mask").to(pred_rgb)
    clothing = _nchw(target_clothing_mask, 1, "target_clothing_mask").to(pred_rgb)
    base_rgb = _nchw(base_rgb, 3, "base_rgb").to(pred_rgb)
    base_alpha = _nchw(base_alpha, 1, "base_alpha").to(pred_rgb)
    for value in (foreground, clothing):
        if torch.any(value < 0) or torch.any(value > 1):
            raise ValueError("mask values must be in [0,1]")
    if torch.any(clothing > foreground + 1e-4):
        raise ValueError("clothing mask must be a subset of foreground")
    non_clothing = (foreground - dilate_binary_mask(clothing, non_clothing_dilation_radius).to(foreground)).clamp_min(0)

    rgb = rgb_reconstruction_loss(pred_rgb, target, foreground, ssim_weight=1.0)
    alpha = alpha_mask_loss(pred_alpha, foreground)
    clothing_rgb = rgb_reconstruction_loss(pred_rgb, target, clothing, ssim_weight=0)
    clothing_alpha = alpha_mask_loss(pred_alpha * clothing, clothing)
    noncloth_rgb = _masked_l1(pred_rgb, base_rgb, non_clothing)
    noncloth_alpha = _masked_l1(pred_alpha, base_alpha, non_clothing)
    lpips = pred_rgb.new_zeros(())
    if weights.lpips:
        if lpips_loss is None:
            raise ValueError("LPIPSLoss is required when lpips weight is nonzero")
        lpips = lpips_loss(pred_rgb * foreground, target * foreground)

    reg = oracle_output.regularization
    parts = {
        "rgb": rgb["l1"],
        "ssim": rgb["ssim"],
        "lpips": lpips,
        "alpha": alpha["total"],
        "clothing_rgb": clothing_rgb["l1"],
        "clothing_alpha": clothing_alpha["total"],
        "non_clothing_rgb": noncloth_rgb,
        "non_clothing_alpha": noncloth_alpha,
        "xyz_magnitude": reg.residual_magnitude["delta_xyz"],
        "scaling_magnitude": reg.residual_magnitude["delta_log_scaling"],
        "rotation_magnitude": reg.residual_magnitude["delta_rotvec"],
        "opacity_magnitude": reg.residual_magnitude["delta_opacity_logit"],
        "sh0_magnitude": reg.residual_magnitude["delta_sh0"],
        "shN_magnitude": reg.residual_magnitude["delta_shN"],
        "geometry_gate_sparsity": reg.geometry_gate_sparsity,
        "appearance_gate_sparsity": reg.appearance_gate_sparsity,
        "gate_binary": reg.gate_binary,
        "graph_gate_smoothness": reg.graph_gate_smoothness,
        "graph_residual_smoothness": reg.graph_residual_smoothness,
    }
    total = pred_rgb.new_zeros(())
    for name, value in parts.items():
        total = total + getattr(weights, name) * value
    parts["total"] = total
    return parts


def mask_iou(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    pred = _nchw(prediction, 1, "prediction") >= threshold
    truth = _nchw(target, 1, "target") >= threshold
    intersection = (pred & truth).sum().float()
    union = (pred | truth).sum().float()
    return torch.where(union > 0, intersection / union, torch.ones_like(union))


def _masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.expand_as(prediction)
    if expanded.sum().item() == 0:
        return prediction.sum() * 0
    return ((prediction - target).abs() * expanded).sum() / expanded.sum()


def _nchw(value: torch.Tensor, channels: int, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
        raise TypeError(f"{name} must be a floating-point tensor")
    if value.ndim == 3 and value.shape[0] == channels:
        value = value.unsqueeze(0)
    elif value.ndim == 3 and value.shape[-1] == channels:
        value = value.permute(2, 0, 1).unsqueeze(0)
    elif value.ndim == 4 and value.shape[1] == channels:
        pass
    elif value.ndim == 4 and value.shape[-1] == channels:
        value = value.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"{name} has ambiguous or invalid channel shape")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return value
