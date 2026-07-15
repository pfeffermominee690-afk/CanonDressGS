from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def rgb_reconstruction_loss(
    pred_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    mask: torch.Tensor | None = None,
    l1_weight: float = 1.0,
    ssim_weight: float = 0.2,
) -> dict[str, torch.Tensor]:
    """Compute masked RGB L1 and SSIM losses with valid-pixel normalization."""

    prediction = _as_nchw(pred_rgb, channels=3, name="pred_rgb")
    target = _as_nchw(target_rgb, channels=3, name="target_rgb").to(
        device=prediction.device,
        dtype=prediction.dtype,
    )
    if prediction.shape != target.shape:
        raise ValueError("pred_rgb and target_rgb must have identical shapes")
    mask_tensor = _prepare_mask(mask, prediction)
    if mask_tensor is None:
        l1 = torch.abs(prediction - target).mean()
        ssim = _repository_or_fallback_ssim(prediction, target)
    elif mask_tensor.sum().item() == 0:
        zero = prediction.sum() * 0
        l1, ssim = zero, zero
    else:
        expanded_mask = mask_tensor.expand_as(prediction)
        l1 = (torch.abs(prediction - target) * expanded_mask).sum() / expanded_mask.sum()
        ssim = _repository_or_fallback_ssim(
            prediction * mask_tensor,
            target * mask_tensor,
        )
    return {"l1": l1, "ssim": ssim, "total": l1_weight * l1 + ssim_weight * ssim}


def alpha_mask_loss(
    pred_alpha: torch.Tensor,
    target_mask: torch.Tensor,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Compute probability-space BCE and soft Dice foreground losses."""

    prediction = _as_nchw(pred_alpha, channels=1, name="pred_alpha")
    target = _as_nchw(target_mask, channels=1, name="target_mask").to(
        device=prediction.device,
        dtype=prediction.dtype,
    )
    if prediction.shape != target.shape:
        raise ValueError("pred_alpha and target_mask must have identical shapes")
    if torch.any(target < 0) or torch.any(target > 1):
        raise ValueError("target_mask values must be in [0,1]")
    clamped = prediction.clamp(eps, 1 - eps)
    bce = F.binary_cross_entropy(clamped, target)
    intersection = (prediction * target).sum()
    dice = 1 - (2 * intersection + eps) / (prediction.sum() + target.sum() + eps)
    return {"bce": bce, "dice": dice, "total": bce_weight * bce + dice_weight * dice}


class LPIPSLoss(nn.Module):
    """One-time LPIPS wrapper; construction fails clearly when unavailable."""

    def __init__(self, net_type: str = "vgg") -> None:
        super().__init__()
        try:
            from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
        except ImportError as error:
            raise RuntimeError(
                "LPIPS requires torchmetrics; install project rendering dependencies "
                "or set lpips_weight=0"
            ) from error
        self.metric = LearnedPerceptualImagePatchSimilarity(
            net_type=net_type,
            normalize=False,
        )
        self.metric.eval()
        for parameter in self.metric.parameters():
            parameter.requires_grad_(False)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = _as_nchw(prediction, channels=3, name="prediction")
        target = _as_nchw(target, channels=3, name="target").to(
            device=prediction.device,
            dtype=prediction.dtype,
        )
        return self.metric(prediction * 2 - 1, target * 2 - 1)


def combined_rendering_loss(
    pred_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    pred_alpha: torch.Tensor,
    target_mask: torch.Tensor,
    rgb_mask: torch.Tensor | None = None,
    rgb_weight: float = 1.0,
    mask_weight: float = 0.5,
    lpips_weight: float = 0.0,
    ssim_weight: float = 0.2,
    lpips_loss: LPIPSLoss | None = None,
) -> dict[str, torch.Tensor]:
    """Combine RGB, alpha, and optional persistent LPIPS supervision."""

    rgb = rgb_reconstruction_loss(
        pred_rgb,
        target_rgb,
        mask=rgb_mask,
        ssim_weight=ssim_weight,
    )
    alpha = alpha_mask_loss(pred_alpha, target_mask)
    if lpips_weight > 0:
        if lpips_loss is None:
            raise ValueError("lpips_loss instance is required when lpips_weight > 0")
        lpips = lpips_loss(pred_rgb, target_rgb)
    else:
        lpips = rgb["total"].new_zeros(())
    total = rgb_weight * rgb["total"] + mask_weight * alpha["total"] + lpips_weight * lpips
    return {
        "rgb_l1": rgb["l1"],
        "rgb_ssim": rgb["ssim"],
        "lpips": lpips,
        "mask_bce": alpha["bce"],
        "mask_dice": alpha["dice"],
        "total": total,
    }


def _as_nchw(tensor: torch.Tensor, channels: int, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor) or not torch.is_floating_point(tensor):
        raise TypeError(f"{name} must be a floating-point torch.Tensor")
    if tensor.ndim == 3:
        if tensor.shape[0] == channels:
            tensor = tensor.unsqueeze(0)
        elif tensor.shape[-1] == channels:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError(f"{name} does not contain {channels} channels")
    elif tensor.ndim == 4:
        if tensor.shape[1] == channels:
            pass
        elif tensor.shape[-1] == channels:
            tensor = tensor.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"{name} does not contain {channels} channels")
    else:
        raise ValueError(f"{name} must be CHW/HWC or NCHW/NHWC")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return tensor


def _prepare_mask(mask: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor | None:
    if mask is None:
        return None
    prepared = _as_nchw(mask, channels=1, name="mask").to(
        device=reference.device,
        dtype=reference.dtype,
    )
    if prepared.shape[0] != reference.shape[0] or prepared.shape[-2:] != reference.shape[-2:]:
        raise ValueError("mask batch and spatial shape must match RGB tensors")
    if torch.any(prepared < 0) or torch.any(prepared > 1):
        raise ValueError("mask values must be in [0,1]")
    return prepared


def _repository_or_fallback_ssim(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    try:
        from utils.loss_utils import ssim_loss as repository_ssim_loss

        losses = [
            repository_ssim_loss(
                prediction[index].permute(1, 2, 0),
                target[index].permute(1, 2, 0),
            )
            for index in range(prediction.shape[0])
        ]
        return torch.stack(losses).mean()
    except (ImportError, ModuleNotFoundError):
        return _local_ssim_loss(prediction, target)


def _local_ssim_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    kernel_size = 3
    padding = kernel_size // 2
    mu_x = F.avg_pool2d(prediction, kernel_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, kernel_size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(prediction.square(), kernel_size, 1, padding) - mu_x.square()
    sigma_y = F.avg_pool2d(target.square(), kernel_size, 1, padding) - mu_y.square()
    sigma_xy = F.avg_pool2d(prediction * target, kernel_size, 1, padding) - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    ssim = numerator / denominator.clamp_min(1e-8)
    return 1 - ssim.mean()
