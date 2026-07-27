from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


TRANSITION_DISTANCE_SAMPLING = (1.0, 1.0)
TRANSITION_DISTANCE_EPSILON = 1e-6
TRANSITION_SMOOTH_L1_BETA = 1.0
TRANSITION_GRADIENT_CAP_EPSILON = 1e-12
TRANSITION_GRADIENT_CAP_FRACTION = 0.5


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


def region_aware_dual_target_loss(
    pred_rgb: torch.Tensor,
    pred_alpha: torch.Tensor,
    target_edit_rgb: torch.Tensor,
    target_base_rgb: torch.Tensor,
    target_edit_core_mask: torch.Tensor,
    target_preserve_mask: torch.Tensor,
    target_protected_mask: torch.Tensor,
    target_transition_mask: torch.Tensor,
    target_clothing_mask: torch.Tensor,
    target_foreground_mask: torch.Tensor,
    target_base_foreground_mask: torch.Tensor,
    *,
    edit_weight: float = 1.0,
    preserve_weight: float = 1.0,
    protected_weight: float = 1.0,
    transition_weight: float = 0.25,
    clothing_weight: float = 1.0,
    alpha_weight: float = 0.5,
    alpha_edit_weight: float = 1.0,
    alpha_transition_weight: float = 0.25,
    alpha_base_weight: float = 1.0,
    transition_alpha_target: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Supervise clothing edits while preserving frozen subject identity pixels.

    Target tensors are consumed only here. The model forward remains conditioned on
    references plus target pose/camera. Empty masks return differentiable zeros.
    """

    prediction = _as_nchw(pred_rgb, channels=3, name="pred_rgb")
    edit_target = _as_nchw(target_edit_rgb, channels=3, name="target_edit_rgb").to(
        device=prediction.device, dtype=prediction.dtype,
    )
    base_target = _as_nchw(target_base_rgb, channels=3, name="target_base_rgb").to(
        device=prediction.device, dtype=prediction.dtype,
    )
    if prediction.shape != edit_target.shape or prediction.shape != base_target.shape:
        raise ValueError("prediction and dual RGB targets must have identical shapes")
    masks = {
        "edit_core": _prepare_mask(target_edit_core_mask, prediction),
        "preserve": _prepare_mask(target_preserve_mask, prediction),
        "protected": _prepare_mask(target_protected_mask, prediction),
        "transition": _prepare_mask(target_transition_mask, prediction),
        "clothing": _prepare_mask(target_clothing_mask, prediction),
    }
    if torch.any(masks["protected"] > masks["preserve"]):
        raise ValueError("protected mask must be a subset of preserve mask")
    if torch.any((masks["edit_core"] > 0) & (masks["protected"] > 0)):
        raise ValueError("edit core and protected masks must not overlap")
    foreground = _prepare_mask(target_foreground_mask, prediction)
    if torch.any((masks["clothing"] > 0) & (masks["protected"] > 0)):
        raise ValueError("clothing supervision mask must exclude protected pixels")
    if torch.any(masks["clothing"] > foreground):
        raise ValueError("clothing supervision mask must be a subset of target foreground")

    edit = _normalized_masked_l1(prediction, edit_target, masks["edit_core"])
    preserve = _normalized_masked_l1(prediction, base_target, masks["preserve"])
    protected = _normalized_masked_l1(prediction, base_target, masks["protected"])
    transition = _normalized_masked_l1(prediction, edit_target, masks["transition"])
    clothing = _normalized_masked_l1(prediction, edit_target, masks["clothing"])
    alpha_reference = _as_nchw(pred_alpha, channels=1, name="pred_alpha")
    alpha_edit_target = _as_nchw(
        foreground, channels=1, name="target_foreground_mask",
    ).to(device=alpha_reference.device, dtype=alpha_reference.dtype)
    alpha_base_target = _as_nchw(
        target_base_foreground_mask,
        channels=1,
        name="target_base_foreground_mask",
    ).to(device=alpha_reference.device, dtype=alpha_reference.dtype)
    if alpha_reference.shape != alpha_edit_target.shape or alpha_reference.shape != alpha_base_target.shape:
        raise ValueError("prediction and dual alpha targets must have identical shapes")
    alpha_protected_mask = masks["protected"].to(
        device=alpha_reference.device, dtype=alpha_reference.dtype,
    )
    alpha_edit_mask = masks["edit_core"].to(device=alpha_reference.device, dtype=alpha_reference.dtype) * (1 - alpha_protected_mask)
    alpha_transition_mask = masks["transition"].to(device=alpha_reference.device, dtype=alpha_reference.dtype) * (1 - alpha_protected_mask)
    alpha_base_mask = torch.maximum(
        masks["preserve"].to(device=alpha_reference.device, dtype=alpha_reference.dtype),
        alpha_protected_mask,
    )
    if transition_alpha_target is None:
        transition_target = boundary_aware_transition_alpha_target(
            alpha_edit_target,
            alpha_base_target,
            masks["edit_core"],
            masks["preserve"],
            masks["protected"],
        )["target"]
    else:
        transition_target = _as_nchw(
            transition_alpha_target,
            channels=1,
            name="transition_alpha_target",
        ).to(device=alpha_reference.device, dtype=alpha_reference.dtype)
        if transition_target.shape != alpha_reference.shape:
            raise ValueError("transition_alpha_target must match predicted alpha shape")
        if torch.any(transition_target < 0) or torch.any(transition_target > 1):
            raise ValueError("transition_alpha_target values must be in [0,1]")
        if torch.any(
            torch.abs(transition_target - alpha_base_target) * alpha_protected_mask > 1e-6
        ):
            raise ValueError("protected transition alpha target must equal base alpha")
    alpha_edit = _normalized_masked_alpha_loss(alpha_reference, alpha_edit_target, alpha_edit_mask)
    alpha_transition = _normalized_masked_smooth_l1(
        alpha_reference,
        transition_target,
        alpha_transition_mask,
        beta=TRANSITION_SMOOTH_L1_BETA,
    )
    alpha_base = _normalized_masked_alpha_loss(alpha_reference, alpha_base_target, alpha_base_mask)
    alpha_total = (
        alpha_edit_weight * alpha_edit["total"]
        + alpha_transition_weight * alpha_transition["total"]
        + alpha_base_weight * alpha_base["total"]
    )
    total = (
        edit_weight * edit
        + preserve_weight * preserve
        + protected_weight * protected
        + transition_weight * transition
        + clothing_weight * clothing
        + alpha_weight * alpha_total
    )
    return {
        "edit": edit,
        "preserve": preserve,
        "protected": protected,
        "transition": transition,
        "clothing": clothing,
        "alpha_edit_bce": alpha_edit["bce"],
        "alpha_edit_dice": alpha_edit["dice"],
        "alpha_edit": alpha_edit["total"],
        "alpha_transition_bce": alpha_transition["zero"],
        "alpha_transition_dice": alpha_transition["zero"],
        "alpha_transition_smooth_l1": alpha_transition["smooth_l1"],
        "alpha_transition": alpha_transition["total"],
        "alpha_base_bce": alpha_base["bce"],
        "alpha_base_dice": alpha_base["dice"],
        "alpha_base": alpha_base["total"],
        "alpha": alpha_total,
        "total": total,
    }


def boundary_aware_transition_alpha_target(
    target_edit_alpha: torch.Tensor,
    target_base_alpha: torch.Tensor,
    edit_core_mask: torch.Tensor,
    preserve_core_mask: torch.Tensor,
    protected_mask: torch.Tensor,
    *,
    epsilon: float = TRANSITION_DISTANCE_EPSILON,
) -> dict[str, torch.Tensor]:
    """Build one globally defined distance-weighted transition alpha target.

    The distance transform is supervision-only and never enters model forward.
    Distances use fixed unit pixel sampling for every sample.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    edit = _as_nchw(target_edit_alpha, channels=1, name="target_edit_alpha")
    base = _as_nchw(target_base_alpha, channels=1, name="target_base_alpha").to(
        device=edit.device, dtype=edit.dtype,
    )
    if edit.shape != base.shape:
        raise ValueError("edit and base alpha targets must have identical shapes")
    if torch.any(edit < 0) or torch.any(edit > 1) or torch.any(base < 0) or torch.any(base > 1):
        raise ValueError("alpha targets must be in [0,1]")
    edit_core = _prepare_mask(edit_core_mask, edit)
    preserve_core = _prepare_mask(preserve_core_mask, edit)
    protected = _prepare_mask(protected_mask, edit)
    weights = []
    edit_distances = []
    base_distances = []
    for batch_index in range(edit.shape[0]):
        edit_binary = edit_core[batch_index, 0].detach().cpu().numpy() >= 0.5
        preserve_binary = preserve_core[batch_index, 0].detach().cpu().numpy() >= 0.5
        d_edit = _euclidean_distance_to_true(edit_binary)
        d_base = _euclidean_distance_to_true(preserve_binary)
        weight = d_base / (d_edit + d_base + epsilon)
        weights.append(torch.from_numpy(weight.astype(np.float32, copy=False)))
        edit_distances.append(torch.from_numpy(d_edit.astype(np.float32, copy=False)))
        base_distances.append(torch.from_numpy(d_base.astype(np.float32, copy=False)))
    w_edit = torch.stack(weights).unsqueeze(1).to(device=edit.device, dtype=edit.dtype).clamp(0, 1)
    d_edit_tensor = torch.stack(edit_distances).unsqueeze(1).to(device=edit.device, dtype=edit.dtype)
    d_base_tensor = torch.stack(base_distances).unsqueeze(1).to(device=edit.device, dtype=edit.dtype)
    w_edit = torch.where(protected > 0, torch.zeros_like(w_edit), w_edit)
    w_base = 1 - w_edit
    target = w_edit * edit + w_base * base
    target = torch.where(protected > 0, base, target).clamp(0, 1)
    if not torch.isfinite(target).all() or not torch.isfinite(w_edit).all():
        raise ValueError("transition alpha target contains NaN or Inf")
    if not torch.allclose(w_edit + w_base, torch.ones_like(w_edit), atol=1e-6, rtol=0):
        raise AssertionError("transition alpha weights do not sum to one")
    return {
        "target": target,
        "w_edit": w_edit,
        "w_base": w_base,
        "d_edit": d_edit_tensor,
        "d_base": d_base_tensor,
    }


def _euclidean_distance_to_true(mask: np.ndarray) -> np.ndarray:
    """Return an exact, dependency-free Euclidean distance transform.

    This is the separable squared-distance transform from Felzenszwalb and
    Huttenlocher. Unit pixel spacing is fixed globally for the V5.3 objective.
    """

    if TRANSITION_DISTANCE_SAMPLING != (1.0, 1.0):
        raise AssertionError("only fixed unit transition distance sampling is supported")
    binary = np.asarray(mask, dtype=np.bool_)
    if binary.ndim != 2:
        raise ValueError("distance transform mask must be two-dimensional")
    if not binary.any():
        raise ValueError("distance transform mask must contain at least one true pixel")
    height, width = binary.shape
    upper_bound = float(height * height + width * width + 1)
    squared = np.where(binary, 0.0, upper_bound)
    vertical = np.empty_like(squared)
    for column in range(width):
        vertical[:, column] = _squared_distance_transform_1d(squared[:, column])
    distance_squared = np.empty_like(squared)
    for row in range(height):
        distance_squared[row, :] = _squared_distance_transform_1d(vertical[row, :])
    return np.sqrt(distance_squared, out=distance_squared)


def compute_static_transition_gradient_cap(
    rgb_gradient_norm: float,
    raw_transition_gradient_norm: float,
    original_coefficient: float,
    *,
    cap_fraction: float = TRANSITION_GRADIENT_CAP_FRACTION,
    epsilon: float = TRANSITION_GRADIENT_CAP_EPSILON,
) -> dict[str, float | bool | str]:
    """Compute the V5.3 coefficient once for the shared decoder trunk.

    The returned coefficient is a static run contract. Callers persist it at
    step 0 and reuse it without recomputation for every optimizer step.
    """

    values = (rgb_gradient_norm, raw_transition_gradient_norm, original_coefficient)
    if any(not np.isfinite(value) or value < 0 for value in values):
        raise ValueError("gradient norms and original coefficient must be finite and non-negative")
    if not np.isfinite(cap_fraction) or cap_fraction <= 0:
        raise ValueError("cap_fraction must be finite and positive")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    calculated = cap_fraction * rgb_gradient_norm / (raw_transition_gradient_norm + epsilon)
    frozen = min(original_coefficient, calculated)
    return {
        "formula": "min(original, cap_fraction * rgb_norm / (raw_transition_norm + epsilon))",
        "rgb_gradient_norm": float(rgb_gradient_norm),
        "raw_transition_gradient_norm": float(raw_transition_gradient_norm),
        "original_coefficient": float(original_coefficient),
        "calculated_coefficient": float(calculated),
        "final_frozen_coefficient": float(frozen),
        "cap_fraction": float(cap_fraction),
        "epsilon": float(epsilon),
        "applied_transition_gradient_norm": float(frozen * raw_transition_gradient_norm),
        "dynamic_updates": False,
    }


def _squared_distance_transform_1d(values: np.ndarray) -> np.ndarray:
    length = int(values.shape[0])
    sites = np.empty(length, dtype=np.int64)
    boundaries = np.empty(length + 1, dtype=np.float64)
    distances = np.empty(length, dtype=np.float64)
    envelope_index = 0
    sites[0] = 0
    boundaries[0] = -np.inf
    boundaries[1] = np.inf
    for query in range(1, length):
        site = int(sites[envelope_index])
        intersection = (
            (float(values[query]) + query * query)
            - (float(values[site]) + site * site)
        ) / (2.0 * (query - site))
        while intersection <= boundaries[envelope_index]:
            envelope_index -= 1
            site = int(sites[envelope_index])
            intersection = (
                (float(values[query]) + query * query)
                - (float(values[site]) + site * site)
            ) / (2.0 * (query - site))
        envelope_index += 1
        sites[envelope_index] = query
        boundaries[envelope_index] = intersection
        boundaries[envelope_index + 1] = np.inf
    envelope_index = 0
    for query in range(length):
        while boundaries[envelope_index + 1] < query:
            envelope_index += 1
        delta = query - int(sites[envelope_index])
        distances[query] = delta * delta + float(values[sites[envelope_index]])
    return distances


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


def _normalized_masked_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if mask.sum().item() == 0:
        return prediction.sum() * 0
    expanded = mask.expand_as(prediction)
    return (torch.abs(prediction - target) * expanded).sum() / expanded.sum()


def _normalized_masked_alpha_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    if mask.sum().item() == 0:
        zero = prediction.sum() * 0
        return {"bce": zero, "dice": zero, "total": zero}
    clamped = prediction.clamp(eps, 1 - eps)
    bce_values = F.binary_cross_entropy(clamped, target, reduction="none")
    bce = (bce_values * mask).sum() / mask.sum()
    intersection = (prediction * target * mask).sum()
    dice = 1 - (2 * intersection + eps) / (
        (prediction * mask).sum() + (target * mask).sum() + eps
    )
    return {"bce": bce, "dice": dice, "total": bce + dice}


def _normalized_masked_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float,
) -> dict[str, torch.Tensor]:
    if beta <= 0:
        raise ValueError("SmoothL1 beta must be positive")
    if mask.sum().item() == 0:
        zero = prediction.sum() * 0
        return {"smooth_l1": zero, "zero": zero, "total": zero}
    values = F.smooth_l1_loss(prediction, target, reduction="none", beta=beta)
    smooth_l1 = (values * mask).sum() / mask.sum()
    zero = smooth_l1 * 0
    return {"smooth_l1": smooth_l1, "zero": zero, "total": smooth_l1}


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
