from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


ALPHA_THRESHOLD = 1.0 / 255.0
EARLY_TERMINATION_TRANSMITTANCE = 1.0e-4
ALPHA_CLAMP_MAX = 0.999
CULLING_REASONS = (
    "outside_frustum",
    "near_far_clipped",
    "projected_radius_below_minimum",
    "tile_not_assigned",
    "covariance_invalid",
    "conic_rejected",
    "alpha_below_cutoff",
    "behind_transmittance_termination",
    "outside_image",
    "opacity_too_low",
    "no_nearby_gaussian_support",
    "depth_order_abnormal",
    "unknown",
)


def _finite_1d(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains NaN/Inf")
    return value


def reference_alpha_composite(single_alphas: torch.Tensor) -> torch.Tensor:
    """Float64 front-to-back alpha without cutoff or early termination."""

    values = _finite_1d(single_alphas, "single_alphas").to(torch.float64)
    if torch.any((values < 0) | (values > 1)):
        raise ValueError("single_alphas must be in [0,1]")
    return 1.0 - torch.prod(1.0 - values)


def production_alpha_composite(
    single_alphas: torch.Tensor,
    *,
    cutoff: float = ALPHA_THRESHOLD,
    termination: float = EARLY_TERMINATION_TRANSMITTANCE,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    """Mirror gsplat 1.5.3's accepted-contribution and exclusive stop rules."""

    values = _finite_1d(single_alphas, "single_alphas")
    if not 0 <= cutoff < 1 or not 0 < termination < 1:
        raise ValueError("invalid production compositor thresholds")
    transmittance = values.new_tensor(1.0)
    accepted: list[torch.Tensor] = []
    last = -1
    for index, alpha in enumerate(values):
        if float(alpha.detach()) < cutoff:
            continue
        next_transmittance = transmittance * (1.0 - alpha)
        if float(next_transmittance.detach()) <= termination:
            break
        accepted.append(alpha)
        transmittance = next_transmittance
        last = index
    stored = torch.stack(accepted) if accepted else values.new_empty((0,))
    return 1.0 - transmittance, last, stored


def single_gaussian_alpha(
    pixel_xy: torch.Tensor,
    mean_xy: torch.Tensor,
    conic: torch.Tensor,
    opacity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pixel_xy.shape != (2,) or mean_xy.shape != (2,) or conic.shape != (3,) or opacity.numel() != 1:
        raise ValueError("invalid single-Gaussian input shape")
    tensors = (pixel_xy, mean_xy, conic, opacity.reshape(()))
    if not all(torch.isfinite(value).all() for value in tensors):
        raise FloatingPointError("single-Gaussian input contains NaN/Inf")
    delta = mean_xy - pixel_xy
    sigma = 0.5 * (conic[0] * delta[0].square() + conic[2] * delta[1].square()) + conic[1] * delta[0] * delta[1]
    alpha = torch.minimum(opacity.reshape(()) * torch.exp(-sigma), opacity.new_tensor(ALPHA_CLAMP_MAX))
    return alpha, sigma


def deterministic_mask_indices(mask: np.ndarray, count: int) -> list[tuple[int, int]]:
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2 or count < 0:
        raise ValueError("mask must be 2D and count non-negative")
    points = np.argwhere(value)
    if len(points) <= count:
        return [(int(y), int(x)) for y, x in points]
    positions = np.linspace(0, len(points) - 1, num=count, dtype=np.int64)
    return [(int(points[index, 0]), int(points[index, 1])) for index in positions]


def binary_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError("binary IoU shapes differ")
    union = np.logical_or(prediction, target).sum()
    return float(np.logical_and(prediction, target).sum() / max(int(union), 1))


def component_statistics(mask: np.ndarray) -> dict[str, float | int]:
    value = np.asarray(mask, dtype=np.uint8)
    try:
        import cv2
        count, _, stats, _ = cv2.connectedComponentsWithStats(value, connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.empty((0,), dtype=np.int32)
        contours = cv2.findContours(value, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
    except ImportError:
        areas = np.asarray(_component_areas(value.astype(bool)), dtype=np.int64)
        count = len(areas) + 1
        padded = np.pad(value.astype(bool), 1)
        center = padded[1:-1, 1:-1]
        perimeter = float(
            (center & ~padded[:-2, 1:-1]).sum()
            + (center & ~padded[2:, 1:-1]).sum()
            + (center & ~padded[1:-1, :-2]).sum()
            + (center & ~padded[1:-1, 2:]).sum()
        )
    return {
        "component_count": int(max(count - 1, 0)),
        "largest_component_area": int(areas.max()) if areas.size else 0,
        "perimeter": perimeter,
        "boundary_roughness": float(perimeter * perimeter / max(4.0 * math.pi * float(value.sum()), 1.0)),
    }


def largest_internal_hole(foreground: np.ndarray, target_domain: np.ndarray | None = None) -> np.ndarray:
    fg = np.asarray(foreground, dtype=np.uint8)
    try:
        import cv2
        flood = fg.copy()
        padded = np.pad(flood, 1, constant_values=0)
        cv2.floodFill(padded, np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8), (0, 0), 1)
        exterior_background = padded[1:-1, 1:-1].astype(bool)
        holes = (~fg.astype(bool)) & (~exterior_background)
    except ImportError:
        background = ~fg.astype(bool)
        exterior_background = _flood_from_border(background)
        holes = background & ~exterior_background
    if target_domain is not None:
        holes &= np.asarray(target_domain, dtype=bool)
    try:
        import cv2
        count, labels, stats, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), connectivity=8)
        if count <= 1:
            return np.zeros_like(holes)
        label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return labels == label
    except ImportError:
        return _largest_component_numpy(holes)


def _flood_from_border(mask: np.ndarray) -> np.ndarray:
    value = np.asarray(mask, dtype=bool)
    visited = np.zeros_like(value)
    stack = [(int(y), int(x)) for y, x in np.argwhere(value & _border_mask(value.shape))]
    while stack:
        y, x = stack.pop()
        if visited[y, x] or not value[y, x]:
            continue
        visited[y, x] = True
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < value.shape[0] and 0 <= nx < value.shape[1] and not visited[ny, nx] and value[ny, nx]:
                stack.append((ny, nx))
    return visited


def _border_mask(shape: tuple[int, int]) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    result[[0, -1], :] = True; result[:, [0, -1]] = True
    return result


def _component_areas(mask: np.ndarray) -> list[int]:
    value = np.asarray(mask, dtype=bool)
    visited = np.zeros_like(value)
    areas = []
    for start_y, start_x in np.argwhere(value):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]; area = 0
        while stack:
            y, x = stack.pop()
            if visited[y, x] or not value[y, x]:
                continue
            visited[y, x] = True; area += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if (dy or dx) and 0 <= ny < value.shape[0] and 0 <= nx < value.shape[1] and value[ny, nx] and not visited[ny, nx]:
                        stack.append((ny, nx))
        areas.append(area)
    return areas


def _largest_component_numpy(mask: np.ndarray) -> np.ndarray:
    value = np.asarray(mask, dtype=bool)
    visited = np.zeros_like(value); best: list[tuple[int, int]] = []
    for start_y, start_x in np.argwhere(value):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]; component = []
        while stack:
            y, x = stack.pop()
            if visited[y, x] or not value[y, x]:
                continue
            visited[y, x] = True; component.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if (dy or dx) and 0 <= ny < value.shape[0] and 0 <= nx < value.shape[1] and value[ny, nx] and not visited[ny, nx]:
                        stack.append((ny, nx))
        if len(component) > len(best):
            best = component
    result = np.zeros_like(value)
    for y, x in best:
        result[y, x] = True
    return result


def threshold_metrics(
    alpha: np.ndarray,
    target_foreground: np.ndarray,
    trusted_expansion: np.ndarray,
    trusted_removal: np.ndarray,
    trusted_background: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    value = np.asarray(alpha, dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("alpha must be finite HxW")
    if value.min() < -1e-6 or value.max() > 1 + 1e-6 or not 0 <= threshold <= 1:
        raise ValueError("alpha/threshold out of range")
    prediction = value >= threshold
    target = np.asarray(target_foreground, dtype=bool)
    expansion = np.asarray(trusted_expansion, dtype=bool)
    removal = np.asarray(trusted_removal, dtype=bool)
    background = np.asarray(trusted_background, dtype=bool)
    trusted_domain = expansion | removal
    trusted_prediction = prediction & trusted_domain
    holes = largest_internal_hole(prediction, target)
    cloud = prediction & (~target)
    shape = component_statistics(prediction)
    return {
        "threshold": float(threshold),
        "raw_iou": binary_iou(prediction, target),
        "trusted_garment_iou": binary_iou(trusted_prediction, expansion),
        "trusted_expansion_recall": float((prediction & expansion).sum() / max(int(expansion.sum()), 1)),
        "trusted_removal_recall": float(((~prediction) & removal).sum() / max(int(removal.sum()), 1)),
        "internal_hole_pixels": int(holes.sum()),
        "internal_hole_ratio": float(holes.sum() / max(int(target.sum()), 1)),
        "background_fp_pixels": int((prediction & background).sum()),
        "background_leakage": float(value[background].mean()) if background.any() else 0.0,
        "trailing_cloud_area": int(cloud.sum()),
        "max_internal_hole_area": int(largest_internal_hole(prediction, target).sum()),
        **shape,
    }


def quantile_summary(values: torch.Tensor | np.ndarray) -> dict[str, float]:
    tensor = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    tensor = tensor[torch.isfinite(tensor)]
    if not tensor.numel():
        return {name: float("nan") for name in ("p50", "p75", "p90", "p95", "p99", "max")}
    qs = torch.quantile(tensor, torch.tensor([.5, .75, .9, .95, .99], dtype=torch.float64))
    return {"p50": float(qs[0]), "p75": float(qs[1]), "p90": float(qs[2]), "p95": float(qs[3]), "p99": float(qs[4]), "max": float(tensor.max())}


def culling_reason(record: Mapping[str, bool]) -> str:
    """Return one exhaustive reason using the frozen precedence order."""

    precedence = (
        "near_far_clipped", "outside_frustum", "covariance_invalid", "outside_image",
        "opacity_too_low", "projected_radius_below_minimum", "tile_not_assigned",
        "conic_rejected", "alpha_below_cutoff", "behind_transmittance_termination",
        "no_nearby_gaussian_support", "depth_order_abnormal",
    )
    for reason in precedence:
        if bool(record.get(reason, False)):
            return reason
    return "unknown"


def assert_exhaustive_reasons(reasons: Iterable[str]) -> None:
    unknown = set(reasons).difference(CULLING_REASONS)
    if unknown:
        raise ValueError(f"unregistered culling reasons: {sorted(unknown)}")


def downsample_supersampled(value: torch.Tensor, channels: int) -> torch.Tensor:
    if value.ndim != 3 or value.shape[0] != channels or value.shape[1] % 2 or value.shape[2] % 2:
        raise ValueError("supersampled tensor must be CHW with even spatial dimensions")
    return F.avg_pool2d(value.unsqueeze(0), kernel_size=2, stride=2)[0]


@dataclass(frozen=True)
class RenderVariant:
    name: str
    changed_setting: str | None
    value: object
    supported: bool


def registered_render_variants() -> tuple[RenderVariant, ...]:
    return (
        RenderVariant("R0", None, "production", True),
        RenderVariant("R1", "early_termination_transmittance", "relaxed", False),
        RenderVariant("R2", "alpha_cutoff", "lower", False),
        RenderVariant("R3", "radius_clip", "lower_than_zero", False),
        RenderVariant("R4", "dtype", "float32", True),
        RenderVariant("R5", "rasterize_mode", "antialiased", True),
        RenderVariant("R6", "supersampling", 2, True),
        RenderVariant("R7", "tile_culling", "disabled", False),
    )


def validate_variant_contract(variants: Sequence[RenderVariant]) -> None:
    if len(variants) > 8 or [item.name for item in variants] != [f"R{i}" for i in range(len(variants))]:
        raise ValueError("render variant ladder changed")
    for variant in variants:
        if variant.name == "R0" and variant.changed_setting is not None:
            raise ValueError("R0 must be the unchanged production renderer")
        if variant.name != "R0" and variant.changed_setting is None:
            raise ValueError("each non-production variant must change exactly one setting")
