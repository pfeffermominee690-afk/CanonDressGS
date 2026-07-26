"""Additive screen-space support proxies for research-only placement audits.

The functions in this module never composite alpha and never read target data.
Image-derived regions belong in an offline caller after a target-independent
candidate grid and proxy field have been constructed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


ALPHA_CUTOFF = 1.0 / 255.0
PROXY_A_NAME = "SOFT_PRE_CONTRIBUTOR_COUNT_V1"
PROXY_B_NAME = "FIXED_RADIUS_CENTER_DENSITY_V1"
PROXY_C_NAME = "SOFT_KNN_SUPPORT_DISTANCE_V1"
FROZEN_PROXY_A_NAME = "DIFFERENTIABLE_SOFT_PRECONTRIBUTOR_PROXY_V1"
FROZEN_PROXY_B_NAME = "DIFFERENTIABLE_CENTER_DENSITY_PROXY_V1"
TEMPERATURES = (0.10, 0.20, 0.40)
RADII_PIXELS = (2.0, 4.0, 8.0)
K_VALUES = (4, 8, 16)


@dataclass(frozen=True)
class CandidateGrid:
    """Detached coarse spatial index; exact gathered distances retain grad."""

    cell_size: int
    width: int
    height: int
    cells: Mapping[int, torch.Tensor]
    source_count: int
    construction: str
    target_fields_used: bool = False


def _require_projected_inputs(
    projected_means: torch.Tensor,
    conic: torch.Tensor | None = None,
    opacity: torch.Tensor | None = None,
) -> None:
    if projected_means.ndim != 2 or projected_means.shape[1] != 2:
        raise ValueError("projected_means must have shape [N,2]")
    if not torch.isfinite(projected_means).all():
        raise FloatingPointError("projected_means contains NaN/Inf")
    count = projected_means.shape[0]
    if conic is not None and conic.shape != (count, 3):
        raise ValueError("conic must have shape [N,3]")
    if opacity is not None and opacity.reshape(-1).shape != (count,):
        raise ValueError("opacity must contain one value per Gaussian")
    for name, value in (("conic", conic), ("opacity", opacity)):
        if value is not None and not torch.isfinite(value).all():
            raise FloatingPointError(f"{name} contains NaN/Inf")


def differentiable_projected_means(
    posed_xyz: torch.Tensor,
    w2c: torch.Tensor,
    K: torch.Tensor,
) -> torch.Tensor:
    """Project posed centers while detaching the camera contract."""

    if posed_xyz.ndim != 2 or posed_xyz.shape[1] != 3:
        raise ValueError("posed_xyz must have shape [N,3]")
    if w2c.shape != (4, 4) or K.shape != (3, 3):
        raise ValueError("w2c and K must have shapes [4,4] and [3,3]")
    view = w2c.detach().to(posed_xyz)
    intrinsic = K.detach().to(posed_xyz)
    camera_xyz = posed_xyz @ view[:3, :3].T + view[:3, 3]
    depth = camera_xyz[:, 2:3]
    if torch.any(depth.detach() <= 0):
        depth = torch.where(depth.detach() > 0, depth, torch.ones_like(depth))
    normalized = camera_xyz[:, :2] / depth
    result = normalized @ intrinsic[:2, :2].T + intrinsic[:2, 2]
    if not torch.isfinite(result).all():
        raise FloatingPointError("differentiable projection produced NaN/Inf")
    return result


def dense_soft_precontributor_count(
    projected_means: torch.Tensor,
    conic: torch.Tensor,
    opacity: torch.Tensor,
    pixels_xy: torch.Tensor,
    *,
    temperature: float,
    valid: torch.Tensor | None = None,
    gaussian_chunk: int = 4096,
) -> torch.Tensor:
    """Exact chunked additive Proxy A on registered query pixels.

    This is deliberately a sum of independent soft eligibility values. It has
    no transmittance, depth ordering, alpha accumulation, or early termination.
    """

    _require_projected_inputs(projected_means, conic, opacity)
    if temperature not in TEMPERATURES:
        raise ValueError(f"temperature must be one of {TEMPERATURES}")
    if pixels_xy.ndim != 2 or pixels_xy.shape[1] != 2:
        raise ValueError("pixels_xy must have shape [P,2]")
    if gaussian_chunk <= 0:
        raise ValueError("gaussian_chunk must be positive")
    count = projected_means.shape[0]
    eligible = (
        torch.ones(count, dtype=torch.bool, device=projected_means.device)
        if valid is None else valid.detach().to(device=projected_means.device, dtype=torch.bool)
    )
    if eligible.shape != (count,):
        raise ValueError("valid must have shape [N]")
    detached_conic = conic.detach().to(projected_means)
    detached_opacity = opacity.detach().reshape(-1).to(projected_means)
    pixels = pixels_xy.detach().to(projected_means)
    threshold = math.log(ALPHA_CUTOFF)
    support = projected_means.new_zeros(pixels.shape[0])
    for start in range(0, count, gaussian_chunk):
        stop = min(start + gaussian_chunk, count)
        means = projected_means[start:stop]
        delta = means[None] - pixels[:, None]
        local_conic = detached_conic[start:stop]
        sigma = (
            0.5 * local_conic[None, :, 0] * delta[:, :, 0].square()
            + local_conic[None, :, 1] * delta[:, :, 0] * delta[:, :, 1]
            + 0.5 * local_conic[None, :, 2] * delta[:, :, 1].square()
        )
        log_alpha = torch.log(detached_opacity[start:stop].clamp_min(torch.finfo(means.dtype).tiny))[None] - sigma
        weights = torch.sigmoid((log_alpha - threshold) / float(temperature))
        weights = torch.where(eligible[start:stop][None] & torch.isfinite(sigma) & (sigma >= 0), weights, torch.zeros_like(weights))
        support = support + weights.sum(dim=1)
    return support


def exact_instrumented_precontributor_count(
    projected_means: torch.Tensor,
    conic: torch.Tensor,
    opacity: torch.Tensor,
    pixels_xy: torch.Tensor,
    *,
    valid: torch.Tensor,
    gaussian_chunk: int = 4096,
) -> torch.Tensor:
    """Vectorized equivalent of the independent instrumented N_pre oracle."""

    _require_projected_inputs(projected_means, conic, opacity)
    if valid.shape != (projected_means.shape[0],):
        raise ValueError("valid must have shape [N]")
    pixels = pixels_xy.detach().to(projected_means)
    result = torch.zeros(pixels.shape[0], dtype=torch.int64, device=projected_means.device)
    for start in range(0, projected_means.shape[0], gaussian_chunk):
        stop = min(start + gaussian_chunk, projected_means.shape[0])
        delta = projected_means[start:stop][None] - pixels[:, None]
        local_conic = conic.detach().to(projected_means)[start:stop]
        sigma = 0.5 * local_conic[None, :, 0] * delta[:, :, 0].square() + local_conic[None, :, 1] * delta[:, :, 0] * delta[:, :, 1] + 0.5 * local_conic[None, :, 2] * delta[:, :, 1].square()
        alpha = opacity.detach().reshape(-1).to(projected_means)[start:stop][None] * torch.exp(-sigma)
        mask = valid.detach().to(result.device, dtype=torch.bool)[start:stop][None] & torch.isfinite(sigma) & (sigma >= 0) & torch.isfinite(alpha) & (alpha >= ALPHA_CUTOFF)
        result = result + mask.sum(dim=1)
    return result


def dense_fixed_center_density(
    projected_means: torch.Tensor,
    pixels_xy: torch.Tensor,
    *,
    radius_pixels: float,
    valid: torch.Tensor | None = None,
    gaussian_chunk: int = 4096,
) -> torch.Tensor:
    """Exact chunked additive Proxy B, independent of Gaussian attributes."""

    _require_projected_inputs(projected_means)
    if float(radius_pixels) not in RADII_PIXELS:
        raise ValueError(f"radius_pixels must be one of {RADII_PIXELS}")
    if pixels_xy.ndim != 2 or pixels_xy.shape[1] != 2:
        raise ValueError("pixels_xy must have shape [P,2]")
    count = projected_means.shape[0]
    eligible = (
        torch.ones(count, dtype=torch.bool, device=projected_means.device)
        if valid is None else valid.detach().to(device=projected_means.device, dtype=torch.bool)
    )
    if eligible.shape != (count,):
        raise ValueError("valid must have shape [N]")
    pixels = pixels_xy.detach().to(projected_means)
    support = projected_means.new_zeros(pixels.shape[0])
    denominator = 2.0 * float(radius_pixels) ** 2
    for start in range(0, count, gaussian_chunk):
        stop = min(start + gaussian_chunk, count)
        delta = projected_means[start:stop][None] - pixels[:, None]
        weights = torch.exp(-delta.square().sum(dim=-1) / denominator)
        support = support + torch.where(eligible[start:stop][None], weights, torch.zeros_like(weights)).sum(dim=1)
    return support


def build_preregistered_candidate_grid(
    projected_means: torch.Tensor,
    *,
    width: int,
    height: int,
    support_radius_xy: torch.Tensor,
    cell_size: int = 16,
    construction: str,
) -> CandidateGrid:
    """Build a target-independent detached bbox-to-cell index."""

    _require_projected_inputs(projected_means)
    if support_radius_xy.shape != projected_means.shape:
        raise ValueError("support_radius_xy must have shape [N,2]")
    if width <= 0 or height <= 0 or cell_size <= 0:
        raise ValueError("image and cell dimensions must be positive")
    means = projected_means.detach().cpu()
    radii = support_radius_xy.detach().cpu()
    cells_x = math.ceil(width / cell_size)
    cells_y = math.ceil(height / cell_size)
    buckets: dict[int, list[int]] = {}
    valid = torch.isfinite(means).all(dim=1) & torch.isfinite(radii).all(dim=1) & (radii >= 0).all(dim=1)
    for index in torch.where(valid)[0].tolist():
        x, y = (float(value) for value in means[index])
        rx, ry = (float(value) for value in radii[index])
        x0 = max(0, min(cells_x - 1, math.floor((x - rx) / cell_size)))
        x1 = max(0, min(cells_x - 1, math.floor((x + rx) / cell_size)))
        y0 = max(0, min(cells_y - 1, math.floor((y - ry) / cell_size)))
        y1 = max(0, min(cells_y - 1, math.floor((y + ry) / cell_size)))
        if x + rx < 0 or x - rx >= width or y + ry < 0 or y - ry >= height:
            continue
        for cell_y in range(y0, y1 + 1):
            for cell_x in range(x0, x1 + 1):
                buckets.setdefault(cell_y * cells_x + cell_x, []).append(index)
    tensors = {
        key: torch.tensor(sorted(set(values)), dtype=torch.long)
        for key, values in buckets.items()
    }
    return CandidateGrid(
        cell_size=cell_size, width=width, height=height, cells=tensors,
        source_count=int(projected_means.shape[0]), construction=construction,
    )


def proxy_a_candidate_radii(
    covariance2d: torch.Tensor,
    opacity: torch.Tensor,
    *,
    tail_log_margin: float = 8.0,
) -> torch.Tensor:
    """Conservative axis-aligned extent where omitted sigmoid mass is tiny."""

    if covariance2d.ndim != 3 or covariance2d.shape[1:] != (2, 2):
        raise ValueError("covariance2d must have shape [N,2,2]")
    values = opacity.detach().reshape(-1).to(covariance2d)
    if values.shape[0] != covariance2d.shape[0]:
        raise ValueError("opacity count mismatch")
    margin = torch.log((values / ALPHA_CUTOFF).clamp_min(1.0)) + float(tail_log_margin)
    diagonal = torch.diagonal(covariance2d.detach(), dim1=1, dim2=2).clamp_min(0)
    return torch.sqrt(2.0 * margin[:, None] * diagonal)


def fixed_radius_candidate_radii(projected_means: torch.Tensor, *, maximum_radius: float = 8.0) -> torch.Tensor:
    if maximum_radius != 8.0:
        raise ValueError("maximum_radius is frozen at 8px")
    return torch.full_like(projected_means.detach(), 6.0 * maximum_radius)


def candidates_for_pixels(grid: CandidateGrid, pixels_xy: torch.Tensor) -> torch.Tensor:
    """Return a deterministic padded candidate matrix without target semantics."""

    if pixels_xy.ndim != 2 or pixels_xy.shape[1] != 2:
        raise ValueError("pixels_xy must have shape [P,2]")
    cells_x = math.ceil(grid.width / grid.cell_size)
    rows: list[torch.Tensor] = []
    for x_value, y_value in pixels_xy.detach().cpu().tolist():
        x, y = int(math.floor(x_value / grid.cell_size)), int(math.floor(y_value / grid.cell_size))
        if x < 0 or y < 0 or x >= cells_x or y >= math.ceil(grid.height / grid.cell_size):
            rows.append(torch.empty(0, dtype=torch.long))
        else:
            rows.append(grid.cells.get(y * cells_x + x, torch.empty(0, dtype=torch.long)))
    maximum = max((row.numel() for row in rows), default=0)
    output = torch.full((len(rows), maximum), -1, dtype=torch.long, device=pixels_xy.device)
    for index, row in enumerate(rows):
        if row.numel():
            output[index, : row.numel()] = row.to(output.device)
    return output


def sparse_soft_precontributor_count(
    projected_means: torch.Tensor,
    conic: torch.Tensor,
    opacity: torch.Tensor,
    pixels_xy: torch.Tensor,
    candidate_indices: torch.Tensor,
    *,
    temperature: float,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    _require_projected_inputs(projected_means, conic, opacity)
    if temperature not in TEMPERATURES:
        raise ValueError(f"temperature must be one of {TEMPERATURES}")
    if candidate_indices.ndim != 2 or candidate_indices.shape[0] != pixels_xy.shape[0]:
        raise ValueError("candidate_indices must have shape [P,K]")
    if candidate_indices.shape[1] == 0:
        return projected_means.sum() * 0 + projected_means.new_zeros(pixels_xy.shape[0])
    mask = candidate_indices >= 0
    safe = candidate_indices.clamp_min(0)
    means = projected_means[safe]
    delta = means - pixels_xy.detach().to(projected_means)[:, None]
    detached_conic = conic.detach().to(projected_means)[safe]
    sigma = 0.5 * detached_conic[:, :, 0] * delta[:, :, 0].square() + detached_conic[:, :, 1] * delta[:, :, 0] * delta[:, :, 1] + 0.5 * detached_conic[:, :, 2] * delta[:, :, 1].square()
    log_alpha = torch.log(opacity.detach().reshape(-1).to(projected_means)[safe].clamp_min(torch.finfo(projected_means.dtype).tiny)) - sigma
    if valid is not None:
        mask = mask & valid.detach().to(mask.device, dtype=torch.bool)[safe]
    weights = torch.sigmoid((log_alpha - math.log(ALPHA_CUTOFF)) / float(temperature))
    return torch.where(mask & torch.isfinite(sigma) & (sigma >= 0), weights, torch.zeros_like(weights)).sum(dim=1)


def sparse_fixed_center_density(
    projected_means: torch.Tensor,
    pixels_xy: torch.Tensor,
    candidate_indices: torch.Tensor,
    *,
    radius_pixels: float,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    _require_projected_inputs(projected_means)
    if float(radius_pixels) not in RADII_PIXELS:
        raise ValueError(f"radius_pixels must be one of {RADII_PIXELS}")
    if candidate_indices.ndim != 2 or candidate_indices.shape[0] != pixels_xy.shape[0]:
        raise ValueError("candidate_indices must have shape [P,K]")
    if candidate_indices.shape[1] == 0:
        return projected_means.sum() * 0 + projected_means.new_zeros(pixels_xy.shape[0])
    mask = candidate_indices >= 0
    safe = candidate_indices.clamp_min(0)
    delta = projected_means[safe] - pixels_xy.detach().to(projected_means)[:, None]
    if valid is not None:
        mask = mask & valid.detach().to(mask.device, dtype=torch.bool)[safe]
    weights = torch.exp(-delta.square().sum(dim=-1) / (2.0 * float(radius_pixels) ** 2))
    return torch.where(mask, weights, torch.zeros_like(weights)).sum(dim=1)


def estimate_soft_knn_resources(
    gaussian_count: int,
    width: int,
    height: int,
    *,
    support_stride: int = 4,
    bytes_per_value: int = 4,
) -> dict[str, float | int | str]:
    pixels = math.ceil(width / support_stride) * math.ceil(height / support_stride)
    elements = int(gaussian_count) * int(pixels)
    matrix_bytes = elements * bytes_per_value
    return {
        "gaussian_count": int(gaussian_count), "support_pixels": int(pixels),
        "distance_matrix_elements": int(elements), "single_matrix_bytes": int(matrix_bytes),
        "single_matrix_gib": float(matrix_bytes / 2**30),
        "minimum_forward_plus_sort_workspace_gib": float(3 * matrix_bytes / 2**30),
        "complexity": "O(P*N) storage and O(P*N*log(N)) sorting",
        "status": "NOT_FEASIBLE_FOR_FORMAL_TRAINING",
    }


def deterministic_support_grid(width: int, height: int, *, stride: int = 4, device: torch.device | str = "cpu") -> torch.Tensor:
    if width <= 0 or height <= 0 or stride <= 0:
        raise ValueError("width, height, and stride must be positive")
    ys = torch.arange(0, height, stride, dtype=torch.float32, device=device) + 0.5
    xs = torch.arange(0, width, stride, dtype=torch.float32, device=device) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)


def average_ranks(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float64)
    if tensor.numel() == 0:
        return tensor
    order = torch.argsort(tensor, stable=True)
    ranks = torch.empty_like(tensor)
    start = 0
    while start < tensor.numel():
        stop = start + 1
        while stop < tensor.numel() and tensor[order[stop]] == tensor[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    x, y = average_ranks(left), average_ranks(right)
    x, y = x - x.mean(), y - y.mean()
    denominator = torch.sqrt(x.square().sum() * y.square().sum())
    if denominator <= 0:
        return 0.0
    return float((x * y).sum() / denominator)


def deterministic_mask_pixels(mask: torch.Tensor, maximum: int) -> torch.Tensor:
    """Select evenly spaced full-resolution pixel centers from a mask."""

    if mask.ndim != 2 or maximum <= 0:
        raise ValueError("mask must be [H,W] and maximum must be positive")
    coordinates = torch.nonzero(mask.detach().cpu().to(torch.bool), as_tuple=False)
    if coordinates.numel() == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=mask.device)
    if coordinates.shape[0] > maximum:
        pick = torch.linspace(0, coordinates.shape[0] - 1, maximum).round().long()
        coordinates = coordinates[pick]
    return torch.stack((coordinates[:, 1].float() + 0.5, coordinates[:, 0].float() + 0.5), dim=1).to(mask.device)


def qualify_proxy_candidate(
    state_rows: Sequence[Mapping[str, object]],
    pixel_groups: Sequence[Mapping[str, object]],
    *,
    focus_ratio_min: float = 1.50,
    global_spearman_min: float = 0.90,
    outfit_spearman_min: float = 0.80,
    pixel_spearman_min: float = 0.75,
) -> dict[str, object]:
    """Apply the frozen static gates without visual or training outcomes."""

    if not state_rows:
        raise ValueError("state_rows cannot be empty")
    global_spearman = spearman_correlation(
        [float(row["proxy_mean"]) for row in state_rows],
        [float(row["npre_mean"]) for row in state_rows],
    )
    outfit_spearman = {
        outfit: spearman_correlation(
            [float(row["proxy_mean"]) for row in state_rows if row["outfit"] == outfit],
            [float(row["npre_mean"]) for row in state_rows if row["outfit"] == outfit],
        )
        for outfit in ("O01", "O08")
    }
    ratios: dict[str, float] = {}
    ordering: dict[str, bool] = {}
    underfill: dict[str, bool] = {}
    for view in ("back", "right"):
        selected = {str(row["state"]): row for row in state_rows if row["outfit"] == "O08" and row["view"] == view}
        if set(selected) != {"P1", "P2", "P3"}:
            ratios[view], ordering[view], underfill[view] = 0.0, False, False
            continue
        p1, p2, p3 = selected["P1"], selected["P2"], selected["P3"]
        ratios[view] = float(p1["proxy_mean"]) / max(float(p3["proxy_mean"]), 1e-12)
        ordering[view] = float(p1["proxy_mean"]) > float(p2["proxy_mean"]) and float(p1["proxy_mean"]) > float(p3["proxy_mean"])
        underfill[view] = float(p2["low_support_fraction"]) > float(p1["low_support_fraction"]) and float(p3["low_support_fraction"]) > float(p1["low_support_fraction"])
    per_view_pixel = [float(row["spearman"]) for row in pixel_groups if int(row.get("pixel_count", 0)) >= 2]
    median_pixel = float(torch.tensor(per_view_pixel, dtype=torch.float64).median()) if per_view_pixel else 0.0
    cloud = [row for row in state_rows if bool(row.get("cloud_required", False))]
    cloud_pass = bool(cloud) and all(bool(row.get("cloud_anomaly", False)) for row in cloud)
    o01_p1 = [float(row["low_support_fraction"]) for row in state_rows if row["outfit"] == "O01" and row["state"] == "P1"]
    o01_stability = bool(o01_p1) and not all(value >= 0.5 for value in o01_p1)
    gates = {
        "A_focus_order_and_ratio": all(ordering.values()) and all(value >= focus_ratio_min for value in ratios.values()),
        "B_global_spearman": global_spearman >= global_spearman_min,
        "C_per_outfit_spearman": all(value >= outfit_spearman_min for value in outfit_spearman.values()),
        "D_pixel_spearman": median_pixel >= pixel_spearman_min,
        "E_underfill_discrimination": all(underfill.values()),
        "F_cloud_detection": cloud_pass,
        "G_O01_stability": o01_stability,
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates, "focus_p1_p3_ratio": ratios,
        "focus_ordering": ordering, "underfill_discrimination": underfill,
        "global_spearman": global_spearman,
        "outfit_spearman": outfit_spearman,
        "median_per_view_pixel_spearman": median_pixel,
        "cloud_detection": cloud_pass, "o01_stability": o01_stability,
    }


def anti_saturation_metrics(values: torch.Tensor) -> dict[str, float | bool]:
    finite = values.detach().flatten().float()
    finite = finite[torch.isfinite(finite)]
    if not finite.numel():
        return {"finite": False, "p50": 0.0, "p95": 0.0, "p95_p50_ratio": 0.0, "near_max_fraction": 1.0}
    p50 = float(torch.quantile(finite, 0.50))
    p95 = float(torch.quantile(finite, 0.95))
    maximum = float(finite.max())
    near_max = float((finite >= maximum * 0.99).float().mean()) if maximum > 0 else 1.0
    return {
        "finite": True, "p50": p50, "p95": p95,
        "p95_p50_ratio": p95 / max(p50, 1e-12),
        "near_max_fraction": near_max,
    }
