"""Read-only Gaussian projection/admission diagnostics for the RF audit.

This module deliberately does not wrap or replace the production renderer.  The
backend helper is opt-in and calls gsplat's already-installed low-level
projection/intersection operations after a production render has completed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


ALPHA_CUTOFF = 1.0 / 255.0
ALPHA_CLAMP_MAX = 0.999
STAGE_NAMES = (
    "S0_CANONICAL_VALID",
    "S1_POSED_VALID",
    "S2_CAMERA_VALID",
    "S3_COVARIANCE_VALID",
    "S4_IMAGE_OVERLAP",
    "S5_TILE_BBOX",
    "S6_TILE_INTERSECTION_EMITTED",
    "S7_DEPTH_SORTED",
    "S8_PIXEL_ALPHA_ELIGIBLE",
    "S9_PIXEL_CONTRIBUTED",
)
REJECTION_REASONS = (
    "NONFINITE_CANONICAL_STATE",
    "NONFINITE_POSED_STATE",
    "NONFINITE_CAMERA_POINT",
    "BEHIND_NEAR_PLANE",
    "BEYOND_FAR_PLANE",
    "INVALID_COVARIANCE",
    "NONPOSITIVE_DETERMINANT",
    "OPACITY_BELOW_CUTOFF",
    "RADIUS_LE_ZERO",
    "NO_IMAGE_OVERLAP",
    "ZERO_TILE_OVERLAP",
    "TILE_BBOX_OVERFLOW",
    "INTERSECTION_ALLOCATION_OMISSION",
    "BACKEND_INTERNAL_REJECTION",
    "BACKEND_UNEXPOSED_REJECTION",
    "NOT_IN_TARGET_PIXEL_TILE",
    "INVALID_PIXEL_CONIC",
    "ALPHA_BELOW_CUTOFF",
    "BEHIND_EARLY_TERMINATION",
)


@dataclass(frozen=True)
class InstrumentationOptions:
    enabled: bool = False
    eps2d: float = 0.3
    near_plane: float = 0.1
    far_plane: float = 1.0e10
    radius_clip: float = 0.0
    tile_size: int = 16
    alpha_cutoff: float = ALPHA_CUTOFF
    alpha_clamp_max: float = ALPHA_CLAMP_MAX


def _require_shape(value: torch.Tensor, shape: tuple[int | None, ...], name: str) -> None:
    if value.ndim != len(shape) or any(expected is not None and actual != expected for actual, expected in zip(value.shape, shape)):
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")


def _covariance_upper_triangle(covars: torch.Tensor) -> torch.Tensor:
    """Return gsplat's [xx, xy, xz, yy, yz, zz] covariance layout."""

    _require_shape(covars, (None, 3, 3), "covars")
    return torch.stack(
        (covars[:, 0, 0], covars[:, 0, 1], covars[:, 0, 2], covars[:, 1, 1], covars[:, 1, 2], covars[:, 2, 2]),
        dim=-1,
    )


def tile_bboxes(
    means2d: torch.Tensor,
    radii_xy: torch.Tensor,
    *,
    tile_size: int,
    tile_width: int,
    tile_height: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mirror gsplat's inclusive-min/exclusive-max tile rectangle contract."""

    _require_shape(means2d, (None, 2), "means2d")
    _require_shape(radii_xy, (means2d.shape[0], 2), "radii_xy")
    if tile_size <= 0 or tile_width <= 0 or tile_height <= 0:
        raise ValueError("tile dimensions must be positive")
    means = means2d.to(torch.float64) / float(tile_size)
    radii = radii_xy.to(torch.float64) / float(tile_size)
    minimum = torch.floor(means - radii).to(torch.int64)
    maximum = torch.ceil(means + radii).to(torch.int64)
    bounds = torch.tensor([tile_width, tile_height], dtype=torch.int64, device=means.device)
    minimum = torch.minimum(torch.maximum(minimum, torch.zeros_like(minimum)), bounds)
    maximum = torch.minimum(torch.maximum(maximum, torch.zeros_like(maximum)), bounds)
    counts = torch.clamp(maximum - minimum, min=0).prod(dim=-1)
    return minimum, maximum, counts


def independent_gaussian_projection_oracle_v1(
    *,
    canonical_xyz: torch.Tensor,
    canonical_scaling: torch.Tensor,
    canonical_quaternion: torch.Tensor,
    canonical_opacity: torch.Tensor,
    posed_xyz: torch.Tensor,
    posed_covariance: torch.Tensor,
    opacity: torch.Tensor,
    w2c: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    options: InstrumentationOptions = InstrumentationOptions(),
) -> dict[str, torch.Tensor | int | float]:
    """Independent float64 pinhole/EWA projection without tile/raster calls.

    Image-derived masks are intentionally absent from the interface.  They may
    only be sampled by an offline consumer after this projection is complete.
    """

    n = int(posed_xyz.shape[0])
    for value, shape, name in (
        (canonical_xyz, (n, 3), "canonical_xyz"),
        (canonical_scaling, (n, 3), "canonical_scaling"),
        (canonical_quaternion, (n, 4), "canonical_quaternion"),
        (posed_xyz, (n, 3), "posed_xyz"),
        (posed_covariance, (n, 3, 3), "posed_covariance"),
    ):
        _require_shape(value, shape, name)
    if canonical_opacity.reshape(-1).shape != (n,) or opacity.reshape(-1).shape != (n,):
        raise ValueError("opacity tensors must contain one value per Gaussian")
    _require_shape(w2c, (4, 4), "w2c")
    _require_shape(K, (3, 3), "K")
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")

    device = posed_xyz.device
    canonical_xyz64 = canonical_xyz.detach().to(device=device, dtype=torch.float64)
    canonical_scaling64 = canonical_scaling.detach().to(device=device, dtype=torch.float64)
    canonical_quaternion64 = canonical_quaternion.detach().to(device=device, dtype=torch.float64)
    canonical_opacity64 = canonical_opacity.detach().reshape(-1).to(device=device, dtype=torch.float64)
    means = posed_xyz.detach().to(device=device, dtype=torch.float64)
    covars = posed_covariance.detach().to(device=device, dtype=torch.float64)
    opacities = opacity.detach().reshape(-1).to(device=device, dtype=torch.float64)
    view = w2c.detach().to(device=device, dtype=torch.float64)
    intrinsic = K.detach().to(device=device, dtype=torch.float64)

    canonical_valid = (
        torch.isfinite(canonical_xyz64).all(dim=1)
        & torch.isfinite(canonical_scaling64).all(dim=1)
        & torch.isfinite(canonical_quaternion64).all(dim=1)
        & torch.isfinite(canonical_opacity64)
    )
    posed_valid = torch.isfinite(means).all(dim=1) & torch.isfinite(covars).all(dim=(1, 2))

    rotation, translation = view[:3, :3], view[:3, 3]
    camera_xyz = means @ rotation.T + translation
    covariance_camera = torch.einsum("ij,njk,lk->nil", rotation, covars, rotation)
    camera_finite = torch.isfinite(camera_xyz).all(dim=1)
    depth = camera_xyz[:, 2]
    near_valid = depth >= float(options.near_plane)
    far_valid = depth <= float(options.far_plane)
    camera_valid = camera_finite & near_valid & far_valid

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    z_safe = torch.where(depth.abs() > torch.finfo(torch.float64).tiny, depth, torch.ones_like(depth))
    x_over_z, y_over_z = camera_xyz[:, 0] / z_safe, camera_xyz[:, 1] / z_safe
    tan_fovx, tan_fovy = 0.5 * width / fx, 0.5 * height / fy
    lim_x_pos = (width - cx) / fx + 0.3 * tan_fovx
    lim_x_neg = cx / fx + 0.3 * tan_fovx
    lim_y_pos = (height - cy) / fy + 0.3 * tan_fovy
    lim_y_neg = cy / fy + 0.3 * tan_fovy
    tx = depth * torch.clamp(x_over_z, min=-lim_x_neg, max=lim_x_pos)
    ty = depth * torch.clamp(y_over_z, min=-lim_y_neg, max=lim_y_pos)
    rz, rz2 = z_safe.reciprocal(), z_safe.reciprocal().square()
    jacobian = torch.zeros((n, 2, 3), dtype=torch.float64, device=device)
    jacobian[:, 0, 0] = fx * rz
    jacobian[:, 0, 2] = -fx * tx * rz2
    jacobian[:, 1, 1] = fy * rz
    jacobian[:, 1, 2] = -fy * ty * rz2
    covariance2d_unblurred = torch.einsum("nij,njk,nlk->nil", jacobian, covariance_camera, jacobian)
    covariance2d = covariance2d_unblurred.clone()
    covariance2d[:, 0, 0] += float(options.eps2d)
    covariance2d[:, 1, 1] += float(options.eps2d)
    determinant = covariance2d[:, 0, 0] * covariance2d[:, 1, 1] - covariance2d[:, 0, 1].square()
    covariance_finite = torch.isfinite(covariance2d).all(dim=(1, 2))
    determinant_valid = torch.isfinite(determinant) & (determinant > 0)
    covariance_valid = covariance_finite & determinant_valid
    determinant_safe = torch.where(determinant_valid, determinant, torch.ones_like(determinant))
    conic = torch.stack(
        (covariance2d[:, 1, 1] / determinant_safe, -covariance2d[:, 0, 1] / determinant_safe, covariance2d[:, 0, 0] / determinant_safe),
        dim=-1,
    )
    conic = torch.where(covariance_valid[:, None], conic, torch.zeros_like(conic))
    eigenvalues = torch.linalg.eigvalsh(torch.nan_to_num(covariance2d, nan=0.0, posinf=0.0, neginf=0.0))

    projected_mean = torch.stack((fx * x_over_z + cx, fy * y_over_z + cy), dim=-1)
    opacity_valid = torch.isfinite(opacities) & (opacities >= float(options.alpha_cutoff))
    ratio = torch.where(opacity_valid, opacities / float(options.alpha_cutoff), torch.ones_like(opacities))
    extend = torch.minimum(torch.full_like(opacities, 3.33), torch.sqrt(torch.clamp(2.0 * torch.log(ratio), min=0.0)))
    extend = torch.where(opacity_valid, extend, torch.zeros_like(extend))
    diagonal = torch.clamp(torch.diagonal(covariance2d, dim1=1, dim2=2), min=0.0)
    radii_xy = torch.ceil(extend[:, None] * torch.sqrt(diagonal)).to(torch.int64)
    major_minor = extend[:, None] * torch.sqrt(torch.clamp(eigenvalues.flip(dims=(1,)), min=0.0))
    radius_valid = (radii_xy > float(options.radius_clip)).any(dim=1)
    image_min = projected_mean - radii_xy.to(torch.float64)
    image_max = projected_mean + radii_xy.to(torch.float64)
    image_overlap = (
        (image_max[:, 0] > 0)
        & (image_min[:, 0] < width)
        & (image_max[:, 1] > 0)
        & (image_min[:, 1] < height)
    )
    tile_width, tile_height = math.ceil(width / options.tile_size), math.ceil(height / options.tile_size)
    tile_min, tile_max, tile_count = tile_bboxes(
        projected_mean,
        radii_xy,
        tile_size=options.tile_size,
        tile_width=tile_width,
        tile_height=tile_height,
    )
    tile_valid = tile_count > 0

    stage_masks = torch.stack(
        (
            canonical_valid,
            canonical_valid & posed_valid,
            canonical_valid & posed_valid & camera_valid,
            canonical_valid & posed_valid & camera_valid & covariance_valid & opacity_valid & radius_valid,
            canonical_valid & posed_valid & camera_valid & covariance_valid & opacity_valid & radius_valid & image_overlap,
            canonical_valid & posed_valid & camera_valid & covariance_valid & opacity_valid & radius_valid & image_overlap & tile_valid,
        ),
        dim=1,
    )
    first_rejection_stage = torch.full((n,), 6, dtype=torch.int16, device=device)
    first_rejection_code = torch.full((n,), -1, dtype=torch.int16, device=device)

    def reject(mask: torch.Tensor, stage: int, reason: str) -> None:
        selected = mask & (first_rejection_code < 0)
        first_rejection_stage[selected] = stage
        first_rejection_code[selected] = REJECTION_REASONS.index(reason)

    reject(~canonical_valid, 0, "NONFINITE_CANONICAL_STATE")
    reject(canonical_valid & ~posed_valid, 1, "NONFINITE_POSED_STATE")
    reject(canonical_valid & posed_valid & ~camera_finite, 2, "NONFINITE_CAMERA_POINT")
    reject(canonical_valid & posed_valid & camera_finite & ~near_valid, 2, "BEHIND_NEAR_PLANE")
    reject(canonical_valid & posed_valid & camera_finite & near_valid & ~far_valid, 2, "BEYOND_FAR_PLANE")
    prefix3 = canonical_valid & posed_valid & camera_valid
    reject(prefix3 & ~covariance_finite, 3, "INVALID_COVARIANCE")
    reject(prefix3 & covariance_finite & ~determinant_valid, 3, "NONPOSITIVE_DETERMINANT")
    reject(prefix3 & covariance_valid & ~opacity_valid, 3, "OPACITY_BELOW_CUTOFF")
    reject(prefix3 & covariance_valid & opacity_valid & ~radius_valid, 3, "RADIUS_LE_ZERO")
    prefix4 = prefix3 & covariance_valid & opacity_valid & radius_valid
    reject(prefix4 & ~image_overlap, 4, "NO_IMAGE_OVERLAP")
    reject(prefix4 & image_overlap & ~tile_valid, 5, "ZERO_TILE_OVERLAP")

    return {
        "gaussian_index": torch.arange(n, dtype=torch.int64, device=device),
        "camera_xyz": camera_xyz,
        "depth": depth,
        "projected_mean": projected_mean,
        "covariance_camera": covariance_camera,
        "covariance2d_unblurred": covariance2d_unblurred,
        "covariance2d": covariance2d,
        "conic": conic,
        "determinant": determinant,
        "eigenvalues": eigenvalues,
        "major_minor_radius": major_minor,
        "radii_xy": radii_xy,
        "image_bbox_min": image_min,
        "image_bbox_max": image_max,
        "tile_bbox_min": tile_min,
        "tile_bbox_max": tile_max,
        "tile_count": tile_count,
        "stage_masks_s0_s5": stage_masks,
        "first_rejection_stage": first_rejection_stage,
        "first_rejection_code": first_rejection_code,
        "opacity": opacities,
        "width": width,
        "height": height,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "float_dtype_bits": 64,
    }


def run_backend_projection_debug(
    *,
    means: torch.Tensor,
    covars: torch.Tensor,
    opacities: torch.Tensor,
    w2c: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    options: InstrumentationOptions = InstrumentationOptions(),
) -> dict[str, torch.Tensor | int] | None:
    """Opt-in read-only low-level gsplat projection and tile intersection."""

    if not options.enabled:
        return None
    from gsplat.cuda._wrapper import fully_fused_projection, isect_tiles

    radii, means2d, depths, conics, compensations = fully_fused_projection(
        means=means.detach(),
        covars=_covariance_upper_triangle(covars.detach()),
        quats=None,
        scales=None,
        viewmats=w2c.detach()[None],
        Ks=K.detach()[None],
        width=width,
        height=height,
        eps2d=options.eps2d,
        near_plane=options.near_plane,
        far_plane=options.far_plane,
        radius_clip=options.radius_clip,
        packed=False,
        sparse_grad=False,
        calc_compensations=False,
        camera_model="pinhole",
        opacities=opacities.detach().reshape(-1),
    )
    tile_width, tile_height = math.ceil(width / options.tile_size), math.ceil(height / options.tile_size)
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size=options.tile_size,
        tile_width=tile_width,
        tile_height=tile_height,
        sort=True,
        segmented=False,
        packed=False,
    )
    return {
        "radii": radii,
        "means2d": means2d,
        "depths": depths,
        "conics": conics,
        "compensations": compensations,
        "tiles_per_gauss": tiles_per_gauss,
        "isect_ids": isect_ids,
        "flatten_ids": flatten_ids,
        "tile_width": tile_width,
        "tile_height": tile_height,
    }


def backend_first_rejections(
    independent: Mapping[str, Any], backend: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve S0-S6 first rejection, preserving unexposed backend failures."""

    stages = independent["first_rejection_stage"].detach().clone()
    codes = independent["first_rejection_code"].detach().clone()
    backend_radii = backend["radii"][0]
    emitted = backend["tiles_per_gauss"][0].to(torch.int64)
    independent_s5 = independent["stage_masks_s0_s5"][:, 5]
    backend_projected = (backend_radii > 0).any(dim=-1)
    unexposed = independent_s5 & ~backend_projected
    codes[unexposed] = REJECTION_REASONS.index("BACKEND_UNEXPOSED_REJECTION")
    stages[unexposed] = 3
    omitted = independent_s5 & backend_projected & (emitted == 0)
    codes[omitted] = REJECTION_REASONS.index("INTERSECTION_ALLOCATION_OMISSION")
    stages[omitted] = 6
    return stages, codes, emitted


def stage_statuses(first_stage: int, first_reason_code: int, *, emitted: bool) -> dict[str, str]:
    """Return exhaustive PASS/REJECT statuses with exactly one first rejection."""

    if first_reason_code < 0:
        return {stage: ("PASS" if index <= 6 and (index < 6 or emitted) else "REJECT_NOT_IN_TARGET_PIXEL_TILE") for index, stage in enumerate(STAGE_NAMES)}
    reason = REJECTION_REASONS[first_reason_code]
    return {stage: ("PASS" if index < first_stage else f"REJECT_{reason}") for index, stage in enumerate(STAGE_NAMES)}


def backend_tile_candidates(backend: Mapping[str, Any], y: int, x: int, tile_size: int) -> torch.Tensor:
    tile_width, tile_height = int(backend["tile_width"]), int(backend["tile_height"])
    tile_x, tile_y = x // tile_size, y // tile_size
    if not (0 <= tile_x < tile_width and 0 <= tile_y < tile_height):
        return torch.empty((0,), dtype=torch.int64)
    tile_index = tile_y * tile_width + tile_x
    isect_ids = backend["isect_ids"].detach()
    flatten_ids = backend["flatten_ids"].detach()
    encoded_tiles = torch.bitwise_right_shift(isect_ids, 32)
    target = torch.tensor(tile_index, dtype=encoded_tiles.dtype, device=encoded_tiles.device)
    start = int(torch.searchsorted(encoded_tiles, target, right=False))
    end = int(torch.searchsorted(encoded_tiles, target, right=True))
    return flatten_ids[start:end].to(torch.int64)


def backend_pixel_pipeline(
    backend: Mapping[str, Any],
    opacity: torch.Tensor,
    y: int,
    x: int,
    *,
    tile_size: int,
    alpha_cutoff: float = ALPHA_CUTOFF,
    termination: float = 1.0e-4,
) -> dict[str, Any]:
    """Trace the backend's sorted tile candidates through pixel compositing."""

    candidates = backend_tile_candidates(backend, y, x, tile_size)
    device = backend["means2d"].device
    if not candidates.numel():
        return {"tile_candidates": [], "alpha_eligible": [], "contributed": [], "active_count": 0, "alpha": 0.0}
    means = backend["means2d"][0, candidates].to(torch.float64)
    conics = backend["conics"][0, candidates].to(torch.float64)
    values = opacity.detach().reshape(-1)[candidates].to(torch.float64)
    pixel = torch.tensor([x + .5, y + .5], dtype=torch.float64, device=device)
    delta = means - pixel
    sigma = .5 * (conics[:, 0] * delta[:, 0].square() + conics[:, 2] * delta[:, 1].square()) + conics[:, 1] * delta[:, 0] * delta[:, 1]
    alphas = torch.minimum(values * torch.exp(-sigma), torch.full_like(values, ALPHA_CLAMP_MAX))
    eligible_mask = torch.isfinite(sigma) & (sigma >= 0) & torch.isfinite(alphas) & (alphas >= alpha_cutoff)
    eligible_ids = candidates[eligible_mask]
    contributed: list[int] = []
    transmittance = 1.0
    for local_index in range(candidates.numel()):
        if not bool(eligible_mask[local_index]):
            continue
        alpha = float(alphas[local_index])
        next_transmittance = transmittance * (1.0 - alpha)
        if next_transmittance <= termination:
            break
        contributed.append(int(candidates[local_index]))
        transmittance = next_transmittance
    return {
        "tile_candidates": [int(value) for value in candidates.detach().cpu()],
        "alpha_eligible": [int(value) for value in eligible_ids.detach().cpu()],
        "contributed": contributed,
        "active_count": len(contributed),
        "alpha": 1.0 - transmittance,
    }


def independent_pixel_support(
    projection: Mapping[str, Any],
    pixels_yx: Sequence[tuple[int, int]],
    *,
    alpha_cutoff: float = ALPHA_CUTOFF,
) -> list[dict[str, Any]]:
    """Evaluate target-independent projection at already-selected offline pixels."""

    means = projection["projected_mean"]
    conics = projection["conic"]
    depths = projection["depth"]
    opacities = projection["opacity"]
    valid = projection["stage_masks_s0_s5"][:, 3]
    result: list[dict[str, Any]] = []
    for y, x in pixels_yx:
        pixel = torch.tensor([x + 0.5, y + 0.5], dtype=torch.float64, device=means.device)
        delta = means - pixel
        sigma = 0.5 * (conics[:, 0] * delta[:, 0].square() + conics[:, 2] * delta[:, 1].square()) + conics[:, 1] * delta[:, 0] * delta[:, 1]
        alpha = torch.minimum(opacities * torch.exp(-sigma), torch.full_like(opacities, ALPHA_CLAMP_MAX))
        eligible_domain = valid & torch.isfinite(sigma) & (sigma >= 0) & torch.isfinite(alpha)
        contributor = eligible_domain & (alpha >= alpha_cutoff)
        count_001 = int((eligible_domain & (alpha >= .01)).sum())
        count_01 = int((eligible_domain & (alpha >= .1)).sum())
        contributor_ids = torch.nonzero(contributor, as_tuple=False).reshape(-1)
        all_alpha = torch.clamp(alpha[eligible_domain], min=0.0, max=ALPHA_CLAMP_MAX)
        accumulated = 1.0 - torch.exp(torch.log1p(-all_alpha).sum()) if all_alpha.numel() else alpha.new_tensor(0.0)
        center_distance = torch.linalg.vector_norm(delta, dim=1)
        mahalanobis = torch.where(eligible_domain, 2.0 * sigma, torch.full_like(sigma, torch.inf))
        nearest_center = torch.topk(center_distance, k=min(10, center_distance.numel()), largest=False).indices
        nearest_mahalanobis = torch.topk(mahalanobis, k=min(10, mahalanobis.numel()), largest=False).indices
        result.append({
            "y": int(y),
            "x": int(x),
            "nearest_projected_center_distance": float(center_distance.min().detach().cpu()),
            "nearest_ellipse_mahalanobis_distance": float(torch.sqrt(torch.clamp(mahalanobis.min(), min=0)).detach().cpu()),
            "maximum_theoretical_single_alpha": float(torch.where(eligible_domain, alpha, torch.zeros_like(alpha)).max().detach().cpu()),
            "pre_tile_contributor_count": int(contributor.sum()),
            "alpha_ge_0_01_count": count_001,
            "alpha_ge_0_1_count": count_01,
            "theoretical_accumulated_alpha": float(accumulated.detach().cpu()),
            "nearest_10_gaussian_indices": [int(value) for value in nearest_center.detach().cpu()],
            "nearest_10_mahalanobis_indices": [int(value) for value in nearest_mahalanobis.detach().cpu()],
            "pre_tile_contributor_indices": [int(value) for value in contributor_ids.detach().cpu()],
        })
    return result


def support_funnel_counts(stage_masks: torch.Tensor, emitted: torch.Tensor | None = None) -> list[int]:
    if stage_masks.ndim != 2 or stage_masks.shape[1] != 6 or stage_masks.dtype != torch.bool:
        raise ValueError("stage_masks must be boolean [N,6]")
    counts = [int(stage_masks[:, index].sum()) for index in range(6)]
    counts.append(int((stage_masks[:, 5] & (emitted > 0)).sum()) if emitted is not None else counts[-1])
    if any(left < right for left, right in zip(counts, counts[1:])):
        raise AssertionError(f"support funnel is not monotonic: {counts}")
    return counts


def same_index_displacement(
    base_indices: torch.Tensor,
    source_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    source_screen: torch.Tensor,
    target_screen: torch.Tensor,
) -> dict[str, torch.Tensor]:
    ids = base_indices.to(torch.int64)
    if ids.ndim != 1 or torch.any(ids < 0) or torch.any(ids >= source_xyz.shape[0]) or source_xyz.shape != target_xyz.shape:
        raise ValueError("invalid same-index match inputs")
    return {
        "gaussian_index": ids,
        "canonical_or_posed_displacement": target_xyz[ids] - source_xyz[ids],
        "screen_displacement": target_screen[ids] - source_screen[ids],
    }


def counterfactual_alpha(
    projection: Mapping[str, Any],
    gaussian_indices: torch.Tensor,
    pixels_yx: Sequence[tuple[int, int]],
) -> torch.Tensor:
    """Pure, no-tile, no-cutoff float64 alpha for a fixed rejected subset."""

    ids = gaussian_indices.detach().to(device=projection["projected_mean"].device, dtype=torch.int64)
    means = projection["projected_mean"][ids]
    conics = projection["conic"][ids]
    opacities = projection["opacity"][ids]
    rows = []
    for y, x in pixels_yx:
        pixel = torch.tensor([x + .5, y + .5], dtype=torch.float64, device=means.device)
        delta = means - pixel
        sigma = .5 * (conics[:, 0] * delta[:, 0].square() + conics[:, 2] * delta[:, 1].square()) + conics[:, 1] * delta[:, 0] * delta[:, 1]
        values = torch.where(sigma >= 0, torch.minimum(opacities * torch.exp(-sigma), torch.full_like(opacities, ALPHA_CLAMP_MAX)), torch.zeros_like(opacities))
        rows.append(1.0 - torch.prod(1.0 - torch.clamp(values, 0.0, ALPHA_CLAMP_MAX)))
    return torch.stack(rows) if rows else torch.empty((0,), dtype=torch.float64, device=means.device)


def projection_agreement(independent: Mapping[str, Any], backend: Mapping[str, Any]) -> dict[str, float | bool | int]:
    backend_radii = backend["radii"][0].detach().to(torch.float64)
    accepted = (backend_radii > 0).any(dim=-1)
    count = int(accepted.sum())
    if not count:
        return {"accepted_count": 0, "pass": False}
    mean_diff = (independent["projected_mean"][accepted] - backend["means2d"][0, accepted].to(torch.float64)).abs().reshape(-1)
    depth = independent["depth"][accepted]
    depth_diff = (depth - backend["depths"][0, accepted].to(torch.float64)).abs() / torch.clamp(depth.abs(), min=1e-12)
    radius_diff = (independent["radii_xy"][accepted].to(torch.float64) - backend_radii[accepted]).abs().reshape(-1)
    radius_rel = radius_diff / torch.clamp(backend_radii[accepted].reshape(-1).abs(), min=1.0)
    conic_diff = (independent["conic"][accepted] - backend["conics"][0, accepted].to(torch.float64)).abs().reshape(-1)
    backend_conic = backend["conics"][0, accepted].to(torch.float64)
    inverse_determinant = backend_conic[:, 0] * backend_conic[:, 2] - backend_conic[:, 1].square()
    backend_covariance = torch.stack(
        (
            backend_conic[:, 2] / inverse_determinant,
            -backend_conic[:, 1] / inverse_determinant,
            -backend_conic[:, 1] / inverse_determinant,
            backend_conic[:, 0] / inverse_determinant,
        ),
        dim=-1,
    ).reshape(-1, 2, 2)
    covariance_diff = (independent["covariance2d"][accepted] - backend_covariance).abs().reshape(-1)

    def q(value: torch.Tensor, fraction: float) -> float:
        return float(torch.quantile(value, torch.tensor(fraction, dtype=value.dtype, device=value.device)).detach().cpu())

    result = {
        "accepted_count": count,
        "projected_mean_abs_median": q(mean_diff, .5),
        "projected_mean_abs_p99": q(mean_diff, .99),
        "depth_relative_p99": q(depth_diff, .99),
        "radius_abs_p99": q(radius_diff, .99),
        "radius_relative_p99": q(radius_rel, .99),
        "conic_abs_p99": q(conic_diff, .99),
        "covariance_abs_p99": q(covariance_diff, .99),
    }
    result["pass"] = bool(
        result["projected_mean_abs_median"] <= .01
        and result["projected_mean_abs_p99"] <= .10
        and result["depth_relative_p99"] <= 1e-5
        and (result["radius_abs_p99"] <= 1.0 or result["radius_relative_p99"] <= .02)
    )
    return result
