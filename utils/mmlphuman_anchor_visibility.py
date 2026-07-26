from __future__ import annotations

import inspect
from importlib import metadata
from typing import Any, Callable

import torch
import torch.nn.functional as F


def describe_gsplat_depth_api(
    rasterization_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Report whether the active gsplat rasterizer exposes official depth modes."""

    if rasterization_fn is None:
        from scene import gaussian_model as gaussian_model_module

        rasterization_fn = gaussian_model_module.rasterization
        try:
            version = metadata.version("gsplat")
        except metadata.PackageNotFoundError:
            version = "unknown"
    else:
        version = "injected-test-rasterizer"
    try:
        signature = inspect.signature(rasterization_fn)
    except (TypeError, ValueError) as error:
        return {
            "supported": False,
            "version": version,
            "signature": "unavailable",
            "reason": f"could not inspect rasterization signature: {error}",
        }
    supported = "render_mode" in signature.parameters
    return {
        "supported": supported,
        "version": version,
        "signature": str(signature),
        "reason": (
            "official render_mode argument is available"
            if supported
            else "rasterization signature has no render_mode argument"
        ),
    }


def render_mmlphuman_expected_depth(
    base_model: Any,
    camera: dict[str, Any],
    background: torch.Tensor,
    canonical_overrides: dict[str, torch.Tensor] | None = None,
    scaling_modifier: float = 1.0,
    override_color: torch.Tensor | None = None,
    *,
    rasterization_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Render official gsplat expected camera-z without changing the RGB path.

    The function uses ``render_mode="ED"``. Unlike accumulated ``D``, ``ED`` is
    normalized by accumulated opacity and is therefore in the same camera-space
    length unit as an anchor's projected z coordinate. Unsupported gsplat builds
    fail explicitly; no approximate depth fallback is used.
    """

    if not isinstance(camera, dict):
        raise TypeError("camera must be a dictionary")
    required_camera = ("w2c", "K", "width", "height")
    missing = [key for key in required_camera if key not in camera]
    if missing:
        raise KeyError(f"camera is missing required fields: {missing}")
    if not isinstance(background, torch.Tensor) or background.shape != (3,):
        raise ValueError("background must be a Tensor[3]")
    if not torch.is_floating_point(background) or not torch.isfinite(background).all():
        raise ValueError("background must be a finite floating-point Tensor[3]")
    if not isinstance(scaling_modifier, (int, float)) or isinstance(
        scaling_modifier, bool
    ):
        raise TypeError("scaling_modifier must be a number")
    if scaling_modifier <= 0:
        raise ValueError("scaling_modifier must be positive")

    api = describe_gsplat_depth_api(rasterization_fn)
    if not api["supported"]:
        raise RuntimeError(
            "the active gsplat rasterization API cannot render verified depth: "
            f"version={api['version']}, signature={api['signature']}, "
            f"reason={api['reason']}"
        )
    if rasterization_fn is None:
        from scene import gaussian_model as gaussian_model_module

        rasterization_fn = gaussian_model_module.rasterization

    cache_object = getattr(base_model, "cache_dict", None)
    if not isinstance(cache_object, dict):
        raise TypeError("base_model.cache_dict must be a dictionary")
    cache_snapshot = dict(cache_object)
    pose_state_names = ("_smpl_poses", "smpl_poses_cuda", "_Rh", "_Th")
    pose_state = {
        name: getattr(base_model, name, None) for name in pose_state_names
    }

    try:
        validator = getattr(base_model, "_validate_canonical_overrides", None)
        if callable(validator):
            validator(canonical_overrides)
        base_model.compute_sh(canonical_overrides)
        covariance = base_model.get_covariance(
            scaling_modifier, canonical_overrides
        )
        if override_color is None:
            camera_position = torch.linalg.inv_ex(camera["w2c"])[0][:3, 3]
            override_color = base_model.get_color(
                camera_position, canonical_overrides
            )
        try:
            rendered, alpha, info = rasterization_fn(
                means=base_model.compute_xyz(canonical_overrides),
                quats=None,
                scales=None,
                opacities=base_model.compute_opacity(canonical_overrides),
                colors=override_color,
                viewmats=camera["w2c"][None],
                Ks=camera["K"][None],
                width=camera["width"],
                height=camera["height"],
                packed=False,
                near_plane=0.1,
                backgrounds=background[None],
                covars=covariance,
                render_mode="ED",
            )
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "gsplat exposes render_mode but rejected official ED depth rendering; "
                f"version={api['version']}, signature={api['signature']}: {error}"
            ) from error
    finally:
        cache_object.clear()
        cache_object.update(cache_snapshot)
        base_model.cache_dict = cache_object

    for name, value in pose_state.items():
        if getattr(base_model, name, None) is not value:
            raise RuntimeError(f"depth rendering changed base pose state: {name}")
    if not isinstance(rendered, torch.Tensor) or rendered.ndim != 4:
        raise RuntimeError("gsplat ED output must be a Tensor[1,H,W,1]")
    if rendered.shape[0] != 1 or rendered.shape[-1] != 1:
        raise RuntimeError(
            f"gsplat ED output has unexpected shape {tuple(rendered.shape)}"
        )
    if not isinstance(alpha, torch.Tensor) or alpha.ndim != 4:
        raise RuntimeError("gsplat alpha output must be a Tensor[1,H,W,1]")
    if alpha.shape[0] != 1 or alpha.shape[-1] != 1:
        raise RuntimeError(
            f"gsplat alpha output has unexpected shape {tuple(alpha.shape)}"
        )
    return {
        "depth": rendered[0, ..., 0],
        "alpha": alpha[0, ..., 0],
        "info": info,
        "render_mode": "ED",
        "gsplat_version": api["version"],
        "gsplat_signature": api["signature"],
        "base_pose_cache_state_restored": True,
    }


def sample_image_map_at_grid(
    image_map: torch.Tensor,
    sampling_grid: torch.Tensor,
    *,
    align_corners: bool,
) -> torch.Tensor:
    """Bilinearly sample ``Tensor[K,C,H,W]`` at ``Tensor[K,A,2]`` grids."""

    if not isinstance(image_map, torch.Tensor) or image_map.ndim != 4:
        raise ValueError("image_map must be a Tensor[K,C,H,W]")
    if not torch.is_floating_point(image_map):
        raise TypeError("image_map must have a floating-point dtype")
    if not isinstance(sampling_grid, torch.Tensor) or sampling_grid.ndim != 3:
        raise ValueError("sampling_grid must be a Tensor[K,A,2]")
    if sampling_grid.shape[-1] != 2:
        raise ValueError("sampling_grid must have shape [K,A,2]")
    if image_map.shape[0] != sampling_grid.shape[0]:
        raise ValueError("image_map and sampling_grid must have the same K")
    if image_map.device != sampling_grid.device:
        raise ValueError("image_map and sampling_grid must be on the same device")
    if image_map.dtype != sampling_grid.dtype:
        raise ValueError("image_map and sampling_grid must have the same dtype")
    sampled = F.grid_sample(
        image_map,
        sampling_grid.unsqueeze(2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=align_corners,
    )
    return sampled.squeeze(-1).permute(0, 2, 1)


def compute_anchor_depth_visibility(
    anchor_depth: torch.Tensor,
    surface_depth_maps: torch.Tensor | None,
    sampling_grid: torch.Tensor,
    positive_depth_mask: torch.Tensor,
    in_frame_mask: torch.Tensor,
    foreground_mask_hit: torch.Tensor,
    *,
    abs_tolerance: float,
    rel_tolerance: float,
    align_corners: bool,
    surface_alpha_maps: torch.Tensor | None = None,
    min_surface_alpha: float = 1e-4,
) -> dict[str, torch.Tensor]:
    """Compare anchor camera-z against bilinearly sampled gsplat ED depth."""

    if surface_depth_maps is None:
        raise RuntimeError(
            "surface_depth_maps is required for depth visibility; refusing to "
            "silently treat every anchor as visible"
        )
    if not isinstance(abs_tolerance, (int, float)) or isinstance(abs_tolerance, bool):
        raise TypeError("abs_tolerance must be a number")
    if not isinstance(rel_tolerance, (int, float)) or isinstance(rel_tolerance, bool):
        raise TypeError("rel_tolerance must be a number")
    if abs_tolerance < 0 or rel_tolerance < 0:
        raise ValueError("depth tolerances must be non-negative")
    if not isinstance(min_surface_alpha, (int, float)) or isinstance(
        min_surface_alpha, bool
    ):
        raise TypeError("min_surface_alpha must be a number")
    if not 0 <= min_surface_alpha <= 1:
        raise ValueError("min_surface_alpha must be in [0,1]")
    if not isinstance(anchor_depth, torch.Tensor) or anchor_depth.ndim != 3:
        raise ValueError("anchor_depth must be a Tensor[K,A,1]")
    expected_shape = anchor_depth.shape
    if expected_shape[-1] != 1:
        raise ValueError("anchor_depth must have shape [K,A,1]")
    for name, value in (
        ("positive_depth_mask", positive_depth_mask),
        ("in_frame_mask", in_frame_mask),
        ("foreground_mask_hit", foreground_mask_hit),
    ):
        if not isinstance(value, torch.Tensor) or value.shape != expected_shape:
            raise ValueError(f"{name} must have shape {tuple(expected_shape)}")
    if not isinstance(surface_depth_maps, torch.Tensor):
        raise TypeError("surface_depth_maps must be a torch.Tensor")
    if surface_depth_maps.ndim != 4 or surface_depth_maps.shape[1] != 1:
        raise ValueError("surface_depth_maps must be a Tensor[K,1,H,W]")
    if surface_depth_maps.shape[0] != anchor_depth.shape[0]:
        raise ValueError("surface_depth_maps must have the same K as anchor_depth")
    if surface_depth_maps.device != anchor_depth.device:
        raise ValueError("surface_depth_maps must match anchor_depth device")
    if surface_depth_maps.dtype != anchor_depth.dtype:
        raise ValueError("surface_depth_maps must match anchor_depth dtype")
    if sampling_grid.shape[:2] != anchor_depth.shape[:2]:
        raise ValueError("sampling_grid must match anchor_depth K and A")

    sampled_surface_depth = sample_image_map_at_grid(
        surface_depth_maps, sampling_grid, align_corners=align_corners
    )
    if surface_alpha_maps is None:
        sampled_surface_alpha = torch.ones_like(sampled_surface_depth)
    else:
        if (
            not isinstance(surface_alpha_maps, torch.Tensor)
            or surface_alpha_maps.shape != surface_depth_maps.shape
        ):
            raise ValueError(
                "surface_alpha_maps must have the same [K,1,H,W] shape as depth"
            )
        if (
            surface_alpha_maps.device != anchor_depth.device
            or surface_alpha_maps.dtype != anchor_depth.dtype
        ):
            raise ValueError("surface_alpha_maps must match anchor depth device/dtype")
        sampled_surface_alpha = sample_image_map_at_grid(
            surface_alpha_maps, sampling_grid, align_corners=align_corners
        )

    anchor_finite = torch.isfinite(anchor_depth)
    surface_finite = torch.isfinite(sampled_surface_depth)
    alpha_finite = torch.isfinite(sampled_surface_alpha)
    depth_valid = (
        anchor_finite
        & surface_finite
        & (sampled_surface_depth > 0)
        & alpha_finite
        & (sampled_surface_alpha >= min_surface_alpha)
    )
    residual = torch.abs(anchor_depth - sampled_surface_depth)
    tolerance = torch.maximum(
        torch.full_like(sampled_surface_depth, float(abs_tolerance)),
        sampled_surface_depth.abs() * float(rel_tolerance),
    )
    geometry_valid = (
        positive_depth_mask.bool()
        & in_frame_mask.bool()
        & foreground_mask_hit.bool()
        & depth_valid
    )
    depth_visible = geometry_valid & (residual <= tolerance)
    epsilon = torch.finfo(anchor_depth.dtype).eps
    safe_tolerance = torch.where(
        torch.isfinite(tolerance),
        tolerance.clamp_min(epsilon),
        torch.full_like(tolerance, epsilon),
    )
    safe_residual = torch.where(
        torch.isfinite(residual), residual, torch.full_like(residual, torch.inf)
    )
    depth_confidence = (
        torch.exp(-safe_residual / safe_tolerance)
        * depth_visible.to(dtype=anchor_depth.dtype)
    )
    return {
        "sampled_surface_depth": sampled_surface_depth,
        "sampled_surface_alpha": sampled_surface_alpha,
        "depth_residual": residual,
        "depth_tolerance": tolerance,
        "depth_valid_mask": depth_valid,
        "depth_visible_mask": depth_visible,
        "depth_confidence": depth_confidence,
    }
