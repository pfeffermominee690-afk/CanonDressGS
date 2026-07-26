from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from utils.surface_alignment import (
    aggregate_gaussians_to_surface_anchors,
    build_surface_aware_anchor_offset_target,
)


_TARGET_KEYS = (
    "anchor_xyz",
    "delta_xyz",
    "delta_scaling",
    "delta_opacity",
    "valid_mask",
    "cloth_region_weight",
)


def aggregate_gaussians_to_anchors(
    gaussians: dict[str, torch.Tensor],
    anchor_xyz: torch.Tensor,
    k: int = 8,
    max_distance: float | None = None,
    temperature: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Aggregate nearby LHM Gaussian raw parameters at canonical anchors."""

    xyz = _require_matrix(gaussians, "xyz", 3)
    scale = _require_matrix(gaussians, "scale", 3, rows=xyz.shape[0])
    opacity = _require_matrix(gaussians, "opacity", 1, rows=xyz.shape[0])
    if not isinstance(anchor_xyz, torch.Tensor) or anchor_xyz.ndim != 2 or anchor_xyz.shape[1] != 3:
        raise ValueError("anchor_xyz must have shape [A, 3]")
    if anchor_xyz.shape[0] == 0 or not torch.isfinite(anchor_xyz).all():
        raise ValueError("anchor_xyz must be non-empty and finite")
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError(f"k must be a positive int, got {k!r}")
    if k > xyz.shape[0]:
        raise ValueError(f"k={k} exceeds Gaussian count M={xyz.shape[0]}")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if max_distance is not None and max_distance < 0:
        raise ValueError("max_distance must be non-negative")

    anchors = anchor_xyz.to(device=xyz.device, dtype=xyz.dtype)
    distances = torch.cdist(anchors, xyz)
    nearest_distances, nearest_indices = torch.topk(
        distances,
        k=k,
        dim=1,
        largest=False,
        sorted=True,
    )
    weights = torch.softmax(-nearest_distances / temperature, dim=1)

    def aggregate(values: torch.Tensor) -> torch.Tensor:
        gathered = values[nearest_indices]
        return (gathered * weights.unsqueeze(-1)).sum(dim=1)

    nearest_distance = nearest_distances[:, :1]
    if max_distance is None:
        valid_mask = torch.ones_like(nearest_distance)
    else:
        valid_mask = (nearest_distance <= max_distance).to(dtype=xyz.dtype)
    return {
        "anchor_xyz": anchors,
        "aggregated_xyz": aggregate(xyz),
        "aggregated_scale": aggregate(scale),
        "aggregated_opacity": aggregate(opacity),
        "nearest_distance": nearest_distance,
        "valid_mask": valid_mask,
    }


def build_anchor_offset_target(
    base_anchor_params: dict[str, torch.Tensor],
    dressed_anchor_params: dict[str, torch.Tensor],
    cloth_region_weight: torch.Tensor | None = None,
    cloth_region_threshold: float = 0.005,
) -> dict[str, torch.Tensor]:
    """Build raw anchor offsets and a soft clothing support estimate."""

    anchor_xyz = _require_matrix(base_anchor_params, "anchor_xyz", 3)
    base_scale = _require_matrix(
        base_anchor_params,
        "scale",
        3,
        rows=anchor_xyz.shape[0],
    ).to(device=anchor_xyz.device, dtype=anchor_xyz.dtype)
    base_opacity = _require_matrix(
        base_anchor_params,
        "opacity",
        1,
        rows=anchor_xyz.shape[0],
    ).to(device=anchor_xyz.device, dtype=anchor_xyz.dtype)
    aggregated_xyz = _require_matrix(
        dressed_anchor_params,
        "aggregated_xyz",
        3,
        rows=anchor_xyz.shape[0],
    ).to(device=anchor_xyz.device, dtype=anchor_xyz.dtype)
    aggregated_scale = _require_matrix(
        dressed_anchor_params,
        "aggregated_scale",
        3,
        rows=anchor_xyz.shape[0],
    ).to(device=anchor_xyz.device, dtype=anchor_xyz.dtype)
    aggregated_opacity = _require_matrix(
        dressed_anchor_params,
        "aggregated_opacity",
        1,
        rows=anchor_xyz.shape[0],
    ).to(device=anchor_xyz.device, dtype=anchor_xyz.dtype)
    valid_mask = _require_matrix(
        dressed_anchor_params,
        "valid_mask",
        1,
        rows=anchor_xyz.shape[0],
    ).to(device=anchor_xyz.device, dtype=anchor_xyz.dtype)
    if torch.any(valid_mask < 0) or torch.any(valid_mask > 1):
        raise ValueError("valid_mask values must be in [0, 1]")
    if cloth_region_threshold <= 0:
        raise ValueError("cloth_region_threshold must be positive")

    delta_xyz = aggregated_xyz - anchor_xyz
    delta_scaling = aggregated_scale - base_scale
    delta_opacity = aggregated_opacity - base_opacity
    if cloth_region_weight is None:
        change = torch.linalg.vector_norm(delta_xyz, dim=-1, keepdim=True)
        change = change + delta_scaling.abs().mean(dim=-1, keepdim=True)
        excess = torch.relu(change - cloth_region_threshold)
        region = 1 - torch.exp(-excess / cloth_region_threshold)
    else:
        region = _normalize_anchor_weight(
            cloth_region_weight,
            anchor_xyz.shape[0],
            anchor_xyz,
            "cloth_region_weight",
        )
        if torch.any(region < 0) or torch.any(region > 1):
            raise ValueError("cloth_region_weight values must be in [0, 1]")
    region = region * valid_mask

    target = {
        "anchor_xyz": anchor_xyz.clone(),
        "delta_xyz": delta_xyz,
        "delta_scaling": delta_scaling,
        "delta_opacity": delta_opacity,
        "valid_mask": valid_mask,
        "cloth_region_weight": region.clamp(0, 1),
    }
    _validate_target(target)
    return target


def save_anchor_offset_target(
    target: dict[str, torch.Tensor],
    out_path: str | Path,
    metadata: dict | None = None,
) -> None:
    """Save a versioned CPU tensor target and optional provenance metadata."""

    _validate_target(target)
    if metadata is not None and not isinstance(metadata, dict):
        raise TypeError("metadata must be a dict or None")
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "target": {key: value.detach().cpu() for key, value in target.items()},
        "metadata": {} if metadata is None else dict(metadata),
        "version": 1,
    }
    torch.save(payload, path)


def load_anchor_offset_target(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load versioned targets and legacy plain tensor dictionaries."""

    target_path = Path(path)
    if not target_path.is_file():
        raise FileNotFoundError(f"anchor target does not exist: {target_path}")
    loaded = torch.load(target_path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise TypeError("anchor target file must contain a dictionary")
    if "target" in loaded:
        target = loaded["target"]
        metadata = loaded.get("metadata", {})
        version = loaded.get("version", 1)
    else:
        target = loaded
        metadata = {}
        version = 0
    if not isinstance(target, dict):
        raise TypeError("anchor target payload must be a tensor dictionary")
    if not isinstance(metadata, dict):
        raise TypeError("anchor target metadata must be a dictionary")
    moved = {
        key: value.to(device=device) if isinstance(value, torch.Tensor) else value
        for key, value in target.items()
    }
    _validate_target(moved)
    return {"target": moved, "metadata": metadata, "version": version}


def _require_matrix(
    values: dict[str, torch.Tensor],
    key: str,
    columns: int,
    rows: int | None = None,
) -> torch.Tensor:
    if key not in values or not isinstance(values[key], torch.Tensor):
        raise KeyError(f"missing tensor field {key!r}")
    value = values[key]
    if value.ndim != 2 or value.shape[1] != columns:
        raise ValueError(f"{key} must have shape [N, {columns}], got {tuple(value.shape)}")
    if rows is not None and value.shape[0] != rows:
        raise ValueError(f"{key} has {value.shape[0]} rows, expected {rows}")
    if value.shape[0] == 0 or not torch.isfinite(value).all():
        raise ValueError(f"{key} must be non-empty and finite")
    return value


def _normalize_anchor_weight(
    weight: torch.Tensor,
    num_anchors: int,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if weight.shape == (num_anchors,):
        weight = weight[:, None]
    if weight.shape != (num_anchors, 1):
        raise ValueError(f"{name} must have shape [A] or [A, 1]")
    weight = weight.to(device=reference.device, dtype=reference.dtype)
    if not torch.isfinite(weight).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return weight


def _validate_target(target: dict[str, torch.Tensor]) -> None:
    if not isinstance(target, dict):
        raise TypeError("target must be a tensor dictionary")
    missing = [key for key in _TARGET_KEYS if key not in target]
    if missing:
        raise KeyError(f"anchor target is missing required keys: {missing}")
    anchor_xyz = _require_matrix(target, "anchor_xyz", 3)
    num_anchors = anchor_xyz.shape[0]
    _require_matrix(target, "delta_xyz", 3, rows=num_anchors)
    _require_matrix(target, "delta_scaling", 3, rows=num_anchors)
    _require_matrix(target, "delta_opacity", 1, rows=num_anchors)
    valid = _require_matrix(target, "valid_mask", 1, rows=num_anchors)
    region = _require_matrix(target, "cloth_region_weight", 1, rows=num_anchors)
    if torch.any(valid < 0) or torch.any(valid > 1):
        raise ValueError("valid_mask values must be in [0, 1]")
    if torch.any(region < 0) or torch.any(region > 1):
        raise ValueError("cloth_region_weight values must be in [0, 1]")
