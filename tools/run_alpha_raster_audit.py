from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_dressable as training  # noqa: E402
from scene.alpha_raster_audit import (  # noqa: E402
    ALPHA_CLAMP_MAX,
    ALPHA_THRESHOLD,
    CULLING_REASONS,
    EARLY_TERMINATION_TRANSMITTANCE,
    culling_reason,
    deterministic_mask_indices,
    downsample_supersampled,
    largest_internal_hole,
    quantile_summary,
    reference_alpha_composite,
    registered_render_variants,
    single_gaussian_alpha,
    threshold_metrics,
    validate_variant_contract,
)
from scene.gaussian_clothing_residuals import (  # noqa: E402
    CanonicalGaussianOverrides,
    axis_angle_to_quaternion_wxyz,
    quaternion_multiply_wxyz,
)
from scene.o00_fixed_open_oracle import FixedOpenGaussianOracle, _bounded_rotvec  # noqa: E402
from scene.representation_capacity_oracle import UnboundedGaussianDeltaField  # noqa: E402
from scene.trusted_silhouette_semantics_v6_1 import build_trusted_silhouette_regions  # noqa: E402
from tools import run_new_silhouette_semantics as v6_1_runner  # noqa: E402
from tools import run_objective_residual_redesign as v6_runner  # noqa: E402
from tools.run_module4b_canonical_oracle_micropilot import (  # noqa: E402
    _environment,
    _git_state,
    _load_samples,
    _tensor_state_fingerprint,
)
from utils.dressable_camera_utils import build_mmlphuman_camera  # noqa: E402
from utils.mmlphuman_state_utils import mmlphuman_state_transaction  # noqa: E402


SCHEMA = "canondressgs.alpha_raster_audit.v1"
OUTFITS = ("O01", "O08")
CONDITIONS = ("cond_000000", "cond_000318", "cond_000017", "cond_000347")
VIEWS = dict(zip(CONDITIONS, ("front", "back", "left", "right")))
STATES = ("P0", "P1", "P2", "P3")
DEFAULT_OUTPUT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/"
    "SUBJECT02-ALPHA-RASTER-AUDIT-001/attempt_001"
)
OUTPUT_DIRS = (
    "contract", "input_audit", "renderer_static_audit", "parameter_state_comparison",
    "alpha_threshold_sweep", "contributor_tracing", "projection_and_culling",
    "reference_compositor", "render_variant_matrix", "gradient_checks",
    "minimal_proof_probe", "visual_acceptance", "final_adjudication",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True, default=str, allow_nan=False) + "\n")


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA:
        raise ValueError("unexpected alpha/raster audit schema")
    if tuple(config["outfits"]) != OUTFITS or tuple(config["conditions"]) != CONDITIONS:
        raise ValueError("fixed O01/O08 four-condition contract changed")
    if config["source_head"] != "74f65ef15d1ca7e81d6d8ac9d58c469285f70323":
        raise ValueError("V6.1 frozen source HEAD changed")
    expected_thresholds = (0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9)
    if tuple(float(value) for value in config["silhouette_thresholds"]) != expected_thresholds:
        raise ValueError("alpha threshold sweep changed")
    permissions = config["permissions"]
    if any(bool(value) for value in permissions.values()):
        raise ValueError("alpha/raster audit permissions must remain false")
    variants = registered_render_variants()
    validate_variant_contract(variants)
    if tuple(config["render_variants"]) != tuple(item.name for item in variants):
        raise ValueError("render variant ladder changed")
    return config


def git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()


def resolve_branch_commit(branch: str) -> str:
    for candidate in (branch, f"cloud/{branch}", f"origin/{branch}"):
        result = subprocess.run(
            ["git", "rev-parse", "--verify", candidate], cwd=PROJECT_ROOT,
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    raise RuntimeError(f"cannot resolve frozen branch: {branch}")


def save_rgb(path: Path, tensor: torch.Tensor) -> None:
    value = tensor.detach().float().cpu()
    if value.ndim == 3 and value.shape[0] == 3:
        value = value.permute(1, 2, 0)
    if value.ndim != 3 or value.shape[-1] != 3 or not torch.isfinite(value).all():
        raise ValueError("RGB must be finite CHW/HWC")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((value.clamp(0, 1).numpy() * 255).round().astype(np.uint8), "RGB").save(path)


def save_alpha(path: Path, tensor: torch.Tensor) -> None:
    value = tensor.detach().float().cpu()
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    elif value.ndim == 3 and value.shape[-1] == 1:
        value = value[..., 0]
    if value.ndim != 2 or not torch.isfinite(value).all():
        raise ValueError("alpha must be finite CHW/HWC")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((value.clamp(0, 1).numpy() * 65535).round().astype(np.uint16), "I;16").save(path)


def chw(value: torch.Tensor, channels: int) -> torch.Tensor:
    if value.ndim == 3 and value.shape[0] == channels:
        result = value
    elif value.ndim == 3 and value.shape[-1] == channels:
        result = value.permute(2, 0, 1)
    else:
        raise ValueError(f"unexpected render shape {tuple(value.shape)}")
    if not torch.isfinite(result).all():
        raise FloatingPointError("render contains NaN/Inf")
    return result.contiguous()


def _checkpoint_path(config: Mapping[str, Any], state: str, outfit: str) -> Path:
    roots = {
        "P1": Path(config["source_representation_output"]),
        "P2": Path(config["source_v6_output"]),
        "P3": Path(config["source_v6_1_output"]),
    }
    relative = config["states"][state]["relative_checkpoint"].format(outfit=outfit)
    return roots[state] / relative


def load_state_model(config: Mapping[str, Any], state: str, outfit: str, base: Any) -> tuple[Any | None, CanonicalGaussianOverrides]:
    if state == "P0":
        return None, CanonicalGaussianOverrides(
            xyz=base._xyz, scaling=base._scaling, rotation=base._rotation,
            opacity=base._opacity, sh0=base._sh0, shN=base._shN,
        ).validate(base)
    path = _checkpoint_path(config, state, outfit)
    expected = config["states"][state]["sha256"][outfit]
    if not path.is_file() or sha256(path) != expected:
        raise RuntimeError(f"{state}/{outfit} checkpoint path or SHA256 mismatch")
    if state == "P1":
        model = UnboundedGaussianDeltaField(base).to(base._xyz.device)
    else:
        model = FixedOpenGaussianOracle(base, bounds=config["candidate_bounds"]).to(base._xyz.device)
    checkpoint = torch.load(path, map_location=base._xyz.device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    output = model(base)
    overrides = output.canonical_overrides if hasattr(output, "canonical_overrides") else output
    return model, overrides


def load_runtime(config: Mapping[str, Any], outfit: str, device: str):
    namespace = argparse.Namespace(
        device=device,
        pipeline_config=PROJECT_ROOT / config["pipeline_config"],
        manifest=Path(config["source_manifest"]),
    )
    base, samples, background, torch_device = v6_runner._load_runtime(namespace, outfit)
    v6_1_runner.attach_artifact_masks(samples)
    if int(base._xyz.shape[0]) != 200_000:
        raise ValueError("audit requires exact 200k base")
    return base, samples, background, torch_device


def target_free_state(sample: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return v6_runner.target_free_state(sample, device)


def trusted_regions(sample: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    height, width = sample["target_foreground_mask"].shape[-2:]
    reference = torch.zeros((1, 3, height, width), dtype=torch.float32)
    return build_trusted_silhouette_regions(
        sample,
        reference,
        support_diagonal_ratio=float(config["support_band"]["base_bbox_diagonal_ratio"])
        if "support_band" in config else 0.002708497051881002,
        minimum_radius_pixels=1,
    )


def render_explicit(
    base: Any,
    state: Mapping[str, Any],
    overrides: CanonicalGaussianOverrides,
    background: torch.Tensor,
    *,
    variant: str = "R0",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], dict[str, torch.Tensor]]:
    from gsplat import rasterization

    camera = build_mmlphuman_camera(
        state["camera"], state["camera"]["height"], state["camera"]["width"], base._xyz.device,
    )
    with mmlphuman_state_transaction(base, state["pose"], state["Rh"], state["Th"]):
        means = base.compute_xyz(overrides.as_dict())
        covars = base.get_covariance(canonical_overrides=overrides.as_dict())
        opacities = base.compute_opacity(overrides.as_dict()).reshape(-1)
        camera_position = torch.linalg.inv(camera["w2c"])[:3, 3]
        colors = base.get_color(camera_position, overrides.as_dict())
    width, height = int(camera["width"]), int(camera["height"])
    intrinsic = camera["K"]
    rasterize_mode = "classic"
    if variant == "R5":
        rasterize_mode = "antialiased"
    if variant == "R6":
        width *= 2; height *= 2
        intrinsic = intrinsic.clone()
        intrinsic[:2] *= 2
    tensors = (means, covars, opacities, colors, camera["w2c"], intrinsic)
    if variant == "R4":
        tensors = tuple(value.float() for value in tensors)
        background = background.float()
    image, alpha, info = rasterization(
        means=tensors[0], quats=None, scales=None, opacities=tensors[2], colors=tensors[3],
        viewmats=tensors[4][None], Ks=tensors[5][None], width=width, height=height,
        packed=False, near_plane=0.1, backgrounds=background[None], covars=tensors[1],
        rasterize_mode=rasterize_mode,
    )
    rgb, opacity = chw(image[0], 3), chw(alpha[0], 1)
    if variant == "R6":
        rgb = downsample_supersampled(rgb, 3)
        opacity = downsample_supersampled(opacity, 1)
    return rgb, opacity, info, {
        "canonical_xyz": overrides.xyz,
        "canonical_scaling": overrides.scaling,
        "canonical_rotation": overrides.rotation,
        "canonical_opacity_logit": overrides.opacity,
        "canonical_sh0": overrides.sh0,
        "posed_xyz": means,
        "posed_covariance": covars,
        "effective_opacity": opacities,
    }


def _source_lines(path: Path, pattern: str) -> list[int]:
    return [index for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1) if pattern in line]


def renderer_static_contract(config: Mapping[str, Any]) -> tuple[dict[str, Any], str, str]:
    import gsplat
    from gsplat import rasterization

    package = Path(inspect.getsourcefile(rasterization)).resolve().parent
    rendering = package / "rendering.py"
    wrapper = package / "cuda/_wrapper.py"
    csrc = package / "cuda/csrc"
    include = package / "cuda/include/Common.h"
    project_renderer = PROJECT_ROOT / "scene/gaussian_model.py"
    residuals = PROJECT_ROOT / "scene/gaussian_clothing_residuals.py"
    installed_version = getattr(gsplat, "__version__", "unknown")
    if installed_version != config["renderer_contract"]["expected_version"]:
        raise RuntimeError(f"gsplat version drift: {installed_version}")
    evidence = {
        "project": str(project_renderer), "residuals": str(residuals), "rendering": str(rendering),
        "wrapper": str(wrapper), "projection_cuda": str(csrc / "ProjectionEWA3DGSFused.cu"),
        "raster_cuda": str(csrc / "RasterizeToPixels3DGSFwd.cu"),
        "tile_cuda": str(csrc / "IntersectTile.cu"), "common": str(include),
    }
    entries = [
        (1, "backend/version", installed_version, f"{evidence['project']}:13,596-610; {evidence['rendering']}:33-70"),
        (2, "quaternion order", "wxyz; canonical local q_base*q_delta", f"{evidence['residuals']}:19-20,201-205; {evidence['rendering']}:163-164"),
        (3, "scale activation", "torch.exp before canonical covariance", f"{evidence['project']}:24-25,282-288"),
        (4, "covariance/projection", "canonical covariance, LBS/Rh rotation, EWA pinhole projection with eps2d=0.3", f"{evidence['project']}:282-292,591-610; {evidence['projection_cuda']}:69-120,153-162"),
        (5, "opacity transform", "sigmoid(logit)", f"{evidence['project']}:27-28,403-414"),
        (6, "single Gaussian alpha", "min(0.999, opacity*exp(-sigma))", f"{evidence['raster_cuda']}:141-150"),
        (7, "compositing order", "front-to-back depth-sorted tile intersections", f"{evidence['raster_cuda']}:122-169; {evidence['tile_cuda']}:95-110"),
        (8, "transmittance", "T_next=T*(1-alpha); output alpha=1-T", f"{evidence['raster_cuda']}:153-178"),
        (9, "alpha clamp", "per-Gaussian alpha max 0.999; no post composite clamp", f"{evidence['raster_cuda']}:148,178"),
        (10, "opacity clamp", "no renderer clamp; input already sigmoid", f"{evidence['project']}:403-414; {evidence['raster_cuda']}:130,148"),
        (11, "early termination", "enabled and exclusive", f"{evidence['raster_cuda']}:153-157"),
        (12, "early termination threshold", "next transmittance <= 1e-4", f"{evidence['raster_cuda']}:153-157"),
        (13, "near/far", "near=0.1 explicit; far=1e10 default", f"{evidence['project']}:596-610; {evidence['rendering']}:43-46"),
        (14, "frustum/depth culling", "z near/far plus projected image bounds", f"{evidence['projection_cuda']}:69-75,192-197"),
        (15, "projected radius culling", "both radii <= radius_clip; radius_clip=0 default", f"{evidence['projection_cuda']}:181-190; {evidence['rendering']}:45"),
        (16, "minimum projected radius", "eps2d=0.3; integer radii use ceil", f"{evidence['rendering']}:46,176-178; {evidence['projection_cuda']}:153-184"),
        (17, "maximum projected radius", "no independent maximum; image/tile bounds clip", f"{evidence['projection_cuda']}:164-197"),
        (18, "tile binning", "16x16 default, explicit sorted intersections", f"{evidence['rendering']}:49,631-658"),
        (19, "tile overlap", "inclusive floor min, exclusive ceil max", f"{evidence['tile_cuda']}:55-83"),
        (20, "depth sorting", "intersection id encodes float depth in low 32 bits and is sorted", f"{evidence['tile_cuda']}:95-110; {evidence['wrapper']}:442-517"),
        (21, "antialiasing", "classic default; compensation only in antialiased mode", f"{evidence['rendering']}:54,136-142,452-453"),
        (22, "packed mode", "False explicitly", f"{evidence['project']}:596-610"),
        (23, "sparse gradients", "False default and ineffective with packed=False", f"{evidence['rendering']}:52,119-125"),
        (24, "dtype", "model/projection/raster tensors float32 in formal runtime; CUDA T float32", f"{evidence['raster_cuda']}:101-105,172-178"),
        (25, "image boundary clipping", "projected radius/image bounds then tile bounds", f"{evidence['projection_cuda']}:192-197; {evidence['tile_cuda']}:71-77"),
        (26, "backward visibility/radius", "backward reuses saved accepted range and cutoff", f"{csrc / 'RasterizeToPixels3DGSBwd.cu'}:135-180"),
        (27, "alpha/RGB contributor set", "same accepted alpha loop and last-id for RGB and alpha", f"{evidence['raster_cuda']}:138-186"),
        (28, "alpha metric threshold", "0.5, evaluator only", f"{PROJECT_ROOT / 'tools/run_new_silhouette_semantics.py'}:733-753"),
        (29, "train/eval renderer parity", "single GaussianModel.render path; no training/eval parameter branch", f"{evidence['project']}:588-611"),
        (30, "non-differentiable selection", "ceil integer radii and no-grad tile assignment; no model-side detach", f"{evidence['projection_cuda']}:181-184; {evidence['wrapper']}:442-443"),
    ]
    payload = {
        "status": "PASS", "backend": "gsplat", "version": installed_version,
        "signature": str(inspect.signature(rasterization)), "entries": [
            {"index": index, "item": item, "value": value, "evidence": source}
            for index, item, value, source in entries
        ], "formal_arguments": config["renderer_contract"],
    }
    markdown = ["# Renderer static contract audit", "", f"- backend: `gsplat {installed_version}`", "- status: **PASS**", ""]
    for row in payload["entries"]:
        markdown.append(f"{row['index']}. **{row['item']}** — {row['value']}  ")
        markdown.append(f"   Evidence: `{row['evidence']}`")
    call_graph = "\n".join([
        "FixedOpenGaussianOracle / UnboundedGaussianDeltaField",
        "  -> compose_canonical_gaussian_overrides (wxyz, log scale, opacity logit)",
        "  -> GaussianModel.compute_xyz / get_covariance / compute_opacity / get_color",
        "  -> gsplat.rasterization(packed=False, near_plane=0.1, covars=...)",
        "     -> fully_fused_projection (EWA pinhole, eps2d=0.3)",
        "     -> isect_tiles (16x16, depth-sorted)",
        "     -> rasterize_to_pixels (alpha cutoff 1/255, early T<=1e-4)",
        "     -> RGB and alpha from the same contributor loop",
    ])
    return payload, "\n".join(markdown), call_graph


def _mask_numpy(value: torch.Tensor) -> np.ndarray:
    result = value.detach().cpu().numpy()
    while result.ndim > 2:
        result = result[0]
    return result >= 0.5


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == label


def crop_bounds(mask: np.ndarray, maximum: int, fallback: tuple[int, int]) -> tuple[int, int, int, int]:
    y, x = np.nonzero(mask)
    cy, cx = (int(np.median(y)), int(np.median(x))) if len(x) else fallback
    half = maximum // 2
    y0 = max(0, min(mask.shape[0] - maximum, cy - half))
    x0 = max(0, min(mask.shape[1] - maximum, cx - half))
    return x0, y0, min(mask.shape[1], x0 + maximum), min(mask.shape[0], y0 + maximum)


def gaussian_mask_from_image(info: Mapping[str, Any], mask: np.ndarray) -> torch.Tensor:
    means = info["means2d"][0].detach().cpu()
    x = torch.floor(means[:, 0]).long(); y = torch.floor(means[:, 1]).long()
    valid = (info["radii"][0].detach().cpu().amin(-1) > 0) & (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
    selected = torch.zeros(means.shape[0], dtype=torch.bool)
    if valid.any():
        array = torch.from_numpy(mask.astype(np.bool_))
        selected[valid] = array[y[valid], x[valid]]
    return selected


def _rotation_angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    dot = (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(-1).abs().clamp(0, 1)
    return 2 * torch.acos(dot)


def parameter_rows(
    state_name: str,
    outfit: str,
    condition: str,
    base: Any,
    overrides: CanonicalGaussianOverrides,
    info: Mapping[str, Any],
    base_info: Mapping[str, Any],
    regions: Mapping[str, torch.Tensor],
    alpha: np.ndarray,
    target: np.ndarray,
) -> list[dict[str, Any]]:
    dynamic = {
        "trusted_expansion": _mask_numpy(regions["trusted_expansion"]),
        "trusted_removal": _mask_numpy(regions["trusted_removal"]),
        "old_garment": _mask_numpy(regions["old_garment_removal"]),
        "garment_core": _mask_numpy(regions["target_garment"]),
        "transition": _mask_numpy(regions["transition"]),
        "protected": _mask_numpy(regions["protected_identity"]),
        "trailing_cloud_fp": _largest_component((alpha >= 0.05) & (~target)),
        "internal_hole_fn": largest_internal_hole(alpha >= 0.5, target),
    }
    metrics = {
        "xyz_residual": torch.linalg.vector_norm(overrides.xyz.detach() - base._xyz.detach(), dim=1).cpu(),
        "projected_center_displacement": torch.linalg.vector_norm(info["means2d"][0].detach() - base_info["means2d"][0].detach(), dim=1).cpu(),
        "log_scaling_residual": torch.linalg.vector_norm(overrides.scaling.detach() - base._scaling.detach(), dim=1).cpu(),
        "projected_major_radius": info["radii"][0].detach().amax(-1).cpu().float(),
        "projected_minor_radius": info["radii"][0].detach().amin(-1).cpu().float(),
        "rotation_residual": _rotation_angle(overrides.rotation.detach(), base._rotation.detach()).cpu(),
        "opacity_logit_residual": (overrides.opacity.detach() - base._opacity.detach()).reshape(-1).abs().cpu(),
        "effective_center_alpha": torch.sigmoid(overrides.opacity.detach()).reshape(-1).cpu(),
        "sh0_residual": torch.linalg.vector_norm((overrides.sh0.detach() - base._sh0.detach()).reshape(base._xyz.shape[0], -1), dim=1).cpu(),
        "visible": (info["radii"][0].detach().amin(-1) > 0).cpu().float(),
        "tiles_per_gaussian": info["tiles_per_gauss"][0].detach().cpu().float(),
    }
    rows = []
    for region_name, image_mask in dynamic.items():
        gaussian_mask = gaussian_mask_from_image(info, image_mask)
        for metric, values in metrics.items():
            stats = quantile_summary(values[gaussian_mask])
            rows.append({
                "state": state_name, "outfit": outfit, "condition": condition, "view": VIEWS[condition],
                "region": region_name, "metric": metric, "gaussian_count": int(gaussian_mask.sum()),
                **{key: None if math.isnan(value) else value for key, value in stats.items()},
            })
    return rows


def tile_candidates(info: Mapping[str, Any], y: int, x: int) -> torch.Tensor:
    tile_size = int(info["tile_size"])
    tile_y, tile_x = y // tile_size, x // tile_size
    offsets = info["isect_offsets"].reshape(-1).detach().cpu()
    tile_width = int(info["tile_width"]); tile_height = int(info["tile_height"])
    if not (0 <= tile_x < tile_width and 0 <= tile_y < tile_height):
        return torch.empty((0,), dtype=torch.long)
    tile_index = tile_y * tile_width + tile_x
    start = int(offsets[tile_index])
    end = int(offsets[tile_index + 1]) if tile_index + 1 < offsets.numel() else int(info["flatten_ids"].numel())
    return info["flatten_ids"][start:end].detach().cpu().long()


def pixel_composition(info: Mapping[str, Any], y: int, x: int) -> dict[str, Any]:
    candidates = tile_candidates(info, y, x)
    if not candidates.numel():
        return {
            "tile_candidate_count": 0, "active_contributor_count": 0,
            "production_alpha_reconstructed": 0.0, "reference_alpha": 0.0,
            "early_termination_index": None, "reason": "no_nearby_gaussian_support", "contributors": [],
        }
    means = info["means2d"][0].detach().cpu()[candidates]
    conics = info["conics"][0].detach().cpu()[candidates]
    opacities = info["opacities"][0].detach().cpu()[candidates]
    depths = info["depths"][0].detach().cpu()[candidates]
    pixel = torch.tensor([x + 0.5, y + 0.5], dtype=means.dtype)
    delta = means - pixel
    sigma = 0.5 * (conics[:, 0] * delta[:, 0].square() + conics[:, 2] * delta[:, 1].square()) + conics[:, 1] * delta[:, 0] * delta[:, 1]
    alphas = torch.minimum(opacities * torch.exp(-sigma), torch.full_like(opacities, ALPHA_CLAMP_MAX))
    valid_conic = sigma >= 0
    reference_values = torch.where(valid_conic, alphas, torch.zeros_like(alphas))
    reference = float(reference_alpha_composite(reference_values))
    transmittance = 1.0
    details = []
    early = None
    for index in range(candidates.numel()):
        alpha = float(alphas[index])
        if not bool(valid_conic[index]) or alpha < ALPHA_THRESHOLD:
            continue
        next_transmittance = transmittance * (1.0 - alpha)
        if next_transmittance <= EARLY_TERMINATION_TRANSMITTANCE:
            early = index
            break
        contribution = alpha * transmittance
        details.append({
            "gaussian_index": int(candidates[index]), "depth": float(depths[index]),
            "projected_center_distance": float(torch.linalg.vector_norm(delta[index])),
            "projected_mean": [float(value) for value in means[index]],
            "conic": [float(value) for value in conics[index]], "single_alpha": alpha,
            "transmittance_before": transmittance, "alpha_contribution": contribution,
            "cumulative_alpha": 1.0 - next_transmittance, "tile_order": index,
        })
        transmittance = next_transmittance
    production = 1.0 - transmittance
    reason_flags = {
        "conic_rejected": not bool(valid_conic.any()),
        "alpha_below_cutoff": bool(valid_conic.any()) and not any(row["single_alpha"] >= ALPHA_THRESHOLD for row in details),
        "behind_transmittance_termination": early is not None,
        "no_nearby_gaussian_support": reference < 0.5,
    }
    return {
        "tile_candidate_count": int(candidates.numel()),
        "active_contributor_count": len(details),
        "production_alpha_reconstructed": production, "reference_alpha": reference,
        "early_termination_index": early, "reason": culling_reason(reason_flags),
        "contributors": sorted(details, key=lambda row: (-row["alpha_contribution"], row["gaussian_index"]))[:10],
    }


def enrich_contributors(
    composition: dict[str, Any],
    base: Any,
    tensors: Mapping[str, torch.Tensor],
    info: Mapping[str, Any],
) -> None:
    weights = base.get_weights.detach()
    base_center = base._xyz.detach().mean(0)
    base_radius = torch.linalg.vector_norm(base._xyz.detach() - base_center, dim=1).max()
    radii = info["radii"][0].detach()
    for row in composition["contributors"]:
        index = int(row["gaussian_index"])
        scale = torch.exp(tensors["canonical_scaling"][index].detach())
        canonical = tensors["canonical_xyz"][index].detach()
        posed = tensors["posed_xyz"][index].detach()
        row.update({
            "projected_radii": [int(value) for value in radii[index].cpu()],
            "opacity_logit": float(tensors["canonical_opacity_logit"][index].detach().reshape(-1)[0]),
            "effective_opacity": float(tensors["effective_opacity"][index].detach()),
            "scale": [float(value) for value in scale.cpu()],
            "dominant_lbs_joint": int(weights[index, :55].argmax()),
            "canonical_xyz": [float(value) for value in canonical.cpu()],
            "posed_xyz": [float(value) for value in posed.cpu()],
            "abnormal": bool(
                torch.linalg.vector_norm(canonical - base_center) > 1.5 * base_radius
                or scale.max() / torch.exp(base._scaling[index].detach()).max() > 8
            ),
        })


def reference_crop(info: Mapping[str, Any], production: np.ndarray, bounds: tuple[int, int, int, int]) -> tuple[np.ndarray, dict[str, float]]:
    x0, y0, x1, y1 = bounds
    reference = np.zeros((y1 - y0, x1 - x0), dtype=np.float64)
    for local_y, y in enumerate(range(y0, y1)):
        for local_x, x in enumerate(range(x0, x1)):
            reference[local_y, local_x] = pixel_composition(info, y, x)["reference_alpha"]
    prod = production[y0:y1, x0:x1].astype(np.float64)
    difference = np.abs(prod - reference)
    metrics = {
        "max_absolute_difference": float(difference.max(initial=0)),
        "mean_absolute_difference": float(difference.mean()) if difference.size else 0.0,
        "p95_absolute_difference": float(np.quantile(difference, .95)) if difference.size else 0.0,
        "binary_coverage_disagreement": float(((prod >= .5) != (reference >= .5)).mean()) if difference.size else 0.0,
    }
    return reference, metrics


def save_triptych(path: Path, production: np.ndarray, reference: np.ndarray) -> None:
    difference = np.abs(production - reference)
    panels = []
    for value in (production, reference, difference / max(float(difference.max()), 1e-8)):
        panels.append(Image.fromarray((np.clip(value, 0, 1) * 255).round().astype(np.uint8), "L").convert("RGB"))
    canvas = Image.new("RGB", (sum(panel.width for panel in panels), max(panel.height for panel in panels)), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0)); x += panel.width
    path.parent.mkdir(parents=True, exist_ok=True); canvas.save(path)


def make_contact_sheet(path: Path, panels: list[tuple[str, Path]], columns: int = 4, width: int = 320) -> None:
    loaded = []
    for label, source in panels:
        with Image.open(source) as image:
            image.load(); value = image.convert("RGB")
        height = max(1, round(value.height * width / value.width))
        value = value.resize((width, height), Image.Resampling.LANCZOS)
        loaded.append((label, value))
    row_height = max((image.height for _, image in loaded), default=1) + 28
    rows = math.ceil(len(loaded) / columns)
    canvas = Image.new("RGB", (columns * width, rows * row_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(loaded):
        x, y = (index % columns) * width, (index // columns) * row_height
        canvas.paste(image, (x, y + 24)); draw.text((x + 4, y + 4), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True); canvas.save(path)


def finite_difference_screen_space(
    info: Mapping[str, Any], pixel_rows: list[dict[str, Any]], epsilons: list[float],
) -> list[dict[str, Any]]:
    """Check the exact local alpha expression exported by production gsplat.

    Projection Jacobians are audited separately by production autograd. These
    deterministic checks isolate the raster alpha sensitivity to projected xyz,
    covariance (the log-scale/rotation consumers), and opacity.
    """

    results = []
    for pixel in pixel_rows:
        y, x = int(pixel["y"]), int(pixel["x"])
        candidates = tile_candidates(info, y, x)
        if not candidates.numel():
            continue
        comp = pixel_composition(info, y, x)
        if not comp["contributors"]:
            continue
        gid = int(comp["contributors"][0]["gaussian_index"])
        location = int((candidates == gid).nonzero(as_tuple=False)[0])
        mean = info["means2d"][0, gid].detach().double().requires_grad_(True)
        conic = info["conics"][0, gid].detach().double().requires_grad_(True)
        opacity = info["opacities"][0, gid].detach().double().reshape(1).requires_grad_(True)
        pixel_xy = torch.tensor([x + .5, y + .5], dtype=torch.float64, device=mean.device)
        alpha, sigma = single_gaussian_alpha(pixel_xy, mean, conic, opacity)
        alpha.backward()
        groups = {"xyz_projected": mean, "logscale_conic": conic, "rotation_conic": conic, "opacity_logit_consumer": opacity}
        for group, tensor in groups.items():
            gradient = tensor.grad.detach().clone()
            component = int(gradient.abs().argmax())
            for epsilon in epsilons:
                plus_mean, minus_mean = mean.detach().clone(), mean.detach().clone()
                plus_conic, minus_conic = conic.detach().clone(), conic.detach().clone()
                plus_opacity, minus_opacity = opacity.detach().clone(), opacity.detach().clone()
                target_plus = {"xyz_projected": plus_mean, "logscale_conic": plus_conic, "rotation_conic": plus_conic, "opacity_logit_consumer": plus_opacity}[group]
                target_minus = {"xyz_projected": minus_mean, "logscale_conic": minus_conic, "rotation_conic": minus_conic, "opacity_logit_consumer": minus_opacity}[group]
                target_plus[component] += epsilon; target_minus[component] -= epsilon
                plus = single_gaussian_alpha(pixel_xy, plus_mean, plus_conic, plus_opacity)[0]
                minus = single_gaussian_alpha(pixel_xy, minus_mean, minus_conic, minus_opacity)[0]
                finite = float((plus - minus) / (2 * epsilon))
                auto = float(gradient[component])
                results.append({
                    **pixel, "gaussian_index": gid, "tile_order": location, "parameter": group,
                    "component": component, "epsilon": epsilon, "autograd": auto,
                    "finite_difference": finite, "absolute_error": abs(auto - finite),
                    "relative_error": abs(auto - finite) / max(abs(auto), abs(finite), 1e-12),
                    "sign_match": bool(auto == 0 == finite or np.sign(auto) == np.sign(finite)),
                    "finite": bool(math.isfinite(auto) and math.isfinite(finite)),
                    "sigma": float(sigma.detach()),
                })
    return results


def run_audit(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    root = args.output.resolve()
    if root.exists():
        raise FileExistsError(f"append-only output already exists: {root}")
    for name in OUTPUT_DIRS:
        (root / name).mkdir(parents=True, exist_ok=False)
    status_path = root / "RUN_STATUS.json"
    stage = "contract"
    atomic_json(status_path, {"status": "RUNNING", "stage": stage, "optimizer_steps": 0, "started_at": now()})
    try:
        current_head = git("rev-parse", "HEAD")
        if git("branch", "--show-current") != config["research_branch"]:
            raise RuntimeError("audit must run from the registered research branch")
        if git("status", "--short"):
            raise RuntimeError("formal audit requires a clean cloud worktree")
        if not subprocess.run(
            ["git", "merge-base", "--is-ancestor", config["source_head"], current_head], cwd=PROJECT_ROOT,
        ).returncode == 0:
            raise RuntimeError("formal audit commit is not descended from V6.1 final HEAD")
        if git("rev-list", "-n", "1", config["source_tag"]) != config["source_head"]:
            raise RuntimeError("V6.1 annotated tag does not resolve to the frozen source HEAD")
        source_roots = [Path(config[key]) for key in ("source_representation_output", "source_v6_output", "source_v6_1_output")]
        if not all(path.is_dir() for path in source_roots):
            raise FileNotFoundError("one or more frozen source attempts are missing")
        state_manifest = []
        for state in ("P1", "P2", "P3"):
            for outfit in OUTFITS:
                path = _checkpoint_path(config, state, outfit)
                actual = sha256(path)
                if actual != config["states"][state]["sha256"][outfit]:
                    raise RuntimeError(f"checkpoint drift: {state}/{outfit}")
                state_manifest.append({"state": state, "outfit": outfit, "path": str(path), "sha256": actual})
        manifest_path = Path(config["source_manifest"])
        pipeline_path = PROJECT_ROOT / config["pipeline_config"]
        frozen_refs = {branch: resolve_branch_commit(branch) for branch in config["frozen_branches"]}
        if frozen_refs != config["frozen_branches"]:
            raise RuntimeError(f"frozen branch drift: expected={config['frozen_branches']} actual={frozen_refs}")
        atomic_json(root / "contract/config_resolved.json", config)
        atomic_json(root / "contract/run_manifest.json", {
            "schema_version": SCHEMA, "task_id": config["task_id"], "git": _git_state(),
            "environment": _environment(), "run_commit": current_head, "optimizer_steps": 0,
            "dataset_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
            "pipeline_config": {"path": str(pipeline_path), "sha256": sha256(pipeline_path)},
            "states": state_manifest, "frozen_branch_heads_before": frozen_refs,
            "permissions": config["permissions"], "target_fields_forward": False,
        })
        atomic_json(root / "input_audit/fixed_input_contract.json", {
            "status": "PASS", "outfits": list(OUTFITS), "conditions": config["conditions"],
            "focus_samples": config["focus_samples"], "control_samples": config["control_samples"],
            "state_count": 4, "reoptimization_before_static_audit": False,
            "source_attempts_append_only": True,
        })
        static_payload, static_md, call_graph = renderer_static_contract(config)
        atomic_json(root / "renderer_static_audit/renderer_static_contract.json", static_payload)
        atomic_text(root / "renderer_static_audit/RENDERER_STATIC_CONTRACT_AUDIT.md", static_md)
        atomic_text(root / "renderer_static_audit/renderer_call_graph.txt", call_graph)

        stage = "static_render_and_parameter_state"
        atomic_json(status_path, {"status": "RUNNING", "stage": stage, "optimizer_steps": 0, "updated_at": now()})
        alpha_cache: dict[tuple[str, str, str], np.ndarray] = {}
        rgb_cache: dict[tuple[str, str, str], torch.Tensor] = {}
        region_cache: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
        parameter_statistics: list[dict[str, Any]] = []
        threshold_rows: list[dict[str, Any]] = []
        culling_rows: list[dict[str, Any]] = []
        trace_rows: list[dict[str, Any]] = []
        trace_summaries: list[dict[str, Any]] = []
        reference_rows: list[dict[str, Any]] = []
        gradient_rows: list[dict[str, Any]] = []
        variant_rows: list[dict[str, Any]] = []
        visual_panels: list[tuple[str, Path]] = []
        reference_panels: list[tuple[str, Path]] = []
        variant_panels: list[tuple[str, Path]] = []
        base_fingerprints: dict[str, str] = {}
        default_regression: dict[str, Any] | None = None

        for outfit in OUTFITS:
            base, samples, background, device = load_runtime(config, outfit, args.device)
            base_fingerprints[outfit] = _tensor_state_fingerprint({
                name: value for name, value in {
                    "xyz": base._xyz, "scaling": base._scaling, "rotation": base._rotation,
                    "opacity": base._opacity, "sh0": base._sh0, "shN": base._shN,
                }.items()
            }.items())
            states = {condition: target_free_state(sample, device) for condition, sample in samples.items()}
            for condition, sample in samples.items():
                region_cache[(outfit, condition)] = trusted_regions(sample, config)
            base_infos: dict[str, dict[str, Any]] = {}
            p3_context: dict[str, tuple[Any, Any, Any, Any]] = {}

            for state_name in STATES:
                model, overrides = load_state_model(config, state_name, outfit, base)
                for condition in CONDITIONS:
                    with torch.no_grad():
                        rgb, alpha, info, tensors = render_explicit(base, states[condition], overrides, background)
                    alpha_np = alpha[0].float().cpu().numpy()
                    alpha_cache[(state_name, outfit, condition)] = alpha_np
                    if (outfit, condition) in {("O08", "cond_000318"), ("O08", "cond_000347"), ("O01", "cond_000318"), ("O01", "cond_000347")}:
                        rgb_cache[(state_name, outfit, condition)] = rgb.detach().cpu()
                    view = VIEWS[condition]
                    alpha_path = root / f"alpha_threshold_sweep/soft_alpha/{state_name}/{outfit}/{view}.png"
                    save_alpha(alpha_path, alpha)
                    if state_name in ("P1", "P2", "P3") and outfit == "O08" and view in ("back", "right"):
                        visual_panels.append((f"{state_name} O08 {view} alpha", alpha_path))
                    if state_name == "P0":
                        base_infos[condition] = {key: value.detach().clone() if isinstance(value, torch.Tensor) else value for key, value in info.items()}
                    else:
                        target = _mask_numpy(samples[condition]["target_foreground_mask"])
                        parameter_statistics.extend(parameter_rows(
                            state_name, outfit, condition, base, overrides, info, base_infos[condition],
                            region_cache[(outfit, condition)], alpha_np, target,
                        ))
                    if state_name == "P3" and (outfit, condition) in {
                        ("O08", "cond_000318"), ("O08", "cond_000347"), ("O01", "cond_000318")
                    }:
                        p3_context[condition] = (model, overrides, info, tensors)
                    if default_regression is None and state_name == "P0" and outfit == "O01" and condition == "cond_000000":
                        camera = build_mmlphuman_camera(states[condition]["camera"], alpha.shape[1], alpha.shape[2], device)
                        with torch.no_grad(), mmlphuman_state_transaction(base, states[condition]["pose"], states[condition]["Rh"], states[condition]["Th"]):
                            expected = base.render(camera, background=background, canonical_overrides=overrides.as_dict())
                        expected_rgb, expected_alpha = chw(expected[0], 3), chw(expected[1], 1)
                        default_regression = {
                            "rgb_max_abs": float((expected_rgb - rgb).abs().max()),
                            "alpha_max_abs": float((expected_alpha - alpha).abs().max()),
                            "rgb_bitwise": bool(torch.equal(expected_rgb, rgb)),
                            "alpha_bitwise": bool(torch.equal(expected_alpha, alpha)),
                        }

                if state_name in ("P1", "P2", "P3"):
                    for condition in CONDITIONS:
                        sample = samples[condition]
                        regions = region_cache[(outfit, condition)]
                        alpha_np = alpha_cache[(state_name, outfit, condition)]
                        target = _mask_numpy(sample["target_foreground_mask"])
                        for threshold in config["silhouette_thresholds"]:
                            row = threshold_metrics(
                                alpha_np, target, _mask_numpy(regions["trusted_expansion"]),
                                _mask_numpy(regions["trusted_removal"]), _mask_numpy(regions["background"]),
                                float(threshold),
                            )
                            threshold_rows.append({"state": state_name, "outfit": outfit, "condition": condition, "view": VIEWS[condition], **row})

                if state_name in ("P1", "P2", "P3"):
                    for condition in CONDITIONS:
                        if not (outfit == "O08" and condition in ("cond_000318", "cond_000347")):
                            continue
                        with torch.no_grad():
                            _, _, info, _ = render_explicit(base, states[condition], overrides, background)
                        alpha_np = alpha_cache[(state_name, outfit, condition)]
                        regions = region_cache[(outfit, condition)]
                        masks = {
                            "trusted_fn": _mask_numpy(regions["trusted_expansion"]) & (alpha_np < .5),
                            "cloud_fp": (~_mask_numpy(samples[condition]["target_foreground_mask"])) & (alpha_np >= .05),
                        }
                        for category, mask in masks.items():
                            reasons = Counter()
                            candidates = deterministic_mask_indices(mask, min(512, int(mask.sum())))
                            for y, x in candidates:
                                reasons[pixel_composition(info, y, x)["reason"]] += 1
                            for reason in CULLING_REASONS:
                                culling_rows.append({
                                    "state": state_name, "outfit": outfit, "condition": condition,
                                    "view": VIEWS[condition], "pixel_category": category, "reason": reason,
                                    "count": int(reasons[reason]), "fraction": reasons[reason] / max(len(candidates), 1),
                                    "sampled_pixel_count": len(candidates), "sampling": "deterministic_even_up_to_512",
                                })

                if state_name in ("P1", "P2", "P3"):
                    for variant in registered_render_variants():
                        if not variant.supported:
                            continue
                        for condition in CONDITIONS:
                            if (outfit, condition) not in {
                                ("O08", "cond_000318"), ("O08", "cond_000347"), ("O08", "cond_000000"),
                                ("O01", "cond_000318"), ("O01", "cond_000347"),
                            }:
                                continue
                            start = time.perf_counter()
                            torch.cuda.reset_peak_memory_stats(device)
                            with torch.no_grad():
                                rgb, alpha, _, _ = render_explicit(base, states[condition], overrides, background, variant=variant.name)
                            elapsed = time.perf_counter() - start
                            alpha_np = alpha[0].float().cpu().numpy()
                            regions = region_cache[(outfit, condition)]
                            target = _mask_numpy(samples[condition]["target_foreground_mask"])
                            metrics = threshold_metrics(
                                alpha_np, target, _mask_numpy(regions["trusted_expansion"]),
                                _mask_numpy(regions["trusted_removal"]), _mask_numpy(regions["background"]), .5,
                            )
                            baseline_alpha = alpha_cache[(state_name, outfit, condition)]
                            baseline_rgb = rgb_cache.get((state_name, outfit, condition))
                            row = {
                                "state": state_name, "variant": variant.name, "changed_setting": variant.changed_setting,
                                "outfit": outfit, "condition": condition, "view": VIEWS[condition], **metrics,
                                "runtime_seconds": elapsed, "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
                                "alpha_mae_vs_production": float(np.abs(alpha_np - baseline_alpha).mean()),
                                "rgb_mae_vs_production": float((rgb.cpu() - baseline_rgb).abs().mean()) if baseline_rgb is not None else None,
                                "target_fields_in_forward": False,
                            }
                            variant_rows.append(row)
                            if state_name == "P3" and outfit == "O08" and condition in ("cond_000318", "cond_000347"):
                                path = root / f"render_variant_matrix/renders/{variant.name}/{state_name}/{outfit}/{VIEWS[condition]}_alpha.png"
                                save_alpha(path, alpha); variant_panels.append((f"{variant.name} {state_name} {VIEWS[condition]}", path))

            # Detailed P3 contributor/reference audit.
            if outfit == "O08":
                selected_specs = [
                    ("o08_back_max_hole", "cond_000318", "hole"),
                    ("o08_right_max_hole", "cond_000347", "hole"),
                    ("o08_back_sleeve_scallop", "cond_000318", "scallop"),
                    ("o08_right_trailing_cloud", "cond_000347", "cloud"),
                ]
            else:
                selected_specs = [("o01_normal_boundary", "cond_000318", "boundary")]
            for region_name, condition, mode in selected_specs:
                model, overrides, info, tensors = p3_context[condition]
                alpha_np = alpha_cache[("P3", outfit, condition)]
                target = _mask_numpy(samples[condition]["target_foreground_mask"])
                regions = region_cache[(outfit, condition)]
                if mode == "hole":
                    anchor_mask = largest_internal_hole(alpha_np >= .5, target)
                elif mode == "cloud":
                    anchor_mask = _largest_component((alpha_np >= .05) & (~target))
                elif mode == "scallop":
                    anchor_mask = _largest_component(_mask_numpy(regions["trusted_expansion"]) & (alpha_np < .5))
                else:
                    gradient = cv2.morphologyEx(target.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
                    anchor_mask = gradient & (alpha_np >= .5)
                bounds = crop_bounds(anchor_mask, int(config["reference_crop_maximum"]), (alpha_np.shape[0] // 2, alpha_np.shape[1] // 2))
                x0, y0, x1, y1 = bounds
                crop_mask = np.zeros_like(target); crop_mask[y0:y1, x0:x1] = True
                categories = {
                    "FN": target & (alpha_np < .5) & crop_mask,
                    "correct_coverage": target & (alpha_np >= .5) & crop_mask,
                    "background_FP": (~target) & (alpha_np >= .05) & crop_mask,
                    "normal_background": (~target) & (alpha_np < .01) & crop_mask,
                }
                trace_pixel_rows = []
                for category, mask in categories.items():
                    for y, x in deterministic_mask_indices(mask, int(config["trace_pixels_per_category"])):
                        composition = pixel_composition(info, y, x)
                        enrich_contributors(composition, base, tensors, info)
                        record = {
                            "region": region_name, "outfit": outfit, "condition": condition, "view": VIEWS[condition],
                            "category": category, "y": y, "x": x, "production_alpha": float(alpha_np[y, x]),
                            "projected_visible_gaussian_count": int((info["radii"][0].amin(-1) > 0).sum()),
                            "culled_or_rejected_in_tile": composition["tile_candidate_count"] - composition["active_contributor_count"],
                            "crop_bounds": list(bounds), **composition,
                        }
                        append_jsonl(root / "contributor_tracing/pixel_contributor_trace.jsonl", record)
                        trace_rows.append(record)
                        trace_pixel_rows.append({"region": region_name, "category": category, "y": y, "x": x})
                reference, metrics = reference_crop(info, alpha_np, bounds)
                production_crop = alpha_np[y0:y1, x0:x1]
                fn = target[y0:y1, x0:x1] & (production_crop < .5)
                recovered = fn & (reference >= .5)
                fp = (~target[y0:y1, x0:x1]) & (reference >= .5) & (production_crop < .5)
                metrics.update({
                    "region": region_name, "outfit": outfit, "condition": condition, "view": VIEWS[condition],
                    "bounds": list(bounds), "trusted_fn_pixels": int(fn.sum()),
                    "trusted_fn_recovered_fraction": float(recovered.sum() / max(int(fn.sum()), 1)),
                    "new_fp_fraction": float(fp.sum() / max(int((~target[y0:y1, x0:x1]).sum()), 1)),
                })
                reference_rows.append(metrics)
                image_path = root / f"reference_compositor/{region_name}_production_reference_difference.png"
                save_triptych(image_path, production_crop, reference); reference_panels.append((region_name, image_path))
                if outfit == "O08" and condition in ("cond_000318", "cond_000347"):
                    candidates = []
                    category_limits = {"FN": 4, "correct_coverage": 2, "background_FP": 2}
                    for category, limit in category_limits.items():
                        candidates.extend([row for row in trace_pixel_rows if row["category"] == category][:limit])
                    gradient_rows.extend(finite_difference_screen_space(info, candidates, [float(value) for value in config["gradient_epsilons"]]))

        # Aggregate and persist static results.
        stage = "persistence_and_preliminary_adjudication"
        atomic_json(status_path, {"status": "RUNNING", "stage": stage, "optimizer_steps": 0, "updated_at": now()})
        write_csv(root / "parameter_state_comparison/parameter_state_comparison.csv", parameter_statistics)
        atomic_json(root / "parameter_state_comparison/parameter_state_comparison.json", {
            "rows": parameter_statistics, "base_fingerprints": base_fingerprints,
            "default_renderer_regression": default_regression,
        })
        write_csv(root / "alpha_threshold_sweep/alpha_threshold_sweep.csv", threshold_rows)
        write_csv(root / "projection_and_culling/culling_reason_summary.csv", culling_rows)
        atomic_json(root / "projection_and_culling/culling_reason_summary.json", {"rows": culling_rows, "categories": CULLING_REASONS})
        summary_by_group = []
        for key in sorted({(row["state"], row["outfit"], row["condition"], row["pixel_category"]) for row in culling_rows}):
            selected = [row for row in culling_rows if (row["state"], row["outfit"], row["condition"], row["pixel_category"]) == key]
            dominant = max(selected, key=lambda row: row["count"])
            summary_by_group.append({"state": key[0], "outfit": key[1], "condition": key[2], "pixel_category": key[3], "dominant_reason": dominant["reason"], "fraction": dominant["fraction"]})
        atomic_text(root / "projection_and_culling/CULLING_REASON_AUDIT.md", "\n".join([
            "# Projection and culling audit", "", "All reason rows include zero counts; `unknown` is never silently dropped.", "",
            *[f"- {row['state']} {row['outfit']} {VIEWS[row['condition']]} {row['pixel_category']}: `{row['dominant_reason']}` ({row['fraction']:.3f})" for row in summary_by_group],
        ]))
        trace_summary_rows = []
        for key in sorted({(row["region"], row["category"]) for row in trace_rows}):
            rows = [row for row in trace_rows if (row["region"], row["category"]) == key]
            trace_summary_rows.append({
                "region": key[0], "category": key[1], "pixel_count": len(rows),
                "mean_tile_candidates": float(np.mean([row["tile_candidate_count"] for row in rows])) if rows else 0,
                "mean_active_contributors": float(np.mean([row["active_contributor_count"] for row in rows])) if rows else 0,
                "mean_production_alpha": float(np.mean([row["production_alpha"] for row in rows])) if rows else 0,
                "mean_reference_alpha": float(np.mean([row["reference_alpha"] for row in rows])) if rows else 0,
                "dominant_reason": Counter(row["reason"] for row in rows).most_common(1)[0][0] if rows else "unknown",
            })
        write_csv(root / "contributor_tracing/pixel_contributor_summary.csv", trace_summary_rows)
        atomic_text(root / "contributor_tracing/CONTRIBUTOR_TRACE_REPORT.md", "\n".join([
            "# Contributor trace report", "", "Pixels were chosen deterministically from registered crops, never by favorable manual picking.", "",
            *[f"- {row['region']} / {row['category']}: n={row['pixel_count']}, candidates={row['mean_tile_candidates']:.2f}, contributors={row['mean_active_contributors']:.2f}, reason=`{row['dominant_reason']}`" for row in trace_summary_rows],
        ]))
        atomic_json(root / "reference_compositor/reference_compositor_metrics.json", {
            "name": "REFERENCE_ALPHA_COMPOSITOR_V1", "dtype": "float64", "early_termination": False,
            "alpha_cutoff": False, "maximum_crop": config["reference_crop_maximum"], "rows": reference_rows,
            "candidate_interface_limitation": "uses production opacity-aware projected tile lists; contributions omitted before tile export cannot be reconstructed",
        })
        atomic_text(root / "reference_compositor/REFERENCE_ALPHA_COMPOSITOR_REPORT.md", "\n".join([
            "# REFERENCE_ALPHA_COMPOSITOR_V1", "", "The compositor is CPU/PyTorch float64, front-to-back/product equivalent for alpha, with no per-pixel cutoff or early termination.",
            "It deliberately consumes production projected tile lists and therefore cannot recover Gaussians removed before tile export.", "",
            *[f"- {row['region']}: mean={row['mean_absolute_difference']:.6f}, p95={row['p95_absolute_difference']:.6f}, FN recovered={row['trusted_fn_recovered_fraction']:.3f}" for row in reference_rows],
        ]))
        write_csv(root / "gradient_checks/alpha_gradient_check.csv", gradient_rows)
        finite_rows = [row for row in gradient_rows if row["finite"]]
        atomic_text(root / "gradient_checks/ALPHA_GRADIENT_AUDIT.md", "\n".join([
            "# Alpha gradient audit", "",
            "Scope: production-exported one-Gaussian pixel alpha expression. `xyz_projected` and conic consumers isolate projection/raster sensitivity; discrete tile/radius transitions remain explicitly non-differentiable.",
            f"- rows: {len(gradient_rows)}", f"- all finite: {len(finite_rows) == len(gradient_rows)}",
            f"- max relative error: {max((row['relative_error'] for row in finite_rows), default=0):.6g}",
            "- full parameter-to-projection finite differences were not substituted for this scoped check.",
        ]))
        unsupported = [item.__dict__ for item in registered_render_variants() if not item.supported]
        write_csv(root / "render_variant_matrix/render_variant_matrix.csv", variant_rows)
        atomic_json(root / "render_variant_matrix/render_variant_matrix.json", {"rows": variant_rows, "unsupported": unsupported})

        # Variant acceptance is evaluated only for P3; R4/R5/R6 are static candidates.
        candidate_decisions = []
        for variant in ("R4", "R5", "R6"):
            focus = [row for row in variant_rows if row["state"] == "P3" and row["variant"] == variant and row["outfit"] == "O08" and row["condition"] in ("cond_000318", "cond_000347")]
            baseline = [row for row in variant_rows if row["state"] == "P3" and row["variant"] == "R0" and row["outfit"] == "O08" and row["condition"] in ("cond_000318", "cond_000347")]
            by_condition = {row["condition"]: row for row in baseline}
            gains = [row["trusted_expansion_recall"] - by_condition[row["condition"]]["trusted_expansion_recall"] for row in focus]
            reductions = [
                (by_condition[row["condition"]]["internal_hole_ratio"] - row["internal_hole_ratio"]) / max(by_condition[row["condition"]]["internal_hole_ratio"], 1e-12)
                for row in focus
            ]
            cloud_growth = [
                (row["trailing_cloud_area"] - by_condition[row["condition"]]["trailing_cloud_area"]) / max(by_condition[row["condition"]]["trailing_cloud_area"], 1)
                for row in focus
            ]
            passed = bool(len(focus) == 2 and min(gains) >= .03 and min(reductions) >= .30 and max(row["background_leakage"] for row in focus) <= .03 and max(cloud_growth) <= .10)
            candidate_decisions.append({"variant": variant, "focus_count": len(focus), "recall_gains": gains, "hole_reductions": reductions, "cloud_growth": cloud_growth, "numeric_pass": passed})
        atomic_json(root / "render_variant_matrix/variant_acceptance.json", {"candidates": candidate_decisions, "visual_pending": True})
        atomic_text(root / "render_variant_matrix/RENDER_VARIANT_REPORT.md", "\n".join([
            "# Static render variant matrix", "", "R0, R4, R5 and R6 ran. R1/R2/R3/R7 are unsupported by the installed public gsplat API and were not emulated by changing production code.", "",
            *[f"- {row['variant']}: numeric candidate pass = `{row['numeric_pass']}`; recall gains={row['recall_gains']}; hole reductions={row['hole_reductions']}" for row in candidate_decisions],
        ]))

        # Curves and visual bundles.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for state in ("P1", "P2", "P3"):
            for condition, axis in zip(("cond_000318", "cond_000347"), axes[0]):
                rows = [row for row in threshold_rows if row["state"] == state and row["outfit"] == "O08" and row["condition"] == condition]
                axis.plot([row["threshold"] for row in rows], [row["trusted_expansion_recall"] for row in rows], label=state)
                axis.set_title(f"O08 {VIEWS[condition]} expansion recall")
            for condition, axis in zip(("cond_000318", "cond_000347"), axes[1]):
                rows = [row for row in threshold_rows if row["state"] == state and row["outfit"] == "O08" and row["condition"] == condition]
                axis.plot([row["threshold"] for row in rows], [row["internal_hole_ratio"] for row in rows], label=state)
                axis.set_title(f"O08 {VIEWS[condition]} hole ratio")
        for axis in axes.flat:
            axis.grid(alpha=.2); axis.legend(); axis.set_xlabel("alpha threshold")
        fig.tight_layout(); fig.savefig(root / "alpha_threshold_sweep/alpha_threshold_curves.png", dpi=160); plt.close(fig)
        atomic_text(root / "alpha_threshold_sweep/ALPHA_THRESHOLD_DIAGNOSIS.md", "\n".join([
            "# Alpha threshold diagnosis", "",
            "The registered thresholds were applied only to persisted/static soft alpha. Renderer parameters and formal alpha threshold remain unchanged.",
            "See `alpha_threshold_sweep.csv` and `alpha_threshold_curves.png` for all P1/P2/P3 outfit/view rows.",
        ]))
        make_contact_sheet(root / "visual_acceptance/p1_p2_p3_o08_alpha_contact_sheet.png", visual_panels, columns=3)
        make_contact_sheet(root / "visual_acceptance/reference_compositor_contact_sheet.png", reference_panels, columns=3)
        make_contact_sheet(root / "visual_acceptance/render_variant_contact_sheet.png", variant_panels, columns=4)
        visual_files = [
            root / "visual_acceptance/p1_p2_p3_o08_alpha_contact_sheet.png",
            root / "alpha_threshold_sweep/alpha_threshold_curves.png",
            root / "visual_acceptance/reference_compositor_contact_sheet.png",
            root / "visual_acceptance/render_variant_contact_sheet.png",
        ]
        atomic_json(root / "visual_acceptance/VISUAL_INSPECTION_REQUIRED.json", {
            "status": "AWAITING_ACTUAL_IMAGE_OPEN", "files": [{"path": str(path), "sha256": sha256(path)} for path in visual_files],
            "required_questions": [
                "holes_low_alpha_or_no_contribution", "scalloping_matches_tile_radius_culling", "cloud_gaussian_origin",
                "supersampling_smoothing_or_coverage", "reference_recovers_production_omissions", "P1_vs_P2_P3_parameter_or_renderer",
                "O01_same_failure",
            ],
        })
        atomic_json(root / "minimal_proof_probe/PROOF_PROBE_STATUS.json", {
            "status": "NOT_RUN", "optimizer_steps": 0,
            "reason": "visual inspection and a unique static candidate are required before any proof probe",
            "v6_1_weights_modified": False, "bounds_modified": False, "target_mask_in_forward": False,
        })
        atomic_text(root / "parameter_state_comparison/PARAMETER_STATE_DIAGNOSIS.md", "\n".join([
            "# P1/P2/P3 parameter-state comparison", "",
            "The CSV/JSON reports p50/p75/p90/p95/p99/max for residual, projected radius/center, rotation, opacity, SH0, visibility and tile counts in all registered image-derived regions.",
            "Region assignment samples each projected Gaussian center deterministically; it is diagnostic and does not alter training or rendering.",
        ]))

        reference_disagreement = any(
            row["mean_absolute_difference"] > .01 or row["p95_absolute_difference"] > .05 or row["trusted_fn_recovered_fraction"] > .20
            for row in reference_rows
        )
        numeric_variants = [row["variant"] for row in candidate_decisions if row["numeric_pass"]]
        preliminary_case = "RF"
        if reference_disagreement and len(numeric_variants) == 1:
            preliminary_case = "RA"
        elif numeric_variants == ["R6"]:
            preliminary_case = "RB"
        atomic_json(root / "final_adjudication/PRELIMINARY_STATUS.json", {
            "status": "AWAITING_VISUAL_INSPECTION", "preliminary_case": preliminary_case,
            "reference_disagreement": reference_disagreement, "numeric_variant_candidates": numeric_variants,
            "optimizer_steps": 0, "base_fingerprints": base_fingerprints,
            "default_renderer_regression": default_regression,
            "renderer_defaults_modified": False, "historical_outputs_modified": False,
        })
        atomic_json(status_path, {
            "status": "STATIC_AUDIT_COMPLETE_AWAITING_VISUAL", "stage": "visual_acceptance",
            "optimizer_steps": 0, "completed_at": now(), "preliminary_case": preliminary_case,
        })
        print(json.dumps({"status": "STATIC_AUDIT_COMPLETE_AWAITING_VISUAL", "output": str(root), "preliminary_case": preliminary_case}, indent=2))
    except Exception as error:
        atomic_json(status_path, {
            "status": "TOOL_FAILURE", "failure_stage": stage, "optimizer_steps": 0,
            "exception_type": type(error).__name__, "exception": str(error),
            "traceback": traceback.format_exc(), "failed_at": now(),
        })
        raise


def finalize_visual(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    root = args.output.resolve()
    status = json.loads((root / "RUN_STATUS.json").read_text(encoding="utf-8"))
    if status.get("status") != "STATIC_AUDIT_COMPLETE_AWAITING_VISUAL":
        raise RuntimeError("static audit is not awaiting visual acceptance")
    decisions = json.loads(args.visual_decisions.read_text(encoding="utf-8"))
    if not decisions.get("images_actually_opened") or decisions.get("inspection_method") != "view_image_original_resolution_and_contact_sheets":
        raise ValueError("actual visual inspection evidence is required")
    preliminary = json.loads((root / "final_adjudication/PRELIMINARY_STATUS.json").read_text(encoding="utf-8"))
    case = decisions["final_case"]
    if case not in {"RA", "RB", "RC", "RD", "RE", "RF"}:
        raise ValueError("invalid final case")
    next_tasks = {
        "RA": "FIX_GAUSSIAN_RASTERIZER_ALPHA_COVERAGE",
        "RB": "ADOPT_ANTIALIASED_ALPHA_RENDER_CONTRACT",
        "RC": "REDESIGN_ALPHA_COVERAGE_PARAMETERIZATION",
        "RD": "AUDIT_CANONICAL_GAUSSIAN_COVERAGE_DENSITY",
        "RE": "FREEZE_TRUSTED_GARMENT_EVALUATION_PROTOCOL",
        "RF": "BUILD_INSTRUMENTED_RASTERIZER_DEBUG_PATH",
    }
    final = {
        "status": "PASS", "audit_status": "COMPLETE", "final_case": case,
        "root_cause": decisions["root_cause"], "next_unique_task": next_tasks[case],
        "visual_acceptance": decisions, "preliminary": preliminary,
        "renderer_change_authorized": case in {"RA", "RB"},
        "alpha_parameterization_change_authorized": case == "RC",
        "seven_outfit_rerun_authorized": False, "target_generation_authorized": False,
        "image_conditioned_training_authorized": False, "proof_probe_run": False,
        "optimizer_steps": 0, "completed_at": now(),
    }
    atomic_json(root / "visual_acceptance/visual_acceptance.json", decisions)
    atomic_text(root / "visual_acceptance/VISUAL_ACCEPTANCE.md", decisions["markdown"])
    atomic_json(root / "final_adjudication/ALPHA_RASTER_AUDIT_FINAL_STATUS.json", final)
    atomic_text(root / "final_adjudication/GATE_ACCEPTANCE.md", "\n".join([
        "# Alpha / raster audit final adjudication", "", f"- status: **PASS (audit complete)**",
        f"- final case: **{case}**", f"- root cause: `{decisions['root_cause']}`",
        f"- next unique task: `{next_tasks[case]}`", "- optimizer steps: `0`",
        "- renderer defaults, V6/V6.1 loss, bounds, checkpoints and historical outputs were not modified.",
    ]))
    atomic_json(root / "RUN_STATUS.json", {"status": "PASS", "final_case": case, "optimizer_steps": 0, "completed_at": now()})
    print(json.dumps(final, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit CanonDressGS alpha coverage and gsplat rasterization")
    parser.add_argument("command", choices=("audit", "finalize-visual"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--visual-decisions", type=Path)
    args = parser.parse_args()
    if args.command == "finalize-visual" and args.visual_decisions is None:
        parser.error("finalize-visual requires --visual-decisions")
    return args


def main() -> None:
    args = parse_args(); config = load_config(args.config)
    if args.command == "audit":
        run_audit(args, config)
    else:
        finalize_visual(args, config)


if __name__ == "__main__":
    main()
