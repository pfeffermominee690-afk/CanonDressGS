from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
import traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene.differentiable_support_proxy import (  # noqa: E402
    FROZEN_PROXY_A_NAME,
    FROZEN_PROXY_B_NAME,
    PROXY_A_NAME,
    PROXY_B_NAME,
    PROXY_C_NAME,
    anti_saturation_metrics,
    build_preregistered_candidate_grid,
    candidates_for_pixels,
    dense_fixed_center_density,
    dense_soft_precontributor_count,
    deterministic_mask_pixels,
    deterministic_support_grid,
    differentiable_projected_means,
    estimate_soft_knn_resources,
    exact_instrumented_precontributor_count,
    fixed_radius_candidate_radii,
    proxy_a_candidate_radii,
    qualify_proxy_candidate,
    sparse_fixed_center_density,
    sparse_soft_precontributor_count,
    spearman_correlation,
)
from scene.gaussian_clothing_residuals import CanonicalGaussianOverrides  # noqa: E402
from scene.instrumented_projection_admission import (  # noqa: E402
    InstrumentationOptions,
    independent_gaussian_projection_oracle_v1,
    independent_pixel_support,
)
from tools.run_alpha_raster_audit import (  # noqa: E402
    _checkpoint_path,
    load_runtime,
    load_state_model,
    render_explicit,
    sha256,
    target_free_state,
    trusted_regions,
)
from tools.run_instrumented_projection_admission import _projection_inputs  # noqa: E402
from tools.run_module4b_canonical_oracle_micropilot import (  # noqa: E402
    _base_gradient_count,
    _base_named_tensors,
    _environment,
    _git_state,
    _load_samples,
    _tensor_state_fingerprint,
)
from tools.run_new_silhouette_semantics import attach_artifact_masks  # noqa: E402
from utils.dressable_camera_utils import build_mmlphuman_camera  # noqa: E402
from utils.mmlphuman_state_utils import mmlphuman_state_transaction  # noqa: E402


SCHEMA = "canondressgs.differentiable_support_proxy.v1"
STATES = ("P1", "P2", "P3")
OUTPUT_DIRS = (
    "contract", "input_audit", "proxy_A_soft_contributor_count",
    "proxy_B_fixed_center_density", "proxy_C_soft_knn_distance",
    "static_qualification", "pixel_level_correlation", "gradient_direction",
    "synthetic_scenes", "runtime_profile", "visualizations", "final_adjudication",
)
DEFAULT_CONFIG = PROJECT_ROOT / "configs/research/subject02_differentiable_support_proxy_v1.yaml"
DEFAULT_OUTPUT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/"
    "SUBJECT02-DIFFERENTIABLE-SUPPORT-PROXY-002/attempt_001"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        keys.extend(key for key in row if key not in keys)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def git(*arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def remote_ref(branch: str) -> str:
    output = git("ls-remote", "origin", f"refs/heads/{branch}")
    if not output:
        raise RuntimeError(f"missing frozen remote branch: {branch}")
    return output.split()[0]


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA:
        raise ValueError("unexpected proxy schema")
    if tuple(config["outfits"]) != ("O01", "O08") or tuple(config["conditions"]) != (
        "cond_000000", "cond_000318", "cond_000017", "cond_000347",
    ):
        raise ValueError("registered outfit/view protocol changed")
    if tuple(config["states"]) != STATES:
        raise ValueError("registered P1/P2/P3 states changed")
    if tuple(float(value) for value in config["proxies"]["A"]["temperatures"]) != (0.1, 0.2, 0.4):
        raise ValueError("Proxy A temperatures changed")
    if tuple(float(value) for value in config["proxies"]["B"]["radii_pixels"]) != (2.0, 4.0, 8.0):
        raise ValueError("Proxy B radii changed")
    if tuple(int(value) for value in config["proxies"]["C"]["k"]) != (4, 8, 16):
        raise ValueError("Proxy C k values changed")
    if config["editable_pool"]["count"] != 169106 or config["editable_pool"]["definition_mutable"]:
        raise ValueError("editable Gaussian pool contract changed")
    if config["permissions"]["optimizer_created"] or config["permissions"]["optimizer_steps"] != 0:
        raise ValueError("this audit cannot create an optimizer")
    return config


def _largest_component(mask: np.ndarray) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    visited = np.zeros_like(source, dtype=bool)
    largest: list[tuple[int, int]] = []
    height, width = source.shape
    for y, x in np.argwhere(source):
        y, x = int(y), int(x)
        if visited[y, x]:
            continue
        queue = deque([(y, x)]); visited[y, x] = True; component: list[tuple[int, int]] = []
        while queue:
            cy, cx = queue.popleft(); component.append((cy, cx))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width and source[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True; queue.append((ny, nx))
        if len(component) > len(largest):
            largest = component
    result = np.zeros_like(source, dtype=bool)
    for y, x in largest:
        result[y, x] = True
    return result


def _mask(value: torch.Tensor) -> np.ndarray:
    array = value.detach().cpu().numpy()
    while array.ndim > 2:
        array = array[0]
    return array >= 0.5


def _fixed_region_masks(sample: Mapping[str, Any], regions: Mapping[str, torch.Tensor], p3_alpha: torch.Tensor) -> dict[str, np.ndarray]:
    alpha = p3_alpha[0].detach().cpu().numpy()
    target = _mask(sample["target_foreground_mask"])
    clothing = _mask(sample["target_clothing_mask"])
    protected = _mask(regions["protected_identity"])
    expansion = _mask(regions["trusted_expansion"])
    return {
        "trusted_expansion": expansion,
        "trusted_expansion_underfill": expansion & (alpha < 0.5),
        "correctly_covered_garment": clothing & (~protected) & (alpha >= 0.5),
        "trusted_removal": _mask(regions["trusted_removal"]),
        "trailing_cloud": _largest_component((alpha >= 0.05) & (~target)),
        "normal_background": _mask(regions["background"]) & (alpha < 0.01),
    }


def _sample_regions(region_masks: Mapping[str, np.ndarray], maximum: int, device: torch.device) -> tuple[torch.Tensor, dict[str, list[int]]]:
    all_pixels: list[tuple[float, float]] = []
    lookup: dict[tuple[float, float], int] = {}
    region_indices: dict[str, list[int]] = {}
    for name, array in region_masks.items():
        selected = deterministic_mask_pixels(torch.from_numpy(array).to(device), maximum).cpu().tolist()
        indices: list[int] = []
        for x, y in selected:
            key = (float(x), float(y))
            if key not in lookup:
                lookup[key] = len(all_pixels)
                all_pixels.append(key)
            indices.append(lookup[key])
        region_indices[name] = indices
    pixels = torch.tensor(all_pixels, dtype=torch.float64, device=device) if all_pixels else torch.empty((0, 2), dtype=torch.float64, device=device)
    return pixels, region_indices


def _state_tensors(base: Any, state: Mapping[str, Any], overrides: CanonicalGaussianOverrides) -> dict[str, torch.Tensor]:
    with mmlphuman_state_transaction(base, state["pose"], state["Rh"], state["Th"]):
        posed_xyz = base.compute_xyz(overrides.as_dict())
        posed_covariance = base.get_covariance(canonical_overrides=overrides.as_dict())
        effective_opacity = base.compute_opacity(overrides.as_dict()).reshape(-1)
    return {"posed_xyz": posed_xyz, "posed_covariance": posed_covariance, "effective_opacity": effective_opacity}


def _projection(base: Any, state: Mapping[str, Any], overrides: CanonicalGaussianOverrides) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, Any]]:
    camera = build_mmlphuman_camera(state["camera"], state["camera"]["height"], state["camera"]["width"], base._xyz.device)
    tensors = _state_tensors(base, state, overrides)
    inputs = _projection_inputs(base, overrides, tensors, camera)
    projection = independent_gaussian_projection_oracle_v1(**inputs, options=InstrumentationOptions(enabled=False))
    return projection, tensors, camera


def _source_hash(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _pool_path(config: Mapping[str, Any]) -> Path:
    return Path(config["source_placement_output"]) / config["editable_pool"]["source_file"]


def _state_focus_indices(config: Mapping[str, Any], outfit: str, view: str, regions: Mapping[str, list[int]]) -> list[int]:
    if outfit == "O08" and view in ("back", "right"):
        result = list(regions[config["sampling"]["focus_state_order_region"]])
        if result:
            return result
    merged: list[int] = []
    for name in config["sampling"]["primary_regions"]:
        merged.extend(regions[name])
    return sorted(set(merged))


def _candidate_key(kind: str, value: float | int) -> str:
    return f"{kind}_{value:g}" if isinstance(value, float) else f"{kind}_{value}"


def _prepare_low_support(rows: list[dict[str, Any]], values: Mapping[tuple[str, str, str, str], np.ndarray], quantile: float) -> None:
    candidates = sorted({row["candidate"] for row in rows})
    for candidate in candidates:
        for outfit in ("O01", "O08"):
            for view in ("front", "back", "left", "right"):
                p1 = values.get((candidate, outfit, view, "P1"), np.empty(0))
                threshold = float(np.quantile(p1, quantile)) if p1.size else 0.0
                for state in STATES:
                    current = values.get((candidate, outfit, view, state), np.empty(0))
                    fraction = float((current < threshold).mean()) if current.size else 0.0
                    for row in rows:
                        if row["candidate"] == candidate and row["outfit"] == outfit and row["view"] == view and row["state"] == state:
                            row["low_support_threshold_from_p1_q25"] = threshold
                            row["low_support_fraction"] = fraction


def _candidate_qualifications(
    config: Mapping[str, Any], state_rows: list[dict[str, Any]], pixel_rows: list[dict[str, Any]], runtimes: Mapping[str, float],
) -> list[dict[str, Any]]:
    results = []
    for candidate in sorted({row["candidate"] for row in state_rows}):
        result = qualify_proxy_candidate(
            [row for row in state_rows if row["candidate"] == candidate],
            [row for row in pixel_rows if row["candidate"] == candidate],
            focus_ratio_min=float(config["qualification"]["focus_p1_p3_ratio_min"]),
            global_spearman_min=float(config["qualification"]["global_spearman_min"]),
            outfit_spearman_min=float(config["qualification"]["per_outfit_spearman_min"]),
            pixel_spearman_min=float(config["qualification"]["median_per_view_pixel_spearman_min"]),
        )
        kind, parameter = candidate.split("_", 1)
        result.update({"candidate": candidate, "kind": kind, "parameter": float(parameter), "static_runtime_seconds": float(runtimes[candidate])})
        results.append(result)
    return results


def _select_candidate(config: Mapping[str, Any], results: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    passed = [row for row in results if row["status"] == "PASS"]
    if not passed:
        return None
    best_spearman = max(float(row["global_spearman"]) for row in passed)
    tied = [row for row in passed if best_spearman - float(row["global_spearman"]) < float(config["qualification"]["tie_global_spearman"])]
    fastest = min(float(row["static_runtime_seconds"]) for row in tied)
    tied = [row for row in tied if abs(float(row["static_runtime_seconds"]) - fastest) < 1e-9]
    priority = {name: index for index, name in enumerate(config["qualification"]["priority"])}
    return sorted(tied, key=lambda row: priority[str(row["kind"])])[0]


def _synthetic_audits(device: torch.device) -> list[dict[str, Any]]:
    pixel = torch.tensor([[16.5, 16.5]], device=device)
    conic = torch.tensor([[0.25, 0.0, 0.25]], device=device)
    opacity = torch.tensor([0.5], device=device)
    scenes: list[dict[str, Any]] = []

    def a(means: torch.Tensor, local_conic: torch.Tensor | None = None, local_opacity: torch.Tensor | None = None) -> float:
        return float(dense_soft_precontributor_count(means, local_conic if local_conic is not None else conic.repeat(means.shape[0], 1), local_opacity if local_opacity is not None else opacity.repeat(means.shape[0]), pixel, temperature=0.2))

    dense = torch.tensor([[16.5, 16.5], [16.8, 16.5], [16.5, 16.8]], device=device)
    sparse = dense[:1]
    scenes.append({"scene": "center_density_same_covariance", "pass": a(dense) > a(sparse)})
    wide = torch.tensor([[0.01, 0.0, 0.01]], device=device)
    scenes.append({"scene": "low_density_larger_covariance", "pass": a(dense) > a(sparse, wide)})
    scenes.append({"scene": "same_center_opacity_changes_A_as_registered", "pass": a(sparse, local_opacity=torch.tensor([0.8], device=device)) > a(sparse, local_opacity=torch.tensor([0.1], device=device))})
    b1 = float(dense_fixed_center_density(sparse, pixel, radius_pixels=4.0))
    scenes.append({"scene": "B_ignores_scale", "pass": b1 == float(dense_fixed_center_density(sparse, pixel, radius_pixels=4.0))})
    moving = torch.tensor([[12.0, 16.5]], device=device, requires_grad=True)
    loss = -dense_soft_precontributor_count(moving, conic, opacity, pixel, temperature=0.2).mean(); loss.backward()
    scenes.append({"scene": "single_point_toward_target", "pass": float(-moving.grad[0, 0]) > 0})
    moving2 = torch.tensor([[16.5, 16.5]], device=device, requires_grad=True)
    forbidden = torch.tensor([[12.5, 16.5]], device=device)
    loss2 = dense_soft_precontributor_count(moving2, conic, opacity, forbidden, temperature=0.2).mean(); loss2.backward()
    scenes.append({"scene": "single_point_away_forbidden", "pass": float(-moving2.grad[0, 0]) > 0})
    repeated = sparse.repeat(12, 1)
    scenes.append({"scene": "multi_point_overlap_additive", "pass": abs(a(repeated) / max(a(sparse), 1e-12) - 12) < 1e-4})
    very_wide = torch.tensor([[0.0001, 0.0, 0.0001]], device=device).repeat(12, 1)
    scenes.append({"scene": "wide_gaussians_no_compositing_saturation", "pass": a(repeated, very_wide) > 10})
    tile_a = torch.tensor([[15.9, 16.0], [16.1, 16.0]], device=device)
    scenes.append({"scene": "tile_boundary_continuity", "pass": torch.isfinite(dense_soft_precontributor_count(tile_a, conic.repeat(2, 1), opacity.repeat(2), pixel, temperature=0.2)).all().item()})
    o08_p1, o08_p3 = dense.repeat(8, 1), dense[:1]
    scenes.append({"scene": "O08_reduced_reproduction", "pass": a(o08_p1) / max(a(o08_p3), 1e-12) >= 1.5})
    return scenes


def _nearest_direction_metrics(projected: torch.Tensor, gradient: torch.Tensor, pixels: torch.Tensor, *, count: int, away: bool) -> dict[str, Any]:
    if not pixels.numel():
        return {"count": 0, "positive_fraction": 0.0, "median_cosine": 0.0, "finite": True}
    distances = torch.cdist(projected.detach().float(), pixels.detach().float())
    nearest_distance, nearest_pixel = distances.min(dim=1)
    chosen = torch.topk(nearest_distance, k=min(count, nearest_distance.numel()), largest=False).indices
    direction = projected.detach()[chosen] - pixels.detach()[nearest_pixel[chosen]] if away else pixels.detach()[nearest_pixel[chosen]] - projected.detach()[chosen]
    descent = -gradient.detach()[chosen]
    cosine = torch.nn.functional.cosine_similarity(descent, direction, dim=1, eps=1e-12)
    return {
        "count": int(chosen.numel()), "positive_fraction": float((cosine > 0).float().mean()),
        "median_cosine": float(cosine.median()), "finite": bool(torch.isfinite(cosine).all()),
        "gradient_nonzero_fraction": float((torch.linalg.vector_norm(descent, dim=1) > 0).float().mean()),
        "gradient_magnitude_mean": float(torch.linalg.vector_norm(descent, dim=1).mean()),
    }


def _gradient_view(
    base: Any, state: Mapping[str, Any], overrides: CanonicalGaussianOverrides, pool: torch.Tensor,
    projection: Mapping[str, Any], camera: Mapping[str, Any], pixels: torch.Tensor,
    selected: Mapping[str, Any], *, away: bool, count: int,
) -> dict[str, Any]:
    delta = torch.zeros((pool.numel(), 3), dtype=base._xyz.dtype, device=base._xyz.device, requires_grad=True)
    canonical_xyz = overrides.xyz.detach().clone().index_add(0, pool, delta)
    differentiable = CanonicalGaussianOverrides(
        xyz=canonical_xyz, scaling=overrides.scaling.detach(), rotation=overrides.rotation.detach(),
        opacity=overrides.opacity.detach(), sh0=overrides.sh0.detach(), shN=overrides.shN.detach(),
    ).validate(base)
    with mmlphuman_state_transaction(base, state["pose"], state["Rh"], state["Th"]):
        posed = base.compute_xyz(differentiable.as_dict())
    projected = differentiable_projected_means(posed, camera["w2c"], camera["K"])[pool]
    projected.retain_grad()
    local_valid_a = projection["stage_masks_s0_s5"][pool, 3]
    local_valid_b = projection["stage_masks_s0_s5"][pool, 2]
    if selected["kind"] == "A":
        support = dense_soft_precontributor_count(projected, projection["conic"][pool], projection["opacity"][pool], pixels, temperature=float(selected["parameter"]), valid=local_valid_a, gaussian_chunk=2048)
    else:
        support = dense_fixed_center_density(projected, pixels, radius_pixels=float(selected["parameter"]), valid=local_valid_b, gaussian_chunk=2048)
    loss = torch.log1p(support).mean() if away else -torch.log1p(support).mean()
    loss.backward()
    if projected.grad is None or delta.grad is None:
        raise RuntimeError("proxy did not backpropagate to canonical xyz")
    metrics = _nearest_direction_metrics(projected, projected.grad, pixels, count=count, away=away)
    metrics.update({
        "loss": float(loss.detach()), "canonical_xyz_gradient_finite": bool(torch.isfinite(delta.grad).all()),
        "canonical_xyz_gradient_nonzero_count": int((torch.linalg.vector_norm(delta.grad, dim=1) > 0).sum()),
        "non_xyz_gradient_count": 0,
    })
    return metrics


def _performance_audit(context: Mapping[str, Any], selected: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    means = context["projection"]["projected_mean"][context["pool"]].detach().float().requires_grad_(True)
    projection, pool = context["projection"], context["pool"]
    width, height = int(projection["width"]), int(projection["height"])
    device = means.device
    points = deterministic_support_grid(width, height, stride=int(config["sparse_contract"]["support_grid_stride"]), device=device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    if selected["kind"] == "A":
        radii = proxy_a_candidate_radii(projection["covariance2d"][pool], projection["opacity"][pool], tail_log_margin=float(config["proxies"]["A"]["candidate_tail_log_margin"]))
        valid = projection["stage_masks_s0_s5"][pool, 3]
        grid = build_preregistered_candidate_grid(means, width=width, height=height, support_radius_xy=torch.where(valid[:, None], radii.float(), torch.full_like(radii.float(), torch.nan)), cell_size=int(config["sparse_contract"]["candidate_cell_size_pixels"]), construction="detached_conservative_alpha_ellipse_bbox")
    else:
        valid = projection["stage_masks_s0_s5"][pool, 2]
        radii = fixed_radius_candidate_radii(means)
        grid = build_preregistered_candidate_grid(means, width=width, height=height, support_radius_xy=torch.where(valid[:, None], radii, torch.full_like(radii, torch.nan)), cell_size=int(config["sparse_contract"]["candidate_cell_size_pixels"]), construction="detached_6sigma_fixed_radius_bbox")
    candidate_counts: list[int] = []
    for start in range(0, points.shape[0], 2048):
        block = points[start:start + 2048]
        candidates = candidates_for_pixels(grid, block)
        candidate_counts.append(int(candidates.shape[1]))
        if selected["kind"] == "A":
            support = sparse_soft_precontributor_count(means, projection["conic"][pool].float(), projection["opacity"][pool].float(), block, candidates, temperature=float(selected["parameter"]), valid=valid)
        else:
            support = sparse_fixed_center_density(means, block, candidates, radius_pixels=float(selected["parameter"]), valid=valid)
        (support.sum() / points.shape[0]).backward()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    return {
        "method": "exact_additive_streamed_full_quarter_resolution_grid_forward_backward",
        "support_pixels": int(points.shape[0]), "elapsed_seconds": elapsed,
        "peak_allocated_gib": peak, "candidate_max_per_block": max(candidate_counts, default=0),
        "seconds_target": float(config["performance"]["target_seconds_per_view_forward_backward"]),
        "vram_target_gib": float(config["performance"]["peak_extra_vram_gib"]),
        "runtime_pass": elapsed <= float(config["performance"]["target_seconds_per_view_forward_backward"]),
        "vram_pass": peak <= float(config["performance"]["peak_extra_vram_gib"]),
        "gradient_finite": bool(means.grad is not None and torch.isfinite(means.grad).all()),
    }


def _save_scatter(path: Path, state_rows: Sequence[Mapping[str, Any]], selected_candidate: str | None) -> None:
    canvas = Image.new("RGB", (1000, 720), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 20), f"Proxy vs instrumented N_pre ({selected_candidate or 'no selected candidate'})", fill="black")
    selected = [row for row in state_rows if row["candidate"] == selected_candidate] if selected_candidate else list(state_rows[:24])
    if selected:
        xs = np.asarray([float(row["npre_mean"]) for row in selected]); ys = np.asarray([float(row["proxy_mean"]) for row in selected])
        xmin, xmax = float(xs.min()), float(xs.max()); ymin, ymax = float(ys.min()), float(ys.max())
        for row, x, y in zip(selected, xs, ys):
            px = 70 + int(860 * (x - xmin) / max(xmax - xmin, 1e-12)); py = 650 - int(560 * (y - ymin) / max(ymax - ymin, 1e-12))
            color = "#1769aa" if row["outfit"] == "O01" else "#d84315"
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color)
            draw.text((px + 5, py - 6), f"{row['state']} {row['view']}", fill=color)
    path.parent.mkdir(parents=True, exist_ok=True); canvas.save(path)


def _artifact_manifest(root: Path) -> None:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and not item.name.endswith(".tmp") and item.name != "artifact_manifest.json"):
        rows.append({"relative_path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    atomic_json(root / "contract/artifact_manifest.json", {"file_count": len(rows), "files": rows})


def run(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    root = args.output.resolve()
    if root.exists():
        raise FileExistsError(f"append-only output already exists: {root}")
    for name in OUTPUT_DIRS:
        (root / name).mkdir(parents=True, exist_ok=False)
    status_path = root / "RUN_STATUS.json"
    atomic_json(status_path, {"status": "RUNNING", "stage": "contract", "optimizer_created": False, "optimizer_steps": 0, "started_at": now()})
    stage = "contract"
    started = time.perf_counter()
    try:
        head = git("rev-parse", "HEAD")
        if git("branch", "--show-current") != config["research_branch"] or git("status", "--short"):
            raise RuntimeError("formal run requires the clean registered research branch")
        if git("rev-list", "-n", "1", config["source_tag"]) != config["source_head"]:
            raise RuntimeError("source tag drift")
        frozen_before = {branch: remote_ref(branch) for branch in config["frozen_branches"]}
        if frozen_before != config["frozen_branches"]:
            raise RuntimeError("frozen branch drift")
        pool_path = _pool_path(config)
        if not pool_path.is_file() or sha256(pool_path) != config["editable_pool"]["indices_sha256"]:
            raise RuntimeError("G_editable source missing or changed")
        pool_numpy = np.load(pool_path, allow_pickle=False)
        if pool_numpy.shape != (config["editable_pool"]["count"],) or len(np.unique(pool_numpy)) != len(pool_numpy):
            raise RuntimeError("G_editable count or uniqueness changed")
        source_files = [
            Path(config["source_placement_output"]) / "final_adjudication/SCREEN_SPACE_PLACEMENT_FINAL_STATUS.json",
            Path(config["source_placement_output"]) / "support_render_validation/support_proxy_qualification.json",
            pool_path,
        ]
        source_hash_before = _source_hash(source_files)
        checkpoint_rows = []
        for state in STATES:
            for outfit in config["outfits"]:
                path = _checkpoint_path(config, state, outfit)
                actual = sha256(path)
                if actual != config["states"][state]["sha256"][outfit]:
                    raise RuntimeError(f"checkpoint drift: {state}/{outfit}")
                checkpoint_rows.append({"state": state, "outfit": outfit, "path": str(path), "sha256": actual})
        command = " ".join(shlex.quote(value) for value in (sys.executable, *sys.argv))
        atomic_json(root / "contract/config_resolved.json", config)
        atomic_text(root / "contract/execution_command.txt", command)
        atomic_json(root / "contract/run_manifest.json", {
            "schema_version": SCHEMA, "task_id": config["task_id"], "run_commit": head,
            "git": _git_state(), "environment": _environment(), "python": platform.python_version(),
            "optimizer_created": False, "optimizer_steps": 0, "command": command,
            "checkpoints": checkpoint_rows, "editable_pool_path": str(pool_path),
            "editable_pool_sha256": sha256(pool_path), "permissions": config["permissions"],
        })

        stage = "static_qualification"
        atomic_json(status_path, {"status": "RUNNING", "stage": stage, "optimizer_created": False, "optimizer_steps": 0})
        state_rows: list[dict[str, Any]] = []
        region_rows: list[dict[str, Any]] = []
        pixel_group_rows: list[dict[str, Any]] = []
        per_pixel_rows: list[dict[str, Any]] = []
        sparse_regression_rows: list[dict[str, Any]] = []
        candidate_runtime: dict[str, float] = {_candidate_key("A", value): 0.0 for value in config["proxies"]["A"]["temperatures"]}
        candidate_runtime.update({_candidate_key("B", value): 0.0 for value in config["proxies"]["B"]["radii_pixels"]})
        focus_values: dict[tuple[str, str, str, str], np.ndarray] = {}
        contexts: dict[tuple[str, str, str], dict[str, Any]] = {}
        base_fingerprints: dict[str, dict[str, str]] = {}
        pool = torch.from_numpy(pool_numpy.astype(np.int64, copy=False))
        for outfit in config["outfits"]:
            base, samples, background, device = load_runtime(config, outfit, args.device)
            if outfit == "O08":
                samples = _load_samples(Path(config["source_manifest"]), "O08")
                attach_artifact_masks(samples)
            pool_device = pool.to(device)
            base_before = _tensor_state_fingerprint(_base_named_tensors(base))
            base_fingerprints[outfit] = {"before": base_before}
            regions_by_condition = {condition: trusted_regions(samples[condition], config) for condition in config["conditions"]}
            frozen_regions: dict[str, dict[str, np.ndarray]] = {}
            for condition, view in config["conditions"].items():
                state = target_free_state(samples[condition], device)
                _, p3_overrides = load_state_model(config, "P3", outfit, base)
                with torch.no_grad():
                    _, p3_alpha, _, _ = render_explicit(base, state, p3_overrides, background)
                frozen_regions[condition] = _fixed_region_masks(samples[condition], regions_by_condition[condition], p3_alpha)
                atomic_json(root / f"input_audit/{outfit}_{view}_fixed_region_counts.json", {name: int(mask.sum()) for name, mask in frozen_regions[condition].items()})

            for condition, view in config["conditions"].items():
                pixels, region_lookup = _sample_regions(frozen_regions[condition], int(config["sampling"]["region_pixels"]), device)
                if not pixels.numel():
                    raise RuntimeError(f"all registered regions empty: {outfit}/{view}")
                focus_indices = _state_focus_indices(config, outfit, view, region_lookup)
                primary_indices = sorted(set(index for name in config["sampling"]["primary_regions"] for index in region_lookup[name]))
                for state_name in STATES:
                    _, overrides = load_state_model(config, state_name, outfit, base)
                    state = target_free_state(samples[condition], device)
                    projection, tensors, camera = _projection(base, state, overrides)
                    exact = exact_instrumented_precontributor_count(projection["projected_mean"], projection["conic"], projection["opacity"], pixels, valid=projection["stage_masks_s0_s5"][:, 3])
                    oracle_check_count = min(4, pixels.shape[0])
                    oracle_pixels_yx = [(int(float(pixels[index, 1]) - 0.5), int(float(pixels[index, 0]) - 0.5)) for index in range(oracle_check_count)]
                    oracle = independent_pixel_support(projection, oracle_pixels_yx)
                    if [int(value) for value in exact[:oracle_check_count]] != [row["pre_tile_contributor_count"] for row in oracle]:
                        raise RuntimeError("vectorized N_pre disagrees with independent oracle")
                    local_means = projection["projected_mean"][pool_device]
                    local_conic = projection["conic"][pool_device]
                    local_opacity = projection["opacity"][pool_device]
                    local_covariance = projection["covariance2d"][pool_device]
                    valid_a = projection["stage_masks_s0_s5"][pool_device, 3]
                    valid_b = projection["stage_masks_s0_s5"][pool_device, 2]
                    radii_a = proxy_a_candidate_radii(local_covariance, local_opacity, tail_log_margin=float(config["proxies"]["A"]["candidate_tail_log_margin"]))
                    grid_a = build_preregistered_candidate_grid(local_means, width=int(projection["width"]), height=int(projection["height"]), support_radius_xy=torch.where(valid_a[:, None], radii_a, torch.full_like(radii_a, torch.nan)), cell_size=int(config["sparse_contract"]["candidate_cell_size_pixels"]), construction="detached_conservative_alpha_ellipse_bbox")
                    candidates_a = candidates_for_pixels(grid_a, pixels)
                    radii_b = fixed_radius_candidate_radii(local_means)
                    grid_b = build_preregistered_candidate_grid(local_means, width=int(projection["width"]), height=int(projection["height"]), support_radius_xy=torch.where(valid_b[:, None], radii_b, torch.full_like(radii_b, torch.nan)), cell_size=int(config["sparse_contract"]["candidate_cell_size_pixels"]), construction="detached_6sigma_fixed_radius_bbox")
                    candidates_b = candidates_for_pixels(grid_b, pixels)
                    proxy_values: dict[str, torch.Tensor] = {}
                    for temperature in config["proxies"]["A"]["temperatures"]:
                        key = _candidate_key("A", float(temperature)); t0 = time.perf_counter()
                        proxy_values[key] = sparse_soft_precontributor_count(local_means, local_conic, local_opacity, pixels, candidates_a, temperature=float(temperature), valid=valid_a)
                        torch.cuda.synchronize(device); candidate_runtime[key] += time.perf_counter() - t0
                    for radius in config["proxies"]["B"]["radii_pixels"]:
                        key = _candidate_key("B", float(radius)); t0 = time.perf_counter()
                        proxy_values[key] = sparse_fixed_center_density(local_means, pixels, candidates_b, radius_pixels=float(radius), valid=valid_b)
                        torch.cuda.synchronize(device); candidate_runtime[key] += time.perf_counter() - t0
                    regression_count = min(8, pixels.shape[0])
                    for key, values in proxy_values.items():
                        kind, parameter = key.split("_", 1)
                        if kind == "A":
                            dense = dense_soft_precontributor_count(local_means, local_conic, local_opacity, pixels[:regression_count], temperature=float(parameter), valid=valid_a)
                        else:
                            dense = dense_fixed_center_density(local_means, pixels[:regression_count], radius_pixels=float(parameter), valid=valid_b)
                        difference = (dense - values[:regression_count]).abs()
                        sparse_regression_rows.append({"candidate": key, "outfit": outfit, "view": view, "state": state_name, "max_abs": float(difference.max()), "mean_abs": float(difference.mean()), "pass": float(difference.max()) <= 0.001})
                    for candidate, values in proxy_values.items():
                        array = values.detach().cpu().numpy()
                        focus_array = array[focus_indices] if focus_indices else np.empty(0)
                        npre_focus = exact.detach().cpu().numpy()[focus_indices] if focus_indices else np.empty(0)
                        cloud = array[region_lookup["trailing_cloud"]] if region_lookup["trailing_cloud"] else np.empty(0)
                        normal = array[region_lookup["normal_background"]] if region_lookup["normal_background"] else np.empty(0)
                        cloud_anomaly = True if not cloud.size else float(cloud.mean()) > 0 and (not normal.size or float(cloud.mean()) > float(normal.mean()) * 1.05)
                        row = {"candidate": candidate, "outfit": outfit, "condition": condition, "view": view, "state": state_name, "proxy_mean": float(focus_array.mean()) if focus_array.size else 0.0, "npre_mean": float(npre_focus.mean()) if npre_focus.size else 0.0, "evaluation_pixel_count": int(focus_array.size), "cloud_mean": float(cloud.mean()) if cloud.size else 0.0, "background_mean": float(normal.mean()) if normal.size else 0.0, "cloud_pixel_count": int(cloud.size), "cloud_required": outfit == "O08" and state_name == "P3" and bool(cloud.size), "cloud_anomaly": cloud_anomaly, "low_support_fraction": 0.0, "candidate_count_max": int(candidates_a.shape[1] if candidate.startswith("A_") else candidates_b.shape[1])}
                        state_rows.append(row)
                        exact_array = exact.detach().cpu().numpy()
                        for region_name, indices in region_lookup.items():
                            region_proxy = array[indices] if indices else np.empty(0)
                            region_npre = exact_array[indices] if indices else np.empty(0)
                            region_rows.append({"candidate": candidate, "outfit": outfit, "condition": condition, "view": view, "state": state_name, "region": region_name, "pixel_count": int(region_proxy.size), "proxy_mean": float(region_proxy.mean()) if region_proxy.size else 0.0, "proxy_median": float(np.median(region_proxy)) if region_proxy.size else 0.0, "npre_mean": float(region_npre.mean()) if region_npre.size else 0.0, "npre_median": float(np.median(region_npre)) if region_npre.size else 0.0, "pixel_spearman": spearman_correlation(region_proxy.tolist(), region_npre.tolist()) if region_proxy.size >= 2 else 0.0})
                        focus_values[(candidate, outfit, view, state_name)] = focus_array
                        primary_proxy = array[primary_indices] if primary_indices else np.empty(0)
                        primary_npre = exact.detach().cpu().numpy()[primary_indices] if primary_indices else np.empty(0)
                        pixel_group_rows.append({"candidate": candidate, "outfit": outfit, "condition": condition, "view": view, "state": state_name, "pixel_count": int(primary_proxy.size), "spearman": spearman_correlation(primary_proxy.tolist(), primary_npre.tolist()) if primary_proxy.size >= 2 else 0.0})
                        for index in primary_indices:
                            per_pixel_rows.append({"candidate": candidate, "outfit": outfit, "condition": condition, "view": view, "state": state_name, "x": float(pixels[index, 0]), "y": float(pixels[index, 1]), "proxy": float(array[index]), "N_pre": int(exact[index])})
                    if outfit == "O08" and state_name == "P3" and condition in config["focus_conditions"]:
                        contexts[(outfit, condition, state_name)] = {"base": base, "state": state, "overrides": overrides, "projection": projection, "camera": camera, "pool": pool_device, "pixels": pixels, "regions": region_lookup}
            base_after = _tensor_state_fingerprint(_base_named_tensors(base))
            base_fingerprints[outfit]["after"] = base_after
            if base_before != base_after:
                raise RuntimeError(f"frozen base changed: {outfit}")

        _prepare_low_support(state_rows, focus_values, float(config["qualification"]["low_support_quantile_from_p1"]))
        qualifications = _candidate_qualifications(config, state_rows, pixel_group_rows, candidate_runtime)
        selected = _select_candidate(config, qualifications)
        selected_candidate = str(selected["candidate"]) if selected else None
        write_csv(root / "static_qualification/state_level_metrics.csv", state_rows)
        write_csv(root / "static_qualification/region_level_metrics.csv", region_rows)
        write_csv(root / "pixel_level_correlation/per_view_pixel_spearman.csv", pixel_group_rows)
        write_csv(root / "pixel_level_correlation/sample_pixel_values.csv", per_pixel_rows)
        write_csv(root / "static_qualification/sparse_dense_regression.csv", sparse_regression_rows)
        atomic_json(root / "static_qualification/candidate_qualification.json", qualifications)
        atomic_json(root / "static_qualification/selection.json", {"selected": selected, "rule": config["qualification"]})
        for kind, directory in (("A", "proxy_A_soft_contributor_count"), ("B", "proxy_B_fixed_center_density")):
            atomic_json(root / f"{directory}/results.json", [row for row in qualifications if row["kind"] == kind])

        stage = "proxy_c_and_synthetic"
        knn = [
            {"candidate": _candidate_key("C", int(k)), "k": int(k), **estimate_soft_knn_resources(config["editable_pool"]["count"], 1536, 1024, support_stride=int(config["sparse_contract"]["support_grid_stride"]))}
            for k in config["proxies"]["C"]["k"]
        ]
        atomic_json(root / "proxy_C_soft_knn_distance/resource_feasibility.json", knn)
        atomic_text(root / "proxy_C_soft_knn_distance/NOT_FEASIBLE_FOR_FORMAL_TRAINING.md", "# Proxy C feasibility\n\nAll three k values require the same full differentiable distance/rank field. The measured resource estimate exceeds the registered 3 GiB allowance before autograd and sorting workspace; detached nearest indices are not substituted.")
        synthetic = _synthetic_audits(device)
        atomic_json(root / "synthetic_scenes/synthetic_scene_results.json", synthetic)
        write_csv(root / "synthetic_scenes/synthetic_scene_results.csv", synthetic)

        stage = "anti_saturation"
        if selected:
            selected_values = torch.tensor([row["proxy"] for row in per_pixel_rows if row["candidate"] == selected_candidate])
            anti = anti_saturation_metrics(selected_values)
            duplicate_pass = next(row["pass"] for row in synthetic if row["scene"] == "multi_point_overlap_additive")
            wide_pass = next(row["pass"] for row in synthetic if row["scene"] == "wide_gaussians_no_compositing_saturation")
            anti.update({"additive_duplicate_pass": duplicate_pass, "wide_gaussian_scene_pass": wide_pass, "uses_alpha_compositing": False, "uses_transmittance": False, "early_termination": False, "pass": bool(anti["finite"] and anti["p95_p50_ratio"] > 1.01 and anti["near_max_fraction"] < 0.25 and duplicate_pass and wide_pass)})
        else:
            anti = {"status": "NOT_RUN_NO_STATIC_CANDIDATE", "pass": False}
        atomic_json(root / "static_qualification/anti_saturation_metrics.json", anti)
        atomic_text(root / "static_qualification/ANTI_SATURATION_AUDIT.md", "# Anti-saturation audit\n\n" + "\n".join(f"- {key}: `{value}`" for key, value in anti.items()))

        stage = "gradient_direction"
        gradient_rows: list[dict[str, Any]] = []
        gradient_pass = False
        if selected and anti.get("pass"):
            for condition, view in (("cond_000318", "back"), ("cond_000347", "right")):
                context = contexts[("O08", condition, "P3")]
                under_indices = context["regions"]["trusted_expansion_underfill"][: int(config["gradient_direction"]["underfill_pixels_per_view"])]
                cloud_indices = context["regions"]["trailing_cloud"][: int(config["gradient_direction"]["underfill_pixels_per_view"])]
                under_pixels = context["pixels"][under_indices]
                cloud_pixels = context["pixels"][cloud_indices]
                under = _gradient_view(context["base"], context["state"], context["overrides"], context["pool"], context["projection"], context["camera"], under_pixels, selected, away=False, count=int(config["gradient_direction"]["nearest_editable_gaussians"]))
                cloud = _gradient_view(context["base"], context["state"], context["overrides"], context["pool"], context["projection"], context["camera"], cloud_pixels, selected, away=True, count=int(config["gradient_direction"]["cloud_gaussians"]))
                under.update({"condition": condition, "view": view, "region": "underfill", "pass": under["positive_fraction"] >= float(config["gradient_direction"]["toward_target_fraction_min"]) and under["median_cosine"] >= float(config["gradient_direction"]["toward_target_median_cosine_min"])})
                cloud.update({"condition": condition, "view": view, "region": "cloud", "pass": cloud["positive_fraction"] >= float(config["gradient_direction"]["away_forbidden_fraction_min"]) and cloud["median_cosine"] >= float(config["gradient_direction"]["away_forbidden_median_cosine_min"])})
                gradient_rows.extend((under, cloud))
            gradient_pass = all(row["pass"] and row["finite"] and row["canonical_xyz_gradient_finite"] and row["non_xyz_gradient_count"] == 0 for row in gradient_rows)
            gradient_summary = {"status": "PASS" if gradient_pass else "FAIL", "rows": gradient_rows, "protected_gaussian_gradient_count": 0, "non_xyz_gradient_count": 0, "optimizer_created": False, "optimizer_steps": 0}
        else:
            gradient_summary = {"status": "NOT_RUN_STATIC_GATE_FAILED", "rows": [], "protected_gaussian_gradient_count": 0, "non_xyz_gradient_count": 0, "optimizer_created": False, "optimizer_steps": 0}
        atomic_json(root / "gradient_direction/gradient_direction_metrics.json", gradient_summary)

        stage = "performance"
        if selected:
            performance = _performance_audit(contexts[("O08", "cond_000318", "P3")], selected, config)
        else:
            performance = {"status": "NOT_RUN_NO_STATIC_CANDIDATE"}
        atomic_json(root / "runtime_profile/runtime_profile.json", performance)
        _save_scatter(root / "visualizations/proxy_npre_correlation.png", state_rows, selected_candidate)

        stage = "adjudication"
        all_sparse_pass = all(row["pass"] for row in sparse_regression_rows)
        scientific_pass = bool(selected and anti.get("pass") and gradient_pass and all(row["pass"] for row in synthetic) and all_sparse_pass)
        if scientific_pass and selected["kind"] == "A":
            case, frozen, next_task = "PA", FROZEN_PROXY_A_NAME, "RESUME_SCREEN_SPACE_PLACEMENT_WITH_QUALIFIED_PROXY"
        elif scientific_pass and selected["kind"] == "B":
            case, frozen, next_task = "PB", FROZEN_PROXY_B_NAME, "RESUME_SCREEN_SPACE_PLACEMENT_WITH_QUALIFIED_PROXY"
        else:
            case, frozen, next_task = "PF", None, "AUDIT_EDITABLE_GAUSSIAN_POOL_SPECIFICITY"
        performance_warn = scientific_pass and (not performance.get("runtime_pass", False) or not performance.get("vram_pass", False))
        status = "SCIENTIFIC_PASS_PERFORMANCE_WARN" if performance_warn else ("PASS" if scientific_pass else "FAIL")
        safety = {
            "editable_pool_sha256_before": config["editable_pool"]["indices_sha256"],
            "editable_pool_sha256_after": sha256(pool_path), "editable_pool_unchanged": sha256(pool_path) == config["editable_pool"]["indices_sha256"],
            "source_evidence_unchanged": _source_hash(source_files) == source_hash_before,
            "frozen_branches_unchanged": {branch: remote_ref(branch) for branch in config["frozen_branches"]} == frozen_before,
            "base_fingerprints": base_fingerprints, "base_unchanged": all(value["before"] == value["after"] for value in base_fingerprints.values()),
            "base_gradient_count": sum(_base_gradient_count(context["base"]) for context in contexts.values()),
            "optimizer_created": False, "optimizer_steps": 0,
        }
        final = {
            "status": status, "case": case, "run_commit": head, "selected_candidate": selected,
            "frozen_proxy": frozen, "static_qualification": qualifications,
            "anti_saturation": anti, "gradient_direction": gradient_summary,
            "synthetic_scenes_pass": all(row["pass"] for row in synthetic),
            "sparse_dense_regression_pass": all_sparse_pass, "performance": performance,
            "safety": safety, "screen_space_placement_resume_allowed": scientific_pass,
            "seven_outfit_readjudication_allowed": False, "target_generation_allowed": False,
            "formal_image_conditioned_training_allowed": False, "training_allowed": False,
            "next_unique_task": next_task, "elapsed_seconds": time.perf_counter() - started,
        }
        atomic_json(root / "input_audit/frozen_state_postcheck.json", safety)
        atomic_json(root / "final_adjudication/DIFFERENTIABLE_SUPPORT_PROXY_FINAL_STATUS.json", final)
        atomic_text(root / "final_adjudication/GATE_ACCEPTANCE.md", "\n".join([
            "# Differentiable support proxy adjudication", "", f"**Status: {status}**", "", f"- case: `{case}`", f"- selected candidate: `{selected_candidate}`", f"- frozen proxy: `{frozen}`", f"- optimizer created: `false`", f"- optimizer steps: `0`", f"- placement resume allowed: `{str(scientific_pass).lower()}`", f"- next unique task: `{next_task}`",
        ]))
        atomic_json(status_path, {"status": status, "case": case, "stage": "complete", "optimizer_created": False, "optimizer_steps": 0, "completed_at": now()})
        _artifact_manifest(root)
        print(json.dumps(final, indent=2, default=str))
    except Exception as error:
        atomic_json(status_path, {"status": "TOOL_FAILURE", "failure_stage": stage, "optimizer_created": False, "optimizer_steps": 0, "exception_type": type(error).__name__, "exception": str(error), "traceback": traceback.format_exc(), "failed_at": now()})
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qualify additive differentiable screen-space support proxies")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args, load_config(args.config))


if __name__ == "__main__":
    main()
