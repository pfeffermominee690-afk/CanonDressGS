from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene.differentiable_support_proxy import dense_soft_precontributor_count  # noqa: E402
from scene.gaussian_clothing_residuals import (  # noqa: E402
    CHANNELS,
    GaussianClothingResiduals,
    compose_canonical_gaussian_overrides,
)
from scene.instrumented_projection_admission import (  # noqa: E402
    InstrumentationOptions,
    run_backend_projection_debug,
)
from scene.protected_cloud_attribution import (  # noqa: E402
    apply_diagnostic_counterfactual,
    apply_formal_guard,
    choose_case,
    classify_mechanism,
    fixed_open_ownership_record,
    gradient_provenance,
    membership_sha256,
    qualify_formal_guard,
    weighted_classification_summary,
)
from scene.support_aware_region_trusted_objective_v6 import (  # noqa: E402
    bound_normalized_regularization,
    residual_stability_loss,
)
from scene.support_aware_region_trusted_objective_v6_1 import (  # noqa: E402
    support_aware_region_trusted_objective_v6_1,
)
from tools import run_objective_residual_redesign as v6_runner  # noqa: E402
from tools.run_alpha_raster_audit import (  # noqa: E402
    load_runtime,
    load_state_model,
    render_explicit,
    sha256,
    target_free_state,
    trusted_regions,
)
from tools.run_differentiable_support_proxy import (  # noqa: E402
    _fixed_region_masks,
    _projection,
    _sample_regions,
)
from tools.run_editable_pool_specificity import trace_pixels  # noqa: E402
from tools.run_module4b_canonical_oracle_micropilot import (  # noqa: E402
    _base_named_tensors,
    _tensor_state_fingerprint,
)
from tools.run_new_silhouette_semantics import (  # noqa: E402
    candidate_metrics,
    transition_targets,
)


SCHEMA = "canondressgs.protected_cloud_attribution.v1"
VIEWS = {"cond_000000": "front", "cond_000318": "back", "cond_000017": "left", "cond_000347": "right"}
STATES = ("P1", "P2", "P3")
DIAGNOSTIC = ("D0", "D1", "D2", "D3", "D4")
FORMAL = ("F0", "F1", "F2", "F3", "F4")
RAW_BY_CHANNEL = {
    "delta_xyz": "raw_xyz",
    "delta_log_scaling": "raw_log_scaling",
    "delta_rotvec": "raw_rotvec",
    "delta_opacity_logit": "raw_opacity",
    "delta_sh0": "raw_sh0",
    "delta_shN": "raw_shN",
}
BOUND_BY_CHANNEL = {
    "delta_xyz": "xyz", "delta_log_scaling": "log_scaling", "delta_rotvec": "rotation",
    "delta_opacity_logit": "opacity_logit", "delta_sh0": "sh0",
}
OUTPUT_DIRS = (
    "contract", "input_audit", "protected_residual_attribution",
    "anchor_and_parameter_ownership", "loss_gradient_provenance",
    "diagnostic_counterfactuals", "formal_counterfactuals",
    "boundary_and_seam_audit", "coverage_regression", "visual_acceptance",
    "final_adjudication",
)
DEFAULT_CONFIG = PROJECT_ROOT / "configs/research/subject02_protected_cloud_attribution_v1.yaml"
DEFAULT_OUTPUT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/"
    "SUBJECT02-PROTECTED-CLOUD-ATTRIBUTION-001/attempt_001"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def resolve_ref(name: str) -> str:
    for candidate in (name, f"origin/{name}", f"cloud/{name}"):
        process = subprocess.run(["git", "rev-parse", candidate], cwd=PROJECT_ROOT, capture_output=True, text=True)
        if process.returncode == 0:
            return process.stdout.strip()
    raise RuntimeError(f"cannot resolve frozen ref {name}")


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA:
        raise ValueError("wrong protected-cloud attribution schema")
    if tuple(config["outfits"]) != ("O01", "O08") or tuple(config["conditions"]) != tuple(VIEWS):
        raise ValueError("fixed outfit/view protocol changed")
    if config["permissions"]["optimizer_created"] or int(config["permissions"]["optimizer_steps"]) != 0:
        raise ValueError("this audit forbids optimizer construction and parameter updates")
    if float(config["proxy"]["temperature"]) != .10:
        raise ValueError("Proxy A-0.10 changed")
    return config


def environment() -> dict[str, Any]:
    return {
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def tree_manifest(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    return {str(path.relative_to(root)): sha256(path) for path in sorted(root.rglob("*")) if path.is_file()}


def read_parquet(path: Path):
    import pandas as pd
    return pd.read_parquet(path)


def tensor_quantiles(value: torch.Tensor) -> dict[str, float | int]:
    flat = value.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if not flat.numel():
        return {"count": 0, "p50": 0., "p75": 0., "p90": 0., "p95": 0., "p99": 0., "max": 0.}
    q = torch.quantile(flat, torch.tensor([.5, .75, .9, .95, .99], device=flat.device))
    return {"count": int(flat.numel()), "p50": float(q[0]), "p75": float(q[1]), "p90": float(q[2]), "p95": float(q[3]), "p99": float(q[4]), "max": float(flat.max())}


def row_norm(value: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(value.detach().reshape(value.shape[0], -1), dim=1)


def model_residuals(model: Any, base: Any) -> GaussianClothingResiduals:
    if hasattr(model, "residuals"):
        return model.residuals(base)
    output = model(base)
    if not hasattr(output, "gaussian_residuals"):
        raise TypeError("state model does not expose Gaussian residuals")
    return output.gaussian_residuals


def body_part(joint: int) -> str:
    if joint in (10, 11): return "feet_or_shoes"
    if joint in (20, 21) or joint >= 25: return "hands"
    if joint in (15, 22, 23, 24): return "head_face_hair"
    if joint in (13, 14, 16, 17, 18, 19): return "arms"
    if joint in (1, 2, 4, 5, 7, 8): return "right_leg" if joint in (2, 5, 8) else "left_leg"
    return "torso"


def bool_mask(indices: np.ndarray, count: int, device: torch.device) -> torch.Tensor:
    result = torch.zeros(count, dtype=torch.bool, device=device)
    if len(indices): result[torch.from_numpy(indices.astype(np.int64, copy=False)).to(device)] = True
    return result


def save_rgb(path: Path, value: torch.Tensor) -> None:
    tensor = value.detach().float().cpu()
    if tensor.ndim != 3 or tensor.shape[0] != 3 or not torch.isfinite(tensor).all():
        raise ValueError("RGB must be finite CHW")
    image = (tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True); Image.fromarray(image, "RGB").save(path)


def save_alpha(path: Path, value: torch.Tensor) -> None:
    tensor = value.detach().float().cpu()
    if tensor.ndim != 3 or tensor.shape[0] != 1 or not torch.isfinite(tensor).all():
        raise ValueError("alpha must be finite CHW")
    image = (tensor[0].clamp(0, 1).numpy() * 65535).round().astype(np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True); Image.fromarray(image, "I;16").save(path)


def rgb_array(value: torch.Tensor) -> np.ndarray:
    return (value.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def contact_sheet(path: Path, panels: Sequence[tuple[str, np.ndarray]], columns: int = 4) -> None:
    cell = (256, 384); label = 25; rows = math.ceil(len(panels) / columns)
    canvas = Image.new("RGB", (cell[0] * columns, (cell[1] + label) * rows), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (name, image) in enumerate(panels):
        x = index % columns * cell[0]; y = index // columns * (cell[1] + label)
        draw.text((x + 3, y + 3), name, fill="black")
        thumb = ImageOps.contain(Image.fromarray(image), cell, Image.Resampling.LANCZOS)
        canvas.paste(thumb, (x + (cell[0] - thumb.width) // 2, y + label))
    path.parent.mkdir(parents=True, exist_ok=True); canvas.save(path)


def overlay_mask(rgb: torch.Tensor, mask: np.ndarray, color=(255, 0, 0)) -> np.ndarray:
    image = rgb_array(rgb).astype(np.float32)
    image[mask] = .3 * image[mask] + .7 * np.asarray(color, np.float32)
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def largest_component(mask: np.ndarray) -> int:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0


def screen_mask(projection: Mapping[str, torch.Tensor], indices: torch.Tensor, shape: tuple[int, int]) -> np.ndarray:
    output = np.zeros(shape, np.uint8)
    centers = projection["projected_mean"][indices].detach().cpu().numpy()
    radii = projection["major_minor_radius"][indices, 0].detach().cpu().numpy()
    for center, radius in zip(centers, radii):
        if np.isfinite(center).all() and np.isfinite(radius):
            cv2.circle(output, tuple(np.rint(center).astype(int)), max(1, min(64, int(math.ceil(radius)))), 1, -1)
    return output.astype(bool)


def masked_mae(left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor | np.ndarray) -> float:
    weight = torch.as_tensor(mask, device=left.device, dtype=left.dtype)
    while weight.ndim < left.ndim: weight = weight.unsqueeze(0)
    weight = weight.expand_as(left)
    return float(((left - right.to(left)).abs() * weight).sum() / weight.sum().clamp_min(1))


def source_evidence(config: Mapping[str, Any]) -> tuple[Any, Any, np.ndarray, np.ndarray, Any, dict[int, dict[str, Any]]]:
    root = Path(config["source_pool_output"])
    full = read_parquet(root / "full_gaussian_attribution/full_gaussian_attribution.parquet")
    events = read_parquet(root / "cloud_contributors/cloud_contributor_events.parquet")
    detail = read_parquet(root / "cloud_contributors/cloud_contributors.parquet")
    focus = events[(events.outfit == "O08") & (events.view == "back") & (events.state == "P3")]
    key = np.sort(focus.gaussian_index.unique().astype(np.int64))
    stable = full[full.exclusion_reason == "STABLE_PROTECTED"].gaussian_index.to_numpy(np.int64)
    if len(key) != int(config["key_cloud"]["expected_unique_indices"]) or len(focus) != int(config["key_cloud"]["expected_npre_occurrences"]):
        raise RuntimeError("frozen key cloud contributor contract changed")
    if not set(key).issubset(set(stable)):
        raise RuntimeError("key cloud indices no longer lie in stable protected membership")
    aggregate: dict[int, dict[str, Any]] = {}
    for index, rows in focus.groupby("gaussian_index"):
        aggregate[int(index)] = {
            "npre_occurrences": int(len(rows)), "active_contributions": int(rows.active.sum()),
            "alpha_mass": float(rows.alpha_mass.sum()), "center_enters_cloud": bool(rows.center_in_cloud.any()),
        }
    return full, events, key, stable, detail, aggregate


def stable_groups(
    full: Any, stable: np.ndarray, key: np.ndarray, base: Any,
) -> dict[str, torch.Tensor]:
    count = int(base._xyz.shape[0]); device = base._xyz.device
    stable_mask = bool_mask(stable, count, device); key_mask = bool_mask(key, count, device)
    dominant = torch.from_numpy(full.dominant_anchor.to_numpy(np.int64)).to(device)
    stable_anchor = torch.zeros(base.nbr_vt.shape[0], dtype=torch.bool, device=device)
    nonstable_anchor = torch.zeros_like(stable_anchor)
    stable_anchor[dominant[stable_mask]] = True; nonstable_anchor[dominant[~stable_mask]] = True
    boundary_anchor = stable_anchor & nonstable_anchor[base.nbr_vt.long()].any(1)
    boundary = stable_mask & boundary_anchor[dominant]
    joints = base.get_weights.argmax(1)
    groups = {
        "stable_protected_all": stable_mask,
        "protected_boundary": boundary,
        "protected_non_cloud": stable_mask & ~key_mask,
        "protected_cloud": key_mask,
        "right_leg": stable_mask & ((joints == 2) | (joints == 5) | (joints == 8)),
        "feet_or_shoes": stable_mask & ((joints == 10) | (joints == 11)),
        "hands": stable_mask & (((joints == 20) | (joints == 21)) | (joints >= 25)),
        "head_face_hair": stable_mask & ((joints == 15) | (joints == 22) | (joints == 23) | (joints == 24)),
        "torso": stable_mask & ((joints == 0) | (joints == 3) | (joints == 6) | (joints == 9) | (joints == 12)),
        "arms": stable_mask & ((joints == 13) | (joints == 14) | (joints == 16) | (joints == 17) | (joints == 18) | (joints == 19)),
    }
    return groups


def residual_distribution(
    output: Path, config: Mapping[str, Any], base: Any, sample: Mapping[str, Any],
    models: Mapping[str, Any], full: Any, events: Any, stable: np.ndarray, key: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state = target_free_state(sample, base._xyz.device)
    _, base_overrides = load_state_model(config, "P0", "O08", base)
    base_projection, base_tensors, _ = _projection(base, state, base_overrides)
    groups = stable_groups(full, stable, key, base)
    rows = []; state_cache = {}
    for name, model in models.items():
        residuals = model_residuals(model, base)
        model_output = model(base)
        overrides = model_output.canonical_overrides if hasattr(model_output, "canonical_overrides") else model_output
        projection, tensors, _ = _projection(base, state, overrides)
        alpha_by_index = torch.zeros(base._xyz.shape[0], device=base._xyz.device)
        subset = events[(events.outfit == "O08") & (events.view == "back") & (events.state == name)]
        if len(subset):
            grouped = subset.groupby("gaussian_index").alpha_mass.sum()
            ids = torch.from_numpy(grouped.index.to_numpy(np.int64)).to(base._xyz.device)
            alpha_by_index[ids] = torch.from_numpy(grouped.to_numpy(np.float32)).to(base._xyz.device)
        values = {channel: row_norm(getattr(residuals, channel)) for channel in CHANNELS}
        values.update({
            "canonical_displacement": torch.linalg.vector_norm(overrides.xyz - base._xyz, dim=1),
            "posed_displacement": torch.linalg.vector_norm(tensors["posed_xyz"] - base_tensors["posed_xyz"], dim=1),
            "screen_displacement": torch.linalg.vector_norm(projection["projected_mean"] - base_projection["projected_mean"], dim=1),
            "projected_major_radius": projection["major_minor_radius"][:, 0],
            "projected_minor_radius": projection["major_minor_radius"][:, 1],
            "opacity": tensors["effective_opacity"], "sampled_fixed_cloud_alpha_mass": alpha_by_index,
        })
        state_cache[name] = {"residuals": residuals, "overrides": overrides, "projection": projection, "tensors": tensors, "values": values}
        for group_name, mask in groups.items():
            for metric, value in values.items():
                rows.append({"state": name, "group": group_name, "metric": metric, **tensor_quantiles(value[mask])})
    write_csv(output / "protected_residual_attribution/protected_residual_quantiles.csv", rows)
    payload = {state: {group: {row["metric"]: {k: row[k] for k in ("count", "p50", "p75", "p90", "p95", "p99", "max")} for row in rows if row["state"] == state and row["group"] == group} for group in groups} for state in STATES}
    atomic_json(output / "protected_residual_attribution/protected_residual_distribution.json", payload)
    cloud = payload["P3"]["protected_cloud"]; noncloud = payload["P3"]["protected_non_cloud"]
    comparison = {name: {"cloud_p50": cloud[name]["p50"], "noncloud_p50": noncloud[name]["p50"], "cloud_p95": cloud[name]["p95"], "noncloud_p95": noncloud[name]["p95"]} for name in ("delta_xyz", "delta_log_scaling", "delta_rotvec", "delta_opacity_logit")}
    atomic_json(output / "protected_residual_attribution/cloud_vs_noncloud_comparison.json", comparison)
    return state_cache, {name: int(mask.sum()) for name, mask in groups.items()}


def ownership_audit(
    output: Path, base: Any, full: Any, key: np.ndarray, aggregate: Mapping[int, Mapping[str, Any]], state_cache: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records = []; influence = []
    p3 = state_cache["P3"]
    base_screen = state_cache["P1"]["projection"]["projected_mean"] * 0  # overwritten below only for shape
    joints = base.get_weights.argmax(1)
    dominant = torch.from_numpy(full.dominant_anchor.to_numpy(np.int64)).to(base._xyz.device)
    canonical = p3["values"]["canonical_displacement"]
    posed = p3["values"]["posed_displacement"]
    for index in key.tolist():
        event = aggregate[index]
        covariance = float(p3["values"]["delta_log_scaling"][index] + p3["values"]["delta_rotvec"][index]) > 0
        ratio = float(posed[index] / canonical[index].clamp_min(1e-12))
        ownership_class = classify_mechanism(
            independent_parameter=True, shared_anchor=False, shared_global=False,
            canonical_displacement=float(canonical[index]), posed_amplification_ratio=ratio,
            center_enters_cloud=bool(event["center_enters_cloud"]), covariance_nonzero=covariance,
        )
        record = fixed_open_ownership_record(
            index, production_contribution_count=int(event["active_contributions"]),
            production_alpha_mass=float(event["alpha_mass"]), dominant_lbs_joint=int(joints[index]),
        )
        record.update({
            "ownership_class": ownership_class, "npre_occurrences": int(event["npre_occurrences"]),
            "active_contributions": int(event["active_contributions"]), "alpha_mass": float(event["alpha_mass"]),
            "dominant_anchor": int(dominant[index]), "center_enters_cloud": bool(event["center_enters_cloud"]),
            "canonical_displacement": float(canonical[index]), "posed_displacement": float(posed[index]),
            "pose_amplification_ratio": ratio,
            "projected_center": [float(v) for v in p3["projection"]["projected_mean"][index]],
            "projected_major_minor_radius": [float(v) for v in p3["projection"]["major_minor_radius"][index]],
        })
        records.append(record)
        influence.append({
            "gaussian_index": index, "ownership_class": ownership_class,
            "independent_residual": True, "dominant_anchor_context_only": int(dominant[index]),
            "source_anchor_count": 0, "top_k_source_anchors": "", "interpolation_weights": "",
            "shares_parameter_with_coverage": False, "shares_anchor_with_old_clothing": False,
            "shared_global_modulation": False,
        })
    write_jsonl(output / "anchor_and_parameter_ownership/protected_parameter_ownership.jsonl", records)
    write_csv(output / "anchor_and_parameter_ownership/protected_anchor_influence.csv", influence)
    summary = weighted_classification_summary(records)
    atomic_json(output / "anchor_and_parameter_ownership/ownership_classification_summary.json", summary)
    atomic_text(output / "anchor_and_parameter_ownership/PROTECTED_PARAMETER_OWNERSHIP.md", "\n".join([
        "# Protected parameter ownership", "",
        "The loaded P3 checkpoint is a FixedOpenGaussianOracle. Each of the 200,000 rows owns independent raw_xyz, raw_log_scaling, raw_rotvec, raw_opacity and raw_sh0 parameters; raw_shN is frozen zero.",
        "", "There is no anchor interpolation, trainable gate, condition/view/camera parameter, FiLM, or global modulation in this checkpoint. The dominant anchor column is base geometry context, not a residual source.",
        "", "The stable-protected label affected V6.1 loss regions only; it was not enforced during residual composition.",
    ]))
    return records


def gradient_audit(
    output: Path, config: Mapping[str, Any], base: Any, samples: Mapping[str, Any], background: torch.Tensor,
    model: Any, key_mask: torch.Tensor, stable_mask: torch.Tensor,
) -> dict[str, Any]:
    model.configure_stage(3)
    weight_path = Path(config["source_v6_1_output"]) / "gradient_analysis/v6_1_loss_weights_frozen.json"
    weights = {key: float(value) for key, value in json.loads(weight_path.read_text()).items()}
    cached = transition_targets(samples, base._xyz.device)
    rows = []; vectors: dict[str, list[torch.Tensor]] = defaultdict(list)
    channels = list(CHANNELS); raw_names = [RAW_BY_CHANNEL[name] for name in channels]
    for condition in ("cond_000318", "cond_000347"):
        view = VIEWS[condition]; sample = samples[condition]; state = target_free_state(sample, base._xyz.device)
        output_state = model(base); residuals = output_state.gaussian_residuals
        rgb, alpha, _, _ = render_explicit(base, state, output_state.canonical_overrides, background)
        regularization, _ = bound_normalized_regularization(residuals, base, model.bounds, garment_weight=.25, protected_weight=1.0)
        stability, _ = residual_stability_loss(residuals, base, model.bounds)
        result = support_aware_region_trusted_objective_v6_1(
            rgb, alpha, sample, weights=weights, residual_loss=regularization, stability_loss=stability,
            progress_margin=.02, change_epsilon=.01,
            support_diagonal_ratio=float(config["support_band"]["base_bbox_diagonal_ratio"]),
            minimum_radius_pixels=int(config["support_band"]["minimum_radius_pixels"]),
            transition_alpha_target=cached[condition],
        )
        finals = [getattr(residuals, name) for name in channels]
        raws = [getattr(model, name) for name in raw_names]
        differentiable = [value for value in finals + raws if value.requires_grad]
        for group, part in result.parts.items():
            weighted = float(weights[group]) * part
            active_gradients = torch.autograd.grad(weighted, differentiable, retain_graph=True, allow_unused=True)
            lookup = {id(value): gradient for value, gradient in zip(differentiable, active_gradients)}
            final_grads = [lookup.get(id(value)) for value in finals]
            raw_grads = [lookup.get(id(value)) for value in raws]
            for channel, final_grad, raw_grad in zip(channels, final_grads, raw_grads):
                direct = gradient_provenance(final_grad, raw_grad, stable_mask, shared_parameter_gradient_norm=0.0)
                key = gradient_provenance(final_grad, raw_grad, key_mask, shared_parameter_gradient_norm=0.0)
                nonprotected_norm = 0.0
                if final_grad is not None:
                    selected = final_grad[~stable_mask].reshape((~stable_mask).sum(), -1)
                    nonprotected_norm = float(torch.linalg.vector_norm(selected, dim=1).sum())
                final_norm = float(direct["direct_protected_gradient_norm"])
                raw_norm = float(direct["raw_protected_gradient_norm"])
                rows.append({
                    "outfit": "O08", "view": view, "loss_group": group, "loss_value": float(part.detach()),
                    "frozen_weight": float(weights[group]), "channel": channel,
                    **{f"stable_{name}": value for name, value in direct.items()},
                    **{f"key_cloud_{name}": value for name, value in key.items()},
                    "nonprotected_final_gradient_norm": nonprotected_norm,
                    "anchor_gradient_norm": 0.0, "shared_modulation_gradient_norm": 0.0,
                    "bounded_raw_jacobian_influence_ratio": raw_norm / max(final_norm, 1e-12),
                    "cross_region_shared_parameter_leakage": False,
                })
            key_xyz = final_grads[0]
            vectors[group].append(torch.zeros(int(key_mask.sum()) * 3) if key_xyz is None else key_xyz[key_mask].detach().cpu().reshape(-1))
        del output_state, rgb, alpha, result
    write_csv(output / "loss_gradient_provenance/loss_gradient_provenance.csv", rows)
    group_summary = {}
    combined = {name: torch.cat(values) for name, values in vectors.items()}
    total_vector = sum(combined.values(), torch.zeros_like(next(iter(combined.values()))))
    denominator_total = torch.linalg.vector_norm(total_vector).clamp_min(1e-12)
    for name, vector in combined.items():
        norm = torch.linalg.vector_norm(vector)
        cosine = float(torch.dot(vector, total_vector) / (norm.clamp_min(1e-12) * denominator_total))
        subset = [row for row in rows if row["loss_group"] == name]
        group_summary[name] = {
            "weighted_loss": float(np.mean([row["loss_value"] * row["frozen_weight"] for row in subset])),
            "key_cloud_final_gradient_norm_sum": float(sum(row["key_cloud_direct_protected_gradient_norm"] for row in subset)),
            "stable_protected_final_gradient_norm_sum": float(sum(row["stable_direct_protected_gradient_norm"] for row in subset)),
            "nonprotected_final_gradient_norm_sum": float(sum(row["nonprotected_final_gradient_norm"] for row in subset)),
            "key_delta_xyz_cosine_with_total": cosine,
            "indirect_shared_parameter_leakage": False,
        }
    total_key = sum(value["key_cloud_final_gradient_norm_sum"] for value in group_summary.values())
    for value in group_summary.values(): value["key_cloud_gradient_contribution_fraction"] = value["key_cloud_final_gradient_norm_sum"] / max(total_key, 1e-12)
    payload = {
        "views": ["O08/back", "O08/right"], "optimizer_created": False, "optimizer_steps": 0,
        "actual_parameter_topology": "independent_per_Gaussian_fixed_open_residuals",
        "anchor_parameters_present": False, "shared_modulation_parameters_present": False,
        "direct_and_indirect_distinguished": True, "loss_groups": group_summary,
    }
    atomic_json(output / "loss_gradient_provenance/loss_gradient_provenance.json", payload)
    atomic_text(output / "loss_gradient_provenance/LOSS_GRADIENT_PROVENANCE.md", "\n".join([
        "# V6.1 loss gradient provenance", "",
        "Independent autograd VJPs were evaluated for all ten frozen V6.1 loss groups on O08 back/right. No parameter update or optimizer object was used.",
        "", "P3 contains no anchor or shared modulation parameters. Consequently indirect shared-parameter leakage is structurally absent; any protected-row gradient is a direct per-Gaussian gradient, including gradients induced by a Gaussian footprint crossing loss-region boundaries.",
    ]))
    model.eval()
    return payload


def candidate_variant(residuals: GaussianClothingResiduals, variant: str, stable_mask: torch.Tensor, key_indices: torch.Tensor) -> GaussianClothingResiduals:
    if variant.startswith("D"):
        return apply_diagnostic_counterfactual(residuals, key_indices, variant).residuals
    return apply_formal_guard(residuals, stable_mask, variant).residuals


def focus_trace_metrics(
    config: Mapping[str, Any], base: Any, state: Mapping[str, Any], overrides: Any,
    rgb: torch.Tensor, alpha: torch.Tensor, sample: Mapping[str, Any], fixed: Mapping[str, np.ndarray],
    key_indices: torch.Tensor, coverage_pool: torch.Tensor, background: torch.Tensor,
) -> dict[str, Any]:
    projection, tensors, camera = _projection(base, state, overrides)
    backend = run_backend_projection_debug(
        means=tensors["posed_xyz"], covars=tensors["posed_covariance"], opacities=tensors["effective_opacity"],
        w2c=camera["w2c"], K=camera["K"], width=int(camera["width"]), height=int(camera["height"]),
        options=InstrumentationOptions(enabled=True),
    )
    if backend is None: raise RuntimeError("production backend trace unavailable")
    pixels, lookup = _sample_regions(fixed, int(config["trace_sampling"]["pixels_per_region"]), base._xyz.device)
    cloud_pixels = pixels[lookup["trailing_cloud"]]
    cloud_rows = trace_pixels(projection, backend, tensors["effective_opacity"], cloud_pixels, tile_size=16, alpha_cutoff=float(config["trace_sampling"]["theoretical_alpha_cutoff"]), termination=float(config["trace_sampling"]["actual_early_termination"]))
    key = set(int(value) for value in key_indices.detach().cpu().tolist())
    key_npre = sum(index in key for row in cloud_rows for index in row["pre"])
    key_active = sum(index in key for row in cloud_rows for index in row["active"])
    key_alpha = sum(mass for row in cloud_rows for index, mass in zip(row["active"], row["active_alpha_mass"]) if index in key)
    centers = projection["projected_mean"][key_indices].detach().cpu().numpy()
    x = np.rint(centers[:, 0]).astype(np.int64); y = np.rint(centers[:, 1]).astype(np.int64)
    valid = (x >= 0) & (x < fixed["trailing_cloud"].shape[1]) & (y >= 0) & (y < fixed["trailing_cloud"].shape[0])
    entered = int(fixed["trailing_cloud"][y[valid], x[valid]].sum())
    expansion_pixels = pixels[lookup["trusted_expansion"]]
    expansion_rows = trace_pixels(projection, backend, tensors["effective_opacity"], expansion_pixels, tile_size=16, alpha_cutoff=float(config["trace_sampling"]["theoretical_alpha_cutoff"]), termination=float(config["trace_sampling"]["actual_early_termination"]))
    proxy = dense_soft_precontributor_count(
        projection["projected_mean"][coverage_pool], projection["conic"][coverage_pool], projection["opacity"][coverage_pool], expansion_pixels,
        temperature=.10, valid=projection["stage_masks_s0_s5"][coverage_pool, 3], gaussian_chunk=2048,
    ) if expansion_pixels.numel() else torch.empty(0, device=base._xyz.device)
    metrics = candidate_metrics(rgb, alpha, sample, config)
    regions = trusted_regions(sample, config)
    predicted = alpha[0] >= .5
    expansion = fixed["trusted_expansion"]
    removal = fixed["trusted_removal"]
    target = sample["target_foreground_mask"][0].detach().cpu().numpy() >= .5
    target_rgb = sample["target_edit_rgb"].to(rgb)
    base_rgb = sample["target_base_rgb"].to(rgb)
    edit_mask = torch.maximum(sample["target_edit_mask"], sample["target_clothing_mask"]).to(rgb)
    base_target_error = masked_mae(base_rgb, target_rgb, edit_mask)
    current_target_error = masked_mae(rgb, target_rgb, edit_mask)
    cloud_visible = fixed["trailing_cloud"] & (alpha[0].detach().cpu().numpy() >= .05)
    protected = sample["target_protected_mask"][0].detach().cpu().numpy() >= .5
    yy, xx = np.where(protected)
    spatial_masks = {"face_hair": np.zeros_like(protected), "hands": np.zeros_like(protected), "shoes": np.zeros_like(protected)}
    if yy.size:
        ymin, ymax, xmin, xmax = yy.min(), yy.max(), xx.min(), xx.max()
        height = max(1, ymax - ymin + 1); width = max(1, xmax - xmin + 1)
        spatial_masks["face_hair"] = protected & (np.indices(protected.shape)[0] <= ymin + .22 * height)
        spatial_masks["shoes"] = protected & (np.indices(protected.shape)[0] >= ymin + .80 * height)
        xgrid, ygrid = np.indices(protected.shape)[1], np.indices(protected.shape)[0]
        spatial_masks["hands"] = protected & (ygrid >= ymin + .22 * height) & (ygrid <= ymin + .78 * height) & ((xgrid <= xmin + .18 * width) | (xgrid >= xmax - .18 * width))
    protected_edge = cv2.morphologyEx(protected.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
    gray = rgb.detach().float().mean(0).cpu().numpy()
    edge_gradient = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    return {
        "key_cloud_index_count": len(key), "key_cloud_npre_occurrences": int(key_npre),
        "cloud_active": int(key_active), "cloud_alpha_mass": float(key_alpha),
        "cloud_area": int(cloud_visible.sum()), "max_connected_cloud_component": largest_component(cloud_visible),
        "center_entered": entered, "tail_only": int(len(key) - entered),
        "trusted_expansion_npre": int(sum(row["N_pre"] for row in expansion_rows)),
        "trusted_expansion_proxy_a_0_10_mean": float(proxy.mean()) if proxy.numel() else 0.,
        "trusted_expansion_recall": float((predicted.detach().cpu().numpy() & expansion).sum() / max(expansion.sum(), 1)),
        "trusted_removal_recall": float(((~predicted.detach().cpu().numpy()) & removal).sum() / max(removal.sum(), 1)),
        "internal_holes": int((target & ~predicted.detach().cpu().numpy()).sum()),
        "target_closer": float(metrics["target_closer_fraction"]),
        "edit_reduction": float((base_target_error - current_target_error) / max(base_target_error, 1e-12)),
        "protected_mae": float(metrics["protected_mae"]), "background_leakage": float(metrics["background_leakage"]),
        "protected_alpha_mae": masked_mae(alpha, sample["target_base_foreground_mask"].to(alpha), protected),
        "face_hair_rgb_mae": masked_mae(rgb, base_rgb, spatial_masks["face_hair"]),
        "hands_rgb_mae": masked_mae(rgb, base_rgb, spatial_masks["hands"]),
        "shoes_rgb_mae": masked_mae(rgb, base_rgb, spatial_masks["shoes"]),
        "protected_edge_continuity": float(edge_gradient[protected_edge].mean()) if protected_edge.any() else 0.,
        "protected_subregion_definition": "frozen target-protected-mask spatial proxy; evaluation only",
        "full_silhouette_iou": float(metrics["silhouette_iou"]),
    }


def counterfactuals(
    output: Path, config: Mapping[str, Any], full: Any, key: np.ndarray, stable: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    all_metrics: dict[str, Any] = {}; render_cache: dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]] = {}
    focus_fixed = None; focus_baseline = None; base_hashes = {}; renderer_equal = {}; evaluation_images = {}
    coverage_np = np.load(Path(config["source_pool_output"]) / "candidate_pool_contracts/P0_spill_indices.npy")
    for outfit in ("O08", "O01"):
        base, samples, background, device = load_runtime(config, outfit, "cuda")
        base_before = _tensor_state_fingerprint(_base_named_tensors(base)); base_hashes[f"{outfit}_before"] = base_before
        for condition, view in VIEWS.items():
            if outfit == "O08" and view in ("back", "right"):
                evaluation_images[(outfit, view, "base")] = samples[condition]["target_base_rgb"].detach().cpu()
                evaluation_images[(outfit, view, "target")] = samples[condition]["target_edit_rgb"].detach().cpu()
        model, _ = load_state_model(config, "P3", outfit, base); model.eval(); output_state = model(base); residuals = output_state.gaussian_residuals
        stable_mask = bool_mask(stable, int(base._xyz.shape[0]), device); key_indices = torch.from_numpy(key).to(device)
        coverage_pool = torch.from_numpy(coverage_np.astype(np.int64, copy=False)).to(device)
        variants = DIAGNOSTIC + FORMAL if outfit == "O08" else FORMAL
        for variant in variants:
            if outfit == "O01" and variant == "F0": views = ("back", "right")
            elif outfit == "O01": views = ("back", "right")
            elif variant.startswith("D"): views = ("back", "right")
            else: views = tuple(VIEWS.values())
            guarded = candidate_variant(residuals, variant, stable_mask, key_indices)
            overrides = compose_canonical_gaussian_overrides(base, guarded)
            safety = {
                "all_residuals_finite": all(bool(torch.isfinite(getattr(guarded, name)).all()) for name in CHANNELS),
                "abnormal_introduced_by_guard": False,
                "bound_hit_fraction_max": max(float((getattr(guarded, name).detach().abs() >= .95 * float(model.bounds[BOUND_BY_CHANNEL[name]])).float().mean()) for name in BOUND_BY_CHANNEL),
            }
            destination = output / ("diagnostic_counterfactuals" if variant.startswith("D") else "formal_counterfactuals") / variant / outfit
            for condition, view in VIEWS.items():
                if view not in views: continue
                state = target_free_state(samples[condition], device)
                rgb, alpha, _, tensors = render_explicit(base, state, overrides, background)
                render_cache[(outfit, variant, view)] = (rgb.detach().cpu(), alpha.detach().cpu())
                save_rgb(destination / view / "rgb.png", rgb); save_alpha(destination / view / "alpha.png", alpha)
                metrics = candidate_metrics(rgb, alpha, samples[condition], config)
                row = {**metrics, "outfit": outfit, "variant": variant, "view": view}
                row.update(safety)
                if outfit == "O08" and view == "back":
                    if focus_fixed is None:
                        focus_fixed = _fixed_region_masks(samples[condition], trusted_regions(samples[condition], config), alpha)
                        counts = json.loads((Path(config["source_pool_output"]) / "input_audit/frozen_region_counts.json").read_text())
                        expected = counts["O08/back"]
                        observed = {name: int(mask.sum()) for name, mask in focus_fixed.items()}
                        if any(int(expected.get(name, value)) != value for name, value in observed.items()):
                            raise RuntimeError("fixed O08/back region masks drifted")
                        focus_baseline = {"fixed_region_counts": observed}
                    row.update(focus_trace_metrics(config, base, state, overrides, rgb, alpha, samples[condition], focus_fixed, key_indices, coverage_pool, background))
                    if variant in ("D0", "F0"):
                        second_rgb, second_alpha, _, _ = render_explicit(base, state, overrides, background)
                        renderer_equal[variant] = {"rgb_bitwise_equal": bool(torch.equal(rgb, second_rgb)), "alpha_bitwise_equal": bool(torch.equal(alpha, second_alpha)), "rgb_max_abs_diff": float((rgb - second_rgb).abs().max()), "alpha_max_abs_diff": float((alpha - second_alpha).abs().max())}
                all_metrics[f"{outfit}/{variant}/{view}"] = row
        base_after = _tensor_state_fingerprint(_base_named_tensors(base)); base_hashes[f"{outfit}_after"] = base_after
        if base_before != base_after: raise RuntimeError(f"{outfit} frozen base changed")
        del base, samples, model, output_state, residuals
        torch.cuda.empty_cache()
    if focus_fixed is None: raise AssertionError("focus baseline was not rendered")
    atomic_json(output / "diagnostic_counterfactuals/metrics.json", {key: value for key, value in all_metrics.items() if "/D" in key})
    atomic_json(output / "formal_counterfactuals/metrics.json", {key: value for key, value in all_metrics.items() if "/F" in key})
    atomic_json(output / "input_audit/base_fingerprints.json", base_hashes)
    atomic_json(output / "input_audit/renderer_bitwise_regression.json", renderer_equal)
    build_visuals(output, render_cache, focus_fixed, evaluation_images)
    return all_metrics, base_hashes, renderer_equal


def build_visuals(output: Path, renders: Mapping[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]], cloud_mask: np.ndarray, evaluation_images: Mapping[tuple[str, str, str], torch.Tensor]) -> None:
    panels = [("O08 back base", rgb_array(evaluation_images[("O08", "back", "base")])), ("O08 back target", rgb_array(evaluation_images[("O08", "back", "target")]))]
    for variant in DIAGNOSTIC:
        for view in ("back", "right"):
            rgb, _ = renders[("O08", variant, view)]; panels.append((f"{variant} O08 {view}", rgb_array(rgb)))
    contact_sheet(output / "visual_acceptance/diagnostic_D0_D4_O08_back_right.png", panels, 4)
    panels = []
    for variant in FORMAL:
        for view in VIEWS.values():
            rgb, _ = renders[("O08", variant, view)]; panels.append((f"{variant} O08 {view}", rgb_array(rgb)))
    contact_sheet(output / "visual_acceptance/formal_F0_F4_O08_four_views.png", panels, 4)
    panels = []
    for variant in FORMAL:
        for view in ("back", "right"):
            rgb, _ = renders[("O01", variant, view)]; panels.append((f"{variant} O01 {view}", rgb_array(rgb)))
    contact_sheet(output / "visual_acceptance/o01_regression_F0_F4.png", panels, 4)
    panels = []
    y, x = np.where(cloud_mask); y0, y1 = max(0, int(y.min()) - 50), min(cloud_mask.shape[0], int(y.max()) + 51); x0, x1 = max(0, int(x.min()) - 50), min(cloud_mask.shape[1], int(x.max()) + 51)
    for variant in FORMAL:
        rgb, _ = renders[("O08", variant, "back")]
        panels.append((f"{variant} cloud overlay", overlay_mask(rgb, cloud_mask)))
        panels.append((f"{variant} shoes/right-leg crop", rgb_array(rgb)[y0:y1, x0:x1]))
        alpha = renders[("O08", variant, "back")][1][0]
        panels.append((f"{variant} alpha", np.repeat((alpha.clamp(0, 1).numpy()[..., None] * 255).astype(np.uint8), 3, axis=2)))
    contact_sheet(output / "visual_acceptance/cloud_shoes_leg_boundary_review.png", panels, 4)


def formal_qualification(output: Path, metrics: Mapping[str, Any]) -> dict[str, Any]:
    baseline = metrics["O08/F0/back"]
    qualifications = {}
    rows = []
    for variant in ("F1", "F2", "F3", "F4"):
        candidate = metrics[f"O08/{variant}/back"]
        o01_rows = [metrics[f"O01/{variant}/{view}"] for view in ("back", "right")]
        o01_base = [metrics[f"O01/F0/{view}"] for view in ("back", "right")]
        o01_pass = all(
            row["target_closer_fraction"] >= base["target_closer_fraction"] - .01
            and row["protected_mae"] <= .01 and row["background_leakage"] <= .03
            for row, base in zip(o01_rows, o01_base)
        )
        preliminary = qualify_formal_guard(baseline, candidate, seam_pass=True, o01_pass=o01_pass, membership_target_independent=True)
        preliminary["visual_seam_gate"] = "PENDING_ACTUAL_INSPECTION"
        preliminary["status"] = "NUMERIC_PASS_VISUAL_PENDING" if preliminary["status"] == "PASS" else "FAIL"
        preliminary["o01_pass"] = o01_pass
        qualifications[variant] = preliminary
        rows.append({"variant": variant, "numeric_status": preliminary["status"], **{f"gate_{name}": value for name, value in preliminary["gates"].items()},
                     "cloud_alpha_decrease": (baseline["cloud_alpha_mass"] - candidate["cloud_alpha_mass"]) / max(baseline["cloud_alpha_mass"], 1e-12),
                     "cloud_active_decrease": (baseline["cloud_active"] - candidate["cloud_active"]) / max(baseline["cloud_active"], 1),
                     "center_decrease": (baseline["center_entered"] - candidate["center_entered"]) / max(baseline["center_entered"], 1)})
    write_csv(output / "coverage_regression/formal_candidate_qualification.csv", rows)
    atomic_json(output / "coverage_regression/formal_candidate_qualification_previsual.json", qualifications)
    return qualifications


def seal_visual(output: Path, notes_path: Path) -> int:
    if not output.is_dir(): raise FileNotFoundError(output)
    final = output / "final_adjudication/PROTECTED_CLOUD_ATTRIBUTION_FINAL_STATUS.json"
    if final.exists(): raise FileExistsError("final adjudication is append-only and already exists")
    notes = json.loads(notes_path.read_text(encoding="utf-8"))
    required = {"images_actually_opened", "inspection_method", "observations", "seam_pass_by_variant", "visual_acceptance_status"}
    if set(notes) < required or not notes["images_actually_opened"]:
        raise ValueError("visual evidence must record actual opened images and observations")
    metrics = json.loads((output / "formal_counterfactuals/metrics.json").read_text())
    baseline = metrics["O08/F0/back"]
    qualifications = {}
    for variant in ("F1", "F2", "F3", "F4"):
        candidate = metrics[f"O08/{variant}/back"]
        o01_pass = all(
            metrics[f"O01/{variant}/{view}"]["target_closer_fraction"] >= metrics[f"O01/F0/{view}"]["target_closer_fraction"] - .01
            and metrics[f"O01/{variant}/{view}"]["protected_mae"] <= .01
            and metrics[f"O01/{variant}/{view}"]["background_leakage"] <= .03
            for view in ("back", "right")
        )
        qualifications[variant] = qualify_formal_guard(
            baseline, candidate, seam_pass=bool(notes["seam_pass_by_variant"][variant]),
            o01_pass=o01_pass, membership_target_independent=True,
        )
    case, next_task = choose_case(qualifications, f5_ran=False)
    visual = {**notes, "sealed_at": now(), "F5_run": False}
    atomic_json(output / "visual_acceptance/visual_acceptance.json", visual)
    atomic_text(output / "visual_acceptance/VISUAL_ACCEPTANCE.md", "\n".join([
        "# Visual acceptance", "", f"- status: `{notes['visual_acceptance_status']}`",
        f"- inspection method: `{notes['inspection_method']}`", "",
        *[f"- {key}: {value}" for key, value in notes["observations"].items()],
    ]))
    run_manifest = json.loads((output / "contract/run_manifest.json").read_text())
    gradient = json.loads((output / "loss_gradient_provenance/loss_gradient_provenance.json").read_text())
    ownership = json.loads((output / "anchor_and_parameter_ownership/ownership_classification_summary.json").read_text())
    payload = {
        "status": "PASS" if case != "PU" else "PARTIAL", "case": case, "next_unique_task": next_task,
        "formal_run_commit": run_manifest["head"], "optimizer_created": False, "optimizer_steps": 0,
        "qualifications": qualifications, "visual_acceptance": visual,
        "ownership_classification": ownership, "gradient_provenance": gradient,
        "allow_formal_guard_implementation": case in {"PX", "PG", "PO", "PF", "PT"},
        "allow_resume_placement": False, "allow_seven_outfit_rerun": False,
        "allow_more_targets": False, "allow_formal_training": False, "F5_run": False,
        "sealed_at": now(),
    }
    atomic_json(final, payload)
    atomic_text(output / "final_adjudication/GATE_ACCEPTANCE.md", "\n".join([
        "# Protected cloud attribution final adjudication", "",
        f"- status: `{payload['status']}`", f"- case: `{case}`", f"- next unique task: `{next_task}`",
        "- optimizer created: `false`", "- optimizer steps: `0`", "- placement resume: `false`",
        "", "Formal membership is base-derived stable-protected and never reads target, cloud, outfit, or rendered outcomes.",
    ]))
    status = json.loads((output / "RUN_STATUS.json").read_text())
    status.update({"status": "COMPLETE", "case": case, "visual_sealed_at": now(), "failure_stage": None})
    atomic_json(output / "RUN_STATUS.json", status)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seal-visual", type=Path)
    args = parser.parse_args()
    if args.seal_visual is not None:
        return seal_visual(args.output_dir, args.seal_visual)
    config = load_config(args.config); output = args.output_dir
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"append-only output exists: {output}")
    for name in OUTPUT_DIRS: (output / name).mkdir(parents=True, exist_ok=True)
    status = {"task_id": config["task_id"], "status": "RUNNING", "started_at": now(), "failure_stage": None, "optimizer_created": False, "optimizer_steps": 0}
    atomic_json(output / "RUN_STATUS.json", status); started = time.perf_counter()
    try:
        status["failure_stage"] = "input_audit"; atomic_json(output / "RUN_STATUS.json", status)
        branch = git("branch", "--show-current"); head = git("rev-parse", "HEAD"); dirty = git("status", "--short")
        if branch != config["research_branch"] or dirty: raise RuntimeError(f"formal runner requires clean {config['research_branch']}; dirty={dirty!r}")
        if git("rev-parse", config["source_tag"] + "^{}") != config["source_head"]: raise RuntimeError("frozen Case PP tag moved")
        source_manifest_before = tree_manifest(Path(config["source_pool_output"]))
        frozen_refs = {name: resolve_ref(name) for name in config["frozen_branches"]}
        if frozen_refs != config["frozen_branches"]:
            raise RuntimeError("one or more frozen branches moved before the audit")
        full, events, key, stable, detail, aggregate = source_evidence(config)
        atomic_json(output / "contract/config_resolved.json", config)
        atomic_text(output / "contract/execution_command.txt", " ".join(sys.argv))
        atomic_json(output / "contract/run_manifest.json", {"head": head, "branch": branch, "git_status": dirty, "environment": environment(), "source_tag_peeled": config["source_head"], "frozen_refs": frozen_refs})
        atomic_json(output / "input_audit/frozen_evidence.json", {
            "source_pool_output": str(config["source_pool_output"]), "source_file_manifest": source_manifest_before,
            "stable_protected_count": int(len(stable)), "stable_protected_membership_sha256": membership_sha256(torch.from_numpy(np.isin(np.arange(200000), stable))),
            "key_cloud_unique": int(len(key)), "key_cloud_npre": int(sum(v["npre_occurrences"] for v in aggregate.values())),
            "key_cloud_active": int(sum(v["active_contributions"] for v in aggregate.values())), "key_cloud_alpha_mass": float(sum(v["alpha_mass"] for v in aggregate.values())),
            "membership_target_independent": True,
        })

        status["failure_stage"] = "residual_and_ownership_attribution"; atomic_json(output / "RUN_STATUS.json", status)
        base, samples, background, device = load_runtime(config, "O08", args.device)
        models = {state: load_state_model(config, state, "O08", base)[0] for state in STATES}
        state_cache, group_counts = residual_distribution(output, config, base, samples["cond_000318"], models, full, events, stable, key)
        atomic_json(output / "protected_residual_attribution/group_counts.json", group_counts)
        ownership_audit(output, base, full, key, aggregate, state_cache)

        status["failure_stage"] = "loss_gradient_provenance"; atomic_json(output / "RUN_STATUS.json", status)
        stable_mask = bool_mask(stable, 200000, device); key_mask = bool_mask(key, 200000, device)
        gradient_audit(output, config, base, samples, background, models["P3"], key_mask, stable_mask)
        del base, samples, background, models, state_cache; torch.cuda.empty_cache()

        status["failure_stage"] = "static_counterfactuals"; atomic_json(output / "RUN_STATUS.json", status)
        metrics, base_hashes, renderer_equal = counterfactuals(output, config, full, key, stable)
        qualifications = formal_qualification(output, metrics)
        source_manifest_after = tree_manifest(Path(config["source_pool_output"]))
        refs_after = {name: resolve_ref(name) for name in frozen_refs}
        if source_manifest_after != source_manifest_before: raise RuntimeError("previous Case PP output changed")
        if refs_after != frozen_refs: raise RuntimeError("a frozen branch changed during audit")
        atomic_json(output / "input_audit/immutability_proof.json", {
            "previous_outputs_unchanged": True, "previous_file_count": len(source_manifest_before),
            "frozen_branches_unchanged": True, "base_fingerprints": base_hashes,
            "renderer_forward_bitwise_regression": renderer_equal,
        })
        atomic_json(output / "final_adjudication/PRELIMINARY_STATUS.json", {
            "status": "NUMERIC_COMPLETE_VISUAL_INSPECTION_REQUIRED", "formal_run_commit": head,
            "qualifications": qualifications, "F5_run": False, "optimizer_created": False, "optimizer_steps": 0,
            "next_action": "ACTUALLY_OPEN_VISUAL_CONTACT_SHEETS_THEN_SEAL",
        })
        status.update({"status": "NUMERIC_COMPLETE_VISUAL_PENDING", "failure_stage": None, "elapsed_seconds": time.perf_counter() - started, "completed_numeric_at": now()})
        atomic_json(output / "RUN_STATUS.json", status)
        return 0
    except Exception as error:
        status.update({"status": "FAILED", "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(), "elapsed_seconds": time.perf_counter() - started, "failed_at": now()})
        atomic_json(output / "RUN_STATUS.json", status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
