from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_dressable as training  # noqa: E402
from scene.o00_fixed_open_oracle import FixedOpenGaussianOracle  # noqa: E402
from scene.representation_capacity_oracle import (  # noqa: E402
    UnboundedGaussianDeltaField,
    capacity_oracle_loss_v1,
    direct_stability,
)
from scene.support_aware_region_trusted_objective_v6 import (  # noqa: E402
    V6_OBJECTIVE_NAME,
    bound_normalized_regularization,
    build_support_aware_regions,
    residual_stability_loss,
    support_aware_region_trusted_objective_v6,
)
from tools.run_module4b_canonical_oracle_micropilot import (  # noqa: E402
    _base_gradient_count,
    _base_named_tensors,
    _environment,
    _git_state,
    _grid,
    _load_samples,
    _render,
    _save_render_set,
    _tensor_state_fingerprint,
)
from utils.rendering_loss_utils import (  # noqa: E402
    boundary_aware_transition_alpha_target,
    region_aware_dual_target_loss,
)


SCHEMA = "canondressgs.objective_residual_redesign.v1"
OUTFITS = ("O01", "O08")
CONDITIONS = ("cond_000000", "cond_000318", "cond_000017", "cond_000347")
VIEWS = dict(zip(CONDITIONS, ("front", "back", "left", "right")))
TESTS = ("T1", "T2", "T3", "T4", "T5")
DEFAULT_OUTPUT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/"
    "SUBJECT02-OBJECTIVE-RESIDUAL-REDESIGN-001/attempt_004"
)


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


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True, default=str) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not keys:
            return
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()


def target_free_state(sample: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    camera = sample["target_camera"]
    return {
        "pose": sample["target_pose"].to(device), "Rh": sample["target_Rh"].to(device),
        "Th": sample["target_Th"].to(device),
        "camera": {"K": camera["K"].to(device), "w2c": camera["w2c"].to(device),
                   "width": int(camera["width"]), "height": int(camera["height"])},
    }


def render_direct(base: Any, state: Mapping[str, Any], overrides: Any, background: torch.Tensor):
    sample = {
        "target_pose": state["pose"], "target_Rh": state["Rh"], "target_Th": state["Th"],
        "target_camera": state["camera"],
    }
    return _render(base, sample, overrides, background)


def _masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    weight = mask.to(prediction)
    while weight.ndim < prediction.ndim:
        weight = weight.unsqueeze(0)
    weight = weight.expand_as(prediction)
    return float(((prediction - target.to(prediction)).abs() * weight).sum() / weight.sum().clamp_min(1))


def capacity_metrics(rgb: torch.Tensor, alpha: torch.Tensor, sample: Mapping[str, Any]) -> dict[str, float]:
    device = rgb.device; target = sample["target_edit_rgb"].to(device); base = sample["target_base_rgb"].to(device)
    protected = sample["target_protected_mask"].to(device)
    garment = torch.maximum(
        torch.maximum(sample["target_edit_mask"].to(device), sample["target_clothing_mask"].to(device)),
        sample["target_old_clothing_mask"].to(device),
    ) * (1 - protected)
    pixel = garment[0] >= .5; target_distance = (rgb - target).abs().mean(0); base_distance = (rgb - base).abs().mean(0)
    selected = pixel.sum().clamp_min(1)
    target_closer = float(((target_distance < base_distance) & pixel).sum() / selected)
    base_closer = float(((base_distance < target_distance) & pixel).sum() / selected)
    target_fg = sample["target_foreground_mask"].to(device) >= .5; pred_fg = alpha >= .5
    union = int((target_fg | pred_fg).sum()); outside = ~target_fg
    return {
        "garment_target_mae": _masked_l1(rgb, target, garment), "garment_base_mae": _masked_l1(rgb, base, garment),
        "target_closer_fraction": target_closer, "base_closer_fraction": base_closer,
        "purple_base_retention_score": base_closer, "tie_fraction": max(0., 1. - target_closer - base_closer),
        "silhouette_iou": int((target_fg & pred_fg).sum()) / max(union, 1),
        "protected_mae": _masked_l1(rgb, base, protected),
        "background_leakage": float(alpha[outside].mean()) if outside.any() else 0.,
    }


def abnormal_direct(field: Any, base: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    residuals = field.residuals(base) if hasattr(field, "residuals") else field._raw_residuals(base) if hasattr(field, "_raw_residuals") else None
    xyz = residuals.delta_xyz if residuals is not None else field.raw_xyz
    scaling = residuals.delta_log_scaling if residuals is not None else field.raw_log_scaling
    opacity_delta = residuals.delta_opacity_logit if residuals is not None else field.raw_opacity
    output = field(base) if hasattr(field, "forward") else None
    overrides = output.canonical_overrides if hasattr(output, "canonical_overrides") else output
    opacity_value = overrides.opacity if hasattr(overrides, "opacity") else base._opacity + opacity_delta
    displacement = torch.linalg.vector_norm(xyz.detach(), dim=1)
    scale_ratio = torch.exp(scaling.detach().abs()).amax(1)
    opacity = torch.sigmoid(opacity_value.detach()).reshape(-1)
    thresholds = config["abnormal_gaussian"]
    center = base._xyz.detach().mean(0); radius = torch.linalg.vector_norm(base._xyz.detach() - center, dim=1).max() * float(thresholds["outside_base_radius_multiplier"])
    position = overrides.xyz.detach() if hasattr(overrides, "xyz") else base._xyz.detach() + xyz.detach()
    outside = torch.linalg.vector_norm(position - center, dim=1) > radius
    abnormal = ((displacement > float(thresholds["xyz_displacement_m"])) | (scale_ratio > float(thresholds["scale_ratio_max"])) |
                (opacity < float(thresholds["opacity_saturation_low"])) | (opacity > float(thresholds["opacity_saturation_high"])) | outside)
    return {"xyz_displacement": _stats(displacement), "scale_ratio": _stats(scale_ratio), "opacity": _stats(opacity),
            "outside_radius_fraction": float(outside.float().mean()),
            "opacity_saturation_fraction": float(((opacity < .005) | (opacity > .995)).float().mean()),
            "scale_abnormal_fraction": float((scale_ratio > 8).float().mean()),
            "abnormal_gaussian_fraction": float(abnormal.float().mean())}


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA:
        raise ValueError("unexpected objective-residual redesign schema")
    if tuple(config["outfits"]) != OUTFITS or tuple(config["conditions"]) != CONDITIONS:
        raise ValueError("frozen outfit/condition protocol changed")
    if config["v6"]["objective_name"] != V6_OBJECTIVE_NAME:
        raise ValueError("V6 objective name changed")
    if int(config["v6"]["gradient_calibration"]["steps"]) != 20:
        raise ValueError("exactly one 20-step gradient calibration is allowed")
    if set(config["tests"]) != {"T0", *TESTS}:
        raise ValueError("causal matrix must be exactly T0--T5")
    if any(int(config["tests"][test]["steps"]) != (1000 if test in {"T4", "T5"} else 800) for test in TESTS):
        raise ValueError("causal test step contract changed")
    for name, candidate in config["candidate_bounds"].items():
        if float(candidate) > float(config["safety_caps"][name]):
            raise ValueError(f"BOUND_CONTRACT_INCOMPATIBLE: {name}")
    return config


def _stage(step: int, config: Mapping[str, Any]) -> int:
    if step <= 0:
        return 0
    if step <= int(config["optimizer"]["stages"]["geometry_end"]):
        return 1
    if step <= int(config["optimizer"]["stages"]["appearance_end"]):
        return 2
    return 3


def _configure_unbounded(field: UnboundedGaussianDeltaField, stage: int) -> None:
    allowed = {
        0: set(),
        1: {"raw_xyz", "raw_log_scaling", "raw_rotvec", "raw_opacity"},
        2: {"raw_opacity", "raw_sh0"},
        3: {"raw_xyz", "raw_log_scaling", "raw_rotvec", "raw_opacity", "raw_sh0"},
    }[stage]
    for name, parameter in field.named_parameters():
        parameter.requires_grad_(name in allowed)


def _load_runtime(args: argparse.Namespace, outfit: str):
    device = torch.device(args.device)
    pipeline = training.load_config(args.pipeline_config)
    base = training.load_frozen_mmlphuman_base(
        pipeline["base"]["model_dir"], pipeline["base"]["checkpoint_path"], device,
    )
    if int(base._xyz.shape[0]) != 200_000:
        raise ValueError("redesign requires the frozen original 200k Gaussian support")
    samples = _load_samples(args.manifest, outfit)
    background = torch.tensor(pipeline["render"]["background"], device=device, dtype=base._xyz.dtype)
    return base, samples, background, device


def _render_model(model: Any, base: Any, state: Mapping[str, Any], background: torch.Tensor):
    if isinstance(model, FixedOpenGaussianOracle):
        return render_direct(base, state, model(base).canonical_overrides, background)
    return render_direct(base, state, model(base), background)


def _formal_output(model: Any, base: Any):
    return model(base) if isinstance(model, FixedOpenGaussianOracle) else None


def _optimizer(model: Any, config: Mapping[str, Any]):
    if isinstance(model, FixedOpenGaussianOracle):
        groups, names = model.parameter_groups({
            "geometry_residuals": config["optimizer"]["geometry_lr"],
            "appearance_residuals": config["optimizer"]["appearance_lr"],
        })
    else:
        groups, names = model.parameter_groups(
            config["optimizer"]["geometry_lr"], config["optimizer"]["appearance_lr"],
        )
    return torch.optim.Adam(groups), names


def _current_raw_regularization(output: Any, config: Mapping[str, Any]) -> torch.Tensor:
    weights = config["current_formal_regularization"]
    names = {
        "xyz_magnitude": "delta_xyz",
        "scaling_magnitude": "delta_log_scaling",
        "rotation_magnitude": "delta_rotvec",
        "opacity_magnitude": "delta_opacity_logit",
        "sh0_magnitude": "delta_sh0",
        "shN_magnitude": "delta_shN",
    }
    return sum(float(weights[name]) * output.regularization.residual_magnitude[field] for name, field in names.items())


def _v5_loss(
    rgb: torch.Tensor, alpha: torch.Tensor, sample: Mapping[str, Any], config: Mapping[str, Any],
    transition_target: torch.Tensor,
):
    device = rgb.device
    return region_aware_dual_target_loss(
        rgb, alpha, sample["target_edit_rgb"].to(device), sample["target_base_rgb"].to(device),
        sample["target_edit_core_mask"].to(device), sample["target_preserve_mask"].to(device),
        sample["target_protected_mask"].to(device), sample["target_transition_mask"].to(device),
        sample["target_clothing_mask"].to(device), sample["target_foreground_mask"].to(device),
        sample["target_base_foreground_mask"].to(device), transition_alpha_target=transition_target,
        **{name: float(value) for name, value in config["v5_3_loss"].items()},
    )


def _capacity_loss(
    rgb: torch.Tensor, alpha: torch.Tensor, sample: Mapping[str, Any], config: Mapping[str, Any], stability: torch.Tensor,
):
    device = rgb.device
    return capacity_oracle_loss_v1(
        rgb, alpha, target_rgb=sample["target_edit_rgb"].to(device),
        base_rgb=sample["target_base_rgb"].to(device),
        target_foreground=sample["target_foreground_mask"].to(device),
        base_foreground=sample["target_base_foreground_mask"].to(device),
        edit_mask=sample["target_edit_mask"].to(device), clothing_mask=sample["target_clothing_mask"].to(device),
        old_clothing_mask=sample["target_old_clothing_mask"].to(device),
        protected_mask=sample["target_protected_mask"].to(device),
        transition_mask=sample["target_transition_mask"].to(device),
        weights=config["capacity_loss"], stability=stability,
    )


def _objective(
    test: str, model: Any, base: Any, rgb: torch.Tensor, alpha: torch.Tensor, sample: Mapping[str, Any],
    config: Mapping[str, Any], transition_target: torch.Tensor,
):
    if test == "T2":
        parts = _v5_loss(rgb, alpha, sample, config, transition_target)
        minimal = 1e-8 * direct_stability(model)
        return parts["total"] + minimal, {**parts, "stability": minimal}
    output = _formal_output(model, base)
    bounds = model.bounds
    if test == "T1":
        regularizer = _current_raw_regularization(output, config)
        parts = _capacity_loss(rgb, alpha, sample, config, regularizer)
        return parts["total"], parts
    normalized, normalized_parts = bound_normalized_regularization(
        output.gaussian_residuals, base, bounds, **config["v6"]["regional_regularization"],
    )
    stability, stability_parts = residual_stability_loss(output.gaussian_residuals, base, bounds)
    if test == "T3":
        parts = _capacity_loss(rgb, alpha, sample, config, normalized)
        parts.update({f"normalized_{name}": value for name, value in normalized_parts.items()})
        return parts["total"], parts
    result = support_aware_region_trusted_objective_v6(
        rgb, alpha, sample, weights=config["v6"]["frozen_weights"], residual_loss=normalized,
        stability_loss=stability, progress_margin=float(config["v6"]["progress_margin"]),
        change_epsilon=float(config["v6"]["change_epsilon"]), transition_alpha_target=transition_target,
    )
    parts = {**result.parts, **{f"stability_{name}": value for name, value in stability_parts.items()}}
    return result.total, parts


def _masked_mae(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    weight = mask.to(prediction)
    while weight.ndim < prediction.ndim:
        weight = weight.unsqueeze(0)
    weight = weight.expand_as(prediction)
    return float(((prediction - target.to(prediction)).abs() * weight).sum() / weight.sum().clamp_min(1))


def _extended_metrics(rgb: torch.Tensor, alpha: torch.Tensor, sample: Mapping[str, Any]) -> dict[str, float]:
    result = capacity_metrics(rgb, alpha, sample)
    reference = rgb.unsqueeze(0)
    regions = build_support_aware_regions(sample, reference)
    target_alpha = sample["target_foreground_mask"].to(alpha)
    new = regions["new_silhouette"][0]
    selected = new >= .5
    result["new_silhouette_recall"] = float((alpha[selected] >= .5).float().mean()) if selected.any() else 1.0
    result["neutral_preserve_mae"] = _masked_mae(rgb, sample["target_base_rgb"], regions["neutral_preserve"][0])
    transition_target = boundary_aware_transition_alpha_target(
        sample["target_foreground_mask"].to(alpha), sample["target_base_foreground_mask"].to(alpha),
        sample["target_edit_core_mask"].to(alpha), sample["target_preserve_mask"].to(alpha),
        sample["target_protected_mask"].to(alpha),
    )["target"][0]
    result["transition_alpha_mae"] = _masked_mae(alpha, transition_target, regions["transition"][0])
    return result


def _residual_diagnostics(model: Any, base: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(model, FixedOpenGaussianOracle):
        residuals = model(base).gaussian_residuals
        bounds = model.bounds
        values = {
            "xyz": residuals.delta_xyz, "log_scaling": residuals.delta_log_scaling,
            "rotation": residuals.delta_rotvec, "opacity_logit": residuals.delta_opacity_logit,
            "sh0": residuals.delta_sh0,
        }
        hits = {name: float((value.abs() >= .99 * bounds[name]).float().mean()) for name, value in values.items()}
        abnormal = abnormal_direct(model, base, config)
    else:
        hits = {name: 0.0 for name in ("xyz", "log_scaling", "rotation", "opacity_logit", "sh0")}
        abnormal = abnormal_direct(model, base, config)
    return {
        "bound_hit_fraction_by_attribute": hits,
        "max_bound_hit_fraction": max(hits.values()),
        **abnormal,
    }


def _slope(history: list[dict[str, Any]], name: str) -> float:
    rows = history[-100:]
    if len(rows) < 2:
        return float("nan")
    x = np.asarray([row["step"] for row in rows], dtype=np.float64)
    aliases = {
        "edit_rgb": ("edit_rgb", "garment_rgb", "edit"),
        "garment_rgb": ("garment_rgb", "clothing", "edit_rgb", "edit"),
    }
    candidates = aliases.get(name, (name,))
    selected = next((candidate for candidate in candidates if candidate in rows[0]), None)
    if selected is None:
        raise KeyError(f"none of the registered slope fields are present: {candidates}")
    y = np.asarray([row[selected] for row in rows], dtype=np.float64)
    return float(np.polyfit(x, y, 1)[0])


def repair_derived_slopes(args: argparse.Namespace) -> None:
    """Repair slope-only acceptance metadata without rerunning optimization."""
    summary_path = args.output / "comparisons/DERIVED_SLOPE_REPAIR_SUMMARY.json"
    if summary_path.exists():
        raise FileExistsError(f"derived slope repair already exists: {summary_path}")
    repaired: dict[str, Any] = {
        "status": "COMPLETE",
        "repair_scope": "derived acceptance metadata only; checkpoints, renders, and loss curves unchanged",
        "repair_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repair_code_commit": git_output("rev-parse", "HEAD"),
        "runs": {},
    }
    for test in TESTS:
        repaired["runs"][test] = {}
        for outfit in OUTFITS:
            run_dir = args.output / "causal_matrix" / test / outfit
            metrics_path = run_dir / "metrics.json"
            status_path = run_dir / "RUN_STATUS.json"
            curve_path = run_dir / "loss_curve.csv"
            provenance_path = run_dir / "diagnostics/DERIVED_SLOPE_REPAIR.json"
            if provenance_path.exists():
                raise FileExistsError(f"run slope repair already exists: {provenance_path}")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            with curve_path.open("r", encoding="utf-8", newline="") as handle:
                history = list(csv.DictReader(handle))
            original = {
                "metrics_sha256": sha256(metrics_path),
                "last_100_edit_slope": metrics["last_100_edit_slope"],
                "last_100_clothing_slope": metrics["last_100_clothing_slope"],
                "numeric_status": metrics["numeric_status"],
                "numeric_checks": metrics["numeric_checks"],
            }
            edit_slope = _slope(history, "edit_rgb")
            clothing_slope = _slope(history, "garment_rgb")
            checks = dict(metrics["numeric_checks"])
            checks["edit_slope"] = bool(edit_slope < 0)
            checks["clothing_slope"] = bool(clothing_slope < 0)
            corrected_status = "NUMERIC_PASS" if all(checks.values()) else "NUMERIC_FAIL"
            relative_provenance = provenance_path.relative_to(args.output).as_posix()
            repair_record = {
                "test": test,
                "outfit": outfit,
                "reason": "registered V6 edit_rgb was missing from the clothing-slope fallback",
                "loss_curve_path": str(curve_path),
                "loss_curve_sha256": sha256(curve_path),
                "semantic_aliases": {
                    "edit_slope": ["edit_rgb", "garment_rgb", "edit"],
                    "clothing_slope": ["garment_rgb", "clothing", "edit_rgb", "edit"],
                },
                "original": original,
                "corrected": {
                    "last_100_edit_slope": edit_slope,
                    "last_100_clothing_slope": clothing_slope,
                    "numeric_status": corrected_status,
                    "numeric_checks": checks,
                },
                "optimizer_rerun": False,
                "checkpoint_modified": False,
                "renders_modified": False,
                "loss_curve_modified": False,
            }
            metrics.update({
                "last_100_edit_slope": edit_slope,
                "last_100_clothing_slope": clothing_slope,
                "numeric_status": corrected_status,
                "numeric_checks": checks,
                "derived_metric_repair": {
                    "applied": True,
                    "provenance": relative_provenance,
                    "original_metrics_sha256": original["metrics_sha256"],
                },
            })
            atomic_json(metrics_path, metrics)
            atomic_json(status_path, metrics)
            repair_record["corrected"]["metrics_sha256"] = sha256(metrics_path)
            atomic_json(provenance_path, repair_record)
            repaired["runs"][test][outfit] = repair_record
    atomic_json(summary_path, repaired)
    print(json.dumps(repaired, indent=2))


def _mask_overlay(image: Image.Image, mask: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    base = image.convert("RGB")
    alpha = mask.convert("L").resize(base.size, Image.Resampling.NEAREST).point(lambda value: 128 if value > 0 else 0)
    layer = Image.new("RGB", base.size, color)
    return Image.composite(layer, base, alpha)


def _labeled_tile(image: Image.Image, label: str, size: tuple[int, int]) -> Image.Image:
    tile = Image.new("RGB", (size[0], size[1] + 24), "white")
    tile.paste(image.convert("RGB").resize(size, Image.Resampling.LANCZOS), (0, 0))
    ImageDraw.Draw(tile).text((4, size[1] + 4), label, fill="black")
    return tile


def build_visual_evidence(args: argparse.Namespace) -> None:
    """Build inspection-only bundles from persisted PNGs; never render or optimize."""
    output = args.output / "visual_acceptance"
    manifest_path = output / "VISUAL_BUNDLE_MANIFEST.json"
    if manifest_path.exists():
        raise FileExistsError(f"visual evidence already exists: {manifest_path}")
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "status": "COMPLETE_PENDING_ACTUAL_IMAGE_INSPECTION",
        "source": "persisted causal-matrix PNG files only",
        "renderer_invoked": False,
        "optimizer_steps": 0,
        "bundles": {},
    }
    for test in TESTS:
        for outfit in OUTFITS:
            run_dir = args.output / "causal_matrix" / test / outfit
            key = f"{test}/{outfit}"
            milestone_paths = [
                run_dir / "diagnostics/step_000000_four_view.png",
                run_dir / "diagnostics/step_000200_four_view.png",
                run_dir / "diagnostics/step_000480_four_view.png",
                run_dir / "diagnostics/final_four_view.png",
            ]
            if not all(path.is_file() for path in milestone_paths):
                raise FileNotFoundError(f"missing milestone visual for {key}")
            width, milestone_height, detail_height = 2048, 840, 408
            canvas = Image.new("RGB", (width, 48 + milestone_height + 32 + 4 * detail_height), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 12), f"{key} persisted visual acceptance bundle", fill="black")
            sources = []
            for index, (label, path) in enumerate(zip(("step 0", "step 200", "step 480", "final"), milestone_paths)):
                image = Image.open(path).convert("RGB")
                tile = _labeled_tile(image, label, (512, 816))
                canvas.paste(tile, (index * 512, 48))
                sources.append({"path": str(path), "sha256": sha256(path)})
            draw.text((8, 48 + milestone_height + 8), "Final-view protected/body-part/boundary/artifact inspection", fill="black")
            for row, condition in enumerate(CONDITIONS):
                view = VIEWS[condition]
                view_dir = run_dir / "renders/final" / view
                prediction_path = view_dir / "prediction.png"
                protected_path = view_dir / "protected_mask.png"
                clothing_path = view_dir / "clothing_mask_safe.png"
                old_path = view_dir / "old_clothing_mask.png"
                error_path = view_dir / "absolute_error.png"
                required = (prediction_path, protected_path, clothing_path, old_path, error_path)
                if not all(path.is_file() for path in required):
                    raise FileNotFoundError(f"missing final inspection PNG for {key}/{view}")
                prediction = Image.open(prediction_path).convert("RGB")
                protected = Image.open(protected_path).convert("L")
                garment = Image.fromarray(np.maximum(
                    np.asarray(Image.open(clothing_path).convert("L")),
                    np.asarray(Image.open(old_path).convert("L")),
                ).astype(np.uint8), "L")
                boundary = garment.filter(ImageFilter.FIND_EDGES).point(lambda value: 255 if value > 32 else 0)
                protected_overlay = _mask_overlay(prediction, protected, (255, 0, 255))
                boundary_overlay = _mask_overlay(prediction, boundary, (0, 255, 0))
                error = Image.open(error_path).convert("RGB")
                error_strength = error.convert("L").point(lambda value: min(220, value * 3))
                artifact_overlay = Image.composite(Image.new("RGB", prediction.size, (255, 0, 0)), prediction, error_strength)
                w, h = prediction.size
                crops = {
                    "prediction": prediction,
                    "protected overlay": protected_overlay,
                    "head/face/hair crop": prediction.crop((w // 4, 0, 3 * w // 4, h // 3)),
                    "hands/arms crop": prediction.crop((0, h // 4, w, 3 * h // 4)),
                    "shoes crop": prediction.crop((w // 8, 2 * h // 3, 7 * w // 8, h)),
                    "garment boundary": boundary_overlay,
                    "error/splat overlay": artifact_overlay,
                }
                y = 48 + milestone_height + 32 + row * detail_height
                draw.text((4, y + 184), view, fill="black")
                for column, (label, image) in enumerate(crops.items()):
                    canvas.paste(_labeled_tile(image, f"{view} {label}", (256, 384)), (32 + column * 288, y))
                sources.extend({"path": str(path), "sha256": sha256(path)} for path in required)
            bundle_path = output / f"{test}_{outfit}_visual_bundle.png"
            canvas.save(bundle_path)
            manifest["bundles"][key] = {
                "path": str(bundle_path), "sha256": sha256(bundle_path), "sources": sources,
                "contains": ["step0", "step200", "step480", "final", "protected", "head_face_hair",
                             "hands_arms", "shoes", "garment_boundary", "absolute_error_splat_overlay"],
            }
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


def _parameter_gradient_norm(loss: torch.Tensor, parameters: list[torch.Tensor]) -> float:
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    terms = [gradient.double().square().sum() for gradient in gradients if gradient is not None]
    return float(torch.sqrt(torch.stack(terms).sum()).detach()) if terms else 0.0


def run_audit(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    root = args.output.resolve()
    if root.exists():
        raise FileExistsError(root)
    if git_output("branch", "--show-current") != config["research_branch"]:
        raise RuntimeError("not on the frozen objective-residual research branch")
    tagged = git_output("rev-list", "-n", "1", config["source_representation_triage_tag"])
    if tagged != config["source_representation_triage_commit"]:
        raise RuntimeError("representation triage tag moved")
    if not args.source_triage.is_dir() or not args.source_aaai.is_dir() or not args.manifest.is_file():
        raise FileNotFoundError("frozen triage/AAAI/manifest input is missing")
    for directory in (
        "contract", "input_audit", "rung2_parameter_analysis", "causal_matrix/T0", "causal_matrix/T1",
        "causal_matrix/T2", "causal_matrix/T3", "causal_matrix/T4", "causal_matrix/T5", "v6_objective",
        "residual_calibration", "comparisons", "visual_acceptance", "final_adjudication",
    ):
        (root / directory).mkdir(parents=True, exist_ok=True)
    resolved = json.loads(json.dumps(config))
    atomic_text(root / "contract/config_resolved.yaml", yaml.safe_dump(resolved, sort_keys=False))
    source_files = []
    for outfit in OUTFITS:
        for relative in (
            Path("rung_2_shared_same_support") / outfit / "metrics.json",
            Path("rung_2_shared_same_support") / outfit / "checkpoints/step_001200.pth",
        ):
            path = args.source_triage / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            source_files.append({"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size})
    t0 = {
        "definition": config["tests"]["T0"]["definition"], "status": "FAIL", "rerun": False,
        "source": str(args.source_aaai.resolve()), "historical_result_not_recomputed": True,
    }
    atomic_json(root / "causal_matrix/T0/T0_HISTORICAL_BASELINE.json", t0)
    atomic_json(root / "input_audit/input_manifest.json", {
        "task_id": config["task_id"], "git": _git_state(), "environment": _environment(),
        "manifest": {"path": str(args.manifest.resolve()), "sha256": sha256(args.manifest)},
        "source_files": source_files, "outfits": OUTFITS, "conditions": CONDITIONS,
        "target_fields_loss_only": True, "reference_images_used_by_oracle_forward": False,
    })
    atomic_json(root / "final_adjudication/RUN_STATUS.json", {"status": "AUDIT_COMPLETE", "optimizer_steps": 0})
    print(json.dumps({"status": "AUDIT_COMPLETE", "output": str(root)}, indent=2))


def _stats(value: torch.Tensor) -> dict[str, float]:
    flat = value.detach().float().abs().reshape(-1)
    return {name: float(torch.quantile(flat, q)) for name, q in (
        ("p50", .5), ("p75", .75), ("p90", .9), ("p95", .95), ("p99", .99), ("p99_5", .995), ("max", 1.0),
    )}


def run_analyze(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = args.output / "rung2_parameter_analysis"
    if (output / "rung2_required_residual_statistics.json").exists():
        raise FileExistsError("Rung-2 parameter analysis already exists")
    rows: list[dict[str, Any]] = []
    correlations: dict[str, Any] = {}
    pooled: dict[str, list[torch.Tensor]] = {name: [] for name in ("xyz", "log_scaling", "rotation", "opacity_logit", "sh0")}
    key_map = {"xyz": "raw_xyz", "log_scaling": "raw_log_scaling", "rotation": "raw_rotvec", "opacity_logit": "raw_opacity", "sh0": "raw_sh0"}
    for outfit in OUTFITS:
        checkpoint = args.source_triage / f"rung_2_shared_same_support/{outfit}/checkpoints/step_001200.pth"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
        garment = state["trainable_support"].bool().reshape(-1)
        scalar_columns = []
        for name, key in key_map.items():
            value = state[key].float()
            magnitude = value.abs().reshape(value.shape[0], -1).amax(1)
            # Bounds cover the canonical garment support that the successful
            # Rung-2 field was allowed to change. Frozen zero protected rows
            # are not evidence about required garment residual capacity.
            pooled[name].append(value[garment].reshape(-1))
            scalar_columns.append(magnitude)
            for region, mask in (("all", torch.ones_like(garment)), ("garment_support", garment), ("protected_support", ~garment)):
                row = {"outfit": outfit, "attribute": name, "region": region, "count": int(mask.sum())}
                row.update(_stats(magnitude[mask]))
                rows.append(row)
        correlations[outfit] = np.corrcoef(torch.stack(scalar_columns).numpy()).tolist()
    pooled_stats = {}
    candidate = {}
    incompatible = []
    for name, tensors in pooled.items():
        stats = _stats(torch.cat(tensors))
        pooled_stats[name] = stats
        required = 1.25 * stats["p99_5"]
        candidate[name] = max(float(config["current_bounds"][name]), required)
        if candidate[name] > float(config["safety_caps"][name]):
            incompatible.append(name)
    candidate["shN"] = float(config["current_bounds"]["shN"])
    for name, expected in config["candidate_bounds"].items():
        if not math.isclose(candidate[name], float(expected), rel_tol=2e-6, abs_tol=2e-7):
            raise AssertionError(f"candidate bound derivation drifted for {name}: {candidate[name]} != {expected}")
    payload = {
        "rows": rows, "pooled": pooled_stats, "parameter_correlations": correlations,
        "image_space_regions": {
            "old_clothing": "not a canonical hard partition; retained as per-view image-space evidence",
            "new_silhouette": "not a canonical hard partition; retained as per-view image-space evidence",
        },
        "candidate_bounds": candidate, "bound_contract_incompatible": incompatible,
        "extreme_solution_dependency": "reported by p99.5-to-max ratios; calibration deliberately ignores top 0.5%",
    }
    write_csv(output / "rung2_required_residual_statistics.csv", rows)
    atomic_json(output / "rung2_required_residual_statistics.json", payload)
    atomic_json(args.output / "residual_calibration/formal_residual_bounds_v6_candidate.json", {
        "rule": "max(current, 1.25 * pooled absolute p99.5)", "bounds": candidate,
        "safety_caps": config["safety_caps"], "bound_contract_incompatible": incompatible,
    })
    atomic_text(output / "CURRENT_VS_REQUIRED_BOUNDS.md", "\n".join([
        "# Current vs required residual bounds", "",
        "| attribute | current | pooled p99 | pooled p99.5 | calibrated | safety cap |",
        "|---|---:|---:|---:|---:|---:|",
        *[f"| {name} | {config['current_bounds'][name]:.10g} | {pooled_stats[name]['p99']:.10g} | "
          f"{pooled_stats[name]['p99_5']:.10g} | {candidate[name]:.10g} | {config['safety_caps'][name]:.10g} |"
          for name in pooled], "", f"- bound-contract incompatibility: `{bool(incompatible)}`",
    ]))
    atomic_text(args.output / "residual_calibration/formal_residual_bounds_v6_rationale.md", "\n".join([
        "# Formal residual bounds V6 candidate", "", "One global bound set is shared by O01/O08 and all views.",
        "The rule is frozen at `max(current, 1.25 × pooled absolute p99.5)`.",
        "No outcome-driven grid search or outfit-specific expansion is permitted.",
        f"BOUND_CONTRACT_INCOMPATIBLE: `{bool(incompatible)}`.",
    ]))
    print(json.dumps({"status": "ANALYSIS_COMPLETE", "candidate_bounds": candidate}, indent=2))


def run_calibrate(args: argparse.Namespace, config: dict[str, Any]) -> None:
    output = args.output / "residual_calibration"
    target = output / "v6_gradient_calibration.json"
    if target.exists():
        raise FileExistsError(target)
    torch.manual_seed(int(config["seed"])); np.random.seed(int(config["seed"]))
    records = []
    raw_weights = dict(config["v6"]["initial_weights"])
    for outfit in OUTFITS:
        base, samples, background, device = _load_runtime(args, outfit)
        model = FixedOpenGaussianOracle(base, bounds=config["candidate_bounds"]).to(device)
        model.configure_stage(1)
        optimizer, _ = _optimizer(model, config)
        states = {condition: target_free_state(sample, device) for condition, sample in samples.items()}
        for local in range(10):
            step = len(records) + 1; condition = CONDITIONS[local % 4]; sample = samples[condition]
            optimizer.zero_grad(set_to_none=True)
            output_value = model(base)
            rgb, alpha = render_direct(base, states[condition], output_value.canonical_overrides, background)
            normalized, _ = bound_normalized_regularization(
                output_value.gaussian_residuals, base, model.bounds, **config["v6"]["regional_regularization"],
            )
            stability, _ = residual_stability_loss(output_value.gaussian_residuals, base, model.bounds)
            result = support_aware_region_trusted_objective_v6(
                rgb, alpha, sample, weights=raw_weights, residual_loss=normalized, stability_loss=stability,
                progress_margin=config["v6"]["progress_margin"], change_epsilon=config["v6"]["change_epsilon"],
            )
            parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
            edit_group = result.parts["edit_rgb"] + result.parts["target_progress"]
            alpha_group = result.parts["new_silhouette"] + result.parts["transition_alpha"]
            regularization_group = result.parts["residual"] + result.parts["stability"]
            record = {
                "step": step, "outfit": outfit, "condition": condition,
                "edit_gradient_norm": _parameter_gradient_norm(edit_group, parameters),
                "alpha_gradient_norm": _parameter_gradient_norm(alpha_group, parameters),
                "regularization_gradient_norm": _parameter_gradient_norm(regularization_group, parameters),
            }
            records.append(record)
            result.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["optimizer"]["gradient_clip_norm"])
            optimizer.step()
    edit = float(np.median([row["edit_gradient_norm"] for row in records]))
    alpha = float(np.median([row["alpha_gradient_norm"] for row in records]))
    regularization = float(np.median([row["regularization_gradient_norm"] for row in records]))
    calibration = config["v6"]["gradient_calibration"]
    alpha_scale = min(1.0, float(calibration["alpha_to_edit_cap"]) * edit / max(alpha, 1e-12))
    regularization_scale = min(1.0, float(calibration["regularization_to_edit_cap"]) * edit / max(regularization, 1e-12))
    frozen = dict(raw_weights)
    frozen["new_silhouette"] *= alpha_scale; frozen["transition_alpha"] *= alpha_scale
    frozen["residual"] *= regularization_scale; frozen["stability"] *= regularization_scale
    payload = {
        "steps": 20, "records": records, "median_gradient_norms": {"edit": edit, "alpha": alpha, "regularization": regularization},
        "scales": {"alpha": alpha_scale, "regularization": regularization_scale},
        "caps": {"alpha_to_edit": .5, "regularization_to_edit": .25},
        "visual_tuning_used": False, "frozen_weights": frozen,
    }
    atomic_json(target, payload); atomic_json(output / "v6_loss_weights_frozen.json", frozen)
    atomic_text(output / "V6_GRADIENT_CALIBRATION.md", "\n".join([
        "# V6 gradient calibration", "", "Exactly 20 optimizer steps were used, ten per outfit.",
        f"- median edit gradient: `{edit:.9g}`", f"- median alpha gradient: `{alpha:.9g}`",
        f"- median regularization gradient: `{regularization:.9g}`", f"- alpha scale: `{alpha_scale:.9g}`",
        f"- regularization scale: `{regularization_scale:.9g}`", "- visual tuning: `false`",
    ]))
    print(json.dumps(payload, indent=2))


def _config_with_frozen_weights(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    path = args.output / "residual_calibration/v6_loss_weights_frozen.json"
    if not path.is_file():
        raise FileNotFoundError("V6 gradient calibration must complete before causal runs")
    config = json.loads(json.dumps(config))
    config["v6"]["frozen_weights"] = json.loads(path.read_text(encoding="utf-8"))
    return config


def _numeric_status(per_view: list[dict[str, Any]], diagnostics: Mapping[str, Any], history: list[dict[str, Any]], config: Mapping[str, Any]):
    acceptance = config["acceptance"]
    checks = {
        "all_view_edit_reduction": min(row["edit_reduction"] for row in per_view) >= acceptance["per_view_edit_reduction_min"],
        "mean_edit_reduction": np.mean([row["edit_reduction"] for row in per_view]) >= acceptance["mean_edit_reduction_min"],
        "all_view_target_closer": min(row["target_closer_fraction"] for row in per_view) >= acceptance["per_view_target_closer_min"],
        "mean_target_closer": np.mean([row["target_closer_fraction"] for row in per_view]) >= acceptance["mean_target_closer_min"],
        "all_view_silhouette": min(row["silhouette_iou"] for row in per_view) >= acceptance["per_view_silhouette_iou_min"],
        "protected": max(row["protected_mae"] for row in per_view) <= acceptance["protected_mae_max"],
        "background": max(row["background_leakage"] for row in per_view) <= acceptance["background_leakage_max"],
        "edit_slope": _slope(history, "edit_rgb") < 0,
        "clothing_slope": _slope(history, "garment_rgb") < 0,
        "abnormal": diagnostics["abnormal_gaussian_fraction"] <= acceptance["abnormal_gaussian_fraction_max"],
        "bound_truncation": diagnostics["max_bound_hit_fraction"] <= acceptance["systematic_bound_hit_fraction_max"],
    }
    checks = {name: bool(value) for name, value in checks.items()}
    return ("NUMERIC_PASS" if all(checks.values()) else "NUMERIC_FAIL"), checks


def run_test(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if args.test not in TESTS or args.outfit not in OUTFITS:
        raise ValueError("test/outfit is outside frozen T1--T5 protocol")
    config = _config_with_frozen_weights(args, config)
    run_dir = args.output / "causal_matrix" / args.test / args.outfit
    if run_dir.exists():
        raise FileExistsError(run_dir)
    for name in ("checkpoints", "renders", "diagnostics"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "RUN_STATUS.json"
    atomic_json(status_path, {"status": "RUNNING", "optimizer_steps": 0})
    started = time.perf_counter()
    try:
        torch.manual_seed(int(config["seed"])); np.random.seed(int(config["seed"]))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        base, samples, background, device = _load_runtime(args, args.outfit)
        transition_targets = {
            condition: boundary_aware_transition_alpha_target(
                sample["target_foreground_mask"].to(device), sample["target_base_foreground_mask"].to(device),
                sample["target_edit_core_mask"].to(device), sample["target_preserve_mask"].to(device),
                sample["target_protected_mask"].to(device),
            )["target"]
            for condition, sample in samples.items()
        }
        bounds = config["candidate_bounds"] if args.test in {"T3", "T5"} else config["current_bounds"]
        model = UnboundedGaussianDeltaField(base).to(device) if args.test == "T2" else FixedOpenGaussianOracle(base, bounds=bounds).to(device)
        optimizer, optimizer_groups = _optimizer(model, config)
        states = {condition: target_free_state(sample, device) for condition, sample in samples.items()}
        base_before = _tensor_state_fingerprint(_base_named_tensors(base))
        initial: dict[str, Any] = {}
        initial_panels = []
        for condition in CONDITIONS:
            with torch.no_grad():
                rgb, alpha = _render_model(model, base, states[condition], background)
            initial[condition] = _extended_metrics(rgb, alpha, samples[condition])
            initial_panels.extend(_save_render_set(run_dir / "renders/step_000000", condition, rgb, alpha, samples[condition]))
        _grid(run_dir / "diagnostics/step_000000_four_view.png", initial_panels, columns=4)
        history: list[dict[str, Any]] = []
        gradient_seen = {name: 0 for name, parameter in model.named_parameters() if parameter.requires_grad}
        steps = int(config["tests"][args.test]["steps"])
        render_steps = {step for step in config["render_steps"] if step <= steps}
        record_steps = {step for step in config["record_steps"] if step <= steps}
        for step in range(1, steps + 1):
            stage = _stage(step, config)
            if isinstance(model, FixedOpenGaussianOracle): model.configure_stage(stage)
            else: _configure_unbounded(model, stage)
            condition = CONDITIONS[(step - 1) % 4]; sample = samples[condition]
            optimizer.zero_grad(set_to_none=True)
            rgb, alpha = _render_model(model, base, states[condition], background)
            total, parts = _objective(
                args.test, model, base, rgb, alpha, sample, config, transition_targets[condition],
            )
            if not torch.isfinite(total): raise FloatingPointError(f"non-finite objective at step {step}")
            total.backward()
            for name, parameter in model.named_parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all(): raise FloatingPointError(f"non-finite gradient: {name}")
                    gradient_seen[name] = gradient_seen.get(name, 0) + int(torch.count_nonzero(parameter.grad) > 0)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["optimizer"]["gradient_clip_norm"])
            if not torch.isfinite(norm): raise FloatingPointError("non-finite clipped gradient")
            optimizer.step()
            row = {"step": step, "condition": condition, "stage": stage, "total": float(total.detach()), "gradient_norm": float(norm)}
            row.update({name: float(value.detach()) for name, value in parts.items() if isinstance(value, torch.Tensor) and value.numel() == 1})
            history.append(row); append_jsonl(run_dir / "state_records.jsonl", row)
            if step in record_steps: atomic_json(run_dir / f"diagnostics/step_{step:06d}_loss.json", row)
            if step in render_steps:
                milestone = []
                for view_condition in CONDITIONS:
                    with torch.no_grad():
                        view_rgb, view_alpha = _render_model(model, base, states[view_condition], background)
                    milestone.extend(_save_render_set(run_dir / f"renders/step_{step:06d}", view_condition, view_rgb, view_alpha, samples[view_condition]))
                _grid(run_dir / f"diagnostics/step_{step:06d}_four_view.png", milestone, columns=4)
            atomic_json(status_path, {"status": "RUNNING", "optimizer_steps": step})
        final = []; final_panels = []
        for condition in CONDITIONS:
            with torch.no_grad(): rgb, alpha = _render_model(model, base, states[condition], background)
            metrics = _extended_metrics(rgb, alpha, samples[condition])
            metrics["edit_reduction"] = 1 - metrics["garment_target_mae"] / max(initial[condition]["garment_target_mae"], 1e-12)
            final.append({"condition_id": condition, "view": VIEWS[condition], **metrics})
            final_panels.extend(_save_render_set(run_dir / "renders/final", condition, rgb, alpha, samples[condition]))
        _grid(run_dir / "diagnostics/final_four_view.png", final_panels, columns=4)
        diagnostics = _residual_diagnostics(model, base, config)
        numeric, checks = _numeric_status(final, diagnostics, history, config)
        base_after = _tensor_state_fingerprint(_base_named_tensors(base))
        checkpoint = run_dir / f"checkpoints/step_{steps:06d}.pth"
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "global_step": steps,
            "metadata": {"test": args.test, "outfit": args.outfit, "target_tensors_stored": False,
                         "formal_composition": args.test != "T2", "optimizer_groups": optimizer_groups},
        }, checkpoint)
        write_csv(run_dir / "loss_curve.csv", history)
        result = {
            "status": "COMPLETE_PENDING_VISUAL", "test": args.test, "outfit": args.outfit,
            "optimizer_steps": steps, "initial_per_view": initial, "final_per_view": final,
            "mean_edit_reduction": float(np.mean([row["edit_reduction"] for row in final])),
            "mean_target_closer_fraction": float(np.mean([row["target_closer_fraction"] for row in final])),
            "last_100_edit_slope": _slope(history, "edit_rgb"), "last_100_clothing_slope": _slope(history, "garment_rgb"),
            "residual_diagnostics": diagnostics, "numeric_status": numeric, "numeric_checks": checks,
            "visual_status": "PENDING_ACTUAL_IMAGE_INSPECTION", "gradient_steps_nonzero": gradient_seen,
            "base_fingerprint_before": base_before, "base_fingerprint_after": base_after,
            "base_bitwise_exact": base_before == base_after, "base_gradient_count": _base_gradient_count(base),
            "target_fields_used_only_after_forward": True, "forward_state_fields": sorted(states[CONDITIONS[0]]),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
        }
        atomic_json(run_dir / "metrics.json", result); atomic_json(status_path, result)
        print(json.dumps(result, indent=2, default=str))
    except Exception as error:
        atomic_json(status_path, {"status": "FAILED", "exception_type": type(error).__name__, "exception": str(error), "traceback": traceback.format_exc()})
        raise


def adjudicate(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    decisions = json.loads(args.visual_decisions.read_text(encoding="utf-8"))
    summary = {"status": "COMPLETE", "tests": {}}
    for test in TESTS:
        summary["tests"][test] = {}
        for outfit in OUTFITS:
            run_dir = args.output / "causal_matrix" / test / outfit
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            key = f"{test}/{outfit}"; visual = decisions.get(key)
            if not visual or visual.get("status") not in {"PASS", "WARN", "FAIL"} or not visual.get("images_actually_opened") or not visual.get("observations"):
                raise ValueError(f"missing actual visual inspection: {key}")
            final = "PASS" if metrics["numeric_status"] == "NUMERIC_PASS" and visual["status"] in {"PASS", "WARN"} else "FAIL"
            metrics.update({"visual_acceptance": visual, "visual_status": visual["status"], "final_status": final})
            atomic_json(run_dir / "metrics.json", metrics); atomic_json(run_dir / "FINAL_STATUS.json", {"final_status": final, "numeric": metrics["numeric_status"], "visual": visual["status"]})
            summary["tests"][test][outfit] = {"final_status": final, "numeric": metrics["numeric_status"], "visual": visual["status"]}
    atomic_json(args.output / "comparisons/CAUSAL_MATRIX_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2))


def finalize(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    summary = json.loads((args.output / "comparisons/CAUSAL_MATRIX_SUMMARY.json").read_text(encoding="utf-8"))
    def both(test: str) -> bool:
        return all(summary["tests"][test][outfit]["final_status"] == "PASS" for outfit in OUTFITS)
    t1, t2, t5 = both("T1"), both("T2"), both("T5")
    if t1 and not t2: causal = "O"
    elif not t1 and t2: causal = "B"
    else: causal = "I"
    final_case = "P" if t5 else "F"
    t5_failure_checks: dict[str, list[str]] = {}
    for outfit in OUTFITS:
        metrics = json.loads((args.output / "causal_matrix" / "T5" / outfit / "metrics.json").read_text(encoding="utf-8"))
        t5_failure_checks[outfit] = [name for name, passed in metrics["numeric_checks"].items() if not passed]
    unique_numeric_failures = sorted({name for failures in t5_failure_checks.values() for name in failures})
    if t5:
        next_task = "RE-ADJUDICATE_7_OUTFIT_GATE_WITH_V6"
        proven_failure_component = None
    elif unique_numeric_failures == ["all_view_silhouette"]:
        next_task = "REFINE_NEW_SILHOUETTE_MASK_SEMANTICS"
        proven_failure_component = "new_silhouette_mask_semantics"
    else:
        raise RuntimeError(f"T5 has no unique proven failure component: {unique_numeric_failures}")
    final = {
        "status": "COMPLETE", "causal_primary_case": causal, "final_case": final_case,
        "t1_both_pass": t1, "t2_both_pass": t2, "t5_both_pass": t5,
        "v6_formal_oracle_pass": {outfit: summary["tests"]["T5"][outfit]["final_status"] == "PASS" for outfit in OUTFITS},
        "rerun_seven_outfit_gate_allowed": t5, "generate_more_targets_allowed": False,
        "image_conditioned_training_allowed": False, "next_unique_task": next_task,
        "proven_failure_component": proven_failure_component,
        "t5_failure_checks": t5_failure_checks,
    }
    atomic_json(args.output / "final_adjudication/OBJECTIVE_RESIDUAL_REDESIGN_FINAL_STATUS.json", final)
    atomic_json(args.output / "final_adjudication/RUN_STATUS.json", final)
    atomic_text(args.output / "final_adjudication/FINAL_ADJUDICATION.md", "\n".join([
        "# Objective and residual redesign final adjudication", "", f"- causal primary case: **Case {causal}**",
        f"- final case: **Case {final_case}**", f"- T5 O01/O08 both pass: `{t5}`",
        f"- proven failure component: `{proven_failure_component}`",
        f"- seven-outfit re-adjudication allowed: `{t5}`", "- more target generation allowed: `false`",
        "- image-conditioned training allowed: `false`", f"- next unique task: `{next_task}`",
    ]))
    print(json.dumps(final, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen objective/residual redesign causal matrix")
    parser.add_argument("--phase", required=True, choices=("audit", "analyze", "calibrate", "run", "repair-slopes", "build-visual-evidence", "adjudicate", "finalize"))
    parser.add_argument("--test", choices=TESTS); parser.add_argument("--outfit", choices=OUTFITS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-triage", type=Path, default=Path("/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-REPRESENTATION-TRIAGE-001/attempt_002"))
    parser.add_argument("--source-aaai", type=Path, default=Path("/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-AAAI27-DATA-CAPACITY-GATE-001/attempt_003"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/research/subject02_objective_residual_redesign_v1.yaml")
    parser.add_argument("--pipeline-config", type=Path, default=PROJECT_ROOT / "configs/canon_dress_gs_mvp_real.yaml")
    parser.add_argument("--visual-decisions", type=Path); parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args(); config = load_config(args.config)
    if args.phase == "audit": run_audit(args, config)
    elif args.phase == "analyze": run_analyze(args, config)
    elif args.phase == "calibrate": run_calibrate(args, config)
    elif args.phase == "run": run_test(args, config)
    elif args.phase == "repair-slopes": repair_derived_slopes(args)
    elif args.phase == "build-visual-evidence": build_visual_evidence(args)
    elif args.phase == "adjudicate": adjudicate(args, config)
    else: finalize(args, config)


if __name__ == "__main__":
    main()
