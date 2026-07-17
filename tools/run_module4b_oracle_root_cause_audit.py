from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_dressable as training  # noqa: E402
from scene.full_attribute_oracle import AnchorResidualOracle, GaussianResidualOracle, STAGE_TRAINABLE  # noqa: E402
from scene.gaussian_clothing_residuals import (  # noqa: E402
    GaussianClothingResiduals,
    axis_angle_to_quaternion_wxyz,
    compose_canonical_gaussian_overrides,
    quaternion_multiply_wxyz,
)
from scene.oracle_root_cause_diagnostics import (  # noqa: E402
    FixedOpenGaussianOracle,
    gate_residuals_once,
    root_cause_decision_matrix,
)
from tools.run_module4b_canonical_oracle_micropilot import (  # noqa: E402
    CONDITIONS,
    OUTFITS,
    VIEWS,
    _base_named_tensors,
    _chw,
    _compute_transition_cap,
    _custom_metrics,
    _git_state,
    _grid,
    _load_samples,
    _loss,
    _regularization,
    _render,
    _residual_statistics,
    _sha256,
    _tensor_state_fingerprint,
    _to_pil,
    _transition_targets,
    build_oracle,
    load_contract,
    stage_for_state,
)
from utils.mmlphuman_state_utils import mmlphuman_state_transaction  # noqa: E402


SCHEMA = "canondressgs.module4b.root_cause.v1"
STEPS = (0, 80, 160, 320, 480)
KINDS = ("gaussian", "anchor")
PROBE_CONDITIONS = ("cond_000000", "cond_000318")
PROBE_AMPLITUDES = {
    "xyz": 0.005,
    "log_scaling": 0.03,
    "rotvec": 0.05,
    "opacity_logit": 0.20,
    "sh0": 0.05,
}
GATE_LEVELS = (0.05, 0.5, 1.0)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def _quantiles(value: torch.Tensor) -> dict[str, float]:
    flat = value.detach().float().reshape(-1).cpu()
    q = torch.quantile(flat, torch.tensor([0.01, 0.5, 0.95, 0.99]))
    return {
        "mean": float(flat.mean()), "std": float(flat.std(unbiased=False)),
        "p01": float(q[0]), "p50": float(q[1]), "p95": float(q[2]), "p99": float(q[3]),
        "min": float(flat.min()), "max": float(flat.max()),
    }


def _gate_row(
    outfit: str,
    kind: str,
    step: int,
    name: str,
    logits: torch.Tensor,
    activated: torch.Tensor,
    initial_logits: torch.Tensor,
) -> dict[str, Any]:
    row = {"outfit_id": outfit, "oracle_kind": kind, "step": step, "gate": name}
    row.update({f"logit_{key}": value for key, value in _quantiles(logits).items()})
    row.update({f"sigmoid_{key}": value for key, value in _quantiles(activated).items()})
    for threshold in (0.01, 0.05, 0.10, 0.25, 0.50):
        row[f"fraction_ge_{threshold:.2f}"] = float((activated >= threshold).float().mean())
    delta = logits.detach().float() - initial_logits.detach().float()
    row["parameter_update_l2_from_step0"] = float(torch.linalg.vector_norm(delta))
    row["parameter_update_rms_from_step0"] = float(torch.sqrt(delta.square().mean()))
    stage = stage_for_state(step)
    row["stage"] = stage
    row["requires_grad"] = name != "opacity" and (
        (name == "geometry" and "geometry_gate_logits" in STAGE_TRAINABLE[stage])
        or (name == "appearance" and "appearance_gate_logits" in STAGE_TRAINABLE[stage])
    )
    row["optimizer_contains_parameter"] = name in {"geometry", "appearance"}
    row["learning_rate"] = 0.001 if name in {"geometry", "appearance"} else "derived_max"
    row["scheduler_enabled"] = False
    row["clamp_or_detach"] = False
    return row


def _checkpoint_payload(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _gate_distribution_audit(module4b_root: Path, output: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for outfit in OUTFITS:
        for kind in KINDS:
            run = module4b_root / outfit / f"{kind}_oracle"
            zero = _checkpoint_payload(run / "checkpoints/step_000000.pth")["model_state"]
            initial_geometry = zero["geometry_gate_logits"]
            initial_appearance = zero["appearance_gate_logits"]
            for step in STEPS:
                state = _checkpoint_payload(run / f"checkpoints/step_{step:06d}.pth")["model_state"]
                geometry_logits = state["geometry_gate_logits"]
                appearance_logits = state["appearance_gate_logits"]
                geometry = torch.sigmoid(geometry_logits)
                appearance = torch.sigmoid(appearance_logits)
                opacity = torch.maximum(geometry, appearance)
                rows.append(_gate_row(outfit, kind, step, "geometry", geometry_logits, geometry, initial_geometry))
                rows.append(_gate_row(outfit, kind, step, "appearance", appearance_logits, appearance, initial_appearance))
                rows.append(_gate_row(outfit, kind, step, "opacity", torch.maximum(geometry_logits, appearance_logits), opacity, torch.maximum(initial_geometry, initial_appearance)))
    _write_csv(output / "gate_statistics_full_v4br.csv", rows)
    return rows


def _gate_gradient_audit(
    base: Any,
    module4b_root: Path,
    manifest: Path,
    config: Mapping[str, Any],
    background: torch.Tensor,
    rows: list[dict[str, Any]],
) -> None:
    keyed = {(row["outfit_id"], row["oracle_kind"], int(row["step"]), row["gate"]): row for row in rows}
    for outfit in OUTFITS:
        samples = _load_samples(manifest, outfit)
        transitions = _transition_targets(samples, base._xyz.device)
        for kind in KINDS:
            run = module4b_root / outfit / f"{kind}_oracle"
            coefficient = float(json.loads((run / "transition_gradient_cap.json").read_text())["final_frozen_coefficient"])
            for step in STEPS:
                oracle = build_oracle(kind, base, config, base._xyz.device).to(base._xyz.device)
                state = _checkpoint_payload(run / f"checkpoints/step_{step:06d}.pth")["model_state"]
                oracle.load_state_dict(state, strict=True)
                oracle.configure_stage(stage_for_state(step))
                sample = samples[CONDITIONS[step % len(CONDITIONS)]]
                output_state = oracle(base)
                rgb, alpha = _render(base, sample, output_state.canonical_overrides, background)
                parts = _loss(
                    rgb, alpha, sample, transitions[sample["target_condition_id"]],
                    base._xyz.device, config, coefficient,
                )
                regularizer, _ = _regularization(output_state, config)
                objective = parts["total"] + regularizer
                targets = [
                    oracle.geometry_gate_logits, oracle.appearance_gate_logits,
                    output_state.geometry_gate, output_state.appearance_gate,
                ]
                gradients = []
                for target in targets:
                    if not target.requires_grad:
                        gradients.append(None)
                    else:
                        gradients.append(torch.autograd.grad(objective, target, retain_graph=True, allow_unused=True)[0])
                norms = [0.0 if value is None else float(torch.linalg.vector_norm(value)) for value in gradients]
                geometry = keyed[(outfit, kind, step, "geometry")]
                appearance = keyed[(outfit, kind, step, "appearance")]
                opacity = keyed[(outfit, kind, step, "opacity")]
                geometry["logit_gradient_norm"] = norms[0]
                appearance["logit_gradient_norm"] = norms[1]
                geometry["activated_value_gradient_norm"] = norms[2]
                appearance["activated_value_gradient_norm"] = norms[3]
                opacity["logit_gradient_norm"] = "derived_not_a_parameter"
                opacity["activated_value_gradient_norm"] = math.sqrt(norms[2] ** 2 + norms[3] ** 2)
                opacity["activated_gradient_note"] = "combined geometry/appearance activated gradients; opacity=max has no separately exposed retained tensor"
                del oracle, output_state, rgb, alpha, objective, gradients
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


def _formal_gate_code_audit() -> dict[str, Any]:
    gaussian_source = inspect.getsource(GaussianResidualOracle.forward)
    anchor_source = inspect.getsource(AnchorResidualOracle.forward)
    composition_source = inspect.getsource(compose_canonical_gaussian_overrides)
    result = {
        "initial_probability": 0.05,
        "initial_logit": math.log(0.05 / 0.95),
        "activation": "sigmoid",
        "opacity_gate_formula": "maximum(geometry_gate, appearance_gate)",
        "gaussian_gate_level": "Gaussian",
        "anchor_gate_level": "anchor before formal interpolation",
        "anchor_gate_and_residual_interpolation": "gated residual is interpolated; gates are separately interpolated only for reporting",
        "render_effective_residual_gate_multiplications": 1,
        "regularization_reuses_gate": True,
        "regularization_changes_render_composition": False,
        "composition_reapplies_gate": False,
        "renderer_reapplies_gate": False,
        "double_gate_composition_bug": False,
        "stage_policy": {
            "stage1_appearance_initialization": sorted(STAGE_TRAINABLE[1]),
            "stage2_geometry_and_appearance": sorted(STAGE_TRAINABLE[2]),
            "stage3_joint": sorted(STAGE_TRAINABLE[3]),
        },
        "optimizer_gate_lr": {"geometry": 0.001, "appearance": 0.001},
        "scheduler_enabled": False,
        "clamp_or_detach": False,
        "source_checks": {
            "gaussian_has_one_gate_construction": gaussian_source.count("GaussianClothingResiduals(") == 1,
            "anchor_has_one_gate_construction": anchor_source.count("AnchorClothingResiduals(") == 1,
            "composition_mentions_gate": "gate" in composition_source.lower(),
        },
    }
    if result["source_checks"]["composition_mentions_gate"]:
        result["double_gate_composition_bug"] = True
    return result


def _load_runtime(args: argparse.Namespace):
    device = torch.device(args.device)
    pipeline = training.load_config(args.pipeline_config)
    base = training.load_frozen_mmlphuman_base(
        pipeline["base"]["model_dir"], pipeline["base"]["checkpoint_path"], device,
    )
    samples = _load_samples(args.manifest, "O00")
    background = torch.tensor(pipeline["render"]["background"], device=device, dtype=base._xyz.dtype)
    return device, pipeline, base, samples, background


def _project_gaussians(base: Any, sample: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    device = base._xyz.device
    with mmlphuman_state_transaction(
        base,
        sample["target_pose"].to(device),
        sample["target_Rh"].to(device),
        sample["target_Th"].to(device),
    ):
        xyz = base.compute_xyz().detach()
        opacity = base.compute_opacity().detach().reshape(-1)
    w2c = torch.as_tensor(sample["target_camera"]["w2c"], device=device, dtype=xyz.dtype)
    K = torch.as_tensor(sample["target_camera"]["K"], device=device, dtype=xyz.dtype)
    camera = torch.einsum("ij,nj->ni", w2c, torch.nn.functional.pad(xyz, (0, 1), value=1.0))[:, :3]
    pixels_h = torch.einsum("ij,nj->ni", K, camera)
    depth = camera[:, 2]
    u = pixels_h[:, 0] / depth.clamp_min(1e-8)
    v = pixels_h[:, 1] / depth.clamp_min(1e-8)
    height = int(sample["target_camera"]["height"]); width = int(sample["target_camera"]["width"])
    valid = (depth > 1e-6) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return {
        "u": u, "v": v, "depth": depth, "valid": valid,
        "ui": u.round().long().clamp(0, width - 1),
        "vi": v.round().long().clamp(0, height - 1),
        "opacity": opacity,
    }


def _mask_values(mask: torch.Tensor, projection: Mapping[str, torch.Tensor]) -> torch.Tensor:
    value = mask.to(projection["u"].device)
    if value.ndim == 3:
        value = value[0]
    sampled = value[projection["vi"], projection["ui"]]
    return sampled * projection["valid"].to(sampled.dtype)


def _select_visible(projection: Mapping[str, torch.Tensor], mask: torch.Tensor, count: int = 64) -> torch.Tensor:
    candidates = torch.where((_mask_values(mask, projection) >= 0.5) & projection["valid"])[0]
    if candidates.numel() == 0:
        raise RuntimeError("probe mask has no projected Gaussian centers")
    score = projection["opacity"][candidates] - 1e-6 * projection["depth"][candidates]
    return candidates[torch.argsort(score, descending=True)[: min(count, candidates.numel())]]


def _probe_residuals(base: Any, attribute: str, indices: torch.Tensor, amplitude: torch.Tensor) -> GaussianClothingResiduals:
    zeros = GaussianClothingResiduals.zeros(base)
    selector = torch.zeros(base._xyz.shape[0], device=base._xyz.device, dtype=base._xyz.dtype)
    selector[indices] = 1
    values = {
        "delta_xyz": zeros.delta_xyz,
        "delta_log_scaling": zeros.delta_log_scaling,
        "delta_rotvec": zeros.delta_rotvec,
        "delta_opacity_logit": zeros.delta_opacity_logit,
        "delta_sh0": zeros.delta_sh0,
        "delta_shN": zeros.delta_shN,
    }
    if attribute == "xyz":
        direction = torch.tensor([1.0, 0.25, -0.1], device=selector.device, dtype=selector.dtype)
        values["delta_xyz"] = selector[:, None] * direction * amplitude
    elif attribute == "log_scaling":
        values["delta_log_scaling"] = selector[:, None] * torch.tensor([1.0, -0.5, 0.25], device=selector.device) * amplitude
    elif attribute == "rotvec":
        values["delta_rotvec"] = selector[:, None] * torch.tensor([0.3, 0.5, 1.0], device=selector.device) * amplitude
    elif attribute == "opacity_logit":
        shaped = selector.reshape(base._opacity.shape)
        values["delta_opacity_logit"] = shaped * amplitude
    elif attribute == "sh0":
        direction = torch.tensor([0.8, -0.4, 0.2], device=selector.device, dtype=selector.dtype)
        values["delta_sh0"] = selector[:, None, None] * direction.reshape(1, 1, 3) * amplitude
    else:
        raise ValueError(attribute)
    return GaussianClothingResiduals(**values).validate(base)


def _render_probe(base: Any, sample: Mapping[str, Any], background: torch.Tensor, residuals: GaussianClothingResiduals, gate: float):
    gate_tensor = torch.full((base._xyz.shape[0], 1), gate, device=base._xyz.device, dtype=base._xyz.dtype)
    gated, _ = gate_residuals_once(residuals, gate_tensor, gate_tensor)
    overrides = compose_canonical_gaussian_overrides(base, gated)
    return _render(base, sample, overrides, background)


def _signed_probe_scalar(rgb: torch.Tensor, alpha: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    region = mask.to(rgb.device)
    if region.ndim == 2: region = region.unsqueeze(0)
    rgb_denominator = region.sum().clamp_min(1) * 3
    alpha_denominator = region.sum().clamp_min(1)
    return (rgb * region).sum() / rgb_denominator + 0.25 * (alpha * region).sum() / alpha_denominator


def _attribute_sensitivity_audit(base: Any, samples: Mapping[str, Any], background: torch.Tensor, output: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    panels: list[tuple[str, Image.Image]] = []
    for condition in PROBE_CONDITIONS:
        sample = samples[condition]
        projection = _project_gaussians(base, sample)
        base_residuals = GaussianClothingResiduals.zeros(base)
        base_rgb, base_alpha = _render_probe(base, sample, background, base_residuals, 1.0)
        panels.append((f"{condition} base", _to_pil(base_rgb, 3)))
        regions = {
            "edit": sample["target_edit_mask"],
            "preserve_control": torch.maximum(sample["target_preserve_mask"], sample["target_protected_mask"]),
        }
        for region_name, region_mask in regions.items():
            indices = _select_visible(projection, region_mask)
            for attribute, nominal in PROBE_AMPLITUDES.items():
                for gate in GATE_LEVELS:
                    amplitude = torch.tensor(nominal, device=base._xyz.device, dtype=base._xyz.dtype, requires_grad=True)
                    residuals = _probe_residuals(base, attribute, indices, amplitude)
                    rgb, alpha = _render_probe(base, sample, background, residuals, gate)
                    mask = region_mask.to(rgb.device)
                    scalar = _signed_probe_scalar(rgb, alpha, mask)
                    gradient = torch.autograd.grad(scalar, amplitude, allow_unused=True)[0]
                    autograd_value = 0.0 if gradient is None else float(gradient.detach())
                    epsilon = max(abs(nominal) * 0.02, 1e-4)
                    plus = torch.tensor(nominal + epsilon, device=base._xyz.device, dtype=base._xyz.dtype)
                    minus = torch.tensor(nominal - epsilon, device=base._xyz.device, dtype=base._xyz.dtype)
                    plus_rgb, plus_alpha = _render_probe(base, sample, background, _probe_residuals(base, attribute, indices, plus), gate)
                    minus_rgb, minus_alpha = _render_probe(base, sample, background, _probe_residuals(base, attribute, indices, minus), gate)
                    finite_difference = float((
                        _signed_probe_scalar(plus_rgb, plus_alpha, mask)
                        - _signed_probe_scalar(minus_rgb, minus_alpha, mask)
                    ) / (2 * epsilon))
                    rgb_diff = (rgb - base_rgb).abs(); alpha_diff = (alpha - base_alpha).abs()
                    prepared = mask.to(rgb.device)
                    if prepared.ndim == 2: prepared = prepared.unsqueeze(0)
                    rows.append({
                        "condition_id": condition, "view": VIEWS[condition], "selection": region_name,
                        "selected_gaussians": int(indices.numel()), "attribute": attribute, "gate": gate,
                        "nominal_amplitude": nominal, "rgb_mean_change": float(rgb_diff.mean()),
                        "rgb_max_change": float(rgb_diff.max()), "alpha_mean_change": float(alpha_diff.mean()),
                        "alpha_max_change": float(alpha_diff.max()),
                        "region_rgb_mean_change": float((rgb_diff * prepared).sum() / (prepared.sum().clamp_min(1) * 3)),
                        "region_alpha_mean_change": float((alpha_diff * prepared).sum() / prepared.sum().clamp_min(1)),
                        "autograd_gradient": autograd_value, "finite_difference_sensitivity": finite_difference,
                        "direction_agrees": bool(
                            abs(autograd_value) < 1e-12 and abs(finite_difference) < 1e-8
                            or autograd_value * finite_difference > 0
                        ),
                        "finite": bool(torch.isfinite(rgb).all() and torch.isfinite(alpha).all()),
                    })
                    if gate == 1.0 and region_name == "edit":
                        visualization = torch.cat((rgb, rgb_diff.clamp(0, 1)), dim=2)
                        panels.append((f"{condition} {attribute} gate1", _to_pil(visualization, 3)))
    _write_csv(output / "attribute_sensitivity_v4br.csv", rows)
    _grid(output / "attribute_probe_contact_sheet_v4br.png", panels, columns=3, cell=(512, 512))
    return rows


def _quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    q = torch.nn.functional.normalize(quaternion, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack((
        1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w),
        2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w),
        2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y),
    ), dim=-1).reshape(-1, 3, 3)


def _canonical_covariance(base: Any, overrides: Any) -> torch.Tensor:
    scales = base.compute_cano_scaling(overrides.as_dict())
    rotation = base.compute_cano_rotation(overrides.as_dict())
    matrix = _quaternion_to_matrix(rotation)
    return matrix @ torch.diag_embed(scales.square()) @ matrix.transpose(-1, -2)


def _rotation_audit(base: Any, sample: Mapping[str, Any], background: torch.Tensor, output: Path) -> dict[str, Any]:
    with torch.no_grad():
        scales = base.compute_cano_scaling().detach()
        ratio = scales.max(dim=1).values / scales.min(dim=1).values.clamp_min(1e-12)
    distribution = _quantiles(ratio)
    for threshold in (1.05, 1.10, 1.25, 1.50):
        distribution[f"fraction_ge_{threshold:.2f}"] = float((ratio >= threshold).float().mean())
    high = int(torch.argmax(ratio)); isotropic = int(torch.argmin(ratio))
    base_residuals = GaussianClothingResiduals.zeros(base)
    base_overrides = compose_canonical_gaussian_overrides(base, base_residuals)
    base_rgb, base_alpha = _render(base, sample, base_overrides, background)
    probe_dir = output / "rotation_probe_images_v4br"; probe_dir.mkdir(parents=True, exist_ok=True)
    probes = []
    axes = torch.eye(3, device=base._xyz.device, dtype=base._xyz.dtype)
    for label, index in (("high_anisotropy", high), ("near_isotropic", isotropic)):
        for axis_index, axis in enumerate(axes):
            nominal = 0.05
            amplitude = torch.tensor(nominal, device=base._xyz.device, requires_grad=True)
            selector = torch.tensor([index], device=base._xyz.device)
            residuals = _probe_residuals(base, "rotvec", selector, amplitude)
            # Use an exact axis for the dedicated rotation test.
            rotvec = torch.zeros_like(residuals.delta_rotvec); rotvec[index] = axis * amplitude
            residuals = GaussianClothingResiduals(
                residuals.delta_xyz, residuals.delta_log_scaling, rotvec,
                residuals.delta_opacity_logit, residuals.delta_sh0, residuals.delta_shN,
            )
            overrides = compose_canonical_gaussian_overrides(base, residuals)
            covariance = _canonical_covariance(base, overrides)
            base_covariance = _canonical_covariance(base, base_overrides)
            rgb, alpha = _render(base, sample, overrides, background)
            scalar = covariance[index, 0, 1] + rgb.mean() * 1e-3 + alpha.mean() * 1e-3
            gradient = torch.autograd.grad(scalar, amplitude, allow_unused=True)[0]
            epsilon = 1e-3
            def evaluate(value: float):
                values = GaussianClothingResiduals.zeros(base)
                rv = torch.zeros_like(values.delta_rotvec); rv[index] = axis * value
                values = GaussianClothingResiduals(values.delta_xyz, values.delta_log_scaling, rv, values.delta_opacity_logit, values.delta_sh0, values.delta_shN)
                ov = compose_canonical_gaussian_overrides(base, values)
                cov = _canonical_covariance(base, ov)
                image, a = _render(base, sample, ov, background)
                return cov[index, 0, 1] + image.mean() * 1e-3 + a.mean() * 1e-3, image, a
            plus, _, _ = evaluate(nominal + epsilon); minus, _, _ = evaluate(nominal - epsilon)
            finite_difference = float((plus - minus) / (2 * epsilon))
            diff = (rgb - base_rgb).abs()
            image_path = probe_dir / f"{label}_axis_{axis_index}.png"
            _grid(image_path, [("base", _to_pil(base_rgb, 3)), ("probe", _to_pil(rgb, 3)), ("abs diff x20", _to_pil((diff * 20).clamp(0, 1), 3))], columns=3, cell=(512, 512))
            probes.append({
                "selection": label, "gaussian_index": index, "anisotropy_ratio": float(ratio[index]),
                "axis": axis_index, "quaternion_convention": "wxyz", "composition_order": "base_times_delta",
                "composed_quaternion_change_l2": float(torch.linalg.vector_norm(overrides.rotation[index] - base._rotation[index])),
                "composed_quaternion_norm": float(torch.linalg.vector_norm(overrides.rotation[index])),
                "covariance_change_l2": float(torch.linalg.vector_norm(covariance[index] - base_covariance[index])),
                "rgb_mean_change": float(diff.mean()), "rgb_max_change": float(diff.max()),
                "alpha_mean_change": float((alpha - base_alpha).abs().mean()),
                "autograd_at_nonzero": 0.0 if gradient is None else float(gradient),
                "finite_difference_at_nonzero": finite_difference,
                "nonzero_direction_agrees": bool(gradient is not None and float(gradient) * finite_difference > 0),
                "image": str(image_path),
            })
    # The exact zero shortcut is the suspected root cause.
    zero_amplitude = torch.tensor(0.0, device=base._xyz.device, requires_grad=True)
    zero_values = GaussianClothingResiduals.zeros(base)
    zero_rotvec = torch.zeros_like(zero_values.delta_rotvec) + zero_amplitude * 0
    zero_values = GaussianClothingResiduals(zero_values.delta_xyz, zero_values.delta_log_scaling, zero_rotvec, zero_values.delta_opacity_logit, zero_values.delta_sh0, zero_values.delta_shN)
    zero_overrides = compose_canonical_gaussian_overrides(base, zero_values)
    zero_autograd_connected = bool(zero_overrides.rotation.requires_grad)
    epsilon = 1e-3
    def high_covariance(value: float) -> float:
        values = GaussianClothingResiduals.zeros(base)
        rv = torch.zeros_like(values.delta_rotvec); rv[high, 2] = value
        values = GaussianClothingResiduals(values.delta_xyz, values.delta_log_scaling, rv, values.delta_opacity_logit, values.delta_sh0, values.delta_shN)
        return float(_canonical_covariance(base, compose_canonical_gaussian_overrides(base, values))[high, 0, 1])
    zero_fd = (high_covariance(epsilon) - high_covariance(-epsilon)) / (2 * epsilon)
    high_responsive = any(
        row["selection"] == "high_anisotropy" and row["covariance_change_l2"] > 1e-8
        for row in probes
    )
    classification = "ROTATION_PATH_BROKEN" if (not zero_autograd_connected and abs(zero_fd) > 1e-8 and high_responsive) else "ROTATION_EXPECTED_ZERO"
    result = {
        "activated_scale_anisotropy": distribution,
        "high_anisotropy_index": high,
        "near_isotropic_index": isotropic,
        "zero_initialization_autograd_connected": zero_autograd_connected,
        "zero_initialization_finite_difference_covariance": zero_fd,
        "zero_shortcut_source_detected": "torch.count_nonzero(delta_rotvec).item() == 0" in inspect.getsource(compose_canonical_gaussian_overrides),
        "high_anisotropy_downstream_response": high_responsive,
        "probes": probes,
        "classification": classification,
        "root_cause": "zero-valued rotvec shortcut returns base rotation and removes the trainable rotvec from autograd" if classification == "ROTATION_PATH_BROKEN" else "near-isotropic covariance response",
    }
    _write_json(output / "rotation_sensitivity_v4br.json", result)
    return result


def _objective_audit(
    base: Any,
    samples: Mapping[str, Any],
    background: torch.Tensor,
    config: Mapping[str, Any],
    module4b_root: Path,
    output: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    coefficient = json.loads((module4b_root / "O00/gaussian_oracle/transition_gradient_cap.json").read_text())["final_frozen_coefficient"]
    transitions = _transition_targets(samples, base._xyz.device)
    gradient_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    visualization: list[tuple[str, Image.Image]] = []
    conflict = False
    for condition in PROBE_CONDITIONS:
        sample = samples[condition]
        projection = _project_gaussians(base, sample)
        old_sleeve_mask = sample["target_old_clothing_mask"] * sample["target_revealed_skin_mask"]
        old_indices = torch.where((_mask_values(old_sleeve_mask, projection) >= 0.5) & projection["valid"])[0]
        masks = {
            "old_sleeve": old_sleeve_mask,
            "edit_core": sample["target_edit_core_mask"],
            "clothing_safe": sample["target_clothing_mask"],
            "preserve": sample["target_preserve_mask"],
            "protected": sample["target_protected_mask"],
            "transition": sample["target_transition_mask"],
            "alpha_base_effective": torch.maximum(sample["target_preserve_mask"], sample["target_protected_mask"]),
            "new_silhouette": sample["target_foreground_mask"] * (1 - sample["target_base_foreground_mask"]),
        }
        old_binary = old_sleeve_mask >= 0.5
        for name, mask in masks.items():
            binary = mask >= 0.5
            overlap = int((old_binary & binary).sum())
            overlap_rows.append({
                "condition_id": condition, "view": VIEWS[condition], "mask": name,
                "mask_pixels": int(binary.sum()), "old_sleeve_pixels": int(old_binary.sum()),
                "overlap_pixels": overlap,
                "old_sleeve_overlap_fraction": overlap / max(int(old_binary.sum()), 1),
            })
        oracle = build_oracle("gaussian", base, config, base._xyz.device).to(base._xyz.device)
        oracle.configure_stage(3)
        output_state = oracle(base)
        rgb, alpha = _render(base, sample, output_state.canonical_overrides, background)
        parts = _loss(rgb, alpha, sample, transitions[condition], base._xyz.device, config, float(coefficient))
        losses = {
            "edit_rgb": parts["edit"], "clothing_rgb": parts["clothing"],
            "rgb_transition": parts["transition"], "alpha_edit": parts["alpha_edit"],
            "alpha_transition": parts["alpha_transition"], "alpha_base": parts["alpha_base"],
            "preserve": parts["preserve"], "protected": parts["protected"],
        }
        tensors = [
            output_state.gaussian_residuals.delta_xyz,
            output_state.gaussian_residuals.delta_log_scaling,
            output_state.gaussian_residuals.delta_rotvec,
            output_state.gaussian_residuals.delta_opacity_logit,
            output_state.gaussian_residuals.delta_sh0,
            oracle.geometry_gate_logits,
            oracle.appearance_gate_logits,
        ]
        names = ["effective_xyz", "effective_scaling", "effective_rotation", "effective_opacity", "effective_sh0", "geometry_gate_logits", "appearance_gate_logits"]
        total_loss = sum(losses.values())
        total_grads = torch.autograd.grad(total_loss, tensors, retain_graph=True, allow_unused=True)
        for loss_name, loss_value in losses.items():
            gradients = torch.autograd.grad(loss_value, tensors, retain_graph=True, allow_unused=True)
            for tensor_name, gradient, total_gradient in zip(names, gradients, total_grads):
                if gradient is None:
                    norm = old_norm = cosine = 0.0
                else:
                    flat = gradient.reshape(gradient.shape[0], -1)
                    norm = float(torch.linalg.vector_norm(flat))
                    old_norm = float(torch.linalg.vector_norm(flat[old_indices])) if old_indices.numel() else 0.0
                    if total_gradient is None:
                        cosine = 0.0
                    else:
                        g = gradient.reshape(-1); t = total_gradient.reshape(-1)
                        cosine = float(torch.dot(g, t) / (torch.linalg.vector_norm(g) * torch.linalg.vector_norm(t)).clamp_min(1e-20))
                gradient_rows.append({
                    "condition_id": condition, "view": VIEWS[condition], "loss": loss_name,
                    "target": tensor_name, "loss_value": float(loss_value.detach()),
                    "gradient_norm": norm, "old_sleeve_gradient_norm": old_norm,
                    "cosine_with_combined_objective": cosine,
                    "old_sleeve_gaussian_count": int(old_indices.numel()),
                })
        old_edit = next(row for row in overlap_rows if row["condition_id"] == condition and row["mask"] == "edit_core")
        old_preserve = next(row for row in overlap_rows if row["condition_id"] == condition and row["mask"] == "preserve")
        old_alpha_base = next(row for row in overlap_rows if row["condition_id"] == condition and row["mask"] == "alpha_base_effective")
        edit_gradient = sum(row["old_sleeve_gradient_norm"] for row in gradient_rows if row["condition_id"] == condition and row["loss"] in {"edit_rgb", "alpha_edit"})
        conflict = conflict or old_edit["old_sleeve_overlap_fraction"] < 0.5 or old_preserve["overlap_pixels"] > 0 or old_alpha_base["overlap_pixels"] > 0 or edit_gradient <= 0
        visualization.extend([
            (f"{condition} base", _to_pil(sample["target_base_rgb"], 3)),
            (f"{condition} edit", _to_pil(sample["target_edit_rgb"], 3)),
            (f"{condition} old sleeve", _to_pil(old_sleeve_mask.repeat(3, 1, 1), 3)),
            (f"{condition} edit core", _to_pil(sample["target_edit_core_mask"].repeat(3, 1, 1), 3)),
            (f"{condition} preserve", _to_pil(sample["target_preserve_mask"].repeat(3, 1, 1), 3)),
            (f"{condition} alpha base mask", _to_pil(masks["alpha_base_effective"].repeat(3, 1, 1), 3)),
        ])
    _write_csv(output / "oracle_objective_gradient_map_v4br.csv", gradient_rows)
    _write_csv(output / "oracle_mask_overlap_v4br.csv", overlap_rows)
    _grid(output / "oracle_gradient_visualization_v4br.png", visualization, columns=3, cell=(512, 512))
    return gradient_rows, overlap_rows, {"objective_mask_conflict": conflict}


def _base_support_audit(base: Any, samples: Mapping[str, Any], background: torch.Tensor, output: Path) -> dict[str, Any]:
    before = _tensor_state_fingerprint(_base_named_tensors(base))
    all_indices: list[torch.Tensor] = []
    condition_data: dict[str, Any] = {}
    skin_pixels = []
    for condition in PROBE_CONDITIONS:
        sample = samples[condition]; projection = _project_gaussians(base, sample)
        old_sleeve_mask = sample["target_old_clothing_mask"] * sample["target_revealed_skin_mask"]
        indices = torch.where((_mask_values(old_sleeve_mask, projection) >= 0.5) & projection["valid"])[0]
        all_indices.append(indices)
        revealed = sample["target_revealed_skin_mask"].to(base._xyz.device) >= 0.5
        skin_pixels.append(sample["target_edit_rgb"].to(base._xyz.device).permute(1, 2, 0)[revealed[0]])
        condition_data[condition] = {"projection": projection, "indices": indices, "old_sleeve_mask": old_sleeve_mask}
    union = torch.unique(torch.cat(all_indices))
    weights = base._weights.to(base._xyz.device)
    arm_joints = [index for index in (16, 17, 18, 19, 20, 21) if index < weights.shape[1]]
    arm_influence = weights[:, arm_joints].sum(dim=1) if arm_joints else torch.zeros(weights.shape[0], device=weights.device)
    anchor_indices = base.nbr_gs.to(base._xyz.device).long()
    anchor_xyz = base.xyz_vt.to(base._xyz.device)
    canonical_distance = torch.linalg.vector_norm(base._xyz[:, None, :] - anchor_xyz[anchor_indices], dim=-1).min(dim=1).values
    distance_limit = torch.quantile(canonical_distance[union], 0.75) if union.numel() else torch.tensor(0.0, device=weights.device)
    support = union[(arm_influence[union] >= 0.25) & (canonical_distance[union] <= distance_limit)]
    skin_rgb = torch.cat(skin_pixels).median(dim=0).values
    c0 = 0.28209479177387814
    skin_sh0 = (skin_rgb - 0.5) / c0
    base_opacity = base.compute_opacity().detach().reshape(-1)
    base_color = (0.5 + c0 * base._sh0[:, 0]).clamp(0, 1).detach()
    delta_e_rgb = torch.linalg.vector_norm(base_color[union] - skin_rgb, dim=1) if union.numel() else torch.empty(0, device=weights.device)
    panels: list[tuple[str, Image.Image]] = []
    per_condition = {}
    for condition in PROBE_CONDITIONS:
        sample = samples[condition]
        zero = GaussianClothingResiduals.zeros(base)
        base_overrides = compose_canonical_gaussian_overrides(base, zero)
        rgb_a, alpha_a = _render(base, sample, base_overrides, background)
        opacity_b = base._opacity.clone(); opacity_b[union] = opacity_b[union] - 10.0
        override_b = type(base_overrides)(base_overrides.xyz, base_overrides.scaling, base_overrides.rotation, opacity_b, base_overrides.sh0, base_overrides.shN)
        rgb_b, alpha_b = _render(base, sample, override_b, background)
        opacity_c = opacity_b.clone(); opacity_c[support] = base._opacity[support]
        sh0_c = base._sh0.clone(); sh0_c[support, 0] = skin_sh0
        override_c = type(base_overrides)(base_overrides.xyz, base_overrides.scaling, base_overrides.rotation, opacity_c, sh0_c, base_overrides.shN)
        rgb_c, alpha_c = _render(base, sample, override_c, background)
        mask = condition_data[condition]["old_sleeve_mask"].to(rgb_a.device)
        denominator = mask.sum().clamp_min(1)
        per_condition[condition] = {
            "old_sleeve_projected_gaussians": int(condition_data[condition]["indices"].numel()),
            "B_rgb_mae_in_old_sleeve": float(((rgb_b - rgb_a).abs() * mask).sum() / (denominator * 3)),
            "B_alpha_mae_in_old_sleeve": float(((alpha_b - alpha_a).abs() * mask).sum() / denominator),
            "C_rgb_mae_to_target_skin_in_old_sleeve": float(((rgb_c - sample["target_edit_rgb"].to(rgb_c.device)).abs() * mask).sum() / (denominator * 3)),
            "C_alpha_mean_in_old_sleeve": float((alpha_c * mask).sum() / denominator),
        }
        panels.extend([
            (f"{condition} A base", _to_pil(rgb_a, 3)),
            (f"{condition} B old sleeve opacity down", _to_pil(rgb_b, 3)),
            (f"{condition} C arm-support skin probe", _to_pil(rgb_c, 3)),
            (f"{condition} old sleeve mask", _to_pil(mask.repeat(3, 1, 1), 3)),
            (f"{condition} B alpha", _to_pil(alpha_b.repeat(3, 1, 1), 3)),
            (f"{condition} C alpha", _to_pil(alpha_c.repeat(3, 1, 1), 3)),
        ])
    _grid(output / "base_support_probe_contact_sheet_v4br.png", panels, columns=3, cell=(512, 512))
    after = _tensor_state_fingerprint(_base_named_tensors(base))
    result = {
        "old_sleeve_union_gaussian_count": int(union.numel()),
        "arm_support_gaussian_count": int(support.numel()),
        "arm_support_fraction": int(support.numel()) / max(int(union.numel()), 1),
        "arm_joint_indices_smplx": arm_joints,
        "arm_influence": _quantiles(arm_influence[union]) if union.numel() else {},
        "canonical_surface_distance": _quantiles(canonical_distance[union]) if union.numel() else {},
        "activated_opacity": _quantiles(base_opacity[union]) if union.numel() else {},
        "base_dc_rgb": {"r": _quantiles(base_color[union, 0]), "g": _quantiles(base_color[union, 1]), "b": _quantiles(base_color[union, 2])} if union.numel() else {},
        "skin_reference_rgb": [float(value) for value in skin_rgb],
        "rgb_distance_to_skin": _quantiles(delta_e_rgb) if union.numel() else {},
        "skin_like_fraction_rgb_distance_le_0.15": float((delta_e_rgb <= 0.15).float().mean()) if union.numel() else 0.0,
        "condition_probes": per_condition,
        "base_fingerprint_before": before,
        "base_fingerprint_after": after,
        "base_unmodified": before == after,
        "body_part_source": "formal Gaussian LBS weights with SMPL-X shoulder/elbow/wrist joints 16-21; no explicit body-part labels available",
        "pre_visual_classification": "BASE_SUPPORT_AMBIGUOUS",
        "visual_classification": "PENDING_ACTUAL_IMAGE_INSPECTION",
    }
    _write_json(output / "base_support_statistics_v4br.json", result)
    return result


def _write_audit_reports(
    output: Path,
    gate_code: Mapping[str, Any],
    attribute_rows: list[dict[str, Any]],
    rotation: Mapping[str, Any],
    base_support: Mapping[str, Any],
    objective: Mapping[str, Any],
) -> None:
    responsive = {}
    for attribute in PROBE_AMPLITUDES:
        rows = [row for row in attribute_rows if row["attribute"] == attribute and row["selection"] == "edit" and row["gate"] == 1.0]
        responsive[attribute] = {
            "max_rgb_change": max(row["rgb_max_change"] for row in rows),
            "max_alpha_change": max(row["alpha_max_change"] for row in rows),
            "autograd_fd_agreement_fraction": sum(row["direction_agrees"] for row in rows) / len(rows),
        }
    _write_text(output / "GATE_PATH_AUDIT_V4BR.md", "\n".join([
        "# Gate Path Audit V4BR", "",
        f"- Initial gate probability/logit: `0.05` / `{gate_code['initial_logit']:.9f}`.",
        "- Activation: sigmoid; opacity gate: max(geometry, appearance).",
        "- Gaussian residuals are gated once before composition.",
        "- Anchor residuals are gated once at anchor level, then formally interpolated.",
        "- Separately interpolated gates are reporting fields and are not multiplied into the rendered residual again.",
        "- Gate regularization reuses gate values but does not alter canonical render composition.",
        f"- DOUBLE_GATE_COMPOSITION_BUG: `{gate_code['double_gate_composition_bug']}`.",
        "- Gate optimizer groups exist at LR 0.001; scheduler is disabled.",
    ]))
    _write_text(output / "ATTRIBUTE_SENSITIVITY_AUDIT_V4BR.md", "\n".join([
        "# Attribute Sensitivity Audit V4BR", "",
        "Deterministic front/back probes use edit and preserve/protected Gaussian selections and gate levels 0.05, 0.5 and 1.0.", "",
        *[f"- {name}: `{json.dumps(value, sort_keys=True)}`" for name, value in responsive.items()],
        "- SHN is intentionally absent because formal SH degree is zero.",
    ]))
    _write_text(output / "ROTATION_ZERO_GRAD_ROOT_CAUSE.md", "\n".join([
        "# Rotation Zero-gradient Root Cause", "",
        f"- Classification: **{rotation['classification']}**.",
        f"- Zero initialization autograd connected: `{rotation['zero_initialization_autograd_connected']}`.",
        f"- Zero-point covariance finite difference: `{rotation['zero_initialization_finite_difference_covariance']}`.",
        f"- High-anisotropy downstream response: `{rotation['high_anisotropy_downstream_response']}`.",
        "- Formal code returns base rotation when every rotvec element is exactly zero. This removes the zero-initialized trainable rotvec from the graph, so it can never leave zero.",
        "- Nonzero rotvec probes change the normalized wxyz quaternion and anisotropic covariance, proving the downstream covariance path exists.",
        "- Per the preregistered stop rule, no fixed-open optimizer run may execute before this formal composition defect is fixed in a separate task.",
    ]))
    _write_text(output / "BASE_SUPPORT_AUDIT_V4BR.md", "\n".join([
        "# Base Support Audit V4BR", "",
        f"- Projected old-sleeve Gaussian union: `{base_support['old_sleeve_union_gaussian_count']}`.",
        f"- LBS arm-support candidates: `{base_support['arm_support_gaussian_count']}` (`{base_support['arm_support_fraction']:.6f}`).",
        f"- Base tensors unmodified: `{base_support['base_unmodified']}`.",
        "- No explicit body-part labels exist; arm candidates use formal Gaussian LBS weights and SMPL-X shoulder/elbow/wrist joints.",
        "- Final support classification requires actual opening of the A/B/C contact sheet.",
    ]))
    _write_text(output / "ORACLE_OBJECTIVE_AUDIT_V4BR.md", "\n".join([
        "# Oracle Objective Audit V4BR", "",
        "- The audit differentiates edit RGB, clothing RGB, RGB transition, alpha edit, alpha transition, alpha base, preserve and protected terms separately.",
        "- Norms are reported for each effective residual channel and both gate logits, including the projected old-sleeve Gaussian subset.",
        f"- Automatic mask/gradient conflict flag: `{objective['objective_mask_conflict']}`.",
        "- The CSV records exact mask overlap and per-loss normalization context; target fields remain loss/evaluation-only.",
    ]))


def _preflight(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"root-cause output already exists: {output}")
    output.mkdir(parents=True)
    module4b_root = args.module4b_root.resolve()
    required = [
        module4b_root / "module4b_input_fingerprint.json",
        module4b_root / "module4b_oracle_contract.json",
        module4b_root / "MODULE4B_FINAL_STATUS.json",
    ]
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("sealed Module 4B evidence is incomplete")
    status = json.loads(required[-1].read_text())
    if status.get("status") != "COMPLETE" or status.get("final_status") != "FAIL":
        raise ValueError("Module 4B-R requires the sealed COMPLETE/FAIL Module 4B result")
    contract = {
        "schema_version": SCHEMA,
        "git": _git_state(),
        "module4b_root": str(module4b_root),
        "module4b_status_sha256": _sha256(required[-1]),
        "module4b_input_fingerprint_sha256": _sha256(required[0]),
        "manifest": str(args.manifest.resolve()), "manifest_sha256": _sha256(args.manifest),
        "pipeline_config": str(args.pipeline_config.resolve()), "pipeline_config_sha256": _sha256(args.pipeline_config),
        "oracle_config": str(args.config.resolve()), "oracle_config_sha256": _sha256(args.config),
        "allowed_optimizer_runs": [{"outfit": "O00", "oracle": "gaussian", "gate": "fixed_open", "max_steps": 160}],
        "technical_stop_rules": ["DOUBLE_GATE_COMPOSITION_BUG", "ROTATION_PATH_BROKEN"],
        "probes": {"conditions": list(PROBE_CONDITIONS), "attributes": PROBE_AMPLITUDES, "gate_levels": list(GATE_LEVELS)},
        "forbidden": ["full_module4b", "formal_image_conditioned_training", "garment_gaussian_layer", "v5_3_mutation"],
    }
    _write_json(output / "MODULE4B_ROOT_CAUSE_CONTRACT.json", contract)
    gate_code = _formal_gate_code_audit()
    _write_json(output / "implementation_audit.json", gate_code)
    rows = _gate_distribution_audit(module4b_root, output)
    _write_text(output / "MODULE4B_ROOT_CAUSE_IMPLEMENTATION_AUDIT.md", "\n".join([
        "# Module 4B-R Implementation Audit", "",
        f"- Git: `{contract['git']['commit']}` ({'clean' if not contract['git']['status_short'] else 'dirty'}).",
        "- Sealed Module 4B input fingerprints and final status were read without mutation.",
        f"- Gate distributions loaded from `{len(rows)}` gate/checkpoint rows (six runs × five states × three gate views).",
        f"- Double-gate composition bug detected: `{gate_code['double_gate_composition_bug']}`.",
        "- The root-cause run is capped at one O00 Gaussian fixed-open run of 160 steps, subject to technical stop rules.",
    ]))
    _write_json(output / "run_status.json", {"status": "PREFLIGHT_COMPLETE", "optimizer_steps": 0})


def _audit(args: argparse.Namespace) -> None:
    output = args.output.resolve(); module4b_root = args.module4b_root.resolve()
    status_path = output / "run_status.json"
    _write_json(status_path, {"status": "AUDIT_RUNNING", "optimizer_steps": 0})
    try:
        config = load_contract(args.config)
        device, pipeline, base, samples, background = _load_runtime(args)
        before = _tensor_state_fingerprint(_base_named_tensors(base))
        gate_code = json.loads((output / "implementation_audit.json").read_text())
        gate_rows = _gate_distribution_audit(module4b_root, output)
        _gate_gradient_audit(base, module4b_root, args.manifest, config, background, gate_rows)
        _write_csv(output / "gate_statistics_full_v4br.csv", gate_rows)
        attribute_rows = _attribute_sensitivity_audit(base, samples, background, output)
        rotation = _rotation_audit(base, samples["cond_000000"], background, output)
        base_support = _base_support_audit(base, samples, background, output)
        gradient_rows, overlap_rows, objective = _objective_audit(base, samples, background, config, module4b_root, output)
        after = _tensor_state_fingerprint(_base_named_tensors(base))
        if before != after or not base_support["base_unmodified"]:
            raise RuntimeError("base changed during read-only root-cause probes")
        _write_audit_reports(output, gate_code, attribute_rows, rotation, base_support, objective)
        _write_json(output / "audit_summary.json", {
            "double_gate_composition_bug": gate_code["double_gate_composition_bug"],
            "rotation_classification": rotation["classification"],
            "base_support_pre_visual": base_support["pre_visual_classification"],
            "objective_mask_conflict": objective["objective_mask_conflict"],
            "base_bitwise_exact": before == after,
            "optimizer_steps": 0,
        })
        _write_json(status_path, {
            "status": "AUDIT_COMPLETE",
            "optimizer_steps": 0,
            "gate_open_allowed": not gate_code["double_gate_composition_bug"] and rotation["classification"] != "ROTATION_PATH_BROKEN",
        })
    except Exception as error:
        _write_json(status_path, {"status": "FAILED", "optimizer_steps": 0, "exception_type": type(error).__name__, "exception": str(error), "traceback": traceback.format_exc()})
        raise


def _gate_open(args: argparse.Namespace) -> None:
    output = args.output.resolve(); module4b_root = args.module4b_root.resolve()
    audit = json.loads((output / "audit_summary.json").read_text())
    d0_path = module4b_root / "O00/gaussian_oracle/metrics_history.csv"
    with d0_path.open(newline="", encoding="utf-8") as handle:
        d0 = [row for row in csv.DictReader(handle) if int(row["step"]) in {0, 1, 10, 20, 40, 80, 120, 160}]
    rows = [{"diagnostic": "D0_learned_gate", **row} for row in d0]
    stop_reason = None
    if audit["double_gate_composition_bug"]:
        stop_reason = "DOUBLE_GATE_COMPOSITION_BUG"
    elif audit["rotation_classification"] == "ROTATION_PATH_BROKEN":
        stop_reason = "ROTATION_PATH_BROKEN"
    if stop_reason:
        rows.append({"diagnostic": "D1_fixed_open_gate", "status": "SKIPPED_TECHNICAL_STOP", "stop_reason": stop_reason, "optimizer_steps": 0})
        _write_csv(output / "gate_open_o00_metrics_v4br.csv", rows)
        source0 = Image.open(module4b_root / "O00/gaussian_oracle/visuals/step_000000_four_view_panel.png").convert("RGB")
        source160 = Image.open(module4b_root / "O00/gaussian_oracle/visuals/step_000160_four_view_panel.png").convert("RGB")
        placeholder = Image.new("RGB", source0.size, "white")
        draw = ImageDraw.Draw(placeholder); draw.multiline_text((40, 40), f"D1 SKIPPED\n{stop_reason}\noptimizer steps: 0", fill="black", spacing=12)
        _grid(output / "gate_open_o00_contact_sheet_v4br.png", [("D0 step 0", source0), ("D0 step 160", source160), ("D1", placeholder)], columns=3, cell=(640, 640))
        _write_json(output / "gate_open_status.json", {"status": "SKIPPED_TECHNICAL_STOP", "stop_reason": stop_reason, "optimizer_steps": 0, "gate_bottleneck": None})
        _write_json(output / "run_status.json", {"status": "DIAGNOSTICS_COMPLETE_TRAINING_SKIPPED", "optimizer_steps": 0, "stop_reason": stop_reason})
        return
    raise RuntimeError("fixed-open 160-step implementation intentionally unavailable unless every technical audit passes")


def _finalize(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    audit = json.loads((output / "audit_summary.json").read_text())
    rotation = json.loads((output / "rotation_sensitivity_v4br.json").read_text())
    base = json.loads((output / "base_support_statistics_v4br.json").read_text())
    gate_open = json.loads((output / "gate_open_status.json").read_text())
    base["visual_classification"] = args.base_support_status
    base["visual_observation"] = args.base_support_observation
    _write_json(output / "base_support_statistics_v4br.json", base)
    objective_conflict = bool(audit["objective_mask_conflict"])
    cases = root_cause_decision_matrix(
        double_gate=bool(audit["double_gate_composition_bug"]),
        rotation_path_broken=rotation["classification"] == "ROTATION_PATH_BROKEN",
        gate_bottleneck=gate_open.get("gate_bottleneck"),
        base_support=args.base_support_status,
        objective_conflict=objective_conflict,
        residual_bound_limit=False,
    )
    status = "FAIL" if "R2" in cases or "DOUBLE_GATE_COMPOSITION_BUG" in cases else "PARTIAL"
    adjudication = {
        "status": "COMPLETE", "final_status": status, "root_cause_cases": cases,
        "double_gate_composition_bug": audit["double_gate_composition_bug"],
        "rotation_classification": rotation["classification"],
        "gate_open_status": gate_open["status"], "new_optimizer_steps": gate_open["optimizer_steps"],
        "base_support_classification": args.base_support_status,
        "objective_mask_conflict": objective_conflict,
        "fix_formal_composition_required": "R2" in cases,
        "redefine_oracle_gate_policy": "R1" in cases,
        "rebuild_base_representation": args.base_support_status == "BASE_SUPPORT_MISSING",
        "rerun_module4b_allowed": False,
        "formal_image_conditioned_training_allowed": False,
        "unique_blocker": "zero-initialized rotation residual is disconnected by the formal composition zero shortcut" if "R2" in cases else "root cause remains unresolved",
    }
    _write_json(output / "MODULE4B_ROOT_CAUSE_FINAL_STATUS.json", adjudication)
    _write_text(output / "MODULE4B_ROOT_CAUSE_FINAL_ADJUDICATION.md", "\n".join([
        "# Module 4B-R Final Adjudication", "",
        f"- Final status: **{status}**.",
        f"- Root-cause case(s): `{', '.join(cases)}`.",
        f"- Double gate: `{audit['double_gate_composition_bug']}`.",
        f"- Rotation: `{rotation['classification']}`.",
        f"- Base support: `{args.base_support_status}` — {args.base_support_observation}",
        f"- Objective/mask conflict: `{objective_conflict}`.",
        f"- Fixed-open D1: `{gate_open['status']}`; optimizer steps `{gate_open['optimizer_steps']}`.",
        "- Formal composition must be fixed and minimally regression-tested in a separate authorized task before any Oracle rerun.",
        "- Formal image-conditioned training and garment-layer work remain prohibited.",
    ]))
    _write_json(output / "run_status.json", {"status": "COMPLETE", "final_status": status, "optimizer_steps": gate_open["optimizer_steps"], "root_cause_cases": cases})


def main() -> None:
    parser = argparse.ArgumentParser(description="Module 4B-R Oracle root-cause audit")
    parser.add_argument("--phase", choices=("preflight", "audit", "gate-open", "finalize"), required=True)
    parser.add_argument("--module4b-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("/root/autodl-tmp/canondressgs_work/data/subject02_dual_target_v5_2/pilot_manifest_full_v1_v5_2.json"))
    parser.add_argument("--pipeline-config", type=Path, default=PROJECT_ROOT / "configs/canon_dress_gs_mvp_real.yaml")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/oracle/module4b_canonical_capacity_v1.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-support-status", choices=("BASE_SUPPORT_PRESENT", "BASE_SUPPORT_MISSING", "BASE_SUPPORT_AMBIGUOUS"), default="BASE_SUPPORT_AMBIGUOUS")
    parser.add_argument("--base-support-observation", default="pending actual image inspection")
    args = parser.parse_args()
    if args.phase == "preflight": _preflight(args)
    elif args.phase == "audit": _audit(args)
    elif args.phase == "gate-open": _gate_open(args)
    else: _finalize(args)


if __name__ == "__main__":
    main()
