from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene.gaussian_clothing_residuals import CHANNELS  # noqa: E402
from tools.check_differentiable_render_path import (  # noqa: E402
    BASE_PARAMETER_NAMES,
    _gradient_snapshot,
)
from tools.check_real_image_conditioned_one_batch import save_render_tensor  # noqa: E402
from tools.run_dual_target_smoke20_v5_1 import (  # noqa: E402
    _forward,
    _state_clone,
    _state_equal,
    _tensor_image,
)
from tools.run_fixed_episode_optimization_v5_2 import (  # noqa: E402
    EXPECTED_OUTFIT,
    EXPECTED_REFERENCES,
    EXPECTED_TARGET,
    _build_runtime,
    _combine_gradients,
    _component_values,
    _cosine,
    _fixed_episode_contract,
    _gradient_norm,
    _gradient_tuple,
    _module_parameter_indices,
    _objective,
    _regularizers,
    _restore_initial_state,
    _sha256,
    _total_gradient_norm,
    _total_parameter_norm,
    _write_csv,
    _write_json,
)
from utils.rendering_loss_utils import (  # noqa: E402
    boundary_aware_transition_alpha_target,
    compute_static_transition_gradient_cap,
    region_aware_dual_target_loss,
)


EXPECTED_INITIAL_STATE_SHA256 = "e9d1d3f8b544fee8e2fda8a19002b4b4e7774357777c5ed1df4f0d37e4005a18"
LOG_STEPS = frozenset({0, 1, 5, 10, 20, 40, 60, 80, 100, 120})
RENDER_STEPS = frozenset({0, 20, 40, 80, 120})
BASELINE_SCALES = {name: 1.0 for name in ("edit_rgb", "identity", "alpha", "regularization")}


def _loss_v5_3(
    rgb: torch.Tensor,
    alpha: torch.Tensor,
    sample: dict[str, Any],
    transition_target: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    return region_aware_dual_target_loss(
        rgb,
        alpha,
        sample["target_edit_rgb"].to(device),
        sample["target_base_rgb"].to(device),
        sample["target_edit_core_mask"].to(device),
        sample["target_preserve_mask"].to(device),
        sample["target_protected_mask"].to(device),
        sample["target_transition_mask"].to(device),
        sample["target_clothing_mask"].to(device),
        sample["target_foreground_mask"].to(device),
        sample["target_base_foreground_mask"].to(device),
        alpha_weight=args.alpha_weight,
        alpha_edit_weight=args.alpha_edit_weight,
        alpha_transition_weight=args.alpha_transition_weight,
        alpha_base_weight=args.alpha_base_weight,
        transition_alpha_target=transition_target,
    )


def _legacy_transition_parts(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float = 1e-6,
) -> dict[str, torch.Tensor]:
    selected = mask.to(device=prediction.device, dtype=prediction.dtype)
    if float(selected.sum()) == 0:
        zero = prediction.sum() * 0
        return {"bce": zero, "dice": zero, "total": zero}
    bounded = prediction.clamp(epsilon, 1 - epsilon)
    bce = (F.binary_cross_entropy(bounded, target, reduction="none") * selected).sum() / selected.sum()
    intersection = (prediction * target * selected).sum()
    dice = 1 - (2 * intersection + epsilon) / (
        (prediction * selected).sum() + (target * selected).sum() + epsilon
    )
    return {"bce": bce, "dice": dice, "total": bce + dice}


def _mask_quantiles(values: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    selected = values.detach().float().cpu()[mask.detach().bool().cpu()]
    if selected.numel() == 0:
        raise ValueError("cannot summarize an empty transition mask")
    return {
        "min": float(selected.min()),
        "q25": float(torch.quantile(selected, 0.25)),
        "median": float(torch.quantile(selected, 0.5)),
        "mean": float(selected.mean()),
        "q75": float(torch.quantile(selected, 0.75)),
        "max": float(selected.max()),
    }


def _single_image(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 4 and value.shape[0] == 1:
        return value[0]
    if value.ndim == 3:
        return value
    raise ValueError(f"expected one NCHW/CHW image, got {tuple(value.shape)}")


def _build_transition_target(
    sample: dict[str, Any],
    output: Path,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    target_data = boundary_aware_transition_alpha_target(
        sample["target_foreground_mask"].to(device),
        sample["target_base_foreground_mask"].to(device),
        sample["target_edit_core_mask"].to(device),
        sample["target_preserve_mask"].to(device),
        sample["target_protected_mask"].to(device),
    )
    destination = output / "transition_alpha_target_v5_3"
    destination.mkdir(parents=True, exist_ok=False)
    torch.save({name: value.detach().cpu() for name, value in target_data.items()}, destination / "target_tensors.pt")
    for name in ("target", "w_edit", "w_base"):
        save_render_tensor(
            destination / f"{name}.png",
            _single_image(target_data[name].detach().cpu()),
            1,
        )
    for name in ("d_edit", "d_base"):
        value = target_data[name].detach().cpu()
        normalized = value / value.max().clamp_min(1e-12)
        save_render_tensor(
            destination / f"{name}_normalized.png", _single_image(normalized), 1,
        )
    transition = sample["target_transition_mask"].to(device).bool()
    protected = sample["target_protected_mask"].to(device).bool()
    alpha_edit = sample["target_foreground_mask"].to(device)
    alpha_base = sample["target_base_foreground_mask"].to(device)
    if transition.ndim == 3:
        transition = transition.unsqueeze(0)
        protected = protected.unsqueeze(0)
        alpha_edit = alpha_edit.unsqueeze(0)
        alpha_base = alpha_base.unsqueeze(0)
    count = int(transition.sum())
    positive = int((transition & (alpha_edit >= 0.5)).sum())
    disagreement = int((transition & ((alpha_edit >= 0.5) != (alpha_base >= 0.5))).sum())
    stats = {
        "definition": "w_edit=d_base/(d_edit+d_base+1e-6); w_base=1-w_edit; target=w_edit*A_edit+w_base*A_base",
        "distance_sampling": [1.0, 1.0],
        "distance_backend": "dependency-free exact separable Euclidean distance transform",
        "transition_pixel_count": count,
        "edit_foreground_positive_count": positive,
        "edit_foreground_positive_ratio": positive / count,
        "edit_foreground_negative_count": count - positive,
        "edit_foreground_negative_ratio": (count - positive) / count,
        "edit_base_disagreement_count": disagreement,
        "edit_base_disagreement_ratio": disagreement / count,
        "d_edit_transition": _mask_quantiles(target_data["d_edit"], transition),
        "d_base_transition": _mask_quantiles(target_data["d_base"], transition),
        "w_edit_transition": _mask_quantiles(target_data["w_edit"], transition),
        "soft_target_transition": _mask_quantiles(target_data["target"], transition),
        "weights_in_range": bool(
            torch.all((target_data["w_edit"] >= 0) & (target_data["w_edit"] <= 1))
            and torch.all((target_data["w_base"] >= 0) & (target_data["w_base"] <= 1))
        ),
        "weight_sum_max_abs_error": float(
            (target_data["w_edit"] + target_data["w_base"] - 1).abs().max()
        ),
        "protected_target_equals_base": bool(
            torch.equal(target_data["target"][protected], alpha_base[protected])
        ),
    }
    _write_json(output / "transition_target_statistics_v5_3.json", stats)
    _target_contact_sheet(output / "transition_weight_contact_sheet_v5_3.png", sample, target_data)
    return target_data, stats


def _target_contact_sheet(path: Path, sample: dict[str, Any], target: dict[str, torch.Tensor]) -> None:
    d_edit = target["d_edit"].detach().cpu()
    d_base = target["d_base"].detach().cpu()
    items = [
        ("A edit", _tensor_image(sample["target_foreground_mask"], 1)),
        ("A base", _tensor_image(sample["target_base_foreground_mask"], 1)),
        ("edit core", _tensor_image(sample["target_edit_core_mask"], 1)),
        ("preserve core", _tensor_image(sample["target_preserve_mask"], 1)),
        ("transition", _tensor_image(sample["target_transition_mask"], 1)),
        ("d edit", _tensor_image(_single_image(d_edit / d_edit.max().clamp_min(1e-12)), 1)),
        ("d base", _tensor_image(_single_image(d_base / d_base.max().clamp_min(1e-12)), 1)),
        ("w edit", _tensor_image(_single_image(target["w_edit"].detach().cpu()), 1)),
        ("w base", _tensor_image(_single_image(target["w_base"].detach().cpu()), 1)),
        ("soft alpha target", _tensor_image(_single_image(target["target"].detach().cpu()), 1)),
    ]
    _save_labeled_grid(path, items, columns=5)


def _save_labeled_grid(path: Path, items: list[tuple[str, Image.Image]], columns: int = 4) -> None:
    cell = (256, 384)
    rows = math.ceil(len(items) / columns)
    canvas = Image.new("RGB", (cell[0] * columns, (cell[1] + 26) * rows), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(items):
        picture = image.convert("RGB").resize(cell, Image.Resampling.LANCZOS)
        x = index % columns * cell[0]
        y = index // columns * (cell[1] + 26)
        canvas.paste(picture, (x, y))
        draw.text((x + 4, y + cell[1] + 4), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _audit_step0(
    runtime: dict[str, Any],
    target_data: dict[str, torch.Tensor],
    stats: dict[str, Any],
    output: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model = runtime["model"]
    result, rgb, alpha = _forward(
        model,
        runtime["base"],
        runtime["adapter"],
        runtime["episode"],
        runtime["geometry"],
        runtime["checkpoint"],
        runtime["background"],
        device,
    )
    parts = _loss_v5_3(rgb, alpha, runtime["sample"], target_data["target"], device, args)
    residual, gate = _regularizers(result)
    components = _component_values(parts, residual, gate, args)
    parameters = [value for value in model.parameters() if value.requires_grad]
    indices = _module_parameter_indices(model, parameters)
    trunk = indices["shared_decoder_trunk"]
    edit_gradient = _gradient_tuple(components["edit"][0], parameters)
    clothing_gradient = _gradient_tuple(components["clothing"][0], parameters)
    rgb_gradient = _combine_gradients([(1.0, edit_gradient), (1.0, clothing_gradient)])
    new_transition_gradient = _gradient_tuple(parts["alpha_transition"], parameters)
    rgb_norm = _gradient_norm(rgb_gradient, trunk)
    raw_transition_norm = _gradient_norm(new_transition_gradient, trunk)
    original_coefficient = args.alpha_weight * args.alpha_transition_weight
    cap = compute_static_transition_gradient_cap(rgb_norm, raw_transition_norm, original_coefficient)
    coefficient = float(cap["final_frozen_coefficient"])
    components["alpha_transition"] = (parts["alpha_transition"], coefficient)
    weighted_transition = _combine_gradients([(coefficient, new_transition_gradient)])
    applied_cosine, _, _ = _cosine(rgb_gradient, weighted_transition, trunk)
    cap["applied_transition_vs_rgb_cosine"] = applied_cosine
    cap["computed_once_at_step"] = 0
    cap["frozen_for_steps"] = [0, args.steps]
    cap["original_effective_coefficient_definition"] = "alpha_weight * alpha_transition_weight"
    _write_json(output / "transition_gradient_cap_v5_3.json", cap)

    sample = runtime["sample"]
    transition_mask = sample["target_transition_mask"].to(device) * (
        1 - sample["target_protected_mask"].to(device)
    )
    legacy = _legacy_transition_parts(
        alpha,
        sample["target_foreground_mask"].to(device),
        transition_mask,
    )
    legacy_gradients = {
        "BCE": _gradient_tuple(legacy["bce"], parameters),
        "Dice": _gradient_tuple(legacy["dice"], parameters),
    }
    rows = []
    for name, gradient in legacy_gradients.items():
        edit_cosine, _, _ = _cosine(gradient, edit_gradient, trunk)
        clothing_cosine, _, _ = _cosine(gradient, clothing_gradient, trunk)
        rows.append({
            "component": name,
            "raw_value": float(legacy[name.lower()].detach()),
            "shared_trunk_gradient_norm": _gradient_norm(gradient, trunk),
            "cosine_vs_edit_rgb": edit_cosine,
            "cosine_vs_clothing_rgb": clothing_cosine,
        })
    _write_csv(output / "alpha_transition_component_gradients_v5_3.csv", rows)
    root_cause = {
        "legacy_definition": "BCE + Dice against A_edit in the narrow transition ring",
        "legacy_bce_raw": float(legacy["bce"].detach()),
        "legacy_dice_raw": float(legacy["dice"].detach()),
        "legacy_total_raw": float(legacy["total"].detach()),
        "transition_statistics": stats,
        "component_gradients": rows,
        "adjudication": "Replace only transition alpha with a continuous distance target and SmoothL1; retain BCE+Dice for edit/base alpha.",
    }
    root_lines = [
        "# Alpha Transition Root Cause V5.3",
        "",
        "The legacy transition term was reproduced at the frozen V5.2 step-0 state before the new objective was used.",
        "",
        f"- transition pixels: `{stats['transition_pixel_count']}`",
        f"- edit-target foreground/background ratio: `{stats['edit_foreground_positive_ratio']:.9f}` / `{stats['edit_foreground_negative_ratio']:.9f}`",
        f"- A_edit/A_base disagreements in ring: `{stats['edit_base_disagreement_count']}`",
        f"- legacy BCE / Dice / total: `{root_cause['legacy_bce_raw']:.9g}` / `{root_cause['legacy_dice_raw']:.9g}` / `{root_cause['legacy_total_raw']:.9g}`",
        "",
        "| Component | trunk norm | cosine vs edit | cosine vs clothing |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        root_lines.append(
            f"| {row['component']} | {row['shared_trunk_gradient_norm']:.9g} | "
            f"{row['cosine_vs_edit_rgb']:.9g} | {row['cosine_vs_clothing_rgb']:.9g} |"
        )
    root_lines.extend(["", root_cause["adjudication"]])
    (output / "ALPHA_TRANSITION_ROOT_CAUSE_V5_3.md").write_text(
        "\n".join(root_lines) + "\n", encoding="utf-8"
    )

    gradients = {
        name: _gradient_tuple(raw, parameters)
        for name, (raw, _) in components.items()
    }
    weighted = {
        name: _combine_gradients([(components[name][1], gradients[name])])
        for name in components
    }
    groups = {
        "edit": weighted["edit"],
        "clothing": weighted["clothing"],
        "new_alpha_transition": weighted["alpha_transition"],
        "rgb_group": _combine_gradients([(1.0, weighted["edit"]), (1.0, weighted["clothing"])]),
        "alpha_group": _combine_gradients([
            (1.0, weighted["alpha_edit"]),
            (1.0, weighted["alpha_transition"]),
            (1.0, weighted["alpha_base"]),
        ]),
    }
    pairs = (
        ("edit", "clothing"),
        ("edit", "new_alpha_transition"),
        ("clothing", "new_alpha_transition"),
        ("rgb_group", "alpha_group"),
    )
    cosines = {}
    for first, second in pairs:
        cosine, first_norm, second_norm = _cosine(groups[first], groups[second], trunk)
        cosines[f"{first}_vs_{second}"] = {
            "cosine": cosine,
            "first_trunk_norm": first_norm,
            "second_trunk_norm": second_norm,
        }
    objective = _objective(components, BASELINE_SCALES)
    model.zero_grad(set_to_none=True)
    objective.backward()
    module_gradients = _gradient_snapshot(model, runtime["base"])
    diagnostics = {
        "objective": float(objective.detach()),
        "losses": {name: float(value.detach()) for name, value in parts.items()},
        "cosines": cosines,
        "transition_cap": cap,
        "module_gradients": module_gradients,
        "six_heads_finite": all(
            math.isfinite(module_gradients["six_channel_heads"][name]["norm"])
            for name in CHANNELS
        ),
        "frozen_base_gradient_zero": module_gradients["frozen_base_grad_count"] == 0,
        "frozen_backbone_gradient_zero": module_gradients["frozen_image_backbone"]["parameter_count_with_grad"] == 0,
        "legacy_root_cause": root_cause,
    }
    _write_json(output / "step0_gradient_diagnostics_v5_3.json", diagnostics)
    return cap, diagnostics


def _linear_slope(rows: list[dict[str, Any]], name: str) -> float:
    return float(np.polyfit(
        np.asarray([row["step"] for row in rows], dtype=np.float64),
        np.asarray([row[name] for row in rows], dtype=np.float64),
        1,
    )[0])


def evaluate_history_v5_3(
    history: list[dict[str, Any]],
    *,
    base_unchanged: bool,
    backbone_unchanged: bool,
    shoe_closer_to_base: bool,
) -> dict[str, Any]:
    if len(history) != 121 or [row["step"] for row in history] != list(range(121)):
        raise ValueError("a V5.3 run must contain states 0 through 120")
    first10 = history[0:10]
    last20 = history[101:121]
    last40 = history[81:121]
    means = {
        "edit_first10": float(np.mean([row["edit"] for row in first10])),
        "edit_last20": float(np.mean([row["edit"] for row in last20])),
        "clothing_first10": float(np.mean([row["clothing"] for row in first10])),
        "clothing_last20": float(np.mean([row["clothing"] for row in last20])),
    }
    means["edit_reduction_fraction"] = 1 - means["edit_last20"] / means["edit_first10"]
    means["clothing_reduction_fraction"] = 1 - means["clothing_last20"] / means["clothing_first10"]
    slopes = {
        "edit_last40": _linear_slope(last40, "edit"),
        "clothing_last40": _linear_slope(last40, "clothing"),
    }
    initial, final = history[0], history[-1]
    criteria = {
        "finite": all(
            math.isfinite(float(value))
            for row in history for key, value in row.items() if key != "step"
        ),
        "total_not_above_step0": final["objective"] <= initial["objective"],
        "edit_last20_mean_reduced_at_least_0_5pct": means["edit_reduction_fraction"] >= 0.005,
        "clothing_last20_mean_reduced_at_least_0_5pct": means["clothing_reduction_fraction"] >= 0.005,
        "edit_last40_slope_negative": slopes["edit_last40"] < 0,
        "clothing_last40_slope_negative": slopes["clothing_last40"] < 0,
        "protected_within_110pct_and_below_0_005": final["protected"] <= initial["protected"] * 1.10 and final["protected"] < 0.005,
        "preserve_within_110pct": final["preserve"] <= initial["preserve"] * 1.10,
        "alpha_base_within_110pct": final["alpha_base"] <= initial["alpha_base"] * 1.10,
        "shoe_closer_to_base_than_raw": shoe_closer_to_base,
        "frozen_base_bitwise_unchanged": base_unchanged,
        "frozen_backbone_bitwise_unchanged": backbone_unchanged,
    }
    pass_status = all(criteria.values())
    safety = all(value for key, value in criteria.items() if not key.startswith(("edit_last", "clothing_last")))
    negative_trends = criteria["edit_last40_slope_negative"] and criteria["clothing_last40_slope_negative"]
    partial = bool(not pass_status and negative_trends and safety)
    return {
        "means": means,
        "slopes": slopes,
        "criteria": criteria,
        "status": "PASS" if pass_status else ("PARTIAL" if partial else "FAIL"),
    }


def _run_training(
    runtime: dict[str, Any],
    target: torch.Tensor,
    w_edit: torch.Tensor,
    frozen_coefficient: float,
    shoe: torch.Tensor,
    output: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model, base, optimizer = runtime["model"], runtime["base"], runtime["optimizer"]
    base_before = {name: getattr(base, name).detach().cpu().clone() for name in BASE_PARAMETER_NAMES}
    backbone_before = _state_clone(model.clothing_observation_encoder.backbone)
    history: list[dict[str, Any]] = []
    renders: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    render_dir = output / "renders"
    render_dir.mkdir(parents=True, exist_ok=False)
    final_gradients = None
    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        result, rgb, alpha = _forward(
            model, base, runtime["adapter"], runtime["episode"], runtime["geometry"],
            runtime["checkpoint"], runtime["background"], device,
        )
        parts = _loss_v5_3(rgb, alpha, runtime["sample"], target, device, args)
        residual, gate = _regularizers(result)
        components = _component_values(parts, residual, gate, args)
        components["alpha_transition"] = (parts["alpha_transition"], frozen_coefficient)
        objective = _objective(components, BASELINE_SCALES)
        if not torch.isfinite(objective):
            raise FloatingPointError(f"non-finite V5.3 objective at step {step}")
        objective.backward()
        gated = result["gated_anchor_residuals"]
        row = {
            "step": step,
            "objective": float(objective.detach()),
            "edit": float(parts["edit"].detach()),
            "clothing": float(parts["clothing"].detach()),
            "preserve": float(parts["preserve"].detach()),
            "protected": float(parts["protected"].detach()),
            "transition": float(parts["transition"].detach()),
            "alpha_edit": float(parts["alpha_edit"].detach()),
            "alpha_transition": float(parts["alpha_transition"].detach()),
            "alpha_base": float(parts["alpha_base"].detach()),
            "residual_magnitude": float(residual.detach()),
            "gate_regularization": float(gate.detach()),
            "geometry_gate_mean": float(result["completion"].geometry_gate.detach().mean()),
            "appearance_gate_mean": float(result["completion"].appearance_gate.detach().mean()),
            "gradient_norm": _total_gradient_norm(model),
            "parameter_norm": _total_parameter_norm(model),
            "xyz_residual_abs_max": float(gated.xyz.detach().abs().max()),
            "opacity_residual_abs_max": float(gated.opacity.detach().abs().max()),
        }
        history.append(row)
        if step in LOG_STEPS:
            print(json.dumps(row), flush=True)
        if step in RENDER_STEPS:
            rgb_cpu, alpha_cpu = rgb.detach().cpu(), alpha.detach().cpu()
            renders[step] = (rgb_cpu, alpha_cpu)
            save_render_tensor(render_dir / f"prediction_step{step:03d}.png", rgb_cpu, 3)
            save_render_tensor(render_dir / f"alpha_step{step:03d}.png", alpha_cpu, 1)
        if step == args.steps:
            final_gradients = _gradient_snapshot(model, base)
            break
        optimizer.step()
    _write_csv(output / "history_v5_3.csv", history)
    _write_json(output / "history_v5_3.json", history)
    base_unchanged = all(
        torch.equal(before, getattr(base, name).detach().cpu())
        for name, before in base_before.items()
    )
    backbone_unchanged = _state_equal(backbone_before, model.clothing_observation_encoder.backbone)
    final_rgb, final_alpha = renders[120]
    sample = runtime["sample"]
    base_rgb, edit_rgb = sample["target_base_rgb"].cpu(), sample["target_edit_rgb"].cpu()
    shoe3 = shoe.unsqueeze(0).expand_as(base_rgb)
    shoe1 = shoe.unsqueeze(0)
    visual_metrics = {
        "raw_edit_vs_base_shoe_rgb_mae": float((edit_rgb - base_rgb).abs()[shoe3].mean()),
        "step120_vs_base_shoe_rgb_mae": float((final_rgb - base_rgb).abs()[shoe3].mean()),
        "step120_vs_raw_shoe_rgb_mae": float((final_rgb - edit_rgb).abs()[shoe3].mean()),
        "step120_vs_base_shoe_alpha_mae": float(
            (final_alpha - sample["target_base_foreground_mask"]).abs()[shoe1].mean()
        ),
    }
    decision = evaluate_history_v5_3(
        history,
        base_unchanged=base_unchanged,
        backbone_unchanged=backbone_unchanged,
        shoe_closer_to_base=(visual_metrics["step120_vs_base_shoe_rgb_mae"] < visual_metrics["step120_vs_raw_shoe_rgb_mae"]),
    )
    _final_contact_sheet(
        output / "visual_acceptance_contact_sheet_v5_3.png",
        sample, renders, target, w_edit, shoe,
    )
    return {
        "history": history,
        "summary": decision,
        "visual_metrics": visual_metrics,
        "final_gradient_snapshot": final_gradients,
        "base_max_change": {
            name: float((getattr(base, name).detach().cpu() - before).abs().max())
            for name, before in base_before.items()
        },
        "backbone_bitwise_unchanged": backbone_unchanged,
    }


def _final_contact_sheet(
    path: Path,
    sample: dict[str, Any],
    renders: dict[int, tuple[torch.Tensor, torch.Tensor]],
    target: torch.Tensor,
    w_edit: torch.Tensor,
    shoe: torch.Tensor,
) -> None:
    base, edit = sample["target_base_rgb"].cpu(), sample["target_edit_rgb"].cpu()
    final_rgb, final_alpha = renders[120]
    edit_mask = sample["target_edit_mask"].cpu()
    clothing = sample["target_clothing_mask"].cpu()
    protected = sample["target_protected_mask"].cpu()
    shoe3 = shoe.unsqueeze(0).expand_as(base)
    items = [
        ("subject02 base", _tensor_image(base, 3)),
        ("raw direct edit", _tensor_image(edit, 3)),
        ("edit core", _tensor_image(sample["target_edit_core_mask"], 1)),
        ("safe clothing", _tensor_image(clothing, 1)),
        ("transition ring", _tensor_image(sample["target_transition_mask"], 1)),
        ("w edit", _tensor_image(_single_image(w_edit.detach().cpu()), 1)),
        ("soft alpha target", _tensor_image(_single_image(target.detach().cpu()), 1)),
        *[(f"prediction step {step}", _tensor_image(renders[step][0], 3)) for step in (0, 20, 40, 80, 120)],
        ("edit-region diff", _tensor_image((final_rgb - edit).abs() * edit_mask, 3)),
        ("clothing-region diff", _tensor_image((final_rgb - edit).abs() * clothing, 3)),
        ("protected-region diff", _tensor_image((final_rgb - base).abs() * protected, 3)),
        ("shoe-region diff", _tensor_image(torch.where(shoe3, (final_rgb - base).abs(), torch.zeros_like(base)), 3)),
        ("predicted alpha", _tensor_image(final_alpha, 1)),
    ]
    _save_labeled_grid(path, items, columns=4)


def _protected_flip_check(
    runtime: dict[str, Any],
    target: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    with torch.no_grad():
        _, rgb, alpha = _forward(
            runtime["model"], runtime["base"], runtime["adapter"], runtime["episode"],
            runtime["geometry"], runtime["checkpoint"], runtime["background"], device,
        )
    sample = runtime["sample"]
    original_rgb = sample["target_edit_rgb"]
    original_alpha = sample["target_foreground_mask"]
    protected = sample["target_protected_mask"].bool()
    flipped_rgb = original_rgb.clone()
    flipped_rgb[protected.expand_as(flipped_rgb)] = 1 - flipped_rgb[protected.expand_as(flipped_rgb)]
    flipped_alpha = original_alpha.clone()
    flipped_alpha[protected] = 1 - flipped_alpha[protected]
    first = _loss_v5_3(rgb, alpha, sample, target, device, args)
    sample["target_edit_rgb"] = flipped_rgb
    sample["target_foreground_mask"] = flipped_alpha
    try:
        second = _loss_v5_3(rgb, alpha, sample, target, device, args)
    finally:
        sample["target_edit_rgb"] = original_rgb
        sample["target_foreground_mask"] = original_alpha
    keys = ("edit", "clothing", "transition", "alpha_edit", "alpha_transition", "total")
    deltas = {name: float((second[name] - first[name]).abs()) for name in keys}
    return {"deltas": deltas, "strict_zero": all(value == 0 for value in deltas.values())}


def _contract_fingerprint(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "inputs": contract["inputs"],
        "external_inputs": [
            {"role": item["role"], "sha256": item["sha256"]}
            for item in contract["external_inputs"]
        ],
        "episode": {
            "outfit_id": contract["outfit_id"],
            "reference_condition_ids": contract["reference_condition_ids"],
            "target_condition_id": contract["target_condition_id"],
            "seed": contract["seed"],
            "sample_index": contract["sample_index"],
            "optimizer": contract["optimizer"],
        },
    }


def _validate_preregistered_config(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    episode = payload["fixed_episode"]
    loss = payload["loss"]
    transition = payload["transition_alpha"]
    expected = {
        "outfit_id": EXPECTED_OUTFIT,
        "references": list(EXPECTED_REFERENCES),
        "target": EXPECTED_TARGET,
        "sample_index": args.sample_index,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "optimization_steps": args.steps,
    }
    if episode != expected:
        raise ValueError(f"V5.3 config fixed episode differs from preregistration: {episode}")
    required_losses = {
        "edit": 1.0, "clothing": 1.0, "preserve": 1.0, "protected": 1.0,
        "rgb_transition": 0.25, "alpha_global": 0.5, "alpha_edit": 1.0,
        "alpha_transition_original": 0.25, "alpha_base": 1.0,
        "residual_regularization": 1e-4, "gate_regularization": 0.0,
    }
    if loss != required_losses:
        raise ValueError("V5.3 loss weights differ from the frozen V5.2 baseline")
    if transition["loss"] != "smooth_l1" or transition["dice"] is not False:
        raise ValueError("V5.3 transition alpha must use SmoothL1 without Dice")
    if transition["gradient_cap"]["dynamic_updates"] is not False:
        raise ValueError("V5.3 transition gradient cap must be static")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the single V5.3 boundary-aware 120-step closure")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gate4-config", type=Path, required=True)
    parser.add_argument("--module2-config", type=Path, required=True)
    parser.add_argument("--module3-checkpoint", type=Path, required=True)
    parser.add_argument("--v5-2-initial-state", type=Path, required=True)
    parser.add_argument("--v5-2-contract", type=Path, required=True)
    parser.add_argument("--v5-3-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=11)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--alpha-weight", type=float, default=0.5)
    parser.add_argument("--alpha-edit-weight", type=float, default=1.0)
    parser.add_argument("--alpha-transition-weight", type=float, default=0.25)
    parser.add_argument("--alpha-base-weight", type=float, default=1.0)
    parser.add_argument("--residual-regularization-weight", type=float, default=1e-4)
    parser.add_argument("--gate-regularization-weight", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    if args.steps != 120 or args.sample_index != 11 or args.seed != 20260717 or args.learning_rate != 1e-5:
        raise ValueError("V5.3 acceptance is frozen to 120 steps, sample 11, seed 20260717, and LR 1e-5")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty V5.3 output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "run_partial.json", {"status": "RUNNING", "stage": "initial_state"})
    try:
        if _sha256(args.v5_2_initial_state) != EXPECTED_INITIAL_STATE_SHA256:
            raise ValueError("V5.2 initial state SHA256 does not match the frozen contract")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        device = torch.device(args.device)
        preregistered_config = _validate_preregistered_config(args.v5_3_config.resolve(), args)
        runtime = _build_runtime(args, device)
        restore = _restore_initial_state(args.v5_2_initial_state, runtime["model"], runtime["optimizer"])
        checkpoint_payload = torch.load(args.v5_2_initial_state, map_location="cpu", weights_only=False)
        restore.update({
            "source_path": str(args.v5_2_initial_state.resolve()),
            "source_sha256": EXPECTED_INITIAL_STATE_SHA256,
            "optimizer_state_entries": len(runtime["optimizer"].state),
            "optimizer_learning_rates": [group["lr"] for group in runtime["optimizer"].param_groups],
            "optimizer_matches_frozen_step0": runtime["optimizer"].state_dict() == checkpoint_payload["optimizer"],
        })
        _write_json(output / "initial_state_verification_v5_3.json", restore)
        contract = _fixed_episode_contract(args.manifest.resolve(), runtime["sample"], args)
        contract["schema_version"] = "canondressgs.fixed_episode.v5.3"
        previous = json.loads(args.v5_2_contract.read_text(encoding="utf-8"))
        current_fingerprint = _contract_fingerprint(contract)
        previous_fingerprint = _contract_fingerprint(previous)
        contract["v5_2_input_fingerprint_exact"] = current_fingerprint == previous_fingerprint
        contract["v5_2_contract_path"] = str(args.v5_2_contract.resolve())
        contract["v5_3_config"] = {
            "path": str(args.v5_3_config.resolve()),
            "sha256": _sha256(args.v5_3_config),
            "schema_version": preregistered_config["schema_version"],
        }
        if not contract["v5_2_input_fingerprint_exact"]:
            raise AssertionError("V5.3 input SHA/episode fingerprint differs from V5.2")
        _write_json(output / "fixed_episode_contract_v5_3.json", contract)
        target_data, target_stats = _build_transition_target(runtime["sample"], output, device)
        cap, step0 = _audit_step0(runtime, target_data, target_stats, output, device, args)
        restore_after_audit = _restore_initial_state(args.v5_2_initial_state, runtime["model"], runtime["optimizer"])
        _write_json(output / "post_audit_initial_state_verification_v5_3.json", restore_after_audit)
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        shoe_path = Path(manifest["protected_region_final_adjudication"]["target_contract"]["shoe_mask"])
        shoe_path = shoe_path if shoe_path.is_absolute() else args.manifest.resolve().parent / shoe_path
        shoe = torch.from_numpy(np.array(Image.open(shoe_path).convert("L"), copy=True) >= 128)
        training_result = _run_training(
            runtime, target_data["target"], target_data["w_edit"],
            float(cap["final_frozen_coefficient"]),
            shoe, output, device, args,
        )
        flip = _protected_flip_check(runtime, target_data["target"], device, args)
        status = training_result["summary"]["status"]
        if not flip["strict_zero"]:
            status = "FAIL"
        report = {
            "status": status,
            "run_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
            "fixed_episode": {
                "outfit": EXPECTED_OUTFIT,
                "references": list(EXPECTED_REFERENCES),
                "target": EXPECTED_TARGET,
                "seed": args.seed,
                "steps": args.steps,
            },
            "legacy_root_cause": {
                "bce": step0["legacy_root_cause"]["legacy_bce_raw"],
                "dice": step0["legacy_root_cause"]["legacy_dice_raw"],
                "evidence_file": "ALPHA_TRANSITION_ROOT_CAUSE_V5_3.md",
            },
            "transition_target": target_stats,
            "transition_cap": cap,
            "step0_gradients": step0,
            "training": training_result,
            "protected_flip_isolation": flip,
            "single_candidate_only": True,
            "phase_b_used": False,
            "source_data_or_masks_modified": False,
        }
        _write_json(output / "V5_3_FINAL_STATUS.json", report)
        (output / "V5_3_FINAL_STATUS.md").write_text(
            "# V5.3 Boundary-Aware Alpha Objective Closure\n\n"
            f"- status: **{status}**\n"
            f"- edit first-10/last-20 reduction: `{training_result['summary']['means']['edit_reduction_fraction']:.6%}`\n"
            f"- clothing first-10/last-20 reduction: `{training_result['summary']['means']['clothing_reduction_fraction']:.6%}`\n"
            f"- edit/clothing last-40 slopes: `{training_result['summary']['slopes']['edit_last40']:.9g}` / `{training_result['summary']['slopes']['clothing_last40']:.9g}`\n"
            f"- frozen transition coefficient: `{cap['final_frozen_coefficient']:.9g}`\n"
            f"- protected flip isolation strict zero: `{flip['strict_zero']}`\n",
            encoding="utf-8",
        )
        _write_json(output / "run_partial.json", {"status": "COMPLETE", "stage": "final", "outcome": status})
        print(json.dumps({"status": status, "output": str(output)}, indent=2))
        if status == "FAIL":
            raise SystemExit(1)
    except Exception as error:
        _write_json(output / "run_partial.json", {
            "status": "FAIL", "stage": "exception",
            "exception_type": type(error).__name__, "message": str(error),
        })
        raise


if __name__ == "__main__":
    main()
