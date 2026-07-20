from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_dressable as training  # noqa: E402
from scene.gaussian_clothing_residuals import (  # noqa: E402
    CHANNELS,
    GaussianClothingResiduals,
    apply_protected_full_residual_guard,
)
from scene.image_conditioned_failure_diagnostics import (  # noqa: E402
    DiagnosticOutfitLatents,
    channel_comparison,
    normalized_residual_regression_loss,
    residual_distance_scalar,
)
from scene.residual_field_parameterizations import (  # noqa: E402
    AnchorSupportTokenResidualField,
    DirectPerGaussianResidualTableControl,
    FIXED_ONE_GATE_MODE,
    GaussianSupportTokenResidualField,
    residual_output_shapes,
)
from tools import diagnose_image_conditioned_overfit_failure as diagnosis  # noqa: E402
from tools import run_image_conditioned_overfit_o01 as o01  # noqa: E402
from tools import run_residual_decoder_capacity_v7 as v7  # noqa: E402
from utils.mmlphuman_anchor_deformation import MMLPHumanAnchorDeformationAdapter  # noqa: E402


SCHEMA = "canondressgs.residual_field_parameterization.v1"
TASK_ID = "SUBJECT02-RESIDUAL-FIELD-PARAMETERIZATION-001"
EXPECTED_BRANCH = "research/residual-field-parameterization-20260720"
EXPECTED_SOURCE_HEAD = "b1ddd084b51c2eba83b3c26ec6b479d1d162b3c3"
OUTFITS = ("O01", "O08")
CONDITIONS = o01.CONDITIONS
VIEWS = o01.VIEWS


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
        handle.flush()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def immutable_tree_metadata_fingerprint(path: Path) -> str:
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for item in sorted((value for value in path.rglob("*") if value.is_file()), key=lambda value: value.as_posix()):
        stat = item.stat()
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA or config.get("task_id") != TASK_ID:
        raise ValueError("not the preregistered residual-field parameterization contract")
    if config.get("branch") != EXPECTED_BRANCH or config.get("source_head") != EXPECTED_SOURCE_HEAD:
        raise ValueError("branch/source contract changed")
    if tuple(config.get("outfits", ())) != OUTFITS or tuple(config.get("conditions", ())) != CONDITIONS:
        raise ValueError("two-outfit/four-view protocol changed")
    if int(config["p0"]["max_steps"]) != 300:
        raise ValueError("P0 must use exactly the preregistered 300-step ceiling")
    if int(config["p1"]["max_steps"]) != 1000 or int(config["p2"]["max_steps"]) != 1000:
        raise ValueError("P1/P2 must use exactly the preregistered 1000-step ceiling")
    if int(config["p1"]["token_dim"]) != 24 or int(config["p2"]["token_dim"]) != 24:
        raise ValueError("token_dim search is forbidden; P1/P2 must use 24")
    if config["p0"]["gate_mode"] != FIXED_ONE_GATE_MODE or any(
        config[name]["gate_mode"] != FIXED_ONE_GATE_MODE for name in ("p1", "p2")
    ):
        raise ValueError("all capacity gates must remain fixed one")
    if any(bool(value) for value in config["permissions"].values()):
        raise ValueError("a forbidden mutation was enabled")
    frozen = {
        "p0": {
            "mean_bound_normalized_rmse_max": 0.02,
            "mean_direction_cosine_min": 0.95,
            "mean_top_10pct_overlap_min": 0.90,
            "per_outfit_render_garment_mae_max": 0.01,
        },
        "p1": {
            "mean_bound_normalized_rmse_max": 0.12,
            "mean_direction_cosine_min": 0.70,
            "mean_top_10pct_overlap_min": 0.50,
            "mean_top_20pct_overlap_min": 0.65,
            "outfit_separation_ratio_min": 0.55,
            "per_outfit_render_garment_mae_max": 0.08,
        },
    }
    for phase, thresholds in frozen.items():
        for name, expected in thresholds.items():
            if float(config[phase]["acceptance"][name]) != expected:
                raise ValueError(f"{phase} acceptance threshold changed: {name}")
    if config["p2"]["acceptance"] != config["p1"]["acceptance"]:
        raise ValueError("P2 acceptance must be identical to P1")
    return config


def bounds(config: Mapping[str, Any]) -> dict[str, float]:
    channels = config["model"]["dressable_channels"]
    return {
        "xyz": float(channels["xyz"]["max_abs"]),
        "log_scaling": float(channels["log_scaling"]["max_abs"]),
        "rotation": float(channels["rotation"]["max_angle_rad"]),
        "opacity_logit": float(channels["opacity_logit"]["max_abs"]),
        "sh0": float(channels["sh0"]["max_abs"]),
        "shN": float(channels["shN"]["max_abs"]),
    }


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def save_checkpoint(
    path: Path,
    *,
    phase: str,
    step: int,
    model_state: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    extra: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "schema_version": SCHEMA,
        "task_id": TASK_ID,
        "phase": phase,
        "global_step": int(step),
        "model": dict(model_state),
        "optimizer": optimizer.state_dict(),
        "rng": rng_state(),
        "target_tensors_stored": False,
        "oracle_residual_in_forward": False,
        "extra": dict(extra),
    }, temporary)
    os.replace(temporary, path)


def set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def slice_residuals(residuals: GaussianClothingResiduals, indices: torch.Tensor) -> GaussianClothingResiduals:
    return GaussianClothingResiduals(**{
        name: value.index_select(0, indices) for name, value in residuals.as_dict().items()
    })


def residual_bundle_cpu(residuals: GaussianClothingResiduals) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in residuals.as_dict().items()}


def _zero_pair_cosine(
    comparison: dict[str, dict[str, Any]],
    prediction: GaussianClothingResiduals,
    target: GaussianClothingResiduals,
    epsilon: float,
) -> dict[str, dict[str, Any]]:
    for name in CHANNELS:
        first = getattr(prediction, name).detach().float()
        second = getattr(target, name).detach().float()
        first_norm = float(first.square().sum().sqrt())
        second_norm = float(second.square().sum().sqrt())
        comparison[name]["zero_prediction"] = first_norm <= epsilon
        comparison[name]["zero_target"] = second_norm <= epsilon
        if first_norm <= epsilon and second_norm <= epsilon:
            comparison[name]["cosine_similarity"] = 1.0
            comparison[name]["cosine_convention"] = "zero_target_zero_prediction_is_exact_match"
    return comparison


def row_magnitude(residuals: GaussianClothingResiduals, channel_bounds: Mapping[str, float]) -> torch.Tensor:
    names = {
        "delta_xyz": "xyz", "delta_log_scaling": "log_scaling", "delta_rotvec": "rotation",
        "delta_opacity_logit": "opacity_logit", "delta_sh0": "sh0", "delta_shN": "shN",
    }
    values = [
        (value.detach().float().reshape(value.shape[0], -1) / float(channel_bounds[names[name]])).square().mean(1)
        for name, value in residuals.as_dict().items()
    ]
    return torch.stack(values, dim=1).mean(1).sqrt()


def residual_metrics(
    prediction: GaussianClothingResiduals,
    target: GaussianClothingResiduals,
    channel_bounds: Mapping[str, float],
    *,
    active_epsilon: float,
) -> dict[str, Any]:
    comparison_10 = _zero_pair_cosine(
        channel_comparison(prediction, target, channel_bounds, active_epsilon=active_epsilon, top_fraction=0.10),
        prediction, target, active_epsilon,
    )
    comparison_20 = channel_comparison(
        prediction, target, channel_bounds, active_epsilon=active_epsilon, top_fraction=0.20,
    )
    pred_magnitude = row_magnitude(prediction, channel_bounds)
    target_magnitude = row_magnitude(target, channel_bounds)
    predicted_active, target_active = pred_magnitude > 1e-5, target_magnitude > 1e-5
    true_positive = int((predicted_active & target_active).sum())
    predicted_count, target_count = int(predicted_active.sum()), int(target_active.sum())
    top_count = max(1, int(round(pred_magnitude.shape[0] * 0.10)))
    pred_top = torch.topk(pred_magnitude, top_count).indices
    target_top = torch.topk(target_magnitude, top_count).indices
    pred_top_mask = torch.zeros_like(predicted_active); pred_top_mask[pred_top] = True
    target_top_mask = torch.zeros_like(target_active); target_top_mask[target_top] = True
    overlap = pred_top_mask & target_top_mask
    return {
        "normalized_rmse": float(np.mean([value["bound_normalized_rmse"] for value in comparison_10.values()])),
        "normalized_mae": float(np.mean([value["bound_normalized_mae"] for value in comparison_10.values()])),
        "direction_cosine": float(np.mean([value["cosine_similarity"] for value in comparison_10.values()])),
        "top_10pct_overlap": float(np.mean([value["top_10pct_row_overlap"] for value in comparison_10.values()])),
        "top_20pct_overlap": float(np.mean([value["top_10pct_row_overlap"] for value in comparison_20.values()])),
        "nonzero_support_precision": true_positive / max(predicted_count, 1),
        "nonzero_support_recall": true_positive / max(target_count, 1),
        "prediction_nonzero_count": predicted_count,
        "target_nonzero_count": target_count,
        "bound_hit_fraction": float(np.mean([value["prediction_bound_hit_ratio"] for value in comparison_10.values()])),
        "spatial_concentration_in_oracle_top10": float(
            pred_magnitude[target_top_mask].sum() / pred_magnitude.sum().clamp_min(1e-12)
        ),
        "top10_true_positive_count": int(overlap.sum()),
        "top10_false_positive_count": int((pred_top_mask & ~target_top_mask).sum()),
        "top10_false_negative_count": int((~pred_top_mask & target_top_mask).sum()),
        "per_attribute": comparison_10,
    }


def mean_metric(per_outfit: Mapping[str, Mapping[str, Any]], name: str) -> float:
    return float(np.mean([float(per_outfit[outfit][name]) for outfit in OUTFITS]))


def render_prediction(
    base: Any,
    sample: Mapping[str, Any],
    residuals: GaussianClothingResiduals,
    background: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return v7.render_prediction(base, sample, residuals, background)


def evaluate_predictions(
    *,
    phase: str,
    step: int,
    predictions: Mapping[str, GaussianClothingResiduals],
    targets: Mapping[str, GaussianClothingResiduals],
    base: Any,
    samples: Mapping[str, Any],
    background: torch.Tensor,
    config: Mapping[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    channel_bounds = bounds(config)
    active_epsilon = float(config["metrics"]["active_epsilon"])
    per_outfit = {
        outfit: residual_metrics(
            predictions[outfit], targets[outfit], channel_bounds, active_epsilon=active_epsilon,
        ) for outfit in OUTFITS
    }
    predicted_separation = residual_distance_scalar(
        _zero_pair_cosine(
            channel_comparison(predictions["O01"], predictions["O08"], channel_bounds),
            predictions["O01"], predictions["O08"], active_epsilon,
        )
    )
    oracle_separation = residual_distance_scalar(
        _zero_pair_cosine(
            channel_comparison(targets["O01"], targets["O08"], channel_bounds),
            targets["O01"], targets["O08"], active_epsilon,
        )
    )
    rows, render_cache = [], {outfit: {} for outfit in OUTFITS}
    render_dir = output_dir / phase / "milestones" / f"step_{step:06d}" / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for outfit in OUTFITS:
            for condition in CONDITIONS:
                sample = samples[f"{outfit}/{condition}"]
                predicted_rgb, predicted_alpha = render_prediction(base, sample, predictions[outfit], background)
                oracle_rgb, oracle_alpha = render_prediction(base, sample, targets[outfit], background)
                garment = diagnosis._garment_mask(sample)
                protected = sample["target_protected_mask"]
                background_mask = 1 - torch.maximum(sample["target_foreground_mask"], sample["target_base_foreground_mask"])
                rows.append({
                    "outfit": outfit,
                    "condition": condition,
                    "view": VIEWS[condition],
                    "garment_rgb_mae_to_oracle": o01._masked_mae(predicted_rgb, oracle_rgb, garment),
                    "protected_rgb_mae_to_oracle": o01._masked_mae(predicted_rgb, oracle_rgb, protected),
                    "background_rgb_mae_to_oracle": o01._masked_mae(predicted_rgb, oracle_rgb, background_mask),
                    "alpha_mae_to_oracle": float((predicted_alpha - oracle_alpha).abs().mean()),
                })
                render_cache[outfit][condition] = {
                    "prediction": (predicted_rgb.detach().cpu(), predicted_alpha.detach().cpu()),
                    "oracle": (oracle_rgb.detach().cpu(), oracle_alpha.detach().cpu()),
                }
                from tools.check_real_image_conditioned_one_batch import save_render_tensor
                save_render_tensor(render_dir / f"{outfit}_{condition}_predicted_rgb.png", predicted_rgb, 3)
                save_render_tensor(render_dir / f"{outfit}_{condition}_predicted_alpha.png", predicted_alpha, 1)
                save_render_tensor(render_dir / f"{outfit}_{condition}_oracle_rgb.png", oracle_rgb, 3)
    render_per_outfit = {
        outfit: float(np.mean([
            row["garment_rgb_mae_to_oracle"] for row in rows if row["outfit"] == outfit
        ])) for outfit in OUTFITS
    }
    loss_per_outfit = {}
    for outfit in OUTFITS:
        loss, _ = normalized_residual_regression_loss(predictions[outfit], targets[outfit], channel_bounds)
        loss_per_outfit[outfit] = float(loss)
    report = {
        "step": step,
        "per_outfit": per_outfit,
        "mean": {
            name: mean_metric(per_outfit, name) for name in (
                "normalized_rmse", "normalized_mae", "direction_cosine", "top_10pct_overlap",
                "top_20pct_overlap", "nonzero_support_precision", "nonzero_support_recall",
                "bound_hit_fraction", "spatial_concentration_in_oracle_top10",
            )
        },
        "loss_per_outfit": loss_per_outfit,
        "mean_loss": float(np.mean(list(loss_per_outfit.values()))),
        "predicted_outfit_separation": predicted_separation,
        "oracle_outfit_separation": oracle_separation,
        "outfit_separation_ratio": predicted_separation / max(oracle_separation, 1e-12),
        "render_rows": rows,
        "render_per_outfit_garment_mae": render_per_outfit,
    }
    milestone = output_dir / phase / "milestones" / f"step_{step:06d}"
    atomic_json(milestone / "metrics.json", report)
    torch.save({outfit: residual_bundle_cpu(value) for outfit, value in predictions.items()}, milestone / "predicted_residuals.pt")
    return report, render_cache


def trainable_snapshot(groups: Mapping[str, list[torch.nn.Parameter]]) -> dict[str, Any]:
    result = {}
    for name, parameters in groups.items():
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        result[name] = {
            "parameter_count": sum(parameter.numel() for parameter in parameters),
            "gradient_tensor_count": len(gradients),
            "gradient_finite": bool(gradients) and all(torch.isfinite(value).all().item() for value in gradients),
            "gradient_l2": float(torch.sqrt(sum(
                (value.detach().float().square().sum() for value in gradients),
                torch.tensor(0.0, device=parameters[0].device),
            ))) if gradients else 0.0,
        }
    return result


def p0_predictions(
    tables: torch.nn.ModuleDict,
    protected_mask: torch.Tensor,
) -> dict[str, GaussianClothingResiduals]:
    return {
        outfit: apply_protected_full_residual_guard(tables[outfit](), protected_mask)
        for outfit in OUTFITS
    }


def load_v7_predictions(context: Mapping[str, Any]) -> dict[str, GaussianClothingResiduals]:
    config, model, base = context["config"], context["model"], context["base"]
    checkpoint = torch.load(Path(config["inputs"]["v7_checkpoint"]), map_location="cpu", weights_only=False)
    model.support_conditioned_residual_decoder_v7.load_state_dict(checkpoint["model"]["decoder"], strict=True)
    latents = DiagnosticOutfitLatents(OUTFITS, 64, 20260720).to(base._xyz)
    latents.load_state_dict(checkpoint["model"]["diagnostic_latents"], strict=True)
    with torch.no_grad():
        return {
            outfit: v7.prediction_from_latent(
                model.support_conditioned_residual_decoder_v7,
                latents(outfit),
                context["protected_mask"],
                chunk_size=16384,
            ) for outfit in OUTFITS
        }


def _p0_contact_sheets(
    context: Mapping[str, Any],
    p0_cache: Mapping[str, Any],
    v7_predictions: Mapping[str, GaussianClothingResiduals],
) -> None:
    visual = context["output_dir"] / "visual_acceptance"
    base, samples, targets, background = (
        context["base"], context["samples"], context["targets"], context["background"],
    )
    v7_cache = {outfit: {} for outfit in OUTFITS}
    with torch.no_grad():
        for outfit in OUTFITS:
            for condition in CONDITIONS:
                sample = samples[f"{outfit}/{condition}"]
                rgb, alpha = render_prediction(base, sample, v7_predictions[outfit], background)
                v7_cache[outfit][condition] = (rgb.detach().cpu(), alpha.detach().cpu())
    for outfit in OUTFITS:
        rows = []
        for condition in CONDITIONS:
            sample = samples[f"{outfit}/{condition}"]
            oracle_rgb = p0_cache[outfit][condition]["oracle"][0]
            p0_rgb = p0_cache[outfit][condition]["prediction"][0]
            garment = diagnosis._garment_mask(sample)
            protected = sample["target_protected_mask"]
            rows.append((f"{outfit}/{VIEWS[condition]}", [
                ("base", sample["target_base_rgb"], 3),
                ("target", sample["target_edit_rgb"], 3),
                ("Oracle", oracle_rgb, 3),
                ("V7", v7_cache[outfit][condition][0], 3),
                ("P0", p0_rgb, 3),
                ("Oracle-P0 abs", (oracle_rgb - p0_rgb).abs(), 3),
                ("garment crop", v7.crop_tensor(p0_rgb, garment), 3),
                ("shoes/protected", v7.crop_tensor(p0_rgb, protected, vertical="lower"), 3),
            ]))
        o01._save_contact_sheet(visual / f"p0_{outfit}_four_view_contact_sheet.png", rows)


def run_p0(context: dict[str, Any]) -> dict[str, Any]:
    config, output_dir, base = context["config"], context["output_dir"], context["base"]
    phase_config = config["p0"]
    torch.manual_seed(int(phase_config["seed"])); torch.cuda.manual_seed_all(int(phase_config["seed"]))
    shapes = residual_output_shapes(base)
    tables = torch.nn.ModuleDict({
        outfit: DirectPerGaussianResidualTableControl(shapes, bounds(config)).to(base._xyz)
        for outfit in OUTFITS
    })
    optimizer = torch.optim.Adam(tables.parameters(), lr=float(phase_config["learning_rate"]))
    groups = {"direct_residual_tables": list(tables.parameters())}
    phase_dir = output_dir / "p0"; phase_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        initial_predictions = p0_predictions(tables, context["protected_mask"])
    initial, _ = evaluate_predictions(
        phase="p0", step=0, predictions=initial_predictions, targets=context["targets"], base=base,
        samples=context["samples"], background=context["background"], config=config, output_dir=output_dir,
    )
    history, milestones = [], {"0": initial}
    final_gradients = {}
    started = time.time(); torch.cuda.reset_peak_memory_stats()
    for step in range(1, int(phase_config["max_steps"]) + 1):
        outfit = OUTFITS[(step - 1) % 2]
        optimizer.zero_grad(set_to_none=True)
        prediction = apply_protected_full_residual_guard(tables[outfit](), context["protected_mask"])
        loss, parts = normalized_residual_regression_loss(prediction, context["targets"][outfit], bounds(config))
        if not torch.isfinite(loss):
            raise FloatingPointError("P0 loss is NaN or Inf")
        loss.backward()
        final_gradients = trainable_snapshot(groups)
        torch.nn.utils.clip_grad_norm_(tables.parameters(), float(phase_config["gradient_clip_norm"]), error_if_nonfinite=True)
        optimizer.step()
        row = {"step": step, "outfit": outfit, "loss": float(loss.detach()), **{
            f"loss_{name}": float(value.detach()) for name, value in parts.items()
        }}
        history.append(row); append_jsonl(phase_dir / "training.jsonl", row)
        if step in phase_config["milestones"]:
            with torch.no_grad():
                predictions = p0_predictions(tables, context["protected_mask"])
            report, cache = evaluate_predictions(
                phase="p0", step=step, predictions=predictions, targets=context["targets"], base=base,
                samples=context["samples"], background=context["background"], config=config, output_dir=output_dir,
            )
            milestones[str(step)] = report
            if step == int(phase_config["max_steps"]):
                final_cache = cache
    elapsed = time.time() - started
    final = milestones[str(phase_config["max_steps"])]
    checkpoint_path = phase_dir / "checkpoints" / "checkpoint_step_000300.pth"
    save_checkpoint(
        checkpoint_path, phase="P0", step=300, model_state={"tables": tables.state_dict()},
        optimizer=optimizer, extra={"next_outfit": "O01", "outfit_update_counts": {"O01": 150, "O08": 150}},
    )
    restored = torch.nn.ModuleDict({
        outfit: DirectPerGaussianResidualTableControl(shapes, bounds(config)).to(base._xyz)
        for outfit in OUTFITS
    })
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=float(phase_config["learning_rate"]))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    restored.load_state_dict(checkpoint["model"]["tables"], strict=True)
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    fixed = torch.arange(256, device=base._xyz.device)
    output_exact = all(
        torch.equal(getattr(tables["O01"](fixed), name), getattr(restored["O01"](fixed), name))
        for name in CHANNELS
    )
    current_rng = rng_state(); o01._restore_rng(checkpoint["rng"])
    rng_exact = o01.object_fingerprint(rng_state()) == o01.object_fingerprint(checkpoint["rng"])
    o01._restore_rng(current_rng)
    resume = {
        "checkpoint": str(checkpoint_path), "sha256": sha256(checkpoint_path),
        "global_step_exact": int(checkpoint["global_step"]) == 300,
        "model_state_exact": v7._tensor_state_fingerprint(tables.state_dict().items()) == v7._tensor_state_fingerprint(restored.state_dict().items()),
        "optimizer_state_exact": o01.object_fingerprint(optimizer.state_dict()) == o01.object_fingerprint(restored_optimizer.state_dict()),
        "rng_state_exact": rng_exact, "fixed_output_bitwise_exact": output_exact,
    }
    resume["pass"] = all(value for name, value in resume.items() if name not in {"checkpoint", "sha256"})
    acceptance = phase_config["acceptance"]
    checks = {
        "mean_normalized_rmse": final["mean"]["normalized_rmse"] <= float(acceptance["mean_bound_normalized_rmse_max"]),
        "mean_direction_cosine": final["mean"]["direction_cosine"] >= float(acceptance["mean_direction_cosine_min"]),
        "mean_top_10pct_overlap": final["mean"]["top_10pct_overlap"] >= float(acceptance["mean_top_10pct_overlap_min"]),
        "o01_render_garment_mae": final["render_per_outfit_garment_mae"]["O01"] <= float(acceptance["per_outfit_render_garment_mae_max"]),
        "o08_render_garment_mae": final["render_per_outfit_garment_mae"]["O08"] <= float(acceptance["per_outfit_render_garment_mae_max"]),
        "finite": all(torch.isfinite(value).all().item() for prediction in p0_predictions(tables, context["protected_mask"]).values() for value in prediction.as_dict().values()),
        "base_bitwise_frozen": context["base_before"] == v7._tensor_state_fingerprint(v7._base_named_tensors(base)),
        "base_gradient_zero": v7._base_gradient_count(base) == 0,
        "checkpoint_resume": resume["pass"],
        "target_images_entered_forward": False,
    }
    numeric_pass = all(checks.values())
    v7_predictions = load_v7_predictions(context)
    _p0_contact_sheets(context, final_cache, v7_predictions)
    result = {
        "status": "NUMERIC_PASS_VISUAL_PENDING" if numeric_pass else "FAIL",
        "optimizer_steps": 300, "milestones": milestones, "final": final,
        "checks": checks, "gradients": final_gradients, "runtime_seconds": elapsed,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "parameter_count": sum(parameter.numel() for parameter in tables.parameters()),
        "checkpoint_resume": resume,
        "visual_status": "PENDING_ACTUAL_INSPECTION" if numeric_pass else "NOT_ELIGIBLE_NUMERIC_FAIL",
    }
    atomic_json(phase_dir / "p0_metrics.json", result)
    atomic_json(phase_dir / "partial_status.json", {
        "status": result["status"], "completed_step": 300,
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": resume["sha256"],
    })
    return result


def make_token_field(context: Mapping[str, Any], phase: str) -> torch.nn.Module:
    config, model, base = context["config"], context["model"], context["base"]
    phase_config = config[phase]
    decoder = model.support_conditioned_residual_decoder_v7
    common = {
        "static_support_descriptor": decoder.static_support_descriptor,
        "gaussian_anchor_indices": decoder.gaussian_anchor_indices,
        "gaussian_anchor_weights": decoder.gaussian_anchor_weights,
        "garment_embedding_dim": int(phase_config["garment_embedding_dim"]),
        "token_dim": int(phase_config["token_dim"]),
        "hidden_dim": int(phase_config["hidden_dim"]),
        "num_blocks": int(phase_config["num_blocks"]),
        "output_shapes": residual_output_shapes(base),
        "channel_bounds": bounds(config),
        "default_chunk_size": int(phase_config["evaluation_chunk_size"]),
    }
    field = GaussianSupportTokenResidualField(**common) if phase == "p1" else AnchorSupportTokenResidualField(
        anchor_count=int(config["base"]["anchor_count"]), **common,
    )
    return field.to(base._xyz)


def token_prediction(
    field: torch.nn.Module,
    garment_embedding: torch.Tensor,
    protected_mask: torch.Tensor,
    *,
    gaussian_indices: torch.Tensor | None = None,
    chunk_size: int | None = None,
) -> GaussianClothingResiduals:
    prediction = field(
        garment_embedding, gaussian_indices=gaussian_indices, chunk_size=chunk_size,
    ).gated_gaussian_residuals
    selected_protected = protected_mask if gaussian_indices is None else protected_mask.index_select(0, gaussian_indices)
    return apply_protected_full_residual_guard(prediction, selected_protected)


def token_trainable_groups(field: torch.nn.Module, latents: DiagnosticOutfitLatents) -> dict[str, list[torch.nn.Parameter]]:
    return {
        "diagnostic_latents": list(latents.parameters()),
        "support_tokens": [field.support_tokens],
        "garment_projection": list(field.garment_to_token.parameters()),
        "geometry_branch": list(field.geometry_input.parameters()) + list(field.geometry_blocks.parameters()) + list(field.geometry_condition_reinject.parameters()),
        "appearance_branch": list(field.appearance_input.parameters()) + list(field.appearance_blocks.parameters()) + list(field.appearance_condition_reinject.parameters()),
        "head_xyz": list(field.xyz_head.parameters()), "head_scaling": list(field.scaling_head.parameters()),
        "head_rotation": list(field.rotation_head.parameters()), "head_opacity": list(field.opacity_head.parameters()),
        "head_sh0": list(field.sh0_head.parameters()), "head_shN": list(field.shN_head.parameters()),
    }


def token_optimizer(field: torch.nn.Module, latents: DiagnosticOutfitLatents, phase_config: Mapping[str, Any]) -> torch.optim.Optimizer:
    network = [parameter for name, parameter in field.named_parameters() if name != "support_tokens"]
    return torch.optim.Adam([
        {"name": "support_tokens", "params": [field.support_tokens], "lr": float(phase_config["token_learning_rate"])},
        {"name": "diagnostic_latents", "params": list(latents.parameters()), "lr": float(phase_config["latent_learning_rate"])},
        {"name": "field_network", "params": network, "lr": float(phase_config["network_learning_rate"])},
    ])


def run_token_phase(context: dict[str, Any], phase: str) -> dict[str, Any]:
    config, output_dir, base = context["config"], context["output_dir"], context["base"]
    phase_config = config[phase]
    seed = int(phase_config["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    field = make_token_field(context, phase)
    latents = DiagnosticOutfitLatents(OUTFITS, int(phase_config["garment_embedding_dim"]), seed).to(base._xyz)
    optimizer = token_optimizer(field, latents, phase_config)
    groups = token_trainable_groups(field, latents)
    phase_dir = output_dir / phase; phase_dir.mkdir(parents=True, exist_ok=True)

    def predict_all() -> dict[str, GaussianClothingResiduals]:
        return {
            outfit: token_prediction(
                field, latents(outfit), context["protected_mask"],
                chunk_size=int(phase_config["evaluation_chunk_size"]),
            ) for outfit in OUTFITS
        }

    field.eval(); latents.eval()
    with torch.no_grad():
        initial_predictions = predict_all()
    initial, _ = evaluate_predictions(
        phase=phase, step=0, predictions=initial_predictions, targets=context["targets"], base=base,
        samples=context["samples"], background=context["background"], config=config, output_dir=output_dir,
    )
    field.train(); latents.train()
    history, milestones = [], {"0": initial}
    final_gradients = {}
    count, chunk = field.gaussian_count, int(phase_config["training_chunk_size"])
    generator = torch.Generator(device=base._xyz.device).manual_seed(seed + 1)
    started = time.time(); torch.cuda.reset_peak_memory_stats()
    for step in range(1, int(phase_config["max_steps"]) + 1):
        outfit = OUTFITS[(step - 1) % 2]
        indices = torch.randint(count, (chunk,), device=base._xyz.device, generator=generator)
        optimizer.zero_grad(set_to_none=True)
        prediction = token_prediction(
            field, latents(outfit), context["protected_mask"], gaussian_indices=indices, chunk_size=chunk,
        )
        target = slice_residuals(context["targets"][outfit], indices)
        loss, parts = normalized_residual_regression_loss(prediction, target, bounds(config))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{phase} loss is NaN or Inf")
        loss.backward()
        final_gradients = trainable_snapshot(groups)
        torch.nn.utils.clip_grad_norm_(
            list(field.parameters()) + list(latents.parameters()),
            float(phase_config["gradient_clip_norm"]), error_if_nonfinite=True,
        )
        optimizer.step()
        row = {"step": step, "outfit": outfit, "loss": float(loss.detach()), **{
            f"loss_{name}": float(value.detach()) for name, value in parts.items()
        }}
        history.append(row); append_jsonl(phase_dir / "training.jsonl", row)
        if step in phase_config["milestones"]:
            field.eval(); latents.eval()
            with torch.no_grad():
                predictions = predict_all()
            milestone, _ = evaluate_predictions(
                phase=phase, step=step, predictions=predictions, targets=context["targets"], base=base,
                samples=context["samples"], background=context["background"], config=config, output_dir=output_dir,
            )
            milestones[str(step)] = milestone
            atomic_json(phase_dir / "partial_status.json", {"status": "RUNNING", "completed_step": step})
            field.train(); latents.train()
    elapsed = time.time() - started
    final = milestones[str(phase_config["max_steps"])]
    checkpoint_path = phase_dir / "checkpoints" / "checkpoint_step_001000.pth"
    save_checkpoint(
        checkpoint_path, phase=phase.upper(), step=1000,
        model_state={"field": field.state_dict(), "diagnostic_latents": latents.state_dict()},
        optimizer=optimizer, extra={"next_outfit": "O01", "outfit_update_counts": {"O01": 500, "O08": 500}},
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    restored_field = make_token_field(context, phase)
    restored_latents = DiagnosticOutfitLatents(OUTFITS, int(phase_config["garment_embedding_dim"]), seed).to(base._xyz)
    restored_field.load_state_dict(checkpoint["model"]["field"], strict=True)
    restored_latents.load_state_dict(checkpoint["model"]["diagnostic_latents"], strict=True)
    restored_optimizer = token_optimizer(restored_field, restored_latents, phase_config)
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    fixed = torch.arange(256, device=base._xyz.device)
    field.eval(); latents.eval(); restored_field.eval(); restored_latents.eval()
    with torch.no_grad():
        current_output = token_prediction(field, latents("O01"), context["protected_mask"], gaussian_indices=fixed, chunk_size=127)
        restored_output = token_prediction(restored_field, restored_latents("O01"), context["protected_mask"], gaussian_indices=fixed, chunk_size=127)
        chunked = token_prediction(field, latents("O01"), context["protected_mask"], gaussian_indices=fixed, chunk_size=37)
        non_chunked = token_prediction(field, latents("O01"), context["protected_mask"], gaussian_indices=fixed, chunk_size=1000)
    output_exact = all(torch.equal(getattr(current_output, name), getattr(restored_output, name)) for name in CHANNELS)
    chunk_parity = all(torch.allclose(getattr(chunked, name), getattr(non_chunked, name), atol=1e-7, rtol=1e-6) for name in CHANNELS)
    current_rng = rng_state(); o01._restore_rng(checkpoint["rng"])
    rng_exact = o01.object_fingerprint(rng_state()) == o01.object_fingerprint(checkpoint["rng"])
    o01._restore_rng(current_rng)
    resume = {
        "checkpoint": str(checkpoint_path), "sha256": sha256(checkpoint_path),
        "global_step_exact": int(checkpoint["global_step"]) == 1000,
        "field_state_exact": v7._tensor_state_fingerprint(field.state_dict().items()) == v7._tensor_state_fingerprint(restored_field.state_dict().items()),
        "latent_state_exact": v7._tensor_state_fingerprint(latents.state_dict().items()) == v7._tensor_state_fingerprint(restored_latents.state_dict().items()),
        "optimizer_state_exact": o01.object_fingerprint(optimizer.state_dict()) == o01.object_fingerprint(restored_optimizer.state_dict()),
        "rng_state_exact": rng_exact, "fixed_output_bitwise_exact": output_exact,
    }
    resume["pass"] = all(value for name, value in resume.items() if name not in {"checkpoint", "sha256"})
    acceptance = phase_config["acceptance"]
    checks = {
        "mean_normalized_rmse": final["mean"]["normalized_rmse"] <= float(acceptance["mean_bound_normalized_rmse_max"]),
        "mean_direction_cosine": final["mean"]["direction_cosine"] >= float(acceptance["mean_direction_cosine_min"]),
        "mean_top_10pct_overlap": final["mean"]["top_10pct_overlap"] >= float(acceptance["mean_top_10pct_overlap_min"]),
        "mean_top_20pct_overlap": final["mean"]["top_20pct_overlap"] >= float(acceptance["mean_top_20pct_overlap_min"]),
        "outfit_separation": final["outfit_separation_ratio"] >= float(acceptance["outfit_separation_ratio_min"]),
        "o01_render_garment_mae": final["render_per_outfit_garment_mae"]["O01"] <= float(acceptance["per_outfit_render_garment_mae_max"]),
        "o08_render_garment_mae": final["render_per_outfit_garment_mae"]["O08"] <= float(acceptance["per_outfit_render_garment_mae_max"]),
        "support_token_gradient_finite_nonzero": final_gradients["support_tokens"]["gradient_finite"] and final_gradients["support_tokens"]["gradient_l2"] > 0,
        "chunked_non_chunked_parity": chunk_parity,
        "base_bitwise_frozen": context["base_before"] == v7._tensor_state_fingerprint(v7._base_named_tensors(base)),
        "base_gradient_zero": v7._base_gradient_count(base) == 0,
        "checkpoint_resume": resume["pass"],
        "target_images_entered_forward": False,
    }
    numeric_pass = all(checks.values())
    result = {
        "status": "NUMERIC_PASS_VISUAL_PENDING" if numeric_pass else "FAIL",
        "optimizer_steps": 1000, "milestones": milestones, "final": final,
        "checks": checks, "gradients": final_gradients, "runtime_seconds": elapsed,
        "throughput_steps_per_second": 1000.0 / max(elapsed, 1e-12),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "parameter_count": sum(parameter.numel() for parameter in field.parameters()) + sum(parameter.numel() for parameter in latents.parameters()),
        "field_parameter_count": sum(parameter.numel() for parameter in field.parameters()),
        "token_count": field.token_count, "token_dim": field.token_dim,
        "checkpoint_resume": resume,
        "visual_status": "PENDING_ACTUAL_INSPECTION" if numeric_pass else "NOT_ELIGIBLE_NUMERIC_FAIL",
    }
    atomic_json(phase_dir / f"{phase}_metrics.json", result)
    atomic_json(phase_dir / "partial_status.json", {
        "status": result["status"], "completed_step": 1000,
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": resume["sha256"],
    })
    return result


def _load_method_predictions(context: Mapping[str, Any], phase: str) -> dict[str, GaussianClothingResiduals]:
    base, config, output_dir = context["base"], context["config"], context["output_dir"]
    if phase == "p0":
        tables = torch.nn.ModuleDict({
            outfit: DirectPerGaussianResidualTableControl(residual_output_shapes(base), bounds(config)).to(base._xyz)
            for outfit in OUTFITS
        })
        checkpoint = torch.load(output_dir / "p0/checkpoints/checkpoint_step_000300.pth", map_location="cpu", weights_only=False)
        tables.load_state_dict(checkpoint["model"]["tables"], strict=True)
        with torch.no_grad():
            return p0_predictions(tables, context["protected_mask"])
    phase_config = config[phase]
    field = make_token_field(context, phase)
    latents = DiagnosticOutfitLatents(OUTFITS, int(phase_config["garment_embedding_dim"]), int(phase_config["seed"])).to(base._xyz)
    checkpoint = torch.load(output_dir / phase / "checkpoints/checkpoint_step_001000.pth", map_location="cpu", weights_only=False)
    field.load_state_dict(checkpoint["model"]["field"], strict=True)
    latents.load_state_dict(checkpoint["model"]["diagnostic_latents"], strict=True)
    field.eval(); latents.eval()
    with torch.no_grad():
        return {
            outfit: token_prediction(
                field, latents(outfit), context["protected_mask"],
                chunk_size=int(phase_config["evaluation_chunk_size"]),
            ) for outfit in OUTFITS
        }


def _render_cache(
    context: Mapping[str, Any],
    predictions: Mapping[str, GaussianClothingResiduals],
) -> dict[str, Any]:
    cache = {outfit: {} for outfit in OUTFITS}
    with torch.no_grad():
        for outfit in OUTFITS:
            for condition in CONDITIONS:
                sample = context["samples"][f"{outfit}/{condition}"]
                rgb, alpha = render_prediction(context["base"], sample, predictions[outfit], context["background"])
                cache[outfit][condition] = (rgb.detach().cpu(), alpha.detach().cpu())
    return cache


def build_unified_visuals(context: Mapping[str, Any]) -> None:
    import matplotlib.pyplot as plt

    output_dir = context["output_dir"]
    predictions = {
        "V7": load_v7_predictions(context),
        "P0": _load_method_predictions(context, "p0"),
        "P1": _load_method_predictions(context, "p1"),
        "P2": _load_method_predictions(context, "p2"),
    }
    caches = {name: _render_cache(context, value) for name, value in predictions.items()}
    oracle_cache = _render_cache(context, context["targets"])
    visual = output_dir / "visual_acceptance"; visual.mkdir(parents=True, exist_ok=True)
    for outfit in OUTFITS:
        rows = []
        for condition in CONDITIONS:
            sample = context["samples"][f"{outfit}/{condition}"]
            oracle_rgb = oracle_cache[outfit][condition][0]
            p1_rgb = caches["P1"][outfit][condition][0]
            p2_rgb = caches["P2"][outfit][condition][0]
            garment = diagnosis._garment_mask(sample)
            protected = sample["target_protected_mask"]
            rows.append((f"{outfit}/{VIEWS[condition]}", [
                ("base", sample["target_base_rgb"], 3),
                ("target", sample["target_edit_rgb"], 3),
                ("Oracle", oracle_rgb, 3),
                ("V7", caches["V7"][outfit][condition][0], 3),
                ("P0", caches["P0"][outfit][condition][0], 3),
                ("P1", p1_rgb, 3),
                ("P2", p2_rgb, 3),
                ("Oracle-P1 abs", (oracle_rgb - p1_rgb).abs(), 3),
                ("Oracle-P2 abs", (oracle_rgb - p2_rgb).abs(), 3),
                ("garment crop P2", v7.crop_tensor(p2_rgb, garment), 3),
                ("torso/sleeve P2", v7.crop_tensor(p2_rgb, garment, vertical="upper"), 3),
                ("shoes/protected P2", v7.crop_tensor(p2_rgb, protected, vertical="lower"), 3),
            ]))
        o01._save_contact_sheet(visual / f"parameterization_{outfit}_four_view_contact_sheet.png", rows)

        xyz = context["base"]._xyz.detach().float().cpu()
        sampled = torch.arange(0, xyz.shape[0], max(1, xyz.shape[0] // 80000))
        figure, axes = plt.subplots(4, 6, figsize=(25, 16), constrained_layout=True)
        for row_index, method in enumerate(("Oracle", "V7", "P1", "P2")):
            residual = context["targets"][outfit] if method == "Oracle" else predictions[method][outfit]
            for column, channel in enumerate(CHANNELS):
                magnitude = getattr(residual, channel).detach().float().cpu().reshape(xyz.shape[0], -1).norm(dim=1)
                plot = axes[row_index, column].scatter(xyz[sampled, 0], xyz[sampled, 2], c=magnitude[sampled], s=1, cmap="magma")
                axes[row_index, column].set_title(f"{method} {channel}")
                figure.colorbar(plot, ax=axes[row_index, column], fraction=0.046)
        for axis in axes.reshape(-1):
            axis.set_aspect("equal"); axis.set_xlabel("canonical x"); axis.set_ylabel("canonical z")
        figure.savefig(visual / f"parameterization_{outfit}_six_channel_support.png", dpi=110)
        plt.close(figure)

        figure, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
        target_magnitude = row_magnitude(context["targets"][outfit], bounds(context["config"])).cpu()
        count = max(1, int(round(xyz.shape[0] * 0.10)))
        target_top = torch.zeros(xyz.shape[0], dtype=torch.bool); target_top[torch.topk(target_magnitude, count).indices] = True
        for row_index, method in enumerate(("P1", "P2")):
            pred_magnitude = row_magnitude(predictions[method][outfit], bounds(context["config"])).cpu()
            pred_top = torch.zeros_like(target_top); pred_top[torch.topk(pred_magnitude, count).indices] = True
            panels = [
                (f"{method} top10", pred_top.float()),
                ("overlap", (pred_top & target_top).float()),
                ("false positive", (pred_top & ~target_top).float()),
                ("false negative", (~pred_top & target_top).float()),
            ]
            for column, (title, colors) in enumerate(panels):
                axes[row_index, column].scatter(xyz[sampled, 0], xyz[sampled, 2], c=colors[sampled], s=1, cmap="coolwarm", vmin=0, vmax=1)
                axes[row_index, column].set_title(title)
                axes[row_index, column].set_aspect("equal")
        figure.savefig(visual / f"parameterization_{outfit}_top10_support.png", dpi=120)
        plt.close(figure)


def _load_visual_observations(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("images_actually_opened") is not True:
        raise ValueError("visual observations must confirm images_actually_opened=true")
    if not payload.get("inspection_method") or not payload.get("images") or not payload.get("observations"):
        raise ValueError("visual observations require method, image list, and concrete observations")
    return payload


def approve_p0(output_dir: Path, observations_path: Path) -> dict[str, Any]:
    metrics = json.loads((output_dir / "p0/p0_metrics.json").read_text(encoding="utf-8"))
    observations = _load_visual_observations(observations_path)
    if observations.get("p0_status") not in {"PASS", "WARN", "FAIL"}:
        raise ValueError("P0 observations require p0_status")
    numeric_pass = metrics["status"] == "NUMERIC_PASS_VISUAL_PENDING"
    observations["p0_gate_pass"] = numeric_pass and observations["p0_status"] == "PASS"
    atomic_json(output_dir / "visual_acceptance/p0_visual_acceptance.json", observations)
    return observations


def _require_p0_pass(output_dir: Path) -> None:
    metrics = json.loads((output_dir / "p0/p0_metrics.json").read_text(encoding="utf-8"))
    visual_path = output_dir / "visual_acceptance/p0_visual_acceptance.json"
    if metrics["status"] != "NUMERIC_PASS_VISUAL_PENDING" or not visual_path.is_file():
        raise RuntimeError("P1/P2 require completed P0 numeric and visual acceptance")
    if json.loads(visual_path.read_text(encoding="utf-8")).get("p0_gate_pass") is not True:
        raise RuntimeError("P0 did not pass its visual gate")


def input_context(config: dict[str, Any], output_dir: Path, *, create: bool) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("formal parameterization closure requires CUDA")
    device = torch.device("cuda")
    seed = int(config["p0"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    branch, dirty = git_output("branch", "--show-current"), git_output("status", "--short")
    if branch != EXPECTED_BRANCH or dirty:
        raise RuntimeError(f"formal run requires clean {EXPECTED_BRANCH}; branch={branch} dirty={bool(dirty)}")
    paths = {
        "source_manifest": Path(config["inputs"]["source_manifest"]),
        "stable_protected_attribution": Path(config["inputs"]["stable_protected_attribution"]),
        "diagnosis_checkpoint": Path(config["inputs"]["diagnosis_checkpoint"]),
        "v7_checkpoint": Path(config["inputs"]["v7_checkpoint"]),
        "oracle_o01_checkpoint": Path(config["inputs"]["oracle_o01_checkpoint"]),
        "oracle_o08_checkpoint": Path(config["inputs"]["oracle_o08_checkpoint"]),
        "base_checkpoint": Path(config["base"]["model_dir"]) / config["base"]["checkpoint_path"],
        "lbs_grid": Path(config["base"]["lbs_grid_path"]),
    }
    expected = {
        "source_manifest": config["inputs"]["source_manifest_sha256"],
        "stable_protected_attribution": config["inputs"]["stable_protected_attribution_sha256"],
        "diagnosis_checkpoint": config["inputs"]["diagnosis_checkpoint_sha256"],
        "v7_checkpoint": config["inputs"]["v7_checkpoint_sha256"],
        "oracle_o01_checkpoint": config["inputs"]["oracle_o01_checkpoint_sha256"],
        "oracle_o08_checkpoint": config["inputs"]["oracle_o08_checkpoint_sha256"],
        "base_checkpoint": config["base"]["checkpoint_sha256"],
    }
    hashes = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[name] = sha256(path)
        if name in expected and hashes[name] != expected[name]:
            raise ValueError(f"immutable input SHA mismatch: {name}")
    immutable_attempts = {
        "o01_attempt": Path(config["inputs"]["o01_attempt"]),
        "diagnosis_attempt": Path(config["inputs"]["diagnosis_attempt"]),
        "v7_attempt": Path(config["inputs"]["v7_attempt"]),
    }
    immutable_fingerprints = {name: immutable_tree_metadata_fingerprint(path) for name, path in immutable_attempts.items()}
    renderer_sources = {
        "gaussian_model": PROJECT_ROOT / "scene/gaussian_model.py",
        "production_render_adapter": PROJECT_ROOT / "tools/run_module4b_canonical_oracle_micropilot.py",
        "registered_sh1_wrapper": PROJECT_ROOT / "tools/run_image_conditioned_overfit_o01.py",
    }
    renderer_fingerprints = {name: sha256(path) for name, path in renderer_sources.items()}
    if create:
        for name in ("contract", "input_audit", "visual_acceptance", "final_adjudication"):
            (output_dir / name).mkdir(parents=True)
        atomic_text(output_dir / "contract/config_resolved.yaml", yaml.safe_dump(config, sort_keys=False))
        atomic_text(output_dir / "contract/command.txt", " ".join([sys.executable, *sys.argv]))
        atomic_json(output_dir / "input_audit/environment.json", o01.environment_snapshot())
        atomic_json(output_dir / "input_audit/input_manifest.json", {
            "task_id": TASK_ID,
            "git": {"branch": branch, "commit": git_output("rev-parse", "HEAD"), "clean": True, "source_head": EXPECTED_SOURCE_HEAD},
            "paths": {name: str(path) for name, path in paths.items()}, "sha256": hashes,
            "immutable_attempt_paths": {name: str(path) for name, path in immutable_attempts.items()},
            "immutable_attempt_tree_metadata_fingerprints": immutable_fingerprints,
            "historical_checkpoint_warning": {
                "historical_sha256": config["base"]["historical_checkpoint_sha256_warning"],
                "current_sha256": config["base"]["checkpoint_sha256"], "same_file": False,
            },
            "teacher_usage": "detached_loss_and_offline_render_only",
            "target_images_in_prediction_forward": False,
            "renderer_source_paths": {name: str(path) for name, path in renderer_sources.items()},
            "renderer_source_sha256": renderer_fingerprints,
        })
    samples, episodes, protocol = diagnosis._generic_outfit_data(paths["source_manifest"])
    base = training.load_frozen_mmlphuman_base(config["base"]["model_dir"], paths["base_checkpoint"], device=device)
    if int(base._xyz.shape[0]) != int(config["base"]["gaussian_count"]):
        raise ValueError("base Gaussian count changed")
    protected_cpu, attribution = o01._load_stable_protected_mask(
        paths["stable_protected_attribution"], int(config["inputs"]["stable_protected_count"]), int(base._xyz.shape[0]),
    )
    protected_mask = protected_cpu.to(device)
    base_before = v7._tensor_state_fingerprint(v7._base_named_tensors(base))
    v7_config = v7.load_config(PROJECT_ROOT / "configs/research/subject02_residual_decoder_capacity_v7.yaml")
    model, graph = v7.construct_model(base, v7_config, device)
    set_requires_grad(model, False)
    backbone_before = o01._state_fingerprint(model.clothing_observation_encoder.backbone.state_dict())
    adapter = MMLPHumanAnchorDeformationAdapter.from_mmlphuman_base(base, lbs_grid_path=paths["lbs_grid"])
    background = torch.tensor(config["render"]["background"], device=device, dtype=base._xyz.dtype)
    for key in list(samples):
        samples[key] = diagnosis._to_device_nested(samples[key], device)
        episodes[key] = diagnosis._to_device_nested(episodes[key], device)
    for condition in CONDITIONS:
        training.prepare_real_reference_geometry(base, adapter, episodes[f"O01/{condition}"], background)
        for outfit in OUTFITS:
            key = f"{outfit}/{condition}"
            support, _ = o01._base_only_protected_support(
                base, samples[key], protected_mask, background,
                float(config["render"]["protected_support_alpha_threshold"]),
            )
            samples[key]["target_protected_mask"] = torch.maximum(samples[key]["target_protected_mask"], support)
    targets, oracle_meta = {}, {}
    for outfit in OUTFITS:
        oracle, oracle_meta[outfit] = diagnosis._load_oracle(base, paths[f"oracle_{outfit.lower()}_checkpoint"], device)
        targets[outfit] = apply_protected_full_residual_guard(oracle.residuals(base), protected_mask)
        if any(value.requires_grad or value.grad_fn is not None for value in targets[outfit].as_dict().values()):
            raise AssertionError("Oracle teacher retained an autograd graph")
    if create:
        atomic_json(output_dir / "input_audit/resolved_protocol.json", {
            "protocol": protocol, "graph": graph, "oracles": oracle_meta, "attribution": attribution,
            "support_descriptor": model.support_conditioned_residual_decoder_v7.descriptor_metadata,
            "p0_has_decoder": False,
            "p1_token_count": int(base._xyz.shape[0]),
            "p2_token_count": int(config["base"]["anchor_count"]),
            "token_dim": 24,
            "target_images_in_prediction_forward": False,
        })
    return {
        "config": config, "output_dir": output_dir, "base": base, "model": model,
        "samples": samples, "episodes": episodes, "targets": targets,
        "protected_mask": protected_mask, "background": background,
        "base_before": base_before, "backbone_before": backbone_before,
        "immutable_fingerprints": immutable_fingerprints,
        "renderer_sources": renderer_sources, "renderer_fingerprints": renderer_fingerprints,
    }


def finalize(context: Mapping[str, Any], observations_path: Path) -> dict[str, Any]:
    output_dir = context["output_dir"]
    p0 = json.loads((output_dir / "p0/p0_metrics.json").read_text(encoding="utf-8"))
    p1_path, p2_path = output_dir / "p1/p1_metrics.json", output_dir / "p2/p2_metrics.json"
    p1 = json.loads(p1_path.read_text(encoding="utf-8")) if p1_path.is_file() else {"status": "NOT_RUN_P0_FAILED", "optimizer_steps": 0}
    p2 = json.loads(p2_path.read_text(encoding="utf-8")) if p2_path.is_file() else {"status": "NOT_RUN_P0_FAILED", "optimizer_steps": 0}
    visual = _load_visual_observations(observations_path)
    for phase in ("p0", "p1", "p2"):
        if f"{phase}_status" not in visual:
            visual[f"{phase}_status"] = "NOT_RUN"
    p0_pass = p0["status"] == "NUMERIC_PASS_VISUAL_PENDING" and visual["p0_status"] == "PASS"
    p1_pass = p1["status"] == "NUMERIC_PASS_VISUAL_PENDING" and visual["p1_status"] == "PASS"
    p2_pass = p2["status"] == "NUMERIC_PASS_VISUAL_PENDING" and visual["p2_status"] == "PASS"
    if not p0_pass:
        case, next_task, allow_reference = "PF-0", "FIX_RESIDUAL_TEACHER_REGRESSION_CONTRACT", False
    elif p2_pass:
        case, next_task, allow_reference = "PF-A", "INTEGRATE_ANCHOR_TOKEN_FIELD_WITH_REFERENCE_CONDITIONING", True
    elif p1_pass:
        case, next_task, allow_reference = "PF-G", "INTEGRATE_GAUSSIAN_TOKEN_FIELD_WITH_REFERENCE_CONDITIONING", True
    else:
        case, next_task, allow_reference = "PF-B", "BUILD_EXPLICIT_LOW_RANK_GAUSSIAN_RESIDUAL_BASIS", False
    manifest = json.loads((output_dir / "input_audit/input_manifest.json").read_text(encoding="utf-8"))
    immutable_after = {
        name: immutable_tree_metadata_fingerprint(Path(path))
        for name, path in manifest["immutable_attempt_paths"].items()
    }
    freeze = {
        "base_before": context["base_before"],
        "base_after": v7._tensor_state_fingerprint(v7._base_named_tensors(context["base"])),
        "base_gradient_count": v7._base_gradient_count(context["base"]),
        "image_backbone_before": context["backbone_before"],
        "image_backbone_after": o01._state_fingerprint(context["model"].clothing_observation_encoder.backbone.state_dict()),
        "image_backbone_gradient_count": sum(parameter.grad is not None for parameter in context["model"].clothing_observation_encoder.backbone.parameters()),
        "legacy_mmlp_gradient_count": sum(parameter.grad is not None for parameter in context["model"].dressable_model.anchor_clothing_mlp.parameters()),
        "historical_outputs_immutable": immutable_after == manifest["immutable_attempt_tree_metadata_fingerprints"],
        "renderer_source_before": context["renderer_fingerprints"],
        "renderer_source_after": {name: sha256(path) for name, path in context["renderer_sources"].items()},
    }
    freeze["base_bitwise_unchanged"] = freeze["base_before"] == freeze["base_after"]
    freeze["image_backbone_bitwise_unchanged"] = freeze["image_backbone_before"] == freeze["image_backbone_after"]
    freeze["renderer_source_unchanged"] = freeze["renderer_source_before"] == freeze["renderer_source_after"]
    if (
        not freeze["base_bitwise_unchanged"]
        or not freeze["image_backbone_bitwise_unchanged"]
        or not freeze["historical_outputs_immutable"]
        or not freeze["renderer_source_unchanged"]
    ):
        raise AssertionError("frozen asset contract failed during finalization")
    result = {
        "task_id": TASK_ID,
        "run_commit": manifest["git"]["commit"],
        "finalization_commit": git_output("rev-parse", "HEAD"),
        "p0": p0, "p1": p1, "p2": p2,
        "visual_acceptance": visual, "freeze": freeze,
        "target_forward_leakage": False,
        "final_case": case, "status": "PASS" if case in {"PF-A", "PF-G"} else "FAIL",
        "real_reference_integration_allowed": allow_reference,
        "next_task": next_task,
    }
    atomic_json(output_dir / "visual_acceptance/visual_acceptance.json", visual)
    visual_lines = [
        f"# {TASK_ID} visual acceptance", "",
        f"- Images actually opened: `{visual['images_actually_opened']}`",
        f"- Inspection method: {visual['inspection_method']}",
        f"- P0: **{visual['p0_status']}**", f"- P1: **{visual['p1_status']}**", f"- P2: **{visual['p2_status']}**",
        "", "## Images", "", *[f"- `{item}`" for item in visual["images"]],
        "", "## Observations", "", *[f"- {item}" for item in visual["observations"]],
    ]
    atomic_text(output_dir / "visual_acceptance/VISUAL_ACCEPTANCE.md", "\n".join(visual_lines))
    atomic_json(output_dir / "final_adjudication/final_adjudication.json", result)
    atomic_text(output_dir / "final_adjudication/FINAL_ADJUDICATION.md", "\n".join([
        f"# {TASK_ID}", "", f"- Run commit: `{result['run_commit']}`",
        f"- Finalization commit: `{result['finalization_commit']}`",
        f"- P0: **{p0['status']} / visual {visual['p0_status']}**",
        f"- P1: **{p1['status']} / visual {visual['p1_status']}**",
        f"- P2: **{p2['status']} / visual {visual['p2_status']}**",
        f"- Final case: **{case}**", f"- Real-reference integration allowed: `{allow_reference}`",
        f"- Next unique task: `{next_task}`",
    ]))
    atomic_json(output_dir / "RUN_STATUS.json", {
        "task_id": TASK_ID, "status": "COMPLETE", "final_case": case,
        "optimizer_steps": int(p0["optimizer_steps"]) + int(p1["optimizer_steps"]) + int(p2["optimizer_steps"]),
        "completed_at": time.time(),
    })
    return result


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config.resolve())
    output_dir = Path(args.output_dir or Path(config["output"]["task_root"]) / config["output"]["attempt"])
    create = args.phase == "p0"
    if create:
        if output_dir.exists():
            raise FileExistsError(f"append-only parameterization attempt already exists: {output_dir}")
        output_dir.mkdir(parents=True)
        atomic_json(output_dir / "RUN_STATUS.json", {"task_id": TASK_ID, "status": "RUNNING", "phase": "input_audit", "optimizer_steps": 0})
    elif not output_dir.is_dir():
        raise FileNotFoundError(output_dir)
    context = input_context(config, output_dir, create=create)
    try:
        if args.phase == "p0":
            atomic_json(output_dir / "RUN_STATUS.json", {"task_id": TASK_ID, "status": "RUNNING", "phase": "P0", "optimizer_steps": 0})
            result = run_p0(context)
            atomic_json(output_dir / "RUN_STATUS.json", {"task_id": TASK_ID, "status": "AWAITING_P0_VISUAL_INSPECTION" if result["status"].startswith("NUMERIC_PASS") else "AWAITING_FINAL_ADJUDICATION", "phase": "P0_COMPLETE", "optimizer_steps": 300})
        elif args.phase == "p1":
            if args.visual_observations is None:
                raise ValueError("P1 requires --visual-observations for the actual P0 inspection")
            p0_visual = approve_p0(output_dir, args.visual_observations)
            if not p0_visual["p0_gate_pass"]:
                raise RuntimeError("P0 did not pass; P1/P2 are forbidden")
            atomic_json(output_dir / "RUN_STATUS.json", {"task_id": TASK_ID, "status": "RUNNING", "phase": "P1", "optimizer_steps": 300})
            result = run_token_phase(context, "p1")
            atomic_json(output_dir / "RUN_STATUS.json", {"task_id": TASK_ID, "status": "P1_COMPLETE_P2_REQUIRED", "phase": "P1_COMPLETE", "optimizer_steps": 1300, "p1_status": result["status"]})
        elif args.phase == "p2":
            _require_p0_pass(output_dir)
            if not (output_dir / "p1/p1_metrics.json").is_file():
                raise RuntimeError("P2 requires completed P1; P1 PASS is not required")
            atomic_json(output_dir / "RUN_STATUS.json", {"task_id": TASK_ID, "status": "RUNNING", "phase": "P2", "optimizer_steps": 1300})
            result = run_token_phase(context, "p2")
            build_unified_visuals(context)
            atomic_json(output_dir / "RUN_STATUS.json", {"task_id": TASK_ID, "status": "AWAITING_FINAL_VISUAL_INSPECTION", "phase": "P2_COMPLETE", "optimizer_steps": 2300, "p2_status": result["status"]})
        elif args.phase == "finalize":
            if args.visual_observations is None:
                raise ValueError("finalize requires --visual-observations")
            finalize(context, args.visual_observations)
        else:
            raise ValueError(args.phase)
    except Exception as error:
        atomic_json(output_dir / "RUN_STATUS.json", {
            "task_id": TASK_ID, "status": "FAILED_TOOL_OR_RUNTIME", "phase": args.phase,
            "exception_type": type(error).__name__, "exception": str(error), "traceback": traceback.format_exc(),
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the preregistered P0/P1/P2 residual-field capacity ladder")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("p0", "p1", "p2", "finalize"), required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--visual-observations", type=Path, default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
