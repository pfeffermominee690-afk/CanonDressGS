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
from typing import Any, Mapping

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
    compose_canonical_gaussian_overrides,
)
from scene.image_conditioned_failure_diagnostics import (  # noqa: E402
    DiagnosticOutfitLatents,
    assert_forward_boundary,
    channel_comparison,
    normalized_residual_regression_loss,
    reference_variant,
    residual_distance_scalar,
)
from scene.support_conditioned_dual_branch_residual_decoder_v7 import (  # noqa: E402
    SupportConditionedDualBranchResidualDecoderV7,
    V7_CAPACITY_GATE_MODE,
    V7_DECODER_TYPE,
)
from tools import diagnose_image_conditioned_overfit_failure as diagnosis  # noqa: E402
from tools import run_image_conditioned_overfit_o01 as o01  # noqa: E402
from tools.check_real_image_conditioned_one_batch import save_render_tensor  # noqa: E402
from tools.run_module4b_canonical_oracle_micropilot import (  # noqa: E402
    _base_gradient_count,
    _base_named_tensors,
    _tensor_state_fingerprint,
)
from utils.mmlphuman_anchor_deformation import MMLPHumanAnchorDeformationAdapter  # noqa: E402


SCHEMA = "canondressgs.residual_decoder_capacity_v7.v1"
TASK_ID = "SUBJECT02-RESIDUAL-DECODER-CAPACITY-V7-001"
EXPECTED_BRANCH = "research/residual-decoder-capacity-v7-20260720"
EXPECTED_SOURCE_HEAD = "b1dcc16f0d2a57f56ed04778792db8f11f92e081"
OUTFITS = ("O01", "O08")
CONDITIONS = o01.CONDITIONS
VIEWS = o01.VIEWS
BOUND_NAMES = {
    "delta_xyz": "xyz",
    "delta_log_scaling": "log_scaling",
    "delta_rotvec": "rotation",
    "delta_opacity_logit": "opacity_logit",
    "delta_sh0": "sh0",
    "delta_shN": "shN",
}


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
        raise ValueError("not the preregistered V7 decoder-capacity contract")
    if config.get("branch") != EXPECTED_BRANCH or config.get("source_head") != EXPECTED_SOURCE_HEAD:
        raise ValueError("V7 branch/source contract changed")
    if tuple(config.get("outfits", ())) != OUTFITS or tuple(config.get("conditions", ())) != CONDITIONS:
        raise ValueError("V7 two-outfit/four-view protocol changed")
    if config["model"]["decoder"]["type"] != V7_DECODER_TYPE:
        raise ValueError("V7 decoder selector changed")
    if config["stage_a"]["gate_mode"] != V7_CAPACITY_GATE_MODE:
        raise ValueError("Stage A must use a fixed-one diagnostic gate")
    if int(config["stage_a"]["max_steps"]) > 1000 or int(config["stage_b"]["max_steps"]) > 1000:
        raise ValueError("V7 probe exceeds the 1000-step ceiling")
    expected_milestones = [0, 100, 300, 600, 1000]
    if config["stage_a"]["milestones"] != expected_milestones or config["stage_b"]["milestones"] != expected_milestones:
        raise ValueError("V7 milestone contract changed")
    if any(bool(value) for value in config["permissions"].values()):
        raise ValueError("V7 contract enabled a forbidden mutation")
    frozen_acceptance = {
        "first_100_to_final_loss_drop_fraction_min": 0.80,
        "mean_bound_normalized_rmse_max": 0.15,
        "mean_direction_cosine_min": 0.65,
        "mean_top_10pct_overlap_min": 0.40,
        "outfit_separation_ratio_min": 0.50,
        "per_outfit_render_garment_mae_max": 0.08,
    }
    for name, expected in frozen_acceptance.items():
        if float(config["stage_a"]["acceptance"][name]) != expected:
            raise ValueError(f"Stage A acceptance threshold changed: {name}")
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


def model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    model = config["model"]
    return {
        "base": {"require_real_base": True},
        "model": {
            "offset_mode": "anchor_film",
            "hidden_dim": int(model["hidden_dim"]),
            "num_layers": int(model["num_layers"]),
            "hyper_hidden_dim": int(model["hyper_hidden_dim"]),
            "anchor_num_frequencies": int(model["anchor_num_frequencies"]),
            "anchor_graph_k": int(model["anchor_graph_k"]),
            "enable_delta_xyz": True,
            "enable_delta_scaling": True,
            "enable_delta_opacity": True,
            "decoder": dict(model["decoder"]),
            "dressable_channels": dict(model["dressable_channels"]),
        },
        "image_conditioning": {
            "embedding_dim": int(model["embedding_dim"]),
            "feature_dim": int(model["image_feature_dim"]),
            "local_feature_dim": int(model["local_feature_dim"]),
            "freeze_backbone": bool(model["freeze_image_backbone"]),
        },
        "optimizer": dict(config["stage_b"]["optimizer"]),
    }


def construct_model(base: Any, config: Mapping[str, Any], device: torch.device):
    model, _, _, _ = training.create_image_conditioned_components(
        o01.DatasetModelContract(), model_config(config), base_model=base, device=device,
    )
    graph_config = config["model"]["online_completion"]
    graph_indices, graph_weights, graph_diagnostics = o01.build_surface_aware_anchor_graph(
        base.xyz_vt.to(device), int(graph_config["graph_k"]), base.nbr_vt.to(device),
    )
    model.initialize_online_completion(
        graph_indices,
        graph_weights,
        hidden_dim=int(graph_config["hidden_dim"]),
        num_blocks=int(graph_config["blocks"]),
    )
    if model.residual_decoder_type != V7_DECODER_TYPE or model.support_conditioned_residual_decoder_v7 is None:
        raise RuntimeError("formal model did not resolve the V7 decoder")
    return model, graph_diagnostics


def set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def slice_residuals(residuals: GaussianClothingResiduals, indices: torch.Tensor) -> GaussianClothingResiduals:
    return GaussianClothingResiduals(**{
        name: value.index_select(0, indices) for name, value in residuals.as_dict().items()
    })


def prediction_from_latent(
    decoder: SupportConditionedDualBranchResidualDecoderV7,
    latent: torch.Tensor,
    protected_mask: torch.Tensor,
    *,
    gaussian_indices: torch.Tensor | None = None,
    chunk_size: int | None = None,
) -> GaussianClothingResiduals:
    local = latent.expand(int(decoder.gaussian_anchor_indices.max()) + 1, -1)
    prediction = decoder(
        latent,
        local,
        gaussian_indices=gaussian_indices,
        chunk_size=chunk_size,
        gate_mode=V7_CAPACITY_GATE_MODE,
    ).gated_gaussian_residuals
    if gaussian_indices is None:
        return apply_protected_full_residual_guard(prediction, protected_mask)
    selected_protected = protected_mask.index_select(0, gaussian_indices)
    return apply_protected_full_residual_guard(prediction, selected_protected)


def row_magnitude(residuals: GaussianClothingResiduals, channel_bounds: Mapping[str, float]) -> torch.Tensor:
    values = []
    for name, value in residuals.as_dict().items():
        values.append((value.detach().float().reshape(value.shape[0], -1) / float(channel_bounds[BOUND_NAMES[name]])).square().mean(1))
    return torch.stack(values, dim=1).mean(1).sqrt()


def residual_metrics(
    prediction: GaussianClothingResiduals,
    target: GaussianClothingResiduals,
    channel_bounds: Mapping[str, float],
) -> dict[str, Any]:
    comparison_10 = channel_comparison(prediction, target, channel_bounds, top_fraction=0.10)
    comparison_20 = channel_comparison(prediction, target, channel_bounds, top_fraction=0.20)
    pred_magnitude, target_magnitude = row_magnitude(prediction, channel_bounds), row_magnitude(target, channel_bounds)
    epsilon = 1e-5
    predicted_active, target_active = pred_magnitude > epsilon, target_magnitude > epsilon
    true_positive = int((predicted_active & target_active).sum())
    predicted_count, target_count = int(predicted_active.sum()), int(target_active.sum())
    top_count = max(1, int(round(pred_magnitude.shape[0] * 0.10)))
    pred_top = torch.topk(pred_magnitude, top_count).indices
    target_top = torch.topk(target_magnitude, top_count).indices
    target_top_mask = torch.zeros_like(target_active); target_top_mask[target_top] = True
    pred_top_mask = torch.zeros_like(predicted_active); pred_top_mask[pred_top] = True
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
        "spatial_concentration_in_oracle_top10": float(pred_magnitude[target_top_mask].sum() / pred_magnitude.sum().clamp_min(1e-12)),
        "top10_true_positive_count": int(overlap.sum()),
        "top10_false_positive_count": int((pred_top_mask & ~target_top_mask).sum()),
        "top10_false_negative_count": int((~pred_top_mask & target_top_mask).sum()),
        "per_attribute": comparison_10,
    }


def mean_metric(per_outfit: Mapping[str, Mapping[str, Any]], name: str) -> float:
    return float(np.mean([float(per_outfit[outfit][name]) for outfit in OUTFITS]))


def residual_bundle_cpu(residuals: GaussianClothingResiduals) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in residuals.as_dict().items()}


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
    stage: str,
    step: int,
    model_state: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    extra: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "schema_version": SCHEMA,
        "task_id": TASK_ID,
        "stage": stage,
        "global_step": int(step),
        "model": dict(model_state),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "rng": rng_state(),
        "target_tensors_stored": False,
        "oracle_residual_in_forward": False,
        "extra": dict(extra),
    }, temporary)
    os.replace(temporary, path)


def masked_mae(first: torch.Tensor, second: torch.Tensor, mask: torch.Tensor) -> float:
    return o01._masked_mae(first, second, mask)


def garment_mask(sample: Mapping[str, Any]) -> torch.Tensor:
    return diagnosis._garment_mask(sample)


def render_prediction(
    base: Any,
    sample: Mapping[str, Any],
    residuals: GaussianClothingResiduals,
    background: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    overrides = compose_canonical_gaussian_overrides(base, residuals, CHANNELS)
    return o01._render_sh1(base, sample, overrides, background)


def render_metrics(
    base: Any,
    samples: Mapping[str, Any],
    predictions: Mapping[str, GaussianClothingResiduals],
    targets: Mapping[str, GaussianClothingResiduals],
    background: torch.Tensor,
    output_dir: Path,
    step: int,
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]]]:
    directory = output_dir / "stage_a" / "milestones" / f"step_{step:06d}" / "renders"
    directory.mkdir(parents=True, exist_ok=True)
    rows, cached = [], {}
    with torch.no_grad():
        for outfit in OUTFITS:
            cached[outfit] = {}
            for condition in CONDITIONS:
                sample = samples[f"{outfit}/{condition}"]
                predicted_rgb, predicted_alpha = render_prediction(base, sample, predictions[outfit], background)
                oracle_rgb, oracle_alpha = render_prediction(base, sample, targets[outfit], background)
                mask = garment_mask(sample)
                protected = sample["target_protected_mask"]
                background_mask = 1 - torch.maximum(sample["target_foreground_mask"], sample["target_base_foreground_mask"])
                row = {
                    "outfit": outfit,
                    "condition": condition,
                    "view": VIEWS[condition],
                    "garment_rgb_mae_to_oracle": masked_mae(predicted_rgb, oracle_rgb, mask),
                    "protected_rgb_mae_to_oracle": masked_mae(predicted_rgb, oracle_rgb, protected),
                    "background_rgb_mae_to_oracle": masked_mae(predicted_rgb, oracle_rgb, background_mask),
                    "alpha_mae_to_oracle": float((predicted_alpha - oracle_alpha).abs().mean()),
                }
                rows.append(row)
                cached[outfit][condition] = {
                    "prediction": (predicted_rgb.detach().cpu(), predicted_alpha.detach().cpu()),
                    "oracle": (oracle_rgb.detach().cpu(), oracle_alpha.detach().cpu()),
                }
                save_render_tensor(directory / f"{outfit}_{condition}_predicted_rgb.png", predicted_rgb, 3)
                save_render_tensor(directory / f"{outfit}_{condition}_predicted_alpha.png", predicted_alpha, 1)
                save_render_tensor(directory / f"{outfit}_{condition}_oracle_rgb.png", oracle_rgb, 3)
    per_outfit = {
        outfit: float(np.mean([row["garment_rgb_mae_to_oracle"] for row in rows if row["outfit"] == outfit]))
        for outfit in OUTFITS
    }
    atomic_json(directory.parent / "render_metrics.json", {"rows": rows, "per_outfit_garment_mae": per_outfit})
    return rows, per_outfit, cached


def load_legacy_predictions(
    base: Any,
    protected_mask: torch.Tensor,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, GaussianClothingResiduals]:
    legacy_config = diagnosis._load_config(PROJECT_ROOT / "configs/research/subject02_image_conditioned_failure_diagnosis_v1.yaml")
    legacy_model, _, _, _, _ = o01._construct_model(base, legacy_config, device)
    checkpoint = torch.load(Path(config["inputs"]["diagnosis_checkpoint"]), map_location="cpu", weights_only=False)
    legacy_model.load_state_dict(checkpoint["model"], strict=True)
    latent = DiagnosticOutfitLatents(OUTFITS, int(legacy_config["model"]["embedding_dim"]), int(legacy_config["diagnosis"]["seed"])).to(base._xyz)
    latent.load_state_dict(checkpoint["diagnostic_latents"], strict=True)
    with torch.no_grad():
        predictions = {
            outfit: diagnosis._predict_from_latent(legacy_model, latent(outfit), protected_mask)[0]
            for outfit in OUTFITS
        }
    return predictions


def crop_tensor(tensor: torch.Tensor, mask: torch.Tensor, *, vertical: str = "all") -> torch.Tensor:
    value = tensor.detach().cpu()
    active = mask.detach().cpu().squeeze() >= 0.5
    points = torch.nonzero(active, as_tuple=False)
    if points.numel() == 0:
        return value
    y0, x0 = points.amin(dim=0).tolist(); y1, x1 = points.amax(dim=0).tolist()
    padding = max(4, int(0.05 * max(y1 - y0 + 1, x1 - x0 + 1)))
    y0, x0 = max(0, y0 - padding), max(0, x0 - padding)
    y1, x1 = min(active.shape[0] - 1, y1 + padding), min(active.shape[1] - 1, x1 + padding)
    height = y1 - y0 + 1
    if vertical == "upper":
        y1 = y0 + max(1, int(height * 0.70))
    elif vertical == "lower":
        y0 = y0 + int(height * 0.60)
    return value[:, y0:y1 + 1, x0:x1 + 1]


def build_stage_a_contact_sheets(
    samples: Mapping[str, Any],
    step0_renders: Mapping[str, Mapping[str, tuple[torch.Tensor, torch.Tensor]]],
    final_renders: Mapping[str, Mapping[str, tuple[torch.Tensor, torch.Tensor]]],
    legacy_renders: Mapping[str, Mapping[str, tuple[torch.Tensor, torch.Tensor]]],
    output_dir: Path,
) -> None:
    visual = output_dir / "visual_acceptance"
    for outfit in OUTFITS:
        rows = []
        for condition in CONDITIONS:
            sample = samples[f"{outfit}/{condition}"]
            step0_rgb = step0_renders[outfit][condition]["prediction"][0]
            final_rgb = final_renders[outfit][condition]["prediction"][0]
            oracle_rgb = final_renders[outfit][condition]["oracle"][0]
            legacy_rgb = legacy_renders[outfit][condition]["prediction"][0]
            mask = garment_mask(sample)
            protected = sample["target_protected_mask"]
            rows.append((f"{outfit}/{VIEWS[condition]}", [
                ("base", sample["target_base_rgb"], 3),
                ("target", sample["target_edit_rgb"], 3),
                ("Oracle", oracle_rgb, 3),
                ("legacy", legacy_rgb, 3),
                ("V7 step0", step0_rgb, 3),
                ("V7 final", final_rgb, 3),
                ("Oracle-V7 abs", (oracle_rgb - final_rgb).abs(), 3),
                ("garment crop", crop_tensor(final_rgb, mask), 3),
                ("torso/sleeve", crop_tensor(final_rgb, mask, vertical="upper"), 3),
                ("shoes/protected", crop_tensor(final_rgb, protected, vertical="lower"), 3),
            ]))
        o01._save_contact_sheet(visual / f"stage_a_{outfit}_four_view_contact_sheet.png", rows)


def build_support_visualizations(
    base: Any,
    predictions: Mapping[str, GaussianClothingResiduals],
    targets: Mapping[str, GaussianClothingResiduals],
    channel_bounds: Mapping[str, float],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    xyz = base._xyz.detach().float().cpu()
    stride = max(1, xyz.shape[0] // 100000)
    sampled = torch.arange(0, xyz.shape[0], stride)
    for outfit in OUTFITS:
        prediction, target = predictions[outfit], targets[outfit]
        figure, axes = plt.subplots(2, 5, figsize=(24, 9), constrained_layout=True)
        for axis, name in zip(axes.reshape(-1)[:6], CHANNELS):
            value = getattr(prediction, name).detach().float().cpu().reshape(xyz.shape[0], -1).norm(dim=1)
            plot = axis.scatter(xyz[sampled, 0], xyz[sampled, 2], c=value[sampled], s=1, cmap="magma")
            axis.set_title(f"V7 {name}"); figure.colorbar(plot, ax=axis, fraction=0.046)
        pred_mag, target_mag = row_magnitude(prediction, channel_bounds).cpu(), row_magnitude(target, channel_bounds).cpu()
        count = max(1, int(round(xyz.shape[0] * .10)))
        pred_top = torch.zeros(xyz.shape[0], dtype=torch.bool); pred_top[torch.topk(pred_mag, count).indices] = True
        target_top = torch.zeros_like(pred_top); target_top[torch.topk(target_mag, count).indices] = True
        support_panels = [
            ("Oracle top10", target_top), ("V7 top10", pred_top),
            ("overlap", pred_top & target_top),
            ("FP red / FN blue", pred_top ^ target_top),
        ]
        for axis, (title, active) in zip(axes.reshape(-1)[6:], support_panels):
            color = active.float()
            if title.startswith("FP"):
                color = pred_top.float() - target_top.float()
            axis.scatter(xyz[sampled, 0], xyz[sampled, 2], c=color[sampled], s=1, cmap="coolwarm", vmin=-1, vmax=1)
            axis.set_title(title)
        for axis in axes.reshape(-1):
            axis.set_aspect("equal"); axis.set_xlabel("canonical x"); axis.set_ylabel("canonical z")
        path = output_dir / "visual_acceptance" / f"stage_a_{outfit}_residual_support.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=120); plt.close(figure)


def stage_a_evaluation(
    *,
    step: int,
    decoder: SupportConditionedDualBranchResidualDecoderV7,
    latents: DiagnosticOutfitLatents,
    targets: Mapping[str, GaussianClothingResiduals],
    protected_mask: torch.Tensor,
    base: Any,
    samples: Mapping[str, Any],
    background: torch.Tensor,
    config: Mapping[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, GaussianClothingResiduals], dict[str, Any]]:
    decoder.eval(); latents.eval()
    channel_bounds = bounds(config)
    with torch.no_grad():
        predictions = {
            outfit: prediction_from_latent(
                decoder, latents(outfit), protected_mask,
                chunk_size=int(config["stage_a"]["evaluation_chunk_size"]),
            )
            for outfit in OUTFITS
        }
        per_outfit = {outfit: residual_metrics(predictions[outfit], targets[outfit], channel_bounds) for outfit in OUTFITS}
        predicted_separation = residual_distance_scalar(channel_comparison(predictions["O01"], predictions["O08"], channel_bounds))
        oracle_separation = residual_distance_scalar(channel_comparison(targets["O01"], targets["O08"], channel_bounds))
        render_rows, render_per_outfit, render_cache = render_metrics(
            base, samples, predictions, targets, background, output_dir, step,
        )
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
        "render_rows": render_rows,
        "render_per_outfit_garment_mae": render_per_outfit,
    }
    milestone = output_dir / "stage_a" / "milestones" / f"step_{step:06d}"
    atomic_json(milestone / "metrics.json", report)
    torch.save({outfit: residual_bundle_cpu(value) for outfit, value in predictions.items()}, milestone / "predicted_residuals.pt")
    decoder.train(); latents.train()
    return report, predictions, render_cache


def trainable_snapshot(groups: Mapping[str, list[torch.nn.Parameter]]) -> dict[str, Any]:
    result = {}
    for name, parameters in groups.items():
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        result[name] = {
            "parameter_count": sum(parameter.numel() for parameter in parameters),
            "gradient_tensor_count": len(gradients),
            "gradient_finite": bool(gradients) and all(torch.isfinite(value).all().item() for value in gradients),
            "gradient_l2": float(torch.sqrt(sum((value.detach().float().square().sum() for value in gradients), torch.tensor(0.0, device=parameters[0].device)))) if gradients else 0.0,
        }
    return result


def run_stage_a(context: dict[str, Any]) -> dict[str, Any]:
    config, output_dir, base = context["config"], context["output_dir"], context["base"]
    model, targets = context["model"], context["targets"]
    protected_mask, samples, background = context["protected_mask"], context["samples"], context["background"]
    decoder = model.support_conditioned_residual_decoder_v7
    if decoder is None:
        raise RuntimeError("Stage A requires V7")
    set_requires_grad(model, False); set_requires_grad(decoder, True)
    latents = DiagnosticOutfitLatents(
        OUTFITS, int(config["stage_a"]["diagnostic_latent_dim"]), int(config["stage_a"]["seed"]),
    ).to(base._xyz)
    optimizer = torch.optim.Adam([
        {"name": "diagnostic_latents", "params": list(latents.parameters()), "lr": float(config["stage_a"]["latent_learning_rate"])},
        {"name": "v7_decoder", "params": list(decoder.parameters()), "lr": float(config["stage_a"]["decoder_learning_rate"])},
    ])
    groups = {
        "diagnostic_latents": list(latents.parameters()),
        "support_encoder": list(decoder.support_encoder.parameters()),
        "geometry_branch": list(decoder.geometry_input.parameters()) + list(decoder.geometry_blocks.parameters()) + list(decoder.geometry_condition_reinject.parameters()),
        "appearance_branch": list(decoder.appearance_input.parameters()) + list(decoder.appearance_blocks.parameters()) + list(decoder.appearance_condition_reinject.parameters()),
        "head_xyz": list(decoder.xyz_head.parameters()), "head_scaling": list(decoder.scaling_head.parameters()),
        "head_rotation": list(decoder.rotation_head.parameters()), "head_opacity": list(decoder.opacity_head.parameters()),
        "head_sh0": list(decoder.sh0_head.parameters()), "head_shN": list(decoder.shN_head.parameters()),
    }
    stage_dir = output_dir / "stage_a"; stage_dir.mkdir(parents=True, exist_ok=True)
    initial, _, step0_renders = stage_a_evaluation(
        step=0, decoder=decoder, latents=latents, targets=targets, protected_mask=protected_mask,
        base=base, samples=samples, background=background, config=config, output_dir=output_dir,
    )
    history, milestones = [], {"0": initial}
    final_gradients: dict[str, Any] = {}
    count = decoder.gaussian_count
    chunk = int(config["stage_a"]["training_chunk_size"])
    generator = torch.Generator(device=base._xyz.device).manual_seed(int(config["stage_a"]["seed"]) + 1)
    started = time.time(); torch.cuda.reset_peak_memory_stats()
    for step in range(1, int(config["stage_a"]["max_steps"]) + 1):
        outfit = OUTFITS[(step - 1) % 2]
        indices = torch.randint(count, (chunk,), device=base._xyz.device, generator=generator)
        optimizer.zero_grad(set_to_none=True)
        prediction = prediction_from_latent(decoder, latents(outfit), protected_mask, gaussian_indices=indices, chunk_size=chunk)
        target = slice_residuals(targets[outfit], indices)
        loss, parts = normalized_residual_regression_loss(prediction, target, bounds(config))
        if not torch.isfinite(loss):
            raise FloatingPointError("Stage A loss is NaN or Inf")
        loss.backward()
        final_gradients = trainable_snapshot(groups)
        torch.nn.utils.clip_grad_norm_(
            list(latents.parameters()) + list(decoder.parameters()),
            float(config["stage_a"]["gradient_clip_norm"]),
            error_if_nonfinite=True,
        )
        optimizer.step()
        row = {"step": step, "outfit": outfit, "loss": float(loss.detach()), **{f"loss_{name}": float(value.detach()) for name, value in parts.items()}}
        history.append(row); append_jsonl(stage_dir / "training.jsonl", row)
        if step in config["stage_a"]["milestones"]:
            report, _, _ = stage_a_evaluation(
                step=step, decoder=decoder, latents=latents, targets=targets, protected_mask=protected_mask,
                base=base, samples=samples, background=background, config=config, output_dir=output_dir,
            )
            milestones[str(step)] = report
            checkpoint = stage_dir / "checkpoints" / f"checkpoint_step_{step:06d}.pth"
            save_checkpoint(
                checkpoint, stage="A", step=step,
                model_state={"decoder": decoder.state_dict(), "diagnostic_latents": latents.state_dict()},
                optimizer=optimizer, scheduler=None,
                extra={"outfit_update_counts": {"O01": (step + 1) // 2, "O08": step // 2}, "next_outfit": OUTFITS[step % 2]},
            )
            atomic_json(stage_dir / "partial_status.json", {"status": "RUNNING", "completed_step": step, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint)})
    elapsed = time.time() - started
    final = milestones[str(config["stage_a"]["max_steps"])]
    with torch.no_grad():
        final_predictions = {outfit: prediction_from_latent(decoder, latents(outfit), protected_mask, chunk_size=int(config["stage_a"]["evaluation_chunk_size"])) for outfit in OUTFITS}
    _, _, final_renders = render_metrics(base, samples, final_predictions, targets, background, output_dir, int(config["stage_a"]["max_steps"]))
    legacy_predictions = load_legacy_predictions(base, protected_mask, config, base._xyz.device)
    legacy_renders: dict[str, dict[str, Any]] = {outfit: {} for outfit in OUTFITS}
    with torch.no_grad():
        for outfit in OUTFITS:
            for condition in CONDITIONS:
                sample = samples[f"{outfit}/{condition}"]
                rgb, alpha = render_prediction(base, sample, legacy_predictions[outfit], background)
                oracle_rgb, oracle_alpha = render_prediction(base, sample, targets[outfit], background)
                legacy_renders[outfit][condition] = {"prediction": (rgb.detach().cpu(), alpha.detach().cpu()), "oracle": (oracle_rgb.detach().cpu(), oracle_alpha.detach().cpu())}
    build_stage_a_contact_sheets(samples, step0_renders, final_renders, legacy_renders, output_dir)
    build_support_visualizations(base, final_predictions, targets, bounds(config), output_dir)

    first_window = float(np.mean([row["loss"] for row in history[:100]]))
    final_window = float(np.mean([row["loss"] for row in history[-100:]]))
    loss_drop = (first_window - final_window) / max(first_window, 1e-12)
    acceptance = config["stage_a"]["acceptance"]
    checks = {
        "first_100_to_final_loss_drop": loss_drop >= float(acceptance["first_100_to_final_loss_drop_fraction_min"]),
        "mean_normalized_rmse": final["mean"]["normalized_rmse"] <= float(acceptance["mean_bound_normalized_rmse_max"]),
        "mean_direction_cosine": final["mean"]["direction_cosine"] >= float(acceptance["mean_direction_cosine_min"]),
        "mean_top_10pct_overlap": final["mean"]["top_10pct_overlap"] >= float(acceptance["mean_top_10pct_overlap_min"]),
        "outfit_separation": final["outfit_separation_ratio"] >= float(acceptance["outfit_separation_ratio_min"]),
        "o01_render_garment_mae": final["render_per_outfit_garment_mae"]["O01"] <= float(acceptance["per_outfit_render_garment_mae_max"]),
        "o08_render_garment_mae": final["render_per_outfit_garment_mae"]["O08"] <= float(acceptance["per_outfit_render_garment_mae_max"]),
        "all_trainable_groups_have_finite_nonzero_gradient": all(value["gradient_finite"] and value["gradient_l2"] > 0 for value in final_gradients.values()),
        "base_bitwise_frozen": context["base_before"] == _tensor_state_fingerprint(_base_named_tensors(base)),
        "base_gradient_zero": _base_gradient_count(base) == 0,
        "image_backbone_bitwise_frozen": context["backbone_before"] == o01._state_fingerprint(model.clothing_observation_encoder.backbone.state_dict()),
        "image_backbone_gradient_zero": sum(parameter.grad is not None for parameter in model.clothing_observation_encoder.backbone.parameters()) == 0,
        "legacy_mlp_gradient_zero": sum(parameter.grad is not None for parameter in model.dressable_model.anchor_clothing_mlp.parameters()) == 0,
        "target_images_entered_forward": False,
    }
    checkpoint_path = stage_dir / "checkpoints" / "checkpoint_step_001000.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    restored_decoder = SupportConditionedDualBranchResidualDecoderV7.from_frozen_support(
        base, model.canonical_anchors, model.dressable_model.gaussian_anchor_indices,
        model.dressable_model.gaussian_anchor_weights,
        {**config["model"]["decoder"], "global_feature_dim": int(config["model"]["embedding_dim"]), "local_feature_dim": int(config["model"]["local_feature_dim"])},
        bounds(config),
    )
    restored_latents = DiagnosticOutfitLatents(OUTFITS, int(config["stage_a"]["diagnostic_latent_dim"]), int(config["stage_a"]["seed"])).to(base._xyz)
    restored_decoder.load_state_dict(checkpoint["model"]["decoder"], strict=True)
    restored_latents.load_state_dict(checkpoint["model"]["diagnostic_latents"], strict=True)
    restored_optimizer = torch.optim.Adam([
        {"name": "diagnostic_latents", "params": list(restored_latents.parameters()), "lr": float(config["stage_a"]["latent_learning_rate"])},
        {"name": "v7_decoder", "params": list(restored_decoder.parameters()), "lr": float(config["stage_a"]["decoder_learning_rate"])},
    ])
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    fixed_indices = torch.arange(256, device=base._xyz.device)
    with torch.no_grad():
        before_resume = prediction_from_latent(decoder, latents("O01"), protected_mask, gaussian_indices=fixed_indices, chunk_size=128)
        after_resume = prediction_from_latent(restored_decoder, restored_latents("O01"), protected_mask, gaussian_indices=fixed_indices, chunk_size=128)
    output_parity = all(
        torch.equal(getattr(before_resume, name), getattr(after_resume, name)) for name in CHANNELS
    )
    current_rng = rng_state()
    o01._restore_rng(checkpoint["rng"])
    rng_restored_exact = o01.object_fingerprint(rng_state()) == o01.object_fingerprint(checkpoint["rng"])
    o01._restore_rng(current_rng)
    resume = {
        "checkpoint": str(checkpoint_path), "sha256": sha256(checkpoint_path),
        "global_step_exact": int(checkpoint["global_step"]) == 1000,
        "optimizer_state_exact": o01.object_fingerprint(optimizer.state_dict()) == o01.object_fingerprint(restored_optimizer.state_dict()),
        "rng_state_exact": rng_restored_exact,
        "decoder_state_exact": _tensor_state_fingerprint(decoder.state_dict().items()) == _tensor_state_fingerprint(restored_decoder.state_dict().items()),
        "latent_state_exact": _tensor_state_fingerprint(latents.state_dict().items()) == _tensor_state_fingerprint(restored_latents.state_dict().items()),
        "fixed_output_bitwise_exact": output_parity,
        "next_outfit_exact": checkpoint["extra"]["next_outfit"] == "O01",
    }
    resume["pass"] = all(value for name, value in resume.items() if name not in {"checkpoint", "sha256"})
    checks["checkpoint_resume"] = resume["pass"]
    numeric_pass = all(checks.values())
    report = {
        "status": "NUMERIC_PASS_VISUAL_PENDING" if numeric_pass else "FAIL",
        "optimizer_steps": int(config["stage_a"]["max_steps"]),
        "first_100_mean_loss": first_window,
        "final_100_mean_loss": final_window,
        "first_100_to_final_loss_drop_fraction": loss_drop,
        "milestones": milestones,
        "final": final,
        "checks": checks,
        "gradients": final_gradients,
        "runtime_seconds": elapsed,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "parameter_count": sum(parameter.numel() for parameter in decoder.parameters()),
        "legacy_parameter_count": sum(parameter.numel() for parameter in model.dressable_model.anchor_clothing_mlp.parameters()) + sum(parameter.numel() for parameter in model.dressable_model.clothing_film_generator.parameters()),
        "descriptor": decoder.descriptor_metadata,
        "checkpoint_resume": resume,
        "freeze": {
            "base_before": context["base_before"],
            "base_after": _tensor_state_fingerprint(_base_named_tensors(base)),
            "base_bitwise_unchanged": context["base_before"] == _tensor_state_fingerprint(_base_named_tensors(base)),
            "base_gradient_count": _base_gradient_count(base),
            "image_backbone_before": context["backbone_before"],
            "image_backbone_after": o01._state_fingerprint(model.clothing_observation_encoder.backbone.state_dict()),
            "image_backbone_bitwise_unchanged": context["backbone_before"] == o01._state_fingerprint(model.clothing_observation_encoder.backbone.state_dict()),
            "image_backbone_gradient_count": sum(parameter.grad is not None for parameter in model.clothing_observation_encoder.backbone.parameters()),
            "legacy_mlp_gradient_count": sum(parameter.grad is not None for parameter in model.dressable_model.anchor_clothing_mlp.parameters()),
        },
        "visual_status": "PENDING_ACTUAL_INSPECTION" if numeric_pass else "NOT_ELIGIBLE_NUMERIC_FAIL",
    }
    atomic_json(stage_dir / "stage_a_metrics.json", report)
    atomic_json(stage_dir / "partial_status.json", {"status": report["status"], "completed_step": 1000, "checkpoint": str(checkpoint_path), "checkpoint_sha256": resume["sha256"]})
    return report


def stage_b_trainable_groups(model: Any, config: Mapping[str, Any]) -> dict[str, list[torch.nn.Parameter]]:
    set_requires_grad(model, False)
    modules = {
        "encoder_projection": model.clothing_observation_encoder.projection_head,
        "encoder_embedding_norm": model.clothing_observation_encoder.embedding_norm,
        "aggregator": model.multiview_aggregator,
        "completion": model.canonical_clothing_completer,
        "v7_decoder": model.support_conditioned_residual_decoder_v7,
    }
    groups = {}
    for name, module in modules.items():
        if module is None:
            raise RuntimeError(f"Stage B module is missing: {name}")
        set_requires_grad(module, True)
        groups[name] = list(module.parameters())
    if any(parameter.requires_grad for parameter in model.clothing_observation_encoder.backbone.parameters()):
        raise AssertionError("Stage B unfroze the image backbone")
    return groups


def stage_b_optimizer(model: Any, config: Mapping[str, Any]):
    groups = stage_b_trainable_groups(model, config)
    optimizer_config = config["stage_b"]["optimizer"]
    parameter_groups = [
        {"name": "encoder_projection", "params": groups["encoder_projection"] + groups["encoder_embedding_norm"], "lr": float(optimizer_config["encoder_lr"])},
        {"name": "aggregator", "params": groups["aggregator"], "lr": float(optimizer_config["aggregator_lr"])},
        {"name": "completion", "params": groups["completion"], "lr": float(optimizer_config["completer_lr"])},
        {"name": "v7_decoder", "params": groups["v7_decoder"], "lr": float(optimizer_config["residual_decoder_v7_lr"])},
    ]
    optimizer = torch.optim.Adam(parameter_groups, weight_decay=float(optimizer_config["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    return optimizer, scheduler, groups


def stage_b_prediction(
    model: Any,
    episode: Mapping[str, Any],
    geometry: Mapping[str, Any],
    protected_mask: torch.Tensor,
    *,
    gaussian_indices: torch.Tensor | None,
    chunk_size: int,
) -> tuple[GaussianClothingResiduals, Mapping[str, Any]]:
    inputs = o01._forward_inputs(episode, geometry)
    assert_forward_boundary(inputs)
    online = model.encode_clothing_online(**inputs)
    decoder = model.support_conditioned_residual_decoder_v7
    decoded = decoder(
        online["global_clothing_embedding"],
        online["completion"].completed_anchor_features,
        gaussian_indices=gaussian_indices,
        chunk_size=chunk_size,
        gate_mode=V7_CAPACITY_GATE_MODE,
    ).gated_gaussian_residuals
    mask = protected_mask if gaussian_indices is None else protected_mask.index_select(0, gaussian_indices)
    return apply_protected_full_residual_guard(decoded, mask), online


def evaluate_stage_b(
    context: Mapping[str, Any],
    model: Any,
    *,
    step: int,
    render_final: bool,
) -> dict[str, Any]:
    config, episodes, geometries = context["config"], context["episodes"], context["geometries"]
    targets, protected_mask = context["targets"], context["protected_mask"]
    base, samples, background = context["base"], context["samples"], context["background"]
    model.eval(); per_episode, predictions = {}, {}
    with torch.no_grad():
        for outfit in OUTFITS:
            for condition in CONDITIONS:
                key = f"{outfit}/{condition}"
                prediction, _ = stage_b_prediction(
                    model, episodes[key], geometries[condition], protected_mask,
                    gaussian_indices=None, chunk_size=int(config["stage_b"]["evaluation_chunk_size"]),
                )
                predictions[key] = prediction
                per_episode[key] = residual_metrics(prediction, targets[outfit], bounds(config))
        outfit_average = {}
        for outfit in OUTFITS:
            values = [predictions[f"{outfit}/{condition}"] for condition in CONDITIONS]
            outfit_average[outfit] = GaussianClothingResiduals(**{
                name: torch.stack([getattr(value, name) for value in values]).mean(0) for name in CHANNELS
            })
        separation = residual_distance_scalar(channel_comparison(outfit_average["O01"], outfit_average["O08"], bounds(config)))
        oracle_separation = residual_distance_scalar(channel_comparison(targets["O01"], targets["O08"], bounds(config)))
    report = {
        "step": step,
        "per_episode": per_episode,
        "mean": {name: float(np.mean([value[name] for value in per_episode.values()])) for name in ("normalized_rmse", "normalized_mae", "direction_cosine", "top_10pct_overlap", "top_20pct_overlap")},
        "outfit_separation_ratio": separation / max(oracle_separation, 1e-12),
    }
    if render_final:
        report.update(stage_b_reference_evaluation(context, model, predictions))
    directory = context["output_dir"] / "stage_b" / "milestones" / f"step_{step:06d}"
    atomic_json(directory / "metrics.json", report)
    model.train()
    return report


def stage_b_reference_evaluation(context: Mapping[str, Any], model: Any, correct_predictions: Mapping[str, GaussianClothingResiduals]) -> dict[str, Any]:
    config, episodes, geometries = context["config"], context["episodes"], context["geometries"]
    targets, protected_mask = context["targets"], context["protected_mask"]
    base, samples, background, output_dir = context["base"], context["samples"], context["background"], context["output_dir"]
    rows, residual_advantages, render_advantages, wins = [], [], [], 0
    contact_rows = []
    with torch.no_grad():
        for outfit in OUTFITS:
            swapped_outfit = "O08" if outfit == "O01" else "O01"
            for condition in CONDITIONS:
                key, swapped_key = f"{outfit}/{condition}", f"{swapped_outfit}/{condition}"
                correct = correct_predictions[key]
                swapped, _ = stage_b_prediction(model, episodes[swapped_key], geometries[condition], protected_mask, gaussian_indices=None, chunk_size=int(config["stage_b"]["evaluation_chunk_size"]))
                correct_loss, _ = normalized_residual_regression_loss(correct, targets[outfit], bounds(config))
                swapped_loss, _ = normalized_residual_regression_loss(swapped, targets[outfit], bounds(config))
                residual_advantage = float(swapped_loss - correct_loss)
                sample = samples[key]
                correct_rgb, _ = render_prediction(base, sample, correct, background)
                swapped_rgb, _ = render_prediction(base, sample, swapped, background)
                oracle_rgb, _ = render_prediction(base, sample, targets[outfit], background)
                mask = garment_mask(sample)
                correct_render = masked_mae(correct_rgb, oracle_rgb, mask)
                swapped_render = masked_mae(swapped_rgb, oracle_rgb, mask)
                render_advantage = swapped_render - correct_render
                residual_advantages.append(residual_advantage); render_advantages.append(render_advantage)
                if residual_advantage > 0 and render_advantage > 0:
                    wins += 1
                rows.append({
                    "outfit": outfit, "condition": condition, "view": VIEWS[condition],
                    "correct_residual_loss": float(correct_loss), "swapped_residual_loss": float(swapped_loss),
                    "correct_vs_swapped_residual_margin": residual_advantage,
                    "correct_render_mae": correct_render, "swapped_render_mae": swapped_render,
                    "correct_vs_swapped_render_margin": render_advantage,
                    "target_pose_camera_fixed": True,
                })
                contact_rows.append((key, [
                    ("target", sample["target_edit_rgb"], 3), ("Oracle", oracle_rgb, 3),
                    ("correct refs", correct_rgb, 3), ("swapped refs", swapped_rgb, 3),
                    ("correct error", (correct_rgb - oracle_rgb).abs(), 3),
                    ("swapped error", (swapped_rgb - oracle_rgb).abs(), 3),
                ]))
        # Reference zeroing is evaluated on one fixed target without changing its pose/camera.
        fixed_key = "O01/cond_000017"
        correct = correct_predictions[fixed_key]
        zero_episode, zero_geometry = reference_variant(episodes[fixed_key], geometries["cond_000017"], "zero_rgb_masks_kept")
        zero, _ = stage_b_prediction(model, zero_episode, zero_geometry, protected_mask, gaussian_indices=None, chunk_size=int(config["stage_b"]["evaluation_chunk_size"]))
        zero_change = residual_distance_scalar(channel_comparison(correct, zero, bounds(config)))
    o01._save_contact_sheet(output_dir / "visual_acceptance" / "stage_b_correct_vs_swapped_contact_sheet.png", contact_rows)
    return {
        "reference_evaluation_rows": rows,
        "mean_correct_vs_swapped_residual_margin": float(np.mean(residual_advantages)),
        "mean_correct_vs_swapped_render_margin": float(np.mean(render_advantages)),
        "correct_episode_wins": wins,
        "reference_zero_residual_change": zero_change,
        "reference_collapse": zero_change <= 1e-5,
        "target_forward_leakage": False,
    }


def run_stage_b(context: dict[str, Any]) -> dict[str, Any]:
    config, output_dir, base = context["config"], context["output_dir"], context["base"]
    stage_a = json.loads((output_dir / "stage_a" / "stage_a_metrics.json").read_text(encoding="utf-8"))
    if stage_a["status"] != "NUMERIC_PASS_VISUAL_PENDING" or context["stage_a_visual_status"] != "PASS":
        raise RuntimeError("Stage B requires numeric and actually-inspected Stage A PASS")
    model = context["model"]
    checkpoint_path = output_dir / "stage_a" / "checkpoints" / "checkpoint_step_001000.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.support_conditioned_residual_decoder_v7.load_state_dict(checkpoint["model"]["decoder"], strict=True)
    optimizer, scheduler, groups = stage_b_optimizer(model, config)
    stage_dir = output_dir / "stage_b"; stage_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "visual_acceptance" / "stage_a_visual_acceptance.json", {
        "status": "PASS", "images_actually_opened": True, "inspection_method": "Codex local image viewer", "recorded_before_stage_b": True,
    })
    milestones = {"0": evaluate_stage_b(context, model, step=0, render_final=False)}
    count = model.support_conditioned_residual_decoder_v7.gaussian_count
    chunk = int(config["stage_b"]["training_chunk_size"])
    generator = torch.Generator(device=base._xyz.device).manual_seed(int(config["stage_a"]["seed"]) + 2)
    schedule = [(outfit, condition) for condition in CONDITIONS for outfit in OUTFITS]
    history, final_gradients = [], {}
    started = time.time(); torch.cuda.reset_peak_memory_stats()
    for step in range(1, int(config["stage_b"]["max_steps"]) + 1):
        outfit, condition = schedule[(step - 1) % len(schedule)]
        key = f"{outfit}/{condition}"
        indices = torch.randint(count, (chunk,), device=base._xyz.device, generator=generator)
        optimizer.zero_grad(set_to_none=True)
        prediction, _ = stage_b_prediction(
            model, context["episodes"][key], context["geometries"][condition], context["protected_mask"],
            gaussian_indices=indices, chunk_size=chunk,
        )
        target = slice_residuals(context["targets"][outfit], indices)
        loss, parts = normalized_residual_regression_loss(prediction, target, bounds(config))
        if not torch.isfinite(loss):
            raise FloatingPointError("Stage B loss is NaN or Inf")
        loss.backward(); final_gradients = trainable_snapshot(groups)
        parameters = [parameter for values in groups.values() for parameter in values]
        torch.nn.utils.clip_grad_norm_(parameters, float(config["stage_b"]["gradient_clip_norm"]), error_if_nonfinite=True)
        optimizer.step(); scheduler.step()
        row = {"step": step, "outfit": outfit, "condition": condition, "loss": float(loss.detach()), **{f"loss_{name}": float(value.detach()) for name, value in parts.items()}}
        history.append(row); append_jsonl(stage_dir / "training.jsonl", row)
        if step in config["stage_b"]["milestones"]:
            report = evaluate_stage_b(context, model, step=step, render_final=step == int(config["stage_b"]["max_steps"]))
            milestones[str(step)] = report
            checkpoint_path = stage_dir / "checkpoints" / f"checkpoint_step_{step:06d}.pth"
            save_checkpoint(
                checkpoint_path, stage="B", step=step, model_state={"model": model.state_dict()}, optimizer=optimizer,
                scheduler=scheduler, extra={"schedule_position": step % len(schedule), "outfit_view_schedule": schedule},
            )
            atomic_json(stage_dir / "partial_status.json", {"status": "RUNNING", "completed_step": step, "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path)})
    final = milestones["1000"]
    acceptance = config["stage_b"]["acceptance"]
    checks = {
        "mean_normalized_rmse": final["mean"]["normalized_rmse"] <= float(acceptance["mean_bound_normalized_rmse_max"]),
        "mean_direction_cosine": final["mean"]["direction_cosine"] >= float(acceptance["mean_direction_cosine_min"]),
        "mean_top_10pct_overlap": final["mean"]["top_10pct_overlap"] >= float(acceptance["mean_top_10pct_overlap_min"]),
        "correct_vs_swapped_residual_margin": final["mean_correct_vs_swapped_residual_margin"] > float(acceptance["correct_vs_swapped_residual_margin_min"]),
        "correct_vs_swapped_render_margin": final["mean_correct_vs_swapped_render_margin"] > float(acceptance["correct_vs_swapped_render_margin_min"]),
        "correct_episode_wins": int(final["correct_episode_wins"]) >= int(acceptance["correct_episode_wins_min"]),
        "reference_not_collapsed": not bool(final["reference_collapse"]),
        "target_forward_leakage_zero": not bool(final["target_forward_leakage"]),
        "trainable_gradients_finite_nonzero": all(value["gradient_finite"] and value["gradient_l2"] > 0 for value in final_gradients.values()),
        "base_bitwise_frozen": context["base_before"] == _tensor_state_fingerprint(_base_named_tensors(base)),
        "base_gradient_zero": _base_gradient_count(base) == 0,
        "image_backbone_bitwise_frozen": context["backbone_before"] == o01._state_fingerprint(model.clothing_observation_encoder.backbone.state_dict()),
        "legacy_mlp_gradient_zero": sum(parameter.grad is not None for parameter in model.dressable_model.anchor_clothing_mlp.parameters()) == 0,
        "image_backbone_gradient_zero": sum(parameter.grad is not None for parameter in model.clothing_observation_encoder.backbone.parameters()) == 0,
    }
    checkpoint_final = stage_dir / "checkpoints" / "checkpoint_step_001000.pth"
    checkpoint = torch.load(checkpoint_final, map_location="cpu", weights_only=False)
    restored_model, _ = construct_model(base, config, base._xyz.device)
    restored_model.load_state_dict(checkpoint["model"]["model"], strict=True)
    restored_optimizer, restored_scheduler, _ = stage_b_optimizer(restored_model, config)
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    restored_scheduler.load_state_dict(checkpoint["scheduler"])
    fixed_indices = torch.arange(256, device=base._xyz.device)
    fixed_key = "O01/cond_000017"
    with torch.no_grad():
        before_resume, _ = stage_b_prediction(
            model, context["episodes"][fixed_key], context["geometries"]["cond_000017"],
            context["protected_mask"], gaussian_indices=fixed_indices, chunk_size=128,
        )
        after_resume, _ = stage_b_prediction(
            restored_model, context["episodes"][fixed_key], context["geometries"]["cond_000017"],
            context["protected_mask"], gaussian_indices=fixed_indices, chunk_size=128,
        )
    current_rng = rng_state(); o01._restore_rng(checkpoint["rng"])
    rng_restored_exact = o01.object_fingerprint(rng_state()) == o01.object_fingerprint(checkpoint["rng"])
    o01._restore_rng(current_rng)
    resume = {
        "global_step_exact": int(checkpoint["global_step"]) == 1000,
        "model_state_exact": _tensor_state_fingerprint(model.state_dict().items()) == _tensor_state_fingerprint(restored_model.state_dict().items()),
        "optimizer_state_exact": o01.object_fingerprint(optimizer.state_dict()) == o01.object_fingerprint(restored_optimizer.state_dict()),
        "scheduler_state_exact": o01.object_fingerprint(scheduler.state_dict()) == o01.object_fingerprint(restored_scheduler.state_dict()),
        "rng_state_exact": rng_restored_exact,
        "schedule_position_exact": int(checkpoint["extra"]["schedule_position"]) == 0,
        "fixed_output_bitwise_exact": all(torch.equal(getattr(before_resume, name), getattr(after_resume, name)) for name in CHANNELS),
    }
    resume["pass"] = all(resume.values())
    checks["checkpoint_scheduler_exact_resume"] = resume["pass"]
    numeric_pass = all(checks.values())
    report = {
        "status": "NUMERIC_PASS_VISUAL_PENDING" if numeric_pass else "FAIL",
        "optimizer_steps": 1000,
        "milestones": milestones,
        "checks": checks,
        "gradients": final_gradients,
        "runtime_seconds": time.time() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "visual_status": "PENDING_ACTUAL_INSPECTION" if numeric_pass else "NOT_ELIGIBLE_NUMERIC_FAIL",
        "checkpoint": str(stage_dir / "checkpoints" / "checkpoint_step_001000.pth"),
        "checkpoint_sha256": sha256(stage_dir / "checkpoints" / "checkpoint_step_001000.pth"),
        "checkpoint_resume": resume,
    }
    atomic_json(stage_dir / "stage_b_metrics.json", report)
    atomic_json(stage_dir / "partial_status.json", {"status": report["status"], "completed_step": 1000, "checkpoint": report["checkpoint"], "checkpoint_sha256": report["checkpoint_sha256"]})
    return report


def input_context(config: dict[str, Any], output_dir: Path, *, create: bool) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("formal V7 capacity closure requires CUDA")
    device = torch.device("cuda")
    seed = int(config["stage_a"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    branch, dirty = git_output("branch", "--show-current"), git_output("status", "--short")
    if branch != EXPECTED_BRANCH or dirty:
        raise RuntimeError(f"formal V7 run requires clean {EXPECTED_BRANCH}; branch={branch} dirty={bool(dirty)}")
    paths = {
        "source_manifest": Path(config["inputs"]["source_manifest"]),
        "stable_protected_attribution": Path(config["inputs"]["stable_protected_attribution"]),
        "diagnosis_checkpoint": Path(config["inputs"]["diagnosis_checkpoint"]),
        "oracle_o01_checkpoint": Path(config["inputs"]["oracle_o01_checkpoint"]),
        "oracle_o08_checkpoint": Path(config["inputs"]["oracle_o08_checkpoint"]),
        "base_checkpoint": Path(config["base"]["model_dir"]) / config["base"]["checkpoint_path"],
        "lbs_grid": Path(config["base"]["lbs_grid_path"]),
    }
    immutable_attempts = {
        "o01_attempt": Path(config["inputs"]["o01_attempt"]),
        "diagnosis_attempt": Path(config["inputs"]["diagnosis_attempt"]),
    }
    expected = {
        "source_manifest": config["inputs"]["source_manifest_sha256"],
        "stable_protected_attribution": config["inputs"]["stable_protected_attribution_sha256"],
        "diagnosis_checkpoint": config["inputs"]["diagnosis_checkpoint_sha256"],
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
    immutable_tree_fingerprints = {
        name: immutable_tree_metadata_fingerprint(path) for name, path in immutable_attempts.items()
    }
    if create:
        (output_dir / "contract").mkdir(parents=True)
        (output_dir / "input_audit").mkdir()
        (output_dir / "visual_acceptance").mkdir()
        (output_dir / "final_adjudication").mkdir()
        (output_dir / "contract" / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        atomic_text(output_dir / "contract" / "command.txt", " ".join([sys.executable, *sys.argv]))
        atomic_json(output_dir / "input_audit" / "environment.json", o01.environment_snapshot())
        atomic_json(output_dir / "input_audit" / "input_manifest.json", {
            "task_id": TASK_ID,
            "git": {"branch": branch, "commit": git_output("rev-parse", "HEAD"), "clean": True, "source_head": EXPECTED_SOURCE_HEAD},
            "paths": {name: str(path) for name, path in paths.items()}, "sha256": hashes,
            "immutable_attempt_paths": {name: str(path) for name, path in immutable_attempts.items()},
            "immutable_attempt_tree_metadata_fingerprints": immutable_tree_fingerprints,
            "o01_attempt_immutable": True, "diagnosis_attempt_immutable": True, "oracle_outputs_immutable": True,
            "historical_checkpoint_warning": {"historical_sha256": config["base"]["historical_checkpoint_sha256_warning"], "current_sha256": config["base"]["checkpoint_sha256"], "same_file": False},
        })
    samples, episodes, protocol = diagnosis._generic_outfit_data(paths["source_manifest"])
    base = training.load_frozen_mmlphuman_base(config["base"]["model_dir"], paths["base_checkpoint"], device=device)
    if int(base._xyz.shape[0]) != int(config["base"]["gaussian_count"]):
        raise ValueError("base Gaussian count differs from V7 contract")
    protected_cpu, attribution = o01._load_stable_protected_mask(
        paths["stable_protected_attribution"], int(config["inputs"]["stable_protected_count"]), int(base._xyz.shape[0]),
    )
    protected_mask = protected_cpu.to(device)
    base_before = _tensor_state_fingerprint(_base_named_tensors(base))
    model, graph = construct_model(base, config, device)
    backbone_before = o01._state_fingerprint(model.clothing_observation_encoder.backbone.state_dict())
    adapter = MMLPHumanAnchorDeformationAdapter.from_mmlphuman_base(base, lbs_grid_path=paths["lbs_grid"])
    background = torch.tensor(config["render"]["background"], device=device, dtype=base._xyz.dtype)
    for key in list(samples):
        samples[key] = diagnosis._to_device_nested(samples[key], device)
        episodes[key] = diagnosis._to_device_nested(episodes[key], device)
    geometries = {}
    for condition in CONDITIONS:
        geometries[condition] = training.prepare_real_reference_geometry(base, adapter, episodes[f"O01/{condition}"], background)
        for outfit in OUTFITS:
            key = f"{outfit}/{condition}"
            support, _ = o01._base_only_protected_support(base, samples[key], protected_mask, background, float(config["render"]["protected_support_alpha_threshold"]))
            samples[key]["target_protected_mask"] = torch.maximum(samples[key]["target_protected_mask"], support)
    targets, oracle_meta = {}, {}
    for outfit in OUTFITS:
        oracle, oracle_meta[outfit] = diagnosis._load_oracle(base, paths[f"oracle_{outfit.lower()}_checkpoint"], device)
        targets[outfit] = apply_protected_full_residual_guard(oracle.residuals(base), protected_mask)
        if any(value.requires_grad or value.grad_fn is not None for value in targets[outfit].as_dict().values()):
            raise AssertionError("V7 Oracle target retained an autograd graph")
    if create:
        atomic_json(output_dir / "input_audit" / "resolved_protocol.json", {
            "protocol": protocol, "graph": graph, "oracles": oracle_meta, "attribution": attribution,
            "target_images_in_prediction_forward": False,
            "decoder_descriptor": model.support_conditioned_residual_decoder_v7.descriptor_metadata,
        })
    return {
        "config": config, "output_dir": output_dir, "base": base, "model": model,
        "samples": samples, "episodes": episodes, "geometries": geometries,
        "targets": targets, "protected_mask": protected_mask, "background": background,
        "base_before": base_before, "backbone_before": backbone_before,
        "immutable_tree_fingerprints": immutable_tree_fingerprints,
    }


def _load_visual_observations(path: Path | None, expected_status: str) -> dict[str, Any]:
    if path is None:
        raise ValueError("finalize requires --visual-observations from an actual image inspection")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("images_actually_opened") is not True:
        raise ValueError("visual observations must confirm images_actually_opened=true")
    if payload.get("stage_a_status") != expected_status:
        raise ValueError(
            f"visual observation status {payload.get('stage_a_status')!r} does not match "
            f"--stage-a-visual-status {expected_status!r}"
        )
    if not payload.get("inspection_method") or not payload.get("images") or not payload.get("observations"):
        raise ValueError("visual observations require inspection_method, images, and observations")
    return payload


def _visual_acceptance_markdown(payload: Mapping[str, Any], stage_b_visual_status: str | None) -> str:
    lines = [
        f"# {TASK_ID} visual acceptance",
        "",
        f"- Images actually opened: `{payload['images_actually_opened']}`",
        f"- Inspection method: {payload['inspection_method']}",
        f"- Stage A visual status: **{payload['stage_a_status']}**",
        f"- Stage B visual status: **{stage_b_visual_status or 'NOT_RUN'}**",
        "",
        "## Images inspected",
        "",
    ]
    lines.extend(f"- `{item}`" for item in payload["images"])
    lines.extend(["", "## Observations", ""])
    lines.extend(f"- {item}" for item in payload["observations"])
    return "\n".join(lines)


def finalize(
    context: Mapping[str, Any],
    stage_a_visual_status: str,
    stage_b_visual_status: str | None,
    visual_observations_path: Path | None = None,
) -> dict[str, Any]:
    output_dir, model, base = context["output_dir"], context["model"], context["base"]
    stage_a = json.loads((output_dir / "stage_a" / "stage_a_metrics.json").read_text(encoding="utf-8"))
    stage_b_path = output_dir / "stage_b" / "stage_b_metrics.json"
    stage_b = json.loads(stage_b_path.read_text(encoding="utf-8")) if stage_b_path.is_file() else {"status": "NOT_RUN_STAGE_A_FAILED", "optimizer_steps": 0}
    stage_a_pass = stage_a["status"] == "NUMERIC_PASS_VISUAL_PENDING" and stage_a_visual_status == "PASS"
    stage_b_pass = stage_b["status"] == "NUMERIC_PASS_VISUAL_PENDING" and stage_b_visual_status == "PASS"
    if not stage_a_pass:
        case = "V7-D"; next_task = "REASSESS_RESIDUAL_FIELD_PARAMETERIZATION"; final_status = "FAIL"
    elif not stage_b_pass:
        case = "V7-C"; next_task = "REDESIGN_REFERENCE_TO_SUPPORT_FEATURE_FUSION"; final_status = "FAIL"
    else:
        case = "V7-P"; next_task = "VALIDATE_V7_IMAGE_SPACE_OBJECTIVE_FROM_TEACHER_INITIALIZATION"; final_status = "PASS"
    freeze = {
        "base_before": context["base_before"],
        "base_after": _tensor_state_fingerprint(_base_named_tensors(base)),
        "base_gradient_count": _base_gradient_count(base),
        "image_backbone_before": context["backbone_before"],
        "image_backbone_after": o01._state_fingerprint(model.clothing_observation_encoder.backbone.state_dict()),
        "image_backbone_gradient_count": sum(parameter.grad is not None for parameter in model.clothing_observation_encoder.backbone.parameters()),
        "legacy_mlp_gradient_count": sum(parameter.grad is not None for parameter in model.dressable_model.anchor_clothing_mlp.parameters()),
    }
    freeze["base_bitwise_unchanged"] = freeze["base_before"] == freeze["base_after"]
    freeze["image_backbone_bitwise_unchanged"] = freeze["image_backbone_before"] == freeze["image_backbone_after"]
    immutable_inputs = json.loads((output_dir / "input_audit" / "input_manifest.json").read_text(encoding="utf-8"))
    immutable_tree_after = {
        name: immutable_tree_metadata_fingerprint(Path(path))
        for name, path in immutable_inputs["immutable_attempt_paths"].items()
    }
    freeze["old_o01_and_diagnosis_outputs_immutable"] = (
        immutable_tree_after == immutable_inputs["immutable_attempt_tree_metadata_fingerprints"]
    )
    if not freeze["base_bitwise_unchanged"] or not freeze["image_backbone_bitwise_unchanged"] or freeze["base_gradient_count"] != 0 or not freeze["old_o01_and_diagnosis_outputs_immutable"]:
        raise AssertionError("V7 frozen base/backbone contract failed")
    result = {
        "task_id": TASK_ID, "run_commit": git_output("rev-parse", "HEAD"),
        "stage_a": stage_a, "stage_b": stage_b,
        "stage_a_visual_status": stage_a_visual_status,
        "stage_b_visual_status": stage_b_visual_status,
        "freeze": freeze, "final_case": case, "status": final_status,
        "image_space_objective_allowed": case == "V7-P", "next_task": next_task,
    }
    visual_observations = _load_visual_observations(visual_observations_path, stage_a_visual_status)
    visual_observations["stage_b_status"] = stage_b_visual_status or "NOT_RUN"
    atomic_json(output_dir / "visual_acceptance" / "visual_acceptance.json", visual_observations)
    atomic_text(
        output_dir / "visual_acceptance" / "VISUAL_ACCEPTANCE.md",
        _visual_acceptance_markdown(visual_observations, stage_b_visual_status),
    )
    atomic_json(output_dir / "final_adjudication" / "final_adjudication.json", result)
    atomic_text(output_dir / "final_adjudication" / "FINAL_ADJUDICATION.md", "\n".join([
        f"# {TASK_ID}", "", f"- Run commit: `{result['run_commit']}`",
        f"- Stage A: **{stage_a['status']} / visual {stage_a_visual_status}**",
        f"- Stage B: **{stage_b['status']} / visual {stage_b_visual_status or 'NOT_RUN'}**",
        f"- Final case: **{case}**", f"- Final status: **{final_status}**",
        f"- Image-space objective allowed: `{result['image_space_objective_allowed']}`",
        f"- Next unique task: `{next_task}`",
    ]))
    atomic_json(output_dir / "RUN_STATUS.json", {
        "task_id": TASK_ID, "status": "COMPLETE", "final_case": case,
        "optimizer_steps": int(stage_a["optimizer_steps"]) + int(stage_b["optimizer_steps"]),
        "completed_at": time.time(),
    })
    return result


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config.resolve())
    output_dir = Path(args.output_dir or Path(config["output"]["task_root"]) / config["output"]["attempt"])
    create = args.phase == "stage-a"
    if create:
        if output_dir.exists():
            raise FileExistsError(f"append-only V7 attempt already exists: {output_dir}")
        output_dir.mkdir(parents=True)
        atomic_json(output_dir / "RUN_STATUS.json", {"task_id": TASK_ID, "status": "RUNNING", "phase": "input_audit", "optimizer_steps": 0, "started_at": time.time()})
    elif not output_dir.is_dir():
        raise FileNotFoundError(f"V7 attempt does not exist: {output_dir}")
    context = input_context(config, output_dir, create=create)
    try:
        if args.phase == "stage-a":
            atomic_json(output_dir / "RUN_STATUS.json", {"task_id": TASK_ID, "status": "RUNNING", "phase": "stage_a", "optimizer_steps": 0})
            report = run_stage_a(context)
            atomic_json(output_dir / "RUN_STATUS.json", {"task_id": TASK_ID, "status": "AWAITING_STAGE_A_VISUAL_INSPECTION" if report["status"].startswith("NUMERIC_PASS") else "AWAITING_FINAL_ADJUDICATION", "phase": "stage_a_complete", "optimizer_steps": report["optimizer_steps"]})
        elif args.phase == "stage-b":
            context["stage_a_visual_status"] = args.stage_a_visual_status
            atomic_json(output_dir / "RUN_STATUS.json", {"task_id": TASK_ID, "status": "RUNNING", "phase": "stage_b", "optimizer_steps": 1000})
            report = run_stage_b(context)
            atomic_json(output_dir / "RUN_STATUS.json", {"task_id": TASK_ID, "status": "AWAITING_STAGE_B_VISUAL_INSPECTION" if report["status"].startswith("NUMERIC_PASS") else "AWAITING_FINAL_ADJUDICATION", "phase": "stage_b_complete", "optimizer_steps": 2000})
        elif args.phase == "finalize":
            finalize(
                context,
                args.stage_a_visual_status,
                args.stage_b_visual_status,
                args.visual_observations,
            )
        else:
            raise ValueError(f"unknown phase: {args.phase}")
    except Exception as error:
        atomic_json(output_dir / "RUN_STATUS.json", {
            "task_id": TASK_ID, "status": "FAILED_TOOL_OR_RUNTIME", "phase": args.phase,
            "exception_type": type(error).__name__, "exception": str(error), "traceback": traceback.format_exc(),
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the preregistered V7 residual decoder capacity closure")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--phase", choices=("stage-a", "stage-b", "finalize"), required=True)
    parser.add_argument("--stage-a-visual-status", choices=("PASS", "WARN", "FAIL"), default=None)
    parser.add_argument("--stage-b-visual-status", choices=("PASS", "WARN", "FAIL"), default=None)
    parser.add_argument("--visual-observations", type=Path, default=None)
    args = parser.parse_args()
    if args.phase in {"stage-b", "finalize"} and args.stage_a_visual_status is None:
        parser.error("stage-b/finalize require --stage-a-visual-status")
    run(args)


if __name__ == "__main__":
    main()
