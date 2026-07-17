from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import random
import shutil
import subprocess
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
from scene.full_attribute_oracle import (  # noqa: E402
    AnchorResidualOracle,
    GaussianResidualOracle,
    STAGE_TRAINABLE,
)
from scene.full_dressable_dataset import (  # noqa: E402
    DUAL_TARGET_FIELDS,
    FullDressableTrainingDataset,
)
from tools.check_real_image_conditioned_one_batch import save_render_tensor  # noqa: E402
from utils.dressable_camera_utils import build_mmlphuman_camera  # noqa: E402
from utils.full_training_checkpoint_utils import (  # noqa: E402
    load_full_training_checkpoint,
    save_full_training_checkpoint,
)
from utils.mmlphuman_anchor_deformation import MMLPHumanAnchorDeformationAdapter  # noqa: E402
from utils.mmlphuman_state_utils import mmlphuman_state_transaction  # noqa: E402
from utils.rendering_loss_utils import (  # noqa: E402
    boundary_aware_transition_alpha_target,
    compute_static_transition_gradient_cap,
    region_aware_dual_target_loss,
)


EXPECTED_SCHEMA = "canondressgs.module4b.canonical_capacity.v1"
EXPECTED_DATA_SCHEMA = "canondressgs.full_dataset.v1"
KINDS = ("gaussian", "anchor")
OUTFITS = ("O00", "O01", "O05")
CONDITIONS = ("cond_000000", "cond_000318", "cond_000017", "cond_000347")
VIEWS = {"cond_000000": "front", "cond_000318": "back", "cond_000017": "left", "cond_000347": "right"}
CHANNELS = (
    "delta_xyz",
    "delta_log_scaling",
    "delta_rotvec",
    "delta_opacity_logit",
    "delta_sh0",
    "delta_shN",
)
RAW_TO_CHANNEL = {
    "raw_xyz": "delta_xyz",
    "raw_log_scaling": "delta_log_scaling",
    "raw_rotvec": "delta_rotvec",
    "raw_opacity": "delta_opacity_logit",
    "raw_sh0": "delta_sh0",
    "raw_shN": "delta_shN",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _tensor_state_fingerprint(named: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(named, key=lambda item: item[0]):
        tensor = value.detach().contiguous().cpu()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _object_fingerprint(value: Any) -> str:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True, default=str) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _git(command: list[str]) -> str:
    return subprocess.run(
        ["git", *command], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()


def _git_state() -> dict[str, Any]:
    return {
        "branch": _git(["branch", "--show-current"]),
        "commit": _git(["rev-parse", "HEAD"]),
        "status_short": _git(["status", "--short"]),
    }


def _environment() -> dict[str, Any]:
    gpu = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(torch.cuda.current_device())
    return {
        "timestamp_unix": time.time(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
    }


def load_contract(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != EXPECTED_SCHEMA:
        raise ValueError(f"unexpected Module 4B config schema: {value.get('schema_version')}")
    if tuple(value["outfits"]) != OUTFITS:
        raise ValueError("Module 4B outfits differ from the preregistered contract")
    if tuple(value["round_robin"]) != CONDITIONS or int(value["steps"]) != 480:
        raise ValueError("Module 4B condition order or step count differs from the contract")
    if value["schedule_policy"] != "module4a_formal_stage_precedence":
        raise ValueError("Module 4A formal stage precedence is mandatory")
    if bool(value["oracle"].get("enable_shn", True)):
        raise ValueError("formal Module 4A degree-0 policy requires SHN disabled")
    return value


def stage_for_state(step: int) -> int:
    if not 0 <= step <= 480:
        raise ValueError("state step must be in [0,480]")
    if step < 80:
        return 1
    if step < 160:
        return 2
    return 3


def _load_samples(manifest: Path, outfit_id: str) -> dict[str, dict[str, Any]]:
    dataset = FullDressableTrainingDataset(manifest, "train", reference_count=1, seed=0)
    if dataset.manifest.get("schema_version") != EXPECTED_DATA_SCHEMA:
        raise ValueError("fixture does not satisfy the full dataset v1 contract")
    if dataset.supervision_mode != "dual_target_region_aware_v1":
        raise ValueError("Module 4B requires dual_target_region_aware_v1")
    selected: dict[str, dict[str, Any]] = {}
    for outfit, observation in dataset.samples:
        if outfit["outfit_id"] != outfit_id or observation["condition_id"] not in CONDITIONS:
            continue
        target = dataset._observation(outfit, observation, True)
        missing = sorted(DUAL_TARGET_FIELDS.difference(observation))
        if missing:
            raise ValueError(f"dual-target observation is missing fields: {missing}")
        sample = {
            "target_edit_rgb": dataset._image(observation["target_edit_rgb"], 3),
            "target_base_rgb": dataset._image(observation["target_base_rgb"], 3),
            "target_foreground_mask": target["foreground_mask"],
            "target_clothing_mask": target["clothing_mask"],
            "target_pose": target["pose"],
            "target_Rh": target["R_global"],
            "target_Th": target["Th"],
            "target_camera": {
                "K": target["K"], "w2c": target["w2c"],
                "width": target["width"], "height": target["height"],
            },
            "target_condition_id": target["condition_id"],
            "outfit_id": outfit_id,
            "outfit_metadata": outfit.get("metadata", {}),
            "source_record": observation,
        }
        for name in sorted(DUAL_TARGET_FIELDS.difference({"target_edit_rgb", "target_base_rgb"})):
            sample[name] = dataset._image(observation[name], 1)
        selected[target["condition_id"]] = sample
    if tuple(key for key in CONDITIONS if key in selected) != CONDITIONS or len(selected) != 4:
        raise ValueError(f"{outfit_id} does not contain the exact four-condition protocol")
    return {condition: selected[condition] for condition in CONDITIONS}


def _gaussian_edges(base: Any, device: torch.device) -> torch.Tensor | None:
    neighbors = getattr(base, "nbr_vt", None)
    if not isinstance(neighbors, torch.Tensor) or neighbors.ndim != 2:
        return None
    neighbors = neighbors.to(device=device, dtype=torch.long)
    rows = torch.arange(neighbors.shape[0], device=device).unsqueeze(1).expand_as(neighbors)
    edges = torch.stack((rows.reshape(-1), neighbors.reshape(-1)), dim=1)
    return edges[(edges[:, 1] >= 0) & (edges[:, 1] < neighbors.shape[0])]


def build_oracle(kind: str, base: Any, config: Mapping[str, Any], device: torch.device):
    common = {
        "bounds": config["oracle"]["bounds"],
        "initial_gate_probability": config["oracle"]["initial_gate_probability"],
        "enable_shn": False,
    }
    if kind == "gaussian":
        return GaussianResidualOracle(base, graph_edges=_gaussian_edges(base, device), **common)
    if kind != "anchor":
        raise ValueError(f"unknown oracle kind: {kind}")
    anchors = getattr(base, "xyz_vt", None)
    indices = getattr(base, "nbr_gs", None)
    weights = getattr(base, "nbr_gs_invdist", None)
    if not all(isinstance(value, torch.Tensor) for value in (anchors, indices, weights)):
        raise ValueError("anchor oracle requires formal xyz_vt/nbr_gs/nbr_gs_invdist tensors")
    normalized = weights.to(device=device, dtype=base._xyz.dtype)
    normalized = normalized / normalized.sum(dim=1, keepdim=True)
    if not torch.allclose(normalized.sum(1), torch.ones_like(normalized[:, 0]), atol=1e-5, rtol=0):
        raise ValueError("formal anchor interpolation weights do not normalize to one")
    edges = training.build_anchor_knn_edges(anchors.to(device), 4)
    return AnchorResidualOracle(
        base,
        int(anchors.shape[0]),
        indices.to(device=device, dtype=torch.long),
        normalized,
        graph_edges=edges,
        **common,
    )


def _optimizer(oracle: Any, config: Mapping[str, Any]):
    optimizer_config = config["oracle"]["optimizer"]
    rates = {
        "geometry_residuals": optimizer_config["geometry_residuals_lr"],
        "appearance_residuals": optimizer_config["appearance_residuals_lr"],
        "geometry_gate": optimizer_config["geometry_gate_lr"],
        "appearance_gate": optimizer_config["appearance_gate_lr"],
    }
    groups, names = oracle.parameter_groups(rates)
    return torch.optim.Adam(groups), names


def _chw(value: torch.Tensor, channels: int) -> torch.Tensor:
    if value.ndim == 3 and value.shape[0] == channels:
        result = value
    elif value.ndim == 3 and value.shape[-1] == channels:
        result = value.permute(2, 0, 1)
    else:
        raise ValueError(f"render tensor must be CHW/HWC with {channels} channels, got {tuple(value.shape)}")
    if not torch.isfinite(result).all():
        raise FloatingPointError("render output contains NaN or Inf")
    return result.contiguous()


def _render(base: Any, sample: Mapping[str, Any], overrides: Any, background: torch.Tensor):
    height = int(sample["target_camera"]["height"])
    width = int(sample["target_camera"]["width"])
    camera = build_mmlphuman_camera(sample["target_camera"], height, width, base._xyz.device)
    with mmlphuman_state_transaction(
        base,
        sample["target_pose"].to(base._xyz.device),
        sample["target_Rh"].to(base._xyz.device),
        sample["target_Th"].to(base._xyz.device),
    ):
        rendered = base.render(camera, background=background, canonical_overrides=overrides.as_dict())
    return _chw(rendered[0], 3), _chw(rendered[1], 1)


def _transition_targets(samples: Mapping[str, Mapping[str, Any]], device: torch.device):
    values = {}
    for condition, sample in samples.items():
        values[condition] = boundary_aware_transition_alpha_target(
            sample["target_foreground_mask"].to(device),
            sample["target_base_foreground_mask"].to(device),
            sample["target_edit_core_mask"].to(device),
            sample["target_preserve_mask"].to(device),
            sample["target_protected_mask"].to(device),
        )["target"]
    return values


def _loss(
    rgb: torch.Tensor,
    alpha: torch.Tensor,
    sample: Mapping[str, Any],
    transition_target: torch.Tensor,
    device: torch.device,
    config: Mapping[str, Any],
    effective_transition_coefficient: float,
) -> dict[str, torch.Tensor]:
    loss = config["loss"]
    alpha_global = float(loss["alpha_global"])
    if alpha_global <= 0:
        raise ValueError("alpha_global must be positive")
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
        edit_weight=float(loss["edit"]),
        clothing_weight=float(loss["clothing"]),
        preserve_weight=float(loss["preserve"]),
        protected_weight=float(loss["protected"]),
        transition_weight=float(loss["rgb_transition"]),
        alpha_weight=alpha_global,
        alpha_edit_weight=float(loss["alpha_edit"]),
        alpha_transition_weight=effective_transition_coefficient / alpha_global,
        alpha_base_weight=float(loss["alpha_base"]),
        transition_alpha_target=transition_target,
    )


def _regularization(output: Any, config: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    weights = config["loss"]["regularization"]
    regularization = output.regularization
    parts = {
        "xyz_magnitude": regularization.residual_magnitude["delta_xyz"],
        "scaling_magnitude": regularization.residual_magnitude["delta_log_scaling"],
        "rotation_magnitude": regularization.residual_magnitude["delta_rotvec"],
        "opacity_magnitude": regularization.residual_magnitude["delta_opacity_logit"],
        "sh0_magnitude": regularization.residual_magnitude["delta_sh0"],
        "shN_magnitude": regularization.residual_magnitude["delta_shN"],
        "geometry_gate_sparsity": regularization.geometry_gate_sparsity,
        "appearance_gate_sparsity": regularization.appearance_gate_sparsity,
        "gate_binary": regularization.gate_binary,
        "graph_gate_smoothness": regularization.graph_gate_smoothness,
        "graph_residual_smoothness": regularization.graph_residual_smoothness,
    }
    total = sum(float(weights[name]) * value for name, value in parts.items())
    return total, {name: float(value.detach()) for name, value in parts.items()}


def _gradient_norm(loss: torch.Tensor, parameters: list[torch.Tensor]) -> float:
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    total = sum((gradient.double().square().sum() for gradient in gradients if gradient is not None), torch.zeros((), dtype=torch.float64, device=loss.device))
    return float(torch.sqrt(total).detach())


def _compute_transition_cap(
    base: Any,
    oracle: Any,
    sample: Mapping[str, Any],
    transition_target: torch.Tensor,
    background: torch.Tensor,
    device: torch.device,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    oracle.configure_stage(1)
    output = oracle(base)
    rgb, alpha = _render(base, sample, output.canonical_overrides, background)
    original = float(config["loss"]["alpha_global"]) * float(config["loss"]["alpha_transition_original"])
    parts = _loss(rgb, alpha, sample, transition_target, device, config, original)
    parameters = [value for value in oracle.parameters() if value.requires_grad]
    rgb_norm = _gradient_norm(parts["edit"] + parts["clothing"], parameters)
    transition_norm = _gradient_norm(parts["alpha_transition"], parameters)
    cap = compute_static_transition_gradient_cap(
        rgb_norm,
        transition_norm,
        original,
        cap_fraction=float(config["loss"]["transition_gradient_cap_fraction"]),
        epsilon=float(config["loss"]["transition_gradient_cap_epsilon"]),
    )
    cap.update({
        "computed_once_at_step": 0,
        "frozen_for_steps": [0, 480],
        "reference_parameter_set": "Module4A formal stage-1 trainable oracle parameters",
        "shared_decoder_trunk_absent": True,
    })
    oracle.zero_grad(set_to_none=True)
    return cap


def _masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    selected = mask.to(prediction).expand_as(prediction)
    denominator = selected.sum()
    return float(((prediction - target.to(prediction)).abs() * selected).sum() / denominator.clamp_min(1))


def _binary_scores(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float, float]:
    prediction = prediction.bool()
    target = target.bool()
    intersection = int((prediction & target).sum())
    union = int((prediction | target).sum())
    precision = intersection / max(int(prediction.sum()), 1)
    recall = intersection / max(int(target.sum()), 1)
    iou = intersection / max(union, 1)
    return precision, recall, iou


def _bbox(binary: torch.Tensor) -> dict[str, float]:
    points = torch.nonzero(binary, as_tuple=False)
    if points.numel() == 0:
        return {"center_x": float("nan"), "center_y": float("nan"), "width": 0.0, "height": 0.0, "scale": 0.0}
    y0, x0 = points.min(0).values.tolist()
    y1, x1 = points.max(0).values.tolist()
    width, height = x1 - x0 + 1, y1 - y0 + 1
    return {
        "center_x": 0.5 * (x0 + x1), "center_y": 0.5 * (y0 + y1),
        "width": float(width), "height": float(height), "scale": float(math.sqrt(width * height)),
    }


def _ssim_value(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    # A fixed global SSIM implementation already used by the repository. The
    # mask is applied identically to both images before evaluation.
    from utils.loss_utils import ssim_loss

    expanded = mask.to(prediction).expand_as(prediction)
    white = torch.ones_like(prediction)
    pred = prediction * expanded + white * (1 - expanded)
    truth = target.to(prediction) * expanded + white * (1 - expanded)
    return 1.0 - float(ssim_loss(pred.permute(1, 2, 0), truth.permute(1, 2, 0)).detach())


def _custom_metrics(rgb: torch.Tensor, alpha: torch.Tensor, sample: Mapping[str, Any]) -> dict[str, float]:
    device = rgb.device
    edit = sample["target_edit_rgb"].to(device)
    base = sample["target_base_rgb"].to(device)
    clothing = sample["target_clothing_mask"].to(device)
    protected = sample["target_protected_mask"].to(device)
    preserve = sample["target_preserve_mask"].to(device)
    foreground = sample["target_foreground_mask"].to(device)
    base_foreground = sample["target_base_foreground_mask"].to(device)
    edit_mask = sample["target_edit_mask"].to(device)
    old_clothing = sample["target_old_clothing_mask"].to(device)
    revealed_skin = sample["target_revealed_skin_mask"].to(device)
    pred_fg = alpha >= 0.5
    truth_fg = foreground >= 0.5
    _, _, foreground_iou = _binary_scores(pred_fg, truth_fg)
    support_region = edit_mask >= 0.5
    support_prediction = pred_fg & support_region
    support_target = (clothing >= 0.5) & support_region
    support_precision, support_recall, _ = _binary_scores(support_prediction, support_target)
    new_silhouette = truth_fg & ~(base_foreground >= 0.5)
    new_silhouette_recall = float((pred_fg & new_silhouette).sum() / new_silhouette.sum().clamp_min(1))
    removal_region = (old_clothing >= 0.5) & ~truth_fg
    old_clothing_removal = float((1 - alpha)[removal_region].mean()) if removal_region.any() else 1.0
    mse = ((rgb - edit).square() * clothing.expand_as(rgb)).sum() / clothing.expand_as(rgb).sum().clamp_min(1)
    psnr = float(-10 * torch.log10(mse.clamp_min(1e-12)))
    bbox = _bbox(pred_fg[0])
    result = {
        "foreground_iou": foreground_iou,
        "clothing_support_alpha_precision": support_precision,
        "clothing_support_alpha_recall": support_recall,
        "new_silhouette_recall": new_silhouette_recall,
        "old_clothing_removal_score": old_clothing_removal,
        "masked_psnr": psnr,
        "masked_ssim": _ssim_value(rgb, edit, clothing),
        "protected_rgb_mae": _masked_l1(rgb, base, protected),
        "preserve_rgb_mae": _masked_l1(rgb, base, preserve),
        "revealed_skin_edit_mae": _masked_l1(rgb, edit, revealed_skin),
        **{f"bbox_{name}": value for name, value in bbox.items()},
    }
    return result


def _specialized_metrics(outfit_id: str, rgb: torch.Tensor, alpha: torch.Tensor, sample: Mapping[str, Any]) -> dict[str, float]:
    device = rgb.device
    edit = sample["target_edit_rgb"].to(device)
    base = sample["target_base_rgb"].to(device)
    clothing = sample["target_clothing_mask"].to(device)
    old = sample["target_old_clothing_mask"].to(device)
    revealed = sample["target_revealed_skin_mask"].to(device)
    foreground = sample["target_foreground_mask"].to(device)
    base_foreground = sample["target_base_foreground_mask"].to(device)
    height, width = alpha.shape[-2:]
    y = torch.arange(height, device=device).reshape(1, height, 1)
    x = torch.arange(width, device=device).reshape(1, 1, width)
    upper = y < height * 0.45
    lower = y > height * 0.55
    outer = (x < width * 0.35) | (x > width * 0.65)
    new_silhouette = (foreground >= 0.5) & ~(base_foreground >= 0.5)

    def recall(mask: torch.Tensor) -> float:
        return float(((alpha >= 0.5) & mask).sum() / mask.sum().clamp_min(1))

    if outfit_id == "O00":
        return {
            "old_sleeve_removal_rgb_improvement": _masked_l1(base, edit, revealed) - _masked_l1(rgb, edit, revealed),
            "short_sleeve_support_recall": recall((clothing >= 0.5) & upper & outer),
            "revealed_arm_edit_mae": _masked_l1(rgb, edit, revealed),
        }
    if outfit_id == "O01":
        return {
            "hood_support_recall": recall((clothing >= 0.5) & (y < height * 0.25)),
            "sleeve_support_recall": recall((clothing >= 0.5) & upper & outer),
            "hoodie_silhouette_recall": recall(new_silhouette),
        }
    return {
        "long_coat_support_recall": recall(clothing >= 0.5),
        "hem_extension_recall": recall(new_silhouette & lower),
        "exterior_region_recall": recall(new_silhouette),
        "old_upper_clothing_removal_rgb_improvement": _masked_l1(base, edit, (old >= 0.5) & upper) - _masked_l1(rgb, edit, (old >= 0.5) & upper),
    }


def _projection_hit(adapter: MMLPHumanAnchorDeformationAdapter, base: Any, sample: Mapping[str, Any]) -> float:
    device, dtype = base._xyz.device, base._xyz.dtype
    posed = adapter.deform_anchors(
        base.xyz_vt,
        sample["target_pose"].to(device),
        sample["target_Rh"].to(device),
        sample["target_Th"].to(device),
    )
    ones = torch.ones((posed.shape[0], 1), device=device, dtype=dtype)
    camera_points = (sample["target_camera"]["w2c"].to(device=device, dtype=dtype) @ torch.cat((posed, ones), 1).T).T[:, :3]
    pixels = (sample["target_camera"]["K"].to(device=device, dtype=dtype) @ camera_points.T).T
    z = pixels[:, 2]
    u = pixels[:, 0] / z.clamp_min(1e-8)
    v = pixels[:, 1] / z.clamp_min(1e-8)
    height, width = sample["target_foreground_mask"].shape[-2:]
    valid = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not valid.any():
        return 0.0
    mask = sample["target_foreground_mask"][0]
    ui = u[valid].round().long().clamp(0, width - 1).cpu()
    vi = v[valid].round().long().clamp(0, height - 1).cpu()
    return float((mask[vi, ui] >= 0.5).float().mean())


def _residual_statistics(output: Any, oracle: Any, config: Mapping[str, Any], base: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"channels": {}, "gates": {}}
    bounds = config["oracle"]["bounds"]
    bound_names = {
        "delta_xyz": "xyz", "delta_log_scaling": "log_scaling", "delta_rotvec": "rotation",
        "delta_opacity_logit": "opacity_logit", "delta_sh0": "sh0", "delta_shN": "shN",
    }
    for name in CHANNELS:
        tensor = getattr(output.gaussian_residuals, name).detach().float()
        flattened = tensor.abs().reshape(tensor.shape[0], -1)
        norm = torch.linalg.vector_norm(flattened, dim=1)
        result["channels"][name] = {
            "abs_p50": float(torch.quantile(flattened.reshape(-1), 0.50)),
            "abs_p95": float(torch.quantile(flattened.reshape(-1), 0.95)),
            "abs_max": float(flattened.max()),
            "norm_p50": float(torch.quantile(norm, 0.50)),
            "norm_p95": float(torch.quantile(norm, 0.95)),
            "norm_max": float(norm.max()),
            "bound_hit_fraction": float((flattened >= float(bounds[bound_names[name]]) * 0.99).float().mean()),
        }
    for name, gate in (
        ("geometry", output.gaussian_geometry_gate),
        ("appearance", output.gaussian_appearance_gate),
    ):
        value = gate.detach().float().reshape(-1)
        result["gates"][name] = {
            "mean": float(value.mean()), "p95": float(torch.quantile(value, 0.95)),
            "max": float(value.max()), "active_fraction": float((value >= 0.5).float().mean()),
        }
    xyz = output.canonical_overrides.xyz.detach()
    center = base._xyz.detach().mean(0)
    radius = torch.linalg.vector_norm(base._xyz.detach() - center, dim=1).max() * 1.25
    result["gaussian_outside_reasonable_body_radius_fraction"] = float(
        (torch.linalg.vector_norm(xyz - center, dim=1) > radius).float().mean()
    )
    result["opacity_saturation_fraction"] = result["channels"]["delta_opacity_logit"]["bound_hit_fraction"]
    result["scaling_saturation_fraction"] = result["channels"]["delta_log_scaling"]["bound_hit_fraction"]
    result["element_count"] = oracle.element_count
    result["gaussian_count"] = int(base._xyz.shape[0])
    return result


def _to_pil(tensor: torch.Tensor, channels: int) -> Image.Image:
    value = _chw(tensor.detach().cpu(), channels).clamp(0, 1)
    if channels == 1:
        return Image.fromarray((value[0].numpy() * 255 + 0.5).astype(np.uint8), "L").convert("RGB")
    return Image.fromarray((value.permute(1, 2, 0).numpy() * 255 + 0.5).astype(np.uint8), "RGB")


def _grid(path: Path, items: list[tuple[str, Image.Image]], columns: int = 4, cell: tuple[int, int] = (256, 384)) -> None:
    rows = math.ceil(len(items) / columns)
    canvas = Image.new("RGB", (cell[0] * columns, (cell[1] + 24) * rows), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(items):
        picture = image.convert("RGB").resize(cell, Image.Resampling.LANCZOS)
        x = index % columns * cell[0]
        y = index // columns * (cell[1] + 24)
        canvas.paste(picture, (x, y))
        draw.text((x + 3, y + cell[1] + 3), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _save_render_set(
    directory: Path,
    condition: str,
    rgb: torch.Tensor,
    alpha: torch.Tensor,
    sample: Mapping[str, Any],
) -> list[tuple[str, Image.Image]]:
    view_dir = directory / VIEWS[condition]
    view_dir.mkdir(parents=True, exist_ok=True)
    base = sample["target_base_rgb"]
    edit = sample["target_edit_rgb"]
    difference = (rgb.detach().cpu() - edit).abs().clamp(0, 1)
    protected = sample["target_protected_mask"].expand_as(base)
    protected_difference = difference * protected
    tensors = {
        "subject02_base.png": (base, 3), "edit_target.png": (edit, 3),
        "prediction.png": (rgb, 3), "alpha.png": (alpha, 1),
        "absolute_error.png": (difference, 3), "protected_error.png": (protected_difference, 3),
        "edit_foreground_mask.png": (sample["target_foreground_mask"], 1),
        "base_foreground_mask.png": (sample["target_base_foreground_mask"], 1),
        "clothing_mask_safe.png": (sample["target_clothing_mask"], 1),
        "edit_core_mask.png": (sample["target_edit_core_mask"], 1),
        "preserve_mask.png": (sample["target_preserve_mask"], 1),
        "protected_mask.png": (sample["target_protected_mask"], 1),
        "transition_mask.png": (sample["target_transition_mask"], 1),
        "old_clothing_mask.png": (sample["target_old_clothing_mask"], 1),
        "revealed_skin_mask.png": (sample["target_revealed_skin_mask"], 1),
    }
    for name, (tensor, channels) in tensors.items():
        save_render_tensor(view_dir / name, tensor, channels)
    return [
        (f"{VIEWS[condition]} base", _to_pil(base, 3)),
        (f"{VIEWS[condition]} edit", _to_pil(edit, 3)),
        (f"{VIEWS[condition]} pred", _to_pil(rgb, 3)),
        (f"{VIEWS[condition]} diff", _to_pil(difference, 3)),
    ]


def _save_field_panel(path: Path, base: Any, output: Any, kind: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xyz = (base._xyz if kind == "gaussian" else base.xyz_vt).detach().float().cpu()
    residual = output.gated_residuals
    fields = {
        "geometry gate": output.geometry_gate[:, 0],
        "appearance gate": output.appearance_gate[:, 0],
        "xyz norm": torch.linalg.vector_norm(residual.delta_xyz.reshape(xyz.shape[0], -1), dim=1),
        "sh0 norm": torch.linalg.vector_norm(residual.delta_sh0.reshape(xyz.shape[0], -1), dim=1),
    }
    stride = max(1, xyz.shape[0] // 20000)
    indices = torch.arange(0, xyz.shape[0], stride)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for axis, (name, values) in zip(axes.flat, fields.items()):
        score = values.detach().float().cpu()[indices]
        scatter = axis.scatter(xyz[indices, 0], xyz[indices, 1], c=score, s=1, cmap="magma")
        axis.set_title(name)
        axis.set_aspect("equal")
        axis.invert_yaxis()
        fig.colorbar(scatter, ax=axis)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _evaluate_views(
    base: Any,
    oracle: Any,
    samples: Mapping[str, Mapping[str, Any]],
    transitions: Mapping[str, torch.Tensor],
    background: torch.Tensor,
    device: torch.device,
    config: Mapping[str, Any],
    coefficient: float,
    outfit_id: str,
    output_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], Any, list[tuple[str, Image.Image]]]:
    rows, panels = [], []
    oracle.eval()
    with torch.no_grad():
        output = oracle(base)
        for condition in CONDITIONS:
            sample = samples[condition]
            rgb, alpha = _render(base, sample, output.canonical_overrides, background)
            parts = _loss(rgb, alpha, sample, transitions[condition], device, config, coefficient)
            custom = _custom_metrics(rgb, alpha, sample)
            specialized = _specialized_metrics(outfit_id, rgb, alpha, sample)
            row = {
                "condition_id": condition,
                "view": VIEWS[condition],
                **{name: float(value.detach()) for name, value in parts.items()},
                **custom,
                **specialized,
            }
            rows.append(row)
            if output_dir is not None:
                panels.extend(_save_render_set(output_dir, condition, rgb, alpha, sample))
    return rows, output, panels


def _aggregate_views(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric = [key for key, value in rows[0].items() if isinstance(value, (int, float)) and math.isfinite(float(value))]
    return {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "worst_max": float(np.max([row[key] for row in rows])),
            "best_min": float(np.min([row[key] for row in rows])),
            "std": float(np.std([row[key] for row in rows])),
        }
        for key in numeric
    }


def _parameter_group_diagnostics(oracle: Any, initial: Mapping[str, torch.Tensor], gradient_seen: Mapping[str, int]) -> dict[str, Any]:
    output = {}
    for name, value in oracle.named_parameters():
        delta = (value.detach().cpu() - initial[name]).abs()
        output[name] = {
            "requires_grad_final": value.requires_grad,
            "gradient_steps_nonzero": int(gradient_seen.get(name, 0)),
            "max_abs_update": float(delta.max()),
            "mean_abs_update": float(delta.mean()),
        }
    return output


def _save_checkpoint(
    path: Path,
    oracle: Any,
    optimizer: torch.optim.Optimizer,
    group_names: list[str],
    step: int,
    config: Mapping[str, Any],
    manifest: Path,
    outfit_id: str,
    kind: str,
    coefficient: float,
) -> None:
    save_full_training_checkpoint(
        path,
        model=oracle,
        optimizer=optimizer,
        optimizer_group_names=group_names,
        scheduler=None,
        scaler=None,
        training_state={
            "global_step": step,
            "optimizer_step": step,
            "epoch": 0,
            "batch_index": step % 4,
            "gradient_accumulation_position": 0,
            "best_metric": None,
            "sampler_state": {"round_robin": list(CONDITIONS), "next_index": step % 4},
            "stage": oracle.stage,
        },
        data_state={
            "manifest": str(manifest.resolve()),
            "manifest_sha256": _sha256(manifest),
            "outfit_id": outfit_id,
            "conditions": list(CONDITIONS),
        },
        method_state={
            "schema_version": EXPECTED_SCHEMA,
            "oracle_kind": kind,
            "config_sha256": _json_fingerprint(config),
            "transition_coefficient": coefficient,
            "shared_canonical_field": True,
        },
    )


def _resume_exact(
    checkpoint: Path,
    base: Any,
    optimizer: torch.optim.Optimizer,
    oracle: Any,
    group_names: list[str],
    config: Mapping[str, Any],
    manifest: Path,
    outfit_id: str,
    kind: str,
    coefficient: float,
    device: torch.device,
) -> tuple[Any, torch.optim.Optimizer, list[str], dict[str, Any]]:
    model_before = _tensor_state_fingerprint(oracle.state_dict().items())
    optimizer_before = _object_fingerprint(optimizer.state_dict())
    replacement = build_oracle(kind, base, config, device).to(device)
    replacement.configure_stage(oracle.stage)
    replacement_optimizer, replacement_names = _optimizer(replacement, config)
    payload = load_full_training_checkpoint(
        checkpoint,
        model=replacement,
        optimizer=replacement_optimizer,
        optimizer_group_names=replacement_names,
        scheduler=None,
        scaler=None,
        expected_data_state={
            "manifest": str(manifest.resolve()),
            "manifest_sha256": _sha256(manifest),
            "outfit_id": outfit_id,
            "conditions": list(CONDITIONS),
        },
        expected_method_state={
            "schema_version": EXPECTED_SCHEMA,
            "oracle_kind": kind,
            "config_sha256": _json_fingerprint(config),
            "transition_coefficient": coefficient,
            "shared_canonical_field": True,
        },
    )
    model_after = _tensor_state_fingerprint(replacement.state_dict().items())
    optimizer_after = _object_fingerprint(replacement_optimizer.state_dict())
    result = {
        "step": 40,
        "model_state_exact": model_before == model_after,
        "optimizer_state_exact": optimizer_before == optimizer_after,
        "global_step_exact": int(payload["training_state"]["global_step"]) == 40,
        "sampler_state_exact": payload["training_state"]["sampler_state"] == {"round_robin": list(CONDITIONS), "next_index": 0},
        "checkpoint_sha256": _sha256(checkpoint),
    }
    result["pass"] = all(value for key, value in result.items() if key.endswith("_exact"))
    if not result["pass"]:
        raise RuntimeError(f"step-40 checkpoint restoration failed: {result}")
    return replacement, replacement_optimizer, replacement_names, result


def _linear_slope(rows: list[dict[str, Any]], name: str) -> float:
    return float(np.polyfit(
        np.asarray([row["step"] for row in rows], dtype=np.float64),
        np.asarray([row[name] for row in rows], dtype=np.float64),
        1,
    )[0])


def evaluate_fit_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    if len(history) != 481 or [row["step"] for row in history] != list(range(481)):
        raise ValueError("Module 4B history must contain states 0 through 480")
    first = history[0:20]
    last = history[441:481]
    slope_rows = history[401:481]
    means = {
        "edit_first20": float(np.mean([row["edit"] for row in first])),
        "edit_last40": float(np.mean([row["edit"] for row in last])),
        "clothing_first20": float(np.mean([row["clothing"] for row in first])),
        "clothing_last40": float(np.mean([row["clothing"] for row in last])),
        "protected_first20": float(np.mean([row["protected"] for row in first])),
        "protected_last40": float(np.mean([row["protected"] for row in last])),
        "preserve_first20": float(np.mean([row["preserve"] for row in first])),
        "preserve_last40": float(np.mean([row["preserve"] for row in last])),
    }
    means["edit_reduction"] = 1 - means["edit_last40"] / max(means["edit_first20"], 1e-12)
    means["clothing_reduction"] = 1 - means["clothing_last40"] / max(means["clothing_first20"], 1e-12)
    slopes = {name: _linear_slope(slope_rows, name) for name in ("edit", "clothing", "total", "objective")}
    finite = all(
        math.isfinite(float(value))
        for row in history for key, value in row.items()
        if key not in {"condition_id", "view"} and isinstance(value, (int, float))
    )
    protected_ok = (
        means["protected_last40"] <= means["protected_first20"] * 1.10 + 1e-12
        and means["protected_last40"] < 0.005
    )
    preserve_ok = means["preserve_last40"] <= means["preserve_first20"] * 1.10 + 1e-12
    strong = (
        means["edit_reduction"] >= 0.10 and means["clothing_reduction"] >= 0.10
        and slopes["edit"] < 0 and slopes["clothing"] < 0
        and history[-1]["total"] < history[0]["total"] and protected_ok and preserve_ok and finite
    )
    partial = (
        means["edit_reduction"] >= 0.02 and means["clothing_reduction"] >= 0.02
        and slopes["edit"] < 0 and slopes["clothing"] < 0 and finite
    )
    classification = "STRONG_FIT" if strong else "PARTIAL_FIT" if partial else "NO_FIT"
    return {
        "classification": classification,
        "means": means,
        "last80_slopes": slopes,
        "finite": finite,
        "protected_ok": protected_ok,
        "preserve_ok": preserve_ok,
    }


def capacity_retention(anchor: Mapping[str, Any], gaussian: Mapping[str, Any]) -> dict[str, float]:
    values = {}
    for name in ("edit_reduction", "clothing_reduction"):
        denominator = float(gaussian["fit"]["means"][name])
        values[name] = float(anchor["fit"]["means"][name]) / denominator if denominator > 0 else float("nan")
    return values


def module4b_capacity_decision(
    gaussian_classes: Mapping[str, str],
    anchor_classes: Mapping[str, str],
    retention_sufficient: bool,
) -> str:
    gaussian_pass = all(value == "STRONG_FIT" for value in gaussian_classes.values())
    anchor_pass = all(value == "STRONG_FIT" for value in anchor_classes.values())
    if gaussian_pass and anchor_pass and retention_sufficient:
        return "A"
    if gaussian_pass and not anchor_pass:
        return "B"
    if gaussian_pass and anchor_pass and not retention_sufficient:
        return "C"
    return "D"


def _run_smoke(args: argparse.Namespace, config: Mapping[str, Any], kind: str) -> None:
    root = args.output.resolve()
    destination = root / "preflight" / f"O00_{kind}_one_step_smoke_{args.attempt_id}"
    if destination.exists():
        raise FileExistsError(f"smoke output already exists: {destination}")
    destination.mkdir(parents=True)
    _write_json(destination / "smoke_status.json", {
        "status": "RUNNING", "optimizer_steps": 0, "candidate_result": False,
    })
    device = torch.device(args.device)
    pipeline = training.load_config(args.pipeline_config)
    base = training.load_frozen_mmlphuman_base(
        pipeline["base"]["model_dir"], pipeline["base"]["checkpoint_path"], device,
    )
    samples = _load_samples(args.manifest, "O00")
    transitions = _transition_targets(samples, device)
    oracle = build_oracle(kind, base, config, device).to(device)
    oracle.configure_stage(1)
    optimizer, group_names = _optimizer(oracle, config)
    background = torch.tensor(pipeline["render"]["background"], device=device, dtype=base._xyz.dtype)
    base_before = _tensor_state_fingerprint(base.named_parameters())
    initial = {name: value.detach().clone() for name, value in oracle.named_parameters()}
    cap = _compute_transition_cap(base, oracle, samples[CONDITIONS[0]], transitions[CONDITIONS[0]], background, device, config)
    coefficient = float(cap["final_frozen_coefficient"])
    output = oracle(base)
    rgb, alpha = _render(base, samples[CONDITIONS[0]], output.canonical_overrides, background)
    parts = _loss(rgb, alpha, samples[CONDITIONS[0]], transitions[CONDITIONS[0]], device, config, coefficient)
    regularizer, _ = _regularization(output, config)
    objective = parts["total"] + regularizer
    objective.backward()
    gradient_count = sum(value.grad is not None and torch.isfinite(value.grad).all() and torch.count_nonzero(value.grad).item() > 0 for value in oracle.parameters())
    gradient_norm = torch.nn.utils.clip_grad_norm_(oracle.parameters(), float(config["oracle"]["optimizer"]["gradient_clip_norm"]))
    optimizer.step()
    checkpoint = destination / "discarded_smoke_checkpoint.pth"
    _save_checkpoint(checkpoint, oracle, optimizer, group_names, 1, config, args.manifest, "O00", kind, coefficient)
    base_after = _tensor_state_fingerprint(base.named_parameters())
    updates = {name: float((value.detach() - initial[name]).abs().max()) for name, value in oracle.named_parameters()}
    result = {
        "candidate_result": False,
        "discarded_update": True,
        "oracle_kind": kind,
        "objective": float(objective.detach()),
        "rgb_shape": list(rgb.shape),
        "alpha_shape": list(alpha.shape),
        "gaussian_residual_shapes": {name: list(getattr(output.gaussian_residuals, name).shape) for name in CHANNELS},
        "canonical_override_shapes": {name: list(value.shape) for name, value in output.canonical_overrides.as_dict().items()},
        "finite": bool(torch.isfinite(objective) and torch.isfinite(gradient_norm)),
        "nonzero_gradient_parameter_count": int(gradient_count),
        "parameter_max_updates": updates,
        "base_exact": base_before == base_after,
        "image_backbone_instantiated": False,
        "pass": bool(torch.isfinite(objective) and gradient_count > 0 and base_before == base_after),
    }
    _write_json(destination / "smoke_result.json", result)
    _write_json(destination / "smoke_status.json", {
        "status": "COMPLETE", "optimizer_steps": 1, "candidate_result": False,
        "discarded_update": True, "pass": result["pass"],
    })
    if not result["pass"]:
        raise RuntimeError(f"{kind} one-step smoke failed")


def _run_one(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    if args.outfit not in OUTFITS or args.oracle_kind not in KINDS:
        raise ValueError("run phase requires a preregistered outfit and oracle kind")
    root = args.output.resolve()
    run_dir = root / args.outfit / f"{args.oracle_kind}_oracle"
    if run_dir.exists():
        raise FileExistsError(f"candidate run directory already exists: {run_dir}")
    for name in ("checkpoints", "renders", "residuals", "gates", "visuals"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "run_status.json"
    _write_json(status_path, {"status": "RUNNING", "optimizer_steps": 0, "failure_stage": None})
    start = time.perf_counter()
    try:
        device = torch.device(args.device)
        pipeline = training.load_config(args.pipeline_config)
        base = training.load_frozen_mmlphuman_base(
            pipeline["base"]["model_dir"], pipeline["base"]["checkpoint_path"], device,
        )
        base_before = _tensor_state_fingerprint(base.named_parameters())
        samples = _load_samples(args.manifest, args.outfit)
        transitions = _transition_targets(samples, device)
        oracle = build_oracle(args.oracle_kind, base, config, device).to(device)
        oracle.configure_stage(1)
        optimizer, group_names = _optimizer(oracle, config)
        initial_parameters = {name: value.detach().cpu().clone() for name, value in oracle.named_parameters()}
        gradient_seen = {name: 0 for name, _ in oracle.named_parameters()}
        background = torch.tensor(pipeline["render"]["background"], device=device, dtype=base._xyz.dtype)
        adapter = MMLPHumanAnchorDeformationAdapter.from_mmlphuman_base(
            base, lbs_grid_path=pipeline["base"]["lbs_grid_path"],
        )
        projection_hits = {condition: _projection_hit(adapter, base, sample) for condition, sample in samples.items()}
        cap = _compute_transition_cap(base, oracle, samples[CONDITIONS[0]], transitions[CONDITIONS[0]], background, device, config)
        coefficient = float(cap["final_frozen_coefficient"])
        _write_json(run_dir / "transition_gradient_cap.json", cap)
        resolved = dict(config)
        resolved["resolved_run"] = {
            "outfit_id": args.outfit, "oracle_kind": args.oracle_kind,
            "manifest": str(args.manifest.resolve()), "pipeline_config": str(args.pipeline_config.resolve()),
            "output": str(run_dir), "transition_coefficient": coefficient,
        }
        (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
        _write_json(run_dir / "run_manifest.json", {
            "git": _git_state(), "environment": _environment(),
            "outfit_id": args.outfit, "oracle_kind": args.oracle_kind,
            "conditions": [{"condition_id": value, "view": VIEWS[value]} for value in CONDITIONS],
            "shared_canonical_field": True, "condition_specific_residuals": False,
            "image_conditioning_used": False, "teacher_used": False,
            "target_fields_used_only_after_render_for_loss_and_evaluation": True,
            "base_gaussian_count": int(base._xyz.shape[0]), "anchor_count": int(base.xyz_vt.shape[0]),
            "formal_interpolation_indices_shape": list(base.nbr_gs.shape),
            "formal_interpolation_weights_shape": list(base.nbr_gs_invdist.shape),
            "projection_foreground_hit": projection_hits,
        })
        history: list[dict[str, Any]] = []
        per_view: dict[str, Any] = {}
        resume_result = None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        for step in range(481):
            oracle.configure_stage(stage_for_state(step))
            if step in set(config["log_steps"]):
                checkpoint = run_dir / "checkpoints" / f"step_{step:06d}.pth"
                _save_checkpoint(checkpoint, oracle, optimizer, group_names, step, config, args.manifest, args.outfit, args.oracle_kind, coefficient)
                if step == 40 and args.outfit == "O00" and args.oracle_kind == "gaussian":
                    oracle, optimizer, group_names, resume_result = _resume_exact(
                        checkpoint, base, optimizer, oracle, group_names, config, args.manifest,
                        args.outfit, args.oracle_kind, coefficient, device,
                    )
                    _write_json(run_dir / "checkpoint_resume_step40.json", resume_result)
            condition = CONDITIONS[step % 4]
            sample = samples[condition]
            oracle.train(step < 480)
            optimizer.zero_grad(set_to_none=True)
            output = oracle(base)
            rgb, alpha = _render(base, sample, output.canonical_overrides, background)
            parts = _loss(rgb, alpha, sample, transitions[condition], device, config, coefficient)
            regularizer, regularizer_parts = _regularization(output, config)
            objective = parts["total"] + regularizer
            if not torch.isfinite(objective):
                raise FloatingPointError(f"non-finite objective at step {step}")
            row = {
                "step": step, "stage": oracle.stage, "condition_id": condition, "view": VIEWS[condition],
                **{name: float(value.detach()) for name, value in parts.items()},
                "regularization": float(regularizer.detach()), "objective": float(objective.detach()),
                **_custom_metrics(rgb, alpha, sample),
                **_specialized_metrics(args.outfit, rgb, alpha, sample),
            }
            history.append(row)
            _append_jsonl(run_dir / "state_records.jsonl", row)
            if step in set(config["render_steps"]):
                view_dir = run_dir / "renders" / f"step_{step:06d}"
                rows, milestone_output, panels = _evaluate_views(
                    base, oracle, samples, transitions, background, device, config,
                    coefficient, args.outfit, view_dir,
                )
                per_view[str(step)] = {"conditions": rows, "aggregate": _aggregate_views(rows)}
                _grid(run_dir / "visuals" / f"step_{step:06d}_four_view_panel.png", panels, columns=4)
                if step in set(config["visual_summary_steps"]):
                    field_panel = run_dir / "visuals" / f"step_{step:06d}_canonical_fields.png"
                    _save_field_panel(
                        field_panel,
                        base, milestone_output, args.oracle_kind,
                    )
                    shutil.copy2(field_panel, run_dir / "gates" / f"step_{step:06d}_gate_and_field_map.png")
                    shutil.copy2(field_panel, run_dir / "residuals" / f"step_{step:06d}_residual_and_gate_map.png")
                _write_json(run_dir / "residuals" / f"step_{step:06d}.json", _residual_statistics(milestone_output, oracle, config, base))
            if step == 480:
                break
            objective.backward()
            for name, value in oracle.named_parameters():
                if value.grad is not None:
                    if not torch.isfinite(value.grad).all():
                        raise FloatingPointError(f"non-finite gradient in {name} at step {step}")
                    if torch.count_nonzero(value.grad).item() > 0:
                        gradient_seen[name] += 1
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                oracle.parameters(), float(config["oracle"]["optimizer"]["gradient_clip_norm"]),
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"non-finite clipped gradient at step {step}")
            optimizer.step()
            _write_json(status_path, {
                "status": "RUNNING", "optimizer_steps": step + 1,
                "last_condition": condition, "failure_stage": None,
            })

        fit = evaluate_fit_history(history)
        final_output = oracle(base)
        residual_stats = _residual_statistics(final_output, oracle, config, base)
        residual_stats["O05_extreme_gaussian_fraction"] = max(
            value["bound_hit_fraction"] for value in residual_stats["channels"].values()
        ) if args.outfit == "O05" else None
        _write_json(run_dir / "residual_statistics.json", residual_stats)
        _write_json(run_dir / "gate_statistics.json", residual_stats["gates"])
        _write_json(run_dir / "per_view_metrics.json", per_view)
        _write_csv(run_dir / "metrics_history.csv", history)
        base_after = _tensor_state_fingerprint(base.named_parameters())
        parameter_diagnostics = _parameter_group_diagnostics(oracle, initial_parameters, gradient_seen)
        expected_trainable = {
            "raw_xyz", "raw_log_scaling", "raw_rotvec", "raw_opacity", "raw_sh0",
            "geometry_gate_logits", "appearance_gate_logits",
        }
        gradient_contract = {
            "expected_nonzero_parameters": sorted(expected_trainable),
            "all_expected_received_nonzero_gradient": all(gradient_seen[name] > 0 for name in expected_trainable),
            "disabled_shn_gradient_steps": gradient_seen["raw_shN"],
            "disabled_shn_strictly_zero": bool(torch.count_nonzero(final_output.raw_residuals.delta_shN).item() == 0),
        }
        gradient_contract["pass"] = (
            gradient_contract["all_expected_received_nonzero_gradient"]
            and gradient_contract["disabled_shn_gradient_steps"] == 0
            and gradient_contract["disabled_shn_strictly_zero"]
        )
        frozen = {
            "base_parameter_fingerprint_before": base_before,
            "base_parameter_fingerprint_after": base_after,
            "base_bitwise_exact": base_before == base_after,
            "base_parameter_with_gradient_count": sum(parameter.grad is not None for parameter in base.parameters()),
            "image_backbone_instantiated": False,
            "image_backbone_parameter_count": 0,
            "image_backbone_gradient_count": 0,
            "image_backbone_update": 0.0,
        }
        metrics = {
            "fit": fit,
            "initial_four_view": per_view["0"],
            "final_four_view": per_view["480"],
            "parameter_diagnostics": parameter_diagnostics,
            "gradient_contract": gradient_contract,
            "frozen": frozen,
            "checkpoint_resume_step40": resume_result,
            "parameter_count": sum(value.numel() for value in oracle.parameters()),
            "optimizer_steps": 480,
            "per_condition_update_count": {condition: 120 for condition in CONDITIONS},
            "elapsed_seconds": time.perf_counter() - start,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            "projection_foreground_hit": projection_hits,
            "visual_acceptance_status": "PENDING_ACTUAL_IMAGE_INSPECTION",
        }
        _write_json(run_dir / "metrics.json", metrics)
        summary_panels = []
        for step in config["visual_summary_steps"]:
            summary_panels.append((f"step {step} four-view", Image.open(run_dir / "visuals" / f"step_{step:06d}_four_view_panel.png").convert("RGB")))
            summary_panels.append((f"step {step} fields", Image.open(run_dir / "visuals" / f"step_{step:06d}_canonical_fields.png").convert("RGB")))
        _grid(run_dir / "visual_summary.png", summary_panels, columns=2, cell=(768, 720))
        status = {
            "status": "COMPLETE",
            "optimizer_steps": 480,
            "fit_classification": fit["classification"],
            "quantitative_pass": (
                fit["classification"] == "STRONG_FIT"
                and frozen["base_bitwise_exact"]
                and frozen["base_parameter_with_gradient_count"] == 0
                and gradient_contract["pass"]
            ),
            "visual_acceptance_status": "PENDING_ACTUAL_IMAGE_INSPECTION",
        }
        _write_json(status_path, status)
        _write_text(run_dir / "RUN_ACCEPTANCE.md", "\n".join([
            f"# {args.outfit} {args.oracle_kind.title()} Oracle Micro-pilot",
            "",
            f"- optimizer steps: `480`",
            f"- fit classification: `{fit['classification']}`",
            f"- frozen base exact: `{frozen['base_bitwise_exact']}`",
            "- image backbone: `not instantiated (oracle bypass)`",
            "- visual acceptance: `PENDING_ACTUAL_IMAGE_INSPECTION`",
            "",
            "This run is not a final PASS until its required images are actually opened and adjudicated.",
        ]))
    except Exception as error:
        _write_json(status_path, {
            "status": "FAILED", "failure_stage": "candidate_micro_pilot",
            "exception_type": type(error).__name__, "exception": str(error),
            "traceback": traceback.format_exc(),
        })
        raise


def _preflight(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    root = args.output.resolve()
    if root.exists():
        raise FileExistsError(f"Module 4B output root already exists: {root}")
    (root / "preflight").mkdir(parents=True)
    git = _git_state()
    if git["status_short"]:
        raise RuntimeError("Module 4B preflight requires a clean cloud worktree")
    manifest_value = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest_value.get("schema_version") != EXPECTED_DATA_SCHEMA:
        raise ValueError("fixture manifest schema is not full_dataset.v1")
    if manifest_value.get("supervision_mode") != "dual_target_region_aware_v1":
        raise ValueError("fixture does not use V5.3 dual-target supervision")
    v5_output = args.v5_3_output.resolve()
    v5_status_path = v5_output / "run_partial.json"
    if not v5_status_path.is_file():
        raise FileNotFoundError(f"V5.3 sealed status is missing: {v5_status_path}")
    v5_status = json.loads(v5_status_path.read_text(encoding="utf-8"))
    v5_outcome = v5_status.get("outcome", v5_status.get("acceptance"))
    if v5_status.get("status") != "COMPLETE" or v5_outcome != "PASS":
        raise RuntimeError(f"V5.3 sealed attempt is not COMPLETE/PASS: {v5_status}")
    for outfit in OUTFITS:
        _load_samples(args.manifest, outfit)
    pipeline = training.load_config(args.pipeline_config)
    checkpoint = Path(pipeline["base"]["model_dir"]) / pipeline["base"]["checkpoint_path"]
    lbs = Path(pipeline["base"]["lbs_grid_path"])
    device = torch.device(args.device)
    base = training.load_frozen_mmlphuman_base(str(checkpoint.parent), checkpoint.name, device)
    weights = base.nbr_gs_invdist.to(base._xyz)
    normalized = weights / weights.sum(1, keepdim=True)
    audit = {
        "git": git,
        "environment": _environment(),
        "base_gaussian_count": int(base._xyz.shape[0]),
        "anchor_count": int(base.xyz_vt.shape[0]),
        "base_shapes": {name: list(getattr(base, name).shape) for name in ("_xyz", "_scaling", "_rotation", "_opacity", "_sh0", "_shN")},
        "anchor_mapping": {
            "indices_shape": list(base.nbr_gs.shape),
            "weights_shape": list(base.nbr_gs_invdist.shape),
            "normalized_row_sum_max_abs_error": float((normalized.sum(1) - 1).abs().max()),
        },
        "stage_trainable_formal": {str(stage): sorted(values) for stage, values in STAGE_TRAINABLE.items()},
        "schedule_conflict_adjudication": {
            "requested_shorthand": "G(1-80), A(81-160), J(161-480)",
            "applied_formal_policy": "Module4A appearance(1-80), geometry+appearance(81-160), joint(161-480)",
            "reason": "the Module 4B preregistration explicitly gives Module 4A formal policy precedence",
        },
        "shn_policy": "formal Module4A effective degree 0; tensor exists, is strictly zero, and is never trainable",
        "gaussian_path": "direct one shared canonical six-field residual and two gate-logit fields per outfit",
        "anchor_path": "one shared anchor field -> gate at anchor level -> formal normalized nbr_gs interpolation -> canonical Gaussian overrides",
        "gate_order": "residuals are gated at their canonical representation level before anchor-to-Gaussian interpolation",
        "quaternion": "wxyz; local q_dressed=normalize(q_base*q_delta)",
        "target_boundary": "target images/masks enter only V5.3 loss and post-render metrics; never oracle forward",
        "image_conditioning": "not instantiated",
        "teacher": "not loaded",
    }
    _write_json(root / "preflight" / "implementation_audit.json", audit)
    _write_text(root / "preflight" / "IMPLEMENTATION_AUDIT.md", "\n".join([
        "# Module 4B Implementation Audit",
        "",
        f"- Git commit: `{git['commit']}` (clean)",
        f"- base Gaussians / anchors: `{audit['base_gaussian_count']}` / `{audit['anchor_count']}`",
        "- Gaussian oracle: one outfit-shared canonical field; no condition axis.",
        "- Anchor oracle: one outfit-shared anchor field, formal normalized interpolation.",
        "- Formal composition: wxyz, local base-times-delta quaternion.",
        "- Forward boundary: no image model, references, teacher, target RGB, or target masks.",
        "- V5.3 targets are consumed only after rendering by the loss/evaluator.",
        "- Stage conflict: Module 4A formal appearance→geometry+appearance→joint policy applied.",
        "- SHN: formal effective degree 0, retained as a strict-zero compatibility tensor.",
    ]))
    inputs = {
        "config": {"path": str(args.config.resolve()), "sha256": _sha256(args.config)},
        "pipeline_config": {"path": str(args.pipeline_config.resolve()), "sha256": _sha256(args.pipeline_config)},
        "fixture_manifest": {"path": str(args.manifest.resolve()), "sha256": _sha256(args.manifest)},
        "base_checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "lbs_grid": {"path": str(lbs), "sha256": _sha256(lbs)},
        "v5_3_fixture": "dual_target_region_aware_v1; O00/O01/O05 x four fixed conditions",
        "v5_3_sealed_attempt": {
            "path": str(v5_output),
            "status_path": str(v5_status_path),
            "status_sha256": _sha256(v5_status_path),
            "status": v5_status.get("status"),
            "acceptance": v5_outcome,
        },
    }
    _write_json(root / "input_fingerprint_manifest.json", inputs)
    _write_json(root / "method_contract.json", {
        "schema_version": EXPECTED_SCHEMA,
        "config": config,
        "implementation_audit": audit,
        "loss_contract": "exact region_aware_dual_target_loss + boundary_aware_transition_alpha_target",
        "regularization_contract": "formal Module4A residual/gate regularizers",
        "stop_conditions": [
            "non-finite objective/gradient", "frozen base change", "fixture contract failure",
            "formal mapping failure", "output collision", "step-40 restoration failure",
        ],
    })
    _write_json(root / "preflight" / "preflight_status.json", {"status": "PASS", "optimizer_steps": 0})


def _parse_visual(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    expected = {f"{outfit}/{kind}" for outfit in OUTFITS for kind in KINDS}
    if set(parsed) != expected:
        raise ValueError(f"visual decisions must cover exactly: {sorted(expected)}")
    for key, item in parsed.items():
        if item.get("grade") not in {"PASS", "WARN", "FAIL"}:
            raise ValueError(f"invalid visual grade for {key}")
        if not item.get("images_actually_opened") or not item.get("observations"):
            raise ValueError(f"visual decision for {key} lacks actual inspection evidence")
    return parsed


def _finalize(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    root = args.output.resolve()
    visual = _parse_visual(args.visual_decisions)
    all_metrics: dict[str, dict[str, Any]] = {}
    comparison_rows = []
    for outfit in OUTFITS:
        all_metrics[outfit] = {}
        for kind in KINDS:
            run_dir = root / outfit / f"{kind}_oracle"
            status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
            if status.get("status") != "COMPLETE" or status.get("optimizer_steps") != 480:
                raise RuntimeError(f"cannot finalize incomplete run: {outfit}/{kind}")
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            metrics["visual_acceptance"] = visual[f"{outfit}/{kind}"]
            metrics["visual_acceptance_status"] = visual[f"{outfit}/{kind}"]["grade"]
            _write_json(run_dir / "metrics.json", metrics)
            status["visual_acceptance_status"] = metrics["visual_acceptance_status"]
            status["final_status"] = "PASS" if status["quantitative_pass"] and metrics["visual_acceptance_status"] == "PASS" else "PARTIAL" if status["quantitative_pass"] else "FAIL"
            _write_json(run_dir / "run_status.json", status)
            all_metrics[outfit][kind] = metrics
            fit = metrics["fit"]
            comparison_rows.append({
                "outfit_id": outfit, "oracle_kind": kind,
                "fit_classification": fit["classification"],
                "visual_grade": metrics["visual_acceptance_status"],
                "edit_reduction": fit["means"]["edit_reduction"],
                "clothing_reduction": fit["means"]["clothing_reduction"],
                "edit_slope": fit["last80_slopes"]["edit"],
                "clothing_slope": fit["last80_slopes"]["clothing"],
                "protected_last40": fit["means"]["protected_last40"],
                "preserve_last40": fit["means"]["preserve_last40"],
                "parameter_count": metrics["parameter_count"],
                "elapsed_seconds": metrics["elapsed_seconds"],
                "peak_gpu_memory_bytes": metrics["peak_gpu_memory_bytes"],
            })
    retention: dict[str, Any] = {}
    grade_order = {"FAIL": 0, "WARN": 1, "PASS": 2}
    for outfit in OUTFITS:
        value = capacity_retention(all_metrics[outfit]["anchor"], all_metrics[outfit]["gaussian"])
        value["visual_grade_not_lower"] = grade_order[visual[f"{outfit}/anchor"]["grade"]] >= grade_order[visual[f"{outfit}/gaussian"]["grade"]]
        value["sufficient"] = (
            all_metrics[outfit]["anchor"]["fit"]["classification"] == "STRONG_FIT"
            and value["edit_reduction"] >= 0.75 and value["clothing_reduction"] >= 0.75
            and value["visual_grade_not_lower"]
        )
        retention[outfit] = value
    gaussian_classes = {outfit: all_metrics[outfit]["gaussian"]["fit"]["classification"] for outfit in OUTFITS}
    anchor_classes = {outfit: all_metrics[outfit]["anchor"]["fit"]["classification"] for outfit in OUTFITS}
    retention_sufficient = all(value["sufficient"] for value in retention.values())
    matrix = module4b_capacity_decision(gaussian_classes, anchor_classes, retention_sufficient)
    decision_labels = {
        "A": "gaussian_and_anchor_capacity_sufficient_continue_image_conditioned_model",
        "B": "gaussian_capacity_sufficient_anchor_bottleneck_review_before_formal_training",
        "C": "anchor_capacity_passes_but_retention_below_threshold_review_interpolation",
        "D": "gaussian_capacity_insufficient_stop_and_revise_representation_or_data",
    }
    overall_visual = "PASS" if all(item["grade"] == "PASS" for item in visual.values()) else "FAIL" if any(item["grade"] == "FAIL" for item in visual.values()) else "WARN"
    final_status = "PASS" if matrix == "A" and overall_visual == "PASS" else "PARTIAL" if matrix in {"A", "C"} and overall_visual != "FAIL" else "FAIL"
    comparison = {
        "gaussian_classifications": gaussian_classes,
        "anchor_classifications": anchor_classes,
        "capacity_retention": retention,
        "capacity_retention_sufficient": retention_sufficient,
        "decision_matrix_case": matrix,
        "decision": decision_labels[matrix],
        "visual_acceptance_status": overall_visual,
        "final_status": final_status,
        "enter_next_stage": final_status == "PASS",
    }
    _write_csv(root / "comparison" / "capacity_metrics.csv", comparison_rows)
    _write_json(root / "comparison" / "capacity_metrics.json", {"runs": comparison_rows, **comparison})
    comparison_items = []
    for outfit in OUTFITS:
        for kind in KINDS:
            comparison_items.append((f"{outfit} {kind}", Image.open(root / outfit / f"{kind}_oracle" / "visual_summary.png").convert("RGB")))
    _grid(root / "comparison" / "gaussian_vs_anchor_contact_sheet.png", comparison_items, columns=2, cell=(768, 720))
    _grid(root / "comparison" / "outfit_capacity_contact_sheet.png", comparison_items, columns=2, cell=(768, 720))
    _write_json(root / "visual_acceptance.json", {
        "inspection_method": "actual image opening through local visual inspection after artifact fetch",
        "runs": visual,
        "visual_acceptance_status": overall_visual,
    })
    _write_text(root / "VISUAL_ACCEPTANCE.md", "\n".join([
        "# Module 4B Visual Acceptance",
        "",
        f"- status: `{overall_visual}`",
        "- inspection method: actual image opening of all six step-0/final four-view sets, O05 intermediate stages, canonical fields, and comparison contact sheets.",
        "",
        *[f"- {key}: **{item['grade']}** — {item['observations']}" for key, item in sorted(visual.items())],
    ]))
    _write_json(root / "final_adjudication.json", comparison)
    _write_text(root / "FINAL_ADJUDICATION.md", "\n".join([
        "# Module 4B Final Adjudication",
        "",
        f"- final status: `{final_status}`",
        f"- decision matrix: `{matrix}`",
        f"- decision: `{decision_labels[matrix]}`",
        f"- visual acceptance: `{overall_visual}`",
        f"- capacity retention sufficient: `{retention_sufficient}`",
        f"- enter next stage: `{comparison['enter_next_stage']}`",
        "",
        "No image-conditioned formal training was started by this adjudication.",
    ]))
    _write_json(root / "run_status.json", {
        "status": "COMPLETE", "optimizer_steps_per_run": 480,
        "candidate_run_count": 6, "final_status": final_status,
        "decision_matrix_case": matrix, "enter_next_stage": comparison["enter_next_stage"],
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Module 4B canonical representation oracle micro-pilot")
    parser.add_argument("--phase", required=True, choices=("preflight", "smoke", "run", "finalize"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/oracle/module4b_canonical_capacity_v1.yaml")
    parser.add_argument("--pipeline-config", type=Path, default=PROJECT_ROOT / "configs/canon_dress_gs_mvp_real.yaml")
    parser.add_argument(
        "--v5-3-output",
        type=Path,
        default=Path("/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-DUAL-TARGET-V5-3-001/attempt_002"),
    )
    parser.add_argument("--oracle-kind", choices=KINDS)
    parser.add_argument("--outfit", choices=OUTFITS)
    parser.add_argument("--visual-decisions", default="")
    parser.add_argument("--attempt-id", default="attempt_001")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_contract(args.config)
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    random.seed(int(config["seed"]))
    if args.phase == "preflight":
        _preflight(args, config)
    elif args.phase == "smoke":
        if args.oracle_kind is None:
            raise ValueError("smoke phase requires --oracle-kind")
        _run_smoke(args, config, args.oracle_kind)
    elif args.phase == "run":
        _run_one(args, config)
    else:
        _finalize(args, config)


if __name__ == "__main__":
    main()
