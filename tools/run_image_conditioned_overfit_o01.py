from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import random
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
from scene.full_dressable_dataset import (  # noqa: E402
    DUAL_TARGET_FIELDS,
    FullDressableTrainingDataset,
)
from scene.gaussian_clothing_residuals import (  # noqa: E402
    CHANNELS,
    CanonicalGaussianOverrides,
)
from scene.support_aware_region_trusted_objective_v6 import (  # noqa: E402
    bound_normalized_regularization,
    residual_stability_loss,
)
from scene.support_aware_region_trusted_objective_v6_1 import (  # noqa: E402
    support_aware_region_trusted_objective_v6_1,
)
from tools.check_real_image_conditioned_one_batch import save_render_tensor  # noqa: E402
from tools.run_module4b_canonical_oracle_micropilot import (  # noqa: E402
    _base_gradient_count,
    _base_named_tensors,
    _render,
    _tensor_state_fingerprint,
)
from utils.anchor_graph_utils import build_surface_aware_anchor_graph  # noqa: E402
from utils.mmlphuman_anchor_deformation import MMLPHumanAnchorDeformationAdapter  # noqa: E402


EXPECTED_SCHEMA = "canondressgs.image_conditioned_overfit_o01.v1"
EXPECTED_DATA_SCHEMA = "canondressgs.full_dataset.v1"
TASK_ID = "SUBJECT02-IMAGE-CONDITIONED-OVERFIT-O01-001"
OUTFIT_ID = "O01"
CONDITIONS = ("cond_000000", "cond_000318", "cond_000017", "cond_000347")
VIEWS = {
    "cond_000000": "front",
    "cond_000318": "back",
    "cond_000017": "left",
    "cond_000347": "right",
}
FORBIDDEN_FORWARD_FIELDS = frozenset({
    "teacher", "teacher_residual", "teacher_active_mask", "target_rgb",
    "target_edit_rgb", "target_base_rgb", "target_foreground_mask",
    "target_clothing_mask", "target_pose", "target_camera", "cloth_id",
    "outfit_id", "reference_only_gate", "oracle_residual",
})
REQUIRED_RECORD_STEPS = (0, 1, 10, 20, 50, 100, 200, 400, 800, 1200, 1600, 2000)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def object_fingerprint(value: Any) -> str:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


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


def git_output(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=PROJECT_ROOT, text=True, encoding="utf-8",
    ).strip()


def environment_snapshot() -> dict[str, Any]:
    gpu = None
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory": torch.cuda.get_device_properties(0).total_memory,
        }
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu,
    }


def load_contract(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != EXPECTED_SCHEMA or config.get("task_id") != TASK_ID:
        raise ValueError("config is not the preregistered O01 image-conditioned overfit contract")
    if config.get("outfit_id") != OUTFIT_ID or tuple(config.get("conditions", ())) != CONDITIONS:
        raise ValueError("outfit or four-condition protocol differs from the frozen contract")
    training_config = config["training"]
    if tuple(training_config["round_robin"]) != CONDITIONS:
        raise ValueError("round-robin order differs from the frozen contract")
    if int(training_config["reference_count"]) != 3 or int(training_config["batch_size"]) != 1:
        raise ValueError("the micro-overfit must use three references and batch size one")
    if int(training_config["smoke_steps"]) != 20 or int(training_config["total_steps"]) != 2000:
        raise ValueError("the frozen 20/2000-step schedule changed")
    if tuple(training_config["record_steps"]) != REQUIRED_RECORD_STEPS:
        raise ValueError("record steps differ from the frozen contract")
    if config["guard"] != {
        "name": "PROTECTED_FULL_RESIDUAL_GUARD_V1",
        "version": 1,
        "membership_source": "base_derived_stable_protected",
        "application_point": "after_anchor_to_gaussian_interpolation_before_canonical_composition",
        "target_independent": True,
        "outfit_independent": True,
        "cloud_region_independent": True,
        "channels": list(CHANNELS),
    }:
        raise ValueError("protected full residual guard contract changed")
    forbidden = [name for name, enabled in config["permissions"].items() if bool(enabled)]
    if forbidden:
        raise ValueError(f"forbidden scope was enabled: {forbidden}")
    return config


def leave_one_out_reference_ids(target_condition: str) -> tuple[str, ...]:
    if target_condition not in CONDITIONS:
        raise KeyError(f"unknown target condition: {target_condition}")
    return tuple(condition for condition in CONDITIONS if condition != target_condition)


class DatasetModelContract:
    num_clothes = 1

    @staticmethod
    def get_anchor_xyz_or_none() -> None:
        return None


def _load_samples_and_episodes(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    dataset = FullDressableTrainingDataset(manifest_path, "train", reference_count=3, seed=0)
    if dataset.manifest.get("schema_version") != EXPECTED_DATA_SCHEMA:
        raise ValueError("manifest does not satisfy the full dataset v1 contract")
    if dataset.supervision_mode != "dual_target_region_aware_v1":
        raise ValueError("O01 overfit requires dual_target_region_aware_v1")
    outfits = [value for value in dataset.outfits if value["outfit_id"] == OUTFIT_ID]
    if len(outfits) != 1:
        raise ValueError("manifest must contain exactly one O01 outfit record")
    outfit = outfits[0]
    by_condition: dict[str, Any] = {}
    for observation in outfit["observations"]:
        condition = observation["condition_id"]
        if condition not in CONDITIONS:
            continue
        if condition in by_condition:
            raise ValueError(f"duplicate O01 condition: {condition}")
        missing = DUAL_TARGET_FIELDS.difference(observation)
        if missing:
            raise ValueError(f"{condition} is missing dual-target fields: {sorted(missing)}")
        target = dataset._observation(outfit, observation, True)
        sample: dict[str, Any] = {
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
            "target_condition_id": condition,
            "outfit_id": OUTFIT_ID,
            "source_record": observation,
        }
        for name in sorted(DUAL_TARGET_FIELDS.difference({"target_edit_rgb", "target_base_rgb"})):
            sample[name] = dataset._image(observation[name], 1)
        by_condition[condition] = sample
    if tuple(condition for condition in CONDITIONS if condition in by_condition) != CONDITIONS:
        raise ValueError("O01 does not contain the exact four-condition protocol")

    episodes: dict[str, Any] = {}
    protocol: dict[str, Any] = {}
    observations = {value["condition_id"]: value for value in outfit["observations"]}
    for target_condition in CONDITIONS:
        reference_ids = leave_one_out_reference_ids(target_condition)
        references = [dataset._observation(outfit, observations[value], True) for value in reference_ids]
        episode = dataset._stack_references(references)
        if target_condition in episode["reference_condition_ids"] or len(set(episode["reference_condition_ids"])) != 3:
            raise AssertionError("target/reference disjointness failed")
        episode["reference_cloth_masks"] = episode.pop("reference_clothing_masks")
        episodes[target_condition] = episode
        protocol[target_condition] = {
            "target": target_condition,
            "target_view": VIEWS[target_condition],
            "references": list(reference_ids),
            "target_in_references": False,
        }
    return by_condition, episodes, {"outfit_id": OUTFIT_ID, "episodes": protocol}


def _load_stable_protected_mask(path: Path, expected_count: int, gaussian_count: int) -> tuple[torch.Tensor, dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("pyarrow is required to read the frozen attribution parquet") from error
    table = parquet.read_table(path, columns=["gaussian_index", "base_projection_region"])
    indices = np.asarray(table.column("gaussian_index").to_numpy(), dtype=np.int64)
    regions = np.asarray(table.column("base_projection_region").to_pylist(), dtype=object)
    if len(indices) != gaussian_count or not np.array_equal(np.sort(indices), np.arange(gaussian_count)):
        raise ValueError("attribution parquet does not contain exactly one row per base Gaussian")
    protected_indices = indices[regions == "stable_protected"]
    if len(protected_indices) != expected_count:
        raise ValueError(f"stable-protected count changed: {len(protected_indices)} != {expected_count}")
    mask = torch.zeros(gaussian_count, dtype=torch.bool)
    mask[torch.from_numpy(protected_indices.copy())] = True
    return mask, {
        "row_count": int(len(indices)),
        "stable_protected_count": int(mask.sum()),
        "membership_source": "base_projection_region == stable_protected",
        "target_independent": True,
        "outfit_independent": True,
        "cloud_region_independent": True,
    }


def _model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    model = config["model"]
    optimizer = config["optimizer"]
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
        },
        "image_conditioning": {
            "embedding_dim": int(model["embedding_dim"]),
            "feature_dim": int(model["image_feature_dim"]),
            "local_feature_dim": int(model["local_feature_dim"]),
            "freeze_backbone": bool(model["freeze_image_backbone"]),
        },
        "optimizer": dict(optimizer),
    }


def _construct_model(base: Any, config: Mapping[str, Any], device: torch.device):
    model, _, anchor_features, anchor_edges = training.create_image_conditioned_components(
        DatasetModelContract(), _model_config(config), base_model=base, device=device,
    )
    model.dressable_model.configure_six_channel_decoder(config["model"]["dressable_channels"])
    graph_config = config["model"]["online_completion"]
    graph_indices, graph_weights, graph_diagnostics = build_surface_aware_anchor_graph(
        base.xyz_vt.to(device), int(graph_config["graph_k"]), base.nbr_vt.to(device),
    )
    model.initialize_online_completion(
        graph_indices, graph_weights,
        hidden_dim=int(graph_config["hidden_dim"]), num_blocks=int(graph_config["blocks"]),
    )
    optimizer = training.build_image_conditioned_optimizer(model, dict(config["optimizer"]))
    completer_parameters = [
        value for value in model.canonical_clothing_completer.parameters() if value.requires_grad
    ]
    optimizer.add_param_group({
        "name": "completer",
        "params": completer_parameters,
        "lr": float(config["optimizer"]["completer_lr"]),
    })
    return model, optimizer, anchor_features, anchor_edges, graph_diagnostics


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {name: _to_device(item, device) for name, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


def _render_sh1(base: Any, sample: Mapping[str, Any], overrides: Any, background: torch.Tensor):
    original_degree = int(base.sh_degree)
    try:
        base.sh_degree = 1
        return _render(base, sample, overrides, background)
    finally:
        base.sh_degree = original_degree


def _base_only_protected_support(
    base: Any,
    sample: Mapping[str, Any],
    stable_mask: torch.Tensor,
    background: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    opacity_mask = stable_mask.reshape(
        stable_mask.shape[0], *([1] * (base._opacity.ndim - 1))
    )
    opacity = torch.where(
        opacity_mask, base._opacity.detach(), torch.full_like(base._opacity, -100.0),
    )
    overrides = CanonicalGaussianOverrides(
        xyz=base._xyz.detach(), scaling=base._scaling.detach(), rotation=base._rotation.detach(),
        opacity=opacity, sh0=base._sh0.detach(), shN=base._shN.detach(),
    ).validate(base)
    with torch.no_grad():
        _, alpha = _render_sh1(base, sample, overrides, background)
    support = (alpha >= threshold).to(alpha)
    return support, {
        "pixel_count": int(support.sum()),
        "pixel_ratio": float(support.mean()),
        "alpha_threshold": threshold,
        "source": "stable_protected_base_gaussians_only",
    }


def _forward_inputs(episode: Mapping[str, Any], geometry: Mapping[str, Any]) -> dict[str, Any]:
    values = {
        "reference_images": episode["reference_images"],
        "reference_cloth_masks": episode["reference_cloth_masks"],
        "reference_foreground_masks": episode["reference_foreground_masks"],
        "reference_poses": episode["reference_poses"],
        "reference_cameras": episode["reference_cameras"],
        "reference_valid_mask": episode["reference_valid_mask"],
        "deformation_fn": geometry["deformation_fn"],
        "surface_depth_maps": geometry["surface_depth_maps"],
        "surface_alpha_maps": geometry["surface_alpha_maps"],
        "require_depth_visibility": True,
    }
    leaked = FORBIDDEN_FORWARD_FIELDS.intersection(values)
    if leaked:
        raise RuntimeError(f"target/oracle data leaked into formal forward: {sorted(leaked)}")
    return values


def _bounds(config: Mapping[str, Any]) -> dict[str, float]:
    channels = config["model"]["dressable_channels"]
    return {
        "xyz": float(channels["xyz"]["max_abs"]),
        "log_scaling": float(channels["log_scaling"]["max_abs"]),
        "rotation": float(channels["rotation"]["max_angle_rad"]),
        "opacity_logit": float(channels["opacity_logit"]["max_abs"]),
        "sh0": float(channels["sh0"]["max_abs"]),
        "shN": float(channels["shN"]["max_abs"]),
    }


def _loss_and_output(
    model: Any,
    base: Any,
    sample: Mapping[str, Any],
    episode: Mapping[str, Any],
    geometry: Mapping[str, Any],
    protected_mask: torch.Tensor,
    background: torch.Tensor,
    config: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], torch.Tensor, torch.Tensor, Any]:
    output = model.compute_online_six_channel_residuals(
        protected_gaussian_mask=protected_mask,
        **_forward_inputs(episode, geometry),
    )
    residuals = output["gaussian_residuals"]
    for name, value in residuals.as_dict().items():
        if value is None or not torch.isfinite(value).all():
            raise FloatingPointError(f"six-channel output {name} is absent or non-finite")
        if torch.count_nonzero(value[protected_mask]).item() != 0:
            raise AssertionError(f"protected full residual guard failed for {name}")
    overrides = model.dressable_model.compose_canonical_gaussian_overrides(residuals, CHANNELS)
    rgb, alpha = _render_sh1(base, sample, overrides, background)
    regularization, regularization_parts = bound_normalized_regularization(
        residuals, base, _bounds(config),
        garment_weight=float(config["v6_1"]["regional_regularization"]["garment_weight"]),
        protected_weight=float(config["v6_1"]["regional_regularization"]["protected_weight"]),
    )
    stability, stability_parts = residual_stability_loss(residuals, base, _bounds(config))
    objective = support_aware_region_trusted_objective_v6_1(
        rgb, alpha, sample, weights=config["v6_1"]["weights"],
        residual_loss=regularization, stability_loss=stability,
        progress_margin=float(config["v6_1"]["progress_margin"]),
        change_epsilon=float(config["v6_1"]["change_epsilon"]),
        support_diagonal_ratio=float(config["v6_1"]["support_diagonal_ratio"]),
        minimum_radius_pixels=int(config["v6_1"]["minimum_radius_pixels"]),
    )
    extras = {
        "regularization_parts": regularization_parts,
        "stability_parts": stability_parts,
        "regions": objective.regions,
    }
    return objective, output, rgb, alpha, extras


def _masked_mae(first: torch.Tensor, second: torch.Tensor, mask: torch.Tensor) -> float:
    active = (mask >= 0.5).to(first)
    while active.ndim > first.ndim and active.shape[0] == 1:
        active = active[0]
    while active.ndim < first.ndim:
        active = active.unsqueeze(0)
    if active.shape != first.shape:
        active = active.expand_as(first)
    return float(((first - second).abs() * active).sum() / active.sum().clamp_min(1))


def _residual_statistics(residuals: Any, bounds: Mapping[str, float]) -> dict[str, Any]:
    bound_names = {
        "delta_xyz": "xyz", "delta_log_scaling": "log_scaling",
        "delta_rotvec": "rotation", "delta_opacity_logit": "opacity_logit",
        "delta_sh0": "sh0", "delta_shN": "shN",
    }
    output = {}
    abnormal = torch.zeros(residuals.delta_xyz.shape[0], dtype=torch.bool, device=residuals.delta_xyz.device)
    for name, value in residuals.as_dict().items():
        absolute = value.detach().abs()
        bound = float(bounds[bound_names[name]])
        row_ratio = absolute.reshape(absolute.shape[0], -1).amax(1) / bound
        abnormal |= row_ratio >= 0.99
        output[name] = {
            "mean_abs": float(absolute.mean()), "max_abs": float(absolute.max()),
            "nonzero_ratio": float((absolute > 1e-9).float().mean()),
            "bound_saturation_ratio": float((row_ratio >= 0.99).float().mean()),
        }
    output["abnormal_gaussian_fraction"] = float(abnormal.float().mean())
    return output


def _evaluation_metrics(
    sample: Mapping[str, Any], objective: Any, output: Mapping[str, Any], rgb: torch.Tensor,
    alpha: torch.Tensor, bounds: Mapping[str, float], protected_mask: torch.Tensor,
) -> dict[str, Any]:
    target = sample["target_edit_rgb"]
    base = sample["target_base_rgb"]
    garment = torch.maximum(
        objective.regions["target_garment"],
        torch.maximum(
            objective.regions["old_garment_removal"],
            torch.maximum(objective.regions["trusted_expansion"], objective.regions["trusted_removal"]),
        ),
    )
    pred_error = _masked_mae(rgb, target, garment)
    base_error = _masked_mae(base, target, garment)
    change = (target - base).abs().mean(0, keepdim=True) >= 0.01
    closer = ((rgb - target).abs().mean(0, keepdim=True) < (base - target).abs().mean(0, keepdim=True)).float()
    closer_fraction = float((closer * change).sum() / change.sum().clamp_min(1))
    raw_target = sample["target_foreground_mask"] >= 0.5
    raw_pred = alpha >= 0.5
    intersection = (raw_target & raw_pred).sum()
    union = (raw_target | raw_pred).sum().clamp_min(1)
    completion = output["completion"]
    gate_stats = {}
    for name in ("geometry_gate", "appearance_gate", "confidence"):
        value = getattr(completion, name).detach()
        gate_stats[name] = {
            "mean": float(value.mean()), "std": float(value.std()),
            "min": float(value.min()), "max": float(value.max()),
            "saturated_ratio": float(((value <= 0.01) | (value >= 0.99)).float().mean()),
        }
    return {
        "edit_pred_mae": pred_error,
        "edit_base_mae": base_error,
        "edit_reduction": (base_error - pred_error) / max(base_error, 1e-12),
        "target_closer_fraction": closer_fraction,
        "protected_rgb_mae": _masked_mae(rgb, base, objective.regions["protected_identity"]),
        "protected_alpha_mae": _masked_mae(alpha, sample["target_base_foreground_mask"], objective.regions["protected_identity"]),
        "background_rgb_mae": _masked_mae(rgb, base, objective.regions["background"]),
        "background_alpha_mae": _masked_mae(alpha, sample["target_base_foreground_mask"], objective.regions["background"]),
        "raw_foreground_iou_report_only": float(intersection / union),
        "raw_foreground_iou_used_as_hard_gate": False,
        "protected_gaussian_count": int(protected_mask.sum()),
        "gates": gate_stats,
        "residuals": _residual_statistics(output["gaussian_residuals"], bounds),
    }


def _parameter_modules(model: Any) -> dict[str, Any]:
    mlp = model.dressable_model.anchor_clothing_mlp
    return {
        "encoder_projection": model.clothing_observation_encoder.projection_head,
        "aggregator": model.multiview_aggregator,
        "completer": model.canonical_clothing_completer,
        "hypernetwork": model.dressable_model.clothing_film_generator,
        "anchor_shared_decoder": mlp.hidden_layers,
        "local_adapter": mlp.local_feature_adapter,
        "head_xyz": mlp.output_layer,
        "head_scaling": mlp.scaling_head,
        "head_rotation": mlp.rotation_head,
        "head_opacity": mlp.opacity_head,
        "head_sh0": mlp.sh0_head,
        "head_shN": mlp.shN_head,
    }


def _gradient_snapshot(model: Any, base: Any) -> dict[str, Any]:
    values = {}
    for name, module in _parameter_modules(model).items():
        parameters = list(module.parameters())
        gradients = [value.grad for value in parameters if value.grad is not None]
        values[name] = {
            "parameter_count": sum(value.numel() for value in parameters),
            "gradient_tensor_count": len(gradients),
            "finite": all(torch.isfinite(value).all().item() for value in gradients),
            "l2": float(torch.sqrt(sum((value.detach().float().square().sum() for value in gradients), torch.tensor(0.0, device=base._xyz.device)))) if gradients else 0.0,
        }
    backbone = list(model.clothing_observation_encoder.backbone.parameters())
    values["frozen_image_backbone"] = {
        "requires_grad_count": sum(value.requires_grad for value in backbone),
        "gradient_tensor_count": sum(value.grad is not None for value in backbone),
    }
    values["frozen_mmlphuman_base"] = {"gradient_tensor_count": _base_gradient_count(base)}
    return values


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _save_checkpoint(path: Path, model: Any, optimizer: Any, step: int, metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "schema_version": "canondressgs.image_conditioned_overfit_o01.checkpoint.v1",
        "step": int(step), "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "rng": _rng_state(), "metadata": dict(metadata),
    }, temporary)
    os.replace(temporary, path)


def _state_fingerprint(state: Mapping[str, torch.Tensor]) -> str:
    return _tensor_state_fingerprint((name, value) for name, value in state.items())


def _roundtrip_check(
    checkpoint_path: Path, model: Any, optimizer: Any, base: Any, config: Mapping[str, Any],
    sample: Mapping[str, Any], episode: Mapping[str, Any], geometry: Mapping[str, Any],
    protected_mask: torch.Tensor, background: torch.Tensor, device: torch.device,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        before = _loss_and_output(model, base, sample, episode, geometry, protected_mask, background, config)
    before_state = _state_fingerprint(model.state_dict())
    before_optimizer = object_fingerprint(optimizer.state_dict())
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    restored_model, restored_optimizer, _, _, graph = _construct_model(base, config, device)
    restored_model.load_state_dict(checkpoint["model"], strict=True)
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    restored_model.eval()
    with torch.no_grad():
        after = _loss_and_output(restored_model, base, sample, episode, geometry, protected_mask, background, config)
    pairs = {
        "global_clothing_embedding": (before[1]["global_clothing_embedding"], after[1]["global_clothing_embedding"]),
        "completed_anchor_features": (before[1]["completion"].completed_anchor_features, after[1]["completion"].completed_anchor_features),
        "geometry_gate": (before[1]["completion"].geometry_gate, after[1]["completion"].geometry_gate),
        "appearance_gate": (before[1]["completion"].appearance_gate, after[1]["completion"].appearance_gate),
        "gaussian_delta_xyz": (before[1]["gaussian_residuals"].delta_xyz, after[1]["gaussian_residuals"].delta_xyz),
        "rendered_rgb": (before[2], after[2]),
        "rendered_alpha": (before[3], after[3]),
    }
    comparisons = {}
    for name, (first, second) in pairs.items():
        difference = (first - second).abs()
        tolerance = 1e-6 if name.startswith("rendered") else 1e-7
        comparisons[name] = {
            "max_abs_diff": float(difference.max()), "mean_abs_diff": float(difference.mean()),
            "atol": tolerance, "rtol": 0.0,
            "allclose": bool(torch.allclose(first, second, atol=tolerance, rtol=0)),
        }
    restored_state = _state_fingerprint(restored_model.state_dict())
    restored_optimizer_fingerprint = object_fingerprint(restored_optimizer.state_dict())
    _restore_rng(checkpoint["rng"])
    rng_restored_exact = object_fingerprint(_rng_state()) == object_fingerprint(checkpoint["rng"])
    result = {
        "checkpoint_step": int(checkpoint["step"]),
        "model_state_bitwise": before_state == restored_state,
        "optimizer_state_exact": before_optimizer == restored_optimizer_fingerprint,
        "rng_state_exact": rng_restored_exact,
        "graph_fingerprint": graph["fingerprint"],
        "comparisons": comparisons,
    }
    result["pass"] = bool(
        result["model_state_bitwise"] and result["optimizer_state_exact"] and result["rng_state_exact"]
        and all(value["allclose"] for value in comparisons.values())
    )
    del restored_model, restored_optimizer, before, after
    torch.cuda.empty_cache()
    model.train()
    return result


def _tensor_to_pil(tensor: torch.Tensor, channels: int) -> Image.Image:
    value = tensor.detach().float().clamp(0, 1).cpu()
    if value.ndim == 3 and value.shape[0] == channels:
        value = value.permute(1, 2, 0)
    if channels == 1:
        value = value[..., 0]
    array = (value.numpy() * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(array, "RGB" if channels == 3 else "L")


def _save_contact_sheet(path: Path, rows: list[tuple[str, list[tuple[str, torch.Tensor, int]]]]) -> None:
    thumb_width, thumb_height, label_height = 256, 384, 28
    columns = max(len(images) for _, images in rows)
    canvas = Image.new("RGB", (columns * thumb_width, len(rows) * (thumb_height + label_height) + 36), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, (row_label, images) in enumerate(rows):
        y = 36 + row_index * (thumb_height + label_height)
        draw.text((4, y - 28), row_label, fill="black")
        for column, (label, tensor, channels) in enumerate(images):
            image = _tensor_to_pil(tensor, channels).convert("RGB")
            image.thumbnail((thumb_width, thumb_height))
            x = column * thumb_width + (thumb_width - image.width) // 2
            canvas.paste(image, (x, y + (thumb_height - image.height) // 2))
            draw.text((column * thumb_width + 4, y + thumb_height + 4), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _evaluate_all(
    step: int, model: Any, base: Any, samples: Mapping[str, Any], episodes: Mapping[str, Any],
    geometries: Mapping[str, Any], protected_mask: torch.Tensor, background: torch.Tensor,
    config: Mapping[str, Any], output_dir: Path,
) -> dict[str, Any]:
    model.eval()
    per_episode = {}
    rows = []
    milestone = output_dir / "milestones" / f"step_{step:06d}"
    milestone.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for condition in CONDITIONS:
            objective, output, rgb, alpha, _ = _loss_and_output(
                model, base, samples[condition], episodes[condition], geometries[condition],
                protected_mask, background, config,
            )
            metrics = _evaluation_metrics(
                samples[condition], objective, output, rgb, alpha, _bounds(config), protected_mask,
            )
            metrics["loss"] = float(objective.total)
            metrics["parts"] = {name: float(value) for name, value in objective.parts.items()}
            per_episode[condition] = metrics
            save_render_tensor(milestone / f"{condition}_predicted_rgb.png", rgb, 3)
            save_render_tensor(milestone / f"{condition}_predicted_alpha.png", alpha, 1)
            rows.append((condition, [
                (f"ref {episodes[condition]['reference_condition_ids'][0]}", episodes[condition]["reference_images"][0], 3),
                (f"ref {episodes[condition]['reference_condition_ids'][1]}", episodes[condition]["reference_images"][1], 3),
                (f"ref {episodes[condition]['reference_condition_ids'][2]}", episodes[condition]["reference_images"][2], 3),
                ("base", samples[condition]["target_base_rgb"], 3),
                ("target", samples[condition]["target_edit_rgb"], 3),
                ("prediction", rgb, 3),
                ("abs error", (rgb - samples[condition]["target_edit_rgb"]).abs(), 3),
                ("alpha", alpha, 1),
            ]))
    _save_contact_sheet(milestone / "comparison.png", rows)
    summary = {
        "step": step, "episodes": per_episode,
        "trainable_parameter_fingerprint": _state_fingerprint(model.state_dict()),
        "mean": {
            name: float(np.mean([value[name] for value in per_episode.values()]))
            for name in (
                "edit_pred_mae", "edit_base_mae", "edit_reduction", "target_closer_fraction",
                "protected_rgb_mae", "protected_alpha_mae", "background_rgb_mae",
                "background_alpha_mae", "raw_foreground_iou_report_only",
            )
        },
        "mean_abnormal_gaussian_fraction": float(np.mean([
            value["residuals"]["abnormal_gaussian_fraction"] for value in per_episode.values()
        ])),
    }
    atomic_json(milestone / "metrics.json", summary)
    model.train()
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _final_learning_decision(history: list[dict[str, Any]], final: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    first = [row["loss"] for row in history[:100]]
    last = [row["loss"] for row in history[-200:]]
    if len(first) != 100 or len(last) != 200:
        raise RuntimeError("loss-window acceptance requires all 2000 step records")
    first_mean, last_mean = float(np.mean(first)), float(np.mean(last))
    drop = (first_mean - last_mean) / max(abs(first_mean), 1e-12)
    mean = final["mean"]
    acceptance = config["acceptance"]
    checks = {
        "loss_drop_fraction": drop >= float(acceptance["loss_drop_fraction_min"]),
        "mean_edit_reduction": mean["edit_reduction"] >= float(acceptance["mean_edit_reduction_min"]),
        "mean_target_closer": mean["target_closer_fraction"] >= float(acceptance["mean_target_closer_min"]),
        "protected_rgb_mae": mean["protected_rgb_mae"] <= float(acceptance["protected_mae_max"]),
        "background_rgb_mae": mean["background_rgb_mae"] <= float(acceptance["background_leakage_max"]),
        "abnormal_gaussian_fraction": final["mean_abnormal_gaussian_fraction"] <= float(acceptance["abnormal_gaussian_fraction_max"]),
    }
    for name in ("geometry_gate", "appearance_gate"):
        gate_values = [episode["gates"][name] for episode in final["episodes"].values()]
        checks[f"{name}_nonconstant_nonsaturated"] = (
            float(np.mean([value["std"] for value in gate_values])) > 1e-6
            and float(np.mean([value["saturated_ratio"] for value in gate_values])) < 0.999
        )
    residual_nonzero = [
        episode["residuals"][channel]["nonzero_ratio"]
        for episode in final["episodes"].values() for channel in CHANNELS
    ]
    checks["all_residual_channels_nonzero"] = min(residual_nonzero) > 0
    return {
        "first_100_mean_loss": first_mean, "last_200_mean_loss": last_mean,
        "loss_drop_fraction": drop, "checks": checks, "pass": all(checks.values()),
    }


def _checkpoint_metadata(config: Mapping[str, Any], graph: Mapping[str, Any], step: int) -> dict[str, Any]:
    return {
        "task_id": TASK_ID, "step": step, "git_commit": git_output("rev-parse", "HEAD"),
        "config_fingerprint": json_fingerprint(config), "graph_fingerprint": graph["fingerprint"],
        "schedule": "four_episode_leave_one_out_round_robin", "outfit": OUTFIT_ID,
        "target_forward_leakage": False, "guard": config["guard"],
    }


def run(args: argparse.Namespace) -> None:
    config_path = args.config.resolve()
    config = load_contract(config_path)
    output_dir = Path(args.output_dir or config["training"]["output_dir"])
    if output_dir.exists():
        raise FileExistsError(f"append-only output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    start = time.time()
    status = {"task_id": TASK_ID, "status": "RUNNING", "failure_stage": None, "started_at": start}
    atomic_json(output_dir / "RUN_STATUS.json", status)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("formal O01 overfit requires CUDA")
        device = torch.device("cuda")
        seed = int(config["training"]["seed"])
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats()
        commit = git_output("rev-parse", "HEAD")
        branch = git_output("branch", "--show-current")
        dirty = git_output("status", "--short")
        if branch != config["branch"] or dirty:
            raise RuntimeError(f"formal execution requires clean {config['branch']}; branch={branch} dirty={bool(dirty)}")
        manifest = Path(config["inputs"]["manifest"])
        attribution = Path(config["inputs"]["stable_protected_attribution"])
        checkpoint = Path(config["base"]["model_dir"]) / config["base"]["checkpoint_path"]
        paths = {
            "config": config_path, "manifest": manifest, "stable_protected_attribution": attribution,
            "base_checkpoint": checkpoint, "lbs_grid": Path(config["base"]["lbs_grid_path"]),
        }
        expected_hashes = {
            "manifest": config["inputs"]["manifest_sha256"],
            "stable_protected_attribution": config["inputs"]["stable_protected_attribution_sha256"],
            "base_checkpoint": config["base"]["checkpoint_sha256"],
        }
        input_hashes = {}
        for name, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(path)
            if name in expected_hashes:
                input_hashes[name] = sha256_file(path)
                if input_hashes[name] != expected_hashes[name]:
                    raise ValueError(f"{name} SHA256 differs from preregistration")
        (output_dir / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        atomic_json(output_dir / "environment.json", environment_snapshot())
        atomic_text(output_dir / "command.txt", " ".join([sys.executable, *sys.argv]))

        status["failure_stage"] = "load_data_and_base"
        atomic_json(output_dir / "RUN_STATUS.json", status)
        samples, episodes, protocol = _load_samples_and_episodes(manifest)
        base = training.load_frozen_mmlphuman_base(
            config["base"]["model_dir"], checkpoint, device=device,
        )
        if int(base._xyz.shape[0]) != int(config["base"]["gaussian_count"]):
            raise ValueError("base Gaussian count differs from contract")
        protected_cpu, attribution_diagnostics = _load_stable_protected_mask(
            attribution, int(config["inputs"]["stable_protected_count"]), int(base._xyz.shape[0]),
        )
        protected_mask = protected_cpu.to(device)
        base_fingerprint_before = _tensor_state_fingerprint(_base_named_tensors(base))
        model, optimizer, _, _, graph = _construct_model(base, config, device)
        backbone_fingerprint_before = _state_fingerprint(
            model.clothing_observation_encoder.backbone.state_dict()
        )
        adapter = MMLPHumanAnchorDeformationAdapter.from_mmlphuman_base(
            base, lbs_grid_path=Path(config["base"]["lbs_grid_path"]),
        )
        background = torch.tensor(config["render"]["background"], device=device, dtype=base._xyz.dtype)

        status["failure_stage"] = "precompute_reference_geometry_and_protected_support"
        atomic_json(output_dir / "RUN_STATUS.json", status)
        geometries, support_diagnostics = {}, {}
        for condition in CONDITIONS:
            episodes[condition] = _to_device(episodes[condition], device)
            samples[condition] = _to_device(samples[condition], device)
            geometries[condition] = training.prepare_real_reference_geometry(
                base, adapter, episodes[condition], background,
            )
            support, diagnostics = _base_only_protected_support(
                base, samples[condition], protected_mask, background,
                float(config["render"]["protected_support_alpha_threshold"]),
            )
            samples[condition]["target_protected_mask"] = torch.maximum(
                samples[condition]["target_protected_mask"], support,
            )
            support_diagnostics[condition] = diagnostics

        input_manifest = {
            "task_id": TASK_ID, "git": {"branch": branch, "commit": commit, "clean": True},
            "paths": {name: str(path) for name, path in paths.items()}, "sha256": input_hashes,
            "historical_checkpoint_warning": {
                "historical_sha256": config["base"]["historical_checkpoint_sha256_warning"],
                "current_sha256": config["base"]["checkpoint_sha256"],
                "same_file": False,
            },
            "protocol": protocol, "target_view_used_in_forward": False,
            "attribution": attribution_diagnostics, "protected_support": support_diagnostics,
            "graph": graph, "config_fingerprint": json_fingerprint(config),
        }
        atomic_json(output_dir / "input_manifest.json", input_manifest)

        status["failure_stage"] = "step_000000_evaluation"
        atomic_json(output_dir / "RUN_STATUS.json", status)
        milestones = {"0": _evaluate_all(
            0, model, base, samples, episodes, geometries, protected_mask, background, config, output_dir,
        )}
        history: list[dict[str, Any]] = []
        cumulative_gradient_nonzero = {name: False for name in _parameter_modules(model)}
        model.train()
        total_steps = int(config["training"]["total_steps"])
        record_steps = set(int(value) for value in config["training"]["record_steps"])
        checkpoint_steps = set(int(value) for value in config["training"]["checkpoint_steps"])
        for step in range(1, total_steps + 1):
            condition = CONDITIONS[(step - 1) % len(CONDITIONS)]
            status["failure_stage"] = f"optimization_step_{step:06d}"
            if step == 1 or step % 25 == 0:
                status["current_step"] = step
                atomic_json(output_dir / "RUN_STATUS.json", status)
            step_start = time.time()
            optimizer.zero_grad(set_to_none=True)
            objective, output, rgb, alpha, extras = _loss_and_output(
                model, base, samples[condition], episodes[condition], geometries[condition],
                protected_mask, background, config,
            )
            objective.total.backward()
            gradient = _gradient_snapshot(model, base)
            for name in cumulative_gradient_nonzero:
                cumulative_gradient_nonzero[name] |= gradient[name]["l2"] > 0 and gradient[name]["finite"]
            if gradient["frozen_image_backbone"]["gradient_tensor_count"] != 0:
                raise AssertionError("frozen image backbone received gradients")
            if gradient["frozen_mmlphuman_base"]["gradient_tensor_count"] != 0:
                raise AssertionError("frozen MMLP-Human base received gradients")
            trainable = [value for group in optimizer.param_groups for value in group["params"]]
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable, float(config["training"]["gradient_clip_norm"]), error_if_nonfinite=True,
            )
            optimizer.step()
            row = {
                "step": step, "condition": condition, "view": VIEWS[condition],
                "loss": float(objective.total.detach()),
                **{f"loss_{name}": float(value.detach()) for name, value in objective.parts.items()},
                "gradient_norm_pre_clip": float(gradient_norm),
                "elapsed_seconds": time.time() - step_start,
                "memory_allocated_bytes": torch.cuda.memory_allocated(),
                "memory_reserved_bytes": torch.cuda.memory_reserved(),
            }
            if not all(np.isfinite(value) for name, value in row.items() if isinstance(value, float)):
                raise FloatingPointError("training record contains NaN or Inf")
            history.append(row)
            append_jsonl(output_dir / "training.jsonl", row)

            if step in checkpoint_steps:
                checkpoint_path = output_dir / "checkpoints" / f"checkpoint_step_{step:06d}.pth"
                checkpoint_metadata = _checkpoint_metadata(config, graph, step)
                checkpoint_metadata.update({
                    "trainable_parameter_fingerprint": _state_fingerprint(model.state_dict()),
                    "optimizer_fingerprint": object_fingerprint(optimizer.state_dict()),
                })
                _save_checkpoint(checkpoint_path, model, optimizer, step, checkpoint_metadata)
            if step == int(config["training"]["smoke_steps"]):
                missing_gradients = sorted(name for name, passed in cumulative_gradient_nonzero.items() if not passed)
                roundtrip = _roundtrip_check(
                    output_dir / "checkpoints" / "checkpoint_step_000020.pth",
                    model, optimizer, base, config, samples[CONDITIONS[0]], episodes[CONDITIONS[0]],
                    geometries[CONDITIONS[0]], protected_mask, background, device,
                )
                smoke = {
                    "steps": step, "all_losses_finite": True,
                    "trainable_modules_cumulative_nonzero_grad": cumulative_gradient_nonzero,
                    "missing_nonzero_gradient_modules": missing_gradients,
                    "frozen_image_backbone_gradient_count": gradient["frozen_image_backbone"]["gradient_tensor_count"],
                    "frozen_base_gradient_count": gradient["frozen_mmlphuman_base"]["gradient_tensor_count"],
                    "target_view_used_in_forward": False, "checkpoint_roundtrip": roundtrip,
                    "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                }
                smoke["pass"] = not missing_gradients and roundtrip["pass"]
                atomic_json(output_dir / "SMOKE_ACCEPTANCE.json", smoke)
                if not smoke["pass"]:
                    raise RuntimeError(f"20-step smoke failed: missing_gradients={missing_gradients}, roundtrip={roundtrip['pass']}")
            if step in record_steps:
                milestones[str(step)] = _evaluate_all(
                    step, model, base, samples, episodes, geometries, protected_mask, background, config, output_dir,
                )
                atomic_json(output_dir / "milestones.json", milestones)

        status["failure_stage"] = "final_acceptance"
        atomic_json(output_dir / "RUN_STATUS.json", status)
        final = milestones[str(total_steps)]
        learning = _final_learning_decision(history, final, config)
        base_fingerprint_after = _tensor_state_fingerprint(_base_named_tensors(base))
        backbone_fingerprint_after = _state_fingerprint(
            model.clothing_observation_encoder.backbone.state_dict()
        )
        base_check = {
            "before": base_fingerprint_before, "after": base_fingerprint_after,
            "bitwise_unchanged": base_fingerprint_before == base_fingerprint_after,
            "gradient_tensor_count": _base_gradient_count(base),
        }
        engineering_checks = {
            "completed_2000_steps": len(history) == 2000 and history[-1]["step"] == 2000,
            "all_losses_finite": all(np.isfinite(row["loss"]) for row in history),
            "frozen_base_bitwise": base_check["bitwise_unchanged"],
            "frozen_base_gradients_zero": base_check["gradient_tensor_count"] == 0,
            "frozen_backbone_gradients_zero": gradient["frozen_image_backbone"]["gradient_tensor_count"] == 0,
            "frozen_backbone_bitwise": backbone_fingerprint_before == backbone_fingerprint_after,
            "checkpoint_roundtrip": json.loads((output_dir / "SMOKE_ACCEPTANCE.json").read_text())["checkpoint_roundtrip"]["pass"],
            "target_not_in_forward": True,
            "all_four_episodes": set(row["condition"] for row in history) == set(CONDITIONS),
            "guard_protected_count_exact": int(protected_mask.sum()) == int(config["inputs"]["stable_protected_count"]),
        }
        engineering_pass = all(engineering_checks.values())
        result = {
            "task_id": TASK_ID, "training_steps": len(history),
            "duration_seconds": time.time() - start,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "engineering": {"checks": engineering_checks, "pass": engineering_pass},
            "learning": learning, "final_evaluation": final,
            "base_freeze": base_check, "gradient_final": gradient,
            "backbone_freeze": {
                "before": backbone_fingerprint_before, "after": backbone_fingerprint_after,
                "bitwise_unchanged": backbone_fingerprint_before == backbone_fingerprint_after,
                "gradient_tensor_count": gradient["frozen_image_backbone"]["gradient_tensor_count"],
            },
            "visual_status": "PENDING_ACTUAL_INSPECTION",
            "preliminary_status": "AWAITING_VISUAL_ACCEPTANCE" if engineering_pass and learning["pass"] else "FAIL",
        }
        atomic_json(output_dir / "metrics.json", result)
        _write_csv(output_dir / "training_curve.csv", history)
        final_sheet = output_dir / "milestones" / "step_002000" / "comparison.png"
        (output_dir / "visual_acceptance_contact_sheet.png").write_bytes(final_sheet.read_bytes())
        atomic_text(output_dir / "RUN_REPORT.md", "\n".join([
            f"# {TASK_ID}", "", f"- Commit: `{commit}`", f"- Steps: {len(history)}",
            f"- Engineering: **{'PASS' if engineering_pass else 'FAIL'}**",
            f"- Quantitative learning: **{'PASS' if learning['pass'] else 'FAIL'}**",
            "- Visual acceptance: **PENDING_ACTUAL_INSPECTION**",
            "- Forward inputs: reference RGB/foreground/clothing masks, pose/camera only",
            "- Target RGB/masks: consumed only after render by V6.1 loss/evaluation",
            "- cloth_id / teacher / oracle residual / target-view conditioning: not used",
            "- Raw foreground IoU: report-only, never a hard gate",
        ]))
        status.update({
            "status": result["preliminary_status"], "failure_stage": None,
            "completed_step": total_steps, "finished_at": time.time(),
        })
        atomic_json(output_dir / "RUN_STATUS.json", status)
    except Exception as error:
        status.update({
            "status": "FAIL", "finished_at": time.time(),
            "exception_type": type(error).__name__, "exception": str(error),
            "traceback": traceback.format_exc(),
        })
        atomic_json(output_dir / "RUN_STATUS.json", status)
        raise


def finalize_visual(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    metrics_path = output_dir / "metrics.json"
    contact_sheet = output_dir / "visual_acceptance_contact_sheet.png"
    if not metrics_path.exists() or not contact_sheet.exists():
        raise FileNotFoundError("cannot seal visual acceptance without completed metrics and contact sheet")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics["preliminary_status"] not in {"AWAITING_VISUAL_ACCEPTANCE", "FAIL"}:
        raise RuntimeError("visual acceptance is already sealed")
    if args.visual_status not in {"PASS", "WARN", "FAIL"}:
        raise ValueError("--visual-status must be PASS, WARN, or FAIL")
    if len(args.visual_note) < 4:
        raise ValueError("actual visual inspection requires at least four concrete observations")
    visual = {
        "images_actually_opened": [str(contact_sheet)],
        "inspection_method": "Codex local image viewer, full contact sheet plus milestone render panels",
        "observations": args.visual_note, "status": args.visual_status,
        "purple_hoodie_dominance_resolved": bool(args.purple_hoodie_resolved),
        "identity_retained": bool(args.identity_retained),
        "clear_o01_view_count": int(args.clear_o01_view_count),
        "mottled_artifact": bool(args.mottled_artifact),
    }
    visual["pass"] = (
        args.visual_status in {"PASS", "WARN"}
        and visual["purple_hoodie_dominance_resolved"] and visual["identity_retained"]
        and visual["clear_o01_view_count"] >= 3 and not visual["mottled_artifact"]
    )
    final_pass = bool(metrics["engineering"]["pass"] and metrics["learning"]["pass"] and visual["pass"])
    final_status = "IMAGE_CONDITIONED_O01_OVERFIT_PASS" if final_pass else "FAIL"
    metrics["visual_acceptance"] = visual
    metrics["visual_status"] = args.visual_status
    metrics["final_status"] = final_status
    metrics["next_task"] = (
        "RUN_TWO_OUTFIT_DISCRIMINATION_O01_O08" if final_pass
        else "DIAGNOSE_IMAGE_CONDITIONED_OVERFIT_FAILURE"
    )
    atomic_json(metrics_path, metrics)
    atomic_json(output_dir / "visual_acceptance.json", visual)
    atomic_text(output_dir / "VISUAL_ACCEPTANCE.md", "\n".join([
        "# O01 Visual Acceptance", "", f"- Status: **{args.visual_status}**",
        f"- Clear O01 target views: {args.clear_o01_view_count}/4",
        f"- Purple hoodie dominance resolved: {args.purple_hoodie_resolved}",
        f"- Identity retained: {args.identity_retained}",
        f"- Mottled artifact: {args.mottled_artifact}", "", "## Concrete observations", "",
        *[f"- {note}" for note in args.visual_note],
    ]))
    atomic_text(output_dir / "GATE_ACCEPTANCE.md", "\n".join([
        f"# {TASK_ID} Final Acceptance", "", f"- Final status: **{final_status}**",
        f"- Engineering: **{'PASS' if metrics['engineering']['pass'] else 'FAIL'}**",
        f"- Learning: **{'PASS' if metrics['learning']['pass'] else 'FAIL'}**",
        f"- Visual: **{args.visual_status}**", f"- Next task: `{metrics['next_task']}`",
        "", "This run is O01 four-view leave-one-out overfit evidence only; it is not multi-outfit generalization evidence.",
    ]))
    status = json.loads((output_dir / "RUN_STATUS.json").read_text(encoding="utf-8"))
    status.update({"status": final_status, "visual_status": args.visual_status, "finished_at": time.time()})
    atomic_json(output_dir / "RUN_STATUS.json", status)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preregistered O01 minimal image-conditioned six-channel overfit")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/sprint/subject02_image_conditioned_overfit_o01_v1.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--finalize-visual", action="store_true")
    parser.add_argument("--visual-status", choices=("PASS", "WARN", "FAIL"))
    parser.add_argument("--visual-note", action="append", default=[])
    parser.add_argument("--clear-o01-view-count", type=int, default=0)
    parser.add_argument("--purple-hoodie-resolved", action="store_true")
    parser.add_argument("--identity-retained", action="store_true")
    parser.add_argument("--mottled-artifact", action="store_true")
    args = parser.parse_args()
    if args.finalize_visual:
        if args.output_dir is None:
            parser.error("--finalize-visual requires --output-dir")
        finalize_visual(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
