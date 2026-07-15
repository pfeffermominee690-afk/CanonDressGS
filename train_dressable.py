from __future__ import annotations

import argparse
import hashlib
import glob
import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from scene.anchor_image_projector import AnchorImageProjector
from scene.clothing_observation_encoder import ClothingObservationEncoder
from scene.dressable_dataset import (
    AnchorOffsetDataset,
    ImageConditionedEpisodeDataset,
    RenderingDressableDataset,
    image_conditioned_episode_collate,
)
from scene.dressable_gaussian_model import DressableGaussianModel
from scene.image_conditioned_dressable_model import ImageConditionedDressableModel
from scene.multiview_clothing_aggregator import MultiViewClothingAggregator
from utils.clothing_loss_utils import (
    anchor_feature_consistency_loss,
    anchor_graph_smoothness_loss,
    anchor_offset_supervision_loss,
    clothing_embedding_consistency_loss,
    clothing_offset_consistency_loss,
    non_clothing_region_loss,
    offset_magnitude_regularization,
)
from utils.gaussian_alignment import save_anchor_offset_target
from utils.dressable_camera_utils import build_mmlphuman_camera
from utils.dressable_checkpoint_utils import (
    freeze_base_model,
    load_frozen_mmlphuman_base,
)
from utils.rendering_loss_utils import LPIPSLoss, combined_rendering_loss
from utils.mmlphuman_anchor_deformation import MMLPHumanAnchorDeformationAdapter
from utils.mmlphuman_anchor_visibility import render_mmlphuman_expected_depth
from utils.mmlphuman_state_utils import mmlphuman_state_transaction


class TargetOnlyGaussianModel(nn.Module):
    """Minimal raw Gaussian container for anchor-only clothing supervision."""

    def __init__(self, anchor_xyz: torch.Tensor) -> None:
        super().__init__()
        if anchor_xyz.ndim != 2 or anchor_xyz.shape[1] != 3:
            raise ValueError("anchor_xyz must have shape [A, 3]")
        num_gaussians = anchor_xyz.shape[0]
        self._xyz = nn.Parameter(anchor_xyz.detach().clone())
        self._scaling = nn.Parameter(anchor_xyz.new_zeros(num_gaussians, 3))
        self._opacity = nn.Parameter(anchor_xyz.new_zeros(num_gaussians))
        rotation = anchor_xyz.new_zeros(num_gaussians, 4)
        rotation[:, 0] = 1
        self._rotation = nn.Parameter(rotation)
        self._sh0 = nn.Parameter(anchor_xyz.new_zeros(num_gaussians, 1, 3))
        self._shN = nn.Parameter(anchor_xyz.new_zeros(num_gaussians, 3, 3))


def build_anchor_features(
    anchor_xyz: torch.Tensor,
    num_frequencies: int = 4,
) -> torch.Tensor:
    """Build centered normalized xyz plus Fourier positional encoding."""

    if not isinstance(anchor_xyz, torch.Tensor) or anchor_xyz.ndim != 2 or anchor_xyz.shape[1] != 3:
        raise ValueError("anchor_xyz must have shape [A, 3]")
    if not isinstance(num_frequencies, int) or num_frequencies < 0:
        raise ValueError("num_frequencies must be a non-negative integer")
    if not torch.isfinite(anchor_xyz).all():
        raise ValueError("anchor_xyz contains NaN or Inf")
    centered = anchor_xyz - anchor_xyz.mean(dim=0, keepdim=True)
    normalization = torch.linalg.vector_norm(centered, dim=-1).max().clamp_min(1e-8)
    normalized = centered / normalization
    features = [normalized]
    for level in range(num_frequencies):
        phase = (2**level) * torch.pi * normalized
        features.extend((torch.sin(phase), torch.cos(phase)))
    return torch.cat(features, dim=-1)


def build_anchor_knn_edges(anchor_xyz: torch.Tensor, k: int = 4) -> torch.Tensor:
    """Build unique undirected kNN anchor edges without retaining dense distances."""

    if anchor_xyz.ndim != 2 or anchor_xyz.shape[1] != 3:
        raise ValueError("anchor_xyz must have shape [A, 3]")
    num_anchors = anchor_xyz.shape[0]
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0 or k >= num_anchors:
        raise ValueError(f"k must satisfy 0 < k < A={num_anchors}")
    distances = torch.cdist(anchor_xyz, anchor_xyz)
    distances.fill_diagonal_(float("inf"))
    neighbors = torch.topk(distances, k=k, largest=False, dim=1).indices.detach().cpu()
    edges = {
        (min(anchor_id, int(neighbor)), max(anchor_id, int(neighbor)))
        for anchor_id in range(num_anchors)
        for neighbor in neighbors[anchor_id]
        if int(neighbor) != anchor_id
    }
    return torch.tensor(sorted(edges), dtype=torch.long, device=anchor_xyz.device)


def validate_batch_size(batch_size: int) -> None:
    if batch_size != 1:
        raise ValueError(
            "CanonDressGS anchor training currently requires batch_size=1; "
            "batched cloth_id is not implemented"
        )


def create_training_components(
    dataset: AnchorOffsetDataset,
    config: dict[str, Any],
    device: torch.device | str = "cpu",
) -> tuple[DressableGaussianModel, torch.optim.Optimizer, torch.Tensor, torch.Tensor]:
    """Create target-only anchor_film model, optimizer, features, and graph."""

    device = torch.device(device)
    model_config = config["model"]
    if model_config.get("offset_mode") != "anchor_film":
        raise ValueError("the first train_dressable version only supports offset_mode=anchor_film")
    anchor_xyz = dataset.anchor_xyz.to(device)
    anchor_features = build_anchor_features(
        anchor_xyz,
        int(model_config.get("anchor_num_frequencies", 4)),
    )
    anchor_edges = build_anchor_knn_edges(
        anchor_xyz,
        int(model_config.get("anchor_graph_k", 4)),
    )
    base_model = TargetOnlyGaussianModel(anchor_xyz).to(device)
    model = DressableGaussianModel(base_model)
    num_anchors = anchor_xyz.shape[0]
    identity_indices = torch.arange(num_anchors, device=device, dtype=torch.long)[:, None]
    identity_weights = torch.ones(num_anchors, 1, device=device, dtype=anchor_xyz.dtype)
    model.initialize_film_clothing_generator(
        num_clothes=dataset.num_clothes,
        anchor_features=anchor_features,
        gaussian_anchor_indices=identity_indices,
        gaussian_anchor_weights=identity_weights,
        embedding_dim=int(model_config["embedding_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        num_layers=int(model_config["num_layers"]),
        hyper_hidden_dim=int(model_config["hyper_hidden_dim"]),
    )
    for parameter in model.base_model.parameters():
        parameter.requires_grad_(False)

    optimizer_config = config["optimizer"]
    if str(optimizer_config.get("type", "adam")).lower() != "adam":
        raise ValueError("the first train_dressable version only supports Adam")
    optimizer = torch.optim.Adam(
        model.clothing_parameters(),
        lr=float(optimizer_config["lr"]),
        weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
    )
    return model, optimizer, anchor_features, anchor_edges


def create_rendering_components(
    dataset: RenderingDressableDataset,
    config: dict[str, Any],
    base_model: Any,
    device: torch.device | str,
) -> tuple[DressableGaussianModel, torch.optim.Optimizer, torch.Tensor, torch.Tensor]:
    """Attach anchor_film to a frozen real or mock GaussianModel-like base."""

    device = torch.device(device)
    anchor_xyz = dataset.anchor_xyz.to(device)
    model_config = config["model"]
    if model_config.get("offset_mode") != "anchor_film":
        raise ValueError("rendering_finetune currently requires offset_mode=anchor_film")
    if not bool(config.get("train", {}).get("freeze_base", True)):
        raise ValueError("rendering_finetune currently requires train.freeze_base=true")
    anchor_features = build_anchor_features(
        anchor_xyz,
        int(model_config.get("anchor_num_frequencies", 4)),
    )
    anchor_edges = build_anchor_knn_edges(
        anchor_xyz,
        int(model_config.get("anchor_graph_k", 4)),
    )
    indices = getattr(base_model, "nbr_gs", None)
    weights = getattr(base_model, "nbr_gs_invdist", None)
    if not isinstance(indices, torch.Tensor) or indices.ndim != 2:
        raise ValueError("rendering base requires nbr_gs Tensor[N,K] anchor indices")
    indices = indices.to(device=device, dtype=torch.long)
    if indices.shape[0] != base_model._xyz.shape[0]:
        raise ValueError("base nbr_gs count does not match Gaussian count")
    if torch.any(indices < 0) or torch.any(indices >= anchor_xyz.shape[0]):
        raise ValueError("base nbr_gs indices do not match anchor target topology")
    if not isinstance(weights, torch.Tensor) or weights.shape != indices.shape:
        raise ValueError("rendering base requires nbr_gs_invdist matching nbr_gs shape")
    weights = weights.to(device=device, dtype=anchor_xyz.dtype)
    if torch.any(weights < 0) or torch.any(weights.sum(dim=1) <= 0):
        raise ValueError("base anchor interpolation weights must be non-negative with positive rows")
    weights = weights / weights.sum(dim=1, keepdim=True)

    freeze_base_model(base_model)
    model = DressableGaussianModel(base_model)
    model.initialize_film_clothing_generator(
        num_clothes=dataset.num_clothes,
        anchor_features=anchor_features,
        gaussian_anchor_indices=indices,
        gaussian_anchor_weights=weights,
        embedding_dim=int(model_config["embedding_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        num_layers=int(model_config["num_layers"]),
        hyper_hidden_dim=int(model_config["hyper_hidden_dim"]),
    )
    optimizer_config = config["optimizer"]
    optimizer = torch.optim.Adam(
        model.clothing_parameters(),
        lr=float(optimizer_config["lr"]),
        weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
    )
    return model, optimizer, anchor_features, anchor_edges


def compute_rendering_training_loss(
    model: DressableGaussianModel,
    sample: dict[str, Any],
    anchor_edges: torch.Tensor,
    config: dict[str, Any],
    device: torch.device | str,
    lpips_loss: LPIPSLoss | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Render one sample and combine anchor teacher and image-space losses."""

    device = torch.device(device)
    base_model = model.base_model
    base_model.smpl_poses = sample["pose"].detach().cpu()
    base_model.Th = sample["Th"].detach().cpu()
    base_model.Rh = sample["Rh"].detach().cpu()
    height, width = sample["target_rgb"].shape[-2:]
    camera = build_mmlphuman_camera(sample["camera"], height, width, device)
    background = torch.as_tensor(
        config.get("render", {}).get("background", [1.0, 1.0, 1.0]),
        device=device,
        dtype=torch.float32,
    )
    cloth_id = sample["cloth_id"].to(device)
    pred_rgb, pred_alpha, _ = model.render(camera, cloth_id, background=background)
    pred_anchor = model.compute_film_anchor_offsets(cloth_id)
    target = sample["anchor_offset_target"]
    target_offsets = {
        "delta_xyz": target["delta_xyz"].to(device),
        "delta_scaling": target["delta_scaling"].to(device),
        "delta_opacity": target["delta_opacity"].to(device),
    }
    valid = target["valid_mask"].to(device)
    region = target["cloth_region_weight"].to(device)
    fallback = target.get("fallback_mask", torch.zeros_like(valid)).to(device)
    loss_config = config["loss"]
    effective_valid = valid * (
        1 - fallback + float(loss_config.get("fallback_weight", 0.5)) * fallback
    )
    supervision = anchor_offset_supervision_loss(
        pred_anchor,
        target_offsets,
        valid_mask=effective_valid,
        cloth_region_weight=region,
        xyz_weight=float(loss_config["xyz_weight"]),
        scaling_weight=float(loss_config["scaling_weight"]),
        opacity_weight=float(loss_config["opacity_weight"]),
    )
    smooth = anchor_graph_smoothness_loss(pred_anchor["delta_xyz"], anchor_edges.to(device))
    noncloth = non_clothing_region_loss(pred_anchor, region)
    magnitude = offset_magnitude_regularization(pred_anchor)
    render_losses = combined_rendering_loss(
        pred_rgb,
        sample["target_rgb"].to(device),
        pred_alpha,
        sample["foreground_mask"].to(device),
        rgb_mask=sample["foreground_mask"].to(device),
        rgb_weight=float(loss_config.get("rgb_weight", 1.0)),
        mask_weight=float(loss_config.get("mask_weight", 0.5)),
        lpips_weight=float(loss_config.get("lpips_weight", 0.0)),
        ssim_weight=float(loss_config.get("ssim_weight", 0.2)),
        lpips_loss=lpips_loss,
    )
    total = (
        float(loss_config.get("anchor_weight", 1.0)) * supervision["total"]
        + render_losses["total"]
        + float(loss_config["smooth_weight"]) * smooth
        + float(loss_config["noncloth_weight"]) * noncloth["total"]
        + float(loss_config["magnitude_weight"]) * magnitude["total"]
    )
    losses = {
        "total": total,
        "anchor": supervision["total"],
        "rgb_l1": render_losses["rgb_l1"],
        "rgb_ssim": render_losses["rgb_ssim"],
        "lpips": render_losses["lpips"],
        "mask_bce": render_losses["mask_bce"],
        "mask_dice": render_losses["mask_dice"],
        "smooth": smooth,
        "noncloth": noncloth["total"],
        "magnitude": magnitude["total"],
    }
    outputs = {
        "pred_rgb": pred_rgb,
        "pred_alpha": pred_alpha,
        "pred_anchor_xyz": pred_anchor["delta_xyz"],
    }
    return losses, outputs


def compute_anchor_training_loss(
    model: DressableGaussianModel,
    sample: dict[str, Any],
    anchor_edges: torch.Tensor,
    loss_config: dict[str, Any],
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Compute all first-version anchor target supervision terms."""

    device = torch.device(device)
    cloth_id = sample["cloth_id"].to(device)
    prediction = model.compute_film_anchor_offsets(cloth_id)
    targets = {
        "delta_xyz": sample["delta_xyz"].to(device),
        "delta_scaling": sample["delta_scaling"].to(device),
        "delta_opacity": sample["delta_opacity"].to(device),
    }
    valid_mask = sample["valid_mask"].to(device)
    region = sample["cloth_region_weight"].to(device)
    fallback = sample.get("fallback_mask")
    if fallback is None:
        fallback = torch.zeros_like(valid_mask)
    else:
        fallback = fallback.to(device)
    fallback_weight = float(loss_config.get("fallback_weight", 0.5))
    effective_valid = valid_mask * (1 - fallback + fallback_weight * fallback)

    supervision = anchor_offset_supervision_loss(
        prediction,
        targets,
        valid_mask=effective_valid,
        cloth_region_weight=region,
        xyz_weight=float(loss_config["xyz_weight"]),
        scaling_weight=float(loss_config["scaling_weight"]),
        opacity_weight=float(loss_config["opacity_weight"]),
    )
    smooth = anchor_graph_smoothness_loss(
        prediction["delta_xyz"],
        anchor_edges.to(device),
    )
    noncloth = non_clothing_region_loss(
        prediction,
        region,
        xyz_weight=float(loss_config["xyz_weight"]),
        scaling_weight=float(loss_config["scaling_weight"]),
        opacity_weight=float(loss_config["opacity_weight"]),
    )
    magnitude = offset_magnitude_regularization(prediction)
    total = (
        supervision["total"]
        + float(loss_config["smooth_weight"]) * smooth
        + float(loss_config["noncloth_weight"]) * noncloth["total"]
        + float(loss_config["magnitude_weight"]) * magnitude["total"]
    )
    return {
        "total": total,
        "xyz": supervision["xyz"],
        "scaling": supervision["scaling"],
        "opacity": supervision["opacity"],
        "smooth": smooth,
        "noncloth": noncloth["total"],
        "magnitude": magnitude["total"],
    }


def save_training_checkpoint(
    path: str | Path,
    model: DressableGaussianModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict[str, Any],
    cloth_id_map: dict[str, int],
    anchor_xyz: torch.Tensor,
    anchor_features: torch.Tensor,
    anchor_edges: torch.Tensor,
    base_fingerprint: str | None = None,
    base_checkpoint_path: str | None = None,
    train_mode: str | None = None,
    image_size: tuple[int, int] | None = None,
    render_loss_config: dict[str, Any] | None = None,
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "config": config,
        "cloth_id_map": cloth_id_map,
        "anchor_xyz": anchor_xyz.detach().cpu(),
        "anchor_features": anchor_features.detach().cpu(),
        "anchor_edges": anchor_edges.detach().cpu(),
        "version": 1,
    }
    if train_mode is not None:
        checkpoint.update(
            {
                "base_fingerprint": base_fingerprint,
                "base_checkpoint_path": base_checkpoint_path,
                "train_mode": train_mode,
                "image_size": None if image_size is None else list(image_size),
                "render_loss_config": {} if render_loss_config is None else render_loss_config,
            }
        )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)


def load_training_checkpoint(
    path: str | Path,
    model: DressableGaussianModel,
    optimizer: torch.optim.Optimizer,
    cloth_id_map: dict[str, int],
    anchor_xyz: torch.Tensor,
    anchor_features: torch.Tensor,
    anchor_edges: torch.Tensor,
    expected_base_fingerprint: str | None = None,
    expected_train_mode: str | None = None,
    expected_image_size: tuple[int, int] | None = None,
) -> int:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("version") != 1:
        raise ValueError(f"unsupported dressable checkpoint version: {checkpoint.get('version')}")
    if checkpoint.get("cloth_id_map") != cloth_id_map:
        raise ValueError("resume cloth_id_map does not match the current dataset")
    _assert_checkpoint_tensor("anchor_xyz", checkpoint, anchor_xyz, floating=True)
    _assert_checkpoint_tensor("anchor_features", checkpoint, anchor_features, floating=True)
    _assert_checkpoint_tensor("anchor_edges", checkpoint, anchor_edges, floating=False)
    if expected_train_mode is not None and checkpoint.get("train_mode") != expected_train_mode:
        raise ValueError("resume train mode does not match current training mode")
    if expected_base_fingerprint is not None and checkpoint.get(
        "base_fingerprint"
    ) != expected_base_fingerprint:
        raise ValueError("resume base fingerprint does not match the loaded base checkpoint")
    if expected_image_size is not None and checkpoint.get("image_size") != list(
        expected_image_size
    ):
        raise ValueError("resume image size does not match current rendering data")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["step"])


@torch.no_grad()
def evaluate_anchor_targets(
    model: DressableGaussianModel,
    dataset: AnchorOffsetDataset,
    step: int,
    output_dir: str | Path,
    loss_config: dict[str, Any],
    device: torch.device | str,
) -> dict[str, Any]:
    """Evaluate every clothing target and export compatible predictions."""

    was_training = model.training
    model.eval()
    output_path = Path(output_dir)
    prediction_root = output_path / "predictions" / f"step_{step:06d}"
    results: dict[str, Any] = {"step": int(step), "clothes": {}}
    for sample in dataset:
        cloth_id = sample["cloth_id"].to(device)
        prediction = model.compute_film_anchor_offsets(cloth_id)
        targets = {
            "delta_xyz": sample["delta_xyz"].to(device),
            "delta_scaling": sample["delta_scaling"].to(device),
            "delta_opacity": sample["delta_opacity"].to(device),
        }
        supervision = anchor_offset_supervision_loss(
            prediction,
            targets,
            valid_mask=sample["valid_mask"].to(device),
            cloth_region_weight=sample["cloth_region_weight"].to(device),
            xyz_weight=float(loss_config["xyz_weight"]),
            scaling_weight=float(loss_config["scaling_weight"]),
            opacity_weight=float(loss_config["opacity_weight"]),
        )
        magnitude = offset_magnitude_regularization(prediction)
        noncloth = non_clothing_region_loss(
            prediction,
            sample["cloth_region_weight"].to(device),
        )
        cloth_name = sample["cloth_name"]
        results["clothes"][cloth_name] = {
            "cloth_id": int(sample["cloth_id"].item()),
            "xyz": supervision["xyz"].item(),
            "scaling": supervision["scaling"].item(),
            "opacity": supervision["opacity"].item(),
            "total": supervision["total"].item(),
            "predicted_offset_magnitude": magnitude["total"].item(),
            "noncloth_offset_magnitude": noncloth["total"].item(),
        }
        predicted_target = {
            "anchor_xyz": sample["anchor_xyz"],
            "delta_xyz": prediction["delta_xyz"].detach().cpu(),
            "delta_scaling": prediction["delta_scaling"].detach().cpu(),
            "delta_opacity": prediction["delta_opacity"].detach().cpu(),
            "valid_mask": sample["valid_mask"],
            "cloth_region_weight": sample["cloth_region_weight"],
        }
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", cloth_name)
        save_anchor_offset_target(
            predicted_target,
            prediction_root / safe_name / "predicted_anchor_offsets.pt",
            metadata={
                "source": "prediction",
                "cloth_name": cloth_name,
                "cloth_id": int(sample["cloth_id"].item()),
                "step": int(step),
            },
        )
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / f"eval_step_{step:06d}.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    if was_training:
        model.train()
    return results


def train_anchor_targets(
    config: dict[str, Any],
    target_paths: list[str | Path],
    output_dir: str | Path | None = None,
    device: torch.device | str | None = None,
    debug_one_batch: bool = False,
    resume: str | Path | None = None,
) -> dict[str, Any]:
    """Train anchor_film on serialized anchor targets and return runtime state."""

    seed = int(config["train"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    validate_batch_size(int(config["data"]["batch_size"]))
    dataset = AnchorOffsetDataset(target_paths)
    model, optimizer, anchor_features, anchor_edges = create_training_components(
        dataset,
        config,
        device,
    )
    output_path = Path(output_dir or config["train"]["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "config_resolved.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)
    with (output_path / "cloth_id_map.json").open("w", encoding="utf-8") as file:
        json.dump(dataset.get_cloth_id_map(), file, ensure_ascii=False, indent=2)

    resume_path = resume if resume is not None else config["train"].get("resume")
    start_step = 0
    if resume_path:
        start_step = load_training_checkpoint(
            resume_path,
            model,
            optimizer,
            dataset.get_cloth_id_map(),
            dataset.anchor_xyz,
            anchor_features,
            anchor_edges,
        )

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=int(config["data"].get("num_workers", 0)),
        collate_fn=_single_sample_collate,
        generator=generator,
    )
    iterator = iter(loader)
    configured_steps = int(config["train"]["steps"])
    end_step = min(configured_steps, start_step + 1) if debug_one_batch else configured_steps
    history: list[dict[str, float]] = []
    log_path = output_path / "train_log.jsonl"
    actual_step = start_step
    for step in range(start_step + 1, end_step + 1):
        try:
            sample = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            sample = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        losses = compute_anchor_training_loss(
            model,
            sample,
            anchor_edges,
            config["loss"],
            device,
        )
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError(f"non-finite loss at step {step}")
        losses["total"].backward()
        gradient_clip = float(
            config["train"].get("grad_clip", config["train"].get("gradient_clip", 1.0))
        )
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(list(model.clothing_parameters()), gradient_clip)
        optimizer.step()
        actual_step = step
        log_record = {"step": step, **{key: value.detach().item() for key, value in losses.items()}}
        history.append(log_record)
        with log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(log_record) + "\n")

        if step == 1 or step % int(config["train"]["log_every"]) == 0:
            print(
                f"step={step} total={log_record['total']:.8f} xyz={log_record['xyz']:.8f} "
                f"scaling={log_record['scaling']:.8f} opacity={log_record['opacity']:.8f} "
                f"smooth={log_record['smooth']:.8f} noncloth={log_record['noncloth']:.8f} "
                f"magnitude={log_record['magnitude']:.8f}"
            )
        if step % int(config["train"]["eval_every"]) == 0:
            evaluate_anchor_targets(model, dataset, step, output_path, config["loss"], device)
        if step % int(config["train"]["save_every"]) == 0:
            _save_step_checkpoints(
                output_path,
                model,
                optimizer,
                step,
                config,
                dataset,
                anchor_features,
                anchor_edges,
            )

    _save_step_checkpoints(
        output_path,
        model,
        optimizer,
        actual_step,
        config,
        dataset,
        anchor_features,
        anchor_edges,
    )
    evaluate_anchor_targets(
        model,
        dataset,
        actual_step,
        output_path,
        config["loss"],
        device,
    )
    return {
        "model": model,
        "optimizer": optimizer,
        "dataset": dataset,
        "anchor_features": anchor_features,
        "anchor_edges": anchor_edges,
        "history": history,
        "step": actual_step,
        "output_dir": output_path,
    }


def export_render_visualization(
    output_dir: str | Path,
    step: int,
    sample: dict[str, Any],
    pred_rgb: torch.Tensor,
    pred_alpha: torch.Tensor,
) -> Path:
    """Export detached target/render/mask/difference images for one sample."""

    sample_name = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        f"{sample['cloth_name']}_{sample['frame_id']}_{sample['view_id']}",
    )
    directory = Path(output_dir) / "vis" / f"step_{step:06d}" / sample_name
    directory.mkdir(parents=True, exist_ok=True)
    target_rgb = _to_chw_image(sample["target_rgb"].detach().cpu(), 3)
    rendered_rgb = _to_chw_image(pred_rgb.detach().cpu(), 3)
    target_mask = _to_chw_image(sample["foreground_mask"].detach().cpu(), 1)
    rendered_alpha = _to_chw_image(pred_alpha.detach().cpu(), 1)
    difference = torch.abs(rendered_rgb - target_rgb)
    _save_tensor_image(directory / "target_rgb.png", target_rgb)
    _save_tensor_image(directory / "rendered_rgb.png", rendered_rgb)
    _save_tensor_image(directory / "target_mask.png", target_mask)
    _save_tensor_image(directory / "rendered_alpha.png", rendered_alpha)
    _save_tensor_image(directory / "rgb_difference.png", difference)
    return directory


def _to_chw_image(tensor: torch.Tensor, channels: int) -> torch.Tensor:
    """Normalize a detached HWC/CHW image tensor to CPU CHW layout."""

    if tensor.ndim != 3:
        raise ValueError("visualization tensors must be three-dimensional")
    if tensor.shape[0] == channels:
        result = tensor
    elif tensor.shape[-1] == channels:
        result = tensor.permute(2, 0, 1)
    else:
        raise ValueError(f"visualization tensor does not contain {channels} channels")
    if not torch.isfinite(result).all():
        raise ValueError("visualization tensor contains NaN or Inf")
    return result.float().cpu().clamp(0, 1).contiguous()


def _save_tensor_image(path: Path, tensor: torch.Tensor) -> None:
    """Save a CHW float tensor in [0,1] without retaining a training graph."""

    array = (tensor.clamp(0, 1) * 255).round().to(torch.uint8)
    if array.shape[0] == 1:
        image_array = array[0].numpy()
        mode = "L"
    elif array.shape[0] == 3:
        image_array = array.permute(1, 2, 0).numpy()
        mode = "RGB"
    else:
        raise ValueError("saved visualization must have one or three channels")
    Image.fromarray(image_array, mode=mode).save(path)


def _load_anchor_pretrained_clothing(
    model: DressableGaussianModel,
    checkpoint_path: str | Path,
) -> None:
    """Transfer only clothing-generator state from an anchor-only checkpoint."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"anchor pretraining checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("version") != 1 or "model" not in checkpoint:
        raise ValueError("anchor pretraining checkpoint has an unsupported format")
    module_prefixes = (
        "clothing_embedding.",
        "anchor_clothing_mlp.",
        "clothing_film_generator.",
    )
    source_state = {
        key: value
        for key, value in checkpoint["model"].items()
        if key.startswith(module_prefixes)
    }
    current_state = model.state_dict()
    transferable_keys = {
        key
        for key in current_state
        if key.startswith(module_prefixes)
    }
    missing = sorted(transferable_keys - set(source_state))
    unexpected = sorted(set(source_state) - transferable_keys)
    if missing or unexpected:
        raise ValueError(
            "anchor pretraining clothing state mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for key in transferable_keys:
        if source_state[key].shape != current_state[key].shape:
            raise ValueError(f"anchor pretraining tensor shape mismatch for {key}")
        current_state[key] = source_state[key]
    model.load_state_dict(current_state, strict=True)


def _save_rendering_checkpoint(
    output_path: Path,
    model: DressableGaussianModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict[str, Any],
    dataset: RenderingDressableDataset,
    anchor_features: torch.Tensor,
    anchor_edges: torch.Tensor,
    base_model: Any,
) -> None:
    """Save latest and step-specific rendering checkpoints with base identity."""

    arguments = (
        model,
        optimizer,
        step,
        config,
        dataset.get_cloth_id_map(),
        dataset.anchor_xyz,
        anchor_features,
        anchor_edges,
    )
    metadata = {
        "base_fingerprint": base_model._dressable_base_fingerprint,
        "base_checkpoint_path": base_model._dressable_checkpoint_path,
        "train_mode": "rendering_finetune",
        "image_size": dataset.image_size,
        "render_loss_config": dict(config["loss"]),
    }
    save_training_checkpoint(output_path / "checkpoint_latest.pth", *arguments, **metadata)
    save_training_checkpoint(
        output_path / f"checkpoint_step_{step:06d}.pth",
        *arguments,
        **metadata,
    )


def train_rendering_finetune(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    device: torch.device | str | None = None,
    debug_one_batch: bool = False,
    resume: str | Path | None = None,
) -> dict[str, Any]:
    """Run frozen-base rendering-aware finetuning in a full MMLPHuman environment."""

    seed = int(config["train"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    validate_batch_size(int(config["train"].get("batch_size", 1)))
    render_data = config["render_data"]
    if not render_data.get("samples_json"):
        raise ValueError("rendering_finetune requires render_data.samples_json")
    image_size = None
    if render_data.get("image_height") is not None or render_data.get("image_width") is not None:
        if render_data.get("image_height") is None or render_data.get("image_width") is None:
            raise ValueError("render_data image_height and image_width must be set together")
        image_size = (int(render_data["image_height"]), int(render_data["image_width"]))
    dataset = RenderingDressableDataset(
        render_data["samples_json"],
        anchor_target_root=render_data.get("anchor_target_root"),
        image_size=image_size,
    )
    base_config = config["base"]
    if not base_config.get("model_dir"):
        raise ValueError("rendering_finetune requires base.model_dir")
    base_model = load_frozen_mmlphuman_base(
        base_config["model_dir"],
        checkpoint_path=base_config.get("checkpoint_path"),
        device=device,
    )
    base_fingerprint = base_model._dressable_base_fingerprint
    if base_config.get("fingerprint_required", True) and not base_fingerprint:
        raise ValueError("loaded base does not provide a fingerprint")
    model, optimizer, anchor_features, anchor_edges = create_rendering_components(
        dataset,
        config,
        base_model,
        device,
    )
    pretrain_path = config["train"].get("anchor_pretrain_checkpoint")
    resume_path = resume if resume is not None else config["train"].get("resume")
    if pretrain_path and resume_path:
        raise ValueError("anchor_pretrain_checkpoint and resume cannot be used together")
    if pretrain_path:
        _load_anchor_pretrained_clothing(model, pretrain_path)

    output_path = Path(output_dir or config["train"]["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "config_resolved.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)
    with (output_path / "cloth_id_map.json").open("w", encoding="utf-8") as file:
        json.dump(dataset.get_cloth_id_map(), file, ensure_ascii=False, indent=2)
    start_step = 0
    if resume_path:
        start_step = load_training_checkpoint(
            resume_path,
            model,
            optimizer,
            dataset.get_cloth_id_map(),
            dataset.anchor_xyz,
            anchor_features,
            anchor_edges,
            expected_base_fingerprint=base_fingerprint,
            expected_train_mode="rendering_finetune",
            expected_image_size=dataset.image_size,
        )

    lpips_weight = float(config["loss"].get("lpips_weight", 0.0))
    lpips_loss = LPIPSLoss().to(device) if lpips_weight > 0 else None
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=int(config["data"].get("num_workers", 0)),
        collate_fn=_single_sample_collate,
        generator=torch.Generator().manual_seed(seed),
    )
    iterator = iter(loader)
    configured_steps = int(config["train"]["steps"])
    end_step = min(configured_steps, start_step + 1) if debug_one_batch else configured_steps
    amp_enabled = bool(config["train"].get("mixed_precision", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history: list[dict[str, float]] = []
    log_path = output_path / "train_log.jsonl"
    actual_step = start_step
    for step in range(start_step + 1, end_step + 1):
        try:
            sample = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            sample = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            losses, outputs = compute_rendering_training_loss(
                model,
                sample,
                anchor_edges,
                config,
                device,
                lpips_loss,
            )
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError(f"non-finite rendering loss at step {step}")
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        grad_clip = float(config["train"].get("grad_clip", 1.0))
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(list(model.clothing_parameters()), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        actual_step = step
        record = {"step": step, **{key: value.detach().item() for key, value in losses.items()}}
        history.append(record)
        with log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
        if step == 1 or step % int(config["train"]["log_every"]) == 0:
            print(
                f"step={step} total={record['total']:.8f} anchor={record['anchor']:.8f} "
                f"rgb_l1={record['rgb_l1']:.8f} rgb_ssim={record['rgb_ssim']:.8f} "
                f"mask_bce={record['mask_bce']:.8f} mask_dice={record['mask_dice']:.8f}"
            )
        if step % int(config["train"]["eval_every"]) == 0:
            export_render_visualization(
                output_path,
                step,
                sample,
                outputs["pred_rgb"],
                outputs["pred_alpha"],
            )
        if step % int(config["train"]["save_every"]) == 0:
            _save_rendering_checkpoint(
                output_path,
                model,
                optimizer,
                step,
                config,
                dataset,
                anchor_features,
                anchor_edges,
                base_model,
            )
    _save_rendering_checkpoint(
        output_path,
        model,
        optimizer,
        actual_step,
        config,
        dataset,
        anchor_features,
        anchor_edges,
        base_model,
    )
    return {
        "model": model,
        "optimizer": optimizer,
        "dataset": dataset,
        "anchor_features": anchor_features,
        "anchor_edges": anchor_edges,
        "history": history,
        "step": actual_step,
        "output_dir": output_path,
        "base_fingerprint": base_fingerprint,
    }


def create_image_conditioned_components(
    dataset: ImageConditionedEpisodeDataset,
    config: dict[str, Any],
    base_model: Any | None = None,
    device: torch.device | str = "cpu",
) -> tuple[
    ImageConditionedDressableModel,
    torch.optim.Optimizer,
    torch.Tensor,
    torch.Tensor,
]:
    """Create the image encoder, local aggregator, frozen base, and optimizer."""

    device = torch.device(device)
    anchor_xyz = resolve_image_conditioned_canonical_anchors(
        dataset,
        base_model,
        device,
        require_real_base=bool(config.get("base", {}).get("require_real_base", False)),
    )
    model_config = config["model"]
    image_config = config["image_conditioning"]
    if model_config.get("offset_mode") != "anchor_film":
        raise ValueError("image-conditioned modes require offset_mode=anchor_film")
    anchor_features = build_anchor_features(
        anchor_xyz, int(model_config.get("anchor_num_frequencies", 4))
    )
    anchor_edges = build_anchor_knn_edges(
        anchor_xyz, int(model_config.get("anchor_graph_k", 4))
    )
    if base_model is None:
        base_model = TargetOnlyGaussianModel(anchor_xyz).to(device)
        indices = torch.arange(anchor_xyz.shape[0], device=device)[:, None]
        weights = torch.ones(anchor_xyz.shape[0], 1, device=device)
    else:
        indices = getattr(base_model, "nbr_gs", None)
        weights = getattr(base_model, "nbr_gs_invdist", None)
        if not isinstance(indices, torch.Tensor) or not isinstance(weights, torch.Tensor):
            raise ValueError("rendering base requires nbr_gs and nbr_gs_invdist")
        indices = indices.to(device=device, dtype=torch.long)
        weights = weights.to(device=device, dtype=anchor_xyz.dtype)
    freeze_base_model(base_model)
    dressable = DressableGaussianModel(base_model)
    dressable.initialize_film_clothing_generator(
        num_clothes=dataset.num_clothes,
        anchor_features=anchor_features,
        gaussian_anchor_indices=indices,
        gaussian_anchor_weights=weights,
        embedding_dim=int(image_config["embedding_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        num_layers=int(model_config["num_layers"]),
        hyper_hidden_dim=int(model_config["hyper_hidden_dim"]),
    )
    dressable.configure_offset_channels(
        enable_delta_xyz=bool(model_config.get("enable_delta_xyz", True)),
        enable_delta_scaling=bool(model_config.get("enable_delta_scaling", True)),
        enable_delta_opacity=bool(model_config.get("enable_delta_opacity", True)),
    )
    for parameter in dressable.clothing_embedding.parameters():
        parameter.requires_grad_(False)
    encoder = ClothingObservationEncoder(
        embedding_dim=int(image_config["embedding_dim"]),
        feature_dim=int(image_config["feature_dim"]),
    ).to(device=device, dtype=anchor_xyz.dtype)
    aggregator = MultiViewClothingAggregator(
        input_dim=int(image_config["feature_dim"]),
        output_dim=int(image_config["local_feature_dim"]),
    ).to(device=device, dtype=anchor_xyz.dtype)
    model = ImageConditionedDressableModel(
        dressable,
        encoder,
        AnchorImageProjector(),
        aggregator,
        canonical_anchors=anchor_xyz,
    )
    optimizer = build_image_conditioned_optimizer(model, config["optimizer"])
    return model, optimizer, anchor_features, anchor_edges


def resolve_image_conditioned_canonical_anchors(
    dataset: ImageConditionedEpisodeDataset,
    base_model: Any | None,
    device: torch.device | str,
    *,
    require_real_base: bool = False,
) -> torch.Tensor:
    """Prefer verified MMLPHuman control anchors and validate optional teachers."""

    teacher_anchors = dataset.get_anchor_xyz_or_none()
    base_anchors = None if base_model is None else getattr(base_model, "xyz_vt", None)
    if base_anchors is None:
        if require_real_base:
            raise AttributeError("real base_model must provide xyz_vt")
        if teacher_anchors is None:
            raise ValueError(
                "image-conditioned training requires base_model.xyz_vt or dataset anchor targets"
            )
        return teacher_anchors.detach().clone().to(device)

    if not isinstance(base_anchors, torch.Tensor):
        raise AttributeError("real base_model must provide xyz_vt")
    if base_anchors.ndim != 2 or base_anchors.shape[1] != 3:
        raise ValueError("base_model.xyz_vt must have shape [A,3]")
    if not torch.is_floating_point(base_anchors) or not torch.isfinite(base_anchors).all():
        raise ValueError("base_model.xyz_vt must be finite and floating point")
    if require_real_base and int(base_anchors.shape[0]) != 10000:
        raise ValueError(
            f"real CanonDressGS base requires A=10000 anchors, got {base_anchors.shape[0]}"
        )
    if teacher_anchors is not None:
        reference = base_anchors.detach().cpu()
        if teacher_anchors.shape != reference.shape or not torch.allclose(
            teacher_anchors, reference, atol=1e-6, rtol=0
        ):
            raise ValueError(
                "dataset teacher anchors do not match base_model.xyz_vt ordering/values"
            )
    return base_anchors.detach().clone().to(device)


def build_episode_deformation_fn(
    adapter: MMLPHumanAnchorDeformationAdapter,
    episode: dict[str, Any],
):
    """Create a stateless per-episode closure over reference Rh/Th tensors."""

    reference_Rh = episode.get("reference_Rh")
    reference_Th = episode.get("reference_Th")
    if not isinstance(reference_Rh, torch.Tensor) or reference_Rh.ndim != 3:
        raise ValueError("episode reference_Rh must have shape [K,3,3]")
    if not isinstance(reference_Th, torch.Tensor) or reference_Th.ndim != 2:
        raise ValueError("episode reference_Th must have shape [K,3]")
    if reference_Rh.shape[0] != reference_Th.shape[0]:
        raise ValueError("reference_Rh/reference_Th must have matching K")

    def deformation_fn(
        anchors: torch.Tensor,
        pose: torch.Tensor,
        view_index: int,
    ) -> torch.Tensor:
        if not isinstance(view_index, int) or not 0 <= view_index < reference_Rh.shape[0]:
            raise IndexError("reference view_index is outside the episode")
        return adapter.deform_anchors(
            anchors,
            pose,
            reference_Rh[view_index].to(device=anchors.device, dtype=anchors.dtype),
            reference_Th[view_index].to(device=anchors.device, dtype=anchors.dtype),
        )

    return deformation_fn


@torch.no_grad()
def prepare_real_reference_geometry(
    base_model: Any,
    deformation_adapter: MMLPHumanAnchorDeformationAdapter,
    episode: dict[str, Any],
    background: torch.Tensor,
) -> dict[str, Any]:
    """Render one official gsplat ED/alpha map per real reference observation."""

    reference_images = episode.get("reference_images")
    reference_poses = episode.get("reference_poses")
    reference_Rh = episode.get("reference_Rh")
    reference_Th = episode.get("reference_Th")
    cameras = episode.get("reference_cameras")
    if not isinstance(reference_images, torch.Tensor) or reference_images.ndim != 4:
        raise ValueError("reference_images must have shape [K,3,H,W]")
    num_views, _, height, width = reference_images.shape
    expected = {
        "reference_poses": (num_views, 165),
        "reference_Rh": (num_views, 3, 3),
        "reference_Th": (num_views, 3),
    }
    for name, shape in expected.items():
        value = {
            "reference_poses": reference_poses,
            "reference_Rh": reference_Rh,
            "reference_Th": reference_Th,
        }[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if not torch.is_floating_point(value) or not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite and floating point")
    if not isinstance(cameras, list) or len(cameras) != num_views:
        raise ValueError("reference_cameras must contain K camera dictionaries")

    base_device = base_model._xyz.device
    base_dtype = base_model._xyz.dtype
    background = torch.as_tensor(background, device=base_device, dtype=base_dtype)
    if tuple(background.shape) != (3,) or not torch.isfinite(background).all():
        raise ValueError("background must be a finite Tensor[3]")
    original_cache = base_model.cache_dict
    original_state = {
        name: getattr(base_model, name)
        for name in ("_smpl_poses", "smpl_poses_cuda", "_Rh", "_Th")
    }
    depths = []
    alphas = []
    per_view = []
    for view_index in range(num_views):
        pose = reference_poses[view_index].to(device=base_device, dtype=base_dtype)
        Rh = reference_Rh[view_index].to(device=base_device, dtype=base_dtype)
        Th = reference_Th[view_index].to(device=base_device, dtype=base_dtype)
        camera = build_mmlphuman_camera(cameras[view_index], height, width, base_device)
        with mmlphuman_state_transaction(base_model, pose, Rh, Th):
            depth_output = render_mmlphuman_expected_depth(
                base_model,
                camera,
                background,
                canonical_overrides=None,
            )
        depth = depth_output["depth"]
        alpha = depth_output["alpha"]
        if tuple(depth.shape) != (height, width) or tuple(alpha.shape) != (height, width):
            raise RuntimeError("gsplat ED depth/alpha shape does not match reference image")
        if not torch.isfinite(depth).all() or not torch.isfinite(alpha).all():
            raise FloatingPointError("reference ED depth/alpha contains NaN or Inf")
        depths.append(depth.to(device=base_device, dtype=base_dtype))
        alphas.append(alpha.to(device=base_device, dtype=base_dtype))
        per_view.append(
            {
                "view_index": view_index,
                "frame_id": episode.get("reference_frame_ids", [None] * num_views)[view_index],
                "view_id": episode.get("reference_view_ids", [None] * num_views)[view_index],
                "render_mode": depth_output["render_mode"],
                "depth_finite_ratio": float(torch.isfinite(depth).float().mean().item()),
                "alpha_nonzero_ratio": float((alpha > 0).float().mean().item()),
            }
        )
    state_restored = base_model.cache_dict is original_cache and all(
        getattr(base_model, name) is value for name, value in original_state.items()
    )
    if not state_restored:
        raise RuntimeError("reference geometry rendering polluted base pose/cache state")
    depth_tensor = torch.stack(depths, dim=0).unsqueeze(1)
    alpha_tensor = torch.stack(alphas, dim=0).unsqueeze(1)
    return {
        "deformation_fn": build_episode_deformation_fn(deformation_adapter, episode),
        "surface_depth_maps": depth_tensor,
        "surface_alpha_maps": alpha_tensor,
        "diagnostics": {
            "num_views": num_views,
            "height": height,
            "width": width,
            "depth_shape": list(depth_tensor.shape),
            "alpha_shape": list(alpha_tensor.shape),
            "depth_finite_ratio": float(
                torch.isfinite(depth_tensor).float().mean().item()
            ),
            "base_state_cache_restored": True,
            "per_view": per_view,
        },
    }


def build_image_conditioned_optimizer(
    model: ImageConditionedDressableModel,
    optimizer_config: dict[str, Any],
) -> torch.optim.Optimizer:
    """Build named, disjoint parameter groups and exclude base/ID embedding state."""

    dressable = model.dressable_model
    local_parameters = list(dressable.anchor_clothing_mlp.local_feature_adapter.parameters())
    local_ids = {id(parameter) for parameter in local_parameters}
    groups = (
        (
            "encoder",
            list(model.clothing_observation_encoder.parameters()),
            float(optimizer_config.get("encoder_lr", optimizer_config.get("lr", 1e-4))),
        ),
        (
            "aggregator",
            list(model.multiview_aggregator.parameters()),
            float(optimizer_config.get("aggregator_lr", optimizer_config.get("lr", 1e-4))),
        ),
        (
            "hypernetwork",
            list(dressable.clothing_film_generator.parameters()),
            float(optimizer_config.get("hypernetwork_lr", optimizer_config.get("lr", 1e-4))),
        ),
        (
            "anchor_mlp",
            [
                parameter
                for parameter in dressable.anchor_clothing_mlp.parameters()
                if id(parameter) not in local_ids
            ],
            float(optimizer_config.get("anchor_mlp_lr", optimizer_config.get("lr", 1e-4))),
        ),
        (
            "local_adapter",
            local_parameters,
            float(optimizer_config.get("local_adapter_lr", optimizer_config.get("lr", 1e-4))),
        ),
    )
    seen: set[int] = set()
    parameter_groups = []
    for name, parameters, learning_rate in groups:
        trainable = [parameter for parameter in parameters if parameter.requires_grad]
        duplicates = [parameter for parameter in trainable if id(parameter) in seen]
        if duplicates:
            raise RuntimeError(f"optimizer parameter group {name!r} contains duplicates")
        seen.update(id(parameter) for parameter in trainable)
        count = sum(parameter.numel() for parameter in trainable)
        print(f"optimizer group {name}: parameters={count} lr={learning_rate:g}")
        parameter_groups.append({"name": name, "params": trainable, "lr": learning_rate})
    return torch.optim.Adam(
        parameter_groups,
        weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
    )


def compute_image_conditioned_training_loss(
    model: ImageConditionedDressableModel,
    episode: dict[str, Any],
    anchor_edges: torch.Tensor,
    config: dict[str, Any],
    device: torch.device | str,
    second_episode: dict[str, Any] | None = None,
    render_target: bool = False,
    deformation_fn=None,
    deformation_adapter: MMLPHumanAnchorDeformationAdapter | None = None,
    lpips_loss: LPIPSLoss | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Compute teacher, regularization, consistency, and optional render losses."""

    device = torch.device(device)
    render_kwargs = None
    background = torch.as_tensor(
        config.get("render", {}).get("background", [1, 1, 1]),
        device=device,
        dtype=model.dressable_model.anchor_features.dtype,
    )
    if render_target:
        render_kwargs = {"background": background}
    image_config = config["image_conditioning"]
    require_depth = bool(image_config.get("require_depth_visibility", False))
    geometry = None
    active_deformation_fn = deformation_fn
    if deformation_adapter is not None:
        if require_depth:
            geometry = prepare_real_reference_geometry(
                model.dressable_model.base_model,
                deformation_adapter,
                episode,
                background,
            )
            active_deformation_fn = geometry["deformation_fn"]
        else:
            active_deformation_fn = build_episode_deformation_fn(
                deformation_adapter, episode
            )
    elif require_depth:
        raise RuntimeError(
            "depth-visible image training requires a real deformation adapter and ED geometry"
        )
    output = model.forward_episode(
        episode,
        deformation_fn=active_deformation_fn,
        render_target=render_target,
        render_kwargs=render_kwargs,
        surface_depth_maps=(None if geometry is None else geometry["surface_depth_maps"]),
        surface_alpha_maps=(None if geometry is None else geometry["surface_alpha_maps"]),
        depth_abs_tolerance=float(image_config.get("depth_abs_tolerance", 0.02)),
        depth_rel_tolerance=float(image_config.get("depth_rel_tolerance", 0.01)),
        depth_alpha_threshold=float(image_config.get("depth_alpha_threshold", 1e-4)),
        require_depth_visibility=require_depth,
        require_mmlphuman_state_transaction=bool(
            config.get("base", {}).get("require_real_base", False)
        ),
        reference_geometry_diagnostics=(
            None if geometry is None else geometry["diagnostics"]
        ),
        reference_only_gate=episode.get("reference_only_gate"),
    )
    prediction = output["anchor_offsets"]
    target = episode.get("anchor_offset_target")
    loss_config = config["loss"]
    zero = prediction["delta_xyz"].sum() * 0
    if target is None:
        supervision = {"xyz": zero, "scaling": zero, "opacity": zero, "total": zero}
        anchor_available = zero
        region = torch.ones(
            prediction["delta_xyz"].shape[0], 1, device=device, dtype=prediction["delta_xyz"].dtype
        )
    else:
        target_offsets = {
            key: target[key].to(device)
            for key in ("delta_xyz", "delta_scaling", "delta_opacity")
        }
        valid = target["valid_mask"].to(device)
        region = target["cloth_region_weight"].to(device)
        supervision = anchor_offset_supervision_loss(
            prediction,
            target_offsets,
            valid_mask=valid,
            cloth_region_weight=region,
            xyz_weight=float(loss_config.get("xyz_weight", 1.0)),
            scaling_weight=float(loss_config.get("scaling_weight", 1.0)),
            opacity_weight=float(loss_config.get("opacity_weight", 1.0)),
        )
        anchor_available = zero + 1
    smooth = anchor_graph_smoothness_loss(
        prediction["delta_xyz"], anchor_edges.to(device)
    )
    noncloth = non_clothing_region_loss(
        prediction,
        region,
        outside_threshold=(
            None
            if loss_config.get("noncloth_outside_threshold") is None
            else float(loss_config["noncloth_outside_threshold"])
        ),
    )["total"]
    magnitude = offset_magnitude_regularization(prediction)["total"]

    latent_consistency = zero
    local_consistency = zero
    offset_consistency = zero
    second_output = None
    if second_episode is not None:
        second_geometry = None
        second_deformation_fn = deformation_fn
        if deformation_adapter is not None:
            if require_depth:
                second_geometry = prepare_real_reference_geometry(
                    model.dressable_model.base_model,
                    deformation_adapter,
                    second_episode,
                    background,
                )
                second_deformation_fn = second_geometry["deformation_fn"]
            else:
                second_deformation_fn = build_episode_deformation_fn(
                    deformation_adapter, second_episode
                )
        second_output = model.forward_episode(
            second_episode,
            deformation_fn=second_deformation_fn,
            render_target=False,
            surface_depth_maps=(
                None if second_geometry is None else second_geometry["surface_depth_maps"]
            ),
            surface_alpha_maps=(
                None if second_geometry is None else second_geometry["surface_alpha_maps"]
            ),
            depth_abs_tolerance=float(image_config.get("depth_abs_tolerance", 0.02)),
            depth_rel_tolerance=float(image_config.get("depth_rel_tolerance", 0.01)),
            depth_alpha_threshold=float(
                image_config.get("depth_alpha_threshold", 1e-4)
            ),
            require_depth_visibility=require_depth,
            reference_geometry_diagnostics=(
                None if second_geometry is None else second_geometry["diagnostics"]
            ),
        )
        latent_consistency = clothing_embedding_consistency_loss(
            output["global_clothing_embedding"],
            second_output["global_clothing_embedding"],
        )
        local_consistency = anchor_feature_consistency_loss(
            output["anchor_clothing_features"],
            second_output["anchor_clothing_features"],
            output["anchor_visibility"],
            second_output["anchor_visibility"],
        )
        offset_consistency = clothing_offset_consistency_loss(
            output["anchor_offsets"], second_output["anchor_offsets"]
        )["total"]

    render_losses = {
        "rgb_l1": zero,
        "rgb_ssim": zero,
        "lpips": zero,
        "mask_bce": zero,
        "mask_dice": zero,
        "total": zero,
    }
    if render_target:
        if not isinstance(output["render"], tuple) or len(output["render"]) < 2:
            raise ValueError("base render must return at least (rgb, alpha)")
        pred_rgb, pred_alpha = output["render"][:2]
        render_losses = combined_rendering_loss(
            pred_rgb,
            episode["target_rgb"].to(device),
            pred_alpha,
            episode["target_foreground_mask"].to(device),
            rgb_mask=episode["target_foreground_mask"].to(device),
            rgb_weight=float(loss_config.get("rgb_weight", 1.0)),
            mask_weight=float(loss_config.get("mask_weight", 0.5)),
            lpips_weight=float(loss_config.get("lpips_weight", 0.0)),
            ssim_weight=float(loss_config.get("ssim_weight", 0.2)),
            lpips_loss=lpips_loss,
        )
    total = (
        float(loss_config.get("anchor_weight", 1.0)) * supervision["total"]
        + float(loss_config.get("smooth_weight", 0.05)) * smooth
        + float(loss_config.get("noncloth_weight", 0.1)) * noncloth
        + float(loss_config.get("magnitude_weight", 0.001)) * magnitude
        + float(loss_config.get("latent_consistency_weight", 0.1)) * latent_consistency
        + float(loss_config.get("anchor_feature_consistency_weight", 0.1)) * local_consistency
        + float(loss_config.get("offset_consistency_weight", 0.1)) * offset_consistency
        + render_losses["total"]
    )
    losses = {
        "total": total,
        "anchor": supervision["total"],
        "anchor_available": anchor_available,
        "smooth": smooth,
        "noncloth": noncloth,
        "magnitude": magnitude,
        "latent_consistency": latent_consistency,
        "anchor_feature_consistency": local_consistency,
        "offset_consistency": offset_consistency,
        "rgb_l1": render_losses["rgb_l1"],
        "rgb_ssim": render_losses["rgb_ssim"],
        "lpips": render_losses["lpips"],
        "mask_bce": render_losses["mask_bce"],
        "mask_dice": render_losses["mask_dice"],
    }
    if not all(torch.isfinite(value) for value in losses.values()):
        raise FloatingPointError("image-conditioned loss contains NaN or Inf")
    return losses, {"primary": output, "secondary": second_output}


def save_image_training_checkpoint(
    path: str | Path,
    model: ImageConditionedDressableModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict[str, Any],
    dataset: ImageConditionedEpisodeDataset,
    base_fingerprint: str | None = None,
) -> None:
    """Save image-conditioned state in an isolated version-2 format."""

    checkpoint = {
        "train_mode": config["train"]["mode"],
        "manifest_fingerprint": dataset.manifest_fingerprint,
        "split": dataset.split,
        "cloth_id_map": dataset.get_cloth_id_map(),
        "reference_sampling_config": {
            "reference_count": list(dataset.reference_count_range),
            "random_reference_sampling": dataset.random_reference_sampling,
            "exclude_target_from_reference": dataset.exclude_target_from_reference,
            "seed": dataset.seed,
        },
        "image_conditioning_config": dict(config["image_conditioning"]),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "base_fingerprint": base_fingerprint,
        "anchor_xyz": model.canonical_anchors.detach().cpu(),
        "image_size": list(dataset.image_size),
        "embedding_dim": model.clothing_observation_encoder.embedding_dim,
        "local_feature_dim": model.multiview_aggregator.output_dim,
        "config": config,
        "version": 2,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)


def load_image_training_checkpoint(
    path: str | Path,
    model: ImageConditionedDressableModel,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    dataset: ImageConditionedEpisodeDataset,
    expected_base_fingerprint: str | None = None,
) -> int:
    """Strictly restore version-2 image-conditioned training state."""

    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    checks = {
        "version": 2,
        "train_mode": config["train"]["mode"],
        "manifest_fingerprint": dataset.manifest_fingerprint,
        "split": dataset.split,
        "cloth_id_map": dataset.get_cloth_id_map(),
        "image_size": list(dataset.image_size),
        "embedding_dim": model.clothing_observation_encoder.embedding_dim,
        "local_feature_dim": model.multiview_aggregator.output_dim,
        "base_fingerprint": expected_base_fingerprint,
    }
    for key, expected in checks.items():
        if checkpoint.get(key) != expected:
            raise ValueError(f"image checkpoint {key} does not match current training state")
    expected_sampling = {
        "reference_count": list(dataset.reference_count_range),
        "random_reference_sampling": dataset.random_reference_sampling,
        "exclude_target_from_reference": dataset.exclude_target_from_reference,
        "seed": dataset.seed,
    }
    if checkpoint.get("reference_sampling_config") != expected_sampling:
        raise ValueError("image checkpoint reference sampling config does not match")
    if checkpoint.get("image_conditioning_config") != config["image_conditioning"]:
        raise ValueError("image checkpoint conditioning config does not match")
    if not torch.allclose(
        checkpoint["anchor_xyz"],
        model.canonical_anchors.detach().cpu(),
        atol=1e-6,
        rtol=0,
    ):
        raise ValueError("image checkpoint anchor topology does not match")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["step"])


@torch.no_grad()
def evaluate_image_conditioned(
    model: ImageConditionedDressableModel,
    dataset: ImageConditionedEpisodeDataset,
    step: int,
    output_dir: str | Path,
    render_target: bool = False,
    deformation_fn=None,
    render_background: list[float] | None = None,
    deformation_adapter: MMLPHumanAnchorDeformationAdapter | None = None,
    image_conditioning_config: dict[str, Any] | None = None,
    reference_only_gate: torch.Tensor | None = None,
    require_mmlphuman_state_transaction: bool = False,
) -> dict[str, Any]:
    """Export one deterministic episode per clothing item."""

    was_training = model.training
    model.eval()
    root = Path(output_dir) / "predictions" / f"step_{step:06d}"
    results = {"step": int(step), "clothes": {}}
    seen: set[str] = set()
    for index, (cloth_name, _) in enumerate(dataset._episodes):
        if cloth_name in seen:
            continue
        seen.add(cloth_name)
        episode = dataset.sample_episode(index, deterministic=True)
        if reference_only_gate is not None:
            episode["reference_only_gate"] = reference_only_gate
        conditioning = image_conditioning_config or {}
        require_depth = bool(conditioning.get("require_depth_visibility", False))
        background_tensor = torch.tensor(
            render_background or [1, 1, 1],
            device=model.dressable_model.anchor_features.device,
            dtype=model.dressable_model.anchor_features.dtype,
        )
        geometry = None
        active_deformation_fn = deformation_fn
        if deformation_adapter is not None:
            if require_depth:
                geometry = prepare_real_reference_geometry(
                    model.dressable_model.base_model,
                    deformation_adapter,
                    episode,
                    background_tensor,
                )
                active_deformation_fn = geometry["deformation_fn"]
            else:
                active_deformation_fn = build_episode_deformation_fn(
                    deformation_adapter, episode
                )
        elif require_depth:
            raise RuntimeError("depth-visible evaluation requires a real adapter")
        render_kwargs = None
        if render_target:
            render_kwargs = {"background": background_tensor}
        output = model.forward_episode(
            episode,
            deformation_fn=active_deformation_fn,
            render_target=render_target,
            render_kwargs=render_kwargs,
            surface_depth_maps=(
                None if geometry is None else geometry["surface_depth_maps"]
            ),
            surface_alpha_maps=(
                None if geometry is None else geometry["surface_alpha_maps"]
            ),
            depth_abs_tolerance=float(conditioning.get("depth_abs_tolerance", 0.02)),
            depth_rel_tolerance=float(conditioning.get("depth_rel_tolerance", 0.01)),
            depth_alpha_threshold=float(
                conditioning.get("depth_alpha_threshold", 1e-4)
            ),
            require_depth_visibility=require_depth,
            require_mmlphuman_state_transaction=require_mmlphuman_state_transaction,
            reference_geometry_diagnostics=(
                None if geometry is None else geometry["diagnostics"]
            ),
        )
        directory = root / cloth_name
        directory.mkdir(parents=True, exist_ok=True)
        target = episode["anchor_offset_target"]
        canonical_anchors = model.canonical_anchors.detach().cpu()
        predicted = {
            "anchor_xyz": canonical_anchors,
            **{key: value.detach().cpu() for key, value in output["anchor_offsets"].items()},
            "valid_mask": (
                torch.ones(canonical_anchors.shape[0], 1)
                if target is None
                else target["valid_mask"]
            ),
            "cloth_region_weight": (
                torch.ones(canonical_anchors.shape[0], 1)
                if target is None
                else target["cloth_region_weight"]
            ),
        }
        save_anchor_offset_target(
            predicted,
            directory / "predicted_anchor_offsets.pt",
            metadata={"cloth_name": cloth_name, "step": int(step), "source": "image_prediction"},
        )
        torch.save(output["global_clothing_embedding"].cpu(), directory / "global_clothing_embedding.pt")
        torch.save(output["anchor_clothing_features"].cpu(), directory / "anchor_clothing_features.pt")
        torch.save(output["anchor_visibility"].cpu(), directory / "anchor_visibility.pt")
        selection = {
            "target_frame_id": episode["target_frame_id"],
            "target_view_id": episode["target_view_id"],
            "reference_frame_ids": episode["reference_frame_ids"],
            "reference_view_ids": episode["reference_view_ids"],
        }
        (directory / "reference_selection.json").write_text(
            json.dumps(selection, indent=2), encoding="utf-8"
        )
        if render_target:
            pred_rgb, pred_alpha = output["render"][:2]
            target_rgb = _to_chw_image(episode["target_rgb"].cpu(), 3)
            rendered_rgb = _to_chw_image(pred_rgb.cpu(), 3)
            target_mask = _to_chw_image(episode["target_foreground_mask"].cpu(), 1)
            rendered_alpha = _to_chw_image(pred_alpha.cpu(), 1)
            _save_tensor_image(directory / "target_rgb.png", target_rgb)
            _save_tensor_image(directory / "rendered_rgb.png", rendered_rgb)
            _save_tensor_image(directory / "target_mask.png", target_mask)
            _save_tensor_image(directory / "rendered_alpha.png", rendered_alpha)
            _save_tensor_image(
                directory / "rgb_difference.png", torch.abs(rendered_rgb - target_rgb)
            )
            export_render_visualization(
                Path(output_dir),
                step,
                {
                    "cloth_name": cloth_name,
                    "frame_id": episode["target_frame_id"],
                    "view_id": episode["target_view_id"],
                    "target_rgb": episode["target_rgb"],
                    "foreground_mask": episode["target_foreground_mask"],
                },
                pred_rgb,
                pred_alpha,
            )
        results["clothes"][cloth_name] = selection
    if was_training:
        model.train()
    return results


def train_image_conditioned(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    device: torch.device | str | None = None,
    debug_one_batch: bool = False,
    resume: str | Path | None = None,
    base_model: Any | None = None,
    deformation_fn=None,
) -> dict[str, Any]:
    """Train image-anchor or image-rendering mode without changing old trainers."""

    seed = int(config["train"].get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    validate_batch_size(int(config["train"].get("batch_size", 1)))
    image_config = config["image_conditioning"]
    base_config = config.get("base", {})
    require_real_base = bool(base_config.get("require_real_base", False))
    if require_real_base:
        if image_config.get("image_height") or image_config.get("image_width"):
            raise ValueError(
                "real MVP forbids loader resize until camera K scaling is verified"
            )
        if bool(image_config.get("allow_clothing_mask_fallback", False)):
            raise ValueError("real MVP forbids clothing-mask fallback")
        if bool(image_config.get("allow_missing_rh_th", False)):
            raise ValueError("real MVP forbids Rh/Th fallback")
        if bool(image_config.get("allow_identity_deformation", False)):
            raise ValueError("real MVP forbids identity deformation")
        if not bool(image_config.get("require_depth_visibility", False)):
            raise ValueError("real MVP requires depth visibility")
    if not image_config.get("manifest_path"):
        raise ValueError("image-conditioned training requires image_conditioning.manifest_path")
    reference_count = config["train"].get("reference_count")
    if reference_count is None:
        reference_count = (
            int(config["train"]["reference_count_min"]),
            int(config["train"]["reference_count_max"]),
        )
    dataset = ImageConditionedEpisodeDataset(
        image_config["manifest_path"],
        split=image_config.get("split", "train"),
        reference_count=reference_count,
        image_size=(
            image_config["image_height"], image_config["image_width"]
        ) if image_config.get("image_height") and image_config.get("image_width") else None,
        seed=seed,
        allow_clothing_mask_fallback=bool(
            image_config.get("allow_clothing_mask_fallback", True)
        ),
        allow_missing_rh_th=bool(image_config.get("allow_missing_rh_th", False)),
    )
    mode = config["train"]["mode"]
    render_target = mode == "image_rendering_finetune"
    if require_real_base and not render_target:
        raise ValueError("require_real_base=true requires image_rendering_finetune mode")
    if render_target and base_model is None:
        if not base_config.get("model_dir"):
            raise ValueError("image_rendering_finetune requires base.model_dir")
        base_model = load_frozen_mmlphuman_base(
            base_config["model_dir"], base_config.get("checkpoint_path"), device
        )
    if require_real_base and base_model is None:
        raise RuntimeError("real image-conditioned training requires a loaded base")
    actual_fingerprint = getattr(base_model, "_dressable_base_fingerprint", None)
    expected_fingerprint = base_config.get("expected_fingerprint")
    if expected_fingerprint and actual_fingerprint != expected_fingerprint:
        raise ValueError(
            "loaded MMLPHuman fingerprint does not match base.expected_fingerprint: "
            f"{actual_fingerprint!r} != {expected_fingerprint!r}"
        )
    identity_allowed = bool(image_config.get("allow_identity_deformation", False))
    if require_real_base and deformation_fn is not None:
        raise ValueError("real mode builds the verified adapter and rejects external deformation_fn")
    if deformation_fn is None and not identity_allowed and not require_real_base:
        raise RuntimeError(
            "image-conditioned training requires a verified anchor deformation_fn; "
            "allow_identity_deformation is only for synthetic tests"
        )
    model, optimizer, anchor_features, anchor_edges = create_image_conditioned_components(
        dataset, config, base_model, device
    )
    reference_only_gate = None
    region_path_value = image_config.get("reference_region_path")
    if region_path_value:
        if image_config.get("target_view_used") is not False:
            raise ValueError("reference-only gate requires target_view_used=false")
        region_path = Path(region_path_value)
        actual_region_sha = hashlib.sha256(region_path.read_bytes()).hexdigest()
        expected_region_sha = image_config.get("reference_region_sha256")
        if expected_region_sha and actual_region_sha != expected_region_sha:
            raise ValueError("reference-only region SHA256 mismatch")
        region_object = torch.load(region_path, map_location="cpu", weights_only=True)
        if not isinstance(region_object, dict) or "cloth_region_weight" not in region_object:
            raise ValueError("reference-only region lacks cloth_region_weight")
        region_score = region_object["cloth_region_weight"].float().reshape(-1)
        if region_score.shape != (model.canonical_anchors.shape[0],):
            raise ValueError("reference-only region anchor count mismatch")
        if not torch.isfinite(region_score).all():
            raise ValueError("reference-only region contains NaN or Inf")
        region_score = region_score.clamp(0, 1)
        threshold = float(image_config.get("reference_gate_threshold", 0.5))
        reference_only_gate = (region_score >= threshold).to(
            device=device, dtype=model.dressable_model.anchor_features.dtype
        )
    deformation_adapter = None
    if require_real_base:
        lbs_grid_path = base_config.get("lbs_grid_path")
        if not lbs_grid_path:
            raise ValueError("real image-conditioned training requires base.lbs_grid_path")
        deformation_adapter = MMLPHumanAnchorDeformationAdapter.from_mmlphuman_base(
            base_model,
            canonical_anchors=model.canonical_anchors,
            lbs_grid_path=lbs_grid_path,
        )
    base_fingerprint = getattr(model.dressable_model.base_model, "_dressable_base_fingerprint", None)
    output_path = Path(output_dir or config["train"]["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "config_resolved.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)
    (output_path / "cloth_id_map.json").write_text(
        json.dumps(dataset.get_cloth_id_map(), indent=2), encoding="utf-8"
    )
    start_step = 0
    resume_path = resume or config["train"].get("resume")
    if resume_path:
        start_step = load_image_training_checkpoint(
            resume_path, model, optimizer, config, dataset, base_fingerprint
        )
    lpips_weight = float(config["loss"].get("lpips_weight", 0))
    lpips = LPIPSLoss().to(device) if render_target and lpips_weight > 0 else None
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=int(config.get("data", {}).get("num_workers", 0)),
        collate_fn=image_conditioned_episode_collate,
        generator=torch.Generator().manual_seed(seed),
    )
    iterator = iter(loader)
    configured_steps = int(config["train"]["steps"])
    end_step = min(configured_steps, start_step + 1) if debug_one_batch else configured_steps
    history = []
    last_losses: dict[str, torch.Tensor] | None = None
    last_outputs: dict[str, Any] | None = None
    last_episode: dict[str, Any] | None = None
    amp_enabled = bool(config["train"].get("mixed_precision", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    actual_step = start_step
    for step in range(start_step + 1, end_step + 1):
        try:
            episode = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            episode = next(iterator)
        if reference_only_gate is not None:
            episode["reference_only_gate"] = reference_only_gate
        second = None
        if bool(config["train"].get("dual_reference_consistency", False)) and step % int(
            config["train"].get("consistency_every", 1)
        ) == 0:
            second = dataset.sample_episode(
                int(episode["episode_index"]), sampling_salt=step + 100000, deterministic=True
            )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            losses, outputs = compute_image_conditioned_training_loss(
                model,
                episode,
                anchor_edges,
                config,
                device,
                second_episode=second,
                render_target=render_target,
                deformation_fn=deformation_fn,
                deformation_adapter=deformation_adapter,
                lpips_loss=lpips,
            )
        last_losses = losses
        last_outputs = outputs
        last_episode = episode
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        grad_clip = float(config["train"].get("grad_clip", 1.0))
        if grad_clip > 0:
            parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
            torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
        scaler.step(optimizer)
        scaler.update()
        actual_step = step
        record = {"step": step, **{key: value.detach().item() for key, value in losses.items()}}
        history.append(record)
        with (output_path / "train_log.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
        if step == 1 or step % int(config["train"].get("log_every", 20)) == 0:
            print(
                f"step={step} total={record['total']:.8f} anchor={record['anchor']:.8f} "
                f"latent={record['latent_consistency']:.8f} "
                f"local={record['anchor_feature_consistency']:.8f}"
            )
        if step % int(config["train"].get("save_every", 500)) == 0:
            save_image_training_checkpoint(
                output_path / "checkpoint_latest.pth",
                model,
                optimizer,
                step,
                config,
                dataset,
                base_fingerprint,
            )
            save_image_training_checkpoint(
                output_path / f"checkpoint_step_{step:06d}.pth",
                model,
                optimizer,
                step,
                config,
                dataset,
                base_fingerprint,
            )
        if step % int(config["train"].get("eval_every", 200)) == 0:
            evaluate_image_conditioned(
                model,
                dataset,
                step,
                output_path,
                render_target=render_target,
                deformation_fn=deformation_fn,
                render_background=config.get("render", {}).get("background"),
                deformation_adapter=deformation_adapter,
                image_conditioning_config=image_config,
                reference_only_gate=reference_only_gate,
                require_mmlphuman_state_transaction=require_real_base,
            )
    save_image_training_checkpoint(
        output_path / "checkpoint_latest.pth",
        model,
        optimizer,
        actual_step,
        config,
        dataset,
        base_fingerprint,
    )
    evaluate_image_conditioned(
        model,
        dataset,
        actual_step,
        output_path,
        render_target=render_target,
        deformation_fn=deformation_fn,
        render_background=config.get("render", {}).get("background"),
        deformation_adapter=deformation_adapter,
        image_conditioning_config=image_config,
        reference_only_gate=reference_only_gate,
        require_mmlphuman_state_transaction=require_real_base,
    )
    return {
        "model": model,
        "optimizer": optimizer,
        "dataset": dataset,
        "anchor_features": anchor_features,
        "anchor_edges": anchor_edges,
        "deformation_adapter": deformation_adapter,
        "reference_only_gate": reference_only_gate,
        "history": history,
        "last_losses": last_losses,
        "last_outputs": last_outputs,
        "last_episode": last_episode,
        "step": actual_step,
        "output_dir": output_path,
    }


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise TypeError("config root must be a mapping")
    return config


def _single_sample_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if len(batch) != 1:
        raise ValueError("anchor_film training collate requires exactly one sample")
    return batch[0]


def _save_step_checkpoints(
    output_path: Path,
    model: DressableGaussianModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict[str, Any],
    dataset: AnchorOffsetDataset,
    anchor_features: torch.Tensor,
    anchor_edges: torch.Tensor,
) -> None:
    arguments = (
        model,
        optimizer,
        step,
        config,
        dataset.get_cloth_id_map(),
        dataset.anchor_xyz,
        anchor_features,
        anchor_edges,
    )
    save_training_checkpoint(output_path / "checkpoint_latest.pth", *arguments)
    save_training_checkpoint(output_path / f"checkpoint_step_{step:06d}.pth", *arguments)


def _assert_checkpoint_tensor(
    key: str,
    checkpoint: dict[str, Any],
    current: torch.Tensor,
    floating: bool,
) -> None:
    if key not in checkpoint or not isinstance(checkpoint[key], torch.Tensor):
        raise KeyError(f"checkpoint is missing tensor {key!r}")
    saved = checkpoint[key].cpu()
    current_cpu = current.detach().cpu()
    if saved.shape != current_cpu.shape:
        raise ValueError(f"resume {key} shape mismatch")
    matches = torch.allclose(saved, current_cpu, atol=1e-6, rtol=0) if floating else torch.equal(
        saved, current_cpu
    )
    if not matches:
        raise ValueError(f"resume {key} values do not match current data")


def _resolve_target_paths(pattern: str) -> list[Path]:
    if not pattern:
        raise ValueError("target_glob must be provided in config or via --target_glob")
    paths = [Path(path) for path in sorted(glob.glob(pattern, recursive=True))]
    if not paths:
        raise FileNotFoundError(f"target_glob matched no files: {pattern}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CanonDressGS clothing offsets")
    parser.add_argument("--config", required=True)
    parser.add_argument("--target_glob", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--debug_one_batch", action="store_true")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.target_glob is not None:
        config["data"]["target_glob"] = args.target_glob
    if args.output_dir is not None:
        config["train"]["output_dir"] = args.output_dir
    if args.resume is not None:
        config["train"]["resume"] = args.resume
    mode = config.get("train", {}).get("mode", "anchor_only")
    if mode == "anchor_only":
        target_paths = _resolve_target_paths(config["data"].get("target_glob", ""))
        result = train_anchor_targets(
            config,
            target_paths,
            output_dir=args.output_dir,
            debug_one_batch=args.debug_one_batch,
            resume=args.resume,
        )
    elif mode == "rendering_finetune":
        result = train_rendering_finetune(
            config,
            output_dir=args.output_dir,
            debug_one_batch=args.debug_one_batch,
            resume=args.resume,
        )
    elif mode in {"image_anchor_only", "image_rendering_finetune"}:
        result = train_image_conditioned(
            config,
            output_dir=args.output_dir,
            debug_one_batch=args.debug_one_batch,
            resume=args.resume,
        )
    else:
        raise ValueError(
            "train.mode must be anchor_only, rendering_finetune, "
            "image_anchor_only, or image_rendering_finetune"
        )
    print(f"training complete at step {result['step']}; outputs: {result['output_dir']}")


if __name__ == "__main__":
    main()
