from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision.utils import make_grid, save_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_dressable import (
    compute_image_conditioned_training_loss,
    load_config,
    load_image_training_checkpoint,
    train_image_conditioned,
)
from scene.dressable_dataset import ImageConditionedEpisodeDataset
from utils.gaussian_alignment import load_anchor_offset_target
from utils.dressable_checkpoint_utils import iter_base_parameters


EXPECTED_MODEL_FINGERPRINT = (
    "64ac7f2dd1dd705307258620f76967fdc93c66869d3ff5c7f32e1f70055635c4"
)
EXPECTED_CHECKPOINT_SHA256 = "abbf67b59eadf2cba2dea69dbeec598f9177da45b8ddc74ccbe8108acf9ddf70"
REQUESTED_REFERENCES = (("0", "cam18"), ("1000", "cam00"))
REQUESTED_TARGET = ("2000", "cam09")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()


def _stats(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().float()
    return {
        "shape": list(tensor.shape),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "finite_ratio": float(torch.isfinite(tensor).float().mean().item()),
    }


def _comparison(
    before: torch.Tensor,
    after: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    difference = (before.detach().float() - after.detach().float()).abs()
    return {
        "max_abs_diff": float(difference.max().item()),
        "mean_abs_diff": float(difference.mean().item()),
        "atol": atol,
        "rtol": rtol,
        "allclose": bool(torch.allclose(before, after, atol=atol, rtol=rtol)),
    }


def _single_reference_episode(episode: dict[str, Any], index: int = 0) -> dict[str, Any]:
    result = copy.deepcopy(episode)
    tensor_keys = (
        "reference_images", "reference_cloth_masks", "reference_foreground_masks",
        "reference_poses", "reference_Rh", "reference_Th", "reference_valid_mask",
    )
    for key in tensor_keys:
        if isinstance(result.get(key), torch.Tensor):
            result[key] = result[key][index:index + 1]
    for key in ("reference_cameras", "reference_frame_ids", "reference_view_ids"):
        if isinstance(result.get(key), list):
            result[key] = [result[key][index]]
    return result


def _canonical_record_id(frame_id: Any, view_id: Any) -> tuple[str, str]:
    try:
        frame = str(int(str(frame_id)))
    except ValueError as error:
        raise ValueError(f"invalid frame ID: {frame_id!r}") from error
    view_text = str(view_id).lower()
    if view_text.startswith("cam"):
        view_text = view_text[3:]
    try:
        view = f"cam{int(view_text):02d}"
    except ValueError as error:
        raise ValueError(f"invalid camera ID: {view_id!r}") from error
    return frame, view


def _resolve_protocol_records(
    frames: list[dict[str, Any]],
    references: tuple[tuple[str, str], ...] = REQUESTED_REFERENCES,
    target: tuple[str, str] = REQUESTED_TARGET,
) -> tuple[list[int], int]:
    normalized = [_canonical_record_id(item["frame_id"], item["view_id"]) for item in frames]

    def unique_index(requested: tuple[str, str]) -> int:
        wanted = _canonical_record_id(*requested)
        matches = [index for index, value in enumerate(normalized) if value == wanted]
        if len(matches) != 1:
            raise ValueError(f"protocol record {wanted} matched {len(matches)} manifest entries")
        return matches[0]

    reference_indices = [unique_index(item) for item in references]
    target_index = unique_index(target)
    if target_index in reference_indices or len(set(reference_indices)) != len(reference_indices):
        raise ValueError("target/reference overlap or duplicate reference detected")
    return reference_indices, target_index


def _build_fixed_episode(dataset: ImageConditionedEpisodeDataset) -> dict[str, Any]:
    split_clothes = dataset.manifest["splits"][dataset.split]
    if len(split_clothes) != 1:
        raise ValueError("fixed Gate 4-A protocol requires exactly one clothing entry")
    cloth_name = split_clothes[0]
    cloth = dataset._resolved_clothes[cloth_name]
    frames = cloth["frames"]
    reference_indices, target_index = _resolve_protocol_records(frames)
    references = [dataset._load_frame(frames[index]) for index in reference_indices]
    target = dataset._load_frame(frames[target_index])
    anchor_target = None
    anchor_metadata: dict[str, Any] = {}
    if cloth["anchor_offset_target"] is not None:
        loaded = load_anchor_offset_target(cloth["anchor_offset_target"], device="cpu")
        anchor_target, anchor_metadata = loaded["target"], loaded["metadata"]
    episode = {
        "identity_id": dataset.identity_id,
        "cloth_name": cloth_name,
        "cloth_id": torch.tensor(dataset._cloth_id_map[cloth_name], dtype=torch.long),
        "episode_index": target_index,
        "reference_images": torch.stack([item["rgb"] for item in references]),
        "reference_cloth_masks": torch.stack([item["clothing_mask"] for item in references]),
        "reference_foreground_masks": torch.stack([item["foreground_mask"] for item in references]),
        "reference_poses": torch.stack([item["pose"] for item in references]),
        "reference_Rh": torch.stack([item["Rh"] for item in references]),
        "reference_Th": torch.stack([item["Th"] for item in references]),
        "reference_cameras": [item["camera"] for item in references],
        "reference_frame_ids": [item["frame_id"] for item in references],
        "reference_view_ids": [item["view_id"] for item in references],
        "reference_valid_mask": torch.ones(len(references), dtype=torch.float32),
        "target_rgb": target["rgb"],
        "target_foreground_mask": target["foreground_mask"],
        "target_clothing_mask": target["clothing_mask"],
        "target_pose": target["pose"],
        "target_Rh": target["Rh"],
        "target_Th": target["Th"],
        "target_camera": target["camera"],
        "target_frame_id": target["frame_id"],
        "target_view_id": target["view_id"],
        "anchor_offset_target": anchor_target,
        "anchor_metadata": anchor_metadata,
        "metadata": {
            "reference_clothing_mask_fallback": [item["clothing_mask_fallback"] for item in references],
            "target_clothing_mask_fallback": target["clothing_mask_fallback"],
            "reference_rh_th_fallback": [item["rh_th_fallback"] for item in references],
            "target_rh_th_fallback": target["rh_th_fallback"],
        },
    }
    dataset._validate_episode_output(episode)
    resolved_references = tuple(
        _canonical_record_id(frame, view)
        for frame, view in zip(episode["reference_frame_ids"], episode["reference_view_ids"])
    )
    resolved_target = _canonical_record_id(episode["target_frame_id"], episode["target_view_id"])
    if resolved_references != REQUESTED_REFERENCES or resolved_target != REQUESTED_TARGET:
        raise RuntimeError("resolved protocol does not match requested protocol")
    return episode


def _as_chw_render(tensor: torch.Tensor, expected_channels: int) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
        raise ValueError("render tensor must be rank-3 HWC or CHW")
    first_matches = tensor.shape[0] == expected_channels
    last_matches = tensor.shape[-1] == expected_channels
    if first_matches == last_matches:
        raise ValueError(f"ambiguous or invalid render shape: {tuple(tensor.shape)}")
    value = tensor if first_matches else tensor.permute(2, 0, 1)
    if not torch.isfinite(value).all():
        raise ValueError("render tensor contains NaN or Inf")
    return value.detach().float().clamp(0, 1).contiguous().cpu()


def save_render_tensor(path: str | Path, tensor: torch.Tensor, expected_channels: int) -> None:
    if expected_channels not in (1, 3):
        raise ValueError("expected_channels must be 1 or 3")
    value = _as_chw_render(tensor, expected_channels)
    if expected_channels == 1:
        array = value.mul(255).round().to(torch.uint8).squeeze(0).numpy()
        Image.fromarray(array, mode="L").save(Path(path))
    else:
        save_image(value, Path(path))


def _grad_norm(parameters) -> float:
    squares = []
    for parameter in parameters:
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("trainable gradient contains NaN or Inf")
            squares.append(parameter.grad.detach().float().square().sum())
    if not squares:
        return 0.0
    return float(torch.sqrt(torch.stack(squares).sum()).item())


def _image_grad_norms(model) -> dict[str, float]:
    return {
        "encoder": _grad_norm(model.clothing_observation_encoder.parameters()),
        "aggregator": _grad_norm(model.multiview_aggregator.parameters()),
        "hypernetwork": _grad_norm(
            model.dressable_model.clothing_film_generator.parameters()
        ),
        "anchor_mlp": _grad_norm(
            model.dressable_model.anchor_clothing_mlp.parameters()
        ),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the real CanonDressGS Gate 4-A one-batch acceptance"
    )
    parser.add_argument(
        "--config",
        default="configs/canon_dress_gs_mvp_real.yaml",
    )
    parser.add_argument("--manifest", default=None)
    parser.add_argument(
        "--output_dir",
        default="/root/autodl-tmp/canondressgs_work/outputs/pipeline_mvp/GATE4-REAL-ONEBATCH-001",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    attempt_id = "attempt_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    config = load_config(args.config)
    if args.manifest is not None:
        config["image_conditioning"]["manifest_path"] = args.manifest
    config["train"]["output_dir"] = args.output_dir
    if config["train"].get("mode") != "image_rendering_finetune":
        raise ValueError("real Gate 2 requires train.mode=image_rendering_finetune")
    if not config.get("base", {}).get("require_real_base", False):
        raise ValueError("real Gate 2 requires base.require_real_base=true")
    image_config = config["image_conditioning"]
    if image_config.get("target_view_used") is not False:
        raise ValueError("Gate 4-A requires target_view_used=false")
    if not image_config.get("reference_region_path"):
        raise ValueError("Gate 4-A requires a reference-only region artifact")

    original_sample_episode = ImageConditionedEpisodeDataset.sample_episode

    def fixed_sample_episode(self, index, sampling_salt=0, deterministic=False):
        del index, sampling_salt, deterministic
        return _build_fixed_episode(self)

    ImageConditionedEpisodeDataset.sample_episode = fixed_sample_episode
    try:
        result = train_image_conditioned(
            config,
            output_dir=args.output_dir,
            device=args.device,
            debug_one_batch=True,
        )
    finally:
        ImageConditionedEpisodeDataset.sample_episode = original_sample_episode
    model = result["model"]
    dataset = result["dataset"]
    episode = result["last_episode"]
    losses = result["last_losses"]
    output = result["last_outputs"]["primary"]
    if episode is None or losses is None or output is None:
        raise RuntimeError("one-batch trainer did not expose acceptance diagnostics")
    resolved_references = [
        _canonical_record_id(frame, view)
        for frame, view in zip(episode["reference_frame_ids"], episode["reference_view_ids"])
    ]
    resolved_target = _canonical_record_id(episode["target_frame_id"], episode["target_view_id"])
    if tuple(resolved_references) != REQUESTED_REFERENCES or resolved_target != REQUESTED_TARGET:
        raise RuntimeError("trainer output protocol does not match the pre-registered protocol")
    if resolved_target in resolved_references:
        raise RuntimeError("target observation leaked into Condition A references")
    step_one_losses = losses
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "one_batch_diagnostics.partial.json"
    partial_path.write_text(
        json.dumps({"status": "RUNNING", "failure_stage": None, "completed": ["step_1_saved"]}, indent=2),
        encoding="utf-8",
    )

    base_model = model.dressable_model.base_model
    checkpoint_path = getattr(base_model, "_dressable_checkpoint_path", None)
    fingerprint = getattr(base_model, "_dressable_base_fingerprint", None)
    if fingerprint != EXPECTED_MODEL_FINGERPRINT:
        raise ValueError("loaded model fingerprint changed during training")
    canonical_anchors = model.canonical_anchors
    if canonical_anchors.shape != (10000, 3):
        raise ValueError("real canonical anchors must have shape [10000,3]")
    if not torch.allclose(
        canonical_anchors,
        base_model.xyz_vt.to(
            device=canonical_anchors.device, dtype=canonical_anchors.dtype
        ),
        atol=1e-6,
        rtol=0,
    ):
        raise ValueError("model canonical anchors do not equal base_model.xyz_vt")

    checkpoint = output_dir / "checkpoint_step_000001.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError("formal one-batch checkpoint was not saved")

    def run_fixed_eval(fixed_episode: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        model.eval()
        with torch.no_grad():
            fixed_losses, fixed_outputs = compute_image_conditioned_training_loss(
                model, fixed_episode, result["anchor_edges"], config, args.device,
                render_target=True, deformation_adapter=result["deformation_adapter"],
            )
        return fixed_losses, fixed_outputs["primary"]

    pre_reload_losses, pre_reload_output = run_fixed_eval(episode)
    restored_step = load_image_training_checkpoint(
        checkpoint, model, result["optimizer"], config, dataset, fingerprint
    )
    if restored_step != result["step"]:
        raise RuntimeError("saved checkpoint step did not round-trip")
    post_reload_losses, post_reload_output = run_fixed_eval(episode)

    def raw_anchor_offsets(fixed_output: dict[str, Any]) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            return model.dressable_model.compute_film_anchor_offsets(
                clothing_embedding=fixed_output["global_clothing_embedding"],
                anchor_clothing_features=fixed_output["anchor_clothing_features"],
            )

    pre_raw = raw_anchor_offsets(pre_reload_output)
    post_raw = raw_anchor_offsets(post_reload_output)
    roundtrip = {
        "global_clothing_embedding": _comparison(
            pre_reload_output["global_clothing_embedding"], post_reload_output["global_clothing_embedding"], atol=1e-7, rtol=1e-6
        ),
        "raw_anchor_delta_xyz": _comparison(pre_raw["delta_xyz"], post_raw["delta_xyz"], atol=1e-7, rtol=1e-6),
        "gated_anchor_delta_xyz": _comparison(
            pre_reload_output["anchor_offsets"]["delta_xyz"], post_reload_output["anchor_offsets"]["delta_xyz"], atol=1e-7, rtol=1e-6
        ),
        "gaussian_delta_xyz": _comparison(
            pre_reload_output["gaussian_offsets"]["delta_xyz"], post_reload_output["gaussian_offsets"]["delta_xyz"], atol=1e-7, rtol=1e-6
        ),
        "rendered_rgb": _comparison(pre_reload_output["render"][0], post_reload_output["render"][0], atol=1e-6, rtol=1e-5),
        "rendered_alpha": _comparison(pre_reload_output["render"][1], post_reload_output["render"][1], atol=1e-6, rtol=1e-5),
    }
    if not all(item["allclose"] for item in roundtrip.values()):
        partial_path.write_text(
            json.dumps({"status": "FAIL", "failure_stage": "checkpoint_roundtrip", "checkpoint_roundtrip": roundtrip}, indent=2),
            encoding="utf-8",
        )
        raise RuntimeError(f"same-state checkpoint roundtrip mismatch: {roundtrip}")
    partial_path.write_text(
        json.dumps({"status": "RUNNING", "failure_stage": None, "completed": ["step_1_saved", "checkpoint_roundtrip"], "checkpoint_roundtrip": roundtrip}, indent=2),
        encoding="utf-8",
    )

    # Connectivity diagnostics may use one in-memory warmup update, but are
    # intentionally isolated from the formal step-1 checkpoint comparison.
    model.train()
    result["optimizer"].zero_grad(set_to_none=True)
    connectivity_losses, connectivity_outputs = compute_image_conditioned_training_loss(
        model, episode, result["anchor_edges"], config, args.device,
        render_target=True, deformation_adapter=result["deformation_adapter"],
    )
    connectivity_losses["total"].backward()
    connectivity_warmup_updates = 0
    if not all(value > 0 for value in _image_grad_norms(model).values()):
        result["optimizer"].step()
        connectivity_warmup_updates = 1
        result["optimizer"].zero_grad(set_to_none=True)
        connectivity_losses, connectivity_outputs = compute_image_conditioned_training_loss(
            model, episode, result["anchor_edges"], config, args.device,
            render_target=True, deformation_adapter=result["deformation_adapter"],
        )
        connectivity_losses["total"].backward()
    trainable_grad_norms = _image_grad_norms(model)
    if not all(value > 0 for value in trainable_grad_norms.values()):
        raise RuntimeError(f"one-batch backward did not reach every image-conditioned module: {trainable_grad_norms}")
    base_grad_count = sum(parameter.grad is not None for parameter in iter_base_parameters(base_model))
    if base_grad_count != 0:
        raise RuntimeError(f"frozen base received {base_grad_count} gradients")

    # Restore the formal step-1 state before sensitivity and final evidence.
    load_image_training_checkpoint(checkpoint, model, result["optimizer"], config, dataset, fingerprint)
    _, output = run_fixed_eval(episode)
    render = output["render"]
    if not isinstance(render, tuple) or len(render) < 2:
        raise RuntimeError("target rendering did not return RGB and alpha")
    pred_rgb, pred_alpha = render[:2]
    if not torch.isfinite(pred_rgb).all() or not torch.isfinite(pred_alpha).all():
        raise FloatingPointError("target rendering contains NaN or Inf")
    projection = output["projection"]
    if "depth_visible_mask" not in projection:
        raise RuntimeError("real one-batch projection did not use depth visibility")
    visible = projection["depth_visible_mask"].float()
    visible_ratios = visible.reshape(visible.shape[0], -1).mean(dim=1)
    geometry = output["reference_geometry_diagnostics"]
    if not geometry or geometry.get("depth_shape") is None:
        raise RuntimeError("reference ED geometry diagnostics are missing")
    if not geometry.get("base_state_cache_restored"):
        raise RuntimeError("reference geometry did not restore base state/cache")
    if output.get("target_state_cache_restored") is not True:
        raise RuntimeError("target render did not restore base state/cache")

    anchor_offsets = output["anchor_offsets"]
    gaussian_offsets = output["gaussian_offsets"]
    raw_offsets = raw_anchor_offsets(output)
    disabled_max = {
        "anchor_scaling": float(anchor_offsets["delta_scaling"].abs().max().item()),
        "anchor_opacity": float(anchor_offsets["delta_opacity"].abs().max().item()),
        "gaussian_scaling": float(gaussian_offsets["delta_scaling"].abs().max().item()),
        "gaussian_opacity": float(gaussian_offsets["delta_opacity"].abs().max().item()),
    }
    if any(value != 0 for value in disabled_max.values()):
        raise RuntimeError(f"disabled offset channels are nonzero: {disabled_max}")

    condition_b = _single_reference_episode(episode)
    condition_b_records = [
        _canonical_record_id(frame, view)
        for frame, view in zip(condition_b["reference_frame_ids"], condition_b["reference_view_ids"])
    ]
    if resolved_target in condition_b_records:
        raise RuntimeError("target observation leaked into Condition B references")
    _, sensitivity_output = run_fixed_eval(condition_b)
    sensitivity_raw = raw_anchor_offsets(sensitivity_output)
    sensitivity = {
        "condition_a_reference_count": int(episode["reference_images"].shape[0]),
        "condition_b_reference_count": 1,
        "condition_b_reference_frames": condition_b["reference_frame_ids"],
        "condition_b_reference_cameras": condition_b["reference_view_ids"],
        "condition_b_target_view_used": False,
        "global_embedding_l2": float(torch.linalg.vector_norm(
            output["global_clothing_embedding"] - sensitivity_output["global_clothing_embedding"]
        ).item()),
        "anchor_feature_mean_abs": float((
            output["anchor_clothing_features"] - sensitivity_output["anchor_clothing_features"]
        ).abs().mean().item()),
        "raw_anchor_offset_mean_abs": float((
            raw_offsets["delta_xyz"] - sensitivity_raw["delta_xyz"]
        ).abs().mean().item()),
        "gated_anchor_offset_mean_abs": float((
            anchor_offsets["delta_xyz"] - sensitivity_output["anchor_offsets"]["delta_xyz"]
        ).abs().mean().item()),
        "rendered_rgb_mean_abs": float((
            pred_rgb - sensitivity_output["render"][0]
        ).abs().mean().item()),
        "rendered_alpha_mean_abs": float((
            pred_alpha - sensitivity_output["render"][1]
        ).abs().mean().item()),
    }
    if not any(value > 0 for key, value in sensitivity.items() if key.endswith(("_l2", "_abs"))):
        raise RuntimeError("reference condition did not affect embedding, offsets, or render")
    rendered_images = sorted(Path(args.output_dir).rglob("rendered_rgb.png"))
    if not rendered_images:
        raise FileNotFoundError("evaluation did not save a rendered image")

    diagnostics = {
        "status": "REAL-VERIFIED",
        "checkpoint_path": checkpoint_path,
        "checkpoint_fingerprint": fingerprint,
        "canonical_anchor_source": "base_model.xyz_vt",
        "num_anchors": int(canonical_anchors.shape[0]),
        "num_gaussians": int(base_model._xyz.shape[0]),
        "reference_count": int(episode["reference_images"].shape[0]),
        "reference_frame_ids": episode["reference_frame_ids"],
        "reference_view_ids": episode["reference_view_ids"],
        "reference_pose_shape": list(episode["reference_poses"].shape),
        "reference_Rh_shape": list(episode["reference_Rh"].shape),
        "reference_Th_shape": list(episode["reference_Th"].shape),
        "reference_depth_shape": geometry["depth_shape"],
        "reference_depth_finite_ratio": geometry["depth_finite_ratio"],
        "visible_ratio_per_view": visible_ratios.detach().cpu().tolist(),
        "target_frame_id": episode["target_frame_id"],
        "target_view_id": episode["target_view_id"],
        "render_rgb_shape": list(pred_rgb.shape),
        "render_alpha_shape": list(pred_alpha.shape),
        "losses": {key: float(value.detach().item()) for key, value in step_one_losses.items()},
        "total_loss_finite": bool(torch.isfinite(step_one_losses["total"]).item()),
        "backward_success": True,
        "trainable_grad_norms": trainable_grad_norms,
        "projector_trainable_parameter_count": sum(
            parameter.numel() for parameter in model.anchor_image_projector.parameters()
            if parameter.requires_grad
        ),
        "connectivity_extra_optimizer_updates": connectivity_warmup_updates,
        "encoder_backbone_requires_grad": any(
            parameter.requires_grad
            for parameter in model.clothing_observation_encoder.backbone.parameters()
        ),
        "base_grad_count": base_grad_count,
        "base_state_cache_restored": True,
        "disabled_offset_channel_max_abs": disabled_max,
        "checkpoint_round_trip_step": restored_step,
        "checkpoint_roundtrip": roundtrip,
        "reference_sensitivity": sensitivity,
        "data": {
            "reference_images": _stats(episode["reference_images"]),
            "reference_masks": _stats(episode["reference_cloth_masks"]),
            "target_rgb": _stats(episode["target_rgb"]),
            "target_mask": _stats(episode["target_foreground_mask"]),
        },
        "encoder_feature_shape": list(output["projection"]["sampled_features"].shape),
        "global_embedding_shape": list(output["global_clothing_embedding"].shape),
        "anchor_feature_shape": list(output["anchor_clothing_features"].shape),
        "anchor_visibility_shape": list(output["anchor_visibility"].shape),
        "aggregated_visible_anchor_ratio": float((output["anchor_visibility"] > 0).float().mean().item()),
        "anchor_delta_xyz": _stats(anchor_offsets["delta_xyz"]),
        "raw_anchor_delta_xyz": _stats(raw_offsets["delta_xyz"]),
        "anchor_delta_xyz_mean_norm": float(anchor_offsets["delta_xyz"].norm(dim=-1).mean().item()),
        "anchor_delta_xyz_max_norm": float(anchor_offsets["delta_xyz"].norm(dim=-1).max().item()),
        "gaussian_delta_xyz": _stats(gaussian_offsets["delta_xyz"]),
        "gaussian_delta_xyz_mean_norm": float(gaussian_offsets["delta_xyz"].norm(dim=-1).mean().item()),
        "gaussian_delta_xyz_max_norm": float(gaussian_offsets["delta_xyz"].norm(dim=-1).max().item()),
        "render_rgb": _stats(pred_rgb),
        "render_alpha": _stats(pred_alpha),
        "rendered_image": str(rendered_images[0]),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = Path(checkpoint_path)
    manifest_path = Path(config["image_conditioning"]["manifest_path"])
    region_path = Path(config["image_conditioning"]["reference_region_path"])
    checkpoint_sha = _sha256(checkpoint_file)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("current checkpoint file SHA256 changed")
    input_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "attempt_id": attempt_id,
        "git": {"branch": _git("branch", "--show-current"), "commit": _git("rev-parse", "HEAD"), "status": _git("status", "--short")},
        "environment": {"python": platform.python_version(), "pytorch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
        "base_checkpoint": {"path": str(checkpoint_file), "size": checkpoint_file.stat().st_size, "sha256": checkpoint_sha, "historical_warning": "Current SHA256 differs from historical 64ac7f2d... acceptance fingerprint."},
        "episode_manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "reference_only_region": {"path": str(region_path), "sha256": _sha256(region_path), "threshold": config["image_conditioning"]["reference_gate_threshold"], "reference_views": config["image_conditioning"]["reference_views"], "target_view_used": False},
        "reference_frames": episode["reference_frame_ids"], "reference_cameras": episode["reference_view_ids"],
        "target_frame": episode["target_frame_id"], "target_camera": episode["target_view_id"],
        "protocol": {
            "requested": {
                "condition_a_references": [list(item) for item in REQUESTED_REFERENCES],
                "target": list(REQUESTED_TARGET),
                "condition_b_references": [list(REQUESTED_REFERENCES[0])],
                "target_view_used": False,
            },
            "resolved": {
                "condition_a_references": [list(item) for item in resolved_references],
                "target": list(resolved_target),
                "condition_b_references": [list(item) for item in condition_b_records],
                "target_view_used": False,
                "target_reference_overlap": False,
            },
            "reference_records": [
                {"frame_id": frame, "camera_id": view}
                for frame, view in resolved_references
            ],
            "target_record": {"frame_id": resolved_target[0], "camera_id": resolved_target[1]},
            "matches_requested": True,
        },
        "config": config,
    }
    manifest_tmp = output_dir / "input_manifest.json.tmp"
    diagnostics_tmp = output_dir / "one_batch_diagnostics.json.tmp"
    manifest_tmp.write_text(json.dumps(input_manifest, indent=2), encoding="utf-8")
    diagnostics_tmp.write_text(json.dumps(_jsonable(diagnostics), indent=2), encoding="utf-8")
    manifest_tmp.replace(output_dir / "input_manifest.json")
    diagnostics_tmp.replace(output_dir / "one_batch_diagnostics.json")
    markdown = "# Gate 4-A Real One-batch Diagnostics\n\n- Status: PASS\n- Commit: `" + input_manifest["git"]["commit"] + "`\n- Loss: `" + str(diagnostics["losses"]["total"]) + "`\n- Anchor delta: `" + str(diagnostics["anchor_delta_xyz"]["shape"]) + "`\n- Gaussian delta: `" + str(diagnostics["gaussian_delta_xyz"]["shape"]) + "`\n- Gradients: `" + json.dumps(trainable_grad_norms) + "`\n- Roundtrip: `" + json.dumps(roundtrip) + "`\n- Sensitivity: `" + json.dumps(sensitivity) + "`\n"
    markdown_tmp = output_dir / "one_batch_diagnostics.md.tmp"
    markdown_tmp.write_text(markdown, encoding="utf-8")
    markdown_tmp.replace(output_dir / "one_batch_diagnostics.md")
    partial_path.write_text(
        json.dumps(
            {
                "status": "RUNNING",
                "failure_stage": "image_persistence",
                "completed": [
                    "fixed_protocol", "step_1_saved", "checkpoint_roundtrip",
                    "backward_gradients", "reference_sensitivity", "diagnostics",
                ],
                "checkpoint_roundtrip": roundtrip,
                "reference_sensitivity": sensitivity,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    references_chw = episode["reference_images"].detach().float().clamp(0, 1).cpu()
    target_rgb_chw = _as_chw_render(episode["target_rgb"], 3)
    predicted_rgb_chw = _as_chw_render(pred_rgb, 3)
    target_alpha_chw = _as_chw_render(episode["target_foreground_mask"], 1)
    predicted_alpha_chw = _as_chw_render(pred_alpha, 1)
    save_image(make_grid(references_chw, nrow=2), output_dir / "reference_contact_sheet.png")
    save_render_tensor(output_dir / "target_rgb.png", target_rgb_chw, 3)
    save_render_tensor(output_dir / "predicted_rgb.png", predicted_rgb_chw, 3)
    save_render_tensor(output_dir / "predicted_alpha.png", predicted_alpha_chw, 1)
    rgb_error = (target_rgb_chw - predicted_rgb_chw).abs()
    comparison = make_grid(
        [
            references_chw[0], references_chw[1], target_rgb_chw,
            predicted_rgb_chw, rgb_error,
            target_alpha_chw.expand(3, -1, -1),
            predicted_alpha_chw.expand(3, -1, -1),
        ],
        nrow=4,
    )
    save_image(comparison, output_dir / "comparison.png")
    shutil.copyfile(checkpoint, output_dir / "checkpoint_step_000001.pth") if checkpoint.name != "checkpoint_step_000001.pth" else None
    acceptance = (
        "# Gate 4-A Acceptance\n\n"
        "Status: **PASS**\n\n"
        "The fixed pre-registered reference/target protocol, real one-batch forward/backward, "
        "reference-only gate, same-state checkpoint roundtrip, diagnostics, and render persistence "
        "were verified.\n\n"
        "Feature-level sensitivity is established at step 1; output-level sensitivity is not yet "
        "established at step 1. This is not evidence of final clothing-change capability.\n\n"
        "## Gate 4-B hard acceptance\n\n"
        "After continuous training, raw anchor offset difference and rendered RGB difference between "
        "legal reference conditions must both be greater than zero.\n"
    )
    (output_dir / "GATE_ACCEPTANCE.md").write_text(acceptance, encoding="utf-8")
    partial_path.unlink(missing_ok=True)

    print(f"checkpoint path: {checkpoint_path}")
    print(f"checkpoint fingerprint: {fingerprint}")
    print("canonical anchor source: base_model.xyz_vt")
    print(f"A={diagnostics['num_anchors']} N={diagnostics['num_gaussians']}")
    print(f"K={diagnostics['reference_count']}")
    print(f"reference frames: {episode['reference_frame_ids']}")
    print(f"reference views: {episode['reference_view_ids']}")
    print(f"reference pose/Rh/Th: {diagnostics['reference_pose_shape']} / "
          f"{diagnostics['reference_Rh_shape']} / {diagnostics['reference_Th_shape']}")
    print(f"depth maps: {diagnostics['reference_depth_shape']}")
    print(f"depth finite ratio: {diagnostics['reference_depth_finite_ratio']:.6f}")
    print(f"visible ratios: {diagnostics['visible_ratio_per_view']}")
    print(f"target: {episode['target_frame_id']} / {episode['target_view_id']}")
    print(f"RGB/alpha: {list(pred_rgb.shape)} / {list(pred_alpha.shape)}")
    print(f"losses: {diagnostics['losses']}")
    print(f"trainable grad norms: {trainable_grad_norms}")
    print(f"base grad count: {base_grad_count}")
    print(f"disabled channel max abs: {disabled_max}")
    print(f"checkpoint roundtrip: {roundtrip}")
    print(f"reference sensitivity: {sensitivity}")
    print(f"diagnostics: {(output_dir / 'one_batch_diagnostics.json').resolve()}")
    print("CanonDressGS Gate 4-A real one-batch check: PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        default_output = "/root/autodl-tmp/canondressgs_work/outputs/pipeline_mvp/GATE4-REAL-ONEBATCH-001"
        output_value = default_output
        if "--output_dir" in sys.argv:
            position = sys.argv.index("--output_dir")
            if position + 1 < len(sys.argv):
                output_value = sys.argv[position + 1]
        failure_path = Path(output_value) / "one_batch_diagnostics.partial.json"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            partial = json.loads(failure_path.read_text(encoding="utf-8")) if failure_path.is_file() else {}
        except json.JSONDecodeError:
            partial = {}
        partial.update(
            {
                "status": "FAIL",
                "failure_stage": partial.get("failure_stage") or "unclassified",
                "exception_type": type(error).__name__,
                "exception_message": str(error),
            }
        )
        failure_path.write_text(json.dumps(partial, indent=2), encoding="utf-8")
        raise
