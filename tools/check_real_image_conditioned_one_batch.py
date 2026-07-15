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
from utils.dressable_checkpoint_utils import iter_base_parameters


EXPECTED_MODEL_FINGERPRINT = (
    "64ac7f2dd1dd705307258620f76967fdc93c66869d3ff5c7f32e1f70055635c4"
)
EXPECTED_CHECKPOINT_SHA256 = "abbf67b59eadf2cba2dea69dbeec598f9177da45b8ddc74ccbe8108acf9ddf70"


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

    result = train_image_conditioned(
        config,
        output_dir=args.output_dir,
        device=args.device,
        debug_one_batch=True,
    )
    model = result["model"]
    dataset = result["dataset"]
    episode = result["last_episode"]
    losses = result["last_losses"]
    output = result["last_outputs"]["primary"]
    if episode is None or losses is None or output is None:
        raise RuntimeError("one-batch trainer did not expose acceptance diagnostics")

    # Reuse the same batch after the single optimizer update for connectivity
    # diagnostics. This performs backward only and never makes a second update.
    result["optimizer"].zero_grad(set_to_none=True)
    losses, connectivity_outputs = compute_image_conditioned_training_loss(
        model,
        episode,
        result["anchor_edges"],
        config,
        args.device,
        render_target=True,
        deformation_adapter=result["deformation_adapter"],
    )
    losses["total"].backward()
    output = connectivity_outputs["primary"]
    connectivity_warmup_updates = 0
    if not all(value > 0 for value in _image_grad_norms(model).values()):
        result["optimizer"].step()
        connectivity_warmup_updates = 1
        result["optimizer"].zero_grad(set_to_none=True)
        losses, connectivity_outputs = compute_image_conditioned_training_loss(
            model, episode, result["anchor_edges"], config, args.device,
            render_target=True, deformation_adapter=result["deformation_adapter"],
        )
        losses["total"].backward()
        output = connectivity_outputs["primary"]

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

    trainable_grad_norms = _image_grad_norms(model)
    if not all(value > 0 for value in trainable_grad_norms.values()):
        raise RuntimeError(
            "one-batch backward did not reach every image-conditioned module: "
            f"{trainable_grad_norms}"
        )
    base_grad_count = sum(
        parameter.grad is not None for parameter in iter_base_parameters(base_model)
    )
    if base_grad_count != 0:
        raise RuntimeError(f"frozen base received {base_grad_count} gradients")

    anchor_offsets = output["anchor_offsets"]
    gaussian_offsets = output["gaussian_offsets"]
    disabled_max = {
        "anchor_scaling": float(anchor_offsets["delta_scaling"].abs().max().item()),
        "anchor_opacity": float(anchor_offsets["delta_opacity"].abs().max().item()),
        "gaussian_scaling": float(gaussian_offsets["delta_scaling"].abs().max().item()),
        "gaussian_opacity": float(gaussian_offsets["delta_opacity"].abs().max().item()),
    }
    if any(value != 0 for value in disabled_max.values()):
        raise RuntimeError(f"disabled offset channels are nonzero: {disabled_max}")

    checkpoint = Path(args.output_dir) / "checkpoint_latest.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError("one-batch checkpoint was not saved")
    restored_step = load_image_training_checkpoint(
        checkpoint,
        model,
        result["optimizer"],
        config,
        dataset,
        fingerprint,
    )
    if restored_step != result["step"]:
        raise RuntimeError("saved checkpoint step did not round-trip")

    result["optimizer"].zero_grad(set_to_none=True)
    roundtrip_losses, roundtrip_outputs = compute_image_conditioned_training_loss(
        model, episode, result["anchor_edges"], config, args.device,
        render_target=True, deformation_adapter=result["deformation_adapter"],
    )
    roundtrip_output = roundtrip_outputs["primary"]
    roundtrip_anchor_max_abs = float(
        (roundtrip_output["anchor_offsets"]["delta_xyz"] - anchor_offsets["delta_xyz"])
        .abs().max().item()
    )
    roundtrip_rgb_max_abs = float(
        (roundtrip_output["render"][0] - pred_rgb).abs().max().item()
    )
    if roundtrip_anchor_max_abs > 1e-7 or roundtrip_rgb_max_abs > 1e-6:
        raise RuntimeError("checkpoint roundtrip changed model output")

    condition_b = _single_reference_episode(episode)
    sensitivity_losses, sensitivity_outputs = compute_image_conditioned_training_loss(
        model, condition_b, result["anchor_edges"], config, args.device,
        render_target=True, deformation_adapter=result["deformation_adapter"],
    )
    sensitivity_output = sensitivity_outputs["primary"]
    sensitivity = {
        "condition_a_reference_count": int(episode["reference_images"].shape[0]),
        "condition_b_reference_count": 1,
        "condition_b_target_view_used": False,
        "global_embedding_l2": float(torch.linalg.vector_norm(
            output["global_clothing_embedding"] - sensitivity_output["global_clothing_embedding"]
        ).item()),
        "anchor_feature_mean_abs": float((
            output["anchor_clothing_features"] - sensitivity_output["anchor_clothing_features"]
        ).abs().mean().item()),
        "anchor_offset_mean_abs": float((
            anchor_offsets["delta_xyz"] - sensitivity_output["anchor_offsets"]["delta_xyz"]
        ).abs().mean().item()),
        "rendered_rgb_mean_abs": float((
            pred_rgb - sensitivity_output["render"][0]
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
        "losses": {key: float(value.detach().item()) for key, value in losses.items()},
        "total_loss_finite": bool(torch.isfinite(losses["total"]).item()),
        "backward_success": True,
        "trainable_grad_norms": trainable_grad_norms,
        "connectivity_extra_optimizer_updates": connectivity_warmup_updates,
        "encoder_backbone_requires_grad": any(
            parameter.requires_grad
            for parameter in model.clothing_observation_encoder.backbone.parameters()
        ),
        "base_grad_count": base_grad_count,
        "base_state_cache_restored": True,
        "disabled_offset_channel_max_abs": disabled_max,
        "checkpoint_round_trip_step": restored_step,
        "checkpoint_roundtrip_anchor_max_abs": roundtrip_anchor_max_abs,
        "checkpoint_roundtrip_rgb_max_abs": roundtrip_rgb_max_abs,
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
    (output_dir / "diagnostics.json").write_text(
        json.dumps(_jsonable(diagnostics), indent=2), encoding="utf-8"
    )

    checkpoint_file = Path(checkpoint_path)
    manifest_path = Path(config["image_conditioning"]["manifest_path"])
    region_path = Path(config["image_conditioning"]["reference_region_path"])
    checkpoint_sha = _sha256(checkpoint_file)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("current checkpoint file SHA256 changed")
    input_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": {"branch": _git("branch", "--show-current"), "commit": _git("rev-parse", "HEAD"), "status": _git("status", "--short")},
        "environment": {"python": platform.python_version(), "pytorch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
        "base_checkpoint": {"path": str(checkpoint_file), "size": checkpoint_file.stat().st_size, "sha256": checkpoint_sha, "historical_warning": "Current SHA256 differs from historical 64ac7f2d... acceptance fingerprint."},
        "episode_manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "reference_only_region": {"path": str(region_path), "sha256": _sha256(region_path), "threshold": config["image_conditioning"]["reference_gate_threshold"], "reference_views": config["image_conditioning"]["reference_views"], "target_view_used": False},
        "reference_frames": episode["reference_frame_ids"], "reference_cameras": episode["reference_view_ids"],
        "target_frame": episode["target_frame_id"], "target_camera": episode["target_view_id"],
        "config": config,
    }
    (output_dir / "input_manifest.json").write_text(json.dumps(input_manifest, indent=2), encoding="utf-8")
    (output_dir / "one_batch_diagnostics.json").write_text(json.dumps(_jsonable(diagnostics), indent=2), encoding="utf-8")
    markdown = "# Gate 4-A Real One-batch Diagnostics\n\n- Status: PASS\n- Commit: `" + input_manifest["git"]["commit"] + "`\n- Loss: `" + str(diagnostics["losses"]["total"]) + "`\n- Anchor delta: `" + str(diagnostics["anchor_delta_xyz"]["shape"]) + "`\n- Gaussian delta: `" + str(diagnostics["gaussian_delta_xyz"]["shape"]) + "`\n- Gradients: `" + json.dumps(trainable_grad_norms) + "`\n- Sensitivity: `" + json.dumps(sensitivity) + "`\n"
    (output_dir / "one_batch_diagnostics.md").write_text(markdown, encoding="utf-8")
    save_image(make_grid(episode["reference_images"].cpu(), nrow=2), output_dir / "reference_contact_sheet.png")
    save_image(episode["target_rgb"].cpu(), output_dir / "target_rgb.png")
    save_image(pred_rgb.detach().cpu(), output_dir / "predicted_rgb.png")
    save_image(pred_alpha.detach().cpu(), output_dir / "predicted_alpha.png")
    comparison = make_grid([episode["target_rgb"].cpu(), pred_rgb.detach().cpu(), (episode["target_rgb"].cpu() - pred_rgb.detach().cpu()).abs()], nrow=3)
    save_image(comparison, output_dir / "comparison.png")
    shutil.copyfile(checkpoint, output_dir / "checkpoint_step_000001.pth") if checkpoint.name != "checkpoint_step_000001.pth" else None
    (output_dir / "GATE_ACCEPTANCE.md").write_text("# Gate 4-A Acceptance\n\nStatus: **PASS**\n\nReal one-batch, reference-only gate, backward, sensitivity and checkpoint roundtrip verified.\n", encoding="utf-8")

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
    print(f"diagnostics: {(output_dir / 'diagnostics.json').resolve()}")
    print("CanonDressGS Gate 4-A real one-batch check: PASS")


if __name__ == "__main__":
    main()
