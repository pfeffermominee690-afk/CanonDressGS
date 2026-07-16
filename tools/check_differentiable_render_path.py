from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_dressable as training
from scene.full_dressable_dataset import FullDressableTrainingDataset
from scene.gaussian_clothing_residuals import CHANNELS
from tools.check_real_image_conditioned_one_batch import (
    _as_chw_render,
    save_render_tensor,
)
from tools.infer_module3_online_completion import _InferenceModelContract, _num_clothes
from utils.dressable_camera_utils import build_mmlphuman_camera
from utils.mmlphuman_anchor_deformation import MMLPHumanAnchorDeformationAdapter
from utils.mmlphuman_state_utils import mmlphuman_state_transaction
from utils.rendering_loss_utils import alpha_mask_loss, rgb_reconstruction_loss


BASE_PARAMETER_NAMES = ("_xyz", "_scaling", "_rotation", "_opacity", "_sh0", "_shN")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_trace(value: torch.Tensor) -> dict[str, Any]:
    numeric = value.detach().float()
    return {
        "shape": list(value.shape),
        "requires_grad": bool(value.requires_grad),
        "grad_fn": type(value.grad_fn).__name__ if value.grad_fn is not None else None,
        "is_leaf": bool(value.is_leaf),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "finite": bool(torch.isfinite(numeric).all()),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "mean": float(numeric.mean()),
    }


def _parameter_grad_metrics(parameters: Iterable[torch.nn.Parameter]) -> dict[str, Any]:
    values = [
        parameter.grad.detach().float()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not values:
        return {
            "parameter_count_with_grad": 0,
            "norm": 0.0,
            "max_abs": 0.0,
            "finite": True,
        }
    return {
        "parameter_count_with_grad": len(values),
        "norm": float(torch.sqrt(sum(value.square().sum() for value in values))),
        "max_abs": max(float(value.abs().max()) for value in values),
        "finite": all(bool(torch.isfinite(value).all()) for value in values),
    }


def _module_grad_metrics(module: torch.nn.Module) -> dict[str, Any]:
    return _parameter_grad_metrics(module.parameters())


def _head_modules(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    anchor_mlp = model.dressable_model.anchor_clothing_mlp
    return {
        "xyz": anchor_mlp.output_layer,
        "scaling": anchor_mlp.scaling_head,
        "rotation": anchor_mlp.rotation_head,
        "opacity": anchor_mlp.opacity_head,
        "sh0": anchor_mlp.sh0_head,
        "shN": anchor_mlp.shN_head,
    }


def _gradient_snapshot(model: torch.nn.Module, base: Any) -> dict[str, Any]:
    heads = _head_modules(model)
    return {
        "six_channel_heads": {
            name: _module_grad_metrics(module) for name, module in heads.items()
        },
        "shared_trunk": _module_grad_metrics(
            model.dressable_model.anchor_clothing_mlp.hidden_layers
        ),
        "hypernetwork": _module_grad_metrics(
            model.dressable_model.clothing_film_generator
        ),
        "online_completer": _module_grad_metrics(model.canonical_clothing_completer),
        "aggregator": _module_grad_metrics(model.multiview_aggregator),
        "encoder_projection_head": _module_grad_metrics(
            model.clothing_observation_encoder.projection_head
        ),
        "frozen_image_backbone": _module_grad_metrics(
            model.clothing_observation_encoder.backbone
        ),
        "frozen_base_grad_count": sum(
            getattr(base, name).grad is not None for name in BASE_PARAMETER_NAMES
        ),
    }


def test_real_render_requires_grad(
    rendered_rgb: torch.Tensor,
    rendered_alpha: torch.Tensor,
) -> bool:
    return bool(
        rendered_rgb.requires_grad
        and rendered_alpha.requires_grad
        and rendered_rgb.grad_fn is not None
        and rendered_alpha.grad_fn is not None
    )


def test_rgb_loss_reaches_six_channel_decoder(
    gradient_metrics: dict[str, Any],
) -> bool:
    heads = gradient_metrics["six_channel_heads"]
    geometry = ("xyz", "scaling", "rotation", "opacity")
    appearance = ("sh0", "shN")
    return bool(
        any(heads[name]["norm"] > 0 for name in geometry)
        and any(heads[name]["norm"] > 0 for name in appearance)
    )


def test_alpha_loss_reaches_geometry_channels(
    gradient_metrics: dict[str, Any],
) -> bool:
    heads = gradient_metrics["six_channel_heads"]
    return bool(
        all(
            heads[name]["norm"] > 0
            for name in ("xyz", "scaling", "rotation", "opacity")
        )
    )


def test_frozen_base_has_no_grad(
    gradient_metrics: dict[str, Any],
    base_max_change: dict[str, float],
) -> bool:
    return bool(
        gradient_metrics["frozen_base_grad_count"] == 0
        and gradient_metrics["frozen_image_backbone"]["parameter_count_with_grad"] == 0
        and not any(base_max_change.values())
    )


def _cpu_camera(camera: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
        for key, value in camera.items()
    }


def _prepare_episode(
    manifest: Path,
    *,
    sample_index: int,
) -> tuple[dict[str, Any], dict[str, Any], FullDressableTrainingDataset]:
    dataset = FullDressableTrainingDataset(
        manifest,
        split="train",
        reference_count=2,
        seed=42,
    )
    sample = dataset[sample_index]
    if sample["target_condition_id"] in sample["reference_condition_ids"]:
        raise RuntimeError("target/reference overlap")
    episode = {
        "reference_images": sample["reference_images"],
        "reference_cloth_masks": sample["reference_clothing_masks"],
        "reference_foreground_masks": sample["reference_foreground_masks"],
        "reference_poses": sample["reference_poses"],
        "reference_Rh": sample["reference_R_global"],
        "reference_Th": sample["reference_Th"],
        "reference_cameras": [
            _cpu_camera(value) for value in sample["reference_cameras"]
        ],
        "reference_valid_mask": sample["reference_valid_mask"],
        "target_rgb": sample["target_rgb"],
        "target_foreground_mask": sample["target_foreground_mask"],
        "target_clothing_mask": sample["target_clothing_mask"],
        "target_pose": sample["target_pose"],
        "target_Rh": sample["target_R_global"],
        "target_Th": sample["target_Th"],
        "target_camera": _cpu_camera(sample["target_camera"]),
    }
    return sample, episode, dataset


def _build_model(
    gate4_config: Path,
    module2_config: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[Any, torch.nn.Module, dict[str, Any], dict[str, Any]]:
    gate4 = training.load_config(gate4_config)
    module2 = training.load_config(module2_config)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = checkpoint["model"]
    base = training.load_frozen_mmlphuman_base(
        gate4["base"]["model_dir"],
        gate4["base"]["checkpoint_path"],
        device,
    )
    model, _, _, _ = training.create_image_conditioned_components(
        _InferenceModelContract(_num_clothes(state)),
        gate4,
        base,
        device,
    )
    model.dressable_model.configure_six_channel_decoder(
        module2["dressable_channels"]
    )
    metadata = checkpoint["metadata"]
    model.initialize_online_completion(
        checkpoint["graph_indices"].to(device),
        checkpoint["graph_weights"].to(device=device, dtype=base._xyz.dtype),
        int(metadata["feature_completion_hidden_dim"]),
        int(metadata["feature_completion_blocks"]),
    )
    model.load_state_dict(state, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    for parameter in model.dressable_model.clothing_embedding.parameters():
        parameter.requires_grad_(False)
    for parameter in model.clothing_observation_encoder.backbone.parameters():
        parameter.requires_grad_(False)
    model.train()
    return base, model, checkpoint, gate4


def _run_real_closure(
    args: argparse.Namespace,
    unit_results: dict[str, str],
) -> dict[str, Any]:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_grad_enabled(True)

    sample, episode, dataset = _prepare_episode(
        args.manifest,
        sample_index=args.sample_index,
    )
    base, model, checkpoint, gate4 = _build_model(
        args.gate4_config,
        args.module2_config,
        args.module3_checkpoint,
        device,
    )
    base_before = {
        name: getattr(base, name).detach().clone() for name in BASE_PARAMETER_NAMES
    }
    adapter = MMLPHumanAnchorDeformationAdapter.from_mmlphuman_base(
        base,
        canonical_anchors=model.canonical_anchors,
        lbs_grid_path=gate4["base"]["lbs_grid_path"],
    )
    background = torch.tensor(
        gate4["render"]["background"],
        device=device,
        dtype=base._xyz.dtype,
    )
    geometry = training.prepare_real_reference_geometry(
        base,
        adapter,
        episode,
        background,
    )
    online_inputs = {
        "reference_images": episode["reference_images"].to(device),
        "reference_cloth_masks": episode["reference_cloth_masks"].to(device),
        "reference_foreground_masks": episode["reference_foreground_masks"].to(device),
        "reference_poses": episode["reference_poses"].to(device),
        "reference_cameras": episode["reference_cameras"],
        "reference_valid_mask": episode["reference_valid_mask"].to(device),
        "deformation_fn": training.build_episode_deformation_fn(adapter, episode),
        "surface_depth_maps": geometry["surface_depth_maps"],
        "surface_alpha_maps": geometry["surface_alpha_maps"],
        "require_depth_visibility": True,
    }
    result = model.compute_online_six_channel_residuals(**online_inputs)
    overrides = model.dressable_model.compose_canonical_gaussian_overrides(
        result["gaussian_residuals"],
        CHANNELS,
    )
    camera = build_mmlphuman_camera(
        episode["target_camera"],
        int(episode["target_rgb"].shape[-2]),
        int(episode["target_rgb"].shape[-1]),
        device,
    )
    original_degree = int(base.sh_degree)
    intermediate: dict[str, torch.Tensor] = {}
    try:
        base.sh_degree = int(
            checkpoint["metadata"]["module2_decoder_metadata"]["effective_sh_degree"]
        )
        with mmlphuman_state_transaction(
            base,
            episode["target_pose"].to(device),
            episode["target_Rh"].to(device),
            episode["target_Th"].to(device),
        ):
            intermediate = {
                "posed_means": base.compute_xyz(overrides.as_dict()),
                "posed_covariances": base.get_covariance(
                    canonical_overrides=overrides.as_dict()
                ),
                "posed_opacities": base.compute_opacity(overrides.as_dict()),
            }
            camera_position = torch.linalg.inv_ex(camera["w2c"])[0][:3, 3]
            intermediate["posed_colors"] = base.get_color(
                camera_position,
                overrides.as_dict(),
            )
            rendered_rgb, rendered_alpha, _ = base.render(
                camera,
                background=background,
                canonical_overrides=overrides.as_dict(),
            )
    finally:
        base.sh_degree = original_degree

    rgb = _as_chw_render(rendered_rgb, 3)
    alpha = _as_chw_render(rendered_alpha, 1)
    target_rgb = episode["target_rgb"].to(device=rgb.device, dtype=rgb.dtype)
    target_foreground = episode["target_foreground_mask"].to(
        device=alpha.device,
        dtype=alpha.dtype,
    )
    target_clothing = episode["target_clothing_mask"].to(
        device=rgb.device,
        dtype=rgb.dtype,
    )
    rgb_parts = rgb_reconstruction_loss(
        rgb,
        target_rgb,
        ssim_weight=0.2,
    )
    clothing_parts = rgb_reconstruction_loss(
        rgb,
        target_rgb,
        mask=target_clothing,
        ssim_weight=0.0,
    )
    alpha_parts = alpha_mask_loss(alpha, target_foreground)
    rgb_only = rgb_parts["total"] + clothing_parts["l1"]
    alpha_only = alpha_parts["total"]
    combined = rgb_only + 0.5 * alpha_only

    gradients: dict[str, Any] = {}
    for name, loss, retain_graph in (
        ("rgb_only", rgb_only, True),
        ("alpha_only", alpha_only, True),
        ("combined_image_only", combined, False),
    ):
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=retain_graph)
        gradients[name] = _gradient_snapshot(model, base)

    trace = {
        "grad_enabled": torch.is_grad_enabled(),
        "inference_mode_enabled": torch.is_inference_mode_enabled(),
        "reference_images": _tensor_trace(online_inputs["reference_images"]),
        "global_clothing_embedding": _tensor_trace(
            result["global_clothing_embedding"]
        ),
        "completed_anchor_features": _tensor_trace(
            result["completion"].completed_anchor_features
        ),
        "geometry_gate": _tensor_trace(result["completion"].geometry_gate),
        "appearance_gate": _tensor_trace(result["completion"].appearance_gate),
        "raw_anchor_residuals": {
            name: _tensor_trace(getattr(result["raw_anchor_residuals"], name))
            for name in CHANNELS
        },
        "gated_anchor_residuals": {
            name: _tensor_trace(getattr(result["gated_anchor_residuals"], name))
            for name in CHANNELS
        },
        "gaussian_residuals": {
            name: _tensor_trace(getattr(result["gaussian_residuals"], name))
            for name in CHANNELS
        },
        "canonical_overrides": {
            name: _tensor_trace(value) for name, value in overrides.as_dict().items()
        },
        "posed_gaussian_attributes": {
            name: _tensor_trace(value) for name, value in intermediate.items()
        },
        "raw_rendered_rgb": _tensor_trace(rendered_rgb),
        "raw_rendered_alpha": _tensor_trace(rendered_alpha),
        "training_rgb_chw": _tensor_trace(rgb),
        "training_alpha_chw": _tensor_trace(alpha),
        "cpu_or_numpy_or_pil_before_loss": False,
        "detached_render_cache_used": False,
    }
    base_max_change = {
        name: float((getattr(base, name).detach() - before).abs().max())
        for name, before in base_before.items()
    }
    combined_metrics = gradients["combined_image_only"]
    tests = {
        "test_real_render_requires_grad": test_real_render_requires_grad(
            rgb,
            alpha,
        ),
        "test_rgb_loss_reaches_six_channel_decoder":
            test_rgb_loss_reaches_six_channel_decoder(
                gradients["rgb_only"],
            ),
        "test_alpha_loss_reaches_geometry_channels":
            test_alpha_loss_reaches_geometry_channels(
                gradients["alpha_only"],
            ),
        "test_frozen_base_has_no_grad": test_frozen_base_has_no_grad(
            combined_metrics,
            base_max_change,
        ),
        "combined_image_loss_reaches_shared_trunk": (
            combined_metrics["shared_trunk"]["norm"] > 0
        ),
        "combined_image_loss_reaches_hypernetwork": (
            combined_metrics["hypernetwork"]["norm"] > 0
        ),
        "combined_image_loss_reaches_online_completer": (
            combined_metrics["online_completer"]["norm"] > 0
        ),
        "combined_image_loss_reaches_aggregator": (
            combined_metrics["aggregator"]["norm"] > 0
        ),
        "combined_image_loss_reaches_encoder_trainable_head": (
            combined_metrics["encoder_projection_head"]["norm"] > 0
        ),
    }
    diagnostics = {
        "status": (
            "PASS"
            if all(tests.values()) and all(value == "PASS" for value in unit_results.values())
            else "FAIL"
        ),
        "scope": "one real forward and image-only backward; no optimizer step and no training",
        "root_cause": (
            "tools.check_real_image_conditioned_one_batch._as_chw_render detached "
            "and moved renderer tensors to CPU before the V2 smoke image loss"
        ),
        "fix": (
            "_as_chw_render now performs only finite shape conversion; "
            "save_render_tensor owns detach/clamp/CPU conversion"
        ),
        "git": {
            "branch": subprocess.check_output(
                ["git", "branch", "--show-current"],
                cwd=PROJECT_ROOT,
                text=True,
            ).strip(),
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
                text=True,
            ).strip(),
            "status_short": subprocess.check_output(
                ["git", "status", "--short"],
                cwd=PROJECT_ROOT,
                text=True,
            ).splitlines(),
        },
        "inputs": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": _sha256(args.manifest),
            "checkpoint": str(args.module3_checkpoint.resolve()),
            "checkpoint_sha256": _sha256(args.module3_checkpoint),
            "outfit_id": sample["outfit_id"],
            "target": sample["target_condition_id"],
            "references": sample["reference_condition_ids"],
            "target_reference_overlap": False,
            "teacher_read": False,
            "target_rgb_or_mask_used_for_gate": False,
            "direct_residual_teacher_losses_enabled": False,
        },
        "losses": {
            "rgb_l1": float(rgb_parts["l1"].detach()),
            "rgb_ssim": float(rgb_parts["ssim"].detach()),
            "clothing_region_rgb_l1": float(clothing_parts["l1"].detach()),
            "alpha_bce": float(alpha_parts["bce"].detach()),
            "alpha_dice": float(alpha_parts["dice"].detach()),
            "rgb_only": float(rgb_only.detach()),
            "alpha_only": float(alpha_only.detach()),
            "combined_image_only": float(combined.detach()),
        },
        "gradients": gradients,
        "base_max_parameter_change": base_max_change,
        "tests": tests,
        "unit_tests": unit_results,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    (output / "differentiable_render_trace.json").write_text(
        json.dumps(trace, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "image_loss_gradient_metrics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "image_loss_backward_trace.json").write_text(
        json.dumps(
            {
                "losses": diagnostics["losses"],
                "gradients": gradients,
                "tests": tests,
                "unit_tests": unit_results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    save_render_tensor(output / "predicted_rgb.png", rendered_rgb, 3)
    save_render_tensor(output / "predicted_alpha.png", rendered_alpha, 1)
    audit = (
        "# Differentiable Render Audit\n\n"
        f"Status: **{diagnostics['status']}**\n\n"
        "The formal model and gsplat path remained differentiable. The V2 smoke "
        "lost the graph in its shared CHW conversion helper, which detached and "
        "moved tensors to CPU before constructing image loss. The helper now "
        "preserves the original tensor graph; only PNG persistence detaches.\n\n"
        f"- raw RGB requires_grad: `{rendered_rgb.requires_grad}`; grad_fn: "
        f"`{type(rendered_rgb.grad_fn).__name__ if rendered_rgb.grad_fn else None}`\n"
        f"- raw alpha requires_grad: `{rendered_alpha.requires_grad}`; grad_fn: "
        f"`{type(rendered_alpha.grad_fn).__name__ if rendered_alpha.grad_fn else None}`\n"
        "- direct residual/teacher loss used: `false`\n"
        f"- target/reference overlap: `false`\n"
        f"- checkpoint SHA256: `{_sha256(args.module3_checkpoint)}`\n\n"
        "See `differentiable_render_trace.json` and "
        "`image_loss_gradient_metrics.json` for per-stage and per-module evidence.\n"
    )
    (output / "DIFFERENTIABLE_RENDER_AUDIT.md").write_text(
        audit,
        encoding="utf-8",
    )
    print(json.dumps(diagnostics, indent=2))
    if diagnostics["status"] != "PASS":
        raise SystemExit(1)
    return diagnostics


def test_png_saving_does_not_detach_training_tensor() -> None:
    source = torch.rand(8, 9, 3, requires_grad=True)
    training_tensor = _as_chw_render(source, 3)
    with tempfile.TemporaryDirectory() as directory:
        save_render_tensor(Path(directory) / "rgb.png", training_tensor, 3)
        if not (Path(directory) / "rgb.png").is_file():
            raise AssertionError("PNG was not saved")
    training_tensor.sum().backward()
    if source.grad is None or not torch.isfinite(source.grad).all():
        raise AssertionError("PNG saving altered the live training graph")


def test_eval_render_may_detach_but_train_render_must_not() -> None:
    source = torch.rand(7, 11, 1, requires_grad=True)
    train_value = _as_chw_render(source, 1)
    evaluation_value = train_value.detach().cpu()
    if not train_value.requires_grad or train_value.grad_fn is None:
        raise AssertionError("training conversion detached the renderer tensor")
    if evaluation_value.requires_grad or evaluation_value.device.type != "cpu":
        raise AssertionError("explicit evaluation detach did not produce a CPU tensor")


def _run_unit_tests() -> dict[str, str]:
    tests = {
        "test_png_saving_does_not_detach_training_tensor":
            test_png_saving_does_not_detach_training_tensor,
        "test_eval_render_may_detach_but_train_render_must_not":
            test_eval_render_may_detach_but_train_render_must_not,
    }
    results: dict[str, str] = {}
    for name, test in tests.items():
        test()
        results[name] = "PASS"
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify real image-loss gradients through the dressable gsplat path"
    )
    parser.add_argument("--unit-only", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--gate4-config", type=Path)
    parser.add_argument("--module2-config", type=Path)
    parser.add_argument("--module3-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    unit_results = _run_unit_tests()
    if args.unit_only:
        print(json.dumps(unit_results, indent=2))
        return
    required = (
        "manifest",
        "gate4_config",
        "module2_config",
        "module3_checkpoint",
        "output_dir",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"missing real closure arguments: {', '.join(missing)}")
    _run_real_closure(args, unit_results)


if __name__ == "__main__":
    main()
