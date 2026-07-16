from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torchvision.utils import save_image

import train_dressable as training
from scene.gaussian_clothing_residuals import AnchorClothingResiduals, CHANNELS
from tools.build_full_pipeline_visual_acceptance import build as build_visual_acceptance
from tools.infer_module3_online_completion import _InferenceModelContract, _num_clothes
from utils.dressable_camera_utils import build_mmlphuman_camera
from utils.mmlphuman_anchor_deformation import MMLPHumanAnchorDeformationAdapter
from utils.mmlphuman_state_utils import mmlphuman_state_transaction


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chw(value: torch.Tensor, channels: int) -> torch.Tensor:
    if value.ndim == 3 and value.shape[0] == channels: return value
    if value.ndim == 3 and value.shape[-1] == channels: return value.permute(2, 0, 1)
    raise ValueError(f"render tensor does not have {channels} channels: {tuple(value.shape)}")


def mae(value: torch.Tensor) -> float:
    return float(value.detach().float().abs().mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate4-config", required=True)
    parser.add_argument("--module2-config", required=True)
    parser.add_argument("--module2-output", required=True)
    parser.add_argument("--module3-output", required=True)
    parser.add_argument("--supplementary-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    source = Path(args.module3_output)
    output = Path(args.supplementary_dir)
    if output.exists(): raise FileExistsError(f"supplementary output already exists: {output}")
    output.mkdir(parents=True)
    device = torch.device(args.device)
    checkpoint_path = source / "checkpoint_final.pth"
    request_path = source / "independent_inference_request.pt"
    independent_output_path = source / "independent_inference_output.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    request = torch.load(request_path, map_location="cpu", weights_only=True)
    independent = torch.load(independent_output_path, map_location="cpu", weights_only=True)
    forbidden = {"teacher", "teacher_gate", "teacher_active_mask", "target_rgb", "target_foreground_mask",
                 "target_clothing_mask", "cloth_id", "outfit_id", "reference_only_gate"}.intersection(request)
    if forbidden: raise ValueError(f"forbidden inference request fields: {sorted(forbidden)}")

    for index in range(2):
        save_image(request["reference_images"][index].clamp(0, 1), output / f"reference_{index+1}.png")
        save_image(request["reference_cloth_masks"][index].clamp(0, 1), output / f"reference_clothing_mask_{index+1}.png")
    visual_names = [
        "observed_probability_S1.png", "observed_probability_S2.png", "observed_probability_S12.png",
        "geometry_gate_S1.png", "geometry_gate_S2.png", "geometry_gate_S12.png",
        "appearance_gate_S1.png", "appearance_gate_S2.png", "appearance_gate_S12.png",
        "teacher_gate.png", "observed_only_gate.png", "diffusion_gate.png", "learned_completed_gate.png",
        "teacher_gate_render.png", "observed_only_render.png", "diffusion_render.png", "learned_gate_render.png",
        "render_comparison.png", "feature_holdout_comparison.png", "loss_curve.png",
    ]
    for name in visual_names: shutil.copy2(source / name, output / name)

    c4 = training.load_config(args.gate4_config)
    c2 = training.load_config(args.module2_config)
    state = checkpoint["model"]
    base = training.load_frozen_mmlphuman_base(c4["base"]["model_dir"], c4["base"]["checkpoint_path"], device)
    model, _, _, _ = training.create_image_conditioned_components(
        _InferenceModelContract(_num_clothes(state)), c4, base, device
    )
    model.dressable_model.configure_six_channel_decoder(c2["dressable_channels"])
    metadata = checkpoint["metadata"]
    graph_indices = checkpoint["graph_indices"].to(device)
    graph_weights = checkpoint["graph_weights"].to(device=device, dtype=base._xyz.dtype)
    model.initialize_online_completion(
        graph_indices, graph_weights, int(metadata["feature_completion_hidden_dim"]),
        int(metadata["feature_completion_blocks"]),
    )
    model.load_state_dict(state, strict=True); model.eval()
    adapter = MMLPHumanAnchorDeformationAdapter.from_mmlphuman_base(
        base, canonical_anchors=model.canonical_anchors, lbs_grid_path=c4["base"]["lbs_grid_path"]
    )
    episode = {key: value for key, value in request.items() if key.startswith("reference_")}
    background = torch.tensor(c4["render"]["background"], device=device, dtype=base._xyz.dtype)
    geometry = training.prepare_real_reference_geometry(base, adapter, episode, background)
    online_inputs = {
        "reference_images": request["reference_images"].to(device),
        "reference_cloth_masks": request["reference_cloth_masks"].to(device),
        "reference_foreground_masks": request["reference_foreground_masks"].to(device),
        "reference_poses": request["reference_poses"].to(device),
        "reference_cameras": request["reference_cameras"],
        "reference_valid_mask": request["reference_valid_mask"].to(device),
        "deformation_fn": training.build_episode_deformation_fn(adapter, episode),
        "surface_depth_maps": geometry["surface_depth_maps"],
        "surface_alpha_maps": geometry["surface_alpha_maps"],
        "require_depth_visibility": True,
    }
    with torch.no_grad(): result = model.compute_online_six_channel_residuals(**online_inputs)
    parity = {}
    parity["completed_features"] = float((result["completion"].completed_anchor_features.cpu() - independent["completed_features"]).abs().max())
    parity["geometry_gate"] = float((result["completion"].geometry_gate.cpu() - independent["geometry_gate"]).abs().max())
    parity["appearance_gate"] = float((result["completion"].appearance_gate.cpu() - independent["appearance_gate"]).abs().max())
    for name in CHANNELS:
        parity[f"anchor.{name}"] = float((getattr(result["gated_anchor_residuals"], name).cpu() - independent[f"anchor.{name}"]).abs().max())

    teacher_dict = torch.load(Path(args.module2_output) / "teacher_anchor_residuals.pt", map_location=device, weights_only=True)
    teacher = AnchorClothingResiduals(**teacher_dict).validate(10000, base)
    geometry_active = ((teacher.delta_xyz.abs().sum(1) + teacher.delta_log_scaling.abs().sum(1)
                        + teacher.delta_rotvec.abs().sum(1)) > 1e-8)
    appearance_active = ((teacher.delta_sh0.abs().sum(1) + teacher.delta_shN.abs().sum(1)) > 1e-8)
    opacity_active = geometry_active | appearance_active
    gates = result["completion"]
    residuals = result["gated_anchor_residuals"]
    inactive_masks = {
        "delta_xyz": ~geometry_active, "delta_log_scaling": ~geometry_active,
        "delta_rotvec": ~geometry_active, "delta_opacity_logit": ~opacity_active,
        "delta_sh0": ~appearance_active, "delta_shN": ~appearance_active,
    }
    residual_leakage = {name: mae(getattr(residuals, name)[inactive_masks[name]]) for name in CHANNELS}

    def render(anchor_residuals: AnchorClothingResiduals | None):
        overrides = None
        if anchor_residuals is not None:
            gaussian = model.dressable_model.interpolate_anchor_clothing_residuals(anchor_residuals)
            overrides = model.dressable_model.compose_canonical_gaussian_overrides(gaussian, CHANNELS)
        camera = build_mmlphuman_camera(request["target_camera"], int(request["target_height"]), int(request["target_width"]), device)
        degree = int(metadata["module2_decoder_metadata"]["effective_sh_degree"])
        original = int(base.sh_degree)
        try:
            base.sh_degree = degree
            with mmlphuman_state_transaction(base, request["target_pose"].to(device), request["target_Rh"].to(device), request["target_Th"].to(device)):
                return base.render(camera, background=background, canonical_overrides=None if overrides is None else overrides.as_dict())
        finally: base.sh_degree = original

    with torch.no_grad():
        base_render = render(None)
        learned_render = render(residuals)
        teacher_render = render(teacher)
        zero = AnchorClothingResiduals.zeros(base, CHANNELS)
        rotation_only = AnchorClothingResiduals(**{
            name: (residuals.delta_rotvec if name == "delta_rotvec" else getattr(zero, name)) for name in CHANNELS
        })
        rotation_render = render(rotation_only)
    base_rgb, base_alpha = chw(base_render[0], 3), chw(base_render[1], 1)
    teacher_rgb, teacher_alpha = chw(teacher_render[0], 3), chw(teacher_render[1], 1)
    learned_rgb, learned_alpha = chw(learned_render[0], 3), chw(learned_render[1], 1)
    rotation_rgb, rotation_alpha = chw(rotation_render[0], 3), chw(rotation_render[1], 1)
    clothing_pixels = ((teacher_rgb - base_rgb).abs().amax(0, keepdim=True) > 1 / 255) | ((teacher_alpha - base_alpha).abs() > 1e-4)
    nonclothing = ~clothing_pixels
    render_leakage = {
        "non_clothing_render_rgb_mae": mae((learned_rgb - base_rgb)[nonclothing.expand_as(learned_rgb)]),
        "non_clothing_render_alpha_mae": mae((learned_alpha - base_alpha)[nonclothing]),
        "rotation_non_clothing_render_rgb_mae": mae((rotation_rgb - base_rgb)[nonclothing.expand_as(rotation_rgb)]),
        "rotation_non_clothing_render_alpha_mae": mae((rotation_alpha - base_alpha)[nonclothing]),
        "non_clothing_pixel_ratio": float(nonclothing.float().mean()),
    }
    def overlay(delta: torch.Tensor) -> torch.Tensor:
        heat = delta.abs().amax(0, keepdim=True).clamp(0, 1) * nonclothing
        return torch.cat((heat * 32, heat * 16, torch.zeros_like(heat)), 0).clamp(0, 1)
    save_image(overlay(learned_rgb - base_rgb), output / "non_clothing_leakage_overlay.png")
    save_image(overlay(rotation_rgb - base_rgb), output / "rotation_non_clothing_leakage_overlay.png")

    leakage = {
        "geometry_gate_inactive_mean": float(gates.geometry_gate[~geometry_active].mean()),
        "geometry_gate_inactive_max": float(gates.geometry_gate[~geometry_active].max()),
        "appearance_gate_inactive_mean": float(gates.appearance_gate[~appearance_active].mean()),
        "appearance_gate_inactive_max": float(gates.appearance_gate[~appearance_active].max()),
        "non_clothing_residual_mae": residual_leakage,
        "rotation_non_clothing_residual_mae": residual_leakage["delta_rotvec"],
        **render_leakage,
    }
    dataflow = {
        "trace": [
            "reference_clothing_masks", "anchor_projection_and_visibility",
            "observed_clothing_probability", "canonical_clothing_completer",
            "geometry_and_appearance_gates", "six_channel_decoder",
            "Gaussian_residual_interpolation", "canonical_composition", "render",
        ],
        "gate_source": metadata["gate_source"],
        "request_fields": sorted(request),
        "forbidden_fields_count": len(forbidden),
        "teacher_forward_input": False,
        "target_rgb_or_mask_forward_input": False,
        "temporary_gate_forward_input": False,
        "cloth_or_outfit_id_forward_input": False,
        "output_shapes": {
            "observed_probability": list(gates.observed_clothing_probability.shape),
            "completed_features": list(gates.completed_anchor_features.shape),
            "geometry_gate": list(gates.geometry_gate.shape),
            "appearance_gate": list(gates.appearance_gate.shape),
            **{f"anchor.{name}": list(getattr(residuals, name).shape) for name in CHANNELS},
        },
        "evaluation_parity_max_abs": parity,
    }
    checkpoint_keys = sorted(checkpoint)
    resume = {
        "checkpoint_keys": checkpoint_keys,
        "model_state_present": "model" in checkpoint,
        "graph_state_present": "graph_indices" in checkpoint and "graph_weights" in checkpoint,
        "graph_fingerprint": metadata["graph_fingerprint"],
        "model_reload_parity_max_abs": parity,
        "optimizer_state_present": "optimizer" in checkpoint,
        "scheduler_state_present": "scheduler" in checkpoint,
        "global_step_present": "global_step" in checkpoint or "step" in checkpoint,
        "random_states_present": any(key in checkpoint for key in ("random_state", "rng_state", "random_states")),
        "full_training_resume_status": "PARTIAL",
        "reason": "checkpoint supports deterministic model/graph inference parity but does not contain optimizer, scheduler, global-step or RNG state; no training step was run",
    }
    summary = {
        "module3_run_commit": metadata["Git_commit"],
        "current_audit_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "module3_output": str(source),
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
        "training_steps_logged": [0, 100], "selected_step": json.loads((source / "forward.log").read_text())["selected_step"],
        "original_status": json.loads((source / "forward.log").read_text())["status"],
        "visual_artifacts_complete_before_supplement": False,
        "supplementary_output": str(output),
    }
    for name, value in (
        ("inactive_region_leakage.json", leakage), ("formal_dataflow_audit.json", dataflow),
        ("checkpoint_resume_audit.json", resume), ("supplementary_summary.json", summary),
    ): (output / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
    build_visual_acceptance("module3", output, output, None)
    print(json.dumps({"summary": summary, "leakage": leakage, "dataflow": dataflow, "resume": resume}, indent=2))


if __name__ == "__main__": main()
