from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_dressable as training
from scene.gaussian_clothing_residuals import CHANNELS
from tools.check_differentiable_render_path import (
    BASE_PARAMETER_NAMES,
    _build_model,
    _cpu_camera,
    _prepare_episode,
)
from tools.check_real_image_conditioned_one_batch import _as_chw_render, save_render_tensor
from utils.dressable_camera_utils import build_mmlphuman_camera
from utils.mmlphuman_anchor_deformation import MMLPHumanAnchorDeformationAdapter
from utils.mmlphuman_state_utils import mmlphuman_state_transaction
from utils.rendering_loss_utils import region_aware_dual_target_loss


def _state_clone(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _state_equal(before: dict[str, torch.Tensor], module: torch.nn.Module) -> bool:
    after = module.state_dict()
    return all(torch.equal(value, after[name].detach().cpu()) for name, value in before.items())


def _tensor_image(value: torch.Tensor, channels: int) -> Image.Image:
    tensor = _as_chw_render(value, channels).detach().float().clamp(0, 1).cpu()
    if channels == 1:
        return Image.fromarray((tensor[0].numpy() * 255).round().astype(np.uint8), "L")
    return Image.fromarray((tensor.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8), "RGB")


def _contact_sheet(items: list[tuple[str, Image.Image]], path: Path) -> None:
    cell = (256, 384)
    canvas = Image.new("RGB", (cell[0] * 3, (cell[1] + 28) * 3), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(items):
        picture = image.convert("RGB").resize(cell, Image.Resampling.LANCZOS)
        x, y = (index % 3) * cell[0], (index // 3) * (cell[1] + 28)
        canvas.paste(picture, (x, y))
        draw.text((x + 5, y + cell[1] + 5), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _forward(
    model: torch.nn.Module,
    base: Any,
    adapter: MMLPHumanAnchorDeformationAdapter,
    episode: dict[str, Any],
    geometry: dict[str, Any],
    checkpoint: dict[str, Any],
    background: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    inputs = {
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
    result = model.compute_online_six_channel_residuals(**inputs)
    overrides = model.dressable_model.compose_canonical_gaussian_overrides(result["gaussian_residuals"], CHANNELS)
    camera = build_mmlphuman_camera(
        episode["target_camera"], episode["target_height"], episode["target_width"], device,
    )
    original_degree = int(base.sh_degree)
    try:
        base.sh_degree = int(checkpoint["metadata"]["module2_decoder_metadata"]["effective_sh_degree"])
        with mmlphuman_state_transaction(
            base, episode["target_pose"].to(device), episode["target_Rh"].to(device), episode["target_Th"].to(device),
        ):
            rendered_rgb, rendered_alpha, _ = base.render(
                camera, background=background, canonical_overrides=overrides.as_dict(),
            )
    finally:
        base.sh_degree = original_degree
    return result, _as_chw_render(rendered_rgb, 3), _as_chw_render(rendered_alpha, 1)


def _loss(
    rgb: torch.Tensor,
    alpha: torch.Tensor,
    sample: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    return region_aware_dual_target_loss(
        rgb, alpha,
        sample["target_edit_rgb"].to(device), sample["target_base_rgb"].to(device),
        sample["target_edit_core_mask"].to(device), sample["target_preserve_mask"].to(device),
        sample["target_protected_mask"].to(device), sample["target_transition_mask"].to(device),
        sample["target_clothing_mask"].to(device), sample["target_foreground_mask"].to(device),
        sample["target_base_foreground_mask"].to(device),
        alpha_weight=args.alpha_weight,
        alpha_edit_weight=args.alpha_edit_weight,
        alpha_transition_weight=args.alpha_transition_weight,
        alpha_base_weight=args.alpha_base_weight,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fixed <=20-step V5.1 supervision smoke")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gate4-config", type=Path, required=True)
    parser.add_argument("--module2-config", type=Path, required=True)
    parser.add_argument("--module3-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=11)
    parser.add_argument("--steps", type=int, default=20)
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
    if not 1 <= args.steps <= 20:
        raise ValueError("steps must be in [1,20]")
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    sample, episode, _ = _prepare_episode(args.manifest, sample_index=args.sample_index)
    if sample["outfit_id"] != "O05" or sample["target_condition_id"] != "cond_000347":
        raise ValueError("V5.1 smoke must target cond_000347_O05")
    base, model, checkpoint, gate4 = _build_model(args.gate4_config, args.module2_config, args.module3_checkpoint, device)
    base_before = {name: getattr(base, name).detach().cpu().clone() for name in BASE_PARAMETER_NAMES}
    backbone_before = _state_clone(model.clothing_observation_encoder.backbone)
    adapter = MMLPHumanAnchorDeformationAdapter.from_mmlphuman_base(
        base, canonical_anchors=model.canonical_anchors, lbs_grid_path=gate4["base"]["lbs_grid_path"],
    )
    background = torch.tensor(gate4["render"]["background"], device=device, dtype=base._xyz.dtype)
    geometry = training.prepare_real_reference_geometry(base, adapter, episode, background)
    optimizer = torch.optim.Adam([value for value in model.parameters() if value.requires_grad], lr=args.learning_rate)
    log_steps = {0, 1, 5, 10, args.steps}
    history = []
    step_images: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        result, rgb, alpha = _forward(model, base, adapter, episode, geometry, checkpoint, background, device)
        parts = _loss(rgb, alpha, sample, device, args)
        residual_magnitude = sum(getattr(result["gated_anchor_residuals"], name).abs().mean() for name in CHANNELS)
        geometry_gate = result["completion"].geometry_gate
        appearance_gate = result["completion"].appearance_gate
        gate_regularization = (
            (geometry_gate * (1 - geometry_gate)).mean()
            + (appearance_gate * (1 - appearance_gate)).mean()
        )
        objective = (
            parts["total"]
            + args.residual_regularization_weight * residual_magnitude
            + args.gate_regularization_weight * gate_regularization
        )
        if not torch.isfinite(objective):
            raise FloatingPointError(f"non-finite objective at step {step}")
        if step in log_steps:
            history.append({
                "step": step,
                **{name: float(value.detach()) for name, value in parts.items()},
                "gate_regularization": float(gate_regularization.detach()),
                "residual_magnitude": float(residual_magnitude.detach()),
                "objective": float(objective.detach()),
            })
        if step in {0, args.steps}:
            step_images[step] = (rgb.detach().cpu(), alpha.detach().cpu())
        if step == args.steps:
            break
        objective.backward()
        optimizer.step()

    base_unchanged = all(torch.equal(value, getattr(base, name).detach().cpu()) for name, value in base_before.items())
    backbone_unchanged = _state_equal(backbone_before, model.clothing_observation_encoder.backbone)
    first, final = history[0], history[-1]
    criteria = {
        "finite": all(math.isfinite(value) for row in history for key, value in row.items() if key != "step"),
        "edit_decreased": final["edit"] < first["edit"],
        "clothing_nonincreasing": final["clothing"] <= first["clothing"] + 1e-6,
        "protected_not_significantly_worse": final["protected"] <= first["protected"] * 1.10 + 1e-6,
        "alpha_base_not_significantly_worse": final["alpha_base"] <= first["alpha_base"] * 1.10 + 1e-6,
        "base_bitwise_unchanged": base_unchanged,
        "backbone_bitwise_unchanged": backbone_unchanged,
    }

    manifest_payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    shoe_path = Path(manifest_payload["protected_region_final_adjudication"]["target_contract"]["shoe_mask"])
    shoe_path = shoe_path if shoe_path.is_absolute() else args.manifest.resolve().parent / shoe_path
    shoe = torch.from_numpy(np.array(Image.open(shoe_path).convert("L"), copy=True) >= 128)
    base_rgb = sample["target_base_rgb"].cpu(); edit_rgb = sample["target_edit_rgb"].cpu()
    rgb0, alpha0 = step_images[0]; rgbf, alphaf = step_images[args.steps]
    shoe3 = shoe.unsqueeze(0).expand_as(base_rgb)
    shoe1 = shoe.unsqueeze(0)
    visual_metrics = {
        "raw_edit_vs_base_shoe_rgb_mae": float((edit_rgb - base_rgb).abs()[shoe3].mean()),
        "step0_vs_base_shoe_rgb_mae": float((rgb0 - base_rgb).abs()[shoe3].mean()),
        "step20_vs_base_shoe_rgb_mae": float((rgbf - base_rgb).abs()[shoe3].mean()),
        "step0_vs_raw_shoe_rgb_mae": float((rgb0 - edit_rgb).abs()[shoe3].mean()),
        "step20_vs_raw_shoe_rgb_mae": float((rgbf - edit_rgb).abs()[shoe3].mean()),
        "step0_vs_base_shoe_alpha_mae": float((alpha0 - sample["target_base_foreground_mask"]).abs()[shoe1].mean()),
        "step20_vs_base_shoe_alpha_mae": float((alphaf - sample["target_base_foreground_mask"]).abs()[shoe1].mean()),
    }
    criteria["shoe_closer_to_base_than_raw_target"] = (
        visual_metrics["step20_vs_base_shoe_rgb_mae"] < visual_metrics["step20_vs_raw_shoe_rgb_mae"]
    )
    status = "PASS" if all(criteria.values()) else "PARTIAL"
    save_render_tensor(output / "prediction_step0.png", rgb0, 3)
    save_render_tensor(output / "prediction_step20.png", rgbf, 3)
    save_render_tensor(output / "prediction_alpha_step0.png", alpha0, 1)
    save_render_tensor(output / "prediction_alpha_step20.png", alphaf, 1)
    shoe_rgb_diff = ((rgbf - base_rgb).abs() * shoe3).clamp(0, 1)
    shoe_alpha_diff = ((alphaf - sample["target_base_foreground_mask"]).abs() * shoe1).clamp(0, 1)
    items = [
        ("subject02 base white shoes", _tensor_image(base_rgb, 3)),
        ("raw edit black shoes", _tensor_image(edit_rgb, 3)),
        ("protected shoe mask", Image.open(shoe_path).convert("L")),
        ("edit mask", _tensor_image(sample["target_edit_mask"], 1)),
        ("preserve mask", _tensor_image(sample["target_preserve_mask"], 1)),
        ("prediction step 0", _tensor_image(rgb0, 3)),
        ("prediction step 20", _tensor_image(rgbf, 3)),
        ("shoe RGB diff vs base", _tensor_image(shoe_rgb_diff, 3)),
        ("shoe alpha diff vs base", _tensor_image(shoe_alpha_diff, 1)),
    ]
    _contact_sheet(items, output / "dual_target_loss_smoke_contact_sheet_v5_1.png")
    report = {
        "status": status,
        "scope": "fixed V5.1 episode; no representation/capacity claim",
        "steps": args.steps,
        "sample": {"outfit_id": sample["outfit_id"], "target": sample["target_condition_id"], "references": sample["reference_condition_ids"]},
        "weights": {
            "learning_rate": args.learning_rate,
            "alpha_global": args.alpha_weight,
            "alpha_edit": args.alpha_edit_weight,
            "alpha_transition": args.alpha_transition_weight,
            "alpha_base": args.alpha_base_weight,
            "residual_regularization": args.residual_regularization_weight,
            "gate_regularization": args.gate_regularization_weight,
        },
        "history": history,
        "criteria": criteria,
        "visual_metrics": visual_metrics,
        "git": {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip(),
            "status_short": subprocess.check_output(["git", "status", "--short"], cwd=Path(__file__).resolve().parents[1], text=True).splitlines(),
        },
    }
    (output / "dual_target_loss_smoke_v5_1.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
