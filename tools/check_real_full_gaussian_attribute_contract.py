from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torchvision.utils import make_grid, save_image

import train_dressable as training
from scene.dressable_dataset import ImageConditionedEpisodeDataset
from scene.gaussian_clothing_residuals import (
    CHANNELS,
    GaussianClothingResiduals,
    compose_canonical_gaussian_overrides,
    residual_contract_metadata,
)
from tools.check_real_image_conditioned_one_batch import _build_fixed_episode, _as_chw_render, save_render_tensor
from utils.dressable_camera_utils import build_mmlphuman_camera
from utils.mmlphuman_state_utils import mmlphuman_state_transaction


DISPLAY = {
    "delta_xyz": "xyz", "delta_log_scaling": "scaling", "delta_rotvec": "rotation",
    "delta_opacity_logit": "opacity", "delta_sh0": "sh0", "delta_shN": "shN",
}
AMPLITUDES = {
    "delta_xyz": 0.01, "delta_log_scaling": 0.25, "delta_rotvec": 0.35,
    "delta_opacity_logit": 0.75, "delta_sh0": 0.15, "delta_shN": 0.15,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_stats(value: torch.Tensor) -> dict[str, Any]:
    numeric = value.detach().float()
    return {
        "shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device),
        "min": float(numeric.min().item()), "max": float(numeric.max().item()),
        "mean": float(numeric.mean().item()),
        "finite_ratio": float(torch.isfinite(numeric).float().mean().item()),
    }


def diff_stats(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    difference = (left.detach().float() - right.detach().float()).abs()
    return {"max_abs_diff": float(difference.max().item()), "mean_abs_diff": float(difference.mean().item())}


def base_tensors(base) -> dict[str, torch.Tensor]:
    return {name: getattr(base, name) for name in ("_xyz", "_scaling", "_rotation", "_opacity", "_sh0", "_shN")}


def clone_base(base) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in base_tensors(base).items()}


def base_difference(base, before) -> dict[str, float]:
    return {name: float((getattr(base, name).detach() - value).abs().max().item()) for name, value in before.items()}


def make_residual(base, channel: str, indices: torch.Tensor, requires_grad: bool) -> tuple[GaussianClothingResiduals, torch.Tensor]:
    values = GaussianClothingResiduals.zeros(base, [channel]).as_dict()
    tensor = values[channel]
    assert tensor is not None
    tensor = tensor.clone()
    amplitude = AMPLITUDES[channel]
    if channel == "delta_xyz": tensor[indices, 0] = amplitude
    elif channel == "delta_log_scaling": tensor[indices, 0] = amplitude
    elif channel == "delta_rotvec": tensor[indices, 1] = amplitude
    elif channel == "delta_opacity_logit": tensor.reshape(tensor.shape[0], -1)[indices, 0] = amplitude
    elif channel == "delta_sh0": tensor.reshape(tensor.shape[0], -1)[indices, 0] = amplitude
    elif channel == "delta_shN": tensor.reshape(tensor.shape[0], -1)[indices, 0] = amplitude
    tensor.requires_grad_(requires_grad)
    values[channel] = tensor
    return GaussianClothingResiduals.from_dict(values).validate(base), tensor


def main() -> None:
    parser = argparse.ArgumentParser(description="Real six-channel Gaussian attribute contract acceptance")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--base-checkpoint", default="chkpnt100000.pth")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--old-image-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    base_path = Path(args.model_dir) / args.base_checkpoint
    base = training.load_frozen_mmlphuman_base(args.model_dir, args.base_checkpoint, device)
    dataset = ImageConditionedEpisodeDataset(
        args.manifest, split="train", reference_count=2, seed=42,
        allow_clothing_mask_fallback=False, allow_missing_rh_th=False,
    )
    episode = _build_fixed_episode(dataset)
    camera = build_mmlphuman_camera(
        episode["target_camera"], episode["target_rgb"].shape[-2], episode["target_rgb"].shape[-1], device
    )
    background = torch.ones(3, device=device, dtype=base._xyz.dtype)
    audited_sh_degree = int(base.sh_degree)

    def render(overrides=None, sh_degree=None):
        original_degree = base.sh_degree
        try:
            if sh_degree is not None:
                base.sh_degree = int(sh_degree)
            with mmlphuman_state_transaction(base, episode["target_pose"], episode["target_Rh"], episode["target_Th"]):
                return base.render(camera, background=background, canonical_overrides=None if overrides is None else overrides.as_dict())
        finally:
            base.sh_degree = original_degree

    before = clone_base(base)
    base_rgb, base_alpha, info = render()
    radii = info.get("radii") if isinstance(info, dict) else None
    if not isinstance(radii, torch.Tensor):
        raise KeyError("renderer info lacks radii needed for visible Gaussian selection")
    visible_mask = radii.reshape(-1, radii.shape[-1]).amax(dim=0) > 0
    visible = torch.nonzero(visible_mask, as_tuple=False).reshape(-1)
    if visible.numel() < 8:
        raise RuntimeError("fewer than eight visible Gaussians")
    count = min(64, visible.numel())
    selected = visible[:count]
    activated_scaling = base.scaling_activation(base._scaling[selected])
    anisotropy = activated_scaling.max(dim=-1).values / activated_scaling.min(dim=-1).values.clamp_min(1e-8)
    rotation_selected = selected[torch.argsort(anisotropy, descending=True)[:count]]

    zero = GaussianClothingResiduals.zeros(base)
    zero_overrides = compose_canonical_gaussian_overrides(base, zero, CHANNELS)
    zero_rgb, zero_alpha, _ = render(zero_overrides)
    zero_equivalence = {"rgb": diff_stats(base_rgb, zero_rgb), "alpha": diff_stats(base_alpha, zero_alpha)}

    render_metrics, gradient_metrics, rendered = {}, {}, {}
    for channel in CHANNELS:
        channel_indices = rotation_selected if channel == "delta_rotvec" else selected
        evaluation_degree = 1 if channel == "delta_shN" else int(base.sh_degree)
        channel_base_rgb, channel_base_alpha = (
            render(sh_degree=evaluation_degree)[:2]
            if channel == "delta_shN" else (base_rgb, base_alpha)
        )
        residuals, leaf = make_residual(base, channel, channel_indices, requires_grad=True)
        overrides = compose_canonical_gaussian_overrides(base, residuals, [channel])
        rgb, alpha, _ = render(overrides, sh_degree=evaluation_degree)
        loss = rgb.float().mean() + 0.1 * alpha.float().mean()
        if loss.requires_grad:
            loss.backward()
        gradient = leaf.grad
        render_metrics[channel] = {
            "selected_gaussian_count": int(channel_indices.numel()),
            "selected_indices": channel_indices.detach().cpu().tolist(),
            "amplitude": AMPLITUDES[channel], "residual": tensor_stats(leaf),
            "evaluation_sh_degree": evaluation_degree,
            "rgb": diff_stats(channel_base_rgb, rgb), "alpha": diff_stats(channel_base_alpha, alpha),
            "rgb_finite_ratio": float(torch.isfinite(rgb).float().mean().item()),
            "alpha_finite_ratio": float(torch.isfinite(alpha).float().mean().item()),
            "base_parameter_max_diff": base_difference(base, before),
        }
        gradient_metrics[channel] = {
            "exists": gradient is not None,
            "finite_ratio": 0.0 if gradient is None else float(torch.isfinite(gradient).float().mean().item()),
            "norm": 0.0 if gradient is None else float(gradient.float().norm().item()),
            "max_abs": 0.0 if gradient is None else float(gradient.float().abs().max().item()),
            "loss_requires_grad": bool(loss.requires_grad),
            "base_gradient_count": sum(value.grad is not None for value in base_tensors(base).values()),
        }
        rendered[channel] = rgb.detach()
        save_render_tensor(output / f"{DISPLAY[channel]}_rgb.png", rgb, 3)
        if leaf.grad is not None:
            leaf.grad = None

    final_rgb, final_alpha, _ = render()
    state_restore = {
        "base_rgb_before_after": diff_stats(base_rgb, final_rgb),
        "base_alpha_before_after": diff_stats(base_alpha, final_alpha),
        "base_parameter_max_diff": base_difference(base, before),
        "state_restore": bool(diff_stats(base_rgb, final_rgb)["max_abs_diff"] == 0 and diff_stats(base_alpha, final_alpha)["max_abs_diff"] == 0),
        "sh_degree_restored": int(base.sh_degree) == audited_sh_degree,
    }
    contract = residual_contract_metadata(base, CHANNELS)
    contract.update({
        "base_checkpoint": str(base_path.resolve()), "base_checkpoint_sha256": sha256(base_path),
        "raw_attributes": {name: tensor_stats(value) for name, value in base_tensors(base).items()},
        "field_aliases": {"_features_dc": "_sh0", "_features_rest": "_shN"},
        "activations": {"scaling": "torch.exp", "opacity": "torch.sigmoid", "rotation": "torch.nn.functional.normalize"},
        "renderer_getters": {"xyz": "compute_xyz", "scaling_rotation": "get_covariance", "opacity": "compute_opacity", "sh": "compute_sh/get_color"},
        "visible_gaussian_count": int(visible.numel()), "seed": args.seed,
        "sh_degree": int(base.sh_degree),
    })
    (output / "base_attribute_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")

    saved = torch.load(args.old_image_checkpoint, map_location="cpu", weights_only=True)
    old_config = saved["config"]
    old_base = training.load_frozen_mmlphuman_base(old_config["base"]["model_dir"], old_config["base"]["checkpoint_path"], device)
    old_model, old_optimizer, _, _ = training.create_image_conditioned_components(dataset, old_config, old_base, device)
    restored_step = training.load_image_training_checkpoint(
        args.old_image_checkpoint, old_model, old_optimizer, old_config, dataset,
        getattr(old_base, "_dressable_base_fingerprint", None),
    )
    compatibility = {
        "checkpoint": str(Path(args.old_image_checkpoint).resolve()), "checkpoint_sha256": sha256(Path(args.old_image_checkpoint)),
        "restored_step": restored_step, "legacy_metadata_present": "residual_contract_version" in saved,
        "migration": residual_contract_metadata(old_base, ["delta_xyz"]),
        "new_channels_default": "disabled/zero", "status": "PASS" if restored_step == 100 else "FAIL",
    }
    del old_model, old_optimizer, old_base

    save_render_tensor(output / "base_rgb.png", base_rgb, 3)
    panels = [_as_chw_render(base_rgb, 3)]
    for channel in CHANNELS:
        panels.extend([_as_chw_render(rendered[channel], 3), (_as_chw_render(rendered[channel], 3) - _as_chw_render(base_rgb, 3)).abs()])
    save_image(make_grid(panels, nrow=3), output / "attribute_comparison.png")
    pass_render = all(value["rgb"]["max_abs_diff"] > 0 or value["alpha"]["max_abs_diff"] > 0 for value in render_metrics.values())
    pass_grad = all(value["exists"] and value["finite_ratio"] == 1 and value["norm"] > 0 and value["base_gradient_count"] == 0 for value in gradient_metrics.values())
    pass_zero = zero_equivalence["rgb"]["max_abs_diff"] == 0 and zero_equivalence["alpha"]["max_abs_diff"] == 0
    pass_state = state_restore["state_restore"] and state_restore["sh_degree_restored"] and not any(state_restore["base_parameter_max_diff"].values())
    passed = pass_render and pass_grad and pass_zero and pass_state and compatibility["status"] == "PASS"
    (output / "base_attribute_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    (output / "acceptance_metrics.json").write_text(json.dumps(render_metrics, indent=2), encoding="utf-8")
    (output / "gradient_metrics.json").write_text(json.dumps(gradient_metrics, indent=2), encoding="utf-8")
    (output / "state_restore_metrics.json").write_text(json.dumps({"zero_equivalence": zero_equivalence, **state_restore}, indent=2), encoding="utf-8")
    (output / "checkpoint_compatibility.json").write_text(json.dumps(compatibility, indent=2), encoding="utf-8")
    (output / "attribute_contract.md").write_text(f"# Full Gaussian Attribute Contract\n\nStatus: **{'PASS' if passed else 'FAIL'}**\n\n- Quaternion: wxyz, local base * delta\n- Visible Gaussians: {visible.numel()}\n- Six render channels: {'PASS' if pass_render else 'FAIL'}\n- Six gradient channels: {'PASS' if pass_grad else 'FAIL'}\n- Zero equivalence: {'PASS' if pass_zero else 'FAIL'}\n- State restore: {'PASS' if pass_state else 'FAIL'}\n- Old checkpoint: {compatibility['status']}\n", encoding="utf-8")
    (output / "GATE_ACCEPTANCE.md").write_text(f"# Gate 5 Full Attribute Contract\n\nStatus: **{'PASS' if passed else 'FAIL'}**\n", encoding="utf-8")
    print(json.dumps({"status": "PASS" if passed else "FAIL", "contract": contract, "render": render_metrics, "gradient": gradient_metrics, "zero": zero_equivalence, "state": state_restore, "compatibility": compatibility}, indent=2))
    if not passed:
        raise RuntimeError("full Gaussian attribute contract acceptance failed")


if __name__ == "__main__":
    main()
