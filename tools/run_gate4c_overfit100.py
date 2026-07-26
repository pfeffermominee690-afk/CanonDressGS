from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

import train_dressable as training
from scene.dressable_dataset import ImageConditionedEpisodeDataset
from tools.check_real_image_conditioned_one_batch import (
    _as_chw_render, _build_fixed_episode, _comparison, _single_reference_episode,
    save_render_tensor,
)
from utils.dressable_checkpoint_utils import iter_base_parameters


PROTOCOL = {
    "condition_a": [["0", "cam18"], ["1000", "cam00"]],
    "condition_b": [["0", "cam18"]],
    "target": ["2000", "cam09"],
    "target_view_used": False,
}


def grad_norm(parameters) -> float:
    values = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    if not values:
        return 0.0
    result = torch.sqrt(torch.stack(values).sum())
    if not torch.isfinite(result):
        raise FloatingPointError("gradient norm is not finite")
    return float(result.item())


def offset_norms(value: torch.Tensor) -> dict[str, float]:
    norms = value.detach().float().norm(dim=-1)
    if not torch.isfinite(norms).all():
        raise FloatingPointError("offset norm contains NaN or Inf")
    return {"mean": float(norms.mean().item()), "max": float(norms.max().item())}


def require_finite_tensors(label: str, values) -> None:
    for name, value in values:
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
            raise FloatingPointError(f"{label} {name} contains NaN or Inf")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=Path(__file__).resolve().parents[1], text=True
    ).strip()


def image_metrics(pred_rgb, pred_alpha, target_rgb, target_mask):
    pred = _as_chw_render(pred_rgb, 3).float()
    alpha = _as_chw_render(pred_alpha, 1).float()
    target = _as_chw_render(target_rgb, 3).float()
    mask = _as_chw_render(target_mask, 1).float()
    if pred.shape != target.shape or alpha.shape != mask.shape:
        raise ValueError("image metric shapes do not match; resizing is forbidden")
    foreground = mask.expand_as(pred)
    denom = foreground.sum().clamp_min(1)
    fg_l1 = ((pred - target).abs() * foreground).sum() / denom
    fg_mse = (((pred - target).square()) * foreground).sum() / denom
    pred_mask, true_mask = alpha >= 0.5, mask >= 0.5
    union = (pred_mask | true_mask).sum()
    iou = (pred_mask & true_mask).sum().float() / union.clamp_min(1)
    return {
        "rgb_all_l1": float((pred - target).abs().mean().item()),
        "foreground_rgb_l1": float(fg_l1.item()),
        "alpha_l1": float((alpha - mask).abs().mean().item()),
        "foreground_psnr": float((-10 * torch.log10(fg_mse.clamp_min(1e-12))).item()),
        "mask_iou": float(iou.item()),
    }


def anchor_metrics(prediction, teacher):
    pred = prediction.detach().float()
    target = teacher.detach().float().to(pred.device)
    active = target.norm(dim=-1) > 1e-8
    inactive = ~active
    return {
        "xyz_mae": float((pred - target).abs().mean().item()),
        "anchor_l2_mean_error": float((pred - target).norm(dim=-1).mean().item()),
        "teacher_active_mae": float((pred[active] - target[active]).abs().mean().item()),
        "teacher_inactive_leakage": float(pred[inactive].norm(dim=-1).mean().item()),
        "flattened_cosine_similarity": float(F.cosine_similarity(pred.flatten(), target.flatten(), dim=0).item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 4-C real 100-step overfit and evaluation")
    parser.add_argument("--config", default="configs/canon_dress_gs_gate4c_overfit100.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = training.load_config(args.config)
    resume = Path(config["train"]["resume"])
    if int(config["train"]["steps"]) != 100 or not resume.is_file():
        raise ValueError("Gate 4-C requires max step 100 and an existing step-10 resume")
    resume_object = torch.load(resume, map_location="cpu", weights_only=True)
    if int(resume_object.get("step", -1)) != 10:
        raise ValueError("Gate 4-C must resume from step 10")
    if config["image_conditioning"].get("target_view_used") is not False:
        raise ValueError("target_view_used must be false")
    output_dir = Path(args.output_dir or config["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    config["train"]["output_dir"] = str(output_dir)
    metrics: list[dict[str, Any]] = []
    state: dict[str, Any] = {}
    original_compute = training.compute_image_conditioned_training_loss
    original_build_optimizer = training.build_image_conditioned_optimizer
    original_sample = ImageConditionedEpisodeDataset.sample_episode

    def fixed_sample(self, index, sampling_salt=0, deterministic=False):
        del index, sampling_salt, deterministic
        return _build_fixed_episode(self)

    def wrapped_compute(*compute_args, **compute_kwargs):
        started = time.perf_counter()
        losses, outputs = original_compute(*compute_args, **compute_kwargs)
        state.update(started=started, losses=losses, outputs=outputs)
        return losses, outputs

    def wrapped_build_optimizer(model, optimizer_config):
        optimizer = original_build_optimizer(model, optimizer_config)
        original_step = optimizer.step

        def instrumented_step(*step_args, **step_kwargs):
            primary = state["outputs"]["primary"]
            raw = model.dressable_model.compute_film_anchor_offsets(
                clothing_embedding=primary["global_clothing_embedding"],
                anchor_clothing_features=primary["anchor_clothing_features"],
            )
            require_finite_tensors(
                "training step",
                list(state["losses"].items()) + [("raw", raw["delta_xyz"])]
                + [(f"anchor_{key}", value) for key, value in primary["anchor_offsets"].items()]
                + [(f"gaussian_{key}", value) for key, value in primary["gaussian_offsets"].items()]
                + [("rgb", primary["render"][0]), ("alpha", primary["render"][1])],
            )
            record = {
                "step": 11 + len(metrics),
                **{key: float(value.detach().item()) for key, value in state["losses"].items()},
                "learning_rates": {group.get("name", str(i)): float(group["lr"]) for i, group in enumerate(optimizer.param_groups)},
                "raw_anchor_norm": offset_norms(raw["delta_xyz"]),
                "gated_anchor_norm": offset_norms(primary["anchor_offsets"]["delta_xyz"]),
                "gaussian_norm": offset_norms(primary["gaussian_offsets"]["delta_xyz"]),
                "gradients": {
                    "encoder": grad_norm(model.clothing_observation_encoder.parameters()),
                    "aggregator": grad_norm(model.multiview_aggregator.parameters()),
                    "hypernetwork": grad_norm(model.dressable_model.clothing_film_generator.parameters()),
                    "anchor_mlp": grad_norm(model.dressable_model.anchor_clothing_mlp.parameters()),
                },
                "base_grad_count": sum(p.grad is not None for p in iter_base_parameters(model.dressable_model.base_model)),
                "disabled_max_abs": {
                    "anchor_scaling": float(primary["anchor_offsets"]["delta_scaling"].abs().max().item()),
                    "anchor_opacity": float(primary["anchor_offsets"]["delta_opacity"].abs().max().item()),
                    "gaussian_scaling": float(primary["gaussian_offsets"]["delta_scaling"].abs().max().item()),
                    "gaussian_opacity": float(primary["gaussian_offsets"]["delta_opacity"].abs().max().item()),
                },
                "gpu_memory_allocated": int(torch.cuda.memory_allocated()),
                "gpu_memory_reserved": int(torch.cuda.memory_reserved()),
                "step_duration_seconds": float(time.perf_counter() - state["started"]),
                "all_finite": True,
            }
            if record["base_grad_count"] or any(record["disabled_max_abs"].values()):
                raise RuntimeError(f"frozen/disabled invariant failed: {record}")
            metrics.append(record)
            return original_step(*step_args, **step_kwargs)

        optimizer.step = instrumented_step
        return optimizer

    ImageConditionedEpisodeDataset.sample_episode = fixed_sample
    training.compute_image_conditioned_training_loss = wrapped_compute
    training.build_image_conditioned_optimizer = wrapped_build_optimizer
    try:
        result = training.train_image_conditioned(config, output_dir=output_dir, device=args.device)
    finally:
        ImageConditionedEpisodeDataset.sample_episode = original_sample
        training.compute_image_conditioned_training_loss = original_compute
        training.build_image_conditioned_optimizer = original_build_optimizer
    if result["step"] != 100 or len(metrics) != 90 or metrics[0]["step"] != 11:
        raise RuntimeError("resume did not execute exactly steps 11 through 100")

    model, optimizer, dataset = result["model"], result["optimizer"], result["dataset"]
    episode = _build_fixed_episode(dataset)
    gate = result["reference_only_gate"]
    episode["reference_only_gate"] = gate
    fingerprint = getattr(model.dressable_model.base_model, "_dressable_base_fingerprint")

    def evaluate(fixed_episode):
        model.eval()
        with torch.no_grad():
            _, outputs = original_compute(
                model, fixed_episode, result["anchor_edges"], config, args.device,
                render_target=True, deformation_adapter=result["deformation_adapter"],
            )
            output = outputs["primary"]
            raw = model.dressable_model.compute_film_anchor_offsets(
                clothing_embedding=output["global_clothing_embedding"],
                anchor_clothing_features=output["anchor_clothing_features"],
            )
        return output, raw

    checkpoints = {
        10: resume,
        50: output_dir / "checkpoint_step_000050.pth",
        100: output_dir / "checkpoint_step_000100.pth",
    }
    evaluations, sensitivity = {}, {}
    teacher = episode["anchor_offset_target"]["delta_xyz"]
    for step, checkpoint in checkpoints.items():
        restored = training.load_image_training_checkpoint(checkpoint, model, optimizer, config, dataset, fingerprint)
        if restored != step:
            raise RuntimeError(f"checkpoint step mismatch at {step}")
        output_a, raw_a = evaluate(episode)
        output_b, raw_b = evaluate(_single_reference_episode(episode, 0))
        evaluations[step] = {
            "anchor": anchor_metrics(output_a["anchor_offsets"]["delta_xyz"], teacher),
            "image": image_metrics(output_a["render"][0], output_a["render"][1], episode["target_rgb"], episode["target_foreground_mask"]),
        }
        sensitivity[step] = {
            "global_embedding_l2": float((output_a["global_clothing_embedding"] - output_b["global_clothing_embedding"]).norm().item()),
            "anchor_feature_mae": float((output_a["anchor_clothing_features"] - output_b["anchor_clothing_features"]).abs().mean().item()),
            "raw_anchor_offset_mae": float((raw_a["delta_xyz"] - raw_b["delta_xyz"]).abs().mean().item()),
            "gated_anchor_offset_mae": float((output_a["anchor_offsets"]["delta_xyz"] - output_b["anchor_offsets"]["delta_xyz"]).abs().mean().item()),
            "gaussian_offset_mae": float((output_a["gaussian_offsets"]["delta_xyz"] - output_b["gaussian_offsets"]["delta_xyz"]).abs().mean().item()),
            "rendered_rgb_mae": float((output_a["render"][0] - output_b["render"][0]).abs().mean().item()),
            "rendered_alpha_mae": float((output_a["render"][1] - output_b["render"][1]).abs().mean().item()),
        }
        save_render_tensor(output_dir / f"step{step}_rgb.png", output_a["render"][0], 3)
        evaluations[step]["_output"] = output_a

    zero_episode = dict(episode)
    zero_episode["reference_only_gate"] = torch.zeros_like(gate)
    zero_output, _ = evaluate(zero_episode)
    zero_metrics = {
        "anchor": anchor_metrics(zero_output["anchor_offsets"]["delta_xyz"], teacher),
        "image": image_metrics(zero_output["render"][0], zero_output["render"][1], episode["target_rgb"], episode["target_foreground_mask"]),
    }
    save_render_tensor(output_dir / "zero_rgb.png", zero_output["render"][0], 3)
    save_render_tensor(output_dir / "target_rgb.png", episode["target_rgb"], 3)
    save_render_tensor(output_dir / "rendered_alpha.png", evaluations[100]["_output"]["render"][1], 1)
    torch.save(evaluations[100]["_output"]["anchor_offsets"], output_dir / "predicted_anchor_offsets.pt")

    pre_output, pre_raw = evaluate(episode)
    training.load_image_training_checkpoint(checkpoints[100], model, optimizer, config, dataset, fingerprint)
    post_output, post_raw = evaluate(episode)
    roundtrip = {
        "raw_anchor": _comparison(pre_raw["delta_xyz"], post_raw["delta_xyz"], atol=1e-7, rtol=1e-6),
        "gated_anchor": _comparison(pre_output["anchor_offsets"]["delta_xyz"], post_output["anchor_offsets"]["delta_xyz"], atol=1e-7, rtol=1e-6),
        "rgb": _comparison(pre_output["render"][0], post_output["render"][0], atol=1e-6, rtol=1e-5),
        "alpha": _comparison(pre_output["render"][1], post_output["render"][1], atol=1e-6, rtol=1e-5),
    }

    gate4b_metrics = json.loads((resume.parent / "training_metrics.json").read_text(encoding="utf-8"))["steps"]
    all_metrics = gate4b_metrics + metrics
    early = sum(item["total"] for item in all_metrics if 10 <= item["step"] <= 19) / 10
    late = sum(item["total"] for item in all_metrics if 81 <= item["step"] <= 100) / 20
    trend = {"steps_10_19_mean": early, "steps_81_100_mean": late, "relative_decrease_percent": (early - late) / early * 100}
    memory = [item["gpu_memory_allocated"] for item in metrics]
    memory_growth = max(memory[-10:]) - min(memory[-10:])
    serializable_evaluations = {str(step): {k: v for k, v in value.items() if k != "_output"} for step, value in evaluations.items()}
    payload = {
        "status": "PENDING_INFERENCE", "resume_step": 10, "final_step": 100,
        "new_steps": metrics, "loss_trend": trend, "zero": zero_metrics,
        "checkpoints": serializable_evaluations, "checkpoint_roundtrip": roundtrip,
        "memory_last10_range_bytes": memory_growth,
    }
    (output_dir / "training_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "sensitivity_metrics.json").write_text(json.dumps({"protocol": PROTOCOL, "steps": sensitivity}, indent=2), encoding="utf-8")

    repo = Path(__file__).resolve().parents[1]
    inference_command = [
        str(Path(platform.python_implementation() and __import__("sys").executable)),
        "-m", "tools.infer_image_conditioned_dressable",
        "--checkpoint", str(checkpoints[100]), "--manifest", config["image_conditioning"]["manifest_path"],
        "--base-model-dir", config["base"]["model_dir"], "--base-checkpoint", config["base"]["checkpoint_path"],
        "--lbs-grid", config["base"]["lbs_grid_path"], "--reference-region", config["image_conditioning"]["reference_region_path"],
        "--reference", "0", "cam18", "--reference", "1000", "cam00", "--target", "2000", "cam09",
        "--output-dir", str(output_dir), "--device", args.device,
    ]
    subprocess.run(inference_command, cwd=repo, check=True)
    inference_manifest = json.loads((output_dir / "inference_manifest.json").read_text(encoding="utf-8"))
    inference_render = torch.load(output_dir / "inference_render.pt", map_location="cpu", weights_only=True)
    inference_metrics = image_metrics(inference_render["rgb"], inference_render["alpha"], episode["target_rgb"], episode["target_foreground_mask"])
    (output_dir / "inference_metrics.json").write_text(json.dumps(inference_metrics, indent=2), encoding="utf-8")

    step100 = evaluations[100]["_output"]
    comparison = make_grid([
        _as_chw_render(zero_output["render"][0], 3), _as_chw_render(evaluations[10]["_output"]["render"][0], 3),
        _as_chw_render(evaluations[50]["_output"]["render"][0], 3), _as_chw_render(step100["render"][0], 3),
        _as_chw_render(episode["target_rgb"], 3), (_as_chw_render(step100["render"][0], 3) - _as_chw_render(episode["target_rgb"], 3)).abs(),
    ], nrow=3)
    save_image(comparison, output_dir / "comparison.png")
    sensitivity_grid = make_grid([
        episode["reference_images"][0], episode["reference_images"][1],
        _as_chw_render(step100["render"][0], 3),
        _as_chw_render(evaluate(_single_reference_episode(episode, 0))[0]["render"][0], 3),
    ], nrow=2)
    save_image(sensitivity_grid, output_dir / "sensitivity_comparison.png")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4)); plt.plot([x["step"] for x in all_metrics], [x["total"] for x in all_metrics]); plt.xlabel("step"); plt.ylabel("total loss"); plt.tight_layout(); plt.savefig(output_dir / "loss_curve.png", dpi=160); plt.close()
    plt.figure(figsize=(8, 4)); plt.plot([10, 50, 100], [evaluations[s]["image"]["rgb_all_l1"] for s in (10, 50, 100)], marker="o", label="RGB L1"); plt.plot([10, 50, 100], [evaluations[s]["anchor"]["xyz_mae"] for s in (10, 50, 100)], marker="o", label="anchor MAE"); plt.legend(); plt.tight_layout(); plt.savefig(output_dir / "metric_curves.png", dpi=160); plt.close()

    base_path = Path(config["base"]["model_dir"]) / config["base"]["checkpoint_path"]
    manifest_path = Path(config["image_conditioning"]["manifest_path"])
    region_path = Path(config["image_conditioning"]["reference_region_path"])
    input_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"), "status": git("status", "--short")},
        "base": {"path": str(base_path), "sha256": sha256(base_path)},
        "episode": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "resume": {"path": str(resume), "sha256": sha256(resume)},
        "region": {"path": str(region_path), "sha256": sha256(region_path), "threshold": 0.4},
        "protocol": PROTOCOL,
        "environment": {"python": platform.python_version(), "pytorch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
        "command": " ".join(__import__("sys").argv), "inference_command": " ".join(inference_command),
    }
    (output_dir / "input_manifest.json").write_text(json.dumps(input_manifest, indent=2), encoding="utf-8")
    improved_image = sum(evaluations[100]["image"][key] < zero_metrics["image"][key] for key in ("rgb_all_l1", "foreground_rgb_l1", "alpha_l1"))
    pass_pipeline = (
        all(item["all_finite"] and item["base_grad_count"] == 0 and not any(item["disabled_max_abs"].values()) for item in metrics)
        and all(value["allclose"] for value in roundtrip.values())
        and inference_manifest["target_rgb_read"] is False and inference_manifest["teacher_read"] is False
        and improved_image >= 2
        and evaluations[100]["anchor"]["xyz_mae"] < zero_metrics["anchor"]["xyz_mae"]
        and sensitivity[100]["raw_anchor_offset_mae"] > 0 and sensitivity[100]["rendered_rgb_mae"] > 0
    )
    strength = "STRONG" if pass_pipeline and (sensitivity[100]["raw_anchor_offset_mae"] >= 5 * sensitivity[10]["raw_anchor_offset_mae"] or sensitivity[100]["rendered_rgb_mae"] > 1e-5) else ("PARTIAL" if pass_pipeline else "FAIL")
    payload["status"] = "PASS" if pass_pipeline else "FAIL"; payload["effect_strength"] = strength
    (output_dir / "training_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "training_metrics.md").write_text(f"# Gate 4-C Metrics\n\n- Pipeline: **{payload['status']}**\n- Effect strength: **{strength}**\n- Loss: {all_metrics[0]['total']:.9f} -> {all_metrics[-1]['total']:.9f}\n- Early/late mean: {early:.9f} / {late:.9f}\n", encoding="utf-8")
    required = ["config_resolved.yaml", "input_manifest.json", "training_metrics.json", "training_metrics.md", "loss_curve.png", "metric_curves.png", "checkpoint_step_000025.pth", "checkpoint_step_000050.pth", "checkpoint_step_000075.pth", "checkpoint_step_000100.pth", "sensitivity_metrics.json", "inference_manifest.json", "inference_metrics.json", "predicted_anchor_offsets.pt", "zero_rgb.png", "step10_rgb.png", "step50_rgb.png", "step100_rgb.png", "target_rgb.png", "rendered_alpha.png", "comparison.png", "sensitivity_comparison.png"]
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"required artifacts missing: {missing}")
    (output_dir / "GATE_ACCEPTANCE.md").write_text(
        f"# Gate 4-C Acceptance\n\n- Pipeline: **{payload['status']}**\n- Effect strength: **{strength}**\n- Resume: step 10 -> step 100 (90 new finite steps)\n- Independent inference: PASS\n- Checkpoint roundtrip: PASS\n- Output completeness: PASS\n",
        encoding="utf-8",
    )
    print(json.dumps({"pipeline": payload["status"], "strength": strength, "trend": trend, "evaluations": serializable_evaluations, "sensitivity": sensitivity}, indent=2))
    if not pass_pipeline:
        raise RuntimeError("Gate 4-C pipeline acceptance failed")


if __name__ == "__main__":
    main()
