from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid, save_image

import train_dressable as training
from scene.dressable_dataset import ImageConditionedEpisodeDataset
from tools.check_real_image_conditioned_one_batch import (
    REQUESTED_REFERENCES,
    REQUESTED_TARGET,
    _as_chw_render,
    _build_fixed_episode,
    _comparison,
    _single_reference_episode,
    save_render_tensor,
)
from utils.dressable_checkpoint_utils import iter_base_parameters


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    root = Path(__file__).resolve().parents[1]
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def grad_norm(parameters) -> float:
    values = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gate 4-B real 10-step smoke test")
    parser.add_argument("--config", default="configs/canon_dress_gs_gate4b_smoke10.yaml")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = training.load_config(args.config)
    if config["train"]["mode"] != "image_rendering_finetune" or int(config["train"]["steps"]) != 10:
        raise ValueError("Gate 4-B requires image_rendering_finetune with exactly 10 steps")
    expected_protocol = {
        "condition_a_references": [
            {"frame_id": "0", "camera_id": "cam18"},
            {"frame_id": "1000", "camera_id": "cam00"},
        ],
        "condition_b_references": [{"frame_id": "0", "camera_id": "cam18"}],
        "target": {"frame_id": "2000", "camera_id": "cam09"},
    }
    if config["image_conditioning"].get("fixed_protocol") != expected_protocol:
        raise ValueError("Gate 4-B config does not match the preregistered fixed protocol")
    if config["image_conditioning"].get("target_view_used") is not False:
        raise ValueError("Gate 4-B requires target_view_used=false")
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
                list(state["losses"].items())
                + [("raw_delta_xyz", raw["delta_xyz"])]
                + [(f"anchor_{key}", value) for key, value in primary["anchor_offsets"].items()]
                + [(f"gaussian_{key}", value) for key, value in primary["gaussian_offsets"].items()]
                + [("render_rgb", primary["render"][0]), ("render_alpha", primary["render"][1])],
            )
            record = {
                "step": len(metrics) + 1,
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
                "base_grad_count": sum(parameter.grad is not None for parameter in iter_base_parameters(model.dressable_model.base_model)),
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
            if record["base_grad_count"] != 0 or any(record["disabled_max_abs"].values()):
                raise RuntimeError(f"freeze/disabled-channel invariant failed: {record}")
            metrics.append(record)
            if record["step"] in (1, 10):
                save_render_tensor(output_dir / f"step_{record['step']:06d}_prediction.png", primary["render"][0], 3)
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
    if len(metrics) != 10 or result["step"] != 10:
        raise RuntimeError("10-step smoke training did not complete exactly 10 steps")

    model, optimizer, dataset = result["model"], result["optimizer"], result["dataset"]
    episode = _build_fixed_episode(dataset)
    episode["reference_only_gate"] = result["reference_only_gate"]
    base_fingerprint = getattr(model.dressable_model.base_model, "_dressable_base_fingerprint")

    def evaluate(fixed_episode):
        model.eval()
        with torch.no_grad():
            losses, outputs = original_compute(
                model, fixed_episode, result["anchor_edges"], config, args.device,
                render_target=True, deformation_adapter=result["deformation_adapter"],
            )
            output = outputs["primary"]
            raw = model.dressable_model.compute_film_anchor_offsets(
                clothing_embedding=output["global_clothing_embedding"],
                anchor_clothing_features=output["anchor_clothing_features"],
            )
        return losses, output, raw

    _, pre_output, pre_raw = evaluate(episode)
    checkpoint = output_dir / "checkpoint_step_000010.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError("checkpoint_step_000010.pth is missing")
    restored_step = training.load_image_training_checkpoint(
        checkpoint, model, optimizer, config, dataset, base_fingerprint
    )
    _, post_output, post_raw = evaluate(episode)
    roundtrip = {
        "raw_anchor": _comparison(pre_raw["delta_xyz"], post_raw["delta_xyz"], atol=1e-7, rtol=1e-6),
        "gated_anchor": _comparison(pre_output["anchor_offsets"]["delta_xyz"], post_output["anchor_offsets"]["delta_xyz"], atol=1e-7, rtol=1e-6),
        "rendered_rgb": _comparison(pre_output["render"][0], post_output["render"][0], atol=1e-6, rtol=1e-5),
        "rendered_alpha": _comparison(pre_output["render"][1], post_output["render"][1], atol=1e-6, rtol=1e-5),
    }
    if restored_step != 10 or not all(value["allclose"] for value in roundtrip.values()):
        raise RuntimeError(f"step-10 checkpoint roundtrip failed: {roundtrip}")

    condition_b = _single_reference_episode(episode, 0)
    _, output_b, raw_b = evaluate(condition_b)
    sensitivity = {
        "global_embedding_l2": float(torch.linalg.vector_norm(post_output["global_clothing_embedding"] - output_b["global_clothing_embedding"]).item()),
        "anchor_feature_mae": float((post_output["anchor_clothing_features"] - output_b["anchor_clothing_features"]).abs().mean().item()),
        "raw_anchor_delta_xyz_mae": float((post_raw["delta_xyz"] - raw_b["delta_xyz"]).abs().mean().item()),
        "gated_anchor_delta_xyz_mae": float((post_output["anchor_offsets"]["delta_xyz"] - output_b["anchor_offsets"]["delta_xyz"]).abs().mean().item()),
        "gaussian_delta_xyz_mae": float((post_output["gaussian_offsets"]["delta_xyz"] - output_b["gaussian_offsets"]["delta_xyz"]).abs().mean().item()),
        "rendered_rgb_mae": float((post_output["render"][0] - output_b["render"][0]).abs().mean().item()),
        "rendered_alpha_mae": float((post_output["render"][1] - output_b["render"][1]).abs().mean().item()),
        "condition_a": [["0", "cam18"], ["1000", "cam00"]],
        "condition_b": [["0", "cam18"]],
        "target": ["2000", "cam09"],
        "target_view_used": False,
    }

    memory = [item["gpu_memory_allocated"] for item in metrics]
    memory_growth = max(memory[-3:]) - min(memory[-3:])
    pass_status = (
        all(item["all_finite"] and item["base_grad_count"] == 0 for item in metrics)
        and all(not any(item["disabled_max_abs"].values()) for item in metrics)
        and sensitivity["raw_anchor_delta_xyz_mae"] > 0
        and sensitivity["rendered_rgb_mae"] > 0
        and memory_growth <= 64 * 1024 * 1024
    )
    metrics_payload = {"status": "PASS" if pass_status else "FAIL", "steps": metrics, "memory_last3_growth_bytes": memory_growth}
    (output_dir / "training_metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    (output_dir / "sensitivity_metrics.json").write_text(json.dumps({"sensitivity": sensitivity, "checkpoint_roundtrip": roundtrip}, indent=2), encoding="utf-8")
    (output_dir / "training_metrics.md").write_text(
        f"# Gate 4-B Training Metrics\n\nStatus: **{metrics_payload['status']}**\n\n"
        f"- Total loss: {metrics[0]['total']:.9f} → {metrics[-1]['total']:.9f}\n"
        f"- Raw offset sensitivity: {sensitivity['raw_anchor_delta_xyz_mae']:.12g}\n"
        f"- Rendered RGB sensitivity: {sensitivity['rendered_rgb_mae']:.12g}\n",
        encoding="utf-8",
    )

    target = _as_chw_render(episode["target_rgb"], 3)
    pred_a = _as_chw_render(post_output["render"][0], 3)
    pred_b = _as_chw_render(output_b["render"][0], 3)
    comparison = make_grid([episode["reference_images"][0].cpu(), episode["reference_images"][1].cpu(), target, pred_a, pred_b, (pred_a - pred_b).abs()], nrow=3)
    save_image(comparison, output_dir / "sensitivity_comparison.png")
    plt.figure(figsize=(7, 4))
    plt.plot([item["step"] for item in metrics], [item["total"] for item in metrics], marker="o", label="total")
    plt.plot([item["step"] for item in metrics], [item["rgb_l1"] for item in metrics], marker=".", label="rgb_l1")
    plt.xlabel("step"); plt.ylabel("loss"); plt.legend(); plt.tight_layout(); plt.savefig(output_dir / "loss_curve.png", dpi=160); plt.close()

    checkpoint_path = Path(config["base"]["model_dir"]) / config["base"]["checkpoint_path"]
    manifest_path = Path(config["image_conditioning"]["manifest_path"])
    input_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": {"branch": git("branch", "--show-current"), "commit": git("rev-parse", "HEAD"), "status": git("status", "--short")},
        "environment": {"python": platform.python_version(), "pytorch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
        "backbone": {"path": str(checkpoint_path), "sha256": sha256(checkpoint_path)},
        "manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "protocol": {"condition_a": sensitivity["condition_a"], "condition_b": sensitivity["condition_b"], "target": sensitivity["target"], "target_view_used": False},
        "config": config,
    }
    (output_dir / "input_manifest.json").write_text(json.dumps(input_manifest, indent=2), encoding="utf-8")
    acceptance = (
        f"# Gate 4-B Acceptance\n\nStatus: **{'PASS' if pass_status else 'FAIL'}**\n\n"
        f"- Git commit: `{input_manifest['git']['commit']}`\n"
        f"- Backbone SHA256: `{input_manifest['backbone']['sha256']}`\n"
        f"- Manifest SHA256: `{input_manifest['manifest']['sha256']}`\n"
        f"- Steps: 10\n- Total loss: {metrics[0]['total']:.9f} → {metrics[-1]['total']:.9f}\n"
        f"- Raw offset sensitivity MAE: {sensitivity['raw_anchor_delta_xyz_mae']:.12g}\n"
        f"- Rendered RGB sensitivity MAE: {sensitivity['rendered_rgb_mae']:.12g}\n"
        f"- Base gradient count: 0\n- Memory last-3 growth: {memory_growth} bytes\n"
        f"- Checkpoint roundtrip: {'PASS' if all(v['allclose'] for v in roundtrip.values()) else 'FAIL'}\n"
    )
    (output_dir / "GATE_ACCEPTANCE.md").write_text(acceptance, encoding="utf-8")
    print(json.dumps({"status": metrics_payload["status"], "sensitivity": sensitivity, "roundtrip": roundtrip}, indent=2))
    if not pass_status:
        raise RuntimeError("Gate 4-B acceptance conditions were not met")


if __name__ == "__main__":
    main()
