from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml

import train_dressable as training
from scene.full_attribute_oracle import AnchorResidualOracle, GaussianResidualOracle
from scene.oracle_outfit_dataset import OracleOutfitDataset, ResumableDeterministicSampler
from utils.dressable_camera_utils import build_mmlphuman_camera
from utils.full_training_checkpoint_utils import load_full_training_checkpoint, save_full_training_checkpoint
from utils.mmlphuman_state_utils import mmlphuman_state_transaction
from utils.oracle_loss_utils import LPIPSLoss, OracleLossWeights, mask_iou, oracle_rendering_loss


def _chw(value: torch.Tensor, channels: int) -> torch.Tensor:
    if value.ndim == 3 and value.shape[0] == channels:
        return value
    if value.ndim == 3 and value.shape[-1] == channels:
        return value.permute(2, 0, 1).contiguous()
    raise ValueError(f"render tensor must be CHW or HWC with {channels} channels")


def _build_oracle(kind: str, base: Any, config: dict[str, Any], device: torch.device):
    common = {
        "bounds": config["bounds"],
        "initial_gate_probability": config["initial_gate_probability"],
        "enable_shn": bool(config.get("enable_shn", False)),
    }
    if kind == "gaussian":
        edges = _gaussian_edges(base, device)
        return GaussianResidualOracle(base, graph_edges=edges, **common)
    anchors = getattr(base, "xyz_vt", None)
    indices, weights = getattr(base, "nbr_gs", None), getattr(base, "nbr_gs_invdist", None)
    if not all(isinstance(value, torch.Tensor) for value in (anchors, indices, weights)):
        raise ValueError("anchor oracle requires base.xyz_vt, base.nbr_gs and base.nbr_gs_invdist")
    weights = weights.to(device=device, dtype=base._xyz.dtype)
    weights = weights / weights.sum(dim=1, keepdim=True)
    edges = training.build_anchor_knn_edges(anchors.to(device), 4)
    return AnchorResidualOracle(
        base, int(anchors.shape[0]), indices.to(device=device, dtype=torch.long), weights,
        graph_edges=edges, **common,
    )


def _gaussian_edges(base: Any, device: torch.device) -> torch.Tensor | None:
    neighbors = getattr(base, "nbr_vt", None)
    if not isinstance(neighbors, torch.Tensor) or neighbors.ndim != 2:
        return None
    rows = torch.arange(neighbors.shape[0], device=device).unsqueeze(1).expand_as(neighbors)
    edges = torch.stack((rows.reshape(-1), neighbors.to(device=device, dtype=torch.long).reshape(-1)), dim=1)
    return edges[(edges[:, 1] >= 0) & (edges[:, 1] < neighbors.shape[0])]


def _render(base: Any, sample: dict[str, Any], overrides, background: torch.Tensor):
    height, width = sample["target_rgb"].shape[-2:]
    camera = build_mmlphuman_camera(sample["target_camera"], height, width, base._xyz.device)
    with mmlphuman_state_transaction(
        base, sample["target_pose"].to(base._xyz.device),
        sample["target_Rh"].to(base._xyz.device), sample["target_Th"].to(base._xyz.device),
    ):
        return base.render(
            camera, background=background,
            canonical_overrides=None if overrides is None else overrides.as_dict(),
        )


def _evaluate(base, oracle, dataset, background, weights, lpips, device) -> dict[str, float]:
    totals: dict[str, float] = {}
    oracle.eval()
    with torch.no_grad():
        output = oracle(base)
        for sample in dataset:
            prediction = _render(base, sample, output.canonical_overrides, background)
            base_render = _render(base, sample, None, background)
            parts = oracle_rendering_loss(
                prediction_rgb=prediction[0], prediction_alpha=prediction[1],
                target_rgb=sample["target_rgb"].to(device),
                target_foreground_mask=sample["target_foreground_mask"].to(device),
                target_clothing_mask=sample["target_clothing_mask"].to(device),
                base_rgb=base_render[0], base_alpha=base_render[1],
                oracle_output=output, weights=weights, lpips_loss=lpips,
            )
            values = {name: float(value.detach()) for name, value in parts.items()}
            values["mask_iou"] = float(mask_iou(prediction[1], sample["target_foreground_mask"].to(device)))
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + value
    return {name: value / len(dataset) for name, value in totals.items()}


def run(args: argparse.Namespace) -> None:
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    oracle_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    pipeline_config = training.load_config(args.pipeline_config)
    oracle_config["enable_shn"] = bool(args.enable_shn)
    oracle_config["configured_sh_degree"] = 1 if args.enable_shn else 0
    oracle_config["effective_sh_degree"] = 1 if args.enable_shn else 0
    (output_dir / "config_resolved.yaml").write_text(yaml.safe_dump(oracle_config, sort_keys=False), encoding="utf-8")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    base = training.load_frozen_mmlphuman_base(
        pipeline_config["base"]["model_dir"], pipeline_config["base"]["checkpoint_path"], device,
    )
    oracle = _build_oracle(args.oracle_type, base, oracle_config, device).to(device)
    train_set = OracleOutfitDataset(args.manifest, args.outfit_id, args.split, "train")
    validation_set = OracleOutfitDataset(args.manifest, args.outfit_id, args.split, "validation")
    test_set = OracleOutfitDataset(args.manifest, args.outfit_id, args.split, "test")
    sampler = ResumableDeterministicSampler(len(train_set), args.seed)
    learning_rates = {
        "geometry_residuals": oracle_config["optimizer"]["geometry_residuals_lr"],
        "appearance_residuals": oracle_config["optimizer"]["appearance_residuals_lr"],
        "geometry_gate": oracle_config["optimizer"]["geometry_gate_lr"],
        "appearance_gate": oracle_config["optimizer"]["appearance_gate_lr"],
    }
    groups, group_names = oracle.parameter_groups(learning_rates)
    optimizer = torch.optim.Adam(groups)
    weights = OracleLossWeights.from_mapping(oracle_config["loss"])
    lpips = LPIPSLoss().to(device) if weights.lpips else None
    background = torch.tensor(pipeline_config["render"]["background"], device=device, dtype=base._xyz.dtype)
    method_state = {
        "oracle_type": oracle.oracle_type, "oracle_kind": args.oracle_type,
        "bounds": oracle_config["bounds"], "enable_shn": args.enable_shn,
    }
    data_state = {
        "manifest": str(args.manifest.resolve()), "outfit_id": args.outfit_id,
        "split_fingerprint": train_set.split_fingerprint,
    }
    global_step = 0
    if args.resume:
        resume_metadata = torch.load(args.resume, map_location="cpu", weights_only=False)
        oracle.configure_stage(int(resume_metadata["training_state"]["stage"]))
        checkpoint = load_full_training_checkpoint(
            args.resume, model=oracle, optimizer=optimizer, optimizer_group_names=group_names,
            scheduler=None, scaler=None, expected_method_state=method_state, expected_data_state=data_state,
        )
        global_step = int(checkpoint["training_state"]["global_step"])
        sampler.load_state_dict(checkpoint["training_state"]["sampler_state"])
    if args.eval_only:
        metrics = {"validation": _evaluate(base, oracle, validation_set, background, weights, lpips, device)}
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return
    base_metrics = _evaluate(base, oracle, validation_set, background, weights, lpips, device)
    best_metric, best_step = float("inf"), global_step
    log_path = output_dir / "training_log.jsonl"
    schedule = oracle_config["schedule"]
    if args.stage is not None:
        schedule = [item for item in schedule if int(item["stage"]) == args.stage]
    start_time = time.perf_counter()
    while global_step < args.max_steps:
        progressed = False
        for stage_item in schedule:
            stage = int(stage_item["stage"])
            if stage == 0:
                continue
            oracle.configure_stage(stage)
            for _ in range(int(stage_item["steps"])):
                if global_step >= args.max_steps:
                    break
                for index in sampler:
                    sample = train_set[index]
                    oracle.train()
                    optimizer.zero_grad(set_to_none=True)
                    output = oracle(base)
                    prediction = _render(base, sample, output.canonical_overrides, background)
                    with torch.no_grad():
                        base_render = _render(base, sample, None, background)
                    parts = oracle_rendering_loss(
                        prediction_rgb=prediction[0], prediction_alpha=prediction[1],
                        target_rgb=sample["target_rgb"].to(device),
                        target_foreground_mask=sample["target_foreground_mask"].to(device),
                        target_clothing_mask=sample["target_clothing_mask"].to(device),
                        base_rgb=base_render[0], base_alpha=base_render[1],
                        oracle_output=output, weights=weights, lpips_loss=lpips,
                    )
                    if not torch.isfinite(parts["total"]):
                        raise FloatingPointError("oracle total loss is NaN or Inf")
                    parts["total"].backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        oracle.parameters(), float(oracle_config["optimizer"]["gradient_clip_norm"]),
                    )
                    if not torch.isfinite(grad_norm):
                        raise FloatingPointError("oracle gradient norm is NaN or Inf")
                    optimizer.step()
                    global_step += 1
                    record = {
                        "step": global_step, "stage": stage, "condition_id": sample["condition_id"],
                        "gradient_norm": float(grad_norm), "elapsed_seconds": time.perf_counter() - start_time,
                        "losses": {name: float(value.detach()) for name, value in parts.items()},
                    }
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record) + "\n")
                    progressed = True
                    if global_step >= args.max_steps:
                        break
                validation = _evaluate(base, oracle, validation_set, background, weights, lpips, device)
                if validation["total"] < best_metric:
                    best_metric, best_step = validation["total"], global_step
                    _save(output_dir / "best_checkpoint.pth", oracle, optimizer, group_names, global_step, sampler, method_state, data_state, best_metric)
                _save(output_dir / "last_checkpoint.pth", oracle, optimizer, group_names, global_step, sampler, method_state, data_state, best_metric)
            if global_step >= args.max_steps:
                break
        if not progressed:
            break
    metrics = {
        "oracle_type": oracle.oracle_type, "oracle_kind": args.oracle_type,
        "base_validation": base_metrics,
        "validation": _evaluate(base, oracle, validation_set, background, weights, lpips, device),
        "test": _evaluate(base, oracle, test_set, background, weights, lpips, device),
        "best_validation_step": best_step, "trainable_parameter_count": sum(p.numel() for p in oracle.parameters()),
        "global_step": global_step, "total_training_seconds": time.perf_counter() - start_time,
        "no_image_conditioned_modules_used": True, "shared_canonical_residual": True,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _save(path, oracle, optimizer, group_names, step, sampler, method_state, data_state, best_metric):
    save_full_training_checkpoint(
        path, model=oracle, optimizer=optimizer, optimizer_group_names=group_names,
        scheduler=None, scaler=None,
        training_state={
            "global_step": step, "optimizer_step": step, "epoch": sampler.epoch,
            "batch_index": sampler.position, "gradient_accumulation_position": 0,
            "best_metric": best_metric, "sampler_state": sampler.state_dict(), "stage": oracle.stage,
        },
        data_state=data_state, method_state=method_state,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--outfit-id", required=True)
    parser.add_argument("--oracle-type", required=True, choices=("gaussian", "anchor"))
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/oracle/full_attribute_oracle_v1.yaml"))
    parser.add_argument("--pipeline-config", type=Path, default=Path("configs/canon_dress_gs_mvp_real.yaml"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stage", type=int, choices=(0, 1, 2, 3, 4))
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--render-protocol", default="full_dataset_target")
    parser.add_argument("--enable-shn", action="store_true")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
