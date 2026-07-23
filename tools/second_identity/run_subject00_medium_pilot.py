#!/usr/bin/env python3
"""Run the frozen subject00 one-pass medium pilot.

This driver deliberately reuses the repaired short-canary implementation for
model construction, optimizer auditing, metrics, and visualization.  It adds
the medium-pilot data order, exact canary-step0 restore, formal LPIPS schedule,
four newly written checkpoints, and a fresh-process final roundtrip.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torchmetrics.functional.image import structural_similarity_index_measure

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.second_identity import run_subject00_short_canary as core
from utils.image_utils import crop_image
from utils.loss_utils import (
    dxyz_smooth_loss,
    gaussian_scaling_loss,
    l1_loss,
    lpips_loss,
)
from utils.smpl_utils import init_smpl
from utils.surface_lbs_utils import SURFACE_LBS_FILES, sha256_file


TASK_ID = "MMLPHUMAN-SUBJECT00-ONE-PASS-MEDIUM-PILOT-001"
SOURCE_BRANCH = (
    "research/mmlphuman-subject00-short-canary-from-repaired-contract-20260723"
)
SOURCE_HEAD = "8c69cdce0139532a33b1d4f839da7467784270e6"
CANARY_EXECUTION_HEAD = "b0e8096589fe18069d95d2137e1db3979b4fe89f"
TARGET_BRANCH = (
    "research/mmlphuman-subject00-one-pass-medium-pilot-20260723"
)
AVAILABILITY_SHA = (
    "cd00ad0a1549bc99d956e26eea300945adfaa10a20a438e931e640977475564e"
)
TRAIN_ORDER_SHA = (
    "0f0e7463d9f9066f54e19ec31f85f722afd58e58aba123af144918fdc97639f8"
)
EVAL_ORDER_SHA = (
    "38476b7d9f6a012e0e21d7f188c5db5131506ec2231e8a7b50a389c7c92ad98a"
)
CAMERA_SPLIT_SHA = (
    "8827cdc05a08cb7068e1f5e89ab7ba5384bf83b89686e8c9f80024b78d49cf44"
)
POSE_SPLIT_SHA = (
    "c12e5eec87c737d3740b6e6199d657c859df7ce1ec9d371ea8d4a987ef62bf06"
)
DERIVED_MANIFEST_SHA = (
    "de6cd51fe3f81e29139b45494860b186038b75a378b101e7428252b3b66c1af8"
)
CANARY_STEP0_PATH = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-MMLPHUMAN-SHORT-CANARY-001/attempt_001/"
    "checkpoints/step_000000.pth"
)
CANARY_STEP0_SHA = (
    "29d0edb11474f4384b6d409141d17f5da91c98824bd070196d50804cd7cb480a"
)
CANARY_STEP0_BYTES = 701_938_720
CANARY_STEP0_PAYLOAD_SOURCE_HEAD = (
    "55cb5a28b8ff0d6a3373582759b8704df2267331"
)
CANARY_SNAPSHOT_PATH = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-MMLPHUMAN-SHORT-CANARY-001/attempt_001/"
    "snapshots/pre_result_snapshot.json"
)
N_VALID = 20_249
CHECKPOINT_STEPS = (0, 5_062, 10_124, 15_186, 20_249)
DIAGNOSTIC_STEPS = CHECKPOINT_STEPS[1:-1]
FINAL_STEP = CHECKPOINT_STEPS[-1]
DIAGNOSTIC_ORDINALS = (1, 24, 25, 48, 49, 72, 73, 96)


def write_json(path: Path, value: object) -> None:
    core.write_json(path, value)


def canonical_sha(value: object) -> str:
    return core.canonical_sha(value)


def configure_core() -> None:
    core.TASK_ID = TASK_ID
    core.SOURCE_BRANCH = SOURCE_BRANCH
    core.SOURCE_HEAD = SOURCE_HEAD
    core.TARGET_BRANCH = TARGET_BRANCH
    core.AVAILABILITY_SHA = AVAILABILITY_SHA
    core.TRAIN_ORDER_SHA = TRAIN_ORDER_SHA
    core.EVAL_ORDER_SHA = EVAL_ORDER_SHA
    core.DERIVED_MANIFEST_SHA = DERIVED_MANIFEST_SHA
    core.CAMERA_SPLIT_SHA = CAMERA_SPLIT_SHA
    core.POSE_SPLIT_SHA = POSE_SPLIT_SHA
    core.CHECKPOINT_STEPS = CHECKPOINT_STEPS
    core.DIAGNOSTIC_STEPS = DIAGNOSTIC_STEPS
    core.FORMAL_EVAL_STEPS = (0, FINAL_STEP)


def load_bundle(availability_path: Path) -> dict[str, Any]:
    contract_path = (
        REPO_ROOT
        / "paper_protocol/second_identity/"
        "subject00_medium_pilot_execution_contract.json"
    )
    records_path = (
        REPO_ROOT
        / "paper_protocol/second_identity/"
        "subject00_medium_pilot_record_manifest.json"
    )
    schedule_path = (
        REPO_ROOT
        / "paper_protocol/second_identity/"
        "subject00_medium_pilot_schedule.json"
    )
    evaluation_path = (
        REPO_ROOT
        / "paper_protocol/second_identity/"
        "subject00_canary_evaluation_manifest.json"
    )
    repaired_config_path = (
        REPO_ROOT / "config/subject00_surface_lbs_short_canary_repaired.yaml"
    )
    repaired_contract_path = (
        REPO_ROOT
        / "paper_protocol/second_identity/"
        "subject00_canary_training_contract_repaired.json"
    )
    availability_contract_path = (
        REPO_ROOT
        / "paper_protocol/second_identity/"
        "subject00_canary_evaluation_availability.json"
    )
    repair_summary_path = (
        REPO_ROOT
        / "paper_protocol/second_identity/"
        "subject00_canary_contract_repair_summary.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    records = json.loads(records_path.read_text(encoding="utf-8"))
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    availability = json.loads(availability_path.read_text(encoding="utf-8"))
    repaired_contract = json.loads(
        repaired_contract_path.read_text(encoding="utf-8")
    )
    availability_contract = json.loads(
        availability_contract_path.read_text(encoding="utf-8")
    )
    repair_summary = json.loads(
        repair_summary_path.read_text(encoding="utf-8")
    )
    hashes = {
        "execution_contract_file_sha256": sha256_file(contract_path),
        "record_manifest_file_sha256": sha256_file(records_path),
        "schedule_file_sha256": sha256_file(schedule_path),
        "repaired_config_file_sha256": sha256_file(repaired_config_path),
        "training_contract_file_sha256": sha256_file(repaired_contract_path),
        "evaluation_availability_file_sha256": sha256_file(
            availability_contract_path
        ),
        "evaluation_manifest_file_sha256": sha256_file(evaluation_path),
        "contract_repair_summary_file_sha256": sha256_file(
            repair_summary_path
        ),
        "availability_canonical_sha256": canonical_sha(
            availability["entries"]
        ),
        "training_data_order_sha256": canonical_sha(records["records"]),
        "evaluation_query_order_sha256": canonical_sha(
            evaluation["queries"]
        ),
    }
    expected = {
        "availability_canonical_sha256": AVAILABILITY_SHA,
        "training_data_order_sha256": TRAIN_ORDER_SHA,
        "evaluation_query_order_sha256": EVAL_ORDER_SHA,
    }
    for key, value in expected.items():
        if hashes[key] != value:
            raise RuntimeError(
                "SUBJECT00_MEDIUM_PILOT_REPAIRED_CONTRACT_HASH_MISMATCH: "
                f"{key}={hashes[key]} expected={value}"
            )
    if len(records["records"]) != N_VALID:
        raise RuntimeError("medium record manifest count changed")
    if len(evaluation["queries"]) != 96:
        raise RuntimeError("fixed evaluation query count changed")
    if schedule["checkpoint_steps"] != list(CHECKPOINT_STEPS):
        raise RuntimeError("medium checkpoint schedule changed")
    if len(schedule["steps"]) != N_VALID:
        raise RuntimeError("medium training schedule count changed")
    if (
        contract["source"]["head"] != SOURCE_HEAD
        or contract["canary_execution_code_head"] != CANARY_EXECUTION_HEAD
    ):
        raise RuntimeError("medium source provenance changed")
    if contract["subject02_formal_training_contract"] != repaired_contract[
        "subject02_formal_training_contract"
    ]:
        raise RuntimeError("subject02 optimizer/loss/scheduler contract changed")
    if len(contract["subject02_formal_training_contract"]["optimizer"]["groups"]) != 14:
        raise RuntimeError("optimizer contract is not 14 groups")
    valid_pairs = {
        (int(item["frame_id"]), int(item["camera_id"]))
        for item in availability["entries"]
        if item["image_available"]
        and item["mask_available"]
        and item["valid_pair"]
    }
    record_pairs = [
        (int(item["pose_id"]), int(item["camera_id"]))
        for item in records["records"]
    ]
    if len(record_pairs) != len(set(record_pairs)):
        raise RuntimeError("duplicate medium record exposure")
    if not set(record_pairs).issubset(valid_pairs):
        raise RuntimeError("invalid medium training record present")
    train_poses = set(records["pose_selection"]["pose_ids"])
    train_cameras = set(records["camera_ids"])
    if train_poses & set(contract["strict_splits"]["heldout_pose_ids"]):
        raise RuntimeError("held-out pose leakage")
    if train_poses & set(contract["strict_splits"]["buffer_pose_ids"]):
        raise RuntimeError("buffer pose leakage")
    if train_cameras & set(contract["strict_splits"]["heldout_camera_ids"]):
        raise RuntimeError("held-out camera leakage")
    return {
        "contract": contract,
        "records": records,
        "schedule": schedule,
        "evaluation": evaluation,
        "availability_contract": availability_contract,
        "repair_summary": repair_summary,
        "repaired_config": OmegaConf.load(repaired_config_path),
        "hashes": hashes,
        "valid_pairs": valid_pairs,
    }


def color_distribution_metrics(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    mask: np.ndarray,
    pred_mask: np.ndarray,
) -> dict[str, float]:
    gt_pixels = ground_truth[mask]
    pred_pixels = predicted[mask]
    if len(pred_pixels):
        variance = float(np.var(pred_pixels, axis=0).mean())
        chroma = float(
            np.mean(np.max(pred_pixels, axis=1) - np.min(pred_pixels, axis=1))
        )
    else:
        variance = 0.0
        chroma = 0.0
    histogram_l1 = 0.0
    if len(gt_pixels) and len(pred_pixels):
        for channel in range(3):
            gt_hist, _ = np.histogram(
                gt_pixels[:, channel], bins=32, range=(0.0, 1.0)
            )
            pred_hist, _ = np.histogram(
                pred_pixels[:, channel], bins=32, range=(0.0, 1.0)
            )
            gt_hist = gt_hist.astype(np.float64) / max(gt_hist.sum(), 1)
            pred_hist = pred_hist.astype(np.float64) / max(pred_hist.sum(), 1)
            histogram_l1 += float(np.abs(gt_hist - pred_hist).sum())
        histogram_l1 /= 6.0
    mask_count = max(int(mask.sum()), 1)
    background_count = max(int((~mask).sum()), 1)
    return {
        "foreground_rgb_variance": variance,
        "foreground_chroma": chroma,
        "gt_pred_color_histogram_l1": histogram_l1,
        "garment_proxy_undercoverage": float(
            (mask & ~pred_mask).sum() / mask_count
        ),
        "garment_proxy_overcoverage": float(
            ((~mask) & pred_mask).sum() / background_count
        ),
    }


@torch.no_grad()
def evaluate_queries(
    model,
    bundle: dict[str, Any],
    data_root: Path,
    attempt_root: Path,
    *,
    step: int,
    queries: list[dict[str, Any]],
    lpips_metric,
    lpips_error: str | None,
    visual_subdir: str,
    smoke_ordinals: set[int] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    poses = sorted({int(item["pose_id"]) for item in queries})
    cameras = sorted({int(item["camera_id"]) for item in queries})
    dataset = core.ThumanDataset(
        datadir=str(data_root),
        frame_ids=poses,
        cam_ids=cameras,
        background=np.ones(3, dtype=np.float32),
        image_scaling=1,
        is_in_memory=False,
    )
    mapping = {
        (int(pose), int(camera)): index
        for index, (pose, camera) in enumerate(dataset.indices)
    }
    expected = {
        (int(item["pose_id"]), int(item["camera_id"])) for item in queries
    }
    if not expected.issubset(mapping):
        raise RuntimeError(
            f"evaluation inputs missing: {sorted(expected - set(mapping))}"
        )
    background = torch.ones(3, dtype=torch.float32, device="cuda")
    records: list[dict[str, Any]] = []
    smoke_arrays: dict[str, np.ndarray] = {}
    all_warnings: list[str] = []
    for query in queries:
        pose_id = int(query["pose_id"])
        camera_id = int(query["camera_id"])
        item = core.prepare_item(dataset[mapping[(pose_id, camera_id)]])
        core.set_model_pose(model, item)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            predicted, alpha, depth, info = model.render(
                item, background=background, return_depth=True
            )
            torch.cuda.synchronize()
        render_seconds = time.perf_counter() - started
        warning_messages = [str(entry.message) for entry in caught]
        all_warnings.extend(warning_messages)
        predicted = torch.clamp(predicted, 0.0, 1.0)
        ground_truth = item["image"].clone()
        ground_truth[~item["mask"]] = background
        ground_truth[item["mask_boundary"]] = background
        predicted_metric = predicted.clone()
        predicted_metric[item["mask_boundary"]] = background
        difference = torch.abs(predicted_metric - ground_truth)
        mse = torch.mean((predicted_metric - ground_truth) ** 2)
        psnr_value = float(
            (-10.0 * torch.log10(torch.clamp(mse, min=1.0e-12))).item()
        )
        ssim_value = float(
            structural_similarity_index_measure(
                predicted_metric.permute(2, 0, 1)[None],
                ground_truth.permute(2, 0, 1)[None],
                data_range=1.0,
            ).item()
        )
        lpips_value = None
        if lpips_metric is not None:
            pred_crop, gt_crop = crop_image(
                background,
                item["mask"],
                512,
                False,
                predicted_metric.permute(2, 0, 1),
                ground_truth.permute(2, 0, 1),
            )
            lpips_value = float(
                lpips_metric(pred_crop[None], gt_crop[None]).item()
            )
        alpha_np = alpha[..., 0].detach().cpu().numpy().astype(np.float32)
        depth_np = depth[..., 0].detach().cpu().numpy().astype(np.float32)
        pred_np = predicted.detach().cpu().numpy().astype(np.float32)
        metric_pred_np = (
            predicted_metric.detach().cpu().numpy().astype(np.float32)
        )
        gt_np = ground_truth.detach().cpu().numpy().astype(np.float32)
        mask_np = item["mask"].detach().cpu().numpy().astype(bool)
        pred_mask = alpha_np >= 0.5
        intersection = int((pred_mask & mask_np).sum())
        union = int((pred_mask | mask_np).sum())
        posed_xyz = model.get_xyz
        extent = posed_xyz.max(dim=0).values - posed_xyz.min(dim=0).values
        means2d = info.get("means2d")
        ordinal = int(query["ordinal"])
        visual_path = (
            attempt_root
            / "visuals"
            / visual_subdir
            / f"query_{ordinal:03d}.png"
        )
        core.save_visual(
            visual_path,
            gt=gt_np,
            mask=mask_np,
            predicted=pred_np,
            alpha=alpha_np,
            depth=depth_np,
            title=(
                f"step={step} quadrant={query['quadrant']} "
                f"pose={pose_id} camera={camera_id}"
            ),
        )
        key = f"query_{ordinal:03d}"
        if smoke_ordinals is not None and ordinal in smoke_ordinals:
            smoke_arrays[f"{key}_rgb"] = pred_np
            smoke_arrays[f"{key}_alpha"] = alpha_np
            smoke_arrays[f"{key}_depth"] = depth_np
        color_metrics = color_distribution_metrics(
            metric_pred_np, gt_np, mask_np, pred_mask
        )
        records.append(
            {
                **query,
                "step": step,
                "visual_path": str(visual_path),
                "rgb_mae": float(difference.mean().item()),
                "psnr": psnr_value,
                "ssim": ssim_value,
                "lpips": lpips_value,
                "silhouette_iou": float(
                    intersection / union if union else 1.0
                ),
                "boundary_f_score": core.boundary_f_score(
                    pred_mask, mask_np
                ),
                "alpha_occupancy": float((alpha_np > 1.0e-4).mean()),
                "depth_finite_ratio": float(np.isfinite(depth_np).mean()),
                "foreground_coverage": float(pred_mask.mean()),
                **color_metrics,
                "render_seconds": render_seconds,
                "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
                "rgb_finite": bool(np.isfinite(metric_pred_np).all()),
                "alpha_finite": bool(np.isfinite(alpha_np).all()),
                "depth_finite": bool(np.isfinite(depth_np).all()),
                "alpha_nonempty": bool((alpha_np > 1.0e-6).any()),
                "mask_nonempty": bool(mask_np.any()),
                "posed_xyz_finite": bool(
                    torch.isfinite(posed_xyz).all().item()
                ),
                "posed_bbox_extent": [
                    float(value) for value in extent.detach().cpu().tolist()
                ],
                "body_explosion": bool(extent.max().item() >= 5.0),
                "means2d_finite": bool(
                    means2d is not None
                    and torch.isfinite(means2d).all().item()
                ),
                "warning_messages": warning_messages,
            }
        )
    aggregate_names = (
        "foreground_rgb_variance",
        "foreground_chroma",
        "gt_pred_color_histogram_l1",
        "garment_proxy_undercoverage",
        "garment_proxy_overcoverage",
    )
    by_quadrant = {}
    for quadrant in bundle["evaluation"]["quadrant_contract_order"]:
        subset = [item for item in records if item["quadrant"] == quadrant]
        metrics = core.aggregate_metrics(subset)
        for name in aggregate_names:
            values = [float(item[name]) for item in subset]
            metrics[name] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        by_quadrant[quadrant] = {
            "query_count": len(subset),
            "metrics": metrics,
        }
    metrics_all = core.aggregate_metrics(records)
    for name in aggregate_names:
        values = [float(item[name]) for item in records]
        metrics_all[name] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    all_finite = all(
        item["rgb_finite"]
        and item["alpha_finite"]
        and item["depth_finite"]
        and item["posed_xyz_finite"]
        and item["means2d_finite"]
        for item in records
    )
    result = {
        "schema_version": "subject00.medium.fixed_evaluation.v1",
        "task_id": TASK_ID,
        "step": step,
        "query_count": len(records),
        "successful_render_count": sum(
            item["rgb_finite"]
            and item["alpha_finite"]
            and item["depth_finite"]
            for item in records
        ),
        "failed_render_count": sum(
            not (
                item["rgb_finite"]
                and item["alpha_finite"]
                and item["depth_finite"]
            )
            for item in records
        ),
        "all_finite": all_finite,
        "alpha_nonempty_count": sum(item["alpha_nonempty"] for item in records),
        "body_explosion_count": sum(item["body_explosion"] for item in records),
        "lpips_available": lpips_metric is not None,
        "lpips_unavailable_reason": lpips_error,
        "primary_image_error": (
            "lpips" if lpips_metric is not None else "rgb_mae"
        ),
        "warning_set": sorted(set(all_warnings)),
        "bin_overflow_warning": any(
            "overflow" in item.lower() for item in all_warnings
        ),
        "representation_metric_contract": {
            "foreground_pixels": "ground-truth human mask",
            "color_histogram": (
                "32 bins per RGB channel; mean normalized L1 divided by 2"
            ),
            "garment_mask_limitation": (
                "No separate garment mask exists; human silhouette is an "
                "explicit garment-coverage proxy."
            ),
        },
        "records": records,
        "quadrants": by_quadrant,
        "metrics_all": metrics_all,
    }
    return result, smoke_arrays


def train_step(model, item: dict[str, Any], args, step: int) -> dict[str, Any]:
    cam = core.prepare_item(item)
    core.set_model_pose(model, cam)
    background = torch.rand(3, device="cuda")
    torch.cuda.synchronize()
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        image, alpha, info = model.render(cam, background=background)
    image = torch.clamp(image, 0.0, 1.0)
    image_gt = cam["image"].clone()
    image_gt[~cam["mask"]] = background
    image_gt[cam["mask_boundary"]] = background
    image[cam["mask_boundary"]] = background
    l1_value = l1_loss(image, image_gt)
    smooth = dxyz_smooth_loss(model) * args.lambda_dxyz_smooth
    scaling = (
        gaussian_scaling_loss(
            model.get_cano_scaling, args.scaling_threshold
        )
        * args.lambda_scaling
    )
    random_patch = step >= int(args.iteration_lpips_random_patch)
    image_crop, image_gt_crop = crop_image(
        background,
        cam["mask"],
        512,
        random_patch,
        image.permute(2, 0, 1),
        image_gt.permute(2, 0, 1),
    )
    if step > int(args.iteration_lpips):
        lpips_value = (
            lpips_loss(
                image_crop.permute(1, 2, 0),
                image_gt_crop.permute(1, 2, 0),
            )
            * args.lambda_lpips
        )
    else:
        lpips_value = torch.zeros((), device="cuda")
    total = l1_value + lpips_value + smooth + scaling
    finite_loss = bool(torch.isfinite(total).item())
    if not finite_loss:
        raise FloatingPointError(f"non-finite loss at step {step}")
    total.backward()
    gradient_norms, parameter_norms = core.group_norms(model)
    gradients_finite = all(
        math.isfinite(value) for value in gradient_norms.values()
    )
    nonzero_groups = sorted(
        name for name, value in gradient_norms.items() if value > 0.0
    )
    if not gradients_finite or not nonzero_groups:
        raise FloatingPointError(
            f"gradient audit failed at step {step}: "
            f"finite={gradients_finite}, nonzero={nonzero_groups}"
        )
    for frozen in (
        model._xyz,
        model.get_weights,
        *model.surface_attachment_state_dict(cpu=False).values(),
    ):
        if frozen.grad is not None:
            raise RuntimeError("frozen attachment/LBS tensor acquired gradient")
    model.optimizer_step()
    if not core.optimizer_state_finite(model):
        raise FloatingPointError(f"optimizer state non-finite at step {step}")
    torch.cuda.synchronize()
    xyz = model.get_xyz.detach()
    opacity = model.compute_opacity().detach()
    scale = model.get_cano_scaling.detach()
    warning_messages = [str(entry.message) for entry in caught]
    means2d = info.get("means2d")
    return {
        "step": step,
        "pose_id": cam["frame_id"],
        "camera_id": cam["cam_id"],
        "record_id": f"pose_{cam['frame_id']:04d}/camera_{cam['cam_id']:02d}",
        "total_loss": float(total.item()),
        "l1_loss": float(l1_value.item()),
        "lpips_loss": float(lpips_value.item()),
        "dxyz_smooth_loss": float(smooth.item()),
        "scaling_loss": float(scaling.item()),
        "learning_rates": {
            name: float(optimizer.param_groups[0]["lr"])
            for name, optimizer in model.optimizers.items()
        },
        "gradient_norms": gradient_norms,
        "parameter_norms": parameter_norms,
        "nonzero_gradient_groups": nonzero_groups,
        "gradient_clipping": "NONE",
        "mixed_precision": False,
        "loss_finite": finite_loss,
        "gradients_finite": gradients_finite,
        "optimizer_state_finite": True,
        "xyz": {
            "min": float(xyz.min().item()),
            "max": float(xyz.max().item()),
            "mean": float(xyz.mean().item()),
        },
        "opacity": {
            "min": float(opacity.min().item()),
            "max": float(opacity.max().item()),
            "mean": float(opacity.mean().item()),
        },
        "scaling": {
            "min": float(scale.min().item()),
            "max": float(scale.max().item()),
            "mean": float(scale.mean().item()),
        },
        "means2d_finite": bool(
            means2d is not None and torch.isfinite(means2d).all().item()
        ),
        "warning_messages": warning_messages,
        "bin_overflow_warning": any(
            "overflow" in message.lower() for message in warning_messages
        ),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "wall_seconds": time.perf_counter() - started,
    }


def save_checkpoint(
    path: Path,
    model,
    *,
    step: int,
    data_order_position: int,
    scene_scale: float,
    execution_source_head: str,
    contract_hashes: dict[str, str],
) -> dict[str, Any]:
    payload = model.capture()
    payload.update(
        {
            "checkpoint_kind": "SUBJECT00_ONE_PASS_MEDIUM_PILOT",
            "checkpoint_version": 1,
            "training_step": step,
            "iteration": step,
            "optimizer_states": {
                name: optimizer.state_dict()
                for name, optimizer in model.optimizers.items()
            },
            "scheduler_states": [
                scheduler.state_dict() for scheduler in model.schedulers
            ],
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all(),
            "data_order_position": data_order_position,
            "scene_scale": float(scene_scale),
            "attachment_provenance": model.surface_attachment_provenance,
            "surface_attachment_state_keys": list(SURFACE_LBS_FILES),
            "template_sha256": (
                "f10a3b516e2b3a2ad38dc4924a3692b2f3e72a6cc9e66f3c0063c4e9cd210031"
            ),
            "sampler_sha256": (
                "98af26a3f578d1e0239f6ea414e3664dae10a4b60f47aef594d0978f61b9ee82"
            ),
            "derived_manifest_sha256": DERIVED_MANIFEST_SHA,
            "camera_split_sha256": CAMERA_SPLIT_SHA,
            "pose_split_sha256": POSE_SPLIT_SHA,
            "source_branch": TARGET_BRANCH,
            "source_head": execution_source_head,
            "contract_hashes": contract_hashes,
            "training_data_order_sha256": TRAIN_ORDER_SHA,
            "evaluation_query_order_sha256": EVAL_ORDER_SHA,
            "initialization_checkpoint": {
                "path": str(CANARY_STEP0_PATH),
                "sha256": CANARY_STEP0_SHA,
            },
            "topology_call_counts": {
                "clone": 0,
                "split": 0,
                "densification": 0,
                "topology_changing_prune": 0,
                "off_surface_rebind": 0,
            },
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {
        "path": str(path),
        "step": step,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "data_order_position": data_order_position,
        "reused": False,
    }


def restore_checkpoint_state(model, runtime_args, scene_scale: float, payload) -> None:
    model.restore(payload)
    model.training_setup(runtime_args, scene_scale)
    if set(payload["optimizer_states"]) != set(model.optimizers):
        raise RuntimeError("optimizer group names changed during restore")
    for name, optimizer in model.optimizers.items():
        optimizer.load_state_dict(payload["optimizer_states"][name])
    if len(payload["scheduler_states"]) != len(model.schedulers):
        raise RuntimeError("scheduler count changed during restore")
    for scheduler, state in zip(model.schedulers, payload["scheduler_states"]):
        scheduler.load_state_dict(state)
    random.setstate(payload["python_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    torch.cuda.set_rng_state_all(
        [value.cpu() for value in payload["cuda_rng_states"]]
    )


def initialization_audit(
    model,
    payload,
    scene_scale: float,
    assets_root: Path,
) -> dict[str, Any]:
    if int(payload["training_step"]) != 0:
        raise RuntimeError("canary initialization checkpoint is not step0")
    if int(payload["data_order_position"]) != 0:
        raise RuntimeError("canary step0 data-order position is not zero")
    if not math.isclose(
        float(payload["scene_scale"]),
        float(scene_scale),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError("canary step0 scene scale changed")
    if payload["source_head"] != CANARY_STEP0_PAYLOAD_SOURCE_HEAD:
        raise RuntimeError("canary step0 payload source provenance changed")
    attachments = core.attachment_inventory(model, assets_root)
    snapshot = json.loads(CANARY_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    inventory = core.parameter_inventory(model)
    if inventory != snapshot["model_inventory"]:
        raise RuntimeError("canary step0 parameter inventory parity failed")
    if attachments != snapshot["attachment_inventory"]:
        raise RuntimeError("canary step0 attachment inventory parity failed")
    optimizer_initial = all(
        len(optimizer.state) == 0 for optimizer in model.optimizers.values()
    )
    scheduler_initial = all(
        int(state.get("last_epoch", -1)) == 0
        for state in payload["scheduler_states"]
    )
    if not optimizer_initial or not scheduler_initial:
        raise RuntimeError("canary step0 optimizer/scheduler is not initial")
    return {
        "status": "PASS",
        "checkpoint_step": 0,
        "data_order_position": 0,
        "checkpoint_sha256": CANARY_STEP0_SHA,
        "checkpoint_bytes": CANARY_STEP0_BYTES,
        "parameter_inventory_exact": True,
        "attachment_inventory_exact": True,
        "optimizer_initial": optimizer_initial,
        "scheduler_initial": scheduler_initial,
        "gaussian_count": int(model._xyz.shape[0]),
        "attachment_count": int(
            model.surface_attachment_state_dict(cpu=True)["face_ids"].shape[0]
        ),
        "lbs_shape": list(model.get_weights.shape),
        "rng_state_present": all(
            key in payload
            for key in (
                "python_rng_state",
                "numpy_rng_state",
                "torch_rng_state",
                "cuda_rng_states",
            )
        ),
        "source_head": payload["source_head"],
    }


def train_phase(args) -> int:
    configure_core()
    if core.git_branch() != TARGET_BRANCH:
        raise RuntimeError(f"unexpected branch: {core.git_branch()}")
    execution_head = core.git_head()
    if execution_head == SOURCE_HEAD:
        raise RuntimeError("pre-result contract commit is absent")
    bundle = load_bundle(args.availability_manifest)
    if sha256_file(CANARY_STEP0_PATH) != CANARY_STEP0_SHA:
        raise RuntimeError(
            "SUBJECT00_MEDIUM_PILOT_REPAIRED_CONTRACT_HASH_MISMATCH: "
            "canary step0 checkpoint"
        )
    if CANARY_STEP0_PATH.stat().st_size != CANARY_STEP0_BYTES:
        raise RuntimeError("canary step0 checkpoint size changed")
    if args.attempt_root.exists() and args.resume_step == 0:
        raise RuntimeError("attempt_001 exists; refusing a second training run")
    for directory in (
        "contract",
        "snapshots",
        "checkpoints",
        "training_logs",
        "evaluations",
        "visuals",
        "audits",
    ):
        (args.attempt_root / directory).mkdir(parents=True, exist_ok=True)
    write_json(
        args.attempt_root / "contract/step0_checkpoint_pointer.json",
        {
            "schema_version": "subject00.medium.step0_pointer.v1",
            "task_id": TASK_ID,
            "reused": True,
            "copied": False,
            "path": str(CANARY_STEP0_PATH),
            "sha256": CANARY_STEP0_SHA,
            "bytes": CANARY_STEP0_BYTES,
            "step": 0,
            "data_order_position": 0,
        },
    )
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model, scene, runtime_args = core.build_model(
        bundle, args.data_root, args.assets_root, args.attempt_root
    )
    training_log_path = (
        args.attempt_root / "training_logs/step_records.jsonl"
    )
    checkpoints = [
        {
            "path": str(CANARY_STEP0_PATH),
            "step": 0,
            "bytes": CANARY_STEP0_BYTES,
            "sha256": CANARY_STEP0_SHA,
            "data_order_position": 0,
            "reused": True,
            "pointer_only": True,
        }
    ]
    training_records: list[dict[str, Any]] = []
    resume_step = int(args.resume_step)
    if resume_step:
        if resume_step not in CHECKPOINT_STEPS[1:-1]:
            raise RuntimeError("resume step is not a frozen checkpoint")
        checkpoint_path = (
            args.attempt_root / f"checkpoints/step_{resume_step:06d}.pth"
        )
        payload = torch.load(checkpoint_path, weights_only=False)
        if (
            int(payload["training_step"]) != resume_step
            or int(payload["data_order_position"]) != resume_step
            or payload["training_data_order_sha256"] != TRAIN_ORDER_SHA
        ):
            raise RuntimeError("resume checkpoint/data-order mismatch")
        restore_checkpoint_state(
            model, runtime_args, scene.scene_scale, payload
        )
        training_records = [
            json.loads(line)
            for line in training_log_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        if len(training_records) != resume_step:
            raise RuntimeError("resume log length differs from checkpoint")
        for index, record in enumerate(training_records):
            expected = bundle["records"]["records"][index]
            if (
                int(record["step"]) != index + 1
                or int(record["pose_id"]) != int(expected["pose_id"])
                or int(record["camera_id"]) != int(expected["camera_id"])
            ):
                raise RuntimeError(f"resume order mismatch at {index}")
        for step in CHECKPOINT_STEPS[1:]:
            if step > resume_step:
                break
            path = args.attempt_root / f"checkpoints/step_{step:06d}.pth"
            checkpoints.append(
                {
                    "path": str(path),
                    "step": step,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "data_order_position": step,
                    "reused": False,
                }
            )
        initialization = json.loads(
            (
                args.attempt_root / "audits/initialization_audit.json"
            ).read_text(encoding="utf-8")
        )
    else:
        payload = torch.load(CANARY_STEP0_PATH, weights_only=False)
        restore_checkpoint_state(
            model, runtime_args, scene.scene_scale, payload
        )
        initialization = initialization_audit(
            model, payload, scene.scene_scale, args.assets_root
        )
        write_json(
            args.attempt_root / "audits/initialization_audit.json",
            initialization,
        )
    initial_attachments = core.attachment_inventory(model, args.assets_root)
    write_json(
        args.attempt_root / "snapshots/initial_attachment_inventory.json",
        initial_attachments,
    )
    optimizer_audit_result = core.optimizer_audit(
        model,
        bundle["contract"],
        scene.scene_scale,
        scheduler_step=resume_step,
    )
    write_json(
        args.attempt_root / "audits/optimizer_membership.json",
        optimizer_audit_result,
    )
    lpips_metric, lpips_error = core.make_lpips_preserving_rng()
    diagnostic = core.diagnostic_queries(bundle["evaluation"])
    diagnostic_reports = {}
    for step in DIAGNOSTIC_STEPS:
        path = (
            args.attempt_root
            / f"evaluations/diagnostic_step_{step:06d}.json"
        )
        if step <= resume_step:
            report = json.loads(path.read_text(encoding="utf-8"))
            diagnostic_reports[str(step)] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "query_count": report["query_count"],
            }
    warning_set: set[str] = {
        message
        for record in training_records
        for message in record["warning_messages"]
    }
    for step, manifest_record in enumerate(
        bundle["records"]["records"], start=1
    ):
        if step <= resume_step:
            continue
        item = scene.trainset[step - 1]
        if (
            int(item["frame_id"]) != int(manifest_record["pose_id"])
            or int(item["cam_id"]) != int(manifest_record["camera_id"])
        ):
            raise RuntimeError(f"record schedule mismatch at step {step}")
        record = train_step(model, item, runtime_args, step)
        training_records.append(record)
        warning_set.update(record["warning_messages"])
        with training_log_path.open(
            "a", encoding="utf-8", newline="\n"
        ) as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if step in DIAGNOSTIC_STEPS:
            checkpoint = save_checkpoint(
                args.attempt_root
                / f"checkpoints/step_{step:06d}.pth",
                model,
                step=step,
                data_order_position=step,
                scene_scale=scene.scene_scale,
                execution_source_head=execution_head,
                contract_hashes=bundle["hashes"],
            )
            checkpoints.append(checkpoint)
            report, _ = evaluate_queries(
                model,
                bundle,
                args.data_root,
                args.attempt_root,
                step=step,
                queries=diagnostic,
                lpips_metric=lpips_metric,
                lpips_error=lpips_error,
                visual_subdir=f"diagnostic_step_{step:06d}",
            )
            report_path = (
                args.attempt_root
                / f"evaluations/diagnostic_step_{step:06d}.json"
            )
            write_json(report_path, report)
            diagnostic_reports[str(step)] = {
                "path": str(report_path),
                "sha256": sha256_file(report_path),
                "query_count": report["query_count"],
            }
    if len(training_records) != N_VALID:
        raise RuntimeError("full strict-train pass did not complete")
    final_checkpoint = save_checkpoint(
        args.attempt_root / f"checkpoints/step_{FINAL_STEP:06d}.pth",
        model,
        step=FINAL_STEP,
        data_order_position=FINAL_STEP,
        scene_scale=scene.scene_scale,
        execution_source_head=execution_head,
        contract_hashes=bundle["hashes"],
    )
    checkpoints.append(final_checkpoint)
    final_evaluation, smoke_arrays = evaluate_queries(
        model,
        bundle,
        args.data_root,
        args.attempt_root,
        step=FINAL_STEP,
        queries=bundle["evaluation"]["queries"],
        lpips_metric=lpips_metric,
        lpips_error=lpips_error,
        visual_subdir=f"step_{FINAL_STEP:06d}",
        smoke_ordinals=set(DIAGNOSTIC_ORDINALS),
    )
    final_eval_path = (
        args.attempt_root
        / f"evaluations/step_{FINAL_STEP:06d}_results.json"
    )
    write_json(final_eval_path, final_evaluation)
    baseline_path = (
        args.attempt_root
        / f"evaluations/step_{FINAL_STEP:06d}_roundtrip_baseline.npz"
    )
    np.savez(
        baseline_path,
        **{key: smoke_arrays[key] for key in sorted(smoke_arrays)},
    )
    post_attachments = core.attachment_inventory(model, args.assets_root)
    frozen_unchanged = post_attachments == initial_attachments
    if not frozen_unchanged:
        raise RuntimeError(
            "SUBJECT00_MEDIUM_PILOT_BLOCKED_SURFACE_LBS_MUTATION"
        )
    visual_paths = [
        item["visual_path"] for item in final_evaluation["records"]
    ]
    contact_sheets = core.build_contact_sheets(
        visual_paths,
        args.attempt_root
        / f"visuals/step_{FINAL_STEP:06d}_contact_sheets",
    )
    window = max(1, int(math.floor(N_VALID * 0.05)))
    total_losses = [item["total_loss"] for item in training_records]
    primary_losses = [item["l1_loss"] for item in training_records]
    first_total = float(np.median(total_losses[:window]))
    last_total = float(np.median(total_losses[-window:]))
    first_primary = float(np.median(primary_losses[:window]))
    last_primary = float(np.median(primary_losses[-window:]))
    lpips_nonzero = [
        int(item["step"])
        for item in training_records
        if float(item["lpips_loss"]) > 0.0
    ]
    result = {
        "schema_version": "subject00.medium.training_result.v1",
        "task_id": TASK_ID,
        "status": "TRAINING_COMPLETE_PENDING_ROUNDTRIP_AND_VISUAL_REVIEW",
        "execution_source_head": execution_head,
        "training_runs": 1,
        "process_segments": 2 if resume_step else 1,
        "infrastructure_resume_from_step": resume_step or None,
        "initialization": initialization,
        "record_count": N_VALID,
        "unique_exposure_count": len(
            {
                (int(item["pose_id"]), int(item["camera_id"]))
                for item in training_records
            }
        ),
        "repeated_training_record_count": 0,
        "optimizer_created": 1,
        "training_forward_batches": N_VALID,
        "backward_calls": N_VALID,
        "optimizer_steps": N_VALID,
        "checkpoint_writes": 4,
        "step0_reused_pointer": 1,
        "data_order_sha256": TRAIN_ORDER_SHA,
        "loss_finite_all": all(
            item["loss_finite"] for item in training_records
        ),
        "gradient_finite_all": all(
            item["gradients_finite"] for item in training_records
        ),
        "optimizer_state_finite_all": all(
            item["optimizer_state_finite"] for item in training_records
        ),
        "intended_gradient_nonzero_all": all(
            bool(item["nonzero_gradient_groups"])
            for item in training_records
        ),
        "loss_window": {"fraction": 0.05, "count": window},
        "total_loss": {
            "first5_percent_median": first_total,
            "last5_percent_median": last_total,
            "relative_reduction": (
                (first_total - last_total) / first_total
            ),
            "gate_last_below_first": last_total < first_total,
        },
        "primary_reconstruction_loss": {
            "name": "l1_loss",
            "first5_percent_median": first_primary,
            "last5_percent_median": last_primary,
            "relative_reduction": (
                (first_primary - last_primary) / first_primary
            ),
            "gate_last_below_first": last_primary < first_primary,
        },
        "lpips_training_loss": {
            "activation_rule": "step > 6000",
            "first_nonzero_step": (
                min(lpips_nonzero) if lpips_nonzero else None
            ),
            "nonzero_step_count": len(lpips_nonzero),
        },
        "optimizer_audit": optimizer_audit_result,
        "runtime_lbs_counters": model.runtime_lbs_counters,
        "topology_call_counts": {
            "clone": 0,
            "split": 0,
            "densification": 0,
            "topology_changing_prune": 0,
            "off_surface_rebind": 0,
        },
        "warning_set": sorted(warning_set),
        "bin_overflow_warning": any(
            "overflow" in item.lower() for item in warning_set
        ),
        "diagnostic_evaluations": diagnostic_reports,
        "final_evaluation": {
            "path": str(final_eval_path),
            "sha256": sha256_file(final_eval_path),
            "successful_render_count": final_evaluation[
                "successful_render_count"
            ],
            "failed_render_count": final_evaluation["failed_render_count"],
            "contact_sheets": contact_sheets,
        },
        "roundtrip_baseline": {
            "path": str(baseline_path),
            "sha256": sha256_file(baseline_path),
            "query_count": 8,
            "array_count": len(smoke_arrays),
        },
        "checkpoints": checkpoints,
        "attachment_template_lbs_unchanged": frozen_unchanged,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "wall_seconds": time.perf_counter() - started,
        "logical_primary_evaluation_records": 216,
        "new_render_count": 120,
        "reused_step0_render_count": 96,
        "secondary_canary_reference_count": 96,
        "PAPER_FINAL": 0,
    }
    write_json(
        args.attempt_root / "training_logs/training_result.json", result
    )
    write_json(
        args.attempt_root
        / "audits/post_training_attachment_inventory.json",
        post_attachments,
    )
    scene.tb_writer.close()
    print(
        json.dumps(
            {
                "phase": "train",
                "status": "PASS",
                "steps": N_VALID,
                "new_checkpoints": 4,
                "new_evaluation_renders": 120,
            },
            sort_keys=True,
        )
    )
    return 0


def roundtrip_phase(args) -> int:
    configure_core()
    bundle = load_bundle(args.availability_manifest)
    checkpoint_path = (
        args.attempt_root / f"checkpoints/step_{FINAL_STEP:06d}.pth"
    )
    baseline_path = (
        args.attempt_root
        / f"evaluations/step_{FINAL_STEP:06d}_roundtrip_baseline.npz"
    )
    payload = torch.load(checkpoint_path, weights_only=False)
    if (
        int(payload["training_step"]) != FINAL_STEP
        or int(payload["data_order_position"]) != FINAL_STEP
    ):
        raise RuntimeError("final checkpoint position changed")
    smpl_path = REPO_ROOT / "smpl_model/smplx/SMPLX_NEUTRAL.npz"
    if not smpl_path.exists():
        smpl_path = Path(
            "/root/autodl-tmp/canondressgs_work/mmlphuman_code/"
            "smpl_model/smplx/SMPLX_NEUTRAL.npz"
        )
    init_smpl(str(smpl_path))
    model = core.GaussianModel()
    runtime_args = core.build_runtime_args(
        bundle, args.data_root, args.assets_root, args.attempt_root
    )
    restore_checkpoint_state(
        model, runtime_args, float(payload["scene_scale"]), payload
    )
    optimizer_audit_result = core.optimizer_audit(
        model,
        bundle["contract"],
        float(payload["scene_scale"]),
        scheduler_step=FINAL_STEP,
    )
    diagnostic = core.diagnostic_queries(bundle["evaluation"])
    current: dict[str, np.ndarray] = {}
    background = torch.ones(3, device="cuda")
    for query in diagnostic:
        pose_id = int(query["pose_id"])
        camera_id = int(query["camera_id"])
        dataset = core.ThumanDataset(
            datadir=str(args.data_root),
            frame_ids=[pose_id],
            cam_ids=[camera_id],
            background=np.ones(3, dtype=np.float32),
            image_scaling=1,
            is_in_memory=False,
        )
        item = core.prepare_item(dataset[0])
        core.set_model_pose(model, item)
        with torch.no_grad():
            rgb, alpha, depth, _ = model.render(
                item, background=background, return_depth=True
            )
        ordinal = int(query["ordinal"])
        current[f"query_{ordinal:03d}_rgb"] = (
            rgb.detach().cpu().numpy().astype(np.float32)
        )
        current[f"query_{ordinal:03d}_alpha"] = (
            alpha[..., 0].detach().cpu().numpy().astype(np.float32)
        )
        current[f"query_{ordinal:03d}_depth"] = (
            depth[..., 0].detach().cpu().numpy().astype(np.float32)
        )
    records = []
    with np.load(baseline_path, allow_pickle=False) as baseline:
        if set(baseline.files) != set(current):
            raise RuntimeError("roundtrip render key set changed")
        for key in sorted(current):
            expected = np.asarray(baseline[key])
            actual = current[key]
            difference = np.abs(
                expected.astype(np.float64) - actual.astype(np.float64)
            )
            records.append(
                {
                    "array": key,
                    "shape": list(actual.shape),
                    "exact": bool(np.array_equal(expected, actual)),
                    "max_abs": float(difference.max(initial=0.0)),
                    "mae": float(difference.mean()),
                }
            )
    max_abs = max(item["max_abs"] for item in records)
    attachment_state = model.surface_attachment_state_dict(cpu=True)
    result = {
        "schema_version": "subject00.medium.checkpoint_roundtrip.v1",
        "task_id": TASK_ID,
        "status": "PASS" if max_abs <= 1.0e-6 else "FAIL",
        "checkpoint": {
            "path": str(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": sha256_file(checkpoint_path),
            "step": int(payload["training_step"]),
            "data_order_position": int(payload["data_order_position"]),
            "source_branch": payload["source_branch"],
            "source_head": payload["source_head"],
        },
        "gaussian_count": int(model._xyz.shape[0]),
        "attachment_count": int(attachment_state["face_ids"].shape[0]),
        "lbs_shape": list(attachment_state["lbs_weights"].shape),
        "cached_weights_exact_to_checkpoint": bool(
            torch.equal(model.get_weights.cpu(), payload["_weights"].cpu())
        ),
        "face_ids_exact_to_checkpoint": bool(
            torch.equal(
                attachment_state["face_ids"],
                payload["surface_attachment"]["face_ids"].cpu(),
            )
        ),
        "barycentric_exact_to_checkpoint": bool(
            torch.equal(
                attachment_state["barycentric"],
                payload["surface_attachment"]["barycentric"].cpu(),
            )
        ),
        "optimizer_state_restored": core.optimizer_state_finite(model),
        "scheduler_state_restored": True,
        "rng_state_present": all(
            key in payload
            for key in (
                "python_rng_state",
                "numpy_rng_state",
                "torch_rng_state",
                "cuda_rng_states",
            )
        ),
        "template_sha256": payload["template_sha256"],
        "sampler_sha256": payload["sampler_sha256"],
        "derived_manifest_sha256": payload["derived_manifest_sha256"],
        "legacy_grid_loads": int(
            model.runtime_lbs_counters["legacy_grid_loads"]
        ),
        "spatial_weight_queries": int(
            model.runtime_lbs_counters["spatial_weight_queries"]
        ),
        "optimizer_audit": optimizer_audit_result,
        "render_query_count": 8,
        "render_array_count": len(records),
        "render_tolerance": 1.0e-6,
        "render_max_abs": max_abs,
        "render_all_exact": all(item["exact"] for item in records),
        "render_records": records,
        "PAPER_FINAL": 0,
    }
    if result["status"] != "PASS":
        raise RuntimeError(f"final checkpoint roundtrip failed: {result}")
    write_json(
        args.attempt_root / "audits/checkpoint_roundtrip.json", result
    )
    print(
        json.dumps(
            {
                "phase": "roundtrip",
                "status": result["status"],
                "checkpoint_sha256": result["checkpoint"]["sha256"],
                "render_max_abs": max_abs,
            },
            sort_keys=True,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train", "roundtrip"), required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/datasets/"
            "thuman4_second_identity_staging/subject00"
        ),
    )
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/datasets/"
            "thuman4_second_identity_staging/derived_assets/subject00"
        ),
    )
    parser.add_argument(
        "--availability-manifest",
        type=Path,
        default=Path(
            "/root/autodl-tmp/datasets/"
            "thuman4_second_identity_staging/reports/"
            "SUBJECT00_VALID_FRAME_CAMERA_MANIFEST.json"
        ),
    )
    parser.add_argument(
        "--attempt-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/canondressgs_work/outputs/"
            "SUBJECT00-MMLPHUMAN-MEDIUM-PILOT-001/attempt_001"
        ),
    )
    parser.add_argument("--resume-step", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.phase == "train":
        return train_phase(args)
    return roundtrip_phase(args)


if __name__ == "__main__":
    raise SystemExit(main())
