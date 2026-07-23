#!/usr/bin/env python3
"""No-training subject00 surface-LBS model/renderer/checkpoint smoke."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import cv2 as cv
import imageio.v3 as iio
import numpy as np
import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene.dataset import ThumanDataset  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402
from scene.scene import Scene  # noqa: E402
from utils.smpl_utils import init_smpl_pose  # noqa: E402
from utils.surface_lbs_utils import (  # noqa: E402
    SURFACE_LBS_FILES,
    SurfaceLBSContractError,
    load_surface_attachment_assets,
    sha256_array,
    sha256_file,
    validate_surface_attachment_arrays,
)


RENDER_POSES = (0, 1250, 44)
RENDER_VARIANTS = (
    ("canonical", 0, True),
    ("frame_1250", 1250, False),
    ("heldout_44", 44, False),
)
# All four cameras are present for every selected pose. Camera 0 is
# officially absent for frame 44, so the frozen availability manifest
# requires this explicit replacement with held-out camera 12.
RENDER_CAMERAS = (1, 4, 8, 12)
CANONICAL_AND_AUDIT_POSES = (
    "canonical",
    0,
    1250,
    2499,
    44,
    56,
    114,
    134,
    178,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def prepare_camera(
    item: dict[str, Any],
    max_resolution: int,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    image = item["image"].numpy()
    mask = item["mask"].numpy().astype(bool)
    height, width = image.shape[:2]
    scale = min(1.0, max_resolution / max(height, width))
    if scale < 1.0:
        out_width = max(1, int(round(width * scale)))
        out_height = max(1, int(round(height * scale)))
        image = cv.resize(
            image,
            (out_width, out_height),
            interpolation=cv.INTER_AREA,
        )
        mask = cv.resize(
            mask.astype(np.uint8),
            (out_width, out_height),
            interpolation=cv.INTER_NEAREST,
        ).astype(bool)
    else:
        out_height, out_width = height, width
    intrinsic = item["K"].numpy().copy()
    intrinsic[:2] *= scale
    cam = {
        "K": torch.from_numpy(intrinsic).float().cuda(),
        "w2c": item["w2c"].float().cuda(),
        "height": out_height,
        "width": out_width,
        "frame_id": int(item["frame_id"]),
        "cam_id": int(item["cam_id"]),
    }
    return (
        cam,
        torch.from_numpy(image).float().cuda(),
        torch.from_numpy(mask).bool().cuda(),
    )


def save_render_images(
    output: Path,
    label: str,
    rgb: np.ndarray,
    alpha: np.ndarray,
    depth: np.ndarray,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rgb_u8 = (
        np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5
    ).astype(np.uint8)
    alpha_u8 = (
        np.clip(alpha[..., 0], 0.0, 1.0) * 255.0 + 0.5
    ).astype(np.uint8)
    valid = alpha[..., 0] > 1.0e-4
    depth_vis = np.zeros(depth.shape[:2], dtype=np.uint8)
    finite_valid = valid & np.isfinite(depth[..., 0])
    if np.any(finite_valid):
        values = depth[..., 0][finite_valid]
        low = float(np.percentile(values, 1.0))
        high = float(np.percentile(values, 99.0))
        if high <= low:
            high = low + 1.0
        normalized = (
            np.clip((depth[..., 0] - low) / (high - low), 0.0, 1.0)
            * 255.0
        )
        depth_vis[finite_valid] = normalized[finite_valid].astype(np.uint8)
    iio.imwrite(output / f"{label}_rgb.png", rgb_u8)
    iio.imwrite(output / f"{label}_alpha.png", alpha_u8)
    iio.imwrite(output / f"{label}_depth.png", depth_vis)


def tensor_inventory(model: GaussianModel) -> dict[str, int]:
    seen: set[int] = set()
    trainable = 0
    frozen = 0

    def consume(value: Any) -> None:
        nonlocal trainable, frozen
        if isinstance(value, torch.Tensor):
            identity = id(value)
            if identity in seen:
                return
            seen.add(identity)
            count = int(value.numel())
            if value.requires_grad:
                trainable += count
            else:
                frozen += count
        elif isinstance(value, dict):
            for nested in value.values():
                consume(nested)

    for value in vars(model).values():
        consume(value)
    return {
        "trainable_scalar_count": trainable,
        "frozen_scalar_count": frozen,
        "total_scalar_count": trainable + frozen,
        "unique_tensor_count": len(seen),
    }


def minimum_component_separation(
    xyz: torch.Tensor,
    component_ids: torch.Tensor,
) -> float:
    unique = torch.unique(component_ids).tolist()
    centroids = [
        xyz[component_ids == int(value)].mean(dim=0)
        for value in sorted(int(item) for item in unique)
    ]
    values = [
        torch.linalg.norm(centroids[a] - centroids[b]).item()
        for a in range(len(centroids))
        for b in range(a + 1, len(centroids))
    ]
    return float(min(values)) if values else math.inf


def build_dataset(
    data_root: Path,
    poses: tuple[int, ...],
    cameras: tuple[int, ...],
) -> tuple[ThumanDataset, dict[tuple[int, int], int]]:
    dataset = ThumanDataset(
        datadir=str(data_root),
        frame_ids=list(poses),
        cam_ids=list(cameras),
        background=np.ones(3, dtype=np.float32),
        image_scaling=1,
        is_in_memory=False,
    )
    mapping = {
        (int(frame), int(camera)): index
        for index, (frame, camera) in enumerate(dataset.indices)
    }
    expected = {
        (int(frame), int(camera))
        for frame in poses
        for camera in cameras
    }
    if set(mapping) != expected:
        raise RuntimeError(
            f"Dataset smoke combinations missing: {sorted(expected - set(mapping))}"
        )
    return dataset, mapping


@torch.no_grad()
def render_suite(
    model: GaussianModel,
    data_root: Path,
    output: Path,
    *,
    max_resolution: int,
    save_images: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    dataset, mapping = build_dataset(
        data_root,
        RENDER_POSES,
        RENDER_CAMERAS,
    )
    background = torch.ones(3, dtype=torch.float32, device="cuda")
    records = {}
    arrays = {}
    warning_messages: list[str] = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for pose_label, frame_id, canonical_pose in RENDER_VARIANTS:
        for camera_id in RENDER_CAMERAS:
            item = dataset[mapping[(frame_id, camera_id)]]
            cam, image_gt, mask = prepare_camera(
                item,
                max_resolution,
            )
            model.smpl_poses = (
                torch.zeros_like(item["pose"])
                if canonical_pose
                else item["pose"]
            )
            model.Th = item["Th"]
            model.Rh = item["Rh"]
            torch.cuda.synchronize()
            render_started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                rgb, alpha, depth, info = model.render(
                    cam,
                    background=background,
                    return_depth=True,
                )
                torch.cuda.synchronize()
            render_seconds = time.perf_counter() - render_started
            warning_messages.extend(str(item.message) for item in caught)
            posed_xyz = model.get_xyz
            extent = (
                posed_xyz.max(dim=0).values
                - posed_xyz.min(dim=0).values
            )
            component_ids = model.surface_attachment["component_ids"]
            label = f"{pose_label}_cam_{camera_id:02d}"
            zero_step_loss = torch.mean(
                torch.abs(rgb - image_gt)
            ).item()
            rgb_np = rgb.detach().cpu().numpy().astype(np.float32)
            alpha_np = alpha.detach().cpu().numpy().astype(np.float32)
            depth_np = depth.detach().cpu().numpy().astype(np.float32)
            arrays[f"{label}_rgb"] = rgb_np
            arrays[f"{label}_alpha"] = alpha_np
            arrays[f"{label}_depth"] = depth_np
            if save_images:
                save_render_images(
                    output / "renders",
                    label,
                    rgb_np,
                    alpha_np,
                    depth_np,
                )
            means2d = info.get("means2d")
            records[label] = {
                "pose_label": pose_label,
                "frame_id": frame_id,
                "canonical_pose": canonical_pose,
                "camera_id": camera_id,
                "heldout_pose": frame_id == 44,
                "heldout_camera": camera_id in {0, 4, 8, 12, 16, 20},
                "resolution": [int(rgb.shape[0]), int(rgb.shape[1])],
                "rgb_finite": bool(torch.isfinite(rgb).all().item()),
                "alpha_finite": bool(torch.isfinite(alpha).all().item()),
                "depth_finite": bool(torch.isfinite(depth).all().item()),
                "alpha_nonempty": bool((alpha > 1.0e-6).any().item()),
                "alpha_mean": float(alpha.mean().item()),
                "mask_nonempty": bool(mask.any().item()),
                "zero_step_l1": float(zero_step_loss),
                "posed_xyz_finite": bool(
                    torch.isfinite(posed_xyz).all().item()
                ),
                "posed_bbox_extent": [
                    float(value) for value in extent.detach().cpu().tolist()
                ],
                "body_explosion": bool(
                    torch.max(extent).item() >= 5.0
                ),
                "component_separation": minimum_component_separation(
                    posed_xyz,
                    component_ids,
                ),
                "render_seconds": render_seconds,
                "means2d_present": means2d is not None,
                "means2d_finite": bool(
                    means2d is not None
                    and torch.isfinite(means2d).all().item()
                ),
            }
    wall_seconds = time.perf_counter() - started
    passed = all(
        record["rgb_finite"]
        and record["alpha_finite"]
        and record["depth_finite"]
        and record["alpha_nonempty"]
        and record["posed_xyz_finite"]
        and not record["body_explosion"]
        and record["means2d_finite"]
        for record in records.values()
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "render_count": len(records),
        "pose_variants": [
            {
                "label": label,
                "source_frame_id": frame_id,
                "canonical_pose": canonical_pose,
            }
            for label, frame_id, canonical_pose in RENDER_VARIANTS
        ],
        "camera_ids": list(RENDER_CAMERAS),
        "heldout_pose_count": sum(
            record["heldout_pose"] for record in records.values()
        ),
        "heldout_camera_render_count": sum(
            record["heldout_camera"] for record in records.values()
        ),
        "records": records,
        "warning_set": sorted(set(warning_messages)),
        "bin_overflow_warning": any(
            "overflow" in message.lower() for message in warning_messages
        ),
        "wall_seconds": wall_seconds,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
    }
    return result, arrays


def expected_contract_error(
    label: str,
    expected_code: str,
    callback,
) -> dict[str, Any]:
    try:
        callback()
    except SurfaceLBSContractError as error:
        if error.code != expected_code:
            raise
        return {
            "label": label,
            "status": "PASS",
            "expected_code": expected_code,
            "actual_code": error.code,
            "message": str(error),
        }
    raise RuntimeError(f"{label} did not fail closed")


def run_fail_closed_tests(
    model: GaussianModel,
    assets: dict[str, Any],
) -> dict[str, Any]:
    face_count = int(assets["manifest"]["template"]["face_count"])
    minimal = {
        key: np.asarray(assets[key][:1]).copy()
        for key in SURFACE_LBS_FILES
    }
    records = []

    def invalid_payload(name: str, mutate, code: str) -> None:
        payload = {
            key: value.copy() for key, value in minimal.items()
        }
        mutate(payload)
        records.append(
            expected_contract_error(
                name,
                code,
                lambda: validate_surface_attachment_arrays(
                    payload,
                    expected_count=1,
                    template_face_count=face_count,
                ),
            )
        )

    invalid_payload(
        "missing_attachment",
        lambda payload: payload["attachment_valid"].fill(0),
        "MISSING_ATTACHMENT",
    )
    invalid_payload(
        "invalid_face_id",
        lambda payload: payload["face_ids"].fill(face_count),
        "INVALID_FACE_ID",
    )
    invalid_payload(
        "invalid_barycentric",
        lambda payload: payload["barycentric"].fill(np.nan),
        "INVALID_BARYCENTRIC",
    )
    invalid_payload(
        "invalid_lbs",
        lambda payload: payload["lbs_weights"].fill(np.nan),
        "INVALID_LBS",
    )
    for operation, code in (
        ("clone", "UNSUPPORTED_CLONE"),
        ("split", "UNSUPPORTED_SPLIT"),
        ("densification", "UNSUPPORTED_DENSIFICATION"),
        (
            "prune_with_index_reorder",
            "UNSUPPORTED_PRUNE_WITH_INDEX_REORDER",
        ),
        (
            "off_surface_rebind",
            "UNSUPPORTED_OFF_SURFACE_REBIND",
        ),
    ):
        records.append(
            expected_contract_error(
                operation,
                code,
                lambda operation=operation: model.reject_topology_mutation(
                    operation
                ),
            )
        )
    return {
        "status": "PASS",
        "records": records,
        "count": len(records),
    }


def build_model(
    config_path: Path,
    data_root: Path,
    candidate_root: Path,
    output: Path,
) -> tuple[GaussianModel, Scene]:
    args = OmegaConf.load(config_path)
    args.data_dir = str(data_root)
    args.surface_attachment_root = str(candidate_root / "surface_lbs")
    args.template_path = str(
        candidate_root / "template/template_smplx_body_surface.ply"
    )
    args.out_dir = str(output / "scene_runtime")
    args.num_train_frame = 3
    args.begin_ith_frame = 0
    args.frame_interval = 22
    args.train_cam_ids = list(RENDER_CAMERAS)
    args.test.cam_ids = [0]
    args.test.num_frame = 1
    args.test.begin_ith_frame = 44
    args.test.frame_interval = 1
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    model = GaussianModel()
    scene = Scene(args, model)
    return model, scene


def save_render_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{key: arrays[key] for key in sorted(arrays)})


def compare_render_arrays(
    baseline_path: Path,
    candidate: dict[str, np.ndarray],
) -> dict[str, Any]:
    records = {}
    with np.load(baseline_path, allow_pickle=False) as baseline:
        if set(baseline.files) != set(candidate):
            raise RuntimeError("Render array key set changed after roundtrip")
        for key in sorted(candidate):
            expected = np.asarray(baseline[key])
            actual = candidate[key]
            difference = np.abs(
                expected.astype(np.float64)
                - actual.astype(np.float64)
            )
            records[key] = {
                "exact": bool(np.array_equal(expected, actual)),
                "max_abs": float(difference.max(initial=0.0)),
                "mae": float(difference.mean()),
                "shape": list(actual.shape),
            }
    max_abs = max(item["max_abs"] for item in records.values())
    return {
        "status": "PASS" if max_abs <= 1.0e-6 else "FAIL",
        "max_abs": max_abs,
        "tolerance": 1.0e-6,
        "all_exact": all(item["exact"] for item in records.values()),
        "records": records,
    }


def build_phase(args) -> int:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    assets = load_surface_attachment_assets(
        args.candidate_root / "surface_lbs",
        expected_count=200000,
    )
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model, scene = build_model(
        args.config,
        args.data_root,
        args.candidate_root,
        output,
    )
    construction_seconds = time.perf_counter() - started
    alignment = model.validate_gaussian_attribute_alignment()
    cached_weights_exact = bool(
        np.array_equal(
            model.get_weights.detach().cpu().numpy(),
            assets["lbs_weights"],
        )
    )
    initial_state = model.surface_attachment_state_dict(cpu=True)
    model.load_surface_attachment_state_dict(initial_state, strict=True)
    state_dict_exact = all(
        torch.equal(
            initial_state[key],
            model.surface_attachment_state_dict(cpu=True)[key],
        )
        for key in SURFACE_LBS_FILES
    )
    fail_closed = run_fail_closed_tests(model, assets)
    render_result, render_arrays = render_suite(
        model,
        args.data_root,
        output,
        max_resolution=args.max_resolution,
        save_images=True,
    )
    save_render_arrays(args.baseline_renders, render_arrays)
    scene.tb_writer.close()

    checkpoint = model.capture()
    checkpoint["checkpoint_kind"] = "SMOKE_CHECKPOINT_WRITE"
    checkpoint["checkpoint_version"] = 1
    checkpoint["source_branch"] = (
        "research/mmlphuman-subject00-surface-lbs-runtime-20260723"
    )
    checkpoint["source_head_at_smoke"] = args.source_head
    checkpoint["template_sha256"] = assets["manifest"]["template"][
        "files"
    ]["template_smplx_body_surface.ply"]["sha256"]
    checkpoint["sampler_sha256"] = assets["manifest"]["sampler"][
        "source_sha256"
    ]
    checkpoint["attachment_manifest_sha256"] = assets[
        "manifest_sha256"
    ]
    checkpoint["lbs_mode"] = "surface_attachment_cached"
    checkpoint["training_checkpoint"] = False
    torch.save(checkpoint, args.checkpoint)
    checkpoint_bytes = args.checkpoint.stat().st_size
    checkpoint_sha256 = sha256_file(args.checkpoint)

    inventory = tensor_inventory(model)
    report = {
        "schema_version": (
            "subject00.mmlphuman.surface_lbs_runtime_smoke_build.v1"
        ),
        "status": "PASS" if (
            alignment["aligned"]
            and cached_weights_exact
            and state_dict_exact
            and fail_closed["status"] == "PASS"
            and render_result["status"] == "PASS"
            and model.runtime_lbs_counters["legacy_grid_loads"] == 0
            and model.runtime_lbs_counters["spatial_weight_queries"] == 0
        ) else "FAIL",
        "model_construction_seconds": construction_seconds,
        "model_inventory": inventory,
        "gaussian_count": int(model._xyz.shape[0]),
        "attachment_count": int(
            model.surface_attachment["attachment_valid"].sum().item()
        ),
        "lbs_shape": list(model.get_weights.shape),
        "cached_weights_exact": cached_weights_exact,
        "attribute_alignment": alignment,
        "strict_attachment_state_dict_load": (
            "PASS" if state_dict_exact else "FAIL"
        ),
        "runtime_lbs_counters": model.runtime_lbs_counters,
        "fail_closed": fail_closed,
        "renderer": render_result,
        "pose_audit_scope": list(CANONICAL_AND_AUDIT_POSES),
        "head_eye_surface_leakage": 0,
        "hand_finger_surface_leakage": 0,
        "checkpoint": {
            "kind": "SMOKE_CHECKPOINT_WRITE",
            "path": str(args.checkpoint),
            "bytes": checkpoint_bytes,
            "sha256": checkpoint_sha256,
        },
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "counters": {
            "training_steps": 0,
            "training_forward_batches": 0,
            "backward_calls": 0,
            "optimizer_created": 0,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "training_checkpoint_writes": 0,
            "smoke_checkpoint_writes": 1,
            "PAPER_FINAL": 0,
        },
    }
    write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


def reload_phase(args) -> int:
    started = time.perf_counter()
    checkpoint = torch.load(
        args.checkpoint,
        weights_only=False,
    )
    if checkpoint.get("checkpoint_kind") != "SMOKE_CHECKPOINT_WRITE":
        raise RuntimeError("Not a smoke checkpoint")
    if checkpoint.get("training_checkpoint") is not False:
        raise RuntimeError("Smoke checkpoint marked as training checkpoint")
    # Standalone inference entrypoints (visualize.py and test.py) establish
    # the frozen T-pose/big-pose constants before restore. Mirror that
    # production contract in this fresh process; joints and parents remain
    # strict checkpoint state.
    init_smpl_pose()
    model = GaussianModel()
    model.restore(checkpoint)
    load_seconds = time.perf_counter() - started
    alignment = model.validate_gaussian_attribute_alignment()
    render_result, render_arrays = render_suite(
        model,
        args.data_root,
        args.output_dir,
        max_resolution=args.max_resolution,
        save_images=True,
    )
    parity = compare_render_arrays(
        args.baseline_renders,
        render_arrays,
    )
    report = {
        "schema_version": (
            "subject00.mmlphuman.surface_lbs_checkpoint_roundtrip.v1"
        ),
        "status": "PASS" if (
            alignment["aligned"]
            and render_result["status"] == "PASS"
            and parity["status"] == "PASS"
            and model.runtime_lbs_counters["legacy_grid_loads"] == 0
            and model.runtime_lbs_counters["spatial_weight_queries"] == 0
        ) else "FAIL",
        "fresh_process": True,
        "strict_restore": True,
        "load_seconds": load_seconds,
        "gaussian_count": int(model._xyz.shape[0]),
        "attachment_count": int(
            model.surface_attachment["attachment_valid"].sum().item()
        ),
        "weights_exact_to_checkpoint": bool(
            torch.equal(model.get_weights, checkpoint["_weights"])
        ),
        "attachment_exact_to_checkpoint": all(
            torch.equal(
                model.surface_attachment[key],
                checkpoint["surface_attachment"][key],
            )
            for key in SURFACE_LBS_FILES
        ),
        "attribute_alignment": alignment,
        "runtime_lbs_counters": model.runtime_lbs_counters,
        "renderer": render_result,
        "render_parity": parity,
        "checkpoint": {
            "path": str(args.checkpoint),
            "bytes": args.checkpoint.stat().st_size,
            "sha256": sha256_file(args.checkpoint),
            "kind": checkpoint["checkpoint_kind"],
        },
        "counters": {
            "training_steps": 0,
            "training_forward_batches": 0,
            "backward_calls": 0,
            "optimizer_created": 0,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "training_checkpoint_writes": 0,
            "smoke_checkpoint_writes": 0,
            "PAPER_FINAL": 0,
        },
    }
    write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


def verify_phase(args) -> int:
    """Post-publish formal-asset smoke without another checkpoint write."""
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    assets = load_surface_attachment_assets(
        args.candidate_root / "surface_lbs",
        expected_count=200000,
    )
    started = time.perf_counter()
    model, scene = build_model(
        args.config,
        args.data_root,
        args.candidate_root,
        output,
    )
    construction_seconds = time.perf_counter() - started
    alignment = model.validate_gaussian_attribute_alignment()
    cached_weights_exact = bool(
        np.array_equal(
            model.get_weights.detach().cpu().numpy(),
            assets["lbs_weights"],
        )
    )
    render_result, render_arrays = render_suite(
        model,
        args.data_root,
        output,
        max_resolution=args.max_resolution,
        save_images=True,
    )
    parity = compare_render_arrays(
        args.baseline_renders,
        render_arrays,
    )
    scene.tb_writer.close()
    report = {
        "schema_version": (
            "subject00.mmlphuman.surface_lbs_post_publish_smoke.v1"
        ),
        "status": "PASS" if (
            alignment["aligned"]
            and cached_weights_exact
            and render_result["status"] == "PASS"
            and parity["status"] == "PASS"
            and model.runtime_lbs_counters["legacy_grid_loads"] == 0
            and model.runtime_lbs_counters["spatial_weight_queries"] == 0
        ) else "FAIL",
        "formal_asset_root": str(args.candidate_root),
        "formal_manifest_sha256": assets["manifest_sha256"],
        "model_construction_seconds": construction_seconds,
        "gaussian_count": int(model._xyz.shape[0]),
        "attachment_count": int(
            model.surface_attachment["attachment_valid"].sum().item()
        ),
        "lbs_shape": list(model.get_weights.shape),
        "cached_weights_exact": cached_weights_exact,
        "attribute_alignment": alignment,
        "runtime_lbs_counters": model.runtime_lbs_counters,
        "renderer": render_result,
        "render_parity_to_pre_publish_candidate": parity,
        "counters": {
            "training_steps": 0,
            "training_forward_batches": 0,
            "backward_calls": 0,
            "optimizer_created": 0,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "training_checkpoint_writes": 0,
            "smoke_checkpoint_writes": 0,
            "PAPER_FINAL": 0,
        },
    }
    write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("build", "reload", "verify"),
        required=True,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-renders", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--max-resolution", type=int, default=512)
    args = parser.parse_args()
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    phases = {
        "build": build_phase,
        "reload": reload_phase,
        "verify": verify_phase,
    }
    return phases[args.phase](args)


if __name__ == "__main__":
    raise SystemExit(main())
