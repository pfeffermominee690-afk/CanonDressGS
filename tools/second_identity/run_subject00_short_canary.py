#!/usr/bin/env python3
"""Execute the frozen subject00 repaired-contract short canary.

The three phases are deliberately separate:

* ``preflight`` constructs the model and renders step 0 without an optimizer.
* ``train`` verifies the preflight state, creates the optimizer, and performs
  the one authorized 384-record pass.
* ``roundtrip`` starts a fresh process, strictly restores step 384, and
  verifies eight frozen render queries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Iterable

import cv2 as cv
import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torchmetrics.functional.image import structural_similarity_index_measure


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene.dataset import ThumanDataset  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402
from scene.scene import Scene  # noqa: E402
from utils.image_utils import crop_image  # noqa: E402
from utils.loss_utils import (  # noqa: E402
    dxyz_smooth_loss,
    gaussian_scaling_loss,
    l1_loss,
)
from utils.smpl_utils import init_smpl  # noqa: E402
from utils.surface_lbs_utils import (  # noqa: E402
    SURFACE_LBS_FILES,
    load_surface_attachment_assets,
    sha256_file,
)


TASK_ID = "MMLPHUMAN-SUBJECT00-SHORT-CANARY-FROM-REPAIRED-CONTRACT-001"
SOURCE_BRANCH = "research/mmlphuman-subject00-canary-contract-repair-20260723"
SOURCE_HEAD = "ebcf40da0fca0345f749f7c111e1a3a00f19dda3"
TARGET_BRANCH = (
    "research/mmlphuman-subject00-short-canary-from-repaired-contract-20260723"
)
AVAILABILITY_SHA = (
    "cd00ad0a1549bc99d956e26eea300945adfaa10a20a438e931e640977475564e"
)
TRAIN_ORDER_SHA = (
    "a64a8d40876946b8f0919a4761e7b814e9ca7182cd0434b0cf97a6b2737386ca"
)
EVAL_ORDER_SHA = (
    "38476b7d9f6a012e0e21d7f188c5db5131506ec2231e8a7b50a389c7c92ad98a"
)
DERIVED_MANIFEST_SHA = (
    "de6cd51fe3f81e29139b45494860b186038b75a378b101e7428252b3b66c1af8"
)
CAMERA_SPLIT_SHA = (
    "8827cdc05a08cb7068e1f5e89ab7ba5384bf83b89686e8c9f80024b78d49cf44"
)
POSE_SPLIT_SHA = (
    "c12e5eec87c737d3740b6e6199d657c859df7ce1ec9d371ea8d4a987ef62bf06"
)
CHECKPOINT_STEPS = (0, 96, 192, 288, 384)
DIAGNOSTIC_STEPS = (96, 192, 288)
FORMAL_EVAL_STEPS = (0, 384)
EXPECTED_OPTIMIZER_GROUPS = (
    "dxyz",
    "scales",
    "quats",
    "opacities",
    "sh0",
    "shN",
    "dxyz_bs",
    "dscales_bs",
    "dquats_bs",
    "dopacities_bs",
    "dsh0_bs",
    "dshN_bs",
    "encoder_feat_params",
    "xyz_offset",
)
DIRECT_TRAINABLES = (
    "dxyz_vt",
    "_scaling",
    "_rotation",
    "_opacity",
    "_sh0",
    "_shN",
    "dxyz_bs",
    "scaling_bs",
    "rotation_bs",
    "opacity_bs",
    "sh0_bs",
    "shN_bs",
    "xyz_offset",
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def canonical_sha(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    array = value.detach().contiguous().cpu().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def git_branch() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "branch", "--show-current"],
        text=True,
    ).strip()


def load_json(relative: str) -> dict[str, Any]:
    return json.loads((REPO_ROOT / relative).read_text(encoding="utf-8"))


def load_contract_bundle(availability_path: Path) -> dict[str, Any]:
    config_path = (
        REPO_ROOT / "config/subject00_surface_lbs_short_canary_repaired.yaml"
    )
    contract_path = (
        REPO_ROOT
        / "paper_protocol/second_identity/"
        "subject00_canary_training_contract_repaired.json"
    )
    record_path = (
        REPO_ROOT
        / "paper_protocol/second_identity/subject00_canary_record_manifest.json"
    )
    evaluation_path = (
        REPO_ROOT
        / "paper_protocol/second_identity/subject00_canary_evaluation_manifest.json"
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
    records = json.loads(record_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    availability_contract = json.loads(
        availability_contract_path.read_text(encoding="utf-8")
    )
    repair_summary = json.loads(repair_summary_path.read_text(encoding="utf-8"))
    repaired_config = OmegaConf.load(config_path)
    availability = json.loads(availability_path.read_text(encoding="utf-8"))

    hashes = {
        "repaired_config_file_sha256": sha256_file(config_path),
        "training_contract_file_sha256": sha256_file(contract_path),
        "record_manifest_file_sha256": sha256_file(record_path),
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
                "SUBJECT00_CANARY_REPAIRED_CONTRACT_HASH_MISMATCH: "
                f"{key}={hashes[key]} expected={value}"
            )
    if len(records["records"]) != 384:
        raise RuntimeError("record manifest is not 384 records")
    if len(evaluation["queries"]) != 96:
        raise RuntimeError("evaluation manifest is not 96 queries")
    if contract["classification"] != "SUBJECT00_CANARY_TRAINING_CONTRACT_REPAIRED":
        raise RuntimeError("repaired contract classification changed")
    if (
        contract["subject02_formal_training_contract"]["provenance_status"]
        != "LIMITED_HISTORICAL_PROVENANCE"
    ):
        raise RuntimeError("limited historical provenance was not retained")
    if len(contract["subject02_formal_training_contract"]["optimizer"]["groups"]) != 14:
        raise RuntimeError("subject02 optimizer contract is not 14 groups")
    valid = {
        (int(item["frame_id"]), int(item["camera_id"]))
        for item in availability["entries"]
        if item["image_available"]
        and item["mask_available"]
        and item["valid_pair"]
    }
    for record in records["records"]:
        pair = (int(record["pose_id"]), int(record["camera_id"]))
        if pair not in valid:
            raise RuntimeError(f"training pair unavailable: {pair}")
    for query in evaluation["queries"]:
        pair = (int(query["pose_id"]), int(query["camera_id"]))
        if pair not in valid:
            raise RuntimeError(f"evaluation pair unavailable: {pair}")
    return {
        "contract": contract,
        "records": records,
        "evaluation": evaluation,
        "availability_contract": availability_contract,
        "repair_summary": repair_summary,
        "repaired_config": repaired_config,
        "hashes": hashes,
        "valid_pairs": valid,
    }


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_runtime_args(
    bundle: dict[str, Any],
    data_root: Path,
    assets_root: Path,
    attempt_root: Path,
):
    base = OmegaConf.load(
        REPO_ROOT / "config/subject00_surface_lbs_canary_ready.yaml"
    )
    contract = bundle["contract"]
    subject02 = contract["subject02_formal_training_contract"]
    formal = subject02["formal_config"]
    groups = subject02["optimizer"]["groups"]
    losses = subject02["loss"]["components"]
    records = bundle["records"]
    evaluation = bundle["evaluation"]

    base.data_dir = str(data_root)
    base.surface_attachment_root = str(assets_root / "surface_lbs")
    base.template_path = str(
        assets_root / "template/template_smplx_body_surface.ply"
    )
    base.out_dir = str(attempt_root / "snapshots/scene_runtime")
    base.seed = 0
    base.detect_anomaly = False
    base.background = [1.0, 1.0, 1.0]
    base.random_background = True
    base.data_in_memory = False
    base.image_scaling = 1
    base.init_num_gs = 200000
    base.num_verts = 10000
    base.num_features = 300
    base.lbs_mode = "surface_attachment_cached"
    base.require_surface_attachment = True
    base.allow_off_surface_fallback = False
    base.allow_topology_mutation = False
    base.train_frame_ids = list(records["pose_selection"]["pose_ids"])
    base.train_cam_ids = list(records["camera_ids"])
    base.train_shuffle = False
    base.train_num_workers = int(formal["dataloader_workers"])
    base.num_train_frame = 64
    base.begin_ith_frame = 0
    base.frame_interval = 1
    base.test.cam_ids = [int(evaluation["train_eval_camera_ids"][0])]
    base.test.num_frame = 1
    base.test.begin_ith_frame = int(evaluation["train_eval_pose_ids"][0])
    base.test.frame_interval = 1
    base.test.image_scaling = 1
    base.iterations = int(subject02["scheduler"]["horizon"])
    base.position_lr = 0.00016
    base.scaling_lr = float(groups["scales"]["lr"])
    base.rotation_lr = float(groups["quats"]["lr"])
    base.opacity_lr = float(groups["opacities"]["lr"])
    base.color_lr = float(groups["sh0"]["lr"])
    base.xyz_offset_lr = float(groups["xyz_offset"]["lr"])
    base.encoder_lr = float(groups["encoder_feat_params"]["lr"])
    base.lambda_lpips = float(losses["lpips"]["weight"])
    base.iteration_lpips = 6000
    base.iteration_lpips_random_patch = 300000
    base.lambda_scaling = float(losses["gaussian_scaling"]["weight"])
    base.scaling_threshold = float(losses["gaussian_scaling"]["threshold"])
    base.lambda_dxyz_smooth = float(losses["dxyz_smooth"]["weight"])
    base.iteration_dxyz_basis = 2000
    base.iteration_gsparam_basis = 2000
    base.iteration_sh_degree = 250000
    base.formal_training_enabled = False
    base.canary_training_enabled = False
    return base


def build_model(
    bundle: dict[str, Any],
    data_root: Path,
    assets_root: Path,
    attempt_root: Path,
) -> tuple[GaussianModel, Scene, Any]:
    args = build_runtime_args(bundle, data_root, assets_root, attempt_root)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    set_seed(0)
    model = GaussianModel()
    scene = Scene(args, model)
    alignment = model.validate_gaussian_attribute_alignment()
    if (
        int(alignment.get("gaussian_count", -1)) != 200000
        or not bool(alignment.get("aligned", False))
    ):
        raise RuntimeError(f"Gaussian alignment failed: {alignment}")
    if tuple(model.get_weights.shape) != (200000, 55):
        raise RuntimeError("cached LBS shape changed")
    if model.runtime_lbs_counters["legacy_grid_loads"] != 0:
        raise RuntimeError("legacy grid was loaded")
    if model.runtime_lbs_counters["spatial_weight_queries"] != 0:
        raise RuntimeError("spatial LBS query occurred")
    expected_indices = [
        (int(item["pose_id"]), int(item["camera_id"]))
        for item in bundle["records"]["records"]
    ]
    actual_indices = [
        (int(pose), int(camera)) for pose, camera in scene.trainset.indices
    ]
    if actual_indices != expected_indices:
        raise RuntimeError("Scene training record order differs from manifest")
    return model, scene, args


def trainable_parameters(model: GaussianModel) -> dict[str, torch.Tensor]:
    result = {
        name: getattr(model, name)
        for name in DIRECT_TRAINABLES
    }
    for name, value in sorted(model.encoder_feat_params.items()):
        result[f"encoder_feat_params.{name}"] = value
    return result


def parameter_inventory(model: GaussianModel) -> dict[str, Any]:
    parameters = trainable_parameters(model)
    records = {}
    for name, value in parameters.items():
        records[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "requires_grad": bool(value.requires_grad),
            "numel": int(value.numel()),
            "sha256": tensor_sha(value),
        }
    frozen = {
        "_xyz": model._xyz,
        "_weights": model.get_weights,
        "t_joints": model.t_joints,
        "joint_parents": model.joint_parents,
        "xyz_vt": model.xyz_vt,
        "xyz_ft": model.xyz_ft,
    }
    frozen_records = {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "requires_grad": bool(value.requires_grad),
            "numel": int(value.numel()),
            "sha256": tensor_sha(value),
        }
        for name, value in frozen.items()
    }
    return {
        "trainable_scalar_count": sum(
            item["numel"] for item in records.values()
        ),
        "frozen_scalar_count_selected": sum(
            item["numel"] for item in frozen_records.values()
        ),
        "trainable_parameters": records,
        "frozen_tensors": frozen_records,
    }


def attachment_inventory(
    model: GaussianModel,
    assets_root: Path,
) -> dict[str, Any]:
    state = model.surface_attachment_state_dict(cpu=True)
    arrays = {
        key: {
            "shape": list(state[key].shape),
            "dtype": str(state[key].dtype),
            "sha256": tensor_sha(state[key]),
        }
        for key in SURFACE_LBS_FILES
    }
    manifest = load_surface_attachment_assets(
        assets_root / "surface_lbs",
        expected_count=200000,
    )
    return {
        "arrays": arrays,
        "manifest_sha256": manifest["manifest_sha256"],
        "template_ply_sha256": sha256_file(
            assets_root / "template/template_smplx_body_surface.ply"
        ),
        "template_vertices_sha256": sha256_file(
            assets_root / "template/template_vertices.npy"
        ),
        "template_faces_sha256": sha256_file(
            assets_root / "template/template_faces.npy"
        ),
        "gaussian_count": int(model._xyz.shape[0]),
        "attachment_count": int(state["face_ids"].shape[0]),
        "lbs_shape": list(state["lbs_weights"].shape),
        "attachment_valid_count": int(
            state["attachment_valid"].to(torch.int64).sum().item()
        ),
        "off_surface_count": int(
            torch.count_nonzero(state["surface_distance"]).item()
        ),
    }


def save_rng_state(path: Path) -> dict[str, Any]:
    payload = {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "python": True,
        "numpy": True,
        "torch": True,
        "cuda_device_count": len(payload["cuda_rng_states"]),
        "torch_rng_sha256": tensor_sha(payload["torch_rng_state"]),
        "cuda_rng_sha256": [
            tensor_sha(value) for value in payload["cuda_rng_states"]
        ],
    }


def prepare_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "K": item["K"].float().cuda(),
        "w2c": item["w2c"].float().cuda(),
        "image": item["image"].float().cuda(),
        "mask": item["mask"].bool().cuda(),
        "mask_boundary": item["mask_boundary"].bool().cuda(),
        "pose": item["pose"].float(),
        "Rh": item["Rh"].float(),
        "Th": item["Th"].float(),
        "height": int(item["height"]),
        "width": int(item["width"]),
        "frame_id": int(item["frame_id"]),
        "cam_id": int(item["cam_id"]),
    }


def set_model_pose(model: GaussianModel, item: dict[str, Any]) -> None:
    model.smpl_poses = item["pose"]
    model.Rh = item["Rh"]
    model.Th = item["Th"]


def zero_step_losses(
    model: GaussianModel,
    item: dict[str, Any],
    args,
) -> dict[str, float]:
    cuda_state = torch.cuda.get_rng_state_all()
    torch_state = torch.get_rng_state()
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        cam = prepare_item(item)
        set_model_pose(model, cam)
        background = torch.rand(3, device="cuda")
        with torch.no_grad():
            image, _, _ = model.render(cam, background=background)
            image = torch.clamp(image, 0.0, 1.0)
            image_gt = cam["image"].clone()
            image_gt[~cam["mask"]] = background
            image_gt[cam["mask_boundary"]] = background
            image[cam["mask_boundary"]] = background
            l1_value = l1_loss(image, image_gt)
            smooth = (
                dxyz_smooth_loss(model) * args.lambda_dxyz_smooth
            )
            scaling = (
                gaussian_scaling_loss(
                    model.get_cano_scaling,
                    args.scaling_threshold,
                )
                * args.lambda_scaling
            )
            total = l1_value + smooth + scaling
        return {
            "total_loss": float(total.item()),
            "l1_loss": float(l1_value.item()),
            "lpips_loss": 0.0,
            "dxyz_smooth_loss": float(smooth.item()),
            "scaling_loss": float(scaling.item()),
        }
    finally:
        torch.cuda.set_rng_state_all(cuda_state)
        torch.set_rng_state(torch_state)
        random.setstate(random_state)
        np.random.set_state(numpy_state)


def make_lpips_preserving_rng():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all()
    error = None
    metric = None
    try:
        from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

        metric = LearnedPerceptualImagePatchSimilarity(
            net_type="vgg",
            normalize=True,
        ).cuda()
        metric.eval()
        for parameter in metric.parameters():
            parameter.requires_grad = False
    except Exception as exc:  # environment support is explicitly optional
        error = f"{type(exc).__name__}: {exc}"
        metric = None
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        torch.cuda.set_rng_state_all(cuda_state)
    return metric, error


def boundary_f_score(
    predicted: np.ndarray,
    target: np.ndarray,
    tolerance: int = 3,
) -> float:
    kernel = np.ones((3, 3), np.uint8)
    pred_u8 = predicted.astype(np.uint8)
    target_u8 = target.astype(np.uint8)
    pred_boundary = cv.morphologyEx(pred_u8, cv.MORPH_GRADIENT, kernel) > 0
    target_boundary = (
        cv.morphologyEx(target_u8, cv.MORPH_GRADIENT, kernel) > 0
    )
    if not pred_boundary.any() and not target_boundary.any():
        return 1.0
    dilation_kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE,
        (2 * tolerance + 1, 2 * tolerance + 1),
    )
    target_dilated = cv.dilate(
        target_boundary.astype(np.uint8),
        dilation_kernel,
    ) > 0
    pred_dilated = cv.dilate(
        pred_boundary.astype(np.uint8),
        dilation_kernel,
    ) > 0
    precision = float(
        (pred_boundary & target_dilated).sum()
        / max(1, pred_boundary.sum())
    )
    recall = float(
        (target_boundary & pred_dilated).sum()
        / max(1, target_boundary.sum())
    )
    return (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )


def depth_visual(depth: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    result = np.zeros(depth.shape, dtype=np.uint8)
    valid = np.isfinite(depth) & (alpha > 1.0e-4)
    if valid.any():
        values = depth[valid]
        low, high = np.percentile(values, [1.0, 99.0])
        if high <= low:
            high = low + 1.0
        normalized = np.clip((depth - low) / (high - low), 0.0, 1.0)
        result[valid] = (normalized[valid] * 255.0 + 0.5).astype(np.uint8)
    return result


def labeled_panel(image: np.ndarray, label: str, height: int = 320) -> np.ndarray:
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    image = np.ascontiguousarray(image)
    width = max(1, int(round(image.shape[1] * height / image.shape[0])))
    resized = cv.resize(image, (width, height), interpolation=cv.INTER_AREA)
    bar = np.zeros((34, width, 3), dtype=np.uint8)
    cv.putText(
        bar,
        label,
        (8, 24),
        cv.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )
    return np.concatenate([bar, resized], axis=0)


def save_visual(
    path: Path,
    *,
    gt: np.ndarray,
    mask: np.ndarray,
    predicted: np.ndarray,
    alpha: np.ndarray,
    depth: np.ndarray,
    title: str,
) -> None:
    gt_u8 = (np.clip(gt, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    pred_u8 = (
        np.clip(predicted, 0.0, 1.0) * 255.0 + 0.5
    ).astype(np.uint8)
    mask_u8 = mask.astype(np.uint8) * 255
    alpha_u8 = (
        np.clip(alpha, 0.0, 1.0) * 255.0 + 0.5
    ).astype(np.uint8)
    depth_u8 = depth_visual(depth, alpha)
    panels = [
        labeled_panel(gt_u8, "GT RGB"),
        labeled_panel(mask_u8, "GT mask"),
        labeled_panel(pred_u8, "Pred RGB"),
        labeled_panel(alpha_u8, "Pred alpha"),
        labeled_panel(depth_u8, "Pred depth"),
    ]
    body = np.concatenate(panels, axis=1)
    header = np.zeros((46, body.shape[1], 3), dtype=np.uint8)
    cv.putText(
        header,
        title,
        (12, 31),
        cv.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, np.concatenate([header, body], axis=0))


def aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = [
        "rgb_mae",
        "psnr",
        "ssim",
        "silhouette_iou",
        "boundary_f_score",
        "alpha_occupancy",
        "depth_finite_ratio",
        "foreground_coverage",
        "render_seconds",
        "peak_vram_bytes",
    ]
    if all(record["lpips"] is not None for record in records):
        metric_names.append("lpips")
    result = {}
    for name in metric_names:
        values = [float(record[name]) for record in records]
        result[name] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return result


@torch.no_grad()
def evaluate_queries(
    model: GaussianModel,
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
    dataset = ThumanDataset(
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
            f"evaluation dataset differs: missing={sorted(expected - set(mapping))}"
        )
    background = torch.ones(3, dtype=torch.float32, device="cuda")
    records = []
    smoke_arrays: dict[str, np.ndarray] = {}
    all_warnings: list[str] = []
    for query in queries:
        pose_id = int(query["pose_id"])
        camera_id = int(query["camera_id"])
        item = prepare_item(dataset[mapping[(pose_id, camera_id)]])
        set_model_pose(model, item)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            predicted, alpha, depth, info = model.render(
                item,
                background=background,
                return_depth=True,
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
        extent = (
            posed_xyz.max(dim=0).values - posed_xyz.min(dim=0).values
        )
        means2d = info.get("means2d")
        ordinal = int(query["ordinal"])
        visual_path = (
            attempt_root
            / "visuals"
            / visual_subdir
            / f"query_{ordinal:03d}.png"
        )
        title = (
            f"step={step} quadrant={query['quadrant']} "
            f"pose={pose_id} camera={camera_id}"
        )
        save_visual(
            visual_path,
            gt=gt_np,
            mask=mask_np,
            predicted=pred_np,
            alpha=alpha_np,
            depth=depth_np,
            title=title,
        )
        key = f"query_{ordinal:03d}"
        if smoke_ordinals is not None and ordinal in smoke_ordinals:
            smoke_arrays[f"{key}_rgb"] = pred_np
            smoke_arrays[f"{key}_alpha"] = alpha_np
            smoke_arrays[f"{key}_depth"] = depth_np
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
                "boundary_f_score": boundary_f_score(pred_mask, mask_np),
                "alpha_occupancy": float((alpha_np > 1.0e-4).mean()),
                "depth_finite_ratio": float(np.isfinite(depth_np).mean()),
                "foreground_coverage": float(pred_mask.mean()),
                "render_seconds": render_seconds,
                "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
                "rgb_finite": bool(np.isfinite(metric_pred_np).all()),
                "alpha_finite": bool(np.isfinite(alpha_np).all()),
                "depth_finite": bool(np.isfinite(depth_np).all()),
                "alpha_nonempty": bool((alpha_np > 1.0e-6).any()),
                "mask_nonempty": bool(mask_np.any()),
                "posed_xyz_finite": bool(torch.isfinite(posed_xyz).all().item()),
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
    by_quadrant = {}
    for quadrant in bundle["evaluation"]["quadrant_contract_order"]:
        subset = [item for item in records if item["quadrant"] == quadrant]
        by_quadrant[quadrant] = {
            "query_count": len(subset),
            "metrics": aggregate_metrics(subset),
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
        "schema_version": "subject00.canary.fixed_evaluation.v1",
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
        "records": records,
        "quadrants": by_quadrant,
        "metrics_all": aggregate_metrics(records),
    }
    return result, smoke_arrays


def diagnostic_queries(
    evaluation: dict[str, Any],
) -> list[dict[str, Any]]:
    result = []
    for quadrant in evaluation["quadrant_contract_order"]:
        subset = [
            item
            for item in evaluation["queries"]
            if item["quadrant"] == quadrant
        ]
        result.extend([subset[0], subset[-1]])
    if len(result) != 8:
        raise RuntimeError("diagnostic query selection is not eight")
    return result


def environment_snapshot() -> dict[str, Any]:
    gpu_query = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu="
            "name,driver_version,memory.total,memory.used,memory.free,"
            "utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_query": gpu_query,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "working_directory": str(Path.cwd()),
    }


def build_contact_sheets(
    visual_paths: list[str],
    output: Path,
    *,
    per_sheet: int = 8,
) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    sheets = []
    for start in range(0, len(visual_paths), per_sheet):
        images = [iio.imread(path) for path in visual_paths[start:start + per_sheet]]
        target_width = 1600
        resized = []
        for image in images:
            target_height = max(
                1,
                int(round(image.shape[0] * target_width / image.shape[1])),
            )
            resized.append(
                cv.resize(
                    image,
                    (target_width, target_height),
                    interpolation=cv.INTER_AREA,
                )
            )
        cell_height = max(image.shape[0] for image in resized)
        while len(resized) < per_sheet:
            resized.append(
                np.zeros((cell_height, target_width, 3), dtype=np.uint8)
            )
        cells = []
        for image in resized:
            if image.shape[0] < cell_height:
                pad = np.zeros(
                    (cell_height - image.shape[0], target_width, 3),
                    dtype=np.uint8,
                )
                image = np.concatenate([image, pad], axis=0)
            cells.append(image)
        rows = [
            np.concatenate(cells[index:index + 2], axis=1)
            for index in range(0, per_sheet, 2)
        ]
        sheet = np.concatenate(rows, axis=0)
        path = output / f"sheet_{start // per_sheet + 1:02d}.jpg"
        iio.imwrite(path, sheet, quality=92)
        sheets.append(str(path))
    return sheets


def optimizer_audit(
    model: GaussianModel,
    contract: dict[str, Any],
    scene_scale: float,
) -> dict[str, Any]:
    if tuple(model.optimizers) != EXPECTED_OPTIMIZER_GROUPS:
        raise RuntimeError(
            f"optimizer groups changed: {tuple(model.optimizers)}"
        )
    memberships: dict[int, list[str]] = {}
    group_records = {}
    for group_name, optimizer in model.optimizers.items():
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        for parameter in parameters:
            memberships.setdefault(id(parameter), []).append(group_name)
        first = optimizer.param_groups[0]
        group_records[group_name] = {
            "class": type(optimizer).__name__,
            "parameter_tensor_count": len(parameters),
            "parameter_scalar_count": sum(
                int(parameter.numel()) for parameter in parameters
            ),
            "lr": float(first["lr"]),
            "betas": [float(value) for value in first["betas"]],
            "eps": float(first["eps"]),
            "weight_decay": float(first["weight_decay"]),
        }
    duplicates = {
        str(identity): groups
        for identity, groups in memberships.items()
        if len(groups) != 1
    }
    trainables = trainable_parameters(model)
    expected_ids = {id(value) for value in trainables.values()}
    actual_ids = set(memberships)
    omitted = sorted(
        name for name, value in trainables.items() if id(value) not in actual_ids
    )
    excluded = {
        "template_vertices": None,
        "template_faces": None,
        "_xyz": model._xyz,
        "_weights": model.get_weights,
        "surface_face_ids": model.surface_attachment["face_ids"],
        "surface_barycentric": model.surface_attachment["barycentric"],
        "surface_component_ids": model.surface_attachment["component_ids"],
        "surface_semantic_region_ids": model.surface_attachment[
            "semantic_region_ids"
        ],
        "surface_attachment_valid": model.surface_attachment[
            "attachment_valid"
        ],
        "surface_cached_lbs": model.surface_attachment["lbs_weights"],
    }
    excluded_present = sorted(
        name
        for name, value in excluded.items()
        if value is not None and id(value) in actual_ids
    )
    expected_lr = {
        "dxyz": 0.00016 * scene_scale,
        "scales": 0.0005,
        "quats": 0.0005,
        "opacities": 0.0005,
        "sh0": 0.0005,
        "shN": 0.000025,
        "dxyz_bs": 0.00016 * scene_scale / 10.0,
        "dscales_bs": 0.0001,
        "dquats_bs": 0.0001,
        "dopacities_bs": 0.0001,
        "dsh0_bs": 0.0001,
        "dshN_bs": 0.0000025,
        "encoder_feat_params": 0.0005,
        "xyz_offset": 0.001,
    }
    lr_mismatches = {
        name: {
            "actual": group_records[name]["lr"],
            "expected": expected,
        }
        for name, expected in expected_lr.items()
        if not math.isclose(
            group_records[name]["lr"],
            expected,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
    }
    contract_groups = contract["subject02_formal_training_contract"][
        "optimizer"
    ]["groups"]
    if len(contract_groups) != 14:
        raise RuntimeError("persisted optimizer contract is not 14 groups")
    passed = (
        not duplicates
        and not omitted
        and not excluded_present
        and not lr_mismatches
        and actual_ids == expected_ids
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "scene_scale": float(scene_scale),
        "group_count": len(group_records),
        "groups": group_records,
        "duplicate_parameter_memberships": duplicates,
        "omitted_trainable_parameters": omitted,
        "excluded_state_in_optimizer": excluded_present,
        "lr_mismatches": lr_mismatches,
        "trainable_parameter_tensor_count": len(trainables),
        "optimizer_parameter_tensor_count": len(actual_ids),
        "scheduler_count": len(model.schedulers),
        "scheduler_horizon": 800000,
    }
    if not passed:
        raise RuntimeError(f"optimizer membership audit failed: {result}")
    return result


def optimizer_state_finite(model: GaussianModel) -> bool:
    for optimizer in model.optimizers.values():
        for state in optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    if not torch.isfinite(value).all().item():
                        return False
    return True


def group_norms(model: GaussianModel) -> tuple[dict[str, float], dict[str, float]]:
    gradient = {}
    parameter = {}
    for name, optimizer in model.optimizers.items():
        params = [
            item
            for group in optimizer.param_groups
            for item in group["params"]
        ]
        param_sq = torch.zeros((), device="cuda")
        grad_sq = torch.zeros((), device="cuda")
        for value in params:
            param_sq = param_sq + torch.sum(value.detach() ** 2)
            if value.grad is not None:
                grad_sq = grad_sq + torch.sum(value.grad.detach() ** 2)
        parameter[name] = float(torch.sqrt(param_sq).item())
        gradient[name] = float(torch.sqrt(grad_sq).item())
    return gradient, parameter


def train_step(
    model: GaussianModel,
    item: dict[str, Any],
    args,
    step: int,
) -> dict[str, Any]:
    cam = prepare_item(item)
    set_model_pose(model, cam)
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
            model.get_cano_scaling,
            args.scaling_threshold,
        )
        * args.lambda_scaling
    )
    lpips_value = torch.zeros((), device="cuda")
    total = l1_value + lpips_value + smooth + scaling
    finite_loss = bool(torch.isfinite(total).item())
    if not finite_loss:
        raise FloatingPointError(f"non-finite loss at step {step}")
    total.backward()
    gradient_norms, parameter_norms = group_norms(model)
    gradients_finite = all(math.isfinite(value) for value in gradient_norms.values())
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
    if not optimizer_state_finite(model):
        raise FloatingPointError(f"optimizer state non-finite at step {step}")
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - started
    xyz = model.get_xyz.detach()
    opacity = model.compute_opacity().detach()
    scale = model.get_cano_scaling.detach()
    means2d = info.get("means2d")
    warning_messages = [str(entry.message) for entry in caught]
    return {
        "step": step,
        "pose_id": cam["frame_id"],
        "camera_id": cam["cam_id"],
        "total_loss": float(total.item()),
        "l1_loss": float(l1_value.item()),
        "lpips_loss": 0.0,
        "dxyz_smooth_loss": float(smooth.item()),
        "scaling_loss": float(scaling.item()),
        "learning_rates": {
            name: float(optimizer.param_groups[0]["lr"])
            for name, optimizer in model.optimizers.items()
        },
        "gradient_norms": gradient_norms,
        "parameter_norms": parameter_norms,
        "nonzero_gradient_groups": nonzero_groups,
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
        "wall_seconds": wall_seconds,
    }


def save_checkpoint(
    path: Path,
    model: GaussianModel,
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
            "checkpoint_kind": "SUBJECT00_SHORT_CANARY_TRAINING",
            "checkpoint_version": 3,
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
            "repaired_contract_hashes": contract_hashes,
            "training_data_order_sha256": TRAIN_ORDER_SHA,
            "evaluation_query_order_sha256": EVAL_ORDER_SHA,
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
    }


def preflight_phase(args) -> int:
    if git_head() != SOURCE_HEAD:
        raise RuntimeError(
            f"preflight must start at repaired HEAD {SOURCE_HEAD}; "
            f"got {git_head()}"
        )
    if git_branch() != TARGET_BRANCH:
        raise RuntimeError(f"unexpected execution branch: {git_branch()}")
    bundle = load_contract_bundle(args.availability_manifest)
    for directory in (
        "contract",
        "snapshots",
        "checkpoints",
        "training_logs",
        "evaluations",
        "renders",
        "visuals",
        "audits",
    ):
        (args.attempt_root / directory).mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model, scene, runtime_args = build_model(
        bundle,
        args.data_root,
        args.assets_root,
        args.attempt_root,
    )
    inventory = parameter_inventory(model)
    attachments = attachment_inventory(model, args.assets_root)
    rng = save_rng_state(
        args.attempt_root / "snapshots/pre_result_rng_state.pt"
    )
    zero_losses = zero_step_losses(model, scene.trainset[0], runtime_args)
    lpips_metric, lpips_error = make_lpips_preserving_rng()
    evaluation, _ = evaluate_queries(
        model,
        bundle,
        args.data_root,
        args.attempt_root,
        step=0,
        queries=bundle["evaluation"]["queries"],
        lpips_metric=lpips_metric,
        lpips_error=lpips_error,
        visual_subdir="step_000000",
    )
    evaluation_path = (
        args.attempt_root / "evaluations/step_000000_results.json"
    )
    write_json(evaluation_path, evaluation)
    visual_paths = [item["visual_path"] for item in evaluation["records"]]
    contact_sheets = build_contact_sheets(
        visual_paths,
        args.attempt_root / "visuals/step_000000_contact_sheets",
    )
    diagnostic = diagnostic_queries(bundle["evaluation"])
    snapshot = {
        "schema_version": "subject00.canary.pre_result_snapshot.v1",
        "task_id": TASK_ID,
        "status": "PASS",
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "execution_branch": TARGET_BRANCH,
        "execution_head_before_pre_result_commit": git_head(),
        "contract_hashes": bundle["hashes"],
        "environment": environment_snapshot(),
        "exact_runtime_config": OmegaConf.to_container(
            runtime_args,
            resolve=True,
        ),
        "record_manifest": {
            "path": str(
                REPO_ROOT
                / "paper_protocol/second_identity/"
                "subject00_canary_record_manifest.json"
            ),
            "count": 384,
            "sha256": TRAIN_ORDER_SHA,
        },
        "evaluation_manifest": {
            "path": str(
                REPO_ROOT
                / "paper_protocol/second_identity/"
                "subject00_canary_evaluation_manifest.json"
            ),
            "count": 96,
            "sha256": EVAL_ORDER_SHA,
            "diagnostic_query_ordinals": [
                int(item["ordinal"]) for item in diagnostic
            ],
        },
        "model_inventory": inventory,
        "attachment_inventory": attachments,
        "rng_state": rng,
        "zero_step_losses": zero_losses,
        "step0_evaluation": {
            "path": str(evaluation_path),
            "sha256": sha256_file(evaluation_path),
            "query_count": evaluation["query_count"],
            "successful_render_count": evaluation[
                "successful_render_count"
            ],
            "failed_render_count": evaluation["failed_render_count"],
            "lpips_available": evaluation["lpips_available"],
            "lpips_unavailable_reason": evaluation[
                "lpips_unavailable_reason"
            ],
            "primary_image_error": evaluation["primary_image_error"],
            "quadrants": evaluation["quadrants"],
            "visual_count": len(visual_paths),
            "contact_sheet_count": len(contact_sheets),
            "contact_sheets": contact_sheets,
        },
        "runtime_lbs_counters": model.runtime_lbs_counters,
        "topology_call_counts": {
            "clone": 0,
            "split": 0,
            "densification": 0,
            "topology_changing_prune": 0,
            "off_surface_rebind": 0,
        },
        "counters": {
            "optimizer_created": 0,
            "training_forwards": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "checkpoint_writes": 0,
            "render_calls": 96,
            "PAPER_FINAL": 0,
        },
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "wall_seconds": time.perf_counter() - started,
    }
    snapshot_path = (
        args.attempt_root / "snapshots/pre_result_snapshot.json"
    )
    write_json(snapshot_path, snapshot)
    execution_contract = {
        "schema_version": "subject00.canary.execution_contract.v1",
        "task_id": TASK_ID,
        "classification": "PRE_RESULT_EXECUTION_CONTRACT_FROZEN",
        "source": {"branch": SOURCE_BRANCH, "head": SOURCE_HEAD},
        "target_branch": TARGET_BRANCH,
        "training_source_head_resolution": (
            "git rev-parse HEAD after the pre-result execution commit"
        ),
        "repaired_contract_hashes": bundle["hashes"],
        "train_record_count": 384,
        "train_data_order_sha256": TRAIN_ORDER_SHA,
        "evaluation_query_count": 96,
        "evaluation_query_order_sha256": EVAL_ORDER_SHA,
        "diagnostic_query_ordinals": [
            int(item["ordinal"]) for item in diagnostic
        ],
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "formal_evaluation_steps": list(FORMAL_EVAL_STEPS),
        "diagnostic_steps": list(DIAGNOSTIC_STEPS),
        "budget": {
            "training_runs": 1,
            "seed": 0,
            "batch_size": 1,
            "training_forwards": 384,
            "backward_calls": 384,
            "optimizer_steps": 384,
            "evaluation_renders": 216,
            "main_visual_reviews": 192,
        },
        "fixed_policies": {
            "training_background": (
                "torch.rand(3, device='cuda') once per training batch"
            ),
            "evaluation_background": [1.0, 1.0, 1.0],
            "lpips_input": (
                "foreground square crop resized to 512, matching formal loss"
            ),
            "primary_image_error": evaluation["primary_image_error"],
            "silhouette_threshold": 0.5,
            "boundary_f_tolerance_pixels": 3,
            "visual_composite_height_pixels": 320,
        },
        "pre_result_snapshot": {
            "external_path": str(snapshot_path),
            "external_sha256": sha256_file(snapshot_path),
            "model_inventory": inventory,
            "attachment_inventory": attachments,
            "rng_state": rng,
            "zero_step_losses": zero_losses,
            "step0_evaluation": snapshot["step0_evaluation"],
            "runtime_lbs_counters": model.runtime_lbs_counters,
        },
        "authorization_boundary": {
            "optimizer_created": 0,
            "training_forwards": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "checkpoint_writes": 0,
            "training_authorized_before_this_contract_is_pushed": False,
        },
        "PAPER_FINAL": 0,
    }
    write_json(args.tracked_execution_contract, execution_contract)
    scene.tb_writer.close()
    print(
        json.dumps(
            {
                "phase": "preflight",
                "status": "PASS",
                "snapshot": str(snapshot_path),
                "execution_contract": str(args.tracked_execution_contract),
                "step0_renders": 96,
                "optimizer_created": 0,
            },
            sort_keys=True,
        )
    )
    return 0


def training_phase(args) -> int:
    if git_branch() != TARGET_BRANCH:
        raise RuntimeError(f"unexpected execution branch: {git_branch()}")
    execution_head = git_head()
    if execution_head == SOURCE_HEAD:
        raise RuntimeError("pre-result execution contract was not committed")
    bundle = load_contract_bundle(args.availability_manifest)
    execution_contract = json.loads(
        args.tracked_execution_contract.read_text(encoding="utf-8")
    )
    snapshot_path = (
        args.attempt_root / "snapshots/pre_result_snapshot.json"
    )
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if (
        execution_contract["pre_result_snapshot"]["external_sha256"]
        != sha256_file(snapshot_path)
    ):
        raise RuntimeError("pre-result snapshot hash changed before training")
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model, scene, runtime_args = build_model(
        bundle,
        args.data_root,
        args.assets_root,
        args.attempt_root,
    )
    initial_inventory = parameter_inventory(model)
    if (
        initial_inventory
        != snapshot["model_inventory"]
    ):
        raise RuntimeError("initial model inventory/hash differs from preflight")
    initial_attachments = attachment_inventory(model, args.assets_root)
    if initial_attachments != snapshot["attachment_inventory"]:
        raise RuntimeError("initial attachment/template state changed")
    current_cuda_rng = [
        tensor_sha(value) for value in torch.cuda.get_rng_state_all()
    ]
    if current_cuda_rng != snapshot["rng_state"]["cuda_rng_sha256"]:
        raise RuntimeError("initial CUDA RNG differs from preflight")
    resume_step = int(args.resume_step)
    training_log_path = (
        args.attempt_root / "training_logs/step_records.jsonl"
    )
    checkpoints = []
    training_records: list[dict[str, Any]] = []
    execution_source_heads = [execution_head]
    if resume_step:
        resume_checkpoint = (
            args.attempt_root
            / f"checkpoints/step_{resume_step:06d}.pth"
        )
        if not resume_checkpoint.exists():
            raise RuntimeError(
                f"resume checkpoint is absent: {resume_checkpoint}"
            )
        payload = torch.load(resume_checkpoint, weights_only=False)
        if (
            int(payload["training_step"]) != resume_step
            or int(payload["data_order_position"]) != resume_step
        ):
            raise RuntimeError(
                "resume checkpoint step/data-order position mismatch"
            )
        if payload["training_data_order_sha256"] != TRAIN_ORDER_SHA:
            raise RuntimeError("resume checkpoint data-order SHA changed")
        if not math.isclose(
            float(payload["scene_scale"]),
            float(scene.scene_scale),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError("resume checkpoint scene scale changed")
        model.restore(payload)
        model.training_setup(runtime_args, scene.scene_scale)
        if set(payload["optimizer_states"]) != set(model.optimizers):
            raise RuntimeError("resume optimizer group names changed")
        for name, optimizer in model.optimizers.items():
            optimizer.load_state_dict(payload["optimizer_states"][name])
        if len(payload["scheduler_states"]) != len(model.schedulers):
            raise RuntimeError("resume scheduler count changed")
        for scheduler, state in zip(
            model.schedulers,
            payload["scheduler_states"],
        ):
            scheduler.load_state_dict(state)
        random.setstate(payload["python_rng_state"])
        np.random.set_state(payload["numpy_rng_state"])
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all(
            [value.cpu() for value in payload["cuda_rng_states"]]
        )
        if not training_log_path.exists():
            raise RuntimeError("resume training log is absent")
        training_records = [
            json.loads(line)
            for line in training_log_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        if len(training_records) != resume_step:
            raise RuntimeError(
                f"resume log has {len(training_records)} records, "
                f"expected {resume_step}"
            )
        for index, record in enumerate(training_records):
            expected_record = bundle["records"]["records"][index]
            if (
                int(record["step"]) != index + 1
                or int(record["pose_id"]) != int(expected_record["pose_id"])
                or int(record["camera_id"])
                != int(expected_record["camera_id"])
            ):
                raise RuntimeError(
                    f"resume log data order mismatch at index {index}"
                )
        execution_source_heads = [
            str(payload["source_head"]),
            execution_head,
        ]
        for step in CHECKPOINT_STEPS:
            if step > resume_step:
                continue
            path = (
                args.attempt_root
                / f"checkpoints/step_{step:06d}.pth"
            )
            if not path.exists():
                raise RuntimeError(
                    f"historical checkpoint missing during resume: {path}"
                )
            checkpoints.append(
                {
                    "path": str(path),
                    "step": step,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "data_order_position": step,
                }
            )
    else:
        if training_log_path.exists():
            raise RuntimeError(
                "training log already exists; refusing a result-driven rerun"
            )
        model.training_setup(runtime_args, scene.scene_scale)
        checkpoints.append(
            save_checkpoint(
                args.attempt_root / "checkpoints/step_000000.pth",
                model,
                step=0,
                data_order_position=0,
                scene_scale=scene.scene_scale,
                execution_source_head=execution_head,
                contract_hashes=bundle["hashes"],
            )
        )
    optimizer_audit_result = optimizer_audit(
        model,
        bundle["contract"],
        scene.scene_scale,
    )
    write_json(
        args.attempt_root / "audits/optimizer_membership.json",
        optimizer_audit_result,
    )
    lpips_metric, lpips_error = make_lpips_preserving_rng()
    diagnostic = diagnostic_queries(bundle["evaluation"])
    diagnostic_reports = {}
    for step in DIAGNOSTIC_STEPS:
        if step > resume_step:
            continue
        diagnostic_path = (
            args.attempt_root
            / f"evaluations/diagnostic_step_{step:06d}.json"
        )
        if diagnostic_path.exists():
            report = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        elif step == resume_step:
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
            write_json(diagnostic_path, report)
        else:
            raise RuntimeError(
                f"completed diagnostic report is missing: {diagnostic_path}"
            )
        diagnostic_reports[str(step)] = {
            "path": str(diagnostic_path),
            "sha256": sha256_file(diagnostic_path),
            "query_count": report["query_count"],
            "successful_render_count": report[
                "successful_render_count"
            ],
        }
    warning_set: set[str] = {
        message
        for record in training_records
        for message in record["warning_messages"]
    }
    for step, manifest_record in enumerate(
        bundle["records"]["records"],
        start=1,
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
        with training_log_path.open("a", encoding="utf-8", newline="\n") as handle:
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
            diagnostic_path = (
                args.attempt_root
                / f"evaluations/diagnostic_step_{step:06d}.json"
            )
            write_json(diagnostic_path, report)
            diagnostic_reports[str(step)] = {
                "path": str(diagnostic_path),
                "sha256": sha256_file(diagnostic_path),
                "query_count": report["query_count"],
                "successful_render_count": report[
                    "successful_render_count"
                ],
            }
    diagnostic_ordinals = {
        int(item["ordinal"]) for item in diagnostic
    }
    final_evaluation, smoke_arrays = evaluate_queries(
        model,
        bundle,
        args.data_root,
        args.attempt_root,
        step=384,
        queries=bundle["evaluation"]["queries"],
        lpips_metric=lpips_metric,
        lpips_error=lpips_error,
        visual_subdir="step_000384",
        smoke_ordinals=diagnostic_ordinals,
    )
    final_eval_path = (
        args.attempt_root / "evaluations/step_000384_results.json"
    )
    write_json(final_eval_path, final_evaluation)
    smoke_path = (
        args.attempt_root / "evaluations/step_000384_roundtrip_baseline.npz"
    )
    np.savez(smoke_path, **{key: smoke_arrays[key] for key in sorted(smoke_arrays)})
    checkpoints.append(
        save_checkpoint(
            args.attempt_root / "checkpoints/step_000384.pth",
            model,
            step=384,
            data_order_position=384,
            scene_scale=scene.scene_scale,
            execution_source_head=execution_head,
            contract_hashes=bundle["hashes"],
        )
    )
    post_attachments = attachment_inventory(model, args.assets_root)
    frozen_unchanged = (
        post_attachments == snapshot["attachment_inventory"]
    )
    if not frozen_unchanged:
        raise RuntimeError("surface attachment/template state changed")
    visual_paths = [
        item["visual_path"] for item in final_evaluation["records"]
    ]
    contact_sheets = build_contact_sheets(
        visual_paths,
        args.attempt_root / "visuals/step_000384_contact_sheets",
    )
    losses = [item["total_loss"] for item in training_records]
    primary = [item["l1_loss"] for item in training_records]
    first48_total = float(np.median(losses[:48]))
    last48_total = float(np.median(losses[-48:]))
    first48_primary = float(np.median(primary[:48]))
    last48_primary = float(np.median(primary[-48:]))
    training_result = {
        "schema_version": "subject00.canary.training_result.v1",
        "task_id": TASK_ID,
        "status": "TRAINING_COMPLETE_PENDING_ROUNDTRIP_AND_VISUAL_REVIEW",
        "execution_source_head": execution_head,
        "execution_source_heads": execution_source_heads,
        "training_runs": 1,
        "process_segments": 2 if resume_step else 1,
        "infrastructure_resume_from_step": (
            resume_step if resume_step else None
        ),
        "repeated_training_record_count": 0,
        "optimizer_created": 1,
        "training_forward_batches": 384,
        "backward_calls": 384,
        "optimizer_steps": 384,
        "checkpoint_writes": 5,
        "data_order_sha256": TRAIN_ORDER_SHA,
        "record_count": len(training_records),
        "unique_exposure_count": len(
            {(item["pose_id"], item["camera_id"]) for item in training_records}
        ),
        "loss_finite_all": all(item["loss_finite"] for item in training_records),
        "gradient_finite_all": all(
            item["gradients_finite"] for item in training_records
        ),
        "optimizer_state_finite_all": all(
            item["optimizer_state_finite"] for item in training_records
        ),
        "intended_gradient_nonzero_all": all(
            bool(item["nonzero_gradient_groups"]) for item in training_records
        ),
        "total_loss": {
            "first48_median": first48_total,
            "last48_median": last48_total,
            "relative_reduction": (
                (first48_total - last48_total) / first48_total
            ),
            "gate_reduction_at_least_5_percent": (
                last48_total <= 0.95 * first48_total
            ),
        },
        "primary_reconstruction_loss": {
            "name": "l1_loss",
            "first48_median": first48_primary,
            "last48_median": last48_primary,
            "relative_reduction": (
                (first48_primary - last48_primary) / first48_primary
            ),
            "gate_last_below_first": last48_primary < first48_primary,
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
        "step384_evaluation": {
            "path": str(final_eval_path),
            "sha256": sha256_file(final_eval_path),
            "successful_render_count": final_evaluation[
                "successful_render_count"
            ],
            "failed_render_count": final_evaluation["failed_render_count"],
            "contact_sheets": contact_sheets,
        },
        "roundtrip_baseline": {
            "path": str(smoke_path),
            "sha256": sha256_file(smoke_path),
            "query_count": 8,
            "array_count": len(smoke_arrays),
            "note": (
                "Eight baseline queries reuse arrays from the formal "
                "96-query step384 evaluation; fresh-restore renders are "
                "counted separately from the 216 evaluation renders."
            ),
        },
        "checkpoints": checkpoints,
        "attachment_template_lbs_unchanged": frozen_unchanged,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(
        args.attempt_root / "training_logs/training_result.json",
        training_result,
    )
    write_json(
        args.attempt_root / "audits/post_training_attachment_inventory.json",
        post_attachments,
    )
    scene.tb_writer.close()
    print(
        json.dumps(
            {
                "phase": "train",
                "status": "PASS",
                "steps": 384,
                "checkpoints": 5,
                "evaluation_renders": 216,
                "roundtrip_baseline_renders": 8,
            },
            sort_keys=True,
        )
    )
    return 0


def roundtrip_phase(args) -> int:
    bundle = load_contract_bundle(args.availability_manifest)
    checkpoint_path = (
        args.attempt_root / "checkpoints/step_000384.pth"
    )
    baseline_path = (
        args.attempt_root / "evaluations/step_000384_roundtrip_baseline.npz"
    )
    payload = torch.load(checkpoint_path, weights_only=False)
    if int(payload["training_step"]) != 384:
        raise RuntimeError("final checkpoint step is not 384")
    init_smpl(
        str(
            REPO_ROOT / "smpl_model/smplx/SMPLX_NEUTRAL.npz"
        )
        if (REPO_ROOT / "smpl_model/smplx/SMPLX_NEUTRAL.npz").exists()
        else str(
            Path(
                "/root/autodl-tmp/canondressgs_work/mmlphuman_code/"
                "smpl_model/smplx/SMPLX_NEUTRAL.npz"
            )
        )
    )
    model = GaussianModel()
    model.restore(payload)
    runtime_args = build_runtime_args(
        bundle,
        args.data_root,
        args.assets_root,
        args.attempt_root,
    )
    model.training_setup(runtime_args, float(payload["scene_scale"]))
    for name, optimizer in model.optimizers.items():
        optimizer.load_state_dict(payload["optimizer_states"][name])
    for scheduler, state in zip(
        model.schedulers,
        payload["scheduler_states"],
    ):
        scheduler.load_state_dict(state)
    optimizer_audit_result = optimizer_audit(
        model,
        bundle["contract"],
        float(payload["scene_scale"]),
    )
    diagnostic = diagnostic_queries(bundle["evaluation"])
    current: dict[str, np.ndarray] = {}
    records = []
    background = torch.ones(3, device="cuda")
    for query in diagnostic:
        pose_id = int(query["pose_id"])
        camera_id = int(query["camera_id"])
        dataset = ThumanDataset(
            datadir=str(args.data_root),
            frame_ids=[pose_id],
            cam_ids=[camera_id],
            background=np.ones(3, dtype=np.float32),
            image_scaling=1,
            is_in_memory=False,
        )
        item = prepare_item(dataset[0])
        set_model_pose(model, item)
        with torch.no_grad():
            rgb, alpha, depth, _ = model.render(
                item,
                background=background,
                return_depth=True,
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
        "schema_version": "subject00.canary.checkpoint_roundtrip.v1",
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
        "optimizer_state_restored": optimizer_state_finite(model),
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
    }
    if result["status"] != "PASS":
        raise RuntimeError(f"final checkpoint roundtrip failed: {result}")
    write_json(
        args.attempt_root / "audits/checkpoint_roundtrip.json",
        result,
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
    parser.add_argument(
        "--phase",
        choices=("preflight", "train", "roundtrip"),
        required=True,
    )
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
            "SUBJECT00-MMLPHUMAN-SHORT-CANARY-001/attempt_001"
        ),
    )
    parser.add_argument(
        "--tracked-execution-contract",
        type=Path,
        default=(
            REPO_ROOT
            / "paper_protocol/second_identity/"
            "subject00_canary_execution_contract.json"
        ),
    )
    parser.add_argument(
        "--resume-step",
        type=int,
        default=0,
        choices=(0, 96, 192, 288),
        help=(
            "Infrastructure-only exact resume point. The checkpoint step, "
            "data-order position, RNG, optimizer, and scheduler must agree."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.phase == "preflight":
        return preflight_phase(args)
    if args.phase == "train":
        return training_phase(args)
    return roundtrip_phase(args)


if __name__ == "__main__":
    raise SystemExit(main())
