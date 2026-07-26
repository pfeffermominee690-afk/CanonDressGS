#!/usr/bin/env python3
"""Run the one-shot Subject00/O03 Teacher micro-pilot from Base step 60747.

This runner is deliberately fail closed.  It consumes only the eight accepted
O03 raw/mask triples, restores the atomically sealed Subject00 formal Base
step-60747 checkpoint, freezes that Base, and optimizes one shared canonical
six-attribute residual field for exactly 1,200 steps.  It never resumes or
overwrites an existing provisional run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import nn
from torchmetrics.functional.image import structural_similarity_index_measure

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.second_identity import run_subject00_formal_base_101245 as formal_base
from tools.second_identity import run_subject00_formal_strict_split as formal
from tools.second_identity import run_subject00_short_canary as core


TASK_ID = "AAAI27-SUBJECT00-BASE60747-O03-PROVISIONAL-TEACHER-MICROPILOT-001"
BRANCH = "research/subject00-base60747-o03-provisional-teacher-micropilot-20260727"
CONFIG_PATH = (
    REPO_ROOT
    / "configs/research/subject00_o03_provisional_teacher_base60747_v1.json"
)
EXPECTED_LOSS = {
    "garment_rgb": 1.0,
    "alpha_foreground": 0.5,
    "new_silhouette_alpha": 1.0,
    "boundary_rgb": 0.25,
    "protected_rgb": 10.0,
    "protected_alpha": 5.0,
    "stability": 0.0001,
}
SLOTS = tuple(f"slot_{index:02d}" for index in range(8))
CAMERAS = (17, 21, 14, 23, 11, 2, 9, 5)
DIRECTIONS = (
    "front",
    "front-left",
    "front-right",
    "left",
    "right",
    "back-left",
    "back-right",
    "back",
)
CHECKPOINT_STEPS = (0, 300, 600, 900, 1200)
GARMENT_BODY_PARTS = frozenset(
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 16, 17, 18, 19}
)
PROTECTED_BODY_PARTS = frozenset({10, 11, 15, 20, 21, *range(22, 55)})
FORMAL_LBS_JOINT_COUNT = 55
MODEL_KEYS = ("_xyz", "_scaling", "_rotation", "_opacity", "_sh0", "_shN")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    encoded = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write((json.dumps(value, sort_keys=True) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())


def git_output(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), *arguments], text=True
    ).strip()


def tensor_sha(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def model_fingerprint(model: nn.Module) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, parameter in model.named_parameters():
        result[f"parameter:{name}"] = tensor_sha(parameter)
    for name in MODEL_KEYS:
        value = getattr(model, name)
        result[f"canonical:{name}"] = tensor_sha(value)
    result["lbs_weights"] = tensor_sha(model.get_weights)
    return result


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config["task_id"] != TASK_ID:
        raise RuntimeError("task id changed")
    if config["source_git"]["execution_branch"] != BRANCH:
        raise RuntimeError("execution branch changed")
    if config["classification"] != "PROVISIONAL_BASE60747_RESULT":
        raise RuntimeError("provisional result classification changed")
    if config["paper_eligible"] is not False:
        raise RuntimeError("paper eligibility must remain false")
    if config["subject"] != "Subject00" or config["garment"] != "O03":
        raise RuntimeError("subject/garment scope changed")
    if int(config["base"]["step"]) != 60747:
        raise RuntimeError("initialization step changed")
    if int(config["teacher"]["steps"]) != 1200:
        raise RuntimeError("Teacher step budget changed")
    if int(config["teacher"]["seed"]) != 20260718:
        raise RuntimeError("Teacher seed changed")
    if config["teacher"]["loss_name"] != "CAPACITY_ORACLE_LOSS_V1":
        raise RuntimeError("Teacher loss implementation changed")
    if config["teacher"]["loss_weights"] != EXPECTED_LOSS:
        raise RuntimeError("Teacher loss weights changed")
    optimizer = config["teacher"]["optimizer"]
    if (
        optimizer["class"] != "Adam"
        or float(optimizer["geometry_lr"]) != 0.001
        or float(optimizer["appearance_lr"]) != 0.002
        or float(optimizer["gradient_clip_norm"]) != 1.0
    ):
        raise RuntimeError("Teacher optimizer contract changed")
    if tuple(config["teacher"]["checkpoint_steps"]) != CHECKPOINT_STEPS:
        raise RuntimeError("Teacher checkpoint cadence changed")
    if (
        tuple(config["targets"]["slots"]) != SLOTS
        or tuple(config["targets"]["camera_ids"]) != CAMERAS
        or tuple(config["targets"]["directions"]) != DIRECTIONS
    ):
        raise RuntimeError("O03 target mapping changed")
    return config


def require_git_state(config: Mapping[str, Any]) -> dict[str, str]:
    branch = git_output("branch", "--show-current")
    head = git_output("rev-parse", "HEAD")
    status = git_output("status", "--porcelain")
    if branch != BRANCH:
        raise RuntimeError(f"wrong execution branch: {branch}")
    if status:
        raise RuntimeError("execution worktree is dirty")
    launch = config["source_git"]["launch_head"]
    ancestor = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", launch, head]
    ).returncode
    if ancestor != 0:
        raise RuntimeError("execution HEAD is not descended from launch HEAD")
    return {"branch": branch, "head": head, "status": "CLEAN"}


def require_storage(path: Path, minimum_free_bytes: int) -> dict[str, Any]:
    usage = os.statvfs(path.parent)
    free = int(usage.f_bavail * usage.f_frsize)
    total = int(usage.f_blocks * usage.f_frsize)
    if free < int(minimum_free_bytes):
        raise RuntimeError(
            f"storage gate failed: {free} < {int(minimum_free_bytes)}"
        )
    return {
        "path": str(path.parent),
        "free_bytes": free,
        "total_bytes": total,
        "minimum_free_bytes": int(minimum_free_bytes),
        "status": "PASS",
    }


def require_gpu(config: Mapping[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    name = torch.cuda.get_device_name(0)
    if name != config["runtime"]["gpu_name"]:
        raise RuntimeError(f"unexpected GPU: {name}")
    query = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    rows = [line.strip() for line in query.splitlines() if line.strip()]
    other_rows = []
    for row in rows:
        try:
            pid = int(row.split(",", 1)[0].strip())
        except ValueError:
            other_rows.append(row)
            continue
        if pid != os.getpid():
            other_rows.append(row)
    if other_rows:
        raise RuntimeError(f"GPU is not exclusive before O03: {other_rows}")
    return {
        "gpu": name,
        "active_compute_processes": rows,
        "other_compute_processes": other_rows,
        "status": "PASS",
    }


def verify_file(path: Path, expected_sha: str, expected_bytes: int | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size <= 0 or (expected_bytes is not None and size != int(expected_bytes)):
        raise RuntimeError(f"file size mismatch: {path}")
    digest = sha256_file(path)
    if digest != expected_sha:
        raise RuntimeError(f"file SHA mismatch: {path}")
    return {"path": str(path), "bytes": size, "sha256": digest, "status": "PASS"}


def load_mask(path: Path, expected_width: int, expected_height: int) -> torch.Tensor:
    with Image.open(path) as image:
        if image.mode != "L":
            raise RuntimeError(f"mask mode must be L: {path}")
        array = np.asarray(image)
    if array.shape != (expected_height, expected_width):
        raise RuntimeError(f"mask dimensions changed: {path}: {array.shape}")
    values = set(np.unique(array).tolist())
    if not values.issubset({0, 255}) or not values:
        raise RuntimeError(f"mask must be strictly binary: {path}: {values}")
    return torch.from_numpy((array > 0).astype(np.float32).copy())[..., None]


def load_rgb(path: Path, expected_width: int, expected_height: int) -> torch.Tensor:
    with Image.open(path) as image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        if image.size != (expected_width, expected_height):
            raise RuntimeError(
                f"accepted raw dimensions changed: {path}: {image.size}"
            )
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy())


def build_target_registry(
    config: Mapping[str, Any],
    run_root: Path,
    mask_registry_path: Path,
    preflight_draft_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evidence = config["source_evidence"]
    verify_file(
        mask_registry_path, evidence["mask_accepted_registry_sha256"]
    )
    verify_file(
        preflight_draft_path, evidence["teacher_target_preflight_draft_sha256"]
    )
    mask_registry = read_json(mask_registry_path)
    draft = read_json(preflight_draft_path)
    if (
        int(mask_registry["total_mask_accepted_cell_count"]) != 24
        or int(mask_registry["mask_pair_accepted_count"]) != 24
        or int(mask_registry["teacher_target_count"]) != 0
    ):
        raise RuntimeError("mask accepted registry counts changed")
    if (
        mask_registry["teacher_target_creation_authorized"] is not False
        or mask_registry["teacher_endpoint_optimization_authorized"] is not False
    ):
        raise RuntimeError("mask promotion task unexpectedly created Teacher targets")
    mask_rows = {
        (row["garment"], row["slot"]): row for row in mask_registry["records"]
    }
    draft_rows = {
        (row["garment"], row["slot"]): row for row in draft["records"]
    }
    records: list[dict[str, Any]] = []
    loaded: list[dict[str, Any]] = []
    for index, (slot, camera, direction) in enumerate(
        zip(SLOTS, CAMERAS, DIRECTIONS, strict=True)
    ):
        key = ("O03", slot)
        if key not in mask_rows or key not in draft_rows:
            raise RuntimeError(f"missing O03 target evidence: {key}")
        source = mask_rows[key]
        binding = draft_rows[key]
        if (
            source["camera"] != f"cam{camera:02d}"
            or source["direction"] != direction
            or int(binding["camera_pose_binding"]["camera_id"]) != camera
            or int(binding["camera_pose_binding"]["pose_frame_id"]) != 0
            or binding["camera_pose_binding"]["strict_split_role"] != "STRICT_TRAIN"
        ):
            raise RuntimeError(f"camera/pose binding changed: {slot}")
        if (
            not source["mask_accepted"]
            or source["person_mask_human_decision"] != "PASS"
            or source["garment_mask_human_decision"] != "PASS"
            or source["mask_pair_human_decision"] != "PASS"
        ):
            raise RuntimeError(f"mask acceptance changed: {slot}")
        width = int(source["native_resolution"]["width"])
        height = int(source["native_resolution"]["height"])
        asset_root = run_root / "inputs" / "targets" / slot
        staged = {
            "accepted_raw": asset_root / "accepted_raw.png",
            "person_mask": asset_root / "person_mask.png",
            "garment_mask": asset_root / "garment_mask.png",
        }
        source_bindings = {
            "accepted_raw": source["accepted_raw"],
            "person_mask": source["person_mask"],
            "garment_mask": source["garment_mask"],
        }
        verified: dict[str, Any] = {}
        for name in staged:
            expected = source_bindings[name]
            verified[name] = verify_file(
                staged[name], expected["sha256"], int(expected["bytes"])
            )
        raw = load_rgb(staged["accepted_raw"], width, height)
        person = load_mask(staged["person_mask"], width, height)
        garment = load_mask(staged["garment_mask"], width, height)
        if int(torch.count_nonzero(garment * (1 - person))) != 0:
            raise RuntimeError(f"garment mask escapes person mask: {slot}")
        protected = person * (1 - garment)
        if int(torch.count_nonzero(garment)) == 0 or int(torch.count_nonzero(protected)) == 0:
            raise RuntimeError(f"empty garment/protected region: {slot}")
        record = {
            "schema_id": "subject00.teacher_target_observation.v1",
            "teacher_target_id": f"subject00/O03/{slot}",
            "subject": "Subject00",
            "garment": "O03",
            "slot": slot,
            "camera": f"cam{camera:02d}",
            "camera_id": camera,
            "direction": direction,
            "pose_frame_id": 0,
            "strict_split_role": "STRICT_TRAIN",
            "native_resolution": {"width": width, "height": height},
            "accepted_raw": {
                "original_path": source_bindings["accepted_raw"]["path"],
                **verified["accepted_raw"],
            },
            "person_mask": {
                "original_path": source_bindings["person_mask"]["path"],
                **verified["person_mask"],
            },
            "garment_mask": {
                "original_path": source_bindings["garment_mask"]["path"],
                **verified["garment_mask"],
            },
            "camera_calibration": binding["camera_pose_binding"]["camera_file"],
            "pose_smplx": binding["camera_pose_binding"]["pose_smplx_file"],
            "calibrated_view_orientation_degrees": binding[
                "camera_pose_binding"
            ]["calibrated_view_orientation_degrees"],
            "limitation_codes": source["limitation_codes"],
            "human_override_status": source["human_override_status"],
            "mask_acceptance_status": source["mask_acceptance_status"],
            "request_id": source["request_id"],
            "derived_loss_masks": {
                "target_foreground": "person_mask",
                "target_clothing": "garment_mask",
                "edit_mask": "garment_mask",
                "old_clothing_mask": "garment_mask (no separate accepted source field)",
                "protected_mask": "person_mask AND NOT garment_mask",
                "transition_mask": "3x3 morphological gradient of garment_mask",
            },
            "base_avatar_checkpoint": {
                "path": config["base"]["checkpoint_path"],
                "bytes": config["base"]["checkpoint_bytes"],
                "sha256": config["base"]["checkpoint_sha256"],
                "step": 60747,
                "class": "PROVISIONAL_INTERMEDIATE_BASE",
            },
            "provenance": {
                "mask_promotion_branch": config["source_git"][
                    "mask_promotion_branch"
                ],
                "mask_promotion_head": config["source_git"]["mask_promotion_head"],
                "mask_registry_sha256": evidence["mask_accepted_registry_sha256"],
                "preflight_draft_sha256": evidence[
                    "teacher_target_preflight_draft_sha256"
                ],
            },
        }
        records.append(record)
        loaded.append(
            {
                "record": record,
                "raw": raw,
                "person": person,
                "garment": garment,
                "protected": protected,
            }
        )
    if len(records) != 8:
        raise RuntimeError("O03 target count is not eight")
    registry = {
        "schema_version": "canondressgs.subject00.o03_teacher_target_registry.v1",
        "task_id": TASK_ID,
        "status": "PROVISIONAL_BASE60747_TARGET_PREFLIGHT_PASS",
        "record_count": 8,
        "denominator": 8,
        "records": records,
        "loader_contract": {
            "accepted_raw_loader": "PIL RGB; source bytes/SHA verified before decode",
            "person_mask_loader": "PIL L; exact values subset of {0,255}",
            "garment_mask_loader": "PIL L; exact values subset of {0,255}",
            "batching": "batch_size=1; native per-view resolution; no resize/pad/reencode",
            "pair_invariants": [
                "raw/person/garment dimensions exactly equal",
                "garment mask subset of person mask",
                "garment outside person pixel count zero",
                "garment and protected regions non-empty",
            ],
        },
        "camera_binding": {
            "source": "formal Subject00 calibration/SMPL-X strict-train frame 0",
            "camera_ids": list(CAMERAS),
            "pose_frame_id": 0,
        },
        "provisional_base": config["base"],
        "paper_eligible": False,
        "human_visual_decision": None,
        "scientific_pass": None,
    }
    return registry, loaded


def axis_angle_to_quaternion_wxyz(rotvec: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.vector_norm(rotvec, dim=-1, keepdim=True)
    half = angle * 0.5
    scale = 0.5 * torch.sinc(angle / (2 * torch.pi))
    return torch.cat([torch.cos(half), rotvec * scale], dim=-1)


def quaternion_multiply_wxyz(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(-1)
    rw, rx, ry, rz = right.unbind(-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def garment_trainable_mask(base_model: Any) -> torch.Tensor:
    weights = base_model.get_weights.detach()
    if weights.ndim != 2 or weights.shape[1] < FORMAL_LBS_JOINT_COUNT:
        raise RuntimeError("Base LBS weights are not [N,>=55]")
    weights = weights[:, :FORMAL_LBS_JOINT_COUNT]
    if not torch.isfinite(weights).all():
        raise FloatingPointError("Base LBS weights contain NaN/Inf")
    if not torch.allclose(
        weights.sum(1),
        torch.ones(weights.shape[0], device=weights.device),
        atol=1e-5,
        rtol=0,
    ):
        raise RuntimeError("Base LBS weights do not sum to one")
    body_part = weights.argmax(1)
    mask = torch.zeros_like(body_part, dtype=torch.bool)
    for index in GARMENT_BODY_PARTS:
        mask |= body_part == index
    if int(mask.sum()) < 30_000:
        raise RuntimeError("canonical garment support has fewer than 30k Gaussians")
    return mask[:, None]


def masked_gaussian(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask
    while expanded.ndim < value.ndim:
        expanded = expanded.unsqueeze(-1)
    return value * expanded.to(value)


class UnboundedGaussianDeltaField(nn.Module):
    """The sealed Subject02 Rung-2 shared canonical Teacher parameterization."""

    def __init__(self, base_model: Any) -> None:
        super().__init__()
        count = int(base_model._xyz.shape[0])
        if count != 200_000:
            raise RuntimeError("Teacher requires the original 200k support")
        self.register_buffer(
            "trainable_support", garment_trainable_mask(base_model), persistent=True
        )
        self.raw_xyz = nn.Parameter(torch.zeros_like(base_model._xyz))
        self.raw_log_scaling = nn.Parameter(torch.zeros_like(base_model._scaling))
        self.raw_rotvec = nn.Parameter(
            torch.zeros(
                count,
                3,
                device=base_model._xyz.device,
                dtype=base_model._xyz.dtype,
            )
        )
        self.raw_opacity = nn.Parameter(torch.zeros_like(base_model._opacity))
        self.raw_sh0 = nn.Parameter(torch.zeros_like(base_model._sh0))

    def overrides(self, base_model: Any) -> dict[str, torch.Tensor]:
        delta_rotation = axis_angle_to_quaternion_wxyz(
            masked_gaussian(self.raw_rotvec, self.trainable_support)
        )
        rotation = F.normalize(
            quaternion_multiply_wxyz(
                F.normalize(base_model._rotation, dim=-1), delta_rotation
            ),
            dim=-1,
        )
        return {
            "xyz": base_model._xyz
            + masked_gaussian(self.raw_xyz, self.trainable_support),
            "scaling": base_model._scaling
            + masked_gaussian(self.raw_log_scaling, self.trainable_support),
            "rotation": rotation,
            "opacity": base_model._opacity
            + masked_gaussian(self.raw_opacity, self.trainable_support),
            "sh0": base_model._sh0
            + masked_gaussian(self.raw_sh0, self.trainable_support),
            "shN": base_model._shN,
        }

    def optimizer_groups(
        self, geometry_lr: float, appearance_lr: float
    ) -> list[dict[str, Any]]:
        return [
            {
                "name": "geometry",
                "params": [self.raw_xyz, self.raw_log_scaling, self.raw_rotvec],
                "lr": float(geometry_lr),
            },
            {
                "name": "appearance",
                "params": [self.raw_opacity, self.raw_sh0],
                "lr": float(appearance_lr),
            },
        ]


def direct_stability(field: UnboundedGaussianDeltaField) -> torch.Tensor:
    return sum(
        masked_gaussian(value, field.trainable_support).square().mean()
        for value in (
            field.raw_xyz,
            field.raw_log_scaling,
            field.raw_rotvec,
            field.raw_opacity,
            field.raw_sh0,
        )
    )


def mask_boundary(mask: torch.Tensor) -> torch.Tensor:
    nchw = mask.permute(2, 0, 1)[None]
    dilation = F.max_pool2d(nchw, 3, stride=1, padding=1)
    erosion = 1 - F.max_pool2d(1 - nchw, 3, stride=1, padding=1)
    return (dilation - erosion).clamp(0, 1)[0].permute(1, 2, 0)


def masked_l1(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    expanded = mask.expand_as(prediction)
    return ((prediction - target.to(prediction)).abs() * expanded).sum() / expanded.sum().clamp_min(1)


def capacity_loss(
    prediction_rgb: torch.Tensor,
    prediction_alpha: torch.Tensor,
    target: Mapping[str, Any],
    base_rgb: torch.Tensor,
    base_alpha: torch.Tensor,
    weights: Mapping[str, float],
    field: UnboundedGaussianDeltaField,
) -> dict[str, torch.Tensor]:
    target_rgb = target["raw"]
    foreground = target["person"]
    garment = target["garment"]
    protected = target["protected"]
    boundary = target["boundary"]
    new_silhouette = foreground * (1 - base_alpha)
    parts = {
        "garment_rgb": masked_l1(prediction_rgb, target_rgb, garment),
        "alpha_foreground": masked_l1(
            prediction_alpha, foreground, torch.ones_like(foreground)
        ),
        "new_silhouette_alpha": masked_l1(
            prediction_alpha, foreground, new_silhouette
        ),
        "boundary_rgb": masked_l1(prediction_rgb, target_rgb, boundary),
        "protected_rgb": masked_l1(prediction_rgb, base_rgb, protected),
        "protected_alpha": masked_l1(prediction_alpha, base_alpha, protected),
        "stability": direct_stability(field),
    }
    parts["total"] = sum(float(weights[name]) * value for name, value in parts.items())
    return parts


def restore_base(
    config: Mapping[str, Any], data_root: Path, assets_root: Path, availability: Path
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    protocol = formal_base.load_protocol_before_rebinding(availability)
    formal_base.apply_formal_base_runtime_binding()
    formal.configure_runtime_modules()
    restore_args = argparse.Namespace(
        attempt_root=formal_base.FORMAL_BASE_ATTEMPT_ROOT,
        data_root=data_root,
        assets_root=assets_root,
    )
    model, runtime_args, payload, sidecar = formal.restore_model_from_checkpoint(
        restore_args, protocol, 60747
    )
    if (
        int(payload["training_step"]) != 60747
        or sidecar["sha256"] != config["base"]["checkpoint_sha256"]
        or int(sidecar["bytes"]) != int(config["base"]["checkpoint_bytes"])
    ):
        raise RuntimeError("restored Base60747 binding changed")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    model.optimizers = {}
    model.schedulers = []
    restore_summary = {
        "sidecar": sidecar,
        "runtime_args_type": type(runtime_args).__name__,
        "training_step": int(payload["training_step"]),
        "data_order_position": int(payload["data_order_position"]),
        "optimizer_state_present": isinstance(payload.get("optimizer_states"), dict),
        "scheduler_state_count": len(payload.get("scheduler_states", [])),
        "rng_state_complete": all(
            payload.get(name) is not None
            for name in (
                "python_rng_state",
                "numpy_rng_state",
                "torch_rng_state",
                "cuda_rng_states",
            )
        ),
    }
    del payload
    torch.cuda.empty_cache()
    return model, protocol, restore_summary


def build_camera_items(
    data_root: Path,
    loaded: list[dict[str, Any]],
) -> None:
    dataset = core.ThumanDataset(
        datadir=str(data_root),
        frame_ids=[0],
        cam_ids=list(CAMERAS),
        background=np.ones(3, dtype=np.float32),
        image_scaling=1,
        is_in_memory=False,
    )
    mapping = {
        (int(pose), int(camera)): index
        for index, (pose, camera) in enumerate(dataset.indices)
    }
    for target in loaded:
        camera = int(target["record"]["camera_id"])
        if (0, camera) not in mapping:
            raise RuntimeError(f"camera binding unavailable in formal dataset: {camera}")
        item = core.prepare_item(dataset[mapping[(0, camera)]])
        expected = target["record"]["native_resolution"]
        if (
            int(item["width"]) != int(expected["width"])
            or int(item["height"]) != int(expected["height"])
        ):
            raise RuntimeError(
                f"native target/formal camera dimensions disagree: {target['record']['slot']}"
            )
        target["item"] = item
        for name in ("raw", "person", "garment", "protected"):
            target[name] = target[name].cuda(non_blocking=False)
        target["boundary"] = mask_boundary(target["garment"])


def render(
    base: Any,
    target: Mapping[str, Any],
    background: torch.Tensor,
    field: UnboundedGaussianDeltaField | None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    core.set_model_pose(base, target["item"])
    torch.cuda.synchronize()
    started = time.perf_counter()
    rgb, alpha, _ = base.render(
        target["item"],
        background=background,
        canonical_overrides=None if field is None else field.overrides(base),
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return rgb.clamp(0, 1), alpha.clamp(0, 1), elapsed


def save_png(path: Path, value: torch.Tensor) -> None:
    array = value.detach().cpu().numpy()
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.dtype != np.uint8:
        array = np.clip(array * 255.0, 0, 255).round().astype(np.uint8)
    image = Image.fromarray(array, mode="L" if array.ndim == 2 else "RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    image.save(temporary, format="PNG")
    os.replace(temporary, path)


def save_checkpoint(
    run_root: Path,
    step: int,
    field: UnboundedGaussianDeltaField,
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    target_registry_sha: str,
    execution_head: str,
    last_record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    path = run_root / "checkpoints" / f"step_{step:06d}.pth"
    sidecar_path = path.with_suffix(".sidecar.json")
    if path.exists() or sidecar_path.exists():
        raise RuntimeError(f"checkpoint overwrite forbidden: {step}")
    payload = {
        "schema_version": "canondressgs.subject00.o03_provisional_teacher_checkpoint.v1",
        "task_id": TASK_ID,
        "classification": "PROVISIONAL_BASE60747_RESULT",
        "paper_eligible": False,
        "subject": "Subject00",
        "garment": "O03",
        "global_step": int(step),
        "optimizer_step": int(step),
        "model": field.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        },
        "base_checkpoint": config["base"],
        "target_registry_sha256": target_registry_sha,
        "target_count": 8,
        "view_schedule": config["teacher"]["view_schedule"],
        "execution_head": execution_head,
        "config_sha256": sha256_file(CONFIG_PATH),
        "last_record": dict(last_record) if last_record is not None else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    with temporary.open("rb+") as handle:
        os.fsync(handle.fileno())
    digest = sha256_file(temporary)
    size = temporary.stat().st_size
    os.replace(temporary, path)
    sidecar = {
        "schema_version": "canondressgs.subject00.o03_provisional_teacher_checkpoint_sidecar.v1",
        "task_id": TASK_ID,
        "step": int(step),
        "bytes": size,
        "sha256": digest,
        "path": str(path),
        "atomically_sealed": True,
        "base_checkpoint_sha256": config["base"]["checkpoint_sha256"],
        "target_registry_sha256": target_registry_sha,
        "checkpoint_overwrite": 0,
        "paper_eligible": False,
    }
    atomic_json(sidecar_path, sidecar)
    return sidecar


def tensor_to_pil(value: torch.Tensor, max_width: int = 300) -> Image.Image:
    array = value.detach().cpu().numpy()
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    array = np.clip(array * 255.0, 0, 255).round().astype(np.uint8)
    image = Image.fromarray(array, mode="L" if array.ndim == 2 else "RGB").convert("RGB")
    if image.width > max_width:
        height = max(1, round(image.height * max_width / image.width))
        image = image.resize((max_width, height), Image.Resampling.LANCZOS)
    return image


def contact_sheet(
    path: Path,
    panels: Iterable[tuple[str, torch.Tensor]],
    columns: int = 4,
) -> None:
    images = [(label, tensor_to_pil(value)) for label, value in panels]
    if not images:
        return
    tile_width = max(image.width for _, image in images)
    tile_height = max(image.height for _, image in images) + 28
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (tile_width * columns, tile_height * rows), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(images):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        sheet.paste(image, (x, y + 24))
        draw.text((x + 4, y + 4), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    sheet.save(temporary, format="PNG")
    os.replace(temporary, path)


def masked_composite(
    value: torch.Tensor, mask: torch.Tensor, background: float = 1.0
) -> torch.Tensor:
    return value * mask + background * (1 - mask)


def metric_lpips(metric: Any, first: torch.Tensor, second: torch.Tensor) -> float | None:
    if metric is None:
        return None
    return float(
        metric(
            first.permute(2, 0, 1)[None],
            second.permute(2, 0, 1)[None],
        ).item()
    )


def evaluate_view(
    rgb: torch.Tensor,
    alpha: torch.Tensor,
    target: Mapping[str, Any],
    lpips_metric: Any,
) -> dict[str, Any]:
    truth = target["raw"]
    garment = target["garment"]
    protected = target["protected"]
    mse = torch.mean((rgb - truth) ** 2)
    garment_mse = ((rgb - truth).square() * garment).sum() / (
        garment.expand_as(rgb).sum().clamp_min(1)
    )
    full_ssim = structural_similarity_index_measure(
        rgb.permute(2, 0, 1)[None],
        truth.permute(2, 0, 1)[None],
        data_range=1.0,
    )
    garment_rgb = masked_composite(rgb, garment)
    garment_truth = masked_composite(truth, garment)
    garment_ssim = structural_similarity_index_measure(
        garment_rgb.permute(2, 0, 1)[None],
        garment_truth.permute(2, 0, 1)[None],
        data_range=1.0,
    )
    pred_mask = alpha[..., 0] >= 0.5
    truth_mask = target["person"][..., 0] >= 0.5
    intersection = int((pred_mask & truth_mask).sum())
    union = int((pred_mask | truth_mask).sum())
    boundary_f = core.boundary_f_score(
        pred_mask.detach().cpu().numpy(),
        truth_mask.detach().cpu().numpy(),
        tolerance=3,
    )
    protected_rgb = masked_composite(rgb, protected)
    protected_truth = masked_composite(truth, protected)
    return {
        "full_image_lpips": metric_lpips(lpips_metric, rgb, truth),
        "psnr": float(-10 * torch.log10(mse.clamp_min(1e-12))),
        "ssim": float(full_ssim),
        "garment_region_lpips": metric_lpips(
            lpips_metric, garment_rgb, garment_truth
        ),
        "garment_region_psnr": float(
            -10 * torch.log10(garment_mse.clamp_min(1e-12))
        ),
        "garment_region_ssim": float(garment_ssim),
        "silhouette_iou": intersection / max(union, 1),
        "boundary_f": float(boundary_f),
        "protected_region_lpips": metric_lpips(
            lpips_metric, protected_rgb, protected_truth
        ),
        "protected_region_rgb_mae": float(
            masked_l1(rgb, truth, protected)
        ),
        "render_finite": bool(torch.isfinite(rgb).all() and torch.isfinite(alpha).all()),
        "background_alpha_mean": float(alpha[~truth_mask].mean())
        if bool((~truth_mask).any())
        else 0.0,
    }


def mean_optional(rows: Iterable[Mapping[str, Any]], name: str) -> float | None:
    values = [row[name] for row in rows if row.get(name) is not None]
    return float(np.mean(values)) if values else None


def abnormal_summary(
    field: UnboundedGaussianDeltaField, base: Any
) -> dict[str, Any]:
    with torch.no_grad():
        support = field.trainable_support[:, 0]
        displacement = torch.linalg.vector_norm(
            masked_gaussian(field.raw_xyz, field.trainable_support), dim=1
        )
        scale_ratio = torch.exp(
            masked_gaussian(field.raw_log_scaling, field.trainable_support).abs()
        ).amax(1)
        opacity = torch.sigmoid(
            base._opacity
            + masked_gaussian(field.raw_opacity, field.trainable_support)
        ).reshape(-1)
        overrides = field.overrides(base)
        center = base._xyz.mean(0)
        radius = torch.linalg.vector_norm(base._xyz - center, dim=1).max() * 1.5
        outside = (
            torch.linalg.vector_norm(overrides["xyz"] - center, dim=1) > radius
        )
        abnormal = support & (
            (displacement > 0.25)
            | (scale_ratio > 8.0)
            | (opacity < 0.005)
            | (opacity > 0.995)
            | outside
        )
    return {
        "trainable_support_count": int(support.sum()),
        "abnormal_gaussian_count": int(abnormal.sum()),
        "abnormal_gaussian_fraction_of_trainable": float(
            abnormal.sum() / support.sum().clamp_min(1)
        ),
        "max_xyz_displacement_m": float(displacement.max()),
        "max_scale_ratio": float(scale_ratio.max()),
        "min_opacity": float(opacity.min()),
        "max_opacity": float(opacity.max()),
        "outside_base_radius_count": int((outside & support).sum()),
    }


def save_review_crops(
    run_root: Path,
    target: Mapping[str, Any],
    teacher_rgb: torch.Tensor,
) -> None:
    slot = target["record"]["slot"]
    person = target["person"][..., 0] >= 0.5
    ys, xs = torch.where(person)
    if not len(xs):
        return
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    height = max(1, y1 - y0)
    width = max(1, x1 - x0)
    crops = {
        "face_head": (x0, y0, x1, y0 + max(1, height // 4)),
        "hands": (
            x0,
            y0 + height // 3,
            x1,
            y0 + 3 * height // 4,
        ),
        "feet": (x0, y0 + 3 * height // 4, x1, y1),
        "protected_region": (x0, y0, x1, y1),
    }
    for name, (left, top, right, bottom) in crops.items():
        value = teacher_rgb[top:bottom, left:right]
        save_png(run_root / "review" / name / f"{slot}.png", value)


def render_novel_queries(
    run_root: Path,
    base: Any,
    field: UnboundedGaussianDeltaField,
    protocol: Mapping[str, Any],
    data_root: Path,
    background: torch.Tensor,
) -> dict[str, Any]:
    queries = protocol["visual"]["queries"]
    target_cameras = set(CAMERAS)
    different_camera = next(
        (
            row
            for row in queries
            if int(row["camera_id"]) not in target_cameras
            and int(row["pose_id"]) == 0
        ),
        next(row for row in queries if int(row["camera_id"]) not in target_cameras),
    )
    different_pose = next(row for row in queries if int(row["pose_id"]) != 0)
    results: dict[str, Any] = {}
    for name, query in (
        ("different_camera", different_camera),
        ("different_pose", different_pose),
    ):
        pose = int(query["pose_id"])
        camera = int(query["camera_id"])
        dataset = core.ThumanDataset(
            datadir=str(data_root),
            frame_ids=[pose],
            cam_ids=[camera],
            background=np.ones(3, dtype=np.float32),
            image_scaling=1,
            is_in_memory=False,
        )
        item = core.prepare_item(dataset[0])
        holder = {"item": item}
        base_rgb, base_alpha, base_seconds = render(
            base, holder, background, None
        )
        teacher_rgb, teacher_alpha, teacher_seconds = render(
            base, holder, background, field
        )
        finite = bool(
            torch.isfinite(teacher_rgb).all()
            and torch.isfinite(teacher_alpha).all()
        )
        save_png(run_root / "review" / name / "base.png", base_rgb)
        save_png(run_root / "review" / name / "teacher.png", teacher_rgb)
        contact_sheet(
            run_root / "review" / name / "comparison.png",
            [("Base60747", base_rgb), ("Teacher60747", teacher_rgb)],
            columns=2,
        )
        results[name] = {
            "status": "PASS_FINITE_RENDER" if finite else "FAIL_NONFINITE_RENDER",
            "pose_id": pose,
            "camera_id": camera,
            "base_render_seconds": base_seconds,
            "teacher_render_seconds": teacher_seconds,
            "height": int(item["height"]),
            "width": int(item["width"]),
        }
    results["animation_compatibility"] = {
        "status": "PASS"
        if results["different_pose"]["status"] == "PASS_FINITE_RENDER"
        else "FAIL",
        "criterion": "frozen deformation/LBS path produced a finite different-pose Teacher render",
    }
    return results


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    run_root = args.output_root / config["runtime"]["attempt"]
    if args.output_root != Path(config["runtime"]["output_root"]):
        raise RuntimeError("output root differs from frozen contract")
    if run_root.name != "attempt_001":
        raise RuntimeError("attempt id differs from frozen contract")
    if not args.output_root.is_dir() or not run_root.is_dir():
        raise RuntimeError("prepared provisional output/attempt root is missing")
    permitted_initial = {"inputs"}
    unexpected = {path.name for path in run_root.iterdir()} - permitted_initial
    if unexpected:
        raise RuntimeError(f"provisional run root is not fresh: {sorted(unexpected)}")
    required_input_paths = {
        args.mask_registry.resolve(),
        args.preflight_draft.resolve(),
        *((
            run_root / "inputs" / "targets" / slot / name
        ).resolve() for slot in SLOTS for name in (
            "accepted_raw.png", "person_mask.png", "garment_mask.png"
        )),
    }
    for path in required_input_paths:
        if not path.is_file() or run_root.resolve() not in path.parents:
            raise RuntimeError(f"input is absent or outside provisional root: {path}")
    git = require_git_state(config)
    storage = require_storage(
        args.output_root, int(config["runtime"]["minimum_free_bytes"])
    )
    gpu = require_gpu(config)
    for name in (
        "audits",
        "checkpoints",
        "contract",
        "evaluations",
        "review",
        "training",
    ):
        (run_root / name).mkdir(exist_ok=False)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    random.seed(int(config["teacher"]["seed"]))
    np.random.seed(int(config["teacher"]["seed"]))
    torch.manual_seed(int(config["teacher"]["seed"]))
    torch.cuda.manual_seed_all(int(config["teacher"]["seed"]))
    try:
        registry, targets = build_target_registry(
            config, run_root, args.mask_registry, args.preflight_draft
        )
        target_registry_path = run_root / "contract" / "target_registry.json"
        atomic_json(target_registry_path, registry)
        target_registry_sha = sha256_file(target_registry_path)
        checkpoint_audit = verify_file(
            Path(config["base"]["checkpoint_path"]),
            config["base"]["checkpoint_sha256"],
            int(config["base"]["checkpoint_bytes"]),
        )
        sidecar = read_json(Path(config["base"]["sidecar_path"]))
        if (
            sidecar["sha256"] != config["base"]["checkpoint_sha256"]
            or int(sidecar["step"]) != 60747
            or sidecar["atomically_sealed"] is not True
        ):
            raise RuntimeError("Base60747 sidecar changed")
        base, protocol, restore = restore_base(
            config, args.data_root, args.assets_root, args.availability_manifest
        )
        base_before = model_fingerprint(base)
        build_camera_items(args.data_root, targets)
        field = UnboundedGaussianDeltaField(base).cuda()
        optimizer = torch.optim.Adam(
            field.optimizer_groups(
                config["teacher"]["optimizer"]["geometry_lr"],
                config["teacher"]["optimizer"]["appearance_lr"],
            )
        )
        background = torch.ones(3, device="cuda")
        base_cache: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = {}
        for target in targets:
            target["record"]["target_registry_sha256"] = target_registry_sha
            base_cache[target["record"]["slot"]] = render(
                base, target, background, None
            )
        contract = {
            "schema_version": "canondressgs.subject00.o03_provisional_teacher_execution_contract.v1",
            "task_id": TASK_ID,
            "git": git,
            "config": config,
            "config_sha256": sha256_file(args.config),
            "target_registry_path": str(target_registry_path),
            "target_registry_sha256": target_registry_sha,
            "checkpoint_audit": checkpoint_audit,
            "checkpoint_sidecar": sidecar,
            "storage_gate": storage,
            "gpu_gate": gpu,
            "base_fingerprint_before": base_before,
            "trainable_groups": {
                "geometry": [
                    "raw_xyz",
                    "raw_log_scaling",
                    "raw_rotvec",
                ],
                "appearance": ["raw_opacity", "raw_sh0"],
            },
            "frozen_groups": [
                "entire Base60747 model",
                "surface attachment and 55-joint LBS",
                "all non-garment-support residual entries",
                "SHN",
            ],
            "scheduler": None,
            "paper_eligible": False,
        }
        atomic_json(run_root / "contract" / "execution_contract.json", contract)
        atomic_json(
            run_root / "audits" / "preflight.json",
            {
                "status": "PASS",
                "target_count": 8,
                "raw_binding_status": "PASS",
                "person_mask_binding_status": "PASS",
                "garment_mask_binding_status": "PASS",
                "camera_pose_binding_status": "PASS",
                "teacher_contract_status": "PASS_UNIQUE",
                "initialization_class": "PROVISIONAL_INTERMEDIATE_BASE",
                "initialization_step": 60747,
                "paper_eligible": False,
            },
        )
        atomic_json(
            run_root / "RUN_STATUS.json",
            {
                "task_id": TASK_ID,
                "status": "RUNNING",
                "stage": "PREFLIGHT_COMPLETE",
                "optimizer_steps": 0,
                "updated_at_unix": time.time(),
            },
        )
        checkpoint_records = [
            save_checkpoint(
                run_root,
                0,
                field,
                optimizer,
                config,
                target_registry_sha,
                git["head"],
                None,
            )
        ]
        gradient_seen = {
            name: 0 for name, value in field.named_parameters() if value.requires_grad
        }
        loss_initial: float | None = None
        last_record: dict[str, Any] | None = None
        for step in range(1, 1201):
            target = targets[(step - 1) % 8]
            slot = target["record"]["slot"]
            optimizer.zero_grad(set_to_none=True)
            rgb, alpha, render_seconds = render(base, target, background, field)
            base_rgb, base_alpha, _ = base_cache[slot]
            parts = capacity_loss(
                rgb,
                alpha,
                target,
                base_rgb,
                base_alpha,
                config["teacher"]["loss_weights"],
                field,
            )
            if not torch.isfinite(parts["total"]):
                raise FloatingPointError(f"non-finite Teacher loss at step {step}")
            parts["total"].backward()
            for name, parameter in field.named_parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(
                            f"non-finite Teacher gradient at {step}: {name}"
                        )
                    if bool(torch.count_nonzero(parameter.grad)):
                        gradient_seen[name] += 1
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                field.parameters(),
                float(config["teacher"]["optimizer"]["gradient_clip_norm"]),
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"non-finite Teacher gradient norm at step {step}"
                )
            optimizer.step()
            row = {
                "step": step,
                "slot": slot,
                "camera_id": target["record"]["camera_id"],
                "view_denominator": 8,
                "loss": {
                    name: float(value.detach()) for name, value in parts.items()
                },
                "gradient_norm": float(gradient_norm),
                "learning_rates": {
                    group["name"]: float(group["lr"])
                    for group in optimizer.param_groups
                },
                "loss_finite": True,
                "gradient_finite": True,
                "render_seconds": render_seconds,
            }
            if loss_initial is None:
                loss_initial = row["loss"]["total"]
            last_record = row
            append_jsonl(run_root / "training" / "state_records.jsonl", row)
            atomic_json(
                run_root / "RUN_STATUS.json",
                {
                    "task_id": TASK_ID,
                    "status": "RUNNING",
                    "stage": "TRAINING",
                    "optimizer_steps": step,
                    "current_step": step,
                    "loss": row["loss"]["total"],
                    "loss_finite": True,
                    "gradient_finite": True,
                    "updated_at_unix": time.time(),
                },
            )
            if step in CHECKPOINT_STEPS:
                checkpoint_records.append(
                    save_checkpoint(
                        run_root,
                        step,
                        field,
                        optimizer,
                        config,
                        target_registry_sha,
                        git["head"],
                        row,
                    )
                )
        if not all(value > 0 for value in gradient_seen.values()):
            raise RuntimeError(f"Teacher parameter group never received gradient: {gradient_seen}")
        base_after = model_fingerprint(base)
        if base_before != base_after:
            raise RuntimeError("frozen Base60747 tensors changed")
        if sha256_file(Path(config["base"]["checkpoint_path"])) != config["base"][
            "checkpoint_sha256"
        ]:
            raise RuntimeError("Base60747 checkpoint bytes changed")
        for record in registry["records"]:
            for name in ("accepted_raw", "person_mask", "garment_mask"):
                if sha256_file(Path(record[name]["path"])) != record[name]["sha256"]:
                    raise RuntimeError(
                        f"target source bytes changed after training: {record['slot']}/{name}"
                    )
        lpips_metric, lpips_error = core.make_lpips_preserving_rng()
        base_rows: list[dict[str, Any]] = []
        teacher_rows: list[dict[str, Any]] = []
        target_panels: list[tuple[str, torch.Tensor]] = []
        base_panels: list[tuple[str, torch.Tensor]] = []
        teacher_panels: list[tuple[str, torch.Tensor]] = []
        person_panels: list[tuple[str, torch.Tensor]] = []
        garment_panels: list[tuple[str, torch.Tensor]] = []
        boundary_panels: list[tuple[str, torch.Tensor]] = []
        protected_panels: list[tuple[str, torch.Tensor]] = []
        render_times: list[float] = []
        for target in targets:
            slot = target["record"]["slot"]
            base_rgb, base_alpha, base_seconds = base_cache[slot]
            teacher_rgb, teacher_alpha, teacher_seconds = render(
                base, target, background, field
            )
            render_times.append(teacher_seconds)
            base_metric = evaluate_view(
                base_rgb, base_alpha, target, lpips_metric
            )
            teacher_metric = evaluate_view(
                teacher_rgb, teacher_alpha, target, lpips_metric
            )
            base_rows.append({"slot": slot, **base_metric, "render_seconds": base_seconds})
            teacher_rows.append(
                {"slot": slot, **teacher_metric, "render_seconds": teacher_seconds}
            )
            save_png(run_root / "review" / "target" / f"{slot}.png", target["raw"])
            save_png(run_root / "review" / "base60747" / f"{slot}.png", base_rgb)
            save_png(run_root / "review" / "teacher60747" / f"{slot}.png", teacher_rgb)
            save_png(run_root / "review" / "person_mask" / f"{slot}.png", target["person"])
            save_png(run_root / "review" / "garment_mask" / f"{slot}.png", target["garment"])
            save_png(run_root / "review" / "boundary" / f"{slot}.png", target["boundary"])
            save_png(run_root / "review" / "protected" / f"{slot}.png", target["protected"])
            save_review_crops(run_root, target, teacher_rgb)
            target_panels.append((f"{slot} target", target["raw"]))
            base_panels.append((f"{slot} Base60747", base_rgb))
            teacher_panels.append((f"{slot} Teacher60747", teacher_rgb))
            person_panels.append((f"{slot} person", target["person"]))
            garment_panels.append((f"{slot} garment", target["garment"]))
            boundary_panels.append((f"{slot} boundary", target["boundary"]))
            protected_panels.append((f"{slot} protected", target["protected"]))
        contact_sheet(run_root / "review" / "target_contact_sheet.png", target_panels)
        contact_sheet(run_root / "review" / "base60747_contact_sheet.png", base_panels)
        contact_sheet(run_root / "review" / "teacher60747_contact_sheet.png", teacher_panels)
        contact_sheet(run_root / "review" / "person_mask_contact_sheet.png", person_panels)
        contact_sheet(run_root / "review" / "garment_mask_contact_sheet.png", garment_panels)
        contact_sheet(run_root / "review" / "boundary_contact_sheet.png", boundary_panels)
        contact_sheet(run_root / "review" / "protected_contact_sheet.png", protected_panels)
        comparisons: list[tuple[str, torch.Tensor]] = []
        for target, base_panel, teacher_panel in zip(
            targets, base_panels, teacher_panels, strict=True
        ):
            comparisons.extend(
                [
                    (f"{target['record']['slot']} target", target["raw"]),
                    base_panel,
                    teacher_panel,
                ]
            )
        contact_sheet(
            run_root / "review" / "comparison_contact_sheet.png",
            comparisons,
            columns=3,
        )
        novel = render_novel_queries(
            run_root, base, field, protocol, args.data_root, background
        )
        abnormal = abnormal_summary(field, base)
        severe_artifact_count = sum(
            not row["render_finite"]
            or row["background_alpha_mean"] > 0.05
            or row["silhouette_iou"] < 0.5
            for row in teacher_rows
        )
        metrics = {
            "schema_version": "canondressgs.subject00.o03_provisional_teacher_evaluation.v1",
            "task_id": TASK_ID,
            "classification": "PROVISIONAL_BASE60747_RESULT",
            "paper_eligible": False,
            "target_count": 8,
            "base60747": {
                "per_view": base_rows,
                "macro": {
                    name: mean_optional(base_rows, name)
                    for name in (
                        "full_image_lpips",
                        "psnr",
                        "ssim",
                        "garment_region_lpips",
                        "garment_region_psnr",
                        "garment_region_ssim",
                        "silhouette_iou",
                        "boundary_f",
                        "protected_region_lpips",
                        "protected_region_rgb_mae",
                    )
                },
            },
            "teacher60747": {
                "per_view": teacher_rows,
                "macro": {
                    name: mean_optional(teacher_rows, name)
                    for name in (
                        "full_image_lpips",
                        "psnr",
                        "ssim",
                        "garment_region_lpips",
                        "garment_region_psnr",
                        "garment_region_ssim",
                        "silhouette_iou",
                        "boundary_f",
                        "protected_region_lpips",
                        "protected_region_rgb_mae",
                    )
                },
            },
            "target_role": "accepted O03 loss-only target; not a model or upper bound",
            "lpips_available": lpips_metric is not None,
            "lpips_unavailable_reason": lpips_error,
            "abnormal_gaussians": abnormal,
            "automated_severe_artifact_view_count": int(severe_artifact_count),
            "automated_severe_artifact_rule": "nonfinite render OR background alpha mean >0.05 OR silhouette IoU <0.5",
            "render_time_seconds": {
                "mean": float(np.mean(render_times)),
                "median": float(np.median(render_times)),
                "maximum": float(np.max(render_times)),
            },
            "novel_render_checks": novel,
            "human_visual_decision": None,
            "scientific_pass": None,
        }
        atomic_json(run_root / "evaluations" / "final_metrics.json", metrics)
        wall_seconds = time.perf_counter() - started
        result = {
            "schema_version": "canondressgs.subject00.o03_provisional_teacher_result.v1",
            "task_id": TASK_ID,
            "status": "TECHNICAL_PASS_PENDING_USER_REVIEW",
            "classification": "PROVISIONAL_BASE60747_RESULT",
            "optimizer_steps": 1200,
            "current_step": 1200,
            "checkpoint_count": len(checkpoint_records),
            "checkpoints": checkpoint_records,
            "loss_initial": loss_initial,
            "loss_final": last_record["loss"]["total"] if last_record else None,
            "loss_finite_all": True,
            "gradient_finite_all": True,
            "nan_inf_status": "NONE",
            "oom_status": "NONE",
            "gradient_steps_nonzero": gradient_seen,
            "base_fingerprint_unchanged": True,
            "base_checkpoint_sha256_unchanged": True,
            "target_source_sha256_unchanged": True,
            "target_count": 8,
            "wall_seconds": wall_seconds,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
            "metrics_path": str(run_root / "evaluations" / "final_metrics.json"),
            "metrics_sha256": sha256_file(
                run_root / "evaluations" / "final_metrics.json"
            ),
            "review_package_path": str(run_root / "review"),
            "different_camera_status": novel["different_camera"]["status"],
            "different_pose_status": novel["different_pose"]["status"],
            "animation_compatibility": novel["animation_compatibility"]["status"],
            "human_visual_decision": None,
            "scientific_pass": None,
            "paper_eligible": False,
            "formal_base_resume_ready": True,
            "formal_base_resume_authorized": False,
            "data_mutations": 0,
            "provisional_checkpoint_mutations": 0,
            "paper_modifications": 0,
            "paper_final": False,
        }
        atomic_json(run_root / "training" / "training_result.json", result)
        atomic_json(
            run_root / "audits" / "post_training_integrity.json",
            {
                "status": "PASS",
                "base_fingerprint_before": base_before,
                "base_fingerprint_after": base_after,
                "base_checkpoint_sha256_unchanged": True,
                "target_source_sha256_unchanged": True,
                "checkpoint_overwrite_count": 0,
                "optimizer_step_count": 1200,
                "duplicate_process_count": 0,
                "paper_modifications": 0,
            },
        )
        atomic_json(
            run_root / "RUN_STATUS.json",
            {
                "task_id": TASK_ID,
                "status": "COMPLETE",
                "stage": "TECHNICAL_PASS_PENDING_USER_REVIEW",
                "optimizer_steps": 1200,
                "current_step": 1200,
                "loss": result["loss_final"],
                "updated_at_unix": time.time(),
                "final_classification": "SUBJECT00_O03_PROVISIONAL_TEACHER_BASE60747_TECHNICAL_PASS_PENDING_USER_REVIEW",
            },
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    except BaseException as exc:
        failure = {
            "task_id": TASK_ID,
            "status": "FAIL",
            "exception": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "updated_at_unix": time.time(),
            "paper_eligible": False,
        }
        atomic_json(run_root / "RUN_STATUS.json", failure)
        atomic_json(run_root / "audits" / "failure.json", failure)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/canondressgs_work/outputs/"
            "SUBJECT00-O03-TEACHER-PROVISIONAL-BASE60747-001"
        ),
    )
    parser.add_argument(
        "--mask-registry",
        type=Path,
        default=Path(
            "/root/autodl-tmp/canondressgs_work/outputs/"
            "SUBJECT00-O03-TEACHER-PROVISIONAL-BASE60747-001/attempt_001/"
            "inputs/subject00_global_mask_accepted_registry_24of24_20260727.json"
        ),
    )
    parser.add_argument(
        "--preflight-draft",
        type=Path,
        default=Path(
            "/root/autodl-tmp/canondressgs_work/outputs/"
            "SUBJECT00-O03-TEACHER-PROVISIONAL-BASE60747-001/attempt_001/"
            "inputs/subject00_24_cell_teacher_target_preflight_manifest_draft_20260727.json"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/datasets/thuman4_second_identity_staging/subject00"
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
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
