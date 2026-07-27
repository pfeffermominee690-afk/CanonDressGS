#!/usr/bin/env python3
"""Train one frozen-contract Subject00 formal Teacher from Base60747."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene.full_dressable_dataset import DUAL_TARGET_FIELDS, FullDressableTrainingDataset
from scene.representation_capacity_oracle import (
    CAPACITY_LOSS_NAME,
    UnboundedGaussianDeltaField,
    capacity_oracle_loss_v1,
    direct_stability,
)
from tools.run_representation_triage_ladder import abnormal_direct, render_direct, target_free_state
from tools.second_identity import run_subject00_o03_provisional_teacher_base60747 as legacy


TASK_ID = "AAAI27-SUBJECT00-BASE60747-ACCELERATED-METHOD-LAUNCH-001"
SCHEMA = "canondressgs.subject00.base60747_three_garment_teachers.v1"
BRANCH = "research/subject00-base60747-accelerated-method-launch-20260727"
GARMENTS = ("O01", "O03", "O04")
CHECKPOINT_STEPS = (0, 300, 600, 900, 1200)
SUPERVISION_FIELDS = (
    "target_edit_rgb",
    "target_base_rgb",
    "target_foreground_mask",
    "target_base_foreground_mask",
    "target_edit_mask",
    "target_clothing_mask",
    "target_old_clothing_mask",
    "target_protected_mask",
    "target_transition_mask",
)
CONFIG_PATH = REPO_ROOT / "configs/research/subject00_base60747_three_garment_teachers_v1.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write((json.dumps(value, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write((json.dumps(value, sort_keys=True) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != SCHEMA or config.get("task_id") != TASK_ID:
        raise RuntimeError("formal Teacher config identity changed")
    if config.get("execution_branch") != BRANCH or config.get("subject") != "Subject00":
        raise RuntimeError("formal Teacher execution binding changed")
    base = config["base"]
    if (
        int(base["step"]) != 60747
        or base["checkpoint_sha256"] != "2d09b1ce6bb19b8b6418314caab8dc1cf92d70e432a9e09e009989332ea4a2f7"
        or base["formal_base_status"] != "USER_AUTHORIZED_PAUSED"
        or base["resume_authorized"] is not False
    ):
        raise RuntimeError("Base60747 freeze contract changed")
    targets = config["formal_targets"]
    if int(targets["training_record_count"]) != 22 or int(targets["quarantine_count"]) != 2:
        raise RuntimeError("formal target denominator changed")
    teacher = config["teacher"]
    optimizer = teacher["optimizer"]
    if (
        teacher["loss_name"] != CAPACITY_LOSS_NAME
        or int(teacher["steps"]) != 1200
        or int(teacher["seed"]) != 20260718
        or optimizer["class"] != "Adam"
        or float(optimizer["geometry_lr"]) != 0.001
        or float(optimizer["appearance_lr"]) != 0.002
        or float(optimizer["gradient_clip_norm"]) != 1.0
        or teacher["scheduler"] is not None
        or tuple(teacher["checkpoint_steps"]) != CHECKPOINT_STEPS
    ):
        raise RuntimeError("shared formal Teacher optimization contract changed")
    if set(config["garments"]) != set(GARMENTS):
        raise RuntimeError("three-garment scope changed")
    return config


def git_state() -> dict[str, Any]:
    branch = subprocess.check_output(["git", "-C", str(REPO_ROOT), "branch", "--show-current"], text=True).strip()
    head = subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True).strip()
    if branch != BRANCH:
        raise RuntimeError(f"wrong execution branch: {branch}")
    if status:
        raise RuntimeError("execution worktree is dirty")
    return {"branch": branch, "head": head, "clean": True}


def require_idle_gpu() -> dict[str, Any]:
    query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if query:
        raise RuntimeError(f"another GPU compute process is active: {query}")
    name = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).strip()
    return {"status": "PASS_IDLE", "gpu_name": name, "process_count": 0}


def cpu_checkpoint_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    base = config["base"]
    path = Path(base["checkpoint_path"])
    if not path.is_file() or path.stat().st_size != int(base["checkpoint_bytes"]):
        raise RuntimeError("Base60747 checkpoint path/size changed")
    actual = sha256_file(path)
    if actual != base["checkpoint_sha256"]:
        raise RuntimeError("Base60747 checkpoint SHA256 changed")
    if list(path.parent.glob(f"{path.name}.tmp*")) or list(path.parent.glob(f".{path.name}.tmp*")):
        raise RuntimeError("Base60747 temporary/partial checkpoint exists")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "_xyz", "xyz_offset", "dxyz_vt", "_scaling", "_rotation", "_opacity",
        "_sh0", "_shN", "_weights", "surface_attachment", "encoder_feat_params",
        "dxyz_bs", "sh0_bs", "shN_bs", "scaling_bs", "rotation_bs", "opacity_bs",
        "optimizer_states", "scheduler_states", "python_rng_state",
        "numpy_rng_state", "torch_rng_state", "cuda_rng_states", "data_order_position",
        "sampler_sha256", "global_data_order_sha256", "single_pass_data_order_sha256",
    }
    missing = sorted(required.difference(payload))
    if missing or int(payload.get("training_step", -1)) != 60747:
        raise RuntimeError(f"Base60747 checkpoint payload incomplete: {missing}")
    result = {
        "status": "PASS",
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": actual,
        "training_step": int(payload["training_step"]),
        "model_state_complete": all(payload[name] is not None for name in (
            "_xyz", "xyz_offset", "dxyz_vt", "_scaling", "_rotation", "_opacity",
            "_sh0", "_shN", "_weights", "surface_attachment", "encoder_feat_params",
            "dxyz_bs", "sh0_bs", "shN_bs", "scaling_bs", "rotation_bs", "opacity_bs",
        )),
        "optimizer_state_complete": bool(payload["optimizer_states"]),
        "scheduler_state_complete": isinstance(payload["scheduler_states"], list),
        "rng_state_complete": all(payload[name] is not None for name in (
            "python_rng_state", "numpy_rng_state", "torch_rng_state", "cuda_rng_states"
        )),
        "sampling_state_complete": all(payload[name] is not None for name in (
            "data_order_position", "sampler_sha256", "global_data_order_sha256",
            "single_pass_data_order_sha256"
        )),
        "temporary_or_partial_count": 0,
    }
    del payload
    return result


def manifest_assets(manifest_path: Path, garment: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    manifest = read_json(manifest_path)
    observations = next(row["observations"] for row in manifest["outfits"] if row["outfit_id"] == garment)
    hashes: dict[str, str] = {}
    for observation in observations:
        for name, expected in observation["checksums"].items():
            path = (manifest_path.parent / observation[name]).resolve()
            actual = sha256_file(path)
            if actual != expected:
                raise RuntimeError(f"formal target checksum mismatch: {observation['condition_id']}/{name}")
            hashes[str(path)] = actual
    return observations, hashes


def load_samples(
    config: Mapping[str, Any], garment: str
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str], Path, Path]:
    root = Path(config["formal_targets"]["root"])
    manifest_path = root / config["formal_targets"]["manifest"]
    index_path = root / config["garments"][garment]["index"]
    if not manifest_path.is_file() or not index_path.is_file():
        raise FileNotFoundError("formal target manifest/index is absent")
    observations, asset_hashes = manifest_assets(manifest_path, garment)
    index = read_json(index_path)
    contract = config["garments"][garment]
    if int(index["count"]) != int(contract["target_count"]):
        raise RuntimeError("formal garment index denominator changed")
    request_ids = [row["request_id"] for row in index["records"]]
    if request_ids != [row["condition_id"] for row in observations]:
        raise RuntimeError("formal loader/index order mismatch")
    if any(not row["training_eligible"] or not row["evaluation_eligible"] or row["review_only"] for row in index["records"]):
        raise RuntimeError("formal index contains a non-training record")
    excluded = set(config["formal_targets"]["excluded_requests"])
    if excluded.intersection(request_ids):
        raise RuntimeError("quarantine request entered formal garment index")
    # Formal materialization freezes native, mixed resolutions and explicitly
    # validates batch-size-one loading without resize/crop/padding.  Teachers do
    # not consume references, so one reference exercises the authoritative
    # loader while preserving every target at its native resolution.
    dataset = FullDressableTrainingDataset(manifest_path, split="train", reference_count=1, seed=20260718)
    dataset_indices = [
        position for position, (outfit, _) in enumerate(dataset.samples)
        if outfit["outfit_id"] == garment
    ]
    compact: list[dict[str, Any]] = []
    for position in dataset_indices:
        sample = dataset[position]
        compact.append({
            "condition_id": sample["target_condition_id"],
            "outfit_id": sample["outfit_id"],
            "target_pose": sample["target_pose"],
            "target_Rh": sample["target_Rh"],
            "target_Th": sample["target_Th"],
            "target_camera": sample["target_camera"],
            **{name: sample[name] for name in SUPERVISION_FIELDS},
        })
    if [row["condition_id"] for row in compact] != request_ids:
        raise RuntimeError("formal FullDressableTrainingDataset output mismatch")
    if len(compact) != int(contract["target_count"]):
        raise RuntimeError("formal loader denominator mismatch")
    return compact, index, asset_hashes, manifest_path, index_path


def to_device_sample(sample: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "condition_id": sample["condition_id"],
        "outfit_id": sample["outfit_id"],
        "target_pose": sample["target_pose"].to(device),
        "target_Rh": sample["target_Rh"].to(device),
        "target_Th": sample["target_Th"].to(device),
        "target_camera": {
            "K": sample["target_camera"]["K"].to(device),
            "w2c": sample["target_camera"]["w2c"].to(device),
            "width": int(sample["target_camera"]["width"]),
            "height": int(sample["target_camera"]["height"]),
        },
        **{name: sample[name].to(device) for name in SUPERVISION_FIELDS},
    }


def loss_for_sample(
    rgb: torch.Tensor,
    alpha: torch.Tensor,
    sample: Mapping[str, Any],
    field: UnboundedGaussianDeltaField,
    weights: Mapping[str, float],
) -> dict[str, torch.Tensor]:
    return capacity_oracle_loss_v1(
        rgb,
        alpha,
        target_rgb=sample["target_edit_rgb"],
        base_rgb=sample["target_base_rgb"],
        target_foreground=sample["target_foreground_mask"],
        base_foreground=sample["target_base_foreground_mask"],
        edit_mask=sample["target_edit_mask"],
        clothing_mask=sample["target_clothing_mask"],
        old_clothing_mask=sample["target_old_clothing_mask"],
        protected_mask=sample["target_protected_mask"],
        transition_mask=sample["target_transition_mask"],
        weights=weights,
        stability=direct_stability(field),
    )


def masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    selected = mask.expand_as(prediction)
    return float(((prediction - target).abs() * selected).sum() / selected.sum().clamp_min(1))


def masked_composite(value: torch.Tensor, mask: torch.Tensor, background: float = 1.0) -> torch.Tensor:
    return value * mask + background * (1 - mask)


def evaluate_view(
    rgb: torch.Tensor,
    alpha: torch.Tensor,
    sample: Mapping[str, Any],
    lpips_metric: Any,
) -> dict[str, Any]:
    target = sample["target_edit_rgb"]
    base = sample["target_base_rgb"]
    protected = sample["target_protected_mask"]
    garment = torch.maximum(
        torch.maximum(sample["target_edit_mask"], sample["target_clothing_mask"]),
        sample["target_old_clothing_mask"],
    ) * (1 - protected)
    target_fg = sample["target_foreground_mask"]
    pred_fg = alpha >= 0.5
    truth_fg = target_fg >= 0.5
    intersection = int((pred_fg & truth_fg).sum())
    union = int((pred_fg | truth_fg).sum())
    boundary_f = legacy.core.boundary_f_score(
        pred_fg[0].detach().cpu().numpy(),
        truth_fg[0].detach().cpu().numpy(),
        tolerance=3,
    )
    rgb_hwc = rgb.permute(1, 2, 0)
    target_hwc = target.permute(1, 2, 0)
    base_hwc = base.permute(1, 2, 0)
    garment_hwc = garment.permute(1, 2, 0)
    protected_hwc = protected.permute(1, 2, 0)
    outside = ~truth_fg
    return {
        "garment_region_lpips": legacy.metric_lpips(
            lpips_metric,
            masked_composite(rgb_hwc, garment_hwc),
            masked_composite(target_hwc, garment_hwc),
        ),
        "silhouette_iou": intersection / max(union, 1),
        "boundary_f": float(boundary_f),
        "protected_region_lpips": legacy.metric_lpips(
            lpips_metric,
            masked_composite(rgb_hwc, protected_hwc),
            masked_composite(base_hwc, protected_hwc),
        ),
        "protected_region_rgb_mae": masked_l1(rgb, base, protected),
        "alpha_foreground_error": float((alpha - target_fg).abs().mean()),
        "render_finite": bool(torch.isfinite(rgb).all() and torch.isfinite(alpha).all()),
        "background_alpha_mean": float(alpha[outside].mean()) if bool(outside.any()) else 0.0,
    }


def mean_optional(rows: Iterable[Mapping[str, Any]], name: str) -> float | None:
    values = [float(row[name]) for row in rows if row.get(name) is not None]
    return float(np.mean(values)) if values else None


def save_png(path: Path, value: torch.Tensor) -> None:
    array = value.detach().cpu().numpy()
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = array.transpose(1, 2, 0)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    array = np.clip(array * 255, 0, 255).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="L" if array.ndim == 2 else "RGB").save(path)


def contact_sheet(path: Path, rows: list[tuple[str, torch.Tensor]], columns: int = 3) -> None:
    thumbs: list[tuple[str, Image.Image]] = []
    for label, value in rows:
        array = value.detach().cpu().permute(1, 2, 0).numpy()
        image = Image.fromarray(np.clip(array * 255, 0, 255).round().astype(np.uint8))
        image.thumbnail((320, 320))
        thumbs.append((label, image))
    width, height = columns * 340, ((len(thumbs) + columns - 1) // columns) * 360
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(thumbs):
        x, y = (index % columns) * 340, (index // columns) * 360
        canvas.paste(image, (x, y + 24))
        draw.text((x + 4, y + 4), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_checkpoint(
    run_root: Path,
    step: int,
    field: UnboundedGaussianDeltaField,
    optimizer: torch.optim.Optimizer,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    path = run_root / "checkpoints" / f"step_{step:06d}.pth"
    if path.exists():
        raise RuntimeError(f"checkpoint overwrite forbidden: {path}")
    temporary = path.with_suffix(".pth.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": "canondressgs.subject00.base60747_formal_teacher_checkpoint.v1",
        "task_id": TASK_ID,
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
        "metadata": dict(metadata),
        "target_tensors_stored": False,
        "paper_eligible": False,
    }, temporary)
    os.replace(temporary, path)
    record = {"step": int(step), "path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    atomic_json(path.with_suffix(".sidecar.json"), record)
    return record


def novel_render_checks(
    run_root: Path,
    base: Any,
    field: UnboundedGaussianDeltaField,
    samples: list[dict[str, Any]],
    background: torch.Tensor,
) -> dict[str, Any]:
    first = target_free_state(samples[0], base._xyz.device)
    second = target_free_state(samples[1], base._xyz.device)
    camera_state = dict(first)
    camera_state["camera"] = second["camera"]
    pose_state = dict(first)
    pose = second["pose"].clone()
    if torch.equal(pose, first["pose"]):
        pose[48] += 0.05
    pose_state["pose"] = pose
    result: dict[str, Any] = {}
    with torch.no_grad():
        for name, state in (("different_camera", camera_state), ("different_pose", pose_state)):
            rgb, alpha = render_direct(base, state, field(base), background)
            finite = bool(torch.isfinite(rgb).all() and torch.isfinite(alpha).all())
            save_png(run_root / "review/novel" / f"{name}.png", rgb.clamp(0, 1))
            result[name] = {"status": "PASS_FINITE" if finite else "FAIL_NONFINITE", "render_finite": finite}
    return result


def preflight(
    config: Mapping[str, Any], garment: str, *, recover_preoptimizer: bool = False
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, str], Path, Path]:
    git = git_state()
    gpu = require_idle_gpu()
    checkpoint = cpu_checkpoint_audit(config)
    samples, index, hashes, manifest_path, index_path = load_samples(config, garment)
    output_root = Path(config["garments"][garment]["output_root"])
    recovery: dict[str, Any] | None = None
    if output_root.exists():
        run_root = output_root / config["runtime"]["attempt"]
        status_path = run_root / "RUN_STATUS.json"
        status = read_json(status_path) if status_path.is_file() else {}
        checkpoints = list((run_root / "checkpoints").glob("*.pth")) if (run_root / "checkpoints").is_dir() else []
        state_records = run_root / "training/state_records.jsonl"
        if (
            not recover_preoptimizer
            or status.get("status") != "FAILED"
            or checkpoints
            or state_records.exists()
        ):
            raise FileExistsError(f"formal Teacher output root already exists: {output_root}")
        recovery = {
            "status": "AUTHORIZED_SAME_ATTEMPT_PREOPTIMIZER_RECOVERY",
            "prior_status_sha256": sha256_file(status_path),
            "prior_exception": status.get("exception"),
            "prior_optimizer_steps": 0,
            "prior_checkpoint_count": 0,
            "scientific_run_count_consumed": 0,
        }
    free = shutil.disk_usage(output_root.parent).free
    minimum = int(config["runtime"]["minimum_free_bytes"])
    if free < minimum:
        raise RuntimeError(f"storage gate failed: {free} < {minimum}")
    evidence = {
        "status": "PASS",
        "git": git,
        "gpu_gate": gpu,
        "storage_gate": {"status": "PASS", "free_bytes": free, "minimum_free_bytes": minimum},
        "base_checkpoint_audit": checkpoint,
        "formal_target_root": config["formal_targets"]["root"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "index_path": str(index_path),
        "index_sha256": sha256_file(index_path),
        "target_count": len(samples),
        "formal_training_record_count": 22,
        "quarantine_count": 2,
        "excluded_request_count_in_samples": 0,
        "recovery": recovery,
    }
    return evidence, samples, index, hashes, manifest_path, index_path


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config.resolve())
    if args.garment not in GARMENTS:
        raise RuntimeError("garment is outside the frozen three-garment contract")
    evidence, cpu_samples, index, asset_hashes, manifest_path, index_path = preflight(
        config, args.garment, recover_preoptimizer=args.recover_preoptimizer
    )
    if args.preflight_only:
        print(json.dumps(evidence, indent=2))
        return 0
    contract = config["garments"][args.garment]
    run_root = Path(contract["output_root"]) / config["runtime"]["attempt"]
    if args.recover_preoptimizer:
        prior_status = read_json(run_root / "RUN_STATUS.json")
        atomic_json(run_root / "audits/preoptimizer_failure_recovery_01.json", prior_status)
    else:
        run_root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        atomic_json(run_root / "audits/preflight.json", evidence)
        atomic_json(run_root / "contract/config_snapshot.json", config)
        atomic_json(run_root / "contract/formal_index_snapshot.json", index)
        atomic_json(run_root / "RUN_STATUS.json", {
            "task_id": TASK_ID, "status": "RUNNING", "stage": "PREFLIGHT_COMPLETE",
            "garment": args.garment, "optimizer_steps": 0, "paper_eligible": False,
        })
        seed = int(config["teacher"]["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
        base, protocol, restore = legacy.restore_base(
            config,
            Path(config["runtime"]["data_root"]),
            Path(config["runtime"]["assets_root"]),
            Path(config["runtime"]["availability_manifest"]),
        )
        device = base._xyz.device
        samples = [to_device_sample(row, device) for row in cpu_samples]
        del cpu_samples
        base_before = legacy.model_fingerprint(base)
        field = UnboundedGaussianDeltaField(base).to(device)
        initial_parameters = {
            name: tensor_sha(value) for name, value in field.named_parameters()
        }
        groups, group_names = field.parameter_groups(
            config["teacher"]["optimizer"]["geometry_lr"],
            config["teacher"]["optimizer"]["appearance_lr"],
        )
        for name, group in zip(group_names, groups, strict=True):
            group["name"] = name
        optimizer = torch.optim.Adam(groups)
        background = torch.ones(3, device=device)
        states = [target_free_state(sample, device) for sample in samples]
        metadata = {
            "garment": args.garment,
            "base_checkpoint_path": config["base"]["checkpoint_path"],
            "base_checkpoint_sha256": config["base"]["checkpoint_sha256"],
            "base_checkpoint_step": 60747,
            "formal_target_root": config["formal_targets"]["root"],
            "target_manifest_path": str(manifest_path),
            "target_manifest_sha256": evidence["manifest_sha256"],
            "target_index_path": str(index_path),
            "target_index_sha256": evidence["index_sha256"],
            "target_request_ids": [row["request_id"] for row in index["records"]],
            "target_count": len(samples),
            "quarantine_count_in_optimizer": 0,
            "loss_contract": CAPACITY_LOSS_NAME,
            "optimizer": config["teacher"]["optimizer"],
            "scheduler": None,
            "seed": seed,
            "total_steps": 1200,
            "execution_head": evidence["git"]["head"],
            "formal_base_resume_authorized": False,
        }
        checkpoints = [save_checkpoint(run_root, 0, field, optimizer, metadata)]
        gradient_seen = {name: 0 for name, value in field.named_parameters() if value.requires_grad}
        counts: Counter[str] = Counter()
        last_row: dict[str, Any] | None = None
        initial_loss: float | None = None
        for step in range(1, 1201):
            index_number = (step - 1) % len(samples)
            sample = samples[index_number]
            optimizer.zero_grad(set_to_none=True)
            rgb, alpha = render_direct(base, states[index_number], field(base), background)
            parts = loss_for_sample(rgb, alpha, sample, field, config["teacher"]["loss_weights"])
            if not torch.isfinite(parts["total"]):
                raise FloatingPointError(f"non-finite loss at optimizer step {step}")
            parts["total"].backward()
            for name, parameter in field.named_parameters():
                if parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(f"non-finite gradient at step {step}: {name}")
                    if bool(torch.count_nonzero(parameter.grad)):
                        gradient_seen[name] += 1
            norm = torch.nn.utils.clip_grad_norm_(
                field.parameters(), float(config["teacher"]["optimizer"]["gradient_clip_norm"])
            )
            if not torch.isfinite(norm):
                raise FloatingPointError(f"non-finite gradient norm at optimizer step {step}")
            optimizer.step()
            counts[sample["condition_id"]] += 1
            last_row = {
                "step": step,
                "condition_id": sample["condition_id"],
                "garment": args.garment,
                "view_denominator": len(samples),
                "loss": {name: float(value.detach()) for name, value in parts.items()},
                "gradient_norm": float(norm),
                "loss_finite": True,
                "gradient_finite": True,
            }
            if initial_loss is None:
                initial_loss = last_row["loss"]["total"]
            append_jsonl(run_root / "training/state_records.jsonl", last_row)
            if step % 10 == 0 or step in CHECKPOINT_STEPS:
                atomic_json(run_root / "RUN_STATUS.json", {
                    "task_id": TASK_ID, "status": "RUNNING", "stage": "TRAINING",
                    "garment": args.garment, "optimizer_steps": step, "current_step": step,
                    "loss": last_row["loss"]["total"], "loss_finite": True,
                    "gradient_finite": True, "paper_eligible": False,
                })
            if step in CHECKPOINT_STEPS:
                checkpoints.append(save_checkpoint(run_root, step, field, optimizer, metadata))
        if max(counts.values()) - min(counts.values()) > 1 or len(counts) != len(samples):
            raise RuntimeError(f"round-robin sampler is not balanced: {dict(counts)}")
        if args.garment == "O04" and set(counts.values()) != {150}:
            raise RuntimeError(f"O04 sampler count must be exactly 150/view: {dict(counts)}")
        final_parameters = {name: tensor_sha(value) for name, value in field.named_parameters()}
        changed = {name: initial_parameters[name] != value for name, value in final_parameters.items()}
        if not all(changed.values()) or not all(value > 0 for value in gradient_seen.values()):
            raise RuntimeError(f"trainable parameter change/gradient gate failed: {changed}/{gradient_seen}")
        base_after = legacy.model_fingerprint(base)
        if base_before != base_after:
            raise RuntimeError("frozen Base60747 tensors mutated")
        if sha256_file(Path(config["base"]["checkpoint_path"])) != config["base"]["checkpoint_sha256"]:
            raise RuntimeError("Base60747 checkpoint bytes mutated")
        for path, expected in asset_hashes.items():
            if sha256_file(Path(path)) != expected:
                raise RuntimeError(f"formal target asset mutated: {path}")
        lpips_metric, lpips_error = legacy.core.make_lpips_preserving_rng()
        if lpips_metric is None:
            raise RuntimeError(f"LPIPS unavailable: {lpips_error}")
        rows: list[dict[str, Any]] = []
        panels: list[tuple[str, torch.Tensor]] = []
        with torch.no_grad():
            for sample, state in zip(samples, states, strict=True):
                rgb, alpha = render_direct(base, state, field(base), background)
                metric = evaluate_view(rgb, alpha, sample, lpips_metric)
                rows.append({"condition_id": sample["condition_id"], **metric})
                save_png(run_root / "review/target" / f"{sample['condition_id']}.png", sample["target_edit_rgb"])
                save_png(run_root / "review/teacher" / f"{sample['condition_id']}.png", rgb.clamp(0, 1))
                panels.extend([
                    (f"{sample['condition_id']} target", sample["target_edit_rgb"]),
                    (f"{sample['condition_id']} teacher", rgb.clamp(0, 1)),
                ])
        contact_sheet(run_root / "review/comparison_contact_sheet.png", panels, columns=2)
        novel = novel_render_checks(run_root, base, field, samples, background)
        severe_count = sum(
            (not row["render_finite"])
            or row["background_alpha_mean"] > 0.05
            or row["silhouette_iou"] < 0.5
            for row in rows
        )
        abnormal = abnormal_direct(field, base, {
            "abnormal_gaussian": {
                "xyz_displacement_m": 0.25,
                "scale_ratio_max": 8.0,
                "opacity_saturation_low": 0.005,
                "opacity_saturation_high": 0.995,
                "outside_base_radius_multiplier": 1.5
            }
        })
        metric_names = (
            "garment_region_lpips", "silhouette_iou", "boundary_f",
            "protected_region_lpips", "protected_region_rgb_mae",
            "alpha_foreground_error", "background_alpha_mean",
        )
        metrics = {
            "schema_version": "canondressgs.subject00.base60747_formal_teacher_evaluation.v1",
            "task_id": TASK_ID,
            "garment": args.garment,
            "denominator": len(rows),
            "per_view": rows,
            "macro": {name: mean_optional(rows, name) for name in metric_names},
            "lpips_available": True,
            "severe_artifact_audit": {
                "status": "PASS_COMPLETED",
                "flagged_view_count": int(severe_count),
                "rule": "nonfinite OR background_alpha_mean>0.05 OR silhouette_iou<0.5",
            },
            "abnormal_gaussian_audit": abnormal,
            "novel_render_checks": novel,
            "human_visual_status": None,
            "scientific_pass": None,
            "paper_eligible": False,
        }
        atomic_json(run_root / "evaluations/final_metrics.json", metrics)
        final_checkpoint = checkpoints[-1]
        result = {
            "schema_version": "canondressgs.subject00.base60747_formal_teacher_result.v1",
            "task_id": TASK_ID,
            "garment": args.garment,
            "status": "TECHNICAL_PASS",
            "optimizer_steps": 1200,
            "current_step": 1200,
            "loss_initial": initial_loss,
            "loss_final": last_row["loss"]["total"] if last_row else None,
            "view_update_counts": dict(counts),
            "sampler_balanced": True,
            "target_count": len(samples),
            "evaluation_denominator": len(rows),
            "quarantine_sample_count": 0,
            "checkpoint_count": len(checkpoints),
            "checkpoints": checkpoints,
            "final_checkpoint": final_checkpoint,
            "trainable_parameter_changed": changed,
            "gradient_steps_nonzero": gradient_seen,
            "base_fingerprint_unchanged": True,
            "base_checkpoint_mutations": 0,
            "target_mutations": 0,
            "mask_mutations": 0,
            "nan_inf_status": "NONE",
            "oom_status": "NONE",
            "different_camera_status": novel["different_camera"]["status"],
            "different_pose_status": novel["different_pose"]["status"],
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
            "wall_seconds": time.perf_counter() - started,
            "metrics_path": str(run_root / "evaluations/final_metrics.json"),
            "human_visual_status": None,
            "scientific_pass": None,
            "paper_eligible": False,
            "paper_modifications": 0,
            "paper_final": False,
            "formal_base_resume_authorized": False,
            "restore_summary": restore,
            "formal_protocol_loaded": bool(protocol),
        }
        atomic_json(run_root / "training/training_result.json", result)
        atomic_json(run_root / "audits/post_training_integrity.json", {
            "status": "PASS",
            "base_fingerprint_before": base_before,
            "base_fingerprint_after": base_after,
            "base_checkpoint_sha256_unchanged": True,
            "formal_target_assets_unchanged": True,
            "trainable_parameter_changes": changed,
            "optimizer_steps": 1200,
            "paper_modifications": 0,
        })
        atomic_json(run_root / "RUN_STATUS.json", {
            "task_id": TASK_ID,
            "status": "COMPLETE",
            "stage": "TECHNICAL_PASS",
            "garment": args.garment,
            "optimizer_steps": 1200,
            "current_step": 1200,
            "loss": result["loss_final"],
            "final_checkpoint": final_checkpoint,
            "paper_eligible": False,
        })
        print(json.dumps(result, sort_keys=True))
        return 0
    except BaseException as error:
        failure = {
            "task_id": TASK_ID,
            "status": "FAILED",
            "garment": args.garment,
            "exception": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "paper_eligible": False,
        }
        atomic_json(run_root / "RUN_STATUS.json", failure)
        atomic_json(run_root / "audits/failure.json", failure)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--garment", required=True, choices=GARMENTS)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--recover-preoptimizer", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
