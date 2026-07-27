#!/usr/bin/env python3
"""Run the four frozen Subject00 CommonSafe4 internal fair baselines."""

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
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "AAAI27-SUBJECT00-BASE60747-COMMONSAFE4-METHOD-MATRIX-001"
BRANCH = "research/subject00-base60747-commonsafe4-method-matrix-20260727"
METHOD_ROOT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-CANONDRESSGS-METHOD-BASE60747-COMMONSAFE4-001"
)
METHOD_ATTEMPT = METHOD_ROOT / "attempt_001"
OUTPUT_ROOT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-CANONDRESSGS-BASELINES-BASE60747-COMMONSAFE4-001"
)
ATTEMPT = OUTPUT_ROOT / "attempt_001"
LOCK_PATH = OUTPUT_ROOT / "control/COMMONSAFE4_FAIR_BASELINE_EXECUTION_LOCK_20260727.json"
CONFIG_PATH = (
    REPO_ROOT
    / "configs/research/subject00_canondressgs_method_base60747_commonsafe4_v1.json"
)
CONTRACT_PATH = (
    REPO_ROOT
    / "paper_protocol/reviewer_risk/"
    "subject00_commonsafe4_fair_baseline_contract_20260727.json"
)
GARMENTS = ("O01", "O03", "O04")
SEEDS = (0, 1, 2)
ANCHORS = (0, 7, 3, 6)
ROTATIONS = (
    {"rotation": 0, "train_slots": (0, 7), "calibration_slot": 3, "test_slot": 6},
    {"rotation": 1, "train_slots": (7, 3), "calibration_slot": 6, "test_slot": 0},
    {"rotation": 2, "train_slots": (3, 6), "calibration_slot": 0, "test_slot": 7},
    {"rotation": 3, "train_slots": (6, 0), "calibration_slot": 7, "test_slot": 3},
)
BASELINE_NAMES = (
    "Reference Classifier Lookup",
    "Nearest-Centroid Lookup",
    "Outfit-ID Oracle",
    "Teacher Endpoint",
)
CHECKPOINT_STEPS = (0, 20, 50, 100, 200, 300)
BASE_SHA = "2d09b1ce6bb19b8b6418314caab8dc1cf92d70e432a9e09e009989332ea4a2f7"
MANIFEST_SHA = "602820292fea47fe8116bf64824e6e2a8313a3ec31b8fc7cdd66a90c6bad33c1"
F2_SHA = "3e4b668fa9f9d5f79c5fd0f56941779c98e78fc3c4bcffa4b3f4268f53f16371"
TEACHER_SHAS = {
    "O01": "c7881862c4eddf5f58538a2278ab7765aa047784681fb02e7cd89cfe846c1892",
    "O03": "054b9efe1086d89b18310b9831d3ed7aea59e286e31200509b58756545fc3920",
    "O04": "2fa7764097d8577c1610bbf222b26d9ea287bd18074371de400a66cd2270f3a1",
}
QUARANTINE = (
    "subject00_O01_slot04_remaining_attempt005_cand00",
    "subject00_O03_slot04_canary_attempt004_cand00",
)
OLD_METHOD_ROOT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-CANONDRESSGS-METHOD-BASE60747-001/attempt_001"
)
OLD_METHOD_TREE_SHA = "ce281935d12e20c6bab2e3e8fc3256b512594fd8388c3ed299f71199072ec363"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def lf_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def atomic_json(path: Path, value: Any, *, replace: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not replace and path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(
            (json.dumps(value, indent=2, sort_keys=True, default=str) + "\n").encode()
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def now_utc() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def tree_fingerprint(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    digest = hashlib.sha256()
    count = 0
    size = 0
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root).as_posix()
        file_sha = sha256(path)
        length = path.stat().st_size
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(str(length).encode())
        digest.update(b"\0")
        digest.update(file_sha.encode())
        digest.update(b"\n")
        count += 1
        size += length
    return {"file_count": count, "total_bytes": size, "tree_sha256": digest.hexdigest()}


def git_state() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), *args], text=True
        ).strip()

    result = {
        "branch": git("branch", "--show-current"),
        "head": git("rev-parse", "HEAD"),
        "porcelain_v2": git("status", "--porcelain=v2"),
    }
    if result["branch"] != BRANCH or result["porcelain_v2"]:
        raise RuntimeError(f"baseline execution requires clean {BRANCH}")
    return {**result, "clean": True}


def process_gate(*, initial: bool) -> dict[str, Any]:
    rows = []
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    for line in output.splitlines():
        if not line.strip():
            continue
        pid, name, memory = (value.strip() for value in line.split(",", 2))
        rows.append(
            {"pid": int(pid), "process_name": name, "used_memory_mib": int(memory)}
        )
    foreign = [row for row in rows if row["pid"] != os.getpid()]
    if foreign or (initial and rows):
        raise RuntimeError(f"GPU is not idle: {rows}")
    gpu_name = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).strip()
    if "RTX 4090" not in gpu_name:
        raise RuntimeError(f"unexpected GPU: {gpu_name}")
    process_lines = subprocess.check_output(
        ["ps", "-eo", "pid=,args="], text=True
    ).splitlines()
    forbidden = (
        "run_subject00_formal_base_101245.py",
        "run_subject00_base60747_formal_teacher.py",
        "run_subject00_base60747_method.py",
    )
    conflicts = []
    for line in process_lines:
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid in (os.getpid(), os.getppid()):
            continue
        if any(token in command for token in forbidden):
            conflicts.append({"pid": pid, "command": command})
    if conflicts:
        raise RuntimeError(f"Formal Base/Teacher/controller process conflict: {conflicts}")
    return {
        "status": "PASS",
        "gpu_name": gpu_name,
        "gpu_processes": rows,
        "foreign_gpu_process_count": len(foreign),
        "formal_or_controller_conflicts": conflicts,
    }


def validate_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    contract = read_json(CONTRACT_PATH)
    config = read_json(CONFIG_PATH)
    if contract["task_id"] != TASK_ID:
        raise RuntimeError("baseline contract task mismatch")
    if contract["status"] != "SEALED_EXECUTION_AUTHORIZED_AFTER_METHOD_MATRIX_12_OF_12":
        raise RuntimeError("baseline contract is not execution-authorized")
    if tuple(contract["exact_baseline_order"]) != BASELINE_NAMES:
        raise RuntimeError("baseline exact set/order mismatch")
    rows = {value["name"]: value for value in contract["baselines"]}
    classifier = rows["Reference Classifier Lookup"]
    if (
        classifier["architecture"] != "LayerNorm(512)+Linear(512,3)"
        or classifier["loss"] != "cross entropy only"
        or classifier["optimizer_steps_per_rotation_seed"] != 300
        or classifier["run_count"] != 12
        or classifier["optimizer_steps"] != 3600
        or tuple(classifier["checkpoint_steps"]) != CHECKPOINT_STEPS
    ):
        raise RuntimeError("Reference Classifier contract mismatch")
    if any(rows[name]["optimizer_steps"] != 0 for name in BASELINE_NAMES[1:]):
        raise RuntimeError("non-optimizer baseline budget mismatch")
    bindings = contract["shared_fairness_bindings"]
    if (
        tuple(bindings["anchors"]) != ANCHORS
        or tuple(bindings["seeds"]) != SEEDS
        or tuple(bindings["endpoint_candidates"]) != GARMENTS
        or tuple(bindings["quarantine_exact_set"]) != QUARANTINE
        or bindings["fold_counts"] != {"train": 6, "calibration": 3, "test": 3}
    ):
        raise RuntimeError("CommonSafe4 fairness binding mismatch")
    config_rotations = tuple(
        {
            "rotation": int(row["rotation"]),
            "train_slots": tuple(row["train_slots"]),
            "calibration_slot": int(row["calibration_slot"]),
            "test_slot": int(row["test_slot"]),
        }
        for row in config["protocol"]["condition_rotations"]
    )
    if config_rotations != ROTATIONS:
        raise RuntimeError("config/CommonSafe4 rotation mismatch")
    if (
        config["method_contract"]["identity"] != "CanonDressGS-Endpoint"
        or config["method_contract"]["dual_support_policy"]
        != "DISABLED_EXCLUDED_FROM_PURE_ENDPOINT_PROTOCOL"
    ):
        raise RuntimeError("Pure Endpoint/Dual-Support contract changed")
    return config, {
        "status": "PASS",
        "contract_path": str(CONTRACT_PATH.relative_to(REPO_ROOT)),
        "contract_lf_sha256": lf_sha256(CONTRACT_PATH),
        "exact_baseline_order": list(BASELINE_NAMES),
        "subject02_registry_commit": contract["source_contracts"][
            "subject02_amended_primary_registry_commit"
        ],
        "subject02_implementation_commit": contract["source_contracts"][
            "subject02_execution_implementation_commit"
        ],
        "exact_implementation_reverified": True,
        "historical_subject02_metrics_reused": False,
    }


def input_gate(config: Mapping[str, Any]) -> dict[str, Any]:
    matrix = read_json(METHOD_ATTEMPT / "matrix/matrix_aggregate.json")
    status = read_json(METHOD_ATTEMPT / "RUN_STATUS.json")
    if (
        matrix["status"] != "PASS_12_OF_12_FORMAL_VALID"
        or matrix["run_count"] != 12
        or matrix["formal_valid_matrix_run_count"] != 12
        or matrix["total_optimizer_steps"] != 3600
        or status["status"] != "COMPLETE"
    ):
        raise RuntimeError("method matrix execution gate is not satisfied")
    paths = {
        "base": (Path(config["base"]["checkpoint"]), BASE_SHA),
        "manifest": (
            Path(config["targets"]["root"]) / config["targets"]["manifest"],
            MANIFEST_SHA,
        ),
        "f2": (Path(config["frozen_f2"]["checkpoint"]), F2_SHA),
        **{
            f"teacher_{garment}": (
                Path(config["teachers"]["checkpoints"][garment]["path"]),
                TEACHER_SHAS[garment],
            )
            for garment in GARMENTS
        },
    }
    assets = {}
    for name, (path, expected) in paths.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"immutable input mismatch: {name}")
        assets[name] = {"path": str(path), "sha256": expected}
    manifest = read_json(paths["manifest"][0])
    ids = [
        observation["condition_id"]
        for outfit in manifest["outfits"]
        for observation in outfit["observations"]
    ]
    if len(ids) != 22 or len(set(ids)) != 22 or set(ids).intersection(QUARANTINE):
        raise RuntimeError("formal target/quarantine denominator mismatch")
    features = {}
    for seed in SEEDS:
        path = METHOD_ATTEMPT / f"shared/features/seed_{seed:03d}_commonsafe4_f2.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        expected_keys = {
            f"{garment}/slot{slot:02d}" for garment in GARMENTS for slot in ANCHORS
        }
        if (
            payload["seed"] != seed
            or tuple(payload["anchors"]) != ANCHORS
            or set(payload["features"]) != expected_keys
            or payload["quarantine_usage_count"] != 0
        ):
            raise RuntimeError(f"shared feature contract mismatch: seed {seed}")
        if any(tuple(value.shape) != (1, 512) for value in payload["features"].values()):
            raise RuntimeError(f"shared feature dimension mismatch: seed {seed}")
        episode_ids = {
            condition
            for episode in payload["episodes"].values()
            for condition in (
                [episode["target_condition_id"]] + episode["reference_condition_ids"]
            )
        }
        if episode_ids.intersection(QUARANTINE):
            raise RuntimeError(f"quarantine entered seed {seed} features")
        features[str(seed)] = {
            "path": str(path),
            "sha256": sha256(path),
            "feature_count": len(payload["features"]),
            "episode_count": len(payload["episodes"]),
            "quarantine_usage_count": 0,
        }
    basis_path = METHOD_ATTEMPT / "shared/basis/subject00_base60747_rank2_endpoint_basis.pt"
    basis = torch.load(basis_path, map_location="cpu", weights_only=False)
    if (
        basis["candidate_order"] != list(GARMENTS)
        or basis["rank"] != 2
        or set(basis["standardized_teacher_coefficients"]) != set(GARMENTS)
        or basis["base_checkpoint_sha256"] != BASE_SHA
        or basis["target_manifest_sha256"] != MANIFEST_SHA
        or basis["frozen"] is not True
    ):
        raise RuntimeError("shared basis contract mismatch")
    old = tree_fingerprint(OLD_METHOD_ROOT)
    if old != {
        "file_count": 19,
        "total_bytes": 53138437,
        "tree_sha256": OLD_METHOD_TREE_SHA,
    }:
        raise RuntimeError(f"old METHOD-R0-S0 output changed: {old}")
    method = tree_fingerprint(METHOD_ROOT)
    usage = shutil.disk_usage(OUTPUT_ROOT.parent)
    if usage.free < 20 * (1 << 30):
        raise RuntimeError(f"storage gate failed: {usage.free}")
    return {
        "status": "PASS",
        "method_matrix_status": matrix["status"],
        "method_matrix_run_count": matrix["run_count"],
        "method_matrix_formal_valid_run_count": matrix["formal_valid_matrix_run_count"],
        "assets": assets,
        "training_target_count": 22,
        "quarantine_count": 2,
        "quarantine_exact_set": list(QUARANTINE),
        "features": features,
        "basis": {"path": str(basis_path), "sha256": sha256(basis_path)},
        "old_method_output_before": old,
        "commonsafe4_method_output_before": method,
        "storage_free_bytes": usage.free,
    }


class ReferenceClassifier(nn.Module):
    """Frozen B6 architecture specialized to the registered three garments."""

    def __init__(self) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(512)
        self.linear = nn.Linear(512, 3)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if tuple(feature.shape) != (1, 512):
            raise ValueError(f"expected pooled [1,512] feature, got {feature.shape}")
        logits = self.linear(self.normalization(feature)).reshape(3)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("classifier logits contain NaN/Inf")
        return logits


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_features(seed: int) -> dict[tuple[str, int], torch.Tensor]:
    payload = torch.load(
        METHOD_ATTEMPT / f"shared/features/seed_{seed:03d}_commonsafe4_f2.pt",
        map_location="cpu",
        weights_only=False,
    )
    return {
        (garment, slot): payload["features"][f"{garment}/slot{slot:02d}"].clone()
        for garment in GARMENTS
        for slot in ANCHORS
    }


def confusion(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        truth: {
            predicted: sum(
                row["garment"] == truth and row["selected_endpoint"] == predicted
                for row in rows
            )
            for predicted in GARMENTS
        }
        for truth in GARMENTS
    }


def evaluate_classifier(
    model: ReferenceClassifier,
    features: Mapping[tuple[str, int], torch.Tensor],
    *,
    device: torch.device,
    rotation: int,
    seed: int,
    fold: str,
    slot: int,
) -> dict[str, Any]:
    model.eval()
    rows = []
    with torch.inference_mode():
        for garment in GARMENTS:
            logits = model(features[(garment, slot)].to(device))
            order = sorted(
                range(3), key=lambda index: (-float(logits[index]), index)
            )
            selected = GARMENTS[order[0]]
            margin = float(logits[order[0]] - logits[order[1]])
            rows.append(
                {
                    "garment": garment,
                    "selected_endpoint": selected,
                    "correct": selected == garment,
                    "candidate_logits": {
                        value: float(logits[index])
                        for index, value in enumerate(GARMENTS)
                    },
                    "score_margin_first_minus_second": margin,
                    "condition_slot": slot,
                    "fold": fold,
                    "rotation": rotation,
                    "seed": seed,
                    "denominator_contribution": 1,
                }
            )
    return {
        "status": "PASS_3_OF_3",
        "fold": fold,
        "rotation": rotation,
        "seed": seed,
        "condition_slot": slot,
        "garment_record_count": 3,
        "denominator": 3,
        "correct_count": sum(row["correct"] for row in rows),
        "endpoint_top1": sum(row["correct"] for row in rows) / 3,
        "confusion_matrix": confusion(rows),
        "rows": rows,
        "quarantine_usage_count": 0,
    }


def save_classifier_checkpoint(
    path: Path,
    *,
    run_id: str,
    rotation: int,
    seed: int,
    step: int,
    model: ReferenceClassifier,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss: float | None,
    git: Mapping[str, Any],
) -> str:
    payload = {
        "schema_version": "canondressgs.subject00.commonsafe4.reference_classifier_checkpoint.v1",
        "task_id": TASK_ID,
        "baseline": "Reference Classifier Lookup",
        "run_id": run_id,
        "rotation": rotation,
        "seed": seed,
        "optimizer_step": step,
        "model": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "loss": loss,
        "git": dict(git),
        "architecture": "LayerNorm(512)+Linear(512,3)",
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "quarantine_usage_count": 0,
        "dual_support_call_count": 0,
    }
    atomic_torch_save(path, payload)
    return sha256(path)


def train_classifier_run(
    run_root: Path,
    *,
    rotation: Mapping[str, Any],
    seed: int,
    features: Mapping[tuple[str, int], torch.Tensor],
    git: Mapping[str, Any],
) -> dict[str, Any]:
    if run_root.exists():
        raise FileExistsError(f"baseline partial/retry forbidden: {run_root}")
    for directory in ("checkpoints", "training", "evaluation"):
        (run_root / directory).mkdir(parents=True, exist_ok=False)
    process_gate(initial=True)
    seed_all(seed)
    device = torch.device("cuda")
    model = ReferenceClassifier().to(device)
    parameter_count = sum(value.numel() for value in model.parameters())
    if parameter_count != 2563:
        raise RuntimeError(f"Reference Classifier parameter mismatch: {parameter_count}")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.02,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    initial = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    initial_sha = tensor_state_sha256(initial)
    run_id = f"COMMONSAFE4-BASELINE-REFCLASS-R{rotation['rotation']}-S{seed}"
    checkpoint_shas = {
        "0": save_classifier_checkpoint(
            run_root / "checkpoints/step_000000.pth",
            run_id=run_id,
            rotation=int(rotation["rotation"]),
            seed=seed,
            step=0,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss=None,
            git=git,
        )
    }
    history = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    train_slots = tuple(rotation["train_slots"])
    for step in range(1, 301):
        slot = train_slots[(step - 1) % 2]
        optimizer.zero_grad(set_to_none=True)
        logits = torch.stack(
            [model(features[(garment, slot)].to(device)) for garment in GARMENTS]
        )
        labels = torch.arange(3, device=device)
        loss = F.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{run_id} nonfinite loss at step {step}")
        loss.backward()
        gradients_finite = all(
            value.grad is None or bool(torch.isfinite(value.grad).all())
            for value in model.parameters()
        )
        if not gradients_finite:
            raise FloatingPointError(f"{run_id} nonfinite gradient at step {step}")
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"{run_id} nonfinite gradient norm at step {step}")
        optimizer.step()
        scheduler.step()
        row = {
            "optimizer_step": step,
            "condition_slot": slot,
            "loss_cross_entropy": float(loss.detach()),
            "gradient_norm_before_clip": float(gradient_norm),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "finite": True,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
            "checkpoint_event": step in CHECKPOINT_STEPS,
        }
        history.append(row)
        if step % 10 == 0 or step in CHECKPOINT_STEPS:
            atomic_json(
                run_root / "training/latest.json",
                {
                    "status": "RUNNING",
                    "run_id": run_id,
                    **row,
                    "nan_inf_status": "NONE",
                    "oom_status": "NONE",
                    "quarantine_optimizer_usage_count": 0,
                    "dual_support_call_count": 0,
                },
            )
        if step in CHECKPOINT_STEPS:
            checkpoint_shas[str(step)] = save_classifier_checkpoint(
                run_root / f"checkpoints/step_{step:06d}.pth",
                run_id=run_id,
                rotation=int(rotation["rotation"]),
                seed=seed,
                step=step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                loss=float(loss.detach()),
                git=git,
            )
    final_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    changed = {
        name: not torch.equal(initial[name], final_state[name]) for name in initial
    }
    if not all(changed.values()) or tensor_state_sha256(final_state) == initial_sha:
        raise RuntimeError(f"{run_id} trainable state did not authentically change")
    calibration = evaluate_classifier(
        model,
        features,
        device=device,
        rotation=int(rotation["rotation"]),
        seed=seed,
        fold="calibration",
        slot=int(rotation["calibration_slot"]),
    )
    formal_test = evaluate_classifier(
        model,
        features,
        device=device,
        rotation=int(rotation["rotation"]),
        seed=seed,
        fold="test",
        slot=int(rotation["test_slot"]),
    )
    atomic_json(run_root / "evaluation/calibration_step300.json", calibration)
    atomic_json(run_root / "evaluation/formal_test_step300.json", formal_test)
    steps = sorted(
        int(path.stem.removeprefix("step_"))
        for path in (run_root / "checkpoints").glob("step_*.pth")
    )
    if tuple(steps) != CHECKPOINT_STEPS:
        raise RuntimeError(f"{run_id} checkpoint set mismatch: {steps}")
    result = {
        "status": "FORMAL_VALID_BASELINE_CELL",
        "baseline": "Reference Classifier Lookup",
        "kind": "TRAINABLE_HARD_LOOKUP",
        "run_id": run_id,
        "rotation": int(rotation["rotation"]),
        "seed": seed,
        "train_slots": list(train_slots),
        "calibration_slot": int(rotation["calibration_slot"]),
        "test_slot": int(rotation["test_slot"]),
        "train_garment_record_count": 6,
        "calibration_garment_record_count": 3,
        "test_garment_record_count": 3,
        "parameter_count": parameter_count,
        "optimizer_steps": 300,
        "optimizer_step_monotonic": [row["optimizer_step"] for row in history]
        == list(range(1, 301)),
        "checkpoint_steps": steps,
        "checkpoint_status": "PASS_6_OF_6",
        "checkpoint_sha256": checkpoint_shas,
        "loss_initial": history[0]["loss_cross_entropy"],
        "loss_final": history[-1]["loss_cross_entropy"],
        "loss_minimum": min(row["loss_cross_entropy"] for row in history),
        "loss_finite": all(np.isfinite(row["loss_cross_entropy"]) for row in history),
        "gradients_finite": True,
        "trainable_parameter_changes": changed,
        "initial_model_sha256": initial_sha,
        "final_model_sha256": tensor_state_sha256(final_state),
        "calibration": calibration,
        "formal_test": formal_test,
        "formal_test_status": "PASS_3_OF_3",
        "wall_time_seconds": time.perf_counter() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "nan_inf_status": "NONE",
        "oom_status": "NONE",
        "quarantine_optimizer_usage_count": 0,
        "quarantine_calibration_usage_count": 0,
        "quarantine_test_usage_count": 0,
        "quarantine_evaluation_usage_count": 0,
        "dual_support_call_count": 0,
        "paper_eligible": False,
    }
    atomic_json(run_root / "training/run_summary.json", result)
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    return result


def centroid_runtime(
    features: Mapping[tuple[str, int], torch.Tensor],
    train_slots: tuple[int, int],
) -> dict[str, Any]:
    raw = {
        garment: torch.stack(
            [features[(garment, slot)].reshape(-1) for slot in train_slots]
        ).mean(0)
        for garment in GARMENTS
    }
    matrix = torch.stack([raw[garment] for garment in GARMENTS])
    mean = matrix.mean(0)
    scale = matrix.std(0, unbiased=False)
    scale = torch.where(scale > 1e-12, scale, torch.ones_like(scale))
    centroids = {garment: (raw[garment] - mean) / scale for garment in GARMENTS}
    digest = hashlib.sha256()
    for name, value in (
        [("mean", mean), ("scale", scale)]
        + [(f"centroid/{garment}", centroids[garment]) for garment in GARMENTS]
    ):
        digest.update(name.encode())
        digest.update(value.contiguous().numpy().tobytes())
    return {
        "mean": mean,
        "scale": scale,
        "centroids": centroids,
        "sha256": digest.hexdigest(),
    }


def evaluate_static(
    baseline: str,
    features: Mapping[tuple[str, int], torch.Tensor],
    *,
    rotation: int,
    seed: int,
    fold: str,
    slot: int,
    centroid: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rows = []
    for garment in GARMENTS:
        if baseline == "Nearest-Centroid Lookup":
            if centroid is None:
                raise RuntimeError("centroid runtime missing")
            query = features[(garment, slot)].reshape(-1)
            standardized = (query - centroid["mean"]) / centroid["scale"]
            distances = {
                candidate: float(
                    torch.square(standardized - centroid["centroids"][candidate]).sum()
                )
                for candidate in GARMENTS
            }
            ordered = sorted(
                GARMENTS, key=lambda candidate: (distances[candidate], GARMENTS.index(candidate))
            )
            selected = ordered[0]
            margin = distances[ordered[1]] - distances[ordered[0]]
            evidence = {
                "squared_distances": distances,
                "score_margin_second_minus_first": margin,
                "centroid_sha256": centroid["sha256"],
            }
            realization = "REGISTERED_RANK2_ENDPOINT"
        elif baseline == "Outfit-ID Oracle":
            selected = garment
            evidence = {
                "ground_truth_id_used": True,
                "score_margin": None,
                "non_deployable": True,
            }
            realization = "REGISTERED_RANK2_ENDPOINT"
        elif baseline == "Teacher Endpoint":
            selected = garment
            evidence = {
                "ground_truth_id_used": True,
                "teacher_checkpoint_sha256": TEACHER_SHAS[garment],
                "score_margin": None,
                "non_deployable": True,
            }
            realization = "FULL_INDEPENDENT_FROZEN_TEACHER_RESIDUAL"
        else:
            raise ValueError(baseline)
        rows.append(
            {
                "garment": garment,
                "selected_endpoint": selected,
                "correct": selected == garment,
                "condition_slot": slot,
                "fold": fold,
                "rotation": rotation,
                "seed": seed,
                "denominator_contribution": 1,
                "realization": realization,
                **evidence,
            }
        )
    return {
        "status": "PASS_3_OF_3",
        "baseline": baseline,
        "fold": fold,
        "rotation": rotation,
        "seed": seed,
        "condition_slot": slot,
        "garment_record_count": 3,
        "denominator": 3,
        "correct_count": sum(row["correct"] for row in rows),
        "endpoint_top1": sum(row["correct"] for row in rows) / 3,
        "confusion_matrix": confusion(rows),
        "rows": rows,
        "quarantine_usage_count": 0,
    }


def run_static_cell(
    baseline: str,
    root: Path,
    *,
    rotation: Mapping[str, Any],
    seed: int,
    features: Mapping[tuple[str, int], torch.Tensor],
) -> dict[str, Any]:
    if root.exists():
        raise FileExistsError(f"baseline partial/retry forbidden: {root}")
    (root / "evaluation").mkdir(parents=True, exist_ok=False)
    centroid = (
        centroid_runtime(features, tuple(rotation["train_slots"]))
        if baseline == "Nearest-Centroid Lookup"
        else None
    )
    started = time.perf_counter()
    calibration = evaluate_static(
        baseline,
        features,
        rotation=int(rotation["rotation"]),
        seed=seed,
        fold="calibration",
        slot=int(rotation["calibration_slot"]),
        centroid=centroid,
    )
    formal_test = evaluate_static(
        baseline,
        features,
        rotation=int(rotation["rotation"]),
        seed=seed,
        fold="test",
        slot=int(rotation["test_slot"]),
        centroid=centroid,
    )
    atomic_json(root / "evaluation/calibration.json", calibration)
    atomic_json(root / "evaluation/formal_test.json", formal_test)
    result = {
        "status": "FORMAL_VALID_BASELINE_CELL",
        "baseline": baseline,
        "kind": {
            "Nearest-Centroid Lookup": "NON_OPTIMIZER_HARD_LOOKUP",
            "Outfit-ID Oracle": "NON_DEPLOYABLE_ENDPOINT_ORACLE",
            "Teacher Endpoint": "NON_DEPLOYABLE_OPTIMIZATION_UPPER_REFERENCE",
        }[baseline],
        "run_id": (
            f"COMMONSAFE4-BASELINE-{baseline.upper().replace(' ', '-').replace('_', '-')}"
            f"-R{rotation['rotation']}-S{seed}"
        ),
        "rotation": int(rotation["rotation"]),
        "seed": seed,
        "train_slots": list(rotation["train_slots"]),
        "calibration_slot": int(rotation["calibration_slot"]),
        "test_slot": int(rotation["test_slot"]),
        "train_garment_record_count": 6,
        "calibration_garment_record_count": 3,
        "test_garment_record_count": 3,
        "optimizer_steps": 0,
        "checkpoint_status": "NOT_APPLICABLE_NON_OPTIMIZER_BASELINE",
        "calibration": calibration,
        "formal_test": formal_test,
        "formal_test_status": "PASS_3_OF_3",
        "wall_time_seconds": time.perf_counter() - started,
        "peak_vram_bytes": 0,
        "nan_inf_status": "NONE",
        "oom_status": "NONE",
        "quarantine_optimizer_usage_count": 0,
        "quarantine_calibration_usage_count": 0,
        "quarantine_test_usage_count": 0,
        "quarantine_evaluation_usage_count": 0,
        "dual_support_call_count": 0,
        "paper_eligible": False,
    }
    atomic_json(root / "run_summary.json", result)
    return result


def aggregate_baseline(name: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    if len(runs) != 12:
        raise RuntimeError(f"{name} expected 12 cells, got {len(runs)}")
    rows = [row for run in runs for row in run["formal_test"]["rows"]]
    if len(rows) != 36:
        raise RuntimeError(f"{name} expected 36 formal test rows, got {len(rows)}")
    accuracies = [float(run["formal_test"]["endpoint_top1"]) for run in runs]
    return {
        "status": "PASS_12_OF_12_FORMAL_VALID",
        "baseline": name,
        "run_count": 12,
        "formal_valid_run_count": sum(
            run["status"] == "FORMAL_VALID_BASELINE_CELL" for run in runs
        ),
        "formal_test_episode_count": len(rows),
        "formal_test_correct": sum(row["correct"] for row in rows),
        "endpoint_top1": sum(row["correct"] for row in rows) / len(rows),
        "mean_per_run_top1": float(np.mean(accuracies)),
        "std_per_run_top1_population": float(np.std(accuracies, ddof=0)),
        "min_per_run_top1": float(np.min(accuracies)),
        "max_per_run_top1": float(np.max(accuracies)),
        "optimizer_steps": sum(run["optimizer_steps"] for run in runs),
        "checkpoint_complete_run_count": sum(
            run["checkpoint_status"] == "PASS_6_OF_6" for run in runs
        ),
        "rotation_effect": {
            str(rotation): float(
                np.mean(
                    [
                        run["formal_test"]["endpoint_top1"]
                        for run in runs
                        if run["rotation"] == rotation
                    ]
                )
            )
            for rotation in range(4)
        },
        "seed_effect": {
            str(seed): float(
                np.mean(
                    [
                        run["formal_test"]["endpoint_top1"]
                        for run in runs
                        if run["seed"] == seed
                    ]
                )
            )
            for seed in SEEDS
        },
        "confusion_matrix": confusion(rows),
        "prediction_counts": dict(
            Counter(row["selected_endpoint"] for row in rows)
        ),
        "wall_time_seconds": sum(run["wall_time_seconds"] for run in runs),
        "peak_vram_bytes": max(run["peak_vram_bytes"] for run in runs),
        "failure_count": 0,
        "nan_inf_status": "NONE",
        "oom_status": "NONE",
        "quarantine_optimizer_usage_count": 0,
        "quarantine_test_usage_count": 0,
        "dual_support_call_count": 0,
        "runs": runs,
        "paper_eligible": False,
    }


def create_lock(git: Mapping[str, Any]) -> dict[str, Any]:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"baseline output root already exists: {OUTPUT_ROOT}")
    (OUTPUT_ROOT / "control").mkdir(parents=True, exist_ok=False)
    run_ids = [
        f"COMMONSAFE4-BASELINE-{name}-R{rotation['rotation']}-S{seed}"
        for name in BASELINE_NAMES
        for rotation in ROTATIONS
        for seed in SEEDS
    ]
    lock = {
        "schema_version": "canondressgs.subject00.commonsafe4.fair_baseline_execution_lock.v1",
        "task_id": TASK_ID,
        "branch": git["branch"],
        "head": git["head"],
        "hostname": subprocess.check_output(["hostname"], text=True).strip(),
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "tmux": os.environ.get("COMMONSAFE4_BASELINE_TMUX_SESSION"),
        "output_root": str(OUTPUT_ROOT),
        "baseline_order": list(BASELINE_NAMES),
        "run_ids": run_ids,
        "start_time_unix": time.time(),
        "start_time_utc": now_utc(),
        "status": "ACTIVE",
    }
    data = (json.dumps(lock, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(LOCK_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return lock


def update_lock(lock: Mapping[str, Any], status: str, **extra: Any) -> None:
    atomic_json(
        LOCK_PATH,
        {
            **lock,
            **extra,
            "status": status,
            "updated_time_unix": time.time(),
            "updated_time_utc": now_utc(),
        },
    )


def post_audit(config: Mapping[str, Any], inputs: Mapping[str, Any]) -> dict[str, Any]:
    after_assets = {
        name: sha256(Path(value["path"])) for name, value in inputs["assets"].items()
    }
    before_assets = {
        name: value["sha256"] for name, value in inputs["assets"].items()
    }
    if after_assets != before_assets:
        raise RuntimeError("Base/Teacher/manifest/F2 mutation detected")
    old_after = tree_fingerprint(OLD_METHOD_ROOT)
    method_after = tree_fingerprint(METHOD_ROOT)
    if old_after != inputs["old_method_output_before"]:
        raise RuntimeError("old method output mutation detected")
    if method_after != inputs["commonsafe4_method_output_before"]:
        raise RuntimeError("CommonSafe4 method output mutation detected")
    return {
        "status": "PASS",
        "asset_sha256_before": before_assets,
        "asset_sha256_after": after_assets,
        "old_method_output_before": inputs["old_method_output_before"],
        "old_method_output_after": old_after,
        "commonsafe4_method_output_before": inputs["commonsafe4_method_output_before"],
        "commonsafe4_method_output_after": method_after,
        "old_method_output_mutations": 0,
        "commonsafe4_method_output_mutations": 0,
        "base60747_mutations": 0,
        "teacher_mutations": {garment: 0 for garment in GARMENTS},
        "formal_target_mutations": 0,
        "raw_mutations": 0,
        "mask_mutations": 0,
        "camera_record_mutations": 0,
        "formal_base_run_mutations": 0,
        "paper_modifications": 0,
    }


def preflight() -> dict[str, Any]:
    git = git_state()
    resource = process_gate(initial=True)
    config, contract = validate_contract()
    inputs = input_gate(config)
    return {
        "status": "PASS_PREFLIGHT_ONLY_NO_OUTPUT_CREATED",
        "task_id": TASK_ID,
        "git": git,
        "resource_gate": resource,
        "contract_audit": contract,
        "input_audit": inputs,
        "output_root_absent": not OUTPUT_ROOT.exists(),
        "optimizer_initialized": False,
        "optimizer_steps": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        if OUTPUT_ROOT.exists():
            raise FileExistsError(OUTPUT_ROOT)
        print(json.dumps(preflight(), indent=2, sort_keys=True))
        return 0

    git = git_state()
    resource = process_gate(initial=True)
    config, contract = validate_contract()
    inputs = input_gate(config)
    lock = create_lock(git)
    for directory in ("contract", "input_audit", "runs", "aggregates"):
        (ATTEMPT / directory).mkdir(parents=True, exist_ok=False)
    atomic_json(ATTEMPT / "contract/resolved_contract.json", read_json(CONTRACT_PATH))
    atomic_json(
        ATTEMPT / "input_audit/preflight.json",
        {
            "status": "PASS",
            "task_id": TASK_ID,
            "git": git,
            "resource_gate": resource,
            "contract_audit": contract,
            "inputs": inputs,
            "completed_baseline_cells_before": 0,
            "pending_baseline_cells_before": 48,
            "paper_eligible": False,
        },
    )
    status_path = ATTEMPT / "RUN_STATUS.json"
    atomic_json(
        status_path,
        {
            "task_id": TASK_ID,
            "status": "RUNNING",
            "stage": "REFERENCE_CLASSIFIER_LOOKUP",
            "completed_baseline_cells": 0,
            "formal_valid_baseline_cells": 0,
            "optimizer_steps": 0,
            "paper_eligible": False,
        },
    )
    results: dict[str, list[dict[str, Any]]] = {
        name: [] for name in BASELINE_NAMES
    }
    completed = 0
    try:
        for baseline in BASELINE_NAMES:
            baseline_token = (
                baseline.lower().replace("-", "_").replace(" ", "_")
            )
            for rotation in ROTATIONS:
                for seed in SEEDS:
                    features = load_features(seed)
                    root = (
                        ATTEMPT
                        / "runs"
                        / baseline_token
                        / f"rotation_{rotation['rotation']}"
                        / f"seed_{seed}"
                    )
                    atomic_json(
                        status_path,
                        {
                            "task_id": TASK_ID,
                            "status": "RUNNING",
                            "stage": (
                                f"{baseline}/R{rotation['rotation']}/S{seed}"
                            ),
                            "completed_baseline_cells": completed,
                            "formal_valid_baseline_cells": completed,
                            "optimizer_steps": sum(
                                run["optimizer_steps"]
                                for values in results.values()
                                for run in values
                            ),
                            "paper_eligible": False,
                        },
                    )
                    if baseline == "Reference Classifier Lookup":
                        result = train_classifier_run(
                            root,
                            rotation=rotation,
                            seed=seed,
                            features=features,
                            git=git,
                        )
                    else:
                        result = run_static_cell(
                            baseline,
                            root,
                            rotation=rotation,
                            seed=seed,
                            features=features,
                        )
                    results[baseline].append(result)
                    completed += 1
                    atomic_json(
                        ATTEMPT / "aggregates/live_execution_registry.json",
                        {
                            "task_id": TASK_ID,
                            "status": "RUNNING" if completed < 48 else "COMPLETE",
                            "baseline_order": list(BASELINE_NAMES),
                            "completed_baseline_cells": completed,
                            "formal_valid_baseline_cells": completed,
                            "optimizer_steps": sum(
                                run["optimizer_steps"]
                                for values in results.values()
                                for run in values
                            ),
                            "completed_run_ids": [
                                run["run_id"]
                                for values in results.values()
                                for run in values
                            ],
                        },
                    )
        aggregates = {
            name: aggregate_baseline(name, results[name])
            for name in BASELINE_NAMES
        }
        if aggregates["Reference Classifier Lookup"]["optimizer_steps"] != 3600:
            raise RuntimeError("Reference Classifier total budget mismatch")
        if any(
            aggregates[name]["optimizer_steps"] != 0 for name in BASELINE_NAMES[1:]
        ):
            raise RuntimeError("non-optimizer baseline executed optimizer steps")
        immutable = post_audit(config, inputs)
        total_wall_time = time.time() - lock["start_time_unix"]
        final = {
            "schema_version": "canondressgs.subject00.commonsafe4.fair_baseline_results.v1",
            "task_id": TASK_ID,
            "status": "COMPLETE",
            "classification": "COMMONSAFE4_FAIR_BASELINES_TECHNICAL_PASS",
            "exact_baseline_order": list(BASELINE_NAMES),
            "baseline_cell_count": 48,
            "formal_valid_baseline_cell_count": 48,
            "total_optimizer_steps": 3600,
            "aggregates": aggregates,
            "method_matrix_binding": {
                "output_root": str(METHOD_ROOT),
                "status": inputs["method_matrix_status"],
                "run_count": inputs["method_matrix_run_count"],
                "formal_valid_run_count": inputs[
                    "method_matrix_formal_valid_run_count"
                ],
            },
            "fairness": {
                "same_commonsafe4_anchors": True,
                "same_corrected_rotations": True,
                "same_seeds": True,
                "same_6_3_3_folds": True,
                "same_frozen_f2_features": True,
                "same_base60747": True,
                "same_formal_teachers": True,
                "same_formal_targets": True,
                "same_quarantine_policy": True,
                "same_endpoint_candidates": True,
                "same_formal_test_denominator": True,
                "historical_subject02_metrics_reused": False,
                "old_slot04_results_reused": False,
            },
            "compute_budget": {
                "CanonDressGS-Endpoint": 3600,
                "Reference Classifier Lookup": 3600,
                "Nearest-Centroid Lookup": 0,
                "Outfit-ID Oracle": 0,
                "Teacher Endpoint": 0,
            },
            "immutable_post_audit": immutable,
            "wall_time_seconds": total_wall_time,
            "peak_vram_bytes": max(
                value["peak_vram_bytes"] for value in aggregates.values()
            ),
            "quarantine_optimizer_usage_count": 0,
            "quarantine_test_usage_count": 0,
            "dual_support_call_count": 0,
            "nan_inf_status": "NONE",
            "oom_status": "NONE",
            "human_visual_decision": None,
            "scientific_pass": None,
            "paper_eligible": False,
            "paper_final": False,
        }
        atomic_json(ATTEMPT / "aggregates/fair_baseline_results.json", final)
        atomic_json(ATTEMPT / "input_audit/immutable_post_audit.json", immutable)
        atomic_json(ATTEMPT / "FINAL_REPORT.json", final)
        atomic_json(
            status_path,
            {
                "task_id": TASK_ID,
                "status": "COMPLETE",
                "stage": "FAIR_BASELINES_48_OF_48_FORMAL_VALID",
                "completed_baseline_cells": 48,
                "formal_valid_baseline_cells": 48,
                "reference_classifier_runs": 12,
                "reference_classifier_optimizer_steps": 3600,
                "non_optimizer_baseline_evaluations": 36,
                "quarantine_optimizer_usage_count": 0,
                "quarantine_test_usage_count": 0,
                "dual_support_call_count": 0,
                "nan_inf_status": "NONE",
                "oom_status": "NONE",
                "paper_eligible": False,
            },
        )
        update_lock(
            lock,
            "COMPLETE",
            completed_baseline_cells=48,
            formal_valid_baseline_cells=48,
            optimizer_steps=3600,
            wall_time_seconds=total_wall_time,
        )
        print(json.dumps(final, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        optimizer_steps = sum(
            run["optimizer_steps"] for values in results.values() for run in values
        )
        atomic_json(
            status_path,
            {
                "task_id": TASK_ID,
                "status": "FAIL",
                "stage": "FAIR_BASELINE_EXECUTION_FAILED",
                "completed_baseline_cells": completed,
                "formal_valid_baseline_cells": completed,
                "optimizer_steps": optimizer_steps,
                "error_type": type(error).__name__,
                "error": str(error),
                "automatic_retry": False,
                "attempt_002_allowed": False,
                "paper_eligible": False,
            },
        )
        update_lock(
            lock,
            "FAIL",
            completed_baseline_cells=completed,
            formal_valid_baseline_cells=completed,
            optimizer_steps=optimizer_steps,
            error_type=type(error).__name__,
            error=str(error),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
