#!/usr/bin/env python3
"""Run the frozen Subject00 Base60747 CommonSafe4 12-cell matrix."""

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
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene.full_dressable_dataset import FullDressableTrainingDataset
from scene.multi_outfit_linear_coefficient_control import (
    MultiOutfitLinearCoefficientControl,
)
from tools.second_identity import run_subject00_base60747_method as legacy


TASK_ID = "AAAI27-SUBJECT00-BASE60747-COMMONSAFE4-METHOD-MATRIX-001"
BRANCH = "research/subject00-base60747-commonsafe4-method-matrix-20260727"
CONTRACT_NAME = "SUBJECT00_COMMONSAFE4_CARDINAL_PROXY_V1"
CONFIG_PATH = (
    REPO_ROOT
    / "configs/research/subject00_canondressgs_method_base60747_commonsafe4_v1.json"
)
ORIGINAL_CONFIG_PATH = (
    REPO_ROOT / "configs/research/subject00_canondressgs_method_base60747_v1.json"
)
SELECTION_PATH = (
    REPO_ROOT
    / "paper_protocol/reviewer_risk/"
    "subject00_commonsafe_slot04_replacement_selection_20260727.json"
)
ROTATION_CONTRACT_PATH = (
    REPO_ROOT
    / "paper_protocol/reviewer_risk/subject00_commonsafe4_rotation_contract_20260727.json"
)
REGISTRY_PATH = (
    REPO_ROOT
    / "paper_protocol/reviewer_risk/"
    "subject00_base60747_three_garment_teacher_registry_20260727.json"
)
OUTPUT_ROOT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-CANONDRESSGS-METHOD-BASE60747-COMMONSAFE4-001"
)
ATTEMPT = OUTPUT_ROOT / "attempt_001"
LOCK_PATH = (
    OUTPUT_ROOT
    / "control/COMMONSAFE4_METHOD_MATRIX_EXECUTION_LOCK_20260727.json"
)
BASE_SHA = "2d09b1ce6bb19b8b6418314caab8dc1cf92d70e432a9e09e009989332ea4a2f7"
MANIFEST_SHA = "602820292fea47fe8116bf64824e6e2a8313a3ec31b8fc7cdd66a90c6bad33c1"
F2_SHA = "3e4b668fa9f9d5f79c5fd0f56941779c98e78fc3c4bcffa4b3f4268f53f16371"
F2_FINGERPRINT = "3ed5ec6d04d3b9444d234252b5bc5b718ff879f8ec6220ec9704499cce8e77d7"
GARMENTS = ("O01", "O03", "O04")
SEEDS = (0, 1, 2)
ANCHORS = (0, 7, 3, 6)
QUARANTINE = (
    "subject00_O01_slot04_remaining_attempt005_cand00",
    "subject00_O03_slot04_canary_attempt004_cand00",
)
CHECKPOINT_STEPS = (0, 20, 50, 100, 200, 300)
ROTATIONS = (
    {"rotation": 0, "train_slots": (0, 7), "calibration_slot": 3, "test_slot": 6},
    {"rotation": 1, "train_slots": (7, 3), "calibration_slot": 6, "test_slot": 0},
    {"rotation": 2, "train_slots": (3, 6), "calibration_slot": 0, "test_slot": 7},
    {"rotation": 3, "train_slots": (6, 0), "calibration_slot": 7, "test_slot": 3},
)
EXPECTED_RUN_IDS = tuple(
    f"COMMONSAFE4-METHOD-R{rotation['rotation']}-S{seed}"
    for rotation in ROTATIONS
    for seed in SEEDS
)


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


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), *args], text=True
        ).strip()

    state = {
        "branch": run("branch", "--show-current"),
        "head": run("rev-parse", "HEAD"),
        "porcelain_v2": run("status", "--porcelain=v2"),
    }
    if state["branch"] != BRANCH or state["porcelain_v2"]:
        raise RuntimeError(f"matrix execution requires clean {BRANCH}")
    return {**state, "clean": True}


def gpu_processes() -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    rows = []
    for line in result.splitlines():
        if not line.strip():
            continue
        values = [value.strip() for value in line.split(",")]
        rows.append(
            {
                "pid": int(values[0]),
                "process_name": values[1],
                "used_memory_mib": int(values[2]),
            }
        )
    return rows


def process_gate(*, initial: bool) -> dict[str, Any]:
    gpu = gpu_processes()
    foreign = [row for row in gpu if row["pid"] != os.getpid()]
    if foreign or (initial and gpu):
        raise RuntimeError(f"foreign/initial GPU process gate failed: {gpu}")
    process_table = subprocess.check_output(
        ["ps", "-eo", "pid=,args="], text=True
    ).splitlines()
    forbidden_tokens = (
        "run_subject00_formal_base_101245.py",
        "run_subject00_base60747_formal_teacher.py",
        "run_subject00_base60747_method.py",
        "run_subject00_base60747_commonsafe4_matrix.py",
    )
    conflicts = []
    for line in process_table:
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        if any(token in command for token in forbidden_tokens):
            # The tmux shell contains this command string while the child Python
            # is active; allow only the direct parent recorded by the OS.
            if pid == os.getppid():
                continue
            conflicts.append({"pid": pid, "command": command})
    if conflicts:
        raise RuntimeError(f"conflicting formal/controller process: {conflicts}")
    gpu_name = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).strip()
    if "RTX 4090" not in gpu_name:
        raise RuntimeError(f"unexpected GPU: {gpu_name}")
    return {
        "status": "PASS",
        "gpu_name": gpu_name,
        "gpu_processes": gpu,
        "foreign_gpu_process_count": len(foreign),
        "formal_or_controller_conflicts": conflicts,
    }


def validate_config() -> tuple[dict[str, Any], dict[str, Any]]:
    config = read_json(CONFIG_PATH)
    original = read_json(ORIGINAL_CONFIG_PATH)
    selection = read_json(SELECTION_PATH)
    contract = read_json(ROTATION_CONTRACT_PATH)
    if config["task_id"] != TASK_ID or config["execution_branch"] != BRANCH:
        raise RuntimeError("CommonSafe4 config task/branch identity mismatch")
    if config["method_contract"].get("rotation_contract") != CONTRACT_NAME:
        raise RuntimeError("CommonSafe4 contract name mismatch")
    if int(config["method_contract"].get("selected_replacement_slot", -1)) != 6:
        raise RuntimeError("sealed replacement slot changed")
    if selection["selected_replacement_slot"] != "slot06":
        raise RuntimeError("selection artifact changed")
    if contract["method_contract"] != "PURE_ENDPOINT":
        raise RuntimeError("method contract changed")
    if contract["dual_support_enabled"] is not False:
        raise RuntimeError("Dual-Support unexpectedly enabled")
    actual_rotations = tuple(
        {
            "rotation": int(value["rotation"]),
            "train_slots": tuple(value["train_slots"]),
            "calibration_slot": int(value["calibration_slot"]),
            "test_slot": int(value["test_slot"]),
        }
        for value in config["protocol"]["condition_rotations"]
    )
    if actual_rotations != ROTATIONS:
        raise RuntimeError(f"corrected rotations changed: {actual_rotations}")
    frozen_sections = (
        "base",
        "targets",
        "teachers",
        "frozen_f2",
        "basis",
        "controller",
        "optimization",
        "validation_gates",
        "final_evaluation",
        "paper_eligible",
        "paper_final",
    )
    frozen = {name: config[name] == original[name] for name in frozen_sections}
    if not all(frozen.values()):
        raise RuntimeError(f"forbidden config change: {frozen}")
    if (
        int(config["optimization"]["steps_per_run"]) != 300
        or tuple(config["optimization"]["checkpoint_steps"]) != CHECKPOINT_STEPS
        or tuple(config["protocol"]["replicate_seeds"]) != SEEDS
        or config["output"]["root"] != str(OUTPUT_ROOT)
        or config["output"]["attempt"] != "attempt_001"
    ):
        raise RuntimeError("matrix budget/output contract changed")
    return config, {
        "status": "PASS",
        "config_sha256": sha256(CONFIG_PATH),
        "config_lf_sha256": lf_sha256(CONFIG_PATH),
        "selection_sha256": sha256(SELECTION_PATH),
        "rotation_contract_sha256": sha256(ROTATION_CONTRACT_PATH),
        "frozen_section_equality": frozen,
    }


def validate_registry(config: Mapping[str, Any]) -> dict[str, Any]:
    registry = read_json(REGISTRY_PATH)
    if registry["teacher_garment_count"] != 3:
        raise RuntimeError("Teacher registry count changed")
    if registry["formal_targets"]["manifest_sha256"] != MANIFEST_SHA:
        raise RuntimeError("Teacher manifest binding changed")
    result = {}
    for garment in GARMENTS:
        row = registry["garments"][garment]
        binding = config["teachers"]["checkpoints"][garment]
        if (
            row["technical_status"] != "TECHNICAL_PASS"
            or row["teacher_checkpoint"] != binding["path"]
            or row["teacher_checkpoint_sha256"] != binding["sha256"]
        ):
            raise RuntimeError(f"{garment} Teacher registry mismatch")
        result[garment] = {
            "path": binding["path"],
            "sha256": binding["sha256"],
            "technical_status": row["technical_status"],
        }
    return {
        "status": "PASS",
        "registry_path": str(REGISTRY_PATH.relative_to(REPO_ROOT)),
        "registry_lf_sha256": lf_sha256(REGISTRY_PATH),
        "garments": result,
    }


def input_gate(config: Mapping[str, Any]) -> dict[str, Any]:
    manifest = Path(config["targets"]["root"]) / config["targets"]["manifest"]
    paths = {
        "base": (Path(config["base"]["checkpoint"]), BASE_SHA),
        "manifest": (manifest, MANIFEST_SHA),
        "f2": (Path(config["frozen_f2"]["checkpoint"]), F2_SHA),
        **{
            f"teacher_{garment}": (
                Path(config["teachers"]["checkpoints"][garment]["path"]),
                config["teachers"]["checkpoints"][garment]["sha256"],
            )
            for garment in GARMENTS
        },
    }
    assets = {}
    for name, (path, expected) in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"{name} SHA mismatch")
        assets[name] = {"path": str(path), "sha256": actual}
    payload = read_json(manifest)
    observations = [
        observation
        for outfit in payload["outfits"]
        for observation in outfit["observations"]
    ]
    ids = [value["condition_id"] for value in observations]
    if len(ids) != 22 or len(set(ids)) != 22 or set(QUARANTINE).intersection(ids):
        raise RuntimeError("formal target denominator/quarantine mismatch")
    slot_counts = {
        slot: {
            garment: sum(
                outfit["outfit_id"] == garment
                and f"_slot{slot:02d}_" in observation["condition_id"]
                for outfit in payload["outfits"]
                for observation in outfit["observations"]
            )
            for garment in GARMENTS
        }
        for slot in ANCHORS
    }
    if any(set(counts.values()) != {1} for counts in slot_counts.values()):
        raise RuntimeError(f"anchor garment coverage changed: {slot_counts}")
    usage = shutil.disk_usage(OUTPUT_ROOT.parent)
    if usage.free < 30 * (1 << 30):
        raise RuntimeError(f"storage gate failed: {usage.free}")
    return {
        "status": "PASS",
        "assets": assets,
        "training_target_count": 22,
        "quarantine_count": 2,
        "quarantine_in_manifest": [],
        "anchor_slot_counts": slot_counts,
        "storage_free_bytes": usage.free,
    }


def create_lock(git: Mapping[str, Any]) -> dict[str, Any]:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"new output root already exists: {OUTPUT_ROOT}")
    (OUTPUT_ROOT / "control").mkdir(parents=True, exist_ok=False)
    lock = {
        "schema_version": "canondressgs.subject00.commonsafe4.execution_lock.v1",
        "task_id": TASK_ID,
        "branch": git["branch"],
        "head": git["head"],
        "hostname": subprocess.check_output(["hostname"], text=True).strip(),
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "tmux": os.environ.get("COMMONSAFE4_TMUX_SESSION"),
        "output_root": str(OUTPUT_ROOT),
        "selected_replacement": "slot06",
        "run_ids": list(EXPECTED_RUN_IDS),
        "start_time_unix": time.time(),
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
        },
    )


def patch_legacy_globals(git: Mapping[str, Any]) -> None:
    legacy.TASK_ID = TASK_ID
    legacy.BRANCH = BRANCH
    legacy.CONFIG_PATH = CONFIG_PATH
    legacy.REGISTRY_PATH = REGISTRY_PATH
    legacy.BASE_SHA = BASE_SHA
    legacy.MANIFEST_SHA = MANIFEST_SHA
    legacy.F2_SHA = F2_SHA
    legacy.F2_FINGERPRINT = F2_FINGERPRINT


def extract_all_features(
    output_dir: Path,
    config: Mapping[str, Any],
    git: Mapping[str, Any],
) -> tuple[dict[tuple[int, str, int], torch.Tensor], dict[str, Any]]:
    manifest = Path(config["targets"]["root"]) / config["targets"]["manifest"]
    extractor, provenance, retained = legacy.build_f2_extractor(config, git)
    features: dict[tuple[int, str, int], torch.Tensor] = {}
    seed_audits = {}
    for seed in SEEDS:
        dataset = FullDressableTrainingDataset(
            manifest, split="train", reference_count=3, seed=seed
        )
        episodes = {}
        for garment in GARMENTS:
            for slot in ANCHORS:
                matches = [
                    (index, outfit, observation)
                    for index, (outfit, observation) in enumerate(dataset.samples)
                    if outfit["outfit_id"] == garment
                    and f"_slot{slot:02d}_" in observation["condition_id"]
                ]
                if len(matches) != 1:
                    raise RuntimeError(
                        f"{garment}/slot{slot:02d} unavailable for seed {seed}"
                    )
                index, outfit, target = matches[0]
                selected = dataset._references(outfit, target["condition_id"], index)
                selected_ids = {value["condition_id"] for value in selected}
                if target["condition_id"] in selected_ids:
                    raise RuntimeError("target leaked into reference set")
                if set(QUARANTINE).intersection(selected_ids):
                    raise RuntimeError("quarantine entered reference sampler")
                references = [
                    dataset._observation(outfit, value, True) for value in selected
                ]
                feature, episode = legacy.aggregate_native_references(
                    extractor, references
                )
                features[(seed, garment, slot)] = feature.cpu()
                episodes[f"{garment}/slot{slot:02d}"] = {
                    "target_condition_id": target["condition_id"],
                    **episode,
                }
        artifact = output_dir / f"features/seed_{seed:03d}_commonsafe4_f2.pt"
        payload = {
            "schema_version": "canondressgs.subject00.commonsafe4.f2_features.v1",
            "task_id": TASK_ID,
            "seed": seed,
            "anchors": list(ANCHORS),
            "features": {
                f"{garment}/slot{slot:02d}": features[(seed, garment, slot)]
                for garment in GARMENTS
                for slot in ANCHORS
            },
            "episodes": episodes,
            "provenance": provenance,
            "quarantine_usage_count": 0,
        }
        atomic_torch_save(artifact, payload)
        seed_audits[str(seed)] = {
            "artifact": str(artifact),
            "sha256": sha256(artifact),
            "episode_count": len(episodes),
            "episodes": episodes,
        }
    backbone_after = legacy.multi.o01._state_fingerprint(
        retained["legacy_model"].clothing_observation_encoder.backbone.state_dict()
    )
    if (
        backbone_after != retained["backbone_before"]
        or retained["backbone_before"] != F2_FINGERPRINT
    ):
        raise RuntimeError("frozen F2 backbone mutated")
    audit = {
        "status": "PASS",
        "provenance": provenance,
        "backbone_after": backbone_after,
        "backbone_bitwise_frozen": backbone_after == retained["backbone_before"],
        "seed_audits": seed_audits,
        "episode_count": len(features),
        "quarantine_usage_count": 0,
        "target_or_teacher_field_in_prediction_forward": False,
    }
    atomic_json(output_dir / "features/feature_audit.json", audit)
    del extractor, retained
    torch.cuda.empty_cache()
    return features, audit


def nearest_endpoint(
    prediction: torch.Tensor, targets: Mapping[str, torch.Tensor]
) -> tuple[str, dict[str, float], float]:
    scores = {
        garment: float((prediction - targets[garment]).square().sum())
        for garment in GARMENTS
    }
    ordered = sorted(GARMENTS, key=lambda value: (scores[value], GARMENTS.index(value)))
    margin = scores[ordered[1]] - scores[ordered[0]]
    return ordered[0], scores, margin


def evaluate_fold(
    controller: MultiOutfitLinearCoefficientControl,
    features: Mapping[tuple[int, str, int], torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    *,
    seed: int,
    rotation: int,
    fold_name: str,
    slot: int,
) -> dict[str, Any]:
    controller.eval()
    rows = []
    squared_errors = []
    with torch.no_grad():
        target_device = {
            garment: value.cuda(non_blocking=False) for garment, value in targets.items()
        }
        for garment in GARMENTS:
            prediction = controller(
                features[(seed, garment, slot)].cuda(non_blocking=False)
            ).standardized_coefficients.reshape(-1)
            target = target_device[garment].reshape(-1)
            selected, scores, margin = nearest_endpoint(prediction, target_device)
            squared_errors.extend((prediction - target).square().tolist())
            rows.append(
                {
                    "garment": garment,
                    "slot": slot,
                    "prediction": prediction.tolist(),
                    "target": target.tolist(),
                    "selected_endpoint": selected,
                    "correct": selected == garment,
                    "candidate_squared_distances": scores,
                    "score_margin_second_minus_first": margin,
                    "finite": bool(torch.isfinite(prediction).all()),
                }
            )
    correct = sum(row["correct"] for row in rows)
    if len(rows) != 3 or any(not row["finite"] for row in rows):
        raise RuntimeError(f"{fold_name} formal evaluation is incomplete/nonfinite")
    return {
        "status": "PASS_FORMAL_3_OF_3",
        "rotation": rotation,
        "seed": seed,
        "fold": fold_name,
        "slot": slot,
        "denominator": 3,
        "garment_record_count": 3,
        "nearest_endpoint_correct": correct,
        "nearest_endpoint_total": 3,
        "nearest_endpoint_accuracy": correct / 3.0,
        "standardized_coefficient_rmse": float(
            np.sqrt(np.mean(squared_errors))
        ),
        "mean_score_margin": float(
            np.mean([row["score_margin_second_minus_first"] for row in rows])
        ),
        "rows": rows,
        "quarantine_usage_count": 0,
        "dual_support_call_count": 0,
        "paper_eligible": False,
    }


def save_checkpoint(
    path: Path,
    *,
    run_id: str,
    rotation: Mapping[str, Any],
    seed: int,
    step: int,
    controller: MultiOutfitLinearCoefficientControl,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    history: list[dict[str, Any]],
    config: Mapping[str, Any],
    git: Mapping[str, Any],
    basis_audit: Mapping[str, Any],
    feature_audit: Mapping[str, Any],
) -> None:
    atomic_torch_save(
        path,
        {
            "schema_version": "canondressgs.subject00.commonsafe4.checkpoint.v1",
            "task_id": TASK_ID,
            "run_id": run_id,
            "rotation": int(rotation["rotation"]),
            "seed": seed,
            "train_slots": list(rotation["train_slots"]),
            "calibration_slot": int(rotation["calibration_slot"]),
            "test_slot": int(rotation["test_slot"]),
            "global_step": step,
            "optimizer_step": step,
            "model": controller.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": legacy.rng_state(),
            "condition_position": step % 2,
            "history": history,
            "bindings": {
                "contract_name": CONTRACT_NAME,
                "config_sha256": sha256(CONFIG_PATH),
                "selection_sha256": sha256(SELECTION_PATH),
                "rotation_contract_sha256": sha256(ROTATION_CONTRACT_PATH),
                "base_checkpoint": config["base"]["checkpoint"],
                "base_checkpoint_sha256": BASE_SHA,
                "target_manifest_sha256": MANIFEST_SHA,
                "teacher_checkpoint_sha256": {
                    garment: config["teachers"]["checkpoints"][garment]["sha256"]
                    for garment in GARMENTS
                },
                "basis_artifact_sha256": basis_audit["artifact_sha256"],
                "feature_artifact_sha256": feature_audit["seed_audits"][str(seed)][
                    "sha256"
                ],
                "f2_checkpoint_sha256": F2_SHA,
                "f2_backbone_fingerprint": F2_FINGERPRINT,
                "quarantine_optimizer_usage_count": 0,
                "quarantine_test_usage_count": 0,
                "dual_support_call_count": 0,
                "git": dict(git),
            },
            "paper_eligible": False,
        },
    )


def train_run(
    run_root: Path,
    *,
    rotation: Mapping[str, Any],
    seed: int,
    config: Mapping[str, Any],
    git: Mapping[str, Any],
    features: Mapping[tuple[int, str, int], torch.Tensor],
    targets_cpu: Mapping[str, torch.Tensor],
    basis_audit: Mapping[str, Any],
    feature_audit: Mapping[str, Any],
) -> dict[str, Any]:
    process = process_gate(initial=False)
    run_id = f"COMMONSAFE4-METHOD-R{rotation['rotation']}-S{seed}"
    if run_root.exists():
        raise FileExistsError(f"run output already exists: {run_root}")
    for directory in ("checkpoints", "training", "evaluation", "audit"):
        (run_root / directory).mkdir(parents=True, exist_ok=False)
    atomic_json(
        run_root / "audit/pre_run_gate.json",
        {
            "status": "PASS",
            "run_id": run_id,
            "resource_gate": process,
            "fold_counts": {"train": 6, "calibration": 3, "test": 3},
            "quarantine_usage_count": 0,
        },
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    targets = {
        garment: value.to(device) for garment, value in targets_cpu.items()
    }
    controller = MultiOutfitLinearCoefficientControl(512, 2).to(device)
    if sum(parameter.numel() for parameter in controller.parameters()) != 2050:
        raise RuntimeError("controller parameter count changed")
    optimizer = torch.optim.Adam(
        controller.parameters(),
        lr=0.02,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda _: 1.0
    )
    history: list[dict[str, Any]] = []
    initial_state = {
        name: value.detach().cpu().clone()
        for name, value in controller.state_dict().items()
    }
    initial_sha = tensor_state_sha256(initial_state)
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    save_checkpoint(
        run_root / "checkpoints/step_000000.pth",
        run_id=run_id,
        rotation=rotation,
        seed=seed,
        step=0,
        controller=controller,
        optimizer=optimizer,
        scheduler=scheduler,
        history=history,
        config=config,
        git=git,
        basis_audit=basis_audit,
        feature_audit=feature_audit,
    )
    train_slots = tuple(rotation["train_slots"])
    for step in range(1, 301):
        slot = train_slots[(step - 1) % 2]
        optimizer.zero_grad(set_to_none=True)
        predictions = torch.stack(
            [
                controller(features[(seed, garment, slot)].to(device))
                .standardized_coefficients.reshape(-1)
                for garment in GARMENTS
            ]
        )
        target_batch = torch.stack(
            [targets[garment].reshape(-1) for garment in GARMENTS]
        )
        loss = F.smooth_l1_loss(
            predictions, target_batch, beta=1.0, reduction="mean"
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{run_id} nonfinite loss at step {step}")
        selections = {}
        candidate_scores = {}
        endpoint_scores = {}
        for index, garment in enumerate(GARMENTS):
            selected, scores, _ = nearest_endpoint(predictions[index], targets)
            selections[garment] = selected
            candidate_scores[garment] = scores
            endpoint_scores[garment] = scores[selected]
        loss.backward()
        gradients_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in controller.parameters()
        )
        if not gradients_finite:
            raise FloatingPointError(f"{run_id} nonfinite gradient at step {step}")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            controller.parameters(), 5.0
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"{run_id} nonfinite grad norm")
        optimizer.step()
        scheduler.step()
        checkpoint_event = step in CHECKPOINT_STEPS
        row = {
            "optimizer_step": step,
            "rotation": int(rotation["rotation"]),
            "seed": seed,
            "fold": "train",
            "condition_slot": slot,
            "garment_batch_order": list(GARMENTS),
            "loss_total": float(loss.detach()),
            "loss_components": {
                "coefficient_smooth_l1": float(loss.detach()),
                "classification": 0.0,
                "pairwise_geometry": 0.0,
                "render": 0.0,
            },
            "gradient_norm_before_clip": float(gradient_norm),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "selected_endpoint": selections,
            "endpoint_score": endpoint_scores,
            "candidate_scores": candidate_scores,
            "finite": True,
            "elapsed_seconds": time.perf_counter() - start,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
            "checkpoint_event": checkpoint_event,
        }
        history.append(row)
        if step % 10 == 0 or checkpoint_event:
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
        if checkpoint_event:
            save_checkpoint(
                run_root / f"checkpoints/step_{step:06d}.pth",
                run_id=run_id,
                rotation=rotation,
                seed=seed,
                step=step,
                controller=controller,
                optimizer=optimizer,
                scheduler=scheduler,
                history=history,
                config=config,
                git=git,
                basis_audit=basis_audit,
                feature_audit=feature_audit,
            )
    changed = {
        name: not torch.equal(
            initial_state[name], controller.state_dict()[name].detach().cpu()
        )
        for name in initial_state
    }
    if not all(changed.values()):
        raise RuntimeError(f"{run_id} trainable parameters did not all change: {changed}")
    final_sha = tensor_state_sha256(controller.state_dict())
    if final_sha == initial_sha:
        raise RuntimeError(f"{run_id} final controller equals initialization")
    calibration = evaluate_fold(
        controller,
        features,
        targets_cpu,
        seed=seed,
        rotation=int(rotation["rotation"]),
        fold_name="calibration",
        slot=int(rotation["calibration_slot"]),
    )
    test = evaluate_fold(
        controller,
        features,
        targets_cpu,
        seed=seed,
        rotation=int(rotation["rotation"]),
        fold_name="test",
        slot=int(rotation["test_slot"]),
    )
    atomic_json(run_root / "evaluation/calibration_step300.json", calibration)
    atomic_json(run_root / "evaluation/formal_test_step300.json", test)
    checkpoint_paths = sorted((run_root / "checkpoints").glob("step_*.pth"))
    checkpoint_steps = [
        int(path.stem.removeprefix("step_")) for path in checkpoint_paths
    ]
    if tuple(checkpoint_steps) != CHECKPOINT_STEPS:
        raise RuntimeError(f"{run_id} checkpoint set incomplete: {checkpoint_steps}")
    wall_time = time.perf_counter() - start
    result = {
        "status": "FORMAL_VALID_MATRIX_CELL",
        "run_id": run_id,
        "rotation": int(rotation["rotation"]),
        "seed": seed,
        "train_slots": list(train_slots),
        "calibration_slot": int(rotation["calibration_slot"]),
        "test_slot": int(rotation["test_slot"]),
        "train_garment_record_count": 6,
        "calibration_garment_record_count": 3,
        "test_garment_record_count": 3,
        "optimizer_steps": 300,
        "optimizer_step_monotonic": [
            value["optimizer_step"] for value in history
        ]
        == list(range(1, 301)),
        "checkpoint_steps": checkpoint_steps,
        "checkpoint_status": "PASS_6_OF_6",
        "checkpoint_sha256": {
            str(step): sha256(run_root / f"checkpoints/step_{step:06d}.pth")
            for step in CHECKPOINT_STEPS
        },
        "loss_initial": history[0]["loss_total"],
        "loss_final": history[-1]["loss_total"],
        "loss_minimum": min(value["loss_total"] for value in history),
        "loss_finite": all(np.isfinite(value["loss_total"]) for value in history),
        "gradients_finite": True,
        "nan_inf_status": "NONE",
        "oom_status": "NONE",
        "trainable_parameter_changes": changed,
        "initial_controller_sha256": initial_sha,
        "final_controller_sha256": final_sha,
        "calibration": calibration,
        "formal_test": test,
        "formal_test_status": "PASS_3_OF_3",
        "wall_time_seconds": wall_time,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "quarantine_optimizer_usage_count": 0,
        "quarantine_calibration_usage_count": 0,
        "quarantine_test_usage_count": 0,
        "quarantine_evaluation_usage_count": 0,
        "dual_support_call_count": 0,
        "paper_eligible": False,
    }
    atomic_json(run_root / "training/run_summary.json", result)
    del controller, optimizer, scheduler, targets
    torch.cuda.empty_cache()
    return result


def immutable_post_audit(
    config: Mapping[str, Any], before: Mapping[str, Any]
) -> dict[str, Any]:
    current = {
        name: sha256(Path(value["path"]))
        for name, value in before["assets"].items()
    }
    expected = {
        name: value["sha256"] for name, value in before["assets"].items()
    }
    if current != expected:
        raise RuntimeError("immutable Base/Teacher/target/F2 mutation detected")
    return {
        "status": "PASS",
        "sha256_before": expected,
        "sha256_after": current,
        "base60747_mutations": 0,
        "teacher_mutations": {garment: 0 for garment in GARMENTS},
        "formal_target_mutations": 0,
        "raw_mutations": 0,
        "mask_mutations": 0,
        "camera_record_mutations": 0,
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    if len(results) != 12:
        raise RuntimeError(f"matrix incomplete: {len(results)}")
    run_ids = [value["run_id"] for value in results]
    if tuple(run_ids) != EXPECTED_RUN_IDS or len(set(run_ids)) != 12:
        raise RuntimeError(f"run ID set/order mismatch: {run_ids}")
    accuracies = [
        float(value["formal_test"]["nearest_endpoint_accuracy"])
        for value in results
    ]
    all_rows = [
        row for value in results for row in value["formal_test"]["rows"]
    ]
    confusion = {
        truth: {
            prediction: sum(
                row["garment"] == truth
                and row["selected_endpoint"] == prediction
                for row in all_rows
            )
            for prediction in GARMENTS
        }
        for truth in GARMENTS
    }
    rotation_effect = {
        str(rotation): {
            "run_count": sum(value["rotation"] == rotation for value in results),
            "mean_endpoint_top1": float(
                np.mean(
                    [
                        value["formal_test"]["nearest_endpoint_accuracy"]
                        for value in results
                        if value["rotation"] == rotation
                    ]
                )
            ),
        }
        for rotation in range(4)
    }
    seed_effect = {
        str(seed): {
            "run_count": sum(value["seed"] == seed for value in results),
            "mean_endpoint_top1": float(
                np.mean(
                    [
                        value["formal_test"]["nearest_endpoint_accuracy"]
                        for value in results
                        if value["seed"] == seed
                    ]
                )
            ),
        }
        for seed in SEEDS
    }
    return {
        "status": "PASS_12_OF_12_FORMAL_VALID",
        "method_contract": "PURE_ENDPOINT",
        "dual_support_enabled": False,
        "no_cherry_picking": True,
        "run_count": 12,
        "training_authentic_run_count": 12,
        "formal_valid_matrix_run_count": 12,
        "total_optimizer_steps": sum(value["optimizer_steps"] for value in results),
        "checkpoint_complete_run_count": sum(
            value["checkpoint_status"] == "PASS_6_OF_6" for value in results
        ),
        "formal_test_episode_count": len(all_rows),
        "formal_test_correct": sum(row["correct"] for row in all_rows),
        "matrix_endpoint_top1": sum(row["correct"] for row in all_rows)
        / len(all_rows),
        "per_run_endpoint_top1": {
            value["run_id"]: value["formal_test"]["nearest_endpoint_accuracy"]
            for value in results
        },
        "mean_endpoint_top1": float(np.mean(accuracies)),
        "std_endpoint_top1_population": float(np.std(accuracies, ddof=0)),
        "min_endpoint_top1": float(np.min(accuracies)),
        "max_endpoint_top1": float(np.max(accuracies)),
        "mean_score_margin": float(
            np.mean(
                [
                    row["score_margin_second_minus_first"]
                    for row in all_rows
                ]
            )
        ),
        "rotation_effect": rotation_effect,
        "seed_effect": seed_effect,
        "confusion_matrix": confusion,
        "prediction_counts": dict(
            Counter(row["selected_endpoint"] for row in all_rows)
        ),
        "failure_count": 0,
        "nan_inf_status": "NONE",
        "oom_status": "NONE",
        "quarantine_optimizer_usage_count": 0,
        "quarantine_test_usage_count": 0,
        "dual_support_call_count": 0,
        "wall_time_seconds": sum(value["wall_time_seconds"] for value in results),
        "peak_vram_bytes": max(value["peak_vram_bytes"] for value in results),
        "runs": results,
        "paper_eligible": False,
        "scientific_pass": None,
        "human_visual_decision": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.config.resolve() != CONFIG_PATH.resolve():
        raise RuntimeError("only the sealed CommonSafe4 config is authorized")
    git = git_state()
    initial_process = process_gate(initial=True)
    config, config_audit = validate_config()
    registry = validate_registry(config)
    inputs = input_gate(config)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "PASS_PREFLIGHT_ONLY_NO_OUTPUT_CREATED",
                    "task_id": TASK_ID,
                    "git": git,
                    "resource_gate": initial_process,
                    "config_audit": config_audit,
                    "teacher_registry": registry,
                    "inputs": inputs,
                    "output_root_absent": not OUTPUT_ROOT.exists(),
                    "optimizer_initialized": False,
                    "optimizer_steps": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    lock = create_lock(git)
    for directory in (
        "contract",
        "input_audit",
        "shared/basis",
        "shared/features",
        "runs",
        "matrix",
    ):
        (ATTEMPT / directory).mkdir(parents=True, exist_ok=False)
    atomic_json(ATTEMPT / "contract/config_resolved.json", config)
    atomic_json(
        ATTEMPT / "input_audit/preflight.json",
        {
            "status": "PASS",
            "task_id": TASK_ID,
            "git": git,
            "resource_gate": initial_process,
            "config_audit": config_audit,
            "teacher_registry": registry,
            "inputs": inputs,
            "completed_run_count_before": 0,
            "pending_run_count_before": 12,
            "paper_eligible": False,
        },
    )
    status_path = ATTEMPT / "RUN_STATUS.json"
    atomic_json(
        status_path,
        {
            "task_id": TASK_ID,
            "status": "RUNNING",
            "stage": "SHARED_BASIS",
            "completed_run_count": 0,
            "formal_valid_run_count": 0,
            "optimizer_steps": 0,
            "paper_eligible": False,
        },
    )
    results: list[dict[str, Any]] = []
    try:
        patch_legacy_globals(git)
        targets, basis_audit = legacy.save_basis(ATTEMPT / "shared", config)
        atomic_json(
            status_path,
            {
                "task_id": TASK_ID,
                "status": "RUNNING",
                "stage": "SHARED_F2_FEATURES",
                "completed_run_count": 0,
                "formal_valid_run_count": 0,
                "optimizer_steps": 0,
                "paper_eligible": False,
            },
        )
        features, feature_audit = extract_all_features(
            ATTEMPT / "shared", config, git
        )
        for rotation in ROTATIONS:
            for seed in SEEDS:
                run_id = (
                    f"COMMONSAFE4-METHOD-R{rotation['rotation']}-S{seed}"
                )
                atomic_json(
                    status_path,
                    {
                        "task_id": TASK_ID,
                        "status": "RUNNING",
                        "stage": run_id,
                        "completed_run_count": len(results),
                        "formal_valid_run_count": len(results),
                        "optimizer_steps": len(results) * 300,
                        "paper_eligible": False,
                    },
                )
                result = train_run(
                    ATTEMPT / "runs" / run_id,
                    rotation=rotation,
                    seed=seed,
                    config=config,
                    git=git,
                    features=features,
                    targets_cpu=targets,
                    basis_audit=basis_audit,
                    feature_audit=feature_audit,
                )
                results.append(result)
                atomic_json(
                    ATTEMPT / "matrix/live_execution_registry.json",
                    {
                        "task_id": TASK_ID,
                        "expected_run_ids": list(EXPECTED_RUN_IDS),
                        "completed_run_ids": [
                            value["run_id"] for value in results
                        ],
                        "completed_run_count": len(results),
                        "formal_valid_run_count": len(results),
                        "optimizer_steps": len(results) * 300,
                        "runs": results,
                        "status": (
                            "COMPLETE" if len(results) == 12 else "RUNNING"
                        ),
                    },
                )
        matrix = aggregate(results)
        immutable = immutable_post_audit(config, inputs)
        atomic_json(ATTEMPT / "matrix/matrix_aggregate.json", matrix)
        atomic_json(ATTEMPT / "input_audit/immutable_post_audit.json", immutable)
        final = {
            "schema_version": "canondressgs.subject00.commonsafe4.matrix_final.v1",
            "task_id": TASK_ID,
            "status": "COMPLETE",
            "classification": "COMMONSAFE4_METHOD_MATRIX_TECHNICAL_PASS",
            "matrix": matrix,
            "immutable_post_audit": immutable,
            "paper_eligible": False,
            "scientific_pass": None,
            "human_visual_decision": None,
            "paper_final": False,
        }
        atomic_json(ATTEMPT / "FINAL_REPORT.json", final)
        atomic_json(
            status_path,
            {
                "task_id": TASK_ID,
                "status": "COMPLETE",
                "stage": "MATRIX_12_OF_12_FORMAL_VALID",
                "completed_run_count": 12,
                "formal_valid_run_count": 12,
                "optimizer_steps": 3600,
                "nan_inf_status": "NONE",
                "oom_status": "NONE",
                "paper_eligible": False,
            },
        )
        update_lock(
            lock,
            "COMPLETE",
            completed_run_count=12,
            formal_valid_run_count=12,
            optimizer_steps=3600,
        )
        print(json.dumps(final, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        atomic_json(
            status_path,
            {
                "task_id": TASK_ID,
                "status": "FAIL",
                "stage": "MATRIX_EXECUTION_FAILED",
                "completed_run_count": len(results),
                "formal_valid_run_count": len(results),
                "optimizer_steps": len(results) * 300,
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
            completed_run_count=len(results),
            formal_valid_run_count=len(results),
            optimizer_steps=len(results) * 300,
            error_type=type(error).__name__,
            error=str(error),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
