#!/usr/bin/env python3
"""Run the frozen Subject00 Base60747 Pure-Endpoint CommonSafe4 matrix.

Dry-run and zero-step-smoke are preflight-only and never create the formal
output root.  Training modes are explicit, collision-intolerant, and support
only a fresh attempt_001 with resume-policy=none.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TASK_ID = "AAAI27-SUBJECT00-BASE60747-COMMONSAFE4-CONTRACT-RUNNER-PREFLIGHT-001"
CONFIG_PATH = (
    REPO_ROOT
    / "configs/research/subject00_canondressgs_method_base60747_commonsafe4_v1.json"
)
OLD_CONFIG_PATH = (
    REPO_ROOT / "configs/research/subject00_canondressgs_method_base60747_v1.json"
)
OLD_RUNNER_PATH = (
    REPO_ROOT / "tools/second_identity/run_subject00_base60747_method.py"
)
FROZEN_RUNTIME_PATH = REPO_ROOT / "tools/paper/formal_batch_runtime.py"
FROZEN_RUNTIME_COMMIT = "ed44835f950fe369dd3eb64b5eaef8c6cc174175"
FROZEN_RUNTIME_LF_SHA = (
    "75211429a628801882d211805a2334fc88c11f6654cfbc4b6cc47fa2906693a6"
)
OLD_RUNNER_LF_SHA = (
    "c1703b209441de6521f5c5120e170090c52fa5a0e9bd78a20c62b35a45306bd9"
)
BASE_SHA = "2d09b1ce6bb19b8b6418314caab8dc1cf92d70e432a9e09e009989332ea4a2f7"
MANIFEST_SHA = "602820292fea47fe8116bf64824e6e2a8313a3ec31b8fc7cdd66a90c6bad33c1"
F2_SHA = "3e4b668fa9f9d5f79c5fd0f56941779c98e78fc3c4bcffa4b3f4268f53f16371"
F2_FINGERPRINT = "3ed5ec6d04d3b9444d234252b5bc5b718ff879f8ec6220ec9704499cce8e77d7"
TEACHER_SHA = {
    "O01": "c7881862c4eddf5f58538a2278ab7765aa047784681fb02e7cd89cfe846c1892",
    "O03": "054b9efe1086d89b18310b9831d3ed7aea59e286e31200509b58756545fc3920",
    "O04": "2fa7764097d8577c1610bbf222b26d9ea287bd18074371de400a66cd2270f3a1",
}
GARMENTS = ("O01", "O03", "O04")
SEEDS = (0, 1, 2)
CHECKPOINT_STEPS = (0, 20, 50, 100, 200, 300)
QUARANTINE = (
    "subject00_O01_slot04_remaining_attempt005_cand00",
    "subject00_O03_slot04_canary_attempt004_cand00",
)
FORMAL_OUTPUT_ROOT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-CANONDRESSGS-METHOD-BASE60747-COMMONSAFE4-001"
)
ZERO_STEP_SMOKE_CELLS = ((0, 0), (1, 1), (2, 2), (3, 0))


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


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), *args], text=True
        ).strip()

    return {
        "branch": run("branch", "--show-current"),
        "head": run("rev-parse", "HEAD"),
        "clean": not bool(run("status", "--porcelain")),
    }


def _rotations(config: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    rows = config["protocol"]["condition_rotations"]
    rotations = {int(row["rotation"]): row for row in rows}
    if set(rotations) != {0, 1, 2, 3} or len(rows) != 4:
        raise RuntimeError("rotation set must be exactly R0-R3")
    anchors = {0, 7, 3, 6}
    expected = {
        0: ([0, 7], 3, 6),
        1: ([7, 3], 6, 0),
        2: ([3, 6], 0, 7),
        3: ([6, 0], 7, 3),
    }
    for rotation, row in rotations.items():
        train = [int(value) for value in row["train_slots"]]
        calibration = int(row["calibration_slot"])
        test = int(row["test_slot"])
        if (train, calibration, test) != expected[rotation]:
            raise RuntimeError(f"rotation R{rotation} differs from frozen CommonSafe4")
        if len(train) != 2 or set(train + [calibration, test]) != anchors:
            raise RuntimeError(f"rotation R{rotation} overlaps or changes denominator")
    return rotations


def validate_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("task_id") != TASK_ID:
        raise RuntimeError("config task identity mismatch")
    if config.get("method_contract", {}).get("rotation_contract") != (
        "SUBJECT00_COMMONSAFE4_CARDINAL_PROXY_V1"
    ):
        raise RuntimeError("CommonSafe4 protocol name changed")
    method = config["method_contract"]
    if (
        method.get("dual_support_policy")
        != "DISABLED_EXCLUDED_FROM_PURE_ENDPOINT_PROTOCOL"
        or method.get("selected_replacement_slot") != 6
        or method.get("selected_replacement_camera") != "cam09"
        or method.get("selected_replacement_direction") != "back-right"
    ):
        raise RuntimeError("Pure-Endpoint/replacement contract changed")
    if config["base"]["sha256"] != BASE_SHA or int(config["base"]["step"]) != 60747:
        raise RuntimeError("Base60747 contract changed")
    if config["base"]["resume_authorized"] is not False:
        raise RuntimeError("Formal Base resume must remain unauthorized")
    targets = config["targets"]
    if (
        targets["manifest_sha256"] != MANIFEST_SHA
        or int(targets["training_record_count"]) != 22
        or tuple(targets["quarantine_exclusions"]) != QUARANTINE
    ):
        raise RuntimeError("target/quarantine contract changed")
    for garment in GARMENTS:
        binding = config["teachers"]["checkpoints"][garment]
        if binding["sha256"] != TEACHER_SHA[garment]:
            raise RuntimeError(f"{garment} Teacher SHA contract changed")
    f2 = config["frozen_f2"]
    if (
        f2["checkpoint_sha256"] != F2_SHA
        or f2["backbone_fingerprint"] != F2_FINGERPRINT
        or int(f2["controller_input_dimension"]) != 512
        or f2["trainable"] is not False
    ):
        raise RuntimeError("frozen F2 contract changed")
    optimization = config["optimization"]
    if (
        optimization["optimizer"] != "torch.optim.Adam"
        or float(optimization["learning_rate"]) != 0.02
        or tuple(optimization["betas"]) != (0.9, 0.999)
        or float(optimization["epsilon"]) != 1e-8
        or float(optimization["weight_decay"]) != 0.0
        or optimization["amsgrad"] is not False
        or int(optimization["steps_per_run"]) != 300
        or tuple(optimization["checkpoint_steps"]) != CHECKPOINT_STEPS
        or optimization["scheduler"] != "constant LambdaLR multiplier 1.0"
        or float(optimization["gradient_clip_norm"]) != 5.0
        or float(optimization["loss"]["coefficient_smooth_l1_beta"]) != 1.0
        or any(
            float(optimization["loss"][name]) != 0.0
            for name in (
                "classification_weight",
                "pairwise_geometry_weight",
                "render_weight",
            )
        )
    ):
        raise RuntimeError("frozen Pure-Endpoint optimizer/loss contract changed")
    if tuple(config["protocol"]["replicate_seeds"]) != SEEDS:
        raise RuntimeError("seed set must be exactly [0,1,2]")
    if config["protocol"].get("fold_denominators") != {
        "train": 6,
        "calibration": 3,
        "test": 3,
    }:
        raise RuntimeError("fold denominator contract changed")
    _rotations(config)
    if Path(config["output"]["root"]) != FORMAL_OUTPUT_ROOT:
        raise RuntimeError("formal output root changed")
    if config["output"]["attempt"] != "attempt_001":
        raise RuntimeError("only attempt_001 is permitted")
    if config.get("paper_final") is not False:
        raise RuntimeError("paper_final must remain false")
    return config


def manifest_records(config: Mapping[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    path = Path(config["targets"]["root"]) / config["targets"]["manifest"]
    payload = read_json(path)
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    all_ids: list[str] = []
    for outfit in payload["outfits"]:
        garment = outfit["outfit_id"]
        for observation in outfit["observations"]:
            request_id = observation["condition_id"]
            marker = request_id.index("_slot") + len("_slot")
            slot = int(request_id[marker : marker + 2])
            key = (garment, slot)
            if key in rows:
                raise RuntimeError(f"duplicate formal target {key}")
            rows[key] = observation
            all_ids.append(request_id)
    if (
        len(rows) != 22
        or len(set(all_ids)) != 22
        or set(all_ids).intersection(QUARANTINE)
    ):
        raise RuntimeError("formal 22-record denominator/quarantine gate failed")
    for slot in (0, 3, 6, 7):
        if any((garment, slot) not in rows for garment in GARMENTS):
            raise RuntimeError(f"CommonSafe4 slot{slot:02d} is incomplete")
    return rows


def fold_record_ids(
    rotation: Mapping[str, Any],
    records: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, list[str]]:
    roles = {
        "train": [int(value) for value in rotation["train_slots"]],
        "calibration": [int(rotation["calibration_slot"])],
        "test": [int(rotation["test_slot"])],
    }
    result = {
        role: [
            str(records[(garment, slot)]["condition_id"])
            for slot in slots
            for garment in GARMENTS
        ]
        for role, slots in roles.items()
    }
    if {role: len(values) for role, values in result.items()} != {
        "train": 6,
        "calibration": 3,
        "test": 3,
    }:
        raise RuntimeError("fold is not exact 6/3/3")
    sets = {role: set(values) for role, values in result.items()}
    if any(
        sets[left].intersection(sets[right])
        for left, right in (
            ("train", "calibration"),
            ("train", "test"),
            ("calibration", "test"),
        )
    ):
        raise RuntimeError("train/calibration/test folds overlap")
    if any(set(values).intersection(QUARANTINE) for values in result.values()):
        raise RuntimeError("quarantine request entered a fold")
    return result


def run_id(rotation: int, seed: int) -> str:
    return f"COMMONSAFE4-METHOD-R{rotation}-S{seed}"


def run_plan(
    config: Mapping[str, Any],
    output_root: Path,
    rotation_ids: Iterable[int],
    seeds: Iterable[int],
) -> list[dict[str, Any]]:
    records = manifest_records(config)
    rotations = _rotations(config)
    rows = []
    for rotation_id in rotation_ids:
        if rotation_id not in rotations:
            raise ValueError(f"rotation-id must be one of {sorted(rotations)}")
        for seed in seeds:
            if seed not in SEEDS:
                raise ValueError(f"seed must be one of {list(SEEDS)}")
            rotation = rotations[rotation_id]
            folds = fold_record_ids(rotation, records)
            identifier = run_id(rotation_id, seed)
            cell = output_root / "attempt_001" / "cells" / identifier
            rows.append(
                {
                    "run_id": identifier,
                    "rotation": rotation_id,
                    "seed": seed,
                    "train_slots": list(rotation["train_slots"]),
                    "calibration_slot": rotation["calibration_slot"],
                    "test_slot": rotation["test_slot"],
                    "train_request_ids": folds["train"],
                    "calibration_request_ids": folds["calibration"],
                    "test_request_ids": folds["test"],
                    "train_count": 6,
                    "calibration_count": 3,
                    "test_count": 3,
                    "output_path": str(cell),
                    "checkpoint_paths": [
                        str(cell / "checkpoints" / f"step_{step:06d}.pth")
                        for step in CHECKPOINT_STEPS
                    ],
                    "metrics_path": str(
                        cell / "evaluation" / "formal_metrics.json"
                    ),
                    "base_sha256": BASE_SHA,
                    "teacher_sha256": TEACHER_SHA,
                    "target_manifest_sha256": MANIFEST_SHA,
                    "quarantine_usage_count": 0,
                    "dual_support_enabled": False,
                    "formal_test_configured": True,
                    "status": "PLANNED_NOT_RUN",
                }
            )
    if len({row["run_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate run ID")
    if len({row["output_path"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate output path")
    return rows


def validate_bindings(config: Mapping[str, Any], *, parse: bool) -> dict[str, Any]:
    bindings = {
        "base": (Path(config["base"]["checkpoint"]), BASE_SHA),
        "manifest": (
            Path(config["targets"]["root"]) / config["targets"]["manifest"],
            MANIFEST_SHA,
        ),
        "f2": (Path(config["frozen_f2"]["checkpoint"]), F2_SHA),
        **{
            f"teacher_{garment}": (
                Path(config["teachers"]["checkpoints"][garment]["path"]),
                TEACHER_SHA[garment],
            )
            for garment in GARMENTS
        },
    }
    result = {}
    for name, (path, expected) in bindings.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"{name} SHA mismatch")
        result[name] = {"path": str(path), "sha256": actual}
    if parse:
        base = torch.load(bindings["base"][0], map_location="cpu", weights_only=False)
        step = int(
            base.get("global_step", base.get("step", base.get("iteration", -1)))
        )
        if step != 60747:
            raise RuntimeError(f"Base internal step mismatch: {step}")
        result["base"]["internal_step"] = step
        for garment in GARMENTS:
            payload = torch.load(
                bindings[f"teacher_{garment}"][0],
                map_location="cpu",
                weights_only=False,
            )
            metadata = payload["metadata"]
            if (
                int(payload.get("global_step", -1)) != 1200
                or int(payload.get("optimizer_step", -1)) != 1200
                or metadata.get("base_checkpoint_sha256") != BASE_SHA
                or int(metadata.get("quarantine_count_in_optimizer", -1)) != 0
            ):
                raise RuntimeError(f"{garment} Teacher metadata mismatch")
            result[f"teacher_{garment}"]["parse_status"] = "PASS"
    return result


@contextlib.contextmanager
def frozen_runtime_snapshot() -> Any:
    """Materialize the exact historical runtime as an isolated detached worktree."""

    temporary_root = Path(tempfile.mkdtemp(prefix="commonsafe4-runtime-"))
    checkout = temporary_root / "runtime"
    common_dir = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--git-common-dir"],
        text=True,
    ).strip()
    common_path = Path(common_dir)
    if not common_path.is_absolute():
        common_path = (REPO_ROOT / common_path).resolve()
    subprocess.run(
        [
            "git",
            f"--git-dir={common_path}",
            "worktree",
            "add",
            "--detach",
            str(checkout),
            FROZEN_RUNTIME_COMMIT,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    sys.path.insert(0, str(checkout))
    try:
        modules = {
            "basis": importlib.import_module("scene.explicit_gaussian_residual_basis"),
            "extractor": importlib.import_module(
                "scene.frozen_f2_linear_coefficient_control"
            ),
            "dataset": importlib.import_module("scene.full_dressable_dataset"),
            "residual": importlib.import_module("scene.gaussian_clothing_residuals"),
            "controller": importlib.import_module(
                "scene.multi_outfit_linear_coefficient_control"
            ),
            "multi": importlib.import_module("tools.run_multi_outfit_explicit_basis"),
        }
        modules["root"] = checkout
        yield modules
    finally:
        if str(checkout) in sys.path:
            sys.path.remove(str(checkout))
        subprocess.run(
            [
                "git",
                f"--git-dir={common_path}",
                "worktree",
                "remove",
                "--force",
                str(checkout),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        shutil.rmtree(temporary_root, ignore_errors=True)


def standardized_targets(
    config: Mapping[str, Any], runtime: Mapping[str, Any]
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    mapping = {
        "raw_xyz": "delta_xyz",
        "raw_log_scaling": "delta_log_scaling",
        "raw_rotvec": "delta_rotvec",
        "raw_opacity": "delta_opacity_logit",
        "raw_sh0": "delta_sh0",
        "raw_shN": "delta_shN",
    }
    teachers = {}
    for garment in GARMENTS:
        payload = torch.load(
            config["teachers"]["checkpoints"][garment]["path"],
            map_location="cpu",
            weights_only=False,
        )
        model = payload["model"]
        teachers[garment] = runtime["residual"].GaussianClothingResiduals(
            **{
                target: model[source].detach().float().contiguous()
                for source, target in mapping.items()
            }
        )
    decomposition = runtime["basis"].build_svd_basis(
        teachers,
        config["basis"]["channel_bounds"],
        GARMENTS,
        rank=2,
    )
    matrix = torch.stack(
        [
            decomposition.teacher_coefficients[garment].double()
            for garment in GARMENTS
        ]
    )
    mean = matrix.mean(0)
    std = matrix.std(0, unbiased=False)
    if torch.any(std <= 0) or not torch.isfinite(matrix).all():
        raise RuntimeError("rank2 endpoint coordinates are degenerate/nonfinite")
    targets = {
        garment: (
            decomposition.teacher_coefficients[garment].double() - mean
        ).div(std).float()
        for garment in GARMENTS
    }
    return targets, {
        "rank": 2,
        "basis_fingerprint": decomposition.metadata["basis_fingerprint"],
        "coefficient_mean": mean.tolist(),
        "coefficient_std_population": std.tolist(),
        "standardized_teacher_coefficients": {
            garment: value.tolist() for garment, value in targets.items()
        },
    }


class FormalFoldEvaluator:
    """Frozen nearest-endpoint evaluator with garment-order tie breaking."""

    def __init__(self, fold: Mapping[str, list[str]]) -> None:
        if len(fold["calibration"]) != 3 or len(fold["test"]) != 3:
            raise RuntimeError("formal evaluator requires exact 3/3 calibration/test")
        self.fold = {name: list(values) for name, values in fold.items()}
        self.candidate_order = GARMENTS

    def evaluate(
        self,
        controller: torch.nn.Module,
        features: Mapping[tuple[str, int], torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        *,
        role: str,
        slot: int,
    ) -> dict[str, Any]:
        rows, errors, correct = [], [], 0
        controller.eval()
        with torch.no_grad():
            for garment in GARMENTS:
                prediction = controller(
                    features[(garment, slot)].cuda()
                ).standardized_coefficients
                target = targets[garment].to(prediction)
                distances = {
                    name: float(
                        (prediction - value.to(prediction)).square().sum()
                    )
                    for name, value in targets.items()
                }
                selected = min(
                    GARMENTS,
                    key=lambda name: (distances[name], GARMENTS.index(name)),
                )
                errors.extend((prediction - target).square().tolist())
                correct += int(selected == garment)
                rows.append(
                    {
                        "request_id": self.fold[role][GARMENTS.index(garment)],
                        "garment": garment,
                        "slot": slot,
                        "selected_endpoint": selected,
                        "correct": selected == garment,
                        "squared_distances": distances,
                    }
                )
        return {
            "role": role,
            "count": 3,
            "finite_prediction_count": 3,
            "standardized_coefficient_rmse": float(
                math.sqrt(float(np.mean(errors)))
            ),
            "nearest_endpoint_correct": correct,
            "nearest_endpoint_total": 3,
            "nearest_endpoint_accuracy": correct / 3.0,
            "rows": rows,
            "tie_break": list(GARMENTS),
        }


def deterministic_smoke_feature(request_id: str, seed: int) -> torch.Tensor:
    value = int(hashlib.sha256(f"{request_id}:{seed}".encode()).hexdigest()[:16], 16)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(value % (2**63 - 1))
    return torch.randn(1, 512, generator=generator, dtype=torch.float32)


def zero_step_smoke(
    config: Mapping[str, Any],
    cells: Iterable[tuple[int, int]],
) -> dict[str, Any]:
    if FORMAL_OUTPUT_ROOT.exists():
        raise FileExistsError("formal CommonSafe4 output root must remain absent")
    if not torch.cuda.is_available():
        raise RuntimeError("zero-step smoke requires CUDA")
    records = manifest_records(config)
    rotations = _rotations(config)
    with frozen_runtime_snapshot() as runtime:
        bindings = validate_bindings(config, parse=True)
        targets, basis = standardized_targets(config, runtime)
        results = []
        for rotation_id, seed in cells:
            if rotation_id not in rotations or seed not in SEEDS:
                raise ValueError("invalid zero-step smoke rotation/seed")
            rotation = rotations[rotation_id]
            fold = fold_record_ids(rotation, records)
            evaluator = FormalFoldEvaluator(fold)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            controller = runtime[
                "controller"
            ].MultiOutfitLinearCoefficientControl(512, 2).cuda()
            if sum(parameter.numel() for parameter in controller.parameters()) != 2050:
                raise RuntimeError("controller parameter count changed")
            first_request = fold["train"][0]
            first_garment = first_request.split("_")[1]
            feature = deterministic_smoke_feature(first_request, seed).cuda()
            prediction = controller(feature).standardized_coefficients
            target = targets[first_garment].to(prediction)
            loss = F.smooth_l1_loss(
                prediction, target, beta=1.0, reduction="mean"
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("zero-step smoke loss is nonfinite")
            loss.backward()
            gradients_finite = all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in controller.parameters()
            )
            if not gradients_finite:
                raise FloatingPointError("zero-step smoke gradient is nonfinite")
            results.append(
                {
                    "cell_id": f"R{rotation_id}/S{seed}",
                    "run_id": run_id(rotation_id, seed),
                    "rotation": rotation_id,
                    "seed": seed,
                    "fold_counts": {
                        role: len(values) for role, values in fold.items()
                    },
                    "quarantine_usage_count": 0,
                    "dual_support_call_count": 0,
                    "controller_parameter_count": 2050,
                    "loss": float(loss.detach()),
                    "loss_finite": True,
                    "gradients_finite": True,
                    "gradient_audit": {
                        name: {
                            "present": parameter.grad is not None,
                            "finite": parameter.grad is None
                            or bool(torch.isfinite(parameter.grad).all()),
                        }
                        for name, parameter in controller.named_parameters()
                    },
                    "formal_test_evaluator_constructed": isinstance(
                        evaluator, FormalFoldEvaluator
                    ),
                    "gpu_forward_calls": 1,
                    "backward_calls": 1,
                    "optimizer_created": False,
                    "optimizer_steps": 0,
                    "feature_policy": (
                        "DETERMINISTIC_SYNTHETIC_512D_STRUCTURAL_SMOKE_ONLY;"
                        "FROZEN_F2_NOT_EVALUATED"
                    ),
                    "formal_output_root_absent_after": not FORMAL_OUTPUT_ROOT.exists(),
                    "status": "PASS_ZERO_STEP",
                }
            )
            del controller, feature, prediction, target, loss
            torch.cuda.empty_cache()
    if FORMAL_OUTPUT_ROOT.exists():
        raise RuntimeError("zero-step smoke created the formal output root")
    return {
        "schema_version": "canondressgs.subject00.commonsafe4.zero_step_smoke.v1",
        "task_id": TASK_ID,
        "cells": results,
        "cell_count": len(results),
        "pass_count": sum(row["status"] == "PASS_ZERO_STEP" for row in results),
        "base_teacher_binding_status": "PASS",
        "binding_count": len(bindings),
        "basis": basis,
        "formal_test_evaluator_status": "PASS_CONSTRUCTED",
        "gpu_forward_calls": len(results),
        "backward_calls": len(results),
        "optimizer_steps": 0,
        "dual_support_call_count": 0,
        "formal_output_root_created": False,
        "status": "PASS",
    }


def build_real_features(
    config: Mapping[str, Any],
    runtime: Mapping[str, Any],
    slots: Iterable[int],
) -> tuple[dict[tuple[str, int], torch.Tensor], dict[str, Any]]:
    manifest = Path(config["targets"]["root"]) / config["targets"]["manifest"]
    dataset = runtime["dataset"].FullDressableTrainingDataset(
        manifest, split="train", reference_count=3, seed=0
    )
    subject02 = yaml.safe_load(
        (
            runtime["root"]
            / "configs/research/subject02_multi_outfit_explicit_basis_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    multi = runtime["multi"]
    old_branch, old_head = multi.EXPECTED_BRANCH, multi.EXPECTED_SOURCE_HEAD
    multi.EXPECTED_BRANCH, multi.EXPECTED_SOURCE_HEAD = "", FROZEN_RUNTIME_COMMIT
    try:
        context = multi._build_cs_context(subject02)
    finally:
        multi.EXPECTED_BRANCH, multi.EXPECTED_SOURCE_HEAD = old_branch, old_head
    if context["backbone_before"] != F2_FINGERPRINT:
        raise RuntimeError("frozen F2 backbone fingerprint mismatch")
    extractor = runtime["extractor"].FrozenF2ReferenceFeatureExtractor(
        context["legacy_model"].clothing_observation_encoder.backbone[:-1],
        feature_dim=128,
    ).cuda().eval()
    features: dict[tuple[str, int], torch.Tensor] = {}
    episodes = {}
    for garment in GARMENTS:
        for slot in slots:
            matches = [
                (index, outfit, observation)
                for index, (outfit, observation) in enumerate(dataset.samples)
                if outfit["outfit_id"] == garment
                and f"_slot{slot:02d}_" in observation["condition_id"]
            ]
            if len(matches) != 1:
                raise RuntimeError(f"{garment}/slot{slot:02d} target unavailable")
            index, outfit, target = matches[0]
            selected = dataset._references(outfit, target["condition_id"], index)
            ids = [value["condition_id"] for value in selected]
            if target["condition_id"] in ids or set(ids).intersection(QUARANTINE):
                raise RuntimeError("target leakage or quarantine in reference set")
            rows = []
            denominators = []
            with torch.no_grad():
                for selected_record in selected:
                    reference = dataset._observation(outfit, selected_record, True)
                    image = reference["rgb"].unsqueeze(0).cuda()
                    mask = reference["clothing_mask"].unsqueeze(0).cuda()
                    output = extractor(
                        image,
                        mask,
                        torch.ones(1, 1, device="cuda", dtype=image.dtype),
                    )
                    rows.append(output.per_reference_f2.reshape(-1))
                    denominators.append(float(output.pooling_denominator.min()))
            per_reference = torch.stack(rows)
            feature = torch.cat(
                (per_reference.mean(0), per_reference.max(0).values)
            ).reshape(1, -1)
            if feature.shape != (1, 512) or not torch.isfinite(feature).all():
                raise RuntimeError("frozen F2 feature invalid")
            if min(denominators) <= 0:
                raise RuntimeError("reference clothing-mask denominator is zero")
            features[(garment, slot)] = feature.detach().cpu()
            episodes[f"{garment}/slot{slot:02d}"] = {
                "target_condition_id": target["condition_id"],
                "reference_condition_ids": ids,
                "reference_count": 3,
                "target_excluded": True,
                "pooling_denominators": denominators,
            }
    return features, {
        "status": "PASS",
        "episode_count": len(features),
        "episodes": episodes,
        "f2_checkpoint_sha256": F2_SHA,
        "f2_backbone_fingerprint": F2_FINGERPRINT,
        "quarantine_usage_count": 0,
    }


def save_checkpoint(
    path: Path,
    *,
    rotation: int,
    seed: int,
    step: int,
    controller: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    history: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> None:
    atomic_torch(
        path,
        {
            "schema_version": (
                "canondressgs.subject00.commonsafe4.pure_endpoint_checkpoint.v1"
            ),
            "task_id": TASK_ID,
            "run_id": run_id(rotation, seed),
            "rotation": rotation,
            "seed": seed,
            "global_step": step,
            "optimizer_step": step,
            "model": controller.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "bindings": {
                "base_checkpoint_sha256": BASE_SHA,
                "teacher_checkpoint_sha256": TEACHER_SHA,
                "target_manifest_sha256": MANIFEST_SHA,
                "f2_checkpoint_sha256": F2_SHA,
                "f2_backbone_fingerprint": F2_FINGERPRINT,
                "quarantine_usage_count": 0,
                "dual_support_call_count": 0,
            },
            "paper_eligible": False,
        },
    )


def train_cell(
    config: Mapping[str, Any],
    runtime: Mapping[str, Any],
    features_cpu: Mapping[tuple[str, int], torch.Tensor],
    targets_cpu: Mapping[str, torch.Tensor],
    rotation_id: int,
    seed: int,
    cell: Path,
) -> dict[str, Any]:
    rotation = _rotations(config)[rotation_id]
    records = manifest_records(config)
    fold = fold_record_ids(rotation, records)
    evaluator = FormalFoldEvaluator(fold)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    controller = runtime[
        "controller"
    ].MultiOutfitLinearCoefficientControl(512, 2).cuda()
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
    features = {key: value.cuda() for key, value in features_cpu.items()}
    targets = {key: value.cuda() for key, value in targets_cpu.items()}
    history: list[dict[str, Any]] = []
    save_checkpoint(
        cell / "checkpoints/step_000000.pth",
        rotation=rotation_id,
        seed=seed,
        step=0,
        controller=controller,
        optimizer=optimizer,
        scheduler=scheduler,
        history=history,
        config=config,
    )
    for step in range(1, 301):
        slot = int(rotation["train_slots"][(step - 1) % 2])
        optimizer.zero_grad(set_to_none=True)
        predictions = torch.stack(
            [
                controller(
                    features[(garment, slot)]
                ).standardized_coefficients
                for garment in GARMENTS
            ]
        )
        target_batch = torch.stack([targets[garment] for garment in GARMENTS])
        loss = F.smooth_l1_loss(
            predictions, target_batch, beta=1.0, reduction="mean"
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at step {step}")
        loss.backward()
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in controller.parameters()
        ):
            raise FloatingPointError(f"nonfinite gradient at step {step}")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            controller.parameters(), 5.0
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"nonfinite gradient norm at step {step}")
        optimizer.step()
        scheduler.step()
        history.append(
            {
                "optimizer_step": step,
                "condition_slot": slot,
                "loss": float(loss.detach()),
                "gradient_norm_before_clip": float(gradient_norm),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if step in CHECKPOINT_STEPS:
            save_checkpoint(
                cell / "checkpoints" / f"step_{step:06d}.pth",
                rotation=rotation_id,
                seed=seed,
                step=step,
                controller=controller,
                optimizer=optimizer,
                scheduler=scheduler,
                history=history,
                config=config,
            )
    calibration = evaluator.evaluate(
        controller,
        features,
        targets,
        role="calibration",
        slot=int(rotation["calibration_slot"]),
    )
    test = evaluator.evaluate(
        controller,
        features,
        targets,
        role="test",
        slot=int(rotation["test_slot"]),
    )
    result = {
        "run_id": run_id(rotation_id, seed),
        "rotation": rotation_id,
        "seed": seed,
        "optimizer_steps": 300,
        "checkpoints": list(CHECKPOINT_STEPS),
        "loss_initial": history[0]["loss"],
        "loss_final": history[-1]["loss"],
        "calibration": calibration,
        "formal_test": test,
        "quarantine_usage_count": 0,
        "dual_support_call_count": 0,
        "status": "COMPLETE",
    }
    atomic_json(cell / "evaluation/formal_metrics.json", result)
    return result


def execute(
    config: Mapping[str, Any],
    output_root: Path,
    cells: list[tuple[int, int]],
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"formal output collision: {output_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("formal execution requires CUDA")
    state = git_state()
    if not state["clean"]:
        raise RuntimeError("formal execution requires a clean Git worktree")
    output_root.mkdir(parents=False, exist_ok=False)
    lock = output_root / "METHOD_MATRIX_EXECUTION_LOCK.json"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, json.dumps({"pid": os.getpid(), "cells": cells}).encode())
    os.close(descriptor)
    attempt = output_root / "attempt_001"
    try:
        validate_bindings(config, parse=False)
        with frozen_runtime_snapshot() as runtime:
            validate_bindings(config, parse=True)
            targets, basis = standardized_targets(config, runtime)
            features, feature_audit = build_real_features(
                config, runtime, slots=(0, 3, 6, 7)
            )
            atomic_json(attempt / "shared/basis_audit.json", basis)
            atomic_json(attempt / "shared/feature_audit.json", feature_audit)
            results = []
            for rotation, seed in cells:
                cell = attempt / "cells" / run_id(rotation, seed)
                cell.mkdir(parents=True, exist_ok=False)
                results.append(
                    train_cell(
                        config,
                        runtime,
                        features,
                        targets,
                        rotation,
                        seed,
                        cell,
                    )
                )
        final = {
            "task_id": TASK_ID,
            "completed_run_count": len(results),
            "optimizer_steps": 300 * len(results),
            "runs": results,
            "status": "COMPLETE",
        }
        atomic_json(attempt / "MATRIX_RESULT.json", final)
        return final
    except Exception:
        atomic_json(
            output_root / "EXECUTION_FAILURE.json",
            {"task_id": TASK_ID, "status": "FAIL", "pid": os.getpid()},
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--rotation-id", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--attempt-id", default="attempt_001")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--zero-step-smoke", action="store_true")
    modes.add_argument("--execute-single-run", action="store_true")
    modes.add_argument("--execute-matrix", action="store_true")
    parser.add_argument("--resume-policy", default="none", choices=("none",))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = validate_config(args.config.resolve())
    if args.attempt_id != "attempt_001":
        raise ValueError("attempt-id must be exactly attempt_001")
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else Path(config["output"]["root"])
    )
    if output_root != FORMAL_OUTPUT_ROOT:
        raise ValueError(f"output-root must be exactly {FORMAL_OUTPUT_ROOT}")
    if args.resume_policy != "none":
        raise ValueError("resume-policy only supports none")
    if args.execute_single_run or (
        args.dry_run and (args.rotation_id is not None or args.seed is not None)
    ):
        if args.rotation_id is None or args.seed is None:
            raise ValueError("rotation-id and seed must be supplied together")
        cells = [(args.rotation_id, args.seed)]
    elif args.zero_step_smoke:
        if (args.rotation_id is None) != (args.seed is None):
            raise ValueError("rotation-id and seed must be supplied together")
        cells = (
            [(args.rotation_id, args.seed)]
            if args.rotation_id is not None
            else list(ZERO_STEP_SMOKE_CELLS)
        )
    else:
        if args.rotation_id is not None or args.seed is not None:
            raise ValueError("matrix mode does not accept rotation-id or seed")
        cells = [(rotation, seed) for rotation in range(4) for seed in SEEDS]
    if FORMAL_OUTPUT_ROOT.exists():
        raise FileExistsError(f"formal output root collision: {FORMAL_OUTPUT_ROOT}")
    if args.dry_run:
        bindings = validate_bindings(config, parse=False)
        plans = (
            run_plan(config, output_root, [cells[0][0]], [cells[0][1]])
            if len(cells) == 1
            else run_plan(config, output_root, range(4), SEEDS)
        )
        payload = {
            "schema_version": "canondressgs.subject00.commonsafe4.runner_dry_run.v1",
            "task_id": TASK_ID,
            "method_contract": "PURE_ENDPOINT",
            "dual_support_enabled": False,
            "cells": plans,
            "cell_count": len(plans),
            "pass_count": len(plans),
            "bindings": bindings,
            "optimizer_steps": 0,
            "formal_output_root_created": False,
            "status": "PASS",
        }
    elif args.zero_step_smoke:
        payload = zero_step_smoke(config, cells)
    elif args.execute_single_run:
        payload = execute(config, output_root, cells)
    else:
        payload = execute(config, output_root, cells)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
