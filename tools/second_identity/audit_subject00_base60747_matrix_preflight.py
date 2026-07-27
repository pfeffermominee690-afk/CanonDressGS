#!/usr/bin/env python3
"""Read-only preflight for the Subject00 Base60747 method matrix continuation."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TASK_ID = "AAAI27-SUBJECT00-BASE60747-METHOD-MATRIX-AND-FAIR-BASELINES-001"
SOURCE_HEAD = "ed44835f950fe369dd3eb64b5eaef8c6cc174175"
BRANCH = "research/subject00-base60747-method-matrix-fair-baselines-20260727"
BASE_SHA = "2d09b1ce6bb19b8b6418314caab8dc1cf92d70e432a9e09e009989332ea4a2f7"
MANIFEST_SHA = "602820292fea47fe8116bf64824e6e2a8313a3ec31b8fc7cdd66a90c6bad33c1"
CHECKPOINT_STEPS = (0, 20, 50, 100, 200, 300)
GARMENTS = ("O01", "O03", "O04")
ROTATIONS = {
    0: {"train_slots": (0, 7), "calibration_slot": 3, "test_slot": 4},
    1: {"train_slots": (7, 3), "calibration_slot": 4, "test_slot": 0},
    2: {"train_slots": (3, 4), "calibration_slot": 0, "test_slot": 7},
    3: {"train_slots": (4, 0), "calibration_slot": 7, "test_slot": 3},
}
SEEDS = (0, 1, 2)
QUARANTINE = (
    "subject00_O01_slot04_remaining_attempt005_cand00",
    "subject00_O03_slot04_canary_attempt004_cand00",
)
BASELINE_NAMES = (
    "Reference Classifier Lookup",
    "Nearest-Centroid Lookup",
    "Outfit-ID Oracle",
    "Teacher Endpoint",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def lf_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def tree_fingerprint(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    rows = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root).as_posix()
        file_sha = sha256(path)
        size = path.stat().st_size
        rows.append({"path": relative, "bytes": size, "sha256": file_sha})
        digest.update(f"{relative}\0{size}\0{file_sha}\n".encode())
    return {
        "sha256": digest.hexdigest(),
        "file_count": len(rows),
        "bytes": sum(row["bytes"] for row in rows),
        "files": rows,
    }


def git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    return {
        "branch": run("branch", "--show-current"),
        "head": run("rev-parse", "HEAD"),
        "merge_base_with_source": run("merge-base", "HEAD", SOURCE_HEAD),
        "porcelain_v2": run("status", "--porcelain=v2"),
    }


def checkpoint_record(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    history = payload["history"]
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "schema_version": payload.get("schema_version"),
        "rotation": payload.get("rotation"),
        "seed": payload.get("seed"),
        "global_step": payload.get("global_step"),
        "optimizer_step": payload.get("optimizer_step"),
        "history_count": len(history),
        "history_step_set": [row["optimizer_step"] for row in history],
        "loss_first": history[0]["loss"] if history else None,
        "loss_last": history[-1]["loss"] if history else None,
        "loss_finite": all(
            isinstance(row["loss"], (int, float))
            and row["loss"] == row["loss"]
            and abs(row["loss"]) != float("inf")
            for row in history
        ),
        "model_keys": sorted(payload["model"]),
        "optimizer_state_count": len(payload["optimizer"]["state"]),
        "scheduler_last_epoch": payload["scheduler"].get("last_epoch"),
        "bindings": payload["bindings"],
        "paper_eligible": payload.get("paper_eligible"),
    }


def main() -> int:
    root = Path.cwd().resolve()
    config_path = root / "configs/research/subject00_canondressgs_method_base60747_v1.json"
    runner_path = root / "tools/second_identity/run_subject00_base60747_method.py"
    implementation_path = root / "tools/paper/formal_batch_runtime.py"
    registry_path = (
        root
        / "paper_protocol/reviewer_risk/subject00_base60747_three_garment_teacher_registry_20260727.json"
    )
    evaluator_path = (
        root / "paper_protocol/reviewer_risk/subject00_second_identity_evaluator_contract.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    evaluator = json.loads(evaluator_path.read_text(encoding="utf-8"))

    base_path = Path(config["base"]["checkpoint"])
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    base_required = {
        "_xyz", "xyz_offset", "dxyz_vt", "_scaling", "_rotation", "_opacity",
        "_sh0", "_shN", "_weights", "surface_attachment", "encoder_feat_params",
        "dxyz_bs", "sh0_bs", "shN_bs", "scaling_bs", "rotation_bs", "opacity_bs",
        "optimizer_states", "scheduler_states", "python_rng_state",
        "numpy_rng_state", "torch_rng_state", "cuda_rng_states", "data_order_position",
        "sampler_sha256", "global_data_order_sha256", "single_pass_data_order_sha256",
    }
    base_audit = {
        "path": str(base_path),
        "bytes": base_path.stat().st_size,
        "sha256": sha256(base_path),
        "internal_step": base.get("training_step"),
        "required_field_count": len(base_required),
        "missing_fields": sorted(base_required.difference(base)),
        "partial_or_tmp_count": len(list(base_path.parent.glob("*tmp*"))),
    }
    del base

    teacher_audits = {}
    for garment in GARMENTS:
        record = registry["garments"][garment]
        path = Path(record["teacher_checkpoint"])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        teacher_audits[garment] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "registry_sha256": record["teacher_checkpoint_sha256"],
            "technical_status": record["technical_status"],
            "human_status": record["human_status"],
            "scientific_pass": record["scientific_pass"],
            "global_step": payload.get("global_step"),
            "optimizer_step": payload.get("optimizer_step"),
            "base_sha256": payload["metadata"].get("base_checkpoint_sha256"),
            "quarantine_count_in_optimizer": payload["metadata"].get(
                "quarantine_count_in_optimizer"
            ),
            "paper_eligible": payload.get("paper_eligible"),
        }
        del payload

    target_root = Path(config["targets"]["root"])
    manifest_path = target_root / config["targets"]["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target_rows = []
    target_mismatches = []
    for outfit in manifest["outfits"]:
        for observation in outfit["observations"]:
            for field, expected in observation["checksums"].items():
                path = (manifest_path.parent / observation[field]).resolve()
                actual = sha256(path)
                target_rows.append({
                    "garment": outfit["outfit_id"],
                    "condition_id": observation["condition_id"],
                    "field": field,
                    "path": str(path),
                    "sha256": actual,
                })
                if actual != expected:
                    target_mismatches.append({
                        "condition_id": observation["condition_id"],
                        "field": field,
                        "expected": expected,
                        "actual": actual,
                    })
    request_ids = [
        observation["condition_id"]
        for outfit in manifest["outfits"]
        for observation in outfit["observations"]
    ]

    coverage = {}
    for slot in (0, 3, 4, 7):
        coverage[str(slot)] = {}
        for garment in GARMENTS:
            coverage[str(slot)][garment] = [
                observation["condition_id"]
                for outfit in manifest["outfits"]
                if outfit["outfit_id"] == garment
                for observation in outfit["observations"]
                if f"_slot{slot:02d}_" in observation["condition_id"]
            ]

    rotation_coverage = {}
    matrix_blockers = []
    for rotation, contract in ROTATIONS.items():
        train = {
            str(slot): {
                garment: len(coverage[str(slot)][garment])
                for garment in GARMENTS
            }
            for slot in contract["train_slots"]
        }
        calibration = {
            garment: len(coverage[str(contract["calibration_slot"])][garment])
            for garment in GARMENTS
        }
        test = {
            garment: len(coverage[str(contract["test_slot"])][garment])
            for garment in GARMENTS
        }
        train_complete = all(
            count == 1
            for slot in train.values()
            for count in slot.values()
        )
        calibration_complete = all(count == 1 for count in calibration.values())
        test_complete = all(count == 1 for count in test.values())
        rotation_coverage[str(rotation)] = {
            "contract": contract,
            "train_counts": train,
            "calibration_counts": calibration,
            "test_counts": test,
            "train_complete": train_complete,
            "calibration_complete": calibration_complete,
            "test_complete": test_complete,
        }
        for role, complete in (
            ("train", train_complete),
            ("calibration", calibration_complete),
            ("test", test_complete),
        ):
            if not complete:
                matrix_blockers.append({
                    "rotation": rotation,
                    "role": role,
                    "reason": "QUARANTINED_SLOT04_REMOVES_O01_O03_REQUIRED_FOLD_RECORDS",
                })

    method_output = Path(config["output"]["root"]) / config["output"]["attempt"]
    completed_tree = tree_fingerprint(method_output)
    checkpoint_dir = method_output / "checkpoints/rotation_00_seed_000"
    checkpoints = [checkpoint_record(path) for path in sorted(checkpoint_dir.glob("*.pth"))]
    final_report = json.loads((method_output / "FINAL_REPORT.json").read_text(encoding="utf-8"))
    initial_summary = json.loads(
        (method_output / "training/initial_run_summary.json").read_text(encoding="utf-8")
    )
    initial_stability = json.loads(
        (method_output / "training/initial_stability_step200.json").read_text(encoding="utf-8")
    )
    evaluation = json.loads(
        (method_output / "evaluation/rotation00_seed000_step300.json").read_text(
            encoding="utf-8"
        )
    )
    mtimes = [
        path.stat().st_mtime
        for path in method_output.rglob("*")
        if path.is_file()
    ]

    completed_cell_checks = {
        "checkpoint_steps_exact": [item["global_step"] for item in checkpoints]
        == list(CHECKPOINT_STEPS),
        "checkpoint_internal_rotation_seed_exact": all(
            item["rotation"] == 0 and item["seed"] == 0 for item in checkpoints
        ),
        "step300_history_exact": checkpoints[-1]["history_step_set"] == list(range(1, 301)),
        "base_binding_exact": all(
            item["bindings"]["base_checkpoint_sha256"] == BASE_SHA for item in checkpoints
        ),
        "teacher_bindings_exact": all(
            item["bindings"]["teacher_checkpoint_sha256"]
            == {
                garment: registry["garments"][garment]["teacher_checkpoint_sha256"]
                for garment in GARMENTS
            }
            for item in checkpoints
        ),
        "target_manifest_binding_exact": all(
            item["bindings"]["target_manifest_sha256"] == MANIFEST_SHA
            for item in checkpoints
        ),
        "quarantine_count_zero": all(
            item["bindings"]["quarantine_count"] == 0 for item in checkpoints
        ),
        "optimizer_steps_exact": checkpoints[-1]["optimizer_step"] == 300,
        "loss_finite": checkpoints[-1]["loss_finite"],
        "parameter_changes_recorded": all(
            initial_summary["trainable_parameter_changes"].values()
        ),
        "nan_inf_none": initial_summary["nan_inf_status"] == "NONE",
        "oom_none": initial_summary["oom_status"] == "NONE",
        "step200_stability_pass": initial_stability["status"] == "PASS",
        "training_fold_evaluation_present": evaluation["episode_count"] == 6,
        "formal_test_fold_evaluation_present": False,
        "formal_test_fold_complete": rotation_coverage["0"]["test_complete"],
    }
    completed_cell_authentic_training = all(
        value
        for key, value in completed_cell_checks.items()
        if key not in {"formal_test_fold_evaluation_present", "formal_test_fold_complete"}
    )
    completed_cell_formal_matrix_valid = all(completed_cell_checks.values())

    expected_cells = [
        {
            "run_id": f"METHOD-R{rotation}-S{seed}",
            "rotation": rotation,
            "seed": seed,
            "status": (
                "COMPLETED_TRAINING_ONLY_FORMAL_TEST_BLOCKED"
                if (rotation, seed) == (0, 0)
                else "BLOCKED_NOT_RUN"
            ),
            "train_slots": list(ROTATIONS[rotation]["train_slots"]),
            "calibration_slot": ROTATIONS[rotation]["calibration_slot"],
            "test_slot": ROTATIONS[rotation]["test_slot"],
            "attempt_id": "attempt_001",
            "expected_checkpoints": list(CHECKPOINT_STEPS),
        }
        for rotation in ROTATIONS
        for seed in SEEDS
    ]

    result = {
        "schema_version": "canondressgs.subject00.base60747_matrix_preflight.v1",
        "task_id": TASK_ID,
        "git": git_state(root),
        "source_head": SOURCE_HEAD,
        "method_files": {
            "implementation": {
                "path": str(implementation_path),
                "sha256": sha256(implementation_path),
                "sha256_lf": lf_sha256(implementation_path),
            },
            "runner": {
                "path": str(runner_path),
                "sha256": sha256(runner_path),
                "sha256_lf": lf_sha256(runner_path),
            },
            "config": {
                "path": str(config_path),
                "sha256": sha256(config_path),
                "sha256_lf": lf_sha256(config_path),
            },
            "teacher_registry": {
                "path": str(registry_path),
                "sha256": sha256(registry_path),
                "sha256_lf": lf_sha256(registry_path),
            },
        },
        "method_contract": "PURE_ENDPOINT",
        "dual_support_enabled": False,
        "base": base_audit,
        "formal_base": {
            "status": config["base"]["formal_status"],
            "resume_authorized": config["base"]["resume_authorized"],
        },
        "teachers": teacher_audits,
        "targets": {
            "root": str(target_root),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "training_count": len(request_ids),
            "evaluation_count": len(request_ids),
            "quarantine_count": len(QUARANTINE),
            "quarantine_ids": list(QUARANTINE),
            "quarantine_in_manifest": sorted(set(QUARANTINE).intersection(request_ids)),
            "checksum_binding_count": len(target_rows),
            "checksum_mismatch_count": len(target_mismatches),
            "checksum_mismatches": target_mismatches,
            "cardinal_slot_coverage": coverage,
        },
        "matrix": {
            "rotations": ROTATIONS,
            "seeds": list(SEEDS),
            "expected_run_count": 12,
            "expected_optimizer_steps": 3600,
            "cells": expected_cells,
            "rotation_coverage": rotation_coverage,
            "blockers": matrix_blockers,
        },
        "completed_rotation0_seed0": {
            "output_root": str(method_output),
            "tree_fingerprint": completed_tree,
            "wall_time_seconds_by_file_mtime": max(mtimes) - min(mtimes),
            "checkpoints": checkpoints,
            "final_report": final_report,
            "initial_run_summary": initial_summary,
            "initial_stability": initial_stability,
            "evaluation": evaluation,
            "checks": completed_cell_checks,
            "authentic_training_run": completed_cell_authentic_training,
            "formal_matrix_cell_valid": completed_cell_formal_matrix_valid,
            "formal_matrix_invalid_reason": (
                "ROTATION0_FORMAL_TEST_SLOT04_HAS_ONLY_O04; RUNNER_EVALUATED_TRAIN_FOLDS_6_OF_6, "
                "NOT THE FROZEN TEST FOLD"
            ),
        },
        "baselines": {
            "contract_source": str(evaluator_path),
            "contract_source_sha256": sha256(evaluator_path),
            "contract_status": evaluator["status"],
            "exact_names": list(BASELINE_NAMES),
            "source_names": [item["name"] for item in evaluator["baselines"]],
            "unique_set_recovered": [item["name"] for item in evaluator["baselines"]]
            == list(BASELINE_NAMES),
            "execution_authorized": False,
            "execution_blocker": "METHOD_MATRIX_NOT_12_OF_12",
        },
        "execution_decision": {
            "new_method_runs_authorized": False,
            "baseline_runs_authorized": False,
            "reason": "FROZEN_ROTATION_FOLDS_CANNOT_BE_INSTANTIATED_WITH_22_RECORD_QUARANTINE_POLICY",
            "automatic_retry": False,
            "attempt_002_allowed": False,
            "gpu_initialized": False,
            "optimizer_created": False,
            "optimizer_steps": 0,
        },
        "paper_eligible": False,
        "scientific_pass": None,
        "human_visual_decision": None,
        "paper_final": False,
    }
    result["critical_checks"] = {
        "branch_exact": result["git"]["branch"] == BRANCH,
        "source_is_merge_base": result["git"]["merge_base_with_source"] == SOURCE_HEAD,
        "worktree_clean": result["git"]["porcelain_v2"] == "",
        "base_sha_exact": base_audit["sha256"] == BASE_SHA,
        "base_step_exact": base_audit["internal_step"] == 60747,
        "base_state_complete": not base_audit["missing_fields"],
        "base_no_partial_tmp": base_audit["partial_or_tmp_count"] == 0,
        "formal_base_paused": config["base"]["formal_status"] == "USER_AUTHORIZED_PAUSED",
        "formal_base_resume_unauthorized": config["base"]["resume_authorized"] is False,
        "teacher_sha_exact": all(
            teacher_audits[garment]["sha256"]
            == teacher_audits[garment]["registry_sha256"]
            for garment in GARMENTS
        ),
        "teacher_technical_pass": all(
            teacher_audits[garment]["technical_status"] == "TECHNICAL_PASS"
            for garment in GARMENTS
        ),
        "teacher_quarantine_zero": all(
            teacher_audits[garment]["quarantine_count_in_optimizer"] == 0
            for garment in GARMENTS
        ),
        "manifest_sha_exact": sha256(manifest_path) == MANIFEST_SHA,
        "target_count_exact": len(request_ids) == 22 and len(set(request_ids)) == 22,
        "quarantine_absent": not set(QUARANTINE).intersection(request_ids),
        "target_checksum_bindings_exact": not target_mismatches and len(target_rows) == 330,
        "completed_tree_immutable": (
            completed_tree["sha256"]
            == "ce281935d12e20c6bab2e3e8fc3256b512594fd8388c3ed299f71199072ec363"
        ),
        "completed_training_authentic": completed_cell_authentic_training,
        "completed_formal_cell_rejected": not completed_cell_formal_matrix_valid,
        "slot04_gap_exact": coverage["4"] == {
            "O01": [], "O03": [], "O04": ["subject00_O04_slot04_cand01"]
        },
        "new_execution_forbidden": not result["execution_decision"]["new_method_runs_authorized"],
        "baseline_execution_forbidden": not result["execution_decision"]["baseline_runs_authorized"],
        "dual_support_disabled": result["dual_support_enabled"] is False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all(result["critical_checks"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
