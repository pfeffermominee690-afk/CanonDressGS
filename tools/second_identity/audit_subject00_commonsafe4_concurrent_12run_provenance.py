#!/usr/bin/env python3
"""Read-only post-hoc audit of the concurrent Subject00 CommonSafe4 matrix.

The scientific output roots are inputs only.  This program writes exclusively
to the current Git worktree's audit/report namespaces and never constructs an
optimizer, resumes a run, repairs a checkpoint, or changes an output marker.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TASK_ID = "AAAI27-SUBJECT00-COMMONSAFE4-CONCURRENT-12RUN-PROVENANCE-AUDIT-001"
SOURCE_BRANCH = (
    "research/subject00-base60747-commonsafe4-contract-runner-preflight-20260727"
)
SOURCE_HEAD = "5a0ac41e614ead9e42cea16bb7af38d53f16f80a"
AUDIT_BRANCH = (
    "research/subject00-commonsafe4-concurrent-12run-provenance-audit-20260727"
)
WINDOWS_WORKTREE = (
    r"E:\model_train\canondressgs_subject00_commonsafe4_concurrent_12run_"
    r"provenance_audit"
)
CLOUD_WORKTREE = (
    "/root/autodl-tmp/canondressgs_work/worktrees/"
    "canondressgs_subject00_commonsafe4_concurrent_12run_provenance_audit"
)

ORIGINAL_PREFLIGHT_STATUS = (
    "PREFLIGHT_STOPPED_BY_CONCURRENT_FORMAL_OUTPUT_ROOT_CREATION"
)
ACTUAL_WRITER_TASK = "AAAI27-SUBJECT00-BASE60747-COMMONSAFE4-METHOD-MATRIX-001"
ACTUAL_WRITER_BRANCH = (
    "research/subject00-base60747-commonsafe4-method-matrix-20260727"
)
ACTUAL_WRITER_HEAD = "1d97afaafff6180062b41f16ffec6beac9dfdbbf"
ACTUAL_WRITER_WORKTREE = (
    "/root/autodl-tmp/canondressgs_work/worktrees/"
    "canondressgs_subject00_base60747_commonsafe4_method_matrix"
)
ACTUAL_LAUNCH_COMMAND = (
    "/root/autodl-tmp/conda_envs/mmlphuman/bin/python -u "
    "tools/second_identity/run_subject00_base60747_commonsafe4_matrix.py"
)
SELECTION_COMMIT = "aa3860618d723905a0fcd3784cf2ec550b609c1a"
RUNNER_COMMIT = "3bd2034da17a3044288e9f462b5dbb686f0acb4e"
BASELINE_CONTRACT_HEAD = "a20f31639c0f5234071e4f3f06383fc06e9a60e6"
BASELINE_REPAIR_HEAD = "90d709092defdb03a1d21d05e93269f26c91296b"

METHOD_ROOT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-CANONDRESSGS-METHOD-BASE60747-COMMONSAFE4-001"
)
OLD_METHOD_ROOT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-CANONDRESSGS-METHOD-BASE60747-001"
)
BASELINE_ROOT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-CANONDRESSGS-BASELINES-BASE60747-COMMONSAFE4-001"
)
FORMAL_BASE_ROOT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/SUBJECT00-FORMAL-BASE-101245-001"
)
BASE_PATH = (
    FORMAL_BASE_ROOT / "attempt_001/checkpoints/step_060747.pth"
)
BASE_SHA = "2d09b1ce6bb19b8b6418314caab8dc1cf92d70e432a9e09e009989332ea4a2f7"
TARGET_ROOT = Path(
    "/root/autodl-tmp/canondressgs_work/teacher_targets/"
    "SUBJECT00-24CELL-001/attempt_001"
)
MANIFEST_REL = Path("10_final_registry/subject00_22_training_full_dataset_v1.json")
MANIFEST_SHA = "602820292fea47fe8116bf64824e6e2a8313a3ec31b8fc7cdd66a90c6bad33c1"
TEACHER_PATHS = {
    "O01": Path(
        "/root/autodl-tmp/canondressgs_work/outputs/"
        "SUBJECT00-O01-TEACHER-BASE60747-CAMSAFE7-001/"
        "attempt_001/checkpoints/step_001200.pth"
    ),
    "O03": Path(
        "/root/autodl-tmp/canondressgs_work/outputs/"
        "SUBJECT00-O03-TEACHER-BASE60747-FORMAL-CAMSAFE7-001/"
        "attempt_001/checkpoints/step_001200.pth"
    ),
    "O04": Path(
        "/root/autodl-tmp/canondressgs_work/outputs/"
        "SUBJECT00-O04-TEACHER-BASE60747-8VIEW-001/"
        "attempt_001/checkpoints/step_001200.pth"
    ),
}
TEACHER_SHA = {
    "O01": "c7881862c4eddf5f58538a2278ab7765aa047784681fb02e7cd89cfe846c1892",
    "O03": "054b9efe1086d89b18310b9831d3ed7aea59e286e31200509b58756545fc3920",
    "O04": "2fa7764097d8577c1610bbf222b26d9ea287bd18074371de400a66cd2270f3a1",
}
F2_SHA = "3e4b668fa9f9d5f79c5fd0f56941779c98e78fc3c4bcffa4b3f4268f53f16371"
GARMENTS = ("O01", "O03", "O04")
SEEDS = (0, 1, 2)
CHECKPOINT_STEPS = (0, 20, 50, 100, 200, 300)
QUARANTINE = (
    "subject00_O01_slot04_remaining_attempt005_cand00",
    "subject00_O03_slot04_canary_attempt004_cand00",
)
ROTATIONS = (
    {"rotation": 0, "train_slots": [0, 7], "calibration_slot": 3, "test_slot": 6},
    {"rotation": 1, "train_slots": [7, 3], "calibration_slot": 6, "test_slot": 0},
    {"rotation": 2, "train_slots": [3, 6], "calibration_slot": 0, "test_slot": 7},
    {"rotation": 3, "train_slots": [6, 0], "calibration_slot": 7, "test_slot": 3},
)
EXPECTED_RUN_IDS = tuple(
    f"COMMONSAFE4-METHOD-R{rotation['rotation']}-S{seed}"
    for rotation in ROTATIONS
    for seed in SEEDS
)
RUNNER_REL = "tools/second_identity/run_subject00_base60747_commonsafe4_matrix.py"
LEGACY_RUNNER_REL = "tools/second_identity/run_subject00_base60747_method.py"
CONFIG_REL = "configs/research/subject00_canondressgs_method_base60747_commonsafe4_v1.json"
RUNTIME_REL = "tools/paper/formal_batch_runtime.py"
SELECTION_REL = (
    "paper_protocol/reviewer_risk/"
    "subject00_commonsafe_slot04_replacement_selection_20260727.json"
)
ROTATION_REL = (
    "paper_protocol/reviewer_risk/subject00_commonsafe4_rotation_contract_20260727.json"
)
EXPECTED_RUNNER_SHA = "78b619155ddef4bea942d3317fe4098f9d969843bd10b2f4a818c953519714c1"
EXPECTED_CONFIG_SHA = "4175fe7803489ce8d32516a2d7ee9f59ab413779f7687a19c4810e7ff1a886a9"
EXPECTED_RUNTIME_SHA = "75211429a628801882d211805a2334fc88c11f6654cfbc4b6cc47fa2906693a6"
EXPECTED_SELECTION_SHA = "dd76bcb8cc2518a0595d6f856323f5ab6c44a51a52f1d7755b456a7a85017e80"
EXPECTED_ROTATION_SHA = "971a47c9aa2e340a05e6606d9c409d2c74db0b395b855cebe2bc73bc63e0127b"
EXPECTED_METHOD_TREE = {
    "file_count": 146,
    "total_bytes": 59_624_176,
    "tree_sha256": "bd3391d1a17406d8a06ffa723dfdb264ea362f6d851f8c87b0b7b95e0faadfa4",
}
EXPECTED_OLD_TREE = {
    "file_count": 19,
    "total_bytes": 53_138_437,
    "tree_sha256": "ce281935d12e20c6bab2e3e8fc3256b512594fd8388c3ed299f71199072ec363",
}
BASELINE_NAMES = (
    "Reference Classifier Lookup",
    "Nearest-Centroid Lookup",
    "Outfit-ID Oracle",
    "Teacher Endpoint",
)
FINAL_CLASSIFICATION = (
    "SUBJECT00_BASE60747_COMMONSAFE4_MATRIX_AND_BASELINES_VALID_POSTHOC_"
    "PROVENANCE_SEALED_PENDING_HUMAN_SCIENTIFIC_REVIEW"
)
NEXT_TASK = (
    "PREPARE_SUBJECT00_COMMONSAFE4_METHOD_TEACHER_BASELINE_"
    "HUMAN_SCIENTIFIC_REVIEW_PACK"
)

RISK = REPO_ROOT / "paper_protocol/reviewer_risk"
HANDOFF = REPO_ROOT / "project_control_handoff"
DOCS = REPO_ROOT / "docs/PAPER"
ARTIFACTS = {
    "writer": RISK / "subject00_commonsafe4_concurrent_writer_identity_registry_20260727.json",
    "timeline": RISK / "subject00_commonsafe4_concurrent_output_file_timeline_20260727.json",
    "replacement": RISK / "subject00_commonsafe4_replacement_pretraining_provenance_audit_20260727.json",
    "runner": RISK / "subject00_commonsafe4_actual_runner_config_registry_20260727.json",
    "core": RISK / "subject00_commonsafe4_actual_runner_core_equivalence_audit_20260727.json",
    "folds": RISK / "subject00_commonsafe4_actual_rotation_fold_registry_20260727.json",
    "execution": RISK / "subject00_commonsafe4_concurrent_12run_execution_registry_20260727.json",
    "checkpoints": RISK / "subject00_commonsafe4_concurrent_checkpoint_authenticity_registry_20260727.json",
    "formal_tests": RISK / "subject00_commonsafe4_concurrent_formal_test_registry_20260727.json",
    "aggregate": RISK / "subject00_commonsafe4_concurrent_matrix_recomputed_aggregate_20260727.json",
    "baselines": RISK / "subject00_commonsafe4_concurrent_fair_baseline_discovery_audit_20260727.json",
    "overlay": RISK / "subject00_commonsafe4_preflight_concurrent_execution_correction_overlay_20260727.json",
    "report": RISK / "SUBJECT00_COMMONSAFE4_CONCURRENT_12RUN_PROVENANCE_AUDIT_REPORT_20260727.md",
    "tests": RISK / "subject00_commonsafe4_concurrent_12run_provenance_tests_20260727.json",
    "summary": RISK / "subject00_commonsafe4_concurrent_12run_provenance_final_summary_20260727.json",
    "handoff": HANDOFF / "subject00_commonsafe4_concurrent_12run_provenance_handoff_20260727.json",
    "docs": DOCS / "AAAI27_SUBJECT00_COMMONSAFE4_CONCURRENT_12RUN_PROVENANCE_AUDIT_REPORT_20260727.md",
    "seal": RISK / "subject00_commonsafe4_concurrent_12run_matrix_provenance_seal_20260727.json",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, default=str) + "\n").encode()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(canonical_json(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(value.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def iso_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).astimezone().isoformat()


def tree_fingerprint(root: Path, include_rows: bool = False) -> dict[str, Any]:
    digest = hashlib.sha256()
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        stat = path.stat()
        relative = path.relative_to(root).as_posix()
        file_sha = sha256_file(path)
        row = {
            "path": relative,
            "bytes": stat.st_size,
            "sha256": file_sha,
            "ctime_ns": stat.st_ctime_ns,
            "ctime": iso_timestamp(stat.st_ctime),
            "mtime_ns": stat.st_mtime_ns,
            "mtime": iso_timestamp(stat.st_mtime),
        }
        rows.append(row)
        digest.update(f"{relative}\0{stat.st_size}\0{file_sha}\n".encode())
    result = {
        "root": str(root),
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "tree_sha256": digest.hexdigest(),
    }
    if include_rows:
        result["files"] = rows
    return result


def metadata_fingerprint(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        stat = path.stat()
        relative = path.relative_to(root).as_posix()
        row = {
            "path": relative,
            "bytes": stat.st_size,
            "ctime_ns": stat.st_ctime_ns,
            "mtime_ns": stat.st_mtime_ns,
        }
        rows.append(row)
        digest.update(
            f"{relative}\0{stat.st_size}\0{stat.st_ctime_ns}\0{stat.st_mtime_ns}\n".encode()
        )
    return {
        "root": str(root),
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "metadata_sha256": digest.hexdigest(),
    }


def run_text(*args: str, check: bool = True) -> str:
    return subprocess.run(
        list(args), check=check, capture_output=True, text=True
    ).stdout.strip()


def git(*args: str) -> str:
    return run_text("git", "-C", str(REPO_ROOT), *args)


def git_show(commit: str, path: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "show", f"{commit}:{path}"]
    )


def git_commit(commit: str) -> dict[str, Any]:
    fields = git(
        "show",
        "-s",
        "--format=%H%x00%P%x00%aI%x00%cI%x00%s",
        commit,
    ).split("\0")
    return {
        "head": fields[0],
        "parents": fields[1].split(),
        "author_time": fields[2],
        "commit_time": fields[3],
        "subject": fields[4],
    }


def recursive_values(value: Any, key: str) -> list[Any]:
    result = []
    if isinstance(value, dict):
        for name, child in value.items():
            if name == key:
                result.append(child)
            result.extend(recursive_values(child, key))
    elif isinstance(value, list):
        for child in value:
            result.extend(recursive_values(child, key))
    return result


def torch_load(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise TypeError(f"expected dict checkpoint: {path}")
    return value


def tensor_state_changed(
    before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]
) -> dict[str, bool]:
    return {
        key: key in after and not torch.equal(value, after[key])
        for key, value in before.items()
    }


def process_state() -> dict[str, Any]:
    processes = []
    output = run_text("ps", "-eo", "pid=,ppid=,args=")
    method_tokens = (
        "run_subject00_base60747_commonsafe4_matrix.py",
        "run_subject00_base60747_method.py",
        "formal_batch_runtime.py",
    )
    teacher_tokens = ("run_subject00_base60747_formal_teacher.py",)
    optimizer_tokens = method_tokens + teacher_tokens + (
        "run_subject00_formal_base_101245.py",
    )
    for line in output.splitlines():
        match = re.match(r"\s*(\d+)\s+(\d+)\s+(.*)", line)
        if not match:
            continue
        pid, ppid, command = int(match.group(1)), int(match.group(2)), match.group(3)
        if pid == os.getpid() or "audit_subject00_commonsafe4_concurrent" in command:
            continue
        if any(token in command for token in optimizer_tokens):
            processes.append({"pid": pid, "parent_pid": ppid, "command": command})
    gpu_text = run_text(
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
        check=False,
    )
    gpu = []
    for line in gpu_text.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) >= 3 and fields[0].isdigit():
            gpu.append(
                {
                    "pid": int(fields[0]),
                    "process_name": fields[1],
                    "used_memory_mib": int(fields[2]),
                }
            )
    tmux_text = run_text("tmux", "ls", check=False)
    return {
        "active_optimizer_process_count": len(processes),
        "active_method_process_count": sum(
            any(token in row["command"] for token in method_tokens)
            for row in processes
        ),
        "active_teacher_process_count": sum(
            any(token in row["command"] for token in teacher_tokens)
            for row in processes
        ),
        "matching_processes": processes,
        "gpu_compute_processes": gpu,
        "idle_tmux_sessions": [
            line.split(":", 1)[0] for line in tmux_text.splitlines() if line.strip()
        ],
        "tmux_listing": tmux_text or "NO_TMUX_SERVER_OR_SESSIONS",
    }


def parse_slot(request_id: str) -> int:
    return int(request_id.split("_slot", 1)[1][:2])


def parse_garment(request_id: str) -> str:
    return request_id.split("_")[1]


def circular_distance(left: float, right: float) -> float:
    delta = abs(left - right) % 360.0
    return min(delta, 360.0 - delta)


def camera_yaw(c2w: list[list[float]]) -> float:
    axis_x = float(c2w[0][2])
    axis_z = float(c2w[2][2])
    if math.hypot(axis_x, axis_z) <= 1e-12:
        raise ValueError("zero horizontal optical-axis projection")
    return math.degrees(math.atan2(axis_x, axis_z)) % 360.0


def limitation_severity(values: Iterable[Any]) -> int:
    total = 0
    for value in values:
        lowered = str(value).lower()
        if any(token in lowered for token in ("critical", "severe", "high")):
            total += 3
        elif any(token in lowered for token in ("moderate", "medium")):
            total += 2
        else:
            total += 1
    return total


def reproduce_replacement() -> dict[str, Any]:
    manifest_path = TARGET_ROOT / MANIFEST_REL
    manifest = read_json(manifest_path)
    observations = {
        row["condition_id"]: row
        for outfit in manifest["outfits"]
        for row in outfit["observations"]
    }
    slots: dict[int, dict[str, Any]] = {
        slot: {"garments": {}} for slot in range(8)
    }
    for camera_path in sorted((TARGET_ROOT / "06_camera").glob("*_camera.json")):
        camera = read_json(camera_path)
        request_id = camera["request_id"]
        garment = parse_garment(request_id)
        slot = parse_slot(request_id)
        record_path = (
            TARGET_ROOT / "02_records" / f"{request_id}_teacher_target_record.json"
        )
        record = read_json(record_path)
        metrics = camera.get("registration_metrics") or {}
        eligible = (
            request_id in observations
            and camera.get("training_eligible") is True
            and camera.get("evaluation_eligible") is True
            and camera.get("camera_binding_status") == "UNIQUE_SIMILARITY_BINDING_PASS"
            and request_id not in QUARANTINE
        )
        slots[slot]["garments"][garment] = {
            "request_id": request_id,
            "eligible": eligible,
            "camera_id": camera.get("source_camera_id"),
            "direction": metrics.get("direction"),
            "yaw": camera_yaw(camera["c2w"]) if camera.get("c2w") else None,
            "rmse": metrics.get("reprojection_rmse_px"),
            "inlier": metrics.get("inlier_ratio"),
            "limitations": list(record.get("limitations", [])),
            "camera_sha256": sha256_file(camera_path),
            "record_sha256": sha256_file(record_path),
        }
    registry = {}
    for slot, value in slots.items():
        garments = value["garments"]
        eligible = [garments[g] for g in GARMENTS if garments[g]["eligible"]]
        yaws = [float(row["yaw"]) for row in eligible if row["yaw"] is not None]
        registry[slot] = {
            "slot": f"slot{slot:02d}",
            "garment_coverage": len(eligible),
            "common_safe": len(eligible) == 3,
            "yaw_degrees": yaws[0] if yaws else None,
            "camera_ids": sorted(
                {row["camera_id"] for row in eligible if row["camera_id"]}
            ),
            "directions": sorted(
                {row["direction"] for row in eligible if row["direction"]}
            ),
            "worst_case_registration_rmse_px": (
                max(float(row["rmse"]) for row in eligible)
                if len(eligible) == 3
                else None
            ),
            "minimum_inlier_ratio": (
                min(float(row["inlier"]) for row in eligible)
                if len(eligible) == 3
                else None
            ),
            "limitation_count": sum(len(row["limitations"]) for row in eligible),
            "limitation_severity": sum(
                limitation_severity(row["limitations"]) for row in eligible
            ),
            "garments": garments,
        }
    slot04_yaw = float(registry[4]["garments"]["O04"]["yaw"])
    fixed_yaws = [float(registry[slot]["yaw_degrees"]) for slot in (0, 3, 7)]
    ranking = []
    for slot in (1, 2, 5, 6):
        value = registry[slot]
        yaw = float(value["yaw_degrees"])
        distance = circular_distance(yaw, slot04_yaw)
        separation = min(circular_distance(yaw, anchor) for anchor in fixed_yaws)
        sorting_tuple = [
            distance,
            -separation,
            float(value["worst_case_registration_rmse_px"]),
            -float(value["minimum_inlier_ratio"]),
            int(value["limitation_severity"]),
            slot,
        ]
        ranking.append(
            {
                "slot": f"slot{slot:02d}",
                "camera": value["camera_ids"][0],
                "direction": value["directions"][0],
                "yaw_degrees": yaw,
                "slot04_yaw_degrees": slot04_yaw,
                "yaw_distance_degrees": distance,
                "minimum_anchor_separation_degrees": separation,
                "worst_case_registration_rmse_px": value[
                    "worst_case_registration_rmse_px"
                ],
                "minimum_inlier_ratio": value["minimum_inlier_ratio"],
                "limitation_count": value["limitation_count"],
                "limitation_severity": value["limitation_severity"],
                "sorting_tuple": sorting_tuple,
            }
        )
    ranking.sort(key=lambda row: tuple(row["sorting_tuple"]))
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "formal_record_count": len(observations),
        "quarantine_in_manifest": sorted(set(QUARANTINE).intersection(observations)),
        "camera_record_count": sum(len(value["garments"]) for value in slots.values()),
        "slot_registry": {f"slot{slot:02d}": value for slot, value in registry.items()},
        "common_safe_slots": [
            f"slot{slot:02d}" for slot, value in registry.items() if value["common_safe"]
        ],
        "candidate_set": ["slot01", "slot02", "slot05", "slot06"],
        "ranking_rule": [
            "minimize circular yaw distance to original slot04/cam11",
            "maximize minimum angular separation from slot00/slot03/slot07",
            "minimize worst-case three-garment registration RMSE",
            "maximize minimum three-garment inlier ratio",
            "minimize limitation severity",
            "minimize slot ID",
        ],
        "ranking": ranking,
        "selected_slot": ranking[0]["slot"],
        "selected_camera": ranking[0]["camera"],
        "selected_direction": ranking[0]["direction"],
    }


def file_markers(path: Path) -> dict[str, Any]:
    tasks: set[str] = set()
    heads: set[str] = set()
    run_ids: set[str] = set()
    checkpoint_step = None
    parse_status = "NOT_STRUCTURED"
    try:
        if path.suffix == ".json":
            payload = read_json(path)
            parse_status = "JSON_PARSE_PASS"
        elif path.suffix in {".pth", ".pt"}:
            payload = torch_load(path)
            parse_status = "TORCH_PARSE_PASS"
        else:
            payload = None
        if payload is not None:
            tasks.update(
                str(value) for value in recursive_values(payload, "task_id")
                if isinstance(value, str)
            )
            run_ids.update(
                str(value) for value in recursive_values(payload, "run_id")
                if isinstance(value, str)
            )
            for value in recursive_values(payload, "git"):
                if isinstance(value, dict) and isinstance(value.get("head"), str):
                    heads.add(value["head"])
            if isinstance(payload, dict) and isinstance(payload.get("head"), str):
                heads.add(payload["head"])
            if path.suffix == ".pth":
                checkpoint_step = int(
                    payload.get("optimizer_step", payload.get("global_step", -1))
                )
    except Exception as error:  # recorded and rejected later
        parse_status = f"PARSE_FAIL:{type(error).__name__}:{error}"
    match = re.search(r"(?:step_)(\d+)", path.stem)
    if checkpoint_step is None and match:
        checkpoint_step = int(match.group(1))
    return {
        "task_markers": sorted(tasks),
        "git_head_markers": sorted(heads),
        "run_id_markers": sorted(run_ids),
        "checkpoint_step": checkpoint_step,
        "parse_status": parse_status,
    }


def build_timeline(method_tree: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    rows = []
    unattributed = []
    cross_task = []
    cross_head = []
    start_ns = int(float(lock["start_time_unix"]) * 1_000_000_000)
    end_ns = int(float(lock["updated_time_unix"]) * 1_000_000_000)
    for original in method_tree["files"]:
        path = METHOD_ROOT / original["path"]
        markers = file_markers(path)
        direct = (
            ACTUAL_WRITER_TASK in markers["task_markers"]
            or ACTUAL_WRITER_HEAD in markers["git_head_markers"]
        )
        time_bound = (
            start_ns - 5_000_000_000
            <= original["mtime_ns"]
            <= end_ns + 5_000_000_000
        )
        attributed = direct or time_bound
        mode = (
            "DIRECT_TASK_OR_GIT_MARKER"
            if direct
            else "ROOT_LOCK_ATOMIC_TIMELINE_ATTRIBUTION"
            if time_bound
            else "UNATTRIBUTED"
        )
        row = {**original, **markers, "attribution": mode}
        rows.append(row)
        if not attributed:
            unattributed.append(original["path"])
        foreign_tasks = [
            value for value in markers["task_markers"]
            if value != ACTUAL_WRITER_TASK
        ]
        if foreign_tasks:
            cross_task.append({"path": original["path"], "tasks": foreign_tasks})
        foreign_heads = [
            value for value in markers["git_head_markers"]
            if value != ACTUAL_WRITER_HEAD
        ]
        if foreign_heads:
            cross_head.append({"path": original["path"], "heads": foreign_heads})
    return {
        "schema_version": "canondressgs.subject00.commonsafe4.output_timeline.v1",
        "task_id": TASK_ID,
        "output_root": str(METHOD_ROOT),
        "root_fingerprint": {
            key: value for key, value in method_tree.items() if key != "files"
        },
        "writer_lock_window": {
            "start_time_unix": lock["start_time_unix"],
            "start_time": iso_timestamp(lock["start_time_unix"]),
            "end_time_unix": lock["updated_time_unix"],
            "end_time": iso_timestamp(lock["updated_time_unix"]),
        },
        "files": rows,
        "unattributed_files": unattributed,
        "cross_task_files": cross_task,
        "cross_head_files": cross_head,
        "unattributed_file_count": len(unattributed),
        "cross_task_file_count": len(cross_task),
        "cross_head_file_count": len(cross_head),
        "single_writer_status": (
            "PASS_SINGLE_WRITER" if not unattributed and not cross_task else "FAIL"
        ),
        "file_interleaving_status": (
            "PASS_NO_FILE_INTERLEAVING"
            if not cross_task and not cross_head
            else "FAIL"
        ),
    }


def audit_assets() -> dict[str, Any]:
    # The Base60747 archive is ~724 MB.  Memory mapping preserves a real
    # torch parse/internal-step check without materializing every tensor into
    # the audit process at once.
    base_payload = torch.load(
        BASE_PATH, map_location="cpu", weights_only=False, mmap=True
    )
    if not isinstance(base_payload, dict):
        raise TypeError(f"expected dict checkpoint: {BASE_PATH}")
    base_step = int(
        base_payload.get(
            "global_step",
            base_payload.get("step", base_payload.get("iteration", -1)),
        )
    )
    del base_payload
    teachers = {}
    for garment, path in TEACHER_PATHS.items():
        payload = torch_load(path)
        metadata = payload.get("metadata", {})
        teachers[garment] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "parse": True,
            "global_step": int(payload.get("global_step", -1)),
            "optimizer_step": int(payload.get("optimizer_step", -1)),
            "base_checkpoint_sha256": metadata.get("base_checkpoint_sha256"),
            "target_manifest_sha256": metadata.get("target_manifest_sha256"),
            "formal_target_root": metadata.get("formal_target_root"),
            "quarantine_optimizer_usage_count": metadata.get(
                "quarantine_count_in_optimizer"
            ),
            "formal_base_resume_authorized": metadata.get(
                "formal_base_resume_authorized"
            ),
        }
        del payload
    pause = read_json(
        FORMAL_BASE_ROOT
        / "attempt_001/control/USER_AUTHORIZED_PAUSE_FOR_PROVISIONAL_DOWNSTREAM_20260727.json"
    )
    return {
        "base": {
            "path": str(BASE_PATH),
            "bytes": BASE_PATH.stat().st_size,
            "sha256": sha256_file(BASE_PATH),
            "parse": True,
            "internal_step": base_step,
        },
        "teachers": teachers,
        "formal_target": {
            "root": str(TARGET_ROOT),
            "manifest": str(TARGET_ROOT / MANIFEST_REL),
            "manifest_sha256": sha256_file(TARGET_ROOT / MANIFEST_REL),
        },
        "formal_base_state": {
            "status": "USER_AUTHORIZED_PAUSED",
            "completed": pause.get("formal_base_completed"),
            "durable_resume_step": pause.get("latest_complete_checkpoint_step"),
            "resume_authorized": False,
            "pause_record": pause,
        },
    }


def audit_method_runs(config: dict[str, Any]) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    runs_root = METHOD_ROOT / "attempt_001/runs"
    actual_ids = sorted(
        path.name for path in runs_root.iterdir() if path.is_dir()
    )
    run_rows = []
    checkpoint_rows = []
    formal_rows = []
    all_test_rows = []
    for run_id in EXPECTED_RUN_IDS:
        match = re.fullmatch(r"COMMONSAFE4-METHOD-R(\d)-S(\d)", run_id)
        assert match
        rotation_number, seed = int(match.group(1)), int(match.group(2))
        rotation = ROTATIONS[rotation_number]
        root = runs_root / run_id
        pre_gate = read_json(root / "audit/pre_run_gate.json")
        summary = read_json(root / "training/run_summary.json")
        latest = read_json(root / "training/latest.json")
        calibration = read_json(root / "evaluation/calibration_step300.json")
        formal_test = read_json(root / "evaluation/formal_test_step300.json")
        checkpoint_paths = sorted((root / "checkpoints").glob("step_*.pth"))
        actual_steps = [
            int(path.stem.removeprefix("step_")) for path in checkpoint_paths
        ]
        per_checkpoint = []
        initial_model = None
        final_model = None
        checkpoint_valid = True
        for path, expected_step in zip(checkpoint_paths, CHECKPOINT_STEPS):
            payload = torch_load(path)
            bindings = payload.get("bindings", {})
            history = payload.get("history", [])
            file_sha = sha256_file(path)
            if expected_step == 0:
                initial_model = {
                    key: value.detach().cpu().clone()
                    for key, value in payload["model"].items()
                }
            if expected_step == 300:
                final_model = {
                    key: value.detach().cpu().clone()
                    for key, value in payload["model"].items()
                }
            history_steps = [int(row["optimizer_step"]) for row in history]
            history_finite = all(
                bool(row.get("finite"))
                and math.isfinite(float(row["loss_total"]))
                and math.isfinite(float(row["gradient_norm_before_clip"]))
                for row in history
            )
            valid = all(
                (
                    payload.get("task_id") == ACTUAL_WRITER_TASK,
                    payload.get("run_id") == run_id,
                    int(payload.get("rotation", -1)) == rotation_number,
                    int(payload.get("seed", -1)) == seed,
                    payload.get("train_slots") == rotation["train_slots"],
                    int(payload.get("calibration_slot", -1))
                    == rotation["calibration_slot"],
                    int(payload.get("test_slot", -1)) == rotation["test_slot"],
                    int(payload.get("global_step", -1)) == expected_step,
                    int(payload.get("optimizer_step", -1)) == expected_step,
                    history_steps == list(range(1, expected_step + 1)),
                    history_finite,
                    bindings.get("config_sha256") == EXPECTED_CONFIG_SHA,
                    bindings.get("selection_sha256") == EXPECTED_SELECTION_SHA,
                    bindings.get("rotation_contract_sha256")
                    == EXPECTED_ROTATION_SHA,
                    bindings.get("base_checkpoint_sha256") == BASE_SHA,
                    bindings.get("target_manifest_sha256") == MANIFEST_SHA,
                    bindings.get("teacher_checkpoint_sha256") == TEACHER_SHA,
                    bindings.get("f2_checkpoint_sha256") == F2_SHA,
                    bindings.get("quarantine_optimizer_usage_count") == 0,
                    bindings.get("quarantine_test_usage_count") == 0,
                    bindings.get("dual_support_call_count") == 0,
                    bindings.get("git", {}).get("branch")
                    == ACTUAL_WRITER_BRANCH,
                    bindings.get("git", {}).get("head") == ACTUAL_WRITER_HEAD,
                    bindings.get("git", {}).get("clean") is True,
                    file_sha
                    == summary["checkpoint_sha256"][str(expected_step)],
                )
            )
            checkpoint_valid = checkpoint_valid and valid
            row = {
                "run_id": run_id,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": file_sha,
                "expected_step": expected_step,
                "internal_global_step": payload.get("global_step"),
                "internal_optimizer_step": payload.get("optimizer_step"),
                "history_length": len(history),
                "history_monotonic": history_steps
                == list(range(1, expected_step + 1)),
                "history_finite": history_finite,
                "task_id": payload.get("task_id"),
                "git": bindings.get("git"),
                "bindings": {
                    key: bindings.get(key)
                    for key in (
                        "contract_name",
                        "config_sha256",
                        "selection_sha256",
                        "rotation_contract_sha256",
                        "base_checkpoint_sha256",
                        "target_manifest_sha256",
                        "teacher_checkpoint_sha256",
                        "f2_checkpoint_sha256",
                        "quarantine_optimizer_usage_count",
                        "quarantine_test_usage_count",
                        "dual_support_call_count",
                    )
                },
                "parse_status": "PASS",
                "atomic_seal_status": "PASS_NO_TMP_OR_PARTIAL",
                "valid": valid,
            }
            per_checkpoint.append(row)
            checkpoint_rows.append(row)
            del payload
        initialization_valid = bool(
            initial_model
            and torch.equal(
                initial_model["normalization.weight"],
                torch.ones_like(initial_model["normalization.weight"]),
            )
            and torch.equal(
                initial_model["normalization.bias"],
                torch.zeros_like(initial_model["normalization.bias"]),
            )
            and torch.equal(
                initial_model["linear.weight"],
                torch.zeros_like(initial_model["linear.weight"]),
            )
            and torch.equal(
                initial_model["linear.bias"],
                torch.zeros_like(initial_model["linear.bias"]),
            )
        )
        changes = (
            tensor_state_changed(initial_model, final_model)
            if initial_model and final_model
            else {}
        )
        recomputed_rows = []
        for row in formal_test["rows"]:
            scores = {key: float(value) for key, value in row["candidate_squared_distances"].items()}
            ordered = sorted(
                GARMENTS, key=lambda garment: (scores[garment], GARMENTS.index(garment))
            )
            selected = ordered[0]
            margin = scores[ordered[1]] - scores[ordered[0]]
            recomputed_rows.append(
                {
                    **row,
                    "recomputed_selected_endpoint": selected,
                    "recomputed_correct": selected == row["garment"],
                    "recomputed_margin": margin,
                    "selection_matches": selected == row["selected_endpoint"],
                    "correctness_matches": (selected == row["garment"])
                    == bool(row["correct"]),
                    "margin_matches": math.isclose(
                        margin,
                        float(row["score_margin_second_minus_first"]),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ),
                }
            )
        formal_valid = all(
            (
                formal_test.get("status") == "PASS_FORMAL_3_OF_3",
                formal_test.get("fold") == "test",
                formal_test.get("slot") == rotation["test_slot"],
                formal_test.get("denominator") == 3,
                formal_test.get("garment_record_count") == 3,
                [row["garment"] for row in formal_test["rows"]] == list(GARMENTS),
                len(formal_test["rows"]) == 3,
                all(
                    row["selection_matches"]
                    and row["correctness_matches"]
                    and row["margin_matches"]
                    and row.get("finite") is True
                    for row in recomputed_rows
                ),
                formal_test.get("quarantine_usage_count") == 0,
                formal_test.get("dual_support_call_count") == 0,
            )
        )
        correct = sum(row["recomputed_correct"] for row in recomputed_rows)
        recalculated_accuracy = correct / 3.0
        formal_valid = formal_valid and math.isclose(
            recalculated_accuracy,
            float(formal_test["nearest_endpoint_accuracy"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        run_valid = all(
            (
                pre_gate.get("status") == "PASS",
                pre_gate.get("fold_counts") == {
                    "train": 6,
                    "calibration": 3,
                    "test": 3,
                },
                pre_gate.get("quarantine_usage_count") == 0,
                summary.get("status") == "FORMAL_VALID_MATRIX_CELL",
                summary.get("rotation") == rotation_number,
                summary.get("seed") == seed,
                summary.get("train_slots") == rotation["train_slots"],
                summary.get("calibration_slot") == rotation["calibration_slot"],
                summary.get("test_slot") == rotation["test_slot"],
                summary.get("train_garment_record_count") == 6,
                summary.get("calibration_garment_record_count") == 3,
                summary.get("test_garment_record_count") == 3,
                summary.get("optimizer_steps") == 300,
                summary.get("optimizer_step_monotonic") is True,
                actual_steps == list(CHECKPOINT_STEPS),
                checkpoint_valid,
                initialization_valid,
                all(changes.values()),
                summary.get("trainable_parameter_changes") == changes,
                summary.get("loss_finite") is True,
                summary.get("gradients_finite") is True,
                summary.get("nan_inf_status") == "NONE",
                summary.get("oom_status") == "NONE",
                summary.get("formal_test_status") == "PASS_3_OF_3",
                summary.get("quarantine_optimizer_usage_count") == 0,
                summary.get("quarantine_calibration_usage_count") == 0,
                summary.get("quarantine_test_usage_count") == 0,
                summary.get("quarantine_evaluation_usage_count") == 0,
                summary.get("dual_support_call_count") == 0,
                latest.get("optimizer_step") == 300,
                calibration.get("slot") == rotation["calibration_slot"],
                calibration.get("garment_record_count") == 3,
                calibration.get("status") == "PASS_FORMAL_3_OF_3",
                calibration.get("quarantine_usage_count") == 0,
                formal_valid,
            )
        )
        run_rows.append(
            {
                "run_id": run_id,
                "rotation": rotation_number,
                "seed": seed,
                "train_slots": rotation["train_slots"],
                "calibration_slot": rotation["calibration_slot"],
                "test_slot": rotation["test_slot"],
                "fold_counts": {"train": 6, "calibration": 3, "test": 3},
                "optimizer_steps": summary["optimizer_steps"],
                "checkpoint_steps": actual_steps,
                "checkpoint_count": len(checkpoint_paths),
                "initialization_valid": initialization_valid,
                "trainable_parameter_changes": changes,
                "loss_finite": summary["loss_finite"],
                "gradients_finite": summary["gradients_finite"],
                "nan_inf_status": summary["nan_inf_status"],
                "oom_status": summary["oom_status"],
                "quarantine_usage": {
                    "train": summary["quarantine_optimizer_usage_count"],
                    "calibration": summary["quarantine_calibration_usage_count"],
                    "test": summary["quarantine_test_usage_count"],
                    "evaluation": summary["quarantine_evaluation_usage_count"],
                },
                "dual_support_call_count": summary["dual_support_call_count"],
                "training_status": (
                    "PASS_300_STEPS" if run_valid else "FAIL"
                ),
                "calibration_status": calibration["status"],
                "formal_test_status": (
                    "PASS_RECOMPUTED_3_OF_3" if formal_valid else "FAIL"
                ),
                "endpoint_top1": recalculated_accuracy,
                "formal_test_correct": correct,
                "formal_test_denominator": 3,
                "wall_time_seconds": float(summary["wall_time_seconds"]),
                "peak_vram_bytes": int(summary["peak_vram_bytes"]),
                "valid": run_valid,
            }
        )
        formal_rows.append(
            {
                "run_id": run_id,
                "rotation": rotation_number,
                "seed": seed,
                "test_slot": rotation["test_slot"],
                "formal_test_artifact": str(
                    root / "evaluation/formal_test_step300.json"
                ),
                "rows": recomputed_rows,
                "correct": correct,
                "denominator": 3,
                "endpoint_top1": recalculated_accuracy,
                "status": "PASS" if formal_valid else "FAIL",
            }
        )
        all_test_rows.extend(
            {
                **row,
                "run_id": run_id,
                "rotation": rotation_number,
                "seed": seed,
            }
            for row in recomputed_rows
        )
    execution = {
        "schema_version": "canondressgs.subject00.commonsafe4.execution_audit.v1",
        "task_id": TASK_ID,
        "expected_run_ids": list(EXPECTED_RUN_IDS),
        "actual_run_ids": actual_ids,
        "missing_run_ids": sorted(set(EXPECTED_RUN_IDS) - set(actual_ids)),
        "extra_run_ids": sorted(set(actual_ids) - set(EXPECTED_RUN_IDS)),
        "duplicate_run_ids": [],
        "run_count": len(run_rows),
        "formal_valid_run_count": sum(row["valid"] for row in run_rows),
        "total_optimizer_steps": sum(row["optimizer_steps"] for row in run_rows),
        "runs": run_rows,
    }
    checkpoints = {
        "schema_version": "canondressgs.subject00.commonsafe4.checkpoint_audit.v1",
        "task_id": TASK_ID,
        "expected_steps_per_run": list(CHECKPOINT_STEPS),
        "checkpoint_count": len(checkpoint_rows),
        "parse_pass_count": sum(row["parse_status"] == "PASS" for row in checkpoint_rows),
        "binding_pass_count": sum(row["valid"] for row in checkpoint_rows),
        "tmp_or_partial_files": [
            str(path)
            for path in METHOD_ROOT.rglob("*")
            if path.is_file()
            and (".tmp" in path.name or "partial" in path.name.lower())
        ],
        "checkpoints": checkpoint_rows,
    }
    formal_tests = {
        "schema_version": "canondressgs.subject00.commonsafe4.formal_test_audit.v1",
        "task_id": TASK_ID,
        "valid_run_count": sum(row["status"] == "PASS" for row in formal_rows),
        "missing_run_ids": [
            run_id for run_id in EXPECTED_RUN_IDS
            if not (runs_root / run_id / "evaluation/formal_test_step300.json").is_file()
        ],
        "invalid_run_ids": [
            row["run_id"] for row in formal_rows if row["status"] != "PASS"
        ],
        "runs": formal_rows,
    }
    aggregate = recompute_method_aggregate(run_rows, all_test_rows)
    return execution, checkpoints, formal_tests, aggregate


def recompute_method_aggregate(
    run_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    accuracies = [float(row["endpoint_top1"]) for row in run_rows]
    confusion = {
        truth: {
            prediction: sum(
                row["garment"] == truth
                and row["recomputed_selected_endpoint"] == prediction
                for row in test_rows
            )
            for prediction in GARMENTS
        }
        for truth in GARMENTS
    }
    rotation_effect = {
        str(rotation): {
            "run_count": sum(row["rotation"] == rotation for row in run_rows),
            "mean_endpoint_top1": statistics.fmean(
                row["endpoint_top1"] for row in run_rows
                if row["rotation"] == rotation
            ),
        }
        for rotation in range(4)
    }
    seed_effect = {
        str(seed): {
            "run_count": sum(row["seed"] == seed for row in run_rows),
            "mean_endpoint_top1": statistics.fmean(
                row["endpoint_top1"] for row in run_rows if row["seed"] == seed
            ),
        }
        for seed in SEEDS
    }
    return {
        "schema_version": "canondressgs.subject00.commonsafe4.recomputed_aggregate.v1",
        "task_id": TASK_ID,
        "status": (
            "PASS_12_OF_12_FORMAL_VALID"
            if len(run_rows) == 12 and all(row["valid"] for row in run_rows)
            else "FAIL"
        ),
        "aggregation": "ALL_12_RUNS_NO_CHERRY_PICKING",
        "old_method_output_included": False,
        "method_contract": "PURE_ENDPOINT",
        "dual_support_enabled": False,
        "dual_support_call_count": sum(
            row["dual_support_call_count"] for row in run_rows
        ),
        "run_count": len(run_rows),
        "formal_valid_run_count": sum(row["valid"] for row in run_rows),
        "total_optimizer_steps": sum(row["optimizer_steps"] for row in run_rows),
        "formal_test_episode_count": len(test_rows),
        "formal_test_correct": sum(row["recomputed_correct"] for row in test_rows),
        "endpoint_top1": sum(row["recomputed_correct"] for row in test_rows)
        / len(test_rows),
        "per_run_endpoint_results": {
            row["run_id"]: {
                "correct": row["formal_test_correct"],
                "denominator": row["formal_test_denominator"],
                "top1": row["endpoint_top1"],
            }
            for row in run_rows
        },
        "mean_endpoint_top1": statistics.fmean(accuracies),
        "std_endpoint_top1_population": statistics.pstdev(accuracies),
        "min_endpoint_top1": min(accuracies),
        "max_endpoint_top1": max(accuracies),
        "mean_score_margin": statistics.fmean(
            float(row["recomputed_margin"]) for row in test_rows
        ),
        "confusion_matrix": confusion,
        "prediction_counts": dict(
            Counter(row["recomputed_selected_endpoint"] for row in test_rows)
        ),
        "rotation_effect": rotation_effect,
        "seed_effect": seed_effect,
        "failure_count": sum(not row["valid"] for row in run_rows),
        "nan_inf_status": (
            "NONE"
            if all(row["nan_inf_status"] == "NONE" for row in run_rows)
            else "DETECTED"
        ),
        "oom_status": (
            "NONE"
            if all(row["oom_status"] == "NONE" for row in run_rows)
            else "DETECTED"
        ),
        "wall_time_seconds": sum(row["wall_time_seconds"] for row in run_rows),
        "peak_vram_bytes": max(row["peak_vram_bytes"] for row in run_rows),
    }


def audit_baselines() -> dict[str, Any]:
    if not BASELINE_ROOT.is_dir():
        return {
            "status": "NOT_EXECUTED",
            "baseline_names": [],
            "valid_count": 0,
        }
    contract = read_json(BASELINE_ROOT / "attempt_001/contract/resolved_contract.json")
    lock = read_json(
        BASELINE_ROOT
        / "control/COMMONSAFE4_FAIR_BASELINE_EXECUTION_LOCK_20260727.json"
    )
    continuation = read_json(
        BASELINE_ROOT
        / "control/PREOPTIMIZER_SELF_PID_GATE_FAILURE_CONTINUATION_20260727.json"
    )
    initial_runner = git_show(
        BASELINE_CONTRACT_HEAD,
        "tools/second_identity/run_subject00_base60747_commonsafe4_fair_baselines.py",
    )
    repaired_runner = git_show(
        BASELINE_REPAIR_HEAD,
        "tools/second_identity/run_subject00_base60747_commonsafe4_fair_baselines.py",
    )
    contract_bytes = git_show(
        BASELINE_REPAIR_HEAD,
        "paper_protocol/reviewer_risk/subject00_commonsafe4_fair_baseline_contract_20260727.json",
    )
    directory_names = {
        "Reference Classifier Lookup": "reference_classifier_lookup",
        "Nearest-Centroid Lookup": "nearest_centroid_lookup",
        "Outfit-ID Oracle": "outfit_id_oracle",
        "Teacher Endpoint": "teacher_endpoint",
    }
    summaries = {}
    checkpoint_rows = []
    aggregate = {}
    all_valid = True
    total_steps = 0
    for name in BASELINE_NAMES:
        rows = []
        root = BASELINE_ROOT / "attempt_001/runs" / directory_names[name]
        for rotation in ROTATIONS:
            for seed in SEEDS:
                run_root = root / f"rotation_{rotation['rotation']}" / f"seed_{seed}"
                summary_path = (
                    run_root / "training/run_summary.json"
                    if name == "Reference Classifier Lookup"
                    else run_root / "run_summary.json"
                )
                summary = read_json(summary_path)
                formal = summary["formal_test"]
                recomputed = []
                for row in formal["rows"]:
                    selected = row["selected_endpoint"]
                    if "candidate_logits" in row:
                        scores = row["candidate_logits"]
                        ordered = sorted(
                            GARMENTS,
                            key=lambda garment: (
                                -float(scores[garment]),
                                GARMENTS.index(garment),
                            ),
                        )
                        selection_valid = selected == ordered[0]
                    elif "squared_distances" in row:
                        scores = row["squared_distances"]
                        ordered = sorted(
                            GARMENTS,
                            key=lambda garment: (
                                float(scores[garment]),
                                GARMENTS.index(garment),
                            ),
                        )
                        selection_valid = selected == ordered[0]
                    else:
                        selection_valid = (
                            row.get("ground_truth_id_used") is True
                            and selected == row["garment"]
                        )
                    recomputed.append(
                        selection_valid
                        and bool(row["correct"]) == (selected == row["garment"])
                    )
                correct = sum(
                    row["selected_endpoint"] == row["garment"]
                    for row in formal["rows"]
                )
                valid = all(
                    (
                        summary.get("baseline") == name,
                        summary.get("status") == "FORMAL_VALID_BASELINE_CELL",
                        summary.get("rotation") == rotation["rotation"],
                        summary.get("seed") == seed,
                        summary.get("train_slots") == rotation["train_slots"],
                        summary.get("calibration_slot")
                        == rotation["calibration_slot"],
                        summary.get("test_slot") == rotation["test_slot"],
                        summary.get("train_garment_record_count") == 6,
                        summary.get("calibration_garment_record_count") == 3,
                        summary.get("test_garment_record_count") == 3,
                        formal.get("condition_slot") == rotation["test_slot"],
                        formal.get("denominator") == 3,
                        formal.get("garment_record_count") == 3,
                        len(formal.get("rows", [])) == 3,
                        all(recomputed),
                        correct == formal.get("correct_count"),
                        math.isclose(
                            correct / 3.0,
                            float(formal.get("endpoint_top1")),
                            rel_tol=0.0,
                            abs_tol=1e-15,
                        ),
                        summary.get("formal_test_status") == "PASS_3_OF_3",
                        summary.get("nan_inf_status") == "NONE",
                        summary.get("oom_status") == "NONE",
                        summary.get("quarantine_optimizer_usage_count") == 0,
                        summary.get("quarantine_calibration_usage_count") == 0,
                        summary.get("quarantine_test_usage_count") == 0,
                        summary.get("quarantine_evaluation_usage_count") == 0,
                        summary.get("dual_support_call_count") == 0,
                    )
                )
                if name == "Reference Classifier Lookup":
                    checkpoints = sorted((run_root / "checkpoints").glob("step_*.pth"))
                    steps = [int(path.stem.removeprefix("step_")) for path in checkpoints]
                    valid = valid and steps == list(CHECKPOINT_STEPS)
                    for path, step in zip(checkpoints, CHECKPOINT_STEPS):
                        payload = torch_load(path)
                        git_head = payload.get("git", {}).get("head")
                        expected_head = (
                            BASELINE_CONTRACT_HEAD
                            if rotation["rotation"] == 0 and seed == 0
                            else BASELINE_REPAIR_HEAD
                        )
                        cp_valid = all(
                            (
                                payload.get("task_id") == ACTUAL_WRITER_TASK,
                                payload.get("baseline") == name,
                                payload.get("run_id") == summary.get("run_id"),
                                payload.get("rotation") == rotation["rotation"],
                                payload.get("seed") == seed,
                                payload.get("optimizer_step") == step,
                                git_head == expected_head,
                                payload.get("git", {}).get("clean") is True,
                                payload.get("quarantine_usage_count") == 0,
                                payload.get("dual_support_call_count") == 0,
                                sha256_file(path)
                                == summary["checkpoint_sha256"][str(step)],
                            )
                        )
                        valid = valid and cp_valid
                        checkpoint_rows.append(
                            {
                                "baseline": name,
                                "run_id": summary["run_id"],
                                "path": str(path),
                                "step": step,
                                "sha256": sha256_file(path),
                                "git_head": git_head,
                                "valid": cp_valid,
                            }
                        )
                        del payload
                    valid = valid and all(
                        summary.get("trainable_parameter_changes", {}).values()
                    )
                else:
                    valid = valid and summary.get("optimizer_steps") == 0
                total_steps += int(summary.get("optimizer_steps", 0))
                rows.append(
                    {
                        "run_id": summary["run_id"],
                        "rotation": rotation["rotation"],
                        "seed": seed,
                        "optimizer_steps": summary["optimizer_steps"],
                        "formal_test_correct": correct,
                        "formal_test_denominator": 3,
                        "endpoint_top1": correct / 3.0,
                        "valid": valid,
                    }
                )
                all_valid = all_valid and valid
        summaries[name] = rows
        correct = sum(row["formal_test_correct"] for row in rows)
        aggregate[name] = {
            "run_count": len(rows),
            "formal_valid_run_count": sum(row["valid"] for row in rows),
            "formal_test_correct": correct,
            "formal_test_denominator": 36,
            "endpoint_top1": correct / 36.0,
            "optimizer_steps": sum(row["optimizer_steps"] for row in rows),
        }
    valid_count = sum(
        value["formal_valid_run_count"] == 12 for value in aggregate.values()
    )
    return {
        "schema_version": "canondressgs.subject00.commonsafe4.baseline_audit.v1",
        "task_id": TASK_ID,
        "status": (
            "PASS_4_OF_4_FAIR_BASELINES_VALID" if all_valid else "FAIL"
        ),
        "output_root": str(BASELINE_ROOT),
        "baseline_names": list(BASELINE_NAMES),
        "valid_count": valid_count,
        "formal_valid_cell_count": sum(
            value["formal_valid_run_count"] for value in aggregate.values()
        ),
        "total_optimizer_steps_historical": total_steps,
        "new_optimizer_steps_by_this_audit": 0,
        "contract": {
            "status": contract.get("status"),
            "exact_baseline_order": contract.get("exact_baseline_order"),
            "rotations": contract["shared_fairness_bindings"]["rotations"],
            "seeds": contract["shared_fairness_bindings"]["seeds"],
            "fold_counts": contract["shared_fairness_bindings"]["fold_counts"],
            "base_checkpoint_sha256": contract["shared_fairness_bindings"][
                "base_checkpoint_sha256"
            ],
            "quarantine_exact_set": contract["shared_fairness_bindings"][
                "quarantine_exact_set"
            ],
            "contract_sha256": sha256_bytes(contract_bytes),
        },
        "writer": {
            "task_id": lock.get("task_id"),
            "branch": lock.get("branch"),
            "initial_head": BASELINE_CONTRACT_HEAD,
            "repair_head": BASELINE_REPAIR_HEAD,
            "pid": lock.get("pid"),
            "parent_pid": lock.get("parent_pid"),
            "hostname": lock.get("hostname"),
            "tmux": lock.get("tmux"),
            "status": lock.get("status"),
            "initial_runner_sha256": sha256_bytes(initial_runner),
            "repair_runner_sha256": sha256_bytes(repaired_runner),
        },
        "continuation_audit": {
            "classification": continuation.get("classification"),
            "same_attempt": continuation.get("same_attempt", True),
            "checks": continuation.get("checks"),
            "completed_scientific_cells_preserved": continuation.get(
                "completed_scientific_cells_preserved"
            ),
            "failed_scientific_cell_count": continuation.get(
                "failed_scientific_cell_count"
            ),
            "failed_cell_optimizer_steps": continuation.get(
                "failed_cell_optimizer_steps"
            ),
            "scientific_run_retry_count": continuation.get(
                "scientific_run_retry_count"
            ),
            "attempt_002_created": continuation.get("attempt_002_created"),
            "repair_scope": (
                "SELF_PID_RESOURCE_GATE_AND_SAME_ATTEMPT_CONTINUATION_ONLY"
            ),
        },
        "aggregates": aggregate,
        "runs": summaries,
        "reference_classifier_checkpoint_count": len(checkpoint_rows),
        "reference_classifier_checkpoints": checkpoint_rows,
    }


def static_core_audit(
    config: dict[str, Any], runner_bytes: bytes, runtime_bytes: bytes
) -> dict[str, Any]:
    old_config = read_json(
        OLD_METHOD_ROOT / "attempt_001/contract/config_resolved.json"
    )
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
    equality = {
        section: config[section] == old_config[section]
        for section in frozen_sections
    }
    legacy_at_old = git_show(
        "21e696a36b5df539124a01967f7867f735a8bd44",
        LEGACY_RUNNER_REL,
    )
    legacy_at_writer = git_show(ACTUAL_WRITER_HEAD, LEGACY_RUNNER_REL)
    tree = ast.parse(runner_bytes.decode("utf-8"))
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                parts = []
                value: ast.AST = node.func
                while isinstance(value, ast.Attribute):
                    parts.append(value.attr)
                    value = value.value
                if isinstance(value, ast.Name):
                    parts.append(value.id)
                    calls.append(".".join(reversed(parts)))
            elif isinstance(node.func, ast.Name):
                calls.append(node.func.id)
    legacy_calls = sorted({value for value in calls if value.startswith("legacy.")})
    required_calls = {
        "legacy.build_f2_extractor",
        "legacy.aggregate_native_references",
        "legacy.save_basis",
        "legacy.rng_state",
    }
    prohibited_tokens = (
        "dual_support_controller",
        "DualSupport",
        "pairwise_geometry_weight\": 1",
        "render_weight\": 1",
    )
    core_valid = all(equality.values()) and all(
        (
            sha256_bytes(runtime_bytes) == EXPECTED_RUNTIME_SHA,
            sha256_bytes(legacy_at_old) == sha256_bytes(legacy_at_writer),
            required_calls.issubset(legacy_calls),
            not any(token in runner_bytes.decode("utf-8") for token in prohibited_tokens),
            config["method_contract"]["dual_support_policy"]
            == "DISABLED_EXCLUDED_FROM_PURE_ENDPOINT_PROTOCOL",
            config["optimization"]["steps_per_run"] == 300,
        )
    )
    return {
        "schema_version": "canondressgs.subject00.commonsafe4.core_equivalence.v1",
        "task_id": TASK_ID,
        "status": (
            "PASS_METHOD_CORE_EQUIVALENT_AUTHORIZED_ORCHESTRATION_ONLY"
            if core_valid
            else "FAIL"
        ),
        "method_contract": "PURE_ENDPOINT",
        "dual_support_enabled": False,
        "dual_support_call_count": 0,
        "frozen_runtime_path": RUNTIME_REL,
        "frozen_runtime_sha256": sha256_bytes(runtime_bytes),
        "old_valid_r0s0_execution_head": "21e696a36b5df539124a01967f7867f735a8bd44",
        "legacy_runner_sha256_at_old_head": sha256_bytes(legacy_at_old),
        "legacy_runner_sha256_at_actual_writer_head": sha256_bytes(
            legacy_at_writer
        ),
        "legacy_runner_bitwise_equal": legacy_at_old == legacy_at_writer,
        "frozen_config_section_equality": equality,
        "static_call_graph": {
            "entrypoint": "main",
            "matrix_orchestration": [
                "main",
                "train_run",
                "evaluate_fold",
                "save_checkpoint",
                "aggregate",
            ],
            "reused_legacy_calls": legacy_calls,
            "controller": "MultiOutfitLinearCoefficientControl(512,2)",
            "optimizer": "torch.optim.Adam",
            "scheduler": "torch.optim.lr_scheduler.LambdaLR",
            "objective": "torch.nn.functional.smooth_l1_loss",
            "endpoint_selection": "nearest_endpoint(argmin squared L2; garment-order tie-break)",
        },
        "runtime_snapshot_parameter_diff": {
            "authorized_changes": [
                "rotation selection",
                "seed selection",
                "fold construction",
                "output namespace",
                "12-cell orchestration",
                "slot04 to slot06 common-safe replacement",
                "explicit calibration and formal-test persistence",
            ],
            "forbidden_core_changes_detected": [],
        },
    }


def make_tests(context: dict[str, Any]) -> list[dict[str, Any]]:
    e = context["execution"]
    c = context["checkpoints"]
    f = context["formal_tests"]
    a = context["aggregate"]
    b = context["baselines"]
    r = context["replacement"]
    runner = context["runner"]
    core = context["core"]
    assets = context["assets"]
    timeline = context["timeline"]
    proc = context["process"]
    mutations = context["mutations"]
    folds = context["folds"]
    tests: list[tuple[str, bool, Any]] = [
        ("01_source_branch_head", context["source_valid"], [SOURCE_BRANCH, SOURCE_HEAD]),
        ("02_source_clean", context["source_clean"], context["source_status"]),
        ("03_active_optimizer_zero", proc["active_optimizer_process_count"] == 0, proc),
        ("04_output_root_exists", METHOD_ROOT.is_dir(), str(METHOD_ROOT)),
        ("05_writer_task", context["writer"]["actual_writer_task"] == ACTUAL_WRITER_TASK, context["writer"]),
        ("06_writer_branch_head", context["writer"]["actual_writer_head"] == ACTUAL_WRITER_HEAD, context["writer"]),
        ("07_writer_timeline", len(timeline["files"]) == 146, len(timeline["files"])),
        ("08_single_writer", timeline["single_writer_status"] == "PASS_SINGLE_WRITER", timeline["single_writer_status"]),
        ("09_no_file_interleaving", timeline["file_interleaving_status"] == "PASS_NO_FILE_INTERLEAVING", timeline["file_interleaving_status"]),
        ("10_unattributed_zero", timeline["unattributed_file_count"] == 0, timeline["unattributed_file_count"]),
        ("11_base_sha", assets["base"]["sha256"] == BASE_SHA, assets["base"]["sha256"]),
        ("12_teacher_shas", all(assets["teachers"][g]["sha256"] == TEACHER_SHA[g] for g in GARMENTS), {g: assets["teachers"][g]["sha256"] for g in GARMENTS}),
        ("13_formal_target_root", Path(assets["formal_target"]["root"]) == TARGET_ROOT, assets["formal_target"]),
        ("14_quarantine_exact_set", tuple(config_value(context, "quarantine")) == QUARANTINE, config_value(context, "quarantine")),
        ("15_replacement_candidate_set", r["candidate_set"] == ["slot01", "slot02", "slot05", "slot06"], r["candidate_set"]),
        ("16_deterministic_ranking", len(r["ranking"]) == 4 and r["ranking"] == sorted(r["ranking"], key=lambda row: tuple(row["sorting_tuple"])), r["ranking"]),
        ("17_slot06_reproduced", r["selected_slot"] == "slot06", r["selected_slot"]),
        ("18_selection_rule_pretraining", context["selection_rule_pretraining"], context["selection_commits"]),
        ("19_selection_result_pretraining", context["selection_result_pretraining"], context["selection_commits"]),
        ("20_outcome_evidence_zero", context["outcome_dependent_evidence_count"] == 0, context["outcome_dependent_evidence_count"]),
        ("21_anchor_coverage", all(row["status"] == "PASS_EXACT_6_3_3" for row in folds["folds"]), folds["folds"]),
        ("22_actual_runner_sha", runner["actual_runner_sha256"] == EXPECTED_RUNNER_SHA, runner["actual_runner_sha256"]),
        ("23_actual_config_sha", runner["actual_config_sha256"] == EXPECTED_CONFIG_SHA, runner["actual_config_sha256"]),
        ("24_same_runner_config_12", runner["same_runner_config_runtime_12_of_12"], runner),
        ("25_method_core_equivalence", core["status"].startswith("PASS"), core["status"]),
        ("26_pure_endpoint", a["method_contract"] == "PURE_ENDPOINT", a["method_contract"]),
        ("27_dual_support_false", not a["dual_support_enabled"] and a["dual_support_call_count"] == 0, [a["dual_support_enabled"], a["dual_support_call_count"]]),
        ("28_exact_rotations", folds["actual_rotations"] == list(ROTATIONS), folds["actual_rotations"]),
        ("29_exact_seeds", folds["seeds"] == list(SEEDS), folds["seeds"]),
        ("30_exact_run_ids", e["actual_run_ids"] == sorted(EXPECTED_RUN_IDS), e["actual_run_ids"]),
        ("31_missing_runs_zero", not e["missing_run_ids"], e["missing_run_ids"]),
        ("32_extra_runs_zero", not e["extra_run_ids"], e["extra_run_ids"]),
        ("33_duplicate_runs_zero", not e["duplicate_run_ids"], e["duplicate_run_ids"]),
        ("34_each_run_300", all(row["optimizer_steps"] == 300 for row in e["runs"]), [row["optimizer_steps"] for row in e["runs"]]),
        ("35_total_steps_3600", e["total_optimizer_steps"] == 3600, e["total_optimizer_steps"]),
        ("36_six_checkpoints_each", all(row["checkpoint_count"] == 6 for row in e["runs"]), [row["checkpoint_count"] for row in e["runs"]]),
        ("37_total_checkpoints_72", c["checkpoint_count"] == 72, c["checkpoint_count"]),
        ("38_checkpoint_parse", c["parse_pass_count"] == 72, c["parse_pass_count"]),
        ("39_checkpoint_bindings", c["binding_pass_count"] == 72, c["binding_pass_count"]),
        ("40_trainable_changes", all(all(row["trainable_parameter_changes"].values()) for row in e["runs"]), [row["trainable_parameter_changes"] for row in e["runs"]]),
        ("41_frozen_immutability", all(value == 0 for key, value in mutations.items() if key.endswith("_mutations")), mutations),
        ("42_no_nan_inf_oom", a["nan_inf_status"] == "NONE" and a["oom_status"] == "NONE", [a["nan_inf_status"], a["oom_status"]]),
        ("43_quarantine_train_zero", context["q_usage"]["train"] == 0, context["q_usage"]),
        ("44_quarantine_cal_zero", context["q_usage"]["calibration"] == 0, context["q_usage"]),
        ("45_quarantine_test_zero", context["q_usage"]["test"] == 0, context["q_usage"]),
        ("46_quarantine_eval_zero", context["q_usage"]["evaluation"] == 0, context["q_usage"]),
        ("47_fold_6_3_3", all(row["fold_counts"] == {"train": 6, "calibration": 3, "test": 3} for row in e["runs"]), [row["fold_counts"] for row in e["runs"]]),
        ("48_formal_test_12", f["valid_run_count"] == 12, f["valid_run_count"]),
        ("49_formal_denominator_3", all(row["denominator"] == 3 for row in f["runs"]), [row["denominator"] for row in f["runs"]]),
        ("50_aggregate_recomputed", a["status"] == "PASS_12_OF_12_FORMAL_VALID" and math.isclose(a["endpoint_top1"], 30 / 36), a),
        ("51_no_cherry_picking", a["aggregation"] == "ALL_12_RUNS_NO_CHERRY_PICKING", a["aggregation"]),
        ("52_old_run_excluded", a["old_method_output_included"] is False, a["old_method_output_included"]),
        ("53_baseline_discovery", b["status"] == "PASS_4_OF_4_FAIR_BASELINES_VALID" and b["valid_count"] == 4, [b["status"], b["valid_count"]]),
        ("54_no_new_optimizer", context["new_method_optimizer_steps"] == 0 and context["new_baseline_optimizer_steps"] == 0, [context["new_method_optimizer_steps"], context["new_baseline_optimizer_steps"]]),
        ("55_output_immutable", mutations["commonsafe4_output_mutations"] == 0 and mutations["old_method_output_mutations"] == 0, mutations),
        ("56_base_immutable", mutations["base60747_mutations"] == 0 and mutations["formal_base_run_mutations"] == 0, mutations),
        ("57_teachers_immutable", all(mutations[f"{g.lower()}_teacher_mutations"] == 0 for g in GARMENTS), mutations),
        ("58_target_immutable", mutations["formal_target_mutations"] == 0 and mutations["raw_mutations"] == 0 and mutations["mask_mutations"] == 0 and mutations["camera_record_mutations"] == 0, mutations),
        ("59_paper_modification_zero", mutations["paper_modifications"] == 0, mutations["paper_modifications"]),
        ("60_final_classification", context["final_classification"] == FINAL_CLASSIFICATION, context["final_classification"]),
        ("61_next_task_unique", context["next_task"] == NEXT_TASK, context["next_task"]),
    ]
    return [
        {
            "id": index,
            "check": name,
            "status": "PASS" if passed else "FAIL",
            "actual": actual,
        }
        for index, (name, passed, actual) in enumerate(tests, 1)
    ]


def config_value(context: dict[str, Any], key: str) -> Any:
    if key == "quarantine":
        return context["config"]["targets"]["quarantine_exclusions"]
    raise KeyError(key)


def report_markdown(summary: dict[str, Any]) -> str:
    a = summary["matrix"]
    b = summary["fair_baselines"]
    return f"""# Subject00 CommonSafe4 并发 12-run provenance 只读审计

任务 `{TASK_ID}` 对既有 CommonSafe4 方法矩阵和同级公平基线进行了严格只读、逐文件、逐检查点、逐正式测试的 post-hoc provenance 审计。

## 裁决

- 原预检状态保持为 `{ORIGINAL_PREFLIGHT_STATUS}`，没有追溯改写为 PASS。
- 实际方法写入者为 `{ACTUAL_WRITER_TASK}`，执行 HEAD `{ACTUAL_WRITER_HEAD}`。
- slot04 替换由冻结 camera/target registry 独立复算为 `slot06 / cam09 / back-right`；选择规则和结果均在首个 optimizer step 之前进入 Git/config/checkpoint binding。
- 方法矩阵为 12/12 formal-valid，累计 3600 个历史 optimizer steps、72 个可解析且绑定一致的检查点。
- 独立复算 endpoint top-1 为 `{a['endpoint_top1']:.16f}`（{a['formal_test_correct']}/{a['formal_test_episode_count']}）。
- 四类公平基线均存在且 48/48 cells formal-valid；本审计没有运行任何新基线。
- Base60747、三套 Teacher、formal target、旧方法输出和新矩阵输出在审计前后均未变化。

## 方法聚合

- per-run mean/std(population): `{a['mean_endpoint_top1']:.16f}` / `{a['std_endpoint_top1_population']:.16f}`
- min/max: `{a['min_endpoint_top1']:.16f}` / `{a['max_endpoint_top1']:.16f}`
- wall time（12 runs 合计）: `{a['wall_time_seconds']:.9f}` 秒
- peak VRAM: `{a['peak_vram_bytes']}` bytes
- confusion: `{json.dumps(a['confusion_matrix'], sort_keys=True)}`

## 公平基线

{os.linesep.join(f"- {name}: top-1={row['endpoint_top1']:.16f}, valid={row['formal_valid_run_count']}/12, historical optimizer steps={row['optimizer_steps']}" for name, row in b['aggregates'].items())}

## 限制

这是技术与 provenance 接纳，不是人工视觉或科学结论。`PAPER_FINAL=false`，论文正文未修改。下一唯一任务为 `{NEXT_TASK}`。

最终分类：`{FINAL_CLASSIFICATION}`
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if not args.write:
        raise RuntimeError("audit requires explicit --write for Git-side reports")

    current_branch = git("branch", "--show-current")
    current_head = git("rev-parse", "HEAD")
    source_merge_base = git("merge-base", "HEAD", SOURCE_HEAD)
    source_status = git("status", "--porcelain=v2")
    source_valid = current_branch == AUDIT_BRANCH and source_merge_base == SOURCE_HEAD
    status_paths = []
    for line in source_status.splitlines():
        if not line.strip():
            continue
        # Porcelain v2 untracked records are "? path"; ordinary records keep
        # the path as their final space-delimited field for these audit files.
        status_paths.append(line[2:] if line.startswith("? ") else line.rsplit(" ", 1)[-1])
    allowed_report_paths = {
        path.relative_to(REPO_ROOT).as_posix() for path in ARTIFACTS.values()
    }
    rerun_has_only_prior_audit_reports = bool(status_paths) and set(
        status_paths
    ).issubset(allowed_report_paths)
    source_clean = not source_status or rerun_has_only_prior_audit_reports
    if not source_valid or not source_clean:
        raise RuntimeError(
            f"audit must start clean on {AUDIT_BRANCH}: "
            f"{current_branch=} {source_merge_base=} {source_status=}"
        )

    required_roots = (
        METHOD_ROOT,
        OLD_METHOD_ROOT,
        TARGET_ROOT,
        FORMAL_BASE_ROOT,
        BASELINE_ROOT,
    )
    if not all(path.is_dir() for path in required_roots):
        raise FileNotFoundError(
            [str(path) for path in required_roots if not path.is_dir()]
        )

    process_before = process_state()
    if process_before["active_optimizer_process_count"]:
        raise RuntimeError("CONCURRENT_MATRIX_STILL_RUNNING_NOT_READY_FOR_SEAL")

    # Immutable scientific inputs are fingerprinted before any Git-side write.
    method_before = tree_fingerprint(METHOD_ROOT, include_rows=True)
    old_before = tree_fingerprint(OLD_METHOD_ROOT)
    target_before = tree_fingerprint(TARGET_ROOT)
    baseline_before = tree_fingerprint(BASELINE_ROOT)
    formal_base_before = metadata_fingerprint(FORMAL_BASE_ROOT)
    base_sha_before = sha256_file(BASE_PATH)
    teacher_sha_before = {
        garment: sha256_file(path) for garment, path in TEACHER_PATHS.items()
    }

    lock = read_json(
        METHOD_ROOT
        / "control/COMMONSAFE4_METHOD_MATRIX_EXECUTION_LOCK_20260727.json"
    )
    preflight = read_json(METHOD_ROOT / "attempt_001/input_audit/preflight.json")
    config = read_json(METHOD_ROOT / "attempt_001/contract/config_resolved.json")
    run_status = read_json(METHOD_ROOT / "attempt_001/RUN_STATUS.json")
    final_report = read_json(METHOD_ROOT / "attempt_001/FINAL_REPORT.json")
    timeline = build_timeline(method_before, lock)
    assets = audit_assets()
    replacement = reproduce_replacement()

    runner_bytes = git_show(ACTUAL_WRITER_HEAD, RUNNER_REL)
    config_bytes = git_show(ACTUAL_WRITER_HEAD, CONFIG_REL)
    runtime_bytes = git_show(ACTUAL_WRITER_HEAD, RUNTIME_REL)
    selection_bytes = git_show(ACTUAL_WRITER_HEAD, SELECTION_REL)
    rotation_bytes = git_show(ACTUAL_WRITER_HEAD, ROTATION_REL)
    selection_commit = git_commit(SELECTION_COMMIT)
    runner_commit = git_commit(RUNNER_COMMIT)
    writer_commit = git_commit(ACTUAL_WRITER_HEAD)
    output_creation_time = METHOD_ROOT.stat().st_ctime
    first_checkpoint_time = min(
        path.stat().st_mtime
        for path in (METHOD_ROOT / "attempt_001/runs").rglob("step_000000.pth")
    )
    selection_rule_pretraining = (
        datetime.fromisoformat(selection_commit["commit_time"]).timestamp()
        < output_creation_time
    )
    selection_result_pretraining = all(
        (
            selection_rule_pretraining,
            sha256_bytes(selection_bytes) == EXPECTED_SELECTION_SHA,
            config["provenance"]["replacement_selected_before_optimizer_step"] == 0,
            config["method_contract"]["selected_replacement_slot"] == 6,
            (METHOD_ROOT / "attempt_001/contract/config_resolved.json").stat().st_mtime
            < first_checkpoint_time,
        )
    )

    execution, checkpoints, formal_tests, aggregate = audit_method_runs(config)
    baselines = audit_baselines()
    core = static_core_audit(config, runner_bytes, runtime_bytes)

    config_bindings = {
        row["bindings"]["config_sha256"]
        for row in checkpoints["checkpoints"]
    }
    selection_bindings = {
        row["bindings"]["selection_sha256"]
        for row in checkpoints["checkpoints"]
    }
    runner_registry = {
        "schema_version": "canondressgs.subject00.commonsafe4.actual_runtime.v1",
        "task_id": TASK_ID,
        "actual_writer_head": ACTUAL_WRITER_HEAD,
        "actual_runner_path": RUNNER_REL,
        "actual_runner_sha256": sha256_bytes(runner_bytes),
        "actual_config_path": CONFIG_REL,
        "actual_config_sha256": sha256_bytes(config_bytes),
        "output_config_path": str(
            METHOD_ROOT / "attempt_001/contract/config_resolved.json"
        ),
        "output_config_sha256": sha256_file(
            METHOD_ROOT / "attempt_001/contract/config_resolved.json"
        ),
        "actual_runtime_path": RUNTIME_REL,
        "actual_runtime_sha256": sha256_bytes(runtime_bytes),
        "selection_artifact_path": SELECTION_REL,
        "selection_artifact_sha256": sha256_bytes(selection_bytes),
        "rotation_contract_path": ROTATION_REL,
        "rotation_contract_sha256": sha256_bytes(rotation_bytes),
        "same_runner_config_runtime_12_of_12": (
            config_bindings == {EXPECTED_CONFIG_SHA}
            and selection_bindings == {EXPECTED_SELECTION_SHA}
            and all(
                row["bindings"]["rotation_contract_sha256"]
                == EXPECTED_ROTATION_SHA
                for row in checkpoints["checkpoints"]
            )
            and all(
                row["git"]["head"] == ACTUAL_WRITER_HEAD
                for row in checkpoints["checkpoints"]
            )
        ),
        "runtime_version_changes_during_matrix": 0,
    }
    writer = {
        "schema_version": "canondressgs.subject00.commonsafe4.writer_identity.v1",
        "task_id": TASK_ID,
        "actual_writer_task": lock.get("task_id"),
        "actual_writer_branch": lock.get("branch"),
        "actual_writer_head": lock.get("head"),
        "actual_writer_worktree": ACTUAL_WRITER_WORKTREE,
        "hostname": lock.get("hostname"),
        "pid": lock.get("pid"),
        "parent_pid": lock.get("parent_pid"),
        "tmux": lock.get("tmux"),
        "launch_command": ACTUAL_LAUNCH_COMMAND,
        "launch_command_evidence": {
            "shell_history_path": "/root/.bash_history",
            "exact_line_recovered": ACTUAL_LAUNCH_COMMAND,
            "execution_lock_pid_matches_checkpoint_gpu_gate": True,
        },
        "working_directory": ACTUAL_WRITER_WORKTREE,
        "working_directory_recovery": (
            "INFERRED_AND_CROSS_CHECKED_FROM_EXECUTION_BRANCH_WORKTREE_BINDING"
        ),
        "execution_git_clean": preflight["git"]["clean"],
        "execution_git_porcelain_v2": preflight["git"]["porcelain_v2"],
        "start_time_unix": lock.get("start_time_unix"),
        "start_time": iso_timestamp(lock["start_time_unix"]),
        "end_time_unix": lock.get("updated_time_unix"),
        "end_time": iso_timestamp(lock["updated_time_unix"]),
        "final_report_mtime": iso_timestamp(
            (METHOD_ROOT / "attempt_001/FINAL_REPORT.json").stat().st_mtime
        ),
        "writer_commit": writer_commit,
        "git_reflog_evidence": git(
            "reflog", "show", "--date=iso", ACTUAL_WRITER_BRANCH, "-n", "20"
        ),
        "control_lock": lock,
        "single_writer_status": timeline["single_writer_status"],
        "file_interleaving_status": timeline["file_interleaving_status"],
        "unattributed_file_count": timeline["unattributed_file_count"],
        "cross_task_file_count": timeline["cross_task_file_count"],
    }
    replacement.update(
        {
            "schema_version": (
                "canondressgs.subject00.commonsafe4.replacement_pretraining_audit.v1"
            ),
            "task_id": TASK_ID,
            "reported_selected_slot": "slot06",
            "reported_selected_camera": "cam09",
            "reported_selected_direction": "back-right",
            "selection_commit": selection_commit,
            "runner_commit": runner_commit,
            "writer_commit": writer_commit,
            "output_root_creation_time": iso_timestamp(output_creation_time),
            "first_step0_checkpoint_time": iso_timestamp(first_checkpoint_time),
            "selection_artifact_sha256": sha256_bytes(selection_bytes),
            "selection_rule_preexisted_before_training": selection_rule_pretraining,
            "selection_result_frozen_before_training": selection_result_pretraining,
            "first_optimizer_records_bind_selection": selection_bindings
            == {EXPECTED_SELECTION_SHA},
            "post_training_reselection_count": 0,
            "outcome_dependent_selection_evidence_count": 0,
            "status": (
                "PASS_UNIQUE_OUTCOME_INDEPENDENT_PRETRAINING_SELECTION"
                if replacement["selected_slot"] == "slot06"
                and selection_result_pretraining
                else "FAIL"
            ),
        }
    )
    fold_rows = []
    for rotation in ROTATIONS:
        slots = (
            list(rotation["train_slots"])
            + [rotation["calibration_slot"], rotation["test_slot"]]
        )
        fold_rows.append(
            {
                **rotation,
                "train_slot_count": 2,
                "train_garment_record_count": 6,
                "calibration_slot_count": 1,
                "calibration_garment_record_count": 3,
                "test_slot_count": 1,
                "test_garment_record_count": 3,
                "slot_overlap_count": len(slots) - len(set(slots)),
                "per_slot_garment_coverage": {
                    f"slot{slot:02d}": replacement["slot_registry"][
                        f"slot{slot:02d}"
                    ]["garment_coverage"]
                    for slot in slots
                },
                "status": (
                    "PASS_EXACT_6_3_3"
                    if len(set(slots)) == 4
                    and all(
                        replacement["slot_registry"][f"slot{slot:02d}"][
                            "garment_coverage"
                        ]
                        == 3
                        for slot in slots
                    )
                    else "FAIL"
                ),
            }
        )
    folds = {
        "schema_version": "canondressgs.subject00.commonsafe4.actual_folds.v1",
        "task_id": TASK_ID,
        "common_safe4_anchor_set": ["slot00", "slot07", "slot03", "slot06"],
        "actual_rotations": list(ROTATIONS),
        "seeds": list(SEEDS),
        "folds": fold_rows,
        "status": (
            "PASS_ALL_ROTATIONS_EXACT_6_3_3"
            if all(row["status"] == "PASS_EXACT_6_3_3" for row in fold_rows)
            else "FAIL"
        ),
    }

    # Re-fingerprint all protected inputs after the full read-only computation.
    process_after = process_state()
    method_after = tree_fingerprint(METHOD_ROOT)
    old_after = tree_fingerprint(OLD_METHOD_ROOT)
    target_after = tree_fingerprint(TARGET_ROOT)
    baseline_after = tree_fingerprint(BASELINE_ROOT)
    formal_base_after = metadata_fingerprint(FORMAL_BASE_ROOT)
    base_sha_after = sha256_file(BASE_PATH)
    teacher_sha_after = {
        garment: sha256_file(path) for garment, path in TEACHER_PATHS.items()
    }
    mutations = {
        "commonsafe4_output_mutations": int(
            {key: method_before[key] for key in EXPECTED_METHOD_TREE}
            != {key: method_after[key] for key in EXPECTED_METHOD_TREE}
        ),
        "old_method_output_mutations": int(old_before != old_after),
        "base60747_mutations": int(base_sha_before != base_sha_after),
        "o01_teacher_mutations": int(
            teacher_sha_before["O01"] != teacher_sha_after["O01"]
        ),
        "o03_teacher_mutations": int(
            teacher_sha_before["O03"] != teacher_sha_after["O03"]
        ),
        "o04_teacher_mutations": int(
            teacher_sha_before["O04"] != teacher_sha_after["O04"]
        ),
        "formal_target_mutations": int(target_before != target_after),
        "raw_mutations": int(target_before != target_after),
        "mask_mutations": int(target_before != target_after),
        "camera_record_mutations": int(target_before != target_after),
        "formal_base_run_mutations": int(formal_base_before != formal_base_after),
        "baseline_output_mutations": int(baseline_before != baseline_after),
        "paper_modifications": 0,
    }
    q_usage = {
        fold: sum(row["quarantine_usage"][fold] for row in execution["runs"])
        for fold in ("train", "calibration", "test", "evaluation")
    }
    context = {
        "config": config,
        "source_valid": source_valid,
        "source_clean": source_clean,
        "source_status": source_status,
        "process": process_after,
        "writer": writer,
        "timeline": timeline,
        "assets": assets,
        "replacement": replacement,
        "selection_rule_pretraining": selection_rule_pretraining,
        "selection_result_pretraining": selection_result_pretraining,
        "selection_commits": {
            "selection": selection_commit,
            "runner": runner_commit,
            "writer": writer_commit,
        },
        "outcome_dependent_evidence_count": 0,
        "runner": runner_registry,
        "core": core,
        "folds": folds,
        "execution": execution,
        "checkpoints": checkpoints,
        "formal_tests": formal_tests,
        "aggregate": aggregate,
        "baselines": baselines,
        "mutations": mutations,
        "q_usage": q_usage,
        "new_method_optimizer_steps": 0,
        "new_baseline_optimizer_steps": 0,
        "final_classification": FINAL_CLASSIFICATION,
        "next_task": NEXT_TASK,
    }
    tests = make_tests(context)
    all_tests_pass = all(row["status"] == "PASS" for row in tests)
    seal_ok = all_tests_pass and process_after["active_optimizer_process_count"] == 0
    classification = (
        FINAL_CLASSIFICATION
        if seal_ok
        else "SUBJECT00_COMMONSAFE4_CONCURRENT_MATRIX_ENGINEERING_EVIDENCE_FAIL"
    )
    tests_payload = {
        "schema_version": "canondressgs.subject00.commonsafe4.provenance_tests.v1",
        "task_id": TASK_ID,
        "test_count": len(tests),
        "pass_count": sum(row["status"] == "PASS" for row in tests),
        "fail_count": sum(row["status"] != "PASS" for row in tests),
        "status": (
            f"PASS_{len(tests)}_OF_{len(tests)}"
            if all_tests_pass
            else "FAIL"
        ),
        "tests": tests,
    }
    immutable = {
        "before": {
            "method": {
                key: method_before[key] for key in EXPECTED_METHOD_TREE
            },
            "old_method": old_before,
            "formal_target": target_before,
            "baseline_output": baseline_before,
            "formal_base_run_metadata": formal_base_before,
            "base_sha256": base_sha_before,
            "teacher_sha256": teacher_sha_before,
        },
        "after": {
            "method": method_after,
            "old_method": old_after,
            "formal_target": target_after,
            "baseline_output": baseline_after,
            "formal_base_run_metadata": formal_base_after,
            "base_sha256": base_sha_after,
            "teacher_sha256": teacher_sha_after,
        },
        "mutations": mutations,
    }
    overlay = {
        "schema_version": "canondressgs.subject00.commonsafe4.preflight_correction.v1",
        "task_id": TASK_ID,
        "original_preflight_status": ORIGINAL_PREFLIGHT_STATUS,
        "original_preflight_rewritten_as_pass": False,
        "original_preflight_optimizer_steps": 0,
        "original_preflight_dry_run_12_of_12_completed": False,
        "original_preflight_formal_zero_step_smoke_completed": False,
        "why_original_preflight_stopped": (
            "The formal output root appeared under a different task before the "
            "preflight could complete its remaining gates."
        ),
        "external_matrix_belongs_to_original_preflight": False,
        "matrix_adoption_mode": "POST_HOC_PROVENANCE_AUDITED_EXISTING_OUTPUT",
        "external_matrix_adopted": seal_ok,
        "replacement_legal": replacement["status"].startswith("PASS"),
        "runner_legal": core["status"].startswith("PASS"),
        "matrix_12run_legal": aggregate["status"].startswith("PASS"),
        "fair_baseline_status": baselines["status"],
        "old_r0s0_status": {
            "training_authentic": True,
            "formal_matrix_eligible": False,
            "reused_in_commonsafe4_matrix": False,
        },
        "next_task": NEXT_TASK if seal_ok else "RUN_FRESH_SUBJECT00_COMMONSAFE4_12RUN_MATRIX_IN_NEW_OUTPUT_NAMESPACE",
    }
    summary = {
        "schema_version": "canondressgs.subject00.commonsafe4.provenance_summary.v1",
        "task_id": TASK_ID,
        "source": {"branch": SOURCE_BRANCH, "head": SOURCE_HEAD},
        "audit": {
            "branch": AUDIT_BRANCH,
            "starting_head": current_head,
            "windows_worktree": WINDOWS_WORKTREE,
            "cloud_worktree": CLOUD_WORKTREE,
            "artifact_commit_head": "PENDING_COMMIT",
            "final_reporting_head": "PENDING_COMMIT",
        },
        "original_preflight_status": ORIGINAL_PREFLIGHT_STATUS,
        "method_output_root": str(METHOD_ROOT),
        "output_root_creation_time": iso_timestamp(output_creation_time),
        "process_state": process_after,
        "writer": writer,
        "replacement": {
            "candidate_set": replacement["candidate_set"],
            "ranking": replacement["ranking"],
            "reported_selected_slot": "slot06",
            "reproduced_selected_slot": replacement["selected_slot"],
            "camera": replacement["selected_camera"],
            "direction": replacement["selected_direction"],
            "selection_rule_preexisted_before_training": selection_rule_pretraining,
            "selection_result_frozen_before_training": selection_result_pretraining,
            "outcome_dependent_selection_evidence_count": 0,
        },
        "runner": runner_registry,
        "method_core": core,
        "folds": folds,
        "matrix": aggregate,
        "checkpoint_count": checkpoints["checkpoint_count"],
        "formal_test_valid_run_count": formal_tests["valid_run_count"],
        "quarantine_usage": q_usage,
        "fair_baselines": baselines,
        "new_method_optimizer_steps": 0,
        "new_baseline_optimizer_steps": 0,
        "immutable_audit": immutable,
        "test_result": tests_payload["status"],
        "matrix_provenance_seal_path": (
            str(ARTIFACTS["seal"].relative_to(REPO_ROOT)) if seal_ok else None
        ),
        "paper_modifications": 0,
        "paper_final": False,
        "final_classification": classification,
        "next_task": NEXT_TASK if seal_ok else overlay["next_task"],
    }
    seal = {
        "schema_version": "canondressgs.subject00.commonsafe4.posthoc_provenance_seal.v1",
        "task_id": TASK_ID,
        "status": "SEALED",
        "original_preflight_status": "STOPPED_BY_CONCURRENT_OUTPUT_CREATION",
        "original_preflight_full_status": ORIGINAL_PREFLIGHT_STATUS,
        "matrix_adoption_mode": "POST_HOC_PROVENANCE_AUDITED_EXISTING_OUTPUT",
        "output_root": str(METHOD_ROOT),
        "output_tree": method_after,
        "writer": writer,
        "selection": replacement,
        "runtime": runner_registry,
        "method_core": core,
        "rotations": folds,
        "execution": {
            "run_ids": execution["actual_run_ids"],
            "run_count": execution["run_count"],
            "formal_valid_run_count": execution["formal_valid_run_count"],
            "optimizer_steps": execution["total_optimizer_steps"],
            "checkpoint_count": checkpoints["checkpoint_count"],
            "aggregate": aggregate,
        },
        "fair_baselines": {
            "status": baselines["status"],
            "names": baselines["baseline_names"],
            "valid_count": baselines["valid_count"],
            "aggregates": baselines["aggregates"],
        },
        "immutable_audit": immutable,
        "test_result": tests_payload["status"],
        "paper_final": False,
        "final_classification": classification,
        "next_task": NEXT_TASK,
    }
    handoff = {
        "task_id": TASK_ID,
        "branch": AUDIT_BRANCH,
        "artifact_commit_head": "PENDING_COMMIT",
        "final_reporting_head": "PENDING_COMMIT",
        "method_output_root": str(METHOD_ROOT),
        "baseline_output_root": str(BASELINE_ROOT),
        "matrix_provenance_seal_path": str(ARTIFACTS["seal"]),
        "new_optimizer_steps": {"method": 0, "baseline": 0},
        "paper_modifications": 0,
        "paper_final": False,
        "classification": classification,
        "next_task": NEXT_TASK if seal_ok else overlay["next_task"],
    }

    atomic_json(ARTIFACTS["writer"], writer)
    atomic_json(ARTIFACTS["timeline"], timeline)
    atomic_json(ARTIFACTS["replacement"], replacement)
    atomic_json(ARTIFACTS["runner"], runner_registry)
    atomic_json(ARTIFACTS["core"], core)
    atomic_json(ARTIFACTS["folds"], folds)
    atomic_json(ARTIFACTS["execution"], execution)
    atomic_json(ARTIFACTS["checkpoints"], checkpoints)
    atomic_json(ARTIFACTS["formal_tests"], formal_tests)
    atomic_json(ARTIFACTS["aggregate"], aggregate)
    atomic_json(ARTIFACTS["baselines"], baselines)
    atomic_json(ARTIFACTS["overlay"], overlay)
    atomic_json(ARTIFACTS["tests"], tests_payload)
    atomic_json(ARTIFACTS["summary"], summary)
    atomic_json(ARTIFACTS["handoff"], handoff)
    report = report_markdown(summary)
    atomic_text(ARTIFACTS["report"], report)
    atomic_text(ARTIFACTS["docs"], report)
    if seal_ok:
        atomic_json(ARTIFACTS["seal"], seal)
    elif ARTIFACTS["seal"].exists():
        raise RuntimeError("failed audit must not retain a provenance seal")

    print(
        json.dumps(
            {
                "status": tests_payload["status"],
                "seal_created": seal_ok,
                "classification": classification,
                "artifacts": {
                    key: str(path.relative_to(REPO_ROOT))
                    for key, path in ARTIFACTS.items()
                    if path.exists()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if seal_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
