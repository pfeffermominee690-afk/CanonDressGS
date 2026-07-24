#!/usr/bin/env python3
"""Execute the frozen Subject00 formal strict-split base experiment.

The protocol artifacts are immutable inputs.  This driver owns execution
only: entry gates, five-pass training, atomic checkpoints, exact resume,
fixed96 trajectory evaluation, final streaming evaluation, and checkpoint
roundtrip verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from torchmetrics.functional.image import structural_similarity_index_measure

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.second_identity import run_subject00_medium_pilot as medium
from tools.second_identity import run_subject00_short_canary as core
from utils.image_utils import crop_image
from utils.smpl_utils import init_smpl, rigid_transform_numba
from utils.surface_lbs_utils import SURFACE_LBS_FILES, sha256_file


TASK_ID = "AAAI27-SUBJECT00-FORMAL-STRICT-SPLIT-BASE-001"
PROTOCOL_TASK_ID = "MMLPHUMAN-SUBJECT00-FORMAL-STRICT-SPLIT-PROTOCOL-001"
PROTOCOL_BRANCH = (
    "research/mmlphuman-subject00-formal-strict-split-protocol-20260723"
)
PROTOCOL_HEAD = "4993f5c865ec19895f35811fa399fc4a1834c6a7"
TARGET_BRANCH = (
    "research/mmlphuman-subject00-formal-strict-split-experiment-20260725"
)
FORMAL_OUTPUT_ROOT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-MMLPHUMAN-FORMAL-STRICT-SPLIT-001"
)
ATTEMPT_ROOT = FORMAL_OUTPUT_ROOT / "attempt_001"
CONFLICTING_REQUEST_OUTPUT_ROOT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-FORMAL-STRICT-SPLIT-BASE-001"
)
CANARY_ROOT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-MMLPHUMAN-SHORT-CANARY-001/attempt_001"
)
MEDIUM_ROOT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-MMLPHUMAN-MEDIUM-PILOT-001/attempt_001"
)
CANARY_STEP0_PATH = CANARY_ROOT / "checkpoints/step_000000.pth"
CANARY_STEP0_SHA = (
    "29d0edb11474f4384b6d409141d17f5da91c98824bd070196d50804cd7cb480a"
)
CANARY_STEP0_BYTES = 701_938_720
MINIMUM_FREE_BYTES = 32_212_254_720
N_VALID = 20_249
PASS_COUNT = 5
FINAL_STEP = 101_245
CHECKPOINT_STEPS = (20_249, 40_498, 60_747, 80_996, 100_000, 101_245)
ALL_TRAJECTORY_STEPS = (0, *CHECKPOINT_STEPS)
FIXED96_SHA = (
    "38476b7d9f6a012e0e21d7f188c5db5131506ec2231e8a7b50a389c7c92ad98a"
)
SINGLE_PASS_SHA = (
    "0f0e7463d9f9066f54e19ec31f85f722afd58e58aba123af144918fdc97639f8"
)
GLOBAL_SCHEDULE_SHA = (
    "1c1b4d3a0adc93e887ab15b9267e1b1284508dba40926eaf7c0e944f5af932c3"
)
FORMAL_ARTIFACTS = {
    "contract": (
        "paper_protocol/second_identity/subject00_formal_training_contract.json",
        "contract_content_sha256",
    ),
    "schedule": (
        "paper_protocol/second_identity/subject00_formal_training_schedule.json",
        "schedule_content_sha256",
    ),
    "records": (
        "paper_protocol/second_identity/subject00_formal_train_record_manifest.json",
        "manifest_content_sha256",
    ),
    "evaluation": (
        "paper_protocol/second_identity/subject00_formal_evaluation_manifests.json",
        "manifests_content_sha256",
    ),
    "visual": (
        "paper_protocol/second_identity/subject00_formal_visual_review_manifest.json",
        "manifest_content_sha256",
    ),
    "resource": (
        "paper_protocol/second_identity/subject00_formal_resource_budget.json",
        "budget_content_sha256",
    ),
    "summary": (
        "paper_protocol/second_identity/subject00_formal_protocol_final_summary.json",
        "summary_content_sha256",
    ),
    "handoff": (
        "project_control_handoff/subject00_formal_protocol_handoff.json",
        "handoff_content_sha256",
    ),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_sha(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, value: object) -> None:
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


def append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")
    with path.open("ab") as handle:
        handle.write(encoded)
        handle.flush()


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), *args], text=True
    ).strip()


def configure_runtime_modules() -> None:
    medium.TASK_ID = TASK_ID
    medium.TARGET_BRANCH = TARGET_BRANCH
    medium.SOURCE_HEAD = PROTOCOL_HEAD
    medium.TRAIN_ORDER_SHA = SINGLE_PASS_SHA
    medium.EVAL_ORDER_SHA = FIXED96_SHA
    core.TASK_ID = TASK_ID
    core.TARGET_BRANCH = TARGET_BRANCH
    core.TRAIN_ORDER_SHA = SINGLE_PASS_SHA
    core.EVAL_ORDER_SHA = FIXED96_SHA


def validate_seal(value: dict[str, Any], seal_key: str) -> None:
    claimed = value.get(seal_key)
    unsigned = dict(value)
    unsigned.pop(seal_key, None)
    if not claimed or canonical_sha(unsigned) != claimed:
        raise RuntimeError(f"formal protocol content seal failed: {seal_key}")


def load_protocol(availability_path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for name, (relative, seal_key) in FORMAL_ARTIFACTS.items():
        path = REPO_ROOT / relative
        value = read_json(path)
        validate_seal(value, seal_key)
        values[name] = value
        hashes[f"{name}_file_sha256"] = sha256_file(path)
        hashes[f"{name}_content_sha256"] = value[seal_key]

    contract = values["contract"]
    schedule = values["schedule"]
    records = values["records"]
    evaluation = values["evaluation"]
    visual = values["visual"]
    resource = values["resource"]
    handoff = values["handoff"]
    if contract["task_id"] != PROTOCOL_TASK_ID:
        raise RuntimeError("formal protocol task id changed")
    if Path(contract["formal_output_root"]) != FORMAL_OUTPUT_ROOT:
        raise RuntimeError("formal protocol output root changed")
    if Path(handoff["formal_output_root"]) != FORMAL_OUTPUT_ROOT:
        raise RuntimeError("formal handoff output root changed")
    if int(resource["resource_gate"]["minimum_free_bytes"]) != MINIMUM_FREE_BYTES:
        raise RuntimeError("formal disk gate changed")
    if schedule["optimizer_steps"] != FINAL_STEP:
        raise RuntimeError("formal optimizer-step count changed")
    if schedule["pass_count"] != PASS_COUNT:
        raise RuntimeError("formal pass count changed")
    if schedule["new_checkpoint_steps"] != list(CHECKPOINT_STEPS):
        raise RuntimeError("formal checkpoint schedule changed")
    if len(schedule["steps"]) != FINAL_STEP:
        raise RuntimeError("formal schedule is incomplete")
    if schedule["global_data_order_sha256"] != GLOBAL_SCHEDULE_SHA:
        raise RuntimeError("formal global schedule hash changed")
    if records["record_count"] != N_VALID or len(records["records"]) != N_VALID:
        raise RuntimeError("formal record count changed")
    if canonical_sha(records["records"]) != SINGLE_PASS_SHA:
        raise RuntimeError("formal single-pass order changed")
    if visual["query_count"] != 96 or len(visual["queries"]) != 96:
        raise RuntimeError("formal fixed96 count changed")
    if canonical_sha(visual["queries"]) != FIXED96_SHA:
        raise RuntimeError("formal fixed96 order changed")
    if evaluation["full_evaluation"]["only_checkpoint_step"] != FINAL_STEP:
        raise RuntimeError("formal full evaluation is not final-only")
    full_count = sum(
        int(evaluation["quadrants"][name]["valid_count"])
        for name in evaluation["quadrant_order"]
    )
    if full_count != 29_954:
        raise RuntimeError("formal full-evaluation count changed")

    availability = read_json(availability_path)
    if sha256_file(availability_path) != evaluation["availability_raw_file_sha256"]:
        raise RuntimeError("availability raw file hash changed")
    if canonical_sha(availability["entries"]) != evaluation["availability_entries_sha256"]:
        raise RuntimeError("availability canonical hash changed")

    source_files = contract["optimizer_loss_scheduler_renderer"]["source_files"]
    for relative, expected in source_files.items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"runtime source closure changed: {relative}")

    medium_contract = read_json(
        REPO_ROOT
        / "paper_protocol/second_identity/subject00_medium_pilot_execution_contract.json"
    )
    canary_evaluation = read_json(
        REPO_ROOT
        / "paper_protocol/second_identity/subject00_canary_evaluation_manifest.json"
    )
    runtime_bundle = {
        "contract": medium_contract,
        "records": records,
        "evaluation": canary_evaluation,
    }
    return {
        **values,
        "availability": availability,
        "hashes": hashes,
        "runtime_bundle": runtime_bundle,
        "full_query_count": full_count,
    }


def tree_metadata_fingerprint(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        stat = path.stat()
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return {
        "file_count": len(rows),
        "bytes": sum(item["size"] for item in rows),
        "metadata_sha256": canonical_sha(rows),
    }


def credential_findings() -> list[str]:
    patterns = (
        "-----BEGIN " + "PRIVATE KEY-----",
        "-----BEGIN OPENSSH " + "PRIVATE KEY-----",
        "gh" + "p_",
        "github_" + "pat_",
        "AK" + "IA",
    )
    findings = []
    tracked = git_output("ls-files", "-z").split("\0")
    for relative in tracked:
        if not relative:
            continue
        path = REPO_ROOT / relative
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if any(pattern in text for pattern in patterns):
            findings.append(relative)
    return findings


def active_gpu_processes() -> list[dict[str, str]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    rows = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:
            rows.append({"pid": parts[0], "name": parts[1], "used_mib": parts[2]})
    return rows


def environment_snapshot() -> dict[str, Any]:
    try:
        import gsplat

        gsplat_version = getattr(gsplat, "__version__", "UNKNOWN")
    except Exception as exc:
        gsplat_version = f"UNAVAILABLE: {type(exc).__name__}: {exc}"
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gsplat": gsplat_version,
        "gpu": gpu,
    }


def preflight(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    *,
    allow_attempt: bool,
) -> dict[str, Any]:
    branch = git_output("branch", "--show-current")
    head = git_output("rev-parse", "HEAD")
    if branch != TARGET_BRANCH:
        raise RuntimeError(f"unexpected execution branch: {branch}")
    subprocess.run(
        ["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", PROTOCOL_HEAD, head],
        check=True,
    )
    if git_output("status", "--porcelain"):
        raise RuntimeError("execution worktree is not clean")
    if args.attempt_root != ATTEMPT_ROOT:
        raise RuntimeError(f"attempt root must be frozen path: {ATTEMPT_ROOT}")
    if CONFLICTING_REQUEST_OUTPUT_ROOT.exists():
        raise RuntimeError(
            "conflicting non-protocol output root exists; refusing ambiguous execution"
        )
    if args.attempt_root.exists() != allow_attempt:
        expected = "present" if allow_attempt else "absent"
        raise RuntimeError(f"attempt_001 must be {expected} for this phase")
    if not CANARY_ROOT.is_dir() or not MEDIUM_ROOT.is_dir():
        raise RuntimeError("canary or medium immutable source root is absent")
    if sha256_file(CANARY_STEP0_PATH) != CANARY_STEP0_SHA:
        raise RuntimeError("authorized step0 checkpoint SHA changed")
    if CANARY_STEP0_PATH.stat().st_size != CANARY_STEP0_BYTES:
        raise RuntimeError("authorized step0 checkpoint byte count changed")

    baseline = protocol["resource"]["output_immutability_baseline"]
    canary_fingerprint = tree_metadata_fingerprint(CANARY_ROOT)
    medium_fingerprint = tree_metadata_fingerprint(MEDIUM_ROOT)
    if canary_fingerprint != baseline["canary_attempt_001"]:
        raise RuntimeError("canary output fingerprint changed")
    if medium_fingerprint != baseline["medium_attempt_001"]:
        raise RuntimeError("medium output fingerprint changed")

    output_parent = FORMAL_OUTPUT_ROOT.parent
    disk = shutil.disk_usage(output_parent)
    if disk.free < MINIMUM_FREE_BYTES:
        raise RuntimeError(
            f"formal disk gate failed: free={disk.free}, required={MINIMUM_FREE_BYTES}"
        )
    processes = active_gpu_processes()
    if processes:
        raise RuntimeError(f"formal GPU exclusivity gate failed: {processes}")
    findings = credential_findings()
    if findings:
        raise RuntimeError(f"credential findings are nonzero: {findings}")
    if not os.access(output_parent, os.W_OK):
        raise RuntimeError("formal output parent is not writable")
    return {
        "schema_version": "subject00.formal.execution_preflight.v1",
        "task_id": TASK_ID,
        "status": "PASS",
        "execution_head": head,
        "protocol_head": PROTOCOL_HEAD,
        "attempt_root": str(args.attempt_root),
        "environment": environment_snapshot(),
        "disk": {
            "free_bytes": disk.free,
            "required_free_bytes": MINIMUM_FREE_BYTES,
        },
        "gpu_compute_processes": processes,
        "credential_findings": findings,
        "canary_fingerprint": canary_fingerprint,
        "medium_fingerprint": medium_fingerprint,
        "protocol_hashes": protocol["hashes"],
        "PAPER_FINAL": 0,
    }


def checkpoint_paths(attempt_root: Path, step: int) -> tuple[Path, Path]:
    payload = attempt_root / f"checkpoints/step_{step:06d}.pth"
    return payload, payload.with_suffix(".sidecar.json")


def save_checkpoint_atomic(
    attempt_root: Path,
    model,
    *,
    step: int,
    scene_scale: float,
    execution_head: str,
    protocol: dict[str, Any],
    elapsed_seconds: float,
    metrics_snapshot: dict[str, Any],
) -> dict[str, Any]:
    path, sidecar_path = checkpoint_paths(attempt_root, step)
    if path.exists() or sidecar_path.exists():
        raise RuntimeError(f"checkpoint overwrite forbidden at step {step}")
    payload = model.capture()
    payload.update(
        {
            "checkpoint_kind": "SUBJECT00_FORMAL_STRICT_SPLIT_BASE",
            "checkpoint_version": 2,
            "attempt": "attempt_001",
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
            "data_order_position": step,
            "scene_scale": float(scene_scale),
            "execution_head": execution_head,
            "source_branch": TARGET_BRANCH,
            "source_head": execution_head,
            "protocol_head": PROTOCOL_HEAD,
            "protocol_sha256": protocol["contract"]["contract_content_sha256"],
            "data_manifest_sha256": protocol["records"]["manifest_content_sha256"],
            "single_pass_data_order_sha256": SINGLE_PASS_SHA,
            "global_data_order_sha256": GLOBAL_SCHEDULE_SHA,
            "elapsed_seconds": float(elapsed_seconds),
            "metrics_snapshot": metrics_snapshot,
            "attachment_provenance": model.surface_attachment_provenance,
            "surface_attachment_state_keys": list(SURFACE_LBS_FILES),
            "template_sha256": (
                "f10a3b516e2b3a2ad38dc4924a3692b2f3e72a6cc9e66f3c0063c4e9cd210031"
            ),
            "sampler_sha256": (
                "98af26a3f578d1e0239f6ea414e3664dae10a4b60f47aef594d0978f61b9ee82"
            ),
            "derived_manifest_sha256": medium.DERIVED_MANIFEST_SHA,
            "camera_split_sha256": medium.CAMERA_SPLIT_SHA,
            "pose_split_sha256": medium.POSE_SPLIT_SHA,
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
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    with temporary.open("rb+") as handle:
        os.fsync(handle.fileno())
    digest = sha256_file(temporary)
    size = temporary.stat().st_size
    os.replace(temporary, path)
    sidecar = {
        "schema_version": "subject00.formal.checkpoint_sidecar.v1",
        "task_id": TASK_ID,
        "attempt": "attempt_001",
        "step": step,
        "training_step": step,
        "data_order_position": step,
        "bytes": size,
        "sha256": digest,
        "path": str(path),
        "execution_head": execution_head,
        "protocol_head": PROTOCOL_HEAD,
        "protocol_sha256": protocol["contract"]["contract_content_sha256"],
        "data_manifest_sha256": protocol["records"]["manifest_content_sha256"],
        "elapsed_seconds": float(elapsed_seconds),
        "metrics_snapshot": metrics_snapshot,
        "atomically_sealed": True,
        "checkpoint_overwrite": 0,
    }
    atomic_write_json(sidecar_path, sidecar)
    return sidecar


def load_complete_checkpoint(attempt_root: Path, step: int) -> tuple[Any, dict[str, Any]]:
    path, sidecar_path = checkpoint_paths(attempt_root, step)
    if not path.is_file() or not sidecar_path.is_file():
        raise RuntimeError(f"checkpoint {step} is incomplete")
    sidecar = read_json(sidecar_path)
    if not sidecar.get("atomically_sealed"):
        raise RuntimeError(f"checkpoint {step} sidecar is not sealed")
    if sidecar["bytes"] != path.stat().st_size:
        raise RuntimeError(f"checkpoint {step} size differs from sidecar")
    if sidecar["sha256"] != sha256_file(path):
        raise RuntimeError(f"checkpoint {step} SHA differs from sidecar")
    payload = torch.load(path, weights_only=False)
    if (
        int(payload["training_step"]) != step
        or int(payload["data_order_position"]) != step
        or payload["execution_head"] != sidecar["execution_head"]
        or payload["protocol_head"] != PROTOCOL_HEAD
    ):
        raise RuntimeError(f"checkpoint {step} payload position/provenance mismatch")
    return payload, sidecar


def fixed96_bundle(protocol: dict[str, Any]) -> dict[str, Any]:
    queries = protocol["visual"]["queries"]
    order = []
    for item in queries:
        if item["quadrant"] not in order:
            order.append(item["quadrant"])
    return {
        "evaluation": {
            "queries": queries,
            "quadrant_contract_order": order,
        }
    }


def evaluate_fixed96(
    model,
    protocol: dict[str, Any],
    args: argparse.Namespace,
    *,
    step: int,
    lpips_metric,
    lpips_error: str | None,
) -> dict[str, Any]:
    states = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all(),
    )
    try:
        report, arrays = medium.evaluate_queries(
            model,
            fixed96_bundle(protocol),
            args.data_root,
            args.attempt_root,
            step=step,
            queries=protocol["visual"]["queries"],
            lpips_metric=lpips_metric,
            lpips_error=lpips_error,
            visual_subdir=f"fixed96/step_{step:06d}",
            smoke_ordinals=set(medium.DIAGNOSTIC_ORDINALS),
        )
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])
        torch.cuda.set_rng_state_all(states[3])
    report["schema_version"] = "subject00.formal.fixed96_evaluation.v1"
    report["task_id"] = TASK_ID
    report["query_order_sha256"] = FIXED96_SHA
    report["checkpoint_selection_authorized"] = False
    path = args.attempt_root / f"evaluations/fixed96_step_{step:06d}.json"
    atomic_write_json(path, report)
    baseline = args.attempt_root / f"evaluations/fixed96_step_{step:06d}_roundtrip.npz"
    np.savez(baseline, **{name: arrays[name] for name in sorted(arrays)})
    return {
        "step": step,
        "path": str(path),
        "sha256": sha256_file(path),
        "query_count": report["query_count"],
        "successful_render_count": report["successful_render_count"],
        "failed_render_count": report["failed_render_count"],
        "roundtrip_baseline": str(baseline),
        "roundtrip_baseline_sha256": sha256_file(baseline),
    }


def read_training_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def validate_resume_log(records: list[dict[str, Any]], schedule: dict[str, Any], step: int) -> None:
    if len(records) != step:
        raise RuntimeError(
            "resume log must end exactly at the sealed checkpoint; replay is forbidden"
        )
    for index, record in enumerate(records):
        expected = schedule["steps"][index]
        if (
            int(record["step"]) != index + 1
            or int(record["pose_id"]) != int(expected["pose_id"])
            or int(record["camera_id"]) != int(expected["camera_id"])
            or int(record["pass_index"]) != int(expected["pass_index"])
        ):
            raise RuntimeError(f"resume training-log order mismatch at index {index}")


def train_phase(args: argparse.Namespace, protocol: dict[str, Any]) -> int:
    configure_runtime_modules()
    resume_step = int(args.resume_step)
    if resume_step and resume_step not in CHECKPOINT_STEPS[:-1]:
        raise RuntimeError("resume step is not a frozen non-final checkpoint")
    if resume_step and not args.interruption_reason:
        raise RuntimeError("resume requires --interruption-reason")
    gate = preflight(args, protocol, allow_attempt=bool(resume_step))
    execution_head = gate["execution_head"]

    if not resume_step:
        args.attempt_root.mkdir(parents=True, exist_ok=False)
        for relative in (
            "audits",
            "checkpoints",
            "contract",
            "evaluations",
            "snapshots",
            "training_logs",
            "visuals",
        ):
            (args.attempt_root / relative).mkdir()
        atomic_write_json(args.attempt_root / "audits/preflight.json", gate)
        atomic_write_json(
            args.attempt_root / "contract/step0_checkpoint_pointer.json",
            {
                "schema_version": "subject00.formal.step0_pointer.v1",
                "task_id": TASK_ID,
                "path": str(CANARY_STEP0_PATH),
                "sha256": CANARY_STEP0_SHA,
                "bytes": CANARY_STEP0_BYTES,
                "step": 0,
                "data_order_position": 0,
                "reused": True,
                "copied": False,
            },
        )
        atomic_write_json(
            args.attempt_root / "contract/protocol_pointer.json",
            {
                "task_id": TASK_ID,
                "protocol_task_id": PROTOCOL_TASK_ID,
                "protocol_branch": PROTOCOL_BRANCH,
                "protocol_head": PROTOCOL_HEAD,
                "execution_branch": TARGET_BRANCH,
                "execution_head": execution_head,
                "protocol_hashes": protocol["hashes"],
            },
        )
    else:
        append_jsonl(
            args.attempt_root / "audits/interruption_events.jsonl",
            {
                "event": "INFRASTRUCTURE_RESUME",
                "resume_step": resume_step,
                "reason": args.interruption_reason,
                "timestamp_unix": time.time(),
                "execution_head": execution_head,
            },
        )

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    runtime_bundle = protocol["runtime_bundle"]
    model, scene, runtime_args = core.build_model(
        runtime_bundle, args.data_root, args.assets_root, args.attempt_root
    )
    log_path = args.attempt_root / "training_logs/step_records.jsonl"
    records = read_training_records(log_path)
    trajectory: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []

    if resume_step:
        payload, _ = load_complete_checkpoint(args.attempt_root, resume_step)
        medium.restore_checkpoint_state(model, runtime_args, scene.scene_scale, payload)
        validate_resume_log(records, protocol["schedule"], resume_step)
        initialization = read_json(args.attempt_root / "audits/initialization_audit.json")
        for step in CHECKPOINT_STEPS:
            if step > resume_step:
                break
            _, sidecar = load_complete_checkpoint(args.attempt_root, step)
            checkpoints.append(sidecar)
            fixed_path = args.attempt_root / f"evaluations/fixed96_step_{step:06d}.json"
            trajectory.append(
                {
                    "step": step,
                    "path": str(fixed_path),
                    "sha256": sha256_file(fixed_path),
                    "query_count": 96,
                }
            )
    else:
        payload = torch.load(CANARY_STEP0_PATH, weights_only=False)
        medium.restore_checkpoint_state(model, runtime_args, scene.scene_scale, payload)
        initialization = medium.initialization_audit(
            model, payload, scene.scene_scale, args.assets_root
        )
        atomic_write_json(
            args.attempt_root / "audits/initialization_audit.json", initialization
        )

    initial_attachments = core.attachment_inventory(model, args.assets_root)
    if not resume_step:
        atomic_write_json(
            args.attempt_root / "snapshots/initial_attachment_inventory.json",
            initial_attachments,
        )
    optimizer_audit = core.optimizer_audit(
        model,
        runtime_bundle["contract"],
        scene.scene_scale,
        scheduler_step=resume_step,
    )
    atomic_write_json(
        args.attempt_root / "audits/optimizer_membership.json", optimizer_audit
    )
    lpips_metric, lpips_error = core.make_lpips_preserving_rng()
    warning_set = {
        message for record in records for message in record.get("warning_messages", [])
    }

    for expected in protocol["schedule"]["steps"]:
        step = int(expected["step"])
        if step <= resume_step:
            continue
        item = scene.trainset[int(expected["record_ordinal"]) - 1]
        if (
            int(item["frame_id"]) != int(expected["pose_id"])
            or int(item["cam_id"]) != int(expected["camera_id"])
        ):
            raise RuntimeError(f"formal schedule mismatch at step {step}")
        record = medium.train_step(model, item, runtime_args, step)
        record.update(
            {
                "pass_index": int(expected["pass_index"]),
                "pass_record_ordinal": int(expected["pass_record_ordinal"]),
                "record_ordinal": int(expected["record_ordinal"]),
            }
        )
        append_jsonl(log_path, record)
        records.append(record)
        warning_set.update(record["warning_messages"])
        if step not in CHECKPOINT_STEPS:
            continue

        checkpoint = save_checkpoint_atomic(
            args.attempt_root,
            model,
            step=step,
            scene_scale=scene.scene_scale,
            execution_head=execution_head,
            protocol=protocol,
            elapsed_seconds=time.perf_counter() - started,
            metrics_snapshot={
                "total_loss": record["total_loss"],
                "l1_loss": record["l1_loss"],
                "lpips_loss": record["lpips_loss"],
            },
        )
        checkpoints.append(checkpoint)
        trajectory.append(
            evaluate_fixed96(
                model,
                protocol,
                args,
                step=step,
                lpips_metric=lpips_metric,
                lpips_error=lpips_error,
            )
        )

    if len(records) != FINAL_STEP:
        raise RuntimeError("formal five-pass schedule did not complete")
    final_attachments = core.attachment_inventory(model, args.assets_root)
    if final_attachments != initial_attachments:
        raise RuntimeError("surface attachment/template/LBS state mutated")
    atomic_write_json(
        args.attempt_root / "audits/post_training_attachment_inventory.json",
        final_attachments,
    )
    total_losses = np.asarray([item["total_loss"] for item in records], dtype=np.float64)
    l1_losses = np.asarray([item["l1_loss"] for item in records], dtype=np.float64)
    window = max(1, int(math.floor(FINAL_STEP * 0.05)))
    result = {
        "schema_version": "subject00.formal.training_result.v1",
        "task_id": TASK_ID,
        "status": "TRAINING_COMPLETE_PENDING_ROUNDTRIP_FULL_EVALUATION_AND_REVIEW",
        "execution_head": execution_head,
        "protocol_head": PROTOCOL_HEAD,
        "attempt": "attempt_001",
        "process_segments": 2 if resume_step else 1,
        "infrastructure_resume_from_step": resume_step or None,
        "repeated_optimizer_steps": 0,
        "skipped_optimizer_steps": 0,
        "checkpoint_overwrite": 0,
        "training_runs": 1,
        "optimizer_created": 1,
        "training_forward_batches": FINAL_STEP,
        "backward_calls": FINAL_STEP,
        "optimizer_steps": FINAL_STEP,
        "scheduler_steps": FINAL_STEP,
        "record_count": N_VALID,
        "pass_count": PASS_COUNT,
        "record_exposures": FINAL_STEP,
        "single_pass_data_order_sha256": SINGLE_PASS_SHA,
        "global_data_order_sha256": GLOBAL_SCHEDULE_SHA,
        "initialization": initialization,
        "optimizer_audit": optimizer_audit,
        "checkpoints": checkpoints,
        "fixed96_trajectory": {
            "step0_reuse": str(CANARY_ROOT / "evaluations/step_000000_results.json"),
            "step0_reuse_sha256": sha256_file(
                CANARY_ROOT / "evaluations/step_000000_results.json"
            ),
            "new_evaluations": trajectory,
        },
        "loss_window": {"fraction": 0.05, "count": window},
        "total_loss": {
            "first5_percent_median": float(np.median(total_losses[:window])),
            "last5_percent_median": float(np.median(total_losses[-window:])),
        },
        "l1_loss": {
            "first5_percent_median": float(np.median(l1_losses[:window])),
            "last5_percent_median": float(np.median(l1_losses[-window:])),
        },
        "loss_finite_all": all(item["loss_finite"] for item in records),
        "gradient_finite_all": all(item["gradients_finite"] for item in records),
        "optimizer_state_finite_all": all(
            item["optimizer_state_finite"] for item in records
        ),
        "attachment_template_lbs_unchanged": True,
        "runtime_lbs_counters": model.runtime_lbs_counters,
        "topology_call_counts": {
            "clone": 0,
            "split": 0,
            "densification": 0,
            "topology_changing_prune": 0,
            "off_surface_rebind": 0,
        },
        "warning_set": sorted(warning_set),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "wall_seconds": time.perf_counter() - started,
        "PAPER_FINAL": 0,
    }
    atomic_write_json(args.attempt_root / "training_logs/training_result.json", result)
    atomic_write_json(
        args.attempt_root / "checkpoints/checkpoint_registry.json",
        {
            "schema_version": "subject00.formal.checkpoint_registry.v1",
            "task_id": TASK_ID,
            "step0_pointer": str(CANARY_STEP0_PATH),
            "new_checkpoint_count": len(checkpoints),
            "checkpoints": checkpoints,
            "checkpoint_overwrite": 0,
        },
    )
    scene.tb_writer.close()
    print(
        json.dumps(
            {
                "phase": "train",
                "status": "PASS",
                "steps": FINAL_STEP,
                "new_checkpoints": len(checkpoints),
            },
            sort_keys=True,
        )
    )
    return 0


def post_training_gate(args: argparse.Namespace, protocol: dict[str, Any]) -> str:
    branch = git_output("branch", "--show-current")
    head = git_output("rev-parse", "HEAD")
    if branch != TARGET_BRANCH or git_output("status", "--porcelain"):
        raise RuntimeError("post-training phase requires clean execution worktree")
    if args.attempt_root != ATTEMPT_ROOT or not args.attempt_root.is_dir():
        raise RuntimeError("formal attempt_001 is absent or path is not frozen")
    pointer = read_json(args.attempt_root / "contract/protocol_pointer.json")
    if (
        pointer["execution_head"] != head
        or pointer["protocol_head"] != PROTOCOL_HEAD
        or pointer["protocol_hashes"] != protocol["hashes"]
    ):
        raise RuntimeError("attempt execution/protocol provenance changed")
    baseline = protocol["resource"]["output_immutability_baseline"]
    if tree_metadata_fingerprint(CANARY_ROOT) != baseline["canary_attempt_001"]:
        raise RuntimeError("canary output fingerprint changed")
    if tree_metadata_fingerprint(MEDIUM_ROOT) != baseline["medium_attempt_001"]:
        raise RuntimeError("medium output fingerprint changed")
    processes = active_gpu_processes()
    if processes:
        raise RuntimeError(f"formal GPU exclusivity gate failed: {processes}")
    return head


def restore_model_from_checkpoint(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    step: int,
):
    payload, sidecar = load_complete_checkpoint(args.attempt_root, step)
    smpl_path = REPO_ROOT / "smpl_model/smplx/SMPLX_NEUTRAL.npz"
    if not smpl_path.exists():
        smpl_path = Path(
            "/root/autodl-tmp/canondressgs_work/mmlphuman_code/"
            "smpl_model/smplx/SMPLX_NEUTRAL.npz"
        )
    init_smpl(str(smpl_path))
    model = core.GaussianModel()
    runtime_args = core.build_runtime_args(
        protocol["runtime_bundle"],
        args.data_root,
        args.assets_root,
        args.attempt_root,
    )
    medium.restore_checkpoint_state(
        model, runtime_args, float(payload["scene_scale"]), payload
    )
    return model, runtime_args, payload, sidecar


def render_smoke_arrays(
    model,
    queries: list[dict[str, Any]],
    data_root: Path,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    background = torch.ones(3, device="cuda")
    for query in queries:
        pose_id = int(query["pose_id"])
        camera_id = int(query["camera_id"])
        dataset = core.ThumanDataset(
            datadir=str(data_root),
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
        result[f"query_{ordinal:03d}_rgb"] = (
            rgb.detach().cpu().numpy().astype(np.float32)
        )
        result[f"query_{ordinal:03d}_alpha"] = (
            alpha[..., 0].detach().cpu().numpy().astype(np.float32)
        )
        result[f"query_{ordinal:03d}_depth"] = (
            depth[..., 0].detach().cpu().numpy().astype(np.float32)
        )
    return result


def roundtrip_phase(args: argparse.Namespace, protocol: dict[str, Any]) -> int:
    configure_runtime_modules()
    execution_head = post_training_gate(args, protocol)
    model, _, payload, sidecar = restore_model_from_checkpoint(
        args, protocol, FINAL_STEP
    )
    baseline_path = Path(
        next(
            item["roundtrip_baseline"]
            for item in read_json(
                args.attempt_root / "training_logs/training_result.json"
            )["fixed96_trajectory"]["new_evaluations"]
            if int(item["step"]) == FINAL_STEP
        )
    )
    smoke_ordinals = set(medium.DIAGNOSTIC_ORDINALS)
    smoke_queries = [
        item
        for item in protocol["visual"]["queries"]
        if int(item["ordinal"]) in smoke_ordinals
    ]
    current = render_smoke_arrays(model, smoke_queries, args.data_root)
    comparisons = []
    with np.load(baseline_path, allow_pickle=False) as baseline:
        if set(baseline.files) != set(current):
            raise RuntimeError("roundtrip baseline key set changed")
        for name in sorted(current):
            expected = np.asarray(baseline[name])
            actual = current[name]
            difference = np.abs(
                expected.astype(np.float64) - actual.astype(np.float64)
            )
            comparisons.append(
                {
                    "array": name,
                    "shape": list(actual.shape),
                    "exact": bool(np.array_equal(expected, actual)),
                    "max_abs": float(difference.max(initial=0.0)),
                    "mae": float(difference.mean()),
                }
            )
    max_abs = max(item["max_abs"] for item in comparisons)
    attachments = model.surface_attachment_state_dict(cpu=True)
    result = {
        "schema_version": "subject00.formal.checkpoint_roundtrip.v1",
        "task_id": TASK_ID,
        "status": "PASS" if max_abs <= 1.0e-6 else "FAIL",
        "execution_head": execution_head,
        "protocol_head": PROTOCOL_HEAD,
        "checkpoint": sidecar,
        "gaussian_count": int(model._xyz.shape[0]),
        "attachment_count": int(attachments["face_ids"].shape[0]),
        "lbs_shape": list(attachments["lbs_weights"].shape),
        "cached_weights_exact_to_checkpoint": bool(
            torch.equal(model.get_weights.cpu(), payload["_weights"].cpu())
        ),
        "face_ids_exact_to_checkpoint": bool(
            torch.equal(
                attachments["face_ids"],
                payload["surface_attachment"]["face_ids"].cpu(),
            )
        ),
        "barycentric_exact_to_checkpoint": bool(
            torch.equal(
                attachments["barycentric"],
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
        "render_query_count": len(smoke_queries),
        "render_array_count": len(comparisons),
        "render_tolerance": 1.0e-6,
        "render_max_abs": max_abs,
        "render_all_exact": all(item["exact"] for item in comparisons),
        "render_records": comparisons,
        "legacy_grid_loads": int(model.runtime_lbs_counters["legacy_grid_loads"]),
        "spatial_weight_queries": int(
            model.runtime_lbs_counters["spatial_weight_queries"]
        ),
        "PAPER_FINAL": 0,
    }
    atomic_write_json(
        args.attempt_root / "audits/checkpoint_roundtrip.json", result
    )
    if result["status"] != "PASS":
        raise RuntimeError(f"formal checkpoint roundtrip failed: max_abs={max_abs}")
    print(
        json.dumps(
            {
                "phase": "roundtrip",
                "status": "PASS",
                "checkpoint_sha256": sidecar["sha256"],
                "render_max_abs": max_abs,
            },
            sort_keys=True,
        )
    )
    return 0


def full_queries(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    queries = []
    global_ordinal = 0
    evaluation = protocol["evaluation"]
    for quadrant in evaluation["quadrant_order"]:
        for item in evaluation["quadrants"][quadrant]["records"]:
            global_ordinal += 1
            queries.append(
                {
                    **item,
                    "quadrant": quadrant,
                    "global_ordinal": global_ordinal,
                }
            )
    if len(queries) != protocol["full_query_count"]:
        raise RuntimeError("formal full query construction count changed")
    return queries


def project_smplx_joints(model, item: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    pose = model.smpl_poses.numpy()
    joints = model.t_joints.numpy()
    parents = model.joint_parents.numpy()
    rotations = Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().astype(
        np.float32
    )
    transforms = rigid_transform_numba(rotations, joints, parents)
    world = transforms[:, :3, 3]
    rh = model.Rh.detach().cpu().numpy()
    th = model.Th.detach().cpu().numpy()
    world = np.einsum("ij,nj->ni", rh, world) + th
    w2c = item["w2c"].detach().cpu().numpy()
    k = item["K"].detach().cpu().numpy()
    homogeneous = np.concatenate(
        [world, np.ones((len(world), 1), dtype=np.float32)], axis=1
    )
    camera = (w2c @ homogeneous.T).T[:, :3]
    projected = (k @ camera.T).T
    z = projected[:, 2]
    pixels = np.full((len(world), 2), np.nan, dtype=np.float32)
    valid = z > 1.0e-6
    pixels[valid] = projected[valid, :2] / z[valid, None]
    try:
        from smplx.joint_names import JOINT_NAMES

        names = list(JOINT_NAMES[: len(world)])
    except Exception:
        names = [f"joint_{index}" for index in range(len(world))]
    return pixels, names


def padded_box(
    points: np.ndarray,
    *,
    width: int,
    height: int,
    padding: int = 14,
) -> tuple[int, int, int, int] | None:
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        return None
    x0 = max(0, int(math.floor(points[:, 0].min())) - padding)
    y0 = max(0, int(math.floor(points[:, 1].min())) - padding)
    x1 = min(width, int(math.ceil(points[:, 0].max())) + padding + 1)
    y1 = min(height, int(math.ceil(points[:, 1].max())) + padding + 1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return x0, y0, x1, y1


def face_hand_boxes(model, item: dict[str, Any]) -> list[tuple[int, int, int, int]]:
    pixels, names = project_smplx_joints(model, item)
    lowered = [name.lower() for name in names]
    groups = [
        [
            index
            for index, name in enumerate(lowered)
            if any(token in name for token in ("head", "jaw", "eye"))
        ],
        [
            index
            for index, name in enumerate(lowered)
            if "left" in name
            and any(
                token in name
                for token in ("wrist", "thumb", "index", "middle", "ring", "pinky")
            )
        ],
        [
            index
            for index, name in enumerate(lowered)
            if "right" in name
            and any(
                token in name
                for token in ("wrist", "thumb", "index", "middle", "ring", "pinky")
            )
        ],
    ]
    boxes = []
    for indices in groups:
        if not indices:
            continue
        box = padded_box(
            pixels[np.asarray(indices)],
            width=int(item["width"]),
            height=int(item["height"]),
        )
        if box is not None:
            boxes.append(box)
    return boxes


def mask_proxy_geometry(
    mask: np.ndarray,
    predicted_mask: np.ndarray,
) -> dict[str, float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return {
            "body_conforming_bias_score": float("nan"),
            "sleeve_bulk_roi_absolute_coverage_error": float("nan"),
            "hem_boundary_vertical_error_px": float("nan"),
        }
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    gt_coverage = float(mask.mean())
    pred_coverage = float(predicted_mask.mean())
    under = float((mask & ~predicted_mask).sum() / max(1, mask.sum()))
    over = float(
        ((~mask) & predicted_mask).sum() / max(1, (~mask).sum())
    )
    upper_y0 = min(y1, y0 + int(round(0.15 * height)))
    upper_y1 = min(y1, y0 + int(round(0.65 * height)))
    third = max(1, width // 3)
    sleeve_rois = [
        (slice(upper_y0, upper_y1), slice(x0, min(x1, x0 + third))),
        (slice(upper_y0, upper_y1), slice(max(x0, x1 - third), x1)),
    ]
    sleeve_errors = []
    for roi in sleeve_rois:
        if mask[roi].size:
            sleeve_errors.append(
                abs(float(mask[roi].mean()) - float(predicted_mask[roi].mean()))
            )
    central_x0 = x0 + width // 4
    central_x1 = x1 - width // 4
    lower_y0 = y0 + height // 2
    gt_roi = mask[lower_y0:y1, central_x0:central_x1]
    pred_roi = predicted_mask[lower_y0:y1, central_x0:central_x1]
    gt_rows = np.where(gt_roi.any(axis=1))[0]
    pred_rows = np.where(pred_roi.any(axis=1))[0]
    hem_error = (
        abs(float(gt_rows.max()) - float(pred_rows.max()))
        if len(gt_rows) and len(pred_rows)
        else float(height)
    )
    return {
        "body_conforming_bias_score": abs(pred_coverage - gt_coverage) + under + over,
        "sleeve_bulk_roi_absolute_coverage_error": float(
            np.mean(sleeve_errors) if sleeve_errors else float("nan")
        ),
        "hem_boundary_vertical_error_px": float(hem_error),
    }


def roi_metrics(
    predicted: torch.Tensor,
    ground_truth: torch.Tensor,
    boxes: list[tuple[int, int, int, int]],
    lpips_metric,
) -> tuple[float | None, float | None]:
    if not boxes:
        return None, None
    predicted_chw = predicted.permute(2, 0, 1)
    ground_truth_chw = ground_truth.permute(2, 0, 1)
    pred_crops = []
    gt_crops = []
    maes = []
    for x0, y0, x1, y1 in boxes:
        pred = predicted_chw[:, y0:y1, x0:x1][None]
        gt = ground_truth_chw[:, y0:y1, x0:x1][None]
        pred = F.interpolate(pred, size=(128, 128), mode="bilinear", align_corners=False)
        gt = F.interpolate(gt, size=(128, 128), mode="bilinear", align_corners=False)
        pred_crops.append(pred)
        gt_crops.append(gt)
        maes.append(float(torch.mean(torch.abs(pred - gt)).item()))
    lpips_value = None
    if lpips_metric is not None:
        lpips_value = float(
            lpips_metric(torch.cat(pred_crops), torch.cat(gt_crops)).item()
        )
    return lpips_value, float(np.mean(maes))


@torch.no_grad()
def render_metric_record(
    model,
    item: dict[str, Any],
    query: dict[str, Any],
    *,
    lpips_metric,
    lpips_error: str | None,
) -> dict[str, Any]:
    core.set_model_pose(model, item)
    background = torch.ones(3, dtype=torch.float32, device="cuda")
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
    predicted = torch.clamp(predicted, 0.0, 1.0)
    ground_truth = item["image"].clone()
    ground_truth[~item["mask"]] = background
    ground_truth[item["mask_boundary"]] = background
    predicted_metric = predicted.clone()
    predicted_metric[item["mask_boundary"]] = background
    difference = torch.abs(predicted_metric - ground_truth)
    mse = torch.mean((predicted_metric - ground_truth) ** 2)
    psnr = float(
        (-10.0 * torch.log10(torch.clamp(mse, min=1.0e-12))).item()
    )
    ssim = float(
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
        lpips_value = float(lpips_metric(pred_crop[None], gt_crop[None]).item())

    alpha_np = alpha[..., 0].detach().cpu().numpy().astype(np.float32)
    depth_np = depth[..., 0].detach().cpu().numpy().astype(np.float32)
    predicted_np = predicted_metric.detach().cpu().numpy().astype(np.float32)
    ground_truth_np = ground_truth.detach().cpu().numpy().astype(np.float32)
    mask_np = item["mask"].detach().cpu().numpy().astype(bool)
    predicted_mask = alpha_np >= 0.5
    intersection = int((predicted_mask & mask_np).sum())
    union = int((predicted_mask | mask_np).sum())
    color = medium.color_distribution_metrics(
        predicted_np, ground_truth_np, mask_np, predicted_mask
    )
    geometry = mask_proxy_geometry(mask_np, predicted_mask)
    boxes = face_hand_boxes(model, item)
    face_hand_lpips, face_hand_mae = roi_metrics(
        predicted_metric, ground_truth, boxes, lpips_metric
    )
    posed_xyz = model.get_xyz
    extent = posed_xyz.max(dim=0).values - posed_xyz.min(dim=0).values
    means2d = info.get("means2d")
    warning_messages = [str(entry.message) for entry in caught]
    finite = bool(
        np.isfinite(predicted_np).all()
        and np.isfinite(alpha_np).all()
        and np.isfinite(depth_np).all()
        and torch.isfinite(posed_xyz).all().item()
        and means2d is not None
        and torch.isfinite(means2d).all().item()
    )
    return {
        **query,
        "step": FINAL_STEP,
        "status": "PASS" if finite else "INVALID_NONFINITE",
        "rgb_mae": float(difference.mean().item()),
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips_value,
        "silhouette_iou": float(intersection / union if union else 1.0),
        "boundary_f_score": core.boundary_f_score(predicted_mask, mask_np),
        "alpha_occupancy": float((alpha_np > 1.0e-4).mean()),
        "depth_finite_ratio": float(np.isfinite(depth_np).mean()),
        "foreground_coverage": float(predicted_mask.mean()),
        **color,
        **geometry,
        "face_hand_roi_lpips": face_hand_lpips,
        "face_hand_roi_rgb_mae": face_hand_mae,
        "face_hand_roi_box_count": len(boxes),
        "face_hand_roi_source": "projected fitted per-frame SMPL-X joints",
        "render_seconds": render_seconds,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "rgb_finite": bool(np.isfinite(predicted_np).all()),
        "alpha_finite": bool(np.isfinite(alpha_np).all()),
        "depth_finite": bool(np.isfinite(depth_np).all()),
        "alpha_nonempty": bool((alpha_np > 1.0e-6).any()),
        "posed_xyz_finite": bool(torch.isfinite(posed_xyz).all().item()),
        "posed_bbox_extent": [float(value) for value in extent.cpu().tolist()],
        "body_explosion": bool(extent.max().item() >= 5.0),
        "means2d_finite": bool(
            means2d is not None and torch.isfinite(means2d).all().item()
        ),
        "warning_messages": warning_messages,
        "lpips_available": lpips_metric is not None,
        "lpips_unavailable_reason": lpips_error,
    }


METRIC_NAMES = (
    "rgb_mae",
    "psnr",
    "ssim",
    "lpips",
    "silhouette_iou",
    "boundary_f_score",
    "alpha_occupancy",
    "depth_finite_ratio",
    "foreground_coverage",
    "render_seconds",
    "peak_vram_bytes",
    "foreground_rgb_variance",
    "foreground_chroma",
    "gt_pred_color_histogram_l1",
    "garment_proxy_undercoverage",
    "garment_proxy_overcoverage",
    "body_conforming_bias_score",
    "sleeve_bulk_roi_absolute_coverage_error",
    "hem_boundary_vertical_error_px",
    "face_hand_roi_lpips",
    "face_hand_roi_rgb_mae",
)


def metric_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for name in METRIC_NAMES:
        values = np.asarray(
            [
                float(item[name])
                for item in records
                if item.get(name) is not None and math.isfinite(float(item[name]))
            ],
            dtype=np.float64,
        )
        if len(values) == 0:
            result[name] = {"count": 0}
            continue
        p05, p25, p75, p95 = np.percentile(values, [5, 25, 75, 95])
        result[name] = {
            "count": int(len(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "standard_deviation": float(np.std(values)),
            "p05": float(p05),
            "p25": float(p25),
            "p75": float(p75),
            "p95": float(p95),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return result


def grouped_statistics(
    records: list[dict[str, Any]], key: str
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record[key])].append(record)
    return {
        name: {"query_count": len(items), "metrics": metric_statistics(items)}
        for name, items in sorted(groups.items(), key=lambda pair: int(pair[0]))
    }


def validate_evaluation_resume(
    records: list[dict[str, Any]], queries: list[dict[str, Any]]
) -> None:
    if len(records) > len(queries):
        raise RuntimeError("evaluation log is longer than frozen query set")
    for index, record in enumerate(records):
        expected = queries[index]
        if (
            int(record["global_ordinal"]) != index + 1
            or int(record["pose_id"]) != int(expected["pose_id"])
            or int(record["camera_id"]) != int(expected["camera_id"])
            or record["quadrant"] != expected["quadrant"]
        ):
            raise RuntimeError(f"evaluation resume order mismatch at index {index}")


def evaluate_phase(args: argparse.Namespace, protocol: dict[str, Any]) -> int:
    configure_runtime_modules()
    execution_head = post_training_gate(args, protocol)
    roundtrip = read_json(args.attempt_root / "audits/checkpoint_roundtrip.json")
    if roundtrip["status"] != "PASS":
        raise RuntimeError("full evaluation requires passing final roundtrip")
    aggregate_path = args.attempt_root / "evaluations/final_full_aggregate.json"
    if aggregate_path.exists():
        raise RuntimeError("formal full evaluation already completed; rerun forbidden")
    records_path = args.attempt_root / "evaluations/final_full_query_records.jsonl"
    records = read_training_records(records_path)
    if records and not args.resume_evaluation:
        raise RuntimeError("partial evaluation exists; explicit --resume-evaluation required")
    queries = full_queries(protocol)
    validate_evaluation_resume(records, queries)

    model, _, _, sidecar = restore_model_from_checkpoint(
        args, protocol, FINAL_STEP
    )
    lpips_metric, lpips_error = core.make_lpips_preserving_rng()
    poses = sorted({int(item["pose_id"]) for item in queries})
    cameras = sorted({int(item["camera_id"]) for item in queries})
    dataset = core.ThumanDataset(
        datadir=str(args.data_root),
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
    expected_pairs = {
        (int(item["pose_id"]), int(item["camera_id"])) for item in queries
    }
    if not expected_pairs.issubset(mapping):
        raise RuntimeError(
            f"full evaluation inputs missing: {sorted(expected_pairs - set(mapping))[:20]}"
        )
    started = time.perf_counter()
    failures = []
    for query in queries[len(records) :]:
        pair = (int(query["pose_id"]), int(query["camera_id"]))
        try:
            item = core.prepare_item(dataset[mapping[pair]])
            record = render_metric_record(
                model,
                item,
                query,
                lpips_metric=lpips_metric,
                lpips_error=lpips_error,
            )
        except Exception as exc:
            record = {
                **query,
                "step": FINAL_STEP,
                "status": "FAILED",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(record)
        append_jsonl(records_path, record)
        records.append(record)
    passing = [item for item in records if item["status"] == "PASS"]
    failed = [item for item in records if item["status"] != "PASS"]
    by_quadrant = {}
    for quadrant in protocol["evaluation"]["quadrant_order"]:
        subset = [item for item in passing if item["quadrant"] == quadrant]
        by_quadrant[quadrant] = {
            "query_count": sum(
                item["quadrant"] == quadrant for item in records
            ),
            "successful_query_count": len(subset),
            "failed_query_count": sum(
                item["quadrant"] == quadrant for item in failed
            ),
            "metrics": metric_statistics(subset),
            "per_camera": grouped_statistics(subset, "camera_id"),
            "per_pose": grouped_statistics(subset, "pose_id"),
        }
    result = {
        "schema_version": "subject00.formal.full_evaluation.v1",
        "task_id": TASK_ID,
        "status": "PASS" if not failed else "FAIL_WITH_QUERY_ERRORS",
        "execution_head": execution_head,
        "protocol_head": PROTOCOL_HEAD,
        "checkpoint": sidecar,
        "step": FINAL_STEP,
        "query_count": len(records),
        "successful_query_count": len(passing),
        "failed_query_count": len(failed),
        "invalid_query_count": sum(
            item.get("status") == "INVALID_NONFINITE" for item in records
        ),
        "full_evaluation_png_count": 0,
        "query_records_path": str(records_path),
        "query_records_sha256": sha256_file(records_path),
        "quadrant_order": protocol["evaluation"]["quadrant_order"],
        "primary_strict_result_quadrants": protocol["evaluation"][
            "primary_strict_result_quadrants"
        ],
        "training_fit_role": protocol["evaluation"]["training_fit_role"],
        "quadrants": by_quadrant,
        "metrics_all_for_audit_only": metric_statistics(passing),
        "failed_query_records": failed,
        "lpips_available": lpips_metric is not None,
        "lpips_unavailable_reason": lpips_error,
        "face_hand_roi_source": "projected fitted per-frame SMPL-X joints",
        "representation_proxy_limitation": protocol["evaluation"][
            "representation_capacity_metric_contract"
        ]["proxy_limitation"],
        "wall_seconds": time.perf_counter() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "PAPER_FINAL": 0,
    }
    atomic_write_json(aggregate_path, result)
    if failed:
        raise RuntimeError(
            f"formal full evaluation had {len(failed)} failed/invalid queries"
        )
    print(
        json.dumps(
            {
                "phase": "evaluate",
                "status": "PASS",
                "query_count": len(records),
                "full_evaluation_png_count": 0,
            },
            sort_keys=True,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("preflight", "train", "roundtrip", "evaluate"),
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
    parser.add_argument("--attempt-root", type=Path, default=ATTEMPT_ROOT)
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--interruption-reason")
    parser.add_argument("--resume-evaluation", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_runtime_modules()
    protocol = load_protocol(args.availability_manifest)
    if args.phase == "preflight":
        result = preflight(args, protocol, allow_attempt=False)
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.phase == "train":
        return train_phase(args, protocol)
    if args.phase == "roundtrip":
        return roundtrip_phase(args, protocol)
    return evaluate_phase(args, protocol)


if __name__ == "__main__":
    raise SystemExit(main())
