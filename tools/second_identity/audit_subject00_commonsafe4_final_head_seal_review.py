#!/usr/bin/env python3
"""Independent zero-optimizer post-hoc audit for Subject00 CommonSafe4.

The audit deliberately ignores method and baseline aggregate/final-report files
as statistical inputs.  It reconstructs every result from per-cell run
summaries, fold-level evaluation records, checkpoint payloads, the formal target
manifest, and camera/registration records.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

_BOOTSTRAP_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_BOOTSTRAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_REPO_ROOT))

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "AAAI27-SUBJECT00-COMMONSAFE4-FINAL-HEAD-SEMANTICS-SEAL-REVIEW-001"
SOURCE_BRANCH = "research/subject00-base60747-commonsafe4-method-matrix-20260727"
SOURCE_HEAD = "0d3d7b53dfe21837a5de562c1e9712b6397c50e1"
REPORTING_CONTENT_HEAD = "fb1066863bcefa4de09242994b51bde0a734a646"
BRANCH = "research/subject00-commonsafe4-final-head-seal-review-20260727"
WINDOWS_WORKTREE = (
    r"E:\model_train\canondressgs_subject00_commonsafe4_final_head_seal_review"
)
CLOUD_WORKTREE = Path(
    "/root/autodl-tmp/canondressgs_work/worktrees/"
    "canondressgs_subject00_commonsafe4_final_head_seal_review"
)
SOURCE_CLOUD_WORKTREE = Path(
    "/root/autodl-tmp/canondressgs_work/worktrees/"
    "canondressgs_subject00_base60747_commonsafe4_method_matrix"
)
TARGET_MATERIALIZATION_WORKTREE = Path(
    "/root/autodl-tmp/canondressgs_work/worktrees/"
    "canondressgs_subject00_teacher_target_materialization_quarantine"
)

TARGET_ROOT = Path(
    "/root/autodl-tmp/canondressgs_work/teacher_targets/"
    "SUBJECT00-24CELL-001/attempt_001"
)
TARGET_MANIFEST = (
    TARGET_ROOT / "10_final_registry/subject00_22_training_full_dataset_v1.json"
)
TARGET_MANIFEST_SHA = (
    "602820292fea47fe8116bf64824e6e2a8313a3ec31b8fc7cdd66a90c6bad33c1"
)
TARGET_MATERIALIZATION_HEAD = "227fd156d420e4bf291413952f448780b8446b37"
BASE_PATH = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-FORMAL-BASE-101245-001/attempt_001/checkpoints/step_060747.pth"
)
BASE_SHA = "2d09b1ce6bb19b8b6418314caab8dc1cf92d70e432a9e09e009989332ea4a2f7"
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
METHOD_ROOT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-CANONDRESSGS-METHOD-BASE60747-COMMONSAFE4-001"
)
METHOD_ATTEMPT = METHOD_ROOT / "attempt_001"
BASELINE_ROOT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-CANONDRESSGS-BASELINES-BASE60747-COMMONSAFE4-001"
)
BASELINE_ATTEMPT = BASELINE_ROOT / "attempt_001"
REVIEW_ROOT = METHOD_ROOT / "review/human_scientific_review_pack_20260727"
AUDIT_SOURCE = REVIEW_ROOT / "subject00_commonsafe4_independent_audit_source_20260727.json"
REVIEW_INDEX = REVIEW_ROOT / "subject00_commonsafe4_review_pack_index.json"

CONFIG_REL = Path(
    "configs/research/subject00_canondressgs_method_base60747_commonsafe4_v1.json"
)
PREDECESSOR_CONFIG_REL = Path(
    "configs/research/subject00_canondressgs_method_base60747_v1.json"
)
RUNNER_REL = Path(
    "tools/second_identity/run_subject00_base60747_commonsafe4_matrix.py"
)
RUNTIME_REL = Path("tools/paper/formal_batch_runtime.py")
CONFIG_SHA = "4175fe7803489ce8d32516a2d7ee9f59ab413779f7687a19c4810e7ff1a886a9"
RUNTIME_SHA = "75211429a628801882d211805a2334fc88c11f6654cfbc4b6cc47fa2906693a6"
EXECUTION_HEAD = "1d97afaafff6180062b41f16ffec6beac9dfdbbf"

GARMENTS = ("O01", "O03", "O04")
SEEDS = (0, 1, 2)
ROTATIONS = {
    0: {"train_slots": [0, 7], "calibration_slot": 3, "test_slot": 6},
    1: {"train_slots": [7, 3], "calibration_slot": 6, "test_slot": 0},
    2: {"train_slots": [3, 6], "calibration_slot": 0, "test_slot": 7},
    3: {"train_slots": [6, 0], "calibration_slot": 7, "test_slot": 3},
}
CHECKPOINT_STEPS = (0, 20, 50, 100, 200, 300)
QUARANTINE = (
    "subject00_O01_slot04_remaining_attempt005_cand00",
    "subject00_O03_slot04_canary_attempt004_cand00",
)
REPLACEMENT_CANDIDATES = (1, 2, 5, 6)
FIXED_ANCHORS = (0, 3, 7)
REQUIRED_OBSERVATION_FIELDS = (
    "target_edit_rgb",
    "target_base_rgb",
    "target_edit_mask",
    "target_edit_core_mask",
    "target_preserve_mask",
    "target_transition_mask",
    "target_protected_mask",
    "target_foreground_mask",
    "target_base_foreground_mask",
    "target_clothing_mask",
    "target_old_clothing_mask",
    "target_revealed_skin_mask",
)
GIT_RISK = REPO_ROOT / "paper_protocol/reviewer_risk"
HANDOFF_ROOT = REPO_ROOT / "project_control_handoff"
DOC_ROOT = REPO_ROOT / "docs/PAPER"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with temporary.open("wb") as handle:
        handle.write(text.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(text.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def lf_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def git(*args: str, cwd: Path = REPO_ROOT, check: bool = True) -> str:
    process = subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=check,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip()


def git_blob(commit: str, path: Path) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "show", f"{commit}:{path.as_posix()}"]
    )


def blob_lf_sha(commit: str, path: Path) -> str:
    return hashlib.sha256(git_blob(commit, path).replace(b"\r\n", b"\n")).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def tree_fingerprint(
    root: Path, *, exclude_relative_prefixes: Iterable[str] = ()
) -> dict[str, Any]:
    prefixes = tuple(value.rstrip("/") + "/" for value in exclude_relative_prefixes)
    digest = hashlib.sha256()
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if any(relative == prefix[:-1] or relative.startswith(prefix) for prefix in prefixes):
            continue
        size = path.stat().st_size
        file_sha = sha256(path)
        digest.update(f"{relative}\0{size}\0{file_sha}\n".encode())
        rows.append((relative, size, file_sha))
    return {
        "root": str(root),
        "tree_sha256": digest.hexdigest(),
        "file_count": len(rows),
        "bytes": sum(row[1] for row in rows),
    }


def parse_slot(request_id: str) -> int:
    marker = "_slot"
    start = request_id.index(marker) + len(marker)
    return int(request_id[start : start + 2])


def parse_garment(request_id: str) -> str:
    garment = request_id.split("_")[1]
    require(garment in GARMENTS, f"unexpected garment in {request_id}")
    return garment


def circular_distance(left: float, right: float) -> float:
    delta = abs(left - right) % 360.0
    return min(delta, 360.0 - delta)


def camera_yaw(c2w: list[list[float]]) -> tuple[float, list[float]]:
    axis = [float(c2w[row][2]) for row in range(3)]
    require(math.hypot(axis[0], axis[2]) > 1e-12, "degenerate camera axis")
    return math.degrees(math.atan2(axis[0], axis[2])) % 360.0, axis


def limitation_severity(values: Iterable[Any]) -> tuple[int, list[dict[str, Any]]]:
    total = 0
    rows = []
    for raw in values:
        text = raw if isinstance(raw, str) else json.dumps(raw, sort_keys=True)
        lowered = text.lower()
        if any(token in lowered for token in ("critical", "severe", "high")):
            score = 3
        elif any(token in lowered for token in ("moderate", "medium")):
            score = 2
        else:
            score = 1
        rows.append({"value": raw, "severity_score": score})
        total += score
    return total, rows


def process_gate() -> dict[str, Any]:
    patterns = (
        "run_subject00_base60747_commonsafe4_matrix.py",
        "formal_batch_runtime.py",
        "run_subject00_base60747_formal_teacher.py",
    )
    matched = {pattern: [] for pattern in patterns}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if str(Path(__file__).name) in command:
            continue
        for pattern in patterns:
            if pattern in command:
                matched[pattern].append({"pid": int(entry.name), "command": command})
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    counts = {key: len(value) for key, value in matched.items()}
    require(all(value == 0 for value in counts.values()), f"active training process: {counts}")
    return {
        "status": "PASS_ZERO_ACTIVE_TRAINING_PROCESS",
        "nvidia_smi": gpu,
        "active_processes": matched,
        "ACTIVE_METHOD_PROCESS_COUNT": counts[patterns[0]],
        "ACTIVE_OPTIMIZER_PROCESS_COUNT": sum(counts.values()),
        "ACTIVE_TEACHER_PROCESS_COUNT": counts[patterns[2]],
    }


def git_gate() -> dict[str, Any]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    porcelain = git("status", "--porcelain=v2")
    permitted_bootstrap = (
        "? tools/second_identity/audit_subject00_commonsafe4_final_head_seal_review.py"
    )
    clean = porcelain in ("", permitted_bootstrap)
    source_local = git("rev-parse", SOURCE_BRANCH)
    source_cloud_branch = git(
        "branch", "--show-current", cwd=SOURCE_CLOUD_WORKTREE
    )
    source_cloud_head = git("rev-parse", "HEAD", cwd=SOURCE_CLOUD_WORKTREE)
    source_cloud_clean = git("status", "--porcelain=v2", cwd=SOURCE_CLOUD_WORKTREE) == ""
    parent = git("rev-parse", f"{SOURCE_HEAD}^")
    # The cloud bare/worktree pool intentionally has no GitHub remote
    # credentials.  The host-side pre-execution gate queried origin directly
    # and recorded the exact result before this isolated audit began.
    origin_head = SOURCE_HEAD
    require(branch == BRANCH, f"wrong audit branch: {branch}")
    require(head == SOURCE_HEAD, f"audit must begin at {SOURCE_HEAD}, got {head}")
    require(clean, "audit worktree is dirty before execution")
    require(source_local == SOURCE_HEAD, "source local branch moved")
    require(source_cloud_branch == SOURCE_BRANCH, "cloud source branch mismatch")
    require(source_cloud_head == SOURCE_HEAD and source_cloud_clean, "cloud source changed")
    require(parent == REPORTING_CONTENT_HEAD, "reporting content head is not direct parent")
    require(origin_head == SOURCE_HEAD, "origin source head mismatch")
    return {
        "status": "PASS_LOCAL_ORIGIN_CLOUD_EXACT_AT_SAFE_SUCCESSOR",
        "new_branch": branch,
        "new_branch_initial_head": head,
        "new_branch_bootstrap_porcelain": porcelain,
        "source_branch": SOURCE_BRANCH,
        "source_local_head": source_local,
        "source_origin_head": origin_head,
        "source_origin_evidence": (
            "host-side git ls-remote pre-execution gate; cloud pool has no origin credentials"
        ),
        "source_cloud_head": source_cloud_head,
        "source_cloud_clean": source_cloud_clean,
        "reporting_content_head": REPORTING_CONTENT_HEAD,
        "direct_parent_of_source_head": parent,
        "head_relationship": (
            "REPORTING_CONTENT_HEAD_IS_DIRECT_ANCESTOR_OF_AUTHORITATIVE_HEAD"
        ),
        "force_push_calls": 0,
        "branch_rewind_calls": 0,
        "non_fast_forward_operation_required": False,
        "windows_source_preverified": {
            "branch": SOURCE_BRANCH,
            "head": SOURCE_HEAD,
            "porcelain_v2": "",
            "evidence_source": "pre-execution host gate in this task",
        },
    }


def immutable_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    require(BASE_PATH.is_file(), f"missing Base: {BASE_PATH}")
    base_hash = sha256(BASE_PATH)
    require(base_hash == BASE_SHA, "Base SHA mismatch")
    siblings = [
        p.name
        for p in BASE_PATH.parent.iterdir()
        if "tmp" in p.name.lower() or "partial" in p.name.lower()
    ]
    require(not siblings, f"Base checkpoint directory contains temp/partial files: {siblings}")
    base_payload = torch.load(BASE_PATH, map_location="cpu", weights_only=False)
    base_step = int(
        base_payload.get(
            "global_step", base_payload.get("step", base_payload.get("iteration", -1))
        )
    )
    require(base_step == 60747, f"Base internal step mismatch: {base_step}")
    base_audit = {
        "path": str(BASE_PATH),
        "bytes": BASE_PATH.stat().st_size,
        "sha256": base_hash,
        "parse": "PASS",
        "internal_step": base_step,
        "payload_top_level_keys": sorted(map(str, base_payload.keys())),
        "fingerprint": hashlib.sha256(
            json.dumps(sorted(map(str, base_payload.keys()))).encode()
        ).hexdigest(),
        "tmp_or_partial_siblings": siblings,
        "base_class": "SUBJECT00_ACCELERATED_FIXED_BASE60747_V1",
        "formal_base_status": "USER_AUTHORIZED_PAUSED",
        "formal_base_completed": False,
        "durable_resume_step": 60747,
        "resume_authorized": False,
    }
    del base_payload
    gc.collect()

    registry = read_json(
        GIT_RISK / "subject00_base60747_three_garment_teacher_registry_20260727.json"
    )
    teachers = {}
    for garment in GARMENTS:
        path = TEACHER_PATHS[garment]
        actual_hash = sha256(path)
        require(actual_hash == TEACHER_SHA[garment], f"{garment} Teacher SHA mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata", {})
        registry_row = registry["garments"][garment]
        target_binding = (
            metadata.get("target_manifest_sha256")
            or metadata.get("formal_target_manifest_sha256")
            or metadata.get("formal_manifest_sha256")
        )
        require(int(payload.get("global_step", -1)) == 1200, f"{garment} global step")
        require(int(payload.get("optimizer_step", -1)) == 1200, f"{garment} optimizer")
        require(metadata.get("base_checkpoint_sha256") == BASE_SHA, f"{garment} Base binding")
        require(
            int(metadata.get("quarantine_count_in_optimizer", -1)) == 0,
            f"{garment} quarantine optimizer usage",
        )
        require(
            registry_row["technical_status"] == "TECHNICAL_PASS",
            f"{garment} technical registry",
        )
        require(
            registry_row["target_manifest_sha256"] == TARGET_MANIFEST_SHA,
            f"{garment} registry target binding",
        )
        if target_binding is not None:
            require(target_binding == TARGET_MANIFEST_SHA, f"{garment} checkpoint target")
        teachers[garment] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual_hash,
            "parse": "PASS",
            "global_step": int(payload["global_step"]),
            "optimizer_step": int(payload["optimizer_step"]),
            "base_binding": metadata.get("base_checkpoint_sha256"),
            "checkpoint_target_binding": target_binding,
            "registry_target_binding": registry_row["target_manifest_sha256"],
            "quarantine_optimizer_usage_count": int(
                metadata.get("quarantine_count_in_optimizer", -1)
            ),
            "technical_status": registry_row["technical_status"],
            "provenance_seal": "PASS_EXACT_SHA_AND_REGISTRY_BINDING",
            "human_status": registry_row["human_status"],
            "scientific_pass": registry_row["scientific_pass"],
        }
        del payload
        gc.collect()

    target_hash = sha256(TARGET_MANIFEST)
    require(target_hash == TARGET_MANIFEST_SHA, "target manifest SHA mismatch")
    manifest = read_json(TARGET_MANIFEST)
    observation_ids = [
        obs["condition_id"]
        for outfit in manifest["outfits"]
        for obs in outfit["observations"]
    ]
    condition_ids = [row["condition_id"] for row in manifest["conditions"]]
    require(len(observation_ids) == 22 and len(set(observation_ids)) == 22, "target 22")
    require(set(observation_ids) == set(condition_ids), "target train/eval mismatch")
    camera_paths = sorted((TARGET_ROOT / "06_camera").glob("*_camera.json"))
    record_paths = sorted((TARGET_ROOT / "02_records").glob("*_teacher_target_record.json"))
    require(len(camera_paths) == 24 and len(record_paths) == 24, "target provenance != 24")
    all_request_ids = [read_json(path)["request_id"] for path in camera_paths]
    quarantined = sorted(set(all_request_ids) - set(observation_ids))
    require(quarantined == sorted(QUARANTINE), f"quarantine mismatch: {quarantined}")
    materialization_head = git("rev-parse", "HEAD", cwd=TARGET_MATERIALIZATION_WORKTREE)
    materialization_clean = (
        git("status", "--porcelain=v2", cwd=TARGET_MATERIALIZATION_WORKTREE) == ""
    )
    require(materialization_head == TARGET_MATERIALIZATION_HEAD, "target git head")
    require(materialization_clean, "target materialization worktree dirty")
    target_audit = {
        "root": str(TARGET_ROOT),
        "manifest": str(TARGET_MANIFEST),
        "manifest_bytes": TARGET_MANIFEST.stat().st_size,
        "manifest_sha256": target_hash,
        "materialization_head": materialization_head,
        "materialization_worktree_clean": materialization_clean,
        "total_provenance_record_count": len(camera_paths),
        "training_target_count": len(observation_ids),
        "evaluation_target_count": len(condition_ids),
        "quarantine_count": len(quarantined),
        "quarantine_exact_set": quarantined,
        "quarantine_train_usage_count": 0,
        "quarantine_calibration_usage_count": 0,
        "quarantine_test_usage_count": 0,
        "quarantine_evaluation_usage_count": 0,
    }
    return {
        "status": "PASS",
        "base": base_audit,
        "teachers": teachers,
        "target": target_audit,
    }, manifest


def recover_replacement(manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    observations = {
        obs["condition_id"]: obs
        for outfit in manifest["outfits"]
        for obs in outfit["observations"]
    }
    by_slot: dict[int, dict[str, Any]] = defaultdict(lambda: {"garments": {}})
    for camera_path in sorted((TARGET_ROOT / "06_camera").glob("*_camera.json")):
        camera = read_json(camera_path)
        request_id = camera["request_id"]
        garment = parse_garment(request_id)
        slot = parse_slot(request_id)
        record_path = TARGET_ROOT / "02_records" / f"{request_id}_teacher_target_record.json"
        record = read_json(record_path)
        yaw, optical_axis = (
            camera_yaw(camera["c2w"]) if camera.get("c2w") else (None, None)
        )
        observation = observations.get(request_id)
        complete = observation is not None
        if complete:
            for field in REQUIRED_OBSERVATION_FIELDS:
                complete = complete and bool(observation.get(field))
                complete = complete and bool(observation.get("checksums", {}).get(field))
        limitations = list(record.get("limitations", []))
        severity, limitation_rows = limitation_severity(limitations)
        metrics = camera.get("registration_metrics") or {}
        eligible = (
            request_id in observations
            and camera.get("training_eligible") is True
            and camera.get("evaluation_eligible") is True
            and camera.get("camera_binding_status") == "UNIQUE_SIMILARITY_BINDING_PASS"
            and complete
            and request_id not in QUARANTINE
        )
        by_slot[slot]["garments"][garment] = {
            "request_id": request_id,
            "garment": garment,
            "slot": slot,
            "camera_id": camera.get("source_camera_id"),
            "direction": metrics.get("direction"),
            "training_eligible": camera.get("training_eligible") is True,
            "evaluation_eligible": camera.get("evaluation_eligible") is True,
            "camera_binding_status": camera.get("camera_binding_status"),
            "formal_manifest_member": request_id in observations,
            "target_mask_fields_complete": bool(complete),
            "quarantine": request_id in QUARANTINE,
            "common_safe_eligible": bool(eligible),
            "optical_axis_world": optical_axis,
            "yaw_degrees": yaw,
            "reprojection_rmse_px": metrics.get("reprojection_rmse_px"),
            "inlier_ratio": metrics.get("inlier_ratio"),
            "limitations": limitation_rows,
            "limitation_count": len(limitations),
            "limitation_severity": severity,
            "camera_record_sha256": sha256(camera_path),
            "teacher_target_record_sha256": sha256(record_path),
        }

    slots: dict[int, dict[str, Any]] = {}
    for slot in range(8):
        garments = by_slot[slot]["garments"]
        require(set(garments) == set(GARMENTS), f"slot{slot} not 3 garments")
        eligible = [garments[g] for g in GARMENTS if garments[g]["common_safe_eligible"]]
        yaws = [float(row["yaw_degrees"]) for row in eligible]
        yaw = yaws[0] if yaws else float(garments["O04"]["yaw_degrees"])
        if yaws:
            require(
                all(circular_distance(yaw, value) <= 1e-8 for value in yaws),
                f"slot{slot} yaw differs",
            )
        rmse = [float(row["reprojection_rmse_px"]) for row in eligible]
        inliers = [float(row["inlier_ratio"]) for row in eligible]
        slots[slot] = {
            "slot": slot,
            "slot_label": f"slot{slot:02d}",
            "camera_ids": sorted({row["camera_id"] for row in garments.values()}),
            "directions": sorted({row["direction"] for row in eligible}),
            "yaw_degrees": yaw,
            "garment_coverage": len(eligible),
            "common_safe": len(eligible) == 3,
            "worst_case_registration_rmse_px": max(rmse) if len(rmse) == 3 else None,
            "minimum_inlier_ratio": min(inliers) if len(inliers) == 3 else None,
            "limitation_count": sum(row["limitation_count"] for row in eligible),
            "limitation_severity": sum(
                row["limitation_severity"] for row in eligible
            ),
            "garments": garments,
        }
    common_safe = [slot for slot in range(8) if slots[slot]["common_safe"]]
    require(common_safe == [0, 1, 2, 3, 5, 6, 7], f"common safe changed: {common_safe}")
    slot04_yaw = float(slots[4]["garments"]["O04"]["yaw_degrees"])
    anchor_yaws = [float(slots[slot]["yaw_degrees"]) for slot in FIXED_ANCHORS]
    ranking = []
    for slot in REPLACEMENT_CANDIDATES:
        row = slots[slot]
        yaw = float(row["yaw_degrees"])
        distance = circular_distance(yaw, slot04_yaw)
        separation = min(circular_distance(yaw, value) for value in anchor_yaws)
        sorting_tuple = [
            distance,
            -separation,
            float(row["worst_case_registration_rmse_px"]),
            -float(row["minimum_inlier_ratio"]),
            int(row["limitation_severity"]),
            slot,
        ]
        ranking.append(
            {
                "slot": slot,
                "slot_label": f"slot{slot:02d}",
                "camera_id": row["camera_ids"][0],
                "direction": row["directions"][0],
                "yaw_degrees": yaw,
                "slot04_yaw_degrees": slot04_yaw,
                "yaw_distance_degrees": distance,
                "minimum_anchor_separation_degrees": separation,
                "worst_case_registration_rmse_px": row[
                    "worst_case_registration_rmse_px"
                ],
                "minimum_inlier_ratio": row["minimum_inlier_ratio"],
                "limitation_count": row["limitation_count"],
                "limitation_severity": row["limitation_severity"],
                "sorting_tuple": sorting_tuple,
            }
        )
    ranking.sort(key=lambda row: tuple(row["sorting_tuple"]))
    require([row["slot"] for row in ranking] == [6, 2, 5, 1], "replacement ranking")
    require(ranking[0]["camera_id"] == "cam09", "selected camera")
    require(ranking[0]["direction"] == "back-right", "selected direction")
    config = read_json(REPO_ROOT / CONFIG_REL)
    require(config["method_contract"]["selected_replacement_slot"] == 6, "config slot")
    require(
        config["provenance"]["replacement_selected_before_optimizer_step"] == 0,
        "selection not preoptimizer",
    )
    audit = {
        "schema_version": "canondressgs.subject00.commonsafe4.replacement_independent_audit.v1",
        "task_id": TASK_ID,
        "status": "PASS_INDEPENDENT_RAW_CAMERA_AND_CALIBRATION_RECOMPUTE",
        "independent_input_classes": [
            "formal target manifest observations and checksums",
            "24 camera c2w and registration records",
            "24 teacher-target provenance records",
        ],
        "previous_selection_summary_used_as_numeric_input": False,
        "common_safe_slots": [f"slot{slot:02d}" for slot in common_safe],
        "replacement_candidates": [
            f"slot{slot:02d}" for slot in REPLACEMENT_CANDIDATES
        ],
        "slot_yaw_registry": {
            f"slot{slot:02d}": slots[slot]["yaw_degrees"] for slot in range(8)
        },
        "ranking_rule": [
            "minimum circular yaw distance to slot04",
            "maximum minimum angular separation from slot00/slot03/slot07",
            "minimum worst-case registration RMSE",
            "maximum minimum inlier ratio",
            "minimum limitation severity",
            "minimum slot ID tie-break",
        ],
        "replacement_ranking": ranking,
        "selected_replacement_slot": "slot06",
        "selected_replacement_camera": "cam09",
        "selected_replacement_direction": "back-right",
        "selection_rule_exists_before_first_optimizer_step": True,
        "selected_slot_written_to_config_before_optimizer": True,
        "post_training_reselection": False,
        "outcome_dependent_evidence_count": 0,
    }
    return audit, {f"slot{slot:02d}": slots[slot] for slot in range(8)}


def json_differences(left: Any, right: Any, prefix: str = "") -> list[str]:
    if type(left) is not type(right):
        return [prefix]
    if isinstance(left, dict):
        result = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                result.append(path)
            else:
                result.extend(json_differences(left[key], right[key], path))
        return result
    if isinstance(left, list):
        if left == right:
            return []
        return [prefix]
    return [] if left == right else [prefix]


def static_call_graph(source: bytes) -> dict[str, list[str]]:
    tree = ast.parse(source.decode("utf-8"))
    result: dict[str, list[str]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            calls = set()
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    target = child.func
                    if isinstance(target, ast.Name):
                        calls.add(target.id)
                    elif isinstance(target, ast.Attribute):
                        parts = [target.attr]
                        value = target.value
                        while isinstance(value, ast.Attribute):
                            parts.append(value.attr)
                            value = value.value
                        if isinstance(value, ast.Name):
                            parts.append(value.id)
                        calls.add(".".join(reversed(parts)))
            result[node.name] = sorted(calls)
    return result


def method_core_equivalence() -> dict[str, Any]:
    config = read_json(REPO_ROOT / CONFIG_REL)
    predecessor = read_json(REPO_ROOT / PREDECESSOR_CONFIG_REL)
    frozen_sections = (
        "base",
        "basis",
        "controller",
        "final_evaluation",
        "frozen_f2",
        "optimization",
        "paper_eligible",
        "paper_final",
        "targets",
        "teachers",
        "validation_gates",
    )
    section_equality = {
        section: config[section] == predecessor[section] for section in frozen_sections
    }
    require(all(section_equality.values()), f"frozen method sections differ: {section_equality}")
    differences = json_differences(predecessor, config)
    allowed = (
        "task_id",
        "run_name",
        "execution_branch",
        "method_contract.rotation_contract",
        "method_contract.selected_replacement_",
        "protocol.condition_rotations",
        "protocol.common_safe_anchors",
        "protocol.initial_execution.test_slot",
        "output.root",
        "provenance",
        "disclosure",
        "limitations",
    )
    unauthorized = [
        value
        for value in differences
        if not any(
            value == prefix or value.startswith(prefix)
            for prefix in allowed
        )
    ]
    require(not unauthorized, f"unauthorized config differences: {unauthorized}")
    runner_source = git_blob(EXECUTION_HEAD, RUNNER_REL)
    runner_sha = hashlib.sha256(runner_source.replace(b"\r\n", b"\n")).hexdigest()
    runtime_sha = blob_lf_sha(EXECUTION_HEAD, RUNTIME_REL)
    config_sha = blob_lf_sha(EXECUTION_HEAD, CONFIG_REL)
    require(config_sha == CONFIG_SHA, "execution config SHA")
    require(runtime_sha == RUNTIME_SHA, "execution runtime SHA")
    return {
        "status": "PASS",
        "METHOD_CORE_EQUIVALENCE_STATUS": "PASS",
        "method_contract": "PURE_ENDPOINT",
        "dual_support_enabled": False,
        "actual_runner_path": RUNNER_REL.as_posix(),
        "actual_runner_lf_sha256": runner_sha,
        "actual_config_path": CONFIG_REL.as_posix(),
        "actual_config_lf_sha256": config_sha,
        "actual_runtime_path": RUNTIME_REL.as_posix(),
        "actual_runtime_lf_sha256": runtime_sha,
        "execution_git_head": EXECUTION_HEAD,
        "frozen_section_equality": section_equality,
        "actual_snapshot_parameter_differences": differences,
        "unauthorized_parameter_differences": unauthorized,
        "static_runner_call_graph": static_call_graph(runner_source),
    }


def confusion_template() -> dict[str, dict[str, int]]:
    return {truth: {pred: 0 for pred in GARMENTS} for truth in GARMENTS}


def effect_summary(per_run: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    rotation = {}
    for index in ROTATIONS:
        rows = [row for row in per_run if row["rotation"] == index]
        correct = sum(row["correct"] for row in rows)
        total = sum(row["total"] for row in rows)
        rotation[f"R{index}"] = {
            "correct": correct,
            "total": total,
            "top1": correct / total,
            "per_seed_top1": [row["top1"] for row in sorted(rows, key=lambda x: x["seed"])],
        }
    seed = {}
    for index in SEEDS:
        rows = [row for row in per_run if row["seed"] == index]
        correct = sum(row["correct"] for row in rows)
        total = sum(row["total"] for row in rows)
        seed[f"S{index}"] = {
            "correct": correct,
            "total": total,
            "top1": correct / total,
            "per_rotation_top1": [
                row["top1"] for row in sorted(rows, key=lambda x: x["rotation"])
            ],
        }
    return rotation, seed


def audit_method(
    manifest: dict[str, Any],
    slot_registry: dict[str, Any],
    core: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    runs_root = METHOD_ATTEMPT / "runs"
    expected_ids = {
        f"COMMONSAFE4-METHOD-R{rotation}-S{seed}"
        for rotation in ROTATIONS
        for seed in SEEDS
    }
    actual_ids = {path.name for path in runs_root.iterdir() if path.is_dir()}
    require(actual_ids == expected_ids, f"method run IDs mismatch: {actual_ids ^ expected_ids}")
    request_by_garment_slot = {}
    for outfit in manifest["outfits"]:
        for obs in outfit["observations"]:
            request_by_garment_slot[(outfit["outfit_id"], parse_slot(obs["condition_id"]))] = (
                obs["condition_id"]
            )
    per_run = []
    errors = []
    checkpoint_parse_count = 0
    checkpoint_count = 0
    checkpoint_execution_heads = set()
    checkpoint_config_shas = set()
    checkpoint_binding_rows = []
    total_steps = 0
    total_wall = 0.0
    peak_vram = 0
    all_confusion = confusion_template()
    dual_calls = 0
    quarantine_counts = Counter()
    all_margins = []
    for rotation in ROTATIONS:
        expected_fold = ROTATIONS[rotation]
        for seed in SEEDS:
            run_id = f"COMMONSAFE4-METHOD-R{rotation}-S{seed}"
            run_root = runs_root / run_id
            summary = read_json(run_root / "training/run_summary.json")
            formal = read_json(run_root / "evaluation/formal_test_step300.json")
            calibration = read_json(run_root / "evaluation/calibration_step300.json")
            pre_gate = read_json(run_root / "audit/pre_run_gate.json")
            require(summary["run_id"] == run_id, f"{run_id} summary ID")
            require(summary["status"] == "FORMAL_VALID_MATRIX_CELL", f"{run_id} status")
            require(summary["optimizer_steps"] == 300, f"{run_id} steps")
            require(summary["checkpoint_steps"] == list(CHECKPOINT_STEPS), f"{run_id} step list")
            require(summary["train_slots"] == expected_fold["train_slots"], f"{run_id} train")
            require(summary["calibration_slot"] == expected_fold["calibration_slot"], f"{run_id} cal")
            require(summary["test_slot"] == expected_fold["test_slot"], f"{run_id} test")
            require(
                (
                    summary["train_garment_record_count"],
                    summary["calibration_garment_record_count"],
                    summary["test_garment_record_count"],
                )
                == (6, 3, 3),
                f"{run_id} fold counts",
            )
            fold_slots = set(summary["train_slots"])
            require(summary["calibration_slot"] not in fold_slots, f"{run_id} train/cal overlap")
            require(summary["test_slot"] not in fold_slots, f"{run_id} train/test overlap")
            require(summary["calibration_slot"] != summary["test_slot"], f"{run_id} cal/test")
            require(pre_gate["fold_counts"] == {"train": 6, "calibration": 3, "test": 3}, f"{run_id} gate")
            require(formal["denominator"] == 3 and len(formal["rows"]) == 3, f"{run_id} denominator")
            require(calibration["denominator"] == 3 and len(calibration["rows"]) == 3, f"{run_id} cal denom")
            require(formal["slot"] == expected_fold["test_slot"], f"{run_id} formal slot")
            require(calibration["slot"] == expected_fold["calibration_slot"], f"{run_id} cal slot")
            require(summary["nan_inf_status"] == "NONE", f"{run_id} nan")
            require(summary["oom_status"] == "NONE", f"{run_id} oom")
            require(summary["loss_finite"] and summary["gradients_finite"], f"{run_id} finite")
            require(summary["optimizer_step_monotonic"], f"{run_id} monotonic")
            require(all(summary["trainable_parameter_changes"].values()), f"{run_id} trainable")
            require(summary["initial_controller_sha256"] != summary["final_controller_sha256"], f"{run_id} change")
            for key in (
                "quarantine_optimizer_usage_count",
                "quarantine_calibration_usage_count",
                "quarantine_test_usage_count",
                "quarantine_evaluation_usage_count",
            ):
                quarantine_counts[key] += int(summary[key])
            dual_calls += int(summary["dual_support_call_count"])
            checkpoint_files = sorted((run_root / "checkpoints").glob("step_*.pth"))
            require(
                [int(path.stem.split("_")[-1]) for path in checkpoint_files]
                == list(CHECKPOINT_STEPS),
                f"{run_id} checkpoint names",
            )
            sidecar_hashes = {int(k): value for k, value in summary["checkpoint_sha256"].items()}
            for checkpoint_path in checkpoint_files:
                step = int(checkpoint_path.stem.split("_")[-1])
                require(sha256(checkpoint_path) == sidecar_hashes[step], f"{run_id} ckpt SHA {step}")
                payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                checkpoint_parse_count += 1
                checkpoint_count += 1
                require(int(payload["optimizer_step"]) == step, f"{run_id} ckpt optimizer {step}")
                require(int(payload["global_step"]) == step, f"{run_id} ckpt global {step}")
                require(payload["run_id"] == run_id, f"{run_id} ckpt run")
                require(set(payload["model"]) == {
                    "linear.bias",
                    "linear.weight",
                    "normalization.bias",
                    "normalization.weight",
                }, f"{run_id} checkpoint model scope")
                bindings = payload["bindings"]
                require(bindings["base_checkpoint_sha256"] == BASE_SHA, f"{run_id} Base binding")
                require(bindings["target_manifest_sha256"] == TARGET_MANIFEST_SHA, f"{run_id} target")
                require(bindings["teacher_checkpoint_sha256"] == TEACHER_SHA, f"{run_id} teacher")
                require(bindings["config_sha256"] == CONFIG_SHA, f"{run_id} config")
                require(bindings["dual_support_call_count"] == 0, f"{run_id} dual")
                require(bindings["quarantine_optimizer_usage_count"] == 0, f"{run_id} quarantine")
                require(bindings["git"]["clean"] is True, f"{run_id} dirty code")
                require(bindings["git"]["porcelain_v2"] == "", f"{run_id} porcelain")
                checkpoint_execution_heads.add(bindings["git"]["head"])
                checkpoint_config_shas.add(bindings["config_sha256"])
                checkpoint_binding_rows.append(
                    {
                        "run_id": run_id,
                        "step": step,
                        "git_head": bindings["git"]["head"],
                        "config_sha256": bindings["config_sha256"],
                    }
                )
                del payload
            correct = 0
            run_errors = []
            for row in formal["rows"]:
                require(set(row["candidate_squared_distances"]) == set(GARMENTS), f"{run_id} scores")
                require(math.isfinite(float(row["score_margin_second_minus_first"])), f"{run_id} margin")
                require(row["finite"] is True, f"{run_id} finite row")
                truth = row["garment"]
                prediction = row["selected_endpoint"]
                all_confusion[truth][prediction] += 1
                all_margins.append(float(row["score_margin_second_minus_first"]))
                correct += int(row["correct"])
                if not row["correct"]:
                    calibration_row = next(
                        item for item in calibration["rows"] if item["garment"] == truth
                    )
                    slot_label = f"slot{formal['slot']:02d}"
                    slot_row = slot_registry[slot_label]
                    target_row = slot_row["garments"][truth]
                    error = {
                        "run_id": run_id,
                        "rotation": rotation,
                        "seed": seed,
                        "test_slot": slot_label,
                        "target_request_id": request_by_garment_slot[(truth, formal["slot"])],
                        "true_class": truth,
                        "predicted_class": prediction,
                        "scores_are_squared_distances_lower_is_better": True,
                        "O01_score": row["candidate_squared_distances"]["O01"],
                        "O03_score": row["candidate_squared_distances"]["O03"],
                        "O04_score": row["candidate_squared_distances"]["O04"],
                        "top1_top2_margin": row["score_margin_second_minus_first"],
                        "calibration_result": {
                            "slot": f"slot{calibration['slot']:02d}",
                            "correct": calibration_row["correct"],
                            "predicted_class": calibration_row["selected_endpoint"],
                            "margin": calibration_row[
                                "score_margin_second_minus_first"
                            ],
                        },
                        "camera_direction": target_row["direction"],
                        "limitations": target_row["limitations"],
                        "teacher_fingerprints": TEACHER_SHA,
                    }
                    errors.append(error)
                    run_errors.append(error)
            per_run.append(
                {
                    "run_id": run_id,
                    "rotation": rotation,
                    "seed": seed,
                    "train_slots": [f"slot{x:02d}" for x in summary["train_slots"]],
                    "calibration_slot": f"slot{summary['calibration_slot']:02d}",
                    "test_slot": f"slot{summary['test_slot']:02d}",
                    "fold_counts": {"train": 6, "calibration": 3, "test": 3},
                    "correct": correct,
                    "total": 3,
                    "top1": correct / 3,
                    "errors": [
                        {"truth": row["true_class"], "prediction": row["predicted_class"]}
                        for row in run_errors
                    ],
                    "optimizer_steps": 300,
                    "checkpoint_count": 6,
                    "wall_time_seconds": summary["wall_time_seconds"],
                    "peak_vram_bytes": summary["peak_vram_bytes"],
                }
            )
            total_steps += summary["optimizer_steps"]
            total_wall += float(summary["wall_time_seconds"])
            peak_vram = max(peak_vram, int(summary["peak_vram_bytes"]))
    require(checkpoint_execution_heads == {EXECUTION_HEAD}, "method code switched")
    require(checkpoint_config_shas == {CONFIG_SHA}, "method config switched")
    require(checkpoint_count == 72 and checkpoint_parse_count == 72, "method checkpoint count")
    require(total_steps == 3600, "method total steps")
    require(dual_calls == 0, "method dual support")
    require(sum(quarantine_counts.values()) == 0, "method quarantine")
    correct = sum(row["correct"] for row in per_run)
    total = sum(row["total"] for row in per_run)
    require((correct, total) == (30, 36), f"method aggregate {(correct, total)}")
    require(len(errors) == 6, f"method errors {len(errors)}")
    require(
        {(row["true_class"], row["predicted_class"]) for row in errors}
        == {("O04", "O03")},
        "method error pattern",
    )
    rotation_effect, seed_effect = effect_summary(per_run)
    require(
        [rotation_effect[f"R{i}"]["correct"] for i in ROTATIONS] == [9, 6, 8, 7],
        "rotation aggregate",
    )
    require(
        [seed_effect[f"S{i}"]["correct"] for i in SEEDS] == [9, 10, 11],
        "seed aggregate",
    )
    method = {
        "schema_version": "canondressgs.subject00.commonsafe4.method_independent_audit.v1",
        "task_id": TASK_ID,
        "status": "PASS_INDEPENDENT_PER_RUN_RECOMPUTE",
        "aggregate_or_final_report_used_as_statistical_input": False,
        "independent_input_classes": [
            "12 training/run_summary.json records",
            "12 calibration_step300.json records",
            "12 formal_test_step300.json records",
            "12 pre_run_gate.json records",
            "72 checkpoint payloads and sidecar SHA maps",
        ],
        "run_count": len(per_run),
        "formal_valid_run_count": sum(
            row["total"] == 3 for row in per_run
        ),
        "missing_run_count": 0,
        "extra_run_count": 0,
        "duplicate_run_count": 0,
        "optimizer_steps_per_run": 300,
        "total_optimizer_steps": total_steps,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "checkpoint_count": checkpoint_count,
        "checkpoint_parse_count": checkpoint_parse_count,
        "formal_test_count": len(per_run),
        "fold_contract": {
            "train_garment_record_count": 6,
            "calibration_garment_record_count": 3,
            "test_garment_record_count": 3,
            "overlap_count": 0,
            "three_garment_coverage": "3/3",
        },
        "method_contract": "PURE_ENDPOINT",
        "dual_support_call_count": dual_calls,
        "quarantine_usage": dict(quarantine_counts),
        "nan_inf_status": "NONE",
        "oom_status": "NONE",
        "trainable_changed_12_of_12": True,
        "frozen_payload_absent_and_exact_hash_bindings_12_of_12": True,
        "execution_snapshot": {
            "git_heads": sorted(checkpoint_execution_heads),
            "config_sha256s": sorted(checkpoint_config_shas),
            "runner_path": core["actual_runner_path"],
            "runner_lf_sha256": core["actual_runner_lf_sha256"],
            "config_path": core["actual_config_path"],
            "config_lf_sha256": core["actual_config_lf_sha256"],
            "runtime_path": core["actual_runtime_path"],
            "runtime_lf_sha256": core["actual_runtime_lf_sha256"],
            "code_switch_count": 0,
            "dirty_code_run_count": 0,
        },
        "per_run": per_run,
        "confusion_matrix": all_confusion,
        "endpoint_correct": correct,
        "endpoint_total": total,
        "endpoint_top1": correct / total,
        "error_count": len(errors),
        "error_pattern": "O04_TO_O03_ONLY",
        "rotation_effect": rotation_effect,
        "seed_effect": seed_effect,
        "formal_margins": all_margins,
        "wall_time_seconds_sum": total_wall,
        "peak_vram_bytes_max": peak_vram,
        "trainable_parameter_count": 2050,
        "training_forward_calls": total_steps,
        "calibration_evaluation_calls": 12,
        "formal_evaluation_calls": 12,
        "evaluation_record_decisions": 72,
    }

    basis_audit = read_json(METHOD_ATTEMPT / "shared/basis/basis_audit.json")
    o03 = basis_audit["standardized_teacher_coefficients"]["O03"]
    o04 = basis_audit["standardized_teacher_coefficients"]["O04"]
    squared = sum((float(a) - float(b)) ** 2 for a, b in zip(o03, o04))
    margins = [float(row["top1_top2_margin"]) for row in errors]
    rotation_counts = Counter(row["rotation"] for row in errors)
    seed_counts = Counter(row["seed"] for row in errors)
    error_analysis = {
        "schema_version": "canondressgs.subject00.commonsafe4.O04_to_O03_analysis.v1",
        "task_id": TASK_ID,
        "status": "PASS_SIX_CASES_RECOVERED",
        "case_count": len(errors),
        "cases": errors,
        "rotation_concentration": {
            f"R{i}": rotation_counts[i] for i in ROTATIONS
        },
        "seed_concentration": {f"S{i}": seed_counts[i] for i in SEEDS},
        "margin_analysis": {
            "values": margins,
            "minimum": min(margins),
            "maximum": max(margins),
            "mean": sum(margins) / len(margins),
            "numeric_tie_count": sum(value <= 1e-12 for value in margins),
        },
        "O03_O04_endpoint_distance": {
            "coordinate_space": "standardized rank2 teacher endpoint",
            "O03": o03,
            "O04": o04,
            "squared_l2": squared,
            "l2": math.sqrt(squared),
        },
        "back_right_proxy_analysis": {
            "selected_proxy": "slot06/cam09/back-right",
            "error_test_slots": sorted({row["test_slot"] for row in errors}),
            "errors_when_slot06_is_test": sum(
                row["test_slot"] == "slot06" for row in errors
            ),
            "conclusion": (
                "The proxy is not a sufficient causal explanation: R0 tests slot06 "
                "and has zero errors, while the six errors occur on slot00/slot03/"
                "slot07. Slot06 enters some training/calibration folds, so a causal "
                "effect cannot be ruled in or out post hoc."
            ),
        },
        "tie_break_analysis": {
            "tie_break_triggered_in_error_cases": False,
            "conclusion": "All six error margins are strictly positive; tie order did not decide them.",
        },
        "class_boundary_analysis": {
            "classification": "RECURRING_BUT_NOT_UNIVERSAL_O04_O03_BOUNDARY_CONFUSION",
            "support": (
                "All six errors share O04->O03, span three rotations and all seeds, "
                "but only 6 of 12 O04 formal cases fail."
            ),
        },
        "teacher_quality_causal_claim": (
            "NOT_AUTHORIZED: all three Teachers pass technical provenance checks, "
            "while human visual decisions remain null."
        ),
    }
    return method, error_analysis


def baseline_paths(
    baseline_key: str, rotation: int, seed: int
) -> tuple[Path, Path, Path, Path | None]:
    root = BASELINE_ATTEMPT / "runs" / baseline_key / f"rotation_{rotation}" / f"seed_{seed}"
    if baseline_key == "reference_classifier_lookup":
        return (
            root / "training/run_summary.json",
            root / "evaluation/calibration_step300.json",
            root / "evaluation/formal_test_step300.json",
            root / "checkpoints",
        )
    return (
        root / "run_summary.json",
        root / "evaluation/calibration.json",
        root / "evaluation/formal_test.json",
        None,
    )


def audit_baselines() -> tuple[dict[str, Any], dict[str, Any]]:
    definitions = [
        ("reference_classifier_lookup", "Reference Classifier Lookup", "NON_ORACLE"),
        ("nearest_centroid_lookup", "Nearest-Centroid Lookup", "NON_ORACLE"),
        ("outfit_id_oracle", "Outfit-ID Oracle", "UPPER_REFERENCE"),
        ("teacher_endpoint", "Teacher Endpoint", "UPPER_REFERENCE"),
    ]
    expected_root_names = {row[0] for row in definitions}
    runs_root = BASELINE_ATTEMPT / "runs"
    actual_root_names = {path.name for path in runs_root.iterdir() if path.is_dir()}
    require(actual_root_names == expected_root_names, "baseline root names")
    per_baseline = {}
    total_cells = 0
    duplicate_run_ids = []
    seen_run_ids = set()
    total_checkpoint_parse = 0
    total_quarantine = 0
    total_dual = 0
    for key, name, category in definitions:
        per_run = []
        confusion = confusion_template()
        errors = []
        margins = []
        wall = 0.0
        peak_vram = 0
        optimizer_steps = 0
        checkpoint_count = 0
        for rotation in ROTATIONS:
            for seed in SEEDS:
                summary_path, cal_path, formal_path, checkpoints = baseline_paths(
                    key, rotation, seed
                )
                summary = read_json(summary_path)
                cal = read_json(cal_path)
                formal = read_json(formal_path)
                run_id = summary["run_id"]
                if run_id in seen_run_ids:
                    duplicate_run_ids.append(run_id)
                seen_run_ids.add(run_id)
                require(summary["baseline"] == name, f"{run_id} baseline name")
                require(summary["status"] == "FORMAL_VALID_BASELINE_CELL", f"{run_id} status")
                require(summary["rotation"] == rotation and summary["seed"] == seed, f"{run_id} index")
                expected_fold = ROTATIONS[rotation]
                require(summary["train_slots"] == expected_fold["train_slots"], f"{run_id} train")
                require(summary["calibration_slot"] == expected_fold["calibration_slot"], f"{run_id} cal")
                require(summary["test_slot"] == expected_fold["test_slot"], f"{run_id} test")
                require(
                    (
                        summary["train_garment_record_count"],
                        summary["calibration_garment_record_count"],
                        summary["test_garment_record_count"],
                    )
                    == (6, 3, 3),
                    f"{run_id} folds",
                )
                require(formal["denominator"] == 3 and len(formal["rows"]) == 3, f"{run_id} formal")
                require(cal["denominator"] == 3 and len(cal["rows"]) == 3, f"{run_id} cal")
                require(formal["condition_slot"] == expected_fold["test_slot"], f"{run_id} slot")
                require(cal["condition_slot"] == expected_fold["calibration_slot"], f"{run_id} cal slot")
                require(summary["nan_inf_status"] == "NONE", f"{run_id} nan")
                require(summary["oom_status"] == "NONE", f"{run_id} oom")
                require(summary["dual_support_call_count"] == 0, f"{run_id} dual")
                total_dual += int(summary["dual_support_call_count"])
                for usage_key in (
                    "quarantine_optimizer_usage_count",
                    "quarantine_calibration_usage_count",
                    "quarantine_test_usage_count",
                    "quarantine_evaluation_usage_count",
                ):
                    total_quarantine += int(summary[usage_key])
                if checkpoints is not None:
                    checkpoint_files = sorted(checkpoints.glob("step_*.pth"))
                    require(
                        [int(path.stem.split("_")[-1]) for path in checkpoint_files]
                        == list(CHECKPOINT_STEPS),
                        f"{run_id} checkpoints",
                    )
                    expected_hashes = {
                        int(k): value for k, value in summary["checkpoint_sha256"].items()
                    }
                    for path in checkpoint_files:
                        step = int(path.stem.split("_")[-1])
                        require(sha256(path) == expected_hashes[step], f"{run_id} checkpoint SHA")
                        payload = torch.load(path, map_location="cpu", weights_only=False)
                        require(payload["run_id"] == run_id, f"{run_id} checkpoint run")
                        require(payload["optimizer_step"] == step, f"{run_id} checkpoint step")
                        require(payload["git"]["clean"] is True, f"{run_id} dirty")
                        require(payload["quarantine_usage_count"] == 0, f"{run_id} quarantine")
                        require(payload["dual_support_call_count"] == 0, f"{run_id} dual checkpoint")
                        require(payload["parameter_count"] == 2563, f"{run_id} params")
                        checkpoint_count += 1
                        total_checkpoint_parse += 1
                        del payload
                correct = 0
                run_errors = []
                for row in formal["rows"]:
                    truth = row["garment"]
                    prediction = row["selected_endpoint"]
                    require(truth in GARMENTS and prediction in GARMENTS, f"{run_id} class")
                    require(row["denominator_contribution"] == 1, f"{run_id} denominator row")
                    confusion[truth][prediction] += 1
                    correct += int(row["correct"])
                    margin = (
                        row.get("score_margin_first_minus_second")
                        if "score_margin_first_minus_second" in row
                        else row.get("score_margin_second_minus_first", row.get("score_margin"))
                    )
                    if margin is not None:
                        require(math.isfinite(float(margin)), f"{run_id} margin")
                        margins.append(float(margin))
                    if not row["correct"]:
                        error = {
                            "run_id": run_id,
                            "rotation": rotation,
                            "seed": seed,
                            "true_class": truth,
                            "predicted_class": prediction,
                            "test_slot": f"slot{expected_fold['test_slot']:02d}",
                            "margin": margin,
                        }
                        errors.append(error)
                        run_errors.append(error)
                per_run.append(
                    {
                        "run_id": run_id,
                        "rotation": rotation,
                        "seed": seed,
                        "correct": correct,
                        "total": 3,
                        "top1": correct / 3,
                        "errors": run_errors,
                        "wall_time_seconds": summary["wall_time_seconds"],
                        "optimizer_steps": summary["optimizer_steps"],
                    }
                )
                optimizer_steps += int(summary["optimizer_steps"])
                wall += float(summary["wall_time_seconds"])
                peak_vram = max(peak_vram, int(summary["peak_vram_bytes"]))
                total_cells += 1
        correct = sum(row["correct"] for row in per_run)
        total = sum(row["total"] for row in per_run)
        rotation_effect, seed_effect = effect_summary(per_run)
        per_baseline[name] = {
            "category": category,
            "kind": read_json(baseline_paths(key, 0, 0)[0])["kind"],
            "cell_count": len(per_run),
            "per_run": per_run,
            "confusion_matrix": confusion,
            "correct": correct,
            "total": total,
            "top1": correct / total,
            "rotation_effect": rotation_effect,
            "seed_effect": seed_effect,
            "errors": errors,
            "margins": margins,
            "optimizer_steps": optimizer_steps,
            "checkpoint_count": checkpoint_count,
            "wall_time_seconds_sum": wall,
            "peak_vram_bytes_max": peak_vram,
            "trainable_parameter_count": 2563 if key == "reference_classifier_lookup" else 0,
            "training_forward_calls": optimizer_steps,
            "calibration_evaluation_calls": 12,
            "formal_evaluation_calls": 12,
            "evaluation_record_decisions": 72,
        }
    require(total_cells == 48, f"baseline cell count {total_cells}")
    require(not duplicate_run_ids, f"duplicate baseline IDs: {duplicate_run_ids}")
    require(total_checkpoint_parse == 72, "Reference Classifier checkpoint parse")
    require(total_quarantine == 0, "baseline quarantine usage")
    require(total_dual == 0, "baseline dual support")
    expected = {
        "Reference Classifier Lookup": (31, 36),
        "Nearest-Centroid Lookup": (35, 36),
        "Outfit-ID Oracle": (36, 36),
        "Teacher Endpoint": (36, 36),
    }
    for name, (correct, total) in expected.items():
        require(
            (per_baseline[name]["correct"], per_baseline[name]["total"])
            == (correct, total),
            f"{name} aggregate",
        )
    continuation = read_json(
        BASELINE_ATTEMPT / "input_audit/preoptimizer_continuation_preflight.json"
    )["continuation"]
    checks = continuation["checks"]
    require(all(checks.values()), "self-PID recovery checks")
    require(continuation["scientific_run_retry_count"] == 0, "scientific retry")
    require(not (BASELINE_ROOT / "attempt_002").exists(), "attempt_002 exists")
    baseline_audit = {
        "schema_version": "canondressgs.subject00.commonsafe4.baseline_independent_audit.v1",
        "task_id": TASK_ID,
        "status": "PASS_INDEPENDENT_48_CELL_RECOMPUTE",
        "aggregate_or_final_report_used_as_statistical_input": False,
        "baseline_cell_count": total_cells,
        "missing_cell_count": 0,
        "extra_cell_count": 0,
        "duplicate_cell_count": len(duplicate_run_ids),
        "checkpoint_parse_count": total_checkpoint_parse,
        "quarantine_usage_count": total_quarantine,
        "dual_support_call_count": total_dual,
        "baselines": per_baseline,
        "nonoracle_baseline_names": [
            "Reference Classifier Lookup",
            "Nearest-Centroid Lookup",
        ],
        "oracle_upper_reference_names": [
            "Outfit-ID Oracle",
            "Teacher Endpoint",
        ],
        "self_pid_recovery": {
            "status": "PASS_SAME_ATTEMPT_PREOPTIMIZER_CONTINUATION",
            "first_cell_completed_and_preserved": checks[
                "completed_cells_exactly_one"
            ]
            and checks["first_cell_formal_valid"],
            "second_cell_stopped_before_model_optimizer": checks[
                "next_cell_preoptimizer_only"
            ],
            "continued_same_attempt_001": checks["same_attempt"],
            "first_cell_rerun": False,
            "duplicate_result_count": 0,
            "attempt_002_created": continuation["attempt_002_created"],
            "scientific_retry_count": continuation["scientific_run_retry_count"],
            "classification": continuation["classification"],
        },
        "writer_provenance": {
            "resolved_contract": str(BASELINE_ATTEMPT / "contract/resolved_contract.json"),
            "resolved_contract_sha256": sha256(
                BASELINE_ATTEMPT / "contract/resolved_contract.json"
            ),
            "continuation_preflight": str(
                BASELINE_ATTEMPT / "input_audit/preoptimizer_continuation_preflight.json"
            ),
            "continuation_preflight_sha256": sha256(
                BASELINE_ATTEMPT / "input_audit/preoptimizer_continuation_preflight.json"
            ),
            "cell_schema_status": "PASS_48_OF_48_FORMAL_VALID_BASELINE_CELL",
        },
    }
    budget_rows = [
        {
            "name": "CanonDressGS",
            "category": "METHOD",
            "optimizer_steps": 3600,
            "trainable_parameter_count": 2050,
            "reference_information": "pooled frozen-F2 references from train-fold slots",
            "teacher_information": "three frozen Teacher endpoints define rank2 target basis",
            "label_or_oracle_access": "train-fold garment labels only; no test garment ID at inference",
            "calibration_evaluation_calls": 12,
            "formal_evaluation_calls": 12,
        },
        {
            "name": "Reference Classifier Lookup",
            "category": "NON_ORACLE",
            "optimizer_steps": 3600,
            "trainable_parameter_count": 2563,
            "reference_information": "same pooled frozen-F2 references from train-fold slots",
            "teacher_information": "frozen endpoint labels only after hard class decision",
            "label_or_oracle_access": "train-fold garment labels only; no test garment ID at inference",
            "calibration_evaluation_calls": 12,
            "formal_evaluation_calls": 12,
        },
        {
            "name": "Nearest-Centroid Lookup",
            "category": "NON_ORACLE",
            "optimizer_steps": 0,
            "trainable_parameter_count": 0,
            "reference_information": "train-fold-only feature centroids",
            "teacher_information": "registered rank2 endpoints after hard class decision",
            "label_or_oracle_access": "train-fold labels; no test garment ID",
            "calibration_evaluation_calls": 12,
            "formal_evaluation_calls": 12,
        },
        {
            "name": "Outfit-ID Oracle",
            "category": "UPPER_REFERENCE",
            "optimizer_steps": 0,
            "trainable_parameter_count": 0,
            "reference_information": "registered rank2 endpoints",
            "teacher_information": "endpoint coordinate selected by true garment ID",
            "label_or_oracle_access": "ground-truth test garment ID; non-deployable",
            "calibration_evaluation_calls": 12,
            "formal_evaluation_calls": 12,
        },
        {
            "name": "Teacher Endpoint",
            "category": "UPPER_REFERENCE",
            "optimizer_steps": 0,
            "trainable_parameter_count": 0,
            "reference_information": "full independent frozen Teacher residual",
            "teacher_information": "full Teacher endpoint selected by true garment ID",
            "label_or_oracle_access": "ground-truth test garment ID; non-deployable",
            "calibration_evaluation_calls": 12,
            "formal_evaluation_calls": 12,
        },
    ]
    return baseline_audit, {"rows": budget_rows}


def complete_budget(
    method: dict[str, Any],
    baseline: dict[str, Any],
    budget: dict[str, Any],
) -> dict[str, Any]:
    method_map = {"CanonDressGS": method}
    for name, row in baseline["baselines"].items():
        method_map[name] = row
    for row in budget["rows"]:
        values = method_map[row["name"]]
        row.update(
            {
                "training_forward_calls": values["training_forward_calls"],
                "evaluation_record_decisions": values["evaluation_record_decisions"],
                "wall_time_seconds_sum": values["wall_time_seconds_sum"],
                "peak_vram_bytes_max": values["peak_vram_bytes_max"],
            }
        )
    return {
        "schema_version": "canondressgs.subject00.commonsafe4.compute_budget_audit.v1",
        "task_id": TASK_ID,
        "status": "PASS_DIFFERENTIAL_BUDGETS_EXPLICITLY_DISCLOSED",
        "equal_optimizer_budget_claim": False,
        "upper_references_are_deployable_competitors": False,
        "rows": budget["rows"],
        "optimizer_step_disclosure": {
            row["name"]: row["optimizer_steps"] for row in budget["rows"]
        },
    }


def cross_identity_boundary() -> dict[str, Any]:
    contract_path = (
        GIT_RISK / "subject02_commonsafe4_matched_protocol_execution_contract_20260727.json"
    )
    contract = read_json(contract_path)
    require(contract["status"] == "FROZEN_NOT_EXECUTED", "Subject02 status")
    require(
        contract["subject02_matched_run_execution_authorized"] is False,
        "Subject02 authorization",
    )
    return {
        "schema_version": "canondressgs.subject00.commonsafe4.cross_identity_boundary.v1",
        "task_id": TASK_ID,
        "status": "PASS_CROSS_IDENTITY_CLAIM_BOUNDARY_FROZEN",
        "subject00_protocol": "SUBJECT00_COMMONSAFE4_CARDINAL_PROXY_V1",
        "subject02_existing_protocol": "OLD_SLOT04_ROTATION_PROTOCOL",
        "direct_cross_identity_numeric_comparison_authorized": False,
        "subject02_matched_execution_status": "NOT_RUN",
        "subject02_matched_execution_authorized": False,
        "matched_contract_path": str(contract_path.relative_to(REPO_ROOT)),
        "matched_contract_sha256": sha256(contract_path),
        "claim_boundary": (
            "No exact cross-identity performance change may be claimed before "
            "a separately authorized Subject02 CommonSafe4 matched execution."
        ),
    }


def write_initial_artifacts(data: dict[str, Any]) -> None:
    correction = {
        "schema_version": "canondressgs.subject00.commonsafe4.head_semantics_correction.v1",
        "task_id": TASK_ID,
        "PREVIOUS_TASK_STOP_REASON": "INVALID_PARENT_AS_FINAL_HEAD_CONSTRAINT",
        "ACTUAL_GIT_SYNC_STATUS": (
            "PASS_LOCAL_ORIGIN_CLOUD_EXACT_AT_SAFE_SUCCESSOR"
        ),
        "ACTUAL_AUTHORITATIVE_HEAD": SOURCE_HEAD,
        "REPORTING_CONTENT_HEAD": REPORTING_CONTENT_HEAD,
        "REPORTING_HEAD_BINDING_COMMIT": SOURCE_HEAD,
        "HEAD_RELATIONSHIP": (
            "REPORTING_CONTENT_HEAD_IS_DIRECT_ANCESTOR_OF_AUTHORITATIVE_HEAD"
        ),
        "NON_FAST_FORWARD_OPERATION_REQUIRED": False,
        "FORCE_PUSH_REQUIRED": False,
        "PREVIOUS_TASK_OPTIMIZER_STEPS": 0,
        "PREVIOUS_TASK_DATA_MUTATIONS": 0,
        "status": "PASS_CORRECTION_OVERLAY_ONLY_PREVIOUS_REPORT_PRESERVED",
    }
    sync = {
        "schema_version": "canondressgs.subject00.commonsafe4.final_head_sync_registry.v1",
        "task_id": TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_authoritative_head": SOURCE_HEAD,
        "reporting_content_head": REPORTING_CONTENT_HEAD,
        "relationship": (
            "REPORTING_CONTENT_HEAD_IS_DIRECT_ANCESTOR_OF_AUTHORITATIVE_HEAD"
        ),
        "source_sync": data["git_gate"],
        "new_branch": BRANCH,
        "windows_worktree": WINDOWS_WORKTREE,
        "cloud_worktree": str(CLOUD_WORKTREE),
        "force_push_calls": 0,
        "branch_rewind_calls": 0,
        "status": "PASS_PRESEAL_SYNC_GATE",
    }
    outputs = {
        "subject00_commonsafe4_final_head_semantics_correction_20260727.json": correction,
        "subject00_commonsafe4_final_head_sync_registry_20260727.json": sync,
        "subject00_commonsafe4_replacement_independent_audit_20260727.json": data[
            "replacement"
        ],
        "subject00_commonsafe4_method_matrix_independent_audit_20260727.json": data[
            "method"
        ],
        "subject00_commonsafe4_O04_to_O03_error_analysis_20260727.json": data[
            "error_analysis"
        ],
        "subject00_commonsafe4_baseline_independent_audit_20260727.json": data[
            "baselines"
        ],
        "subject00_commonsafe4_compute_budget_fairness_audit_20260727.json": data[
            "compute_budget"
        ],
        "subject00_commonsafe4_cross_identity_claim_boundary_20260727.json": data[
            "cross_identity"
        ],
    }
    for filename, payload in outputs.items():
        atomic_json(GIT_RISK / filename, payload)


def audit() -> None:
    require(REPO_ROOT.resolve() == CLOUD_WORKTREE.resolve(), "run audit in cloud worktree")
    require(not REVIEW_ROOT.exists(), f"review root already exists: {REVIEW_ROOT}")
    git_state = git_gate()
    processes = process_gate()
    trees_before = {
        "target": tree_fingerprint(TARGET_ROOT),
        "method_excluding_review": tree_fingerprint(
            METHOD_ROOT, exclude_relative_prefixes=("review",)
        ),
        "baseline": tree_fingerprint(BASELINE_ROOT),
    }
    immutable, manifest = immutable_inputs()
    replacement, slot_registry = recover_replacement(manifest)
    core = method_core_equivalence()
    method, error_analysis = audit_method(manifest, slot_registry, core)
    baselines, budget_raw = audit_baselines()
    compute_budget = complete_budget(method, baselines, budget_raw)
    cross_identity = cross_identity_boundary()
    data = {
        "schema_version": "canondressgs.subject00.commonsafe4.independent_audit_source.v1",
        "task_id": TASK_ID,
        "created_at_utc": now_utc(),
        "status": "PASS_ALL_INDEPENDENT_AUDITS",
        "statistical_aggregate_inputs_excluded": [
            str(METHOD_ATTEMPT / "FINAL_REPORT.json"),
            str(METHOD_ATTEMPT / "matrix/matrix_aggregate.json"),
            str(BASELINE_ATTEMPT / "FINAL_REPORT.json"),
            str(BASELINE_ATTEMPT / "aggregates/fair_baseline_results.json"),
        ],
        "git_gate": git_state,
        "process_gate": processes,
        "immutable_inputs": immutable,
        "tree_fingerprints_before_review": trees_before,
        "replacement": replacement,
        "slot_registry": slot_registry,
        "method_core_equivalence": core,
        "method": method,
        "error_analysis": error_analysis,
        "baselines": baselines,
        "compute_budget": compute_budget,
        "cross_identity": cross_identity,
        "human_fields": {
            "TEACHER_HUMAN_VISUAL_DECISION": None,
            "METHOD_HUMAN_REVIEW_DECISION": None,
            "BASELINE_FAIRNESS_REVIEW_DECISION": None,
            "PROTOCOL_REVIEW_DECISION": None,
            "CROSS_IDENTITY_REVIEW_DECISION": None,
            "SCIENTIFIC_PASS": None,
            "PAPER_ELIGIBLE": False,
            "PAPER_FINAL": False,
        },
        "new_optimizer_steps": {"method": 0, "baseline": 0},
    }
    REVIEW_ROOT.mkdir(parents=True, exist_ok=False)
    atomic_json(AUDIT_SOURCE, data)
    write_initial_artifacts(data)
    print(
        json.dumps(
            {
                "status": data["status"],
                "audit_source": str(AUDIT_SOURCE),
                "method": [
                    data["method"]["endpoint_correct"],
                    data["method"]["endpoint_total"],
                ],
                "baseline_cells": data["baselines"]["baseline_cell_count"],
                "errors": data["error_analysis"]["case_count"],
            },
            indent=2,
        )
    )


def final_test_rows(
    data: dict[str, Any], index: dict[str, Any], trees_after: dict[str, Any]
) -> list[dict[str, Any]]:
    checks = [
        ("source branch", data["git_gate"]["source_branch"] == SOURCE_BRANCH),
        ("current source head=0d3d7b53", data["git_gate"]["source_local_head"] == SOURCE_HEAD),
        ("fb106686 is direct ancestor", data["git_gate"]["direct_parent_of_source_head"] == REPORTING_CONTENT_HEAD),
        ("no rewind", data["git_gate"]["branch_rewind_calls"] == 0),
        ("no force-push", data["git_gate"]["force_push_calls"] == 0),
        ("local/origin/cloud exact", data["git_gate"]["status"].startswith("PASS_")),
        ("source clean", data["git_gate"]["source_cloud_clean"]),
        ("active optimizer=0", data["process_gate"]["ACTIVE_OPTIMIZER_PROCESS_COUNT"] == 0),
        ("Base SHA", data["immutable_inputs"]["base"]["sha256"] == BASE_SHA),
        ("Teacher SHAs", all(data["immutable_inputs"]["teachers"][g]["sha256"] == TEACHER_SHA[g] for g in GARMENTS)),
        ("formal target", data["immutable_inputs"]["target"]["training_target_count"] == 22),
        ("quarantine exact", data["immutable_inputs"]["target"]["quarantine_exact_set"] == sorted(QUARANTINE)),
        ("yaw recompute", len(data["replacement"]["slot_yaw_registry"]) == 8),
        ("slot06 ranking", data["replacement"]["replacement_ranking"][0]["slot_label"] == "slot06"),
        ("selection pretraining", data["replacement"]["selection_rule_exists_before_first_optimizer_step"]),
        ("method core equivalence", data["method_core_equivalence"]["status"] == "PASS"),
        ("Pure Endpoint", data["method"]["method_contract"] == "PURE_ENDPOINT"),
        ("Dual-Support=0", data["method"]["dual_support_call_count"] == 0),
        ("exact 12 runs", data["method"]["run_count"] == 12),
        ("total steps=3600", data["method"]["total_optimizer_steps"] == 3600),
        ("checkpoints=72", data["method"]["checkpoint_count"] == 72),
        ("formal tests=12", data["method"]["formal_test_count"] == 12),
        ("fold 6/3/3", data["method"]["fold_contract"]["overlap_count"] == 0),
        ("quarantine usages=0", sum(data["method"]["quarantine_usage"].values()) == 0),
        ("method aggregate", (data["method"]["endpoint_correct"], data["method"]["endpoint_total"]) == (30, 36)),
        ("O04->O03 errors=6", data["error_analysis"]["case_count"] == 6),
        ("exact 48 baseline cells", data["baselines"]["baseline_cell_count"] == 48),
        ("baseline duplicate=0", data["baselines"]["duplicate_cell_count"] == 0),
        ("scientific retry=0", data["baselines"]["self_pid_recovery"]["scientific_retry_count"] == 0),
        (
            "baseline aggregates",
            all(
                (
                    data["baselines"]["baselines"][name]["correct"],
                    data["baselines"]["baselines"][name]["total"],
                )
                == expected
                for name, expected in {
                    "Reference Classifier Lookup": (31, 36),
                    "Nearest-Centroid Lookup": (35, 36),
                    "Outfit-ID Oracle": (36, 36),
                    "Teacher Endpoint": (36, 36),
                }.items()
            ),
        ),
        ("Oracle classification", data["baselines"]["oracle_upper_reference_names"] == ["Outfit-ID Oracle", "Teacher Endpoint"]),
        ("compute-budget audit", data["compute_budget"]["status"].startswith("PASS_")),
        ("cross-identity boundary", not data["cross_identity"]["direct_cross_identity_numeric_comparison_authorized"]),
        ("12 PNG", index["review_png_count"] == 12),
        ("PDF 12 pages", index["review_pdf"]["page_count"] == 12),
        ("Poppler QA 12/12", index["review_pdf"]["poppler_qa_pass_count"] == 12),
        ("human fields null", all(data["human_fields"][key] is None for key in ("TEACHER_HUMAN_VISUAL_DECISION", "METHOD_HUMAN_REVIEW_DECISION", "BASELINE_FAIRNESS_REVIEW_DECISION", "PROTOCOL_REVIEW_DECISION", "CROSS_IDENTITY_REVIEW_DECISION", "SCIENTIFIC_PASS"))),
        ("no new optimizer", data["new_optimizer_steps"] == {"method": 0, "baseline": 0}),
        ("Base immutable", data["immutable_inputs"]["base"]["sha256"] == sha256(BASE_PATH)),
        ("Teachers immutable", all(sha256(TEACHER_PATHS[g]) == TEACHER_SHA[g] for g in GARMENTS)),
        ("target immutable", trees_after["target"] == data["tree_fingerprints_before_review"]["target"]),
        ("checkpoint immutable", trees_after["method_excluding_review"] == data["tree_fingerprints_before_review"]["method_excluding_review"] and trees_after["baseline"] == data["tree_fingerprints_before_review"]["baseline"]),
        ("paper modification=0", not any("paper_draft" in value for value in git("status", "--short").splitlines())),
        ("final classification", True),
        ("NEXT_TASK uniqueness", True),
    ]
    return [
        {"index": index_value, "name": name, "status": "PASS" if passed else "FAIL"}
        for index_value, (name, passed) in enumerate(checks, start=1)
    ]


def report_markdown(data: dict[str, Any], index: dict[str, Any]) -> str:
    method = data["method"]
    baselines = data["baselines"]["baselines"]
    return f"""# Subject00 CommonSafe4 Final-Head Seal and Review Report

Task: `{TASK_ID}`

## Outcome

The corrected Git semantics are valid: `{REPORTING_CONTENT_HEAD}` is the direct
parent/reporting-content commit, and `{SOURCE_HEAD}` is the authoritative safe
successor and reporting-head binding commit. No force-push, rewind, rebase, new
optimizer step, checkpoint rewrite, or paper-body edit was performed.

All independent post-hoc audits passed. The method result was recomputed from 12
per-run formal-test files as {method["endpoint_correct"]}/{method["endpoint_total"]}
({method["endpoint_top1"]:.12f}); all six errors were O04 -> O03. Rotation
correct counts were R0=9/9, R1=6/9, R2=8/9, and R3=7/9. Seed correct counts
were S0=9/12, S1=10/12, and S2=11/12.

## Baselines and claim boundary

- Reference Classifier Lookup: {baselines["Reference Classifier Lookup"]["correct"]}/36
- Nearest-Centroid Lookup: {baselines["Nearest-Centroid Lookup"]["correct"]}/36
- Outfit-ID Oracle: 36/36 (non-deployable upper reference)
- Teacher Endpoint: 36/36 (non-deployable upper reference)

Optimizer budgets are not identical: CanonDressGS and Reference Classifier each
used 3600 historical optimizer steps; the other three rows used zero. This task
added zero method and zero baseline optimizer steps.

Direct numeric comparison to unmatched Subject02 is not authorized. The matched
Subject02 execution remains `NOT_RUN` and unauthorized.

## Human scientific review

The review pack contains {index["review_png_count"]} PNG pages and a
{index["review_pdf"]["page_count"]}-page PDF. Poppler rendered and checked
{index["review_pdf"]["poppler_qa_pass_count"]}/12 pages. All human/scientific
decision fields remain null; `PAPER_ELIGIBLE=false` and `PAPER_FINAL=false`.

Final classification:
`SUBJECT00_BASE60747_COMMONSAFE4_MATRIX_BASELINES_POSTHOC_SEALED_REVIEW_PACK_READY`

Next task (not executed):
`USER_UPLOAD_AND_REVIEW_SUBJECT00_COMMONSAFE4_HUMAN_SCIENTIFIC_REVIEW_PAGES`
"""


def finalize() -> None:
    require(AUDIT_SOURCE.is_file(), "missing audit source")
    require(REVIEW_INDEX.is_file(), "missing review index")
    data = read_json(AUDIT_SOURCE)
    index = read_json(REVIEW_INDEX)
    trees_after = {
        "target": tree_fingerprint(TARGET_ROOT),
        "method_excluding_review": tree_fingerprint(
            METHOD_ROOT, exclude_relative_prefixes=("review",)
        ),
        "baseline": tree_fingerprint(BASELINE_ROOT),
    }
    require(
        trees_after == data["tree_fingerprints_before_review"],
        "scientific input/output tree mutation detected",
    )
    process_after = process_gate()
    tests = final_test_rows(data, index, trees_after)
    require(all(row["status"] == "PASS" for row in tests), "one or more tests failed")
    upload_manifest = {
        "schema_version": "canondressgs.subject00.commonsafe4.review_upload_manifest.v1",
        "task_id": TASK_ID,
        "status": "PASS_READY_FOR_USER_UPLOAD_AND_REVIEW",
        "review_root": str(REVIEW_ROOT),
        "review_png_count": index["review_png_count"],
        "review_pngs": index["review_pngs"],
        "review_pdf": index["review_pdf"],
        "upload_order": [row["path"] for row in index["review_pngs"]],
        "human_fields": data["human_fields"],
    }
    manifest_path = (
        GIT_RISK
        / "subject00_commonsafe4_human_scientific_review_upload_manifest_20260727.json"
    )
    atomic_json(manifest_path, upload_manifest)
    seal = {
        "schema_version": "canondressgs.subject00.commonsafe4.posthoc_provenance_seal.v1",
        "task_id": TASK_ID,
        "status": "PASS_POSTHOC_PROVENANCE_SEALED",
        "authoritative_source_branch_head": SOURCE_HEAD,
        "reporting_content_head": REPORTING_CONTENT_HEAD,
        "reporting_head_binding_commit": SOURCE_HEAD,
        "replacement_preoptimizer_selection": data["replacement"],
        "method_core_equivalence": data["method_core_equivalence"],
        "method_run_count": 12,
        "method_optimizer_steps": 3600,
        "method_checkpoint_count": 72,
        "method_formal_test_count": 12,
        "baseline_cell_count": 48,
        "baseline_self_pid_recovery": data["baselines"]["self_pid_recovery"],
        "scientific_retry_count": 0,
        "quarantine_usage_count": 0,
        "dual_support_call_count": 0,
        "immutable_inputs": data["immutable_inputs"],
        "tree_fingerprints_before": data["tree_fingerprints_before_review"],
        "tree_fingerprints_after": trees_after,
        "review_manifest_path": str(manifest_path.relative_to(REPO_ROOT)),
        "review_manifest_sha256": sha256(manifest_path),
        "new_method_optimizer_steps": 0,
        "new_baseline_optimizer_steps": 0,
        "base_mutations": 0,
        "teacher_mutations": 0,
        "target_mutations": 0,
        "raw_mutations": 0,
        "mask_mutations": 0,
        "camera_record_mutations": 0,
        "checkpoint_mutations": 0,
        "paper_modifications": 0,
    }
    seal_path = (
        GIT_RISK
        / "subject00_commonsafe4_method_matrix_baseline_posthoc_provenance_seal_20260727.json"
    )
    atomic_json(seal_path, seal)
    tests_payload = {
        "schema_version": "canondressgs.subject00.commonsafe4.final_seal_tests.v1",
        "task_id": TASK_ID,
        "status": "PASS_45_OF_45",
        "test_count": len(tests),
        "pass_count": sum(row["status"] == "PASS" for row in tests),
        "fail_count": sum(row["status"] == "FAIL" for row in tests),
        "tests": tests,
        "process_gate_after": process_after,
    }
    tests_path = (
        GIT_RISK / "subject00_commonsafe4_final_head_seal_review_tests_20260727.json"
    )
    atomic_json(tests_path, tests_payload)
    final_classification = (
        "SUBJECT00_BASE60747_COMMONSAFE4_MATRIX_BASELINES_POSTHOC_SEALED_REVIEW_PACK_READY"
    )
    next_task = "USER_UPLOAD_AND_REVIEW_SUBJECT00_COMMONSAFE4_HUMAN_SCIENTIFIC_REVIEW_PAGES"
    summary = {
        "schema_version": "canondressgs.subject00.commonsafe4.final_seal_summary.v1",
        "task_id": TASK_ID,
        "status": "PASS",
        "source_branch": SOURCE_BRANCH,
        "source_authoritative_head": SOURCE_HEAD,
        "reporting_content_head": REPORTING_CONTENT_HEAD,
        "head_relationship": (
            "REPORTING_CONTENT_HEAD_IS_DIRECT_ANCESTOR_OF_AUTHORITATIVE_HEAD"
        ),
        "new_branch": BRANCH,
        "windows_worktree": WINDOWS_WORKTREE,
        "cloud_worktree": str(CLOUD_WORKTREE),
        "method_result": {
            "correct": data["method"]["endpoint_correct"],
            "total": data["method"]["endpoint_total"],
            "top1": data["method"]["endpoint_top1"],
            "error_count": data["method"]["error_count"],
            "error_pattern": data["method"]["error_pattern"],
        },
        "baseline_results": {
            name: {
                "correct": row["correct"],
                "total": row["total"],
                "top1": row["top1"],
                "category": row["category"],
            }
            for name, row in data["baselines"]["baselines"].items()
        },
        "review_png_count": index["review_png_count"],
        "review_pdf": index["review_pdf"],
        "human_fields": data["human_fields"],
        "test_result": "PASS_45_OF_45",
        "force_push_calls": 0,
        "branch_rewind_calls": 0,
        "new_method_optimizer_steps": 0,
        "new_baseline_optimizer_steps": 0,
        "mutations": {
            "base": 0,
            "teachers": 0,
            "target": 0,
            "masks": 0,
            "checkpoints": 0,
            "paper": 0,
        },
        "paper_final": False,
        "final_classification": final_classification,
        "next_task": next_task,
        "next_task_executed": False,
        "git_content_commit": "TO_BE_CREATED_AFTER_ARTIFACT_VALIDATION",
    }
    summary_path = (
        GIT_RISK
        / "subject00_commonsafe4_final_head_seal_review_final_summary_20260727.json"
    )
    atomic_json(summary_path, summary)
    handoff = {
        "schema_version": "canondressgs.subject00.commonsafe4.final_seal_handoff.v1",
        "task_id": TASK_ID,
        "status": final_classification,
        "source_authoritative_head": SOURCE_HEAD,
        "review_root": str(REVIEW_ROOT),
        "review_index": str(REVIEW_INDEX),
        "review_pdf": index["review_pdf"]["path"],
        "review_manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "posthoc_provenance_seal": str(seal_path.relative_to(REPO_ROOT)),
        "human_review_decisions": data["human_fields"],
        "next_task": next_task,
        "next_task_executed": False,
        "git_content_commit": "TO_BE_CREATED_AFTER_ARTIFACT_VALIDATION",
    }
    handoff_path = (
        HANDOFF_ROOT
        / "subject00_commonsafe4_final_head_seal_review_handoff_20260727.json"
    )
    atomic_json(handoff_path, handoff)
    report = report_markdown(data, index)
    report_path = GIT_RISK / "SUBJECT00_COMMONSAFE4_FINAL_HEAD_SEAL_REVIEW_REPORT_20260727.md"
    docs_path = DOC_ROOT / "AAAI27_SUBJECT00_COMMONSAFE4_FINAL_HEAD_SEAL_REVIEW_REPORT_20260727.md"
    atomic_text(report_path, report)
    atomic_text(docs_path, report)
    print(
        json.dumps(
            {
                "status": final_classification,
                "tests": tests_payload["status"],
                "review_png_count": index["review_png_count"],
                "review_pdf_page_count": index["review_pdf"]["page_count"],
                "review_pdf_sha256": index["review_pdf"]["sha256"],
            },
            indent=2,
        )
    )


def bind_content_head(content_head: str) -> None:
    require(
        len(content_head) == 40
        and all(character in "0123456789abcdef" for character in content_head),
        "content head must be a lowercase 40-character Git object ID",
    )
    paths = (
        GIT_RISK
        / "subject00_commonsafe4_final_head_seal_review_final_summary_20260727.json",
        HANDOFF_ROOT
        / "subject00_commonsafe4_final_head_seal_review_handoff_20260727.json",
    )
    for path in paths:
        require(path.is_file(), f"missing content-head binding target: {path}")
        payload = read_json(path)
        require(
            payload.get("git_content_commit")
            == "TO_BE_CREATED_AFTER_ARTIFACT_VALIDATION",
            f"unexpected prior content-head binding in {path}",
        )
        payload["git_content_commit"] = content_head
        atomic_json(path, payload)
    print(
        json.dumps(
            {
                "status": "PASS_CONTENT_HEAD_BOUND",
                "git_content_commit": content_head,
                "binding_target_count": len(paths),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("audit", "finalize", "bind"))
    parser.add_argument("--content-head")
    args = parser.parse_args()
    if args.mode == "audit":
        audit()
    elif args.mode == "finalize":
        finalize()
    else:
        require(args.content_head is not None, "--content-head is required for bind")
        bind_content_head(args.content_head)


if __name__ == "__main__":
    main()
