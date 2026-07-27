#!/usr/bin/env python3
"""Atomically seal the Subject00 method-matrix lock after a blocked preflight."""

from __future__ import annotations

import json
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path


TASK_ID = "AAAI27-SUBJECT00-BASE60747-REMAINING-METHOD-MATRIX-FAIR-BASELINES-001"
BRANCH = "research/subject00-base60747-remaining-method-matrix-fair-baselines-20260727"
OUTPUT_ROOT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-CANONDRESSGS-METHOD-BASE60747-001"
)
LOCK_PATH = OUTPUT_ROOT / "control/METHOD_MATRIX_EXECUTION_LOCK_20260727.json"
RUNNER_NAMES = (
    "run_subject00_base60747_method.py",
    "run_subject00_base60747_formal_teacher.py",
    "run_subject00_formal_base_101245.py",
)


def active_optimizer_processes() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(name in command for name in RUNNER_NAMES):
            rows.append({"pid": int(entry.name), "command": command.strip()})
    return rows


def main() -> int:
    repo = Path.cwd().resolve()
    branch = subprocess.check_output(
        ["git", "-C", str(repo), "branch", "--show-current"], text=True
    ).strip()
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if branch != BRANCH:
        raise RuntimeError(f"unexpected branch: {branch}")
    processes = active_optimizer_processes()
    if processes:
        raise RuntimeError(f"optimizer process exists: {processes}")
    if LOCK_PATH.exists():
        raise FileExistsError(f"lock already exists: {LOCK_PATH}")

    now = datetime.now(timezone.utc).isoformat()
    pending = [
        f"METHOD-R{rotation}-S{seed}"
        for rotation in range(4)
        for seed in range(3)
        if (rotation, seed) != (0, 0)
    ]
    payload = {
        "schema_version": "canondressgs.subject00.method_matrix_execution_lock.v1",
        "task_id": TASK_ID,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "tmux_or_session": os.environ.get("TMUX") or os.environ.get("STY"),
        "git_branch": branch,
        "git_head": head,
        "start_timestamp": now,
        "end_timestamp": now,
        "method_output_root": str(OUTPUT_ROOT),
        "expected_pending_run_ids": pending,
        "active_optimizer_process_count_before": 0,
        "status": "CLOSED_BLOCKED_PREFLIGHT",
        "close_reason": (
            "FROZEN_ROTATION_FOLDS_REQUIRE_QUARANTINED_O01_O03_SLOT04_RECORDS; "
            "NO_OPTIMIZER_WAS_CREATED"
        ),
        "new_method_optimizer_steps": 0,
        "new_baseline_optimizer_steps": 0,
    }
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(LOCK_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
