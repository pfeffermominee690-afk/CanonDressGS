from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import torch

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from tools.paper import formal_batch_runtime as historical
from tools.paper.p0_formal_candidate_runtime import (
    CANDIDATE_MANIFEST_PATH,
    M4_PARITY_PATH,
    REGISTRY_PATH,
    RUN_BRANCH,
    SOURCE_HEAD,
    TASK_ID,
    TRAINING_AUDIT_PATH,
    TRUNK_AUDIT_PATH,
    aggregate_all,
    atomic_json,
    atomic_yaml,
    execute_run,
    git,
    load_registry,
    preflight,
    prepare_context,
)


RUNTIME_REGISTRY = "manifests/p0_formal_candidate_run_registry_runtime.yaml"


def _run_map(registry: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["formal_run_id"]: row for row in registry["runs"]}


def _atomic_claim(path: Path, value: Mapping[str, Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(dict(value), sort_keys=True) + "\n")
    return descriptor


@contextmanager
def single_gpu_lock(output_root: Path):
    lock = output_root / "audits/SINGLE_GPU_RUN_LOCK.json"
    if lock.exists():
        payload = json.loads(lock.read_text(encoding="utf-8"))
        pid = int(payload.get("pid", -1))
        alive = False
        if pid > 0:
            try:
                os.kill(pid, 0)
                alive = True
            except OSError:
                alive = False
        if alive:
            raise RuntimeError(f"another formal GPU runner is active: pid={pid}")
        stale = output_root / "audits" / f"stale_gpu_lock_{int(time.time())}.json"
        os.replace(lock, stale)
    _atomic_claim(lock, {
        "task_id": TASK_ID, "pid": os.getpid(), "branch": RUN_BRANCH,
        "head": git("rev-parse", "HEAD"), "created_at_unix": time.time(),
    })
    try:
        yield
    finally:
        if lock.exists():
            lock.unlink()


def _root_directories(output_root: Path) -> None:
    for name in (
        "formal_runs", "checkpoints", "renders", "metrics", "aggregates",
        "visuals", "audits", "manifests",
    ):
        (output_root / name).mkdir(parents=True, exist_ok=True)


def _runtime_registry(output_root: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    path = output_root / RUNTIME_REGISTRY
    if path.is_file():
        value = __import__("yaml").safe_load(path.read_text(encoding="utf-8"))
    else:
        value = json.loads(json.dumps(source))
        value["runtime_registry_created_at_unix"] = time.time()
        atomic_yaml(path, value)
    return value


def _save_runtime_registry(output_root: Path, value: Mapping[str, Any]) -> None:
    payload = dict(value)
    payload["runtime_registry_updated_at_unix"] = time.time()
    atomic_yaml(output_root / RUNTIME_REGISTRY, payload)


def _attempt_step(attempt: Path) -> int:
    latest = attempt / "checkpoints/checkpoint_latest.pth"
    if not latest.is_file():
        return 0
    payload = torch.load(latest, map_location="cpu", weights_only=False)
    return int(payload.get("global_step", 0))


def _next_attempt(run_root: Path) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    indices = [
        int(path.name.rsplit("_", 1)[1]) for path in run_root.glob("attempt_*")
        if path.name.rsplit("_", 1)[1].isdigit()
    ]
    attempt = run_root / f"attempt_{max(indices, default=0) + 1:03d}"
    attempt.mkdir(parents=False, exist_ok=False)
    return attempt


def _select_attempt(output_root: Path, run: Mapping[str, Any]) -> tuple[Path, str]:
    run_root = output_root / "formal_runs" / run["output_subdir"]
    attempts = sorted(run_root.glob("attempt_*")) if run_root.exists() else []
    if not attempts:
        return _next_attempt(run_root), "FRESH"
    latest = attempts[-1]
    status_path = latest / "RUN_STATUS.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "MANUAL_REVIEW_REQUIRED":
            return latest, "COMPLETE"
    step = _attempt_step(latest)
    if step > 0:
        return latest, "EXACT_RESUME"
    return _next_attempt(run_root), "ZERO_STEP_NEW_ATTEMPT"


def _snapshot_attempt(
    *, attempt: Path, run: Mapping[str, Any], preflight_report: Mapping[str, Any],
    mode: str,
) -> None:
    for directory in ("checkpoints", "renders", "metrics", "visuals", "manifests", "logs", "provenance", "contract"):
        (attempt / directory).mkdir(parents=True, exist_ok=True)
    for source in (
        REGISTRY_PATH, CANDIDATE_MANIFEST_PATH, TRAINING_AUDIT_PATH,
        TRUNK_AUDIT_PATH, M4_PARITY_PATH,
    ):
        shutil.copy2(source, attempt / "contract" / source.name)
    atomic_json(attempt / "provenance/run_provenance.json", {
        "schema_version": "canondressgs.paper.p0_formal_run_provenance.v1",
        "task_id": TASK_ID, "formal_run": dict(run), "attempt": str(attempt),
        "attempt_mode": mode, "source_head": SOURCE_HEAD,
        "run_commit": git("rev-parse", "HEAD"), "branch": RUN_BRANCH,
        "command": [sys.executable, *sys.argv],
        "preflight": dict(preflight_report),
        "created_at_unix": time.time(), "paper_final": False,
    })
    atomic_json(attempt / "RUN_STATUS.json", {
        "status": "AUTHORIZED_NOT_RUN", "formal_run_id": run["formal_run_id"],
        "optimizer_steps": 0, "paper_final": False,
    })


def _update_run(
    runtime_registry: dict[str, Any], run_id: str, **changes: Any,
) -> None:
    matches = [row for row in runtime_registry["runs"] if row["formal_run_id"] == run_id]
    if len(matches) != 1:
        raise KeyError(run_id)
    matches[0].update(changes)


def _queue(registry: Mapping[str, Any], selected: str | None) -> list[dict[str, Any]]:
    if selected is not None:
        rows = [row for row in registry["runs"] if row["formal_run_id"] == selected]
        if len(rows) != 1:
            raise KeyError(selected)
        return rows
    return list(registry["runs"])


def run_queue(
    *, output_root: Path, asset_root: Path, formal_root: Path,
    selected: str | None,
) -> dict[str, Any]:
    report = preflight(asset_root=asset_root, formal_root=formal_root)
    registry = load_registry()
    _root_directories(output_root)
    preflight_path = output_root / "audits/preflight.json"
    if not preflight_path.exists():
        atomic_json(preflight_path, report)
    runtime_registry = _runtime_registry(output_root, registry)
    queue = _queue(registry, selected)
    state_path = output_root / "manifests/queue_state.json"
    if state_path.is_file():
        previous_state = json.loads(state_path.read_text(encoding="utf-8"))
        existing_attempts = list((output_root / "formal_runs").rglob("attempt_*"))
        if previous_state.get("status") == "RUNNING" and not existing_attempts:
            atomic_json(output_root / "audits/zero_step_startup_failure_001.json", {
                "schema_version": "canondressgs.paper.p0_zero_step_startup_failure.v1",
                "classification": "ZERO_STEP_TOOLING_ERROR",
                "status": "PRESERVED_BEFORE_MINIMAL_FIX",
                "run_commit": "0222a6eb5fac1681c22e1d410b4c3784557f1e08",
                "optimizer_steps": 0, "attempt_created": False,
                "exception_type": "RuntimeError",
                "exception_message": "median CUDA with indices output does not have a deterministic implementation",
                "root_cause": "The deterministic-algorithm guard was enabled before constructing the frozen historical context.",
                "contract_change": False,
                "previous_queue_state": previous_state,
            })
    state = {
        "schema_version": "canondressgs.paper.p0_formal_queue.v1",
        "task_id": TASK_ID, "status": "RUNNING", "queue": [row["formal_run_id"] for row in queue],
        "completed": [], "failed": [], "current": None,
        "started_at_unix": time.time(), "paper_final": False,
    }
    atomic_json(state_path, state)
    canary = formal_root / "PAPER-OURS-S0/seed_0/attempt_001"
    context = prepare_context(canary)
    cache, cache_path, cache_sha256, cache_seconds = historical.build_shared_feature_cache(
        context, formal_root
    )
    atomic_json(output_root / "audits/shared_feature_cache.json", {
        "status": "PASS", "path": str(cache_path), "sha256": cache_sha256,
        "load_seconds": cache_seconds, "target_forward_leakage": False,
    })
    torch.use_deterministic_algorithms(True, warn_only=True)
    with single_gpu_lock(output_root):
        for run in queue:
            run_id = run["formal_run_id"]
            attempt, mode = _select_attempt(output_root, run)
            if mode == "COMPLETE":
                state["completed"].append({"formal_run_id": run_id, "attempt": str(attempt), "reused": True})
                atomic_json(state_path, state)
                continue
            if mode in {"FRESH", "ZERO_STEP_NEW_ATTEMPT"}:
                _snapshot_attempt(
                    attempt=attempt, run=run, preflight_report=report, mode=mode
                )
            state["current"] = run_id
            state["current_attempt"] = str(attempt)
            atomic_json(state_path, state)
            _update_run(
                runtime_registry, run_id, status="RUNNING", attempt=str(attempt),
                provenance=str(attempt / "provenance/run_provenance.json"),
            )
            _save_runtime_registry(output_root, runtime_registry)
            atomic_json(attempt / "RUN_STATUS.json", {
                "status": "RUNNING", "formal_run_id": run_id,
                "optimizer_steps": _attempt_step(attempt), "paper_final": False,
            })
            try:
                result = execute_run(
                    context=context, cache=cache, cache_path=cache_path,
                    run=run, attempt=attempt, asset_root=asset_root,
                )
                _update_run(
                    runtime_registry, run_id, status="MANUAL_REVIEW_REQUIRED",
                    attempt=str(attempt), checkpoint=result["final_checkpoint"],
                    evaluation=result["evaluation"]["evaluated_metrics"],
                    provenance=str(attempt / "provenance/executor_result.json"),
                )
                _save_runtime_registry(output_root, runtime_registry)
                state["completed"].append({
                    "formal_run_id": run_id, "attempt": str(attempt),
                    "optimizer_steps": result["optimizer_steps"],
                })
            except Exception as error:
                step = _attempt_step(attempt)
                classification = "INTERRUPTED_RESUMABLE" if step > 0 else "FAILED"
                failure = {
                    "formal_run_id": run_id, "attempt": str(attempt),
                    "status": classification, "optimizer_steps": step,
                    "exception_type": type(error).__name__,
                    "exception_message": str(error), "traceback": traceback.format_exc(),
                    "updated_at_unix": time.time(),
                }
                atomic_json(attempt / "provenance/runtime_failure.json", failure)
                atomic_json(attempt / "RUN_STATUS.json", {
                    "status": classification, "formal_run_id": run_id,
                    "optimizer_steps": step, "paper_final": False,
                })
                _update_run(runtime_registry, run_id, status=classification, attempt=str(attempt))
                _save_runtime_registry(output_root, runtime_registry)
                state["failed"].append(failure)
                state["status"] = "BLOCKED"
                state["current"] = None
                atomic_json(state_path, state)
                raise
            finally:
                torch.cuda.empty_cache()
            state["current"] = None
            state["updated_at_unix"] = time.time()
            atomic_json(state_path, state)
    if selected is None:
        summary = aggregate_all(
            output_root=output_root, formal_root=formal_root, registry=registry
        )
    else:
        summary = {"status": "PARTIAL_QUEUE_COMPLETE", "run_count": len(state["completed"])}
    state["status"] = "COMPLETE"
    state["finished_at_unix"] = time.time()
    state["aggregate"] = summary
    atomic_json(state_path, state)
    return state


def status(output_root: Path, registry: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for run in registry["runs"]:
        root = output_root / "formal_runs" / run["output_subdir"]
        attempts = sorted(root.glob("attempt_*")) if root.exists() else []
        if not attempts:
            rows.append({
                "formal_run_id": run["formal_run_id"], "attempt": None,
                "status": "AUTHORIZED_NOT_RUN", "optimizer_steps": 0,
            })
            continue
        attempt = attempts[-1]
        value = json.loads((attempt / "RUN_STATUS.json").read_text(encoding="utf-8"))
        rows.append({
            "formal_run_id": run["formal_run_id"], "attempt": attempt.name,
            "attempt_count": len(attempts), "status": value["status"],
            "optimizer_steps": value["optimizer_steps"],
        })
    return {
        "schema_version": "canondressgs.paper.p0_formal_status.v1",
        "run_count": 13,
        "manual_review_required_count": sum(row["status"] == "MANUAL_REVIEW_REQUIRED" for row in rows),
        "total_optimizer_steps": sum(row["optimizer_steps"] for row in rows),
        "paper_final": False, "runs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Serial P0 formal candidate runner")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    for command in ("validate", "run", "aggregate"):
        item = sub.add_parser(command)
        item.add_argument("--asset-root", type=Path, required=True)
        item.add_argument("--formal-root", type=Path, required=True)
        item.add_argument("--output-root", type=Path, required=True)
        if command == "run":
            item.add_argument("--run-id")
    item = sub.add_parser("status")
    item.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    registry = load_registry()
    if args.command == "plan":
        print(json.dumps({
            "run_count": len(registry["runs"]), "trainable_run_count": 12,
            "non_trainable_run_count": 1, "planned_optimizer_steps": 3600,
            "serial_single_gpu": True, "runs": registry["runs"],
        }, sort_keys=True))
        return
    if args.command == "status":
        print(json.dumps(status(args.output_root, registry), sort_keys=True))
        return
    if args.command == "validate":
        if args.output_root.exists():
            raise RuntimeError("validate must run before creating the new formal output root")
        print(json.dumps(preflight(asset_root=args.asset_root, formal_root=args.formal_root), sort_keys=True))
        return
    if args.command == "run":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        value = run_queue(
            output_root=args.output_root, asset_root=args.asset_root,
            formal_root=args.formal_root, selected=args.run_id,
        )
        print(json.dumps({
            "status": value["status"], "completed": len(value["completed"]),
            "failed": len(value["failed"]), "queue_state": str(args.output_root / "manifests/queue_state.json"),
        }, sort_keys=True))
        return
    if args.command == "aggregate":
        value = aggregate_all(
            output_root=args.output_root, formal_root=args.formal_root, registry=registry
        )
        print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
