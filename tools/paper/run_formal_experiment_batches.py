from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import yaml

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from tools.paper import formal_runtime as core
from tools.paper.evaluate_seen_outfit import evaluate_records, load_raw_records
from tools.paper.formal_batch_runtime import (
    ASSET_FINGERPRINT,
    build_shared_feature_cache,
    run_experiment,
)
from tools.paper.method_adapters import adapter_for
from tools.paper.path_resolver import resolve_paths
from tools.paper.registry_state import transition_registry
from tools.paper.run_contract import (
    create_attempt,
    snapshot_attempt,
    strict_preflight,
    validate_executor_result,
)


REGISTRY = "paper_protocol/experiment_registry.yaml"
CONFIG = "configs/paper/aaai27_seen_outfit_explicit_basis_v1.yaml"
MANIFEST = "paper_protocol/frozen_asset_manifest.json"


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _registry(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _batch_for(experiment_id: str) -> str:
    if experiment_id.startswith("PAPER-A"):
        return "batch_2"
    return "batch_1"


def _priority(experiment_id: str) -> tuple[int, str]:
    order = {
        "PAPER-OURS-S1": 0, "PAPER-OURS-S2": 1,
        "PAPER-B0-FIXED": 2, "PAPER-B1-FIXED": 3, "PAPER-B2-FIXED": 4,
    }
    if experiment_id in order:
        return order[experiment_id], experiment_id
    for prefix, base in (("PAPER-B3-", 10), ("PAPER-B4-", 20), ("PAPER-B5-", 30)):
        if experiment_id.startswith(prefix):
            return base + int(experiment_id.rsplit("S", 1)[1]), experiment_id
    if experiment_id.startswith("PAPER-A1-"):
        rank = int(experiment_id.split("-K", 1)[1].split("-", 1)[0])
        seed = int(experiment_id.rsplit("S", 1)[1])
        return 100 + rank * 10 + seed, experiment_id
    for index, prefix in enumerate(("PAPER-A2-", "PAPER-A3-", "PAPER-A4-", "PAPER-A5-", "PAPER-A6-"), 1):
        if experiment_id.startswith(prefix):
            return 200 + index * 10 + int(experiment_id.rsplit("S", 1)[1]), experiment_id
    if experiment_id.startswith("PAPER-A7-"):
        count = int(experiment_id.split("KREF", 1)[1].split("-", 1)[0])
        seed = int(experiment_id.rsplit("S", 1)[1])
        return 300 + count * 10 + seed, experiment_id
    return 9999, experiment_id


def _queue(registry: Mapping[str, Any], selected_batch: str) -> list[str]:
    rows = []
    for experiment in registry["experiments"]:
        if not experiment.get("executable", False):
            continue
        if experiment["experiment_id"] == "PAPER-OURS-S0":
            continue
        if experiment["status"] != "NOT_RUN":
            continue
        batch = _batch_for(experiment["experiment_id"])
        if selected_batch != "all" and batch != selected_batch:
            continue
        rows.append(experiment["experiment_id"])
    return sorted(rows, key=_priority)


def _find(registry: Mapping[str, Any], experiment_id: str) -> dict[str, Any]:
    matches = [
        row for row in registry["experiments"]
        if row["experiment_id"] == experiment_id
    ]
    if len(matches) != 1:
        raise KeyError(experiment_id)
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen seen-outfit paper batches serially")
    parser.add_argument("--batch", choices=("batch_1", "batch_2", "all"), default="all")
    args = parser.parse_args()
    paths = resolve_paths()
    config_path = paths.repo_root / CONFIG
    manifest_path = paths.repo_root / MANIFEST
    registry_path = paths.repo_root / REGISTRY
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    registry_before = _registry(registry_path)
    queue = _queue(registry_before, args.batch)
    queue_path = paths.formal_output_root / "queue_state.json"
    state = {
        "task_id": "AAAI27-FROZEN-PAPER-EXPERIMENT-BATCHES-001",
        "batch": args.batch, "status": "PREFLIGHT",
        "pid": os.getpid(), "queue": queue, "completed": [], "failed": [],
        "current_experiment": None, "next_experiment": queue[0] if queue else None,
        "started_at_unix": time.time(), "updated_at_unix": time.time(),
    }
    _atomic_json(queue_path, state)
    preflight = strict_preflight(
        paths=paths, config_path=config_path, registry_path=registry_path,
        manifest_path=manifest_path,
    )
    canary_attempt = (
        paths.formal_output_root / "PAPER-OURS-S0/seed_0/attempt_001"
    )
    context = core._legacy_context(canary_attempt)
    cache, cache_path, cache_sha256, cache_seconds = build_shared_feature_cache(
        context, paths.formal_output_root
    )
    state.update({
        "status": "RUNNING", "shared_feature_cache": str(cache_path),
        "shared_feature_cache_sha256": cache_sha256,
        "shared_feature_cache_seconds": cache_seconds,
    })
    _atomic_json(queue_path, state)
    for position, experiment_id in enumerate(queue):
        registry = _registry(registry_path)
        experiment = _find(registry, experiment_id)
        if experiment["status"] != "NOT_RUN":
            raise RuntimeError(f"queue registry state drift: {experiment_id}")
        adapter = adapter_for(experiment, config)
        contract = adapter.validate_contract()
        training_plan = adapter.build_training_plan()
        attempt = create_attempt(
            paths.experiment_root(experiment_id, experiment.get("seed"))
        )
        snapshot_attempt(
            attempt,
            experiment=experiment,
            config_path=config_path,
            registry_path=registry_path,
            manifest_path=manifest_path,
            preflight=preflight,
            command=[sys.executable, *sys.argv, "--experiment-id", experiment_id],
            repo_root=paths.repo_root,
            adapter_contract=contract,
            training_plan=training_plan,
            model_build_spec=asdict(adapter.build_model()),
        )
        state.update({
            "current_experiment": experiment_id,
            "current_attempt": str(attempt),
            "next_experiment": queue[position + 1] if position + 1 < len(queue) else None,
            "updated_at_unix": time.time(),
        })
        _atomic_json(queue_path, state)
        try:
            result = run_experiment(
                context, cache, cache_path, cache_sha256, cache_seconds,
                attempt, experiment, contract, training_plan,
            )
            validate_executor_result(attempt, training_plan)
            evaluated = evaluate_records(
                load_raw_records(Path(result["raw_metrics"])),
                expected_asset_fingerprint=registry["frozen_asset_manifest_sha256"],
            )
            evaluated_path = attempt / "evaluated_metrics/evaluated_metrics.json"
            _atomic_json(evaluated_path, evaluated)
            transition_registry(registry_path, experiment_id, "PREFLIGHT_PASS", evidence={
                "commit": core._git("rev-parse", "HEAD"), "attempt": str(attempt),
                "asset_fingerprint": ASSET_FINGERPRINT,
            })
            if training_plan["optimizer_required"]:
                transition_registry(registry_path, experiment_id, "RUNNING", evidence={
                    "commit": core._git("rev-parse", "HEAD"), "attempt": str(attempt),
                })
                transition_registry(registry_path, experiment_id, "TRAINED", evidence={
                    "optimizer_steps": training_plan["steps"],
                    "checkpoint": result["training"]["checkpoints"][-1]["path"],
                })
            transition_registry(registry_path, experiment_id, "EVALUATED", evidence={
                "evaluator_version": experiment["evaluator_version"],
                "metric_count": evaluated["metric_count"],
                "evaluated_output": str(evaluated_path),
                "manual_review_pending": True,
            })
            _atomic_json(attempt / "RUN_STATUS.json", {
                "status": "EVALUATED", "optimizer_steps": training_plan["steps"],
                "experiment_id": experiment_id, "updated_at_unix": time.time(),
            })
            state["completed"].append({
                "experiment_id": experiment_id, "attempt": str(attempt),
                "optimizer_steps": training_plan["steps"],
                "metrics": evaluated["metrics"],
            })
        except Exception as error:
            failure = {
                "experiment_id": experiment_id, "attempt": str(attempt),
                "exception_type": type(error).__name__, "exception_message": str(error),
                "traceback": traceback.format_exc(), "updated_at_unix": time.time(),
            }
            _atomic_json(attempt / "provenance/batch_runtime_failure.json", failure)
            state["failed"].append(failure)
            state["status"] = "BLOCKED"
            state["updated_at_unix"] = time.time()
            _atomic_json(queue_path, state)
            raise
        state["updated_at_unix"] = time.time()
        _atomic_json(queue_path, state)
    state.update({
        "status": "COMPLETE", "current_experiment": None,
        "next_experiment": None, "finished_at_unix": time.time(),
        "updated_at_unix": time.time(),
    })
    _atomic_json(queue_path, state)
    print(json.dumps({
        "status": "COMPLETE", "batch": args.batch,
        "completed": len(state["completed"]), "failed": len(state["failed"]),
        "queue_state": str(queue_path),
    }))


if __name__ == "__main__":
    main()
