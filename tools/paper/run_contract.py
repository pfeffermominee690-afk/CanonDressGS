from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .method_adapters import adapter_for
from .path_resolver import PaperPaths
from .protocol_audit import audit_protocol, load_documents
from .verify_seen_outfit_paper_assets import verify_manifest


ATTEMPT_DIRECTORIES = (
    "contract", "preflight", "logs", "checkpoints", "raw_metrics",
    "evaluated_metrics", "visuals", "tables", "provenance", "final_adjudication",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_asset_report_pass(report: Mapping[str, Any]) -> None:
    if report.get("status") != "PASS" or report.get("failed_assets"):
        raise RuntimeError("PAPER_ASSET_MISMATCH")


def strict_preflight(
    *,
    paths: PaperPaths,
    config_path: Path,
    registry_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    config, manifest, registry = load_documents(config_path, manifest_path, registry_path)
    protocol = audit_protocol(config, manifest, registry)
    if protocol["status"] != "PASS":
        raise RuntimeError(protocol["status"])
    if paths.asset_root is None:
        raise ValueError("--asset-root or CANONDRESSGS_ASSET_ROOT is required for strict preflight")
    assets = verify_manifest(manifest, paths.repo_root, paths.asset_root, verify_external=True)
    assert_asset_report_pass(assets)
    return {"status": "PASS", "protocol": protocol, "assets": assets}


def next_attempt_path(root: Path) -> Path:
    for number in range(1, 10000):
        candidate = root / f"attempt_{number:03d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("attempt namespace exhausted")


def create_attempt(root: Path) -> Path:
    attempt = next_attempt_path(root)
    attempt.mkdir(parents=True, exist_ok=False)
    for name in ATTEMPT_DIRECTORIES:
        (attempt / name).mkdir()
    return attempt


def git_metadata(repo_root: Path) -> dict[str, Any]:
    def output(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=repo_root, text=True).strip()
    status = output("status", "--short")
    return {
        "commit": output("rev-parse", "HEAD"), "branch": output("branch", "--show-current"),
        "status_short": status, "clean": not bool(status),
    }


def environment_metadata() -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "python_executable": sys.executable,
        "platform": sys.platform, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def snapshot_attempt(
    attempt: Path,
    *,
    experiment: Mapping[str, Any],
    config_path: Path,
    registry_path: Path,
    manifest_path: Path,
    preflight: Mapping[str, Any],
    command: Sequence[str],
    repo_root: Path,
    adapter_contract: Mapping[str, Any],
    training_plan: Mapping[str, Any],
    model_build_spec: Mapping[str, Any],
) -> None:
    contract = attempt / "contract"
    shutil.copy2(config_path, contract / "method_config_snapshot.yaml")
    shutil.copy2(registry_path, contract / "experiment_registry_snapshot.yaml")
    shutil.copy2(manifest_path, contract / "frozen_asset_manifest.json")
    (attempt / "preflight/asset_verification.json").write_text(json.dumps(preflight, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    provenance = {
        "experiment": dict(experiment), "git": git_metadata(repo_root),
        "environment": environment_metadata(), "command": list(command),
        "seed": experiment.get("seed"), "dataset_split": ["O01", "O02", "O03", "O04", "O08"],
        "evaluator_version": experiment["evaluator_version"],
        "adapter_contract": dict(adapter_contract), "training_plan": dict(training_plan),
        "model_trainable_frozen_parameter_summary": dict(model_build_spec),
    }
    (attempt / "provenance/run_provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (attempt / "RUN_STATUS.json").write_text(json.dumps({"status": "PREFLIGHT_PASS", "optimizer_steps": 0}, indent=2) + "\n", encoding="utf-8")


def prepare_real_run(
    *,
    paths: PaperPaths,
    config_path: Path,
    registry_path: Path,
    manifest_path: Path,
    experiment: Mapping[str, Any],
    command: Sequence[str],
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    preflight = strict_preflight(paths=paths, config_path=config_path, registry_path=registry_path, manifest_path=manifest_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    adapter = adapter_for(experiment, config)
    adapter_contract = adapter.validate_contract()
    training_plan = adapter.build_training_plan()
    model_build_spec = asdict(adapter.build_model())
    attempt = create_attempt(paths.experiment_root(experiment["experiment_id"], experiment.get("seed")))
    snapshot_attempt(
        attempt, experiment=experiment, config_path=config_path, registry_path=registry_path,
        manifest_path=manifest_path, preflight=preflight, command=command,
        repo_root=paths.repo_root, adapter_contract=adapter_contract, training_plan=training_plan,
        model_build_spec=model_build_spec,
    )
    return attempt, adapter_contract, training_plan


def validate_executor_result(attempt: Path, training_plan: Mapping[str, Any]) -> dict[str, Any]:
    path = attempt / "provenance/executor_result.json"
    if not path.is_file():
        raise ValueError("executor did not persist provenance/executor_result.json")
    result = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "optimizer_created", "optimizer_parameter_scope", "global_step",
        "checkpoint_milestones", "frozen_parameter_max_change",
        "frozen_gradient_count", "target_forward_leakage", "outfit_id_in_model",
    }
    if not required.issubset(result):
        raise ValueError("executor result contract is incomplete")
    if bool(result["optimizer_created"]) is not bool(training_plan["optimizer_required"]):
        raise ValueError("optimizer creation differs from adapter contract")
    if result["optimizer_parameter_scope"] != training_plan["optimizer_parameter_scope"]:
        raise ValueError("optimizer parameter set differs from adapter contract")
    if int(result["global_step"]) != int(training_plan["steps"]):
        raise ValueError("executor did not reach the exact frozen step budget")
    if list(result["checkpoint_milestones"]) != list(training_plan["milestones"]):
        raise ValueError("checkpoint milestones differ from frozen contract")
    if float(result["frozen_parameter_max_change"]) != 0.0 or int(result["frozen_gradient_count"]) != 0:
        raise ValueError("frozen parameter contract failed")
    if result["target_forward_leakage"] or result["outfit_id_in_model"]:
        raise ValueError("prediction boundary contract failed")
    return result
