from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from tools.paper.aggregation import (
    aggregate_evaluations,
    aggregate_smoke_evaluation,
    write_aggregation,
    write_smoke_aggregation,
)
from tools.paper.evaluate_seen_outfit import evaluate_records, load_raw_records
from tools.paper.method_adapters import ADAPTER_TYPES, adapter_for
from tools.paper.paper_exports import export_figure_layouts, export_tables
from tools.paper.path_resolver import PaperPaths, resolve_paths
from tools.paper.planning import build_run_plan, write_run_plan
from tools.paper.protocol_audit import audit_protocol, load_documents
from tools.paper.registry_state import transition_registry
from tools.paper.run_contract import prepare_real_run, strict_preflight, validate_executor_result
from tools.paper.verify_seen_outfit_paper_assets import verify_manifest


DEFAULT_REGISTRY = "paper_protocol/experiment_registry.yaml"
DEFAULT_CONFIG = "configs/paper/aaai27_seen_outfit_explicit_basis_v1.yaml"
DEFAULT_MANIFEST = "paper_protocol/frozen_asset_manifest.json"
DEFAULT_AUDIT = "paper_protocol/audits/protocol_registry_audit.json"
DEFAULT_GENERATED = "paper_protocol/generated"


def _repo_path(paths: PaperPaths, value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else paths.repo_root / candidate


def _documents(args: argparse.Namespace, paths: PaperPaths):
    registry_path = _repo_path(paths, args.registry)
    config_path = _repo_path(paths, args.config)
    manifest_path = _repo_path(paths, DEFAULT_MANIFEST)
    config, manifest, registry = load_documents(config_path, manifest_path, registry_path)
    return config_path, manifest_path, registry_path, config, manifest, registry


def _experiment(registry: Mapping[str, Any], experiment_id: str | None, seed: int | None):
    if experiment_id is None:
        raise ValueError("--experiment-id is required")
    matches = [item for item in registry["experiments"] if item["experiment_id"] == experiment_id]
    if len(matches) != 1:
        raise KeyError(f"experiment id must resolve exactly once: {experiment_id}")
    experiment = matches[0]
    if seed is not None and experiment.get("seed") != seed:
        raise ValueError("CLI seed differs from frozen registry seed")
    return experiment


def _print(payload: Mapping[str, Any]) -> None:
    print(json.dumps(dict(payload), indent=2, ensure_ascii=False))


def command_validate(args: argparse.Namespace, paths: PaperPaths) -> dict[str, Any]:
    config_path, manifest_path, registry_path, config, manifest, registry = _documents(args, paths)
    claim_matrix = (paths.repo_root / "docs/PAPER/AAAI27_CLAIM_EVIDENCE_MATRIX_SEEN_OUTFIT_20260720.md").read_text(encoding="utf-8")
    table_figure_plan = (paths.repo_root / "docs/PAPER/AAAI27_PAPER_TABLE_AND_FIGURE_PLAN_20260720.md").read_text(encoding="utf-8")
    audit = audit_protocol(
        config, manifest, registry, claim_matrix_text=claim_matrix,
        table_figure_plan_text=table_figure_plan,
    )
    audit_path = _repo_path(paths, DEFAULT_AUDIT)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if audit["status"] != "PASS":
        raise RuntimeError(audit["status"])
    assets = verify_manifest(
        manifest, paths.repo_root, paths.asset_root,
        verify_external=paths.asset_root is not None and not args.dry_run,
    )
    if paths.asset_root is not None and not args.dry_run and assets["status"] != "PASS":
        raise RuntimeError("PAPER_ASSET_MISMATCH")
    result = {
        "status": "PASS", "protocol_audit": audit,
        "asset_verification": {
            "status": assets["status"], "verified_external": assets["verified_external"],
            "asset_count": assets["asset_count"], "failed_assets": assets["failed_assets"],
        },
        "adapter_count": len(ADAPTER_TYPES), "dry_run": args.dry_run,
    }
    _print(result); return result


def command_plan(args: argparse.Namespace, paths: PaperPaths) -> dict[str, Any]:
    _, _, _, config, _, registry = _documents(args, paths)
    plan = build_run_plan(registry, config, paths)
    generated = _repo_path(paths, args.generated_dir or DEFAULT_GENERATED)
    files = write_run_plan(plan, generated)
    result = {"status": "PASS", "dry_run": True, "run_count": len(plan["runs"]), "files": [str(path) for path in files]}
    _print(result); return result


def _execute(command: str, attempt: Path, *, resume_checkpoint: Path | None = None) -> None:
    environment = os.environ.copy()
    environment["CANONDRESSGS_PAPER_ATTEMPT"] = str(attempt)
    if resume_checkpoint is not None:
        environment["CANONDRESSGS_PAPER_RESUME_CHECKPOINT"] = str(resume_checkpoint)
    completed = subprocess.run(shlex.split(command, posix=os.name != "nt"), cwd=attempt, env=environment, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"paper executor failed with exit code {completed.returncode}")


def command_run(args: argparse.Namespace, paths: PaperPaths) -> dict[str, Any]:
    config_path, manifest_path, registry_path, config, _, registry = _documents(args, paths)
    experiment = _experiment(registry, args.experiment_id, args.seed)
    if not experiment.get("executable", False):
        raise ValueError("historical A8 evidence cannot be run")
    adapter = adapter_for(experiment, config)
    if args.dry_run:
        result = {"status": "DRY_RUN", "experiment_id": experiment["experiment_id"], "contract": adapter.validate_contract(), "training_plan": adapter.build_training_plan(), "optimizer_created": False}
        _print(result); return result
    if not args.executor_command:
        raise ValueError("real run requires an explicitly authorized --executor-command")
    attempt, contract, training = prepare_real_run(
        paths=paths, config_path=config_path, registry_path=registry_path,
        manifest_path=manifest_path, experiment=experiment, command=sys.argv,
    )
    try:
        # Keep the checkout bitwise clean while the production executor loads the
        # legacy frozen stack.  Those audited loaders intentionally reject dirty
        # source trees.  The attempt itself is the durable RUNNING state; the
        # canonical registry transitions are appended atomically after the
        # executor has returned.
        _execute(args.executor_command, attempt)
        validate_executor_result(attempt, training)
        transition_registry(registry_path, experiment["experiment_id"], "PREFLIGHT_PASS")
        transition_registry(registry_path, experiment["experiment_id"], "RUNNING")
        transition_registry(registry_path, experiment["experiment_id"], "TRAINED")
        (attempt / "RUN_STATUS.json").write_text(json.dumps({"status": "TRAINED"}, indent=2) + "\n", encoding="utf-8")
    except Exception:
        current = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        current_status = next(
            item["status"] for item in current["experiments"]
            if item["experiment_id"] == experiment["experiment_id"]
        )
        if current_status != "FAILED":
            if current_status == "NOT_RUN":
                transition_registry(registry_path, experiment["experiment_id"], "FAILED")
            elif current_status in {"PREFLIGHT_PASS", "RUNNING", "TRAINED", "EVALUATED", "MANUAL_REVIEW_REQUIRED"}:
                transition_registry(registry_path, experiment["experiment_id"], "FAILED")
        raise
    result = {"status": "TRAINED", "attempt": str(attempt), "contract": contract, "training_plan": training}
    _print(result); return result


def command_resume(args: argparse.Namespace, paths: PaperPaths) -> dict[str, Any]:
    config_path, manifest_path, registry_path, _, _, registry = _documents(args, paths)
    experiment = _experiment(registry, args.experiment_id, args.seed)
    if args.attempt is None or args.checkpoint is None:
        raise ValueError("resume requires --attempt and --checkpoint")
    attempt, checkpoint = Path(args.attempt), Path(args.checkpoint)
    if not attempt.is_dir() or not checkpoint.is_file() or attempt not in checkpoint.parents:
        raise ValueError("resume checkpoint must exist inside the selected attempt")
    required_sidecar = checkpoint.with_suffix(checkpoint.suffix + ".resume.json")
    if not required_sidecar.is_file():
        raise ValueError("resume checkpoint sidecar is missing")
    resume_state = json.loads(required_sidecar.read_text(encoding="utf-8"))
    required = {"model", "optimizer", "rng", "scheduler", "global_step", "condition_position", "fixed_output_parity"}
    if not required.issubset(resume_state):
        raise ValueError("resume checkpoint state is incomplete")
    if args.dry_run:
        result = {"status": "DRY_RUN", "attempt": str(attempt), "checkpoint": str(checkpoint), "state_fields": sorted(required), "optimizer_created": False}
        _print(result); return result
    if not args.executor_command:
        raise ValueError("real resume requires an explicitly authorized --executor-command")
    strict_preflight(paths=paths, config_path=config_path, registry_path=registry_path, manifest_path=manifest_path)
    _execute(args.executor_command, attempt, resume_checkpoint=checkpoint)
    result = {"status": "RESUMED", "attempt": str(attempt), "checkpoint": str(checkpoint)}
    _print(result); return result


def command_evaluate(args: argparse.Namespace, paths: PaperPaths) -> dict[str, Any]:
    config_path, manifest_path, registry_path, _, _, registry = _documents(args, paths)
    experiment = _experiment(registry, args.experiment_id, args.seed)
    if not experiment.get("executable", False):
        raise ValueError("historical A8 evidence cannot be evaluated as a formal run")
    if args.raw_metrics is None:
        raise ValueError("evaluate requires --raw-metrics")
    if not args.dry_run:
        strict_preflight(paths=paths, config_path=config_path, registry_path=registry_path, manifest_path=manifest_path)
    result = evaluate_records(load_raw_records(Path(args.raw_metrics)), expected_asset_fingerprint=registry["frozen_asset_manifest_sha256"])
    if args.dry_run:
        _print({"status": "DRY_RUN_EVALUATION_PASS", "metric_count": result["metric_count"]}); return result
    if args.evaluated_output is None:
        raise ValueError("evaluate requires --evaluated-output")
    output = Path(args.evaluated_output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    transition_registry(registry_path, experiment["experiment_id"], "EVALUATED")
    _print({"status": "EVALUATED", "output": str(output), "metric_count": result["metric_count"]}); return result


def command_aggregate(args: argparse.Namespace, paths: PaperPaths) -> dict[str, Any]:
    if args.smoke_only:
        if len(args.evaluations or []) != 1:
            raise ValueError("--smoke-only aggregate requires exactly one seed-0 evaluation")
        evaluation = json.loads(Path(args.evaluations[0]).read_text(encoding="utf-8"))
        result = aggregate_smoke_evaluation(evaluation)
        if args.aggregate_output is not None:
            write_smoke_aggregation(result, Path(args.aggregate_output))
        _print({
            "status": "SMOKE_ONLY",
            "seeds": [0],
            "best_seed_selected": False,
            "three_seed_statistics_generated": False,
        })
        return result
    if len(args.evaluations or []) != 3:
        raise ValueError("MISSING_REQUIRED_PAPER_SEEDS: formal aggregate requires seeds 0, 1, 2")
    evaluations = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.evaluations]
    result = aggregate_evaluations(evaluations)
    if args.aggregate_output is not None:
        write_aggregation(result, Path(args.aggregate_output))
    _print({"status": "PASS", "seeds": result["seeds"], "best_seed_selected": False}); return result


def command_export_tables(args: argparse.Namespace, paths: PaperPaths) -> dict[str, Any]:
    if args.source_data is None or args.export_output is None:
        raise ValueError("export-tables requires --source-data and --export-output")
    source = json.loads(Path(args.source_data).read_text(encoding="utf-8"))
    result = export_tables(source, Path(args.export_output)); _print({"status": "PASS", "tables": result}); return result


def command_export_figures(args: argparse.Namespace, paths: PaperPaths) -> dict[str, Any]:
    if args.export_output is None:
        raise ValueError("export-figures requires --export-output")
    source_manifest = json.loads(Path(args.figure_source_manifest).read_text(encoding="utf-8")) if args.figure_source_manifest else None
    result = export_figure_layouts(Path(args.export_output), synthetic=args.synthetic_fixture, source_manifest=source_manifest)
    _print({"status": "PASS", "figure_count": len(result["figures"])}); return result


def command_archive(args: argparse.Namespace, paths: PaperPaths) -> dict[str, Any]:
    _, manifest_path, registry_path, _, _, registry = _documents(args, paths)
    result = {
        "status": "ARCHIVE_PLAN_ONLY" if args.dry_run else "ARCHIVE_MANIFEST_READY",
        "anonymous": True, "copy_performed": False,
        "required": [str(registry_path.relative_to(paths.repo_root)), str(manifest_path.relative_to(paths.repo_root)), DEFAULT_CONFIG, DEFAULT_AUDIT, DEFAULT_GENERATED],
        "registry_status_counts": {status: sum(item["status"] == status for item in registry["experiments"]) for status in registry["paper_status_values"]},
    }
    if args.archive_output:
        output = Path(args.archive_output); output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _print(result); return result


def command_status(args: argparse.Namespace, paths: PaperPaths) -> dict[str, Any]:
    _, _, _, _, _, registry = _documents(args, paths)
    counts = {status: sum(item["status"] == status for item in registry["experiments"]) for status in registry["paper_status_values"]}
    artifacts = {}
    if paths.output_root is not None:
        for item in registry["experiments"]:
            root = paths.experiment_root(item["experiment_id"], item.get("seed"))
            attempts = sorted(path.name for path in root.glob("attempt_[0-9][0-9][0-9]") if path.is_dir()) if root.is_dir() else []
            artifacts[item["experiment_id"]] = attempts
    result = {"status": "READ_ONLY", "total": len(registry["experiments"]), "counts": counts, "executable": sum(bool(item.get("executable")) for item in registry["experiments"]), "actual_attempts": artifacts}
    _print(result); return result


COMMANDS = {
    "validate": command_validate, "plan": command_plan, "run": command_run,
    "resume": command_resume, "evaluate": command_evaluate, "aggregate": command_aggregate,
    "export-tables": command_export_tables, "export-figures": command_export_figures,
    "archive": command_archive, "status": command_status,
}


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--experiment-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--repo-root", type=Path)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Unified frozen seen-outfit paper runner and evaluator")
    subparsers = root.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        child = subparsers.add_parser(command); _add_common(child)
        child.add_argument("--generated-dir", type=Path)
        child.add_argument("--executor-command")
        child.add_argument("--attempt", type=Path)
        child.add_argument("--checkpoint", type=Path)
        child.add_argument("--raw-metrics", type=Path)
        child.add_argument("--evaluated-output", type=Path)
        child.add_argument("--evaluations", type=Path, action="append")
        child.add_argument("--aggregate-output", type=Path)
        child.add_argument("--source-data", type=Path)
        child.add_argument("--export-output", type=Path)
        child.add_argument("--archive-output", type=Path)
        child.add_argument("--synthetic-fixture", action="store_true")
        child.add_argument("--smoke-only", action="store_true")
        child.add_argument("--figure-source-manifest", type=Path)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    paths = resolve_paths(repo_root=args.repo_root, asset_root=args.asset_root, output_root=args.output_root)
    COMMANDS[args.command](args, paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
