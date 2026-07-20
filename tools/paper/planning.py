from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

from .method_adapters import adapter_for
from .path_resolver import PaperPaths


def _write_lf(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def build_run_plan(registry: Mapping[str, Any], config: Mapping[str, Any], paths: PaperPaths) -> dict[str, Any]:
    rows = []
    for experiment in registry["experiments"]:
        if not experiment.get("executable", False):
            continue
        adapter = adapter_for(experiment, config)
        contract = adapter.validate_contract()
        training = adapter.build_training_plan()
        linux = (
            'python tools/paper/aaai27_paper_pipeline.py run '
            '--registry "$CANONDRESSGS_REPO_ROOT/paper_protocol/experiment_registry.yaml" '
            '--config "$CANONDRESSGS_REPO_ROOT/configs/paper/aaai27_seen_outfit_explicit_basis_v1.yaml" '
            f'--experiment-id {experiment["experiment_id"]} '
            '--repo-root "$CANONDRESSGS_REPO_ROOT" --asset-root "$CANONDRESSGS_ASSET_ROOT" '
            '--output-root "$CANONDRESSGS_OUTPUT_ROOT"'
        )
        windows = (
            '& $Python tools/paper/aaai27_paper_pipeline.py run '
            '--registry (Join-Path $RepoRoot "paper_protocol/experiment_registry.yaml") '
            '--config (Join-Path $RepoRoot "configs/paper/aaai27_seen_outfit_explicit_basis_v1.yaml") '
            f'--experiment-id {experiment["experiment_id"]} '
            '--repo-root $RepoRoot --asset-root $AssetRoot --output-root $OutputRoot'
        )
        if experiment.get("seed") is not None:
            linux += f' --seed {experiment["seed"]}'
            windows += f' --seed {experiment["seed"]}'
        linux += ' --executor-command "$CANONDRESSGS_PAPER_EXECUTOR"'
        windows += " --executor-command $Executor"
        rows.append({
            "experiment_id": experiment["experiment_id"], "method": experiment["method"],
            "seed": experiment.get("seed"), "status": experiment["status"],
            "evaluator_version": experiment["evaluator_version"],
            "adapter": contract["adapter"], "steps": training["steps"],
            "optimizer_required": training["optimizer_required"],
            "output_root": str(paths.experiment_root(experiment["experiment_id"], experiment.get("seed"))) if paths.output_root else f"${{CANONDRESSGS_OUTPUT_ROOT}}/AAAI27-SEEN-OUTFIT-PAPER/{experiment['experiment_id']}/seed_{experiment.get('seed') if experiment.get('seed') is not None else 'fixed'}",
            "linux_command": linux, "windows_command": windows,
        })
    return {
        "task_id": "AAAI27-UNIFIED-PAPER-RUNNER-EVALUATOR-001",
        "protocol_task_id": registry["task_id"], "marker": "DRY_RUN_PLAN_ONLY",
        "executable_count": len(rows), "historical_excluded": 4,
        "commands_execute_automatically": False, "runs": rows,
    }


def write_run_plan(plan: Mapping[str, Any], generated_dir: Path) -> list[Path]:
    generated_dir.mkdir(parents=True, exist_ok=True)
    json_path = generated_dir / "paper_run_plan.json"
    csv_path = generated_dir / "paper_run_plan.csv"
    linux_path = generated_dir / "paper_run_commands_linux.sh"
    windows_path = generated_dir / "paper_run_commands_windows.ps1"
    json_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rows = list(plan["runs"])
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    _write_lf(linux_path, "#!/usr/bin/env bash\nset -euo pipefail\n: \"${CANONDRESSGS_REPO_ROOT:?required}\"\n: \"${CANONDRESSGS_ASSET_ROOT:?required}\"\n: \"${CANONDRESSGS_OUTPUT_ROOT:?required}\"\n: \"${CANONDRESSGS_PAPER_EXECUTOR:?required}\"\ncd \"$CANONDRESSGS_REPO_ROOT\"\n\n" + "\n".join(row["linux_command"] for row in rows) + "\n")
    _write_lf(windows_path, "$ErrorActionPreference = 'Stop'\n$RepoRoot = $env:CANONDRESSGS_REPO_ROOT\n$AssetRoot = $env:CANONDRESSGS_ASSET_ROOT\n$OutputRoot = $env:CANONDRESSGS_OUTPUT_ROOT\n$Executor = $env:CANONDRESSGS_PAPER_EXECUTOR\n$Python = 'python'\nif (-not $RepoRoot -or -not $AssetRoot -or -not $OutputRoot -or -not $Executor) { throw 'CANONDRESSGS roots and paper executor are required' }\nSet-Location -LiteralPath $RepoRoot\n\n" + "\n".join(row["windows_command"] for row in rows) + "\n")
    return [json_path, csv_path, linux_path, windows_path]
