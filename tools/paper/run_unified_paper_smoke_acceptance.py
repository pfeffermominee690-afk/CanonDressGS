from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from PIL import Image, ImageDraw

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from tools.paper.aggregation import aggregate_evaluations
from tools.paper.method_adapters import adapter_for
from tools.paper.paper_exports import export_figure_layouts, export_tables
from tools.paper.smoke_contract import (
    EXPECTED_SMOKE_IDS,
    SMOKE_MARKER,
    assert_no_formal_destination,
    create_smoke_attempt,
    git_blob,
    load_smoke_registry,
    sha256_file,
    smoke_output_root,
    write_status,
)
from tools.paper.smoke_runtime import environment_report, run_smoke_experiment
from tools.paper.verify_seen_outfit_paper_assets import tree_metadata_fingerprint, verify_manifest


DEFAULT_SMOKE_REGISTRY = "paper_protocol/smoke/smoke_experiment_registry.yaml"
DEFAULT_FORMAL_REGISTRY = "paper_protocol/experiment_registry.yaml"
DEFAULT_CONFIG = "configs/paper/aaai27_seen_outfit_explicit_basis_v1.yaml"
DEFAULT_MANIFEST = "paper_protocol/frozen_asset_manifest.json"


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo_root, text=True).strip()


def _tree_or_absent(path: Path) -> str:
    return tree_metadata_fingerprint(path) if path.is_dir() else "ABSENT"


def _snapshot_attempt(
    attempt: Path,
    experiment: Mapping[str, Any],
    repo_root: Path,
    smoke_registry_path: Path,
    formal_registry_path: Path,
    config_path: Path,
    manifest_path: Path,
    preflight: Mapping[str, Any],
    command: Sequence[str],
) -> None:
    for source in (smoke_registry_path, formal_registry_path, config_path, manifest_path):
        shutil.copy2(source, attempt / "contract" / source.name)
    (attempt / "contract/smoke_contract.json").write_text(
        json.dumps(dict(experiment), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (attempt / "preflight/preflight.json").write_text(
        json.dumps(preflight, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    provenance = {
        "marker": SMOKE_MARKER,
        "paper_final_eligible": False,
        "git": {
            "commit": _git(repo_root, "rev-parse", "HEAD"),
            "branch": _git(repo_root, "branch", "--show-current"),
            "clean": not bool(_git(repo_root, "status", "--porcelain=v1")),
        },
        "command": list(command),
        "environment": environment_report(),
        "experiment": dict(experiment),
    }
    (attempt / "provenance/run_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_status(attempt, "PREFLIGHT_PASS", optimizer_steps=0)


def _entry_preflight(experiment: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    adapter_experiment = {**experiment, "experiment_id": experiment["smoke_experiment_id"]}
    adapter = adapter_for(adapter_experiment, config)
    steps = int(experiment["optimizer_steps"])
    checks = {
        "adapter_exists": True,
        "seed_is_zero": experiment.get("seed") == 0,
        "smoke_steps_exact": steps in {0, 5, 20},
        "seen_split_exact": list(config["data"]["seen_outfits"]) == ["O01", "O02", "O03", "O04", "O08"],
        "o07_not_training": config["data"]["held_out_diagnostic"] == "O07",
        "o06_unused": config["data"]["unused_reserve"] == "O06",
        "no_outfit_id": experiment.get("outfit_id_in_model") is False,
        "no_target_forward_image": experiment.get("target_forward_input") is False,
        "basis_rank_matches": int(experiment.get("basis_rank", 4)) <= int(config["basis"]["rank"]),
        "optimizer_scope_registered": (steps == 0 and adapter.parameter_scope in {"none"}) or (steps > 0 and adapter.parameter_scope != "none"),
        "paper_final_ineligible": experiment.get("paper_final_eligible") is False,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "experiment_id": experiment["smoke_experiment_id"],
        "adapter": type(adapter).__name__,
        "optimizer_parameter_scope": adapter.parameter_scope if steps else "none",
        "checks": checks,
    }


def _metric_cell(value: float | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {"mean": float(value), "std": None, "aggregation": "single_seed_smoke"}


def _table_source(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    method_ids = {
        "B0": "S-B0", "B1": "S-B1", "B2": "S-B2", "B3": "S-B3",
        "B4": "S-B4", "B5": "S-B5", "Ours": "S-OURS",
    }
    methods = {}
    for label, identifier in method_ids.items():
        evaluated = json.loads(Path(results[identifier]["evaluated_metrics"]).read_text(encoding="utf-8"))
        methods[label] = {key: _metric_cell(value) for key, value in evaluated["metrics"].items()}
    ablations = {}
    for identifier in ("S-A1", "S-A5", "S-A7"):
        evaluated = json.loads(Path(results[identifier]["evaluated_metrics"]).read_text(encoding="utf-8"))
        ablations[identifier] = {key: _metric_cell(value) for key, value in evaluated["metrics"].items()}
    return {
        "marker": "SMOKE_ONLY",
        "paper_final_eligible": False,
        "methods": methods,
        "ablations": ablations,
        "held_out": {
            "O07 teacher": {"status": "REFERENCE_ONLY"},
            "O07 projection": {"status": "FAIL"},
            "O07 reference prediction": {"status": "FAIL"},
        },
    }


def _write_not_run_panel(path: Path, label: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (256, 128), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 28), fill=(110, 0, 0))
    draw.text((6, 7), "SMOKE ONLY - NOT PAPER RESULTS", fill="white")
    draw.text((12, 55), label, fill=(40, 40, 40))
    draw.text((12, 78), "NOT_RUN IN THIS SMOKE MATRIX", fill=(110, 0, 0))
    image.save(path)
    return str(path)


def _figure_sources(attempts: Mapping[str, Path], placeholder_root: Path) -> dict[str, Any]:
    ours = attempts["S-OURS"]
    b5 = attempts["S-B5"]
    figure2 = []
    for outfit in ("O01", "O02", "O03", "O04", "O08"):
        for view in ("cond_000000", "cond_000318", "cond_000017", "cond_000347"):
            ours_source = ours / "visuals/source_panels"
            figure2.extend([
                str(ours_source / f"{outfit}_{view}_base.png"),
                str(ours_source / f"{outfit}_{view}_target.png"),
                str(ours_source / f"{outfit}_{view}_teacher.png"),
                str(ours_source / f"{outfit}_{view}_ours.png"),
                str(b5 / "visuals/episodes" / f"{outfit}_{view}_prediction.png"),
            ])
    ours_contact = str(ours / "visuals/S-OURS_five_outfit_four_view_contact_sheet.png")
    swap = str(ours / "visuals/Ours_reference_swap.png")
    a1 = str(attempts["S-A1"] / "visuals/S-A1_five_outfit_four_view_contact_sheet.png")
    a5 = str(attempts["S-A5"] / "visuals/S-A5_five_outfit_four_view_contact_sheet.png")
    o07 = [str(ours / "visuals/O07" / f"{name}.png") for name in ("teacher", "projection", "prediction", "O03_endpoint")]
    rank_sources = [
        _write_not_run_panel(placeholder_root / "A1_K1_NOT_RUN.png", "A1 K=1"),
        _write_not_run_panel(placeholder_root / "A1_K2_NOT_RUN.png", "A1 K=2"),
        a1,
        _write_not_run_panel(placeholder_root / "A1_K4_NOT_RUN.png", "A1 K=4"),
    ]
    supervision_sources = [
        a5,
        _write_not_run_panel(placeholder_root / "A5_old_gradient_NOT_RUN.png", "old gradient"),
        _write_not_run_panel(placeholder_root / "A5_signed_raw_logit_NOT_RUN.png", "signed raw-logit"),
        _write_not_run_panel(placeholder_root / "A5_CS_PASS_NOT_RUN.png", "CS-PASS"),
    ]
    return {
        "marker": "SMOKE_ONLY",
        "figures": {
            "figure_1": {"source_paths": [ours_contact, swap]},
            "figure_2": {"source_paths": figure2},
            "figure_3": {"source_paths": [str(ours / "visuals/source_panels" / f"{outfit}_cond_000000_ours.png") for outfit in ("O01", "O02", "O03", "O04", "O08")]},
            "figure_4": {"source_paths": rank_sources},
            "figure_5": {"source_paths": supervision_sources},
            "figure_6": {"source_paths": o07},
        },
    }


def _visual_review_index(attempts: Mapping[str, Path]) -> dict[str, str]:
    ours = attempts["S-OURS"]
    return {
        "ours_five_outfit_four_view_contact_sheet": str(ours / "visuals/S-OURS_five_outfit_four_view_contact_sheet.png"),
        "ours_reference_swap": str(ours / "visuals/Ours_reference_swap.png"),
        "b0_output": str(attempts["S-B0"] / "visuals/S-B0_five_outfit_four_view_contact_sheet.png"),
        "b1_teacher_upper_bound": str(attempts["S-B1"] / "visuals/S-B1_five_outfit_four_view_contact_sheet.png"),
        "b5_legacy_output": str(attempts["S-B5"] / "visuals/S-B5_five_outfit_four_view_contact_sheet.png"),
        "a1_rank3_output": str(attempts["S-A1"] / "visuals/S-A1_five_outfit_four_view_contact_sheet.png"),
        "a5_endpoint_output": str(attempts["S-A5"] / "visuals/S-A5_five_outfit_four_view_contact_sheet.png"),
        "o07_limitation_layout": str(ours / "visuals/O07_held_out_limitation_layout.png"),
    }


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    output_parent = args.output_root.resolve()
    artifact_root = args.artifact_root.resolve()
    assert_no_formal_destination(artifact_root)
    smoke_registry_path = repo_root / args.smoke_registry
    formal_registry_path = repo_root / DEFAULT_FORMAL_REGISTRY
    config_path = repo_root / DEFAULT_CONFIG
    manifest_path = repo_root / DEFAULT_MANIFEST
    registry = load_smoke_registry(smoke_registry_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_head = _git(repo_root, "rev-parse", "HEAD")
    if actual_head != args.expected_commit:
        raise RuntimeError(f"run commit mismatch: {actual_head}")
    if _git(repo_root, "status", "--porcelain=v1"):
        raise RuntimeError("smoke run requires a clean worktree")
    if git_blob(repo_root, DEFAULT_FORMAL_REGISTRY) != registry["formal_registry_git_blob"]:
        raise RuntimeError("formal registry Git blob differs from the preregistered smoke contract")
    environment = environment_report()
    if environment["cuda_visible_devices"] != "-1" or environment["torch_cuda_available"]:
        raise RuntimeError("smoke acceptance must run with CUDA_VISIBLE_DEVICES=-1")
    formal_registry_before = sha256_file(formal_registry_path)
    formal_blob_before = git_blob(repo_root, DEFAULT_FORMAL_REGISTRY)
    formal_output = output_parent / "AAAI27-SEEN-OUTFIT-PAPER"
    formal_output_before = _tree_or_absent(formal_output)
    asset_report = verify_manifest(manifest, repo_root, args.asset_root.resolve(), verify_external=True)
    if asset_report["status"] != "PASS" or asset_report["asset_count"] != 19:
        raise RuntimeError("PAPER_ASSET_MISMATCH")
    entry_reports = [_entry_preflight(item, config) for item in registry["experiments"]]
    if any(item["status"] != "PASS" for item in entry_reports):
        raise RuntimeError("smoke entry preflight failed")
    preflight = {
        "status": "PASS",
        "marker": SMOKE_MARKER,
        "asset_verification": asset_report,
        "git": {"commit": actual_head, "branch": _git(repo_root, "branch", "--show-current"), "clean": True},
        "environment": environment,
        "seed": 0,
        "seen_outfits": list(config["data"]["seen_outfits"]),
        "held_out_excluded_from_training": config["data"]["held_out_diagnostic"] == "O07",
        "unused_outfit": config["data"]["unused_reserve"],
        "target_forward_input": False,
        "outfit_id_in_model": False,
        "entry_reports": entry_reports,
    }
    task_root = smoke_output_root(output_parent)
    attempts: dict[str, Path] = {}
    results: dict[str, dict[str, Any]] = {}
    for source in registry["experiments"]:
        experiment = dict(source)
        experiment["frozen_asset_fingerprint"] = registry["frozen_asset_fingerprint"]
        identifier = experiment["smoke_experiment_id"]
        attempt = create_smoke_attempt(task_root / identifier / "seed_0")
        attempts[identifier] = attempt
        _snapshot_attempt(
            attempt, experiment, repo_root, smoke_registry_path, formal_registry_path,
            config_path, manifest_path, preflight, sys.argv,
        )
        try:
            results[identifier] = run_smoke_experiment(experiment, config, attempt)
        except Exception as error:
            write_status(attempt, "FAILED", failure=f"{type(error).__name__}: {error}")
            raise
    first_evaluation = json.loads(Path(results["S-OURS"]["evaluated_metrics"]).read_text(encoding="utf-8"))
    try:
        aggregate_evaluations([first_evaluation])
    except ValueError as error:
        formal_rejection = str(error)
    else:
        raise RuntimeError("formal aggregate incorrectly accepted one seed")
    if "MISSING_REQUIRED_PAPER_SEEDS" not in formal_rejection:
        raise RuntimeError("formal aggregate rejection status is incorrect")
    table_source = _table_source(results)
    table_source_path = artifact_root / "tables/smoke_table_source.json"
    table_source_path.parent.mkdir(parents=True, exist_ok=True)
    table_source_path.write_text(json.dumps(table_source, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tables = export_tables(table_source, artifact_root / "tables")
    figure_source = _figure_sources(attempts, artifact_root / "figures/source_placeholders")
    figure_source_path = artifact_root / "figures/smoke_figure_source_manifest.json"
    figure_source_path.parent.mkdir(parents=True, exist_ok=True)
    figure_source_path.write_text(json.dumps(figure_source, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    figures = export_figure_layouts(
        artifact_root / "figures", synthetic=False, source_manifest=figure_source, smoke_only=True,
    )
    formal_registry_after = sha256_file(formal_registry_path)
    formal_blob_after = git_blob(repo_root, DEFAULT_FORMAL_REGISTRY)
    formal_output_after = _tree_or_absent(formal_output)
    if formal_registry_before != formal_registry_after or formal_blob_before != formal_blob_after:
        raise RuntimeError("formal registry changed during smoke")
    if formal_output_before != formal_output_after:
        raise RuntimeError("formal paper output tree changed during smoke")
    if _git(repo_root, "status", "--porcelain=v1"):
        raise RuntimeError("smoke runner dirtied the execution worktree")
    summary = {
        "task_id": registry["task_id"],
        "status": "MANUAL_REVIEW_REQUIRED",
        "marker": SMOKE_MARKER,
        "run_commit": actual_head,
        "branch": _git(repo_root, "branch", "--show-current"),
        "worktree": str(repo_root),
        "output_root": str(task_root),
        "artifact_root": str(artifact_root),
        "environment": environment,
        "preflight": preflight,
        "experiment_count": len(results),
        "successful_experiment_count": len(results),
        "attempts": {key: str(value) for key, value in attempts.items()},
        "results": results,
        "formal_aggregate_rejection": formal_rejection,
        "tables": tables,
        "figures": figures,
        "visual_review_index": _visual_review_index(attempts),
        "formal_registry": {
            "before_sha256": formal_registry_before,
            "after_sha256": formal_registry_after,
            "before_git_blob": formal_blob_before,
            "after_git_blob": formal_blob_after,
        },
        "historical_formal_output": {
            "before_fingerprint": formal_output_before,
            "after_fingerprint": formal_output_after,
            "unchanged": True,
        },
        "paper_final_outputs_created": False,
    }
    summary_path = task_root / "smoke_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "summary": str(summary_path), "experiments": len(results)}, ensure_ascii=False))
    return summary


def export_only(args: argparse.Namespace) -> dict[str, Any]:
    summary = json.loads(args.summary.resolve().read_text(encoding="utf-8"))
    artifact_root = args.artifact_root.resolve()
    assert_no_formal_destination(artifact_root)
    attempts = {key: Path(value) for key, value in summary["attempts"].items()}
    results = summary["results"]
    table_source = _table_source(results)
    table_source_path = artifact_root / "tables/smoke_table_source.json"
    table_source_path.parent.mkdir(parents=True, exist_ok=True)
    table_source_path.write_text(json.dumps(table_source, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tables = export_tables(table_source, artifact_root / "tables")
    figure_source = _figure_sources(attempts, artifact_root / "figures/source_placeholders")
    figure_source_path = artifact_root / "figures/smoke_figure_source_manifest.json"
    figure_source_path.parent.mkdir(parents=True, exist_ok=True)
    figure_source_path.write_text(json.dumps(figure_source, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    figures = export_figure_layouts(
        artifact_root / "figures", synthetic=False, source_manifest=figure_source, smoke_only=True,
    )
    result = {
        "status": "MANUAL_REVIEW_REQUIRED",
        "marker": SMOKE_MARKER,
        "source_run_commit": summary["run_commit"],
        "source_attempts": summary["attempts"],
        "optimizer_steps_executed": 0,
        "tables": tables,
        "figures": figures,
    }
    (artifact_root / "artifact_export_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "artifact_root": str(artifact_root)}))
    return result


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = args.summary.resolve()
    review = json.loads(args.review_json.resolve().read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if review.get("status") != "PASS" or review.get("images_actually_opened") is not True:
        raise ValueError("manual visual review is not a PASS")
    expected = set(summary["visual_review_index"])
    if set(review.get("observations", {})) != expected:
        raise ValueError("manual visual review does not cover all eight preregistered categories")
    for identifier, attempt_value in summary["attempts"].items():
        attempt = Path(attempt_value)
        adjudication = {
            "status": "SMOKE_ACCEPTED",
            "marker": SMOKE_MARKER,
            "paper_final_eligible": False,
            "manual_visual_review": review,
        }
        (attempt / "final_adjudication/SMOKE_ACCEPTANCE.json").write_text(
            json.dumps(adjudication, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        write_status(attempt, "SMOKE_ACCEPTED", optimizer_steps=summary["results"][identifier]["optimizer_steps"])
    summary["status"] = "SMOKE_ACCEPTED"
    summary["manual_visual_review"] = review
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "SMOKE_ACCEPTED", "summary": str(summary_path)}))
    return summary


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Run isolated unified-paper smoke acceptance")
    sub = root.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-all")
    run.add_argument("--repo-root", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--asset-root", type=Path, required=True)
    run.add_argument("--artifact-root", type=Path, required=True)
    run.add_argument("--expected-commit", required=True)
    run.add_argument("--smoke-registry", default=DEFAULT_SMOKE_REGISTRY)
    finish = sub.add_parser("finalize")
    finish.add_argument("--summary", type=Path, required=True)
    finish.add_argument("--review-json", type=Path, required=True)
    export = sub.add_parser("export-only")
    export.add_argument("--summary", type=Path, required=True)
    export.add_argument("--artifact-root", type=Path, required=True)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "run-all":
        run_all(args)
    elif args.command == "finalize":
        finalize(args)
    else:
        export_only(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
