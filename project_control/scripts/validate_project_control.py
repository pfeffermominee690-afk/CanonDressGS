#!/usr/bin/env python3
"""Validate CanonDressGS control-center registries and live Git metadata."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


STATUS_CLASSES = {
    "ACTIVE",
    "SEALED_CANONICAL",
    "SEALED_HISTORICAL",
    "DOCUMENTATION_ACTIVE",
    "DATA_STAGING",
    "FAILED_PRESERVED",
    "SUPERSEDED",
    "UNKNOWN_REQUIRES_REVIEW",
}
REQUIRED_WORKTREE_FIELDS = {
    "id", "local_path", "cloud_path", "repository_common_dir", "branch", "head",
    "upstream", "origin_head", "cloud_head", "clean_local", "clean_cloud",
    "source_branch", "source_head", "purpose", "task_id", "created_date",
    "latest_report", "latest_summary_json", "formal_output_root",
    "artifact_fingerprint", "push_status", "cloud_sync_status",
    "paper_final_count", "status_class", "superseded_by", "safe_to_remove", "notes",
}
EXPECTED_HEADS = {
    "paper/aaai27-p0-formal-candidate-runs-20260721": "32e6058ea64078f5ccded6f86e8bffe3cb81b30e",
    "paper/aaai27-p0-color-spatial-soft-control-evaluations-20260722": "371d812864614cd561e33edfe3f6c043b38415d2",
    "paper/aaai27-deterministic-initialization-protocol-20260721": "42a28b386f6f32e23e5e16680408820efd8a65c7",
    "paper/aaai27-p0-candidate-adapters-runner-20260721": "d75c4fa5a426fe90314eba08e94509940ecafb23",
    "paper/aaai27-p0-evaluation-protocol-repair-20260721": "8578fb3143dd916ff7e42240e7508a29c784d995",
    "paper/aaai27-manuscript-stable-sections-20260721": "1a5a55fee0f6a3344debdc63f93bad60944f9e96",
    "research/continuous-control-artifact-root-cause-20260722": "68623c36eee70c0b41aefb480f8f85e668eae201",
    "research/continuous-control-causal-attribution-20260722": "8c43524b7ca0ee8b3c795dbe349466f30b29354f",
    "research/geometry-dual-support-micro-pilot-20260722": "b216acd02323c822a7aca12f424efed6ba2e0a81",
    "research/dual-support-all-pair-evaluation-20260722": "d802f427f1e9c23595e1bdb4f10135f7de2c3f08",
}
FORMAL_CHAIN_BRANCHES = {
    "research/continuous-control-causal-attribution-20260722",
    "research/geometry-dual-support-micro-pilot-20260722",
    "research/dual-support-all-pair-evaluation-20260722",
    "research/reference-conditioned-dual-support-controller-20260722",
}
SUBJECT00_FINGERPRINT = "2c0f894f70d944fa8d6ebd48ab1188b78cb19b673f0b7c006f1922a592b1ea7b"


def git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip() if result.returncode == 0 else ""


def norm(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def live_worktrees(repo: Path) -> dict[str, dict[str, str]]:
    raw = git(repo, "-c", "core.quotePath=false", "worktree", "list", "--porcelain")
    result: dict[str, dict[str, str]] = {}
    for block in re.split(r"\r?\n\r?\n", raw):
        if not block.strip():
            continue
        row: dict[str, str] = {}
        for line in block.splitlines():
            key, _, value = line.partition(" ")
            if key == "branch":
                row[key] = value.removeprefix("refs/heads/")
            elif key == "detached":
                row["branch"] = "DETACHED"
            else:
                row[key] = value
        result[norm(row["worktree"])] = row
    return result


def is_ancestor(repo: Path, older: str, newer: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", older, newer],
        capture_output=True,
    )
    return result.returncode == 0


def markdown_section(text: str, heading: str) -> str:
    match = re.search(
        rf"(?ims)^##\s+{re.escape(heading)}\s*$\n(.*?)(?=^##\s+|\Z)", text,
    )
    return match.group(1) if match else ""


def has_dependency_cycle(graph: dict[str, list[str]]) -> bool:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for dependency in graph.get(node, []):
            if dependency not in graph or visit(dependency):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in graph)


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    root = repo / "project_control"
    errors: list[str] = []
    warnings: list[str] = []

    yaml_files = sorted(root.glob("*.yaml"))
    parsed: dict[str, Any] = {}
    for path in yaml_files:
        try:
            parsed[path.name] = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"YAML/JSON-compatible parse failed: {path.name}: {exc}")
    print(f"YAML parse: {'PASS' if len(parsed) == len(yaml_files) else 'FAIL'} ({len(parsed)}/{len(yaml_files)})")

    json_files = sorted(root.rglob("*.json"))
    json_ok = 0
    for path in json_files:
        try:
            json.loads(path.read_text(encoding="utf-8"))
            json_ok += 1
        except Exception as exc:  # noqa: BLE001
            errors.append(f"JSON parse failed: {path.relative_to(root)}: {exc}")
    print(f"JSON parse: {'PASS' if json_ok == len(json_files) else 'FAIL'} ({json_ok}/{len(json_files)})")

    registry = parsed.get("WORKTREE_REGISTRY.yaml", {})
    rows = registry.get("worktrees", [])
    if registry.get("scan", {}).get("discovered_git_worktree_count") != len(rows):
        errors.append("Worktree count does not match registry row count")
    branches: dict[str, list[dict[str, Any]]] = defaultdict(list)
    heads: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        missing = REQUIRED_WORKTREE_FIELDS - set(row)
        if missing:
            errors.append(f"{row.get('id', '<unknown>')} missing fields: {sorted(missing)}")
        if row.get("status_class") not in STATUS_CLASSES:
            errors.append(f"{row.get('id')} invalid status_class: {row.get('status_class')}")
        if row.get("safe_to_remove") is not False:
            errors.append(f"{row.get('id')} safe_to_remove must be false in this refresh")
        branches[str(row.get("branch"))].append(row)
        heads[str(row.get("head"))].append(row)
        expected = EXPECTED_HEADS.get(row.get("branch"))
        if expected and row.get("head") != expected:
            errors.append(f"Known milestone mismatch: {row.get('branch')} expected {expected}, got {row.get('head')}")
        for field in ("latest_report", "latest_summary_json"):
            value = row.get(field)
            if value and not Path(value).exists():
                errors.append(f"Missing registered {field}: {value}")
    duplicate_branches = {key: value for key, value in branches.items() if len(value) > 1}
    if duplicate_branches:
        errors.append(f"Duplicate branches: {sorted(duplicate_branches)}")
    duplicate_heads = {key: value for key, value in heads.items() if len(value) > 1}
    unexplained_duplicate_heads = {
        head: [row.get("id") for row in group]
        for head, group in duplicate_heads.items()
        if any(not row.get("duplicate_head_explanation") for row in group)
    }
    if unexplained_duplicate_heads:
        errors.append(f"Duplicate HEAD groups lack explanations: {unexplained_duplicate_heads}")
    print(f"Duplicate branch detection: {'PASS' if not duplicate_branches else 'FAIL'}")
    print(f"Duplicate HEAD explanation check: {'PASS' if not unexplained_duplicate_heads else 'FAIL'} ({len(duplicate_heads)} group(s))")

    live = live_worktrees(repo)
    registered_paths = {norm(row["local_path"]): row for row in rows}
    missing_live = sorted(set(registered_paths) - set(live))
    unregistered_live = sorted(set(live) - set(registered_paths))
    if missing_live:
        errors.append(f"Registered worktrees missing locally: {missing_live}")
    if unregistered_live:
        errors.append(f"Live worktrees missing from registry: {unregistered_live}")
    print(f"Missing worktree detection: {'PASS' if not missing_live and not unregistered_live else 'FAIL'}")
    mismatch_count = 0
    for key in sorted(set(live) & set(registered_paths)):
        expected = registered_paths[key]
        actual = live[key]
        if expected["branch"] != actual.get("branch", "DETACHED"):
            errors.append(f"Branch mismatch at {expected['local_path']}")
            mismatch_count += 1
        if expected["head"] != actual.get("HEAD"):
            if expected["branch"] == "project/canondressgs-control-center-20260722" and is_ancestor(repo, expected["head"], actual["HEAD"]):
                warnings.append("Control-center self row is an ancestor snapshot, as documented in README.md")
            else:
                errors.append(f"HEAD mismatch at {expected['local_path']}: {expected['head']} != {actual.get('HEAD')}")
                mismatch_count += 1
    print(f"Branch/HEAD mismatch detection: {'PASS' if mismatch_count == 0 else 'FAIL'}")

    chain_errors = 0
    for row in rows:
        if row.get("branch") not in FORMAL_CHAIN_BRANCHES:
            continue
        summary_path = Path(row["latest_summary_json"])
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary_branch = summary.get("run_branch")
        if summary_branch and summary_branch != row["branch"]:
            errors.append(f"Summary branch mismatch for {row['branch']}: {summary_branch}")
            chain_errors += 1
        source_head = summary.get("source_head")
        if source_head != row.get("source_head"):
            errors.append(f"Summary source HEAD mismatch for {row['branch']}: {source_head} != {row.get('source_head')}")
            chain_errors += 1
        execution_head = summary.get("execution_head")
        if execution_head and execution_head != row["head"] and not is_ancestor(Path(row["local_path"]), execution_head, row["head"]):
            errors.append(f"HEAD_OR_REPORT_MISMATCH for {row['branch']}: execution {execution_head}, archive {row['head']}")
            chain_errors += 1
        report_text = Path(row["latest_report"]).read_text(encoding="utf-8", errors="replace")
        if source_head and source_head[:8] not in report_text:
            errors.append(f"Formal report does not cite summary source HEAD for {row['branch']}")
            chain_errors += 1
    print(f"Report/summary HEAD consistency: {'PASS' if chain_errors == 0 else 'FAIL'}")

    artifact_registry = parsed.get("ARTIFACT_REGISTRY.yaml", {})
    artifacts = artifact_registry.get("artifacts", [])
    cloud_count = sum(bool(item.get("cloud_output_root")) for item in artifacts)
    if artifact_registry.get("cloud_output_roots_registered_count") != cloud_count:
        errors.append("Artifact cloud-output-root count is inconsistent")
    artifact_missing = 0
    for artifact in artifacts:
        for value in artifact.get("metadata_sources", []):
            if value and re.match(r"^[A-Za-z]:[\\/]", value) and not Path(value).exists():
                artifact_missing += 1
                errors.append(f"Artifact metadata source missing: {value}")
        local_root = artifact.get("local_copy_path")
        if artifact.get("local_copy_status") == "PRESENT" and local_root and not Path(local_root).exists():
            artifact_missing += 1
            errors.append(f"Artifact local copy marked PRESENT but missing: {local_root}")
        if artifact.get("paper_final_count") not in (None, 0):
            errors.append(f"Artifact has nonzero PAPER_FINAL count: {artifact.get('id')}")
    print(f"Artifact path existence: {'PASS' if artifact_missing == 0 else 'FAIL'}")

    dataset_registry = parsed.get("DATASET_REGISTRY.yaml", {})
    dataset_missing = 0
    for dataset in dataset_registry.get("datasets", []):
        local_path = dataset.get("local_path")
        if local_path and re.match(r"^[A-Za-z]:[\\/]", local_path) and not Path(local_path).exists():
            dataset_missing += 1
            errors.append(f"Dataset local path missing: {local_path}")
        for report in dataset.get("reports", []):
            if re.match(r"^[A-Za-z]:[\\/]", report) and not Path(report).exists():
                dataset_missing += 1
                errors.append(f"Dataset report missing: {report}")
    print(f"Dataset path existence: {'PASS' if dataset_missing == 0 else 'FAIL'}")
    subject00 = next((item for item in dataset_registry.get("datasets", []) if item.get("id") == "DATASET-THUMAN4-SUBJECT00-ARCHIVE"), None)
    subject_ok = bool(
        subject00
        and subject00.get("raw_data_status") == "SUBJECT00_CLOUD_DATA_READY"
        and subject00.get("tree_fingerprint") == SUBJECT00_FINGERPRINT
        and subject00.get("mmlphuman_status") == "QUEUED_NOT_STARTED"
        and subject00.get("garment_benchmark_status") == "NOT_STARTED"
        and subject00.get("cross_identity_evaluation") == "NOT_STARTED"
    )
    if not subject_ok:
        errors.append("subject00 status/fingerprint/readiness boundary is incorrect")
    print(f"subject00 fingerprint/status fields: {'PASS' if subject_ok else 'FAIL'}")

    link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    bad_links = []
    for path in sorted(root.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        for target in link_pattern.findall(text):
            target = target.strip().strip("<>").split("#", 1)[0]
            if not target or re.match(r"^(?:https?://|mailto:|[A-Za-z]:[\\/]|/)", target):
                continue
            if not (path.parent / target).resolve().exists():
                bad_links.append(f"{path.relative_to(root)} -> {target}")
    if bad_links:
        errors.extend(f"Broken Markdown link: {item}" for item in bad_links)
    print(f"Markdown links/path checks: {'PASS' if not bad_links else 'FAIL'}")

    claims_text = (root / "CLAIM_STATUS.md").read_text(encoding="utf-8")
    confirmed = markdown_section(claims_text, "CONFIRMED").lower()
    partial = markdown_section(claims_text, "PARTIAL").lower()
    not_confirmed = markdown_section(claims_text, "NOT_CONFIRMED").lower()
    limitations = markdown_section(claims_text, "FAILED / LIMITATIONS").lower()
    required_confirmed = [
        "geometry interpolation is the primary source", "dual-support endpoint parity",
        "all `10/10` seen-garment pairs", "macro garment lpips", "silhouette iou",
        "boundary f-score", "identity-contamination maximum remains zero",
    ]
    required_partial = ["reference soft control", "dual-support intermediate-state utility", "reference-conditioned controller design"]
    required_not = [
        "trained reference-conditioned dual-support control", "unseen garment generation",
        "cross-identity generalization", "second-identity performance", "novel view",
        "novel pose", "second public dataset generalization",
    ]
    required_limits = ["2x", "2.625864x", "seven of ten", "o01_o03", "closed, seen-garment", "held-view or held-pose"]
    claim_missing = [
        *(f"CONFIRMED:{item}" for item in required_confirmed if item not in confirmed),
        *(f"PARTIAL:{item}" for item in required_partial if item not in partial),
        *(f"NOT_CONFIRMED:{item}" for item in required_not if item not in not_confirmed),
        *(f"LIMITATIONS:{item}" for item in required_limits if item not in limitations),
    ]
    if claim_missing:
        errors.append(f"Claim-state consistency missing entries: {claim_missing}")
    print(f"Claim-state consistency: {'PASS' if not claim_missing else 'FAIL'}")

    milestone_registry = parsed.get("MILESTONE_REGISTRY.yaml", {})
    graph = milestone_registry.get("task_dependencies", {})
    cycle = has_dependency_cycle(graph) if graph else True
    if cycle:
        errors.append("Task dependency graph is missing, references unknown tasks, or contains a cycle")
    print(f"Task dependency cycle detection: {'PASS' if not cycle else 'FAIL'}")

    controller = next((row for row in rows if row.get("branch") == "research/reference-conditioned-dual-support-controller-20260722"), None)
    controller_ok = bool(
        controller
        and controller.get("status_class") == "ACTIVE"
        and controller.get("cloud_sync_status") == "UNVERIFIED_CLOUD_REMOTE_UNREACHABLE"
        and controller.get("paper_final_count") == 0
    )
    if not controller_ok:
        errors.append("Controller conservative ACTIVE/cloud-unverified classification is inconsistent")

    diff = subprocess.run(
        ["git", "-C", str(repo), "diff", "--check"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if diff.returncode:
        errors.append(f"git diff --check failed: {diff.stdout.strip()} {diff.stderr.strip()}")
    print(f"git diff --check: {'PASS' if diff.returncode == 0 else 'FAIL'}")

    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    print(f"Validation result: {'PASS' if not errors else 'FAIL'} ({len(errors)} error(s), {len(warnings)} warning(s))")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
