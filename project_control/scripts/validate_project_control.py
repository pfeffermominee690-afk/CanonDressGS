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
}


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
    raw = git(repo, "worktree", "list", "--porcelain")
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
    branches: dict[str, list[str]] = defaultdict(list)
    heads: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        missing = REQUIRED_WORKTREE_FIELDS - set(row)
        if missing:
            errors.append(f"{row.get('id', '<unknown>')} missing fields: {sorted(missing)}")
        if row.get("status_class") not in STATUS_CLASSES:
            errors.append(f"{row.get('id')} invalid status_class: {row.get('status_class')}")
        if row.get("safe_to_remove") is not False:
            errors.append(f"{row.get('id')} safe_to_remove must initially be false")
        branches[str(row.get("branch"))].append(str(row.get("id")))
        heads[str(row.get("head"))].append(str(row.get("id")))
        expected = EXPECTED_HEADS.get(row.get("branch"))
        if expected and row.get("head") != expected:
            errors.append(f"Known milestone mismatch: {row.get('branch')} expected {expected}, got {row.get('head')}")
        for field in ("latest_report", "latest_summary_json"):
            value = row.get(field)
            if value and not Path(value).exists():
                errors.append(f"Missing registered {field}: {value}")
    duplicate_branches = {key: value for key, value in branches.items() if len(value) > 1}
    if duplicate_branches:
        errors.append(f"Duplicate branches: {duplicate_branches}")
    duplicate_heads = {key: value for key, value in heads.items() if len(value) > 1}
    if duplicate_heads:
        warnings.append(f"Duplicate HEAD groups detected (allowed when documented): {duplicate_heads}")
    print(f"Duplicate branch detection: {'PASS' if not duplicate_branches else 'FAIL'}")
    print(f"Duplicate HEAD detection: PASS ({len(duplicate_heads)} group(s) reported)")

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

    artifact_registry = parsed.get("ARTIFACT_REGISTRY.yaml", {})
    artifact_missing = 0
    for artifact in artifact_registry.get("artifacts", []):
        for value in artifact.get("metadata_sources", []):
            if value and re.match(r"^[A-Za-z]:[\\/]", value) and not Path(value).exists():
                artifact_missing += 1
                errors.append(f"Artifact metadata source missing: {value}")
        local_root = artifact.get("local_copy_path")
        if artifact.get("local_copy_status") == "PRESENT" and local_root and not Path(local_root).exists():
            artifact_missing += 1
            errors.append(f"Artifact local copy marked PRESENT but missing: {local_root}")
    print(f"Artifact path existence: {'PASS' if artifact_missing == 0 else 'FAIL'}")

    dataset_registry = parsed.get("DATASET_REGISTRY.yaml", {})
    subject00 = next((item for item in dataset_registry.get("datasets", []) if item.get("id") == "DATASET-THUMAN4-SUBJECT00-ARCHIVE"), None)
    if not subject00 or subject00.get("status") != "DOWNLOADED_LOCAL_PENDING_VERIFICATION":
        errors.append("subject00 archive status is missing or incorrect")
    elif not Path(subject00["local_path"]).exists():
        errors.append("subject00 archive local path does not exist")

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

    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    print(f"Validation result: {'PASS' if not errors else 'FAIL'} ({len(errors)} error(s), {len(warnings)} warning(s))")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
