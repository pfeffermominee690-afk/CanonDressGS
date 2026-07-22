#!/usr/bin/env python3
"""Refresh the CanonDressGS worktree inventory without touching experiment data.

The registry files use JSON syntax because JSON is a strict, machine-parseable
subset of YAML 1.2.  This keeps the refresh/validation path dependency-free.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
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

EXPECTED_HEADS = {
    "paper/aaai27-p0-formal-candidate-runs-20260721": "32e6058ea64078f5ccded6f86e8bffe3cb81b30e",
    "paper/aaai27-p0-color-spatial-soft-control-evaluations-20260722": "371d812864614cd561e33edfe3f6c043b38415d2",
    "paper/aaai27-deterministic-initialization-protocol-20260721": "42a28b386f6f32e23e5e16680408820efd8a65c7",
    "paper/aaai27-p0-candidate-adapters-runner-20260721": "d75c4fa5a426fe90314eba08e94509940ecafb23",
    "paper/aaai27-p0-evaluation-protocol-repair-20260721": "8578fb3143dd916ff7e42240e7508a29c784d995",
    "paper/aaai27-manuscript-stable-sections-20260721": "1a5a55fee0f6a3344debdc63f93bad60944f9e96",
    "research/continuous-control-artifact-root-cause-20260722": "68623c36eee70c0b41aefb480f8f85e668eae201",
}


def override(status: str, purpose: str, **kwargs: Any) -> dict[str, Any]:
    if status not in STATUS_CLASSES:
        raise ValueError(status)
    return {"status_class": status, "purpose": purpose, **kwargs}


OVERRIDES: dict[str, dict[str, Any]] = {
    "pipeline/imagecond-mvp-20260715": override(
        "UNKNOWN_REQUIRES_REVIEW",
        "Legacy image-conditioned MVP/main working directory",
        notes="Dirty user worktree; classification requires owner review.",
    ),
    "sprint/aaai27-two-outfit-prep-20260720": override(
        "DATA_STAGING",
        "O01/O08 two-outfit reference preparation",
        task_id="AAAI27-TWO-OUTFIT-PREP",
    ),
    "paper/aaai27-deterministic-initialization-protocol-20260721": override(
        "SEALED_CANONICAL",
        "Deterministic initialization protocol",
        task_id="AAAI27-DETERMINISTIC-INITIALIZATION-PROTOCOL",
    ),
    "paper/aaai27-frozen-experiment-batches-20260720": override(
        "SEALED_CANONICAL",
        "Frozen seen-outfit paper experiment batches",
        task_id="AAAI27-SEEN-OUTFIT-PAPER",
        formal_output_root="/root/autodl-tmp/canondressgs_work/outputs/AAAI27-SEEN-OUTFIT-PAPER",
        artifact_fingerprint="2205acdcf3a7a6f8019a584f4e4b9ac0f50a8cdd3f2b342642a949b9791b22cc",
        paper_final_count=0,
    ),
    "paper/aaai27-manuscript-stable-sections-20260721": override(
        "SEALED_CANONICAL",
        "Stable manuscript sections aligned to P0 contracts",
        task_id="AAAI27-MANUSCRIPT-STABLE-SECTIONS",
        paper_final_count=0,
    ),
    "paper/aaai27-p0-candidate-adapters-runner-20260721": override(
        "SEALED_CANONICAL",
        "P0 candidate adapters and runner",
        task_id="AAAI27-P0-CANDIDATE-ADAPTERS-RUNNER",
        paper_final_count=0,
    ),
    "paper/aaai27-p0-color-spatial-soft-control-20260721": override(
        "SUPERSEDED",
        "Pre-execution staging branch for the color/spatial/soft-control evaluation",
        superseded_by="paper/aaai27-p0-color-spatial-soft-control-evaluations-20260722",
        paper_final_count=0,
        notes="Branch has no verified origin/upstream; it is not a cleanup candidate.",
    ),
    "paper/aaai27-p0-color-spatial-soft-control-evaluations-20260722": override(
        "SEALED_CANONICAL",
        "P0 color, spatial, and soft-control archive",
        task_id="AAAI27-P0-COLOR-SPATIAL-SOFT-CONTROL-002",
        source_branch="paper/aaai27-p0-evaluation-protocol-repair-20260721",
        source_head="8578fb3143dd916ff7e42240e7508a29c784d995",
        latest_report="docs/PAPER/AAAI27_P0_COLOR_EXTENDED_SPATIAL_SOFT_CONTROL_RESULTS_20260722.md",
        latest_summary_json="paper_protocol/reviewer_risk/p0_color_spatial_soft_control_final_summary.json",
        formal_output_root="/root/autodl-tmp/canondressgs_work/outputs/AAAI27-P0-COLOR-SPATIAL-SOFT-CONTROL/attempt_004",
        artifact_fingerprint="f90b9aeaa5ab93a951ae81f91e48dc77792d4b6bd513e6c8af9af8ba481228ab",
        paper_final_count=0,
    ),
    "paper/aaai27-p0-evaluation-protocol-repair-20260721": override(
        "SEALED_CANONICAL",
        "P0 evaluation protocol repair",
        task_id="AAAI27-P0-EVALUATION-PROTOCOL-REPAIR",
        paper_final_count=0,
    ),
    "paper/aaai27-p0-formal-candidate-runs-20260721": override(
        "SEALED_CANONICAL",
        "P0 formal candidate run archive",
        task_id="AAAI27-P0-FORMAL-CANDIDATE-RUNS-001",
        latest_report="docs/PAPER/AAAI27_P0_FORMAL_CANDIDATE_RUN_RESULTS_20260721.md",
        latest_summary_json="paper_protocol/reviewer_risk/p0_formal_candidate_final_summary.json",
        formal_output_root="/root/autodl-tmp/canondressgs_work/outputs/AAAI27-P0-FORMAL-CANDIDATE-RUNS",
        artifact_fingerprint="f90b9aeaa5ab93a951ae81f91e48dc77792d4b6bd513e6c8af9af8ba481228ab",
        paper_final_count=0,
        notes="Origin matches the archive HEAD; cached cloud ref is older and live cloud verification failed.",
    ),
    "paper/aaai27-p0-reviewer-risk-closure-20260721": override(
        "SEALED_HISTORICAL",
        "P0 seed-propagation failure audit",
        task_id="AAAI27-P0-REVIEWER-RISK-CLOSURE",
        paper_final_count=0,
    ),
    "paper/aaai27-real-teaser-figure-20260722": override(
        "DOCUMENTATION_ACTIVE",
        "Real-asset Figure 1 teaser pending author signoff",
        task_id="AAAI27-CANONDRESSGS-TEASER-REAL-ASSETS-001",
        source_branch="paper/aaai27-p0-color-spatial-soft-control-evaluations-20260722",
        source_head="371d812864614cd561e33edfe3f6c043b38415d2",
        latest_report="docs/PAPER/AAAI27_CANONDRESSGS_TEASER_FIGURE_20260722.md",
        latest_summary_json="paper_protocol/figure_manifests/canondressgs_teaser_visual_review.json",
        artifact_fingerprint="6ce8c00d77f4f6c472fee8d03ddaa8aa6086934657abc484fde40d18b6cfe641",
        paper_final_count=0,
        notes="Formal-asset audit PASS; visual status MANUAL_REVIEW_REQUIRED.",
    ),
    "paper/aaai27-reviewer-risk-adjudication-20260721": override(
        "SEALED_HISTORICAL",
        "Reviewer-risk evidence adjudication",
        task_id="AAAI27-REVIEWER-RISK-ADJUDICATION",
        paper_final_count=0,
    ),
    "paper/aaai27-reviewer-risk-paper-materials-20260721": override(
        "SEALED_HISTORICAL",
        "Reviewer-risk paper materials",
        task_id="AAAI27-REVIEWER-RISK-PAPER-MATERIALS",
        paper_final_count=0,
    ),
    "paper/aaai27-seen-outfit-protocol-20260720": override(
        "SEALED_CANONICAL",
        "Seen-outfit paper protocol",
        task_id="AAAI27-SEEN-OUTFIT-PAPER",
        formal_output_root="/root/autodl-tmp/canondressgs_work/outputs/AAAI27-SEEN-OUTFIT-PAPER",
        artifact_fingerprint="ff90540e56c9c2db3fd6a1e45c31a1d1eb5a8a32effabde71158584d9be538bb",
        paper_final_count=0,
    ),
    "paper/aaai27-unified-runner-evaluator-20260720": override(
        "SEALED_CANONICAL",
        "Unified paper runner and evaluator",
        task_id="AAAI27-UNIFIED-RUNNER-EVALUATOR",
        paper_final_count=0,
    ),
    "paper/aaai27-unified-smoke-acceptance-20260720": override(
        "SEALED_CANONICAL",
        "Unified paper smoke acceptance",
        task_id="AAAI27-UNIFIED-SMOKE-ACCEPTANCE",
        paper_final_count=0,
    ),
    "research/continuous-control-causal-attribution-20260722": override(
        "ACTIVE",
        "Endpoint-anchored continuous-control causal attribution",
        task_id="AAAI27-CONTINUOUS-CONTROL-CAUSAL-ATTRIBUTION-001",
        source_branch="research/continuous-control-artifact-root-cause-20260722",
        source_head="68623c36eee70c0b41aefb480f8f85e668eae201",
        latest_report="docs/PAPER/AAAI27_CAUSAL_ATTRIBUTION_ALPHA_GRID_CORRECTION_20260722.md",
        latest_summary_json="paper_protocol/reviewer_risk/continuous_control_causal_attribution_protocol_correction.json",
        formal_output_root="/root/autodl-tmp/canondressgs_work/outputs/CONTINUOUS-CONTROL-CAUSAL-ATTRIBUTION-001",
        artifact_fingerprint="aa5cc2bc0f5deb1f9a4dacde26178fb1c322ed2fd314dfca308b00f6b98d21c4",
        paper_final_count=0,
        notes="attempt_001 is preserved as FAILED_PRE_RESULT_ALPHA_GRID_ASSET_MISMATCH; next append-only attempt is attempt_002; no final summary exists locally.",
    ),
    "research/continuous-control-artifact-root-cause-20260722": override(
        "SEALED_HISTORICAL",
        "Previous continuous-control root-cause diagnostic",
        task_id="AAAI27-CONTINUOUS-CONTROL-ROOT-CAUSE-001",
        source_branch="paper/aaai27-p0-color-spatial-soft-control-evaluations-20260722",
        source_head="371d812864614cd561e33edfe3f6c043b38415d2",
        latest_report="docs/PAPER/AAAI27_CONTINUOUS_CONTROL_ARTIFACT_ROOT_CAUSE_20260722.md",
        latest_summary_json="paper_protocol/reviewer_risk/continuous_control_root_cause_final_summary.json",
        formal_output_root="/root/autodl-tmp/canondressgs_work/outputs/CONTINUOUS-CONTROL-ROOT-CAUSE-001",
        artifact_fingerprint="5540c8da3fe2e94c64aef1962f057b06e94d3f75ba845bd8c0ecc6216a536340",
        paper_final_count=0,
        notes="Channel attribution has midpoint degeneracy and cannot be used as the final causal conclusion.",
    ),
    "research/explicit-gaussian-residual-basis-20260720": override(
        "SEALED_CANONICAL",
        "Explicit Gaussian residual basis ladder",
        task_id="SUBJECT02-EXPLICIT-GAUSSIAN-RESIDUAL-BASIS-001",
    ),
    "research/protected-cloud-attribution-20260719": override(
        "UNKNOWN_REQUIRES_REVIEW",
        "Protected Gaussian cloud leakage attribution",
        notes="Dirty worktree; do not infer sealed status.",
    ),
    "research/image-conditioned-failure-diagnosis-20260720": override(
        "FAILED_PRESERVED",
        "O01 image-conditioned decoder capacity failure diagnosis",
        task_id="SUBJECT02-IMAGE-CONDITIONED-FAILURE-DIAGNOSIS",
    ),
    "sprint/aaai27-minimal-image-overfit-20260719": override(
        "UNKNOWN_REQUIRES_REVIEW",
        "Minimal O01 image-overfit sprint",
        notes="Clean worktree, but no current formal classification was found in the scanned handoff set.",
    ),
    "research/multi-outfit-explicit-basis-20260720": override(
        "SEALED_CANONICAL",
        "Five-seen-outfit teacher bank and rank-4 basis",
        task_id="SUBJECT02-MULTI-OUTFIT-EXPLICIT-BASIS-001",
        formal_output_root="/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-MULTI-OUTFIT-EXPLICIT-BASIS-001/attempt_003",
    ),
    "project/canondressgs-control-center-20260722": override(
        "DOCUMENTATION_ACTIVE",
        "Unique CanonDressGS project control center",
        task_id="CANONDRESSGS-PROJECT-CONTROL-001",
        source_branch="paper/aaai27-p0-color-spatial-soft-control-evaluations-20260722",
        source_head="371d812864614cd561e33edfe3f6c043b38415d2",
        notes="The recorded HEAD is the pre-commit scan HEAD; this registry cannot self-record the hash of the commit that contains it.",
    ),
    "research/reference-basis-coefficient-fusion-20260720": override(
        "SEALED_HISTORICAL",
        "Reference-to-basis coefficient fusion diagnosis",
        task_id="SUBJECT02-REFERENCE-BASIS-COEFFICIENT-FUSION-001",
    ),
    "research/reference-coefficient-supervision-calibration-20260720": override(
        "SEALED_CANONICAL",
        "Reference coefficient supervision calibration (CS-PASS)",
        task_id="SUBJECT02-REFERENCE-COEFFICIENT-SUPERVISION-CALIBRATION-001",
    ),
    "research/residual-decoder-capacity-v7-20260720": override(
        "FAILED_PRESERVED",
        "Residual decoder V7 capacity adjudication",
        task_id="SUBJECT02-RESIDUAL-DECODER-CAPACITY-V7-001",
    ),
    "research/residual-field-parameterization-20260720": override(
        "FAILED_PRESERVED",
        "Residual-field parameterization adjudication",
        task_id="SUBJECT02-RESIDUAL-FIELD-PARAMETERIZATION-001",
    ),
}


def run_git(repo: Path, *args: str, check: bool = True, timeout: int = 30) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
    )
    if check and completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "git command failed")
    return completed.stdout.strip() if completed.returncode == 0 else ""


def parse_worktrees(repo: Path) -> list[dict[str, Any]]:
    raw = run_git(repo, "worktree", "list", "--porcelain")
    records: list[dict[str, Any]] = []
    for block in re.split(r"\r?\n\r?\n", raw):
        if not block.strip():
            continue
        fields: dict[str, Any] = {"locked": False, "prunable": False}
        for line in block.splitlines():
            key, _, value = line.partition(" ")
            if key in {"locked", "prunable"}:
                fields[key] = True
                if value:
                    fields[f"{key}_reason"] = value
            elif key == "branch":
                fields[key] = value.removeprefix("refs/heads/")
            elif key == "detached":
                fields["branch"] = "DETACHED"
            else:
                fields[key] = value
        records.append(fields)
    return records


def cached_remote_head(path: Path, remote: str, branch: str) -> str | None:
    if branch == "DETACHED":
        return None
    value = run_git(path, "rev-parse", f"refs/remotes/{remote}/{branch}", check=False)
    return value or None


def ls_remote(repo: Path, remote: str) -> tuple[dict[str, str], str]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "ls-remote", "--heads", remote],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=25,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {}, f"FAILED: {exc}"
    if completed.returncode:
        message = " ".join(completed.stderr.strip().splitlines())
        return {}, f"FAILED: {message[:300]}"
    refs = {}
    for line in completed.stdout.splitlines():
        head, ref = line.split(maxsplit=1)
        refs[ref.removeprefix("refs/heads/")] = head
    return refs, "PASS"


def rel_if_present(worktree: Path, value: str | None) -> str | None:
    if not value:
        return None
    candidate = worktree / Path(value)
    return str(candidate.resolve()) if candidate.exists() else str(candidate.resolve())


def scan_non_git(scan_root: Path, worktree_paths: set[str], main_repo: Path) -> list[dict[str, str]]:
    pattern = re.compile(r"^(?:canondressgs_|aaai27_|alpha_)", re.IGNORECASE)
    candidates = [p for p in scan_root.iterdir() if p.is_dir() and pattern.match(p.name)]
    if main_repo.parent == scan_root and main_repo not in candidates:
        candidates.append(main_repo)
    result = []
    for candidate in sorted(candidates, key=lambda p: str(p).casefold()):
        key = os.path.normcase(os.path.abspath(candidate))
        if key not in worktree_paths:
            top = run_git(candidate, "rev-parse", "--show-toplevel", check=False)
            result.append(
                {
                    "local_path": str(candidate.resolve()),
                    "classification": "GIT_NOT_REGISTERED_WORKTREE" if top else "NON_GIT_DIRECTORY",
                    "git_top_level": top or None,
                }
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--scan-root", type=Path, default=Path(r"E:\model_train"))
    parser.add_argument("--verify-remotes", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    output = (args.output or repo / "project_control" / "WORKTREE_REGISTRY.yaml").resolve()
    common_dir_raw = run_git(repo, "rev-parse", "--git-common-dir")
    common_dir = (repo / common_dir_raw).resolve() if not Path(common_dir_raw).is_absolute() else Path(common_dir_raw).resolve()
    remotes_text = run_git(repo, "remote", "-v", check=False)
    remote_urls: dict[str, dict[str, str]] = {}
    for line in remotes_text.splitlines():
        match = re.match(r"(\S+)\s+(\S+)\s+\((fetch|push)\)", line)
        if match:
            remote_urls.setdefault(match.group(1), {})[match.group(3)] = match.group(2)

    origin_refs: dict[str, str] = {}
    cloud_refs: dict[str, str] = {}
    origin_check = "NOT_REQUESTED"
    cloud_check = "NOT_REQUESTED"
    if args.verify_remotes:
        origin_refs, origin_check = ls_remote(repo, "origin")
        cloud_refs, cloud_check = ls_remote(repo, "cloud")

    rows = []
    for index, raw in enumerate(parse_worktrees(repo), start=1):
        path = Path(raw["worktree"]).resolve()
        branch = raw.get("branch", "DETACHED")
        head = raw["HEAD"]
        status_lines = run_git(path, "status", "--porcelain", check=False).splitlines()
        upstream = run_git(path, "rev-parse", "--abbrev-ref", "@{upstream}", check=False) or None
        local_common_raw = run_git(path, "rev-parse", "--git-common-dir")
        local_common = (path / local_common_raw).resolve() if not Path(local_common_raw).is_absolute() else Path(local_common_raw).resolve()
        cached_origin = cached_remote_head(path, "origin", branch)
        cached_cloud = cached_remote_head(path, "cloud", branch)
        origin_head = origin_refs.get(branch) if origin_check == "PASS" else cached_origin
        cloud_head = cloud_refs.get(branch) if cloud_check == "PASS" else cached_cloud
        metadata = OVERRIDES.get(
            branch,
            override("UNKNOWN_REQUIRES_REVIEW", "Unclassified worktree; formal owner review required"),
        ).copy()
        notes = metadata.pop("notes", None)
        push_matches = (origin_head == head) or (cloud_head == head)
        if push_matches:
            push_status = "PASS"
        elif origin_head or cloud_head:
            push_status = "HEAD_MISMATCH"
        else:
            push_status = "UNVERIFIED_OR_NOT_PUSHED"
        if cloud_check == "PASS":
            cloud_sync = "PASS" if cloud_head == head else ("HEAD_MISMATCH" if cloud_head else "BRANCH_ABSENT")
        elif cached_cloud:
            cloud_sync = "CACHED_HEAD_MATCH_LIVE_UNREACHABLE" if cached_cloud == head else "CACHED_HEAD_MISMATCH_LIVE_UNREACHABLE"
        else:
            cloud_sync = "UNVERIFIED_CLOUD_REMOTE_UNREACHABLE" if cloud_check.startswith("FAILED") else "NOT_VERIFIED"
        try:
            created_date = dt.datetime.fromtimestamp(path.stat().st_ctime).date().isoformat()
        except OSError:
            created_date = None
        latest_report = metadata.pop("latest_report", None)
        latest_summary = metadata.pop("latest_summary_json", None)
        row = {
            "id": f"WT-{index:03d}",
            "local_path": str(path),
            "cloud_path": metadata.pop("cloud_path", None),
            "repository_common_dir": str(local_common),
            "branch": branch,
            "head": head,
            "upstream": upstream,
            "origin_head": origin_head,
            "cloud_head": cloud_head,
            "clean_local": not status_lines,
            "clean_cloud": None,
            "source_branch": metadata.pop("source_branch", None),
            "source_head": metadata.pop("source_head", None),
            "purpose": metadata.pop("purpose"),
            "task_id": metadata.pop("task_id", None),
            "created_date": created_date,
            "latest_report": rel_if_present(path, latest_report),
            "latest_summary_json": rel_if_present(path, latest_summary),
            "formal_output_root": metadata.pop("formal_output_root", None),
            "artifact_fingerprint": metadata.pop("artifact_fingerprint", None),
            "push_status": push_status,
            "cloud_sync_status": cloud_sync,
            "paper_final_count": metadata.pop("paper_final_count", None),
            "status_class": metadata.pop("status_class"),
            "superseded_by": metadata.pop("superseded_by", None),
            "safe_to_remove": False,
            "locked": bool(raw.get("locked")),
            "expected_head": EXPECTED_HEADS.get(branch),
            "expected_head_match": EXPECTED_HEADS.get(branch) in {None, head},
            "notes": notes,
        }
        if metadata:
            row["additional_metadata"] = metadata
        rows.append(row)

    worktree_paths = {os.path.normcase(os.path.abspath(row["local_path"])) for row in rows}
    non_git = scan_non_git(args.scan_root.resolve(), worktree_paths, repo)
    counts = {name: sum(row["status_class"] == name for row in rows) for name in sorted(STATUS_CLASSES)}
    registry = {
        "schema_version": "canondressgs.project_control.worktree_registry.v1",
        "task_id": "CANONDRESSGS-PROJECT-CONTROL-001",
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "scan": {
            "scan_root": str(args.scan_root.resolve()),
            "repository_common_dir": str(common_dir),
            "discovered_git_worktree_count": len(rows),
            "non_git_directory_count": len(non_git),
            "remote_verification": {"origin": origin_check, "cloud": cloud_check},
            "remote_urls": remote_urls,
            "status_counts": counts,
            "safe_to_remove_count": 0,
        },
        "worktrees": rows,
        "non_git_directories": non_git,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    print(f"Git worktrees: {len(rows)}; non-Git directories: {len(non_git)}")
    print(f"Origin: {origin_check}")
    print(f"Cloud: {cloud_check}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
