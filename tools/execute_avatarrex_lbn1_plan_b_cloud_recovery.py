#!/usr/bin/env python3
"""Run the recovered-cloud Plan B extraction without a loader canary."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import execute_avatarrex_lbn1_plan_b as base


TASK_ID = "AAAI27-AVATARREX-LBN1-PLAN-B-CLOUD-ACCESS-RECOVERY-AND-EXECUTION-001"
SOURCE_BRANCH = "research/avatarrex-lbn1-targeted-extraction-plan-b-20260726"
SOURCE_HEAD = "95d84853ac1ba8d9928613f66310d23b6fd08c0c"
NEW_BRANCH = "research/avatarrex-lbn1-plan-b-cloud-recovery-execution-20260726"
WINDOWS_WORKTREE = r"E:\model_train\canondressgs_avatarrex_lbn1_plan_b_cloud_recovery_execution"
CLOUD_WORKTREE = (
    "/root/autodl-tmp/canondressgs_work/worktrees/"
    "canondressgs_avatarrex_lbn1_plan_b_cloud_recovery_execution"
)
PROJECT_ANCHOR = Path("/root/autodl-tmp/canondressgs_work")
MIN_PROJECTED_FREE_BYTES = 42_949_672_960
FINAL_CLASSIFICATION = "AVATARREX_LBN1_PLAN_B_TARGETED_EXTRACTION_PASS_PENDING_LOADER_REVIEW"
NEXT_TASK = "USER_REVIEW_AVATARREX_PLAN_B_EXTRACTION_AND_AUTHORIZE_LOADER_CANARY"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=base.PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def assert_execution_gate() -> tuple[list[base.Member], dict[str, Any]]:
    branch = git("branch", "--show-current")
    status = git("status", "--short")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", SOURCE_HEAD, "HEAD"],
        cwd=base.PROJECT_ROOT,
    ).returncode
    if branch != NEW_BRANCH or status or ancestor != 0:
        raise base.PlanBFailure(
            f"Cloud Git gate mismatch: branch={branch!r} status={status!r} ancestor={ancestor}",
            base.FINAL_CLASSIFICATION_CONTRACT_FAIL,
            "CLOUD_GIT_GATE_FAILED",
        )
    hostname = platform.node()
    if not hostname.startswith("autodl-container-") or not PROJECT_ANCHOR.is_dir():
        raise base.PlanBFailure(
            f"Cloud identity gate failed: hostname={hostname!r}",
            base.FINAL_CLASSIFICATION_CONTRACT_FAIL,
            "CLOUD_IDENTITY_GATE_FAILED",
        )

    staging_root = Path(str(base.STAGING_ROOT))
    audit_root = Path(str(base.AUDIT_ROOT))
    base.ensure_execute_roots_absent(staging_root, audit_root)
    members = base.load_members()

    df = base.run_command(["df", "-B1", str(base.DF_TARGET)])
    if df.returncode:
        raise base.PlanBFailure(
            "df -B1 failed before attempt creation.",
            base.FINAL_CLASSIFICATION_STORAGE_FAIL,
            "DF_COMMAND_FAILED",
        )
    free_bytes = base.parse_df_free_bytes(df.stdout)
    projected = free_bytes - base.EXPECTED_UNCOMPRESSED_BYTES - base.PROJECT_AUDIT_RESERVE_BYTES
    if projected < MIN_PROJECTED_FREE_BYTES:
        raise base.PlanBFailure(
            f"Recovered-cloud storage gate failed: projected={projected}",
            base.FINAL_CLASSIFICATION_STORAGE_FAIL,
            "RECOVERED_CLOUD_STORAGE_GATE_FAILED",
        )
    return members, {
        "cloud_free_bytes_before": free_bytes,
        "projected_free_bytes_after": projected,
        "required_projected_free_bytes": MIN_PROJECTED_FREE_BYTES,
        "status": "PASS",
    }


def make_summary(
    classification: str,
    storage_gate: dict[str, Any],
    preflight: dict[str, Any],
    extraction: dict[str, Any],
    verification: dict[str, Any],
    failure: base.PlanBFailure | None,
) -> dict[str, Any]:
    return {
        "schema_version": "avatarrex_lbn1.plan_b.cloud_recovery_execution.v1",
        "task_id": TASK_ID,
        "previous_task_id": base.TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "new_branch": NEW_BRANCH,
        "windows_worktree": WINDOWS_WORKTREE,
        "cloud_connection_method": "WINDOWS_OPENSSH_EXISTING_ALIAS",
        "cloud_host": platform.node(),
        "cloud_worktree": CLOUD_WORKTREE,
        "archive_path": str(base.ARCHIVE_PATH),
        "archive_bytes_expected": base.ARCHIVE_BYTES,
        "archive_sha256_expected": base.ARCHIVE_SHA256,
        "plan_id": base.PLAN_ID,
        "allowlist_path": str(base.ALLOWLIST_TXT.relative_to(base.PROJECT_ROOT)),
        "allowlist_count": base.ALLOWLIST_COUNT,
        "camera_ids": list(base.CAMERA_IDS),
        "frame_ids": list(base.FRAME_IDS),
        "expected_uncompressed_bytes": base.EXPECTED_UNCOMPRESSED_BYTES,
        "staging_root": str(base.STAGING_ROOT),
        "audit_root": str(base.AUDIT_ROOT),
        "storage_gate": storage_gate,
        "preflight": preflight,
        "extraction": extraction,
        "verification": verification,
        "archive_mutations": 0,
        "formal_data_root_status": "NONE",
        "preprocessing_steps": 0,
        "template_generation": 0,
        "lbs_grid_generation": 0,
        "base_avatar_training_steps": 0,
        "training_steps": 0,
        "generation_calls": 0,
        "garment_generation_calls": 0,
        "paper_modifications": 0,
        "paper_final": False,
        "loader_canary_calls": 0,
        "final_classification": classification,
        "next_task": NEXT_TASK if failure is None else "STOP_AFTER_SINGLE_EXTRACTION_ATTEMPT_FAILURE",
        "failure": None
        if failure is None
        else {
            "classification": failure.classification,
            "kind": failure.failure_kind,
            "message": str(failure),
        },
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    verification = summary["verification"]
    lines = [
        "# AvatarReX LBN1 Plan B Cloud-Recovery Execution",
        "",
        f"- TASK_ID: `{summary['task_id']}`",
        f"- FINAL_CLASSIFICATION: `{summary['final_classification']}`",
        f"- CLOUD_HOST: `{summary['cloud_host']}`",
        f"- ARCHIVE_PATH: `{summary['archive_path']}`",
        f"- STAGING_ROOT: `{summary['staging_root']}`",
        f"- EXTRACTION_CALLS: `{summary['extraction'].get('extraction_calls', 0)}`",
        f"- EXTRACTED_FILE_COUNT: `{verification.get('extracted_file_count', 0)}`",
        f"- EXTRACTED_TOTAL_BYTES: `{verification.get('extracted_total_bytes', 0)}`",
        f"- LOADER_CANARY_CALLS: `{summary['loader_canary_calls']}`",
        f"- NEXT_TASK: `{summary['next_task']}`",
        "",
        "The frozen allowlist is the only extraction scope. No wildcard, full-sequence extraction, preprocessing, formal-data promotion, loader canary, training, generation, or paper modification is performed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    audit_root = Path(str(base.AUDIT_ROOT))
    storage_gate: dict[str, Any] = {}
    preflight: dict[str, Any] = {}
    extraction: dict[str, Any] = {"extraction_calls": 0}
    verification: dict[str, Any] = {}
    failure: base.PlanBFailure | None = None
    classification = FINAL_CLASSIFICATION

    try:
        members, storage_gate = assert_execution_gate()
        audit_root.mkdir(parents=True, exist_ok=False)
        hashes = base.copy_attempt_manifests(audit_root)
        base.write_json(audit_root / "manifest_copy_hashes.json", hashes)
        preflight = base.preflight(Path(str(base.ARCHIVE_PATH)), audit_root)
        listfile = base.write_7z_listfile(members, audit_root)
        extraction = base.run_extraction(
            Path(str(base.ARCHIVE_PATH)),
            Path(str(base.STAGING_ROOT)),
            audit_root,
            listfile,
        )
        verification = base.verify_extraction(
            Path(str(base.STAGING_ROOT)),
            members,
            audit_root,
            run_loader_check=False,
        )
        sha_after = base.run_command(
            ["sha256sum", str(base.ARCHIVE_PATH)],
            audit_root / "preflight_logs" / "sha256sum_after.json",
        )
        if sha_after.returncode or sha_after.stdout.split()[0].lower() != base.ARCHIVE_SHA256:
            raise base.PlanBFailure(
                "Archive SHA changed after targeted extraction.",
                base.FINAL_CLASSIFICATION_ARCHIVE_MUTATION,
                "ARCHIVE_SHA_AFTER_MISMATCH",
            )
    except base.PlanBFailure as exc:
        failure = exc
        classification = exc.classification
    except Exception as exc:
        failure = base.PlanBFailure(
            f"Unhandled execution failure: {type(exc).__name__}: {exc}",
            base.FINAL_CLASSIFICATION_CONTRACT_FAIL,
            "UNHANDLED_EXECUTION_FAILURE",
        )
        classification = failure.classification

    summary = make_summary(
        classification,
        storage_gate,
        preflight,
        extraction,
        verification,
        failure,
    )
    if audit_root.exists():
        base.write_json(audit_root / "avatarrex_lbn1_plan_b_cloud_recovery_final_summary.json", summary)
        write_report(
            audit_root / "AVATARREX_LBN1_PLAN_B_CLOUD_RECOVERY_EXECUTION_REPORT.md",
            summary,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if failure is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
