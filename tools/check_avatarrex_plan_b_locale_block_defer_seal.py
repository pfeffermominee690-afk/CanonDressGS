#!/usr/bin/env python3
"""Audit the additive AvatarReX Plan B locale-block defer seal."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "paper_protocol" / "avatarrex_lbn1_plan_b"
EVIDENCE_PATH = PROTOCOL / "avatarrex_lbn1_plan_b_cloud_recovery_evidence_20260726.json"
PRIOR_SUMMARY_PATH = PROTOCOL / "avatarrex_lbn1_plan_b_cloud_recovery_final_summary_20260726.json"
PRIOR_REPORT_PATH = PROTOCOL / "AVATARREX_LBN1_PLAN_B_CLOUD_RECOVERY_FINAL_REPORT_20260726.md"
LISTING_OVERLAY_PATH = PROTOCOL / "avatarrex_plan_b_listing_fingerprint_failure_overlay_20260726.json"
CORRECTION_PATH = PROTOCOL / "avatarrex_plan_b_locale_fingerprint_classification_correction_20260726.json"
SUMMARY_PATH = PROTOCOL / "avatarrex_plan_b_locale_block_defer_final_summary_20260726.json"
REPORT_PATH = PROTOCOL / "AVATARREX_PLAN_B_LOCALE_FINGERPRINT_BLOCK_AND_DEFER_REPORT_20260726.md"
TESTS_PATH = PROTOCOL / "avatarrex_plan_b_locale_block_defer_tests_20260726.json"
HANDOFF_PATH = ROOT / "project_control_handoff" / "avatarrex_plan_b_locale_block_defer_seal_handoff_20260726.json"
ALLOWLIST_PATH = PROTOCOL / "manifests" / "avatarrex_lbn1_plan_b_allowlist.json"
SOURCE_WORKTREE = Path(r"E:\model_train\canondressgs_avatarrex_lbn1_plan_b_cloud_recovery_execution")

TASK_ID = "AAAI27-AVATARREX-PLAN-B-LOCALE-BLOCK-DEFER-SEAL-001"
SOURCE_BRANCH = "research/avatarrex-lbn1-plan-b-cloud-recovery-execution-20260726"
SOURCE_HEAD = "914efe09f286d3cd16a5d5f283c7a2e74c65ea54"
BRANCH = "research/avatarrex-plan-b-locale-block-defer-seal-20260726"
ORIGINAL = "AVATARREX_PLAN_B_EXTRACTION_CONTRACT_FAIL"
CORRECTED = "AVATARREX_PLAN_B_PREFLIGHT_BLOCKED_BY_LOCALE_SENSITIVE_LISTING_FINGERPRINT"
FINAL = "AVATARREX_PLAN_B_LOCALE_BLOCK_SEALED_DEFERRED_FOR_SUBJECT00_FORMAL_BASE"
NEXT = "RESUME_AVATARREX_PLAN_B_AFTER_SUBJECT00_AND_REUPLOAD_VERIFIED_ARCHIVE"
ARCHIVE_SHA = "531bd1c71ad9b35f6ae0e2595ee531aa7ba1f83c242f18f2d0505b2dcd5fbcc1"
ALLOWED_CHANGED_PATHS = {
    "paper_protocol/avatarrex_lbn1_plan_b/AVATARREX_PLAN_B_LOCALE_FINGERPRINT_BLOCK_AND_DEFER_REPORT_20260726.md",
    "paper_protocol/avatarrex_lbn1_plan_b/avatarrex_plan_b_locale_block_defer_final_summary_20260726.json",
    "paper_protocol/avatarrex_lbn1_plan_b/avatarrex_plan_b_locale_block_defer_tests_20260726.json",
    "paper_protocol/avatarrex_lbn1_plan_b/avatarrex_plan_b_locale_fingerprint_classification_correction_20260726.json",
    "project_control_handoff/avatarrex_plan_b_locale_block_defer_seal_handoff_20260726.json",
    "tools/check_avatarrex_plan_b_locale_block_defer_seal.py",
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_sha(value: dict[str, Any]) -> str:
    clone = dict(value)
    clone.pop("content_sha256", None)
    raw = json.dumps(clone, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def changed_paths() -> set[str]:
    tracked = git("diff", "--name-only", SOURCE_HEAD).splitlines()
    untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
    return set(filter(None, [*tracked, *untracked]))


def main() -> int:
    evidence = load(EVIDENCE_PATH)
    prior_summary = load(PRIOR_SUMMARY_PATH)
    listing = load(LISTING_OVERLAY_PATH)
    correction = load(CORRECTION_PATH)
    summary = load(SUMMARY_PATH)
    handoff = load(HANDOFF_PATH)
    allowlist = load(ALLOWLIST_PATH)
    report = REPORT_PATH.read_text(encoding="utf-8")
    prior_report = PRIOR_REPORT_PATH.read_text(encoding="utf-8")
    checks: list[dict[str, Any]] = []

    def check(name: str, predicate: Callable[[], bool], detail: str) -> None:
        try:
            passed = bool(predicate())
        except Exception as exc:
            passed = False
            detail = f"{type(exc).__name__}: {exc}"
        checks.append({"detail": detail, "name": name, "passed": passed})

    check("source_branch_head", lambda: correction["source_branch"] == summary["source_branch"] == handoff["source_branch"] == SOURCE_BRANCH and correction["source_head"] == summary["source_head"] == handoff["source_head"] == SOURCE_HEAD, "source branch and HEAD are exact")
    check("source_worktree_clean", lambda: SOURCE_WORKTREE.is_dir() and git("status", "--porcelain", cwd=SOURCE_WORKTREE) == "", "source worktree is clean")
    check("target_branch", lambda: git("branch", "--show-current") == BRANCH, "new branch is exact")
    check("source_ancestor", lambda: subprocess.run(["git", "merge-base", "--is-ancestor", SOURCE_HEAD, "HEAD"], cwd=ROOT).returncode == 0, "new branch descends from source HEAD")
    check("task_id", lambda: correction["task_id"] == summary["task_id"] == handoff["task_id"] == TASK_ID, "task IDs match")
    check("archive_facts_unchanged", lambda: evidence["archive"]["bytes"] == summary["archive_bytes"] == 12_569_755_256 and evidence["archive"]["sha256_before"] == evidence["archive"]["sha256_after_failure"] == summary["archive_sha256"] == ARCHIVE_SHA, "archive bytes and SHA inherit unchanged evidence")
    check("archive_integrity", lambda: evidence["archive"]["test_return_code"] == 0 and summary["archive_integrity_status"] == evidence["archive"]["test_status"] == "PASS_7ZIP_16_02_EVERYTHING_IS_OK", "prior 7z integrity result is preserved")
    check("allowlist_1602", lambda: evidence["allowlist"]["count"] == summary["allowlist_count"] == len(allowlist["members"]) == 1602, "allowlist remains exactly 1602 members")
    check("extraction_calls_zero", lambda: evidence["execution"]["extraction_calls"] == summary["extraction_calls"] == handoff["extraction_calls"] == 0, "extraction calls remain zero")
    check("staging_not_created", lambda: evidence["execution"]["staging_root_exists_after"] is False and summary["staging_created"] is False and handoff["staging_created"] is False, "attempt_001 staging was not created")
    check("locale_mismatch_exact", lambda: correction["observed_environment"] == {"locale": "C.UTF-8", "utf16": "on"} and correction["frozen_environment"] == {"locale": "C", "utf16": "off"} and summary["locale_observed"] == "C.UTF-8" and summary["utf16_observed"] == "on", "locale and Utf16 mismatch is exact")
    check("banner_only_difference", lambda: listing["archive_member_difference_count"] == correction["archive_member_difference_count"] == 0 and summary["fingerprint_difference_class"] == "TOOL_ENVIRONMENT_BANNER_ONLY_NO_ARCHIVE_MEMBER_DIFFERENCE", "difference is banner-only with no member change")
    check("original_classification_preserved", lambda: prior_summary["final_classification"] == correction["original_classification"] == summary["original_classification"] == ORIGINAL and ORIGINAL in prior_report and git("diff", "--name-only", SOURCE_HEAD, "--", str(PRIOR_SUMMARY_PATH.relative_to(ROOT)).replace("\\", "/"), str(PRIOR_REPORT_PATH.relative_to(ROOT)).replace("\\", "/")) == "", "original report and classification are unchanged")
    check("correction_overlay", lambda: CORRECTION_PATH.is_file() and correction["corrected_classification"] == summary["corrected_classification"] == CORRECTED and correction["previous_report_modified"] is False, "additive correction overlay exists")
    check("attempt_002_not_authorized", lambda: summary["attempt_002_authorized"] is False and handoff["attempt_002_authorized"] is False, "attempt_002 is not authorized")
    check("deferred_status", lambda: summary["avatarrex_plan_b_execution_priority"] == "DEFERRED_FOR_SUBJECT00_FORMAL_BASE" and summary["avatarrex_plan_b_status"] == handoff["avatarrex_plan_b_status"] == "DEFERRED_PENDING_REUPLOAD_FROM_VERIFIED_WINDOWS_MIRROR", "Plan B is deferred for Subject00")
    check("windows_mirror_preserved", lambda: summary["windows_mirror_status"] == handoff["windows_mirror_status"] == "PRESERVED_BYTE_EXACT" and Path(summary["windows_mirror"]["path"]).stat().st_size == 12_569_755_256 and file_sha256(Path(summary["windows_mirror"]["path"])) == ARCHIVE_SHA, "Windows mirror exists and is byte-exact")
    check("no_archive_deletion", lambda: summary["cloud_archive_deletion_executed_by_this_task"] is False and handoff["cloud_archive_deletion_executed_by_this_task"] is False and summary["cloud_archive_status"] == "PRESENT_AT_LAST_OBSERVATION_PENDING_SEPARATE_AUTHORIZED_RECLAMATION", "this task did not delete the cloud archive")
    check("no_extraction", lambda: summary["extraction_calls"] == 0 and summary["staging_created"] is False, "no extraction or staging creation")
    check("no_preprocessing", lambda: summary["preprocessing_steps"] == handoff["preprocessing_steps"] == 0, "no preprocessing")
    check("no_training", lambda: summary["training_steps"] == handoff["training_steps"] == 0, "no training or GPU work")
    check("no_generation", lambda: summary["generation_calls"] == handoff["generation_calls"] == 0, "no generation")
    check("no_paper_modification", lambda: summary["paper_modifications"] == handoff["paper_modifications"] == 0 and summary["paper_final"] is False and handoff["paper_final"] is False, "no paper modification")
    check("changed_path_scope", lambda: changed_paths() <= ALLOWED_CHANGED_PATHS, "only seal audit paths changed")
    check("future_policy_design_only", lambda: summary["future_fingerprint_policy"] == handoff["future_fingerprint_policy"] == "LOCALE_INSENSITIVE_ARCHIVE_CONTENT_FINGERPRINT" and "does not implement it" in report, "future policy is recorded but not implemented")
    check("final_classification", lambda: summary["final_classification"] == handoff["final_classification"] == FINAL and FINAL in report, "final classification is exact")
    check("next_task_unique", lambda: summary["next_task"] == handoff["next_task"] == NEXT and report.count(NEXT) == 1, "one exact next task is recorded")

    failures = [item["name"] for item in checks if not item["passed"]]
    payload: dict[str, Any] = {
        "check_count": len(checks),
        "checks": checks,
        "failed_checks": failures,
        "paper_final": False,
        "result": "PASS_LOCALE_BLOCK_DEFER_SEAL" if not failures else "FAIL",
        "schema_version": "avatarrex_lbn1.plan_b.locale_block_defer_tests.v1",
        "task_id": TASK_ID,
    }
    payload["content_sha256"] = canonical_sha(payload)
    TESTS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{payload['result']}: {len(checks)} checks")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
