#!/usr/bin/env python3
"""Audit the safe-stop result of the recovered-cloud Plan B attempt."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "paper_protocol" / "avatarrex_lbn1_plan_b"
EVIDENCE_PATH = PROTOCOL / "avatarrex_lbn1_plan_b_cloud_recovery_evidence_20260726.json"
SUMMARY_PATH = PROTOCOL / "avatarrex_lbn1_plan_b_cloud_recovery_final_summary_20260726.json"
TESTS_PATH = PROTOCOL / "avatarrex_lbn1_plan_b_cloud_recovery_tests_20260726.json"
HANDOFF_PATH = ROOT / "project_control_handoff" / "avatarrex_lbn1_plan_b_cloud_recovery_execution_handoff_20260726.json"
LISTING_OVERLAY = PROTOCOL / "avatarrex_plan_b_listing_fingerprint_failure_overlay_20260726.json"
PREVIOUS_OVERLAY = PROTOCOL / "avatarrex_plan_b_previous_classification_correction_overlay_20260726.json"
ALLOWLIST_PATH = PROTOCOL / "manifests" / "avatarrex_lbn1_plan_b_allowlist.json"

TASK_ID = "AAAI27-AVATARREX-LBN1-PLAN-B-CLOUD-ACCESS-RECOVERY-AND-EXECUTION-001"
SOURCE_HEAD = "95d84853ac1ba8d9928613f66310d23b6fd08c0c"
BRANCH = "research/avatarrex-lbn1-plan-b-cloud-recovery-execution-20260726"
FINAL_CLASSIFICATION = "AVATARREX_PLAN_B_EXTRACTION_CONTRACT_FAIL"
NEXT_TASK = "USER_AUTHORIZE_AVATARREX_PLAN_B_ATTEMPT_002_AFTER_LOCALE_STABLE_LISTING_FINGERPRINT_REPAIR"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_sha(value: dict[str, Any]) -> str:
    clone = dict(value)
    clone.pop("content_sha256", None)
    raw = json.dumps(clone, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def main() -> int:
    evidence = load(EVIDENCE_PATH)
    summary = load(SUMMARY_PATH)
    handoff = load(HANDOFF_PATH)
    listing = load(LISTING_OVERLAY)
    previous = load(PREVIOUS_OVERLAY)
    allowlist = load(ALLOWLIST_PATH)
    members = allowlist["members"]
    checks: list[dict[str, Any]] = []

    def check(name: str, predicate: Callable[[], bool], detail: str) -> None:
        try:
            passed = bool(predicate())
        except Exception as exc:
            passed = False
            detail = f"{type(exc).__name__}: {exc}"
        checks.append({"detail": detail, "name": name, "passed": passed})

    check("task_id", lambda: evidence["task_id"] == summary["task_id"] == handoff["task_id"] == TASK_ID, "task IDs match")
    check("source_head", lambda: evidence["source_head"] == summary["source_head"] == handoff["source_head"] == SOURCE_HEAD, "source HEAD matches")
    check("target_branch", lambda: git("branch", "--show-current") == BRANCH, "target branch matches")
    check("source_ancestor", lambda: subprocess.run(["git", "merge-base", "--is-ancestor", SOURCE_HEAD, "HEAD"], cwd=ROOT).returncode == 0, "target descends from source")
    check("cloud_identity", lambda: evidence["cloud"]["identity_status"] == "PASS_CANONDRESSGS_FORMAL_CLOUD", "cloud identity passed")
    check("cloud_container", lambda: evidence["cloud"]["container"] == "autodl-container-ef19489c10-464381bb", "container is fixed")
    check("project_anchor", lambda: evidence["cloud"]["project_anchor"] == "/root/autodl-tmp/canondressgs_work", "project anchor matches")
    check("formal_asset_anchors", lambda: evidence["cloud"]["formal_asset_anchors_present"] is True, "formal asset anchors were present")
    check("archive_exists", lambda: evidence["archive"]["exists"] is True, "archive existed")
    check("archive_bytes", lambda: evidence["archive"]["bytes"] == 12_569_755_256, "archive bytes match")
    check("archive_sha_before", lambda: evidence["archive"]["sha256_before"] == "531bd1c71ad9b35f6ae0e2595ee531aa7ba1f83c242f18f2d0505b2dcd5fbcc1", "archive pre-SHA matches")
    check("archive_sha_after", lambda: evidence["archive"]["sha256_after_failure"] == evidence["archive"]["sha256_before"], "archive post-SHA is unchanged")
    check("archive_7z_test", lambda: evidence["archive"]["test_return_code"] == 0 and evidence["archive"]["test_status"] == "PASS_7ZIP_16_02_EVERYTHING_IS_OK", "7z test passed")
    check("allowlist_count", lambda: evidence["allowlist"]["count"] == len(members) == 1602, "allowlist has 1602 members")
    check("allowlist_unique", lambda: evidence["allowlist"]["unique_path_count"] == len({item["path"] for item in members}) == 1602, "allowlist paths are unique")
    check("allowlist_bytes", lambda: evidence["allowlist"]["expected_uncompressed_bytes"] == sum(int(item["bytes"]) for item in members) == 501_198_133, "allowlist bytes match")
    check("rgb_count_contract", lambda: evidence["allowlist"]["rgb_count"] == 800, "RGB contract count is 800")
    check("pha_count_contract", lambda: evidence["allowlist"]["pha_count"] == 800, "PHA contract count is 800")
    check("metadata_count_contract", lambda: evidence["allowlist"]["metadata_count"] == 2, "metadata contract count is 2")
    check("camera_set", lambda: set(evidence["allowlist"]["camera_ids"]) == {"22053908", "22053926", "22010708", "22010710", "22010716", "22010714", "22070935", "22053923"}, "camera set is exact")
    check("frame_count", lambda: evidence["allowlist"]["frame_count"] == 100, "frame count is 100")
    check("storage_free", lambda: evidence["storage"]["cloud_free_bytes_before"] == 45_987_901_440, "live free bytes recorded")
    check("storage_projection", lambda: evidence["storage"]["projected_free_bytes_after"] == 43_339_219_659, "projected free bytes recorded")
    check("storage_gate", lambda: evidence["storage"]["status"] == "PASS" and evidence["storage"]["projected_free_bytes_after"] >= evidence["storage"]["required_projected_free_bytes"], "storage gate passed")
    check("staging_preexisting", lambda: evidence["execution"]["staging_preexisted"] is False, "staging did not preexist")
    check("audit_preexisting", lambda: evidence["execution"]["audit_preexisted"] is False, "audit did not preexist")
    check("audit_preserved", lambda: evidence["execution"]["audit_root_exists_after"] is True and listing["audit_root_preserved"] is True, "partial audit is preserved")
    check("staging_not_created", lambda: evidence["execution"]["staging_root_exists_after"] is False and summary["staging_root_created"] is False, "staging was not created")
    check("extraction_calls_zero", lambda: evidence["execution"]["extraction_calls"] == summary["extraction_calls"] == handoff["extraction_calls"] == 0, "no extraction call occurred")
    check("retry_zero", lambda: evidence["execution"]["retry_count"] == summary["retry_count"] == 0 and listing["retry_performed"] is False, "no retry occurred")
    check("extracted_count_zero", lambda: evidence["execution"]["extracted_file_count"] == summary["extracted_file_count"] == 0, "no files were extracted")
    check("extracted_bytes_zero", lambda: evidence["execution"]["extracted_total_bytes"] == summary["extracted_total_bytes"] == 0, "no bytes were extracted")
    check("result_counts_zero", lambda: evidence["execution"]["rgb_count"] == evidence["execution"]["pha_count"] == evidence["execution"]["metadata_count"] == 0, "result asset counts are zero")
    check("failure_kind", lambda: evidence["execution"]["failure_kind"] == summary["failure_kind"] == handoff["failure_kind"] == "ARCHIVE_LISTING_FINGERPRINT_MISMATCH", "failure kind matches")
    check("listing_frozen_sha", lambda: listing["frozen_listing_sha256"] == "d33cb311687a1034c37e795d26966f1a93b16bc5e7de3e96cce28b387081fc70", "frozen listing SHA matches")
    check("listing_captured_sha", lambda: listing["captured_listing_sha256"] == "1c07db32d62550e6a99ddbd407bbc6d53e4290bef336638dd0b9474a731f26f7", "captured listing SHA matches")
    check("listing_members_unchanged", lambda: listing["archive_member_difference_count"] == 0, "archive member set was unchanged")
    check("archive_mutations_zero", lambda: evidence["archive"]["mutations"] == summary["archive_mutations"] == handoff["archive_mutations"] == 0, "archive mutations are zero")
    check("formal_root_none", lambda: evidence["forbidden_work"]["formal_data_root_status"] == summary["formal_data_root_status"] == handoff["formal_data_root_status"] == "NONE", "formal root remains absent")
    check("preprocessing_zero", lambda: evidence["forbidden_work"]["preprocessing_steps"] == summary["preprocessing_steps"] == handoff["preprocessing_steps"] == 0, "preprocessing is zero")
    check("training_zero", lambda: evidence["forbidden_work"]["training_steps"] == summary["training_steps"] == handoff["training_steps"] == 0, "training is zero")
    check("generation_zero", lambda: evidence["forbidden_work"]["generation_calls"] == summary["generation_calls"] == handoff["generation_calls"] == 0, "generation is zero")
    check("loader_zero", lambda: evidence["forbidden_work"]["loader_canary_calls"] == 0, "loader canary is zero")
    check("paper_zero", lambda: evidence["forbidden_work"]["paper_modifications"] == summary["paper_modifications"] == handoff["paper_modifications"] == 0, "paper modifications are zero")
    check("paper_final_false", lambda: evidence["forbidden_work"]["paper_final"] is False and summary["paper_final"] is False and handoff["paper_final"] is False, "paper final remains false")
    check("previous_correction", lambda: previous["corrected_interpretation"] == "PREFLIGHT_BLOCKED_BY_CLOUD_SOURCE_UNREACHABLE" and previous["previous_report_modified"] is False, "previous classification correction is present")
    check("final_classification", lambda: evidence["execution"]["failure_classification"] == summary["final_classification"] == handoff["final_classification"] == FINAL_CLASSIFICATION, "final classification matches safe stop")
    check("next_task", lambda: summary["next_task"] == handoff["next_task"] == NEXT_TASK, "next task uniquely requires user authorization")
    check("no_inventory", lambda: evidence["external_audit"]["inventory_path"] == summary["inventory_path"] == "NOT_CREATED_EXTRACTION_NOT_STARTED", "inventory was not fabricated")
    check("no_sha_registry", lambda: evidence["external_audit"]["sha_registry_path"] == summary["sha_registry_path"] == "NOT_CREATED_EXTRACTION_NOT_STARTED", "SHA registry was not fabricated")

    failures = [item["name"] for item in checks if not item["passed"]]
    payload: dict[str, Any] = {
        "check_count": len(checks),
        "checks": checks,
        "failed_checks": failures,
        "paper_final": False,
        "result": "PASS_SAFE_STOP_EVIDENCE" if not failures else "FAIL",
        "schema_version": "avatarrex_lbn1.plan_b.cloud_recovery_tests.v1",
        "task_id": TASK_ID,
    }
    payload["content_sha256"] = canonical_sha(payload)
    TESTS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{payload['result']}: {len(checks)} checks")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
