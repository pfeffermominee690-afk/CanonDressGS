#!/usr/bin/env python3
"""Verify the evidence-only Subject00 portrait canary visual-fail adjudication."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Callable

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
HANDOFF_PATH = ROOT / "project_control_handoff" / "subject00_portrait_canary_visual_fail_adjudication_handoff.json"
ATTEMPT_002 = Path(r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-GENERATION-001\attempt_002_portrait_canary")
TASK_ID = "AAAI27-SUBJECT00-PORTRAIT-CANARY-VISUAL-FAIL-ADJUDICATION-001"
SOURCE_BRANCH = "research/subject00-managed-portrait-canary-20260725"
SOURCE_HEAD = "4a1eeb2e53fc1a8b5a5bf1c3b706aac374e156dc"
PORTRAIT_CANARY_RESULT_HEAD = "9742c675943c2b105a5b153888e5b160b73471f2"
BRANCH = "research/subject00-portrait-canary-visual-fail-adjudication-20260726"
REVIEWER = "USER_AND_GPT_MANUAL_REVIEW"
CLASSIFICATION = "SUBJECT00_PORTRAIT_CANARY_VISUAL_FAIL_FORMALLY_RECORDED_NEXT_PROTOCOL_PENDING_USER_DECISION"
NEXT_TASK = "USER_SELECT_SUBJECT00_PORTRAIT_CANARY_V2_PROTOCOL"

EVIDENCE_PATH = RISK / "subject00_portrait_canary_visual_fail_evidence_registry_20260726.json"
ADJUDICATION_PATH = RISK / "subject00_portrait_canary_manual_visual_adjudication_20260726.json"
OVERLAY_PATH = RISK / "subject00_portrait_canary_review_overlay_20260726.json"
SUMMARY_PATH = RISK / "subject00_portrait_canary_visual_fail_final_summary_20260726.json"
TESTS_PATH = RISK / "subject00_portrait_canary_visual_fail_tests_20260726.json"
REPORT_PATH = RISK / "SUBJECT00_PORTRAIT_CANARY_VISUAL_FAIL_REPORT_20260726.md"
CONTRACT_PATH = RISK / "SUBJECT00_RESOLUTION_CAMERA_REGISTRATION_CONTRACT_ADJUDICATION_20260726.md"


def canonical_sha256(value: Any) -> str:
    if isinstance(value, dict):
        value = dict(value)
        value.pop("content_sha256", None)
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload.pop("content_sha256", None)
    payload["content_sha256"] = canonical_sha256(payload)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def run(*command: str) -> str:
    return subprocess.run(
        list(command), cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def assert_true(value: Any, message: str) -> None:
    if not value:
        raise AssertionError(message)


def current_inventory(root: Path) -> dict[str, tuple[int, str]]:
    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, file_sha256(path))
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def main() -> int:
    evidence = load_json(EVIDENCE_PATH)
    adjudication = load_json(ADJUDICATION_PATH)
    overlay = load_json(OVERLAY_PATH)
    summary = load_json(SUMMARY_PATH)
    handoff = load_json(HANDOFF_PATH)
    source_outputs = load_json(RISK / "subject00_portrait_canary_output_registry.json")
    original_manifest_path = Path(evidence["original_review_manifest"]["path"])
    original_manifest = load_json(original_manifest_path)
    checks: list[dict[str, Any]] = []

    def check(name: str, function: Callable[[], None]) -> None:
        try:
            function()
            checks.append({"name": name, "status": "PASS"})
        except Exception as error:  # noqa: BLE001 - every failed contract gate is evidence
            checks.append({"name": name, "status": "FAIL", "detail": f"{type(error).__name__}: {error}"})

    check("source_provenance", lambda: assert_true(
        evidence["source_branch"] == adjudication["source_branch"] == summary["source_branch"] == SOURCE_BRANCH
        and evidence["source_head"] == adjudication["source_head"] == summary["source_head"] == SOURCE_HEAD
        and adjudication["portrait_canary_result_head"] == PORTRAIT_CANARY_RESULT_HEAD,
        "source provenance mismatch",
    ))
    check("source_head", lambda: assert_true(
        run("git", "rev-parse", SOURCE_BRANCH) == SOURCE_HEAD
        and subprocess.run(["git", "merge-base", "--is-ancestor", SOURCE_HEAD, "HEAD"], cwd=ROOT).returncode == 0,
        "source HEAD is not the exact ancestor",
    ))
    check("worktree_clean", lambda: assert_true(run("git", "status", "--short") == "", "worktree is not clean at test start"))
    check("adjudication_branch", lambda: assert_true(run("git", "branch", "--show-current") == BRANCH, "wrong branch"))

    def verify_attempt_001() -> None:
        assert_true(evidence["attempt_001_output_count"] == len(evidence["attempt_001_outputs"]) == 48, "attempt_001 count mismatch")
        for item in evidence["attempt_001_outputs"]:
            path = Path(item["path"])
            assert_true(path.is_file() and path.stat().st_size == item["bytes"] and file_sha256(path) == item["sha256"], f"attempt_001 changed: {path}")

    check("attempt_001_immutability", verify_attempt_001)
    check("attempt_001_native_pass_five", lambda: assert_true(
        len(evidence["attempt_001_preserved_native_pass_outputs"]) == 5
        and all(item["preserved_native_pass_output"] for item in evidence["attempt_001_preserved_native_pass_outputs"]),
        "preserved native PASS evidence mismatch",
    ))

    def verify_attempt_002() -> None:
        expected = {
            item["relative_path"]: (item["bytes"], item["sha256"])
            for item in evidence["attempt_002_files"]
        }
        assert_true(current_inventory(ATTEMPT_002) == expected, "attempt_002 file inventory changed")

    check("attempt_002_complete_immutability", verify_attempt_002)
    check("attempt_002_image_immutability", lambda: [
        assert_true(Path(item["path"]).is_file() and Path(item["path"]).stat().st_size == item["bytes"] and file_sha256(Path(item["path"])) == item["sha256"], f"image changed: {item['path']}")
        for item in evidence["derived_canvases"] + evidence["generated_outputs"] + [evidence["contact_sheet"]]
    ])
    check("four_request_completeness", lambda: assert_true(
        adjudication["request_count"] == len(adjudication["records"]) == len(source_outputs["records"]) == 4,
        "four-request evidence incomplete",
    ))
    check("four_output_sha", lambda: [
        assert_true(file_sha256(Path(item["generated_output_path"])) == item["generated_output_sha256"], f"output SHA mismatch: {item['request_id']}")
        for item in adjudication["records"]
    ])

    def verify_native_resolution() -> None:
        for item in adjudication["records"]:
            with Image.open(item["generated_output_path"]) as image:
                image.load()
                assert_true(image.format == "PNG" and image.size == (1024, 1536), f"native resolution failed: {item['request_id']}")
            assert_true(item["native_resolution_pass"] and item["png_parse_pass"], "recorded native evidence mismatch")

    check("native_resolution_evidence", verify_native_resolution)
    check("contact_sheet_readable", lambda: assert_true(
        evidence["contact_sheet"]["format"] == "PNG"
        and evidence["contact_sheet_group_count_confirmed_by_manual_open"] == 4
        and file_sha256(Path(evidence["contact_sheet"]["path"])) == evidence["contact_sheet"]["sha256"],
        "contact sheet evidence mismatch",
    ))
    check("original_manifest_preservation", lambda: assert_true(
        original_manifest_path.stat().st_size == evidence["original_review_manifest"]["bytes"]
        and file_sha256(original_manifest_path) == evidence["original_review_manifest"]["sha256"]
        and overlay["original_manifest_sha256"] == evidence["original_review_manifest"]["sha256"]
        and not overlay["original_manifest_modified"],
        "original manifest changed",
    ))
    check("original_manifest_visual_fields_null", lambda: assert_true(
        original_manifest["visual_decision_count"] == 0
        and all(item["decision"] is None for item in original_manifest["records"]),
        "original manifest was adjudicated in place",
    ))
    check("human_review_overlay_schema", lambda: assert_true(
        overlay["overlay_only"] is True
        and len(overlay["records"]) == 4
        and all(item["human_visual_decision"] == "FAIL" and not item["accepted"] and not item["teacher_target"] for item in overlay["records"]),
        "overlay contract mismatch",
    ))
    required_manual_fields = {
        "request_id", "garment", "slot", "camera", "original_condition_sha256", "derived_canvas_sha256",
        "generated_output_sha256", "native_resolution_pass", "png_parse_pass", "single_person",
        "garment_correctness", "pose_direction_preservation", "subject_scale_preservation",
        "camera_framing_preservation", "background_extension_quality", "identity_reviewability",
        "garment_boundary_reviewability", "human_visual_decision", "failure_reasons", "accepted",
        "teacher_target", "reviewer", "review_timestamp", "overall_classification",
    }
    check("manual_adjudication_schema", lambda: assert_true(
        all(required_manual_fields.issubset(item) for item in adjudication["records"]),
        "manual adjudication field missing",
    ))
    check("fixed_human_visual_fail_4_of_4", lambda: assert_true(
        adjudication["human_visual_pass_count"] == 0
        and adjudication["human_visual_fail_count"] == 4
        and all(item["human_visual_decision"] == "FAIL" for item in adjudication["records"]),
        "fixed human verdict changed",
    ))
    check("reviewer_identity", lambda: assert_true(
        adjudication["reviewer"] == REVIEWER and all(item["reviewer"] == REVIEWER for item in adjudication["records"]),
        "reviewer field mismatch",
    ))
    check("subject_scale_camera_background_fail", lambda: assert_true(
        all(
            item["subject_scale_preservation"] == "FAIL"
            and item["camera_framing_preservation"] == "FAIL"
            and item["background_extension_quality"] == "FAIL"
            for item in adjudication["records"]
        ),
        "common root cause not recorded",
    ))
    check("accepted_count_zero", lambda: assert_true(
        adjudication["accepted_count"] == summary["accepted_count"] == handoff["accepted_count"] == 0
        and not any(item["accepted"] for item in adjudication["records"]),
        "accepted count is nonzero",
    ))
    check("teacher_target_count_zero", lambda: assert_true(
        adjudication["teacher_target_count"] == summary["teacher_target_count"] == handoff["teacher_target_count"] == 0
        and not any(item["teacher_target"] for item in adjudication["records"]),
        "Teacher target count is nonzero",
    ))
    check("remaining_39_not_generated", lambda: assert_true(
        summary["remaining_39_generated_count"] == 0
        and not summary["remaining_39_rerun_authorized"]
        and not handoff["remaining_39_rerun_authorized"]
        and len(evidence["generated_outputs"]) == 4
        and not ATTEMPT_002.with_name("attempt_003").exists(),
        "remaining generation was materialized or authorized",
    ))
    check("no_new_generation_calls", lambda: assert_true(
        evidence["new_generation_calls"] == summary["new_generation_calls"] == 0,
        "new generation call recorded",
    ))
    check("no_external_api", lambda: assert_true(
        evidence["external_api_calls"] == summary["external_api_calls"] == 0,
        "external API call recorded",
    ))
    check("no_api_key_read", lambda: assert_true(
        evidence["api_key_reads"] == summary["api_key_reads"] == 0,
        "API key read recorded",
    ))
    check("no_cloud_image_write", lambda: assert_true(
        evidence["cloud_image_writes"] == summary["cloud_image_writes"] == 0,
        "cloud image write recorded",
    ))
    check("formal_base_pending", lambda: assert_true(
        adjudication["formal_base"] == summary["formal_base"] == handoff["formal_base"] == "PENDING",
        "Formal Base status changed",
    ))
    check("technical_and_visual_status_separated", lambda: assert_true(
        summary["engineering_pass"] is True
        and summary["native_resolution_pass_count"] == 4
        and summary["human_visual_pass_count"] == 0
        and handoff["engineering_status"] == "PASS"
        and handoff["visual_status"] == "FAIL_4_OF_4",
        "engineering/native and visual status were collapsed",
    ))

    changed = run("git", "diff", "--name-only", SOURCE_HEAD, "--").splitlines()
    image_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
    check("no_image_git_commit", lambda: assert_true(
        not any(Path(path).suffix.lower() in image_suffixes for path in changed),
        "image added to Git changes",
    ))
    check("no_paper_modification", lambda: assert_true(
        not any(Path(path).suffix.lower() == ".tex" or Path(path).name == "main.tex" for path in changed)
        and summary["paper_modifications"] == 0,
        "paper source changed",
    ))
    check("required_adjudication_documents", lambda: assert_true(
        REPORT_PATH.is_file() and CONTRACT_PATH.is_file(),
        "required adjudication report missing",
    ))
    check("candidate_protocols_not_executed", lambda: assert_true(
        "does not select or execute a candidate" in CONTRACT_PATH.read_text(encoding="utf-8")
        and summary["new_generation_calls"] == 0,
        "candidate protocol was selected or executed",
    ))
    check("content_sha256_seals", lambda: [
        assert_true(canonical_sha256(payload) == payload["content_sha256"], f"seal mismatch: {path.name}")
        for path, payload in [
            (EVIDENCE_PATH, evidence), (ADJUDICATION_PATH, adjudication), (OVERLAY_PATH, overlay),
            (SUMMARY_PATH, summary), (HANDOFF_PATH, handoff),
        ]
    ])
    check("no_secret_like_values", lambda: assert_true(
        re.search(
            r"(?i)(sk-[a-z0-9_-]{16,}|bearer\s+[a-z0-9._-]{16,}|api[_-]?key\s*[:=]\s*[\"'][^\"']{8,})",
            "\n".join(path.read_text(encoding="utf-8") for path in [EVIDENCE_PATH, ADJUDICATION_PATH, OVERLAY_PATH, SUMMARY_PATH, HANDOFF_PATH]),
        ) is None,
        "secret-like value found",
    ))
    check("final_classification", lambda: assert_true(
        summary["classification"] == handoff["classification"] == CLASSIFICATION,
        "final classification mismatch",
    ))
    check("next_task_unique", lambda: assert_true(
        {summary["next_task"], handoff["next_task"]} == {NEXT_TASK}
        and not summary["next_task_authorized"]
        and not handoff["next_task_authorized"],
        "NEXT_TASK mismatch or authorization leak",
    ))
    check("paper_final_false", lambda: assert_true(
        all(payload["paper_final"] is False for payload in [evidence, adjudication, overlay, summary, handoff]),
        "PAPER_FINAL is true",
    ))

    failed = [item for item in checks if item["status"] == "FAIL"]
    status = f"PASS_{len(checks)}_VISUAL_FAIL_ADJUDICATION_CHECKS" if not failed else f"FAIL_{len(failed)}_OF_{len(checks)}"
    tests = {
        "schema_version": "canondressgs.subject00.portrait_canary_visual_fail_tests.v1",
        "task_id": TASK_ID,
        "status": status,
        "total": len(checks),
        "passed": len(checks) - len(failed),
        "failed": failed,
        "checks": checks,
        "paper_final": False,
    }
    write_json(TESTS_PATH, tests)
    for path in [SUMMARY_PATH, HANDOFF_PATH]:
        payload = load_json(path)
        payload["tests"] = status
        write_json(path, payload)
    print(json.dumps({"status": status, "total": len(checks), "failed": failed}, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
