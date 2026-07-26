#!/usr/bin/env python3
"""Independently check the Subject00 V2 fail and valid-region audit contract."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
SOURCE_WORKTREE = Path(r"E:\model_train\canondressgs_subject00_portrait_canary_v2_registered_outpaint")
ATTEMPTS_ROOT = Path(r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-GENERATION-001")
ATTEMPTS = {
    "attempt_001": ATTEMPTS_ROOT / "attempt_001",
    "attempt_002": ATTEMPTS_ROOT / "attempt_002_portrait_canary",
    "attempt_003": ATTEMPTS_ROOT / "attempt_003_portrait_canary_v2_registered_outpaint",
}

TASK_ID = "AAAI27-SUBJECT00-PORTRAIT-CANARY-V2-FAIL-AND-VALID-REGION-AUDIT-001"
SOURCE_BRANCH = "research/subject00-portrait-canary-v2-registered-outpaint-20260726"
SOURCE_HEAD = "95370bc256c2e98c849471481fd6248579584f44"
BRANCH = "research/subject00-v2-visual-fail-valid-region-audit-20260726"
REVIEWER = "USER_AND_GPT_MANUAL_REVIEW"
VISUAL_CLASSIFICATION = "SUBJECT00_PORTRAIT_CANARY_V2_TECHNICAL_PASS_VISUAL_FAIL_BACKGROUND_OUTPAINT"
FINAL_CLASSIFICATION = "SUBJECT00_PORTRAIT_CANARY_V2_VISUAL_FAIL_RECORDED_VALID_REGION_PROTOCOL_AUDITED"
NEXT_TASK = "USER_SELECT_SUBJECT00_VALID_REGION_OR_NATIVE_LANDSCAPE_PROTOCOL"
CONTACT_SHA256 = "cde7de35fb0a18a6c2c719fc1c7cd7ad78921545f7be9bf46a70ca7a6abe7f2a"
REVIEW_SHA256 = "3b457c99d029b9ae91f9ccc93a059c779427a391dff7aa865f563f4773af1657"
REQUEST_IDS = [
    "subject00_O01_slot00_cand00",
    "subject00_O03_slot03_cand00",
    "subject00_O04_slot02_cand00",
    "subject00_O01_slot06_cand00",
]

MANUAL_PATH = RISK / "subject00_portrait_canary_v2_manual_visual_adjudication_20260726.json"
EVIDENCE_PATH = RISK / "subject00_portrait_canary_v2_visual_fail_evidence_registry_20260726.json"
CODE_AUDIT_PATH = RISK / "subject00_valid_region_teacher_pipeline_code_audit_20260726.json"
PROTOCOL_PATH = RISK / "subject00_valid_region_protocol_comparison_20260726.json"
SUMMARY_PATH = RISK / "subject00_portrait_canary_v2_visual_fail_valid_region_audit_final_summary_20260726.json"
HANDOFF_PATH = ROOT / "project_control_handoff" / "subject00_v2_visual_fail_valid_region_audit_handoff_20260726.json"
TESTS_PATH = RISK / "subject00_portrait_canary_v2_visual_fail_valid_region_audit_tests_20260726.json"
CONTACT_PATH = ATTEMPTS["attempt_003"] / "07_human_review" / "subject00_portrait_canary_v2_contact_sheet.png"
REVIEW_MANIFEST_PATH = ATTEMPTS["attempt_003"] / "07_human_review" / "subject00_portrait_canary_v2_review_manifest.json"


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
    value = dict(payload)
    value.pop("content_sha256", None)
    value["content_sha256"] = canonical_sha256(value)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def run(*command: str, cwd: Path = ROOT, check: bool = True) -> str:
    return subprocess.run(
        list(command), cwd=cwd, check=check, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def assert_true(value: Any, message: str) -> None:
    if not value:
        raise AssertionError(message)


def current_inventory(root: Path) -> dict[str, tuple[int, str]]:
    return {
        item.relative_to(root).as_posix(): (item.stat().st_size, file_sha256(item))
        for item in sorted(path for path in root.rglob("*") if path.is_file())
    }


def frozen_inventory(snapshot: dict[str, Any]) -> dict[str, tuple[int, str]]:
    return {
        item["relative_path"]: (item["bytes"], item["sha256"])
        for item in snapshot["files"]
    }


def changed_paths() -> list[str]:
    tracked = run("git", "diff", "--name-only", SOURCE_HEAD, "--").splitlines()
    untracked = run("git", "ls-files", "--others", "--exclude-standard").splitlines()
    return sorted(set(path for path in tracked + untracked if path))


def verify_json_hashes(paths: list[Path]) -> None:
    for path in paths:
        payload = load_json(path)
        assert_true(payload.get("content_sha256") == canonical_sha256(payload), f"bad content hash: {path.name}")


def main() -> int:
    manual = load_json(MANUAL_PATH)
    evidence = load_json(EVIDENCE_PATH)
    code = load_json(CODE_AUDIT_PATH)
    protocol = load_json(PROTOCOL_PATH)
    summary = load_json(SUMMARY_PATH)
    handoff = load_json(HANDOFF_PATH)
    review = load_json(REVIEW_MANIFEST_PATH)
    paths = changed_paths()
    checks: list[dict[str, Any]] = []

    def check(name: str, function: Callable[[], None]) -> None:
        try:
            function()
            checks.append({"name": name, "status": "PASS"})
        except Exception as error:  # noqa: BLE001 - retain every gate result
            checks.append({"name": name, "status": "FAIL", "detail": f"{type(error).__name__}: {error}"})

    check("source_branch", lambda: assert_true(
        run("git", "branch", "--show-current", cwd=SOURCE_WORKTREE) == SOURCE_BRANCH,
        "source branch mismatch",
    ))
    check("source_head", lambda: assert_true(
        run("git", "rev-parse", "HEAD", cwd=SOURCE_WORKTREE) == SOURCE_HEAD,
        "source HEAD mismatch",
    ))
    check("source_worktree_clean", lambda: assert_true(
        run("git", "status", "--short", cwd=SOURCE_WORKTREE) == "",
        "source worktree is dirty",
    ))
    check("target_branch", lambda: assert_true(
        run("git", "branch", "--show-current") == BRANCH,
        "target branch mismatch",
    ))
    check("source_head_ancestor", lambda: assert_true(
        subprocess.run(["git", "merge-base", "--is-ancestor", SOURCE_HEAD, "HEAD"], cwd=ROOT).returncode == 0,
        "source HEAD is not an ancestor",
    ))
    check("attempt_001_immutability", lambda: assert_true(
        current_inventory(ATTEMPTS["attempt_001"]) == frozen_inventory(evidence["attempt_snapshots"]["attempt_001"]),
        "attempt_001 changed",
    ))
    check("attempt_002_immutability", lambda: assert_true(
        current_inventory(ATTEMPTS["attempt_002"]) == frozen_inventory(evidence["attempt_snapshots"]["attempt_002"]),
        "attempt_002 changed",
    ))
    check("attempt_003_immutability", lambda: assert_true(
        current_inventory(ATTEMPTS["attempt_003"]) == frozen_inventory(evidence["attempt_snapshots"]["attempt_003"]),
        "attempt_003 changed",
    ))
    check("contact_sheet_sha", lambda: assert_true(
        file_sha256(CONTACT_PATH) == manual["contact_sheet_sha256"] == CONTACT_SHA256,
        "contact sheet SHA mismatch",
    ))
    check("review_manifest_sha", lambda: assert_true(
        file_sha256(REVIEW_MANIFEST_PATH) == manual["original_review_manifest_sha256"] == REVIEW_SHA256,
        "review manifest SHA mismatch",
    ))
    check("frozen_review_manifest_unmodified", lambda: assert_true(
        review["visual_decision_count"] == review["accepted_count"] == review["teacher_target_count"] == 0
        and all(item["visual_review_status"] == "PENDING_USER_REVIEW" for item in review["records"]),
        "frozen review manifest was modified",
    ))
    check("four_request_completeness", lambda: assert_true(
        [item["request_id"] for item in manual["records"]] == REQUEST_IDS,
        "request set/order mismatch",
    ))

    required_fields = {
        "request_id", "garment", "slot", "camera", "source_sha256", "derived_canvas_sha256",
        "output_sha256", "native_resolution_pass", "registration_preflight_pass",
        "subject_scale_visual_result", "pose_direction_visual_result", "garment_correctness_visual_result",
        "human_completeness_visual_result", "background_extension_visual_result", "identity_reviewability",
        "garment_boundary_reviewability", "human_visual_decision", "failure_reasons", "accepted",
        "teacher_target", "reviewer", "review_timestamp",
    }
    check("manual_adjudication_schema", lambda: assert_true(
        all(required_fields.issubset(item) for item in manual["records"]),
        "manual record schema incomplete",
    ))
    check("reviewer_contract", lambda: assert_true(
        manual["reviewer"] == REVIEWER and all(item["reviewer"] == REVIEWER for item in manual["records"]),
        "reviewer contract mismatch",
    ))
    check("visual_pass_count_zero", lambda: assert_true(
        manual["human_visual_pass_count"] == 0
        and not any(item["human_visual_decision"] == "PASS" for item in manual["records"]),
        "visual pass is nonzero",
    ))
    check("visual_fail_count_four", lambda: assert_true(
        manual["human_visual_fail_count"] == 4
        and all(item["human_visual_decision"] == "FAIL" for item in manual["records"]),
        "visual fail count mismatch",
    ))
    check("local_engineering_passes_retained", lambda: assert_true(
        all(item["native_resolution_pass"] and item["registration_preflight_pass"] for item in manual["records"])
        and all(item["local_passes"] for item in manual["records"]),
        "local engineering pass evidence missing",
    ))
    check("background_failure_each_request", lambda: assert_true(
        all(item["background_extension_visual_result"].startswith("FAIL_") and item["failure_reasons"] for item in manual["records"]),
        "background failure not recorded for every request",
    ))
    check("accepted_count_zero", lambda: assert_true(
        manual["accepted_count"] == summary["accepted_count"] == handoff["accepted_count"] == 0
        and not any(item["accepted"] for item in manual["records"]),
        "accepted count is nonzero",
    ))
    check("teacher_target_count_zero", lambda: assert_true(
        manual["teacher_target_count"] == summary["teacher_target_count"] == handoff["teacher_target_count"] == 0
        and not any(item["teacher_target"] for item in manual["records"]),
        "Teacher target count is nonzero",
    ))
    check("remaining_39_zero_and_denied", lambda: assert_true(
        manual["remaining_39_authorization"] == summary["remaining_39_authorization"] == "DENIED"
        and manual["remaining_39_generated_count"] == summary["remaining_39_generated_count"] == 0,
        "remaining-39 state changed",
    ))
    prohibited = evidence["prohibited_action_counts"]
    check("no_generation_call", lambda: assert_true(
        prohibited["new_generation_calls"] == summary["new_generation_calls"] == 0,
        "new generation recorded",
    ))
    check("no_external_api", lambda: assert_true(
        prohibited["external_api_calls"] == summary["external_api_calls"] == 0,
        "external API call recorded",
    ))
    check("no_api_key_read", lambda: assert_true(
        prohibited["api_key_reads"] == summary["api_key_reads"] == 0,
        "API key read recorded",
    ))
    check("no_cloud_image_write", lambda: assert_true(
        prohibited["cloud_image_writes"] == summary["cloud_image_writes"] == 0,
        "cloud image write recorded",
    ))
    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
    check("no_image_git_commit", lambda: assert_true(
        not any(Path(path).suffix.lower() in image_extensions for path in paths),
        "image path entered Git changes",
    ))

    conclusions = code["conclusions"]
    check("arbitrary_resolution_code_audit", lambda: assert_true(
        conclusions["arbitrary_resolution_support"] == "PARTIAL"
        and "msk" in (ROOT / "scene" / "dataset.py").read_text(encoding="utf-8")
        and "image_scaling: 1" in (ROOT / "config" / "subject00.yaml").read_text(encoding="utf-8"),
        "arbitrary-resolution audit incomplete",
    ))
    check("per_view_resolution_code_audit", lambda: assert_true(
        conclusions["per_view_resolution_support"] == "PARTIAL_BASE_PIPELINE_BUT_NO_FORMAL_TEACHER_DATASET"
        and "all rendering samples must share one image size" in (ROOT / "scene" / "dressable_dataset.py").read_text(encoding="utf-8"),
        "per-view resolution evidence missing",
    ))
    check("camera_intrinsics_code_audit", lambda: assert_true(
        conclusions["camera_intrinsics_override_support"].startswith("YES_IF_MATERIALIZED")
        and 'camera_data.get("K", camera_data.get("intrinsics"))' in (ROOT / "utils" / "dressable_camera_utils.py").read_text(encoding="utf-8")
        and "Ks=cam['K'][None]" in (ROOT / "scene" / "gaussian_model.py").read_text(encoding="utf-8"),
        "camera intrinsics evidence missing",
    ))
    check("validity_mask_code_audit", lambda: assert_true(
        conclusions["validity_mask_support"] == "NO_INDEPENDENT_HxW_PIXEL_VALIDITY_MASK"
        and "reference_valid_mask must have shape [K] or [K,1]" in (ROOT / "scene" / "clothing_observation_encoder.py").read_text(encoding="utf-8"),
        "validity-mask audit incomplete",
    ))
    check("loss_mask_propagation_audit", lambda: assert_true(
        conclusions["validity_mask_loss_coverage"] == "NONE_COMPLETE"
        and "lpips = lpips_loss(pred_rgb, target_rgb)" in (ROOT / "utils" / "rendering_loss_utils.py").read_text(encoding="utf-8")
        and "alpha_mask_loss(pred_alpha, foreground)" in (ROOT / "utils" / "oracle_loss_utils.py").read_text(encoding="utf-8"),
        "loss propagation audit incomplete",
    ))
    check("evaluation_mask_propagation_audit", lambda: assert_true(
        conclusions["validity_mask_evaluation_coverage"] == "NONE"
        and "mask_iou(prediction[1]" in (ROOT / "train_full_attribute_oracle.py").read_text(encoding="utf-8")
        and "psnr_test += psnr(image, image_gt)" in (ROOT / "train.py").read_text(encoding="utf-8"),
        "evaluation propagation audit incomplete",
    ))
    check("registered_1024x1150_audit", lambda: assert_true(
        conclusions["registered_1024x1150_support"].startswith("TECHNICALLY_SUPPORTED")
        and code["camera_registration"]["central_only_supervision_with_full_viewport"].startswith("NOT_CURRENTLY_SUPPORTED"),
        "1024x1150 support conclusion incomplete",
    ))
    check("attempt_001_landscape_reuse_audit", lambda: assert_true(
        protocol["attempt_001_landscape_reuse"]["classification"] == "CONDITIONAL_AUDIT_REQUIRED"
        and protocol["attempt_001_landscape_reuse"]["non_exact_output_count"] == 43
        and protocol["attempt_001_landscape_reuse"]["landscape_count"] == 38
        and protocol["attempt_001_landscape_reuse"]["reliable_registered_subset"] == "NOT_ESTABLISHED",
        "attempt_001 reuse audit incomplete",
    ))
    check("candidate_protocol_completeness", lambda: assert_true(
        set(protocol["candidates"]) == {"A", "B", "C", "D"}
        and all({"name", "classification", "recommendation_level"}.issubset(value) for value in protocol["candidates"].values()),
        "candidate table incomplete",
    ))
    check("candidate_A_blocked", lambda: assert_true(
        protocol["candidates"]["A"]["classification"] == "BLOCKED_BY_MISSING_END_TO_END_PIXEL_VALIDITY_SUPPORT"
        and len(protocol["candidates"]["A"]["required_code_changes"]) >= 9,
        "Candidate A not correctly blocked",
    ))
    check("candidate_D_rejected", lambda: assert_true(
        protocol["candidates"]["D"]["classification"] == protocol["candidates"]["D"]["recommendation_level"] == "REJECT",
        "Candidate D not rejected",
    ))
    check("no_protocol_executed", lambda: assert_true(
        not protocol["execution_authorized"] and protocol["recommended_protocol"] == "NO_EXECUTABLE_PROTOCOL_CURRENTLY",
        "a candidate protocol was executed or selected",
    ))

    training_roots = ("scene/", "utils/")
    training_files = {"train.py", "test.py", "train_dressable.py", "train_full_attribute_oracle.py"}
    check("no_training_code_modification", lambda: assert_true(
        not any(path in training_files or path.startswith(training_roots) for path in paths),
        "training code changed",
    ))
    check("no_paper_modification", lambda: assert_true(
        summary["paper_modifications"] == 0
        and not any(path.startswith("docs/PAPER/") or Path(path).suffix.lower() in {".tex", ".bib"} for path in paths),
        "paper source changed",
    ))
    check("json_content_hashes", lambda: verify_json_hashes([
        MANUAL_PATH, EVIDENCE_PATH, CODE_AUDIT_PATH, PROTOCOL_PATH, SUMMARY_PATH, HANDOFF_PATH,
    ]))
    check("visual_classification", lambda: assert_true(
        manual["classification"] == summary["visual_classification"] == VISUAL_CLASSIFICATION,
        "visual classification mismatch",
    ))
    check("final_classification", lambda: assert_true(
        summary["final_classification"] == handoff["final_classification"] == FINAL_CLASSIFICATION,
        "final classification mismatch",
    ))
    check("paper_final_false", lambda: assert_true(
        manual["paper_final"] is summary["paper_final"] is handoff["paper_final"] is protocol["paper_final"] is False,
        "PAPER_FINAL changed",
    ))
    check("next_task_unique", lambda: assert_true(
        {summary["next_task"], handoff["next_task"], protocol["next_task"]} == {NEXT_TASK},
        "NEXT_TASK is not unique",
    ))

    failed = [item for item in checks if item["status"] != "PASS"]
    payload = {
        "schema_version": "canondressgs.subject00.v2_visual_fail_valid_region_audit_tests.v1",
        "task_id": TASK_ID,
        "source_head": SOURCE_HEAD,
        "branch": BRANCH,
        "test_count": len(checks),
        "pass_count": len(checks) - len(failed),
        "fail_count": len(failed),
        "result": "PASS" if not failed else "FAIL",
        "checks": checks,
        "final_classification": FINAL_CLASSIFICATION if not failed else "SUBJECT00_VALID_REGION_AUDIT_CONTRACT_VIOLATION",
        "paper_final": False,
        "next_task": NEXT_TASK if not failed else "REPAIR_SUBJECT00_VALID_REGION_AUDIT_CONTRACT",
    }
    write_json(TESTS_PATH, payload)
    print(json.dumps({
        "result": payload["result"],
        "pass_count": payload["pass_count"],
        "fail_count": payload["fail_count"],
        "tests_path": str(TESTS_PATH),
        "failed": failed,
    }, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
