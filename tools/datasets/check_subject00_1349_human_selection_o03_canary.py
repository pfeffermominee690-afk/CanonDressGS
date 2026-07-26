#!/usr/bin/env python3
"""Independently audit the frozen Subject00 selection and O03 canary draft."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
SOURCE_WORKTREE = Path(
    r"E:\model_train\canondressgs_subject00_1349_cohort_correction_human_review_prep"
)
SOURCE_BRANCH = "research/subject00-1349-cohort-correction-human-review-prep-20260726"
SOURCE_HEAD = "f6f634c029abc72191ce1edbc99e56fe781f0bfd"
TARGET_BRANCH = "research/subject00-1349-human-selection-o03-canary-prep-20260726"
TASK_ID = "AAAI27-SUBJECT00-1349-HUMAN-SELECTION-AND-O03-CANARY-PREP-001"
FINAL_CLASSIFICATION = "SUBJECT00_1349_HUMAN_SELECTION_FROZEN_O03_CANARY_PENDING_AUTHORIZATION"
NEXT_TASK = "USER_AUTHORIZE_SUBJECT00_O03_HOOD_REMOVAL_CANARY_4_REQUESTS"
FAILURE_TAG = "SOURCE_GARMENT_HOOD_RESIDUAL_IN_O03_SUIT"

DATA_ROOT = Path(r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-GENERATION-001")
ATTEMPTS = {
    "attempt_001": DATA_ROOT / "attempt_001",
    "attempt_002": DATA_ROOT / "attempt_002_portrait_canary",
    "attempt_003": DATA_ROOT / "attempt_003_portrait_canary_v2_registered_outpaint",
}
AUDIT_ROOT = DATA_ROOT / "attempt_001_native_landscape_registration_audit"
BASELINE_PATH = AUDIT_ROOT / "00_provenance" / "attempt_immutability_baseline.json"
EXTERNAL_OLD_REVIEW = (
    AUDIT_ROOT / "08_corrected_1349_human_review" / "corrected_1349x1166_human_review_manifest.json"
)
REPO_OLD_REVIEW = RISK / "corrected_1349x1166_human_review_manifest.json"

OVERLAY = RISK / "subject00_1349_final_human_selection_overlay_20260726.json"
CANDIDATES = RISK / "subject00_1349_candidate_decision_registry_20260726.json"
SELECTED_REGISTRY = RISK / "subject00_1349_selected_cell_registry_20260726.json"
MISSING_REGISTRY = RISK / "subject00_1349_final_missing_cell_registry_20260726.json"
SUMMARY = RISK / "subject00_1349_human_selection_final_summary_20260726.json"
CANARY = RISK / "subject00_o03_hood_removal_canary_manifest_draft_20260726.json"
TESTS = RISK / "subject00_1349_human_selection_o03_canary_tests_20260726.json"
HANDOFF = ROOT / "project_control_handoff" / "subject00_1349_human_selection_o03_canary_prep_handoff_20260726.json"

SELECTED = {
    "subject00_O01_slot00_cand01",
    "subject00_O01_slot01_cand01",
    "subject00_O01_slot02_cand01",
    "subject00_O01_slot03_cand01",
    "subject00_O01_slot05_cand00",
    "subject00_O01_slot06_cand01",
    "subject00_O03_slot02_cand00",
    "subject00_O03_slot03_cand01",
    "subject00_O04_slot00_cand01",
    "subject00_O04_slot01_cand01",
    "subject00_O04_slot02_cand00",
    "subject00_O04_slot03_cand01",
    "subject00_O04_slot04_cand01",
    "subject00_O04_slot07_cand01",
}
FAILED = {
    "subject00_O03_slot00_cand00",
    "subject00_O03_slot00_cand01",
    "subject00_O03_slot01_cand00",
    "subject00_O03_slot01_cand01",
    "subject00_O03_slot02_cand01",
    "subject00_O03_slot03_cand00",
    "subject00_O03_slot04_cand00",
    "subject00_O03_slot05_cand00",
}
MISSING = {
    "O01/slot_04",
    "O01/slot_07",
    "O03/slot_00",
    "O03/slot_01",
    "O03/slot_04",
    "O03/slot_05",
    "O03/slot_06",
    "O03/slot_07",
    "O04/slot_05",
    "O04/slot_06",
}
CANARY_IDS = {
    "subject00_O03_slot00_canary_attempt004_cand00",
    "subject00_O03_slot04_canary_attempt004_cand00",
    "subject00_O03_slot05_canary_attempt004_cand00",
    "subject00_O03_slot07_canary_attempt004_cand00",
}
CANARY_CELLS = {
    ("slot_00", "cam17", "front"),
    ("slot_04", "cam11", "right"),
    ("slot_05", "cam02", "back-left"),
    ("slot_07", "cam05", "back"),
}
PROMPT_CLAUSES = {
    "Replace the entire source hoodie with a complete formal suit.",
    "Remove the original blue-and-white hoodie completely.",
    "Remove the source hood from the head and neck region.",
    "The final outfit must contain no hood.",
    "The head, hair, face, ears, and neck must remain naturally visible according to the source identity and camera direction.",
    "Preserve identity, face structure, skin tone, body proportions, pose, hands, feet, camera, framing, lighting, and background.",
    "Generate one person only.",
    "Do not retain blue hoodie fabric around the head, neck, shoulders, torso, sleeves, or waist.",
    "Do not add a coat hood, sweatshirt hood, scarf-like hood, or head covering.",
    "Produce a complete suit jacket, shirt, trousers, and formal styling under the frozen O03 garment contract.",
}
ACCEPTANCE_GATES = {
    "exact_resolution",
    "similarity_registration",
    "complete_hood_removal",
    "correct_O03_suit",
    "identity_consistency",
    "pose_camera_preservation",
    "hands_feet_completeness",
    "background_geometry",
    "garment_boundary",
    "single_person",
}

ALLOWED_CHANGED_PATHS = {
    "paper_protocol/reviewer_risk/subject00_1349_final_human_selection_overlay_20260726.json",
    "paper_protocol/reviewer_risk/SUBJECT00_1349_FINAL_HUMAN_SELECTION_REPORT_20260726.md",
    "paper_protocol/reviewer_risk/subject00_1349_candidate_decision_registry_20260726.json",
    "paper_protocol/reviewer_risk/subject00_1349_selected_cell_registry_20260726.json",
    "paper_protocol/reviewer_risk/subject00_1349_final_missing_cell_registry_20260726.json",
    "paper_protocol/reviewer_risk/subject00_1349_human_selection_final_summary_20260726.json",
    "paper_protocol/reviewer_risk/SUBJECT00_O03_HOOD_REMOVAL_CANARY_CONTRACT_20260726.md",
    "paper_protocol/reviewer_risk/subject00_o03_hood_removal_canary_manifest_draft_20260726.json",
    "paper_protocol/reviewer_risk/subject00_1349_human_selection_o03_canary_tests_20260726.json",
    "project_control_handoff/subject00_1349_human_selection_o03_canary_prep_handoff_20260726.json",
    "tools/datasets/freeze_subject00_1349_human_selection_o03_canary.py",
    "tools/datasets/check_subject00_1349_human_selection_o03_canary.py",
}


def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: dict[str, Any]) -> str:
    clone = dict(value)
    clone.pop("content_sha256", None)
    raw = json.dumps(clone, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def inventory(root: Path) -> dict[str, tuple[int, str]]:
    result: dict[str, tuple[int, str]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        result[path.relative_to(root).as_posix()] = (path.stat().st_size, file_sha(path))
    return result


def baseline_inventory(value: dict[str, Any]) -> dict[str, tuple[int, str]]:
    return {item["relative_path"]: (item["bytes"], item["sha256"]) for item in value["files"]}


def changed_paths() -> set[str]:
    committed = set(filter(None, git("diff", "--name-only", SOURCE_HEAD, "--").splitlines()))
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    untracked = {
        line[3:].replace("\\", "/")
        for line in status.splitlines()
        if line.startswith("?? ")
    }
    return committed | untracked


def write_results(checks: list[dict[str, Any]]) -> None:
    failed = [item["name"] for item in checks if not item["passed"]]
    payload: dict[str, Any] = {
        "check_count": len(checks),
        "checks": checks,
        "failed_checks": failed,
        "paper_final": False,
        "result": "PASS" if not failed else "FAIL",
        "schema_version": "canondressgs.subject00.1349_human_selection_o03_canary_tests.v1",
        "task_id": TASK_ID,
    }
    payload["content_sha256"] = canonical_sha(payload)
    TESTS.parent.mkdir(parents=True, exist_ok=True)
    with TESTS.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    overlay = load(OVERLAY)
    candidates = load(CANDIDATES)
    selected_registry = load(SELECTED_REGISTRY)
    missing_registry = load(MISSING_REGISTRY)
    summary = load(SUMMARY)
    canary = load(CANARY)
    handoff = load(HANDOFF)
    old_review = load(REPO_OLD_REVIEW)
    records = candidates["records"]
    by_id = {item["request_id"]: item for item in records}
    selected_ids = {item["request_id"] for item in records if item["selected_for_cell"]}
    failed_ids = {
        item["request_id"] for item in records if item["human_visual_decision"] == "FAIL"
    }
    pass_not_selected = [
        item for item in records if item["human_visual_decision"] == "PASS_NOT_SELECTED"
    ]
    missing_cells = {item["cell_key"] for item in missing_registry["records"]}
    canary_ids = {item["request_id"] for item in canary["requests"]}
    canary_cells = {
        (item["slot"], item["camera"], item["orientation"]) for item in canary["requests"]
    }
    baseline = load(BASELINE_PATH)
    old_repo_sha = file_sha(REPO_OLD_REVIEW)
    old_external_sha = file_sha(EXTERNAL_OLD_REVIEW)
    external_old_review = load(EXTERNAL_OLD_REVIEW)

    checks: list[dict[str, Any]] = []

    def check(name: str, predicate: Callable[[], bool], detail: str) -> None:
        try:
            passed = bool(predicate())
            observed = detail if passed else f"FAILED: {detail}"
        except Exception as error:  # audit output must preserve every failed assertion
            passed = False
            observed = f"{type(error).__name__}: {error}"
        checks.append({"detail": observed, "name": name, "passed": passed})

    check(
        "source_branch_and_head",
        lambda: git("branch", "--show-current", cwd=SOURCE_WORKTREE) == SOURCE_BRANCH
        and git("rev-parse", "HEAD", cwd=SOURCE_WORKTREE) == SOURCE_HEAD,
        "source branch and HEAD exactly match the frozen contract",
    )
    check(
        "source_worktree_clean",
        lambda: git("status", "--short", cwd=SOURCE_WORKTREE) == "",
        "source worktree has no tracked or untracked changes",
    )
    check(
        "target_branch_and_ancestry",
        lambda: git("branch", "--show-current") == TARGET_BRANCH
        and subprocess.run(
            ["git", "merge-base", "--is-ancestor", SOURCE_HEAD, "HEAD"], cwd=ROOT
        ).returncode
        == 0,
        "target branch is correct and descends from the frozen source HEAD",
    )
    check(
        "attempt_immutability",
        lambda: all(inventory(path) == baseline_inventory(baseline[name]) for name, path in ATTEMPTS.items()),
        "attempt_001, attempt_002, and attempt_003 exactly match their frozen inventories",
    )
    check(
        "old_review_manifest_sha",
        lambda: old_review["content_sha256"] == external_old_review["content_sha256"]
        and overlay["old_review_manifest_sha256"] == old_external_sha
        and overlay["old_review_manifest_repo_copy_sha256"] == old_repo_sha
        and overlay["old_review_manifest_canonical_content_sha256"]
        == old_review["content_sha256"]
        and overlay["repo_and_external_canonical_content_match"] is True
        and overlay["old_review_manifest_was_modified"] is False,
        "external evidence SHA, repo-copy SHA, and equal canonical content are recorded",
    )
    check("generated_json_content_hashes", lambda: all(load(path)["content_sha256"] == canonical_sha(load(path)) for path in (OVERLAY, CANDIDATES, SELECTED_REGISTRY, MISSING_REGISTRY, SUMMARY, CANARY, HANDOFF)), "all generated non-test JSON content hashes verify")
    check("candidate_count_31", lambda: candidates["record_count"] == len(records) == 31 and overlay["reviewed_candidate_count"] == 31, "31 high-resolution review candidates are bound")
    check("pass_count_23", lambda: Counter(item["human_visual_decision"] != "FAIL" for item in records)[True] == 23 and candidates["pass_candidate_count"] == 23, "23 candidates pass the final visual review")
    check("fail_count_8", lambda: failed_ids == FAILED and candidates["fail_count"] == 8, "the exact eight O03 hood-residual candidates fail")
    check("uncertain_count_zero", lambda: candidates["uncertain_count"] == 0 and all(item["human_visual_decision"] != "UNCERTAIN" for item in records), "no candidate is uncertain")
    check("revised_o03_slot02_cand01", lambda: by_id["subject00_O03_slot02_cand01"]["decision_revision"] == {"previous_preliminary_decision": "PASS_CANDIDATE", "final_high_res_cell_review_decision": "FAIL", "revision_reason": "DARK_BLUE_SOURCE_HOOD_REVEALED_BY_HIGH_RES_SIDE_BY_SIDE_REVIEW"}, "the preliminary pass is explicitly revised to final fail")
    check("selected_count_14", lambda: selected_ids == SELECTED and selected_registry["record_count"] == 14, "the exact 14 preferred requests are selected")
    check("unique_selection_per_cell", lambda: len({item["cell_key"] for item in selected_registry["records"]}) == 14 and selected_registry["unique_selection_per_cell"] is True, "each selected request belongs to a unique garment-slot cell")
    check("o01_selection_complete", lambda: {item for item in selected_ids if "_O01_" in item} == {item for item in SELECTED if "_O01_" in item} and sum("_O01_" in item for item in selected_ids) == 6, "the six frozen O01 selections match exactly")
    check("o03_selection_complete", lambda: {item for item in selected_ids if "_O03_" in item} == {item for item in SELECTED if "_O03_" in item} and sum("_O03_" in item for item in selected_ids) == 2, "the two frozen O03 selections match exactly")
    check("o04_selection_complete", lambda: {item for item in selected_ids if "_O04_" in item} == {item for item in SELECTED if "_O04_" in item} and sum("_O04_" in item for item in selected_ids) == 6, "the six frozen O04 selections match exactly")
    check("missing_count_10", lambda: missing_registry["missing_cell_count"] == len(missing_registry["records"]) == 10, "ten cells remain missing")
    check("missing_list_exact", lambda: missing_cells == MISSING, "the missing-cell set exactly matches the user decision")
    check("missing_rerun_state", lambda: all(item["rerun_required"] is True and item["rerun_authorized"] is False for item in missing_registry["records"]), "all missing cells require rerun and none is authorized")
    check("failed_candidates_exact", lambda: failed_ids == FAILED and all(by_id[item]["failure_tags"] == [FAILURE_TAG] for item in FAILED), "failed request IDs and failure tags match exactly")
    check("pass_not_selected_semantics", lambda: len(pass_not_selected) == 9 and all(not item["selected_for_cell"] and not item["accepted"] and not item["teacher_target"] and not item["failure_tags"] for item in pass_not_selected), "nine visually passing alternatives remain unselected without being mislabeled as failures")
    check("accepted_zero", lambda: candidates["accepted_count"] == selected_registry["accepted_count"] == summary["accepted_count"] == handoff["accepted_count"] == 0 and all(not item["accepted"] for item in records), "accepted promotion count is zero")
    check("teacher_target_zero", lambda: candidates["teacher_target_count"] == selected_registry["teacher_target_count"] == summary["teacher_target_count"] == handoff["teacher_target_count"] == 0 and all(not item["teacher_target"] for item in records), "Teacher-target promotion count is zero")
    check("canary_request_count_and_ids", lambda: canary["request_count"] == len(canary["requests"]) == 4 and canary_ids == CANARY_IDS, "the draft contains exactly four new request IDs")
    check("canary_cells_exact", lambda: canary_cells == CANARY_CELLS and all(item["garment"] == "O03" for item in canary["requests"]), "front, right, back-left, and back fixed cells match exactly")
    check("canary_source_bindings", lambda: all(Path(item["local_source_condition_path"]).is_file() and file_sha(Path(item["local_source_condition_path"])) == item["source_condition_sha256"] and Path(item["source_request_path"]).is_file() and file_sha(Path(item["source_request_path"])) == item["source_request_sha256"] for item in canary["requests"]), "every canary request is bound to an existing original request and registered condition SHA")
    check("canary_native_resolution", lambda: canary["expected_resolution"] == {"width": 1349, "height": 1166, "orientation": "landscape"} and all(item["expected_resolution"] == canary["expected_resolution"] for item in canary["requests"]) and canary["expected_output_format"] == "PNG" and canary["native_resolution_required"] is True and canary["png_parse_required"] is True, "all draft outputs require parseable native 1349x1166 landscape PNG")
    check("canary_prompt_contract", lambda: set(canary["prompt_contract"]) == PROMPT_CLAUSES, "the complete hood-removal and preservation prompt contract is frozen")
    check("canary_retry_and_postprocessing", lambda: canary["retry_policy"] == {"automatic_retry": False, "candidate_count_per_cell": 1, "retry_count": 0} and all(value is False for value in canary["postprocessing_policy"].values()), "one candidate per cell, zero retries, and no postprocessing")
    check("canary_acceptance_gates", lambda: set(canary["acceptance_gates"]) == ACCEPTANCE_GATES and canary["all_four_requests_must_pass"] is True and canary["automatic_acceptance"] is False, "all ten human gates require 4/4 passage with no automatic acceptance")
    check("generation_not_authorized", lambda: canary["generation_authorized"] is False and summary["canary_generation_authorized"] is False and handoff["canary_generation_authorized"] is False, "canary execution remains unauthorized")
    check("no_generation_calls", lambda: canary["generation_call_count"] == summary["new_generation_calls"] == handoff["new_generation_calls"] == 0 and summary["canary_generated_count"] == summary["targeted_rerun_generated_count"] == handoff["targeted_rerun_generated_count"] == 0, "no image generation or targeted rerun occurred")
    check("no_external_api_or_key_read", lambda: canary["external_api_allowed"] is False and canary["api_key_read_allowed"] is False and canary["cloud_image_write_allowed"] is False and summary["api_key_reads"] == 0, "no external API, API-key read, or cloud image write is allowed or recorded")
    check("allowed_git_paths_only", lambda: changed_paths() <= ALLOWED_CHANGED_PATHS, "all target-branch changes are restricted to approved sidecars, scripts, tests, report, and handoff")
    check("no_images_or_pdfs_in_git", lambda: not any(Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".pdf"} for path in changed_paths()), "no image or PDF path is changed")
    check("no_paper_or_training_modification", lambda: summary["paper_modifications"] == handoff["paper_modifications"] == 0 and not any("paper_draft/" in path or path.endswith(".tex") or path.startswith("train") for path in changed_paths()), "no paper or training-code path is changed")
    check("overlay_schema", lambda: overlay["schema_version"] == "canondressgs.subject00.1349_final_human_selection_overlay.v1" and overlay["reviewer"] == "USER_AND_GPT_MANUAL_REVIEW" and overlay["automatic_acceptance"] is False, "overlay is a separate, manually reviewed, non-accepting sidecar")
    check("failure_interpretation_boundary", lambda: summary["o03_systematic_failure_classification"] == "O03_HOOD_REMOVAL_FAILURE_SYSTEMATIC_IN_EXISTING_CANDIDATE_SET" and summary["o03_selected_cell_count"] == 2 and summary["o03_missing_cell_count"] == 6, "O03 finding is limited to the existing image-generation candidate set")
    check("formal_base_pending", lambda: summary["formal_base_status"] == handoff["formal_base_status"] == "PENDING" and summary["subject00_paper_positive_claim"] == 0, "Formal Base remains pending and no positive paper claim is made")
    check("final_classification", lambda: summary["final_classification"] == handoff["final_classification"] == FINAL_CLASSIFICATION, "success classification matches the contract")
    check("next_task_unique", lambda: summary["next_task"] == handoff["next_task"] == NEXT_TASK, "the only next task is explicit user authorization of four canary requests")
    check("paper_final_false", lambda: old_review["paper_final"] is False and all(value["paper_final"] is False for value in (overlay, candidates, selected_registry, missing_registry, summary, canary, handoff)), "PAPER_FINAL remains false in every applicable registry")

    write_results(checks)
    failed = [item for item in checks if not item["passed"]]
    if failed:
        for item in failed:
            print(f"FAIL {item['name']}: {item['detail']}")
        return 1
    print(f"PASS: {len(checks)} independent checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
