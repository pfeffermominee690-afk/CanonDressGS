#!/usr/bin/env python3
"""Freeze the manual Subject00 1349 selection and draft the O03 canary."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
HANDOFF_ROOT = ROOT / "project_control_handoff"
SOURCE_WORKTREE = Path(
    r"E:\model_train\canondressgs_subject00_1349_cohort_correction_human_review_prep"
)
SOURCE_BRANCH = "research/subject00-1349-cohort-correction-human-review-prep-20260726"
SOURCE_HEAD = "f6f634c029abc72191ce1edbc99e56fe781f0bfd"
BRANCH = "research/subject00-1349-human-selection-o03-canary-prep-20260726"
WORKTREE = Path(r"E:\model_train\canondressgs_subject00_1349_human_selection_o03_canary_prep")

ATTEMPTS_ROOT = Path(
    r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-GENERATION-001"
)
ATTEMPTS = {
    "attempt_001": ATTEMPTS_ROOT / "attempt_001",
    "attempt_002": ATTEMPTS_ROOT / "attempt_002_portrait_canary",
    "attempt_003": ATTEMPTS_ROOT / "attempt_003_portrait_canary_v2_registered_outpaint",
}
AUDIT_ROOT = ATTEMPTS_ROOT / "attempt_001_native_landscape_registration_audit"
EXTERNAL_REVIEW_MANIFEST = (
    AUDIT_ROOT / "08_corrected_1349_human_review" / "corrected_1349x1166_human_review_manifest.json"
)
IMMUTABILITY_BASELINE = AUDIT_ROOT / "00_provenance" / "attempt_immutability_baseline.json"

TASK_ID = "AAAI27-SUBJECT00-1349-HUMAN-SELECTION-AND-O03-CANARY-PREP-001"
REVIEWER = "USER_AND_GPT_MANUAL_REVIEW"
FAILURE_TAG = "SOURCE_GARMENT_HOOD_RESIDUAL_IN_O03_SUIT"
FINAL_CLASSIFICATION = "SUBJECT00_1349_HUMAN_SELECTION_FROZEN_O03_CANARY_PENDING_AUTHORIZATION"
NEXT_TASK = "USER_AUTHORIZE_SUBJECT00_O03_HOOD_REMOVAL_CANARY_4_REQUESTS"
CANARY_NAME = "SUBJECT00_O03_HOOD_REMOVAL_TARGETED_CANARY"
CANARY_NAMESPACE = "attempt_004_o03_hood_removal_targeted_canary"
EXPECTED_RESOLUTION = {"height": 1166, "orientation": "landscape", "width": 1349}

SOURCE_CANDIDATES = RISK / "corrected_attempt001_1349x1166_candidate_registry.json"
SOURCE_COVERAGE = RISK / "corrected_attempt001_1349x1166_cell_coverage.json"
SOURCE_REVIEW = RISK / "corrected_1349x1166_human_review_manifest.json"
SOURCE_PACK = RISK / "subject00_attempt001_1349_high_res_review_pack_registry_20260726.json"

OVERLAY_PATH = RISK / "subject00_1349_final_human_selection_overlay_20260726.json"
REPORT_PATH = RISK / "SUBJECT00_1349_FINAL_HUMAN_SELECTION_REPORT_20260726.md"
CANDIDATE_PATH = RISK / "subject00_1349_candidate_decision_registry_20260726.json"
SELECTED_PATH = RISK / "subject00_1349_selected_cell_registry_20260726.json"
MISSING_PATH = RISK / "subject00_1349_final_missing_cell_registry_20260726.json"
SUMMARY_PATH = RISK / "subject00_1349_human_selection_final_summary_20260726.json"
CANARY_CONTRACT_PATH = RISK / "SUBJECT00_O03_HOOD_REMOVAL_CANARY_CONTRACT_20260726.md"
CANARY_MANIFEST_PATH = RISK / "subject00_o03_hood_removal_canary_manifest_draft_20260726.json"
TESTS_PATH = RISK / "subject00_1349_human_selection_o03_canary_tests_20260726.json"
HANDOFF_PATH = HANDOFF_ROOT / "subject00_1349_human_selection_o03_canary_prep_handoff_20260726.json"

SELECTED = (
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
)

FAILED = (
    "subject00_O03_slot00_cand00",
    "subject00_O03_slot00_cand01",
    "subject00_O03_slot01_cand00",
    "subject00_O03_slot01_cand01",
    "subject00_O03_slot02_cand01",
    "subject00_O03_slot03_cand00",
    "subject00_O03_slot04_cand00",
    "subject00_O03_slot05_cand00",
)

MISSING = (
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
)

HUMAN_VISUAL_MISSING = {
    "O03/slot_00",
    "O03/slot_01",
    "O03/slot_04",
    "O03/slot_05",
}

CANARY_CELLS = (
    {"camera": "cam17", "orientation": "front", "slot": "slot_00"},
    {"camera": "cam11", "orientation": "right", "slot": "slot_04"},
    {"camera": "cam02", "orientation": "back-left", "slot": "slot_05"},
    {"camera": "cam05", "orientation": "back", "slot": "slot_07"},
)

PROMPT_CLAUSES = (
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
)

ACCEPTANCE_GATES = (
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
)


def canonical_sha256(value: Any) -> str:
    if isinstance(value, dict):
        value = dict(value)
        value.pop("content_sha256", None)
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8", newline="\n")


def run(*command: str, cwd: Path = ROOT) -> str:
    return subprocess.run(
        list(command), cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def current_inventory(root: Path) -> dict[str, tuple[int, str]]:
    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, file_sha256(path))
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def frozen_inventory(snapshot: dict[str, Any]) -> dict[str, tuple[int, str]]:
    return {item["relative_path"]: (item["bytes"], item["sha256"]) for item in snapshot["files"]}


def validate_gate() -> None:
    facts = {
        "source_branch": run("git", "branch", "--show-current", cwd=SOURCE_WORKTREE),
        "source_head": run("git", "rev-parse", "HEAD", cwd=SOURCE_WORKTREE),
        "source_status": run("git", "status", "--short", cwd=SOURCE_WORKTREE),
        "target_branch": run("git", "branch", "--show-current"),
    }
    expected = {
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "source_status": "",
        "target_branch": BRANCH,
    }
    if facts != expected:
        raise RuntimeError(f"source gate mismatch: {facts!r}")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", SOURCE_HEAD, "HEAD"], cwd=ROOT
    )
    if ancestor.returncode:
        raise RuntimeError("source HEAD is not an ancestor of target HEAD")
    baseline = load_json(IMMUTABILITY_BASELINE)
    for name, root in ATTEMPTS.items():
        if current_inventory(root) != frozen_inventory(baseline[name]):
            raise RuntimeError(f"{name} changed relative to the frozen baseline")


def decision_record(
    source: dict[str, Any], coverage: dict[str, Any], selected: bool, failed: bool
) -> dict[str, Any]:
    if failed:
        decision = "FAIL"
        visual_comparison = "HOOD_RESIDUAL_CONFIRMED_BY_HIGH_RES_SIDE_BY_SIDE_REVIEW"
        identity_confidence = "NOT_DISPOSITIVE_AFTER_GARMENT_FAILURE"
        garment_correctness = "FAIL"
        selection_rationale = "Not selectable because the source hoodie remains in the O03 suit output."
        failure_tags = [FAILURE_TAG]
    elif selected:
        decision = "PASS_CANDIDATE"
        visual_comparison = "PREFERRED_WITHIN_CELL_BY_HIGH_RES_SIDE_BY_SIDE_REVIEW"
        identity_confidence = "PASS"
        garment_correctness = "PASS"
        selection_rationale = (
            "Preferred within its cell by the user/GPT high-resolution comparison; machine "
            "registration is supporting evidence only."
        )
        failure_tags = []
    else:
        decision = "PASS_NOT_SELECTED"
        visual_comparison = "VISUALLY_PASSING_ALTERNATIVE_NOT_PREFERRED_WITHIN_CELL"
        identity_confidence = "PASS"
        garment_correctness = "PASS"
        selection_rationale = (
            "Visually passed, but another candidate in the same cell was preferred; non-selection "
            "is not a failure."
        )
        failure_tags = []
    value: dict[str, Any] = {
        "accepted": False,
        "actual_resolution": source["actual_resolution"],
        "camera": source["camera"],
        "cell_key": source["cell_key"],
        "failure_tags": failure_tags,
        "final_pass_candidate": not failed,
        "garment": source["garment"],
        "garment_correctness": garment_correctness,
        "human_visual_decision": decision,
        "identity_confidence": identity_confidence,
        "machine_registration_classification": source["primary_machine_classification"],
        "machine_registration_pass": source["machine_pass"],
        "orientation": coverage["orientation"],
        "output_path": source["output_path"],
        "output_sha256": source["output_sha256"],
        "registration_metrics": source["registration_metrics"],
        "request_id": source["request_id"],
        "review_asset_count": source["review_asset_count"],
        "review_assets": source["review_assets"],
        "reviewer": REVIEWER,
        "selected_for_cell": selected,
        "selection_rationale": selection_rationale,
        "slot": source["slot"],
        "source_path": source["source_path"],
        "source_sha256": source["source_sha256"],
        "teacher_target": False,
        "visual_comparison": visual_comparison,
    }
    if source["request_id"] == "subject00_O03_slot02_cand01":
        value["decision_revision"] = {
            "final_high_res_cell_review_decision": "FAIL",
            "previous_preliminary_decision": "PASS_CANDIDATE",
            "revision_reason": "DARK_BLUE_SOURCE_HOOD_REVEALED_BY_HIGH_RES_SIDE_BY_SIDE_REVIEW",
        }
    return value


def candidate_status(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if not record:
        return None
    return {
        "actual_resolution": record["actual_resolution"],
        "machine_registration_status": record["primary_machine_classification"],
        "request_id": record["request_id"],
        "review_eligible": record["review_eligible"],
    }


def build_missing_record(cell: dict[str, Any]) -> dict[str, Any]:
    visual_missing = cell["cell_key"] in HUMAN_VISUAL_MISSING
    if visual_missing:
        reason = "ALL_EXACT_COHORT_CANDIDATES_FAILED_HOOD_REMOVAL_VISUAL_GATE"
        exact_status = "EXACT_1349x1166_MACHINE_PASS_EXISTS_BUT_NO_HUMAN_VISUAL_PASS"
        human_status = "FAIL_SOURCE_GARMENT_HOOD_RESIDUAL_IN_O03_SUIT"
    else:
        reason = "NO_REVIEW_ELIGIBLE_EXACT_1349x1166_MACHINE_PASS_CANDIDATE"
        exact_status = "NO_REVIEW_ELIGIBLE_EXACT_1349x1166_MACHINE_PASS_CANDIDATE"
        human_status = "NOT_REVIEWABLE_AS_AN_EXACT_REGISTERED_CANDIDATE"
    return {
        "camera": cell["camera"],
        "cell_key": cell["cell_key"],
        "exact_resolution_status": exact_status,
        "existing_candidate_status": [
            candidate_status(cell.get("cand00")),
            candidate_status(cell.get("cand01")),
        ],
        "garment": cell["garment"],
        "human_status": human_status,
        "machine_registration_status": {
            "exact_cohort_machine_pass_candidate_count": cell["pass_candidate_count_1349x1166"],
            "machine_registration_does_not_imply_visual_pass": True,
        },
        "orientation": cell["orientation"],
        "reason": reason,
        "rerun_authorized": False,
        "rerun_required": True,
        "slot": cell["slot"],
    }


def build_canary_requests(all_records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    requests = []
    for cell in CANARY_CELLS:
        slot_number = cell["slot"][-2:]
        source_request_id = f"subject00_O03_slot{slot_number}_cand00"
        source = all_records[source_request_id]
        source_request_path = (
            ATTEMPTS["attempt_001"]
            / "03_generation_requests"
            / "requests"
            / f"{source_request_id}.json"
        )
        source_request = load_json(source_request_path)
        local_path = Path(source_request["managed_tool_local_input_path"])
        source_sha = source_request["source_input_sha256"]["identity_condition_rgb"]
        if source["source_sha256"] != source_sha or file_sha256(local_path) != source_sha:
            raise RuntimeError(f"source condition SHA conflict for {source_request_id}")
        requests.append(
            {
                "camera": cell["camera"],
                "expected_resolution": EXPECTED_RESOLUTION,
                "garment": "O03",
                "local_source_condition_path": str(local_path),
                "orientation": cell["orientation"],
                "request_id": f"subject00_O03_slot{slot_number}_canary_attempt004_cand00",
                "slot": cell["slot"],
                "source_condition_path": source_request["source_input_paths"]["identity_condition_rgb"],
                "source_condition_sha256": source_sha,
                "source_mask_path": source_request["source_input_paths"]["identity_condition_mask"],
                "source_mask_sha256": source_request["source_input_sha256"]["identity_condition_mask"],
                "source_request_id": source_request_id,
                "source_request_path": str(source_request_path),
                "source_request_sha256": file_sha256(source_request_path),
            }
        )
    return requests


def markdown_list(values: list[str] | tuple[str, ...]) -> str:
    return "\n".join(f"- `{value}`" for value in values)


def main() -> int:
    validate_gate()
    candidates = load_json(SOURCE_CANDIDATES)
    coverage = load_json(SOURCE_COVERAGE)
    review = load_json(SOURCE_REVIEW)
    pack = load_json(SOURCE_PACK)
    review_candidates = {item["request_id"]: item for item in candidates["review_candidates"]}
    all_records = {item["request_id"]: item for item in candidates["records"]}
    review_records = {item["request_id"]: item for item in review["records"]}
    coverage_by_cell = {item["cell_key"]: item for item in coverage["records"]}

    if set(review_candidates) != set(review_records) or len(review_candidates) != 31:
        raise RuntimeError("31-candidate evidence binding failed")
    if not set(SELECTED).isdisjoint(FAILED):
        raise RuntimeError("selected and failed request sets overlap")
    pass_not_selected = tuple(sorted(set(review_candidates) - set(SELECTED) - set(FAILED)))
    if len(SELECTED) != 14 or len(FAILED) != 8 or len(pass_not_selected) != 9:
        raise RuntimeError("manual decision partition is not 14 selected / 8 fail / 9 pass-not-selected")

    decisions = []
    for request_id in sorted(review_candidates):
        source = review_candidates[request_id]
        decisions.append(
            decision_record(
                source,
                coverage_by_cell[source["cell_key"]],
                request_id in SELECTED,
                request_id in FAILED,
            )
        )
    decisions_by_id = {item["request_id"]: item for item in decisions}
    selected_records = [decisions_by_id[request_id] for request_id in SELECTED]
    missing_records = [build_missing_record(coverage_by_cell[cell]) for cell in MISSING]
    canary_requests = build_canary_requests(all_records)

    old_repo_sha = file_sha256(SOURCE_REVIEW)
    old_external_sha = file_sha256(EXTERNAL_REVIEW_MANIFEST)
    external_review = load_json(EXTERNAL_REVIEW_MANIFEST)
    if review["content_sha256"] != external_review["content_sha256"]:
        raise RuntimeError("old review manifest copies disagree in canonical content")

    overlay = {
        "accepted_count": 0,
        "automatic_acceptance": False,
        "candidate_decisions": decisions,
        "fail_count": 8,
        "machine_registration_pass_is_not_human_visual_pass": True,
        "old_review_manifest_canonical_content_sha256": review["content_sha256"],
        "old_review_manifest_path": str(EXTERNAL_REVIEW_MANIFEST),
        "old_review_manifest_repo_copy_path": str(SOURCE_REVIEW),
        "old_review_manifest_repo_copy_sha256": old_repo_sha,
        "old_review_manifest_sha256": old_external_sha,
        "old_review_manifest_was_modified": False,
        "repo_and_external_canonical_content_match": True,
        "paper_final": False,
        "pass_candidate_count": 23,
        "pass_not_selected_count": 9,
        "reviewed_candidate_count": 31,
        "reviewer": REVIEWER,
        "revised_decisions": ["subject00_O03_slot02_cand01"],
        "schema_version": "canondressgs.subject00.1349_final_human_selection_overlay.v1",
        "selected_for_cell_count": 14,
        "task_id": TASK_ID,
        "teacher_target_count": 0,
    }
    write_json(OVERLAY_PATH, overlay)

    candidate_registry = {
        "accepted_count": 0,
        "decision_distribution": dict(
            sorted(Counter(item["human_visual_decision"] for item in decisions).items())
        ),
        "fail_count": 8,
        "paper_final": False,
        "pass_candidate_count": 23,
        "pass_not_selected_count": 9,
        "record_count": 31,
        "records": decisions,
        "reviewer": REVIEWER,
        "schema_version": "canondressgs.subject00.1349_candidate_decision_registry.v1",
        "selected_for_cell_count": 14,
        "source_candidate_registry_path": str(SOURCE_CANDIDATES),
        "source_candidate_registry_sha256": file_sha256(SOURCE_CANDIDATES),
        "task_id": TASK_ID,
        "teacher_target_count": 0,
        "uncertain_count": 0,
    }
    write_json(CANDIDATE_PATH, candidate_registry)

    selected_registry = {
        "accepted_count": 0,
        "paper_final": False,
        "record_count": 14,
        "records": selected_records,
        "reviewer": REVIEWER,
        "schema_version": "canondressgs.subject00.1349_selected_cell_registry.v1",
        "selection_counts_by_garment": dict(
            sorted(Counter(item["garment"] for item in selected_records).items())
        ),
        "task_id": TASK_ID,
        "teacher_target_count": 0,
        "unique_selection_per_cell": True,
    }
    write_json(SELECTED_PATH, selected_registry)

    missing_registry = {
        "missing_cell_count": 10,
        "paper_final": False,
        "records": missing_records,
        "rerun_authorized_count": 0,
        "rerun_required_count": 10,
        "schema_version": "canondressgs.subject00.1349_final_missing_cell_registry.v1",
        "task_id": TASK_ID,
    }
    write_json(MISSING_PATH, missing_registry)

    canary_manifest = {
        "acceptance_gates": list(ACCEPTANCE_GATES),
        "all_four_requests_must_pass": True,
        "attempt_namespace": CANARY_NAMESPACE,
        "automatic_acceptance": False,
        "api_key_read_allowed": False,
        "canary_name": CANARY_NAME,
        "cloud_image_write_allowed": False,
        "expected_resolution": EXPECTED_RESOLUTION,
        "expected_output_format": "PNG",
        "external_api_allowed": False,
        "generation_authorized": False,
        "generation_call_count": 0,
        "native_resolution_required": True,
        "paper_final": False,
        "png_parse_required": True,
        "postprocessing_policy": {
            "background_outpainting": False,
            "crop": False,
            "padding": False,
            "person_scaling": False,
            "portrait_canvas": False,
            "re_encoding_repair": False,
            "resize": False,
        },
        "prompt_contract": list(PROMPT_CLAUSES),
        "request_count": 4,
        "requests": canary_requests,
        "retry_policy": {
            "automatic_retry": False,
            "candidate_count_per_cell": 1,
            "retry_count": 0,
        },
        "schema_version": "canondressgs.subject00.o03_hood_removal_canary_manifest_draft.v1",
        "task_id": TASK_ID,
    }
    write_json(CANARY_MANIFEST_PATH, canary_manifest)

    canary_contract = f"""# Subject00 O03 Hood-Removal Canary Contract

- Task: `{TASK_ID}`
- Canary: `{CANARY_NAME}`
- Attempt namespace: `{CANARY_NAMESPACE}`
- Generation authorized: `false`
- Request count: `4`
- Expected output: native `1349x1166` landscape PNG

## Purpose

Test whether a targeted edit can replace the complete O01 hoodie with the frozen O03 formal suit while preserving the registered Subject00 condition. This is an image-generation candidate canary, not a Base Avatar representation test.

## Requests

{markdown_list([item['request_id'] for item in canary_requests])}

The fixed cells are O03/slot00/cam17/front, O03/slot04/cam11/right, O03/slot05/cam02/back-left, and O03/slot07/cam05/back. Each request is bound to the original registered condition path and SHA in `{CANARY_MANIFEST_PATH.name}`.

## Prompt Contract

{markdown_list(PROMPT_CLAUSES)}

The execution prompt must not request face beautification, a different hairstyle, a camera change, subject movement, zoom, or background reconstruction.

## Execution Contract

- Exactly four generation calls, one per request, only after explicit user authorization.
- Zero retries and one candidate per cell; no generate-many-then-select behavior.
- No resize, crop, padding, re-encoding repair, portrait canvas, output postprocessing, or background outpainting.
- No external API, API-key read, cloud image write, automatic acceptance, accepted promotion, or Teacher-target promotion.
- Output must parse as PNG at exactly 1349x1166.

## Human Gate

All four outputs must pass every gate: {', '.join(ACCEPTANCE_GATES)}. If fewer than 4/4 pass, the remaining six O03 missing cells must not be batch-generated.

## Status

`generation_authorized = false`. This task drafts and freezes the contract only; it performs no image generation.
"""
    write_text(CANARY_CONTRACT_PATH, canary_contract)

    summary = {
        "accepted_count": 0,
        "api_key_reads": 0,
        "canary_expected_resolution": EXPECTED_RESOLUTION,
        "canary_generated_count": 0,
        "canary_generation_authorized": False,
        "canary_request_count": 4,
        "canary_request_ids": [item["request_id"] for item in canary_requests],
        "failed_requests": list(FAILED),
        "final_classification": FINAL_CLASSIFICATION,
        "final_fail_count": 8,
        "final_pass_candidate_count": 23,
        "formal_base_status": "PENDING",
        "missing_cell_count": 10,
        "missing_cells": list(MISSING),
        "new_branch": BRANCH,
        "new_generation_calls": 0,
        "next_task": NEXT_TASK,
        "o03_missing_cell_count": 6,
        "o03_selected_cell_count": 2,
        "o03_systematic_failure_classification": "O03_HOOD_REMOVAL_FAILURE_SYSTEMATIC_IN_EXISTING_CANDIDATE_SET",
        "paper_final": False,
        "paper_modifications": 0,
        "pass_not_selected_count": 9,
        "reviewed_candidate_count": 31,
        "selected_for_cell_count": 14,
        "selected_requests": list(SELECTED),
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "source_review_manifest_sha256": old_external_sha,
        "subject00_paper_positive_claim": 0,
        "targeted_rerun_generated_count": 0,
        "task_id": TASK_ID,
        "teacher_target_count": 0,
        "worktree": str(WORKTREE),
    }
    write_json(SUMMARY_PATH, summary)

    report = f"""# Subject00 1349 Final Human Selection Report

- Task: `{TASK_ID}`
- Reviewer: `{REVIEWER}`
- Source: `{SOURCE_BRANCH}` at `{SOURCE_HEAD}`
- Old review manifest evidence SHA256: `{old_external_sha}`
- Repository worktree copy SHA256: `{old_repo_sha}`
- Canonical content SHA256: `{review['content_sha256']}`
- Final classification: `{FINAL_CLASSIFICATION}`

## Manual Decision Freeze

The 31 high-resolution candidates are frozen as 23 visual passes and 8 failures, with no uncertain decisions. Machine registration PASS is supporting geometric evidence and is not equivalent to human Visual PASS.

- Selected for cell: 14
- Pass but not selected: 9
- Failed: 8
- Accepted: 0
- Teacher target: 0

### Selected Requests

{markdown_list(SELECTED)}

### Failed Requests

{markdown_list(FAILED)}

All eight failures use `{FAILURE_TAG}`. `subject00_O03_slot02_cand01` is revised from preliminary `PASS_CANDIDATE` to final `FAIL` because high-resolution side-by-side review revealed the dark-blue source hood.

## Missing Cells

{markdown_list(MISSING)}

All ten cells require a rerun, but rerun authorization remains false.

## O03 Finding

O03 hoodie-removal failure is systematic in the existing candidate set: 2/8 cells are selected, 6/8 are missing, and 8 candidate images have explicit hood contamination. Managed Image Edit frequently changes the suit torso while retaining the source O01 hood around the head. This is an image-generation candidate failure, not evidence that O03 is unrepresentable, that MMLP-Human cannot represent suits, or that an O03 Teacher must fail.

## Canary

The four-request `{CANARY_NAME}` contract is drafted but not authorized. It covers front, right, back-left, and back views at native 1349x1166. No image generation, retry, external API call, API-key read, accepted promotion, Teacher-target promotion, Formal Base training, Teacher optimization, or paper modification occurred.

## Next Task

`{NEXT_TASK}`
"""
    write_text(REPORT_PATH, report)

    handoff = {
        "accepted_count": 0,
        "artifact_paths": {
            "candidate_registry": str(CANDIDATE_PATH.relative_to(ROOT)),
            "canary_contract": str(CANARY_CONTRACT_PATH.relative_to(ROOT)),
            "canary_manifest": str(CANARY_MANIFEST_PATH.relative_to(ROOT)),
            "final_report": str(REPORT_PATH.relative_to(ROOT)),
            "final_summary": str(SUMMARY_PATH.relative_to(ROOT)),
            "missing_registry": str(MISSING_PATH.relative_to(ROOT)),
            "overlay": str(OVERLAY_PATH.relative_to(ROOT)),
            "selected_registry": str(SELECTED_PATH.relative_to(ROOT)),
            "tests": str(TESTS_PATH.relative_to(ROOT)),
        },
        "canary_generation_authorized": False,
        "canary_request_count": 4,
        "execution_status": "HUMAN_SELECTION_FROZEN_CANARY_DRAFTED_WAITING_FOR_AUTHORIZATION",
        "final_classification": FINAL_CLASSIFICATION,
        "formal_base_status": "PENDING",
        "missing_cell_count": 10,
        "new_branch": BRANCH,
        "new_generation_calls": 0,
        "next_task": NEXT_TASK,
        "paper_final": False,
        "paper_modifications": 0,
        "reviewed_candidate_count": 31,
        "schema_version": "canondressgs.subject00.1349_human_selection_o03_canary_prep_handoff.v1",
        "selected_for_cell_count": 14,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "targeted_rerun_generated_count": 0,
        "task_id": TASK_ID,
        "teacher_target_count": 0,
        "worktree": str(WORKTREE),
    }
    write_json(HANDOFF_PATH, handoff)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
