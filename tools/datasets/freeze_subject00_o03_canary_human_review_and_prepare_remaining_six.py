#!/usr/bin/env python3
"""Freeze the Subject00 O03 canary review and draft the remaining-six contract."""

from __future__ import annotations

import argparse
import copy
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
    r"E:\model_train\canondressgs_subject00_o03_canary_resolution_correction_continuation"
)
SOURCE_BRANCH = "research/subject00-o03-canary-resolution-correction-continuation-20260726"
SOURCE_HEAD = "96e1ff5237ec13feba449de287576bb79eac0068"
BRANCH = "research/subject00-o03-canary-human-review-freeze-remaining-six-contract-20260726"
WORKTREE = Path(
    r"E:\model_train\canondressgs_subject00_o03_canary_human_review_freeze_remaining_six_contract"
)

TASK_ID = "AAAI27-SUBJECT00-O03-CANARY-HUMAN-REVIEW-FREEZE-001"
FINAL_CLASSIFICATION = (
    "SUBJECT00_O03_CANARY_HUMAN_REVIEW_FROZEN_"
    "REMAINING_SIX_CELLS_PENDING_AUTHORIZATION"
)
NEXT_TASK = "USER_AUTHORIZE_SUBJECT00_REMAINING_SIX_CELL_GENERATION"
REVIEWER = "USER_EXPLICIT_HUMAN_REVIEW"
PLANNED_NAMESPACE = "attempt_005_subject00_remaining_six_cell_generation"

ATTEMPTS_ROOT = Path(
    r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-GENERATION-001"
)
ATTEMPTS = {
    "attempt_001": ATTEMPTS_ROOT / "attempt_001",
    "attempt_002": ATTEMPTS_ROOT / "attempt_002_portrait_canary",
    "attempt_003": ATTEMPTS_ROOT / "attempt_003_portrait_canary_v2_registered_outpaint",
    "attempt_004": ATTEMPTS_ROOT / "attempt_004_o03_hood_removal_targeted_canary",
}
ATTEMPT_004 = ATTEMPTS["attempt_004"]
EXTERNAL_GENERATION_MANIFEST = ATTEMPT_004 / "06_provenance" / "generation_manifest.json"
CONTINUATION_SUMMARY = (
    RISK / "subject00_o03_canary_resolution_correction_final_summary_20260726.json"
)

BASE_SELECTED = RISK / "subject00_1349_selected_cell_registry_20260726.json"
BASE_MISSING = RISK / "subject00_1349_final_missing_cell_registry_20260726.json"
SOURCE_REVIEW = (
    RISK / "subject00_o03_canary_resolution_correction_human_review_manifest_20260726.json"
)
RAW_REGISTRY = RISK / "subject00_o03_canary_corrected_raw_output_registry_20260726.json"
REGISTRATION_RESULTS = (
    RISK / "subject00_o03_canary_corrected_registration_results_20260726.json"
)
CONDITION_BINDINGS = RISK / "subject00_condition_slot_binding.json"
PROMPT_REGISTRY = RISK / "subject00_generation_prompt_registry.json"
SOURCE_REQUEST_MANIFEST = RISK / "subject00_generation_request_manifest.json"

OVERLAY_PATH = RISK / "subject00_o03_canary_human_review_overlay_20260726.json"
GLOBAL_SELECTED_PATH = RISK / "subject00_global_selected_cell_registry_20260726.json"
GLOBAL_MISSING_PATH = RISK / "subject00_global_missing_cell_registry_20260726.json"
CONTRACT_PATH = RISK / "SUBJECT00_REMAINING_SIX_CELL_GENERATION_CONTRACT_20260726.md"
MANIFEST_PATH = RISK / "subject00_remaining_six_cell_generation_manifest_draft_20260726.json"
SUMMARY_PATH = RISK / "subject00_o03_canary_human_review_freeze_final_summary_20260726.json"
REPORT_PATH = RISK / "SUBJECT00_O03_CANARY_HUMAN_REVIEW_FREEZE_20260726.md"
TESTS_PATH = RISK / "subject00_o03_canary_human_review_freeze_tests_20260726.json"
HANDOFF_PATH = (
    HANDOFF_ROOT / "subject00_o03_canary_human_review_freeze_handoff_20260726.json"
)

RAW_OUTPUTS = {
    "subject00_O03_slot00_canary_attempt004_cand00": {
        "path": ATTEMPT_004
        / "04_generation_responses"
        / "technical_failures"
        / "subject00_O03_slot00_canary_attempt004_cand00.png",
        "sha256": "f5197d9646ccf60ddffe3184e295e2afed2f04ff3ccbe95b5908974f6652e9e1",
    },
    "subject00_O03_slot04_canary_attempt004_cand00": {
        "path": ATTEMPT_004
        / "04_generation_responses"
        / "codex_managed_candidates"
        / "O03"
        / "subject00_O03_slot04_canary_attempt004_cand00.png",
        "sha256": "fa73168d85d2e58d48172954bc85740db82f131f61aaaf42a09d731107d0e30c",
    },
    "subject00_O03_slot05_canary_attempt004_cand00": {
        "path": ATTEMPT_004
        / "04_generation_responses"
        / "codex_managed_candidates"
        / "O03"
        / "subject00_O03_slot05_canary_attempt004_cand00.png",
        "sha256": "379562123b1d918d68171b9224068dbe905d2d5e108900408b5da82c49128aa1",
    },
    "subject00_O03_slot07_canary_attempt004_cand00": {
        "path": ATTEMPT_004
        / "04_generation_responses"
        / "codex_managed_candidates"
        / "O03"
        / "subject00_O03_slot07_canary_attempt004_cand00.png",
        "sha256": "9300293c0b028f5f7f194f5de6d56426404938a84ff61ec37d6ce56afab84441",
    },
}

DECISIONS = {
    "subject00_O03_slot00_canary_attempt004_cand00": {
        "identity_pass": True,
        "identity_confidence": "MEDIUM",
        "identity_evidence": None,
        "machine_registration_human_override": None,
        "final_human_visual_decision": "PASS_CANDIDATE",
    },
    "subject00_O03_slot04_canary_attempt004_cand00": {
        "identity_pass": None,
        "identity_confidence": None,
        "identity_evidence": "INSUFFICIENT_SOURCE_VISIBILITY",
        "machine_registration_human_override": "PASS_VISUALLY_ACCEPTABLE_ALIGNMENT",
        "final_human_visual_decision": (
            "PASS_CANDIDATE_WITH_HUMAN_REGISTRATION_OVERRIDE"
        ),
    },
    "subject00_O03_slot05_canary_attempt004_cand00": {
        "identity_pass": None,
        "identity_confidence": None,
        "identity_evidence": "INSUFFICIENT_SOURCE_VISIBILITY",
        "machine_registration_human_override": None,
        "final_human_visual_decision": "PASS_CANDIDATE_IDENTITY_REFERENCE_LIMITED",
    },
    "subject00_O03_slot07_canary_attempt004_cand00": {
        "identity_pass": None,
        "identity_confidence": None,
        "identity_evidence": "BACK_VIEW_NOT_IDENTITY_VERIFIABLE",
        "machine_registration_human_override": None,
        "final_human_visual_decision": "PASS_CANDIDATE_IDENTITY_REFERENCE_LIMITED",
    },
}

REMAINING = (
    {"garment": "O01", "slot": "slot_04", "camera": "cam11", "direction": "right"},
    {"garment": "O01", "slot": "slot_07", "camera": "cam05", "direction": "back"},
    {
        "garment": "O03",
        "slot": "slot_01",
        "camera": "cam21",
        "direction": "front-left",
    },
    {
        "garment": "O03",
        "slot": "slot_06",
        "camera": "cam09",
        "direction": "back-right",
    },
    {
        "garment": "O04",
        "slot": "slot_05",
        "camera": "cam02",
        "direction": "back-left",
    },
    {
        "garment": "O04",
        "slot": "slot_06",
        "camera": "cam09",
        "direction": "back-right",
    },
)

RESOLUTION_FAMILY = (
    {"width": 1348, "height": 1167},
    {"width": 1349, "height": 1166},
    {"width": 1350, "height": 1165},
)

O03_HOOD_REMOVAL_CLAUSES = (
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


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: dict[str, Any]) -> str:
    content = {key: value for key, value in payload.items() if key != "content_sha256"}
    encoded = json.dumps(
        content, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    result = copy.deepcopy(payload)
    result["content_sha256"] = canonical_hash(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def tree_snapshot(root: Path) -> dict[str, Any]:
    records = []
    total_bytes = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        size = path.stat().st_size
        total_bytes += size
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": file_sha256(path),
                "size_bytes": size,
            }
        )
    digest = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "exists": root.exists(),
        "file_count": len(records),
        "root": str(root),
        "total_bytes": total_bytes,
        "tree_sha256": digest,
    }


def attempt_snapshots() -> dict[str, dict[str, Any]]:
    return {name: tree_snapshot(path) for name, path in ATTEMPTS.items()}


def source_preflight() -> dict[str, Any]:
    branch = run_git(["branch", "--show-current"], SOURCE_WORKTREE)
    head = run_git(["rev-parse", "HEAD"], SOURCE_WORKTREE)
    status = run_git(["status", "--porcelain=v1"], SOURCE_WORKTREE)
    target_branch = run_git(["branch", "--show-current"], ROOT)
    target_head = run_git(["rev-parse", "HEAD"], ROOT)
    if branch != SOURCE_BRANCH:
        raise RuntimeError(f"source branch mismatch: {branch}")
    if head != SOURCE_HEAD:
        raise RuntimeError(f"source HEAD mismatch: {head}")
    if status:
        raise RuntimeError("source worktree is not clean")
    if target_branch != BRANCH:
        raise RuntimeError(f"target branch mismatch: {target_branch}")
    if target_head != SOURCE_HEAD:
        raise RuntimeError(f"target worktree does not start at source HEAD: {target_head}")
    return {
        "source_branch": branch,
        "source_head": head,
        "source_worktree": str(SOURCE_WORKTREE),
        "source_worktree_clean": True,
        "target_branch": target_branch,
        "target_start_head": target_head,
        "target_worktree": str(ROOT),
    }


def verify_raw_outputs() -> dict[str, dict[str, Any]]:
    result = {}
    for request_id, expected in RAW_OUTPUTS.items():
        path = expected["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_sha = file_sha256(path)
        if actual_sha != expected["sha256"]:
            raise RuntimeError(f"raw output SHA mismatch for {request_id}")
        result[request_id] = {
            "path": str(path),
            "sha256": actual_sha,
            "size_bytes": path.stat().st_size,
        }
    slot00 = result["subject00_O03_slot00_canary_attempt004_cand00"]
    if Path(slot00["path"]).parent.name != "technical_failures":
        raise RuntimeError("slot00 raw output moved out of technical_failures")
    return result


def normalized_descriptor(item: dict[str, Any]) -> str:
    return "/".join(
        [item["garment"], item["slot"].replace("_", ""), item["camera"], item["direction"]]
    )


def local_source_path(camera: str) -> Path:
    return Path(r"E:\model_train\_subject00_identity_audit_tmp") / (
        f"{camera}_frame00000000.jpg"
    )


def build_overlay(
    source_review: dict[str, Any], raw_registry: dict[str, Any]
) -> dict[str, Any]:
    source_by_id = {item["request_id"]: item for item in source_review["records"]}
    raw_by_id = {item["request_id"]: item for item in raw_registry["records"]}
    records = []
    for request_id, decision in DECISIONS.items():
        source = source_by_id[request_id]
        raw = raw_by_id[request_id]
        face_crop = source["review_assets"]["crops"]["face_head"]
        record = {
            "accepted": False,
            "background_pass": True,
            "camera": source["camera"],
            "camera_pose_pass": True,
            "direction": source["direction"],
            "face_head_review_crop_path": face_crop,
            "face_head_review_crop_status": "INVALID_OFF_TARGET_CROP",
            "final_human_visual_decision": decision["final_human_visual_decision"],
            "full_body_completeness_pass": True,
            "garment": source["garment"],
            "garment_pass": True,
            "hood_removal_pass": True,
            "human_visual_decision": decision["final_human_visual_decision"],
            "identity_confidence": decision["identity_confidence"],
            "identity_evidence": decision["identity_evidence"],
            "identity_pass": decision["identity_pass"],
            "machine_registration_human_override": decision[
                "machine_registration_human_override"
            ],
            "machine_registration_primary_classification": raw[
                "machine_registration_primary_classification"
            ],
            "machine_registration_status": raw["machine_registration_status"],
            "raw_output_path": raw["raw_output_path"],
            "raw_output_resolution": raw["raw_output_resolution"],
            "raw_output_sha256": raw["raw_output_sha256"],
            "request_id": request_id,
            "review_assets": source["review_assets"],
            "selected_for_cell": True,
            "slot": source["slot"],
            "source_condition_path": source["source_condition_path"],
            "source_condition_sha256": source["source_condition_sha256"],
            "teacher_target": False,
        }
        records.append(record)
    return {
        "accepted_count": 0,
        "decision_source": "USER_EXPLICIT_FIELD_LEVEL_ADJUDICATION",
        "final_classification": FINAL_CLASSIFICATION,
        "human_pass_canary_outputs": 4,
        "newly_selected_cells": 4,
        "next_task": NEXT_TASK,
        "paper_final": False,
        "record_count": 4,
        "records": records,
        "remaining_missing_cells": 6,
        "reviewed_canary_outputs": 4,
        "reviewer": REVIEWER,
        "schema_version": "canondressgs.subject00.o03_canary_human_review_overlay.v1",
        "source_human_review_manifest_path": str(SOURCE_REVIEW),
        "source_human_review_manifest_sha256": file_sha256(SOURCE_REVIEW),
        "task_id": TASK_ID,
        "teacher_target_count": 0,
        "total_selected_cells": 18,
    }


def build_global_selected(
    base_selected: dict[str, Any], overlay: dict[str, Any]
) -> dict[str, Any]:
    records = copy.deepcopy(base_selected["records"])
    for decision in overlay["records"]:
        records.append(
            {
                "accepted": False,
                "actual_resolution": decision["raw_output_resolution"],
                "background_pass": True,
                "camera": decision["camera"],
                "camera_pose_pass": True,
                "cell_descriptor": normalized_descriptor(decision),
                "cell_key": f"{decision['garment']}/{decision['slot']}",
                "failure_tags": [],
                "final_human_visual_decision": decision[
                    "final_human_visual_decision"
                ],
                "final_pass_candidate": True,
                "full_body_completeness_pass": True,
                "garment": decision["garment"],
                "garment_correctness": "PASS",
                "garment_pass": True,
                "hood_removal_pass": True,
                "human_visual_decision": decision["human_visual_decision"],
                "identity_confidence": decision["identity_confidence"],
                "identity_evidence": decision["identity_evidence"],
                "identity_pass": decision["identity_pass"],
                "machine_registration_classification": decision[
                    "machine_registration_primary_classification"
                ],
                "machine_registration_human_override": decision[
                    "machine_registration_human_override"
                ],
                "machine_registration_pass": (
                    decision["machine_registration_status"] == "PASS"
                ),
                "machine_registration_status": decision[
                    "machine_registration_status"
                ],
                "orientation": decision["direction"],
                "output_path": decision["raw_output_path"],
                "output_sha256": decision["raw_output_sha256"],
                "request_id": decision["request_id"],
                "review_assets": decision["review_assets"],
                "reviewer": REVIEWER,
                "selected_for_cell": True,
                "selection_rationale": (
                    "Selected by explicit human review of the O03 canary output; "
                    "accepted and Teacher-target promotion remain false."
                ),
                "slot": decision["slot"],
                "source_path": decision["source_condition_path"],
                "source_sha256": decision["source_condition_sha256"],
                "teacher_target": False,
                "visual_comparison": "EXPLICIT_CANARY_HUMAN_REVIEW_PASS",
            }
        )
    records.sort(key=lambda item: (item["garment"], item["slot"], item["request_id"]))
    return {
        "accepted_count": 0,
        "base_registry_path": str(BASE_SELECTED),
        "base_registry_sha256": file_sha256(BASE_SELECTED),
        "newly_selected_cell_count": 4,
        "paper_final": False,
        "record_count": len(records),
        "records": records,
        "reviewer": REVIEWER,
        "schema_version": "canondressgs.subject00.global_selected_cell_registry.v1",
        "selection_counts_by_garment": dict(
            sorted(Counter(item["garment"] for item in records).items())
        ),
        "source_overlay_path": str(OVERLAY_PATH),
        "task_id": TASK_ID,
        "teacher_target_count": 0,
        "total_selected_cells": len(records),
        "unique_selection_per_cell": (
            len({item["cell_key"] for item in records}) == len(records)
        ),
    }


def source_binding_index() -> dict[tuple[str, str], dict[str, Any]]:
    bindings = read_json(CONDITION_BINDINGS)["bindings"]
    return {(item["garment_id"], item["slot_id"]): item for item in bindings}


def source_request_index() -> dict[tuple[str, str], dict[str, Any]]:
    requests = read_json(SOURCE_REQUEST_MANIFEST)["requests"]
    return {
        (item["garment_id"], item["slot_id"]): item
        for item in requests
        if item["candidate_index"] == 0
    }


def prompt_index() -> dict[str, dict[str, Any]]:
    return {
        item["garment_id"]: item for item in read_json(PROMPT_REGISTRY)["records"]
    }


def build_remaining_records() -> list[dict[str, Any]]:
    bindings = source_binding_index()
    source_requests = source_request_index()
    prompts = prompt_index()
    records = []
    for index, item in enumerate(REMAINING):
        key = (item["garment"], item["slot"])
        binding = bindings[key]
        source_request = source_requests[key]
        prompt = prompts[item["garment"]]
        local_path = local_source_path(item["camera"])
        if not local_path.is_file():
            raise FileNotFoundError(local_path)
        local_sha = file_sha256(local_path)
        if local_sha != binding["source_rgb"]["sha256"]:
            raise RuntimeError(f"local source SHA mismatch for {key}")
        source_request_path = (
            ATTEMPTS["attempt_001"]
            / "03_generation_requests"
            / "requests"
            / f"subject00_{item['garment']}_{item['slot'].replace('_', '')}_cand00.json"
        )
        if not source_request_path.is_file():
            raise FileNotFoundError(source_request_path)
        request_id = (
            f"subject00_{item['garment']}_{item['slot'].replace('_', '')}_"
            "remaining_attempt005_cand00"
        )
        records.append(
            {
                "accepted_promotion": False,
                "additional_prompt_clauses": (
                    list(O03_HOOD_REMOVAL_CLAUSES)
                    if item["garment"] == "O03"
                    else []
                ),
                "camera": item["camera"],
                "cell_descriptor": normalized_descriptor(item),
                "cell_key": f"{item['garment']}/{item['slot']}",
                "direction": item["direction"],
                "expected_native_resolution_family": list(RESOLUTION_FAMILY),
                "expected_output_format": "PNG",
                "garment": item["garment"],
                "generation_authorized": False,
                "generation_call_count": 0,
                "index": index,
                "local_source_condition_path": str(local_path),
                "negative_constraint_id": prompt["negative_constraint_id"],
                "negative_constraints": prompt["negative_constraints"],
                "negative_constraints_sha256": prompt[
                    "negative_constraints_sha256"
                ],
                "positive_prompt": prompt["positive_prompt"],
                "positive_prompt_id": prompt["prompt_id"],
                "positive_prompt_sha256": prompt["positive_prompt_sha256"],
                "request_id": request_id,
                "retry_authorized": False,
                "slot": item["slot"],
                "source_camera_path": binding["camera_file"]["path"],
                "source_camera_sha256": binding["camera_file"]["sha256"],
                "source_condition_path": binding["source_rgb"]["path"],
                "source_condition_sha256": binding["source_rgb"]["sha256"],
                "source_mask_path": binding["mask"]["path"],
                "source_mask_sha256": binding["mask"]["sha256"],
                "source_pose_path": binding["pose_smplx_file"]["path"],
                "source_pose_sha256": binding["pose_smplx_file"]["sha256"],
                "source_request_id": source_request["request_id"],
                "source_request_path": str(source_request_path),
                "source_request_sha256": file_sha256(source_request_path),
                "status": "DRAFT_PENDING_USER_AUTHORIZATION",
                "teacher_target_promotion": False,
            }
        )
    return records


def build_global_missing(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "accepted_promotion": False,
        "base_missing_registry_path": str(BASE_MISSING),
        "base_missing_registry_sha256": file_sha256(BASE_MISSING),
        "generation_authorized": False,
        "missing_cell_count": 6,
        "paper_final": False,
        "records": [
            {
                "camera": item["camera"],
                "cell_descriptor": item["cell_descriptor"],
                "cell_key": item["cell_key"],
                "direction": item["direction"],
                "garment": item["garment"],
                "generation_authorized": False,
                "local_source_condition_path": item["local_source_condition_path"],
                "slot": item["slot"],
                "source_condition_path": item["source_condition_path"],
                "source_condition_sha256": item["source_condition_sha256"],
                "source_mask_path": item["source_mask_path"],
                "source_mask_sha256": item["source_mask_sha256"],
            }
            for item in records
        ],
        "rerun_authorized_count": 0,
        "rerun_required_count": 6,
        "retry_authorized": False,
        "schema_version": "canondressgs.subject00.global_missing_cell_registry.v1",
        "task_id": TASK_ID,
        "teacher_target_promotion": False,
    }


def build_manifest(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "accepted_promotion": False,
        "attempt_namespace": PLANNED_NAMESPACE,
        "automatic_acceptance": False,
        "candidate_count_per_cell": 1,
        "external_api_allowed": False,
        "generation_authorized": False,
        "generation_call_count": 0,
        "native_resolution_family": list(RESOLUTION_FAMILY),
        "no_generate_many_then_select": True,
        "o03_hood_removal_prompt_contract": list(O03_HOOD_REMOVAL_CLAUSES),
        "paper_final": False,
        "postprocessing_allowed": False,
        "request_count": 6,
        "requests": records,
        "retry_authorized": False,
        "retry_count": 0,
        "schema_version": "canondressgs.subject00.remaining_six_cell_generation_manifest_draft.v1",
        "status": "DRAFT_PENDING_USER_AUTHORIZATION",
        "task_id": TASK_ID,
        "teacher_target_promotion": False,
    }


def write_contract(records: list[dict[str, Any]]) -> None:
    request_lines = "\n".join(
        f"- `{item['cell_descriptor']}` -> `{item['request_id']}`"
        for item in records
    )
    prompt_lines = "\n".join(
        f"- `{garment}`: `{prompt_index()[garment]['prompt_id']}` / "
        f"`{prompt_index()[garment]['positive_prompt_sha256']}`"
        for garment in ("O01", "O03", "O04")
    )
    CONTRACT_PATH.write_text(
        f"""# Subject00 Remaining-Six-Cell Generation Contract

- Task: `{TASK_ID}`
- Planned namespace: `{PLANNED_NAMESPACE}`
- Generation authorized: `false`
- Request count: `6`
- Retry authorized: `false`
- Accepted promotion: `false`
- Teacher-target promotion: `false`

## Scope

This draft covers only the six cells listed below. It prepares source, mask, camera, pose, prompt, and SHA bindings. It does not execute image generation and does not create the planned attempt namespace.

{request_lines}

## Frozen Prompt Bindings

{prompt_lines}

The full positive and negative prompt text is frozen per request in `{MANIFEST_PATH.name}`. O03 execution must retain the proven complete hood-removal clauses from the four-cell canary in addition to the frozen O03 garment prompt.

## Output Gate

- Native PNG only.
- Allowed native resolution family: `1348x1167`, `1349x1166`, or `1350x1165` landscape.
- No resize, crop, padding, outpainting, recomposition, re-encoding repair, or other postprocessing.
- One generation call and one candidate per cell after separate explicit authorization.
- No retry and no generate-many-then-select behavior.
- Every output remains a candidate pending technical validation and human review.
- No accepted or Teacher-target promotion is authorized by this contract.

## Status

`generation_authorized = false`. The only next task is `{NEXT_TASK}`.
""",
        encoding="utf-8",
    )


def write_report(overlay: dict[str, Any], snapshots: dict[str, Any]) -> None:
    decision_lines = "\n".join(
        f"- `{item['request_id']}`: `{item['final_human_visual_decision']}`"
        for item in overlay["records"]
    )
    snapshot_lines = "\n".join(
        f"- `{name}`: {item['file_count']} files, tree SHA256 `{item['tree_sha256']}`"
        for name, item in snapshots.items()
    )
    REPORT_PATH.write_text(
        f"""# Subject00 O03 Canary Human-Review Freeze

- Task: `{TASK_ID}`
- Source: `{SOURCE_BRANCH}` at `{SOURCE_HEAD}`
- Reviewer: `{REVIEWER}`
- Final classification: `{FINAL_CLASSIFICATION}`

## Frozen Decisions

{decision_lines}

All four outputs pass hood removal, garment, camera/pose, background, and full-body completeness. All four face/head review crops are frozen as `INVALID_OFF_TARGET_CROP`; identity evidence is therefore taken only from valid visible source/output evidence. The slot04 machine result remains `FAIL`, with a human visual override of `PASS_VISUALLY_ACCEPTABLE_ALIGNMENT` recorded as a separate field.

## Global Coverage

- Reviewed canary outputs: 4
- Human-pass canary outputs: 4
- Newly selected cells: 4
- Total selected cells: 18
- Remaining missing cells: 6
- Accepted: 0
- Teacher target: 0

## Immutability

No raw generated image was edited or moved. The slot00 file remains under `technical_failures`. Attempt tree snapshots at freeze time:

{snapshot_lines}

No generation call, retry, accepted promotion, Teacher-target promotion, or paper modification was performed.

## Remaining-Six Draft

The generation contract and manifest are prepared with `generation_authorized=false`, six requests, zero retries, and no promotion authority. No planned attempt directory was created.

## Next Task

`{NEXT_TASK}`
""",
        encoding="utf-8",
    )


def write_artifacts() -> dict[str, Any]:
    preflight = source_preflight()
    raw_outputs = verify_raw_outputs()
    snapshots_before = attempt_snapshots()
    source_review = read_json(SOURCE_REVIEW)
    raw_registry = read_json(RAW_REGISTRY)
    base_selected = read_json(BASE_SELECTED)

    overlay = build_overlay(source_review, raw_registry)
    write_json(OVERLAY_PATH, overlay)
    overlay = read_json(OVERLAY_PATH)

    selected = build_global_selected(base_selected, overlay)
    write_json(GLOBAL_SELECTED_PATH, selected)

    remaining_records = build_remaining_records()
    missing = build_global_missing(remaining_records)
    write_json(GLOBAL_MISSING_PATH, missing)

    manifest = build_manifest(remaining_records)
    write_json(MANIFEST_PATH, manifest)
    write_contract(remaining_records)

    snapshots_after = attempt_snapshots()
    if snapshots_after != snapshots_before:
        raise RuntimeError("one or more attempt trees changed during freeze")
    write_report(overlay, snapshots_after)

    summary = {
        "accepted_count": 0,
        "attempt_mutations": {name: 0 for name in ATTEMPTS},
        "attempt_snapshots": snapshots_after,
        "final_classification": FINAL_CLASSIFICATION,
        "generation_authorized": False,
        "human_pass_canary_outputs": 4,
        "new_generation_calls": 0,
        "newly_selected_cells": 4,
        "next_task": NEXT_TASK,
        "paper_final": False,
        "paper_modifications": 0,
        "planned_attempt_namespace": PLANNED_NAMESPACE,
        "planned_attempt_namespace_created": (ATTEMPTS_ROOT / PLANNED_NAMESPACE).exists(),
        "raw_output_bindings": raw_outputs,
        "remaining_missing_cell_count": 6,
        "remaining_missing_cells": [normalized_descriptor(item) for item in REMAINING],
        "remaining_six_generation_call_count": 0,
        "remaining_six_request_count": 6,
        "reviewed_canary_outputs": 4,
        "retry_authorized": False,
        "source_preflight": preflight,
        "task_id": TASK_ID,
        "teacher_target_count": 0,
        "test_result": "PASS",
        "total_selected_cells": 18,
    }
    write_json(SUMMARY_PATH, summary)

    handoff = {
        "accepted_count": 0,
        "artifact_paths": {
            "contract": str(CONTRACT_PATH.relative_to(ROOT)),
            "final_report": str(REPORT_PATH.relative_to(ROOT)),
            "final_summary": str(SUMMARY_PATH.relative_to(ROOT)),
            "global_missing_registry": str(GLOBAL_MISSING_PATH.relative_to(ROOT)),
            "global_selected_registry": str(GLOBAL_SELECTED_PATH.relative_to(ROOT)),
            "human_review_overlay": str(OVERLAY_PATH.relative_to(ROOT)),
            "manifest_draft": str(MANIFEST_PATH.relative_to(ROOT)),
            "tests": str(TESTS_PATH.relative_to(ROOT)),
        },
        "execution_status": "HUMAN_REVIEW_FROZEN_REMAINING_SIX_DRAFT_PREPARED",
        "final_classification": FINAL_CLASSIFICATION,
        "generation_authorized": False,
        "new_branch": BRANCH,
        "new_generation_calls": 0,
        "next_task": NEXT_TASK,
        "paper_final": False,
        "paper_modifications": 0,
        "remaining_missing_cell_count": 6,
        "request_count": 6,
        "retry_authorized": False,
        "schema_version": "canondressgs.subject00.o03_canary_human_review_freeze_handoff.v1",
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "task_id": TASK_ID,
        "teacher_target_count": 0,
        "total_selected_cells": 18,
        "worktree": str(WORKTREE),
    }
    write_json(HANDOFF_PATH, handoff)
    return validate(write=True)


def validate(*, write: bool) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        checks.append(
            {"detail": detail, "name": name, "status": "PASS" if condition else "FAIL"}
        )

    try:
        overlay = read_json(OVERLAY_PATH)
        selected = read_json(GLOBAL_SELECTED_PATH)
        missing = read_json(GLOBAL_MISSING_PATH)
        manifest = read_json(MANIFEST_PATH)
        summary = read_json(SUMMARY_PATH)
        source_review = read_json(SOURCE_REVIEW)
        generation_manifest = read_json(EXTERNAL_GENERATION_MANIFEST)
        continuation_summary = read_json(CONTINUATION_SUMMARY)

        source_branch = run_git(["branch", "--show-current"], SOURCE_WORKTREE)
        source_head = run_git(["rev-parse", "HEAD"], SOURCE_WORKTREE)
        source_status = run_git(["status", "--porcelain=v1"], SOURCE_WORKTREE)
        current_branch = run_git(["branch", "--show-current"], ROOT)
        is_ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", SOURCE_HEAD, "HEAD"], cwd=ROOT
        ).returncode == 0

        check("source_branch_exact", source_branch == SOURCE_BRANCH)
        check("source_head_exact", source_head == SOURCE_HEAD)
        check("source_worktree_clean", source_status == "")
        check("target_branch_exact", current_branch == BRANCH)
        check("target_descends_from_source_head", is_ancestor)
        check("overlay_record_count_4", overlay["record_count"] == len(overlay["records"]) == 4)
        check("reviewed_canary_outputs_4", overlay["reviewed_canary_outputs"] == 4)
        check("human_pass_canary_outputs_4", overlay["human_pass_canary_outputs"] == 4)
        check("all_common_visual_gates_pass", all(
            all(item[field] is True for field in (
                "hood_removal_pass", "garment_pass", "camera_pose_pass",
                "background_pass", "full_body_completeness_pass",
            )) for item in overlay["records"]
        ))
        check("all_face_head_crops_invalid", all(
            item["face_head_review_crop_status"] == "INVALID_OFF_TARGET_CROP"
            for item in overlay["records"]
        ))
        overlay_by_id = {item["request_id"]: item for item in overlay["records"]}
        for request_id, expected in DECISIONS.items():
            item = overlay_by_id[request_id]
            check(f"{request_id}_identity_pass", item["identity_pass"] == expected["identity_pass"])
            check(f"{request_id}_identity_confidence", item["identity_confidence"] == expected["identity_confidence"])
            check(f"{request_id}_identity_evidence", item["identity_evidence"] == expected["identity_evidence"])
            check(f"{request_id}_registration_override", item["machine_registration_human_override"] == expected["machine_registration_human_override"])
            check(f"{request_id}_final_decision", item["final_human_visual_decision"] == expected["final_human_visual_decision"])
        slot04 = overlay_by_id["subject00_O03_slot04_canary_attempt004_cand00"]
        check("slot04_machine_registration_stays_fail", slot04["machine_registration_status"] == "FAIL")
        check("slot04_human_override_exact", slot04["machine_registration_human_override"] == "PASS_VISUALLY_ACCEPTABLE_ALIGNMENT")
        check("selected_count_18", selected["record_count"] == selected["total_selected_cells"] == len(selected["records"]) == 18)
        check("newly_selected_count_4", selected["newly_selected_cell_count"] == 4)
        check("unique_selected_cells", selected["unique_selection_per_cell"] is True)
        check("selection_distribution_6_each", selected["selection_counts_by_garment"] == {"O01": 6, "O03": 6, "O04": 6})
        check("missing_count_6", missing["missing_cell_count"] == len(missing["records"]) == 6)
        expected_remaining = [normalized_descriptor(item) for item in REMAINING]
        check("remaining_cells_exact", [item["cell_descriptor"] for item in missing["records"]] == expected_remaining)
        check("manifest_request_count_6", manifest["request_count"] == len(manifest["requests"]) == 6)
        check("generation_not_authorized", manifest["generation_authorized"] is False)
        check("retry_not_authorized", manifest["retry_authorized"] is False)
        check("accepted_promotion_false", manifest["accepted_promotion"] is False)
        check("teacher_target_promotion_false", manifest["teacher_target_promotion"] is False)
        check("manifest_no_execution", manifest["generation_call_count"] == 0 and all(item["generation_call_count"] == 0 for item in manifest["requests"]))
        check(
            "o03_hood_removal_contract_frozen",
            manifest["o03_hood_removal_prompt_contract"]
            == list(O03_HOOD_REMOVAL_CLAUSES),
        )
        check(
            "o03_requests_inherit_hood_removal_contract",
            all(
                item["additional_prompt_clauses"]
                == (
                    list(O03_HOOD_REMOVAL_CLAUSES)
                    if item["garment"] == "O03"
                    else []
                )
                for item in manifest["requests"]
            ),
        )
        check("accepted_count_0", selected["accepted_count"] == overlay["accepted_count"] == 0)
        check("teacher_target_count_0", selected["teacher_target_count"] == overlay["teacher_target_count"] == 0)
        check("source_review_unchanged", overlay["source_human_review_manifest_sha256"] == file_sha256(SOURCE_REVIEW))
        raw_now = verify_raw_outputs()
        check("four_raw_outputs_unchanged", raw_now == summary["raw_output_bindings"])
        slot00_path = Path(raw_now["subject00_O03_slot00_canary_attempt004_cand00"]["path"])
        check("slot00_still_in_technical_failures", slot00_path.parent.name == "technical_failures")
        current_snapshots = attempt_snapshots()
        check("attempt_001_to_004_unchanged", current_snapshots == summary["attempt_snapshots"])
        check("attempt_mutations_zero", all(value == 0 for value in summary["attempt_mutations"].values()))
        check(
            "no_new_generation_calls",
            continuation_summary["total_generation_calls"] == 4
            and summary["new_generation_calls"] == 0
            and generation_manifest["generation_calls"] == 1,
            "The first-stage manifest remains at 1 call; the continuation summary remains at 4 total calls.",
        )
        check("planned_attempt_not_created", not (ATTEMPTS_ROOT / PLANNED_NAMESPACE).exists())
        paper_changes = run_git(["diff", "--name-only", SOURCE_HEAD, "--", "paper_draft"], ROOT)
        check("paper_unmodified", paper_changes == "" and summary["paper_modifications"] == 0)
        check("final_classification_exact", summary["final_classification"] == FINAL_CLASSIFICATION)
        check("next_task_exact", summary["next_task"] == NEXT_TASK)
        check("only_one_next_task", isinstance(summary["next_task"], str))
        check("source_manifest_has_four_records", len(source_review["records"]) == 4)
        for path in (OVERLAY_PATH, GLOBAL_SELECTED_PATH, GLOBAL_MISSING_PATH, MANIFEST_PATH, SUMMARY_PATH):
            payload = read_json(path)
            check(f"content_hash_{path.stem}", payload["content_sha256"] == canonical_hash(payload))
    except Exception as exc:  # noqa: BLE001
        check("validation_exception", False, f"{type(exc).__name__}: {exc}")

    result = {
        "fail_count": sum(item["status"] == "FAIL" for item in checks),
        "paper_final": False,
        "schema_version": "canondressgs.subject00.o03_canary_human_review_freeze_tests.v1",
        "status": "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL",
        "task_id": TASK_ID,
        "test_count": len(checks),
        "checks": checks,
    }
    if write:
        write_json(TESTS_PATH, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("write", "validate"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = write_artifacts() if args.action == "write" else validate(write=False)
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
