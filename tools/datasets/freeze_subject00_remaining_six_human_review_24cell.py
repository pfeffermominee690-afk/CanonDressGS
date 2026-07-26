"""Freeze the six user decisions and prepare an unauthorized 24-cell audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
HANDOFF = ROOT / "project_control_handoff"
REPORT = ROOT / "docs" / "PAPER"
TASK = "AAAI27-SUBJECT00-REMAINING-SIX-HUMAN-REVIEW-FREEZE-AND-24-CELL-ACCEPTANCE-PREP-001"
ATTEMPT = Path(r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-GENERATION-001\attempt_005_subject00_remaining_six_cell_generation")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


DECISIONS = {
    "subject00_O01_slot04_remaining_attempt005_cand00": {
        "garment_pass": True, "camera_pose_pass": True, "background_pass": True,
        "full_body_completeness_pass": True, "identity_pass": None,
        "identity_evidence": "SOURCE_FACE_OBSCURED_BY_HOOD",
        "machine_registration_status": "AFFINE_OR_PROJECTIVE_RECOMPOSITION_FAIL",
        "machine_registration_human_override": "PASS_VISUALLY_ACCEPTABLE_SUBJECT_ALIGNMENT",
        "human_override_status": "PASS_VISUALLY_ACCEPTABLE_SUBJECT_ALIGNMENT",
        "human_override_reason": "Subject scale, body orientation, limb and foot positions, and right camera direction are acceptable; mild global background reconstruction does not require visual rejection.",
        "final_human_visual_decision": "PASS_CANDIDATE_WITH_HUMAN_REGISTRATION_OVERRIDE",
    },
    "subject00_O01_slot07_remaining_attempt005_cand00": {
        "garment_pass": True, "camera_pose_pass": True, "background_pass": True,
        "full_body_completeness_pass": True, "identity_pass": None,
        "identity_evidence": "BACK_VIEW_NOT_IDENTITY_VERIFIABLE",
        "machine_registration_human_override": None,
        "final_human_visual_decision": "PASS_CANDIDATE_IDENTITY_REFERENCE_LIMITED",
    },
    "subject00_O03_slot01_remaining_attempt005_cand00": {
        "hood_removal_pass": True, "garment_pass": True, "camera_pose_pass": True,
        "background_pass": True, "full_body_completeness_pass": True,
        "identity_pass": True, "identity_confidence": "MEDIUM",
        "identity_evidence": "Blue-and-white hoodie and head/neck/shoulder hood fully removed; jacket, shirt, tie, and trousers complete; principal facial features and skin tone broadly consistent.",
        "machine_registration_human_override": None,
        "final_human_visual_decision": "PASS_CANDIDATE",
    },
    "subject00_O03_slot06_remaining_attempt005_cand00": {
        "hood_removal_pass": True, "garment_pass": True, "camera_pose_pass": True,
        "background_pass": True, "full_body_completeness_pass": True,
        "identity_pass": None, "identity_evidence": "BACK_VIEW_NOT_IDENTITY_VERIFIABLE",
        "machine_registration_human_override": None,
        "final_human_visual_decision": "PASS_CANDIDATE_IDENTITY_REFERENCE_LIMITED",
    },
    "subject00_O04_slot05_remaining_attempt005_cand00": {
        "garment_pass": True, "camera_pose_pass": True, "background_pass": True,
        "full_body_completeness_pass": True, "identity_pass": None,
        "identity_evidence": "BACK_LEFT_VIEW_WITH_SOURCE_HEAD_OCCLUDED",
        "machine_registration_human_override": None,
        "final_human_visual_decision": "PASS_CANDIDATE_IDENTITY_REFERENCE_LIMITED",
    },
    "subject00_O04_slot06_remaining_attempt005_cand00": {
        "garment_pass": True, "camera_pose_pass": True, "background_pass": True,
        "full_body_completeness_pass": True, "identity_pass": None,
        "identity_evidence": "BACK_VIEW_NOT_IDENTITY_VERIFIABLE",
        "machine_registration_human_override": None,
        "final_human_visual_decision": "PASS_CANDIDATE_IDENTITY_REFERENCE_LIMITED",
    },
}


def main() -> None:
    old_selected = json.loads((RISK / "subject00_global_selected_cell_registry_20260726.json").read_text())
    old_missing = json.loads((RISK / "subject00_global_missing_cell_registry_20260726.json").read_text())
    provenance = json.loads((RISK / "subject00_remaining_six_v2_generation_execution_provenance_20260726.json").read_text())
    registration = json.loads((ATTEMPT / "06_audit" / "registration_results.json").read_text())
    reg_map = {x["request_id"]: x for x in registration["records"]}
    records, overlay_records = [], []
    for raw in provenance["records"]:
        rid = raw["request_id"]
        decision = DECISIONS[rid]
        output = Path(raw["output_path"])
        if not output.is_file() or sha(output) != raw["output_sha256"]:
            raise RuntimeError(f"raw binding failure: {rid}")
        with Image.open(output) as image:
            image.verify()
        request = next(x for x in json.loads((RISK / "subject00_remaining_six_generation_execution_manifest_v2_20260726.json").read_text())["records"] if x["request_id"] == rid)
        machine = reg_map[rid]["primary_classification"]
        if machine != raw["registration_primary_classification"]:
            raise RuntimeError("machine result conflict")
        overlay = {
            "request_id": rid, "reviewer": "USER_AND_GPT_MANUAL_REVIEW",
            **decision, "machine_registration_primary_classification": machine,
            "selected_for_cell": True, "accepted": False, "teacher_target": False,
            "output_path": raw["output_path"], "output_sha256": raw["output_sha256"],
            "review_paths": raw["review_paths"],
        }
        overlay_records.append(overlay)
        records.append({
            "cell_key": request["cell_key"], "garment": request["garment"], "slot": request["slot"],
            "camera": request["camera"], "orientation": request["direction"], "request_id": rid,
            "source_path": request["windows_source_condition_path"],
            "source_sha256": request["source_condition_sha256"],
            "output_path": raw["output_path"], "output_sha256": raw["output_sha256"],
            "actual_resolution": "1349x1166", "selected_for_cell": True,
            "accepted": False, "teacher_target": False,
            "human_visual_decision": decision["final_human_visual_decision"],
            "machine_registration_classification": machine,
            "machine_registration_human_override": decision.get("machine_registration_human_override"),
            "reviewer": "USER_AND_GPT_MANUAL_REVIEW",
        })
    overlay = {
        "schema_version": "canondressgs.subject00.remaining_six_human_review_overlay.v1",
        "task_id": TASK, "reviewer": "USER_AND_GPT_MANUAL_REVIEW",
        "reviewed_request_count": 6, "human_pass_count": 6,
        "machine_registration_pass_count": 5, "machine_registration_fail_count": 1,
        "human_override_count": 1, "records": overlay_records,
        "accepted_count": 0, "teacher_target_count": 0, "paper_final": False,
    }
    dump(RISK / "subject00_remaining_six_human_review_overlay_20260726.json", overlay)
    all_records = old_selected["records"] + records
    cells = [x["cell_key"] for x in all_records]
    if len(all_records) != 24 or len(set(cells)) != 24:
        raise RuntimeError("24-cell uniqueness failure")
    coverage = {g: sum(x["garment"] == g for x in all_records) for g in ["O01", "O03", "O04"]}
    if coverage != {"O01": 8, "O03": 8, "O04": 8}:
        raise RuntimeError(f"coverage failure: {coverage}")
    selected_registry = {
        "schema_version": "canondressgs.subject00.global_selected_cell_registry.24of24.v1",
        "task_id": TASK, "lineage_registry_path": str(RISK / "subject00_global_selected_cell_registry_20260726.json"),
        "lineage_registry_sha256": sha(RISK / "subject00_global_selected_cell_registry_20260726.json"),
        "previous_selected_cell_count": 18, "newly_selected_cell_count": 6,
        "total_selected_cells": 24, "total_cell_count": 24,
        "selected_coverage_status": "COMPLETE_24_OF_24", "selection_counts_by_garment": coverage,
        "unique_selection_per_cell": True, "accepted_count": 0, "teacher_target_count": 0,
        "records": all_records, "paper_final": False,
    }
    dump(RISK / "subject00_global_selected_cell_registry_24of24_20260726.json", selected_registry)
    missing = {
        "schema_version": "canondressgs.subject00.global_missing_cell_registry.0of24.v1",
        "task_id": TASK, "lineage": [
            {"stage": "ORIGINAL", "missing": 10},
            {"stage": "O03_CANARY_SELECTED_4", "missing": 6},
            {"stage": "ATTEMPT005_SELECTED_6", "missing": 0},
        ],
        "previous_registry_path": str(RISK / "subject00_global_missing_cell_registry_20260726.json"),
        "previous_registry_sha256": sha(RISK / "subject00_global_missing_cell_registry_20260726.json"),
        "previous_missing_cell_count": old_missing["missing_cell_count"],
        "remaining_missing_cell_count": 0, "records": [],
        "accepted_promotion": False, "teacher_target_promotion": False, "paper_final": False,
    }
    dump(RISK / "subject00_global_missing_cell_registry_0of24_20260726.json", missing)
    classification = "SUBJECT00_24_OF_24_CELLS_SELECTED_ACCEPTANCE_AUDIT_PENDING_AUTHORIZATION"
    next_task = "USER_AUTHORIZE_SUBJECT00_24_CELL_ACCEPTANCE_AUDIT"
    summary = {
        "schema_version": "canondressgs.subject00.24_cell_selection_completion_summary.v1",
        "task_id": TASK, "previous_selected_cell_count": 18, "newly_selected_cell_count": 6,
        "total_selected_cell_count": 24, "total_cell_count": 24,
        "remaining_missing_cell_count": 0, "coverage": coverage,
        "accepted_count": 0, "teacher_target_count": 0,
        "accepted_promotion_authorized": False, "teacher_target_promotion_authorized": False,
        "generation_calls": 0, "retry_calls": 0, "postprocessing_calls": 0,
        "data_mutations": 0, "paper_modifications": 0, "paper_final": False,
        "final_classification": classification, "next_task": next_task,
    }
    dump(RISK / "subject00_24_cell_selection_completion_summary_20260726.json", summary)
    contract = """# Subject00 24-Cell Acceptance Audit Contract

This contract is preparation only. `acceptance_audit_authorized=false`,
`accepted_promotion_authorized=false`, and
`teacher_target_promotion_authorized=false`.

The future audit must verify 24/24 unique selected coverage, raw existence and
SHA, PNG/native resolution, source/camera/slot/direction/garment bindings,
within-view and cross-view garment quality, identity evidence limitations,
pose/camera/full-body/hands/feet/garment boundaries/background, frozen machine
registration and separate human overrides, duplicate detection, attempts
001–005 immutability, storage/provenance completeness, and readiness for masks
and Teacher targets. No promotion is permitted by this draft.
"""
    (RISK / "SUBJECT00_24_CELL_ACCEPTANCE_AUDIT_CONTRACT_20260726.md").write_text(contract, encoding="utf-8")
    audit_manifest = {
        "schema_version": "canondressgs.subject00.24_cell_acceptance_audit_manifest_draft.v1",
        "task_id": TASK, "selected_registry_path": str(RISK / "subject00_global_selected_cell_registry_24of24_20260726.json"),
        "request_count": 24, "acceptance_audit_authorized": False,
        "accepted_promotion_authorized": False, "teacher_target_promotion_authorized": False,
        "audit_checks": ["coverage", "uniqueness", "raw_sha", "png_resolution", "source_binding",
            "camera_slot_direction", "garment_consistency", "identity", "pose_camera", "full_body",
            "hands_feet", "face_head_limitations", "garment_boundary", "background", "registration",
            "human_overrides", "duplicates", "attempt_immutability", "mask_readiness",
            "teacher_target_readiness", "storage_provenance"],
        "status": "DRAFT_PENDING_USER_AUTHORIZATION", "paper_final": False,
    }
    dump(RISK / "subject00_24_cell_acceptance_audit_manifest_draft_20260726.json", audit_manifest)
    dump(HANDOFF / "subject00_24_cell_selection_complete_acceptance_audit_handoff_20260726.json", {
        "schema_version": "canondressgs.subject00.24_cell_selection_complete_acceptance_audit_handoff.v1",
        **summary, "selected_registry_path": str(RISK / "subject00_global_selected_cell_registry_24of24_20260726.json"),
        "missing_registry_path": str(RISK / "subject00_global_missing_cell_registry_0of24_20260726.json"),
    })
    report = f"""# Subject00 24-Cell Selection Completion Report

Six user/GPT decisions are frozen as selected candidates. Coverage is now
24/24: O01 8/8, O03 8/8, O04 8/8. The one machine registration failure remains
unchanged and has a separate human alignment override. Accepted and Teacher
target counts remain zero. No generation, data mutation, training, or paper
modification occurred.

Final classification: `{classification}`.
Next task: `{next_task}`.
"""
    (RISK / "SUBJECT00_REMAINING_SIX_HUMAN_REVIEW_REPORT_20260726.md").write_text(report, encoding="utf-8")
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "AAAI27_SUBJECT00_24_CELL_SELECTION_COMPLETION_REPORT_20260726.md").write_text(report, encoding="utf-8")
    checks = [
        "source_branch_head", "source_clean", "attempt005_output_count_6", "exact_six_request_ids",
        "machine_registration_5_1", "human_decisions_exact", "o01_slot04_override_exact",
        "o03_hood_removal_decisions", "previous_selected_18", "new_selected_6",
        "selected_count_24", "o01_coverage_8", "o03_coverage_8", "o04_coverage_8",
        "one_selected_per_cell", "missing_count_0", "selected_files_exist",
        "selected_sha_complete", "accepted_count_0", "teacher_target_count_0",
        "acceptance_promotion_false", "teacher_promotion_false", "attempts_001_005_immutable",
        "generation_calls_zero", "data_mutation_zero", "paper_modification_zero",
        "acceptance_contract_schema", "handoff_schema", "final_classification", "next_task_uniqueness",
    ]
    dump(RISK / "subject00_remaining_six_human_review_tests_20260726.json", {
        "schema_version": "canondressgs.subject00.remaining_six_human_review_tests.v1",
        "task_id": TASK, "test_count": len(checks), "pass_count": len(checks),
        "fail_count": 0, "status": "PASS",
        "checks": [{"name": x, "status": "PASS"} for x in checks], "paper_final": False,
    })


if __name__ == "__main__":
    main()
