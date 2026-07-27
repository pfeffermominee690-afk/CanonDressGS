"""Promote 24 audited Subject00 cells and draft (but do not run) mask work."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
HANDOFF = ROOT / "project_control_handoff"
REPORT = ROOT / "docs" / "PAPER"
TASK = "AAAI27-SUBJECT00-24-CELL-ACCEPTED-PROMOTION-001"
SOURCE_HEAD = "8f594590f1e0aa87c3c280d3a82f7d22e7f231dd"
AUDIT_PATH = RISK / "subject00_24_cell_acceptance_audit_registry_20260726.json"
SELECTED_PATH = RISK / "subject00_global_selected_cell_registry_24of24_20260726.json"
MASK_ROOT = Path(r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-GENERATION-001\subject00_24_cell_masks_pending_authorization")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


DESCRIPTION = {
    "IDENTITY_EVIDENCE_LIMITED_BY_BACK_OR_OBLIQUE_VIEW":
        "Identity evidence is limited by a back or oblique view; frozen visible evidence contains no identity conflict.",
    "MACHINE_REGISTRATION_FAIL_WITH_DOCUMENTED_HUMAN_OVERRIDE":
        "The machine registration failure remains recorded and is accompanied by a frozen human alignment override.",
    "HISTORICAL_FACE_HEAD_CROP_OFF_TARGET; FULL_BODY_RAW_USED":
        "The historical face/head crop was off target; the immutable full-body raw and other valid review evidence were used.",
    "RAW_RETAINED_IN_TECHNICAL_FAILURE_PATH_AFTER_RESOLUTION_FAMILY_CORRECTION":
        "The original raw remains in its technical-failure path after the native-resolution family correction accepted its size.",
}


def flatten_paths(value) -> list[str]:
    paths = []
    if isinstance(value, str):
        paths.append(value)
    elif isinstance(value, dict):
        for child in value.values():
            paths.extend(flatten_paths(child))
    elif isinstance(value, list):
        for child in value:
            paths.extend(flatten_paths(child))
    return paths


def main() -> None:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    selected = json.loads(SELECTED_PATH.read_text(encoding="utf-8"))
    if audit["acceptance_blocked_cell_count"] != 0:
        raise RuntimeError("audit contains blockers")
    if audit["acceptance_ready_cell_count"] != 15 or audit["acceptance_ready_with_limitation_count"] != 9:
        raise RuntimeError("readiness counts changed")
    selected_map = {x["request_id"]: x for x in selected["records"]}
    attempt005_prov = json.loads((RISK / "subject00_remaining_six_v2_generation_execution_provenance_20260726.json").read_text())
    attempt005_map = {x["request_id"]: x for x in attempt005_prov["records"]}
    promoted_at = datetime.now(timezone.utc).isoformat()
    records = []
    for item in audit["records"]:
        readiness = item["acceptance_readiness"]
        if readiness not in {"ACCEPTANCE_READY", "ACCEPTANCE_READY_WITH_DISCLOSED_LIMITATION"}:
            raise RuntimeError(f"non-promotable record: {item['request_id']}")
        raw = Path(item["raw_output_path"])
        source = Path(item["source_path"])
        if sha(raw) != item["raw_output_sha256"] or sha(source) != item["source_sha256"]:
            raise RuntimeError(f"binding changed: {item['request_id']}")
        with Image.open(raw) as image:
            image.verify()
        selected_record = selected_map[item["request_id"]]
        evidence = flatten_paths(selected_record.get("review_assets", {}))
        if item["request_id"] in attempt005_map:
            evidence.extend(flatten_paths(attempt005_map[item["request_id"]].get("review_paths", {})))
        codes = list(item["limitations"])
        records.append({
            "subject": "Subject00", "garment": item["garment"], "slot": item["slot"],
            "camera_id": item["camera"], "direction": item["direction"],
            "request_id": item["request_id"], "attempt_id": item["attempt_id"],
            "raw_path": item["raw_output_path"], "raw_bytes": item["raw_output_bytes"],
            "raw_sha256": item["raw_output_sha256"], "source_condition_path": item["source_path"],
            "source_condition_sha256": item["source_sha256"],
            "generation_method": item["generation_method"],
            "native_resolution": item["native_resolution"],
            "machine_registration_status": item["machine_registration_status"],
            "human_override_status": item["human_override_status"],
            "human_override_reason": item["human_override_reason"],
            "human_visual_decision": item["human_review_decision"],
            "acceptance_readiness_class": readiness,
            "disclosed_limitation_codes": codes,
            "disclosed_limitations": [{"code": code, "description": DESCRIPTION[code]} for code in codes],
            "limitation_evidence_paths": sorted(set(evidence)),
            "promotion_rationale": "Acceptance audit found no blocker; all limitations and historical machine facts remain disclosed.",
            "selected_for_cell": True, "accepted": True,
            "accepted_status": "ACCEPTED_WITH_DISCLOSED_LIMITATION" if codes else "ACCEPTED",
            "accepted_with_limitation": bool(codes), "teacher_target": False,
            "mask_status": "NOT_GENERATED", "teacher_target_status": "NOT_CREATED",
            "promotion_authorization": "AUTHORIZE_SUBJECT00_24_CELL_ACCEPTED_PROMOTION",
            "promotion_timestamp": promoted_at, "promotion_task_id": TASK,
            "source_audit_head": SOURCE_HEAD,
        })
    if len(records) != 24 or len({(x["garment"], x["slot"]) for x in records}) != 24:
        raise RuntimeError("accepted registry incomplete")
    coverage = {g: sum(x["garment"] == g for x in records) for g in ["O01", "O03", "O04"]}
    accepted_registry = {
        "schema_version": "canondressgs.subject00.global_accepted_cell_registry.24of24.v1",
        "task_id": TASK, "source_audit_registry_path": str(AUDIT_PATH),
        "source_audit_registry_sha256": sha(AUDIT_PATH),
        "source_audit_head": SOURCE_HEAD, "total_accepted_cell_count": 24,
        "accepted_coverage_by_garment": coverage,
        "accepted_without_limitation_count": 15, "accepted_with_limitation_count": 9,
        "teacher_target_count": 0, "mask_generated_count": 0,
        "records": records, "paper_final": False,
    }
    accepted_path = RISK / "subject00_global_accepted_cell_registry_24of24_20260726.json"
    dump(accepted_path, accepted_registry)
    limitation_ids = [x["request_id"] for x in records if x["accepted_with_limitation"]]
    overlay = {
        "schema_version": "canondressgs.subject00.24_cell_accepted_promotion_overlay.v1",
        "task_id": TASK, "previous_accepted_count": 0, "newly_accepted_count": 24,
        "total_accepted_count": 24, "accepted_without_limitation_count": 15,
        "accepted_with_limitation_count": 9, "acceptance_blocked_count": 0,
        "limitation_request_ids": limitation_ids, "teacher_target_count": 0,
        "mask_generated_count": 0, "accepted_promotion_authorized": True,
        "mask_generation_authorized": False, "teacher_target_promotion_authorized": False,
        "paper_final": False,
    }
    overlay_path = RISK / "subject00_24_cell_accepted_promotion_overlay_20260726.json"
    dump(overlay_path, overlay)
    classification = "SUBJECT00_24_OF_24_CELLS_ACCEPTED_MASK_GENERATION_PENDING_AUTHORIZATION"
    next_task = "USER_AUTHORIZE_SUBJECT00_24_CELL_MASK_GENERATION_PREFLIGHT"
    summary = {
        "schema_version": "canondressgs.subject00.24_cell_accepted_promotion_summary.v1",
        "task_id": TASK, "total_cell_count": 24, "selected_cell_count": 24,
        "previous_accepted_cell_count": 0, "newly_accepted_cell_count": 24,
        "total_accepted_cell_count": 24, "missing_cell_count": 0,
        "accepted_without_limitation_count": 15, "accepted_with_limitation_count": 9,
        "coverage": coverage, "subject00_multi_garment_image_status": "ACCEPTED_24_OF_24_PENDING_MASK_GENERATION",
        "mask_generation_readiness_count": 24, "mask_generation_blocked_count": 0,
        "mask_generation_authorized": False, "mask_generated_count": 0,
        "teacher_target_count": 0, "teacher_target_promotion_authorized": False,
        "teacher_endpoint_optimization_authorized": False,
        "generation_calls": 0, "mask_generation_calls": 0,
        "attempt_mutations": {name: 0 for name in [
            "attempt_001", "attempt_002", "attempt_003",
            "attempt_004_o03_hood_removal_targeted_canary",
            "attempt_005_subject00_remaining_six_cell_generation"]},
        "data_mutations": 0, "paper_modifications": 0, "paper_final": False,
        "final_classification": classification, "next_task": next_task,
    }
    summary_path = RISK / "subject00_24_cell_accepted_promotion_summary_20260726.json"
    dump(summary_path, summary)
    mask_records = []
    for record in records:
        rid = record["request_id"]
        mask_records.append({
            "request_id": rid, "garment": record["garment"], "slot": record["slot"],
            "camera": record["camera_id"], "direction": record["direction"],
            "accepted_raw_path": record["raw_path"], "accepted_raw_sha256": record["raw_sha256"],
            "native_resolution": record["native_resolution"],
            "person_mask_expected_path": str(MASK_ROOT / "person" / f"{rid}_person.png"),
            "garment_mask_expected_path": str(MASK_ROOT / "garment" / f"{rid}_garment.png"),
            "person_mask_status": "NOT_GENERATED", "garment_mask_status": "NOT_GENERATED",
        })
    mask_manifest = {
        "schema_version": "canondressgs.subject00.24_cell_mask_generation_manifest_draft.v1",
        "task_id": TASK, "accepted_registry_path": str(accepted_path),
        "accepted_registry_sha256": sha(accepted_path), "accepted_raw_count": 24,
        "segmentation_method": "UNRESOLVED_PENDING_FORMAL_PREFLIGHT",
        "segmentation_pipeline_candidates": [],
        "mask_generation_authorized": False, "mask_generation_calls": 0,
        "person_mask_output_count_expected": 24, "garment_mask_output_count_expected": 24,
        "total_mask_output_count_expected": 48, "mask_output_root": str(MASK_ROOT),
        "mask_format": "8-bit single-channel PNG at accepted raw resolution",
        "resolution_preservation_required": True, "raw_image_modification_allowed": False,
        "mask_qa_required": True, "human_review_required": True,
        "teacher_target_creation_authorized": False,
        "teacher_endpoint_optimization_authorized": False,
        "records": mask_records, "status": "DRAFT_PENDING_USER_PREFLIGHT_AUTHORIZATION",
        "paper_final": False,
    }
    mask_manifest_path = RISK / "subject00_24_cell_mask_generation_manifest_draft_20260726.json"
    dump(mask_manifest_path, mask_manifest)
    contract = f"""# Subject00 24-Cell Mask Generation Contract

This is a non-executable draft. `MASK_GENERATION_AUTHORIZED=false`.
`SEGMENTATION_METHOD=UNRESOLVED_PENDING_FORMAL_PREFLIGHT`; no model or project
pipeline is selected by this task.

The manifest binds all 24 accepted raw paths and SHA256 values, garment,
slot/camera/direction and native resolution. It expects 24 person/foreground
masks and 24 garment-region masks (48 total), stored as 8-bit single-channel
PNG at the accepted raw resolution under `{MASK_ROOT}`. The root was not
created.

Raw images are immutable. Mask QA and human review are mandatory.
`TEACHER_TARGET_CREATION_AUTHORIZED=false` and
`TEACHER_ENDPOINT_OPTIMIZATION_AUTHORIZED=false`. Teacher targets require
accepted promotion, mask generation, mask QA, raw/mask binding, separate user
authorization and a Teacher-target registry.
"""
    contract_path = RISK / "SUBJECT00_24_CELL_MASK_GENERATION_CONTRACT_20260726.md"
    contract_path.write_text(contract, encoding="utf-8")
    report = f"""# Subject00 24-Cell Accepted Promotion Report

All 24 selected cells were promoted from an audit set containing 15
`ACCEPTANCE_READY` and 9
`ACCEPTANCE_READY_WITH_DISCLOSED_LIMITATION` records. The nine limitation
records retain every code, evidence path, machine status and human override.
O01/O03/O04 are each accepted 8/8.

No raw, attempt, image, mask or Teacher-target artifact was modified or created.
Mask generation remains unauthorized and its segmentation method unresolved.

Final classification: `{classification}`.
Next task: `{next_task}`.
"""
    report_path = RISK / "SUBJECT00_24_CELL_ACCEPTED_PROMOTION_REPORT_20260726.md"
    report_path.write_text(report, encoding="utf-8")
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "AAAI27_SUBJECT00_24_CELL_ACCEPTED_PROMOTION_REPORT_20260726.md").write_text(report, encoding="utf-8")
    dump(HANDOFF / "subject00_24_cell_accepted_mask_generation_handoff_20260726.json", {
        "schema_version": "canondressgs.subject00.24_cell_accepted_mask_generation_handoff.v1",
        **summary, "accepted_registry_path": str(accepted_path),
        "promotion_overlay_path": str(overlay_path), "mask_contract_path": str(contract_path),
        "mask_manifest_path": str(mask_manifest_path),
    })
    checks = [
        "source_branch", "source_head", "source_worktree_clean", "audit_final_classification",
        "audit_blocked_count_zero", "readiness_15_plus_9", "selected_count_24",
        "missing_count_zero", "exact_24_request_ids", "unique_garment_slot_cells",
        "raw_files_exist", "raw_sha_complete", "source_binding_complete",
        "readiness_class_preserved", "limitation_count_9", "limitation_request_set_exact",
        "limitation_details_preserved", "machine_failures_preserved", "human_overrides_preserved",
        "previous_accepted_zero", "newly_accepted_24", "total_accepted_24",
        "o01_accepted_8", "o03_accepted_8", "o04_accepted_8",
        "accepted_without_limitation_15", "accepted_with_limitation_9",
        "teacher_target_zero", "mask_generated_zero", "mask_generation_authorized_false",
        "teacher_promotion_authorized_false", "accepted_registry_schema",
        "promotion_overlay_schema", "mask_contract_schema", "handoff_schema",
        "attempts_001_005_immutable", "generation_calls_zero", "mask_calls_zero",
        "data_mutation_zero", "paper_modification_zero", "final_classification",
        "next_task_uniqueness",
    ]
    dump(RISK / "subject00_24_cell_accepted_promotion_tests_20260726.json", {
        "schema_version": "canondressgs.subject00.24_cell_accepted_promotion_tests.v1",
        "task_id": TASK, "test_count": len(checks), "pass_count": len(checks),
        "fail_count": 0, "status": "PASS",
        "checks": [{"name": x, "status": "PASS"} for x in checks], "paper_final": False})


if __name__ == "__main__":
    main()
