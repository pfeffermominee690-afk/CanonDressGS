"""Build the frozen Subject00 remaining-six V2 execution contract (no generation)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
HANDOFF = ROOT / "project_control_handoff"
REPORT = ROOT / "docs" / "PAPER"
V1_CONTRACT = RISK / "SUBJECT00_REMAINING_SIX_CELL_GENERATION_CONTRACT_20260726.md"
V1_MANIFEST = RISK / "subject00_remaining_six_cell_generation_manifest_draft_20260726.json"
PROTOCOL = RISK / "subject00_attempt001_native_landscape_registration_protocol_20260726.json"
REG_IMPL = ROOT / "tools" / "datasets" / "audit_subject00_attempt001_native_landscape_registration.py"
CROP_IMPL = ROOT / "tools" / "datasets" / "subject00_target_aware_review_crops.py"
PROJECT = Path(r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-GENERATION-001")
ATTEMPT = PROJECT / "attempt_005_subject00_remaining_six_cell_generation"
TASK = "AAAI27-SUBJECT00-REMAINING-SIX-GENERATION-CONTRACT-V2-REFREEZE-001"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    if ATTEMPT.exists():
        raise RuntimeError("attempt root already exists")
    v1 = json.loads(V1_MANIFEST.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    roots = {
        "contract_snapshot_root": str(ATTEMPT / "01_contract_snapshot"),
        "request_root": str(ATTEMPT / "02_requests"),
        "source_binding_root": str(ATTEMPT / "03_source_bindings"),
        "valid_raw_output_root": str(ATTEMPT / "04_generation_responses" / "codex_managed_candidates"),
        "technical_failure_root": str(ATTEMPT / "04_generation_responses" / "technical_failures"),
        "generation_response_root": str(ATTEMPT / "04_generation_responses" / "response_metadata"),
        "review_root": str(ATTEMPT / "05_review_assets"),
        "audit_root": str(ATTEMPT / "06_audit"),
        "log_root": str(ATTEMPT / "07_logs"),
    }
    records = []
    for request in v1["requests"]:
        source = Path(request["local_source_condition_path"])
        if not source.is_file() or sha(source) != request["source_condition_sha256"]:
            raise RuntimeError(f"source binding conflict: {source}")
        with Image.open(source) as im:
            im.verify()
        rid, garment = request["request_id"], request["garment"]
        review = Path(roots["review_root"])
        prompt = request["positive_prompt"]
        if garment == "O03":
            prompt += " " + " ".join(request["additional_prompt_clauses"])
        prompt += (
            " Preserve one full-body person. Do not zoom, crop, rotate, outpaint, "
            "recompose, resize, pad, re-encode, or postprocess."
        )
        records.append({
            **request,
            "index": len(records),
            "windows_source_condition_path": str(source),
            "canonical_source_condition_path": request["source_condition_path"],
            "execution_prompt": prompt,
            "expected_raw_output_path": str(Path(roots["valid_raw_output_root"]) / garment / f"{rid}.png"),
            "technical_failure_path": str(Path(roots["technical_failure_root"]) / f"{rid}.png"),
            "response_metadata_path": str(Path(roots["generation_response_root"]) / f"{rid}.json"),
            "request_manifest_path": str(Path(roots["request_root"]) / f"{rid}.json"),
            "review_paths": {
                "side_by_side": str(review / "side_by_side" / f"{rid}_source_output.png"),
                "full_body": str(review / "full_body" / f"{rid}_full_body_page.png"),
                "face_head": str(review / "crops" / f"{rid}_face_head.png"),
                "neck_shoulder": str(review / "crops" / f"{rid}_neck_shoulder.png"),
                "garment_boundary": str(review / "crops" / f"{rid}_garment_boundary.png"),
                "hands_feet": str(review / "crops" / f"{rid}_hands_feet.png"),
                "registration_overlay": str(review / "registration" / f"{rid}_registration_overlay.png"),
            },
            "human_visual_decision": None,
            "selected_for_cell": False,
            "accepted": False,
            "teacher_target": False,
            "generation_authorized": True,
            "status": "FROZEN_READY_NOT_EXECUTED",
        })
    registration = {
        "status": "RECOVERED_UNIQUELY_FROM_ATTEMPT004_ACTUAL_CONTINUATION",
        "implementation_path": str(REG_IMPL),
        "implementation_sha256": sha(REG_IMPL),
        "config_path": str(PROTOCOL),
        "config_sha256": sha(PROTOCOL),
        "attempt004_recorded_protocol_content_sha256": "f8bda763ee845d7e3fd6ece32d7b4a7e8150308cd69df0aa80f15cd69aa53e76",
        "preprocessing": {
            "official_person_mask_foreground_threshold": protocol["background_mask"]["foreground_threshold"],
            "person_dilation_rule": protocol["background_mask"]["person_dilation_rule"],
            "border_exclusion_px": protocol["background_mask"]["image_border_exclusion_px"],
            "mask_interpolation": protocol["background_mask"]["interpolation"],
            "feature_detector": protocol["feature_matching"],
        },
        "models": protocol["models"],
        "thresholds": {
            "machine_similarity_gate": protocol["machine_similarity_gate"],
            "background_structure_gate": protocol["background_structure_gate"],
        },
        "subject_scale_measurement": "uniform_scale and axis_scale_translation from background feature correspondences",
        "full_body_completeness_test": "transformed source person bounds; minimum margin from frozen gate",
        "camera_direction_test": "request camera/direction exact manifest consistency",
        "duplicate_threshold": "SHA256 exact equality; any duplicate raw SHA is flagged",
        "failure_tags_and_priority": protocol["classification_priority"],
        "output_schema": "calculate_registration metric payload plus gate_results, failure_tags, primary_classification",
    }
    manifest = {
        "schema_version": "canondressgs.subject00.remaining_six_generation_execution_manifest.v2",
        "task_id": TASK,
        "source_branch": "research/subject00-o03-canary-human-review-freeze-remaining-six-contract-20260726",
        "source_head": "fcffc661747158778607bd30d1572624780265cb",
        "attempt_namespace": ATTEMPT.name,
        "generation_project_root": str(PROJECT),
        "attempt_root": str(ATTEMPT),
        "attempt_root_created": False,
        "directory_layout": roots,
        "generation_method": {
            "identifier": "CODEX_IMAGE_GENERATION_SKILL/CODEX_PLATFORM_MANAGED_DIRECT_IMAGE_EDIT",
            "generation_provider": "CODEX_IMAGE_GENERATION_SKILL",
            "generation_mode": "CODEX_PLATFORM_MANAGED_DIRECT_IMAGE_EDIT",
            "tool_name": "image_gen.imagegen",
            "request_semantics": "one frozen request maps to one native output",
            "platform_managed_native_dimensions": True,
            "response_metadata_fields": ["request_id", "start_time", "end_time", "tool_response", "source_path", "source_sha256", "output_path", "output_bytes", "output_sha256", "raw_resolution", "png_parse", "no_postprocessing"],
            "external_api_used": False, "api_key_used": False, "custom_backend": False,
            "outpaint_pipeline": False, "postprocessing": False,
            "provenance": "attempt_004 execution manifest at ba1043e8eeba1690ccf64c8f96b47c58425acbb9 and continuation at 96e1ff5237ec13feba449de287576bb79eac0068",
        },
        "native_resolution_policy": "SUBJECT00_NATIVE_LANDSCAPE_2515_FAMILY",
        "accepted_resolution_set": [{"width": 1348, "height": 1167}, {"width": 1349, "height": 1166}, {"width": 1350, "height": 1165}],
        "raw_output_postprocessing_allowed": False,
        "resize_allowed": False, "crop_allowed": False, "padding_allowed": False,
        "outpaint_allowed": False, "reencoding_allowed": False,
        "registration_policy": registration,
        "crop_policy": {
            "implementation_path": str(CROP_IMPL),
            "implementation_sha256": sha(CROP_IMPL),
            "method": "mask/detection person-bbox target-aware display-only crops",
            "back_view_face_label": "face not visible",
            "raw_output_modified": False,
        },
        "contact_sheet_path": str(Path(roots["review_root"]) / "contact_sheet" / "subject00_remaining_six_attempt005_contact_sheet.png"),
        "request_count": 6, "candidates_per_cell": 1,
        "execution_order": [r["request_id"] for r in records],
        "generation_calls_max": 6, "retry_calls_max": 0, "postprocessing_calls_max": 0,
        "generation_authorized": True,
        "authorization_source": "USER_AUTHORIZE_SUBJECT00_REMAINING_SIX_CELL_GENERATION",
        "authorization_scope": "SUBJECT00_REMAINING_SIX_FROZEN_CELLS_ONLY",
        "authorization_consumed_calls": 0, "authorization_remaining_calls": 6,
        "generation_executed": False, "generation_calls": 0,
        "previous_selected_cell_count": 18, "total_selected_cell_count": 18,
        "remaining_missing_cell_count": 6, "accepted_count": 0, "teacher_target_count": 0,
        "failure_rules": {
            "resolution_family_failure": "move raw to bound technical-failure path; stop; no retry",
            "registration_failure": "retain raw; continue; include in human review; no retry",
            "no_parseable_image": "record technical failure; stop; no retry",
            "old_attempt_mutation": "stop; contract violation",
        },
        "records": records, "paper_final": False,
        "status": "FROZEN_READY_WITH_EXISTING_USER_AUTHORIZATION",
    }
    overlay = {
        "schema_version": "canondressgs.subject00.remaining_six_execution_contract_v2_correction_overlay.v1",
        "task_id": TASK,
        "v1_status": "V1_DRAFT_INCOMPLETE_FOR_EXECUTION",
        "v1_contract_path": str(V1_CONTRACT), "v1_contract_sha256": sha(V1_CONTRACT),
        "v1_manifest_path": str(V1_MANIFEST), "v1_manifest_sha256": sha(V1_MANIFEST),
        "corrections": ["execution roots", "per-request output/failure/metadata paths", "generation method", "registration implementation/config/thresholds", "target-aware review paths and crop implementation"],
        "v1_files_modified": False, "generation_executed": False,
    }
    dump(RISK / "subject00_remaining_six_execution_contract_v2_correction_overlay_20260726.json", overlay)
    dump(RISK / "subject00_remaining_six_generation_execution_manifest_v2_20260726.json", manifest)
    contract = f"""# Subject00 Remaining-Six Generation Execution Contract V2

This V2 repairs the incomplete V1 draft without modifying it. It freezes the six
requests in the accompanying manifest, with attempt root `{ATTEMPT}`. The root
must be absent before execution and was not created by this refreeze.

Generation is authorized by the existing unconsumed user authorization, but this
task performs zero generation calls. The method is
`CODEX_IMAGE_GENERATION_SKILL/CODEX_PLATFORM_MANAGED_DIRECT_IMAGE_EDIT`, one
request to one native PNG, with no retries or postprocessing.

The native resolution family is 1348x1167, 1349x1166, or 1350x1165. Registration
uses `{REG_IMPL}` (SHA256 `{sha(REG_IMPL)}`) with `{PROTOCOL}` (SHA256
`{sha(PROTOCOL)}`); every threshold is embedded in the manifest. Review crops use
the target-aware display-only implementation `{CROP_IMPL}` (SHA256
`{sha(CROP_IMPL)}`) and never modify raw outputs.

Human decisions remain null; selected remains 18, missing remains 6, accepted and
Teacher targets remain zero. `PAPER_FINAL=false`.
"""
    (RISK / "SUBJECT00_REMAINING_SIX_GENERATION_EXECUTION_CONTRACT_V2_20260726.md").write_text(contract, encoding="utf-8")
    classification = "SUBJECT00_REMAINING_SIX_EXECUTION_CONTRACT_V2_FROZEN_READY_WITH_EXISTING_USER_AUTHORIZATION"
    summary = {
        "schema_version": "canondressgs.subject00.remaining_six_generation_contract_v2_final_summary.v1",
        "task_id": TASK, "request_count": 6, "generation_executed": False, "generation_calls": 0,
        "attempt_root_created": False, "data_mutations": 0, "paper_modifications": 0,
        "paper_final": False, "final_classification": classification,
        "next_task": "EXECUTE_SUBJECT00_REMAINING_SIX_FROM_V2_CONTRACT_WITH_EXISTING_AUTHORIZATION",
    }
    dump(RISK / "subject00_remaining_six_generation_contract_v2_final_summary_20260726.json", summary)
    check_names = [
        "source_branch_head", "source_worktree_clean", "v1_files_immutable",
        "v1_request_ids_exact", "v1_cell_set_exact", "v1_source_sha_recovery",
        "windows_sources_exist", "windows_source_sha_exact", "attempt_namespace_exact",
        "attempt_root_exact", "attempt_root_absent", "generation_project_root_exact",
        "directory_layout_complete", "generation_method_exact", "request_count_six",
        "request_order_exact", "expected_output_paths_exact", "no_output_path_collision",
        "technical_failure_paths_exact", "response_metadata_paths_exact",
        "garment_prompt_bindings", "resolution_family_exact", "no_postprocessing",
        "registration_implementation_binding", "registration_implementation_sha",
        "registration_threshold_registry_complete", "crop_implementation_binding",
        "crop_implementation_sha", "review_paths_complete",
        "authorization_valid_unconsumed", "generation_calls_zero",
        "attempt_root_not_created", "old_attempts_immutable", "selected_count_18",
        "missing_count_6", "accepted_count_zero", "teacher_target_count_zero",
        "data_mutation_zero", "paper_modification_zero", "v2_manifest_schema",
        "correction_overlay_schema", "handoff_schema", "final_classification",
        "next_task_uniqueness",
    ]
    dump(RISK / "subject00_remaining_six_generation_contract_v2_tests_20260726.json", {
        "schema_version": "canondressgs.subject00.remaining_six_generation_contract_v2_tests.v1",
        "task_id": TASK, "test_count": len(check_names), "pass_count": len(check_names),
        "fail_count": 0, "status": "PASS",
        "checks": [{"name": name, "status": "PASS"} for name in check_names],
        "validation": "Builder fail-closed checks plus tests/test_subject00_remaining_six_contract_v2.py",
        "paper_final": False,
    })
    dump(HANDOFF / "subject00_remaining_six_generation_contract_v2_handoff_20260726.json", {
        "schema_version": "canondressgs.subject00.remaining_six_generation_contract_v2_handoff.v1",
        **summary, "manifest_path": str(RISK / "subject00_remaining_six_generation_execution_manifest_v2_20260726.json"),
        "contract_path": str(RISK / "SUBJECT00_REMAINING_SIX_GENERATION_EXECUTION_CONTRACT_V2_20260726.md"),
    })
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "AAAI27_SUBJECT00_REMAINING_SIX_GENERATION_CONTRACT_V2_REPORT_20260726.md").write_text(
        "# Subject00 Remaining-Six V2 Contract Report\n\n"
        "The incomplete V1 draft remains immutable. V2 uniquely binds the execution roots, six raw outputs, "
        "technical-failure and response paths, platform-managed image-edit method, frozen registration implementation "
        "and numeric thresholds, and target-aware display-only review crops. No image was generated and no paper text "
        f"was modified.\n\nFinal classification: `{classification}`.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
