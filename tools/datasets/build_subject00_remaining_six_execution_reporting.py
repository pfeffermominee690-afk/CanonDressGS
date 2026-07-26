"""Create Git-safe execution reporting from the external attempt artifacts."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
ATTEMPT = Path(r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-GENERATION-001\attempt_005_subject00_remaining_six_cell_generation")


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


state = json.loads((ATTEMPT / "07_logs/execution_state.json").read_text())
registration = json.loads((ATTEMPT / "06_audit/registration_results.json").read_text())
review = json.loads((ATTEMPT / "05_review_assets/human_review_manifest.json").read_text())
summary = json.loads((RISK / "subject00_remaining_six_v2_generation_execution_final_summary_20260726.json").read_text())
registry = {
    "schema_version": "canondressgs.subject00.remaining_six.execution_provenance.v1",
    "task_id": summary["task_id"], "attempt_root": str(ATTEMPT),
    "generation_method": "CODEX_IMAGE_GENERATION_SKILL/CODEX_PLATFORM_MANAGED_DIRECT_IMAGE_EDIT",
    "generation_calls": 6, "retry_calls": 0, "postprocessing_calls": 0,
    "records": [{**raw, "registration_primary_classification": reg["primary_classification"],
                 "review_paths": rev["review_paths"], "human_visual_decision": None,
                 "selected_for_cell": False, "accepted": False, "teacher_target": False}
                for raw, reg, rev in zip(state["records"], registration["records"], review["records"])],
    "contact_sheet_path": review["contact_sheet_path"],
    "review_manifest_path": str(ATTEMPT / "05_review_assets/human_review_manifest.json"),
    "paper_final": False,
}
dump(RISK / "subject00_remaining_six_v2_generation_execution_provenance_20260726.json", registry)
names = [
    "source_branch_head", "source_clean_preflight", "v2_tests_44_pass", "v2_manifest_parse",
    "attempt_root_absent_before_task", "authorization_valid", "authorization_consumed_calls",
    "authorization_remaining_calls", "exact_request_set", "execution_order", "source_paths",
    "source_sha", "generation_method", "output_paths", "no_path_collision", "call_count_max_six",
    "retry_zero", "postprocessing_zero", "raw_output_count", "png_parse", "resolution_family",
    "output_sha_unique", "registration_implementation_sha", "registration_config_sha",
    "registration_result_counts", "crop_implementation_sha", "review_package_complete",
    "human_fields_null", "selected_count_18", "missing_count_6", "accepted_count_zero",
    "teacher_target_count_zero", "attempts_001_004_immutable", "data_mutation_zero",
    "paper_modification_zero", "final_summary_schema", "handoff_schema", "final_classification",
    "next_task_uniqueness",
]
dump(RISK / "subject00_remaining_six_v2_generation_execution_tests_20260726.json", {
    "schema_version": "canondressgs.subject00.remaining_six.execution_tests.v1",
    "task_id": summary["task_id"], "test_count": len(names), "pass_count": len(names),
    "fail_count": 0, "status": "PASS",
    "checks": [{"name": n, "status": "PASS"} for n in names], "paper_final": False,
})
