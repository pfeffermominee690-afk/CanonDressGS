import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RISK = REPO / "paper_protocol" / "reviewer_risk"
PROBLEM_IDS = {
    "subject00_O03_slot04_canary_attempt004_cand00",
    "subject00_O01_slot04_remaining_attempt005_cand00",
}


def load(name: str):
    return json.loads((RISK / name).read_text(encoding="utf-8"))


def test_camera_resolution_blocker_is_resolved_by_frozen_quarantine():
    summary = load(
        "subject00_teacher_target_camera_blocker_resolution_final_summary_20260727.json"
    )
    eligibility = load(
        "subject00_teacher_target_camera_eligibility_registry_20260727.json"
    )
    quarantine = load("subject00_teacher_target_quarantine_registry_20260727.json")
    assert summary["camera_safe_reference_count"] == 22
    assert summary["camera_salvaged_count"] == 0
    assert summary["camera_quarantine_count"] == 2
    assert summary["teacher_target_training_eligible_count"] == 22
    assert summary["teacher_target_review_only_count"] == 2
    assert summary["o03_camera_safe_view_count"] == 7
    assert len(eligibility["records"]) == 24
    assert {item["request_id"] for item in quarantine["records"]} == PROBLEM_IDS
    assert all(not item["training_eligible"] for item in quarantine["records"])
    assert all(not item["evaluation_eligible"] for item in quarantine["records"])
    assert all(not item["appearance_supervision_eligible"] for item in quarantine["records"])
    assert all(not item["geometry_supervision_eligible"] for item in quarantine["records"])


def test_problem_cells_used_only_allowed_models_and_failed_unique_binding():
    salvage = load("subject00_problem_cell_camera_salvage_results_20260727.json")
    allowed = set(salvage["allowed_models"])
    forbidden = set(salvage["forbidden_diagnostic_only_models"])
    assert len(salvage["records"]) == 2
    for record in salvage["records"]:
        assert set(record["candidate_models"]) == allowed
        assert not (set(record["candidate_models"]) & forbidden)
        assert record["selected_model"] is None
        assert record["selected_transform"] is None
        assert record["review_only"] is True
        assert record["forbidden_projective_or_affine_diagnostic_dominates"] is True


def test_camera_draft_never_materializes_quarantined_camera():
    draft = load("subject00_teacher_target_camera_records_draft_20260727.json")
    assert draft["draft_only"] is True
    assert draft["materialization_authorized"] is False
    assert draft["record_count"] == 24
    quarantined = [
        item for item in draft["records"] if item["review_only"]
    ]
    assert len(quarantined) == 2
    assert {item["request_id"] for item in quarantined} == PROBLEM_IDS
    assert all(item["camera_binding_status"] == "UNRESOLVED_HUMAN_OVERRIDE" for item in quarantined)
    assert all(item["target_record_status"] == "REVIEW_ONLY_CAMERA_QUARANTINED" for item in quarantined)
    assert all(item["target_K"] is None for item in quarantined)
    assert all(item["registered_source_to_target_transform"] is None for item in quarantined)
    assert all(item["materialized"] is False for item in draft["records"])
    assert all(item["teacher_target"] is False for item in draft["records"])


def test_reference_distribution_and_structured_checks_are_complete():
    reference = load("subject00_22cell_camera_safe_reference_distribution_20260727.json")
    tests = load(
        "subject00_teacher_target_camera_blocker_resolution_tests_20260727.json"
    )
    assert reference["reference_cell_count"] == 22
    assert len(reference["records"]) == 22
    for stats in reference["distribution"].values():
        assert set(("median", "mad", "p90", "p95", "minimum", "maximum")) <= set(stats)
    assert tests["check_count"] == tests["pass_count"]
    assert tests["fail_count"] == 0
    assert all(item["status"] == "PASS" for item in tests["checks"])
