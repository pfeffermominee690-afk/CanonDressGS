import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / "paper_protocol/reviewer_risk"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_human_review_and_24_cell_completion():
    overlay = json.loads((RISK / "subject00_remaining_six_human_review_overlay_20260726.json").read_text())
    selected = json.loads((RISK / "subject00_global_selected_cell_registry_24of24_20260726.json").read_text())
    missing = json.loads((RISK / "subject00_global_missing_cell_registry_0of24_20260726.json").read_text())
    assert overlay["reviewed_request_count"] == overlay["human_pass_count"] == 6
    assert overlay["machine_registration_pass_count"] == 5
    assert overlay["machine_registration_fail_count"] == overlay["human_override_count"] == 1
    assert sum(x["selected_for_cell"] for x in overlay["records"]) == 6
    assert all(not x["accepted"] and not x["teacher_target"] for x in overlay["records"])
    slot04 = next(x for x in overlay["records"] if "O01_slot04" in x["request_id"])
    assert slot04["machine_registration_primary_classification"] == "AFFINE_OR_PROJECTIVE_RECOMPOSITION_FAIL"
    assert slot04["machine_registration_human_override"] == "PASS_VISUALLY_ACCEPTABLE_SUBJECT_ALIGNMENT"
    assert selected["total_selected_cells"] == selected["total_cell_count"] == 24
    assert selected["selection_counts_by_garment"] == {"O01": 8, "O03": 8, "O04": 8}
    assert len({x["cell_key"] for x in selected["records"]}) == 24
    assert all(Path(x["output_path"]).is_file() and sha(x["output_path"]) == x["output_sha256"] for x in selected["records"])
    assert missing["remaining_missing_cell_count"] == 0 and missing["records"] == []
    assert selected["accepted_count"] == selected["teacher_target_count"] == 0


def test_acceptance_audit_stays_unauthorized():
    manifest = json.loads((RISK / "subject00_24_cell_acceptance_audit_manifest_draft_20260726.json").read_text())
    summary = json.loads((RISK / "subject00_24_cell_selection_completion_summary_20260726.json").read_text())
    assert not manifest["acceptance_audit_authorized"]
    assert not manifest["accepted_promotion_authorized"]
    assert not manifest["teacher_target_promotion_authorized"]
    assert summary["generation_calls"] == summary["retry_calls"] == summary["postprocessing_calls"] == 0
    assert summary["final_classification"] == "SUBJECT00_24_OF_24_CELLS_SELECTED_ACCEPTANCE_AUDIT_PENDING_AUTHORIZATION"
    assert summary["next_task"] == "USER_AUTHORIZE_SUBJECT00_24_CELL_ACCEPTANCE_AUDIT"
