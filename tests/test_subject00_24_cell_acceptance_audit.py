import hashlib
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / "paper_protocol/reviewer_risk"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_registry_and_readiness_are_evidence_backed():
    reg = json.loads((RISK / "subject00_24_cell_acceptance_audit_registry_20260726.json").read_text())
    assert reg["selected_cell_count"] == reg["raw_file_exist_count"] == reg["raw_parse_pass_count"] == 24
    assert reg["machine_registration_pass_count"] == 22
    assert reg["machine_registration_fail_count"] == 2
    assert reg["machine_registration_unavailable_count"] == 0
    assert reg["human_override_count"] == 2
    assert reg["acceptance_ready_cell_count"] + reg["acceptance_ready_with_limitation_count"] == 24
    assert reg["acceptance_blocked_cell_count"] == reg["severe_artifact_cell_count"] == 0
    assert reg["exact_duplicate_count"] == 0
    for item in reg["records"]:
        output, source = Path(item["raw_output_path"]), Path(item["source_path"])
        assert output.is_file() and source.is_file()
        assert sha(output) == item["raw_output_sha256"]
        assert sha(source) == item["source_sha256"]
        with Image.open(output) as image:
            image.verify()
        assert item["retry_count"] == item["postprocessing_count"] == 0
        assert item["accepted"] is item["teacher_target"] is False


def test_promotion_stays_unauthorized_and_review_complete():
    summary = json.loads((RISK / "subject00_24_cell_acceptance_audit_final_summary_20260726.json").read_text())
    readiness = json.loads((RISK / "subject00_24_cell_mask_teacher_readiness_20260726.json").read_text())
    assert summary["accepted_count"] == summary["teacher_target_count"] == 0
    assert not summary["accepted_promotion_authorized"]
    assert not summary["teacher_target_promotion_authorized"]
    assert not summary["mask_generation_authorized"]
    assert summary["generation_calls"] == summary["retry_calls"] == summary["postprocessing_calls"] == 0
    assert all(value == 0 for value in summary["attempt_mutations"].values())
    assert all(Path(path).is_file() for path in summary["review_package_files"])
    assert len(summary["review_package_files"]) == 10
    assert readiness["teacher_target_readiness"] == "READY_AFTER_ACCEPTED_PROMOTION_AND_MASK_GENERATION"
    assert summary["final_classification"] == "SUBJECT00_24_CELL_ACCEPTANCE_AUDIT_PASS_PENDING_USER_PROMOTION"
    assert summary["next_task"] == "USER_AUTHORIZE_SUBJECT00_24_CELL_ACCEPTED_PROMOTION"
