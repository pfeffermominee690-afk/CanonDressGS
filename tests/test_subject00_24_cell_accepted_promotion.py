import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / "paper_protocol/reviewer_risk"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_accepted_registry_preserves_audit_and_limitations():
    accepted = json.loads((RISK / "subject00_global_accepted_cell_registry_24of24_20260726.json").read_text())
    audit = json.loads((RISK / "subject00_24_cell_acceptance_audit_registry_20260726.json").read_text())
    assert accepted["total_accepted_cell_count"] == 24
    assert accepted["accepted_coverage_by_garment"] == {"O01": 8, "O03": 8, "O04": 8}
    assert accepted["accepted_without_limitation_count"] == 15
    assert accepted["accepted_with_limitation_count"] == 9
    assert len({(x["garment"], x["slot"]) for x in accepted["records"]}) == 24
    audit_map = {x["request_id"]: x for x in audit["records"]}
    for item in accepted["records"]:
        source = audit_map[item["request_id"]]
        assert Path(item["raw_path"]).is_file() and sha(item["raw_path"]) == item["raw_sha256"]
        assert item["acceptance_readiness_class"] == source["acceptance_readiness"]
        assert item["disclosed_limitation_codes"] == source["limitations"]
        assert item["selected_for_cell"] and item["accepted"]
        assert not item["teacher_target"]
        assert item["mask_status"] == "NOT_GENERATED"
    failures = [x for x in accepted["records"] if x["machine_registration_status"] != "REGISTERED_SIMILARITY_PASS_CANDIDATE"]
    assert len(failures) == 2 and all(x["human_override_status"] for x in failures)


def test_mask_contract_is_draft_only():
    manifest = json.loads((RISK / "subject00_24_cell_mask_generation_manifest_draft_20260726.json").read_text())
    summary = json.loads((RISK / "subject00_24_cell_accepted_promotion_summary_20260726.json").read_text())
    assert not manifest["mask_generation_authorized"]
    assert manifest["segmentation_method"] == "UNRESOLVED_PENDING_FORMAL_PREFLIGHT"
    assert manifest["person_mask_output_count_expected"] == 24
    assert manifest["garment_mask_output_count_expected"] == 24
    assert manifest["total_mask_output_count_expected"] == 48
    assert not Path(manifest["mask_output_root"]).exists()
    assert not manifest["teacher_target_creation_authorized"]
    assert summary["total_accepted_cell_count"] == 24
    assert summary["mask_generated_count"] == summary["teacher_target_count"] == 0
    assert summary["generation_calls"] == summary["mask_generation_calls"] == 0
    assert summary["final_classification"] == "SUBJECT00_24_OF_24_CELLS_ACCEPTED_MASK_GENERATION_PENDING_AUTHORIZATION"
    assert summary["next_task"] == "USER_AUTHORIZE_SUBJECT00_24_CELL_MASK_GENERATION_PREFLIGHT"
