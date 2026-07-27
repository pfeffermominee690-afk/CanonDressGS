import hashlib
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / "paper_protocol/reviewer_risk"
ATTEMPT = Path(r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-GENERATION-001\attempt_005_subject00_remaining_six_cell_generation")
STATE = json.loads((ATTEMPT / "07_logs/execution_state.json").read_text())
V2 = json.loads((RISK / "subject00_remaining_six_generation_execution_manifest_v2_20260726.json").read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_execution_contract_and_raw_outputs():
    assert STATE["generation_calls"] == 6
    assert STATE["retry_calls"] == STATE["postprocessing_calls"] == 0
    assert STATE["authorization_calls_consumed"] == 6
    assert STATE["authorization_calls_remaining"] == 0
    assert [x["request_id"] for x in STATE["records"]] == V2["execution_order"]
    assert len({x["output_path"] for x in STATE["records"]}) == 6
    assert len({x["output_sha256"] for x in STATE["records"]}) == 6
    for record in STATE["records"]:
        assert Path(record["output_path"]).is_file()
        assert sha(record["output_path"]) == record["output_sha256"]
        assert record["png_parse"] and record["resolution_family_pass"]
        assert record["raw_resolution"] == {"width": 1349, "height": 1166}
        assert record["no_postprocessing"]


def test_registration_review_and_boundaries():
    reg = json.loads((ATTEMPT / "06_audit/registration_results.json").read_text())
    review = json.loads((ATTEMPT / "05_review_assets/human_review_manifest.json").read_text())
    assert len(reg["records"]) == len(review["records"]) == 6
    assert sum(x["primary_classification"] == "REGISTERED_SIMILARITY_PASS_CANDIDATE" for x in reg["records"]) == 5
    assert Path(review["contact_sheet_path"]).is_file()
    for item in review["records"]:
        assert item["human_visual_decision"] is None
        assert item["garment_pass"] is item["identity_pass"] is None
        assert not item["selected_for_cell"] and not item["accepted"] and not item["teacher_target"]
        assert all(Path(x).is_file() for x in item["review_paths"].values())
    assert STATE["old_attempt_mutations"] == {
        "attempt_001": 0, "attempt_002": 0, "attempt_003": 0,
        "attempt_004_o03_hood_removal_targeted_canary": 0}
