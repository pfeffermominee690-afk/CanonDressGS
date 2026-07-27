import hashlib
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
V1 = json.loads((RISK / "subject00_remaining_six_cell_generation_manifest_draft_20260726.json").read_text())
V2 = json.loads((RISK / "subject00_remaining_six_generation_execution_manifest_v2_20260726.json").read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_contract_v2_frozen_bindings():
    assert V2["request_count"] == 6
    assert V2["execution_order"] == [r["request_id"] for r in V2["records"]]
    assert [r["request_id"] for r in V2["records"]] == [r["request_id"] for r in V1["requests"]]
    assert len({r["cell_descriptor"] for r in V2["records"]}) == 6
    assert len({r["expected_raw_output_path"] for r in V2["records"]}) == 6
    assert len({r["technical_failure_path"] for r in V2["records"]}) == 6
    assert len({r["response_metadata_path"] for r in V2["records"]}) == 6
    assert not Path(V2["attempt_root"]).exists()
    assert V2["attempt_root_created"] is False
    assert V2["generation_method"]["identifier"] == "CODEX_IMAGE_GENERATION_SKILL/CODEX_PLATFORM_MANAGED_DIRECT_IMAGE_EDIT"
    assert V2["accepted_resolution_set"] == [{"width": 1348, "height": 1167}, {"width": 1349, "height": 1166}, {"width": 1350, "height": 1165}]
    assert not any(V2[k] for k in ["raw_output_postprocessing_allowed", "resize_allowed", "crop_allowed", "padding_allowed", "outpaint_allowed", "reencoding_allowed"])
    reg = V2["registration_policy"]
    assert sha(reg["implementation_path"]) == reg["implementation_sha256"]
    assert sha(reg["config_path"]) == reg["config_sha256"]
    assert reg["thresholds"]["machine_similarity_gate"]["minimum_inlier_ratio"] == 0.65
    crop = V2["crop_policy"]
    assert sha(crop["implementation_path"]) == crop["implementation_sha256"]
    assert crop["raw_output_modified"] is False
    for old, new in zip(V1["requests"], V2["records"]):
        assert new["source_condition_sha256"] == old["source_condition_sha256"]
        assert sha(new["windows_source_condition_path"]) == old["source_condition_sha256"]
        with Image.open(new["windows_source_condition_path"]) as image:
            image.verify()
        assert new["human_visual_decision"] is None
        assert not new["selected_for_cell"] and not new["accepted"] and not new["teacher_target"]
        assert all(new["review_paths"].values())
    assert V2["authorization_consumed_calls"] == 0 and V2["authorization_remaining_calls"] == 6
    assert V2["generation_executed"] is False and V2["generation_calls"] == 0
    assert V2["total_selected_cell_count"] == 18 and V2["remaining_missing_cell_count"] == 6
    assert V2["accepted_count"] == 0 and V2["teacher_target_count"] == 0
    assert V2["paper_final"] is False


def test_target_aware_crop_hits_person_head():
    from tools.datasets.subject00_target_aware_review_crops import review_crop_boxes
    bbox = (559, 288, 748, 902)
    box = review_crop_boxes(bbox, 1330, 1150)["face_head"]
    assert box[0] <= bbox[0] < bbox[2] <= box[2]
    assert box[1] <= bbox[1] < box[3]
    assert box[3] < 700
