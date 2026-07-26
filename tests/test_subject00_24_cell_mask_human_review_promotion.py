from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
HANDOFF = ROOT / "project_control_handoff"
EXPECTED_CLASSIFICATION = (
    "SUBJECT00_24_OF_24_MASKS_ACCEPTED_TEACHER_TARGET_PREFLIGHT_PENDING"
)
EXPECTED_NEXT_TASK = (
    "PREFLIGHT_AND_FREEZE_SUBJECT00_24_CELL_TEACHER_TARGET_CREATION_CONTRACT"
)
TEACHER_ROOT = Path(
    r"E:\model_train\canondressgs_work\datasets"
    r"\subject00_three_garment\09_teacher_targets"
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def test_human_review_overlay_and_mask_accepted_registry() -> None:
    overlay = load(RISK / "subject00_24_cell_mask_human_review_overlay_20260727.json")
    accepted = load(
        RISK / "subject00_global_mask_accepted_registry_24of24_20260727.json"
    )
    assert overlay["reviewer"] == "USER_AND_GPT_MANUAL_REVIEW"
    assert overlay["reviewed_cell_count"] == 24
    assert overlay["person_mask_human_pass_count"] == 24
    assert overlay["garment_mask_human_pass_count"] == 24
    assert overlay["mask_pair_human_pass_count"] == 24
    assert overlay["mask_human_blocked_count"] == 0
    assert len(overlay["records"]) == 24
    assert len(accepted["records"]) == 24
    assert accepted["mask_accepted_coverage_by_garment"] == {
        "O01": 8,
        "O03": 8,
        "O04": 8,
    }
    assert accepted["mask_accepted_without_limitation_count"] == 15
    assert accepted["mask_accepted_with_disclosed_source_limitation_count"] == 9
    assert accepted["machine_registration_pass_count"] == 22
    assert accepted["machine_registration_fail_count"] == 2
    assert accepted["human_override_count"] == 2
    assert all(record["mask_accepted"] for record in accepted["records"])
    assert not any(record["teacher_target"] for record in accepted["records"])
    assert all(
        record["person_mask_human_decision"]
        == record["garment_mask_human_decision"]
        == record["mask_pair_human_decision"]
        == "PASS"
        for record in accepted["records"]
    )
    records = {record["request_id"]: record for record in accepted["records"]}
    o03 = records["subject00_O03_slot04_canary_attempt004_cand00"]
    assert o03["machine_registration_status"] == "FAIL"
    assert (
        o03["machine_registration_classification"]
        == "AFFINE_OR_PROJECTIVE_RECOMPOSITION_FAIL"
    )
    assert o03["human_override_status"] == "PASS_VISUALLY_ACCEPTABLE_ALIGNMENT"
    o01 = records["subject00_O01_slot04_remaining_attempt005_cand00"]
    assert (
        o01["machine_registration_status"]
        == "AFFINE_OR_PROJECTIVE_RECOMPOSITION_FAIL"
    )
    assert (
        o01["human_override_status"]
        == "PASS_VISUALLY_ACCEPTABLE_SUBJECT_ALIGNMENT"
    )


def test_all_raw_and_mask_bindings_remain_exact() -> None:
    accepted = load(
        RISK / "subject00_global_mask_accepted_registry_24of24_20260727.json"
    )
    for record in accepted["records"]:
        resolution = (
            record["native_resolution"]["width"],
            record["native_resolution"]["height"],
        )
        for key in ("accepted_raw", "person_mask", "garment_mask"):
            binding = record[key]
            path = Path(binding["path"])
            assert path.is_file()
            assert path.stat().st_size == binding["bytes"]
            assert sha256(path) == binding["sha256"]
            with Image.open(path) as image:
                assert image.size == resolution
        assert record["technical_qa"]["person"]["status"] == "PASS"
        assert record["technical_qa"]["garment"]["status"] == "PASS"
        assert record["technical_qa"]["pair"]["status"] == "PASS"
        assert (
            record["technical_qa"]["pair"][
                "garment_outside_person_pixel_count"
            ]
            == 0
        )
        assert record["technical_qa"]["pair"]["protected_region_area"] > 0


def test_frozen_attempt_baseline_and_review_evidence_are_immutable() -> None:
    snapshot = load(
        RISK / "subject00_24_cell_mask_human_review_pack_pre_snapshot_20260726.json"
    )
    assert snapshot["baseline"]["file_count"] == 368
    for record in snapshot["baseline"]["records"]:
        path = Path(record["path"])
        assert path.is_file()
        assert path.stat().st_size == record["bytes"]
        assert sha256(path) == record["sha256"]
    summary = load(RISK / "subject00_24_cell_mask_human_review_summary_20260727.json")
    evidence = summary["review_evidence"]
    pdf = Path(evidence["pdf"]["path"])
    assert pdf.is_file()
    assert pdf.stat().st_size == evidence["pdf"]["bytes"] == 100056298
    assert (
        sha256(pdf)
        == evidence["pdf"]["sha256"]
        == "7f9da88729be097265b0a8e8843ef10c46105ca68c4474785bc88cc4fc4f0c7b"
    )
    upload_index = load(Path(evidence["upload_index"]["path"]))
    assert upload_index["review_page_count"] == 8
    for page in upload_index["pages"]:
        path = Path(page["path"])
        assert path.is_file()
        assert path.stat().st_size == page["bytes"]
        assert sha256(path) == page["sha256"]


def test_teacher_preflight_is_non_executable_and_preserves_limitations() -> None:
    manifest = load(
        RISK
        / "subject00_24_cell_teacher_target_preflight_manifest_draft_20260727.json"
    )
    assert manifest["status"] == "DRAFT_PREFLIGHT_ONLY_NOT_AUTHORIZED"
    assert manifest["accepted_raw_count"] == 24
    assert manifest["person_mask_count"] == 24
    assert manifest["garment_mask_count"] == 24
    assert manifest["teacher_target_count"] == 0
    assert not manifest["teacher_target_creation_authorized"]
    assert not manifest["teacher_endpoint_optimization_authorized"]
    assert not manifest["loader_contract"]["resize_allowed"]
    assert not manifest["loader_contract"]["reencode_allowed"]
    assert manifest["base_avatar_dependency"]["status"] == (
        "PENDING_SUBJECT00_FORMAL_BASE_FINALIZATION"
    )
    assert len(manifest["records"]) == 24
    assert sum(bool(record["limitation_codes"]) for record in manifest["records"]) == 9
    assert not any(record["teacher_target"] for record in manifest["records"])
    assert not TEACHER_ROOT.exists()
    assert manifest["storage_estimate"]["materialized_bytes_in_this_task"] == 0


def test_promotion_summary_tests_reports_and_handoff() -> None:
    promotion = load(
        RISK
        / "subject00_24_cell_mask_accepted_promotion_overlay_20260727.json"
    )
    summary = load(RISK / "subject00_24_cell_mask_human_review_summary_20260727.json")
    tests = load(RISK / "subject00_24_cell_mask_human_review_tests_20260727.json")
    handoff = load(
        HANDOFF
        / "subject00_24_cell_mask_accepted_teacher_target_preflight_handoff_20260727.json"
    )
    assert promotion["previous_mask_accepted_count"] == 0
    assert promotion["newly_mask_accepted_count"] == 24
    assert promotion["total_mask_accepted_count"] == 24
    assert summary["coverage"] == {"O01": 8, "O03": 8, "O04": 8}
    assert summary["mask_missing_count"] == summary["mask_blocked_count"] == 0
    assert summary["teacher_target_count"] == 0
    assert summary["segmentation_inference_calls"] == 0
    assert summary["model_forward_calls"] == 0
    assert summary["image_generation_calls"] == 0
    assert summary["data_mutations"] == summary["paper_modifications"] == 0
    assert summary["final_classification"] == EXPECTED_CLASSIFICATION
    assert summary["next_task"] == EXPECTED_NEXT_TASK
    assert tests["test_count"] == tests["pass_count"] == 47
    assert tests["fail_count"] == 0
    assert all(check["status"] == "PASS" for check in tests["checks"])
    assert handoff["final_classification"] == EXPECTED_CLASSIFICATION
    assert handoff["next_task"] == EXPECTED_NEXT_TASK
    assert not handoff["teacher_target_creation_authorized"]
    assert not handoff["teacher_endpoint_optimization_authorized"]
    assert (
        RISK / "SUBJECT00_24_CELL_MASK_HUMAN_REVIEW_REPORT_20260727.md"
    ).is_file()
    assert (
        RISK / "SUBJECT00_24_CELL_TEACHER_TARGET_PREFLIGHT_CONTRACT_20260727.md"
    ).is_file()
    assert (
        ROOT
        / "docs"
        / "PAPER"
        / "AAAI27_SUBJECT00_24_CELL_MASK_HUMAN_REVIEW_REPORT_20260727.md"
    ).is_file()
