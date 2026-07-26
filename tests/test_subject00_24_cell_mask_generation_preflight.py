from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
TASK_ID = "AAAI27-SUBJECT00-24-CELL-MASK-GENERATION-PREFLIGHT-001"
SOURCE_HEAD = "fdbe9e74104641b242d1c33242a362f8822c113f"
ATTEMPT_ROOT = Path(
    r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-MASKS-001"
    r"\attempt_001_subject00_24_cell_person_garment_masks"
)
WEIGHTS = Path(
    r"E:\data_pre\audit_subject02_layered_composite_v3"
    r"\mask_backend_closure_v3a\m\model.safetensors"
)
WEIGHTS_SHA = "8f86fd90c567afd4370b3cc3a7e81ed767a632b2832a738331af660acc0c4c68"


def load(name: str) -> dict:
    return json.loads((RISK / name).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def test_frozen_manifest_and_bindings() -> None:
    manifest = load(
        "subject00_24_cell_mask_generation_execution_manifest_20260726.json"
    )
    accepted = load("subject00_global_accepted_cell_registry_24of24_20260726.json")
    assert manifest["task_id"] == TASK_ID
    assert manifest["source_head"] == SOURCE_HEAD
    assert manifest["accepted_cell_count"] == 24
    assert manifest["accepted_without_limitation_count"] == 15
    assert manifest["accepted_with_limitation_count"] == 9
    assert len(manifest["records"]) == 24
    assert len({record["request_id"] for record in manifest["records"]}) == 24
    assert manifest["request_order"] == [
        record["request_id"] for record in manifest["records"]
    ]
    assert [record["garment"] for record in manifest["records"]].count("O01") == 8
    assert [record["garment"] for record in manifest["records"]].count("O03") == 8
    assert [record["garment"] for record in manifest["records"]].count("O04") == 8
    assert sha256(Path(manifest["accepted_registry_path"])) == manifest[
        "accepted_registry_sha256"
    ]
    accepted_map = {record["request_id"]: record for record in accepted["records"]}
    limited = []
    for record in manifest["records"]:
        source = accepted_map[record["request_id"]]
        raw = Path(record["accepted_raw_path"])
        condition = Path(record["source_condition_path"])
        assert raw.is_file() and condition.is_file()
        assert raw.stat().st_size == record["accepted_raw_bytes"]
        assert sha256(raw) == record["accepted_raw_sha256"] == source["raw_sha256"]
        assert (
            sha256(condition)
            == record["source_condition_sha256"]
            == source["source_condition_sha256"]
        )
        with Image.open(raw) as opened:
            opened.verify()
        with Image.open(raw) as opened:
            assert [opened.width, opened.height] == [
                record["native_resolution"]["width"],
                record["native_resolution"]["height"],
            ]
        prior = Path(record["source_reference_person_mask_path"])
        assert prior.is_file()
        assert sha256(prior) == record["source_reference_person_mask_sha256"]
        assert record["source_reference_person_mask_role"] == (
            "REVIEW_ONLY_NOT_GENERATION"
        )
        assert not Path(record["person_mask_output_path"]).exists()
        assert not Path(record["garment_mask_output_path"]).exists()
        assert record["person_mask_human_decision"] is None
        assert record["garment_mask_human_decision"] is None
        assert record["mask_pair_human_decision"] is None
        assert not record["mask_accepted"]
        if record["accepted_with_limitation"]:
            limited.append(record)
    assert len(limited) == 9
    assert all(
        not record["mask_risk_assessment"][
            "registration_override_used_for_generation"
        ]
        for record in limited
    )


def test_semantics_pipeline_qa_storage_and_boundaries() -> None:
    manifest = load(
        "subject00_24_cell_mask_generation_execution_manifest_20260726.json"
    )
    semantics = load("subject00_24_cell_mask_semantics_registry_20260726.json")
    pipeline = load(
        "subject00_24_cell_mask_pipeline_feasibility_audit_20260726.json"
    )
    quality = load("subject00_24_cell_mask_quality_gate_registry_20260726.json")
    storage = load(
        "subject00_24_cell_mask_generation_storage_estimate_20260726.json"
    )
    tests = load(
        "subject00_24_cell_mask_generation_preflight_tests_20260726.json"
    )
    summary = load(
        "subject00_24_cell_mask_generation_preflight_final_summary_20260726.json"
    )
    handoff = json.loads(
        (
            ROOT
            / "project_control_handoff"
            / "subject00_24_cell_mask_generation_execution_handoff_20260726.json"
        ).read_text(encoding="utf-8")
    )

    assert semantics["conflict_status"] == "NO_SEMANTIC_CONFLICT"
    person = semantics["person_foreground_mask"]
    garment = semantics["garment_region_mask"]
    loader = semantics["loader_semantics"]
    assert person["label_ids_included"] == list(range(1, 16)) + [17]
    assert person["label_ids_excluded"] == [0, 16]
    assert garment["label_ids_included"] == [4, 5, 6, 7, 8, 17]
    assert garment["garment_mask_must_be_person_subset"]
    assert loader["format"] == "PNG mode L"
    assert loader["bit_depth"] == 8
    assert loader["binary_value_set"] == [0, 255]
    assert loader["formal_subject00_resize"].startswith("FORBIDDEN")

    assert pipeline["selected_pipeline_id"] == manifest["segmentation_method_id"]
    assert pipeline["selected_method"] == "HUMAN_PARSING_WITH_GARMENT_LABEL_MAPPING"
    assert pipeline["person_mask_method_status"] == "READY"
    assert pipeline["garment_mask_method_status"].startswith("READY")
    assert pipeline["model"]["weights_path"] == str(WEIGHTS)
    assert sha256(WEIGHTS) == WEIGHTS_SHA
    assert pipeline["model"]["weights_sha256"] == WEIGHTS_SHA
    assert pipeline["model"]["license_status"].startswith("RESEARCH_ONLY")
    assert not pipeline["model"]["download_required"]
    assert not pipeline["model"]["network_required"]
    assert pipeline["environment"]["dependency_status"] == (
        "READY_NO_INSTALL_REQUIRED"
    )

    assert manifest["attempt_root"] == str(ATTEMPT_ROOT)
    assert not manifest["attempt_root_preexisted"]
    assert not manifest["attempt_root_created"]
    assert not ATTEMPT_ROOT.exists()
    assert manifest["person_mask_count_expected"] == 24
    assert manifest["garment_mask_count_expected"] == 24
    assert manifest["total_mask_count_expected"] == 48
    assert manifest["person_mask_count_generated"] == 0
    assert manifest["garment_mask_count_generated"] == 0
    assert not manifest["mask_generation_authorized"]
    assert manifest["model_download_bytes"] == 0
    assert manifest["environment_install_calls"] == 0
    assert manifest["mask_inference_calls"] == 0
    assert manifest["teacher_target_count"] == 0
    assert not manifest["teacher_target_creation_authorized"]
    assert not manifest["teacher_endpoint_optimization_authorized"]
    teacher = manifest["teacher_pipeline_compatibility"]
    assert teacher["status"].startswith("SCHEMA_COMPATIBLE")
    assert {
        "raw_path",
        "raw_sha256",
        "person_mask_path",
        "person_mask_sha256",
        "garment_mask_path",
        "garment_mask_sha256",
        "camera_id",
        "slot",
        "garment",
        "direction",
    }.issubset(teacher["required_future_target_registry_fields"])
    assert teacher["teacher_target_creation_remains_forbidden"]

    assert quality["limitation_propagation"]["status"] == "COMPLETE_9_OF_9"
    assert quality["limitation_propagation"]["expected_count"] == 9
    assert quality["review_package_schema"]["raw_person_overlays"] == 24
    assert quality["review_package_schema"]["raw_garment_overlays"] == 24
    assert quality["review_package_schema"]["person_vs_garment_composites"] == 24
    assert quality["cross_view_matrix_shape"]["cells"] == 24
    assert storage["storage_gate"] == "PASS"
    assert storage["target_disk_free_bytes"] > storage["total_projected_bytes"]

    assert tests["status"] == "PASS"
    assert tests["pass_count"] == tests["test_count"] >= 47
    assert tests["fail_count"] == 0
    assert summary["mask_inference_calls"] == 0
    assert summary["data_mutations"] == 0
    assert summary["paper_modifications"] == 0
    assert not summary["paper_final"]
    assert summary["final_classification"] == (
        "SUBJECT00_24_CELL_MASK_GENERATION_CONTRACT_READY_FOR_EXECUTION"
    )
    assert summary["next_task"] == (
        "EXECUTE_SUBJECT00_24_CELL_MASK_GENERATION_FROM_FROZEN_CONTRACT"
    )
    assert handoff["final_classification"] == summary["final_classification"]
    assert handoff["next_task"] == summary["next_task"]


def test_executor_is_local_only_and_uninvoked() -> None:
    executor = (
        ROOT
        / "tools"
        / "datasets"
        / "execute_subject00_24_cell_masks_from_frozen_contract.py"
    )
    source = executor.read_text(encoding="utf-8")
    assert "local_files_only=True" in source
    assert "EXECUTE_SUBJECT00_24_CELL_MASK_GENERATION_FROM_FROZEN_CONTRACT" in source
    assert "MASK_PAIR_GENERATED_QA_PENDING_HUMAN" in source
    assert "TECHNICAL_FAILURE_NO_RETRY" in source
    assert "no_candidate_masks_saved" in source
    assert "requests." not in source
    assert "http://" not in source and "https://" not in source
    assert not ATTEMPT_ROOT.exists()


def test_required_artifacts_exist() -> None:
    required = [
        RISK / "SUBJECT00_24_CELL_MASK_GENERATION_EXECUTION_CONTRACT_20260726.md",
        RISK / "subject00_24_cell_mask_generation_execution_manifest_20260726.json",
        RISK / "subject00_24_cell_mask_semantics_registry_20260726.json",
        RISK / "subject00_24_cell_mask_pipeline_feasibility_audit_20260726.json",
        RISK / "subject00_24_cell_mask_quality_gate_registry_20260726.json",
        RISK / "subject00_24_cell_mask_generation_storage_estimate_20260726.json",
        RISK / "subject00_24_cell_mask_generation_preflight_tests_20260726.json",
        RISK / "subject00_24_cell_mask_generation_preflight_final_summary_20260726.json",
        ROOT
        / "project_control_handoff"
        / "subject00_24_cell_mask_generation_execution_handoff_20260726.json",
        ROOT
        / "docs"
        / "PAPER"
        / "AAAI27_SUBJECT00_24_CELL_MASK_GENERATION_PREFLIGHT_REPORT_20260726.md",
    ]
    assert all(path.is_file() for path in required)
