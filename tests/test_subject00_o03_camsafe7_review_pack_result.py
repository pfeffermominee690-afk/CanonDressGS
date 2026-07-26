from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
TASK_ID = "AAAI27-SUBJECT00-O03-CAMSAFE7-PROVISIONAL-TEACHER-REVIEW-PACK-001"
FINAL_CLASSIFICATION = (
    "SUBJECT00_O03_CAMSAFE7_PROVISIONAL_TEACHER_REVIEW_PACK_"
    "READY_FOR_USER_REVIEW"
)
NEXT_TASK = "USER_UPLOAD_AND_REVIEW_SUBJECT00_O03_CAMSAFE7_REVIEW_PAGES"
FILES = {
    "manifest": RISK
    / "subject00_O03_camsafe7_human_review_upload_manifest_20260727.json",
    "tests": RISK
    / "subject00_O03_camsafe7_review_pack_execution_tests_20260727.json",
    "summary": RISK
    / "subject00_O03_camsafe7_review_pack_final_summary_20260727.json",
    "handoff": ROOT
    / "project_control_handoff"
    / "subject00_O03_camsafe7_review_pack_handoff_20260727.json",
}


def load(name: str):
    return json.loads(FILES[name].read_text(encoding="utf-8"))


def test_required_git_artifacts_exist_and_parse():
    for path in FILES.values():
        assert path.is_file(), path
        value = json.loads(path.read_text(encoding="utf-8"))
        assert value.get("task_id", value.get("TASK_ID")) == TASK_ID
    assert (RISK / "SUBJECT00_O03_CAMSAFE7_REVIEW_PACK_REPORT_20260727.md").is_file()


def test_manifest_has_eleven_ordered_primary_pngs_and_one_pdf():
    manifest = load("manifest")
    assert manifest["review_png_count"] == 11
    assert len(manifest["review_pages"]) == 11
    assert [row["page_number"] for row in manifest["review_pages"]] == list(
        range(1, 12)
    )
    assert manifest["pdf"]["page_count"] == 11
    assert len(manifest["pdf"]["sha256"]) == 64
    assert manifest["excluded_slot04_page_number"] == 11


def test_all_primary_page_resolutions_and_hashes():
    manifest = load("manifest")
    expected = [
        {"width": 6400, "height": 4000},
        *[{"width": 6400, "height": 4800}] * 7,
        *[{"width": 6400, "height": 4000}] * 3,
    ]
    assert [row["resolution"] for row in manifest["review_pages"]] == expected
    assert all(row["bytes"] > 0 for row in manifest["review_pages"])
    assert all(len(row["sha256"]) == 64 for row in manifest["review_pages"])


def test_exact_seven_view_to_page_mapping():
    mapping = load("manifest")["view_page_mapping"]
    assert list(mapping) == [
        "slot_00",
        "slot_01",
        "slot_02",
        "slot_03",
        "slot_05",
        "slot_06",
        "slot_07",
    ]
    assert [mapping[slot]["page_number"] for slot in mapping] == list(range(2, 9))
    assert [mapping[slot]["sample_count"] for slot in mapping] == [
        172,
        172,
        172,
        171,
        171,
        171,
        171,
    ]


def test_every_per_view_page_declares_required_visual_coverage():
    pages = load("manifest")["review_pages"][1:8]
    required = {
        "target raw",
        "person mask",
        "garment mask",
        "Base60747 render",
        "clean Teacher7 render",
        "historical contaminated Teacher render",
        "target-clean difference",
        "garment crop",
        "protected crop",
        "silhouette/boundary overlay",
        "face/head crop",
        "hands crop",
        "feet crop",
        "per-view metrics",
    }
    assert all(required <= set(page["required_content"]) for page in pages)


def test_region_comparison_animation_and_quarantine_pages():
    pages = load("manifest")["review_pages"]
    assert pages[8]["role"] == "region_risk_summary"
    assert "metric-risk pointers" in pages[8]["required_content"]
    assert pages[9]["role"] == "clean_vs_contaminated"
    assert "historical contaminated8" in pages[9]["required_content"]
    assert pages[10]["role"] == "animation_and_quarantine"
    assert "slot04 original target" in pages[10]["required_content"]
    assert "slot04 sample_count=0" in pages[10]["required_content"]


def test_human_fields_remain_null_and_paper_ineligible():
    manifest = load("manifest")
    fields = manifest["human_fields"]
    assert fields["paper_eligible"] is False
    assert all(value is None for key, value in fields.items() if key != "paper_eligible")
    for row in manifest["view_page_mapping"].values():
        assert row["human_fields"] == fields


def test_all_runtime_checks_are_nonempty_unique_and_passing():
    tests = load("tests")
    assert tests["test_count"] >= 33
    assert tests["pass_count"] == tests["test_count"]
    assert tests["fail_count"] == 0
    assert tests["result"] == f"PASS_{tests['test_count']}_OF_{tests['test_count']}"
    assert all(row["passed"] is True for row in tests["tests"])
    assert len({row["name"] for row in tests["tests"]}) == tests["test_count"]


def test_immutability_and_zero_optimizer_contract():
    manifest = load("manifest")
    mutations = manifest["immutability"]
    assert manifest["optimizer_steps"] == 0
    assert mutations["before_equals_after"] is True
    for key in (
        "target_mutations",
        "mask_mutations",
        "base_checkpoint_mutations",
        "clean_teacher_checkpoint_mutations",
        "contaminated_run_mutations",
        "formal_base_run_mutations",
        "paper_modifications",
    ):
        assert mutations[key] == 0


def test_summary_final_classification_and_next_task():
    summary = load("summary")
    assert summary["TARGET_VIEW_COUNT"] == 7
    assert summary["EXCLUDED_VIEW_COUNT"] == 1
    assert summary["REVIEW_PNG_COUNT"] == 11
    assert summary["REVIEW_PDF_PAGE_COUNT"] == 11
    assert summary["HUMAN_VISUAL_DECISION"] is None
    assert summary["SCIENTIFIC_PASS"] is None
    assert summary["PAPER_ELIGIBLE"] is False
    assert summary["OPTIMIZER_STEPS"] == 0
    assert summary["PAPER_FINAL"] is False
    assert summary["FINAL_CLASSIFICATION"] == FINAL_CLASSIFICATION
    assert summary["NEXT_TASK"] == NEXT_TASK


def test_no_binary_review_assets_are_tracked_in_git():
    tracked = subprocess.check_output(
        ["git", "-C", str(ROOT), "ls-files"], text=True
    ).splitlines()
    assert not any(
        "human_review_upload_pack_20260727" in path
        and path.lower().endswith((".png", ".pdf", ".jpg", ".jpeg"))
        for path in tracked
    )
