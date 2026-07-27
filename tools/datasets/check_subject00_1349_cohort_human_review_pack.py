#!/usr/bin/env python3
"""Independently check the corrected Subject00 1349 cohort review pack."""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from PIL import Image
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[2]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
SOURCE_WORKTREE = Path(r"E:\model_train\canondressgs_subject00_attempt001_native_landscape_registration_audit")
ATTEMPTS_ROOT = Path(r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-GENERATION-001")
ATTEMPTS = {
    "attempt_001": ATTEMPTS_ROOT / "attempt_001",
    "attempt_002": ATTEMPTS_ROOT / "attempt_002_portrait_canary",
    "attempt_003": ATTEMPTS_ROOT / "attempt_003_portrait_canary_v2_registered_outpaint",
}
OLD_AUDIT_ROOT = ATTEMPTS_ROOT / "attempt_001_native_landscape_registration_audit"
OUTPUT_ROOT = OLD_AUDIT_ROOT / "08_corrected_1349_human_review"

TASK_ID = "AAAI27-SUBJECT00-1349x1166-COHORT-CORRECTION-HUMAN-REVIEW-PREP-001"
SOURCE_BRANCH = "research/subject00-attempt001-native-landscape-registration-audit-20260726"
SOURCE_HEAD = "e8af4f852fb08021d4dd7ae070bf196c7bedc589"
BRANCH = "research/subject00-1349-cohort-correction-human-review-prep-20260726"
TARGET_RESOLUTION = "1349x1166"
PASS_CLASSIFICATION = "REGISTERED_SIMILARITY_PASS_CANDIDATE"
FINAL_CLASSIFICATION = "SUBJECT00_1349_COHORT_CORRECTED_READY_FOR_HIGH_RES_USER_REVIEW"
NEXT_TASK = "USER_REVIEW_CORRECTED_1349x1166_DOMINANT_COHORT"
GARMENTS = ("O01", "O03", "O04")
SLOTS = tuple(f"slot_{index:02d}" for index in range(8))
EXPECTED_MISSING = [
    "O01/slot_04", "O01/slot_07", "O03/slot_06", "O03/slot_07",
    "O04/slot_05", "O04/slot_06",
]
HYPOTHESIZED_MISSING = [
    "O01/slot_04", "O01/slot_07", "O03/slot_06", "O04/slot_01",
    "O04/slot_05", "O04/slot_06",
]

PROTOCOL_PATH = RISK / "subject00_attempt001_1349_cohort_correction_review_protocol_20260726.json"
COHORTS_PATH = RISK / "corrected_attempt001_exact_resolution_cohorts.json"
COVERAGE_PATH = RISK / "corrected_attempt001_1349x1166_cell_coverage.json"
CANDIDATES_PATH = RISK / "corrected_attempt001_1349x1166_candidate_registry.json"
REVIEW_PATH = RISK / "corrected_1349x1166_human_review_manifest.json"
OVERLAY_PATH = RISK / "subject00_attempt001_1349_cohort_correction_overlay_20260726.json"
PACK_PATH = RISK / "subject00_attempt001_1349_high_res_review_pack_registry_20260726.json"
SUMMARY_PATH = RISK / "subject00_attempt001_1349_cohort_correction_final_summary_20260726.json"
REPORT_PATH = RISK / "SUBJECT00_ATTEMPT001_1349_COHORT_CORRECTION_REPORT_20260726.md"
HANDOFF_PATH = ROOT / "project_control_handoff" / "subject00_attempt001_1349_cohort_correction_human_review_prep_handoff_20260726.json"
TESTS_PATH = RISK / "subject00_attempt001_1349_cohort_correction_tests_20260726.json"
EXTERNAL_TESTS_PATH = OUTPUT_ROOT / "subject00_attempt001_1349_cohort_correction_tests_20260726.json"
BASELINE_PATH = OLD_AUDIT_ROOT / "00_provenance" / "attempt_immutability_baseline.json"

DUAL_PATHS = {
    COHORTS_PATH: OUTPUT_ROOT / "registries" / "corrected_attempt001_exact_resolution_cohorts.json",
    COVERAGE_PATH: OUTPUT_ROOT / "registries" / "corrected_attempt001_1349x1166_cell_coverage.json",
    CANDIDATES_PATH: OUTPUT_ROOT / "registries" / "corrected_attempt001_1349x1166_candidate_registry.json",
    REVIEW_PATH: OUTPUT_ROOT / "corrected_1349x1166_human_review_manifest.json",
    OVERLAY_PATH: OUTPUT_ROOT / "registries" / "subject00_attempt001_1349_cohort_correction_overlay_20260726.json",
    PACK_PATH: OUTPUT_ROOT / "subject00_attempt001_1349_high_res_review_pack_registry_20260726.json",
    SUMMARY_PATH: OUTPUT_ROOT / "subject00_attempt001_1349_cohort_correction_final_summary_20260726.json",
}


def canonical_sha256(value: Any) -> str:
    if isinstance(value, dict):
        value = dict(value)
        value.pop("content_sha256", None)
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    value = dict(payload)
    value.pop("content_sha256", None)
    value["content_sha256"] = canonical_sha256(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def run(*command: str, cwd: Path = ROOT) -> str:
    return subprocess.run(
        list(command), cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def assert_true(value: Any, message: str) -> None:
    if not value:
        raise AssertionError(message)


def assert_equal(actual: Any, expected: Any, message: str) -> None:
    if actual != expected:
        raise AssertionError(f"{message}: {actual!r} != {expected!r}")


def current_inventory(root: Path) -> dict[str, tuple[int, str]]:
    return {
        item.relative_to(root).as_posix(): (item.stat().st_size, file_sha256(item))
        for item in sorted(path for path in root.rglob("*") if path.is_file())
    }


def frozen_inventory(snapshot: dict[str, Any]) -> dict[str, tuple[int, str]]:
    return {item["relative_path"]: (item["bytes"], item["sha256"]) for item in snapshot["files"]}


def parse_png(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    assert_true(header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR", f"bad PNG {path}")
    width, height = struct.unpack(">II", header[16:24])
    with Image.open(path) as image:
        assert_equal(image.size, (width, height), f"PNG size {path}")
        image.verify()
    return width, height


def changed_paths() -> list[str]:
    tracked = run("git", "diff", "--name-only", SOURCE_HEAD, "--").splitlines()
    untracked = run("git", "ls-files", "--others", "--exclude-standard").splitlines()
    return sorted(set(path for path in tracked + untracked if path))


def verify_content_hashes(paths: list[Path]) -> None:
    for path in paths:
        payload = load_json(path)
        assert_equal(payload.get("content_sha256"), canonical_sha256(payload), f"content hash {path.name}")


def pdf_text(path: Path) -> list[str]:
    return [(page.extract_text() or "") for page in PdfReader(str(path)).pages]


def main() -> int:
    protocol = load_json(PROTOCOL_PATH)
    cohorts = load_json(COHORTS_PATH)
    coverage = load_json(COVERAGE_PATH)
    candidates = load_json(CANDIDATES_PATH)
    review = load_json(REVIEW_PATH)
    overlay = load_json(OVERLAY_PATH)
    pack = load_json(PACK_PATH)
    summary = load_json(SUMMARY_PATH)
    handoff = load_json(HANDOFF_PATH)
    baseline = load_json(BASELINE_PATH)
    records = {item["request_id"]: item for item in candidates["records"]}
    review_candidates = {item["request_id"]: item for item in candidates["review_candidates"]}
    coverage_by_cell = {item["cell_key"]: item for item in coverage["records"]}
    review_records = {item["request_id"]: item for item in review["records"]}
    paths = changed_paths()
    checks: list[dict[str, Any]] = []

    def check(name: str, function: Callable[[], None]) -> None:
        try:
            function()
            checks.append({"name": name, "status": "PASS"})
        except Exception as error:  # noqa: BLE001 - preserve every contract failure
            checks.append({"name": name, "status": "FAIL", "detail": f"{type(error).__name__}: {error}"})

    check("source_branch", lambda: assert_equal(run("git", "branch", "--show-current", cwd=SOURCE_WORKTREE), SOURCE_BRANCH, "source branch"))
    check("source_head", lambda: assert_equal(run("git", "rev-parse", "HEAD", cwd=SOURCE_WORKTREE), SOURCE_HEAD, "source HEAD"))
    check("source_worktree_clean", lambda: assert_equal(run("git", "status", "--short", cwd=SOURCE_WORKTREE), "", "source dirty"))
    check("target_branch", lambda: assert_equal(run("git", "branch", "--show-current"), BRANCH, "target branch"))
    check("source_ancestor", lambda: assert_true(
        subprocess.run(["git", "merge-base", "--is-ancestor", SOURCE_HEAD, "HEAD"], cwd=ROOT).returncode == 0,
        "source is not ancestor",
    ))
    check("attempt_001_immutable", lambda: assert_equal(current_inventory(ATTEMPTS["attempt_001"]), frozen_inventory(baseline["attempt_001"]), "attempt_001 changed"))
    check("attempt_002_immutable", lambda: assert_equal(current_inventory(ATTEMPTS["attempt_002"]), frozen_inventory(baseline["attempt_002"]), "attempt_002 changed"))
    check("attempt_003_immutable", lambda: assert_equal(current_inventory(ATTEMPTS["attempt_003"]), frozen_inventory(baseline["attempt_003"]), "attempt_003 changed"))

    check("task_ids", lambda: assert_true(all(
        item["task_id"] == TASK_ID for item in (protocol, cohorts, coverage, candidates, review, overlay, pack, summary, handoff)
    ), "task ID mismatch"))
    check("request_count_48", lambda: assert_equal(candidates["record_count"], 48, "request count"))
    check("request_id_set", lambda: assert_equal(
        sorted(records),
        sorted(f"subject00_{garment}_slot{slot[-2:]}_cand{candidate:02d}" for garment in GARMENTS for slot in SLOTS for candidate in range(2)),
        "request IDs",
    ))

    actual_sizes: dict[str, tuple[int, int]] = {}

    def verify_pngs() -> None:
        for request_id, record in records.items():
            path = Path(record["output_path"])
            actual_sizes[request_id] = parse_png(path)
            assert_equal(file_sha256(path), record["output_sha256"], f"output SHA {request_id}")

    check("png_parse_48", verify_pngs)
    check("actual_png_dimensions", lambda: assert_true(all(
        actual_sizes[request_id] == (record["actual_width"], record["actual_height"])
        and record["actual_resolution"] == f"{actual_sizes[request_id][0]}x{actual_sizes[request_id][1]}"
        for request_id, record in records.items()
    ), "actual PNG dimension mismatch"))
    check("manifest_binding_48", lambda: assert_true(all(
        record["provenance_binding_status"] == "PASS" and all(record["binding_checks"].values())
        for record in records.values()
    ), "manifest binding failure"))
    check("unique_outputs", lambda: assert_true(
        len({record["output_sha256"] for record in records.values()}) == 48
        and all(record["duplicate_status"] == "UNIQUE" for record in records.values()),
        "duplicate output",
    ))
    check("postprocessing_evidence_preserved", lambda: assert_true(all(
        record["postprocessing_count"] == "NOT_RECORDED"
        and not record["hidden_postprocessing_replacement_detected"]
        and record["no_postprocessing_or_replacement_evidence"]
        for record in records.values()
    ), "postprocessing evidence changed"))

    distribution = Counter(record["actual_resolution"] for record in records.values())
    check("resolution_distribution", lambda: assert_equal(distribution, Counter(cohorts["exact_resolution_distribution"]), "resolution distribution"))
    check("exact_1349_output_count", lambda: assert_true(
        distribution[TARGET_RESOLUTION] == cohorts["dominant_output_count"] == summary["recomputed_1349_output_count"] == 34,
        "1349 output count",
    ))
    raw_cells = {record["cell_key"] for record in records.values() if record["actual_resolution"] == TARGET_RESOLUTION}
    pass_records = [record for record in records.values() if record["actual_resolution"] == TARGET_RESOLUTION and record["is_similarity_pass_candidate"]]
    pass_cells = {record["cell_key"] for record in pass_records}
    check("exact_1349_raw_cell_coverage", lambda: assert_true(
        len(raw_cells) == cohorts["dominant_raw_cell_coverage"] == coverage["raw_cell_coverage"] == 20,
        "raw cell coverage",
    ))
    check("machine_pass_filter", lambda: assert_true(
        len(pass_records) == cohorts["dominant_machine_pass_candidate_count"] == coverage["machine_pass_candidate_count"] == 31
        and all(record["primary_machine_classification"] == PASS_CLASSIFICATION for record in pass_records),
        "machine-pass filtering",
    ))
    check("machine_pass_cell_coverage", lambda: assert_true(
        len(pass_cells) == cohorts["dominant_machine_pass_cell_coverage"] == coverage["machine_pass_cell_coverage"] == 18,
        "machine-pass cell coverage",
    ))

    check("O03_slot07_cand01_adjudication", lambda: assert_true(
        actual_sizes["subject00_O03_slot07_cand01"] == (1350, 1165)
        and records["subject00_O03_slot07_cand01"]["primary_machine_classification"] == PASS_CLASSIFICATION
        and not records["subject00_O03_slot07_cand01"]["belongs_to_1349x1166_cohort"],
        "O03/slot07/cand01 adjudication",
    ))
    check("O04_slot01_cand00_adjudication", lambda: assert_true(
        actual_sizes["subject00_O04_slot01_cand00"] == (1349, 1166)
        and records["subject00_O04_slot01_cand00"]["primary_machine_classification"] == PASS_CLASSIFICATION
        and records["subject00_O04_slot01_cand00"]["review_eligible"],
        "O04/slot01/cand00 adjudication",
    ))
    check("O04_slot01_cand01_adjudication", lambda: assert_true(
        actual_sizes["subject00_O04_slot01_cand01"] == (1349, 1166)
        and records["subject00_O04_slot01_cand01"]["primary_machine_classification"] == PASS_CLASSIFICATION
        and records["subject00_O04_slot01_cand01"]["review_eligible"],
        "O04/slot01/cand01 adjudication",
    ))
    check("old_vs_corrected_gap_comparison", lambda: assert_true(
        overlay["old_missing_cell_list"] == coverage["missing_cell_list"] == summary["corrected_missing_cell_list"] == EXPECTED_MISSING
        and overlay["hypothesized_missing_cell_list"] == HYPOTHESIZED_MISSING
        and overlay["correction_adjudication"] == "OLD_REGISTRY_CONFIRMED_CORRECT_PROPOSED_O03_O04_SWAP_REJECTED",
        "gap adjudication mismatch",
    ))
    check("no_evidence_conflict", lambda: assert_true(
        not overlay["cohort_registry_evidence_conflict"]
        and not overlay["underlying_json_was_wrong"]
        and not overlay["registry_generation_code_was_wrong"],
        "evidence conflict recorded",
    ))
    check("old_report_preserved", lambda: assert_true(
        file_sha256(Path(overlay["old_report_path"])) == overlay["old_report_sha256"]
        and file_sha256(Path(overlay["old_final_summary_path"])) == overlay["old_final_summary_sha256"]
        and file_sha256(Path(overlay["old_coverage_registry_path"])) == overlay["old_coverage_registry_sha256"]
        and file_sha256(Path(overlay["old_contact_sheet_path"])) == overlay["old_contact_sheet_sha256"]
        and overlay["old_report_mutations"] == 0,
        "old evidence changed",
    ))

    expected_review_ids = {record["request_id"] for record in pass_records if record["review_eligible"]}
    check("review_candidate_completeness", lambda: assert_true(
        set(review_candidates) == set(review_records) == expected_review_ids
        and candidates["review_candidate_count"] == review["record_count"] == pack["review_candidate_count"] == 31,
        "review candidate completeness",
    ))
    check("review_cell_count", lambda: assert_true(
        len({item["cell_key"] for item in review_candidates.values()})
        == candidates["review_cell_count"] == review["cell_count"] == pack["review_cell_count"] == 18,
        "review cell count",
    ))
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in review_candidates.values():
        by_cell[item["cell_key"]].append(item)
    check("multi_candidate_cell_count", lambda: assert_true(
        sum(len(items) == 2 for items in by_cell.values())
        == candidates["multi_candidate_cell_count"] == pack["multi_candidate_cell_count"] == 13,
        "multi-candidate cell count",
    ))
    check("no_cand00_representative_bias", lambda: assert_true(all(
        {item["request_id"] for item in by_cell[cell]}
        == {record["request_id"] for record in pass_records if record["cell_key"] == cell}
        for cell in by_cell
    ), "candidate omitted from multi-candidate cell"))
    check("coverage_24_cells", lambda: assert_true(
        coverage["cell_count"] == len(coverage_by_cell) == 24
        and set(coverage_by_cell) == {f"{garment}/{slot}" for garment in GARMENTS for slot in SLOTS},
        "coverage cell set",
    ))
    check("coverage_missing_flags", lambda: assert_true(all(
        item["missing_1349_pass_candidate"] == (item["cell_key"] in EXPECTED_MISSING)
        and item["pass_candidate_count_1349x1166"] == len(by_cell.get(item["cell_key"], []))
        and item["human_review_status"] is None
        for item in coverage["records"]
    ), "coverage missing flag"))

    required_assets = {
        "original_condition_full_frame", "generated_output_full_frame", "similarity_aligned_overlay",
        "background_edge_overlay", "full_body_source_output_pair", "face_head_source_output_pair",
        "left_hand_source_output_pair", "right_hand_source_output_pair", "feet_source_output_pair",
        "upper_garment_boundary_pair", "lower_garment_boundary_pair",
        "official_source_mask_and_transformed_expected_silhouette", "high_error_background_regions",
    }
    check("candidate_asset_schema", lambda: assert_true(all(
        set(item["review_assets"]) == required_assets and item["review_asset_count"] == 13
        for item in review_candidates.values()
    ), "candidate asset schema"))

    def verify_review_assets() -> None:
        for item in review_candidates.values():
            for path_value in item["review_assets"].values():
                path = Path(path_value)
                assert_true(path.is_file(), f"missing review asset {path}")
                with Image.open(path) as image:
                    assert_true(image.width >= 100 and image.height >= 100, f"undersized review asset {path}")
                    image.verify()

    check("candidate_assets_readable", verify_review_assets)
    check("face_crop_completeness", lambda: assert_true(all(Path(item["review_assets"]["face_head_source_output_pair"]).is_file() for item in review_candidates.values()), "face crop missing"))
    check("hand_crop_completeness", lambda: assert_true(all(
        Path(item["review_assets"]["left_hand_source_output_pair"]).is_file()
        and Path(item["review_assets"]["right_hand_source_output_pair"]).is_file()
        for item in review_candidates.values()
    ), "hand crop missing"))
    check("feet_crop_completeness", lambda: assert_true(all(Path(item["review_assets"]["feet_source_output_pair"]).is_file() for item in review_candidates.values()), "feet crop missing"))
    check("garment_crop_completeness", lambda: assert_true(all(
        Path(item["review_assets"]["upper_garment_boundary_pair"]).is_file()
        and Path(item["review_assets"]["lower_garment_boundary_pair"]).is_file()
        for item in review_candidates.values()
    ), "garment crop missing"))
    check("silhouette_error_assets", lambda: assert_true(all(
        Path(item["review_assets"]["official_source_mask_and_transformed_expected_silhouette"]).is_file()
        and Path(item["review_assets"]["high_error_background_regions"]).is_file()
        for item in review_candidates.values()
    ), "silhouette/error asset missing"))

    pdf_paths = {key: Path(value["path"]) for key, value in pack["pdfs"].items()}
    pdf_texts = {key: pdf_text(path) for key, path in pdf_paths.items()}
    check("pdf_files_and_shas", lambda: assert_true(all(
        path.is_file() and file_sha256(path) == pack["pdfs"][key]["sha256"] and path.stat().st_size == pack["pdfs"][key]["bytes"]
        for key, path in pdf_paths.items()
    ), "PDF file/SHA mismatch"))
    check("garment_pdf_page_counts", lambda: assert_true(all(
        len(pdf_texts[garment]) == pack["pdfs"][garment]["page_count"] == 8 for garment in GARMENTS
    ), "garment PDF page count"))
    check("high_risk_pdf_page_count", lambda: assert_true(
        len(pdf_texts["high_risk"]) == pack["pdfs"]["high_risk"]["page_count"] == 31,
        "high-risk PDF page count",
    ))

    def verify_garment_pages() -> None:
        for garment in GARMENTS:
            for page_index, slot in enumerate(SLOTS):
                text = pdf_texts[garment][page_index]
                cell = f"{garment}/{slot}"
                assert_true(cell in text and "HUMAN REVIEW STATUS: null" in text, f"page header {cell}")
                items = sorted(by_cell.get(cell, []), key=lambda item: item["candidate_id"])
                if not items:
                    assert_true("NO_1349_PASS_CANDIDATE" in text and "No placeholder image was inserted" in text, f"missing page {cell}")
                else:
                    for item in items:
                        assert_true(item["request_id"] in text and "REVIEW DECISION: null" in text, f"candidate page {item['request_id']}")

    check("garment_pdf_page_completeness", verify_garment_pages)
    check("dual_candidates_same_page", lambda: assert_true(all(
        review_records[item[0]["request_id"]]["garment_review_pdf_page"]
        == review_records[item[1]["request_id"]]["garment_review_pdf_page"]
        and item[0]["request_id"] in pdf_texts[item[0]["garment"]][review_records[item[0]["request_id"]]["garment_review_pdf_page"] - 1]
        and item[1]["request_id"] in pdf_texts[item[1]["garment"]][review_records[item[1]["request_id"]]["garment_review_pdf_page"] - 1]
        for item in (sorted(items, key=lambda value: value["candidate_id"]) for items in by_cell.values() if len(items) == 2)
    ), "dual candidates not on same page"))
    check("high_risk_order", lambda: assert_true(all(
        request_id in pdf_texts["high_risk"][index]
        and f"HIGH-RISK REVIEW PRIORITY {index + 1:02d}/31" in pdf_texts["high_risk"][index]
        and "REVIEW STATUS: null" in pdf_texts["high_risk"][index]
        for index, request_id in enumerate(candidates["risk_order"])
    ), "high-risk order mismatch"))
    check("metadata_completeness", lambda: assert_true(all(
        all(token in pdf_texts[item["garment"]][int(item["slot"][-2:])] for token in (
            "output resolution:", "output SHA-256:", "uniform scale:", "tx / ty:", "rotation:",
            "median / p95 reprojection:", "RANSAC inlier ratio:", "machine classification:", "exact cohort: True",
        ))
        for item in review_candidates.values()
    ), "metadata panel incomplete"))

    sheet_path = Path(pack["corrected_coverage_sheet"]["path"])
    check("coverage_sheet_readable", lambda: assert_true(
        sheet_path.is_file() and parse_png(sheet_path) == tuple(pack["corrected_coverage_sheet"]["dimensions"])
        and file_sha256(sheet_path) == pack["corrected_coverage_sheet"]["sha256"],
        "coverage sheet mismatch",
    ))
    check("pack_asset_tree_hash", lambda: assert_equal(
        canonical_sha256([{
            "relative_path": path.relative_to(OUTPUT_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        } for path in sorted(path for path in (OUTPUT_ROOT / "assets").rglob("*") if path.is_file())]),
        pack["candidate_asset_tree_sha256"],
        "asset tree hash",
    ))
    check("pack_asset_count", lambda: assert_equal(
        len([path for path in (OUTPUT_ROOT / "assets").rglob("*") if path.is_file()]),
        pack["candidate_asset_file_count"],
        "asset file count",
    ))

    null_fields = {
        "identity_consistency", "face_consistency", "pose_preservation", "camera_direction",
        "single_person", "hands_complete", "feet_complete", "garment_correctness",
        "garment_boundary_quality", "silhouette_quality", "background_geometry",
        "visible_artifacts", "human_decision", "rejection_reasons", "selected_for_cell",
    }
    check("human_fields_null", lambda: assert_true(all(
        all(item[field] is None for field in null_fields) for item in review_records.values()
    ), "human field was prefilled"))
    check("human_decision_non_null_zero", lambda: assert_true(
        review["human_decision_non_null_count"] == summary["human_decision_non_null_count"] == 0,
        "human decision count nonzero",
    ))
    check("accepted_zero", lambda: assert_true(
        review["accepted_count"] == pack["accepted_count"] == summary["accepted_count"] == 0
        and not any(item["accepted"] for item in review_records.values()),
        "accepted count nonzero",
    ))
    check("teacher_target_zero", lambda: assert_true(
        review["teacher_target_count"] == pack["teacher_target_count"] == summary["teacher_target_count"] == 0
        and not any(item["teacher_target"] for item in review_records.values()),
        "Teacher target count nonzero",
    ))
    check("no_automatic_visual_decision", lambda: assert_true(
        candidates["automatic_visual_decision_count"] == candidates["automatic_candidate_selection_count"] == 0
        and not protocol["risk_ranking"]["automatic_pass_fail"]
        and not protocol["risk_ranking"]["automatic_candidate_selection"],
        "automatic visual decision recorded",
    ))
    check("targeted_rerun_denied", lambda: assert_true(
        summary["targeted_rerun_authorization"] == overlay["targeted_rerun_authorization"] == "DENIED"
        and summary["targeted_rerun_generated_count"] == overlay["targeted_rerun_generated_count"] == 0,
        "targeted rerun state changed",
    ))
    check("no_generation_api_model_download", lambda: assert_true(
        summary["new_generation_calls"] == summary["external_api_calls"] == summary["api_key_reads"] == summary["model_downloads"] == 0,
        "prohibited generation/API/model action",
    ))
    check("formal_base_paper_state", lambda: assert_true(
        summary["formal_base_status"] == "PENDING"
        and summary["subject00_paper_positive_claim_count"] == summary["paper_modifications"] == 0,
        "Formal Base/paper state changed",
    ))

    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".pdf"}
    check("no_image_or_pdf_git_changes", lambda: assert_true(
        not any(Path(path).suffix.lower() in image_extensions for path in paths),
        "image/PDF entered Git changes",
    ))
    check("no_paper_modification", lambda: assert_true(
        not any(path.startswith("docs/PAPER/") or Path(path).suffix.lower() in {".tex", ".bib"} for path in paths),
        "paper source changed",
    ))
    check("no_training_modification", lambda: assert_true(
        not any(path.startswith(("scene/", "utils/", "config/")) or Path(path).name.startswith("train") for path in paths),
        "training code changed",
    ))
    check("correction_overlay_schema", lambda: assert_true({
        "old_report_path", "old_report_sha256", "old_missing_cell_list", "recomputed_missing_cell_list",
        "root_cause", "affected_downstream_decision", "old_evidence_preserved",
        "targeted_rerun_authorization", "cohort_registry_evidence_conflict",
    }.issubset(overlay), "correction overlay incomplete"))
    check("dual_json_exact", lambda: assert_true(all(
        file_sha256(repo_path) == file_sha256(external_path) for repo_path, external_path in DUAL_PATHS.items()
    ), "Git/external JSON mismatch"))
    check("json_content_hashes", lambda: verify_content_hashes(list(DUAL_PATHS) + [HANDOFF_PATH]))
    check("final_classification", lambda: assert_true(
        summary["final_classification"] == handoff["final_classification"] == FINAL_CLASSIFICATION,
        "final classification mismatch",
    ))
    check("next_task_unique", lambda: assert_true(
        summary["next_task"] == handoff["next_task"] == protocol["next_task_on_success"] == NEXT_TASK,
        "next task mismatch",
    ))
    check("paper_final_false", lambda: assert_true(all(
        item["paper_final"] is False for item in (protocol, cohorts, coverage, candidates, review, overlay, pack, summary, handoff)
    ), "PAPER_FINAL changed"))

    failed = [item for item in checks if item["status"] != "PASS"]
    payload = {
        "schema_version": "canondressgs.subject00.attempt001_1349_cohort_correction_tests.v1",
        "task_id": TASK_ID,
        "source_head": SOURCE_HEAD,
        "branch": BRANCH,
        "test_count": len(checks),
        "pass_count": len(checks) - len(failed),
        "fail_count": len(failed),
        "result": "PASS" if not failed else "FAIL",
        "checks": checks,
        "final_classification": FINAL_CLASSIFICATION if not failed else "SUBJECT00_1349_COHORT_CORRECTION_CONTRACT_VIOLATION",
        "paper_final": False,
        "next_task": NEXT_TASK if not failed else "REPAIR_SUBJECT00_1349_COHORT_CORRECTION_CONTRACT",
    }
    write_json(TESTS_PATH, payload)
    write_json(EXTERNAL_TESTS_PATH, payload)
    print(json.dumps({
        "result": payload["result"],
        "test_count": payload["test_count"],
        "pass_count": payload["pass_count"],
        "fail_count": payload["fail_count"],
        "tests_path": str(TESTS_PATH),
        "external_tests_path": str(EXTERNAL_TESTS_PATH),
        "failed": failed,
    }, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
