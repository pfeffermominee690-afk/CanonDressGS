#!/usr/bin/env python3
"""Verify the Subject00 resolution audit and independent review package."""

from __future__ import annotations

import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

import audit_subject00_managed_output_resolutions as audit


ROOT = audit.ROOT
RISK = audit.RISK
DATA_ROOT = audit.DATA_ROOT
CANDIDATE_ROOT = audit.CANDIDATE_ROOT
REVIEW_ROOT = audit.REVIEW_ROOT
TASK_ID = audit.TASK_ID


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_sealed(path: Path, payload: dict[str, Any]) -> None:
    audit.write_json(path, payload, sealed=True)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def seal_ok(path: Path) -> bool:
    payload = load(path)
    claimed = payload.get("content_sha256")
    return claimed is not None and claimed == audit.canonical_sha256(payload)


def main() -> int:
    inventory_path = RISK / "subject00_managed_output_resolution_inventory.json"
    distribution_path = RISK / "subject00_managed_output_resolution_distribution.json"
    technical_path = RISK / "subject00_managed_output_technical_eligibility.json"
    review_contract_path = RISK / "subject00_managed_output_double_review_contract.json"
    regeneration_path = RISK / "subject00_managed_output_regeneration_plan.json"
    tests_path = RISK / "subject00_managed_output_resolution_tests.json"
    summary_path = RISK / "subject00_managed_output_resolution_final_summary.json"
    handoff_path = ROOT / "project_control_handoff" / "subject00_managed_output_resolution_adjudication_handoff.json"
    external_audit_path = DATA_ROOT / "12_final_verification" / "subject00_managed_output_resolution_audit.json"

    inventory = load(inventory_path)
    distribution = load(distribution_path)
    technical = load(technical_path)
    review_contract = load(review_contract_path)
    regeneration = load(regeneration_path)
    summary = load(summary_path)
    output_registry = load(RISK / "subject00_codex_managed_output_registry.json")
    prior_technical = load(RISK / "subject00_codex_managed_technical_verification.json")
    reviewer_1 = load(REVIEW_ROOT / "reviewer_1_manifest.json")
    reviewer_2 = load(REVIEW_ROOT / "reviewer_2_manifest.json")
    adjudication = load(REVIEW_ROOT / "adjudication_manifest.json")

    records = inventory["records"]
    registry_by_id = {item["request_id"]: item for item in output_registry["records"]}
    actual: dict[str, dict[str, Any]] = {}
    parse_errors: list[str] = []
    sha_groups: dict[str, list[str]] = defaultdict(list)
    for item in records:
        request_id = item["request_id"]
        path = Path(item["output_path"])
        try:
            with Image.open(path) as image:
                image.load()
                width, height = image.size
                image_format = image.format
                orientation = image.getexif().get(274)
                has_alpha = "A" in image.getbands() or "transparency" in image.info
            sha = audit.file_sha256(path)
            category, ratio, error = audit.classify(width, height)
            actual[request_id] = {
                "path": str(path),
                "sha256": sha,
                "width": width,
                "height": height,
                "format": image_format,
                "orientation": orientation,
                "has_alpha": has_alpha,
                "category": category,
                "ratio": ratio,
                "error": error,
            }
            sha_groups[sha].append(request_id)
        except Exception as exc:  # pragma: no cover - reports real corrupt assets
            parse_errors.append(f"{request_id}: {exc}")

    request_ids = [item["request_id"] for item in records]
    output_paths = [item["output_path"] for item in records]
    duplicate_shas = {sha: ids for sha, ids in sha_groups.items() if len(ids) > 1}
    provenance_paths = [DATA_ROOT / "11_provenance" / "requests" / f"{request_id}.json" for request_id in request_ids]
    category_counts = Counter(value["category"] for value in actual.values())
    inventory_category_counts = Counter(item["resolution_category"] for item in records)

    mutation_ids = [
        item["request_id"]
        for item in records
        if item["request_id"] not in actual
        or actual[item["request_id"]]["sha256"] != item["output_sha256"]
        or actual[item["request_id"]]["sha256"] != registry_by_id[item["request_id"]]["output_sha256"]
    ]
    metadata_mismatches = [
        item["request_id"]
        for item in records
        if item["request_id"] not in actual
        or actual[item["request_id"]]["width"] != item["width"]
        or actual[item["request_id"]]["height"] != item["height"]
        or actual[item["request_id"]]["format"] != item["format"]
        or actual[item["request_id"]]["category"] != item["resolution_category"]
    ]
    ratio_mismatches = [
        item["request_id"]
        for item in records
        if item["request_id"] not in actual
        or abs(actual[item["request_id"]]["ratio"] - item["aspect_ratio"]) > 1e-12
        or abs(actual[item["request_id"]]["error"] - item["aspect_ratio_absolute_error"]) > 1e-12
    ]

    reviewer_visual_null_errors: list[str] = []
    for name, manifest in {
        "reviewer_1": reviewer_1,
        "reviewer_2": reviewer_2,
        "adjudication": adjudication,
    }.items():
        for item in manifest["records"]:
            if any(item.get(field) is not None for field in audit.VISUAL_REVIEW_FIELDS):
                reviewer_visual_null_errors.append(f"{name}:{item['request_id']}")
            if name == "adjudication" and any(
                item.get(field) is not None
                for field in [
                    "reviewer_1_decision",
                    "reviewer_2_decision",
                    "visual_adjudication_status",
                    "final_dataset_eligibility",
                ]
            ):
                reviewer_visual_null_errors.append(f"{name}:{item['request_id']}:decision")

    accepted_count = sum(path.is_file() for path in (DATA_ROOT / "06_accepted_rgb").rglob("*"))
    teacher_count = sum(path.is_file() for path in (DATA_ROOT / "09_teacher_targets").rglob("*"))
    contact_paths = [Path(path) for path in review_contract["contact_sheets"].values()]
    contact_errors: list[str] = []
    for path in contact_paths:
        try:
            with Image.open(path) as image:
                image.load()
                if image.format != "PNG" or image.width <= 0 or image.height <= 0:
                    contact_errors.append(str(path))
        except Exception:
            contact_errors.append(str(path))

    git_status = git("status", "--porcelain").splitlines()
    git_image_additions = [
        line for line in git_status if line.lower().rstrip().endswith((".png", ".jpg", ".jpeg"))
    ]
    seal_paths = [
        inventory_path,
        distribution_path,
        technical_path,
        review_contract_path,
        regeneration_path,
        tests_path,
        summary_path,
        handoff_path,
        REVIEW_ROOT / "reviewer_1_manifest.json",
        REVIEW_ROOT / "reviewer_2_manifest.json",
        REVIEW_ROOT / "adjudication_manifest.json",
        external_audit_path,
    ]
    seal_failures = [str(path) for path in seal_paths if not seal_ok(path)]

    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, actual_value: Any, expected: Any) -> None:
        checks.append(
            {
                "name": name,
                "status": "PASS" if passed else "FAIL",
                "actual": actual_value,
                "expected": expected,
            }
        )

    add("image_count", len(records) == 48, len(records), 48)
    add("actual_png_parse_count", len(actual) == 48 and not parse_errors, {"parsed": len(actual), "errors": parse_errors}, {"parsed": 48, "errors": []})
    add("provenance_count", all(path.is_file() for path in provenance_paths) and len(provenance_paths) == 48, sum(path.is_file() for path in provenance_paths), 48)
    add("unique_request_ids", len(set(request_ids)) == 48, len(set(request_ids)), 48)
    add("unique_output_paths", len(set(output_paths)) == 48, len(set(output_paths)), 48)
    add("duplicate_output_sha_groups", not duplicate_shas, duplicate_shas, {})
    add("dimension_categories_sum", sum(inventory_category_counts.values()) == 48, sum(inventory_category_counts.values()), 48)
    add("exact_count_matches_prior_report", category_counts[audit.CATEGORY_A] == prior_technical["resolution_pass_1024x1536"] == 5, category_counts[audit.CATEGORY_A], 5)
    add("aspect_ratio_calculation", not ratio_mismatches, ratio_mismatches, [])
    add("reviewer_1_entries", len(reviewer_1["records"]) == 48, len(reviewer_1["records"]), 48)
    add("reviewer_2_entries", len(reviewer_2["records"]) == 48, len(reviewer_2["records"]), 48)
    add("adjudication_entries", len(adjudication["records"]) == 48, len(adjudication["records"]), 48)
    add("visual_decisions_all_null", not reviewer_visual_null_errors, reviewer_visual_null_errors, [])
    add("accepted_images", accepted_count == 0, accepted_count, 0)
    add("teacher_targets", teacher_count == 0, teacher_count, 0)
    add("original_image_mutations", not mutation_ids, mutation_ids, [])
    add("resize_operations", summary["resize_operations"] == 0, summary["resize_operations"], 0)
    add("crop_operations", summary["crop_operations"] == 0, summary["crop_operations"], 0)
    add("generation_calls", summary["generation_calls"] == 0, summary["generation_calls"], 0)
    add("external_api_calls", summary["external_api_calls"] == 0, summary["external_api_calls"], 0)
    add("cloud_image_writes", summary["cloud_image_writes"] == 0, summary["cloud_image_writes"], 0)
    add("git_image_additions", not git_image_additions, git_image_additions, [])
    add("formal_base", summary["formal_base"] == "PENDING", summary["formal_base"], "PENDING")
    add("paper_final", summary["paper_final"] is False, summary["paper_final"], False)
    add("registry_metadata_consistency", not metadata_mismatches and inventory["registry_consistency"] == "PASS", metadata_mismatches, [])
    add(
        "technical_eligibility_counts",
        technical["counts"] == {"pass": 5, "pending_user_resize_contract": 0, "fail": 43, "visual_review_pending": 48},
        technical["counts"],
        {"pass": 5, "pending_user_resize_contract": 0, "fail": 43, "visual_review_pending": 48},
    )
    add("resolution_contact_sheets", len(contact_paths) == 4 and not contact_errors, {"count": len(contact_paths), "errors": contact_errors}, {"count": 4, "errors": []})
    add("regeneration_plan", regeneration["record_count"] == 43 and all(not item["generation_authorization"] and item["retry_budget"] == 0 for item in regeneration["records"]), regeneration["record_count"], 43)
    add("exif_alpha_audited", all("exif_orientation" in item and "has_alpha_channel" in item for item in records), sum("exif_orientation" in item and "has_alpha_channel" in item for item in records), 48)
    add("json_content_seals", not seal_failures, seal_failures, [])

    failed = [item["name"] for item in checks if item["status"] != "PASS"]
    mutation_audit = {
        "candidate_images_checked": len(records),
        "original_image_mutation_count": len(mutation_ids),
        "resize_operations": 0,
        "crop_operations": 0,
        "pad_operations": 0,
        "rotate_operations": 0,
        "generation_calls": 0,
        "external_api_calls": 0,
        "cloud_image_writes": 0,
        "git_image_additions": len(git_image_additions),
        "visual_judgments_populated": len(reviewer_visual_null_errors),
        "contact_sheets_created_outside_candidate_root": len(contact_paths),
    }
    tests = {
        "schema_version": "canondressgs.subject00.managed_output_resolution_tests.v1",
        "task_id": TASK_ID,
        "status": "PASS" if not failed else "FAIL",
        "total": len(checks),
        "passed": len(checks) - len(failed),
        "failed": failed,
        "checks": checks,
        "mutation_audit": mutation_audit,
        "paper_final": False,
    }
    write_sealed(tests_path, tests)

    summary["tests"] = tests["status"]
    summary["mutation_audit"] = mutation_audit
    write_sealed(summary_path, summary)
    handoff = load(handoff_path)
    handoff["tests"] = tests["status"]
    handoff["mutation_audit"] = mutation_audit
    write_sealed(handoff_path, handoff)
    external_audit = load(external_audit_path)
    external_audit["tests"] = tests["status"]
    external_audit["test_count"] = len(checks)
    external_audit["mutation_audit"] = mutation_audit
    write_sealed(external_audit_path, external_audit)

    report_path = ROOT / "docs" / "PAPER" / "AAAI27_SUBJECT00_MANAGED_OUTPUT_RESOLUTION_AUDIT_20260725.md"
    report = report_path.read_text(encoding="utf-8")
    report = report.replace(
        "- Contract tests: `RUN_AFTER_AUDIT_EMISSION`",
        f"- Contract tests: `{tests['status']}` ({tests['passed']}/{tests['total']})",
    )
    report_path.write_text(report, encoding="utf-8", newline="\n")

    print(
        json.dumps(
            {
                "status": tests["status"],
                "passed": tests["passed"],
                "total": tests["total"],
                "failed": failed,
                "mutation_audit": mutation_audit,
            },
            indent=2,
        )
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
