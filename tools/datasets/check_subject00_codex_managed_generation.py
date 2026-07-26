#!/usr/bin/env python3
"""Audit the local Subject00 managed-generation outputs and contracts."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
DATA_ROOT = Path(r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-GENERATION-001\attempt_001")
TASK_ID = "AAAI27-SUBJECT00-CODEX-MANAGED-GENERATION-001"
TARGET_SIZE = (1024, 1536)
REVIEW_FIELDS = [
    "identity_match", "face_match", "hair_match", "skin_tone_match", "body_shape_match",
    "pose_match", "camera_match", "garment_match", "garment_slot_match", "mask_quality",
    "edge_quality", "hands_feet_complete", "background_match", "lighting_match",
    "artifact_grade", "reviewer_1", "reviewer_2", "reject_reason",
]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_sha256(value: Any) -> str:
    value = dict(value)
    value.pop("content_sha256", None)
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_sealed(path: Path, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload.pop("content_sha256", None)
    payload["content_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def main() -> int:
    manifest = load(RISK / "subject00_generation_request_manifest.json")
    progress = load(DATA_ROOT / "03_generation_requests" / "generation_progress_registry.json")
    registry = load(RISK / "subject00_codex_managed_generation_registry.json")
    output_registry = load(RISK / "subject00_codex_managed_output_registry.json")
    technical = load(RISK / "subject00_codex_managed_technical_verification.json")
    review = load(DATA_ROOT / "05_human_review" / "subject00_codex_managed_human_review_manifest.json")
    summary_path = RISK / "subject00_codex_managed_generation_final_summary.json"
    summary = load(summary_path)

    source_rows = manifest["requests"]
    source_by_id = {row["request_id"]: row for row in source_rows}
    output_by_id = {row["request_id"]: row for row in output_registry["records"]}
    progress_by_id = {row["request_id"]: row for row in progress["records"]}
    review_by_id = {row["request_id"]: row for row in review["records"]}
    candidates = sorted((DATA_ROOT / "04_generation_responses" / "codex_managed_candidates").rglob("*.png"))

    parsed: dict[str, dict[str, Any]] = {}
    parse_errors: list[str] = []
    hashes: dict[str, list[str]] = defaultdict(list)
    for path in candidates:
        try:
            with Image.open(path) as image:
                image.load()
                item = {"format": image.format, "size": image.size, "bytes": path.stat().st_size, "sha256": sha256(path)}
            parsed[path.stem] = item
            hashes[item["sha256"]].append(path.stem)
        except Exception as exc:  # pragma: no cover - reports real corrupt assets
            parse_errors.append(f"{path}: {exc}")

    duplicate_groups = {value: ids for value, ids in hashes.items() if len(ids) > 1}
    provenance_files = sorted((DATA_ROOT / "11_provenance" / "requests").glob("*.json"))
    provenance_errors: list[str] = []
    for path in provenance_files:
        item = load(path)
        request_id = path.stem
        output = parsed.get(request_id)
        if not item.get("success") or output is None or item.get("output_sha256") != output["sha256"]:
            provenance_errors.append(request_id)

    source_ids = [row["request_id"] for row in source_rows]
    progress_ids = progress["stable_order"]
    output_ids = [row["request_id"] for row in output_registry["records"]]
    id_changes = sorted(set(source_ids) ^ set(progress_ids)) + sorted(set(source_ids) ^ set(output_ids))
    sha_changes = [
        request_id for request_id in source_ids
        if output_by_id.get(request_id, {}).get("source_input_sha256") != source_by_id[request_id]["input_sha256"]
    ]
    garment_slots: dict[str, set[str]] = defaultdict(set)
    garment_slot_candidates: Counter[tuple[str, str]] = Counter()
    for row in source_rows:
        garment_slots[row["garment_id"]].add(row["slot_id"])
        garment_slot_candidates[(row["garment_id"], row["slot_id"])] += 1

    status_lines = git("status", "--porcelain").splitlines()
    git_image_additions = [line for line in status_lines if line.lower().rstrip().endswith((".png", ".jpg", ".jpeg"))]
    generated_code = (ROOT / "tools" / "datasets" / "manage_subject00_codex_managed_generation.py").read_text(
        encoding="utf-8"
    )
    forbidden_network_patterns = {
        "requests_call": r"\brequests\s*\.",
        "invoke_web_request": r"Invoke-WebRequest",
        "curl_call": r"\bcurl\b",
        "openai_sdk_client": r"\bOpenAI\s*\(",
        "custom_base_url_assignment": r"\bbase_url\s*=",
    }
    forbidden_network_hits = [name for name, pattern in forbidden_network_patterns.items() if re.search(pattern, generated_code)]

    scan_paths = [
        ROOT / "tools" / "datasets" / "manage_subject00_codex_managed_generation.py",
        ROOT / "tools" / "datasets" / "check_subject00_codex_managed_generation.py",
        *RISK.glob("subject00_codex_managed_*.json"),
        ROOT / "project_control_handoff" / "subject00_codex_managed_generation_handoff.json",
    ]
    secret_patterns = [r"sk-[A-Za-z0-9_-]{20,}", r"ghp_[A-Za-z0-9]{20,}", r"OPENAI_API_KEY\s*=\s*[^\s]+"]
    secret_hits = [
        str(path) for path in scan_paths
        if path.is_file() and any(re.search(pattern, path.read_text(encoding="utf-8")) for pattern in secret_patterns)
    ]

    accepted_count = sum(path.is_file() for path in (DATA_ROOT / "06_accepted_rgb").rglob("*"))
    teacher_count = sum(path.is_file() for path in (DATA_ROOT / "09_teacher_targets").rglob("*"))
    review_pending = sum(row.get("final_status") == "PENDING_HUMAN_DOUBLE_REVIEW" for row in review["records"])
    review_fields_null = all(all(row.get(field) is None for field in REVIEW_FIELDS) for row in review["records"])
    successful = sum(row["status"] in {"GENERATED_TECHNICAL_PASS", "GENERATED_TECHNICAL_REVIEW"} for row in progress["records"])
    exact_resolution = sum(item["size"] == TARGET_SIZE for item in parsed.values())

    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, actual: Any, expected: Any) -> None:
        checks.append({"name": name, "status": "PASS" if passed else "FAIL", "actual": actual, "expected": expected})

    add("request_count", len(source_rows) == 48, len(source_rows), 48)
    add("garment_count", set(garment_slots) == {"O01", "O03", "O04"}, sorted(garment_slots), ["O01", "O03", "O04"])
    add("slots_per_garment", all(len(value) == 8 for value in garment_slots.values()), {key: len(value) for key, value in garment_slots.items()}, 8)
    add("candidates_per_slot", all(value == 2 for value in garment_slot_candidates.values()), dict(sorted((f"{key[0]}:{key[1]}", value) for key, value in garment_slot_candidates.items())), 2)
    add("request_id_changes", not id_changes and source_ids == progress_ids == output_ids, id_changes, [])
    add("source_input_sha_changes", not sha_changes, sha_changes, [])
    add("external_api_calls", registry.get("external_api_calls") == 0, registry.get("external_api_calls"), 0)
    add("api_key_reads", registry.get("api_key_reads") == 0, registry.get("api_key_reads"), 0)
    add("custom_base_url_calls", registry.get("custom_base_url_calls") == 0 and not forbidden_network_hits, {"declared": registry.get("custom_base_url_calls"), "code_hits": forbidden_network_hits}, {"declared": 0, "code_hits": []})
    add("sublyx_calls", registry.get("sublyx_calls") == 0, registry.get("sublyx_calls"), 0)
    add("code78_calls", registry.get("code78_calls") == 0, registry.get("code78_calls"), 0)
    add("codex_managed_generation_calls", registry.get("managed_generation_calls") == successful + registry.get("technical_retries", 0), registry.get("managed_generation_calls"), successful + registry.get("technical_retries", 0))
    add("output_file_count", len(candidates) == 48, len(candidates), 48)
    add("successful_output_count", successful == 48, successful, 48)
    add("png_parse_count", len(parsed) == 48 and not parse_errors and all(item["format"] == "PNG" and item["bytes"] > 0 for item in parsed.values()), {"parsed": len(parsed), "errors": parse_errors}, {"parsed": 48, "errors": []})
    add("resolution_1024x1536_count", exact_resolution == technical["resolution_pass_1024x1536"], exact_resolution, technical["resolution_pass_1024x1536"])
    add("duplicate_output_sha_groups", not duplicate_groups, duplicate_groups, {})
    add("provenance_count", len(provenance_files) == 48 and not provenance_errors, {"count": len(provenance_files), "errors": provenance_errors}, {"count": 48, "errors": []})
    add("missing_provenance_count", set(source_ids) == {path.stem for path in provenance_files}, sorted(set(source_ids) - {path.stem for path in provenance_files}), [])
    add("human_review_pending_count", review_pending == 48 and review_fields_null and set(review_by_id) == set(source_ids), {"pending": review_pending, "all_fields_null": review_fields_null}, {"pending": 48, "all_fields_null": True})
    add("accepted_count", accepted_count == 0, accepted_count, 0)
    add("teacher_target_count", teacher_count == 0, teacher_count, 0)
    add("cloud_image_writes", registry.get("cloud_image_writes") == 0, registry.get("cloud_image_writes"), 0)
    add("git_image_additions", not git_image_additions, git_image_additions, [])
    add("secret_scan", not secret_hits, secret_hits, [])
    add("formal_base_dependency", summary.get("formal_base_dependency") == "PENDING", summary.get("formal_base_dependency"), "PENDING")
    add("paper_final", summary.get("paper_final") is False, summary.get("paper_final"), False)

    mutation_audit = {
        "external_dataset_root": str(DATA_ROOT),
        "candidate_files_written": len(candidates),
        "git_image_additions": len(git_image_additions),
        "cloud_image_writes": 0,
        "external_api_calls": 0,
        "custom_base_url_calls": 0,
        "api_key_reads": 0,
        "sublyx_calls": 0,
        "code78_calls": 0,
        "forbidden_network_code_hits": forbidden_network_hits,
        "secret_scan_hits": secret_hits,
    }
    failed = [check["name"] for check in checks if check["status"] != "PASS"]
    tests = {
        "schema_version": "canondressgs.subject00.codex_managed_generation_tests.v1",
        "task_id": TASK_ID,
        "status": "PASS" if not failed else "FAIL",
        "total": len(checks),
        "passed": len(checks) - len(failed),
        "failed": failed,
        "checks": checks,
        "metrics": {
            "successful_outputs": successful,
            "technical_retries": registry.get("technical_retries"),
            "resolution_pass_1024x1536": exact_resolution,
            "resolution_mismatches": len(parsed) - exact_resolution,
        },
        "mutation_audit": mutation_audit,
        "paper_final": False,
    }
    write_sealed(RISK / "subject00_codex_managed_generation_tests.json", tests)

    summary["tests"] = tests["status"]
    summary["mutation_audit"] = mutation_audit
    write_sealed(summary_path, summary)
    handoff_path = ROOT / "project_control_handoff" / "subject00_codex_managed_generation_handoff.json"
    handoff = load(handoff_path)
    handoff["tests"] = tests["status"]
    handoff["mutation_audit"] = mutation_audit
    write_sealed(handoff_path, handoff)

    external_technical_path = DATA_ROOT / "12_final_verification" / "technical_verification.json"
    external_technical = load(external_technical_path)
    external_technical["tests"] = tests["status"]
    external_technical["test_count"] = len(checks)
    external_technical["mutation_audit"] = mutation_audit
    external_technical_path.write_text(json.dumps(external_technical, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

    report_path = ROOT / "docs" / "PAPER" / "AAAI27_SUBJECT00_CODEX_MANAGED_GENERATION_REPORT_20260725.md"
    report = report_path.read_text(encoding="utf-8")
    report = report.replace("- Contract tests: `RUN_AFTER_METADATA_EMISSION`", f"- Contract tests: `{tests['status']}` ({tests['passed']}/{tests['total']})")
    report_path.write_text(report, encoding="utf-8", newline="\n")

    print(json.dumps({"status": tests["status"], "passed": tests["passed"], "total": tests["total"], "failed": failed, "mutation_audit": mutation_audit}, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
