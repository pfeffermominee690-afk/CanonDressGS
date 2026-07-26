#!/usr/bin/env python3
"""Validate frozen AAAI-27 preflight manifests and governance invariants."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Callable


SCHEMA_VERSION = "aaai27_sprint_preflight_v1"
EXPECTED_LONG_TERM_HEAD = "9fc88033b407136073f7bddc6ffca6dd4dd1e0f5"
EXPECTED_CANDIDATES = {"O01", "O02", "O03", "O04", "O06", "O07", "O08"}
EXPECTED_UNSEEN = {"O03", "O08"}
EXPECTED_JSON_ARTIFACTS = {
    "candidate_outfit_audit.json": "candidate_outfit_audit",
    "condition_selection_32.json": "condition_selection_32",
    "target_generation_gate_28_manifest.json": "target_generation_gate_28_manifest",
    "target_generation_full_192_manifest.json": "target_generation_full_192_manifest",
    "experiment_registry.json": "experiment_registry",
    "paper_figure_registry.json": "paper_figure_registry",
    "paper_table_registry.json": "paper_table_registry",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/aaai27_sprint"))
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("schemas/aaai27_sprint_preflight_v1.schema.json"),
    )
    parser.add_argument("--expected-long-term-head", default=EXPECTED_LONG_TERM_HEAD)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_aaai_candidate_outfits_are_explicit(context: dict[str, Any]) -> None:
    records = context["candidate_audit"]["records"]
    ids = {item["outfit_id"] for item in records}
    require(ids == EXPECTED_CANDIDATES, f"Candidate set mismatch: {ids}")
    require(len(records) == 7, "Candidate records must be unique")
    for item in records:
        for key in (
            "definition_en",
            "mapping_status",
            "reference_image_count",
            "reference_view_coverage",
            "source_file_hash_matches",
            "exposed_skin_required",
            "large_exterior_silhouette_required",
            "support_risk",
            "existing_subject02_target_count",
            "gate_status",
        ):
            require(key in item, f"{item['outfit_id']} missing {key}")
        require(
            item["source_file_hash_matches"] == item["source_file_count"],
            f"{item['outfit_id']} source hash mismatch",
        )
        require(item["gate_status"] == "REQUIRES_28_IMAGE_DATA_AND_CAPACITY_GATE", "Candidate bypasses gate")


def test_aaai_condition_split_has_no_leakage(context: dict[str, Any]) -> None:
    selection = context["selection"]["records"]
    ids = [item["condition_id"] for item in selection]
    require(len(ids) == 32 and len(set(ids)) == 32, "Condition IDs must be 32 unique records")
    counts = Counter(item["condition_split"] for item in selection)
    require(counts == {"train": 24, "validation": 4, "novel_pose_test": 4}, f"Bad split counts: {counts}")
    split_sets = {
        split: {item["condition_id"] for item in selection if item["condition_split"] == split}
        for split in counts
    }
    require(not (split_sets["train"] & split_sets["validation"]), "Train/validation leakage")
    require(not (split_sets["train"] & split_sets["novel_pose_test"]), "Train/test leakage")
    require(not (split_sets["validation"] & split_sets["novel_pose_test"]), "Validation/test leakage")


def test_aaai_four_view_balance(context: dict[str, Any]) -> None:
    selection = context["selection"]["records"]
    views = Counter(item["view"] for item in selection)
    require(views == {"front": 8, "back": 8, "left": 8, "right": 8}, f"View imbalance: {views}")
    for view in views:
        subset = [item for item in selection if item["view"] == view]
        difficulty = Counter(item["difficulty"] for item in subset)
        splits = Counter(item["condition_split"] for item in subset)
        require(difficulty == {"easy": 3, "medium": 3, "hard": 2}, f"{view} difficulty mismatch: {difficulty}")
        require(splits == {"train": 6, "validation": 1, "novel_pose_test": 1}, f"{view} split mismatch: {splits}")
        require(all(item["full_body_uncropped"] for item in subset), f"{view} has cropped sample")
        require(all(item["hand_screening_status"].startswith("PASS") for item in subset), f"{view} has failed hand audit")


def test_aaai_unseen_outfits_not_in_training(context: dict[str, Any]) -> None:
    full = context["full_manifest"]
    require(set(full["unseen_test_outfits"]) == EXPECTED_UNSEEN, "Unseen outfit set changed")
    for record in full["records"]:
        if record["outfit_id"] in EXPECTED_UNSEEN:
            require(record["split"] == "unseen_outfit_evaluation_only", f"Unseen outfit leaked: {record['record_id']}")
        elif record["split"] == "train":
            require(record["outfit_id"] not in EXPECTED_UNSEEN, f"Unseen outfit in training: {record['record_id']}")


def test_aaai_generation_manifest_has_no_target_dependency(context: dict[str, Any]) -> None:
    for manifest_name in ("gate_manifest", "full_manifest"):
        manifest = context[manifest_name]
        require(manifest["network_requests"] == 0, f"{manifest_name} records network requests")
        require(manifest["images_generated"] == 0, f"{manifest_name} records generated images")
        for record in manifest["records"]:
            require(record["target_dependency"] is False, f"Target dependency: {record['record_id']}")
            require(record["generated_target_used_as_input"] is False, f"Generated target input: {record['record_id']}")
            require(record["teacher_used"] is False, f"Teacher input: {record['record_id']}")
            require(record["target_rgb_mask_inference_input"] is False, f"Target RGB/mask input: {record['record_id']}")
            require(
                record["model_conditioning_fields"] == ["donor_references", "target_pose", "target_camera"],
                f"Unexpected conditioning fields: {record['record_id']}",
            )
            require(record["status"] == "PLANNED_NOT_GENERATED", f"Unexpected generation status: {record['record_id']}")


def test_post_aaai_handoff_has_required_sections(context: dict[str, Any]) -> None:
    text = context["handoff_text"]
    required = [
        "## 1. Original goal",
        "## 2. Formal method modules and contracts",
        "## 3. Data asset status",
        "## 4. Module and experiment status ledger",
        "## 5. Mandatory conclusions",
        "## 6. Paused or deprecated routes",
        "## 7. Unresolved items",
        "## 8. Post-AAAI recovery roadmap",
        "## 9. Exact recovery instructions",
        "## 10. Non-misread guardrails",
    ]
    for heading in required:
        require(heading in text, f"Handoff missing section: {heading}")
    for phase in range(1, 10):
        require(f"### P{phase} —" in text, f"Handoff missing recovery P{phase}")
    for status_token in ("Module 4B", "R3-CLEAN", "GEOMCAM", "V5.3", "PASS_WITH_CUDA_NONDETERMINISM"):
        require(status_token in text, f"Handoff missing status token: {status_token}")


def test_claims_have_required_evidence(context: dict[str, Any]) -> None:
    text = context["claims_text"]
    sections = []
    for index in range(1, 13):
        marker = f"## C{index} —"
        require(marker in text, f"Claim matrix missing C{index}")
        start = text.index(marker)
        next_positions = [text.find(f"## C{other} —", start + len(marker)) for other in range(index + 1, 13)]
        next_positions = [position for position in next_positions if position >= 0]
        end = min(next_positions) if next_positions else text.find("## Abstract and conclusion release gate", start)
        sections.append(text[start:end if end >= 0 else None])
    required_fields = (
        "**Claim:**",
        "**Required experiment:**",
        "**Baseline:**",
        "**Metric:**",
        "**Qualitative figure:**",
        "**Artifact/output path:**",
        "**Current status:**",
        "**Owner:**",
        "**Deadline:**",
        "**Risk:**",
        "**Fallback wording:**",
    )
    for index, section in enumerate(sections, 1):
        for field in required_fields:
            require(field in section, f"Claim C{index} missing {field}")
    for priority in ("Unseen-outfit", "No per-outfit", "Graph completion", "Region-trusted", "Target identity"):
        require(priority.lower() in text.lower(), f"Priority claim missing: {priority}")
    require("no claim with status other than `SUPPORTED` may enter the abstract or conclusion" in text, "Missing claim release rule")


def test_long_term_branch_is_unchanged(context: dict[str, Any]) -> None:
    head = subprocess.check_output(
        ["git", "rev-parse", "refs/heads/pipeline/full-dressable-20260715"],
        cwd=context["repo_root"],
        text=True,
        encoding="utf-8",
    ).strip()
    require(head == context["expected_long_term_head"], f"Long-term branch changed: {head}")


def test_manifest_schema_structure(context: dict[str, Any]) -> None:
    schema = context["schema"]
    require(schema["$schema"].endswith("2020-12/schema"), "Unexpected schema draft")
    require("$defs" in schema and "generationRecord" in schema["$defs"], "Schema definitions incomplete")
    for filename, artifact_type in EXPECTED_JSON_ARTIFACTS.items():
        payload = context["artifacts_by_filename"][filename]
        require(payload["schema_version"] == SCHEMA_VERSION, f"{filename} schema version mismatch")
        require(payload["artifact_type"] == artifact_type, f"{filename} artifact type mismatch")
        require(isinstance(payload["status"], str) and payload["status"], f"{filename} missing status")
    require(context["gate_manifest"]["record_count"] == 28, "Gate manifest must have 28 records")
    require(len(context["gate_manifest"]["records"]) == 28, "Gate record count mismatch")
    require(context["full_manifest"]["record_count"] == 192, "Full manifest must have 192 records")
    require(len(context["full_manifest"]["records"]) == 192, "Full record count mismatch")
    ids = set(context["experiments"]["method_ids"])
    require(ids == {"M0", "M1", "M2", "M3", "M4", "M5", "A1", "A2", "A3", "A4", "A5", "A6"}, "Experiment registry mismatch")


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    artifact_dir = args.artifact_dir if args.artifact_dir.is_absolute() else repo_root / args.artifact_dir
    schema_path = args.schema if args.schema.is_absolute() else repo_root / args.schema
    artifacts_by_filename = {
        filename: read_json(artifact_dir / filename) for filename in EXPECTED_JSON_ARTIFACTS
    }
    context = {
        "repo_root": repo_root,
        "expected_long_term_head": args.expected_long_term_head,
        "artifacts_by_filename": artifacts_by_filename,
        "candidate_audit": artifacts_by_filename["candidate_outfit_audit.json"],
        "selection": artifacts_by_filename["condition_selection_32.json"],
        "gate_manifest": artifacts_by_filename["target_generation_gate_28_manifest.json"],
        "full_manifest": artifacts_by_filename["target_generation_full_192_manifest.json"],
        "experiments": artifacts_by_filename["experiment_registry.json"],
        "schema": read_json(schema_path),
        "handoff_text": (repo_root / "docs" / "CANONDRESSGS_POST_AAAI_CONTINUATION_HANDOFF_20260718.md").read_text(encoding="utf-8"),
        "claims_text": (repo_root / "docs" / "AAAI27_CLAIM_EVIDENCE_MATRIX_20260718.md").read_text(encoding="utf-8"),
    }
    tests: list[Callable[[dict[str, Any]], None]] = [
        test_aaai_candidate_outfits_are_explicit,
        test_aaai_condition_split_has_no_leakage,
        test_aaai_four_view_balance,
        test_aaai_unseen_outfits_not_in_training,
        test_aaai_generation_manifest_has_no_target_dependency,
        test_post_aaai_handoff_has_required_sections,
        test_claims_have_required_evidence,
        test_long_term_branch_is_unchanged,
        test_manifest_schema_structure,
    ]
    for test in tests:
        test(context)
        print(f"PASS {test.__name__}")
    print(json.dumps({"status": "PASS", "tests": len(tests), "network_requests": 0, "training_runs": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
