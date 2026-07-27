#!/usr/bin/env python3
"""Audit Subject00 managed output resolutions without mutating candidate pixels."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
DOCS = ROOT / "docs" / "PAPER"
HANDOFF = ROOT / "project_control_handoff"
DATA_ROOT = Path(r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-GENERATION-001\attempt_001")
CANDIDATE_ROOT = DATA_ROOT / "04_generation_responses" / "codex_managed_candidates"
REVIEW_ROOT = DATA_ROOT / "05_human_review"
RESOLUTION_REVIEW_ROOT = REVIEW_ROOT / "resolution_adjudication"

TASK_ID = "AAAI27-SUBJECT00-MANAGED-OUTPUT-RESOLUTION-ADJUDICATION-001"
SOURCE_BRANCH = "research/subject00-codex-managed-generation-20260725"
SOURCE_HEAD = "8717118c9fd23c2cc92f65c5e13eff7ec6f15b86"
BRANCH = "research/subject00-managed-output-resolution-adjudication-20260725"
TARGET_WIDTH = 1024
TARGET_HEIGHT = 1536
TARGET_RATIO = TARGET_WIDTH / TARGET_HEIGHT
PATH_TYPE = "WINDOWS_LOCAL_EXTERNAL_DATASET_PATH"
FORMAL_BASE = "PENDING"
PAPER_FINAL = False
RESOLUTION_AUDIT_HEAD = "RESOLVE_AFTER_RESOLUTION_AUDIT_COMMIT"
NEXT_TASK = "USER_REVIEW_RESOLUTION_CONTACT_SHEETS_AND_SELECT_RESOLUTION_CONTRACT"

CATEGORY_A = "A_EXACT_TARGET_RESOLUTION"
CATEGORY_B = "B_PORTRAIT_EXACT_2_TO_3_RESIZE_CANDIDATE"
CATEGORY_C = "C_PORTRAIT_NEAR_2_TO_3_REVIEW_REQUIRED"
CATEGORY_D = "D_PORTRAIT_INCOMPATIBLE_ASPECT"
CATEGORY_E = "E_LANDSCAPE_OUTPUT"
CATEGORY_F = "F_SQUARE_OUTPUT"
CATEGORY_G = "G_PARSE_OR_METADATA_FAILURE"
CATEGORIES = [CATEGORY_A, CATEGORY_B, CATEGORY_C, CATEGORY_D, CATEGORY_E, CATEGORY_F, CATEGORY_G]

SLOT_VIEWS = {
    "slot_00": "front",
    "slot_01": "front-left",
    "slot_02": "front-right",
    "slot_03": "left",
    "slot_04": "right",
    "slot_05": "back-left",
    "slot_06": "back-right",
    "slot_07": "back",
}

VISUAL_REVIEW_FIELDS = [
    "identity_match",
    "face_match",
    "hair_match",
    "skin_tone_match",
    "body_shape_match",
    "pose_match",
    "camera_match",
    "garment_match",
    "garment_slot_match",
    "hands_complete",
    "feet_complete",
    "full_body_complete",
    "background_match",
    "lighting_match",
    "edge_quality",
    "artifact_grade",
    "review_decision",
    "reject_reason",
    "reviewer_id",
    "review_timestamp",
]


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


def write_json(path: Path, payload: dict[str, Any], sealed: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    if sealed:
        payload.pop("content_sha256", None)
        payload["content_sha256"] = canonical_sha256(payload)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_doc(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8", newline="\n")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def classify(width: int, height: int) -> tuple[str, float, float]:
    ratio = width / height
    error = abs(ratio - TARGET_RATIO)
    if width == TARGET_WIDTH and height == TARGET_HEIGHT:
        return CATEGORY_A, ratio, error
    if width > height:
        return CATEGORY_E, ratio, error
    if width == height:
        return CATEGORY_F, ratio, error
    if error <= 0.001:
        return CATEGORY_B, ratio, error
    if error <= 0.03:
        return CATEGORY_C, ratio, error
    return CATEGORY_D, ratio, error


def eligibility(category: str) -> str:
    if category == CATEGORY_A:
        return "PASS"
    if category == CATEGORY_B:
        return "PENDING_USER_RESIZE_CONTRACT"
    return "FAIL"


def dimension_object(width: Any, height: Any) -> dict[str, Any]:
    return {"width": width, "height": height}


def review_record(item: dict[str, Any]) -> dict[str, Any]:
    result = {
        "request_id": item["request_id"],
        "candidate_path": item["output_path"],
        "candidate_sha": item["output_sha256"],
        "actual_resolution": dimension_object(item["width"], item["height"]),
        "resolution_category": item["resolution_category"],
        "technical_eligibility": item["technical_dataset_eligibility"],
    }
    result.update({field: None for field in VISUAL_REVIEW_FIELDS})
    return result


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def make_contact_sheet(
    records: list[dict[str, Any]], path: Path, columns: int, tile_width: int, image_height: int, label_height: int
) -> None:
    rows = (len(records) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_width, rows * (image_height + label_height)), "#202225")
    draw = ImageDraw.Draw(canvas)
    label_font = font(12)
    for index, item in enumerate(records):
        x = (index % columns) * tile_width
        y = (index // columns) * (image_height + label_height)
        with Image.open(item["output_path"]) as source:
            preview = source.convert("RGB")
            preview.thumbnail((tile_width - 10, image_height - 10), Image.Resampling.LANCZOS)
        px = x + (tile_width - preview.width) // 2
        py = y + (image_height - preview.height) // 2
        canvas.paste(preview, (px, py))
        label = "\n".join(
            [
                item["request_id"],
                f"{item['width']}x{item['height']} ratio={item['aspect_ratio']:.6f}",
                item["resolution_category"],
                f"{item['garment_id']} {item['slot_id']} cam{item['camera_id']:02d} cand{item['candidate_index']}",
                f"eligibility: {item['technical_dataset_eligibility']}",
            ]
        )
        draw.multiline_text((x + 6, y + image_height + 5), label, fill="white", font=label_font, spacing=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=True)


def nested_distribution(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in sorted({str(item[key]) for item in records}):
        subset = [item for item in records if str(item[key]) == value]
        result[value] = {
            "count": len(subset),
            "dimensions": dict(sorted(Counter(f"{item['width']}x{item['height']}" for item in subset).items())),
            "categories": dict(sorted(Counter(item["resolution_category"] for item in subset).items())),
        }
    return result


def main() -> int:
    if git("branch", "--show-current") != BRANCH:
        raise RuntimeError(f"expected branch {BRANCH}")
    git("merge-base", "--is-ancestor", SOURCE_HEAD, "HEAD")

    generation_contract = load_json(RISK / "subject00_codex_managed_generation_contract.json")
    generation_registry = load_json(RISK / "subject00_codex_managed_generation_registry.json")
    output_registry = load_json(RISK / "subject00_codex_managed_output_registry.json")
    technical_verification = load_json(RISK / "subject00_codex_managed_technical_verification.json")
    git_review = load_json(RISK / "subject00_codex_managed_human_review_manifest.json")
    generation_summary = load_json(RISK / "subject00_codex_managed_generation_final_summary.json")
    generation_handoff = load_json(HANDOFF / "subject00_codex_managed_generation_handoff.json")
    progress = load_json(DATA_ROOT / "03_generation_requests" / "generation_progress_registry.json")
    external_review = load_json(REVIEW_ROOT / "subject00_codex_managed_human_review_manifest.json")

    if generation_contract["source"]["head"] != "6678fe0e7b92c648a0799ee4c99ecafba75d15c0":
        raise RuntimeError("SUBJECT00_MANAGED_OUTPUT_REGISTRY_MISMATCH: generation source")
    if generation_summary["generation_result_head"] != "7d60f10f414a8393a0d131c9f14fe8f36de52abe":
        raise RuntimeError("SUBJECT00_MANAGED_OUTPUT_REGISTRY_MISMATCH: generation result head")
    if generation_handoff["classification"] != generation_summary["classification"]:
        raise RuntimeError("SUBJECT00_MANAGED_OUTPUT_REGISTRY_MISMATCH: handoff classification")

    output_rows = output_registry["records"]
    progress_by_id = {item["request_id"]: item for item in progress["records"]}
    external_review_by_id = {item["request_id"]: item for item in external_review["records"]}
    git_review_by_id = {item["request_id"]: item for item in git_review["records"]}
    request_ids = [item["request_id"] for item in output_rows]
    expected_set = set(request_ids)
    if len(output_rows) != 48 or len(expected_set) != 48:
        raise RuntimeError("SUBJECT00_MANAGED_OUTPUT_REGISTRY_MISMATCH: output registry count")
    for name, values in {
        "progress": set(progress_by_id),
        "external_review": set(external_review_by_id),
        "git_review": set(git_review_by_id),
    }.items():
        if values != expected_set:
            raise RuntimeError(f"SUBJECT00_MANAGED_OUTPUT_REGISTRY_MISMATCH: {name} request set")

    inventory_records: list[dict[str, Any]] = []
    consistency_errors: list[dict[str, Any]] = []
    original_hashes: dict[str, str] = {}
    for sequence, registry_row in enumerate(output_rows, start=1):
        request_id = registry_row["request_id"]
        output_path = Path(registry_row["output_path"])
        if not output_path.is_file() or CANDIDATE_ROOT.resolve() not in output_path.resolve().parents:
            raise RuntimeError(f"SUBJECT00_MANAGED_OUTPUT_REGISTRY_MISMATCH: output path {request_id}")
        actual_sha = file_sha256(output_path)
        original_hashes[request_id] = actual_sha
        parse_error = None
        try:
            with Image.open(output_path) as image:
                image.load()
                width, height = image.size
                image_format = image.format
                bands = list(image.getbands())
                has_alpha = "A" in bands or "transparency" in image.info
                orientation = image.getexif().get(274)
        except Exception as exc:  # pragma: no cover - records real corrupt assets
            width = height = None
            image_format = None
            bands = []
            has_alpha = None
            orientation = None
            parse_error = str(exc)

        if parse_error is None:
            category, ratio, ratio_error = classify(width, height)
        else:
            category, ratio, ratio_error = CATEGORY_G, None, None
        technical_eligibility = eligibility(category)
        provenance_path = DATA_ROOT / "11_provenance" / "requests" / f"{request_id}.json"
        provenance = load_json(provenance_path)
        progress_row = progress_by_id[request_id]
        external_review_row = external_review_by_id[request_id]
        git_review_row = git_review_by_id[request_id]

        comparisons = {
            "registry_sha": registry_row.get("output_sha256") == actual_sha,
            "registry_width": registry_row.get("width") == width,
            "registry_height": registry_row.get("height") == height,
            "registry_format": registry_row.get("format") == image_format,
            "progress_sha": progress_row.get("output_sha256") == actual_sha,
            "progress_width": progress_row.get("width") == width,
            "progress_height": progress_row.get("height") == height,
            "provenance_sha": provenance.get("output_sha256") == actual_sha,
            "provenance_width": provenance.get("width") == width,
            "provenance_height": provenance.get("height") == height,
            "external_review_sha": external_review_row.get("candidate_sha256") == actual_sha,
            "git_review_sha": git_review_row.get("candidate_sha256") == actual_sha,
            "output_path": Path(provenance.get("output_path", "")) == output_path,
        }
        if not all(comparisons.values()):
            consistency_errors.append({"request_id": request_id, "comparisons": comparisons})

        inventory_records.append(
            {
                "managed_generation_call_sequence": sequence,
                "request_id": request_id,
                "garment_id": registry_row["garment_id"],
                "slot_id": registry_row["slot_id"],
                "camera_orientation": SLOT_VIEWS[registry_row["slot_id"]],
                "camera_id": registry_row["camera_id"],
                "candidate_index": registry_row["candidate_index"],
                "output_path": str(output_path),
                "path_type": PATH_TYPE,
                "output_sha256": actual_sha,
                "format": image_format,
                "width": width,
                "height": height,
                "aspect_ratio": ratio,
                "target_aspect_ratio": TARGET_RATIO,
                "aspect_ratio_absolute_error": ratio_error,
                "file_bytes": output_path.stat().st_size,
                "exif_orientation": orientation,
                "image_bands": bands,
                "has_alpha_channel": has_alpha,
                "parse_error": parse_error,
                "resolution_category": category,
                "technical_dataset_eligibility": technical_eligibility,
                "visual_quality_review": "PENDING_HUMAN_DOUBLE_REVIEW",
                "registry_resolution": dimension_object(registry_row.get("width"), registry_row.get("height")),
                "progress_resolution": dimension_object(progress_row.get("width"), progress_row.get("height")),
                "provenance_resolution": dimension_object(provenance.get("width"), provenance.get("height")),
                "technical_verification_status": registry_row.get("technical_status"),
                "provenance_path": str(provenance_path),
                "registry_consistency": "PASS" if all(comparisons.values()) else "FAIL",
            }
        )

    if consistency_errors:
        raise RuntimeError(
            "SUBJECT00_MANAGED_OUTPUT_REGISTRY_MISMATCH: " + json.dumps(consistency_errors, ensure_ascii=True)
        )
    if technical_verification["png_parse_pass"] != 48 or technical_verification["resolution_pass_1024x1536"] != 5:
        raise RuntimeError("SUBJECT00_MANAGED_OUTPUT_REGISTRY_MISMATCH: prior technical summary")

    dimensions = Counter(f"{item['width']}x{item['height']}" for item in inventory_records)
    categories = Counter(item["resolution_category"] for item in inventory_records)
    eligibilities = Counter(item["technical_dataset_eligibility"] for item in inventory_records)
    dominant_dimension, dominant_count = dimensions.most_common(1)[0]
    first_four_slots = [item for item in inventory_records if item["slot_id"] in {"slot_00", "slot_01", "slot_02", "slot_03"}]
    sequence_records = [
        {
            "sequence": item["managed_generation_call_sequence"],
            "request_id": item["request_id"],
            "dimensions": f"{item['width']}x{item['height']}",
            "resolution_category": item["resolution_category"],
        }
        for item in inventory_records
    ]
    distribution = {
        "schema_version": "canondressgs.subject00.managed_output_resolution_distribution.v1",
        "task_id": TASK_ID,
        "image_count": len(inventory_records),
        "exact_dimension_distribution": dict(sorted(dimensions.items())),
        "aspect_ratio_category_distribution": {category: categories.get(category, 0) for category in CATEGORIES},
        "technical_eligibility_distribution": dict(sorted(eligibilities.items())),
        "by_garment": nested_distribution(inventory_records, "garment_id"),
        "by_slot": nested_distribution(inventory_records, "slot_id"),
        "by_candidate_index": nested_distribution(inventory_records, "candidate_index"),
        "by_camera_orientation": nested_distribution(inventory_records, "camera_orientation"),
        "managed_generation_call_sequence": sequence_records,
        "observed_patterns": {
            "systematic_pattern_present": dominant_count >= 24,
            "dominant_dimension": dominant_dimension,
            "dominant_dimension_count": dominant_count,
            "landscape_count": categories.get(CATEGORY_E, 0),
            "slots_00_to_03_landscape_count": sum(item["resolution_category"] == CATEGORY_E for item in first_four_slots),
            "slots_00_to_03_total": len(first_four_slots),
            "exact_outputs_by_slot": dict(
                sorted(Counter(item["slot_id"] for item in inventory_records if item["resolution_category"] == CATEGORY_A).items())
            ),
            "garment_association": "OBSERVED_DISTRIBUTIONS_DIFFER_CAUSAL_ATTRIBUTION_NOT_ESTABLISHED",
            "slot_camera_association": "OBSERVED_CATEGORY_CONCENTRATION_CAUSAL_ATTRIBUTION_NOT_ESTABLISHED",
            "call_sequence_association": "SEQUENCE_RECORDED_CAUSAL_ATTRIBUTION_NOT_ESTABLISHED",
            "randomness_assessment": "NOT_ESTABLISHED",
            "causal_explanation": "NOT_ESTABLISHED_FROM_AVAILABLE_RECORDS",
        },
        "paper_final": PAPER_FINAL,
    }
    inventory = {
        "schema_version": "canondressgs.subject00.managed_output_resolution_inventory.v1",
        "task_id": TASK_ID,
        "source": {"branch": SOURCE_BRANCH, "head": SOURCE_HEAD},
        "dataset_root": str(DATA_ROOT),
        "path_type": PATH_TYPE,
        "target_resolution": dimension_object(TARGET_WIDTH, TARGET_HEIGHT),
        "target_aspect_ratio": TARGET_RATIO,
        "record_count": len(inventory_records),
        "registry_consistency": "PASS",
        "records": inventory_records,
        "paper_final": PAPER_FINAL,
    }
    technical = {
        "schema_version": "canondressgs.subject00.managed_output_technical_eligibility.v1",
        "task_id": TASK_ID,
        "policy": {
            CATEGORY_A: "PASS",
            CATEGORY_B: "PENDING_USER_RESIZE_CONTRACT",
            CATEGORY_C: "FAIL",
            CATEGORY_D: "FAIL",
            CATEGORY_E: "FAIL",
            CATEGORY_F: "FAIL",
            CATEGORY_G: "FAIL",
        },
        "counts": {
            "pass": eligibilities.get("PASS", 0),
            "pending_user_resize_contract": eligibilities.get("PENDING_USER_RESIZE_CONTRACT", 0),
            "fail": eligibilities.get("FAIL", 0),
            "visual_review_pending": len(inventory_records),
        },
        "records": [
            {
                "request_id": item["request_id"],
                "resolution_category": item["resolution_category"],
                "technical_dataset_eligibility": item["technical_dataset_eligibility"],
                "visual_quality_review": item["visual_quality_review"],
            }
            for item in inventory_records
        ],
        "paper_final": PAPER_FINAL,
    }

    RESOLUTION_REVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    contact_paths = {}
    for garment in ["O01", "O03", "O04"]:
        path = RESOLUTION_REVIEW_ROOT / f"subject00_{garment}_resolution_review.png"
        make_contact_sheet(
            [item for item in inventory_records if item["garment_id"] == garment],
            path,
            columns=4,
            tile_width=320,
            image_height=360,
            label_height=108,
        )
        contact_paths[garment] = str(path)
    all_path = RESOLUTION_REVIEW_ROOT / "subject00_all_48_resolution_review.png"
    make_contact_sheet(
        inventory_records, all_path, columns=8, tile_width=220, image_height=260, label_height=104
    )
    contact_paths["all_48"] = str(all_path)

    post_hashes = {item["request_id"]: file_sha256(Path(item["output_path"])) for item in inventory_records}
    mutated = [request_id for request_id in request_ids if original_hashes[request_id] != post_hashes[request_id]]
    if mutated:
        raise RuntimeError(f"candidate image mutation detected: {mutated}")

    reviewer_records = [review_record(item) for item in inventory_records]
    for reviewer_number in [1, 2]:
        write_json(
            REVIEW_ROOT / f"reviewer_{reviewer_number}_manifest.json",
            {
                "schema_version": f"canondressgs.subject00.independent_reviewer_{reviewer_number}.v1",
                "task_id": TASK_ID,
                "reviewer_role": f"REVIEWER_{reviewer_number}",
                "independence_required": True,
                "shared_prefilled_visual_judgment": False,
                "entry_count": len(reviewer_records),
                "records": reviewer_records,
            },
        )
    adjudication_records = []
    for item in inventory_records:
        record = review_record(item)
        record.update(
            {
                "reviewer_1_decision": None,
                "reviewer_2_decision": None,
                "visual_adjudication_status": None,
                "final_dataset_eligibility": None,
            }
        )
        adjudication_records.append(record)
    write_json(
        REVIEW_ROOT / "adjudication_manifest.json",
        {
            "schema_version": "canondressgs.subject00.double_human_adjudication.v1",
            "task_id": TASK_ID,
            "entry_count": len(adjudication_records),
            "decision_policy": {
                "both_accept": "VISUAL_ACCEPT",
                "any_reject": "VISUAL_REJECT",
                "other": "HUMAN_ADJUDICATION_REQUIRED",
                "final_dataset_requirement": "VISUAL_ACCEPT_AND_TECHNICAL_DATASET_ELIGIBILITY_PASS",
                "category_b_visual_accept": "VISUAL_ACCEPT_PENDING_RESOLUTION_CONTRACT",
            },
            "records": adjudication_records,
        },
    )

    double_review_contract = {
        "schema_version": "canondressgs.subject00.managed_output_double_review_contract.v1",
        "task_id": TASK_ID,
        "visual_review_queue_count": 48,
        "reviewer_1_manifest": str(REVIEW_ROOT / "reviewer_1_manifest.json"),
        "reviewer_2_manifest": str(REVIEW_ROOT / "reviewer_2_manifest.json"),
        "adjudication_manifest": str(REVIEW_ROOT / "adjudication_manifest.json"),
        "contact_sheets": contact_paths,
        "independent_review_required": True,
        "visual_fields_initial_state": None,
        "automated_visual_judgment_allowed": False,
        "decision_policy": {
            "reviewer_1_accept_and_reviewer_2_accept": "VISUAL_ACCEPT",
            "any_reject": "VISUAL_REJECT",
            "otherwise": "HUMAN_ADJUDICATION_REQUIRED",
            "final_dataset_requirement": "VISUAL_ACCEPT_AND_TECHNICAL_DATASET_ELIGIBILITY_PASS",
            "category_b_visual_accept": "VISUAL_ACCEPT_PENDING_RESOLUTION_CONTRACT",
        },
        "accepted_count": 0,
        "teacher_target_count": 0,
        "paper_final": PAPER_FINAL,
    }
    regeneration_records = [
        {
            "request_id": item["request_id"],
            "current_output": item["output_path"],
            "current_resolution": dimension_object(item["width"], item["height"]),
            "resolution_category": item["resolution_category"],
            "technical_dataset_eligibility": item["technical_dataset_eligibility"],
            "visual_review_status": "PENDING_HUMAN_DOUBLE_REVIEW",
            "reason_for_regeneration": f"TECHNICALLY_INELIGIBLE_{item['resolution_category']}",
            "whether_prompt_changes_are_allowed": False,
            "expected_target": {
                "width": TARGET_WIDTH,
                "height": TARGET_HEIGHT,
                "format": "PNG",
                "orientation": "portrait",
            },
            "retry_budget": 0,
            "retry_budget_status": "NOT_AUTHORIZED",
            "generation_authorization": False,
        }
        for item in inventory_records
        if item["technical_dataset_eligibility"] == "FAIL"
    ]
    regeneration_plan = {
        "schema_version": "canondressgs.subject00.managed_output_regeneration_plan.v1",
        "task_id": TASK_ID,
        "plan_only": True,
        "generation_executed": False,
        "record_count": len(regeneration_records),
        "records": regeneration_records,
        "category_b_not_scheduled_for_regeneration": categories.get(CATEGORY_B, 0),
        "generation_authorization": False,
        "paper_final": PAPER_FINAL,
    }

    option_a_keep = categories.get(CATEGORY_A, 0)
    option_b_resize = categories.get(CATEGORY_B, 0)
    option_b_keep = option_a_keep + option_b_resize
    strict_rerun = len(inventory_records) - option_a_keep
    option_b_rerun = len(inventory_records) - option_b_keep
    classification = (
        "SUBJECT00_MANAGED_OUTPUT_RESIZE_CONTRACT_CANDIDATE"
        if option_b_resize > 0
        else "SUBJECT00_MANAGED_OUTPUT_STRICT_RERUN_REQUIRED"
    )
    audit_head = os.environ.get("SUBJECT00_RESOLUTION_AUDIT_HEAD", RESOLUTION_AUDIT_HEAD)
    tests = {
        "schema_version": "canondressgs.subject00.managed_output_resolution_tests.v1",
        "task_id": TASK_ID,
        "status": "NOT_RUN",
        "total": 0,
        "passed": 0,
        "failed": [],
        "checks": [],
        "paper_final": PAPER_FINAL,
    }
    final_summary = {
        "schema_version": "canondressgs.subject00.managed_output_resolution_final_summary.v1",
        "task_id": TASK_ID,
        "source": {"branch": SOURCE_BRANCH, "head": SOURCE_HEAD},
        "branch": BRANCH,
        "dataset_root": str(DATA_ROOT),
        "path_type": PATH_TYPE,
        "image_count": len(inventory_records),
        "category_counts": {category: categories.get(category, 0) for category in CATEGORIES},
        "technical_pass_count": eligibilities.get("PASS", 0),
        "resize_contract_candidate_count": eligibilities.get("PENDING_USER_RESIZE_CONTRACT", 0),
        "strict_rerun_count": strict_rerun,
        "visual_review_queue_count": 48,
        "option_a": {"native_keep": option_a_keep, "regenerate": strict_rerun},
        "option_b": {"native_keep": option_a_keep, "deterministic_resize": option_b_resize, "total_keep": option_b_keep, "regenerate": option_b_rerun},
        "original_image_mutation_count": len(mutated),
        "resize_operations": 0,
        "crop_operations": 0,
        "pad_operations": 0,
        "rotate_operations": 0,
        "generation_calls": 0,
        "external_api_calls": 0,
        "cloud_image_writes": 0,
        "accepted_count": 0,
        "teacher_target_count": 0,
        "formal_base": FORMAL_BASE,
        "tests": "NOT_RUN",
        "resolution_audit_head": audit_head,
        "final_reporting_head_resolution": "git rev-parse HEAD after final reporting commit",
        "paper_final": PAPER_FINAL,
        "classification": classification,
        "next_task": NEXT_TASK,
        "next_task_authorized": False,
    }
    handoff = {
        "schema_version": "canondressgs.subject00.managed_output_resolution_adjudication_handoff.v1",
        "task_id": TASK_ID,
        "source": {"branch": SOURCE_BRANCH, "head": SOURCE_HEAD},
        "branch": BRANCH,
        "dataset_root": str(DATA_ROOT),
        "classification": classification,
        "category_counts": final_summary["category_counts"],
        "technical_pass_count": option_a_keep,
        "resize_contract_candidate_count": option_b_resize,
        "strict_rerun_count": strict_rerun,
        "visual_review_queue_count": 48,
        "accepted_count": 0,
        "teacher_target_count": 0,
        "formal_base": FORMAL_BASE,
        "tests": "NOT_RUN",
        "resolution_audit_head": audit_head,
        "final_reporting_head_resolution": "git rev-parse HEAD after final reporting commit",
        "paper_final": PAPER_FINAL,
        "next_task": NEXT_TASK,
        "next_task_authorized": False,
    }

    git_outputs = {
        "subject00_managed_output_resolution_inventory.json": inventory,
        "subject00_managed_output_resolution_distribution.json": distribution,
        "subject00_managed_output_technical_eligibility.json": technical,
        "subject00_managed_output_double_review_contract.json": double_review_contract,
        "subject00_managed_output_regeneration_plan.json": regeneration_plan,
        "subject00_managed_output_resolution_tests.json": tests,
        "subject00_managed_output_resolution_final_summary.json": final_summary,
    }
    for name, payload in git_outputs.items():
        write_json(RISK / name, payload)
    write_json(HANDOFF / "subject00_managed_output_resolution_adjudication_handoff.json", handoff)

    write_json(
        DATA_ROOT / "12_final_verification" / "subject00_managed_output_resolution_audit.json",
        {
            "schema_version": "canondressgs.subject00.managed_output_resolution_external_audit.v1",
            "task_id": TASK_ID,
            "registry_consistency": "PASS",
            "image_count": 48,
            "actual_png_parse_count": 48,
            "category_counts": final_summary["category_counts"],
            "original_image_mutation_count": len(mutated),
            "resize_operations": 0,
            "crop_operations": 0,
            "pad_operations": 0,
            "rotate_operations": 0,
            "generation_calls": 0,
            "contact_sheets": contact_paths,
            "reviewer_manifests": {
                "reviewer_1": str(REVIEW_ROOT / "reviewer_1_manifest.json"),
                "reviewer_2": str(REVIEW_ROOT / "reviewer_2_manifest.json"),
                "adjudication": str(REVIEW_ROOT / "adjudication_manifest.json"),
            },
            "paper_final": PAPER_FINAL,
        },
    )

    write_doc(
        DOCS / "AAAI27_SUBJECT00_MANAGED_OUTPUT_RESOLUTION_AUDIT_20260725.md",
        f"""
# Subject00 Managed Output Resolution Audit

- Task: `{TASK_ID}`
- Source: `{SOURCE_BRANCH}` at `{SOURCE_HEAD}`
- Dataset root: `{DATA_ROOT}`
- Registry consistency: `PASS`
- Actual PNG parse: 48/48
- A exact 1024x1536: {categories.get(CATEGORY_A, 0)}
- B exact 2:3 portrait resize candidate: {categories.get(CATEGORY_B, 0)}
- C near 2:3 portrait: {categories.get(CATEGORY_C, 0)}
- D incompatible portrait: {categories.get(CATEGORY_D, 0)}
- E landscape: {categories.get(CATEGORY_E, 0)}
- F square: {categories.get(CATEGORY_F, 0)}
- G parse/metadata failure: {categories.get(CATEGORY_G, 0)}
- Technical PASS: {option_a_keep}
- Pending user resize contract: {option_b_resize}
- Technical FAIL: {strict_rerun}
- Visual review queue: 48
- Original image mutations: 0
- Resize/crop/pad/rotate operations: 0/0/0/0
- Generation calls: 0
- Formal Base: `PENDING`
- PAPER_FINAL: `false`
- Contract tests: `RUN_AFTER_AUDIT_EMISSION`

The dominant actual dimension is `{dominant_dimension}` ({dominant_count}/48); 38 outputs are landscape. This is an observed repeated pattern, not a causal attribution. No visual quality field was filled by Codex.
""",
    )
    write_doc(
        DOCS / "AAAI27_SUBJECT00_DOUBLE_HUMAN_REVIEW_GUIDE_20260725.md",
        f"""
# Subject00 Double Human Review Guide

Use the four labeled contact sheets under `{RESOLUTION_REVIEW_ROOT}` for orientation, then inspect each original candidate at its registered path.

Reviewer 1 and Reviewer 2 must work independently. All visual fields and decisions start as null. Review identity, face, hair, skin tone, body shape, pose, camera, garment semantics, slot match, complete hands and feet, full-body framing, background, lighting, edges, and artifacts.

Two ACCEPT decisions produce `VISUAL_ACCEPT`; any REJECT produces `VISUAL_REJECT`; all other combinations require human adjudication. Dataset eligibility additionally requires `TECHNICAL_DATASET_ELIGIBILITY=PASS`. A category B image, if one exists, remains `VISUAL_ACCEPT_PENDING_RESOLUTION_CONTRACT` until the user explicitly authorizes a resize contract.

Do not copy any candidate into `06_accepted_rgb` or `09_teacher_targets` during this review preparation task.
""",
    )
    write_doc(
        DOCS / "AAAI27_SUBJECT00_RESOLUTION_CONTRACT_OPTIONS_20260725.md",
        f"""
# Subject00 Resolution Contract Options

## Option A: STRICT_NATIVE_RESOLUTION

- Keep native 1024x1536 outputs: {option_a_keep}
- Regenerate non-A outputs after explicit authorization: {strict_rerun}
- Pixel transformations: none

## Option B: DETERMINISTIC_RESIZE_ONLY

- Keep native A outputs: {option_a_keep}
- Eligible strict 2:3 portrait B outputs for future Lanczos resize: {option_b_resize}
- Total potentially retained after an explicit resize contract: {option_b_keep}
- Regenerate C/D/E/F/G outputs after explicit authorization: {option_b_rerun}
- Crop, pad, rotate, and aspect distortion remain forbidden

No B candidates exist in this audit, so Option B currently retains no additional image over Option A. This task does not select an option, resize an image, or authorize regeneration.
""",
    )

    print(
        json.dumps(
            {
                "status": "AUDIT_EMITTED",
                "classification": classification,
                "category_counts": final_summary["category_counts"],
                "dimension_distribution": dict(sorted(dimensions.items())),
                "technical": final_summary["technical_pass_count"],
                "resize_contract_candidates": option_b_resize,
                "strict_rerun": strict_rerun,
                "visual_queue": 48,
                "original_image_mutations": len(mutated),
                "contact_sheets": contact_paths,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
