#!/usr/bin/env python3
"""Manage the local, resumable Subject00 Codex managed image-edit run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
DOCS = ROOT / "docs" / "PAPER"
HANDOFF = ROOT / "project_control_handoff"
DATA_ROOT = Path(r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-GENERATION-001\attempt_001")
CACHE_ROOT = Path(r"E:\model_train\_subject00_identity_audit_tmp")

TASK_ID = "AAAI27-SUBJECT00-CODEX-MANAGED-GENERATION-001"
SOURCE_BRANCH = "research/subject00-generation-backend-adjudication-20260725"
SOURCE_HEAD = "6678fe0e7b92c648a0799ee4c99ecafba75d15c0"
DATA_PREP_BRANCH = "research/subject00-data-preparation-generation-ready-20260725"
DATA_PREP_HEAD = "4e5e76a2a19ffdb94385660ec1d8ea4ce4c5d6b3"
BRANCH = "research/subject00-codex-managed-generation-20260725"
DATA_PATH_TYPE = "WINDOWS_LOCAL_EXTERNAL_DATASET_PATH"
FORMAL_DEPENDENCY = "PENDING"
PRE_FORMAL_LABEL = "PRE_FORMAL_BASE_GENERATED_CANDIDATE"
DOWNSTREAM_BLOCK = "DOWNSTREAM_USE_BLOCKED_UNTIL_FORMAL_BASE_AND_HUMAN_REVIEW"
TARGET_SIZE = (1024, 1536)
MIN_FREE_BYTES = 10 * 1024**3
GENERATION_RESULT_HEAD = "RESOLVE_AFTER_GENERATION_METADATA_COMMIT"

DIRECTORIES = [
    "00_contract",
    "01_identity_sources",
    "02_garment_references",
    "03_generation_requests",
    "04_generation_responses",
    "05_human_review",
    "06_accepted_rgb",
    "07_accepted_masks",
    "08_camera_pose",
    "09_teacher_targets",
    "10_controller_splits",
    "11_provenance",
    "12_final_verification",
]

SLOT_CACHE = {
    "slot_00": (17, "front"),
    "slot_01": (21, "front-left three-quarter"),
    "slot_02": (14, "front-right three-quarter"),
    "slot_03": (23, "left side"),
    "slot_04": (11, "right side"),
    "slot_05": (2, "back-left three-quarter"),
    "slot_06": (9, "back-right three-quarter"),
    "slot_07": (5, "back"),
}

REVIEW_FIELDS = [
    "identity_match",
    "face_match",
    "hair_match",
    "skin_tone_match",
    "body_shape_match",
    "pose_match",
    "camera_match",
    "garment_match",
    "garment_slot_match",
    "mask_quality",
    "edge_quality",
    "hands_feet_complete",
    "background_match",
    "lighting_match",
    "artifact_grade",
    "reviewer_1",
    "reviewer_2",
    "reject_reason",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(value: Any) -> str:
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


def seal(payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    payload.pop("content_sha256", None)
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


def write_json(path: Path, payload: dict[str, Any], sealed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if sealed:
        payload = seal(payload)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def write_doc(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8", newline="\n")


def git(*args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8")
    return completed.stdout.strip()


def source_manifest() -> dict[str, Any]:
    return load_json(RISK / "subject00_generation_request_manifest.json")


def prompt_registry() -> dict[str, Any]:
    return load_json(RISK / "subject00_generation_prompt_registry.json")


def progress_path() -> Path:
    return DATA_ROOT / "03_generation_requests" / "generation_progress_registry.json"


def provenance_path(request_id: str) -> Path:
    return DATA_ROOT / "11_provenance" / "requests" / f"{request_id}.json"


def request_path(request_id: str) -> Path:
    return DATA_ROOT / "03_generation_requests" / "requests" / f"{request_id}.json"


def output_path(row: dict[str, Any]) -> Path:
    return DATA_ROOT / "04_generation_responses" / "codex_managed_candidates" / row["garment_id"] / f"{row['request_id']}.png"


def local_source_path(row: dict[str, Any]) -> Path:
    camera, _ = SLOT_CACHE[row["slot_id"]]
    return CACHE_ROOT / f"cam{camera:02d}_frame00000000.jpg"


def actual_prompt(row: dict[str, Any], prompt: dict[str, Any]) -> str:
    _, view = SLOT_CACHE[row["slot_id"]]
    return "\n".join(
        [
            "Use case: identity-preserve",
            f"Asset type: Subject00 pre-Formal candidate {row['request_id']}",
            f"Primary request: {prompt['positive_prompt']}",
            "Input images: Image 1 is the sole identity, body, pose, camera, framing, background, and lighting authority.",
            f"Camera/view invariant: preserve exactly the Image 1 {view} view, camera {row['camera_id']}, pose/frame 0.",
            f"Garment invariant: apply only {prompt['semantic_description']}",
            "Composition/framing: one person, full body from head through both feet, portrait 1024x1536, no crop or reframing.",
            "Constraints: edit clothing only; preserve the same face, visible hairline and hairstyle, skin tone, body proportions, height ratio, pose, hands, feet, shoes, camera orientation, room background, and lighting.",
            f"Avoid: {prompt['negative_constraints']}",
            "Output: exactly one clean PNG candidate; no text, number, label, watermark, border, collage, or extra person.",
        ]
    )


def validate_source_row(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if row["identity_id"] != "subject00":
        errors.append("IDENTITY_NOT_SUBJECT00")
    expected_camera, _ = SLOT_CACHE[row["slot_id"]]
    if row["camera_id"] != expected_camera:
        errors.append("CAMERA_SLOT_MISMATCH")
    if row["pose_frame_id"] != 0:
        errors.append("POSE_NOT_ZERO")
    path = local_source_path(row)
    if not path.is_file():
        errors.append("LOCAL_INPUT_MISSING")
    else:
        if file_sha256(path) != row["input_sha256"]["identity_condition_rgb"]:
            errors.append("LOCAL_INPUT_SHA_MISMATCH")
        with Image.open(path) as image:
            if image.size != (1330, 1150):
                errors.append("LOCAL_INPUT_RESOLUTION_MISMATCH")
    return errors


def progress_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(record["status"] for record in records)
    return {
        "total": len(records),
        "pending": counts["PENDING"],
        "input_blocked": counts["INPUT_BINDING_BLOCKED"],
        "retry_pending": counts["TECHNICAL_RETRY_PENDING"],
        "generated_technical_pass": counts["GENERATED_TECHNICAL_PASS"],
        "generated_technical_review": counts["GENERATED_TECHNICAL_REVIEW"],
        "failed": counts["FAILED"],
    }


def refresh_progress(payload: dict[str, Any]) -> None:
    payload["counts"] = progress_counts(payload["records"])
    payload["updated_at"] = now()
    write_json(progress_path(), payload)


def prepare() -> None:
    if git("branch", "--show-current") != BRANCH:
        raise RuntimeError(f"expected branch {BRANCH}")
    git("merge-base", "--is-ancestor", SOURCE_HEAD, "HEAD")
    free = shutil.disk_usage(DATA_ROOT.drive + "\\").free
    if free < MIN_FREE_BYTES:
        raise RuntimeError("SUBJECT00_CODEX_MANAGED_GENERATION_LOCAL_STORAGE_BLOCKED")

    manifest = source_manifest()
    prompts = {record["garment_id"]: record for record in prompt_registry()["records"]}
    rows = manifest["requests"]
    if len(rows) != 48:
        raise RuntimeError("frozen request count is not 48")
    for directory in DIRECTORIES:
        (DATA_ROOT / directory).mkdir(parents=True, exist_ok=True)
    for path in [
        DATA_ROOT / "03_generation_requests" / "requests",
        DATA_ROOT / "04_generation_responses" / "codex_managed_candidates" / "O01",
        DATA_ROOT / "04_generation_responses" / "codex_managed_candidates" / "O03",
        DATA_ROOT / "04_generation_responses" / "codex_managed_candidates" / "O04",
        DATA_ROOT / "05_human_review" / "contact_sheets",
        DATA_ROOT / "11_provenance" / "requests",
    ]:
        path.mkdir(parents=True, exist_ok=True)

    previous = load_json(progress_path()) if progress_path().is_file() else None
    previous_by_id = {record["request_id"]: record for record in previous["records"]} if previous else {}
    progress_records = []
    input_pass = 0
    input_blocked = 0
    for row in rows:
        errors = validate_source_row(row)
        prompt = prompts[row["garment_id"]]
        request_payload = {
            "schema_version": "canondressgs.subject00.codex_managed_request.v1",
            "task_id": TASK_ID,
            "request_id": row["request_id"],
            "source_request_manifest_sha256": row["request_manifest_sha256"],
            "source_request_set_sha256": manifest["request_set_sha256"],
            "identity_id": row["identity_id"],
            "identity_source_set_id": row["identity_source_set_id"],
            "garment_id": row["garment_id"],
            "slot_id": row["slot_id"],
            "semantic_pose_slot": row["semantic_pose_slot"],
            "candidate_index": row["candidate_index"],
            "camera_id": row["camera_id"],
            "pose_frame_id": row["pose_frame_id"],
            "source_input_paths": row["input_file_paths"],
            "source_input_sha256": row["input_sha256"],
            "managed_tool_local_input_path": str(local_source_path(row)),
            "prompt_id": row["prompt_id"],
            "prompt_sha256": row["prompt_sha256"],
            "negative_constraint_id": row["negative_constraint_id"],
            "negative_constraint_sha256": row["negative_constraint_sha256"],
            "actual_prompt": actual_prompt(row, prompt),
            "actual_prompt_sha256": hashlib.sha256(actual_prompt(row, prompt).encode("utf-8")).hexdigest(),
            "generation_backend": "CODEX_MANAGED_IMAGE_EDIT",
            "generation_mode": "IMAGE_EDIT",
            "provider_response_available": False,
            "provider_model_id": "NOT_EXPOSED_BY_MANAGED_TOOL",
            "provider_revision": "NOT_EXPOSED_BY_MANAGED_TOOL",
            "target_resolution": {"width": 1024, "height": 1536, "orientation": "portrait"},
            "output_path": str(output_path(row)),
            "path_type": DATA_PATH_TYPE,
            "input_validation": {"status": "PASS" if not errors else "INPUT_BINDING_BLOCKED", "errors": errors},
            "authorized": True,
            "executed": False,
            "pre_formal_base_label": PRE_FORMAL_LABEL,
            "downstream_use_status": DOWNSTREAM_BLOCK,
        }
        write_json(request_path(row["request_id"]), request_payload)
        if errors:
            input_blocked += 1
            record = {
                "request_id": row["request_id"],
                "garment_id": row["garment_id"],
                "slot_id": row["slot_id"],
                "candidate_index": row["candidate_index"],
                "status": "INPUT_BINDING_BLOCKED",
                "technical_retry_count": 0,
                "failure_reason": ";".join(errors),
            }
        else:
            input_pass += 1
            record = previous_by_id.get(row["request_id"])
            if record and record["status"] in {"GENERATED_TECHNICAL_PASS", "GENERATED_TECHNICAL_REVIEW"}:
                out = Path(record["output_path"])
                if not out.is_file() or file_sha256(out) != record["output_sha256"]:
                    record = None
            if not record:
                record = {
                    "request_id": row["request_id"],
                    "garment_id": row["garment_id"],
                    "slot_id": row["slot_id"],
                    "candidate_index": row["candidate_index"],
                    "status": "PENDING",
                    "technical_retry_count": 0,
                    "failure_reason": None,
                }
        progress_records.append(record)

    contract = {
        "schema_version": "canondressgs.subject00.codex_managed_generation_external_contract.v1",
        "task_id": TASK_ID,
        "source": {"branch": SOURCE_BRANCH, "head": SOURCE_HEAD},
        "branch": BRANCH,
        "dataset_root": str(DATA_ROOT),
        "path_type": DATA_PATH_TYPE,
        "directories": DIRECTORIES,
        "request_count": len(rows),
        "request_set_sha256": manifest["request_set_sha256"],
        "input_binding_pass": input_pass,
        "input_binding_blocked": input_blocked,
        "backend": "CODEX_MANAGED_IMAGE_EDIT",
        "external_api_allowed": False,
        "technical_retry_limit": 1,
        "formal_base_dependency": FORMAL_DEPENDENCY,
        "pre_formal_base_label": PRE_FORMAL_LABEL,
        "accepted_directories_must_remain_empty": ["06_accepted_rgb", "09_teacher_targets"],
        "created_at": now(),
    }
    write_json(DATA_ROOT / "00_contract" / "subject00_codex_managed_generation_contract.json", contract)
    progress = {
        "schema_version": "canondressgs.subject00.codex_managed_progress.v1",
        "task_id": TASK_ID,
        "request_set_sha256": manifest["request_set_sha256"],
        "stable_order": [row["request_id"] for row in rows],
        "records": progress_records,
        "counts": {},
        "api_calls": 0,
        "sublyx_calls": 0,
        "code78_calls": 0,
        "api_key_reads": 0,
        "cloud_image_writes": 0,
        "created_at": previous["created_at"] if previous else now(),
        "updated_at": now(),
    }
    refresh_progress(progress)
    print(json.dumps({"status": "PREPARED", "dataset_root": str(DATA_ROOT), "input_pass": input_pass, "input_blocked": input_blocked, "counts": progress["counts"]}, indent=2))


def next_request() -> None:
    progress = load_json(progress_path())
    for record in progress["records"]:
        if record["status"] in {"PENDING", "TECHNICAL_RETRY_PENDING"}:
            payload = load_json(request_path(record["request_id"]))
            payload["technical_retry_count"] = record["technical_retry_count"]
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return
    print(json.dumps({"status": "NO_PENDING_REQUEST"}))


def record_success(args: argparse.Namespace) -> None:
    source = Path(args.generated_source)
    if not source.is_file():
        raise FileNotFoundError(source)
    progress = load_json(progress_path())
    record = next((item for item in progress["records"] if item["request_id"] == args.request_id), None)
    if record is None:
        raise KeyError(args.request_id)
    if record["status"] in {"GENERATED_TECHNICAL_PASS", "GENERATED_TECHNICAL_REVIEW"}:
        raise RuntimeError("successful request must not be generated again")
    request = load_json(request_path(args.request_id))
    target = Path(request["output_path"])
    with Image.open(source) as image:
        image.load()
        width, height = image.size
        image_format = image.format
    if image_format != "PNG":
        raise RuntimeError(f"generated source is {image_format}, not PNG")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if file_sha256(target) != file_sha256(source):
            raise RuntimeError("refusing to overwrite a different candidate")
    else:
        shutil.copy2(source, target)
    sha = file_sha256(target)
    technical_status = "PASS" if (width, height) == TARGET_SIZE else "OUTPUT_RESOLUTION_MISMATCH"
    record.update(
        {
            "status": "GENERATED_TECHNICAL_PASS" if technical_status == "PASS" else "GENERATED_TECHNICAL_REVIEW",
            "output_path": str(target),
            "output_sha256": sha,
            "output_bytes": target.stat().st_size,
            "width": width,
            "height": height,
            "format": image_format,
            "technical_status": technical_status,
            "failure_reason": None,
            "ended_at": args.ended_at,
        }
    )
    provenance = {
        "schema_version": "canondressgs.subject00.codex_managed_request_provenance.v1",
        "task_id": TASK_ID,
        "request_id": args.request_id,
        "source_request_manifest_sha256": request["source_request_manifest_sha256"],
        "source_request_set_sha256": request["source_request_set_sha256"],
        "input_paths": request["source_input_paths"],
        "input_sha256": request["source_input_sha256"],
        "managed_tool_local_input_path": request["managed_tool_local_input_path"],
        "garment_id": request["garment_id"],
        "slot_id": request["slot_id"],
        "camera_id": request["camera_id"],
        "pose_frame_id": request["pose_frame_id"],
        "prompt_id": request["prompt_id"],
        "actual_prompt": request["actual_prompt"],
        "actual_prompt_sha256": request["actual_prompt_sha256"],
        "generation_backend": "CODEX_MANAGED_IMAGE_EDIT",
        "generation_mode": "IMAGE_EDIT",
        "provider_response_available": False,
        "provider_model_id": "NOT_EXPOSED_BY_MANAGED_TOOL",
        "provider_revision": "NOT_EXPOSED_BY_MANAGED_TOOL",
        "tool_output_hint": args.tool_output_hint,
        "output_path": str(target),
        "path_type": DATA_PATH_TYPE,
        "output_sha256": sha,
        "output_bytes": target.stat().st_size,
        "width": width,
        "height": height,
        "format": image_format,
        "started_at": args.started_at,
        "ended_at": args.ended_at,
        "success": True,
        "technical_status": technical_status,
        "failure_reason": None,
        "technical_retry_count": int(args.technical_retry_count),
        "human_review_status": "PENDING_HUMAN_DOUBLE_REVIEW",
        "pre_formal_base_label": PRE_FORMAL_LABEL,
        "downstream_use_status": DOWNSTREAM_BLOCK,
    }
    write_json(provenance_path(args.request_id), provenance)
    refresh_progress(progress)
    print(json.dumps({"request_id": args.request_id, "status": record["status"], "output_path": str(target), "sha256": sha, "width": width, "height": height}, indent=2))


def record_failure(args: argparse.Namespace) -> None:
    progress = load_json(progress_path())
    record = next((item for item in progress["records"] if item["request_id"] == args.request_id), None)
    if record is None:
        raise KeyError(args.request_id)
    retry_count = int(args.technical_retry_count)
    record.update(
        {
            "status": "TECHNICAL_RETRY_PENDING" if retry_count < 1 else "FAILED",
            "technical_retry_count": retry_count + 1 if retry_count < 1 else retry_count,
            "failure_reason": args.reason,
            "ended_at": args.ended_at,
        }
    )
    request = load_json(request_path(args.request_id))
    provenance = {
        "schema_version": "canondressgs.subject00.codex_managed_request_provenance.v1",
        "task_id": TASK_ID,
        "request_id": args.request_id,
        "source_request_manifest_sha256": request["source_request_manifest_sha256"],
        "input_paths": request["source_input_paths"],
        "input_sha256": request["source_input_sha256"],
        "generation_backend": "CODEX_MANAGED_IMAGE_EDIT",
        "generation_mode": "IMAGE_EDIT",
        "provider_response_available": False,
        "provider_model_id": "NOT_EXPOSED_BY_MANAGED_TOOL",
        "provider_revision": "NOT_EXPOSED_BY_MANAGED_TOOL",
        "output_path": str(output_path(request)),
        "path_type": DATA_PATH_TYPE,
        "started_at": args.started_at,
        "ended_at": args.ended_at,
        "success": False,
        "failure_reason": args.reason,
        "technical_retry_count": record["technical_retry_count"],
        "human_review_status": "PENDING_HUMAN_DOUBLE_REVIEW",
        "pre_formal_base_label": PRE_FORMAL_LABEL,
        "downstream_use_status": DOWNSTREAM_BLOCK,
    }
    write_json(provenance_path(args.request_id), provenance)
    refresh_progress(progress)
    print(json.dumps({"request_id": args.request_id, "status": record["status"], "failure_reason": args.reason}, indent=2))


def make_contact_sheet(records: list[dict[str, Any]], path: Path, columns: int, tile: tuple[int, int]) -> None:
    if not records:
        return
    tile_width, image_height = tile
    label_height = 54
    rows = (len(records) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_width, rows * (image_height + label_height)), "#202225")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, record in enumerate(records):
        x = (index % columns) * tile_width
        y = (index // columns) * (image_height + label_height)
        with Image.open(record["output_path"]) as image:
            image = image.convert("RGB")
            image.thumbnail((tile_width - 8, image_height - 8), Image.Resampling.LANCZOS)
            px = x + (tile_width - image.width) // 2
            py = y + (image_height - image.height) // 2
            canvas.paste(image, (px, py))
        label = f"{record['request_id']}\n{record['garment_id']} {record['slot_id']} cam{SLOT_CACHE[record['slot_id']][0]:02d} cand{record['candidate_index']}"
        draw.multiline_text((x + 6, y + image_height + 4), label, fill="white", font=font, spacing=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=True)


def classification_for(progress: dict[str, Any]) -> tuple[str, str]:
    counts = progress["counts"]
    generated = counts["generated_technical_pass"] + counts["generated_technical_review"]
    if counts["input_blocked"]:
        return "SUBJECT00_CODEX_MANAGED_GENERATION_INPUT_BLOCKED", "RESUME_SUBJECT00_CODEX_MANAGED_GENERATION_FROM_PROGRESS_REGISTRY"
    if generated < 48:
        return "SUBJECT00_CODEX_MANAGED_GENERATION_PARTIAL_RESUMABLE", "RESUME_SUBJECT00_CODEX_MANAGED_GENERATION_FROM_PROGRESS_REGISTRY"
    if counts["generated_technical_review"]:
        return "SUBJECT00_CODEX_MANAGED_GENERATION_COMPLETE_WITH_TECHNICAL_REVIEW", "USER_PERFORM_SUBJECT00_DOUBLE_HUMAN_REVIEW"
    return "SUBJECT00_CODEX_MANAGED_GENERATION_COMPLETE_PENDING_HUMAN_REVIEW", "USER_PERFORM_SUBJECT00_DOUBLE_HUMAN_REVIEW"


def finalize() -> None:
    generation_result_head = os.environ.get("SUBJECT00_GENERATION_RESULT_HEAD", GENERATION_RESULT_HEAD)
    manifest = source_manifest()
    progress = load_json(progress_path())
    by_id = {row["request_id"]: row for row in manifest["requests"]}
    generated = [
        record for record in progress["records"]
        if record["status"] in {"GENERATED_TECHNICAL_PASS", "GENERATED_TECHNICAL_REVIEW"}
    ]
    for record in generated:
        path = Path(record["output_path"])
        if not path.is_file() or file_sha256(path) != record["output_sha256"]:
            raise RuntimeError(f"output/progress mismatch: {record['request_id']}")

    contact_root = DATA_ROOT / "05_human_review" / "contact_sheets"
    for garment in ["O01", "O03", "O04"]:
        rows = [record for record in generated if record["garment_id"] == garment]
        make_contact_sheet(rows, contact_root / f"subject00_{garment}_candidates.png", 4, (256, 384))
    make_contact_sheet(generated, contact_root / "subject00_all_48_candidates.png", 8, (192, 288))

    review_records = []
    for progress_record in progress["records"]:
        row = by_id[progress_record["request_id"]]
        review = {
            "request_id": row["request_id"],
            "candidate_path": progress_record.get("output_path"),
            "candidate_sha256": progress_record.get("output_sha256"),
            "garment_id": row["garment_id"],
            "slot_id": row["slot_id"],
            "camera_id": row["camera_id"],
            "candidate_index": row["candidate_index"],
            "generation_status": progress_record["status"],
            "final_status": "PENDING_HUMAN_DOUBLE_REVIEW",
        }
        review.update({field: None for field in REVIEW_FIELDS})
        review_records.append(review)
    review_manifest = {
        "schema_version": "canondressgs.subject00.codex_managed_human_review_external.v1",
        "task_id": TASK_ID,
        "candidate_count": len(review_records),
        "records": review_records,
        "decision_policy": {
            "both_accept": "ACCEPT",
            "any_reject": "REJECT",
            "other_pair": "MANUAL_ADJUDICATION",
            "vlm_tie_break_allowed": False,
        },
        "pending_count": len(review_records),
        "accepted_count": 0,
        "rejected_count": 0,
    }
    review_path = DATA_ROOT / "05_human_review" / "subject00_codex_managed_human_review_manifest.json"
    write_json(review_path, review_manifest)

    hashes: dict[str, list[str]] = defaultdict(list)
    for record in generated:
        hashes[record["output_sha256"]].append(record["request_id"])
    duplicate_groups = [
        {"sha256": sha, "request_ids": ids}
        for sha, ids in sorted(hashes.items()) if len(ids) > 1
    ]
    technical = {
        "schema_version": "canondressgs.subject00.codex_managed_technical_external.v1",
        "task_id": TASK_ID,
        "manifest_requests": len(progress["records"]),
        "generated_count": len(generated),
        "file_count": sum(Path(record["output_path"]).is_file() for record in generated),
        "png_parse_pass": sum(record.get("format") == "PNG" for record in generated),
        "resolution_pass_1024x1536": sum((record.get("width"), record.get("height")) == TARGET_SIZE for record in generated),
        "resolution_mismatch_count": sum((record.get("width"), record.get("height")) != TARGET_SIZE for record in generated),
        "duplicate_sha_groups": duplicate_groups,
        "provenance_count": sum(provenance_path(record["request_id"]).is_file() for record in progress["records"]),
        "missing_provenance_count": sum(not provenance_path(record["request_id"]).is_file() for record in progress["records"]),
        "accepted_rgb_count": len(list((DATA_ROOT / "06_accepted_rgb").rglob("*.png"))),
        "teacher_target_count": len(list((DATA_ROOT / "09_teacher_targets").rglob("*.png"))),
        "external_api_calls": 0,
        "custom_base_url_calls": 0,
        "sublyx_calls": 0,
        "code78_calls": 0,
        "api_key_reads": 0,
        "cloud_image_writes": 0,
        "formal_base_dependency": FORMAL_DEPENDENCY,
    }
    write_json(DATA_ROOT / "12_final_verification" / "technical_verification.json", technical)

    classification, next_task = classification_for(progress)
    contract = {
        "schema_version": "canondressgs.subject00.codex_managed_generation_contract.v1",
        "task_id": TASK_ID,
        "source": {"branch": SOURCE_BRANCH, "head": SOURCE_HEAD},
        "data_prep_source": {"branch": DATA_PREP_BRANCH, "head": DATA_PREP_HEAD},
        "branch": BRANCH,
        "backend": "CODEX_MANAGED_IMAGE_EDIT",
        "provider_response_available": False,
        "provider_model_id": "NOT_EXPOSED_BY_MANAGED_TOOL",
        "provider_revision": "NOT_EXPOSED_BY_MANAGED_TOOL",
        "request_count": 48,
        "technical_retry_limit": 1,
        "target_resolution": {"width": 1024, "height": 1536, "format": "PNG"},
        "dataset_root": str(DATA_ROOT),
        "path_type": DATA_PATH_TYPE,
        "formal_base_dependency": FORMAL_DEPENDENCY,
        "pre_formal_base_label": PRE_FORMAL_LABEL,
        "downstream_use_status": DOWNSTREAM_BLOCK,
        "external_api_calls": 0,
        "custom_base_url_calls": 0,
        "sublyx_calls": 0,
        "code78_calls": 0,
        "api_key_reads": 0,
        "cloud_image_writes": 0,
        "paper_final": False,
    }
    output_records = []
    for record in progress["records"]:
        row = by_id[record["request_id"]]
        output_records.append(
            {
                "request_id": record["request_id"],
                "garment_id": row["garment_id"],
                "slot_id": row["slot_id"],
                "camera_id": row["camera_id"],
                "pose_frame_id": row["pose_frame_id"],
                "candidate_index": row["candidate_index"],
                "source_input_sha256": row["input_sha256"],
                "status": record["status"],
                "output_path": record.get("output_path"),
                "output_sha256": record.get("output_sha256"),
                "output_bytes": record.get("output_bytes"),
                "width": record.get("width"),
                "height": record.get("height"),
                "format": record.get("format"),
                "technical_status": record.get("technical_status"),
                "provenance_path": str(provenance_path(record["request_id"])),
                "path_type": DATA_PATH_TYPE,
                "review_status": "PENDING_HUMAN_DOUBLE_REVIEW",
                "pre_formal_base_label": PRE_FORMAL_LABEL,
                "downstream_use_status": DOWNSTREAM_BLOCK,
            }
        )
    generation_registry = {
        "schema_version": "canondressgs.subject00.codex_managed_generation_registry.v1",
        "task_id": TASK_ID,
        "request_count": 48,
        "request_set_sha256": manifest["request_set_sha256"],
        "counts": progress["counts"],
        "generation_backend": "CODEX_MANAGED_IMAGE_EDIT",
        "managed_generation_calls": (
            len(generated)
            + progress["counts"]["failed"]
            + sum(record["technical_retry_count"] for record in progress["records"])
        ),
        "technical_retries": sum(record["technical_retry_count"] for record in progress["records"]),
        "external_api_calls": 0,
        "custom_base_url_calls": 0,
        "sublyx_calls": 0,
        "code78_calls": 0,
        "api_key_reads": 0,
        "cloud_image_writes": 0,
        "paper_final": False,
    }
    output_registry = {
        "schema_version": "canondressgs.subject00.codex_managed_output_registry.v1",
        "task_id": TASK_ID,
        "path_type": DATA_PATH_TYPE,
        "records": output_records,
        "output_count": len(generated),
        "duplicate_sha_groups": duplicate_groups,
        "paper_final": False,
    }
    technical_git = dict(technical)
    technical_git["schema_version"] = "canondressgs.subject00.codex_managed_technical_verification.v1"
    technical_git["paper_final"] = False
    review_git = {
        "schema_version": "canondressgs.subject00.codex_managed_human_review_manifest.v1",
        "task_id": TASK_ID,
        "path_type": DATA_PATH_TYPE,
        "external_manifest_path": str(review_path),
        "records": review_records,
        "pending_count": 48,
        "accepted_count": 0,
        "rejected_count": 0,
        "paper_final": False,
    }
    tests = {
        "schema_version": "canondressgs.subject00.codex_managed_generation_tests.v1",
        "task_id": TASK_ID,
        "status": "NOT_RUN",
        "total": 0,
        "passed": 0,
        "failed": [],
        "checks": [],
        "paper_final": False,
    }
    summary = {
        "schema_version": "canondressgs.subject00.codex_managed_generation_final_summary.v1",
        "task_id": TASK_ID,
        "source": {"branch": SOURCE_BRANCH, "head": SOURCE_HEAD},
        "branch": BRANCH,
        "classification": classification,
        "dataset_root": str(DATA_ROOT),
        "path_type": DATA_PATH_TYPE,
        "request_count": 48,
        "managed_generation_calls": generation_registry["managed_generation_calls"],
        "technical_retries": generation_registry["technical_retries"],
        "successful_outputs": len(generated),
        "failed_outputs": progress["counts"]["failed"],
        "missing_outputs": 48 - len(generated),
        "png_parse_pass": technical["png_parse_pass"],
        "resolution_pass_1024x1536": technical["resolution_pass_1024x1536"],
        "resolution_mismatches": technical["resolution_mismatch_count"],
        "duplicate_output_groups": len(duplicate_groups),
        "provenance_count": technical["provenance_count"],
        "review_pending_count": 48,
        "accepted_count": 0,
        "rejected_count": 0,
        "teacher_target_count": 0,
        "external_api_calls": 0,
        "custom_base_url_calls": 0,
        "sublyx_calls": 0,
        "code78_calls": 0,
        "api_key_reads": 0,
        "cloud_image_writes": 0,
        "git_image_commits": 0,
        "formal_base_dependency": FORMAL_DEPENDENCY,
        "downstream_use_status": DOWNSTREAM_BLOCK,
        "tests": "NOT_RUN",
        "generation_result_head": generation_result_head,
        "final_reporting_head_resolution": "git rev-parse HEAD after final reporting commit",
        "paper_final": False,
        "next_task": next_task,
        "next_task_authorized": False,
    }
    handoff = {
        "schema_version": "canondressgs.subject00.codex_managed_generation_handoff.v1",
        "task_id": TASK_ID,
        "source": {"branch": SOURCE_BRANCH, "head": SOURCE_HEAD},
        "branch": BRANCH,
        "classification": classification,
        "dataset_root": str(DATA_ROOT),
        "path_type": DATA_PATH_TYPE,
        "successful_outputs": len(generated),
        "review_pending_count": 48,
        "accepted_count": 0,
        "teacher_target_count": 0,
        "external_api_calls": 0,
        "custom_base_url_calls": 0,
        "sublyx_calls": 0,
        "code78_calls": 0,
        "api_key_reads": 0,
        "cloud_image_writes": 0,
        "formal_base_dependency": FORMAL_DEPENDENCY,
        "downstream_use_status": DOWNSTREAM_BLOCK,
        "generation_result_head": generation_result_head,
        "final_reporting_head_resolution": "git rev-parse HEAD after final reporting commit",
        "tests": "NOT_RUN",
        "paper_final": False,
        "next_task": next_task,
        "next_task_authorized": False,
    }
    git_outputs = {
        "subject00_codex_managed_generation_contract.json": contract,
        "subject00_codex_managed_generation_registry.json": generation_registry,
        "subject00_codex_managed_output_registry.json": output_registry,
        "subject00_codex_managed_technical_verification.json": technical_git,
        "subject00_codex_managed_human_review_manifest.json": review_git,
        "subject00_codex_managed_generation_tests.json": tests,
        "subject00_codex_managed_generation_final_summary.json": summary,
    }
    for name, payload in git_outputs.items():
        write_json(RISK / name, payload, sealed=True)
    write_json(HANDOFF / "subject00_codex_managed_generation_handoff.json", handoff, sealed=True)
    write_doc(
        DOCS / "AAAI27_SUBJECT00_CODEX_MANAGED_GENERATION_REPORT_20260725.md",
        f"""
# Subject00 Codex Managed Generation Report

- Task: `{TASK_ID}`
- Source: `{SOURCE_BRANCH}` at `{SOURCE_HEAD}`
- Classification: `{classification}`
- Backend: `CODEX_MANAGED_IMAGE_EDIT`
- External dataset root: `{DATA_ROOT}`
- Requests: 48
- Managed outputs: {len(generated)}
- Technical 1024x1536 passes: {technical['resolution_pass_1024x1536']}
- Technical retries: {generation_registry['technical_retries']}
- Pending double human review: 48
- Accepted images: 0
- Teacher targets: 0
- External API/Sublyx/78Code calls: 0
- Custom base URL calls/API key reads/cloud image writes: 0
- Formal Base dependency: `PENDING`
- Downstream use: `{DOWNSTREAM_BLOCK}`

Candidates are local Windows assets and are not committed to Git or copied to cloud storage. They are pre-Formal candidates, not accepted Teacher targets, final dataset images, or paper results.

- Contract tests: `RUN_AFTER_METADATA_EMISSION`
""",
    )
    write_doc(
        DOCS / "AAAI27_SUBJECT00_GENERATED_CANDIDATE_REVIEW_GUIDE_20260725.md",
        f"""
# Subject00 Generated Candidate Review Guide

Review the four contact sheets under `{contact_root}` and open original candidates before deciding.

Each candidate requires two independent reviewers. Review identity, face, hair, skin tone, body shape, pose, camera, garment semantics, slot consistency, edges, hands/feet, background, lighting, and artifacts. Two ACCEPT decisions accept; any REJECT rejects; every other combination requires manual adjudication. VLM or generator self-review is not a final tie-break.

All records begin at `PENDING_HUMAN_DOUBLE_REVIEW`. Do not copy candidates into `06_accepted_rgb` or `09_teacher_targets` until Formal Base is sealed, both reviewers accept, camera/pose and mask gates pass, provenance is complete, and the user explicitly authorizes downstream use.
""",
    )
    print(json.dumps({"classification": classification, "next_task": next_task, "generated": len(generated), "technical": technical, "review_manifest": str(review_path)}, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("next")
    success = sub.add_parser("record-success")
    success.add_argument("--request-id", required=True)
    success.add_argument("--generated-source", required=True)
    success.add_argument("--started-at", required=True)
    success.add_argument("--ended-at", required=True)
    success.add_argument("--technical-retry-count", type=int, default=0)
    success.add_argument("--tool-output-hint", default="")
    failure = sub.add_parser("record-failure")
    failure.add_argument("--request-id", required=True)
    failure.add_argument("--reason", required=True)
    failure.add_argument("--started-at", required=True)
    failure.add_argument("--ended-at", required=True)
    failure.add_argument("--technical-retry-count", type=int, default=0)
    sub.add_parser("finalize")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "next":
        next_request()
    elif args.command == "record-success":
        record_success(args)
    elif args.command == "record-failure":
        record_failure(args)
    elif args.command == "finalize":
        finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
