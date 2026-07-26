#!/usr/bin/env python3
"""Manage the four-request Subject00 native portrait canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[2]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
DOCS = ROOT / "docs" / "PAPER"
HANDOFF = ROOT / "project_control_handoff"
ATTEMPT_001 = Path(r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-GENERATION-001\attempt_001")
DATA_ROOT = Path(r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-GENERATION-001\attempt_002_portrait_canary")

TASK_ID = "AAAI27-SUBJECT00-MANAGED-PORTRAIT-CANARY-001"
SOURCE_BRANCH = "research/subject00-managed-output-resolution-adjudication-20260725"
SOURCE_HEAD = "90f5563988513088fd8da690e0527bc2b377466c"
BRANCH = "research/subject00-managed-portrait-canary-20260725"
TARGET_SIZE = (1024, 1536)
PATH_TYPE = "WINDOWS_LOCAL_EXTERNAL_DATASET_PATH"
FORMAL_BASE = "PENDING"
PAPER_FINAL = False
CANARY_RESULT_HEAD = "RESOLVE_AFTER_CANARY_RESULT_COMMIT"
PASS_CLASSIFICATION = "SUBJECT00_PORTRAIT_CANARY_TECHNICAL_PASS_PENDING_USER_REVIEW"
FAIL_CLASSIFICATION = "SUBJECT00_PORTRAIT_CANARY_TECHNICAL_FAIL"
PARTIAL_CLASSIFICATION = "SUBJECT00_PORTRAIT_CANARY_PARTIAL"
PASS_NEXT_TASK = "USER_REVIEW_PORTRAIT_CANARY_THEN_AUTHORIZE_REMAINING_39_RERUN"
FAIL_NEXT_TASK = "ADJUDICATE_ALTERNATIVE_NATIVE_PORTRAIT_GENERATION_STRATEGY"

DIRECTORIES = [
    "00_contract",
    "01_derived_edit_canvases",
    "03_generation_requests",
    "04_generation_responses",
    "05_human_review",
    "11_provenance",
    "12_final_verification",
]

SLOT_CAMERAS = {
    "slot_00": 17,
    "slot_01": 21,
    "slot_02": 14,
    "slot_03": 23,
    "slot_04": 11,
    "slot_05": 2,
    "slot_06": 9,
    "slot_07": 5,
}

CANVAS_CONTROL_CLAUSE = """Edit the provided portrait canvas directly.
Return a native portrait PNG at exactly 1024 pixels wide
and 1536 pixels high.
Preserve the complete full body from head to feet.
Do not rotate the canvas, change it to landscape,
crop the person, add borders, add text,
or change the camera orientation."""

INPUT_ORDER_CLAUSE = (
    "Input order clarification: Image 1 is the DERIVED_EDIT_INPUT_CANVAS and primary edit target. "
    "Image 2 is the unchanged original Subject00 identity/condition reference for identity fidelity only. "
    "Do not copy Image 2's landscape canvas dimensions or override Image 1's portrait composition."
)

VISUAL_FIELDS = [
    "identity_match",
    "pose_match",
    "camera_match",
    "garment_match",
    "full_body_complete",
    "hands_complete",
    "feet_complete",
    "background_match",
    "artifact_grade",
    "decision",
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


def progress_path() -> Path:
    return DATA_ROOT / "03_generation_requests" / "generation_progress_registry.json"


def request_path(request_id: str) -> Path:
    return DATA_ROOT / "03_generation_requests" / "requests" / f"{request_id}.json"


def provenance_path(request_id: str) -> Path:
    return DATA_ROOT / "11_provenance" / "requests" / f"{request_id}.json"


def output_path(item: dict[str, Any]) -> Path:
    return DATA_ROOT / "04_generation_responses" / item["garment_id"] / f"{item['request_id']}.png"


def canvas_path(request_id: str) -> Path:
    return DATA_ROOT / "01_derived_edit_canvases" / f"{request_id}_portrait_canvas.png"


def select_canaries() -> list[str]:
    plan = load_json(RISK / "subject00_managed_output_regeneration_plan.json")
    inventory = load_json(RISK / "subject00_managed_output_resolution_inventory.json")
    by_id = {item["request_id"]: item for item in inventory["records"]}
    rows = plan["records"]
    if len(rows) != 43:
        raise RuntimeError("strict rerun set is not 43")

    def first(garment: str, slot: str) -> str:
        values = sorted(
            item["request_id"]
            for item in rows
            if by_id[item["request_id"]]["garment_id"] == garment
            and by_id[item["request_id"]]["slot_id"] == slot
        )
        if not values:
            raise RuntimeError(f"missing canary selector {garment}/{slot}")
        return values[0]

    selected = [first("O01", "slot_00"), first("O03", "slot_03"), first("O04", "slot_02")]
    portrait_values = sorted(
        item["request_id"]
        for item in rows
        if by_id[item["request_id"]]["slot_id"] in {"slot_06", "slot_07"}
        and by_id[item["request_id"]]["resolution_category"] == "D_PORTRAIT_INCOMPATIBLE_ASPECT"
    )
    fallback_values = sorted(
        item["request_id"]
        for item in rows
        if by_id[item["request_id"]]["slot_id"] in {"slot_06", "slot_07"}
    )
    selected.append(portrait_values[0] if portrait_values else fallback_values[0])
    if len(set(selected)) != 4:
        raise RuntimeError("canary selection is not unique")
    return selected


def border_background(image: Image.Image) -> tuple[tuple[int, int, int], dict[str, Any]]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    edge = min(8, width // 4, height // 4)
    pixels = (
        list(rgb.crop((0, 0, width, edge)).getdata())
        + list(rgb.crop((0, height - edge, width, height)).getdata())
        + list(rgb.crop((0, edge, edge, height - edge)).getdata())
        + list(rgb.crop((width - edge, edge, width, height - edge)).getdata())
    )
    median = tuple(int(round(statistics.median(pixel[channel] for pixel in pixels))) for channel in range(3))
    mad = tuple(
        float(statistics.median(abs(pixel[channel] - median[channel]) for pixel in pixels))
        for channel in range(3)
    )
    reliable = max(mad) <= 64.0 and max(median) >= 16
    background = median if reliable else (245, 245, 245)
    return background, {
        "method": "BORDER_PIXEL_MEDIAN" if reliable else "FALLBACK_RGB_245",
        "border_width_pixels": edge,
        "border_median_rgb": list(median),
        "border_median_absolute_deviation": list(mad),
        "reliable": reliable,
    }


def make_canvas(source_path: Path, target_path: Path) -> dict[str, Any]:
    with Image.open(source_path) as source:
        source.load()
        source_rgb = source.convert("RGB")
        source_width, source_height = source_rgb.size
        background, background_audit = border_background(source_rgb)
        contained = ImageOps.contain(source_rgb, (960, 1472), Image.Resampling.LANCZOS)
    offset_x = (TARGET_SIZE[0] - contained.width) // 2
    offset_y = (TARGET_SIZE[1] - contained.height) // 2
    if offset_x < 32 or offset_y < 32:
        raise RuntimeError("derived canvas margin contract failed")
    canvas = Image.new("RGB", TARGET_SIZE, background)
    canvas.paste(contained, (offset_x, offset_y))
    target_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target_path, format="PNG", optimize=True)
    with Image.open(target_path) as check:
        check.load()
        if check.format != "PNG" or check.mode != "RGB" or check.size != TARGET_SIZE:
            raise RuntimeError("derived canvas verification failed")
    return {
        "source_path": str(source_path),
        "source_sha256": file_sha256(source_path),
        "source_resolution": {"width": source_width, "height": source_height},
        "contain_box": {"width": 960, "height": 1472},
        "resized_resolution": {"width": contained.width, "height": contained.height},
        "resize_scale_x": contained.width / source_width,
        "resize_scale_y": contained.height / source_height,
        "paste_offset": {"x": offset_x, "y": offset_y},
        "margins": {
            "left": offset_x,
            "right": TARGET_SIZE[0] - offset_x - contained.width,
            "top": offset_y,
            "bottom": TARGET_SIZE[1] - offset_y - contained.height,
        },
        "background_rgb": list(background),
        "background_audit": background_audit,
        "derived_canvas_path": str(target_path),
        "derived_canvas_sha256": file_sha256(target_path),
        "derived_canvas_resolution": {"width": TARGET_SIZE[0], "height": TARGET_SIZE[1]},
        "derived_canvas_mode": "RGB",
        "derived_canvas_format": "PNG",
        "full_image_contain": True,
        "crop_operations": 0,
        "rotate_operations": 0,
        "stretch_operations": 0,
        "asset_role": "DERIVED_EDIT_INPUT_CANVAS",
    }


def prepare() -> None:
    if git("branch", "--show-current") != BRANCH:
        raise RuntimeError(f"expected branch {BRANCH}")
    git("merge-base", "--is-ancestor", SOURCE_HEAD, "HEAD")
    resolution_summary = load_json(RISK / "subject00_managed_output_resolution_final_summary.json")
    if resolution_summary["classification"] != "SUBJECT00_MANAGED_OUTPUT_STRICT_RERUN_REQUIRED":
        raise RuntimeError("resolution source classification mismatch")
    if resolution_summary["technical_pass_count"] != 5 or resolution_summary["strict_rerun_count"] != 43:
        raise RuntimeError("resolution source counts mismatch")

    inventory = load_json(RISK / "subject00_managed_output_resolution_inventory.json")
    output_registry = load_json(RISK / "subject00_codex_managed_output_registry.json")
    by_id = {item["request_id"]: item for item in inventory["records"]}
    output_by_id = {item["request_id"]: item for item in output_registry["records"]}
    selected = select_canaries()
    if any(by_id[request_id]["technical_dataset_eligibility"] != "FAIL" for request_id in selected):
        raise RuntimeError("selected canary outside strict rerun set")
    native_pass_ids = sorted(
        item["request_id"] for item in inventory["records"] if item["technical_dataset_eligibility"] == "PASS"
    )
    if len(native_pass_ids) != 5 or set(selected) & set(native_pass_ids):
        raise RuntimeError("native pass preservation contract failed")

    for directory in DIRECTORIES:
        (DATA_ROOT / directory).mkdir(parents=True, exist_ok=True)
    for path in [
        DATA_ROOT / "03_generation_requests" / "requests",
        DATA_ROOT / "11_provenance" / "requests",
        *[DATA_ROOT / "04_generation_responses" / garment for garment in ["O01", "O03", "O04"]],
    ]:
        path.mkdir(parents=True, exist_ok=True)
    forbidden = ["06_accepted_rgb", "09_teacher_targets", "10_controller_splits", "accepted_rgb", "teacher_targets"]
    if any((DATA_ROOT / name).exists() for name in forbidden):
        raise RuntimeError("forbidden attempt_002 directory exists")

    baseline_records = []
    for item in inventory["records"]:
        path = Path(item["output_path"])
        current_sha = file_sha256(path)
        if current_sha != item["output_sha256"] or current_sha != output_by_id[item["request_id"]]["output_sha256"]:
            raise RuntimeError(f"attempt_001 mutation before canary: {item['request_id']}")
        baseline_records.append(
            {
                "request_id": item["request_id"],
                "path": str(path),
                "sha256": current_sha,
                "preserved_native_pass_output": item["request_id"] in native_pass_ids,
            }
        )

    old_progress = load_json(progress_path()) if progress_path().is_file() else None
    old_by_id = {item["request_id"]: item for item in old_progress["records"]} if old_progress else {}
    selection_records = []
    canvas_records = []
    progress_records = []
    for selector_index, request_id in enumerate(selected, start=1):
        inventory_item = by_id[request_id]
        source_request_path = ATTEMPT_001 / "03_generation_requests" / "requests" / f"{request_id}.json"
        source_request = load_json(source_request_path)
        source_path = Path(source_request["managed_tool_local_input_path"])
        source_sha = file_sha256(source_path) if source_path.is_file() else None
        expected_sha = source_request["source_input_sha256"]["identity_condition_rgb"]
        errors = []
        if source_sha != expected_sha:
            errors.append("SOURCE_SHA_MISMATCH")
        if source_request["identity_id"] != "subject00":
            errors.append("IDENTITY_NOT_SUBJECT00")
        if source_request["pose_frame_id"] != 0:
            errors.append("POSE_NOT_ZERO")
        if "strict_train" not in source_request["identity_source_set_id"]:
            errors.append("NOT_STRICT_TRAIN")
        if source_request["camera_id"] != SLOT_CAMERAS[source_request["slot_id"]]:
            errors.append("CAMERA_SLOT_MISMATCH")
        if errors:
            raise RuntimeError(f"canary input blocked {request_id}: {errors}")

        target_canvas = canvas_path(request_id)
        prior_contract_path = DATA_ROOT / "00_contract" / "subject00_portrait_canvas_registry_external.json"
        if target_canvas.exists() and prior_contract_path.is_file():
            prior = load_json(prior_contract_path)
            prior_item = next(item for item in prior["records"] if item["request_id"] == request_id)
            if file_sha256(target_canvas) != prior_item["derived_canvas_sha256"]:
                raise RuntimeError(f"existing derived canvas mismatch: {request_id}")
            canvas_record = dict(prior_item)
        elif target_canvas.exists():
            raise RuntimeError(f"unregistered derived canvas exists: {request_id}")
        else:
            canvas_record = make_canvas(source_path, target_canvas)
            canvas_record.update(
                {
                    "request_id": request_id,
                    "garment_id": inventory_item["garment_id"],
                    "slot_id": inventory_item["slot_id"],
                    "camera_id": inventory_item["camera_id"],
                }
            )
        canvas_records.append(canvas_record)
        actual_prompt = "\n\n".join(
            [source_request["actual_prompt"], INPUT_ORDER_CLAUSE, CANVAS_CONTROL_CLAUSE]
        )
        request_payload = {
            "schema_version": "canondressgs.subject00.portrait_canary_request.v1",
            "task_id": TASK_ID,
            "selector_index": selector_index,
            "request_id": request_id,
            "garment_id": inventory_item["garment_id"],
            "slot_id": inventory_item["slot_id"],
            "camera_id": inventory_item["camera_id"],
            "pose_frame_id": source_request["pose_frame_id"],
            "identity_id": source_request["identity_id"],
            "identity_source_set_id": source_request["identity_source_set_id"],
            "source_request_path": str(source_request_path),
            "source_request_manifest_sha256": source_request["source_request_manifest_sha256"],
            "source_input_paths": source_request["source_input_paths"],
            "source_input_sha256": source_request["source_input_sha256"],
            "managed_tool_original_reference_path": str(source_path),
            "derived_edit_canvas_path": str(target_canvas),
            "derived_edit_canvas_sha256": canvas_record["derived_canvas_sha256"],
            "attempt_001_output_path": inventory_item["output_path"],
            "attempt_001_output_sha256": inventory_item["output_sha256"],
            "attempt_001_resolution": {
                "width": inventory_item["width"],
                "height": inventory_item["height"],
            },
            "attempt_001_resolution_category": inventory_item["resolution_category"],
            "prompt_id": source_request["prompt_id"],
            "negative_constraint_id": source_request["negative_constraint_id"],
            "source_actual_prompt": source_request["actual_prompt"],
            "actual_prompt": actual_prompt,
            "actual_prompt_sha256": hashlib.sha256(actual_prompt.encode("utf-8")).hexdigest(),
            "input_order": [str(target_canvas), str(source_path)],
            "generation_backend": "CODEX_MANAGED_IMAGE_EDIT",
            "generation_mode": "IMAGE_EDIT",
            "target_resolution": {"width": 1024, "height": 1536, "format": "PNG", "orientation": "portrait"},
            "output_path": str(output_path(inventory_item)),
            "output_postprocessing_allowed": False,
            "input_validation": {"status": "PASS", "errors": []},
            "formal_base": FORMAL_BASE,
            "pre_formal_candidate": True,
        }
        write_json(request_path(request_id), request_payload)
        selection_records.append(
            {
                "selector_index": selector_index,
                "request_id": request_id,
                "garment_id": inventory_item["garment_id"],
                "slot_id": inventory_item["slot_id"],
                "camera_id": inventory_item["camera_id"],
                "attempt_001_resolution_category": inventory_item["resolution_category"],
                "strict_rerun_member": True,
                "native_pass_overlap": False,
            }
        )
        progress_record = old_by_id.get(request_id)
        if progress_record and progress_record["status"] in {"NATIVE_RESOLUTION_PASS", "NATIVE_RESOLUTION_FAIL"}:
            path = Path(progress_record["output_path"])
            if not path.is_file() or file_sha256(path) != progress_record["output_sha256"]:
                progress_record = None
        if not progress_record:
            progress_record = {
                "request_id": request_id,
                "garment_id": inventory_item["garment_id"],
                "slot_id": inventory_item["slot_id"],
                "status": "PENDING",
                "call_attempts": 0,
                "technical_retry_count": 0,
                "failure_reason": None,
            }
        progress_records.append(progress_record)

    selection = {
        "schema_version": "canondressgs.subject00.portrait_canary_selection_external.v1",
        "task_id": TASK_ID,
        "strict_rerun_set_count": 43,
        "native_pass_count": 5,
        "selected_count": 4,
        "records": selection_records,
    }
    canvas_registry = {
        "schema_version": "canondressgs.subject00.portrait_canvas_registry_external.v1",
        "task_id": TASK_ID,
        "canvas_count": 4,
        "records": canvas_records,
        "canvas_role": "DERIVED_EDIT_INPUT_CANVAS",
        "generated_candidate": False,
        "output_postprocessing": False,
    }
    contract = {
        "schema_version": "canondressgs.subject00.portrait_canary_external_contract.v1",
        "task_id": TASK_ID,
        "source": {"branch": SOURCE_BRANCH, "head": SOURCE_HEAD},
        "branch": BRANCH,
        "attempt_001": str(ATTEMPT_001),
        "attempt_002": str(DATA_ROOT),
        "resolution_contract": "STRICT_NATIVE_RESOLUTION",
        "target_resolution": {"width": 1024, "height": 1536, "format": "PNG"},
        "selected_request_ids": selected,
        "attempt_001_baseline": baseline_records,
        "preserved_native_pass_request_ids": native_pass_ids,
        "forbidden_output_postprocessing": ["resize", "crop", "pad", "rotate", "stretch", "canvas_embedding"],
        "formal_base": FORMAL_BASE,
    }
    progress_payload = {
        "schema_version": "canondressgs.subject00.portrait_canary_progress.v1",
        "task_id": TASK_ID,
        "stable_order": selected,
        "records": progress_records,
        "external_api_calls": 0,
        "sublyx_calls": 0,
        "code78_calls": 0,
        "api_key_reads": 0,
        "cloud_image_writes": 0,
    }
    write_json(DATA_ROOT / "00_contract" / "subject00_portrait_canary_contract.json", contract)
    write_json(DATA_ROOT / "00_contract" / "subject00_portrait_canary_request_selection_external.json", selection)
    write_json(DATA_ROOT / "00_contract" / "subject00_portrait_canvas_registry_external.json", canvas_registry)
    write_json(progress_path(), progress_payload)
    print(
        json.dumps(
            {
                "status": "PREPARED",
                "dataset_root": str(DATA_ROOT),
                "selected_request_ids": selected,
                "canvas_count": len(canvas_records),
                "pending": sum(item["status"] in {"PENDING", "TECHNICAL_RETRY_PENDING"} for item in progress_records),
            },
            indent=2,
        )
    )


def next_request() -> None:
    progress = load_json(progress_path())
    for record in progress["records"]:
        if record["status"] in {"PENDING", "TECHNICAL_RETRY_PENDING"}:
            payload = load_json(request_path(record["request_id"]))
            payload["technical_retry_count"] = record["technical_retry_count"]
            payload["call_attempts"] = record["call_attempts"]
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return
    print(json.dumps({"status": "NO_PENDING_REQUEST"}))


def refresh_progress(progress: dict[str, Any]) -> None:
    counts = Counter(item["status"] for item in progress["records"])
    progress["counts"] = dict(sorted(counts.items()))
    progress["generation_calls"] = sum(item["call_attempts"] for item in progress["records"])
    progress["primary_calls"] = sum(item["call_attempts"] >= 1 for item in progress["records"])
    progress["technical_retries"] = sum(max(item["call_attempts"] - 1, 0) for item in progress["records"])
    write_json(progress_path(), progress)


def record_success(args: argparse.Namespace) -> None:
    source = Path(args.generated_source)
    if not source.is_file():
        raise FileNotFoundError(source)
    with Image.open(source) as image:
        image.load()
        width, height = image.size
        image_format = image.format
    if image_format != "PNG":
        raise RuntimeError(f"generated output is {image_format}, not PNG")
    progress = load_json(progress_path())
    record = next(item for item in progress["records"] if item["request_id"] == args.request_id)
    if record["status"] in {"NATIVE_RESOLUTION_PASS", "NATIVE_RESOLUTION_FAIL"}:
        raise RuntimeError("completed canary must not be regenerated")
    request = load_json(request_path(args.request_id))
    target = Path(request["output_path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and file_sha256(target) != file_sha256(source):
        raise RuntimeError("refusing to overwrite a different canary output")
    if not target.exists():
        shutil.copy2(source, target)
    output_sha = file_sha256(target)
    native_pass = (width, height) == TARGET_SIZE
    record.update(
        {
            "status": "NATIVE_RESOLUTION_PASS" if native_pass else "NATIVE_RESOLUTION_FAIL",
            "call_attempts": record["call_attempts"] + 1,
            "technical_retry_count": int(args.technical_retry_count),
            "output_path": str(target),
            "output_sha256": output_sha,
            "output_bytes": target.stat().st_size,
            "width": width,
            "height": height,
            "format": image_format,
            "failure_reason": None if native_pass else "NATIVE_RESOLUTION_FAIL",
            "started_at": args.started_at,
            "ended_at": args.ended_at,
        }
    )
    provenance = {
        "schema_version": "canondressgs.subject00.portrait_canary_provenance.v1",
        "task_id": TASK_ID,
        "request_id": args.request_id,
        "garment_id": request["garment_id"],
        "slot_id": request["slot_id"],
        "camera_id": request["camera_id"],
        "pose_frame_id": request["pose_frame_id"],
        "input_order": request["input_order"],
        "derived_edit_canvas_sha256": request["derived_edit_canvas_sha256"],
        "source_input_paths": request["source_input_paths"],
        "source_input_sha256": request["source_input_sha256"],
        "prompt_id": request["prompt_id"],
        "negative_constraint_id": request["negative_constraint_id"],
        "actual_prompt": request["actual_prompt"],
        "actual_prompt_sha256": request["actual_prompt_sha256"],
        "generation_backend": "CODEX_MANAGED_IMAGE_EDIT",
        "generation_mode": "IMAGE_EDIT",
        "provider_response_available": False,
        "provider_model_id": "NOT_EXPOSED_BY_MANAGED_TOOL",
        "provider_revision": "NOT_EXPOSED_BY_MANAGED_TOOL",
        "tool_output_hint": args.tool_output_hint,
        "output_path": str(target),
        "output_sha256": output_sha,
        "output_bytes": target.stat().st_size,
        "width": width,
        "height": height,
        "format": image_format,
        "native_resolution_status": "PASS" if native_pass else "FAIL",
        "output_postprocessing_count": 0,
        "technical_retry_count": int(args.technical_retry_count),
        "started_at": args.started_at,
        "ended_at": args.ended_at,
        "success": True,
        "visual_review_status": "PENDING_HUMAN_REVIEW",
        "formal_base": FORMAL_BASE,
    }
    write_json(provenance_path(args.request_id), provenance)
    refresh_progress(progress)
    print(
        json.dumps(
            {
                "request_id": args.request_id,
                "status": record["status"],
                "output_path": str(target),
                "sha256": output_sha,
                "width": width,
                "height": height,
            },
            indent=2,
        )
    )


def record_failure(args: argparse.Namespace) -> None:
    progress = load_json(progress_path())
    record = next(item for item in progress["records"] if item["request_id"] == args.request_id)
    retry_count = int(args.technical_retry_count)
    record.update(
        {
            "status": "TECHNICAL_RETRY_PENDING" if retry_count < 1 else "FAILED",
            "call_attempts": record["call_attempts"] + 1,
            "technical_retry_count": retry_count + 1 if retry_count < 1 else retry_count,
            "failure_reason": args.reason,
            "started_at": args.started_at,
            "ended_at": args.ended_at,
        }
    )
    request = load_json(request_path(args.request_id))
    write_json(
        provenance_path(args.request_id),
        {
            "schema_version": "canondressgs.subject00.portrait_canary_provenance.v1",
            "task_id": TASK_ID,
            "request_id": args.request_id,
            "input_order": request["input_order"],
            "derived_edit_canvas_sha256": request["derived_edit_canvas_sha256"],
            "actual_prompt": request["actual_prompt"],
            "generation_backend": "CODEX_MANAGED_IMAGE_EDIT",
            "generation_mode": "IMAGE_EDIT",
            "provider_response_available": False,
            "provider_model_id": "NOT_EXPOSED_BY_MANAGED_TOOL",
            "provider_revision": "NOT_EXPOSED_BY_MANAGED_TOOL",
            "success": False,
            "failure_reason": args.reason,
            "technical_retry_count": record["technical_retry_count"],
            "output_postprocessing_count": 0,
            "started_at": args.started_at,
            "ended_at": args.ended_at,
            "formal_base": FORMAL_BASE,
        },
    )
    refresh_progress(progress)
    print(json.dumps({"request_id": args.request_id, "status": record["status"], "reason": args.reason}, indent=2))


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def make_contact_sheet(records: list[dict[str, Any]], inventory_by_id: dict[str, Any]) -> Path:
    panel_width = 300
    header_height = 58
    image_height = 300
    label_height = 44
    row_height = header_height + image_height + label_height
    canvas = Image.new("RGB", (panel_width * 4, row_height * len(records)), "#202225")
    draw = ImageDraw.Draw(canvas)
    label_font = font(12)
    roles = ["ORIGINAL CONDITION", "DERIVED PORTRAIT CANVAS", "ATTEMPT_001 OUTPUT", "ATTEMPT_002 CANARY"]
    for row_index, record in enumerate(records):
        request = load_json(request_path(record["request_id"]))
        inventory = inventory_by_id[record["request_id"]]
        y0 = row_index * row_height
        new_resolution = f"{record.get('width', 'MISSING')}x{record.get('height', 'MISSING')}"
        header = (
            f"{record['request_id']} | {record['garment_id']} {record['slot_id']} cam{request['camera_id']:02d} | "
            f"old {inventory['width']}x{inventory['height']} | new {new_resolution}"
        )
        draw.text((8, y0 + 8), header, fill="white", font=label_font)
        paths = [
            Path(request["managed_tool_original_reference_path"]),
            Path(request["derived_edit_canvas_path"]),
            Path(request["attempt_001_output_path"]),
            Path(record["output_path"]) if record.get("output_path") else None,
        ]
        for column, (role, path) in enumerate(zip(roles, paths)):
            x0 = column * panel_width
            if path and path.is_file():
                with Image.open(path) as source:
                    preview = source.convert("RGB")
                    source_size = source.size
                    preview.thumbnail((panel_width - 10, image_height - 10), Image.Resampling.LANCZOS)
                px = x0 + (panel_width - preview.width) // 2
                py = y0 + header_height + (image_height - preview.height) // 2
                canvas.paste(preview, (px, py))
                size_label = f"{source_size[0]}x{source_size[1]}"
            else:
                size_label = "MISSING"
            draw.text((x0 + 6, y0 + header_height + image_height + 4), f"{role} | {size_label}", fill="white", font=label_font)
    path = DATA_ROOT / "05_human_review" / "subject00_portrait_canary_contact_sheet.png"
    canvas.save(path, format="PNG", optimize=True)
    return path


def finalize() -> None:
    progress = load_json(progress_path())
    inventory = load_json(RISK / "subject00_managed_output_resolution_inventory.json")
    inventory_by_id = {item["request_id"]: item for item in inventory["records"]}
    contract = load_json(DATA_ROOT / "00_contract" / "subject00_portrait_canary_contract.json")
    selection_external = load_json(DATA_ROOT / "00_contract" / "subject00_portrait_canary_request_selection_external.json")
    canvas_external = load_json(DATA_ROOT / "00_contract" / "subject00_portrait_canvas_registry_external.json")

    mutation_ids = [
        item["request_id"]
        for item in contract["attempt_001_baseline"]
        if file_sha256(Path(item["path"])) != item["sha256"]
    ]
    preserved_native_errors = [
        item["request_id"]
        for item in contract["attempt_001_baseline"]
        if item["preserved_native_pass_output"] and file_sha256(Path(item["path"])) != item["sha256"]
    ]
    if mutation_ids:
        raise RuntimeError(f"attempt_001 mutation detected: {mutation_ids}")

    output_records = []
    for record in progress["records"]:
        request = load_json(request_path(record["request_id"]))
        item = {
            "request_id": record["request_id"],
            "garment_id": record["garment_id"],
            "slot_id": record["slot_id"],
            "camera_id": request["camera_id"],
            "status": record["status"],
            "output_path": record.get("output_path"),
            "output_sha256": record.get("output_sha256"),
            "output_bytes": record.get("output_bytes"),
            "width": record.get("width"),
            "height": record.get("height"),
            "format": record.get("format"),
            "native_resolution_pass": record["status"] == "NATIVE_RESOLUTION_PASS",
            "technical_retry_count": record["technical_retry_count"],
            "call_attempts": record["call_attempts"],
            "provenance_path": str(provenance_path(record["request_id"])),
            "visual_review_status": "PENDING_HUMAN_REVIEW",
            "output_postprocessing_count": 0,
        }
        if record.get("output_path"):
            path = Path(record["output_path"])
            with Image.open(path) as image:
                image.load()
                if image.format != record["format"] or image.size != (record["width"], record["height"]):
                    raise RuntimeError(f"output registry mismatch: {record['request_id']}")
            if file_sha256(path) != record["output_sha256"]:
                raise RuntimeError(f"output SHA mismatch: {record['request_id']}")
        output_records.append(item)

    output_count = sum(item["output_path"] is not None for item in output_records)
    parse_count = sum(item["format"] == "PNG" for item in output_records)
    native_pass_count = sum(item["native_resolution_pass"] for item in output_records)
    output_shas = [item["output_sha256"] for item in output_records if item["output_sha256"]]
    duplicate_count = len(output_shas) - len(set(output_shas))
    provenance_count = sum(provenance_path(item["request_id"]).is_file() for item in output_records)
    if output_count < 4:
        classification = PARTIAL_CLASSIFICATION
    elif native_pass_count == 4 and parse_count == 4 and duplicate_count == 0 and provenance_count == 4:
        classification = PASS_CLASSIFICATION
    else:
        classification = FAIL_CLASSIFICATION
    next_task = PASS_NEXT_TASK if classification == PASS_CLASSIFICATION else FAIL_NEXT_TASK

    contact_path = make_contact_sheet(output_records, inventory_by_id)
    review_records = []
    for item in output_records:
        review = {
            "request_id": item["request_id"],
            "candidate_path": item["output_path"],
            "candidate_sha256": item["output_sha256"],
            "garment_id": item["garment_id"],
            "slot_id": item["slot_id"],
            "camera_id": item["camera_id"],
            "native_resolution_status": "PASS" if item["native_resolution_pass"] else "FAIL",
            "visual_review_status": "PENDING_HUMAN_REVIEW",
        }
        review.update({field: None for field in VISUAL_FIELDS})
        review_records.append(review)
    review_manifest = {
        "schema_version": "canondressgs.subject00.portrait_canary_review_manifest.v1",
        "task_id": TASK_ID,
        "record_count": 4,
        "records": review_records,
        "visual_decision_count": 0,
        "accepted_count": 0,
        "teacher_target_count": 0,
    }
    review_manifest_path = DATA_ROOT / "05_human_review" / "portrait_canary_review_manifest.json"
    write_json(review_manifest_path, review_manifest)

    result_head = os.environ.get("SUBJECT00_PORTRAIT_CANARY_RESULT_HEAD", CANARY_RESULT_HEAD)
    selection_git = {
        "schema_version": "canondressgs.subject00.portrait_canary_request_selection.v1",
        "task_id": TASK_ID,
        "source": {"branch": SOURCE_BRANCH, "head": SOURCE_HEAD},
        "strict_rerun_set_count": 43,
        "selected_count": 4,
        "selected_request_ids": progress["stable_order"],
        "native_pass_overlap_count": 0,
        "records": selection_external["records"],
        "paper_final": PAPER_FINAL,
    }
    canvas_git = {
        "schema_version": "canondressgs.subject00.portrait_canvas_registry.v1",
        "task_id": TASK_ID,
        "path_type": PATH_TYPE,
        "canvas_count": 4,
        "records": canvas_external["records"],
        "asset_role": "DERIVED_EDIT_INPUT_CANVAS",
        "generated_candidate": False,
        "paper_final": PAPER_FINAL,
    }
    outputs_git = {
        "schema_version": "canondressgs.subject00.portrait_canary_output_registry.v1",
        "task_id": TASK_ID,
        "path_type": PATH_TYPE,
        "record_count": 4,
        "records": output_records,
        "output_count": output_count,
        "paper_final": PAPER_FINAL,
    }
    technical_git = {
        "schema_version": "canondressgs.subject00.portrait_canary_technical_verification.v1",
        "task_id": TASK_ID,
        "selected_count": 4,
        "generation_primary_calls": progress.get("primary_calls", 0),
        "technical_retries": progress.get("technical_retries", 0),
        "generation_calls": progress.get("generation_calls", 0),
        "output_count": output_count,
        "png_parse_count": parse_count,
        "native_1024x1536_pass_count": native_pass_count,
        "unique_output_sha_count": len(set(output_shas)),
        "duplicate_output_sha_count": duplicate_count,
        "provenance_count": provenance_count,
        "output_postprocessing_count": 0,
        "attempt_001_mutation_count": len(mutation_ids),
        "preserved_native_pass_integrity_errors": preserved_native_errors,
        "external_api_calls": 0,
        "sublyx_calls": 0,
        "code78_calls": 0,
        "api_key_reads": 0,
        "cloud_image_writes": 0,
        "accepted_count": 0,
        "teacher_target_count": 0,
        "formal_base": FORMAL_BASE,
        "paper_final": PAPER_FINAL,
    }
    review_contract = {
        "schema_version": "canondressgs.subject00.portrait_canary_review_contract.v1",
        "task_id": TASK_ID,
        "contact_sheet_path": str(contact_path),
        "review_manifest_path": str(review_manifest_path),
        "record_count": 4,
        "visual_fields_initial_state": None,
        "visual_decision_count": 0,
        "automated_acceptance_allowed": False,
        "remaining_39_rerun_authorized": False,
        "paper_final": PAPER_FINAL,
    }
    tests = {
        "schema_version": "canondressgs.subject00.portrait_canary_tests.v1",
        "task_id": TASK_ID,
        "status": "NOT_RUN",
        "total": 0,
        "passed": 0,
        "failed": [],
        "checks": [],
        "paper_final": PAPER_FINAL,
    }
    final_summary = {
        "schema_version": "canondressgs.subject00.portrait_canary_final_summary.v1",
        "task_id": TASK_ID,
        "source": {"branch": SOURCE_BRANCH, "head": SOURCE_HEAD},
        "branch": BRANCH,
        "attempt_002_root": str(DATA_ROOT),
        "path_type": PATH_TYPE,
        "resolution_contract": "STRICT_NATIVE_RESOLUTION",
        "selected_request_ids": progress["stable_order"],
        "generation_primary_calls": progress.get("primary_calls", 0),
        "technical_retries": progress.get("technical_retries", 0),
        "generation_calls": progress.get("generation_calls", 0),
        "output_count": output_count,
        "png_parse_count": parse_count,
        "native_1024x1536_pass_count": native_pass_count,
        "duplicate_output_sha_count": duplicate_count,
        "attempt_001_mutation_count": len(mutation_ids),
        "preserved_native_pass_count": 5,
        "preserved_native_pass_integrity_errors": preserved_native_errors,
        "output_postprocessing_count": 0,
        "remaining_39_rerun_executed": 0,
        "external_api_calls": 0,
        "sublyx_calls": 0,
        "code78_calls": 0,
        "api_key_reads": 0,
        "cloud_image_writes": 0,
        "visual_decision_count": 0,
        "accepted_count": 0,
        "teacher_target_count": 0,
        "formal_base": FORMAL_BASE,
        "tests": "NOT_RUN",
        "portrait_canary_result_head": result_head,
        "final_reporting_head_resolution": "git rev-parse HEAD after final reporting commit",
        "paper_final": PAPER_FINAL,
        "classification": classification,
        "next_task": next_task,
        "next_task_authorized": False,
    }
    handoff = {
        "schema_version": "canondressgs.subject00.managed_portrait_canary_handoff.v1",
        "task_id": TASK_ID,
        "source": {"branch": SOURCE_BRANCH, "head": SOURCE_HEAD},
        "branch": BRANCH,
        "attempt_002_root": str(DATA_ROOT),
        "classification": classification,
        "selected_request_ids": progress["stable_order"],
        "native_1024x1536_pass_count": native_pass_count,
        "remaining_39_rerun_authorized": False,
        "formal_base": FORMAL_BASE,
        "tests": "NOT_RUN",
        "portrait_canary_result_head": result_head,
        "final_reporting_head_resolution": "git rev-parse HEAD after final reporting commit",
        "paper_final": PAPER_FINAL,
        "next_task": next_task,
        "next_task_authorized": False,
    }
    git_outputs = {
        "subject00_portrait_canary_request_selection.json": selection_git,
        "subject00_portrait_canvas_registry.json": canvas_git,
        "subject00_portrait_canary_output_registry.json": outputs_git,
        "subject00_portrait_canary_technical_verification.json": technical_git,
        "subject00_portrait_canary_review_contract.json": review_contract,
        "subject00_portrait_canary_tests.json": tests,
        "subject00_portrait_canary_final_summary.json": final_summary,
    }
    for name, payload in git_outputs.items():
        write_json(RISK / name, payload)
    write_json(HANDOFF / "subject00_managed_portrait_canary_handoff.json", handoff)
    write_json(
        DATA_ROOT / "12_final_verification" / "subject00_portrait_canary_technical_verification.json",
        technical_git,
    )

    write_doc(
        DOCS / "AAAI27_SUBJECT00_MANAGED_PORTRAIT_CANARY_REPORT_20260725.md",
        f"""
# Subject00 Managed Portrait Canary Report

- Task: `{TASK_ID}`
- Source: `{SOURCE_BRANCH}` at `{SOURCE_HEAD}`
- Resolution contract: `STRICT_NATIVE_RESOLUTION`
- Attempt root: `{DATA_ROOT}`
- Selected requests: {', '.join(progress['stable_order'])}
- Primary calls: {progress.get('primary_calls', 0)}
- Technical retries: {progress.get('technical_retries', 0)}
- Output count: {output_count}
- PNG parse count: {parse_count}
- Native 1024x1536 pass count: {native_pass_count}
- Output postprocessing count: 0
- Attempt 001 mutation count: 0
- Preserved native PASS outputs: 5/5
- Remaining 39 reruns executed: 0
- Accepted/Teacher targets: 0/0
- Formal Base: `PENDING`
- Classification: `{classification}`
- PAPER_FINAL: `false`
- Contract tests: `RUN_AFTER_CANARY_EMISSION`

This task does not authorize or execute the remaining 39 reruns. Visual review fields remain null.
""",
    )
    write_doc(
        DOCS / "AAAI27_SUBJECT00_PORTRAIT_CANVAS_CONTRACT_20260725.md",
        f"""
# Subject00 Portrait Canvas Contract

Each canary uses a deterministic RGB PNG canvas at 1024x1536. The unchanged condition image is contained with Lanczos inside a 960x1472 box, centered without crop, rotation, stretch, text, borders, arrows, labels, or watermark. Background color is the stable median of source border pixels, with RGB(245,245,245) reserved as fallback.

The portrait canvas is `DERIVED_EDIT_INPUT_CANVAS`, not a generated candidate, accepted image, Teacher target, or scientific result. It is Image 1 and the primary edit target; the unchanged original Subject00 condition image is Image 2. Attempt 001 generated outputs are never generation references.

Generated outputs must be natively returned as 1024x1536 PNG. Output resize, crop, pad, rotate, stretch, and canvas embedding are forbidden. A valid but wrong-size model output is `NATIVE_RESOLUTION_FAIL` and is not retried or repaired.

The remaining 39 reruns require a 4/4 technical pass, user inspection of `{contact_path}`, and explicit authorization.
""",
    )
    print(
        json.dumps(
            {
                "classification": classification,
                "next_task": next_task,
                "output_count": output_count,
                "png_parse_count": parse_count,
                "native_pass_count": native_pass_count,
                "primary_calls": progress.get("primary_calls", 0),
                "technical_retries": progress.get("technical_retries", 0),
                "contact_sheet": str(contact_path),
                "review_manifest": str(review_manifest_path),
            },
            indent=2,
        )
    )


def verify() -> None:
    selection = load_json(RISK / "subject00_portrait_canary_request_selection.json")
    canvases = load_json(RISK / "subject00_portrait_canvas_registry.json")
    outputs = load_json(RISK / "subject00_portrait_canary_output_registry.json")
    technical = load_json(RISK / "subject00_portrait_canary_technical_verification.json")
    review_contract = load_json(RISK / "subject00_portrait_canary_review_contract.json")
    summary = load_json(RISK / "subject00_portrait_canary_final_summary.json")
    handoff_path = HANDOFF / "subject00_managed_portrait_canary_handoff.json"
    handoff = load_json(handoff_path)
    progress = load_json(progress_path())
    contract = load_json(DATA_ROOT / "00_contract" / "subject00_portrait_canary_contract.json")
    review_path = DATA_ROOT / "05_human_review" / "portrait_canary_review_manifest.json"
    review = load_json(review_path)

    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, actual: Any, expected: Any) -> None:
        checks.append(
            {
                "name": name,
                "status": "PASS" if passed else "FAIL",
                "actual": actual,
                "expected": expected,
            }
        )

    selected_ids = selection["selected_request_ids"]
    expected_ids = [
        "subject00_O01_slot00_cand00",
        "subject00_O03_slot03_cand00",
        "subject00_O04_slot02_cand00",
        "subject00_O01_slot06_cand00",
    ]
    add("selected_request_count", len(selected_ids) == 4, len(selected_ids), 4)
    add("selected_request_order", selected_ids == expected_ids, selected_ids, expected_ids)
    add("selected_request_unique_count", len(set(selected_ids)) == 4, len(set(selected_ids)), 4)
    add(
        "selected_all_strict_rerun_members",
        all(item["strict_rerun_member"] for item in selection["records"]),
        sum(item["strict_rerun_member"] for item in selection["records"]),
        4,
    )
    add("native_pass_overlap_count", selection["native_pass_overlap_count"] == 0, selection["native_pass_overlap_count"], 0)

    source_sha_errors = [
        item["request_id"]
        for item in canvases["records"]
        if not Path(item["source_path"]).is_file()
        or file_sha256(Path(item["source_path"])) != item["source_sha256"]
    ]
    add("source_sha_change_count", not source_sha_errors, source_sha_errors, [])
    add("derived_canvas_count", canvases["canvas_count"] == 4, canvases["canvas_count"], 4)

    canvas_parse_errors: list[str] = []
    canvas_sha_errors: list[str] = []
    canvas_pixel_contract_errors: list[str] = []
    for item in canvases["records"]:
        path = Path(item["derived_canvas_path"])
        if not path.is_file():
            canvas_parse_errors.append(item["request_id"])
            continue
        if file_sha256(path) != item["derived_canvas_sha256"]:
            canvas_sha_errors.append(item["request_id"])
        with Image.open(path) as image:
            image.load()
            if image.format != "PNG" or image.mode != "RGB" or image.size != TARGET_SIZE:
                canvas_parse_errors.append(item["request_id"])
                continue
            with Image.open(Path(item["source_path"])) as source:
                contained = ImageOps.contain(source.convert("RGB"), (960, 1472), Image.Resampling.LANCZOS)
            expected = Image.new("RGB", TARGET_SIZE, tuple(item["background_rgb"]))
            expected.paste(contained, (item["paste_offset"]["x"], item["paste_offset"]["y"]))
            if expected.tobytes() != image.tobytes():
                canvas_pixel_contract_errors.append(item["request_id"])
    add("derived_canvas_png_rgb_1024x1536_count", not canvas_parse_errors, canvas_parse_errors, [])
    add("derived_canvas_sha_match_count", not canvas_sha_errors, canvas_sha_errors, [])
    add("derived_canvas_exact_contain_pixel_contract", not canvas_pixel_contract_errors, canvas_pixel_contract_errors, [])
    add(
        "full_image_contain_count",
        sum(item["full_image_contain"] for item in canvases["records"]) == 4,
        sum(item["full_image_contain"] for item in canvases["records"]),
        4,
    )
    add(
        "canvas_minimum_margin_contract",
        all(min(item["margins"].values()) >= 32 for item in canvases["records"]),
        [item["margins"] for item in canvases["records"]],
        "all margins >= 32",
    )
    add("canvas_crop_operation_count", sum(item["crop_operations"] for item in canvases["records"]) == 0, sum(item["crop_operations"] for item in canvases["records"]), 0)
    add("canvas_rotate_operation_count", sum(item["rotate_operations"] for item in canvases["records"]) == 0, sum(item["rotate_operations"] for item in canvases["records"]), 0)
    add("canvas_stretch_operation_count", sum(item["stretch_operations"] for item in canvases["records"]) == 0, sum(item["stretch_operations"] for item in canvases["records"]), 0)
    add("canvas_asset_role", canvases["asset_role"] == "DERIVED_EDIT_INPUT_CANVAS", canvases["asset_role"], "DERIVED_EDIT_INPUT_CANVAS")
    add("canvas_generated_candidate_false", canvases["generated_candidate"] is False, canvases["generated_candidate"], False)

    request_records = [load_json(request_path(request_id)) for request_id in selected_ids]
    add(
        "managed_input_order_count",
        all(
            request["input_order"]
            == [request["derived_edit_canvas_path"], request["managed_tool_original_reference_path"]]
            for request in request_records
        ),
        sum(
            request["input_order"]
            == [request["derived_edit_canvas_path"], request["managed_tool_original_reference_path"]]
            for request in request_records
        ),
        4,
    )
    add(
        "attempt_001_generation_input_count",
        all(str(ATTEMPT_001).casefold() not in path.casefold() for request in request_records for path in request["input_order"]),
        sum(str(ATTEMPT_001).casefold() in path.casefold() for request in request_records for path in request["input_order"]),
        0,
    )
    add(
        "frozen_portrait_control_clause_count",
        sum(CANVAS_CONTROL_CLAUSE in request["actual_prompt"] for request in request_records) == 4,
        sum(CANVAS_CONTROL_CLAUSE in request["actual_prompt"] for request in request_records),
        4,
    )
    add("primary_call_count", progress["primary_calls"] == 4, progress["primary_calls"], 4)
    add("primary_call_upper_bound", progress["primary_calls"] <= 4, progress["primary_calls"], "<= 4")
    add("technical_retry_count", progress["technical_retries"] == 0, progress["technical_retries"], 0)
    add("technical_retry_upper_bound", progress["technical_retries"] <= 4, progress["technical_retries"], "<= 4")
    add("generation_call_count", progress["generation_calls"] == 4, progress["generation_calls"], 4)

    output_parse_errors: list[str] = []
    output_sha_errors: list[str] = []
    for item in outputs["records"]:
        path = Path(item["output_path"])
        if not path.is_file():
            output_parse_errors.append(item["request_id"])
            continue
        if file_sha256(path) != item["output_sha256"]:
            output_sha_errors.append(item["request_id"])
        with Image.open(path) as image:
            image.load()
            if image.format != "PNG" or image.size != TARGET_SIZE:
                output_parse_errors.append(item["request_id"])
    output_shas = [item["output_sha256"] for item in outputs["records"]]
    add("output_count", outputs["output_count"] == 4, outputs["output_count"], 4)
    add("output_png_parse_count", not output_parse_errors, output_parse_errors, [])
    add("native_1024x1536_pass_count", sum(item["native_resolution_pass"] for item in outputs["records"]) == 4, sum(item["native_resolution_pass"] for item in outputs["records"]), 4)
    add("output_sha_match_count", not output_sha_errors, output_sha_errors, [])
    add("unique_output_sha_count", len(set(output_shas)) == 4, len(set(output_shas)), 4)
    add("duplicate_output_sha_count", len(output_shas) - len(set(output_shas)) == 0, len(output_shas) - len(set(output_shas)), 0)
    add("provenance_record_count", all(provenance_path(request_id).is_file() for request_id in selected_ids), sum(provenance_path(request_id).is_file() for request_id in selected_ids), 4)
    provenance = [load_json(provenance_path(request_id)) for request_id in selected_ids]
    add("managed_backend_provenance_count", sum(item["generation_backend"] == "CODEX_MANAGED_IMAGE_EDIT" for item in provenance) == 4, sum(item["generation_backend"] == "CODEX_MANAGED_IMAGE_EDIT" for item in provenance), 4)
    add("output_postprocessing_count", technical["output_postprocessing_count"] == 0, technical["output_postprocessing_count"], 0)

    for field in ["external_api_calls", "sublyx_calls", "code78_calls", "api_key_reads", "cloud_image_writes"]:
        add(field, technical[field] == 0, technical[field], 0)

    attempt_mutations = [
        item["request_id"]
        for item in contract["attempt_001_baseline"]
        if not Path(item["path"]).is_file() or file_sha256(Path(item["path"])) != item["sha256"]
    ]
    native_pass_mutations = [
        item["request_id"]
        for item in contract["attempt_001_baseline"]
        if item["preserved_native_pass_output"]
        and (not Path(item["path"]).is_file() or file_sha256(Path(item["path"])) != item["sha256"])
    ]
    add("attempt_001_mutation_count", not attempt_mutations, attempt_mutations, [])
    add("preserved_native_pass_integrity", not native_pass_mutations, native_pass_mutations, [])
    add("preserved_native_pass_count", sum(item["preserved_native_pass_output"] for item in contract["attempt_001_baseline"]) == 5, sum(item["preserved_native_pass_output"] for item in contract["attempt_001_baseline"]), 5)

    forbidden_names = {"accepted_rgb", "teacher", "teacher_targets", "controller_split"}
    forbidden_dirs = [str(path) for path in DATA_ROOT.rglob("*") if path.is_dir() and path.name.casefold() in forbidden_names]
    add("forbidden_output_directory_count", not forbidden_dirs, forbidden_dirs, [])
    add("accepted_count", technical["accepted_count"] == 0, technical["accepted_count"], 0)
    add("teacher_target_count", technical["teacher_target_count"] == 0, technical["teacher_target_count"], 0)
    visual_non_null = [
        f"{item['request_id']}:{field}"
        for item in review["records"]
        for field in VISUAL_FIELDS
        if item[field] is not None
    ]
    add("visual_field_non_null_count", not visual_non_null, visual_non_null, [])
    add("visual_decision_count", review["visual_decision_count"] == 0, review["visual_decision_count"], 0)
    add("formal_base_pending", technical["formal_base"] == FORMAL_BASE, technical["formal_base"], FORMAL_BASE)
    add("paper_final_false", all(item["paper_final"] is False for item in [selection, canvases, outputs, technical, review_contract, summary, handoff]), [item["paper_final"] for item in [selection, canvases, outputs, technical, review_contract, summary, handoff]], [False] * 7)
    add("remaining_39_rerun_executed", summary["remaining_39_rerun_executed"] == 0, summary["remaining_39_rerun_executed"], 0)
    add("contact_sheet_exists", Path(review_contract["contact_sheet_path"]).is_file(), review_contract["contact_sheet_path"], "existing external PNG")
    add("review_manifest_exists", review_path.is_file(), str(review_path), "existing external JSON")
    add("technical_classification", summary["classification"] == PASS_CLASSIFICATION, summary["classification"], PASS_CLASSIFICATION)
    add("next_task", summary["next_task"] == PASS_NEXT_TASK, summary["next_task"], PASS_NEXT_TASK)
    tracked_canary_pngs = [path for path in git("ls-files", "*.png").splitlines() if "portrait_canary" in path.casefold()]
    add("git_tracked_canary_png_count", not tracked_canary_pngs, tracked_canary_pngs, [])

    failed = [item["name"] for item in checks if item["status"] == "FAIL"]
    status = "PASS" if not failed else "FAIL"
    tests = {
        "schema_version": "canondressgs.subject00.portrait_canary_tests.v1",
        "task_id": TASK_ID,
        "status": status,
        "total": len(checks),
        "passed": len(checks) - len(failed),
        "failed": failed,
        "checks": checks,
        "paper_final": PAPER_FINAL,
    }
    write_json(RISK / "subject00_portrait_canary_tests.json", tests)
    write_json(DATA_ROOT / "12_final_verification" / "subject00_portrait_canary_tests.json", tests)

    result_head = os.environ.get("SUBJECT00_PORTRAIT_CANARY_RESULT_HEAD")
    if result_head:
        summary["portrait_canary_result_head"] = result_head
        handoff["portrait_canary_result_head"] = result_head
    summary["tests"] = f"{status} ({tests['passed']}/{tests['total']})"
    handoff["tests"] = summary["tests"]
    write_json(RISK / "subject00_portrait_canary_final_summary.json", summary)
    write_json(handoff_path, handoff)

    report_path = DOCS / "AAAI27_SUBJECT00_MANAGED_PORTRAIT_CANARY_REPORT_20260725.md"
    report_lines = report_path.read_text(encoding="utf-8").splitlines()
    report_lines = [
        f"- Contract tests: `{status} ({tests['passed']}/{tests['total']})`"
        if line.startswith("- Contract tests:")
        else line
        for line in report_lines
    ]
    write_doc(report_path, "\n".join(report_lines))
    print(json.dumps({"status": status, "passed": tests["passed"], "total": tests["total"], "failed": failed}, indent=2))
    if failed:
        raise SystemExit(1)


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
    success.add_argument("--technical-retry-count", type=int, required=True)
    success.add_argument("--tool-output-hint", default="")
    failure = sub.add_parser("record-failure")
    failure.add_argument("--request-id", required=True)
    failure.add_argument("--reason", required=True)
    failure.add_argument("--started-at", required=True)
    failure.add_argument("--ended-at", required=True)
    failure.add_argument("--technical-retry-count", type=int, required=True)
    sub.add_parser("finalize")
    sub.add_parser("verify")
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
    elif args.command == "verify":
        verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
