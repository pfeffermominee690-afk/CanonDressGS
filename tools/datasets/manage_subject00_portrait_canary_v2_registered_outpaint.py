#!/usr/bin/env python3
"""Manage the four-request Subject00 registered-crop vertical-outpaint canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[2]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
DOCS = ROOT / "docs" / "PAPER"
HANDOFF = ROOT / "project_control_handoff"
ATTEMPT_001 = Path(r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-GENERATION-001\attempt_001")
ATTEMPT_002 = Path(r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-GENERATION-001\attempt_002_portrait_canary")
DATA_ROOT = Path(r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-GENERATION-001\attempt_003_portrait_canary_v2_registered_outpaint")
SOURCE_CACHE = DATA_ROOT / "00_source_evidence"
PREFLIGHT = DATA_ROOT / "01_registration_preflight"

TASK_ID = "AAAI27-SUBJECT00-PORTRAIT-CANARY-V2-REGISTERED-OUTPAINT-001"
SOURCE_BRANCH = "research/subject00-portrait-canary-visual-fail-adjudication-20260726"
SOURCE_HEAD = "5a992921a5e33d1a23a332094ee9383f441217bd"
BRANCH = "research/subject00-portrait-canary-v2-registered-outpaint-20260726"
PROTOCOL = "REGISTERED_HORIZONTAL_CROP_PLUS_VERTICAL_OUTPAINT"
FORMAL_BASE = "PENDING"
PAPER_FINAL = False
TARGET_SIZE = (1024, 1536)
SOURCE_SIZE = (1330, 1150)
TOP_EXTENSION = 193
BOTTOM_EXTENSION = 193
PASS_CLASSIFICATION = "SUBJECT00_PORTRAIT_CANARY_V2_TECHNICAL_PASS_PENDING_USER_REVIEW"
INCOMPLETE_CLASSIFICATION = "SUBJECT00_PORTRAIT_CANARY_V2_TECHNICAL_INCOMPLETE"
BLOCKED_CLASSIFICATION = "SUBJECT00_PORTRAIT_CANARY_V2_BLOCKED_BY_REGISTRATION_CONTRACT"
PASS_NEXT_TASK = "USER_REVIEW_SUBJECT00_PORTRAIT_CANARY_V2_CONTACT_SHEET"
BLOCKED_NEXT_TASK = "USER_ADJUDICATE_SUBJECT00_CAMERA_REGISTRATION_BLOCKER"

REQUESTS = [
    ("subject00_O01_slot00_cand00", "O01", "slot_00", "cam17", "front"),
    ("subject00_O03_slot03_cand00", "O03", "slot_03", "cam23", "left"),
    ("subject00_O04_slot02_cand00", "O04", "slot_02", "cam14", "front-right"),
    ("subject00_O01_slot06_cand00", "O01", "slot_06", "cam09", "back-right"),
]

RGB_PATHS = {
    "cam17": Path(r"E:\model_train\_subject00_identity_audit_tmp\cam17_frame00000000.jpg"),
    "cam23": Path(r"E:\model_train\_subject00_identity_audit_tmp\cam23_frame00000000.jpg"),
    "cam14": Path(r"E:\model_train\_subject00_identity_audit_tmp\cam14_frame00000000.jpg"),
    "cam09": Path(r"E:\model_train\_subject00_identity_audit_tmp\cam09_frame00000000.jpg"),
}

MASK_PATHS = {
    camera: SOURCE_CACHE / f"{camera}_mask_00000000.jpg" for camera in RGB_PATHS
}

EXPECTED_RGB_SHA = {
    "cam17": "b5e790c6e978fa0ecbf7bcad3fe64aa64382f67f22ff3f1b0bb3d0d33764abdf",
    "cam23": "a0c34bd8bee414ef2aa06b8457ef661790ff9956aa9b1b16079e15dc6fb1a23b",
    "cam14": "575d04d79c3b89e1f94171ab42c80591fa843429ee815fef823bdd13c24fba33",
    "cam09": "67c2616c073c57165efe54982bf2cd18a47d0be18cd1d67576ad4020af70c373",
}

EXPECTED_MASK_SHA = {
    "cam17": "c54df74796fb9fdb500f264ebf6c9d534200a9051e8ce80eb2f2c985c153e234",
    "cam23": "65d115eb03bdf718ac37159aa47919343fbca202d94071a4308cd6943ca3ef1d",
    "cam14": "7acb6949e6743ce978fa520f3ec76d9495224789d64e9f75e7ef9fb209fdb71e",
    "cam09": "cd48ea97b973f503caaf71f266c3216740b510cac6943e281e09d63d3b10a2ec",
}

CALIBRATION_SHA = "4b2d98d20740da03be209040851fab7e8801e5670937ed9ad290a20264d1dde7"
VISUAL_FIELDS = [
    "identity_match", "pose_match", "camera_match", "garment_match", "full_body_complete",
    "hands_complete", "feet_complete", "background_match", "artifact_grade", "decision",
]

V2_CLAUSE = """Input roles:
Image 1 is the registered 1024x1536 edit target. Its central rows 193 through 1342 are a 1:1 pixel crop from the unchanged condition. Replace the reflected seed content in the top and bottom extension bands with natural continuation of the same ceiling, walls, and floor.
Image 2 is the unchanged original Subject00 condition and is the sole authority for identity, pose, camera direction, subject pixel scale, subject center, ground contact, lighting, and scene structure.
Image 3 is a technical edit-region map: red is top scene outpaint, blue is bottom scene outpaint, green is the conservative person/garment edit envelope, and black is protected. The map is guidance only and must not appear in the output.

Preserve the central registered scene geometry and keep the subject at the same pixel size and location. Edit clothing only inside the person envelope while preserving head, face, visible hairline, skin, hands, feet, shoes, body shape, and pose from Image 2. Do not zoom out, reframe, shrink, translate, rotate, warp, duplicate, crop, or recompose the person. Do not retain mirrored extension artifacts, flat-color padding, borders, text, labels, masks, or watermarks.

Return exactly one native RGB PNG at 1024 pixels wide and 1536 pixels high. Do not return any alternate candidate."""


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
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.pop("content_sha256", None)
    payload["content_sha256"] = canonical_sha256(payload)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_doc(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8", newline="\n")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def full_inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path),
            "relative_path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


def threshold_mask(path: Path) -> Image.Image:
    with Image.open(path) as source:
        return source.convert("L").point(lambda value: 255 if value >= 128 else 0)


def request_path(request_id: str) -> Path:
    return DATA_ROOT / "05_generation_requests" / "requests" / f"{request_id}.json"


def progress_path() -> Path:
    return DATA_ROOT / "05_generation_requests" / "generation_progress.json"


def output_path(request_id: str) -> Path:
    return DATA_ROOT / "06_generation_responses" / request_id / f"{request_id}.png"


def provenance_path(request_id: str) -> Path:
    return DATA_ROOT / "08_provenance" / "requests" / f"{request_id}.json"


def preflight() -> None:
    if git("branch", "--show-current") != BRANCH:
        raise RuntimeError("wrong V2 branch")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", SOURCE_HEAD, "HEAD"], cwd=ROOT
    ).returncode
    if ancestor != 0:
        raise RuntimeError("source HEAD is not an ancestor")

    calibration_path = SOURCE_CACHE / "calibration.json"
    if file_sha256(calibration_path) != CALIBRATION_SHA:
        raise RuntimeError("calibration SHA mismatch")
    calibration = load_json(calibration_path)
    source_requests = {
        request_id: load_json(ATTEMPT_002 / "03_generation_requests" / "requests" / f"{request_id}.json")
        for request_id, *_ in REQUESTS
    }
    attempt_002_manifest = ATTEMPT_002 / "05_human_review" / "portrait_canary_review_manifest.json"
    baseline = {
        "attempt_001": full_inventory(ATTEMPT_001),
        "attempt_002": full_inventory(ATTEMPT_002),
        "attempt_002_review_manifest_sha256": file_sha256(attempt_002_manifest),
    }
    write_json(DATA_ROOT / "00_source_evidence" / "prior_attempt_immutability_baseline.json", baseline)

    records = []
    for request_id, garment, slot, camera, orientation in REQUESTS:
        rgb_path = RGB_PATHS[camera]
        mask_path = MASK_PATHS[camera]
        if file_sha256(rgb_path) != EXPECTED_RGB_SHA[camera]:
            raise RuntimeError(f"RGB SHA mismatch: {camera}")
        if file_sha256(mask_path) != EXPECTED_MASK_SHA[camera]:
            raise RuntimeError(f"mask SHA mismatch: {camera}")
        with Image.open(rgb_path) as source_image:
            source = source_image.convert("RGB")
        if source.size != SOURCE_SIZE:
            raise RuntimeError(f"source size mismatch: {camera}")
        person_mask = threshold_mask(mask_path)
        if person_mask.size != SOURCE_SIZE or person_mask.getbbox() is None:
            raise RuntimeError(f"person mask invalid: {camera}")

        camera_data = calibration[camera]
        k = camera_data["K"]
        fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])
        if camera_data["imgSize"] != [1330, 1150]:
            raise RuntimeError(f"calibration image size mismatch: {camera}")
        x_left = max(0, min(round(cx - 512), SOURCE_SIZE[0] - 1024))
        x_right = x_left + 1024
        outside = Image.new("L", SOURCE_SIZE, 255)
        draw = ImageDraw.Draw(outside)
        draw.rectangle((x_left, 0, x_right - 1, SOURCE_SIZE[1] - 1), fill=0)
        removed_person = ImageChops.logical_and(person_mask.convert("1"), outside.convert("1"))
        intersection_pixels = sum(1 for value in removed_person.getdata() if value)
        person_bbox = person_mask.getbbox()
        registration_pass = intersection_pixels == 0 and person_bbox is not None

        crop = source.crop((x_left, 0, x_right, SOURCE_SIZE[1]))
        crop_path = DATA_ROOT / "02_registered_crops" / f"{request_id}_registered_crop.png"
        crop.save(crop_path, format="PNG", optimize=True)

        canvas = Image.new("RGB", TARGET_SIZE)
        top_seed = ImageOps.flip(crop.crop((0, 0, 1024, TOP_EXTENSION)))
        bottom_seed = ImageOps.flip(crop.crop((0, SOURCE_SIZE[1] - BOTTOM_EXTENSION, 1024, SOURCE_SIZE[1])))
        canvas.paste(top_seed, (0, 0))
        canvas.paste(crop, (0, TOP_EXTENSION))
        canvas.paste(bottom_seed, (0, TOP_EXTENSION + SOURCE_SIZE[1]))
        canvas_path = DATA_ROOT / "03_derived_edit_canvases" / f"{request_id}_registered_outpaint_canvas.png"
        canvas.save(canvas_path, format="PNG", optimize=True)

        crop_mask = person_mask.crop((x_left, 0, x_right, SOURCE_SIZE[1]))
        edit_map = Image.new("RGB", TARGET_SIZE, (0, 0, 0))
        edit_map.paste((255, 0, 0), (0, 0, 1024, TOP_EXTENSION))
        edit_map.paste((0, 0, 255), (0, TOP_EXTENSION + SOURCE_SIZE[1], 1024, TARGET_SIZE[1]))
        green = Image.new("RGB", (1024, SOURCE_SIZE[1]), (0, 255, 0))
        edit_map.paste(green, (0, TOP_EXTENSION), crop_mask)
        edit_map_path = DATA_ROOT / "04_edit_masks" / f"{request_id}_edit_regions.png"
        edit_map.save(edit_map_path, format="PNG", optimize=True)

        bbox_crop = [person_bbox[0] - x_left, person_bbox[1], person_bbox[2] - x_left, person_bbox[3]]
        bbox_canvas = [bbox_crop[0], bbox_crop[1] + TOP_EXTENSION, bbox_crop[2], bbox_crop[3] + TOP_EXTENSION]
        record = {
            "request_id": request_id,
            "garment_id": garment,
            "slot_id": slot,
            "camera_id": int(camera[3:]),
            "camera_name": camera,
            "orientation": orientation,
            "source_path": str(rgb_path),
            "source_sha256": file_sha256(rgb_path),
            "source_size": {"width": 1330, "height": 1150},
            "source_mask_path": str(mask_path),
            "source_mask_sha256": file_sha256(mask_path),
            "calibration_path": str(calibration_path),
            "calibration_sha256": CALIBRATION_SHA,
            "camera_binding": source_requests[request_id]["source_input_paths"],
            "camera_intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
            "distortion_contract": {"distCoeff": camera_data["distCoeff"], "rectifyAlpha": camera_data["rectifyAlpha"]},
            "extrinsics": {
                "world_to_camera_rotation_R": camera_data["R"],
                "world_to_camera_translation_T": camera_data["T"],
                "provenance": "scene/dataset.py constructs w2c = [[R,T],[0,0,0,1]]",
            },
            "crop_window": {"x_left": x_left, "x_right": x_right, "y_top": 0, "y_bottom": 1150},
            "updated_intrinsics": {"fx": fx, "fy": fy, "cx": cx - x_left, "cy": cy + TOP_EXTENSION},
            "top_extension": TOP_EXTENSION,
            "bottom_extension": BOTTOM_EXTENSION,
            "source_region_destination": {"x_left": 0, "x_right": 1024, "y_top": TOP_EXTENSION, "y_bottom": TOP_EXTENSION + 1150},
            "source_region_scale_x": 1.0,
            "source_region_scale_y": 1.0,
            "expected_subject_height_ratio": 1.0,
            "person_bbox_source": list(person_bbox),
            "person_bbox_registered_crop": bbox_crop,
            "person_bbox_derived_canvas": bbox_canvas,
            "garment_bbox_contract": {"status": "CONSERVATIVE_PERSON_ENVELOPE", "bbox": list(person_bbox)},
            "hands_feet_bbox_contract": {"status": "CONSERVATIVE_PERSON_ENVELOPE", "bbox": list(person_bbox)},
            "crop_person_intersection_pixels": intersection_pixels,
            "crop_garment_intersection_pixels": intersection_pixels,
            "crop_hands_feet_intersection_pixels": intersection_pixels,
            "registered_crop_path": str(crop_path),
            "registered_crop_sha256": file_sha256(crop_path),
            "derived_canvas_path": str(canvas_path),
            "derived_canvas_sha256": file_sha256(canvas_path),
            "edit_mask_path": str(edit_map_path),
            "edit_mask_sha256": file_sha256(edit_map_path),
            "seed_contract": "REFLECTED_SCENE_EDGE_SEED_REPLACED_BY_MASKED_OUTPAINT",
            "registration_decision": "REGISTRATION_PREFLIGHT_PASS" if registration_pass else "REGISTRATION_PREFLIGHT_FAIL",
        }
        records.append(record)

        source_request = source_requests[request_id]
        actual_prompt = source_request["source_actual_prompt"] + "\n\n" + V2_CLAUSE
        request = {
            "schema_version": "canondressgs.subject00.portrait_canary_v2_request.v1",
            "task_id": TASK_ID,
            "request_id": request_id,
            "garment_id": garment,
            "slot_id": slot,
            "camera_id": int(camera[3:]),
            "orientation": orientation,
            "protocol": PROTOCOL,
            "actual_prompt": actual_prompt,
            "actual_prompt_sha256": hashlib.sha256(actual_prompt.encode("utf-8")).hexdigest(),
            "input_order": [str(canvas_path), str(rgb_path), str(edit_map_path)],
            "input_roles": ["REGISTERED_OUTPAINT_EDIT_TARGET", "UNCHANGED_CONDITION_AUTHORITY", "TECHNICAL_EDIT_REGION_MAP"],
            "authority_reference_sha256": file_sha256(rgb_path),
            "derived_canvas_sha256": file_sha256(canvas_path),
            "edit_mask_sha256": file_sha256(edit_map_path),
            "target_resolution": {"width": 1024, "height": 1536, "format": "PNG"},
            "generation_backend": "CODEX_MANAGED_IMAGE_EDIT",
            "retry_allowed": False,
            "output_postprocessing_allowed": False,
        }
        write_json(request_path(request_id), request)

    phase_pass = len(records) == 4 and all(item["registration_decision"] == "REGISTRATION_PREFLIGHT_PASS" for item in records)
    camera_audit = {
        "schema_version": "canondressgs.subject00.v2.camera_binding_audit.v1",
        "task_id": TASK_ID,
        "calibration_sha256": CALIBRATION_SHA,
        "camera_count": len(records),
        "records": records,
        "camera_binding_status": "PASS" if phase_pass else "FAIL",
        "camera_intrinsics_status": "PASS" if phase_pass else "FAIL",
        "phase_0_status": "REGISTRATION_PREFLIGHT_PASS" if phase_pass else "REGISTRATION_PREFLIGHT_FAIL",
    }
    crop_manifest = {
        "schema_version": "canondressgs.subject00.v2.crop_geometry_manifest.v1",
        "task_id": TASK_ID,
        "protocol": PROTOCOL,
        "records": records,
        "phase_0_pass_count": sum(item["registration_decision"] == "REGISTRATION_PREFLIGHT_PASS" for item in records),
    }
    transform_manifest = {
        "schema_version": "canondressgs.subject00.v2.portrait_canvas_transform_manifest.v1",
        "task_id": TASK_ID,
        "target_size": {"width": 1024, "height": 1536},
        "source_region_size": {"width": 1024, "height": 1150},
        "top_extension": TOP_EXTENSION,
        "bottom_extension": BOTTOM_EXTENSION,
        "source_region_scale_x": 1.0,
        "source_region_scale_y": 1.0,
        "records": records,
    }
    protected_audit = {
        "schema_version": "canondressgs.subject00.v2.protected_region_intersection_audit.v1",
        "task_id": TASK_ID,
        "garment_and_hands_feet_contract": "PERSON_MASK_CONSERVATIVE_ENVELOPE",
        "records": [
            {
                "request_id": item["request_id"],
                "person_intersection_pixels": item["crop_person_intersection_pixels"],
                "garment_intersection_pixels": item["crop_garment_intersection_pixels"],
                "hands_feet_intersection_pixels": item["crop_hands_feet_intersection_pixels"],
                "decision": item["registration_decision"],
            }
            for item in records
        ],
        "phase_0_status": "REGISTRATION_PREFLIGHT_PASS" if phase_pass else "REGISTRATION_PREFLIGHT_FAIL",
    }
    write_json(PREFLIGHT / "camera_binding_audit.json", camera_audit)
    write_json(PREFLIGHT / "crop_geometry_manifest.json", crop_manifest)
    write_json(PREFLIGHT / "portrait_canvas_transform_manifest.json", transform_manifest)
    write_json(PREFLIGHT / "protected_region_intersection_audit.json", protected_audit)
    make_registration_contact_sheet(records)
    write_doc(
        PREFLIGHT / "registration_preflight_report.md",
        f"""
# Subject00 V2 Registration Preflight

- Task: `{TASK_ID}`
- Protocol: `{PROTOCOL}`
- Calibration SHA: `{CALIBRATION_SHA}`
- Camera binding: `{'PASS' if phase_pass else 'FAIL'}`
- Camera intrinsics: `{'PASS' if phase_pass else 'FAIL'}`
- Registered crop count: `{len(records)}`
- Source-region scale: `1.0 x 1.0`
- Top/bottom extensions: `{TOP_EXTENSION} / {BOTTOM_EXTENSION}`
- Protected-region crop intersections: `{sum(item['crop_person_intersection_pixels'] for item in records)}` pixels
- Phase 0: `{'REGISTRATION_PREFLIGHT_PASS' if phase_pass else 'REGISTRATION_PREFLIGHT_FAIL'}`

`R` and `T` are interpreted as world-to-camera rotation and translation by `scene/dataset.py`. The crop is centered on the calibrated principal point and clamps only to image bounds. The official person mask is used as a conservative envelope for person, garment, hands, and feet; zero person-envelope intersection therefore proves the narrower protected regions are retained.
""",
    )
    progress = {
        "schema_version": "canondressgs.subject00.portrait_canary_v2_progress.v1",
        "task_id": TASK_ID,
        "phase_0_status": "REGISTRATION_PREFLIGHT_PASS" if phase_pass else "REGISTRATION_PREFLIGHT_FAIL",
        "stable_order": [item[0] for item in REQUESTS],
        "generation_calls": 0,
        "technical_retries": 0,
        "external_api_calls": 0,
        "api_key_reads": 0,
        "cloud_image_writes": 0,
        "records": [
            {
                "request_id": request_id,
                "status": "PENDING" if phase_pass else "BLOCKED_PHASE_0",
                "generation_call_id": None,
                "output_path": None,
                "output_sha256": None,
                "format": None,
                "width": None,
                "height": None,
                "started_at": None,
                "ended_at": None,
                "failure_reason": None,
                "retry_count": 0,
                "postprocessing_count": 0,
            }
            for request_id, *_ in REQUESTS
        ],
    }
    write_json(progress_path(), progress)
    print(json.dumps({"phase_0_status": progress["phase_0_status"], "pass_count": sum(item["registration_decision"] == "REGISTRATION_PREFLIGHT_PASS" for item in records)}, indent=2))


def make_registration_contact_sheet(records: list[dict[str, Any]]) -> Path:
    panel_width, row_height = 520, 430
    sheet = Image.new("RGB", (panel_width * 4, row_height * 4), (28, 31, 34))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for row, item in enumerate(records):
        y0 = row * row_height
        source_path = Path(item["source_path"])
        with Image.open(source_path) as image:
            source = image.convert("RGB")
        overlay = source.copy()
        overlay_draw = ImageDraw.Draw(overlay)
        xl, xr = item["crop_window"]["x_left"], item["crop_window"]["x_right"]
        overlay_draw.line((xl, 0, xl, source.height - 1), fill=(255, 50, 50), width=5)
        overlay_draw.line((xr - 1, 0, xr - 1, source.height - 1), fill=(255, 50, 50), width=5)
        overlay_draw.rectangle(tuple(item["person_bbox_source"]), outline=(0, 255, 80), width=5)
        paths = [source_path, Path(item["registered_crop_path"]), Path(item["derived_canvas_path"]), None]
        labels = ["ORIGINAL", "REGISTERED CROP", "DERIVED CANVAS", "REGISTRATION OVERLAY"]
        images = []
        for path in paths[:3]:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        images.append(overlay)
        draw.text((8, y0 + 7), f"{item['request_id']} {item['camera_name']} crop={xl}:{xr}", fill="white", font=font)
        for col, (image, label) in enumerate(zip(images, labels)):
            preview = image.copy()
            preview.thumbnail((panel_width - 16, row_height - 58), Image.Resampling.LANCZOS)
            x0 = col * panel_width + (panel_width - preview.width) // 2
            py = y0 + 30 + (row_height - 58 - preview.height) // 2
            sheet.paste(preview, (x0, py))
            draw.text((col * panel_width + 8, y0 + row_height - 22), label, fill="white", font=font)
    path = PREFLIGHT / "registration_overlay_contact_sheet.png"
    sheet.save(path, format="PNG", optimize=True)
    return path


def next_request() -> None:
    progress = load_json(progress_path())
    if progress["phase_0_status"] != "REGISTRATION_PREFLIGHT_PASS":
        print(json.dumps({"status": "BLOCKED_PHASE_0"}))
        return
    for record in progress["records"]:
        if record["status"] == "PENDING":
            request = load_json(request_path(record["request_id"]))
            request["call_index"] = progress["generation_calls"] + 1
            print(json.dumps(request, indent=2))
            return
    print(json.dumps({"status": "NO_PENDING_REQUEST"}))


def record_success(args: argparse.Namespace) -> None:
    progress = load_json(progress_path())
    record = next(item for item in progress["records"] if item["request_id"] == args.request_id)
    if record["status"] != "PENDING":
        raise RuntimeError("request is not pending")
    source = Path(args.generated_source)
    if not source.is_file():
        raise FileNotFoundError(source)
    with Image.open(source) as image:
        image.load()
        image_format = image.format
        width, height = image.size
    destination = output_path(args.request_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if file_sha256(source) != file_sha256(destination):
        raise RuntimeError("byte-for-byte output copy failed")
    status = "NATIVE_RESOLUTION_PASS" if image_format == "PNG" and (width, height) == TARGET_SIZE else "NATIVE_RESOLUTION_FAIL"
    record.update({
        "status": status,
        "generation_call_id": args.generation_call_id,
        "output_path": str(destination),
        "output_sha256": file_sha256(destination),
        "format": image_format,
        "width": width,
        "height": height,
        "started_at": args.started_at,
        "ended_at": args.ended_at,
        "failure_reason": None,
    })
    progress["generation_calls"] += 1
    write_json(progress_path(), progress)
    request = load_json(request_path(args.request_id))
    write_json(
        provenance_path(args.request_id),
        {
            "schema_version": "canondressgs.subject00.portrait_canary_v2_provenance.v1",
            "task_id": TASK_ID,
            "request_id": args.request_id,
            "generation_backend": "CODEX_MANAGED_IMAGE_EDIT",
            "provider_model_id": "NOT_EXPOSED_BY_MANAGED_TOOL",
            "generation_call_id": args.generation_call_id,
            "input_order": request["input_order"],
            "authority_reference_sha256": request["authority_reference_sha256"],
            "derived_canvas_sha256": request["derived_canvas_sha256"],
            "edit_mask_sha256": request["edit_mask_sha256"],
            "output_path": str(destination),
            "output_sha256": file_sha256(destination),
            "actual_resolution": {"width": width, "height": height},
            "format": image_format,
            "started_at": args.started_at,
            "ended_at": args.ended_at,
            "retry_count": 0,
            "postprocessing_count": 0,
            "status": status,
        },
    )
    print(json.dumps({"request_id": args.request_id, "status": status, "width": width, "height": height, "sha256": file_sha256(destination)}, indent=2))


def record_failure(args: argparse.Namespace) -> None:
    progress = load_json(progress_path())
    record = next(item for item in progress["records"] if item["request_id"] == args.request_id)
    if record["status"] != "PENDING":
        raise RuntimeError("request is not pending")
    record.update({
        "status": "TECHNICAL_FAILURE",
        "generation_call_id": args.generation_call_id,
        "started_at": args.started_at,
        "ended_at": args.ended_at,
        "failure_reason": args.reason,
    })
    progress["generation_calls"] += 1
    write_json(progress_path(), progress)
    print(json.dumps({"request_id": args.request_id, "status": "TECHNICAL_FAILURE"}, indent=2))


def flat_neutral_fraction(path: Path, bands: list[tuple[int, int]]) -> float:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    count = 0
    total = 0
    pixels = rgb.load()
    for y0, y1 in bands:
        for y in range(y0, y1):
            for x in range(rgb.width):
                r, g, b = pixels[x, y]
                total += 1
                if max(r, g, b) - min(r, g, b) <= 2 and 20 <= r <= 245:
                    count += 1
    return count / total if total else 0.0


def make_final_contact_sheet(records: list[dict[str, Any]], geometry: dict[str, dict[str, Any]]) -> Path:
    panel_width, row_height = 440, 430
    columns = ["ORIGINAL", "REGISTERED CROP", "DERIVED TARGET", "ATTEMPT 001", "V1 FAILED", "V2 OUTPUT", "OVERLAY", "SUBJECT DETAIL"]
    sheet = Image.new("RGB", (panel_width * len(columns), row_height * len(records)), (26, 29, 32))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for row, record in enumerate(records):
        item = geometry[record["request_id"]]
        y0 = row * row_height
        attempt_001_path = load_json(ATTEMPT_002 / "03_generation_requests" / "requests" / f"{record['request_id']}.json")["attempt_001_output_path"]
        v1_path = load_json(RISK / "subject00_portrait_canary_output_registry.json")["records"]
        v1_path = next(value["output_path"] for value in v1_path if value["request_id"] == record["request_id"])
        output = Path(record["output_path"]) if record.get("output_path") else None
        source = Path(item["source_path"])
        overlay = source
        detail = source
        paths = [source, Path(item["registered_crop_path"]), Path(item["derived_canvas_path"]), Path(attempt_001_path), Path(v1_path), output, overlay, detail]
        draw.text((8, y0 + 6), f"{record['request_id']} {item['garment_id']} {item['slot_id']} {item['camera_name']} crop={item['crop_window']['x_left']}:{item['crop_window']['x_right']} ext=193/193 ratio=NOT_AVAILABLE", fill="white", font=font)
        for col, (label, path) in enumerate(zip(columns, paths)):
            if path and path.is_file():
                with Image.open(path) as image:
                    preview = image.convert("RGB")
                if label == "SUBJECT DETAIL":
                    bbox = item["person_bbox_source"]
                    margin = 30
                    preview = preview.crop((max(0, bbox[0]-margin), max(0, bbox[1]-margin), min(preview.width, bbox[2]+margin), min(preview.height, bbox[3]+margin)))
                if label == "OVERLAY":
                    overlay_image = preview.copy()
                    od = ImageDraw.Draw(overlay_image)
                    xl, xr = item["crop_window"]["x_left"], item["crop_window"]["x_right"]
                    od.line((xl, 0, xl, overlay_image.height-1), fill=(255, 50, 50), width=5)
                    od.line((xr-1, 0, xr-1, overlay_image.height-1), fill=(255, 50, 50), width=5)
                    od.rectangle(tuple(item["person_bbox_source"]), outline=(0, 255, 80), width=5)
                    preview = overlay_image
                preview.thumbnail((panel_width - 12, row_height - 56), Image.Resampling.LANCZOS)
                px = col * panel_width + (panel_width - preview.width) // 2
                py = y0 + 28 + (row_height - 56 - preview.height) // 2
                sheet.paste(preview, (px, py))
            draw.text((col * panel_width + 6, y0 + row_height - 20), label, fill="white", font=font)
    path = DATA_ROOT / "07_human_review" / "subject00_portrait_canary_v2_contact_sheet.png"
    sheet.save(path, format="PNG", optimize=True)
    return path


def finalize() -> None:
    progress = load_json(progress_path())
    geometry_records = load_json(PREFLIGHT / "crop_geometry_manifest.json")["records"]
    geometry = {item["request_id"]: item for item in geometry_records}
    output_records = []
    for record in progress["records"]:
        item = dict(record)
        item["subject_height_ratio"] = "NOT_AVAILABLE"
        item["subject_center_displacement"] = "NOT_AVAILABLE"
        item["keypoint_drift"] = "NOT_AVAILABLE"
        item["person_mask_status"] = "NOT_AVAILABLE_FOR_GENERATED_OUTPUT"
        item["flat_neutral_extension_fraction"] = (
            flat_neutral_fraction(Path(item["output_path"]), [(0, TOP_EXTENSION), (TOP_EXTENSION + 1150, 1536)])
            if item.get("output_path") else None
        )
        output_records.append(item)
    output_count = sum(item.get("output_path") is not None for item in output_records)
    native_count = sum(item["status"] == "NATIVE_RESOLUTION_PASS" for item in output_records)
    parse_count = sum(item.get("format") == "PNG" for item in output_records)
    shas = [item["output_sha256"] for item in output_records if item.get("output_sha256")]
    duplicate_count = len(shas) - len(set(shas))
    if progress["phase_0_status"] != "REGISTRATION_PREFLIGHT_PASS":
        classification = BLOCKED_CLASSIFICATION
        next_task = BLOCKED_NEXT_TASK
    elif output_count == native_count == parse_count == 4 and duplicate_count == 0:
        classification = PASS_CLASSIFICATION
        next_task = PASS_NEXT_TASK
    else:
        classification = INCOMPLETE_CLASSIFICATION
        next_task = PASS_NEXT_TASK
    contact_sheet = make_final_contact_sheet(output_records, geometry)
    review_records = []
    for item in output_records:
        review = {
            "request_id": item["request_id"],
            "candidate_path": item.get("output_path"),
            "candidate_sha256": item.get("output_sha256"),
            "technical_status": item["status"],
            "subject_height_ratio": item["subject_height_ratio"],
            "subject_center_displacement": item["subject_center_displacement"],
            "keypoint_drift": item["keypoint_drift"],
            "visual_review_status": "PENDING_USER_REVIEW",
        }
        review.update({field: None for field in VISUAL_FIELDS})
        review_records.append(review)
    review_manifest = {
        "schema_version": "canondressgs.subject00.portrait_canary_v2_review_manifest.v1",
        "task_id": TASK_ID,
        "records": review_records,
        "visual_decision_count": 0,
        "accepted_count": 0,
        "teacher_target_count": 0,
    }
    review_path = DATA_ROOT / "07_human_review" / "subject00_portrait_canary_v2_review_manifest.json"
    write_json(review_path, review_manifest)
    output_registry = {
        "schema_version": "canondressgs.subject00.portrait_canary_v2_output_registry.v1",
        "task_id": TASK_ID,
        "record_count": 4,
        "records": output_records,
        "generation_calls": progress["generation_calls"],
        "technical_retries": 0,
        "output_count": output_count,
        "native_resolution_pass_count": native_count,
        "png_parse_pass_count": parse_count,
        "duplicate_sha_count": duplicate_count,
        "postprocessing_count": 0,
        "paper_final": PAPER_FINAL,
    }
    summary = {
        "schema_version": "canondressgs.subject00.portrait_canary_v2_final_summary.v1",
        "task_id": TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "branch": BRANCH,
        "attempt_root": str(DATA_ROOT),
        "protocol": PROTOCOL,
        "phase_0_status": progress["phase_0_status"],
        "generation_calls": progress["generation_calls"],
        "technical_retries": 0,
        "output_count": output_count,
        "native_resolution_pass_count": native_count,
        "png_parse_pass_count": parse_count,
        "duplicate_sha_count": duplicate_count,
        "postprocessing_count": 0,
        "external_api_calls": 0,
        "api_key_reads": 0,
        "cloud_image_writes": 0,
        "accepted_count": 0,
        "teacher_target_count": 0,
        "remaining_39_generated_count": 0,
        "formal_base": FORMAL_BASE,
        "paper_modifications": 0,
        "contact_sheet_path": str(contact_sheet),
        "review_manifest_path": str(review_path),
        "tests": "NOT_RUN",
        "classification": classification,
        "next_task": next_task,
        "next_task_authorized": False,
        "paper_final": PAPER_FINAL,
    }
    write_json(DATA_ROOT / "09_final_verification" / "subject00_portrait_canary_v2_output_registry.json", output_registry)
    write_json(DATA_ROOT / "09_final_verification" / "subject00_portrait_canary_v2_final_summary.json", summary)
    write_json(RISK / "subject00_portrait_canary_v2_registration_preflight.json", load_json(PREFLIGHT / "camera_binding_audit.json"))
    write_json(RISK / "subject00_portrait_canary_v2_output_registry.json", output_registry)
    write_json(RISK / "subject00_portrait_canary_v2_final_summary.json", summary)
    write_json(
        RISK / "subject00_portrait_canary_v2_review_contract.json",
        {
            "schema_version": "canondressgs.subject00.portrait_canary_v2_review_contract.v1",
            "task_id": TASK_ID,
            "contact_sheet_path": str(contact_sheet),
            "review_manifest_path": str(review_path),
            "visual_fields_initial_state": None,
            "visual_decision_count": 0,
            "automated_acceptance_allowed": False,
            "remaining_39_rerun_authorized": False,
            "paper_final": PAPER_FINAL,
        },
    )
    write_json(
        RISK / "subject00_portrait_canary_v2_tests.json",
        {"schema_version": "canondressgs.subject00.portrait_canary_v2_tests.v1", "task_id": TASK_ID, "status": "NOT_RUN", "total": 0, "passed": 0, "failed": [], "checks": [], "paper_final": PAPER_FINAL},
    )
    write_json(
        HANDOFF / "subject00_portrait_canary_v2_registered_outpaint_handoff.json",
        {
            "schema_version": "canondressgs.subject00.portrait_canary_v2_handoff.v1",
            "task_id": TASK_ID,
            "source_branch": SOURCE_BRANCH,
            "source_head": SOURCE_HEAD,
            "branch": BRANCH,
            "attempt_root": str(DATA_ROOT),
            "classification": classification,
            "phase_0_status": progress["phase_0_status"],
            "accepted_count": 0,
            "teacher_target_count": 0,
            "remaining_39_rerun_authorized": False,
            "formal_base": FORMAL_BASE,
            "tests": "NOT_RUN",
            "next_task": next_task,
            "next_task_authorized": False,
            "paper_final": PAPER_FINAL,
        },
    )
    write_doc(
        DOCS / "AAAI27_SUBJECT00_PORTRAIT_CANARY_V2_REGISTERED_OUTPAINT_REPORT_20260726.md",
        f"""
# Subject00 Portrait Canary V2 Registered Outpaint Report

- Task: `{TASK_ID}`
- Source: `{SOURCE_BRANCH}` at `{SOURCE_HEAD}`
- Protocol: `{PROTOCOL}`
- Phase 0: `{progress['phase_0_status']}`
- Generation calls / retries: `{progress['generation_calls']} / 0`
- Outputs / native PNG: `{output_count} / {native_count}`
- Duplicate SHA / postprocessing: `{duplicate_count} / 0`
- Accepted / Teacher targets: `0 / 0`
- Remaining 39 generated: `0`
- Formal Base: `PENDING`
- PAPER_FINAL: `false`
- Classification: `{classification}`
- Next task: `{next_task}`

Visual decision fields remain null. This report records engineering and native-resolution evidence only and does not claim Visual PASS.
""",
    )
    print(json.dumps({"classification": classification, "next_task": next_task, "output_count": output_count, "native_count": native_count, "contact_sheet": str(contact_sheet)}, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    sub.add_parser("next")
    success = sub.add_parser("record-success")
    success.add_argument("--request-id", required=True)
    success.add_argument("--generated-source", required=True)
    success.add_argument("--generation-call-id", required=True)
    success.add_argument("--started-at", required=True)
    success.add_argument("--ended-at", required=True)
    failure = sub.add_parser("record-failure")
    failure.add_argument("--request-id", required=True)
    failure.add_argument("--generation-call-id", required=True)
    failure.add_argument("--reason", required=True)
    failure.add_argument("--started-at", required=True)
    failure.add_argument("--ended-at", required=True)
    sub.add_parser("finalize")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.command == "preflight":
        preflight()
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
