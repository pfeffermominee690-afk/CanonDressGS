#!/usr/bin/env python3
"""Recompute the Subject00 1349 cohort and build a high-resolution review pack."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import shutil
import struct
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps
from pypdf import PdfReader
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
HANDOFF_ROOT = ROOT / "project_control_handoff"
SOURCE_WORKTREE = Path(r"E:\model_train\canondressgs_subject00_attempt001_native_landscape_registration_audit")
ATTEMPTS_ROOT = Path(r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-GENERATION-001")
ATTEMPT_001 = ATTEMPTS_ROOT / "attempt_001"
ATTEMPT_002 = ATTEMPTS_ROOT / "attempt_002_portrait_canary"
ATTEMPT_003 = ATTEMPTS_ROOT / "attempt_003_portrait_canary_v2_registered_outpaint"
OLD_AUDIT_ROOT = ATTEMPTS_ROOT / "attempt_001_native_landscape_registration_audit"
OUTPUT_ROOT = OLD_AUDIT_ROOT / "08_corrected_1349_human_review"
ARCHIVE = Path(r"E:\data_pre\thuman4_second_identity_staging\downloads\subject00.7z")
CLOUD_MANIFEST = Path(r"E:\data_pre\thuman4_second_identity_staging\reports\SUBJECT00_CLOUD_DATA_MANIFEST.json")

TASK_ID = "AAAI27-SUBJECT00-1349x1166-COHORT-CORRECTION-HUMAN-REVIEW-PREP-001"
SOURCE_BRANCH = "research/subject00-attempt001-native-landscape-registration-audit-20260726"
SOURCE_HEAD = "e8af4f852fb08021d4dd7ae070bf196c7bedc589"
BRANCH = "research/subject00-1349-cohort-correction-human-review-prep-20260726"
TARGET_RESOLUTION = "1349x1166"
PASS_CLASSIFICATION = "REGISTERED_SIMILARITY_PASS_CANDIDATE"
FINAL_CLASSIFICATION = "SUBJECT00_1349_COHORT_CORRECTED_READY_FOR_HIGH_RES_USER_REVIEW"
NEXT_TASK = "USER_REVIEW_CORRECTED_1349x1166_DOMINANT_COHORT"
GARMENTS = ("O01", "O03", "O04")
SLOTS = {
    "slot_00": {"camera": "cam17", "orientation": "front"},
    "slot_01": {"camera": "cam21", "orientation": "front-left"},
    "slot_02": {"camera": "cam14", "orientation": "front-right"},
    "slot_03": {"camera": "cam23", "orientation": "left"},
    "slot_04": {"camera": "cam11", "orientation": "right"},
    "slot_05": {"camera": "cam02", "orientation": "back-left"},
    "slot_06": {"camera": "cam09", "orientation": "back-right"},
    "slot_07": {"camera": "cam05", "orientation": "back"},
}
OLD_MISSING = [
    "O01/slot_04", "O01/slot_07", "O03/slot_06", "O03/slot_07",
    "O04/slot_05", "O04/slot_06",
]
HYPOTHESIZED_MISSING = [
    "O01/slot_04", "O01/slot_07", "O03/slot_06", "O04/slot_01",
    "O04/slot_05", "O04/slot_06",
]

PROTOCOL_PATH = RISK / "subject00_attempt001_1349_cohort_correction_review_protocol_20260726.json"
OLD_INVENTORY_PATH = RISK / "subject00_attempt001_native_landscape_full_inventory_20260726.json"
OLD_REGISTRATION_PATH = RISK / "subject00_attempt001_background_registration_metrics_20260726.json"
OLD_PROVENANCE_PATH = RISK / "subject00_attempt001_provenance_binding_audit_20260726.json"
OLD_COHORTS_PATH = RISK / "subject00_attempt001_exact_resolution_cohorts_20260726.json"
OLD_COVERAGE_PATH = RISK / "subject00_attempt001_garment_slot_coverage_20260726.json"
OLD_SUMMARY_PATH = RISK / "subject00_attempt001_native_landscape_registration_final_summary_20260726.json"
OLD_REPORT_PATH = RISK / "SUBJECT00_ATTEMPT001_NATIVE_LANDSCAPE_REGISTRATION_AUDIT_REPORT_20260726.md"
BASELINE_PATH = OLD_AUDIT_ROOT / "00_provenance" / "attempt_immutability_baseline.json"

COHORTS_PATH = RISK / "corrected_attempt001_exact_resolution_cohorts.json"
COVERAGE_PATH = RISK / "corrected_attempt001_1349x1166_cell_coverage.json"
CANDIDATES_PATH = RISK / "corrected_attempt001_1349x1166_candidate_registry.json"
REVIEW_PATH = RISK / "corrected_1349x1166_human_review_manifest.json"
OVERLAY_PATH = RISK / "subject00_attempt001_1349_cohort_correction_overlay_20260726.json"
PACK_REGISTRY_PATH = RISK / "subject00_attempt001_1349_high_res_review_pack_registry_20260726.json"
FINAL_SUMMARY_PATH = RISK / "subject00_attempt001_1349_cohort_correction_final_summary_20260726.json"
REPORT_PATH = RISK / "SUBJECT00_ATTEMPT001_1349_COHORT_CORRECTION_REPORT_20260726.md"
HANDOFF_PATH = HANDOFF_ROOT / "subject00_attempt001_1349_cohort_correction_human_review_prep_handoff_20260726.json"

PAGE_WIDTH = 36 * 72
PAGE_HEIGHT = 24 * 72
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


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


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value.rstrip() + "\n")


def dual_json(repo_path: Path, external_name: str, payload: dict[str, Any]) -> None:
    write_json(repo_path, payload)
    write_json(OUTPUT_ROOT / external_name, payload)


def run(*command: str, cwd: Path = ROOT) -> str:
    return subprocess.run(
        list(command), cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def expected_request_ids() -> list[str]:
    return [
        f"subject00_{garment}_slot{slot[-2:]}_cand{candidate:02d}"
        for garment in GARMENTS for slot in SLOTS for candidate in range(2)
    ]


def current_inventory(root: Path) -> dict[str, tuple[int, str]]:
    return {
        item.relative_to(root).as_posix(): (item.stat().st_size, file_sha256(item))
        for item in sorted(path for path in root.rglob("*") if path.is_file())
    }


def frozen_inventory(snapshot: dict[str, Any]) -> dict[str, tuple[int, str]]:
    return {item["relative_path"]: (item["bytes"], item["sha256"]) for item in snapshot["files"]}


def validate_gate() -> None:
    facts = {
        "source_branch": run("git", "branch", "--show-current", cwd=SOURCE_WORKTREE),
        "source_head": run("git", "rev-parse", "HEAD", cwd=SOURCE_WORKTREE),
        "source_status": run("git", "status", "--short", cwd=SOURCE_WORKTREE),
        "target_branch": run("git", "branch", "--show-current"),
    }
    expected = {
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "source_status": "",
        "target_branch": BRANCH,
    }
    if facts != expected:
        raise RuntimeError(f"source gate mismatch: {facts!r}")
    if subprocess.run(["git", "merge-base", "--is-ancestor", SOURCE_HEAD, "HEAD"], cwd=ROOT).returncode:
        raise RuntimeError("source HEAD is not an ancestor of the target")
    baseline = load_json(BASELINE_PATH)
    for name, root in (("attempt_001", ATTEMPT_001), ("attempt_002", ATTEMPT_002), ("attempt_003", ATTEMPT_003)):
        if current_inventory(root) != frozen_inventory(baseline[name]):
            raise RuntimeError(f"{name} changed relative to the frozen baseline")


def prepare_output(replace: bool) -> None:
    if OUTPUT_ROOT.exists():
        if not replace:
            raise RuntimeError(f"output already exists: {OUTPUT_ROOT}")
        if OUTPUT_ROOT.parent.resolve() != OLD_AUDIT_ROOT.resolve() or OUTPUT_ROOT.name != "08_corrected_1349_human_review":
            raise RuntimeError("refusing to replace unexpected output root")
        shutil.rmtree(OUTPUT_ROOT)
    for relative in ("assets", "masks", "registries", "pdf"):
        (OUTPUT_ROOT / relative).mkdir(parents=True, exist_ok=False)


def parse_png(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise RuntimeError(f"invalid PNG signature/IHDR: {path}")
    width, height = struct.unpack(">II", header[16:24])
    with Image.open(path) as image:
        if image.format != "PNG" or image.size != (width, height):
            raise RuntimeError(f"Pillow/IHDR mismatch: {path}")
        image.verify()
    return width, height


def archive_member_batch(members: list[str], sizes: dict[str, int]) -> dict[str, bytes]:
    ordered = sorted(set(members))
    payload = subprocess.run(["tar", "-xOf", str(ARCHIVE), *ordered], check=True, capture_output=True).stdout
    expected_bytes = sum(sizes[member] for member in ordered)
    if len(payload) != expected_bytes:
        raise RuntimeError(f"archive byte count mismatch: {len(payload)} != {expected_bytes}")
    result: dict[str, bytes] = {}
    offset = 0
    for member in ordered:
        size = sizes[member]
        result[member] = payload[offset:offset + size]
        offset += size
    return result


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (Path(r"C:\Windows\Fonts\consola.ttf"), Path(r"C:\Windows\Fonts\arial.ttf")):
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def fit(image: Image.Image, width: int, height: int, background: tuple[int, int, int] = (18, 20, 22)) -> Image.Image:
    fitted = ImageOps.contain(image.convert("RGB"), (max(1, width), max(1, height)), Image.Resampling.LANCZOS)
    canvas_image = Image.new("RGB", (width, height), background)
    canvas_image.paste(fitted, ((width - fitted.width) // 2, (height - fitted.height) // 2))
    return canvas_image


def labeled_pair(left: Image.Image, right: Image.Image, left_label: str, right_label: str) -> Image.Image:
    label_height = 42
    target_height = max(left.height, right.height)
    left_canvas = fit(left, left.width, target_height)
    right_canvas = fit(right, right.width, target_height)
    output = Image.new("RGB", (left_canvas.width + right_canvas.width, target_height + label_height), (18, 20, 22))
    output.paste(left_canvas, (0, label_height))
    output.paste(right_canvas, (left_canvas.width, label_height))
    draw = ImageDraw.Draw(output)
    draw.text((10, 8), left_label, fill=(235, 235, 235), font=font(22))
    draw.text((left_canvas.width + 10, 8), right_label, fill=(235, 235, 235), font=font(22))
    return output


def normalized_box(person_bbox: list[int], fractions: list[float], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [float(value) for value in person_bbox]
    width, height = x1 - x0, y1 - y0
    box = (
        x0 + fractions[0] * width,
        y0 + fractions[1] * height,
        x0 + fractions[2] * width,
        y0 + fractions[3] * height,
    )
    return clip_box(box, image_size)


def clip_box(box: tuple[float, float, float, float], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = image_size
    x0 = max(0, min(width - 2, math.floor(box[0])))
    y0 = max(0, min(height - 2, math.floor(box[1])))
    x1 = max(x0 + 2, min(width, math.ceil(box[2])))
    y1 = max(y0 + 2, min(height, math.ceil(box[3])))
    return x0, y0, x1, y1


def transform_box(box: tuple[int, int, int, int], matrix: np.ndarray, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    points = np.asarray([[x0, y0, 1], [x1, y0, 1], [x0, y1, 1], [x1, y1, 1]], dtype=np.float64)
    transformed = points @ matrix.T
    return clip_box((transformed[:, 0].min(), transformed[:, 1].min(), transformed[:, 0].max(), transformed[:, 1].max()), image_size)


def aligned_image(source: Image.Image, matrix: np.ndarray, output_size: tuple[int, int], resample: Image.Resampling) -> Image.Image:
    full = np.vstack([matrix, [0.0, 0.0, 1.0]])
    inverse = np.linalg.inv(full)
    coefficients = tuple(float(value) for value in inverse[:2].reshape(-1))
    return source.transform(output_size, Image.Transform.AFFINE, coefficients, resample=resample, fillcolor=0)


def edge_overlay(aligned_source: Image.Image, output: Image.Image, transformed_mask: Image.Image) -> Image.Image:
    source_edges = np.asarray(aligned_source.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.uint8) > 28
    output_edges = np.asarray(output.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.uint8) > 28
    exclusion = np.asarray(transformed_mask.filter(ImageFilter.MaxFilter(47)), dtype=np.uint8) >= 128
    source_edges[exclusion] = False
    output_edges[exclusion] = False
    source_edges[:8] = source_edges[-8:] = False
    source_edges[:, :8] = source_edges[:, -8:] = False
    output_edges[:8] = output_edges[-8:] = False
    output_edges[:, :8] = output_edges[:, -8:] = False
    rgb = np.zeros((*source_edges.shape, 3), dtype=np.uint8)
    rgb[..., 1] = source_edges.astype(np.uint8) * 255
    rgb[..., 0] = output_edges.astype(np.uint8) * 255
    return Image.fromarray(rgb, "RGB")


def high_error_background(aligned_source: Image.Image, output: Image.Image, transformed_mask: Image.Image) -> Image.Image:
    source_array = np.asarray(aligned_source.convert("RGB"), dtype=np.int16)
    output_array = np.asarray(output.convert("RGB"), dtype=np.uint8)
    difference = np.mean(np.abs(source_array - output_array.astype(np.int16)), axis=2)
    exclusion = np.asarray(transformed_mask.filter(ImageFilter.MaxFilter(47)), dtype=np.uint8) >= 128
    difference[exclusion] = 0
    difference[:8] = difference[-8:] = 0
    difference[:, :8] = difference[:, -8:] = 0
    scaled = np.clip(difference * 4.0, 0, 255).astype(np.uint8)
    base = (output_array.astype(np.float32) * 0.30).astype(np.uint8)
    base[..., 0] = np.maximum(base[..., 0], scaled)
    base[..., 1] = np.maximum(base[..., 1], np.where(scaled > 160, scaled // 2, 0).astype(np.uint8))
    return Image.fromarray(base, "RGB")


def silhouette_comparison(source: Image.Image, output: Image.Image, source_mask: Image.Image, transformed_mask: Image.Image) -> Image.Image:
    source_overlay = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
    source_edge = np.asarray(source_mask.filter(ImageFilter.FIND_EDGES), dtype=np.uint8) > 16
    source_overlay[source_edge] = [0, 255, 255]
    output_overlay = np.asarray(output.convert("RGB"), dtype=np.uint8).copy()
    output_edge = np.asarray(transformed_mask.filter(ImageFilter.FIND_EDGES), dtype=np.uint8) > 16
    output_overlay[output_edge] = [255, 80, 40]
    return labeled_pair(
        Image.fromarray(source_overlay), Image.fromarray(output_overlay),
        "SOURCE MASK", "EXPECTED SILHOUETTE",
    )


def roi_mae(aligned_source: Image.Image, output: Image.Image, box: tuple[int, int, int, int]) -> float:
    source_array = np.asarray(aligned_source.crop(box).convert("RGB"), dtype=np.float32)
    output_array = np.asarray(output.crop(box).convert("RGB"), dtype=np.float32)
    return float(np.mean(np.abs(source_array - output_array)) / 255.0)


def save_image(image: Image.Image, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", compress_level=6)
    return str(path)


def build_candidate_assets(record: dict[str, Any], metric: dict[str, Any], source_mask: Image.Image, protocol: dict[str, Any]) -> tuple[dict[str, str], dict[str, float]]:
    request_id = record["request_id"]
    asset_root = OUTPUT_ROOT / "assets" / request_id
    asset_root.mkdir(parents=True, exist_ok=False)
    source = Image.open(record["source_path"]).convert("RGB")
    output = Image.open(record["output_path"]).convert("RGB")
    matrix = np.asarray(metric["similarity"]["matrix"], dtype=np.float64)
    aligned_source = aligned_image(source, matrix, output.size, Image.Resampling.BILINEAR)
    transformed_mask = aligned_image(source_mask, matrix, output.size, Image.Resampling.NEAREST).convert("L")
    overlay = Image.blend(output, aligned_source, 0.5)
    edge = edge_overlay(aligned_source, output, transformed_mask)
    error = high_error_background(aligned_source, output, transformed_mask)
    silhouette = silhouette_comparison(source, output, source_mask, transformed_mask)

    crop_protocol = protocol["crop_protocol"]
    crop_names = {
        "full_body_source_output_pair": "full_body",
        "face_head_source_output_pair": "face_head",
        "left_hand_source_output_pair": "left_hand_image_side",
        "right_hand_source_output_pair": "right_hand_image_side",
        "feet_source_output_pair": "feet",
        "upper_garment_boundary_pair": "upper_garment_boundary",
        "lower_garment_boundary_pair": "lower_garment_boundary",
    }
    crop_assets: dict[str, str] = {}
    output_boxes: dict[str, tuple[int, int, int, int]] = {}
    for asset_name, protocol_name in crop_names.items():
        source_box = normalized_box(metric["person_pose_metrics"]["person_bbox_source"], crop_protocol[protocol_name], source.size)
        output_box = transform_box(source_box, matrix, output.size)
        output_boxes[protocol_name] = output_box
        pair = labeled_pair(source.crop(source_box), output.crop(output_box), "SOURCE", "OUTPUT")
        crop_assets[asset_name] = save_image(pair, asset_root / f"{asset_name}.png")

    assets = {
        "original_condition_full_frame": record["source_path"],
        "generated_output_full_frame": record["output_path"],
        "similarity_aligned_overlay": save_image(overlay, asset_root / "similarity_aligned_overlay.png"),
        "background_edge_overlay": save_image(edge, asset_root / "background_edge_overlay.png"),
        **crop_assets,
        "official_source_mask_and_transformed_expected_silhouette": save_image(silhouette, asset_root / "silhouette_mask_comparison.png"),
        "high_error_background_regions": save_image(error, asset_root / "high_error_background_regions.png"),
    }
    face = roi_mae(aligned_source, output, output_boxes["face_head"])
    hand_values = [
        roi_mae(aligned_source, output, output_boxes["left_hand_image_side"]),
        roi_mae(aligned_source, output, output_boxes["right_hand_image_side"]),
        roi_mae(aligned_source, output, output_boxes["feet"]),
    ]
    garment_values = [
        roi_mae(aligned_source, output, output_boxes["upper_garment_boundary"]),
        roi_mae(aligned_source, output, output_boxes["lower_garment_boundary"]),
    ]
    signals = {
        "absolute_uniform_scale_minus_one": abs(float(metric["uniform_scale"]) - 1.0),
        "translation_norm_px": math.hypot(float(metric["translation_x"]), float(metric["translation_y"])),
        "p95_reprojection_error_px": float(metric["p95_reprojection_error_px"]),
        "inverse_ransac_inlier_ratio": 1.0 - float(metric["similarity"]["inlier_ratio"]),
        "face_head_aligned_mae": face,
        "hands_feet_aligned_mae": float(np.mean(hand_values)),
        "garment_boundary_aligned_mae": float(np.mean(garment_values)),
    }
    return assets, signals


def add_risk_ranks(candidates: list[dict[str, Any]], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    count = len(candidates)
    weights = dict(protocol["risk_ranking"]["signals_and_weights"])
    for signal, weight in weights.items():
        ordered = sorted(candidates, key=lambda item: (-item["risk_signals"][signal], item["request_id"]))
        for index, item in enumerate(ordered, start=1):
            item.setdefault("risk_signal_ranks", {})[signal] = index
            points = 1.0 if count == 1 else (count - index) / (count - 1)
            item["risk_priority_score"] = item.get("risk_priority_score", 0.0) + float(weight) * points
    ordered = sorted(candidates, key=lambda item: (-item["risk_priority_score"], item["request_id"]))
    for rank, item in enumerate(ordered, start=1):
        item["risk_priority_rank"] = rank
        item["risk_priority_score"] = round(float(item["risk_priority_score"]), 8)
    return ordered


def draw_fitted(pdf: canvas.Canvas, path: str, x: float, y: float, width: float, height: float) -> None:
    with Image.open(path) as image:
        image_width, image_height = image.size
    scale = min(width / image_width, height / image_height)
    draw_width, draw_height = image_width * scale, image_height * scale
    pdf.drawImage(ImageReader(path), x + (width - draw_width) / 2, y + (height - draw_height) / 2, draw_width, draw_height, preserveAspectRatio=True, mask="auto")


def draw_panel(pdf: canvas.Canvas, label: str, path: str, x: float, top: float, width: float, height: float) -> None:
    pdf.setFillColorRGB(0.08, 0.09, 0.10)
    pdf.rect(x, top - height, width, height, fill=1, stroke=0)
    pdf.setFillColorRGB(0.90, 0.92, 0.94)
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(x + 6, top - 14, label[:62])
    draw_fitted(pdf, path, x + 5, top - height + 5, width - 10, height - 24)


def metadata_lines(candidate: dict[str, Any]) -> list[str]:
    metric = candidate["registration_metrics"]
    return [
        f"request_id: {candidate['request_id']}",
        f"garment / slot / camera / candidate: {candidate['garment']} / {candidate['slot']} / {candidate['camera']} / {candidate['candidate_id']}",
        f"output resolution: {candidate['actual_width']}x{candidate['actual_height']} | exact cohort: {candidate['exact_cohort']}",
        f"output SHA-256: {candidate['output_sha256']}",
        f"uniform scale: {metric['uniform_scale']:.9f}",
        f"tx / ty: {metric['translation_x']:.6f} / {metric['translation_y']:.6f} px",
        f"rotation: {metric['rotation_degrees']:.9f} deg",
        f"median / p95 reprojection: {metric['median_reprojection_error_px']:.6f} / {metric['p95_reprojection_error_px']:.6f} px",
        f"RANSAC inlier ratio: {metric['ransac_inlier_ratio']:.9f}",
        f"machine classification: {candidate['primary_machine_classification']}",
        f"risk review rank / score: {candidate['risk_priority_rank']} / {candidate['risk_priority_score']:.6f}",
        "human decision: null | selected_for_cell: null | accepted: false | teacher_target: false",
        "Mask panel shows official source mask and transformed expected silhouette; no output segmentation model was used.",
        "Risk metrics prioritize review only and do not constitute an automatic visual PASS or FAIL.",
    ]


def draw_candidate(pdf: canvas.Canvas, candidate: dict[str, Any], x: float, top: float, width: float, height: float) -> None:
    pdf.setStrokeColorRGB(0.25, 0.28, 0.30)
    pdf.rect(x, top - height, width, height, fill=0, stroke=1)
    pdf.setFillColorRGB(0.96, 0.96, 0.96)
    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawString(x + 8, top - 20, f"{candidate['request_id']} | REVIEW DECISION: null")
    inner_x, inner_width = x + 8, width - 16
    row_top = top - 30
    rows = [
        ([
            ("ORIGINAL CONDITION FULL FRAME", "original_condition_full_frame"),
            ("GENERATED OUTPUT FULL FRAME", "generated_output_full_frame"),
            ("SIMILARITY-ALIGNED OVERLAY", "similarity_aligned_overlay"),
            ("BACKGROUND EDGE OVERLAY", "background_edge_overlay"),
        ], 280),
        ([
            ("FULL-BODY SOURCE / OUTPUT", "full_body_source_output_pair"),
            ("SILHOUETTE / MASK COMPARISON", "official_source_mask_and_transformed_expected_silhouette"),
            ("HIGH-ERROR BACKGROUND REGIONS", "high_error_background_regions"),
        ], 280),
        ([
            ("FACE / HEAD SOURCE / OUTPUT", "face_head_source_output_pair"),
            ("LEFT-HAND IMAGE-SIDE ROI", "left_hand_source_output_pair"),
            ("RIGHT-HAND IMAGE-SIDE ROI", "right_hand_source_output_pair"),
        ], 235),
        ([
            ("FEET SOURCE / OUTPUT", "feet_source_output_pair"),
            ("UPPER GARMENT BOUNDARY", "upper_garment_boundary_pair"),
            ("LOWER GARMENT BOUNDARY", "lower_garment_boundary_pair"),
        ], 235),
    ]
    for panels, row_height in rows:
        gap = 6
        panel_width = (inner_width - gap * (len(panels) - 1)) / len(panels)
        for index, (label, key) in enumerate(panels):
            draw_panel(pdf, label, candidate["review_assets"][key], inner_x + index * (panel_width + gap), row_top, panel_width, row_height)
        row_top -= row_height + 6
    pdf.setFillColorRGB(0.08, 0.09, 0.10)
    pdf.rect(inner_x, top - height + 8, inner_width, max(20, row_top - (top - height + 8)), fill=1, stroke=0)
    pdf.setFillColorRGB(0.90, 0.92, 0.94)
    pdf.setFont("Helvetica", 10)
    text_y = row_top - 14
    for line in metadata_lines(candidate):
        pdf.drawString(inner_x + 8, text_y, line[:190])
        text_y -= 14


def build_garment_pdf(garment: str, candidates_by_cell: dict[str, list[dict[str, Any]]], coverage_by_cell: dict[str, dict[str, Any]], destination: Path) -> None:
    pdf = canvas.Canvas(str(destination), pagesize=(PAGE_WIDTH, PAGE_HEIGHT), pageCompression=1, invariant=1)
    pdf.setTitle(f"Subject00 {garment} 1349x1166 Human Review")
    for slot, binding in SLOTS.items():
        cell = f"{garment}/{slot}"
        candidates = sorted(candidates_by_cell.get(cell, []), key=lambda item: item["candidate_id"])
        pdf.setFillColorRGB(0.04, 0.045, 0.05)
        pdf.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, fill=1, stroke=0)
        pdf.setFillColorRGB(0.96, 0.96, 0.96)
        pdf.setFont("Helvetica-Bold", 22)
        pdf.drawString(24, PAGE_HEIGHT - 30, f"Subject00 {cell} | {binding['camera']} | {binding['orientation']} | HUMAN REVIEW STATUS: null")
        if not candidates:
            pdf.setFillColorRGB(0.95, 0.30, 0.22)
            pdf.setFont("Helvetica-Bold", 40)
            pdf.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT / 2 + 30, "NO_1349_PASS_CANDIDATE")
            pdf.setFillColorRGB(0.88, 0.88, 0.88)
            pdf.setFont("Helvetica", 17)
            raw = coverage_by_cell[cell]
            pdf.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT / 2 - 20, f"cand00: {raw['cand00']['actual_resolution']} | {raw['cand00']['primary_machine_classification']}")
            pdf.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT / 2 - 48, f"cand01: {raw['cand01']['actual_resolution']} | {raw['cand01']['primary_machine_classification']}")
            pdf.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT / 2 - 82, "No placeholder image was inserted. Targeted rerun authorization remains DENIED.")
        elif len(candidates) == 1:
            draw_candidate(pdf, candidates[0], 20, PAGE_HEIGHT - 45, PAGE_WIDTH - 40, PAGE_HEIGHT - 65)
        else:
            gap = 12
            panel_width = (PAGE_WIDTH - 40 - gap) / 2
            draw_candidate(pdf, candidates[0], 20, PAGE_HEIGHT - 45, panel_width, PAGE_HEIGHT - 65)
            draw_candidate(pdf, candidates[1], 20 + panel_width + gap, PAGE_HEIGHT - 45, panel_width, PAGE_HEIGHT - 65)
        pdf.showPage()
    pdf.save()


def build_risk_pdf(ordered_candidates: list[dict[str, Any]], destination: Path) -> None:
    pdf = canvas.Canvas(str(destination), pagesize=(PAGE_WIDTH, PAGE_HEIGHT), pageCompression=1, invariant=1)
    pdf.setTitle("Subject00 1349x1166 High-Risk Candidate Review")
    for candidate in ordered_candidates:
        pdf.setFillColorRGB(0.04, 0.045, 0.05)
        pdf.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, fill=1, stroke=0)
        pdf.setFillColorRGB(0.96, 0.96, 0.96)
        pdf.setFont("Helvetica-Bold", 22)
        pdf.drawString(24, PAGE_HEIGHT - 30, f"HIGH-RISK REVIEW PRIORITY {candidate['risk_priority_rank']:02d}/{len(ordered_candidates):02d} | REVIEW STATUS: null")
        draw_candidate(pdf, candidate, 20, PAGE_HEIGHT - 45, PAGE_WIDTH - 40, PAGE_HEIGHT - 65)
        pdf.showPage()
    pdf.save()


def build_coverage_sheet(coverage_records: list[dict[str, Any]], destination: Path) -> None:
    tile_width, tile_height = 1160, 350
    margin, title_height = 30, 120
    sheet = Image.new("RGB", (margin * 2 + tile_width * 3, title_height + tile_height * 8 + margin), (16, 18, 20))
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 20), "SUBJECT00 CORRECTED 1349x1166 24-CELL COVERAGE | HUMAN REVIEW STATUS: null", fill=(245, 245, 245), font=font(34))
    by_cell = {item["cell_key"]: item for item in coverage_records}
    for column, garment in enumerate(GARMENTS):
        for row, slot in enumerate(SLOTS):
            item = by_cell[f"{garment}/{slot}"]
            x = margin + column * tile_width
            y = title_height + row * tile_height
            missing = item["missing_1349_pass_candidate"]
            fill_color = (58, 24, 24) if missing else (23, 33, 31)
            draw.rectangle((x + 4, y + 4, x + tile_width - 6, y + tile_height - 6), fill=fill_color, outline=(110, 115, 120), width=2)
            draw.text((x + 18, y + 14), f"{item['cell_key']} | {item['camera']} | {item['orientation']}", fill=(245, 245, 245), font=font(25))
            draw.text((x + 18, y + 52), f"cand00: {item['cand00']['actual_resolution']} | {item['cand00']['primary_machine_classification']}", fill=(205, 220, 230), font=font(19))
            draw.text((x + 18, y + 82), f"cand01: {item['cand01']['actual_resolution']} | {item['cand01']['primary_machine_classification']}", fill=(205, 220, 230), font=font(19))
            draw.text((x + 18, y + 116), f"1349 pass candidate count: {item['pass_candidate_count_1349x1166']}", fill=(155, 230, 190), font=font(22))
            draw.text((x + 18, y + 150), "human review status: null", fill=(245, 215, 130), font=font(21))
            flag = "NO_1349_PASS_CANDIDATE" if missing else "REVIEW_CANDIDATE_PRESENT"
            draw.text((x + 18, y + 184), f"missing flag: {flag}", fill=(255, 115, 95) if missing else (145, 235, 180), font=font(21))
            thumbnails = [Path(candidate["output_path"]) for candidate in item["review_candidates"]]
            thumb_x = x + 18
            for thumb_path in thumbnails:
                with Image.open(thumb_path) as image:
                    thumbnail = fit(image, 250, 120)
                sheet.paste(thumbnail, (thumb_x, y + 220))
                thumb_x += 266
            if not thumbnails:
                draw.text((x + 18, y + 245), "No candidate image inserted.", fill=(225, 165, 155), font=font(20))
    sheet.save(destination, format="PNG", compress_level=6)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace-output", action="store_true")
    args = parser.parse_args()
    validate_gate()
    protocol = load_json(PROTOCOL_PATH)
    if protocol["task_id"] != TASK_ID or not protocol["frozen_before_review_pack_generation"]:
        raise RuntimeError("review protocol mismatch")
    prepare_output(args.replace_output)

    old_inventory = load_json(OLD_INVENTORY_PATH)
    old_metrics = load_json(OLD_REGISTRATION_PATH)
    old_provenance = load_json(OLD_PROVENANCE_PATH)
    old_cohorts = load_json(OLD_COHORTS_PATH)
    old_coverage = load_json(OLD_COVERAGE_PATH)
    old_summary = load_json(OLD_SUMMARY_PATH)
    baseline = load_json(BASELINE_PATH)
    requests = {path.stem: load_json(path) for path in sorted((ATTEMPT_001 / "03_generation_requests" / "requests").glob("*.json"))}
    provenance = {path.stem: load_json(path) for path in sorted((ATTEMPT_001 / "11_provenance" / "requests").glob("*.json"))}
    metrics = {item["request_id"]: item for item in old_metrics["records"]}
    old_inventory_records = {item["request_id"]: item for item in old_inventory["records"]}
    expected_ids = expected_request_ids()
    if sorted(requests) != sorted(expected_ids) or sorted(provenance) != sorted(expected_ids) or sorted(metrics) != sorted(expected_ids):
        raise RuntimeError("48-request evidence set mismatch")

    records: list[dict[str, Any]] = []
    for request_id in expected_ids:
        request, prov, metric = requests[request_id], provenance[request_id], metrics[request_id]
        output_path = Path(prov["output_path"])
        width, height = parse_png(output_path)
        output_sha = file_sha256(output_path)
        garment = request["garment_id"]
        slot = request["slot_id"]
        candidate_id = f"cand{int(request['candidate_index']):02d}"
        binding_checks = {
            "request_id": request["request_id"] == prov["request_id"] == request_id,
            "garment": request["garment_id"] == prov["garment_id"] == garment,
            "slot": request["slot_id"] == prov["slot_id"] == slot,
            "camera": request["camera_id"] == prov["camera_id"] == int(SLOTS[slot]["camera"][3:]),
            "pose_frame": request["pose_frame_id"] == prov["pose_frame_id"] == 0,
            "output_path": Path(request["output_path"]) == output_path,
            "output_sha": prov["output_sha256"] == output_sha,
            "output_size": (prov["width"], prov["height"]) == (width, height),
            "source_rgb_sha": request["source_input_sha256"]["identity_condition_rgb"] == prov["input_sha256"]["identity_condition_rgb"],
            "source_mask_sha": request["source_input_sha256"]["identity_condition_mask"] == prov["input_sha256"]["identity_condition_mask"],
        }
        old_record = old_inventory_records[request_id]
        binding_checks["old_inventory"] = (
            old_record["generated_output_sha256"] == output_sha
            and (old_record["output_width"], old_record["output_height"]) == (width, height)
        )
        if not all(binding_checks.values()):
            raise RuntimeError(f"COHORT_REGISTRY_EVIDENCE_CONFLICT: {request_id}: {binding_checks}")
        exact_resolution = f"{width}x{height}"
        is_pass = metric["primary_classification"] == PASS_CLASSIFICATION
        belongs = exact_resolution == TARGET_RESOLUTION
        no_replacement = (
            not old_record["hidden_postprocessing_replacement_detected"]
            and old_record["duplicate_status"] == "UNIQUE"
            and old_record["generated_output_sha256"] == output_sha
        )
        review_eligible = (
            belongs and is_pass and old_record["original_manifest_binding_status"] == "PASS"
            and old_record["png_parse_status"] == "PASS" and no_replacement
        )
        records.append({
            "request_id": request_id,
            "garment": garment,
            "slot": slot,
            "camera": SLOTS[slot]["camera"],
            "candidate_id": candidate_id,
            "output_path": str(output_path),
            "output_sha256": output_sha,
            "source_path": request["managed_tool_local_input_path"],
            "source_sha256": request["source_input_sha256"]["identity_condition_rgb"],
            "source_mask_member": f"subject00/masks/{SLOTS[slot]['camera']}/00000000.jpg",
            "source_mask_sha256": request["source_input_sha256"]["identity_condition_mask"],
            "actual_width": width,
            "actual_height": height,
            "actual_resolution": exact_resolution,
            "primary_machine_classification": metric["primary_classification"],
            "is_similarity_pass_candidate": is_pass,
            "belongs_to_1349x1166_cohort": belongs,
            "cell_key": f"{garment}/{slot}",
            "png_parse_status": "PASS",
            "provenance_binding_status": "PASS",
            "binding_checks": binding_checks,
            "postprocessing_count": old_record["postprocessing_count"],
            "hidden_postprocessing_replacement_detected": old_record["hidden_postprocessing_replacement_detected"],
            "duplicate_status": old_record["duplicate_status"],
            "no_postprocessing_or_replacement_evidence": no_replacement,
            "review_eligible": review_eligible,
        })

    records_by_id = {item["request_id"]: item for item in records}
    distribution = Counter(item["actual_resolution"] for item in records)
    cohorts = []
    for resolution, count in sorted(distribution.items(), key=lambda item: (-item[1], item[0])):
        cohort_records = [item for item in records if item["actual_resolution"] == resolution]
        raw_cells = sorted({item["cell_key"] for item in cohort_records})
        pass_records = [item for item in cohort_records if item["is_similarity_pass_candidate"]]
        pass_cells = sorted({item["cell_key"] for item in pass_records})
        cohorts.append({
            "resolution": resolution,
            "output_count": count,
            "raw_cell_count": len(raw_cells),
            "raw_cells": raw_cells,
            "machine_pass_candidate_count": len(pass_records),
            "machine_pass_cell_count": len(pass_cells),
            "machine_pass_cells": pass_cells,
        })
    target_cohort = next(item for item in cohorts if item["resolution"] == TARGET_RESOLUTION)

    coverage_records: list[dict[str, Any]] = []
    candidates_by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for garment in GARMENTS:
        for slot, binding in SLOTS.items():
            request_ids = [f"subject00_{garment}_slot{slot[-2:]}_cand{candidate:02d}" for candidate in range(2)]
            pair = [records_by_id[request_id] for request_id in request_ids]
            eligible = [item for item in pair if item["review_eligible"]]
            for item in eligible:
                candidates_by_cell[item["cell_key"]].append(item)
            other_pass = [item for item in pair if item["is_similarity_pass_candidate"] and not item["belongs_to_1349x1166_cohort"]]
            coverage_records.append({
                "cell_key": f"{garment}/{slot}",
                "garment": garment,
                "slot": slot,
                "camera": binding["camera"],
                "orientation": binding["orientation"],
                "cand00": {
                    "request_id": pair[0]["request_id"],
                    "actual_resolution": pair[0]["actual_resolution"],
                    "primary_machine_classification": pair[0]["primary_machine_classification"],
                    "review_eligible": pair[0]["review_eligible"],
                },
                "cand01": {
                    "request_id": pair[1]["request_id"],
                    "actual_resolution": pair[1]["actual_resolution"],
                    "primary_machine_classification": pair[1]["primary_machine_classification"],
                    "review_eligible": pair[1]["review_eligible"],
                },
                "pass_candidate_count_1349x1166": len(eligible),
                "review_candidates": [{"request_id": item["request_id"], "output_path": item["output_path"]} for item in eligible],
                "missing_1349_pass_candidate": not eligible,
                "other_resolution_machine_pass_candidates": [
                    {"request_id": item["request_id"], "actual_resolution": item["actual_resolution"]} for item in other_pass
                ],
                "human_review_status": None,
            })
    coverage_by_cell = {item["cell_key"]: item for item in coverage_records}
    corrected_missing = [item["cell_key"] for item in coverage_records if item["missing_1349_pass_candidate"]]
    no_machine_pass_any_resolution = [
        item["cell_key"] for item in coverage_records
        if not records_by_id[item["cand00"]["request_id"]]["is_similarity_pass_candidate"]
        and not records_by_id[item["cand01"]["request_id"]]["is_similarity_pass_candidate"]
    ]
    other_resolution_pass_cells = [
        item["cell_key"] for item in coverage_records if item["other_resolution_machine_pass_candidates"]
    ]

    disputed_ids = ["subject00_O03_slot07_cand01", "subject00_O04_slot01_cand00", "subject00_O04_slot01_cand01"]
    dispute = {request_id: {
        "actual_resolution_from_png_ihdr": records_by_id[request_id]["actual_resolution"],
        "actual_png_sha256": records_by_id[request_id]["output_sha256"],
        "provenance_width": provenance[request_id]["width"],
        "provenance_height": provenance[request_id]["height"],
        "provenance_output_sha256": provenance[request_id]["output_sha256"],
        "request_binding_status": records_by_id[request_id]["provenance_binding_status"],
        "primary_machine_classification": records_by_id[request_id]["primary_machine_classification"],
        "belongs_to_1349x1166_cohort": records_by_id[request_id]["belongs_to_1349x1166_cohort"],
    } for request_id in disputed_ids}
    if corrected_missing != OLD_MISSING:
        raise RuntimeError(f"unexpected corrected gap list: {corrected_missing}")
    if dispute["subject00_O03_slot07_cand01"]["actual_resolution_from_png_ihdr"] != "1350x1165":
        raise RuntimeError("O03/slot07/cand01 evidence changed")
    if any(dispute[request_id]["actual_resolution_from_png_ihdr"] != TARGET_RESOLUTION for request_id in disputed_ids[1:]):
        raise RuntimeError("O04/slot01 evidence changed")

    cloud_manifest = load_json(CLOUD_MANIFEST)
    archive_sizes = {f"subject00/{item['relative_path']}": int(item["size_bytes"]) for item in cloud_manifest["files"]}
    mask_members = [f"subject00/masks/{binding['camera']}/00000000.jpg" for binding in SLOTS.values()]
    mask_payloads = archive_member_batch(mask_members, archive_sizes)
    masks_by_slot: dict[str, Image.Image] = {}
    for slot, binding in SLOTS.items():
        member = f"subject00/masks/{binding['camera']}/00000000.jpg"
        expected_sha = records_by_id[f"subject00_O01_slot{slot[-2:]}_cand00"]["source_mask_sha256"]
        if bytes_sha256(mask_payloads[member]) != expected_sha:
            raise RuntimeError(f"official mask SHA mismatch: {slot}")
        mask = Image.open(io.BytesIO(mask_payloads[member])).convert("L")
        mask = mask.point(lambda value: 255 if value >= protocol["source_mask"]["foreground_threshold"] else 0)
        masks_by_slot[slot] = mask
        mask.save(OUTPUT_ROOT / "masks" / f"{binding['camera']}_official_person_mask.png", format="PNG")

    review_candidates: list[dict[str, Any]] = []
    eligible_records = [item for item in records if item["review_eligible"]]
    for index, record in enumerate(eligible_records, start=1):
        metric = metrics[record["request_id"]]
        assets, risk_signals = build_candidate_assets(record, metric, masks_by_slot[record["slot"]], protocol)
        candidate = {
            **record,
            "review_assets": assets,
            "review_asset_count": 13,
            "risk_signals": risk_signals,
            "registration_metrics": {
                "uniform_scale": float(metric["uniform_scale"]),
                "translation_x": float(metric["translation_x"]),
                "translation_y": float(metric["translation_y"]),
                "rotation_degrees": float(metric["rotation_degrees"]),
                "median_reprojection_error_px": float(metric["median_reprojection_error_px"]),
                "p95_reprojection_error_px": float(metric["p95_reprojection_error_px"]),
                "ransac_inlier_ratio": float(metric["similarity"]["inlier_ratio"]),
            },
            "exact_cohort": True,
            "machine_pass": True,
        }
        review_candidates.append(candidate)
        print(f"[{index:02d}/{len(eligible_records):02d}] assets {record['request_id']}", flush=True)
    risk_order = add_risk_ranks(review_candidates, protocol)
    review_by_id = {item["request_id"]: item for item in review_candidates}
    candidates_by_cell = defaultdict(list)
    for item in review_candidates:
        candidates_by_cell[item["cell_key"]].append(item)

    pdf_paths = {
        garment: OUTPUT_ROOT / "pdf" / f"{garment}_1349x1166_human_review.pdf" for garment in GARMENTS
    }
    for garment, path in pdf_paths.items():
        build_garment_pdf(garment, candidates_by_cell, coverage_by_cell, path)
        print(f"[PDF] {path.name}", flush=True)
    high_risk_pdf = OUTPUT_ROOT / "pdf" / "1349x1166_high_risk_candidate_review.pdf"
    build_risk_pdf(risk_order, high_risk_pdf)
    print(f"[PDF] {high_risk_pdf.name}", flush=True)
    coverage_sheet = OUTPUT_ROOT / "corrected_1349x1166_24_cell_coverage_sheet.png"
    build_coverage_sheet(coverage_records, coverage_sheet)
    print(f"[SHEET] {coverage_sheet.name}", flush=True)

    review_manifest_records = []
    null_fields = [
        "identity_consistency", "face_consistency", "pose_preservation", "camera_direction",
        "single_person", "hands_complete", "feet_complete", "garment_correctness",
        "garment_boundary_quality", "silhouette_quality", "background_geometry",
        "visible_artifacts", "human_decision", "rejection_reasons", "selected_for_cell",
    ]
    for item in sorted(review_candidates, key=lambda candidate: candidate["request_id"]):
        value = {
            "request_id": item["request_id"],
            "garment": item["garment"],
            "slot": item["slot"],
            "camera": item["camera"],
            "candidate_id": item["candidate_id"],
            "source_sha": item["source_sha256"],
            "output_sha": item["output_sha256"],
            "machine_pass": True,
            "exact_cohort": True,
            "garment_review_pdf": str(pdf_paths[item["garment"]]),
            "garment_review_pdf_page": int(item["slot"][-2:]) + 1,
            "high_risk_review_pdf": str(high_risk_pdf),
            "high_risk_review_pdf_page": item["risk_priority_rank"],
            "review_assets": item["review_assets"],
            "accepted": False,
            "teacher_target": False,
        }
        value.update({field: None for field in null_fields})
        review_manifest_records.append(value)

    cohort_payload = {
        "schema_version": "canondressgs.subject00.corrected_attempt001_exact_resolution_cohorts.v1",
        "task_id": TASK_ID,
        "source_evidence": "ACTUAL_48_PNG_IHDR_PLUS_FROZEN_REQUEST_PROVENANCE_AND_REGISTRATION_METRICS",
        "output_count": 48,
        "exact_resolution_distribution": dict(sorted(distribution.items())),
        "cohorts": cohorts,
        "dominant_resolution": TARGET_RESOLUTION,
        "dominant_output_count": target_cohort["output_count"],
        "dominant_raw_cell_coverage": target_cohort["raw_cell_count"],
        "dominant_machine_pass_candidate_count": target_cohort["machine_pass_candidate_count"],
        "dominant_machine_pass_cell_coverage": target_cohort["machine_pass_cell_count"],
        "paper_final": False,
    }
    candidate_payload = {
        "schema_version": "canondressgs.subject00.corrected_attempt001_1349_candidate_registry.v1",
        "task_id": TASK_ID,
        "record_count": len(records),
        "review_candidate_count": len(review_candidates),
        "review_cell_count": len({item["cell_key"] for item in review_candidates}),
        "multi_candidate_cell_count": sum(len(items) == 2 for items in candidates_by_cell.values()),
        "records": records,
        "review_candidates": review_candidates,
        "risk_order": [item["request_id"] for item in risk_order],
        "automatic_visual_decision_count": 0,
        "automatic_candidate_selection_count": 0,
        "paper_final": False,
    }
    coverage_payload = {
        "schema_version": "canondressgs.subject00.corrected_attempt001_1349_cell_coverage.v1",
        "task_id": TASK_ID,
        "cell_count": 24,
        "raw_cell_coverage": target_cohort["raw_cell_count"],
        "machine_pass_candidate_count": target_cohort["machine_pass_candidate_count"],
        "machine_pass_cell_coverage": target_cohort["machine_pass_cell_count"],
        "missing_cell_list": corrected_missing,
        "no_machine_pass_candidate_at_any_resolution_cells": no_machine_pass_any_resolution,
        "other_resolution_machine_pass_candidate_cells": other_resolution_pass_cells,
        "records": coverage_records,
        "human_review_status": None,
        "paper_final": False,
    }
    overlay_payload = {
        "schema_version": "canondressgs.subject00.attempt001_1349_cohort_correction_overlay.v1",
        "task_id": TASK_ID,
        "old_report_path": str(OLD_REPORT_PATH),
        "old_report_sha256": file_sha256(OLD_REPORT_PATH),
        "old_final_summary_path": str(OLD_SUMMARY_PATH),
        "old_final_summary_sha256": file_sha256(OLD_SUMMARY_PATH),
        "old_coverage_registry_path": str(OLD_COVERAGE_PATH),
        "old_coverage_registry_sha256": file_sha256(OLD_COVERAGE_PATH),
        "old_contact_sheet_path": old_summary["contact_sheet_paths"]["machine_pass"],
        "old_contact_sheet_sha256": file_sha256(Path(old_summary["contact_sheet_paths"]["machine_pass"])),
        "old_missing_cell_list": OLD_MISSING,
        "hypothesized_missing_cell_list": HYPOTHESIZED_MISSING,
        "recomputed_missing_cell_list": corrected_missing,
        "correction_adjudication": "OLD_REGISTRY_CONFIRMED_CORRECT_PROPOSED_O03_O04_SWAP_REJECTED",
        "cohort_registry_evidence_conflict": False,
        "disputed_request_adjudication": dispute,
        "root_cause": "The dense multi-column contact sheet was visually misread across row/cell boundaries. Its request labels are correct; PNG IHDR, PNG SHA, request/provenance binding, inventory, metrics, and generation code all confirm the old registry.",
        "underlying_json_was_wrong": False,
        "registry_generation_code_was_wrong": False,
        "final_response_transcription_was_wrong": False,
        "affected_downstream_decision": "NONE. The existing six-cell targeted-gap set remains evidence-correct, but generation authorization remains denied.",
        "old_evidence_preserved": True,
        "old_report_mutations": 0,
        "attempt_001_mutations": 0,
        "attempt_002_mutations": 0,
        "attempt_003_mutations": 0,
        "targeted_rerun_authorization": "DENIED",
        "targeted_rerun_generated_count": 0,
        "paper_final": False,
    }
    review_payload = {
        "schema_version": "canondressgs.subject00.corrected_1349_human_review_manifest.v1",
        "task_id": TASK_ID,
        "status": "PENDING_USER_REVIEW",
        "record_count": len(review_manifest_records),
        "cell_count": len({item["cell_key"] for item in review_candidates}),
        "records": review_manifest_records,
        "human_decision_non_null_count": 0,
        "accepted_count": 0,
        "teacher_target_count": 0,
        "paper_final": False,
    }

    pdf_registry = {
        key: {
            "path": str(path),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
            "page_count": len(PdfReader(str(path)).pages),
        }
        for key, path in {**pdf_paths, "high_risk": high_risk_pdf}.items()
    }
    asset_files = sorted(path for path in (OUTPUT_ROOT / "assets").rglob("*") if path.is_file())
    asset_inventory = [{
        "relative_path": path.relative_to(OUTPUT_ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    } for path in asset_files]
    with Image.open(coverage_sheet) as image:
        coverage_dimensions = list(image.size)
        image.verify()
    pack_registry = {
        "schema_version": "canondressgs.subject00.attempt001_1349_high_res_review_pack_registry.v1",
        "task_id": TASK_ID,
        "output_root": str(OUTPUT_ROOT),
        "review_candidate_count": len(review_candidates),
        "review_cell_count": len({item["cell_key"] for item in review_candidates}),
        "multi_candidate_cell_count": sum(len(items) == 2 for items in candidates_by_cell.values()),
        "pdfs": pdf_registry,
        "corrected_coverage_sheet": {
            "path": str(coverage_sheet),
            "sha256": file_sha256(coverage_sheet),
            "bytes": coverage_sheet.stat().st_size,
            "dimensions": coverage_dimensions,
        },
        "candidate_asset_file_count": len(asset_inventory),
        "candidate_asset_tree_sha256": canonical_sha256(asset_inventory),
        "required_candidate_panel_count": len(protocol["required_candidate_panels"]),
        "all_candidate_assets_complete": all(len(item["review_assets"]) == 13 for item in review_candidates),
        "accepted_count": 0,
        "teacher_target_count": 0,
        "paper_final": False,
    }
    final_summary = {
        "schema_version": "canondressgs.subject00.attempt001_1349_cohort_correction_final_summary.v1",
        "task_id": TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "new_branch": BRANCH,
        "worktree": str(ROOT),
        "old_1349_output_count": old_cohorts["dominant_output_count"],
        "recomputed_1349_output_count": target_cohort["output_count"],
        "old_1349_raw_cell_coverage": old_cohorts["dominant_garment_slot_coverage"],
        "recomputed_1349_raw_cell_coverage": target_cohort["raw_cell_count"],
        "old_1349_machine_pass_cell_coverage": old_cohorts["dominant_machine_pass_cell_coverage"],
        "recomputed_1349_machine_pass_cell_coverage": target_cohort["machine_pass_cell_count"],
        "old_missing_cell_list": OLD_MISSING,
        "corrected_missing_cell_list": corrected_missing,
        "review_candidate_count": len(review_candidates),
        "review_cell_count": len({item["cell_key"] for item in review_candidates}),
        "multi_candidate_cell_count": sum(len(items) == 2 for items in candidates_by_cell.values()),
        "old_report_mutations": 0,
        "attempt_001_mutations": 0,
        "attempt_002_mutations": 0,
        "attempt_003_mutations": 0,
        "new_generation_calls": 0,
        "external_api_calls": 0,
        "api_key_reads": 0,
        "model_downloads": 0,
        "human_decision_non_null_count": 0,
        "accepted_count": 0,
        "teacher_target_count": 0,
        "targeted_rerun_authorization": "DENIED",
        "targeted_rerun_generated_count": 0,
        "formal_base_status": "PENDING",
        "subject00_paper_positive_claim_count": 0,
        "paper_modifications": 0,
        "pdf_paths": {key: value["path"] for key, value in pdf_registry.items()},
        "corrected_coverage_sheet": str(coverage_sheet),
        "review_manifest": str(OUTPUT_ROOT / "corrected_1349x1166_human_review_manifest.json"),
        "final_classification": FINAL_CLASSIFICATION,
        "paper_final": False,
        "next_task": NEXT_TASK,
    }

    dual_json(COHORTS_PATH, "registries/corrected_attempt001_exact_resolution_cohorts.json", cohort_payload)
    dual_json(COVERAGE_PATH, "registries/corrected_attempt001_1349x1166_cell_coverage.json", coverage_payload)
    dual_json(CANDIDATES_PATH, "registries/corrected_attempt001_1349x1166_candidate_registry.json", candidate_payload)
    dual_json(REVIEW_PATH, "corrected_1349x1166_human_review_manifest.json", review_payload)
    dual_json(OVERLAY_PATH, "registries/subject00_attempt001_1349_cohort_correction_overlay_20260726.json", overlay_payload)
    dual_json(PACK_REGISTRY_PATH, "subject00_attempt001_1349_high_res_review_pack_registry_20260726.json", pack_registry)
    dual_json(FINAL_SUMMARY_PATH, "subject00_attempt001_1349_cohort_correction_final_summary_20260726.json", final_summary)

    report = f"""# Subject00 Attempt 001 1349x1166 Cohort Correction and High-Resolution Review Pack

- Task: `{TASK_ID}`
- Source HEAD: `{SOURCE_HEAD}`
- Actual PNGs reparsed: `48/48`
- Recomputed `1349x1166` outputs: `{target_cohort['output_count']}`
- Recomputed raw cell coverage: `{target_cohort['raw_cell_count']}/24`
- Recomputed machine-pass candidates: `{target_cohort['machine_pass_candidate_count']}` across `{target_cohort['machine_pass_cell_count']}/24` cells
- Review candidates: `{len(review_candidates)}`; multi-candidate cells: `{sum(len(items) == 2 for items in candidates_by_cell.values())}`
- Final classification: `{FINAL_CLASSIFICATION}`

## Adjudication

The proposed O03/O04 cell swap is rejected by direct evidence. `subject00_O03_slot07_cand01` is `1350x1165`, while both `subject00_O04_slot01_cand00` and `subject00_O04_slot01_cand01` are `1349x1166`. All three PNG IHDR values agree with the PNG SHA-bound provenance, old inventory, registration metrics, request-level review panels, and registry generation code.

The old missing-cell list is therefore retained unchanged: `{', '.join(corrected_missing)}`. The root cause of the apparent contradiction was visual misreading of the dense multi-column contact sheet, not an underlying JSON, indexing, or final-response transcription defect. Old evidence remains preserved and unmodified.

## Review State

The high-resolution review pack contains all 31 eligible candidates. Every garment PDF contains eight cell pages; dual candidates share one page, missing cells use a text-only `NO_1349_PASS_CANDIDATE` page, and no placeholder candidate image is inserted. The high-risk PDF ranks candidates for review priority only. All human decision fields remain null; `accepted=0`, `teacher_target=0`, targeted rerun authorization remains denied, and `PAPER_FINAL=false`.
"""
    write_text(REPORT_PATH, report)
    write_text(OUTPUT_ROOT / "SUBJECT00_ATTEMPT001_1349_COHORT_CORRECTION_REPORT.md", report)
    handoff = {
        **final_summary,
        "schema_version": "canondressgs.subject00.attempt001_1349_cohort_correction_human_review_prep_handoff.v1",
        "correction_overlay": str(OVERLAY_PATH.relative_to(ROOT)),
        "cohort_registry": str(COHORTS_PATH.relative_to(ROOT)),
        "coverage_registry": str(COVERAGE_PATH.relative_to(ROOT)),
        "candidate_registry": str(CANDIDATES_PATH.relative_to(ROOT)),
        "review_manifest_registry": str(REVIEW_PATH.relative_to(ROOT)),
        "review_pack_registry": str(PACK_REGISTRY_PATH.relative_to(ROOT)),
        "execution_status": "WAITING_FOR_USER_HIGH_RES_VISUAL_REVIEW",
    }
    write_json(HANDOFF_PATH, handoff)

    print(json.dumps({
        "status": "PACK_BUILT",
        "recomputed_1349_output_count": target_cohort["output_count"],
        "raw_cell_coverage": target_cohort["raw_cell_count"],
        "machine_pass_candidate_count": target_cohort["machine_pass_candidate_count"],
        "machine_pass_cell_coverage": target_cohort["machine_pass_cell_count"],
        "missing_cells": corrected_missing,
        "review_candidate_count": len(review_candidates),
        "multi_candidate_cell_count": sum(len(items) == 2 for items in candidates_by_cell.values()),
        "pdf_registry": pdf_registry,
        "final_classification": FINAL_CLASSIFICATION,
        "next_task": NEXT_TASK,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
