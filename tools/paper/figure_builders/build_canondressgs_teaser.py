#!/usr/bin/env python3
"""Build Figure 1 from archived CanonDressGS experiment pixels only.

This script never invokes a model, renderer, evaluator, or training code. It
only verifies archived source hashes, crops the reference clothing-mask bbox,
resizes raster panels, and draws vector-compatible figure furniture.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


TASK_ID = "AAAI27-CANONDRESSGS-TEASER-REAL-ASSETS-001"
SOURCE_BRANCH = "paper/aaai27-p0-color-spatial-soft-control-evaluations-20260722"
SOURCE_HEAD = "371d812864614cd561e33edfe3f6c043b38415d2"
NEW_BRANCH = "paper/aaai27-real-teaser-figure-20260722"
GARMENTS = ("O01", "O03", "O08")
OURS_RUN_ID = "P0-FORMAL-OURS-V2-R0"
OURS_ATTEMPT = (
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "AAAI27-P0-FORMAL-CANDIDATE-RUNS/formal_runs/ours_v2/"
    "replicate_0/attempt_002"
)
DATASET_ROOT = (
    "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/"
    "SUBJECT02-AAAI27-DATA-CAPACITY-GATE-001/attempt_003/dataset"
)
FORMAL_ROOT = (
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "AAAI27-P0-FORMAL-CANDIDATE-RUNS"
)
SEEN_ROOT = (
    "/root/autodl-tmp/canondressgs_work/outputs/AAAI27-SEEN-OUTFIT-PAPER"
)

WIDTH = 3600
HEIGHT = 1714
DPI = 300
BLUE = (31, 111, 235)
BLACK = (24, 29, 38)
GRAY = (111, 119, 132)
LIGHT = (218, 223, 230)
WHITE = (255, 255, 255)


REFERENCE_HASHES = {
    "O01": {
        "rgb": "9a089db7ffd92bc394faa91c32053289aca071aaf9592adf5a45a1c5d8b84501",
        "mask": "a9071f551c67efc7d933c3a8cb749b5d55a6aa93bc80599800198fd27322a92b",
    },
    "O03": {
        "rgb": "b11e0feb45b747cffe74613b4fb7d6cf9e290e08f2251317ffa49ee64c512117",
        "mask": "a25b9f0e7586fbfe3ce5e6c6035cd5df2fb5cae540a0a46c27be4027d0a222c6",
    },
    "O08": {
        "rgb": "40accfe8a3513da1d424fea5437b1979f7ad85ef17854c7757ee6c22eb938f21",
        "mask": "438eff8699c1c6cadc1ec952ae88cdb7811c36ae46963e548140565dcd8f0c49",
    },
}

BASE_SPEC = {
    "cached_path": "paper_draft/aaai27/figure_sources/figure1/base/B0_O01_cond_000000_prediction.png",
    "source_path": (
        f"{SEEN_ROOT}/PAPER-B0-FIXED/seed_fixed/attempt_001/visuals/episodes/"
        "O01_cond_000000_prediction.png"
    ),
    "sha256": "ff037b1fba9e00db7f47eab534d794fd54236e8c8aada9bd42406b3c5712a28f",
    "method": "B0_base_avatar",
    "episode": "O01_cond_000000",
    "condition_id": "cond_000000",
}

RESULT_HASHES = {
    ("O01", "cond_000318"): "2338c683760718c876325f783bfce97c0994134a80e9ff1b38b2363196480171",
    ("O01", "cond_000017"): "c5338ea726e2c065d4e83fe855bfa4f6feb25d7fb7efa2c03d05ff0a90e71d66",
    ("O03", "cond_000318"): "be3dea00440f60b6f3893fa2b579b360a08c9e7d1fe7fb79a90ee56435f1efab",
    ("O03", "cond_000017"): "29212dd7048947f210d5192a24f509f0dc5669ab85ccc49c1e5bf4b29994be3e",
    ("O08", "cond_000318"): "e02e2c1e391815f2b055542f7e828442ad41fe6348420f6f354191fcf8050890",
    ("O08", "cond_000017"): "6d77929f7e4ff678faee904ffaecfcf10d81c856bd22f6ae6208c776da2a5c23",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=9)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def relative(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def font(size: int) -> ImageFont.FreeTypeFont:
    # Pillow ships this sans-serif font, avoiding host font substitutions.
    return ImageFont.load_default(size=size)


def condition_map(dataset: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["condition_id"]: row for row in dataset["conditions"]}


def select_pose_pair(dataset: dict[str, Any]) -> tuple[str, str, float]:
    rows = dataset["conditions"]
    candidates: list[tuple[float, str, str]] = []
    for left, right in itertools.combinations(rows, 2):
        squared = sum(
            (float(a) - float(b)) ** 2
            for a, b in zip(left["pose"], right["pose"], strict=True)
        )
        candidates.append(
            (math.sqrt(squared), left["condition_id"], right["condition_id"])
        )
    distance, pose_a, pose_b = sorted(
        candidates, key=lambda row: (-row[0], row[1], row[2])
    )[0]
    if (pose_a, pose_b) != ("cond_000318", "cond_000017"):
        raise RuntimeError(
            f"Unexpected maximum-distance pose pair: {pose_a}, {pose_b}"
        )
    return pose_a, pose_b, distance


def observation(dataset: dict[str, Any], garment: str, condition: str) -> dict[str, Any]:
    outfit = next(row for row in dataset["outfits"] if row["outfit_id"] == garment)
    return next(
        row for row in outfit["observations"] if row["condition_id"] == condition
    )


def verify_hash(path: Path, expected: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"Source hash mismatch for {path}: {actual} != {expected}")
    return actual


def mask_crop_box(mask: Image.Image) -> tuple[int, int, int, int]:
    binary = mask.convert("L").point(lambda value: 255 if value > 127 else 0)
    bbox = binary.getbbox()
    if bbox is None:
        raise RuntimeError("Clothing mask is empty")
    left, top, right, bottom = bbox
    pad_x = int(round((right - left) * 0.05))
    pad_y = int(round((bottom - top) * 0.05))
    return (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(mask.width, right + pad_x),
        min(mask.height, bottom + pad_y),
    )


def fit_panel(
    source: Image.Image,
    slot: tuple[int, int],
) -> tuple[Image.Image, tuple[int, int]]:
    panel = Image.new("RGB", slot, WHITE)
    scale = min(slot[0] / source.width, slot[1] / source.height)
    size = (
        max(1, int(round(source.width * scale))),
        max(1, int(round(source.height * scale))),
    )
    resized = source.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    offset = ((slot[0] - size[0]) // 2, (slot[1] - size[1]) // 2)
    panel.paste(resized, offset)
    return panel, size


def draw_arrow(draw: ImageDraw.ImageDraw, x1: int, x2: int, y: int) -> None:
    draw.line((x1, y, x2 - 18, y), fill=BLUE, width=5)
    draw.polygon(((x2, y), (x2 - 22, y - 13), (x2 - 22, y + 13)), fill=BLUE)


def draw_text_center(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    size: int,
    fill: tuple[int, int, int] = BLACK,
) -> None:
    draw.text(xy, text, font=font(size), fill=fill, anchor="mm")


def pdf_y(top: float, height: float = 0.0) -> float:
    scale = 72.0 / DPI
    return HEIGHT * scale - (top + height) * scale


def pdf_arrow(pdf: canvas.Canvas, x1: int, x2: int, y: int) -> None:
    scale = 72.0 / DPI
    pdf.setStrokeColorRGB(*(value / 255.0 for value in BLUE))
    pdf.setFillColorRGB(*(value / 255.0 for value in BLUE))
    pdf.setLineWidth(1.2)
    py = pdf_y(y)
    pdf.line(x1 * scale, py, (x2 - 18) * scale, py)
    path = pdf.beginPath()
    path.moveTo(x2 * scale, py)
    path.lineTo((x2 - 22) * scale, pdf_y(y - 13))
    path.lineTo((x2 - 22) * scale, pdf_y(y + 13))
    path.close()
    pdf.drawPath(path, fill=1, stroke=0)


def pdf_text_center(
    pdf: canvas.Canvas,
    xy: tuple[int, int],
    text: str,
    size: int,
    color: tuple[int, int, int] = BLACK,
) -> None:
    scale = 72.0 / DPI
    pdf.setFillColorRGB(*(value / 255.0 for value in color))
    pdf.setFont("Helvetica", size * scale)
    pdf.drawCentredString(xy[0] * scale, pdf_y(xy[1] + size * 0.34), text)


def build(
    repo_root: Path,
    png_path: Path,
    pdf_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    source_root = repo_root / "paper_draft/aaai27/figure_sources/figure1"
    dataset_path = source_root / "manifests/aaai_gate_28_manifest.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    formal_registry_path = (
        repo_root / "paper_protocol/reviewer_risk/p0_formal_candidate_run_registry.yaml"
    )
    registry_text = formal_registry_path.read_text(encoding="utf-8")
    block_start = registry_text.index(f"- formal_run_id: {OURS_RUN_ID}")
    block_end = registry_text.find("\n- formal_run_id:", block_start + 1)
    ours_registry_block = registry_text[
        block_start : block_end if block_end >= 0 else len(registry_text)
    ]
    if "\n  method: Ours-v2\n" not in ours_registry_block:
        raise RuntimeError("Formal registry does not classify selected run as Ours-v2")
    if f"\n  attempt: {OURS_ATTEMPT}\n" not in ours_registry_block:
        raise RuntimeError("Formal Ours-v2 attempt differs from the sealed registry")

    pose_a, pose_b, pose_distance = select_pose_pair(dataset)
    target_conditions = (pose_a, pose_b)
    all_condition_ids = [row["condition_id"] for row in dataset["conditions"]]
    reference_condition = next(
        condition for condition in all_condition_ids if condition not in target_conditions
    )
    if reference_condition != "cond_000000":
        raise RuntimeError("Reference condition selection changed")
    conditions = condition_map(dataset)
    if conditions[pose_a]["source_checksum"]["pose_sha256"] == conditions[pose_b][
        "source_checksum"
    ]["pose_sha256"]:
        raise RuntimeError("POSE_DIVERSITY_NOT_VERIFIED")

    layout = {
        "reference_slots": [(180, 275 + 410 * index, 560, 340) for index in range(3)],
        "base_slot": (930, 350, 540, 1000),
        "result_slots_a": [(1690, 255 + 410 * index, 760, 380) for index in range(3)],
        "result_slots_b": [(2610, 255 + 410 * index, 760, 380) for index in range(3)],
        "row_centers": [445, 855, 1265],
    }

    figure = Image.new("RGB", (WIDTH, HEIGHT), WHITE)
    raster_draw = ImageDraw.Draw(figure)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    page_size = (WIDTH * 72.0 / DPI, HEIGHT * 72.0 / DPI)
    pdf = canvas.Canvas(
        str(pdf_path), pagesize=page_size, invariant=1, pageCompression=1
    )
    pdf.setTitle("CanonDressGS Figure 1 Teaser")
    pdf.setAuthor("CanonDressGS")
    pdf.setCreator("build_canondressgs_teaser.py")
    pdf.setSubject("Archived real experiment assets; manual review required")

    for x in (850, 1560):
        raster_draw.line((x, 45, x, 1560), fill=LIGHT, width=3)
        pdf.setStrokeColorRGB(*(value / 255.0 for value in LIGHT))
        pdf.setLineWidth(0.7)
        pdf.line(x * 72 / DPI, pdf_y(45), x * 72 / DPI, pdf_y(1560))
    for y in (665, 1075):
        raster_draw.line((80, y, 800, y), fill=LIGHT, width=2)
        raster_draw.line((1630, y, 3450, y), fill=LIGHT, width=2)
        pdf.setStrokeColorRGB(*(value / 255.0 for value in LIGHT))
        pdf.setLineWidth(0.5)
        pdf.line(80 * 72 / DPI, pdf_y(y), 800 * 72 / DPI, pdf_y(y))
        pdf.line(1630 * 72 / DPI, pdf_y(y), 3450 * 72 / DPI, pdf_y(y))

    for y in layout["row_centers"]:
        draw_arrow(raster_draw, 770, 920, y)
        draw_arrow(raster_draw, 1480, 1680, y)
        pdf_arrow(pdf, 770, 920, y)
        pdf_arrow(pdf, 1480, 1680, y)

    headers = [
        ((440, 74), "Garment References", 48, BLACK),
        ((1200, 74), "Base Avatar", 48, BLACK),
        ((2530, 64), "Edited Animatable Results", 48, BLACK),
        ((2530, 114), "Ours-v2 formal endpoints", 28, BLUE),
        ((2070, 176), "Pose A", 38, BLACK),
        ((2070, 214), pose_a, 24, GRAY),
        ((2990, 176), "Pose B", 38, BLACK),
        ((2990, 214), pose_b, 24, GRAY),
    ]
    for xy, text, size, color in headers:
        draw_text_center(raster_draw, xy, text, size, color)
        pdf_text_center(pdf, xy, text, size, color)

    reference_records: list[dict[str, Any]] = []
    result_records: list[dict[str, Any]] = []
    panel_images: list[tuple[Image.Image, tuple[int, int, int, int]]] = []

    for index, garment in enumerate(GARMENTS):
        draw_text_center(
            raster_draw, (112, layout["row_centers"][index]), garment, 34, BLUE
        )
        pdf_text_center(
            pdf, (112, layout["row_centers"][index]), garment, 34, BLUE
        )
        obs = observation(dataset, garment, reference_condition)
        rgb_rel = (
            f"paper_draft/aaai27/figure_sources/figure1/references/"
            f"{garment}_{reference_condition}_rgb.png"
        )
        mask_rel = (
            f"paper_draft/aaai27/figure_sources/figure1/references/"
            f"{garment}_{reference_condition}_clothing_mask.png"
        )
        rgb_path = repo_root / rgb_rel
        mask_path = repo_root / mask_rel
        rgb_sha = verify_hash(rgb_path, REFERENCE_HASHES[garment]["rgb"])
        mask_sha = verify_hash(mask_path, REFERENCE_HASHES[garment]["mask"])
        if obs["rgb"] != f"{DATASET_ROOT}/rgb/edit/{garment}/{reference_condition}.png":
            raise RuntimeError(f"Reference RGB provenance mismatch for {garment}")
        if obs["clothing_mask"] != (
            f"{DATASET_ROOT}/masks/target_clothing_mask/"
            f"{garment}/{reference_condition}.png"
        ):
            raise RuntimeError(f"Reference mask provenance mismatch for {garment}")
        with Image.open(rgb_path) as rgb_image, Image.open(mask_path) as mask_image:
            crop_box = mask_crop_box(mask_image)
            cropped = rgb_image.convert("RGB").crop(crop_box)
            slot = layout["reference_slots"][index]
            panel, resized_size = fit_panel(cropped, (slot[2], slot[3]))
            panel_images.append((panel, slot))
            reference_records.append(
                {
                    "garment_id": garment,
                    "condition_id": reference_condition,
                    "role": "archived_formal_reference_input",
                    "source_path": obs["rgb"],
                    "source_sha256": rgb_sha,
                    "cached_path": rgb_rel,
                    "mask_source_path": obs["clothing_mask"],
                    "mask_sha256": mask_sha,
                    "mask_cached_path": mask_rel,
                    "archived_generation_provider": obs.get("raw_generation_provider"),
                    "source_size": list(rgb_image.size),
                    "crop_rule": "clothing-mask bbox plus 5 percent per side",
                    "crop_box": list(crop_box),
                    "resize_size": list(resized_size),
                    "panel_slot_size": [slot[2], slot[3]],
                    "output_panel_sha256": sha256_image(panel),
                }
            )

        for pose_label, condition_id, slot in (
            ("Pose A", pose_a, layout["result_slots_a"][index]),
            ("Pose B", pose_b, layout["result_slots_b"][index]),
        ):
            result_rel = (
                "paper_draft/aaai27/figure_sources/figure1/results/"
                f"{garment}_{condition_id}_prediction.png"
            )
            result_path = repo_root / result_rel
            result_sha = verify_hash(
                result_path, RESULT_HASHES[(garment, condition_id)]
            )
            canonical_path = (
                f"{OURS_ATTEMPT}/renders/episodes/"
                f"{garment}_{condition_id}_prediction.png"
            )
            with Image.open(result_path) as result_image:
                panel, resized_size = fit_panel(
                    result_image.convert("RGB"), (slot[2], slot[3])
                )
                panel_images.append((panel, slot))
                condition = conditions[condition_id]
                result_records.append(
                    {
                        "garment_id": garment,
                        "pose_label": pose_label,
                        "condition_id": condition_id,
                        "method": "Ours-v2",
                        "formal_run_id": OURS_RUN_ID,
                        "replicate_index": 0,
                        "attempt": "attempt_002",
                        "identity": "subject02",
                        "source_path": canonical_path,
                        "source_sha256": result_sha,
                        "cached_path": result_rel,
                        "pose_sha256": condition["source_checksum"]["pose_sha256"],
                        "camera_sha256": condition["source_checksum"]["camera_sha256"],
                        "source_size": list(result_image.size),
                        "crop_box": [0, 0, result_image.width, result_image.height],
                        "resize_size": list(resized_size),
                        "panel_slot_size": [slot[2], slot[3]],
                        "output_panel_sha256": sha256_image(panel),
                    }
                )

    base_path = repo_root / BASE_SPEC["cached_path"]
    base_sha = verify_hash(base_path, BASE_SPEC["sha256"])
    base_slot = layout["base_slot"]
    with Image.open(base_path) as base_image:
        base_panel, base_resize = fit_panel(
            base_image.convert("RGB"), (base_slot[2], base_slot[3])
        )
        panel_images.append((base_panel, base_slot))
        base_record = {
            **BASE_SPEC,
            "source_sha256": base_sha,
            "identity": "subject02",
            "pose_sha256": conditions[BASE_SPEC["condition_id"]]["source_checksum"][
                "pose_sha256"
            ],
            "camera_sha256": conditions[BASE_SPEC["condition_id"]][
                "source_checksum"
            ]["camera_sha256"],
            "source_size": list(base_image.size),
            "crop_box": [0, 0, base_image.width, base_image.height],
            "resize_size": list(base_resize),
            "panel_slot_size": [base_slot[2], base_slot[3]],
            "output_panel_sha256": sha256_image(base_panel),
        }
    draw_text_center(raster_draw, (1200, 1408), "subject02 / formal B0", 28, GRAY)
    pdf_text_center(pdf, (1200, 1408), "subject02 / formal B0", 28, GRAY)

    # Paste source panels after arrows so white panel backgrounds preserve pixels.
    for panel, slot in panel_images:
        figure.paste(panel, (slot[0], slot[1]))
        raster_draw.rectangle(
            (slot[0], slot[1], slot[0] + slot[2], slot[1] + slot[3]),
            outline=LIGHT,
            width=2,
        )
        scale = 72.0 / DPI
        pdf.drawImage(
            ImageReader(panel),
            slot[0] * scale,
            pdf_y(slot[1], slot[3]),
            width=slot[2] * scale,
            height=slot[3] * scale,
            preserveAspectRatio=False,
            mask="auto",
        )
        pdf.setStrokeColorRGB(*(value / 255.0 for value in LIGHT))
        pdf.setLineWidth(0.5)
        pdf.rect(
            slot[0] * scale,
            pdf_y(slot[1], slot[3]),
            slot[2] * scale,
            slot[3] * scale,
            fill=0,
            stroke=1,
        )

    raster_draw.line((80, 1570, 3520, 1570), fill=LIGHT, width=3)
    pdf.setStrokeColorRGB(*(value / 255.0 for value in LIGHT))
    pdf.setLineWidth(0.7)
    pdf.line(80 * 72 / DPI, pdf_y(1570), 3520 * 72 / DPI, pdf_y(1570))
    callouts = [
        ((440, 1634), "Reference-Controlled"),
        ((1200, 1634), "Same Identity"),
        ((2530, 1634), "Pose-Driven Animation"),
    ]
    for xy, label in callouts:
        callout_font = font(32)
        text_box = raster_draw.textbbox(xy, label, font=callout_font, anchor="mm")
        dot_x = text_box[0] - 24
        raster_draw.ellipse((dot_x - 8, 1627, dot_x + 8, 1643), fill=BLUE)
        raster_draw.text(xy, label, font=callout_font, fill=BLACK, anchor="mm")
        pdf.setFillColorRGB(*(value / 255.0 for value in BLUE))
        pdf_font_size = 32 * 72 / DPI
        half_width = pdf.stringWidth(label, "Helvetica", pdf_font_size) / 2
        pdf.circle(
            xy[0] * 72 / DPI - half_width - 10 * 72 / DPI,
            pdf_y(1635),
            8 * 72 / DPI,
            fill=1,
            stroke=0,
        )
        pdf_text_center(pdf, xy, label, 32, BLACK)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.save(png_path, format="PNG", dpi=(DPI, DPI), compress_level=9)
    pdf.showPage()
    pdf.save()

    pose_records = {
        label: {
            "condition_id": condition_id,
            "pose_sha256": conditions[condition_id]["source_checksum"]["pose_sha256"],
            "camera_sha256": conditions[condition_id]["source_checksum"][
                "camera_sha256"
            ],
            "source_frame_id": conditions[condition_id]["source_frame_id"],
        }
        for label, condition_id in (("Pose A", pose_a), ("Pose B", pose_b))
    }
    manifest = {
        "schema_version": "canondressgs.paper.figure1_teaser_sources.v1",
        "task_id": TASK_ID,
        "status": "MANUAL_REVIEW_REQUIRED",
        "paper_final": False,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "new_branch": NEW_BRANCH,
        "garment_ids": list(GARMENTS),
        "identity": "subject02",
        "selection_policy": {
            "pose_pair": "maximum L2 distance over all frozen pose vectors",
            "pose_l2_distance": pose_distance,
            "method_visual_quality_used_for_selection": False,
            "reference_condition": (
                "first frozen condition not used as either target pose"
            ),
            "reference_condition_id": reference_condition,
        },
        "pose_conditions": pose_records,
        "reference_sources": reference_records,
        "base_avatar": base_record,
        "edited_results": result_records,
        "source_manifests": {
            "target_reference_manifest": {
                "source_path": f"{DATASET_ROOT}/aaai_gate_28_manifest.json",
                "cached_path": relative(repo_root, dataset_path),
                "sha256": sha256_file(dataset_path),
            },
            "formal_candidate_registry": {
                "path": relative(repo_root, formal_registry_path),
                "sha256": sha256_file(formal_registry_path),
            },
        },
        "checks": {
            "references_from_archived_reference_set": True,
            "reference_target_overlap_condition_ids": [],
            "reference_target_overlap": False,
            "all_edited_results_method": "Ours-v2",
            "forbidden_method_count": 0,
            "interpolation_source_count": 0,
            "mixed_reference_source_count": 0,
            "color_counterfactual_source_count": 0,
            "pose_ids_shared_across_garments": True,
            "pose_hashes_are_different": True,
            "identity_shared": True,
            "source_hashes_recorded": True,
            "formal_asset_mutation_count": 0,
            "new_model_render_count": 0,
            "training_count": 0,
            "evaluation_count": 0,
            "checkpoint_write_count": 0,
            "ai_generated_image_count_current_task": 0,
        },
        "allowed_pixel_operations": [
            "crop",
            "resize",
            "layout composition",
            "figure labels",
            "vector arrows",
            "vector borders",
        ],
        "forbidden_pixel_operations_used": [],
        "outputs": {
            "png": relative(repo_root, png_path),
            "png_sha256": sha256_file(png_path),
            "png_size": [WIDTH, HEIGHT],
            "png_dpi": DPI,
            "pdf": relative(repo_root, pdf_path),
            "pdf_sha256": sha256_file(pdf_path),
            "pdf_page_count": 1,
            "pdf_vector_furniture": True,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument("--png", type=Path)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    png_path = args.png or (
        repo_root / "paper_draft/aaai27/figures/figure1_canondressgs_teaser.png"
    )
    pdf_path = args.pdf or (
        repo_root / "paper_draft/aaai27/figures/figure1_canondressgs_teaser.pdf"
    )
    manifest_path = args.manifest or (
        repo_root
        / "paper_protocol/figure_manifests/figure1_canondressgs_teaser_sources.json"
    )
    manifest = build(repo_root, png_path.resolve(), pdf_path.resolve(), manifest_path.resolve())
    print(json.dumps(manifest["outputs"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
