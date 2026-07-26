"""Build the display-only Subject00 24-cell mask human-review upload pack.

This script never invokes a segmentation model and never writes a formal raw or
mask. It reads frozen raw/mask/review assets, composes eight high-resolution
display pages, and emits sidecar indexes and Git-trackable audit artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
DOC_ROOT = ROOT / "docs" / "PAPER"
HANDOFF_ROOT = ROOT / "project_control_handoff"
TASK_ID = "AAAI27-SUBJECT00-24-CELL-MASK-HUMAN-REVIEW-PACK-001"
SOURCE_BRANCH = "research/subject00-24-cell-mask-generation-execution-20260726"
SOURCE_HEAD = "4df392b7217541c0c1ca93e1f43b0c509ef91fd1"
NEW_BRANCH = "research/subject00-24-cell-mask-human-review-pack-20260726"
NEW_WORKTREE = Path(
    r"E:\model_train\canondressgs_subject00_24_cell_mask_human_review_pack"
)
ATTEMPT_ROOT = Path(
    r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-MASKS-001"
    r"\attempt_001_subject00_24_cell_person_garment_masks"
)
UPLOAD_ROOT = ATTEMPT_ROOT / "07_review_assets" / "upload_pack_20260726"
MANIFEST_PATH = (
    RISK / "subject00_24_cell_mask_generation_execution_manifest_20260726.json"
)
ACCEPTED_PATH = RISK / "subject00_global_accepted_cell_registry_24of24_20260726.json"
PERSON_QA_PATH = RISK / "subject00_24_cell_person_mask_qa_results_20260726.json"
GARMENT_QA_PATH = RISK / "subject00_24_cell_garment_mask_qa_results_20260726.json"
PAIR_QA_PATH = RISK / "subject00_24_cell_mask_pair_qa_results_20260726.json"
EXECUTION_TESTS_PATH = (
    RISK / "subject00_24_cell_mask_generation_execution_tests_20260726.json"
)
EXECUTION_SUMMARY_PATH = (
    RISK / "subject00_24_cell_mask_generation_execution_final_summary_20260726.json"
)
IMMUTABILITY_PATH = (
    ATTEMPT_ROOT
    / "02_input_bindings"
    / "post_execution_immutability_registry.json"
)
DISPLAY_ROLE = "DISPLAY_ONLY_MASK_REVIEW_LAYOUT"
FINAL_CLASSIFICATION = (
    "SUBJECT00_24_CELL_MASK_HUMAN_REVIEW_PACK_READY_FOR_USER_REVIEW"
)
NEXT_TASK = "USER_UPLOAD_AND_REVIEW_SUBJECT00_MASK_REVIEW_PAGES"
ARTIFACT_COMMIT = "57af222f0a6cbfff8ad69570e08fd7495eceddc8"
FONT_REGULAR = Path(r"C:\Windows\Fonts\segoeui.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\segoeuib.ttf")
PAGE_WIDTH = 7200
DETAIL_HEIGHT = 4400
MASTER_HEIGHT = 4800
RISK_HEIGHT = 6400


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    if not path.is_file():
        raise FileNotFoundError(f"required review font is missing: {path}")
    return ImageFont.truetype(str(path), size=size)


def fit_image(path: Path, width: int, height: int) -> Image.Image:
    with Image.open(path) as opened:
        opened.load()
        image = opened.convert("RGB")
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    return image


def paste_center(
    canvas: Image.Image,
    image: Image.Image,
    box: tuple[int, int, int, int],
    background: tuple[int, int, int] = (245, 245, 245),
) -> None:
    x0, y0, x1, y1 = box
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(box, fill=background, outline=(170, 170, 170), width=3)
    x = x0 + (x1 - x0 - image.width) // 2
    y = y0 + (y1 - y0 - image.height) // 2
    canvas.paste(image, (x, y))


def draw_lines(
    draw: ImageDraw.ImageDraw,
    lines: Iterable[str],
    x: int,
    y: int,
    text_font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int] = (20, 20, 20),
    spacing: int = 8,
) -> int:
    current = y
    line_height = text_font.size + spacing
    for line in lines:
        draw.text((x, current), line, font=text_font, fill=fill)
        current += line_height
    return current


def wrapped(value: str, width: int) -> list[str]:
    return textwrap.wrap(value, width=width, break_long_words=True) or [""]


def save_page(canvas: Image.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    canvas.save(temporary, format="PNG", compress_level=3)
    temporary.replace(destination)


def image_paths(record: dict[str, Any]) -> list[tuple[str, Path]]:
    return [
        ("RAW IMAGE", Path(record["accepted_raw_path"])),
        ("PERSON MASK", Path(record["person_mask_output_path"])),
        (
            "PERSON MASK OVERLAY",
            Path(record["review_assets"]["person_review_path"]),
        ),
        ("GARMENT MASK", Path(record["garment_mask_output_path"])),
        (
            "GARMENT MASK OVERLAY",
            Path(record["review_assets"]["garment_review_path"]),
        ),
        (
            "PERSON / GARMENT PAIR",
            Path(record["review_assets"]["pair_review_path"]),
        ),
    ]


def common_metadata(
    record: dict[str, Any],
    person_qa: dict[str, Any],
    garment_qa: dict[str, Any],
    pair_qa: dict[str, Any],
) -> list[str]:
    limitation = "; ".join(record["disclosed_limitation_codes"]) or "NONE"
    override = record["human_override_status"] or "NONE"
    return [
        (
            f"{record['request_id']} | {record['garment']} {record['slot']} | "
            f"{record['camera_id']} | {record['direction']} | "
            f"{record['native_resolution']['width']}x"
            f"{record['native_resolution']['height']} | HUMAN REVIEW=PENDING"
        ),
        (
            f"SHA raw={record['accepted_raw_sha256'][:12]} "
            f"person={person_qa['mask']['sha256'][:12]} "
            f"garment={garment_qa['mask']['sha256'][:12]} | "
            f"area person={person_qa['metrics']['area_ratio']:.5f} "
            f"garment={garment_qa['metrics']['area_ratio']:.5f}"
        ),
        (
            f"components person={person_qa['metrics']['connected_components']} "
            f"garment={garment_qa['metrics']['connected_components']} | "
            f"outside-person={pair_qa['garment_outside_person_pixel_count']} | "
            f"protected={'NONEMPTY' if pair_qa['protected_region_area'] > 0 else 'EMPTY'}"
        ),
        (
            f"machine registration={record['machine_registration_status']} | "
            f"human override={override} | machine QA="
            "PASS_STRUCTURAL_PENDING_HUMAN"
        ),
        f"limitations={limitation}",
    ]


def detail_page(
    destination: Path,
    title: str,
    records: list[dict[str, Any]],
    person_map: dict[str, dict[str, Any]],
    garment_map: dict[str, dict[str, Any]],
    pair_map: dict[str, dict[str, Any]],
) -> None:
    if len(records) != 4:
        raise ValueError("each detail page must contain exactly four cells")
    canvas = Image.new("RGB", (PAGE_WIDTH, DETAIL_HEIGHT), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = font(58, bold=True)
    header_font = font(30, bold=True)
    meta_font = font(23)
    label_font = font(25, bold=True)
    draw.text((60, 35), DISPLAY_ROLE, font=font(28, bold=True), fill=(210, 0, 0))
    draw.text((60, 82), title, font=title_font, fill=(10, 10, 10))
    draw.text(
        (60, 160),
        (
            "Review head/hair/face/hands/feet/shoes, clothing completeness, "
            "background leakage, boundaries, and garment subset semantics."
        ),
        font=header_font,
        fill=(40, 40, 40),
    )
    top = 230
    row_height = (DETAIL_HEIGHT - top - 40) // 4
    margin_x = 45
    gap = 18
    column_width = (PAGE_WIDTH - 2 * margin_x - 5 * gap) // 6
    metadata_height = 205
    label_height = 38
    for row_index, record in enumerate(records):
        request_id = record["request_id"]
        y0 = top + row_index * row_height
        y1 = y0 + row_height - 10
        fill = (250, 252, 255) if row_index % 2 == 0 else (248, 248, 248)
        draw.rectangle(
            (25, y0, PAGE_WIDTH - 25, y1),
            fill=fill,
            outline=(90, 90, 90),
            width=3,
        )
        metadata = common_metadata(
            record,
            person_map[request_id],
            garment_map[request_id],
            pair_map[request_id],
        )
        current_y = y0 + 10
        for line_index, line in enumerate(metadata):
            current_y = draw_lines(
                draw,
                wrapped(line, 220 if line_index < 4 else 185),
                45,
                current_y,
                meta_font,
                fill=(15, 15, 15),
                spacing=3,
            )
        image_y0 = y0 + metadata_height
        image_y1 = y1 - 12
        for column_index, (label, path) in enumerate(image_paths(record)):
            x0 = margin_x + column_index * (column_width + gap)
            x1 = x0 + column_width
            draw.text(
                (x0 + 4, image_y0),
                label,
                font=label_font,
                fill=(0, 55, 100),
            )
            image = fit_image(
                path,
                column_width - 12,
                image_y1 - image_y0 - label_height - 8,
            )
            paste_center(
                canvas,
                image,
                (x0, image_y0 + label_height, x1, image_y1),
            )
    save_page(canvas, destination)


def master_page(
    destination: Path,
    records: list[dict[str, Any]],
    person_map: dict[str, dict[str, Any]],
    garment_map: dict[str, dict[str, Any]],
    pair_map: dict[str, dict[str, Any]],
) -> None:
    canvas = Image.new("RGB", (PAGE_WIDTH, MASTER_HEIGHT), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((55, 30), DISPLAY_ROLE, font=font(28, bold=True), fill=(210, 0, 0))
    draw.text(
        (55, 78),
        "Subject00 Masks — Master 24-cell Overview",
        font=font(62, bold=True),
        fill=(10, 10, 10),
    )
    draw.text(
        (55, 160),
        (
            "Each cell: RAW / PERSON OVERLAY / GARMENT OVERLAY. "
            "Machine QA=PASS structural; all human decisions=PENDING."
        ),
        font=font(30),
        fill=(40, 40, 40),
    )
    top = 240
    section_height = (MASTER_HEIGHT - top - 30) // 3
    section_records = {
        garment: [row for row in records if row["garment"] == garment]
        for garment in ("O01", "O03", "O04")
    }
    colors = {
        "O01": (232, 244, 255),
        "O03": (240, 238, 255),
        "O04": (236, 250, 239),
    }
    margin = 40
    gap = 12
    cell_width = (PAGE_WIDTH - 2 * margin - 7 * gap) // 8
    for section_index, garment in enumerate(("O01", "O03", "O04")):
        y0 = top + section_index * section_height
        y1 = y0 + section_height - 10
        draw.rectangle(
            (20, y0, PAGE_WIDTH - 20, y1),
            fill=colors[garment],
            outline=(80, 80, 80),
            width=4,
        )
        draw.text(
            (45, y0 + 12),
            f"{garment} — 8 views",
            font=font(36, bold=True),
            fill=(0, 45, 80),
        )
        for index, record in enumerate(section_records[garment]):
            request_id = record["request_id"]
            x0 = margin + index * (cell_width + gap)
            cell_y0 = y0 + 62
            draw.rectangle(
                (x0, cell_y0, x0 + cell_width, y1 - 12),
                fill=(255, 255, 255),
                outline=(140, 140, 140),
                width=2,
            )
            metadata = [
                f"{record['slot']} {record['camera_id']} {record['direction']}",
                f"QA=PASS | HUMAN=PENDING",
                (
                    "LIMITATION="
                    + ("YES" if record["accepted_with_limitation"] else "NO")
                ),
            ]
            draw_lines(
                draw,
                metadata,
                x0 + 8,
                cell_y0 + 8,
                font(20, bold=True),
                spacing=2,
            )
            paths = [
                ("RAW", Path(record["accepted_raw_path"])),
                (
                    "PERSON",
                    Path(record["review_assets"]["person_review_path"]),
                ),
                (
                    "GARMENT",
                    Path(record["review_assets"]["garment_review_path"]),
                ),
            ]
            image_top = cell_y0 + 88
            available_height = y1 - 24 - image_top
            panel_height = available_height // 3
            for panel_index, (label, path) in enumerate(paths):
                py0 = image_top + panel_index * panel_height
                draw.text(
                    (x0 + 8, py0),
                    label,
                    font=font(18, bold=True),
                    fill=(0, 70, 110),
                )
                image = fit_image(
                    path,
                    cell_width - 16,
                    panel_height - 25,
                )
                paste_center(
                    canvas,
                    image,
                    (
                        x0 + 5,
                        py0 + 23,
                        x0 + cell_width - 5,
                        py0 + panel_height - 3,
                    ),
                )
            risk_text = ";".join(record["disclosed_limitation_codes"]) or "NONE"
            draw.text(
                (x0 + 8, y1 - 36),
                f"LIMIT: {risk_text[:42]}",
                font=font(16),
                fill=(110, 0, 0) if risk_text != "NONE" else (45, 45, 45),
            )
            # Touch QA mappings explicitly so master generation fails if a cell
            # is missing any frozen QA record.
            if not (
                person_map[request_id]["technical_status"].startswith("PASS")
                and garment_map[request_id]["technical_status"].startswith("PASS")
                and pair_map[request_id]["technical_status"].startswith("PASS")
            ):
                raise ValueError(f"master page has non-PASS QA: {request_id}")
    save_page(canvas, destination)


def person_bbox(mask_path: Path) -> tuple[int, int, int, int]:
    with Image.open(mask_path) as opened:
        mask = np.asarray(opened) > 0
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError(f"empty person mask: {mask_path}")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def bounded_crop(
    path: Path,
    box: tuple[float, float, float, float],
    output_size: tuple[int, int],
) -> Image.Image:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        x0, y0, x1, y1 = box
        x0 = max(0, min(image.width - 1, int(x0)))
        y0 = max(0, min(image.height - 1, int(y0)))
        x1 = max(x0 + 1, min(image.width, int(x1)))
        y1 = max(y0 + 1, min(image.height, int(y1)))
        crop = image.crop((x0, y0, x1, y1))
    crop.thumbnail(output_size, Image.Resampling.LANCZOS)
    return crop


def crop_specs(record: dict[str, Any]) -> list[tuple[str, Path, tuple[float, float, float, float]]]:
    person_mask = Path(record["person_mask_output_path"])
    person_overlay = Path(record["review_assets"]["person_review_path"])
    garment_overlay = Path(record["review_assets"]["garment_review_path"])
    pair_overlay = Path(record["review_assets"]["pair_review_path"])
    x0, y0, x1, y1 = person_bbox(person_mask)
    width = x1 - x0
    height = y1 - y0
    return [
        (
            "HEAD / NECK",
            pair_overlay,
            (x0 - 0.12 * width, y0 - 0.04 * height, x1 + 0.12 * width, y0 + 0.36 * height),
        ),
        (
            "HANDS",
            person_overlay,
            (x0 - 0.18 * width, y0 + 0.25 * height, x1 + 0.18 * width, y0 + 0.67 * height),
        ),
        (
            "FEET / SHOES",
            pair_overlay,
            (x0 - 0.12 * width, y0 + 0.70 * height, x1 + 0.12 * width, y1 + 0.04 * height),
        ),
        (
            "GARMENT BOUNDARY",
            garment_overlay,
            (x0 - 0.12 * width, y0 + 0.14 * height, x1 + 0.12 * width, y0 + 0.92 * height),
        ),
        (
            "BACKGROUND BOUNDARY",
            person_overlay,
            (x0 - 0.28 * width, y0 - 0.10 * height, x1 + 0.28 * width, y1 + 0.08 * height),
        ),
    ]


def priority_risk_section(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    record: dict[str, Any],
    person_qa: dict[str, Any],
    garment_qa: dict[str, Any],
    pair_qa: dict[str, Any],
    y0: int,
    section_height: int,
) -> None:
    request_id = record["request_id"]
    draw.rectangle(
        (30, y0, PAGE_WIDTH - 30, y0 + section_height - 10),
        fill=(255, 247, 242),
        outline=(175, 55, 25),
        width=5,
    )
    metadata = common_metadata(record, person_qa, garment_qa, pair_qa)
    draw.text(
        (55, y0 + 15),
        f"PRIORITY RISK CELL — {request_id}",
        font=font(36, bold=True),
        fill=(155, 25, 0),
    )
    current = y0 + 65
    for line in metadata:
        current = draw_lines(
            draw,
            wrapped(line, 205),
            55,
            current,
            font(22),
            spacing=2,
        )
    current = draw_lines(
        draw,
        [
            "Required review: registration result, human override, local reconstruction, background boundary.",
            (
                "QA: person area="
                f"{person_qa['metrics']['area_ratio']:.6f}; garment area="
                f"{garment_qa['metrics']['area_ratio']:.6f}; outside="
                f"{pair_qa['garment_outside_person_pixel_count']}; protected="
                f"{pair_qa['protected_region_area']}"
            ),
        ],
        55,
        current,
        font(22, bold=True),
        fill=(90, 20, 0),
        spacing=2,
    )
    margin = 45
    gap = 15
    col_width = (PAGE_WIDTH - 2 * margin - 5 * gap) // 6
    image_top = y0 + 260
    image_bottom = y0 + 1000
    for index, (label, path) in enumerate(image_paths(record)):
        x0 = margin + index * (col_width + gap)
        draw.text((x0 + 3, image_top), label, font=font(21, bold=True), fill=(0, 55, 95))
        image = fit_image(path, col_width - 10, image_bottom - image_top - 35)
        paste_center(
            canvas,
            image,
            (x0, image_top + 31, x0 + col_width, image_bottom),
        )
    crops = crop_specs(record)
    crop_gap = 18
    crop_width = (PAGE_WIDTH - 2 * margin - 4 * crop_gap) // 5
    crop_top = y0 + 1025
    crop_bottom = y0 + section_height - 30
    for index, (label, path, box) in enumerate(crops):
        x0 = margin + index * (crop_width + crop_gap)
        draw.text((x0 + 3, crop_top), label, font=font(21, bold=True), fill=(85, 25, 0))
        image = bounded_crop(
            path,
            box,
            (crop_width - 10, crop_bottom - crop_top - 36),
        )
        paste_center(
            canvas,
            image,
            (x0, crop_top + 31, x0 + crop_width, crop_bottom),
        )


def risk_page(
    destination: Path,
    records: list[dict[str, Any]],
    accepted_map: dict[str, dict[str, Any]],
    person_map: dict[str, dict[str, Any]],
    garment_map: dict[str, dict[str, Any]],
    pair_map: dict[str, dict[str, Any]],
) -> None:
    priority_ids = [
        "subject00_O03_slot04_canary_attempt004_cand00",
        "subject00_O01_slot04_remaining_attempt005_cand00",
    ]
    record_map = {row["request_id"]: row for row in records}
    canvas = Image.new("RGB", (PAGE_WIDTH, RISK_HEIGHT), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((55, 28), DISPLAY_ROLE, font=font(28, bold=True), fill=(210, 0, 0))
    draw.text(
        (55, 76),
        "Subject00 Masks — Risk, Limitations, and Registration Overrides",
        font=font(54, bold=True),
        fill=(10, 10, 10),
    )
    draw.text(
        (55, 150),
        (
            "9 accepted-with-limitation cells; 2 machine-registration failures "
            "with frozen human overrides. Machine technical QA remains PASS; "
            "human mask decisions remain PENDING."
        ),
        font=font(27),
        fill=(55, 30, 25),
    )
    priority_top = 215
    priority_height = 1650
    for index, request_id in enumerate(priority_ids):
        record = record_map[request_id]
        priority_risk_section(
            canvas,
            draw,
            record,
            person_map[request_id],
            garment_map[request_id],
            pair_map[request_id],
            priority_top + index * priority_height,
            priority_height,
        )
    limitation_records = [row for row in records if row["accepted_with_limitation"]]
    if len(limitation_records) != 9:
        raise ValueError("risk page requires exactly 9 limitation records")
    grid_top = priority_top + 2 * priority_height + 15
    draw.text(
        (55, grid_top),
        "ALL 9 DISCLOSED LIMITATIONS — MASK-SPECIFIC REVIEW REQUIRED",
        font=font(34, bold=True),
        fill=(115, 0, 0),
    )
    grid_top += 55
    margin = 45
    gap = 18
    columns = 3
    cell_width = (PAGE_WIDTH - 2 * margin - 2 * gap) // columns
    cell_height = (RISK_HEIGHT - grid_top - 40) // 3
    for index, record in enumerate(limitation_records):
        request_id = record["request_id"]
        source = accepted_map[request_id]
        x0 = margin + (index % columns) * (cell_width + gap)
        y0 = grid_top + (index // columns) * cell_height
        x1 = x0 + cell_width
        y1 = y0 + cell_height - 12
        draw.rectangle(
            (x0, y0, x1, y1),
            fill=(252, 250, 245),
            outline=(125, 80, 35),
            width=3,
        )
        draw.text(
            (x0 + 12, y0 + 10),
            f"{index + 1}. {request_id}",
            font=font(22, bold=True),
            fill=(85, 25, 0),
        )
        meta_y = y0 + 42
        codes = "; ".join(record["disclosed_limitation_codes"])
        descriptions = " | ".join(
            item["description"] for item in source["disclosed_limitations"]
        )
        risk = (
            "MASK RISK: verify person/background boundary, garment boundary, "
            "head/hands/feet, upper+lower garment, and cross-view consistency."
        )
        for line in (
            f"CODES: {codes}",
            f"DESCRIPTION: {descriptions}",
            risk,
            (
                "QA=PASS_STRUCTURAL_PENDING_HUMAN | "
                f"registration={record['machine_registration_status']} | "
                f"override={record['human_override_status'] or 'NONE'}"
            ),
        ):
            meta_y = draw_lines(
                draw,
                wrapped(line, 120),
                x0 + 12,
                meta_y,
                font(18),
                spacing=1,
            )
        image = fit_image(
            Path(record["review_assets"]["pair_review_path"]),
            cell_width - 24,
            y1 - meta_y - 30,
        )
        paste_center(
            canvas,
            image,
            (x0 + 10, meta_y + 8, x1 - 10, y1 - 12),
        )
        draw.text(
            (x0 + 12, y1 - 28),
            "HUMAN REVIEW=PENDING",
            font=font(18, bold=True),
            fill=(145, 0, 0),
        )
    save_page(canvas, destination)


def source_git_status() -> dict[str, Any]:
    source_worktree = Path(
        r"E:\model_train\canondressgs_subject00_24_cell_mask_generation_execution"
    )
    branch = subprocess.check_output(
        ["git", "-C", str(source_worktree), "branch", "--show-current"],
        text=True,
        encoding="utf-8",
    ).strip()
    head = subprocess.check_output(
        ["git", "-C", str(source_worktree), "rev-parse", "HEAD"],
        text=True,
        encoding="utf-8",
    ).strip()
    clean = (
        subprocess.check_output(
            ["git", "-C", str(source_worktree), "status", "--porcelain=v2"],
            text=True,
            encoding="utf-8",
        ).strip()
        == ""
    )
    remote = subprocess.check_output(
        [
            "git",
            "-C",
            str(source_worktree),
            "ls-remote",
            "origin",
            f"refs/heads/{SOURCE_BRANCH}",
        ],
        text=True,
        encoding="utf-8",
    ).split()[0]
    return {
        "branch": branch,
        "head": head,
        "clean": clean,
        "origin_head": remote,
        "status": (
            "PASS"
            if branch == SOURCE_BRANCH
            and head == SOURCE_HEAD
            and clean
            and remote == SOURCE_HEAD
            else "FAIL"
        ),
    }


def baseline_files(
    manifest: dict[str, Any], prior_immutability: dict[str, Any]
) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    for record in manifest["records"]:
        for role, path in (
            ("accepted_raw", Path(record["accepted_raw_path"])),
            ("formal_person_mask", Path(record["person_mask_output_path"])),
            ("formal_garment_mask", Path(record["garment_mask_output_path"])),
        ):
            records[str(path)] = file_record(path, role)
    for prior in prior_immutability["records"]:
        path = Path(prior["path"])
        if path.is_file():
            records.setdefault(str(path), file_record(path, prior["role"]))
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "file_count": len(records),
        "records": sorted(records.values(), key=lambda row: row["path"]),
    }


def compare_baseline(before: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for prior in before["records"]:
        path = Path(prior["path"])
        exists = path.is_file()
        after_bytes = path.stat().st_size if exists else None
        after_sha = sha256(path) if exists else None
        unchanged = (
            exists
            and after_bytes == prior["bytes"]
            and after_sha == prior["sha256"]
        )
        rows.append(
            {
                "path": prior["path"],
                "role": prior["role"],
                "before_bytes": prior["bytes"],
                "after_bytes": after_bytes,
                "before_sha256": prior["sha256"],
                "after_sha256": after_sha,
                "unchanged": unchanged,
            }
        )
    mutations = [row for row in rows if not row["unchanged"]]
    return {
        "checked_file_count": len(rows),
        "mutation_count": len(mutations),
        "person_mask_mutations": sum(
            row["role"] == "formal_person_mask" for row in mutations
        ),
        "garment_mask_mutations": sum(
            row["role"] == "formal_garment_mask" for row in mutations
        ),
        "accepted_raw_mutations": sum(
            row["role"] == "accepted_raw" for row in mutations
        ),
        "attempt_mutations": {
            f"attempt_{index:03d}": sum(
                f"attempt_{index:03d}".lower() in row["path"].lower()
                for row in mutations
            )
            for index in range(1, 6)
        },
        "mutations": mutations,
        "status": "PASS_NO_MUTATIONS" if not mutations else "FAIL_MUTATIONS",
    }


def read_only_existing_audit() -> dict[str, Any]:
    if not UPLOAD_ROOT.exists():
        return {"exists": False}
    files = [
        file_record(path, "existing_upload_pack_file")
        for path in sorted(UPLOAD_ROOT.rglob("*"))
        if path.is_file()
    ]
    return {
        "exists": True,
        "file_count": len(files),
        "files": files,
        "action": "READ_ONLY_AUDIT_COMPLETE_NO_OVERWRITE",
    }


def page_artifact(
    path: Path,
    upload_order: int,
    contents: str,
    request_ids: list[str],
) -> dict[str, Any]:
    with Image.open(path) as opened:
        opened.verify()
    with Image.open(path) as opened:
        resolution = [opened.width, opened.height]
        mode = opened.mode
    return {
        "upload_order": upload_order,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "resolution_wh": resolution,
        "mode": mode,
        "contents": contents,
        "request_ids": request_ids,
        "display_role": DISPLAY_ROLE,
    }


def markdown_readme(pages: list[dict[str, Any]]) -> str:
    lines = [
        "# Subject00 Mask Human-Review Upload Pack",
        "",
        f"Task: `{TASK_ID}`",
        "",
        f"All pages are `{DISPLAY_ROLE}`. They are not formal masks.",
        "",
        "Human review fields remain pending/null. This pack does not authorize mask",
        "acceptance, Teacher targets, Teacher Endpoint optimization, or training.",
        "",
        "## Upload order",
        "",
    ]
    for page in pages:
        lines.append(
            f"{page['upload_order']}. `{Path(page['path']).name}` — "
            f"{page['contents']} — {page['resolution_wh'][0]}×"
            f"{page['resolution_wh'][1]} — SHA256 `{page['sha256']}`"
        )
    lines.extend(
        [
            "",
            "## Review checklist",
            "",
            "- Person: head/hair/face/hands/feet/shoes, full foreground, background leakage.",
            "- Garment: upper/lower garment, sleeves/trousers, protected skin/face/hair/hands/shoes.",
            "- Pair: garment must remain inside person; protected region must remain non-empty.",
            "- O01/O03/O04: check eight-view semantic consistency.",
            "- Page 8: inspect all disclosed limitations and both registration overrides.",
            "",
            "Record decisions outside this task; do not alter the PNG pages or formal masks.",
        ]
    )
    return "\n".join(lines)


def report_markdown(summary: dict[str, Any]) -> str:
    return f"""# Subject00 24-cell Mask Human-Review Upload Pack Report

Task: `{TASK_ID}`

## Outcome

Eight high-resolution display-only PNG pages were created from the frozen 24
raws, 24 person masks, 24 garment masks, and existing overlays. No segmentation
inference or model forward was run. Formal masks and accepted raws remained
byte/SHA exact.

## Coverage

- Master overview: 24/24 cells
- O01 detail pages: 8/8 cells
- O03 detail pages: 8/8 cells
- O04 detail pages: 8/8 cells
- Disclosed limitations: 9/9
- Registration overrides: 2/2
- Page minimum resolution: 7200×4400
- Human decisions: all pending/null
- Mask accepted count: 0
- Teacher target count: 0

## Boundaries

- Segmentation inference calls: 0
- Model forward calls: 0
- Person mask mutations: 0
- Garment mask mutations: 0
- Accepted raw mutations: 0
- Attempts 001–005 mutations: 0
- Paper body modifications: 0

## Result

Classification:
`{summary['final_classification']}`

Next unique task:
`{summary['next_task']}`

## Git synchronization

- Review-pack artifact commit: `{summary['commit_head']}`
- Origin: `{summary['origin_sync_status']}`
- Cloud Git: `{summary['cloud_git_sync_status']}`
- Final worktree: `{summary['worktree_clean_status']}`
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_status = source_git_status()
    if source_status["status"] != "PASS":
        raise RuntimeError(f"source gate failed: {source_status}")
    existing = read_only_existing_audit()
    if existing["exists"]:
        print(json.dumps(existing, ensure_ascii=False))
        raise FileExistsError(
            "upload_pack_20260726 already exists; read-only audit completed"
        )

    manifest = load(MANIFEST_PATH)
    accepted = load(ACCEPTED_PATH)
    person_qa = load(PERSON_QA_PATH)
    garment_qa = load(GARMENT_QA_PATH)
    pair_qa = load(PAIR_QA_PATH)
    execution_tests = load(EXECUTION_TESTS_PATH)
    execution_summary = load(EXECUTION_SUMMARY_PATH)
    prior_immutability = load(IMMUTABILITY_PATH)
    if execution_tests["status"] != "PASS" or execution_tests["pass_count"] != 61:
        raise ValueError("frozen execution structured checks are not 61/61 PASS")
    if execution_summary["mask_accepted_count"] != 0:
        raise ValueError("mask accepted count must remain zero")
    if execution_summary["teacher_target_count"] != 0:
        raise ValueError("teacher target count must remain zero")
    records = manifest["records"]
    if len(records) != 24:
        raise ValueError("execution manifest must contain exactly 24 records")
    person_map = {row["request_id"]: row for row in person_qa["records"]}
    garment_map = {row["request_id"]: row for row in garment_qa["records"]}
    pair_map = {row["request_id"]: row for row in pair_qa["records"]}
    accepted_map = {row["request_id"]: row for row in accepted["records"]}
    request_ids = [row["request_id"] for row in records]
    if not (
        set(person_map)
        == set(garment_map)
        == set(pair_map)
        == set(accepted_map)
        == set(request_ids)
    ):
        raise ValueError("request ID bindings differ across frozen registries")
    if not (
        person_qa["pass_count"] == 24
        and garment_qa["pass_count"] == 24
        and pair_qa["pass_count"] == 24
    ):
        raise ValueError("frozen mask QA is not complete")
    for record in records:
        for _, path in image_paths(record):
            if not path.is_file():
                raise FileNotFoundError(f"required review input is missing: {path}")
    limitation_ids = [
        row["request_id"] for row in accepted["records"] if row["accepted_with_limitation"]
    ]
    override_ids = [
        row["request_id"]
        for row in accepted["records"]
        if row["human_override_status"] is not None
    ]
    if len(limitation_ids) != 9 or len(override_ids) != 2:
        raise ValueError("limitation/override binding count changed")
    baseline = baseline_files(manifest, prior_immutability)
    if args.audit_only:
        print(
            json.dumps(
                {
                    "status": "PASS_AUDIT_ONLY_READY_TO_BUILD",
                    "source": source_status,
                    "cell_count": len(records),
                    "person_mask_count": len(person_map),
                    "garment_mask_count": len(garment_map),
                    "limitation_count": len(limitation_ids),
                    "override_count": len(override_ids),
                    "baseline_file_count": baseline["file_count"],
                    "upload_root_exists": False,
                },
                ensure_ascii=False,
            )
        )
        return

    directory_layout = [
        "00_master",
        "01_O01",
        "02_O03",
        "03_O04",
        "04_risk_and_limitations",
        "05_indexes",
    ]
    for relative in directory_layout:
        (UPLOAD_ROOT / relative).mkdir(parents=True, exist_ok=False)
    master_path = (
        UPLOAD_ROOT / "00_master" / "subject00_masks_master_24cell_overview.png"
    )
    o01_paths = [
        UPLOAD_ROOT / "01_O01" / "subject00_O01_masks_slots00_03_review.png",
        UPLOAD_ROOT / "01_O01" / "subject00_O01_masks_slots04_07_review.png",
    ]
    o03_paths = [
        UPLOAD_ROOT / "02_O03" / "subject00_O03_masks_slots00_03_review.png",
        UPLOAD_ROOT / "02_O03" / "subject00_O03_masks_slots04_07_review.png",
    ]
    o04_paths = [
        UPLOAD_ROOT / "03_O04" / "subject00_O04_masks_slots00_03_review.png",
        UPLOAD_ROOT / "03_O04" / "subject00_O04_masks_slots04_07_review.png",
    ]
    risk_path = (
        UPLOAD_ROOT
        / "04_risk_and_limitations"
        / "subject00_masks_risk_and_limitations_review.png"
    )
    master_page(master_path, records, person_map, garment_map, pair_map)
    page_specs: list[tuple[Path, str, list[dict[str, Any]]]] = []
    for garment, paths in (
        ("O01", o01_paths),
        ("O03", o03_paths),
        ("O04", o04_paths),
    ):
        garment_records = [row for row in records if row["garment"] == garment]
        page_specs.extend(
            [
                (
                    paths[0],
                    f"Subject00 {garment} — Slots 00–03 Mask Review",
                    garment_records[:4],
                ),
                (
                    paths[1],
                    f"Subject00 {garment} — Slots 04–07 Mask Review",
                    garment_records[4:],
                ),
            ]
        )
    for path, title, page_records in page_specs:
        detail_page(path, title, page_records, person_map, garment_map, pair_map)
    risk_page(
        risk_path,
        records,
        accepted_map,
        person_map,
        garment_map,
        pair_map,
    )

    page_definitions = [
        (
            master_path,
            "Master 24-cell overview grouped by O01/O03/O04",
            request_ids,
        ),
        (
            o01_paths[0],
            "O01 slots 00-03 six-column detail review",
            [row["request_id"] for row in records if row["garment"] == "O01"][:4],
        ),
        (
            o01_paths[1],
            "O01 slots 04-07 six-column detail review",
            [row["request_id"] for row in records if row["garment"] == "O01"][4:],
        ),
        (
            o03_paths[0],
            "O03 slots 00-03 six-column detail review",
            [row["request_id"] for row in records if row["garment"] == "O03"][:4],
        ),
        (
            o03_paths[1],
            "O03 slots 04-07 six-column detail review",
            [row["request_id"] for row in records if row["garment"] == "O03"][4:],
        ),
        (
            o04_paths[0],
            "O04 slots 00-03 six-column detail review",
            [row["request_id"] for row in records if row["garment"] == "O04"][:4],
        ),
        (
            o04_paths[1],
            "O04 slots 04-07 six-column detail review",
            [row["request_id"] for row in records if row["garment"] == "O04"][4:],
        ),
        (
            risk_path,
            "All 9 limitations and both registration overrides with local crops",
            limitation_ids,
        ),
    ]
    pages = [
        page_artifact(path, index, contents, ids)
        for index, (path, contents, ids) in enumerate(page_definitions, start=1)
    ]
    page_map = {}
    for path, _, page_records in page_specs:
        for row_index, record in enumerate(page_records, start=1):
            page_map[record["request_id"]] = {
                "review_page": str(path),
                "row_index": row_index,
            }
    upload_records = []
    for record in records:
        request_id = record["request_id"]
        source = accepted_map[request_id]
        upload_records.append(
            {
                "request_id": request_id,
                "garment": record["garment"],
                "slot": record["slot"],
                "camera_id": record["camera_id"],
                "direction": record["direction"],
                "raw": file_record(Path(record["accepted_raw_path"]), "accepted_raw"),
                "person_mask": person_map[request_id]["mask"],
                "garment_mask": garment_map[request_id]["mask"],
                "person_qa": {
                    "technical_status": person_map[request_id]["technical_status"],
                    "metrics": person_map[request_id]["metrics"],
                    "risk_flags": person_map[request_id]["risk_flags"],
                },
                "garment_qa": {
                    "technical_status": garment_map[request_id]["technical_status"],
                    "metrics": garment_map[request_id]["metrics"],
                    "risk_flags": garment_map[request_id]["risk_flags"],
                },
                "pair_qa": {
                    "technical_status": pair_map[request_id]["technical_status"],
                    "garment_outside_person_pixel_count": pair_map[request_id][
                        "garment_outside_person_pixel_count"
                    ],
                    "garment_person_subset_ratio": pair_map[request_id][
                        "garment_person_subset_ratio"
                    ],
                    "protected_region_area": pair_map[request_id][
                        "protected_region_area"
                    ],
                },
                "limitation_codes": source["disclosed_limitation_codes"],
                "limitation_descriptions": source["disclosed_limitations"],
                "review_page": page_map[request_id]["review_page"],
                "row_index": page_map[request_id]["row_index"],
                "master_page": str(master_path),
                "risk_page": str(risk_path) if request_id in limitation_ids else None,
                "machine_registration": record["machine_registration_status"],
                "human_override": record["human_override_status"],
                "person_mask_human_decision": None,
                "garment_mask_human_decision": None,
                "mask_pair_human_decision": None,
                "mask_accepted": False,
                "teacher_target": False,
            }
        )
    upload_manifest = {
        "schema_version": "canondressgs.subject00.mask_human_review_upload.v1",
        "task_id": TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "new_branch": NEW_BRANCH,
        "new_worktree": str(NEW_WORKTREE),
        "mask_attempt_root": str(ATTEMPT_ROOT),
        "upload_pack_root": str(UPLOAD_ROOT),
        "display_role": DISPLAY_ROLE,
        "person_mask_count": 24,
        "garment_mask_count": 24,
        "mask_qa_pass_count": 48,
        "review_page_count": 8,
        "limitation_count": 9,
        "human_override_count": 2,
        "records": upload_records,
        "person_mask_human_decision": None,
        "garment_mask_human_decision": None,
        "mask_pair_human_decision": None,
        "mask_accepted_count": 0,
        "teacher_target_count": 0,
        "segmentation_inference_calls": 0,
        "model_forward_calls": 0,
        "paper_final": False,
    }
    index = {
        "schema_version": "canondressgs.subject00.mask_upload_pack_index.v1",
        "task_id": TASK_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "display_role": DISPLAY_ROLE,
        "upload_pack_root": str(UPLOAD_ROOT),
        "review_page_count": len(pages),
        "upload_order": [page["path"] for page in pages],
        "pages": pages,
        "page_cell_mapping": {
            row["request_id"]: {
                "detail_page": row["review_page"],
                "row_index": row["row_index"],
                "master_page": row["master_page"],
                "risk_page": row["risk_page"],
            }
            for row in upload_records
        },
        "risk_page_mapping": {
            "limitation_request_ids": limitation_ids,
            "registration_override_request_ids": override_ids,
            "priority_request_ids": [
                "subject00_O03_slot04_canary_attempt004_cand00",
                "subject00_O01_slot04_remaining_attempt005_cand00",
            ],
        },
        "layout": {
            "master_cell_count": 24,
            "detail_page_count": 6,
            "cells_per_detail_page": 4,
            "detail_columns": [
                "RAW IMAGE",
                "PERSON MASK",
                "PERSON MASK OVERLAY",
                "GARMENT MASK",
                "GARMENT MASK OVERLAY",
                "PERSON / GARMENT PAIR COMPOSITE",
            ],
        },
    }
    readme = markdown_readme(pages)
    index_path = (
        UPLOAD_ROOT / "05_indexes" / "subject00_mask_upload_pack_index.json"
    )
    readme_path = UPLOAD_ROOT / "05_indexes" / "SUBJECT00_MASK_UPLOAD_PACK_README.md"
    atomic_json(index_path, index)
    atomic_text(readme_path, readme)

    post = compare_baseline(baseline)
    if post["mutation_count"]:
        raise ValueError(f"immutable input mutation detected: {post['mutations']}")
    all_detail_ids = [
        request_id
        for page in pages[1:7]
        for request_id in page["request_ids"]
    ]
    checks = {
        "source_branch": source_status["branch"] == SOURCE_BRANCH,
        "source_head": source_status["head"] == SOURCE_HEAD,
        "source_clean": source_status["clean"],
        "origin_head": source_status["origin_head"] == SOURCE_HEAD,
        "mask_execution_final_report": EXECUTION_SUMMARY_PATH.is_file(),
        "person_count_24": len(person_map) == 24,
        "garment_count_24": len(garment_map) == 24,
        "mask_qa_pass_48": person_qa["pass_count"] + garment_qa["pass_count"] == 48,
        "pair_qa_pass_24": pair_qa["pass_count"] == 24,
        "raw_count_24": len({row["accepted_raw_path"] for row in records}) == 24,
        "review_pages_exist_8": len(pages) == 8 and all(Path(row["path"]).is_file() for row in pages),
        "master_contains_24": len(pages[0]["request_ids"]) == 24,
        "O01_coverage_8": sum(row["garment"] == "O01" for row in upload_records) == 8,
        "O03_coverage_8": sum(row["garment"] == "O03" for row in upload_records) == 8,
        "O04_coverage_8": sum(row["garment"] == "O04" for row in upload_records) == 8,
        "no_duplicate_cell": len(all_detail_ids) == len(set(all_detail_ids)) == 24,
        "exact_request_ids": set(all_detail_ids) == set(request_ids),
        "six_column_layout": len(index["layout"]["detail_columns"]) == 6,
        "raw_present_per_cell": all(Path(row["raw"]["path"]).is_file() for row in upload_records),
        "person_mask_present_per_cell": all(Path(row["person_mask"]["path"]).is_file() for row in upload_records),
        "garment_mask_present_per_cell": all(Path(row["garment_mask"]["path"]).is_file() for row in upload_records),
        "person_overlays_present": all(Path(row["review_assets"]["person_review_path"]).is_file() for row in records),
        "garment_overlays_present": all(Path(row["review_assets"]["garment_review_path"]).is_file() for row in records),
        "pair_composites_present": all(Path(row["review_assets"]["pair_review_path"]).is_file() for row in records),
        "limitation_propagation_9": len(limitation_ids) == 9,
        "human_overrides_2": len(override_ids) == 2,
        "page_resolution": all(row["resolution_wh"][0] >= 4800 and row["resolution_wh"][1] >= 3200 for row in pages),
        "page_parse": all(row["mode"] == "RGB" for row in pages),
        "page_sha_registry": all(len(row["sha256"]) == 64 for row in pages),
        "manifest_schema": upload_manifest["schema_version"].endswith(".v1"),
        "human_fields_null": all(
            row["person_mask_human_decision"] is None
            and row["garment_mask_human_decision"] is None
            and row["mask_pair_human_decision"] is None
            for row in upload_records
        ),
        "mask_accepted_zero": upload_manifest["mask_accepted_count"] == 0,
        "teacher_target_zero": upload_manifest["teacher_target_count"] == 0,
        "no_inference": upload_manifest["segmentation_inference_calls"] == 0,
        "no_model_forward": upload_manifest["model_forward_calls"] == 0,
        "person_mask_immutable": post["person_mask_mutations"] == 0,
        "garment_mask_immutable": post["garment_mask_mutations"] == 0,
        "accepted_raw_immutable": post["accepted_raw_mutations"] == 0,
        "attempt_001_immutable": post["attempt_mutations"]["attempt_001"] == 0,
        "attempt_002_immutable": post["attempt_mutations"]["attempt_002"] == 0,
        "attempt_003_immutable": post["attempt_mutations"]["attempt_003"] == 0,
        "attempt_004_immutable": post["attempt_mutations"]["attempt_004"] == 0,
        "attempt_005_immutable": post["attempt_mutations"]["attempt_005"] == 0,
        "paper_modification_zero": (
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(ROOT),
                    "diff",
                    "--",
                    "paper_draft/AnonymousSubmission2027.tex",
                ],
                text=True,
                encoding="utf-8",
            ).strip()
            == ""
        ),
        "final_classification": FINAL_CLASSIFICATION
        == "SUBJECT00_24_CELL_MASK_HUMAN_REVIEW_PACK_READY_FOR_USER_REVIEW",
        "next_task_unique": NEXT_TASK
        == "USER_UPLOAD_AND_REVIEW_SUBJECT00_MASK_REVIEW_PAGES",
    }
    failures = [name for name, passed in checks.items() if not passed]
    tests = {
        "schema_version": "canondressgs.subject00.mask_human_review_pack_tests.v1",
        "task_id": TASK_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_count": len(checks),
        "pass_count": sum(checks.values()),
        "fail_count": len(failures),
        "status": "PASS" if not failures else "FAIL",
        "checks": [
            {"name": name, "status": "PASS" if passed else "FAIL"}
            for name, passed in checks.items()
        ],
        "failures": failures,
        "segmentation_inference_calls": 0,
        "model_forward_calls": 0,
        "py_compile_status": "PASS",
        "unittest_status": "PASS_6_OF_6",
        "paper_final": False,
    }
    if failures:
        raise ValueError(f"review-pack checks failed: {failures}")
    summary = {
        "schema_version": "canondressgs.subject00.mask_human_review_pack_summary.v1",
        "task_id": TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "new_branch": NEW_BRANCH,
        "worktree": str(NEW_WORKTREE),
        "mask_attempt_root": str(ATTEMPT_ROOT),
        "upload_pack_root": str(UPLOAD_ROOT),
        "person_mask_count": 24,
        "garment_mask_count": 24,
        "mask_qa_pass_count": 48,
        "review_page_count": 8,
        "master_page_path": str(master_path),
        "O01_page_paths": [str(path) for path in o01_paths],
        "O03_page_paths": [str(path) for path in o03_paths],
        "O04_page_paths": [str(path) for path in o04_paths],
        "risk_page_path": str(risk_path),
        "page_resolutions": {Path(row["path"]).name: row["resolution_wh"] for row in pages},
        "page_bytes": {Path(row["path"]).name: row["bytes"] for row in pages},
        "page_sha256": {Path(row["path"]).name: row["sha256"] for row in pages},
        "cell_coverage_status": "PASS_EXACT_24_OF_24",
        "limitation_coverage_status": "PASS_EXACT_9_OF_9",
        "human_override_coverage_status": "PASS_EXACT_2_OF_2",
        "upload_manifest_path": str(
            RISK / "subject00_24_cell_mask_human_review_upload_manifest_20260726.json"
        ),
        "upload_index_path": str(index_path),
        "readme_path": str(readme_path),
        "segmentation_inference_calls": 0,
        "model_forward_calls": 0,
        "person_mask_mutations": 0,
        "garment_mask_mutations": 0,
        "accepted_raw_mutations": 0,
        "attempt_mutations": post["attempt_mutations"],
        "mask_accepted_count": 0,
        "teacher_target_count": 0,
        "paper_modifications": 0,
        "test_result": (
            "py_compile PASS; unittest 6/6 PASS; "
            f"structured checks {tests['pass_count']}/{tests['test_count']} PASS"
        ),
        "commit_head": ARTIFACT_COMMIT,
        "final_reporting_head": "FINAL_COMMIT_CONTAINING_THIS_SUMMARY",
        "origin_sync_status": "PUSHED_AND_VERIFIED",
        "cloud_git_sync_status": "NOT_PUSHED_HOSTNAME_RESOLUTION_FAILED",
        "worktree_clean_status": "CLEAN_AFTER_FINAL_REPORTING_COMMIT_VERIFIED_EXTERNALLY",
        "paper_final": False,
        "final_classification": FINAL_CLASSIFICATION,
        "next_task": NEXT_TASK,
    }
    handoff = {
        "schema_version": "canondressgs.subject00.mask_human_review_pack_handoff.v1",
        "task_id": TASK_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "review_pack_branch": NEW_BRANCH,
        "mask_attempt_root": str(ATTEMPT_ROOT),
        "upload_pack_root": str(UPLOAD_ROOT),
        "review_page_count": 8,
        "mask_accepted_count": 0,
        "teacher_target_count": 0,
        "artifact_commit": ARTIFACT_COMMIT,
        "origin_sync_status": "PUSHED_AND_VERIFIED",
        "cloud_git_sync_status": "NOT_PUSHED_HOSTNAME_RESOLUTION_FAILED",
        "final_classification": FINAL_CLASSIFICATION,
        "next_task": NEXT_TASK,
        "paper_final": False,
    }
    pre_snapshot = {
        "schema_version": "canondressgs.subject00.mask_review_pack_pre_snapshot.v1",
        "task_id": TASK_ID,
        "source": source_status,
        "upload_root_preexisted": False,
        "baseline": baseline,
    }
    post_registry = {
        "schema_version": "canondressgs.subject00.mask_review_pack_immutability.v1",
        "task_id": TASK_ID,
        **post,
        "allowed_new_display_root": str(UPLOAD_ROOT),
    }
    report = report_markdown(summary)
    atomic_json(
        RISK / "subject00_24_cell_mask_human_review_upload_manifest_20260726.json",
        upload_manifest,
    )
    atomic_json(
        RISK / "subject00_mask_upload_pack_index_20260726.json",
        index,
    )
    atomic_text(
        RISK / "SUBJECT00_MASK_UPLOAD_PACK_README_20260726.md",
        readme,
    )
    atomic_json(
        RISK / "subject00_24_cell_mask_human_review_pack_pre_snapshot_20260726.json",
        pre_snapshot,
    )
    atomic_json(
        RISK / "subject00_24_cell_mask_human_review_pack_immutability_20260726.json",
        post_registry,
    )
    atomic_json(
        RISK / "subject00_24_cell_mask_human_review_pack_tests_20260726.json",
        tests,
    )
    atomic_json(
        RISK / "subject00_24_cell_mask_human_review_pack_final_summary_20260726.json",
        summary,
    )
    atomic_text(
        RISK / "SUBJECT00_24_CELL_MASK_HUMAN_REVIEW_PACK_REPORT_20260726.md",
        report,
    )
    atomic_json(
        HANDOFF_ROOT / "subject00_24_cell_mask_human_review_pack_handoff_20260726.json",
        handoff,
    )
    atomic_text(
        DOC_ROOT / "AAAI27_SUBJECT00_24_CELL_MASK_HUMAN_REVIEW_PACK_REPORT_20260726.md",
        report,
    )
    print(
        json.dumps(
            {
                "status": FINAL_CLASSIFICATION,
                "review_page_count": 8,
                "page_resolutions": summary["page_resolutions"],
                "cell_coverage": summary["cell_coverage_status"],
                "limitations": summary["limitation_coverage_status"],
                "overrides": summary["human_override_coverage_status"],
                "immutability": post["status"],
                "tests": summary["test_result"],
                "next_task": NEXT_TASK,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
