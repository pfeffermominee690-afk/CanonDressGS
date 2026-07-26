#!/usr/bin/env python3
"""Build the 11-page camera-safe7 upload PDF from sealed PNG review pages."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from PIL import Image
from pypdf import PdfReader
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


PAGE_FILENAMES = (
    "subject00_O03_camsafe7_teacher_master_overview.png",
    "subject00_O03_slot00_camsafe7_review.png",
    "subject00_O03_slot01_camsafe7_review.png",
    "subject00_O03_slot02_camsafe7_review.png",
    "subject00_O03_slot03_camsafe7_review.png",
    "subject00_O03_slot05_camsafe7_review.png",
    "subject00_O03_slot06_camsafe7_review.png",
    "subject00_O03_slot07_camsafe7_review.png",
    "subject00_O03_camsafe7_region_risk_summary.png",
    "subject00_O03_clean7_vs_contaminated8_comparison.png",
    "subject00_O03_camsafe7_animation_and_slot04_quarantine.png",
)


def resolve_pages(root: Path) -> list[Path]:
    matches = {path.name: path for path in root.rglob("*.png")}
    missing = [name for name in PAGE_FILENAMES if name not in matches]
    if missing:
        raise RuntimeError(f"review pages are missing: {missing}")
    pages = [matches[name] for name in PAGE_FILENAMES]
    if len(set(pages)) != 11:
        raise RuntimeError("review page order is not unique")
    return pages


def build_pdf(pages: list[Path], output: Path) -> None:
    if output.exists():
        raise RuntimeError(f"PDF overwrite is forbidden: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    document = canvas.Canvas(str(temporary), pageCompression=1)
    for page in pages:
        with Image.open(page) as image:
            image.verify()
        with Image.open(page) as image:
            width_px, height_px = image.size
        width_pt = width_px * 72.0 / 150.0
        height_pt = height_px * 72.0 / 150.0
        document.setPageSize((width_pt, height_pt))
        document.drawImage(
            ImageReader(str(page)),
            0,
            0,
            width=width_pt,
            height=height_pt,
            preserveAspectRatio=True,
            anchor="c",
            mask="auto",
        )
        document.showPage()
    document.save()
    reader = PdfReader(str(temporary))
    if len(reader.pages) != 11:
        raise RuntimeError(f"PDF page count differs: {len(reader.pages)}")
    for index, (pdf_page, png_path) in enumerate(
        zip(reader.pages, pages, strict=True), start=1
    ):
        with Image.open(png_path) as image:
            expected_ratio = image.width / image.height
        box = pdf_page.mediabox
        actual_ratio = float(box.width) / float(box.height)
        if abs(expected_ratio - actual_ratio) > 1e-6:
            raise RuntimeError(f"PDF page aspect changed at page {index}")
    os.replace(temporary, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ordered = resolve_pages(args.pages_root)
    build_pdf(ordered, args.output)
    print(
        json.dumps(
            {
                "status": "PASS",
                "page_count": len(ordered),
                "output": str(args.output),
                "page_order": [path.name for path in ordered],
            },
            sort_keys=True,
        )
    )
