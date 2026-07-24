#!/usr/bin/env python3
"""Build deterministic review-only contact sheets from an asset registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import textwrap
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


SHEET_DEFINITIONS: Sequence[Tuple[str, Callable[[Dict[str, Any]], bool]]] = (
    (
        "old_main_methods",
        lambda a: a["source_category"] == "HISTORICAL_51_RUN"
        and any(token in a["experiment_id"] for token in ("PAPER-OURS", "PAPER-B")),
    ),
    (
        "rank_ablation",
        lambda a: a["source_category"] == "HISTORICAL_51_RUN" and "PAPER-A1-" in a["experiment_id"],
    ),
    (
        "representation_ablations",
        lambda a: a["source_category"] == "HISTORICAL_51_RUN"
        and any(f"PAPER-A{index}-" in a["experiment_id"] for index in range(2, 7)),
    ),
    (
        "reference_count",
        lambda a: a["source_category"] == "HISTORICAL_51_RUN" and "PAPER-A7-" in a["experiment_id"],
    ),
    (
        "o07",
        lambda a: "O07" in a["original_relative_path"].upper(),
    ),
    (
        "early_failures",
        lambda a: a["source_category"] == "METHOD_DEVELOPMENT_FAILURE_TRIAGE",
    ),
    (
        "dual_support",
        lambda a: a["source_category"] == "DUAL_SUPPORT_ALL_PAIR",
    ),
    (
        "controller_visuals",
        lambda a: a["source_category"] == "CONTROLLER_DIAGNOSTIC",
    ),
    (
        "subject00",
        lambda a: a["source_category"] == "SUBJECT00_BASE_AVATAR",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--transform-registry", type=Path)
    parser.add_argument("--max-assets", type=int, default=24)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_assets(
    assets: Sequence[Dict[str, Any]],
    predicate: Callable[[Dict[str, Any]], bool],
    limit: int,
) -> List[Dict[str, Any]]:
    selected = [asset for asset in assets if predicate(asset) and asset.get("thumbnail_absolute_path")]
    selected.sort(key=lambda item: item["asset_id"])
    return selected[:limit]


def open_thumbnail(asset: Dict[str, Any], size: Tuple[int, int]) -> Image.Image:
    path = Path(asset["thumbnail_absolute_path"])
    with Image.open(path) as image:
        image.load()
        image = image.convert("RGB")
        image.thumbnail(size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", size, "white")
        x = (size[0] - image.width) // 2
        y = (size[1] - image.height) // 2
        canvas.paste(image, (x, y))
        return canvas


def build_sheet(name: str, assets: Sequence[Dict[str, Any]], columns: int) -> Tuple[Image.Image, Dict[str, Any]]:
    font = ImageFont.load_default()
    title_height = 48
    image_size = (240, 210)
    label_height = 94
    cell_size = (image_size[0] + 16, image_size[1] + label_height + 16)
    rows = max(1, math.ceil(max(1, len(assets)) / columns))
    sheet = Image.new("RGB", (columns * cell_size[0] + 16, rows * cell_size[1] + title_height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), f"{name} | REVIEW CONTACT SHEET | NOT A PAPER FIGURE", fill="black", font=font)
    draw.line((12, 31, sheet.width - 12, 31), fill=(100, 100, 100), width=1)
    if not assets:
        draw.text((12, title_height + 18), "NO SEALED VISUAL ASSETS DISCOVERED", fill=(150, 0, 0), font=font)
    for index, asset in enumerate(assets):
        row, column = divmod(index, columns)
        x = 8 + column * cell_size[0]
        y = title_height + row * cell_size[1]
        tile = open_thumbnail(asset, image_size)
        sheet.paste(tile, (x + 8, y + 8))
        draw.rectangle((x + 8, y + 8, x + 8 + image_size[0], y + 8 + image_size[1]), outline=(80, 80, 80))
        garment = asset.get("garment_pair")
        if not garment or garment == "NOT_RECORDED":
            garment = asset.get("garment", "NOT_RECORDED")
        label_lines = [
            asset["asset_id"],
            str(asset.get("experiment_id", "NOT_RECORDED")),
            f"garment={garment} view={asset.get('camera')} seed={asset.get('seed')}",
            f"eligibility={asset.get('paper_eligibility')}",
            f"warning={asset.get('claim_status')}",
        ]
        label = "\n".join(
            wrapped
            for line in label_lines
            for wrapped in textwrap.wrap(line, width=39, break_long_words=True) or [""]
        )
        draw.multiline_text((x + 9, y + image_size[1] + 14), label, fill="black", font=font, spacing=1)
    return sheet, {
        "name": name,
        "asset_ids": [asset["asset_id"] for asset in assets],
        "columns": columns,
        "cell_size": list(cell_size),
        "thumbnail_box": list(image_size),
        "selection": "LEXICOGRAPHIC_ASSET_ID_PREFIX_NOT_VISUAL_QUALITY",
        "claim_boundary": "REVIEW_CONTACT_SHEET_NOT_PAPER_FIGURE",
    }


def main() -> int:
    args = parse_args()
    registry = load_json(args.registry)
    assets = registry["assets"]
    definitions = [item for item in SHEET_DEFINITIONS if not args.include or item[0] in args.include]
    definitions = [item for item in definitions if item[0] not in args.exclude]
    if args.dry_run:
        for name, predicate in definitions:
            print(f"{name}: {len(select_assets(assets, predicate, args.max_assets))}")
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    transforms: List[Dict[str, Any]] = []
    for name, predicate in definitions:
        selected = select_assets(assets, predicate, args.max_assets)
        image, transform = build_sheet(name, selected, args.columns)
        output = args.output_dir / f"{name}.png"
        image.save(output, format="PNG", compress_level=9, optimize=False)
        transform.update({
            "transform_id": f"CONTACT-SHEET-{name.upper().replace('_', '-')}",
            "output_path": str(output),
            "output_sha256": sha256_file(output),
            "operation": "DETERMINISTIC_LOSSLESS_PNG_COMPOSITION_WITH_LANCZOS_THUMBNAILS",
            "crop": None,
            "method_specific_enhancement": False,
            "ai_generated": False,
        })
        transforms.append(transform)
    if args.transform_registry:
        write_json(args.transform_registry, {
            "schema_version": "paper_figure_transform_registry.v1",
            "transforms": transforms,
        })
    print(json.dumps({"contact_sheets": len(transforms)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
