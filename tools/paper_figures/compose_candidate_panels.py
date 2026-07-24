#!/usr/bin/env python3
"""Compose deterministic, no-crop candidate panels from registered assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--panel-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contain_image(path: Path, size: Tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        source.load()
        source = source.convert("RGB")
        source.thumbnail(size, Image.Resampling.LANCZOS)
        target = Image.new("RGB", size, "white")
        target.paste(source, ((size[0] - source.width) // 2, (size[1] - source.height) // 2))
        return target


def compose(panel: Dict[str, Any], by_id: Dict[str, Dict[str, Any]], output_dir: Path) -> Dict[str, Any]:
    tile_size = tuple(panel.get("tile_size", [256, 256]))
    columns = int(panel.get("columns", 4))
    assets = [by_id[item["asset_id"]] for item in panel["items"]]
    rows = max(1, math.ceil(len(assets) / columns))
    label_height = 42
    title_height = 52
    canvas = Image.new("RGB", (columns * tile_size[0], title_height + rows * (tile_size[1] + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 8), panel["title"], fill="black", font=font)
    draw.text((8, 27), panel["claim_status"], fill=(130, 20, 20), font=font)
    for index, (item, asset) in enumerate(zip(panel["items"], assets)):
        row, column = divmod(index, columns)
        x = column * tile_size[0]
        y = title_height + row * (tile_size[1] + label_height)
        image_path = Path(asset.get("thumbnail_absolute_path") or asset["original_absolute_path"])
        canvas.paste(contain_image(image_path, tile_size), (x, y))
        draw.rectangle((x, y, x + tile_size[0] - 1, y + tile_size[1] - 1), outline=(80, 80, 80))
        draw.text((x + 5, y + tile_size[1] + 4), item["label"], fill="black", font=font)
        draw.text((x + 5, y + tile_size[1] + 20), asset["asset_id"], fill=(80, 80, 80), font=font)
    output = output_dir / f"{panel['panel_id']}.png"
    canvas.save(output, format="PNG", compress_level=9, optimize=False)
    return {
        "panel_id": panel["panel_id"],
        "asset_ids": [asset["asset_id"] for asset in assets],
        "output_path": str(output),
        "output_sha256": sha(output),
        "crop": None,
        "resize": {"mode": "CONTAIN", "filter": "LANCZOS", "tile_size": list(tile_size)},
        "method_specific_enhancement": False,
        "selection_rule": panel["selection_rule"],
        "claim_status": panel["claim_status"],
    }


def main() -> int:
    args = parse_args()
    registry = load(args.registry)
    specs = load(args.panel_spec)
    by_id = {asset["asset_id"]: asset for asset in registry["assets"]}
    missing = sorted({item["asset_id"] for panel in specs["panels"] for item in panel["items"]} - set(by_id))
    if missing:
        raise SystemExit(f"unregistered panel assets: {missing}")
    if args.dry_run:
        print(json.dumps({"panels": len(specs["panels"]), "missing": missing}, sort_keys=True))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "paper_candidate_panel_manifest.v1",
        "panels": [compose(panel, by_id, args.output_dir) for panel in sorted(specs["panels"], key=lambda x: x["panel_id"])],
    }
    write(args.output_dir / "candidate_panel_manifest.json", manifest)
    print(json.dumps({"candidate_panels": len(manifest["panels"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
