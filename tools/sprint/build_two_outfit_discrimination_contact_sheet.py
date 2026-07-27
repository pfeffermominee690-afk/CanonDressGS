from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageDraw


LAYOUT_SCHEMA_VERSION = "canondressgs.two_outfit_contact_sheet_layout.v1"
PANEL_KEYS = (
    "reference_first",
    "reference_second",
    "first_correct_prediction",
    "first_swapped_prediction",
    "second_correct_prediction",
    "second_swapped_prediction",
    "first_target",
    "second_target",
    "rgb_error",
    "alpha_support_error",
)
REQUIRED_CROPS = ("torso_garment", "sleeves_arms", "shoes_protected", "garment_boundary")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def validate_contact_sheet_layout(layout: Mapping[str, Any], root: str | Path = ".") -> dict[str, Any]:
    if layout.get("schema_version") != LAYOUT_SCHEMA_VERSION:
        raise ValueError("unexpected contact-sheet layout schema")
    outfits = list(layout["outfits"])
    if len(outfits) != 2 or len(set(outfits)) != 2:
        raise ValueError("discrimination contact sheet requires exactly two selected outfits")
    rows = list(layout["rows"])
    if not rows:
        raise ValueError("contact sheet must contain at least one target pose row")
    base = Path(root)
    expected_size: tuple[int, int] | None = None
    for row in rows:
        panels = row["panels"]
        if tuple(panels) != PANEL_KEYS:
            raise ValueError("panel keys/order differs from the frozen ten-column layout")
        fingerprints = row.get("panel_camera_fingerprints", {})
        if fingerprints and any(value != row["target_pose_camera_fingerprint"] for value in fingerprints.values()):
            raise ValueError("a prediction/target panel uses a different pose or camera")
        crops = row["crop_boxes_xyxy"]
        if set(crops) != set(REQUIRED_CROPS):
            raise ValueError("row does not contain the four required crop definitions")
        row_size: tuple[int, int] | None = None
        for key in PANEL_KEYS:
            path = _resolve(base, panels[key])
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as image:
                size = image.size
            if row_size is None:
                row_size = size
            elif size != row_size:
                raise ValueError("panels in a row must have identical resolution and scaling")
        if expected_size is None:
            expected_size = row_size
        elif row_size != expected_size:
            raise ValueError("all rows must use identical resolution and scaling")
        assert row_size is not None
        for name, box in crops.items():
            if len(box) != 4:
                raise ValueError(f"crop {name} must be xyxy")
            left, top, right, bottom = map(int, box)
            if not (0 <= left < right <= row_size[0] and 0 <= top < bottom <= row_size[1]):
                raise ValueError(f"crop {name} lies outside the shared panel bounds")
    return {
        "row_count": len(rows),
        "panel_count_per_row": len(PANEL_KEYS),
        "same_camera": True,
        "same_crop": True,
        "same_scaling": True,
        "manual_color_adjustment": False,
    }


def _labels(outfits: list[str]) -> list[str]:
    first, second = outfits
    return [
        f"Reference {first}",
        f"Reference {second}",
        f"{first} correct prediction",
        f"{first} swapped-reference prediction",
        f"{second} correct prediction",
        f"{second} swapped-reference prediction",
        f"{first} target",
        f"{second} target",
        "RGB error",
        "Alpha/support error",
    ]


def _compose_grid(rows: list[tuple[str, list[Image.Image]]], labels: list[str]) -> Image.Image:
    panel_width, panel_height = rows[0][1][0].size
    label_height, row_label_height = 36, 28
    canvas = Image.new(
        "RGB",
        (panel_width * len(labels), label_height + len(rows) * (row_label_height + panel_height)),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    for index, label in enumerate(labels):
        draw.text((index * panel_width + 4, 8), label, fill=(0, 0, 0))
    y = label_height
    for row_label, panels in rows:
        draw.text((4, y + 5), row_label, fill=(0, 0, 0))
        y += row_label_height
        for index, panel in enumerate(panels):
            canvas.paste(panel.convert("RGB"), (index * panel_width, y))
        y += panel_height
    return canvas


def build_contact_sheet(layout_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    path = Path(layout_path).resolve()
    layout = json.loads(path.read_text(encoding="utf-8"))
    validation = validate_contact_sheet_layout(layout, path.parent)
    output = Path(output_dir)
    source_data = output / "source_data"
    output.mkdir(parents=True, exist_ok=True)
    source_data.mkdir(parents=True, exist_ok=True)
    labels = _labels(list(layout["outfits"]))

    full_rows: list[tuple[str, list[Image.Image]]] = []
    loaded_rows: list[tuple[Mapping[str, Any], list[Image.Image]]] = []
    asset_hashes: dict[str, str] = {}
    for row in layout["rows"]:
        panels = []
        for key in PANEL_KEYS:
            panel_path = _resolve(path.parent, row["panels"][key])
            asset_hashes[str(panel_path)] = _sha256(panel_path)
            with Image.open(panel_path) as image:
                panels.append(image.convert("RGB").copy())
        row_label = f"{row['target_view']} | {row['target_condition_id']} | correct/swapped labeled above"
        full_rows.append((row_label, panels))
        loaded_rows.append((row, panels))
    sheet = _compose_grid(full_rows, labels)
    main_path = output / "two_outfit_discrimination_contact_sheet.png"
    sheet.save(main_path)

    crop_outputs: dict[str, str] = {}
    for crop_name in REQUIRED_CROPS:
        cropped_rows = []
        for row, panels in loaded_rows:
            box = tuple(map(int, row["crop_boxes_xyxy"][crop_name]))
            cropped_rows.append((f"{row['target_view']} | {crop_name}", [panel.crop(box) for panel in panels]))
        crop_sheet = _compose_grid(cropped_rows, labels)
        crop_path = output / f"two_outfit_discrimination__{crop_name}.png"
        crop_sheet.save(crop_path)
        crop_outputs[crop_name] = str(crop_path)

    source_record = {
        **layout,
        "validated": validation,
        "source_layout_sha256": _sha256(path),
        "source_asset_sha256": asset_hashes,
        "rendering_contract": {
            "same_pose_camera": True,
            "same_crop": True,
            "same_scaling": True,
            "manual_color_adjustment": False,
        },
    }
    source_path = source_data / "layout.json"
    source_path.write_text(json.dumps(source_record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        **validation,
        "main_png": str(main_path),
        "crop_pngs": crop_outputs,
        "source_data": str(source_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the fixed two-outfit discrimination contact sheet.")
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_contact_sheet(args.layout, args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
