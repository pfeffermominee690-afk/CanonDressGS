from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FIXED_GAIN = 32.0

MODULE_FILES = {
    "module1": [
        "base_rgb.png", "zero_override_rgb.png", "post_restore_rgb.png",
        "xyz_rgb.png", "scaling_rgb.png", "rotation_rgb.png", "opacity_rgb.png",
        "sh0_rgb.png", "shN_degree1_zero_control_rgb.png", "shN_rgb.png",
        "attribute_comparison.png",
    ],
    "module2": [
        "teacher_reference_contact_sheet.png", "teacher_target_rgb.png", "base_rgb.png",
        "predicted_rgb.png", "predicted_alpha.png", "xyz_only.png", "scaling_only.png",
        "rotation_only.png", "opacity_only.png", "sh0_only.png",
        "shN_degree1_zero_control.png", "shN_only.png", "all_channels.png",
        "channel_comparison.png",
    ],
    "module3": [
        "observed_probability_S1.png", "observed_probability_S2.png", "observed_probability_S12.png",
        "geometry_gate_S1.png", "geometry_gate_S2.png", "geometry_gate_S12.png",
        "appearance_gate_S1.png", "appearance_gate_S2.png", "appearance_gate_S12.png",
        "teacher_gate.png", "observed_only_gate.png", "diffusion_gate.png", "learned_completed_gate.png",
        "teacher_gate_render.png", "observed_only_render.png", "diffusion_render.png", "learned_gate_render.png",
        "render_comparison.png", "feature_holdout_comparison.png", "loss_curve.png",
    ],
}

DIFF_CONTROLS = {
    "module1": {
        "zero_override_rgb.png": "base_rgb.png", "post_restore_rgb.png": "base_rgb.png",
        "xyz_rgb.png": "base_rgb.png", "scaling_rgb.png": "base_rgb.png",
        "rotation_rgb.png": "base_rgb.png", "opacity_rgb.png": "base_rgb.png",
        "sh0_rgb.png": "base_rgb.png", "shN_rgb.png": "shN_degree1_zero_control_rgb.png",
    },
    "module2": {
        "predicted_rgb.png": "base_rgb.png", "xyz_only.png": "base_rgb.png",
        "scaling_only.png": "base_rgb.png", "rotation_only.png": "base_rgb.png",
        "opacity_only.png": "base_rgb.png", "sh0_only.png": "base_rgb.png",
        "shN_only.png": "shN_degree1_zero_control.png", "all_channels.png": "base_rgb.png",
    },
    "module3": {
        "observed_only_gate.png": "teacher_gate.png", "diffusion_gate.png": "teacher_gate.png",
        "learned_completed_gate.png": "teacher_gate.png",
        "observed_only_render.png": "teacher_gate_render.png", "diffusion_render.png": "teacher_gate_render.png",
        "learned_gate_render.png": "teacher_gate_render.png",
    },
}


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def _font(size: int) -> ImageFont.ImageFont:
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _tile(image: Image.Image, label: str, width: int = 320, height: int = 240) -> Image.Image:
    canvas = Image.new("RGB", (width, height + 36), "white")
    thumb = image.copy(); thumb.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas.paste(thumb, ((width - thumb.width) // 2, (height - thumb.height) // 2))
    ImageDraw.Draw(canvas).text((6, height + 8), label, fill="black", font=_font(16))
    return canvas


def _difference(control: Image.Image, candidate: Image.Image) -> tuple[Image.Image, Image.Image, Image.Image, float]:
    if control.size != candidate.size:
        raise ValueError(f"comparison size mismatch: {control.size} != {candidate.size}")
    left = np.asarray(control, dtype=np.float32) / 255.0
    right = np.asarray(candidate, dtype=np.float32) / 255.0
    delta = np.abs(right - left)
    scalar = delta.max(axis=2)
    raw = Image.fromarray(np.round(np.clip(delta, 0, 1) * 255).astype(np.uint8), "RGB")
    gain = Image.fromarray(np.round(np.clip(delta * FIXED_GAIN, 0, 1) * 255).astype(np.uint8), "RGB")
    heat = np.stack((np.clip(scalar * FIXED_GAIN, 0, 1), np.clip(scalar * FIXED_GAIN / 2, 0, 1), np.zeros_like(scalar)), axis=2)
    overlay = np.clip(0.55 * left + 0.45 * heat, 0, 1)
    return raw, gain, Image.fromarray(np.round(overlay * 255).astype(np.uint8), "RGB"), float(delta.max())


def build(module: str, input_dir: Path, output_dir: Path, inspection: dict[str, Any] | None) -> None:
    required = MODULE_FILES[module]
    missing = [name for name in required if not (input_dir / name).is_file()]
    if missing: raise FileNotFoundError(f"missing required visual inputs: {missing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    images = {name: load_rgb(input_dir / name) for name in required}
    tiles: list[Image.Image] = []
    diff_max: dict[str, float] = {}
    for name in required:
        tiles.append(_tile(images[name], name))
        control_name = DIFF_CONTROLS[module].get(name)
        if control_name:
            raw, gain, overlay, maximum = _difference(images[control_name], images[name])
            diff_max[name] = maximum
            tiles.extend([
                _tile(raw, f"{name} | absolute diff | max={maximum:.6f}"),
                _tile(gain, f"{name} | absolute diff x{int(FIXED_GAIN)} fixed gain"),
                _tile(overlay, f"{name} | fixed-gain locality overlay"),
            ])
    columns = 4; tile_w, tile_h = tiles[0].size
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_w, rows * tile_h), (230, 230, 230))
    for index, tile in enumerate(tiles): sheet.paste(tile, ((index % columns) * tile_w, (index // columns) * tile_h))
    contact_name = f"{module}_visual_acceptance_contact_sheet.png"
    sheet.save(output_dir / contact_name)
    records = {} if inspection is None else inspection.get("records", {})
    status = "PARTIAL" if inspection is None else inspection["visual_acceptance_status"]
    payload = {
        "module": module, "inspection_method": "pixel-rendered contact sheet plus individually opened source PNGs",
        "images_actually_opened": False if inspection is None else bool(inspection.get("images_actually_opened")),
        "inspected_files": sorted(records), "checks": {} if inspection is None else inspection.get("checks", {}),
        "diff_max_from_png": diff_max, "warnings": [] if inspection is None else inspection.get("warnings", []),
        "failures": [] if inspection is None else inspection.get("failures", []), "visual_acceptance_status": status,
    }
    (output_dir / "visual_acceptance.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [f"# {module.title()} Visual Acceptance", "", f"Status: **{status}**", ""]
    for name, record in records.items():
        lines.extend([
            f"## {name}", "", f"- actual_open_status: {record['actual_open_status']}",
            f"- visual_status: {record['visual_status']}", f"- expected_semantics: {record['expected_semantics']}",
            f"- observed_behavior: {record['observed_behavior']}", f"- changed_region: {record['changed_region']}",
            f"- abnormal_region: {record['abnormal_region']}", f"- severity: {record['severity']}",
            f"- conclusion: {record['conclusion']}", "",
        ])
    (output_dir / "VISUAL_ACCEPTANCE.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", choices=sorted(MODULE_FILES), required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--inspection-record", type=Path)
    args = parser.parse_args()
    inspection = None if args.inspection_record is None else json.loads(args.inspection_record.read_text(encoding="utf-8"))
    build(args.module, args.input_dir, args.output_dir or args.input_dir, inspection)


if __name__ == "__main__": main()
