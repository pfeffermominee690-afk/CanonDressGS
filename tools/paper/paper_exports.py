from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageOps

from .evaluate_seen_outfit import FIXED_VIEWS, SEEN_OUTFITS


SMOKE_MARKER = "SMOKE ONLY — NOT PAPER RESULTS"


METHOD_ROWS = ("B0", "B1", "B2", "B3", "B4", "B5", "Ours")
METHOD_LABELS = {
    "B0": "Base Avatar", "B1": "B1 — Optimization Upper Bound",
    "B2": "B2 — Seen-only Lookup", "B3": "Global Reference Feature",
    "B4": "Clothing Mean Only", "B5": "Legacy Complex Fusion", "Ours": "Ours",
}
TABLE_COLUMNS = {
    "table_1": (
        "garment_rgb_mae", "edit_reduction", "target_closer_fraction",
        "protected_rgb_mae", "background_rgb_mae", "nearest_teacher_accuracy",
        "correct_vs_swapped_wins", "trainable_parameter_count",
    ),
    "table_2": (
        "standardized_coefficient_rmse", "nearest_teacher_accuracy",
        "correct_vs_swapped_wins", "garment_rgb_mae", "protected_rgb_mae",
    ),
    "table_3": (
        "trainable_parameter_count", "basis_storage_bytes", "peak_vram_bytes",
        "training_time_seconds", "inference_time_seconds",
    ),
}
UNITS = {
    "peak_vram_bytes": "bytes", "basis_storage_bytes": "bytes",
    "training_time_seconds": "s", "inference_time_seconds": "s",
    "render_time_seconds": "s", "trainable_parameter_count": "parameters",
}


def _format(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, Mapping) and "mean" in value:
        if value["mean"] is None:
            return "N/A"
        if value.get("std") is None:
            return f"{float(value['mean']):.6g}"
        return f"{float(value['mean']):.6g} ± {float(value.get('std', 0.0)):.3g}"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_table(
    name: str,
    title: str,
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    marker: str | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    csv_path, md_path, tex_path, json_path = (
        output_dir / f"{name}.csv", output_dir / f"{name}.md",
        output_dir / f"{name}.tex", output_dir / f"{name}.json",
    )
    formatted = [{key: _format(value) for key, value in row.items()} for row in rows]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(formatted)
    md_lines = [f"# {title}", ""]
    if marker:
        md_lines.extend([f"> {marker}", ""])
    md_lines.extend(["| " + " | ".join(fields) + " |", "|" + "|".join("---" for _ in fields) + "|"])
    md_lines.extend("| " + " | ".join(row[field] for field in fields) + " |" for row in formatted)
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    tex_lines = ["\\begin{tabular}{" + "l" * len(fields) + "}", " & ".join(fields) + " \\\\", "\\hline"]
    tex_lines.extend(" & ".join(row[field].replace("±", "$\\pm$") for field in fields) + " \\\\" for row in formatted)
    tex_lines.append("\\end{tabular}")
    if marker:
        tex_lines.insert(0, f"% {marker}")
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps({"title": title, "marker": marker, "units": UNITS, "rows": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return [csv_path, md_path, tex_path, json_path]


def export_tables(source: Mapping[str, Any], output_dir: Path) -> dict[str, list[str]]:
    source_marker = source.get("marker")
    if source_marker in {"SYNTHETIC_TEST_ONLY", "SMOKE_ONLY"} and "paper_ready" in output_dir.parts:
        raise ValueError("non-formal source cannot enter formal paper table directory")
    marker = SMOKE_MARKER if source_marker == "SMOKE_ONLY" else None
    outputs: dict[str, list[str]] = {}
    table1 = []
    for method in METHOD_ROWS:
        metrics = source["methods"][method]
        table1.append({"method": METHOD_LABELS[method], **{key: metrics.get(key) for key in TABLE_COLUMNS["table_1"]}})
    outputs["table_1"] = [str(path) for path in _write_table("table_1", "Seen-outfit main comparison", table1, output_dir, marker=marker)]
    table2 = [{"ablation": name, **{key: values.get(key) for key in TABLE_COLUMNS["table_2"]}} for name, values in source["ablations"].items()]
    outputs["table_2"] = [str(path) for path in _write_table("table_2", "Method ablation", table2, output_dir, marker=marker)]
    table3 = [{"method": METHOD_LABELS[name], **{key: values.get(key) for key in TABLE_COLUMNS["table_3"]}} for name, values in source["methods"].items()]
    outputs["table_3"] = [str(path) for path in _write_table("table_3", "Efficiency", table3, output_dir, marker=marker)]
    table4 = [{"entry": name, "status": value.get("status", "N/A")} for name, value in source["held_out"].items()]
    outputs["table_4"] = [str(path) for path in _write_table("table_4", "Held-out Diagnostic — FAIL", table4, output_dir, marker=marker)]
    if marker:
        (output_dir / "SMOKE_ONLY.txt").write_text(marker + "\nNOT_FOR_PAPER_NUMBERS\n", encoding="utf-8")
    return outputs


FIGURE_LAYOUTS = {
    "figure_1": {"rows": ["method"], "columns": ["reference-only prediction", "target supervision/evaluation"]},
    "figure_2": {"rows": list(SEEN_OUTFITS), "views": list(FIXED_VIEWS), "columns": ["base", "target", "teacher", "ours", "B5"]},
    "figure_3": {"rows": ["fixed target"], "columns": list(SEEN_OUTFITS)},
    "figure_4": {"rows": ["basis rank"], "columns": ["K=1", "K=2", "K=3", "K=4"]},
    "figure_5": {"rows": ["supervision"], "columns": ["old collapse", "old gradient", "signed raw-logit", "CS-PASS"]},
    "figure_6": {"rows": ["O07"], "columns": ["teacher", "projection", "prediction", "O03 endpoint"]},
}


def validate_figure_layouts(layouts: Mapping[str, Any]) -> None:
    if list(layouts) != [f"figure_{index}" for index in range(1, 7)]:
        raise ValueError("figure set or order differs from frozen protocol")
    figure2 = layouts["figure_2"]
    if figure2["rows"] != list(SEEN_OUTFITS) or figure2["views"] != list(FIXED_VIEWS):
        raise ValueError("Figure 2 outfit/view order mismatch")
    if figure2["columns"] != ["base", "target", "teacher", "ours", "B5"]:
        raise ValueError("Figure 2 columns must use preregistered B5 without cherry-picking")


def export_figure_layouts(
    output_dir: Path,
    *,
    synthetic: bool,
    source_manifest: Mapping[str, Any] | None = None,
    smoke_only: bool = False,
) -> dict[str, Any]:
    validate_figure_layouts(FIGURE_LAYOUTS)
    if synthetic and "paper_ready" in output_dir.parts:
        raise ValueError("synthetic figure cannot enter formal paper figure directory")
    if smoke_only and "paper_ready" in output_dir.parts:
        raise ValueError("smoke figure cannot enter formal paper figure directory")
    if not synthetic and source_manifest is None:
        raise ValueError("formal figure export requires an explicit source path manifest")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "marker": "SYNTHETIC_TEST_ONLY" if synthetic else "SMOKE_ONLY" if smoke_only else "FORMAL_SOURCE_PATHS",
        "paper_numbers_status": "NOT_FOR_PAPER_NUMBERS" if smoke_only else None,
        "fixed_outfit_order": list(SEEN_OUTFITS), "fixed_view_order": list(FIXED_VIEWS),
        "strongest_baseline": "B5", "selection_rule": "fixed_not_metric_selected",
        "allowed_operations": ["concatenate", "aspect_preserving_crop", "label", "uniform_crop"],
        "forbidden_operations": ["retouch", "artifact_removal", "color_change", "successful_region_selection"],
        "crop_definitions": {view: "uniform_full_frame" for view in FIXED_VIEWS},
        "figures": {},
    }
    colors = ((52, 82, 120), (110, 68, 104), (74, 116, 84), (130, 96, 58), (85, 85, 85), (120, 52, 52))
    for index, (name, layout) in enumerate(FIGURE_LAYOUTS.items()):
        columns = len(layout["columns"])
        rows = len(layout.get("rows", [])) * len(layout.get("views", [None]))
        canvas = Image.new("RGB", (columns * 96, max(1, rows) * 72), "white")
        draw = ImageDraw.Draw(canvas)
        expected_count = max(1, rows) * columns
        formal_sources = list(source_manifest["figures"][name]["source_paths"]) if source_manifest is not None else []
        if not synthetic and len(formal_sources) != expected_count:
            raise ValueError(f"{name} source count differs from frozen layout")
        for row in range(max(1, rows)):
            for column in range(columns):
                x0, y0 = column * 96, row * 72
                if synthetic:
                    color = colors[(index + row + column) % len(colors)]
                    draw.rectangle((x0 + 2, y0 + 2, x0 + 93, y0 + 69), fill=color)
                    draw.text((x0 + 5, y0 + 5), f"{name}\nr{row} c{column}", fill="white")
                else:
                    source = Path(formal_sources[row * columns + column])
                    if not source.is_file():
                        raise FileNotFoundError(source)
                    with Image.open(source) as opened:
                        panel = ImageOps.contain(opened.convert("RGB"), (92, 68))
                    canvas.paste(panel, (x0 + (96 - panel.width) // 2, y0 + (72 - panel.height) // 2))
        path = output_dir / f"{name}.png"
        if smoke_only:
            banner_height = 20
            marked = Image.new("RGB", (canvas.width, canvas.height + banner_height), "white")
            marked.paste(canvas, (0, banner_height))
            banner = ImageDraw.Draw(marked)
            banner.rectangle((0, 0, marked.width, banner_height), fill=(125, 0, 0))
            banner.text((5, 4), "SMOKE ONLY - NOT PAPER RESULTS", fill="white")
            canvas = marked
        canvas.save(path)
        source_entries = [f"SYNTHETIC_TEST_ONLY/{name}/row_{row}/column_{column}" for row in range(max(1, rows)) for column in range(columns)] if synthetic else formal_sources
        manifest["figures"][name] = {"path": str(path), "layout": layout, "source_paths": source_entries}
    manifest_path = output_dir / "figure_source_path_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest
