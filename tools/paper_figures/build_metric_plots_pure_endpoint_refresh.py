#!/usr/bin/env python3
"""Render Pure Endpoint refresh plots with visible N/A and denominators."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "svg.hashsalt": "canondressgs-pure-endpoint-refresh-v1",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
import matplotlib.pyplot as plt  # noqa: E402


COLORS = ["#1f5a7a", "#b24b3f", "#2f7d50", "#77538c", "#8a6a19", "#555555"]
MARKERS = ["o", "s", "^", "D", "v", "P"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def denominator_note(plot: Dict[str, Any]) -> str:
    if "denominator" in plot:
        return f"denominator: n={plot['denominator']}"
    if "denominator_per_cell" in plot:
        return f"denominator: n={plot['denominator_per_cell']} per cell"
    if "denominators" in plot:
        values = ", ".join(f"{key}={value}" for key, value in sorted(plot["denominators"].items()))
        return f"denominators: {values}"
    if "counts" in plot:
        values = ", ".join(f"{key}={value}" for key, value in sorted(plot["counts"].items()))
        return f"registered counts: {values}"
    raise ValueError(f"plot has no denominator metadata: {plot['plot_id']}")


def render_plot(plot: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    figure, axis = plt.subplots(figsize=(7.2, 4.6), dpi=200)
    figure.subplots_adjust(left=0.12, right=0.98, top=0.80, bottom=0.30)
    kind = plot.get("kind", "line")
    for index, series in enumerate(plot["series"]):
        values = series["values"]
        xs = [item["x"] for item in values]
        ys = [item["y"] for item in values]
        color = COLORS[index % len(COLORS)]
        if kind == "bar":
            width = 0.8 / max(1, len(plot["series"]))
            offsets = [float(x) + (index - (len(plot["series"]) - 1) / 2) * width for x in xs]
            errors = [item.get("yerr", 0.0) for item in values]
            yerr = errors if any(error != 0.0 for error in errors) else None
            axis.bar(
                offsets, ys, width=width, label=series["label"], color=color,
                yerr=yerr, capsize=2 if yerr is not None else 0,
            )
        else:
            errors = [item.get("yerr") for item in values]
            if any(value is not None for value in errors):
                axis.errorbar(
                    xs, ys, yerr=[value or 0.0 for value in errors], label=series["label"],
                    color=color, marker=MARKERS[index % len(MARKERS)], linewidth=1.5, capsize=2,
                )
            else:
                axis.plot(
                    xs, ys, label=series["label"], color=color,
                    marker=MARKERS[index % len(MARKERS)], linewidth=1.5,
                )
    figure.suptitle(plot["title"], y=0.965, fontsize=12)
    axis.set_xlabel(plot["x_label"])
    axis.set_ylabel(plot["y_label"])
    axis.grid(axis="y", color="#dddddd", linewidth=0.6)
    tick_positions: Dict[str, float] = {}
    if plot.get("x_ticks"):
        positions = [item["value"] for item in plot["x_ticks"]]
        labels = [item["label"] for item in plot["x_ticks"]]
        tick_positions = dict(zip(labels, positions))
        rotate_labels = len(labels) > 6 or sum(len(label) for label in labels) > 48
        if rotate_labels:
            axis.set_xticks(positions, labels, rotation=25, ha="right", fontsize=8)
        else:
            axis.set_xticks(positions, labels)
    if plot.get("y_limits"):
        axis.set_ylim(*plot["y_limits"])
    if len(plot["series"]) > 1:
        axis.legend(frameon=False, fontsize=8)
    y_min, y_max = axis.get_ylim()
    for label in plot.get("not_applicable", {}):
        if label in tick_positions:
            axis.text(
                tick_positions[label], y_min + 0.03 * (y_max - y_min), "N/A",
                ha="center", va="bottom", fontsize=8, color="#555555", fontweight="bold",
            )
    figure.text(
        0.98, 0.90, textwrap.fill(denominator_note(plot), width=95),
        ha="right", va="top", fontsize=7,
    )
    figure.text(
        0.12, 0.025, plot["claim_status"], ha="left", va="bottom",
        fontsize=7, color="#7a1f1f",
    )
    outputs: Dict[str, str] = {}
    for suffix in ("png", "svg", "pdf"):
        path = output_dir / f"{plot['plot_id']}.{suffix}"
        metadata = {"Creator": "CanonDressGS Pure Endpoint refresh plot tool", "Title": plot["title"]}
        if suffix == "pdf":
            metadata.update({"CreationDate": None, "ModDate": None})
        figure.savefig(path, format=suffix, dpi=300 if suffix == "png" else None, metadata=metadata)
        if suffix == "svg":
            normalized = "\n".join(
                line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()
            ) + "\n"
            path.write_text(normalized, encoding="utf-8", newline="\n")
        outputs[suffix] = sha256_file(path)
    plt.close(figure)
    source_path = output_dir / f"{plot['plot_id']}.source.json"
    write_json(source_path, plot)
    outputs["source_json"] = sha256_file(source_path)
    return {
        "plot_id": plot["plot_id"],
        "outputs": outputs,
        "source_files": plot["source_files"],
        "claim_status": plot["claim_status"],
        "data_origin": "STRUCTURED_JSON_OR_REGISTRY_NOT_SCREENSHOT_TRANSCRIPTION",
        "denominator_note": denominator_note(plot),
        "not_applicable": plot.get("not_applicable", {}),
    }


def main() -> int:
    args = parse_args()
    source = load_json(args.source_data)
    plots: List[Dict[str, Any]] = sorted(source["plots"], key=lambda item: item["plot_id"])
    plots = [plot for plot in plots if not args.include or plot["plot_id"] in args.include]
    plots = [plot for plot in plots if plot["plot_id"] not in args.exclude]
    if args.dry_run:
        print("\n".join(plot["plot_id"] for plot in plots))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "paper_metric_plot_pure_endpoint_refresh_manifest.v1",
        "plots": [render_plot(plot, args.output_dir) for plot in plots],
    }
    write_json(args.output_dir / "metric_plot_manifest.json", manifest)
    print(json.dumps({"metric_plots": len(plots), "formats_each": ["PDF", "PNG", "SVG"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
