#!/usr/bin/env python3
"""Render deterministic PNG, SVG, and PDF plots from normalized JSON data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "svg.hashsalt": "canondressgs-paper-figure-evidence-bank-v1",
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


def render_plot(plot: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    figure, axis = plt.subplots(figsize=(6.4, 3.8), dpi=200, constrained_layout=True)
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
            axis.bar(offsets, ys, width=width, label=series["label"], color=color, yerr=errors, capsize=2)
        else:
            errors = [item.get("yerr") for item in values]
            if any(value is not None for value in errors):
                axis.errorbar(xs, ys, yerr=[value or 0.0 for value in errors], label=series["label"], color=color,
                              marker=MARKERS[index % len(MARKERS)], linewidth=1.5, capsize=2)
            else:
                axis.plot(xs, ys, label=series["label"], color=color,
                          marker=MARKERS[index % len(MARKERS)], linewidth=1.5)
    axis.set_title(plot["title"])
    axis.set_xlabel(plot["x_label"])
    axis.set_ylabel(plot["y_label"])
    axis.grid(axis="y", color="#dddddd", linewidth=0.6)
    if plot.get("x_ticks"):
        positions = [item["value"] for item in plot["x_ticks"]]
        labels = [item["label"] for item in plot["x_ticks"]]
        if len(labels) > 6:
            axis.set_xticks(positions, labels, rotation=30, ha="right")
        else:
            axis.set_xticks(positions, labels)
    if plot.get("y_limits"):
        axis.set_ylim(*plot["y_limits"])
    if len(plot["series"]) > 1:
        axis.legend(frameon=False, fontsize=8)
    axis.text(
        0.0,
        -0.25,
        plot["claim_status"],
        transform=axis.transAxes,
        fontsize=7,
        color="#7a1f1f",
    )
    outputs: Dict[str, str] = {}
    for suffix in ("png", "svg", "pdf"):
        path = output_dir / f"{plot['plot_id']}.{suffix}"
        metadata = {
            "Creator": "CanonDressGS deterministic metric plot tool",
            "Title": plot["title"],
        }
        if suffix == "pdf":
            metadata.update({"CreationDate": None, "ModDate": None})
        figure.savefig(path, format=suffix, dpi=300 if suffix == "png" else None, metadata=metadata)
        if suffix == "svg":
            normalized = "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n"
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
        "schema_version": "paper_metric_plot_manifest.v1",
        "plots": [render_plot(plot, args.output_dir) for plot in plots],
    }
    write_json(args.output_dir / "metric_plot_manifest.json", manifest)
    print(json.dumps({"metric_plots": len(plots), "formats_each": ["PDF", "PNG", "SVG"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
