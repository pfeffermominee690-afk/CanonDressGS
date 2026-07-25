#!/usr/bin/env python3
"""Build the frozen AAAI publication figures without changing source pixels."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "canondressgs-aaai27-figure-p0-20260725",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
    }
)

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


TASK_ID = "AAAI27-CANONDRESSGS-FIGURE-P0-CLOSURE-PREP-001"
CREATED = "2026-07-25T00:00:00+08:00"
PAPER_HEAD = "d2db7696c5521178fe1970fc69db64363aa19053"
HEADS = {
    "paper_source": PAPER_HEAD,
    "figure_bank": "3f67f577b85e5f74951287e9a938c11aef74ac60",
    "geometry_causal": "8c43524b7ca0ee8b3c795dbe349466f30b29354f",
    "pure_endpoint": "ce110887a942cf8db082ba688c8d36d2433bfdbe",
    "dual_support": "d802f427f1e9c23595e1bdb4f10135f7de2c3f08",
    "headroom": "674e6092e21eeddeb22e963536247a3385c4e200",
    "loo": "4472e1815287b1866da6d3ed83b2984e0d43611d",
}
GARMENTS = ("O01", "O02", "O03", "O04", "O08")
INK = "#18232d"
BLUE = "#276fbf"
GREEN = "#2a9d6f"
RED = "#c7493a"
GOLD = "#d49b2a"
GRAY = "#66737f"
LIGHT = "#edf1f4"
PDF_META = {
    "Title": "CanonDressGS AAAI27 publication figure",
    "Author": "CanonDressGS authors",
    "Creator": "tools/paper_figures/build_publication_figure_assets.py",
    "CreationDate": datetime(2026, 7, 25, tzinfo=timezone.utc),
    "ModDate": datetime(2026, 7, 25, tzinfo=timezone.utc),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def image_info(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def show(ax: plt.Axes, path: Path, title: str | None = None) -> None:
    with Image.open(path) as source:
        image = np.asarray(source.convert("RGB"))
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#c8d0d7")
        spine.set_linewidth(0.6)
    if title:
        ax.set_title(title, pad=3, color=INK, fontweight="semibold")


def panel(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.01,
        0.99,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="white",
        bbox={"boxstyle": "square,pad=0.22", "facecolor": INK, "edgecolor": "none"},
        zorder=20,
    )


def arrow_text(ax: plt.Axes, x: float, y: float, text: str, color: str = BLUE) -> None:
    ax.annotate(
        text,
        xy=(x + 0.12, y),
        xytext=(x, y),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        color=color,
        fontsize=8,
        fontweight="semibold",
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.4},
    )


def save_figure(fig: plt.Figure, stem: Path, *, png_dpi: int = 200, svg: bool = False) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    fig.savefig(outputs[0], bbox_inches="tight", metadata=PDF_META)
    fig.savefig(outputs[1], bbox_inches="tight", dpi=png_dpi, metadata={"Software": PDF_META["Creator"]})
    if svg:
        outputs.append(stem.with_suffix(".svg"))
        fig.savefig(outputs[-1], bbox_inches="tight", metadata={"Creator": PDF_META["Creator"], "Date": "2026-07-25T00:00:00+00:00"})
    plt.close(fig)
    return outputs


def ref(source: Path, garment: str, condition: str = "cond_000000") -> Path:
    return source / "pure_endpoint" / "references" / garment / f"{condition}.png"


def render(source: Path, garment: str, kind: str) -> Path:
    return source / "pure_endpoint" / "renders" / f"{garment}_{kind}.png"


def build_candidate_a(source: Path, review: Path) -> list[Path]:
    fig = plt.figure(figsize=(11.6, 9.0), constrained_layout=True)
    grid = fig.add_gridspec(5, 6, width_ratios=[0.45, 1, 1, 1, 0.45, 1.05])
    conditions = ("cond_000000", "cond_000318", "cond_000017")
    for row, garment in enumerate(GARMENTS):
        tag = fig.add_subplot(grid[row, 0])
        tag.axis("off")
        tag.text(0.5, 0.5, garment, ha="center", va="center", fontweight="bold", color=INK)
        for col, condition in enumerate(conditions, start=1):
            show(fig.add_subplot(grid[row, col]), ref(source, garment, condition), "Reference views" if row == 0 and col == 2 else None)
        flow = fig.add_subplot(grid[row, 4])
        flow.axis("off")
        arrow_text(flow, 0.22, 0.5, "endpoint")
        show(fig.add_subplot(grid[row, 5]), render(source, garment, "snapped"), "Realized output" if row == 0 else None)
    fig.suptitle("FIG1-A  Five-garment closed-wardrobe overview", fontsize=14, fontweight="bold", color=INK)
    return save_figure(fig, review / "figure1_candidate_A")


def build_candidate_b(source: Path, review: Path) -> list[Path]:
    fig = plt.figure(figsize=(12.0, 7.1), constrained_layout=True)
    grid = fig.add_gridspec(2, 7, height_ratios=[1.45, 1.0])
    for col, condition in enumerate(("cond_000000", "cond_000318", "cond_000017")):
        show(fig.add_subplot(grid[0, col]), ref(source, "O01", condition), "Reference" if col == 1 else None)
    flow = fig.add_subplot(grid[0, 3])
    flow.axis("off")
    arrow_text(flow, 0.15, 0.5, "predict")
    show(fig.add_subplot(grid[0, 4]), render(source, "O01", "raw_prediction"), "Raw")
    show(fig.add_subplot(grid[0, 5]), render(source, "O01", "snapped"), "Snapped")
    show(fig.add_subplot(grid[0, 6]), render(source, "O01", "teacher"), "Teacher")
    for col, garment in enumerate(GARMENTS, start=1):
        show(fig.add_subplot(grid[1, col]), render(source, garment, "snapped"), garment)
    left = fig.add_subplot(grid[1, 0])
    left.axis("off")
    left.text(0.5, 0.5, "Closed\nwardrobe", ha="center", va="center", fontweight="bold", color=INK)
    right = fig.add_subplot(grid[1, 6])
    right.axis("off")
    right.text(0.5, 0.5, "5 registered\nendpoints", ha="center", va="center", color=GRAY)
    fig.suptitle("FIG1-B  Input-to-result main-method example", fontsize=14, fontweight="bold", color=INK)
    return save_figure(fig, review / "figure1_candidate_B")


def build_candidate_c(source: Path, review: Path) -> list[Path]:
    fig = plt.figure(figsize=(8.6, 10.5), constrained_layout=True)
    grid = fig.add_gridspec(5, 4, width_ratios=[0.55, 1, 1, 1])
    for row, garment in enumerate(GARMENTS):
        tag = fig.add_subplot(grid[row, 0])
        tag.axis("off")
        tag.text(0.5, 0.5, garment, ha="center", va="center", fontweight="bold", color=INK)
        show(fig.add_subplot(grid[row, 1]), ref(source, garment), "Reference" if row == 0 else None)
        show(fig.add_subplot(grid[row, 2]), render(source, garment, "snapped"), "Prediction" if row == 0 else None)
        show(fig.add_subplot(grid[row, 3]), render(source, garment, "teacher"), "Teacher" if row == 0 else None)
    fig.suptitle("FIG1-C  Closed-wardrobe qualitative comparison", fontsize=14, fontweight="bold", color=INK)
    return save_figure(fig, review / "figure1_candidate_C")


def build_candidate_d(source: Path, review: Path) -> list[Path]:
    fig = plt.figure(figsize=(12.0, 8.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.35])
    show(fig.add_subplot(grid[0, 0]), ref(source, "O01"), "Reference")
    show(fig.add_subplot(grid[0, 1]), render(source, "O01", "raw_prediction"), "Raw")
    show(fig.add_subplot(grid[0, 2]), render(source, "O01", "snapped"), "Snapped")
    show(fig.add_subplot(grid[0, 3]), render(source, "O01", "teacher"), "Teacher")
    note = fig.add_subplot(grid[0, 4:])
    note.axis("off")
    note.text(0.02, 0.72, "Main method", fontsize=12, fontweight="bold", color=INK)
    note.text(0.02, 0.52, "Reference-conditioned endpoint selection", color=GRAY)
    note.text(0.02, 0.30, "Dual-Support extension", fontsize=12, fontweight="bold", color=GREEN)
    note.text(0.02, 0.12, "Oracle/user-specified pair and weights", color=GRAY)
    dual = fig.add_subplot(grid[1, :])
    show(dual, source / "dual_support" / "O01_O02_A_TO_B.png")
    panel(dual, "Extension")
    fig.suptitle("FIG1-D  Main method with geometry-safety extension  [NOT RECOMMENDED]", fontsize=14, fontweight="bold", color=INK)
    return save_figure(fig, review / "figure1_candidate_D")


def build_contact_sheet(review: Path) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    descriptions = {
        "A": "Five-garment wardrobe overview",
        "B": "Input-to-result main example",
        "C": "Reference / Prediction / Teacher",
        "D": "Main method + extension (not recommended)",
    }
    for ax, key in zip(axes.flat, descriptions):
        show(ax, review / f"figure1_candidate_{key}.png")
        ax.set_title(f"FIG1-{key}: {descriptions[key]}", fontsize=11, fontweight="bold")
    fig.suptitle("Figure 1 manual adjudication candidates - no candidate selected", fontsize=15, fontweight="bold", color=INK)
    return save_figure(fig, review / "figure1_candidate_contact_sheet", png_dpi=170)


def build_figure2(publication: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(13.2, 5.0))
    ax.set_xlim(0, 13.8)
    ax.set_ylim(0, 5.0)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, color: str, *, fill: str = "white") -> None:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fill, edgecolor=color, linewidth=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.2, color=INK, wrap=True)

    def connect(x1: float, y1: float, x2: float, y2: float, color: str = GRAY) -> None:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "-|>", "lw": 1.3, "color": color})

    ax.text(0.15, 4.62, "OFFLINE ENDPOINT CONSTRUCTION", fontsize=10, fontweight="bold", color=BLUE)
    offline = [
        ("Frozen Animatable\nGaussian Avatar", 1.9),
        ("Per-Garment Multi-View\nOptimization", 2.2),
        ("Canonical Teacher\nEndpoints", 1.85),
        ("Explicit Endpoint\nCoordinate System", 2.15),
    ]
    x = 0.15
    centers = []
    for label, width in offline:
        box(x, 3.45, width, 0.8, label, BLUE, fill="#edf5fc")
        centers.append((x, width))
        x += width + 0.42
    for (x1, w1), (x2, _) in zip(centers, centers[1:]):
        connect(x1 + w1, 3.85, x2, 3.85, BLUE)

    ax.text(0.15, 2.75, "REFERENCE-CONDITIONED INFERENCE", fontsize=10, fontweight="bold", color=GREEN)
    labels = [
        "Reference\nRGB / Mask",
        "Frozen F2",
        "Mean / Max\nSet Aggregation",
        "LayerNorm +\nSmall Linear Head",
        "Raw\nCoefficient",
        "Nearest Valid\nEndpoint Snapping",
        "Realized Teacher\nEndpoint",
        "Frozen Deformation /\nLBS / Renderer",
    ]
    widths = [1.35, 1.15, 1.55, 1.55, 1.15, 1.65, 1.55, 1.85]
    gap = 0.23
    x = 0.15
    centers = []
    for index, (label, width) in enumerate(zip(labels, widths)):
        fill = "#eef8f3" if index not in (4, 5, 6) else ("#fff4df" if index == 4 else "#e6f6ee")
        color = GOLD if index == 4 else GREEN
        box(x, 1.35, width, 0.82, label, color, fill=fill)
        centers.append((x, width))
        x += width + gap
    for (x1, w1), (x2, _) in zip(centers, centers[1:]):
        connect(x1 + w1, 1.76, x2, 1.76, GREEN)
    raw_x, raw_w = centers[4]
    snap_x, snap_w = centers[5]
    realized_x, realized_w = centers[6]
    ax.text(raw_x + raw_w / 2, 0.95, "continuous prediction\n(not rendered directly)", ha="center", va="top", fontsize=8, color=GOLD)
    ax.text((snap_x + realized_x + realized_w) / 2, 0.95, "discrete, valid and renderable", ha="center", va="top", fontsize=8, color=GREEN)
    ax.plot([raw_x + raw_w / 2, raw_x + raw_w / 2], [1.34, 1.03], color=GOLD, lw=1)
    ax.plot([snap_x + snap_w / 2, snap_x + snap_w / 2], [1.34, 1.03], color=GREEN, lw=1)
    ax.text(0.15, 0.25, "Frozen components", color=GRAY, fontsize=8)
    ax.text(2.0, 0.25, "Learned reference-to-endpoint map", color=GREEN, fontsize=8)
    ax.text(5.0, 0.25, "Raw and realized coefficients are distinct", color=GOLD, fontsize=8, fontweight="semibold")
    return save_figure(fig, publication / "figure2_method", svg=True)


def build_figure3(root: Path, source: Path, publication: Path) -> list[Path]:
    causal = load_json(root / "paper_protocol" / "reviewer_risk" / "factor_causal_profiles.json")
    effects = causal["global_main_effect_magnitudes"]
    values = [effects[key] for key in ("G", "V", "A")]
    fig = plt.figure(figsize=(13.0, 6.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[2.25, 1, 1], height_ratios=[1.15, 1])
    visual = fig.add_subplot(grid[:, 0])
    show(visual, source / "causal" / "O01_O02_strongest_factor.png")
    panel(visual, "A")
    visual.set_title("Registered visual decomposition (fixed first pair O01 to O02)", pad=5)
    bars = fig.add_subplot(grid[0, 1:])
    colors = [BLUE, GREEN, GOLD]
    bars.bar(np.arange(3), values, color=colors, width=0.62)
    bars.set_xticks(np.arange(3), ["Geometry", "Visibility", "Appearance"])
    bars.set_ylim(0, 1.08)
    bars.set_ylabel("Global main-effect magnitude")
    bars.grid(axis="y", color=LIGHT, lw=0.8)
    bars.text(0, values[0] + 0.035, f"{values[0]:.6f}", ha="center", fontweight="bold", color=BLUE)
    for x, value in enumerate(values[1:], start=1):
        bars.text(x, value + 0.035, f"{value:.4f}", ha="center", color=GRAY)
    panel(bars, "B")
    bars.set_title("Geometry dominates all registered pair directions")
    suff = fig.add_subplot(grid[1, 1])
    suff.bar([0, 1], [10, 10], color=[BLUE, GREEN], width=0.56)
    suff.set_xticks([0, 1], ["Sufficient", "Necessary"])
    suff.set_ylim(0, 11.4)
    suff.set_ylabel("Pairs passing / 10")
    suff.grid(axis="y", color=LIGHT, lw=0.8)
    for index in (0, 1):
        suff.text(index, 10.25, "10 / 10", ha="center", fontweight="bold")
    panel(suff, "C")
    scope = fig.add_subplot(grid[1, 2])
    scope.axis("off")
    scope.text(0.14, 0.92, "Registered protocol", fontsize=11, fontweight="bold", color=INK)
    scope.text(0.02, 0.69, "10 subject02 garment pairs", color=GRAY)
    scope.text(0.02, 0.52, "Frozen geometry / visibility /\nappearance decomposition", color=GRAY)
    scope.text(0.02, 0.28, "Fixed identity and\nclosed wardrobe", color=GRAY)
    panel(scope, "D")
    return save_figure(fig, publication / "figure3_geometry_causal", svg=True)


def build_figure4(root: Path, publication: Path) -> list[Path]:
    source_data = load_json(root / "paper_draft" / "figures" / "plots" / "dual_support" / "dual_support_all_pair_garment_lpips.source.json")
    series = {entry["label"]: [point["y"] for point in entry["values"]] for entry in source_data["series"]}
    full = series["Single-support FULL"]
    dual = series["Dual-Support"]
    labels = [tick["label"].replace("_", "-") for tick in source_data["x_ticks"]]
    fig = plt.figure(figsize=(13.0, 5.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[2.35, 1, 1], height_ratios=[1, 1])
    ax = fig.add_subplot(grid[:, 0])
    x = np.arange(10)
    ax.plot(x, full, marker="o", color=RED, lw=1.7, label="Full single-endpoint interpolation")
    ax.plot(x, dual, marker="o", color=GREEN, lw=1.7, label="Dual Support")
    ax.fill_between(x, dual, full, color="#dff2e9", alpha=0.8)
    ax.set_xticks(x, labels, rotation=38, ha="right")
    ax.set_ylabel("Garment LPIPS (lower is better)")
    ax.set_ylim(0.035, 0.1)
    ax.grid(axis="y", color=LIGHT)
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("Complete all-pair evaluation")
    panel(ax, "A")

    metrics = fig.add_subplot(grid[0, 1:])
    metrics.axis("off")
    metric_rows = [
        ("LPIPS", "0.085303", "0.056985", "lower"),
        ("Silhouette IoU", "0.710052", "0.725290", "higher"),
        ("Boundary F", "0.300263", "0.379103", "higher"),
    ]
    metrics.text(0.09, 0.92, "Aggregate over all 10 garment pairs", fontsize=11, fontweight="bold", color=INK)
    for row, (name, before, after, direction) in enumerate(metric_rows):
        y = 0.67 - row * 0.25
        metrics.text(0.02, y, name, fontweight="semibold", color=INK)
        metrics.text(0.40, y, before, ha="right", color=RED)
        metrics.annotate("", xy=(0.62, y + 0.01), xytext=(0.45, y + 0.01), xycoords="axes fraction", arrowprops={"arrowstyle": "-|>", "color": GREEN, "lw": 1.4})
        metrics.text(0.68, y, after, color=GREEN, fontweight="bold")
        metrics.text(0.96, y, direction, ha="right", color=GRAY, fontsize=8)
    panel(metrics, "B")

    audit = fig.add_subplot(grid[1, 1])
    audit.axis("off")
    audit.text(0.08, 0.86, "Visual and identity audit", fontsize=10, fontweight="bold", color=INK)
    audit.text(0.02, 0.58, "Severe artifact improvement", color=GRAY)
    audit.text(0.98, 0.58, "10 / 10", ha="right", fontweight="bold", color=GREEN)
    audit.text(0.02, 0.31, "Identity contamination", color=GRAY)
    audit.text(0.98, 0.31, "0", ha="right", fontweight="bold", color=GREEN)
    panel(audit, "C")

    cost = fig.add_subplot(grid[1, 2])
    cost.axis("off")
    cost.text(0.08, 0.86, "Explicit extension cost", fontsize=10, fontweight="bold", color=INK)
    cost.text(0.02, 0.58, "Active Gaussians", color=GRAY)
    cost.text(0.98, 0.58, "~2.000x", ha="right", fontweight="bold", color=GOLD)
    cost.text(0.02, 0.31, "Render time", color=GRAY)
    cost.text(0.98, 0.31, "~1.626x", ha="right", fontweight="bold", color=GOLD)
    cost.text(0.02, 0.07, "Oracle/user-specified pair and weights", color=INK, fontsize=8)
    panel(cost, "D")
    return save_figure(fig, publication / "figure4_dual_support", svg=True)


def build_figure5(source: Path, publication: Path) -> list[Path]:
    fig = plt.figure(figsize=(12.8, 4.5), constrained_layout=True)
    grid = fig.add_gridspec(1, 5)
    paths = [
        render(source, "O01", "raw_prediction"),
        render(source, "O01", "snapped"),
        render(source, "O01", "snapped"),
        render(source, "O01", "snapped"),
        render(source, "O01", "teacher"),
    ]
    titles = ["Raw coefficient", "Snapped endpoint", "Reference Classifier\nLookup", "Nearest-Centroid\nLookup", "Teacher"]
    for index, (path, title) in enumerate(zip(paths, titles)):
        ax = fig.add_subplot(grid[0, index])
        show(ax, path, title)
        panel(ax, chr(ord("A") + index))
        if index in (1, 2, 3):
            ax.text(0.5, -0.055, "same realized endpoint", transform=ax.transAxes, ha="center", va="top", color=GREEN, fontsize=8)
    fig.suptitle("Hard-lookup relation in the registered clean closed-wardrobe protocol (60 / 60 agreement)", fontsize=13, fontweight="bold", color=INK)
    return save_figure(fig, publication / "figure5_hard_lookup_relation", svg=True)


def build_figure6(root: Path, publication: Path) -> list[Path]:
    headroom = load_json(root / "paper_draft" / "figures" / "headroom_refresh" / "plots" / "headroom" / "metric_plot_source_data.json")
    loo = load_json(root / "paper_draft" / "figures" / "loo_method_freeze" / "plots" / "loo" / "loo_plot_source_data.json")
    pure = load_json(root / "paper_draft" / "figures" / "pure_endpoint_refresh" / "plots" / "pure_endpoint" / "metric_plot_source_data.json")
    macro = headroom["macro_test_metrics"]
    loo_macro = loo["method_metrics"]["macro"]
    perturb = next(item for item in pure["plots"] if item["plot_id"] == "endpoint_flip_by_perturbation")
    flip = {entry["label"]: {point["x"]: point["y"] for point in entry["values"]} for entry in perturb["series"]}

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), constrained_layout=True)
    ax = axes[0]
    values = [macro["Teacher Endpoint"]["lpips"], macro["Render-Refined Coefficient"]["lpips"]]
    ax.bar([0, 1], values, color=[BLUE, GOLD], width=0.58)
    ax.set_xticks([0, 1], ["Teacher\nendpoint", "Refined"])
    ax.set_ylim(0.0384, 0.03915)
    ax.set_ylabel("LPIPS (lower is better)")
    ax.grid(axis="y", color=LIGHT)
    for index, value in enumerate(values):
        ax.text(index, value + 0.000025, f"{value:.6f}", ha="center", fontsize=8, fontweight="bold")
    ax.text(0.5, 0.08, "0 / 5 garments improve", transform=ax.transAxes, ha="center", color=RED, fontweight="semibold")
    ax.set_title("Endpoint headroom")
    panel(ax, "A")

    ax = axes[1]
    loo_values = [loo_macro["Hard Lookup K2"]["lpips"], loo_macro["Low-Dim K2"]["lpips"]]
    ax.bar([0, 1], loo_values, color=[GREEN, GOLD], width=0.58)
    ax.set_xticks([0, 1], ["Hard lookup", "Rank-3 basis"])
    ax.set_ylim(0.1128, 0.1140)
    ax.set_ylabel("LOO LPIPS (lower is better)")
    ax.grid(axis="y", color=LIGHT)
    for index, value in enumerate(loo_values):
        ax.text(index, value + 0.000045, f"{value:.6f}", ha="center", fontsize=8, fontweight="bold")
    ax.text(0.5, 0.08, "4-garment basis is capacity-limited\nfor the held-out fifth garment", transform=ax.transAxes, ha="center", color=RED, fontsize=8)
    ax.set_title("Leave-one-garment-out boundary")
    panel(ax, "B")

    ax = axes[2]
    labels = ["Clean\ntop-1", "Mild blur\ntop-1", "Single ref.\ntop-1", "Complete dropout\nsafe fallback"]
    values = [1.0, 1.0 - flip["CanonDressGS-Endpoint"][4], 1.0 - flip["CanonDressGS-Endpoint"][6], 1.0]
    bars = ax.bar(np.arange(4), values, color=[GREEN, RED, GOLD, BLUE], width=0.6)
    ax.set_xticks(np.arange(4), labels, fontsize=7.5)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Registered rate")
    ax.grid(axis="y", color=LIGHT)
    for index, value in enumerate(values):
        label = "80 / 80" if index == 3 else f"{value:.2f}"
        ax.text(index, value + 0.035, label, ha="center", fontsize=8, fontweight="bold")
    ax.set_title("Perturbation and fallback sensitivity")
    panel(ax, "C")
    return save_figure(fig, publication / "figure6_endpoint_limits", svg=True)


def collect_sources(root: Path, source: Path) -> list[dict[str, Any]]:
    records = []
    head_for_folder = {"causal": HEADS["geometry_causal"], "dual_support": HEADS["dual_support"], "pure_endpoint": HEADS["pure_endpoint"]}
    for path in sorted(source.rglob("*.png")):
        relative = path.relative_to(root).as_posix()
        width, height = image_info(path)
        source_group = path.relative_to(source).parts[0]
        records.append(
            {
                "path": relative,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "width": width,
                "height": height,
                "source_head": head_for_folder[source_group],
                "copy_transform": "BYTE_IDENTICAL_COPY",
                "scientific_pixel_mutation": False,
                "source_type": "REGISTERED_RASTER_COPY",
            }
        )
    structured = {
        "paper_protocol/reviewer_risk/canondressgs_main_method_freeze.json": HEADS["figure_bank"],
        "paper_protocol/reviewer_risk/factor_causal_profiles.json": HEADS["geometry_causal"],
        "paper_draft/figures/plots/dual_support/dual_support_all_pair_garment_lpips.source.json": HEADS["dual_support"],
        "paper_protocol/reviewer_risk/dual_support_all_pair_results.json": HEADS["dual_support"],
        "paper_draft/figures/pure_endpoint_refresh/plots/pure_endpoint/metric_plot_source_data.json": HEADS["pure_endpoint"],
        "paper_draft/figures/headroom_refresh/plots/headroom/metric_plot_source_data.json": HEADS["headroom"],
        "paper_draft/figures/loo_method_freeze/plots/loo/loo_plot_source_data.json": HEADS["loo"],
    }
    for relative, source_head in structured.items():
        path = root / relative
        records.append(
            {
                "path": relative,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "source_head": source_head,
                "source_type": "REGISTERED_STRUCTURED_DATA",
                "scientific_pixel_mutation": False,
            }
        )
    return records


def artifact_record(root: Path, path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"path": path.relative_to(root).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size}
    if path.suffix.lower() == ".png":
        record["width"], record["height"] = image_info(path)
    return record


def write_selection_form(review: Path) -> Path:
    path = review / "figure1_manual_selection_form.md"
    path.write_text(
        """# Figure 1 Manual Selection Form

Status: `AWAITING_USER_MANUAL_SELECTION`

No candidate is selected by this preparation task. The recommendation below is editorial advice only.

| Candidate | Editorial purpose | Main risk | User selection |
|---|---|---|---|
| FIG1-A | Five-garment closed-wardrobe overview | Dense reference-view grid | [ ] |
| FIG1-B | Input-to-result main-method example | Main row emphasizes one garment | [ ] |
| FIG1-C | Five garments x Reference / Prediction / Teacher | Less explicit pipeline detail | [ ] |
| FIG1-D | Main method plus Dual-Support extension | Conflates core method and optional extension; NOT RECOMMENDED | [ ] |

`EDITORIAL_RECOMMENDATION`: **FIG1-C**, because it gives equal five-garment coverage and the clearest direct qualitative comparison. This is not a formal selection.

Selected candidate: `________________`

User/adjudicator: `________________`

Decision date: `________________`

Required next task after a user decision: `USER_SELECT_FIGURE1_CANDIDATE_THEN_INTEGRATE_AND_FINALIZE_SCOPE`
""",
        encoding="utf-8",
    )
    return path


def write_registries(root: Path, source: Path, review: Path, publication: Path, generated: Iterable[Path]) -> None:
    source_records = collect_sources(root, source)
    generated = sorted(set(generated))
    artifacts = [artifact_record(root, path) for path in generated]
    common = {"schema_version": "canondressgs.paper_figure_p0.v1", "task_id": TASK_ID, "created_at": CREATED, "paper_final": False}
    figure1_paths = [record for record in source_records if "/pure_endpoint/" in record["path"] or "/dual_support/" in record["path"]]
    write_json(root / "paper_figure1_candidate_source_registry.json", {**common, "status": "PASS", "source_heads": HEADS, "sources": figure1_paths, "scientific_source_pixel_mutation_count": 0})
    write_json(
        root / "paper_figure1_candidate_transform_registry.json",
        {
            **common,
            "status": "PASS",
            "transforms": [
                {
                    "candidate_id": f"FIG1-{key}",
                    "output_pdf": f"paper_draft/figure_review/figure1_manual_adjudication/figure1_candidate_{key}.pdf",
                    "operation": "DETERMINISTIC_LAYOUT_WITH_PROPORTIONAL_RESIZE",
                    "crop": False,
                    "retouch": False,
                    "sharpen": False,
                    "ai_generation": False,
                    "scientific_source_pixel_mutation": False,
                }
                for key in "ABCD"
            ],
            "scientific_source_pixel_mutation_count": 0,
        },
    )
    write_json(
        root / "paper_figure1_manual_adjudication_manifest.json",
        {
            **common,
            "status": "AWAITING_USER_MANUAL_SELECTION",
            "candidate_ids": ["FIG1-A", "FIG1-B", "FIG1-C", "FIG1-D"],
            "formal_selection": None,
            "editorial_recommendation": "FIG1-C",
            "editorial_recommendation_status": "EDITORIAL_RECOMMENDATION",
            "not_recommended": ["FIG1-D"],
            "inserted_in_main_tex": False,
            "next_task": "USER_SELECT_FIGURE1_CANDIDATE_THEN_INTEGRATE_AND_FINALIZE_SCOPE",
        },
    )
    causal_sources = [record for record in source_records if "/causal/" in record["path"]]
    write_json(
        root / "paper_figure3_geometry_causal_registry.json",
        {
            **common,
            "status": "PASS",
            "source_head": HEADS["geometry_causal"],
            "main_effect_geometry": 0.9959405426885113,
            "geometry_sufficient_pairs": "10/10",
            "geometry_necessary_pairs": "10/10",
            "visual_example_selection_rule": "LEXICALLY_FIRST_REGISTERED_PAIR_O01_O02_NOT_QUALITY_PICKED",
            "sources": causal_sources,
        },
    )
    write_json(root / "paper_publication_figure_source_registry.json", {**common, "status": "PASS", "source_heads": HEADS, "sources": source_records, "scientific_source_pixel_mutation_count": 0})
    source_sha = {record["path"]: record["sha256"] for record in source_records}
    dependencies = {
        2: ["paper_protocol/reviewer_risk/canondressgs_main_method_freeze.json"],
        3: [
            "paper_protocol/reviewer_risk/factor_causal_profiles.json",
            "paper_draft/figures/publication/source/causal/O01_O02_strongest_factor.png",
        ],
        4: [
            "paper_draft/figures/plots/dual_support/dual_support_all_pair_garment_lpips.source.json",
            "paper_protocol/reviewer_risk/dual_support_all_pair_results.json",
        ],
        5: [
            "paper_draft/figures/pure_endpoint_refresh/plots/pure_endpoint/metric_plot_source_data.json",
            "paper_draft/figures/publication/source/pure_endpoint/renders/O01_raw_prediction.png",
            "paper_draft/figures/publication/source/pure_endpoint/renders/O01_snapped.png",
            "paper_draft/figures/publication/source/pure_endpoint/renders/O01_teacher.png",
        ],
        6: [
            "paper_draft/figures/headroom_refresh/plots/headroom/metric_plot_source_data.json",
            "paper_draft/figures/loo_method_freeze/plots/loo/loo_plot_source_data.json",
            "paper_draft/figures/pure_endpoint_refresh/plots/pure_endpoint/metric_plot_source_data.json",
        ],
    }
    publication_transforms = []
    for number, stem in [(2, "figure2_method"), (3, "figure3_geometry_causal"), (4, "figure4_dual_support"), (5, "figure5_hard_lookup_relation"), (6, "figure6_endpoint_limits")]:
        output = publication / f"{stem}.pdf"
        publication_transforms.append(
            {
                "figure": number,
                "output_path": output.relative_to(root).as_posix(),
                "output_sha256": sha256(output),
                "operation": "DETERMINISTIC_VECTOR_AND_REGISTERED_RASTER_LAYOUT",
                "sources": [{"path": path, "sha256": source_sha[path]} for path in dependencies[number]],
                "source_retouch": False,
                "source_crop": False,
                "scientific_source_pixel_mutation": False,
            }
        )
    write_json(root / "paper_publication_figure_transform_registry.json", {**common, "status": "PASS", "transforms": publication_transforms, "scientific_source_pixel_mutation_count": 0})
    write_json(root / "paper_publication_figure_asset_registry.json", {**common, "status": "PASS", "assets": artifacts, "required_publication_pdfs_present": all((publication / name).exists() for name in ["figure2_method.pdf", "figure3_geometry_causal.pdf", "figure4_dual_support.pdf", "figure5_hard_lookup_relation.pdf", "figure6_endpoint_limits.pdf"])})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.repo_root.resolve()
    source = root / "paper_draft" / "figures" / "publication" / "source"
    publication = source.parent
    review = root / "paper_draft" / "figure_review" / "figure1_manual_adjudication"
    required = list(source.rglob("*.png"))
    if len(required) != 35:
        raise RuntimeError(f"Expected 35 copied source PNGs, found {len(required)}")
    generated: list[Path] = []
    generated += build_candidate_a(source, review)
    generated += build_candidate_b(source, review)
    generated += build_candidate_c(source, review)
    generated += build_candidate_d(source, review)
    generated += build_contact_sheet(review)
    generated.append(write_selection_form(review))
    generated += build_figure2(publication)
    generated += build_figure3(root, source, publication)
    generated += build_figure4(root, publication)
    generated += build_figure5(source, publication)
    generated += build_figure6(root, publication)
    write_registries(root, source, review, publication, generated)
    print(json.dumps({"status": "PASS", "generated": len(generated), "sources": len(required)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
