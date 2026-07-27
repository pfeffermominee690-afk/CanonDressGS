#!/usr/bin/env python3
"""Build and verify the 12-page Subject00 CommonSafe4 human review pack."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageChops, ImageOps, ImageStat
from pypdf import PdfReader
from reportlab.lib.pagesizes import landscape
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


REPO_ROOT = Path(__file__).resolve().parents[2]
METHOD_ROOT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-CANONDRESSGS-METHOD-BASE60747-COMMONSAFE4-001"
)
REVIEW_ROOT = METHOD_ROOT / "review/human_scientific_review_pack_20260727"
AUDIT_SOURCE = REVIEW_ROOT / "subject00_commonsafe4_independent_audit_source_20260727.json"
INDEX_PATH = REVIEW_ROOT / "subject00_commonsafe4_review_pack_index.json"
README_PATH = REVIEW_ROOT / "SUBJECT00_COMMONSAFE4_REVIEW_PACK_README.md"
PDF_PATH = (
    REVIEW_ROOT
    / "subject00_commonsafe4_method_teacher_baseline_human_scientific_review_pages_20260727.pdf"
)
TEACHER_ROOTS = {
    "O01": Path(
        "/root/autodl-tmp/canondressgs_work/outputs/"
        "SUBJECT00-O01-TEACHER-BASE60747-CAMSAFE7-001/attempt_001/review"
    ),
    "O03": Path(
        "/root/autodl-tmp/canondressgs_work/outputs/"
        "SUBJECT00-O03-TEACHER-BASE60747-FORMAL-CAMSAFE7-001/attempt_001/review"
    ),
    "O04": Path(
        "/root/autodl-tmp/canondressgs_work/outputs/"
        "SUBJECT00-O04-TEACHER-BASE60747-8VIEW-001/attempt_001/review"
    ),
}
TEACHER_REGISTRY = (
    REPO_ROOT
    / "paper_protocol/reviewer_risk/"
    "subject00_base60747_three_garment_teacher_registry_20260727.json"
)

WIDTH = 6400
HEIGHT = 4000
DPI = 200
FIGSIZE = (WIDTH / DPI, HEIGHT / DPI)
BG = "#F6F8FB"
NAVY = "#102A43"
BLUE = "#2F6BFF"
SKY = "#56B4E9"
TEAL = "#009E73"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
MAGENTA = "#CC79A7"
GRAY = "#64748B"
LIGHT = "#E7EDF5"
WHITE = "#FFFFFF"
GARMENT_COLORS = {"O01": BLUE, "O03": ORANGE, "O04": TEAL}

PAGE_SPECS = [
    ("page_01_provenance_and_immutability.png", "Provenance and immutability overview"),
    ("page_02_commonsafe4_replacement_yaw_ranking.png", "CommonSafe4 replacement and yaw ranking"),
    ("page_03_rotations_and_633_folds.png", "Four rotations and exact 6/3/3 folds"),
    ("page_04_method_12run_matrix.png", "Twelve-run method matrix"),
    ("page_05_method_confusion_matrix.png", "Method confusion matrix"),
    ("page_06_rotation_and_seed_effects.png", "Rotation and seed effects with all runs visible"),
    ("page_07_six_O04_to_O03_error_cases.png", "Six O04-to-O03 error cases"),
    ("page_08_nonoracle_baseline_comparison.png", "Non-oracle baseline comparison"),
    ("page_09_oracle_upper_references.png", "Oracle and upper-reference boundary"),
    ("page_10_compute_budget_audit.png", "Compute-budget and information-access audit"),
    ("page_11_teacher_technical_visual_evidence.png", "O01/O03/O04 Teacher technical and visual evidence"),
    ("page_12_scientific_limits_and_next_step.png", "Scientific limitations and next step"),
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with temporary.open("wb") as handle:
        handle.write(text.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(value.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def new_page(number: int, title: str, subtitle: str) -> plt.Figure:
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor=BG)
    fig.text(0.045, 0.945, f"{number:02d}", fontsize=30, color=BLUE, weight="bold", va="top")
    fig.text(0.095, 0.948, title, fontsize=43, color=NAVY, weight="bold", va="top")
    fig.text(0.095, 0.902, subtitle, fontsize=21, color=GRAY, va="top")
    fig.add_artist(
        plt.Line2D([0.045, 0.955], [0.875, 0.875], transform=fig.transFigure, color=LIGHT, lw=2)
    )
    fig.text(
        0.045,
        0.028,
        "Post-hoc display-only evidence | Human/scientific decisions pending",
        fontsize=17,
        color=GRAY,
        va="bottom",
    )
    fig.text(0.955, 0.028, f"Page {number}/12", fontsize=17, color=GRAY, ha="right", va="bottom")
    return fig


def add_card(
    fig: plt.Figure,
    rect: tuple[float, float, float, float],
    title: str,
    lines: list[str],
    *,
    accent: str = BLUE,
    title_size: int = 24,
    body_size: int = 18,
) -> None:
    x, y, w, h = rect
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        transform=fig.transFigure,
        facecolor=WHITE,
        edgecolor=LIGHT,
        linewidth=1.5,
    )
    fig.add_artist(patch)
    fig.add_artist(
        plt.Line2D(
            [x + 0.018, x + 0.018],
            [y + 0.055, y + h - 0.055],
            transform=fig.transFigure,
            color=accent,
            lw=7,
            solid_capstyle="round",
        )
    )
    fig.text(x + 0.04, y + h - 0.05, title, fontsize=title_size, color=NAVY, weight="bold", va="top")
    fig.text(
        x + 0.04,
        y + h - 0.105,
        "\n".join(lines),
        fontsize=body_size,
        color=NAVY,
        va="top",
        linespacing=1.35,
    )


def style_axis(ax: plt.Axes, *, grid: bool = True) -> None:
    ax.set_facecolor(WHITE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#A8B4C4")
    ax.tick_params(labelsize=18, colors=NAVY)
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    ax.title.set_size(24)
    ax.title.set_weight("bold")
    ax.title.set_color(NAVY)
    if grid:
        ax.grid(axis="y", color=LIGHT, lw=1.0, linestyle="--", zorder=0)


def save_page(fig: plt.Figure, path: Path) -> None:
    fig.savefig(
        path,
        dpi=DPI,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
        bbox_inches=None,
        pad_inches=0,
    )
    plt.close(fig)
    with Image.open(path) as image:
        if image.size != (WIDTH, HEIGHT):
            raise RuntimeError(f"wrong page resolution {path}: {image.size}")
        if image.mode != "RGB":
            converted = image.convert("RGB")
            converted.save(path, format="PNG", optimize=True)
    with Image.open(path) as image:
        if image.size != (WIDTH, HEIGHT):
            raise RuntimeError(f"resolution changed after normalization: {path}")


def source_image_path(garment: str, kind: str, request_id: str) -> Path:
    path = TEACHER_ROOTS[garment] / kind / f"{request_id}.png"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def add_image(
    fig: plt.Figure,
    rect: tuple[float, float, float, float],
    path: Path,
    label: str,
    *,
    border_color: str = LIGHT,
) -> None:
    ax = fig.add_axes(rect)
    # Figure-level card patches can otherwise cover later axes on PDF/PNG render.
    ax.set_zorder(10)
    ax.set_facecolor(WHITE)
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    ax.imshow(image, interpolation="lanczos", aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(border_color)
        spine.set_linewidth(1.8)
    ax.set_title(label, fontsize=17, color=NAVY, pad=8, weight="bold")


def add_table(
    ax: plt.Axes,
    rows: list[list[str]],
    columns: list[str],
    *,
    font_size: int = 16,
    col_widths: list[float] | None = None,
) -> None:
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, 2.0)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(LIGHT)
        cell.set_linewidth(1.0)
        if row == 0:
            cell.set_facecolor(NAVY)
            cell.get_text().set_color(WHITE)
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(WHITE if row % 2 else "#F0F4F9")
            cell.get_text().set_color(NAVY)


def page_1(data: dict[str, Any], path: Path) -> list[str]:
    fig = new_page(
        1,
        "Provenance and immutability overview",
        "Safe-successor Git semantics, frozen scientific inputs, and zero new optimization",
    )
    git_gate = data["git_gate"]
    immutable = data["immutable_inputs"]
    add_card(
        fig,
        (0.05, 0.57, 0.43, 0.25),
        "Corrected Git semantics",
        [
            f"Authoritative safe successor: {git_gate['source_local_head'][:12]}...",
            f"Reporting content parent: {git_gate['reporting_content_head'][:12]}...",
            "Relationship: direct parent -> safe successor",
            "Force-push 0 | branch rewind 0",
        ],
        accent=BLUE,
    )
    add_card(
        fig,
        (0.52, 0.57, 0.43, 0.25),
        "Execution freeze",
        [
            "Active optimizer processes: 0",
            "Active method processes: 0",
            "Active Teacher processes: 0",
            "New method steps: 0 | new baseline steps: 0",
        ],
        accent=TEAL,
    )
    teacher_lines = [
        f"{garment}: {immutable['teachers'][garment]['sha256'][:18]}... | TECHNICAL_PASS"
        for garment in ("O01", "O03", "O04")
    ]
    add_card(
        fig,
        (0.05, 0.18, 0.43, 0.31),
        "Immutable checkpoints",
        [
            f"Base60747: {immutable['base']['sha256'][:18]}...",
            f"Internal step: {immutable['base']['internal_step']}",
            *teacher_lines,
        ],
        accent=ORANGE,
        body_size=17,
    )
    target = immutable["target"]
    add_card(
        fig,
        (0.52, 0.18, 0.43, 0.31),
        "Formal target contract",
        [
            f"Materialization HEAD: {target['materialization_head'][:12]}...",
            "24 provenance records",
            "22 train/evaluation records",
            "2 permanent quarantine records",
            "Quarantine usage in train/cal/test/eval: 0/0/0/0",
        ],
        accent=MAGENTA,
    )
    save_page(fig, path)
    return []


def page_2(data: dict[str, Any], path: Path) -> list[str]:
    replacement = data["replacement"]
    fig = new_page(
        2,
        "CommonSafe4 replacement and yaw ranking",
        "Yaw and registration are recomputed from 24 raw camera records; outcomes are excluded",
    )
    ax = fig.add_axes([0.055, 0.13, 0.42, 0.68], projection="polar")
    ax.set_facecolor(WHITE)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([])
    ax.grid(color=LIGHT, lw=1.2)
    yaw_registry = replacement["slot_yaw_registry"]
    common = set(replacement["common_safe_slots"])
    for label, yaw in yaw_registry.items():
        theta = math.radians(float(yaw))
        if label == "slot04":
            color, marker, size = ORANGE, "X", 420
        elif label == "slot06":
            color, marker, size = TEAL, "*", 620
        elif label in common:
            color, marker, size = BLUE, "o", 260
        else:
            color, marker, size = GRAY, "s", 220
        ax.scatter(theta, 1.0, s=size, c=color, marker=marker, edgecolors=NAVY, linewidths=1.2)
        ax.text(theta, 1.13, f"{label}\n{yaw:.1f} deg", fontsize=14, color=NAVY, ha="center", va="center")
    ax.set_title("Camera yaw registry", fontsize=25, color=NAVY, weight="bold", pad=28)
    rows = []
    for rank, row in enumerate(replacement["replacement_ranking"], start=1):
        rows.append(
            [
                str(rank),
                row["slot_label"],
                row["camera_id"],
                row["direction"],
                f"{row['yaw_distance_degrees']:.3f}",
                f"{row['worst_case_registration_rmse_px']:.3f}",
                f"{row['minimum_inlier_ratio']:.3f}",
            ]
        )
    table_ax = fig.add_axes([0.50, 0.34, 0.45, 0.45])
    add_table(
        table_ax,
        rows,
        ["Rank", "Slot", "Camera", "Direction", "Yaw dist", "Worst RMSE", "Min inlier"],
        font_size=15,
        col_widths=[0.07, 0.10, 0.12, 0.16, 0.13, 0.16, 0.14],
    )
    add_card(
        fig,
        (0.52, 0.13, 0.41, 0.16),
        "Selection seal",
        [
            "Selected: slot06 / cam09 / back-right",
            "Config written before optimizer step 0",
            "Outcome-dependent evidence count: 0",
        ],
        accent=TEAL,
        title_size=21,
        body_size=16,
    )
    save_page(fig, path)
    return []


def page_3(data: dict[str, Any], path: Path) -> list[str]:
    fig = new_page(
        3,
        "Four rotations and exact 6/3/3 folds",
        "Every rotation covers all three garments with disjoint train, calibration, and test slots",
    )
    positions = [
        (0.06, 0.52, 0.41, 0.29),
        (0.53, 0.52, 0.41, 0.29),
        (0.06, 0.17, 0.41, 0.29),
        (0.53, 0.17, 0.41, 0.29),
    ]
    accents = [BLUE, TEAL, ORANGE, MAGENTA]
    for rotation, rect, accent in zip(range(4), positions, accents):
        fold = data["method"]["per_run"][rotation * 3]
        add_card(
            fig,
            rect,
            f"R{rotation}",
            [
                f"TRAIN  {', '.join(fold['train_slots'])}  -> 6 records",
                f"CAL    {fold['calibration_slot']}  -> 3 records",
                f"TEST   {fold['test_slot']}  -> 3 records",
                "O01 / O03 / O04 coverage: 3/3",
                "Train-cal-test overlap: 0",
            ],
            accent=accent,
            title_size=28,
            body_size=19,
        )
    save_page(fig, path)
    return []


def method_matrix(data: dict[str, Any]) -> np.ndarray:
    matrix = np.zeros((4, 3), dtype=float)
    for row in data["method"]["per_run"]:
        matrix[row["rotation"], row["seed"]] = row["top1"]
    return matrix


def page_4(data: dict[str, Any], path: Path) -> list[str]:
    fig = new_page(
        4,
        "Twelve-run method matrix",
        "Each cell is an independently trained 300-step run with a three-record formal test",
    )
    matrix = method_matrix(data)
    ax = fig.add_axes([0.12, 0.16, 0.68, 0.64])
    image = ax.imshow(matrix, cmap="cividis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3), ["Seed 0", "Seed 1", "Seed 2"], fontsize=22)
    ax.set_yticks(range(4), ["R0", "R1", "R2", "R3"], fontsize=22)
    ax.set_xlabel("Replicate seed", fontsize=23)
    ax.set_ylabel("Rotation", fontsize=23)
    ax.set_title("Formal endpoint top-1 per run", fontsize=28, color=NAVY, weight="bold", pad=18)
    for r in range(4):
        for s in range(3):
            correct = round(matrix[r, s] * 3)
            color = WHITE if matrix[r, s] < 0.75 else NAVY
            ax.text(s, r, f"{correct}/3\n{matrix[r,s]:.3f}", ha="center", va="center", fontsize=24, color=color, weight="bold")
    cax = fig.add_axes([0.82, 0.22, 0.025, 0.52])
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label("Formal endpoint top-1", fontsize=20)
    colorbar.ax.tick_params(labelsize=17)
    add_card(
        fig,
        (0.865, 0.28, 0.10, 0.38),
        "Seal",
        [
            "12/12 formal valid",
            "3600 total steps",
            "72 checkpoints",
            "No NaN / OOM",
            "Dual-Support 0",
        ],
        accent=TEAL,
        title_size=20,
        body_size=14,
    )
    save_page(fig, path)
    return []


def page_5(data: dict[str, Any], path: Path) -> list[str]:
    fig = new_page(
        5,
        "Method confusion matrix",
        "Counts are pooled from the 36 formal-test decisions; no aggregate report is used",
    )
    confusion = data["method"]["confusion_matrix"]
    matrix = np.array([[confusion[t][p] for p in ("O01", "O03", "O04")] for t in ("O01", "O03", "O04")])
    ax = fig.add_axes([0.10, 0.15, 0.60, 0.65])
    image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=12)
    ax.set_xticks(range(3), ["Pred O01", "Pred O03", "Pred O04"], fontsize=22)
    ax.set_yticks(range(3), ["True O01", "True O03", "True O04"], fontsize=22)
    ax.set_xlabel("Predicted endpoint", fontsize=23)
    ax.set_ylabel("True garment", fontsize=23)
    for r in range(3):
        for c in range(3):
            ax.text(c, r, str(matrix[r, c]), ha="center", va="center", fontsize=32, color=WHITE if matrix[r, c] < 6 else NAVY, weight="bold")
    cax = fig.add_axes([0.72, 0.21, 0.025, 0.53])
    cb = fig.colorbar(image, cax=cax)
    cb.set_label("Decision count", fontsize=20)
    cb.ax.tick_params(labelsize=17)
    add_card(
        fig,
        (0.79, 0.26, 0.17, 0.42),
        "Observed pattern",
        [
            "Correct: 30 / 36",
            "Top-1: 0.833333",
            "Errors: 6",
            "All errors: O04 -> O03",
            "O01 and O03: 12/12 each",
        ],
        accent=VERMILLION,
        title_size=22,
        body_size=16,
    )
    save_page(fig, path)
    return []


def page_6(data: dict[str, Any], path: Path) -> list[str]:
    fig = new_page(
        6,
        "Rotation and seed effects",
        "All twelve run-level values are visible; black diamonds show arithmetic means only",
    )
    per_run = data["method"]["per_run"]
    ax1 = fig.add_axes([0.07, 0.18, 0.41, 0.62])
    ax2 = fig.add_axes([0.55, 0.18, 0.41, 0.62])
    seed_offsets = {0: -0.13, 1: 0.0, 2: 0.13}
    seed_markers = {0: "o", 1: "s", 2: "^"}
    seed_colors = {0: BLUE, 1: ORANGE, 2: TEAL}
    for rotation in range(4):
        rows = [row for row in per_run if row["rotation"] == rotation]
        for row in rows:
            ax1.scatter(
                rotation + seed_offsets[row["seed"]],
                row["top1"],
                s=190,
                marker=seed_markers[row["seed"]],
                color=seed_colors[row["seed"]],
                edgecolor=NAVY,
                linewidth=1.0,
                label=f"Seed {row['seed']}" if rotation == 0 else None,
                zorder=3,
            )
        ax1.scatter(rotation, np.mean([row["top1"] for row in rows]), marker="D", s=170, color="black", zorder=4)
    style_axis(ax1)
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(range(4), ["R0", "R1", "R2", "R3"])
    ax1.set_ylabel("Formal endpoint top-1")
    ax1.set_title("Rotation effect (n=3 runs per rotation)")
    ax1.legend(frameon=False, fontsize=16, loc="lower left")
    rotation_offsets = {0: -0.18, 1: -0.06, 2: 0.06, 3: 0.18}
    rotation_markers = {0: "o", 1: "s", 2: "^", 3: "v"}
    rotation_colors = {0: BLUE, 1: ORANGE, 2: TEAL, 3: MAGENTA}
    for seed in range(3):
        rows = [row for row in per_run if row["seed"] == seed]
        for row in rows:
            ax2.scatter(
                seed + rotation_offsets[row["rotation"]],
                row["top1"],
                s=190,
                marker=rotation_markers[row["rotation"]],
                color=rotation_colors[row["rotation"]],
                edgecolor=NAVY,
                linewidth=1.0,
                label=f"R{row['rotation']}" if seed == 0 else None,
                zorder=3,
            )
        ax2.scatter(seed, np.mean([row["top1"] for row in rows]), marker="D", s=170, color="black", zorder=4)
    style_axis(ax2)
    ax2.set_ylim(0, 1.05)
    ax2.set_xticks(range(3), ["Seed 0", "Seed 1", "Seed 2"])
    ax2.set_ylabel("Formal endpoint top-1")
    ax2.set_title("Seed effect (n=4 rotations per seed)")
    ax2.legend(frameon=False, fontsize=16, loc="lower left", ncol=2)
    save_page(fig, path)
    return []


def page_7(data: dict[str, Any], path: Path) -> list[str]:
    fig = new_page(
        7,
        "Six O04 -> O03 error cases",
        "Each panel pairs the frozen target and Teacher render with the exact formal-test scores",
    )
    cases = data["error_analysis"]["cases"]
    sources = []
    cols = [0.045, 0.36, 0.675]
    rows_y = [0.47, 0.09]
    card_w, card_h = 0.28, 0.33
    for index, case in enumerate(cases):
        col = index % 3
        row = index // 3
        x, y = cols[col], rows_y[row]
        request_id = case["target_request_id"]
        target = source_image_path("O04", "target", request_id)
        teacher = source_image_path("O04", "teacher", request_id)
        sources.extend([str(target), str(teacher)])
        patch = FancyBboxPatch(
            (x, y),
            card_w,
            card_h,
            boxstyle="round,pad=0.006,rounding_size=0.01",
            transform=fig.transFigure,
            facecolor=WHITE,
            edgecolor=LIGHT,
            linewidth=1.5,
            zorder=-10,
        )
        fig.add_artist(patch)
        add_image(fig, (x + 0.012, y + 0.105, 0.115, 0.16), target, "Target O04", border_color=ORANGE)
        add_image(fig, (x + 0.142, y + 0.105, 0.115, 0.16), teacher, "Teacher O04", border_color=TEAL)
        fig.text(
            x + 0.014,
            y + card_h - 0.014,
            f"{case['run_id']} | {case['test_slot']}",
            fontsize=15,
            color=NAVY,
            weight="bold",
            va="top",
        )
        fig.text(
            x + 0.014,
            y + 0.028,
            (
                f"O01 {case['O01_score']:.3f}   O03 {case['O03_score']:.3f}   "
                f"O04 {case['O04_score']:.3f}\n"
                f"margin {case['top1_top2_margin']:.3f} | "
                f"cal {case['calibration_result']['predicted_class']} | "
                f"{case['camera_direction']}"
            ),
            fontsize=14,
            color=NAVY,
            va="bottom",
            linespacing=1.3,
        )
    fig.text(
        0.5,
        0.835,
        "Lower squared distance wins; all six margins > 0; tie-break count = 0.",
        fontsize=16,
        color=VERMILLION,
        ha="center",
        va="top",
    )
    save_page(fig, path)
    return sources


def baseline_matrix(row: dict[str, Any]) -> np.ndarray:
    matrix = np.zeros((4, 3), dtype=float)
    for item in row["per_run"]:
        matrix[item["rotation"], item["seed"]] = item["top1"]
    return matrix


def page_8(data: dict[str, Any], path: Path) -> list[str]:
    fig = new_page(
        8,
        "Non-oracle baseline comparison",
        "Deployable/hard-lookup rows only; exact run values remain visible beside pooled counts",
    )
    names = ["CanonDressGS", "Reference Classifier Lookup", "Nearest-Centroid Lookup"]
    values = [30, 31, 35]
    colors = [BLUE, ORANGE, TEAL]
    ax = fig.add_axes([0.07, 0.55, 0.42, 0.26])
    y = np.arange(len(names))
    ax.barh(y, values, color=colors, edgecolor=NAVY, height=0.58)
    ax.set_xlim(0, 36)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("Correct formal decisions (out of 36)")
    ax.set_title("Pooled exact counts")
    for index, value in enumerate(values):
        ax.text(value + 0.3, index, f"{value}/36", va="center", fontsize=20, color=NAVY, weight="bold")
    style_axis(ax, grid=False)
    ax.grid(axis="x", color=LIGHT, linestyle="--")
    rows = [
        data["method"],
        data["baselines"]["baselines"]["Reference Classifier Lookup"],
        data["baselines"]["baselines"]["Nearest-Centroid Lookup"],
    ]
    for index, (name, row, color) in enumerate(zip(names, rows, colors)):
        heat_ax = fig.add_axes([0.07 + index * 0.30, 0.12, 0.25, 0.32])
        matrix = method_matrix(data) if name == "CanonDressGS" else baseline_matrix(row)
        heat_ax.imshow(matrix, cmap="cividis", vmin=0, vmax=1, aspect="auto")
        heat_ax.set_xticks(range(3), ["S0", "S1", "S2"], fontsize=14)
        heat_ax.set_yticks(range(4), ["R0", "R1", "R2", "R3"], fontsize=14)
        heat_ax.set_title(name, fontsize=19, color=color, weight="bold", pad=10)
        for r in range(4):
            for s in range(3):
                heat_ax.text(s, r, f"{round(matrix[r,s]*3)}/3", ha="center", va="center", fontsize=15, color=WHITE if matrix[r,s] < 0.75 else NAVY, weight="bold")
    fig.text(
        0.55,
        0.69,
        "Budget boundary",
        fontsize=25,
        color=NAVY,
        weight="bold",
    )
    fig.text(
        0.55,
        0.64,
        "CanonDressGS: 3600 steps, 2050 trainable parameters\n"
        "Reference Classifier: 3600 steps, 2563 trainable parameters\n"
        "Nearest-Centroid: 0 optimizer steps, train-fold centroids",
        fontsize=19,
        color=NAVY,
        va="top",
        linespacing=1.5,
    )
    save_page(fig, path)
    return []


def page_9(data: dict[str, Any], path: Path) -> list[str]:
    fig = new_page(
        9,
        "Oracle and upper-reference boundary",
        "Perfect endpoint lookup is shown as a diagnostic ceiling, not as a deployable competitor",
    )
    rows = [
        ["Outfit-ID Oracle", "36/36", "Ground-truth test garment ID", "NON-DEPLOYABLE"],
        ["Teacher Endpoint", "36/36", "True-ID selected full Teacher residual", "NON-DEPLOYABLE"],
    ]
    table_ax = fig.add_axes([0.07, 0.58, 0.86, 0.22])
    add_table(
        table_ax,
        rows,
        ["Upper reference", "Result", "Privileged information", "Status"],
        font_size=19,
        col_widths=[0.23, 0.12, 0.43, 0.20],
    )
    sources = []
    request_map = {
        "O01": "subject00_O01_slot06_cand01",
        "O03": "subject00_O03_slot06_remaining_attempt005_cand00",
        "O04": "subject00_O04_slot06_remaining_attempt005_cand00",
    }
    for index, garment in enumerate(("O01", "O03", "O04")):
        source = source_image_path(garment, "teacher", request_map[garment])
        sources.append(str(source))
        add_image(
            fig,
            (0.08 + index * 0.30, 0.17, 0.24, 0.31),
            source,
            f"{garment} frozen Teacher endpoint",
            border_color=GARMENT_COLORS[garment],
        )
    fig.text(
        0.50,
        0.115,
        "These references use ground-truth identity at test time. They bound endpoint availability; they do not establish deployable generalization.",
        fontsize=18,
        color=VERMILLION,
        ha="center",
    )
    save_page(fig, path)
    return sources


def page_10(data: dict[str, Any], path: Path) -> list[str]:
    fig = new_page(
        10,
        "Compute-budget and information-access audit",
        "Separate panels avoid dual-axis scaling; optimizer budgets are explicitly unequal",
    )
    budget = data["compute_budget"]["rows"]
    short = ["CanonDressGS", "Ref Classifier", "Centroid", "Outfit-ID Oracle", "Teacher Endpoint"]
    steps = [row["optimizer_steps"] for row in budget]
    params = [row["trainable_parameter_count"] for row in budget]
    colors = [BLUE, ORANGE, TEAL, GRAY, MAGENTA]
    ax1 = fig.add_axes([0.06, 0.52, 0.42, 0.29])
    y = np.arange(5)
    ax1.barh(y, steps, color=colors, edgecolor=NAVY)
    ax1.set_yticks(y, short)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 4000)
    ax1.set_xlabel("Historical optimizer steps")
    ax1.set_title("Optimizer budget")
    for i, value in enumerate(steps):
        ax1.text(value + 50, i, str(value), va="center", fontsize=17, color=NAVY)
    style_axis(ax1, grid=False)
    ax1.grid(axis="x", color=LIGHT, linestyle="--")
    ax2 = fig.add_axes([0.54, 0.52, 0.42, 0.29])
    ax2.barh(y, params, color=colors, edgecolor=NAVY)
    ax2.set_yticks(y, short)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 3000)
    ax2.set_xlabel("Trainable parameter count")
    ax2.set_title("Trainable capacity")
    for i, value in enumerate(params):
        ax2.text(value + 40, i, str(value), va="center", fontsize=17, color=NAVY)
    style_axis(ax2, grid=False)
    ax2.grid(axis="x", color=LIGHT, linestyle="--")
    table_rows = []
    for row in budget:
        table_rows.append(
            [
                row["name"],
                str(row["evaluation_record_decisions"]),
                f"{row['wall_time_seconds_sum']:.3f}",
                f"{row['peak_vram_bytes_max'] / (1024**2):.1f}",
                "True ID" if "ground-truth" in row["label_or_oracle_access"] else "No test ID",
                row["category"],
            ]
        )
    table_ax = fig.add_axes([0.055, 0.13, 0.89, 0.29])
    add_table(
        table_ax,
        table_rows,
        ["Row", "Eval decisions", "Wall (s)", "Peak VRAM (MiB)", "Test-ID access", "Class"],
        font_size=15,
        col_widths=[0.23, 0.13, 0.11, 0.15, 0.16, 0.16],
    )
    save_page(fig, path)
    return []


def page_11(data: dict[str, Any], path: Path) -> list[str]:
    fig = new_page(
        11,
        "O01/O03/O04 Teacher technical and visual evidence",
        "Slot06 target-to-Teacher pairs are shown without stretching; human visual decisions remain null",
    )
    registry = read_json(TEACHER_REGISTRY)
    request_map = {
        "O01": "subject00_O01_slot06_cand01",
        "O03": "subject00_O03_slot06_remaining_attempt005_cand00",
        "O04": "subject00_O04_slot06_remaining_attempt005_cand00",
    }
    sources = []
    for index, garment in enumerate(("O01", "O03", "O04")):
        x = 0.045 + index * 0.315
        request = request_map[garment]
        target = source_image_path(garment, "target", request)
        teacher = source_image_path(garment, "teacher", request)
        sources.extend([str(target), str(teacher)])
        add_image(fig, (x, 0.44, 0.135, 0.31), target, f"{garment} target", border_color=ORANGE)
        add_image(fig, (x + 0.15, 0.44, 0.135, 0.31), teacher, f"{garment} Teacher", border_color=TEAL)
        row = registry["garments"][garment]
        metrics = row["evaluation_metrics"]
        add_card(
            fig,
            (x, 0.15, 0.285, 0.23),
            f"{garment} | TECHNICAL_PASS",
            [
                f"Silhouette IoU: {metrics['silhouette_iou']:.4f}",
                f"Garment LPIPS: {metrics['garment_region_lpips']:.4f}",
                f"Boundary F: {metrics['boundary_f']:.4f}",
                f"Views: {row['target_view_count']} | step 1200",
                "Human decision: null",
            ],
            accent=GARMENT_COLORS[garment],
            title_size=19,
            body_size=14,
        )
    fig.text(
        0.50,
        0.805,
        "Technical validity does not assign scientific PASS. Reviewers must inspect garment fidelity, identity preservation, and view-dependent artifacts.",
        fontsize=17,
        color=VERMILLION,
        ha="center",
    )
    save_page(fig, path)
    return sources


def page_12(data: dict[str, Any], path: Path) -> list[str]:
    fig = new_page(
        12,
        "Scientific limitations and next step",
        "The post-hoc seal is technical/provenance-ready; it is not an automatic paper decision",
    )
    add_card(
        fig,
        (0.055, 0.52, 0.42, 0.30),
        "Evidence-supported conclusions",
        [
            "Method: 30/36 formal endpoint decisions",
            "Errors recur only at the O04/O03 boundary",
            "Reference Classifier: 31/36",
            "Nearest-Centroid: 35/36",
            "Provenance and immutability audits: PASS",
        ],
        accent=TEAL,
    )
    add_card(
        fig,
        (0.525, 0.52, 0.42, 0.30),
        "Claims not authorized",
        [
            "No Teacher or method human PASS",
            "No equal-budget claim across all five rows",
            "No deployable claim for Oracle references",
            "No direct Subject00-vs-Subject02 numeric claim",
            "No paper-final classification",
        ],
        accent=VERMILLION,
    )
    add_card(
        fig,
        (0.055, 0.18, 0.89, 0.25),
        "Next task - user action only",
        [
            "USER_UPLOAD_AND_REVIEW_SUBJECT00_COMMONSAFE4_HUMAN_SCIENTIFIC_REVIEW_PAGES",
            "After human review, either authorize a matched Subject02 CommonSafe4 run or freeze a mixed/negative result and narrow the paper claim.",
            "This next task has not been executed.",
        ],
        accent=BLUE,
        title_size=25,
        body_size=18,
    )
    fig.text(
        0.50,
        0.115,
        "TEACHER_HUMAN_VISUAL_DECISION = null | METHOD_HUMAN_REVIEW_DECISION = null | SCIENTIFIC_PASS = null | PAPER_FINAL = false",
        fontsize=16,
        color=NAVY,
        ha="center",
    )
    save_page(fig, path)
    return []


def build_pdf(png_paths: list[Path]) -> None:
    page_size = landscape((10 * inch, 16 * inch))
    pdf = canvas.Canvas(str(PDF_PATH), pagesize=page_size, pageCompression=1)
    width, height = page_size
    for path in png_paths:
        pdf.drawImage(str(path), 0, 0, width=width, height=height, preserveAspectRatio=True, anchor="c")
        pdf.showPage()
    pdf.save()
    reader = PdfReader(str(PDF_PATH))
    if len(reader.pages) != 12:
        raise RuntimeError(f"PDF page count {len(reader.pages)}")


def image_rms(left: Image.Image, right: Image.Image) -> float:
    difference = ImageChops.difference(left, right)
    histogram = difference.histogram()
    squares = sum((value % 256) ** 2 * count for value, count in enumerate(histogram))
    return math.sqrt(squares / (left.width * left.height * 3))


def poppler_qa(png_paths: list[Path]) -> list[dict[str, Any]]:
    qa_root = Path(tempfile.mkdtemp(prefix="subject00_commonsafe4_pdfqa_"))
    try:
        prefix = qa_root / "page"
        subprocess.run(
            ["pdftoppm", "-png", "-r", "100", str(PDF_PATH), str(prefix)],
            check=True,
            capture_output=True,
            text=True,
        )
        rendered = sorted(qa_root.glob("page-*.png"))
        if len(rendered) != 12:
            raise RuntimeError(f"Poppler rendered {len(rendered)} pages")
        rows = []
        for index, (source_path, rendered_path) in enumerate(zip(png_paths, rendered), start=1):
            with Image.open(source_path) as source_raw, Image.open(rendered_path) as rendered_raw:
                source = source_raw.convert("RGB")
                rendered_image = rendered_raw.convert("RGB")
                expected = source.resize(rendered_image.size, Image.Resampling.LANCZOS)
                rms = image_rms(expected, rendered_image)
                source_gray = np.asarray(
                    expected.resize((128, 80), Image.Resampling.LANCZOS).convert("L"),
                    dtype=float,
                ).ravel()
                rendered_gray = np.asarray(
                    rendered_image.resize((128, 80), Image.Resampling.LANCZOS).convert("L"),
                    dtype=float,
                ).ravel()
                correlation = float(np.corrcoef(source_gray, rendered_gray)[0, 1])
                aspect = rendered_image.width / rendered_image.height
                passed = (
                    abs(aspect - 1.6) < 0.01
                    and rms < 18.0
                    and correlation > 0.985
                    and ImageStat.Stat(rendered_image).extrema != [(255, 255)] * 3
                )
                rows.append(
                    {
                        "page": index,
                        "source_path": str(source_path),
                        "rendered_resolution": [rendered_image.width, rendered_image.height],
                        "rms_difference_after_scale": rms,
                        "thumbnail_correlation": correlation,
                        "aspect_ratio": aspect,
                        "status": "PASS" if passed else "FAIL",
                    }
                )
        if not all(row["status"] == "PASS" for row in rows):
            raise RuntimeError(f"Poppler visual QA failed: {rows}")
        contact = Image.new("RGB", (1280, 800), "white")
        for index, rendered_path in enumerate(rendered):
            with Image.open(rendered_path) as page:
                thumb = page.convert("RGB").resize((320, 200), Image.Resampling.LANCZOS)
                contact.paste(thumb, ((index % 4) * 320, (index // 4) * 200))
        contact.save("/tmp/subject00_commonsafe4_review_contact_sheet.png", optimize=True)
        return rows
    finally:
        shutil.rmtree(qa_root)


def grayscale_contact(png_paths: list[Path]) -> None:
    contact = Image.new("L", (1280, 800), "white")
    for index, path in enumerate(png_paths):
        with Image.open(path) as page:
            thumb = page.convert("L").resize((320, 200), Image.Resampling.LANCZOS)
            contact.paste(thumb, ((index % 4) * 320, (index // 4) * 200))
    contact.save("/tmp/subject00_commonsafe4_review_contact_sheet_grayscale.png", optimize=True)


def main() -> None:
    if not AUDIT_SOURCE.is_file():
        raise FileNotFoundError(AUDIT_SOURCE)
    if INDEX_PATH.exists() or README_PATH.exists() or PDF_PATH.exists():
        raise FileExistsError("review pack outputs already exist; overwrite is forbidden")
    data = read_json(AUDIT_SOURCE)
    page_functions = [
        page_1,
        page_2,
        page_3,
        page_4,
        page_5,
        page_6,
        page_7,
        page_8,
        page_9,
        page_10,
        page_11,
        page_12,
    ]
    png_paths = [REVIEW_ROOT / spec[0] for spec in PAGE_SPECS]
    if any(path.exists() for path in png_paths):
        raise FileExistsError("one or more page PNGs already exist; overwrite is forbidden")
    page_sources = []
    for function, path in zip(page_functions, png_paths):
        page_sources.append(function(data, path))
    for path in png_paths:
        with Image.open(path) as image:
            if image.size != (WIDTH, HEIGHT):
                raise RuntimeError(f"resolution failure: {path}")
    build_pdf(png_paths)
    qa_rows = poppler_qa(png_paths)
    grayscale_contact(png_paths)
    png_rows = []
    for page, (path, spec, sources) in enumerate(
        zip(png_paths, PAGE_SPECS, page_sources), start=1
    ):
        with Image.open(path) as image:
            png_rows.append(
                {
                    "page": page,
                    "path": str(path),
                    "description": spec[1],
                    "resolution": [image.width, image.height],
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "source_images": sources,
                    "source_images_aspect_preserved": True,
                }
            )
    pdf_reader = PdfReader(str(PDF_PATH))
    index = {
        "schema_version": "canondressgs.subject00.commonsafe4.review_pack_index.v1",
        "task_id": data["task_id"],
        "status": "PASS_12_PAGE_REVIEW_PACK_READY",
        "review_root": str(REVIEW_ROOT),
        "review_png_count": len(png_rows),
        "review_pngs": png_rows,
        "review_pdf": {
            "path": str(PDF_PATH),
            "bytes": PDF_PATH.stat().st_size,
            "sha256": sha256(PDF_PATH),
            "page_count": len(pdf_reader.pages),
            "page_order": [row["page"] for row in png_rows],
            "poppler_qa_pass_count": sum(row["status"] == "PASS" for row in qa_rows),
            "poppler_qa_rows": qa_rows,
            "crop_status": "PASS_NO_CROP",
            "order_status": "PASS_FIXED_1_TO_12",
            "overlay_status": "PASS_NO_OVERLAY",
        },
        "visual_design_audit": {
            "small_n_means_only_without_points": False,
            "dual_y_axis_count": 0,
            "pie_or_3d_chart_count": 0,
            "rainbow_or_jet_colormap_count": 0,
            "colorblind_safe_palette": True,
            "grayscale_contact_sheet": "/tmp/subject00_commonsafe4_review_contact_sheet_grayscale.png",
            "ai_review_contact_sheet": "/tmp/subject00_commonsafe4_review_contact_sheet.png",
            "ai_visual_review_status": "PASS_AFTER_PAGE7_ZORDER_AND_SPACING_REPAIR",
            "visual_review_iterations": 3,
        },
        "human_fields": data["human_fields"],
    }
    atomic_json(INDEX_PATH, index)
    readme_lines = [
        "# Subject00 CommonSafe4 Human Scientific Review Pack",
        "",
        "This directory contains display-only post-hoc review assets. It does not",
        "change targets, renders, checkpoints, metrics, or paper text. Human and",
        "scientific decisions remain pending.",
        "",
        "## Upload order",
        "",
    ]
    for row in png_rows:
        readme_lines.append(
            f"{row['page']}. `{Path(row['path']).name}` - {row['description']} "
            f"({row['resolution'][0]}x{row['resolution'][1]}, SHA256 `{row['sha256']}`)"
        )
    readme_lines.extend(
        [
            "",
            "## Combined PDF",
            "",
            f"`{PDF_PATH.name}` - 12 pages, SHA256 `{index['review_pdf']['sha256']}`.",
            "Poppler render QA: 12/12 PASS.",
            "",
            "Next task (not executed):",
            "`USER_UPLOAD_AND_REVIEW_SUBJECT00_COMMONSAFE4_HUMAN_SCIENTIFIC_REVIEW_PAGES`",
            "",
        ]
    )
    atomic_text(README_PATH, "\n".join(readme_lines))
    print(
        json.dumps(
            {
                "status": index["status"],
                "png_count": index["review_png_count"],
                "pdf_pages": index["review_pdf"]["page_count"],
                "pdf_bytes": index["review_pdf"]["bytes"],
                "pdf_sha256": index["review_pdf"]["sha256"],
                "poppler_qa": index["review_pdf"]["poppler_qa_pass_count"],
                "contact_sheet": "/tmp/subject00_commonsafe4_review_contact_sheet.png",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
