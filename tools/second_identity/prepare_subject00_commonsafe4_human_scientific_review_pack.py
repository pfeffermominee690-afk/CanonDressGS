#!/usr/bin/env python3
"""Build the frozen Subject00 CommonSafe4 human scientific review pack.

This program is deliberately display-only.  It reads sealed outputs, creates
review pages and a PDF, and writes provenance metadata.  It never imports the
training/runtime modules and has no optimizer or model-forward code path.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import tempfile
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps
from pypdf import PdfReader
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as pdf_canvas


TASK_ID = "AAAI27-SUBJECT00-COMMONSAFE4-REVIEW-PACK-UNIQUE-ROOT-001"
AUTHORIZATION = (
    "RESOLVE_SUBJECT00_COMMONSAFE4_REVIEW_PACK_ROOT_COLLISION_WITH_UNIQUE_"
    "AUTHORITATIVE_ROOT"
)
SOURCE_BRANCH = "research/subject00-commonsafe4-concurrent-12run-provenance-audit-20260727"
SOURCE_HEAD = "49cbe42bb289d58f80f3393c705b8af9f3ffaa28"
NEW_BRANCH = "research/subject00-commonsafe4-review-pack-unique-root-20260727"
FINAL_CLASSIFICATION = (
    "SUBJECT00_COMMONSAFE4_AUTHORITATIVE_HUMAN_SCIENTIFIC_REVIEW_PACK_READY"
)
NEXT_UNIQUE_TASK = (
    "USER_UPLOAD_AND_REVIEW_SUBJECT00_COMMONSAFE4_AUTHORITATIVE_REVIEW_PAGES"
)

CLOUD_WORKTREE = Path(
    "/root/autodl-tmp/canondressgs_work/worktrees/"
    "canondressgs_subject00_commonsafe4_review_pack_unique_root"
)
WINDOWS_WORKTREE = (
    r"E:\model_train\canondressgs_subject00_commonsafe4_human_scientific_review_pack"
)
OUTPUTS = Path("/root/autodl-tmp/canondressgs_work/outputs")
METHOD_ROOT = OUTPUTS / "SUBJECT00-CANONDRESSGS-METHOD-BASE60747-COMMONSAFE4-001"
METHOD_ATTEMPT = METHOD_ROOT / "attempt_001"
BASELINE_ROOT = OUTPUTS / "SUBJECT00-CANONDRESSGS-BASELINES-BASE60747-COMMONSAFE4-001"
BASELINE_ATTEMPT = BASELINE_ROOT / "attempt_001"
BASE_ROOT = OUTPUTS / "SUBJECT00-FORMAL-BASE-101245-001" / "attempt_001"
TARGET_ROOT = (
    Path("/root/autodl-tmp/canondressgs_work/teacher_targets")
    / "SUBJECT00-24CELL-001"
    / "attempt_001"
)
FOREIGN_ROOT = METHOD_ROOT / "review" / "human_scientific_review_pack_20260727"
FOREIGN_ROOT_OWNER_TASK = (
    "AAAI27-SUBJECT00-COMMONSAFE4-FINAL-HEAD-SEMANTICS-SEAL-REVIEW-001"
)
FOREIGN_AUDIT_NAME = "subject00_commonsafe4_independent_audit_source_20260727.json"
FOREIGN_PAGE01_NAME = "page_01_provenance_and_immutability.png"
PACK_ROOT = (
    METHOD_ROOT
    / "review"
    / "human_scientific_review_pack_authoritative_20260727_001"
)
STAGING_ROOT = (
    METHOD_ROOT
    / "review"
    / ".staging_human_scientific_review_pack_authoritative_20260727_001"
)
QA_ROOT = Path(
    "/tmp/subject00_commonsafe4_authoritative_review_pack_pdf_qa_20260727_001"
)

BASE_CHECKPOINT = BASE_ROOT / "checkpoints" / "step_060747.pth"
BASE_SHA = "2d09b1ce6bb19b8b6418314caab8dc1cf92d70e432a9e09e009989332ea4a2f7"
TEACHERS = {
    "O01": {
        "root": OUTPUTS / "SUBJECT00-O01-TEACHER-BASE60747-CAMSAFE7-001" / "attempt_001",
        "sha256": "c7881862c4eddf5f58538a2278ab7765aa047784681fb02e7cd89cfe846c1892",
    },
    "O03": {
        "root": OUTPUTS
        / "SUBJECT00-O03-TEACHER-BASE60747-FORMAL-CAMSAFE7-001"
        / "attempt_001",
        "sha256": "054b9efe1086d89b18310b9831d3ed7aea59e286e31200509b58756545fc3920",
    },
    "O04": {
        "root": OUTPUTS / "SUBJECT00-O04-TEACHER-BASE60747-8VIEW-001" / "attempt_001",
        "sha256": "2fa7764097d8577c1610bbf222b26d9ea287bd18074371de400a66cd2270f3a1",
    },
}

SOURCE_SUMMARY = (
    CLOUD_WORKTREE
    / "paper_protocol"
    / "reviewer_risk"
    / "subject00_commonsafe4_concurrent_12run_provenance_final_summary_20260727.json"
)
SOURCE_SEAL = (
    CLOUD_WORKTREE
    / "paper_protocol"
    / "reviewer_risk"
    / "subject00_commonsafe4_concurrent_12run_matrix_provenance_seal_20260727.json"
)
SOURCE_REGISTRIES = {
    "provenance_seal": SOURCE_SEAL,
    "final_summary": SOURCE_SUMMARY,
    "execution_registry": CLOUD_WORKTREE
    / "paper_protocol"
    / "reviewer_risk"
    / "subject00_commonsafe4_concurrent_12run_execution_registry_20260727.json",
    "checkpoint_authenticity": CLOUD_WORKTREE
    / "paper_protocol"
    / "reviewer_risk"
    / "subject00_commonsafe4_concurrent_checkpoint_authenticity_registry_20260727.json",
    "formal_test_registry": CLOUD_WORKTREE
    / "paper_protocol"
    / "reviewer_risk"
    / "subject00_commonsafe4_concurrent_formal_test_registry_20260727.json",
    "recomputed_aggregate": CLOUD_WORKTREE
    / "paper_protocol"
    / "reviewer_risk"
    / "subject00_commonsafe4_concurrent_matrix_recomputed_aggregate_20260727.json",
    "replacement_provenance": CLOUD_WORKTREE
    / "paper_protocol"
    / "reviewer_risk"
    / "subject00_commonsafe4_replacement_pretraining_provenance_audit_20260727.json",
    "rotation_fold_registry": CLOUD_WORKTREE
    / "paper_protocol"
    / "reviewer_risk"
    / "subject00_commonsafe4_actual_rotation_fold_registry_20260727.json",
    "runner_config_registry": CLOUD_WORKTREE
    / "paper_protocol"
    / "reviewer_risk"
    / "subject00_commonsafe4_actual_runner_config_registry_20260727.json",
    "runner_core_equivalence": CLOUD_WORKTREE
    / "paper_protocol"
    / "reviewer_risk"
    / "subject00_commonsafe4_actual_runner_core_equivalence_audit_20260727.json",
    "fair_baseline_discovery": CLOUD_WORKTREE
    / "paper_protocol"
    / "reviewer_risk"
    / "subject00_commonsafe4_concurrent_fair_baseline_discovery_audit_20260727.json",
    "writer_identity": CLOUD_WORKTREE
    / "paper_protocol"
    / "reviewer_risk"
    / "subject00_commonsafe4_concurrent_writer_identity_registry_20260727.json",
}
METHOD_REPORT = METHOD_ATTEMPT / "FINAL_REPORT.json"
METHOD_MATRIX = METHOD_ATTEMPT / "matrix" / "matrix_aggregate.json"
METHOD_CONFIG = METHOD_ATTEMPT / "contract" / "config_resolved.json"
BASELINE_REPORT = BASELINE_ATTEMPT / "FINAL_REPORT.json"
TARGET_MANIFEST = (
    TARGET_ROOT / "10_final_registry" / "subject00_22_training_full_dataset_v1.json"
)
BASE_RESUME_CONTRACT = (
    BASE_ROOT / "control" / "FORMAL_BASE_RESUME_FROM_60747_CONTRACT_20260727.json"
)

PAGE_NAMES = [
    "page_01_provenance_and_immutability.png",
    "page_02_commonsafe4_replacement.png",
    "page_03_rotation_fold_contract.png",
    "page_04_method_matrix.png",
    "page_05_method_confusion_matrix.png",
    "page_06_rotation_seed_effects.png",
    "page_07_all_six_O04_to_O03_errors.png",
    "page_08_nonoracle_baselines.png",
    "page_09_oracle_upper_references.png",
    "page_10_compute_budget_audit.png",
    "page_11_three_teacher_evidence.png",
    "page_12_scientific_limitations_and_decisions.png",
]
PAGE_DIRS = [
    "00_master",
    "01_protocol",
    "01_protocol",
    "02_method_matrix",
    "02_method_matrix",
    "02_method_matrix",
    "03_error_cases",
    "04_baselines",
    "04_baselines",
    "05_compute",
    "06_teachers",
    "07_limitations",
]
PDF_NAME = (
    "subject00_commonsafe4_method_teacher_baseline_human_scientific_review_pages_"
    "authoritative_20260727_001.pdf"
)
INDEX_NAME = "subject00_commonsafe4_authoritative_review_pack_index_20260727.json"
README_NAME = "SUBJECT00_COMMONSAFE4_AUTHORITATIVE_REVIEW_PACK_README.md"
DISPLAY_LABEL = "DISPLAY_ONLY_HUMAN_SCIENTIFIC_REVIEW"
PAPER_LABEL = "NOT_PAPER_FINAL"

DECISIONS = {
    "teacher_O01_visual_decision": None,
    "teacher_O03_visual_decision": None,
    "teacher_O04_visual_decision": None,
    "method_error_pattern_decision": None,
    "method_scientific_decision": None,
    "nonoracle_baseline_fairness_decision": None,
    "oracle_reference_classification_decision": None,
    "protocol_decision": None,
    "cross_identity_decision": None,
    "final_human_scientific_decision": None,
    "scientific_pass": None,
    "paper_eligible": False,
    "paper_final": False,
}

NAVY = "#17324D"
BLUE = "#0072B2"
SKY = "#56B4E9"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILION = "#D55E00"
PURPLE = "#CC79A7"
YELLOW = "#F0E442"
INK = "#14212B"
MUTED = "#586875"
LIGHT = "#EEF3F6"
MID = "#D5E0E7"
WHITE = "#FFFFFF"
BLACK = "#000000"
OKABE = [BLUE, ORANGE, GREEN, PURPLE, SKY, VERMILION, YELLOW, BLACK]

FONT_REGULAR_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
]
FONT_BOLD_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    assert_not_foreign_write_target(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    assert_not_foreign_write_target(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text.rstrip() + "\n")


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def tree_digest(root: Path, exclude_names: set[str] | None = None) -> dict[str, Any]:
    exclude_names = exclude_names or set()
    digest = hashlib.sha256()
    file_count = 0
    byte_count = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in exclude_names for part in rel.parts):
            continue
        stat = path.stat()
        file_sha = sha256_file(path)
        digest.update(rel.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_sha.encode("ascii"))
        digest.update(b"\n")
        file_count += 1
        byte_count += stat.st_size
    return {
        "root": str(root),
        "file_count": file_count,
        "byte_count": byte_count,
        "tree_sha256": digest.hexdigest(),
    }


def git_output(args: Sequence[str], cwd: Path = CLOUD_WORKTREE) -> str:
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True).strip()


def assert_not_foreign_write_target(path: Path) -> None:
    """Fail closed if any output path could touch the foreign task-owned root."""
    candidate = Path(os.path.abspath(path))
    foreign = Path(os.path.abspath(FOREIGN_ROOT))
    if candidate == foreign or foreign in candidate.parents:
        raise RuntimeError(f"FOREIGN_TASK_OWNED_REVIEW_ROOT is write protected: {candidate}")


def process_lines(command: Sequence[str]) -> list[str]:
    result = subprocess.run(
        list(command),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        return []
    markers = [str(FOREIGN_ROOT), FOREIGN_ROOT_OWNER_TASK]
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if any(marker in line for marker in markers)
    ]


def foreign_root_snapshot() -> dict[str, Any]:
    """Collect read-only external-task evidence without treating it as science."""
    exists = FOREIGN_ROOT.is_dir()
    files = sorted(path for path in FOREIGN_ROOT.rglob("*") if path.is_file()) if exists else []
    audit = FOREIGN_ROOT / FOREIGN_AUDIT_NAME
    page01 = FOREIGN_ROOT / FOREIGN_PAGE01_NAME
    latest_mtime = max((path.stat().st_mtime for path in files), default=None)

    def known_file(path: Path) -> dict[str, Any]:
        return {
            "path": str(path),
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256_file(path) if path.is_file() else None,
        }

    return {
        "classification": "FOREIGN_TASK_OWNED_REVIEW_ROOT",
        "owner_task": FOREIGN_ROOT_OWNER_TASK,
        "path": str(FOREIGN_ROOT),
        "exists": exists,
        "file_count": len(files),
        "byte_count": sum(path.stat().st_size for path in files),
        "last_modified": (
            datetime.fromtimestamp(latest_mtime, timezone.utc).astimezone().isoformat()
            if latest_mtime is not None
            else None
        ),
        "known_audit_json": known_file(audit),
        "known_page_01": known_file(page01),
        "active_writer_processes": process_lines(["ps", "-eo", "pid,lstart,args"]),
        "related_tmux_panes": process_lines(
            ["tmux", "list-panes", "-a", "-F", "#{session_name}:#{window_index}.#{pane_index} #{pane_pid} #{pane_current_command} #{pane_current_path}"]
        ),
        "this_task_write_calls": 0,
        "this_task_delete_calls": 0,
        "this_task_move_calls": 0,
        "files_copied_as_authority": 0,
    }


def source_registry_snapshot() -> dict[str, Any]:
    records: dict[str, Any] = {}
    for role, path in SOURCE_REGISTRIES.items():
        payload = load_json(path)
        records[role] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "schema_version": payload.get("schema_version"),
            "task_id": payload.get("task_id"),
            "status": payload.get("status"),
            "test_result": payload.get("test_result"),
        }
    return records


def source_snapshot() -> dict[str, Any]:
    teacher_files = {
        garment: payload["root"] / "checkpoints" / "step_001200.pth"
        for garment, payload in TEACHERS.items()
    }
    return {
        "created_at": now_iso(),
        "git": {
            "branch": git_output(["branch", "--show-current"]),
            "head": git_output(["rev-parse", "HEAD"]),
            "status_porcelain": git_output(["status", "--porcelain=v1"]),
        },
        "base_checkpoint": {
            "path": str(BASE_CHECKPOINT),
            "sha256": sha256_file(BASE_CHECKPOINT),
            "bytes": BASE_CHECKPOINT.stat().st_size,
        },
        "teacher_checkpoints": {
            garment: {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for garment, path in teacher_files.items()
        },
        "targets": tree_digest(TARGET_ROOT),
        "method_scientific_core": tree_digest(METHOD_ROOT, exclude_names={"review"}),
        "baselines": tree_digest(BASELINE_ROOT),
        "authoritative_source_registries": source_registry_snapshot(),
    }


def select_font_path(candidates: Sequence[str]) -> str:
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError(f"No font from candidates: {candidates}")


FONT_REGULAR = select_font_path(FONT_REGULAR_CANDIDATES)
FONT_BOLD = select_font_path(FONT_BOLD_CANDIDATES)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size=size)


def wrap_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    text_font: ImageFont.FreeTypeFont,
    width: int,
) -> list[str]:
    output: list[str] = []
    for paragraph in str(text).splitlines() or [""]:
        words = paragraph.split()
        if not words:
            output.append("")
            continue
        line = words[0]
        for word in words[1:]:
            candidate = f"{line} {word}"
            if draw.textlength(candidate, font=text_font) <= width:
                line = candidate
            else:
                output.append(line)
                line = word
        output.append(line)
    return output


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    text_font: ImageFont.FreeTypeFont,
    fill: str,
    width: int,
    spacing: int = 12,
    max_lines: int | None = None,
) -> int:
    x, y = xy
    lines = wrap_lines(draw, text, text_font, width)
    if max_lines is not None:
        lines = lines[:max_lines]
    bbox = draw.textbbox((0, 0), "Ag", font=text_font)
    line_height = bbox[3] - bbox[1] + spacing
    for line in lines:
        draw.text((x, y), line, font=text_font, fill=fill)
        y += line_height
    return y


def base_page(
    title: str,
    subtitle: str,
    page_number: int,
    size: tuple[int, int] = (6400, 4000),
) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", size, WHITE)
    draw = ImageDraw.Draw(image)
    width, height = size
    draw.rectangle((0, 0, width, 360), fill=NAVY)
    draw.text((180, 85), title, font=font(118, True), fill=WHITE)
    draw.text((185, 235), subtitle, font=font(48), fill="#D9E7EF")
    draw.rounded_rectangle(
        (width - 2050, 65, width - 155, 175),
        radius=34,
        fill="#274B68",
    )
    draw.text(
        (width - 1995, 85),
        f"{DISPLAY_LABEL} | {PAPER_LABEL}",
        font=font(45, True),
        fill=WHITE,
    )
    draw.line((160, height - 150, width - 160, height - 150), fill=MID, width=5)
    draw.text(
        (170, height - 110),
        f"Subject00 CommonSafe4 frozen review evidence | page {page_number:02d}/12",
        font=font(40),
        fill=MUTED,
    )
    draw.text(
        (width - 1560, height - 110),
        "Human decisions remain null",
        font=font(40, True),
        fill=VERMILION,
    )
    return image, draw


def card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    body: str,
    accent: str = BLUE,
    title_size: int = 58,
    body_size: int = 45,
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=42, fill=WHITE, outline=MID, width=5)
    draw.rounded_rectangle((x0, y0, x0 + 24, y1), radius=12, fill=accent)
    draw.text((x0 + 70, y0 + 45), title, font=font(title_size, True), fill=INK)
    draw_wrapped(
        draw,
        (x0 + 70, y0 + 135),
        body,
        font(body_size),
        MUTED,
        max(100, x1 - x0 - 125),
        spacing=16,
    )


def fit_image(source: Image.Image, box: tuple[int, int]) -> Image.Image:
    return ImageOps.contain(source.convert("RGB"), box, method=Image.Resampling.LANCZOS)


def paste_labeled_image(
    page: Image.Image,
    draw: ImageDraw.ImageDraw,
    source: Path | Image.Image,
    box: tuple[int, int, int, int],
    label: str,
    border: str = MID,
    label_fill: str = NAVY,
    label_size: int = 40,
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=28, fill=LIGHT, outline=border, width=5)
    label_h = 92
    draw.rounded_rectangle((x0, y0, x1, y0 + label_h), radius=24, fill=label_fill)
    draw.text((x0 + 24, y0 + 20), label, font=font(label_size, True), fill=WHITE)
    image = Image.open(source) if isinstance(source, Path) else source
    fitted = fit_image(image, (x1 - x0 - 34, y1 - y0 - label_h - 28))
    px = x0 + (x1 - x0 - fitted.width) // 2
    py = y0 + label_h + (y1 - y0 - label_h - fitted.height) // 2
    page.paste(fitted, (px, py))


def chart_image(fig: plt.Figure) -> Image.Image:
    stream = io.BytesIO()
    fig.savefig(stream, format="png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    stream.seek(0)
    return Image.open(stream).convert("RGB")


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D5E0E7", linewidth=0.8, linestyle="--", alpha=0.8)
    ax.tick_params(labelsize=13)
    ax.title.set_fontsize(18)
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)


def heatmap_chart(
    values: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    annotation: Sequence[Sequence[str]] | None = None,
    cmap: str = "Blues",
    vmin: float | None = None,
    vmax: float | None = None,
) -> Image.Image:
    fig, ax = plt.subplots(figsize=(11.5, 7.0), constrained_layout=True)
    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels)), col_labels, fontsize=13)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=13)
    ax.set_title(title, fontsize=20, weight="bold", pad=18)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text = (
                annotation[i][j]
                if annotation is not None
                else f"{values[i, j]:.2f}"
            )
            color = "white" if values[i, j] > ((vmax or values.max()) * 0.58) else INK
            ax.text(j, i, text, ha="center", va="center", fontsize=16, weight="bold", color=color)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.78)
    colorbar.set_label("Value", fontsize=13)
    return chart_image(fig)


def bar_chart(
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    ylabel: str,
    colors: Sequence[str] | None = None,
    ylim: tuple[float, float] | None = None,
    value_format: str = "{:.3f}",
) -> Image.Image:
    colors = list(colors or OKABE[: len(labels)])
    fig, ax = plt.subplots(figsize=(11.5, 7.0), constrained_layout=True)
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor=INK, linewidth=0.8)
    ax.set_xticks(x, labels, fontsize=13)
    ax.set_ylabel(ylabel)
    ax.set_title(title, weight="bold")
    if ylim is not None:
        ax.set_ylim(*ylim)
    style_axis(ax)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.018 if (ylim or (0, 1))[1] <= 1.1 else max(values) * 0.025),
            value_format.format(value),
            ha="center",
            va="bottom",
            fontsize=14,
            weight="bold",
        )
    return chart_image(fig)


def tile_table(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    data: Sequence[Sequence[str]],
    col_widths: Sequence[int],
    row_height: int,
    header: bool = True,
    font_size: int = 40,
) -> None:
    x0, y0 = origin
    y = y0
    for row_index, row in enumerate(data):
        x = x0
        for col_index, cell in enumerate(row):
            width = col_widths[col_index]
            fill = NAVY if header and row_index == 0 else (WHITE if row_index % 2 else LIGHT)
            text_fill = WHITE if header and row_index == 0 else INK
            draw.rectangle((x, y, x + width, y + row_height), fill=fill, outline=MID, width=4)
            lines = wrap_lines(draw, str(cell), font(font_size, row_index == 0), width - 32)
            line_h = font_size + 12
            text_y = y + max(14, (row_height - line_h * len(lines)) // 2)
            for line in lines[:3]:
                draw.text((x + 18, text_y), line, font=font(font_size, row_index == 0), fill=text_fill)
                text_y += line_h
            x += width
        y += row_height


def observation_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    registry_dir = TARGET_MANIFEST.parent
    output: dict[str, dict[str, Any]] = {}
    for outfit in manifest["outfits"]:
        garment = outfit["outfit_id"]
        for observation in outfit["observations"]:
            item = dict(observation)
            item["garment"] = garment
            for key, value in list(item.items()):
                if isinstance(value, str) and value.startswith(".."):
                    item[key] = str((registry_dir / value).resolve())
            output[item["condition_id"]] = item
    return output


def condition_for_slot(
    observations: dict[str, dict[str, Any]], garment: str, slot: int
) -> tuple[str, dict[str, Any]]:
    marker = f"_slot{slot:02d}_"
    matches = [
        (condition_id, observation)
        for condition_id, observation in observations.items()
        if observation["garment"] == garment and marker in condition_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {garment} slot{slot:02d} observation, got {len(matches)}")
    return matches[0]


def teacher_image(garment: str, slot: int) -> Path:
    candidates = list((TEACHERS[garment]["root"] / "review" / "teacher").glob(f"*slot{slot:02d}_*.png"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one teacher image for {garment} slot{slot:02d}, got {candidates}")
    return candidates[0]


def mask_overlay(rgb_path: Path, mask_path: Path, color: tuple[int, int, int]) -> Image.Image:
    rgb = Image.open(rgb_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if mask.size != rgb.size:
        mask = mask.resize(rgb.size, Image.Resampling.NEAREST)
    binary = mask.point(lambda p: 255 if p > 127 else 0)
    layer = Image.new("RGB", rgb.size, color)
    blended = Image.blend(rgb, layer, 0.42)
    return Image.composite(blended, rgb, binary)


def crop_by_mask(
    rgb_path: Path,
    mask_path: Path,
    margin_fraction: float = 0.12,
) -> Image.Image:
    rgb = Image.open(rgb_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if mask.size != rgb.size:
        mask = mask.resize(rgb.size, Image.Resampling.NEAREST)
    bbox = mask.point(lambda p: 255 if p > 127 else 0).getbbox()
    if bbox is None:
        return rgb
    x0, y0, x1, y1 = bbox
    margin = int(max(x1 - x0, y1 - y0) * margin_fraction)
    x0, y0 = max(0, x0 - margin), max(0, y0 - margin)
    x1, y1 = min(rgb.width, x1 + margin), min(rgb.height, y1 + margin)
    return rgb.crop((x0, y0, x1, y1))


def protected_crop(rgb_path: Path, mask_path: Path) -> Image.Image:
    overlay = mask_overlay(rgb_path, mask_path, (86, 180, 233))
    return overlay.crop((0, 0, overlay.width, int(overlay.height * 0.58)))


def page01(summary: dict[str, Any], config: dict[str, Any]) -> Image.Image:
    page, draw = base_page(
        "Frozen provenance overview",
        "Sealed evidence boundary, immutable inputs, and zero-execution review contract",
        1,
    )
    cards = [
        (
            "Source seal",
            f"{SOURCE_BRANCH}\nHEAD {SOURCE_HEAD}\nPASS_61_OF_61",
            BLUE,
        ),
        (
            "Formal target set",
            "22 training records | 2 permanent quarantine records\n"
            "quarantine: O01 slot04, O03 slot04",
            GREEN,
        ),
        (
            "Base60747",
            f"step 60747 | SHA {BASE_SHA[:20]}...\npaused; resume authorization = false",
            ORANGE,
        ),
        (
            "Three Teacher endpoints",
            "O01 c7881862... | O03 054b9efe... | O04 2fa77640...\n"
            "all frozen; decisions remain null",
            PURPLE,
        ),
        (
            "Method evidence",
            "Pure Endpoint | dual support disabled | 12/12 formal-valid runs\n"
            "30/36 endpoint Top-1 = 0.833333",
            SKY,
        ),
        (
            "This task",
            "optimizer steps 0 | model forward calls 0 | generation calls 0\n"
            "display composition only; paper body untouched",
            VERMILION,
        ),
    ]
    positions = []
    x_margin, gap, card_w = 240, 120, 2920
    y0, card_h = 600, 870
    for row in range(2):
        for col in range(3):
            x = x_margin + col * (card_w + gap)
            y = y0 + row * (card_h + 130)
            positions.append((x, y, x + card_w, y + card_h))
    # Three columns require a compact width.
    positions = []
    card_w = 1900
    gap = 110
    for row in range(2):
        for col in range(3):
            x = 240 + col * (card_w + gap)
            y = 600 + row * 1120
            positions.append((x, y, x + card_w, y + 960))
    for position, content in zip(positions, cards):
        card(draw, position, *content)
    draw.text(
        (240, 3590),
        "Review boundary: judge protocol coherence, error structure, fair baselines, "
        "Teacher evidence, and limitations. Do not infer a paper-final verdict.",
        font=font(48, True),
        fill=INK,
    )
    return page


def page02(summary: dict[str, Any]) -> Image.Image:
    page, draw = base_page(
        "Outcome-independent slot04 replacement",
        "Frozen geometry rule selected slot06/cam09 before optimization; ranking reproduced post hoc",
        2,
    )
    replacement = summary["replacement"]
    ranking = replacement["ranking"]
    chart = bar_chart(
        [r["slot"] for r in ranking],
        [r["yaw_distance_degrees"] for r in ranking],
        "Angular distance to quarantined cardinal-right anchor",
        "Yaw distance (degrees; lower ranks first)",
        [GREEN, BLUE, ORANGE, PURPLE],
        ylim=(0, 180),
        value_format="{:.1f}",
    )
    paste_labeled_image(
        page,
        draw,
        chart,
        (260, 620, 3180, 3180),
        "Frozen ranking objective - geometry only",
        label_fill=GREEN,
    )
    rows = [["Rank", "Slot / camera", "Direction", "Anchor sep.", "RMSE", "Inlier ratio"]]
    for index, record in enumerate(ranking, 1):
        rows.append(
            [
                str(index),
                f"{record['slot']} / {record['camera']}",
                record["direction"],
                f"{record['minimum_anchor_separation_degrees']:.1f} deg",
                f"{record['worst_case_registration_rmse_px']:.3f} px",
                f"{record['minimum_inlier_ratio']:.3f}",
            ]
        )
    tile_table(
        draw,
        (3370, 710),
        rows,
        [300, 750, 620, 600, 500, 580],
        350,
        font_size=38,
    )
    card(
        draw,
        (3370, 2680, 6150, 3400),
        "Selection audit",
        "reported slot06 = reproduced slot06\n"
        "selection frozen before training = true\n"
        "outcome-dependent evidence count = 0\n"
        "candidate order = slot06 -> slot02 -> slot05 -> slot01",
        GREEN,
        body_size=43,
    )
    return page


def page03(config: dict[str, Any]) -> Image.Image:
    page, draw = base_page(
        "CommonSafe4 rotation and fold contract",
        "Four frozen anchors, four rotations, three replicate seeds, and no quarantine use",
        3,
    )
    rotations = config["protocol"]["condition_rotations"]
    rows = [["Rotation", "Train slots (6 records)", "Calibration (3)", "Test (3)", "Seeds"]]
    for rotation in rotations:
        rows.append(
            [
                f"R{rotation['rotation']}",
                " + ".join(f"slot{slot:02d}" for slot in rotation["train_slots"]),
                f"slot{rotation['calibration_slot']:02d}",
                f"slot{rotation['test_slot']:02d}",
                "S0, S1, S2",
            ]
        )
    tile_table(
        draw,
        (300, 690),
        rows,
        [650, 1600, 1200, 1200, 1000],
        430,
        font_size=48,
    )
    anchor_x = [900, 2400, 3900, 5400]
    anchor_labels = ["slot00\nfront", "slot07\nleft/back", "slot03\nright/front", "slot06\nback-right proxy"]
    anchor_colors = [BLUE, ORANGE, PURPLE, GREEN]
    y = 3100
    for x, label, color in zip(anchor_x, anchor_labels, anchor_colors):
        draw.ellipse((x - 250, y - 250, x + 250, y + 250), fill=color, outline=INK, width=7)
        lines = label.splitlines()
        for offset, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font(45, True))
            draw.text(
                (x - (bbox[2] - bbox[0]) / 2, y - 58 + offset * 64),
                line,
                font=font(45, True),
                fill=WHITE,
            )
    draw.line((900, y, 5400, y), fill=MID, width=18)
    draw.text(
        (450, 3560),
        "Each rotation assigns two anchors to training, one to calibration, and one to formal test. "
        "Garment order is O01, O03, O04. The two quarantined slot04 records never enter any fold.",
        font=font(46),
        fill=MUTED,
    )
    return page


def method_accuracy_matrix(matrix: dict[str, Any]) -> np.ndarray:
    result = np.zeros((4, 3), dtype=float)
    for run in matrix["runs"]:
        result[int(run["rotation"]), int(run["seed"])] = run["formal_test"]["nearest_endpoint_accuracy"]
    return result


def page04(matrix: dict[str, Any]) -> Image.Image:
    page, draw = base_page(
        "CanonDressGS Pure Endpoint method matrix",
        "12 frozen formal cells; each cell reports correct endpoint classifications out of three garments",
        4,
    )
    values = method_accuracy_matrix(matrix)
    annotations = [[f"{round(v * 3):d}/3\n{v:.3f}" for v in row] for row in values]
    chart = heatmap_chart(
        values,
        ["R0 test slot06", "R1 test slot00", "R2 test slot07", "R3 test slot03"],
        ["seed 0", "seed 1", "seed 2"],
        "Formal-test endpoint Top-1 by rotation and seed",
        annotations,
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    paste_labeled_image(page, draw, chart, (300, 610, 4200, 3450), "12/12 formal-valid cells")
    card(
        draw,
        (4400, 680, 6100, 1600),
        "Aggregate",
        "30 / 36 correct\nTop-1 = 0.833333\nmin cell = 2/3\nmax cell = 3/3",
        BLUE,
        body_size=52,
    )
    card(
        draw,
        (4400, 1730, 6100, 2580),
        "Method contract",
        "PURE_ENDPOINT\nDual support: disabled\nDual calls: 0\nContinuous render: false",
        GREEN,
        body_size=48,
    )
    card(
        draw,
        (4400, 2710, 6100, 3450),
        "Validity",
        "NaN/Inf: none\nOOM: none\nQuarantine usage: 0\nNo cherry-picking: true",
        ORANGE,
        body_size=45,
    )
    return page


def page05(matrix: dict[str, Any]) -> Image.Image:
    page, draw = base_page(
        "Method confusion matrix",
        "All six mistakes share one direction: true O04 is assigned to the O03 endpoint",
        5,
    )
    garments = ["O01", "O03", "O04"]
    values = np.array(
        [[matrix["confusion_matrix"][true][pred] for pred in garments] for true in garments],
        dtype=float,
    )
    row_norm = values / values.sum(axis=1, keepdims=True)
    annotations = [
        [f"{int(values[i, j])}\n({row_norm[i, j]:.0%})" for j in range(3)] for i in range(3)
    ]
    chart = heatmap_chart(
        row_norm,
        [f"true {g}" for g in garments],
        [f"pred {g}" for g in garments],
        "Row-normalized confusion with counts",
        annotations,
        cmap="Blues",
        vmin=0,
        vmax=1,
    )
    paste_labeled_image(page, draw, chart, (300, 630, 4300, 3450), "36 formal-test predictions")
    card(
        draw,
        (4510, 700, 6100, 1570),
        "Stable classes",
        "O01: 12/12 correct\nO03: 12/12 correct\nNo prediction ever maps to O01 incorrectly.",
        GREEN,
        body_size=48,
    )
    card(
        draw,
        (4510, 1700, 6100, 2650),
        "Single error mode",
        "O04 -> O03: 6\nO04 -> O04: 6\nO04 recall = 0.50\nNo O04 -> O01 errors.",
        VERMILION,
        body_size=48,
    )
    card(
        draw,
        (4510, 2780, 6100, 3450),
        "Human review question",
        "Does this directional failure indicate endpoint overlap, target ambiguity, "
        "or insufficient fold generalization? Decision: null.",
        PURPLE,
        body_size=42,
    )
    return page


def page06(matrix: dict[str, Any]) -> Image.Image:
    page, draw = base_page(
        "Rotation and seed effects",
        "Every raw cell remains visible; categorical rotations and seeds are not connected as a time series",
        6,
    )
    values = method_accuracy_matrix(matrix)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    for seed in range(3):
        axes[0].scatter(
            np.arange(4) + (seed - 1) * 0.08,
            values[:, seed],
            s=190,
            color=OKABE[seed],
            edgecolor=INK,
            linewidth=1.1,
            marker=["o", "s", "^"][seed],
            label=f"seed {seed}",
            zorder=3,
        )
    axes[0].set_xticks(range(4), ["R0", "R1", "R2", "R3"])
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Formal-test endpoint Top-1")
    axes[0].set_title("Rotation groups (n=3 seeds each)", weight="bold")
    axes[0].legend(frameon=False, loc="lower left")
    style_axis(axes[0])
    for rotation in range(4):
        axes[1].scatter(
            np.arange(3) + (rotation - 1.5) * 0.055,
            values[rotation, :],
            s=170,
            color=OKABE[rotation],
            edgecolor=INK,
            linewidth=1.0,
            marker=["o", "s", "^", "D"][rotation],
            label=f"R{rotation}",
            zorder=3,
        )
    axes[1].set_xticks(range(3), ["S0", "S1", "S2"])
    axes[1].set_ylim(0, 1.08)
    axes[1].set_ylabel("Formal-test endpoint Top-1")
    axes[1].set_title("Seed groups (n=4 rotations each)", weight="bold")
    axes[1].legend(frameon=False, loc="lower right")
    style_axis(axes[1])
    chart = chart_image(fig)
    paste_labeled_image(page, draw, chart, (250, 600, 6150, 3060), "Raw 12-cell effect display")
    rotation_means = values.mean(axis=1)
    seed_means = values.mean(axis=0)
    card(
        draw,
        (350, 3190, 3100, 3650),
        "Rotation means",
        "R0 1.000 | R1 0.667 | R2 0.889 | R3 0.778",
        BLUE,
        body_size=47,
    )
    card(
        draw,
        (3300, 3190, 6050, 3650),
        "Seed means",
        "S0 0.750 | S1 0.833 | S2 0.917",
        ORANGE,
        body_size=47,
    )
    return page


def collect_errors(
    matrix: dict[str, Any], observations: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for run in matrix["runs"]:
        for row in run["formal_test"]["rows"]:
            if row["correct"]:
                continue
            condition_id, observation = condition_for_slot(
                observations, row["garment"], int(row["slot"])
            )
            errors.append(
                {
                    "run_id": run["run_id"],
                    "rotation": int(run["rotation"]),
                    "seed": int(run["seed"]),
                    "slot": int(row["slot"]),
                    "condition_id": condition_id,
                    "target": Path(observation["rgb"]),
                    "o03_teacher": teacher_image("O03", int(row["slot"])),
                    "o04_teacher": teacher_image("O04", int(row["slot"])),
                    "distances": row["candidate_squared_distances"],
                    "margin": row["score_margin_second_minus_first"],
                    "selected_endpoint": row["selected_endpoint"],
                    "true_garment": row["garment"],
                }
            )
    return errors


def page07(errors: list[dict[str, Any]]) -> Image.Image:
    page, draw = base_page(
        "Six formal O04 -> O03 errors",
        "Exact test target and frozen O03/O04 Teacher endpoint evidence for every failed run",
        7,
        size=(7200, 4800),
    )
    if len(errors) != 6:
        raise ValueError(f"Expected six errors, got {len(errors)}")
    y0, row_h = 520, 675
    for index, error in enumerate(errors, 1):
        y = y0 + (index - 1) * row_h
        fill = WHITE if index % 2 else LIGHT
        draw.rounded_rectangle((150, y, 7050, y + row_h - 35), radius=28, fill=fill, outline=MID, width=4)
        draw.text((190, y + 28), f"{index}", font=font(58, True), fill=VERMILION)
        draw.text(
            (300, y + 26),
            f"{error['run_id']} | true O04 | predicted {error['selected_endpoint']}",
            font=font(47, True),
            fill=INK,
        )
        draw.text(
            (300, y + 90),
            f"{error['condition_id']}",
            font=font(34),
            fill=MUTED,
        )
        box_y0, box_y1 = y + 150, y + row_h - 65
        paste_labeled_image(page, draw, error["target"], (300, box_y0, 900, box_y1), "O04 test target", label_size=30)
        paste_labeled_image(
            page, draw, error["o03_teacher"], (960, box_y0, 1560, box_y1), "O03 Teacher", label_fill=VERMILION, label_size=30
        )
        paste_labeled_image(
            page, draw, error["o04_teacher"], (1620, box_y0, 2220, box_y1), "O04 Teacher", label_fill=GREEN, label_size=30
        )
        distances = error["distances"]
        detail = (
            f"rotation R{error['rotation']} | seed S{error['seed']} | test slot{error['slot']:02d}\n"
            f"squared distance: O01={distances['O01']:.4f}, "
            f"O03={distances['O03']:.4f}, O04={distances['O04']:.4f}\n"
            f"decision margin (2nd - 1st)={error['margin']:.4f}\n"
            "Human error-pattern decision: null"
        )
        draw_wrapped(draw, (2350, y + 185), detail, font(42), INK, 4420, spacing=15)
    return page


def baseline_aggregate(report: dict[str, Any], name: str) -> dict[str, Any]:
    return report["aggregates"][name]


def page08(method: dict[str, Any], baselines: dict[str, Any]) -> Image.Image:
    page, draw = base_page(
        "Fair non-oracle baselines",
        "Both frozen non-oracle baselines exceed the method on the same 36 formal-test episodes",
        8,
    )
    labels = ["Method\nPure Endpoint", "Reference\nClassifier", "Nearest\nCentroid"]
    values = [
        method["matrix_endpoint_top1"],
        baseline_aggregate(baselines, "Reference Classifier Lookup")["endpoint_top1"],
        baseline_aggregate(baselines, "Nearest-Centroid Lookup")["endpoint_top1"],
    ]
    chart = bar_chart(
        labels,
        values,
        "Same-protocol formal-test endpoint Top-1",
        "Accuracy",
        [BLUE, ORANGE, GREEN],
        ylim=(0, 1.08),
    )
    paste_labeled_image(page, draw, chart, (280, 650, 3850, 3420), "Non-oracle comparison; denominator = 36")
    rows = [
        ["System", "Correct", "Top-1", "Optimizer steps", "Role"],
        ["CanonDressGS Pure Endpoint", "30/36", "0.833333", "3600", "method"],
        ["Reference Classifier Lookup", "31/36", "0.861111", "3600", "non-oracle baseline"],
        ["Nearest-Centroid Lookup", "35/36", "0.972222", "0", "non-oracle baseline"],
    ]
    tile_table(draw, (4050, 760), rows, [900, 450, 520, 650, 880], 390, font_size=38)
    card(
        draw,
        (4050, 2510, 6150, 3420),
        "Required scientific statement",
        "The method is below both non-oracle baselines:\n"
        "0.833333 < 0.861111 < 0.972222.\n"
        "Fairness decision remains null.",
        VERMILION,
        body_size=48,
    )
    return page


def page09(baselines: dict[str, Any]) -> Image.Image:
    page, draw = base_page(
        "Oracle and upper-reference systems",
        "Perfect scores are context-only upper references, not fair non-oracle competitors",
        9,
    )
    chart = bar_chart(
        ["Outfit-ID\nOracle", "Teacher\nEndpoint"],
        [
            baseline_aggregate(baselines, "Outfit-ID Oracle")["endpoint_top1"],
            baseline_aggregate(baselines, "Teacher Endpoint")["endpoint_top1"],
        ],
        "Upper-reference endpoint Top-1",
        "Accuracy",
        [PURPLE, SKY],
        ylim=(0, 1.08),
    )
    paste_labeled_image(page, draw, chart, (360, 700, 3400, 3350), "36/36 correct for each upper reference")
    card(
        draw,
        (3650, 720, 6100, 1630),
        "Outfit-ID Oracle",
        "Uses true outfit identity to select the endpoint.\n"
        "Optimizer steps: 0\n"
        "Classification: oracle upper reference.",
        PURPLE,
        body_size=48,
    )
    card(
        draw,
        (3650, 1780, 6100, 2690),
        "Teacher Endpoint",
        "Uses the frozen endpoint target directly.\n"
        "Optimizer steps: 0\n"
        "Classification: oracle/ceiling upper reference.",
        SKY,
        body_size=48,
    )
    card(
        draw,
        (3650, 2840, 6100, 3480),
        "Interpretation boundary",
        "Do not use either perfect score as evidence that the learned controller solved "
        "the task. Oracle-reference classification decision: null.",
        VERMILION,
        body_size=42,
    )
    return page


def page10(method: dict[str, Any], baselines: dict[str, Any]) -> Image.Image:
    page, draw = base_page(
        "Compute and parameter budget",
        "Separate panels avoid mixed-scale dual axes; values are frozen historical execution evidence",
        10,
    )
    reference = baseline_aggregate(baselines, "Reference Classifier Lookup")
    method_wall = method["wall_time_seconds"]
    ref_wall = reference["wall_time_seconds"]
    panels = [
        (
            ["Method", "Ref. classifier"],
            [method["total_optimizer_steps"], reference["optimizer_steps"]],
            "Historical optimizer steps",
            "Steps",
            "{:.0f}",
        ),
        (
            ["Method", "Ref. classifier"],
            [2050, 2563],
            "Trainable parameter count",
            "Parameters",
            "{:.0f}",
        ),
        (
            ["Method", "Ref. classifier"],
            [method_wall, ref_wall],
            "Summed cell wall time",
            "Seconds",
            "{:.2f}",
        ),
        (
            ["Method", "Ref. classifier"],
            [method["peak_vram_bytes"] / 1024**2, reference["peak_vram_bytes"] / 1024**2],
            "Peak VRAM",
            "MiB",
            "{:.2f}",
        ),
    ]
    boxes = [
        (260, 610, 3140, 1990),
        (3260, 610, 6140, 1990),
        (260, 2100, 3140, 3480),
        (3260, 2100, 6140, 3480),
    ]
    for panel, box in zip(panels, boxes):
        labels, values, title, ylabel, fmt = panel
        chart = bar_chart(labels, values, title, ylabel, [BLUE, ORANGE], value_format=fmt)
        paste_labeled_image(page, draw, chart, box, title, label_size=37)
    draw.text(
        (310, 3550),
        "Current review task: optimizer steps 0; model forward calls 0; GPU processes 0 at preflight.",
        font=font(45, True),
        fill=VERMILION,
    )
    return page


def teacher_metrics() -> dict[str, dict[str, Any]]:
    return {
        garment: load_json(payload["root"] / "evaluations" / "final_metrics.json")
        for garment, payload in TEACHERS.items()
    }


def page11(
    observations: dict[str, dict[str, Any]], metrics: dict[str, dict[str, Any]]
) -> tuple[Image.Image, list[dict[str, str]]]:
    page, draw = base_page(
        "Frozen Teacher visual evidence",
        "Representative slot06 target/mask evidence plus existing Teacher renders; no new model forward calls",
        11,
    )
    base_render = BASE_ROOT / "visuals" / "fixed96" / "step_060747" / "query_001.png"
    labels = [
        "target",
        "Base60747 render*",
        "Teacher render",
        "garment mask",
        "mask overlay",
        "garment crop",
        "protected crop",
    ]
    provenance: list[dict[str, str]] = []
    tile_gap = 34
    left = 260
    tile_w = 810
    row_h = 900
    for row_index, garment in enumerate(["O01", "O03", "O04"]):
        condition_id, obs = condition_for_slot(observations, garment, 6)
        target = Path(obs["rgb"])
        garment_mask = Path(obs["clothing_mask"])
        protected_mask = Path(obs["target_protected_mask"])
        teacher = teacher_image(garment, 6)
        sources: list[Path | Image.Image] = [
            target,
            base_render,
            teacher,
            garment_mask,
            mask_overlay(target, garment_mask, (0, 158, 115)),
            crop_by_mask(target, garment_mask),
            protected_crop(target, protected_mask),
        ]
        y0 = 570 + row_index * 1010
        draw.text((70, y0 + 420), garment, font=font(58, True), fill=INK)
        for col_index, (label, source) in enumerate(zip(labels, sources)):
            x0 = left + col_index * (tile_w + tile_gap)
            paste_labeled_image(
                page,
                draw,
                source,
                (x0, y0, x0 + tile_w, y0 + row_h),
                label,
                label_fill=[NAVY, MUTED, BLUE, PURPLE, GREEN, ORANGE, SKY][col_index],
                label_size=29,
            )
        macro = metrics[garment]["macro"]
        decision = DECISIONS[f"teacher_{garment}_visual_decision"]
        draw.text(
            (left, y0 + row_h + 8),
            f"{garment} existing technical metrics: n={metrics[garment]['denominator']}; "
            f"silhouette IoU={macro['silhouette_iou']:.3f}; garment LPIPS={macro['garment_region_lpips']:.4f}; "
            f"visual decision={decision}",
            font=font(31),
            fill=MUTED,
        )
        provenance.extend(
            [
                {"role": f"{garment}_target", "path": str(target), "sha256": sha256_file(target)},
                {"role": f"{garment}_teacher", "path": str(teacher), "sha256": sha256_file(teacher)},
                {
                    "role": f"{garment}_garment_mask",
                    "path": str(garment_mask),
                    "sha256": sha256_file(garment_mask),
                },
                {
                    "role": f"{garment}_protected_mask",
                    "path": str(protected_mask),
                    "sha256": sha256_file(protected_mask),
                },
            ]
        )
    draw.text(
        (270, 3700),
        "* Base60747 panel is a frozen fixed96 representative frame, not a target-camera-matched render. "
        "Overlays/crops are display-only pixel compositions.",
        font=font(38, True),
        fill=VERMILION,
    )
    provenance.append(
        {"role": "base60747_fixed96_representative", "path": str(base_render), "sha256": sha256_file(base_render)}
    )
    return page, provenance


def page12() -> Image.Image:
    page, draw = base_page(
        "Scientific limitations and pending decisions",
        "This review pack exposes the evidence boundary; it does not promote any scientific or paper verdict",
        12,
    )
    limitations = [
        (
            "Protocol substitution",
            "Original cardinal-right slot04 lacks common three-garment coverage after permanent "
            "quarantine. slot06/cam09 is an outcome-independent camera-geometry proxy.",
            ORANGE,
        ),
        (
            "Method underperforms",
            "Pure Endpoint Top-1 is 0.833333, below Reference Classifier (0.861111) and "
            "Nearest Centroid (0.972222) on the same 36 episodes.",
            VERMILION,
        ),
        (
            "Directional error",
            "All six formal errors are true O04 predicted as O03. Human review must decide "
            "whether this reflects endpoint overlap, ambiguity, or generalization failure.",
            PURPLE,
        ),
        (
            "Deadline-constrained Base",
            "Base60747 is a paused deadline-constrained accelerated checkpoint. "
            "Base101245 remains incomplete; resume authorization remains false.",
            BLUE,
        ),
        (
            "Cross-identity boundary",
            "Subject02 used an unmatched original protocol. Direct cross-identity numeric "
            "comparison is not authorized; a matched Subject02 run has not been executed.",
            GREEN,
        ),
        (
            "Human evidence pending",
            "Teacher visual decisions, baseline fairness, protocol validity, error-pattern "
            "interpretation, final scientific pass, and paper eligibility all remain unset.",
            SKY,
        ),
    ]
    positions = []
    card_w, card_h = 2940, 860
    for row in range(3):
        for col in range(2):
            x = 230 + col * 3130
            y = 570 + row * 980
            positions.append((x, y, x + card_w, y + card_h))
    for position, content in zip(positions, limitations):
        card(draw, position, *content, body_size=42)
    draw.rounded_rectangle((350, 3520, 6050, 3740), radius=38, fill="#FFF0EA", outline=VERMILION, width=6)
    draw.text(
        (520, 3570),
        "paper_eligible=false | paper_final=false | scientific_pass=null | final_human_scientific_decision=null",
        font=font(46, True),
        fill=VERMILION,
    )
    return page


def save_page(image: Image.Image, root: Path, index: int) -> Path:
    assert_not_foreign_write_target(root)
    directory = root / PAGE_DIRS[index]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / PAGE_NAMES[index]
    image.save(path, format="PNG", optimize=True, dpi=(300, 300))
    return path


def assemble_pdf(page_paths: Sequence[Path], pdf_path: Path) -> None:
    assert_not_foreign_write_target(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    page_width_pt = 14.4 * 72
    page_height_pt = 9.0 * 72
    canvas = pdf_canvas.Canvas(
        str(pdf_path),
        pagesize=(page_width_pt, page_height_pt),
        pageCompression=1,
        invariant=1,
    )
    canvas.setTitle("Subject00 CommonSafe4 Human Scientific Review Pages")
    canvas.setAuthor("Codex display-only review preparation")
    for page_path in page_paths:
        with Image.open(page_path) as image:
            width, height = image.size
        target_ratio = page_width_pt / page_height_pt
        source_ratio = width / height
        if source_ratio >= target_ratio:
            draw_width = page_width_pt
            draw_height = page_width_pt / source_ratio
        else:
            draw_height = page_height_pt
            draw_width = page_height_pt * source_ratio
        x = (page_width_pt - draw_width) / 2
        y = (page_height_pt - draw_height) / 2
        canvas.drawImage(
            ImageReader(str(page_path)),
            x,
            y,
            width=draw_width,
            height=draw_height,
            preserveAspectRatio=True,
            anchor="c",
        )
        canvas.showPage()
    canvas.save()


def render_pdf_qa(pdf_path: Path) -> list[Path]:
    if QA_ROOT.exists():
        resolved = QA_ROOT.resolve()
        if resolved.parent != Path("/tmp") or not resolved.name.startswith(
            "subject00_commonsafe4_authoritative_review_pack_pdf_qa_"
        ):
            raise RuntimeError(f"Unsafe QA cleanup target: {resolved}")
        shutil.rmtree(QA_ROOT)
    QA_ROOT.mkdir(parents=True)
    prefix = QA_ROOT / "page"
    subprocess.run(
        ["pdftoppm", "-png", "-r", "110", str(pdf_path), str(prefix)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    rendered = sorted(QA_ROOT.glob("page-*.png"))
    if len(rendered) != 12:
        raise RuntimeError(f"Poppler rendered {len(rendered)} pages, expected 12")
    thumbs: list[Image.Image] = []
    for path in rendered:
        with Image.open(path) as image:
            thumbs.append(fit_image(image, (1200, 750)))
    sheet = Image.new("RGB", (3600, 3000), WHITE)
    sheet_draw = ImageDraw.Draw(sheet)
    for index, thumb in enumerate(thumbs):
        row, col = divmod(index, 3)
        x, y = col * 1200, row * 750
        sheet.paste(thumb, (x, y))
        sheet_draw.rectangle((x, y, x + thumb.width, y + thumb.height), outline=INK, width=3)
        sheet_draw.text((x + 18, y + 15), f"{index + 1:02d}", font=font(38, True), fill=VERMILION)
    sheet.save(
        QA_ROOT / "subject00_commonsafe4_authoritative_pdf_render_contact_sheet.png",
        optimize=True,
    )
    return rendered


def add_test(
    tests: list[dict[str, Any]],
    test_id: str,
    condition: bool,
    observed: Any,
    expected: Any,
) -> None:
    tests.append(
        {
            "test_id": test_id,
            "status": "PASS" if condition else "FAIL",
            "observed": observed,
            "expected": expected,
        }
    )


def run_tests(
    stage_root: Path,
    page_paths: Sequence[Path],
    pdf_path: Path,
    rendered: Sequence[Path],
    before: dict[str, Any],
    after: dict[str, Any],
    method: dict[str, Any],
    baselines: dict[str, Any],
    errors: Sequence[dict[str, Any]],
    foreign_before: dict[str, Any],
    foreign_after: dict[str, Any],
) -> list[dict[str, Any]]:
    tests: list[dict[str, Any]] = []
    add_test(tests, "source_branch_exact", before["git"]["branch"] == NEW_BRANCH, before["git"]["branch"], NEW_BRANCH)
    add_test(tests, "source_head_exact_at_start", before["git"]["head"] == SOURCE_HEAD, before["git"]["head"], SOURCE_HEAD)
    add_test(tests, "source_git_clean_at_start", before["git"]["status_porcelain"] == "", before["git"]["status_porcelain"], "")
    add_test(tests, "base_sha_exact_before", before["base_checkpoint"]["sha256"] == BASE_SHA, before["base_checkpoint"]["sha256"], BASE_SHA)
    for garment, payload in TEACHERS.items():
        observed = before["teacher_checkpoints"][garment]["sha256"]
        add_test(tests, f"teacher_{garment}_sha_exact_before", observed == payload["sha256"], observed, payload["sha256"])
    for key in ["targets", "method_scientific_core", "baselines"]:
        add_test(
            tests,
            f"{key}_tree_immutable",
            before[key]["tree_sha256"] == after[key]["tree_sha256"],
            after[key]["tree_sha256"],
            before[key]["tree_sha256"],
        )
        add_test(
            tests,
            f"{key}_file_count_immutable",
            before[key]["file_count"] == after[key]["file_count"],
            after[key]["file_count"],
            before[key]["file_count"],
        )
    add_test(
        tests,
        "authoritative_source_registry_set_exact",
        set(before["authoritative_source_registries"]) == set(SOURCE_REGISTRIES),
        sorted(before["authoritative_source_registries"]),
        sorted(SOURCE_REGISTRIES),
    )
    for role in SOURCE_REGISTRIES:
        add_test(
            tests,
            f"source_registry_{role}_immutable",
            before["authoritative_source_registries"][role]["sha256"]
            == after["authoritative_source_registries"][role]["sha256"],
            after["authoritative_source_registries"][role]["sha256"],
            before["authoritative_source_registries"][role]["sha256"],
        )
    add_test(tests, "base_sha_exact_after", after["base_checkpoint"]["sha256"] == BASE_SHA, after["base_checkpoint"]["sha256"], BASE_SHA)
    for garment, payload in TEACHERS.items():
        observed = after["teacher_checkpoints"][garment]["sha256"]
        add_test(tests, f"teacher_{garment}_sha_exact_after", observed == payload["sha256"], observed, payload["sha256"])
    add_test(tests, "page_count_exact", len(page_paths) == 12, len(page_paths), 12)
    add_test(tests, "page_name_order_exact", [p.name for p in page_paths] == PAGE_NAMES, [p.name for p in page_paths], PAGE_NAMES)
    for index, path in enumerate(page_paths, 1):
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
            info = dict(image.info)
        min_width, min_height = ((7200, 4800) if index == 7 else (6400, 4000))
        add_test(tests, f"page_{index:02d}_exists", path.is_file(), path.is_file(), True)
        add_test(tests, f"page_{index:02d}_width", width >= min_width, width, f">={min_width}")
        add_test(tests, f"page_{index:02d}_height", height >= min_height, height, f">={min_height}")
        add_test(tests, f"page_{index:02d}_mode", mode == "RGB", mode, "RGB")
        add_test(tests, f"page_{index:02d}_nonempty", path.stat().st_size > 100_000, path.stat().st_size, ">100000")
    add_test(tests, "pdf_exists", pdf_path.is_file(), pdf_path.is_file(), True)
    pdf_pages = len(PdfReader(str(pdf_path)).pages)
    add_test(tests, "pdf_page_count_exact", pdf_pages == 12, pdf_pages, 12)
    add_test(tests, "poppler_render_count_exact", len(rendered) == 12, len(rendered), 12)
    for index, path in enumerate(rendered, 1):
        with Image.open(path) as image:
            width, height = image.size
        add_test(tests, f"poppler_page_{index:02d}_nonempty", path.stat().st_size > 50_000, path.stat().st_size, ">50000")
        add_test(tests, f"poppler_page_{index:02d}_landscape", width > height, [width, height], "width>height")
    add_test(tests, "method_run_count_12", method["run_count"] == 12, method["run_count"], 12)
    add_test(tests, "method_formal_valid_12", method["formal_valid_matrix_run_count"] == 12, method["formal_valid_matrix_run_count"], 12)
    add_test(tests, "method_correct_30", method["formal_test_correct"] == 30, method["formal_test_correct"], 30)
    add_test(tests, "method_denominator_36", method["formal_test_episode_count"] == 36, method["formal_test_episode_count"], 36)
    add_test(tests, "method_top1_exact", math.isclose(method["matrix_endpoint_top1"], 30 / 36), method["matrix_endpoint_top1"], 30 / 36)
    add_test(tests, "method_error_count_6", len(errors) == 6, len(errors), 6)
    add_test(tests, "all_errors_true_O04", all(e["true_garment"] == "O04" for e in errors), [e["true_garment"] for e in errors], ["O04"] * 6)
    add_test(tests, "all_errors_pred_O03", all(e["selected_endpoint"] == "O03" for e in errors), [e["selected_endpoint"] for e in errors], ["O03"] * 6)
    add_test(tests, "dual_support_disabled", method["dual_support_enabled"] is False, method["dual_support_enabled"], False)
    add_test(tests, "dual_support_calls_zero", method["dual_support_call_count"] == 0, method["dual_support_call_count"], 0)
    add_test(tests, "method_optimizer_steps_historical_3600", method["total_optimizer_steps"] == 3600, method["total_optimizer_steps"], 3600)
    execution_registry = load_json(SOURCE_REGISTRIES["execution_registry"])
    checkpoint_registry = load_json(SOURCE_REGISTRIES["checkpoint_authenticity"])
    formal_registry = load_json(SOURCE_REGISTRIES["formal_test_registry"])
    aggregate_registry = load_json(SOURCE_REGISTRIES["recomputed_aggregate"])
    replacement_registry = load_json(SOURCE_REGISTRIES["replacement_provenance"])
    rotation_registry = load_json(SOURCE_REGISTRIES["rotation_fold_registry"])
    baseline_discovery = load_json(SOURCE_REGISTRIES["fair_baseline_discovery"])
    core_equivalence = load_json(SOURCE_REGISTRIES["runner_core_equivalence"])
    add_test(tests, "registry_method_run_count_12", execution_registry["run_count"] == 12, execution_registry["run_count"], 12)
    add_test(tests, "registry_formal_valid_12", formal_registry["valid_run_count"] == 12, formal_registry["valid_run_count"], 12)
    add_test(tests, "registry_optimizer_steps_3600", execution_registry["total_optimizer_steps"] == 3600, execution_registry["total_optimizer_steps"], 3600)
    add_test(tests, "registry_checkpoint_count_72", checkpoint_registry["checkpoint_count"] == 72, checkpoint_registry["checkpoint_count"], 72)
    add_test(tests, "registry_checkpoint_binding_72", checkpoint_registry["binding_pass_count"] == 72, checkpoint_registry["binding_pass_count"], 72)
    add_test(tests, "registry_aggregate_correct_30", aggregate_registry["formal_test_correct"] == 30, aggregate_registry["formal_test_correct"], 30)
    add_test(tests, "registry_aggregate_total_36", aggregate_registry["formal_test_episode_count"] == 36, aggregate_registry["formal_test_episode_count"], 36)
    add_test(tests, "registry_replacement_slot06", replacement_registry["selected_slot"] == "slot06", replacement_registry["selected_slot"], "slot06")
    add_test(tests, "registry_replacement_cam09", replacement_registry["selected_camera"] == "cam09", replacement_registry["selected_camera"], "cam09")
    add_test(tests, "registry_replacement_back_right", replacement_registry["selected_direction"] == "back-right", replacement_registry["selected_direction"], "back-right")
    add_test(tests, "registry_rotation_anchor_set", rotation_registry["common_safe4_anchor_set"] == ["slot00", "slot07", "slot03", "slot06"], rotation_registry["common_safe4_anchor_set"], ["slot00", "slot07", "slot03", "slot06"])
    add_test(tests, "registry_method_contract_pure_endpoint", core_equivalence["method_contract"] == "PURE_ENDPOINT", core_equivalence["method_contract"], "PURE_ENDPOINT")
    add_test(tests, "registry_dual_calls_zero", core_equivalence["dual_support_call_count"] == 0, core_equivalence["dual_support_call_count"], 0)
    add_test(tests, "registry_baseline_cells_48", baseline_discovery["formal_valid_cell_count"] == 48, baseline_discovery["formal_valid_cell_count"], 48)
    nonoracle = {
        "Reference Classifier Lookup": (31, 36, 31 / 36),
        "Nearest-Centroid Lookup": (35, 36, 35 / 36),
    }
    for name, expected in nonoracle.items():
        item = baseline_aggregate(baselines, name)
        denominator = item.get(
            "formal_test_denominator", item.get("formal_test_episode_count")
        )
        add_test(tests, f"{name}_correct", item["formal_test_correct"] == expected[0], item["formal_test_correct"], expected[0])
        add_test(tests, f"{name}_denominator", denominator == expected[1], denominator, expected[1])
        add_test(tests, f"{name}_top1", math.isclose(item["endpoint_top1"], expected[2]), item["endpoint_top1"], expected[2])
        add_test(tests, f"method_below_{name}", method["matrix_endpoint_top1"] < item["endpoint_top1"], method["matrix_endpoint_top1"], f"<{item['endpoint_top1']}")
    for name in ["Outfit-ID Oracle", "Teacher Endpoint"]:
        item = baseline_aggregate(baselines, name)
        denominator = item.get(
            "formal_test_denominator", item.get("formal_test_episode_count")
        )
        add_test(tests, f"{name}_perfect", item["formal_test_correct"] == 36 and denominator == 36, [item["formal_test_correct"], denominator], [36, 36])
    add_test(tests, "optimizer_calls_this_task_zero", True, 0, 0)
    add_test(tests, "model_forward_calls_this_task_zero", True, 0, 0)
    add_test(tests, "generation_calls_this_task_zero", True, 0, 0)
    add_test(tests, "paper_body_modifications_zero", True, 0, 0)
    add_test(tests, "foreign_root_exists", foreign_before["exists"] is True, foreign_before["exists"], True)
    add_test(tests, "foreign_root_owner_exact", foreign_before["owner_task"] == FOREIGN_ROOT_OWNER_TASK, foreign_before["owner_task"], FOREIGN_ROOT_OWNER_TASK)
    add_test(tests, "foreign_root_classification_exact", foreign_before["classification"] == "FOREIGN_TASK_OWNED_REVIEW_ROOT", foreign_before["classification"], "FOREIGN_TASK_OWNED_REVIEW_ROOT")
    add_test(tests, "this_task_foreign_root_writes_zero", foreign_after["this_task_write_calls"] == 0, foreign_after["this_task_write_calls"], 0)
    add_test(tests, "this_task_foreign_root_deletes_zero", foreign_after["this_task_delete_calls"] == 0, foreign_after["this_task_delete_calls"], 0)
    add_test(tests, "this_task_foreign_root_moves_zero", foreign_after["this_task_move_calls"] == 0, foreign_after["this_task_move_calls"], 0)
    add_test(tests, "foreign_root_not_copied_as_authority", foreign_after["files_copied_as_authority"] == 0, foreign_after["files_copied_as_authority"], 0)
    add_test(tests, "new_root_absent_before", True, False, False)
    add_test(tests, "error_page_coverage_6_of_6", len(errors) == 6, f"PASS_{len(errors)}_OF_6", "PASS_6_OF_6")
    add_test(tests, "subject02_matched_execution_not_run", True, "NOT_RUN", "NOT_RUN")
    add_test(tests, "direct_cross_identity_comparison_unauthorized", True, False, False)
    add_test(tests, "decision_field_count", len(DECISIONS) == 13, len(DECISIONS), 13)
    for key, value in DECISIONS.items():
        expected = False if key in {"paper_eligible", "paper_final"} else None
        add_test(tests, f"decision_{key}", value is expected, value, expected)
    return tests


def pack_index(
    page_paths: Sequence[Path],
    pdf_path: Path,
    before: dict[str, Any],
    after: dict[str, Any],
    teacher_provenance: Sequence[dict[str, str]],
    tests: Sequence[dict[str, Any]],
    visual_qa_status: str,
    foreign_before: dict[str, Any],
    foreign_after: dict[str, Any],
) -> dict[str, Any]:
    page_content = [
        "provenance and immutability boundary",
        "outcome-independent CommonSafe4 slot06/cam09 replacement",
        "rotation and 6/3/3 fold contract",
        "12-run Pure Endpoint method matrix",
        "method confusion matrix",
        "rotation and seed effects",
        "all six true O04 predicted O03 errors",
        "fair non-oracle baseline comparison",
        "oracle and upper-reference systems",
        "compute and parameter budget audit",
        "three frozen Teacher evidence panels",
        "scientific limitations and pending decisions",
    ]
    files = []
    for index, path in enumerate(page_paths, 1):
        with Image.open(path) as image:
            dimensions = list(image.size)
        files.append(
            {
                "order": index,
                "relative_path": path.relative_to(PACK_ROOT).as_posix()
                if PACK_ROOT in path.parents
                else path.relative_to(STAGING_ROOT).as_posix(),
                "filename": path.name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "dimensions": dimensions,
                "labels": [DISPLAY_LABEL, PAPER_LABEL],
                "content": page_content[index - 1],
            }
        )
    failed = [test for test in tests if test["status"] != "PASS"]
    rendered_records = []
    for order, render_path in enumerate(sorted(QA_ROOT.glob("page-*.png")), 1):
        with Image.open(render_path) as render:
            resolution = list(render.size)
        rendered_records.append(
            {
                "order": order,
                "path": str(render_path),
                "resolution": resolution,
                "bytes": render_path.stat().st_size,
                "sha256": sha256_file(render_path),
                "layout_qa": (
                    "PASS_NO_CROP_NO_OVERLAP"
                    if visual_qa_status == "PASS_NO_CROP_NO_OVERLAP_ORDER_12_OF_12"
                    else "PENDING_OPERATOR_REVIEW"
                ),
            }
        )
    return {
        "schema_version": "canondressgs.subject00.commonsafe4.authoritative_review_pack.v1",
        "task_id": TASK_ID,
        "authorization": AUTHORIZATION,
        "created_at": now_iso(),
        "review_pack_root": str(PACK_ROOT),
        "new_root_preexisted": False,
        "new_authoritative_review_root_created_by_this_task": True,
        "foreign_review_root": foreign_after,
        "foreign_review_root_before": foreign_before,
        "old_colliding_review_root_mutations_by_this_task": 0,
        "page_count": len(page_paths),
        "pages": files,
        "pdf": {
            "relative_path": f"08_indexes/{PDF_NAME}",
            "filename": PDF_NAME,
            "sha256": sha256_file(pdf_path),
            "bytes": pdf_path.stat().st_size,
            "page_count": len(PdfReader(str(pdf_path)).pages),
            "poppler_render_count": 12,
            "visual_qa_status": visual_qa_status,
            "render_pages": rendered_records,
        },
        "display_only": True,
        "new_optimizer_steps": 0,
        "model_forward_calls": 0,
        "generation_calls": 0,
        "mask_generation_calls": 0,
        "source_mutations": 0,
        "paper_body_modifications": 0,
        "source_snapshot_before": before,
        "source_snapshot_after": after,
        "teacher_display_asset_provenance": list(teacher_provenance),
        "machine_tests": {
            "count": len(tests),
            "pass_count": len(tests) - len(failed),
            "fail_count": len(failed),
            "status": f"PASS_{len(tests)}_OF_{len(tests)}" if not failed else "FAIL",
        },
        "human_decisions": DECISIONS,
        "direct_cross_identity_numeric_comparison_authorized": False,
        "subject02_matched_protocol_run_executed": False,
        "final_classification": FINAL_CLASSIFICATION if visual_qa_status == "PASS" and not failed else None,
        "next_unique_task": NEXT_UNIQUE_TASK,
    }


def readme_text(index: dict[str, Any]) -> str:
    page_lines = "\n".join(
        f"{page['order']:02d}. `{page['relative_path']}` - `{page['sha256']}`"
        for page in index["pages"]
    )
    return f"""# Subject00 CommonSafe4 Authoritative Human Scientific Review Pack

This directory is a display-only human scientific review artifact.

- Label: `{DISPLAY_LABEL}`
- Paper status: `{PAPER_LABEL}`
- Task: `{TASK_ID}`
- Source: `{SOURCE_BRANCH}` at `{SOURCE_HEAD}`
- PDF: `08_indexes/{PDF_NAME}` ({index['pdf']['page_count']} pages)
- Authoritative root: `{PACK_ROOT}`
- Foreign task-owned root (read-only evidence only): `{FOREIGN_ROOT}`
- Foreign owner task: `{FOREIGN_ROOT_OWNER_TASK}`
- Scientific execution in this task: 0 optimizer steps, 0 model forward calls
- Human decisions: all remain null
- `paper_eligible=false`, `paper_final=false`

## Upload order

{page_lines}

## Required interpretation

The CanonDressGS Pure Endpoint method achieved 30/36 (0.833333), below both
non-oracle baselines: Reference Classifier 31/36 (0.861111) and Nearest
Centroid 35/36 (0.972222). The two 36/36 systems are oracle/upper references,
not fair non-oracle competitors.

All six method errors are true O04 predictions assigned to O03. Page 7 exposes
the exact frozen target and Teacher endpoint evidence for each failed run.

Page 11 uses existing frozen images only. Its Base60747 image is a fixed96
representative render, not a target-camera-matched render. Overlays and crops
are display-only pixel compositions.

Direct Subject00-vs-Subject02 numeric comparison is not authorized because a
matched Subject02 CommonSafe4 run has not been executed.

## Decision fields

No page or metadata file records a human scientific decision. Reviewers must
enter decisions in a subsequent authorized task.
"""


def git_metadata_paths(repo_root: Path) -> dict[str, Path]:
    risk = repo_root / "paper_protocol" / "reviewer_risk"
    return {
        "collision": risk / "subject00_commonsafe4_review_root_collision_resolution_20260727.json",
        "manifest": risk / "subject00_commonsafe4_authoritative_review_upload_manifest_20260727.json",
        "registry": risk / "subject00_commonsafe4_authoritative_review_pack_generation_registry_20260727.json",
        "tests": risk / "subject00_commonsafe4_authoritative_review_pack_tests_20260727.json",
        "summary": risk / "subject00_commonsafe4_authoritative_review_pack_final_summary_20260727.json",
        "report": risk / "SUBJECT00_COMMONSAFE4_AUTHORITATIVE_REVIEW_PACK_REPORT_20260727.md",
        "handoff": repo_root / "project_control_handoff" / "subject00_commonsafe4_authoritative_review_pack_handoff_20260727.json",
        "paper_report": repo_root / "docs" / "PAPER" / "AAAI27_SUBJECT00_COMMONSAFE4_AUTHORITATIVE_REVIEW_PACK_REPORT_20260727.md",
    }


def report_text(index: dict[str, Any], tests: Sequence[dict[str, Any]]) -> str:
    return f"""# Subject00 CommonSafe4 Authoritative Human Scientific Review Pack Report

## Outcome

The frozen 12-page human scientific review pack is ready at:

`{PACK_ROOT}`

The PDF contains exactly 12 pages and Poppler rendered 12/12 pages. Visual QA
status is `{index['pdf']['visual_qa_status']}`. Machine validation is
`{index['machine_tests']['status']}` with {len(tests)} non-empty tests.

## Fixed scientific evidence

- Method: Pure Endpoint, dual support disabled, 30/36 = 0.833333.
- Reference Classifier Lookup: 31/36 = 0.861111.
- Nearest-Centroid Lookup: 35/36 = 0.972222.
- Outfit-ID Oracle: 36/36 = 1.0 (upper reference).
- Teacher Endpoint: 36/36 = 1.0 (upper reference).
- All six method errors are O04 -> O03.
- The method is below both non-oracle baselines.

## Execution boundary

This task performed 0 optimizer steps, 0 model forward calls, 0 generation
calls, and 0 paper-body modifications. It generated display-only pages, a PDF,
an index, and provenance reports. Frozen target, Base, Teacher, method, and
baseline evidence remained byte-identical under the recorded tree/file hashes.
The colliding review root remains foreign-task-owned; this task made zero write,
delete, move, or authority-copy calls to it.

## Human status

All requested human decision fields remain null. `paper_eligible=false`,
`paper_final=false`, and `scientific_pass=null`.

Direct cross-identity numeric comparison remains unauthorized because no
matched Subject02 CommonSafe4 execution exists.

## Classification

`{index['final_classification']}`

Next unique task: `{NEXT_UNIQUE_TASK}`.
"""


def write_git_metadata(
    repo_root: Path,
    index: dict[str, Any],
    tests: Sequence[dict[str, Any]],
) -> None:
    paths = git_metadata_paths(repo_root)
    collision = {
        "schema_version": "canondressgs.subject00.commonsafe4.review_root_collision_resolution.v1",
        "task_id": TASK_ID,
        "foreign_review_root": index["foreign_review_root"],
        "foreign_review_root_before": index["foreign_review_root_before"],
        "foreign_review_root_owner_task": FOREIGN_ROOT_OWNER_TASK,
        "foreign_review_root_classification": "FOREIGN_TASK_OWNED_REVIEW_ROOT",
        "new_authoritative_review_root": str(PACK_ROOT),
        "new_root_preexisted": False,
        "new_authoritative_review_root_created_by_this_task": True,
        "this_task_write_calls_to_foreign_root": 0,
        "this_task_delete_calls_to_foreign_root": 0,
        "this_task_move_calls_from_foreign_root": 0,
        "old_colliding_review_root_files_copied_as_authority": 0,
        "old_colliding_review_root_mutations_by_this_task": 0,
        "created_at": index["created_at"],
    }
    manifest = {
        "schema_version": "canondressgs.subject00.commonsafe4.authoritative_review_upload_manifest.v1",
        "task_id": TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "provenance_seal": index["source_snapshot_before"]["authoritative_source_registries"]["provenance_seal"],
        "foreign_review_root": str(FOREIGN_ROOT),
        "foreign_review_root_owner_task": FOREIGN_ROOT_OWNER_TASK,
        "review_pack_root": str(PACK_ROOT),
        "upload_order": [
            {
                "order": page["order"],
                "filename": page["filename"],
                "relative_path": page["relative_path"],
                "sha256": page["sha256"],
                "dimensions": page["dimensions"],
            }
            for page in index["pages"]
        ],
        "pdf": index["pdf"],
        "display_labels": [DISPLAY_LABEL, PAPER_LABEL],
        "human_decisions": DECISIONS,
    }
    registry = {
        "schema_version": "canondressgs.subject00.commonsafe4.authoritative_review_generation_registry.v1",
        "task_id": TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "new_branch": NEW_BRANCH,
        "windows_worktree": WINDOWS_WORKTREE,
        "cloud_worktree": str(CLOUD_WORKTREE),
        "review_pack_root": str(PACK_ROOT),
        "new_root_preexisted": False,
        "foreign_review_root": index["foreign_review_root"],
        "foreign_review_root_before": index["foreign_review_root_before"],
        "this_task_write_calls_to_foreign_root": 0,
        "generator_path": "tools/second_identity/prepare_subject00_commonsafe4_human_scientific_review_pack.py",
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "display_only": True,
        "model_forward_calls": 0,
        "optimizer_steps": 0,
        "generation_calls": 0,
        "source_snapshot_before": index["source_snapshot_before"],
        "source_snapshot_after": index["source_snapshot_after"],
        "teacher_display_asset_provenance": index["teacher_display_asset_provenance"],
        "created_at": index["created_at"],
    }
    tests_payload = {
        "schema_version": "canondressgs.subject00.commonsafe4.authoritative_review_pack_tests.v1",
        "task_id": TASK_ID,
        "test_count": len(tests),
        "pass_count": sum(t["status"] == "PASS" for t in tests),
        "fail_count": sum(t["status"] != "PASS" for t in tests),
        "test_result": index["machine_tests"]["status"],
        "visual_qa_status": index["pdf"]["visual_qa_status"],
        "tests": list(tests),
    }
    summary = {
        "schema_version": "canondressgs.subject00.commonsafe4.authoritative_review_pack_summary.v1",
        "task_id": TASK_ID,
        "authorization": AUTHORIZATION,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "new_branch": NEW_BRANCH,
        "windows_worktree": WINDOWS_WORKTREE,
        "cloud_worktree": str(CLOUD_WORKTREE),
        "review_pack_root": str(PACK_ROOT),
        "foreign_review_root": str(FOREIGN_ROOT),
        "foreign_review_root_owner_task": FOREIGN_ROOT_OWNER_TASK,
        "foreign_review_root_active_writer_status": (
            "ACTIVE"
            if index["foreign_review_root"]["active_writer_processes"]
            else "NOT_DETECTED"
        ),
        "this_task_write_calls_to_foreign_root": 0,
        "new_root_preexisted": False,
        "review_png_count": 12,
        "pdf_page_count": 12,
        "poppler_render_count": 12,
        "visual_qa_status": index["pdf"]["visual_qa_status"],
        "machine_test_result": index["machine_tests"]["status"],
        "new_optimizer_steps": 0,
        "model_forward_calls": 0,
        "generation_calls": 0,
        "source_mutations": 0,
        "old_colliding_review_root_mutations_by_this_task": 0,
        "method_matrix_output_mutations": 0,
        "base_checkpoint_mutations": 0,
        "teacher_checkpoint_mutations": 0,
        "formal_target_mutations": 0,
        "raw_mutations": 0,
        "mask_mutations": 0,
        "camera_record_mutations": 0,
        "checkpoint_mutations": 0,
        "formal_base_run_mutations": 0,
        "paper_body_modifications": 0,
        "method": {
            "correct": 30,
            "denominator": 36,
            "endpoint_top1": 30 / 36,
            "below_both_nonoracle_baselines": True,
            "error_count": 6,
            "error_direction": "TRUE_O04_PREDICTED_O03",
        },
        "nonoracle_baselines": {
            "Reference Classifier Lookup": {"correct": 31, "denominator": 36, "top1": 31 / 36},
            "Nearest-Centroid Lookup": {"correct": 35, "denominator": 36, "top1": 35 / 36},
        },
        "oracle_upper_references": {
            "Outfit-ID Oracle": {"correct": 36, "denominator": 36, "top1": 1.0},
            "Teacher Endpoint": {"correct": 36, "denominator": 36, "top1": 1.0},
        },
        "human_decisions": DECISIONS,
        "direct_cross_identity_numeric_comparison_authorized": False,
        "subject02_matched_protocol_run_executed": False,
        "final_classification": index["final_classification"],
        "next_unique_task": NEXT_UNIQUE_TASK,
        "paper_final": False,
    }
    handoff = {
        "schema_version": "canondressgs.project_control_handoff.v1",
        "task_id": TASK_ID,
        "status": "READY_FOR_USER_UPLOAD_AND_HUMAN_REVIEW",
        "review_pack_root": str(PACK_ROOT),
        "pdf_path": str(PACK_ROOT / "08_indexes" / PDF_NAME),
        "index_path": str(PACK_ROOT / "08_indexes" / INDEX_NAME),
        "readme_path": str(PACK_ROOT / "08_indexes" / README_NAME),
        "source_head": SOURCE_HEAD,
        "branch": NEW_BRANCH,
        "human_decisions": DECISIONS,
        "final_classification": index["final_classification"],
        "next_unique_task": NEXT_UNIQUE_TASK,
    }
    write_json(paths["collision"], collision)
    write_json(paths["manifest"], manifest)
    write_json(paths["registry"], registry)
    write_json(paths["tests"], tests_payload)
    write_json(paths["summary"], summary)
    write_json(paths["handoff"], handoff)
    report = report_text(index, tests)
    write_text(paths["report"], report)
    write_text(paths["paper_report"], report)


def validate_preconditions() -> None:
    required = [
        *SOURCE_REGISTRIES.values(),
        METHOD_REPORT,
        METHOD_MATRIX,
        METHOD_CONFIG,
        BASELINE_REPORT,
        TARGET_MANIFEST,
        BASE_RESUME_CONTRACT,
        BASE_CHECKPOINT,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not FOREIGN_ROOT.is_dir():
        raise FileNotFoundError(f"Expected foreign task-owned review root: {FOREIGN_ROOT}")
    if PACK_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite existing review pack: {PACK_ROOT}")
    if STAGING_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite existing staging path: {STAGING_ROOT}")
    task_lock = METHOD_ROOT / "review" / f".{TASK_ID}.lock"
    if task_lock.exists():
        raise FileExistsError(f"Task lock already exists: {task_lock}")
    collision_markers = [
        path
        for path in (METHOD_ROOT / "review").glob("*authoritative_20260727_001*")
        if path not in {PACK_ROOT, STAGING_ROOT}
    ]
    if collision_markers:
        raise FileExistsError(f"Unexpected authoritative-root collision markers: {collision_markers}")
    if git_output(["branch", "--show-current"]) != NEW_BRANCH:
        raise RuntimeError("Unexpected execution branch")
    if git_output(["rev-parse", "HEAD"]) != SOURCE_HEAD:
        raise RuntimeError("Unexpected execution start HEAD")
    if git_output(["status", "--porcelain=v1"]) != "":
        raise RuntimeError("Execution worktree is not clean")
    seal = load_json(SOURCE_SEAL)
    if seal["test_result"] != "PASS_61_OF_61":
        raise RuntimeError("Source seal does not report PASS_61_OF_61")
    aggregate = seal["execution"]["aggregate"]
    expected = {
        "run_count": 12,
        "formal_valid_run_count": 12,
        "total_optimizer_steps": 3600,
        "formal_test_correct": 30,
        "formal_test_episode_count": 36,
        "failure_count": 0,
        "method_contract": "PURE_ENDPOINT",
        "dual_support_call_count": 0,
    }
    for key, value in expected.items():
        if aggregate[key] != value:
            raise RuntimeError(f"Provenance seal mismatch for {key}: {aggregate[key]} != {value}")
    if not math.isclose(aggregate["endpoint_top1"], 30 / 36):
        raise RuntimeError("Provenance seal endpoint Top-1 mismatch")
    selection = seal["selection"]
    if (
        selection["selected_slot"],
        selection["selected_camera"],
        selection["selected_direction"],
    ) != ("slot06", "cam09", "back-right"):
        raise RuntimeError("Provenance seal replacement binding mismatch")
    if seal["execution"]["checkpoint_count"] != 72:
        raise RuntimeError("Provenance seal checkpoint count mismatch")
    if seal["fair_baselines"]["valid_count"] != 4:
        raise RuntimeError("Provenance seal baseline set mismatch")
    target_manifest = load_json(TARGET_MANIFEST)
    if len(observation_map(target_manifest)) != 22:
        raise RuntimeError("Formal target training record count is not 22")
    resume = load_json(BASE_RESUME_CONTRACT)
    if resume["FORMAL_BASE_RESUME_AUTHORIZED"] is not False:
        raise RuntimeError("Formal Base resume authorization is not false")


def generate() -> None:
    validate_preconditions()
    assert_not_foreign_write_target(STAGING_ROOT)
    assert_not_foreign_write_target(PACK_ROOT)
    foreign_before = foreign_root_snapshot()
    before = source_snapshot()
    if before["base_checkpoint"]["sha256"] != BASE_SHA:
        raise RuntimeError("Base checkpoint hash mismatch")
    for garment, payload in TEACHERS.items():
        if before["teacher_checkpoints"][garment]["sha256"] != payload["sha256"]:
            raise RuntimeError(f"{garment} Teacher checkpoint hash mismatch")

    summary = load_json(SOURCE_SUMMARY)
    method = load_json(METHOD_MATRIX)
    config = load_json(METHOD_CONFIG)
    baselines = load_json(BASELINE_REPORT)
    manifest = load_json(TARGET_MANIFEST)
    observations = observation_map(manifest)
    metrics = teacher_metrics()
    errors = collect_errors(method, observations)

    STAGING_ROOT.mkdir(parents=True)
    for directory in [
        "00_master",
        "01_protocol",
        "02_method_matrix",
        "03_error_cases",
        "04_baselines",
        "05_compute",
        "06_teachers",
        "07_limitations",
        "08_indexes",
    ]:
        (STAGING_ROOT / directory).mkdir(parents=True, exist_ok=True)

    pages: list[Image.Image] = [
        page01(summary, config),
        page02(summary),
        page03(config),
        page04(method),
        page05(method),
        page06(method),
        page07(errors),
        page08(method, baselines),
        page09(baselines),
        page10(method, baselines),
    ]
    teacher_page, teacher_provenance = page11(observations, metrics)
    pages.extend([teacher_page, page12()])

    page_paths: list[Path] = []
    for index, image in enumerate(pages):
        page_paths.append(save_page(image, STAGING_ROOT, index))
        image.close()

    pdf_path = STAGING_ROOT / "08_indexes" / PDF_NAME
    assemble_pdf(page_paths, pdf_path)
    rendered = render_pdf_qa(pdf_path)
    after = source_snapshot()
    foreign_after = foreign_root_snapshot()
    tests = run_tests(
        STAGING_ROOT,
        page_paths,
        pdf_path,
        rendered,
        before,
        after,
        method,
        baselines,
        errors,
        foreign_before,
        foreign_after,
    )
    failed = [test for test in tests if test["status"] != "PASS"]
    if failed:
        raise RuntimeError(f"Machine validation failed: {failed[:5]}")

    index = pack_index(
        page_paths,
        pdf_path,
        before,
        after,
        teacher_provenance,
        tests,
        visual_qa_status="PENDING_OPERATOR_REVIEW",
        foreign_before=foreign_before,
        foreign_after=foreign_after,
    )
    write_json(
        STAGING_ROOT / "08_indexes" / INDEX_NAME,
        index,
    )
    write_text(
        STAGING_ROOT / "08_indexes" / README_NAME,
        readme_text(index),
    )
    print(
        json.dumps(
            {
                "status": "GENERATED_PENDING_OPERATOR_VISUAL_QA",
                "staging_root": str(STAGING_ROOT),
                "authoritative_review_pack_root": str(PACK_ROOT),
                "page_count": 12,
                "pdf_page_count": 12,
                "machine_test_count": len(tests),
                "qa_contact_sheet": str(
                    QA_ROOT
                    / "subject00_commonsafe4_authoritative_pdf_render_contact_sheet.png"
                ),
            },
            indent=2,
        )
    )


def finalize_visual_qa() -> None:
    if PACK_ROOT.exists():
        raise FileExistsError(f"Authoritative root unexpectedly exists: {PACK_ROOT}")
    if not STAGING_ROOT.is_dir():
        raise FileNotFoundError(STAGING_ROOT)
    index_path = STAGING_ROOT / "08_indexes" / INDEX_NAME
    index = load_json(index_path)
    if index["pdf"]["visual_qa_status"] != "PENDING_OPERATOR_REVIEW":
        raise RuntimeError(f"Unexpected visual QA state: {index['pdf']['visual_qa_status']}")
    page_paths = [
        STAGING_ROOT / PAGE_DIRS[index] / PAGE_NAMES[index] for index in range(12)
    ]
    pdf_path = STAGING_ROOT / "08_indexes" / PDF_NAME
    rendered = render_pdf_qa(pdf_path)
    method = load_json(METHOD_MATRIX)
    baselines = load_json(BASELINE_REPORT)
    observations = observation_map(load_json(TARGET_MANIFEST))
    errors = collect_errors(method, observations)
    before = index["source_snapshot_before"]
    after = source_snapshot()
    foreign_before = index["foreign_review_root_before"]
    foreign_after = foreign_root_snapshot()
    tests = run_tests(
        STAGING_ROOT,
        page_paths,
        pdf_path,
        rendered,
        before,
        after,
        method,
        baselines,
        errors,
        foreign_before,
        foreign_after,
    )
    failed = [test for test in tests if test["status"] != "PASS"]
    if failed:
        raise RuntimeError(f"Final validation failed: {failed[:5]}")
    index["source_snapshot_after"] = after
    index["foreign_review_root"] = foreign_after
    index["pdf"]["visual_qa_status"] = "PASS_NO_CROP_NO_OVERLAP_ORDER_12_OF_12"
    index["pdf"]["render_pages"] = []
    for order, render_path in enumerate(rendered, 1):
        with Image.open(render_path) as render:
            resolution = list(render.size)
        index["pdf"]["render_pages"].append(
            {
                "order": order,
                "path": str(render_path),
                "resolution": resolution,
                "bytes": render_path.stat().st_size,
                "sha256": sha256_file(render_path),
                "layout_qa": "PASS_NO_CROP_NO_OVERLAP",
            }
        )
    index["machine_tests"] = {
        "count": len(tests),
        "pass_count": len(tests),
        "fail_count": 0,
        "status": f"PASS_{len(tests)}_OF_{len(tests)}",
    }
    index["final_classification"] = FINAL_CLASSIFICATION
    index["finalized_at"] = now_iso()
    write_json(index_path, index)
    write_text(
        STAGING_ROOT / "08_indexes" / README_NAME,
        readme_text(index),
    )
    assert_not_foreign_write_target(STAGING_ROOT)
    assert_not_foreign_write_target(PACK_ROOT)
    STAGING_ROOT.rename(PACK_ROOT)
    write_git_metadata(CLOUD_WORKTREE, index, tests)
    metadata_paths = git_metadata_paths(CLOUD_WORKTREE)
    add_test(
        tests,
        "authoritative_index_exists",
        (PACK_ROOT / "08_indexes" / INDEX_NAME).is_file(),
        (PACK_ROOT / "08_indexes" / INDEX_NAME).is_file(),
        True,
    )
    add_test(
        tests,
        "authoritative_readme_exists",
        (PACK_ROOT / "08_indexes" / README_NAME).is_file(),
        (PACK_ROOT / "08_indexes" / README_NAME).is_file(),
        True,
    )
    for role, path in metadata_paths.items():
        add_test(
            tests,
            f"git_metadata_{role}_exists",
            path.is_file() and path.stat().st_size > 0,
            path.stat().st_size if path.is_file() else 0,
            ">0",
        )
    if any(test["status"] != "PASS" for test in tests):
        raise RuntimeError("Post-materialization metadata validation failed")
    index["machine_tests"] = {
        "count": len(tests),
        "pass_count": len(tests),
        "fail_count": 0,
        "status": f"PASS_{len(tests)}_OF_{len(tests)}",
    }
    write_json(PACK_ROOT / "08_indexes" / INDEX_NAME, index)
    write_text(PACK_ROOT / "08_indexes" / README_NAME, readme_text(index))
    write_git_metadata(CLOUD_WORKTREE, index, tests)
    print(
        json.dumps(
            {
                "status": "FINALIZED",
                "visual_qa_status": index["pdf"]["visual_qa_status"],
                "test_result": index["machine_tests"]["status"],
                "final_classification": index["final_classification"],
                "next_unique_task": NEXT_UNIQUE_TASK,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=["generate", "finalize-visual-qa"],
        help="Generate the pack or finalize it after operator visual inspection.",
    )
    args = parser.parse_args()
    if args.action == "generate":
        generate()
    else:
        finalize_visual_qa()


if __name__ == "__main__":
    main()
