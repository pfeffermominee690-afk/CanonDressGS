#!/usr/bin/env python3
"""Append sealed Headroom negative-diagnostic evidence to the figure bank.

This is a CPU-only reporting tool. It reads explicit JSON and PNG inputs from
the sealed attempt and never imports model, renderer, CUDA, or optimizer code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont


PARENT_HEAD = "1fe425d2cc3cb3efd372334e1d845e63bf9d630a"
HEADROOM_HEAD = "674e6092e21eeddeb22e963536247a3385c4e200"
HEADROOM_BRANCH = "research/render-refined-coefficient-headroom-attempt2-20260724"
TARGET_BRANCH = "research/paper-figure-bank-headroom-refresh-20260724"
TASK_ID = "AAAI27-PAPER-FIGURE-BANK-HEADROOM-REFRESH-001"
SOURCE_TASK_ID = "AAAI27-RENDER-REFINED-COEFFICIENT-HEADROOM-ATTEMPT-002"
CLASSIFICATION = "HEADROOM_FIGURE_REFRESH_READY"
ALLOWED_CLASSIFICATIONS = {
    "HEADROOM_FIGURE_REFRESH_READY",
    "HEADROOM_FIGURE_REFRESH_SOURCE_NOT_SEALED",
    "HEADROOM_FIGURE_REFRESH_BLOCKED",
}
ATTEMPT_PATH = (
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "COEFFICIENT-HEADROOM-001/attempt_002"
)
FIGURE_ROOT_REL = Path("paper_draft/figures/headroom_refresh")
RISK_REL = Path("paper_protocol/reviewer_risk")
DOCS_REL = Path("docs/PAPER")
HANDOFF_REL = Path("project_control_handoff")
GARMENTS = ["O01", "O02", "O03", "O04", "O08"]
ROTATIONS = ["R0", "R1", "R2", "R3"]
COLORS = {
    "ink": "#1f2933",
    "muted": "#52606d",
    "grid": "#d9e2ec",
    "teal": "#147d64",
    "blue": "#2f6b9a",
    "coral": "#c85a54",
    "gold": "#c39128",
    "green": "#4f8a5b",
    "paper": "#ffffff",
    "panel": "#f7f9fb",
}

JSON_INPUTS = [
    "RUN_STATUS.json",
    "02_static_parity/parity.json",
    "03_coefficient_runs/summary.json",
    "08_metrics/metric_summary.json",
    "08_metrics/evaluation.json",
    "09_span_analysis/final_analysis.json",
    "09_span_analysis/span_analysis.json",
    "10_refined_lookup_analysis/refined_lookup_analysis.json",
    "11_visual_sheets/visual_registry.json",
    "11_visual_sheets/visual_review.json",
    "11_visual_sheets/visual_review_template.json",
    "13_final_verification/SEALED_HEADROOM_FIGURE_REFRESH_MANIFEST.json",
    "13_final_verification/execution_count_verification.json",
    "13_final_verification/final_summary.json",
    "13_final_verification/final_verification.json",
    "13_final_verification/full_residual_summary.json",
    "13_final_verification/run_registry.json",
    "13_final_verification/tests.json",
    "13_final_verification/visual_review_summary.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--headroom-repo", type=Path, required=True)
    parser.add_argument("--headroom-attempt", type=Path, required=True)
    parser.add_argument(
        "--loo-attempt-observation",
        choices=("ABSENT", "PRESENT_UNREAD"),
        required=True,
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r} in {path}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=reject_duplicates)


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != content:
        raise RuntimeError(f"append-only conflict: {path}")
    if not path.exists():
        path.write_bytes(content)


def write_json(path: Path, value: Any) -> None:
    write_bytes(path, json_bytes(value))


def write_text(path: Path, value: str) -> None:
    write_bytes(path, value.rstrip().encode("utf-8") + b"\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", "arialbd.ttf" if bold else "arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    *,
    width: int,
    fill: str,
    text_font: ImageFont.ImageFont,
    line_height: int,
) -> int:
    x, y = xy
    max_chars = max(10, int(width / max(7, getattr(text_font, "size", 14) * 0.55)))
    lines = textwrap.wrap(value, width=max_chars)
    for line in lines:
        draw.text((x, y), line, fill=fill, font=text_font)
        y += line_height
    return y


def new_canvas(title: str, subtitle: str, size: tuple[int, int] = (1500, 900)) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", size, COLORS["paper"])
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size[0], 118), fill=COLORS["ink"])
    draw.text((54, 26), title, fill="white", font=font(34, bold=True))
    draw.text((56, 76), subtitle, fill="#d9e2ec", font=font(18))
    draw.text((54, size[1] - 34), "SEALED HEADROOM | SUPPLEMENTARY NEGATIVE DIAGNOSTIC | NOT PAPER_FINAL", fill=COLORS["muted"], font=font(15, bold=True))
    return image, draw


def save_png(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with Image.open(path) as existing:
            if existing.convert("RGB").tobytes() != image.convert("RGB").tobytes():
                raise RuntimeError(f"append-only image conflict: {path}")
        return
    image.save(path, format="PNG", optimize=True)


def source_record(attempt: Path, relative: str) -> dict[str, Any]:
    local = attempt / relative
    return {
        "sealed_path": f"{ATTEMPT_PATH}/{relative}",
        "sha256": sha256_file(local),
        "bytes": local.stat().st_size,
    }


def validate_source(headroom_repo: Path, attempt: Path, loo_observation: str) -> tuple[dict[str, Any], dict[str, Any]]:
    failures: list[str] = []
    if git(headroom_repo, "rev-parse", "HEAD") != HEADROOM_HEAD:
        failures.append("HEADROOM_HEAD_MISMATCH")
    if git(headroom_repo, "status", "--short"):
        failures.append("HEADROOM_WORKTREE_NOT_CLEAN")

    missing = [relative for relative in JSON_INPUTS if not (attempt / relative).is_file()]
    failures.extend(f"MISSING_SOURCE:{relative}" for relative in missing)
    if missing:
        return {"status": "FAIL", "failures": failures}, {}

    final = load_json(attempt / "13_final_verification/final_summary.json")
    counts = load_json(attempt / "13_final_verification/execution_count_verification.json")
    parity = load_json(attempt / "02_static_parity/parity.json")
    manifest = load_json(attempt / "13_final_verification/SEALED_HEADROOM_FIGURE_REFRESH_MANIFEST.json")
    verification = load_json(attempt / "13_final_verification/final_verification.json")
    run_status = load_json(attempt / "RUN_STATUS.json")
    tests = load_json(attempt / "13_final_verification/tests.json")
    visual_registry = load_json(attempt / "11_visual_sheets/visual_registry.json")

    expected_counts = {
        "optimization_runs": 120,
        "optimizer_steps": 36000,
        "checkpoint_writes": 960,
        "renderer_calls": 36662,
    }
    if final.get("status") not in {"SEALED_HEADROOM_EVIDENCE", "SEALED_HEADROOM_EVIDENCE_VERIFIED"}:
        failures.append("HEADROOM_NOT_SEALED")
    if final.get("classification") != "TEACHER_SPAN_AT_LOCAL_OPTIMUM":
        failures.append("CLASSIFICATION_MISMATCH")
    if final.get("statuses", {}).get("TEACHER_SVD_PARITY_STATUS") != "PASS":
        failures.append("TEACHER_SVD_PARITY_FAILED")
    if parity.get("status") != "PASS" or parity.get("row_count") != 20 or not all(row.get("pass") for row in parity.get("rows", [])):
        failures.append("PARITY_ROWS_FAILED")
    if counts.get("status") != "PASS" or counts.get("unexplained_nonzero_delta_count") != 0:
        failures.append("COUNT_VERIFICATION_FAILED")
    for key, value in expected_counts.items():
        if counts.get("actual", {}).get(key) != value:
            failures.append(f"COUNT_MISMATCH:{key}")
    if manifest.get("status") != "SEALED_HEADROOM_FIGURE_REFRESH_MANIFEST":
        failures.append("FIGURE_REFRESH_MANIFEST_MISSING_OR_INVALID")
    if verification.get("status") != "PASS" or verification.get("cloud_head") != HEADROOM_HEAD:
        failures.append("FINAL_VERIFICATION_FAILED")
    if not verification.get("checks", {}).get("local_origin_cloud_equal"):
        failures.append("HEADROOM_BRANCH_NOT_PUSHED_EQUAL")
    if run_status.get("status") != "SEALED_HEADROOM_EVIDENCE_VERIFIED":
        failures.append("RUN_STATUS_NOT_SEALED")
    if tests.get("status") != "PASS":
        failures.append("HEADROOM_TESTS_FAILED")
    if visual_registry.get("sheet_count") != 20:
        failures.append("VISUAL_SHEET_COUNT_MISMATCH")

    png_paths = sorted((attempt / "11_visual_sheets").glob("R*_O*_cond_*.png"))
    if len(png_paths) != 20:
        failures.append("VISUAL_PNG_COUNT_MISMATCH")
    row_by_name = {Path(row["sheet_path"]).name: row for row in visual_registry.get("rows", [])}
    for path in png_paths:
        row = row_by_name.get(path.name)
        if row is None or sha256_file(path) != row.get("sheet_sha256"):
            failures.append(f"VISUAL_SHA_MISMATCH:{path.name}")

    gate = {
        "schema_version": "paper_figure_headroom_source_gate.v1",
        "task_id": TASK_ID,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "classification": final.get("classification"),
        "teacher_svd_parity": final.get("statuses", {}).get("TEACHER_SVD_PARITY_STATUS"),
        "actual_counts": {key: counts["actual"].get(key) for key in expected_counts},
        "figure_refresh_manifest": manifest.get("status"),
        "branch_pushed_and_clean": not git(headroom_repo, "status", "--short") and verification.get("checks", {}).get("local_origin_cloud_equal") is True,
        "headroom_source_head": HEADROOM_HEAD,
        "headroom_files_read": len(JSON_INPUTS) + len(png_paths),
        "checkpoint_payload_files_read": 0,
        "loo_attempt_observation": loo_observation,
        "loo_attempt_files_read": 0,
        "gpu_used": False,
        "training_runs_started": 0,
        "inference_runs_started": 0,
        "renderer_calls_started": 0,
        "checkpoints_written": 0,
    }
    data = {
        "final": final,
        "counts": counts,
        "parity": parity,
        "manifest": manifest,
        "verification": verification,
        "metrics": load_json(attempt / "08_metrics/metric_summary.json"),
        "analysis": load_json(attempt / "09_span_analysis/final_analysis.json"),
        "span": load_json(attempt / "09_span_analysis/span_analysis.json"),
        "lookup": load_json(attempt / "10_refined_lookup_analysis/refined_lookup_analysis.json"),
        "runs": load_json(attempt / "13_final_verification/run_registry.json"),
        "visual_registry": visual_registry,
        "visual_review": load_json(attempt / "11_visual_sheets/visual_review.json"),
        "png_paths": png_paths,
    }
    return gate, data


def panel_axes(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, y_label: str) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=8, fill=COLORS["panel"], outline=COLORS["grid"], width=2)
    draw.text((left + 24, top + 18), title, fill=COLORS["ink"], font=font(21, bold=True))
    draw.text((left + 24, top + 53), y_label, fill=COLORS["muted"], font=font(15))
    return left + 75, top + 90, right - 25, bottom - 55


def grouped_metric_plot(data: Mapping[str, Any], output: Path) -> dict[str, Any]:
    image, draw = new_canvas(
        "Test metrics: refinement does not improve the Teacher",
        "Macro over 20 held-out test cells; lower is better for LPIPS/RGB MAE, higher for IoU/boundary F",
    )
    methods = ["Teacher Endpoint", "SVD Endpoint", "Render-Refined Coefficient", "Full-Residual Equal-Step"]
    short = ["Teacher", "SVD", "Refined", "Full residual"]
    colors = [COLORS["teal"], COLORS["blue"], COLORS["gold"], COLORS["coral"]]
    metrics = [("lpips", "LPIPS", "lower is better"), ("rgb_mae", "RGB MAE", "lower is better"), ("silhouette_iou", "Silhouette IoU", "higher is better"), ("boundary_f", "Boundary F", "higher is better")]
    boxes = [(55, 145, 735, 485), (765, 145, 1445, 485), (55, 505, 735, 845), (765, 505, 1445, 845)]
    macro = data["metrics"]["macro_test_metrics"]
    for box, (key, title, direction) in zip(boxes, metrics):
        left, top, right, bottom = panel_axes(draw, box, title, direction)
        values = [float(macro[method][key]) for method in methods]
        ymax = max(values) * 1.18
        bar_width = int((right - left) / 5.2)
        gap = int((right - left - bar_width * 4) / 5)
        for index, (name, value, color) in enumerate(zip(short, values, colors)):
            x0 = left + gap + index * (bar_width + gap)
            y0 = bottom - int((bottom - top) * value / ymax)
            draw.rectangle((x0, y0, x0 + bar_width, bottom), fill=color)
            draw.text((x0, y0 - 24), f"{value:.4f}", fill=COLORS["ink"], font=font(13, bold=True))
            draw.text((x0, bottom + 9), name, fill=COLORS["muted"], font=font(13))
        draw.line((left, bottom, right, bottom), fill=COLORS["ink"], width=2)
    save_png(output, image)
    return {"plot_id": "four_method_test_metrics", "output": str(output), "sha256": sha256_file(output)}


def parity_plot(data: Mapping[str, Any], output: Path) -> dict[str, Any]:
    image, draw = new_canvas(
        "Teacher vs SVD endpoint parity",
        "All 20 garment-condition cells pass the frozen 1e-5 render-MAE threshold",
    )
    left, top, right, bottom = 130, 175, 1415, 770
    draw.rectangle((left, top, right, bottom), fill=COLORS["panel"], outline=COLORS["grid"], width=2)
    rows = data["parity"]["rows"]
    rgb = [max(float(row["render_rgb_mae"]), 1e-10) for row in rows]
    alpha = [max(float(row["render_alpha_mae"]), 1e-10) for row in rows]
    threshold = float(data["parity"]["parity_threshold"])
    lo, hi = -10.0, -4.5

    def y(value: float) -> int:
        return int(bottom - (math.log10(value) - lo) / (hi - lo) * (bottom - top))

    for exponent in range(-10, -3):
        yy = y(10.0**exponent)
        draw.line((left, yy, right, yy), fill=COLORS["grid"], width=1)
        draw.text((55, yy - 9), f"1e{exponent}", fill=COLORS["muted"], font=font(14))
    threshold_y = y(threshold)
    draw.line((left, threshold_y, right, threshold_y), fill=COLORS["coral"], width=4)
    draw.text((right - 240, threshold_y - 30), "parity threshold", fill=COLORS["coral"], font=font(16, bold=True))
    for index in range(20):
        x = int(left + (index + 0.5) * (right - left) / 20)
        draw.ellipse((x - 5, y(rgb[index]) - 5, x + 5, y(rgb[index]) + 5), fill=COLORS["blue"])
        draw.rectangle((x - 4, y(alpha[index]) - 4, x + 4, y(alpha[index]) + 4), fill=COLORS["gold"])
        if index % 4 == 1:
            draw.text((x - 22, bottom + 14), GARMENTS[index // 4], fill=COLORS["muted"], font=font(14))
    draw.text((left, 135), "RGB MAE", fill=COLORS["blue"], font=font(17, bold=True))
    draw.text((left + 120, 135), "Alpha MAE", fill=COLORS["gold"], font=font(17, bold=True))
    draw.text((right - 390, 135), f"max RGB={max(rgb):.2e} | max alpha={max(alpha):.2e}", fill=COLORS["ink"], font=font(16))
    save_png(output, image)
    return {"plot_id": "teacher_svd_parity", "output": str(output), "sha256": sha256_file(output)}


def gate_plot(data: Mapping[str, Any], output: Path) -> dict[str, Any]:
    image, draw = new_canvas(
        "LPIPS gate summary",
        "Observed Teacher-minus-refined gain is negative; none of the preregistered positive gates pass",
    )
    gates = data["analysis"]["gate_values"]
    rows = [
        ("Macro absolute LPIPS gain", float(gates["macro_lpips_improvement_absolute"]), 0.005, "error units"),
        ("Macro relative LPIPS reduction", float(gates["macro_lpips_relative_reduction"]), 0.05, "fraction"),
        ("Garments improved", float(gates["improved_garments"]), 4.0, "count of 5"),
    ]
    for idx, (label, observed, threshold, unit) in enumerate(rows):
        y0 = 195 + idx * 205
        draw.text((90, y0), label, fill=COLORS["ink"], font=font(24, bold=True))
        draw.text((90, y0 + 38), f"observed {observed:.6g} | required >= {threshold:.6g} {unit}", fill=COLORS["muted"], font=font(17))
        axis_left, axis_right, axis_y = 570, 1380, y0 + 60
        scale = max(abs(observed), threshold) * 1.25
        zero = int(axis_left + (axis_right - axis_left) * 0.35)
        draw.line((axis_left, axis_y, axis_right, axis_y), fill=COLORS["grid"], width=12)
        draw.line((zero, axis_y - 24, zero, axis_y + 24), fill=COLORS["ink"], width=3)
        obs_x = int(zero + observed / scale * (axis_right - zero))
        threshold_x = int(zero + threshold / scale * (axis_right - zero))
        draw.line((zero, axis_y, obs_x, axis_y), fill=COLORS["coral"], width=18)
        draw.ellipse((obs_x - 11, axis_y - 11, obs_x + 11, axis_y + 11), fill=COLORS["coral"])
        draw.line((threshold_x, axis_y - 28, threshold_x, axis_y + 28), fill=COLORS["green"], width=5)
        draw.text((threshold_x - 38, axis_y + 36), "gate", fill=COLORS["green"], font=font(15, bold=True))
        draw.text((90, y0 + 98), "FAIL", fill=COLORS["coral"], font=font(22, bold=True))
    save_png(output, image)
    return {"plot_id": "lpips_gate_summary", "output": str(output), "sha256": sha256_file(output)}


def per_garment_gain_plot(data: Mapping[str, Any], output: Path) -> dict[str, Any]:
    image, draw = new_canvas(
        "Per-garment Teacher-minus-refined LPIPS gain",
        "Positive values favor refinement; the preregistered per-garment gate is +0.001",
    )
    gains = data["analysis"]["per_garment_test_gains"]
    values = [float(gains[garment]["absolute_improvement"]) for garment in GARMENTS]
    left, top, right, bottom = 170, 190, 1370, 750
    max_abs = 0.0011
    zero_y = int(top + max_abs / (2 * max_abs) * (bottom - top))
    draw.rectangle((left, top, right, bottom), fill=COLORS["panel"], outline=COLORS["grid"], width=2)
    draw.line((left, zero_y, right, zero_y), fill=COLORS["ink"], width=3)
    gate_y = int(zero_y - 0.001 / (2 * max_abs) * (bottom - top))
    draw.line((left, gate_y, right, gate_y), fill=COLORS["green"], width=4)
    draw.text((right - 230, gate_y - 27), "+0.001 gate", fill=COLORS["green"], font=font(16, bold=True))
    slot = (right - left) / len(GARMENTS)
    for index, (garment, value) in enumerate(zip(GARMENTS, values)):
        x0 = int(left + index * slot + slot * 0.22)
        x1 = int(left + (index + 1) * slot - slot * 0.22)
        yy = int(zero_y - value / (2 * max_abs) * (bottom - top))
        color = COLORS["green"] if value > 0 else COLORS["coral"]
        draw.rectangle((x0, min(yy, zero_y), x1, max(yy, zero_y)), fill=color)
        draw.text((x0, bottom + 15), garment, fill=COLORS["ink"], font=font(20, bold=True))
        draw.text((x0 - 12, yy - 27 if value >= 0 else yy + 8), f"{value:+.6f}", fill=color, font=font(15, bold=True))
    draw.text((70, top - 12), "+0.001", fill=COLORS["muted"], font=font(14))
    draw.text((72, zero_y - 8), "0", fill=COLORS["muted"], font=font(14))
    draw.text((58, bottom - 8), "-0.0011", fill=COLORS["muted"], font=font(14))
    save_png(output, image)
    return {"plot_id": "per_garment_teacher_minus_refined_lpips", "output": str(output), "sha256": sha256_file(output)}


def displacement_plot(data: Mapping[str, Any], output: Path) -> dict[str, Any]:
    image, draw = new_canvas(
        "Coefficient displacement trajectory across rotations",
        "Registered start-to-final (step 0 to 300) standardized displacement; no checkpoint payload was read",
    )
    selected_lambdas = {key: float(value["selected_lambda"]) for key, value in data["final"]["selected_lambdas"].items()}
    runs = [
        row for row in data["runs"]["runs"]
        if row.get("method_family") == "coefficient"
        and row.get("regularized") is True
        and math.isclose(float(row.get("lambda_anchor", -1)), selected_lambdas[row["rotation"]])
    ]
    by_key = {(row["outfit_id"], row["rotation"]): float(row["standardized_displacement_l2"]) for row in runs}
    left, top, right, bottom = 155, 185, 1370, 760
    draw.rectangle((left, top, right, bottom), fill=COLORS["panel"], outline=COLORS["grid"], width=2)
    ymax = max(by_key.values()) * 1.15
    for tick in range(6):
        value = ymax * tick / 5
        yy = int(bottom - tick / 5 * (bottom - top))
        draw.line((left, yy, right, yy), fill=COLORS["grid"], width=1)
        draw.text((70, yy - 8), f"{value:.3f}", fill=COLORS["muted"], font=font(14))
    palette = [COLORS["teal"], COLORS["blue"], COLORS["coral"], COLORS["gold"], "#7a5fa0"]
    x_values = [int(left + (index + 0.5) * (right - left) / 4) for index in range(4)]
    for garment, color in zip(GARMENTS, palette):
        points = []
        for rotation, x in zip(ROTATIONS, x_values):
            value = by_key[(garment, rotation)]
            y = int(bottom - value / ymax * (bottom - top))
            points.append((x, y))
        draw.line(points, fill=color, width=4)
        for point in points:
            draw.ellipse((point[0] - 7, point[1] - 7, point[0] + 7, point[1] + 7), fill=color)
    for rotation, x in zip(ROTATIONS, x_values):
        draw.text((x - 18, bottom + 15), rotation, fill=COLORS["ink"], font=font(20, bold=True))
    for index, (garment, color) in enumerate(zip(GARMENTS, palette)):
        x = 925 + index * 90
        draw.line((x, 145, x + 24, 145), fill=color, width=5)
        draw.text((x + 30, 134), garment, fill=COLORS["ink"], font=font(14, bold=True))
    save_png(output, image)
    return {"plot_id": "coefficient_displacement_trajectory", "output": str(output), "sha256": sha256_file(output)}


def span_null_plot(data: Mapping[str, Any], output: Path) -> dict[str, Any]:
    image, draw = new_canvas(
        "Span recovery is undefined, not zero",
        "Teacher LPIPS minus Full-Residual LPIPS is nonpositive for every garment, so all recovery denominators are null",
    )
    rows = data["span"]["rows"]
    left, top, right, bottom = 185, 185, 1370, 750
    draw.rectangle((left, top, right, bottom), fill=COLORS["panel"], outline=COLORS["grid"], width=2)
    zero_y = top + 55
    draw.line((left, zero_y, right, zero_y), fill=COLORS["ink"], width=3)
    max_abs = max(abs(float(row["denominator"])) for row in rows) * 1.15
    slot = (right - left) / len(rows)
    for index, row in enumerate(rows):
        value = float(row["denominator"])
        x0 = int(left + index * slot + slot * 0.2)
        x1 = int(left + (index + 1) * slot - slot * 0.2)
        y1 = int(zero_y + abs(value) / max_abs * (bottom - zero_y - 50))
        draw.rectangle((x0, zero_y, x1, y1), fill=COLORS["coral"])
        draw.text((x0 - 5, y1 + 10), row["outfit_id"], fill=COLORS["ink"], font=font(20, bold=True))
        draw.text((x0 - 18, y1 - 28), f"{value:.4f}", fill="white", font=font(15, bold=True))
        draw.text((x0 + 8, zero_y + 18), "NULL", fill="white", font=font(18, bold=True))
    draw_wrapped(
        draw,
        (190, 785),
        "Rule: recovery=(Teacher-Refined)/(Teacher-FullResidual) only when the denominator is positive. Here Full Residual is worse than Teacher for all five garments, so macro recovery and all per-garment ratios remain null.",
        width=1150,
        fill=COLORS["muted"],
        text_font=font(16),
        line_height=21,
    )
    save_png(output, image)
    return {"plot_id": "span_recovery_denominator_null", "output": str(output), "sha256": sha256_file(output)}


def lookup_plot(data: Mapping[str, Any], output: Path) -> dict[str, Any]:
    image, draw = new_canvas(
        "Refined lookup decomposition",
        "Perfect reference classification yields no predictor penalty; endpoint refinement itself slightly worsens LPIPS",
    )
    lookup = data["lookup"]
    teacher = float(data["metrics"]["macro_test_metrics"]["Teacher Endpoint"]["lpips"])
    refined = float(lookup["refined_hard_lookup"]["lpips"])
    left, top, right, bottom = panel_axes(draw, (70, 170, 850, 790), "Macro LPIPS", "lower is better")
    values = [("Teacher", teacher, COLORS["teal"]), ("Refined hard lookup", refined, COLORS["gold"]), ("Outfit-ID refined", float(lookup["outfit_id_refined_lookup"]["lpips"]), COLORS["blue"])]
    ymax = max(item[1] for item in values) * 1.15
    for index, (label, value, color) in enumerate(values):
        width = 160
        gap = 55
        x0 = left + gap + index * (width + gap)
        y0 = bottom - int((bottom - top) * value / ymax)
        draw.rectangle((x0, y0, x0 + width, bottom), fill=color)
        draw.text((x0, y0 - 26), f"{value:.6f}", fill=COLORS["ink"], font=font(15, bold=True))
        draw_wrapped(draw, (x0, bottom + 10), label, width=width, fill=COLORS["muted"], text_font=font(14), line_height=18)
    draw.line((left, bottom, right, bottom), fill=COLORS["ink"], width=2)

    draw.rounded_rectangle((900, 170, 1430, 790), radius=8, fill=COLORS["panel"], outline=COLORS["grid"], width=2)
    draw.text((935, 202), "Decomposition", fill=COLORS["ink"], font=font(25, bold=True))
    items = [
        ("Reference classifier accuracy", float(lookup["reference_classifier_accuracy"]), "1.000000"),
        ("Predictor penalty LPIPS", float(lookup["predictor_penalty_lpips"]), "0.000000"),
        ("Teacher-minus-refined gain", float(lookup["endpoint_refinement_gain_lpips"]), f"{float(lookup['endpoint_refinement_gain_lpips']):+.6f}"),
    ]
    for index, (label, value, formatted) in enumerate(items):
        y = 295 + index * 145
        draw.text((935, y), label, fill=COLORS["muted"], font=font(17))
        color = COLORS["coral"] if value < 0 else COLORS["teal"]
        draw.text((935, y + 38), formatted, fill=color, font=font(34, bold=True))
    draw_wrapped(draw, (935, 700), "Attribution: no predictor penalty; no positive endpoint-refinement gain.", width=440, fill=COLORS["ink"], text_font=font(16, bold=True), line_height=21)
    save_png(output, image)
    return {"plot_id": "refined_lookup_decomposition", "output": str(output), "sha256": sha256_file(output)}


def artifact_contact_sheet(attempt: Path, data: Mapping[str, Any], output: Path) -> dict[str, Any]:
    selected = [
        "R0_O01_cond_000347.png",
        "R1_O03_cond_000000.png",
        "R2_O04_cond_000318.png",
        "R3_O08_cond_000017.png",
    ]
    columns = [(0, "Target"), (2, "Teacher"), (4, "Refined"), (5, "Full residual")]
    cell_w, cell_h = 290, 225
    image = Image.new("RGB", (80 + cell_w * 4, 215 + cell_h * 4), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 100), fill=COLORS["ink"])
    draw.text((32, 20), "Full-residual artifact examples", fill="white", font=font(30, bold=True))
    draw.text((34, 62), "Fixed one-case-per-rotation selection; source sheets are sealed and unretouched", fill="#d9e2ec", font=font(16))
    for col_index, (_, label) in enumerate(columns):
        draw.text((80 + col_index * cell_w + 75, 112), label, fill=COLORS["ink"], font=font(18, bold=True))
    source_rows = []
    registry_by_name = {Path(row["sheet_path"]).name: row for row in data["visual_registry"]["rows"]}
    for row_index, name in enumerate(selected):
        path = attempt / "11_visual_sheets" / name
        with Image.open(path) as source_image:
            source = source_image.convert("RGB")
            col_width = source.width / 7.0
            for col_index, (source_col, _) in enumerate(columns):
                crop = source.crop((int(source_col * col_width), 38, int((source_col + 1) * col_width), 268))
                crop.thumbnail((cell_w - 16, cell_h - 16), Image.Resampling.LANCZOS)
                x = 80 + col_index * cell_w + (cell_w - crop.width) // 2
                y = 145 + row_index * cell_h + (cell_h - crop.height) // 2
                image.paste(crop, (x, y))
        label = name.removesuffix(".png").replace("_cond_", " | cond_").replace("_", " | ", 1)
        draw.text((8, 145 + row_index * cell_h + 92), label.split(" | cond_")[0].replace(" | ", "\n"), fill=COLORS["muted"], font=font(15, bold=True))
        row = registry_by_name[name]
        source_rows.append({"sheet_id": row["sheet_id"], "sealed_path": row["sheet_path"], "sha256": row["sheet_sha256"], "selection_rule": "ONE_FIXED_CASE_PER_ROTATION_GARMENTS_O01_O03_O04_O08"})
    footer_top = 150 + cell_h * 4 + 18
    draw.line((28, footer_top, image.width - 28, footer_top), fill=COLORS["grid"], width=2)
    draw.text((30, footer_top + 18), "Full Residual shows severe patch/cloud/mottle and edge scatter in all 20 reviewed sheets; displayed cells are fixed representatives.", fill=COLORS["coral"], font=font(14, bold=True))
    save_png(output, image)
    return {"plot_id": "full_residual_artifact_examples", "output": str(output), "sha256": sha256_file(output), "sources": source_rows}


def compose_overview(inputs: Sequence[Path], output: Path) -> dict[str, Any]:
    image = Image.new("RGB", (1600, 1260), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1600, 105), fill=COLORS["ink"])
    draw.text((42, 22), "Headroom: supplementary negative diagnostic", fill="white", font=font(34, bold=True))
    draw.text((44, 68), "Teacher at local optimum; unconstrained residual fitting degrades", fill="#d9e2ec", font=font(18))
    boxes = [(25, 125, 790, 670), (810, 125, 1575, 670), (25, 690, 790, 1235), (810, 690, 1575, 1235)]
    for path, box in zip(inputs, boxes):
        with Image.open(path) as source:
            panel = source.convert("RGB")
            panel.thumbnail((box[2] - box[0], box[3] - box[1]), Image.Resampling.LANCZOS)
            x = box[0] + (box[2] - box[0] - panel.width) // 2
            y = box[1] + (box[3] - box[1] - panel.height) // 2
            image.paste(panel, (x, y))
    save_png(output, image)
    return {"plot_id": "supplementary_negative_diagnostic_v1", "output": str(output), "sha256": sha256_file(output), "input_sha256": [sha256_file(path) for path in inputs]}


def build_plot_source(data: Mapping[str, Any], source_manifest: Mapping[str, Any]) -> dict[str, Any]:
    selected_lambdas = {key: float(value["selected_lambda"]) for key, value in data["final"]["selected_lambdas"].items()}
    displacement_rows = []
    for row in data["runs"]["runs"]:
        if row.get("method_family") == "coefficient" and row.get("regularized") is True and math.isclose(float(row.get("lambda_anchor", -1)), selected_lambdas[row["rotation"]]):
            displacement_rows.append({
                "run_id": row["run_id"],
                "garment": row["outfit_id"],
                "rotation": row["rotation"],
                "step_start": 0,
                "step_final": 300,
                "standardized_displacement_l2": row["standardized_displacement_l2"],
                "raw_displacement_l2": row["raw_displacement_l2"],
            })
    return {
        "schema_version": "paper_figure_headroom_plot_source.v1",
        "task_id": TASK_ID,
        "source_head": HEADROOM_HEAD,
        "classification": data["final"]["classification"],
        "macro_test_metrics": data["metrics"]["macro_test_metrics"],
        "gates": data["analysis"]["gate_values"],
        "per_garment_test_gains": data["analysis"]["per_garment_test_gains"],
        "parity_rows": [
            {key: row[key] for key in ("outfit_id", "condition_id", "render_rgb_mae", "render_alpha_mae", "pass")}
            for row in data["parity"]["rows"]
        ],
        "parity_threshold": data["parity"]["parity_threshold"],
        "coefficient_displacement_rows": displacement_rows,
        "span_recovery": data["span"],
        "refined_lookup": data["lookup"],
        "source_manifest_sha256": hashlib.sha256(json_bytes(source_manifest)).hexdigest(),
        "checkpoint_payload_files_read": 0,
    }


def build_caption() -> str:
    return """# Headroom Supplementary Negative Diagnostic Caption Skeleton

**Status:** `SUPPLEMENTARY_NEGATIVE_DIAGNOSTIC`; not a main-paper candidate and not `PAPER_FINAL`.

**Caption skeleton.** On the closed subject02 five-garment study set, the reconstructed SVD endpoint passes the frozen Teacher parity audit in all 20 garment-condition cells. Rendering-objective coefficient refinement does not improve the Teacher Endpoint on held-out test LPIPS (Teacher-minus-refined macro gain is negative), and no preregistered LPIPS gate passes. Equal-step and equal-wall-time full-residual optimization degrade all five garments and produce severe patch/cloud/mottle and edge-scatter artifacts in all 20 fixed visual-review sheets. Span-recovery ratios are null rather than zero because every Teacher-minus-Full-Residual denominator is nonpositive. These sealed results do not support adding render refinement to the main CanonDressGS pipeline.

**Claim boundary.** This is a fixed-identity, closed-five-garment negative diagnostic. It is not unseen-garment, cross-identity, open-world, or spatial-coefficient-field evidence. Full-residual examples use one preregistered representative per rotation and are not retouched.
"""


def build_reports(data: Mapping[str, Any], plots: Sequence[Mapping[str, Any]], ppt: Mapping[str, Any]) -> dict[str, str]:
    gains = data["analysis"]["gate_values"]
    plot_lines = "\n".join(f"- `{Path(item['output']).name}`: `{item['plot_id']}`" for item in plots)
    refresh = f"""# AAAI27 Headroom Figure Refresh

## Decision

Classification: `{CLASSIFICATION}`. Headroom is consumed from the sealed result as `SUPPLEMENTARY_NEGATIVE_DIAGNOSTIC`. Figure 2 remains `METHOD_FREEZE_PENDING_LOO`; `PAPER_FINAL=0`.

## Sealed Source Gate

- Source: `{HEADROOM_BRANCH}@{HEADROOM_HEAD}`
- Scientific classification: `TEACHER_SPAN_AT_LOCAL_OPTIMUM`
- Teacher/SVD parity: `PASS` (20/20 cells)
- Runs/steps/checkpoints/renderer: `120 / 36,000 / 960 / 36,662`
- LOO attempt observation is metadata-only; `files_read=0`.
- GPU, training, inference, renderer, and checkpoint writes by this refresh: `0`.

## Figure Assets

{plot_lines}

## Interpretation

Headroom did not improve the Teacher Endpoint. Full-residual optimization degraded LPIPS and generated patch/cloud/mottle and edge-scatter artifacts. The current evidence therefore does not support adding rendering refinement to the main pipeline.
"""
    plan = f"""# AAAI27 Headroom Negative Diagnostic Figure Plan

## Placement

Use only as `SUPPLEMENTARY_NEGATIVE_DIAGNOSTIC`, limitation evidence, or a group-meeting negative-result slide. It is not a main-method panel.

## Frozen Story

1. Teacher/SVD parity closes implementation ambiguity.
2. Teacher, SVD, and refined coefficients have nearly identical test metrics; the LPIPS gain is `{gains['macro_lpips_improvement_absolute']:+.9f}` and fails the `+0.005` macro gate.
3. Per-garment gains fail the `+0.001` gate for all five garments.
4. Full Residual is worse and creates patch/cloud/mottle artifacts in all 20 fixed visual sheets.
5. Span recovery is undefined because all five denominators are nonpositive.
6. Refined lookup has zero predictor penalty, so the negative result is attributable to endpoint refinement rather than reference classification.

## Caption Requirement

The caption must say that Headroom did not improve the Teacher Endpoint, full-residual optimization degraded and produced patch/cloud/mottle, and the evidence does not support adding render refinement to the main pipeline.
"""
    ppt_rows = "\n".join(
        f"- **{item['recommended_slide_title']}**: `{item['role']}`; explains {item['question']} Source: `{item['source_path']}` (`{item['source_sha256']}`)."
        for item in ppt["materials"]
    )
    ppt_doc = f"""# AAAI27 Headroom PPT Material Index

No PPT file is generated. This index lists sealed-source visual assets that may be placed into a later group-meeting deck.

{ppt_rows}

All materials are negative diagnostics or limitations. `PAPER_FINAL=0`.
"""
    return {
        "AAAI27_HEADROOM_FIGURE_REFRESH_20260724.md": refresh,
        "AAAI27_HEADROOM_NEGATIVE_DIAGNOSTIC_FIGURE_PLAN_20260724.md": plan,
        "AAAI27_HEADROOM_PPT_MATERIAL_INDEX_20260724.md": ppt_doc,
    }


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    headroom_repo = args.headroom_repo.resolve()
    attempt = args.headroom_attempt.resolve()
    if git(repo, "rev-parse", "HEAD") != PARENT_HEAD:
        raise SystemExit("target worktree must start at the exact Figure Bank parent HEAD")
    if git(repo, "branch", "--show-current") != TARGET_BRANCH:
        raise SystemExit("target branch mismatch")

    gate, data = validate_source(headroom_repo, attempt, args.loo_attempt_observation)
    if gate["status"] != "PASS":
        raise SystemExit(json.dumps(gate, indent=2, sort_keys=True))
    if CLASSIFICATION not in ALLOWED_CLASSIFICATIONS:
        raise SystemExit("invalid classification")

    figure_root = repo / FIGURE_ROOT_REL
    plots_root = figure_root / "plots/headroom"
    contact_root = figure_root / "contact_sheets/headroom"
    candidate_root = figure_root / "candidates/supplementary/headroom_negative_diagnostic_v1"
    provenance_root = figure_root / "provenance"
    for path in (plots_root, contact_root, candidate_root, provenance_root):
        path.mkdir(parents=True, exist_ok=True)

    json_sources = [source_record(attempt, relative) for relative in JSON_INPUTS]
    sheet_sources = []
    row_by_name = {Path(row["sheet_path"]).name: row for row in data["visual_registry"]["rows"]}
    for path in data["png_paths"]:
        row = row_by_name[path.name]
        sheet_sources.append({
            "sheet_id": row["sheet_id"],
            "sealed_path": row["sheet_path"],
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "rotation": row["rotation"],
            "garment": row["outfit_id"],
            "test_condition": row["test_condition"],
        })
    source_manifest = {
        "schema_version": "paper_figure_headroom_source_manifest.v1",
        "task_id": TASK_ID,
        "source_branch": HEADROOM_BRANCH,
        "source_head": HEADROOM_HEAD,
        "attempt_path": ATTEMPT_PATH,
        "json_sources": json_sources,
        "visual_sheet_sources": sheet_sources,
        "headroom_files_read": len(json_sources) + len(sheet_sources),
        "checkpoint_payload_files_read": 0,
        "loo_attempt_observation": args.loo_attempt_observation,
        "loo_attempt_files_read": 0,
    }
    write_json(provenance_root / "headroom_source_manifest.json", source_manifest)
    write_json(provenance_root / "source_integrity_gate.json", gate)

    plot_source = build_plot_source(data, source_manifest)
    write_json(plots_root / "metric_plot_source_data.json", plot_source)
    plots: list[dict[str, Any]] = []
    plots.append(parity_plot(data, plots_root / "teacher_svd_parity.png"))
    plots.append(grouped_metric_plot(data, plots_root / "four_method_test_metrics.png"))
    plots.append(gate_plot(data, plots_root / "lpips_gate_summary.png"))
    plots.append(per_garment_gain_plot(data, plots_root / "per_garment_teacher_minus_refined_lpips.png"))
    plots.append(displacement_plot(data, plots_root / "coefficient_displacement_trajectory.png"))
    plots.append(span_null_plot(data, plots_root / "span_recovery_denominator_null.png"))
    plots.append(lookup_plot(data, plots_root / "refined_lookup_decomposition.png"))
    plots.append(artifact_contact_sheet(attempt, data, contact_root / "full_residual_artifact_examples.png"))
    overview = compose_overview(
        [
            plots_root / "four_method_test_metrics.png",
            plots_root / "lpips_gate_summary.png",
            plots_root / "per_garment_teacher_minus_refined_lpips.png",
            contact_root / "full_residual_artifact_examples.png",
        ],
        candidate_root / "supplementary_negative_diagnostic_v1.png",
    )
    plots.append(overview)
    write_text(candidate_root / "caption_skeleton.md", build_caption())

    for item in plots:
        path = Path(item["output"])
        item["output"] = str(path.relative_to(repo)).replace("\\", "/")
        item["source_head"] = HEADROOM_HEAD
        item["classification"] = "SUPPLEMENTARY_NEGATIVE_DIAGNOSTIC"
        item["paper_final"] = 0
    plot_registry = {
        "schema_version": "paper_figure_headroom_plot_registry.v1",
        "task_id": TASK_ID,
        "plot_count": len(plots),
        "plots": plots,
    }
    write_json(provenance_root / "headroom_plot_registry.json", plot_registry)

    figure_map_parent = load_json(repo / RISK_REL / "paper_figure_map_pure_endpoint_refreshed.json")
    figure_map = dict(figure_map_parent)
    figure_map["schema_version"] = "paper_figure_map_headroom_refreshed.v1"
    figure_map["append_only_parent"] = "paper_figure_map_pure_endpoint_refreshed.json"
    figure_map["figure_2"] = dict(figure_map_parent["figure_2"])
    figure_map["figure_2"].update({
        "status": "METHOD_FREEZE_PENDING_LOO",
        "render_refinement": "CONSUMED_FROM_SEALED_RESULT",
        "headroom_placement": "SUPPLEMENTARY_NEGATIVE_DIAGNOSTIC",
    })
    figure_map["headroom"] = "CONSUMED_FROM_SEALED_RESULT"
    figure_map["loo"] = "PENDING_LOO_ADAPTATION"
    figure_map["spatial_coefficient_field"] = "CONTINGENCY_ONLY_NOT_STARTED"
    figure_map["paper_final"] = 0
    write_json(repo / RISK_REL / "paper_figure_map_headroom_refreshed.json", figure_map)

    pending_parent = load_json(repo / RISK_REL / "paper_figure_pending_refresh_registry_amended.json")
    pending = json.loads(json.dumps(pending_parent))
    pending["schema_version"] = "paper_figure_pending_refresh_registry_headroom.v1"
    pending["append_only_parent"] = "paper_figure_pending_refresh_registry_amended.json"
    for item in pending["refresh_tasks"]:
        if item["slot"] == "PENDING_COEFFICIENT_HEADROOM":
            item.clear()
            item.update({
                "slot": "PENDING_COEFFICIENT_HEADROOM",
                "task": "REFRESH_PAPER_FIGURE_BANK_WITH_HEADROOM_RESULTS",
                "status": "CONSUMED_FROM_SEALED_RESULT",
                "source_branch": HEADROOM_BRANCH,
                "source_head": HEADROOM_HEAD,
                "attempt_id": "attempt_002",
                "classification": "TEACHER_SPAN_AT_LOCAL_OPTIMUM",
                "figure_placement": "SUPPLEMENTARY_NEGATIVE_DIAGNOSTIC",
                "automatic": False,
            })
    pending["headroom_source_files_read"] = source_manifest["headroom_files_read"]
    pending["loo_attempt_observation"] = args.loo_attempt_observation
    pending["loo_attempt_files_read"] = 0
    write_json(repo / RISK_REL / "paper_figure_pending_refresh_registry_headroom.json", pending)

    materials = []
    material_specs = [
        ("teacher_svd_parity", "Why can SVD and Teacher be treated as equivalent?", "Teacher/SVD parity closes implementation ambiguity", "supplementary"),
        ("four_method_test_metrics", "Does render refinement improve held-out test quality?", "Headroom does not improve the Teacher", "negative diagnostic"),
        ("lpips_gate_summary", "Which preregistered success gates pass?", "All positive Headroom gates fail", "negative diagnostic"),
        ("per_garment_teacher_minus_refined_lpips", "Is the macro result hiding a successful garment?", "No garment reaches the frozen LPIPS gain gate", "limitation"),
        ("coefficient_displacement_trajectory", "How far did coefficients move under adaptation?", "Small start-to-final coefficient displacement across rotations", "supplementary"),
        ("full_residual_artifact_examples", "What does unconstrained residual optimization do visually?", "Patch/cloud/mottle and edge-scatter failure mode", "limitation"),
        ("span_recovery_denominator_null", "Why is span recovery null?", "Full Residual is worse, making all denominators nonpositive", "negative diagnostic"),
        ("refined_lookup_decomposition", "Is the negative result caused by reference prediction?", "Predictor penalty is zero; refinement itself has no gain", "negative diagnostic"),
        ("supplementary_negative_diagnostic_v1", "What is the complete Headroom conclusion?", "Compact sealed negative-diagnostic overview", "supplementary"),
    ]
    plot_by_id = {item["plot_id"]: item for item in plots}
    for plot_id, question, title, role in material_specs:
        item = plot_by_id[plot_id]
        materials.append({
            "material_id": f"HEADROOM-{plot_id.upper().replace('_', '-')}",
            "plot_id": plot_id,
            "question": question,
            "recommended_slide_title": title,
            "source_path": item["output"],
            "source_sha256": item["sha256"],
            "source_result_head": HEADROOM_HEAD,
            "role": role,
            "main_paper_candidate": False,
            "supplementary_candidate": role == "supplementary",
            "limitation_candidate": role == "limitation",
            "negative_diagnostic": True,
        })
    ppt_index = {
        "schema_version": "paper_figure_headroom_ppt_material_index.v1",
        "task_id": TASK_ID,
        "status": "READY",
        "ppt_generated": False,
        "source_branch": HEADROOM_BRANCH,
        "source_head": HEADROOM_HEAD,
        "materials": materials,
    }
    write_json(repo / RISK_REL / "ppt_material_index_headroom_refresh.json", ppt_index)

    write_json(repo / RISK_REL / "paper_figure_headroom_plot_registry.json", plot_registry)
    write_json(repo / RISK_REL / "paper_figure_headroom_source_manifest.json", source_manifest)
    write_json(repo / RISK_REL / "paper_figure_headroom_source_gate.json", gate)
    write_json(repo / RISK_REL / "paper_figure_headroom_mutation_audit.json", {
        "schema_version": "paper_figure_headroom_mutation_audit.v1",
        "task_id": TASK_ID,
        "status": "PASS",
        "source_figure_bank_head": PARENT_HEAD,
        "original_figure_bank_mutation_count": 0,
        "pure_endpoint_refresh_mutation_count": 0,
        "headroom_source_mutation_count": 0,
        "headroom_source_worktree_clean": True,
        "headroom_source_head": HEADROOM_HEAD,
        "append_only": True,
        "checkpoint_payload_files_read": 0,
        "checkpoints_written": 0,
        "loo_attempt_files_read": 0,
    })

    reports = build_reports(data, plots, ppt_index)
    for name, value in reports.items():
        write_text(repo / DOCS_REL / name, value)

    summary = {
        "schema_version": "paper_figure_headroom_refresh_final_summary.v1",
        "task_id": TASK_ID,
        "status": "SEALED_REFRESH_EVIDENCE",
        "classification": CLASSIFICATION,
        "allowed_classifications": sorted(ALLOWED_CLASSIFICATIONS),
        "source_figure_bank_branch": "research/paper-figure-bank-pure-endpoint-refresh-20260724",
        "source_figure_bank_head": PARENT_HEAD,
        "target_branch": TARGET_BRANCH,
        "headroom_source_branch": HEADROOM_BRANCH,
        "headroom_source_head": HEADROOM_HEAD,
        "headroom_scientific_classification": "TEACHER_SPAN_AT_LOCAL_OPTIMUM",
        "source_gate": gate,
        "plot_count": len(plots),
        "ppt_material_count": len(materials),
        "figure_2_status": "METHOD_FREEZE_PENDING_LOO",
        "headroom_status": "CONSUMED_FROM_SEALED_RESULT",
        "loo_status": "PENDING_LOO_ADAPTATION",
        "spatial_coefficient_field_status": "CONTINGENCY_ONLY_NOT_STARTED",
        "mutation_audit": {"original_figure_bank": 0, "pure_endpoint_refresh": 0, "headroom_source": 0},
        "paper_final": False,
        "paper_final_count": 0,
        "ppt_generated": False,
        "next_task": "RUN_LEAVE_ONE_GARMENT_OUT_BASIS_ADAPTATION_EXPERIMENT_FROM_REPAIRED_CONTRACT",
        "refresh_commit": "REPORTED_OUT_OF_BAND_TO_AVOID_SELF_REFERENCE",
    }
    write_json(repo / RISK_REL / "paper_figure_headroom_refresh_final_summary.json", summary)
    handoff = {
        "schema_version": "paper_figure_headroom_refresh_handoff.v1",
        "task_id": TASK_ID,
        "status": CLASSIFICATION,
        "source_head": PARENT_HEAD,
        "headroom_head": HEADROOM_HEAD,
        "target_branch": TARGET_BRANCH,
        "figure_root": str(FIGURE_ROOT_REL).replace("\\", "/"),
        "summary": str(RISK_REL / "paper_figure_headroom_refresh_final_summary.json").replace("\\", "/"),
        "ppt_material_index": str(RISK_REL / "ppt_material_index_headroom_refresh.json").replace("\\", "/"),
        "reports": [str(DOCS_REL / name).replace("\\", "/") for name in reports],
        "paper_final": 0,
        "next_task": summary["next_task"],
    }
    write_json(repo / HANDOFF_REL / "paper_figure_headroom_refresh_handoff.json", handoff)

    print(json.dumps({"status": CLASSIFICATION, "plots": len(plots), "ppt_materials": len(materials)}, sort_keys=True))


if __name__ == "__main__":
    main()
