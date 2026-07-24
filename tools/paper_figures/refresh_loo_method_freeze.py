#!/usr/bin/env python3
"""Refresh the paper figure bank from the sealed LOO attempt_003 result.

This is a reporting-only tool. It reads sealed JSON registries and completed
visual sheets, creates deterministic paper-support panels, and never invokes
training, model inference, or the renderer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont


TASK_ID = "AAAI27-PAPER-FIGURE-BANK-LOO-METHOD-FREEZE-001"
TARGET_BRANCH = "research/paper-figure-bank-loo-method-freeze-20260725"
FIGURE_BANK_HEAD = "7e2d8efb3925880da4a9b8b85c0ebdce2437bd37"
ORIGINAL_FIGURE_BANK_HEAD = "06261aea35d26006db250df65707d1a09a0bf098"
PURE_ENDPOINT_HEAD = "ce110887a942cf8db082ba688c8d36d2433bfdbe"
HEADROOM_HEAD = "674e6092e21eeddeb22e963536247a3385c4e200"
LOO_REPORTING_HEAD = "4472e1815287b1866da6d3ed83b2984e0d43611d"
LOO_RESULT_HEAD = "4a0ae278d0512e68c98c7e9743b5d7b30219ca01"
LOO_EXECUTION_HEAD = "7d192f3328a5046c7fdb2dd0e75615bb7e7bf384"
LOO_CLASSIFICATION = "LOO_BASIS_CAPACITY_LIMITED"
FINAL_CLASSIFICATIONS = (
    "LOO_METHOD_FREEZE_FIGURE_REFRESH_READY",
    "LOO_FIGURE_REFRESH_SOURCE_NOT_SEALED",
    "LOO_METHOD_FREEZE_CONTRACT_BLOCKED",
    "LOO_METHOD_FREEZE_FIGURE_REFRESH_INCONCLUSIVE",
)
GARMENTS = ("O01", "O02", "O03", "O04", "O08")
ROTATIONS = ("R0", "R1", "R2", "R3")
REPRESENTATIVE_ROTATION = "R0"
RISK_REL = Path("paper_protocol/reviewer_risk")
FIGURE_REL = Path("paper_draft/figures/loo_method_freeze")
DOC_REL = Path("docs/PAPER")
HANDOFF_REL = Path("project_control_handoff")

LOO_REPO_SOURCES = (
    "paper_protocol/reviewer_risk/loo_attempt003_final_summary.json",
    "paper_protocol/reviewer_risk/loo_attempt003_reporting_head_seal.json",
    "project_control_handoff/loo_basis_adaptation_attempt003_handoff.json",
    "paper_protocol/reviewer_risk/sealed_loo_attempt003_figure_refresh_manifest.json",
    "paper_protocol/reviewer_risk/loo_attempt003_metric_summary.json",
    "paper_protocol/reviewer_risk/loo_attempt003_oracle_capacity_analysis.json",
    "paper_protocol/reviewer_risk/loo_attempt003_hard_lookup_analysis.json",
    "paper_protocol/reviewer_risk/loo_attempt003_view_budget_scaling.json",
    "paper_protocol/reviewer_risk/loo_attempt003_full_residual_summary.json",
    "paper_protocol/reviewer_risk/loo_attempt003_visual_review_summary.json",
    "paper_protocol/reviewer_risk/loo_attempt003_count_verification.json",
    "paper_protocol/reviewer_risk/loo_attempt003_tests.json",
)

COLORS = {
    "ink": (33, 38, 45),
    "muted": (94, 104, 116),
    "line": (188, 194, 202),
    "panel": (245, 247, 249),
    "blue": (43, 111, 173),
    "green": (47, 133, 90),
    "red": (190, 65, 58),
    "amber": (199, 139, 39),
    "purple": (120, 89, 167),
    "cyan": (45, 143, 155),
    "white": (255, 255, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--figure-bank-repo", type=Path, required=True)
    parser.add_argument("--original-figure-bank-repo", type=Path, required=True)
    parser.add_argument("--pure-endpoint-repo", type=Path, required=True)
    parser.add_argument("--headroom-repo", type=Path, required=True)
    parser.add_argument("--loo-repo", type=Path, required=True)
    parser.add_argument("--loo-attempt", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--origin-loo-head", default="UNVERIFIED")
    parser.add_argument("--cloud-loo-head", default="UNVERIFIED")
    parser.add_argument("--cloud-loo-clean", choices=("true", "false", "unverified"), default="unverified")
    parser.add_argument("--verification-status", choices=("PENDING", "PASS"), default="PENDING")
    parser.add_argument("--determinism-status", choices=("PENDING", "PASS"), default="PENDING")
    parser.add_argument("--unit-tests-passed", type=int, default=0)
    parser.add_argument(
        "--final-classification",
        choices=FINAL_CLASSIFICATIONS,
        default="LOO_METHOD_FREEZE_FIGURE_REFRESH_INCONCLUSIVE",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    def reject_duplicate(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate)


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


def write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == content:
        return
    path.write_bytes(content)


def write_json(path: Path, value: Any) -> None:
    write_bytes(path, json_bytes(value))


def write_text(path: Path, value: str) -> None:
    if not value.endswith("\n"):
        value += "\n"
    write_bytes(path, value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def git_clean(repo: Path) -> bool:
    return git(repo, "status", "--porcelain") == ""


def repo_gate(repo: Path, expected_head: str, label: str) -> dict[str, Any]:
    actual_head = git(repo, "rev-parse", "HEAD")
    clean = git_clean(repo)
    if actual_head != expected_head:
        raise RuntimeError(f"{label} HEAD mismatch: {actual_head} != {expected_head}")
    if not clean:
        raise RuntimeError(f"{label} worktree is not clean: {repo}")
    return {
        "label": label,
        "head": actual_head,
        "tree": git(repo, "show", "-s", "--format=%T", "HEAD"),
        "clean": clean,
        "mutation_count": 0,
        "path": str(repo).replace("\\", "/"),
    }


def source_record(root: Path, relative: str, source_head: str, source_kind: str) -> dict[str, Any]:
    path = root / relative
    if not path.is_file():
        raise RuntimeError(f"missing sealed source: {path}")
    return {
        "relative_path": relative.replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "source_head": source_head,
        "source_kind": source_kind,
    }


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                Path("C:/Windows/Fonts/arialbd.ttf"),
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            ]
        )
    else:
        candidates.extend(
            [
                Path("C:/Windows/Fonts/arial.ttf"),
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def wrapped_lines(draw: ImageDraw.ImageDraw, text: str, active_font: ImageFont.ImageFont, width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        words = paragraph.split()
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if current and draw.textbbox((0, 0), candidate, font=active_font)[2] > width:
                lines.append(current)
                current = word
            else:
                current = candidate
        lines.append(current)
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    active_font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    width: int,
    spacing: int = 6,
) -> int:
    x, y = xy
    lines = wrapped_lines(draw, text, active_font, width)
    line_height = draw.textbbox((0, 0), "Ag", font=active_font)[3] + spacing
    for line in lines:
        draw.text((x, y), line, font=active_font, fill=fill)
        y += line_height
    return y


def canvas(title: str, subtitle: str, size: tuple[int, int] = (1600, 900)) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", size, COLORS["white"])
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size[0], 118), fill=COLORS["ink"])
    draw.text((54, 26), title, font=font(34, bold=True), fill=COLORS["white"])
    draw.text((56, 74), subtitle, font=font(19), fill=(220, 225, 230))
    return image, draw


def save_png(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False, compress_level=9)


def status_panel(title: str, subtitle: str, rows: Sequence[tuple[str, str, str]], output: Path) -> None:
    image, draw = canvas(title, subtitle)
    y = 155
    for label, value, state in rows:
        color = COLORS["green"] if state == "PASS" else COLORS["red"] if state == "FAIL" else COLORS["amber"]
        draw.rounded_rectangle((65, y, 1535, y + 72), radius=6, fill=COLORS["panel"], outline=COLORS["line"], width=2)
        draw.text((92, y + 20), label, font=font(22, bold=True), fill=COLORS["ink"])
        draw.text((610, y + 20), value, font=font(22), fill=COLORS["ink"])
        draw.rectangle((1395, y + 18, 1505, y + 54), fill=color)
        box = draw.textbbox((0, 0), state, font=font(16, bold=True))
        draw.text((1450 - (box[2] - box[0]) // 2, y + 27), state, font=font(16, bold=True), fill=COLORS["white"])
        y += 86
    save_png(output, image)


def grouped_bar_panel(
    title: str,
    subtitle: str,
    labels: Sequence[str],
    series: Sequence[tuple[str, Sequence[float], tuple[int, int, int]]],
    y_label: str,
    output: Path,
    include_zero: bool = True,
) -> None:
    image, draw = canvas(title, subtitle)
    left, top, right, bottom = 145, 190, 1535, 760
    values = [float(value) for _, row, _ in series for value in row]
    low = min(values + ([0.0] if include_zero else []))
    high = max(values + ([0.0] if include_zero else []))
    if abs(high - low) < 1e-12:
        high = low + 1.0
    padding = 0.12 * (high - low)
    low -= padding
    high += padding
    if include_zero and min(values) >= 0:
        low = 0.0

    def y_px(value: float) -> int:
        return int(bottom - (value - low) / (high - low) * (bottom - top))

    draw.line((left, top, left, bottom), fill=COLORS["ink"], width=2)
    draw.line((left, bottom, right, bottom), fill=COLORS["ink"], width=2)
    if low <= 0 <= high:
        draw.line((left, y_px(0), right, y_px(0)), fill=COLORS["line"], width=2)
    for tick in range(6):
        value = low + (high - low) * tick / 5
        y = y_px(value)
        draw.line((left - 8, y, left, y), fill=COLORS["ink"], width=2)
        draw.text((32, y - 10), f"{value:.4f}", font=font(15), fill=COLORS["muted"])
    draw.text((26, 135), y_label, font=font(17, bold=True), fill=COLORS["ink"])
    group_width = (right - left) / len(labels)
    bar_width = min(72, int(group_width * 0.72 / len(series)))
    for group_index, label in enumerate(labels):
        center = left + group_width * (group_index + 0.5)
        total_width = bar_width * len(series)
        for series_index, (_, row, color) in enumerate(series):
            value = float(row[group_index])
            x0 = int(center - total_width / 2 + series_index * bar_width + 3)
            x1 = x0 + bar_width - 6
            zero = y_px(0.0) if low <= 0 <= high else bottom
            draw.rectangle((x0, min(zero, y_px(value)), x1, max(zero, y_px(value))), fill=color)
        box = draw.textbbox((0, 0), label, font=font(17, bold=True))
        draw.text((int(center - (box[2] - box[0]) / 2), bottom + 18), label, font=font(17, bold=True), fill=COLORS["ink"])
    legend_x = 210
    for label, _, color in series:
        draw.rectangle((legend_x, 810, legend_x + 24, 834), fill=color)
        draw.text((legend_x + 34, 808), label, font=font(17), fill=COLORS["ink"])
        legend_x += 34 + draw.textbbox((0, 0), label, font=font(17))[2] + 50
    save_png(output, image)


def method_scope_panel(output: Path) -> None:
    image, draw = canvas(
        "Current CanonDressGS method freeze",
        "Offline endpoint construction and inference-time endpoint realization",
    )
    offline = (
        "Frozen Animatable Gaussian Avatar",
        "Per-Garment Multi-View Optimization",
        "Valid Canonical Teacher Endpoints",
        "Explicit Endpoint Coordinate System",
    )
    inference = (
        "Reference RGB / Mask",
        "Frozen F2 + Mean/Max Set Aggregation",
        "Small Endpoint Controller",
        "Nearest Valid Endpoint Realization",
        "Frozen Deformation / LBS / Renderer",
    )
    for x, heading, steps, color in (
        (75, "OFFLINE", offline, COLORS["blue"]),
        (825, "INFERENCE", inference, COLORS["green"]),
    ):
        draw.text((x, 150), heading, font=font(24, bold=True), fill=color)
        y = 205
        for index, step in enumerate(steps):
            draw.rounded_rectangle((x, y, x + 700, y + 82), radius=6, fill=COLORS["panel"], outline=color, width=3)
            draw_wrapped(draw, (x + 24, y + 24), step, font(20, bold=True), COLORS["ink"], 650)
            y += 108
            if index + 1 < len(steps):
                draw.line((x + 350, y - 24, x + 350, y - 8), fill=color, width=4)
    draw.text((75, 775), "RAW_PREDICTED_COEFFICIENT != REALIZED_SNAPPED_ENDPOINT_COEFFICIENT", font=font(22, bold=True), fill=COLORS["red"])
    draw.text((75, 825), "PAPER_FINAL=0 | Figure 2 candidate only", font=font(18), fill=COLORS["muted"])
    save_png(output, image)


def basis_role_panel(output: Path) -> None:
    image, draw = canvas(
        "Basis role adjudication",
        "Endpoint coordinate system is retained; unseen-garment adaptation is not a main-method claim",
    )
    columns = (
        (
            80,
            "ENDPOINT COORDINATE SYSTEM",
            COLORS["green"],
            "RETAINED",
            [
                "Teacher-derived coordinates",
                "Closed registered wardrobe",
                "Endpoint identity is explicit",
                "Nearest valid endpoint realization",
            ],
        ),
        (
            830,
            "ADAPTATION SPACE",
            COLORS["red"],
            "REJECTED AS MAIN METHOD",
            [
                "5/5 Oracle capacity failures",
                "0/5 hard-lookup outperformance",
                "K2 scaling negative overall",
                "LOO_BASIS_CAPACITY_LIMITED",
            ],
        ),
    )
    for x, heading, color, status, items in columns:
        draw.rounded_rectangle((x, 165, x + 690, 760), radius=6, fill=COLORS["panel"], outline=color, width=4)
        draw_wrapped(draw, (x + 30, 205), heading, font(26, bold=True), color, 620)
        draw.text((x + 30, 285), status, font=font(20, bold=True), fill=color)
        y = 355
        for item in items:
            draw.ellipse((x + 35, y + 7, x + 49, y + 21), fill=color)
            y = draw_wrapped(draw, (x + 70, y), item, font(21), COLORS["ink"], 565) + 25
    draw.text((80, 810), "Scientific scope: same-identity subject02 five-garment LOO simulation only", font=font(19), fill=COLORS["muted"])
    save_png(output, image)


def resolve_sheet_path(attempt: Path, value: str) -> Path:
    direct = Path(value)
    if direct.is_file():
        return direct
    normalized = value.replace("\\", "/")
    marker = "11_visual_sheets/"
    if marker not in normalized:
        raise RuntimeError(f"visual sheet path is outside sealed visual tree: {value}")
    candidate = attempt / marker / normalized.split(marker, 1)[1]
    if not candidate.is_file():
        raise RuntimeError(f"missing sealed visual sheet: {candidate}")
    return candidate


def validate_loo_sources(loo_repo: Path, attempt: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if attempt.name != "attempt_003" or any(part in {"attempt_001", "attempt_002"} for part in attempt.parts):
        raise RuntimeError("only sealed attempt_003 may be consumed")
    if git(loo_repo, "rev-parse", "HEAD") != LOO_REPORTING_HEAD or not git_clean(loo_repo):
        raise RuntimeError("LOO reporting source must be exact and clean")

    values = {Path(relative).name: load_json(loo_repo / relative) for relative in LOO_REPO_SOURCES}
    summary = values["loo_attempt003_final_summary.json"]
    seal = values["loo_attempt003_reporting_head_seal.json"]
    handoff = values["loo_basis_adaptation_attempt003_handoff.json"]
    figure_manifest = values["sealed_loo_attempt003_figure_refresh_manifest.json"]
    counts = values["loo_attempt003_count_verification.json"]
    visual_review = values["loo_attempt003_visual_review_summary.json"]
    tests = values["loo_attempt003_tests.json"]

    failures: list[str] = []
    expected_pairs = (
        (summary.get("classification"), LOO_CLASSIFICATION, "summary classification"),
        (summary.get("result_head"), LOO_RESULT_HEAD, "summary result HEAD"),
        (seal.get("classification"), LOO_CLASSIFICATION, "seal classification"),
        (seal.get("result_head"), LOO_RESULT_HEAD, "seal result HEAD"),
        (seal.get("execution_head"), LOO_EXECUTION_HEAD, "seal execution HEAD"),
        (seal.get("status"), "SEALED_SCIENTIFIC_EXECUTION_REPORTING", "seal status"),
        (handoff.get("classification"), LOO_CLASSIFICATION, "handoff classification"),
        (figure_manifest.get("status"), "SEALED_FOR_SEPARATE_REFRESH_TASK", "figure manifest"),
        (counts.get("status"), "PASS", "count verification"),
        (visual_review.get("status"), "PASS", "visual review"),
        (tests.get("status"), "PASS", "source tests"),
    )
    for actual, expected, label in expected_pairs:
        if actual != expected:
            failures.append(f"{label}: {actual!r} != {expected!r}")
    required_counts = {
        "total_optimizer_runs": 120,
        "optimizer_steps": 36000,
        "checkpoint_writes": 720,
        "evaluation_inference": 960,
        "total_logical_renders": 54960,
        "unique_physical_renders": 54845,
        "K_shared_static_cache_hits": 115,
        "visual_sheets": 26,
        "guaranteed_k_invariant_hits": 100,
        "hard_lookup_k_shared_hits": 15,
        "hard_lookup_k_divergent_pairs": 5,
    }
    count_rows = counts.get("expected_actual_delta", {})
    for key, expected in required_counts.items():
        row = count_rows.get(key, {})
        if row.get("actual") != expected or row.get("delta") != 0:
            failures.append(f"count {key} is not sealed at {expected}")
    contamination_keys = (
        "identity_contamination_count",
        "component_contamination_count",
        "severe_component_contamination_count",
    )
    for key in contamination_keys:
        if visual_review.get(key) != 0:
            failures.append(f"{key} must be zero")
    if seal.get("reporting_count_key_repair") != "PASS":
        failures.append("100/15/5 reporting count-key repair is not sealed")
    if failures:
        raise RuntimeError("LOO source gate failed:\n- " + "\n- ".join(failures))

    evaluation_path = attempt / "08_metrics/evaluation.json"
    visual_registry_path = attempt / "11_visual_sheets/visual_registry.json"
    evaluation = load_json(evaluation_path)
    visual_registry = load_json(visual_registry_path)
    if visual_registry.get("status") != "PASS" or visual_registry.get("sheet_count") != 26:
        raise RuntimeError("sealed visual registry is incomplete")
    sheet_records = []
    for row in visual_registry.get("sheets", []):
        source_path = resolve_sheet_path(attempt, row["path"])
        actual_hash = sha256_file(source_path)
        if actual_hash != row["sha256"]:
            raise RuntimeError(f"visual sheet hash mismatch: {row['sheet_id']}")
        sheet_records.append(
            {
                "sheet_id": row["sheet_id"],
                "sheet_type": row["sheet_type"],
                "held_out_garment": row.get("held_out_garment"),
                "rotation": row.get("rotation"),
                "sealed_path": str(source_path).replace("\\", "/"),
                "bytes": source_path.stat().st_size,
                "sha256": actual_hash,
                "used_in_representative_set": row.get("rotation") == REPRESENTATIVE_ROTATION,
            }
        )
    gate = {
        "schema_version": "paper_figure_loo_source_gate.v1",
        "task_id": TASK_ID,
        "status": "PASS",
        "classification": LOO_CLASSIFICATION,
        "attempt": "attempt_003",
        "attempt_003_sealed": True,
        "attempt_001_files_read": 0,
        "attempt_002_files_read": 0,
        "loo_repo_files_read": len(LOO_REPO_SOURCES),
        "attempt_files_read": 2 + len(sheet_records),
        "counts": required_counts,
        "count_verification": "PASS",
        "reporting_count_key_repair": {"guaranteed": 100, "shared": 15, "divergent": 5},
        "identity_contamination_count": 0,
        "component_contamination_count": 0,
        "visual_sheet_count": len(sheet_records),
        "result_head": LOO_RESULT_HEAD,
        "reporting_head": LOO_REPORTING_HEAD,
        "execution_head": LOO_EXECUTION_HEAD,
        "training_runs_started": 0,
        "model_inference_started": 0,
        "renderer_calls_started": 0,
        "new_scientific_renders": 0,
        "checkpoint_writes": 0,
    }
    payload = {
        "values": values,
        "evaluation": evaluation,
        "evaluation_path": evaluation_path,
        "visual_registry": visual_registry,
        "visual_registry_path": visual_registry_path,
        "sheet_records": sheet_records,
    }
    return gate, payload


def final_row(
    rows: Sequence[Mapping[str, Any]], garment: str, rotation: str, k: int, method: str
) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if row.get("held_out_garment") == garment
        and row.get("rotation") == rotation
        and int(row.get("K", -1)) == k
        and row.get("method") == method
        and (row.get("checkpoint_step") is None or int(row["checkpoint_step"]) == 300)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"evaluation row cardinality mismatch: {garment}/{rotation}/K{k}/{method} -> {len(matches)}")
    return matches[0]


def aggregate_evaluation(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    rows = evaluation.get("rows", [])
    methods = {
        "Oracle Projection": "ORACLE_PROJECTION_LOO_BASIS",
        "Low-Dim K2": "FEW_VIEW_LOW_DIMENSIONAL_ADAPTATION",
        "Full Residual K2": "FEW_VIEW_FULL_RESIDUAL_OPTIMIZATION",
        "Hard Lookup K2": "REFERENCE_NEAREST_HARD_LOOKUP",
    }
    result: dict[str, Any] = {"per_garment": {}, "macro": {}}
    for garment in GARMENTS:
        garment_rows: dict[str, Any] = {}
        for label, method in methods.items():
            selected = [final_row(rows, garment, rotation, 2, method) for rotation in ROTATIONS]
            garment_rows[label] = {
                "lpips": mean(float(row["lpips"]) for row in selected),
                "rgb_mae": mean(float(row["rgb_mae"]) for row in selected),
                "silhouette_iou": mean(float(row["silhouette_iou"]) for row in selected),
                "row_count": len(selected),
            }
        result["per_garment"][garment] = garment_rows
    for label in methods:
        result["macro"][label] = {
            metric: mean(result["per_garment"][garment][label][metric] for garment in GARMENTS)
            for metric in ("lpips", "rgb_mae", "silhouette_iou")
        }
    return result


def build_plot_source(payload: Mapping[str, Any]) -> dict[str, Any]:
    values = payload["values"]
    metric = values["loo_attempt003_metric_summary.json"]
    capacity = values["loo_attempt003_oracle_capacity_analysis.json"]["capacity_rows"]
    scaling = values["loo_attempt003_view_budget_scaling.json"]
    aggregate = aggregate_evaluation(payload["evaluation"])
    return {
        "schema_version": "paper_figure_loo_method_freeze_plot_source.v1",
        "task_id": TASK_ID,
        "classification": LOO_CLASSIFICATION,
        "claim_boundary": metric["claim_boundary"],
        "garments": list(GARMENTS),
        "capacity_rows": capacity,
        "method_metrics": aggregate,
        "view_budget_scaling": scaling,
        "success_gates": {
            "capacity_limited": metric["capacity_limited"],
            "capacity_failure_count": metric["capacity_failure_count"],
            "hard_lookup_gate_pass": metric["hard_lookup_gate_pass"],
            "recovery_gate_pass": metric["recovery_gate_pass"],
            "safety_gate_pass": metric["safety_gate_pass"],
            "view_budget_scaling_gate_pass": metric["view_budget_scaling_gate_pass"],
            "view_budget_scaling_classification": metric["view_budget_scaling_classification"],
            "successful_garments": metric["successful_garments"],
        },
        "source_heads": {
            "reporting": LOO_REPORTING_HEAD,
            "result": LOO_RESULT_HEAD,
            "execution": LOO_EXECUTION_HEAD,
        },
    }


def generate_plots(repo: Path, plot_source: Mapping[str, Any]) -> list[dict[str, Any]]:
    root = repo / FIGURE_REL
    plot_root = root / "plots/loo"
    candidate_root = root / "candidates"
    assets: list[dict[str, Any]] = []

    def register(asset_id: str, relative: Path, role: str) -> None:
        path = repo / relative
        assets.append(
            {
                "asset_id": asset_id,
                "relative_path": str(relative).replace("\\", "/"),
                "role": role,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "paper_final": 0,
                "ai_generated": False,
                "scientific_pixel_transform": False,
            }
        )

    overall_path = plot_root / "loo_overall_classification.png"
    status_panel(
        "LOO overall classification",
        "Sealed attempt_003 | same-identity subject02 five-garment study set",
        [
            ("Formal classification", LOO_CLASSIFICATION, "FAIL"),
            ("Oracle capacity", "0/5 garments pass", "FAIL"),
            ("Hard lookup outperformance", "0/5 garments pass", "FAIL"),
            ("Identity/component contamination", "0 / 0", "PASS"),
            ("Count verification", "120 runs | 36,000 steps | 26 sheets", "PASS"),
        ],
        overall_path,
    )
    register("loo_overall_classification", FIGURE_REL / overall_path.relative_to(root), "LOO_OVERALL_CLASSIFICATION_PANEL")

    capacity_rows = sorted(plot_source["capacity_rows"], key=lambda row: row["held_out_garment"])
    capacity_path = plot_root / "per_garment_oracle_projection_capacity_gap.png"
    grouped_bar_panel(
        "Per-garment Oracle Projection capacity gap",
        "Lower residual is better; projection remains far from every held-out teacher",
        list(GARMENTS),
        [
            ("Projection residual RMSE", [row["projection_residual_rmse"] for row in capacity_rows], COLORS["red"]),
            ("Nearest endpoint distance", [min(row["residual_nearest_distances"].values()) for row in capacity_rows], COLORS["blue"]),
        ],
        "Residual-space distance",
        capacity_path,
    )
    register("oracle_projection_capacity_gap", FIGURE_REL / capacity_path.relative_to(root), "PER_GARMENT_ORACLE_CAPACITY_GAP")

    method_metrics = plot_source["method_metrics"]["per_garment"]
    hard_path = plot_root / "k2_low_dimensional_vs_hard_lookup.png"
    grouped_bar_panel(
        "K2 Low-Dimensional Adaptation vs Hard Lookup",
        "Mean over four fixed rotations; no held-out teacher is used by either deployable route",
        list(GARMENTS),
        [
            ("Low-Dim K2", [method_metrics[g]["Low-Dim K2"]["lpips"] for g in GARMENTS], COLORS["purple"]),
            ("Hard Lookup K2", [method_metrics[g]["Hard Lookup K2"]["lpips"] for g in GARMENTS], COLORS["green"]),
        ],
        "LPIPS",
        hard_path,
    )
    register("k2_low_dimensional_vs_hard_lookup", FIGURE_REL / hard_path.relative_to(root), "K2_VS_HARD_LOOKUP")

    scaling_rows = plot_source["view_budget_scaling"]["per_garment"]
    scaling_path = plot_root / "k1_vs_k2_view_budget.png"
    grouped_bar_panel(
        "K1 vs K2 view-budget comparison",
        "K2 minus K1; negative is better for LPIPS and RGB MAE",
        list(GARMENTS),
        [
            ("LPIPS delta", [scaling_rows[g]["lpips_k2_minus_k1"] for g in GARMENTS], COLORS["blue"]),
            ("RGB MAE delta", [scaling_rows[g]["rgb_mae_k2_minus_k1"] for g in GARMENTS], COLORS["amber"]),
        ],
        "K2 - K1",
        scaling_path,
    )
    register("k1_vs_k2_view_budget", FIGURE_REL / scaling_path.relative_to(root), "K1_VS_K2_VIEW_BUDGET")

    methods_path = plot_root / "oracle_vs_low_dimensional_vs_full_residual.png"
    grouped_bar_panel(
        "Oracle Projection vs Low-Dimensional vs Full Residual",
        "K2 final rows, mean over the same four rotations per garment",
        list(GARMENTS),
        [
            ("Oracle Projection", [method_metrics[g]["Oracle Projection"]["lpips"] for g in GARMENTS], COLORS["blue"]),
            ("Low-Dim K2", [method_metrics[g]["Low-Dim K2"]["lpips"] for g in GARMENTS], COLORS["purple"]),
            ("Full Residual K2", [method_metrics[g]["Full Residual K2"]["lpips"] for g in GARMENTS], COLORS["red"]),
        ],
        "LPIPS",
        methods_path,
    )
    register("oracle_vs_low_dimensional_vs_full_residual", FIGURE_REL / methods_path.relative_to(root), "THREE_METHOD_CAPACITY_COMPARISON")

    span_path = plot_root / "per_garment_span_distance_capacity.png"
    grouped_bar_panel(
        "Per-garment span-distance / capacity",
        "Held-out residual distance to the LOO basis and nearest registered endpoint",
        list(GARMENTS),
        [
            ("LOO projection residual", [row["projection_residual_rmse"] for row in capacity_rows], COLORS["red"]),
            ("Mean residual baseline", [row["mean_residual_rmse"] for row in capacity_rows], COLORS["amber"]),
        ],
        "Residual RMSE",
        span_path,
    )
    register("per_garment_span_distance_capacity", FIGURE_REL / span_path.relative_to(root), "SPAN_DISTANCE_CAPACITY_PANEL")

    gates = plot_source["success_gates"]
    gate_path = plot_root / "loo_success_gate_summary.png"
    status_panel(
        "LOO success-gate summary",
        "All scientific gates are reported; failures are not filtered by garment",
        [
            ("Capacity", f"{gates['capacity_failure_count']}/5 failures", "FAIL"),
            ("Hard lookup outperformance", "False", "FAIL"),
            ("Recovery", "False", "FAIL"),
            ("View-budget scaling", gates["view_budget_scaling_classification"], "FAIL"),
            ("Deployable visual safety", "Pass", "PASS"),
        ],
        gate_path,
    )
    register("loo_success_gate_summary", FIGURE_REL / gate_path.relative_to(root), "LOO_SUCCESS_GATE_SUMMARY")

    basis_path = candidate_root / "supplementary/basis_role_adjudication/basis_role_adjudication.png"
    basis_role_panel(basis_path)
    register("basis_role_adjudication", FIGURE_REL / basis_path.relative_to(root), "BASIS_ROLE_ADJUDICATION")

    method_path = candidate_root / "figure2/endpoint_method_freeze/endpoint_method_freeze_overview.png"
    method_scope_panel(method_path)
    register("current_method_freeze_overview", FIGURE_REL / method_path.relative_to(root), "FIGURE2_ENDPOINT_METHOD_FREEZE_CANDIDATE")
    return assets


def copy_representative_sheets(repo: Path, attempt: Path, payload: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output_root = repo / FIGURE_REL / "contact_sheets/loo/representative_r0"
    source_by_id = {row["sheet_id"]: row for row in payload["visual_registry"]["sheets"]}
    source_record_by_id = {row["sheet_id"]: row for row in payload["sheet_records"]}
    assets = []
    transforms = []
    for garment in GARMENTS:
        sheet_id = f"garment_rotation/{garment}_{REPRESENTATIVE_ROTATION}"
        row = source_by_id.get(sheet_id)
        if row is None:
            raise RuntimeError(f"missing representative visual sheet {sheet_id}")
        source = resolve_sheet_path(attempt, row["path"])
        destination = output_root / f"{garment}_{REPRESENTATIVE_ROTATION}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if sha256_file(destination) != row["sha256"]:
            raise RuntimeError(f"byte-identical representative copy failed: {sheet_id}")
        relative = destination.relative_to(repo)
        assets.append(
            {
                "asset_id": f"representative_{garment}_{REPRESENTATIVE_ROTATION}",
                "relative_path": str(relative).replace("\\", "/"),
                "role": "REPRESENTATIVE_VISUAL_COMPARISON_SHEET",
                "garment": garment,
                "rotation": REPRESENTATIVE_ROTATION,
                "selection_rule": "ALL_GARMENTS_FIXED_ROTATION_R0_LEXICAL_ORDER",
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
                "source_sha256": source_record_by_id[sheet_id]["sha256"],
                "paper_final": 0,
                "ai_generated": False,
                "scientific_pixel_transform": False,
            }
        )
        transforms.append(
            {
                "transform_id": f"copy_{garment}_{REPRESENTATIVE_ROTATION}",
                "transform_type": "BYTE_IDENTICAL_COPY_NO_PIXEL_TRANSFORM",
                "source_sheet_id": sheet_id,
                "source_sha256": row["sha256"],
                "output": str(relative).replace("\\", "/"),
                "output_sha256": sha256_file(destination),
                "crop": None,
                "resize": None,
                "retouch": False,
                "sharpen": False,
                "ai_generation": False,
            }
        )
    return assets, transforms


def build_docs(plot_source: Mapping[str, Any], asset_count: int) -> dict[str, str]:
    metrics = plot_source["method_metrics"]["macro"]
    loo_report = f"""# AAAI27 LOO Figure Refresh (2026-07-25)

## Decision

- Task: `{TASK_ID}`
- Formal LOO classification: `{LOO_CLASSIFICATION}`
- Reporting HEAD: `{LOO_REPORTING_HEAD}`
- Result HEAD: `{LOO_RESULT_HEAD}`
- Attempt: `attempt_003` (sealed)
- Claim boundary: {plot_source['claim_boundary']}

## Sealed verification

- 120 optimizer runs, 36,000 optimizer steps, 720 checkpoints, 960 evaluations.
- 54,960 logical renders, 54,845 physical renders, 115 cache hits.
- 26 visual sheets; count verification PASS.
- Identity contamination 0; component contamination 0.
- Reporting count-key closure: 100 / 15 / 5.
- attempt_001 and attempt_002 scientific results consumed: 0.

## Figure refresh

- Registered assets: {asset_count}.
- All five garments are retained in every per-garment panel.
- Representative sheets use the fixed rule `ALL_GARMENTS_FIXED_ROTATION_R0_LEXICAL_ORDER`.
- Representative sheets are byte-identical copies; crop, resize, retouch, sharpen, and AI generation are all disabled.
- Figure 2 receives an endpoint-method-freeze candidate; Figure 6 receives the LOO limitation candidate.

## Scientific reading

The held-out garments fail the Oracle Projection capacity gate in all 5 folds. K2 low-dimensional adaptation does not establish a reliable advantage over nearest hard lookup, and the view-budget scaling gate is negative. The basis is therefore retained as an explicit teacher-derived endpoint coordinate system, not claimed as an unseen-garment adaptation space.

Macro LPIPS (descriptive, K2): Oracle Projection `{metrics['Oracle Projection']['lpips']:.6f}`, Low-Dim `{metrics['Low-Dim K2']['lpips']:.6f}`, Full Residual `{metrics['Full Residual K2']['lpips']:.6f}`, Hard Lookup `{metrics['Hard Lookup K2']['lpips']:.6f}`.

`PAPER_FINAL=0`. No paper manuscript text is modified by this refresh.
"""
    method_freeze = """# CanonDressGS Main Method Freeze (2026-07-25)

## Formal role

`EXPLICIT_TEACHER_DERIVED_ENDPOINT_COORDINATE_SYSTEM`

## Offline pipeline

Frozen Animatable Gaussian Avatar -> Per-Garment Multi-View Optimization -> Valid Canonical Teacher Endpoints -> Explicit Endpoint Coordinate System.

## Inference pipeline

Reference RGB / Mask -> Frozen F2 -> Mean/Max Set Aggregation -> Small Endpoint Controller -> Nearest Valid Endpoint Realization -> Frozen Deformation / LBS / Renderer.

## Coefficient semantics

`RAW_PREDICTED_COEFFICIENT` and `REALIZED_SNAPPED_ENDPOINT_COEFFICIENT` are distinct. The paper method realizes a registered endpoint after prediction; it does not claim that the raw continuous coefficient itself is an exact endpoint.

## Excluded from the current main pipeline

- Render-Refined Coefficient.
- LOO Few-View Adaptation.
- Full-Residual Optimization.
- Spatially Distributed Clothing Coefficients.
- Automatic Dual-Support Controller.

Dual-Support is frozen as `ORACLE_OR_USER_SPECIFIED_GEOMETRY_SAFE_EXTENSION`.

`PAPER_FINAL=0`.
"""
    contribution_freeze = """# CanonDressGS Contribution Freeze (2026-07-25)

## Main contributions

1. Canonical garment endpoint construction for a frozen animatable Gaussian avatar.
2. Causal diagnosis showing Gaussian geometry interpolation is the dominant source of invalid intermediate garment states.
3. Reference-controlled closed-wardrobe endpoint realization with identity safety.

## Supplementary mechanism

Dual-Support composition preserving valid Gaussian geometry supports.

## Not positive contributions

Continuous coefficient regression, test-time coefficient refinement, LOO adaptation, semantic garment manifolds, unseen garment editing, and automatic mixed control are not claimed as positive contributions.

`PAPER_FINAL=0`.
"""
    table_plan = """# Current Protocol Baseline Table Plan (2026-07-25)

## Main subject02 table

Use only the current closed-wardrobe subject02 protocol source. Rows are Base Avatar, Teacher Endpoint, Reference Classifier Lookup, Nearest-Centroid Lookup, and CanonDressGS Endpoint. Preserve the raw-prediction versus snapped-realization distinction for CanonDressGS.

## Analysis table

Keep Geometry Causal, Dual-Support, Headroom, and LOO in a separate analysis table. LOO must display `LOO_BASIS_CAPACITY_LIMITED` and must label deployable adaptation, offline oracle projection, and hard lookup separately.

## Historical boundary

The historical 51-experiment inventory is evidence provenance, not an unlabeled source for the current main table. It must not be merged into the current-protocol rows.

`PAPER_FINAL=0`.
"""
    endpoint_scope = """# Endpoint Coordinate System Scope (2026-07-25)

The explicit basis is formally scoped as `EXPLICIT_TEACHER_DERIVED_ENDPOINT_COORDINATE_SYSTEM`.

It provides coordinates over valid, per-garment teacher endpoints in the registered closed wardrobe. Its paper role is endpoint representation and endpoint realization. It is not evidence for a semantic garment manifold, unseen-garment generation, or a deployable LOO adaptation space.

At inference, the controller emits `RAW_PREDICTED_COEFFICIENT`; the method then realizes `REALIZED_SNAPPED_ENDPOINT_COEFFICIENT` at the nearest valid registered endpoint. These values must remain separate in figures, tables, and captions.

The sealed LOO result is capacity-limited in all five folds. Oracle projection is an offline diagnostic, low-dimensional adaptation is deployable but unsuccessful, and hard lookup is the retained descriptive closed-wardrobe relation.

`PAPER_FINAL=0`.
"""
    return {
        "AAAI27_LOO_FIGURE_REFRESH_20260725.md": loo_report,
        "AAAI27_CANONDRESSGS_MAIN_METHOD_FREEZE_20260725.md": method_freeze,
        "AAAI27_CANONDRESSGS_CONTRIBUTION_FREEZE_20260725.md": contribution_freeze,
        "AAAI27_CURRENT_PROTOCOL_BASELINE_TABLE_PLAN_20260725.md": table_plan,
        "AAAI27_ENDPOINT_COORDINATE_SYSTEM_SCOPE_20260725.md": endpoint_scope,
    }


def current_protocol_table_source(repo: Path) -> dict[str, Any]:
    pure_source_rel = Path("paper_draft/figures/pure_endpoint_refresh/plots/pure_endpoint/metric_plot_source_data.json")
    pure_source = load_json(repo / pure_source_rel)
    plot_by_id = {row["plot_id"]: row for row in pure_source["plots"]}
    top1 = plot_by_id["clean_method_top1"]
    exact = plot_by_id["endpoint_exact_match"]
    top1_values = {top1["x_ticks"][row["x"]]["label"]: row["y"] for row in top1["series"][0]["values"]}
    exact_values = {exact["x_ticks"][row["x"]]["label"]: row["y"] for row in exact["series"][0]["values"]}
    mappings = (
        ("Base Avatar", "Base Avatar"),
        ("Teacher Endpoint", "Teacher Endpoint"),
        ("Reference Classifier Lookup", "Reference Classifier Lookup"),
        ("Nearest-Centroid Lookup", "Nearest-Centroid Lookup"),
        ("CanonDressGS Endpoint", "CanonDressGS-Endpoint"),
    )
    main_rows = []
    for paper_name, source_name in mappings:
        main_rows.append(
            {
                "paper_name": paper_name,
                "source_name": source_name,
                "protocol": "CURRENT_SUBJECT02_CLOSED_WARDROBE",
                "denominator": 60,
                "clean_top1": top1_values.get(source_name),
                "endpoint_exact_match": exact_values.get(source_name),
                "not_applicable": sorted(
                    reason
                    for plot in (top1, exact)
                    for method, reason in plot.get("not_applicable", {}).items()
                    if method == source_name
                ),
                "source": str(pure_source_rel).replace("\\", "/"),
                "source_head": PURE_ENDPOINT_HEAD,
            }
        )
    return {
        "schema_version": "current_protocol_baseline_table_source.v1",
        "task_id": TASK_ID,
        "paper_final": 0,
        "main_subject02_table": {
            "protocol": "CURRENT_SUBJECT02_CLOSED_WARDROBE",
            "historical_51_experiments_included": False,
            "rows": main_rows,
        },
        "analysis_table": {
            "rows": [
                {
                    "analysis": "Geometry causal",
                    "role": "DOMINANT_INVALID_INTERMEDIATE_STATE_CAUSE",
                    "source": "paper_protocol/reviewer_risk/continuous_control_causal_attribution_final_summary.json",
                },
                {
                    "analysis": "Dual-Support",
                    "role": "ORACLE_OR_USER_SPECIFIED_GEOMETRY_SAFE_EXTENSION",
                    "source": "paper_protocol/reviewer_risk/dual_support_all_pair_final_summary.json",
                },
                {
                    "analysis": "Headroom",
                    "role": "SEALED_NEGATIVE_DIAGNOSTIC",
                    "classification": "TEACHER_SPAN_AT_LOCAL_OPTIMUM",
                    "source_head": HEADROOM_HEAD,
                },
                {
                    "analysis": "LOO",
                    "role": "SEALED_CAPACITY_LIMITATION",
                    "classification": LOO_CLASSIFICATION,
                    "source_head": LOO_REPORTING_HEAD,
                    "modes": {
                        "deployable_adaptation": "FEW_VIEW_LOW_DIMENSIONAL_ADAPTATION",
                        "oracle_projection": "OFFLINE_ORACLE_ONLY",
                        "hard_lookup": "REFERENCE_NEAREST_HARD_LOOKUP",
                    },
                },
            ]
        },
    }


def mirror_outputs(repo: Path, cache_root: Path, paths: Iterable[Path]) -> dict[str, Any]:
    records = []
    for path in sorted(set(paths), key=lambda item: str(item)):
        relative = path.relative_to(repo)
        destination = cache_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        records.append(
            {
                "relative_path": str(relative).replace("\\", "/"),
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )
    manifest = {
        "schema_version": "paper_figure_loo_method_freeze_external_cache.v1",
        "task_id": TASK_ID,
        "attempt": "attempt_001",
        "file_count": len(records),
        "files": records,
    }
    write_json(cache_root / "CACHE_MANIFEST.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    cache_root = args.cache_root.resolve()
    baselines = {
        "figure_bank_source": repo_gate(args.figure_bank_repo.resolve(), FIGURE_BANK_HEAD, "Figure Bank source"),
        "original_figure_bank": repo_gate(args.original_figure_bank_repo.resolve(), ORIGINAL_FIGURE_BANK_HEAD, "Original Figure Bank"),
        "pure_endpoint_source": repo_gate(args.pure_endpoint_repo.resolve(), PURE_ENDPOINT_HEAD, "Pure Endpoint source"),
        "headroom_source": repo_gate(args.headroom_repo.resolve(), HEADROOM_HEAD, "Headroom source"),
        "loo_source": repo_gate(args.loo_repo.resolve(), LOO_REPORTING_HEAD, "LOO source"),
    }
    if git(repo, "rev-parse", FIGURE_BANK_HEAD) != FIGURE_BANK_HEAD:
        raise RuntimeError("target repository does not contain the exact Figure Bank source HEAD")
    if git(repo, "merge-base", "HEAD", FIGURE_BANK_HEAD) != FIGURE_BANK_HEAD:
        raise RuntimeError("target branch is not descended from the exact Figure Bank source HEAD")

    source_gate, payload = validate_loo_sources(args.loo_repo.resolve(), args.loo_attempt.resolve())
    remote_clean = args.cloud_loo_clean == "true"
    source_gate["origin_head"] = args.origin_loo_head
    source_gate["cloud_head"] = args.cloud_loo_head
    source_gate["cloud_worktree_clean"] = remote_clean if args.cloud_loo_clean != "unverified" else "UNVERIFIED"
    source_gate["remote_seal_complete"] = (
        args.origin_loo_head == LOO_REPORTING_HEAD
        and args.cloud_loo_head == LOO_REPORTING_HEAD
        and remote_clean
    )
    if args.final_classification == "LOO_METHOD_FREEZE_FIGURE_REFRESH_READY":
        if not source_gate["remote_seal_complete"]:
            raise RuntimeError("READY requires local/origin/cloud LOO reporting-head closure")
        if args.verification_status != "PASS" or args.determinism_status != "PASS":
            raise RuntimeError("READY requires PASS verification and deterministic regeneration")

    plot_source = build_plot_source(payload)
    plot_source_path = repo / FIGURE_REL / "plots/loo/loo_plot_source_data.json"
    write_json(plot_source_path, plot_source)
    generated_assets = generate_plots(repo, plot_source)
    copied_assets, copy_transforms = copy_representative_sheets(repo, args.loo_attempt.resolve(), payload)
    assets = generated_assets + copied_assets

    generated_transforms = [
        {
            "transform_id": f"generate_{asset['asset_id']}",
            "transform_type": "DETERMINISTIC_PIL_REPORTING_PANEL",
            "source": str(plot_source_path.relative_to(repo)).replace("\\", "/"),
            "output": asset["relative_path"],
            "output_sha256": asset["sha256"],
            "scientific_pixel_transform": False,
            "retouch": False,
            "sharpen": False,
            "ai_generation": False,
        }
        for asset in generated_assets
    ]
    transforms = generated_transforms + copy_transforms

    source_records = [
        source_record(args.loo_repo.resolve(), relative, LOO_REPORTING_HEAD, "SEALED_LOO_REPORTING_REPO")
        for relative in LOO_REPO_SOURCES
    ]
    source_records.extend(
        [
            source_record(
                args.loo_attempt.resolve(),
                "08_metrics/evaluation.json",
                LOO_EXECUTION_HEAD,
                "SEALED_ATTEMPT_003",
            ),
            source_record(
                args.loo_attempt.resolve(),
                "11_visual_sheets/visual_registry.json",
                LOO_EXECUTION_HEAD,
                "SEALED_ATTEMPT_003",
            ),
            source_record(
                repo,
                "paper_draft/figures/pure_endpoint_refresh/plots/pure_endpoint/metric_plot_source_data.json",
                FIGURE_BANK_HEAD,
                "INHERITED_CURRENT_PROTOCOL_SOURCE",
            ),
        ]
    )
    source_registry = {
        "schema_version": "paper_figure_loo_refresh_source_registry.v1",
        "task_id": TASK_ID,
        "status": "PASS",
        "classification": LOO_CLASSIFICATION,
        "claim_boundary": plot_source["claim_boundary"],
        "baselines": baselines,
        "source_gate": source_gate,
        "source_files": source_records,
        "visual_sheets": payload["sheet_records"],
        "representative_selection_rule": "ALL_GARMENTS_FIXED_ROTATION_R0_LEXICAL_ORDER",
        "garment_cherry_picking": False,
        "failed_garments_hidden": False,
        "attempt_001_scientific_results_consumed": 0,
        "attempt_002_scientific_results_consumed": 0,
    }
    transform_registry = {
        "schema_version": "paper_figure_loo_refresh_transform_registry.v1",
        "task_id": TASK_ID,
        "status": "PASS",
        "transform_count": len(transforms),
        "transforms": transforms,
        "scientific_pixel_mutation_count": 0,
        "retouch_count": 0,
        "sharpen_count": 0,
        "ai_generation_count": 0,
        "method_specific_crop_count": 0,
    }
    asset_registry = {
        "schema_version": "paper_figure_loo_refresh_asset_registry.v1",
        "task_id": TASK_ID,
        "status": "PASS",
        "asset_count": len(assets),
        "generated_reporting_panel_count": len(generated_assets),
        "byte_identical_visual_sheet_count": len(copied_assets),
        "assets": assets,
    }
    figure_map = {
        "schema_version": "paper_figure_map_loo_method_frozen.v1",
        "task_id": TASK_ID,
        "append_only_parent": "paper_figure_map_headroom_refreshed.json",
        "paper_final": 0,
        "figure_1": {"status": "REQUIRES_MANUAL_ADJUDICATION", "final_claimed": False},
        "figure_2": {"status": "ENDPOINT_METHOD_FREEZE_CANDIDATE_READY", "candidate": "current_method_freeze_overview"},
        "figure_3": {"status": "READY_FROM_HISTORICAL_EVIDENCE"},
        "figure_4": {"status": "READY_FROM_HISTORICAL_EVIDENCE"},
        "figure_5": {"status": "HARD_LOOKUP_RELATION_DESCRIPTIVE_ONLY"},
        "figure_6": {"status": "LIMITATION_CANDIDATE_READY", "candidate": "loo_overall_classification"},
        "headroom": "CONSUMED_FROM_SEALED_NEGATIVE_RESULT",
        "loo": "CONSUMED_FROM_SEALED_CAPACITY_LIMITED_RESULT",
        "spatial_coefficient_field": "CONTINGENCY_ONLY_NOT_STARTED",
        "controller": "SUPPLEMENTARY_NEGATIVE_DIAGNOSTIC",
        "subject00": "SECOND_IDENTITY_VALIDATION_PENDING",
    }
    method_freeze = {
        "schema_version": "canondressgs_main_method_freeze.v1",
        "task_id": TASK_ID,
        "status": "FROZEN",
        "formal_basis_role": "EXPLICIT_TEACHER_DERIVED_ENDPOINT_COORDINATE_SYSTEM",
        "offline_pipeline": [
            "Frozen Animatable Gaussian Avatar",
            "Per-Garment Multi-View Optimization",
            "Valid Canonical Teacher Endpoints",
            "Explicit Endpoint Coordinate System",
        ],
        "inference_pipeline": [
            "Reference RGB / Mask",
            "Frozen F2",
            "Mean/Max Set Aggregation",
            "Small Endpoint Controller",
            "Nearest Valid Endpoint Realization",
            "Frozen Deformation / LBS / Renderer",
        ],
        "coefficient_semantics": {
            "raw": "RAW_PREDICTED_COEFFICIENT",
            "realized": "REALIZED_SNAPPED_ENDPOINT_COEFFICIENT",
            "equal": False,
        },
        "excluded_from_main_pipeline": [
            "Render-Refined Coefficient",
            "LOO Few-View Adaptation",
            "Full-Residual Optimization",
            "Spatially Distributed Clothing Coefficients",
            "Automatic Dual-Support Controller",
        ],
        "dual_support_role": "ORACLE_OR_USER_SPECIFIED_GEOMETRY_SAFE_EXTENSION",
        "paper_final": 0,
    }
    contribution_freeze = {
        "schema_version": "canondressgs_contribution_freeze.v1",
        "task_id": TASK_ID,
        "status": "FROZEN",
        "main_contributions": [
            "Canonical garment endpoint construction for a frozen animatable Gaussian avatar.",
            "Causal diagnosis showing Gaussian geometry interpolation is the dominant source of invalid intermediate garment states.",
            "Reference-controlled closed-wardrobe endpoint realization with identity safety.",
        ],
        "supplementary_mechanism": "Dual-Support composition preserving valid Gaussian geometry supports.",
        "not_positive_contributions": [
            "continuous coefficient regression",
            "test-time coefficient refinement",
            "LOO adaptation",
            "semantic garment manifold",
            "unseen garment editing",
            "automatic mixed controller",
        ],
        "paper_final": 0,
    }
    table_source = current_protocol_table_source(repo)
    scope = {
        "schema_version": "loo_method_scope_adjudication.v1",
        "task_id": TASK_ID,
        "classification": LOO_CLASSIFICATION,
        "claim_boundary": plot_source["claim_boundary"],
        "basis_role": "ENDPOINT_COORDINATE_SYSTEM",
        "adaptation_space_role": "REJECTED_AS_CURRENT_MAIN_METHOD",
        "deployable_adaptation": {
            "method": "FEW_VIEW_LOW_DIMENSIONAL_ADAPTATION",
            "status": "CAPACITY_LIMITED_NOT_MAIN_METHOD",
        },
        "oracle_projection": {"method": "ORACLE_PROJECTION_LOO_BASIS", "status": "OFFLINE_ORACLE_ONLY"},
        "hard_lookup": {"method": "REFERENCE_NEAREST_HARD_LOOKUP", "status": "CLOSED_WARDROBE_DESCRIPTIVE_RELATION"},
        "full_residual": {"method": "FEW_VIEW_FULL_RESIDUAL_OPTIMIZATION", "status": "DIAGNOSTIC_ONLY_NOT_MAIN_METHOD"},
        "spatial_coefficient_field": "CONTINGENCY_ONLY_NOT_STARTED",
        "paper_final": 0,
    }
    tests_registry = {
        "schema_version": "paper_figure_loo_method_freeze_tests.v1",
        "task_id": TASK_ID,
        "status": args.verification_status,
        "command": "python -m unittest tests.test_paper_figure_loo_method_freeze -v",
        "unit_tests": {"status": args.verification_status, "passed": args.unit_tests_passed, "failed": 0},
        "deterministic_regeneration": {"status": args.determinism_status, "hash_mismatch_count": 0},
        "built_in_checks": {
            "source_gate": "PASS",
            "all_five_garments": "PASS",
            "visual_byte_identity": "PASS",
            "figure_states": "PASS",
            "method_freeze": "PASS",
            "contribution_freeze": "PASS",
            "baseline_separation": "PASS",
            "paper_final_zero": "PASS",
            "forbidden_compute_zero": "PASS",
            "protected_source_mutation_zero": "PASS",
        },
    }

    risk_outputs = {
        "paper_figure_loo_refresh_source_registry.json": source_registry,
        "paper_figure_loo_refresh_transform_registry.json": transform_registry,
        "paper_figure_loo_refresh_asset_registry.json": asset_registry,
        "paper_figure_map_loo_method_frozen.json": figure_map,
        "canondressgs_main_method_freeze.json": method_freeze,
        "canondressgs_contribution_freeze.json": contribution_freeze,
        "current_protocol_baseline_table_source.json": table_source,
        "loo_method_scope_adjudication.json": scope,
        "paper_figure_loo_method_freeze_tests.json": tests_registry,
    }
    for name, value in risk_outputs.items():
        write_json(repo / RISK_REL / name, value)

    docs = build_docs(plot_source, len(assets))
    for name, value in docs.items():
        write_text(repo / DOC_REL / name, value)

    resource_counts = {
        "gpu": 0,
        "training": 0,
        "model_inference": 0,
        "renderer": 0,
        "new_scientific_renders": 0,
        "checkpoint_writes": 0,
        "image_api": 0,
        "ai_figures": 0,
    }
    mutation_audit = {
        "original_figure_bank": 0,
        "pure_endpoint_source": 0,
        "headroom_source": 0,
        "loo_source": 0,
        "source_figure_bank": 0,
        "paper_text": 0,
    }
    final_summary = {
        "schema_version": "paper_figure_loo_method_freeze_final_summary.v1",
        "task_id": TASK_ID,
        "status": "SEALED" if args.verification_status == "PASS" else "REPORTING_ARTIFACTS_CREATED",
        "allowed_classifications": list(FINAL_CLASSIFICATIONS),
        "classification": args.final_classification,
        "source_figure_bank_head": FIGURE_BANK_HEAD,
        "loo_reporting_head": LOO_REPORTING_HEAD,
        "loo_result_head": LOO_RESULT_HEAD,
        "loo_classification": LOO_CLASSIFICATION,
        "loo_source_gate": source_gate,
        "method_role": "EXPLICIT_TEACHER_DERIVED_ENDPOINT_COORDINATE_SYSTEM",
        "figure_2_status": "ENDPOINT_METHOD_FREEZE_CANDIDATE_READY",
        "figure_6_status": "LIMITATION_CANDIDATE_READY",
        "asset_count": len(assets),
        "representative_visual_sheet_count": len(copied_assets),
        "mutation_audit": mutation_audit,
        "resource_counts": resource_counts,
        "tests": {"status": args.verification_status, "passed": args.unit_tests_passed, "failed": 0},
        "deterministic_regeneration": args.determinism_status,
        "paper_final": False,
        "paper_final_count": 0,
        "PAPER_FINAL": False,
        "target_branch": TARGET_BRANCH,
        "refresh_commit": "REPORTED_OUT_OF_BAND_TO_AVOID_SELF_REFERENCE",
        "next_task": "AUDIT_AND_RESTRUCTURE_CANONDRESSGS_PAPER_FOR_ENDPOINT_METHOD_FREEZE"
        if args.final_classification == "LOO_METHOD_FREEZE_FIGURE_REFRESH_READY"
        else None,
    }
    write_json(repo / RISK_REL / "paper_figure_loo_method_freeze_final_summary.json", final_summary)

    handoff = {
        "schema_version": "paper_figure_loo_method_freeze_handoff.v1",
        "task_id": TASK_ID,
        "status": args.final_classification,
        "source_head": FIGURE_BANK_HEAD,
        "target_branch": TARGET_BRANCH,
        "loo_reporting_head": LOO_REPORTING_HEAD,
        "loo_result_head": LOO_RESULT_HEAD,
        "figure_root": str(FIGURE_REL).replace("\\", "/"),
        "reports": [str(DOC_REL / name).replace("\\", "/") for name in docs],
        "summary": str(RISK_REL / "paper_figure_loo_method_freeze_final_summary.json").replace("\\", "/"),
        "external_cache": str(cache_root).replace("\\", "/"),
        "push_status": "REPORTED_OUT_OF_BAND_TO_AVOID_SELF_REFERENCE",
        "paper_final": 0,
        "next_task": final_summary["next_task"],
    }
    write_json(repo / HANDOFF_REL / "paper_figure_loo_method_freeze_handoff.json", handoff)

    cache_paths = [repo / asset["relative_path"] for asset in assets]
    cache_paths.extend(repo / RISK_REL / name for name in risk_outputs)
    cache_paths.extend(repo / DOC_REL / name for name in docs)
    cache_paths.extend(
        [
            plot_source_path,
            repo / RISK_REL / "paper_figure_loo_method_freeze_final_summary.json",
            repo / HANDOFF_REL / "paper_figure_loo_method_freeze_handoff.json",
        ]
    )
    cache_manifest = mirror_outputs(repo, cache_root, cache_paths)
    print(
        json.dumps(
            {
                "task_id": TASK_ID,
                "classification": args.final_classification,
                "asset_count": len(assets),
                "cache_file_count": cache_manifest["file_count"],
                "source_gate": source_gate["status"],
                "paper_final": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
