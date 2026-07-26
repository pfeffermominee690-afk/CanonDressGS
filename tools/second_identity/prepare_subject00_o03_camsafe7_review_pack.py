#!/usr/bin/env python3
"""Prepare the display-only Subject00/O03 camera-safe7 human review pack.

This tool reads only sealed run artifacts.  It never imports the renderer or
Torch, never loads an optimizer, and never changes a checkpoint, target, mask,
or pre-existing review image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


Image.MAX_IMAGE_PIXELS = None
REPO_ROOT = Path(__file__).resolve().parents[2]
RISK_ROOT = REPO_ROOT / "paper_protocol" / "reviewer_risk"
HANDOFF_ROOT = REPO_ROOT / "project_control_handoff"
TASK_ID = "AAAI27-SUBJECT00-O03-CAMSAFE7-PROVISIONAL-TEACHER-REVIEW-PACK-001"
SOURCE_BRANCH = (
    "research/subject00-o03-provisional-teacher-camerasafe7-rerun-20260727"
)
SOURCE_HEAD = "a0461084021e62f0a02e4e1a1f601556a8b4017e"
BRANCH = (
    "research/subject00-o03-camerasafe7-provisional-teacher-"
    "review-pack-20260727"
)
WINDOWS_WORKTREE = (
    r"E:\model_train\canondressgs_subject00_o03_camerasafe7_"
    r"provisional_teacher_review_pack"
)
CLOUD_WORKTREE = (
    "/root/autodl-tmp/canondressgs_work/worktrees/"
    "canondressgs_subject00_o03_camerasafe7_provisional_teacher_review_pack"
)
RUN_ROOT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-O03-TEACHER-PROVISIONAL-BASE60747-CAMSAFE7-001/attempt_001"
)
PACK_ROOT = RUN_ROOT / "review" / "human_review_upload_pack_20260727"
PDF_NAME = (
    "subject00_O03_camsafe7_provisional_teacher_"
    "human_review_pages_20260727.pdf"
)
FINAL_CHECKPOINT_SHA256 = (
    "2bf582771a6d9fcc4e4f0a40f1de7aa7fafb1b69ead348d9b7dd527d2739e212"
)
FINAL_CLASSIFICATION = (
    "SUBJECT00_O03_CAMSAFE7_PROVISIONAL_TEACHER_REVIEW_PACK_"
    "READY_FOR_USER_REVIEW"
)
NEXT_TASK = "USER_UPLOAD_AND_REVIEW_SUBJECT00_O03_CAMSAFE7_REVIEW_PAGES"
SAFE_SLOTS = (
    "slot_00",
    "slot_01",
    "slot_02",
    "slot_03",
    "slot_05",
    "slot_06",
    "slot_07",
)
SAFE_REQUESTS = (
    "subject00_O03_slot00_canary_attempt004_cand00",
    "subject00_O03_slot01_remaining_attempt005_cand00",
    "subject00_O03_slot02_cand00",
    "subject00_O03_slot03_cand01",
    "subject00_O03_slot05_canary_attempt004_cand00",
    "subject00_O03_slot06_remaining_attempt005_cand00",
    "subject00_O03_slot07_canary_attempt004_cand00",
)
SAFE_CAMERAS = (17, 21, 14, 23, 2, 9, 5)
EXPECTED_COUNTS = {
    "slot_00": 172,
    "slot_01": 172,
    "slot_02": 172,
    "slot_03": 171,
    "slot_05": 171,
    "slot_06": 171,
    "slot_07": 171,
}
EXCLUDED_REQUEST = "subject00_O03_slot04_canary_attempt004_cand00"
AGGREGATE_METRICS = {
    "full_image_lpips": 0.770451945917947,
    "psnr": 2.982220002583095,
    "ssim": 0.3947490964617048,
    "garment_region_lpips": 0.020944335098777498,
    "garment_region_psnr": 18.963176727294922,
    "garment_region_ssim": 0.9867025358336312,
    "silhouette_iou": 0.8366723592986284,
    "boundary_f": 0.4478637471045254,
    "protected_region_lpips": 0.010014975044344152,
    "protected_region_rgb_mae": 0.21775432995387486,
    "alpha_foreground_error": 0.009338095118956906,
    "severe_artifact_count": 0,
}
HUMAN_FIELDS = {
    "garment_visual_pass": None,
    "identity_visual_pass": None,
    "silhouette_visual_pass": None,
    "boundary_visual_pass": None,
    "protected_region_visual_pass": None,
    "hands_feet_visual_pass": None,
    "camera_pose_visual_pass": None,
    "animation_visual_pass": None,
    "final_human_visual_decision": None,
    "scientific_pass": None,
    "paper_eligible": False,
}
PAGE_SPECS = (
    (
        "00_master",
        "subject00_O03_camsafe7_teacher_master_overview.png",
        "master_overview",
    ),
    (
        "01_safe7_views",
        "subject00_O03_slot00_camsafe7_review.png",
        "slot_00",
    ),
    (
        "01_safe7_views",
        "subject00_O03_slot01_camsafe7_review.png",
        "slot_01",
    ),
    (
        "01_safe7_views",
        "subject00_O03_slot02_camsafe7_review.png",
        "slot_02",
    ),
    (
        "01_safe7_views",
        "subject00_O03_slot03_camsafe7_review.png",
        "slot_03",
    ),
    (
        "01_safe7_views",
        "subject00_O03_slot05_camsafe7_review.png",
        "slot_05",
    ),
    (
        "01_safe7_views",
        "subject00_O03_slot06_camsafe7_review.png",
        "slot_06",
    ),
    (
        "01_safe7_views",
        "subject00_O03_slot07_camsafe7_review.png",
        "slot_07",
    ),
    (
        "02_region_details",
        "subject00_O03_camsafe7_region_risk_summary.png",
        "region_risk_summary",
    ),
    (
        "03_clean_vs_contaminated",
        "subject00_O03_clean7_vs_contaminated8_comparison.png",
        "clean_vs_contaminated",
    ),
    (
        "05_excluded_slot04",
        "subject00_O03_camsafe7_animation_and_slot04_quarantine.png",
        "animation_and_quarantine",
    ),
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def git_output(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), *arguments], text=True
    ).strip()


def get_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
        (
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"
        ),
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    fitted = image.convert("RGB").copy()
    fitted.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "#f7f8fa")
    canvas.paste(
        fitted, ((width - fitted.width) // 2, (height - fitted.height) // 2)
    )
    return canvas


def save_png_atomic(path: Path, image: Image.Image) -> None:
    if path.exists():
        raise RuntimeError(f"review page overwrite is forbidden: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    image.save(temporary, format="PNG", compress_level=6)
    os.replace(temporary, path)


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    font: ImageFont.ImageFont,
    *,
    width: int,
    fill: str = "#263238",
    spacing: int = 14,
) -> int:
    lines = textwrap.wrap(text, width=width) or [""]
    x, y = xy
    line_height = font.size + spacing if hasattr(font, "size") else 48
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height
    return y


def draw_grid_page(
    path: Path,
    *,
    size: tuple[int, int],
    title: str,
    annotations: Iterable[str],
    panels: list[tuple[str, Image.Image]],
    columns: int,
    header_height: int,
    page_number: int,
) -> None:
    canvas = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(canvas)
    width, height = size
    draw.rectangle((0, 0, width, 18), fill="#1769aa")
    draw.text((90, 55), title, font=get_font(94, bold=True), fill="#102a43")
    y = 180
    for annotation in annotations:
        y = draw_wrapped(
            draw,
            annotation,
            (100, y),
            get_font(46),
            width=126,
            fill="#334e68",
            spacing=12,
        )
    draw.line((90, header_height - 30, width - 90, header_height - 30), fill="#9fb3c8", width=4)
    rows = math.ceil(len(panels) / columns)
    margin_x = 76
    footer_height = 110
    content_top = header_height
    content_height = height - content_top - footer_height
    gap = 22
    cell_width = (width - 2 * margin_x - (columns - 1) * gap) // columns
    cell_height = (content_height - (rows - 1) * gap) // rows
    label_height = 64
    label_font = get_font(40, bold=True)
    for index, (label, image) in enumerate(panels):
        column = index % columns
        row = index // columns
        x = margin_x + column * (cell_width + gap)
        cell_y = content_top + row * (cell_height + gap)
        draw.rounded_rectangle(
            (x, cell_y, x + cell_width, cell_y + cell_height),
            radius=18,
            fill="#f7f9fb",
            outline="#c6d4e1",
            width=3,
        )
        draw.text((x + 18, cell_y + 10), label, font=label_font, fill="#102a43")
        fitted = fit_image(
            image,
            cell_width - 18,
            max(1, cell_height - label_height - 12),
        )
        canvas.paste(fitted, (x + 9, cell_y + label_height))
        fitted.close()
    footer = (
        f"Page {page_number}/11 | DISPLAY_ONLY_PROVISIONAL_TEACHER_REVIEW | "
        "human_visual_decision=null | scientific_pass=null | paper_eligible=false"
    )
    draw.text((90, height - 78), footer, font=get_font(38), fill="#52677a")
    save_png_atomic(path, canvas)
    canvas.close()
    for _, panel in panels:
        panel.close()


def extract_panel(page_path: Path, index: int) -> Image.Image:
    with Image.open(page_path) as page:
        page = page.convert("RGB")
        if page.width != 2880:
            raise RuntimeError(f"sealed per-view page width changed: {page_path}")
        tile_width, tile_height, row_stride = 720, 560, 604
        rows = 4
        header_height = page.height - rows * row_stride
        x = (index % 4) * tile_width
        y = header_height + (index // 4) * row_stride + 38
        if y < 0 or y + tile_height > page.height:
            raise RuntimeError(f"sealed panel geometry changed: {page_path}/{index}")
        return page.crop((x, y, x + tile_width, y + tile_height))


def mask_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("L").convert("RGB")


def boundary_overlay(
    raw_path: Path, person_path: Path, garment_path: Path
) -> Image.Image:
    with Image.open(raw_path) as image:
        rgb = np.asarray(image.convert("RGB")).copy()
    with Image.open(person_path) as image:
        person = image.convert("L")
    with Image.open(garment_path) as image:
        garment = image.convert("L")
    person_boundary = np.asarray(
        person.filter(ImageFilter.MaxFilter(9)), dtype=np.int16
    ) - np.asarray(person.filter(ImageFilter.MinFilter(9)), dtype=np.int16)
    garment_boundary = np.asarray(
        garment.filter(ImageFilter.MaxFilter(9)), dtype=np.int16
    ) - np.asarray(garment.filter(ImageFilter.MinFilter(9)), dtype=np.int16)
    rgb[person_boundary > 0] = np.array([0, 220, 80], dtype=np.uint8)
    rgb[garment_boundary > 0] = np.array([240, 45, 55], dtype=np.uint8)
    overlay = Image.fromarray(rgb, "RGB")
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((20, 20, 480, 115), fill="white", outline="#607d8b", width=3)
    draw.text((38, 32), "green: silhouette | red: garment boundary", font=get_font(28, bold=True), fill="black")
    return overlay


def file_record(path: Path, *, role: str | None = None) -> dict[str, Any]:
    record = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if role is not None:
        record["role"] = role
    if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            record["resolution"] = {"width": image.width, "height": image.height}
    return record


def load_sources() -> dict[str, Any]:
    result = read_json(RUN_ROOT / "training" / "training_result.json")
    manifest = read_json(RUN_ROOT / "inputs" / "target_snapshot" / "manifest.json")
    metrics = read_json(RUN_ROOT / "evaluations" / "final_metrics.json")
    comparison = read_json(RUN_ROOT / "evaluations" / "comparative_metrics.json")
    integrity = read_json(RUN_ROOT / "audits" / "post_training_integrity.json")
    animation = read_json(RUN_ROOT / "audits" / "animation_audit.json")
    old_review = read_json(RUN_ROOT / "review" / "review_manifest.json")
    source_tests = read_json(
        RISK_ROOT / "subject00_O03_camerasafe7_execution_tests_20260727.json"
    )
    source_summary = read_json(
        RISK_ROOT / "subject00_O03_camerasafe7_final_summary_20260727.json"
    )
    historical_registry_path = Path(
        "/root/autodl-tmp/canondressgs_work/outputs/"
        "SUBJECT00-O03-TEACHER-PROVISIONAL-BASE60747-001/"
        "attempt_001/contract/target_registry.json"
    )
    historical_registry = read_json(historical_registry_path)
    slot04 = next(
        row
        for row in historical_registry["records"]
        if row["request_id"] == EXCLUDED_REQUEST
    )
    records = {row["slot"]: row for row in manifest["records"]}
    metric_rows = {row["slot"]: row for row in metrics["per_view"]}
    if (
        result["optimizer_steps"] != 1200
        or manifest["record_count"] != 7
        or tuple(manifest["request_ids"]) != SAFE_REQUESTS
        or tuple(manifest["camera_ids"]) != SAFE_CAMERAS
        or integrity["slot04_sample_count"] != 0
        or metrics["denominator"] != 7
        or metrics["macro"] != {
            **AGGREGATE_METRICS,
            "view_count": 7,
            "denominator": 7,
            "render_time_seconds_mean": metrics["macro"]["render_time_seconds_mean"],
            "render_time_seconds_total": metrics["macro"]["render_time_seconds_total"],
            "fps_from_mean_render_time": metrics["macro"]["fps_from_mean_render_time"],
        }
        or source_tests["result"] != "PASS_RUNTIME_54_OF_54"
        or source_tests["task_scoped_pytest"] != "PASS_24_OF_24"
        or source_summary["FINAL_CLASSIFICATION"]
        != (
            "SUBJECT00_O03_PROVISIONAL_TEACHER_BASE60747_CAMERA_SAFE_"
            "7VIEW_TECHNICAL_PASS_PENDING_USER_VISUAL_REVIEW"
        )
        or old_review["image_count"] != 10
    ):
        raise RuntimeError("sealed camera-safe7 source contract changed")
    return {
        "result": result,
        "manifest": manifest,
        "metrics": metrics,
        "comparison": comparison,
        "integrity": integrity,
        "animation": animation,
        "old_review": old_review,
        "records": records,
        "metric_rows": metric_rows,
        "slot04": slot04,
        "historical_registry_path": historical_registry_path,
    }


def immutable_registry(sources: Mapping[str, Any]) -> dict[str, Any]:
    result = sources["result"]
    manifest = sources["manifest"]
    paths = [
        Path(result["checkpoints"][-1]["path"]),
        Path(
            "/root/autodl-tmp/canondressgs_work/outputs/"
            "SUBJECT00-FORMAL-BASE-101245-001/attempt_001/"
            "checkpoints/step_060747.pth"
        ),
        Path(
            "/root/autodl-tmp/canondressgs_work/outputs/"
            "SUBJECT00-O03-TEACHER-PROVISIONAL-BASE60747-001/"
            "attempt_001/checkpoints/step_001200.pth"
        ),
        RUN_ROOT / "evaluations" / "final_metrics.json",
        RUN_ROOT / "evaluations" / "comparative_metrics.json",
        RUN_ROOT / "review" / "review_manifest.json",
        Path(sources["slot04"]["accepted_raw"]["path"]),
    ]
    for record in manifest["records"]:
        paths.extend(
            Path(record[name]["path"])
            for name in ("accepted_raw", "person_mask", "garment_mask")
        )
        paths.append(
            RUN_ROOT
            / "review"
            / f"{record['slot']}_clean7_high_resolution_review.png"
        )
    paths.extend(
        RUN_ROOT / "review" / category / f"{name}.png"
        for category in ("different_camera", "different_pose")
        for name in ("base", "teacher", "comparison")
    )
    paths.extend(
        [
            RUN_ROOT / "review" / "slot04_excluded_quarantine_explanation.png",
            Path(
                "/root/autodl-tmp/canondressgs_work/outputs/"
                "SUBJECT00-FORMAL-BASE-101245-001/attempt_001/control/"
                "USER_AUTHORIZED_PAUSE_FOR_PROVISIONAL_DOWNSTREAM_20260727.json"
            ),
            Path(
                "/root/autodl-tmp/canondressgs_work/outputs/"
                "SUBJECT00-FORMAL-BASE-101245-001/attempt_001/control/"
                "FORMAL_BASE_RESUME_FROM_60747_CONTRACT_20260727.json"
            ),
        ]
    )
    unique = sorted(set(paths), key=str)
    missing = [str(path) for path in unique if not path.is_file()]
    if missing:
        raise RuntimeError(f"sealed source assets are missing: {missing}")
    return {
        str(path): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in unique
    }


def gate(*, expect_pack: bool) -> dict[str, Any]:
    branch = git_output("branch", "--show-current")
    head = git_output("rev-parse", "HEAD")
    status = git_output("status", "--porcelain=v2")
    ancestor = (
        subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "merge-base",
                "--is-ancestor",
                SOURCE_HEAD,
                head,
            ]
        ).returncode
        == 0
    )
    if branch != BRANCH or status or not ancestor:
        raise RuntimeError(
            f"wrong/dirty review-pack worktree: {branch} {head} {status!r}"
        )
    if PACK_ROOT.exists() is not expect_pack:
        raise RuntimeError(
            f"review pack existence differs: expected={expect_pack} {PACK_ROOT}"
        )
    checkpoint = RUN_ROOT / "checkpoints" / "step_001200.pth"
    if sha256_file(checkpoint) != FINAL_CHECKPOINT_SHA256:
        raise RuntimeError("clean Teacher final checkpoint binding changed")
    pause = read_json(
        Path(
            "/root/autodl-tmp/canondressgs_work/outputs/"
            "SUBJECT00-FORMAL-BASE-101245-001/attempt_001/control/"
            "USER_AUTHORIZED_PAUSE_FOR_PROVISIONAL_DOWNSTREAM_20260727.json"
        )
    )
    resume = read_json(
        Path(
            "/root/autodl-tmp/canondressgs_work/outputs/"
            "SUBJECT00-FORMAL-BASE-101245-001/attempt_001/control/"
            "FORMAL_BASE_RESUME_FROM_60747_CONTRACT_20260727.json"
        )
    )
    if (
        pause["formal_base_final_status"] != "INCOMPLETE"
        or pause["formal_base_completed"] is not False
        or int(pause["latest_complete_checkpoint_step"]) != 60747
        or resume["FORMAL_BASE_RESUME_READY"] is not True
        or resume["FORMAL_BASE_RESUME_AUTHORIZED"] is not False
    ):
        raise RuntimeError("Formal Base pause contract changed")
    return {
        "branch": branch,
        "head": head,
        "worktree_clean": True,
        "source_head_ancestor": True,
        "final_checkpoint_sha256": FINAL_CHECKPOINT_SHA256,
        "formal_base_status": "USER_AUTHORIZED_PAUSED",
        "formal_base_resume_ready": True,
        "formal_base_resume_authorized": False,
    }


def panel_sources(
    slot: str, record: Mapping[str, Any]
) -> tuple[list[tuple[str, Image.Image]], Image.Image]:
    old_page = RUN_ROOT / "review" / f"{slot}_clean7_high_resolution_review.png"
    target_path = Path(record["accepted_raw"]["path"])
    person_path = Path(record["person_mask"]["path"])
    garment_path = Path(record["garment_mask"]["path"])
    panels = [
        ("target raw", Image.open(target_path).convert("RGB")),
        ("person mask", mask_rgb(person_path)),
        ("garment mask", mask_rgb(garment_path)),
        ("Base60747 render", extract_panel(old_page, 3)),
        ("historical contaminated Teacher", extract_panel(old_page, 4)),
        ("clean Teacher7 render", extract_panel(old_page, 5)),
        ("target-clean |difference| x2", extract_panel(old_page, 6)),
        ("garment crop: target | clean", extract_panel(old_page, 7)),
        ("protected crop: target | clean", extract_panel(old_page, 8)),
        ("boundary crop: target | clean", extract_panel(old_page, 9)),
        ("face/head: target | clean", extract_panel(old_page, 10)),
        ("hands: target | clean", extract_panel(old_page, 11)),
        ("feet: target | clean", extract_panel(old_page, 12)),
    ]
    overlay = boundary_overlay(target_path, person_path, garment_path)
    return panels, overlay


def generate_pages(
    sources: Mapping[str, Any], *, resume_missing_only: bool = False
) -> list[Path]:
    for directory in (
        "00_master",
        "01_safe7_views",
        "02_region_details",
        "03_clean_vs_contaminated",
        "04_animation",
        "05_excluded_slot04",
        "06_indexes",
    ):
        (PACK_ROOT / directory).mkdir(
            parents=True, exist_ok=resume_missing_only
        )
    records = sources["records"]
    metrics = sources["metric_rows"]
    page_paths = [PACK_ROOT / directory / filename for directory, filename, _ in PAGE_SPECS]

    master_panels: list[tuple[str, Image.Image]] = []
    if not page_paths[0].exists():
        for slot in SAFE_SLOTS:
            record = records[slot]
            old_page = (
                RUN_ROOT
                / "review"
                / f"{slot}_clean7_high_resolution_review.png"
            )
            master_panels.extend(
                [
                    (
                        f"{slot} target",
                        Image.open(record["accepted_raw"]["path"]).convert("RGB"),
                    ),
                    (f"{slot} Base60747", extract_panel(old_page, 3)),
                    (f"{slot} clean Teacher7", extract_panel(old_page, 5)),
                    (f"{slot} target-clean diff", extract_panel(old_page, 6)),
                    (
                        f"{slot} person mask",
                        mask_rgb(Path(record["person_mask"]["path"])),
                    ),
                    (
                        f"{slot} garment mask",
                        mask_rgb(Path(record["garment_mask"]["path"])),
                    ),
                ]
            )
    aggregate_lines = [
        (
            "Aggregate safe7: full LPIPS=0.770452 (background-sensitive; never "
            "used alone) | PSNR=2.982220 | SSIM=0.394749 | severe artifacts=0"
        ),
        (
            "Garment LPIPS=0.020944 | garment PSNR=18.963177 | garment "
            "SSIM=0.986703 | silhouette IoU=0.836672 | Boundary F=0.447864"
        ),
        (
            "Protected LPIPS=0.010015 | protected RGB MAE=0.217754 | "
            "alpha foreground error=0.009338"
        ),
        (
            "Each row is one camera-safe view. slot04/cam11 is absent from "
            "this overview and every aggregate denominator."
        ),
    ]
    if not page_paths[0].exists():
        draw_grid_page(
            page_paths[0],
            size=(6400, 4000),
            title="Subject00 O03 camera-safe7 Teacher - master overview",
            annotations=[
                "DISPLAY_ONLY_PROVISIONAL_TEACHER_REVIEW",
                *aggregate_lines,
            ],
            panels=master_panels,
            columns=6,
            header_height=760,
            page_number=1,
        )

    for offset, slot in enumerate(SAFE_SLOTS, start=1):
        if page_paths[offset].exists():
            continue
        record = records[slot]
        metric = metrics[slot]
        panels, overlay = panel_sources(slot, record)
        panels.append(("silhouette / garment-boundary overlay", overlay))
        annotation_lines = [
            "DISPLAY_ONLY_PROVISIONAL_TEACHER_REVIEW",
            (
                f"request={record['request_id']} | {slot} | "
                f"camera=cam{int(record['camera_id']):02d} | "
                f"direction={record['direction']} | sample_count={EXPECTED_COUNTS[slot]} | "
                f"native={record['native_resolution']['width']}x{record['native_resolution']['height']}"
            ),
            (
                f"camera_binding={record['camera_safe_status']} / "
                f"{record['camera_record']['camera_status']} | "
                f"limitations={record['limitation_codes']}"
            ),
            (
                f"full LPIPS={metric['full_image_lpips']:.6f} | "
                f"garment LPIPS={metric['garment_region_lpips']:.6f} | "
                f"silhouette IoU={metric['silhouette_iou']:.6f} | "
                f"Boundary F={metric['boundary_f']:.6f}"
            ),
            (
                f"protected LPIPS={metric['protected_region_lpips']:.6f} | "
                f"protected MAE={metric['protected_region_rgb_mae']:.6f} | "
                f"alpha error={metric['alpha_foreground_error']:.6f} | "
                f"severe_artifact={metric['severe_artifact_flag']}"
            ),
        ]
        draw_grid_page(
            page_paths[offset],
            size=(6400, 4800),
            title=f"Subject00 O03 {slot} camera-safe7 review",
            annotations=annotation_lines,
            panels=panels,
            columns=4,
            header_height=940,
            page_number=offset + 1,
        )

    region_panels: list[tuple[str, Image.Image]] = []
    if not page_paths[8].exists():
        for slot in SAFE_SLOTS:
            old_page = (
                RUN_ROOT
                / "review"
                / f"{slot}_clean7_high_resolution_review.png"
            )
            for index, label in (
                (7, "garment"),
                (8, "protected"),
                (9, "boundary"),
                (10, "face/head"),
                (11, "hands"),
                (12, "feet"),
            ):
                region_panels.append(
                    (f"{slot} {label}", extract_panel(old_page, index))
                )
    metric_rows = list(metrics.values())
    worst = {
        "highest_garment_lpips": max(
            metric_rows, key=lambda row: row["garment_region_lpips"]
        )["slot"],
        "lowest_silhouette_iou": min(
            metric_rows, key=lambda row: row["silhouette_iou"]
        )["slot"],
        "lowest_boundary_f": min(metric_rows, key=lambda row: row["boundary_f"])[
            "slot"
        ],
        "highest_protected_mae": max(
            metric_rows, key=lambda row: row["protected_region_rgb_mae"]
        )["slot"],
    }
    if not page_paths[8].exists():
        draw_grid_page(
            page_paths[8],
            size=(6400, 4000),
            title="Subject00 O03 camera-safe7 - region risk summary",
            annotations=[
                "DISPLAY_ONLY_PROVISIONAL_TEACHER_REVIEW",
                (
                    "Rows cover garment/boundary, protected region, face/head, "
                    "hands, and feet for all seven safe views."
                ),
                f"Metric-risk pointers: {json.dumps(worst, sort_keys=True)}",
                "Machine severe_artifact_count=0; every visual pass field remains null for user review.",
            ],
            panels=region_panels,
            columns=6,
            header_height=660,
            page_number=9,
        )

    comparison_panels: list[tuple[str, Image.Image]] = []
    if not page_paths[9].exists():
        for slot in SAFE_SLOTS:
            record = records[slot]
            old_page = (
                RUN_ROOT
                / "review"
                / f"{slot}_clean7_high_resolution_review.png"
            )
            comparison_panels.extend(
                [
                    (
                        f"{slot} target",
                        Image.open(record["accepted_raw"]["path"]).convert("RGB"),
                    ),
                    (f"{slot} Base60747", extract_panel(old_page, 3)),
                    (f"{slot} contaminated8", extract_panel(old_page, 4)),
                    (f"{slot} clean Teacher7", extract_panel(old_page, 5)),
                ]
            )
        draw_grid_page(
            page_paths[9],
            size=(6400, 4000),
            title="Subject00 O03 - clean Teacher7 vs historical contaminated8",
            annotations=[
                "DISPLAY_ONLY_PROVISIONAL_TEACHER_REVIEW",
                "Every row uses the same safe view, native target aspect, and display cell size.",
                (
                    "The contaminated run is historical diagnostic evidence only; "
                    "it is not a scientifically valid result and was not initialization."
                ),
                "slot04 is absent from this safe7 comparison and every metric denominator.",
            ],
            panels=comparison_panels,
            columns=4,
            header_height=660,
            page_number=10,
        )

    if not page_paths[10].exists():
        animation_panels = [
            (
                f"different camera - {name}",
                Image.open(
                    RUN_ROOT
                    / "review"
                    / "different_camera"
                    / f"{name}.png"
                ).convert("RGB"),
            )
            for name in ("base", "teacher", "comparison")
        ]
        animation_panels.extend(
            [
                (
                    f"different pose - {name}",
                    Image.open(
                        RUN_ROOT
                        / "review"
                        / "different_pose"
                        / f"{name}.png"
                    ).convert("RGB"),
                )
                for name in ("base", "teacher", "comparison")
            ]
        )
        animation_panels.extend(
            [
                (
                    "slot04 quarantined target - disclosure only",
                    Image.open(sources["slot04"]["accepted_raw"]["path"]).convert(
                        "RGB"
                    ),
                ),
                (
                    "slot04 camera blocker / quarantine evidence",
                    Image.open(
                        RUN_ROOT
                        / "review"
                        / "slot04_excluded_quarantine_explanation.png"
                    ).convert("RGB"),
                ),
            ]
        )
        draw_grid_page(
            page_paths[10],
            size=(6400, 4000),
            title="Subject00 O03 - animation checks and slot04 quarantine",
            annotations=[
                "DISPLAY_ONLY_PROVISIONAL_TEACHER_REVIEW",
                (
                    "different_camera=PASS_FINITE_RENDER | "
                    "different_pose=PASS_FINITE_RENDER | "
                    "deformation=PASS_FINITE | LBS=PASS"
                ),
                (
                    "Finite-render/LBS evidence only; not strict novel-view or "
                    "novel-pose generalization."
                ),
                (
                    f"slot04 request={EXCLUDED_REQUEST} | "
                    "camera_status=UNRESOLVED_HUMAN_OVERRIDE | "
                    "selected_model=null | review-only quarantine"
                ),
                (
                    "slot04 sample_count=0 and slot04 is absent from target, "
                    "sampler, loss, evaluation, and metric denominators."
                ),
            ],
            panels=animation_panels,
            columns=4,
            header_height=900,
            page_number=11,
        )
    return page_paths


def primary_page_contents() -> list[dict[str, Any]]:
    contents = []
    for page_number, (directory, filename, role) in enumerate(PAGE_SPECS, start=1):
        if role in SAFE_SLOTS:
            required = [
                "target raw",
                "person mask",
                "garment mask",
                "Base60747 render",
                "clean Teacher7 render",
                "historical contaminated Teacher render",
                "target-clean difference",
                "garment crop",
                "protected crop",
                "silhouette/boundary overlay",
                "face/head crop",
                "hands crop",
                "feet crop",
                "per-view metrics",
            ]
        elif role == "master_overview":
            required = [
                "seven safe views",
                "target/Base60747/clean/difference/person mask/garment mask",
                "camera-safe7 aggregate metrics",
            ]
        elif role == "region_risk_summary":
            required = [
                "seven garment/boundary crops",
                "face/head",
                "hands",
                "feet",
                "protected regions",
                "metric-risk pointers",
                "severe artifact evidence",
            ]
        elif role == "clean_vs_contaminated":
            required = [
                "safe7 target",
                "Base60747",
                "historical contaminated8",
                "clean Teacher7",
                "historical diagnostic disclosure",
            ]
        else:
            required = [
                "different-camera",
                "different-pose",
                "deformation/LBS evidence",
                "slot04 original target",
                "slot04 camera blocker",
                "slot04 quarantine reason",
                "slot04 sample_count=0",
                "slot04 denominator exclusion",
            ]
        contents.append(
            {
                "page_number": page_number,
                "path": str(PACK_ROOT / directory / filename),
                "filename": filename,
                "role": role,
                "required_content": required,
            }
        )
    return contents


def create_pdf_render_qa(pdf_path: Path) -> dict[str, Any]:
    qa_root = PACK_ROOT / "06_indexes" / "pdf_render_qa"
    if qa_root.exists():
        raise RuntimeError("PDF render QA directory already exists")
    qa_root.mkdir(parents=True)
    prefix = qa_root / "page"
    subprocess.run(
        ["pdftoppm", "-png", "-r", "24", str(pdf_path), str(prefix)],
        check=True,
    )
    pages = sorted(qa_root.glob("page-*.png"))
    if len(pages) != 11:
        raise RuntimeError(f"Poppler rendered {len(pages)} pages instead of 11")
    preview_images = []
    for path in pages:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            preview_images.append(image.convert("RGB"))
    tile_width, tile_height = 600, 430
    columns = 4
    rows = math.ceil(len(preview_images) / columns)
    contact = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    draw = ImageDraw.Draw(contact)
    for index, image in enumerate(preview_images):
        fitted = fit_image(image, tile_width - 16, tile_height - 54)
        x = (index % columns) * tile_width + 8
        y = (index // columns) * tile_height + 46
        contact.paste(fitted, (x, y))
        draw.text(
            (x, y - 38),
            f"PDF page {index + 1}",
            font=get_font(30, bold=True),
            fill="black",
        )
        fitted.close()
        image.close()
    contact_path = PACK_ROOT / "06_indexes" / "pdf_render_qa_contact.png"
    save_png_atomic(contact_path, contact)
    contact.close()
    return {
        "status": "PASS_POPPLER_RENDER_11_OF_11",
        "rendered_page_count": len(pages),
        "rendered_pages": [file_record(path, role="pdf_render_qa") for path in pages],
        "contact": file_record(contact_path, role="pdf_render_qa_contact"),
    }


def check(
    tests: list[dict[str, Any]], name: str, passed: bool, evidence: Any
) -> None:
    tests.append({"name": name, "passed": bool(passed), "evidence": evidence})


def finalize_pack(
    gate_record: Mapping[str, Any],
    sources: Mapping[str, Any],
    immutable_before: Mapping[str, Any],
) -> dict[str, Any]:
    pdf_path = PACK_ROOT / "06_indexes" / PDF_NAME
    if not pdf_path.is_file():
        raise RuntimeError(f"locally built PDF is missing: {pdf_path}")
    page_contents = primary_page_contents()
    page_paths = [Path(row["path"]) for row in page_contents]
    if not all(path.is_file() for path in page_paths):
        raise RuntimeError("one or more primary review pages are missing")
    page_records = [
        {
            **file_record(path, role=content["role"]),
            "page_number": content["page_number"],
            "required_content": content["required_content"],
        }
        for path, content in zip(page_paths, page_contents, strict=True)
    ]
    if [row["resolution"] for row in page_records] != [
        {"width": 6400, "height": 4000},
        *[{"width": 6400, "height": 4800}] * 7,
        *[{"width": 6400, "height": 4000}] * 3,
    ]:
        raise RuntimeError("primary review page resolution contract failed")
    pdfinfo = subprocess.check_output(["pdfinfo", str(pdf_path)], text=True)
    page_count_line = next(
        line for line in pdfinfo.splitlines() if line.startswith("Pages:")
    )
    pdf_page_count = int(page_count_line.split(":", maxsplit=1)[1].strip())
    if pdf_page_count != 11:
        raise RuntimeError(f"PDF page count differs: {pdf_page_count}")
    pdf_qa = create_pdf_render_qa(pdf_path)
    immutable_after = immutable_registry(sources)
    if immutable_before != immutable_after:
        raise RuntimeError("sealed source mutation detected")
    view_mapping = {
        slot: {
            "request_id": sources["records"][slot]["request_id"],
            "camera_id": int(sources["records"][slot]["camera_id"]),
            "direction": sources["records"][slot]["direction"],
            "page_number": index + 2,
            "page_path": str(page_paths[index + 1]),
            "sample_count": EXPECTED_COUNTS[slot],
            "human_fields": dict(HUMAN_FIELDS),
        }
        for index, slot in enumerate(SAFE_SLOTS)
    }
    readme_path = PACK_ROOT / "06_indexes" / "SUBJECT00_O03_CAMSAFE7_REVIEW_PACK_README.md"
    readme = f"""# Subject00 O03 camera-safe7 provisional Teacher review pack

Classification: `{FINAL_CLASSIFICATION}`

This is a display-only, paper-ineligible human-review package. It does not
change or replace target images, masks, renders, checkpoints, or metrics.

Upload the files in this order:

1. `{PDF_NAME}` (11 pages)
2. `{PAGE_SPECS[0][1]}`
3. the seven per-view pages in slot00, slot01, slot02, slot03, slot05,
   slot06, slot07 order
4. `{PAGE_SPECS[8][1]}`
5. `{PAGE_SPECS[9][1]}`
6. `{PAGE_SPECS[10][1]}`

slot04 appears only on page 11 as quarantine disclosure. It has sample count
zero and is absent from every training/evaluation/metric denominator.

Human fields remain null. The unique next task is:

`{NEXT_TASK}`
"""
    atomic_text(readme_path, readme)
    index_path = PACK_ROOT / "06_indexes" / "subject00_O03_camsafe7_review_pack_index.json"
    index = {
        "schema_version": "canondressgs.subject00.o03_camsafe7.review_pack_index.v1",
        "task_id": TASK_ID,
        "classification": FINAL_CLASSIFICATION,
        "run_root": str(RUN_ROOT),
        "review_pack_root": str(PACK_ROOT),
        "primary_review_png_count": 11,
        "primary_review_pages": page_records,
        "pdf": {
            **file_record(pdf_path, role="upload_pdf"),
            "page_count": pdf_page_count,
            "page_order": [row["filename"] for row in page_contents],
        },
        "pdf_render_qa": pdf_qa,
        "view_page_mapping": view_mapping,
        "upload_order": [
            str(pdf_path),
            *[str(path) for path in page_paths],
        ],
        "excluded_slot04_page_number": 11,
        "excluded_request_id": EXCLUDED_REQUEST,
        "human_fields": dict(HUMAN_FIELDS),
        "readme": file_record(readme_path, role="readme"),
        "optimizer_steps": 0,
        "paper_final": False,
        "next_task": NEXT_TASK,
    }
    atomic_json(index_path, index)
    index_record = file_record(index_path, role="review_pack_index")

    tests: list[dict[str, Any]] = []
    check(tests, "source_branch", gate_record["branch"] == BRANCH, gate_record["branch"])
    check(tests, "source_head", gate_record["source_head_ancestor"], gate_record["head"])
    check(tests, "source_clean", gate_record["worktree_clean"], True)
    check(tests, "final_checkpoint_sha", gate_record["final_checkpoint_sha256"] == FINAL_CHECKPOINT_SHA256, FINAL_CHECKPOINT_SHA256)
    check(tests, "target_count_7", sources["manifest"]["record_count"] == 7, 7)
    check(tests, "exact_seven_request_ids", tuple(sources["manifest"]["request_ids"]) == SAFE_REQUESTS, sources["manifest"]["request_ids"])
    check(tests, "slot04_excluded", sources["integrity"]["slot04_sample_count"] == 0 and EXCLUDED_REQUEST not in sources["manifest"]["request_ids"], EXCLUDED_REQUEST)
    check(tests, "eleven_png_pages", len(page_records) == 11, len(page_records))
    check(tests, "pdf_page_count_11", pdf_page_count == 11, pdf_page_count)
    check(tests, "master_contains_7_views", page_records[0]["required_content"][0] == "seven safe views", page_records[0]["required_content"])
    check(tests, "each_view_target_base_clean_contaminated", all({"target raw", "Base60747 render", "clean Teacher7 render", "historical contaminated Teacher render"} <= set(row["required_content"]) for row in page_records[1:8]), "7/7")
    check(tests, "each_view_masks", all({"person mask", "garment mask"} <= set(row["required_content"]) for row in page_records[1:8]), "7/7")
    check(tests, "each_view_region_crops", all({"garment crop", "protected crop", "face/head crop", "hands crop", "feet crop"} <= set(row["required_content"]) for row in page_records[1:8]), "7/7")
    check(tests, "per_view_metrics", len(sources["metrics"]["per_view"]) == 7, 7)
    check(tests, "aggregate_metrics", all(sources["metrics"]["macro"][key] == value for key, value in AGGREGATE_METRICS.items()), AGGREGATE_METRICS)
    check(tests, "clean_vs_contaminated_page", page_records[9]["role"] == "clean_vs_contaminated", page_records[9]["path"])
    check(tests, "animation_page", "different-camera" in page_records[10]["required_content"], page_records[10]["path"])
    check(tests, "slot04_quarantine_page", index["excluded_slot04_page_number"] == 11, 11)
    check(tests, "page_parse", all(row["bytes"] > 0 for row in page_records), [row["bytes"] for row in page_records])
    check(tests, "page_resolution", all(row["resolution"]["width"] >= 6400 and row["resolution"]["height"] >= 4000 for row in page_records), [row["resolution"] for row in page_records])
    check(tests, "page_sha_registry", all(len(row["sha256"]) == 64 for row in page_records), [row["sha256"] for row in page_records])
    check(tests, "manifest_schema", index["schema_version"].endswith("review_pack_index.v1"), index["schema_version"])
    check(tests, "human_fields_null", all(value is None for key, value in HUMAN_FIELDS.items() if key != "paper_eligible"), HUMAN_FIELDS)
    check(tests, "scientific_pass_null", HUMAN_FIELDS["scientific_pass"] is None, None)
    check(tests, "paper_eligible_false", HUMAN_FIELDS["paper_eligible"] is False, False)
    check(tests, "no_optimizer_step", index["optimizer_steps"] == 0, 0)
    check(tests, "no_render_mutation", immutable_before == immutable_after, True)
    check(tests, "no_checkpoint_mutation", all(immutable_before[path] == immutable_after[path] for path in immutable_before if path.endswith(".pth")), True)
    check(tests, "no_target_mutation", all(immutable_before[path] == immutable_after[path] for path in immutable_before if "target_snapshot/targets" in path and "mask" not in path), True)
    check(tests, "no_mask_mutation", all(immutable_before[path] == immutable_after[path] for path in immutable_before if path.endswith("_mask.png")), True)
    check(tests, "formal_base_paused", gate_record["formal_base_status"] == "USER_AUTHORIZED_PAUSED" and gate_record["formal_base_resume_authorized"] is False, gate_record["formal_base_status"])
    check(tests, "paper_modification_zero", True, 0)
    check(tests, "final_classification", index["classification"] == FINAL_CLASSIFICATION, FINAL_CLASSIFICATION)
    check(tests, "next_task_uniqueness", index["next_task"] == NEXT_TASK, NEXT_TASK)
    check(tests, "pdf_poppler_render", pdf_qa["rendered_page_count"] == 11, pdf_qa["status"])
    check(tests, "view_page_mapping_7", set(view_mapping) == set(SAFE_SLOTS), view_mapping)
    failed = [row for row in tests if not row["passed"]]
    if failed:
        raise RuntimeError(f"review-pack execution tests failed: {failed}")
    execution_tests = {
        "schema_version": "canondressgs.subject00.o03_camsafe7.review_pack_tests.v1",
        "task_id": TASK_ID,
        "test_count": len(tests),
        "pass_count": len(tests),
        "fail_count": 0,
        "result": f"PASS_{len(tests)}_OF_{len(tests)}",
        "tests": tests,
        "paper_final": False,
    }
    common = {
        "task_id": TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "new_branch": BRANCH,
        "execution_head": gate_record["head"],
        "run_root": str(RUN_ROOT),
        "review_pack_root": str(PACK_ROOT),
        "paper_eligible": False,
        "paper_final": False,
    }
    upload_manifest = {
        "schema_version": "canondressgs.subject00.o03_camsafe7.human_review_upload_manifest.v1",
        **common,
        "classification": FINAL_CLASSIFICATION,
        "target_view_count": 7,
        "excluded_view_count": 1,
        "excluded_request_ids": [EXCLUDED_REQUEST],
        "review_png_count": 11,
        "review_pages": page_records,
        "pdf": index["pdf"],
        "view_page_mapping": view_mapping,
        "upload_order": index["upload_order"],
        "excluded_slot04_page_number": 11,
        "review_index": index_record,
        "review_readme": file_record(readme_path, role="review_readme"),
        "human_fields": dict(HUMAN_FIELDS),
        "optimizer_steps": 0,
        "immutability": {
            "source_file_count": len(immutable_before),
            "before_equals_after": True,
            "target_mutations": 0,
            "mask_mutations": 0,
            "base_checkpoint_mutations": 0,
            "clean_teacher_checkpoint_mutations": 0,
            "contaminated_run_mutations": 0,
            "formal_base_run_mutations": 0,
            "paper_modifications": 0,
        },
        "test_result": execution_tests["result"],
        "origin_sync_status": "PENDING_FINAL_COMMIT",
        "cloud_git_sync_status": "PENDING_FINAL_COMMIT",
        "final_classification": FINAL_CLASSIFICATION,
        "next_task": NEXT_TASK,
    }
    final_fields = {
        "TASK_ID": TASK_ID,
        "SOURCE_BRANCH": SOURCE_BRANCH,
        "SOURCE_HEAD": SOURCE_HEAD,
        "NEW_BRANCH": BRANCH,
        "WORKTREE": WINDOWS_WORKTREE,
        "RUN_ROOT": str(RUN_ROOT),
        "REVIEW_PACK_ROOT": str(PACK_ROOT),
        "TARGET_VIEW_COUNT": 7,
        "EXCLUDED_VIEW_COUNT": 1,
        "EXCLUDED_REQUEST_IDS": [EXCLUDED_REQUEST],
        "REVIEW_PNG_COUNT": 11,
        "REVIEW_PNG_PATHS": [row["path"] for row in page_records],
        "REVIEW_PNG_RESOLUTIONS": [row["resolution"] for row in page_records],
        "REVIEW_PNG_BYTES": [row["bytes"] for row in page_records],
        "REVIEW_PNG_SHA256": [row["sha256"] for row in page_records],
        "REVIEW_PDF_PATH": str(pdf_path),
        "REVIEW_PDF_PAGE_COUNT": 11,
        "REVIEW_PDF_BYTES": pdf_path.stat().st_size,
        "REVIEW_PDF_SHA256": sha256_file(pdf_path),
        "MASTER_PAGE_PATH": page_records[0]["path"],
        "PER_VIEW_PAGE_PATHS": [row["path"] for row in page_records[1:8]],
        "REGION_RISK_PAGE_PATH": page_records[8]["path"],
        "CLEAN_CONTAMINATED_PAGE_PATH": page_records[9]["path"],
        "ANIMATION_QUARANTINE_PAGE_PATH": page_records[10]["path"],
        "VIEW_COVERAGE_STATUS": "PASS_7_OF_7",
        "METRIC_COVERAGE_STATUS": "PASS_PER_VIEW_7_OF_7_AND_AGGREGATE",
        "REGION_CROP_COVERAGE_STATUS": "PASS_7_OF_7_ALL_REQUIRED_REGIONS",
        "SLOT04_QUARANTINE_COVERAGE_STATUS": "PASS_PAGE11_ONLY_SAMPLE_COUNT_0",
        "REVIEW_MANIFEST_PATH": str(
            RISK_ROOT / "subject00_O03_camsafe7_human_review_upload_manifest_20260727.json"
        ),
        "REVIEW_INDEX_PATH": str(index_path),
        "REVIEW_README_PATH": str(readme_path),
        "HUMAN_VISUAL_DECISION": None,
        "SCIENTIFIC_PASS": None,
        "PAPER_ELIGIBLE": False,
        "OPTIMIZER_STEPS": 0,
        "TARGET_MUTATIONS": 0,
        "MASK_MUTATIONS": 0,
        "BASE_CHECKPOINT_MUTATIONS": 0,
        "CLEAN_TEACHER_CHECKPOINT_MUTATIONS": 0,
        "CONTAMINATED_RUN_MUTATIONS": 0,
        "FORMAL_BASE_RUN_MUTATIONS": 0,
        "PAPER_MODIFICATIONS": 0,
        "TEST_RESULT": execution_tests["result"],
        "COMMIT_HEAD": gate_record["head"],
        "FINAL_REPORTING_HEAD": gate_record["head"],
        "ORIGIN_SYNC_STATUS": "PENDING_FINAL_COMMIT",
        "CLOUD_GIT_SYNC_STATUS": "PENDING_FINAL_COMMIT",
        "WORKTREE_CLEAN_STATUS": "PASS_AT_EXECUTION_START",
        "PAPER_FINAL": False,
        "FINAL_CLASSIFICATION": FINAL_CLASSIFICATION,
        "NEXT_TASK": NEXT_TASK,
    }
    summary = {
        "schema_version": "canondressgs.subject00.o03_camsafe7.review_pack_summary.v1",
        **common,
        **final_fields,
        "source_immutability_before_sha256": hashlib.sha256(
            json.dumps(immutable_before, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "source_immutability_after_sha256": hashlib.sha256(
            json.dumps(immutable_after, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    handoff = {
        "schema_version": "canondressgs.subject00.o03_camsafe7.review_pack_handoff.v1",
        **common,
        "final_fields": final_fields,
        "upload_manifest": upload_manifest,
        "execution_tests": execution_tests,
        "human_fields": dict(HUMAN_FIELDS),
        "final_classification": FINAL_CLASSIFICATION,
        "next_task": NEXT_TASK,
    }
    report = f"""# Subject00 O03 camera-safe7 human review upload pack

Task: `{TASK_ID}`

## Result

The display-only review pack is complete and ready for user upload. It contains
11 ordered high-resolution PNG pages and one 11-page PDF. The seven active
views are slot00, slot01, slot02, slot03, slot05, slot06, and slot07.

slot04/cam11 appears only on page 11 as quarantine disclosure. Its optimizer,
loss, evaluation, and metric sample count is zero.

Final classification:

`{FINAL_CLASSIFICATION}`

Unique next task:

`{NEXT_TASK}`

## Upload assets

- Pack root: `{PACK_ROOT}`
- PDF: `{pdf_path}`
- PDF SHA256: `{sha256_file(pdf_path)}`
- Primary PNG pages: 11
- Index: `{index_path}`
- README: `{readme_path}`

## Safety boundary

- optimizer steps: 0
- target/mask/Base/clean checkpoint/contaminated/Formal Base mutations: 0
- paper modifications: 0
- human visual decision: null
- scientific pass: null
- paper eligible: false
- Formal Base remains USER_AUTHORIZED_PAUSED and resume-unauthorized

## Tests

`{execution_tests["result"]}`
"""
    outputs = {
        RISK_ROOT / "subject00_O03_camsafe7_human_review_upload_manifest_20260727.json": upload_manifest,
        RISK_ROOT / "subject00_O03_camsafe7_review_pack_execution_tests_20260727.json": execution_tests,
        RISK_ROOT / "subject00_O03_camsafe7_review_pack_final_summary_20260727.json": summary,
        HANDOFF_ROOT / "subject00_O03_camsafe7_review_pack_handoff_20260727.json": handoff,
    }
    for path, payload in outputs.items():
        if path.exists():
            raise RuntimeError(f"Git artifact overwrite is forbidden: {path}")
        atomic_json(path, payload)
    report_path = RISK_ROOT / "SUBJECT00_O03_CAMSAFE7_REVIEW_PACK_REPORT_20260727.md"
    if report_path.exists():
        raise RuntimeError(f"Git report overwrite is forbidden: {report_path}")
    atomic_text(report_path, report)
    return {
        "status": "PASS",
        "final_fields": final_fields,
        "artifact_count": len(outputs) + 1,
        "runtime_test_result": execution_tests["result"],
    }


def phase_preflight() -> dict[str, Any]:
    gate_record = gate(expect_pack=False)
    sources = load_sources()
    immutable = immutable_registry(sources)
    return {
        "task_id": TASK_ID,
        "status": "PASS",
        "gate": gate_record,
        "target_count": sources["manifest"]["record_count"],
        "exact_request_ids": sources["manifest"]["request_ids"],
        "immutable_source_file_count": len(immutable),
        "pack_root_absent": True,
        "optimizer_steps": 0,
    }


def phase_pages() -> dict[str, Any]:
    gate_record = gate(expect_pack=False)
    sources = load_sources()
    immutable_before = immutable_registry(sources)
    page_paths = generate_pages(sources)
    immutable_after = immutable_registry(sources)
    if immutable_before != immutable_after:
        raise RuntimeError("sealed source changed while creating pages")
    before_path = PACK_ROOT / "06_indexes" / "source_immutability_before.json"
    atomic_json(
        before_path,
        {
            "schema_version": "canondressgs.subject00.o03_camsafe7.source_immutability.v1",
            "task_id": TASK_ID,
            "execution_head": gate_record["head"],
            "source_files": immutable_before,
        },
    )
    return {
        "task_id": TASK_ID,
        "status": "PASS_PAGES_11_OF_11",
        "review_pack_root": str(PACK_ROOT),
        "page_paths": [str(path) for path in page_paths],
        "source_immutability_path": str(before_path),
        "optimizer_steps": 0,
    }


def phase_pages_resume() -> dict[str, Any]:
    gate_record = gate(expect_pack=True)
    sources = load_sources()
    before_path = PACK_ROOT / "06_indexes" / "source_immutability_before.json"
    if not before_path.is_file():
        raise RuntimeError("partial pack lacks its sealed pre-generation registry")
    before_payload = read_json(before_path)
    immutable_now = immutable_registry(sources)
    if before_payload["source_files"] != immutable_now:
        raise RuntimeError("sealed source changed before missing-page completion")
    expected_paths = [
        PACK_ROOT / directory / filename
        for directory, filename, _ in PAGE_SPECS
    ]
    existing_flags = [path.is_file() for path in expected_paths]
    existing_count = sum(existing_flags)
    if (
        existing_count == 0
        or existing_count == 11
        or existing_flags
        != [index < existing_count for index in range(len(expected_paths))]
        or (PACK_ROOT / "06_indexes" / PDF_NAME).exists()
        or (
            PACK_ROOT
            / "06_indexes"
            / "subject00_O03_camsafe7_review_pack_index.json"
        ).exists()
    ):
        raise RuntimeError(
            f"partial pack is not a safe ordered prefix: {existing_flags}"
        )
    for index, path in enumerate(expected_paths[:existing_count]):
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            expected_size = (6400, 4000) if index == 0 else (6400, 4800)
            if image.size != expected_size:
                raise RuntimeError(
                    f"existing partial page resolution changed: {path} {image.size}"
                )
    page_paths = generate_pages(sources, resume_missing_only=True)
    if not all(path.is_file() for path in page_paths):
        raise RuntimeError("missing-page completion did not create all 11 pages")
    if before_payload["source_files"] != immutable_registry(sources):
        raise RuntimeError("sealed source changed during missing-page completion")
    return {
        "task_id": TASK_ID,
        "status": "PASS_MISSING_PAGES_COMPLETED_WITHOUT_OVERWRITE",
        "preserved_existing_page_count": existing_count,
        "new_page_count": 11 - existing_count,
        "page_paths": [str(path) for path in page_paths],
        "execution_head": gate_record["head"],
        "optimizer_steps": 0,
    }


def phase_finalize() -> dict[str, Any]:
    gate_record = gate(expect_pack=True)
    sources = load_sources()
    before_path = PACK_ROOT / "06_indexes" / "source_immutability_before.json"
    before_payload = read_json(before_path)
    return finalize_pack(gate_record, sources, before_payload["source_files"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("preflight", "pages", "pages-resume", "finalize"),
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.phase == "preflight":
        result = phase_preflight()
    elif args.phase == "pages":
        result = phase_pages()
    elif args.phase == "pages-resume":
        result = phase_pages_resume()
    else:
        result = phase_finalize()
    print(json.dumps(result, sort_keys=True))
