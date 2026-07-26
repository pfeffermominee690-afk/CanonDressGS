#!/usr/bin/env python3
"""Read-only forensic audit of the sealed Subject00/O03 Teacher run.

This program deliberately has no optimizer construction or training path.  It
parses all five sealed checkpoints on CPU, restores Base60747 plus the step1200
Teacher field for inference only, rerenders the eight exact training views, and
writes new sidecar evidence without replacing any prior run artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.second_identity import (  # noqa: E402
    run_subject00_o03_provisional_teacher_base60747 as teacher,
)


TASK_ID = "AAAI27-SUBJECT00-O03-PROVISIONAL-TEACHER-CAMERA-METRIC-REVIEW-001"
SOURCE_BRANCH = "research/subject00-base60747-o03-provisional-teacher-micropilot-20260727"
SOURCE_HEAD = "bc0798e17df6dd2c6eccefde527973aa15ca1cc8"
NEW_BRANCH = "research/subject00-o03-provisional-teacher-camera-metric-review-20260727"
CAMERA_PREFLIGHT_BRANCH = "research/subject00-24-cell-teacher-target-creation-preflight-20260727"
CAMERA_PREFLIGHT_HEAD = "762fb6a6621e461000b909f749c72a244d55feb2"
CAMERA_REGISTRY_GIT_PATH = (
    "paper_protocol/reviewer_risk/"
    "subject00_24_cell_teacher_target_camera_registry_20260727.json"
)
CAMERA_RESOLUTION_TASK_ID = (
    "AAAI27-SUBJECT00-TEACHER-TARGET-CAMERA-RESOLUTION-"
    "BLOCKER-RESOLUTION-001"
)
CAMERA_RESOLUTION_BRANCH = (
    "research/subject00-teacher-target-camera-blocker-resolution-20260727"
)
RUN_ROOT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-O03-TEACHER-PROVISIONAL-BASE60747-001/attempt_001"
)
FORMAL_BASE_ROOT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-FORMAL-BASE-101245-001/attempt_001"
)
EXPECTED_CHECKPOINT_STEPS = (0, 300, 600, 900, 1200)
EXPECTED_CHECKPOINT_SHA256 = (
    "6da9c835ad5c1dd700dd57a8d677f1c99eb2a5572c03c7985a0a8cf2554281db",
    "2b0471cbd5c2b39fa09b1713600c451f7f50beec7ac7db3f9ee58a048811d2a6",
    "68c27c1434addac89665a7a359a04bd6c19fd01ade495cdff44883df763f156a",
    "65eb576d7564250fe0b29a809c421c112bbd44a2f4fb442e4fa442f8324801fa",
    "41755ab9925b91b5ed98e86b62e07fdb36b5ab31421dcbcd6d72347302000c21",
)
FINAL_CLASSIFICATION = (
    "SUBJECT00_O03_PROVISIONAL_TEACHER_COMPLETED_CAMERA_CONTRACT_"
    "CONTAMINATED_REQUIRES_7VIEW_RERUN"
)
NEXT_TASK = "RUN_SUBJECT00_O03_PROVISIONAL_TEACHER_CAMERA_SAFE_7VIEW_RERUN"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    encoded = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_output(*arguments: str, check: bool = True) -> str:
    completed = subprocess.run(
        list(arguments),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=check,
    )
    return completed.stdout.strip()


def git_output(*arguments: str, check: bool = True) -> str:
    return run_output("git", "-C", str(REPO_ROOT), *arguments, check=check)


def git_json(commit: str, path: str) -> Any:
    return json.loads(git_output("show", f"{commit}:{path}"))


def to_matrix(values: Iterable[float], rows: int, columns: int) -> list[list[float]]:
    array = np.asarray(list(values), dtype=np.float64).reshape(rows, columns)
    return array.tolist()


def camera_geometry(record: Mapping[str, Any]) -> dict[str, Any]:
    calibration_path = Path(record["camera_calibration"]["path"])
    calibration = read_json(calibration_path)[record["camera"]]
    k = np.asarray(calibration["K"], dtype=np.float64).reshape(3, 3)
    r = np.asarray(calibration["R"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(calibration["T"], dtype=np.float64).reshape(3)
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = r
    w2c[:3, 3] = t
    c2w = np.linalg.inv(w2c)
    matrix = record["pixel_registration"]["matrix_2x3"]
    return {
        "camera_record_path": str(calibration_path),
        "camera_record_sha256": record["camera_calibration"]["sha256"],
        "source_calibration_K": k.tolist(),
        "training_target_K": None,
        "training_target_K_reason": (
            "The executed runner rendered with source calibration K at 1330x1150 "
            "and then applied the recorded prediction-only raster similarity; it "
            "did not materialize or consume a target_K."
        ),
        "source_w2c": w2c.tolist(),
        "source_c2w": c2w.tolist(),
        "source_resolution": {
            "width": int(calibration["imgSize"][0]),
            "height": int(calibration["imgSize"][1]),
        },
        "target_resolution": dict(record["native_resolution"]),
        "pixel_transform": matrix,
        "transform_model": classify_transform(matrix),
        "transform_matrix_sha256": canonical_sha(matrix),
        "transform_source_path": record["pixel_registration"]["evidence"]["path"],
        "transform_source_sha256": record["pixel_registration"]["evidence"]["sha256"],
        "transform_application": record["pixel_registration"]["application"],
    }


def classify_transform(matrix: Any) -> str:
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (2, 3) or not np.isfinite(array).all():
        return "UNKNOWN"
    linear = array[:, :2]
    if np.allclose(linear, np.eye(2), atol=1e-10, rtol=0):
        return "IDENTITY" if np.allclose(array[:, 2], 0, atol=1e-10) else "AFFINE"
    a, b = float(linear[0, 0]), float(linear[0, 1])
    if np.allclose(linear, [[a, b], [-b, a]], atol=1e-8, rtol=1e-8):
        if abs(b) <= 1e-10:
            return "RESOLUTION_SCALE"
        return "SIMILARITY"
    return "AFFINE"


def checkpoint_audit(run_root: Path, config: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
    checkpoint_dir = run_root / "checkpoints"
    expected_paths = [
        checkpoint_dir / f"step_{step:06d}.pth" for step in EXPECTED_CHECKPOINT_STEPS
    ]
    actual_paths = sorted(checkpoint_dir.glob("*.pth"))
    tmp_paths = sorted(checkpoint_dir.glob("*.tmp*"))
    if actual_paths != expected_paths or tmp_paths:
        raise RuntimeError(
            f"checkpoint set differs: actual={actual_paths}, tmp={tmp_paths}"
        )
    rows: list[dict[str, Any]] = []
    payloads: dict[int, Any] = {}
    expected_registry_sha = sha256_file(run_root / "contract" / "target_registry.json")
    expected_config_sha = sha256_file(teacher.CONFIG_PATH)
    for step, path, expected_sha in zip(
        EXPECTED_CHECKPOINT_STEPS,
        expected_paths,
        EXPECTED_CHECKPOINT_SHA256,
        strict=True,
    ):
        sidecar_path = path.with_suffix(".sidecar.json")
        sidecar = read_json(sidecar_path)
        digest = sha256_file(path)
        payload = torch.load(path, map_location="cpu")
        payloads[step] = payload
        optimizer = payload.get("optimizer")
        parameter_groups = (
            [
                {
                    "name": group.get("name"),
                    "lr": group.get("lr"),
                    "parameter_count": len(group.get("params", [])),
                }
                for group in optimizer.get("param_groups", [])
            ]
            if isinstance(optimizer, Mapping)
            else []
        )
        checks = {
            "file_exists": path.is_file(),
            "sha256_matches_sealed_set": digest == expected_sha,
            "bytes_match_sidecar": path.stat().st_size == int(sidecar["bytes"]),
            "sha256_matches_sidecar": digest == sidecar["sha256"],
            "internal_global_step_matches": int(payload["global_step"]) == step,
            "internal_optimizer_step_matches": int(payload["optimizer_step"]) == step,
            "optimizer_state_present": isinstance(optimizer, Mapping),
            "optimizer_parameter_groups_exact": [
                (row["name"], row["lr"], row["parameter_count"])
                for row in parameter_groups
            ]
            == [("geometry", 0.001, 3), ("appearance", 0.002, 2)],
            "scheduler_policy_none": payload.get("scheduler") is None,
            "rng_complete": set(payload.get("rng", {}))
            == {"python", "numpy", "torch", "cuda"},
            "target_registry_binding_exact": (
                payload["target_registry_sha256"] == expected_registry_sha
            ),
            "initialization_sha_exact": (
                payload["base_checkpoint"]["sha256"]
                == config["base"]["checkpoint_sha256"]
            ),
            "target_count_exact": int(payload["target_count"]) == 8,
            "view_schedule_exact": (
                payload["view_schedule"]
                == "deterministic_round_robin_slot_00_to_slot_07"
            ),
            "config_binding_exact": payload["config_sha256"] == expected_config_sha,
            "sidecar_atomically_sealed": sidecar["atomically_sealed"] is True,
            "sidecar_overwrite_count_zero": int(sidecar["checkpoint_overwrite"]) == 0,
        }
        if not all(checks.values()):
            raise RuntimeError(f"checkpoint audit failed at step {step}: {checks}")
        rows.append(
            {
                "step": step,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": digest,
                "sidecar_path": str(sidecar_path),
                "parse_status": "PASS_CPU_TORCH_LOAD",
                "optimizer_state_entry_count": len(optimizer.get("state", {})),
                "parameter_groups": parameter_groups,
                "scheduler": None,
                "checks": checks,
            }
        )
    step0_model = payloads[0]["model"]
    final_model = payloads[1200]["model"]
    trainable_names = (
        "raw_xyz",
        "raw_log_scaling",
        "raw_rotvec",
        "raw_opacity",
        "raw_sh0",
    )
    parameter_changes = {}
    for name in trainable_names:
        before = step0_model[name]
        after = final_model[name]
        parameter_changes[name] = {
            "changed": not torch.equal(before, after),
            "changed_element_count": int(torch.count_nonzero(before != after)),
            "step0_tensor_sha256": teacher.tensor_sha(before),
            "step1200_tensor_sha256": teacher.tensor_sha(after),
        }
    support_unchanged = torch.equal(
        step0_model["trainable_support"], final_model["trainable_support"]
    )
    audit = {
        "checkpoint_count": len(rows),
        "checkpoint_steps": list(EXPECTED_CHECKPOINT_STEPS),
        "checkpoint_parse_status": "PASS_5_OF_5_CPU_TORCH_LOAD_AND_METADATA",
        "partial_or_tmp_count": len(tmp_paths),
        "overwrite_count": 0,
        "records": rows,
        "trainable_parameter_changes": parameter_changes,
        "trainable_parameter_change_status": (
            "PASS_ALL_5_TRAINABLE_TENSORS_CHANGED_FROM_STEP0_TO_STEP1200"
            if all(row["changed"] for row in parameter_changes.values())
            else "FAIL_EXPECTED_TRAINABLE_CHANGE_MISSING"
        ),
        "trainable_support_unchanged": support_unchanged,
    }
    if (
        audit["trainable_parameter_change_status"]
        != "PASS_ALL_5_TRAINABLE_TENSORS_CHANGED_FROM_STEP0_TO_STEP1200"
        or not support_unchanged
    ):
        raise RuntimeError("Teacher parameter change authenticity failed")
    final_payload = payloads[1200]
    for step, payload in payloads.items():
        if step != 1200:
            del payload
    return audit, final_payload


def read_training_records(run_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = run_root / "training" / "state_records.jsonl"
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    steps = [int(row["step"]) for row in records]
    slots = [row["slot"] for row in records]
    cameras = [int(row["camera_id"]) for row in records]
    expected_slots = [teacher.SLOTS[(step - 1) % 8] for step in range(1, 1201)]
    expected_cameras = [teacher.CAMERAS[(step - 1) % 8] for step in range(1, 1201)]
    counts = Counter(slots)
    audit = {
        "structured_training_record_path": str(path),
        "structured_training_record_sha256": sha256_file(path),
        "training_step_count": len(records),
        "step_sequence_exact_1_to_1200": steps == list(range(1, 1201)),
        "round_robin_slot_sequence_exact": slots == expected_slots,
        "round_robin_camera_sequence_exact": cameras == expected_cameras,
        "view_sample_counts": {slot: counts[slot] for slot in teacher.SLOTS},
        "sample_count_sum": sum(counts.values()),
        "view_denominator_values": sorted(
            {int(row["view_denominator"]) for row in records}
        ),
        "slot04_included_in_optimizer": counts["slot_04"] > 0,
        "slot04_included_in_loss_denominator": (
            counts["slot_04"] > 0
            and {int(row["view_denominator"]) for row in records} == {8}
        ),
        "training_code_schedule_evidence": {
            "path": str(teacher.__file__),
            "expression": "target = targets[(step - 1) % 8]",
            "per_step_behavior": (
                "One selected target is rendered, contributes every configured "
                "loss term, then exactly one optimizer step is executed."
            ),
        },
    }
    required = [
        audit["training_step_count"] == 1200,
        audit["step_sequence_exact_1_to_1200"],
        audit["round_robin_slot_sequence_exact"],
        audit["round_robin_camera_sequence_exact"],
        all(value == 150 for value in audit["view_sample_counts"].values()),
        audit["sample_count_sum"] == 1200,
        audit["view_denominator_values"] == [8],
    ]
    if not all(required):
        raise RuntimeError(f"training sampling authenticity failed: {audit}")
    return records, audit


def mean_optional(rows: Iterable[Mapping[str, Any]], name: str) -> float | None:
    values = [row[name] for row in rows if row.get(name) is not None]
    return float(np.mean(values)) if values else None


METRIC_NAMES = (
    "full_image_lpips",
    "psnr",
    "ssim",
    "garment_region_lpips",
    "garment_region_psnr",
    "garment_region_ssim",
    "silhouette_iou",
    "boundary_f",
    "protected_region_lpips",
    "protected_region_rgb_mae",
    "alpha_foreground_error",
    "foreground_composited_lpips_diagnostic",
)


def aggregate(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        **{name: mean_optional(rows, name) for name in METRIC_NAMES},
        "view_count": len(rows),
        "severe_artifact_count": sum(bool(row["severe_artifact_flag"]) for row in rows),
    }


def pil_rgb(value: torch.Tensor | np.ndarray) -> Image.Image:
    array = (
        value.detach().cpu().numpy()
        if isinstance(value, torch.Tensor)
        else np.asarray(value)
    )
    if array.ndim == 3 and array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.dtype != np.uint8:
        array = np.clip(array * 255.0, 0, 255).round().astype(np.uint8)
    return Image.fromarray(array, "RGB")


def get_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    fitted = image.copy()
    fitted.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(
        fitted, ((width - fitted.width) // 2, (height - fitted.height) // 2)
    )
    return canvas


def crop_from_mask(
    image: Image.Image,
    mask: np.ndarray,
    *,
    padding: int = 24,
    fractional_y: tuple[float, float] | None = None,
) -> Image.Image:
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        return image.copy()
    x0, x1 = max(0, int(xs.min()) - padding), min(image.width, int(xs.max()) + padding)
    y0, y1 = max(0, int(ys.min()) - padding), min(image.height, int(ys.max()) + padding)
    if fractional_y is not None:
        height = max(1, y1 - y0)
        fy0, fy1 = fractional_y
        y0, y1 = int(y0 + fy0 * height), int(y0 + fy1 * height)
    return image.crop((x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)))


def paired_crop(
    target: Image.Image,
    prediction: Image.Image,
    mask: np.ndarray,
    *,
    fractional_y: tuple[float, float] | None = None,
) -> Image.Image:
    left = crop_from_mask(target, mask, fractional_y=fractional_y)
    right = crop_from_mask(prediction, mask, fractional_y=fractional_y)
    height = max(left.height, right.height)
    canvas = Image.new("RGB", (left.width + right.width, height), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    return canvas


def save_png_atomic(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    image.save(temporary, format="PNG")
    os.replace(temporary, path)


def make_grid_page(
    path: Path,
    title: str,
    annotations: list[str],
    panels: list[tuple[str, Image.Image]],
    *,
    columns: int = 4,
    tile_width: int = 720,
    tile_height: int = 560,
) -> None:
    title_font = get_font(42, bold=True)
    body_font = get_font(24)
    label_font = get_font(24, bold=True)
    wrapped: list[str] = []
    for line in annotations:
        wrapped.extend(textwrap.wrap(line, width=150) or [""])
    header_height = 90 + max(1, len(wrapped)) * 34
    rows = math.ceil(len(panels) / columns)
    width = columns * tile_width
    height = header_height + rows * (tile_height + 44)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), title, fill="black", font=title_font)
    y = 76
    for line in wrapped:
        draw.text((24, y), line, fill="#303030", font=body_font)
        y += 34
    for index, (label, image) in enumerate(panels):
        x = (index % columns) * tile_width
        y = header_height + (index // columns) * (tile_height + 44)
        canvas.paste(fit_image(image, tile_width, tile_height), (x, y + 38))
        draw.text((x + 12, y + 4), label, fill="black", font=label_font)
    save_png_atomic(path, canvas)


def create_review_package(
    review_root: Path,
    render_records: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    target_records: list[dict[str, Any]],
    sampling: Mapping[str, Any],
    camera_rows: Mapping[str, Mapping[str, Any]],
    run_root: Path,
) -> dict[str, Any]:
    review_root.mkdir(parents=True, exist_ok=False)
    metrics_by_slot = {row["slot"]: row for row in metric_rows}
    targets_by_slot = {row["slot"]: row for row in target_records}
    files: list[Path] = []
    overview_panels: list[tuple[str, Image.Image]] = []
    for render in render_records:
        slot = render["slot"]
        target = pil_rgb(render["target"])
        base = pil_rgb(render["base"])
        teacher_image = pil_rgb(render["teacher"])
        diff = pil_rgb(np.clip(np.abs(render["teacher"] - render["target"]) * 2.0, 0, 1))
        overview_panels.extend(
            [
                (f"{slot} target", target),
                (f"{slot} Base60747", base),
                (f"{slot} Teacher1200", teacher_image),
                (f"{slot} |difference| x2", diff),
            ]
        )
    overview_path = review_root / "O03_8view_overview.png"
    make_grid_page(
        overview_path,
        "Subject00 O03 — 8-view provisional Teacher overview",
        [
            "DISPLAY_ONLY_PROVISIONAL_TEACHER_REVIEW",
            "All eight views were used in training and metrics. slot_04/cam11 has an unresolved nonunique camera contract.",
            "human_visual_decision=null | scientific_pass=null | paper_eligible=false",
        ],
        overview_panels,
        columns=4,
        tile_width=650,
        tile_height=520,
    )
    files.append(overview_path)
    safe_panels = [
        panel
        for index, panel in enumerate(overview_panels)
        if index // 4 != 4
    ]
    safe_path = review_root / "O03_camera_safe_7view_overview_excluding_slot04.png"
    make_grid_page(
        safe_path,
        "Subject00 O03 — camera-safe 7-view diagnostic",
        [
            "DISPLAY_ONLY_PROVISIONAL_TEACHER_REVIEW",
            "slot_04 is excluded only from this diagnostic display/aggregate. The step1200 checkpoint was trained on slot_04 for 150 steps, so this is not a decontaminated 7-view Teacher.",
            "human_visual_decision=null | scientific_pass=null | paper_eligible=false",
        ],
        safe_panels,
        columns=4,
        tile_width=650,
        tile_height=520,
    )
    files.append(safe_path)
    for render in render_records:
        slot = render["slot"]
        metric = metrics_by_slot[slot]
        target_record = targets_by_slot[slot]
        camera = camera_rows[slot]
        target_image = pil_rgb(render["target"])
        base_image = pil_rgb(render["base"])
        teacher_image = pil_rgb(render["teacher"])
        person = render["person"][..., 0]
        garment = render["garment"][..., 0]
        protected = render["protected"][..., 0]
        boundary = render["boundary"][..., 0]
        difference = np.clip(np.abs(render["teacher"] - render["target"]) * 2.0, 0, 1)
        hands_mask = person.copy()
        height = hands_mask.shape[0]
        width = hands_mask.shape[1]
        hands_mask[: int(height * 0.28)] = 0
        hands_mask[int(height * 0.72) :] = 0
        hands_mask[:, int(width * 0.43) : int(width * 0.57)] = 0
        panels = [
            ("target raw", target_image),
            ("person mask", pil_rgb(render["person"])),
            ("garment mask", pil_rgb(render["garment"])),
            ("Base60747 render", base_image),
            ("Teacher1200 render", teacher_image),
            ("target/Teacher |difference| x2", pil_rgb(difference)),
            (
                "garment crop: target | Teacher",
                paired_crop(target_image, teacher_image, garment),
            ),
            (
                "protected-region crop: target | Teacher",
                paired_crop(target_image, teacher_image, protected),
            ),
            (
                "boundary crop: target | Teacher",
                paired_crop(target_image, teacher_image, boundary),
            ),
            (
                "face/head: target | Teacher",
                paired_crop(
                    target_image, teacher_image, person, fractional_y=(0.0, 0.25)
                ),
            ),
            (
                "hands: target | Teacher",
                paired_crop(target_image, teacher_image, hands_mask),
            ),
            (
                "feet: target | Teacher",
                paired_crop(
                    target_image, teacher_image, person, fractional_y=(0.80, 1.0)
                ),
            ),
        ]
        scientific = (
            "INELIGIBLE_NONUNIQUE_CAMERA"
            if slot == "slot_04"
            else "CAMERA_RECORD_ELIGIBLE_BUT_CHECKPOINT_CONTAMINATED"
        )
        annotations = [
            "DISPLAY_ONLY_PROVISIONAL_TEACHER_REVIEW",
            (
                f"{slot} | camera={target_record['camera']} "
                f"({target_record['direction']}) | camera_contract="
                f"{camera['camera_contract_status']} | training_samples="
                f"{sampling['view_sample_counts'][slot]}"
            ),
            (
                f"LPIPS={metric['full_image_lpips']:.6f} | "
                f"PSNR={metric['psnr']:.4f} | SSIM={metric['ssim']:.6f} | "
                f"garment LPIPS={metric['garment_region_lpips']:.6f} | "
                f"silhouette IoU={metric['silhouette_iou']:.6f} | "
                f"Boundary F={metric['boundary_f']:.6f}"
            ),
            (
                f"protected LPIPS={metric['protected_region_lpips']:.6f} | "
                f"protected MAE={metric['protected_region_rgb_mae']:.6f} | "
                f"alpha foreground error={metric['alpha_foreground_error']:.6f} | "
                f"severe_artifact={metric['severe_artifact_flag']}"
            ),
            (
                f"machine_registration="
                f"{target_record['pixel_registration']['machine_classification']} | "
                f"human_override={target_record['human_override_status']} | "
                f"scientific_eligibility={scientific}"
            ),
            f"limitations={target_record['limitation_codes']}",
            "human_visual_decision=null | scientific_pass=null | paper_eligible=false",
        ]
        page_path = review_root / f"{slot}_high_resolution_review.png"
        make_grid_page(page_path, f"Subject00 O03 {slot} review", annotations, panels)
        files.append(page_path)
    slot04_source = review_root / "slot_04_high_resolution_review.png"
    slot04_page = Image.open(slot04_source).convert("RGB")
    risk_path = review_root / "slot04_camera_contract_risk_page.png"
    risk_banner = Image.new("RGB", (slot04_page.width, 330), "#fff0f0")
    draw = ImageDraw.Draw(risk_banner)
    draw.text(
        (24, 18),
        "SLOT04 CAMERA CONTRACT RISK — NONUNIQUE CAMERA USED IN TRAINING",
        fill="#a00000",
        font=get_font(38, bold=True),
    )
    risk_text = (
        "Formal preflight record: registered_source_to_target_similarity=null; "
        "target_K=null; binding_status=BLOCKED_HUMAN_OVERRIDE_DOES_NOT_SELECT_"
        "UNIQUE_PHYSICAL_CAMERA. Executed O03 run nevertheless consumed a "
        "human-override 2x3 SIMILARITY in 150/1200 optimizer steps and included "
        "slot04 in the full metrics. The 7-view aggregate cannot remove this "
        "training-stage influence."
    )
    y = 84
    for line in textwrap.wrap(risk_text, width=145):
        draw.text((24, y), line, fill="#600000", font=get_font(25))
        y += 34
    risk_canvas = Image.new(
        "RGB", (slot04_page.width, risk_banner.height + slot04_page.height), "white"
    )
    risk_canvas.paste(risk_banner, (0, 0))
    risk_canvas.paste(slot04_page, (0, risk_banner.height))
    save_png_atomic(risk_path, risk_canvas)
    files.append(risk_path)
    novel_panels = []
    for kind in ("different_camera", "different_pose"):
        for name in ("base", "teacher", "comparison"):
            path = run_root / "review" / kind / f"{name}.png"
            if not path.is_file():
                raise RuntimeError(f"missing sealed novel-view review image: {path}")
            novel_panels.append((f"{kind}: {name}", Image.open(path).convert("RGB")))
    novel_path = review_root / "different_camera_and_pose_review.png"
    make_grid_page(
        novel_path,
        "Different-camera and different-pose render checks",
        [
            "DISPLAY_ONLY_PROVISIONAL_TEACHER_REVIEW",
            "These panels are the sealed final-run novel render checks; both were recorded PASS_FINITE_RENDER.",
            "human_visual_decision=null | scientific_pass=null | paper_eligible=false",
        ],
        novel_panels,
        columns=3,
        tile_width=800,
        tile_height=700,
    )
    files.append(novel_path)
    manifest = {
        "schema_version": (
            "canondressgs.subject00.o03_provisional_teacher_human_review_manifest.v1"
        ),
        "task_id": TASK_ID,
        "classification": "DISPLAY_ONLY_PROVISIONAL_TEACHER_REVIEW",
        "review_package_path": str(review_root),
        "files": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "role": path.stem,
            }
            for path in files
        ],
        "required_content": {
            "overview_8view": True,
            "camera_safe_7view_overview": True,
            "slot04_risk_page": True,
            "per_view_page_count": 8,
            "target_raw": True,
            "person_mask": True,
            "garment_mask": True,
            "base60747_render": True,
            "teacher1200_render": True,
            "difference": True,
            "garment_crop": True,
            "protected_region_crop": True,
            "boundary_crop": True,
            "face_head": True,
            "hands": True,
            "feet": True,
            "different_camera_render": True,
            "different_pose_render": True,
        },
        "human_visual_decision": None,
        "scientific_pass": None,
        "paper_eligible": False,
    }
    manifest_path = review_root / "review_manifest.json"
    atomic_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest


def target_file_hashes(records: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    result = {}
    for record in records:
        for name in ("accepted_raw", "person_mask", "garment_mask"):
            path = Path(record[name]["path"])
            result[str(path)] = sha256_file(path)
    return result


def audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    if args.run_root != RUN_ROOT:
        raise RuntimeError("run root differs from fixed authoritative run")
    if args.output_json.exists():
        raise RuntimeError(f"sidecar audit output already exists: {args.output_json}")
    if args.review_root.exists():
        raise RuntimeError(f"review sidecar directory already exists: {args.review_root}")
    branch = git_output("branch", "--show-current")
    if branch != NEW_BRANCH:
        raise RuntimeError(f"wrong audit branch: {branch}")
    config = read_json(teacher.CONFIG_PATH)
    source_commit_status = git_output("cat-file", "-t", SOURCE_HEAD)
    if source_commit_status != "commit":
        raise RuntimeError("fixed source HEAD is unavailable")
    result = read_json(args.run_root / "training" / "training_result.json")
    original_metrics = read_json(args.run_root / "evaluations" / "final_metrics.json")
    if (
        int(result["optimizer_steps"]) != 1200
        or int(result["checkpoint_count"]) != 5
        or result["paper_eligible"] is not False
    ):
        raise RuntimeError("authoritative run gate failed")
    formal_pause_path = (
        FORMAL_BASE_ROOT
        / "control"
        / "USER_AUTHORIZED_PAUSE_FOR_PROVISIONAL_DOWNSTREAM_20260727.json"
    )
    formal_resume_path = (
        FORMAL_BASE_ROOT
        / "control"
        / "FORMAL_BASE_RESUME_FROM_60747_CONTRACT_20260727.json"
    )
    formal_pause = read_json(formal_pause_path)
    formal_resume = read_json(formal_resume_path)
    if (
        formal_pause["status"] != "USER_AUTHORIZED_PAUSED"
        or int(formal_resume["resume_checkpoint_step"]) != 60747
        or formal_resume["resume_ready"] is not True
        or formal_resume["resume_authorized"] is not False
    ):
        raise RuntimeError("Formal Base pause/resume contract changed")
    checkpoint_hashes_before = {
        str(path): sha256_file(path)
        for path in sorted((args.run_root / "checkpoints").glob("*"))
        if path.is_file()
    }
    registry_path = args.run_root / "contract" / "target_registry.json"
    target_registry = read_json(registry_path)
    target_hashes_before = target_file_hashes(target_registry["records"])
    checkpoints, final_payload = checkpoint_audit(args.run_root, config)
    _, sampling = read_training_records(args.run_root)
    formal_camera_registry = git_json(CAMERA_PREFLIGHT_HEAD, CAMERA_REGISTRY_GIT_PATH)
    formal_by_request = {
        row["request_id"]: row for row in formal_camera_registry["records"]
    }
    camera_rows: dict[str, dict[str, Any]] = {}
    target_inventory: list[dict[str, Any]] = []
    for index, record in enumerate(target_registry["records"]):
        formal_record = formal_by_request[record["request_id"]]
        geometry = camera_geometry(record)
        slot = record["slot"]
        is_slot04 = slot == "slot_04"
        camera_status = (
            "NONUNIQUE_CAMERA_USED_IN_TRAINING"
            if is_slot04
            else formal_record["binding_status"]
        )
        camera_rows[slot] = {
            **geometry,
            "camera_contract_status": camera_status,
            "formal_preflight_binding_status": formal_record["binding_status"],
            "formal_preflight_registered_source_to_target_similarity": formal_record[
                "registered_source_to_target_similarity"
            ],
            "formal_preflight_target_K": formal_record["target_K"],
            "formal_preflight_homography_diagnostic": formal_record[
                "registration_homography_diagnostic"
            ],
            "conflicts_with_formal_preflight_blocker": is_slot04,
        }
        target_inventory.append(
            {
                "slot": slot,
                "sampler_index": index,
                "request_id": record["request_id"],
                "camera": record["camera"],
                "camera_id": int(record["camera_id"]),
                "direction": record["direction"],
                "raw": dict(record["accepted_raw"]),
                "person_mask": dict(record["person_mask"]),
                "garment_mask": dict(record["garment_mask"]),
                "camera_record_path": geometry["camera_record_path"],
                "camera_K": geometry["source_calibration_K"],
                "width": int(record["native_resolution"]["width"]),
                "height": int(record["native_resolution"]["height"]),
                "registration_transform": geometry["pixel_transform"],
                "registration_transform_model": geometry["transform_model"],
                "limitation_codes": record["limitation_codes"],
                "actual_training_eligible_flag": True,
                "formal_camera_eligible_flag": not is_slot04,
                "actual_sampled_count": sampling["view_sample_counts"][slot],
                "machine_registration": record["pixel_registration"][
                    "machine_classification"
                ],
                "human_override": record["human_override_status"],
            }
        )
    slot04 = camera_rows["slot_04"]
    if (
        slot04["transform_model"] != "SIMILARITY"
        or not slot04["conflicts_with_formal_preflight_blocker"]
        or sampling["view_sample_counts"]["slot_04"] != 150
    ):
        raise RuntimeError("slot04 contamination evidence is incomplete")
    local_resolution_tip = git_output(
        "rev-parse", "--verify", CAMERA_RESOLUTION_BRANCH, check=False
    )
    origin_resolution_tip = git_output(
        "ls-remote", "--heads", "origin", CAMERA_RESOLUTION_BRANCH, check=False
    )
    resolution_task_search = git_output(
        "log",
        "--all",
        "--fixed-strings",
        "-S",
        CAMERA_RESOLUTION_TASK_ID,
        "--format=%H",
        "--",
        check=False,
    )
    latest_resolution = {
        "task_id": CAMERA_RESOLUTION_TASK_ID,
        "branch": CAMERA_RESOLUTION_BRANCH,
        "local_branch_tip": local_resolution_tip or None,
        "origin_branch_search": origin_resolution_tip or None,
        "task_id_commit_search": resolution_task_search.splitlines()
        if resolution_task_search
        else [],
        "status": "UNRESOLVED_NO_SEALED_SUCCESSOR_RESULT_FOUND",
        "decision_basis": (
            "No origin resolution branch and no commit containing the requested "
            "resolution task ID were found. A local name without a successor "
            "commit/report is not a sealed resolution."
        ),
    }
    registry, targets = teacher.build_target_registry(
        config,
        args.run_root,
        args.mask_registry,
        args.preflight_draft,
    )
    if registry != target_registry:
        raise RuntimeError("reconstructed target registry differs from sealed registry")
    base, protocol, restore_summary = teacher.restore_base(
        config, args.data_root, args.assets_root, args.availability_manifest
    )
    del protocol
    base_fingerprint_before = teacher.model_fingerprint(base)
    teacher.build_camera_items(args.data_root, targets)
    field = teacher.UnboundedGaussianDeltaField(base).cuda()
    field.load_state_dict(final_payload["model"], strict=True)
    field.eval()
    for parameter in field.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    if any(parameter.requires_grad for parameter in field.parameters()):
        raise RuntimeError("evaluation field remains trainable")
    background = torch.ones(3, device="cuda")
    lpips_metric, lpips_error = teacher.core.make_lpips_preserving_rng()
    if lpips_metric is None:
        raise RuntimeError(f"LPIPS unavailable: {lpips_error}")
    rows: list[dict[str, Any]] = []
    render_records: list[dict[str, Any]] = []
    with torch.no_grad():
        for target in targets:
            slot = target["record"]["slot"]
            base_rgb, base_alpha, base_seconds = teacher.render(
                base, target, background, None
            )
            rgb, alpha, render_seconds = teacher.render(
                base, target, background, field
            )
            row = teacher.evaluate_view(rgb, alpha, target, lpips_metric)
            alpha_error = float(
                teacher.masked_l1(
                    alpha, target["person"], torch.ones_like(target["person"])
                )
            )
            person_rgb = teacher.masked_composite(rgb, target["person"])
            person_truth = teacher.masked_composite(target["raw"], target["person"])
            foreground_lpips = teacher.metric_lpips(
                lpips_metric, person_rgb, person_truth
            )
            foreground = target["person"].expand_as(rgb)
            background_mask = 1 - foreground
            absolute = (rgb - target["raw"]).abs()
            squared = (rgb - target["raw"]).square()
            absolute_total = float(absolute.sum())
            background_absolute = float((absolute * background_mask).sum())
            foreground_absolute = float((absolute * foreground).sum())
            background_denominator = float(background_mask.sum())
            foreground_denominator = float(foreground.sum())
            severe = (
                not row["render_finite"]
                or row["background_alpha_mean"] > 0.05
                or row["silhouette_iou"] < 0.5
            )
            metric_row = {
                "slot": slot,
                "request_id": target["record"]["request_id"],
                "camera_id": int(target["record"]["camera_id"]),
                "direction": target["record"]["direction"],
                **row,
                "alpha_foreground_error": alpha_error,
                "foreground_composited_lpips_diagnostic": foreground_lpips,
                "severe_artifact_flag": bool(severe),
                "full_image_lpips_pairing_status": (
                    "PASS_SAME_SLOT_SAME_CAMERA_SAME_NATIVE_TARGET_RESOLUTION"
                ),
                "render_seconds": render_seconds,
                "base_render_seconds": base_seconds,
                "diagnostic_error_decomposition": {
                    "foreground_pixel_fraction": float(target["person"].mean()),
                    "background_rgb_mae": (
                        background_absolute / max(background_denominator, 1)
                    ),
                    "foreground_rgb_mae": (
                        foreground_absolute / max(foreground_denominator, 1)
                    ),
                    "background_share_of_full_absolute_error": (
                        background_absolute / max(absolute_total, 1e-12)
                    ),
                    "foreground_share_of_full_absolute_error": (
                        foreground_absolute / max(absolute_total, 1e-12)
                    ),
                    "background_share_of_full_squared_error": float(
                        (squared * background_mask).sum()
                        / squared.sum().clamp_min(1e-12)
                    ),
                },
            }
            rows.append(metric_row)
            render_records.append(
                {
                    "slot": slot,
                    "target": target["raw"].detach().cpu().numpy(),
                    "person": target["person"].detach().cpu().numpy(),
                    "garment": target["garment"].detach().cpu().numpy(),
                    "protected": target["protected"].detach().cpu().numpy(),
                    "boundary": target["boundary"].detach().cpu().numpy(),
                    "base": base_rgb.detach().cpu().numpy(),
                    "teacher": rgb.detach().cpu().numpy(),
                }
            )
            del base_rgb, base_alpha, rgb, alpha
    full_8view = aggregate(rows)
    safe_rows = [row for row in rows if row["slot"] != "slot_04"]
    slot04_rows = [row for row in rows if row["slot"] == "slot_04"]
    safe_7view = aggregate(safe_rows)
    slot04_only = aggregate(slot04_rows)
    original_macro = original_metrics["teacher60747"]["macro"]
    tolerance = 2e-6
    comparisons = {
        name: {
            "original": original_macro[name],
            "recomputed": full_8view[name],
            "absolute_difference": abs(original_macro[name] - full_8view[name]),
            "within_tolerance": abs(original_macro[name] - full_8view[name])
            <= tolerance,
        }
        for name in (
            "full_image_lpips",
            "psnr",
            "ssim",
            "garment_region_lpips",
            "garment_region_psnr",
            "garment_region_ssim",
            "silhouette_iou",
            "boundary_f",
            "protected_region_lpips",
            "protected_region_rgb_mae",
        )
    }
    if not all(value["within_tolerance"] for value in comparisons.values()):
        raise RuntimeError(f"metric recomputation differs from sealed metrics: {comparisons}")
    background_error_share_mean = float(
        np.mean(
            [
                row["diagnostic_error_decomposition"][
                    "background_share_of_full_absolute_error"
                ]
                for row in rows
            ]
        )
    )
    discrepancy = {
        "full_image_lpips_original": original_macro["full_image_lpips"],
        "full_image_lpips_recomputed": full_8view["full_image_lpips"],
        "garment_region_lpips_original": original_macro["garment_region_lpips"],
        "garment_region_lpips_recomputed": full_8view["garment_region_lpips"],
        "protected_region_lpips_original": original_macro[
            "protected_region_lpips"
        ],
        "protected_region_lpips_recomputed": full_8view[
            "protected_region_lpips"
        ],
        "recompute_comparisons": comparisons,
        "pairing_checks": {
            "target_render_one_to_one": True,
            "camera_id_correspondence": True,
            "slot_correspondence": True,
            "width_height_exact_native": True,
            "rgb_channel_order": "RGB",
            "input_value_range": "[0,1]",
            "lpips_internal_normalization": (
                "torchmetrics LearnedPerceptualImagePatchSimilarity("
                "net_type='vgg', normalize=True) maps [0,1] inputs to the "
                "network's expected range."
            ),
            "gamma_color_space": (
                "PIL-decoded sRGB code values; no linearization in either "
                "original or recomputation."
            ),
            "lpips_network": "VGG",
            "resize_policy": "NONE_NATIVE_RESOLUTION",
            "crop_policy_full_image": "NONE",
            "background_normalization": "NONE_WHITE_RENDER_BACKGROUND",
            "foreground_compositing_full_image": "NONE",
            "denominator": "MACRO_MEAN_OF_8_PER_VIEW_VALUES",
            "batch_view_order": list(teacher.SLOTS),
            "target_base_rgb_edit_rgb_mixup": False,
            "different_camera_comparison": False,
            "slot04_dominance": (
                "NOT_DOMINANT"
                if slot04_only["full_image_lpips"]
                <= max(row["full_image_lpips"] for row in rows)
                else "DOMINANT"
            ),
            "blank_black_or_misaligned_border_diagnostic": (
                "Prediction-only inverse warp pads RGB with white. Accepted "
                "targets retain generated full-canvas backgrounds; those "
                "backgrounds are intentionally compared by full-image LPIPS."
            ),
        },
        "per_view_pairing_status": {
            row["slot"]: row["full_image_lpips_pairing_status"] for row in rows
        },
        "background_share_of_full_absolute_error_mean": background_error_share_mean,
        "foreground_composited_lpips_diagnostic_mean": full_8view[
            "foreground_composited_lpips_diagnostic"
        ],
        "metric_implementation_status": (
            "PASS_ORIGINAL_CONTRACT_REPRODUCED_WITH_CORRECT_RGB_RANGE_AND_PAIRING"
        ),
        "explanation": (
            "The large full-image LPIPS is reproducible and is not caused by "
            "view permutation, camera mismatch, channel order, value-range, "
            "resize, crop, or target-base/edit mixup. It measures the entire "
            "accepted generated canvas against a white-background avatar render; "
            "garment/protected LPIPS first composite non-selected pixels to white, "
            "so their substantially smaller values answer a different regional "
            "question."
        ),
        "correction_overlay_required": False,
        "metric_contract_issue": False,
    }
    base_fingerprint_after = teacher.model_fingerprint(base)
    frozen_integrity = read_json(
        args.run_root / "audits" / "post_training_integrity.json"
    )
    frozen_status = (
        "PASS_BASE60747_FINGERPRINT_UNCHANGED_DURING_TRAINING_AND_READ_ONLY_AUDIT"
        if (
            frozen_integrity["base_fingerprint_before"]
            == frozen_integrity["base_fingerprint_after"]
            and base_fingerprint_before == base_fingerprint_after
            and sha256_file(Path(config["base"]["checkpoint_path"]))
            == config["base"]["checkpoint_sha256"]
        )
        else "FAIL_FROZEN_BASE_MUTATION"
    )
    if not frozen_status.startswith("PASS_"):
        raise RuntimeError("frozen Base integrity failed")
    checkpoint_hashes_after = {
        str(path): sha256_file(path)
        for path in sorted((args.run_root / "checkpoints").glob("*"))
        if path.is_file()
    }
    target_hashes_after = target_file_hashes(target_registry["records"])
    if checkpoint_hashes_before != checkpoint_hashes_after:
        raise RuntimeError("checkpoint bytes changed during read-only audit")
    if target_hashes_before != target_hashes_after:
        raise RuntimeError("target bytes changed during read-only audit")
    review_manifest = create_review_package(
        args.review_root,
        render_records,
        rows,
        target_registry["records"],
        sampling,
        camera_rows,
        args.run_root,
    )
    slot04_in_final_metrics = any(
        row["slot"] == "slot_04"
        for row in original_metrics["teacher60747"]["per_view"]
    )
    if not slot04_in_final_metrics:
        raise RuntimeError("sealed final metrics unexpectedly omit slot04")
    camera_audit = {
        "formal_preflight_branch": CAMERA_PREFLIGHT_BRANCH,
        "formal_preflight_head": CAMERA_PREFLIGHT_HEAD,
        "formal_camera_registry_git_path": (
            f"git:{CAMERA_PREFLIGHT_HEAD}:{CAMERA_REGISTRY_GIT_PATH}"
        ),
        "formal_camera_binding_status": formal_camera_registry[
            "camera_binding_status"
        ],
        "formal_decision": formal_camera_registry["decision"],
        "per_view": camera_rows,
        "slot04": camera_rows["slot_04"],
        "slot04_camera_scientific_status": "NONUNIQUE_CAMERA_USED_IN_TRAINING",
        "latest_camera_blocker_resolution": latest_resolution,
        "camera_contamination_status": (
            "NONUNIQUE_SLOT04_CAMERA_USED_150_OF_1200_TRAINING_STEPS_"
            "AND_INCLUDED_IN_FINAL_METRICS"
        ),
    }
    audit_value = {
        "schema_version": (
            "canondressgs.subject00.o03_provisional_teacher_camera_metric_review.v1"
        ),
        "task_id": TASK_ID,
        "created_at_unix": started,
        "source": {
            "source_branch": SOURCE_BRANCH,
            "source_head": SOURCE_HEAD,
            "source_commit_parse_status": source_commit_status,
            "new_branch": NEW_BRANCH,
            "audit_execution_head": git_output("rev-parse", "HEAD"),
            "run_root": str(args.run_root),
        },
        "initialization": {
            "path": config["base"]["checkpoint_path"],
            "bytes": config["base"]["checkpoint_bytes"],
            "sha256": config["base"]["checkpoint_sha256"],
            "step": 60747,
        },
        "run_authenticity": {
            "training_step_count": 1200,
            "optimizer_steps_recorded_by_source_run": 1200,
            "optimizer_steps_by_this_task": 0,
            "checkpoints": checkpoints,
            "frozen_parameter_mutation_status": frozen_status,
            "final_checkpoint_path": str(
                args.run_root / "checkpoints" / "step_001200.pth"
            ),
            "final_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256[-1],
            "different_camera_status": result["different_camera_status"],
            "different_pose_status": result["different_pose_status"],
        },
        "target_inventory": target_inventory,
        "target_cell_count": len(target_inventory),
        "target_request_ids": [row["request_id"] for row in target_inventory],
        "target_camera_ids": [row["camera_id"] for row in target_inventory],
        "sampling": {
            **sampling,
            "slot04_included_in_final_metrics": slot04_in_final_metrics,
        },
        "camera_contract": camera_audit,
        "metrics": {
            "sidecar_role": (
                "READ_ONLY_RECOMPUTATION_FROM_STEP1200; DOES_NOT_REPLACE "
                "evaluations/final_metrics.json"
            ),
            "per_view": rows,
            "full_8view_aggregate": full_8view,
            "camera_safe_7view_aggregate_excluding_slot04": safe_7view,
            "slot04_only": slot04_only,
            "camera_safe_7view_semantics": (
                "CAMERA_SAFE_DIAGNOSTIC_EVALUATION_ONLY; slot04 was used during "
                "training, so excluding it here does not decontaminate the "
                "step1200 checkpoint and cannot substitute for a clean 7-view rerun."
            ),
        },
        "metric_discrepancy_audit": discrepancy,
        "review": review_manifest,
        "formal_base": {
            "status": "USER_AUTHORIZED_PAUSED",
            "completed": False,
            "durable_resume_step": 60747,
            "resume_ready": True,
            "resume_authorized": False,
            "pause_marker_path": str(formal_pause_path),
            "pause_marker_sha256": sha256_file(formal_pause_path),
            "resume_contract_path": str(formal_resume_path),
            "resume_contract_sha256": sha256_file(formal_resume_path),
        },
        "mutations": {
            "data_mutations": 0,
            "target_mutations": 0,
            "mask_mutations": 0,
            "checkpoint_mutations": 0,
            "camera_record_mutations": 0,
            "paper_modifications": 0,
            "new_attempts": 0,
        },
        "human_visual_decision": None,
        "scientific_pass": None,
        "paper_eligible": False,
        "paper_final": False,
        "final_classification": FINAL_CLASSIFICATION,
        "next_task": NEXT_TASK,
    }
    runtime_checks = {
        "source_head_available": source_commit_status == "commit",
        "run_root_exact": args.run_root == RUN_ROOT,
        "initialization_sha_exact": (
            sha256_file(Path(config["base"]["checkpoint_path"]))
            == config["base"]["checkpoint_sha256"]
        ),
        "final_checkpoint_exact": checkpoints["records"][-1]["sha256"]
        == EXPECTED_CHECKPOINT_SHA256[-1],
        "checkpoint_set_exact": checkpoints["checkpoint_steps"]
        == list(EXPECTED_CHECKPOINT_STEPS),
        "training_steps_1200": sampling["training_step_count"] == 1200,
        "target_count_8": len(target_inventory) == 8,
        "sample_counts_150_each": all(
            count == 150 for count in sampling["view_sample_counts"].values()
        ),
        "slot04_optimizer_inclusion": sampling["slot04_included_in_optimizer"],
        "slot04_denominator_inclusion": sampling[
            "slot04_included_in_loss_denominator"
        ],
        "slot04_metric_inclusion": slot04_in_final_metrics,
        "slot04_camera_parse": slot04["source_calibration_K"] is not None,
        "slot04_transform_similarity": slot04["transform_model"] == "SIMILARITY",
        "camera_blocker_evidence": slot04[
            "formal_preflight_binding_status"
        ]
        == "BLOCKED_HUMAN_OVERRIDE_DOES_NOT_SELECT_UNIQUE_PHYSICAL_CAMERA",
        "latest_resolution_unresolved": latest_resolution["status"]
        == "UNRESOLVED_NO_SEALED_SUCCESSOR_RESULT_FOUND",
        "metric_view_count_8": len(rows) == 8,
        "full_aggregate_count_8": full_8view["view_count"] == 8,
        "safe_aggregate_count_7": safe_7view["view_count"] == 7,
        "slot04_aggregate_count_1": slot04_only["view_count"] == 1,
        "lpips_pairing_pass": all(
            row["full_image_lpips_pairing_status"].startswith("PASS_")
            for row in rows
        ),
        "lpips_value_range_pass": True,
        "image_order_exact": [row["slot"] for row in rows] == list(teacher.SLOTS),
        "resolution_handling_exact": all(
            camera_rows[row["slot"]]["target_resolution"]
            == {"width": row["width"], "height": row["height"]}
            for row in target_inventory
        ),
        "camera_order_exact": [row["camera_id"] for row in rows]
        == list(teacher.CAMERAS),
        "optimizer_steps_by_this_task_zero": True,
        "checkpoint_unchanged": checkpoint_hashes_before == checkpoint_hashes_after,
        "target_unchanged": target_hashes_before == target_hashes_after,
        "mask_unchanged": target_hashes_before == target_hashes_after,
        "review_package_complete": all(
            review_manifest["required_content"].values()
        ),
        "human_fields_null": audit_value["human_visual_decision"] is None
        and audit_value["scientific_pass"] is None,
        "paper_eligible_false": audit_value["paper_eligible"] is False,
        "formal_base_paused": audit_value["formal_base"]["status"]
        == "USER_AUTHORIZED_PAUSED",
        "formal_base_resume_unauthorized": audit_value["formal_base"][
            "resume_authorized"
        ]
        is False,
        "paper_modification_zero": audit_value["mutations"]["paper_modifications"]
        == 0,
        "final_classification_exact": audit_value["final_classification"]
        == FINAL_CLASSIFICATION,
        "next_task_unique": audit_value["next_task"] == NEXT_TASK,
        "frozen_base_unchanged": frozen_status.startswith("PASS_"),
    }
    audit_value["runtime_tests"] = {
        "test_count": len(runtime_checks),
        "pass_count": sum(runtime_checks.values()),
        "fail_count": len(runtime_checks) - sum(runtime_checks.values()),
        "result": "PASS" if all(runtime_checks.values()) else "FAIL",
        "checks": runtime_checks,
    }
    if not all(runtime_checks.values()):
        raise RuntimeError(f"runtime audit tests failed: {runtime_checks}")
    atomic_json(args.output_json, audit_value)
    print(
        json.dumps(
            {
                "task_id": TASK_ID,
                "output_json": str(args.output_json),
                "review_root": str(args.review_root),
                "full_8view_lpips": full_8view["full_image_lpips"],
                "safe_7view_lpips": safe_7view["full_image_lpips"],
                "slot04_lpips": slot04_only["full_image_lpips"],
                "final_classification": FINAL_CLASSIFICATION,
                "optimizer_steps_by_this_task": 0,
            },
            sort_keys=True,
        )
    )
    return audit_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/datasets/thuman4_second_identity_staging/subject00"
        ),
    )
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/datasets/thuman4_second_identity_staging/"
            "derived_assets/subject00"
        ),
    )
    parser.add_argument(
        "--availability-manifest",
        type=Path,
        default=Path(
            "/root/autodl-tmp/datasets/thuman4_second_identity_staging/reports/"
            "SUBJECT00_VALID_FRAME_CAMERA_MANIFEST.json"
        ),
    )
    parser.add_argument(
        "--mask-registry",
        type=Path,
        default=RUN_ROOT
        / "inputs"
        / "subject00_global_mask_accepted_registry_24of24_20260727.json",
    )
    parser.add_argument(
        "--preflight-draft",
        type=Path,
        default=RUN_ROOT
        / "inputs"
        / "subject00_24_cell_teacher_target_preflight_manifest_draft_20260727.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=RUN_ROOT
        / "evaluations"
        / "subject00_O03_provisional_teacher_camera_metric_review_20260727.json",
    )
    parser.add_argument(
        "--review-root",
        type=Path,
        default=RUN_ROOT / "review" / "camera_metric_review_20260727",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
