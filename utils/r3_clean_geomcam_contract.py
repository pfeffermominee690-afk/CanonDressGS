from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np


SCHEMA = "canondressgs.r3_clean_geomcam_contract.v1"


@dataclass(frozen=True)
class BinaryMaskMetrics:
    iou: float
    dice: float
    boundary_fscore: float
    bbox_center_residual_diag: float
    bbox_scale_ratio: float
    foreground_area_ratio: float
    top_residual_fraction: float
    bottom_residual_fraction: float
    best_orientation: str


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2 or not value.any():
        raise ValueError("mask must be a non-empty [H,W] array")
    y, x = np.where(value)
    return int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1


def fixed_safe_box_from_render_only(
    rendered_mask: np.ndarray,
    output_height: int,
    output_width: int,
    safe_box_fraction: float = 5.0 / 6.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the frozen source postprocess without reading a target mask."""
    from PIL import Image

    value = np.asarray(rendered_mask, dtype=bool)
    x0, y0, x1, y1 = mask_bbox(value)
    crop = Image.fromarray(value[y0:y1, x0:x1].astype(np.uint8) * 255, "L")
    safe_width = int(round(output_width * safe_box_fraction))
    safe_height = int(round(output_height * safe_box_fraction))
    scale = min(safe_width / crop.width, safe_height / crop.height)
    fitted_width = max(1, int(round(crop.width * scale)))
    fitted_height = max(1, int(round(crop.height * scale)))
    resized = crop.resize((fitted_width, fitted_height), Image.Resampling.NEAREST)
    left = (output_width - fitted_width) // 2
    top = (output_height - fitted_height) // 2
    output = np.zeros((output_height, output_width), dtype=bool)
    output[top : top + fitted_height, left : left + fitted_width] = np.asarray(resized) >= 128
    return output, {
        "source_bbox": [x0, y0, x1, y1],
        "safe_box_fraction": float(safe_box_fraction),
        "scale": float(scale),
        "fitted_size": [fitted_width, fitted_height],
        "paste_xy": [left, top],
        "target_mask_used": False,
    }


def transform_screen_points_by_safe_box(points: np.ndarray, metadata: Mapping[str, Any]) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64).reshape(-1, 2).copy()
    x0, y0, _, _ = metadata["source_bbox"]
    left, top = metadata["paste_xy"]
    scale = float(metadata["scale"])
    value[:, 0] = (value[:, 0] - x0) * scale + left
    value[:, 1] = (value[:, 1] - y0) * scale + top
    return value


def boundary(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2 or radius < 0:
        raise ValueError("mask must be [H,W] and radius must be non-negative")
    dilated = _binary_dilate(value, radius)
    eroded = _binary_erode(value, radius)
    return dilated & ~eroded


def _binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius == 0:
        return mask.copy()
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    height, width = mask.shape
    output = np.zeros_like(mask)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            output |= padded[dy : dy + height, dx : dx + width]
    return output


def _binary_erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius == 0:
        return mask.copy()
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    height, width = mask.shape
    output = np.ones_like(mask)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            output &= padded[dy : dy + height, dx : dx + width]
    return output


def boundary_fscore(prediction: np.ndarray, target: np.ndarray, tolerance: int = 2) -> float:
    pred = boundary(prediction)
    truth = boundary(target)
    pred_dilated = _binary_dilate(pred, tolerance)
    truth_dilated = _binary_dilate(truth, tolerance)
    precision = float((pred & truth_dilated).sum() / max(pred.sum(), 1))
    recall = float((truth & pred_dilated).sum() / max(truth.sum(), 1))
    return 2 * precision * recall / max(precision + recall, 1e-12)


def binary_mask_metrics(prediction: np.ndarray, target: np.ndarray) -> BinaryMaskMetrics:
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    if pred.shape != truth.shape or pred.ndim != 2:
        raise ValueError("mask shape mismatch")
    intersection = int((pred & truth).sum())
    union = int((pred | truth).sum())
    pbox, tbox = mask_bbox(pred), mask_bbox(truth)
    pcenter = np.array([(pbox[0] + pbox[2]) / 2, (pbox[1] + pbox[3]) / 2])
    tcenter = np.array([(tbox[0] + tbox[2]) / 2, (tbox[1] + tbox[3]) / 2])
    diagonal = math.hypot(pred.shape[1], pred.shape[0])
    orientations = {
        "identity": pred,
        "flip_lr": np.fliplr(pred),
        "flip_ud": np.flipud(pred),
        "flip_both": np.flipud(np.fliplr(pred)),
    }
    orientation_iou = {
        name: float((value & truth).sum() / max((value | truth).sum(), 1))
        for name, value in orientations.items()
    }
    return BinaryMaskMetrics(
        iou=float(intersection / max(union, 1)),
        dice=float(2 * intersection / max(pred.sum() + truth.sum(), 1)),
        boundary_fscore=boundary_fscore(pred, truth),
        bbox_center_residual_diag=float(np.linalg.norm(pcenter - tcenter) / diagonal),
        bbox_scale_ratio=float(math.sqrt(pred.sum() / max(truth.sum(), 1))),
        foreground_area_ratio=float(pred.sum() / max(truth.sum(), 1)),
        top_residual_fraction=float(abs(pbox[1] - tbox[1]) / pred.shape[0]),
        bottom_residual_fraction=float(abs(pbox[3] - tbox[3]) / pred.shape[0]),
        best_orientation=max(orientation_iou, key=orientation_iou.get),
    )


def clean_body_outside_fraction(clean: np.ndarray, source: np.ndarray) -> float:
    body = np.asarray(clean, dtype=bool)
    shell = np.asarray(source, dtype=bool)
    if body.shape != shell.shape or not body.any():
        raise ValueError("invalid clean/source masks")
    return float((body & ~shell).sum() / body.sum())


def source_coverage_by_clean_body(clean: np.ndarray, source: np.ndarray) -> float:
    body = np.asarray(clean, dtype=bool)
    shell = np.asarray(source, dtype=bool)
    if body.shape != shell.shape or not body.any():
        raise ValueError("invalid clean/source masks")
    return float((body & shell).sum() / body.sum())


def validate_frame_mapping(records: list[Mapping[str, Any]], expected_ids: list[str]) -> None:
    if [row["condition_id"] for row in records] != expected_ids:
        raise ValueError("condition ID order/mapping differs from frozen protocol")
    for row in records:
        if int(row["source_frame"]) != int(row["pose_source_frame"]):
            raise ValueError(f"condition/source-frame mismatch: {row['condition_id']}")


def validate_rh_th_contract(record: Mapping[str, Any]) -> None:
    global_orient = np.asarray(record["global_orient"], dtype=np.float64)
    rendered_translation = np.asarray(record["transl_rendered"], dtype=np.float64)
    if not np.allclose(global_orient, 0) or not np.allclose(rendered_translation, 0):
        raise ValueError("synthetic source contract requires rendered global orient and translation to be zero")
    if record.get("apply_source_global_orient", False) or record.get("apply_source_translation", False):
        raise ValueError("source Rh/Th would be applied twice")


def adjudicate_geomcam(
    source_replay_pass: bool,
    pose_contract_pass: bool,
    correct_clean_nested_pass: bool,
    audited_r3_path_pass: bool,
    original_generator_available: bool,
) -> dict[str, Any]:
    if not original_generator_available and not source_replay_pass:
        case, status = "GC5", "BLOCKED_INSUFFICIENT_EVIDENCE"
    elif not source_replay_pass:
        case, status = "GC1", "CAMERA_OR_METADATA_CONTRACT_FAIL"
    elif not pose_contract_pass:
        case, status = "GC2", "POSE_FRAME_OR_TRANSFORM_MISMATCH"
    elif correct_clean_nested_pass and not audited_r3_path_pass:
        case, status = "GC2", "POSE_FRAME_OR_TRANSFORM_MISMATCH"
    elif correct_clean_nested_pass:
        case, status = "GC3", "OLD_GEOMETRY_GATE_INVALID"
    else:
        case, status = "GC4", "ACTUAL_CLEAN_BODY_SHAPE_MISMATCH"
    return {
        "root_cause_case": case,
        "status": status,
        "formal_adapter_change_allowed": False,
        "skin_shell_support_allowed": False,
        "module4b_allowed": False,
    }
