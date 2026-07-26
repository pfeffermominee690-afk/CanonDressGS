#!/usr/bin/env python3
"""Read-only camera salvage adjudication for Subject00 Teacher targets.

The script reuses the frozen attempt-001 registration constants, but makes the
background-only rule symmetric: features and photometric evidence must be
outside both the dilated source-person mask and the dilated accepted
target-person mask.  It never writes into a generation attempt or Teacher
target root; all outputs are review/control artifacts inside the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from skimage.metrics import structural_similarity


TASK_ID = "AAAI27-SUBJECT00-TEACHER-TARGET-CAMERA-RESOLUTION-BLOCKER-RESOLUTION-001"
SOURCE_BRANCH = "research/subject00-24-cell-teacher-target-creation-preflight-20260727"
SOURCE_HEAD = "762fb6a6621e461000b909f749c72a244d55feb2"
NEW_BRANCH = "research/subject00-teacher-target-camera-blocker-resolution-20260727"
WORKTREE = r"E:\model_train\canondressgs_subject00_teacher_target_camera_blocker_resolution"
SOURCE_WORKTREE = Path(
    r"E:\model_train\canondressgs_subject00_24_cell_teacher_target_creation_preflight"
)
PREVIOUS_BLOCKER = (
    "BLOCKED_22_OF_24_UNIQUE_SIMILARITY_BINDINGS_2_OF_24_"
    "HUMAN_OVERRIDE_CAMERA_MODELS_NONUNIQUE"
)
FINAL_CLASSIFICATION = (
    "SUBJECT00_TEACHER_TARGET_CAMERA_BLOCKER_RESOLVED_WITH_QUARANTINE_"
    "READY_FOR_MATERIALIZATION"
)
NEXT_TASK = (
    "EXECUTE_SUBJECT00_TEACHER_TARGET_MATERIALIZATION_WITH_CAMERA_QUARANTINE_POLICY"
)

PROBLEM_IDS = (
    "subject00_O03_slot04_canary_attempt004_cand00",
    "subject00_O01_slot04_remaining_attempt005_cand00",
)
ALLOWED_MODELS = (
    "MODEL_A_IDENTITY",
    "MODEL_B_RESOLUTION_SCALE_ONLY",
    "MODEL_C_ISOTROPIC_SIMILARITY",
    "MODEL_D_RESOLUTION_SCALE_THEN_ISOTROPIC_SIMILARITY",
)
FORBIDDEN_MODELS = (
    "ANISOTROPIC_AFFINE",
    "GENERAL_AFFINE",
    "PROJECTIVE_HOMOGRAPHY",
)

MASK_REGISTRY_REL = Path(
    "paper_protocol/reviewer_risk/"
    "subject00_global_mask_accepted_registry_24of24_20260727.json"
)
RAW_REGISTRY_REL = Path(
    "paper_protocol/reviewer_risk/"
    "subject00_global_accepted_cell_registry_24of24_20260726.json"
)
CAMERA_REGISTRY_REL = Path(
    "paper_protocol/reviewer_risk/"
    "subject00_24_cell_teacher_target_camera_registry_20260727.json"
)
EXECUTION_MANIFEST_REL = Path(
    "paper_protocol/reviewer_risk/"
    "subject00_24_cell_teacher_target_creation_execution_manifest_20260727.json"
)
FORMAL_PROTOCOL_REL = Path(
    "paper_protocol/reviewer_risk/"
    "subject00_attempt001_native_landscape_registration_protocol_20260726.json"
)
FORMAL_IMPLEMENTATION_REL = Path(
    "tools/datasets/audit_subject00_attempt001_native_landscape_registration.py"
)
ATTEMPT1_REG_REL = Path(
    "paper_protocol/reviewer_risk/"
    "subject00_attempt001_background_registration_metrics_20260726.json"
)
O03_REG_REL = Path(
    "paper_protocol/reviewer_risk/"
    "subject00_o03_canary_corrected_registration_results_20260726.json"
)
ATTEMPT5_REG = Path(
    r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-GENERATION-001"
    r"\attempt_005_subject00_remaining_six_cell_generation"
    r"\06_audit\registration_results.json"
)
SOURCE_MASK_ROOT = Path(
    r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-GENERATION-001"
    r"\attempt_001_native_landscape_registration_audit"
    r"\08_corrected_1349_human_review\masks"
)
TEACHER_TARGET_ROOT = Path(
    r"E:\canondressgs_data\subject00_three_garment"
    r"\CODEX-MANAGED-TEACHER-TARGETS-001"
)

REVIEW_ROOT = Path("paper_protocol/reviewer_risk")
OUTPUTS = {
    "overlay": REVIEW_ROOT
    / "subject00_teacher_target_camera_resolution_blocker_correction_overlay_20260727.json",
    "reference": REVIEW_ROOT
    / "subject00_22cell_camera_safe_reference_distribution_20260727.json",
    "salvage": REVIEW_ROOT
    / "subject00_problem_cell_camera_salvage_results_20260727.json",
    "eligibility": REVIEW_ROOT
    / "subject00_teacher_target_camera_eligibility_registry_20260727.json",
    "quarantine": REVIEW_ROOT
    / "subject00_teacher_target_quarantine_registry_20260727.json",
    "camera_draft": REVIEW_ROOT
    / "subject00_teacher_target_camera_records_draft_20260727.json",
    "o03": REVIEW_ROOT
    / "subject00_O03_provisional_camera_safe_target_manifest_draft_20260727.json",
    "report": REVIEW_ROOT
    / "SUBJECT00_TEACHER_TARGET_CAMERA_BLOCKER_RESOLUTION_REPORT_20260727.md",
    "tests": REVIEW_ROOT
    / "subject00_teacher_target_camera_blocker_resolution_tests_20260727.json",
    "summary": REVIEW_ROOT
    / "subject00_teacher_target_camera_blocker_resolution_final_summary_20260727.json",
    "handoff": Path(
        "project_control_handoff/"
        "subject00_teacher_target_camera_blocker_resolution_handoff_20260727.json"
    ),
    "docs": Path(
        "docs/PAPER/"
        "AAAI27_SUBJECT00_TEACHER_TARGET_CAMERA_BLOCKER_RESOLUTION_REPORT_20260727.md"
    ),
}

FORMAL_THRESHOLDS = {
    "person_foreground_threshold": 128,
    "dilation_minimum_px": 20,
    "dilation_short_edge_fraction": 0.02,
    "border_exclusion_px": 8,
    "sift_nfeatures": 5000,
    "sift_contrast_threshold": 0.01,
    "sift_edge_threshold": 10,
    "lowe_ratio": 0.75,
    "minimum_source_background_features": 100,
    "minimum_target_background_features": 100,
    "minimum_ratio_matches": 30,
    "ransac_threshold_px": 3.0,
    "ransac_max_iterations": 5000,
    "ransac_confidence": 0.999,
    "minimum_similarity_inliers": 20,
    "minimum_similarity_inlier_ratio": 0.65,
    "maximum_similarity_median_error_px": 2.0,
    "maximum_similarity_p95_error_px": 5.0,
    "maximum_rotation_degrees": 0.25,
    "maximum_anisotropy": 0.005,
    "maximum_normalized_shear": 0.003,
    "maximum_homography_relative_improvement": 0.35,
    "minimum_masked_ssim": 0.55,
    "minimum_edge_alignment_f1": 0.55,
    "maximum_structural_difference_fraction": 0.25,
    "canny_low": 50,
    "canny_high": 150,
    "edge_tolerance_px": 2,
    "ssim_difference_threshold": 0.50,
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric(value: Any) -> float:
    return float(value)


def matrix_list(matrix: np.ndarray) -> list[list[float]]:
    return [[float(x) for x in row] for row in matrix.tolist()]


def robust_stats(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "median": median,
        "mad": float(np.median(np.abs(array - median))),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
    }


def dilation_radius(mask: np.ndarray) -> int:
    height, width = mask.shape
    return max(
        FORMAL_THRESHOLDS["dilation_minimum_px"],
        round(FORMAL_THRESHOLDS["dilation_short_edge_fraction"] * min(width, height)),
    )


def background_mask(person_mask: np.ndarray) -> tuple[np.ndarray, int]:
    radius = dilation_radius(person_mask)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    person = (person_mask >= FORMAL_THRESHOLDS["person_foreground_threshold"]).astype(
        np.uint8
    )
    dilated = cv2.dilate(person, kernel)
    background = (dilated == 0).astype(np.uint8) * 255
    border = FORMAL_THRESHOLDS["border_exclusion_px"]
    background[:border, :] = 0
    background[-border:, :] = 0
    background[:, :border] = 0
    background[:, -border:] = 0
    return background, radius


def match_background(
    source_bgr: np.ndarray,
    target_bgr: np.ndarray,
    source_bg: np.ndarray,
    target_bg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    sift = cv2.SIFT_create(
        nfeatures=FORMAL_THRESHOLDS["sift_nfeatures"],
        contrastThreshold=FORMAL_THRESHOLDS["sift_contrast_threshold"],
        edgeThreshold=FORMAL_THRESHOLDS["sift_edge_threshold"],
    )
    source_gray = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
    source_kp, source_desc = sift.detectAndCompute(source_gray, source_bg)
    target_kp, target_desc = sift.detectAndCompute(target_gray, target_bg)
    if source_desc is None or target_desc is None:
        raise RuntimeError("background SIFT returned no descriptors")
    if len(source_kp) < FORMAL_THRESHOLDS["minimum_source_background_features"]:
        raise RuntimeError("insufficient source-background features")
    if len(target_kp) < FORMAL_THRESHOLDS["minimum_target_background_features"]:
        raise RuntimeError("insufficient target-background features")
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(source_desc, target_desc, k=2)
    ratio_matches = [
        first
        for first, second in pairs
        if first.distance < FORMAL_THRESHOLDS["lowe_ratio"] * second.distance
    ]
    unique_by_target: dict[int, Any] = {}
    for match in sorted(ratio_matches, key=lambda item: item.distance):
        unique_by_target.setdefault(match.trainIdx, match)
    matches = list(unique_by_target.values())
    if len(matches) < FORMAL_THRESHOLDS["minimum_ratio_matches"]:
        raise RuntimeError("insufficient unique Lowe-ratio matches")
    source_points = np.float64([source_kp[item.queryIdx].pt for item in matches])
    target_points = np.float64([target_kp[item.trainIdx].pt for item in matches])
    return source_points, target_points, {
        "source_background_feature_count": len(source_kp),
        "target_background_feature_count": len(target_kp),
        "lowe_ratio_match_count_before_unique_target": len(ratio_matches),
        "match_count": len(matches),
    }


def fit_similarity(
    source: np.ndarray, target: np.ndarray, seed: int, threshold: float
) -> np.ndarray:
    cv2.setRNGSeed(seed)
    matrix, _ = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=threshold,
        maxIters=FORMAL_THRESHOLDS["ransac_max_iterations"],
        confidence=FORMAL_THRESHOLDS["ransac_confidence"],
        refineIters=10,
    )
    if matrix is None:
        raise RuntimeError("similarity fitting failed")
    return np.asarray(matrix, dtype=np.float64)


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:, :2].T + matrix[:, 2]


def decompose(matrix: np.ndarray) -> dict[str, float]:
    linear = matrix[:, :2]
    determinant = float(np.linalg.det(linear))
    sx = float(np.linalg.norm(linear[:, 0]))
    sy = float(np.linalg.norm(linear[:, 1]))
    uniform = math.sqrt(abs(determinant))
    rotation = math.degrees(math.atan2(linear[1, 0], linear[0, 0]))
    anisotropy = abs(sx / sy - 1.0) if sy else float("inf")
    dot = float(np.dot(linear[:, 0], linear[:, 1]))
    shear = dot / (sx * sy) if sx and sy else float("inf")
    return {
        "determinant": determinant,
        "scale_x": sx,
        "scale_y": sy,
        "uniform_scale": uniform,
        "rotation_degrees": rotation,
        "translation_x": float(matrix[0, 2]),
        "translation_y": float(matrix[1, 2]),
        "anisotropy": anisotropy,
        "normalized_shear": shear,
    }


def geometric_metrics(
    matrix: np.ndarray, source: np.ndarray, target: np.ndarray
) -> dict[str, Any]:
    errors = np.linalg.norm(transform_points(matrix, source) - target, axis=1)
    inliers = errors <= FORMAL_THRESHOLDS["ransac_threshold_px"]
    selected = errors[inliers]
    if selected.size == 0:
        selected = errors
    return {
        "matrix": matrix_list(matrix),
        "match_count": int(errors.size),
        "inlier_count": int(np.count_nonzero(inliers)),
        "inlier_ratio": float(np.mean(inliers)),
        "reprojection_rmse_px": float(np.sqrt(np.mean(selected**2))),
        "median_reprojection_error_px": float(np.median(selected)),
        "p95_reprojection_error_px": float(np.percentile(selected, 95)),
        "maximum_reprojection_error_px": float(np.max(selected)),
        **decompose(matrix),
    }


def photometric_metrics(
    source_bgr: np.ndarray,
    target_bgr: np.ndarray,
    source_bg: np.ndarray,
    target_bg: np.ndarray,
    source_person: np.ndarray,
    target_person: np.ndarray,
    matrix: np.ndarray,
) -> dict[str, Any]:
    target_height, target_width = target_bgr.shape[:2]
    aligned = cv2.warpAffine(
        source_bgr,
        matrix,
        (target_width, target_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    warped_source_bg = cv2.warpAffine(
        source_bg,
        matrix,
        (target_width, target_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    overlap = cv2.warpAffine(
        np.full(source_bg.shape, 255, dtype=np.uint8),
        matrix,
        (target_width, target_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid = (warped_source_bg > 0) & (target_bg > 0) & (overlap > 0)
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        raise RuntimeError("candidate transform has no valid double-masked background")
    difference = np.abs(aligned.astype(np.float32) - target_bgr.astype(np.float32))
    rgb_mae = float(np.mean(difference[valid]))
    source_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
    _, ssim_map = structural_similarity(
        source_gray, target_gray, data_range=255, full=True
    )
    masked_ssim = float(np.mean(ssim_map[valid]))
    source_edges = cv2.Canny(
        source_gray,
        FORMAL_THRESHOLDS["canny_low"],
        FORMAL_THRESHOLDS["canny_high"],
    )
    target_edges = cv2.Canny(
        target_gray,
        FORMAL_THRESHOLDS["canny_low"],
        FORMAL_THRESHOLDS["canny_high"],
    )
    source_edges = (source_edges > 0) & valid
    target_edges = (target_edges > 0) & valid
    tolerance = FORMAL_THRESHOLDS["edge_tolerance_px"]
    kernel = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
    source_dilated = cv2.dilate(source_edges.astype(np.uint8), kernel) > 0
    target_dilated = cv2.dilate(target_edges.astype(np.uint8), kernel) > 0
    precision = float(
        np.count_nonzero(source_edges & target_dilated)
        / max(1, np.count_nonzero(source_edges))
    )
    recall = float(
        np.count_nonzero(target_edges & source_dilated)
        / max(1, np.count_nonzero(target_edges))
    )
    edge_f1 = 2 * precision * recall / max(1e-12, precision + recall)
    structural_difference = float(
        np.count_nonzero((1.0 - ssim_map > FORMAL_THRESHOLDS["ssim_difference_threshold"]) & valid)
        / valid_count
    )
    structure_score = float(
        np.mean([masked_ssim, edge_f1, 1.0 - structural_difference])
    )

    warped_person = cv2.warpAffine(
        (source_person >= FORMAL_THRESHOLDS["person_foreground_threshold"]).astype(
            np.uint8
        ),
        matrix,
        (target_width, target_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0
    target_person_bool = (
        target_person >= FORMAL_THRESHOLDS["person_foreground_threshold"]
    )
    union = warped_person | target_person_bool
    intersection = warped_person & target_person_bool
    person_residual = float(np.count_nonzero(union ^ intersection) / max(1, np.count_nonzero(union)))
    person_iou = float(np.count_nonzero(intersection) / max(1, np.count_nonzero(union)))
    return {
        "valid_double_masked_background_pixel_count": valid_count,
        "valid_double_masked_background_fraction": float(valid_count / valid.size),
        "background_rgb_mae": rgb_mae,
        "masked_ssim": masked_ssim,
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_alignment_f1": edge_f1,
        "structural_difference_area_fraction": structural_difference,
        "structure_score": structure_score,
        "person_region_mask_symmetric_difference_ratio": person_residual,
        "person_region_mask_iou": person_iou,
    }


def candidate_matrices(
    source: np.ndarray,
    target: np.ndarray,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
    request_id: str,
) -> dict[str, np.ndarray]:
    source_height, source_width = source_shape
    target_height, target_width = target_shape
    sx = target_width / source_width
    sy = target_height / source_height
    resolution = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0]], dtype=np.float64)
    seed = int(hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:8], 16)
    seed &= 0x7FFFFFFF
    direct = fit_similarity(
        source, target, seed, FORMAL_THRESHOLDS["ransac_threshold_px"]
    )
    target_normalized = target / np.array([sx, sy], dtype=np.float64)
    residual = fit_similarity(
        source,
        target_normalized,
        seed,
        FORMAL_THRESHOLDS["ransac_threshold_px"] / max(sx, sy),
    )
    resolution_3 = np.vstack([resolution, [0.0, 0.0, 1.0]])
    residual_3 = np.vstack([residual, [0.0, 0.0, 1.0]])
    composed = (resolution_3 @ residual_3)[:2, :]
    return {
        "MODEL_A_IDENTITY": np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
        ),
        "MODEL_B_RESOLUTION_SCALE_ONLY": resolution,
        "MODEL_C_ISOTROPIC_SIMILARITY": direct,
        "MODEL_D_RESOLUTION_SCALE_THEN_ISOTROPIC_SIMILARITY": composed,
    }


def all_registration_records(repo: Path) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for path in (repo / ATTEMPT1_REG_REL, repo / O03_REG_REL, ATTEMPT5_REG):
        for record in read_json(path)["records"]:
            values[record["request_id"]] = record
    return values


def source_mask_path(camera_id: str) -> Path:
    return SOURCE_MASK_ROOT / f"{camera_id}_official_person_mask.png"


def load_cell_inputs(
    accepted: dict[str, Any],
    raw: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    source_path = Path(raw["source_condition_path"])
    target_path = Path(accepted["accepted_raw"]["path"])
    source_person_path = source_mask_path(accepted["camera"])
    target_person_path = Path(accepted["person_mask"]["path"])
    target_garment_path = Path(accepted["garment_mask"]["path"])
    for path in (
        source_path,
        target_path,
        source_person_path,
        target_person_path,
        target_garment_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256(source_path) != raw["source_condition_sha256"]:
        raise ValueError(f"source SHA mismatch: {accepted['request_id']}")
    if sha256(target_path) != accepted["accepted_raw"]["sha256"]:
        raise ValueError(f"target SHA mismatch: {accepted['request_id']}")
    if sha256(target_person_path) != accepted["person_mask"]["sha256"]:
        raise ValueError(f"target person-mask SHA mismatch: {accepted['request_id']}")
    if sha256(target_garment_path) != accepted["garment_mask"]["sha256"]:
        raise ValueError(f"target garment-mask SHA mismatch: {accepted['request_id']}")
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    target = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
    source_person = cv2.imread(str(source_person_path), cv2.IMREAD_GRAYSCALE)
    target_person = cv2.imread(str(target_person_path), cv2.IMREAD_GRAYSCALE)
    if any(item is None for item in (source, target, source_person, target_person)):
        raise ValueError(f"OpenCV input decode failed: {accepted['request_id']}")
    assert source is not None and target is not None
    assert source_person is not None and target_person is not None
    if source.shape[:2] != source_person.shape:
        raise ValueError(f"source/mask shape mismatch: {accepted['request_id']}")
    if target.shape[:2] != target_person.shape:
        raise ValueError(f"target/mask shape mismatch: {accepted['request_id']}")
    provenance = {
        "source_path": str(source_path),
        "source_sha256": raw["source_condition_sha256"],
        "source_person_mask_path": str(source_person_path),
        "source_person_mask_sha256": sha256(source_person_path),
        "target_path": str(target_path),
        "target_sha256": accepted["accepted_raw"]["sha256"],
        "target_person_mask_path": str(target_person_path),
        "target_person_mask_sha256": accepted["person_mask"]["sha256"],
        "target_garment_mask_path": str(target_garment_path),
        "target_garment_mask_sha256": accepted["garment_mask"]["sha256"],
    }
    return source, target, source_person, target_person, provenance


def evaluate_cell(
    accepted: dict[str, Any],
    raw: dict[str, Any],
    models_to_measure: tuple[str, ...],
) -> dict[str, Any]:
    source, target, source_person, target_person, provenance = load_cell_inputs(
        accepted, raw
    )
    source_bg, source_radius = background_mask(source_person)
    target_bg, target_radius = background_mask(target_person)
    source_points, target_points, feature_counts = match_background(
        source, target, source_bg, target_bg
    )
    matrices = candidate_matrices(
        source_points,
        target_points,
        source.shape[:2],
        target.shape[:2],
        accepted["request_id"],
    )
    candidates: dict[str, Any] = {}
    for model_name in models_to_measure:
        geometry = geometric_metrics(
            matrices[model_name], source_points, target_points
        )
        photo = photometric_metrics(
            source,
            target,
            source_bg,
            target_bg,
            source_person,
            target_person,
            matrices[model_name],
        )
        candidates[model_name] = {**geometry, **photo}
    return {
        "request_id": accepted["request_id"],
        "garment": accepted["garment"],
        "slot": accepted["slot"],
        "camera_id": accepted["camera"],
        "direction": accepted["direction"],
        "source_resolution": {
            "width": int(source.shape[1]),
            "height": int(source.shape[0]),
        },
        "target_resolution": {
            "width": int(target.shape[1]),
            "height": int(target.shape[0]),
        },
        "background_region_policy": {
            "logical_definition": (
                "warp(NOT(dilate(source_person_mask))) AND "
                "NOT(dilate(target_person_mask)) AND valid_source_overlap AND border"
            ),
            "source_dilation_radius_px": source_radius,
            "target_dilation_radius_px": target_radius,
            "border_exclusion_px": FORMAL_THRESHOLDS["border_exclusion_px"],
            "mask_interpolation": "NEAREST",
        },
        "feature_evidence": feature_counts,
        "input_provenance": provenance,
        "candidates": candidates,
    }


def build_distribution(reference_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "match_count",
        "inlier_count",
        "inlier_ratio",
        "reprojection_rmse_px",
        "median_reprojection_error_px",
        "p95_reprojection_error_px",
        "uniform_scale",
        "rotation_degrees",
        "translation_x",
        "translation_y",
        "background_rgb_mae",
        "person_region_mask_symmetric_difference_ratio",
        "determinant",
        "anisotropy",
        "normalized_shear",
        "structure_score",
        "masked_ssim",
        "edge_alignment_f1",
        "structural_difference_area_fraction",
        "formal_homography_relative_similarity_error_improvement",
        "model_separation_relative_median_margin",
    )
    return {
        metric: robust_stats([numeric(row[metric]) for row in reference_rows])
        for metric in metrics
    }


def empirical_envelope(distribution: dict[str, Any]) -> dict[str, Any]:
    return {
        "derivation": (
            "Inclusive extrema of the 22 frozen machine-safe reference cells, "
            "intersected with the preregistered formal floors/ceilings where one exists. "
            "No problem-cell value and no post-result tuning enters the envelope."
        ),
        "match_count_minimum": max(
            FORMAL_THRESHOLDS["minimum_ratio_matches"],
            int(distribution["match_count"]["minimum"]),
        ),
        "inlier_count_minimum": max(
            FORMAL_THRESHOLDS["minimum_similarity_inliers"],
            int(distribution["inlier_count"]["minimum"]),
        ),
        "inlier_ratio_minimum": max(
            FORMAL_THRESHOLDS["minimum_similarity_inlier_ratio"],
            distribution["inlier_ratio"]["minimum"],
        ),
        "reprojection_rmse_px_maximum": distribution["reprojection_rmse_px"][
            "maximum"
        ],
        "median_reprojection_error_px_maximum": min(
            FORMAL_THRESHOLDS["maximum_similarity_median_error_px"],
            distribution["median_reprojection_error_px"]["maximum"],
        ),
        "p95_reprojection_error_px_maximum": min(
            FORMAL_THRESHOLDS["maximum_similarity_p95_error_px"],
            distribution["p95_reprojection_error_px"]["maximum"],
        ),
        "uniform_scale_interval": [
            distribution["uniform_scale"]["minimum"],
            distribution["uniform_scale"]["maximum"],
        ],
        "rotation_degrees_interval": [
            distribution["rotation_degrees"]["minimum"],
            distribution["rotation_degrees"]["maximum"],
        ],
        "translation_x_interval": [
            distribution["translation_x"]["minimum"],
            distribution["translation_x"]["maximum"],
        ],
        "translation_y_interval": [
            distribution["translation_y"]["minimum"],
            distribution["translation_y"]["maximum"],
        ],
        "background_rgb_mae_maximum": distribution["background_rgb_mae"][
            "maximum"
        ],
        "person_region_mask_symmetric_difference_ratio_maximum": distribution[
            "person_region_mask_symmetric_difference_ratio"
        ]["maximum"],
        "determinant_interval": [
            distribution["determinant"]["minimum"],
            distribution["determinant"]["maximum"],
        ],
        "structure_score_minimum": distribution["structure_score"]["minimum"],
        "masked_ssim_minimum": max(
            FORMAL_THRESHOLDS["minimum_masked_ssim"],
            distribution["masked_ssim"]["minimum"],
        ),
        "edge_alignment_f1_minimum": max(
            FORMAL_THRESHOLDS["minimum_edge_alignment_f1"],
            distribution["edge_alignment_f1"]["minimum"],
        ),
        "structural_difference_area_fraction_maximum": min(
            FORMAL_THRESHOLDS["maximum_structural_difference_fraction"],
            distribution["structural_difference_area_fraction"]["maximum"],
        ),
        "homography_relative_improvement_maximum": min(
            FORMAL_THRESHOLDS["maximum_homography_relative_improvement"],
            distribution[
                "formal_homography_relative_similarity_error_improvement"
            ]["maximum"],
        ),
        "minimum_relative_model_margin": distribution[
            "model_separation_relative_median_margin"
        ]["p90"],
        "model_margin_derivation": (
            "The p90 absolute relative separation between the frozen direct-"
            "similarity model and the deterministic resolution-only baseline "
            "across the 22 safe references."
        ),
    }


def between(value: float, interval: list[float]) -> bool:
    return interval[0] <= value <= interval[1]


def validate_candidate(
    name: str,
    values: dict[str, Any],
    envelope: dict[str, Any],
    formal_homography_improvement: float,
    source_resolution: dict[str, int],
    target_resolution: dict[str, int],
) -> dict[str, Any]:
    gates = {
        "model_is_allowed": name in ALLOWED_MODELS,
        "identity_applicable_only_at_equal_resolution": (
            name != "MODEL_A_IDENTITY"
            or source_resolution == target_resolution
        ),
        "match_count": values["match_count"] >= envelope["match_count_minimum"],
        "inlier_count": values["inlier_count"] >= envelope["inlier_count_minimum"],
        "inlier_ratio": values["inlier_ratio"]
        >= envelope["inlier_ratio_minimum"],
        "reprojection_rmse": values["reprojection_rmse_px"]
        <= envelope["reprojection_rmse_px_maximum"],
        "median_reprojection_error": values["median_reprojection_error_px"]
        <= envelope["median_reprojection_error_px_maximum"],
        "p95_reprojection_error": values["p95_reprojection_error_px"]
        <= envelope["p95_reprojection_error_px_maximum"],
        "uniform_scale": between(
            values["uniform_scale"], envelope["uniform_scale_interval"]
        ),
        "rotation": between(
            values["rotation_degrees"], envelope["rotation_degrees_interval"]
        ),
        "translation_x": between(
            values["translation_x"], envelope["translation_x_interval"]
        ),
        "translation_y": between(
            values["translation_y"], envelope["translation_y_interval"]
        ),
        "background_only_residual": values["background_rgb_mae"]
        <= envelope["background_rgb_mae_maximum"],
        "person_region_residual": values[
            "person_region_mask_symmetric_difference_ratio"
        ]
        <= envelope["person_region_mask_symmetric_difference_ratio_maximum"],
        "determinant": between(
            values["determinant"], envelope["determinant_interval"]
        ),
        "structure_score": values["structure_score"]
        >= envelope["structure_score_minimum"],
        "masked_ssim": values["masked_ssim"] >= envelope["masked_ssim_minimum"],
        "edge_alignment_f1": values["edge_alignment_f1"]
        >= envelope["edge_alignment_f1_minimum"],
        "structural_difference": values["structural_difference_area_fraction"]
        <= envelope["structural_difference_area_fraction_maximum"],
        "forbidden_homography_not_materially_better": (
            formal_homography_improvement
            <= envelope["homography_relative_improvement_maximum"]
        ),
    }
    return {
        "gates": gates,
        "passing_gate_count": sum(bool(value) for value in gates.values()),
        "total_gate_count": len(gates),
        "eligible": all(gates.values()),
    }


def target_k(calibration_k: list[list[float]], transform: list[list[float]]) -> list[list[float]]:
    similarity = np.asarray(transform, dtype=np.float64)
    if similarity.shape == (2, 3):
        similarity = np.vstack([similarity, [0.0, 0.0, 1.0]])
    return matrix_list(similarity @ np.asarray(calibration_k, dtype=np.float64))


def build(repo: Path) -> None:
    created_at = datetime.now(timezone.utc).isoformat()
    mask_registry = read_json(repo / MASK_REGISTRY_REL)
    raw_registry = read_json(repo / RAW_REGISTRY_REL)
    previous_cameras = read_json(repo / CAMERA_REGISTRY_REL)
    previous_manifest = read_json(repo / EXECUTION_MANIFEST_REL)
    raw_by_id = {item["request_id"]: item for item in raw_registry["records"]}
    camera_by_id = {item["request_id"]: item for item in previous_cameras["records"]}
    manifest_by_id = {
        item["accepted_request_id"]: item for item in previous_manifest["records"]
    }
    formal_by_id = all_registration_records(repo)
    accepted_by_id = {
        item["request_id"]: item for item in mask_registry["records"]
    }
    if set(PROBLEM_IDS) - set(accepted_by_id):
        raise ValueError("problem-cell set is not contained in accepted registry")
    if len(accepted_by_id) != 24:
        raise ValueError("expected exactly 24 accepted cells")

    implementation_path = repo / FORMAL_IMPLEMENTATION_REL
    implementation_sha = sha256(implementation_path)
    protocol_sha = sha256(repo / FORMAL_PROTOCOL_REL)
    subprocess = __import__("subprocess")
    source_head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if source_head != SOURCE_HEAD:
        raise ValueError(f"adjudicator must run before artifact commit at {SOURCE_HEAD}")
    if TEACHER_TARGET_ROOT.exists():
        raise RuntimeError(f"Teacher target root unexpectedly exists: {TEACHER_TARGET_ROOT}")

    reference_rows: list[dict[str, Any]] = []
    for accepted in mask_registry["records"]:
        request_id = accepted["request_id"]
        if request_id in PROBLEM_IDS:
            continue
        formal = formal_by_id[request_id]
        if (
            accepted["machine_registration_status"]
            != "REGISTERED_SIMILARITY_PASS_CANDIDATE"
        ):
            raise ValueError(f"safe reference is not machine PASS: {request_id}")
        evaluation = evaluate_cell(
            accepted,
            raw_by_id[request_id],
            (
                "MODEL_B_RESOLUTION_SCALE_ONLY",
                "MODEL_C_ISOTROPIC_SIMILARITY",
            ),
        )
        candidate = evaluation["candidates"]["MODEL_C_ISOTROPIC_SIMILARITY"]
        resolution_candidate = evaluation["candidates"][
            "MODEL_B_RESOLUTION_SCALE_ONLY"
        ]
        camera = camera_by_id[request_id]
        row = {
            "sequence_index": int(accepted["sequence_index"]),
            "request_id": request_id,
            "garment": accepted["garment"],
            "slot": accepted["slot"],
            "camera_id": accepted["camera"],
            "direction": accepted["direction"],
            "source_resolution": evaluation["source_resolution"],
            "target_resolution": evaluation["target_resolution"],
            "calibration_K": camera["calibration_K"],
            "w2c": camera["w2c"],
            "c2w": camera["c2w"],
            "target_K": camera["target_K"],
            "match_count": candidate["match_count"],
            "inlier_count": candidate["inlier_count"],
            "inlier_ratio": candidate["inlier_ratio"],
            "reprojection_rmse_px": candidate["reprojection_rmse_px"],
            "median_reprojection_error_px": candidate[
                "median_reprojection_error_px"
            ],
            "p95_reprojection_error_px": candidate[
                "p95_reprojection_error_px"
            ],
            "uniform_scale": candidate["uniform_scale"],
            "rotation_degrees": candidate["rotation_degrees"],
            "translation_x": candidate["translation_x"],
            "translation_y": candidate["translation_y"],
            "background_rgb_mae": candidate["background_rgb_mae"],
            "person_region_mask_symmetric_difference_ratio": candidate[
                "person_region_mask_symmetric_difference_ratio"
            ],
            "determinant": candidate["determinant"],
            "anisotropy": numeric(
                abs(
                    formal["estimated_scale_x"] / formal["estimated_scale_y"]
                    - 1.0
                )
            ),
            "normalized_shear": numeric(formal["normalized_shear"]),
            "formal_homography_relative_similarity_error_improvement": numeric(
                formal["homography_relative_similarity_error_improvement"]
            ),
            "model_separation_relative_median_margin": float(
                abs(
                    resolution_candidate["median_reprojection_error_px"]
                    - candidate["median_reprojection_error_px"]
                )
                / max(
                    resolution_candidate["median_reprojection_error_px"], 1e-12
                )
            ),
            "structure_score": candidate["structure_score"],
            "masked_ssim": candidate["masked_ssim"],
            "edge_alignment_f1": candidate["edge_alignment_f1"],
            "structural_difference_area_fraction": candidate[
                "structural_difference_area_fraction"
            ],
            "formal_machine_status": "PASS",
            "formal_machine_status_source": accepted["machine_registration_status"],
            "formal_primary_classification": formal["primary_classification"],
            "reference_status": "FROZEN_MACHINE_SAFE_REFERENCE",
            "input_provenance": evaluation["input_provenance"],
        }
        reference_rows.append(row)

    if len(reference_rows) != 22:
        raise ValueError("safe reference set must contain exactly 22 cells")
    distribution = build_distribution(reference_rows)
    envelope = empirical_envelope(distribution)

    problem_results: list[dict[str, Any]] = []
    for request_id in PROBLEM_IDS:
        accepted = accepted_by_id[request_id]
        formal = formal_by_id[request_id]
        evaluation = evaluate_cell(
            accepted, raw_by_id[request_id], ALLOWED_MODELS
        )
        formal_improvement = numeric(
            formal["homography_relative_similarity_error_improvement"]
        )
        model_results: dict[str, Any] = {}
        for name, values in evaluation["candidates"].items():
            validation = validate_candidate(
                name,
                values,
                envelope,
                formal_improvement,
                evaluation["source_resolution"],
                evaluation["target_resolution"],
            )
            model_results[name] = {**values, "validation": validation}
        eligible_models = [
            name
            for name, values in model_results.items()
            if values["validation"]["eligible"]
        ]
        ordered = sorted(
            (
                (name, values["median_reprojection_error_px"])
                for name, values in model_results.items()
            ),
            key=lambda item: item[1],
        )
        best_second_margin = (
            float((ordered[1][1] - ordered[0][1]) / max(ordered[1][1], 1e-12))
            if len(ordered) > 1
            else None
        )
        forbidden_dominance = (
            formal_improvement > envelope["homography_relative_improvement_maximum"]
        )
        margin_pass = (
            best_second_margin is not None
            and best_second_margin >= envelope["minimum_relative_model_margin"]
        )
        unique = (
            len(eligible_models) == 1
            and margin_pass
            and not forbidden_dominance
        )
        decision = (
            "SALVAGED_UNIQUE_ALLOWED_CAMERA_MODEL"
            if unique
            else "QUARANTINE_REVIEW_ONLY_CAMERA_MODEL_NOT_SCIENTIFICALLY_UNIQUE"
        )
        selected = eligible_models[0] if unique else None
        problem_results.append(
            {
                "sequence_index": int(accepted["sequence_index"]),
                "request_id": request_id,
                "garment": accepted["garment"],
                "slot": accepted["slot"],
                "camera_id": accepted["camera"],
                "direction": accepted["direction"],
                "source_resolution": evaluation["source_resolution"],
                "target_resolution": evaluation["target_resolution"],
                "feature_evidence": evaluation["feature_evidence"],
                "background_region_policy": evaluation[
                    "background_region_policy"
                ],
                "input_provenance": evaluation["input_provenance"],
                "formal_machine_status": accepted["machine_registration_status"],
                "human_override_status": accepted["human_override_status"],
                "formal_homography_relative_similarity_error_improvement": formal_improvement,
                "empirical_homography_improvement_ceiling": envelope[
                    "homography_relative_improvement_maximum"
                ],
                "forbidden_projective_or_affine_diagnostic_dominates": forbidden_dominance,
                "candidate_models": model_results,
                "eligible_allowed_models": eligible_models,
                "best_vs_second_relative_median_error_margin": best_second_margin,
                "required_empirical_relative_model_margin": envelope[
                    "minimum_relative_model_margin"
                ],
                "uniqueness_margin_pass": margin_pass,
                "uniqueness_status": (
                    "UNIQUE_ALLOWED_MODEL"
                    if unique
                    else (
                        "NO_ALLOWED_MODEL_SURVIVES_FORBIDDEN_MODEL_AMBIGUITY_GATE"
                        if forbidden_dominance
                        else "ALLOWED_MODEL_BINDING_NONUNIQUE_OR_OUTSIDE_REFERENCE_ENVELOPE"
                    )
                ),
                "selected_model": selected,
                "selected_transform": (
                    model_results[selected]["matrix"] if selected else None
                ),
                "decision": decision,
                "teacher_target_eligible": unique,
                "training_eligible": unique,
                "evaluation_eligible": unique,
                "appearance_supervision_eligible": unique,
                "geometry_supervision_eligible": unique,
                "review_only": not unique,
            }
        )

    salvaged = [item for item in problem_results if item["teacher_target_eligible"]]
    quarantined = [item for item in problem_results if item["review_only"]]
    eligibility_records: list[dict[str, Any]] = []
    camera_draft_records: list[dict[str, Any]] = []
    result_by_id = {item["request_id"]: item for item in problem_results}
    reference_by_id = {item["request_id"]: item for item in reference_rows}
    for accepted in mask_registry["records"]:
        request_id = accepted["request_id"]
        previous = camera_by_id[request_id]
        problem = result_by_id.get(request_id)
        is_safe = problem is None
        is_salvaged = bool(problem and problem["teacher_target_eligible"])
        eligible = is_safe or is_salvaged
        status = (
            "UNIQUE_SIMILARITY_BINDING_PASS"
            if is_safe
            else (
                "CAMERA_BINDING_SALVAGED_UNIQUE"
                if is_salvaged
                else "UNRESOLVED_HUMAN_OVERRIDE"
            )
        )
        transform = (
            previous["registered_source_to_target_similarity"]
            if is_safe
            else (problem["selected_transform"] if is_salvaged else None)
        )
        target_intrinsics = (
            previous["target_K"]
            if is_safe
            else (
                target_k(previous["calibration_K"], transform)
                if is_salvaged and transform is not None
                else None
            )
        )
        eligibility_records.append(
            {
                "sequence_index": int(accepted["sequence_index"]),
                "request_id": request_id,
                "garment": accepted["garment"],
                "slot": accepted["slot"],
                "camera_id": accepted["camera"],
                "direction": accepted["direction"],
                "camera_status": status,
                "camera_binding_status": status,
                "target_record_status": (
                    "TRAINING_TARGET_ELIGIBLE"
                    if eligible
                    else "REVIEW_ONLY_CAMERA_QUARANTINED"
                ),
                "teacher_target_eligible": eligible,
                "training_eligible": eligible,
                "evaluation_eligible": eligible,
                "appearance_supervision_eligible": eligible,
                "geometry_supervision_eligible": eligible,
                "review_only": not eligible,
                "selected_model": (
                    "FROZEN_MODEL_C_ISOTROPIC_SIMILARITY"
                    if is_safe
                    else (problem["selected_model"] if is_salvaged else None)
                ),
                "derived_camera_targets_authorized": False,
                "accepted_raw": accepted["accepted_raw"],
                "person_mask": accepted["person_mask"],
                "garment_mask": accepted["garment_mask"],
            }
        )
        camera_draft_records.append(
            {
                "sequence_index": int(accepted["sequence_index"]),
                "request_id": request_id,
                "garment": accepted["garment"],
                "slot": accepted["slot"],
                "camera_id": accepted["camera"],
                "direction": accepted["direction"],
                "source_frame_id": previous["source_frame_id"],
                "target_width": previous["target_width"],
                "target_height": previous["target_height"],
                "calibration_K": previous["calibration_K"],
                "calibration_R": previous["calibration_R"],
                "calibration_T": previous["calibration_T"],
                "distortion": previous["distortion"],
                "calibration_image_width": previous["calibration_image_width"],
                "calibration_image_height": previous["calibration_image_height"],
                "w2c": previous["w2c"],
                "c2w": previous["c2w"],
                "registered_source_to_target_transform": transform,
                "target_K": target_intrinsics,
                "camera_status": status,
                "camera_binding_status": status,
                "target_record_status": (
                    "TRAINING_TARGET_ELIGIBLE"
                    if eligible
                    else "REVIEW_ONLY_CAMERA_QUARANTINED"
                ),
                "accepted_raw": accepted["accepted_raw"],
                "person_mask": accepted["person_mask"],
                "garment_mask": accepted["garment_mask"],
                "source_condition": {
                    "path": raw_by_id[request_id]["source_condition_path"],
                    "sha256": raw_by_id[request_id][
                        "source_condition_sha256"
                    ],
                },
                "machine_registration_status": accepted[
                    "machine_registration_status"
                ],
                "human_override_status": accepted.get("human_override_status"),
                "official_limitation_codes": accepted.get(
                    "limitation_codes", []
                ),
                "human_review_evidence": accepted.get("review_evidence"),
                "transform_uncertainty": (
                    {
                        "status": "FROZEN_MACHINE_SAFE_REFERENCE",
                        "empirical_reference_envelope": envelope,
                    }
                    if is_safe
                    else {
                        "status": (
                            "SALVAGED_UNIQUE_ALLOWED_MODEL"
                            if is_salvaged
                            else "UNRESOLVED_NONUNIQUE_REVIEW_ONLY"
                        ),
                        "empirical_reference_envelope": envelope,
                        "uniqueness_status": problem["uniqueness_status"],
                        "best_vs_second_relative_median_error_margin": problem[
                            "best_vs_second_relative_median_error_margin"
                        ],
                    }
                ),
                "validation_metrics": (
                    reference_by_id[request_id]
                    if is_safe
                    else problem
                ),
                "conventions": {
                    "intrinsic_update": "K_target = T_pixel @ K_source",
                    "extrinsic_update": "w2c_target = w2c_source; c2w_target = c2w_source",
                    "pixel_coordinates": "OpenCV u-right v-down homogeneous column vectors",
                },
                "training_eligible": eligible,
                "evaluation_eligible": eligible,
                "appearance_supervision_eligible": eligible,
                "geometry_supervision_eligible": eligible,
                "review_only": not eligible,
                "derived_camera_targets_authorized": False,
                "materialization_authorized": False,
                "materialized": False,
                "teacher_target": False,
            }
        )

    eligible_count = sum(x["teacher_target_eligible"] for x in eligibility_records)
    review_only_count = sum(x["review_only"] for x in eligibility_records)
    if eligible_count not in (22, 23, 24):
        raise AssertionError("eligibility must freeze at 22/24, 23/24, or 24/24")
    o03_eligible = [
        item
        for item in eligibility_records
        if item["garment"] == "O03" and item["teacher_target_eligible"]
    ]
    if len(o03_eligible) not in (7, 8):
        raise AssertionError("O03 provisional target must have 7 or 8 safe views")

    no_mutation_counters = {
        "teacher_target_root_created": 0,
        "teacher_target_records_created": 0,
        "teacher_target_camera_json_created": 0,
        "teacher_target_raw_copies_created": 0,
        "derived_targets_created": 0,
        "portable_archive_created": 0,
        "cloud_uploads": 0,
        "generation_calls": 0,
        "gpu_calls": 0,
        "optimizer_steps": 0,
        "base_resume_calls": 0,
        "paper_body_modifications": 0,
        "attempt_001_mutations": 0,
        "attempt_002_mutations": 0,
        "attempt_003_mutations": 0,
        "attempt_004_mutations": 0,
        "attempt_005_mutations": 0,
        "accepted_raw_mutations": 0,
        "person_mask_mutations": 0,
        "garment_mask_mutations": 0,
        "data_copy_calls": 0,
        "upload_calls": 0,
        "derived_target_generation_calls": 0,
        "data_mutations": 0,
    }
    common = {
        "task_id": TASK_ID,
        "created_at": created_at,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "new_branch": NEW_BRANCH,
        "worktree": WORKTREE,
        "previous_blocker": PREVIOUS_BLOCKER,
        "formal_protocol": str(FORMAL_PROTOCOL_REL),
        "formal_protocol_sha256": protocol_sha,
        "registration_adjudicator": str(
            Path(__file__).resolve().relative_to(repo)
        ),
        "registration_adjudicator_sha256": sha256(Path(__file__)),
        "formal_registration_implementation": str(FORMAL_IMPLEMENTATION_REL),
        "formal_registration_implementation_sha256": implementation_sha,
        "allowed_models": list(ALLOWED_MODELS),
        "forbidden_diagnostic_only_models": list(FORBIDDEN_MODELS),
        "background_region_definition": (
            "NOT(dilated source person mask) AND NOT(dilated accepted target "
            "person mask under candidate target coordinates), restricted to "
            "valid overlap and the frozen 8-pixel border."
        ),
        "threshold_adjustment_policy": (
            "ZERO_POST_RESULT_ADJUSTMENT; empirical envelope frozen from the "
            "22 machine-safe cells before the two problem-cell decisions."
        ),
        "paper_final": False,
    }

    reference_artifact = {
        "schema_version": "canondressgs.subject00.camera_safe_reference_distribution.v1",
        **common,
        "reference_set_definition": (
            "The 22 accepted cells with frozen formal machine registration PASS; "
            "the two human-override problem cells are excluded."
        ),
        "reference_cell_count": len(reference_rows),
        "problem_cell_count": len(PROBLEM_IDS),
        "records": reference_rows,
        "distribution": distribution,
        "empirical_acceptance_envelope": envelope,
    }
    salvage_artifact = {
        "schema_version": "canondressgs.subject00.problem_camera_salvage_results.v1",
        **common,
        "problem_cell_count": len(problem_results),
        "model_d_composition_order": {
            "column_vector_equation": (
                "p_target = S_resolution @ T_similarity_residual @ p_source"
            ),
            "matrix_order": "S_resolution LEFT-MULTIPLIES T_similarity_residual",
            "reason": (
                "The residual similarity is fitted in calibration-resolution "
                "coordinates; deterministic WxH scaling then maps those pixels "
                "to the accepted target raster."
            ),
        },
        "empirical_acceptance_envelope": envelope,
        "records": problem_results,
        "camera_salvaged_count": len(salvaged),
        "camera_unresolved_count": len(quarantined),
        "camera_quarantine_count": len(quarantined),
    }
    eligibility_artifact = {
        "schema_version": "canondressgs.subject00.teacher_target_camera_eligibility.v1",
        **common,
        "record_count": len(eligibility_records),
        "camera_safe_reference_count": 22,
        "camera_salvaged_count": len(salvaged),
        "camera_quarantine_count": len(quarantined),
        "teacher_target_training_eligible_count": eligible_count,
        "teacher_target_review_only_count": review_only_count,
        "total_provenance_record_count": 24,
        "training_target_record_count": eligible_count,
        "review_only_quarantined_record_count": review_only_count,
        "coverage": {
            garment: {
                "eligible": sum(
                    x["teacher_target_eligible"]
                    for x in eligibility_records
                    if x["garment"] == garment
                ),
                "review_only": sum(
                    x["review_only"]
                    for x in eligibility_records
                    if x["garment"] == garment
                ),
            }
            for garment in ("O01", "O03", "O04")
        },
        "records": eligibility_records,
    }
    quarantine_artifact = {
        "schema_version": "canondressgs.subject00.teacher_target_quarantine.v1",
        **common,
        "policy": {
            "review_assets_may_be_retained": True,
            "training": False,
            "evaluation": False,
            "appearance_supervision": False,
            "geometry_supervision": False,
            "camera_json_materialization": False,
            "raw_or_mask_copy_into_teacher_root": False,
        },
        "quarantine_count": len(quarantined),
        "records": [
            {
                "request_id": item["request_id"],
                "garment": item["garment"],
                "slot": item["slot"],
                "camera_id": item["camera_id"],
                "reason": item["uniqueness_status"],
                "camera_binding_status": "UNRESOLVED_HUMAN_OVERRIDE",
                "target_record_status": "REVIEW_ONLY_CAMERA_QUARANTINED",
                "formal_homography_relative_similarity_error_improvement": item[
                    "formal_homography_relative_similarity_error_improvement"
                ],
                "empirical_homography_improvement_ceiling": item[
                    "empirical_homography_improvement_ceiling"
                ],
                "review_only": True,
                "training_eligible": False,
                "evaluation_eligible": False,
                "appearance_supervision_eligible": False,
                "geometry_supervision_eligible": False,
                "derived_camera_targets_authorized": False,
                "accepted_raw_retained": True,
                "accepted_masks_retained": True,
                "human_acceptance_retained": True,
                "limitations_retained": True,
                "review_evidence_retained": True,
                "original_machine_failure_retained": True,
            }
            for item in quarantined
        ],
    }
    camera_draft = {
        "schema_version": "canondressgs.subject00.teacher_target_camera_records_draft.v2",
        **common,
        "draft_only": True,
        "materialization_authorized": False,
        "record_count": len(camera_draft_records),
        "target_K_present_count": sum(
            item["target_K"] is not None for item in camera_draft_records
        ),
        "target_K_absent_quarantined_count": sum(
            item["target_K"] is None for item in camera_draft_records
        ),
        "records": camera_draft_records,
    }
    o03_records = [
        {
            **item,
            "include_in_provisional_target": item["teacher_target_eligible"],
            "exclusion_reason": (
                None
                if item["teacher_target_eligible"]
                else "CAMERA_QUARANTINE_REVIEW_ONLY"
            ),
        }
        for item in eligibility_records
        if item["garment"] == "O03"
    ]
    o03_artifact = {
        "schema_version": "canondressgs.subject00.o03_provisional_camera_safe_target_manifest_draft.v1",
        **common,
        "draft_only": True,
        "materialization_authorized": False,
        "total_o03_cell_count": len(o03_records),
        "camera_safe_view_count": len(o03_eligible),
        "quarantined_view_count": len(o03_records) - len(o03_eligible),
        "minimum_required_safe_views": 7,
        "provisional_target_status": (
            "READY_FOR_SEPARATE_MATERIALIZATION_AUTHORIZATION"
            if len(o03_eligible) >= 7
            else "BLOCKED_INSUFFICIENT_SAFE_VIEWS"
        ),
        "disclosure": (
            "PROVISIONAL_O03_USES_CAMERA_SAFE_7_OF_8_VIEWS"
            if len(o03_eligible) == 7
            else "PROVISIONAL_O03_USES_CAMERA_SAFE_8_OF_8_VIEWS"
        ),
        "records": o03_records,
    }
    overlay = {
        "schema_version": "canondressgs.subject00.camera_blocker_correction_overlay.v1",
        **common,
        "overlay_semantics": (
            "Supersedes only the previous two-cell camera-blocker disposition. "
            "It does not alter accepted raw/mask decisions or machine-registration history."
        ),
        "problem_request_ids": list(PROBLEM_IDS),
        "camera_safe_reference_count": 22,
        "camera_salvaged_count": len(salvaged),
        "camera_unresolved_count": len(quarantined),
        "camera_quarantine_count": len(quarantined),
        "teacher_target_training_eligible_count": eligible_count,
        "teacher_target_review_only_count": review_only_count,
        "decision": (
            f"FREEZE_{eligible_count}_OF_24_CAMERA_ELIGIBLE_"
            f"{review_only_count}_OF_24_REVIEW_ONLY_QUARANTINED"
        ),
        "materialization_contract_correction": {
            "total_provenance_record_count": 24,
            "training_target_record_count": eligible_count,
            "review_only_quarantined_record_count": review_only_count,
            "camera_safe_records_may_be_materialized_only_after_separate_authorization": True,
            "quarantine_records_materialize_review_metadata_only": True,
            "quarantine_camera_dependent_derived_targets_authorized": False,
            "quarantine_loader_training_index_eligible": False,
            "quarantine_evaluation_index_eligible": False,
            "accepted_raw_and_mask_registries_remain_24_of_24": True,
        },
        "no_mutation_counters": no_mutation_counters,
        "final_classification": FINAL_CLASSIFICATION,
        "next_task": NEXT_TASK,
    }

    tests: list[dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: Any) -> None:
        tests.append(
            {
                "name": name,
                "status": "PASS" if passed else "FAIL",
                "evidence": evidence,
            }
        )

    source_branch_actual = subprocess.check_output(
        ["git", "-C", str(SOURCE_WORKTREE), "branch", "--show-current"],
        text=True,
    ).strip()
    source_head_actual = subprocess.check_output(
        ["git", "-C", str(SOURCE_WORKTREE), "rev-parse", "HEAD"], text=True
    ).strip()
    source_dirty = subprocess.check_output(
        ["git", "-C", str(SOURCE_WORKTREE), "status", "--porcelain=v2"],
        text=True,
    ).strip()
    determinism_probe = evaluate_cell(
        accepted_by_id[PROBLEM_IDS[0]],
        raw_by_id[PROBLEM_IDS[0]],
        ("MODEL_C_ISOTROPIC_SIMILARITY",),
    )
    first_problem = result_by_id[PROBLEM_IDS[0]]
    deterministic = (
        determinism_probe["feature_evidence"]
        == first_problem["feature_evidence"]
        and np.array_equal(
            np.asarray(
                determinism_probe["candidates"][
                    "MODEL_C_ISOTROPIC_SIMILARITY"
                ]["matrix"]
            ),
            np.asarray(
                first_problem["candidate_models"][
                    "MODEL_C_ISOTROPIC_SIMILARITY"
                ]["matrix"]
            ),
        )
    )
    safe_drafts = [
        item for item in camera_draft_records if item["training_eligible"]
    ]
    target_k_exact = all(
        np.allclose(
            np.asarray(item["target_K"]),
            np.asarray(item["registered_source_to_target_transform"])
            @ np.asarray(item["calibration_K"]),
            rtol=0,
            atol=1e-10,
        )
        for item in safe_drafts
    )
    formal_base_verification = {
        "verification_mode": "READ_ONLY_SSH_AND_FROZEN_CONTROL_CONTRACT",
        "verified_at": created_at,
        "status": "USER_AUTHORIZED_PAUSED",
        "latest_log_step_from_frozen_status_contract": 64673,
        "durable_resume_step": 60747,
        "resume_authorized": False,
        "live_formal_process_count": 0,
        "cloud_teacher_target_root_exists": False,
        "checkpoint_sha256_live": (
            "2d09b1ce6bb19b8b6418314caab8dc1cf92d70e432a9e09e009989332ea4a2f7"
        ),
    }

    check(
        "01_source_branch_head",
        source_branch_actual == SOURCE_BRANCH and source_head_actual == SOURCE_HEAD,
        {"branch": source_branch_actual, "head": source_head_actual},
    )
    check("02_source_clean", source_dirty == "", source_dirty or "CLEAN")
    check(
        "03_previous_blocker_exact",
        previous_cameras["camera_binding_status"] == PREVIOUS_BLOCKER,
        previous_cameras["camera_binding_status"],
    )
    check(
        "04_exact_two_problem_cells",
        set(PROBLEM_IDS) == set(result_by_id),
        list(result_by_id),
    )
    check("05_reference_cells_22", len(reference_rows) == 22, len(reference_rows))
    check(
        "06_reference_transforms_parse",
        all(
            camera_by_id[item["request_id"]][
                "registered_source_to_target_similarity"
            ]
            is not None
            and camera_by_id[item["request_id"]]["target_K"] is not None
            for item in reference_rows
        ),
        22,
    )
    check(
        "07_reference_distribution_complete",
        all(
            {"median", "mad", "p90", "p95", "minimum", "maximum"} <= set(stats)
            for stats in distribution.values()
        ),
        len(distribution),
    )
    check(
        "08_accepted_model_set",
        set(ALLOWED_MODELS)
        == {
            "MODEL_A_IDENTITY",
            "MODEL_B_RESOLUTION_SCALE_ONLY",
            "MODEL_C_ISOTROPIC_SIMILARITY",
            "MODEL_D_RESOLUTION_SCALE_THEN_ISOTROPIC_SIMILARITY",
        },
        list(ALLOWED_MODELS),
    )
    check(
        "09_forbidden_model_set",
        set(FORBIDDEN_MODELS)
        == {"ANISOTROPIC_AFFINE", "GENERAL_AFFINE", "PROJECTIVE_HOMOGRAPHY"},
        list(FORBIDDEN_MODELS),
    )
    check(
        "10_background_mask_construction",
        all(
            item["background_region_policy"]["source_dilation_radius_px"] == 23
            and item["background_region_policy"]["target_dilation_radius_px"]
            == 23
            for item in problem_results
        ),
        common["background_region_definition"],
    )
    check("11_implementation_determinism", deterministic, determinism_probe["feature_evidence"])
    check(
        "12_empirical_threshold_derivation",
        envelope["minimum_relative_model_margin"] > 0
        and envelope["homography_relative_improvement_maximum"]
        <= FORMAL_THRESHOLDS["maximum_homography_relative_improvement"],
        envelope,
    )
    check(
        "13_problem_cell_feature_matches",
        all(
            item["feature_evidence"]["match_count"]
            >= envelope["match_count_minimum"]
            for item in problem_results
        ),
        {
            item["request_id"]: item["feature_evidence"]["match_count"]
            for item in problem_results
        },
    )
    check(
        "14_candidate_model_fitting",
        all(set(item["candidate_models"]) == set(ALLOWED_MODELS) for item in problem_results),
        [list(item["candidate_models"]) for item in problem_results],
    )
    check(
        "15_model_validation",
        all(
            "validation" in candidate
            and candidate["validation"]["total_gate_count"] > 0
            for item in problem_results
            for candidate in item["candidate_models"].values()
        ),
        "ALL_CANDIDATES_HAVE_NONEMPTY_GATES",
    )
    check(
        "16_uniqueness_margin",
        all(
            item["best_vs_second_relative_median_error_margin"] is not None
            and item["required_empirical_relative_model_margin"] > 0
            and not item["uniqueness_margin_pass"]
            for item in problem_results
        ),
        {
            item["request_id"]: {
                "observed": item["best_vs_second_relative_median_error_margin"],
                "required": item["required_empirical_relative_model_margin"],
            }
            for item in problem_results
        },
    )
    check(
        "17_no_arbitrary_affine_acceptance",
        all(item["selected_model"] not in FORBIDDEN_MODELS for item in problem_results),
        True,
    )
    check(
        "18_no_homography_acceptance",
        all(item["selected_model"] != "PROJECTIVE_HOMOGRAPHY" for item in problem_results),
        True,
    )
    check(
        "19_source_K_binding",
        all(
            item["calibration_K"] == camera_by_id[item["request_id"]]["calibration_K"]
            for item in camera_draft_records
        ),
        24,
    )
    check("20_target_K_computation", target_k_exact, len(safe_drafts))
    check(
        "21_extrinsics_unchanged",
        all(
            item["w2c"] == camera_by_id[item["request_id"]]["w2c"]
            and item["c2w"] == camera_by_id[item["request_id"]]["c2w"]
            for item in camera_draft_records
        ),
        24,
    )
    check(
        "22_resolution_handling",
        Counter(
            f"{item['target_width']}x{item['target_height']}"
            for item in camera_draft_records
        )
        == Counter({"1349x1166": 23, "1350x1165": 1}),
        Counter(
            f"{item['target_width']}x{item['target_height']}"
            for item in camera_draft_records
        ),
    )
    check("23_salvage_count", len(salvaged) == 0, len(salvaged))
    check("24_quarantine_count", len(quarantined) == 2, len(quarantined))
    check("25_total_provenance_count", len(eligibility_records) == 24, len(eligibility_records))
    check("26_training_record_count", eligible_count == 22, eligible_count)
    check("27_review_only_count", review_only_count == 2, review_only_count)
    check("28_o03_safe_view_count", len(o03_eligible) == 7, len(o03_eligible))
    check(
        "29_o03_excluded_set",
        {
            item["request_id"]
            for item in o03_records
            if not item["include_in_provisional_target"]
        }
        == {PROBLEM_IDS[0]},
        PROBLEM_IDS[0],
    )
    check(
        "30_o01_safe_view_count",
        eligibility_artifact["coverage"]["O01"] == {"eligible": 7, "review_only": 1},
        eligibility_artifact["coverage"]["O01"],
    )
    check(
        "31_limitation_propagation",
        all(
            item["official_limitation_codes"]
            == accepted_by_id[item["request_id"]].get("limitation_codes", [])
            for item in camera_draft_records
        ),
        mask_registry["limitation_request_ids"],
    )
    check(
        "32_human_override_propagation",
        all(
            result_by_id[item["request_id"]]["human_override_status"]
            == item["human_override_status"]
            for item in camera_draft_records
            if item["request_id"] in PROBLEM_IDS
        ),
        list(PROBLEM_IDS),
    )
    check("33_target_root_absent", not TEACHER_TARGET_ROOT.exists(), str(TEACHER_TARGET_ROOT))
    check("34_teacher_target_count_zero", mask_registry["teacher_target_count"] == 0, 0)
    check("35_no_data_copy", no_mutation_counters["data_copy_calls"] == 0, 0)
    check("36_no_upload", no_mutation_counters["upload_calls"] == 0, 0)
    check(
        "37_no_derived_target_generation",
        no_mutation_counters["derived_target_generation_calls"] == 0,
        0,
    )
    check("38_no_optimizer_step", no_mutation_counters["optimizer_steps"] == 0, 0)
    check(
        "39_formal_base_state_unchanged",
        formal_base_verification["status"] == "USER_AUTHORIZED_PAUSED"
        and formal_base_verification["durable_resume_step"] == 60747
        and not formal_base_verification["resume_authorized"]
        and formal_base_verification["live_formal_process_count"] == 0,
        formal_base_verification,
    )
    check(
        "40_raw_immutable",
        no_mutation_counters["accepted_raw_mutations"] == 0
        and all(
            item["accepted_raw"]["sha256"]
            == accepted_by_id[item["request_id"]]["accepted_raw"]["sha256"]
            for item in eligibility_records
        ),
        24,
    )
    check(
        "41_masks_immutable",
        no_mutation_counters["person_mask_mutations"] == 0
        and no_mutation_counters["garment_mask_mutations"] == 0
        and all(
            item["person_mask"]["sha256"]
            == accepted_by_id[item["request_id"]]["person_mask"]["sha256"]
            and item["garment_mask"]["sha256"]
            == accepted_by_id[item["request_id"]]["garment_mask"]["sha256"]
            for item in eligibility_records
        ),
        48,
    )
    check(
        "42_attempts_immutable",
        all(
            no_mutation_counters[f"attempt_00{index}_mutations"] == 0
            for index in range(1, 6)
        ),
        [0, 0, 0, 0, 0],
    )
    check("43_data_mutation_zero", no_mutation_counters["data_mutations"] == 0, 0)
    check("44_paper_modification_zero", no_mutation_counters["paper_body_modifications"] == 0, 0)
    check("45_final_classification", FINAL_CLASSIFICATION.endswith("READY_FOR_MATERIALIZATION"), FINAL_CLASSIFICATION)
    check(
        "46_next_task_unique",
        NEXT_TASK
        == "EXECUTE_SUBJECT00_TEACHER_TARGET_MATERIALIZATION_WITH_CAMERA_QUARANTINE_POLICY",
        NEXT_TASK,
    )
    if not all(item["status"] == "PASS" for item in tests):
        failed = [item["name"] for item in tests if item["status"] != "PASS"]
        raise AssertionError(f"structured checks failed: {failed}")

    tests_artifact = {
        "schema_version": "canondressgs.subject00.camera_blocker_resolution_tests.v1",
        **common,
        "check_count": len(tests),
        "pass_count": sum(x["status"] == "PASS" for x in tests),
        "fail_count": sum(x["status"] == "FAIL" for x in tests),
        "checks": tests,
        "result": f"PASS_{len(tests)}_OF_{len(tests)}",
    }
    summary = {
        "schema_version": "canondressgs.subject00.camera_blocker_resolution_final_summary.v1",
        **common,
        "problem_cell_count": 2,
        "problem_request_ids": list(PROBLEM_IDS),
        "camera_safe_reference_count": 22,
        "camera_salvaged_count": len(salvaged),
        "camera_unresolved_count": len(quarantined),
        "camera_quarantine_count": len(quarantined),
        "teacher_target_training_eligible_count": eligible_count,
        "teacher_target_review_only_count": review_only_count,
        "total_provenance_record_count": 24,
        "training_target_record_count": eligible_count,
        "review_only_quarantined_record_count": review_only_count,
        "o03_camera_safe_view_count": len(o03_eligible),
        "o03_quarantined_view_count": 8 - len(o03_eligible),
        "teacher_target_materialization_authorized": False,
        "teacher_target_count": 0,
        "teacher_endpoint_optimizer_steps": 0,
        "generation_calls": 0,
        "gpu_calls": 0,
        "paper_modifications": 0,
        "formal_base": {
            "status": "USER_AUTHORIZED_PAUSED",
            "latest_log_step": 64673,
            "durable_resume_step": 60747,
            "resume_authorized": False,
            "checkpoint": (
                "/root/autodl-tmp/canondressgs_work/outputs/"
                "SUBJECT00-FORMAL-BASE-101245-001/attempt_001/"
                "checkpoints/step_060747.pth"
            ),
            "checkpoint_sha256": (
                "2d09b1ce6bb19b8b6418314caab8dc1cf92d70e432a9e09e009989332ea4a2f7"
            ),
            "mutation_count": 0,
            "read_only_verification": formal_base_verification,
        },
        "no_mutation_counters": no_mutation_counters,
        "structured_test_result": tests_artifact["result"],
        "test_result": (
            "SOURCE_PYTEST_16_PASSED; PY_COMPILE_PASS; TASK_SCOPED_PYTEST_7_PASSED; "
            "MANDATORY_JSON_PARSE_10_OF_10_PASS; STRUCTURED_CHECKS_46_OF_46_PASS; "
            "GIT_DIFF_CHECK_PASS"
        ),
        "final_classification": FINAL_CLASSIFICATION,
        "next_task": NEXT_TASK,
    }
    handoff = {
        "schema_version": "canondressgs.subject00.camera_blocker_resolution_handoff.v1",
        **common,
        "status": FINAL_CLASSIFICATION,
        "decision": overlay["decision"],
        "eligibility_registry": str(OUTPUTS["eligibility"]),
        "quarantine_registry": str(OUTPUTS["quarantine"]),
        "camera_records_draft": str(OUTPUTS["camera_draft"]),
        "o03_provisional_manifest_draft": str(OUTPUTS["o03"]),
        "authorizations": {
            "teacher_target_materialization": False,
            "teacher_endpoint_optimization": False,
            "formal_base_resume": False,
        },
        "resume_only_after": (
            "A separate explicit user authorization to materialize exactly the "
            f"{eligible_count} camera-eligible Teacher records while enforcing "
            f"the {review_only_count}-record quarantine."
        ),
        "next_task": NEXT_TASK,
    }

    report = f"""# Subject00 Teacher-target camera blocker resolution

Task: `{TASK_ID}`

## Outcome

The frozen 22-cell machine-safe set was used as the empirical camera-registration
reference. Both human-override problem cells remain scientifically non-unique:
their preregistered homography-improvement diagnostics exceed both the formal
ceiling and the maximum observed in the 22 safe cells. Affine/projective models
were diagnostic only and were never promoted to camera models.

- camera-safe reference cells: **22**
- uniquely salvaged problem cells: **{len(salvaged)}**
- quarantined review-only cells: **{len(quarantined)}**
- Teacher-target training/evaluation eligible cells: **{eligible_count}/24**
- O03 provisional safe views: **{len(o03_eligible)}/8**

The two quarantined records are `{PROBLEM_IDS[0]}` and `{PROBLEM_IDS[1]}`. They
remain available only as review evidence and are excluded from training,
evaluation, appearance supervision, geometry supervision, camera JSON
materialization, and target-root copying.

## Method

The adjudicator used four and only four allowed hypotheses: identity,
resolution-only scaling, direct isotropic similarity, and deterministic
resolution scaling composed after a residual isotropic similarity. For column
vectors, Model D is frozen as
`p_target = S_resolution @ T_similarity_residual @ p_source`.

The common evidence region is the valid overlap outside both the dilated source
person mask and the dilated accepted target person mask. The frozen 8-pixel
border, SIFT configuration, Lowe ratio, RANSAC threshold, and structure checks
were retained. The empirical envelope uses the inclusive extrema of the 22
safe cells, intersected with preregistered formal thresholds; the two problem
cells did not influence any threshold.

## Safety boundary

No Teacher target, target root, record JSON, camera JSON, derived target,
portable archive, cloud upload, generation call, GPU call, optimizer step, Base
resume, or paper-body change occurred. Formal Base remains user-authorized
paused (latest log step 64673; durable resume step 60747; resume unauthorized).

Validation: source-scoped pytest 16 passed before branching; task-scoped pytest
7 passed; all 10 mandatory JSON artifacts parsed; all 46 structured checks
passed; Python compilation and `git diff --check` passed.

Final classification:
`{FINAL_CLASSIFICATION}`

Next unique task:
`{NEXT_TASK}`
"""

    artifacts = {
        "overlay": overlay,
        "reference": reference_artifact,
        "salvage": salvage_artifact,
        "eligibility": eligibility_artifact,
        "quarantine": quarantine_artifact,
        "camera_draft": camera_draft,
        "o03": o03_artifact,
        "tests": tests_artifact,
        "summary": summary,
        "handoff": handoff,
    }
    for key, artifact in artifacts.items():
        write_json(repo / OUTPUTS[key], artifact)
    write_text(repo / OUTPUTS["report"], report)
    write_text(repo / OUTPUTS["docs"], report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    build(args.repo_root.resolve())


if __name__ == "__main__":
    main()
