#!/usr/bin/env python3
"""Audit Attempt 001 outputs for native-resolution background registration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from skimage.metrics import structural_similarity


ROOT = Path(__file__).resolve().parents[2]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
HANDOFF = ROOT / "project_control_handoff"
SOURCE_WORKTREE = Path(r"E:\model_train\canondressgs_subject00_v2_visual_fail_valid_region_audit")
ATTEMPTS_ROOT = Path(r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-GENERATION-001")
ATTEMPT_001 = ATTEMPTS_ROOT / "attempt_001"
ATTEMPT_002 = ATTEMPTS_ROOT / "attempt_002_portrait_canary"
ATTEMPT_003 = ATTEMPTS_ROOT / "attempt_003_portrait_canary_v2_registered_outpaint"
AUDIT_ROOT = ATTEMPTS_ROOT / "attempt_001_native_landscape_registration_audit"
ARCHIVE = Path(r"E:\data_pre\thuman4_second_identity_staging\downloads\subject00.7z")
CLOUD_DATA_MANIFEST = Path(r"E:\data_pre\thuman4_second_identity_staging\reports\SUBJECT00_CLOUD_DATA_MANIFEST.json")
LOCAL_CALIBRATION = ATTEMPT_003 / "00_source_evidence" / "calibration.json"

TASK_ID = "AAAI27-SUBJECT00-ATTEMPT001-NATIVE-LANDSCAPE-REGISTRATION-AUDIT-001"
SOURCE_BRANCH = "research/subject00-v2-visual-fail-valid-region-audit-20260726"
SOURCE_HEAD = "34231cd3a203732b426f4f88701f22908cd26a56"
BRANCH = "research/subject00-attempt001-native-landscape-registration-audit-20260726"
AUDIT_TIMESTAMP = "2026-07-26T09:54:20+08:00"
PROTOCOL_PATH = RISK / "subject00_attempt001_native_landscape_registration_protocol_20260726.json"

GARMENTS = ["O01", "O03", "O04"]
SLOTS = {
    "slot_00": {"camera_id": 17, "camera": "cam17", "orientation": "front"},
    "slot_01": {"camera_id": 21, "camera": "cam21", "orientation": "front-left"},
    "slot_02": {"camera_id": 14, "camera": "cam14", "orientation": "front-right"},
    "slot_03": {"camera_id": 23, "camera": "cam23", "orientation": "left"},
    "slot_04": {"camera_id": 11, "camera": "cam11", "orientation": "right"},
    "slot_05": {"camera_id": 2, "camera": "cam02", "orientation": "back-left"},
    "slot_06": {"camera_id": 9, "camera": "cam09", "orientation": "back-right"},
    "slot_07": {"camera_id": 5, "camera": "cam05", "orientation": "back"},
}
REQUEST_PATTERN = re.compile(r"^subject00_(O01|O03|O04)_slot(\d{2})_cand(00|01)$")
CLASSIFICATIONS = [
    "REGISTERED_SIMILARITY_PASS_CANDIDATE",
    "REGISTRATION_EVIDENCE_INSUFFICIENT",
    "AFFINE_OR_PROJECTIVE_RECOMPOSITION_FAIL",
    "BACKGROUND_GEOMETRY_CHANGED_FAIL",
    "PERSON_OR_POSE_REGISTRATION_FAIL",
    "GARMENT_OR_HUMAN_VISUAL_FAIL",
]


def canonical_sha256(value: Any) -> str:
    if isinstance(value, dict):
        value = dict(value)
        value.pop("content_sha256", None)
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(payload)
    value.pop("content_sha256", None)
    value["content_sha256"] = canonical_sha256(value)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value.rstrip() + "\n")


def run(*command: str, cwd: Path = ROOT) -> str:
    return subprocess.run(
        list(command), cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def archive_bytes(member: str) -> bytes:
    return subprocess.run(
        ["tar", "-xOf", str(ARCHIVE), member], check=True, capture_output=True
    ).stdout


def archive_member_batch(members: list[str], sizes: dict[str, int]) -> dict[str, bytes]:
    """Read solid-archive members in one scan and split using frozen manifest sizes."""
    ordered = sorted(members)
    payload = subprocess.run(
        ["tar", "-xOf", str(ARCHIVE), *ordered], check=True, capture_output=True
    ).stdout
    expected_bytes = sum(sizes[member] for member in ordered)
    if len(payload) != expected_bytes:
        raise RuntimeError(f"archive batch byte count mismatch: {len(payload)} != {expected_bytes}")
    result: dict[str, bytes] = {}
    offset = 0
    for member in ordered:
        size = sizes[member]
        result[member] = payload[offset:offset + size]
        offset += size
    return result


def inventory(root: Path) -> dict[str, Any]:
    files = [
        {
            "relative_path": item.relative_to(root).as_posix(),
            "bytes": item.stat().st_size,
            "sha256": file_sha256(item),
        }
        for item in sorted(path for path in root.rglob("*") if path.is_file())
    ]
    return {
        "root": str(root),
        "file_count": len(files),
        "tree_sha256": canonical_sha256(files),
        "files": files,
    }


def as_float(value: Any) -> float:
    return float(np.asarray(value).item())


def percentile(values: list[float], q: float) -> float | str:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else "NOT_RELIABLE"


def validate_source_gate() -> None:
    facts = {
        "source_branch": run("git", "branch", "--show-current", cwd=SOURCE_WORKTREE),
        "source_head": run("git", "rev-parse", "HEAD", cwd=SOURCE_WORKTREE),
        "source_status": run("git", "status", "--short", cwd=SOURCE_WORKTREE),
        "target_branch": run("git", "branch", "--show-current"),
    }
    expected = {
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "source_status": "",
        "target_branch": BRANCH,
    }
    if facts != expected:
        raise RuntimeError(f"source gate mismatch: {facts!r}")
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", SOURCE_HEAD, "HEAD"], cwd=ROOT)
    if ancestor.returncode != 0:
        raise RuntimeError("source HEAD is not an ancestor of target HEAD")


def prepare_output(replace: bool) -> None:
    if AUDIT_ROOT.exists():
        if not replace:
            raise RuntimeError(f"audit output already exists: {AUDIT_ROOT}")
        if AUDIT_ROOT.resolve().parent != ATTEMPTS_ROOT.resolve() or AUDIT_ROOT.name != "attempt_001_native_landscape_registration_audit":
            raise RuntimeError("refusing to replace unexpected audit path")
        shutil.rmtree(AUDIT_ROOT)
    for name in [
        "00_provenance", "01_inventory", "02_registration_metrics", "03_camera_candidates",
        "04_coverage", "05_contact_sheets/requests", "06_human_review", "07_final_summary",
    ]:
        (AUDIT_ROOT / name).mkdir(parents=True, exist_ok=False)


def decode_image(value: bytes, mode: int) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), mode)
    if image is None:
        raise RuntimeError("OpenCV could not decode archive image")
    return image


def expected_request_ids() -> list[str]:
    return [
        f"subject00_{garment}_slot{slot[-2:]}_cand{candidate:02d}"
        for garment in GARMENTS for slot in SLOTS for candidate in range(2)
    ]


def fit_uniform_scale_translation(
    source: np.ndarray, target: np.ndarray, seed: int, threshold: float, max_iterations: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if len(source) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    best_mask: np.ndarray | None = None
    best_score = (-1, float("inf"))
    for _ in range(max_iterations):
        indices = rng.choice(len(source), size=2, replace=False)
        delta_source = source[indices[1]] - source[indices[0]]
        delta_target = target[indices[1]] - target[indices[0]]
        denominator = float(np.dot(delta_source, delta_source))
        if denominator < 1e-8:
            continue
        scale = float(np.dot(delta_source, delta_target) / denominator)
        translation = np.mean(target[indices] - scale * source[indices], axis=0)
        predicted = scale * source + translation
        errors = np.linalg.norm(predicted - target, axis=1)
        mask = errors <= threshold
        count = int(mask.sum())
        median = float(np.median(errors[mask])) if count else float("inf")
        score = (count, -median)
        if score > best_score:
            best_score = score
            best_mask = mask
    if best_mask is None or int(best_mask.sum()) < 2:
        return None, None
    for _ in range(3):
        rows = []
        values = []
        for (x, y), (u, v) in zip(source[best_mask], target[best_mask]):
            rows.extend([[x, 1.0, 0.0], [y, 0.0, 1.0]])
            values.extend([u, v])
        scale, tx, ty = np.linalg.lstsq(np.asarray(rows), np.asarray(values), rcond=None)[0]
        matrix = np.asarray([[scale, 0.0, tx], [0.0, scale, ty]], dtype=np.float64)
        errors = affine_errors(matrix, source, target)
        best_mask = errors <= threshold
        if int(best_mask.sum()) < 2:
            return None, None
    return matrix, best_mask.astype(np.uint8)


def fit_axis_scale_translation(source: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    selected_source = source[mask]
    selected_target = target[mask]
    if len(selected_source) < 3:
        return None
    design_x = np.column_stack([selected_source[:, 0], np.ones(len(selected_source))])
    design_y = np.column_stack([selected_source[:, 1], np.ones(len(selected_source))])
    sx, tx = np.linalg.lstsq(design_x, selected_target[:, 0], rcond=None)[0]
    sy, ty = np.linalg.lstsq(design_y, selected_target[:, 1], rcond=None)[0]
    return np.asarray([[sx, 0.0, tx], [0.0, sy, ty]], dtype=np.float64)


def affine_errors(matrix: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    predicted = source @ matrix[:, :2].T + matrix[:, 2]
    return np.linalg.norm(predicted - target, axis=1)


def homography_errors(matrix: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([source, np.ones(len(source))])
    predicted = homogeneous @ matrix.T
    predicted = predicted[:, :2] / np.maximum(np.abs(predicted[:, 2:3]), 1e-12)
    return np.linalg.norm(predicted - target, axis=1)


def error_summary(errors: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    if mask is not None:
        errors = errors[np.asarray(mask).reshape(-1) > 0]
    if not len(errors):
        return {"median_px": "NOT_RELIABLE", "p95_px": "NOT_RELIABLE", "maximum_px": "NOT_RELIABLE"}
    return {
        "median_px": float(np.median(errors)),
        "p95_px": float(np.percentile(errors, 95)),
        "maximum_px": float(np.max(errors)),
    }


def affine_decomposition(matrix: np.ndarray | None) -> dict[str, Any]:
    if matrix is None:
        return {
            "scale_x": "NOT_RELIABLE", "scale_y": "NOT_RELIABLE",
            "rotation_degrees": "NOT_RELIABLE", "normalized_shear": "NOT_RELIABLE",
        }
    linear = matrix[:, :2]
    first = linear[:, 0]
    second = linear[:, 1]
    scale_x = float(np.linalg.norm(first))
    scale_y = float(np.linalg.norm(second))
    shear = float(np.dot(first, second) / max(scale_x * scale_y, 1e-12))
    rotation = float(np.degrees(np.arctan2(first[1], first[0])))
    return {
        "scale_x": scale_x,
        "scale_y": scale_y,
        "rotation_degrees": rotation,
        "normalized_shear": shear,
    }


def person_bounds_after_transform(person_bbox: list[int], matrix: np.ndarray, width: int, height: int, margin: int) -> dict[str, Any]:
    x0, y0, x1, y1 = person_bbox
    corners = np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
    transformed = corners @ matrix[:, :2].T + matrix[:, 2]
    bbox = [
        float(transformed[:, 0].min()), float(transformed[:, 1].min()),
        float(transformed[:, 0].max()), float(transformed[:, 1].max()),
    ]
    passed = bbox[0] >= margin and bbox[1] >= margin and bbox[2] <= width - margin and bbox[3] <= height - margin
    return {"transformed_bbox": bbox, "minimum_margin_px": margin, "pass": bool(passed)}


def photometric_metrics(
    source_bgr: np.ndarray,
    output_bgr: np.ndarray,
    source_background_mask: np.ndarray,
    matrix: np.ndarray,
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    height, width = output_bgr.shape[:2]
    warped_source = cv2.warpAffine(source_bgr, matrix, (width, height), flags=cv2.INTER_LINEAR)
    warped_background = cv2.warpAffine(source_background_mask, matrix, (width, height), flags=cv2.INTER_NEAREST) > 0
    overlap = cv2.warpAffine(
        np.full(source_background_mask.shape, 255, dtype=np.uint8), matrix, (width, height), flags=cv2.INTER_NEAREST
    ) > 0
    valid = warped_background & overlap
    border = int(protocol["background_mask"]["image_border_exclusion_px"])
    valid[:border] = False
    valid[-border:] = False
    valid[:, :border] = False
    valid[:, -border:] = False
    valid = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), dtype=np.uint8), iterations=1) > 0
    if int(valid.sum()) < 1000:
        return ({
            "status": "NOT_RELIABLE", "valid_background_pixel_count": int(valid.sum()),
            "rgb_mae": "NOT_RELIABLE", "psnr_db": "NOT_RELIABLE", "masked_ssim": "NOT_RELIABLE",
            "edge_alignment_f1": "NOT_RELIABLE", "structural_difference_area_fraction": "NOT_RELIABLE",
        }, {"warped_source": warped_source, "valid": valid.astype(np.uint8) * 255})

    difference = np.abs(warped_source.astype(np.float32) - output_bgr.astype(np.float32))
    mae = float(difference[valid].mean())
    mse = float(np.square(difference[valid]).mean())
    psnr = float(20.0 * math.log10(255.0 / math.sqrt(max(mse, 1e-12))))
    source_gray = cv2.cvtColor(warped_source, cv2.COLOR_BGR2GRAY)
    output_gray = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2GRAY)
    _, ssim_map = structural_similarity(source_gray, output_gray, data_range=255, full=True)
    masked_ssim = float(ssim_map[valid].mean())

    structure = protocol["background_structure_gate"]
    source_edges = cv2.Canny(source_gray, int(structure["canny_low"]), int(structure["canny_high"])) > 0
    output_edges = cv2.Canny(output_gray, int(structure["canny_low"]), int(structure["canny_high"])) > 0
    source_edges &= valid
    output_edges &= valid
    tolerance = int(structure["edge_tolerance_px"])
    kernel = np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype=np.uint8)
    dilated_source = cv2.dilate(source_edges.astype(np.uint8), kernel) > 0
    dilated_output = cv2.dilate(output_edges.astype(np.uint8), kernel) > 0
    precision = float((output_edges & dilated_source).sum() / max(int(output_edges.sum()), 1))
    recall = float((source_edges & dilated_output).sum() / max(int(source_edges.sum()), 1))
    f1 = float(2 * precision * recall / max(precision + recall, 1e-12))
    structural_area = float((ssim_map[valid] < float(structure["ssim_difference_threshold"])).mean())
    difference_gray = np.clip(difference.mean(axis=2) * 4.0, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(difference_gray, cv2.COLORMAP_TURBO)
    heatmap[~valid] = 0
    edge_overlay = np.zeros_like(output_bgr)
    edge_overlay[source_edges] = (0, 255, 0)
    edge_overlay[output_edges] = np.maximum(edge_overlay[output_edges], np.asarray([0, 0, 255], dtype=np.uint8))
    return ({
        "status": "PASS_COMPUTED",
        "valid_background_pixel_count": int(valid.sum()),
        "valid_background_fraction": float(valid.mean()),
        "rgb_mae": mae,
        "psnr_db": psnr,
        "masked_ssim": masked_ssim,
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_alignment_f1": f1,
        "structural_difference_area_fraction": structural_area,
    }, {
        "warped_source": warped_source,
        "valid": valid.astype(np.uint8) * 255,
        "heatmap": heatmap,
        "edge_overlay": edge_overlay,
    })


def calculate_registration(
    request_id: str,
    source: dict[str, Any],
    output_bgr: np.ndarray,
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    feature = protocol["feature_matching"]
    models = protocol["models"]
    gray_output = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(
        nfeatures=int(feature["nfeatures"]),
        contrastThreshold=float(feature["contrast_threshold"]),
        edgeThreshold=float(feature["edge_threshold"]),
    )
    output_keypoints, output_descriptors = sift.detectAndCompute(gray_output, None)
    source_keypoints = source["keypoints"]
    source_descriptors = source["descriptors"]
    good_matches: list[cv2.DMatch] = []
    if source_descriptors is not None and output_descriptors is not None:
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        candidates = matcher.knnMatch(source_descriptors, output_descriptors, k=2)
        ratio = float(feature["lowe_ratio"])
        ratio_matches = [first for first, second in candidates if first.distance < ratio * second.distance]
        used_train: set[int] = set()
        for item in sorted(ratio_matches, key=lambda match: match.distance):
            if item.trainIdx not in used_train:
                used_train.add(item.trainIdx)
                good_matches.append(item)

    source_points = np.float64([source_keypoints[item.queryIdx].pt for item in good_matches])
    output_points = np.float64([output_keypoints[item.trainIdx].pt for item in good_matches])
    seed = int(hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF
    threshold = float(models["ransac_reprojection_threshold_px"])
    iterations = int(models["ransac_max_iterations"])
    uniform_matrix, uniform_inliers = fit_uniform_scale_translation(
        source_points, output_points, seed, threshold, iterations
    )
    similarity_matrix = affine_matrix = homography_matrix = None
    similarity_inliers = affine_inliers = homography_inliers = None
    if len(source_points) >= 3:
        cv2.setRNGSeed(seed)
        similarity_matrix, similarity_inliers = cv2.estimateAffinePartial2D(
            source_points, output_points, method=cv2.RANSAC, ransacReprojThreshold=threshold,
            maxIters=iterations, confidence=float(models["ransac_confidence"]), refineIters=10,
        )
        cv2.setRNGSeed(seed)
        affine_matrix, affine_inliers = cv2.estimateAffine2D(
            source_points, output_points, method=cv2.RANSAC, ransacReprojThreshold=threshold,
            maxIters=iterations, confidence=float(models["ransac_confidence"]), refineIters=10,
        )
    if len(source_points) >= 4:
        cv2.setRNGSeed(seed)
        homography_matrix, homography_inliers = cv2.findHomography(
            source_points, output_points, cv2.RANSAC, threshold,
            maxIters=iterations, confidence=float(models["ransac_confidence"]),
        )

    def affine_record(matrix: np.ndarray | None, inliers: np.ndarray | None) -> dict[str, Any]:
        if matrix is None or inliers is None:
            return {"status": "NOT_RELIABLE", "matrix": None, "inlier_count": 0, "inlier_ratio": 0.0, "errors": error_summary(np.asarray([]))}
        errors = affine_errors(matrix, source_points, output_points)
        return {
            "status": "PASS_COMPUTED",
            "matrix": np.asarray(matrix).tolist(),
            "inlier_count": int(inliers.sum()),
            "inlier_ratio": float(inliers.mean()),
            "errors": error_summary(errors, inliers),
        }

    uniform_record = affine_record(uniform_matrix, uniform_inliers)
    similarity_record = affine_record(similarity_matrix, similarity_inliers)
    affine_model_record = affine_record(affine_matrix, affine_inliers)
    if homography_matrix is None or homography_inliers is None:
        homography_record = {"status": "NOT_RELIABLE", "matrix": None, "inlier_count": 0, "inlier_ratio": 0.0, "errors": error_summary(np.asarray([]))}
    else:
        h_errors = homography_errors(homography_matrix, source_points, output_points)
        homography_record = {
            "status": "PASS_COMPUTED", "matrix": homography_matrix.tolist(),
            "inlier_count": int(homography_inliers.sum()), "inlier_ratio": float(homography_inliers.mean()),
            "errors": error_summary(h_errors, homography_inliers),
        }

    axis_matrix = None
    axis_record: dict[str, Any] = {"status": "NOT_RELIABLE", "matrix": None}
    if similarity_matrix is not None and similarity_inliers is not None:
        similarity_mask = similarity_inliers.reshape(-1) > 0
        axis_matrix = fit_axis_scale_translation(source_points, output_points, similarity_mask)
        if axis_matrix is not None:
            axis_errors = affine_errors(axis_matrix, source_points, output_points)
            axis_record = {
                "status": "PASS_COMPUTED", "matrix": axis_matrix.tolist(),
                "scale_x": float(axis_matrix[0, 0]), "scale_y": float(axis_matrix[1, 1]),
                "translation_x": float(axis_matrix[0, 2]), "translation_y": float(axis_matrix[1, 2]),
                "errors_on_similarity_inliers": error_summary(axis_errors, similarity_inliers),
            }

    affine_parts = affine_decomposition(affine_matrix)
    similarity_parts = affine_decomposition(similarity_matrix)
    homography_improvement: float | str = "NOT_RELIABLE"
    if similarity_matrix is not None and similarity_inliers is not None and homography_matrix is not None:
        mask = similarity_inliers.reshape(-1) > 0
        similarity_errors = affine_errors(similarity_matrix, source_points, output_points)[mask]
        homography_errors_shared = homography_errors(homography_matrix, source_points, output_points)[mask]
        if len(similarity_errors):
            sim_median = float(np.median(similarity_errors))
            hom_median = float(np.median(homography_errors_shared))
            homography_improvement = float((sim_median - hom_median) / max(sim_median, 1e-12))

    photo: dict[str, Any]
    visuals: dict[str, np.ndarray]
    if similarity_matrix is not None:
        photo, visuals = photometric_metrics(
            source["image"], output_bgr, source["background_mask"], similarity_matrix, protocol
        )
    else:
        photo = {
            "status": "NOT_RELIABLE", "valid_background_pixel_count": 0,
            "rgb_mae": "NOT_RELIABLE", "psnr_db": "NOT_RELIABLE", "masked_ssim": "NOT_RELIABLE",
            "edge_alignment_f1": "NOT_RELIABLE", "structural_difference_area_fraction": "NOT_RELIABLE",
        }
        visuals = {}

    feature_sufficient = (
        len(source_keypoints) >= int(feature["minimum_source_background_features"])
        and len(output_keypoints) >= int(feature["minimum_output_features"])
        and len(good_matches) >= int(feature["minimum_ratio_matches"])
    )
    gate = protocol["machine_similarity_gate"]
    structure_gate = protocol["background_structure_gate"]
    similarity_stable = (
        similarity_record["status"] == "PASS_COMPUTED"
        and similarity_record["inlier_count"] >= int(models["minimum_similarity_inliers"])
        and similarity_record["inlier_ratio"] >= float(gate["minimum_inlier_ratio"])
    )

    if similarity_matrix is not None:
        person_bounds = person_bounds_after_transform(
            source["person_bbox"], similarity_matrix, output_bgr.shape[1], output_bgr.shape[0],
            int(gate["minimum_transformed_person_margin_px"]),
        )
    else:
        person_bounds = {"transformed_bbox": None, "minimum_margin_px": int(gate["minimum_transformed_person_margin_px"]), "pass": False}

    scale_x = axis_record.get("scale_x", "NOT_RELIABLE")
    scale_y = axis_record.get("scale_y", "NOT_RELIABLE")
    anisotropy_pass = isinstance(scale_x, float) and isinstance(scale_y, float) and abs(scale_x / scale_y - 1.0) <= float(gate["maximum_abs_scale_ratio_minus_one"])
    rotation_value = similarity_parts["rotation_degrees"]
    rotation_pass = isinstance(rotation_value, float) and abs(rotation_value) <= float(gate["maximum_abs_rotation_degrees"])
    shear_value = affine_parts["normalized_shear"]
    shear_pass = isinstance(shear_value, float) and abs(shear_value) <= float(gate["maximum_abs_normalized_shear"])
    sim_errors = similarity_record["errors"]
    reprojection_pass = (
        isinstance(sim_errors["median_px"], float) and isinstance(sim_errors["p95_px"], float)
        and sim_errors["median_px"] <= float(gate["maximum_median_reprojection_error_px"])
        and sim_errors["p95_px"] <= float(gate["maximum_p95_reprojection_error_px"])
    )
    homography_pass = isinstance(homography_improvement, float) and homography_improvement <= float(gate["maximum_homography_relative_median_error_improvement"])
    structure_pass = (
        photo["status"] == "PASS_COMPUTED"
        and photo["masked_ssim"] >= float(structure_gate["minimum_masked_ssim"])
        and photo["edge_alignment_f1"] >= float(structure_gate["minimum_edge_alignment_f1"])
        and photo["structural_difference_area_fraction"] <= float(structure_gate["maximum_structural_difference_area_fraction"])
    )
    geometry_pass = similarity_stable and anisotropy_pass and rotation_pass and shear_pass and reprojection_pass and homography_pass
    failure_tags = [
        name for name, passed in {
            "INSUFFICIENT_FEATURE_EVIDENCE": feature_sufficient,
            "UNSTABLE_SIMILARITY_RANSAC": similarity_stable,
            "SCALE_ANISOTROPY_EXCEEDS_GATE": anisotropy_pass,
            "ROTATION_EXCEEDS_GATE": rotation_pass,
            "SHEAR_EXCEEDS_GATE": shear_pass,
            "SIMILARITY_REPROJECTION_EXCEEDS_GATE": reprojection_pass,
            "HOMOGRAPHY_IMPROVEMENT_EXCEEDS_GATE": homography_pass,
            "TRANSFORMED_SOURCE_PERSON_BOUNDS_FAIL": person_bounds["pass"],
            "BACKGROUND_STRUCTURE_GATE_FAIL": structure_pass,
        }.items() if not passed
    ]
    if not feature_sufficient or not similarity_stable:
        primary = "REGISTRATION_EVIDENCE_INSUFFICIENT"
    elif not person_bounds["pass"]:
        primary = "PERSON_OR_POSE_REGISTRATION_FAIL"
    elif not geometry_pass:
        primary = "AFFINE_OR_PROJECTIVE_RECOMPOSITION_FAIL"
    elif not structure_pass:
        primary = "BACKGROUND_GEOMETRY_CHANGED_FAIL"
    else:
        primary = "REGISTERED_SIMILARITY_PASS_CANDIDATE"

    metric = {
        "request_id": request_id,
        "background_feature_count": len(source_keypoints),
        "output_feature_count": len(output_keypoints),
        "match_count": len(good_matches),
        "uniform_scale_translation": uniform_record,
        "similarity": similarity_record,
        "axis_scale_translation": axis_record,
        "affine": {**affine_model_record, **affine_parts},
        "homography": homography_record,
        "homography_relative_similarity_error_improvement": homography_improvement,
        "estimated_scale_x": scale_x,
        "estimated_scale_y": scale_y,
        "uniform_scale": similarity_parts["scale_x"],
        "translation_x": float(similarity_matrix[0, 2]) if similarity_matrix is not None else "NOT_RELIABLE",
        "translation_y": float(similarity_matrix[1, 2]) if similarity_matrix is not None else "NOT_RELIABLE",
        "rotation_degrees": rotation_value,
        "normalized_shear": shear_value,
        "median_reprojection_error_px": sim_errors["median_px"],
        "p95_reprojection_error_px": sim_errors["p95_px"],
        "maximum_reprojection_error_px": sim_errors["maximum_px"],
        "affine_residual": affine_model_record["errors"],
        "homography_residual": homography_record["errors"],
        "registration_confidence": "HIGH" if primary == "REGISTERED_SIMILARITY_PASS_CANDIDATE" else ("LOW" if primary == "REGISTRATION_EVIDENCE_INSUFFICIENT" else "FAIL"),
        "transformed_source_person_bounds": person_bounds,
        "person_pose_metrics": {
            "person_bbox_source": source["person_bbox"],
            "person_bbox_output": "NOT_AVAILABLE",
            "person_height_ratio": "NOT_AVAILABLE",
            "person_width_ratio": "NOT_AVAILABLE",
            "subject_center_displacement": "NOT_AVAILABLE",
            "feet_position_displacement": "NOT_AVAILABLE",
            "hand_position_displacement": "NOT_AVAILABLE",
            "non_garment_keypoint_drift": "NOT_AVAILABLE",
            "reason": "No frozen output person/keypoint model is available; no new model was downloaded.",
        },
        "photometric": photo,
        "gate_results": {
            "feature_evidence": feature_sufficient,
            "similarity_stable": similarity_stable,
            "scale_anisotropy": anisotropy_pass,
            "rotation": rotation_pass,
            "shear": shear_pass,
            "reprojection": reprojection_pass,
            "homography_diagnostic": homography_pass,
            "transformed_source_person_bounds": bool(person_bounds["pass"]),
            "background_structure": structure_pass,
            "camera_direction_manifest_consistent": True,
        },
        "failure_tags": failure_tags,
        "primary_classification": primary,
        "human_visual_decision": None,
    }
    return metric, visuals


def fit_image(image: np.ndarray, width: int, height: int, background: tuple[int, int, int] = (32, 32, 32)) -> np.ndarray:
    canvas = np.full((height, width, 3), background, dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def labeled_panel(image: np.ndarray, label: str, width: int = 500, height: int = 300) -> np.ndarray:
    panel = fit_image(image, width, height - 28)
    output = np.full((height, width, 3), 24, dtype=np.uint8)
    output[28:] = panel
    cv2.putText(output, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    return output


def request_visualization(
    record: dict[str, Any], metric: dict[str, Any], source: dict[str, Any], output: np.ndarray, visuals: dict[str, np.ndarray], destination: Path
) -> None:
    source_display = source["image"].copy()
    contour_source = source["person_mask"]
    contours, _ = cv2.findContours(contour_source, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(source_display, contours, -1, (0, 0, 255), 3)
    similarity_matrix = np.asarray(metric["similarity"]["matrix"], dtype=np.float64) if metric["similarity"]["matrix"] else None
    if similarity_matrix is not None and "warped_source" in visuals:
        overlay = cv2.addWeighted(visuals["warped_source"], 0.5, output, 0.5, 0)
        edge_overlay = visuals["edge_overlay"]
        heatmap = visuals["heatmap"]
    else:
        overlay = np.zeros_like(output)
        edge_overlay = np.zeros_like(output)
        heatmap = np.zeros_like(output)
    bbox = metric["transformed_source_person_bounds"]["transformed_bbox"]
    details = np.full((300, 500, 3), 32, dtype=np.uint8)
    if bbox:
        sx0, sy0, sx1, sy1 = source["person_bbox"]
        ox0, oy0, ox1, oy1 = [int(round(value)) for value in bbox]
        ox0, oy0 = max(0, ox0), max(0, oy0)
        ox1, oy1 = min(output.shape[1], ox1), min(output.shape[0], oy1)
        source_crop = source["image"][sy0:sy1, sx0:sx1]
        output_crop = output[oy0:oy1, ox0:ox1]
        if source_crop.size and output_crop.size:
            details[:, :245] = fit_image(source_crop, 245, 300)
            details[:, 255:] = fit_image(output_crop, 245, 300)
            cv2.putText(details, "SOURCE SUBJECT", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(details, "OUTPUT EXPECTED REGION", (260, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            for fraction, label in [(0.23, "FACE"), (0.67, "GARMENT/HANDS"), (0.9, "FEET")]:
                y = int(300 * fraction)
                cv2.line(details, (0, y), (500, y), (0, 220, 220), 1)
                cv2.putText(details, label, (5, max(15, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 220), 1, cv2.LINE_AA)

    panels = [
        labeled_panel(source_display, "ORIGINAL CONDITION + OFFICIAL PERSON EXCLUSION"),
        labeled_panel(output, "ATTEMPT_001 OUTPUT (UNMODIFIED)"),
        labeled_panel(overlay, "SIMILARITY-ALIGNED 50/50 OVERLAY"),
        labeled_panel(edge_overlay, "EDGE OVERLAY: SOURCE GREEN / OUTPUT RED"),
        labeled_panel(heatmap, "BACKGROUND DIFFERENCE HEATMAP"),
        labeled_panel(details, "SUBJECT DETAIL: FACE / GARMENT+HANDS / FEET"),
    ]
    canvas = np.full((730, 1520, 3), 18, dtype=np.uint8)
    title = f"{record['request_id']} | {record['output_width']}x{record['output_height']} | {metric['primary_classification']}"
    cv2.putText(canvas, title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (245, 245, 245), 2, cv2.LINE_AA)
    for index, panel in enumerate(panels):
        row, column = divmod(index, 3)
        x, y = 10 + column * 505, 48 + row * 305
        canvas[y:y + 300, x:x + 500] = panel
    metadata = (
        f"features={metric['background_feature_count']}/{metric['output_feature_count']} matches={metric['match_count']} "
        f"inliers={metric['similarity']['inlier_count']} ratio={metric['similarity']['inlier_ratio']:.3f} "
        f"scale={metric['uniform_scale']} tx={metric['translation_x']} ty={metric['translation_y']} "
        f"median/p95={metric['median_reprojection_error_px']}/{metric['p95_reprojection_error_px']} | HUMAN DECISION: PENDING"
    )
    cv2.putText(canvas, metadata[:220], (15, 685), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.imwrite(str(destination), canvas)


def thumbnail_sheet(title: str, records: list[dict[str, Any]], destination: Path, columns: int = 4) -> None:
    tile_width, tile_height = 360, 245
    rows = max(1, math.ceil(len(records) / columns))
    canvas = np.full((55 + rows * tile_height, columns * tile_width, 3), 20, dtype=np.uint8)
    cv2.putText(canvas, title, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 245, 245), 2, cv2.LINE_AA)
    if not records:
        cv2.putText(canvas, "NO MACHINE-PASS CANDIDATES", (25, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 210, 255), 2, cv2.LINE_AA)
    for index, record in enumerate(records):
        image = cv2.imread(record["request_contact_sheet_path"])
        thumb = fit_image(image, tile_width - 8, tile_height - 30)
        row, column = divmod(index, columns)
        x, y = column * tile_width + 4, 55 + row * tile_height
        tile = np.full((tile_height, tile_width - 8, 3), 20, dtype=np.uint8)
        tile[:thumb.shape[0], :thumb.shape[1]] = thumb
        class_label = {
            "REGISTERED_SIMILARITY_PASS_CANDIDATE": "SIMILARITY_PASS_PENDING_HUMAN",
            "REGISTRATION_EVIDENCE_INSUFFICIENT": "EVIDENCE_INSUFFICIENT",
            "AFFINE_OR_PROJECTIVE_RECOMPOSITION_FAIL": "AFFINE_PROJECTIVE_FAIL",
            "BACKGROUND_GEOMETRY_CHANGED_FAIL": "BACKGROUND_GEOMETRY_FAIL",
            "PERSON_OR_POSE_REGISTRATION_FAIL": "PERSON_POSE_FAIL",
            "GARMENT_OR_HUMAN_VISUAL_FAIL": "VISUAL_REVIEW_FAIL",
        }[record["primary_classification"]]
        cv2.putText(tile, record["request_id"], (4, tile_height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(
            tile, f"{record['output_width']}x{record['output_height']} | {class_label}",
            (4, tile_height - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (190, 210, 225), 1, cv2.LINE_AA,
        )
        canvas[y:y + tile_height, x:x + tile.shape[1]] = tile
    cv2.imwrite(str(destination), canvas)


def dual_json(relative_data_path: str, risk_name: str, payload: dict[str, Any]) -> None:
    write_json(AUDIT_ROOT / relative_data_path, payload)
    write_json(RISK / risk_name, payload)


def resolution_cohorts(records: list[dict[str, Any]], metrics_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    all_ids = {item["request_id"] for item in records}
    for item in records:
        groups[(item["output_width"], item["output_height"])].append(item)
    cohorts = []
    for (width, height), items in sorted(groups.items(), key=lambda pair: (-len(pair[1]), pair[0])):
        cells: dict[str, list[str]] = defaultdict(list)
        for item in items:
            cells[f"{item['garment']}/{item['slot']}"] .append(item["request_id"])
        machine_cells = {
            cell for cell, request_ids in cells.items()
            if any(metrics_by_id[request_id]["primary_classification"] == "REGISTERED_SIMILARITY_PASS_CANDIDATE" for request_id in request_ids)
        }
        cohort_ids = {item["request_id"] for item in items}
        cohorts.append({
            "resolution": {"width": width, "height": height},
            "resolution_label": f"{width}x{height}",
            "output_count": len(items),
            "garment_count": len({item["garment"] for item in items}),
            "slot_count": len({item["slot"] for item in items}),
            "garment_slot_cell_count": len(cells),
            "candidate_count_by_cell": dict(sorted(cells.items())),
            "covers_24_of_24_cells": len(cells) == 24,
            "every_cell_at_least_one": len(cells) == 24 and all(cells.values()),
            "every_cell_has_two": len(cells) == 24 and all(len(value) == 2 for value in cells.values()),
            "missing_request_ids": sorted(all_ids - cohort_ids),
            "all_landscape": width > height,
            "same_size_formal_dataset_possible_raw": len(cells) == 24,
            "machine_pass_cell_count": len(machine_cells),
            "machine_pass_covers_24_of_24_cells": len(machine_cells) == 24,
        })
    dominant = cohorts[0]
    raw_complete = [item["resolution_label"] for item in cohorts if item["covers_24_of_24_cells"]]
    reliable_complete = [item["resolution_label"] for item in cohorts if item["machine_pass_covers_24_of_24_cells"]]
    return {
        "schema_version": "canondressgs.subject00.attempt001_exact_resolution_cohorts.v1",
        "task_id": TASK_ID,
        "cohort_count": len(cohorts),
        "cohorts": cohorts,
        "dominant_resolution": dominant["resolution_label"],
        "dominant_output_count": dominant["output_count"],
        "dominant_garment_slot_coverage": dominant["garment_slot_cell_count"],
        "dominant_machine_pass_cell_coverage": dominant["machine_pass_cell_count"],
        "raw_complete_single_resolution_cohorts": raw_complete,
        "reliable_complete_single_resolution_cohorts": reliable_complete,
        "complete_single_resolution_cohort_found": bool(reliable_complete),
        "paper_final": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace-audit-output", action="store_true")
    args = parser.parse_args()
    validate_source_gate()
    protocol = load_json(PROTOCOL_PATH)
    if protocol["task_id"] != TASK_ID or protocol["threshold_adjustment_policy"] != "NO_PER_IMAGE_OR_POST_RESULT_THRESHOLD_ADJUSTMENT":
        raise RuntimeError("registration protocol mismatch")
    protocol_sha = file_sha256(PROTOCOL_PATH)
    prepare_output(args.replace_audit_output)

    baseline = {
        "schema_version": "canondressgs.subject00.attempt001_registration_immutability_baseline.v1",
        "task_id": TASK_ID,
        "captured_at": AUDIT_TIMESTAMP,
        "attempt_001": inventory(ATTEMPT_001),
        "attempt_002": inventory(ATTEMPT_002),
        "attempt_003": inventory(ATTEMPT_003),
        "paper_final": False,
    }
    write_json(AUDIT_ROOT / "00_provenance" / "attempt_immutability_baseline.json", baseline)
    write_json(AUDIT_ROOT / "00_provenance" / "registration_audit_protocol.json", {**protocol, "protocol_file_sha256": protocol_sha})

    calibration_bytes = LOCAL_CALIBRATION.read_bytes()
    calibration = json.loads(calibration_bytes.decode("utf-8"))
    request_paths = sorted((ATTEMPT_001 / "03_generation_requests" / "requests").glob("*.json"))
    provenance_paths = sorted((ATTEMPT_001 / "11_provenance" / "requests").glob("*.json"))
    requests = {path.stem: load_json(path) for path in request_paths}
    provenance = {path.stem: load_json(path) for path in provenance_paths}
    expected = expected_request_ids()
    if sorted(requests) != sorted(expected) or sorted(provenance) != sorted(expected):
        raise RuntimeError("48-request manifest/provenance set mismatch")

    cloud_manifest = load_json(CLOUD_DATA_MANIFEST)
    archive_sizes = {
        f"subject00/{item['relative_path']}": int(item["size_bytes"])
        for item in cloud_manifest["files"]
    }
    mask_members = [f"subject00/masks/{binding['camera']}/00000000.jpg" for binding in SLOTS.values()]
    mask_payloads = archive_member_batch(mask_members, archive_sizes)

    source_cache: dict[str, dict[str, Any]] = {}
    for slot, binding in SLOTS.items():
        request = requests[f"subject00_O01_slot{slot[-2:]}_cand00"]
        source_path = Path(request["managed_tool_local_input_path"])
        if file_sha256(source_path) != request["source_input_sha256"]["identity_condition_rgb"]:
            raise RuntimeError(f"source RGB SHA mismatch: {slot}")
        mask_member = f"subject00/masks/{binding['camera']}/00000000.jpg"
        mask_bytes = mask_payloads[mask_member]
        if bytes_sha256(mask_bytes) != request["source_input_sha256"]["identity_condition_mask"]:
            raise RuntimeError(f"source mask SHA mismatch: {slot}")
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        mask_image = decode_image(mask_bytes, cv2.IMREAD_GRAYSCALE)
        person_mask = (mask_image >= int(protocol["background_mask"]["foreground_threshold"])).astype(np.uint8)
        ys, xs = np.where(person_mask > 0)
        if not len(xs):
            raise RuntimeError(f"empty person mask: {slot}")
        radius = max(20, round(0.02 * min(image.shape[:2])))
        expected_radius = int(protocol["background_mask"]["person_dilation_radius_px_for_1330x1150"])
        if radius != expected_radius:
            raise RuntimeError(f"person dilation changed: {radius}")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        exclusion = cv2.dilate(person_mask, kernel)
        background_mask = ((1 - exclusion) * 255).astype(np.uint8)
        border = int(protocol["background_mask"]["image_border_exclusion_px"])
        background_mask[:border] = 0
        background_mask[-border:] = 0
        background_mask[:, :border] = 0
        background_mask[:, -border:] = 0
        sift = cv2.SIFT_create(
            nfeatures=int(protocol["feature_matching"]["nfeatures"]),
            contrastThreshold=float(protocol["feature_matching"]["contrast_threshold"]),
            edgeThreshold=float(protocol["feature_matching"]["edge_threshold"]),
        )
        keypoints, descriptors = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), background_mask)
        source_cache[slot] = {
            "path": source_path,
            "sha256": file_sha256(source_path),
            "image": image,
            "mask_member": mask_member,
            "mask_sha256": bytes_sha256(mask_bytes),
            "person_mask": person_mask,
            "person_bbox": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
            "dilation_radius_px": radius,
            "background_mask": background_mask,
            "keypoints": keypoints,
            "descriptors": descriptors,
            "camera_intrinsics": calibration[binding["camera"]]["K"],
        }

    records: list[dict[str, Any]] = []
    metric_records: list[dict[str, Any]] = []
    photometric_records: list[dict[str, Any]] = []
    camera_candidates: list[dict[str, Any]] = []
    output_hashes: Counter[str] = Counter()
    visual_cache: dict[str, dict[str, np.ndarray]] = {}
    outputs_cache: dict[str, np.ndarray] = {}
    for index, request_id in enumerate(expected, start=1):
        request = requests[request_id]
        proof = provenance[request_id]
        match = REQUEST_PATTERN.fullmatch(request_id)
        if match is None:
            raise RuntimeError(f"bad request ID: {request_id}")
        garment, slot_number, candidate_text = match.groups()
        slot = f"slot_{slot_number}"
        candidate_index = int(candidate_text)
        binding = SLOTS[slot]
        output_path = Path(proof["output_path"])
        output_sha = file_sha256(output_path)
        output_hashes[output_sha] += 1
        output = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
        if output is None:
            raise RuntimeError(f"PNG parse failed: {request_id}")
        height, width = output.shape[:2]
        source = source_cache[slot]
        binding_checks = {
            "request_id": request["request_id"] == proof["request_id"] == request_id,
            "garment": request["garment_id"] == proof["garment_id"] == garment,
            "slot": request["slot_id"] == proof["slot_id"] == slot,
            "camera": request["camera_id"] == proof["camera_id"] == binding["camera_id"],
            "candidate": request["candidate_index"] == candidate_index,
            "frame": request["pose_frame_id"] == proof["pose_frame_id"] == 0,
            "source_rgb_sha": request["source_input_sha256"]["identity_condition_rgb"] == proof["input_sha256"]["identity_condition_rgb"] == source["sha256"],
            "source_mask_sha": request["source_input_sha256"]["identity_condition_mask"] == proof["input_sha256"]["identity_condition_mask"] == source["mask_sha256"],
            "calibration_sha": request["source_input_sha256"]["camera"] == proof["input_sha256"]["camera"] == bytes_sha256(calibration_bytes),
            "output_sha": proof["output_sha256"] == output_sha,
            "output_size": proof["width"] == width and proof["height"] == height,
            "output_format": proof["format"] == "PNG",
            "prompt_sha": request["actual_prompt_sha256"] == proof["actual_prompt_sha256"],
        }
        if not all(binding_checks.values()):
            raise RuntimeError(f"provenance binding failed: {request_id}: {binding_checks}")
        metric, visuals = calculate_registration(request_id, source, output, protocol)
        visual_cache[request_id] = visuals
        outputs_cache[request_id] = output
        metric.update({
            "garment": garment, "slot": slot, "camera": binding["camera"],
            "candidate_id": f"cand{candidate_index:02d}", "orientation": binding["orientation"],
            "output_resolution": {"width": width, "height": height},
            "protocol_sha256": protocol_sha,
        })
        metric_records.append(metric)
        photometric_records.append({
            "request_id": request_id,
            **metric["photometric"],
            "primary_classification": metric["primary_classification"],
        })
        record = {
            "request_id": request_id,
            "garment": garment,
            "slot": slot,
            "camera": binding["camera"],
            "camera_id": binding["camera_id"],
            "candidate_id": f"cand{candidate_index:02d}",
            "candidate_index": candidate_index,
            "orientation": binding["orientation"],
            "pose_frame_id": 0,
            "original_condition_path": str(source["path"]),
            "original_condition_sha256": source["sha256"],
            "original_condition_mask_archive": str(ARCHIVE),
            "original_condition_mask_member": source["mask_member"],
            "original_condition_mask_sha256": source["mask_sha256"],
            "original_width": source["image"].shape[1],
            "original_height": source["image"].shape[0],
            "generated_output_path": str(output_path),
            "generated_output_sha256": output_sha,
            "output_width": width,
            "output_height": height,
            "generation_timestamp": proof["ended_at"],
            "provider": proof["generation_backend"],
            "provider_model_id": proof["provider_model_id"],
            "retry_count": proof["technical_retry_count"],
            "postprocessing_count": "NOT_RECORDED",
            "hidden_postprocessing_replacement_detected": False,
            "duplicate_status": "PENDING_GLOBAL_CHECK",
            "original_manifest_binding_status": "PASS" if all(binding_checks.values()) else "FAIL",
            "binding_checks": binding_checks,
            "png_parse_status": "PASS",
            "primary_classification": metric["primary_classification"],
        }
        records.append(record)
        print(f"[{index:02d}/48] {request_id} {width}x{height} {metric['primary_classification']}", flush=True)

    duplicates = {sha for sha, count in output_hashes.items() if count > 1}
    for record in records:
        record["duplicate_status"] = "DUPLICATE" if record["generated_output_sha256"] in duplicates else "UNIQUE"

    metrics_by_id = {item["request_id"]: item for item in metric_records}
    records_by_id = {item["request_id"]: item for item in records}
    distribution = Counter(f"{item['output_width']}x{item['output_height']}" for item in records)
    historical = {
        "1024x1536": 5, "1166x1349": 1, "1168x1346": 1, "1168x1347": 1,
        "1173x1341": 1, "1179x1334": 1, "1348x1167": 1, "1349x1166": 34, "1350x1165": 3,
    }
    distribution_payload = {
        "schema_version": "canondressgs.subject00.attempt001_resolution_distribution_recomputed.v1",
        "task_id": TASK_ID,
        "source": "ACTUAL_48_PNG_FILES_REPARSED",
        "output_count": len(records),
        "exact_distribution": dict(sorted(distribution.items())),
        "exact_1024x1536_count": distribution["1024x1536"],
        "landscape_count": sum(count for label, count in distribution.items() if int(label.split("x")[0]) > int(label.split("x")[1])),
        "portrait_incompatible_non_native_count": sum(count for label, count in distribution.items() if int(label.split("x")[0]) < int(label.split("x")[1]) and label != "1024x1536"),
        "parse_failure_count": 0,
        "historical_distribution": historical,
        "historical_match": dict(sorted(distribution.items())) == historical,
        "paper_final": False,
    }
    cohorts = resolution_cohorts(records, metrics_by_id)

    for record in records:
        request_id = record["request_id"]
        metric = metrics_by_id[request_id]
        camera_data = calibration[record["camera"]]
        if metric["primary_classification"] == "REGISTERED_SIMILARITY_PASS_CANDIDATE":
            axis = metric["axis_scale_translation"]
            source_k = np.asarray(camera_data["K"], dtype=np.float64).reshape(3, 3)
            sx, sy = float(axis["scale_x"]), float(axis["scale_y"])
            tx, ty = float(axis["translation_x"]), float(axis["translation_y"])
            candidate_k = source_k.copy()
            candidate_k[0, 0] = sx * source_k[0, 0]
            candidate_k[1, 1] = sy * source_k[1, 1]
            candidate_k[0, 2] = sx * source_k[0, 2] + tx
            candidate_k[1, 2] = sy * source_k[1, 2] + ty
            camera_candidates.append({
                "request_id": request_id,
                "source_calibration_archive": str(ARCHIVE),
                "source_calibration_member": "subject00/calibration.json",
                "source_calibration_resolved_local_path": str(LOCAL_CALIBRATION),
                "source_calibration_sha256": bytes_sha256(calibration_bytes),
                "camera": record["camera"],
                "source_intrinsics": {"fx": source_k[0, 0], "fy": source_k[1, 1], "cx": source_k[0, 2], "cy": source_k[1, 2]},
                "transform": {"scale_x": sx, "scale_y": sy, "translation_x": tx, "translation_y": ty, "rotation_degrees": metric["rotation_degrees"], "normalized_shear": metric["normalized_shear"]},
                "derived_intrinsics": {"fx": candidate_k[0, 0], "fy": candidate_k[1, 1], "cx": candidate_k[0, 2], "cy": candidate_k[1, 2]},
                "width_out": record["output_width"],
                "height_out": record["output_height"],
                "residual": axis["errors_on_similarity_inliers"],
                "camera_compatible": True,
                "materialized": False,
            })

    classification_counts = Counter(item["primary_classification"] for item in metric_records)
    coverage_records = []
    for garment in GARMENTS:
        for slot in SLOTS:
            request_ids = [f"subject00_{garment}_slot{slot[-2:]}_cand{candidate:02d}" for candidate in range(2)]
            candidates = [records_by_id[request_id] for request_id in request_ids]
            machine_count = sum(metrics_by_id[request_id]["primary_classification"] == "REGISTERED_SIMILARITY_PASS_CANDIDATE" for request_id in request_ids)
            coverage_records.append({
                "cell": f"{garment}/{slot}",
                "garment": garment,
                "slot": slot,
                "camera": SLOTS[slot]["camera"],
                "cand00_classification": metrics_by_id[request_ids[0]]["primary_classification"],
                "cand01_classification": metrics_by_id[request_ids[1]]["primary_classification"],
                "machine_pass_candidate_count": machine_count,
                "exact_resolution_cohort": [f"{item['output_width']}x{item['output_height']}" for item in candidates],
                "user_review_required_count": machine_count,
                "no_valid_candidate": machine_count == 0,
                "human_final_decision": None,
            })
    machine_pass_cells = sum(item["machine_pass_candidate_count"] > 0 for item in coverage_records)
    no_valid_cells = [item["cell"] for item in coverage_records if item["no_valid_candidate"]]
    dominant_label = cohorts["dominant_resolution"]
    dominant_missing_cells = []
    minimum_reruns = []
    for cell in coverage_records:
        request_ids = [f"subject00_{cell['garment']}_slot{cell['slot'][-2:]}_cand{candidate:02d}" for candidate in range(2)]
        dominant_machine = [
            request_id for request_id in request_ids
            if f"{records_by_id[request_id]['output_width']}x{records_by_id[request_id]['output_height']}" == dominant_label
            and metrics_by_id[request_id]["primary_classification"] == "REGISTERED_SIMILARITY_PASS_CANDIDATE"
        ]
        if not dominant_machine:
            dominant_missing_cells.append(cell["cell"])
            minimum_reruns.append({
                "cell": cell["cell"],
                "source_request_ids": request_ids,
                "target_resolution": dominant_label,
                "minimum_new_candidate_count": 1,
                "generation_authorized": False,
            })
    coverage_payload = {
        "schema_version": "canondressgs.subject00.attempt001_garment_slot_coverage.v1",
        "task_id": TASK_ID,
        "cell_count": len(coverage_records),
        "machine_pass_24_cell_coverage": machine_pass_cells,
        "no_valid_candidate_cells": no_valid_cells,
        "dominant_resolution": dominant_label,
        "dominant_resolution_machine_pass_missing_cells": dominant_missing_cells,
        "minimum_targeted_rerun_request_count": len(minimum_reruns),
        "minimum_targeted_rerun_requests": minimum_reruns,
        "records": coverage_records,
        "accepted_count": 0,
        "teacher_target_count": 0,
        "paper_final": False,
    }

    if cohorts["complete_single_resolution_cohort_found"]:
        registration_feasibility = "FULLY_FEASIBLE"
        cohort_feasibility = "COMPLETE_SINGLE_RESOLUTION_COHORT"
        per_view_requirement = "NOT_REQUIRED"
        salvage = "FULL_24_CELL_SALVAGE_CANDIDATE"
        final_classification = "SUBJECT00_NATIVE_LANDSCAPE_FULL_COVERAGE_READY_FOR_USER_VISUAL_REVIEW"
        next_task = "USER_REVIEW_SUBJECT00_NATIVE_LANDSCAPE_FULL_COHORT"
    elif classification_counts["REGISTERED_SIMILARITY_PASS_CANDIDATE"] > 0:
        registration_feasibility = "PARTIALLY_FEASIBLE"
        cohort_feasibility = "PARTIAL_SINGLE_RESOLUTION_COHORT"
        per_view_requirement = "REQUIRED_BUT_SMALL_ADAPTER" if machine_pass_cells == 24 else "UNRESOLVED"
        salvage = "PARTIAL_SALVAGE_TARGETED_RERUN_CANDIDATE"
        final_classification = "SUBJECT00_NATIVE_LANDSCAPE_PARTIAL_SALVAGE_TARGETED_GAPS_IDENTIFIED"
        next_task = "USER_REVIEW_SALVAGE_CANDIDATES_AND_AUTHORIZE_ONLY_MISSING_CELL_CANARY"
    elif classification_counts["REGISTRATION_EVIDENCE_INSUFFICIENT"] == 48:
        registration_feasibility = "INSUFFICIENT_EVIDENCE"
        cohort_feasibility = "NO_USEFUL_SINGLE_RESOLUTION_COHORT"
        per_view_requirement = "UNRESOLVED"
        salvage = "INSUFFICIENT_FOR_FORMAL_DATASET"
        final_classification = "SUBJECT00_NATIVE_LANDSCAPE_AUDIT_BLOCKED_BY_MISSING_EVIDENCE"
        next_task = "USER_RESOLVE_SUBJECT00_NATIVE_LANDSCAPE_EVIDENCE_BLOCKER"
    else:
        registration_feasibility = "NOT_FEASIBLE"
        cohort_feasibility = "NO_USEFUL_SINGLE_RESOLUTION_COHORT"
        per_view_requirement = "UNRESOLVED"
        salvage = "REGISTRATION_FAILURE_NO_SALVAGE"
        final_classification = "SUBJECT00_NATIVE_LANDSCAPE_REGISTRATION_NOT_RELIABLE"
        next_task = "USER_SELECT_VALIDITY_MASK_IMPLEMENTATION_OR_1024x1150_BACKEND_CANARY"

    for record in records:
        request_id = record["request_id"]
        contact_path = AUDIT_ROOT / "05_contact_sheets" / "requests" / f"{request_id}_registration_review.png"
        request_visualization(record, metrics_by_id[request_id], source_cache[record["slot"]], outputs_cache[request_id], visual_cache[request_id], contact_path)
        record["request_contact_sheet_path"] = str(contact_path)

    sheet_paths = {
        "overall_resolution_cohort": AUDIT_ROOT / "05_contact_sheets" / "attempt001_resolution_cohort_sheet.png",
        "O01": AUDIT_ROOT / "05_contact_sheets" / "attempt001_O01_16_sheet.png",
        "O03": AUDIT_ROOT / "05_contact_sheets" / "attempt001_O03_16_sheet.png",
        "O04": AUDIT_ROOT / "05_contact_sheets" / "attempt001_O04_16_sheet.png",
        "machine_pass": AUDIT_ROOT / "05_contact_sheets" / "attempt001_machine_pass_candidates_sheet.png",
        "failed_examples": AUDIT_ROOT / "05_contact_sheets" / "attempt001_failed_registration_examples_sheet.png",
        "coverage_24_cell": AUDIT_ROOT / "05_contact_sheets" / "attempt001_24_cell_coverage_sheet.png",
    }
    thumbnail_sheet("ATTEMPT 001: ALL 48 OUTPUTS BY EXACT RESOLUTION", records, sheet_paths["overall_resolution_cohort"])
    for garment in GARMENTS:
        thumbnail_sheet(f"ATTEMPT 001: {garment} 16-CANDIDATE REGISTRATION AUDIT", [item for item in records if item["garment"] == garment], sheet_paths[garment])
    machine_records = [item for item in records if metrics_by_id[item["request_id"]]["primary_classification"] == "REGISTERED_SIMILARITY_PASS_CANDIDATE"]
    failed_records = [item for item in records if item not in machine_records][:16]
    thumbnail_sheet("MACHINE-PASS CANDIDATES: HUMAN REVIEW STILL REQUIRED", machine_records, sheet_paths["machine_pass"])
    thumbnail_sheet("FAILED REGISTRATION EXAMPLES", failed_records, sheet_paths["failed_examples"])
    coverage_representatives = [records_by_id[f"subject00_{garment}_slot{slot[-2:]}_cand00"] for garment in GARMENTS for slot in SLOTS]
    thumbnail_sheet("24 GARMENT-SLOT CELLS: CAND00 REPRESENTATIVE", coverage_representatives, sheet_paths["coverage_24_cell"])

    human_fields = [
        "single_person", "identity_consistency", "garment_correctness", "pose_preservation",
        "camera_direction", "hands_complete", "feet_complete", "subject_scale", "subject_center",
        "silhouette", "garment_boundary", "background_perspective", "background_geometry",
        "visible_generation_artifacts", "final_decision",
    ]
    review_records = []
    for record in records:
        review_records.append({
            "request_id": record["request_id"],
            "machine_classification": metrics_by_id[record["request_id"]]["primary_classification"],
            "request_contact_sheet_path": record["request_contact_sheet_path"],
            **{field: None for field in human_fields},
            "accepted": None,
            "teacher_target": None,
            "reviewer": None,
            "review_timestamp": None,
        })
    review_payload = {
        "schema_version": "canondressgs.subject00.attempt001_native_landscape_human_review.v1",
        "task_id": TASK_ID,
        "status": "PENDING_USER_REVIEW",
        "record_count": 48,
        "records": review_records,
        "accepted_count": 0,
        "teacher_target_count": 0,
        "contact_sheet_paths": {key: str(value) for key, value in sheet_paths.items()},
        "paper_final": False,
    }

    inventory_payload = {
        "schema_version": "canondressgs.subject00.attempt001_native_landscape_full_inventory.v1",
        "task_id": TASK_ID,
        "request_count": len(requests),
        "output_count": len(records),
        "png_parse_pass_count": sum(item["png_parse_status"] == "PASS" for item in records),
        "provenance_binding_pass_count": sum(item["original_manifest_binding_status"] == "PASS" for item in records),
        "duplicate_sha_groups": sorted(duplicates),
        "postprocessing_evidence": "No derivative output replacement was found; per-request postprocessing count was not recorded by Attempt 001 provenance.",
        "records": records,
        "attempt_tree_digests": {
            "attempt_001": {key: baseline["attempt_001"][key] for key in ["root", "file_count", "tree_sha256"]},
            "attempt_002": {key: baseline["attempt_002"][key] for key in ["root", "file_count", "tree_sha256"]},
            "attempt_003": {key: baseline["attempt_003"][key] for key in ["root", "file_count", "tree_sha256"]},
        },
        "paper_final": False,
    }
    provenance_payload = {
        "schema_version": "canondressgs.subject00.attempt001_provenance_binding_audit.v1",
        "task_id": TASK_ID,
        "protocol_sha256": protocol_sha,
        "request_count": 48,
        "pass_count": 48,
        "source_archive": str(ARCHIVE),
        "source_calibration_sha256": bytes_sha256(calibration_bytes),
        "records": [{
            "request_id": item["request_id"],
            "binding_status": item["original_manifest_binding_status"],
            "binding_checks": item["binding_checks"],
            "source_rgb_sha256": item["original_condition_sha256"],
            "source_mask_sha256": item["original_condition_mask_sha256"],
            "output_sha256": item["generated_output_sha256"],
        } for item in records],
        "paper_final": False,
    }
    registration_payload = {
        "schema_version": "canondressgs.subject00.attempt001_background_registration_metrics.v1",
        "task_id": TASK_ID,
        "protocol_sha256": protocol_sha,
        "audit_count": len(metric_records),
        "classification_counts": {name: classification_counts[name] for name in CLASSIFICATIONS},
        "records": metric_records,
        "paper_final": False,
    }
    photometric_payload = {
        "schema_version": "canondressgs.subject00.attempt001_background_photometric_metrics.v1",
        "task_id": TASK_ID,
        "protocol_sha256": protocol_sha,
        "audit_count": len(photometric_records),
        "records": photometric_records,
        "paper_final": False,
    }
    camera_payload = {
        "schema_version": "canondressgs.subject00.attempt001_camera_intrinsics_candidates.v1",
        "task_id": TASK_ID,
        "candidate_count": len(camera_candidates),
        "invalid_count": 48 - len(camera_candidates),
        "materialized_count": 0,
        "records": camera_candidates,
        "paper_final": False,
    }
    summary = {
        "schema_version": "canondressgs.subject00.attempt001_native_landscape_registration_final_summary.v1",
        "task_id": TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "new_branch": BRANCH,
        "worktree": str(ROOT),
        "attempt_001_root": str(ATTEMPT_001),
        "audit_output_root": str(AUDIT_ROOT),
        "protocol_sha256": protocol_sha,
        "request_count": 48,
        "output_count": 48,
        "png_parse_pass_count": 48,
        "provenance_binding_pass_count": 48,
        "resolution_distribution": dict(sorted(distribution.items())),
        "exact_resolution_cohorts": [item["resolution_label"] for item in cohorts["cohorts"]],
        "dominant_resolution": cohorts["dominant_resolution"],
        "dominant_cohort_output_count": cohorts["dominant_output_count"],
        "dominant_cohort_garment_slot_coverage": cohorts["dominant_garment_slot_coverage"],
        "complete_single_resolution_cohort_found": cohorts["complete_single_resolution_cohort_found"],
        "background_registration_audit_count": 48,
        "classification_counts": {name: classification_counts[name] for name in CLASSIFICATIONS},
        "camera_intrinsics_candidate_count": len(camera_candidates),
        "camera_intrinsics_invalid_count": 48 - len(camera_candidates),
        "machine_pass_24_cell_coverage": machine_pass_cells,
        "no_valid_candidate_cells": no_valid_cells,
        "minimum_targeted_rerun_requests": minimum_reruns,
        "registration_feasibility": registration_feasibility,
        "exact_resolution_cohort_feasibility": cohort_feasibility,
        "per_view_resolution_requirement": per_view_requirement,
        "dataset_salvage_classification": salvage,
        "attempt_001_mutations": 0,
        "attempt_002_mutations": 0,
        "attempt_003_mutations": 0,
        "new_generation_calls": 0,
        "external_api_calls": 0,
        "api_key_reads": 0,
        "cloud_image_writes": 0,
        "accepted_count": 0,
        "teacher_target_count": 0,
        "remaining_bulk_generation_authorization": "DENIED",
        "formal_base_status": "PENDING",
        "subject00_paper_positive_claim_count": 0,
        "paper_modifications": 0,
        "contact_sheet_paths": {key: str(value) for key, value in sheet_paths.items()},
        "review_manifest_path": str(AUDIT_ROOT / "06_human_review" / "attempt001_human_review_manifest.json"),
        "paper_final": False,
        "final_classification": final_classification,
        "next_task": next_task,
    }

    dual_json("01_inventory/attempt001_full_inventory.json", "subject00_attempt001_native_landscape_full_inventory_20260726.json", inventory_payload)
    dual_json("01_inventory/attempt001_resolution_distribution.json", "subject00_attempt001_resolution_distribution_20260726.json", distribution_payload)
    dual_json("01_inventory/attempt001_exact_resolution_cohorts.json", "subject00_attempt001_exact_resolution_cohorts_20260726.json", cohorts)
    dual_json("01_inventory/attempt001_provenance_binding_audit.json", "subject00_attempt001_provenance_binding_audit_20260726.json", provenance_payload)
    dual_json("02_registration_metrics/attempt001_background_registration_metrics.json", "subject00_attempt001_background_registration_metrics_20260726.json", registration_payload)
    write_json(AUDIT_ROOT / "02_registration_metrics" / "attempt001_background_photometric_metrics.json", photometric_payload)
    dual_json("03_camera_candidates/attempt001_camera_intrinsics_candidates.json", "subject00_attempt001_camera_intrinsics_candidates_20260726.json", camera_payload)
    dual_json("04_coverage/attempt001_garment_slot_coverage_matrix.json", "subject00_attempt001_garment_slot_coverage_20260726.json", coverage_payload)
    write_json(AUDIT_ROOT / "04_coverage" / "attempt001_minimum_targeted_rerun_requests.json", {
        "schema_version": "canondressgs.subject00.attempt001_minimum_targeted_reruns.v1",
        "task_id": TASK_ID, "count": len(minimum_reruns), "records": minimum_reruns,
        "generation_authorized": False, "paper_final": False,
    })
    dual_json("06_human_review/attempt001_human_review_manifest.json", "subject00_attempt001_native_landscape_human_review_registry_20260726.json", review_payload)
    dual_json("07_final_summary/attempt001_native_landscape_registration_final_summary.json", "subject00_attempt001_native_landscape_registration_final_summary_20260726.json", summary)

    report = f"""# Subject00 Attempt 001 Native-Landscape Registration Audit

- Task: `{TASK_ID}`
- Protocol SHA-256: `{protocol_sha}`
- Outputs audited: `48/48`
- Dominant exact resolution: `{cohorts['dominant_resolution']}` (`{cohorts['dominant_output_count']}` outputs, `{cohorts['dominant_garment_slot_coverage']}/24` raw cells)
- Machine registration pass candidates: `{classification_counts['REGISTERED_SIMILARITY_PASS_CANDIDATE']}`
- Machine-pass cell coverage: `{machine_pass_cells}/24`
- Complete reliable single-resolution cohort: `{str(cohorts['complete_single_resolution_cohort_found']).lower()}`
- Registration feasibility: `{registration_feasibility}`
- Dataset salvage classification: `{salvage}`
- Final classification: `{final_classification}`

The audit used frozen official person masks expanded by 23 px, SIFT background matching, deterministic RANSAC, and the preregistered similarity/structure gates. Machine-pass candidates remain pending human visual review. No output is accepted or written as a Teacher target.

Human review must use the request-level panels and preserve all final decision fields as null until the user adjudicates them. `PAPER_FINAL=false`.
"""
    write_text(AUDIT_ROOT / "07_final_summary" / "ATTEMPT001_NATIVE_LANDSCAPE_REGISTRATION_AUDIT_REPORT.md", report)
    write_text(RISK / "SUBJECT00_ATTEMPT001_NATIVE_LANDSCAPE_REGISTRATION_AUDIT_REPORT_20260726.md", report)
    handoff = {
        **summary,
        "schema_version": "canondressgs.subject00.attempt001_native_landscape_registration_handoff.v1",
        "inventory_registry": "paper_protocol/reviewer_risk/subject00_attempt001_native_landscape_full_inventory_20260726.json",
        "registration_registry": "paper_protocol/reviewer_risk/subject00_attempt001_background_registration_metrics_20260726.json",
        "coverage_registry": "paper_protocol/reviewer_risk/subject00_attempt001_garment_slot_coverage_20260726.json",
        "human_review_registry": "paper_protocol/reviewer_risk/subject00_attempt001_native_landscape_human_review_registry_20260726.json",
        "execution_status": "WAITING_FOR_USER_VISUAL_REVIEW_OR_PROTOCOL_SELECTION",
    }
    write_json(HANDOFF / "subject00_attempt001_native_landscape_registration_audit_handoff_20260726.json", handoff)
    print(json.dumps({
        "status": "AUDIT_BUILT",
        "classification_counts": dict(classification_counts),
        "dominant_resolution": cohorts["dominant_resolution"],
        "dominant_raw_cell_coverage": cohorts["dominant_garment_slot_coverage"],
        "dominant_machine_cell_coverage": cohorts["dominant_machine_pass_cell_coverage"],
        "machine_pass_24_cell_coverage": machine_pass_cells,
        "minimum_targeted_rerun_count": len(minimum_reruns),
        "final_classification": final_classification,
        "next_task": next_task,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
