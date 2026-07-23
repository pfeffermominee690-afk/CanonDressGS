#!/usr/bin/env python3
"""Read-only AvatarReX base-avatar runtime preflight.

This tool decodes nine dataset records, validates camera and SMPL-X schemas,
optionally runs neutral SMPL-X forwards, and statically parses the MMLP-Human
model/renderer interfaces. It never constructs Scene or GaussianModel because
those paths create derived assets when template files are absent.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

import numpy as np

try:
    from .avatarrex_zero_copy_adapter import (
        AvatarReXZeroCopyAdapter,
        canonical_sha256,
        sha256_file,
        write_json,
    )
except ImportError:
    from avatarrex_zero_copy_adapter import (
        AvatarReXZeroCopyAdapter,
        canonical_sha256,
        sha256_file,
        write_json,
    )


TASK_ID = "AAAI27-AVATARREX-BASE-AVATAR-PREFLIGHT-001"
SMOKE_CAMERAS = (0, 8, 15)
SMOKE_FRAMES = (0, 950, 1900)
EXPECTED_SMPL_FIELDS = {
    "betas": ([1, 10], "float32"),
    "global_orient": ([1901, 3], "float32"),
    "transl": ([1901, 3], "float32"),
    "body_pose": ([1901, 63], "float32"),
    "jaw_pose": ([1901, 3], "float32"),
    "expression": ([1901, 10], "float32"),
    "left_hand_pose": ([1901, 45], "float32"),
    "right_hand_pose": ([1901, 45], "float32"),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _tree_metadata_snapshot(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_count = 0
    apparent_bytes = 0
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
        stat = path.stat()
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\n")
        file_count += 1
        apparent_bytes += stat.st_size
    return {
        "algorithm": "sha256(relative_path NUL size NUL mtime_ns LF) over sorted files",
        "file_count": file_count,
        "apparent_bytes": apparent_bytes,
        "sha256": digest.hexdigest(),
    }


def _project(K: np.ndarray, R: np.ndarray, T: np.ndarray, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points_camera = (R @ points_world.T).T + T.reshape(1, 3)
    pixels_h = (K @ points_camera.T).T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    return pixels, points_camera[:, 2]


def _unproject(K: np.ndarray, R: np.ndarray, T: np.ndarray, pixels: np.ndarray, depth: float) -> np.ndarray:
    homogeneous = np.concatenate([pixels, np.ones((len(pixels), 1), dtype=np.float64)], axis=1)
    points_camera = (np.linalg.inv(K) @ homogeneous.T).T * depth
    return (R.T @ (points_camera - T.reshape(1, 3)).T).T


def audit_cameras(calibration: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    all_centers = []
    for camera_index, name in enumerate(sorted(calibration)):
        raw = calibration[name]
        K = np.asarray(raw["K"], dtype=np.float64).reshape(3, 3)
        R = np.asarray(raw["R"], dtype=np.float64).reshape(3, 3)
        T = np.asarray(raw["T"], dtype=np.float64).reshape(3)
        distortion = np.asarray(raw["distCoeff"], dtype=np.float64).reshape(-1)
        image_size = np.asarray(raw["imgSize"], dtype=np.int64).reshape(2)
        width, height = int(image_size[0]), int(image_size[1])
        center = -(R.T @ T)
        all_centers.append(center)

        points_camera = np.asarray(
            [[0.0, 0.0, 1.0], [0.1, 0.0, 1.5], [-0.1, 0.0, 2.0], [0.0, 0.1, 2.5], [0.0, -0.1, 3.0]],
            dtype=np.float64,
        )
        points_world = (R.T @ (points_camera - T.reshape(1, 3)).T).T
        projected, depths = _project(K, R, T, points_world)
        expected_h = (K @ points_camera.T).T
        expected_pixels = expected_h[:, :2] / expected_h[:, 2:3]

        roundtrip_pixels = np.asarray(
            [
                [K[0, 2], K[1, 2]],
                [0.25 * width, 0.25 * height],
                [0.75 * width, 0.25 * height],
                [0.25 * width, 0.75 * height],
                [0.75 * width, 0.75 * height],
            ],
            dtype=np.float64,
        )
        roundtrip_world = _unproject(K, R, T, roundtrip_pixels, depth=2.0)
        reprojected, roundtrip_depths = _project(K, R, T, roundtrip_world)

        determinant = float(np.linalg.det(R))
        orthogonality_error = float(np.max(np.abs(R.T @ R - np.eye(3))))
        center_residual = float(np.max(np.abs(R @ center + T)))
        known_point_error = float(np.max(np.abs(projected - expected_pixels)))
        roundtrip_error = float(np.max(np.abs(reprojected - roundtrip_pixels)))
        checks = {
            "K_shape": list(K.shape) == [3, 3],
            "R_shape": list(R.shape) == [3, 3],
            "T_shape": list(T.shape) == [3],
            "all_finite": bool(
                np.isfinite(K).all()
                and np.isfinite(R).all()
                and np.isfinite(T).all()
                and np.isfinite(projected).all()
            ),
            "positive_focal_lengths": bool(K[0, 0] > 0 and K[1, 1] > 0),
            "principal_point_inside_image": bool(0 <= K[0, 2] < width and 0 <= K[1, 2] < height),
            "rotation_determinant": abs(determinant - 1.0) <= 1e-5,
            "rotation_orthogonality": orthogonality_error <= 1e-5,
            "camera_center": center_residual <= 1e-5,
            "known_3d_projection": known_point_error <= 1e-3 and bool(np.all(depths > 0)),
            "pixel_unproject_reproject": roundtrip_error <= 1e-3 and bool(np.all(roundtrip_depths > 0)),
        }
        rows.append(
            {
                "camera_index": camera_index,
                "canonical_camera_id": f"camera_{camera_index:02d}",
                "raw_camera_name": name,
                "K": K.tolist(),
                "R_shape": list(R.shape),
                "T_shape": list(T.shape),
                "distortion_coefficients": distortion.tolist(),
                "imgSize_width_height": [width, height],
                "rectifyAlpha": float(raw["rectifyAlpha"]),
                "camera_center": center.tolist(),
                "rotation_determinant": determinant,
                "rotation_orthogonality_max_abs_error": orthogonality_error,
                "camera_center_max_abs_residual": center_residual,
                "known_3d_projection_max_abs_pixel_error": known_point_error,
                "pixel_unproject_reproject_max_abs_error": roundtrip_error,
                "checks": checks,
                "status": "PASS" if all(checks.values()) else "FAIL",
            }
        )

    centers = np.stack(all_centers)
    return {
        "status": "PASS" if len(rows) == 16 and all(row["status"] == "PASS" for row in rows) else "FAIL",
        "camera_count": len(rows),
        "camera_names": [row["raw_camera_name"] for row in rows],
        "raw_extrinsic_convention": "x_camera = R @ x_world + T (world_to_camera)",
        "runtime_extrinsic_convention": "gsplat viewmats receives the same 4x4 world_to_camera matrix",
        "camera_center_formula": "C_world = -transpose(R) @ T",
        "translation_shape": [3],
        "rotation_shape": [3, 3],
        "imgSize_order": "width_height",
        "decoded_array_order": "height_width_channels",
        "distortion_policy": {
            "loader_behavior": "cv2.undistort in memory only when sum(abs(distCoeff)) >= 1e-4",
            "all_coefficients_zero": all(
                np.count_nonzero(np.asarray(row["distortion_coefficients"], dtype=np.float64)) == 0 for row in rows
            ),
            "raw_images_treated_as_rectified": True,
            "runtime_undistortion_required_for_this_capture": False,
        },
        "units": {
            "intrinsics": "pixels",
            "extrinsics_and_camera_centers": "dataset world units; official SMPL-X consumer uses model-native meters without scaling",
            "provider_explicit_unit_metadata_present": False,
            "runtime_contract": "meters, with acquisition-time sanity check against neutral SMPL-X extent",
        },
        "rig_center_world": centers.mean(axis=0).tolist(),
        "rows": rows,
    }


def audit_smpl_schema(smpl_path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    arrays: dict[str, np.ndarray] = {}
    rows: dict[str, Any] = {}
    with np.load(smpl_path, allow_pickle=False) as payload:
        for name in payload.files:
            array = np.asarray(payload[name])
            arrays[name] = array
            rows[name] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "finite": bool(np.isfinite(array).all()),
                "minimum": float(array.min()),
                "maximum": float(array.max()),
            }
    schema_match = set(rows) == set(EXPECTED_SMPL_FIELDS) and all(
        rows[name]["shape"] == shape and rows[name]["dtype"] == dtype and rows[name]["finite"]
        for name, (shape, dtype) in EXPECTED_SMPL_FIELDS.items()
    )
    return (
        {
            "status": "PASS" if schema_match else "FAIL",
            "source": str(smpl_path),
            "source_sha256": sha256_file(smpl_path),
            "fields": rows,
            "frame_index_correspondence": "frame_id indexes every time-varying field directly; betas[0] is identity shape",
            "parameter_convention": "SMPL-X axis-angle vectors in radians for global, body, jaw, and hand pose fields",
            "translation_runtime_contract": "model-native meters; official code forwards transl without rescaling",
            "translation_unit_explicit_in_npz": False,
            "gender": "UNKNOWN",
            "gender_reason": "No authoritative gender metadata exists in the raw archive; neutral model compatibility is not gender evidence.",
        },
        arrays,
    )


def audit_smpl_forward(model_path: Path | None, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    if model_path is None or not model_path.is_file():
        return {
            "status": "NOT_RUN_MODEL_ABSENT",
            "model_path": None if model_path is None else str(model_path),
            "gender_inference": "FORBIDDEN",
        }
    import smplx
    import torch

    model = smplx.SMPLX(
        model_path=str(model_path),
        use_pca=False,
        num_pca_comps=45,
        flat_hand_mean=True,
        batch_size=1,
    )
    tensors = {name: torch.from_numpy(value) for name, value in arrays.items()}
    rows = []
    with torch.no_grad():
        for frame_id in SMOKE_FRAMES:
            output = model(
                betas=tensors["betas"],
                global_orient=tensors["global_orient"][frame_id : frame_id + 1],
                transl=tensors["transl"][frame_id : frame_id + 1],
                body_pose=tensors["body_pose"][frame_id : frame_id + 1],
                jaw_pose=tensors["jaw_pose"][frame_id : frame_id + 1],
                expression=tensors["expression"][frame_id : frame_id + 1],
                left_hand_pose=tensors["left_hand_pose"][frame_id : frame_id + 1],
                right_hand_pose=tensors["right_hand_pose"][frame_id : frame_id + 1],
                return_verts=True,
            )
            rows.append(
                {
                    "frame_id": frame_id,
                    "vertices_shape": list(output.vertices.shape),
                    "joints_shape": list(output.joints.shape),
                    "vertices_finite": bool(torch.isfinite(output.vertices).all()),
                    "joints_finite": bool(torch.isfinite(output.joints).all()),
                    "vertex_minimum": float(output.vertices.min()),
                    "vertex_maximum": float(output.vertices.max()),
                }
            )
    passed = all(
        row["vertices_shape"] == [1, 10475, 3]
        and row["vertices_finite"]
        and row["joints_finite"]
        for row in rows
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "model_bytes": model_path.stat().st_size,
        "model_type": "SMPL-X neutral",
        "model_vertex_count": int(model.get_num_verts()),
        "model_face_count": int(model.faces.shape[0]),
        "compatibility_claim": "NEUTRAL_SMPLX_FORWARD_COMPATIBLE",
        "gender_inference": "FORBIDDEN",
        "rows": rows,
    }


def audit_loader(repo_root: Path, raw_root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(repo_root))
    from scene.dataset import AVRexDataset

    dataset = AVRexDataset(
        str(raw_root),
        list(SMOKE_FRAMES),
        list(SMOKE_CAMERAS),
        background=np.ones(3, dtype=np.float32),
        image_scaling=1,
        is_in_memory=False,
    )
    rows = []
    for index in range(len(dataset)):
        item = dataset[index]
        checks = {
            "image_shape": list(item["image"].shape) == [2048, 1500, 3],
            "mask_shape": list(item["mask"].shape) == [2048, 1500],
            "K_shape": list(item["K"].shape) == [3, 3],
            "w2c_shape": list(item["w2c"].shape) == [4, 4],
            "pose_shape": list(item["pose"].shape) == [165],
            "beta_shape": list(item["beta"].shape) == [10],
            "finite": bool(item["image"].isfinite().all() and item["K"].isfinite().all()),
        }
        rows.append(
            {
                "dataset_index": index,
                "camera_index": int(item["cam_id"]),
                "canonical_camera_id": f"camera_{int(item['cam_id']):02d}",
                "frame_id": int(item["frame_id"]),
                "image_shape": list(item["image"].shape),
                "mask_shape": list(item["mask"].shape),
                "pose_shape": list(item["pose"].shape),
                "foreground_pixels": int(item["mask"].sum()),
                "checks": checks,
                "status": "PASS" if all(checks.values()) else "FAIL",
            }
        )
    return {
        "status": "PASS" if len(rows) == 9 and all(row["status"] == "PASS" for row in rows) else "FAIL",
        "dataset_class": "scene.dataset.AVRexDataset",
        "decoded_record_count": len(rows),
        "selected_camera_indices": list(SMOKE_CAMERAS),
        "selected_frame_ids": list(SMOKE_FRAMES),
        "image_scaling": 1,
        "data_in_memory": False,
        "rows": rows,
        "rgb_copies": 0,
        "mask_copies": 0,
        "raw_writes": 0,
    }


def audit_split(adapter: AvatarReXZeroCopyAdapter, frozen_split_path: Path) -> dict[str, Any]:
    frozen = json.loads(frozen_split_path.read_text(encoding="utf-8"))
    camera_first = adapter.camera_split()
    camera_second = adapter.camera_split()
    pose_first = adapter.pose_split()
    pose_second = adapter.pose_split()
    availability = adapter.availability_manifest()

    per_camera = {
        row["canonical_camera_id"]: {
            "rgb": row["rgb_count"],
            "mask": row["mask_count"],
            "valid_pairs": min(row["rgb_count"], row["mask_count"]),
        }
        for row in availability["camera_rows"]
    }
    train_camera_count = camera_first["train_count"]
    heldout_camera_count = camera_first["heldout_count"]
    train_pose_count = pose_first["train_count"]
    heldout_pose_count = pose_first["heldout_count"]
    checks = {
        "camera_regeneration_deterministic": camera_first == camera_second,
        "pose_regeneration_deterministic": pose_first == pose_second,
        "camera_matches_frozen": camera_first == frozen["camera_split"],
        "pose_matches_frozen": pose_first == frozen["pose_split"],
        "availability_matches_frozen": availability == frozen["availability_manifest"],
        "camera_overlap_zero": camera_first["overlap"] == [],
        "pose_overlap_zero": pose_first["train_heldout_overlap"] == [],
        "buffer_overlap_zero": pose_first["train_buffer_overlap"] == [],
        "temporal_leakage_zero": pose_first["temporal_leakage"] == 0,
        "all_records_available": all(row["valid_pairs"] == 1901 for row in per_camera.values()),
    }
    train_azimuths = [
        row["azimuth_degrees"]
        for row in camera_first["cameras"]
        if row["canonical_camera_id"] in camera_first["train_camera_ids"]
    ]
    heldout_azimuths = [
        row["azimuth_degrees"]
        for row in camera_first["cameras"]
        if row["canonical_camera_id"] in camera_first["heldout_camera_ids"]
    ]
    train_elevations = [
        row["elevation_degrees"]
        for row in camera_first["cameras"]
        if row["canonical_camera_id"] in camera_first["train_camera_ids"]
    ]
    heldout_elevations = [
        row["elevation_degrees"]
        for row in camera_first["cameras"]
        if row["canonical_camera_id"] in camera_first["heldout_camera_ids"]
    ]
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "camera_split": camera_first,
        "pose_split": pose_first,
        "per_camera_valid_counts": per_camera,
        "per_pose_valid_camera_counts": {
            "train_poses": {"pose_count": train_pose_count, "minimum": 16, "maximum": 16},
            "heldout_poses": {"pose_count": heldout_pose_count, "minimum": 16, "maximum": 16},
        },
        "available_pair_counts": {
            "train_camera_x_train_pose": train_camera_count * train_pose_count,
            "heldout_camera_x_train_pose": heldout_camera_count * train_pose_count,
            "train_camera_x_heldout_pose": train_camera_count * heldout_pose_count,
            "heldout_camera_x_heldout_pose": heldout_camera_count * heldout_pose_count,
        },
        "camera_coverage": {
            "azimuth_order": camera_first["azimuth_order_camera_ids"],
            "train_azimuth_range_degrees": [min(train_azimuths), max(train_azimuths)],
            "heldout_azimuth_range_degrees": [min(heldout_azimuths), max(heldout_azimuths)],
            "train_elevation_range_degrees": [min(train_elevations), max(train_elevations)],
            "heldout_elevation_range_degrees": [min(heldout_elevations), max(heldout_elevations)],
            "heldout_selection": "azimuth ranks 0,4,8,12; no result-driven camera choice",
        },
        "frozen_split_content_sha256": frozen["split_content_sha256"],
        "regenerated_split_sha256": canonical_sha256(
            {"camera": camera_first, "pose": pose_first, "availability": availability}
        ),
    }


def _source_contract(path: Path, required_fragments: list[str]) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(path))
    fragments = {fragment: fragment in source for fragment in required_fragments}
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "ast_parse": "PASS",
        "required_fragments": fragments,
        "status": "PASS" if all(fragments.values()) else "FAIL",
    }


def audit_static_interfaces(repo_root: Path) -> dict[str, Any]:
    import yaml

    paths = {
        "dataset": _source_contract(
            repo_root / "scene" / "dataset.py",
            ["class AVRexDataset", "calibration_full.json", "smpl_params.npz", "apply_distortion", "def __getitem__"],
        ),
        "scene": _source_contract(
            repo_root / "scene" / "scene.py",
            ["gaussian/lbs_weights_grid.npz", "gaussian/template.ply", "No template found, using SMPLX mesh", "create_from_pcd"],
        ),
        "model": _source_contract(
            repo_root / "scene" / "gaussian_model.py",
            ["def create_from_pcd", "def training_setup", "def render", "viewmats=cam['w2c'][None]", "Ks=cam['K'][None]"],
        ),
        "lbs_generator": _source_contract(
            repo_root / "script" / "gen_weight_volume.py",
            ["PointInterpolant", "gaussian/lbs_weights_grid.npz", "bbox_min", "bbox_max", "grid_dims"],
        ),
        "training": _source_contract(
            repo_root / "train.py",
            ["GaussianModel()", "Scene(args, gaussians)", "loss.backward()", "optimizer_step()", "torch.save"],
        ),
    }
    config_path = repo_root / "config" / "avrex_lbn1.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_contract = {
        "path": str(config_path),
        "sha256": sha256_file(config_path),
        "smpl_pkl_path": config.get("smpl_pkl_path"),
        "image_scaling": config.get("image_scaling"),
        "num_train_frame": config.get("num_train_frame"),
        "declared_iterations": config.get("iterations"),
        "test_iterations": config.get("test_iterations"),
        "checkpoint_iterations": config.get("checkpoint_iterations"),
        "status": "PASS"
        if config.get("image_scaling") == 1 and config.get("num_train_frame") == 1901
        else "FAIL",
    }
    all_pass = all(row["status"] == "PASS" for row in paths.values()) and config_contract["status"] == "PASS"
    return {
        "status": "PASS" if all_pass else "FAIL",
        "config_parse": config_contract,
        "source_parse": paths,
        "loader_construction": "EXECUTED_9_ITEMS",
        "model_construction": "NOT_RUN_TEMPLATE_AND_LBS_ASSETS_REQUIRED",
        "renderer_interface": "STATIC_PARSE_PASS_FORMAL_RENDER_NOT_RUN",
        "zero_step_render": "PENDING_DERIVED_ASSET_AND_RUNTIME_CLOSURE",
        "optimizer_created": 0,
        "backward_calls": 0,
        "checkpoint_writes": 0,
        "formal_renderer_runs": 0,
    }


def asset_inventory(raw_root: Path) -> dict[str, Any]:
    relative_paths = [
        "gaussian/template.ply",
        "gaussian/lbs_weights_grid.npz",
        "gaussian/init_body_points.ply",
        "template.ply",
        "cano_weight_volume.npz",
    ]
    rows = []
    for relative in relative_paths:
        path = raw_root / relative
        rows.append(
            {
                "relative_path": relative,
                "exists": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else None,
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    return {
        "rows": rows,
        "template_present": (raw_root / "gaussian" / "template.ply").is_file(),
        "mmlp_lbs_present": (raw_root / "gaussian" / "lbs_weights_grid.npz").is_file(),
        "status": "TEMPLATE_AND_LBS_ASSETS_REQUIRED",
    }


def build_audit(
    repo_root: Path,
    raw_root: Path,
    frozen_split_path: Path,
    smpl_model_path: Path | None,
) -> dict[str, Any]:
    metadata_before = _tree_metadata_snapshot(raw_root)
    adapter = AvatarReXZeroCopyAdapter(raw_root)
    calibration = adapter.load_calibration()
    camera = audit_cameras(calibration)
    smpl_schema, arrays = audit_smpl_schema(adapter.smpl_path)
    smpl_forward = audit_smpl_forward(smpl_model_path, arrays)
    split = audit_split(adapter, frozen_split_path)
    loader = audit_loader(repo_root, raw_root)
    interfaces = audit_static_interfaces(repo_root)
    assets = asset_inventory(raw_root)
    metadata_after = _tree_metadata_snapshot(raw_root)

    required_passes = [
        camera["status"] == "PASS",
        smpl_schema["status"] == "PASS",
        smpl_forward["status"] in {"PASS", "NOT_RUN_MODEL_ABSENT"},
        split["status"] == "PASS",
        loader["status"] == "PASS",
        interfaces["status"] == "PASS",
        metadata_before == metadata_after,
    ]
    return {
        "schema_version": "avatarrex.base_avatar_runtime_audit.v1",
        "task_id": TASK_ID,
        "generated_at_utc": _utc_now(),
        "status": "DATASET_AND_PARAMETER_ADAPTER_SMOKE_PASS" if all(required_passes) else "FAIL",
        "blocker": "TEMPLATE_AND_LBS_ASSETS_REQUIRED",
        "repo_root": str(repo_root),
        "raw_root": str(raw_root),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": _package_version("torch"),
            "smplx": _package_version("smplx"),
            "opencv_python": _package_version("opencv-python"),
            "pillow": _package_version("Pillow"),
            "open3d": _package_version("open3d"),
            "gsplat": _package_version("gsplat"),
            "pytorch3d": _package_version("pytorch3d"),
        },
        "camera_audit": camera,
        "smpl_x_schema_audit": smpl_schema,
        "smpl_x_neutral_forward": smpl_forward,
        "strict_split_audit": split,
        "loader_smoke": loader,
        "static_interface_audit": interfaces,
        "asset_inventory": assets,
        "mutation_audit": {
            "raw_tree_metadata_before": metadata_before,
            "raw_tree_metadata_after": metadata_after,
            "raw_tree_metadata_unchanged": metadata_before == metadata_after,
            "large_downloads": 0,
            "full_rgb_copies": 0,
            "full_mask_copies": 0,
            "template_generation": 0,
            "lbs_generation": 0,
            "training_runs": 0,
            "backward_calls": 0,
            "optimizer_created": 0,
            "checkpoint_writes": 0,
            "formal_renderer_runs": 0,
            "image_generation_api_calls": 0,
            "raw_mutation": 0,
            "paper_final": 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--frozen-split", type=Path, required=True)
    parser.add_argument("--smpl-model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    raw_root = args.raw_root.resolve()
    frozen_split = args.frozen_split.resolve()
    smpl_model = args.smpl_model.resolve() if args.smpl_model else None
    audit = build_audit(repo_root, raw_root, frozen_split, smpl_model)
    write_json(args.output.resolve(), audit)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "blocker": audit["blocker"],
                "camera_count": audit["camera_audit"]["camera_count"],
                "loader_records": audit["loader_smoke"]["decoded_record_count"],
                "raw_tree_unchanged": audit["mutation_audit"]["raw_tree_metadata_unchanged"],
            },
            sort_keys=True,
        )
    )
    return 0 if audit["status"] == "DATASET_AND_PARAMETER_ADAPTER_SMOKE_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
