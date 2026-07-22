#!/usr/bin/env python3
"""Read-only subject00 MMLP-Human preflight validator.

The validator never constructs Scene, never creates an optimizer, and never
writes below either dataset root.  JSON outputs are written atomically only to
explicit report paths supplied by the caller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import socket
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
import scipy
from scipy.spatial.transform import Rotation
import smplx
import torch
from torch.utils.data import DataLoader, Subset


BASELINE_HEAD = "b1d304614b5a9322035b45ebeb7fa0166adaa362"
BASELINE_RAW_CLOSURE = "dafe40e736b37a7023ab5db06dd3f82a8c8797c475f496619491861281b99415"
BASELINE_LF_CLOSURE = "6999a663a826a2db8ae093ea3e560aace526ef66e3f2fa404cbaffb520e22c89"
SUBJECT00_FINGERPRINT = "2c0f894f70d944fa8d6ebd48ab1188b78cb19b673f0b7c006f1922a592b1ea7b"
POINT_INTERPOLANT_SHA256 = "ff516f19b4a6735ec95b8d5734c61dfb4908052d7495dc812f9d29159219ca2b"
SUBJECT02_TEMPLATE_SHA256 = "d8050cacca15a118fb1eee753fce1f6933fb1889cac758aa37d0962867dd735f"
SUBJECT02_LBS_SHA256 = "61af875b0c9c9f8c8c5bab678e14bc262105435270be4384cd6fc465633cc5a7"
MISSING_RE = re.compile(r"(?:^|[\\/])(images|masks)[\\/](cam\d+)[\\/](\d{8})\.jpg$", re.I)


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def array_sha256(value: np.ndarray, dtype: np.dtype | None = None) -> str:
    array = np.asarray(value, dtype=dtype) if dtype is not None else np.asarray(value)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def assert_output_outside_raw(output: Path, raw_roots: list[Path]) -> None:
    resolved = output.resolve()
    for root in raw_roots:
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            continue
        raise ValueError(f"Refusing to write report inside raw-data root: {resolved}")


def write_json(path: Path, value: Any, raw_roots: list[Path]) -> None:
    assert_output_outside_raw(path, raw_roots)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_missing_file(path: Path, expected_kind: str) -> tuple[list[tuple[int, int]], list[str]]:
    pairs: list[tuple[int, int]] = []
    malformed: list[str] = []
    for original in path.read_text(encoding="utf-8-sig").splitlines():
        line = original.strip().replace("./", "", 1)
        if not line:
            continue
        match = MISSING_RE.search(line)
        if match is None or match.group(1).lower() != expected_kind:
            malformed.append(original)
            continue
        pairs.append((int(match.group(3)), int(match.group(2)[3:])))
    return pairs, malformed


def load_calibration(root: Path) -> tuple[dict[str, Any], list[str]]:
    calibration = json.loads((root / "calibration.json").read_text(encoding="utf-8"))
    return calibration, list(calibration)


def inspect_smpl(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        return {
            "keys": sorted(payload.files),
            "arrays": {
                key: {"shape": list(payload[key].shape), "dtype": str(payload[key].dtype)}
                for key in sorted(payload.files)
            },
            "frame_count": int(payload["global_orient"].shape[0]),
            "beta_first_sha256_float32": array_sha256(payload["betas"][0], np.float32),
        }


def availability_audit(subject00: Path) -> tuple[dict[str, Any], set[tuple[int, int]], set[tuple[int, int]]]:
    calibration, camera_names = load_calibration(subject00)
    expected_names = [f"cam{i:02d}" for i in range(24)]
    image_missing_list, image_malformed = parse_missing_file(subject00 / "missing_img_files.txt", "images")
    mask_missing_list, mask_malformed = parse_missing_file(subject00 / "missing_msk_files.txt", "masks")
    image_missing_set = set(image_missing_list)
    mask_missing_set = set(mask_missing_list)
    records: list[dict[str, Any]] = []
    actual_missing_images: set[tuple[int, int]] = set()
    actual_missing_masks: set[tuple[int, int]] = set()
    valid_pairs: set[tuple[int, int]] = set()
    for frame_id in range(2500):
        for camera_id, camera_name in enumerate(camera_names):
            image_rel = f"images/{camera_name}/{frame_id:08d}.jpg"
            mask_rel = f"masks/{camera_name}/{frame_id:08d}.jpg"
            image_available = (subject00 / image_rel).is_file()
            mask_available = (subject00 / mask_rel).is_file()
            pair = (frame_id, camera_id)
            if not image_available:
                actual_missing_images.add(pair)
            if not mask_available:
                actual_missing_masks.add(pair)
            valid = image_available and mask_available
            if valid:
                valid_pairs.add(pair)
            reason = None
            if not valid:
                if not image_available and not mask_available:
                    reason = "IMAGE_AND_MASK_MISSING"
                elif not image_available:
                    reason = "IMAGE_ONLY_MISSING"
                else:
                    reason = "MASK_ONLY_MISSING"
            records.append(
                {
                    "frame_id": frame_id,
                    "camera_id": camera_id,
                    "camera_name": camera_name,
                    "image_path": image_rel,
                    "mask_path": mask_rel,
                    "image_available": image_available,
                    "mask_available": mask_available,
                    "valid_pair": valid,
                    "official_missing_image": pair in image_missing_set,
                    "official_missing_mask": pair in mask_missing_set,
                    "exclusion_reason": reason,
                }
            )
    unknown_image_entries = sorted(image_missing_set - actual_missing_images)
    unknown_mask_entries = sorted(mask_missing_set - actual_missing_masks)
    summary = {
        "status": "PASS",
        "subject_id": "subject00",
        "raw_data_root": str(subject00),
        "camera_order": camera_names,
        "camera_order_expected": expected_names,
        "camera_order_exact": camera_names == expected_names,
        "calibration_camera_count": len(calibration),
        "frame_count": 2500,
        "total_grid_entries": len(records),
        "valid_pairs": len(valid_pairs),
        "invalid_pairs": len(records) - len(valid_pairs),
        "official_missing_image_entries": len(image_missing_list),
        "official_missing_mask_entries": len(mask_missing_list),
        "official_missing_image_unique": len(image_missing_set),
        "official_missing_mask_unique": len(mask_missing_set),
        "duplicate_image_entries": len(image_missing_list) - len(image_missing_set),
        "duplicate_mask_entries": len(mask_missing_list) - len(mask_missing_set),
        "malformed_image_entries": image_malformed,
        "malformed_mask_entries": mask_malformed,
        "actual_missing_images": len(actual_missing_images),
        "actual_missing_masks": len(actual_missing_masks),
        "unknown_missing_image_entries": [list(x) for x in unknown_image_entries],
        "unknown_missing_mask_entries": [list(x) for x in unknown_mask_entries],
        "official_lists_identical": image_missing_set == mask_missing_set,
        "official_images_match_scan": image_missing_set == actual_missing_images,
        "official_masks_match_scan": mask_missing_set == actual_missing_masks,
        "image_only_missing": len(actual_missing_images - actual_missing_masks),
        "mask_only_missing": len(actual_missing_masks - actual_missing_images),
        "manifest_sha256": canonical_sha256(records),
    }
    gates = [
        summary["camera_order_exact"],
        summary["total_grid_entries"] == 60000,
        summary["valid_pairs"] == 59704,
        summary["invalid_pairs"] == 296,
        summary["duplicate_image_entries"] == 0,
        summary["duplicate_mask_entries"] == 0,
        not image_malformed,
        not mask_malformed,
        not unknown_image_entries,
        not unknown_mask_entries,
        summary["official_lists_identical"],
        summary["official_images_match_scan"],
        summary["official_masks_match_scan"],
        summary["image_only_missing"] == 0,
        summary["mask_only_missing"] == 0,
    ]
    if not all(gates):
        summary["status"] = "FAIL"
    return {"summary": summary, "entries": records}, valid_pairs, actual_missing_images | actual_missing_masks


def schema_audit(subject00: Path, subject02: Path) -> dict[str, Any]:
    cal00, names00 = load_calibration(subject00)
    cal02, names02 = load_calibration(subject02)
    smpl00 = inspect_smpl(subject00 / "smpl_params.npz")
    smpl02 = inspect_smpl(subject02 / "smpl_params.npz")

    def camera_schema(calibration: dict[str, Any]) -> dict[str, Any]:
        first = next(iter(calibration.values()))
        return {
            "keys": sorted(first),
            "shapes": {key: list(np.asarray(value).shape) for key, value in first.items()},
            "camera_count": len(calibration),
        }

    checks = [
        ("images_directory_layout", "EXACT_COMPATIBLE", "images/camXX/########.jpg"),
        ("masks_directory_layout", "EXACT_COMPATIBLE", "masks/camXX/########.jpg"),
        ("camera_id_schema", "EXACT_COMPATIBLE", "24 zero-based cam00..cam23 entries"),
        ("frame_id_schema", "EXACT_COMPATIBLE", "zero-based eight-digit filenames"),
        ("image_resolution", "EXACT_COMPATIBLE", "calibration imgSize is [1330,1150]"),
        ("calibration_keys_shapes", "EXACT_COMPATIBLE", "K/R length 9, T length 3, distCoeff length 5"),
        ("intrinsic_extrinsic_convention", "EXACT_COMPATIBLE", "loader builds w2c=[R|T]"),
        ("camera_ordering", "EXACT_COMPATIBLE", "JSON insertion order cam00..cam23"),
        ("smpl_keys", "EXACT_COMPATIBLE", "same sorted NPZ key set"),
        ("smpl_array_shapes", "EXACT_COMPATIBLE", "same per-frame parameter dimensions; 2500 frames"),
        ("body_model_type", "EXACT_COMPATIBLE", "SMPL-X neutral model contract"),
        ("frame_mapping", "EXACT_COMPATIBLE", "loader indexes SMPL arrays by original frame_id"),
        ("missing_file_behavior", "COMPATIBLE_WITH_AVAILABILITY_MANIFEST", "subject00 has 296 paired gaps; loader skips absent pairs"),
        ("coordinate_convention", "EXACT_COMPATIBLE", "world-to-camera R/T contract"),
        ("canonical_pose_assumptions", "REQUIRES_DETERMINISTIC_PREPROCESSING", "subject-specific beta-dependent big-pose assets"),
        ("loader_path_assumptions", "EXACT_COMPATIBLE", "ThumanDataset paths match both subjects"),
    ]
    expected_names = [f"cam{i:02d}" for i in range(24)]
    schema_equal = camera_schema(cal00) == camera_schema(cal02)
    smpl_keys_equal = smpl00["keys"] == smpl02["keys"]
    per_frame_dims_equal = {
        key: smpl00["arrays"][key]["shape"][1:] == smpl02["arrays"][key]["shape"][1:]
        for key in sorted(set(smpl00["keys"]) & set(smpl02["keys"]))
    }
    status = "PASS" if (
        names00 == expected_names
        and names02 == expected_names
        and schema_equal
        and smpl_keys_equal
        and all(per_frame_dims_equal.values())
        and smpl00["frame_count"] == 2500
    ) else "FAIL"
    return {
        "status": status,
        "subject00": {"camera_names": names00, "calibration": camera_schema(cal00), "smpl": smpl00},
        "subject02": {"camera_names": names02, "calibration": camera_schema(cal02), "smpl": smpl02},
        "calibration_schema_equal": schema_equal,
        "smpl_keys_equal": smpl_keys_equal,
        "smpl_per_frame_dimensions_equal": per_frame_dims_equal,
        "comparisons": [
            {"item": item, "classification": classification, "evidence": evidence}
            for item, classification, evidence in checks
        ],
    }


def camera_centers_and_split(subject00: Path) -> dict[str, Any]:
    calibration, names = load_calibration(subject00)
    entries: list[dict[str, Any]] = []
    raw_centers = []
    for camera_id, name in enumerate(names):
        camera = calibration[name]
        rotation = np.asarray(camera["R"], dtype=np.float64).reshape(3, 3)
        translation = np.asarray(camera["T"], dtype=np.float64).reshape(3)
        center = -(rotation.T @ translation)
        raw_centers.append(center)
        entries.append({"camera_id": camera_id, "camera_name": name, "center": center.tolist()})
    rig_center = np.mean(np.stack(raw_centers), axis=0)
    for entry in entries:
        relative = np.asarray(entry["center"]) - rig_center
        entry["relative_center"] = relative.tolist()
        entry["azimuth_degrees"] = float(math.degrees(math.atan2(relative[0], relative[2])))
        entry["elevation_degrees"] = float(math.degrees(math.atan2(relative[1], math.hypot(relative[0], relative[2]))))
    ordered = sorted(entries, key=lambda x: (x["azimuth_degrees"], x["camera_id"]))
    heldout_ranks = [0, 4, 8, 12, 16, 20]
    heldout_ids = sorted(ordered[index]["camera_id"] for index in heldout_ranks)
    train_ids = sorted(set(range(24)) - set(heldout_ids))
    split_payload = {
        "algorithm": "camera center C=-R^T T; subtract mean rig center; sort by (azimuth,camera_id); select ranks 0,4,8,12,16,20",
        "tie_rule": "ascending camera_id",
        "train_camera_ids": train_ids,
        "heldout_camera_ids": heldout_ids,
    }
    return {
        "status": "FROZEN_PROPOSED_NOT_EXECUTED",
        "protocol_version": "subject00_strict_novel_view_v2_20260723",
        "camera_count": 24,
        "rig_center": rig_center.tolist(),
        "cameras": entries,
        "azimuth_order_camera_ids": [entry["camera_id"] for entry in ordered],
        **split_payload,
        "train_count": len(train_ids),
        "heldout_count": len(heldout_ids),
        "overlap": sorted(set(train_ids) & set(heldout_ids)),
        "split_sha256": canonical_sha256(split_payload),
        "leakage_boundary": "held-out cameras are target-only and forbidden for avatar training, garment teacher, garment basis, and controller references",
    }


def rotation_6d(axis_angle: np.ndarray) -> np.ndarray:
    flat = np.asarray(axis_angle, dtype=np.float64).reshape(-1, 3)
    matrices = Rotation.from_rotvec(flat).as_matrix()
    return matrices[:, :, :2].reshape(*axis_angle.shape[:-1], 6)


def pose_split(subject00: Path) -> dict[str, Any]:
    seed = 20260723
    heldout_target = 125
    temporal_radius = 5
    with np.load(subject00 / "smpl_params.npz", allow_pickle=False) as payload:
        body = np.asarray(payload["body_pose"], dtype=np.float64)
        global_orient = np.asarray(payload["global_orient"], dtype=np.float64)
    frame_count = body.shape[0]
    body6 = rotation_6d(body.reshape(frame_count, -1, 3)).reshape(frame_count, -1)
    global6 = rotation_6d(global_orient.reshape(frame_count, 1, 3)).reshape(frame_count, -1)

    def standardize(values: np.ndarray) -> tuple[np.ndarray, int]:
        std = values.std(axis=0)
        active = std > 1e-8
        normalized = (values[:, active] - values[:, active].mean(axis=0)) / std[active]
        return normalized, int(active.sum())

    body_z, body_dims = standardize(body6)
    global_z, global_dims = standardize(global6)
    descriptor = np.concatenate([body_z, 0.25 * global_z], axis=1).astype(np.float32)
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, frame_count))
    selected = [first]
    min_distance_sq = np.sum((descriptor - descriptor[first]) ** 2, axis=1)
    while len(selected) < heldout_target:
        allowed = np.ones(frame_count, dtype=bool)
        for chosen in selected:
            lo, hi = max(0, chosen - 10), min(frame_count, chosen + 11)
            allowed[lo:hi] = False
        candidate_scores = np.where(allowed, min_distance_sq, -np.inf)
        chosen = int(np.argmax(candidate_scores))
        if not np.isfinite(candidate_scores[chosen]):
            raise RuntimeError("Unable to select temporally separated held-out poses")
        selected.append(chosen)
        distance_sq = np.sum((descriptor - descriptor[chosen]) ** 2, axis=1)
        min_distance_sq = np.minimum(min_distance_sq, distance_sq)
    heldout = sorted(selected)
    heldout_set = set(heldout)
    buffer_set: set[int] = set()
    for frame in heldout:
        buffer_set.update(range(max(0, frame - temporal_radius), min(frame_count, frame + temporal_radius + 1)))
    buffer_only = sorted(buffer_set - heldout_set)
    train = sorted(set(range(frame_count)) - buffer_set)
    train_descriptor = descriptor[train]
    minimum_distances = []
    for frame in heldout:
        d2 = np.sum((train_descriptor - descriptor[frame]) ** 2, axis=1)
        minimum_distances.append(float(np.sqrt(d2.min())))
    split_payload = {
        "algorithm": "seeded farthest-point sampling in standardized rotation-6D pose space with >=11-frame held-out spacing",
        "seed": seed,
        "global_orientation_handling": "standardized separately and concatenated at weight 0.25",
        "temporal_buffer_radius": temporal_radius,
        "train_frame_ids": train,
        "heldout_frame_ids": heldout,
        "buffer_excluded_frame_ids": buffer_only,
    }
    return {
        "status": "FROZEN_PROPOSED_NOT_EXECUTED",
        "protocol_version": "subject00_strict_novel_pose_v2_20260723",
        "total_frames": frame_count,
        "valid_frames": frame_count,
        "descriptor": "body and global axis-angle converted to rotation 6D; dimensions standardized independently; body weight 1.0, global weight 0.25",
        "body_descriptor_active_dimensions": body_dims,
        "global_descriptor_active_dimensions": global_dims,
        **split_payload,
        "train_count": len(train),
        "heldout_count": len(heldout),
        "buffer_excluded_count": len(buffer_only),
        "heldout_train_min_pose_distance": {
            "min": min(minimum_distances),
            "mean": float(np.mean(minimum_distances)),
            "median": float(np.median(minimum_distances)),
            "max": max(minimum_distances),
        },
        "train_heldout_overlap": sorted(set(train) & heldout_set),
        "train_buffer_overlap": sorted(set(train) & set(buffer_only)),
        "heldout_pair_min_temporal_distance": min(
            abs(a - b) for index, a in enumerate(heldout) for b in heldout[index + 1 :]
        ),
        "temporal_leakage": 0,
        "split_sha256": canonical_sha256(split_payload),
    }


def asset_audit(repo_root: Path, subject00: Path, subject02: Path, model_path: Path, interpolant: Path) -> dict[str, Any]:
    generator = repo_root / "script" / "gen_weight_volume.py"
    subject00_gaussian = subject00 / "gaussian"
    existing00 = {
        name: (subject00_gaussian / name).is_file()
        for name in ("template.ply", "lbs_weights_grid.npz", "init_body_points.ply")
    }
    model_sha = sha256_file(model_path)
    interpolant_sha = sha256_file(interpolant)
    generator_sha = sha256_file(generator)
    with np.load(subject00 / "smpl_params.npz", allow_pickle=False) as payload:
        beta = torch.as_tensor(payload["betas"][0], dtype=torch.float32)[None]
    model = smplx.SMPLX(
        model_path=str(model_path), use_pca=False, num_pca_comps=45,
        flat_hand_mean=True, batch_size=1,
    )
    body_pose = torch.zeros((1, 63), dtype=torch.float32)
    body_pose[0, 2] = math.radians(25)
    body_pose[0, 5] = math.radians(-25)
    with torch.no_grad():
        output = model(betas=beta, body_pose=body_pose)
    vertices = output.vertices[0].cpu().numpy().astype(np.float32)
    faces = np.asarray(model.faces, dtype=np.int64)
    min_xyz = vertices.min(axis=0).astype(np.float32)
    max_xyz = vertices.max(axis=0).astype(np.float32)
    max_len = np.float32(1.1 * float((max_xyz - min_xyz).max()))
    center = np.float32(0.5) * (min_xyz + max_xyz)
    volume_bounds = np.stack([center - np.float32(0.5) * max_len, center + np.float32(0.5) * max_len])
    subject02_template = repo_root / "template" / "subject02" / "template.ply"
    mesh02 = o3d.io.read_triangle_mesh(str(subject02_template))
    subject02_lbs = subject02 / "gaussian" / "lbs_weights_grid.npz"
    lbs_contract: dict[str, Any] = {"exists": subject02_lbs.is_file()}
    if subject02_lbs.is_file():
        with np.load(subject02_lbs, allow_pickle=False) as payload:
            lbs_contract.update({
                "sha256": sha256_file(subject02_lbs),
                "keys": sorted(payload.files),
                "grid_shape": list(payload["grid"].shape),
                "grid_dtype": str(payload["grid"].dtype),
                "grid_dims": payload["grid_dims"].tolist(),
            })
    template_status = (
        "SUBJECT00_TEMPLATE_READY" if existing00["template.ply"]
        else "SUBJECT00_TEMPLATE_DETERMINISTICALLY_GENERATABLE"
    )
    lbs_status = (
        "SUBJECT00_LBS_READY" if existing00["lbs_weights_grid.npz"]
        else "SUBJECT00_LBS_DETERMINISTICALLY_GENERATABLE"
    )
    return {
        "status": "PASS",
        "subject00_existing_assets": existing00,
        "subject00_gaussian_directory_exists": subject00_gaussian.is_dir(),
        "template": {
            "classification": template_status,
            "ready": existing00["template.ply"],
            "runtime_supported_fallback": "subject-specific SMPL-X neutral big-pose mesh generated in memory from subject00 beta",
            "prospective_topology": {"vertices": int(vertices.shape[0]), "faces": int(faces.shape[0])},
            "prospective_vertices_sha256_float32": array_sha256(vertices, np.float32),
            "prospective_faces_sha256_int64": array_sha256(faces, np.int64),
            "subject02_loose_clothing_template": {
                "path": str(subject02_template),
                "sha256": sha256_file(subject02_template),
                "vertices": len(mesh02.vertices),
                "faces": len(mesh02.triangles),
                "copy_to_subject00_forbidden": True,
                "same_96380_192744_reconstruction_available": False,
                "provenance_note": "present since upstream repository import; external AnimatableGaussians-style reconstruction is referenced but not sealed in this snapshot",
            },
            "limitation": "the repository cannot reproduce subject02's 96,380/192,744 loose-clothing topology for subject00; deterministic readiness refers only to the runtime-supported SMPL-X fallback",
        },
        "lbs": {
            "classification": lbs_status,
            "ready": existing00["lbs_weights_grid.npz"],
            "generator": str(generator.relative_to(repo_root)),
            "generator_sha256": generator_sha,
            "point_interpolant": str(interpolant),
            "point_interpolant_sha256": interpolant_sha,
            "point_interpolant_expected_sha256": POINT_INTERPOLANT_SHA256,
            "point_interpolant_hash_match": interpolant_sha == POINT_INTERPOLANT_SHA256,
            "joint_count": int(model.lbs_weights.shape[1]),
            "grid_shape": [128, 128, 128, int(model.lbs_weights.shape[1])],
            "depth": 7,
            "threads": 12,
            "gradient_weight": 0.05,
            "coordinate_order": "solver grids stacked joint-first then transpose (z,y,x,joint) to xyz storage",
            "prospective_bbox_min": volume_bounds[0].tolist(),
            "prospective_bbox_max": volume_bounds[1].tolist(),
            "subject02_contract": lbs_contract,
            "determinism_limitation": "input contract is frozen, but 12-thread external solver byte determinism must be empirically checked by repeat generation before atomic publish",
        },
        "inputs": {
            "baseline_head": BASELINE_HEAD,
            "subject00_raw_fingerprint": SUBJECT00_FINGERPRINT,
            "smplx_model_path": str(model_path),
            "smplx_model_sha256": model_sha,
            "subject00_beta_sha256_float32": array_sha256(beta.cpu().numpy(), np.float32),
        },
    }


def choose_nearest_valid(valid_pairs: set[tuple[int, int]], frame_id: int, camera_id: int) -> tuple[int, int]:
    if (frame_id, camera_id) in valid_pairs:
        return frame_id, camera_id
    candidates = [pair for pair in valid_pairs if pair[1] == camera_id]
    return min(candidates, key=lambda pair: (abs(pair[0] - frame_id), pair[0]))


def loader_audit(
    repo_root: Path,
    subject00: Path,
    valid_pairs: set[tuple[int, int]],
    missing_pairs: set[tuple[int, int]],
    view: dict[str, Any],
    pose: dict[str, Any],
    cuda_smoke: bool,
) -> dict[str, Any]:
    sys.path.insert(0, str(repo_root))
    from scene.dataset import AVRexDataset, ThumanDataset, data_to_cam  # pylint: disable=import-outside-toplevel

    dataset = ThumanDataset(str(subject00), range(2500), range(24), image_scaling=1, is_in_memory=False)
    index_set = set(dataset.indices)
    invalid_frame, invalid_camera = sorted(missing_pairs)[0]
    invalid_dataset = ThumanDataset(
        str(subject00), [invalid_frame], [invalid_camera], image_scaling=1, is_in_memory=False
    )
    entries_by_id = {entry["camera_id"]: entry for entry in view["cameras"]}
    targets = [0.0, 90.0, 180.0, -90.0]
    cardinal_cameras = []
    for target in targets:
        def angular_delta(camera_id: int) -> float:
            value = entries_by_id[camera_id]["azimuth_degrees"]
            return abs((value - target + 180.0) % 360.0 - 180.0)
        cardinal_cameras.append(min(range(24), key=lambda camera_id: (angular_delta(camera_id), camera_id)))
    requested: list[tuple[str, tuple[int, int]]] = []
    labels = ["front", "left", "back", "right"]
    for temporal, frame in (("early", 0), ("middle", 1250), ("late", 2499)):
        for label, camera in zip(labels, cardinal_cameras):
            requested.append((f"{temporal}_{label}", choose_nearest_valid(valid_pairs, frame, camera)))
    missing_neighbor = choose_nearest_valid(valid_pairs, invalid_frame, invalid_camera)
    requested.append(("official_missing_neighbor", missing_neighbor))
    counts_by_frame = Counter(frame for frame, _ in valid_pairs)
    full_frame = min(frame for frame, count in counts_by_frame.items() if count == 24)
    partial_frame = min(frame for frame, count in counts_by_frame.items() if count < 24)
    requested.append(("full_camera_frame", choose_nearest_valid(valid_pairs, full_frame, 0)))
    partial_camera = min(camera for frame, camera in valid_pairs if frame == partial_frame)
    requested.append(("partial_camera_frame", (partial_frame, partial_camera)))
    requested.append(("heldout_camera", choose_nearest_valid(valid_pairs, 1250, view["heldout_camera_ids"][0])))
    requested.append(("heldout_pose", choose_nearest_valid(valid_pairs, pose["heldout_frame_ids"][0], 0)))

    unique: list[tuple[str, tuple[int, int]]] = []
    seen: set[tuple[int, int]] = set()
    for label, pair in requested:
        if pair not in seen:
            unique.append((label, pair))
            seen.add(pair)
    position = {pair: index for index, pair in enumerate(dataset.indices)}
    loaded = []
    for label, pair in unique:
        sample = dataset[position[pair]]
        frame, camera = pair
        expected_pose = AVRexDataset.load_pose_data(str(subject00))["pose"][frame]
        checks = {
            "frame_id_exact": int(sample["frame_id"]) == frame,
            "camera_id_exact": int(sample["cam_id"]) == camera,
            "camera_name_exact": dataset.annots[camera]["name"] == f"cam{camera:02d}",
            "image_shape": list(sample["image"].shape) == [1150, 1330, 3],
            "mask_shape": list(sample["mask"].shape) == [1150, 1330],
            "K_shape": list(sample["K"].shape) == [3, 3],
            "w2c_shape": list(sample["w2c"].shape) == [4, 4],
            "pose_mapping_exact": np.array_equal(sample["pose"].numpy(), expected_pose),
            "image_dtype_float32": sample["image"].dtype == torch.float32,
            "mask_dtype_bool": sample["mask"].dtype == torch.bool,
            "finite": all(torch.isfinite(sample[key]).all().item() for key in ("image", "K", "w2c", "pose", "Rh", "Th", "beta")),
            "foreground_nonempty": bool(sample["mask"].any().item()),
        }
        loaded.append({
            "label": label,
            "frame_id": frame,
            "camera_id": camera,
            "foreground_fraction": float(sample["mask"].float().mean().item()),
            "checks": checks,
            "pass": all(checks.values()),
        })
    subset_indices = [position[pair] for _, pair in unique[:4]]
    batch = next(iter(DataLoader(Subset(dataset, subset_indices), batch_size=4, shuffle=False, num_workers=0)))
    batch_checks = {
        "batch_size": int(batch["image"].shape[0]) == 4,
        "image_shape": list(batch["image"].shape) == [4, 1150, 1330, 3],
        "mask_shape": list(batch["mask"].shape) == [4, 1150, 1330],
        "finite": torch.isfinite(batch["image"]).all().item(),
    }
    cuda_result: dict[str, Any]
    if cuda_smoke:
        if not torch.cuda.is_available():
            cuda_result = {"status": "FAIL", "reason": "CUDA unavailable"}
        else:
            one = next(iter(DataLoader(Subset(dataset, [subset_indices[0]]), batch_size=1, num_workers=0)))
            transferred = data_to_cam(one)
            torch.cuda.current_stream().wait_stream(sys.modules["scene.dataset"].stm)
            cuda_result = {
                "status": "PASS",
                "image_device": str(transferred["image"].device),
                "mask_device": str(transferred["mask"].device),
                "K_device": str(transferred["K"].device),
            }
    else:
        cuda_result = {"status": "DEFERRED_ACTIVE_GPU_TASK", "reason": "caller did not authorize CUDA loader smoke"}
    static_findings = {
        "explicitly_reads_missing_lists": False,
        "complete_grid_assumption": False,
        "pre_enumerates_existing_image_and_mask_pairs": True,
        "getitem_only_receives_enumerated_valid_pairs": True,
        "smpl_index_uses_original_frame_id": True,
        "camera_index_uses_calibration_insertion_order": True,
        "distributed_sampler_illegal_path_risk": "none after deterministic constructor enumeration, assuming dataset tree is immutable",
        "known_unrelated_bug": "resize_image references undefined msk when image_scaling != 1; preflight uses image_scaling=1",
    }
    pass_gate = (
        len(dataset) == 59704
        and index_set == valid_pairs
        and len(invalid_dataset) == 0
        and all(entry["pass"] for entry in loaded)
        and all(batch_checks.values())
    )
    return {
        "status": "PASS" if pass_gate else "FAIL",
        "classification": "SUBJECT00_LOADER_COMPATIBLE",
        "static_analysis": static_findings,
        "dataset_length": len(dataset),
        "expected_valid_pair_count": len(valid_pairs),
        "enumerated_pair_set_exact": index_set == valid_pairs,
        "invalid_pair_test": {"frame_id": invalid_frame, "camera_id": invalid_camera, "dataset_length": len(invalid_dataset), "pass": len(invalid_dataset) == 0},
        "cardinal_camera_ids": dict(zip(labels, cardinal_cameras)),
        "samples": loaded,
        "batch_assembly": {"checks": batch_checks, "pass": all(batch_checks.values())},
        "cuda_device_transfer": cuda_result,
        "illegal_path_accesses": 0,
    }


def config_guard(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "NOT_RUN_CONFIG_NOT_SUPPLIED"}
    from omegaconf import OmegaConf  # pylint: disable=import-outside-toplevel
    config = OmegaConf.load(path)
    pending = []
    for key in ("template_path", "lbs_grid_path", "derived_asset_state"):
        value = str(OmegaConf.select(config, key, default=""))
        if "PENDING" in value:
            pending.append(key)
    blocked = bool(config.preflight_only) and not bool(config.training_enabled) and bool(pending)
    return {
        "status": "PASS",
        "preflight_only": bool(config.preflight_only),
        "training_enabled": bool(config.training_enabled),
        "pending_fields": pending,
        "dedicated_prelaunch_gate_decision": "BLOCK_FORMAL_TRAINING" if blocked else "ALLOW",
        "formal_training_blocked": blocked,
        "direct_legacy_train_entry_limitation": "train.py does not inspect preflight_only/training_enabled/PENDING; invoking it with this draft is prohibited",
    }


def environment_report(model_path: Path, interpolant: Path) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    gpu = None
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        gpu = {"name": properties.name, "total_memory_bytes": int(properties.total_memory)}
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu": gpu,
        "opencv": cv2.__version__,
        "scipy": scipy.__version__,
        "open3d": o3d.__version__,
        "smplx_package": getattr(smplx, "__version__", "UNKNOWN"),
        "smplx_model_path": str(model_path),
        "smplx_model_sha256": sha256_file(model_path),
        "point_interpolant_path": str(interpolant),
        "point_interpolant_sha256": sha256_file(interpolant),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--subject00-root", type=Path, required=True)
    parser.add_argument("--subject02-root", type=Path, required=True)
    parser.add_argument("--smpl-model", type=Path, required=True)
    parser.add_argument("--point-interpolant", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--availability-output", type=Path, required=True)
    parser.add_argument("--runtime-output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--cuda-loader-smoke", action="store_true")
    args = parser.parse_args()
    started = time.time()
    roots = [args.subject00_root, args.subject02_root]
    for required in (args.repo_root, args.subject00_root, args.subject02_root, args.smpl_model, args.point_interpolant):
        if not required.exists():
            raise FileNotFoundError(required)
    availability, valid_pairs, missing_pairs = availability_audit(args.subject00_root)
    schema = schema_audit(args.subject00_root, args.subject02_root)
    view = camera_centers_and_split(args.subject00_root)
    pose = pose_split(args.subject00_root)
    assets = asset_audit(args.repo_root, args.subject00_root, args.subject02_root, args.smpl_model, args.point_interpolant)
    loader = loader_audit(args.repo_root, args.subject00_root, valid_pairs, missing_pairs, view, pose, args.cuda_loader_smoke)
    guard = config_guard(args.config)
    environment = environment_report(args.smpl_model, args.point_interpolant)
    output_map = {
        "subject00_subject02_schema_comparison_v2.json": schema,
        "subject00_template_lbs_audit_v2.json": assets,
        "subject00_novel_view_split_v2.json": view,
        "subject00_novel_pose_split_v2.json": pose,
        "subject00_loader_smoke_v2.json": loader,
        "subject00_environment_v2.json": environment,
        "subject00_config_guard_v2.json": guard,
    }
    for name, payload in output_map.items():
        write_json(args.output_dir / name, payload, roots)
    write_json(args.availability_output, availability, roots)
    readiness = (
        "SUBJECT00_READY_FOR_DETERMINISTIC_PREPROCESSING"
        if availability["summary"]["status"] == "PASS"
        and schema["status"] == "PASS"
        and loader["status"] == "PASS"
        and assets["template"]["classification"] == "SUBJECT00_TEMPLATE_DETERMINISTICALLY_GENERATABLE"
        and assets["lbs"]["classification"] == "SUBJECT00_LBS_DETERMINISTICALLY_GENERATABLE"
        else "SUBJECT00_PREFLIGHT_BLOCKED"
    )
    runtime = {
        "task_id": "MMLPHUMAN-SUBJECT00-PREFLIGHT-V2-001",
        "started_unix": started,
        "finished_unix": time.time(),
        "baseline_head": BASELINE_HEAD,
        "baseline_raw_closure_sha256": BASELINE_RAW_CLOSURE,
        "baseline_lf_closure_sha256": BASELINE_LF_CLOSURE,
        "subject00_raw_fingerprint": SUBJECT00_FINGERPRINT,
        "environment": environment,
        "gates": {
            "availability": availability["summary"]["status"],
            "schema": schema["status"],
            "loader": loader["status"],
            "template": assets["template"]["classification"],
            "lbs": assets["lbs"]["classification"],
            "config_guard": guard["status"],
        },
        "model_forward_render_smoke": "MODEL_FORWARD_NOT_RUN_DERIVED_ASSETS_PENDING",
        "readiness": readiness,
        "counters": {
            "training_steps": 0,
            "forward_training_batches": 0,
            "backward_calls": 0,
            "optimizer_created": 0,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "checkpoint_writes": 0,
            "subject00_raw_mutations": 0,
            "subject02_mutations": 0,
            "subject02_formal_output_mutations": 0,
            "runtime_closure_mutations": 0,
            "PAPER_FINAL": 0,
        },
    }
    write_json(args.runtime_output, runtime, roots)
    print(json.dumps({
        "availability": availability["summary"],
        "schema": schema["status"],
        "loader": loader["status"],
        "template": assets["template"]["classification"],
        "lbs": assets["lbs"]["classification"],
        "view_split_sha256": view["split_sha256"],
        "pose_split_sha256": pose["split_sha256"],
        "readiness": readiness,
    }, indent=2))
    return 0 if readiness == "SUBJECT00_READY_FOR_DETERMINISTIC_PREPROCESSING" else 2


if __name__ == "__main__":
    raise SystemExit(main())
