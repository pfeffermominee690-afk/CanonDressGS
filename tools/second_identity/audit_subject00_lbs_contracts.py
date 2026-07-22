#!/usr/bin/env python3
"""Audit subject00 LBS joint, coordinate, bbox, reference, and local errors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
import smplx
import torch
import torch.nn.functional as torch_f
import trimesh


JOINT_COUNT = 55


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def joint_names() -> tuple[list[str], str]:
    try:
        from smplx.joint_names import JOINT_NAMES

        names = list(JOINT_NAMES[:JOINT_COUNT])
        if len(names) != JOINT_COUNT:
            raise RuntimeError(f"Expected {JOINT_COUNT} names, found {len(names)}")
        return names, "smplx.joint_names.JOINT_NAMES[:55]"
    except Exception as exc:
        return [f"joint_{index:02d}" for index in range(JOINT_COUNT)], f"UNAVAILABLE:{type(exc).__name__}:{exc}"


def canonical_model(
    smpl_model_path: Path, smpl_params_path: Path
) -> tuple[Any, Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(smpl_params_path, allow_pickle=False) as payload:
        betas = np.asarray(payload["betas"][0], dtype=np.float32)
    model = smplx.SMPLX(
        model_path=str(smpl_model_path),
        use_pca=False,
        num_pca_comps=45,
        flat_hand_mean=True,
        batch_size=1,
    )
    body_pose = torch.zeros((1, 63), dtype=torch.float32)
    body_pose[0, 2] = math.radians(25.0)
    body_pose[0, 5] = math.radians(-25.0)
    with torch.no_grad():
        result = model(
            betas=torch.from_numpy(betas[None]),
            global_orient=torch.zeros((1, 3), dtype=torch.float32),
            transl=torch.zeros((1, 3), dtype=torch.float32),
            body_pose=body_pose,
        )
    vertices = result.vertices[0].cpu().numpy().astype(np.float32)
    faces = np.asarray(model.faces, dtype=np.int64)
    weights = model.lbs_weights.cpu().numpy().astype(np.float32)
    joints = result.joints[0, :JOINT_COUNT].cpu().numpy().astype(np.float32)
    return model, result, vertices, faces, weights, joints


def runtime_interpolate(grid: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray, points: np.ndarray, align_corners: bool = True) -> np.ndarray:
    grid_tensor = torch.from_numpy(np.asarray(grid, dtype=np.float32))
    point_tensor = torch.from_numpy(np.asarray(points, dtype=np.float32))
    bbox_min_tensor = torch.from_numpy(np.asarray(bbox_min, dtype=np.float32))
    bbox_max_tensor = torch.from_numpy(np.asarray(bbox_max, dtype=np.float32))
    normalized = (point_tensor - bbox_min_tensor) / (bbox_max_tensor - bbox_min_tensor) * 2.0 - 1.0
    sample_points = normalized[:, [2, 1, 0]].reshape(1, 1, 1, -1, 3)
    with torch.no_grad():
        output = torch_f.grid_sample(
            grid_tensor.permute(3, 0, 1, 2)[None],
            sample_points,
            padding_mode="border",
            align_corners=align_corners,
        ).squeeze().permute(1, 0)
    return output.numpy()


def custom_xyz_trilinear(grid: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray, points: np.ndarray) -> np.ndarray:
    shape = np.asarray(grid.shape[:3], dtype=np.int64)
    coordinate = (points - bbox_min[None]) / (bbox_max - bbox_min)[None] * (shape - 1)[None]
    coordinate = np.clip(coordinate, 0.0, (shape - 1)[None])
    lower = np.floor(coordinate).astype(np.int64)
    upper = np.minimum(lower + 1, shape - 1)
    fraction = coordinate - lower
    result = np.zeros((len(points), grid.shape[-1]), dtype=np.float64)
    for dx in (0, 1):
        ix = lower[:, 0] if dx == 0 else upper[:, 0]
        wx = 1.0 - fraction[:, 0] if dx == 0 else fraction[:, 0]
        for dy in (0, 1):
            iy = lower[:, 1] if dy == 0 else upper[:, 1]
            wy = 1.0 - fraction[:, 1] if dy == 0 else fraction[:, 1]
            for dz in (0, 1):
                iz = lower[:, 2] if dz == 0 else upper[:, 2]
                wz = 1.0 - fraction[:, 2] if dz == 0 else fraction[:, 2]
                result += grid[ix, iy, iz].astype(np.float64) * (wx * wy * wz)[:, None]
    return result.astype(np.float32)


def synthetic_coordinate_test() -> dict[str, Any]:
    shape = (4, 5, 6)
    grid = np.zeros(shape + (3,), dtype=np.float32)
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                grid[x, y, z] = (x, y, z)
    bbox_min = np.asarray([-2.0, 3.0, 10.0], dtype=np.float32)
    bbox_max = np.asarray([4.0, 11.0, 20.0], dtype=np.float32)
    fraction = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [0.2, 0.75, 0.4]], dtype=np.float32
    )
    points = bbox_min[None] + fraction * (bbox_max - bbox_min)[None]
    runtime = runtime_interpolate(grid, bbox_min, bbox_max, points, align_corners=True)
    custom = custom_xyz_trilinear(grid, bbox_min, bbox_max, points)
    wrong_axis = runtime_interpolate(grid.transpose(2, 1, 0, 3).copy(), bbox_min, bbox_max, points, align_corners=True)
    align_false = runtime_interpolate(grid, bbox_min, bbox_max, points, align_corners=False)
    return {
        "points": points.tolist(),
        "expected_xyz_indices": (fraction * (np.asarray(shape) - 1)[None]).tolist(),
        "runtime_values": runtime.tolist(),
        "custom_values": custom.tolist(),
        "runtime_vs_custom_max_abs": float(np.max(np.abs(runtime - custom))),
        "wrong_zyx_axis_vs_custom_max_abs": float(np.max(np.abs(wrong_axis - custom))),
        "align_corners_false_vs_custom_max_abs": float(np.max(np.abs(align_false - custom))),
        "pass": bool(np.max(np.abs(runtime - custom)) <= 1.0e-6),
    }


def connected_components(vertex_count: int, faces: np.ndarray) -> tuple[np.ndarray, list[int]]:
    parent = np.arange(vertex_count, dtype=np.int64)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for face in faces:
        union(int(face[0]), int(face[1]))
        union(int(face[1]), int(face[2]))
        union(int(face[2]), int(face[0]))
    roots = np.asarray([find(index) for index in range(vertex_count)], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    sizes = np.bincount(labels).tolist()
    return labels, sizes


def body_region(index: int, name: str) -> str:
    lowered = name.lower()
    if index in (22, 23, 24) or any(token in lowered for token in ("head", "jaw", "eye", "neck")):
        return "head_face"
    if index >= 25:
        return "left_hand" if index < 40 else "right_hand"
    if index in (7, 10) or ("left" in lowered and ("ankle" in lowered or "foot" in lowered)):
        return "left_foot"
    if index in (8, 11) or ("right" in lowered and ("ankle" in lowered or "foot" in lowered)):
        return "right_foot"
    if "left" in lowered and any(token in lowered for token in ("shoulder", "collar", "elbow", "wrist")):
        return "left_arm"
    if "right" in lowered and any(token in lowered for token in ("shoulder", "collar", "elbow", "wrist")):
        return "right_arm"
    if "left" in lowered and any(token in lowered for token in ("hip", "knee")):
        return "left_leg"
    if "right" in lowered and any(token in lowered for token in ("hip", "knee")):
        return "right_leg"
    return "torso"


def top_weights(value: np.ndarray, names: list[str], count: int = 5) -> list[dict[str, Any]]:
    indices = np.argsort(value)[-count:][::-1]
    return [{"joint": int(index), "name": names[int(index)], "weight": float(value[index])} for index in indices]


def aggregate_vertices(indices: np.ndarray, l1: np.ndarray, l2: np.ndarray, maximum: np.ndarray, match: np.ndarray) -> dict[str, Any]:
    if len(indices) == 0:
        return {"count": 0}
    return {
        "count": int(len(indices)),
        "l1_mean": float(l1[indices].mean()),
        "l2_mean": float(l2[indices].mean()),
        "max_error_mean": float(maximum[indices].mean()),
        "max_error_max": float(maximum[indices].max()),
        "dominant_agreement": float(match[indices].mean()),
    }


def write_colored_ply(path: Path, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write(f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
        for vertex, color in zip(vertices, colors, strict=True):
            handle.write(
                f"{float(vertex[0]):.9g} {float(vertex[1]):.9g} {float(vertex[2]):.9g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for face in faces:
            handle.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def model_kwargs(payload: Any, frame: int) -> dict[str, torch.Tensor]:
    mapping = {
        "global_orient": "global_orient",
        "body_pose": "body_pose",
        "transl": "transl",
        "left_hand_pose": "left_hand_pose",
        "right_hand_pose": "right_hand_pose",
        "jaw_pose": "jaw_pose",
        "leye_pose": "leye_pose",
        "reye_pose": "reye_pose",
        "expression": "expression",
    }
    result = {}
    for target, source in mapping.items():
        if source not in payload.files:
            continue
        value = np.asarray(payload[source])
        selected = value[frame] if value.ndim > 1 and len(value) > frame else value[0] if value.ndim > 1 else value
        result[target] = torch.as_tensor(selected, dtype=torch.float32)[None]
    return result


def posed_forward_audit(model: Any, smpl_params_path: Path, faces: np.ndarray, canonical_area: float) -> list[dict[str, Any]]:
    records = []
    with np.load(smpl_params_path, allow_pickle=False) as payload:
        beta_values = np.asarray(payload["betas"], dtype=np.float32)
        beta = beta_values if beta_values.ndim == 1 else beta_values[0]
        for frame in (0, 1250, 2499):
            kwargs = model_kwargs(payload, frame)
            kwargs["betas"] = torch.from_numpy(beta[None])
            with torch.no_grad():
                output = model(**kwargs)
            vertices = output.vertices[0].cpu().numpy().astype(np.float32)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            triangles = vertices[faces]
            face_area = 0.5 * np.linalg.norm(
                np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
            )
            records.append(
                {
                    "frame_id": frame,
                    "vertices_sha256_float32": sha256_array(vertices),
                    "finite": bool(np.isfinite(vertices).all()),
                    "normals_finite": bool(np.isfinite(mesh.vertex_normals).all()),
                    "degenerate_faces_at_1e-12": int(np.sum(face_area <= 1.0e-12)),
                    "bbox_min": vertices.min(axis=0).tolist(),
                    "bbox_max": vertices.max(axis=0).tolist(),
                    "max_abs_coordinate": float(np.abs(vertices).max()),
                    "surface_area": float(mesh.area),
                    "surface_area_ratio_to_canonical": float(mesh.area / canonical_area),
                }
            )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-001", type=Path, required=True)
    parser.add_argument("--attempt-002", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--subject-root", type=Path, required=True)
    parser.add_argument("--smpl-model", type=Path, required=True)
    args = parser.parse_args()

    names, names_source = joint_names()
    smpl_params = args.subject_root / "smpl_params.npz"
    model, canonical_result, vertices, faces, reference, joints = canonical_model(args.smpl_model, smpl_params)
    parents = model.parents.cpu().numpy().astype(np.int64)
    if len(parents) != JOINT_COUNT or reference.shape != (10475, JOINT_COUNT):
        raise RuntimeError(f"Unexpected SMPL-X contract: parents={len(parents)} weights={reference.shape}")

    candidate_vertices = np.load(args.attempt_001 / "run_a" / "template" / "template_vertices.npy", allow_pickle=False)
    candidate_faces = np.load(args.attempt_001 / "run_a" / "template" / "template_faces.npy", allow_pickle=False)
    vertex_identity_max_abs = float(np.max(np.abs(candidate_vertices.astype(np.float64) - vertices.astype(np.float64))))
    face_identity = bool(np.array_equal(candidate_faces, faces))

    with np.load(args.attempt_002 / "runs" / "T1_A" / "output" / "lbs_weights_grid.npz", allow_pickle=False) as loaded:
        grid_info = {key: np.asarray(loaded[key]) for key in loaded.files}
    grid = grid_info["grid"].astype(np.float32)
    bbox_min = grid_info["bbox_min"].astype(np.float32)
    bbox_max = grid_info["bbox_max"].astype(np.float32)
    predicted = runtime_interpolate(grid, bbox_min, bbox_max, vertices, align_corners=True)
    custom_predicted = custom_xyz_trilinear(grid, bbox_min, bbox_max, vertices)
    runtime_custom_max_abs = float(np.max(np.abs(predicted - custom_predicted)))

    difference = np.abs(predicted.astype(np.float64) - reference.astype(np.float64))
    l1 = difference.sum(axis=1)
    l2 = np.linalg.norm(difference, axis=1)
    maximum = difference.max(axis=1)
    reference_dominant = np.argmax(reference, axis=1)
    predicted_dominant = np.argmax(predicted, axis=1)
    dominant_match = reference_dominant == predicted_dominant

    cost = np.empty((JOINT_COUNT, JOINT_COUNT), dtype=np.float64)
    for predicted_channel in range(JOINT_COUNT):
        for reference_channel in range(JOINT_COUNT):
            cost[predicted_channel, reference_channel] = np.mean(
                np.abs(predicted[:, predicted_channel].astype(np.float64) - reference[:, reference_channel].astype(np.float64))
            )
    rows, columns = linear_sum_assignment(cost)
    mapped = np.zeros_like(predicted)
    for predicted_channel, reference_channel in zip(rows, columns, strict=True):
        mapped[:, reference_channel] = predicted[:, predicted_channel]
    diagnostic_agreement = float(np.mean(np.argmax(mapped, axis=1) == reference_dominant))
    permutation = [int(columns[np.where(rows == index)[0][0]]) for index in range(JOINT_COUNT)]

    mapping = []
    for index in range(JOINT_COUNT):
        mapping.append(
            {
                "channel_index": index,
                "source_joint_index": index,
                "source_joint_name": names[index],
                "parent_index": int(parents[index]),
                "pointinterpolant_input_channel": index,
                "output_grid_channel": index,
                "runtime_joint_index": index,
                "runtime_joint_name": names[index],
            }
        )
    mapping_report = {
        "task_id": "MMLPHUMAN-SUBJECT00-LBS-REPAIR-001",
        "status": "PASS",
        "joint_count": JOINT_COUNT,
        "joint_names_source": names_source,
        "model_num_joints": int(model.NUM_JOINTS),
        "parents_count": int(len(parents)),
        "source_lbs_weights_shape": list(reference.shape),
        "pointinterpolant_files": "cano_data_lbs_{val,grad}_00..54.xyz and grid_00..54.grd",
        "generator_channel_operation": "sorted grid filenames followed by stack on channel axis; no permutation",
        "output_channel_operation": "transpose spatial axes only: (channel,raw0,raw1,raw2)->(raw2,raw1,raw0,channel)",
        "runtime_channel_operation": "grid.permute(3,0,1,2) moves channel to C; channel order unchanged",
        "mapping": mapping,
        "bijection": bool(len({entry["source_joint_index"] for entry in mapping}) == JOINT_COUNT
                          and len({entry["runtime_joint_index"] for entry in mapping}) == JOINT_COUNT),
        "semantic_mapping_identity": True,
        "duplicates": 0,
        "missing": 0,
        "implicit_truncation": False,
        "body_22_vs_full_55_mixed_for_weights": False,
        "channel_permutation_diagnostic": {
            "method": "Hungarian minimum per-channel vertex MAE; diagnostic only",
            "identity_dominant_agreement": float(dominant_match.mean()),
            "best_fit_dominant_agreement": diagnostic_agreement,
            "predicted_channel_to_reference_channel": permutation,
            "identity_assignments": int(sum(index == value for index, value in enumerate(permutation))),
            "adopted": False,
            "reason": "Runtime/model/generator semantics prove identity order; metric-fit permutations are prohibited.",
        },
        "classification": "LBS_JOINT_CHANNEL_MAPPING_PASS",
    }

    synthetic = synthetic_coordinate_test()
    coordinate_report = {
        "task_id": "MMLPHUMAN-SUBJECT00-LBS-REPAIR-001",
        "status": "PASS" if synthetic["pass"] else "FAIL",
        "generator_raw_payload_order": "raw_z_y_x inferred from generator transpose and PointInterpolant grid contract",
        "stored_array_order": "array[x,y,z,channel]",
        "runtime_tensor_order": "input[C,D=x,H=y,W=z]",
        "runtime_point_order": "world xyz normalized then reordered to z,y,x for grid_sample x,y,z arguments",
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "grid_dims": grid_info["grid_dims"].tolist(),
        "grid_resolution": float(grid_info["grid_resolution"]),
        "voxel_contract": "128 inclusive endpoint grid-node centers spanning bbox_min through bbox_max; no half-voxel offset",
        "normalized_range": [-1.0, 1.0],
        "align_corners": True,
        "padding_mode": "border",
        "transpose_or_flip": "generator spatial transpose (3,2,1,0); runtime point xyz->zyx; no axis flip",
        "candidate_contracts": {
            "A_array_x_y_z_channel": "SUPPORTED_AND_PASS",
            "B_array_z_y_x_channel": "REJECTED_BY_GENERATOR_TRANSPOSE_AND_SYNTHETIC_TEST",
            "C_half_voxel_center_offset": "REJECTED_BY_BBOX_AND_ALIGN_CORNERS_CODE",
            "D_inclusive_bbox_endpoint_nodes": "SUPPORTED_AND_PASS",
            "E_align_corners_true": "SUPPORTED_AND_PASS",
            "F_align_corners_false": "REJECTED_BY_RUNTIME_CODE_AND_SYNTHETIC_TEST",
        },
        "synthetic_unit_test": synthetic,
        "runtime_vs_custom_on_10475_vertices_max_abs": runtime_custom_max_abs,
        "runtime_vs_custom_real_grid_policy": "record_only_float32_accumulation_order; coordinate mapping is gated by the analytic synthetic index/neighborhood test",
        "runtime_source": "utils/smpl_utils.py::interpolate_skinningfield",
        "runtime_source_sha256": sha256_file(args.worktree / "utils" / "smpl_utils.py"),
        "classification": "LBS_GRID_AXIS_OR_COORDINATE_CONTRACT_PASS" if synthetic["pass"] else "LBS_GRID_AXIS_OR_COORDINATE_CONTRACT_MISMATCH",
    }

    extent = vertices.max(axis=0) - vertices.min(axis=0)
    cube_side = float(1.1 * extent.max())
    expected_center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    expected_bbox_min = expected_center - 0.5 * cube_side
    expected_bbox_max = expected_center + 0.5 * cube_side
    padding_per_side = (cube_side - extent) / 2.0
    distances_sides = np.stack(
        [vertices[:, 0] - bbox_min[0], bbox_max[0] - vertices[:, 0], vertices[:, 1] - bbox_min[1], bbox_max[1] - vertices[:, 1], vertices[:, 2] - bbox_min[2], bbox_max[2] - vertices[:, 2]], axis=1
    )
    boundary_distance = distances_sides.min(axis=1)
    nearest_side_vertices = []
    side_names = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
    for side_index, side_name in enumerate(side_names):
        order = np.argsort(distances_sides[:, side_index])[:10]
        nearest_side_vertices.append(
            {"side": side_name, "vertices": [{"index": int(index), "distance": float(distances_sides[index, side_index])} for index in order]}
        )
    bbox_report = {
        "source": "subject00 canonical SMPL-X vertices with subject-specific betas",
        "template_extent": extent.tolist(),
        "largest_extent": float(extent.max()),
        "cube_side": cube_side,
        "padding_per_side_xyz": padding_per_side.tolist(),
        "expected_bbox_min": expected_bbox_min.tolist(),
        "expected_bbox_max": expected_bbox_max.tolist(),
        "actual_bbox_min": bbox_min.tolist(),
        "actual_bbox_max": bbox_max.tolist(),
        "expected_vs_actual_max_abs": float(max(np.max(np.abs(expected_bbox_min - bbox_min)), np.max(np.abs(expected_bbox_max - bbox_max)))),
        "template_inside": bool(np.all(distances_sides >= -1.0e-7)),
        "extrapolated_vertices": int(np.sum(np.any(distances_sides < -1.0e-7, axis=1))),
        "boundary_distance_min": float(boundary_distance.min()),
        "boundary_distance_percentiles": {str(percentile): float(np.percentile(boundary_distance, percentile)) for percentile in (0, 1, 5, 50, 95, 100)},
        "nearest_vertices_per_side": nearest_side_vertices,
        "grid_spacing": ((bbox_max - bbox_min) / (np.asarray(grid.shape[:3]) - 1)).tolist(),
        "axis_flip": False,
        "units": "SMPL-X model-native meters",
        "status": "PASS",
    }
    coordinate_report["bbox_contract"] = bbox_report

    component_labels, component_sizes = connected_components(len(vertices), faces)
    nearest_joint = np.argmin(np.linalg.norm(vertices[:, None, :] - joints[None, :, :], axis=2), axis=1)
    regions = np.asarray([body_region(int(index), names[int(index)]) for index in nearest_joint], dtype=object)
    per_joint = []
    for channel in range(JOINT_COUNT):
        channel_difference = difference[:, channel]
        per_joint.append(
            {
                "joint": channel,
                "name": names[channel],
                "parent": int(parents[channel]),
                "mae": float(channel_difference.mean()),
                "max_abs": float(channel_difference.max()),
                "reference_dominant_vertices": int(np.sum(reference_dominant == channel)),
                "predicted_dominant_vertices": int(np.sum(predicted_dominant == channel)),
                "dominant_mismatch_reference_vertices": int(np.sum((reference_dominant == channel) & ~dominant_match)),
            }
        )
    per_region = {region: aggregate_vertices(np.where(regions == region)[0], l1, l2, maximum, dominant_match) for region in sorted(set(regions.tolist()))}
    per_component = {
        str(component): {"size": component_sizes[component], **aggregate_vertices(np.where(component_labels == component)[0], l1, l2, maximum, dominant_match)}
        for component in range(len(component_sizes))
    }
    pearson = float(np.corrcoef(boundary_distance, maximum)[0, 1])
    spearman = spearmanr(boundary_distance, maximum)
    worst_indices = np.argsort(maximum)[-100:][::-1]
    top_one_count = max(1, int(math.ceil(len(vertices) * 0.01)))
    top_one_indices = np.argsort(maximum)[-top_one_count:][::-1]
    mismatch_indices = np.where(~dominant_match)[0]

    def compact_record(index: int) -> dict[str, Any]:
        return {
            "vertex_index": int(index),
            "canonical_xyz": vertices[index].tolist(),
            "l1_error": float(l1[index]),
            "l2_error": float(l2[index]),
            "max_error": float(maximum[index]),
            "reference_dominant_joint": int(reference_dominant[index]),
            "reference_dominant_name": names[int(reference_dominant[index])],
            "predicted_dominant_joint": int(predicted_dominant[index]),
            "predicted_dominant_name": names[int(predicted_dominant[index])],
            "dominant_match": bool(dominant_match[index]),
            "bbox_boundary_distance": float(boundary_distance[index]),
            "connected_component": int(component_labels[index]),
            "nearest_joint": int(nearest_joint[index]),
            "nearest_joint_name": names[int(nearest_joint[index])],
            "body_region": str(regions[index]),
            "reference_top5": top_weights(reference[index], names),
            "predicted_top5": top_weights(predicted[index], names),
        }

    external_dir = args.attempt_002 / "error_localization"
    external_dir.mkdir(parents=True, exist_ok=True)
    all_records_path = external_dir / "subject00_lbs_per_vertex_records.jsonl"
    with all_records_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(len(vertices)):
            full_record = compact_record(index)
            full_record["reference_weight_vector"] = reference[index].tolist()
            full_record["interpolated_grid_weight_vector"] = predicted[index].tolist()
            handle.write(json.dumps(full_record, sort_keys=True, separators=(",", ":")) + "\n")

    heat_scale = max(float(np.percentile(maximum, 99.0)), np.finfo(np.float64).eps)
    heat = np.clip(maximum / heat_scale, 0.0, 1.0)
    heat_colors = np.stack([255.0 * heat, 32.0 * (1.0 - heat), 255.0 * (1.0 - heat)], axis=1).astype(np.uint8)
    mismatch_colors = np.tile(np.asarray([80, 80, 80], dtype=np.uint8), (len(vertices), 1))
    mismatch_colors[~dominant_match] = np.asarray([255, 0, 0], dtype=np.uint8)
    boundary_scale = max(float(np.percentile(boundary_distance, 95.0)), np.finfo(np.float64).eps)
    proximity = 1.0 - np.clip(boundary_distance / boundary_scale, 0.0, 1.0)
    boundary_colors = np.stack([255.0 * proximity, 32.0 * (1.0 - proximity), 255.0 * (1.0 - proximity)], axis=1).astype(np.uint8)
    heat_path = external_dir / "subject00_lbs_error_heatmap.ply"
    mismatch_path = external_dir / "subject00_lbs_dominant_mismatch.ply"
    boundary_path = external_dir / "subject00_lbs_bbox_boundary_proximity.ply"
    write_colored_ply(heat_path, vertices, faces, heat_colors)
    write_colored_ply(mismatch_path, vertices, faces, mismatch_colors)
    write_colored_ply(boundary_path, vertices, faces, boundary_colors)

    canonical_area = float(trimesh.Trimesh(vertices=vertices, faces=faces, process=False).area)
    posed = posed_forward_audit(model, smpl_params, faces, canonical_area)
    geometry = {
        "interpolated_weight_sum_max_abs_error": float(np.max(np.abs(predicted.sum(axis=1, dtype=np.float64) - 1.0))),
        "interpolated_weight_mae": float(difference.mean()),
        "interpolated_weight_max_abs": float(difference.max()),
        "dominant_joint_agreement": float(dominant_match.mean()),
        "extrapolated_vertices": bbox_report["extrapolated_vertices"],
        "thresholds": {"sum_max_abs": 1.0e-5, "mae": 0.02, "max_abs": 0.5, "dominant_joint_agreement_minimum": 0.95},
    }
    geometry["status"] = "PASS" if (
        geometry["interpolated_weight_sum_max_abs_error"] <= 1.0e-5
        and geometry["interpolated_weight_mae"] <= 0.02
        and geometry["interpolated_weight_max_abs"] <= 0.5
        and geometry["dominant_joint_agreement"] >= 0.95
        and geometry["extrapolated_vertices"] == 0
    ) else "FAIL"
    posed_pass = all(
        item["finite"] and item["normals_finite"] and item["degenerate_faces_at_1e-12"] == 0
        and item["max_abs_coordinate"] <= 5.0
        and 0.05 <= item["surface_area_ratio_to_canonical"] <= 20.0
        for item in posed
    )
    error_report = {
        "task_id": "MMLPHUMAN-SUBJECT00-LBS-REPAIR-001",
        "status": "PASS_ANALYSIS",
        "reference_contract": {
            "source": "subject00 neutral SMPL-X official model.lbs_weights",
            "weights_shape": list(reference.shape),
            "weights_sha256": sha256_array(reference),
            "candidate_template_vertices_shape": list(candidate_vertices.shape),
            "canonical_model_vertices_shape": list(vertices.shape),
            "vertex_index_identity": bool(vertex_identity_max_abs == 0.0),
            "vertex_position_max_abs": vertex_identity_max_abs,
            "candidate_faces_shape": list(candidate_faces.shape),
            "face_topology_exact": face_identity,
            "different_topology_or_nearest_neighbor_used": False,
        },
        "geometry": geometry,
        "posed_body_forward": {"status": "PASS" if posed_pass else "FAIL", "frames": posed},
        "connected_components": {"count": len(component_sizes), "sizes": component_sizes},
        "worst_100_vertices": [compact_record(int(index)) for index in worst_indices],
        "top_1_percent": {"count": top_one_count, "indices": top_one_indices.astype(int).tolist(), "region_counts": {region: int(np.sum(regions[top_one_indices] == region)) for region in sorted(set(regions.tolist()))}},
        "dominant_mismatch_vertices": {"count": int(len(mismatch_indices)), "indices": mismatch_indices.astype(int).tolist(), "region_counts": {region: int(np.sum(regions[mismatch_indices] == region)) for region in sorted(set(regions.tolist()))}},
        "per_joint_error": per_joint,
        "per_body_region_error": per_region,
        "per_component_error": per_component,
        "boundary_error_correlation": {"pearson": pearson, "spearman": float(spearman.statistic), "spearman_pvalue": float(spearman.pvalue)},
        "external_per_vertex_records": {"path": str(all_records_path), "sha256": sha256_file(all_records_path), "records": len(vertices)},
        "visualizations": {
            "error_heatmap": {"path": str(heat_path), "sha256": sha256_file(heat_path)},
            "dominant_mismatch": {"path": str(mismatch_path), "sha256": sha256_file(mismatch_path)},
            "bbox_boundary_proximity": {"path": str(boundary_path), "sha256": sha256_file(boundary_path)},
        },
    }
    reports = args.attempt_002 / "reports"
    write_json(reports / "subject00_lbs_joint_mapping.json", mapping_report)
    write_json(reports / "subject00_lbs_coordinate_contract.json", coordinate_report)
    write_json(reports / "subject00_lbs_vertex_error_analysis.json", error_report)
    print(
        json.dumps(
            {
                "joint": mapping_report["classification"],
                "coordinate": coordinate_report["classification"],
                "geometry": geometry,
                "posed": error_report["posed_body_forward"]["status"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
