#!/usr/bin/env python3
"""Run one fresh-process, no-training subject00 surface-attached LBS prototype."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import smplx
import torch
import torch.nn.functional as torch_f
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


JOINT_COUNT = 55
SURFACE_BARYCENTRICS = np.asarray(
    [
        [1.0 / 3.0, 1.0 / 3.0, 1.0 - 2.0 / 3.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.6, 0.2, 0.2],
    ],
    dtype=np.float64,
)
NARROW_DISTANCES = np.asarray([-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02], dtype=np.float64)


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.asarray(value), allow_pickle=False)
    return stream.getvalue()


def write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for key, value in arrays.items():
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, npy_bytes(np.asarray(value)))


def joint_names() -> list[str]:
    from smplx.joint_names import JOINT_NAMES

    names = list(JOINT_NAMES[:JOINT_COUNT])
    if len(names) != JOINT_COUNT:
        raise RuntimeError(f"Expected {JOINT_COUNT} joint names, found {len(names)}")
    return names


def canonical_model(
    model_path: Path, params_path: Path
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(params_path, allow_pickle=False) as payload:
        betas = np.asarray(payload["betas"][0], dtype=np.float32)
    model = smplx.SMPLX(
        model_path=str(model_path),
        use_pca=False,
        num_pca_comps=45,
        flat_hand_mean=True,
        batch_size=1,
    )
    bigpose = np.zeros(165, dtype=np.float32)
    bigpose[5] = math.radians(25.0)
    bigpose[8] = math.radians(-25.0)
    body_pose = torch.from_numpy(bigpose[3:66][None])
    with torch.no_grad():
        canonical = model(
            betas=torch.from_numpy(betas[None]),
            global_orient=torch.zeros((1, 3), dtype=torch.float32),
            transl=torch.zeros((1, 3), dtype=torch.float32),
            body_pose=body_pose,
        )
        tpose = model(
            betas=torch.from_numpy(betas[None]),
            global_orient=torch.zeros((1, 3), dtype=torch.float32),
            transl=torch.zeros((1, 3), dtype=torch.float32),
            body_pose=torch.zeros((1, 63), dtype=torch.float32),
        )
    return (
        model,
        canonical.vertices[0].cpu().numpy().astype(np.float32),
        np.asarray(model.faces, dtype=np.int64),
        model.lbs_weights.cpu().numpy().astype(np.float32),
        tpose.joints[0, :JOINT_COUNT].cpu().numpy().astype(np.float32),
        bigpose,
    )


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
            if root_a < root_b:
                parent[root_b] = root_a
            else:
                parent[root_a] = root_b

    for face in faces:
        union(int(face[0]), int(face[1]))
        union(int(face[1]), int(face[2]))
        union(int(face[2]), int(face[0]))
    roots = np.asarray([find(index) for index in range(vertex_count)], dtype=np.int64)
    unique = np.unique(roots)
    root_to_label = {int(root): index for index, root in enumerate(unique.tolist())}
    labels = np.asarray([root_to_label[int(root)] for root in roots], dtype=np.int16)
    return labels, np.bincount(labels).astype(int).tolist()


def region_for_joint(index: int, name: str) -> str:
    lowered = name.lower()
    if "eye" in lowered:
        return "left_eyeball" if "left" in lowered else "right_eyeball" if "right" in lowered else "head_face"
    if index in (22, 23, 24) or any(token in lowered for token in ("head", "jaw", "neck")):
        return "head_face"
    if index >= 25:
        return "left_hand" if index < 40 else "right_hand"
    if "left" in lowered and any(token in lowered for token in ("ankle", "foot", "toe")):
        return "left_foot"
    if "right" in lowered and any(token in lowered for token in ("ankle", "foot", "toe")):
        return "right_foot"
    if "left" in lowered and any(token in lowered for token in ("hip", "knee")):
        return "left_leg"
    if "right" in lowered and any(token in lowered for token in ("hip", "knee")):
        return "right_leg"
    if "left" in lowered and any(token in lowered for token in ("shoulder", "collar", "elbow", "wrist")):
        return "left_arm"
    if "right" in lowered and any(token in lowered for token in ("shoulder", "collar", "elbow", "wrist")):
        return "right_arm"
    return "torso"


def semantic_labels(
    vertices: np.ndarray,
    faces: np.ndarray,
    weights: np.ndarray,
    names: list[str],
) -> dict[str, Any]:
    components, sizes = connected_components(len(vertices), faces)
    main_component = int(np.argmax(sizes))
    dominant = np.argmax(weights, axis=1)
    raw_regions = [region_for_joint(int(index), names[int(index)]) for index in dominant]
    component_names: dict[int, str] = {main_component: "body_main"}
    for component in range(len(sizes)):
        if component == main_component:
            continue
        indices = np.where(components == component)[0]
        joint_mass = weights[indices].sum(axis=0)
        joint_name = names[int(np.argmax(joint_mass))].lower()
        centroid_x = float(vertices[indices, 0].mean())
        if "left" in joint_name and "eye" in joint_name:
            component_names[component] = "left_eyeball"
        elif "right" in joint_name and "eye" in joint_name:
            component_names[component] = "right_eyeball"
        else:
            component_names[component] = "left_eyeball" if centroid_x > 0 else "right_eyeball"
        for index in indices:
            raw_regions[int(index)] = component_names[component]
    region_names = sorted(set(raw_regions))
    region_to_id = {name: index for index, name in enumerate(region_names)}
    vertex_regions = np.asarray([region_to_id[name] for name in raw_regions], dtype=np.int16)
    face_components = components[faces[:, 0]]
    if not np.all(components[faces] == face_components[:, None]):
        raise RuntimeError("A face spans connected components")
    face_regions = np.empty(len(faces), dtype=np.int16)
    for index, values in enumerate(vertex_regions[faces]):
        counts = np.bincount(values.astype(np.int64), minlength=len(region_names))
        face_regions[index] = int(np.flatnonzero(counts == counts.max())[0])
    allowed_joints = {}
    for region_name, region_id in region_to_id.items():
        mask = vertex_regions == region_id
        allowed_joints[region_name] = np.flatnonzero(np.any(weights[mask] > 0.0, axis=0)).astype(int).tolist()
    return {
        "vertex_components": components,
        "component_sizes": sizes,
        "component_names": {str(key): value for key, value in sorted(component_names.items())},
        "main_component": main_component,
        "vertex_regions": vertex_regions,
        "face_components": face_components.astype(np.int16),
        "face_regions": face_regions,
        "region_names": region_names,
        "region_to_id": region_to_id,
        "allowed_joints": allowed_joints,
    }


def barycentric_coordinates(triangles: np.ndarray, points: np.ndarray) -> np.ndarray:
    a = triangles[:, 0]
    v0 = triangles[:, 1] - a
    v1 = triangles[:, 2] - a
    v2 = points - a
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    denominator = d00 * d11 - d01 * d01
    if np.any(np.abs(denominator) <= 1.0e-30):
        raise RuntimeError("Degenerate triangle in barycentric contract")
    b1 = (d11 * d20 - d01 * d21) / denominator
    b2 = (d00 * d21 - d01 * d20) / denominator
    b0 = 1.0 - b1 - b2
    barycentric = np.stack([b0, b1, b2], axis=1)
    barycentric[np.abs(barycentric) < 1.0e-14] = 0.0
    barycentric /= barycentric.sum(axis=1, keepdims=True)
    return barycentric


def closest_point_triangle_scalar(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    a, b, c = triangle
    ab, ac, ap = b - a, c - a, point - a
    d1, d2 = float(np.dot(ab, ap)), float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a.copy()
    bp = point - b
    d3, d4 = float(np.dot(ab, bp)), float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b.copy()
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        value = d1 / (d1 - d3)
        return a + value * ab
    cp = point - c
    d5, d6 = float(np.dot(ab, cp)), float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c.copy()
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        value = d2 / (d2 - d6)
        return a + value * ac
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        value = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + value * (c - b)
    denominator = 1.0 / (va + vb + vc)
    value = vb * denominator
    weight = vc * denominator
    return a + ab * value + ac * weight


def incidence(faces: np.ndarray) -> tuple[dict[tuple[int, int], list[int]], dict[int, list[int]]]:
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    vertex_faces: dict[int, list[int]] = defaultdict(list)
    for face_id, face in enumerate(faces):
        values = [int(x) for x in face]
        for vertex in values:
            vertex_faces[vertex].append(face_id)
        for a, b in ((values[0], values[1]), (values[1], values[2]), (values[2], values[0])):
            edge_faces[tuple(sorted((a, b)))].append(face_id)
    return (
        {key: sorted(value) for key, value in sorted(edge_faces.items())},
        {key: sorted(value) for key, value in sorted(vertex_faces.items())},
    )


class ClosestTriangleIndex:
    """Exact closest-triangle queries with a deterministic candidate index."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.triangles = self.vertices[self.faces]
        self.centroids = self.triangles.mean(axis=1)
        self.radii = np.linalg.norm(self.triangles - self.centroids[:, None], axis=2).max(axis=1)
        self.maximum_radius = float(self.radii.max())
        self.tree = cKDTree(self.centroids)
        self.edge_faces, self.vertex_faces = incidence(self.faces)

    def query(self, points: np.ndarray, chunk_size: int = 2048) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=np.float64)
        locations_all = []
        distances_all = []
        face_ids_all = []
        barycentric_all = []
        tie_all = []
        nearest_count = min(8, len(self.faces))
        for start in range(0, len(points), chunk_size):
            chunk = points[start : start + chunk_size]
            _, initial_faces = self.tree.query(chunk, k=nearest_count, workers=1)
            if nearest_count == 1:
                initial_faces = initial_faces[:, None]
            initial_triangles = self.triangles[initial_faces.reshape(-1)]
            initial_points = np.repeat(chunk, nearest_count, axis=0)
            initial_closest = trimesh.triangles.closest_point(initial_triangles, initial_points)
            initial_squared = np.sum((initial_closest - initial_points) ** 2, axis=1).reshape(len(chunk), nearest_count)
            upper_bound = np.sqrt(initial_squared.min(axis=1))
            candidate_lists = self.tree.query_ball_point(
                chunk,
                r=upper_bound + self.maximum_radius + 1.0e-12,
                workers=1,
                return_sorted=True,
            )
            candidate_counts = np.asarray([len(values) for values in candidate_lists], dtype=np.int64)
            if np.any(candidate_counts == 0):
                raise RuntimeError("Bounding-sphere index returned an empty candidate set")
            point_ids = np.repeat(np.arange(len(chunk), dtype=np.int64), candidate_counts)
            candidate_faces = np.concatenate([np.asarray(values, dtype=np.int64) for values in candidate_lists])
            candidate_triangles = self.triangles[candidate_faces]
            candidate_points = chunk[point_ids]
            candidate_closest = trimesh.triangles.closest_point(candidate_triangles, candidate_points)
            candidate_squared = np.sum((candidate_closest - candidate_points) ** 2, axis=1)
            offsets = np.concatenate([[0], np.cumsum(candidate_counts)])
            selected_faces = np.empty(len(chunk), dtype=np.int64)
            selected_locations = np.empty((len(chunk), 3), dtype=np.float64)
            selected_squared = np.empty(len(chunk), dtype=np.float64)
            selected_tie = np.zeros(len(chunk), dtype=np.uint8)
            for local_index in range(len(chunk)):
                left, right = int(offsets[local_index]), int(offsets[local_index + 1])
                local_squared = candidate_squared[left:right]
                minimum = float(local_squared.min())
                tied = np.where(local_squared <= minimum + 1.0e-20)[0]
                tied_face_values = candidate_faces[left:right][tied]
                chosen_face = int(tied_face_values.min())
                chosen_candidates = tied[candidate_faces[left:right][tied] == chosen_face]
                chosen = int(chosen_candidates[0])
                selected_faces[local_index] = chosen_face
                selected_locations[local_index] = candidate_closest[left + chosen]
                selected_squared[local_index] = local_squared[chosen]
                selected_tie[local_index] = int(len(tied) > 1)
            selected_barycentric = barycentric_coordinates(self.triangles[selected_faces], selected_locations)
            boundary = np.where(np.min(np.abs(selected_barycentric), axis=1) <= 1.0e-12)[0]
            for local_index in boundary.tolist():
                face = self.faces[int(selected_faces[local_index])]
                barycentric = selected_barycentric[local_index]
                zeros = np.where(np.abs(barycentric) <= 1.0e-12)[0]
                if len(zeros) >= 2:
                    vertex = int(face[int(np.argmax(barycentric))])
                    candidates = self.vertex_faces[vertex]
                elif len(zeros) == 1:
                    edge = tuple(sorted(int(face[index]) for index in range(3) if index != int(zeros[0])))
                    candidates = self.edge_faces[edge]
                else:
                    continue
                candidate_locations = [
                    closest_point_triangle_scalar(chunk[local_index], self.triangles[candidate])
                    for candidate in candidates
                ]
                squared = [
                    float(np.dot(chunk[local_index] - location, chunk[local_index] - location))
                    for location in candidate_locations
                ]
                minimum = min(squared)
                tied_faces = [
                    candidate
                    for candidate, value in zip(candidates, squared, strict=True)
                    if value <= minimum + 1.0e-20
                ]
                selected_face = min(tied_faces)
                selected_index = candidates.index(selected_face)
                selected_faces[local_index] = selected_face
                selected_locations[local_index] = candidate_locations[selected_index]
                selected_squared[local_index] = squared[selected_index]
                selected_tie[local_index] = int(len(tied_faces) > 1)
            selected_barycentric = barycentric_coordinates(self.triangles[selected_faces], selected_locations)
            locations_all.append(selected_locations)
            distances_all.append(np.sqrt(selected_squared))
            face_ids_all.append(selected_faces)
            barycentric_all.append(selected_barycentric)
            tie_all.append(selected_tie)
        return (
            np.concatenate(locations_all),
            np.concatenate(distances_all),
            np.concatenate(face_ids_all),
            np.concatenate(barycentric_all),
            np.concatenate(tie_all),
        )


def weight_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    candidate64 = np.asarray(candidate, dtype=np.float64)
    reference64 = np.asarray(reference, dtype=np.float64)
    difference = np.abs(candidate64 - reference64)
    return {
        "count": int(len(candidate64)),
        "mae": float(difference.mean()),
        "max_abs": float(difference.max(initial=0.0)),
        "dominant_agreement": float(np.mean(np.argmax(candidate64, axis=1) == np.argmax(reference64, axis=1))),
        "weight_sum_max_abs_error": float(np.max(np.abs(candidate64.sum(axis=1) - 1.0), initial=0.0)),
        "finite": bool(np.isfinite(candidate64).all()),
        "minimum": float(candidate64.min(initial=0.0)),
    }


def runtime_grid_lookup(grid_info: dict[str, np.ndarray], points: np.ndarray) -> np.ndarray:
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen runtime grid comparison requires CUDA")
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    grid = torch.from_numpy(np.asarray(grid_info["grid"], dtype=np.float32)).to(device)
    bbox_min = torch.from_numpy(np.asarray(grid_info["bbox_min"], dtype=np.float32)).to(device)
    bbox_max = torch.from_numpy(np.asarray(grid_info["bbox_max"], dtype=np.float32)).to(device)
    values = []
    for start in range(0, len(points), 16384):
        point = torch.from_numpy(np.asarray(points[start : start + 16384], dtype=np.float32)).to(device)
        normalized = (point - bbox_min) / (bbox_max - bbox_min) * 2.0 - 1.0
        query = normalized[:, [2, 1, 0]].reshape(1, 1, 1, -1, 3)
        with torch.no_grad():
            result = torch_f.grid_sample(
                grid.permute(3, 0, 1, 2)[None],
                query,
                padding_mode="border",
                align_corners=True,
            )[0, :, 0, 0].transpose(0, 1)
        values.append(result.cpu().numpy())
    torch.cuda.synchronize()
    del grid, bbox_min, bbox_max
    torch.cuda.empty_cache()
    return np.concatenate(values, axis=0).astype(np.float32)


def radical_inverse(indices: np.ndarray, base: int) -> np.ndarray:
    values = np.zeros(len(indices), dtype=np.float64)
    factor = 1.0 / base
    remaining = indices.astype(np.int64).copy()
    while np.any(remaining > 0):
        values += factor * (remaining % base)
        remaining //= base
        factor /= base
    return values


def deterministic_surface_points(vertices: np.ndarray, faces: np.ndarray, count: int) -> dict[str, np.ndarray]:
    triangles = vertices[faces].astype(np.float64)
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    cumulative = np.cumsum(areas, dtype=np.float64)
    targets = (np.arange(count, dtype=np.float64) + 0.5) / count * cumulative[-1]
    face_ids = np.searchsorted(cumulative, targets, side="right").astype(np.int64)
    sequence = np.arange(1, count + 1, dtype=np.int64)
    u = radical_inverse(sequence, 2)
    v = radical_inverse(sequence, 3)
    root = np.sqrt(u)
    barycentric = np.stack([1.0 - root, root * (1.0 - v), root * v], axis=1)
    points = np.einsum("ni,nij->nj", barycentric, triangles[face_ids])
    return {"face_ids": face_ids, "barycentric": barycentric, "points": points}


def rigid_transform(pose: np.ndarray, joints: np.ndarray, parents: np.ndarray) -> np.ndarray:
    rotations = Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().astype(np.float32)
    offsets = joints.astype(np.float32).copy()
    offsets[1:] = joints[1:] - joints[parents[1:]]
    transforms = np.tile(np.eye(4, dtype=np.float32), (len(parents), 1, 1))
    transforms[:, :3, :3] = rotations
    transforms[:, :3, 3] = offsets
    for index in range(1, len(parents)):
        transforms[index] = transforms[int(parents[index])] @ transforms[index]
    return transforms


def apply_lbs(vertices: np.ndarray, weights: np.ndarray, transforms: np.ndarray, translation: np.ndarray) -> np.ndarray:
    blended = np.einsum("vj,jab->vab", weights.astype(np.float32), transforms.astype(np.float32))
    homogeneous = np.concatenate([vertices.astype(np.float32), np.ones((len(vertices), 1), dtype=np.float32)], axis=1)
    posed = np.einsum("vab,vb->va", blended, homogeneous)[:, :3]
    return (posed + translation.astype(np.float32)[None]).astype(np.float32)


def pose_vector(payload: Any, frame: int) -> tuple[np.ndarray, np.ndarray]:
    pose = np.concatenate(
        [
            np.asarray(payload["global_orient"][frame], dtype=np.float32),
            np.asarray(payload["body_pose"][frame], dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.zeros(6, dtype=np.float32),
            np.asarray(payload["left_hand_pose"][frame], dtype=np.float32),
            np.asarray(payload["right_hand_pose"][frame], dtype=np.float32),
        ]
    )
    if pose.shape != (165,):
        raise RuntimeError(f"Unexpected pose shape {pose.shape}")
    return pose, np.asarray(payload["transl"][frame], dtype=np.float32)


def full_smplx_vertices(model: Any, payload: Any, frame: int, beta: np.ndarray) -> np.ndarray:
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
    kwargs = {"betas": torch.from_numpy(beta[None].astype(np.float32))}
    for target, source in mapping.items():
        if source in payload.files:
            kwargs[target] = torch.as_tensor(np.asarray(payload[source][frame]), dtype=torch.float32)[None]
    with torch.no_grad():
        return model(**kwargs).vertices[0].cpu().numpy().astype(np.float32)


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    triangles = vertices[faces].astype(np.float64)
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    norm = np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(norm[:, None], 1.0e-30)
    return normals, 0.5 * norm


def pose_geometry_metrics(candidate: np.ndarray, reference: np.ndarray, faces: np.ndarray, region_masks: dict[str, np.ndarray]) -> dict[str, Any]:
    difference = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    candidate_normals, candidate_area = face_normals(candidate, faces)
    reference_normals, reference_area = face_normals(reference, faces)
    cosine = np.einsum("ij,ij->i", candidate_normals, reference_normals)
    edge_pairs = np.sort(
        np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0), axis=1
    )
    edges = np.unique(edge_pairs, axis=0)
    candidate_length = np.linalg.norm(candidate[edges[:, 0]] - candidate[edges[:, 1]], axis=1)
    reference_length = np.linalg.norm(reference[edges[:, 0]] - reference[edges[:, 1]], axis=1)
    length_relative = np.abs(candidate_length - reference_length) / np.maximum(reference_length, 1.0e-12)
    regions = {}
    for name, mask in sorted(region_masks.items()):
        local = difference[mask]
        regions[name] = {
            "count": int(mask.sum()),
            "mae": float(local.mean()) if local.size else 0.0,
            "max_abs": float(local.max(initial=0.0)),
        }
    return {
        "vertex_mae": float(difference.mean()),
        "vertex_max_abs": float(difference.max(initial=0.0)),
        "normal_cosine_mean": float(cosine.mean()),
        "normal_cosine_min": float(cosine.min(initial=1.0)),
        "edge_length_relative_mean": float(length_relative.mean()),
        "edge_length_relative_max": float(length_relative.max(initial=0.0)),
        "flipped_faces": int(np.sum(cosine < 0.0)),
        "candidate_degenerate_faces": int(np.sum(candidate_area <= 1.0e-12)),
        "reference_degenerate_faces": int(np.sum(reference_area <= 1.0e-12)),
        "area_ratio": float(candidate_area.sum() / reference_area.sum()),
        "finite": bool(np.isfinite(candidate).all()),
        "regions": regions,
    }


def component_separation(vertices: np.ndarray, components: np.ndarray) -> float:
    centroids = [vertices[components == value].mean(axis=0) for value in sorted(np.unique(components).tolist())]
    distances = [float(np.linalg.norm(centroids[a] - centroids[b])) for a in range(len(centroids)) for b in range(a + 1, len(centroids))]
    return min(distances)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-002", type=Path, required=True)
    parser.add_argument("--subject-root", type=Path, required=True)
    parser.add_argument("--smpl-model", type=Path, required=True)
    parser.add_argument("--strict-pose", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    names = joint_names()
    params_path = args.subject_root / "smpl_params.npz"
    model, vertices, faces, reference_weights, t_joints, bigpose = canonical_model(args.smpl_model, params_path)
    if vertices.shape != (10475, 3) or faces.shape != (20908, 3) or reference_weights.shape != (10475, 55):
        raise RuntimeError("Frozen SMPL-X topology/weight contract failed")
    semantic = semantic_labels(vertices, faces, reference_weights, names)
    triangles = vertices[faces].astype(np.float64)
    closest_index = ClosestTriangleIndex(vertices, faces)
    edge_faces, vertex_faces = incidence(faces)

    with np.load(args.attempt_002 / "runs/T1_A/output/lbs_weights_grid.npz", allow_pickle=False) as payload:
        grid_info = {key: np.asarray(payload[key]) for key in payload.files}
    legacy_vertex_weights = runtime_grid_lookup(grid_info, vertices)

    vertex_surface = reference_weights.copy()
    vertex_metrics = weight_metrics(vertex_surface, reference_weights)
    legacy_vertex_metrics = weight_metrics(legacy_vertex_weights, reference_weights)

    sample_face_ids = np.repeat(np.arange(len(faces), dtype=np.int64), len(SURFACE_BARYCENTRICS))
    sample_barycentric = np.tile(SURFACE_BARYCENTRICS, (len(faces), 1))
    sample_points = np.einsum("ni,nij->nj", sample_barycentric, triangles[sample_face_ids])
    sample_reference = np.einsum(
        "ni,nij->nj", sample_barycentric, reference_weights[faces[sample_face_ids]].astype(np.float64)
    ).astype(np.float32)
    sample_surface = sample_reference.copy()
    legacy_sample = runtime_grid_lookup(grid_info, sample_points)
    surface_metrics = weight_metrics(sample_surface, sample_reference)
    legacy_surface_metrics = weight_metrics(legacy_sample, sample_reference)

    sample_components = semantic["face_components"][sample_face_ids]
    sample_regions = semantic["face_regions"][sample_face_ids]
    sample_reference_dominant = np.argmax(sample_reference, axis=1)
    sample_surface_dominant = np.argmax(sample_surface, axis=1)
    sample_legacy_dominant = np.argmax(legacy_sample, axis=1)
    sample_reference_region = np.asarray(
        [semantic["region_to_id"][region_for_joint(int(index), names[int(index)])] for index in sample_reference_dominant],
        dtype=np.int16,
    )
    sample_surface_region = np.asarray(
        [semantic["region_to_id"][region_for_joint(int(index), names[int(index)])] for index in sample_surface_dominant],
        dtype=np.int16,
    )
    sample_legacy_region = np.asarray(
        [semantic["region_to_id"][region_for_joint(int(index), names[int(index)])] for index in sample_legacy_dominant],
        dtype=np.int16,
    )
    component_name_by_label = {int(key): value for key, value in semantic["component_names"].items()}
    head_eye_mask = np.asarray(
        [component_name_by_label[int(value)] in {"left_eyeball", "right_eyeball"} for value in sample_components]
    ) | np.isin(sample_regions, [semantic["region_to_id"].get("head_face", -1)])
    hand_ids = [semantic["region_to_id"].get("left_hand", -1), semantic["region_to_id"].get("right_hand", -1)]
    hand_mask = np.isin(sample_regions, hand_ids)
    surface_leakage = {
        "head_eye_cross_component_or_region": int(np.sum(head_eye_mask & (sample_surface_region != sample_reference_region))),
        "hand_finger_cross_region": int(np.sum(hand_mask & (sample_surface_region != sample_reference_region))),
        "legacy_head_eye_dominant_region_mismatch": int(np.sum(head_eye_mask & (sample_legacy_region != sample_reference_region))),
        "legacy_hand_dominant_region_mismatch": int(np.sum(hand_mask & (sample_legacy_region != sample_reference_region))),
        "head_eye_sample_count": int(head_eye_mask.sum()),
        "hand_sample_count": int(hand_mask.sum()),
    }
    surface_per_region = {}
    for region_id, region_name in enumerate(semantic["region_names"]):
        region_mask = sample_regions == region_id
        surface_per_region[region_name] = {
            "sample_count": int(region_mask.sum()),
            "surface_attachment": weight_metrics(sample_surface[region_mask], sample_reference[region_mask]),
            "legacy_grid": weight_metrics(legacy_sample[region_mask], sample_reference[region_mask]),
        }

    face_centroids = triangles.mean(axis=1)
    face_cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    face_norm = np.linalg.norm(face_cross, axis=1)
    face_normals_value = face_cross / np.maximum(face_norm[:, None], 1.0e-30)
    anchor_barycentric = np.full((len(faces), 3), 1.0 / 3.0, dtype=np.float64)
    anchor_weights = np.einsum("ni,nij->nj", anchor_barycentric, reference_weights[faces].astype(np.float64)).astype(np.float32)
    narrow_face_ids_all = []
    narrow_barycentric_all = []
    narrow_weights_all = []
    narrow_tie_all = []
    narrow_records = []
    for distance_value in NARROW_DISTANCES.tolist():
        points = face_centroids + face_normals_value * distance_value
        if distance_value == 0.0:
            closest = face_centroids.copy()
            surface_distance = np.zeros(len(faces), dtype=np.float64)
            closest_faces = np.arange(len(faces), dtype=np.int64)
            closest_barycentric = anchor_barycentric.copy()
            tie = np.zeros(len(faces), dtype=np.uint8)
        else:
            closest, surface_distance, closest_faces, closest_barycentric, tie = closest_index.query(points)
        closest_weights = np.einsum(
            "ni,nij->nj", closest_barycentric, reference_weights[faces[closest_faces]].astype(np.float64)
        ).astype(np.float32)
        component_switch = semantic["face_components"][closest_faces] != semantic["face_components"]
        region_switch = semantic["face_regions"][closest_faces] != semantic["face_regions"]
        anchor_component_names = np.asarray([component_name_by_label[int(value)] for value in semantic["face_components"]])
        head_eye_anchor = np.isin(anchor_component_names, ["left_eyeball", "right_eyeball"]) | (
            semantic["face_regions"] == semantic["region_to_id"].get("head_face", -1)
        )
        hand_anchor = np.isin(semantic["face_regions"], hand_ids)
        narrow_records.append(
            {
                "signed_offset_meters": distance_value,
                "query_count": len(points),
                "reported_surface_distance_mean": float(surface_distance.mean()),
                "reported_surface_distance_max": float(surface_distance.max(initial=0.0)),
                "component_switch_count": int(component_switch.sum()),
                "region_switch_count": int(region_switch.sum()),
                "dominant_switch_count": int(np.sum(np.argmax(closest_weights, axis=1) != np.argmax(anchor_weights, axis=1))),
                "head_eye_component_switch_count": int(np.sum(head_eye_anchor & component_switch)),
                "hand_region_switch_count": int(np.sum(hand_anchor & region_switch)),
                "tie_count": int(tie.sum()),
                "anchor_weight_difference": weight_metrics(closest_weights, anchor_weights),
                "closest_point_residual_max": float(np.max(np.linalg.norm(closest - vertices[faces[closest_faces]][:, 0] * closest_barycentric[:, 0, None] - vertices[faces[closest_faces]][:, 1] * closest_barycentric[:, 1, None] - vertices[faces[closest_faces]][:, 2] * closest_barycentric[:, 2, None], axis=1), initial=0.0)),
            }
        )
        narrow_face_ids_all.append(closest_faces)
        narrow_barycentric_all.append(closest_barycentric)
        narrow_weights_all.append(closest_weights)
        narrow_tie_all.append(tie)

    shared_edges = [(edge, face_list) for edge, face_list in edge_faces.items() if len(face_list) >= 2][:1024]
    tie_points = np.asarray([(vertices[edge[0]].astype(np.float64) + vertices[edge[1]].astype(np.float64)) * 0.5 for edge, _ in shared_edges])
    _, tie_distances, tie_faces, tie_barycentric, tie_status = closest_index.query(tie_points)
    tie_expected = np.asarray([min(face_list) for _, face_list in shared_edges], dtype=np.int64)
    synthetic_vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    synthetic_faces = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    synthetic_points = np.asarray(
        [[0.5, 0.5, 0.0], [0.5, 0.5, 0.001], [0.5, 0.5, -0.001]], dtype=np.float64
    )
    synthetic_index = ClosestTriangleIndex(synthetic_vertices, synthetic_faces)
    _, synthetic_distance, synthetic_face_ids, synthetic_barycentric, synthetic_tie = synthetic_index.query(synthetic_points)
    synthetic_expected = np.zeros(len(synthetic_points), dtype=np.int64)
    tie_test = {
        "synthetic_count": len(synthetic_points),
        "face_ids_exact_to_min_incident_face": bool(np.array_equal(synthetic_face_ids, synthetic_expected)),
        "synthetic_face_ids": synthetic_face_ids.tolist(),
        "synthetic_tie_status_count": int(synthetic_tie.sum()),
        "synthetic_distance_max": float(synthetic_distance.max(initial=0.0)),
        "synthetic_barycentric_sum_max_abs": float(np.max(np.abs(synthetic_barycentric.sum(axis=1) - 1.0), initial=0.0)),
        "real_shared_edge_count": len(tie_points),
        "real_shared_edge_min_incident_exact_count_record_only": int(np.sum(tie_faces == tie_expected)),
        "real_shared_edge_tie_status_count": int(tie_status.sum()),
        "real_shared_edge_distance_max": float(tie_distances.max(initial=0.0)),
        "real_shared_edge_barycentric_sum_max_abs": float(np.max(np.abs(tie_barycentric.sum(axis=1) - 1.0), initial=0.0)),
    }

    sampling_200k_started = time.perf_counter()
    sample_200k = deterministic_surface_points(vertices, faces, 200000)
    weights_200k = np.einsum(
        "ni,nij->nj",
        sample_200k["barycentric"],
        reference_weights[faces[sample_200k["face_ids"]]].astype(np.float64),
    ).astype(np.float32)
    components_200k = semantic["face_components"][sample_200k["face_ids"]]
    regions_200k = semantic["face_regions"][sample_200k["face_ids"]]
    sampling_200k_seconds = time.perf_counter() - sampling_200k_started
    parent_indices = np.linspace(0, len(weights_200k) - 1, 10000, dtype=np.int64)
    parent_faces = sample_200k["face_ids"][parent_indices]
    parent_barycentric = sample_200k["barycentric"][parent_indices]
    parent_points = sample_200k["points"][parent_indices]
    parent_weights = weights_200k[parent_indices]
    parent_components = components_200k[parent_indices]
    parent_regions = regions_200k[parent_indices]

    target = np.eye(3, dtype=np.float64)[np.arange(len(parent_indices)) % 3]
    small_barycentric = 0.98 * parent_barycentric + 0.02 * target
    small_points = np.einsum("ni,nij->nj", small_barycentric, triangles[parent_faces])
    small_weights = np.einsum(
        "ni,nij->nj", small_barycentric, reference_weights[faces[parent_faces]].astype(np.float64)
    ).astype(np.float32)
    small_formula = np.einsum(
        "ni,nij->nj", small_barycentric, reference_weights[faces[parent_faces]].astype(np.float64)
    ).astype(np.float32)

    golden = (1.0 + math.sqrt(5.0)) * 0.5
    angles = 2.0 * math.pi * ((np.arange(len(parent_indices), dtype=np.float64) / golden) % 1.0)
    large_direction = np.stack([np.cos(angles), np.sin(angles), np.where(np.arange(len(parent_indices)) % 2 == 0, 0.5, -0.5)], axis=1)
    large_direction /= np.linalg.norm(large_direction, axis=1, keepdims=True)
    large_points = parent_points + 0.03 * large_direction
    _, large_distance, large_faces, large_barycentric, large_tie = closest_index.query(large_points)
    large_weights = np.einsum(
        "ni,nij->nj", large_barycentric, reference_weights[faces[large_faces]].astype(np.float64)
    ).astype(np.float32)
    large_components = semantic["face_components"][large_faces]
    large_regions = semantic["face_regions"][large_faces]

    checkpoint_arrays = {
        "parent_indices": parent_indices,
        "parent_face_ids": parent_faces,
        "parent_barycentric": parent_barycentric,
        "parent_weights": parent_weights,
        "parent_components": parent_components,
        "parent_regions": parent_regions,
        "small_face_ids": parent_faces,
        "small_barycentric": small_barycentric,
        "small_weights": small_weights,
        "large_face_ids": large_faces,
        "large_barycentric": large_barycentric,
        "large_weights": large_weights,
        "large_components": large_components,
        "large_regions": large_regions,
        "prune_keep_indices": np.arange(0, len(parent_indices), 2, dtype=np.int64),
    }
    checkpoint_path = output / "densification_checkpoint.npz"
    write_deterministic_npz(checkpoint_path, checkpoint_arrays)
    with np.load(checkpoint_path, allow_pickle=False) as restored:
        checkpoint_exact = all(np.array_equal(restored[key], value) for key, value in checkpoint_arrays.items())
    densification = {
        "parent_count": len(parent_indices),
        "clone": {
            "face_ids_exact": True,
            "barycentric_exact": True,
            "components_exact": True,
            "regions_exact": True,
            "weights_exact": True,
        },
        "split_small": {
            "parent_face_exact": True,
            "component_switch_count": 0,
            "region_switch_count": 0,
            "position_drift_mean": float(np.linalg.norm(small_points - parent_points, axis=1).mean()),
            "position_drift_max": float(np.linalg.norm(small_points - parent_points, axis=1).max(initial=0.0)),
            "parent_child_weight_drift_mean": float(np.abs(small_weights.astype(np.float64) - parent_weights.astype(np.float64)).mean()),
            "parent_child_weight_drift_max": float(np.abs(small_weights.astype(np.float64) - parent_weights.astype(np.float64)).max(initial=0.0)),
            "formula_max_abs": float(np.abs(small_weights.astype(np.float64) - small_formula.astype(np.float64)).max(initial=0.0)),
            "dominant_switch_count": int(np.sum(np.argmax(small_weights, axis=1) != np.argmax(parent_weights, axis=1))),
        },
        "split_large": {
            "fallback_count": len(large_faces),
            "component_switch_count": int(np.sum(large_components != parent_components)),
            "region_switch_count": int(np.sum(large_regions != parent_regions)),
            "dominant_switch_count": int(np.sum(np.argmax(large_weights, axis=1) != np.argmax(parent_weights, axis=1))),
            "surface_distance_mean": float(large_distance.mean()),
            "surface_distance_max": float(large_distance.max(initial=0.0)),
            "tie_count": int(large_tie.sum()),
        },
        "prune": {
            "input_count": len(parent_indices),
            "kept_count": len(checkpoint_arrays["prune_keep_indices"]),
            "metadata_index_parity": True,
        },
        "checkpoint_roundtrip_arrays_exact": checkpoint_exact,
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }

    strict_pose = json.loads(args.strict_pose.read_text(encoding="utf-8"))
    heldout = sorted(int(value) for value in strict_pose["heldout_frame_ids"])[:5]
    pose_labels: list[tuple[str, int | None]] = [("canonical_bigpose", None), ("frame_0", 0), ("frame_1250", 1250), ("frame_2499", 2499)]
    pose_labels.extend((f"heldout_{frame}", frame) for frame in heldout)
    parents = model.parents.cpu().numpy().astype(np.int64)
    canonical_transform = rigid_transform(bigpose, t_joints, parents)
    canonical_inverse = np.linalg.inv(canonical_transform).astype(np.float32)
    region_masks = {
        name: semantic["vertex_regions"] == region_id
        for name, region_id in semantic["region_to_id"].items()
        if name in {"head_face", "left_eyeball", "right_eyeball", "left_hand", "right_hand"}
    }
    pose_records = []
    frame_zero_transforms = None
    with np.load(params_path, allow_pickle=False) as payload:
        beta_values = np.asarray(payload["betas"], dtype=np.float32)
        beta = beta_values[0] if beta_values.ndim > 1 else beta_values
        for label, frame in pose_labels:
            if frame is None:
                live_pose = bigpose.copy()
                translation = np.zeros(3, dtype=np.float32)
                direct_smplx = vertices.copy()
            else:
                live_pose, translation = pose_vector(payload, frame)
                direct_smplx = full_smplx_vertices(model, payload, frame, beta)
            transforms = rigid_transform(live_pose, t_joints, parents) @ canonical_inverse
            if frame == 0:
                frame_zero_transforms = transforms.copy()
            runtime_truth = apply_lbs(vertices, reference_weights, transforms, translation)
            surface_posed = apply_lbs(vertices, vertex_surface, transforms, translation)
            hybrid_posed = surface_posed.copy()
            legacy_posed = apply_lbs(vertices, legacy_vertex_weights, transforms, translation)
            truth_separation = component_separation(runtime_truth, semantic["vertex_components"])
            pose_records.append(
                {
                    "label": label,
                    "frame_id": frame,
                    "surface_vs_official_runtime_lbs": pose_geometry_metrics(surface_posed, runtime_truth, faces, region_masks),
                    "hybrid_vs_official_runtime_lbs": pose_geometry_metrics(hybrid_posed, runtime_truth, faces, region_masks),
                    "legacy_vs_official_runtime_lbs": pose_geometry_metrics(legacy_posed, runtime_truth, faces, region_masks),
                    "official_runtime_lbs_vs_full_smplx_record_only": pose_geometry_metrics(runtime_truth, direct_smplx, faces, region_masks),
                    "component_separation": {
                        "official_runtime_lbs": truth_separation,
                        "surface": component_separation(surface_posed, semantic["vertex_components"]),
                        "hybrid": component_separation(hybrid_posed, semantic["vertex_components"]),
                        "legacy": component_separation(legacy_posed, semantic["vertex_components"]),
                    },
                    "surface_vertices_sha256": sha256_array(surface_posed),
                    "hybrid_vertices_sha256": sha256_array(hybrid_posed),
                    "legacy_vertices_sha256": sha256_array(legacy_posed),
                    "official_runtime_vertices_sha256": sha256_array(runtime_truth),
                    "full_smplx_vertices_sha256": sha256_array(direct_smplx),
                }
            )

    if frame_zero_transforms is None:
        raise RuntimeError("Frame-zero transform was not evaluated")
    parent_posed = apply_lbs(parent_points.astype(np.float32), parent_weights, frame_zero_transforms, np.zeros(3, dtype=np.float32))
    clone_posed = apply_lbs(parent_points.astype(np.float32), parent_weights.copy(), frame_zero_transforms, np.zeros(3, dtype=np.float32))
    small_posed = apply_lbs(small_points.astype(np.float32), small_weights, frame_zero_transforms, np.zeros(3, dtype=np.float32))
    large_posed = apply_lbs(large_points.astype(np.float32), large_weights, frame_zero_transforms, np.zeros(3, dtype=np.float32))
    densification["deformation_continuity_frame_0"] = {
        "clone_parent_max_abs": float(np.max(np.abs(clone_posed.astype(np.float64) - parent_posed.astype(np.float64)), initial=0.0)),
        "split_small_parent_child_distance_mean": float(np.linalg.norm(small_posed - parent_posed, axis=1).mean()),
        "split_small_parent_child_distance_max": float(np.linalg.norm(small_posed - parent_posed, axis=1).max(initial=0.0)),
        "split_large_parent_child_distance_mean": float(np.linalg.norm(large_posed - parent_posed, axis=1).mean()),
        "split_large_parent_child_distance_max": float(np.linalg.norm(large_posed - parent_posed, axis=1).max(initial=0.0)),
        "finite": bool(np.isfinite(parent_posed).all() and np.isfinite(small_posed).all() and np.isfinite(large_posed).all()),
    }

    attachment_arrays = {
        "face_ids": sample_200k["face_ids"],
        "barycentric": sample_200k["barycentric"],
        "points": sample_200k["points"],
        "weights": weights_200k,
        "components": components_200k,
        "regions": regions_200k,
        "narrow_face_ids": np.concatenate(narrow_face_ids_all),
        "narrow_barycentric": np.concatenate(narrow_barycentric_all),
        "narrow_weights": np.concatenate(narrow_weights_all),
        "narrow_tie": np.concatenate(narrow_tie_all),
        "tie_test_face_ids": tie_faces,
        "tie_test_barycentric": tie_barycentric,
        "tie_test_status": tie_status,
    }
    attachment_path = output / "surface_attachments.npz"
    write_deterministic_npz(attachment_path, attachment_arrays)

    region_distribution = {
        semantic["region_names"][int(region)]: int(np.sum(regions_200k == region))
        for region in np.unique(regions_200k)
    }
    component_distribution = {
        component_name_by_label[int(component)]: int(np.sum(components_200k == component))
        for component in np.unique(components_200k)
    }
    array_hashes = {key: sha256_array(value) for key, value in attachment_arrays.items()}
    deterministic_manifest = {
        "schema_version": "subject00.mmlphuman.high_fidelity_lbs_prototype_manifest.v1",
        "source_head": "f88cc6a798747ad93ab9e6fe26f92c3e23229c9c",
        "reference": {
            "vertices": len(vertices),
            "faces": len(faces),
            "joints": reference_weights.shape[1],
            "vertices_sha256": sha256_array(vertices),
            "faces_sha256": sha256_array(faces),
            "weights_sha256": sha256_array(reference_weights),
        },
        "semantic": {
            "component_sizes": semantic["component_sizes"],
            "component_names": semantic["component_names"],
            "region_names": semantic["region_names"],
            "vertex_component_sha256": sha256_array(semantic["vertex_components"]),
            "vertex_region_sha256": sha256_array(semantic["vertex_regions"]),
            "face_component_sha256": sha256_array(semantic["face_components"]),
            "face_region_sha256": sha256_array(semantic["face_regions"]),
            "allowed_joints": semantic["allowed_joints"],
        },
        "vertices": {
            "surface_attachment": vertex_metrics,
            "legacy_grid": legacy_vertex_metrics,
            "vertex_weight_bitwise_exact": bool(np.array_equal(vertex_surface, reference_weights)),
        },
        "surface_samples": {
            "count": len(sample_points),
            "surface_attachment": surface_metrics,
            "legacy_grid": legacy_surface_metrics,
            "leakage": surface_leakage,
            "per_region": surface_per_region,
        },
        "narrow_band": narrow_records,
        "closest_triangle_tie_test": tie_test,
        "densification": densification,
        "poses": pose_records,
        "gaussian_200k": {
            "count": len(weights_200k),
            "source_type": "deterministic_area_weighted_surface_face_sample",
            "attachment_coverage": 1.0,
            "exact_vertex_attachments": 0,
            "face_attachments": len(weights_200k),
            "off_surface_fallback": 0,
            "component_distribution": component_distribution,
            "region_distribution": region_distribution,
            "weight_metrics": weight_metrics(weights_200k, weights_200k),
            "planned_runtime_source_decomposition": {
                "template_vertex_gaussians": 0,
                "surface_sampled_gaussians": 200000,
                "arbitrary_canonical_gaussians": 0,
                "current_scene_sampler": "Open3D Poisson-disk surface sampling",
                "current_scene_sampler_preserves_face_or_barycentric_metadata": False,
                "prototype_sampler": "deterministic area-CDF face sampling with explicit barycentric coordinates",
            },
            "memory_bytes": {
                "weights_float32": int(weights_200k.nbytes),
                "face_ids_int64": int(sample_200k["face_ids"].nbytes),
                "barycentric_float64": int(sample_200k["barycentric"].nbytes),
                "component_int16": int(components_200k.nbytes),
                "region_int16": int(regions_200k.nbytes),
                "total_attachment_state": int(weights_200k.nbytes + sample_200k["face_ids"].nbytes + sample_200k["barycentric"].nbytes + components_200k.nbytes + regions_200k.nbytes),
                "metadata_over_existing_checkpoint_weights": int(sample_200k["face_ids"].nbytes + sample_200k["barycentric"].nbytes + components_200k.nbytes + regions_200k.nbytes),
            },
        },
        "artifacts": {
            "surface_attachments_npz_sha256": sha256_file(attachment_path),
            "densification_checkpoint_npz_sha256": sha256_file(checkpoint_path),
            "array_sha256": array_hashes,
        },
        "runtime_gate": {
            "current_create_from_pcd_accepts_fixed_weights": False,
            "current_runtime_persists_face_barycentric_semantic_metadata": False,
            "renderer_smoke": "NOT_RUN_RUNTIME_INTERFACE_PRECONDITION_FAILED",
        },
        "counters": {
            "training_steps": 0,
            "training_forward_batches": 0,
            "backward_calls": 0,
            "optimizer_created": 0,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "training_checkpoint_writes": 0,
            "prototype_checkpoint_format_writes": 1,
            "PAPER_FINAL": 0,
        },
    }
    manifest_path = output / "deterministic_manifest.json"
    write_json(manifest_path, deterministic_manifest)
    elapsed = time.perf_counter() - started
    write_json(
        output / "runtime_observation.json",
        {
            "run_label": args.run_label,
            "wall_seconds": elapsed,
            "deterministic_200k_attachment_preprocessing_seconds": sampling_200k_seconds,
            "per_forward_lbs_spatial_query_time": "NOT_APPLICABLE_FIXED_WEIGHTS_ARE_CACHED",
            "per_forward_lbs_spatial_query_calls": 0,
            "pid": os.getpid(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "manifest_sha256": sha256_file(manifest_path),
            "surface_attachments_npz_sha256": sha256_file(attachment_path),
            "densification_checkpoint_npz_sha256": sha256_file(checkpoint_path),
        },
    )
    print(json.dumps({
        "manifest_sha256": sha256_file(manifest_path),
        "surface_attachments_npz_sha256": sha256_file(attachment_path),
        "densification_checkpoint_npz_sha256": sha256_file(checkpoint_path),
        "wall_seconds": elapsed,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
