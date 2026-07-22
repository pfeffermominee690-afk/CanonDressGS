#!/usr/bin/env python3
"""Generate one isolated deterministic subject00 SMPL-X body-surface template."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import smplx
import torch
import trimesh
import yaml


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray, dtype: np.dtype) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype=dtype).tobytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def deterministic_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    lines.extend(" ".join(format(float(component), ".9g") for component in vertex) for vertex in vertices)
    lines.extend(f"3 {int(face[0])} {int(face[1])} {int(face[2])}" for face in faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def duplicate_row_count(array: np.ndarray) -> int:
    return int(len(array) - len(np.unique(np.ascontiguousarray(array), axis=0)))


def mesh_audit(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, validate=False)
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    face_areas = 0.5 * np.linalg.norm(cross, axis=1)
    sorted_faces = np.sort(faces, axis=1)
    edges = np.sort(
        np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0),
        axis=1,
    )
    _, edge_counts = np.unique(edges, axis=0, return_counts=True)
    components = trimesh.graph.connected_components(
        mesh.face_adjacency,
        nodes=np.arange(len(faces), dtype=np.int64),
        min_len=1,
    )
    normals = np.asarray(mesh.vertex_normals)
    return {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "vertices_dtype": str(vertices.dtype),
        "faces_dtype": str(faces.dtype),
        "vertices_finite": bool(np.isfinite(vertices).all()),
        "face_index_min": int(faces.min()),
        "face_index_max": int(faces.max()),
        "face_indices_valid": bool(faces.min() >= 0 and faces.max() < len(vertices)),
        "degenerate_faces_at_1e-12": int(np.count_nonzero(face_areas <= 1.0e-12)),
        "duplicate_vertices": duplicate_row_count(vertices),
        "duplicate_faces_unordered": duplicate_row_count(sorted_faces),
        "connected_components": int(len(components)),
        "boundary_edges": int(np.count_nonzero(edge_counts == 1)),
        "manifold_edges": int(np.count_nonzero(edge_counts == 2)),
        "nonmanifold_edges": int(np.count_nonzero(edge_counts > 2)),
        "watertight": bool(mesh.is_watertight),
        "normals_finite": bool(np.isfinite(normals).all()),
        "surface_area": float(face_areas.sum()),
        "bbox_min": vertices.min(axis=0).tolist(),
        "bbox_max": vertices.max(axis=0).tolist(),
        "bbox_extent": np.ptp(vertices, axis=0).tolist(),
        "centroid": vertices.mean(axis=0).tolist(),
    }


def audit_shape(payload: Any, tolerance: float) -> tuple[np.ndarray, dict[str, Any]]:
    if "betas" not in payload.files:
        raise KeyError("smpl_params.npz has no betas")
    betas = np.asarray(payload["betas"], dtype=np.float32)
    if betas.ndim == 1:
        vectors = betas[None]
    elif betas.ndim == 2:
        vectors = betas
    else:
        raise ValueError(f"Unexpected betas shape: {betas.shape}")
    beta0 = np.ascontiguousarray(vectors[0], dtype=np.float32)
    max_abs = float(np.max(np.abs(vectors - beta0[None]))) if len(vectors) > 1 else 0.0
    if not np.isfinite(vectors).all() or max_abs > tolerance:
        raise ValueError(f"Betas are non-finite or not constant: max_abs={max_abs}")
    audit = {
        "keys": sorted(payload.files),
        "arrays": {
            key: {"shape": list(payload[key].shape), "dtype": str(payload[key].dtype), "finite": bool(np.isfinite(payload[key]).all())}
            for key in sorted(payload.files)
        },
        "betas_shape": list(betas.shape),
        "stored_shape_vector_count": int(len(vectors)),
        "cross_frame_constant": bool(max_abs <= tolerance),
        "cross_frame_max_abs": max_abs,
        "cross_frame_tolerance": tolerance,
        "selection": "betas[0]",
        "shape_vector_sha256_float32": sha256_array(beta0, np.float32),
        "finite": bool(np.isfinite(vectors).all()),
    }
    return beta0, audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--subject-data", type=Path, required=True)
    parser.add_argument("--smpl-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", choices=("run_a", "run_b"), required=True)
    args = parser.parse_args()
    started = time.time()
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    expected_output_root = Path(protocol["publication"]["staging_root"]) / args.run_id / "template"
    if args.output_dir.resolve() != expected_output_root.resolve():
        raise ValueError(f"Output must equal frozen run root: {expected_output_root}")
    raw_root = Path(protocol["data"]["raw_root"]).resolve()
    try:
        args.output_dir.resolve().relative_to(raw_root)
    except ValueError:
        pass
    else:
        raise ValueError("Refusing to write beneath subject00 raw root")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if sha256_file(args.smpl_model) != protocol["body_model"]["model_sha256"]:
        raise ValueError("SMPL-X model hash mismatch")
    params_path = args.subject_data / protocol["data"]["smpl_params"]
    with np.load(params_path, allow_pickle=False) as payload:
        beta, shape_audit = audit_shape(payload, float(protocol["shape"]["frame_policy"].rsplit("_", 1)[-1]))
    if shape_audit["shape_vector_sha256_float32"] != protocol["shape"]["expected_sha256"]:
        raise ValueError("Subject00 shape-vector hash mismatch")

    canonical = protocol["canonical_transform"]
    global_orient = np.asarray(canonical["global_orientation_axis_angle"], dtype=np.float32)
    transl = np.asarray(canonical["translation"], dtype=np.float32)
    body_pose = np.zeros(int(canonical["body_pose_axis_angle_length"]), dtype=np.float32)
    for index, value in canonical["body_pose_nonzero"].items():
        body_pose[int(index)] = np.float32(value)
    canonical_vector = np.concatenate([global_orient, body_pose, transl]).astype(np.float32)
    canonical_hash = sha256_array(canonical_vector, np.float32)

    model_cfg = protocol["body_model"]
    model = smplx.SMPLX(
        model_path=str(args.smpl_model),
        use_pca=bool(model_cfg["use_pca"]),
        num_pca_comps=int(model_cfg["num_pca_comps"]),
        flat_hand_mean=bool(model_cfg["flat_hand_mean"]),
        batch_size=int(model_cfg["batch_size"]),
    )
    with torch.no_grad():
        output = model(
            betas=torch.from_numpy(beta[None]),
            global_orient=torch.from_numpy(global_orient[None]),
            body_pose=torch.from_numpy(body_pose[None]),
            transl=torch.from_numpy(transl[None]),
        )
    vertices = np.ascontiguousarray(output.vertices[0].cpu().numpy(), dtype=np.float32)
    faces = np.ascontiguousarray(model.faces, dtype=np.int64)
    root_joint = np.asarray(output.joints[0, 0].cpu().numpy(), dtype=np.float32)
    template_cfg = protocol["template"]
    if vertices.shape != (int(template_cfg["expected_vertices"]), 3):
        raise ValueError(f"Unexpected vertex shape: {vertices.shape}")
    if faces.shape != (int(template_cfg["expected_faces"]), 3):
        raise ValueError(f"Unexpected face shape: {faces.shape}")
    vertex_hash = sha256_array(vertices, np.float32)
    face_hash = sha256_array(faces, np.int64)
    if vertex_hash != template_cfg["expected_vertices_float32_sha256"]:
        raise ValueError(f"Unexpected vertex hash: {vertex_hash}")
    if face_hash != template_cfg["expected_faces_int64_sha256"]:
        raise ValueError(f"Unexpected face hash: {face_hash}")

    vertices_path = args.output_dir / template_cfg["output_names"]["vertices"]
    faces_path = args.output_dir / template_cfg["output_names"]["faces"]
    ply_path = args.output_dir / template_cfg["output_names"]["ply"]
    np.save(vertices_path, vertices, allow_pickle=False)
    np.save(faces_path, faces, allow_pickle=False)
    deterministic_ply(ply_path, vertices, faces)
    audit = mesh_audit(vertices, faces)
    thresholds = template_cfg["topology_thresholds"]
    gate = (
        audit["vertices_finite"]
        and audit["face_indices_valid"]
        and audit["degenerate_faces_at_1e-12"] <= int(thresholds["maximum_degenerate_faces"])
        and audit["duplicate_vertices"] <= int(thresholds["maximum_duplicate_vertices"])
        and audit["duplicate_faces_unordered"] <= int(thresholds["maximum_duplicate_faces"])
        and audit["normals_finite"]
    )
    report = {
        "task_id": protocol["task_id"],
        "run_id": args.run_id,
        "status": "PASS" if gate else "FAIL",
        "template_mode": template_cfg["mode"],
        "loose_clothing_template_available": False,
        "source_head": protocol["source"]["head"],
        "environment": {
            "hostname": socket.gethostname(),
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "smplx": getattr(smplx, "__version__", "UNKNOWN"),
            "trimesh": trimesh.__version__,
        },
        "inputs": {
            "subject_data": str(args.subject_data),
            "smpl_params": str(params_path),
            "smpl_params_sha256": sha256_file(params_path),
            "smpl_model": str(args.smpl_model),
            "smpl_model_sha256": sha256_file(args.smpl_model),
            "protocol": str(args.protocol),
            "protocol_sha256": sha256_file(args.protocol),
        },
        "shape_audit": shape_audit,
        "canonical": {
            "pose_vector_layout": "global_orient[3]+body_pose[63]+translation[3]",
            "pose_vector_sha256_float32": canonical_hash,
            "global_orientation": global_orient.tolist(),
            "translation": transl.tolist(),
            "body_pose_nonzero": {str(index): float(body_pose[index]) for index in np.flatnonzero(body_pose)},
            "root_joint": root_joint.tolist(),
            "coordinate_system": canonical["coordinate_system"],
            "units": canonical["units"],
        },
        "mesh_audit": audit,
        "outputs": {
            "vertices": str(vertices_path),
            "vertices_file_sha256": sha256_file(vertices_path),
            "vertices_array_sha256": vertex_hash,
            "faces": str(faces_path),
            "faces_file_sha256": sha256_file(faces_path),
            "faces_array_sha256": face_hash,
            "ply": str(ply_path),
            "ply_sha256": sha256_file(ply_path),
        },
        "counters": {
            "training_steps": 0,
            "forward_training_batches": 0,
            "backward_calls": 0,
            "optimizer_created": 0,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "checkpoint_writes": 0,
        },
        "wall_clock_seconds": time.time() - started,
    }
    write_json(args.output_dir / "template_generation_report.json", report)
    print(json.dumps({
        "run_id": args.run_id,
        "status": report["status"],
        "shape_sha256": shape_audit["shape_vector_sha256_float32"],
        "canonical_pose_sha256": canonical_hash,
        "vertices_sha256": vertex_hash,
        "faces_sha256": face_hash,
        "ply_sha256": report["outputs"]["ply_sha256"],
        "mesh_audit": audit,
    }, indent=2))
    return 0 if gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
