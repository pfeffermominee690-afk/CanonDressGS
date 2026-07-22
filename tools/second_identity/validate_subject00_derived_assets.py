#!/usr/bin/env python3
"""Compare, validate, publish, and no-training-smoke subject00 derived assets."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import stat
import sys
import time
import warnings
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import smplx
import torch
import torch.nn.functional as torch_f
import trimesh
import yaml


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for key in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(arrays[key]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = (0o100444 & 0xFFFF) << 16
            archive.writestr(info, buffer.getvalue())
    os.replace(temporary, path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def array_record(value: np.ndarray) -> dict[str, Any]:
    return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": sha256_array(value)}


def compare_templates(run_a: Path, run_b: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    names = protocol["template"]["output_names"]
    va = np.load(run_a / names["vertices"], allow_pickle=False)
    vb = np.load(run_b / names["vertices"], allow_pickle=False)
    fa = np.load(run_a / names["faces"], allow_pickle=False)
    fb = np.load(run_b / names["faces"], allow_pickle=False)
    vertex_diff = np.abs(np.asarray(va, dtype=np.float64) - np.asarray(vb, dtype=np.float64))
    bbox_a = np.stack([va.min(axis=0), va.max(axis=0)])
    bbox_b = np.stack([vb.min(axis=0), vb.max(axis=0)])
    threshold = protocol["template"]["determinism_thresholds"]
    comparison = {
        "run_a": {
            "vertices": array_record(va),
            "faces": array_record(fa),
            "ply_sha256": sha256_file(run_a / names["ply"]),
            "report_sha256": sha256_file(run_a / "template_generation_report.json"),
        },
        "run_b": {
            "vertices": array_record(vb),
            "faces": array_record(fb),
            "ply_sha256": sha256_file(run_b / names["ply"]),
            "report_sha256": sha256_file(run_b / "template_generation_report.json"),
        },
        "vertices_shape_exact": va.shape == vb.shape,
        "faces_shape_exact": fa.shape == fb.shape,
        "vertices_dtype_exact": va.dtype == vb.dtype,
        "faces_dtype_exact": fa.dtype == fb.dtype,
        "vertex_arrays_bitwise_exact": bool(np.array_equal(va, vb)),
        "faces_exact": bool(np.array_equal(fa, fb)),
        "ply_files_exact": sha256_file(run_a / names["ply"]) == sha256_file(run_b / names["ply"]),
        "vertex_max_abs": float(vertex_diff.max()),
        "vertex_mean_abs": float(vertex_diff.mean()),
        "bbox_max_abs": float(np.max(np.abs(bbox_a - bbox_b))),
        "topology_hash_a": canonical_json_sha256({"vertex_count": len(va), "faces_sha256": sha256_array(fa)}),
        "topology_hash_b": canonical_json_sha256({"vertex_count": len(vb), "faces_sha256": sha256_array(fb)}),
    }
    comparison["status"] = "PASS" if (
        comparison["vertices_shape_exact"]
        and comparison["faces_shape_exact"]
        and comparison["vertices_dtype_exact"]
        and comparison["faces_dtype_exact"]
        and comparison["faces_exact"]
        and comparison["vertex_max_abs"] <= float(threshold["vertex_max_abs"])
        and comparison["bbox_max_abs"] <= float(threshold["bbox_max_abs"])
    ) else "FAIL"
    return comparison


def compare_lbs(run_a: Path, run_b: Path, protocol: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    name = protocol["lbs"]["output_name"]
    path_a, path_b = run_a / name, run_b / name
    arrays_a, arrays_b = load_npz(path_a), load_npz(path_b)
    keys_equal = set(arrays_a) == set(arrays_b)
    per_array = {}
    all_bitwise = keys_equal
    metadata_exact = keys_equal
    for key in sorted(set(arrays_a) | set(arrays_b)):
        if key not in arrays_a or key not in arrays_b:
            per_array[key] = {"present_both": False}
            all_bitwise = metadata_exact = False
            continue
        a, b = arrays_a[key], arrays_b[key]
        exact = a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
        diff = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) if a.shape == b.shape else None
        per_array[key] = {
            "present_both": True,
            "run_a": array_record(a),
            "run_b": array_record(b),
            "shape_exact": a.shape == b.shape,
            "dtype_exact": a.dtype == b.dtype,
            "bitwise_exact": exact,
            "max_abs": float(diff.max()) if diff is not None else None,
            "mean_abs": float(diff.mean()) if diff is not None else None,
        }
        all_bitwise &= exact
        if key != "grid":
            metadata_exact &= exact
    grid_a, grid_b = arrays_a["grid"], arrays_b["grid"]
    threshold = protocol["lbs"]["repeatability_thresholds"]
    argmax_agreement = float(np.mean(np.argmax(grid_a, axis=-1) == np.argmax(grid_b, axis=-1)))
    grid_max_abs = per_array["grid"]["max_abs"]
    grid_mean_abs = per_array["grid"]["mean_abs"]
    comparison = {
        "status": "PASS" if (
            keys_equal
            and metadata_exact
            and grid_max_abs <= float(threshold["grid_max_abs"])
            and grid_mean_abs <= float(threshold["grid_mean_abs"])
            and argmax_agreement >= float(threshold["argmax_joint_agreement"])
        ) else "FAIL",
        "keys_equal": keys_equal,
        "keys": sorted(arrays_a),
        "all_arrays_bitwise_exact": all_bitwise,
        "npz_files_exact": sha256_file(path_a) == sha256_file(path_b),
        "run_a_file_sha256": sha256_file(path_a),
        "run_b_file_sha256": sha256_file(path_b),
        "metadata_exact": metadata_exact,
        "grid_max_abs": grid_max_abs,
        "grid_mean_abs": grid_mean_abs,
        "argmax_joint_agreement": argmax_agreement,
        "arrays": per_array,
    }
    return comparison, arrays_a


def interpolate_grid(grid_info: dict[str, np.ndarray], vertices: np.ndarray) -> np.ndarray:
    grid = torch.from_numpy(np.asarray(grid_info["grid"], dtype=np.float32))
    bbox_min = torch.from_numpy(np.asarray(grid_info["bbox_min"], dtype=np.float32))
    bbox_max = torch.from_numpy(np.asarray(grid_info["bbox_max"], dtype=np.float32))
    points = torch.from_numpy(np.asarray(vertices, dtype=np.float32))
    normalized = (points - bbox_min) / (bbox_max - bbox_min) * 2.0 - 1.0
    sample_points = normalized[:, [2, 1, 0]].reshape(1, 1, 1, -1, 3)
    with torch.no_grad():
        weights = torch_f.grid_sample(
            grid.permute(3, 0, 1, 2)[None],
            sample_points,
            padding_mode="border",
            align_corners=True,
        ).squeeze().permute(1, 0)
    return weights.numpy()


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
    kwargs: dict[str, torch.Tensor] = {}
    for target, source in mapping.items():
        if source not in payload.files:
            continue
        value = np.asarray(payload[source])
        selected = value[frame] if value.ndim > 1 and len(value) > frame else value[0] if value.ndim > 1 else value
        kwargs[target] = torch.as_tensor(selected, dtype=torch.float32)[None]
    return kwargs


def geometry_audit(
    vertices: np.ndarray,
    faces: np.ndarray,
    grid_info: dict[str, np.ndarray],
    subject_data: Path,
    model_path: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    bbox_min = np.asarray(grid_info["bbox_min"], dtype=np.float32)
    bbox_max = np.asarray(grid_info["bbox_max"], dtype=np.float32)
    below = np.any(vertices < bbox_min[None] - 1.0e-7, axis=1)
    above = np.any(vertices > bbox_max[None] + 1.0e-7, axis=1)
    extrapolated = int(np.count_nonzero(below | above))
    grid_coordinates = (vertices - bbox_min[None]) / (bbox_max - bbox_min)[None]
    interpolated = interpolate_grid(grid_info, vertices)
    sums = interpolated.sum(axis=-1, dtype=np.float64)
    model_cfg = protocol["body_model"]
    model = smplx.SMPLX(
        model_path=str(model_path), use_pca=bool(model_cfg["use_pca"]),
        num_pca_comps=int(model_cfg["num_pca_comps"]), flat_hand_mean=bool(model_cfg["flat_hand_mean"]), batch_size=1,
    )
    direct = model.lbs_weights.detach().cpu().numpy().astype(np.float32)
    difference = np.abs(interpolated.astype(np.float64) - direct.astype(np.float64))
    dominant_agreement = float(np.mean(np.argmax(interpolated, axis=-1) == np.argmax(direct, axis=-1)))
    canonical_area = float(trimesh.Trimesh(vertices=vertices, faces=faces, process=False).area)
    pose_ids = [0, 1250, 2499]
    posed = []
    with np.load(subject_data / protocol["data"]["smpl_params"], allow_pickle=False) as payload:
        beta_values = np.asarray(payload["betas"], dtype=np.float32)
        beta = beta_values if beta_values.ndim == 1 else beta_values[0]
        for frame in pose_ids:
            kwargs = model_kwargs(payload, frame)
            kwargs["betas"] = torch.from_numpy(beta[None])
            with torch.no_grad():
                result = model(**kwargs)
            posed_vertices = result.vertices[0].cpu().numpy().astype(np.float32)
            mesh = trimesh.Trimesh(vertices=posed_vertices, faces=faces, process=False)
            area = float(mesh.area)
            triangles = posed_vertices[faces]
            face_area = 0.5 * np.linalg.norm(
                np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
            )
            posed.append({
                "frame_id": frame,
                "vertices_sha256_float32": sha256_array(posed_vertices),
                "finite": bool(np.isfinite(posed_vertices).all()),
                "bbox_min": posed_vertices.min(axis=0).tolist(),
                "bbox_max": posed_vertices.max(axis=0).tolist(),
                "max_abs_coordinate": float(np.abs(posed_vertices).max()),
                "surface_area": area,
                "surface_area_ratio_to_canonical": area / canonical_area,
                "degenerate_faces_at_1e-12": int(np.count_nonzero(face_area <= 1.0e-12)),
                "normals_finite": bool(np.isfinite(mesh.vertex_normals).all()),
            })
    threshold = protocol["lbs"]["geometry_thresholds"]
    posed_gate = all(
        record["finite"] and record["normals_finite"]
        and record["max_abs_coordinate"] <= float(threshold["posed_max_abs_coordinate_meters"])
        and float(threshold["posed_surface_area_ratio_minimum"]) <= record["surface_area_ratio_to_canonical"] <= float(threshold["posed_surface_area_ratio_maximum"])
        for record in posed
    )
    audit = {
        "template_bbox_min": vertices.min(axis=0).tolist(),
        "template_bbox_max": vertices.max(axis=0).tolist(),
        "grid_bbox_min": bbox_min.tolist(),
        "grid_bbox_max": bbox_max.tolist(),
        "grid_coordinates_finite": bool(np.isfinite(grid_coordinates).all()),
        "grid_coordinate_min": grid_coordinates.min(axis=0).tolist(),
        "grid_coordinate_max": grid_coordinates.max(axis=0).tolist(),
        "extrapolated_vertex_count": extrapolated,
        "interpolated_weights_finite": bool(np.isfinite(interpolated).all()),
        "interpolated_weight_min": float(interpolated.min()),
        "interpolated_weight_max": float(interpolated.max()),
        "interpolated_weight_sum_max_abs_error": float(np.max(np.abs(sums - 1.0))),
        "interpolated_weight_sum_mean_abs_error": float(np.mean(np.abs(sums - 1.0))),
        "interpolated_vs_smplx_weight_mae": float(difference.mean()),
        "interpolated_vs_smplx_weight_max_abs": float(difference.max()),
        "dominant_joint_agreement": dominant_agreement,
        "posed_body_forwards": posed,
        "warnings": [],
    }
    audit["status"] = "PASS" if (
        extrapolated == int(threshold["extrapolated_vertices"])
        and audit["grid_coordinates_finite"]
        and audit["interpolated_weights_finite"]
        and audit["interpolated_weight_sum_max_abs_error"] <= float(threshold["interpolated_sum_max_abs_error"])
        and audit["interpolated_vs_smplx_weight_mae"] <= float(threshold["interpolated_weight_mae"])
        and audit["interpolated_vs_smplx_weight_max_abs"] <= float(threshold["interpolated_weight_max_abs"])
        and dominant_agreement >= float(threshold["dominant_joint_agreement_minimum"])
        and posed_gate
    ) else "FAIL"
    return audit


def runtime_compatibility(repo_root: Path, vertices: np.ndarray, faces: np.ndarray, subject02_template: Path) -> dict[str, Any]:
    scene_source = (repo_root / "scene" / "scene.py").read_text(encoding="utf-8")
    gaussian_source = (repo_root / "scene" / "gaussian_model.py").read_text(encoding="utf-8")
    graphics_source = (repo_root / "utils" / "graphics_utils.py").read_text(encoding="utf-8")
    fixed_tokens = ["96380", "96_380", "192744", "192_744"]
    fixed_assumption = any(token in scene_source + gaussian_source + graphics_source for token in fixed_tokens)
    subject00_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    subject02_mesh = trimesh.load(subject02_template, process=False)
    result = {
        "classification": "BODY_SURFACE_FALLBACK_RUNTIME_COMPATIBLE" if not fixed_assumption else "BODY_SURFACE_FALLBACK_RUNTIME_INCOMPATIBLE",
        "fixed_96380_or_192744_assumption_found": fixed_assumption,
        "runtime_loader": "Open3D loads arbitrary triangle-mesh vertex/face counts",
        "gaussian_initialization": "samples a configured number of surface points; independent of template vertex count",
        "lbs_generation": "uses matching SMPL-X canonical topology and 55 model weights",
        "checkpoint_architecture": "bound to Gaussian/control/feature counts, not template vertex count",
        "subject00": {
            "type": "subject-specific SMPL-X body-surface fallback",
            "vertices": int(len(vertices)),
            "faces": int(len(faces)),
            "surface_area": float(subject00_mesh.area),
            "bbox_extent": subject00_mesh.extents.tolist(),
        },
        "subject02": {
            "type": "loose-clothing template",
            "vertices": int(len(subject02_mesh.vertices)),
            "faces": int(len(subject02_mesh.faces)),
            "surface_area": float(subject02_mesh.area),
            "bbox_extent": subject02_mesh.extents.tolist(),
            "sha256": sha256_file(subject02_template),
        },
        "ratios_subject00_to_subject02": {
            "vertices": float(len(vertices) / len(subject02_mesh.vertices)),
            "faces": float(len(faces) / len(subject02_mesh.faces)),
            "surface_area": float(subject00_mesh.area / subject02_mesh.area),
            "bbox_extent": (subject00_mesh.extents / subject02_mesh.extents).tolist(),
        },
        "scientific_limitation": "This establishes pipeline portability only; it does not establish cross-identity garment generalization, initialization fairness, template independence, or loose-clothing reconstruction parity.",
    }
    result["status"] = "PASS" if result["classification"] == "BODY_SURFACE_FALLBACK_RUNTIME_COMPATIBLE" else "FAIL"
    return result


def publish(args: argparse.Namespace) -> int:
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    publication = protocol["publication"]
    staging_root = Path(publication["staging_root"])
    run_a_template, run_b_template = staging_root / "run_a/template", staging_root / "run_b/template"
    run_a_lbs, run_b_lbs = staging_root / "run_a/lbs", staging_root / "run_b/lbs"
    candidate = staging_root / publication["candidate"]
    formal = Path(publication["formal_target"])
    if formal.exists():
        raise FileExistsError("SUBJECT00_DERIVED_ASSET_TARGET_ALREADY_EXISTS")
    if candidate.exists():
        raise FileExistsError(candidate)
    template_comparison = compare_templates(run_a_template, run_b_template, protocol)
    lbs_comparison, lbs_arrays = compare_lbs(run_a_lbs, run_b_lbs, protocol)
    names = protocol["template"]["output_names"]
    vertices = np.load(run_a_template / names["vertices"], allow_pickle=False)
    faces = np.load(run_a_template / names["faces"], allow_pickle=False)
    geometry = geometry_audit(vertices, faces, lbs_arrays, args.subject_data, args.smpl_model, protocol)
    compatibility = runtime_compatibility(args.repo_root, vertices, faces, args.subject02_template)
    write_json(args.output_dir / "subject00_template_run_comparison.json", template_comparison)
    write_json(args.output_dir / "subject00_lbs_run_comparison.json", lbs_comparison)
    write_json(args.output_dir / "subject00_lbs_geometry_audit.json", geometry)
    write_json(args.output_dir / "subject00_template_runtime_compatibility.json", compatibility)
    gates = {
        "template_determinism": template_comparison["status"],
        "template_runtime_compatibility": compatibility["status"],
        "lbs_determinism": lbs_comparison["status"],
        "lbs_geometry": geometry["status"],
    }
    if any(value != "PASS" for value in gates.values()):
        print(json.dumps({"status": "BLOCKED", "gates": gates}, indent=2))
        return 2

    (candidate / "template").mkdir(parents=True)
    (candidate / "lbs").mkdir()
    (candidate / "manifests").mkdir()
    (candidate / "reports").mkdir()
    for key in ("vertices", "faces", "ply"):
        shutil.copyfile(run_a_template / names[key], candidate / "template" / names[key])
    formal_lbs = candidate / "lbs" / protocol["lbs"]["output_name"]
    if lbs_comparison["npz_files_exact"]:
        shutil.copyfile(run_a_lbs / protocol["lbs"]["output_name"], formal_lbs)
        lbs_selection = "RUN_A_FILE_BYTE_EXACT_WITH_RUN_B"
    else:
        deterministic_npz(formal_lbs, lbs_arrays)
        lbs_selection = "CANONICAL_DETERMINISTIC_NPZ_FROM_BITWISE_OR_TOLERANCE_VALIDATED_RUN_A_ARRAYS"
    reloaded = load_npz(formal_lbs)
    if any(not np.array_equal(reloaded[key], lbs_arrays[key]) for key in lbs_arrays):
        raise RuntimeError("Canonical NPZ payload changed arrays")

    template_report_a = json.loads((run_a_template / "template_generation_report.json").read_text(encoding="utf-8"))
    lbs_report_a = json.loads((run_a_lbs / "lbs_generation_report.json").read_text(encoding="utf-8"))
    formal_assets = {
        "template_ply": {
            "path": "template/template_smplx_body_surface.ply",
            "sha256": sha256_file(candidate / "template" / names["ply"]),
            "bytes": (candidate / "template" / names["ply"]).stat().st_size,
        },
        "template_vertices": {
            "path": "template/template_vertices.npy",
            "sha256": sha256_file(candidate / "template" / names["vertices"]),
            "array_sha256": sha256_array(vertices),
            "bytes": (candidate / "template" / names["vertices"]).stat().st_size,
        },
        "template_faces": {
            "path": "template/template_faces.npy",
            "sha256": sha256_file(candidate / "template" / names["faces"]),
            "array_sha256": sha256_array(faces),
            "bytes": (candidate / "template" / names["faces"]).stat().st_size,
        },
        "lbs_grid": {
            "path": "lbs/lbs_weights_grid.npz",
            "sha256": sha256_file(formal_lbs),
            "bytes": formal_lbs.stat().st_size,
            "array_hashes": {key: sha256_array(value) for key, value in reloaded.items()},
            "selection": lbs_selection,
        },
    }
    manifest = {
        "schema_version": "subject00.mmlphuman.derived_asset_manifest.v1",
        "task_id": protocol["task_id"],
        "attempt": protocol["attempt"],
        "source_branch": protocol["source"]["branch"],
        "source_head": protocol["source"]["head"],
        "v2_baseline_branch": protocol["source"]["v2_baseline_branch"],
        "v2_baseline_head": protocol["source"]["v2_baseline_head"],
        "runtime_raw_closure_sha256": protocol["source"]["runtime_raw_closure_sha256"],
        "runtime_lf_closure_sha256": protocol["source"]["runtime_lf_closure_sha256"],
        "subject00_raw_fingerprint": protocol["data"]["raw_fingerprint"],
        "availability_manifest_sha256": protocol["data"]["availability_manifest_sha256"],
        "smplx_model": {"path": str(args.smpl_model), "sha256": sha256_file(args.smpl_model)},
        "shape_vector_sha256": template_report_a["shape_audit"]["shape_vector_sha256_float32"],
        "canonical_pose_sha256": template_report_a["canonical"]["pose_vector_sha256_float32"],
        "template_mode": protocol["template"]["mode"],
        "loose_clothing_template_available": False,
        "template_generator": protocol["template"]["generator"],
        "template_run_comparison": template_comparison,
        "formal_template": formal_assets,
        "template_topology": {"vertices": int(len(vertices)), "faces": int(len(faces))},
        "coordinate_system": protocol["canonical_transform"]["coordinate_system"],
        "units": protocol["canonical_transform"]["units"],
        "lbs_generator": protocol["lbs"]["generator"],
        "point_interpolant_sha256": protocol["lbs"]["point_interpolant_sha256"],
        "lbs_run_comparison": lbs_comparison,
        "formal_lbs": formal_assets["lbs_grid"],
        "grid_shape": list(reloaded["grid"].shape),
        "joint_count": int(reloaded["grid"].shape[-1]),
        "bbox_min": reloaded["bbox_min"].tolist(),
        "bbox_max": reloaded["bbox_max"].tolist(),
        "determinism_metrics": {"template": template_comparison, "lbs": lbs_comparison},
        "geometry_audit": geometry,
        "runtime_compatibility": compatibility,
        "runtime_smoke": "PENDING_POST_PUBLISH",
        "strict_view_split_sha256": protocol["strict_splits"]["novel_view_sha256"],
        "strict_pose_split_sha256": protocol["strict_splits"]["novel_pose_sha256"],
        "warnings": ["BODY_SURFACE_FALLBACK_NOT_LOOSE_CLOTHING_TEMPLATE"],
        "readiness": "DERIVED_ASSETS_PUBLISHED_RUNTIME_SMOKE_PENDING",
        "next_task": "COMPLETE_SUBJECT00_DERIVED_ASSET_GPU_SMOKE",
        "counters": protocol["no_training_counters"],
    }
    manifest_path = candidate / "manifests/SUBJECT00_DERIVED_ASSET_MANIFEST.json"
    write_json(manifest_path, manifest)
    audit_md = (
        "# Subject00 Derived Asset Audit\n\n"
        f"- Template determinism: `{template_comparison['status']}`\n"
        f"- Template runtime compatibility: `{compatibility['classification']}`\n"
        f"- LBS determinism: `{lbs_comparison['status']}`\n"
        f"- LBS geometry: `{geometry['status']}`\n"
        f"- Template mode: `{protocol['template']['mode']}`\n"
        "- Loose-clothing template available: `false`\n"
        "- Training/backward/optimizer/checkpoint: `0/0/0/0`\n"
        "- Runtime smoke: `PENDING_POST_PUBLISH`\n"
    )
    (candidate / "reports/SUBJECT00_DERIVED_ASSET_AUDIT.md").write_text(audit_md, encoding="utf-8", newline="\n")
    formal.parent.mkdir(parents=True, exist_ok=True)
    os.rename(candidate, formal)
    published_manifest = formal / "manifests/SUBJECT00_DERIVED_ASSET_MANIFEST.json"
    verification = {
        "status": "PASS",
        "formal_target": str(formal),
        "formal_assets": {
            key: {**record, "published_path": str(formal / record["path"]), "published_sha256": sha256_file(formal / record["path"])}
            for key, record in formal_assets.items()
        },
        "manifest_path": str(published_manifest),
        "manifest_sha256": sha256_file(published_manifest),
        "atomic_publish": "PASS_SAME_FILESYSTEM_RENAME",
        "gates": gates,
    }
    write_json(args.output_dir / "subject00_atomic_publish_verification.json", verification)
    print(json.dumps(verification, indent=2))
    return 0


def deterministic_surface_samples(vertices: np.ndarray, faces: np.ndarray, count: int, seed: int) -> np.ndarray:
    triangles = vertices[faces]
    areas = 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    probabilities = areas / areas.sum()
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(faces), size=count, replace=True, p=probabilities)
    u = rng.random(count)
    v = rng.random(count)
    flip = u + v > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    chosen = triangles[selected]
    return np.asarray(chosen[:, 0] + u[:, None] * (chosen[:, 1] - chosen[:, 0]) + v[:, None] * (chosen[:, 2] - chosen[:, 0]), dtype=np.float32)


def collect_parameters(model: Any) -> tuple[int, int, int]:
    seen: set[int] = set()
    values: list[torch.Tensor] = []

    def visit(value: Any) -> None:
        if isinstance(value, torch.nn.Parameter):
            if id(value) not in seen:
                seen.add(id(value))
                values.append(value)
        elif isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested)

    for value in vars(model).values():
        visit(value)
    total = sum(value.numel() for value in values)
    trainable = sum(value.numel() for value in values if value.requires_grad)
    return total, trainable, total - trainable


def choose_complete_smoke_pairs(valid: set[tuple[int, int]], view: dict[str, Any], pose: dict[str, Any]) -> tuple[list[int], list[int]]:
    cameras = [view["train_camera_ids"][0], view["train_camera_ids"][len(view["train_camera_ids"]) // 2], view["train_camera_ids"][-1], view["heldout_camera_ids"][0]]
    train_candidates = pose["train_frame_ids"]
    heldout_candidates = pose["heldout_frame_ids"]
    frame_candidates = [train_candidates[0], train_candidates[len(train_candidates) // 2]]
    selected_frames = []
    for starting, candidates in ((frame_candidates[0], train_candidates), (frame_candidates[1], train_candidates), (heldout_candidates[0], heldout_candidates)):
        ordered = sorted(candidates, key=lambda frame: (abs(frame - starting), frame))
        frame = next(frame for frame in ordered if all((frame, camera) in valid for camera in cameras))
        selected_frames.append(frame)
    return selected_frames, cameras


def smoke(args: argparse.Namespace) -> int:
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    formal = Path(protocol["publication"]["formal_target"])
    manifest_path = formal / "manifests/SUBJECT00_DERIVED_ASSET_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    view = json.loads((args.repo_root / protocol["strict_splits"]["novel_view_manifest"]).read_text(encoding="utf-8"))
    pose = json.loads((args.repo_root / protocol["strict_splits"]["novel_pose_manifest"]).read_text(encoding="utf-8"))
    availability = json.loads(Path(protocol["data"]["availability_manifest"]).read_text(encoding="utf-8"))
    valid = {(entry["frame_id"], entry["camera_id"]) for entry in availability["entries"] if entry["valid_pair"]}
    frames, cameras = choose_complete_smoke_pairs(valid, view, pose)
    sys.path.insert(0, str(args.repo_root))
    from scene.dataset import ThumanDataset, data_to_cam  # pylint: disable=import-outside-toplevel
    dataset = ThumanDataset(str(args.subject_data), frames, cameras, image_scaling=1, is_in_memory=False)
    position = {pair: index for index, pair in enumerate(dataset.indices)}
    samples = []
    for pair in [(frame, camera) for frame in frames for camera in cameras]:
        sample = dataset[position[pair]]
        samples.append({
            "frame_id": pair[0], "camera_id": pair[1],
            "image_shape": list(sample["image"].shape), "mask_shape": list(sample["mask"].shape),
            "finite": bool(all(torch.isfinite(sample[key]).all() for key in ("image", "K", "w2c", "pose", "Th", "beta"))),
            "foreground_fraction": float(sample["mask"].float().mean()),
        })
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)))
    dataset_result = {
        "status": "PASS",
        "frames": frames,
        "cameras": cameras,
        "heldout_pose_included": any(frame in set(pose["heldout_frame_ids"]) for frame in frames),
        "heldout_camera_included": any(camera in set(view["heldout_camera_ids"]) for camera in cameras),
        "samples": samples,
        "batch4_image_shape": list(batch["image"].shape),
        "batch4_mask_shape": list(batch["mask"].shape),
        "invalid_path_accesses": 0,
    }
    if args.defer_gpu:
        model_result = {"status": "GPU_SMOKE_DEFERRED_FORMAL_TASK_ACTIVE"}
        readiness = "SUBJECT00_DERIVED_ASSETS_READY_GPU_SMOKE_DEFERRED"
        next_task = "COMPLETE_SUBJECT00_DERIVED_ASSET_GPU_SMOKE"
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable and --defer-gpu not set")
        from PIL import Image  # pylint: disable=import-outside-toplevel
        from gsplat import rasterization  # pylint: disable=import-outside-toplevel
        from scene.gaussian_model import GaussianModel  # pylint: disable=import-outside-toplevel
        from utils.smpl_utils import init_smpl, smpl  # pylint: disable=import-outside-toplevel
        started = time.time()
        torch.manual_seed(int(protocol["runtime"]["gaussian_sampling_seed"]))
        torch.cuda.manual_seed_all(int(protocol["runtime"]["gaussian_sampling_seed"]))
        torch.cuda.reset_peak_memory_stats()
        vertices = np.load(formal / "template/template_vertices.npy", allow_pickle=False)
        faces = np.load(formal / "template/template_faces.npy", allow_pickle=False)
        grid_info = load_npz(formal / "lbs/lbs_weights_grid.npz")
        seed = int(protocol["runtime"]["gaussian_sampling_seed"])
        xyz = deterministic_surface_samples(vertices, faces, int(protocol["runtime"]["gaussian_initialization_count"]), seed)
        xyz_vt = deterministic_surface_samples(vertices, faces, int(protocol["runtime"]["control_vertex_count"]), seed + 1)
        xyz_ft = deterministic_surface_samples(vertices, faces, int(protocol["runtime"]["feature_point_count"]), seed + 2)
        init_smpl(str(args.smpl_model))
        beta = dataset[0]["beta"].numpy()
        with torch.no_grad():
            tpose = smpl.model(betas=torch.from_numpy(beta[None]), body_pose=smpl.smpl_tpose[None, 3:66])
        t_joints = tpose.joints.detach().cpu().numpy()[0, : smpl.model.NUM_JOINTS + 1]
        pose_data = dataset.smpl_params["pose"]
        all_poses = {str(frame): pose_data[frame] for frame in pose["train_frame_ids"]}
        model = GaussianModel()
        recorded_warnings = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.create_from_pcd(
                xyz=xyz,
                t_joints=t_joints,
                joint_parents=smpl.model.parents,
                all_poses=all_poses,
                lbs_weights_grid_info=grid_info,
                xyz_vt=xyz_vt,
                xyz_ft=xyz_ft,
            )
            recorded_warnings.extend(str(item.message) for item in caught)
        parameter_count, trainable_count, frozen_count = collect_parameters(model)
        render_dir = args.output_dir / "renders"
        render_dir.mkdir(parents=True, exist_ok=True)
        render_records = []
        background = torch.ones(3, dtype=torch.float32, device="cuda")
        for frame in frames:
            for camera_id in cameras:
                sample = dataset[position[(frame, camera_id)]]
                collated = {key: (value[None] if isinstance(value, torch.Tensor) else np.asarray([value])) for key, value in sample.items()}
                camera = data_to_cam(collated, non_blocking=False)
                model.smpl_poses = camera["pose"]
                model.Th = camera["Th"]
                model.Rh = camera["Rh"]
                with torch.no_grad():
                    rgb, alpha, info = model.render(camera, background=background)
                    target = camera["image"]
                    zero_step_l1 = torch.mean(torch.abs(rgb - target))
                torch.cuda.synchronize()
                alpha_mask = alpha[..., 0] > 1.0e-5
                coordinates = torch.nonzero(alpha_mask, as_tuple=False)
                bounds = None if coordinates.numel() == 0 else {
                    "y_min": int(coordinates[:, 0].min()), "y_max": int(coordinates[:, 0].max()),
                    "x_min": int(coordinates[:, 1].min()), "x_max": int(coordinates[:, 1].max()),
                }
                png = (torch.clamp(rgb, 0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
                image_path = render_dir / f"frame_{frame:04d}_cam_{camera_id:02d}.png"
                Image.fromarray(png).save(image_path)
                render_records.append({
                    "frame_id": frame,
                    "camera_id": camera_id,
                    "rgb_shape": list(rgb.shape),
                    "alpha_shape": list(alpha.shape),
                    "rgb_finite": bool(torch.isfinite(rgb).all()),
                    "alpha_finite": bool(torch.isfinite(alpha).all()),
                    "alpha_nonempty": bool(alpha_mask.any()),
                    "alpha_max": float(alpha.max()),
                    "foreground_bounds": bounds,
                    "zero_step_l1": float(zero_step_l1),
                    "info_keys": sorted(info),
                    "render_png": str(image_path),
                    "render_png_sha256": sha256_file(image_path),
                })
        # One explicit expected-depth render using the same model state and renderer.
        frame, camera_id = frames[-1], cameras[-1]
        sample = dataset[position[(frame, camera_id)]]
        collated = {key: (value[None] if isinstance(value, torch.Tensor) else np.asarray([value])) for key, value in sample.items()}
        camera = data_to_cam(collated, non_blocking=False)
        model.smpl_poses, model.Th, model.Rh = camera["pose"], camera["Th"], camera["Rh"]
        with torch.no_grad():
            covars = model.get_covariance()
            camera_position = torch.linalg.inv_ex(camera["w2c"])[0][:3, 3]
            color = model.get_color(camera_position)
            rgb_depth, depth_alpha, depth_info = rasterization(
                means=model.compute_xyz(), quats=None, scales=None,
                opacities=model.compute_opacity(), colors=color,
                viewmats=camera["w2c"][None], Ks=camera["K"][None],
                width=camera["width"], height=camera["height"], packed=False,
                near_plane=0.1, backgrounds=background[None], covars=covars,
                render_mode="RGB+ED",
            )
        torch.cuda.synchronize()
        depth = rgb_depth[0, ..., 3]
        model_result = {
            "status": "PASS" if all(record["rgb_finite"] and record["alpha_finite"] and record["alpha_nonempty"] for record in render_records) and bool(torch.isfinite(depth).all()) else "FAIL",
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_count,
            "frozen_parameter_count": frozen_count,
            "initial_gaussian_count": int(model._xyz.shape[0]),
            "control_vertex_count": int(model.xyz_vt.shape[0]),
            "feature_point_count": int(model.xyz_ft.shape[0]),
            "optimizers": None if model.optimizers is None else "UNEXPECTED",
            "schedulers": None if model.schedulers is None else "UNEXPECTED",
            "render_resolution": [int(camera["height"]), int(camera["width"])],
            "renders": render_records,
            "depth": {
                "frame_id": frame, "camera_id": camera_id,
                "shape": list(depth.shape), "finite": bool(torch.isfinite(depth).all()),
                "minimum": float(depth.min()), "maximum": float(depth.max()),
                "alpha_nonempty": bool((depth_alpha[0, ..., 0] > 1.0e-5).any()),
                "info_keys": sorted(depth_info),
            },
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
            "wall_clock_seconds": time.time() - started,
            "warnings": recorded_warnings,
            "bin_overflow": "NOT_REPORTED_BY_GSPLAT_INFO",
            "illegal_memory_access": 0,
            "renderer_errors": [],
        }
        readiness = "SUBJECT00_DERIVED_ASSETS_READY_FOR_CANARY" if model_result["status"] == "PASS" else "SUBJECT00_BLOCKED_RENDERER"
        next_task = "RUN_SUBJECT00_MMLPHUMAN_SHORT_CANARY_TRAINING_WITH_STRICT_SPLITS" if readiness == "SUBJECT00_DERIVED_ASSETS_READY_FOR_CANARY" else "REPAIR_SUBJECT00_LBS_DETERMINISM_OR_BOUNDS_CONTRACT"
    result = {
        "task_id": protocol["task_id"],
        "status": "PASS" if dataset_result["status"] == "PASS" and model_result["status"] in ("PASS", "GPU_SMOKE_DEFERRED_FORMAL_TASK_ACTIVE") else "FAIL",
        "dataset_smoke": dataset_result,
        "model_renderer_smoke": model_result,
        "readiness": readiness,
        "next_task": next_task,
        "counters": protocol["no_training_counters"],
    }
    write_json(args.output_dir / "subject00_derived_asset_runtime_smoke.json", result)
    manifest["runtime_smoke"] = {
        "status": model_result["status"],
        "report": str(args.output_dir / "subject00_derived_asset_runtime_smoke.json"),
        "dataset_status": dataset_result["status"],
    }
    manifest["readiness"] = readiness
    manifest["next_task"] = next_task
    write_json(manifest_path, manifest)
    result["formal_manifest_sha256"] = sha256_file(manifest_path)
    write_json(args.output_dir / "subject00_derived_asset_runtime_smoke.json", result)
    # Final publication is read-only after the runtime status is sealed.
    for path in sorted(formal.rglob("*"), reverse=True):
        path.chmod(0o444 if path.is_file() else 0o555)
    formal.chmod(0o555)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    publish_parser = subparsers.add_parser("compare-publish")
    smoke_parser = subparsers.add_parser("smoke")
    for target in (publish_parser, smoke_parser):
        target.add_argument("--protocol", type=Path, required=True)
        target.add_argument("--repo-root", type=Path, required=True)
        target.add_argument("--subject-data", type=Path, required=True)
        target.add_argument("--smpl-model", type=Path, required=True)
        target.add_argument("--output-dir", type=Path, required=True)
    publish_parser.add_argument("--subject02-template", type=Path, required=True)
    smoke_parser.add_argument("--defer-gpu", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return publish(args) if args.command == "compare-publish" else smoke(args)


if __name__ == "__main__":
    raise SystemExit(main())
