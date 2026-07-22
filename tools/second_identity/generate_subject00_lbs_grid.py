#!/usr/bin/env python3
"""Run the frozen subject00 LBS generator in one isolated staging directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import smplx
import torch
import yaml


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_template_and_verify(
    template_dir: Path,
    subject_data: Path,
    model_path: Path,
    protocol: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.load(template_dir / protocol["template"]["output_names"]["vertices"], allow_pickle=False)
    faces = np.load(template_dir / protocol["template"]["output_names"]["faces"], allow_pickle=False)
    with np.load(subject_data / protocol["data"]["smpl_params"], allow_pickle=False) as payload:
        beta = np.asarray(payload["betas"], dtype=np.float32)
        beta = beta if beta.ndim == 1 else beta[0]
    body_pose = np.zeros(63, dtype=np.float32)
    for index, value in protocol["canonical_transform"]["body_pose_nonzero"].items():
        body_pose[int(index)] = np.float32(value)
    model_cfg = protocol["body_model"]
    model = smplx.SMPLX(
        model_path=str(model_path),
        use_pca=bool(model_cfg["use_pca"]),
        num_pca_comps=int(model_cfg["num_pca_comps"]),
        flat_hand_mean=bool(model_cfg["flat_hand_mean"]),
        batch_size=1,
    )
    with torch.no_grad():
        canonical = model(
            betas=torch.from_numpy(beta[None]),
            global_orient=torch.zeros((1, 3), dtype=torch.float32),
            body_pose=torch.from_numpy(body_pose[None]),
            transl=torch.zeros((1, 3), dtype=torch.float32),
        )
    expected_vertices = np.asarray(canonical.vertices[0].cpu().numpy(), dtype=np.float32)
    expected_faces = np.asarray(model.faces, dtype=np.int64)
    max_abs = float(np.max(np.abs(np.asarray(vertices, dtype=np.float32) - expected_vertices)))
    if max_abs > 1.0e-7 or not np.array_equal(np.asarray(faces, dtype=np.int64), expected_faces):
        raise ValueError(f"Template is not the frozen subject-specific SMPL-X canonical mesh: max_abs={max_abs}")
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def grid_audit(path: Path, expected_shape: tuple[int, int, int, int]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    required = {"grid", "bbox_min", "bbox_max", "grid_dims", "grid_resolution"}
    if set(arrays) != required:
        raise ValueError(f"Unexpected NPZ keys: {sorted(arrays)}")
    grid = arrays["grid"]
    sums = grid.sum(axis=-1, dtype=np.float64)
    finite = np.isfinite(grid)
    dominant = np.argmax(grid, axis=-1)
    boundary = np.concatenate(
        [grid[0].reshape(-1, grid.shape[-1]), grid[-1].reshape(-1, grid.shape[-1]),
         grid[:, 0].reshape(-1, grid.shape[-1]), grid[:, -1].reshape(-1, grid.shape[-1]),
         grid[:, :, 0].reshape(-1, grid.shape[-1]), grid[:, :, -1].reshape(-1, grid.shape[-1])],
        axis=0,
    )
    audit = {
        "keys": sorted(arrays),
        "arrays": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": sha256_array(value)}
            for key, value in arrays.items()
        },
        "shape": list(grid.shape),
        "shape_exact": tuple(grid.shape) == expected_shape,
        "dtype": str(grid.dtype),
        "finite": bool(finite.all()),
        "nan_count": int(np.isnan(grid).sum()),
        "inf_count": int(np.isinf(grid).sum()),
        "minimum": float(np.nanmin(grid)),
        "maximum": float(np.nanmax(grid)),
        "weight_sum_max_abs_error": float(np.max(np.abs(sums - 1.0))),
        "weight_sum_mean_abs_error": float(np.mean(np.abs(sums - 1.0))),
        "zero_sum_voxels": int(np.count_nonzero(np.abs(sums) <= 1.0e-12)),
        "unknown_joint_indices": int(np.count_nonzero((dominant < 0) | (dominant >= expected_shape[-1]))),
        "sparsity_abs_le_1e-8": float(np.mean(np.abs(grid) <= 1.0e-8)),
        "boundary": {
            "sample_count_with_face_duplicates": int(len(boundary)),
            "minimum": float(boundary.min()),
            "maximum": float(boundary.max()),
            "weight_sum_max_abs_error": float(np.max(np.abs(boundary.sum(axis=-1, dtype=np.float64) - 1.0))),
        },
        "bbox_min": arrays["bbox_min"].tolist(),
        "bbox_max": arrays["bbox_max"].tolist(),
        "grid_dims": arrays["grid_dims"].tolist(),
        "grid_resolution": float(np.asarray(arrays["grid_resolution"]).reshape(-1)[0]),
    }
    return audit, arrays


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--subject-data", type=Path, required=True)
    parser.add_argument("--smpl-model", type=Path, required=True)
    parser.add_argument("--point-interpolant", type=Path, required=True)
    parser.add_argument("--template-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", choices=("run_a", "run_b"), required=True)
    args = parser.parse_args()
    started = time.time()
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    expected_output = Path(protocol["publication"]["staging_root"]) / args.run_id / "lbs"
    expected_template = Path(protocol["publication"]["staging_root"]) / args.run_id / "template"
    if args.output_dir.resolve() != expected_output.resolve() or args.template_dir.resolve() != expected_template.resolve():
        raise ValueError("Run directories do not match frozen protocol")
    raw_root = Path(protocol["data"]["raw_root"]).resolve()
    for output in (args.output_dir,):
        try:
            output.resolve().relative_to(raw_root)
        except ValueError:
            pass
        else:
            raise ValueError("Refusing to write beneath raw subject00")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lbs_cfg = protocol["lbs"]
    frozen_generator = args.repo_root / lbs_cfg["generator"]
    if sha256_file(frozen_generator) != lbs_cfg["generator_sha256"]:
        raise ValueError("Frozen LBS generator hash mismatch")
    if sha256_file(args.point_interpolant) != lbs_cfg["point_interpolant_sha256"]:
        raise ValueError("PointInterpolant hash mismatch")
    if sha256_file(args.smpl_model) != protocol["body_model"]["model_sha256"]:
        raise ValueError("SMPL-X model hash mismatch")
    template_vertices, template_faces = load_template_and_verify(
        args.template_dir, args.subject_data, args.smpl_model, protocol
    )

    work = args.output_dir / "work"
    data = work / "data"
    data.mkdir(parents=True)
    params_source = args.subject_data / protocol["data"]["smpl_params"]
    params_copy = data / "smpl_params.npz"
    shutil.copyfile(params_source, params_copy)
    interpolant_copy = work / "PointInterpolant"
    shutil.copyfile(args.point_interpolant, interpolant_copy)
    interpolant_copy.chmod(interpolant_copy.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    if sha256_file(params_copy) != sha256_file(params_source) or sha256_file(interpolant_copy) != lbs_cfg["point_interpolant_sha256"]:
        raise ValueError("Isolated input copy verification failed")

    stdout_path = args.output_dir / "generator_stdout.log"
    stderr_path = args.output_dir / "generator_stderr.log"
    command = [
        sys.executable,
        str(frozen_generator),
        "--data_dir",
        str(data),
        "--smpl_path",
        str(args.smpl_model),
    ]
    environment = dict(os.environ)
    environment.update({"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    with stdout_path.open("w", encoding="utf-8", newline="\n") as stdout, stderr_path.open("w", encoding="utf-8", newline="\n") as stderr:
        completed = subprocess.run(
            command,
            cwd=work,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
            text=True,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"Frozen LBS generator failed with exit code {completed.returncode}")
    grid_files = sorted((work / "tmp").glob("grid_*.grd"))
    if len(grid_files) != int(lbs_cfg["joint_count"]):
        raise RuntimeError(f"Expected 55 solver grids, found {len(grid_files)}")
    generated = data / "gaussian" / lbs_cfg["output_name"]
    if not generated.is_file():
        raise FileNotFoundError(generated)
    final = args.output_dir / lbs_cfg["output_name"]
    shutil.copyfile(generated, final)
    audit, arrays = grid_audit(
        final,
        (int(lbs_cfg["grid_resolution"]),) * 3 + (int(lbs_cfg["joint_count"]),),
    )
    threshold = lbs_cfg["value_thresholds"]
    gate = (
        audit["shape_exact"]
        and audit["finite"]
        and audit["nan_count"] == 0
        and audit["inf_count"] == 0
        and audit["minimum"] >= float(threshold["weight_min"])
        and audit["weight_sum_max_abs_error"] <= float(threshold["weight_sum_max_abs_error"])
        and audit["zero_sum_voxels"] == int(threshold["zero_sum_voxels"])
        and audit["unknown_joint_indices"] == int(threshold["unknown_joint_indices"])
    )
    report = {
        "task_id": protocol["task_id"],
        "run_id": args.run_id,
        "status": "PASS" if gate else "FAIL",
        "environment": {
            "hostname": socket.gethostname(),
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "smplx": getattr(smplx, "__version__", "UNKNOWN"),
            "pythonhashseed": environment["PYTHONHASHSEED"],
            "omp_num_threads": environment["OMP_NUM_THREADS"],
        },
        "inputs": {
            "template_vertices_sha256": sha256_array(template_vertices),
            "template_faces_sha256": sha256_array(template_faces),
            "smpl_params_sha256": sha256_file(params_source),
            "smpl_model_sha256": sha256_file(args.smpl_model),
            "generator": str(frozen_generator),
            "generator_sha256": sha256_file(frozen_generator),
            "point_interpolant_sha256": sha256_file(interpolant_copy),
            "threads": int(lbs_cfg["threads"]),
            "depth": int(lbs_cfg["depth"]),
            "gradient_weight": float(lbs_cfg["gradient_weight"]),
        },
        "command": command,
        "returncode": completed.returncode,
        "solver_grid_count": len(grid_files),
        "solver_grid_total_bytes": sum(path.stat().st_size for path in grid_files),
        "audit": audit,
        "output": {
            "path": str(final),
            "file_sha256": sha256_file(final),
            "file_bytes": final.stat().st_size,
            "array_hashes": {key: sha256_array(value) for key, value in arrays.items()},
        },
        "logs": {"stdout": str(stdout_path), "stderr": str(stderr_path)},
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
    write_json(args.output_dir / "lbs_generation_report.json", report)
    print(json.dumps({
        "run_id": args.run_id,
        "status": report["status"],
        "output_sha256": report["output"]["file_sha256"],
        "array_hashes": report["output"]["array_hashes"],
        "audit": audit,
        "wall_clock_seconds": report["wall_clock_seconds"],
    }, indent=2))
    return 0 if gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
