#!/usr/bin/env python3
"""Run one isolated subject00 PointInterpolant diagnostic with staged outputs.

This tool deliberately leaves the frozen ``script/gen_weight_volume.py``
unchanged.  It imports that script's gradient-input implementation, invokes
the sealed PointInterpolant binary with an explicit thread count, and records
each numeric/serialization stage separately.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import importlib.util
import io
import json
import locale
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
from typing import Any
import zipfile

import numpy as np
import smplx
import torch
import trimesh


THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
NPZ_KEY_ORDER = ("grid", "bbox_min", "bbox_max", "grid_dims", "grid_resolution")
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_gen_weight_volume", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import frozen generator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def npy_bytes(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.asarray(value), allow_pickle=False)
    return buffer.getvalue()


def deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for key in NPZ_KEY_ORDER:
            info = zipfile.ZipInfo(f"{key}.npy", date_time=FIXED_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, npy_bytes(arrays[key]))


def channel_statistics(values: np.ndarray) -> list[dict[str, Any]]:
    result = []
    for channel in range(values.shape[0]):
        item = np.asarray(values[channel])
        result.append(
            {
                "channel": channel,
                "sha256": sha256_array(item),
                "minimum": float(item.min()),
                "maximum": float(item.max()),
                "mean": float(item.mean(dtype=np.float64)),
                "nonzero_fraction": float(np.mean(item != 0.0)),
            }
        )
    return result


def raw_solver_manifest(paths: list[Path], grid_size: int) -> dict[str, Any]:
    aggregate = hashlib.sha256()
    files = []
    expected_payload_bytes = grid_size**3 * 8
    for channel, path in enumerate(paths):
        size = path.stat().st_size
        file_hash = sha256_file(path)
        header_bytes = size - expected_payload_bytes
        with path.open("rb") as handle:
            header = handle.read(header_bytes)
        header_hash = hashlib.sha256(header).hexdigest()
        aggregate.update(channel.to_bytes(4, "little"))
        aggregate.update(bytes.fromhex(file_hash))
        files.append(
            {
                "channel": channel,
                "path": str(path),
                "bytes": size,
                "header_bytes": header_bytes,
                "file_sha256": file_hash,
                "header_sha256": header_hash,
            }
        )
    return {"aggregate_sha256": aggregate.hexdigest(), "files": files}


def parse_raw_grids(paths: list[Path], grid_size: int) -> np.ndarray:
    grids = []
    payload_count = grid_size**3
    payload_bytes = payload_count * 8
    for path in paths:
        raw = path.read_bytes()
        header_len = len(raw) - payload_bytes
        if header_len < 0:
            raise RuntimeError(f"Short PointInterpolant output: {path}")
        payload = np.asarray(array.array("d", raw[header_len:]), dtype=np.float64)
        if payload.size != payload_count:
            raise RuntimeError(f"Unexpected payload count in {path}: {payload.size}")
        grids.append(payload.reshape(grid_size, grid_size, grid_size))
    return np.stack(grids, axis=0)


def basic_grid_audit(grid: np.ndarray) -> dict[str, Any]:
    sums = grid.sum(axis=-1, dtype=np.float64)
    dominant = np.argmax(grid, axis=-1)
    return {
        "shape": list(grid.shape),
        "dtype": str(grid.dtype),
        "sha256": sha256_array(grid),
        "finite": bool(np.isfinite(grid).all()),
        "nan_count": int(np.isnan(grid).sum()),
        "inf_count": int(np.isinf(grid).sum()),
        "minimum": float(np.nanmin(grid)),
        "maximum": float(np.nanmax(grid)),
        "weight_sum_max_abs_error": float(np.max(np.abs(sums - 1.0))),
        "weight_sum_mean_abs_error": float(np.mean(np.abs(sums - 1.0))),
        "zero_sum_voxels": int(np.sum(sums == 0.0)),
        "unknown_joint_indices": int(np.sum((dominant < 0) | (dominant >= grid.shape[-1]))),
    }


def environment_record() -> dict[str, Any]:
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "trimesh": trimesh.__version__,
        "locale_preferred_encoding": locale.getpreferredencoding(False),
        "locale_environment": {key: os.environ.get(key) for key in ("LANG", "LC_ALL", "LC_NUMERIC")},
        "thread_environment": {key: os.environ.get(key) for key in THREAD_VARIABLES},
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cpu_count": os.cpu_count(),
    }


def build_model_and_inputs(
    generator_module,
    smpl_model_path: Path,
    smpl_params_path: Path,
    input_dir: Path,
    generate_inputs: bool,
) -> tuple[Any, trimesh.Trimesh, np.ndarray, dict[str, Any]]:
    params = np.load(smpl_params_path, allow_pickle=False)
    betas = np.asarray(params["betas"][0], dtype=np.float32)
    model = smplx.SMPLX(
        model_path=str(smpl_model_path),
        use_pca=False,
        num_pca_comps=45,
        flat_hand_mean=True,
        batch_size=1,
    )
    with torch.no_grad():
        output = model.forward(
            betas=torch.as_tensor(betas)[None],
            global_orient=generator_module.config.cano_smpl_global_orient[None],
            transl=generator_module.config.cano_smpl_transl[None],
            body_pose=generator_module.config.cano_smpl_body_pose[None],
        )
    vertices = output.vertices[0].cpu().numpy().astype(np.float32)
    faces = np.asarray(model.faces, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    if generate_inputs:
        generator_module.tmp_dir = str(input_dir)
        generator_module.compute_lbs_grad(mesh, model.lbs_weights.cpu().numpy())
    audit = {
        "vertices_shape": list(vertices.shape),
        "vertices_sha256": sha256_array(vertices),
        "faces_shape": list(faces.shape),
        "faces_sha256": sha256_array(faces),
        "lbs_weights_shape": list(model.lbs_weights.shape),
        "lbs_weights_sha256": sha256_array(model.lbs_weights.cpu().numpy()),
        "betas_sha256": sha256_array(betas),
    }
    return model, mesh, vertices, audit


def input_manifest(input_dir: Path, joint_count: int) -> dict[str, Any]:
    aggregate = hashlib.sha256()
    files = []
    for channel in range(joint_count):
        for kind in ("val", "grad"):
            path = input_dir / f"cano_data_lbs_{kind}_{channel:02d}.xyz"
            digest = sha256_file(path)
            aggregate.update(path.name.encode("utf-8"))
            aggregate.update(bytes.fromhex(digest))
            files.append({"channel": channel, "kind": kind, "bytes": path.stat().st_size, "sha256": digest})
    return {"aggregate_sha256": aggregate.hexdigest(), "files": files}


def run_solver(
    binary: Path,
    input_dir: Path,
    raw_dir: Path,
    joint_count: int,
    depth: int,
    threads: int,
    log_path: Path,
) -> list[list[str]]:
    raw_dir.mkdir(parents=True, exist_ok=False)
    commands: list[list[str]] = []
    with log_path.open("wb") as log:
        for channel in range(joint_count):
            command = [
                str(binary),
                "--inValues",
                str(input_dir / f"cano_data_lbs_val_{channel:02d}.xyz"),
                "--inGradients",
                str(input_dir / f"cano_data_lbs_grad_{channel:02d}.xyz"),
                "--gradientWeight",
                "0.05",
                "--dim",
                "3",
                "--verbose",
                "--grid",
                str(raw_dir / f"grid_{channel:02d}.grd"),
                "--depth",
                str(depth),
                "--threads",
                str(threads),
            ]
            commands.append(command)
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"PointInterpolant channel {channel} returned {completed.returncode}")
    return commands


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--subject-root", type=Path, required=True)
    parser.add_argument("--smpl-model", type=Path, required=True)
    parser.add_argument("--point-interpolant", type=Path, required=True)
    parser.add_argument("--threads", type=int, choices=(1, 12), required=True)
    parser.add_argument("--frozen-input-root", type=Path)
    parser.add_argument("--frozen-bbox-template-vertices", type=Path)
    args = parser.parse_args()

    run_root = args.output_root.resolve()
    if run_root.exists():
        raise FileExistsError(f"Run root already exists: {run_root}")
    run_root.mkdir(parents=True)
    work_dir = run_root / "work"
    input_dir = work_dir / "tmp_inputs"
    data_dir = work_dir / "data"
    raw_dir = run_root / "raw_solver"
    stages_dir = run_root / "stages"
    output_dir = run_root / "output"
    reports_dir = run_root / "reports"
    for path in (input_dir, data_dir, stages_dir, output_dir, reports_dir):
        path.mkdir(parents=True)

    expected_threads = str(args.threads)
    captured = {key: os.environ.get(key) for key in THREAD_VARIABLES}
    if any(value != expected_threads for value in captured.values()):
        raise RuntimeError(f"Thread environment must all equal {expected_threads}: {captured}")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED must equal 0")

    frozen_generator = (args.worktree / "script" / "gen_weight_volume.py").resolve()
    smpl_params_source = (args.subject_root / "smpl_params.npz").resolve()
    local_smpl_params = data_dir / "smpl_params.npz"
    local_binary = work_dir / "PointInterpolant"
    shutil.copy2(smpl_params_source, local_smpl_params)
    shutil.copy2(args.point_interpolant.resolve(), local_binary)
    local_binary.chmod(0o755)

    started = time.perf_counter()
    generator_module = load_module(frozen_generator)
    model, mesh, vertices, reference_audit = build_model_and_inputs(
        generator_module,
        args.smpl_model.resolve(),
        local_smpl_params,
        input_dir,
        generate_inputs=args.frozen_input_root is None,
    )
    if args.frozen_input_root is not None:
        frozen_input_root = args.frozen_input_root.resolve()
        for channel in range(model.lbs_weights.shape[-1]):
            for kind in ("val", "grad"):
                source = frozen_input_root / f"cano_data_lbs_{kind}_{channel:02d}.xyz"
                target = input_dir / source.name
                shutil.copy2(source, target)
    inputs = input_manifest(input_dir, model.lbs_weights.shape[-1])
    solver_log = reports_dir / "point_interpolant.log"
    commands = run_solver(
        local_binary,
        input_dir,
        raw_dir,
        model.lbs_weights.shape[-1],
        7,
        args.threads,
        solver_log,
    )

    raw_paths = [raw_dir / f"grid_{channel:02d}.grd" for channel in range(model.lbs_weights.shape[-1])]
    raw_manifest = raw_solver_manifest(raw_paths, 128)
    parsed = parse_raw_grids(raw_paths, 128)
    parsed_path = stages_dir / "parsed_unnormalized_float64.npy"
    np.save(parsed_path, parsed, allow_pickle=False)

    normalized = np.clip(parsed, 0.0, 1.0)
    denominator = normalized.sum(axis=0, dtype=np.float64)
    zero_sum_before_normalization = int(np.sum(denominator == 0.0))
    if zero_sum_before_normalization:
        raise RuntimeError(f"Zero-sum voxels before normalization: {zero_sum_before_normalization}")
    normalized = normalized / denominator[None]
    normalized_path = stages_dir / "normalized_channel_first_float64.npy"
    np.save(normalized_path, normalized, allow_pickle=False)

    canonical_grid = normalized.transpose((3, 2, 1, 0)).astype(np.float32)
    canonical_grid_path = stages_dir / "canonical_grid_xyz_joint_float32.npy"
    np.save(canonical_grid_path, canonical_grid, allow_pickle=False)

    bbox_vertices = vertices
    if args.frozen_bbox_template_vertices is not None:
        bbox_vertices = np.load(args.frozen_bbox_template_vertices.resolve(), allow_pickle=False).astype(np.float32)
    min_xyz = bbox_vertices.min(axis=0).astype(np.float32)
    max_xyz = bbox_vertices.max(axis=0).astype(np.float32)
    # Preserve the frozen generator's scalar-expression semantics exactly.
    max_len = 1.1 * (max_xyz - min_xyz).max()
    center = 0.5 * (min_xyz + max_xyz)
    volume_bounds = np.stack([center - 0.5 * max_len, center + 0.5 * max_len], axis=0)
    bbox_min = volume_bounds[0]
    bbox_max = volume_bounds[1]
    grid_dims = np.asarray(canonical_grid.shape[:3], dtype=np.int64)
    grid_resolution = np.asarray((bbox_max[0] - bbox_min[0]) / (grid_dims[0] - 1), dtype=np.float64)
    output_arrays = {
        "grid": canonical_grid,
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "grid_dims": grid_dims,
        "grid_resolution": grid_resolution,
    }
    npz_path = output_dir / "lbs_weights_grid.npz"
    deterministic_npz(npz_path, output_arrays)

    with np.load(npz_path, allow_pickle=False) as loaded:
        reload_exact = all(np.array_equal(loaded[key], output_arrays[key]) for key in NPZ_KEY_ORDER)
        reloaded_keys = list(loaded.files)
    if not reload_exact or reloaded_keys != list(NPZ_KEY_ORDER):
        raise RuntimeError("Deterministic NPZ reload contract failed")

    child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    self_usage = resource.getrusage(resource.RUSAGE_SELF)
    report = {
        "task_id": "MMLPHUMAN-SUBJECT00-LBS-REPAIR-001",
        "run_id": args.run_id,
        "status": "PASS_GENERATION_AND_BASIC_CONTRACT",
        "threads": args.threads,
        "environment": environment_record(),
        "working_directory": str(Path.cwd()),
        "run_root": str(run_root),
        "temporary_paths": {
            "input_dir": str(input_dir),
            "raw_solver_dir": str(raw_dir),
            "stages_dir": str(stages_dir),
        },
        "inputs": {
            "frozen_generator": str(frozen_generator),
            "frozen_generator_sha256": sha256_file(frozen_generator),
            "smpl_params_source": str(smpl_params_source),
            "smpl_params_sha256": sha256_file(local_smpl_params),
            "smpl_model": str(args.smpl_model.resolve()),
            "smpl_model_sha256": sha256_file(args.smpl_model.resolve()),
            "point_interpolant_source": str(args.point_interpolant.resolve()),
            "point_interpolant_sha256": sha256_file(local_binary),
            "gradient_inputs": inputs,
            "reference": reference_audit,
            "solver_input_policy": "independently_generated" if args.frozen_input_root is None else "copied_from_frozen_common_input_pack",
            "frozen_input_root": None if args.frozen_input_root is None else str(args.frozen_input_root.resolve()),
            "bbox_template_vertices": None if args.frozen_bbox_template_vertices is None else str(args.frozen_bbox_template_vertices.resolve()),
            "bbox_template_vertices_sha256": None if args.frozen_bbox_template_vertices is None else sha256_file(args.frozen_bbox_template_vertices.resolve()),
        },
        "solver": {
            "joint_count": int(model.lbs_weights.shape[-1]),
            "command_template": commands[0][:-2] + ["--threads", str(args.threads)],
            "commands_executed": len(commands),
            "raw": raw_manifest,
            "log_path": str(solver_log),
            "log_sha256": sha256_file(solver_log),
        },
        "stages": {
            "parsed_unnormalized": {
                "path": str(parsed_path),
                "file_sha256": sha256_file(parsed_path),
                "array_sha256": sha256_array(parsed),
                "shape": list(parsed.shape),
                "dtype": str(parsed.dtype),
                "minimum": float(parsed.min()),
                "maximum": float(parsed.max()),
                "nan_count": int(np.isnan(parsed).sum()),
                "inf_count": int(np.isinf(parsed).sum()),
                "channels": channel_statistics(parsed),
            },
            "normalized_channel_first": {
                "path": str(normalized_path),
                "file_sha256": sha256_file(normalized_path),
                "array_sha256": sha256_array(normalized),
                "shape": list(normalized.shape),
                "dtype": str(normalized.dtype),
                "minimum": float(normalized.min()),
                "maximum": float(normalized.max()),
                "zero_sum_before_normalization": zero_sum_before_normalization,
                "nan_count": int(np.isnan(normalized).sum()),
                "inf_count": int(np.isinf(normalized).sum()),
                "channels": channel_statistics(normalized),
            },
            "canonical_grid": {
                "path": str(canonical_grid_path),
                "file_sha256": sha256_file(canonical_grid_path),
                **basic_grid_audit(canonical_grid),
            },
            "canonical_npz": {
                "path": str(npz_path),
                "file_sha256": sha256_file(npz_path),
                "file_bytes": npz_path.stat().st_size,
                "fixed_key_order": list(NPZ_KEY_ORDER),
                "fixed_zip_timestamp": list(FIXED_ZIP_TIMESTAMP),
                "compression": "ZIP_STORED",
                "reload_exact": reload_exact,
            },
        },
        "metadata": {
            "bbox_min": bbox_min.tolist(),
            "bbox_max": bbox_max.tolist(),
            "grid_dims": grid_dims.tolist(),
            "grid_resolution": float(grid_resolution),
        },
        "resources": {
            "wall_clock_seconds": time.perf_counter() - started,
            "self_max_rss_kib": int(self_usage.ru_maxrss),
            "children_max_rss_kib": int(child_usage.ru_maxrss),
        },
        "counters": {
            "training_steps": 0,
            "forward_training_batches": 0,
            "backward_calls": 0,
            "optimizer_created": 0,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "checkpoint_writes": 0,
            "PAPER_FINAL": 0,
        },
    }
    write_json(reports_dir / "run_report.json", report)
    print(json.dumps({"status": report["status"], "run_id": args.run_id, "report": str(reports_dir / "run_report.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
