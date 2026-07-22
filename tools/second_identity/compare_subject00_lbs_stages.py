#!/usr/bin/env python3
"""Compare prior and repair LBS stages without changing frozen attempts."""

from __future__ import annotations

import argparse
import array
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


GRID_SIZE = 128
JOINT_COUNT = 55
PAYLOAD_COUNT = GRID_SIZE**3
PAYLOAD_BYTES = PAYLOAD_COUNT * 8


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def sha256_memmap(path: Path) -> str:
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    digest = hashlib.sha256()
    for index in range(value.shape[0]):
        digest.update(np.ascontiguousarray(value[index]).tobytes(order="C"))
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_grd(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    header_len = len(raw) - PAYLOAD_BYTES
    if header_len < 0:
        raise RuntimeError(f"Short grid: {path}")
    value = np.asarray(array.array("d", raw[header_len:]), dtype=np.float64)
    if value.size != PAYLOAD_COUNT:
        raise RuntimeError(f"Bad payload count: {path} {value.size}")
    return value.reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE)


def aggregate_input_hash(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    files = []
    for channel in range(JOINT_COUNT):
        for kind in ("val", "grad"):
            path = root / f"cano_data_lbs_{kind}_{channel:02d}.xyz"
            item_hash = sha256_file(path)
            digest.update(path.name.encode("utf-8"))
            digest.update(bytes.fromhex(item_hash))
            files.append({"channel": channel, "kind": kind, "sha256": item_hash, "bytes": path.stat().st_size})
    return {"aggregate_sha256": digest.hexdigest(), "files": files}


def compare_grd_sets(paths_a: list[Path], paths_b: list[Path]) -> dict[str, Any]:
    total_abs = 0.0
    count = 0
    maximum = 0.0
    file_exact = []
    payload_exact = []
    per_channel = []
    for channel, (path_a, path_b) in enumerate(zip(paths_a, paths_b, strict=True)):
        file_equal = sha256_file(path_a) == sha256_file(path_b)
        a = parse_grd(path_a)
        b = parse_grd(path_b)
        difference = np.abs(a - b)
        channel_max = float(difference.max())
        channel_mean = float(difference.mean(dtype=np.float64))
        numeric_equal = bool(np.array_equal(a, b))
        maximum = max(maximum, channel_max)
        total_abs += float(difference.sum(dtype=np.float64))
        count += difference.size
        file_exact.append(file_equal)
        payload_exact.append(numeric_equal)
        per_channel.append(
            {
                "channel": channel,
                "file_exact": file_equal,
                "payload_exact": numeric_equal,
                "max_abs": channel_max,
                "mean_abs": channel_mean,
            }
        )
    return {
        "file_bitwise_exact": bool(all(file_exact)),
        "file_exact_channels": int(sum(file_exact)),
        "payload_bitwise_exact": bool(all(payload_exact)),
        "payload_exact_channels": int(sum(payload_exact)),
        "max_abs": maximum,
        "mean_abs": total_abs / count,
        "per_channel": per_channel,
    }


def compare_npy(path_a: Path, path_b: Path, argmax: bool = False) -> dict[str, Any]:
    a = np.load(path_a, mmap_mode="r", allow_pickle=False)
    b = np.load(path_b, mmap_mode="r", allow_pickle=False)
    if a.shape != b.shape or a.dtype != b.dtype:
        return {"shape_a": list(a.shape), "shape_b": list(b.shape), "dtype_a": str(a.dtype), "dtype_b": str(b.dtype), "compatible": False}
    total_abs = 0.0
    count = 0
    maximum = 0.0
    exact = True
    argmax_equal = 0
    argmax_count = 0
    for index in range(a.shape[0]):
        slice_a = np.asarray(a[index])
        slice_b = np.asarray(b[index])
        exact = exact and bool(np.array_equal(slice_a, slice_b))
        difference = np.abs(slice_a.astype(np.float64) - slice_b.astype(np.float64))
        maximum = max(maximum, float(difference.max()))
        total_abs += float(difference.sum(dtype=np.float64))
        count += difference.size
        if argmax:
            map_a = np.argmax(slice_a, axis=-1)
            map_b = np.argmax(slice_b, axis=-1)
            argmax_equal += int(np.sum(map_a == map_b))
            argmax_count += map_a.size
    result = {
        "compatible": True,
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "bitwise_exact": exact,
        "max_abs": maximum,
        "mean_abs": total_abs / count,
        "array_sha256_a": sha256_memmap(path_a),
        "array_sha256_b": sha256_memmap(path_b),
    }
    if argmax:
        result["argmax_joint_agreement"] = argmax_equal / argmax_count
    return result


def reconstruct_prior(run_root: Path, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=False)
    parsed_path = output_root / "parsed_unnormalized_float64.npy"
    normalized_path = output_root / "normalized_channel_first_float64.npy"
    canonical_path = output_root / "canonical_grid_xyz_joint_float32.npy"
    parsed = np.lib.format.open_memmap(
        parsed_path, mode="w+", dtype=np.float64, shape=(JOINT_COUNT, GRID_SIZE, GRID_SIZE, GRID_SIZE)
    )
    raw_paths = [run_root / "lbs" / "work" / "tmp" / f"grid_{channel:02d}.grd" for channel in range(JOINT_COUNT)]
    for channel, path in enumerate(raw_paths):
        parsed[channel] = parse_grd(path)
    parsed.flush()
    denominator = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.float64)
    for channel in range(JOINT_COUNT):
        denominator += np.clip(parsed[channel], 0.0, 1.0)
    zero_sum = int(np.sum(denominator == 0.0))
    if zero_sum:
        raise RuntimeError(f"Prior run contains {zero_sum} zero-sum voxels")
    normalized = np.lib.format.open_memmap(
        normalized_path, mode="w+", dtype=np.float64, shape=parsed.shape
    )
    canonical = np.lib.format.open_memmap(
        canonical_path, mode="w+", dtype=np.float32, shape=(GRID_SIZE, GRID_SIZE, GRID_SIZE, JOINT_COUNT)
    )
    for channel in range(JOINT_COUNT):
        normalized[channel] = np.clip(parsed[channel], 0.0, 1.0) / denominator
        canonical[..., channel] = normalized[channel].transpose(2, 1, 0).astype(np.float32)
    normalized.flush()
    canonical.flush()
    final_npz = run_root / "lbs" / "lbs_weights_grid.npz"
    with np.load(final_npz, allow_pickle=False) as loaded:
        archived_grid = loaded["grid"]
        canonical_matches_archived = bool(np.array_equal(canonical, archived_grid))
        metadata = {key: np.asarray(loaded[key]).tolist() for key in ("bbox_min", "bbox_max", "grid_dims", "grid_resolution")}
    return {
        "raw_paths": [str(path) for path in raw_paths],
        "parsed_path": str(parsed_path),
        "normalized_path": str(normalized_path),
        "canonical_path": str(canonical_path),
        "canonical_matches_archived_grid_exact": canonical_matches_archived,
        "archived_npz_sha256": sha256_file(final_npz),
        "metadata": metadata,
        "stage_hashes": {
            "parsed": sha256_memmap(parsed_path),
            "normalized": sha256_memmap(normalized_path),
            "canonical": sha256_memmap(canonical_path),
        },
    }


def compare_run_reports(run_a: Path, run_b: Path) -> dict[str, Any]:
    report_a = load_json(run_a / "reports" / "run_report.json")
    report_b = load_json(run_b / "reports" / "run_report.json")
    raw_paths_a = [run_a / "raw_solver" / f"grid_{channel:02d}.grd" for channel in range(JOINT_COUNT)]
    raw_paths_b = [run_b / "raw_solver" / f"grid_{channel:02d}.grd" for channel in range(JOINT_COUNT)]
    raw = compare_grd_sets(raw_paths_a, raw_paths_b)
    parsed = compare_npy(
        run_a / "stages" / "parsed_unnormalized_float64.npy",
        run_b / "stages" / "parsed_unnormalized_float64.npy",
    )
    normalized = compare_npy(
        run_a / "stages" / "normalized_channel_first_float64.npy",
        run_b / "stages" / "normalized_channel_first_float64.npy",
    )
    canonical = compare_npy(
        run_a / "stages" / "canonical_grid_xyz_joint_float32.npy",
        run_b / "stages" / "canonical_grid_xyz_joint_float32.npy",
        argmax=True,
    )
    metadata_exact = report_a["metadata"] == report_b["metadata"]
    npz_hash_a = report_a["stages"]["canonical_npz"]["file_sha256"]
    npz_hash_b = report_b["stages"]["canonical_npz"]["file_sha256"]
    if not raw["payload_bitwise_exact"]:
        first_difference = "SOLVER_OUTPUT_DIFFERENCE"
    elif not parsed["bitwise_exact"]:
        first_difference = "PARSER_DIFFERENCE"
    elif not normalized["bitwise_exact"]:
        first_difference = "NORMALIZATION_DIFFERENCE"
    elif npz_hash_a != npz_hash_b:
        first_difference = "NPZ_CONTAINER_ONLY_DIFFERENCE"
    else:
        first_difference = "NO_DIFFERENCE"
    thresholds_pass = bool(
        canonical["max_abs"] <= 1.0e-6
        and canonical["mean_abs"] <= 1.0e-8
        and canonical["argmax_joint_agreement"] == 1.0
        and metadata_exact
    )
    return {
        "run_a": report_a["run_id"],
        "run_b": report_b["run_id"],
        "threads_a": report_a["threads"],
        "threads_b": report_b["threads"],
        "gradient_inputs_exact": report_a["inputs"]["gradient_inputs"]["aggregate_sha256"]
        == report_b["inputs"]["gradient_inputs"]["aggregate_sha256"],
        "raw_solver": raw,
        "parsed_unnormalized": parsed,
        "normalized": normalized,
        "canonical_grid": canonical,
        "metadata_exact": metadata_exact,
        "npz_sha256_a": npz_hash_a,
        "npz_sha256_b": npz_hash_b,
        "npz_bitwise_exact": npz_hash_a == npz_hash_b,
        "first_difference": first_difference,
        "repeatability_thresholds_pass": thresholds_pass,
        "wall_clock_seconds_a": report_a["resources"]["wall_clock_seconds"],
        "wall_clock_seconds_b": report_b["resources"]["wall_clock_seconds"],
        "peak_rss_kib_a": max(report_a["resources"]["self_max_rss_kib"], report_a["resources"]["children_max_rss_kib"]),
        "peak_rss_kib_b": max(report_b["resources"]["self_max_rss_kib"], report_b["resources"]["children_max_rss_kib"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-001", type=Path, required=True)
    parser.add_argument("--attempt-002", type=Path, required=True)
    args = parser.parse_args()
    reports = args.attempt_002 / "reports"
    reconstructed = args.attempt_002 / "reconstructed_attempt_001"
    prior_a = reconstruct_prior(args.attempt_001 / "run_a", reconstructed / "run_a")
    prior_b = reconstruct_prior(args.attempt_001 / "run_b", reconstructed / "run_b")
    prior_input_a = aggregate_input_hash(args.attempt_001 / "run_a" / "lbs" / "work" / "tmp")
    prior_input_b = aggregate_input_hash(args.attempt_001 / "run_b" / "lbs" / "work" / "tmp")
    prior_raw = compare_grd_sets(
        [Path(path) for path in prior_a["raw_paths"]], [Path(path) for path in prior_b["raw_paths"]]
    )
    prior_parsed = compare_npy(Path(prior_a["parsed_path"]), Path(prior_b["parsed_path"]))
    prior_normalized = compare_npy(Path(prior_a["normalized_path"]), Path(prior_b["normalized_path"]))
    prior_canonical = compare_npy(Path(prior_a["canonical_path"]), Path(prior_b["canonical_path"]), argmax=True)
    prior_first = "SOLVER_OUTPUT_DIFFERENCE" if not prior_raw["payload_bitwise_exact"] else "NO_DIFFERENCE"
    prior_comparison = {
        "task_id": "MMLPHUMAN-SUBJECT00-LBS-REPAIR-001",
        "status": "PASS_DIAGNOSIS",
        "attempt": "attempt_001_read_only_reconstruction",
        "gradient_inputs_exact": prior_input_a["aggregate_sha256"] == prior_input_b["aggregate_sha256"],
        "gradient_input_sha256_a": prior_input_a["aggregate_sha256"],
        "gradient_input_sha256_b": prior_input_b["aggregate_sha256"],
        "raw_solver": prior_raw,
        "parsed_unnormalized": prior_parsed,
        "normalized": prior_normalized,
        "canonical_grid": prior_canonical,
        "metadata_exact": prior_a["metadata"] == prior_b["metadata"],
        "npz_sha256_a": prior_a["archived_npz_sha256"],
        "npz_sha256_b": prior_b["archived_npz_sha256"],
        "canonical_matches_archived_grid_exact_a": prior_a["canonical_matches_archived_grid_exact"],
        "canonical_matches_archived_grid_exact_b": prior_b["canonical_matches_archived_grid_exact"],
        "first_difference": prior_first,
        "classification": prior_first,
    }
    write_json(reports / "subject00_lbs_solver_stage_comparison.json", prior_comparison)

    t1 = compare_run_reports(args.attempt_002 / "runs" / "T1_A", args.attempt_002 / "runs" / "T1_B")
    t12 = compare_run_reports(args.attempt_002 / "runs" / "T12_A", args.attempt_002 / "runs" / "T12_B")
    cross = compare_npy(
        args.attempt_002 / "runs" / "T1_A" / "stages" / "canonical_grid_xyz_joint_float32.npy",
        args.attempt_002 / "runs" / "T12_A" / "stages" / "canonical_grid_xyz_joint_float32.npy",
        argmax=True,
    )
    if t1["repeatability_thresholds_pass"] and not t12["repeatability_thresholds_pass"]:
        classification = "MULTITHREAD_ONLY_NONDETERMINISM"
    elif not t1["repeatability_thresholds_pass"]:
        classification = "THREAD_INDEPENDENT_NONDETERMINISM"
    elif t1["canonical_grid"]["bitwise_exact"] and t12["canonical_grid"]["bitwise_exact"]:
        classification = "SINGLE_THREAD_BITWISE_DETERMINISTIC"
    else:
        classification = "SINGLE_THREAD_NUMERICALLY_DETERMINISTIC"
    thread_report = {
        "task_id": "MMLPHUMAN-SUBJECT00-LBS-REPAIR-001",
        "status": "PASS_DIAGNOSIS",
        "T1_A_vs_T1_B": t1,
        "T12_A_vs_T12_B": t12,
        "T1_A_vs_T12_A": cross,
        "classification": classification,
        "thresholds_unchanged": {
            "max_abs": 1.0e-6,
            "mean_abs": 1.0e-8,
            "argmax_joint_agreement": 1.0,
            "metadata_exact": True,
        },
        "single_thread_runtime_mean_seconds": (t1["wall_clock_seconds_a"] + t1["wall_clock_seconds_b"]) / 2.0,
        "twelve_thread_runtime_mean_seconds": (t12["wall_clock_seconds_a"] + t12["wall_clock_seconds_b"]) / 2.0,
    }
    write_json(reports / "subject00_lbs_thread_determinism.json", thread_report)
    print(json.dumps({"prior_first_difference": prior_first, "thread_classification": classification}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
