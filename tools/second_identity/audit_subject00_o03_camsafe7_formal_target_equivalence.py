#!/usr/bin/env python3
"""Read-only equivalence audit for the O03 camera-safe7 Teacher targets.

This program deliberately performs no rendering, GPU forward pass, optimizer
construction, target materialization, or checkpoint mutation.  It reads the
sealed rerun snapshot and the formal materialized O03 index, executes both CPU
loader contracts at zero optimization steps, and emits one machine snapshot
used to build the Git-side audit records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


TASK_ID = "AAAI27-SUBJECT00-O03-CAMSAFE7-FORMAL-TARGET-EQUIVALENCE-AUDIT-001"
EXPECTED_REQUESTS = [
    "subject00_O03_slot00_canary_attempt004_cand00",
    "subject00_O03_slot01_remaining_attempt005_cand00",
    "subject00_O03_slot02_cand00",
    "subject00_O03_slot03_cand01",
    "subject00_O03_slot05_canary_attempt004_cand00",
    "subject00_O03_slot06_remaining_attempt005_cand00",
    "subject00_O03_slot07_canary_attempt004_cand00",
]
EXPECTED_SLOTS = [
    "slot_00",
    "slot_01",
    "slot_02",
    "slot_03",
    "slot_05",
    "slot_06",
    "slot_07",
]
EXPECTED_CAMERAS = [17, 21, 14, 23, 2, 9, 5]
EXCLUDED_REQUEST = "subject00_O03_slot04_canary_attempt004_cand00"
CHECKPOINT_STEPS = [0, 300, 600, 900, 1200]
FORMAL_SCIENTIFIC_FIELDS = [
    "target_edit_rgb",
    "target_foreground_mask",
    "target_clothing_mask",
    "target_base_rgb",
    "target_edit_mask",
    "target_edit_core_mask",
    "target_preserve_mask",
    "target_transition_mask",
    "target_protected_mask",
    "target_base_foreground_mask",
    "target_old_clothing_mask",
    "target_revealed_skin_mask",
]
DERIVED_FIELD_INVENTORY = [
    "target_edit_rgb",
    "target_foreground_mask",
    "target_clothing_mask",
    "target_protected_mask",
    "boundary_target",
    "garment_rgb_masked_target",
    "target_base_rgb",
    "target_edit_mask",
    "target_edit_core_mask",
    "target_preserve_mask",
    "target_transition_mask",
    "target_base_foreground_mask",
    "target_old_clothing_mask",
    "target_revealed_skin_mask",
]
SCIENTIFIC_MISMATCH_FIELDS = [
    "target_base_rgb",
    "target_edit_mask",
    "target_edit_core_mask",
    "target_preserve_mask",
    "target_transition_mask",
    "target_base_foreground_mask",
    "target_old_clothing_mask",
    "target_revealed_skin_mask",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun-root", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--formal-repo", type=Path, required=True)
    parser.add_argument("--rerun-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    return array_sha(value.detach().cpu().contiguous().numpy())


def json_sha(value: Any) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def image_array(path: Path) -> tuple[np.ndarray, str, tuple[int, int]]:
    with Image.open(path) as image:
        mode = image.mode
        size = image.size
        array = np.asarray(image)
    return array, mode, size


def load_rgb_hwc(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def load_mask_hwc(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image, dtype=np.uint8)
        require(image.mode == "L", f"mask mode is not L: {path}")
    values = set(np.unique(array).tolist())
    require(values and values.issubset({0, 255}), f"mask is not binary: {path}")
    return (array > 0).astype(np.float32)[..., None]


def mask_stats(array: np.ndarray) -> dict[str, Any]:
    value = np.asarray(array)
    if value.ndim == 3:
        require(value.shape[-1] == 1, f"mask has unsupported shape: {value.shape}")
        value = value[..., 0]
    binary = value > 0
    count = int(np.count_nonzero(binary))
    # Count 8-connected components with a row-run union-find.  This avoids a
    # dependency on scipy/opencv in the sealed cloud environment and is much
    # cheaper than a Python flood fill over every foreground pixel.
    parents: list[int] = []
    areas: list[int] = []

    def add_run(length: int) -> int:
        index = len(parents)
        parents.append(index)
        areas.append(length)
        return index

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        parents[right_root] = left_root
        areas[left_root] += areas[right_root]

    previous: list[tuple[int, int, int]] = []
    for row in binary:
        padded = np.pad(row.astype(np.int8), (1, 1))
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        stops = np.flatnonzero(changes == -1)
        current = [
            (int(start), int(stop) - 1, add_run(int(stop - start)))
            for start, stop in zip(starts, stops, strict=True)
        ]
        previous_index = 0
        for start, stop, run_id in current:
            while (
                previous_index < len(previous)
                and previous[previous_index][1] < start - 1
            ):
                previous_index += 1
            candidate = previous_index
            while candidate < len(previous) and previous[candidate][0] <= stop + 1:
                union(run_id, previous[candidate][2])
                candidate += 1
        previous = current
    root_areas = [areas[index] for index in range(len(parents)) if find(index) == index]
    components = len(root_areas)
    if count:
        ys, xs = np.nonzero(binary)
        bbox: list[int] | None = [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        ]
    else:
        bbox = None
    component_areas = sorted((int(x) for x in root_areas), reverse=True)
    return {
        "value_set": sorted(int(x) for x in np.unique(value).tolist()),
        "foreground_pixel_count": count,
        "connected_components": components,
        "component_areas_descending": component_areas,
        "bbox_xyxy_exclusive": bbox,
    }


def boundary(mask_hwc: np.ndarray) -> np.ndarray:
    value = torch.from_numpy(mask_hwc.astype(np.float32, copy=False))
    nchw = value.permute(2, 0, 1)[None]
    dilation = F.max_pool2d(nchw, 3, stride=1, padding=1)
    erosion = 1 - F.max_pool2d(1 - nchw, 3, stride=1, padding=1)
    return (dilation - erosion).clamp(0, 1)[0].permute(1, 2, 0).numpy()


def formal_path(manifest_path: Path, observation: dict[str, Any], field: str) -> Path:
    return (manifest_path.parent / observation[field]).resolve()


def camera_id(value: Any) -> int:
    text = str(value)
    return int(text[3:]) if text.startswith("cam") else int(text)


def flatten_numeric(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def numerical_difference(left: Any, right: Any) -> tuple[float, float]:
    a, b = flatten_numeric(left), flatten_numeric(right)
    require(a.shape == b.shape, f"numeric shapes differ: {a.shape} != {b.shape}")
    diff = np.abs(a - b)
    denominator = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-15)
    return float(diff.max(initial=0.0)), float((diff / denominator).max(initial=0.0))


def normalized_formal_hwc(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().contiguous().numpy()
    require(array.ndim == 3, f"formal tensor is not CHW: {array.shape}")
    return np.ascontiguousarray(np.transpose(array, (1, 2, 0)))


def record_asset(
    rerun_path: Path,
    formal_path_value: Path,
    *,
    mask: bool,
) -> dict[str, Any]:
    rerun_array, rerun_mode, rerun_size = image_array(rerun_path)
    formal_array, formal_mode, formal_size = image_array(formal_path_value)
    if mask:
        rerun_metric = mask_stats(rerun_array)
        formal_metric = mask_stats(formal_array)
    else:
        rerun_metric = {
            "pixel_value_sha256": array_sha(rerun_array),
            "value_range": [int(rerun_array.min()), int(rerun_array.max())],
        }
        formal_metric = {
            "pixel_value_sha256": array_sha(formal_array),
            "value_range": [int(formal_array.min()), int(formal_array.max())],
        }
    return {
        "rerun": {
            "absolute_path": str(rerun_path),
            "bytes": rerun_path.stat().st_size,
            "sha256": sha256_file(rerun_path),
            "width": rerun_size[0],
            "height": rerun_size[1],
            "mode": rerun_mode,
            "parse_status": "PASS",
            **rerun_metric,
        },
        "formal": {
            "absolute_path": str(formal_path_value),
            "bytes": formal_path_value.stat().st_size,
            "sha256": sha256_file(formal_path_value),
            "width": formal_size[0],
            "height": formal_size[1],
            "mode": formal_mode,
            "parse_status": "PASS",
            **formal_metric,
        },
        "byte_exact": sha256_file(rerun_path) == sha256_file(formal_path_value),
        "pixel_exact": bool(np.array_equal(rerun_array, formal_array)),
        "path_class": (
            "SAME_PATH"
            if rerun_path == formal_path_value
            else "BYTE_EXACT_EQUIVALENT_DIFFERENT_PATH"
        ),
    }


def main() -> None:
    args = parse_args()
    rerun_root = args.rerun_root.resolve()
    formal_root = args.formal_root.resolve()
    rerun_manifest_path = (
        rerun_root / "inputs" / "target_snapshot" / "manifest.json"
    )
    rerun_training_index_path = (
        rerun_root / "inputs" / "target_snapshot" / "training_index.json"
    )
    rerun_evaluation_index_path = (
        rerun_root / "inputs" / "target_snapshot" / "evaluation_index.json"
    )
    rerun_derived_path = (
        rerun_root / "inputs" / "target_snapshot" / "derived_target_registry.json"
    )
    formal_index_path = (
        formal_root
        / "10_final_registry"
        / "indexes"
        / "O03_provisional_base60747_records.json"
    )
    formal_manifest_path = (
        formal_root
        / "10_final_registry"
        / "subject00_22_training_full_dataset_v1.json"
    )
    for path in (
        rerun_manifest_path,
        rerun_training_index_path,
        rerun_evaluation_index_path,
        rerun_derived_path,
        formal_index_path,
        formal_manifest_path,
    ):
        require(path.is_file(), f"required source is missing: {path}")

    rerun_manifest = read_json(rerun_manifest_path)
    rerun_training_index = read_json(rerun_training_index_path)
    rerun_evaluation_index = read_json(rerun_evaluation_index_path)
    rerun_derived = read_json(rerun_derived_path)
    formal_index = read_json(formal_index_path)
    formal_manifest = read_json(formal_manifest_path)

    rerun_records = rerun_manifest["records"]
    formal_index_records = formal_index["records"]
    rerun_request_order = [record["request_id"] for record in rerun_records]
    formal_request_order = [record["request_id"] for record in formal_index_records]
    require(rerun_request_order == EXPECTED_REQUESTS, "rerun request order changed")
    require(formal_request_order == EXPECTED_REQUESTS, "formal request order changed")
    require(int(rerun_manifest["denominator"]) == 7, "rerun denominator changed")
    require(int(formal_index["denominator"]) == 7, "formal denominator changed")
    require(EXCLUDED_REQUEST not in rerun_request_order, "slot04 leaked into rerun")
    require(EXCLUDED_REQUEST not in formal_request_order, "slot04 leaked into formal index")
    require(
        all(bool(row["training_eligible"]) for row in formal_index_records),
        "formal O03 record is not training eligible",
    )
    require(
        all(bool(row["evaluation_eligible"]) for row in formal_index_records),
        "formal O03 record is not evaluation eligible",
    )
    require(
        all(not bool(row["review_only"]) for row in formal_index_records),
        "formal O03 index contains quarantine",
    )

    formal_outfit = next(
        outfit for outfit in formal_manifest["outfits"] if outfit["outfit_id"] == "O03"
    )
    observations = {
        row["condition_id"]: row for row in formal_outfit["observations"]
    }
    require(list(observations) == EXPECTED_REQUESTS, "formal manifest O03 order changed")

    raw_mask_records: list[dict[str, Any]] = []
    camera_records: list[dict[str, Any]] = []
    maximum_abs = 0.0
    maximum_rel = 0.0
    camera_canonical_exact_count = 0
    for rerun_record, formal_index_record in zip(
        rerun_records, formal_index_records, strict=True
    ):
        request_id = rerun_record["request_id"]
        require(request_id == formal_index_record["request_id"], "record order differs")
        formal_record_path = formal_root / formal_index_record["record_path"]
        formal_record = read_json(formal_record_path)
        observation = observations[request_id]
        assets = {
            "raw": record_asset(
                Path(rerun_record["accepted_raw"]["path"]),
                formal_path(formal_manifest_path, observation, "target_edit_rgb"),
                mask=False,
            ),
            "person_mask": record_asset(
                Path(rerun_record["person_mask"]["path"]),
                formal_path(formal_manifest_path, observation, "target_foreground_mask"),
                mask=True,
            ),
            "garment_mask": record_asset(
                Path(rerun_record["garment_mask"]["path"]),
                formal_path(formal_manifest_path, observation, "target_clothing_mask"),
                mask=True,
            ),
        }
        require(
            assets["raw"]["formal"]["sha256"]
            == formal_record["raw_binding"]["materialized"]["sha256"]
            == observation["checksums"]["target_edit_rgb"],
            f"formal raw binding changed: {request_id}",
        )
        require(
            assets["person_mask"]["formal"]["sha256"]
            == formal_record["mask_bindings"]["person"]["materialized"]["sha256"]
            == observation["checksums"]["target_foreground_mask"],
            f"formal person-mask binding changed: {request_id}",
        )
        require(
            assets["garment_mask"]["formal"]["sha256"]
            == formal_record["mask_bindings"]["garment"]["materialized"]["sha256"]
            == observation["checksums"]["target_clothing_mask"],
            f"formal garment-mask binding changed: {request_id}",
        )
        person = load_mask_hwc(Path(rerun_record["person_mask"]["path"]))
        garment = load_mask_hwc(Path(rerun_record["garment_mask"]["path"]))
        garment_outside_person = int(np.count_nonzero((garment > 0) & ~(person > 0)))
        assets["garment_mask"]["garment_outside_person_pixel_count"] = (
            garment_outside_person
        )
        raw_mask_records.append(
            {
                "request_id": request_id,
                "slot": rerun_record["slot"],
                "camera_id": int(rerun_record["camera_id"]),
                "formal_record_path": str(formal_record_path),
                "assets": assets,
            }
        )

        formal_camera_path = formal_root / formal_index_record["camera_path"]
        formal_camera = read_json(formal_camera_path)
        require(
            sha256_file(formal_camera_path)
            == formal_record["camera_binding"]["sha256"],
            f"formal camera binding changed: {request_id}",
        )
        rerun_camera = rerun_record["camera_record"]
        rerun_t_pixel = [
            *rerun_record["pixel_registration"]["matrix_2x3"],
            [0.0, 0.0, 1.0],
        ]
        comparisons = {
            "K_source": numerical_difference(
                rerun_camera["calibration_K"], formal_camera["K_source"]
            ),
            "T_pixel": numerical_difference(rerun_t_pixel, formal_camera["T_pixel"]),
            "K_target": numerical_difference(
                rerun_camera["target_K"], formal_camera["K_target"]
            ),
            "w2c": numerical_difference(rerun_camera["w2c"], formal_camera["w2c"]),
            "c2w": numerical_difference(rerun_camera["c2w"], formal_camera["c2w"]),
        }
        record_abs = max(value[0] for value in comparisons.values())
        record_rel = max(value[1] for value in comparisons.values())
        maximum_abs = max(maximum_abs, record_abs)
        maximum_rel = max(maximum_rel, record_rel)
        canonical_exact = (
            camera_id(formal_camera["source_camera_id"])
            == int(rerun_record["camera_id"])
            == EXPECTED_CAMERAS[len(camera_records)]
            and formal_index_record["slot"] == rerun_record["slot"]
            and formal_index_record["direction"] == rerun_record["direction"]
            and int(formal_camera["width"])
            == int(rerun_record["native_resolution"]["width"])
            and int(formal_camera["height"])
            == int(rerun_record["native_resolution"]["height"])
            and record_abs == 0.0
            and record_rel == 0.0
            and formal_camera["transform_model"]
            == "ISOTROPIC_SIMILARITY_LEFT_MULTIPLY_K"
            and rerun_record["pixel_registration"]["method"] == "similarity"
        )
        camera_canonical_exact_count += int(canonical_exact)
        camera_records.append(
            {
                "request_id": request_id,
                "slot": rerun_record["slot"],
                "direction": rerun_record["direction"],
                "camera_id": int(rerun_record["camera_id"]),
                "rerun_camera_container": "embedded manifest camera_record plus pixel_registration",
                "rerun_camera_record_sha256": rerun_record["camera_record_sha256"],
                "formal_camera_path": str(formal_camera_path),
                "formal_camera_sha256": sha256_file(formal_camera_path),
                "camera_byte_exact": False,
                "camera_canonical_exact": canonical_exact,
                "canonical_comparison": {
                    field: {
                        "shape": list(flatten_numeric(formal_camera[
                            {
                                "K_source": "K_source",
                                "T_pixel": "T_pixel",
                                "K_target": "K_target",
                                "w2c": "w2c",
                                "c2w": "c2w",
                            }[field]
                        ]).shape),
                        "dtype": "float64_canonical",
                        "raw_serialized_value_exact": difference == (0.0, 0.0),
                        "max_abs_diff": difference[0],
                        "max_rel_diff": difference[1],
                    }
                    for field, difference in comparisons.items()
                },
                "width": int(formal_camera["width"]),
                "height": int(formal_camera["height"]),
                "transform_model": formal_camera["transform_model"],
                "transform_sha256": formal_camera["transform_sha256"],
                "source_checksum": formal_camera["source_checksum"],
                "target_checksum": formal_camera["target_checksum"],
                "conventions": formal_camera["conventions"],
            }
        )

    require(camera_canonical_exact_count == 7, "camera canonical comparison failed")

    # Execute the frozen formal CPU loader.  No model or renderer is imported.
    sys.path.insert(0, str(args.formal_repo.resolve()))
    from scene.full_dressable_dataset import (  # noqa: PLC0415
        FullDressableTrainingDataset,
    )

    formal_dataset = FullDressableTrainingDataset(
        formal_manifest_path, "train", reference_count=1, seed=0
    )
    formal_samples_all = [formal_dataset[index] for index in range(len(formal_dataset))]
    formal_samples = [
        sample
        for sample in formal_samples_all
        if sample["target_condition_id"] in EXPECTED_REQUESTS
    ]
    require(
        [sample["target_condition_id"] for sample in formal_samples]
        == EXPECTED_REQUESTS,
        "formal loader O03 order differs",
    )

    rerun_derived_by_id = {
        row["request_id"]: row for row in rerun_derived["records"]
    }
    rerun_loader_records: list[dict[str, Any]] = []
    formal_loader_records: list[dict[str, Any]] = []
    derived_records: list[dict[str, Any]] = []
    shared_shape_exact = True
    shared_value_exact = True
    protected_exact_count = 0
    boundary_runtime_exact_count = 0
    garment_rgb_runtime_exact_count = 0
    transition_boundary_equal_count = 0
    for rerun_record, formal_sample in zip(
        rerun_records, formal_samples, strict=True
    ):
        request_id = rerun_record["request_id"]
        raw = load_rgb_hwc(Path(rerun_record["accepted_raw"]["path"]))
        person = load_mask_hwc(Path(rerun_record["person_mask"]["path"]))
        garment = load_mask_hwc(Path(rerun_record["garment_mask"]["path"]))
        protected = person * (1.0 - garment)
        boundary_value = boundary(garment)
        garment_rgb = raw * garment
        rerun_values = {
            "raw": raw,
            "person": person,
            "garment": garment,
            "protected": protected,
            "boundary": boundary_value,
            "garment_rgb_masked_target": garment_rgb,
        }
        sealed = rerun_derived_by_id[request_id]
        require(
            array_sha(raw) == sealed["raw_tensor_sha256"],
            f"rerun raw tensor hash changed: {request_id}",
        )
        require(
            array_sha(person) == sealed["person_mask_tensor_sha256"],
            f"rerun person tensor hash changed: {request_id}",
        )
        require(
            array_sha(garment) == sealed["garment_mask_tensor_sha256"],
            f"rerun garment tensor hash changed: {request_id}",
        )
        require(
            array_sha(protected) == sealed["protected_mask_tensor_sha256"],
            f"rerun protected tensor hash changed: {request_id}",
        )
        require(
            array_sha(boundary_value) == sealed["boundary_mask_tensor_sha256"],
            f"rerun boundary tensor hash changed: {request_id}",
        )

        formal_values = {
            field: normalized_formal_hwc(formal_sample[field])
            for field in FORMAL_SCIENTIFIC_FIELDS
        }
        shared_pairs = {
            "raw_to_target_edit_rgb": (raw, formal_values["target_edit_rgb"]),
            "person_to_target_foreground_mask": (
                person,
                formal_values["target_foreground_mask"],
            ),
            "garment_to_target_clothing_mask": (
                garment,
                formal_values["target_clothing_mask"],
            ),
            "protected_to_target_protected_mask": (
                protected,
                formal_values["target_protected_mask"],
            ),
        }
        pair_results = {}
        for name, (left, right) in shared_pairs.items():
            shape_exact = left.shape == right.shape
            value_exact = shape_exact and bool(np.array_equal(left, right))
            shared_shape_exact &= shape_exact
            shared_value_exact &= value_exact
            pair_results[name] = {
                "shape_exact": shape_exact,
                "value_exact": value_exact,
                "rerun_shape": list(left.shape),
                "formal_shape_normalized_hwc": list(right.shape),
                "rerun_value_sha256": array_sha(left),
                "formal_value_sha256_normalized_hwc": array_sha(right),
            }
        protected_exact_count += int(
            pair_results["protected_to_target_protected_mask"]["value_exact"]
        )

        formal_boundary_runtime = boundary(formal_values["target_clothing_mask"])
        boundary_runtime_exact = bool(
            np.array_equal(boundary_value, formal_boundary_runtime)
        )
        boundary_runtime_exact_count += int(boundary_runtime_exact)
        formal_garment_rgb = (
            formal_values["target_edit_rgb"]
            * formal_values["target_clothing_mask"]
        )
        garment_rgb_exact = bool(np.array_equal(garment_rgb, formal_garment_rgb))
        garment_rgb_runtime_exact_count += int(garment_rgb_exact)
        transition_boundary_equal = bool(
            np.array_equal(
                boundary_value, formal_values["target_transition_mask"]
            )
        )
        transition_boundary_equal_count += int(transition_boundary_equal)

        rerun_loader_records.append(
            {
                "request_id": request_id,
                "field_keys": sorted(rerun_values),
                "fields": {
                    field: {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "value_sha256": array_sha(value),
                        "min": float(value.min()),
                        "max": float(value.max()),
                        "sum": float(value.sum()),
                        "nonzero_count": int(np.count_nonzero(value)),
                    }
                    for field, value in rerun_values.items()
                },
            }
        )
        formal_loader_records.append(
            {
                "request_id": request_id,
                "field_keys": sorted(FORMAL_SCIENTIFIC_FIELDS),
                "fields": {
                    field: {
                        "shape_chw": list(formal_sample[field].shape),
                        "shape_normalized_hwc": list(value.shape),
                        "dtype": str(value.dtype),
                        "value_sha256_normalized_hwc": array_sha(value),
                        "min": float(value.min()),
                        "max": float(value.max()),
                        "sum": float(value.sum()),
                        "nonzero_count": int(np.count_nonzero(value)),
                    }
                    for field, value in formal_values.items()
                },
            }
        )
        derived_records.append(
            {
                "request_id": request_id,
                "shared_field_comparisons": pair_results,
                "boundary_target": {
                    "classification": "SEMANTICALLY_EQUIVALENT_RUNTIME_DERIVATION",
                    "rerun_definition": "3x3 square morphological gradient of garment mask",
                    "formal_runtime_source": "target_clothing_mask",
                    "formal_persisted_equivalent_field": None,
                    "value_exact": boundary_runtime_exact,
                    "rerun_value_sha256": array_sha(boundary_value),
                    "formal_runtime_value_sha256": array_sha(formal_boundary_runtime),
                },
                "garment_rgb_masked_target": {
                    "classification": "SEMANTICALLY_EQUIVALENT_RUNTIME_DERIVATION",
                    "rerun_definition": "raw * garment",
                    "formal_runtime_definition": (
                        "target_edit_rgb * target_clothing_mask"
                    ),
                    "value_exact": garment_rgb_exact,
                    "rerun_value_sha256": array_sha(garment_rgb),
                    "formal_runtime_value_sha256": array_sha(formal_garment_rgb),
                },
                "formal_transition_vs_rerun_boundary": {
                    "value_exact": transition_boundary_equal,
                    "different_pixel_count": int(
                        np.count_nonzero(
                            formal_values["target_transition_mask"]
                            != boundary_value
                        )
                    ),
                    "formal_value_sha256": array_sha(
                        formal_values["target_transition_mask"]
                    ),
                    "rerun_value_sha256": array_sha(boundary_value),
                },
                "scientific_mismatch_fields": {
                    field: {
                        "classification": "SCIENTIFIC_FIELD_MISMATCH",
                        "formal_present": True,
                        "rerun_loader_present": False,
                        "formal_shape_normalized_hwc": list(
                            formal_values[field].shape
                        ),
                        "formal_dtype": str(formal_values[field].dtype),
                        "formal_value_sha256_normalized_hwc": array_sha(
                            formal_values[field]
                        ),
                        "formal_nonzero_count": int(
                            np.count_nonzero(formal_values[field])
                        ),
                        "reason": (
                            "rerun binds a live Base60747 render/alpha instead of "
                            "the formal precomputed base target"
                            if field
                            in {"target_base_rgb", "target_base_foreground_mask"}
                            else "field is required by the formal loader but absent "
                            "from the rerun loader and Teacher loss contract"
                        ),
                    }
                    for field in SCIENTIFIC_MISMATCH_FIELDS
                },
            }
        )

    require(protected_exact_count == 7, "protected target mismatch")
    require(boundary_runtime_exact_count == 7, "runtime boundary mismatch")
    require(garment_rgb_runtime_exact_count == 7, "garment RGB target mismatch")
    require(
        transition_boundary_equal_count < 7,
        "formal transition unexpectedly aliases rerun boundary",
    )

    checkpoint_records: list[dict[str, Any]] = []
    for step in CHECKPOINT_STEPS:
        stem = f"step_{step:06d}"
        checkpoint_path = rerun_root / "checkpoints" / f"{stem}.pth"
        sidecar_path = rerun_root / "checkpoints" / f"{stem}.sidecar.json"
        sidecar = read_json(sidecar_path)
        actual_sha = sha256_file(checkpoint_path)
        require(actual_sha == sidecar["sha256"], f"checkpoint SHA changed: {step}")
        require(
            sidecar["request_ids"] == EXPECTED_REQUESTS,
            f"checkpoint request binding changed: {step}",
        )
        require(
            sidecar["target_manifest_sha256"] == sha256_file(rerun_manifest_path),
            f"checkpoint manifest binding changed: {step}",
        )
        require(
            sidecar["excluded_request_id"] == EXCLUDED_REQUEST,
            f"checkpoint exclusion binding changed: {step}",
        )
        checkpoint_records.append(
            {
                "step": step,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_bytes": checkpoint_path.stat().st_size,
                "checkpoint_sha256": actual_sha,
                "sidecar_path": str(sidecar_path),
                "sidecar_sha256": sha256_file(sidecar_path),
                "target_manifest_sha256": sidecar["target_manifest_sha256"],
                "request_ids": sidecar["request_ids"],
                "camera_ids": EXPECTED_CAMERAS,
                "excluded_request_id": sidecar["excluded_request_id"],
                "denominator": 7,
                "view_sample_counts": sidecar["view_sample_counts"],
                "rerun_snapshot_binding_status": "PASS",
                "formal_shared_asset_binding_status": "PASS_RAW_PERSON_GARMENT",
                "formal_full_scientific_target_binding_status": (
                    "FAIL_DERIVED_AND_LOADER_FIELD_MISMATCH"
                ),
            }
        )

    raw_sha_exact = sum(
        row["assets"]["raw"]["byte_exact"] for row in raw_mask_records
    )
    raw_pixel_exact = sum(
        row["assets"]["raw"]["pixel_exact"] for row in raw_mask_records
    )
    person_sha_exact = sum(
        row["assets"]["person_mask"]["byte_exact"] for row in raw_mask_records
    )
    person_pixel_exact = sum(
        row["assets"]["person_mask"]["pixel_exact"] for row in raw_mask_records
    )
    garment_sha_exact = sum(
        row["assets"]["garment_mask"]["byte_exact"] for row in raw_mask_records
    )
    garment_pixel_exact = sum(
        row["assets"]["garment_mask"]["pixel_exact"] for row in raw_mask_records
    )
    require(
        (
            raw_sha_exact,
            raw_pixel_exact,
            person_sha_exact,
            person_pixel_exact,
            garment_sha_exact,
            garment_pixel_exact,
        )
        == (7, 7, 7, 7, 7, 7),
        "raw/mask exact equivalence failed",
    )

    rerun_implementation = (
        args.rerun_repo
        / "tools"
        / "second_identity"
        / "run_subject00_o03_provisional_teacher_camerasafe7.py"
    )
    formal_loader = args.formal_repo / "scene" / "full_dressable_dataset.py"
    output = {
        "schema_version": (
            "canondressgs.subject00.o03_camsafe7.formal_equivalence."
            "machine_snapshot.v1"
        ),
        "task_id": TASK_ID,
        "execution_mode": "READ_ONLY_ZERO_OPTIMIZER_CPU_AUDIT",
        "paths": {
            "rerun_root": str(rerun_root),
            "formal_root": str(formal_root),
            "rerun_manifest": str(rerun_manifest_path),
            "rerun_training_index": str(rerun_training_index_path),
            "rerun_evaluation_index": str(rerun_evaluation_index_path),
            "rerun_derived_registry": str(rerun_derived_path),
            "formal_O03_index": str(formal_index_path),
            "formal_manifest": str(formal_manifest_path),
        },
        "source_hashes": {
            "rerun_manifest_sha256": sha256_file(rerun_manifest_path),
            "rerun_training_index_sha256": sha256_file(rerun_training_index_path),
            "rerun_evaluation_index_sha256": sha256_file(
                rerun_evaluation_index_path
            ),
            "rerun_derived_registry_sha256": sha256_file(rerun_derived_path),
            "formal_O03_index_sha256": sha256_file(formal_index_path),
            "formal_manifest_sha256": sha256_file(formal_manifest_path),
            "rerun_loader_implementation_path": str(rerun_implementation),
            "rerun_loader_implementation_sha256": sha256_file(
                rerun_implementation
            ),
            "formal_loader_implementation_path": str(formal_loader),
            "formal_loader_implementation_sha256": sha256_file(formal_loader),
        },
        "request_binding": {
            "expected_request_order": EXPECTED_REQUESTS,
            "rerun_request_order": rerun_request_order,
            "formal_request_order": formal_request_order,
            "rerun_camera_order": [
                int(record["camera_id"]) for record in rerun_records
            ],
            "formal_camera_order": [
                camera_id(record["camera_id"]) for record in formal_index_records
            ],
            "rerun_denominator": int(rerun_manifest["denominator"]),
            "formal_denominator": int(formal_index["denominator"]),
            "slot04_present_in_rerun": EXCLUDED_REQUEST in rerun_request_order,
            "slot04_present_in_formal": EXCLUDED_REQUEST in formal_request_order,
            "rerun_training_index": rerun_training_index,
            "rerun_evaluation_index": rerun_evaluation_index,
        },
        "raw_mask_equivalence": {
            "raw_sha_exact_match_count": raw_sha_exact,
            "raw_pixel_exact_match_count": raw_pixel_exact,
            "person_mask_sha_exact_match_count": person_sha_exact,
            "person_mask_pixel_exact_match_count": person_pixel_exact,
            "garment_mask_sha_exact_match_count": garment_sha_exact,
            "garment_mask_pixel_exact_match_count": garment_pixel_exact,
            "records": raw_mask_records,
        },
        "camera_equivalence": {
            "camera_record_count": len(camera_records),
            "camera_byte_exact_count": 0,
            "camera_canonical_exact_count": camera_canonical_exact_count,
            "camera_numerical_max_abs_diff": maximum_abs,
            "camera_numerical_max_rel_diff": maximum_rel,
            "records": camera_records,
        },
        "derived_target_equivalence": {
            "field_inventory": DERIVED_FIELD_INVENTORY,
            "derived_field_count": len(DERIVED_FIELD_INVENTORY),
            "derived_byte_exact_count": 3,
            "derived_tensor_exact_count": 1,
            "derived_runtime_equivalent_count": 2,
            "derived_mismatch_count": len(SCIENTIFIC_MISMATCH_FIELDS),
            "scientific_mismatch_fields": SCIENTIFIC_MISMATCH_FIELDS,
            "protected_exact_count": protected_exact_count,
            "boundary_runtime_exact_count": boundary_runtime_exact_count,
            "garment_rgb_runtime_exact_count": garment_rgb_runtime_exact_count,
            "formal_transition_equals_rerun_boundary_count": (
                transition_boundary_equal_count
            ),
            "records": derived_records,
        },
        "loader_equivalence": {
            "rerun_loader": {
                "implementation_path": str(rerun_implementation),
                "implementation_sha256": sha256_file(rerun_implementation),
                "schema_label": rerun_manifest["schema_version"],
                "batch_size": rerun_manifest["loader_contract"]["batch_size"],
                "normalization": rerun_manifest["loader_contract"]["raw"],
                "mask_conversion": rerun_manifest["loader_contract"]["masks"],
                "records": rerun_loader_records,
            },
            "formal_loader": {
                "implementation_path": str(formal_loader),
                "implementation_sha256": sha256_file(formal_loader),
                "schema_label": formal_manifest["schema_version"],
                "batch_size": 1,
                "dataset_length_all_training": len(formal_dataset),
                "O03_subset_length": len(formal_samples),
                "records": formal_loader_records,
            },
            "implementation_status": "DIFFERENT_IMPLEMENTATIONS_AND_CONTRACTS",
            "field_set_status": "FAIL_SCIENTIFIC_FIELD_SET_MISMATCH_8_FIELDS",
            "shared_shape_status": (
                "PASS_7_OF_7_SHARED_FIELDS" if shared_shape_exact else "FAIL"
            ),
            "shared_value_hash_status": (
                "PASS_7_OF_7_SHARED_FIELDS" if shared_value_exact else "FAIL"
            ),
            "full_value_hash_status": (
                "FAIL_FORMAL_FIELDS_ABSENT_OR_DIFFERENTLY_BOUND_IN_RERUN"
            ),
            "request_order_status": "EXACT",
            "camera_order_status": "EXACT",
            "optimizer_steps": 0,
            "gpu_forward_calls": 0,
        },
        "checkpoint_target_binding": {
            "checkpoint_count": len(checkpoint_records),
            "checkpoint_steps": CHECKPOINT_STEPS,
            "status": (
                "PASS_RERUN_SNAPSHOT_5_OF_5; "
                "FAIL_FORMAL_FULL_SCIENTIFIC_TARGET_BINDING"
            ),
            "records": checkpoint_records,
        },
        "classification": {
            "target_equivalence_class": "D",
            "target_equivalence_label": "FORMAL_TARGET_SCIENTIFIC_FIELD_MISMATCH",
            "scientific_field_mismatch_count": len(SCIENTIFIC_MISMATCH_FIELDS),
            "mismatch_fields": SCIENTIFIC_MISMATCH_FIELDS,
            "rerun_trained_on_formal_equivalent_targets": False,
            "binding_overlay_authorized": False,
            "scientific_pass": None,
            "paper_eligible": False,
        },
        "immutability": {
            "optimizer_steps": 0,
            "gpu_forward_calls": 0,
            "target_mutations": 0,
            "mask_mutations": 0,
            "camera_record_mutations": 0,
            "checkpoint_mutations": 0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
