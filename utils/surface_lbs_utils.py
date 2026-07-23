"""Deterministic surface-LBS asset loading and fail-closed validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SURFACE_LBS_FILES = {
    "gaussian_indices": "gaussian_indices.npy",
    "source_type_ids": "source_type_ids.npy",
    "canonical_xyz": "canonical_gaussian_positions.npy",
    "face_ids": "face_ids.npy",
    "barycentric": "barycentric.npy",
    "component_ids": "component_ids.npy",
    "semantic_region_ids": "semantic_region_ids.npy",
    "lbs_weights": "lbs_weights.npy",
    "attachment_valid": "attachment_valid.npy",
    "surface_distance": "surface_distance.npy",
}


class SurfaceLBSContractError(RuntimeError):
    """A fail-closed surface-attachment contract violation."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def radical_inverse(indices: np.ndarray, base: int) -> np.ndarray:
    """Exact copy of the frozen design-stage sequence."""
    values = np.zeros(len(indices), dtype=np.float64)
    factor = 1.0 / base
    remaining = indices.astype(np.int64).copy()
    while np.any(remaining > 0):
        values += factor * (remaining % base)
        remaining //= base
        factor /= base
    return values


def deterministic_surface_points(
    vertices: np.ndarray,
    faces: np.ndarray,
    count: int,
) -> dict[str, np.ndarray]:
    """Exact frozen area-CDF sampler used by the design-stage prototype."""
    triangles = vertices[faces].astype(np.float64)
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    cumulative = np.cumsum(areas, dtype=np.float64)
    targets = (
        (np.arange(count, dtype=np.float64) + 0.5)
        / count
        * cumulative[-1]
    )
    face_ids = np.searchsorted(
        cumulative,
        targets,
        side="right",
    ).astype(np.int64)
    sequence = np.arange(1, count + 1, dtype=np.int64)
    u = radical_inverse(sequence, 2)
    v = radical_inverse(sequence, 3)
    root = np.sqrt(u)
    barycentric = np.stack(
        [1.0 - root, root * (1.0 - v), root * v],
        axis=1,
    )
    points = np.einsum(
        "ni,nij->nj",
        barycentric,
        triangles[face_ids],
    )
    return {
        "face_ids": face_ids,
        "barycentric": barycentric,
        "points": points,
    }


def _shape_error(name: str, actual: tuple[int, ...], expected: tuple[int, ...]) -> None:
    code = "INVALID_LBS" if name == "lbs_weights" else "MISSING_ATTACHMENT"
    raise SurfaceLBSContractError(
        code,
        f"{name} has shape {actual}, expected {expected}",
    )


def validate_surface_attachment_arrays(
    arrays: dict[str, np.ndarray],
    *,
    expected_count: int | None,
    template_face_count: int,
    barycentric_tolerance: float = 1.0e-8,
    weight_sum_tolerance: float = 1.0e-6,
    nonnegative_tolerance: float = 1.0e-7,
    surface_distance_tolerance: float = 1.0e-7,
) -> dict[str, Any]:
    missing = sorted(set(SURFACE_LBS_FILES).difference(arrays))
    if missing:
        raise SurfaceLBSContractError(
            "MISSING_ATTACHMENT",
            f"missing arrays: {missing}",
        )

    count = int(len(arrays["canonical_xyz"]))
    if expected_count is not None and count != expected_count:
        raise SurfaceLBSContractError(
            "MISSING_ATTACHMENT",
            f"point count {count}, expected {expected_count}",
        )
    expected_shapes = {
        "gaussian_indices": (count,),
        "source_type_ids": (count,),
        "canonical_xyz": (count, 3),
        "face_ids": (count,),
        "barycentric": (count, 3),
        "component_ids": (count,),
        "semantic_region_ids": (count,),
        "lbs_weights": (count, 55),
        "attachment_valid": (count,),
        "surface_distance": (count,),
    }
    for name, expected in expected_shapes.items():
        actual = tuple(arrays[name].shape)
        if actual != expected:
            _shape_error(name, actual, expected)

    expected_indices = np.arange(count, dtype=np.int64)
    if not np.array_equal(
        arrays["gaussian_indices"].astype(np.int64, copy=False),
        expected_indices,
    ):
        raise SurfaceLBSContractError(
            "MISSING_ATTACHMENT",
            "gaussian_indices are not the identity sequence",
        )
    if not np.all(arrays["source_type_ids"] == 1):
        raise SurfaceLBSContractError(
            "MISSING_ATTACHMENT",
            "source_type_ids must all be SURFACE_FACE_SAMPLE=1",
        )
    if not np.all(arrays["attachment_valid"]):
        invalid_count = int(np.sum(~arrays["attachment_valid"].astype(bool)))
        raise SurfaceLBSContractError(
            "MISSING_ATTACHMENT",
            f"{invalid_count} attachment_valid entries are false",
        )

    face_ids = arrays["face_ids"].astype(np.int64, copy=False)
    if (
        np.any(face_ids < 0)
        or np.any(face_ids >= int(template_face_count))
    ):
        raise SurfaceLBSContractError(
            "INVALID_FACE_ID",
            "face ID is outside the frozen template range",
        )

    xyz = arrays["canonical_xyz"]
    barycentric = arrays["barycentric"]
    if not np.isfinite(xyz).all():
        raise SurfaceLBSContractError(
            "MISSING_ATTACHMENT",
            "canonical_xyz contains NaN or Inf",
        )
    if not np.isfinite(barycentric).all():
        raise SurfaceLBSContractError(
            "INVALID_BARYCENTRIC",
            "barycentric contains NaN or Inf",
        )
    bary_sum_error = float(
        np.max(np.abs(barycentric.sum(axis=1) - 1.0), initial=0.0)
    )
    bary_min = float(barycentric.min(initial=0.0))
    if (
        bary_sum_error > barycentric_tolerance
        or bary_min < -barycentric_tolerance
    ):
        raise SurfaceLBSContractError(
            "INVALID_BARYCENTRIC",
            f"sum error={bary_sum_error}, minimum={bary_min}",
        )

    weights = arrays["lbs_weights"]
    if not np.isfinite(weights).all():
        raise SurfaceLBSContractError(
            "INVALID_LBS",
            "weights contain NaN or Inf",
        )
    weight_sum_error = float(
        np.max(np.abs(weights.sum(axis=1, dtype=np.float64) - 1.0), initial=0.0)
    )
    weight_min = float(weights.min(initial=0.0))
    if (
        weight_sum_error > weight_sum_tolerance
        or weight_min < -nonnegative_tolerance
    ):
        raise SurfaceLBSContractError(
            "INVALID_LBS",
            f"sum error={weight_sum_error}, minimum={weight_min}",
        )

    surface_distance = arrays["surface_distance"]
    if (
        not np.isfinite(surface_distance).all()
        or np.any(np.abs(surface_distance) > surface_distance_tolerance)
    ):
        raise SurfaceLBSContractError(
            "MISSING_ATTACHMENT",
            "surface_distance exceeds the surface-only tolerance",
        )

    return {
        "point_count": count,
        "attachment_valid_count": int(
            arrays["attachment_valid"].astype(bool).sum()
        ),
        "off_surface_count": int(
            np.sum(np.abs(surface_distance) > surface_distance_tolerance)
        ),
        "invalid_face_count": 0,
        "invalid_barycentric_count": 0,
        "barycentric_sum_max_abs_error": bary_sum_error,
        "barycentric_minimum": bary_min,
        "weight_sum_max_abs_error": weight_sum_error,
        "weight_minimum": weight_min,
        "weights_finite": True,
    }


def load_surface_attachment_assets(
    root: str | Path,
    *,
    expected_count: int | None = None,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    root = Path(root)
    manifest_path = root / "surface_attachment_manifest.json"
    if not manifest_path.is_file():
        raise SurfaceLBSContractError(
            "MISSING_ATTACHMENT",
            f"manifest does not exist: {manifest_path}",
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arrays: dict[str, np.ndarray] = {}
    for name, relative in SURFACE_LBS_FILES.items():
        path = root / relative
        if not path.is_file():
            raise SurfaceLBSContractError(
                "MISSING_ATTACHMENT",
                f"asset does not exist: {path}",
            )
        arrays[name] = np.load(path, allow_pickle=False)
        if verify_hashes:
            expected = manifest["files"][relative]["sha256"]
            actual = sha256_file(path)
            if actual != expected:
                raise SurfaceLBSContractError(
                    "MISSING_ATTACHMENT",
                    f"SHA256 mismatch for {relative}: {actual} != {expected}",
                )
    validation = validate_surface_attachment_arrays(
        arrays,
        expected_count=expected_count,
        template_face_count=int(manifest["template"]["face_count"]),
        barycentric_tolerance=float(
            manifest["tolerances"]["barycentric_sum"]
        ),
        weight_sum_tolerance=float(
            manifest["tolerances"]["weight_sum"]
        ),
        nonnegative_tolerance=float(
            manifest["tolerances"]["weight_nonnegative"]
        ),
        surface_distance_tolerance=float(
            manifest["tolerances"]["surface_distance_meters"]
        ),
    )
    return {
        **arrays,
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "validation": validation,
    }
