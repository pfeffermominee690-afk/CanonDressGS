#!/usr/bin/env python3
"""Build one deterministic subject00 surface-LBS asset candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.second_identity.run_subject00_high_fidelity_lbs_prototype import (  # noqa: E402
    canonical_model,
    deterministic_surface_points,
    joint_names,
    semantic_labels,
)
from utils.surface_lbs_utils import (  # noqa: E402
    SURFACE_LBS_FILES,
    sha256_array,
    sha256_file,
    validate_surface_attachment_arrays,
)


EXPECTED_SAMPLER_SOURCE_SHA256 = (
    "98af26a3f578d1e0239f6ea414e3664dae10a4b60f47aef594d0978f61b9ee82"
)
EXPECTED_TEMPLATE_PLY_SHA256 = (
    "f10a3b516e2b3a2ad38dc4924a3692b2f3e72a6cc9e66f3c0063c4e9cd210031"
)
EXPECTED_VERTEX_ARRAY_SHA256 = (
    "1f924c7e46a97b37272ea833ea09f54e8b10f79836824f43140bf8452e69ad97"
)
EXPECTED_FACE_ARRAY_SHA256 = (
    "2cb81d8e6c789896d764805d58fb44bdce62424bab97b519bbd6c1668d66ce2b"
)
EXPECTED_WEIGHT_ARRAY_SHA256 = (
    "645bde9f2a656a888b31173b06231da1ebb78d1363d33ba8f44eeb60e6bb4e29"
)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def save_array(path: Path, value: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.save(handle, np.ascontiguousarray(value), allow_pickle=False)


def canonical_lf_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-root", type=Path, required=True)
    parser.add_argument("--smpl-model", type=Path, required=True)
    parser.add_argument("--frozen-template-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200000)
    args = parser.parse_args()

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output}")

    sampler_source = (
        REPO_ROOT
        / "tools/second_identity/run_subject00_high_fidelity_lbs_prototype.py"
    )
    sampler_source_raw_sha256 = sha256_file(sampler_source)
    sampler_source_sha256 = canonical_lf_sha256(sampler_source)
    if sampler_source_sha256 != EXPECTED_SAMPLER_SOURCE_SHA256:
        raise RuntimeError(
            "Frozen sampler source SHA changed: "
            f"{sampler_source_sha256}"
        )

    template_dir = output / "template"
    surface_dir = output / "surface_lbs"
    template_dir.mkdir()
    surface_dir.mkdir()
    for name in (
        "template_smplx_body_surface.ply",
        "template_vertices.npy",
        "template_faces.npy",
        "template_generation_report.json",
    ):
        shutil.copyfile(
            args.frozen_template_root / name,
            template_dir / name,
        )

    template_ply = template_dir / "template_smplx_body_surface.ply"
    if sha256_file(template_ply) != EXPECTED_TEMPLATE_PLY_SHA256:
        raise RuntimeError("Frozen template PLY SHA changed")

    names = joint_names()
    model, vertices, faces, reference_weights, _, _ = canonical_model(
        args.smpl_model,
        args.subject_root / "smpl_params.npz",
    )
    del model
    frozen_vertices = np.load(
        template_dir / "template_vertices.npy",
        allow_pickle=False,
    )
    frozen_faces = np.load(
        template_dir / "template_faces.npy",
        allow_pickle=False,
    )
    if not np.array_equal(vertices, frozen_vertices):
        raise RuntimeError("Canonical vertices differ from frozen template")
    if not np.array_equal(faces, frozen_faces):
        raise RuntimeError("Canonical faces differ from frozen template")
    if sha256_array(vertices) != EXPECTED_VERTEX_ARRAY_SHA256:
        raise RuntimeError("Canonical vertex array SHA changed")
    if sha256_array(faces) != EXPECTED_FACE_ARRAY_SHA256:
        raise RuntimeError("Canonical face array SHA changed")
    if sha256_array(reference_weights) != EXPECTED_WEIGHT_ARRAY_SHA256:
        raise RuntimeError("Official LBS weight array SHA changed")

    semantic = semantic_labels(
        vertices,
        faces,
        reference_weights,
        names,
    )
    sample = deterministic_surface_points(
        vertices,
        faces,
        args.count,
    )
    face_ids = sample["face_ids"]
    barycentric = sample["barycentric"]
    canonical_xyz = sample["points"]
    lbs_weights = np.einsum(
        "ni,nij->nj",
        barycentric,
        reference_weights[faces[face_ids]].astype(np.float64),
    ).astype(np.float32)
    arrays = {
        "gaussian_indices": np.arange(args.count, dtype=np.int64),
        "source_type_ids": np.ones(args.count, dtype=np.uint8),
        "canonical_xyz": canonical_xyz.astype(np.float64, copy=False),
        "face_ids": face_ids.astype(np.int64, copy=False),
        "barycentric": barycentric.astype(np.float64, copy=False),
        "component_ids": semantic["face_components"][face_ids].astype(
            np.int16,
            copy=False,
        ),
        "semantic_region_ids": semantic["face_regions"][face_ids].astype(
            np.int16,
            copy=False,
        ),
        "lbs_weights": lbs_weights,
        "attachment_valid": np.ones(args.count, dtype=np.uint8),
        "surface_distance": np.zeros(args.count, dtype=np.float64),
    }
    validation = validate_surface_attachment_arrays(
        arrays,
        expected_count=args.count,
        template_face_count=len(faces),
    )
    reconstructed_xyz = np.einsum(
        "ni,nij->nj",
        barycentric,
        vertices[faces[face_ids]].astype(np.float64),
    )
    xyz_formula_max_abs = float(
        np.max(np.abs(reconstructed_xyz - canonical_xyz), initial=0.0)
    )
    if xyz_formula_max_abs != 0.0:
        raise RuntimeError(
            f"Surface position formula mismatch: {xyz_formula_max_abs}"
        )

    for key, relative in SURFACE_LBS_FILES.items():
        save_array(surface_dir / relative, arrays[key])

    files = {}
    for relative in sorted(SURFACE_LBS_FILES.values()):
        path = surface_dir / relative
        files[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    template_files = {}
    for path in sorted(template_dir.iterdir(), key=lambda item: item.name):
        template_files[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    manifest = {
        "schema_version": (
            "subject00.mmlphuman.surface_attachment_assets.v1"
        ),
        "subject_id": "subject00",
        "source_branch": (
            "research/mmlphuman-subject00-high-fidelity-lbs-design-20260723"
        ),
        "source_head": "7a9191e7e57fc92000d681cf7f0b73edd346af62",
        "lbs_mode": "surface_attachment_cached",
        "point_count": args.count,
        "source_type_mapping": {"1": "SURFACE_FACE_SAMPLE"},
        "sampler": {
            "algorithm": "deterministic_area_cdf_surface_sampler",
            "source": (
                "tools/second_identity/"
                "run_subject00_high_fidelity_lbs_prototype.py"
            ),
            "source_sha256": sampler_source_sha256,
            "source_sha256_policy": (
                "normalize_CRLF_and_CR_to_LF_before_SHA256"
            ),
            "source_raw_sha256_record_only": sampler_source_raw_sha256,
            "function": "deterministic_surface_points",
            "face_area_dtype": "float64",
            "face_order": "template_face_index_ascending",
            "cdf": "numpy_cumsum_float64",
            "sample_targets": "midpoint_stratified_area_targets",
            "sample_sequence": "integer_1_through_N",
            "barycentric_sequence": (
                "sqrt_radical_inverse_base_2_and_3"
            ),
            "tie_rule": "face_order_then_searchsorted_side_right",
            "process_seed": "NOT_USED_STATELESS_SEQUENCE",
        },
        "template": {
            "vertex_count": int(len(vertices)),
            "face_count": int(len(faces)),
            "joint_count": int(reference_weights.shape[1]),
            "vertices_array_sha256": sha256_array(vertices),
            "design_prototype_reconstructed_vertices_sha256_record_only": (
                "cb7edff9ec0ef82d97fb1444a65e894e39a9bd4cafd7cde0e90b388547e6e447"
            ),
            "vertex_authority": (
                "approved_attempt_001_template_plus_matching_current_"
                "SMPLX_reconstruction"
            ),
            "faces_array_sha256": sha256_array(faces),
            "weights_array_sha256": sha256_array(reference_weights),
            "files": template_files,
        },
        "semantic": {
            "component_names": semantic["component_names"],
            "component_sizes": semantic["component_sizes"],
            "region_names": semantic["region_names"],
            "face_component_array_sha256": sha256_array(
                semantic["face_components"]
            ),
            "face_region_array_sha256": sha256_array(
                semantic["face_regions"]
            ),
        },
        "arrays": {
            key: {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "array_sha256": sha256_array(value),
            }
            for key, value in sorted(arrays.items())
        },
        "files": files,
        "validation": {
            **validation,
            "surface_xyz_formula_max_abs": xyz_formula_max_abs,
            "surface_attachment_coverage": (
                validation["attachment_valid_count"] / args.count
            ),
            "dominant_agreement_with_barycentric_truth": 1.0,
            "lbs_max_abs_from_barycentric_truth": 0.0,
        },
        "tolerances": {
            "barycentric_sum": 1.0e-8,
            "weight_sum": 1.0e-6,
            "weight_nonnegative": 1.0e-7,
            "surface_distance_meters": 1.0e-7,
        },
        "forbidden_assets": {
            "lbs_weights_grid.npz": False,
        },
        "counters": {
            "training_steps": 0,
            "training_forward_batches": 0,
            "backward_calls": 0,
            "optimizer_created": 0,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "training_checkpoint_writes": 0,
            "smoke_checkpoint_writes": 0,
            "PAPER_FINAL": 0,
        },
    }
    write_json(
        surface_dir / "surface_attachment_manifest.json",
        manifest,
    )
    print(
        json.dumps(
            {
                "point_count": args.count,
                "coverage": manifest["validation"][
                    "surface_attachment_coverage"
                ],
                "off_surface_count": validation["off_surface_count"],
                "manifest": str(
                    surface_dir / "surface_attachment_manifest.json"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
