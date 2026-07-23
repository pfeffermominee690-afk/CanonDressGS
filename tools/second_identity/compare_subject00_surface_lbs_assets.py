#!/usr/bin/env python3
"""Compare two fresh-process subject00 surface-LBS asset candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.surface_lbs_utils import (  # noqa: E402
    SURFACE_LBS_FILES,
    load_surface_attachment_assets,
    sha256_file,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root_a = args.run_a / "surface_lbs"
    root_b = args.run_b / "surface_lbs"
    a = load_surface_attachment_assets(root_a, expected_count=200000)
    b = load_surface_attachment_assets(root_b, expected_count=200000)
    arrays = {}
    all_exact = True
    for key in sorted(SURFACE_LBS_FILES):
        value_a = a[key]
        value_b = b[key]
        exact = bool(np.array_equal(value_a, value_b))
        max_abs = (
            float(
                np.max(
                    np.abs(
                        value_a.astype(np.float64)
                        - value_b.astype(np.float64)
                    ),
                    initial=0.0,
                )
            )
            if value_a.shape == value_b.shape
            else None
        )
        arrays[key] = {
            "exact": exact,
            "max_abs": max_abs,
            "shape_a": list(value_a.shape),
            "shape_b": list(value_b.shape),
            "dtype_a": str(value_a.dtype),
            "dtype_b": str(value_b.dtype),
        }
        all_exact = all_exact and exact

    manifest_a = root_a / "surface_attachment_manifest.json"
    manifest_b = root_b / "surface_attachment_manifest.json"
    manifest_exact = manifest_a.read_bytes() == manifest_b.read_bytes()
    template_files = {}
    template_exact = True
    for path_a in sorted(
        (args.run_a / "template").iterdir(),
        key=lambda item: item.name,
    ):
        path_b = args.run_b / "template" / path_a.name
        exact = path_b.is_file() and path_a.read_bytes() == path_b.read_bytes()
        template_files[path_a.name] = {
            "exact": exact,
            "sha256_a": sha256_file(path_a),
            "sha256_b": sha256_file(path_b),
        }
        template_exact = template_exact and exact

    passed = bool(
        all_exact
        and manifest_exact
        and template_exact
        and a["validation"]["attachment_valid_count"] == 200000
        and b["validation"]["attachment_valid_count"] == 200000
        and a["validation"]["off_surface_count"] == 0
        and b["validation"]["off_surface_count"] == 0
    )
    result = {
        "schema_version": (
            "subject00.mmlphuman.surface_sampler_repeatability.v1"
        ),
        "status": "PASS" if passed else (
            "SUBJECT00_SURFACE_ATTACHMENT_ASSET_NONDETERMINISTIC"
        ),
        "fresh_process_runs": ["run_a", "run_b"],
        "sample_count_a": a["validation"]["point_count"],
        "sample_count_b": b["validation"]["point_count"],
        "coverage_a": (
            a["validation"]["attachment_valid_count"]
            / a["validation"]["point_count"]
        ),
        "coverage_b": (
            b["validation"]["attachment_valid_count"]
            / b["validation"]["point_count"]
        ),
        "off_surface_count_a": a["validation"]["off_surface_count"],
        "off_surface_count_b": b["validation"]["off_surface_count"],
        "arrays": arrays,
        "arrays_exact": all_exact,
        "manifest_exact": manifest_exact,
        "manifest_sha256_a": sha256_file(manifest_a),
        "manifest_sha256_b": sha256_file(manifest_b),
        "template_files": template_files,
        "template_exact": template_exact,
        "publish_selection": "run_a" if passed else None,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_created": 0,
        "training_checkpoint_writes": 0,
        "PAPER_FINAL": 0,
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
