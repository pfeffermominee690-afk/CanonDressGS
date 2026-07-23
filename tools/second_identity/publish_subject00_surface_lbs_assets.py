#!/usr/bin/env python3
"""Atomically publish a verified subject00 surface-LBS candidate."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.surface_lbs_utils import (  # noqa: E402
    load_surface_attachment_assets,
    sha256_file,
)


def snapshot_files(root: Path) -> dict[str, dict[str, object]]:
    return {
        path.relative_to(root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(
            (item for item in root.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(root).as_posix(),
        )
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--repeatability", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    repeatability = json.loads(args.repeatability.read_text(encoding="utf-8"))
    if repeatability["status"] != "PASS":
        raise RuntimeError("Repeatability gate did not pass")
    if repeatability["publish_selection"] != "run_a":
        raise RuntimeError("Frozen publish selection is not run_a")
    if args.target.exists():
        raise RuntimeError(f"Formal target already exists: {args.target}")

    candidate = args.candidate.resolve()
    load_surface_attachment_assets(
        candidate / "surface_lbs",
        expected_count=200000,
    )
    candidate_files = snapshot_files(candidate)
    staging = args.target.parent / (
        f".{args.target.name}.surface_lbs_publish_staging"
    )
    if staging.exists():
        raise RuntimeError(f"Publish staging path already exists: {staging}")
    shutil.copytree(candidate, staging)
    staging_files = snapshot_files(staging)
    if candidate_files != staging_files:
        raise RuntimeError("Staging copy hash verification failed")
    load_surface_attachment_assets(
        staging / "surface_lbs",
        expected_count=200000,
    )
    os.replace(staging, args.target)
    published_files = snapshot_files(args.target)
    if candidate_files != published_files:
        raise RuntimeError("Post-publish hash verification failed")

    report = {
        "schema_version": (
            "subject00.mmlphuman.surface_lbs_atomic_publish.v1"
        ),
        "status": "PASS",
        "candidate": str(candidate),
        "target": str(args.target),
        "selection": "run_a",
        "atomic_operation": "os.replace_same_filesystem_directory",
        "file_count": len(published_files),
        "total_bytes": sum(
            int(item["bytes"]) for item in published_files.values()
        ),
        "files": published_files,
        "forbidden_lbs_grid_present": (
            args.target / "surface_lbs/lbs_weights_grid.npz"
        ).exists(),
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_created": 0,
        "training_checkpoint_writes": 0,
        "PAPER_FINAL": 0,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
