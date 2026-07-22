#!/usr/bin/env python3
"""Snapshot immutable inputs for the subject00 high-fidelity LBS design task."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


RUNTIME_PATHS = (
    "config/subject02_formal_800k.yaml",
    "scene/dataset.py",
    "scene/gaussian_model.py",
    "scene/mlp.py",
    "scene/net_vis.py",
    "scene/scene.py",
    "train.py",
    "utils/config_utils.py",
    "utils/general_utils.py",
    "utils/graphics_utils.py",
    "utils/image_utils.py",
    "utils/loss_utils.py",
    "utils/net_utils.py",
    "utils/sh_utils.py",
    "utils/smpl_utils.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def snapshot_tree(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.relative_to(root).as_posix()):
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    identity = [{"path": item["path"], "bytes": item["bytes"], "sha256": item["sha256"]} for item in files]
    return {
        "root": str(root),
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "aggregate_sha256": canonical_sha256(identity),
        "files": files,
    }


def git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--attempt-001", type=Path, required=True)
    parser.add_argument("--attempt-002", type=Path, required=True)
    parser.add_argument("--subject00-root", type=Path, required=True)
    parser.add_argument("--availability-manifest", type=Path, required=True)
    parser.add_argument("--subject02-template", type=Path, required=True)
    parser.add_argument("--subject02-lbs", type=Path, required=True)
    parser.add_argument("--subject02-checkpoint", type=Path, required=True)
    parser.add_argument("--formal-target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    availability = json.loads(args.availability_manifest.read_text(encoding="utf-8"))
    strict_paths = (
        repo / "paper_protocol/second_identity/subject00_novel_view_split_v2.json",
        repo / "paper_protocol/second_identity/subject00_novel_pose_split_v2.json",
    )
    strict = {}
    for path in strict_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        strict[path.name] = {
            "file_sha256": sha256_file(path),
            "split_sha256": data["split_sha256"],
        }
    runtime_files = [
        {
            "path": relative,
            "bytes": (repo / relative).stat().st_size,
            "sha256": sha256_file(repo / relative),
        }
        for relative in RUNTIME_PATHS
    ]
    raw_sentinels = {}
    for name in (
        "smpl_params.npz",
        "calibration.json",
        "missing_img_files.txt",
        "missing_msk_files.txt",
    ):
        path = args.subject00_root / name
        raw_sentinels[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}

    output = {
        "schema_version": "subject00.mmlphuman.lbs_design_immutable_snapshot.v1",
        "repo": {
            "root": str(repo),
            "head": git(repo, "rev-parse", "HEAD"),
            "branch": git(repo, "branch", "--show-current"),
            "status_porcelain": git(repo, "status", "--porcelain"),
        },
        "attempt_001": snapshot_tree(args.attempt_001),
        "attempt_002": snapshot_tree(args.attempt_002),
        "subject00_raw": {
            "frozen_fingerprint": "2c0f894f70d944fa8d6ebd48ab1188b78cb19b673f0b7c006f1922a592b1ea7b",
            "sentinels": raw_sentinels,
            "availability_file_sha256": sha256_file(args.availability_manifest),
            "availability_content_sha256": availability["summary"]["manifest_sha256"],
        },
        "subject02": {
            "template_sha256": sha256_file(args.subject02_template),
            "lbs_sha256": sha256_file(args.subject02_lbs),
            "checkpoint_sha256": sha256_file(args.subject02_checkpoint),
        },
        "runtime_closure": {
            "file_count": len(runtime_files),
            "aggregate_sha256": canonical_sha256(runtime_files),
            "files": runtime_files,
        },
        "strict_splits": strict,
        "formal_target_exists": args.formal_target.exists(),
        "counters": {
            "training_steps": 0,
            "training_forward_batches": 0,
            "backward_calls": 0,
            "optimizer_created": 0,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "training_checkpoint_writes": 0,
            "PAPER_FINAL": 0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({
        "attempt_001": {key: output["attempt_001"][key] for key in ("file_count", "total_bytes", "aggregate_sha256")},
        "attempt_002": {key: output["attempt_002"][key] for key in ("file_count", "total_bytes", "aggregate_sha256")},
        "runtime_closure": output["runtime_closure"]["aggregate_sha256"],
        "formal_target_exists": output["formal_target_exists"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
