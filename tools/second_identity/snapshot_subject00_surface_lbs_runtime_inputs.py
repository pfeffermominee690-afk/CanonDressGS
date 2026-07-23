#!/usr/bin/env python3
"""Snapshot immutable inputs for the subject00 surface-LBS runtime task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from snapshot_subject00_lbs_design_inputs import (
    RUNTIME_PATHS,
    canonical_sha256,
    git,
    sha256_file,
    snapshot_tree,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--attempt-001", type=Path, required=True)
    parser.add_argument("--attempt-002", type=Path, required=True)
    parser.add_argument("--attempt-003", type=Path, required=True)
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
    strict = {}
    for name in ("subject00_novel_view_split_v2.json", "subject00_novel_pose_split_v2.json"):
        path = repo / "paper_protocol/second_identity" / name
        data = json.loads(path.read_text(encoding="utf-8"))
        strict[name] = {"file_sha256": sha256_file(path), "split_sha256": data["split_sha256"]}

    runtime_files = [
        {"path": relative, "bytes": (repo / relative).stat().st_size, "sha256": sha256_file(repo / relative)}
        for relative in RUNTIME_PATHS
    ]
    raw_sentinels = {}
    for name in ("smpl_params.npz", "calibration.json", "missing_img_files.txt", "missing_msk_files.txt"):
        path = args.subject00_root / name
        raw_sentinels[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}

    output = {
        "schema_version": "subject00.mmlphuman.surface_lbs_runtime_immutable_snapshot.v1",
        "repo": {
            "root": str(repo),
            "head": git(repo, "rev-parse", "HEAD"),
            "branch": git(repo, "branch", "--show-current"),
            "status_porcelain": git(repo, "status", "--porcelain"),
        },
        "attempt_001": snapshot_tree(args.attempt_001),
        "attempt_002": snapshot_tree(args.attempt_002),
        "attempt_003": snapshot_tree(args.attempt_003),
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
            "smoke_checkpoint_writes": 0,
            "PAPER_FINAL": 0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({
        "attempt_001": {key: output["attempt_001"][key] for key in ("file_count", "total_bytes", "aggregate_sha256")},
        "attempt_002": {key: output["attempt_002"][key] for key in ("file_count", "total_bytes", "aggregate_sha256")},
        "attempt_003": {key: output["attempt_003"][key] for key in ("file_count", "total_bytes", "aggregate_sha256")},
        "runtime_closure": output["runtime_closure"]["aggregate_sha256"],
        "formal_target_exists": output["formal_target_exists"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
