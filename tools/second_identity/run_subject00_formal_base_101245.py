#!/usr/bin/env python3
"""Bind the frozen Subject00 formal runner to the 101245-step base attempt.

The imported strict-split runner owns the training algorithm. This wrapper only
loads its already sealed protocol, then rebases execution identity and output
paths to the user-authorized Subject00 formal base attempt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.second_identity import run_subject00_formal_strict_split as formal


TASK_ID = "AAAI27-SUBJECT00-FORMAL-BASE-101245-001"
BRANCH = "research/subject00-formal-base-101245-execution-20260726"
FORMAL_BASE_OUTPUT_ROOT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-FORMAL-BASE-101245-001"
)
FORMAL_BASE_ATTEMPT_ROOT = FORMAL_BASE_OUTPUT_ROOT / "attempt_001"
DELETION_EXECUTION_RECORD = (
    REPO_ROOT
    / "paper_protocol"
    / "storage"
    / "avatarrex_cloud_duplicate_deletion_execution_20260726.json"
)
LEGACY_REQUEST_ALIAS = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-FORMAL-STRICT-SPLIT-BASE-001"
)
AVATARREX_DELETED_PATH = Path("/root/autodl-tmp/avatarrex_lbn1.7z")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("preflight", "train", "roundtrip", "evaluate"),
        required=True,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/datasets/"
            "thuman4_second_identity_staging/subject00"
        ),
    )
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/datasets/"
            "thuman4_second_identity_staging/derived_assets/subject00"
        ),
    )
    parser.add_argument(
        "--availability-manifest",
        type=Path,
        default=Path(
            "/root/autodl-tmp/datasets/"
            "thuman4_second_identity_staging/reports/"
            "SUBJECT00_VALID_FRAME_CAMERA_MANIFEST.json"
        ),
    )
    parser.add_argument("--attempt-root", type=Path, default=FORMAL_BASE_ATTEMPT_ROOT)
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--interruption-reason")
    parser.add_argument("--resume-evaluation", action="store_true")
    return parser.parse_args()


def require_deletion_execution_gate() -> None:
    record = read_json(DELETION_EXECUTION_RECORD)
    if record["deletion_authorized"] is not True:
        raise RuntimeError("AvatarReX deletion is not authorized")
    if record["deletion_executed"] is not True:
        raise RuntimeError("AvatarReX cloud duplicate deletion is not executed")
    if record["cloud_duplicate"]["target_path"] != str(AVATARREX_DELETED_PATH):
        raise RuntimeError("AvatarReX deletion target changed")
    if record["cloud_duplicate"]["exists_after_delete"] is not False:
        raise RuntimeError("AvatarReX deletion record does not prove absence")
    if Path("/root/autodl-tmp").exists() and AVATARREX_DELETED_PATH.exists():
        raise RuntimeError("AvatarReX cloud duplicate still exists")
    if record["storage"]["status"] != "PASS_AFTER_DELETION_STORAGE_RECHECK":
        raise RuntimeError("post-deletion storage gate did not pass")


def apply_formal_base_runtime_binding() -> None:
    formal.TASK_ID = TASK_ID
    formal.TARGET_BRANCH = BRANCH
    formal.CANONICAL_FORMAL_OUTPUT_ROOT = FORMAL_BASE_OUTPUT_ROOT
    formal.FORMAL_OUTPUT_ROOT = FORMAL_BASE_OUTPUT_ROOT
    formal.ATTEMPT_ROOT = FORMAL_BASE_ATTEMPT_ROOT
    formal.REQUEST_LEVEL_OUTPUT_ROOT_ALIAS = LEGACY_REQUEST_ALIAS
    formal.CONFLICTING_REQUEST_OUTPUT_ROOT = LEGACY_REQUEST_ALIAS
    formal.OUTPUT_ROOT_CLASSIFICATION = "SUBJECT00_FORMAL_BASE_101245_OUTPUT_ROOT_FREEZE"


def load_protocol_before_rebinding(availability_manifest: Path) -> dict:
    return formal.load_protocol(availability_manifest)


def main() -> int:
    args = parse_args()
    if args.attempt_root != FORMAL_BASE_ATTEMPT_ROOT:
        raise RuntimeError(f"attempt root must be frozen path: {FORMAL_BASE_ATTEMPT_ROOT}")
    require_deletion_execution_gate()
    protocol = load_protocol_before_rebinding(args.availability_manifest)
    apply_formal_base_runtime_binding()
    formal.configure_runtime_modules()

    if args.phase == "preflight":
        result = formal.preflight(args, protocol, allow_attempt=False)
        result["task_id"] = TASK_ID
        result["formal_base_binding"] = {
            "branch": BRANCH,
            "output_root": str(FORMAL_BASE_OUTPUT_ROOT),
            "attempt_root": str(FORMAL_BASE_ATTEMPT_ROOT),
            "deletion_execution_record": str(DELETION_EXECUTION_RECORD),
        }
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.phase == "train":
        return formal.train_phase(args, protocol)
    if args.phase == "roundtrip":
        return formal.roundtrip_phase(args, protocol)
    return formal.evaluate_phase(args, protocol)


if __name__ == "__main__":
    raise SystemExit(main())
