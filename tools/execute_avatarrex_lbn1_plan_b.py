#!/usr/bin/env python3
"""Execute and verify the AvatarReX LBN1 Plan B targeted extraction.

This script is intentionally narrow. It extracts only the frozen allowlist
members for the Plan B canary and writes attempt-scoped audit artifacts. It
does not create a formal raw-data root, preprocess, train, generate garments,
or edit paper files.
"""

from __future__ import annotations

import argparse
import binascii
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST_DIR = PROJECT_ROOT / "paper_protocol" / "avatarrex_lbn1_plan_b" / "manifests"
ALLOWLIST_TXT = ALLOWLIST_DIR / "avatarrex_lbn1_plan_b_allowlist.txt"
ALLOWLIST_JSON = ALLOWLIST_DIR / "avatarrex_lbn1_plan_b_allowlist.json"

TASK_ID = "AAAI27-AVATARREX-LBN1-TARGETED-EXTRACTION-PLAN-B-001"
CONTRACT_TASK_ID = "AAAI27-AVATARREX-LBN1-ARCHIVE-STRUCTURE-AND-EXTRACTION-CONTRACT-001"
PLAN_ID = "PLAN_B_CANARY_001"
ATTEMPT_ID = "attempt_001"
CONTRACT_ATTEMPT_ID = "AAAI27-AVATARREX-LBN1-TARGETED-EXTRACTION-001"

SOURCE_BRANCH = "research/external-baseline-feasibility-from-bundle-rerun-20260726"
SOURCE_HEAD = "572637f08aef21fde35dcd7f8879ff74549c2afc"
NEW_BRANCH = "research/avatarrex-lbn1-targeted-extraction-plan-b-20260726"
WINDOWS_WORKTREE = r"E:\model_train\canondressgs_avatarrex_lbn1_targeted_extraction_plan_b"
CLOUD_WORKTREE = "/root/autodl-tmp/canondressgs_work/worktrees/canondressgs_avatarrex_lbn1_targeted_extraction_plan_b"

ARCHIVE_PATH = PurePosixPath("/root/autodl-tmp/avatarrex_lbn1.7z")
ARCHIVE_BYTES = 12_569_755_256
ARCHIVE_SHA256 = "531bd1c71ad9b35f6ae0e2595ee531aa7ba1f83c242f18f2d0505b2dcd5fbcc1"
ARCHIVE_LISTING_SLT_SHA256 = "d33cb311687a1034c37e795d26966f1a93b16bc5e7de3e96cce28b387081fc70"

ALLOWLIST_TXT_SHA256 = "a8e2825294fc2cf02422669a5c9e9ec2f660e9ec48cda61dafb7ff804c20ab70"
ALLOWLIST_JSON_SHA256 = "943e1e98ff8b7082150d4007d17c5ef34e0503f99ac5d75b555b44c2596599ea"
ALLOWLIST_COUNT = 1602
EXPECTED_UNCOMPRESSED_BYTES = 501_198_133

SAFETY_FLOOR_BYTES = 40_802_189_312
PROJECT_AUDIT_RESERVE_BYTES = 2 * 1024 * 1024 * 1024
CONTRACT_OVERHEAD_BYTES = 536_870_912
DF_TARGET = PurePosixPath("/root/autodl-tmp")
STAGING_ROOT = PurePosixPath("/root/autodl-tmp/datasets/avatarrex_lbn1_staging/PLAN_B_CANARY_001/attempt_001")
AUDIT_ROOT = PurePosixPath("/root/autodl-tmp/datasets/avatarrex_lbn1_staging/PLAN_B_CANARY_001/attempt_001_audit")
FORMAL_DATA_ROOT = PurePosixPath("/root/autodl-tmp/datasets/avatarrex_lbn1")

IDENTITY = "implicit_single_identity_lbn1"
SEQUENCE = "implicit_single_sequence"
CAMERA_IDS = (
    "22053908",
    "22053926",
    "22010708",
    "22010710",
    "22010716",
    "22010714",
    "22070935",
    "22053923",
)
FRAME_IDS = (
    "00000000",
    "00000019",
    "00000038",
    "00000057",
    "00000076",
    "00000095",
    "00000115",
    "00000134",
    "00000153",
    "00000172",
    "00000191",
    "00000211",
    "00000230",
    "00000249",
    "00000268",
    "00000287",
    "00000307",
    "00000326",
    "00000345",
    "00000364",
    "00000383",
    "00000403",
    "00000422",
    "00000441",
    "00000460",
    "00000479",
    "00000498",
    "00000518",
    "00000537",
    "00000556",
    "00000575",
    "00000594",
    "00000614",
    "00000633",
    "00000652",
    "00000671",
    "00000690",
    "00000710",
    "00000729",
    "00000748",
    "00000767",
    "00000786",
    "00000806",
    "00000825",
    "00000844",
    "00000863",
    "00000882",
    "00000902",
    "00000921",
    "00000940",
    "00000959",
    "00000978",
    "00000997",
    "00001017",
    "00001036",
    "00001055",
    "00001074",
    "00001093",
    "00001113",
    "00001132",
    "00001151",
    "00001170",
    "00001189",
    "00001209",
    "00001228",
    "00001247",
    "00001266",
    "00001285",
    "00001305",
    "00001324",
    "00001343",
    "00001362",
    "00001381",
    "00001401",
    "00001420",
    "00001439",
    "00001458",
    "00001477",
    "00001496",
    "00001516",
    "00001535",
    "00001554",
    "00001573",
    "00001592",
    "00001612",
    "00001631",
    "00001650",
    "00001669",
    "00001688",
    "00001708",
    "00001727",
    "00001746",
    "00001765",
    "00001784",
    "00001804",
    "00001823",
    "00001842",
    "00001861",
    "00001880",
    "00001900",
)

FINAL_CLASSIFICATION_PASS = "AVATARREX_LBN1_PLAN_B_TARGETED_EXTRACTION_PASS_PENDING_LOADER_REVIEW"
FINAL_CLASSIFICATION_ALLOWLIST_BLOCKED = "AVATARREX_PLAN_B_BLOCKED_BY_AMBIGUOUS_EXTRACTION_ALLOWLIST"
FINAL_CLASSIFICATION_STORAGE_FAIL = "AVATARREX_PLAN_B_STORAGE_GATE_FAIL"
FINAL_CLASSIFICATION_CONTRACT_FAIL = "AVATARREX_PLAN_B_EXTRACTION_CONTRACT_FAIL"
FINAL_CLASSIFICATION_ARCHIVE_MUTATION = "AVATARREX_PLAN_B_ARCHIVE_IMMUTABILITY_VIOLATION"

NEXT_TASK_SUCCESS = "USER_REVIEW_AVATARREX_PLAN_B_EXTRACTION_AND_AUTHORIZE_LOADER_CANARY"

MEDIA_RE = re.compile(r"^(?P<camera>\d{8})/(?:(?P<pha>mask/pha)/)?(?P<frame>\d{8})\.jpg$")


class PlanBFailure(RuntimeError):
    """A fatal Plan B contract failure with a stable final classification."""

    def __init__(self, message: str, classification: str, failure_kind: str) -> None:
        super().__init__(message)
        self.classification = classification
        self.failure_kind = failure_kind


@dataclass(frozen=True)
class Member:
    path: str
    bytes: int
    crc32: str
    sha256: str
    camera: str | None
    frame: str | None
    asset_type: str


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_and_crc32(path: Path) -> tuple[str, str, int]:
    sha = hashlib.sha256()
    crc = 0
    total = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
            crc = binascii.crc32(chunk, crc)
            total += len(chunk)
    return sha.hexdigest(), f"{crc & 0xFFFFFFFF:08X}", total


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def classify_member(path_text: str) -> tuple[str | None, str | None, str]:
    if path_text == "calibration_full.json":
        return None, None, "calibration"
    if path_text == "smpl_params.npz":
        return None, None, "smpl_params"
    match = MEDIA_RE.match(path_text)
    if not match:
        return None, None, "unknown"
    asset_type = "pha" if match.group("pha") else "rgb"
    return match.group("camera"), match.group("frame"), asset_type


def assert_safe_archive_member(path_text: str) -> None:
    if not path_text or path_text.strip() != path_text:
        raise PlanBFailure(
            f"Unsafe allowlist member spelling: {path_text!r}",
            FINAL_CLASSIFICATION_ALLOWLIST_BLOCKED,
            "ALLOWLIST_UNSAFE_MEMBER_SPELLING",
        )
    if "\\" in path_text or ":" in path_text:
        raise PlanBFailure(
            f"Unsafe allowlist member contains backslash or drive marker: {path_text}",
            FINAL_CLASSIFICATION_ALLOWLIST_BLOCKED,
            "ALLOWLIST_UNSAFE_MEMBER_SEPARATOR",
        )
    if any(token in path_text for token in ("*", "?", "[")):
        raise PlanBFailure(
            f"Glob-like allowlist member is forbidden: {path_text}",
            FINAL_CLASSIFICATION_ALLOWLIST_BLOCKED,
            "ALLOWLIST_GLOB_FORBIDDEN",
        )
    pure = Path(path_text)
    if pure.is_absolute() or ".." in pure.parts:
        raise PlanBFailure(
            f"Absolute or parent-traversal allowlist member is forbidden: {path_text}",
            FINAL_CLASSIFICATION_ALLOWLIST_BLOCKED,
            "ALLOWLIST_TRAVERSAL_FORBIDDEN",
        )


def load_members() -> list[Member]:
    if sha256_file(ALLOWLIST_TXT) != ALLOWLIST_TXT_SHA256:
        raise PlanBFailure(
            "Copied allowlist text hash does not match the frozen contract.",
            FINAL_CLASSIFICATION_ALLOWLIST_BLOCKED,
            "ALLOWLIST_TXT_HASH_MISMATCH",
        )
    if sha256_file(ALLOWLIST_JSON) != ALLOWLIST_JSON_SHA256:
        raise PlanBFailure(
            "Copied allowlist JSON hash does not match the frozen contract.",
            FINAL_CLASSIFICATION_ALLOWLIST_BLOCKED,
            "ALLOWLIST_JSON_HASH_MISMATCH",
        )

    payload = json.loads(ALLOWLIST_JSON.read_text(encoding="utf-8"))
    raw_members = payload.get("members")
    if not isinstance(raw_members, list):
        raise PlanBFailure(
            "Allowlist JSON has no members list.",
            FINAL_CLASSIFICATION_ALLOWLIST_BLOCKED,
            "ALLOWLIST_SCHEMA_MISMATCH",
        )

    members: list[Member] = []
    for row in raw_members:
        path_text = str(row["path"])
        assert_safe_archive_member(path_text)
        camera, frame, asset_type = classify_member(path_text)
        members.append(
            Member(
                path=path_text,
                bytes=int(row["bytes"]),
                crc32=str(row["crc32"]).upper(),
                sha256=str(row["sha256"]).lower(),
                camera=camera,
                frame=frame,
                asset_type=asset_type,
            )
        )
    validate_manifest_members(members)
    return members


def validate_manifest_members(members: list[Member]) -> dict[str, Any]:
    paths = [member.path for member in members]
    duplicates = sorted({path for path in paths if paths.count(path) > 1})
    if duplicates:
        raise PlanBFailure(
            f"Allowlist contains duplicate members: {duplicates[:5]}",
            FINAL_CLASSIFICATION_ALLOWLIST_BLOCKED,
            "ALLOWLIST_DUPLICATE_PATHS",
        )
    if len(members) != ALLOWLIST_COUNT:
        raise PlanBFailure(
            f"Allowlist count {len(members)} does not equal {ALLOWLIST_COUNT}.",
            FINAL_CLASSIFICATION_ALLOWLIST_BLOCKED,
            "ALLOWLIST_COUNT_MISMATCH",
        )
    total = sum(member.bytes for member in members)
    if total != EXPECTED_UNCOMPRESSED_BYTES:
        raise PlanBFailure(
            f"Allowlist bytes {total} do not equal {EXPECTED_UNCOMPRESSED_BYTES}.",
            FINAL_CLASSIFICATION_ALLOWLIST_BLOCKED,
            "ALLOWLIST_BYTE_TOTAL_MISMATCH",
        )

    unknown = [member.path for member in members if member.asset_type == "unknown"]
    if unknown:
        raise PlanBFailure(
            f"Allowlist contains unexpected member shapes: {unknown[:5]}",
            FINAL_CLASSIFICATION_ALLOWLIST_BLOCKED,
            "ALLOWLIST_MEMBER_SHAPE_MISMATCH",
        )

    rgb_pairs = {(member.camera, member.frame) for member in members if member.asset_type == "rgb"}
    pha_pairs = {(member.camera, member.frame) for member in members if member.asset_type == "pha"}
    cameras = tuple(sorted({camera for camera, _ in rgb_pairs if camera is not None}))
    frames = tuple(sorted({frame for _, frame in rgb_pairs if frame is not None}))
    if set(cameras) != set(CAMERA_IDS) or len(cameras) != len(CAMERA_IDS):
        raise PlanBFailure(
            f"Camera set mismatch: {cameras}",
            FINAL_CLASSIFICATION_ALLOWLIST_BLOCKED,
            "ALLOWLIST_CAMERA_SET_MISMATCH",
        )
    if frames != FRAME_IDS:
        raise PlanBFailure(
            "Frame set mismatch against frozen 100-frame Plan B set.",
            FINAL_CLASSIFICATION_ALLOWLIST_BLOCKED,
            "ALLOWLIST_FRAME_SET_MISMATCH",
        )
    if rgb_pairs != pha_pairs:
        raise PlanBFailure(
            "RGB/PHA path pairs are not exact in the allowlist.",
            FINAL_CLASSIFICATION_ALLOWLIST_BLOCKED,
            "ALLOWLIST_RGB_PHA_PAIRING_MISMATCH",
        )

    counts: dict[str, int] = {}
    for member in members:
        counts[member.asset_type] = counts.get(member.asset_type, 0) + 1
    expected_counts = {"rgb": 800, "pha": 800, "calibration": 1, "smpl_params": 1}
    if counts != expected_counts:
        raise PlanBFailure(
            f"Allowlist asset counts mismatch: {counts}",
            FINAL_CLASSIFICATION_ALLOWLIST_BLOCKED,
            "ALLOWLIST_ASSET_COUNT_MISMATCH",
        )
    return {
        "allowlist_count": len(members),
        "expected_uncompressed_bytes": total,
        "camera_ids": CAMERA_IDS,
        "frame_ids": FRAME_IDS,
        "asset_counts": counts,
    }


def run_command(args: list[str], log_path: Path | None = None, check: bool = False) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            json.dumps(
                {
                    "command": args,
                    "returncode": proc.returncode,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "timestamp_utc": now_utc(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if check and proc.returncode:
        raise PlanBFailure(
            f"Command failed: {' '.join(args)}",
            FINAL_CLASSIFICATION_CONTRACT_FAIL,
            "COMMAND_FAILED",
        )
    return proc


def parse_df_free_bytes(output: str) -> int:
    lines = [line for line in output.splitlines() if line.strip()]
    if len(lines) < 2:
        raise PlanBFailure(
            "df -B1 output did not include a data line.",
            FINAL_CLASSIFICATION_STORAGE_FAIL,
            "DF_OUTPUT_PARSE_FAILED",
        )
    fields = lines[1].split()
    if len(fields) < 4:
        raise PlanBFailure(
            f"df -B1 output data line is too short: {lines[1]}",
            FINAL_CLASSIFICATION_STORAGE_FAIL,
            "DF_OUTPUT_PARSE_FAILED",
        )
    return int(fields[3])


def require_tool_version(tool_output: str) -> None:
    if "7-Zip" not in tool_output:
        raise PlanBFailure(
            "7z i output does not identify 7-Zip.",
            FINAL_CLASSIFICATION_CONTRACT_FAIL,
            "SEVENZIP_TOOL_MISSING",
        )
    if "16.02" not in tool_output:
        raise PlanBFailure(
            "7z version differs from the frozen p7zip 16.02 contract.",
            FINAL_CLASSIFICATION_CONTRACT_FAIL,
            "SEVENZIP_VERSION_MISMATCH",
        )


def ensure_execute_roots_absent(staging_root: Path, audit_root: Path) -> None:
    if staging_root.exists():
        raise PlanBFailure(
            f"Attempt staging root already exists: {staging_root}",
            FINAL_CLASSIFICATION_CONTRACT_FAIL,
            "NO_OVERWRITE_STAGING_ROOT_EXISTS",
        )
    if audit_root.exists():
        raise PlanBFailure(
            f"Attempt audit root already exists: {audit_root}",
            FINAL_CLASSIFICATION_CONTRACT_FAIL,
            "NO_RESUME_AUDIT_ROOT_EXISTS",
        )
    formal_data_root = Path(str(FORMAL_DATA_ROOT))
    if formal_data_root.exists():
        raise PlanBFailure(
            f"Formal raw-data root exists and this task must not touch it: {FORMAL_DATA_ROOT}",
            FINAL_CLASSIFICATION_CONTRACT_FAIL,
            "FORMAL_DATA_ROOT_PRESENT",
        )


def copy_attempt_manifests(audit_root: Path) -> dict[str, str]:
    manifest_dir = audit_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=False)
    copied_txt = manifest_dir / "avatarrex_lbn1_plan_b_allowlist.txt"
    copied_json = manifest_dir / "avatarrex_lbn1_plan_b_allowlist.json"
    shutil.copyfile(ALLOWLIST_TXT, copied_txt)
    shutil.copyfile(ALLOWLIST_JSON, copied_json)
    hashes = {
        str(copied_txt): sha256_file(copied_txt),
        str(copied_json): sha256_file(copied_json),
    }
    if hashes[str(copied_txt)] != ALLOWLIST_TXT_SHA256 or hashes[str(copied_json)] != ALLOWLIST_JSON_SHA256:
        raise PlanBFailure(
            "Attempt manifest copy hash mismatch.",
            FINAL_CLASSIFICATION_ALLOWLIST_BLOCKED,
            "ATTEMPT_MANIFEST_COPY_HASH_MISMATCH",
        )
    return hashes


def preflight(archive_path: Path, audit_root: Path) -> dict[str, Any]:
    if not archive_path.exists():
        raise PlanBFailure(
            f"Cloud archive is not present at the fixed path: {archive_path}",
            FINAL_CLASSIFICATION_CONTRACT_FAIL,
            "ARCHIVE_SOURCE_UNREACHABLE",
        )
    if not archive_path.is_file() or archive_path.is_symlink():
        raise PlanBFailure(
            f"Cloud archive is not a regular file: {archive_path}",
            FINAL_CLASSIFICATION_CONTRACT_FAIL,
            "ARCHIVE_SOURCE_NOT_REGULAR_FILE",
        )
    if archive_path.stat().st_size != ARCHIVE_BYTES:
        raise PlanBFailure(
            f"Archive byte count mismatch: {archive_path.stat().st_size}",
            FINAL_CLASSIFICATION_CONTRACT_FAIL,
            "ARCHIVE_BYTES_MISMATCH",
        )

    logs = audit_root / "preflight_logs"
    stat_proc = run_command(["stat", str(archive_path)], logs / "stat.json")
    if stat_proc.returncode:
        raise PlanBFailure("stat failed for archive.", FINAL_CLASSIFICATION_CONTRACT_FAIL, "ARCHIVE_STAT_FAILED")

    sha_proc = run_command(["sha256sum", str(archive_path)], logs / "sha256sum_before.json")
    if sha_proc.returncode:
        raise PlanBFailure("sha256sum failed for archive.", FINAL_CLASSIFICATION_CONTRACT_FAIL, "ARCHIVE_SHA_COMMAND_FAILED")
    live_sha = sha_proc.stdout.split()[0].lower()
    if live_sha != ARCHIVE_SHA256:
        raise PlanBFailure(
            f"Archive SHA mismatch: {live_sha}",
            FINAL_CLASSIFICATION_CONTRACT_FAIL,
            "ARCHIVE_SHA_MISMATCH",
        )

    tool_proc = run_command(["7z", "i"], logs / "7z_i.json")
    if tool_proc.returncode:
        raise PlanBFailure("7z i failed.", FINAL_CLASSIFICATION_CONTRACT_FAIL, "SEVENZIP_TOOL_FAILED")
    require_tool_version(tool_proc.stdout + tool_proc.stderr)

    test_proc = run_command(["7z", "t", str(archive_path)], logs / "7z_test.json")
    if test_proc.returncode:
        raise PlanBFailure("7z t failed.", FINAL_CLASSIFICATION_CONTRACT_FAIL, "ARCHIVE_TEST_FAILED")

    df_proc = run_command(["df", "-B1", str(DF_TARGET)], logs / "df_B1.json")
    if df_proc.returncode:
        raise PlanBFailure("df -B1 failed.", FINAL_CLASSIFICATION_STORAGE_FAIL, "DF_COMMAND_FAILED")
    free_bytes = parse_df_free_bytes(df_proc.stdout)
    run_command(["df", "-hT", str(DF_TARGET)], logs / "df_hT.json")

    free_after = free_bytes - EXPECTED_UNCOMPRESSED_BYTES - PROJECT_AUDIT_RESERVE_BYTES
    contract_required_before = SAFETY_FLOOR_BYTES + EXPECTED_UNCOMPRESSED_BYTES + CONTRACT_OVERHEAD_BYTES
    project_required_before = SAFETY_FLOOR_BYTES + EXPECTED_UNCOMPRESSED_BYTES + PROJECT_AUDIT_RESERVE_BYTES
    required_before = max(contract_required_before, project_required_before)
    if free_bytes < required_before or free_after < SAFETY_FLOOR_BYTES:
        raise PlanBFailure(
            f"Storage gate failed: free={free_bytes}, free_after={free_after}",
            FINAL_CLASSIFICATION_STORAGE_FAIL,
            "STORAGE_GATE_FAILED",
        )

    listing_proc = run_command(["7z", "l", "-slt", str(archive_path)], None)
    live_listing = audit_root / "archive_listing_slt_live.txt"
    live_listing.write_text(listing_proc.stdout, encoding="utf-8", errors="replace")
    if listing_proc.returncode:
        raise PlanBFailure("7z l -slt failed.", FINAL_CLASSIFICATION_CONTRACT_FAIL, "ARCHIVE_LISTING_FAILED")
    listing_sha = sha256_file(live_listing)
    if listing_sha != ARCHIVE_LISTING_SLT_SHA256:
        raise PlanBFailure(
            f"Archive listing fingerprint mismatch: {listing_sha}",
            FINAL_CLASSIFICATION_CONTRACT_FAIL,
            "ARCHIVE_LISTING_FINGERPRINT_MISMATCH",
        )

    return {
        "archive_path": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": live_sha,
        "archive_test_status": "PASS",
        "archive_listing_slt_sha256": listing_sha,
        "cloud_free_bytes_before": free_bytes,
        "storage_gate_status": "PASS",
        "free_after_project_reserve_bytes": free_after,
        "contract_required_before_bytes": contract_required_before,
        "project_required_before_bytes": project_required_before,
        "sevenzip_tool_status": "PASS_16.02",
    }


def write_7z_listfile(members: list[Member], audit_root: Path) -> Path:
    listfile = audit_root / "avatarrex_lbn1_plan_b_7z_listfile.txt"
    text = "".join(f"{member.path}\n" for member in members)
    with listfile.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    if sha256_file(listfile) != ALLOWLIST_TXT_SHA256:
        raise PlanBFailure(
            "Generated 7z listfile differs from the frozen allowlist text.",
            FINAL_CLASSIFICATION_ALLOWLIST_BLOCKED,
            "LISTFILE_HASH_MISMATCH",
        )
    return listfile


def run_extraction(archive_path: Path, staging_root: Path, audit_root: Path, listfile: Path) -> dict[str, Any]:
    staging_root.parent.mkdir(parents=True, exist_ok=True)
    start = now_utc()
    command = ["7z", "x", str(archive_path), f"-o{staging_root}", "-aos", f"@{listfile}", "-y"]
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    end = now_utc()
    extraction_log = {
        "command": command,
        "start_utc": start,
        "end_utc": end,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    write_json(audit_root / "extraction_call.json", extraction_log)
    if proc.returncode:
        raise PlanBFailure(
            "7z targeted extraction command failed.",
            FINAL_CLASSIFICATION_CONTRACT_FAIL,
            "TARGETED_EXTRACTION_EXECUTION_FAILED",
        )
    return {"extraction_calls": 1, "extraction_command": command, "start_utc": start, "end_utc": end}


def walk_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for current, dirs, filenames in os.walk(root):
        dirs.sort()
        for name in sorted(filenames):
            path = Path(current) / name
            files.append(path)
    return files


def image_dimensions(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
        return height, width
    except ImportError:
        pass
    except Exception as exc:
        raise PlanBFailure(
            f"Image parse failed for {path}: {exc}",
            FINAL_CLASSIFICATION_CONTRACT_FAIL,
            "IMAGE_PARSE_FAILED",
        ) from exc

    try:
        import imageio.v3 as iio

        image = iio.imread(path)
        return int(image.shape[0]), int(image.shape[1])
    except Exception as exc:
        raise PlanBFailure(
            f"Image parse failed for {path}: {exc}",
            FINAL_CLASSIFICATION_CONTRACT_FAIL,
            "IMAGE_PARSE_FAILED",
        ) from exc


def assert_finite_numbers(name: str, value: Any) -> None:
    def flatten(obj: Any) -> list[float]:
        if isinstance(obj, (int, float)):
            return [float(obj)]
        if isinstance(obj, list):
            values: list[float] = []
            for item in obj:
                values.extend(flatten(item))
            return values
        raise TypeError(f"Unsupported numeric field shape in {name}: {type(obj)!r}")

    values = flatten(value)
    if not values or not all(math.isfinite(item) for item in values):
        raise PlanBFailure(
            f"Calibration field {name} contains non-finite values.",
            FINAL_CLASSIFICATION_CONTRACT_FAIL,
            "CALIBRATION_NONFINITE",
        )


def verify_calibration(staging_root: Path) -> dict[str, Any]:
    calibration_path = staging_root / "calibration_full.json"
    try:
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PlanBFailure(
            f"calibration_full.json is not parseable JSON: {exc}",
            FINAL_CLASSIFICATION_CONTRACT_FAIL,
            "CALIBRATION_PARSE_FAILED",
        ) from exc
    if not isinstance(payload, dict) or len(payload) != 16:
        raise PlanBFailure(
            "calibration_full.json must contain exactly 16 camera records.",
            FINAL_CLASSIFICATION_CONTRACT_FAIL,
            "CALIBRATION_CAMERA_COUNT_MISMATCH",
        )
    for camera_name, row in payload.items():
        if not isinstance(row, dict):
            raise PlanBFailure(
                f"Calibration camera {camera_name} is not an object.",
                FINAL_CLASSIFICATION_CONTRACT_FAIL,
                "CALIBRATION_CAMERA_SCHEMA_MISMATCH",
            )
        for field in ("K", "R", "T", "distCoeff"):
            if field not in row:
                raise PlanBFailure(
                    f"Calibration camera {camera_name} is missing {field}.",
                    FINAL_CLASSIFICATION_CONTRACT_FAIL,
                    "CALIBRATION_FIELD_MISSING",
                )
            assert_finite_numbers(f"{camera_name}.{field}", row[field])
    missing_selected = sorted(set(CAMERA_IDS) - set(payload))
    if missing_selected:
        raise PlanBFailure(
            f"Calibration missing selected cameras: {missing_selected}",
            FINAL_CLASSIFICATION_CONTRACT_FAIL,
            "CALIBRATION_SELECTED_CAMERA_MISSING",
        )
    return {"status": "PASS_16_CAMERAS_JSON_K_R_T_DISTCOEFF_FINITE", "camera_count": len(payload)}


def verify_smpl_params(staging_root: Path) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise PlanBFailure(
            "numpy is required to parse smpl_params.npz.",
            FINAL_CLASSIFICATION_CONTRACT_FAIL,
            "NUMPY_MISSING",
        ) from exc

    smpl_path = staging_root / "smpl_params.npz"
    required = ("betas", "global_orient", "body_pose", "left_hand_pose", "right_hand_pose", "transl")
    try:
        with np.load(smpl_path, allow_pickle=True) as params:
            keys = set(params.files)
            missing = sorted(set(required) - keys)
            if missing:
                raise PlanBFailure(
                    f"smpl_params.npz missing keys: {missing}",
                    FINAL_CLASSIFICATION_CONTRACT_FAIL,
                    "SMPL_PARAMS_KEY_MISSING",
                )
            frame_lengths = {key: int(len(params[key])) for key in required if key != "betas"}
            beta_count = int(len(params["betas"]))
            for key, length in frame_lengths.items():
                if length < 1901:
                    raise PlanBFailure(
                        f"smpl_params.npz key {key} covers {length} frames, expected at least 1901.",
                        FINAL_CLASSIFICATION_CONTRACT_FAIL,
                        "SMPL_PARAMS_FRAME_COVERAGE_MISMATCH",
                    )
            for key in required:
                if not np.all(np.isfinite(params[key])):
                    raise PlanBFailure(
                        f"smpl_params.npz key {key} contains non-finite values.",
                        FINAL_CLASSIFICATION_CONTRACT_FAIL,
                        "SMPL_PARAMS_NONFINITE",
                    )
    except PlanBFailure:
        raise
    except Exception as exc:
        raise PlanBFailure(
            f"smpl_params.npz is not parseable: {exc}",
            FINAL_CLASSIFICATION_CONTRACT_FAIL,
            "SMPL_PARAMS_PARSE_FAILED",
        ) from exc
    return {
        "status": "PASS_SMPLX_STYLE_NPZ_1901_FRAMES_ALL_LOADER_KEYS",
        "frame_lengths": frame_lengths,
        "beta_count": beta_count,
    }


def run_loader_smoke(staging_root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from scene.dataset import AVRexDataset, get_dataset_type
    except Exception as exc:
        raise PlanBFailure(
            f"AVRexDataset import failed: {exc}",
            FINAL_CLASSIFICATION_CONTRACT_FAIL,
            "LOADER_IMPORT_FAILED",
        ) from exc

    dataset_type = get_dataset_type(str(staging_root))
    if dataset_type is not AVRexDataset:
        raise PlanBFailure(
            "get_dataset_type did not select AVRexDataset.",
            FINAL_CLASSIFICATION_CONTRACT_FAIL,
            "LOADER_TYPE_MISMATCH",
        )
    cams = AVRexDataset.load_cams_data(str(staging_root))
    cam_to_index = {cam["name"]: idx for idx, cam in enumerate(cams)}
    cam_index = cam_to_index[CAMERA_IDS[0]]
    frame_id = int(FRAME_IDS[0])
    dataset = AVRexDataset(str(staging_root), frame_ids=[frame_id], cam_ids=[cam_index], is_in_memory=False)
    if len(dataset) != 1:
        raise PlanBFailure(
            f"AVRexDataset canary length is {len(dataset)}, expected 1.",
            FINAL_CLASSIFICATION_CONTRACT_FAIL,
            "LOADER_LENGTH_MISMATCH",
        )
    item = dataset[0]
    if int(item["height"]) != 1500 or int(item["width"]) != 2048:
        raise PlanBFailure(
            f"AVRexDataset canary image size mismatch: {item['height']}x{item['width']}",
            FINAL_CLASSIFICATION_CONTRACT_FAIL,
            "LOADER_IMAGE_SIZE_MISMATCH",
        )
    if tuple(item["mask"].shape) != (1500, 2048):
        raise PlanBFailure(
            f"AVRexDataset mask shape mismatch: {tuple(item['mask'].shape)}",
            FINAL_CLASSIFICATION_CONTRACT_FAIL,
            "LOADER_MASK_SHAPE_MISMATCH",
        )
    return {
        "status": "PASS_AVREXDATASET_READ_ONLY_CANARY",
        "frame_id": FRAME_IDS[0],
        "camera_id": CAMERA_IDS[0],
        "length": len(dataset),
        "height": int(item["height"]),
        "width": int(item["width"]),
    }


def verify_extraction(
    staging_root: Path,
    members: list[Member],
    audit_root: Path,
    *,
    run_loader_check: bool = True,
) -> dict[str, Any]:
    if not staging_root.exists():
        raise PlanBFailure(
            f"Staging root does not exist: {staging_root}",
            FINAL_CLASSIFICATION_CONTRACT_FAIL,
            "STAGING_ROOT_MISSING",
        )
    expected = {member.path: member for member in members}
    actual_files = walk_files(staging_root)
    actual_rel = []
    symlinks = []
    for path in actual_files:
        if path.is_symlink():
            symlinks.append(str(path))
        actual_rel.append(path.relative_to(staging_root).as_posix())
    if symlinks:
        raise PlanBFailure(
            f"Extracted symlinks are forbidden: {symlinks[:5]}",
            FINAL_CLASSIFICATION_CONTRACT_FAIL,
            "EXTRACTED_SYMLINK_FORBIDDEN",
        )

    missing = sorted(set(expected) - set(actual_rel))
    unexpected = sorted(set(actual_rel) - set(expected))
    if missing or unexpected:
        write_json(audit_root / "path_set_mismatch.json", {"missing": missing, "unexpected": unexpected})
        raise PlanBFailure(
            f"Extracted path set mismatch: missing={len(missing)} unexpected={len(unexpected)}",
            FINAL_CLASSIFICATION_CONTRACT_FAIL,
            "EXTRACTED_PATH_SET_MISMATCH",
        )

    inventory = []
    total_bytes = 0
    empty_files = []
    size_mismatch = []
    hash_mismatch = []
    crc_mismatch = []
    image_size_mismatch = []
    image_parse_count = 0

    for rel_path in sorted(actual_rel):
        file_path = staging_root / rel_path
        member = expected[rel_path]
        sha, crc, size = hash_and_crc32(file_path)
        total_bytes += size
        if size == 0:
            empty_files.append(rel_path)
        if size != member.bytes:
            size_mismatch.append({"path": rel_path, "expected": member.bytes, "actual": size})
        if sha != member.sha256:
            hash_mismatch.append({"path": rel_path, "expected": member.sha256, "actual": sha})
        if crc != member.crc32:
            crc_mismatch.append({"path": rel_path, "expected": member.crc32, "actual": crc})

        height = width = None
        if member.asset_type in {"rgb", "pha"}:
            height, width = image_dimensions(file_path)
            image_parse_count += 1
            if (height, width) != (1500, 2048):
                image_size_mismatch.append({"path": rel_path, "height": height, "width": width})

        inventory.append(
            {
                "relative_path": rel_path,
                "archive_internal_path": rel_path,
                "bytes": size,
                "sha256": sha,
                "crc32": crc,
                "camera": member.camera,
                "frame": member.frame,
                "asset_type": member.asset_type,
                "height": height,
                "width": width,
            }
        )

    if empty_files or size_mismatch or hash_mismatch or crc_mismatch or image_size_mismatch:
        write_json(
            audit_root / "content_mismatch.json",
            {
                "empty_files": empty_files,
                "size_mismatch": size_mismatch,
                "hash_mismatch": hash_mismatch,
                "crc_mismatch": crc_mismatch,
                "image_size_mismatch": image_size_mismatch,
            },
        )
        raise PlanBFailure(
            "Extracted content does not match the frozen manifest.",
            FINAL_CLASSIFICATION_CONTRACT_FAIL,
            "EXTRACTED_CONTENT_MISMATCH",
        )

    rgb_pairs = {(row["camera"], row["frame"]) for row in inventory if row["asset_type"] == "rgb"}
    pha_pairs = {(row["camera"], row["frame"]) for row in inventory if row["asset_type"] == "pha"}
    if rgb_pairs != pha_pairs:
        raise PlanBFailure(
            "Extracted RGB/PHA pairs are not exact.",
            FINAL_CLASSIFICATION_CONTRACT_FAIL,
            "EXTRACTED_RGB_PHA_PAIRING_MISMATCH",
        )
    calibration_status = verify_calibration(staging_root)
    body_parameter_status = verify_smpl_params(staging_root)
    loader_status = (
        run_loader_smoke(staging_root)
        if run_loader_check
        else {"status": "NOT_RUN_PENDING_USER_AUTHORIZATION"}
    )

    inventory_path = audit_root / "avatarrex_lbn1_plan_b_inventory.json"
    sha_registry_path = audit_root / "avatarrex_lbn1_plan_b_sha_registry.json"
    loader_canary_path = audit_root / "avatarrex_lbn1_plan_b_loader_canary_manifest.json"
    write_json(
        inventory_path,
        {
            "schema_version": "avatarrex_lbn1.plan_b.inventory.v1",
            "task_id": TASK_ID,
            "plan_id": PLAN_ID,
            "attempt_id": ATTEMPT_ID,
            "staging_root": str(staging_root),
            "file_count": len(inventory),
            "total_bytes": total_bytes,
            "files": inventory,
        },
    )
    write_json(
        sha_registry_path,
        {
            "schema_version": "avatarrex_lbn1.plan_b.sha_registry.v1",
            "task_id": TASK_ID,
            "plan_id": PLAN_ID,
            "attempt_id": ATTEMPT_ID,
            "file_count": len(inventory),
            "files": [
                {"relative_path": row["relative_path"], "bytes": row["bytes"], "sha256": row["sha256"]}
                for row in inventory
            ],
        },
    )
    if run_loader_check:
        write_json(
            loader_canary_path,
            {
                "schema_version": "avatarrex_lbn1.plan_b.loader_canary_manifest.v1",
                "task_id": TASK_ID,
                "plan_id": PLAN_ID,
                "attempt_id": ATTEMPT_ID,
                "camera_ids": CAMERA_IDS,
                "frame_ids": FRAME_IDS,
                "loader_status": loader_status,
            },
        )
    return {
        "extracted_file_count": len(inventory),
        "extracted_total_bytes": total_bytes,
        "rgb_count": len(rgb_pairs),
        "pha_count": len(pha_pairs),
        "metadata_count": 2,
        "rgb_pha_pairing_status": "PASS_EXACT_800_PAIRS",
        "calibration_status": calibration_status,
        "body_parameter_status": body_parameter_status,
        "loader_status": loader_status,
        "missing_file_count": 0,
        "unexpected_file_count": 0,
        "empty_file_count": 0,
        "duplicate_relative_path_count": 0,
        "image_parse_count": image_parse_count,
        "inventory_path": str(inventory_path),
        "sha_registry_path": str(sha_registry_path),
        "loader_canary_manifest_path": (
            str(loader_canary_path)
            if run_loader_check
            else "NOT_CREATED_PENDING_USER_AUTHORIZATION"
        ),
    }


def final_summary(
    classification: str,
    preflight_data: dict[str, Any] | None,
    extraction_data: dict[str, Any] | None,
    verification_data: dict[str, Any] | None,
    failure: PlanBFailure | None,
) -> dict[str, Any]:
    return {
        "schema_version": "avatarrex_lbn1.plan_b.final_summary.v1",
        "task_id": TASK_ID,
        "contract_task_id": CONTRACT_TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "new_branch": NEW_BRANCH,
        "windows_worktree": WINDOWS_WORKTREE,
        "cloud_worktree": CLOUD_WORKTREE,
        "archive_path": str(ARCHIVE_PATH),
        "archive_bytes_expected": ARCHIVE_BYTES,
        "archive_sha256_expected": ARCHIVE_SHA256,
        "plan_id": PLAN_ID,
        "attempt_id": ATTEMPT_ID,
        "contract_attempt_id": CONTRACT_ATTEMPT_ID,
        "staging_root": str(STAGING_ROOT),
        "audit_root": str(AUDIT_ROOT),
        "formal_data_root_status": "NONE",
        "identity": IDENTITY,
        "sequence": SEQUENCE,
        "camera_ids": CAMERA_IDS,
        "frame_ids": FRAME_IDS,
        "allowlist_count": ALLOWLIST_COUNT,
        "expected_uncompressed_bytes": EXPECTED_UNCOMPRESSED_BYTES,
        "safety_floor_bytes": SAFETY_FLOOR_BYTES,
        "project_audit_reserve_bytes": PROJECT_AUDIT_RESERVE_BYTES,
        "preflight": preflight_data or {},
        "extraction": extraction_data or {"extraction_calls": 0},
        "verification": verification_data or {},
        "archive_mutations": 0,
        "preprocessing_steps": 0,
        "training_steps": 0,
        "generation_calls": 0,
        "paper_modifications": 0,
        "paper_final": False,
        "final_classification": classification,
        "next_task": NEXT_TASK_SUCCESS if classification == FINAL_CLASSIFICATION_PASS else "RETRY_AFTER_CLOUD_SOURCE_PATH_IS_REACHABLE",
        "failure": None
        if failure is None
        else {"kind": failure.failure_kind, "message": str(failure), "classification": failure.classification},
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "timestamp_utc": now_utc(),
        },
    }


def write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    failure = summary.get("failure") or {}
    lines = [
        "# AvatarReX LBN1 Plan B Targeted Extraction Report",
        "",
        f"- TASK_ID: `{summary['task_id']}`",
        f"- PLAN_ID: `{summary['plan_id']}`",
        f"- ATTEMPT_ID: `{summary['attempt_id']}`",
        f"- FINAL_CLASSIFICATION: `{summary['final_classification']}`",
        f"- ARCHIVE_PATH: `{summary['archive_path']}`",
        f"- STAGING_ROOT: `{summary['staging_root']}`",
        f"- ALLOWLIST_COUNT: `{summary['allowlist_count']}`",
        f"- EXPECTED_UNCOMPRESSED_BYTES: `{summary['expected_uncompressed_bytes']}`",
        f"- EXTRACTION_CALLS: `{summary['extraction'].get('extraction_calls', 0)}`",
        f"- ARCHIVE_MUTATIONS: `{summary['archive_mutations']}`",
        f"- FORMAL_DATA_ROOT_STATUS: `{summary['formal_data_root_status']}`",
        f"- PREPROCESSING_STEPS: `{summary['preprocessing_steps']}`",
        f"- TRAINING_STEPS: `{summary['training_steps']}`",
        f"- GENERATION_CALLS: `{summary['generation_calls']}`",
        f"- PAPER_MODIFICATIONS: `{summary['paper_modifications']}`",
        f"- PAPER_FINAL: `{str(summary['paper_final']).lower()}`",
        f"- NEXT_TASK: `{summary['next_task']}`",
        "",
        "## Failure",
        "",
    ]
    if failure:
        lines.extend([f"- kind: `{failure['kind']}`", f"- message: {failure['message']}"])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "No formal raw-data root was created. No preprocessing, template generation, LBS grid generation, Base Avatar training, garment generation, API call, or paper modification is authorized or performed by this script.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def execute(args: argparse.Namespace) -> dict[str, Any]:
    members = load_members()
    preflight_data: dict[str, Any] | None = None
    extraction_data: dict[str, Any] | None = None
    verification_data: dict[str, Any] | None = None
    failure: PlanBFailure | None = None
    classification = FINAL_CLASSIFICATION_PASS

    archive_path = Path(args.archive_path)
    staging_root = Path(args.staging_root)
    audit_root = Path(args.audit_root)

    try:
        ensure_execute_roots_absent(staging_root, audit_root)
        audit_root.mkdir(parents=True, exist_ok=False)
        manifest_hashes = copy_attempt_manifests(audit_root)
        write_json(audit_root / "manifest_copy_hashes.json", manifest_hashes)
        preflight_data = preflight(archive_path, audit_root)
        listfile = write_7z_listfile(members, audit_root)
        extraction_data = run_extraction(archive_path, staging_root, audit_root, listfile)
        verification_data = verify_extraction(staging_root, members, audit_root)

        sha_after = run_command(
            ["sha256sum", str(archive_path)],
            audit_root / "preflight_logs" / "sha256sum_after.json",
        )
        if sha_after.returncode:
            raise PlanBFailure(
                "Post-extraction archive SHA check failed.",
                FINAL_CLASSIFICATION_ARCHIVE_MUTATION,
                "ARCHIVE_SHA_AFTER_COMMAND_FAILED",
            )
        live_after = sha_after.stdout.split()[0].lower()
        if live_after != ARCHIVE_SHA256:
            raise PlanBFailure(
                f"Archive mutated or source changed after extraction: {live_after}",
                FINAL_CLASSIFICATION_ARCHIVE_MUTATION,
                "ARCHIVE_SHA_AFTER_MISMATCH",
            )
    except PlanBFailure as exc:
        failure = exc
        classification = exc.classification

    summary = final_summary(classification, preflight_data, extraction_data, verification_data, failure)
    summary_path = audit_root / "avatarrex_lbn1_plan_b_final_summary.json"
    report_path = audit_root / "AVATARREX_LBN1_PLAN_B_TARGETED_EXTRACTION_REPORT.md"
    if audit_root.exists():
        write_json(summary_path, summary)
        write_markdown_report(report_path, summary)
    if failure:
        raise failure
    return summary


def manifest_check(args: argparse.Namespace) -> dict[str, Any]:
    members = load_members()
    payload = validate_manifest_members(members)
    result = {
        "schema_version": "avatarrex_lbn1.plan_b.manifest_check.v1",
        "task_id": TASK_ID,
        "status": "PASS",
        "allowlist_txt": str(ALLOWLIST_TXT),
        "allowlist_json": str(ALLOWLIST_JSON),
        "allowlist_txt_sha256": sha256_file(ALLOWLIST_TXT),
        "allowlist_json_sha256": sha256_file(ALLOWLIST_JSON),
        **payload,
    }
    if args.output_json:
        write_json(Path(args.output_json), result)
    return result


def verify_existing(args: argparse.Namespace) -> dict[str, Any]:
    members = load_members()
    audit_root = Path(args.audit_root)
    audit_root.mkdir(parents=True, exist_ok=True)
    verification = verify_extraction(Path(args.staging_root), members, audit_root)
    summary = final_summary(FINAL_CLASSIFICATION_PASS, None, {"extraction_calls": 0}, verification, None)
    write_json(audit_root / "avatarrex_lbn1_plan_b_verify_existing_summary.json", summary)
    return summary


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("execute", "manifest-check", "verify-existing"), default="execute")
    parser.add_argument("--archive-path", default=str(ARCHIVE_PATH))
    parser.add_argument("--staging-root", default=str(STAGING_ROOT))
    parser.add_argument("--audit-root", default=str(AUDIT_ROOT))
    parser.add_argument("--output-json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.mode == "manifest-check":
            result = manifest_check(args)
        elif args.mode == "verify-existing":
            result = verify_existing(args)
        else:
            result = execute(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except PlanBFailure as exc:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "classification": exc.classification,
                    "failure_kind": exc.failure_kind,
                    "message": str(exc),
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
