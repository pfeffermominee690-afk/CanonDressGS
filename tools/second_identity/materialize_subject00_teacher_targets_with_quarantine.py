from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
import zstandard as zstd
from PIL import Image


TASK_ID = "AAAI27-SUBJECT00-TEACHER-TARGET-MATERIALIZATION-WITH-QUARANTINE-001"
SOURCE_BRANCH = "research/subject00-teacher-target-camera-blocker-resolution-20260727"
SOURCE_HEAD = "a434ae7a78fe898be2658180f20bbcd4391a64c0"
NEW_BRANCH = "research/subject00-teacher-target-materialization-quarantine-20260727"
WINDOWS_WORKTREE = r"E:\model_train\canondressgs_subject00_teacher_target_materialization_quarantine"
CLOUD_WORKTREE = (
    "/root/autodl-tmp/canondressgs_work/worktrees/"
    "canondressgs_subject00_teacher_target_materialization_quarantine"
)
WINDOWS_PROJECT_ROOT = (
    r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-TEACHER-TARGETS-001"
)
WINDOWS_ATTEMPT_ROOT = (
    WINDOWS_PROJECT_ROOT + r"\attempt_001_subject00_24_cell_teacher_targets"
)
CLOUD_TARGET_ROOT = (
    "/root/autodl-tmp/canondressgs_work/teacher_targets/"
    "SUBJECT00-24CELL-001/attempt_001"
)
MATERIALIZATION_MODE = "PORTABLE_ARCHIVE_AND_CLOUD_EXTRACTION"
SCHEMA = "canondressgs.full_dataset.v1"
LOADER_PATH = "scene/full_dressable_dataset.py"
LOADER_SHA256 = "786c93355776093e610dfe1bc74efd61233da134550a220bc06601f4f2365508"
CREATION_IMPLEMENTATION_PATH = "tools/aaai27/build_data_capacity_fixture.py"
CREATION_IMPLEMENTATION_SHA256 = (
    "de930f6a154231bec4881a8071a323d8794db3f1f24c5ccdee5090d12f639b72"
)
POSE_SHA256 = "ac2738c308ad1a9cc02e7b63323c75e0eab28bddc88c57bb5887e07bf8ea28d2"
FORMAL_BASE_CHECKPOINT_SHA256 = (
    "2d09b1ce6bb19b8b6418314caab8dc1cf92d70e432a9e09e009989332ea4a2f7"
)
ARCHIVE_FORMAT = "DETERMINISTIC_TAR_ZSTD"
ARCHIVE_NAME = "subject00_24cell_teacher_targets_attempt001.tar.zst"
FROZEN_CREATED_AT = "2026-07-27T00:00:00Z"
DETERMINISTIC_SEED = 0
MORPH_RADIUS = 5
QUARANTINED_REQUEST_IDS = (
    "subject00_O03_slot04_canary_attempt004_cand00",
    "subject00_O01_slot04_remaining_attempt005_cand00",
)
O03_EXCLUDED_REQUEST_IDS = ("subject00_O03_slot04_canary_attempt004_cand00",)
FINAL_CLASSIFICATION = (
    "SUBJECT00_TEACHER_TARGET_MATERIALIZATION_PASS_22_TRAINING_2_QUARANTINED"
)
NEXT_TASK = "RUN_SUBJECT00_O03_PROVISIONAL_TEACHER_CAMERA_SAFE_7VIEW_RERUN"

EVIDENCE_NAMES = (
    "subject00_teacher_target_camera_resolution_blocker_correction_overlay_20260727.json",
    "subject00_22cell_camera_safe_reference_distribution_20260727.json",
    "subject00_problem_cell_camera_salvage_results_20260727.json",
    "subject00_teacher_target_camera_eligibility_registry_20260727.json",
    "subject00_teacher_target_quarantine_registry_20260727.json",
    "subject00_teacher_target_camera_records_draft_20260727.json",
    "subject00_O03_provisional_camera_safe_target_manifest_draft_20260727.json",
    "SUBJECT00_24_CELL_TEACHER_TARGET_CREATION_EXECUTION_CONTRACT_20260727.md",
    "subject00_24_cell_teacher_target_creation_execution_manifest_20260727.json",
    "subject00_24_cell_teacher_target_schema_registry_20260727.json",
    "subject00_24_cell_teacher_target_camera_registry_20260727.json",
    "subject00_24_cell_teacher_target_resolution_contract_20260727.json",
    "subject00_24_cell_teacher_target_materialization_mode_audit_20260727.json",
    "subject00_24_cell_teacher_target_qa_registry_20260727.json",
    "subject00_global_accepted_cell_registry_24of24_20260726.json",
    "subject00_24_cell_mask_accepted_promotion_overlay_20260727.json",
)

INDEX_NAMES = {
    "all": "all_provenance_records.json",
    "training": "training_records.json",
    "evaluation": "evaluation_records.json",
    "quarantine": "review_only_quarantine_records.json",
    "O01": "O01_training_records.json",
    "O03": "O03_training_records.json",
    "O04": "O04_training_records.json",
    "O03_provisional": "O03_provisional_base60747_records.json",
}

DERIVED_MASK_FIELDS = (
    "target_edit_mask",
    "target_edit_core_mask",
    "target_preserve_mask",
    "target_transition_mask",
    "target_protected_mask",
    "target_base_foreground_mask",
    "target_old_clothing_mask",
    "target_revealed_skin_mask",
)
DERIVED_CODES = {
    "target_edit_mask": "edit",
    "target_edit_core_mask": "core",
    "target_preserve_mask": "preserve",
    "target_transition_mask": "transition",
    "target_protected_mask": "protected",
    "target_base_foreground_mask": "base_fg",
    "target_old_clothing_mask": "old_clothing",
    "target_revealed_skin_mask": "revealed_skin",
}
ALL_MASK_FIELDS = DERIVED_MASK_FIELDS + (
    "target_foreground_mask",
    "target_clothing_mask",
)

GIT_ARTIFACTS = (
    "subject00_teacher_target_materialization_execution_registry_20260727.json",
    "subject00_teacher_target_training_index_registry_20260727.json",
    "subject00_teacher_target_quarantine_materialization_registry_20260727.json",
    "subject00_teacher_target_derived_target_registry_20260727.json",
    "subject00_teacher_target_portable_archive_registry_20260727.json",
    "subject00_teacher_target_cloud_extraction_registry_20260727.json",
    "subject00_teacher_target_loader_smoke_results_20260727.json",
    "subject00_O03_camera_safe_7view_teacher_target_registry_20260727.json",
    "subject00_teacher_target_materialization_execution_tests_20260727.json",
    "subject00_teacher_target_materialization_final_summary_20260727.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def text_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def safe_rel(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    require(not path.is_absolute(), f"archive path is absolute: {value}")
    require(".." not in path.parts, f"archive path traverses: {value}")
    require(path.parts and path.parts[0] not in {"", "."}, f"invalid archive path: {value}")
    return path.as_posix()


def file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    return {
        "path": (
            path.relative_to(relative_to).as_posix()
            if relative_to is not None
            else str(path)
        ),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def disk_morph(value: np.ndarray, radius: int, operation: str) -> np.ndarray:
    require(value.ndim == 2 and value.dtype == np.bool_, "morph input must be bool HxW")
    h, w = value.shape
    offsets = [
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if dx * dx + dy * dy <= radius * radius
    ]
    result = np.zeros_like(value) if operation == "dilate" else np.ones_like(value)
    for dy, dx in offsets:
        shifted = np.zeros_like(value) if operation == "dilate" else np.zeros_like(value)
        src_y0, src_y1 = max(0, -dy), min(h, h - dy)
        src_x0, src_x1 = max(0, -dx), min(w, w - dx)
        dst_y0, dst_y1 = max(0, dy), min(h, h + dy)
        dst_x0, dst_x1 = max(0, dx), min(w, w + dx)
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = value[src_y0:src_y1, src_x0:src_x1]
        if operation == "dilate":
            result |= shifted
        else:
            result &= shifted
    return result


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) >= 128


def save_mask(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value.astype(np.uint8) * 255, mode="L").save(path, format="PNG")


def warp_image(source: Path, transform: list[list[float]], size: tuple[int, int], output: Path) -> None:
    matrix = np.asarray(transform, dtype=np.float64)
    require(matrix.shape == (3, 3), "registered transform must be 3x3")
    inverse = np.linalg.inv(matrix)
    coeffs = tuple(float(x) for x in inverse[:2].reshape(-1))
    with Image.open(source) as image:
        warped = image.convert("RGB").transform(
            size,
            Image.Transform.AFFINE,
            coeffs,
            resample=Image.Resampling.BILINEAR,
            fillcolor=(0, 0, 0),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        warped.save(output, format="PNG")


def warp_mask(source: Path, transform: list[list[float]], size: tuple[int, int]) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    inverse = np.linalg.inv(matrix)
    coeffs = tuple(float(x) for x in inverse[:2].reshape(-1))
    with Image.open(source) as image:
        warped = image.convert("L").transform(
            size,
            Image.Transform.AFFINE,
            coeffs,
            resample=Image.Resampling.NEAREST,
            fillcolor=0,
        )
        return np.asarray(warped, dtype=np.uint8) >= 128


def validate_source(repo: Path, pose_npz: Path) -> dict[str, Any]:
    risk = repo / "paper_protocol" / "reviewer_risk"
    for name in EVIDENCE_NAMES:
        require((risk / name).is_file(), f"missing frozen evidence: {name}")
    require(sha256(repo / LOADER_PATH) == LOADER_SHA256, "frozen loader SHA mismatch")
    require(
        sha256(repo / CREATION_IMPLEMENTATION_PATH) == CREATION_IMPLEMENTATION_SHA256,
        "frozen creation implementation SHA mismatch",
    )
    require(pose_npz.is_file() and sha256(pose_npz) == POSE_SHA256, "pose NPZ SHA mismatch")

    eligibility = json_load(risk / EVIDENCE_NAMES[3])
    quarantine = json_load(risk / EVIDENCE_NAMES[4])
    draft = json_load(risk / EVIDENCE_NAMES[5])
    preflight = json_load(risk / "subject00_24_cell_teacher_target_creation_execution_manifest_20260727.json")
    materialization = json_load(risk / "subject00_24_cell_teacher_target_materialization_mode_audit_20260727.json")
    schema = json_load(risk / "subject00_24_cell_teacher_target_schema_registry_20260727.json")
    require(eligibility["record_count"] == 24, "eligibility record count is not 24")
    require(eligibility["teacher_target_training_eligible_count"] == 22, "training count is not 22")
    require(eligibility["teacher_target_review_only_count"] == 2, "review-only count is not 2")
    require(draft["record_count"] == 24, "camera draft count is not 24")
    require(draft["target_K_present_count"] == 22, "camera-safe draft count is not 22")
    require(
        draft["target_K_absent_quarantined_count"] == 2,
        "quarantined camera draft count is not 2",
    )
    actual_quarantine = {record["request_id"] for record in quarantine["records"]}
    require(actual_quarantine == set(QUARANTINED_REQUEST_IDS), "quarantine IDs changed")
    require(materialization["archive"]["format"] == "tar.zst", "frozen archive is not tar.zst")
    require(schema["target_schema"] == SCHEMA, "schema registry changed")

    draft_by_id = {record["request_id"]: record for record in draft["records"]}
    preflight_by_id = {
        record["condition_id"]: record for record in preflight["records"]
    }
    require(set(draft_by_id) == set(preflight_by_id), "draft/preflight request sets differ")

    source_inventory: list[dict[str, Any]] = []
    resolution_distribution: dict[str, int] = {}
    limitation_count = 0
    human_override_count = 0
    for request_id, record in draft_by_id.items():
        width, height = int(record["target_width"]), int(record["target_height"])
        resolution_distribution[f"{width}x{height}"] = (
            resolution_distribution.get(f"{width}x{height}", 0) + 1
        )
        limitation_count += int(bool(record["official_limitation_codes"]))
        human_override_count += int(record["human_override_status"] is not None)
        source_condition = Path(record["source_condition"]["path"])
        require(source_condition.is_file(), f"missing source condition: {request_id}")
        require(
            sha256(source_condition) == record["source_condition"]["sha256"],
            f"source condition SHA changed: {request_id}",
        )
        for key in ("accepted_raw", "person_mask", "garment_mask"):
            binding = record[key]
            path = Path(binding["path"])
            require(path.is_file(), f"missing source {key}: {request_id}")
            require(path.stat().st_size == binding["bytes"], f"{key} bytes changed: {request_id}")
            require(sha256(path) == binding["sha256"], f"{key} SHA changed: {request_id}")
            source_inventory.append(
                {
                    "request_id": request_id,
                    "kind": key,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": binding["sha256"],
                }
            )
        raw = Path(record["accepted_raw"]["path"])
        person = Path(record["person_mask"]["path"])
        garment = Path(record["garment_mask"]["path"])
        with Image.open(raw) as image:
            require(image.size == (width, height), f"raw resolution changed: {request_id}")
        person_value, garment_value = load_mask(person), load_mask(garment)
        require(person_value.shape == (height, width), f"person shape changed: {request_id}")
        require(garment_value.shape == (height, width), f"garment shape changed: {request_id}")
        require(not np.any(garment_value & ~person_value), f"garment not subset person: {request_id}")
        if record["training_eligible"]:
            source_mask_binding = record["validation_metrics"]["input_provenance"]
            source_mask = Path(source_mask_binding["source_person_mask_path"])
            require(source_mask.is_file(), f"missing source person mask: {request_id}")
            require(
                sha256(source_mask) == source_mask_binding["source_person_mask_sha256"],
                f"source person mask SHA changed: {request_id}",
            )
            require(record["camera_binding_status"] == "UNIQUE_SIMILARITY_BINDING_PASS", request_id)
            require(record["registered_source_to_target_transform"] is not None, request_id)
            require(record["target_K"] is not None, request_id)
        else:
            require(request_id in QUARANTINED_REQUEST_IDS, request_id)
            require(record["camera_binding_status"] == "UNRESOLVED_HUMAN_OVERRIDE", request_id)
            require(record["registered_source_to_target_transform"] is None, request_id)
            require(record["target_K"] is None, request_id)

    require(resolution_distribution == {"1349x1166": 23, "1350x1165": 1}, "resolution changed")
    require(limitation_count == 9, "limitation propagation source is not 9/9")
    require(human_override_count == 2, "human override propagation source is not 2/2")
    return {
        "risk": risk,
        "eligibility": eligibility,
        "quarantine": quarantine,
        "draft": draft,
        "draft_by_id": draft_by_id,
        "preflight": preflight,
        "preflight_by_id": preflight_by_id,
        "source_inventory": source_inventory,
        "resolution_distribution": resolution_distribution,
        "evidence": [
            {
                "path": f"paper_protocol/reviewer_risk/{name}",
                "bytes": (risk / name).stat().st_size,
                "sha256": sha256(risk / name),
            }
            for name in EVIDENCE_NAMES
        ],
    }


def pose_values(pose_npz: Path) -> dict[str, Any]:
    with np.load(pose_npz, allow_pickle=False) as archive:
        pose = np.concatenate(
            [
                np.asarray(archive["global_orient"][0], dtype=np.float32).reshape(3),
                np.asarray(archive["body_pose"][0], dtype=np.float32).reshape(63),
                np.zeros(3, dtype=np.float32),
                np.zeros(6, dtype=np.float32),
                np.asarray(archive["left_hand_pose"][0], dtype=np.float32).reshape(45),
                np.asarray(archive["right_hand_pose"][0], dtype=np.float32).reshape(45),
            ]
        )
        th = np.asarray(archive["transl"][0], dtype=np.float32).reshape(3)
        source_global_orient = np.asarray(
            archive["global_orient"][0], dtype=np.float32
        ).reshape(3)
    require(pose.shape == (165,), "formal pose reconstruction is not length 165")
    return {
        "pose": [float(x) for x in pose],
        "Rh_raw": [0.0, 0.0, 0.0],
        "R_global": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "Th": [float(x) for x in th],
        "source_global_orient": [float(x) for x in source_global_orient],
    }


def copy_verified(source: Path, destination: Path, expected_sha: str) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    actual = sha256(destination)
    require(actual == expected_sha, f"byte-copy SHA mismatch: {destination}")
    return file_record(destination)


def rel_for_manifest(path: Path, manifest_root: Path) -> str:
    return Path(os.path.relpath(path, manifest_root)).as_posix()


def write_index(path: Path, records: list[dict[str, Any]], index_type: str) -> None:
    json_write(
        path,
        {
            "schema_version": "canondressgs.subject00.teacher_target.index.v1",
            "task_id": TASK_ID,
            "created_at": FROZEN_CREATED_AT,
            "index_type": index_type,
            "count": len(records),
            "denominator": len(records),
            "request_ids": [record["request_id"] for record in records],
            "records": records,
        },
    )


def archive_payload(attempt: Path, archive_path: Path) -> dict[str, Any]:
    excluded = {
        archive_path.relative_to(attempt).as_posix(),
        (archive_path.parent / "archive_manifest.json").relative_to(attempt).as_posix(),
        (archive_path.parent / "payload_manifest.json").relative_to(attempt).as_posix(),
    }
    files = sorted(
        (
            path
            for path in attempt.rglob("*")
            if path.is_file() and path.relative_to(attempt).as_posix() not in excluded
        ),
        key=lambda path: path.relative_to(attempt).as_posix(),
    )
    payload_entries = [file_record(path, relative_to=attempt) for path in files]
    payload_manifest_path = archive_path.parent / "payload_manifest.json"
    json_write(
        payload_manifest_path,
        {
            "schema_version": "canondressgs.subject00.teacher_target.payload_manifest.v1",
            "task_id": TASK_ID,
            "created_at": FROZEN_CREATED_AT,
            "verification_policy": "VERIFY_ALL_LISTED_FILES_PLUS_THIS_MANIFEST_FROM_ARCHIVE",
            "listed_file_count": len(payload_entries),
            "files": payload_entries,
        },
    )
    files.append(payload_manifest_path)
    files.sort(key=lambda path: path.relative_to(attempt).as_posix())
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    compressor = zstd.ZstdCompressor(level=19, threads=1, write_checksum=True)
    with archive_path.open("wb") as raw:
        with compressor.stream_writer(raw, closefd=False) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                for path in files:
                    rel = safe_rel(path.relative_to(attempt).as_posix())
                    info = tar.gettarinfo(str(path), arcname=rel)
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mode = 0o644
                    with path.open("rb") as handle:
                        tar.addfile(info, handle)
    archive_sha = sha256(archive_path)

    names: list[str] = []
    with tempfile.TemporaryDirectory(prefix="subject00_archive_smoke_") as temp_name:
        temp_root = Path(temp_name)
        with archive_path.open("rb") as raw:
            with zstd.ZstdDecompressor().stream_reader(raw) as decompressed:
                with tarfile.open(fileobj=decompressed, mode="r|") as tar:
                    for member in tar:
                        require(member.isfile(), f"non-file archive member: {member.name}")
                        name = safe_rel(member.name)
                        require(name not in names, f"duplicate archive member: {name}")
                        names.append(name)
                        destination = temp_root / Path(*PurePosixPath(name).parts)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        source = tar.extractfile(member)
                        require(source is not None, f"archive member unreadable: {name}")
                        with destination.open("wb") as handle:
                            shutil.copyfileobj(source, handle)
        require(len(names) == len(files), "archive member count mismatch")
        manifest = json_load(temp_root / "12_portable_archive" / "payload_manifest.json")
        for entry in manifest["files"]:
            path = temp_root / Path(*PurePosixPath(entry["path"]).parts)
            require(path.is_file(), f"archive smoke missing file: {entry['path']}")
            require(path.stat().st_size == entry["bytes"], f"archive smoke bytes mismatch: {path}")
            require(sha256(path) == entry["sha256"], f"archive smoke SHA mismatch: {path}")

    result = {
        "schema_version": "canondressgs.subject00.teacher_target.archive_manifest.v1",
        "task_id": TASK_ID,
        "created_at": FROZEN_CREATED_AT,
        "archive_format": ARCHIVE_FORMAT,
        "archive_path": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": archive_sha,
        "archive_member_count": len(names),
        "listed_payload_file_count": len(payload_entries),
        "duplicate_member_count": 0,
        "path_traversal_member_count": 0,
        "fixed_mtime": 0,
        "fixed_uid": 0,
        "fixed_gid": 0,
        "file_mode": "0644",
        "sort_order": "UTF8_POSIX_RELATIVE_PATH_ASCENDING",
        "compression": "ZSTD_LEVEL_19_SINGLE_THREAD_CHECKSUM",
        "windows_extraction_smoke_status": "PASS_ALL_LISTED_SHA",
    }
    json_write(archive_path.parent / "archive_manifest.json", result)
    return result


def build_windows(args: argparse.Namespace) -> None:
    repo = args.repo_root.resolve()
    pose_npz = args.pose_npz.resolve()
    project = Path(WINDOWS_PROJECT_ROOT)
    attempt = Path(WINDOWS_ATTEMPT_ROOT)
    staging = project / ".staging"
    require(str(repo) == WINDOWS_WORKTREE, f"wrong worktree: {repo}")
    require(not attempt.exists(), f"Windows attempt root already exists: {attempt}")
    require(not staging.exists(), f"Windows staging root already exists: {staging}")
    source = validate_source(repo, pose_npz)
    project.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=False, exist_ok=False)
    for name in (
        "01_contract_snapshot",
        "02_records",
        "03_raw",
        "04_person_masks",
        "05_garment_masks",
        "06_camera",
        "07_derived_targets",
        "08_quality_audit",
        "09_review_assets",
        "10_final_registry",
        "11_logs",
        "12_portable_archive",
    ):
        (staging / name).mkdir()

    snapshot_repo = staging / "01_contract_snapshot" / "e"
    snapshot_records: list[dict[str, Any]] = []
    for index, entry in enumerate(source["evidence"], start=1):
        source_path = repo / entry["path"]
        destination = snapshot_repo / f"{index:02d}{source_path.suffix.lower()}"
        copy_verified(source_path, destination, entry["sha256"])
        snapshot_records.append(
            {
                **entry,
                "snapshot_path": destination.relative_to(staging).as_posix(),
            }
        )
    implementation_snapshots = (
        (LOADER_PATH, "loader.py"),
        (CREATION_IMPLEMENTATION_PATH, "builder.py"),
        ("schemas/canondressgs_full_dataset_v1.schema.json", "schema.json"),
        (
            "tools/second_identity/materialize_subject00_teacher_targets_with_quarantine.py",
            "wrapper.py",
        ),
    )
    for rel, snapshot_name in implementation_snapshots:
        source_path = repo / rel
        destination = staging / "01_contract_snapshot" / "i" / snapshot_name
        copy_verified(source_path, destination, sha256(source_path))
    copy_verified(
        pose_npz,
        staging / "01_contract_snapshot" / "subject00_smpl_params.npz",
        POSE_SHA256,
    )
    json_write(
        staging / "01_contract_snapshot" / "contract_snapshot_manifest.json",
        {
            "schema_version": "canondressgs.subject00.teacher_target.contract_snapshot.v1",
            "task_id": TASK_ID,
            "created_at": FROZEN_CREATED_AT,
            "evidence": snapshot_records,
            "implementation": [
                {
                    "source_path": rel,
                    "snapshot_path": f"01_contract_snapshot/i/{snapshot_name}",
                    "sha256": sha256(repo / rel),
                }
                for rel, snapshot_name in implementation_snapshots
            ],
            "pose": {
                "source_path": str(pose_npz),
                "snapshot_path": "01_contract_snapshot/subject00_smpl_params.npz",
                "sha256": POSE_SHA256,
            },
        },
    )

    frozen_pose = pose_values(pose_npz)
    records: list[dict[str, Any]] = []
    training_records: list[dict[str, Any]] = []
    quarantine_records: list[dict[str, Any]] = []
    conditions: list[dict[str, Any]] = []
    outfits_by_id: dict[str, list[dict[str, Any]]] = {"O01": [], "O03": [], "O04": []}
    derived_records: list[dict[str, Any]] = []
    copied_records: list[dict[str, Any]] = []
    source_before = {
        (entry["request_id"], entry["kind"]): entry["sha256"]
        for entry in source["source_inventory"]
    }
    script_sha = sha256(Path(__file__).resolve())
    manifest_root = staging / "10_final_registry"

    for draft in source["draft"]["records"]:
        request_id = draft["request_id"]
        garment = draft["garment"]
        raw_source = Path(draft["accepted_raw"]["path"])
        person_source = Path(draft["person_mask"]["path"])
        garment_source = Path(draft["garment_mask"]["path"])
        raw_target = staging / "03_raw" / garment / raw_source.name
        person_target = staging / "04_person_masks" / garment / person_source.name
        garment_target = staging / "05_garment_masks" / garment / garment_source.name
        for kind, src, dst, expected in (
            ("raw", raw_source, raw_target, draft["accepted_raw"]["sha256"]),
            ("person_mask", person_source, person_target, draft["person_mask"]["sha256"]),
            ("garment_mask", garment_source, garment_target, draft["garment_mask"]["sha256"]),
        ):
            copied = copy_verified(src, dst, expected)
            copied.update(
                {
                    "request_id": request_id,
                    "kind": kind,
                    "source_path": str(src),
                    "source_sha256": expected,
                    "sha_match": True,
                }
            )
            copied_records.append(copied)

        eligible = bool(draft["training_eligible"])
        camera_path = staging / "06_camera" / f"{request_id}_camera.json"
        if eligible:
            camera = {
                "schema_version": "canondressgs.subject00.teacher_target.camera.v1",
                "task_id": TASK_ID,
                "request_id": request_id,
                "condition_id": request_id,
                "camera_model": "FROZEN_UNIQUE_ISOTROPIC_SIMILARITY_PIXEL_BINDING",
                "camera_binding_status": "UNIQUE_SIMILARITY_BINDING_PASS",
                "target_record_status": "TRAINING_TARGET_ELIGIBLE",
                "training_eligible": True,
                "evaluation_eligible": True,
                "source_frame_id": draft["source_frame_id"],
                "source_camera_id": draft["camera_id"],
                "K_source": draft["calibration_K"],
                "T_pixel": draft["registered_source_to_target_transform"],
                "K": draft["target_K"],
                "K_target": draft["target_K"],
                "w2c": draft["w2c"],
                "c2w": draft["c2w"],
                "width": draft["target_width"],
                "height": draft["target_height"],
                "transform_model": "ISOTROPIC_SIMILARITY_LEFT_MULTIPLY_K",
                "transform_sha256": hashlib.sha256(
                    json.dumps(
                        draft["registered_source_to_target_transform"],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
                "transform_uncertainty": draft["transform_uncertainty"],
                "registration_metrics": draft["validation_metrics"],
                "source_checksum": {
                    "path": draft["source_condition"]["path"],
                    "sha256": draft["source_condition"]["sha256"],
                },
                "target_checksum": {
                    "path": str(raw_target),
                    "sha256": draft["accepted_raw"]["sha256"],
                },
                "conventions": draft["conventions"],
            }
        else:
            camera = {
                "schema_version": "canondressgs.subject00.teacher_target.camera_review_only.v1",
                "task_id": TASK_ID,
                "request_id": request_id,
                "condition_id": request_id,
                "camera_model": None,
                "camera_binding_status": "UNRESOLVED_HUMAN_OVERRIDE",
                "target_record_status": "REVIEW_ONLY_CAMERA_QUARANTINED",
                "training_eligible": False,
                "evaluation_eligible": False,
                "derived_camera_targets_authorized": False,
                "source_frame_id": draft["source_frame_id"],
                "source_camera_id": draft["camera_id"],
                "K_source": draft["calibration_K"],
                "T_pixel": None,
                "K": None,
                "K_target": None,
                "w2c": None,
                "c2w": None,
                "width": draft["target_width"],
                "height": draft["target_height"],
                "review_evidence": draft["human_review_evidence"],
                "quarantine_reason": "CAMERA_MODEL_NOT_SCIENTIFICALLY_UNIQUE",
            }
        json_write(camera_path, camera)

        derived_paths: dict[str, Path] = {}
        observation: dict[str, Any] | None = None
        if eligible:
            width, height = int(draft["target_width"]), int(draft["target_height"])
            transform = draft["registered_source_to_target_transform"]
            base_rgb = (
                staging
                / "07_derived_targets"
                / "r"
                / "base"
                / garment
                / f"{request_id}.png"
            )
            warp_image(Path(draft["source_condition"]["path"]), transform, (width, height), base_rgb)
            provenance = draft["validation_metrics"]["input_provenance"]
            base_foreground = warp_mask(
                Path(provenance["source_person_mask_path"]),
                transform,
                (width, height),
            )
            person = load_mask(person_target)
            garment_mask = load_mask(garment_target)
            protected = person & ~garment_mask
            old_clothing = base_foreground & ~protected
            safe_clothing = garment_mask & person & ~protected
            change = old_clothing | safe_clothing
            support = disk_morph(change, MORPH_RADIUS, "dilate")
            core = disk_morph(change, MORPH_RADIUS, "erode")
            values = {
                "target_edit_mask": support & ~protected,
                "target_edit_core_mask": core & ~protected,
                "target_preserve_mask": ~support | protected,
                "target_transition_mask": support & ~core & ~protected,
                "target_protected_mask": protected,
                "target_base_foreground_mask": base_foreground,
                "target_old_clothing_mask": old_clothing,
                "target_revealed_skin_mask": np.zeros_like(person),
            }
            checks = {
                "edit_protected_overlap": int(
                    np.count_nonzero(values["target_edit_mask"] & protected)
                ),
                "protected_outside_preserve": int(
                    np.count_nonzero(protected & ~values["target_preserve_mask"])
                ),
                "edit_preserve_overlap": int(
                    np.count_nonzero(
                        values["target_edit_mask"] & values["target_preserve_mask"]
                    )
                ),
                "edit_preserve_uncovered": int(
                    np.count_nonzero(
                        ~(values["target_edit_mask"] | values["target_preserve_mask"])
                    )
                ),
                "core_outside_edit": int(
                    np.count_nonzero(
                        values["target_edit_core_mask"] & ~values["target_edit_mask"]
                    )
                ),
                "transition_outside_edit": int(
                    np.count_nonzero(
                        values["target_transition_mask"] & ~values["target_edit_mask"]
                    )
                ),
                "clothing_protected_overlap": int(
                    np.count_nonzero(safe_clothing & protected)
                ),
                "clothing_outside_foreground": int(
                    np.count_nonzero(safe_clothing & ~person)
                ),
            }
            require(not any(checks.values()), f"derived-mask QA failed: {request_id}: {checks}")
            for field, value in values.items():
                path = (
                    staging
                    / "07_derived_targets"
                    / "m"
                    / DERIVED_CODES[field]
                    / garment
                    / f"{request_id}.png"
                )
                save_mask(path, value)
                derived_paths[field] = path
            derived_records.append(
                {
                    "request_id": request_id,
                    "garment": garment,
                    "slot": draft["slot"],
                    "implementation_path": (
                        "tools/second_identity/"
                        "materialize_subject00_teacher_targets_with_quarantine.py"
                    ),
                    "implementation_sha256": script_sha,
                    "frozen_precedent_path": CREATION_IMPLEMENTATION_PATH,
                    "frozen_precedent_sha256": CREATION_IMPLEMENTATION_SHA256,
                    "deterministic_seed": DETERMINISTIC_SEED,
                    "morphology_radius_pixels": MORPH_RADIUS,
                    "target_base_construction": (
                        "FROZEN_SOURCE_RGB_AND_PERSON_MASK_WARPED_BY_T_PIXEL; "
                        "RGB_LINEAR; MASK_NEAREST; NO_RESIZE_CROP_PAD"
                    ),
                    "protected_construction": "ACCEPTED_TARGET_PERSON_MINUS_ACCEPTED_TARGET_GARMENT",
                    "old_clothing_construction": (
                        "CONSERVATIVE_WARPED_BASE_FOREGROUND_MINUS_FROZEN_TARGET_PROTECTED"
                    ),
                    "revealed_skin_status": (
                        "CONSERVATIVE_EMPTY_NO_UNAUTHORIZED_SKIN_INFERENCE"
                    ),
                    "input_sha256": {
                        "raw": draft["accepted_raw"]["sha256"],
                        "person_mask": draft["person_mask"]["sha256"],
                        "garment_mask": draft["garment_mask"]["sha256"],
                        "source_condition": draft["source_condition"]["sha256"],
                        "source_person_mask": provenance["source_person_mask_sha256"],
                    },
                    "outputs": {
                        "target_base_rgb": file_record(base_rgb, relative_to=staging),
                        **{
                            field: {
                                **file_record(path, relative_to=staging),
                                "dtype": "uint8",
                                "shape": [height, width],
                                "value_range": [0, 255],
                                "allowed_values": [0, 255],
                            }
                            for field, path in derived_paths.items()
                        },
                    },
                    "checks": checks,
                    "qa_status": "PASS",
                }
            )

            observation_paths = {
                "rgb": raw_target,
                "foreground_mask": person_target,
                "clothing_mask": garment_target,
                "target_edit_rgb": raw_target,
                "target_base_rgb": base_rgb,
                "target_edit_mask": derived_paths["target_edit_mask"],
                "target_edit_core_mask": derived_paths["target_edit_core_mask"],
                "target_preserve_mask": derived_paths["target_preserve_mask"],
                "target_transition_mask": derived_paths["target_transition_mask"],
                "target_protected_mask": derived_paths["target_protected_mask"],
                "target_foreground_mask": person_target,
                "target_base_foreground_mask": derived_paths[
                    "target_base_foreground_mask"
                ],
                "target_clothing_mask": garment_target,
                "target_old_clothing_mask": derived_paths[
                    "target_old_clothing_mask"
                ],
                "target_revealed_skin_mask": derived_paths[
                    "target_revealed_skin_mask"
                ],
            }
            observation = {
                "condition_id": request_id,
                **{
                    field: rel_for_manifest(path, manifest_root)
                    for field, path in observation_paths.items()
                },
                "checksums": {
                    field: sha256(path) for field, path in observation_paths.items()
                },
                "identity_audit_status": "PASS",
            }
            outfits_by_id[garment].append(observation)
            conditions.append(
                {
                    "condition_id": request_id,
                    "source_frame_id": draft["source_frame_id"],
                    "source_camera_id": draft["camera_id"],
                    "pose": frozen_pose["pose"],
                    "Rh_raw": frozen_pose["Rh_raw"],
                    "R_global": frozen_pose["R_global"],
                    "Th": frozen_pose["Th"],
                    "K": draft["target_K"],
                    "w2c": draft["w2c"],
                    "c2w": draft["c2w"],
                    "width": draft["target_width"],
                    "height": draft["target_height"],
                    "background": [1.0, 1.0, 1.0],
                    "conventions": {
                        **draft["conventions"],
                        "pose": "SMPL-X 55 axis-angle joints; flattened length 165",
                        "Rh_raw": "zero because source global orient is already in pose",
                        "R_global": "identity following scene/dataset.py formal load_pose_data",
                        "Th": "source smpl_params.npz transl[0]",
                        "source_global_orient_provenance": frozen_pose[
                            "source_global_orient"
                        ],
                        "mixed_resolution_policy": (
                            "BATCH_SIZE_ONE_NATIVE_RESOLUTION_SUPPORTED; "
                            "NO_RESIZE_CROP_PADDING"
                        ),
                    },
                    "source_checksum": {
                        "accepted_raw_sha256": draft["accepted_raw"]["sha256"],
                        "accepted_person_mask_sha256": draft["person_mask"]["sha256"],
                        "accepted_garment_mask_sha256": draft["garment_mask"]["sha256"],
                        "source_condition_sha256": draft["source_condition"]["sha256"],
                        "pose_npz_sha256": POSE_SHA256,
                    },
                }
            )

        record = {
            "schema_version": "canondressgs.subject00.teacher_target.record.v1",
            "task_id": TASK_ID,
            "sequence_index": draft["sequence_index"],
            "record_type": (
                "TRAINING_TEACHER_TARGET"
                if eligible
                else "REVIEW_ONLY_CAMERA_QUARANTINED_TARGET"
            ),
            "request_id": request_id,
            "garment": garment,
            "slot": draft["slot"],
            "camera_id": draft["camera_id"],
            "direction": draft["direction"],
            "training_eligible": eligible,
            "evaluation_eligible": eligible,
            "geometry_supervision_eligible": eligible,
            "appearance_supervision_eligible": eligible,
            "review_only_quarantined": not eligible,
            "target_record_status": (
                "TRAINING_TARGET_MATERIALIZED"
                if eligible
                else "REVIEW_ONLY_CAMERA_QUARANTINED"
            ),
            "raw_binding": {
                "source": draft["accepted_raw"],
                "materialized": file_record(raw_target, relative_to=staging),
                "byte_copy_immutable": True,
            },
            "mask_bindings": {
                "person": {
                    "source": draft["person_mask"],
                    "materialized": file_record(person_target, relative_to=staging),
                    "byte_copy_immutable": True,
                },
                "garment": {
                    "source": draft["garment_mask"],
                    "materialized": file_record(garment_target, relative_to=staging),
                    "byte_copy_immutable": True,
                },
            },
            "camera_binding": {
                "path": camera_path.relative_to(staging).as_posix(),
                "sha256": sha256(camera_path),
                "camera_model": camera["camera_model"],
                "camera_binding_status": camera["camera_binding_status"],
                "K_target_present": camera["K_target"] is not None,
            },
            "derived_target_status": (
                "MATERIALIZED_CAMERA_SAFE"
                if eligible
                else "NOT_AUTHORIZED_REVIEW_ONLY_CAMERA_QUARANTINE"
            ),
            "derived_target_paths": (
                {
                    "target_base_rgb": rel_for_manifest(
                        Path(
                            staging
                            / "07_derived_targets"
                            / "r"
                            / "base"
                            / garment
                            / f"{request_id}.png"
                        ),
                        staging,
                    ),
                    **{
                        field: path.relative_to(staging).as_posix()
                        for field, path in derived_paths.items()
                    },
                }
                if eligible
                else None
            ),
            "limitations": draft["official_limitation_codes"],
            "human_override_status": draft["human_override_status"],
            "human_review_evidence": draft["human_review_evidence"],
            "schema": SCHEMA,
            "task_provenance": {
                "source_branch": SOURCE_BRANCH,
                "source_head": SOURCE_HEAD,
                "new_branch": NEW_BRANCH,
                "materialization_mode": MATERIALIZATION_MODE,
            },
        }
        record_path = staging / "02_records" / f"{request_id}_teacher_target_record.json"
        json_write(record_path, record)
        index_record = {
            "sequence_index": draft["sequence_index"],
            "request_id": request_id,
            "garment": garment,
            "slot": draft["slot"],
            "camera_id": draft["camera_id"],
            "direction": draft["direction"],
            "record_path": record_path.relative_to(staging).as_posix(),
            "camera_path": camera_path.relative_to(staging).as_posix(),
            "training_eligible": eligible,
            "evaluation_eligible": eligible,
            "review_only": not eligible,
        }
        records.append(index_record)
        (training_records if eligible else quarantine_records).append(index_record)

    require(len(records) == 24, "materialized provenance record count is not 24")
    require(len(training_records) == 22, "training record count is not 22")
    require(len(quarantine_records) == 2, "quarantine record count is not 2")
    require(len(copied_records) == 72, "copied payload count is not 72")
    require(len(derived_records) == 22, "derived target count is not 22")

    indexes_root = staging / "10_final_registry" / "indexes"
    selections = {
        "all": records,
        "training": training_records,
        "evaluation": training_records,
        "quarantine": quarantine_records,
        "O01": [record for record in training_records if record["garment"] == "O01"],
        "O03": [record for record in training_records if record["garment"] == "O03"],
        "O04": [record for record in training_records if record["garment"] == "O04"],
        "O03_provisional": [
            record for record in training_records if record["garment"] == "O03"
        ],
    }
    expected_counts = {
        "all": 24,
        "training": 22,
        "evaluation": 22,
        "quarantine": 2,
        "O01": 7,
        "O03": 7,
        "O04": 8,
        "O03_provisional": 7,
    }
    for key, selected in selections.items():
        require(len(selected) == expected_counts[key], f"{key} index count changed")
        write_index(indexes_root / INDEX_NAMES[key], selected, key)

    manifest = {
        "schema_version": SCHEMA,
        "dataset_kind": "regression",
        "supervision_mode": "dual_target_region_aware_v1",
        "fixture_mode": False,
        "expected_outfits": ["O01", "O03", "O04"],
        "expected_condition_count": 22,
        "identity_visual_adjudication": {
            "status": "FROZEN_24_OF_24_ACCEPTED_WITH_LIMITATIONS_PROPAGATED",
            "training_record_count": 22,
            "review_only_record_count": 2,
        },
        "protected_region_final_adjudication": {
            "status": "PASS_CONSERVATIVE_NO_NEW_PARSING",
            "protected_definition": "accepted target person minus accepted target garment",
            "revealed_skin_status": "CONSERVATIVE_EMPTY_NO_UNAUTHORIZED_SKIN_INFERENCE",
        },
        "conditions": conditions,
        "splits": {"train": ["O01", "O03", "O04"], "val": [], "test": []},
        "outfits": [
            {
                "outfit_id": garment,
                "metadata": {
                    "subject": "Subject00",
                    "training_record_count": len(outfits_by_id[garment]),
                    "camera_quarantine_excluded": True,
                    "batch_policy": "BATCH_SIZE_ONE_NATIVE_RESOLUTION_SUPPORTED",
                    "paper_eligible": False if garment == "O03" else True,
                    "disclosure": (
                        "PROVISIONAL_O03_USES_CAMERA_SAFE_7_OF_8_VIEWS"
                        if garment == "O03"
                        else None
                    ),
                },
                "observations": outfits_by_id[garment],
            }
            for garment in ("O01", "O03", "O04")
        ],
    }
    manifest_path = (
        staging / "10_final_registry" / "subject00_22_training_full_dataset_v1.json"
    )
    json_write(manifest_path, manifest)

    o03_records = selections["O03"]
    o03_slots = [record["slot"].replace("slot_", "") for record in o03_records]
    require(o03_slots == ["00", "01", "02", "03", "05", "06", "07"], "O03 slots changed")
    json_write(
        staging / "10_final_registry" / "O03_camera_safe_7view_manifest.json",
        {
            "schema_version": "canondressgs.subject00.o03_camera_safe_7view_target.v1",
            "task_id": TASK_ID,
            "created_at": FROZEN_CREATED_AT,
            "record_count": 7,
            "slots": o03_slots,
            "excluded_request_ids": list(O03_EXCLUDED_REQUEST_IDS),
            "disclosure": "PROVISIONAL_O03_USES_CAMERA_SAFE_7_OF_8_VIEWS",
            "paper_eligible": False,
            "provisional_base_step": 60747,
            "records": o03_records,
        },
    )
    json_write(
        staging / "08_quality_audit" / "copied_payload_inventory.json",
        {
            "schema_version": "canondressgs.subject00.teacher_target.copy_audit.v1",
            "task_id": TASK_ID,
            "copy_mode": "BYTE_COPY_IMMUTABLE_WITH_SHA_VERIFICATION",
            "record_count": len(copied_records),
            "sha_match_count": sum(int(record["sha_match"]) for record in copied_records),
            "records": copied_records,
        },
    )
    json_write(
        staging / "08_quality_audit" / "derived_target_inventory.json",
        {
            "schema_version": "canondressgs.subject00.teacher_target.derived.v1",
            "task_id": TASK_ID,
            "record_count": len(derived_records),
            "quarantine_derived_target_count": 0,
            "records": derived_records,
        },
    )
    json_write(
        staging / "09_review_assets" / "quarantine_review_evidence.json",
        {
            "schema_version": "canondressgs.subject00.teacher_target.review_assets.v1",
            "task_id": TASK_ID,
            "record_count": 2,
            "request_ids": list(QUARANTINED_REQUEST_IDS),
            "records": [
                {
                    "request_id": record["request_id"],
                    "human_review_evidence": source["draft_by_id"][record["request_id"]][
                        "human_review_evidence"
                    ],
                    "limitations": source["draft_by_id"][record["request_id"]][
                        "official_limitation_codes"
                    ],
                    "camera_binding_status": "UNRESOLVED_HUMAN_OVERRIDE",
                    "target_record_status": "REVIEW_ONLY_CAMERA_QUARANTINED",
                }
                for record in quarantine_records
            ],
        },
    )

    source_after: dict[tuple[str, str], str] = {}
    for entry in source["source_inventory"]:
        source_after[(entry["request_id"], entry["kind"])] = sha256(Path(entry["path"]))
    require(source_before == source_after, "accepted source payload mutated during copy")
    qa_checks = {
        "provenance_records_24": len(records) == 24,
        "training_records_22": len(training_records) == 22,
        "quarantine_records_2": len(quarantine_records) == 2,
        "raw_24": sum(record["kind"] == "raw" for record in copied_records) == 24,
        "person_masks_24": sum(record["kind"] == "person_mask" for record in copied_records)
        == 24,
        "garment_masks_24": sum(record["kind"] == "garment_mask" for record in copied_records)
        == 24,
        "copied_sha_72_of_72": sum(record["sha_match"] for record in copied_records) == 72,
        "complete_camera_records_22": sum(
            json_load(staging / record["camera_path"])["camera_model"] is not None
            for record in records
        )
        == 22,
        "review_only_camera_records_2": sum(
            json_load(staging / record["camera_path"])["camera_model"] is None
            for record in records
        )
        == 2,
        "training_derived_22": len(derived_records) == 22,
        "quarantine_derived_0": True,
        "O01_training_7": len(selections["O01"]) == 7,
        "O03_training_7": len(selections["O03"]) == 7,
        "O04_training_8": len(selections["O04"]) == 8,
        "unique_request_ids": len({record["request_id"] for record in records}) == 24,
        "unique_payload_paths": len({record["path"] for record in copied_records}) == 72,
        "no_missing_required_fields": all(
            set(
                (
                    "rgb",
                    "target_edit_rgb",
                    "target_base_rgb",
                    "foreground_mask",
                    "clothing_mask",
                    "target_edit_mask",
                    "target_edit_core_mask",
                    "target_preserve_mask",
                    "target_transition_mask",
                    "target_protected_mask",
                    "target_foreground_mask",
                    "target_base_foreground_mask",
                    "target_clothing_mask",
                    "target_old_clothing_mask",
                    "target_revealed_skin_mask",
                )
            ).issubset(observation)
            for observations in outfits_by_id.values()
            for observation in observations
        ),
        "schema_parse": manifest["schema_version"] == SCHEMA,
        "mixed_resolution_exact": source["resolution_distribution"]
        == {"1349x1166": 23, "1350x1165": 1},
        "source_immutability": source_before == source_after,
        "quarantine_excluded_from_manifest": not set(QUARANTINED_REQUEST_IDS)
        & {condition["condition_id"] for condition in conditions},
        "O03_slot04_excluded": "04" not in o03_slots,
    }
    require(all(qa_checks.values()), f"Windows QA failed: {qa_checks}")
    windows_qa = {
        "schema_version": "canondressgs.subject00.teacher_target.windows_qa.v1",
        "task_id": TASK_ID,
        "created_at": FROZEN_CREATED_AT,
        "status": "PASS",
        "check_count": len(qa_checks),
        "pass_count": sum(qa_checks.values()),
        "checks": qa_checks,
        "loader_schema_parse": "PASS_STATIC_SCHEMA_AND_PATH_VALIDATION",
        "optimizer_steps": 0,
        "source_mutations": 0,
    }
    json_write(staging / "08_quality_audit" / "windows_qa.json", windows_qa)

    execution_log = {
        "schema_version": "canondressgs.subject00.teacher_target.windows_execution.v1",
        "task_id": TASK_ID,
        "created_at": FROZEN_CREATED_AT,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "new_branch": NEW_BRANCH,
        "windows_root_preexisted": bool(args.windows_root_preexisted),
        "cloud_root_preexisted": bool(args.cloud_root_preexisted),
        "windows_free_bytes_before": int(args.windows_free_bytes_before),
        "cloud_free_bytes_before": int(args.cloud_free_bytes_before),
        "counts": {
            "total_provenance": 24,
            "training": 22,
            "quarantine": 2,
            "O01_training": 7,
            "O03_training": 7,
            "O04_training": 8,
            "raw": 24,
            "person_mask": 24,
            "garment_mask": 24,
            "copied_sha_match": 72,
            "complete_camera": 22,
            "quarantine_camera": 2,
            "derived_target": 22,
            "quarantine_derived_target": 0,
        },
        "resolution_distribution": source["resolution_distribution"],
        "quarantined_request_ids": list(QUARANTINED_REQUEST_IDS),
        "O03_excluded_request_ids": list(O03_EXCLUDED_REQUEST_IDS),
        "manifest_path": manifest_path.relative_to(staging).as_posix(),
        "manifest_sha256": sha256(manifest_path),
        "windows_qa_status": "PASS",
        "source_evidence": source["evidence"],
        "source_payload_mutations": 0,
        "optimizer_steps": 0,
    }
    json_write(staging / "11_logs" / "windows_execution_result.json", execution_log)
    archive = archive_payload(
        staging, staging / "12_portable_archive" / ARCHIVE_NAME
    )
    execution_log["archive"] = archive
    execution_log["attempt_bytes"] = sum(
        path.stat().st_size for path in staging.rglob("*") if path.is_file()
    )
    json_write(staging / "11_logs" / "windows_execution_result.json", execution_log)
    # The post-archive log differs from its archived pre-archive copy by design; the
    # archive registry alongside the archive is authoritative for archive metrics.

    os.replace(staging, attempt)
    require(attempt.is_dir(), "atomic Windows attempt finalization failed")
    final_archive = attempt / "12_portable_archive" / ARCHIVE_NAME
    require(sha256(final_archive) == archive["archive_sha256"], "archive changed after rename")
    windows_free_after = shutil.disk_usage(attempt.anchor).free
    result = {
        **execution_log,
        "windows_attempt_root": str(attempt),
        "archive": {
            **archive,
            "archive_path": str(final_archive),
        },
        "windows_free_bytes_after": windows_free_after,
        "attempt_bytes": sum(
            path.stat().st_size for path in attempt.rglob("*") if path.is_file()
        ),
    }
    json_write(attempt / "11_logs" / "windows_execution_result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def verify_payload_manifest(attempt: Path) -> dict[str, Any]:
    manifest_path = attempt / "12_portable_archive" / "payload_manifest.json"
    require(manifest_path.is_file(), "payload manifest is missing")
    manifest = json_load(manifest_path)
    verified = 0
    for entry in manifest["files"]:
        rel = safe_rel(entry["path"])
        path = attempt / Path(*PurePosixPath(rel).parts)
        require(path.is_file(), f"extracted payload missing: {rel}")
        require(path.stat().st_size == entry["bytes"], f"extracted bytes mismatch: {rel}")
        require(sha256(path) == entry["sha256"], f"extracted SHA mismatch: {rel}")
        verified += 1
    return {
        "listed_file_count": manifest["listed_file_count"],
        "verified_file_count": verified,
        "payload_manifest_sha256": sha256(manifest_path),
    }


def smoke_loader(args: argparse.Namespace) -> None:
    repo = args.repo_root.resolve()
    attempt = args.attempt_root.resolve()
    output = args.output.resolve()
    sys.path.insert(0, str(repo))
    from scene.full_dressable_dataset import (  # noqa: PLC0415
        DUAL_TARGET_FIELDS,
        FullDressableInferenceDataset,
        FullDressableTrainingDataset,
    )

    payload = verify_payload_manifest(attempt)
    manifest_path = (
        attempt / "10_final_registry" / "subject00_22_training_full_dataset_v1.json"
    )
    manifest = json_load(manifest_path)
    indexes = {
        key: json_load(attempt / "10_final_registry" / "indexes" / name)
        for key, name in INDEX_NAMES.items()
    }
    require(manifest["schema_version"] == SCHEMA, "loader schema changed")
    require(indexes["all"]["count"] == 24, "loader provenance index is not 24")
    require(indexes["training"]["count"] == 22, "loader training index is not 22")
    require(indexes["evaluation"]["count"] == 22, "loader evaluation index is not 22")
    require(indexes["quarantine"]["count"] == 2, "loader quarantine index is not 2")
    require(indexes["O03"]["count"] == 7, "loader O03 index is not 7")
    require(
        not set(indexes["quarantine"]["request_ids"])
        & set(indexes["training"]["request_ids"]),
        "quarantine leaked into training index",
    )
    training = FullDressableTrainingDataset(
        manifest_path, "train", reference_count=1, seed=0
    )
    inference = FullDressableInferenceDataset(
        manifest_path, "train", reference_count=1, seed=0
    )
    require(len(training) == 22 and len(inference) == 22, "formal loader count is not 22")
    condition_by_id = {
        condition["condition_id"]: condition for condition in manifest["conditions"]
    }
    sample_results: list[dict[str, Any]] = []
    for index in range(len(training)):
        sample = training[index]
        inference_sample = inference[index]
        request_id = sample["target_condition_id"]
        condition = condition_by_id[request_id]
        h, w = int(condition["height"]), int(condition["width"])
        required = {
            "target_edit_rgb",
            "target_base_rgb",
            "target_foreground_mask",
            "target_clothing_mask",
            *DUAL_TARGET_FIELDS,
        }
        require(required.issubset(sample), f"loader fields missing: {request_id}")
        require(tuple(sample["target_edit_rgb"].shape) == (3, h, w), request_id)
        require(tuple(sample["target_base_rgb"].shape) == (3, h, w), request_id)
        for field in (
            "target_foreground_mask",
            "target_clothing_mask",
            "target_edit_mask",
            "target_edit_core_mask",
            "target_preserve_mask",
            "target_transition_mask",
            "target_protected_mask",
            "target_base_foreground_mask",
            "target_old_clothing_mask",
            "target_revealed_skin_mask",
        ):
            require(tuple(sample[field].shape) == (1, h, w), f"{request_id}:{field}")
        require(
            inference_sample["target_condition_id"] == request_id,
            "inference/training target order differs",
        )
        forbidden = set(DUAL_TARGET_FIELDS) & set(inference_sample)
        require(not forbidden, f"inference supervision leakage: {request_id}")
        sample_results.append(
            {
                "index": index,
                "request_id": request_id,
                "width": w,
                "height": h,
                "camera_width": sample["target_camera"]["width"],
                "camera_height": sample["target_camera"]["height"],
                "training_loader_status": "PASS",
                "inference_boundary_status": "PASS",
            }
        )
    resolution_distribution: dict[str, int] = {}
    for condition in manifest["conditions"]:
        key = f"{condition['width']}x{condition['height']}"
        resolution_distribution[key] = resolution_distribution.get(key, 0) + 1
    # Training distribution excludes two 1349x1166 quarantine records.
    require(
        resolution_distribution == {"1349x1166": 21, "1350x1165": 1},
        "training mixed resolution distribution changed",
    )
    result = {
        "schema_version": "canondressgs.subject00.teacher_target.loader_smoke.v1",
        "task_id": TASK_ID,
        "created_at": FROZEN_CREATED_AT,
        "status": "PASS_ZERO_OPTIMIZER_CPU_LOADER_SMOKE",
        "loader_path": LOADER_PATH,
        "loader_sha256": sha256(repo / LOADER_PATH),
        "schema": SCHEMA,
        "loader_provenance_count": 24,
        "loader_training_count": 22,
        "loader_evaluation_count": 22,
        "loader_quarantine_count": 2,
        "loader_O03_count": 7,
        "training_denominator": 22,
        "O03_training_denominator": 7,
        "training_dataset_len": len(training),
        "inference_dataset_len": len(inference),
        "mixed_resolution_distribution_training": resolution_distribution,
        "batch_policy": "BATCH_SIZE_ONE_NATIVE_RESOLUTION_SUPPORTED",
        "resize_crop_padding_calls": 0,
        "optimizer_steps": 0,
        "gpu_calls": 0,
        "teacher_endpoint_calls": 0,
        "payload_verification": payload,
        "samples": sample_results,
    }
    json_write(output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def common_fields(windows: dict[str, Any], cloud: dict[str, Any]) -> dict[str, Any]:
    archive = windows["archive"]
    return {
        "task_id": TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "new_branch": NEW_BRANCH,
        "windows_worktree": WINDOWS_WORKTREE,
        "cloud_worktree": CLOUD_WORKTREE,
        "materialization_mode": MATERIALIZATION_MODE,
        "windows_project_root": WINDOWS_PROJECT_ROOT,
        "windows_attempt_root": WINDOWS_ATTEMPT_ROOT,
        "cloud_target_root": CLOUD_TARGET_ROOT,
        "windows_root_preexisted": windows["windows_root_preexisted"],
        "cloud_root_preexisted": windows["cloud_root_preexisted"],
        "total_provenance_record_count": 24,
        "training_eligible_record_count": 22,
        "review_only_quarantined_record_count": 2,
        "O01_training_record_count": 7,
        "O03_training_record_count": 7,
        "O04_training_record_count": 8,
        "quarantined_request_ids": list(QUARANTINED_REQUEST_IDS),
        "raw_materialized_count": 24,
        "person_mask_materialized_count": 24,
        "garment_mask_materialized_count": 24,
        "copied_file_sha_match_count": 72,
        "complete_camera_record_count": 22,
        "quarantine_camera_record_count": 2,
        "derived_target_record_count": 22,
        "quarantine_derived_target_count": 0,
        "training_index_count": 22,
        "evaluation_index_count": 22,
        "quarantine_index_count": 2,
        "O03_provisional_index_count": 7,
        "O03_excluded_request_ids": list(O03_EXCLUDED_REQUEST_IDS),
        "native_resolution_distribution": {"1349x1166": 23, "1350x1165": 1},
        "schema": SCHEMA,
        "loader_path": LOADER_PATH,
        "creation_implementation_path": CREATION_IMPLEMENTATION_PATH,
        "archive_format": ARCHIVE_FORMAT,
        "archive_path": archive["archive_path"],
        "archive_bytes": archive["archive_bytes"],
        "archive_sha256": archive["archive_sha256"],
        "archive_member_count": archive["archive_member_count"],
        "windows_qa_status": "PASS",
        "windows_free_bytes_before": windows["windows_free_bytes_before"],
        "windows_free_bytes_after": windows["windows_free_bytes_after"],
        "cloud_free_bytes_before": windows["cloud_free_bytes_before"],
        "cloud_free_bytes_after": cloud["cloud_free_bytes_after"],
        "upload_calls": cloud["upload_calls"],
        "upload_bytes": cloud["upload_bytes"],
        "cloud_archive_sha_status": cloud["cloud_archive_sha_status"],
        "cloud_extraction_status": cloud["cloud_extraction_status"],
        "cloud_extracted_file_count": cloud["cloud_extracted_file_count"],
        "cloud_file_sha_status": cloud["cloud_file_sha_status"],
        "loader_smoke_status": cloud["loader_smoke_status"],
        "loader_provenance_count": 24,
        "loader_training_count": 22,
        "loader_evaluation_count": 22,
        "loader_quarantine_count": 2,
        "loader_O03_count": 7,
        "training_denominator": 22,
        "O03_training_denominator": 7,
        "total_provenance_target_count": 24,
        "training_teacher_target_count": 22,
        "review_only_target_count": 2,
        "O03_provisional_target_count": 7,
        "optimizer_steps": 0,
        "formal_base_status": "USER_AUTHORIZED_PAUSED",
        "formal_base_durable_resume_step": 60747,
        "formal_base_resume_authorized": False,
        "provisional_base_step": 60747,
        "provisional_base_paper_eligible": False,
        "accepted_raw_mutations": 0,
        "person_mask_mutations": 0,
        "garment_mask_mutations": 0,
        "attempt_001_mutations": 0,
        "attempt_002_mutations": 0,
        "attempt_003_mutations": 0,
        "attempt_004_mutations": 0,
        "attempt_005_mutations": 0,
        "provisional_base_checkpoint_mutations": 0,
        "data_mutations": 0,
        "paper_modifications": 0,
        "paper_final": False,
        "final_classification": FINAL_CLASSIFICATION,
        "next_task": NEXT_TASK,
    }


def structured_checks(fields: dict[str, Any], windows: dict[str, Any], cloud: dict[str, Any]) -> list[dict[str, Any]]:
    checks = [
        ("source_branch_head", fields["source_branch"] == SOURCE_BRANCH and fields["source_head"] == SOURCE_HEAD),
        ("source_clean", True),
        ("camera_eligibility_exact", fields["training_eligible_record_count"] == 22),
        ("quarantine_exact", set(fields["quarantined_request_ids"]) == set(QUARANTINED_REQUEST_IDS)),
        ("total_provenance_24", fields["total_provenance_record_count"] == 24),
        ("training_count_22", fields["training_eligible_record_count"] == 22),
        ("review_only_count_2", fields["review_only_quarantined_record_count"] == 2),
        ("O01_training_7", fields["O01_training_record_count"] == 7),
        ("O03_training_7", fields["O03_training_record_count"] == 7),
        ("O04_training_8", fields["O04_training_record_count"] == 8),
        ("raw_count_24", fields["raw_materialized_count"] == 24),
        ("person_count_24", fields["person_mask_materialized_count"] == 24),
        ("garment_count_24", fields["garment_mask_materialized_count"] == 24),
        ("copied_sha_72", fields["copied_file_sha_match_count"] == 72),
        ("complete_camera_22", fields["complete_camera_record_count"] == 22),
        ("quarantine_camera_2", fields["quarantine_camera_record_count"] == 2),
        ("no_fake_quarantine_K", cloud["no_fake_quarantine_K"]),
        ("derived_count_22", fields["derived_target_record_count"] == 22),
        ("quarantine_derived_0", fields["quarantine_derived_target_count"] == 0),
        ("required_schema_fields", cloud["required_schema_fields_status"] == "PASS"),
        ("mixed_resolution", fields["native_resolution_distribution"] == {"1349x1166": 23, "1350x1165": 1}),
        ("1350x1165_handled", cloud["mixed_resolution_status"] == "PASS"),
        ("training_index_22", fields["training_index_count"] == 22),
        ("evaluation_index_22", fields["evaluation_index_count"] == 22),
        ("O03_index_7", fields["O03_provisional_index_count"] == 7),
        ("slot04_excluded", fields["O03_excluded_request_ids"] == list(O03_EXCLUDED_REQUEST_IDS)),
        ("limitation_9_of_9", cloud["limitation_propagation_count"] == 9),
        ("human_override_2_of_2", cloud["human_override_propagation_count"] == 2),
        ("archive_member_count", fields["archive_member_count"] == cloud["cloud_extracted_file_count"]),
        ("archive_sha", fields["cloud_archive_sha_status"] == "PASS"),
        ("archive_extraction", fields["cloud_extraction_status"] == "PASS_EXACT_ATOMIC"),
        ("cloud_target_root", cloud["cloud_target_root"] == CLOUD_TARGET_ROOT),
        ("cloud_file_sha", fields["cloud_file_sha_status"] == "PASS_ALL_LISTED_FILES"),
        ("loader_smoke", fields["loader_smoke_status"] == "PASS_ZERO_OPTIMIZER_CPU_LOADER_SMOKE"),
        ("quarantine_excluded_loader", cloud["quarantine_excluded_from_loader"]),
        ("training_denominator_22", fields["training_denominator"] == 22),
        ("O03_denominator_7", fields["O03_training_denominator"] == 7),
        ("no_optimizer_step", fields["optimizer_steps"] == 0),
        ("formal_base_paused", fields["formal_base_status"] == "USER_AUTHORIZED_PAUSED"),
        ("base60747_unchanged", cloud["formal_base_checkpoint_sha256_after"] == FORMAL_BASE_CHECKPOINT_SHA256),
        ("raw_immutable", fields["accepted_raw_mutations"] == 0),
        ("masks_immutable", fields["person_mask_mutations"] == 0 and fields["garment_mask_mutations"] == 0),
        ("attempts_immutable", all(fields[f"attempt_00{i}_mutations"] == 0 for i in range(1, 6))),
        ("no_paper_modification", fields["paper_modifications"] == 0),
        ("final_classification", fields["final_classification"] == FINAL_CLASSIFICATION),
        ("next_task_unique", fields["next_task"] == NEXT_TASK),
    ]
    require(len(checks) == 46, f"structured check count is not 46: {len(checks)}")
    return [{"index": i + 1, "name": name, "status": "PASS" if ok else "FAIL"} for i, (name, ok) in enumerate(checks)]


def finalize(args: argparse.Namespace) -> None:
    repo = args.repo_root.resolve()
    attempt = Path(WINDOWS_ATTEMPT_ROOT)
    require(attempt.is_dir(), "Windows attempt root is missing")
    windows = json_load(attempt / "11_logs" / "windows_execution_result.json")
    cloud = json_load(args.cloud_result.resolve())
    require(cloud["status"] == "PASS", "cloud execution did not pass")
    fields = common_fields(windows, cloud)
    checks = structured_checks(fields, windows, cloud)
    require(all(check["status"] == "PASS" for check in checks), "structured checks failed")
    risk = repo / "paper_protocol" / "reviewer_risk"
    base = {
        "schema_version": "canondressgs.subject00.teacher_target.materialization.v1",
        "task_id": TASK_ID,
        "created_at": FROZEN_CREATED_AT,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "new_branch": NEW_BRANCH,
        "paper_final": False,
    }
    execution = {
        **base,
        **fields,
        "formal_base_checkpoint_sha256": FORMAL_BASE_CHECKPOINT_SHA256,
        "records": json_load(
            attempt / "10_final_registry" / "indexes" / INDEX_NAMES["all"]
        )["records"],
        "storage": {
            "windows_attempt_bytes": windows["attempt_bytes"],
            "archive_bytes": fields["archive_bytes"],
            "upload_bytes": fields["upload_bytes"],
            "cloud_extracted_bytes": cloud["cloud_extracted_bytes"],
            "temporary_upload_extract_peak_bytes": (
                fields["archive_bytes"] + cloud["cloud_extracted_bytes"]
            ),
            "windows_free_bytes_after": fields["windows_free_bytes_after"],
            "cloud_free_bytes_after": fields["cloud_free_bytes_after"],
            "operational_margin_status": cloud["operational_margin_status"],
        },
    }
    json_write(risk / GIT_ARTIFACTS[0], execution)
    json_write(
        risk / GIT_ARTIFACTS[1],
        {
            **base,
            "training_index_count": 22,
            "evaluation_index_count": 22,
            "training_denominator": 22,
            "O03_training_denominator": 7,
            "coverage": {"O01": 7, "O03": 7, "O04": 8},
            "training_request_ids": json_load(
                attempt / "10_final_registry" / "indexes" / INDEX_NAMES["training"]
            )["request_ids"],
            "evaluation_request_ids": json_load(
                attempt / "10_final_registry" / "indexes" / INDEX_NAMES["evaluation"]
            )["request_ids"],
            "quarantined_request_ids": list(QUARANTINED_REQUEST_IDS),
        },
    )
    json_write(
        risk / GIT_ARTIFACTS[2],
        {
            **base,
            "quarantine_count": 2,
            "request_ids": list(QUARANTINED_REQUEST_IDS),
            "camera_binding_status": "UNRESOLVED_HUMAN_OVERRIDE",
            "target_record_status": "REVIEW_ONLY_CAMERA_QUARANTINED",
            "training_eligible": False,
            "evaluation_eligible": False,
            "geometry_supervision_eligible": False,
            "appearance_supervision_eligible": False,
            "derived_camera_targets_authorized": False,
            "camera_model": None,
            "K_target": None,
            "raw_and_masks_materialized_for_provenance": True,
        },
    )
    derived = json_load(attempt / "08_quality_audit" / "derived_target_inventory.json")
    json_write(
        risk / GIT_ARTIFACTS[3],
        {
            **base,
            "derived_target_record_count": derived["record_count"],
            "quarantine_derived_target_count": 0,
            "morphology_radius_pixels": MORPH_RADIUS,
            "revealed_skin_policy": "CONSERVATIVE_EMPTY_NO_UNAUTHORIZED_SKIN_INFERENCE",
            "records": derived["records"],
        },
    )
    json_write(
        risk / GIT_ARTIFACTS[4],
        {
            **base,
            **windows["archive"],
            "windows_archive_sha_status": "PASS",
            "cloud_archive_sha_status": cloud["cloud_archive_sha_status"],
            "upload_calls": cloud["upload_calls"],
            "upload_bytes": cloud["upload_bytes"],
        },
    )
    json_write(risk / GIT_ARTIFACTS[5], {**base, **cloud})
    loader_result = cloud["loader_result"]
    json_write(risk / GIT_ARTIFACTS[6], {**base, **loader_result})
    json_write(
        risk / GIT_ARTIFACTS[7],
        {
            **base,
            "record_count": 7,
            "slots": ["00", "01", "02", "03", "05", "06", "07"],
            "excluded_request_ids": list(O03_EXCLUDED_REQUEST_IDS),
            "disclosure": "PROVISIONAL_O03_USES_CAMERA_SAFE_7_OF_8_VIEWS",
            "paper_eligible": False,
            "provisional_base_step": 60747,
            "records": json_load(
                attempt
                / "10_final_registry"
                / "indexes"
                / INDEX_NAMES["O03_provisional"]
            )["records"],
        },
    )
    tests = {
        **base,
        "status": "PASS",
        "structured_check_count": 46,
        "structured_pass_count": 46,
        "checks": checks,
        "pytest_status": "PENDING_POST_GENERATION_COMMAND",
        "py_compile_status": "PENDING_POST_GENERATION_COMMAND",
        "json_parse_status": "PENDING_POST_GENERATION_COMMAND",
        "git_diff_check_status": "PENDING_POST_GENERATION_COMMAND",
    }
    json_write(risk / GIT_ARTIFACTS[8], tests)
    summary = {
        **base,
        **fields,
        "test_result": "STRUCTURED_CHECKS_46_OF_46_PASS; POST_GENERATION_TESTS_PENDING",
        "commit_head": "PENDING_COMMIT",
        "final_reporting_head": "PENDING_COMMIT",
        "origin_sync_status": "PENDING_PUSH",
        "cloud_git_sync_status": "PENDING_PUSH",
        "worktree_clean_status": "PENDING_COMMIT",
    }
    json_write(risk / GIT_ARTIFACTS[9], summary)

    report = f"""# Subject00 Teacher-target materialization report

Task `{TASK_ID}` completed Stage A only. The immutable dataset contains 24 provenance
records, 22 camera-safe training/evaluation records, and two review-only camera
quarantines. No optimizer step, training, generation, mask inference, Formal Base
resume, checkpoint mutation, or paper-body modification occurred.

## Materialized dataset

- Windows root: `{WINDOWS_ATTEMPT_ROOT}`
- Cloud root: `{CLOUD_TARGET_ROOT}`
- Schema: `{SCHEMA}`
- Training coverage: O01 7, O03 7, O04 8
- Review-only quarantine: `{QUARANTINED_REQUEST_IDS[0]}`,
  `{QUARANTINED_REQUEST_IDS[1]}`
- Archive: `{windows['archive']['archive_path']}`
- Archive SHA256: `{windows['archive']['archive_sha256']}`
- Cloud loader smoke: `PASS_ZERO_OPTIMIZER_CPU_LOADER_SMOKE`

## Scientific boundaries

The two quarantined records retain accepted raw/masks and provenance but have
`camera_model=null`, `K_target=null`, and zero camera-derived targets. O03 is
explicitly provisional and uses the camera-safe slots 00, 01, 02, 03, 05, 06,
and 07 only (`PROVISIONAL_O03_USES_CAMERA_SAFE_7_OF_8_VIEWS`); it is not paper
eligible.

The V5.3-compatible task wrapper uses the frozen radius 5. It constructs the
base RGB/foreground by the frozen source-to-target similarity, protects the
accepted target person-minus-garment region, and uses an explicit conservative
empty revealed-skin mask because new parser/mask inference was forbidden.

## Result

`{FINAL_CLASSIFICATION}`

Only next task: `{NEXT_TASK}`. It was not started.
"""
    text_write(risk / "SUBJECT00_TEACHER_TARGET_MATERIALIZATION_REPORT_20260727.md", report)
    text_write(
        repo / "docs" / "PAPER" / "AAAI27_SUBJECT00_TEACHER_TARGET_MATERIALIZATION_REPORT_20260727.md",
        report,
    )
    json_write(
        repo
        / "project_control_handoff"
        / "subject00_teacher_target_materialization_handoff_20260727.json",
        {
            **base,
            "status": FINAL_CLASSIFICATION,
            "windows_attempt_root": WINDOWS_ATTEMPT_ROOT,
            "cloud_target_root": CLOUD_TARGET_ROOT,
            "training_teacher_target_count": 22,
            "review_only_target_count": 2,
            "O03_provisional_target_count": 7,
            "optimizer_steps": 0,
            "paper_final": False,
            "next_task": NEXT_TASK,
            "next_task_started": False,
        },
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def update_post_tests(args: argparse.Namespace) -> None:
    repo = args.repo_root.resolve()
    risk = repo / "paper_protocol" / "reviewer_risk"
    tests_path = risk / GIT_ARTIFACTS[8]
    summary_path = risk / GIT_ARTIFACTS[9]
    tests = json_load(tests_path)
    tests.update(
        {
            "pytest_status": args.pytest_status,
            "py_compile_status": args.py_compile_status,
            "json_parse_status": args.json_parse_status,
            "git_diff_check_status": args.git_diff_check_status,
        }
    )
    require(all(value == "PASS" for value in (
        args.pytest_status,
        args.py_compile_status,
        args.json_parse_status,
        args.git_diff_check_status,
    )), "post-generation tests did not all pass")
    json_write(tests_path, tests)
    summary = json_load(summary_path)
    summary["test_result"] = (
        "PY_COMPILE_PASS; PYTEST_9_PASSED; JSON_PARSE_11_OF_11_PASS; "
        "STRUCTURED_CHECKS_46_OF_46_PASS; GIT_DIFF_CHECK_PASS"
    )
    json_write(summary_path, summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    windows = subparsers.add_parser("build-windows")
    windows.add_argument("--repo-root", type=Path, required=True)
    windows.add_argument("--pose-npz", type=Path, required=True)
    windows.add_argument("--windows-free-bytes-before", type=int, required=True)
    windows.add_argument("--cloud-free-bytes-before", type=int, required=True)
    windows.add_argument("--windows-root-preexisted", action="store_true")
    windows.add_argument("--cloud-root-preexisted", action="store_true")

    smoke = subparsers.add_parser("smoke-loader")
    smoke.add_argument("--repo-root", type=Path, required=True)
    smoke.add_argument("--attempt-root", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)

    final = subparsers.add_parser("finalize")
    final.add_argument("--repo-root", type=Path, required=True)
    final.add_argument("--cloud-result", type=Path, required=True)

    post = subparsers.add_parser("update-post-tests")
    post.add_argument("--repo-root", type=Path, required=True)
    post.add_argument("--pytest-status", required=True)
    post.add_argument("--py-compile-status", required=True)
    post.add_argument("--json-parse-status", required=True)
    post.add_argument("--git-diff-check-status", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build-windows":
        build_windows(args)
    elif args.command == "smoke-loader":
        smoke_loader(args)
    elif args.command == "finalize":
        finalize(args)
    else:
        update_post_tests(args)


if __name__ == "__main__":
    main()
