#!/usr/bin/env python3
"""Freeze, validate, and verify the authorized Subject00 reclamation sets."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable


TASK_ID = "AAAI27-SUBJECT00-COMBINED-STORAGE-RECLAMATION-EXECUTION-001"
CHECKPOINT_HEAD = "b5a6ac95175c97c0649af06dbda88b464a10bc48"
STORAGE_HEAD = "34e91445ef45c001cebcd789a4a6a88cad7b9ad8"
EXECUTION_BRANCH = "research/subject00-storage-reclamation-execution-20260725"
CANARY_PATH = (
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-MMLPHUMAN-SHORT-CANARY-001/attempt_001/"
    "checkpoints/step_000000.pth"
)
CANARY_SHA256 = "29d0edb11474f4384b6d409141d17f5da91c98824bd070196d50804cd7cb480a"
FORMAL_PATHS = [
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-MMLPHUMAN-FORMAL-STRICT-SPLIT-001",
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-FORMAL-STRICT-SPLIT-BASE-001",
]
AVATAR_ARCHIVE_ROOT = r"E:\canondressgs_archive\cloud_datasets\AvatarReX"
CHECKPOINT_ARCHIVE_ROOT = r"E:\canondressgs_archive\cloud_checkpoints"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_destination(archive_root: str, source: str) -> str:
    relative = source.lstrip("/").replace("/", "\\")
    return archive_root.rstrip("\\") + "\\payload\\" + relative


def stable_metadata_root(archive_root: str, source: str) -> str:
    relative = source.lstrip("/").replace("/", "\\")
    return archive_root.rstrip("\\") + "\\metadata\\" + relative + ".archive"


def is_under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def tree_stats(
    path: Path, source_type: str, include_directory_blocks: bool = False
) -> dict[str, int]:
    if source_type == "file":
        stat = path.stat()
        return {
            "file_count": 1,
            "logical_bytes": stat.st_size,
            "allocated_bytes": stat.st_blocks * 512,
        }
    files = 0
    logical = 0
    allocated = 0
    for current, directories, names in os.walk(path, followlinks=False):
        if include_directory_blocks:
            allocated += Path(current).stat().st_blocks * 512
        directories[:] = [
            name for name in directories if not os.path.islink(os.path.join(current, name))
        ]
        for name in names:
            item = Path(current, name)
            if item.is_symlink():
                continue
            stat = item.stat()
            files += 1
            logical += stat.st_size
            allocated += stat.st_blocks * 512
    return {
        "file_count": files,
        "logical_bytes": logical,
        "allocated_bytes": allocated,
    }


def freeze(repo: Path, output: Path, cloud_free: int, windows_free: int) -> None:
    risk = repo / "paper_protocol/reviewer_risk"
    storage_plan = load_json(risk / "subject00_storage_migration_plan.json")["plan_c"]
    avatar_audit = load_json(risk / "subject00_storage_avatarrex_audit.json")
    checkpoint_archive = load_json(risk / "subject00_checkpoint_archive_candidates.json")
    deletion = load_json(risk / "subject00_checkpoint_deletion_candidates.json")
    duplicate_groups = load_json(risk / "subject00_checkpoint_duplicate_groups.json")
    inventory_path = risk / "subject00_checkpoint_full_inventory.json"
    inventory = load_json(inventory_path)

    source_items = {item["source"]: item for item in storage_plan["source_items"]}
    archive_info = storage_plan["archive_already_at_destination"]
    avatar_roles = {
        storage_plan["strict_source_retirement_scope"][0]: "AVATARREX_EXTRACTED_DATA",
        storage_plan["strict_source_retirement_scope"][1]: "AVATARREX_REPORTS",
        storage_plan["strict_source_retirement_scope"][2]: "AVATARREX_SOURCE_ARCHIVE",
    }
    avatar_set: list[dict[str, Any]] = []
    for source in storage_plan["strict_source_retirement_scope"]:
        if source in source_items:
            source_item = source_items[source]
            expected_files = source_item["file_count"]
            expected_logical = source_item["apparent_file_bytes"]
            expected_allocated = source_item["allocated_bytes"]
            expected_sha = None
            source_type = "directory"
        elif source == archive_info["cloud_source"]:
            expected_files = 1
            expected_logical = archive_info["logical_bytes"]
            expected_allocated = archive_info["allocated_cloud_bytes"]
            expected_sha = archive_info["sha256"]
            source_type = "file"
        else:
            raise ValueError(f"Unmapped AvatarReX retirement source: {source}")
        avatar_set.append(
            {
                "source_path": source,
                "source_role": avatar_roles[source],
                "source_type": source_type,
                "expected_file_count": expected_files,
                "expected_logical_bytes": expected_logical,
                "expected_allocated_bytes": expected_allocated,
                "allocated_bytes_policy": "FILES_AND_DIRECTORIES",
                "expected_file_sha256": expected_sha,
                "destination_path": stable_destination(AVATAR_ARCHIVE_ROOT, source),
                "metadata_root": stable_metadata_root(AVATAR_ARCHIVE_ROOT, source),
                "archive_reason": "AUTHORIZED_STORAGE_PLAN_C",
                "manifest_references": avatar_audit.get("references", {}),
                "pre_execution_validation_status": "PENDING_LIVE_VALIDATION",
            }
        )

    checkpoint_archive_set = []
    for item in checkpoint_archive["candidates"]:
        source = item["cloud_source"]
        checkpoint_archive_set.append(
            {
                "source_path": source,
                "source_role": "WHOLE_OLD_CHECKPOINT_EXPERIMENT",
                "source_type": "directory",
                "expected_file_count": item["file_count"],
                "expected_logical_bytes": item["logical_bytes"],
                "expected_allocated_bytes": item["allocated_bytes"],
                "allocated_bytes_policy": "FILES_ONLY",
                "expected_checkpoint_count": item["checkpoint_count"],
                "expected_checkpoint_logical_bytes": item["checkpoint_logical_bytes"],
                "destination_path": stable_destination(CHECKPOINT_ARCHIVE_ROOT, source),
                "adjudicated_destination_path": item["windows_destination"],
                "metadata_root": stable_metadata_root(CHECKPOINT_ARCHIVE_ROOT, source),
                "archive_reason": "AUTHORIZED_CHECKPOINT_PLAN_CKPT_D_WHOLE_EXPERIMENT",
                "manifest_references": [],
                "pre_execution_validation_status": "PENDING_LIVE_VALIDATION",
            }
        )

    group_by_candidate: dict[str, dict[str, Any]] = {}
    for group in duplicate_groups["groups"]:
        for candidate in group["safe_delete_candidate_paths"]:
            group_by_candidate[candidate] = group
    duplicate_set = []
    for item in deletion["duplicate_safe_delete_candidates"]:
        group = group_by_candidate[item["path"]]
        duplicate_set.append(
            {
                "source_path": item["path"],
                "source_role": "BYTE_IDENTICAL_DUPLICATE_CHECKPOINT",
                "expected_bytes": item["bytes"],
                "expected_sha256": item["sha256"],
                "retained_duplicate_path": group["recommended_keep_path"],
                "retained_duplicate_sha256": group["sha256"],
                "manifest_references": item["references"],
                "archive_or_delete_reason": item["classification_reason"],
                "pre_execution_validation_status": "PENDING_LIVE_VALIDATION",
            }
        )

    optimizer_set = []
    for item in deletion["temporary_optimizer_state_candidates"]:
        optimizer_set.append(
            {
                "source_path": item["path"],
                "source_role": "OPTIMIZER_ONLY_COMPLETED_RUN_STATE",
                "expected_bytes": item["bytes"],
                "expected_sha256": item["sha256"],
                "manifest_references": item["references"],
                "archive_or_delete_reason": item["classification_reason"],
                "pre_execution_validation_status": "PENDING_LIVE_VALIDATION",
            }
        )

    if deletion["regenerable_delete_candidates"]:
        raise ValueError("The adjudicated regenerable set must remain empty")
    if len(duplicate_set) != 70 or len(optimizer_set) != 1:
        raise ValueError("Unexpected compact deletion-set cardinality")

    planned_transfer = sum(item["expected_logical_bytes"] for item in avatar_set)
    planned_transfer += sum(
        item["expected_logical_bytes"] for item in checkpoint_archive_set
    )
    required_windows = planned_transfer + 20 * 1024**3
    result = {
        "schema_version": "canondressgs.subject00.combined_reclamation_execution_set.v1",
        "task_id": TASK_ID,
        "frozen_at_utc": utc_now(),
        "source_heads": {
            "checkpoint_branch": (
                "research/subject00-checkpoint-reclamation-adjudication-20260725"
            ),
            "checkpoint_head": CHECKPOINT_HEAD,
            "storage_branch": (
                "research/mmlphuman-subject00-storage-migration-adjudication-20260725"
            ),
            "storage_head": STORAGE_HEAD,
        },
        "execution_branch": EXECUTION_BRANCH,
        "checkpoint_inventory": {
            "path": "paper_protocol/reviewer_risk/subject00_checkpoint_full_inventory.json",
            "sha256": sha256_file(inventory_path),
            "checkpoint_count": inventory["checkpoint_count"],
            "checkpoint_total_bytes": inventory["checkpoint_total_bytes"],
        },
        "initial_capacity": {
            "cloud_free_bytes": cloud_free,
            "windows_e_free_bytes": windows_free,
            "planned_transfer_bytes": planned_transfer,
            "required_windows_bytes_including_20_gib_reserve": required_windows,
            "windows_capacity_gate_pass": windows_free >= required_windows,
        },
        "protected_baseline": {
            "short_canary_path": CANARY_PATH,
            "short_canary_sha256": CANARY_SHA256,
            "classification_summary": inventory["classification_summary"],
            "paper_or_figure_checkpoint_count": sum(
                item["reference_flags"]["paper_or_figure"]
                for item in inventory["checkpoints"]
            ),
            "figure_bank_checkpoint_count": sum(
                item["reference_flags"]["figure_bank"]
                for item in inventory["checkpoints"]
            ),
        },
        "avatarrex_archive_set": avatar_set,
        "checkpoint_archive_set": checkpoint_archive_set,
        "duplicate_delete_set": duplicate_set,
        "optimizer_delete_set": optimizer_set,
        "regenerable_delete_set": [],
        "set_totals": {
            "avatarrex_items": len(avatar_set),
            "avatarrex_transfer_bytes": sum(
                item["expected_logical_bytes"] for item in avatar_set
            ),
            "checkpoint_archive_experiments": len(checkpoint_archive_set),
            "checkpoint_archive_files": sum(
                item["expected_file_count"] for item in checkpoint_archive_set
            ),
            "checkpoint_archive_transfer_bytes": sum(
                item["expected_logical_bytes"] for item in checkpoint_archive_set
            ),
            "duplicate_files": len(duplicate_set),
            "duplicate_bytes": sum(item["expected_bytes"] for item in duplicate_set),
            "optimizer_files": len(optimizer_set),
            "optimizer_bytes": sum(item["expected_bytes"] for item in optimizer_set),
        },
        "execution_scope_expansion_allowed": False,
        "pre_execution_validation_status": "PENDING_LIVE_VALIDATION",
        "actual_deletion_count": 0,
    }
    if not result["initial_capacity"]["windows_capacity_gate_pass"]:
        raise RuntimeError("Windows capacity gate failed")
    write_json_atomic(output, result)


def open_fd_matches(roots: Iterable[str]) -> list[dict[str, Any]]:
    normalized = list(roots)
    matches: list[dict[str, Any]] = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            cmdline = (
                (process / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode("utf-8", "replace")
                .strip()
            )
            for fd in (process / "fd").iterdir():
                try:
                    target = os.path.realpath(fd)
                except OSError:
                    continue
                if any(is_under(target, root) for root in normalized):
                    matches.append(
                        {"pid": int(process.name), "cmdline": cmdline, "path": target}
                    )
        except (OSError, PermissionError):
            continue
    return matches


def matching_processes() -> list[dict[str, Any]]:
    patterns = (
        "train_dressable",
        "torchrun",
        "accelerate launch",
        "rsync",
        "rclone",
        "scp ",
        "run_subject00_formal",
        "formal_strict_split",
    )
    result = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            cmdline = (
                (process / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode("utf-8", "replace")
                .strip()
            )
        except OSError:
            continue
        lowered = cmdline.lower()
        if any(pattern in lowered for pattern in patterns) and "validate-live" not in lowered:
            result.append({"pid": int(process.name), "cmdline": cmdline})
    return result


def validate_live(execution_set_path: Path, output: Path) -> None:
    execution = load_json(execution_set_path)
    archive_items = execution["avatarrex_archive_set"] + execution[
        "checkpoint_archive_set"
    ]
    archive_results = []
    changed: list[str] = []
    all_candidate_roots = [item["source_path"] for item in archive_items]
    all_candidate_roots += [
        item["source_path"] for item in execution["duplicate_delete_set"]
    ]
    all_candidate_roots += [
        item["source_path"] for item in execution["optimizer_delete_set"]
    ]
    open_matches = open_fd_matches(all_candidate_roots)

    for item in archive_items:
        path = Path(item["source_path"])
        observed: dict[str, Any] = {
            "source_path": item["source_path"],
            "exists": path.exists(),
        }
        if path.exists():
            observed.update(
                tree_stats(
                    path,
                    item["source_type"],
                    item.get("allocated_bytes_policy") == "FILES_AND_DIRECTORIES",
                )
            )
            observed["file_sha256"] = (
                sha256_file(path) if item["source_type"] == "file" else None
            )
        matches = bool(
            observed["exists"]
            and observed.get("file_count") == item["expected_file_count"]
            and observed.get("logical_bytes") == item["expected_logical_bytes"]
            and observed.get("allocated_bytes") == item["expected_allocated_bytes"]
            and (
                not item.get("expected_file_sha256")
                or observed.get("file_sha256") == item["expected_file_sha256"]
            )
        )
        observed["validation_status"] = (
            "MATCHES_ADJUDICATION" if matches else "SOURCE_CHANGED_SINCE_ADJUDICATION"
        )
        if not matches:
            changed.append(item["source_path"])
        archive_results.append(observed)

    duplicate_results = []
    for item in execution["duplicate_delete_set"]:
        source = Path(item["source_path"])
        retained = Path(item["retained_duplicate_path"])
        observed = {
            "source_path": item["source_path"],
            "retained_duplicate_path": item["retained_duplicate_path"],
            "source_exists": source.is_file(),
            "retained_exists": retained.is_file(),
        }
        if source.is_file():
            observed["source_bytes"] = source.stat().st_size
            observed["source_sha256"] = sha256_file(source)
        if retained.is_file():
            observed["retained_bytes"] = retained.stat().st_size
            observed["retained_sha256"] = sha256_file(retained)
        matches = bool(
            observed["source_exists"]
            and observed["retained_exists"]
            and observed.get("source_bytes") == item["expected_bytes"]
            and observed.get("source_sha256") == item["expected_sha256"]
            and observed.get("retained_sha256") == item["retained_duplicate_sha256"]
            and observed.get("source_sha256") == observed.get("retained_sha256")
            and not item["manifest_references"]
        )
        observed["validation_status"] = (
            "MATCHES_ADJUDICATION" if matches else "SOURCE_CHANGED_SINCE_ADJUDICATION"
        )
        if not matches:
            changed.append(item["source_path"])
        duplicate_results.append(observed)

    optimizer_results = []
    for item in execution["optimizer_delete_set"]:
        path = Path(item["source_path"])
        observed = {"source_path": item["source_path"], "exists": path.is_file()}
        if path.is_file():
            observed["bytes"] = path.stat().st_size
            observed["sha256"] = sha256_file(path)
        matches = bool(
            observed["exists"]
            and observed.get("bytes") == item["expected_bytes"]
            and observed.get("sha256") == item["expected_sha256"]
            and not item["manifest_references"]
        )
        observed["validation_status"] = (
            "MATCHES_ADJUDICATION" if matches else "SOURCE_CHANGED_SINCE_ADJUDICATION"
        )
        if not matches:
            changed.append(item["source_path"])
        optimizer_results.append(observed)

    canary = Path(CANARY_PATH)
    canary_sha = sha256_file(canary) if canary.is_file() else None
    processes = matching_processes()
    formal_present = [path for path in FORMAL_PATHS if Path(path).exists()]
    status = "PASS"
    if changed or open_matches or processes or formal_present or canary_sha != CANARY_SHA256:
        status = "BLOCKED"
    result = {
        "schema_version": "canondressgs.subject00.combined_reclamation_live_validation.v1",
        "task_id": TASK_ID,
        "validated_at_utc": utc_now(),
        "archive_results": archive_results,
        "duplicate_results": duplicate_results,
        "optimizer_results": optimizer_results,
        "source_changed_paths": sorted(set(changed)),
        "open_candidate_files": open_matches,
        "blocking_processes": processes,
        "formal_paths_present": formal_present,
        "short_canary": {
            "path": CANARY_PATH,
            "expected_sha256": CANARY_SHA256,
            "observed_sha256": canary_sha,
            "matches": canary_sha == CANARY_SHA256,
        },
        "overall_status": status,
    }
    write_json_atomic(output, result)


def merge_validation(execution_path: Path, validation_path: Path) -> None:
    execution = load_json(execution_path)
    validation = load_json(validation_path)
    archive_by_path = {
        item["source_path"]: item for item in validation["archive_results"]
    }
    duplicate_by_path = {
        item["source_path"]: item for item in validation["duplicate_results"]
    }
    optimizer_by_path = {
        item["source_path"]: item for item in validation["optimizer_results"]
    }
    for key in ("avatarrex_archive_set", "checkpoint_archive_set"):
        for item in execution[key]:
            result = archive_by_path[item["source_path"]]
            item["pre_execution_validation_status"] = result["validation_status"]
            item["live_validation"] = result
    for item in execution["duplicate_delete_set"]:
        result = duplicate_by_path[item["source_path"]]
        item["pre_execution_validation_status"] = result["validation_status"]
        item["live_validation"] = result
    for item in execution["optimizer_delete_set"]:
        result = optimizer_by_path[item["source_path"]]
        item["pre_execution_validation_status"] = result["validation_status"]
        item["live_validation"] = result
    execution["live_validation"] = validation
    execution["pre_execution_validation_status"] = validation["overall_status"]
    write_json_atomic(execution_path, execution)


def iter_manifest_files(source: Path, source_type: str) -> Iterable[tuple[str, Path]]:
    if source_type == "file":
        yield source.name, source
        return
    for current, directories, names in os.walk(source, followlinks=False):
        directories[:] = sorted(
            name for name in directories if not os.path.islink(os.path.join(current, name))
        )
        for name in sorted(names):
            path = Path(current, name)
            if not path.is_symlink():
                yield path.relative_to(source).as_posix(), path


def build_manifest(source: Path, source_type: str, output: Path) -> None:
    rows = []
    for relative, path in iter_manifest_files(source, source_type):
        stat = path.stat()
        rows.append(
            {
                "relative_path": relative,
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "mode": stat.st_mode & 0o7777,
                "sha256": sha256_file(path),
            }
        )
    result = {
        "schema_version": "canondressgs.subject00.byte_exact_archive_manifest.v1",
        "task_id": TASK_ID,
        "created_at_utc": utc_now(),
        "source_path": str(source),
        "source_type": source_type,
        "source_branch": EXECUTION_BRANCH,
        "source_head": CHECKPOINT_HEAD,
        "file_count": len(rows),
        "logical_bytes": sum(row["bytes"] for row in rows),
        "files": rows,
        "restore_instruction": (
            "Restore each relative path to the recorded source_path, then require the same "
            "file count, logical bytes, and per-file SHA256 registry."
        ),
    }
    write_json_atomic(output, result)


def verify_manifest(manifest_path: Path, destination: Path, output: Path) -> None:
    manifest = load_json(manifest_path)
    expected = {item["relative_path"]: item for item in manifest["files"]}
    observed_rows = []
    destination_type = manifest["source_type"]
    for relative, path in iter_manifest_files(destination, destination_type):
        stat = path.stat()
        observed_rows.append(
            {
                "relative_path": relative,
                "bytes": stat.st_size,
                "sha256": sha256_file(path),
            }
        )
    observed = {item["relative_path"]: item for item in observed_rows}
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    mismatches = []
    for relative in sorted(set(expected) & set(observed)):
        if (
            expected[relative]["bytes"] != observed[relative]["bytes"]
            or expected[relative]["sha256"] != observed[relative]["sha256"]
        ):
            mismatches.append(relative)
    passed = not missing and not extra and not mismatches
    result = {
        "schema_version": "canondressgs.subject00.archive_manifest_verification.v1",
        "task_id": TASK_ID,
        "verified_at_utc": utc_now(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "source_path": manifest["source_path"],
        "destination_path": str(destination),
        "expected_file_count": manifest["file_count"],
        "destination_file_count": len(observed_rows),
        "expected_logical_bytes": manifest["logical_bytes"],
        "destination_logical_bytes": sum(item["bytes"] for item in observed_rows),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "sha_or_size_mismatch_count": len(mismatches),
        "missing_sample": missing[:20],
        "extra_sample": extra[:20],
        "mismatch_sample": mismatches[:20],
        "file_count_match": len(observed_rows) == manifest["file_count"],
        "total_bytes_match": (
            sum(item["bytes"] for item in observed_rows) == manifest["logical_bytes"]
        ),
        "sha256_all_match": passed,
        "overall_status": "PASS" if passed else "FAIL",
    }
    write_json_atomic(output, result)


def init_progress(execution_set_path: Path, output: Path) -> None:
    execution = load_json(execution_set_path)
    result = {
        "schema_version": "canondressgs.subject00.storage_reclamation_progress.v1",
        "task_id": TASK_ID,
        "execution_branch": EXECUTION_BRANCH,
        "execution_set_path": str(execution_set_path),
        "execution_set_sha256": sha256_file(execution_set_path),
        "created_at_utc": utc_now(),
        "current_phase": "EXECUTION_SET_FROZEN",
        "current_status": "PASS",
        "events": [
            {
                "sequence": 1,
                "timestamp_utc": utc_now(),
                "phase": "EXECUTION_SET_FROZEN",
                "status": "PASS",
                "note": (
                    "Exact live validation passed; no archive transfer or source deletion "
                    "had started at registry creation."
                ),
            }
        ],
        "archive_items": {
            item["source_path"]: "PENDING_TRANSFER"
            for item in execution["avatarrex_archive_set"]
            + execution["checkpoint_archive_set"]
        },
        "duplicate_items": {
            item["source_path"]: "PENDING_PRE_DELETION_GATE"
            for item in execution["duplicate_delete_set"]
        },
        "optimizer_items": {
            item["source_path"]: "PENDING_PRE_DELETION_GATE"
            for item in execution["optimizer_delete_set"]
        },
        "actual_deleted_paths": [],
        "checkpoint_deletion_count": 0,
    }
    write_json_atomic(output, result)


def append_progress_event(
    progress_path: Path, phase: str, status: str, note: str
) -> None:
    progress = load_json(progress_path)
    progress["events"].append(
        {
            "sequence": len(progress["events"]) + 1,
            "timestamp_utc": utc_now(),
            "phase": phase,
            "status": status,
            "note": note,
        }
    )
    progress["current_phase"] = phase
    progress["current_status"] = status
    write_json_atomic(progress_path, progress)


def archive_item_matches(item: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    path = Path(item["source_path"])
    observed: dict[str, Any] = {"source_path": str(path), "exists": path.exists()}
    if path.exists():
        observed.update(
            tree_stats(
                path,
                item["source_type"],
                item.get("allocated_bytes_policy") == "FILES_AND_DIRECTORIES",
            )
        )
        observed["file_sha256"] = (
            sha256_file(path) if item["source_type"] == "file" else None
        )
    matches = bool(
        observed["exists"]
        and observed.get("file_count") == item["expected_file_count"]
        and observed.get("logical_bytes") == item["expected_logical_bytes"]
        and observed.get("allocated_bytes") == item["expected_allocated_bytes"]
        and (
            not item.get("expected_file_sha256")
            or observed.get("file_sha256") == item["expected_file_sha256"]
        )
    )
    observed["matches_adjudication"] = matches
    return matches, observed


def verification_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["source_path"]: item for item in report.get("items", [])}


def build_pre_deletion_gate(
    execution_path: Path,
    avatar_verification_path: Path,
    checkpoint_verification_path: Path,
    inventory_path: Path,
    output: Path,
) -> None:
    execution = load_json(execution_path)
    avatar_verification = load_json(avatar_verification_path)
    checkpoint_verification = load_json(checkpoint_verification_path)
    inventory = load_json(inventory_path)

    live_path = output.with_name(output.name + ".live.tmp")
    validate_live(execution_path, live_path)
    live = load_json(live_path)
    live_path.unlink()

    avatar_map = verification_map(avatar_verification)
    checkpoint_map = verification_map(checkpoint_verification)
    expected_avatar = {
        item["source_path"] for item in execution["avatarrex_archive_set"]
    }
    expected_checkpoint = {
        item["source_path"] for item in execution["checkpoint_archive_set"]
    }
    verified_avatar = {
        path for path, item in avatar_map.items() if item.get("overall_status") == "PASS"
    }
    verified_checkpoint = {
        path
        for path, item in checkpoint_map.items()
        if item.get("overall_status") == "PASS"
    }

    duplicate_paths = {
        item["source_path"] for item in execution["duplicate_delete_set"]
    }
    optimizer_paths = {
        item["source_path"] for item in execution["optimizer_delete_set"]
    }
    checkpoint_roots = [
        item["source_path"] for item in execution["checkpoint_archive_set"]
    ]
    planned_checkpoint_rows = [
        row
        for row in inventory["checkpoints"]
        if row["path"] in duplicate_paths | optimizer_paths
        or any(is_under(row["path"], root) for root in checkpoint_roots)
    ]
    class_counts: dict[str, int] = {}
    for row in planned_checkpoint_rows:
        classification = row["classification"]
        class_counts[classification] = class_counts.get(classification, 0) + 1
    protected_classes = {
        "CRITICAL_ACTIVE_KEEP",
        "SEALED_PROVENANCE_KEEP",
        "UNIQUE_FINAL_KEEP",
        "UNKNOWN_KEEP",
    }
    protected_planned = [
        row["path"]
        for row in planned_checkpoint_rows
        if row["classification"] in protected_classes
    ]
    paper_planned = [
        row["path"]
        for row in planned_checkpoint_rows
        if row.get("reference_flags", {}).get("paper_or_figure")
    ]
    figure_planned = [
        row["path"]
        for row in planned_checkpoint_rows
        if row.get("reference_flags", {}).get("figure_bank")
    ]
    sealed_planned = [
        row["path"] for row in planned_checkpoint_rows if row.get("sealed_attempt")
    ]

    checks = {
        "task_id_matches": execution.get("task_id") == TASK_ID,
        "execution_scope_is_frozen": not execution.get(
            "execution_scope_expansion_allowed", True
        ),
        "live_validation_pass": live.get("overall_status") == "PASS",
        "avatarrex_archive_all_verified": verified_avatar == expected_avatar,
        "checkpoint_archive_all_verified": verified_checkpoint == expected_checkpoint,
        "duplicate_count_matches": len(duplicate_paths) == 70,
        "optimizer_count_matches": len(optimizer_paths) == 1,
        "paper_or_figure_checkpoint_delete_count_zero": not paper_planned,
        "figure_bank_checkpoint_delete_count_zero": not figure_planned,
        "sealed_provenance_delete_count_zero": not sealed_planned,
        "protected_class_delete_count_zero": not protected_planned,
        "archive_checkpoint_class_count_matches": class_counts.get(
            "ARCHIVE_THEN_DELETE_CANDIDATE", 0
        )
        == 11,
        "duplicate_class_count_matches": class_counts.get(
            "DUPLICATE_SAFE_DELETE_CANDIDATE", 0
        )
        == 70,
        "optimizer_class_count_matches": class_counts.get(
            "TEMPORARY_OPTIMIZER_STATE_CANDIDATE", 0
        )
        == 1,
        "short_canary_matches": live.get("short_canary", {}).get("matches", False),
        "no_active_training_or_transfer": not live.get("blocking_processes"),
        "no_candidate_open_files": not live.get("open_candidate_files"),
        "no_formal_attempt": not live.get("formal_paths_present"),
    }
    passed = all(checks.values())
    result = {
        "schema_version": "canondressgs.subject00.pre_deletion_final_gate.v1",
        "task_id": TASK_ID,
        "generated_at_utc": utc_now(),
        "execution_set_path": str(execution_path),
        "execution_set_sha256": sha256_file(execution_path),
        "avatarrex_verification_path": str(avatar_verification_path),
        "avatarrex_verification_sha256": sha256_file(avatar_verification_path),
        "checkpoint_verification_path": str(checkpoint_verification_path),
        "checkpoint_verification_sha256": sha256_file(
            checkpoint_verification_path
        ),
        "checks": checks,
        "planned_checkpoint_class_counts": class_counts,
        "protected_planned_paths": protected_planned,
        "paper_or_figure_planned_paths": paper_planned,
        "figure_bank_planned_paths": figure_planned,
        "sealed_planned_paths": sealed_planned,
        "archive_deletion_eligible_paths": sorted(
            expected_avatar | expected_checkpoint
        ),
        "duplicate_deletion_eligible_paths": sorted(duplicate_paths),
        "optimizer_deletion_eligible_paths": sorted(optimizer_paths),
        "live_validation": live,
        "overall_status": "PASS" if passed else "BLOCKED",
    }
    write_json_atomic(output, result)
    if not passed:
        raise RuntimeError("Pre-deletion final gate is blocked")


def delete_archive_item(
    execution_path: Path, gate_path: Path, source_path: str, output: Path
) -> None:
    execution = load_json(execution_path)
    gate = load_json(gate_path)
    items = execution["avatarrex_archive_set"] + execution["checkpoint_archive_set"]
    by_path = {item["source_path"]: item for item in items}
    if source_path not in by_path:
        raise RuntimeError("Requested source is outside the frozen archive set")
    if gate.get("task_id") != TASK_ID or gate.get("overall_status") != "PASS":
        raise RuntimeError("Pre-deletion final gate is not PASS")
    if source_path not in gate.get("archive_deletion_eligible_paths", []):
        raise RuntimeError("Requested source is not eligible in the final gate")

    item = by_path[source_path]
    matches, observed = archive_item_matches(item)
    blockers = matching_processes()
    open_matches = open_fd_matches([source_path])
    canary_before = sha256_file(Path(CANARY_PATH))
    if not matches or blockers or open_matches or canary_before != CANARY_SHA256:
        raise RuntimeError("Archive source failed immediate pre-deletion validation")

    source = Path(source_path)
    free_before = shutil.disk_usage("/root/autodl-tmp").free
    if item["source_type"] == "file":
        source.unlink()
    else:
        shutil.rmtree(source)
    os.sync()
    free_after = shutil.disk_usage("/root/autodl-tmp").free
    released = free_after - free_before
    expected = item["expected_allocated_bytes"]
    release_error_ratio = abs(released - expected) / expected
    canary_after = sha256_file(Path(CANARY_PATH))
    passed = bool(
        not source.exists()
        and release_error_ratio <= 0.05
        and canary_after == CANARY_SHA256
    )
    result = {
        "schema_version": "canondressgs.subject00.archive_source_deletion_item.v1",
        "task_id": TASK_ID,
        "deleted_at_utc": utc_now(),
        "source_path": source_path,
        "source_type": item["source_type"],
        "pre_deletion_observation": observed,
        "free_bytes_before": free_before,
        "free_bytes_after": free_after,
        "released_bytes": released,
        "expected_released_bytes": expected,
        "release_error_ratio": release_error_ratio,
        "source_absent_after": not source.exists(),
        "short_canary_sha256_before": canary_before,
        "short_canary_sha256_after": canary_after,
        "overall_status": "PASS" if passed else "PAUSED_RELEASE_DELTA_MISMATCH",
    }
    write_json_atomic(output, result)
    if not passed:
        raise RuntimeError("Archive deletion paused after release-delta audit")


def duplicate_matches(item: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    source = Path(item["source_path"])
    retained = Path(item["retained_duplicate_path"])
    observed = {
        "source_path": str(source),
        "retained_duplicate_path": str(retained),
        "source_exists": source.is_file(),
        "retained_exists": retained.is_file(),
    }
    if source.is_file():
        observed["source_bytes"] = source.stat().st_size
        observed["source_sha256"] = sha256_file(source)
    if retained.is_file():
        observed["retained_bytes"] = retained.stat().st_size
        observed["retained_sha256"] = sha256_file(retained)
    matches = bool(
        observed["source_exists"]
        and observed["retained_exists"]
        and observed.get("source_bytes") == item["expected_bytes"]
        and observed.get("source_sha256") == item["expected_sha256"]
        and observed.get("retained_sha256") == item["retained_duplicate_sha256"]
        and observed.get("source_sha256") == observed.get("retained_sha256")
        and not item["manifest_references"]
    )
    observed["matches_adjudication"] = matches
    return matches, observed


def delete_low_risk_checkpoints(
    execution_path: Path,
    gate_path: Path,
    inventory_path: Path,
    duplicate_output: Path,
    optimizer_output: Path,
) -> None:
    execution = load_json(execution_path)
    gate = load_json(gate_path)
    inventory = load_json(inventory_path)
    inventory_by_path = {row["path"]: row for row in inventory["checkpoints"]}
    if gate.get("task_id") != TASK_ID or gate.get("overall_status") != "PASS":
        raise RuntimeError("Pre-deletion final gate is not PASS")
    for item in execution["avatarrex_archive_set"] + execution["checkpoint_archive_set"]:
        if Path(item["source_path"]).exists():
            raise RuntimeError("Archive-source deletion stage is not complete")
    if matching_processes() or any(Path(path).exists() for path in FORMAL_PATHS):
        raise RuntimeError("Blocking process or Formal path detected")

    duplicate_report = {
        "schema_version": "canondressgs.subject00.duplicate_deletion_execution.v1",
        "task_id": TASK_ID,
        "started_at_utc": utc_now(),
        "items": [],
        "overall_status": "IN_PROGRESS",
    }
    write_json_atomic(duplicate_output, duplicate_report)
    for item in execution["duplicate_delete_set"]:
        source_path = item["source_path"]
        inventory_row = inventory_by_path.get(source_path)
        matches, observed = duplicate_matches(item)
        if (
            source_path not in gate["duplicate_deletion_eligible_paths"]
            or not inventory_row
            or inventory_row["classification"]
            != "DUPLICATE_SAFE_DELETE_CANDIDATE"
            or inventory_row.get("sealed_attempt")
            or not matches
            or open_fd_matches([source_path])
        ):
            duplicate_report["overall_status"] = "BLOCKED"
            write_json_atomic(duplicate_output, duplicate_report)
            raise RuntimeError(f"Duplicate failed validation: {source_path}")
        Path(source_path).unlink()
        row = {
            **observed,
            "deleted": not Path(source_path).exists(),
            "retained_exists_after": Path(item["retained_duplicate_path"]).is_file(),
        }
        duplicate_report["items"].append(row)
        write_json_atomic(duplicate_output, duplicate_report)
    duplicate_report["completed_at_utc"] = utc_now()
    duplicate_report["deleted_count"] = len(duplicate_report["items"])
    duplicate_report["deleted_bytes"] = sum(
        item["expected_bytes"] for item in execution["duplicate_delete_set"]
    )
    duplicate_report["overall_status"] = (
        "PASS" if duplicate_report["deleted_count"] == 70 else "FAIL"
    )
    write_json_atomic(duplicate_output, duplicate_report)

    optimizer_report = {
        "schema_version": "canondressgs.subject00.optimizer_deletion_execution.v1",
        "task_id": TASK_ID,
        "started_at_utc": utc_now(),
        "items": [],
        "overall_status": "IN_PROGRESS",
    }
    write_json_atomic(optimizer_output, optimizer_report)
    for item in execution["optimizer_delete_set"]:
        path = Path(item["source_path"])
        inventory_row = inventory_by_path.get(item["source_path"])
        observed = {
            "source_path": item["source_path"],
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256_file(path) if path.is_file() else None,
        }
        valid = bool(
            item["source_path"] in gate["optimizer_deletion_eligible_paths"]
            and inventory_row
            and inventory_row["classification"]
            == "TEMPORARY_OPTIMIZER_STATE_CANDIDATE"
            and not inventory_row.get("sealed_attempt")
            and observed["exists"]
            and observed["bytes"] == item["expected_bytes"]
            and observed["sha256"] == item["expected_sha256"]
            and not item["manifest_references"]
            and not open_fd_matches([item["source_path"]])
        )
        if not valid:
            optimizer_report["overall_status"] = "BLOCKED"
            write_json_atomic(optimizer_output, optimizer_report)
            raise RuntimeError("Optimizer-only candidate failed validation")
        path.unlink()
        optimizer_report["items"].append(
            {**observed, "deleted": not path.exists()}
        )
        write_json_atomic(optimizer_output, optimizer_report)
    optimizer_report["completed_at_utc"] = utc_now()
    optimizer_report["deleted_count"] = len(optimizer_report["items"])
    optimizer_report["deleted_bytes"] = sum(
        item["expected_bytes"] for item in execution["optimizer_delete_set"]
    )
    optimizer_report["short_canary_sha256_after"] = sha256_file(Path(CANARY_PATH))
    optimizer_report["overall_status"] = (
        "PASS"
        if optimizer_report["deleted_count"] == 1
        and optimizer_report["short_canary_sha256_after"] == CANARY_SHA256
        else "FAIL"
    )
    write_json_atomic(optimizer_output, optimizer_report)


def audit_protection(
    inventory_path: Path,
    protection_registry_path: Path,
    storage_inventory_path: Path,
    output: Path,
) -> None:
    inventory = load_json(inventory_path)
    registry = load_json(protection_registry_path)
    storage_inventory = load_json(storage_inventory_path)
    protected_classes = {
        "CRITICAL_ACTIVE_KEEP",
        "SEALED_PROVENANCE_KEEP",
        "UNIQUE_FINAL_KEEP",
        "UNKNOWN_KEEP",
    }
    deleted_classes = {
        "ARCHIVE_THEN_DELETE_CANDIDATE",
        "DUPLICATE_SAFE_DELETE_CANDIDATE",
        "TEMPORARY_OPTIMIZER_STATE_CANDIDATE",
    }
    protected_results = []
    deleted_results = []
    for row in inventory["checkpoints"]:
        path = Path(row["path"])
        if row["classification"] in protected_classes:
            exists = path.is_file()
            observed_sha = sha256_file(path) if exists else None
            protected_results.append(
                {
                    "path": row["path"],
                    "classification": row["classification"],
                    "exists": exists,
                    "bytes_match": exists and path.stat().st_size == row["bytes"],
                    "sha256_match": observed_sha == row["sha256"],
                    "paper_or_figure": row.get("reference_flags", {}).get(
                        "paper_or_figure", False
                    ),
                    "figure_bank": row.get("reference_flags", {}).get(
                        "figure_bank", False
                    ),
                }
            )
        elif row["classification"] in deleted_classes:
            deleted_results.append(
                {
                    "path": row["path"],
                    "classification": row["classification"],
                    "absent": not path.exists(),
                }
            )
    protected_failures = [
        item
        for item in protected_results
        if not item["exists"] or not item["bytes_match"] or not item["sha256_match"]
    ]
    deletion_failures = [item for item in deleted_results if not item["absent"]]
    real_scopes = [
        item
        for item in registry["protected_scopes"]
        if str(item["path"]).startswith("/")
    ]
    scope_results = [
        {"path": item["path"], "exists": Path(item["path"]).exists()}
        for item in real_scopes
    ]
    protected_dataset_results = []
    for item in storage_inventory["large_candidates"]:
        if item["classification"] not in {"A", "B"}:
            continue
        path = Path(item["path"])
        observed = (
            tree_stats(path, "directory", include_directory_blocks=True)
            if path.is_dir()
            else {}
        )
        protected_dataset_results.append(
            {
                "path": item["path"],
                "classification": item["classification"],
                "exists": path.is_dir(),
                "expected_file_count": item["file_count"],
                "observed_file_count": observed.get("file_count"),
                "expected_logical_bytes": item["apparent_file_bytes"],
                "observed_logical_bytes": observed.get("logical_bytes"),
                "expected_allocated_bytes": item["allocated_bytes"],
                "observed_allocated_bytes": observed.get("allocated_bytes"),
                "matches": bool(
                    path.is_dir()
                    and observed.get("file_count") == item["file_count"]
                    and observed.get("logical_bytes") == item["apparent_file_bytes"]
                    and observed.get("allocated_bytes") == item["allocated_bytes"]
                ),
            }
        )
    class_summary: dict[str, int] = {}
    for item in protected_results:
        classification = item["classification"]
        class_summary[classification] = class_summary.get(classification, 0) + 1
    passed = bool(
        not protected_failures
        and not deletion_failures
        and all(item["exists"] for item in scope_results)
        and all(item["matches"] for item in protected_dataset_results)
        and sha256_file(Path(CANARY_PATH)) == CANARY_SHA256
    )
    result = {
        "schema_version": "canondressgs.subject00.post_reclamation_protection_audit.v1",
        "task_id": TASK_ID,
        "audited_at_utc": utc_now(),
        "protected_class_counts": class_summary,
        "protected_checkpoint_count": len(protected_results),
        "protected_checkpoint_failures": protected_failures[:20],
        "authorized_deleted_checkpoint_count": len(deleted_results),
        "authorized_deletion_failures": deletion_failures[:20],
        "paper_or_figure_protected_count": sum(
            bool(item["paper_or_figure"]) for item in protected_results
        ),
        "figure_bank_protected_count": sum(
            bool(item["figure_bank"]) for item in protected_results
        ),
        "protected_scope_results": scope_results,
        "protected_dataset_results": protected_dataset_results,
        "short_canary_path": CANARY_PATH,
        "short_canary_sha256": sha256_file(Path(CANARY_PATH)),
        "overall_status": "PASS" if passed else "FAIL",
    }
    write_json_atomic(output, result)
    if not passed:
        raise RuntimeError("Post-reclamation protection audit failed")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--repo-root", required=True)
    freeze_parser.add_argument("--output", required=True)
    freeze_parser.add_argument("--cloud-free-bytes", type=int, required=True)
    freeze_parser.add_argument("--windows-free-bytes", type=int, required=True)

    live_parser = subparsers.add_parser("validate-live")
    live_parser.add_argument("--execution-set", required=True)
    live_parser.add_argument("--output", required=True)

    merge_parser = subparsers.add_parser("merge-validation")
    merge_parser.add_argument("--execution-set", required=True)
    merge_parser.add_argument("--validation", required=True)

    manifest_parser = subparsers.add_parser("build-manifest")
    manifest_parser.add_argument("--source", required=True)
    manifest_parser.add_argument("--source-type", choices=("file", "directory"), required=True)
    manifest_parser.add_argument("--output", required=True)

    verify_parser = subparsers.add_parser("verify-manifest")
    verify_parser.add_argument("--manifest", required=True)
    verify_parser.add_argument("--destination", required=True)
    verify_parser.add_argument("--output", required=True)

    progress_init_parser = subparsers.add_parser("init-progress")
    progress_init_parser.add_argument("--execution-set", required=True)
    progress_init_parser.add_argument("--output", required=True)

    progress_event_parser = subparsers.add_parser("progress-event")
    progress_event_parser.add_argument("--progress", required=True)
    progress_event_parser.add_argument("--phase", required=True)
    progress_event_parser.add_argument("--status", required=True)
    progress_event_parser.add_argument("--note", required=True)

    gate_parser = subparsers.add_parser("build-pre-deletion-gate")
    gate_parser.add_argument("--execution-set", required=True)
    gate_parser.add_argument("--avatar-verification", required=True)
    gate_parser.add_argument("--checkpoint-verification", required=True)
    gate_parser.add_argument("--inventory", required=True)
    gate_parser.add_argument("--output", required=True)

    archive_delete_parser = subparsers.add_parser("delete-archive-item")
    archive_delete_parser.add_argument("--execution-set", required=True)
    archive_delete_parser.add_argument("--gate", required=True)
    archive_delete_parser.add_argument("--source", required=True)
    archive_delete_parser.add_argument("--output", required=True)

    low_risk_delete_parser = subparsers.add_parser(
        "delete-low-risk-checkpoints"
    )
    low_risk_delete_parser.add_argument("--execution-set", required=True)
    low_risk_delete_parser.add_argument("--gate", required=True)
    low_risk_delete_parser.add_argument("--inventory", required=True)
    low_risk_delete_parser.add_argument("--duplicate-output", required=True)
    low_risk_delete_parser.add_argument("--optimizer-output", required=True)

    protection_parser = subparsers.add_parser("audit-protection")
    protection_parser.add_argument("--inventory", required=True)
    protection_parser.add_argument("--protection-registry", required=True)
    protection_parser.add_argument("--storage-inventory", required=True)
    protection_parser.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.command == "freeze":
        freeze(
            Path(args.repo_root).resolve(),
            Path(args.output).resolve(),
            args.cloud_free_bytes,
            args.windows_free_bytes,
        )
    elif args.command == "validate-live":
        validate_live(Path(args.execution_set), Path(args.output))
    elif args.command == "merge-validation":
        merge_validation(Path(args.execution_set), Path(args.validation))
    elif args.command == "build-manifest":
        build_manifest(Path(args.source), args.source_type, Path(args.output))
    elif args.command == "verify-manifest":
        verify_manifest(
            Path(args.manifest), Path(args.destination), Path(args.output)
        )
    elif args.command == "init-progress":
        init_progress(Path(args.execution_set), Path(args.output))
    elif args.command == "progress-event":
        append_progress_event(
            Path(args.progress), args.phase, args.status, args.note
        )
    elif args.command == "build-pre-deletion-gate":
        build_pre_deletion_gate(
            Path(args.execution_set),
            Path(args.avatar_verification),
            Path(args.checkpoint_verification),
            Path(args.inventory),
            Path(args.output),
        )
    elif args.command == "delete-archive-item":
        delete_archive_item(
            Path(args.execution_set), Path(args.gate), args.source, Path(args.output)
        )
    elif args.command == "delete-low-risk-checkpoints":
        delete_low_risk_checkpoints(
            Path(args.execution_set),
            Path(args.gate),
            Path(args.inventory),
            Path(args.duplicate_output),
            Path(args.optimizer_output),
        )
    elif args.command == "audit-protection":
        audit_protection(
            Path(args.inventory),
            Path(args.protection_registry),
            Path(args.storage_inventory),
            Path(args.output),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
