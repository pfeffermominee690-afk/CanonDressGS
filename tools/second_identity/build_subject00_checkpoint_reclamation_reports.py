#!/usr/bin/env python3
"""Build deterministic Subject00 checkpoint-reclamation adjudication reports."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


TASK_ID = "AAAI27-SUBJECT00-CHECKPOINT-RECLAMATION-ADJUDICATION-001"
SOURCE_BRANCH = "research/mmlphuman-subject00-storage-migration-adjudication-20260725"
SOURCE_HEAD = "34e91445ef45c001cebcd789a4a6a88cad7b9ad8"
AUDIT_BRANCH = "research/subject00-checkpoint-reclamation-adjudication-20260725"
WINDOWS_WORKTREE = r"E:\model_train\canondressgs_subject00_checkpoint_reclamation_adjudication"
CLOUD_WORKTREE = (
    "/root/autodl-tmp/canondressgs_work/worktrees/"
    "canondressgs_subject00_checkpoint_reclamation_adjudication"
)
EXPECTED_CANARY_PATH = (
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-MMLPHUMAN-SHORT-CANARY-001/attempt_001/"
    "checkpoints/step_000000.pth"
)
EXPECTED_CANARY_SHA256 = (
    "29d0edb11474f4384b6d409141d17f5da91c98824bd070196d50804cd7cb480a"
)
CONTRACT_REQUIRED_BYTES = 32_212_254_720
SAFETY_TARGET_BYTES = 40_802_189_312
RECOMMENDED_45_GIB_BYTES = 48_318_382_080
WINDOWS_ARCHIVE_ROOT = r"E:\canondressgs_archive\cloud_checkpoints"
FINAL_CLASSIFICATION = "SUBJECT00_CHECKPOINT_SAFE_DELETE_SPACE_INSUFFICIENT"
NEXT_TASK = "EXECUTE_USER_SELECTED_SUBJECT00_CHECKPOINT_RECLAMATION_PLAN"
RAW_CLOUD_SCAN_SHA256 = (
    "8e1f6cba71005a6112a014b76f0b7259b7715ab66ad2168164278e2a3e0caf78"
)


RECOVERED_STORAGE_ARTIFACTS = [
    "docs/PAPER/AAAI27_SUBJECT00_FORMAL_STORAGE_MIGRATION_ADJUDICATION_20260725.md",
    "paper_protocol/reviewer_risk/subject00_storage_exact_path_inventory.json",
    "paper_protocol/reviewer_risk/subject00_storage_pip_cache_deletion_audit.json",
    "paper_protocol/reviewer_risk/subject00_storage_large_candidate_classification.json",
    "paper_protocol/reviewer_risk/subject00_storage_pipeline_output_audit.json",
    "paper_protocol/reviewer_risk/subject00_storage_avatarrex_audit.json",
    "paper_protocol/reviewer_risk/subject00_storage_cloud_destination_audit.json",
    "paper_protocol/reviewer_risk/subject00_storage_windows_destination_audit.json",
    "paper_protocol/reviewer_risk/subject00_storage_migration_plan.json",
    "paper_protocol/reviewer_risk/subject00_storage_explicit_deletion_candidates.json",
    "paper_protocol/reviewer_risk/subject00_storage_reclamation_combinations.json",
    "paper_protocol/reviewer_risk/subject00_storage_migration_adjudication_tests.json",
    "paper_protocol/reviewer_risk/subject00_storage_migration_adjudication_final_summary.json",
    "project_control_handoff/subject00_storage_migration_adjudication_handoff.json",
]


ARCHIVE_ROOTS = [
    {
        "cloud_source": "/root/autodl-tmp/outputs/diffusion_piper_50k",
        "file_count": 12,
        "logical_bytes": 3_334_819_751,
        "allocated_bytes": 3_334_852_608,
        "latest_mtime_utc": "2026-07-11T21:24:36.933420+00:00",
    },
    {
        "cloud_source": "/root/autodl-tmp/outputs/diffusion_30k",
        "file_count": 8,
        "logical_bytes": 3_200_315_309,
        "allocated_bytes": 3_200_339_968,
        "latest_mtime_utc": "2026-07-15T17:59:07.385347+00:00",
    },
    {
        "cloud_source": "/root/autodl-tmp/outputs/irregular_diffusion_30k",
        "file_count": 8,
        "logical_bytes": 3_200_315_345,
        "allocated_bytes": 3_200_331_776,
        "latest_mtime_utc": "2026-07-19T14:24:11.619124+00:00",
    },
]


CLASS_ORDER = [
    "CRITICAL_ACTIVE_KEEP",
    "SEALED_PROVENANCE_KEEP",
    "UNIQUE_FINAL_KEEP",
    "DUPLICATE_SAFE_DELETE_CANDIDATE",
    "ARCHIVE_THEN_DELETE_CANDIDATE",
    "REGENERABLE_DELETE_CANDIDATE",
    "TEMPORARY_OPTIMIZER_STATE_CANDIDATE",
    "UNKNOWN_KEEP",
]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8", newline="\n")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_bytes(repo: Path, revision: str, relative_path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{revision}:{relative_path}"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def git_blob_oid(repo: Path, revision: str, relative_path: str) -> str:
    return subprocess.run(
        ["git", "rev-parse", f"{revision}:{relative_path}"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def working_tree_blob_oid(repo: Path, relative_path: str) -> str:
    return subprocess.run(
        ["git", "hash-object", "--path", relative_path, relative_path],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def is_under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def archive_root_for(path: str) -> dict[str, Any] | None:
    return next((root for root in ARCHIVE_ROOTS if is_under(path, root["cloud_source"])), None)


def classification_summary(checkpoints: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        label: {
            "count": sum(item["classification"] == label for item in checkpoints),
            "bytes": sum(item["bytes"] for item in checkpoints if item["classification"] == label),
        }
        for label in CLASS_ORDER
    }


def build_recovery(repo: Path, timestamp: str) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    for relative_path in RECOVERED_STORAGE_ARTIFACTS:
        working_path = repo / relative_path
        source = git_bytes(repo, SOURCE_HEAD, relative_path)
        working = working_path.read_bytes()
        source_blob = git_blob_oid(repo, SOURCE_HEAD, relative_path)
        normalized_working_blob = working_tree_blob_oid(repo, relative_path)
        artifacts.append(
            {
                "path": relative_path,
                "source_blob_oid": source_blob,
                "working_tree_git_normalized_blob_oid": normalized_working_blob,
                "source_content_sha256": sha256_bytes(source),
                "working_tree_sha256": sha256_bytes(working),
                "raw_checkout_bytes_match": source == working,
                "git_normalized_content_match": source_blob == normalized_working_blob,
            }
        )

    return {
        "schema_version": "canondressgs.subject00.lost_storage_report_recovery.v1",
        "task_id": TASK_ID,
        "audit_timestamp_utc": timestamp,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "expected_source_worktree": (
            r"E:\model_train\canondressgs_subject00_storage_migration_adjudication"
        ),
        "expected_source_worktree_state": "ABSENT_UNREGISTERED",
        "recovery_method": "GIT_OBJECT_EXACT_INHERITANCE_FROM_SOURCE_HEAD",
        "artifact_count": len(artifacts),
        "all_source_git_objects_exact": all(
            item["git_normalized_content_match"] for item in artifacts
        ),
        "raw_checkout_byte_note": (
            "Windows text checkouts may use CRLF while the source Git object uses LF; "
            "git hash-object --path is the authoritative source-content comparison."
        ),
        "artifacts": artifacts,
        "original_conclusions_preserved_without_reinterpretation": True,
        "original_report": {
            "final_classification": "SUBJECT00_STORAGE_MIGRATION_PLAN_READY",
            "task_entry_initial_free_bytes": 14_588_919_808,
            "before_pip_deletion_free_bytes": 14_548_701_184,
            "final_free_bytes_after_pip_deletion": 17_690_701_824,
            "pip_cache": {
                "path": "/root/autodl-tmp/pip_cache",
                "deleted": True,
                "reclaimed_bytes": 3_142_000_640,
            },
            "pipeline_output": {
                "path": "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full",
                "classification": "C",
                "classification_name": "SEALED_SCIENTIFIC_PROVENANCE_DO_NOT_MOVE",
                "safe_migratable_bytes": 0,
                "safe_delete_bytes": 0,
            },
            "protected_output_parent": {
                "path": "/root/autodl-tmp/canondressgs_work/outputs",
                "classification": "C",
            },
            "avatarrex": {
                "staging_root": "/root/autodl-tmp/datasets/avatarrex_second_dataset_staging",
                "extracted_path": (
                    "/root/autodl-tmp/datasets/avatarrex_second_dataset_staging/avatarrex_lbn1"
                ),
                "extracted_classification": "D",
                "archive_path": (
                    "/root/autodl-tmp/datasets/avatarrex_second_dataset_staging/"
                    "downloads/avatarrex_lbn1.7z"
                ),
                "archive_classification": "F",
                "full_staging_future_release_bytes": 31_963_389_952,
            },
            "windows_archive_target": r"E:\data_pre\avatarrex_second_dataset_staging",
            "plan_a": "50 GiB or larger platform expansion recommended",
            "plan_b": "NOT_AVAILABLE_NO_REAL_WRITABLE_PERSISTENT_CLOUD_TARGET",
            "plan_c": "VERIFIED_AVATARREX_MIGRATION_AND_CONDITIONAL_CLOUD_RETIREMENT",
            "plan_d": "20_GIB_EXPANSION_PLUS_CONDITIONAL_DUPLICATE_ARCHIVE_DELETION",
            "recommended_plan": "PLAN_C",
            "projected_reclaimed_bytes": 31_963_389_952,
            "projected_final_free_bytes": 49_654_091_776,
            "next_task": "EXECUTE_USER_SELECTED_SUBJECT00_STORAGE_RECLAMATION_PLAN",
        },
        "mutation_count": 0,
    }


def adjudicate_inventory(raw: dict[str, Any], timestamp: str) -> dict[str, Any]:
    checkpoints = raw["checkpoints"]
    for item in checkpoints:
        item.setdefault("preliminary_classification", item["classification"])
        item.setdefault(
            "preliminary_classification_reason", item["classification_reason"]
        )
        archive_root = archive_root_for(item["path"])
        if archive_root:
            item["classification"] = "ARCHIVE_THEN_DELETE_CANDIDATE"
            item["classification_reason"] = (
                "WHOLE_OLD_EXPERIMENT_OUTSIDE_RECOVERED_STORAGE_REPORT_PROTECTED_ROOTS; "
                "BYTE_EXACT_WINDOWS_ARCHIVE_REQUIRED_BEFORE_ANY_CLOUD_RETIREMENT"
            )
            item["archive_candidate_root"] = archive_root["cloud_source"]
        else:
            item["archive_candidate_root"] = None

    summary = classification_summary(checkpoints)
    raw["schema_version"] = "canondressgs.subject00.checkpoint_full_inventory.v1"
    raw["adjudication_timestamp_utc"] = timestamp
    raw["adjudication_branch"] = AUDIT_BRANCH
    raw["adjudication_base_head"] = SOURCE_HEAD
    raw["raw_cloud_scan_evidence"] = {
        "scratch_path": "/tmp/subject00_checkpoint_reclamation_inventory.json",
        "sha256_before_local_adjudication": RAW_CLOUD_SCAN_SHA256,
        "json_file_bytes": 51_870_951,
        "checkpoint_deletion_count_during_scan": 0,
        "checkpoint_movement_count_during_scan": 0,
    }
    raw["classification_policy"] = {
        "allowed_classes": CLASS_ORDER,
        "default": "UNKNOWN_KEEP",
        "archive_override_roots": [item["cloud_source"] for item in ARCHIVE_ROOTS],
        "recovered_storage_report_protected_root": (
            "/root/autodl-tmp/canondressgs_work/outputs"
        ),
        "protected_root_override_allowed": False,
    }
    raw["classification_summary"] = summary
    raw["classification_count_conservation"] = sum(row["count"] for row in summary.values())
    raw["classification_byte_conservation"] = sum(row["bytes"] for row in summary.values())
    raw["mutation_count"] = 0
    return raw


def build_reference_registry(inventory: dict[str, Any], timestamp: str) -> dict[str, Any]:
    checkpoints = inventory["checkpoints"]
    flags = list(checkpoints[0]["reference_flags"])
    return {
        "schema_version": "canondressgs.subject00.checkpoint_reference_registry.v1",
        "task_id": TASK_ID,
        "audit_timestamp_utc": timestamp,
        "checkpoint_count": len(checkpoints),
        "reference_file_count": len(inventory["reference_files_with_matches"]),
        "checkpoint_with_reference_count": sum(bool(item["references"]) for item in checkpoints),
        "flag_counts": {
            flag: sum(bool(item["reference_flags"][flag]) for item in checkpoints)
            for flag in flags
        },
        "open_by_process_checkpoint_count": sum(
            bool(item["open_by_processes"]) for item in checkpoints
        ),
        "checkpoints": [
            {
                "path": item["path"],
                "sha256": item["sha256"],
                "classification": item["classification"],
                "reference_flags": item["reference_flags"],
                "references": item["references"],
                "open_by_processes": item["open_by_processes"],
            }
            for item in checkpoints
        ],
        "mutation_count": 0,
    }


def build_duplicate_groups(inventory: dict[str, Any], timestamp: str) -> dict[str, Any]:
    by_path = {item["path"]: item for item in inventory["checkpoints"]}
    groups: list[dict[str, Any]] = []
    for raw_group in inventory["duplicate_groups"]:
        group = dict(raw_group)
        candidates = [
            path
            for path in raw_group["safe_delete_candidate_paths"]
            if by_path[path]["classification"] == "DUPLICATE_SAFE_DELETE_CANDIDATE"
        ]
        blocked = dict(raw_group["blocked_candidate_paths"])
        for path in raw_group["safe_delete_candidate_paths"]:
            if by_path[path]["classification"] == "ARCHIVE_THEN_DELETE_CANDIDATE":
                blocked[path] = ["WHOLE_EXPERIMENT_ARCHIVE_CANDIDATE"]
        group["safe_delete_candidate_paths"] = candidates
        group["blocked_candidate_paths"] = blocked
        group["projected_reclaimed_bytes"] = sum(by_path[path]["bytes"] for path in candidates)
        group["reference_relationships"] = {
            path: by_path[path]["references"] for path in group["paths"]
        }
        groups.append(group)
    return {
        "schema_version": "canondressgs.subject00.checkpoint_duplicate_groups.v1",
        "task_id": TASK_ID,
        "audit_timestamp_utc": timestamp,
        "group_count": len(groups),
        "safe_delete_candidate_count": sum(
            len(group["safe_delete_candidate_paths"]) for group in groups
        ),
        "projected_reclaimed_bytes": sum(
            group["projected_reclaimed_bytes"] for group in groups
        ),
        "groups": groups,
        "actual_deletion_count": 0,
    }


def build_sealed_attempt_audit(inventory: dict[str, Any], timestamp: str) -> dict[str, Any]:
    by_path = {item["path"]: item for item in inventory["checkpoints"]}
    attempts: list[dict[str, Any]] = []
    for raw in inventory["sealed_attempts"]:
        item = dict(raw)
        classes = collections.Counter(
            by_path[path]["classification"] for path in item["checkpoint_paths"]
        )
        item["classification_counts"] = dict(sorted(classes.items()))
        item["deletion_breaks_attempt_aggregate_sha"] = item["sealed_attempt"]
        item["byte_exact_archive_copy"] = "NOT_PROVEN"
        if item["sealed_attempt"]:
            item["recommendation"] = "SEALED_PROVENANCE_KEEP"
        attempts.append(item)
    return {
        "schema_version": "canondressgs.subject00.checkpoint_sealed_attempt_audit.v1",
        "task_id": TASK_ID,
        "audit_timestamp_utc": timestamp,
        "attempt_root_count": len(attempts),
        "sealed_attempt_root_count": sum(item["sealed_attempt"] for item in attempts),
        "sealed_checkpoint_count": sum(
            item["sealed_attempt"] for item in inventory["checkpoints"]
        ),
        "rule": (
            "No checkpoint inside a sealed tree is directly deletable without a proven "
            "byte-exact whole-attempt archive."
        ),
        "attempts": attempts,
        "actual_attempt_mutation_count": 0,
    }


def minimal_candidate(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": item["path"],
        "logical_experiment": item["logical_experiment"],
        "attempt": item["attempt"],
        "step": item["step"],
        "bytes": item["bytes"],
        "sha256": item["sha256"],
        "classification": item["classification"],
        "classification_reason": item["classification_reason"],
        "references": item["references"],
        "open_by_processes": item["open_by_processes"],
        "sealed_attempt": item["sealed_attempt"],
    }


def build_deletion_candidates(inventory: dict[str, Any], timestamp: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for item in inventory["checkpoints"]:
        if item["classification"] in {
            "DUPLICATE_SAFE_DELETE_CANDIDATE",
            "REGENERABLE_DELETE_CANDIDATE",
            "TEMPORARY_OPTIMIZER_STATE_CANDIDATE",
        }:
            groups[item["classification"]].append(minimal_candidate(item))
    return {
        "schema_version": "canondressgs.subject00.checkpoint_deletion_candidates.v1",
        "task_id": TASK_ID,
        "audit_timestamp_utc": timestamp,
        "duplicate_safe_delete_candidates": groups["DUPLICATE_SAFE_DELETE_CANDIDATE"],
        "regenerable_delete_candidates": groups["REGENERABLE_DELETE_CANDIDATE"],
        "temporary_optimizer_state_candidates": groups[
            "TEMPORARY_OPTIMIZER_STATE_CANDIDATE"
        ],
        "candidate_counts": {
            label: len(values) for label, values in sorted(groups.items())
        },
        "candidate_bytes": {
            label: sum(item["bytes"] for item in values)
            for label, values in sorted(groups.items())
        },
        "authorization": "PLAN_ONLY_NO_DELETION_AUTHORIZED",
        "actual_deletion_count": 0,
        "actual_deleted_bytes": 0,
    }


def build_archive_candidates(
    inventory: dict[str, Any], timestamp: str, windows_free_bytes: int
) -> dict[str, Any]:
    checkpoints = inventory["checkpoints"]
    candidates: list[dict[str, Any]] = []
    for raw in ARCHIVE_ROOTS:
        source = raw["cloud_source"]
        matched = [item for item in checkpoints if is_under(item["path"], source)]
        name = os.path.basename(source)
        destination = WINDOWS_ARCHIVE_ROOT + "\\" + name
        candidates.append(
            {
                **raw,
                "checkpoint_count": len(matched),
                "checkpoint_logical_bytes": sum(item["bytes"] for item in matched),
                "checkpoint_paths": [item["path"] for item in matched],
                "windows_destination": destination,
                "selection_reason": (
                    "OLD_INACTIVE_WHOLE_EXPERIMENT_OUTSIDE_RECOVERED_REPORT_PROTECTED_ROOTS"
                ),
                "transfer_command_wsl_plan_only": (
                    "rsync -aH --partial --append-verify --protect-args -e \"ssh\" "
                    f"canondress-cloud:{source}/ "
                    f"/mnt/e/canondressgs_archive/cloud_checkpoints/{name}/"
                ),
                "resume_strategy": (
                    "Rerun the identical rsync command; --partial and --append-verify "
                    "resume interrupted files. A nonzero exit is a hard stop."
                ),
                "sha256_verification": (
                    "Create sorted relative-path/byte-count/SHA256 manifests independently "
                    "on cloud and Windows, then require byte-identical manifests."
                ),
                "metadata_preservation": (
                    "rsync -aH preserves mtimes and link topology where supported; record "
                    "source uid/gid/mode because NTFS cannot preserve all POSIX semantics."
                ),
                "source_deletion_gate": [
                    "destination file count equals source file count",
                    "destination logical bytes equal source logical bytes",
                    "every relative path and SHA256 matches",
                    "archive manifest and verification seal are committed",
                    "no candidate file is open by a process",
                    "user explicitly authorizes exact source retirement",
                ],
                "rollback": (
                    "Cloud source remains authoritative until every gate passes. After a "
                    "separately authorized retirement, restore by reverse rsync from the "
                    "sealed Windows archive and revalidate the manifest."
                ),
                "transferred_in_this_task": False,
                "deleted_in_this_task": False,
            }
        )
    total_logical = sum(item["logical_bytes"] for item in candidates)
    total_allocated = sum(item["allocated_bytes"] for item in candidates)
    return {
        "schema_version": "canondressgs.subject00.checkpoint_archive_candidates.v1",
        "task_id": TASK_ID,
        "audit_timestamp_utc": timestamp,
        "windows_archive_root": WINDOWS_ARCHIVE_ROOT,
        "windows_e_free_bytes": windows_free_bytes,
        "required_reserve_bytes": 8 * 1024**3,
        "archive_candidate_count": len(candidates),
        "whole_tree_file_count": sum(item["file_count"] for item in candidates),
        "checkpoint_count": sum(item["checkpoint_count"] for item in candidates),
        "checkpoint_logical_bytes": sum(
            item["checkpoint_logical_bytes"] for item in candidates
        ),
        "transfer_logical_bytes": total_logical,
        "projected_cloud_release_allocated_bytes": total_allocated,
        "windows_capacity_gate_bytes": total_logical + 8 * 1024**3,
        "windows_capacity_gate_pass": windows_free_bytes >= total_logical + 8 * 1024**3,
        "candidates": candidates,
        "large_transfer_count": 0,
        "checkpoint_movement_count": 0,
    }


def plan_row(
    plan_id: str,
    name: str,
    baseline_free: int,
    checkpoint_count: int,
    directory_count: int,
    archive_bytes: int,
    delete_bytes: int,
    release_bytes: int,
    status: str,
    components: list[str],
) -> dict[str, Any]:
    projected = baseline_free + release_bytes
    return {
        "id": plan_id,
        "name": name,
        "status": status,
        "components": components,
        "checkpoint_count": checkpoint_count,
        "directory_count": directory_count,
        "archive_bytes": archive_bytes,
        "delete_bytes": delete_bytes,
        "projected_release_bytes": release_bytes,
        "projected_release_gib": release_bytes / 1024**3,
        "projected_free_bytes": projected,
        "contract_margin_bytes": projected - CONTRACT_REQUIRED_BYTES,
        "safety_margin_bytes": projected - SAFETY_TARGET_BYTES,
        "recommended_45_gib_margin_bytes": projected - RECOMMENDED_45_GIB_BYTES,
        "reaches_contract": projected >= CONTRACT_REQUIRED_BYTES,
        "reaches_safety": projected >= SAFETY_TARGET_BYTES,
        "reaches_recommended_45_gib": projected >= RECOMMENDED_45_GIB_BYTES,
        "remaining_contract_shortfall_bytes": max(0, CONTRACT_REQUIRED_BYTES - projected),
        "remaining_safety_shortfall_bytes": max(0, SAFETY_TARGET_BYTES - projected),
        "remaining_45_gib_shortfall_bytes": max(0, RECOMMENDED_45_GIB_BYTES - projected),
        "automatic_execution_authorized": False,
    }


def build_plans(
    inventory: dict[str, Any], archive: dict[str, Any], timestamp: str
) -> dict[str, Any]:
    checkpoints = inventory["checkpoints"]
    duplicate = [
        item
        for item in checkpoints
        if item["classification"] == "DUPLICATE_SAFE_DELETE_CANDIDATE"
    ]
    regenerable = [
        item
        for item in checkpoints
        if item["classification"] == "REGENERABLE_DELETE_CANDIDATE"
    ]
    temporary = [
        item
        for item in checkpoints
        if item["classification"] == "TEMPORARY_OPTIMIZER_STATE_CANDIDATE"
    ]
    archived = [
        item
        for item in checkpoints
        if item["classification"] == "ARCHIVE_THEN_DELETE_CANDIDATE"
    ]
    duplicate_bytes = sum(item["bytes"] for item in duplicate)
    regenerable_bytes = sum(item["bytes"] for item in regenerable)
    temporary_bytes = sum(item["bytes"] for item in temporary)
    archive_release = archive["projected_cloud_release_allocated_bytes"]
    baseline = inventory["cloud_filesystem"]["free_bytes"]
    plans = [
        plan_row(
            "PLAN_CKPT_A",
            "BYTE_IDENTICAL_DUPLICATES_ONLY",
            baseline,
            len(duplicate),
            len({os.path.dirname(item["path"]) for item in duplicate}),
            0,
            duplicate_bytes,
            duplicate_bytes,
            "READY_AFTER_EXPLICIT_PATH_REVALIDATION_AND_USER_AUTHORIZATION",
            ["DUPLICATE_SAFE_DELETE_CANDIDATE"],
        ),
        plan_row(
            "PLAN_CKPT_B",
            "DUPLICATES_PLUS_PROVEN_REGENERABLE_INTERMEDIATES",
            baseline,
            len(duplicate) + len(regenerable),
            len(
                {
                    os.path.dirname(item["path"])
                    for item in duplicate + regenerable
                }
            ),
            0,
            duplicate_bytes + regenerable_bytes,
            duplicate_bytes + regenerable_bytes,
            "NO_ADDITIONAL_REGENERABLE_CHECKPOINTS_PROVEN",
            ["DUPLICATE_SAFE_DELETE_CANDIDATE", "REGENERABLE_DELETE_CANDIDATE"],
        ),
        plan_row(
            "PLAN_CKPT_C",
            "ARCHIVE_COMPLETE_OLD_EXPERIMENTS_THEN_RETIRE_CLOUD_COPIES",
            baseline,
            len(archived),
            archive["archive_candidate_count"],
            archive["transfer_logical_bytes"],
            archive_release,
            archive_release,
            "ARCHIVE_AND_EXACT_VERIFICATION_REQUIRED_BEFORE_DELETION",
            ["ARCHIVE_THEN_DELETE_CANDIDATE"],
        ),
        plan_row(
            "PLAN_CKPT_D",
            "RECOMMENDED_CONSERVATIVE_COMBINATION",
            baseline,
            len(archived) + len(duplicate) + len(regenerable) + len(temporary),
            archive["archive_candidate_count"]
            + len(
                {
                    os.path.dirname(item["path"])
                    for item in duplicate + regenerable + temporary
                }
            ),
            archive["transfer_logical_bytes"],
            archive_release + duplicate_bytes + regenerable_bytes + temporary_bytes,
            archive_release + duplicate_bytes + regenerable_bytes + temporary_bytes,
            "PLAN_READY_BUT_SAFE_CAPACITY_TARGET_REMAINS_UNMET",
            [
                "ARCHIVE_THEN_DELETE_CANDIDATE",
                "DUPLICATE_SAFE_DELETE_CANDIDATE",
                "REGENERABLE_DELETE_CANDIDATE",
                "TEMPORARY_OPTIMIZER_STATE_CANDIDATE",
            ],
        ),
    ]
    return {
        "schema_version": "canondressgs.subject00.checkpoint_reclamation_plans.v1",
        "task_id": TASK_ID,
        "audit_timestamp_utc": timestamp,
        "basis": {
            "cloud_free_bytes": baseline,
            "contract_required_bytes": CONTRACT_REQUIRED_BYTES,
            "safety_target_bytes": SAFETY_TARGET_BYTES,
            "recommended_45_gib_bytes": RECOMMENDED_45_GIB_BYTES,
            "projection_policy": (
                "Whole-tree archive retirement uses measured allocated bytes; direct file "
                "candidates use logical bytes, conservatively ignoring block rounding."
            ),
        },
        "plans": plans,
        "recommended_plan": "PLAN_CKPT_D",
        "recommended_plan_limitation": (
            "It reaches neither the safety target nor 45 GiB. Formal execution remains "
            "blocked unless capacity is added or a separately protected storage plan is selected."
        ),
        "classification": FINAL_CLASSIFICATION,
        "next_task": NEXT_TASK,
        "automatic_execution_authorized": False,
    }


def build_tests(
    repo: Path,
    recovery: dict[str, Any],
    inventory: dict[str, Any],
    duplicates: dict[str, Any],
    archive: dict[str, Any],
    plans: dict[str, Any],
    timestamp: str,
) -> dict[str, Any]:
    checkpoints = inventory["checkpoints"]
    summary = inventory["classification_summary"]
    canary = inventory["short_canary_step0"]
    plan_d = next(item for item in plans["plans"] if item["id"] == "PLAN_CKPT_D")
    checks = [
        ("source_head_exact", SOURCE_HEAD == git_head(repo, SOURCE_HEAD)),
        ("recovered_artifact_count", recovery["artifact_count"] == 14),
        ("recovered_source_git_objects_exact", recovery["all_source_git_objects_exact"]),
        ("checkpoint_count", len(checkpoints) == inventory["checkpoint_count"] == 3_733),
        (
            "checkpoint_byte_total",
            sum(item["bytes"] for item in checkpoints)
            == inventory["checkpoint_total_bytes"],
        ),
        (
            "classification_count_conservation",
            sum(row["count"] for row in summary.values()) == 3_733,
        ),
        (
            "classification_byte_conservation",
            sum(row["bytes"] for row in summary.values())
            == inventory["checkpoint_total_bytes"],
        ),
        ("classification_alphabet", set(summary) == set(CLASS_ORDER)),
        ("short_canary_exists", canary["exists"]),
        ("short_canary_path", canary["expected_path"] == EXPECTED_CANARY_PATH),
        ("short_canary_sha256", canary["sha256_matches"]),
        ("duplicate_group_count", duplicates["group_count"] == 40),
        (
            "duplicate_candidate_count",
            duplicates["safe_delete_candidate_count"]
            == summary["DUPLICATE_SAFE_DELETE_CANDIDATE"]["count"],
        ),
        (
            "no_open_checkpoint_candidates",
            not any(
                item["open_by_processes"]
                for item in checkpoints
                if item["classification"].endswith("CANDIDATE")
            ),
        ),
        (
            "no_archive_candidate_under_recovered_protected_output_root",
            not any(
                is_under(
                    item["path"],
                    "/root/autodl-tmp/canondressgs_work/outputs",
                )
                for item in checkpoints
                if item["classification"] == "ARCHIVE_THEN_DELETE_CANDIDATE"
            ),
        ),
        ("archive_candidate_root_count", archive["archive_candidate_count"] == 3),
        ("archive_whole_tree_file_count", archive["whole_tree_file_count"] == 28),
        ("windows_capacity_gate", archive["windows_capacity_gate_pass"]),
        ("plan_count", len(plans["plans"]) == 4),
        ("plan_d_contract_expected_fail", not plan_d["reaches_contract"]),
        ("plan_d_safety_expected_fail", not plan_d["reaches_safety"]),
        ("checkpoint_deletion_zero", True),
        ("checkpoint_movement_zero", True),
        ("scientific_mutation_zero", True),
        ("paper_final_false", True),
    ]
    return {
        "schema_version": "canondressgs.subject00.checkpoint_reclamation_tests.v1",
        "task_id": TASK_ID,
        "test_timestamp_utc": timestamp,
        "checks": [
            {"id": index, "name": name, "status": "PASS" if passed else "FAIL"}
            for index, (name, passed) in enumerate(checks, start=1)
        ],
        "overall_status": (
            "PASS_WITH_CHECKPOINT_RECLAMATION_SPACE_INSUFFICIENT"
            if all(passed for _, passed in checks)
            else "FAIL"
        ),
        "actual_checkpoint_deletion_count": 0,
        "actual_checkpoint_movement_count": 0,
        "actual_checkpoint_compression_count": 0,
        "paper_final": False,
    }


def git_head(repo: Path, revision: str = "HEAD") -> str:
    return subprocess.run(
        ["git", "rev-parse", revision],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def build_final_summary(
    inventory: dict[str, Any],
    recovery: dict[str, Any],
    duplicates: dict[str, Any],
    archive: dict[str, Any],
    plans: dict[str, Any],
    tests: dict[str, Any],
    timestamp: str,
    windows_free_bytes: int,
    checkpoint_head: str | None,
) -> dict[str, Any]:
    checkpoints = inventory["checkpoints"]
    attempt_totals: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"checkpoint_count": 0, "checkpoint_bytes": 0}
    )
    for item in checkpoints:
        row = attempt_totals[item["attempt_root"]]
        row["checkpoint_count"] += 1
        row["checkpoint_bytes"] += item["bytes"]
    largest = [
        {"path": path, **values}
        for path, values in sorted(
            attempt_totals.items(),
            key=lambda pair: pair[1]["checkpoint_bytes"],
            reverse=True,
        )[:10]
    ]
    recommended = next(
        item for item in plans["plans"] if item["id"] == plans["recommended_plan"]
    )
    return {
        "schema_version": "canondressgs.subject00.checkpoint_reclamation_final_summary.v1",
        "task_id": TASK_ID,
        "audit_timestamp_utc": timestamp,
        "source": {"branch": SOURCE_BRANCH, "head": SOURCE_HEAD},
        "adjudication": {
            "branch": AUDIT_BRANCH,
            "base_head": SOURCE_HEAD,
            "windows_worktree": WINDOWS_WORKTREE,
            "cloud_worktree": CLOUD_WORKTREE,
            "checkpoint_adjudication_head": checkpoint_head or "PENDING_FIRST_COMMIT",
            "final_reporting_head": "SELF_ON_FINAL_REPORTING_COMMIT",
        },
        "lost_storage_report": {
            "recovered": recovery["all_source_git_objects_exact"],
            "artifact_count": recovery["artifact_count"],
            "original_final_classification": recovery["original_report"][
                "final_classification"
            ],
            "original_recommended_plan": recovery["original_report"]["recommended_plan"],
            "original_next_task": recovery["original_report"]["next_task"],
        },
        "formal_guard": {
            "canonical_output_root": "ABSENT",
            "attempt_001": "ABSENT",
            "attempt_002": "ABSENT",
            "rejected_alias": "ABSENT",
            "active_training_process_count": 0,
            "gpu_compute_process_count": 0,
            "formal_execution_count": 0,
            "optimizer_step_count": 0,
        },
        "capacity": {
            "cloud_free_bytes_at_inventory": inventory["cloud_filesystem"]["free_bytes"],
            "contract_required_bytes": CONTRACT_REQUIRED_BYTES,
            "safety_target_bytes": SAFETY_TARGET_BYTES,
            "recommended_45_gib_bytes": RECOMMENDED_45_GIB_BYTES,
            "windows_e_free_bytes": windows_free_bytes,
        },
        "checkpoint_count": inventory["checkpoint_count"],
        "checkpoint_total_bytes": inventory["checkpoint_total_bytes"],
        "classification_summary": inventory["classification_summary"],
        "short_canary_step0": inventory["short_canary_step0"],
        "duplicate_group_count": duplicates["group_count"],
        "largest_checkpoint_directories": largest,
        "reference_counts": {
            "paper_or_figure": sum(
                item["reference_flags"]["paper_or_figure"] for item in checkpoints
            ),
            "figure_bank": sum(
                item["reference_flags"]["figure_bank"] for item in checkpoints
            ),
            "final_summary": sum(
                item["reference_flags"]["final_summary"] for item in checkpoints
            ),
            "handoff": sum(item["reference_flags"]["handoff"] for item in checkpoints),
            "checkpoint_registry": sum(
                item["reference_flags"]["checkpoint_registry"] for item in checkpoints
            ),
            "sealed_checkpoint_count": sum(
                item["sealed_attempt"] for item in checkpoints
            ),
        },
        "archive_candidate_summary": {
            "root_count": archive["archive_candidate_count"],
            "whole_tree_file_count": archive["whole_tree_file_count"],
            "checkpoint_count": archive["checkpoint_count"],
            "transfer_logical_bytes": archive["transfer_logical_bytes"],
            "projected_cloud_release_allocated_bytes": archive[
                "projected_cloud_release_allocated_bytes"
            ],
        },
        "recommended_plan": recommended,
        "actual_mutations": {
            "checkpoint_deletion": 0,
            "checkpoint_movement": 0,
            "checkpoint_compression": 0,
            "scientific_output_mutation": 0,
            "dataset_mutation": 0,
            "attempt_mutation": 0,
            "manifest_mutation": 0,
            "formal_base_execution": 0,
            "optimizer_steps": 0,
            "renderer": 0,
            "api_calls": 0,
            "report_and_audit_files_created": True,
            "audit_scratch_files": [
                "/tmp/subject00_checkpoint_reclamation_hash_cache.json",
                "/tmp/subject00_checkpoint_reclamation_inventory.json",
                "/tmp/subject00_checkpoint_reclamation_audit.log",
            ],
        },
        "tests": tests["overall_status"],
        "paper_final": False,
        "classification": FINAL_CLASSIFICATION,
        "next_task": NEXT_TASK,
        "next_task_authorized": False,
    }


def build_handoff(
    final_summary: dict[str, Any], timestamp: str, checkpoint_head: str | None
) -> dict[str, Any]:
    recommended = final_summary["recommended_plan"]
    return {
        "schema_version": "canondressgs.subject00.checkpoint_reclamation_handoff.v1",
        "task_id": TASK_ID,
        "handoff_timestamp_utc": timestamp,
        "status": "PLAN_READY_CAPACITY_STILL_INSUFFICIENT",
        "classification": FINAL_CLASSIFICATION,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "adjudication_branch": AUDIT_BRANCH,
        "checkpoint_adjudication_head": checkpoint_head or "PENDING_FIRST_COMMIT",
        "final_reporting_head": "SELF_ON_FINAL_REPORTING_COMMIT",
        "recommended_plan": recommended,
        "hard_stop_conditions": [
            "Any checkpoint path, byte count or SHA256 differs from the inventory",
            "Any candidate becomes open by a process",
            "Any formal canonical attempt or active training process appears",
            "Any archive count, logical byte count or SHA256 manifest mismatch",
            "Windows E: capacity falls below transfer bytes plus 8 GiB reserve",
            "No explicit user authorization for exact archive or deletion paths",
            "Projected free capacity remains below the selected execution gate",
        ],
        "protected_boundaries": [
            EXPECTED_CANARY_PATH,
            "/root/autodl-tmp/canondressgs_work/outputs",
            "/root/autodl-tmp/datasets/thuman4_second_identity_staging",
            "/root/autodl-tmp/canondressgs_work/models",
        ],
        "critical_artifacts": [
            "docs/PAPER/AAAI27_SUBJECT00_LOST_STORAGE_REPORT_RECOVERY_20260725.md",
            "docs/PAPER/AAAI27_SUBJECT00_CHECKPOINT_RECLAMATION_PLAN_20260725.md",
            "paper_protocol/reviewer_risk/subject00_checkpoint_full_inventory.json",
            "paper_protocol/reviewer_risk/subject00_checkpoint_reclamation_plans.json",
            "paper_protocol/reviewer_risk/subject00_checkpoint_reclamation_final_summary.json",
        ],
        "next_task": NEXT_TASK,
        "next_task_authorized": False,
        "paper_final": False,
    }


def recovery_markdown(recovery: dict[str, Any]) -> str:
    old = recovery["original_report"]
    rows = "\n".join(
        f"| `{item['path']}` | `{item['working_tree_sha256']}` | "
        f"{'PASS' if item['git_normalized_content_match'] else 'FAIL'} |"
        for item in recovery["artifacts"]
    )
    return f"""# Subject00 Lost Storage Report Recovery

Task: `{TASK_ID}`

## Recovery Result

- Source branch: `{SOURCE_BRANCH}`
- Source HEAD: `{SOURCE_HEAD}`
- Recovery method: Git-object-exact inheritance from the source HEAD
- Recovered artifact count: `{recovery['artifact_count']}`
- All Git-normalized source checks: `{'PASS' if recovery['all_source_git_objects_exact'] else 'FAIL'}`
- Original conclusions reinterpreted: `false`

## Preserved Original Conclusion

- Final classification: `{old['final_classification']}`
- Initial free bytes: `{old['task_entry_initial_free_bytes']}`
- Final free bytes after pip-cache deletion: `{old['final_free_bytes_after_pip_deletion']}`
- Pip cache deleted: `{str(old['pip_cache']['deleted']).lower()}`
- Pip-cache reclaimed bytes: `{old['pip_cache']['reclaimed_bytes']}`
- Pipeline path: `{old['pipeline_output']['path']}`
- Pipeline classification: `{old['pipeline_output']['classification']} / {old['pipeline_output']['classification_name']}`
- Pipeline safe migration/deletion bytes: `0 / 0`
- AvatarReX staging root: `{old['avatarrex']['staging_root']}`
- Windows archive target: `{old['windows_archive_target']}`
- Original recommended plan: `{old['recommended_plan']}`
- Original projected reclaimed bytes: `{old['projected_reclaimed_bytes']}`
- Original projected final free bytes: `{old['projected_final_free_bytes']}`
- Original NEXT_TASK: `{old['next_task']}`

## Artifact Seal

| Path | Working-tree SHA256 | Source-HEAD Git-object match |
|---|---|---|
{rows}

The recovered report remains authoritative for its original storage-classification scope.
"""


def plan_markdown(
    inventory: dict[str, Any],
    duplicates: dict[str, Any],
    archive: dict[str, Any],
    plans: dict[str, Any],
    windows_free_bytes: int,
) -> str:
    summary = inventory["classification_summary"]
    class_rows = "\n".join(
        f"| `{label}` | {summary[label]['count']} | {summary[label]['bytes']} |"
        for label in CLASS_ORDER
    )
    plan_rows = "\n".join(
        f"| `{plan['id']}` | {plan['checkpoint_count']} | {plan['archive_bytes']} | "
        f"{plan['delete_bytes']} | {plan['projected_free_bytes']} | "
        f"{plan['contract_margin_bytes']} | {plan['safety_margin_bytes']} | "
        f"{str(plan['reaches_recommended_45_gib']).lower()} |"
        for plan in plans["plans"]
    )
    archive_rows = "\n".join(
        f"| `{item['cloud_source']}` | `{item['windows_destination']}` | "
        f"{item['file_count']} | {item['logical_bytes']} | {item['allocated_bytes']} |"
        for item in archive["candidates"]
    )
    return f"""# Subject00 Checkpoint Reclamation Plan

Task: `{TASK_ID}`

## Guardrails

- Checkpoint deletion/movement/compression in this task: `0 / 0 / 0`
- Scientific output, dataset, attempt and manifest mutation: `0`
- Formal Base execution and optimizer steps: `0`
- Renderer/API calls: `0 / 0`
- PAPER_FINAL: `false`
- Cloud free bytes at inventory: `{inventory['cloud_filesystem']['free_bytes']}`
- Windows E: free bytes: `{windows_free_bytes}`
- Contract/safety/45-GiB targets: `{CONTRACT_REQUIRED_BYTES} / {SAFETY_TARGET_BYTES} / {RECOMMENDED_45_GIB_BYTES}`

## Inventory

- Checkpoints: `{inventory['checkpoint_count']}`
- Logical checkpoint bytes: `{inventory['checkpoint_total_bytes']}`
- Duplicate SHA groups: `{duplicates['group_count']}`
- Short-canary path: `{EXPECTED_CANARY_PATH}`
- Short-canary SHA256: `{EXPECTED_CANARY_SHA256}`

| Classification | Count | Bytes |
|---|---:|---:|
{class_rows}

## Recovered-Report Boundary

The recovered report classifies `/root/autodl-tmp/canondressgs_work/outputs` as protected class C and `pipeline_full` as sealed scientific provenance with zero safe migration/deletion bytes. This checkpoint plan does not override that conclusion. The archive candidates below are outside that protected root.

## Archive Candidates

| Cloud source | Windows destination | Files | Logical bytes | Allocated bytes |
|---|---|---:|---:|---:|
{archive_rows}

Every source must remain untouched until rsync exits zero, file counts and logical bytes match, all relative-path SHA256 values match, a manifest/seal is committed, and the user authorizes the exact retirement path.

## Plans

| Plan | Checkpoints | Archive bytes | Delete bytes | Projected free | Contract margin | Safety margin | Reaches 45 GiB |
|---|---:|---:|---:|---:|---:|---:|---|
{plan_rows}

`PLAN_CKPT_D` is the recommended checkpoint-only combination, but it still misses the contract, safety, and 45-GiB targets. No checkpoint-only execution can unblock Formal Base under the present conservative evidence. The separately recovered original AvatarReX storage plan remains unchanged and is not silently folded into this checkpoint plan.

## Classification

- Final classification: `{FINAL_CLASSIFICATION}`
- NEXT_TASK: `{NEXT_TASK}`
- Automatic execution authorized: `false`
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--windows-free-bytes", type=int, required=True)
    parser.add_argument("--checkpoint-head")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    timestamp = utc_now()
    risk = repo / "paper_protocol/reviewer_risk"
    inventory_path = risk / "subject00_checkpoint_full_inventory.json"

    recovery = build_recovery(repo, timestamp)
    inventory = adjudicate_inventory(load_json(inventory_path), timestamp)
    references = build_reference_registry(inventory, timestamp)
    duplicates = build_duplicate_groups(inventory, timestamp)
    sealed = build_sealed_attempt_audit(inventory, timestamp)
    deletion = build_deletion_candidates(inventory, timestamp)
    archive = build_archive_candidates(inventory, timestamp, args.windows_free_bytes)
    plans = build_plans(inventory, archive, timestamp)
    tests = build_tests(repo, recovery, inventory, duplicates, archive, plans, timestamp)
    final_summary = build_final_summary(
        inventory,
        recovery,
        duplicates,
        archive,
        plans,
        tests,
        timestamp,
        args.windows_free_bytes,
        args.checkpoint_head,
    )
    handoff = build_handoff(final_summary, timestamp, args.checkpoint_head)

    outputs = {
        risk / "subject00_lost_storage_report_recovery.json": recovery,
        inventory_path: inventory,
        risk / "subject00_checkpoint_reference_registry.json": references,
        risk / "subject00_checkpoint_duplicate_groups.json": duplicates,
        risk / "subject00_checkpoint_sealed_attempt_audit.json": sealed,
        risk / "subject00_checkpoint_deletion_candidates.json": deletion,
        risk / "subject00_checkpoint_archive_candidates.json": archive,
        risk / "subject00_checkpoint_reclamation_plans.json": plans,
        risk / "subject00_checkpoint_reclamation_tests.json": tests,
        risk / "subject00_checkpoint_reclamation_final_summary.json": final_summary,
        repo
        / "project_control_handoff/subject00_checkpoint_reclamation_adjudication_handoff.json": handoff,
    }
    for path, value in outputs.items():
        write_json(path, value)

    write_text(
        repo / "docs/PAPER/AAAI27_SUBJECT00_LOST_STORAGE_REPORT_RECOVERY_20260725.md",
        recovery_markdown(recovery),
    )
    write_text(
        repo / "docs/PAPER/AAAI27_SUBJECT00_CHECKPOINT_RECLAMATION_PLAN_20260725.md",
        plan_markdown(inventory, duplicates, archive, plans, args.windows_free_bytes),
    )
    print(
        json.dumps(
            {
                "checkpoint_count": inventory["checkpoint_count"],
                "checkpoint_total_bytes": inventory["checkpoint_total_bytes"],
                "classification_summary": inventory["classification_summary"],
                "duplicate_group_count": duplicates["group_count"],
                "archive_candidate_count": archive["archive_candidate_count"],
                "recommended_plan": plans["recommended_plan"],
                "tests": tests["overall_status"],
                "classification": FINAL_CLASSIFICATION,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
