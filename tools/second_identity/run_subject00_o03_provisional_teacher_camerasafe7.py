#!/usr/bin/env python3
"""Execute the one-shot clean Subject00/O03 camera-safe seven-view Teacher.

The only initialization is the sealed Subject00 Formal Base step60747.  The
historical eight-view Teacher checkpoint is loaded only after all 1,200 clean
optimizer steps have completed, and then only for read-only comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.second_identity import (  # noqa: E402
    audit_subject00_o03_provisional_teacher_camera_metrics as review_tools,
)
from tools.second_identity import (  # noqa: E402
    run_subject00_o03_provisional_teacher_base60747 as teacher,
)


TASK_ID = (
    "AAAI27-SUBJECT00-O03-PROVISIONAL-TEACHER-CAMERA-SAFE-7VIEW-RERUN-001"
)
SOURCE_HEAD = "d541edce4b7b1da7ce6d055275993c1faf10e45a"
BRANCH = "research/subject00-o03-provisional-teacher-camerasafe7-rerun-20260727"
CAMERA_HEAD = "a434ae7a78fe898be2658180f20bbcd4391a64c0"
MASK_HEAD = "4ed89d9ac5076d13fef1c2cad8fcf28e3d236ac0"
CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "research"
    / "subject00_o03_provisional_teacher_camerasafe7_v1.json"
)
SAFE_SLOTS = (
    "slot_00",
    "slot_01",
    "slot_02",
    "slot_03",
    "slot_05",
    "slot_06",
    "slot_07",
)
SAFE_CAMERAS = (17, 21, 14, 23, 2, 9, 5)
SAFE_DIRECTIONS = (
    "front",
    "front-left",
    "front-right",
    "left",
    "back-left",
    "back-right",
    "back",
)
SAFE_REQUESTS = (
    "subject00_O03_slot00_canary_attempt004_cand00",
    "subject00_O03_slot01_remaining_attempt005_cand00",
    "subject00_O03_slot02_cand00",
    "subject00_O03_slot03_cand01",
    "subject00_O03_slot05_canary_attempt004_cand00",
    "subject00_O03_slot06_remaining_attempt005_cand00",
    "subject00_O03_slot07_canary_attempt004_cand00",
)
EXCLUDED_SLOT = "slot_04"
EXCLUDED_CAMERA = 11
EXCLUDED_REQUEST = "subject00_O03_slot04_canary_attempt004_cand00"
EXPECTED_COUNTS = {
    "slot_00": 172,
    "slot_01": 172,
    "slot_02": 172,
    "slot_03": 171,
    "slot_05": 171,
    "slot_06": 171,
    "slot_07": 171,
}
CHECKPOINT_STEPS = (0, 300, 600, 900, 1200)
CAMERA_EVIDENCE_PATHS = {
    "eligibility": (
        "paper_protocol/reviewer_risk/"
        "subject00_teacher_target_camera_eligibility_registry_20260727.json"
    ),
    "quarantine": (
        "paper_protocol/reviewer_risk/"
        "subject00_teacher_target_quarantine_registry_20260727.json"
    ),
    "camera_records": (
        "paper_protocol/reviewer_risk/"
        "subject00_teacher_target_camera_records_draft_20260727.json"
    ),
    "safe_manifest": (
        "paper_protocol/reviewer_risk/"
        "subject00_O03_provisional_camera_safe_target_manifest_draft_20260727.json"
    ),
    "final_summary": (
        "paper_protocol/reviewer_risk/"
        "subject00_teacher_target_camera_blocker_resolution_final_summary_20260727.json"
    ),
}
MASK_REGISTRY_PATH = (
    "paper_protocol/reviewer_risk/"
    "subject00_global_mask_accepted_registry_24of24_20260727.json"
)
FINAL_CLASSIFICATION = (
    "SUBJECT00_O03_PROVISIONAL_TEACHER_BASE60747_CAMERA_SAFE_7VIEW_"
    "TECHNICAL_PASS_PENDING_USER_VISUAL_REVIEW"
)
NEXT_TASK = "USER_REVIEW_SUBJECT00_O03_CAMERASAFE7_PROVISIONAL_TEACHER_RESULTS"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    encoded = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write((json.dumps(value, sort_keys=True) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())


def git_output(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), *arguments], text=True
    ).strip()


def git_bytes(commit: str, path: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "show", f"{commit}:{path}"]
    )


def git_json(commit: str, path: str) -> Any:
    return json.loads(git_bytes(commit, path))


def git_blob_sha(commit: str, path: str) -> str:
    return hashlib.sha256(git_bytes(commit, path)).hexdigest()


def tensor_registry(field: teacher.UnboundedGaussianDeltaField) -> dict[str, Any]:
    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "requires_grad": bool(value.requires_grad),
            "tensor_sha256": teacher.tensor_sha(value),
        }
        for name, value in field.state_dict().items()
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config() -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    if (
        config["task_id"] != TASK_ID
        or config["source_git"]["head"] != SOURCE_HEAD
        or config["source_git"]["execution_branch"] != BRANCH
        or config["evidence"]["camera_resolution_head"] != CAMERA_HEAD
        or config["evidence"]["mask_acceptance_head"] != MASK_HEAD
        or tuple(config["targets"]["slots"]) != SAFE_SLOTS
        or tuple(config["targets"]["camera_ids"]) != SAFE_CAMERAS
        or tuple(config["targets"]["request_ids"]) != SAFE_REQUESTS
        or int(config["targets"]["count"]) != 7
        or int(config["targets"]["denominator"]) != 7
        or config["targets"]["excluded"]["request_id"] != EXCLUDED_REQUEST
        or int(config["teacher"]["steps"]) != 1200
        or int(config["teacher"]["seed"]) != 20260718
        or config["teacher"]["loss_name"] != "CAPACITY_ORACLE_LOSS_V1"
        or config["teacher"]["loss_weights"] != teacher.EXPECTED_LOSS
        or config["teacher"]["expected_view_sample_counts"] != EXPECTED_COUNTS
        or tuple(config["teacher"]["checkpoint_steps"]) != CHECKPOINT_STEPS
        or config["historical_contaminated_run"]["initialization_allowed"]
        is not False
        or config["paper_eligible"] is not False
    ):
        raise RuntimeError("camera-safe7 frozen configuration changed")
    return config


def execution_gate(config: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    branch = git_output("branch", "--show-current")
    head = git_output("rev-parse", "HEAD")
    status = git_output("status", "--porcelain=v2")
    if branch != BRANCH or status:
        raise RuntimeError(f"wrong or dirty execution worktree: {branch!r} {status!r}")
    if (
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", SOURCE_HEAD, head]
        ).returncode
        != 0
    ):
        raise RuntimeError("execution head does not descend from fixed audit source")
    if output_root.exists():
        raise RuntimeError(f"fresh output root already exists: {output_root}")
    free_bytes = shutil.disk_usage(output_root.parent).free
    if free_bytes < int(config["runtime"]["minimum_free_bytes"]):
        raise RuntimeError(f"storage gate failed: {free_bytes}")
    process_rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        text=True,
    ).splitlines()
    other_gpu_processes = []
    for line in process_rows:
        pid_text, _, _ = (part.strip() for part in line.split(",", maxsplit=2))
        if int(pid_text) != os.getpid():
            other_gpu_processes.append(line)
    if other_gpu_processes:
        raise RuntimeError(
            f"another GPU compute process is active: {other_gpu_processes}"
        )
    gpu_name = torch.cuda.get_device_name(0)
    if gpu_name != config["runtime"]["gpu_name"]:
        raise RuntimeError(f"wrong GPU: {gpu_name}")
    allowed_process_ids = {os.getpid()}
    parent_pid = os.getppid()
    while parent_pid > 1 and parent_pid not in allowed_process_ids:
        allowed_process_ids.add(parent_pid)
        status_path = Path(f"/proc/{parent_pid}/status")
        if not status_path.is_file():
            break
        parent_line = next(
            (
                line
                for line in status_path.read_text(encoding="utf-8").splitlines()
                if line.startswith("PPid:")
            ),
            None,
        )
        if parent_line is None:
            break
        parent_pid = int(parent_line.split(":", maxsplit=1)[1].strip())
    process_lines = subprocess.check_output(
        ["ps", "-eo", "pid=,args="], text=True
    )
    forbidden = [
        {"pid": int(line.split(maxsplit=1)[0]), "command": line}
        for line in process_lines.splitlines()
        if line.strip()
        and int(line.split(maxsplit=1)[0]) not in allowed_process_ids
        if any(
            marker in line
            for marker in (
                "run_subject00_formal_base",
                "run_subject00_o03_provisional_teacher_base60747.py",
                "run_subject00_o03_provisional_teacher_camerasafe7.py",
                "FullAvatar",
                "GS-VTON",
            )
        )
    ]
    if forbidden:
        raise RuntimeError(f"conflicting process detected: {forbidden}")
    return {
        "branch": branch,
        "head": head,
        "worktree_clean": True,
        "free_bytes": free_bytes,
        "gpu_name": gpu_name,
        "other_gpu_compute_processes": other_gpu_processes,
        "conflicting_processes": [],
    }


def formal_base_gate(config: Mapping[str, Any]) -> dict[str, Any]:
    base_root = Path(config["base"]["checkpoint_path"]).parents[1]
    pause_path = (
        base_root
        / "control"
        / "USER_AUTHORIZED_PAUSE_FOR_PROVISIONAL_DOWNSTREAM_20260727.json"
    )
    resume_path = (
        base_root
        / "control"
        / "FORMAL_BASE_RESUME_FROM_60747_CONTRACT_20260727.json"
    )
    pause = read_json(pause_path)
    resume = read_json(resume_path)
    checkpoint = Path(config["base"]["checkpoint_path"])
    if (
        not pause_path.is_file()
        or not resume_path.is_file()
        or pause["formal_base_final_status"] != "INCOMPLETE"
        or pause["formal_base_completed"] is not False
        or pause["formal_base_resumable"] is not True
        or int(pause["latest_complete_checkpoint_step"]) != 60747
        or resume["FORMAL_BASE_RESUME_READY"] is not True
        or resume["FORMAL_BASE_RESUME_AUTHORIZED"] is not False
        or resume["FORMAL_BASE_COMPLETED"] is not False
        or int(resume["RESUME_CHECKPOINT_STEP"]) != 60747
        or checkpoint.stat().st_size != int(config["base"]["checkpoint_bytes"])
        or sha256_file(checkpoint) != config["base"]["checkpoint_sha256"]
    ):
        raise RuntimeError("Formal Base pause or Base60747 binding changed")
    return {
        "status": "USER_AUTHORIZED_PAUSED",
        "completed": False,
        "durable_resume_step": 60747,
        "resume_ready": True,
        "resume_authorized": False,
        "pause_marker_path": str(pause_path),
        "pause_marker_sha256": sha256_file(pause_path),
        "resume_contract_path": str(resume_path),
        "resume_contract_sha256": sha256_file(resume_path),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": config["base"]["checkpoint_sha256"],
    }


def evidence_bundle(config: Mapping[str, Any]) -> dict[str, Any]:
    camera = {
        name: git_json(CAMERA_HEAD, path)
        for name, path in CAMERA_EVIDENCE_PATHS.items()
    }
    mask = git_json(MASK_HEAD, MASK_REGISTRY_PATH)
    final = camera["final_summary"]
    if (
        final["final_classification"]
        != config["evidence"]["camera_resolution_classification"]
        or int(camera["eligibility"]["teacher_target_training_eligible_count"]) != 22
        or int(camera["eligibility"]["teacher_target_review_only_count"]) != 2
        or int(camera["safe_manifest"]["camera_safe_view_count"]) != 7
        or int(camera["safe_manifest"]["quarantined_view_count"]) != 1
        or int(mask["total_mask_accepted_cell_count"]) != 24
        or int(mask["mask_pair_accepted_count"]) != 24
    ):
        raise RuntimeError("frozen camera/mask evidence counts changed")
    return {
        "payloads": {**camera, "mask": mask},
        "bindings": {
            **{
                name: {
                    "git_path": f"git:{CAMERA_HEAD}:{path}",
                    "sha256": git_blob_sha(CAMERA_HEAD, path),
                }
                for name, path in CAMERA_EVIDENCE_PATHS.items()
            },
            "mask": {
                "git_path": f"git:{MASK_HEAD}:{MASK_REGISTRY_PATH}",
                "sha256": git_blob_sha(MASK_HEAD, MASK_REGISTRY_PATH),
            },
        },
    }


def copy_verified(source: Path, target: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    if (
        not source.is_file()
        or source.stat().st_size != int(expected["bytes"])
        or sha256_file(source) != expected["sha256"]
    ):
        raise RuntimeError(f"source asset binding failed: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise RuntimeError(f"snapshot target unexpectedly exists: {target}")
    shutil.copyfile(source, target)
    if (
        target.stat().st_size != int(expected["bytes"])
        or sha256_file(target) != expected["sha256"]
    ):
        raise RuntimeError(f"snapshot copy verification failed: {target}")
    return {
        "path": str(target),
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
        "source_path": str(source),
        "source_sha256": expected["sha256"],
        "copy_mode": "BYTE_IDENTICAL_RUN_LOCAL_COPY",
    }


def build_snapshot(
    run_root: Path,
    config: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    historical_root = Path(config["historical_contaminated_run"]["run_root"])
    historical_registry_path = historical_root / "contract" / "target_registry.json"
    historical_registry = read_json(historical_registry_path)
    old_by_request = {
        row["request_id"]: row for row in historical_registry["records"]
    }
    payloads = evidence["payloads"]
    eligibility = {row["request_id"]: row for row in payloads["eligibility"]["records"]}
    safe_manifest = {
        row["request_id"]: row for row in payloads["safe_manifest"]["records"]
    }
    camera_records = {
        row["request_id"]: row for row in payloads["camera_records"]["records"]
    }
    mask_records = {
        row["request_id"]: row for row in payloads["mask"]["records"]
    }
    quarantine = {
        row["request_id"]: row for row in payloads["quarantine"]["records"]
    }
    excluded_rows = (
        eligibility[EXCLUDED_REQUEST],
        safe_manifest[EXCLUDED_REQUEST],
        camera_records[EXCLUDED_REQUEST],
        mask_records[EXCLUDED_REQUEST],
        quarantine[EXCLUDED_REQUEST],
    )
    if (
        any(row["slot"] != EXCLUDED_SLOT for row in excluded_rows)
        or eligibility[EXCLUDED_REQUEST]["training_eligible"] is not False
        or eligibility[EXCLUDED_REQUEST]["evaluation_eligible"] is not False
        or safe_manifest[EXCLUDED_REQUEST]["include_in_provisional_target"] is not False
        or camera_records[EXCLUDED_REQUEST]["target_K"] is not None
        or quarantine[EXCLUDED_REQUEST]["review_only"] is not True
    ):
        raise RuntimeError("excluded slot04 quarantine binding changed")
    records: list[dict[str, Any]] = []
    for sampler_index, (slot, camera_id, direction, request_id) in enumerate(
        zip(SAFE_SLOTS, SAFE_CAMERAS, SAFE_DIRECTIONS, SAFE_REQUESTS, strict=True)
    ):
        old = old_by_request[request_id]
        eligible = eligibility[request_id]
        safe = safe_manifest[request_id]
        camera = camera_records[request_id]
        mask = mask_records[request_id]
        if (
            old["slot"] != slot
            or int(old["camera_id"]) != camera_id
            or old["direction"] != direction
            or eligible["training_eligible"] is not True
            or eligible["evaluation_eligible"] is not True
            or eligible["review_only"] is not False
            or safe["include_in_provisional_target"] is not True
            or safe["training_eligible"] is not True
            or camera["camera_status"] != "UNIQUE_SIMILARITY_BINDING_PASS"
            or camera["target_K"] is None
            or camera["materialized"] is not False
            or mask["mask_accepted"] is not True
            or mask["person_mask_human_decision"] != "PASS"
            or mask["garment_mask_human_decision"] != "PASS"
            or mask["mask_pair_human_decision"] != "PASS"
        ):
            raise RuntimeError(f"safe target evidence failed: {request_id}")
        for name, formal in (
            ("accepted_raw", camera["accepted_raw"]),
            ("person_mask", camera["person_mask"]),
            ("garment_mask", camera["garment_mask"]),
        ):
            old_name = name
            if (
                old[old_name]["sha256"] != formal["sha256"]
                or int(old[old_name]["bytes"]) != int(formal["bytes"])
                or mask[old_name]["sha256"] != formal["sha256"]
                or int(mask[old_name]["bytes"]) != int(formal["bytes"])
            ):
                raise RuntimeError(f"three-way asset binding failed: {request_id}/{name}")
        old_matrix = np.asarray(
            old["pixel_registration"]["matrix_2x3"], dtype=np.float64
        )
        formal_matrix = np.asarray(
            camera["registered_source_to_target_transform"], dtype=np.float64
        )[:2]
        if not np.allclose(old_matrix, formal_matrix, atol=1e-12, rtol=0):
            raise RuntimeError(f"camera transform binding differs: {request_id}")
        asset_root = run_root / "inputs" / "target_snapshot" / "targets" / slot
        staged = {
            "accepted_raw": copy_verified(
                Path(old["accepted_raw"]["path"]),
                asset_root / "accepted_raw.png",
                camera["accepted_raw"],
            ),
            "person_mask": copy_verified(
                Path(old["person_mask"]["path"]),
                asset_root / "person_mask.png",
                camera["person_mask"],
            ),
            "garment_mask": copy_verified(
                Path(old["garment_mask"]["path"]),
                asset_root / "garment_mask.png",
                camera["garment_mask"],
            ),
        }
        records.append(
            {
                "schema_version": "canondressgs.full_dataset.record.v1",
                "subject": "Subject00",
                "garment": "O03",
                "slot": slot,
                "sampler_index": sampler_index,
                "request_id": request_id,
                "camera": f"cam{camera_id:02d}",
                "camera_id": camera_id,
                "direction": direction,
                "pose_frame_id": 0,
                "strict_split_role": "STRICT_TRAIN",
                "native_resolution": {
                    "width": int(camera["target_width"]),
                    "height": int(camera["target_height"]),
                },
                "accepted_raw": staged["accepted_raw"],
                "person_mask": staged["person_mask"],
                "garment_mask": staged["garment_mask"],
                "pixel_registration": {
                    "mapping": "formal_source_pixels_to_accepted_target_pixels",
                    "method": "similarity",
                    "matrix_2x3": formal_matrix.tolist(),
                    "source_resolution": {
                        "width": int(camera["calibration_image_width"]),
                        "height": int(camera["calibration_image_height"]),
                    },
                    "target_resolution": {
                        "width": int(camera["target_width"]),
                        "height": int(camera["target_height"]),
                    },
                    "target_K": camera["target_K"],
                    "application": (
                        "formal target_K is recorded; runtime uses the equivalent "
                        "source-camera render plus exact prediction-only inverse "
                        "similarity raster warp used by the frozen Teacher loss path"
                    ),
                },
                "camera_record": {
                    "record_sha256": canonical_sha(camera),
                    "calibration_K": camera["calibration_K"],
                    "target_K": camera["target_K"],
                    "w2c": camera["w2c"],
                    "c2w": camera["c2w"],
                    "camera_status": camera["camera_status"],
                    "camera_binding_status": camera["camera_binding_status"],
                },
                "camera_record_sha256": canonical_sha(camera),
                "camera_safe_status": "TRAINING_TARGET_ELIGIBLE",
                "training_eligible": True,
                "evaluation_eligible": True,
                "appearance_supervision_eligible": True,
                "geometry_supervision_eligible": True,
                "review_only": False,
                "limitation_codes": mask["limitation_codes"],
                "machine_registration": mask["machine_registration_classification"],
                "human_override": mask["human_override_status"],
                "derived_masks": {
                    "person": "person_mask",
                    "garment": "garment_mask",
                    "protected": "person_mask AND NOT garment_mask",
                    "boundary": "3x3 morphological gradient of garment_mask",
                },
                "provenance": {
                    "historical_snapshot_record_sha256": canonical_sha(old),
                    "camera_eligibility_record_sha256": canonical_sha(eligible),
                    "camera_safe_manifest_record_sha256": canonical_sha(safe),
                    "mask_acceptance_record_sha256": canonical_sha(mask),
                    "historical_run_class": (
                        "HISTORICAL_TECHNICAL_RUN_WITH_CAMERA_CONTAMINATION"
                    ),
                },
            }
        )
    if (
        len(records) != 7
        or [row["slot"] for row in records] != list(SAFE_SLOTS)
        or [row["request_id"] for row in records] != list(SAFE_REQUESTS)
        or any(
            token in json.dumps(records, sort_keys=True)
            for token in (EXCLUDED_SLOT, "cam11", EXCLUDED_REQUEST)
        )
    ):
        raise RuntimeError("safe7 snapshot exact-set or exclusion check failed")
    manifest = {
        "schema_version": "canondressgs.full_dataset.v1",
        "task_id": TASK_ID,
        "target_data_class": "RUN_LOCAL_PROVISIONAL_CAMERA_SAFE_7VIEW_SNAPSHOT",
        "paper_eligible": False,
        "materialization_scope": "RUN_LOCAL_ONLY",
        "record_count": 7,
        "denominator": 7,
        "batch_size": 1,
        "mixed_native_resolution": True,
        "slots": list(SAFE_SLOTS),
        "camera_ids": list(SAFE_CAMERAS),
        "request_ids": list(SAFE_REQUESTS),
        "records": records,
        "loader_contract": {
            "implementation_path": str(Path(__file__).resolve()),
            "implementation_sha256": sha256_file(Path(__file__).resolve()),
            "raw": "PIL RGB float32 [0,1], native resolution",
            "masks": "PIL L exact binary {0,255}, native resolution",
            "batch_size": 1,
            "derived_masks": ["protected", "boundary"],
            "no_resize_pad_reencode": True,
        },
        "source_bindings": evidence["bindings"],
        "initialization": config["base"],
    }
    manifest_path = run_root / "inputs" / "target_snapshot" / "manifest.json"
    atomic_json(manifest_path, manifest)
    training_index = {
        "schema_version": "canondressgs.camera_safe7.training_index.v1",
        "task_id": TASK_ID,
        "denominator": 7,
        "slots": list(SAFE_SLOTS),
        "camera_ids": list(SAFE_CAMERAS),
        "request_ids": list(SAFE_REQUESTS),
        "records": [
            {
                "sampler_index": row["sampler_index"],
                "slot": row["slot"],
                "camera_id": row["camera_id"],
                "request_id": row["request_id"],
                "training_eligible": True,
            }
            for row in records
        ],
    }
    evaluation_index = {
        **training_index,
        "schema_version": "canondressgs.camera_safe7.evaluation_index.v1",
    }
    training_index_path = run_root / "inputs" / "target_snapshot" / "training_index.json"
    evaluation_index_path = (
        run_root / "inputs" / "target_snapshot" / "evaluation_index.json"
    )
    atomic_json(training_index_path, training_index)
    atomic_json(evaluation_index_path, evaluation_index)
    excluded_registry = {
        "schema_version": "canondressgs.camera_safe7.excluded_view_registry.v1",
        "task_id": TASK_ID,
        "excluded_count": 1,
        "records": [
            {
                "slot": EXCLUDED_SLOT,
                "camera_id": EXCLUDED_CAMERA,
                "request_id": EXCLUDED_REQUEST,
                "reason": "REVIEW_ONLY_CAMERA_QUARANTINED",
                "training_eligible": False,
                "evaluation_eligible": False,
                "allowed_uses": [
                    "excluded-view registry",
                    "quarantine registry",
                    "historical contaminated-run comparison",
                    "risk disclosure report",
                ],
            }
        ],
    }
    excluded_path = run_root / "contract" / "excluded_view_registry.json"
    atomic_json(excluded_path, excluded_registry)
    for path in (manifest_path, training_index_path, evaluation_index_path):
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in (EXCLUDED_SLOT, "cam11", EXCLUDED_REQUEST)):
            raise RuntimeError(f"slot04 appears in training input: {path}")
    return manifest, {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "training_index_path": str(training_index_path),
        "training_index_sha256": sha256_file(training_index_path),
        "evaluation_index_path": str(evaluation_index_path),
        "evaluation_index_sha256": sha256_file(evaluation_index_path),
        "excluded_registry_path": str(excluded_path),
        "excluded_registry_sha256": sha256_file(excluded_path),
        "historical_target_registry_path": str(historical_registry_path),
        "historical_target_registry_sha256": sha256_file(historical_registry_path),
        "slot04_search_status": "PASS_ABSENT_FROM_ALL_TRAINING_INPUTS",
    }


def load_snapshot(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    loaded = []
    for record in manifest["records"]:
        width = int(record["native_resolution"]["width"])
        height = int(record["native_resolution"]["height"])
        raw = teacher.load_rgb(Path(record["accepted_raw"]["path"]), width, height)
        person = teacher.load_mask(Path(record["person_mask"]["path"]), width, height)
        garment = teacher.load_mask(Path(record["garment_mask"]["path"]), width, height)
        if int(torch.count_nonzero(garment * (1 - person))) != 0:
            raise RuntimeError(f"garment mask escapes person: {record['slot']}")
        protected = person * (1 - garment)
        if not bool(garment.any()) or not bool(protected.any()):
            raise RuntimeError(f"empty derived region: {record['slot']}")
        loaded.append(
            {
                "record": dict(record),
                "raw": raw,
                "person": person,
                "garment": garment,
                "protected": protected,
            }
        )
    if len(loaded) != 7:
        raise RuntimeError("loader did not parse seven records")
    return loaded


def build_camera_items_safe7(
    data_root: Path, loaded: list[dict[str, Any]]
) -> dict[str, Any]:
    """Bind only the seven allowed cameras; cam11 never enters the dataset."""
    dataset = teacher.core.ThumanDataset(
        datadir=str(data_root),
        frame_ids=[0],
        cam_ids=list(SAFE_CAMERAS),
        background=np.ones(3, dtype=np.float32),
        image_scaling=1,
        is_in_memory=False,
    )
    mapping = {
        (int(pose), int(camera)): index
        for index, (pose, camera) in enumerate(dataset.indices)
    }
    if set(mapping) != {(0, camera) for camera in SAFE_CAMERAS}:
        raise RuntimeError(f"safe7 dataset index binding changed: {sorted(mapping)}")
    for target in loaded:
        camera = int(target["record"]["camera_id"])
        if (0, camera) not in mapping or camera == EXCLUDED_CAMERA:
            raise RuntimeError(f"camera binding unavailable or excluded: {camera}")
        item = teacher.core.prepare_item(dataset[mapping[(0, camera)]])
        expected = target["record"]["pixel_registration"]["source_resolution"]
        if (
            int(item["width"]) != int(expected["width"])
            or int(item["height"]) != int(expected["height"])
        ):
            raise RuntimeError(
                f"formal source camera dimensions changed: {target['record']['slot']}"
            )
        target["item"] = item
        target["pixel_registration"] = torch.tensor(
            target["record"]["pixel_registration"]["matrix_2x3"],
            dtype=torch.float32,
            device="cuda",
        )
        for name in ("raw", "person", "garment", "protected"):
            target[name] = target[name].cuda(non_blocking=False)
        target["boundary"] = teacher.mask_boundary(target["garment"])
    return {
        "dataset_camera_ids": list(SAFE_CAMERAS),
        "dataset_camera_count": 7,
        "excluded_camera_loaded": False,
        "excluded_camera_id": EXCLUDED_CAMERA,
        "index_count": len(mapping),
    }


def seal_derived_target_registry(
    run_root: Path, loaded: list[dict[str, Any]]
) -> dict[str, Any]:
    records = []
    for target in loaded:
        record = target["record"]
        records.append(
            {
                "slot": record["slot"],
                "request_id": record["request_id"],
                "camera_id": int(record["camera_id"]),
                "native_resolution": record["native_resolution"],
                "raw_shape": list(target["raw"].shape),
                "person_mask_shape": list(target["person"].shape),
                "garment_mask_shape": list(target["garment"].shape),
                "protected_mask_shape": list(target["protected"].shape),
                "boundary_mask_shape": list(target["boundary"].shape),
                "raw_tensor_sha256": teacher.tensor_sha(target["raw"]),
                "person_mask_tensor_sha256": teacher.tensor_sha(target["person"]),
                "garment_mask_tensor_sha256": teacher.tensor_sha(target["garment"]),
                "protected_mask_tensor_sha256": teacher.tensor_sha(
                    target["protected"]
                ),
                "boundary_mask_tensor_sha256": teacher.tensor_sha(
                    target["boundary"]
                ),
                "camera_record_sha256": record["camera_record_sha256"],
                "limitation_codes": record["limitation_codes"],
                "training_eligible": True,
                "evaluation_eligible": True,
            }
        )
    if (
        len(records) != 7
        or [row["request_id"] for row in records] != list(SAFE_REQUESTS)
        or any(
            token in json.dumps(records, sort_keys=True)
            for token in (EXCLUDED_SLOT, "cam11", EXCLUDED_REQUEST)
        )
    ):
        raise RuntimeError("derived target registry is not exact safe7")
    registry = {
        "schema_version": "canondressgs.camera_safe7.derived_target_registry.v1",
        "task_id": TASK_ID,
        "record_count": 7,
        "denominator": 7,
        "records": records,
    }
    path = (
        run_root
        / "inputs"
        / "target_snapshot"
        / "derived_target_registry.json"
    )
    atomic_json(path, registry)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "record_count": 7,
        "slot04_search_status": "PASS_ABSENT",
        "records": records,
    }


def target_hashes(manifest: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(Path(record[name]["path"])): sha256_file(Path(record[name]["path"]))
        for record in manifest["records"]
        for name in ("accepted_raw", "person_mask", "garment_mask")
    }


def save_checkpoint(
    run_root: Path,
    step: int,
    field: teacher.UnboundedGaussianDeltaField,
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    target_binding: Mapping[str, Any],
    view_counts: Mapping[str, int],
    execution_head: str,
    last_record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    path = run_root / "checkpoints" / f"step_{step:06d}.pth"
    sidecar_path = path.with_suffix(".sidecar.json")
    if path.exists() or sidecar_path.exists():
        raise RuntimeError(f"checkpoint overwrite forbidden: {step}")
    payload = {
        "schema_version": (
            "canondressgs.subject00.o03_camerasafe7_teacher_checkpoint.v1"
        ),
        "task_id": TASK_ID,
        "classification": "PROVISIONAL_BASE60747_CAMERA_SAFE_7VIEW_RESULT",
        "paper_eligible": False,
        "global_step": int(step),
        "optimizer_step": int(step),
        "model": field.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        },
        "initialization": config["base"],
        "historical_contaminated_checkpoint_used_for_initialization": False,
        "target_manifest_sha256": target_binding["manifest_sha256"],
        "training_index_sha256": target_binding["training_index_sha256"],
        "derived_target_registry_sha256": target_binding[
            "derived_target_registry_sha256"
        ],
        "exact_request_ids": list(SAFE_REQUESTS),
        "exact_slots": list(SAFE_SLOTS),
        "exact_camera_ids": list(SAFE_CAMERAS),
        "excluded_request_id": EXCLUDED_REQUEST,
        "target_count": 7,
        "denominator": 7,
        "view_schedule": config["teacher"]["view_schedule"],
        "view_sample_counts": {
            slot: int(view_counts.get(slot, 0)) for slot in SAFE_SLOTS
        },
        "trainable_registry": tensor_registry(field),
        "frozen_registry": {
            "entire_base60747": True,
            "surface_attachment_and_55_joint_lbs": True,
            "non_garment_support_residual_entries": True,
            "shN": True,
        },
        "loss_contract": {
            "name": config["teacher"]["loss_name"],
            "weights": config["teacher"]["loss_weights"],
            "optimizer": config["teacher"]["optimizer"],
            "scheduler": None,
        },
        "execution_head": execution_head,
        "config_sha256": sha256_file(CONFIG_PATH),
        "last_record": dict(last_record) if last_record is not None else None,
    }
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    with temporary.open("rb+") as handle:
        os.fsync(handle.fileno())
    digest = sha256_file(temporary)
    size = temporary.stat().st_size
    os.replace(temporary, path)
    sidecar = {
        "schema_version": (
            "canondressgs.subject00.o03_camerasafe7_teacher_checkpoint_sidecar.v1"
        ),
        "task_id": TASK_ID,
        "step": int(step),
        "path": str(path),
        "bytes": size,
        "sha256": digest,
        "atomically_sealed": True,
        "checkpoint_overwrite": 0,
        "initialization_sha256": config["base"]["checkpoint_sha256"],
        "target_manifest_sha256": target_binding["manifest_sha256"],
        "request_ids": list(SAFE_REQUESTS),
        "excluded_request_id": EXCLUDED_REQUEST,
        "view_sample_counts": {
            slot: int(view_counts.get(slot, 0)) for slot in SAFE_SLOTS
        },
        "paper_eligible": False,
    }
    atomic_json(sidecar_path, sidecar)
    return sidecar


METRIC_NAMES = (
    "full_image_lpips",
    "psnr",
    "ssim",
    "garment_region_lpips",
    "garment_region_psnr",
    "garment_region_ssim",
    "silhouette_iou",
    "boundary_f",
    "protected_region_lpips",
    "protected_region_rgb_mae",
    "alpha_foreground_error",
)


def mean_optional(rows: Iterable[Mapping[str, Any]], name: str) -> float | None:
    values = [row[name] for row in rows if row.get(name) is not None]
    return float(np.mean(values)) if values else None


def aggregate(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        **{name: mean_optional(rows, name) for name in METRIC_NAMES},
        "view_count": len(rows),
        "denominator": len(rows),
        "severe_artifact_count": sum(bool(row["severe_artifact_flag"]) for row in rows),
        "render_time_seconds_mean": float(
            np.mean([row["render_seconds"] for row in rows])
        ),
        "render_time_seconds_total": float(
            np.sum([row["render_seconds"] for row in rows])
        ),
        "fps_from_mean_render_time": float(
            1.0 / max(np.mean([row["render_seconds"] for row in rows]), 1e-12)
        ),
    }


def evaluate(
    rgb: torch.Tensor,
    alpha: torch.Tensor,
    target: Mapping[str, Any],
    metric: Any,
    seconds: float,
) -> dict[str, Any]:
    row = teacher.evaluate_view(rgb, alpha, target, metric)
    alpha_error = float(
        teacher.masked_l1(
            alpha, target["person"], torch.ones_like(target["person"])
        )
    )
    severe = (
        not row["render_finite"]
        or row["background_alpha_mean"] > 0.05
        or row["silhouette_iou"] < 0.5
    )
    return {
        "slot": target["record"]["slot"],
        "request_id": target["record"]["request_id"],
        "camera_id": int(target["record"]["camera_id"]),
        "direction": target["record"]["direction"],
        **row,
        "alpha_foreground_error": alpha_error,
        "severe_artifact_flag": bool(severe),
        "render_seconds": float(seconds),
    }


def create_review(
    run_root: Path,
    render_records: list[dict[str, Any]],
    clean_rows: list[dict[str, Any]],
    sample_counts: Mapping[str, int],
    novel: Mapping[str, Any],
    historical_root: Path,
) -> dict[str, Any]:
    review_root = run_root / "review"
    metrics = {row["slot"]: row for row in clean_rows}
    files: list[Path] = []
    contact_panels = []
    for row in render_records:
        slot = row["slot"]
        for label, value in (
            ("target", row["target"]),
            ("Base60747", row["base"]),
            ("contaminated8", row["contaminated"]),
            ("clean7", row["clean"]),
            ("target-clean |diff| x2", np.clip(np.abs(row["target"] - row["clean"]) * 2, 0, 1)),
        ):
            contact_panels.append((f"{slot} {label}", review_tools.pil_rgb(value)))
    contact_path = review_root / "clean7_total_contact_sheet.png"
    review_tools.make_grid_page(
        contact_path,
        "Subject00 O03 clean camera-safe7 Teacher overview",
        [
            "DISPLAY_ONLY_PROVISIONAL_TEACHER_REVIEW",
            "Seven camera-safe views only. slot04/cam11 is absent from training and metrics.",
            "Columns: target, Base60747, historical contaminated8, clean safe7, target/clean difference.",
            "human_visual_decision=null | scientific_pass=null | paper_eligible=false",
        ],
        contact_panels,
        columns=5,
        tile_width=560,
        tile_height=460,
    )
    files.append(contact_path)
    for row in render_records:
        slot = row["slot"]
        metric = metrics[slot]
        target = review_tools.pil_rgb(row["target"])
        base = review_tools.pil_rgb(row["base"])
        contaminated = review_tools.pil_rgb(row["contaminated"])
        clean = review_tools.pil_rgb(row["clean"])
        person = row["person"][..., 0]
        garment = row["garment"][..., 0]
        protected = row["protected"][..., 0]
        boundary = row["boundary"][..., 0]
        hands = person.copy()
        height, width = hands.shape
        hands[: int(height * 0.25)] = 0
        hands[int(height * 0.75) :] = 0
        hands[:, int(width * 0.43) : int(width * 0.57)] = 0
        panels = [
            ("target raw", target),
            ("person mask", review_tools.pil_rgb(row["person"])),
            ("garment mask", review_tools.pil_rgb(row["garment"])),
            ("Base60747", base),
            ("historical contaminated8", contaminated),
            ("clean safe7", clean),
            (
                "target/clean |difference| x2",
                review_tools.pil_rgb(
                    np.clip(np.abs(row["target"] - row["clean"]) * 2, 0, 1)
                ),
            ),
            (
                "garment crop: target | clean",
                review_tools.paired_crop(target, clean, garment),
            ),
            (
                "protected crop: target | clean",
                review_tools.paired_crop(target, clean, protected),
            ),
            (
                "boundary crop: target | clean",
                review_tools.paired_crop(target, clean, boundary),
            ),
            (
                "face/head: target | clean",
                review_tools.paired_crop(
                    target, clean, person, fractional_y=(0.0, 0.25)
                ),
            ),
            ("hands: target | clean", review_tools.paired_crop(target, clean, hands)),
            (
                "feet: target | clean",
                review_tools.paired_crop(
                    target, clean, person, fractional_y=(0.80, 1.0)
                ),
            ),
        ]
        page = review_root / f"{slot}_clean7_high_resolution_review.png"
        review_tools.make_grid_page(
            page,
            f"Subject00 O03 {slot} clean safe7 review",
            [
                "DISPLAY_ONLY_PROVISIONAL_TEACHER_REVIEW",
                (
                    f"request={row['request_id']} | camera=cam{row['camera_id']:02d} "
                    f"| sample_count={sample_counts[slot]} | "
                    "camera_safe_status=TRAINING_TARGET_ELIGIBLE"
                ),
                (
                    f"LPIPS={metric['full_image_lpips']:.6f} | "
                    f"PSNR={metric['psnr']:.4f} | SSIM={metric['ssim']:.6f} | "
                    f"garment LPIPS={metric['garment_region_lpips']:.6f} | "
                    f"silhouette IoU={metric['silhouette_iou']:.6f} | "
                    f"Boundary F={metric['boundary_f']:.6f}"
                ),
                (
                    f"protected LPIPS={metric['protected_region_lpips']:.6f} | "
                    f"protected MAE={metric['protected_region_rgb_mae']:.6f} | "
                    f"alpha error={metric['alpha_foreground_error']:.6f} | "
                    f"severe_artifact={metric['severe_artifact_flag']}"
                ),
                f"limitations={row['limitation_codes']}",
                "human_visual_decision=null | scientific_pass=null | paper_eligible=false",
            ],
            panels,
        )
        files.append(page)
    historical_risk = (
        historical_root
        / "review"
        / "camera_metric_review_20260727"
        / "slot04_camera_contract_risk_page.png"
    )
    if not historical_risk.is_file():
        raise RuntimeError("historical slot04 risk page is missing")
    quarantine_path = review_root / "slot04_excluded_quarantine_explanation.png"
    review_tools.make_grid_page(
        quarantine_path,
        "slot04 excluded from clean camera-safe7 rerun",
        [
            "DISPLAY_ONLY_PROVISIONAL_TEACHER_REVIEW",
            f"Excluded request: {EXCLUDED_REQUEST}",
            "Latest sealed camera decision: REVIEW_ONLY_CAMERA_QUARANTINED; training_eligible=false; evaluation_eligible=false; selected_model=null.",
            "The historical page below is disclosure evidence only. This request is absent from the clean target manifest, sampler, loss, checkpoints' active request set, and metrics.",
            "human_visual_decision=null | scientific_pass=null | paper_eligible=false",
        ],
        [("historical contamination disclosure", Image.open(historical_risk).convert("RGB"))],
        columns=1,
        tile_width=2600,
        tile_height=2200,
    )
    files.append(quarantine_path)
    novel_panels = []
    for kind in ("different_camera", "different_pose"):
        for name in ("base", "teacher", "comparison"):
            path = review_root / kind / f"{name}.png"
            if not path.is_file():
                raise RuntimeError(f"new finite-render review image missing: {path}")
            novel_panels.append((f"{kind}: {name}", Image.open(path).convert("RGB")))
    novel_path = review_root / "different_camera_and_pose_finite_render_review.png"
    review_tools.make_grid_page(
        novel_path,
        "Different-camera and different-pose finite-render checks",
        [
            "DISPLAY_ONLY_PROVISIONAL_TEACHER_REVIEW",
            f"different_camera={novel['different_camera']['status']} | different_pose={novel['different_pose']['status']}",
            "Finite-render/LBS compatibility check only; not a strict generalization claim.",
            "human_visual_decision=null | scientific_pass=null | paper_eligible=false",
        ],
        novel_panels,
        columns=3,
        tile_width=800,
        tile_height=700,
    )
    files.append(novel_path)
    manifest = {
        "schema_version": "canondressgs.subject00.o03_camerasafe7_review.v1",
        "task_id": TASK_ID,
        "classification": "DISPLAY_ONLY_PROVISIONAL_TEACHER_REVIEW",
        "review_package_path": str(review_root),
        "image_count": len(files),
        "files": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "role": path.stem,
            }
            for path in files
        ],
        "required_content": {
            "clean7_total_contact_sheet": True,
            "safe_view_pages": 7,
            "target_raw": True,
            "person_mask": True,
            "garment_mask": True,
            "base60747": True,
            "historical_contaminated_teacher": True,
            "clean_teacher7": True,
            "target_clean_difference": True,
            "garment_crop": True,
            "protected_crop": True,
            "boundary_crop": True,
            "face_head": True,
            "hands": True,
            "feet": True,
            "different_camera": True,
            "different_pose": True,
            "slot04_quarantine_explanation": True,
        },
        "human_visual_decision": None,
        "scientific_pass": None,
        "paper_eligible": False,
    }
    manifest_path = review_root / "review_manifest.json"
    atomic_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest


def run(args: argparse.Namespace) -> int:
    config = load_config()
    output_root = Path(config["runtime"]["output_root"])
    run_root = output_root / config["runtime"]["attempt"]
    gate = execution_gate(config, output_root)
    formal = formal_base_gate(config)
    historical_checkpoint = Path(
        config["historical_contaminated_run"]["checkpoint_path"]
    )
    if (
        historical_checkpoint.stat().st_size
        != int(config["historical_contaminated_run"]["checkpoint_bytes"])
        or sha256_file(historical_checkpoint)
        != config["historical_contaminated_run"]["checkpoint_sha256"]
    ):
        raise RuntimeError("historical contaminated checkpoint binding changed")
    evidence = evidence_bundle(config)
    output_root.mkdir(parents=True, exist_ok=False)
    run_root.mkdir(exist_ok=False)
    for name in (
        "audits",
        "checkpoints",
        "contract",
        "evaluations",
        "inputs",
        "review",
        "training",
    ):
        (run_root / name).mkdir(exist_ok=False)
    started = time.perf_counter()
    base_checkpoint_hash_before = sha256_file(Path(config["base"]["checkpoint_path"]))
    historical_checkpoint_hash_before = sha256_file(historical_checkpoint)
    try:
        manifest, target_binding = build_snapshot(run_root, config, evidence)
        target_hash_before = target_hashes(manifest)
        targets = load_snapshot(manifest)
        if [row["record"]["request_id"] for row in targets] != list(SAFE_REQUESTS):
            raise RuntimeError("loaded target order differs")
        base, protocol, restore = teacher.restore_base(
            config,
            Path(config["runtime"]["data_root"]),
            Path(config["runtime"]["assets_root"]),
            Path(config["runtime"]["availability_manifest"]),
        )
        base_fingerprint_before = teacher.model_fingerprint(base)
        camera_loader = build_camera_items_safe7(
            Path(config["runtime"]["data_root"]), targets
        )
        derived_target_registry = seal_derived_target_registry(run_root, targets)
        target_binding["derived_target_registry_path"] = (
            derived_target_registry["path"]
        )
        target_binding["derived_target_registry_sha256"] = (
            derived_target_registry["sha256"]
        )
        field = teacher.UnboundedGaussianDeltaField(base).cuda()
        optimizer = torch.optim.Adam(
            field.optimizer_groups(
                config["teacher"]["optimizer"]["geometry_lr"],
                config["teacher"]["optimizer"]["appearance_lr"],
            )
        )
        background = torch.ones(3, device="cuda")
        base_cache: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = {}
        for target in targets:
            base_cache[target["record"]["slot"]] = teacher.render(
                base, target, background, None
            )
        initial_registry = tensor_registry(field)
        smoke_param_before = {
            name: value.detach().clone() for name, value in field.named_parameters()
        }
        set_seed(int(config["teacher"]["seed"]))
        smoke_rgb, smoke_alpha, smoke_seconds = teacher.render(
            base, targets[0], background, field
        )
        smoke_base_rgb, smoke_base_alpha, _ = base_cache[SAFE_SLOTS[0]]
        smoke_parts = teacher.capacity_loss(
            smoke_rgb,
            smoke_alpha,
            targets[0],
            smoke_base_rgb,
            smoke_base_alpha,
            config["teacher"]["loss_weights"],
            field,
        )
        smoke_parts["total"].backward()
        smoke_gradients = {
            name: {
                "present": parameter.grad is not None,
                "finite": bool(
                    parameter.grad is not None and torch.isfinite(parameter.grad).all()
                ),
                "nonzero": bool(
                    parameter.grad is not None and torch.count_nonzero(parameter.grad)
                ),
            }
            for name, parameter in field.named_parameters()
        }
        if not all(
            row["present"] and row["finite"] and row["nonzero"]
            for row in smoke_gradients.values()
        ):
            raise RuntimeError(f"pre-loop gradient smoke failed: {smoke_gradients}")
        optimizer.zero_grad(set_to_none=True)
        smoke_unchanged = all(
            torch.equal(value, smoke_param_before[name])
            for name, value in field.named_parameters()
        )
        if not smoke_unchanged:
            raise RuntimeError("pre-loop smoke changed Teacher parameters")
        preloop = {
            "status": "PASS",
            "target_loader_parse": "PASS_7_OF_7",
            "exact_request_set": list(SAFE_REQUESTS),
            "excluded_request_absent": True,
            "model_initialization": "PASS_BASE60747_ONLY",
            "forward": "PASS_FINITE",
            "loss_construction": "PASS_CAPACITY_ORACLE_LOSS_V1",
            "backward": "PASS_ALL_5_TRAINABLE_TENSORS_FINITE_NONZERO_GRADIENT",
            "gradient_audit": smoke_gradients,
            "optimizer_constructed": True,
            "optimizer_state_entry_count": len(optimizer.state_dict()["state"]),
            "optimizer_steps": 0,
            "parameter_mutation": False,
            "render_seconds": smoke_seconds,
            "cuda_memory_bytes": int(torch.cuda.memory_allocated()),
        }
        atomic_json(run_root / "audits" / "pre_loop_smoke.json", preloop)
        del smoke_rgb, smoke_alpha, smoke_parts, smoke_param_before
        set_seed(int(config["teacher"]["seed"]))
        torch.cuda.reset_peak_memory_stats()
        contract = {
            "schema_version": "canondressgs.subject00.o03_camerasafe7_contract.v1",
            "task_id": TASK_ID,
            "git": gate,
            "config": config,
            "config_sha256": sha256_file(CONFIG_PATH),
            "formal_base": formal,
            "target_binding": target_binding,
            "target_manifest": manifest,
            "camera_loader": camera_loader,
            "derived_target_registry": derived_target_registry,
            "initial_trainable_registry": initial_registry,
            "base_restore": restore,
            "pre_loop_smoke": preloop,
            "historical_contaminated_checkpoint_used_for_initialization": False,
            "paper_eligible": False,
        }
        atomic_json(run_root / "contract" / "execution_contract.json", contract)
        view_counts: Counter[str] = Counter()
        checkpoints = [
            save_checkpoint(
                run_root,
                0,
                field,
                optimizer,
                config,
                target_binding,
                view_counts,
                gate["head"],
                None,
            )
        ]
        gradient_seen = {
            name: 0 for name, parameter in field.named_parameters() if parameter.requires_grad
        }
        loss_initial: float | None = None
        last_record: dict[str, Any] | None = None
        atomic_json(
            run_root / "RUN_STATUS.json",
            {
                "task_id": TASK_ID,
                "status": "RUNNING",
                "stage": "PRE_LOOP_SMOKE_PASS",
                "optimizer_steps": 0,
            },
        )
        training_started = time.perf_counter()
        for step in range(1, 1201):
            target = targets[(step - 1) % 7]
            slot = target["record"]["slot"]
            request_id = target["record"]["request_id"]
            if request_id == EXCLUDED_REQUEST or slot == EXCLUDED_SLOT:
                raise RuntimeError("slot04 reached optimizer path")
            optimizer.zero_grad(set_to_none=True)
            rgb, alpha, render_seconds = teacher.render(
                base, target, background, field
            )
            base_rgb, base_alpha, _ = base_cache[slot]
            parts = teacher.capacity_loss(
                rgb,
                alpha,
                target,
                base_rgb,
                base_alpha,
                config["teacher"]["loss_weights"],
                field,
            )
            if not torch.isfinite(parts["total"]):
                raise FloatingPointError(f"non-finite loss at step {step}")
            parts["total"].backward()
            for name, parameter in field.named_parameters():
                if parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(f"non-finite gradient: {step}/{name}")
                    if bool(torch.count_nonzero(parameter.grad)):
                        gradient_seen[name] += 1
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                field.parameters(),
                float(config["teacher"]["optimizer"]["gradient_clip_norm"]),
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"non-finite gradient norm: {step}")
            optimizer.step()
            view_counts[slot] += 1
            row = {
                "step": step,
                "slot": slot,
                "request_id": request_id,
                "camera_id": int(target["record"]["camera_id"]),
                "view_denominator": 7,
                "loss": {name: float(value.detach()) for name, value in parts.items()},
                "gradient_norm": float(gradient_norm),
                "learning_rates": {
                    group["name"]: float(group["lr"])
                    for group in optimizer.param_groups
                },
                "loss_finite": True,
                "gradient_finite": True,
                "render_seconds": render_seconds,
                "elapsed_seconds": time.perf_counter() - training_started,
                "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
                "checkpoint_event": (
                    f"step_{step:06d}.pth"
                    if step in CHECKPOINT_STEPS
                    else None
                ),
                "view_sample_counts": {
                    name: int(view_counts.get(name, 0)) for name in SAFE_SLOTS
                },
            }
            if loss_initial is None:
                loss_initial = row["loss"]["total"]
            last_record = row
            append_jsonl(run_root / "training" / "state_records.jsonl", row)
            if step in CHECKPOINT_STEPS:
                checkpoints.append(
                    save_checkpoint(
                        run_root,
                        step,
                        field,
                        optimizer,
                        config,
                        target_binding,
                        view_counts,
                        gate["head"],
                        row,
                    )
                )
            if step % 25 == 0 or step == 1:
                atomic_json(
                    run_root / "RUN_STATUS.json",
                    {
                        "task_id": TASK_ID,
                        "status": "RUNNING",
                        "stage": "TRAINING",
                        "optimizer_steps": step,
                        "loss": row["loss"]["total"],
                        "view_sample_counts": row["view_sample_counts"],
                    },
                )
        if (
            dict(view_counts) != EXPECTED_COUNTS
            or sum(view_counts.values()) != 1200
            or view_counts.get(EXCLUDED_SLOT, 0) != 0
            or not all(value > 0 for value in gradient_seen.values())
        ):
            raise RuntimeError(
                f"post-training sampling/gradient audit failed: {view_counts}, {gradient_seen}"
            )
        final_registry = tensor_registry(field)
        trainable_change = {
            name: initial_registry[name]["tensor_sha256"]
            != final_registry[name]["tensor_sha256"]
            for name in (
                "raw_xyz",
                "raw_log_scaling",
                "raw_rotvec",
                "raw_opacity",
                "raw_sh0",
            )
        }
        if not all(trainable_change.values()):
            raise RuntimeError(f"trainable tensors did not all change: {trainable_change}")
        base_fingerprint_after_training = teacher.model_fingerprint(base)
        if base_fingerprint_before != base_fingerprint_after_training:
            raise RuntimeError("frozen Base changed during clean training")
        if sha256_file(Path(config["base"]["checkpoint_path"])) != base_checkpoint_hash_before:
            raise RuntimeError("Base60747 checkpoint changed")
        if target_hashes(manifest) != target_hash_before:
            raise RuntimeError("safe target snapshot changed")
        if sha256_file(historical_checkpoint) != historical_checkpoint_hash_before:
            raise RuntimeError("historical contaminated checkpoint changed before comparison")
        # Historical checkpoint enters memory only here, after all 1,200 clean steps.
        historical_payload = torch.load(historical_checkpoint, map_location="cpu")
        if (
            int(historical_payload["global_step"]) != 1200
            or historical_payload["base_checkpoint"]["checkpoint_sha256"]
            != config["base"]["checkpoint_sha256"]
        ):
            raise RuntimeError("historical checkpoint parse failed")
        contaminated_field = teacher.UnboundedGaussianDeltaField(base).cuda()
        contaminated_field.load_state_dict(historical_payload["model"], strict=True)
        contaminated_field.eval()
        for parameter in contaminated_field.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        historical_loaded_after_optimizer_step = 1200
        lpips_metric, lpips_error = teacher.core.make_lpips_preserving_rng()
        if lpips_metric is None:
            raise RuntimeError(f"LPIPS unavailable: {lpips_error}")
        base_rows: list[dict[str, Any]] = []
        contaminated_rows: list[dict[str, Any]] = []
        clean_rows: list[dict[str, Any]] = []
        render_records: list[dict[str, Any]] = []
        with torch.no_grad():
            for target in targets:
                slot = target["record"]["slot"]
                base_rgb, base_alpha, base_seconds = base_cache[slot]
                contaminated_rgb, contaminated_alpha, contaminated_seconds = teacher.render(
                    base, target, background, contaminated_field
                )
                clean_rgb, clean_alpha, clean_seconds = teacher.render(
                    base, target, background, field
                )
                base_rows.append(
                    evaluate(
                        base_rgb,
                        base_alpha,
                        target,
                        lpips_metric,
                        base_seconds,
                    )
                )
                contaminated_rows.append(
                    evaluate(
                        contaminated_rgb,
                        contaminated_alpha,
                        target,
                        lpips_metric,
                        contaminated_seconds,
                    )
                )
                clean_rows.append(
                    evaluate(
                        clean_rgb,
                        clean_alpha,
                        target,
                        lpips_metric,
                        clean_seconds,
                    )
                )
                render_records.append(
                    {
                        "slot": slot,
                        "request_id": target["record"]["request_id"],
                        "camera_id": int(target["record"]["camera_id"]),
                        "limitation_codes": target["record"]["limitation_codes"],
                        "target": target["raw"].detach().cpu().numpy(),
                        "person": target["person"].detach().cpu().numpy(),
                        "garment": target["garment"].detach().cpu().numpy(),
                        "protected": target["protected"].detach().cpu().numpy(),
                        "boundary": target["boundary"].detach().cpu().numpy(),
                        "base": base_rgb.detach().cpu().numpy(),
                        "contaminated": contaminated_rgb.detach().cpu().numpy(),
                        "clean": clean_rgb.detach().cpu().numpy(),
                    }
                )
        base_macro = aggregate(base_rows)
        contaminated_macro = aggregate(contaminated_rows)
        clean_macro = aggregate(clean_rows)
        comparative = {
            "schema_version": "canondressgs.subject00.o03_camerasafe7_comparison.v1",
            "task_id": TASK_ID,
            "denominator": 7,
            "request_ids": list(SAFE_REQUESTS),
            "historical_run_role": "HISTORICAL_DIAGNOSTIC_ONLY_NOT_INITIALIZATION",
            "base60747": {"per_view": base_rows, "macro": base_macro},
            "historical_contaminated8_on_safe7": {
                "per_view": contaminated_rows,
                "macro": contaminated_macro,
            },
            "clean_camerasafe7": {"per_view": clean_rows, "macro": clean_macro},
            "clean_minus_contaminated_per_view": [
                {
                    "slot": clean["slot"],
                    **{
                        name: clean[name] - contaminated[name]
                        for name in METRIC_NAMES
                        if clean.get(name) is not None
                        and contaminated.get(name) is not None
                    },
                }
                for clean, contaminated in zip(
                    clean_rows, contaminated_rows, strict=True
                )
            ],
            "clean_minus_contaminated_macro": {
                name: clean_macro[name] - contaminated_macro[name]
                for name in METRIC_NAMES
                if clean_macro.get(name) is not None
                and contaminated_macro.get(name) is not None
            },
            "paper_eligible": False,
        }
        atomic_json(run_root / "evaluations" / "comparative_metrics.json", comparative)
        final_metrics = {
            "schema_version": "canondressgs.subject00.o03_camerasafe7_evaluation.v1",
            "task_id": TASK_ID,
            "classification": "PROVISIONAL_BASE60747_CAMERA_SAFE_7VIEW_RESULT",
            "paper_eligible": False,
            "target_count": 7,
            "denominator": 7,
            "request_ids": list(SAFE_REQUESTS),
            "excluded_request_absent": True,
            "per_view": clean_rows,
            "macro": clean_macro,
            "full_image_lpips_interpretation": (
                "Retained for completeness; dominated by full-canvas background "
                "difference and not used alone as success/failure."
            ),
            "human_visual_decision": None,
            "scientific_pass": None,
        }
        atomic_json(run_root / "evaluations" / "final_metrics.json", final_metrics)
        novel = teacher.render_novel_queries(
            run_root,
            base,
            field,
            protocol,
            Path(config["runtime"]["data_root"]),
            background,
        )
        abnormal = teacher.abnormal_summary(field, base)
        animation = {
            "schema_version": "canondressgs.subject00.o03_camerasafe7_animation.v1",
            "task_id": TASK_ID,
            "different_camera": novel["different_camera"],
            "different_pose": novel["different_pose"],
            "animation_compatibility": novel["animation_compatibility"],
            "deformation_status": (
                "PASS_FINITE" if novel["animation_compatibility"]["status"] == "PASS" else "FAIL"
            ),
            "lbs_status": novel["animation_compatibility"]["status"],
            "geometry_health": abnormal,
            "collapse_status": (
                "NONE"
                if clean_macro["severe_artifact_count"] == 0
                else "REVIEW_REQUIRED"
            ),
            "severe_artifact_count": clean_macro["severe_artifact_count"],
            "interpretation_boundary": (
                "Finite-render and LBS compatibility audit only; not a strict "
                "novel-view or novel-pose generalization claim."
            ),
            "paper_eligible": False,
        }
        atomic_json(run_root / "audits" / "animation_audit.json", animation)
        review = create_review(
            run_root,
            render_records,
            clean_rows,
            view_counts,
            novel,
            Path(config["historical_contaminated_run"]["run_root"]),
        )
        base_fingerprint_after = teacher.model_fingerprint(base)
        target_hash_after = target_hashes(manifest)
        checkpoint_paths = sorted((run_root / "checkpoints").glob("*.pth"))
        temporary_paths = sorted((run_root / "checkpoints").glob("*.tmp*"))
        if (
            [int(path.stem.split("_")[1]) for path in checkpoint_paths]
            != list(CHECKPOINT_STEPS)
            or temporary_paths
            or len(checkpoints) != 5
            or base_fingerprint_before != base_fingerprint_after
            or target_hash_before != target_hash_after
            or sha256_file(Path(config["base"]["checkpoint_path"]))
            != base_checkpoint_hash_before
            or sha256_file(historical_checkpoint)
            != historical_checkpoint_hash_before
        ):
            raise RuntimeError("post-training immutable/checkpoint audit failed")
        checkpoint_parse = []
        for path in checkpoint_paths:
            payload = torch.load(path, map_location="cpu")
            step = int(path.stem.split("_")[1])
            checks = {
                "step": int(payload["global_step"]) == step
                and int(payload["optimizer_step"]) == step,
                "initialization": payload["initialization"]["checkpoint_sha256"]
                == config["base"]["checkpoint_sha256"],
                "historical_not_initialization": payload[
                    "historical_contaminated_checkpoint_used_for_initialization"
                ]
                is False,
                "manifest": payload["target_manifest_sha256"]
                == target_binding["manifest_sha256"],
                "derived_targets": payload["derived_target_registry_sha256"]
                == target_binding["derived_target_registry_sha256"],
                "requests": payload["exact_request_ids"] == list(SAFE_REQUESTS),
                "excluded": payload["excluded_request_id"] == EXCLUDED_REQUEST,
                "target_count": int(payload["target_count"]) == 7,
                "denominator": int(payload["denominator"]) == 7,
                "optimizer": isinstance(payload["optimizer"], dict),
                "rng": set(payload["rng"]) == {"python", "numpy", "torch", "cuda"},
                "scheduler_none": payload["scheduler"] is None,
            }
            if not all(checks.values()):
                raise RuntimeError(f"checkpoint parse failed: {path}: {checks}")
            checkpoint_parse.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "step": step,
                    "checks": checks,
                }
            )
        wall_seconds = time.perf_counter() - started
        integrity = {
            "schema_version": "canondressgs.subject00.o03_camerasafe7_integrity.v1",
            "task_id": TASK_ID,
            "optimizer_steps": 1200,
            "view_sample_counts": dict(view_counts),
            "slot04_sample_count": 0,
            "trainable_parameter_change_status": (
                "PASS_ALL_5_TRAINABLE_TENSORS_CHANGED"
            ),
            "trainable_parameter_changes": trainable_change,
            "frozen_parameter_mutation_status": (
                "PASS_BASE60747_FINGERPRINT_UNCHANGED"
            ),
            "base_checkpoint_unchanged": True,
            "target_snapshot_unchanged": True,
            "historical_contaminated_run_unchanged": True,
            "historical_checkpoint_loaded_after_optimizer_step": (
                historical_loaded_after_optimizer_step
            ),
            "historical_checkpoint_used_for_initialization": False,
            "checkpoint_count": 5,
            "checkpoint_parse_status": "PASS_5_OF_5_CPU_TORCH_LOAD",
            "checkpoints": checkpoint_parse,
            "partial_tmp_count": 0,
            "duplicate_process_count": 0,
            "silent_restart_count": 0,
            "nan_inf_status": "NONE",
            "oom_status": "NONE",
            "data_mutations": 0,
            "target_mutations": 0,
            "mask_mutations": 0,
            "base_checkpoint_mutations": 0,
            "historical_run_mutations": 0,
            "paper_modifications": 0,
        }
        atomic_json(run_root / "audits" / "post_training_integrity.json", integrity)
        result = {
            "schema_version": "canondressgs.subject00.o03_camerasafe7_result.v1",
            "task_id": TASK_ID,
            "status": "TECHNICAL_PASS_PENDING_USER_VISUAL_REVIEW",
            "final_classification": FINAL_CLASSIFICATION,
            "next_task": NEXT_TASK,
            "optimizer_steps": 1200,
            "training_steps": 1200,
            "view_sample_counts": dict(view_counts),
            "target_count": 7,
            "excluded_count": 1,
            "excluded_request_id": EXCLUDED_REQUEST,
            "slot04_sample_count": 0,
            "loss_initial": loss_initial,
            "loss_final": last_record["loss"]["total"] if last_record else None,
            "nan_inf_status": "NONE",
            "oom_status": "NONE",
            "wall_seconds": wall_seconds,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
            "checkpoints": checkpoint_parse,
            "metrics_path": str(run_root / "evaluations" / "final_metrics.json"),
            "metrics_sha256": sha256_file(
                run_root / "evaluations" / "final_metrics.json"
            ),
            "comparative_metrics_path": str(
                run_root / "evaluations" / "comparative_metrics.json"
            ),
            "comparative_metrics_sha256": sha256_file(
                run_root / "evaluations" / "comparative_metrics.json"
            ),
            "animation_audit_path": str(
                run_root / "audits" / "animation_audit.json"
            ),
            "review_package_path": str(run_root / "review"),
            "review_manifest_path": review["manifest_path"],
            "different_camera_status": novel["different_camera"]["status"],
            "different_pose_status": novel["different_pose"]["status"],
            "historical_contaminated_checkpoint_used_for_initialization": False,
            "human_visual_decision": None,
            "scientific_pass": None,
            "paper_eligible": False,
            "formal_base_status": "USER_AUTHORIZED_PAUSED",
            "formal_base_resume_ready": True,
            "formal_base_resume_authorized": False,
            "data_mutations": 0,
            "target_mutations": 0,
            "mask_mutations": 0,
            "base_checkpoint_mutations": 0,
            "historical_run_mutations": 0,
            "paper_modifications": 0,
            "paper_final": False,
        }
        atomic_json(run_root / "training" / "training_result.json", result)
        atomic_json(
            run_root / "RUN_STATUS.json",
            {
                "task_id": TASK_ID,
                "status": "COMPLETE",
                "stage": "TECHNICAL_PASS_PENDING_USER_VISUAL_REVIEW",
                "optimizer_steps": 1200,
                "final_classification": FINAL_CLASSIFICATION,
                "next_task": NEXT_TASK,
            },
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    except BaseException as exc:
        failure_text = f"{type(exc).__name__}: {exc}"
        if any(
            token in failure_text
            for token in ("slot04", "slot_04", "cam11", EXCLUDED_REQUEST)
        ):
            failure_classification = (
                "SUBJECT00_O03_CAMSAFE7_RERUN_CONTRACT_VIOLATION_SLOT04_PRESENT"
            )
            failure_next_task = "RESOLVE_SUBJECT00_O03_CAMERASAFE7_TARGET_CONTRACT"
        elif any(
            token in failure_text
            for token in (
                "target evidence",
                "target contract",
                "target snapshot",
                "asset binding",
                "camera transform binding",
            )
        ):
            failure_classification = (
                "SUBJECT00_O03_CAMSAFE7_RERUN_BLOCKED_BY_TARGET_CONTRACT"
            )
            failure_next_task = "RESOLVE_SUBJECT00_O03_CAMERASAFE7_TARGET_CONTRACT"
        elif any(
            token in failure_text
            for token in (
                "checkpoint parse",
                "trainable tensors",
                "frozen Base changed",
                "immutable/checkpoint audit",
            )
        ):
            failure_classification = (
                "SUBJECT00_O03_CAMSAFE7_RERUN_ENGINEERING_EVIDENCE_FAIL"
            )
            failure_next_task = (
                "USER_REVIEW_SUBJECT00_O03_CAMERASAFE7_ENGINEERING_FAILURE"
            )
        else:
            failure_classification = (
                "SUBJECT00_O03_CAMSAFE7_RERUN_ENGINEERING_FAIL"
            )
            failure_next_task = (
                "USER_REVIEW_SUBJECT00_O03_CAMERASAFE7_ENGINEERING_FAILURE"
            )
        failure = {
            "task_id": TASK_ID,
            "status": "FAIL",
            "exception": failure_text,
            "traceback": traceback.format_exc(),
            "optimizer_steps": (
                sum(
                    1
                    for _ in (
                        run_root / "training" / "state_records.jsonl"
                    ).open("r", encoding="utf-8")
                )
                if (run_root / "training" / "state_records.jsonl").is_file()
                else 0
            ),
            "automatic_retry": False,
            "attempt_002_created": False,
            "paper_eligible": False,
            "final_classification": failure_classification,
            "next_task": failure_next_task,
        }
        atomic_json(run_root / "audits" / "failure.json", failure)
        atomic_json(run_root / "RUN_STATUS.json", failure)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Frozen config path; alternate values are not accepted.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.config.resolve() != CONFIG_PATH.resolve():
        raise RuntimeError("alternate configuration is forbidden")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
