#!/usr/bin/env python3
"""Freeze the outcome-independent Subject00 CommonSafe4 rotation contract.

This tool is deliberately pre-optimizer.  It reads the frozen formal target,
camera, and review registries, ranks the allowed slot04 replacements by the
preregistered camera-geometry tuple, verifies immutable inputs, and writes
only Git-side contract/report artifacts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TASK_ID = "AAAI27-SUBJECT00-BASE60747-COMMONSAFE4-METHOD-MATRIX-001"
SOURCE_BRANCH = "research/subject00-o03-loss-binding-concurrent-provenance-20260727"
SOURCE_HEAD = "37d566dbc3ddcda70f136089b8e8e6c11abbc5a6"
BLOCKER_BRANCH = "research/subject00-base60747-method-matrix-fair-baselines-20260727"
BLOCKER_HEAD = "d3a6500344c8d1fe6a52edf6cfbcc7aea219f864"
BRANCH = "research/subject00-base60747-commonsafe4-method-matrix-20260727"
CONTRACT_NAME = "SUBJECT00_COMMONSAFE4_CARDINAL_PROXY_V1"

TARGET_ROOT = Path(
    "/root/autodl-tmp/canondressgs_work/teacher_targets/"
    "SUBJECT00-24CELL-001/attempt_001"
)
MANIFEST_RELATIVE = Path(
    "10_final_registry/subject00_22_training_full_dataset_v1.json"
)
MANIFEST_SHA = "602820292fea47fe8116bf64824e6e2a8313a3ec31b8fc7cdd66a90c6bad33c1"
BASE_PATH = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-FORMAL-BASE-101245-001/attempt_001/"
    "checkpoints/step_060747.pth"
)
BASE_SHA = "2d09b1ce6bb19b8b6418314caab8dc1cf92d70e432a9e09e009989332ea4a2f7"
TEACHER_PATHS = {
    "O01": Path(
        "/root/autodl-tmp/canondressgs_work/outputs/"
        "SUBJECT00-O01-TEACHER-BASE60747-CAMSAFE7-001/attempt_001/"
        "checkpoints/step_001200.pth"
    ),
    "O03": Path(
        "/root/autodl-tmp/canondressgs_work/outputs/"
        "SUBJECT00-O03-TEACHER-BASE60747-FORMAL-CAMSAFE7-001/attempt_001/"
        "checkpoints/step_001200.pth"
    ),
    "O04": Path(
        "/root/autodl-tmp/canondressgs_work/outputs/"
        "SUBJECT00-O04-TEACHER-BASE60747-8VIEW-001/attempt_001/"
        "checkpoints/step_001200.pth"
    ),
}
TEACHER_SHA = {
    "O01": "c7881862c4eddf5f58538a2278ab7765aa047784681fb02e7cd89cfe846c1892",
    "O03": "054b9efe1086d89b18310b9831d3ed7aea59e286e31200509b58756545fc3920",
    "O04": "2fa7764097d8577c1610bbf222b26d9ea287bd18074371de400a66cd2270f3a1",
}
GARMENTS = ("O01", "O03", "O04")
QUARANTINE = (
    "subject00_O01_slot04_remaining_attempt005_cand00",
    "subject00_O03_slot04_canary_attempt004_cand00",
)
REQUESTED_REPLACEMENT_CANDIDATES = (1, 2, 5, 6)
ORIGINAL_ANCHORS = (0, 7, 3, 4)
FIXED_ANCHORS = (0, 7, 3)
SEEDS = (0, 1, 2)
OUTPUT_ROOT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-CANONDRESSGS-METHOD-BASE60747-COMMONSAFE4-001"
)
OLD_OUTPUT_ROOT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-CANONDRESSGS-METHOD-BASE60747-001/attempt_001"
)
OLD_OUTPUT_TREE_SHA = "ce281935d12e20c6bab2e3e8fc3256b512594fd8388c3ed299f71199072ec363"
OLD_OUTPUT_FILE_COUNT = 19
OLD_OUTPUT_BYTES = 53_138_437
REQUIRED_OBSERVATION_FIELDS = (
    "target_edit_rgb",
    "target_base_rgb",
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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def lf_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(
            (json.dumps(value, indent=2, sort_keys=True, default=str) + "\n").encode()
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def tree_fingerprint(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        file_sha = sha256(path)
        size = path.stat().st_size
        rows.append({"path": relative, "bytes": size, "sha256": file_sha})
        digest.update(f"{relative}\0{size}\0{file_sha}\n".encode())
    return {
        "sha256": digest.hexdigest(),
        "file_count": len(rows),
        "bytes": sum(row["bytes"] for row in rows),
    }


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), *args], text=True
        ).strip()

    state = {
        "branch": run("branch", "--show-current"),
        "head": run("rev-parse", "HEAD"),
        "porcelain_v2": run("status", "--porcelain=v2"),
        "source_merge_base": run("merge-base", "HEAD", SOURCE_HEAD),
        "blocker_merge_base": run("merge-base", "HEAD", BLOCKER_HEAD),
    }
    if state["branch"] != BRANCH or state["porcelain_v2"]:
        raise RuntimeError(f"contract preparation requires clean {BRANCH}")
    if state["source_merge_base"] != SOURCE_HEAD:
        raise RuntimeError("scientific provenance HEAD is not an ancestor")
    if state["blocker_merge_base"] != BLOCKER_HEAD:
        raise RuntimeError("blocker evidence HEAD is not an ancestor")
    return state


def parse_slot(request_id: str) -> int:
    marker = "_slot"
    start = request_id.index(marker) + len(marker)
    return int(request_id[start : start + 2])


def parse_garment(request_id: str) -> str:
    value = request_id.split("_")[1]
    if value not in GARMENTS:
        raise RuntimeError(f"unexpected garment in {request_id}")
    return value


def circular_distance(left: float, right: float) -> float:
    delta = abs(left - right) % 360.0
    return min(delta, 360.0 - delta)


def camera_yaw_degrees(c2w: list[list[float]]) -> tuple[float, list[float]]:
    # OpenCV camera looks along +Z.  World Y is vertical in the frozen
    # calibration, so the horizontal projection is the X/Z pair from the
    # third c2w rotation column.
    axis = [float(c2w[row][2]) for row in range(3)]
    horizontal_norm = math.hypot(axis[0], axis[2])
    if horizontal_norm <= 1e-12:
        raise RuntimeError("camera optical axis has zero horizontal projection")
    yaw = math.degrees(math.atan2(axis[0], axis[2])) % 360.0
    return yaw, axis


def limitation_severity(limitations: Iterable[Any]) -> tuple[int, list[dict[str, Any]]]:
    score = 0
    rows = []
    for value in limitations:
        text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
        lowered = text.lower()
        if any(token in lowered for token in ("critical", "severe", "high")):
            item_score = 3
        elif any(token in lowered for token in ("moderate", "medium")):
            item_score = 2
        else:
            item_score = 1
        score += item_score
        rows.append({"value": value, "severity_score": item_score})
    return score, rows


def immutable_gate() -> dict[str, Any]:
    manifest = TARGET_ROOT / MANIFEST_RELATIVE
    required = {
        "base": (BASE_PATH, BASE_SHA),
        "manifest": (manifest, MANIFEST_SHA),
        **{
            f"teacher_{garment}": (TEACHER_PATHS[garment], TEACHER_SHA[garment])
            for garment in GARMENTS
        },
    }
    actual = {}
    for name, (path, expected) in required.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual[name] = {"path": str(path), "sha256": sha256(path)}
        if actual[name]["sha256"] != expected:
            raise RuntimeError(f"{name} immutable SHA mismatch")

    base_payload = torch.load(BASE_PATH, map_location="cpu", weights_only=False)
    base_step = int(
        base_payload.get(
            "global_step",
            base_payload.get("step", base_payload.get("iteration", -1)),
        )
    )
    if base_step != 60747:
        raise RuntimeError(f"Base internal step changed: {base_step}")
    teacher_metadata = {}
    for garment, path in TEACHER_PATHS.items():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        teacher_metadata[garment] = {
            "global_step": int(payload.get("global_step", -1)),
            "optimizer_step": int(payload.get("optimizer_step", -1)),
            "base_sha256": payload["metadata"].get("base_checkpoint_sha256"),
            "quarantine_optimizer_usage_count": int(
                payload["metadata"].get("quarantine_count_in_optimizer", -1)
            ),
        }
        if teacher_metadata[garment] != {
            "global_step": 1200,
            "optimizer_step": 1200,
            "base_sha256": BASE_SHA,
            "quarantine_optimizer_usage_count": 0,
        }:
            raise RuntimeError(f"{garment} Teacher metadata changed")

    old_tree = tree_fingerprint(OLD_OUTPUT_ROOT)
    if old_tree != {
        "sha256": OLD_OUTPUT_TREE_SHA,
        "file_count": OLD_OUTPUT_FILE_COUNT,
        "bytes": OLD_OUTPUT_BYTES,
    }:
        raise RuntimeError("old METHOD-R0-S0 output tree changed")
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"new isolated output root already exists: {OUTPUT_ROOT}")
    return {
        "status": "PASS",
        "assets": actual,
        "base_internal_step": base_step,
        "teacher_metadata": teacher_metadata,
        "old_output_tree": old_tree,
        "new_output_root_absent": True,
    }


def recover_slots() -> tuple[dict[int, Any], list[int], dict[str, Any]]:
    manifest_path = TARGET_ROOT / MANIFEST_RELATIVE
    manifest = read_json(manifest_path)
    observations = {
        observation["condition_id"]: observation
        for outfit in manifest["outfits"]
        for observation in outfit["observations"]
    }
    conditions = {value["condition_id"]: value for value in manifest["conditions"]}
    if len(observations) != 22 or set(observations) != set(conditions):
        raise RuntimeError("formal manifest condition/observation sets disagree")
    if set(QUARANTINE).intersection(observations):
        raise RuntimeError("quarantine entered formal 22-record manifest")

    by_slot: dict[int, dict[str, Any]] = {}
    camera_paths = sorted((TARGET_ROOT / "06_camera").glob("*_camera.json"))
    if len(camera_paths) != 24:
        raise RuntimeError(f"expected 24 camera records, found {len(camera_paths)}")
    for camera_path in camera_paths:
        camera = read_json(camera_path)
        request_id = camera["request_id"]
        garment = parse_garment(request_id)
        slot = parse_slot(request_id)
        record_path = (
            TARGET_ROOT / "02_records" / f"{request_id}_teacher_target_record.json"
        )
        record = read_json(record_path)
        yaw, optical_axis = (
            camera_yaw_degrees(camera["c2w"]) if camera.get("c2w") else (None, None)
        )
        observation = observations.get(request_id)
        complete = observation is not None
        if complete:
            for field in REQUIRED_OBSERVATION_FIELDS:
                complete = complete and bool(observation.get(field))
                complete = complete and bool(observation.get("checksums", {}).get(field))
        limitations = list(record.get("limitations", []))
        severity, limitation_rows = limitation_severity(limitations)
        metrics = camera.get("registration_metrics") or {}
        eligible = (
            request_id in observations
            and camera.get("training_eligible") is True
            and camera.get("evaluation_eligible") is True
            and camera.get("camera_binding_status")
            == "UNIQUE_SIMILARITY_BINDING_PASS"
            and complete
            and request_id not in QUARANTINE
        )
        item = {
            "request_id": request_id,
            "garment": garment,
            "slot": slot,
            "camera_id": camera.get("source_camera_id"),
            "direction": metrics.get("direction"),
            "training_eligible": camera.get("training_eligible") is True,
            "evaluation_eligible": camera.get("evaluation_eligible") is True,
            "camera_binding_status": camera.get("camera_binding_status"),
            "formal_manifest_member": request_id in observations,
            "target_mask_fields_complete": bool(complete),
            "quarantine": request_id in QUARANTINE,
            "common_safe_eligible": bool(eligible),
            "optical_axis_world": optical_axis,
            "yaw_degrees": yaw,
            "reprojection_rmse_px": metrics.get("reprojection_rmse_px"),
            "inlier_ratio": metrics.get("inlier_ratio"),
            "limitations": limitation_rows,
            "limitation_count": len(limitations),
            "limitation_severity": severity,
            "camera_record_sha256": sha256(camera_path),
            "teacher_target_record_sha256": sha256(record_path),
        }
        by_slot.setdefault(slot, {"garments": {}})["garments"][garment] = item

    slot_registry: dict[int, Any] = {}
    for slot in range(8):
        garments = by_slot.get(slot, {}).get("garments", {})
        if set(garments) != set(GARMENTS):
            raise RuntimeError(f"slot{slot:02d} does not have three provenance records")
        eligible = [garments[name] for name in GARMENTS if garments[name]["common_safe_eligible"]]
        yaws = [float(item["yaw_degrees"]) for item in eligible if item["yaw_degrees"] is not None]
        yaw = yaws[0] if yaws else None
        if yaws and any(circular_distance(yaw, value) > 1e-8 for value in yaws[1:]):
            raise RuntimeError(f"slot{slot:02d} camera yaw differs across garments")
        cameras = sorted(
            {item["camera_id"] for item in garments.values() if item["camera_id"]}
        )
        directions = sorted(
            {item["direction"] for item in eligible if item["direction"]}
        )
        rmse = [
            float(item["reprojection_rmse_px"])
            for item in eligible
            if item["reprojection_rmse_px"] is not None
        ]
        inliers = [
            float(item["inlier_ratio"])
            for item in eligible
            if item["inlier_ratio"] is not None
        ]
        slot_registry[slot] = {
            "slot": slot,
            "slot_label": f"slot{slot:02d}",
            "camera_ids": cameras,
            "directions": directions,
            "yaw_degrees": yaw,
            "garment_coverage": len(eligible),
            "training_eligible_count": sum(
                item["training_eligible"] and item["formal_manifest_member"]
                for item in garments.values()
            ),
            "evaluation_eligible_count": sum(
                item["evaluation_eligible"] and item["formal_manifest_member"]
                for item in garments.values()
            ),
            "common_safe": len(eligible) == 3,
            "worst_case_registration_rmse_px": max(rmse) if len(rmse) == 3 else None,
            "minimum_inlier_ratio": min(inliers) if len(inliers) == 3 else None,
            "limitation_count": sum(
                item["limitation_count"] for item in eligible
            ),
            "limitation_severity": sum(
                item["limitation_severity"] for item in eligible
            ),
            "garments": garments,
        }
    common_safe = [slot for slot, value in slot_registry.items() if value["common_safe"]]
    manifest_audit = {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "training_record_count": len(observations),
        "evaluation_record_count": len(conditions),
        "quarantine_count": len(QUARANTINE),
        "quarantine_in_formal_manifest": sorted(set(QUARANTINE).intersection(observations)),
        "per_outfit_record_count": {
            outfit["outfit_id"]: len(outfit["observations"])
            for outfit in manifest["outfits"]
        },
    }
    return slot_registry, common_safe, manifest_audit


def select_replacement(
    slots: dict[int, Any], common_safe: list[int]
) -> tuple[int, list[dict[str, Any]]]:
    if not all(slots[slot]["common_safe"] for slot in FIXED_ANCHORS):
        raise RuntimeError("slot00/slot03/slot07 are not all common-safe")
    if slots[4]["common_safe"]:
        raise RuntimeError("slot04 unexpectedly became common-safe")
    candidates = [
        slot for slot in REQUESTED_REPLACEMENT_CANDIDATES if slot in common_safe
    ]
    if candidates != list(REQUESTED_REPLACEMENT_CANDIDATES):
        raise RuntimeError(
            f"formal registry changed replacement candidates: {candidates}"
        )
    slot04_yaw = float(slots[4]["garments"]["O04"]["yaw_degrees"])
    anchor_yaws = [float(slots[slot]["yaw_degrees"]) for slot in FIXED_ANCHORS]
    ranking = []
    for slot in candidates:
        value = slots[slot]
        yaw = float(value["yaw_degrees"])
        yaw_distance = circular_distance(yaw, slot04_yaw)
        minimum_anchor_separation = min(
            circular_distance(yaw, anchor_yaw) for anchor_yaw in anchor_yaws
        )
        sorting_tuple = [
            yaw_distance,
            -minimum_anchor_separation,
            float(value["worst_case_registration_rmse_px"]),
            -float(value["minimum_inlier_ratio"]),
            int(value["limitation_severity"]),
            slot,
        ]
        ranking.append(
            {
                "slot": slot,
                "slot_label": f"slot{slot:02d}",
                "camera_id": value["camera_ids"][0],
                "direction": value["directions"][0],
                "yaw_degrees": yaw,
                "slot04_yaw_degrees": slot04_yaw,
                "yaw_distance_degrees": yaw_distance,
                "minimum_anchor_separation_degrees": minimum_anchor_separation,
                "worst_case_registration_rmse_px": value[
                    "worst_case_registration_rmse_px"
                ],
                "minimum_inlier_ratio": value["minimum_inlier_ratio"],
                "limitation_count": value["limitation_count"],
                "limitation_severity": value["limitation_severity"],
                "target_mask_complete_3_of_3": all(
                    item["target_mask_fields_complete"]
                    for item in value["garments"].values()
                ),
                "sorting_tuple": sorting_tuple,
            }
        )
    ranking.sort(key=lambda item: tuple(item["sorting_tuple"]))
    return int(ranking[0]["slot"]), ranking


def rotation_definitions(replacement: int) -> list[dict[str, Any]]:
    return [
        {"rotation": 0, "train_slots": [0, 7], "calibration_slot": 3, "test_slot": replacement},
        {"rotation": 1, "train_slots": [7, 3], "calibration_slot": replacement, "test_slot": 0},
        {"rotation": 2, "train_slots": [3, replacement], "calibration_slot": 0, "test_slot": 7},
        {"rotation": 3, "train_slots": [replacement, 0], "calibration_slot": 7, "test_slot": 3},
    ]


def build_config(
    selected: int, selected_record: dict[str, Any], rotations: list[dict[str, Any]]
) -> tuple[Path, dict[str, Any]]:
    source_path = (
        REPO_ROOT
        / "configs/research/subject00_canondressgs_method_base60747_v1.json"
    )
    config = copy.deepcopy(read_json(source_path))
    config["task_id"] = TASK_ID
    config["run_name"] = OUTPUT_ROOT.name
    config["execution_branch"] = BRANCH
    config["method_contract"]["rotation_contract"] = CONTRACT_NAME
    config["method_contract"]["selected_replacement_slot"] = selected
    config["method_contract"]["selected_replacement_camera"] = selected_record[
        "camera_id"
    ]
    config["method_contract"]["selected_replacement_direction"] = selected_record[
        "direction"
    ]
    config["protocol"]["condition_rotations"] = rotations
    config["protocol"]["common_safe_anchors"] = [0, 7, 3, selected]
    config["protocol"]["initial_execution"]["test_slot"] = selected
    config["output"]["root"] = str(OUTPUT_ROOT)
    config["provenance"] = {
        "scientific_source_branch": SOURCE_BRANCH,
        "scientific_source_head": SOURCE_HEAD,
        "blocker_evidence_branch": BLOCKER_BRANCH,
        "blocker_evidence_head": BLOCKER_HEAD,
        "replacement_selection_artifact": (
            "paper_protocol/reviewer_risk/"
            "subject00_commonsafe_slot04_replacement_selection_20260727.json"
        ),
        "replacement_selected_before_optimizer_step": 0,
    }
    config["disclosure"] = {
        "contract_name": CONTRACT_NAME,
        "original_cardinal_right_replaced": True,
        "replacement_selection_outcome_independent": True,
        "quarantine_policy_unchanged": True,
        "subject02_original_protocol_identical": False,
        "cross_identity_direct_numeric_comparison_requires_matched_protocol": True,
        "paper_disclosure_required": True,
    }
    config["limitations"] = [
        (
            "The original slot04/cardinal-right anchor lacks common three-garment "
            "coverage after permanent quarantine and is replaced by an "
            "outcome-independent camera-geometry proxy."
        ),
        (
            "The CommonSafe4 contract is not numerically comparable to the "
            "unmatched original Subject02 protocol."
        ),
        "Base60747 is accelerated and is not Base101245.",
        "Human visual and scientific review remain pending.",
    ]
    output_path = (
        REPO_ROOT
        / "configs/research/subject00_canondressgs_method_base60747_commonsafe4_v1.json"
    )
    atomic_json(output_path, config)
    return output_path, config


def allowed_config_diff(
    original: dict[str, Any], corrected: dict[str, Any]
) -> dict[str, Any]:
    frozen_sections = (
        "base",
        "targets",
        "teachers",
        "frozen_f2",
        "basis",
        "controller",
        "optimization",
        "validation_gates",
        "final_evaluation",
        "paper_eligible",
        "paper_final",
    )
    checks = {
        section: original[section] == corrected[section] for section in frozen_sections
    }
    if not all(checks.values()):
        raise RuntimeError(f"forbidden config section changed: {checks}")
    return {
        "status": "PASS_ONLY_AUTHORIZED_FIELDS_CHANGED",
        "frozen_section_equality": checks,
        "authorized_changed_fields": [
            "task_id",
            "run_name",
            "execution_branch",
            "method_contract.rotation_contract",
            "method_contract.selected_replacement_*",
            "protocol.condition_rotations",
            "protocol.common_safe_anchors",
            "protocol.initial_execution.test_slot",
            "output.root",
            "provenance",
            "disclosure",
            "limitations",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    git = git_state()
    immutable = immutable_gate()
    slots, common_safe, manifest_audit = recover_slots()
    selected, ranking = select_replacement(slots, common_safe)
    selected_record = ranking[0]
    rotations = rotation_definitions(selected)
    anchors = [0, 7, 3, selected]
    if any(not slots[slot]["common_safe"] for slot in anchors):
        raise RuntimeError("selected four-anchor contract is not common-safe")

    folds = []
    for rotation in rotations:
        fold = {
            **rotation,
            "train_garment_record_count": 6,
            "calibration_garment_record_count": 3,
            "test_garment_record_count": 3,
            "train_coverage": {
                f"slot{slot:02d}": slots[slot]["garment_coverage"]
                for slot in rotation["train_slots"]
            },
            "calibration_coverage": slots[rotation["calibration_slot"]][
                "garment_coverage"
            ],
            "test_coverage": slots[rotation["test_slot"]]["garment_coverage"],
        }
        fold["status"] = (
            "PASS_EXACT_6_3_3"
            if (
                set(fold["train_coverage"].values()) == {3}
                and fold["calibration_coverage"] == 3
                and fold["test_coverage"] == 3
            )
            else "FAIL"
        )
        folds.append(fold)
    if any(fold["status"] != "PASS_EXACT_6_3_3" for fold in folds):
        raise RuntimeError("corrected fold is not exact 6/3/3")

    original_config = read_json(
        REPO_ROOT / "configs/research/subject00_canondressgs_method_base60747_v1.json"
    )
    corrected_config = None
    config_path = (
        REPO_ROOT
        / "configs/research/subject00_canondressgs_method_base60747_commonsafe4_v1.json"
    )
    if args.write:
        config_path, corrected_config = build_config(
            selected, selected_record, rotations
        )
        config_audit = allowed_config_diff(original_config, corrected_config)
    else:
        preview = copy.deepcopy(original_config)
        config_audit = {"status": "DRY_RUN_CONFIG_NOT_WRITTEN"}

    payload = {
        "schema_version": "canondressgs.subject00.commonsafe4.preoptimizer_selection.v1",
        "task_id": TASK_ID,
        "git": git,
        "immutable_gate": immutable,
        "manifest_audit": manifest_audit,
        "common_safe_slot_set": [f"slot{slot:02d}" for slot in common_safe],
        "requested_replacement_candidate_set": [
            f"slot{slot:02d}" for slot in REQUESTED_REPLACEMENT_CANDIDATES
        ],
        "actual_replacement_candidate_set": [
            f"slot{slot:02d}"
            for slot in REQUESTED_REPLACEMENT_CANDIDATES
            if slot in common_safe
        ],
        "slot_yaw_registry": {
            f"slot{slot:02d}": {
                "camera_ids": value["camera_ids"],
                "directions": value["directions"],
                "yaw_degrees": value["yaw_degrees"],
                "garment_coverage": value["garment_coverage"],
                "common_safe": value["common_safe"],
            }
            for slot, value in slots.items()
        },
        "slot_registry": {
            f"slot{slot:02d}": value for slot, value in slots.items()
        },
        "ranking_rule": [
            "minimize absolute circular yaw distance to slot04/cam11/right",
            "maximize minimum angular separation from slot00/slot03/slot07",
            "minimize worst-case three-garment reprojection RMSE",
            "maximize minimum three-garment inlier ratio",
            "minimize disclosed limitation severity",
            "minimize slot ID",
        ],
        "replacement_ranking": ranking,
        "selected_replacement_slot": f"slot{selected:02d}",
        "selected_replacement_camera": selected_record["camera_id"],
        "selected_replacement_direction": selected_record["direction"],
        "selected_before_optimizer_step": 0,
        "optimizer_initialized": False,
        "optimizer_steps": 0,
        "common_safe4_anchor_set": [f"slot{slot:02d}" for slot in anchors],
        "corrected_rotations": rotations,
        "fold_coverage": folds,
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "config_audit": config_audit,
        "status": "PASS_UNIQUE_OUTCOME_INDEPENDENT_REPLACEMENT",
        "paper_eligible": False,
        "paper_final": False,
    }

    if args.write:
        risk = REPO_ROOT / "paper_protocol/reviewer_risk"
        atomic_json(
            risk / "subject00_commonsafe_slot04_replacement_selection_20260727.json",
            payload,
        )
        atomic_json(
            risk / "subject00_base60747_original_rotation_matrix_blocker_correction_20260727.json",
            {
                "schema_version": "canondressgs.subject00.original_matrix_blocker.correction.v1",
                "task_id": TASK_ID,
                "original_matrix_engineering_status": "TRAINER_AND_GPU_HEALTHY",
                "original_matrix_scientific_status": (
                    "BLOCKED_BY_MISSING_COMMON_GARMENT_COVERAGE_AT_SLOT04"
                ),
                "old_method_r0_s0_training_authentic": True,
                "old_method_r0_s0_formal_matrix_eligible": False,
                "old_method_r0_s0_formal_test_completed": False,
                "old_method_r0_s0_reuse_in_new_matrix": False,
                "old_output_tree": immutable["old_output_tree"],
                "blocker_evidence_branch": BLOCKER_BRANCH,
                "blocker_evidence_head": BLOCKER_HEAD,
                "overlay_only": True,
            },
        )
        atomic_json(
            risk / "subject00_commonsafe4_rotation_contract_20260727.json",
            {
                "schema_version": "canondressgs.subject00.commonsafe4.rotation_contract.v1",
                "task_id": TASK_ID,
                "contract_name": CONTRACT_NAME,
                "anchors": [f"slot{slot:02d}" for slot in anchors],
                "rotations": rotations,
                "rotation_count": 4,
                "seeds": list(SEEDS),
                "expected_run_count": 12,
                "optimizer_steps_per_run": 300,
                "expected_optimizer_steps": 3600,
                "checkpoint_steps": [0, 20, 50, 100, 200, 300],
                "method_contract": "PURE_ENDPOINT",
                "dual_support_enabled": False,
                "original_cardinal_right_replaced": True,
                "replacement_selection_outcome_independent": True,
                "quarantine_policy_unchanged": True,
                "subject02_original_protocol_identical": False,
                "cross_identity_direct_numeric_comparison_requires_matched_protocol": True,
                "paper_disclosure_required": True,
            },
        )
        atomic_json(
            risk / "subject00_commonsafe4_fold_coverage_registry_20260727.json",
            {
                "schema_version": "canondressgs.subject00.commonsafe4.fold_coverage.v1",
                "task_id": TASK_ID,
                "common_safe_slots": [f"slot{slot:02d}" for slot in common_safe],
                "anchors": [f"slot{slot:02d}" for slot in anchors],
                "slot_registry": payload["slot_registry"],
                "folds": folds,
                "status": "PASS_ALL_ROTATIONS_EXACT_6_3_3",
            },
        )
        run_rows = [
            {
                "run_id": f"COMMONSAFE4-METHOD-R{rotation['rotation']}-S{seed}",
                "rotation": rotation["rotation"],
                "seed": seed,
                "attempt_id": "attempt_001",
                "train_slots": rotation["train_slots"],
                "calibration_slot": rotation["calibration_slot"],
                "test_slot": rotation["test_slot"],
                "expected_optimizer_steps": 300,
                "expected_checkpoints": [0, 20, 50, 100, 200, 300],
                "status": "PENDING_NOT_STARTED",
            }
            for rotation in rotations
            for seed in SEEDS
        ]
        atomic_json(
            risk / "subject00_commonsafe4_method_12run_execution_registry_20260727.json",
            {
                "schema_version": "canondressgs.subject00.commonsafe4.method_matrix_registry.v1",
                "task_id": TASK_ID,
                "completed_run_count_before": 0,
                "pending_run_count_before": 12,
                "runs": run_rows,
                "status": "FROZEN_PRE_EXECUTION",
            },
        )
        atomic_json(
            risk / "subject02_commonsafe4_matched_protocol_execution_contract_20260727.json",
            {
                "schema_version": "canondressgs.subject02.commonsafe4.matched_contract.v1",
                "task_id": TASK_ID,
                "subject02_matched_run_execution_authorized": False,
                "anchors": [f"slot{slot:02d}" for slot in anchors],
                "rotations": rotations,
                "seeds": list(SEEDS),
                "subject00_selected_replacement": {
                    "slot": f"slot{selected:02d}",
                    "camera": selected_record["camera_id"],
                    "direction": selected_record["direction"],
                },
                "status": "FROZEN_NOT_EXECUTED",
            },
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
