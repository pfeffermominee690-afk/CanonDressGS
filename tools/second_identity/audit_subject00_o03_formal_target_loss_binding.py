#!/usr/bin/env python3
"""Read-only preflight for the formal-target O03 camera-safe7 Teacher rerun.

The loss-field binding gate is intentionally evaluated before importing a
renderer, constructing an optimizer, creating the run output root, or touching
CUDA.  The script executes the frozen formal CPU loader, validates Base60747
with a CPU-only torch load, and recovers the actual CAPACITY_ORACLE_LOSS_V1
field bindings from the materialization implementation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


TASK_ID = "AAAI27-SUBJECT00-O03-FORMAL-TARGET-CAMSAFE7-TEACHER-RERUN-001"
SOURCE_BRANCH = (
    "research/subject00-o03-camsafe7-formal-target-equivalence-audit-20260727"
)
SOURCE_HEAD = "113f669eac37fa1f178886f3a7dc1a3a81751441"
MATERIALIZATION_BRANCH = (
    "research/subject00-teacher-target-materialization-quarantine-20260727"
)
MATERIALIZATION_HEAD = "227fd156d420e4bf291413952f448780b8446b37"
FORMAL_SCHEMA = "canondressgs.full_dataset.v1"
EXPECTED_LOADER_SHA_LF = (
    "ee1270ca18a4a85698a03f8fdab693ef78d4d7eb303c4fe4effc4efe912c7d00"
)
EXPECTED_LOADER_SHA_WINDOWS = (
    "786c93355776093e610dfe1bc74efd61233da134550a220bc06601f4f2365508"
)
EXPECTED_BASE_SHA = (
    "2d09b1ce6bb19b8b6418314caab8dc1cf92d70e432a9e09e009989332ea4a2f7"
)
EXPECTED_BASE_BYTES = 724_584_413
EXPECTED_REQUESTS = [
    "subject00_O03_slot00_canary_attempt004_cand00",
    "subject00_O03_slot01_remaining_attempt005_cand00",
    "subject00_O03_slot02_cand00",
    "subject00_O03_slot03_cand01",
    "subject00_O03_slot05_canary_attempt004_cand00",
    "subject00_O03_slot06_remaining_attempt005_cand00",
    "subject00_O03_slot07_canary_attempt004_cand00",
]
EXPECTED_SLOTS = [
    "slot_00",
    "slot_01",
    "slot_02",
    "slot_03",
    "slot_05",
    "slot_06",
    "slot_07",
]
EXPECTED_CAMERAS = ["cam17", "cam21", "cam14", "cam23", "cam02", "cam09", "cam05"]
EXCLUDED_REQUEST = "subject00_O03_slot04_canary_attempt004_cand00"
REQUIRED_SCIENTIFIC_FIELDS = [
    "target_edit_rgb",
    "target_foreground_mask",
    "target_clothing_mask",
    "target_base_rgb",
    "target_edit_mask",
    "target_edit_core_mask",
    "target_preserve_mask",
    "target_transition_mask",
    "target_protected_mask",
    "target_base_foreground_mask",
    "target_old_clothing_mask",
    "target_revealed_skin_mask",
]
MISMATCH_FIELDS = [
    "target_base_rgb",
    "target_edit_mask",
    "target_edit_core_mask",
    "target_preserve_mask",
    "target_transition_mask",
    "target_base_foreground_mask",
    "target_old_clothing_mask",
    "target_revealed_skin_mask",
]
LOSS_ARGUMENT_BINDINGS = {
    "target_base_rgb": "base_rgb",
    "target_edit_mask": "edit_mask",
    "target_transition_mask": "transition_mask",
    "target_base_foreground_mask": "base_foreground",
    "target_old_clothing_mask": "old_clothing_mask",
}
UNBOUND_FIELDS = [
    "target_edit_core_mask",
    "target_preserve_mask",
    "target_revealed_skin_mask",
]
FINAL_CLASSIFICATION = (
    "SUBJECT00_O03_FORMAL_TARGET_RERUN_BLOCKED_BY_LOSS_FIELD_BINDING_AMBIGUITY"
)
NEXT_TASK = "RESOLVE_SUBJECT00_FORMAL_TARGET_LOSS_FIELD_BINDING"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--materialization-repo", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8"
    ).strip()


def function_node(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise RuntimeError(f"function not found: {path}:{name}")


def keyword_sample_bindings(path: Path, function_name: str) -> dict[str, str]:
    node = function_node(path, function_name)
    mappings: dict[str, str] = {}
    for candidate in ast.walk(node):
        if not isinstance(candidate, ast.Call):
            continue
        called = candidate.func
        called_name = (
            called.id
            if isinstance(called, ast.Name)
            else called.attr
            if isinstance(called, ast.Attribute)
            else ""
        )
        if called_name != "capacity_oracle_loss_v1":
            continue
        for keyword in candidate.keywords:
            value = keyword.value
            if (
                keyword.arg
                and isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "to"
            ):
                value = value.func.value
            if (
                keyword.arg
                and isinstance(value, ast.Subscript)
                and isinstance(value.value, ast.Name)
                and value.value.id == "sample"
            ):
                slice_value = value.slice
                if isinstance(slice_value, ast.Constant) and isinstance(
                    slice_value.value, str
                ):
                    mappings[keyword.arg] = slice_value.value
        break
    return mappings


def names_in_function(path: Path, name: str) -> set[str]:
    node = function_node(path, name)
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}


def source_path_for_field(
    manifest_path: Path, observation: dict[str, Any], field: str
) -> Path:
    source_key = {
        "target_foreground_mask": "foreground_mask",
        "target_clothing_mask": "clothing_mask",
    }.get(field, field)
    require(source_key in observation, f"manifest observation lacks {source_key}")
    path = Path(observation[source_key])
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def file_registry(paths: Iterable[Path]) -> dict[str, str]:
    return {
        str(path): sha256_file(path)
        for path in sorted(set(paths), key=lambda item: str(item))
    }


def checkpoint_step(payload: Any) -> int | None:
    if isinstance(payload, dict):
        for key in ("step", "iteration", "global_step", "internal_step"):
            value = payload.get(key)
            if isinstance(value, (int, np.integer)):
                return int(value)
        for value in payload.values():
            if isinstance(value, dict):
                nested = checkpoint_step(value)
                if nested is not None:
                    return nested
    return None


def main() -> None:
    args = parse_args()
    source_repo = args.source_repo.resolve()
    materialization_repo = args.materialization_repo.resolve()
    formal_root = args.formal_root.resolve()
    base_checkpoint = args.base_checkpoint.resolve()
    intended_output_root = args.output_root.resolve()

    require(git(source_repo, "rev-parse", "HEAD") == SOURCE_HEAD, "source HEAD changed")
    require(
        git(materialization_repo, "rev-parse", "HEAD") == MATERIALIZATION_HEAD,
        "materialization HEAD changed",
    )
    require(not git(source_repo, "status", "--porcelain"), "source worktree is dirty")
    require(
        not git(materialization_repo, "status", "--porcelain"),
        "materialization worktree is dirty",
    )
    require(not intended_output_root.exists(), "new output root already exists")

    formal_loader = materialization_repo / "scene" / "full_dressable_dataset.py"
    capacity_oracle = (
        materialization_repo / "scene" / "representation_capacity_oracle.py"
    )
    capacity_adapter = (
        materialization_repo / "tools" / "run_representation_triage_ladder.py"
    )
    objective_adapter = (
        materialization_repo / "tools" / "run_objective_residual_redesign.py"
    )
    support_objective = (
        materialization_repo
        / "scene"
        / "support_aware_region_trusted_objective_v6_1.py"
    )
    support_semantics = (
        materialization_repo / "scene" / "trusted_silhouette_semantics_v6_1.py"
    )
    legacy_runner = (
        source_repo
        / "tools"
        / "second_identity"
        / "run_subject00_o03_provisional_teacher_camerasafe7.py"
    )
    legacy_loss = (
        source_repo
        / "tools"
        / "second_identity"
        / "run_subject00_o03_provisional_teacher_base60747.py"
    )
    for path in (
        formal_loader,
        capacity_oracle,
        capacity_adapter,
        objective_adapter,
        support_objective,
        support_semantics,
        legacy_runner,
        legacy_loss,
    ):
        require(path.is_file(), f"implementation path is missing: {path}")

    loader_sha = sha256_file(formal_loader)
    require(loader_sha == EXPECTED_LOADER_SHA_LF, "cloud loader SHA changed")
    require(
        not (source_repo / "scene" / "full_dressable_dataset.py").exists(),
        "source branch unexpectedly contains the formal loader",
    )
    require(
        not (source_repo / "scene" / "representation_capacity_oracle.py").exists(),
        "source branch unexpectedly contains the formal capacity loss",
    )

    formal_index_path = (
        formal_root
        / "10_final_registry"
        / "indexes"
        / "O03_provisional_base60747_records.json"
    )
    formal_manifest_path = (
        formal_root
        / "10_final_registry"
        / "subject00_22_training_full_dataset_v1.json"
    )
    materialization_summary_path = (
        materialization_repo
        / "paper_protocol"
        / "reviewer_risk"
        / "subject00_teacher_target_materialization_final_summary_20260727.json"
    )
    for path in (formal_index_path, formal_manifest_path, materialization_summary_path):
        require(path.is_file(), f"formal contract path is missing: {path}")
    index = read_json(formal_index_path)
    manifest = read_json(formal_manifest_path)
    materialization_summary = read_json(materialization_summary_path)
    require(index["request_ids"] == EXPECTED_REQUESTS, "formal request order changed")
    require([row["slot"] for row in index["records"]] == EXPECTED_SLOTS, "slot order changed")
    require(
        [row["camera_id"] for row in index["records"]] == EXPECTED_CAMERAS,
        "camera order changed",
    )
    require(EXCLUDED_REQUEST not in index["request_ids"], "slot04 entered formal index")
    require(index["count"] == index["denominator"] == 7, "O03 denominator is not 7")
    require(manifest["schema_version"] == FORMAL_SCHEMA, "formal schema changed")
    require(
        materialization_summary["total_provenance_record_count"] == 24,
        "formal provenance count changed",
    )
    require(
        materialization_summary["training_eligible_record_count"] == 22,
        "formal training count changed",
    )
    require(
        materialization_summary["review_only_quarantined_record_count"] == 2,
        "formal quarantine count changed",
    )
    require(
        materialization_summary["O03_training_record_count"] == 7,
        "formal O03 count changed",
    )

    observations = {
        observation["condition_id"]: observation
        for outfit in manifest["outfits"]
        if outfit["outfit_id"] == "O03"
        for observation in outfit["observations"]
    }
    require(
        list(observations) == EXPECTED_REQUESTS,
        "formal manifest O03 observation order changed",
    )

    # Import and execute only the formal CPU loader.
    sys.path.insert(0, str(materialization_repo))
    from scene.full_dressable_dataset import (  # noqa: PLC0415
        FullDressableTrainingDataset,
    )

    dataset = FullDressableTrainingDataset(
        formal_manifest_path, "train", reference_count=1, seed=20260718
    )
    samples = [
        sample
        for sample in (dataset[index] for index in range(len(dataset)))
        if sample["target_condition_id"] in EXPECTED_REQUESTS
    ]
    require(
        [sample["target_condition_id"] for sample in samples] == EXPECTED_REQUESTS,
        "formal loader O03 order changed",
    )
    require(len(samples) == 7, "formal loader did not produce exactly seven O03 samples")

    record_by_id = {row["request_id"]: row for row in index["records"]}
    field_rows: list[dict[str, Any]] = []
    immutable_paths: list[Path] = [formal_index_path, formal_manifest_path]
    for sample in samples:
        request_id = sample["target_condition_id"]
        observation = observations[request_id]
        index_record = record_by_id[request_id]
        record_path = formal_root / index_record["record_path"]
        camera_path = formal_root / index_record["camera_path"]
        immutable_paths.extend((record_path, camera_path))
        for field in REQUIRED_SCIENTIFIC_FIELDS:
            require(field in sample, f"{request_id} loader output lacks {field}")
            value = sample[field]
            require(isinstance(value, torch.Tensor), f"{request_id}:{field} is not tensor")
            require(torch.isfinite(value).all().item(), f"{request_id}:{field} non-finite")
            require(value.ndim == 3, f"{request_id}:{field} rank is not 3")
            source_path = source_path_for_field(formal_manifest_path, observation, field)
            require(source_path.is_file(), f"{request_id}:{field} source missing")
            immutable_paths.append(source_path)
            field_rows.append(
                {
                    "request_id": request_id,
                    "slot": index_record["slot"],
                    "camera_id": index_record["camera_id"],
                    "field": field,
                    "source_path": str(source_path),
                    "source_sha256": sha256_file(source_path),
                    "implementation_path": str(formal_loader),
                    "implementation_sha256_lf": loader_sha,
                    "implementation_sha256_windows_frozen": EXPECTED_LOADER_SHA_WINDOWS,
                    "loader_field_key": field,
                    "dtype": str(value.dtype),
                    "shape_chw": list(value.shape),
                    "minimum": float(value.min()),
                    "maximum": float(value.max()),
                    "nonzero_count": int(torch.count_nonzero(value)),
                    "value_sha256": tensor_sha(value),
                    "finite": True,
                }
            )

    before_hashes = file_registry(immutable_paths)

    # CPU-only Base60747 validation; no model construction and no CUDA.
    require(base_checkpoint.is_file(), "Base60747 checkpoint is missing")
    require(base_checkpoint.stat().st_size == EXPECTED_BASE_BYTES, "Base60747 bytes changed")
    base_sha = sha256_file(base_checkpoint)
    require(base_sha == EXPECTED_BASE_SHA, "Base60747 SHA changed")
    require(
        ".partial" not in base_checkpoint.name and ".tmp" not in base_checkpoint.name,
        "Base60747 path is partial/tmp",
    )
    payload = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    base_step = checkpoint_step(payload)
    require(base_step == 60747, f"Base60747 internal step is {base_step}")
    top_keys = sorted(str(key) for key in payload) if isinstance(payload, dict) else []
    model_state_present = any(
        token in key.lower()
        for key in top_keys
        for token in ("model", "state_dict", "gaussian", "avatar")
    )
    frozen_state_present = any("frozen" in key.lower() for key in top_keys)
    del payload

    # Recover actual mappings from the formal capacity implementation.
    capacity_node = function_node(capacity_oracle, "capacity_oracle_loss_v1")
    capacity_arguments = [
        argument.arg
        for argument in (
            capacity_node.args.posonlyargs
            + capacity_node.args.args
            + capacity_node.args.kwonlyargs
        )
    ]
    ladder_bindings = keyword_sample_bindings(
        capacity_adapter, "loss_for_sample"
    )
    redesign_bindings = keyword_sample_bindings(
        objective_adapter, "_capacity_loss"
    )
    require(ladder_bindings == redesign_bindings, "formal capacity adapters disagree")
    require(
        ladder_bindings
        == {
            "target_rgb": "target_edit_rgb",
            "base_rgb": "target_base_rgb",
            "target_foreground": "target_foreground_mask",
            "base_foreground": "target_base_foreground_mask",
            "edit_mask": "target_edit_mask",
            "clothing_mask": "target_clothing_mask",
            "old_clothing_mask": "target_old_clothing_mask",
            "protected_mask": "target_protected_mask",
            "transition_mask": "target_transition_mask",
        },
        "formal CAPACITY_ORACLE_LOSS_V1 adapter mapping changed",
    )
    inverse_bindings = {field: argument for argument, field in ladder_bindings.items()}
    require(
        all(inverse_bindings.get(field) == argument for field, argument in LOSS_ARGUMENT_BINDINGS.items()),
        "expected mismatch-field bindings changed",
    )
    require(
        all(field not in inverse_bindings for field in UNBOUND_FIELDS),
        "an expected unbound field gained a formal capacity binding",
    )

    capacity_source = capacity_oracle.read_text(encoding="utf-8")
    legacy_source = legacy_loss.read_text(encoding="utf-8")
    support_source = support_objective.read_text(encoding="utf-8")
    support_semantics_source = support_semantics.read_text(encoding="utf-8")
    require('CAPACITY_LOSS_NAME = "CAPACITY_ORACLE_LOSS_V1"' in capacity_source, "loss name changed")
    require(
        not any(field in legacy_source for field in MISMATCH_FIELDS),
        "legacy source loss unexpectedly consumes a mismatch field",
    )
    require(
        "V6_1_OBJECTIVE_NAME" in support_source
        and 'V6_1_OBJECTIVE_NAME = "SUPPORT_AWARE_REGION_TRUSTED_OBJECTIVE_V6_1"'
        in support_semantics_source,
        "support-aware objective identity changed",
    )

    loss_consumers = {
        "target_base_rgb": {
            "status": "BOUND_IN_MATERIALIZATION_ADAPTER_BUT_NOT_FROZEN_SOURCE_EXECUTION",
            "argument": "base_rgb",
            "loss_components": ["protected_rgb"],
        },
        "target_edit_mask": {
            "status": "BOUND_IN_MATERIALIZATION_ADAPTER_BUT_NOT_FROZEN_SOURCE_EXECUTION",
            "argument": "edit_mask",
            "loss_components": ["garment_rgb"],
        },
        "target_edit_core_mask": {
            "status": "UNBOUND_IN_CAPACITY_ORACLE_LOSS_V1",
            "argument": None,
            "loss_components": [],
        },
        "target_preserve_mask": {
            "status": "UNBOUND_IN_CAPACITY_ORACLE_LOSS_V1",
            "argument": None,
            "loss_components": [],
        },
        "target_transition_mask": {
            "status": "BOUND_IN_MATERIALIZATION_ADAPTER_BUT_NOT_FROZEN_SOURCE_EXECUTION",
            "argument": "transition_mask",
            "loss_components": ["boundary_rgb"],
        },
        "target_base_foreground_mask": {
            "status": "BOUND_IN_MATERIALIZATION_ADAPTER_BUT_NOT_FROZEN_SOURCE_EXECUTION",
            "argument": "base_foreground",
            "loss_components": ["new_silhouette_alpha", "protected_alpha"],
        },
        "target_old_clothing_mask": {
            "status": "BOUND_IN_MATERIALIZATION_ADAPTER_BUT_NOT_FROZEN_SOURCE_EXECUTION",
            "argument": "old_clothing_mask",
            "loss_components": ["garment_rgb"],
        },
        "target_revealed_skin_mask": {
            "status": "UNBOUND_IN_CAPACITY_ORACLE_LOSS_V1",
            "argument": None,
            "loss_components": [],
        },
    }
    unique_consumer_count = sum(
        row["status"].startswith("BOUND_IN_MATERIALIZATION")
        for row in loss_consumers.values()
    )
    require(unique_consumer_count == 5, "binding count changed")
    require(len(UNBOUND_FIELDS) == 3, "unbound count changed")

    after_hashes = file_registry(immutable_paths)
    require(before_hashes == after_hashes, "formal target files mutated during preflight")
    require(sha256_file(base_checkpoint) == base_sha, "Base60747 mutated during preflight")
    require(not intended_output_root.exists(), "output root was created during blocked preflight")

    snapshot = {
        "schema_version": (
            "canondressgs.subject00.o03.formal_target_loss_binding_preflight.v1"
        ),
        "task_id": TASK_ID,
        "execution_mode": "READ_ONLY_CPU_PREFLIGHT_BEFORE_OPTIMIZER_GATE",
        "source": {
            "branch": SOURCE_BRANCH,
            "head": SOURCE_HEAD,
            "repo": str(source_repo),
            "formal_loader_present": False,
            "formal_capacity_loss_present": False,
        },
        "materialization": {
            "branch": MATERIALIZATION_BRANCH,
            "head": MATERIALIZATION_HEAD,
            "repo": str(materialization_repo),
            "summary_path": str(materialization_summary_path),
            "total_provenance_count": 24,
            "training_count": 22,
            "quarantine_count": 2,
            "O03_count": 7,
        },
        "formal_target": {
            "root": str(formal_root),
            "schema": FORMAL_SCHEMA,
            "index_path": str(formal_index_path),
            "index_sha256": sha256_file(formal_index_path),
            "manifest_path": str(formal_manifest_path),
            "manifest_sha256": sha256_file(formal_manifest_path),
            "request_ids": EXPECTED_REQUESTS,
            "slots": EXPECTED_SLOTS,
            "camera_ids": EXPECTED_CAMERAS,
            "excluded_request_ids": [EXCLUDED_REQUEST],
            "slot04_present": False,
            "field_rows": field_rows,
            "required_field_count": len(REQUIRED_SCIENTIFIC_FIELDS),
            "required_fields": REQUIRED_SCIENTIFIC_FIELDS,
            "mismatch_field_count": len(MISMATCH_FIELDS),
            "mismatch_fields": MISMATCH_FIELDS,
            "load_status": "PASS_12_FIELDS_X_7_RECORDS",
            "value_hash_status": "PASS_84_OF_84_FINITE_VALUE_HASHES_RECORDED",
            "native_resolutions": sorted(
                {
                    (
                        int(sample["target_camera"]["width"]),
                        int(sample["target_camera"]["height"]),
                    )
                    for sample in samples
                }
            ),
            "batch_size": 1,
            "immutable_registry_entry_count": len(before_hashes),
            "immutable_before_sha256": before_hashes,
            "immutable_after_sha256": after_hashes,
        },
        "loader": {
            "path": str(formal_loader),
            "sha256_lf_runtime": loader_sha,
            "sha256_windows_frozen": EXPECTED_LOADER_SHA_WINDOWS,
            "zero_optimizer_smoke": "PASS_7_OF_7_CPU_ONLY",
            "gpu_forward_calls": 0,
        },
        "base_checkpoint": {
            "path": str(base_checkpoint),
            "bytes": base_checkpoint.stat().st_size,
            "sha256": base_sha,
            "cpu_torch_load": "PASS",
            "internal_step": base_step,
            "top_level_keys": top_keys,
            "model_state_present_by_key": model_state_present,
            "frozen_state_present_by_key": frozen_state_present,
            "partial_or_tmp": False,
            "mutation_count": 0,
        },
        "implementation": {
            "capacity_oracle_path": str(capacity_oracle),
            "capacity_oracle_sha256": sha256_file(capacity_oracle),
            "capacity_oracle_arguments": capacity_arguments,
            "capacity_adapter_path": str(capacity_adapter),
            "capacity_adapter_sha256": sha256_file(capacity_adapter),
            "objective_adapter_path": str(objective_adapter),
            "objective_adapter_sha256": sha256_file(objective_adapter),
            "adapter_bindings": ladder_bindings,
            "legacy_runner_path": str(legacy_runner),
            "legacy_runner_sha256": sha256_file(legacy_runner),
            "legacy_loss_path": str(legacy_loss),
            "legacy_loss_sha256": sha256_file(legacy_loss),
            "legacy_loss_names": sorted(names_in_function(legacy_loss, "capacity_loss")),
            "support_objective_path": str(support_objective),
            "support_objective_sha256": sha256_file(support_objective),
            "support_semantics_path": str(support_semantics),
            "support_semantics_sha256": sha256_file(support_semantics),
            "support_objective_contract": (
                "SUPPORT_AWARE_REGION_TRUSTED_OBJECTIVE_V6_1"
            ),
        },
        "loss_binding": {
            "loss_contract": "CAPACITY_ORACLE_LOSS_V1",
            "weights": {
                "garment_rgb": 1.0,
                "alpha_foreground": 0.5,
                "new_silhouette_alpha": 1.0,
                "boundary_rgb": 0.25,
                "protected_rgb": 10.0,
                "protected_alpha": 5.0,
                "stability": 0.0001,
            },
            "consumer_map": loss_consumers,
            "uniquely_bound_in_materialization_adapter_count": unique_consumer_count,
            "unbound_field_count": len(UNBOUND_FIELDS),
            "unbound_fields": UNBOUND_FIELDS,
            "source_execution_path_status": (
                "FAIL_SOURCE_HEAD_HAS_ONLY_RUNLOCAL_SNAPSHOT_LOSS_PATH"
            ),
            "status": (
                "FAIL_3_FIELDS_UNBOUND_AND_5_BINDINGS_NOT_IN_FROZEN_SOURCE_EXECUTION"
            ),
            "gate_pass": False,
        },
        "execution": {
            "intended_output_root": str(intended_output_root),
            "output_root_created": False,
            "optimizer_constructed": False,
            "optimizer_steps_completed": 0,
            "gpu_forward_calls": 0,
            "backward_calls": 0,
            "checkpoint_count": 0,
            "automatic_retry": False,
            "run_count": 0,
            "target_mutations": 0,
            "base_checkpoint_mutations": 0,
            "paper_modifications": 0,
        },
        "final_classification": FINAL_CLASSIFICATION,
        "next_task": NEXT_TASK,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
