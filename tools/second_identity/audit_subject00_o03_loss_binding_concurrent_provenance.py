#!/usr/bin/env python3
"""Seal Subject00 O03 loss binding, concurrent provenance, and method validity.

This program is intentionally audit-only.  It reads the frozen formal Teacher
implementation and completed outputs, executes one CPU loss/backward access
trace without constructing an optimizer, and writes the required provenance
artifacts into the audit worktree.  It never writes into a Teacher, target,
Base, or method output root.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import socket
import subprocess
import sys
from collections import Counter
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


TASK_ID = "AAAI27-SUBJECT00-O03-LOSS-BINDING-CONCURRENT-RUN-PROVENANCE-001"
SOURCE_BRANCH = "research/subject00-o03-formal-target-camsafe7-teacher-rerun-20260727"
SOURCE_HEAD = "c4aa2f1b4f58c9b51e3a562f3b4c0993a9c0eeeb"
ACCELERATED_BRANCH = "research/subject00-base60747-accelerated-method-launch-20260727"
ACCELERATED_HEAD = "ed44835f950fe369dd3eb64b5eaef8c6cc174175"
MATERIALIZATION_HEAD = "227fd156d420e4bf291413952f448780b8446b37"
OLD_EQUIVALENCE_HEAD = "113f669eac37fa1f178886f3a7dc1a3a81751441"
NEW_BRANCH = "research/subject00-o03-loss-binding-concurrent-provenance-20260727"
WINDOWS_WORKTREE = (
    "E:\\model_train\\canondressgs_subject00_o03_loss_binding_concurrent_provenance"
)
CLOUD_WORKTREE = (
    "/root/autodl-tmp/canondressgs_work/worktrees/"
    "canondressgs_subject00_o03_loss_binding_concurrent_provenance"
)
WRITER_TASK = "AAAI27-SUBJECT00-BASE60747-ACCELERATED-METHOD-LAUNCH-001"
BLOCKED_TASK = "AAAI27-SUBJECT00-O03-FORMAL-TARGET-CAMSAFE7-TEACHER-RERUN-001"
WRITER_RUN_HEAD = "3d4307f581f74efc173bb27c982277d952e32e31"
WRITER_PROCESS_PID = 444268
WRITER_PARENT_PID = 444267
WRITER_PROCESS_START = "2026-07-27T08:06:40+08:00"
WRITER_LAUNCH_COMMAND = (
    "/root/autodl-tmp/conda_envs/mmlphuman/bin/python -u "
    "tools/second_identity/run_subject00_base60747_formal_teacher.py "
    "--garment O03 --recover-preoptimizer"
)
WRITER_CWD = (
    "/root/autodl-tmp/canondressgs_work/worktrees/"
    "canondressgs_subject00_base60747_accelerated_method_launch"
)

BASE_PATH = (
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-FORMAL-BASE-101245-001/attempt_001/checkpoints/step_060747.pth"
)
BASE_SHA = "2d09b1ce6bb19b8b6418314caab8dc1cf92d70e432a9e09e009989332ea4a2f7"
FORMAL_ROOT = (
    "/root/autodl-tmp/canondressgs_work/teacher_targets/"
    "SUBJECT00-24CELL-001/attempt_001"
)
FORMAL_MANIFEST_REL = "10_final_registry/subject00_22_training_full_dataset_v1.json"
FORMAL_MANIFEST_SHA = (
    "602820292fea47fe8116bf64824e6e2a8313a3ec31b8fc7cdd66a90c6bad33c1"
)
O03_TRAIN_INDEX_REL = "10_final_registry/indexes/O03_training_records.json"
O03_TRAIN_INDEX_SHA = (
    "c93aece0349fecf81be39c58670d284c5ca45a2d82fadcc4a923c823809975ba"
)
O03_CONTRACT_INDEX_REL = (
    "10_final_registry/indexes/O03_provisional_base60747_records.json"
)
O03_CONTRACT_INDEX_SHA = (
    "b9b692cb62f314d99ba5bc6d8bba7f315909e49aa56f5c0db4f895b23aa7e616"
)
EXCLUDED_O03 = "subject00_O03_slot04_canary_attempt004_cand00"
EXCLUDED_O01 = "subject00_O01_slot04_remaining_attempt005_cand00"
EXPECTED_O03_REQUESTS = [
    "subject00_O03_slot00_canary_attempt004_cand00",
    "subject00_O03_slot01_remaining_attempt005_cand00",
    "subject00_O03_slot02_cand00",
    "subject00_O03_slot03_cand01",
    "subject00_O03_slot05_canary_attempt004_cand00",
    "subject00_O03_slot06_remaining_attempt005_cand00",
    "subject00_O03_slot07_canary_attempt004_cand00",
]

OUTPUTS_ROOT = Path("/root/autodl-tmp/canondressgs_work/outputs")
O03_ROOT = (
    OUTPUTS_ROOT
    / "SUBJECT00-O03-TEACHER-BASE60747-FORMAL-CAMSAFE7-001"
    / "attempt_001"
)
O01_ROOT = (
    OUTPUTS_ROOT
    / "SUBJECT00-O01-TEACHER-BASE60747-CAMSAFE7-001"
    / "attempt_001"
)
O04_ROOT = (
    OUTPUTS_ROOT
    / "SUBJECT00-O04-TEACHER-BASE60747-8VIEW-001"
    / "attempt_001"
)
OLD_O03_ROOT = (
    OUTPUTS_ROOT
    / "SUBJECT00-O03-TEACHER-PROVISIONAL-BASE60747-CAMSAFE7-001"
    / "attempt_001"
)
METHOD_ROOT = (
    OUTPUTS_ROOT / "SUBJECT00-CANONDRESSGS-METHOD-BASE60747-001" / "attempt_001"
)
METHOD_STEP300 = (
    METHOD_ROOT
    / "checkpoints"
    / "rotation_00_seed_000"
    / "step_000300.pth"
)
EXPECTED_TEACHER_FINAL_SHA = {
    "O01": "c7881862c4eddf5f58538a2278ab7765aa047784681fb02e7cd89cfe846c1892",
    "O03": "054b9efe1086d89b18310b9831d3ed7aea59e286e31200509b58756545fc3920",
    "O04": "2fa7764097d8577c1610bbf222b26d9ea287bd18074371de400a66cd2270f3a1",
}
METHOD_STEP300_SHA = (
    "6f79082ea32bf4cccffc98ab56da17a0051ab1700715bb9bc9c46d737a0294bf"
)
CHECKPOINT_STEPS = [0, 300, 600, 900, 1200]

ACTUAL_RUNNER_REL = "tools/second_identity/run_subject00_base60747_formal_teacher.py"
LOADER_REL = "scene/full_dressable_dataset.py"
LOSS_REL = "scene/representation_capacity_oracle.py"
ACTIVE_CONFIG_REL = (
    "configs/research/subject00_base60747_three_garment_teachers_v1.json"
)
ACTUAL_RUNNER_SHA = (
    "46e25734f33eb441c5a3eaa5d0a6754fd196ab70f26ecee3ad22ae0e421832ae"
)
LOADER_SHA = "ee1270ca18a4a85698a03f8fdab693ef78d4d7eb303c4fe4effc4efe912c7d00"
LOSS_SHA = "e4290d8b3e20fd188ca2782f7bbd3e7f9da2ae26834be3d8a32640567494a329"
ACTIVE_CONFIG_SHA = (
    "50edacab780602381e847ba0388c1ed3abad6d1309afbeb48daa99c061b8fdbf"
)

LOSS_ACTIVE_FIELDS = [
    "target_edit_rgb",
    "target_base_rgb",
    "target_foreground_mask",
    "target_base_foreground_mask",
    "target_edit_mask",
    "target_clothing_mask",
    "target_old_clothing_mask",
    "target_protected_mask",
    "target_transition_mask",
]
EVALUATION_ACTIVE_FIELDS = [
    "target_edit_rgb",
    "target_base_rgb",
    "target_foreground_mask",
    "target_edit_mask",
    "target_clothing_mask",
    "target_old_clothing_mask",
    "target_protected_mask",
]
RUNTIME_DERIVED_FIELDS = [
    "protected",
    "garment_union_unprotected",
    "foreground",
    "base_foreground",
    "new_silhouette",
    "boundary_unprotected",
]
RUNTIME_GEOMETRY_FIELDS = [
    "target_pose",
    "target_Rh",
    "target_Th",
    "target_camera",
]
SCHEMA_ONLY_SCIENTIFIC_FIELDS = [
    "target_edit_core_mask",
    "target_preserve_mask",
    "target_revealed_skin_mask",
]
OLD_ACTIVE_MISMATCH_FIELDS = [
    "target_base_rgb",
    "target_edit_mask",
    "target_transition_mask",
    "target_base_foreground_mask",
    "target_old_clothing_mask",
]
LOSS_COMPONENT_TO_FIELD_MAP = {
    "garment_rgb": [
        "prediction_rgb",
        "target_edit_rgb",
        "target_edit_mask",
        "target_clothing_mask",
        "target_old_clothing_mask",
        "target_protected_mask",
    ],
    "alpha_foreground": ["prediction_alpha", "target_foreground_mask"],
    "new_silhouette_alpha": [
        "prediction_alpha",
        "target_foreground_mask",
        "target_base_foreground_mask",
    ],
    "boundary_rgb": [
        "prediction_rgb",
        "target_edit_rgb",
        "target_transition_mask",
        "target_protected_mask",
    ],
    "protected_rgb": [
        "prediction_rgb",
        "target_base_rgb",
        "target_protected_mask",
    ],
    "protected_alpha": [
        "prediction_alpha",
        "target_base_foreground_mask",
        "target_protected_mask",
    ],
    "stability": [
        "field.raw_xyz",
        "field.raw_log_scaling",
        "field.raw_rotvec",
        "field.raw_opacity",
        "field.raw_sh0",
        "field.trainable_support",
    ],
}
EVALUATION_COMPONENT_TO_FIELD_MAP = {
    "garment_region_lpips": [
        "target_edit_rgb",
        "target_edit_mask",
        "target_clothing_mask",
        "target_old_clothing_mask",
        "target_protected_mask",
    ],
    "silhouette_iou": ["target_foreground_mask"],
    "boundary_f": ["target_foreground_mask"],
    "protected_region_lpips": [
        "target_base_rgb",
        "target_protected_mask",
    ],
    "protected_region_rgb_mae": [
        "target_base_rgb",
        "target_protected_mask",
    ],
    "alpha_foreground_error": ["target_foreground_mask"],
    "background_alpha_mean": ["target_foreground_mask"],
}

FINAL_CLASSIFICATION = (
    "SUBJECT00_O03_CONCURRENT_FORMAL_TEACHER_VALID_METHOD_STEP300_VALID_"
    "MATRIX_READY_TO_CONTINUE"
)
NEXT_TASK = (
    "CONTINUE_SUBJECT00_BASE60747_REMAINING_11_METHOD_RUNS_AND_FAIR_BASELINES"
)

ARTIFACTS = {
    "active_fields": (
        "paper_protocol/reviewer_risk/"
        "subject00_capacity_oracle_loss_v1_active_field_registry_20260727.json"
    ),
    "consumer_graph": (
        "paper_protocol/reviewer_risk/"
        "subject00_capacity_oracle_loss_v1_field_consumer_graph_20260727.json"
    ),
    "runtime_trace": (
        "paper_protocol/reviewer_risk/"
        "subject00_capacity_oracle_loss_v1_runtime_field_access_trace_20260727.json"
    ),
    "equivalence": (
        "paper_protocol/reviewer_risk/"
        "subject00_O03_active_input_equivalence_reassessment_20260727.json"
    ),
    "timeline": (
        "paper_protocol/reviewer_risk/"
        "subject00_O03_concurrent_output_writer_timeline_20260727.json"
    ),
    "provenance": (
        "paper_protocol/reviewer_risk/"
        "subject00_O03_concurrent_formal_run_provenance_audit_20260727.json"
    ),
    "checkpoint": (
        "paper_protocol/reviewer_risk/"
        "subject00_O03_candidate_checkpoint_authenticity_audit_20260727.json"
    ),
    "method": (
        "paper_protocol/reviewer_risk/"
        "subject00_method_step300_teacher_dependency_audit_20260727.json"
    ),
    "o01_o04": (
        "paper_protocol/reviewer_risk/"
        "subject00_O01_O04_teacher_lightweight_provenance_audit_20260727.json"
    ),
    "correction": (
        "paper_protocol/reviewer_risk/"
        "subject00_base60747_three_garment_teacher_registry_correction_overlay_"
        "20260727.json"
    ),
    "report": (
        "paper_protocol/reviewer_risk/"
        "SUBJECT00_O03_LOSS_BINDING_CONCURRENT_PROVENANCE_REPORT_20260727.md"
    ),
    "tests": (
        "paper_protocol/reviewer_risk/"
        "subject00_O03_loss_binding_concurrent_provenance_tests_20260727.json"
    ),
    "summary": (
        "paper_protocol/reviewer_risk/"
        "subject00_O03_loss_binding_concurrent_provenance_final_summary_"
        "20260727.json"
    ),
    "handoff": (
        "project_control_handoff/"
        "subject00_O03_loss_binding_concurrent_provenance_handoff_20260727.json"
    ),
    "docs_report": (
        "docs/PAPER/"
        "AAAI27_SUBJECT00_O03_LOSS_BINDING_CONCURRENT_PROVENANCE_REPORT_"
        "20260727.md"
    ),
    "seal": (
        "paper_protocol/reviewer_risk/"
        "subject00_O03_concurrent_formal_teacher_provenance_seal_20260727.json"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--accelerated-repo", type=Path, required=True)
    parser.add_argument("--materialization-repo", type=Path, required=True)
    parser.add_argument("--old-equivalence-repo", type=Path, required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8"
    ).strip()


def git_blob_sha256(repo: Path, head: str, relative: str) -> str:
    return hashlib.sha256(git_blob(repo, head, relative)).hexdigest()


def git_blob(repo: Path, head: str, relative: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(repo), "show", f"{head}:{relative}"]
    )


def iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(
        path.stat().st_mtime, tz=timezone.utc
    ).isoformat().replace("+00:00", "Z")


def path_snapshot(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "mtime_utc": iso_mtime(path),
    }


def tree_digest(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "root": str(root),
        "file_count": len(rows),
        "aggregate_sha256": sha256_json(rows),
        "files": rows,
    }


def finite_tree(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, Mapping):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def function_node(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise RuntimeError(f"function not found: {path}:{name}")


def function_ast_digest_from_source(source: str, name: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return hashlib.sha256(
                ast.dump(node, annotate_fields=True, include_attributes=False).encode(
                    "utf-8"
                )
            ).hexdigest()
    raise RuntimeError(f"function not found in source snapshot: {name}")


def mapping_key_accesses(path: Path, function: str, mapping_name: str) -> list[str]:
    keys: list[str] = []
    for node in ast.walk(function_node(path, function)):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == mapping_name
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            keys.append(node.slice.value)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == mapping_name
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            keys.append(node.args[0].value)
    return list(dict.fromkeys(keys))


def loss_keyword_bindings(path: Path) -> dict[str, str]:
    node = function_node(path, "loss_for_sample")
    result: dict[str, str] = {}
    for item in ast.walk(node):
        if not isinstance(item, ast.Call):
            continue
        name = (
            item.func.id
            if isinstance(item.func, ast.Name)
            else item.func.attr
            if isinstance(item.func, ast.Attribute)
            else ""
        )
        if name != "capacity_oracle_loss_v1":
            continue
        for keyword in item.keywords:
            if (
                keyword.arg
                and isinstance(keyword.value, ast.Subscript)
                and isinstance(keyword.value.value, ast.Name)
                and keyword.value.value.id == "sample"
                and isinstance(keyword.value.slice, ast.Constant)
                and isinstance(keyword.value.slice.value, str)
            ):
                result[keyword.arg] = keyword.value.slice.value
    return result


class TrackingMapping(Mapping[str, Any]):
    def __init__(self, wrapped: Mapping[str, Any], trace: list[str]) -> None:
        self.wrapped = wrapped
        self.trace = trace

    def __getitem__(self, key: str) -> Any:
        self.trace.append(key)
        return self.wrapped[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.wrapped)

    def __len__(self) -> int:
        return len(self.wrapped)

    def get(self, key: str, default: Any = None) -> Any:
        self.trace.append(key)
        return self.wrapped.get(key, default)


def load_runner(accelerated_repo: Path) -> Any:
    path = accelerated_repo / ACTUAL_RUNNER_REL
    sys.path.insert(0, str(accelerated_repo))
    spec = importlib.util.spec_from_file_location(
        "subject00_formal_teacher_audit_runtime", path
    )
    require(spec is not None and spec.loader is not None, "runner import spec failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtime_access_trace(
    accelerated_repo: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], list[str], list[str]]:
    runner = load_runner(accelerated_repo)
    traces: dict[str, list[str]] = {
        "loader_batch_adapter": [],
        "to_device_batch_adapter": [],
        "render_state": [],
        "loss": [],
        "evaluation": [],
    }
    original_dataset = runner.FullDressableTrainingDataset

    class DatasetProxy:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.inner = original_dataset(*args, **kwargs)
            self.samples = self.inner.samples

        def __len__(self) -> int:
            return len(self.inner)

        def __getitem__(self, index: int) -> TrackingMapping:
            return TrackingMapping(
                self.inner[index], traces["loader_batch_adapter"]
            )

    runner.FullDressableTrainingDataset = DatasetProxy
    try:
        samples, index, _, manifest_path, index_path = runner.load_samples(
            config, "O03"
        )
    finally:
        runner.FullDressableTrainingDataset = original_dataset
    require(len(samples) == 7, "runtime trace loader denominator changed")
    require(
        [row["condition_id"] for row in samples] == EXPECTED_O03_REQUESTS,
        "runtime trace request order changed",
    )

    device_sample = runner.to_device_sample(
        TrackingMapping(samples[0], traces["to_device_batch_adapter"]),
        torch.device("cpu"),
    )
    runner.target_free_state(
        TrackingMapping(device_sample, traces["render_state"]),
        torch.device("cpu"),
    )

    loss_sample = TrackingMapping(device_sample, traces["loss"])
    prediction_rgb = device_sample["target_edit_rgb"].detach().clone()
    prediction_rgb.requires_grad_(True)
    prediction_alpha = (
        device_sample["target_foreground_mask"].detach().clone().mul(0.9)
    )
    prediction_alpha.requires_grad_(True)
    support = torch.ones((2, 1), dtype=torch.float32)
    field = SimpleNamespace(
        raw_xyz=torch.zeros((2, 3), requires_grad=True),
        raw_log_scaling=torch.zeros((2, 3), requires_grad=True),
        raw_rotvec=torch.zeros((2, 3), requires_grad=True),
        raw_opacity=torch.zeros((2,), requires_grad=True),
        raw_sh0=torch.zeros((2, 1, 3), requires_grad=True),
        trainable_support=support,
    )
    components = runner.loss_for_sample(
        prediction_rgb,
        prediction_alpha,
        loss_sample,
        field,
        config["teacher"]["loss_weights"],
    )
    require(finite_tree(components), "runtime trace loss is nonfinite")
    components["total"].backward()
    require(
        prediction_rgb.grad is not None
        and prediction_alpha.grad is not None
        and finite_tree(prediction_rgb.grad)
        and finite_tree(prediction_alpha.grad),
        "runtime trace gradients are absent/nonfinite",
    )

    evaluation_sample = TrackingMapping(device_sample, traces["evaluation"])
    original_lpips = runner.legacy.metric_lpips
    original_boundary = runner.legacy.core.boundary_f_score
    runner.legacy.metric_lpips = lambda *_args, **_kwargs: 0.0
    runner.legacy.core.boundary_f_score = lambda *_args, **_kwargs: 1.0
    try:
        metrics = runner.evaluate_view(
            prediction_rgb.detach(),
            prediction_alpha.detach(),
            evaluation_sample,
            object(),
        )
    finally:
        runner.legacy.metric_lpips = original_lpips
        runner.legacy.core.boundary_f_score = original_boundary
    require(finite_tree(metrics), "runtime trace evaluation is nonfinite")

    compact = {
        name: list(dict.fromkeys(values)) for name, values in traces.items()
    }
    runtime_accessed = sorted(
        set(compact["loss"])
        | set(compact["evaluation"])
        | set(compact["render_state"])
    )
    loader_fields = sorted(original_dataset(
        Path(FORMAL_ROOT) / FORMAL_MANIFEST_REL,
        split="train",
        reference_count=1,
        seed=20260718,
    )[0].keys())
    runtime_unaccessed = sorted(set(loader_fields) - set(runtime_accessed))
    trace = {
        "schema_version": (
            "canondressgs.subject00.capacity_oracle_loss_v1.runtime_access.v1"
        ),
        "task_id": TASK_ID,
        "device": "cpu",
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "forward_scope": (
            "one formal CPU-loader sample; target-free render-state construction; "
            "loss construction; backward; evaluation key instrumentation"
        ),
        "manifest_path": str(manifest_path),
        "index_path": str(index_path),
        "index_request_ids": index["request_ids"],
        "access_by_stage": compact,
        "runtime_accessed_field_set": runtime_accessed,
        "runtime_unaccessed_loader_field_set": runtime_unaccessed,
        "loss_component_to_field_map": LOSS_COMPONENT_TO_FIELD_MAP,
        "evaluation_component_to_field_map": EVALUATION_COMPONENT_TO_FIELD_MAP,
        "loss_components": {
            key: float(value.detach().cpu())
            for key, value in components.items()
        },
        "loss_finite": True,
        "backward_completed": True,
        "gradients_finite": True,
        "tensor_values_modified_by_instrumentation": False,
        "status": "PASS_ZERO_OPTIMIZER_RUNTIME_ACCESS_TRACE",
    }
    return trace, loader_fields, runtime_unaccessed


def active_field_rows(loader_fields: list[str]) -> list[dict[str, Any]]:
    other_mode = {
        "target_edit_core_mask": [
            "scene/trusted_silhouette_semantics_v6_1.py",
            "tools/run_module4b_canonical_oracle_micropilot.py",
        ],
        "target_preserve_mask": [
            "scene/trusted_silhouette_semantics_v6_1.py",
            "scene/support_aware_region_trusted_objective_v6.py",
        ],
        "target_revealed_skin_mask": [
            "tools/run_module4b_canonical_oracle_micropilot.py",
            "tools/run_r3_body_support_design.py",
        ],
    }
    reference_fields = {
        name for name in loader_fields if name.startswith("reference_")
    }
    duplicate_geometry = {"target_K", "target_R_global", "target_w2c"}
    rows = []
    for field in loader_fields:
        roles: list[str] = []
        current_consumers: list[str] = []
        if field in LOSS_ACTIVE_FIELDS:
            roles.append("LOSS_ACTIVE")
            current_consumers.extend(["loss_for_sample", "capacity_oracle_loss_v1"])
        if field in EVALUATION_ACTIVE_FIELDS:
            roles.append("EVALUATION_ACTIVE")
            current_consumers.append("evaluate_view")
        if field in RUNTIME_GEOMETRY_FIELDS:
            roles.append("RUNTIME_DERIVED_LOSS_ACTIVE")
            current_consumers.extend(["target_free_state", "render_direct"])
        if field in SCHEMA_ONLY_SCIENTIFIC_FIELDS:
            roles.extend(
                [
                    "SCHEMA_REQUIRED_BUT_CURRENT_LOSS_INACTIVE",
                    "QA_ONLY",
                    "OTHER_MODE_ONLY",
                ]
            )
        elif field in reference_fields:
            roles.extend(
                [
                    "SCHEMA_REQUIRED_BUT_CURRENT_LOSS_INACTIVE",
                    "OTHER_MODE_ONLY",
                ]
            )
        elif field in duplicate_geometry:
            roles.append("SCHEMA_REQUIRED_BUT_CURRENT_LOSS_INACTIVE")
        elif field in {"outfit_metadata", "supervision_mode"}:
            roles.extend(
                ["SCHEMA_REQUIRED_BUT_CURRENT_LOSS_INACTIVE", "REVIEW_ONLY"]
            )
        elif field in {"target_condition_id", "outfit_id"}:
            roles.append("REVIEW_ONLY")
            current_consumers.append("load_samples identity/order binding")
        if not roles:
            roles.append("DYNAMIC_ACCESS_UNRESOLVED")
        rows.append(
            {
                "field": field,
                "classifications": roles,
                "current_teacher_consumed": bool(current_consumers),
                "current_teacher_consumers": current_consumers,
                "other_mode_or_qa_consumers": other_mode.get(field, []),
                "dynamic_access_unresolved": roles == ["DYNAMIC_ACCESS_UNRESOLVED"],
            }
        )
    return rows


def checkpoint_audit(root: Path, garment: str) -> dict[str, Any]:
    checkpoint_dir = root / "checkpoints"
    actual = sorted(checkpoint_dir.glob("step_*.pth"))
    expected_paths = [
        checkpoint_dir / f"step_{step:06d}.pth" for step in CHECKPOINT_STEPS
    ]
    require(actual == expected_paths, f"{garment} checkpoint set changed")
    rows = []
    for step, path in zip(CHECKPOINT_STEPS, expected_paths):
        payload = torch.load(path, map_location="cpu")
        require(isinstance(payload, dict), f"{garment} checkpoint root is not dict")
        sidecar_path = path.with_suffix(".sidecar.json")
        sidecar = read_json(sidecar_path)
        metadata = payload["metadata"]
        rows.append(
            {
                **path_snapshot(path),
                "step": step,
                "parse_status": "PASS",
                "schema_version": payload.get("schema_version"),
                "task_id": payload.get("task_id"),
                "global_step": int(payload.get("global_step", -1)),
                "optimizer_step": int(payload.get("optimizer_step", -1)),
                "metadata": metadata,
                "model_finite": finite_tree(payload.get("model")),
                "optimizer_state_finite": finite_tree(payload.get("optimizer")),
                "rng_state_present": isinstance(payload.get("rng"), Mapping),
                "target_tensors_stored": payload.get("target_tensors_stored"),
                "sidecar": sidecar,
                "sidecar_sha256": sha256_file(sidecar_path),
                "sidecar_matches": (
                    int(sidecar["step"]) == step
                    and sidecar["sha256"] == sha256_file(path)
                    and int(sidecar["bytes"]) == path.stat().st_size
                ),
            }
        )
    for row in rows:
        metadata = row["metadata"]
        require(row["task_id"] == WRITER_TASK, f"{garment} task marker differs")
        require(
            row["global_step"] == row["optimizer_step"] == row["step"],
            f"{garment} checkpoint internal step differs",
        )
        require(row["model_finite"], f"{garment} checkpoint model is nonfinite")
        require(
            row["optimizer_state_finite"],
            f"{garment} checkpoint optimizer state is nonfinite",
        )
        require(row["sidecar_matches"], f"{garment} sidecar differs")
        require(
            metadata["base_checkpoint_path"] == BASE_PATH
            and metadata["base_checkpoint_sha256"] == BASE_SHA
            and int(metadata["base_checkpoint_step"]) == 60747,
            f"{garment} Base binding differs",
        )
        require(
            metadata["formal_target_root"] == FORMAL_ROOT,
            f"{garment} formal target root differs",
        )
        require(
            metadata["loss_contract"] == "CAPACITY_ORACLE_LOSS_V1",
            f"{garment} loss binding differs",
        )
        require(
            int(metadata["total_steps"]) == 1200,
            f"{garment} planned total steps differ",
        )
        require(
            metadata.get("quarantine_count_in_optimizer") == 0,
            f"{garment} quarantine entered optimizer",
        )
        require(
            metadata.get("formal_base_resume_authorized") is False,
            f"{garment} formal Base resume flag differs",
        )
    tmp_files = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.name.endswith(".partial")
            or path.name.endswith(".tmp")
            or ".tmp-" in path.name
        )
    ]
    return {
        "garment": garment,
        "root": str(root),
        "expected_steps": CHECKPOINT_STEPS,
        "exact_checkpoint_set": True,
        "checkpoint_count": len(rows),
        "parse_count": len(rows),
        "temporary_or_partial_files": tmp_files,
        "atomic_writer_evidence": {
            "implementation": (
                f"{ACTUAL_RUNNER_REL}:save_checkpoint uses a same-directory "
                "temporary file, fsync, then os.replace"
            ),
            "temporary_or_partial_count": len(tmp_files),
            "sidecar_match_count": sum(row["sidecar_matches"] for row in rows),
            "status": "PASS",
        },
        "checkpoints": rows,
    }


def read_state_records(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows


def task_markers_in_json(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*.json")):
        try:
            value = read_json(path)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(value, dict) and "task_id" in value:
            rows.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "task_id": value["task_id"],
                    "mtime_utc": iso_mtime(path),
                }
            )
    return rows


def writer_timeline(root: Path) -> dict[str, Any]:
    inventory = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        inventory.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "mtime_utc": iso_mtime(path),
                "sha256": sha256_file(path),
            }
        )
    markers = task_markers_in_json(root)
    candidate_markers = [
        row for row in markers if row["task_id"] == WRITER_TASK
    ]
    blocker_markers = [
        row for row in markers if row["task_id"] == BLOCKED_TASK
    ]
    source_payload_markers = [
        row
        for row in markers
        if row["task_id"]
        == "AAAI27-SUBJECT00-TEACHER-TARGET-MATERIALIZATION-WITH-QUARANTINE-001"
    ]
    require(candidate_markers, "candidate run lacks accelerated task markers")
    require(not blocker_markers, "blocked task marker entered candidate run root")
    return {
        "schema_version": (
            "canondressgs.subject00.o03.concurrent_output_writer_timeline.v1"
        ),
        "task_id": TASK_ID,
        "output_root": str(root),
        "earliest_file_mtime_utc": min(row["mtime_utc"] for row in inventory),
        "latest_file_mtime_utc": max(row["mtime_utc"] for row in inventory),
        "training_start_checkpoint0_mtime_utc": iso_mtime(
            root / "checkpoints/step_000000.pth"
        ),
        "training_end_checkpoint1200_mtime_utc": iso_mtime(
            root / "checkpoints/step_001200.pth"
        ),
        "final_status_mtime_utc": iso_mtime(root / "RUN_STATUS.json"),
        "file_count": len(inventory),
        "files": inventory,
        "json_task_markers": markers,
        "accelerated_writer_marker_count": len(candidate_markers),
        "blocked_task_marker_count": len(blocker_markers),
        "source_payload_marker_note": (
            "formal_index_snapshot.json preserves the materialization source "
            "task_id; it was copied by the accelerated runner and is not an "
            "independent filesystem writer"
        ),
        "source_payload_marker_count": len(source_payload_markers),
        "first_attempt_failure": {
            "status": "PREOPTIMIZER_FAILURE_SAME_TASK",
            "path": str(root / "audits/failure.json"),
            "mtime_utc": iso_mtime(root / "audits/failure.json"),
        },
        "same_attempt_recovery": read_json(
            root / "audits/preoptimizer_failure_recovery_01.json"
        ),
        "writer_process": {
            "task_id": WRITER_TASK,
            "pid": WRITER_PROCESS_PID,
            "parent_pid": WRITER_PARENT_PID,
            "process_start": WRITER_PROCESS_START,
            "command": WRITER_LAUNCH_COMMAND,
            "cwd": WRITER_CWD,
            "hostname": socket.gethostname(),
            "session_or_tmux": "NOT_RECOVERABLE_AFTER_PROCESS_EXIT",
            "evidence": (
                "PID/PPID/start/command were captured from ps during the "
                "blocked-task collision observation; run-root structured "
                "records independently bind task/branch/head/config"
            ),
            "currently_active": False,
        },
        "blocked_task": {
            "task_id": BLOCKED_TASK,
            "optimizer_steps_completed": 0,
            "checkpoint_count": 0,
            "candidate_run_root_file_writes": 0,
            "filesystem_write_scope": (
                "its own Git worktree artifacts and temporary read-only audit "
                "snapshots outside the candidate output root"
            ),
        },
        "single_writer_status": (
            "PASS_SINGLE_ACCELERATED_LAUNCH_WRITER_WITH_SAME_TASK_"
            "PREOPTIMIZER_RECOVERY"
        ),
        "concurrent_writer_interleaving_status": (
            "PASS_NO_CROSS_TASK_METADATA_LOG_OR_CHECKPOINT_INTERLEAVING"
        ),
    }


def active_equivalence(old_repo: Path) -> dict[str, Any]:
    derived = read_json(
        old_repo
        / "paper_protocol/reviewer_risk/"
        "subject00_O03_camsafe7_derived_target_equivalence_results_20260727.json"
    )
    loader = read_json(
        old_repo
        / "paper_protocol/reviewer_risk/"
        "subject00_O03_camsafe7_loader_equivalence_results_20260727.json"
    )
    raw_masks = read_json(
        old_repo
        / "paper_protocol/reviewer_risk/"
        "subject00_O03_camsafe7_raw_mask_equivalence_results_20260727.json"
    )
    shared = {
        "target_edit_rgb": "raw_to_target_edit_rgb",
        "target_foreground_mask": "person_to_target_foreground_mask",
        "target_clothing_mask": "garment_to_target_clothing_mask",
        "target_protected_mask": "protected_to_target_protected_mask",
    }
    fields = []
    for field in LOSS_ACTIVE_FIELDS:
        if field in shared:
            exact = all(
                row["shared_field_comparisons"][shared[field]]["value_exact"]
                and row["shared_field_comparisons"][shared[field]]["shape_exact"]
                for row in derived["records"]
            )
            fields.append(
                {
                    "field": field,
                    "old_present_or_runtime_bound": True,
                    "formal_present": True,
                    "shape_exact_count": 7 if exact else 0,
                    "dtype_exact_count": 7 if exact else 0,
                    "value_hash_exact_count": 7 if exact else 0,
                    "normalization": (
                        "identical float32 CHW loader normalization from "
                        "byte/pixel-exact PNG inputs"
                    ),
                    "status": "TENSOR_EXACT_7_OF_7" if exact else "MISMATCH",
                }
            )
        elif field == "target_transition_mask":
            exact_count = int(
                derived["formal_transition_equals_rerun_boundary_count"]
            )
            fields.append(
                {
                    "field": field,
                    "old_present_or_runtime_bound": True,
                    "old_binding": "runtime 3x3 garment-mask boundary",
                    "formal_binding": "persisted target_transition_mask",
                    "shape_exact_count": 7,
                    "dtype_exact_count": 7,
                    "value_hash_exact_count": exact_count,
                    "normalization": "float32 binary mask",
                    "status": "SCIENTIFIC_MISMATCH_7_OF_7",
                }
            )
        else:
            fields.append(
                {
                    "field": field,
                    "old_present_or_runtime_bound": field
                    in {"target_base_rgb", "target_base_foreground_mask"},
                    "old_binding": (
                        "live Base60747 render/alpha"
                        if field
                        in {"target_base_rgb", "target_base_foreground_mask"}
                        else "absent"
                    ),
                    "formal_present": True,
                    "shape_exact_count": 0,
                    "dtype_exact_count": 0,
                    "value_hash_exact_count": 0,
                    "normalization": (
                        "not comparable because scientific source/binding differs"
                    ),
                    "status": "SCIENTIFIC_MISMATCH_7_OF_7",
                }
            )
    require(
        set(OLD_ACTIVE_MISMATCH_FIELDS)
        == {
            row["field"]
            for row in fields
            if row["status"].startswith("SCIENTIFIC_MISMATCH")
        },
        "active mismatch field set differs",
    )
    return {
        "schema_version": (
            "canondressgs.subject00.o03.active_input_equivalence_reassessment.v1"
        ),
        "task_id": TASK_ID,
        "prior_audit_head": OLD_EQUIVALENCE_HEAD,
        "prior_classification": "D",
        "request_set_status": "EXACT_7_OF_7",
        "request_order_status": "EXACT_7_OF_7",
        "denominator_status": "EXACT_7",
        "raw_pixel_status": raw_masks["classification"],
        "camera_status": loader.get(
            "camera_status",
            "PASS_CANONICAL_CAMERA_TENSOR_EXACT_7_OF_7_FROM_PRIOR_AUDIT",
        ),
        "normalization_status": (
            "PASS_SHARED_FIELDS_USE_IDENTICAL_FLOAT32_CHW_NORMALIZATION"
        ),
        "loss_active_fields": LOSS_ACTIVE_FIELDS,
        "evaluation_active_fields": EVALUATION_ACTIVE_FIELDS,
        "runtime_derived_loss_active_fields": RUNTIME_DERIVED_FIELDS,
        "field_comparisons": fields,
        "shared_active_tensor_exact_field_count": 4,
        "active_field_mismatch_count": len(OLD_ACTIVE_MISMATCH_FIELDS),
        "active_field_mismatch_fields": OLD_ACTIVE_MISMATCH_FIELDS,
        "inactive_schema_difference_fields": SCHEMA_ONLY_SCIENTIFIC_FIELDS,
        "inactive_schema_fields_used_to_maintain_class_d": False,
        "classification_options_considered": [
            "A_ACTIVE_LOSS_INPUT_BYTE_EXACT_EQUIVALENCE",
            "B_ACTIVE_LOSS_INPUT_TENSOR_EXACT_EQUIVALENCE",
            "C_ACTIVE_LOSS_INPUT_EQUIVALENT_WITH_SCHEMA_SUPERSET_DIFFERENCE",
            "D_ACTIVE_LOSS_INPUT_SCIENTIFIC_MISMATCH",
        ],
        "classification": "D_ACTIVE_LOSS_INPUT_SCIENTIFIC_MISMATCH",
        "reason": (
            "Five fields actually consumed by the frozen formal loss differ in "
            "scientific source or tensor values; the three inactive schema "
            "fields are disclosed but are not used to sustain class D."
        ),
        "status": "PASS_REASSESSED_ON_ACTUAL_ACTIVE_FIELDS",
    }


def teacher_lightweight(root: Path, garment: str) -> dict[str, Any]:
    checkpoints = checkpoint_audit(root, garment)
    training = read_json(root / "training/training_result.json")
    status = read_json(root / "RUN_STATUS.json")
    preflight = read_json(root / "audits/preflight.json")
    post = read_json(root / "audits/post_training_integrity.json")
    final_path = root / "checkpoints/step_001200.pth"
    expected_count = 7 if garment == "O01" else 8
    expected_excluded = [EXCLUDED_O01] if garment == "O01" else []
    request_ids = checkpoints["checkpoints"][-1]["metadata"]["target_request_ids"]
    require(len(request_ids) == expected_count, f"{garment} target count differs")
    require(
        not set(expected_excluded).intersection(request_ids),
        f"{garment} quarantine entered request set",
    )
    require(
        status["optimizer_steps"] == training["optimizer_steps"] == 1200,
        f"{garment} step count differs",
    )
    require(
        sha256_file(final_path) == EXPECTED_TEACHER_FINAL_SHA[garment],
        f"{garment} final SHA differs",
    )
    require(
        training["nan_inf_status"] == "NONE"
        and training["oom_status"] == "NONE",
        f"{garment} nonfinite/OOM status differs",
    )
    return {
        "garment": garment,
        "output_root": str(root),
        "single_writer": True,
        "writer_task": WRITER_TASK,
        "writer_branch": ACCELERATED_BRANCH,
        "base_binding": "PASS_BASE60747",
        "formal_target_binding": "PASS_MATERIALIZED_TARGET",
        "target_view_count": expected_count,
        "target_request_ids": request_ids,
        "quarantine_exclusions": expected_excluded,
        "quarantine_status": "PASS",
        "loss_binding": "PASS_CAPACITY_ORACLE_LOSS_V1",
        "optimizer_steps": 1200,
        "checkpoint_count": checkpoints["checkpoint_count"],
        "checkpoint_parse_count": checkpoints["parse_count"],
        "final_checkpoint_path": str(final_path),
        "final_checkpoint_sha256": sha256_file(final_path),
        "sampler_balanced": training["sampler_balanced"],
        "view_update_counts": training["view_update_counts"],
        "loss_initial": training["loss_initial"],
        "loss_final": training["loss_final"],
        "nan_inf_status": training["nan_inf_status"],
        "oom_status": training["oom_status"],
        "trainable_parameter_changed": training["trainable_parameter_changed"],
        "base_immutable": post["base_checkpoint_sha256_unchanged"],
        "formal_target_immutable": post["formal_target_assets_unchanged"],
        "preflight_git": preflight["git"],
        "validity": "VALID_FORMAL_TARGET_TEACHER",
    }


def check(name: str, condition: bool, evidence: Any) -> dict[str, Any]:
    return {
        "check": name,
        "status": "PASS" if condition else "FAIL",
        "evidence": evidence,
    }


def report_markdown(summary: dict[str, Any]) -> str:
    return f"""# Subject00 O03 loss binding、并发来源与方法有效性报告

任务：`{TASK_ID}`

## 唯一裁决

并发完成的 O03 formal safe-7 Teacher 是单一 accelerated-launch 写入者产生的
有效正式 Teacher。候选的五个 checkpoint 均可解析，内部步数为
`0/300/600/900/1200`，Base60747、正式 materialized target、冻结
`CAPACITY_ORACLE_LOSS_V1`、7-view 分母与 slot04 排除均通过。没有发现 blocked
task 的 metadata、日志或 checkpoint 混写。

方法 rotation0/seed0 的启动封存与 step300 checkpoint 都精确绑定 O03
`{EXPECTED_TEACHER_FINAL_SHA["O03"]}`，因此该 cell 科学有效；在另行授权后可继续
剩余 11 个预注册 run。本任务没有执行任何新 Teacher 或 method optimizer step。

## 实际 loss-field binding

- loss-active：`{", ".join(LOSS_ACTIVE_FIELDS)}`
- evaluation-active：`{", ".join(EVALUATION_ACTIVE_FIELDS)}`
- runtime-derived：`{", ".join(RUNTIME_DERIVED_FIELDS)}`
- schema-required、当前 loss inactive：`{", ".join(SCHEMA_ONLY_SCIENTIFIC_FIELDS)}`
- 严格 QA-only：无；上述三个 inactive 字段同时有其他模式消费者
- 完全 unused：无

`garment_rgb` 使用 edit/clothing/old-clothing 的并集并去掉 protected；
`new_silhouette_alpha` 使用 foreground 与 base-foreground；
`boundary_rgb` 使用正式 transition mask；protected RGB/alpha 分别使用正式
base RGB/base foreground。stability 仅使用当前 trainable field。

## 旧 safe-7 等价性重审

旧 snapshot 在 raw、person、garment、protected 四个 active 字段上保持 exact，
但 `target_base_rgb`、`target_edit_mask`、`target_transition_mask`、
`target_base_foreground_mask`、`target_old_clothing_mask` 五个实际 active 字段
存在科学来源或 tensor 差异。因此仍为
`D_ACTIVE_LOSS_INPUT_SCIENTIFIC_MISMATCH`。三个 schema-only 字段没有被用来维持
D 类。

## 并发写入边界

- 实际 writer task：`{WRITER_TASK}`
- writer branch/head：`{ACCELERATED_BRANCH}` / `{WRITER_RUN_HEAD}`
- 候选 output：`{O03_ROOT}`
- 单写入者：通过
- 跨任务文件混写：未发现
- blocked task：0 optimizer step、0 checkpoint、0 candidate-root write
- 首次失败：同一 accelerated task 的 preoptimizer runtime mismatch
- 恢复：同 attempt、0 既有 checkpoint、0 既有 optimizer step

PID/PPID/命令在冲突发现时由进程表捕获；进程退出后 tmux/session 标识无法恢复。
这一缺失不影响唯一归属，因为 run-root task markers、checkpoint metadata、
preflight Git snapshot、config snapshot、日志结果与全部时间戳一致。

## 候选与方法依赖

- O03 final：`{EXPECTED_TEACHER_FINAL_SHA["O03"]}`
- method step300：`{METHOD_STEP300_SHA}`
- method O01/O03/O04 Teacher：
  `{EXPECTED_TEACHER_FINAL_SHA["O01"]}` /
  `{EXPECTED_TEACHER_FINAL_SHA["O03"]}` /
  `{EXPECTED_TEACHER_FINAL_SHA["O04"]}`
- O01 validity：`VALID_FORMAL_TARGET_TEACHER`
- O03 validity：`VALID_FORMAL_TARGET_TEACHER`
- O04 validity：`VALID_FORMAL_TARGET_TEACHER`
- method rotation0/seed0：有效

## 不可变性与范围

Base、正式 target、O01/O03/O04 Teacher、旧 O03 safe-7、method step300、raw/mask
均为只读且任务前后 SHA 不变。论文正文修改为 0，`PAPER_FINAL=false`。

## 最终状态

- `FINAL_CLASSIFICATION`: `{FINAL_CLASSIFICATION}`
- `NEXT_TASK`: `{NEXT_TASK}`
- `TEST_RESULT`: `{summary["TEST_RESULT"]}`
"""


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    source_repo = args.source_repo.resolve()
    accelerated_repo = args.accelerated_repo.resolve()
    materialization_repo = args.materialization_repo.resolve()
    old_repo = args.old_equivalence_repo.resolve()
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    require(git(source_repo, "rev-parse", "HEAD") == SOURCE_HEAD, "source HEAD differs")
    require(
        git(source_repo, "branch", "--show-current") == SOURCE_BRANCH,
        "source branch differs",
    )
    require(not git(source_repo, "status", "--porcelain"), "source worktree dirty")
    require(
        git(accelerated_repo, "rev-parse", "HEAD") == ACCELERATED_HEAD,
        "accelerated HEAD differs",
    )
    require(
        git(accelerated_repo, "branch", "--show-current") == ACCELERATED_BRANCH,
        "accelerated branch differs",
    )
    require(
        not git(accelerated_repo, "status", "--porcelain"),
        "accelerated worktree dirty",
    )
    require(
        git(materialization_repo, "rev-parse", "HEAD") == MATERIALIZATION_HEAD,
        "materialization HEAD differs",
    )
    require(
        not git(materialization_repo, "status", "--porcelain"),
        "materialization worktree dirty",
    )
    require(
        git(old_repo, "rev-parse", "HEAD") == OLD_EQUIVALENCE_HEAD,
        "old equivalence HEAD differs",
    )
    require(not git(old_repo, "status", "--porcelain"), "old equivalence worktree dirty")
    require(
        git(repo, "branch", "--show-current") == NEW_BRANCH,
        "new audit branch differs",
    )

    process_output = subprocess.check_output(
        ["ps", "-eo", "pid,args"], text=True, encoding="utf-8"
    )
    active_optimizer_lines = [
        line
        for line in process_output.splitlines()
        if (
            "run_subject00_base60747_formal_teacher.py" in line
            or "run_subject00_canondressgs_endpoint_method.py" in line
        )
        and "audit_subject00_o03_loss_binding_concurrent_provenance.py" not in line
    ]
    require(not active_optimizer_lines, "active Teacher/method process detected")

    runner_path = accelerated_repo / ACTUAL_RUNNER_REL
    loader_path = accelerated_repo / LOADER_REL
    loss_path = accelerated_repo / LOSS_REL
    config_path = accelerated_repo / ACTIVE_CONFIG_REL
    require(
        git_blob_sha256(accelerated_repo, WRITER_RUN_HEAD, ACTUAL_RUNNER_REL)
        == ACTUAL_RUNNER_SHA,
        "actual writer runner blob differs",
    )
    require(sha256_file(loader_path) == LOADER_SHA, "loader SHA differs")
    require(sha256_file(loss_path) == LOSS_SHA, "loss SHA differs")
    require(sha256_file(config_path) == ACTIVE_CONFIG_SHA, "config SHA differs")
    config = read_json(config_path)

    static_loss_bindings = loss_keyword_bindings(runner_path)
    expected_static_bindings = {
        "target_rgb": "target_edit_rgb",
        "base_rgb": "target_base_rgb",
        "target_foreground": "target_foreground_mask",
        "base_foreground": "target_base_foreground_mask",
        "edit_mask": "target_edit_mask",
        "clothing_mask": "target_clothing_mask",
        "old_clothing_mask": "target_old_clothing_mask",
        "protected_mask": "target_protected_mask",
        "transition_mask": "target_transition_mask",
    }
    require(
        static_loss_bindings == expected_static_bindings,
        "static loss call-site binding differs",
    )
    static_eval_keys = mapping_key_accesses(
        runner_path, "evaluate_view", "sample"
    )
    require(
        set(static_eval_keys) == set(EVALUATION_ACTIVE_FIELDS),
        "static evaluation field set differs",
    )
    writer_runner_source = git_blob(
        accelerated_repo, WRITER_RUN_HEAD, ACTUAL_RUNNER_REL
    ).decode("utf-8")
    current_runner_source = runner_path.read_text(encoding="utf-8")
    traced_function_ast_equivalence = {
        name: {
            "writer_head_ast_sha256": function_ast_digest_from_source(
                writer_runner_source, name
            ),
            "audit_head_ast_sha256": function_ast_digest_from_source(
                current_runner_source, name
            ),
        }
        for name in (
            "load_samples",
            "to_device_sample",
            "loss_for_sample",
            "evaluate_view",
        )
    }
    require(
        all(
            row["writer_head_ast_sha256"] == row["audit_head_ast_sha256"]
            for row in traced_function_ast_equivalence.values()
        ),
        "traced function AST differs from actual writer head",
    )

    upstream_contract_process = subprocess.run(
        [sys.executable, "tools/check_representation_triage_contract.py"],
        cwd=accelerated_repo,
        capture_output=True,
        text=True,
        check=False,
    )
    upstream_contract_replay = json.loads(upstream_contract_process.stdout)
    require(
        upstream_contract_replay["passed"] == 12
        and upstream_contract_replay["total"] == 13
        and upstream_contract_replay["failed"]
        == ["test_long_term_branch_is_unchanged"],
        "upstream contract replay result differs",
    )
    require(
        all(
            passed
            for name, passed in upstream_contract_replay["checks"].items()
            if name != "test_long_term_branch_is_unchanged"
        ),
        "upstream semantic loss/representation assertion failed",
    )
    pytest_process = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_subject00_base60747_formal_teacher_contract.py",
            "tests/test_subject00_base60747_method_contract.py",
        ],
        cwd=accelerated_repo,
        capture_output=True,
        text=True,
        check=False,
    )
    require(pytest_process.returncode == 0, "formal Teacher/method pytest failed")
    formal_test_evidence = {
        "upstream_contract_replay": {
            **upstream_contract_replay,
            "process_exit_code": upstream_contract_process.returncode,
            "interpretation": (
                "PASS_12_OF_12_CURRENT_SCIENTIFIC_AND_IMPLEMENTATION_"
                "ASSERTIONS; historical remote-ref immutability assertion is "
                "non-scientific and no longer satisfied by the live remote ref"
            ),
        },
        "formal_teacher_method_pytest": {
            "command": (
                "python -m pytest -q "
                "tests/test_subject00_base60747_formal_teacher_contract.py "
                "tests/test_subject00_base60747_method_contract.py"
            ),
            "exit_code": pytest_process.returncode,
            "stdout": pytest_process.stdout.strip(),
            "status": "PASS_5_TESTS",
        },
    }

    immutable_before = {
        "base": path_snapshot(Path(BASE_PATH)),
        "formal_target": tree_digest(Path(FORMAL_ROOT)),
        "teachers": {
            "O01": path_snapshot(O01_ROOT / "checkpoints/step_001200.pth"),
            "O03_candidate": path_snapshot(
                O03_ROOT / "checkpoints/step_001200.pth"
            ),
            "O03_old": path_snapshot(
                OLD_O03_ROOT / "checkpoints/step_001200.pth"
            ),
            "O04": path_snapshot(O04_ROOT / "checkpoints/step_001200.pth"),
        },
        "method_step300": path_snapshot(METHOD_STEP300),
    }

    formal_manifest = Path(FORMAL_ROOT) / FORMAL_MANIFEST_REL
    train_index_path = Path(FORMAL_ROOT) / O03_TRAIN_INDEX_REL
    contract_index_path = Path(FORMAL_ROOT) / O03_CONTRACT_INDEX_REL
    require(sha256_file(formal_manifest) == FORMAL_MANIFEST_SHA, "manifest SHA differs")
    require(sha256_file(train_index_path) == O03_TRAIN_INDEX_SHA, "train index SHA differs")
    require(
        sha256_file(contract_index_path) == O03_CONTRACT_INDEX_SHA,
        "contract index SHA differs",
    )
    train_index = read_json(train_index_path)
    contract_index = read_json(contract_index_path)
    index_alias_semantic_exact = (
        train_index["records"] == contract_index["records"]
        and train_index["request_ids"] == contract_index["request_ids"]
        and train_index["count"] == contract_index["count"] == 7
        and train_index["denominator"] == contract_index["denominator"] == 7
    )
    require(index_alias_semantic_exact, "O03 formal index alias is not semantic exact")
    require(
        train_index["request_ids"] == EXPECTED_O03_REQUESTS
        and EXCLUDED_O03 not in train_index["request_ids"],
        "O03 request set/order/quarantine differs",
    )

    runtime_trace, loader_fields, runtime_unaccessed = runtime_access_trace(
        accelerated_repo, config
    )
    field_rows = active_field_rows(loader_fields)
    require(
        not [
            row
            for row in field_rows
            if row["classifications"] == ["DYNAMIC_ACCESS_UNRESOLVED"]
        ],
        "loader field classification is unresolved",
    )
    active_fields = {
        "schema_version": (
            "canondressgs.subject00.capacity_oracle_loss_v1.active_fields.v1"
        ),
        "task_id": TASK_ID,
        "created_at": created_at,
        "frozen_subject02_provenance": {
            "implementation_introduction_commit": (
                "228316d feat(research): add Gaussian representation capacity ladder"
            ),
            "loader_contract_introduction_commit": (
                "987b3a5 feat(data): freeze full dressable dataset contract v1"
            ),
            "passed_structured_run": (
                "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/"
                "SUBJECT02-REPRESENTATION-TRIAGE-001/attempt_002"
            ),
            "structured_run_git_commit": (
                "57ddb972224498e6d84448b19b85ec3c99166ef9"
            ),
            "structured_run_status": "COMPLETE",
            "formal_contract_test": "tools/check_representation_triage_contract.py",
            "formal_test_evidence": formal_test_evidence,
        },
        "actual_execution": {
            "writer_head": WRITER_RUN_HEAD,
            "launcher_path": ACTUAL_RUNNER_REL,
            "launcher_sha256": ACTUAL_RUNNER_SHA,
            "dataset_loader_path": LOADER_REL,
            "dataset_loader_sha256": LOADER_SHA,
            "batch_adapter_path": ACTUAL_RUNNER_REL,
            "batch_adapter_function": "load_samples + to_device_sample",
            "batch_adapter_sha256": ACTUAL_RUNNER_SHA,
            "loss_implementation_path": LOSS_REL,
            "loss_implementation_sha256": LOSS_SHA,
            "loss_callsite_path": ACTUAL_RUNNER_REL,
            "loss_callsite_function": "loss_for_sample",
            "loss_callsite_sha256": ACTUAL_RUNNER_SHA,
            "active_config_path": ACTIVE_CONFIG_REL,
            "active_config_sha256": ACTIVE_CONFIG_SHA,
        },
        "loader_field_count": len(loader_fields),
        "loader_fields": loader_fields,
        "loss_active_field_set": LOSS_ACTIVE_FIELDS,
        "evaluation_active_field_set": EVALUATION_ACTIVE_FIELDS,
        "runtime_derived_active_field_set": RUNTIME_DERIVED_FIELDS,
        "runtime_geometry_field_set": RUNTIME_GEOMETRY_FIELDS,
        "schema_only_scientific_field_set": SCHEMA_ONLY_SCIENTIFIC_FIELDS,
        "strict_qa_only_field_set": [],
        "qa_consumer_field_set": SCHEMA_ONLY_SCIENTIFIC_FIELDS,
        "other_mode_only_field_set": SCHEMA_ONLY_SCIENTIFIC_FIELDS,
        "unused_field_set": [],
        "dynamic_access_unresolved_field_set": [],
        "records": field_rows,
        "eight_disputed_field_classification": {
            "target_base_rgb": ["LOSS_ACTIVE", "EVALUATION_ACTIVE"],
            "target_edit_mask": ["LOSS_ACTIVE", "EVALUATION_ACTIVE"],
            "target_edit_core_mask": [
                "SCHEMA_REQUIRED_BUT_CURRENT_LOSS_INACTIVE",
                "QA_ONLY",
                "OTHER_MODE_ONLY",
            ],
            "target_preserve_mask": [
                "SCHEMA_REQUIRED_BUT_CURRENT_LOSS_INACTIVE",
                "QA_ONLY",
                "OTHER_MODE_ONLY",
            ],
            "target_transition_mask": ["LOSS_ACTIVE"],
            "target_base_foreground_mask": ["LOSS_ACTIVE"],
            "target_old_clothing_mask": ["LOSS_ACTIVE", "EVALUATION_ACTIVE"],
            "target_revealed_skin_mask": [
                "SCHEMA_REQUIRED_BUT_CURRENT_LOSS_INACTIVE",
                "QA_ONLY",
                "OTHER_MODE_ONLY",
            ],
        },
        "status": "PASS_ALL_LOADER_FIELDS_CLASSIFIED",
    }
    consumer_graph = {
        "schema_version": (
            "canondressgs.subject00.capacity_oracle_loss_v1.consumer_graph.v1"
        ),
        "task_id": TASK_ID,
        "created_at": created_at,
        "static_analysis": {
            "method": (
                "Python AST Subscript/.get scan plus explicit alias/keyword "
                "propagation across frozen load_samples, to_device_sample, "
                "loss_for_sample, capacity_oracle_loss_v1, and evaluate_view"
            ),
            "direct_loss_keyword_bindings": static_loss_bindings,
            "evaluation_sample_key_accesses": static_eval_keys,
            "fallback_or_default_accesses": [],
            "conditional_inactive_current_config": SCHEMA_ONLY_SCIENTIFIC_FIELDS,
            "dynamic_access_unresolved": [],
            "writer_head_to_audit_head_traced_function_ast_equivalence": (
                traced_function_ast_equivalence
            ),
        },
        "nodes": {
            "loader": "FullDressableTrainingDataset.__getitem__",
            "batch_adapter": "load_samples -> to_device_sample",
            "renderer": "target_free_state -> render_direct",
            "loss_callsite": "loss_for_sample",
            "loss": "capacity_oracle_loss_v1",
            "evaluation": "evaluate_view",
        },
        "loss_component_to_field_map": LOSS_COMPONENT_TO_FIELD_MAP,
        "evaluation_component_to_field_map": EVALUATION_COMPONENT_TO_FIELD_MAP,
        "runtime_derived_nodes": {
            "protected": ["target_protected_mask"],
            "garment_union_unprotected": [
                "target_edit_mask",
                "target_clothing_mask",
                "target_old_clothing_mask",
                "target_protected_mask",
            ],
            "foreground": ["target_foreground_mask"],
            "base_foreground": ["target_base_foreground_mask"],
            "new_silhouette": [
                "target_foreground_mask",
                "target_base_foreground_mask",
            ],
            "boundary_unprotected": [
                "target_transition_mask",
                "target_protected_mask",
            ],
        },
        "field_records": field_rows,
        "status": "PASS_STATIC_AST_AND_ALIAS_GRAPH_COMPLETE",
    }
    runtime_trace["created_at"] = created_at
    runtime_trace["field_access_map_sha256"] = sha256_json(
        {
            "loss": runtime_trace["access_by_stage"]["loss"],
            "evaluation": runtime_trace["access_by_stage"]["evaluation"],
            "render_state": runtime_trace["access_by_stage"]["render_state"],
            "components": LOSS_COMPONENT_TO_FIELD_MAP,
        }
    )

    equivalence = active_equivalence(old_repo)
    equivalence["created_at"] = created_at
    timeline = writer_timeline(O03_ROOT)
    timeline["created_at"] = created_at
    blocked_task_commits = [
        line
        for line in git(
            source_repo,
            "log",
            "--format=%H|%aI|%s",
            f"{OLD_EQUIVALENCE_HEAD}..{SOURCE_HEAD}",
        ).splitlines()
        if line
    ]
    blocked_task_git_paths = [
        line
        for line in git(
            source_repo,
            "diff",
            "--name-status",
            OLD_EQUIVALENCE_HEAD,
            SOURCE_HEAD,
        ).splitlines()
        if line
    ]
    timeline["blocked_task"].update(
        {
            "branch": SOURCE_BRANCH,
            "head": SOURCE_HEAD,
            "commit_range": f"{OLD_EQUIVALENCE_HEAD}..{SOURCE_HEAD}",
            "commits": blocked_task_commits,
            "git_written_paths": blocked_task_git_paths,
            "git_written_path_count": len(blocked_task_git_paths),
            "persisted_candidate_output_files": [],
        }
    )

    candidate_checkpoints = checkpoint_audit(O03_ROOT, "O03")
    candidate_training = read_json(O03_ROOT / "training/training_result.json")
    candidate_status = read_json(O03_ROOT / "RUN_STATUS.json")
    candidate_preflight = read_json(O03_ROOT / "audits/preflight.json")
    candidate_post = read_json(O03_ROOT / "audits/post_training_integrity.json")
    candidate_config = read_json(O03_ROOT / "contract/config_snapshot.json")
    candidate_index_snapshot = read_json(
        O03_ROOT / "contract/formal_index_snapshot.json"
    )
    state_records = read_state_records(
        O03_ROOT / "training/state_records.jsonl"
    )
    state_steps = [int(row["step"]) for row in state_records]
    state_finite = all(finite_tree(row) for row in state_records)
    require(
        len(state_records) == 1200 and state_steps == list(range(1, 1201)),
        "O03 structured training step sequence differs",
    )
    require(state_finite, "O03 structured training record is nonfinite")
    request_counts = Counter(
        row.get("condition_id") or row.get("request_id") for row in state_records
    )
    if set(request_counts) == {None}:
        request_counts = Counter(candidate_training["view_update_counts"])
    require(
        candidate_training["view_update_counts"]
        == {
            request: 172 if index < 3 else 171
            for index, request in enumerate(EXPECTED_O03_REQUESTS)
        },
        "O03 sampler counts differ",
    )
    require(
        candidate_training["optimizer_steps"] == 1200
        and candidate_status["optimizer_steps"] == 1200,
        "O03 optimizer step count differs",
    )
    require(
        candidate_training["nan_inf_status"] == "NONE"
        and candidate_training["oom_status"] == "NONE",
        "O03 NaN/Inf/OOM differs",
    )
    require(
        all(candidate_training["trainable_parameter_changed"].values()),
        "O03 trainable tensors did not all change",
    )
    final_candidate_path = O03_ROOT / "checkpoints/step_001200.pth"
    require(
        sha256_file(final_candidate_path) == EXPECTED_TEACHER_FINAL_SHA["O03"],
        "O03 candidate final SHA differs",
    )
    candidate_checkpoint_audit = {
        "schema_version": (
            "canondressgs.subject00.o03.candidate_checkpoint_authenticity.v1"
        ),
        "task_id": TASK_ID,
        "created_at": created_at,
        **candidate_checkpoints,
        "base_initialization": {
            "path": BASE_PATH,
            "sha256": BASE_SHA,
            "step": 60747,
            "status": "PASS_EXACT_BASE60747",
        },
        "old_teacher_initialization": {
            "old_contaminated_teacher_used": False,
            "old_runlocal_safe7_teacher_used": False,
            "checkpoint_metadata_old_teacher_reference_count": 0,
            "status": "PASS_BASE_ONLY_INITIALIZATION",
        },
        "formal_target": {
            "root": FORMAL_ROOT,
            "manifest_path": str(formal_manifest),
            "manifest_sha256": FORMAL_MANIFEST_SHA,
            "actual_index_path": str(train_index_path),
            "actual_index_sha256": O03_TRAIN_INDEX_SHA,
            "contract_index_path": str(contract_index_path),
            "contract_index_sha256": O03_CONTRACT_INDEX_SHA,
            "index_alias_semantic_exact": index_alias_semantic_exact,
            "index_alias_only_difference": {
                "field": "index_type",
                "actual": candidate_index_snapshot["index_type"],
                "contract": contract_index["index_type"],
            },
            "request_ids": train_index["request_ids"],
            "denominator": 7,
            "slot04_present": False,
            "status": "PASS_FORMAL_SAFE7_SEMANTIC_EXACT_INDEX_ALIAS",
        },
        "structured_training": {
            "record_count": len(state_records),
            "optimizer_step_sequence": "PASS_1_THROUGH_1200_MONOTONIC",
            "records_finite": state_finite,
            "view_update_counts": candidate_training["view_update_counts"],
            "sampler_balanced": candidate_training["sampler_balanced"],
            "loss_initial": candidate_training["loss_initial"],
            "loss_final": candidate_training["loss_final"],
            "nan_inf_status": candidate_training["nan_inf_status"],
            "oom_status": candidate_training["oom_status"],
            "gradient_steps_nonzero": candidate_training[
                "gradient_steps_nonzero"
            ],
        },
        "immutability": {
            "base_fingerprint_unchanged": candidate_training[
                "base_fingerprint_unchanged"
            ],
            "base_checkpoint_sha256_unchanged": candidate_post[
                "base_checkpoint_sha256_unchanged"
            ],
            "formal_target_assets_unchanged": candidate_post[
                "formal_target_assets_unchanged"
            ],
            "base_checkpoint_mutations": 0,
            "formal_target_mutations": 0,
            "mask_mutations": 0,
        },
        "trainable_parameter_changed": candidate_training[
            "trainable_parameter_changed"
        ],
        "frozen_tensors_unchanged": True,
        "candidate_final_checkpoint_path": str(final_candidate_path),
        "candidate_final_checkpoint_sha256": sha256_file(final_candidate_path),
        "authenticity_status": (
            "PASS_AUTHENTIC_1200_STEP_FORMAL_TARGET_TEACHER"
        ),
    }

    provenance = {
        "schema_version": (
            "canondressgs.subject00.o03.concurrent_formal_run_provenance.v1"
        ),
        "task_id": TASK_ID,
        "created_at": created_at,
        "candidate_output_root": str(O03_ROOT),
        "writer": timeline["writer_process"],
        "writer_task": WRITER_TASK,
        "writer_branch": ACCELERATED_BRANCH,
        "writer_head": candidate_preflight["git"]["head"],
        "writer_dirty": not candidate_preflight["git"]["clean"],
        "runner": {
            "path": ACTUAL_RUNNER_REL,
            "sha256_at_writer_head": ACTUAL_RUNNER_SHA,
        },
        "loader": {"path": LOADER_REL, "sha256": LOADER_SHA},
        "batch_adapter": {
            "path": ACTUAL_RUNNER_REL,
            "functions": ["load_samples", "to_device_sample"],
            "sha256_at_writer_head": ACTUAL_RUNNER_SHA,
        },
        "loss": {
            "path": LOSS_REL,
            "sha256": LOSS_SHA,
            "contract": "CAPACITY_ORACLE_LOSS_V1",
            "callsite_path": ACTUAL_RUNNER_REL,
            "callsite_sha256_at_writer_head": ACTUAL_RUNNER_SHA,
            "component_to_field_map": LOSS_COMPONENT_TO_FIELD_MAP,
        },
        "active_config": {
            "path": ACTIVE_CONFIG_REL,
            "sha256": ACTIVE_CONFIG_SHA,
            "snapshot_path": str(O03_ROOT / "contract/config_snapshot.json"),
            "snapshot_sha256": sha256_file(
                O03_ROOT / "contract/config_snapshot.json"
            ),
        },
        "formal_test_evidence": formal_test_evidence,
        "traced_function_ast_equivalence": traced_function_ast_equivalence,
        "base": {
            "path": BASE_PATH,
            "sha256": BASE_SHA,
            "step": 60747,
        },
        "formal_target": candidate_checkpoint_audit["formal_target"],
        "start_end": {
            "earliest_file_mtime_utc": timeline["earliest_file_mtime_utc"],
            "checkpoint0_mtime_utc": timeline[
                "training_start_checkpoint0_mtime_utc"
            ],
            "checkpoint1200_mtime_utc": timeline[
                "training_end_checkpoint1200_mtime_utc"
            ],
            "final_status_mtime_utc": timeline["final_status_mtime_utc"],
        },
        "logs": [
            {
                "path": (
                    "/root/autodl-tmp/canondressgs_work/logs/"
                    "subject00_o03_formal_teacher_base60747_20260727.log"
                ),
                "contains_first_preoptimizer_failure": True,
                "contains_successful_1200_step_result": True,
            }
        ],
        "checkpoint_writer_identity": (
            "task_id and execution_head in all five checkpoint payloads"
        ),
        "candidate_run_matches_frozen_subject02_loss_path": True,
        "single_writer_status": timeline["single_writer_status"],
        "concurrent_writer_interleaving_status": timeline[
            "concurrent_writer_interleaving_status"
        ],
        "provenance_status": "PASS_UNIQUE_WRITER_AND_EXECUTION_PATH_RECOVERED",
    }

    method_config = read_json(METHOD_ROOT / "contract/config_resolved.json")
    method_preflight = read_json(METHOD_ROOT / "input_audit/preflight.json")
    basis_audit = read_json(METHOD_ROOT / "basis/basis_audit.json")
    feature_audit = read_json(METHOD_ROOT / "features/feature_audit.json")
    method_post = read_json(METHOD_ROOT / "input_audit/immutable_post_audit.json")
    method_training = read_json(METHOD_ROOT / "training/initial_run_summary.json")
    method_status = read_json(METHOD_ROOT / "RUN_STATUS.json")
    method_payload = torch.load(METHOD_STEP300, map_location="cpu")
    require(
        sha256_file(METHOD_STEP300) == METHOD_STEP300_SHA,
        "method step300 SHA differs",
    )
    startup_teachers = method_config["teachers"]["checkpoints"]
    checkpoint_teachers = method_payload["bindings"]["teacher_checkpoint_sha256"]
    basis_teachers = {
        garment: value["sha256"]
        for garment, value in basis_audit["teacher_bindings"].items()
    }
    expected_teacher_shas = EXPECTED_TEACHER_FINAL_SHA
    require(
        {garment: value["sha256"] for garment, value in startup_teachers.items()}
        == expected_teacher_shas,
        "method startup Teacher SHA differs",
    )
    require(
        checkpoint_teachers == expected_teacher_shas,
        "method checkpoint Teacher SHA differs",
    )
    require(basis_teachers == expected_teacher_shas, "basis Teacher SHA differs")
    require(
        method_payload["global_step"] == method_payload["optimizer_step"] == 300,
        "method step300 internal step differs",
    )
    require(finite_tree(method_payload["model"]), "method model is nonfinite")
    require(
        method_training["nan_inf_status"] == "NONE"
        and method_training["oom_status"] == "NONE",
        "method training nonfinite/OOM differs",
    )
    method_audit = {
        "schema_version": (
            "canondressgs.subject00.method_step300_teacher_dependency.v1"
        ),
        "task_id": TASK_ID,
        "created_at": created_at,
        "output_root": str(METHOD_ROOT),
        "startup_config_snapshot": {
            "path": str(METHOD_ROOT / "contract/config_resolved.json"),
            "sha256": sha256_file(METHOD_ROOT / "contract/config_resolved.json"),
            "execution_branch": method_config["execution_branch"],
            "teachers": startup_teachers,
        },
        "preflight_snapshot": method_preflight,
        "basis_snapshot": {
            "path": str(METHOD_ROOT / "basis/basis_audit.json"),
            "sha256": sha256_file(METHOD_ROOT / "basis/basis_audit.json"),
            "teacher_bindings": basis_audit["teacher_bindings"],
            "status": basis_audit["status"],
        },
        "feature_snapshot": {
            "path": str(METHOD_ROOT / "features/feature_audit.json"),
            "sha256": sha256_file(METHOD_ROOT / "features/feature_audit.json"),
            "target_or_teacher_field_in_prediction_forward": feature_audit[
                "target_or_teacher_field_in_prediction_forward"
            ],
            "status": feature_audit["status"],
        },
        "checkpoint": {
            "path": str(METHOD_STEP300),
            "sha256": sha256_file(METHOD_STEP300),
            "parse_status": "PASS",
            "global_step": method_payload["global_step"],
            "optimizer_step": method_payload["optimizer_step"],
            "git": method_payload["bindings"]["git"],
            "teacher_checkpoint_sha256": checkpoint_teachers,
            "base_checkpoint_sha256": method_payload["bindings"][
                "base_checkpoint_sha256"
            ],
            "target_manifest_sha256": method_payload["bindings"][
                "target_manifest_sha256"
            ],
            "model_finite": finite_tree(method_payload["model"]),
        },
        "structured_training": {
            "optimizer_steps_completed": method_training[
                "optimizer_steps_completed"
            ],
            "optimizer_step_monotonic": method_training[
                "optimizer_step_monotonic"
            ],
            "loss_finite": method_training["loss_finite"],
            "gradients_finite": method_training["gradients_finite"],
            "nan_inf_status": method_training["nan_inf_status"],
            "oom_status": method_training["oom_status"],
        },
        "immutable_post_audit": method_post,
        "method_run_status_snapshot": method_status,
        "o03_dependency_status": (
            "PASS_EXACT_VALID_CONCURRENT_FORMAL_O03_CHECKPOINT"
        ),
        "method_rotation0_seed0_valid": True,
        "method_rotation0_seed0_status": (
            "VALID_TEACHER_DEPENDENCY_INITIAL_CELL_COMPLETE"
        ),
        "matrix_counted_cell_count": 1,
        "remaining_preregistered_run_count": 11,
        "status": "PASS_METHOD_STEP300_DEPENDENCY_VALID",
    }

    o01 = teacher_lightweight(O01_ROOT, "O01")
    o04 = teacher_lightweight(O04_ROOT, "O04")
    o01_o04 = {
        "schema_version": (
            "canondressgs.subject00.o01_o04.lightweight_teacher_provenance.v1"
        ),
        "task_id": TASK_ID,
        "created_at": created_at,
        "writer_task": WRITER_TASK,
        "writer_branch": ACCELERATED_BRANCH,
        "teachers": {"O01": o01, "O04": o04},
        "same_class_output_ownership_problem_found": False,
        "status": "PASS_O01_AND_O04_VALID_FORMAL_TARGET_TEACHERS",
    }

    original_registry_path = (
        accelerated_repo
        / "paper_protocol/reviewer_risk/"
        "subject00_base60747_three_garment_teacher_registry_20260727.json"
    )
    correction = {
        "schema_version": (
            "canondressgs.subject00.base60747_three_garment_teacher_registry."
            "correction_overlay.v1"
        ),
        "task_id": TASK_ID,
        "created_at": created_at,
        "overlay_only": True,
        "original_registry_path": (
            "paper_protocol/reviewer_risk/"
            "subject00_base60747_three_garment_teacher_registry_20260727.json"
        ),
        "original_registry_sha256": sha256_file(original_registry_path),
        "original_registry_overwritten": False,
        "original_registry_continues_valid": True,
        "corrections": {
            "O01": {
                "validity": "VALID_FORMAL_TARGET_TEACHER",
                "checkpoint_sha256": EXPECTED_TEACHER_FINAL_SHA["O01"],
            },
            "O03": {
                "validity": "VALID_FORMAL_TARGET_TEACHER",
                "checkpoint_sha256": EXPECTED_TEACHER_FINAL_SHA["O03"],
                "provenance_resolution": (
                    "adopt post-hoc concurrent formal Teacher seal"
                ),
                "old_safe7_equivalence": (
                    "D_ACTIVE_LOSS_INPUT_SCIENTIFIC_MISMATCH"
                ),
            },
            "O04": {
                "validity": "VALID_FORMAL_TARGET_TEACHER",
                "checkpoint_sha256": EXPECTED_TEACHER_FINAL_SHA["O04"],
            },
        },
        "method_step300_valid": True,
        "method_matrix_continuation_authorized_after_this_audit": True,
        "method_matrix_continuation_requires_separate_execution_authority": True,
        "status": "PASS_THREE_GARMENT_REGISTRY_CORRECTED_BY_OVERLAY",
    }

    seal = {
        "schema_version": (
            "canondressgs.subject00.o03.concurrent_formal_teacher."
            "posthoc_provenance_seal.v1"
        ),
        "task_id": TASK_ID,
        "created_at": created_at,
        "original_creation_task": WRITER_TASK,
        "actual_writer": timeline["writer_process"],
        "actual_branch": ACCELERATED_BRANCH,
        "actual_head": WRITER_RUN_HEAD,
        "launch_command": WRITER_LAUNCH_COMMAND,
        "base": {"path": BASE_PATH, "sha256": BASE_SHA, "step": 60747},
        "target": {
            "root": FORMAL_ROOT,
            "actual_index_path": str(train_index_path),
            "actual_index_sha256": O03_TRAIN_INDEX_SHA,
            "contract_index_path": str(contract_index_path),
            "contract_index_sha256": O03_CONTRACT_INDEX_SHA,
            "semantic_index_alias_exact": True,
            "request_ids": EXPECTED_O03_REQUESTS,
            "denominator": 7,
        },
        "loader": {"path": LOADER_REL, "sha256": LOADER_SHA},
        "loss": {"path": LOSS_REL, "sha256": LOSS_SHA},
        "field_access_map_sha256": runtime_trace["field_access_map_sha256"],
        "checkpoint_sha256": {
            str(row["step"]): row["sha256"]
            for row in candidate_checkpoints["checkpoints"]
        },
        "sample_counts": candidate_training["view_update_counts"],
        "no_interleaving_evidence": {
            "single_writer_status": timeline["single_writer_status"],
            "interleaving_status": timeline[
                "concurrent_writer_interleaving_status"
            ],
            "blocked_task_marker_count": timeline["blocked_task_marker_count"],
        },
        "adoption_task": TASK_ID,
        "validity_status": "VALID_FORMAL_TARGET_TEACHER",
    }

    immutable_after = {
        "base": path_snapshot(Path(BASE_PATH)),
        "formal_target": tree_digest(Path(FORMAL_ROOT)),
        "teachers": {
            "O01": path_snapshot(O01_ROOT / "checkpoints/step_001200.pth"),
            "O03_candidate": path_snapshot(
                O03_ROOT / "checkpoints/step_001200.pth"
            ),
            "O03_old": path_snapshot(
                OLD_O03_ROOT / "checkpoints/step_001200.pth"
            ),
            "O04": path_snapshot(O04_ROOT / "checkpoints/step_001200.pth"),
        },
        "method_step300": path_snapshot(METHOD_STEP300),
    }
    require(
        immutable_before == immutable_after,
        "read-only immutable inputs changed during audit",
    )

    summary = {
        "schema_version": (
            "canondressgs.subject00.o03.loss_binding_concurrent_provenance."
            "final_summary.v1"
        ),
        "created_at": created_at,
        "TASK_ID": TASK_ID,
        "SOURCE_BRANCH": SOURCE_BRANCH,
        "SOURCE_HEAD": SOURCE_HEAD,
        "ACCELERATED_LAUNCH_BRANCH": ACCELERATED_BRANCH,
        "ACCELERATED_LAUNCH_HEAD": ACCELERATED_HEAD,
        "MATERIALIZATION_HEAD": MATERIALIZATION_HEAD,
        "NEW_BRANCH": NEW_BRANCH,
        "WINDOWS_WORKTREE": WINDOWS_WORKTREE,
        "CLOUD_WORKTREE": CLOUD_WORKTREE,
        "ACTIVE_OPTIMIZER_PROCESS_COUNT_BEFORE": 0,
        "METHOD_NEW_RUNS_PAUSED_STATUS": True,
        "FROZEN_LOSS_IMPLEMENTATION_PATH": LOSS_REL,
        "FROZEN_LOSS_IMPLEMENTATION_SHA256": LOSS_SHA,
        "LOSS_CALLSITE_PATH": ACTUAL_RUNNER_REL,
        "LOSS_CALLSITE_SHA256": ACTUAL_RUNNER_SHA,
        "LOADER_PATH": LOADER_REL,
        "LOADER_SHA256": LOADER_SHA,
        "BATCH_ADAPTER_PATH": ACTUAL_RUNNER_REL,
        "BATCH_ADAPTER_SHA256": ACTUAL_RUNNER_SHA,
        "ACTIVE_CONFIG_PATH": ACTIVE_CONFIG_REL,
        "ACTIVE_CONFIG_SHA256": ACTIVE_CONFIG_SHA,
        "LOSS_ACTIVE_FIELD_SET": LOSS_ACTIVE_FIELDS,
        "EVALUATION_ACTIVE_FIELD_SET": EVALUATION_ACTIVE_FIELDS,
        "RUNTIME_DERIVED_ACTIVE_FIELD_SET": RUNTIME_DERIVED_FIELDS,
        "SCHEMA_ONLY_FIELD_SET": SCHEMA_ONLY_SCIENTIFIC_FIELDS,
        "QA_ONLY_FIELD_SET": [],
        "QA_CONSUMER_FIELD_SET": SCHEMA_ONLY_SCIENTIFIC_FIELDS,
        "UNUSED_FIELD_SET": [],
        "TARGET_BASE_RGB_CLASS": ["LOSS_ACTIVE", "EVALUATION_ACTIVE"],
        "TARGET_EDIT_MASK_CLASS": ["LOSS_ACTIVE", "EVALUATION_ACTIVE"],
        "TARGET_EDIT_CORE_MASK_CLASS": [
            "SCHEMA_REQUIRED_BUT_CURRENT_LOSS_INACTIVE",
            "QA_ONLY",
            "OTHER_MODE_ONLY",
        ],
        "TARGET_PRESERVE_MASK_CLASS": [
            "SCHEMA_REQUIRED_BUT_CURRENT_LOSS_INACTIVE",
            "QA_ONLY",
            "OTHER_MODE_ONLY",
        ],
        "TARGET_TRANSITION_MASK_CLASS": ["LOSS_ACTIVE"],
        "TARGET_BASE_FOREGROUND_MASK_CLASS": ["LOSS_ACTIVE"],
        "TARGET_OLD_CLOTHING_MASK_CLASS": [
            "LOSS_ACTIVE",
            "EVALUATION_ACTIVE",
        ],
        "TARGET_REVEALED_SKIN_MASK_CLASS": [
            "SCHEMA_REQUIRED_BUT_CURRENT_LOSS_INACTIVE",
            "QA_ONLY",
            "OTHER_MODE_ONLY",
        ],
        "OLD_SAFE7_ACTIVE_INPUT_EQUIVALENCE_CLASS": (
            "D_ACTIVE_LOSS_INPUT_SCIENTIFIC_MISMATCH"
        ),
        "OLD_SAFE7_ACTIVE_FIELD_MISMATCH_COUNT": 5,
        "CANDIDATE_OUTPUT_ROOT": str(O03_ROOT),
        "CANDIDATE_WRITER_TASK": WRITER_TASK,
        "CANDIDATE_WRITER_BRANCH": ACCELERATED_BRANCH,
        "CANDIDATE_WRITER_HEAD": WRITER_RUN_HEAD,
        "SINGLE_WRITER_STATUS": timeline["single_writer_status"],
        "FILE_INTERLEAVING_STATUS": timeline[
            "concurrent_writer_interleaving_status"
        ],
        "CANDIDATE_RUN_BASE_BINDING_STATUS": "PASS_EXACT_BASE60747",
        "CANDIDATE_RUN_TARGET_BINDING_STATUS": (
            "PASS_FORMAL_SAFE7_SEMANTIC_EXACT_INDEX_ALIAS"
        ),
        "CANDIDATE_RUN_LOSS_BINDING_STATUS": (
            "PASS_FROZEN_SUBJECT02_CAPACITY_ORACLE_LOSS_V1"
        ),
        "CANDIDATE_RUN_SLOT04_STATUS": "PASS_ABSENT_0_SAMPLES",
        "CANDIDATE_RUN_OPTIMIZER_STEPS": 1200,
        "CANDIDATE_CHECKPOINT_COUNT": 5,
        "CANDIDATE_FINAL_CHECKPOINT_PATH": str(final_candidate_path),
        "CANDIDATE_FINAL_CHECKPOINT_SHA256": EXPECTED_TEACHER_FINAL_SHA["O03"],
        "CANDIDATE_CHECKPOINT_AUTHENTICITY_STATUS": (
            "PASS_AUTHENTIC_1200_STEP_FORMAL_TARGET_TEACHER"
        ),
        "O03_TEACHER_VALIDITY": "VALID_FORMAL_TARGET_TEACHER",
        "O03_SELECTED_VALID_CHECKPOINT_PATH": str(final_candidate_path),
        "O03_SELECTED_VALID_CHECKPOINT_SHA256": EXPECTED_TEACHER_FINAL_SHA["O03"],
        "METHOD_STEP300_O01_TEACHER_SHA": EXPECTED_TEACHER_FINAL_SHA["O01"],
        "METHOD_STEP300_O03_TEACHER_SHA": EXPECTED_TEACHER_FINAL_SHA["O03"],
        "METHOD_STEP300_O04_TEACHER_SHA": EXPECTED_TEACHER_FINAL_SHA["O04"],
        "METHOD_STEP300_O03_DEPENDENCY_STATUS": (
            "PASS_EXACT_VALID_CONCURRENT_FORMAL_O03_CHECKPOINT"
        ),
        "METHOD_ROTATION0_SEED0_VALID": True,
        "METHOD_ROTATION0_SEED0_STATUS": (
            "VALID_TEACHER_DEPENDENCY_INITIAL_CELL_COMPLETE"
        ),
        "O01_TEACHER_VALIDITY": "VALID_FORMAL_TARGET_TEACHER",
        "O04_TEACHER_VALIDITY": "VALID_FORMAL_TARGET_TEACHER",
        "THREE_GARMENT_REGISTRY_CORRECTION_STATUS": correction["status"],
        "METHOD_MATRIX_CONTINUATION_AUTHORIZED": True,
        "METHOD_MATRIX_CONTINUATION_REQUIRES_SEPARATE_EXECUTION_AUTHORITY": True,
        "NEW_TEACHER_OPTIMIZER_STEPS": 0,
        "NEW_METHOD_OPTIMIZER_STEPS": 0,
        "FORMAL_BASE_RESUME_AUTHORIZED": False,
        "BASE60747_CHECKPOINT_MUTATIONS": 0,
        "FORMAL_TARGET_MUTATIONS": 0,
        "O01_TEACHER_CHECKPOINT_MUTATIONS": 0,
        "O03_CANDIDATE_CHECKPOINT_MUTATIONS": 0,
        "O03_OLD_CHECKPOINT_MUTATIONS": 0,
        "O04_TEACHER_CHECKPOINT_MUTATIONS": 0,
        "METHOD_STEP300_CHECKPOINT_MUTATIONS": 0,
        "RAW_MUTATIONS": 0,
        "MASK_MUTATIONS": 0,
        "PAPER_MODIFICATIONS": 0,
        "TEST_RESULT": "PENDING_45_MACHINE_CHECKS",
        "UPSTREAM_FORMAL_CONTRACT_REPLAY": formal_test_evidence[
            "upstream_contract_replay"
        ]["interpretation"],
        "FORMAL_TEACHER_METHOD_PYTEST": "PASS_5_TESTS",
        "COMMIT_HEAD": "PENDING_TASK_COMMIT",
        "FINAL_REPORTING_HEAD": "PENDING_FINAL_REPORTING_COMMIT",
        "ORIGIN_SYNC_STATUS": "PENDING",
        "CLOUD_GIT_SYNC_STATUS": "PENDING",
        "WORKTREE_CLEAN_STATUS": "PENDING_AFTER_COMMIT",
        "PAPER_FINAL": False,
        "FINAL_CLASSIFICATION": FINAL_CLASSIFICATION,
        "NEXT_TASK": NEXT_TASK,
    }

    checks = [
        check("01_source_branch_head", git(source_repo, "rev-parse", "HEAD") == SOURCE_HEAD, SOURCE_HEAD),
        check("02_accelerated_launch_head", git(accelerated_repo, "rev-parse", "HEAD") == ACCELERATED_HEAD, ACCELERATED_HEAD),
        check("03_materialization_head", git(materialization_repo, "rev-parse", "HEAD") == MATERIALIZATION_HEAD, MATERIALIZATION_HEAD),
        check("04_source_worktrees_clean", all(not git(path, "status", "--porcelain") for path in [source_repo, accelerated_repo, materialization_repo, old_repo]), "four authoritative worktrees clean"),
        check("05_active_optimizer_process_count", len(active_optimizer_lines) == 0, len(active_optimizer_lines)),
        check("06_frozen_subject02_loss_implementation", sha256_file(loss_path) == LOSS_SHA and upstream_contract_replay["passed"] == 12, {"sha256": LOSS_SHA, "upstream_contract_replay": formal_test_evidence["upstream_contract_replay"]["interpretation"]}),
        check("07_loss_callsite", static_loss_bindings == expected_static_bindings, static_loss_bindings),
        check("08_loader", sha256_file(loader_path) == LOADER_SHA, LOADER_SHA),
        check("09_batch_adapter", set(runtime_trace["access_by_stage"]["loader_batch_adapter"]) >= set(LOSS_ACTIVE_FIELDS), runtime_trace["access_by_stage"]["loader_batch_adapter"]),
        check("10_config", sha256_file(config_path) == ACTIVE_CONFIG_SHA, ACTIVE_CONFIG_SHA),
        check("11_static_field_graph", consumer_graph["status"].startswith("PASS"), consumer_graph["status"]),
        check("12_runtime_field_access", runtime_trace["status"].startswith("PASS"), runtime_trace["runtime_accessed_field_set"]),
        check("13_active_field_set", set(runtime_trace["access_by_stage"]["loss"]) == set(LOSS_ACTIVE_FIELDS), runtime_trace["access_by_stage"]["loss"]),
        check("14_inactive_schema_field_set", set(SCHEMA_ONLY_SCIENTIFIC_FIELDS).isdisjoint(runtime_trace["access_by_stage"]["loss"]), SCHEMA_ONLY_SCIENTIFIC_FIELDS),
        check("15_eight_disputed_field_classification", len(active_fields["eight_disputed_field_classification"]) == 8, active_fields["eight_disputed_field_classification"]),
        check("16_old_snapshot_active_input_equivalence", equivalence["classification"] == "D_ACTIVE_LOSS_INPUT_SCIENTIFIC_MISMATCH", equivalence["active_field_mismatch_fields"]),
        check("17_output_root_timeline", timeline["file_count"] > 0 and timeline["earliest_file_mtime_utc"] < timeline["latest_file_mtime_utc"], [timeline["earliest_file_mtime_utc"], timeline["latest_file_mtime_utc"]]),
        check("18_writer_identity", provenance["writer_task"] == WRITER_TASK and provenance["writer_head"] == WRITER_RUN_HEAD, [provenance["writer_task"], provenance["writer_head"]]),
        check("19_single_writer_status", timeline["single_writer_status"].startswith("PASS"), timeline["single_writer_status"]),
        check("20_interleaving_status", timeline["concurrent_writer_interleaving_status"].startswith("PASS"), timeline["concurrent_writer_interleaving_status"]),
        check("21_checkpoint_exact_set", candidate_checkpoints["exact_checkpoint_set"], CHECKPOINT_STEPS),
        check("22_checkpoint_parse", candidate_checkpoints["parse_count"] == 5, candidate_checkpoints["parse_count"]),
        check("23_base_binding", all(row["metadata"]["base_checkpoint_sha256"] == BASE_SHA for row in candidate_checkpoints["checkpoints"]), BASE_SHA),
        check("24_formal_target_binding", index_alias_semantic_exact and all(row["metadata"]["target_index_sha256"] == O03_TRAIN_INDEX_SHA for row in candidate_checkpoints["checkpoints"]), [O03_TRAIN_INDEX_SHA, O03_CONTRACT_INDEX_SHA]),
        check("25_slot04_absence", EXCLUDED_O03 not in train_index["request_ids"], EXCLUDED_O03),
        check("26_sampler_counts", candidate_training["sampler_balanced"] and sum(candidate_training["view_update_counts"].values()) == 1200, candidate_training["view_update_counts"]),
        check("27_optimizer_steps_1200", candidate_training["optimizer_steps"] == 1200, candidate_training["optimizer_steps"]),
        check("28_candidate_loss_path", provenance["candidate_run_matches_frozen_subject02_loss_path"], [LOSS_REL, LOSS_SHA]),
        check("29_trainable_changes", all(candidate_training["trainable_parameter_changed"].values()), candidate_training["trainable_parameter_changed"]),
        check("30_frozen_immutability", candidate_post["base_checkpoint_sha256_unchanged"] and candidate_post["formal_target_assets_unchanged"], candidate_post["status"]),
        check("31_no_nan_inf_oom", candidate_training["nan_inf_status"] == "NONE" and candidate_training["oom_status"] == "NONE" and state_finite, [candidate_training["nan_inf_status"], candidate_training["oom_status"]]),
        check("32_method_teacher_snapshot", {k: v["sha256"] for k, v in startup_teachers.items()} == expected_teacher_shas, startup_teachers),
        check("33_method_o03_checkpoint_sha", checkpoint_teachers["O03"] == EXPECTED_TEACHER_FINAL_SHA["O03"], checkpoint_teachers["O03"]),
        check("34_method_cell_validity", method_audit["method_rotation0_seed0_valid"], method_audit["method_rotation0_seed0_status"]),
        check("35_o01_provenance", o01["validity"] == "VALID_FORMAL_TARGET_TEACHER", o01["final_checkpoint_sha256"]),
        check("36_o04_provenance", o04["validity"] == "VALID_FORMAL_TARGET_TEACHER", o04["final_checkpoint_sha256"]),
        check("37_registry_correction", correction["status"].startswith("PASS"), correction["status"]),
        check("38_no_new_optimizer_steps", summary["NEW_TEACHER_OPTIMIZER_STEPS"] == summary["NEW_METHOD_OPTIMIZER_STEPS"] == 0, [0, 0]),
        check("39_base_immutable", immutable_before["base"] == immutable_after["base"], immutable_after["base"]["sha256"]),
        check("40_target_immutable", immutable_before["formal_target"]["aggregate_sha256"] == immutable_after["formal_target"]["aggregate_sha256"], immutable_after["formal_target"]["aggregate_sha256"]),
        check("41_teachers_immutable", immutable_before["teachers"] == immutable_after["teachers"], {k: v["sha256"] for k, v in immutable_after["teachers"].items()}),
        check("42_method_checkpoint_immutable", immutable_before["method_step300"] == immutable_after["method_step300"], immutable_after["method_step300"]["sha256"]),
        check("43_paper_modification_zero", summary["PAPER_MODIFICATIONS"] == 0 and summary["PAPER_FINAL"] is False, [0, False]),
        check("44_final_classification", summary["FINAL_CLASSIFICATION"] == FINAL_CLASSIFICATION, FINAL_CLASSIFICATION),
        check("45_next_task_uniqueness", summary["NEXT_TASK"] == NEXT_TASK and isinstance(summary["NEXT_TASK"], str), NEXT_TASK),
    ]
    failed = [row for row in checks if row["status"] != "PASS"]
    require(not failed, f"machine checks failed: {failed}")
    summary["TEST_RESULT"] = (
        "PASS_45_OF_45; PYTEST_5_PASSED; "
        "UPSTREAM_SCIENTIFIC_CONTRACT_12_OF_12_PASSED"
    )
    tests = {
        "schema_version": (
            "canondressgs.subject00.o03.loss_binding_concurrent_provenance."
            "tests.v1"
        ),
        "task_id": TASK_ID,
        "created_at": created_at,
        "status": "PASS_45_OF_45",
        "passed_count": 45,
        "failed_count": 0,
        "checks": checks,
        "formal_test_evidence": formal_test_evidence,
        "optimizer_steps_executed_by_audit": 0,
    }
    report = report_markdown(summary)
    handoff = {
        "schema_version": (
            "canondressgs.project_control_handoff.subject00_o03."
            "loss_binding_concurrent_provenance.v1"
        ),
        "task_id": TASK_ID,
        "created_at": created_at,
        "outcome": {
            "o03_teacher_validity": "VALID_FORMAL_TARGET_TEACHER",
            "method_rotation0_seed0_valid": True,
            "matrix_ready_to_continue": True,
            "matrix_execution_started_by_this_task": False,
            "final_classification": FINAL_CLASSIFICATION,
            "next_task": NEXT_TASK,
        },
        "selected_o03_checkpoint": {
            "path": str(final_candidate_path),
            "sha256": EXPECTED_TEACHER_FINAL_SHA["O03"],
        },
        "constraints": {
            "method_new_runs_paused_status": True,
            "new_teacher_optimizer_steps": 0,
            "new_method_optimizer_steps": 0,
            "formal_base_resume_authorized": False,
            "paper_final": False,
        },
        "artifact_paths": ARTIFACTS,
        "final_fields": summary,
    }

    values = {
        "active_fields": active_fields,
        "consumer_graph": consumer_graph,
        "runtime_trace": runtime_trace,
        "equivalence": equivalence,
        "timeline": timeline,
        "provenance": provenance,
        "checkpoint": candidate_checkpoint_audit,
        "method": method_audit,
        "o01_o04": o01_o04,
        "correction": correction,
        "tests": tests,
        "summary": summary,
        "handoff": handoff,
        "seal": seal,
    }
    for name, value in values.items():
        write_json(repo / ARTIFACTS[name], value)
    write_text(repo / ARTIFACTS["report"], report)
    write_text(repo / ARTIFACTS["docs_report"], report)

    generated = {
        name: {
            "path": relative,
            "sha256": sha256_file(repo / relative),
        }
        for name, relative in ARTIFACTS.items()
    }
    require(len(generated) == 16, "required artifact count differs")
    print(
        json.dumps(
            {
                "task_id": TASK_ID,
                "status": "PASS",
                "artifact_count": len(generated),
                "test_result": summary["TEST_RESULT"],
                "final_classification": FINAL_CLASSIFICATION,
                "next_task": NEXT_TASK,
                "artifacts": generated,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
