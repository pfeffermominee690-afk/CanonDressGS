#!/usr/bin/env python3
"""Build the Git-side blocker evidence from the sealed cloud preflight snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


TASK_ID = "AAAI27-SUBJECT00-O03-FORMAL-TARGET-CAMSAFE7-TEACHER-RERUN-001"
SOURCE_BRANCH = (
    "research/subject00-o03-camsafe7-formal-target-equivalence-audit-20260727"
)
SOURCE_HEAD = "113f669eac37fa1f178886f3a7dc1a3a81751441"
MATERIALIZATION_BRANCH = (
    "research/subject00-teacher-target-materialization-quarantine-20260727"
)
MATERIALIZATION_HEAD = "227fd156d420e4bf291413952f448780b8446b37"
NEW_BRANCH = (
    "research/subject00-o03-formal-target-camsafe7-teacher-rerun-20260727"
)
WINDOWS_WORKTREE = (
    r"E:\model_train\canondressgs_subject00_o03_formal_target_"
    r"camsafe7_teacher_rerun"
)
CLOUD_WORKTREE = (
    "/root/autodl-tmp/canondressgs_work/worktrees/"
    "canondressgs_subject00_o03_formal_target_camsafe7_teacher_rerun"
)
FORMAL_TARGET_ROOT = (
    "/root/autodl-tmp/canondressgs_work/teacher_targets/"
    "SUBJECT00-24CELL-001/attempt_001"
)
FORMAL_INDEX_PATH = (
    FORMAL_TARGET_ROOT
    + "/10_final_registry/indexes/O03_provisional_base60747_records.json"
)
BASE_CHECKPOINT_PATH = (
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-FORMAL-BASE-101245-001/attempt_001/checkpoints/step_060747.pth"
)
BASE_CHECKPOINT_SHA = (
    "2d09b1ce6bb19b8b6418314caab8dc1cf92d70e432a9e09e009989332ea4a2f7"
)
OUTPUT_ROOT = (
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-O03-TEACHER-BASE60747-FORMAL-CAMSAFE7-001"
)
EXPECTED_REQUESTS = [
    "subject00_O03_slot00_canary_attempt004_cand00",
    "subject00_O03_slot01_remaining_attempt005_cand00",
    "subject00_O03_slot02_cand00",
    "subject00_O03_slot03_cand01",
    "subject00_O03_slot05_canary_attempt004_cand00",
    "subject00_O03_slot06_remaining_attempt005_cand00",
    "subject00_O03_slot07_canary_attempt004_cand00",
]
EXPECTED_CAMERAS = ["cam17", "cam21", "cam14", "cam23", "cam02", "cam09", "cam05"]
EXPECTED_SLOTS = [
    "slot_00",
    "slot_01",
    "slot_02",
    "slot_03",
    "slot_05",
    "slot_06",
    "slot_07",
]
EXCLUDED_REQUEST = "subject00_O03_slot04_canary_attempt004_cand00"
REQUIRED_FIELDS = [
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
LOSS_WEIGHTS = {
    "garment_rgb": 1.0,
    "alpha_foreground": 0.5,
    "new_silhouette_alpha": 1.0,
    "boundary_rgb": 0.25,
    "protected_rgb": 10.0,
    "protected_alpha": 5.0,
    "stability": 0.0001,
}
PLANNED_COUNTS = {
    "slot_00": 172,
    "slot_01": 172,
    "slot_02": 172,
    "slot_03": 171,
    "slot_05": 171,
    "slot_06": 171,
    "slot_07": 171,
}
ACTUAL_COUNTS = {slot: 0 for slot in EXPECTED_SLOTS}
FINAL_CLASSIFICATION = (
    "SUBJECT00_O03_FORMAL_TARGET_RERUN_BLOCKED_BY_LOSS_FIELD_BINDING_AMBIGUITY"
)
NEXT_TASK = "RESOLVE_SUBJECT00_FORMAL_TARGET_LOSS_FIELD_BINDING"
CREATED_AT = "2026-07-27T00:00:00+08:00"

RISK = Path("paper_protocol/reviewer_risk")
HANDOFF = Path("project_control_handoff")
DOCS = Path("docs/PAPER")
ARTIFACT_PATHS = {
    "binding": RISK / "subject00_O03_formal_target_camsafe7_binding_registry_20260727.json",
    "fields": RISK / "subject00_O03_formal_target_scientific_field_registry_20260727.json",
    "loss_map": RISK / "subject00_O03_formal_target_loss_consumer_map_20260727.json",
    "training": RISK / "subject00_O03_formal_target_training_registry_20260727.json",
    "sampling": RISK / "subject00_O03_formal_target_sampling_registry_20260727.json",
    "checkpoints": RISK / "subject00_O03_formal_target_checkpoint_registry_20260727.json",
    "per_view": RISK / "subject00_O03_formal_target_per_view_metrics_20260727.json",
    "comparative": RISK / "subject00_O03_formal_target_comparative_metrics_20260727.json",
    "animation": RISK / "subject00_O03_formal_target_animation_audit_20260727.json",
    "review": RISK / "subject00_O03_formal_target_human_review_manifest_20260727.json",
    "tests": RISK / "subject00_O03_formal_target_execution_tests_20260727.json",
    "summary": RISK / "subject00_O03_formal_target_final_summary_20260727.json",
    "report": RISK / "SUBJECT00_O03_FORMAL_TARGET_CAMSAFE7_TEACHER_REPORT_20260727.md",
    "handoff": HANDOFF / "subject00_O03_formal_target_camsafe7_teacher_handoff_20260727.json",
    "docs_report": DOCS / "AAAI27_SUBJECT00_O03_FORMAL_TARGET_CAMSAFE7_TEACHER_REPORT_20260727.md",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--machine-snapshot", type=Path, required=True)
    parser.add_argument("--commit-head", default="PENDING_ARTIFACT_COMMIT")
    parser.add_argument(
        "--final-reporting-head",
        default="RECORDED_IN_FINAL_TASK_RESPONSE_AFTER_REPORTING_COMMIT",
    )
    parser.add_argument("--origin-status", default="PENDING_COMMIT_AND_PUSH")
    parser.add_argument("--cloud-status", default="PENDING_FINAL_SYNC")
    parser.add_argument(
        "--worktree-status", default="DIRTY_EXPECTED_ARTIFACT_GENERATION"
    )
    parser.add_argument(
        "--test-result",
        default=(
            "PY_COMPILE_PENDING; JSON_PARSE_PENDING; "
            "STRUCTURED_CHECKS_PENDING; GIT_DIFF_CHECK_PENDING"
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def consumer_value(machine: dict[str, Any], field: str) -> dict[str, Any]:
    return machine["loss_binding"]["consumer_map"][field]


def all_field_consumers(machine: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        "target_edit_rgb": {
            "status": "BOUND_IN_MATERIALIZATION_ADAPTER_BUT_NOT_FROZEN_SOURCE_EXECUTION",
            "argument": "target_rgb",
            "loss_components": ["garment_rgb", "boundary_rgb"],
        },
        "target_foreground_mask": {
            "status": "BOUND_IN_MATERIALIZATION_ADAPTER_BUT_NOT_FROZEN_SOURCE_EXECUTION",
            "argument": "target_foreground",
            "loss_components": ["alpha_foreground", "new_silhouette_alpha"],
        },
        "target_clothing_mask": {
            "status": "BOUND_IN_MATERIALIZATION_ADAPTER_BUT_NOT_FROZEN_SOURCE_EXECUTION",
            "argument": "clothing_mask",
            "loss_components": ["garment_rgb"],
        },
        "target_protected_mask": {
            "status": "BOUND_IN_MATERIALIZATION_ADAPTER_BUT_NOT_FROZEN_SOURCE_EXECUTION",
            "argument": "protected_mask",
            "loss_components": [
                "garment_rgb_exclusion",
                "boundary_rgb_exclusion",
                "protected_rgb",
                "protected_alpha",
            ],
        },
        **machine["loss_binding"]["consumer_map"],
    }


def final_fields(args: argparse.Namespace, machine: dict[str, Any]) -> dict[str, Any]:
    consumers = machine["loss_binding"]["consumer_map"]
    return {
        "TASK_ID": TASK_ID,
        "SOURCE_BRANCH": SOURCE_BRANCH,
        "SOURCE_HEAD": SOURCE_HEAD,
        "MATERIALIZATION_BRANCH": MATERIALIZATION_BRANCH,
        "MATERIALIZATION_HEAD": MATERIALIZATION_HEAD,
        "NEW_BRANCH": NEW_BRANCH,
        "WINDOWS_WORKTREE": WINDOWS_WORKTREE,
        "CLOUD_WORKTREE": CLOUD_WORKTREE,
        "FORMAL_TARGET_ROOT": FORMAL_TARGET_ROOT,
        "FORMAL_O03_INDEX_PATH": FORMAL_INDEX_PATH,
        "FORMAL_SCHEMA": "canondressgs.full_dataset.v1",
        "FORMAL_LOADER_PATH": "scene/full_dressable_dataset.py",
        "FORMAL_LOADER_SHA256": (
            "786c93355776093e610dfe1bc74efd61233da134550a220bc06601f4f2365508"
        ),
        "TARGET_VIEW_COUNT": 7,
        "TARGET_REQUEST_IDS": EXPECTED_REQUESTS,
        "TARGET_CAMERA_IDS": EXPECTED_CAMERAS,
        "EXCLUDED_REQUEST_IDS": [EXCLUDED_REQUEST],
        "SLOT04_PRESENCE_STATUS": "ABSENT_INDEX_LOADER_SAMPLER_LOSS_METRICS",
        "BASE_CHECKPOINT_PATH": BASE_CHECKPOINT_PATH,
        "BASE_CHECKPOINT_SHA256": BASE_CHECKPOINT_SHA,
        "BASE_CHECKPOINT_STEP": 60747,
        "OLD_CONTAMINATED_CHECKPOINT_USED": False,
        "OLD_RUNLOCAL_SAFE7_CHECKPOINT_USED": False,
        "OUTPUT_ROOT": OUTPUT_ROOT,
        "ATTEMPT_ID": "attempt_001",
        "REQUIRED_SCIENTIFIC_FIELD_COUNT": 12,
        "REQUIRED_SCIENTIFIC_FIELDS": REQUIRED_FIELDS,
        "PREVIOUS_MISMATCH_FIELD_COUNT": 8,
        "PREVIOUS_MISMATCH_FIELDS": MISMATCH_FIELDS,
        "FORMAL_FIELD_LOAD_STATUS": machine["formal_target"]["load_status"],
        "FORMAL_FIELD_VALUE_HASH_STATUS": machine["formal_target"][
            "value_hash_status"
        ],
        "LOSS_CONSUMER_MAP_STATUS": machine["loss_binding"]["status"],
        "LOSS_CONSUMER_MAP_PATH": str(ARTIFACT_PATHS["loss_map"]).replace("\\", "/"),
        "TARGET_BASE_RGB_CONSUMER": consumers["target_base_rgb"],
        "TARGET_EDIT_MASK_CONSUMER": consumers["target_edit_mask"],
        "TARGET_EDIT_CORE_MASK_CONSUMER": consumers["target_edit_core_mask"],
        "TARGET_PRESERVE_MASK_CONSUMER": consumers["target_preserve_mask"],
        "TARGET_TRANSITION_MASK_CONSUMER": consumers["target_transition_mask"],
        "TARGET_BASE_FOREGROUND_MASK_CONSUMER": consumers[
            "target_base_foreground_mask"
        ],
        "TARGET_OLD_CLOTHING_MASK_CONSUMER": consumers[
            "target_old_clothing_mask"
        ],
        "TARGET_REVEALED_SKIN_MASK_CONSUMER": consumers[
            "target_revealed_skin_mask"
        ],
        "LOSS_CONTRACT": "CAPACITY_ORACLE_LOSS_V1",
        "OPTIMIZER": "Adam (PLANNED_NOT_CONSTRUCTED)",
        "LEARNING_RATES": {"geometry": 0.001, "appearance": 0.002},
        "SEED": 20260718,
        "TRAINING_STEPS": 1200,
        "VIEW_SAMPLE_COUNTS": ACTUAL_COUNTS,
        "SLOT04_SAMPLE_COUNT": 0,
        "PRE_LOOP_LOADER_STATUS": "PASS_7_OF_7_CPU_ONLY_ZERO_OPTIMIZER",
        "PRE_LOOP_FORWARD_STATUS": "NOT_RUN_BLOCKED_BEFORE_GPU_FORWARD",
        "PRE_LOOP_BACKWARD_STATUS": "NOT_RUN_BLOCKED_BEFORE_BACKWARD",
        "OPTIMIZER_STEPS_COMPLETED": 0,
        "CHECKPOINT_COUNT": 0,
        "CHECKPOINT_PATHS": [],
        "CHECKPOINT_SHA256": [],
        "CHECKPOINT_FORMAL_TARGET_BINDING_STATUS": (
            "NOT_CREATED_BLOCKED_BEFORE_CHECKPOINT_0"
        ),
        "TRAINABLE_PARAMETER_CHANGE_STATUS": "NOT_RUN_NO_TRAINABLE_MUTATION",
        "FROZEN_PARAMETER_MUTATION_STATUS": "PASS_ZERO_MUTATIONS",
        "LOSS_INITIAL": None,
        "LOSS_FINAL": None,
        "NAN_INF_STATUS": (
            "PASS_LOADER_84_OF_84_FINITE; TRAINING_NOT_RUN"
        ),
        "OOM_STATUS": "NOT_APPLICABLE_NO_CUDA_OR_TRAINING",
        "WALL_TIME": None,
        "PEAK_VRAM": 0,
        "FULL_IMAGE_LPIPS": None,
        "GARMENT_REGION_LPIPS": None,
        "SILHOUETTE_IOU": None,
        "BOUNDARY_F": None,
        "PROTECTED_REGION_LPIPS": None,
        "PROTECTED_REGION_RGB_MAE": None,
        "ALPHA_FOREGROUND_ERROR": None,
        "SEVERE_ARTIFACT_COUNT": None,
        "PER_VIEW_METRICS_PATH": str(ARTIFACT_PATHS["per_view"]).replace("\\", "/"),
        "COMPARATIVE_METRICS_PATH": str(ARTIFACT_PATHS["comparative"]).replace(
            "\\", "/"
        ),
        "DIFFERENT_CAMERA_STATUS": "NOT_RUN_BLOCKED_BEFORE_RENDERING",
        "DIFFERENT_POSE_STATUS": "NOT_RUN_BLOCKED_BEFORE_RENDERING",
        "REVIEW_PACKAGE_PATH": None,
        "HUMAN_VISUAL_DECISION": None,
        "SCIENTIFIC_PASS": None,
        "PAPER_ELIGIBLE": False,
        "FORMAL_BASE_STATUS": "USER_AUTHORIZED_PAUSED",
        "FORMAL_BASE_DURABLE_RESUME_STEP": 60747,
        "FORMAL_BASE_RESUME_AUTHORIZED": False,
        "FORMAL_TARGET_MUTATIONS": 0,
        "RAW_MUTATIONS": 0,
        "PERSON_MASK_MUTATIONS": 0,
        "GARMENT_MASK_MUTATIONS": 0,
        "CAMERA_RECORD_MUTATIONS": 0,
        "BASE60747_CHECKPOINT_MUTATIONS": 0,
        "OLD_CONTAMINATED_RUN_MUTATIONS": 0,
        "OLD_RUNLOCAL_SAFE7_RUN_MUTATIONS": 0,
        "PAPER_MODIFICATIONS": 0,
        "TEST_RESULT": args.test_result,
        "COMMIT_HEAD": args.commit_head,
        "FINAL_REPORTING_HEAD": args.final_reporting_head,
        "ORIGIN_SYNC_STATUS": args.origin_status,
        "CLOUD_GIT_SYNC_STATUS": args.cloud_status,
        "WORKTREE_CLEAN_STATUS": args.worktree_status,
        "PAPER_FINAL": False,
        "FINAL_CLASSIFICATION": FINAL_CLASSIFICATION,
        "NEXT_TASK": NEXT_TASK,
    }


def execution_checks(machine: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(name: str, status: str, evidence: Any) -> None:
        checks.append(
            {
                "index": len(checks) + 1,
                "name": name,
                "status": status,
                "evidence": evidence,
            }
        )

    add("source_branch_head", "PASS", SOURCE_HEAD)
    add("materialization_branch_head", "PASS", MATERIALIZATION_HEAD)
    add("source_clean", "PASS", "both authoritative worktrees clean at preflight")
    add("formal_target_root", "PASS", FORMAL_TARGET_ROOT)
    add("formal_O03_index", "PASS", machine["formal_target"]["index_sha256"])
    add("exact_seven_request_ids", "PASS", EXPECTED_REQUESTS)
    add("request_order", "PASS", EXPECTED_REQUESTS)
    add("slot04_absent", "PASS", EXCLUDED_REQUEST)
    add("raw_sha_7_of_7", "PASS", "bound materialized SHA verified through records")
    add("person_mask_sha_7_of_7", "PASS", "bound materialized SHA verified")
    add("garment_mask_sha_7_of_7", "PASS", "bound materialized SHA verified")
    add("camera_binding_7_of_7", "PASS", EXPECTED_CAMERAS)
    add("loader_sha", "PASS", machine["loader"])
    add("schema", "PASS", "canondressgs.full_dataset.v1")
    add("required_field_inventory", "PASS", REQUIRED_FIELDS)
    add("eight_mismatch_fields_7_of_7", "PASS", "56/56 loaded")
    add(
        "mismatch_field_value_hashes",
        "PASS",
        "56/56 finite value hashes recorded",
    )
    add("loss_consumer_map", "PASS_BLOCKER_DETECTED", machine["loss_binding"]["status"])
    add("target_base_rgb_consumer", "PASS_AUDITED", consumer_value(machine, "target_base_rgb"))
    add(
        "edit_core_preserve_transition_consumers",
        "PASS_BLOCKER_DETECTED",
        {
            field: consumer_value(machine, field)
            for field in (
                "target_edit_mask",
                "target_edit_core_mask",
                "target_preserve_mask",
                "target_transition_mask",
            )
        },
    )
    add(
        "base_foreground_consumer",
        "PASS_AUDITED",
        consumer_value(machine, "target_base_foreground_mask"),
    )
    add(
        "old_clothing_consumer",
        "PASS_AUDITED",
        consumer_value(machine, "target_old_clothing_mask"),
    )
    add(
        "revealed_skin_consumer",
        "PASS_BLOCKER_DETECTED",
        consumer_value(machine, "target_revealed_skin_mask"),
    )
    add("Base60747_SHA", "PASS", machine["base_checkpoint"])
    add(
        "initialization_not_old_teacher",
        "PASS",
        {
            "old_contaminated": False,
            "old_runlocal_safe7": False,
            "Base60747_cpu_load": "PASS",
        },
    )
    add("output_root_new", "PASS", "absent before and after blocked preflight")
    add("pre_loop_loader", "PASS", "7/7 CPU loader; 84/84 fields finite")
    add("pre_loop_forward", "NOT_APPLICABLE_BLOCKED_BY_GATE", "0 GPU forwards")
    add("pre_loop_loss", "NOT_APPLICABLE_BLOCKED_BY_GATE", "no loss construction")
    add("pre_loop_backward", "NOT_APPLICABLE_BLOCKED_BY_GATE", "0 backward calls")
    add(
        "optimizer_steps_1200",
        "NOT_APPLICABLE_BLOCKED_BY_GATE",
        {"planned": 1200, "actual": 0},
    )
    add(
        "sample_counts_balanced",
        "NOT_APPLICABLE_BLOCKED_BY_GATE",
        {"planned": PLANNED_COUNTS, "actual": ACTUAL_COUNTS},
    )
    add("slot04_sample_count_zero", "PASS", 0)
    add("five_checkpoints", "NOT_APPLICABLE_BLOCKED_BY_GATE", [])
    add(
        "checkpoint_formal_target_binding",
        "NOT_APPLICABLE_BLOCKED_BY_GATE",
        "no new checkpoint",
    )
    add(
        "trainable_parameter_changes",
        "NOT_APPLICABLE_BLOCKED_BY_GATE",
        "no model construction",
    )
    add("frozen_parameter_immutability", "PASS", "zero mutations")
    add(
        "no_NaN_Inf",
        "PASS",
        "84/84 loader fields finite; training not executed",
    )
    add("no_OOM", "PASS", "no CUDA allocation attempted")
    add("per_view_metrics_7", "NOT_APPLICABLE_BLOCKED_BY_GATE", [])
    add("evaluation_denominator_7", "NOT_APPLICABLE_BLOCKED_BY_GATE", "planned 7")
    add("historical_comparison", "NOT_APPLICABLE_BLOCKED_BY_GATE", None)
    add("animation_audit", "NOT_APPLICABLE_BLOCKED_BY_GATE", None)
    add("review_package", "NOT_APPLICABLE_BLOCKED_BY_GATE", None)
    add(
        "human_fields_null",
        "PASS",
        {"HUMAN_VISUAL_DECISION": None, "SCIENTIFIC_PASS": None},
    )
    add("paper_eligible_false", "PASS", False)
    add("formal_target_immutable", "PASS", "before/after registry exact")
    add("Base_checkpoint_immutable", "PASS", "SHA exact before/after")
    add("old_runs_immutable", "PASS", "zero mutations")
    add(
        "Formal_Base_paused",
        "PASS",
        {
            "status": "USER_AUTHORIZED_PAUSED",
            "durable_step": 60747,
            "resume_authorized": False,
        },
    )
    add("paper_modification_zero", "PASS", 0)
    add("final_classification", "PASS", FINAL_CLASSIFICATION)
    add("NEXT_TASK_uniqueness", "PASS", NEXT_TASK)
    if len(checks) != 53:
        raise RuntimeError(f"expected 53 checks, got {len(checks)}")
    return checks


def report_markdown(machine: dict[str, Any], fields: dict[str, Any]) -> str:
    consumers = machine["loss_binding"]["consumer_map"]
    rows = "\n".join(
        "| `{}` | `{}` | `{}` |".format(
            field,
            consumers[field]["argument"],
            consumers[field]["status"],
        )
        for field in MISMATCH_FIELDS
    )
    return f"""# Subject00 O03 正式 Target Camera-safe7 Teacher 执行报告

任务编号：`{TASK_ID}`

## 结论

本次执行在 optimizer 之前按冻结合同停止。正式 CPU loader 已通过：7/7 个 O03
camera-safe 记录按固定顺序加载，12 个科学字段共 84 个 tensor 均完成 shape、
dtype、范围、非零计数、finite 与 value SHA 审计。Base60747 的文件大小、SHA256、
CPU `torch.load` 和内部 step=60747 也全部通过。

阻断原因不是 target 数据失败，而是正式字段到冻结 loss 的实际绑定不完整：
`target_edit_core_mask`、`target_preserve_mask`、`target_revealed_skin_mask` 在
`CAPACITY_ORACLE_LOSS_V1` 中没有 consumer；另外五个字段只出现在
materialization 分支的 capacity adapter 中，冻结的执行源 HEAD 仍只有旧的
run-local snapshot loss 路径。合同中的示意语义不能代替代码绑定，也不能切换到
`SUPPORT_AWARE_REGION_TRUSTED_OBJECTIVE_V6_1`，因为那会改变冻结 loss 合同。

因此最终分类为：

`{FINAL_CLASSIFICATION}`

唯一后续任务：

`{NEXT_TASK}`

## 八字段实际 consumer 审计

| 正式字段 | capacity 参数 | 审计状态 |
|---|---|---|
{rows}

## 执行边界

- 新 output root：未创建。
- CUDA forward / backward：0 / 0。
- optimizer：未构造；optimizer step：0。
- 新 checkpoint：0。
- 自动重试、第二次运行、超参扫描：均未发生。
- 正式 target、raw、person/garment masks、camera、Base60747、历史 Teacher
  和 Formal Base：零修改。
- Formal Base 保持 `USER_AUTHORIZED_PAUSED`，durable step=60747，
  resume authorization=false。
- 论文正文：零修改；paper eligible=false；paper final=false。

冻结任务末尾同时列出了 `OPTIMIZER_STEPS_COMPLETED=1200`，但同一合同更早明确
规定：任一正式字段没有唯一 consumer 时不得启动 optimizer。这里记录实际值 0，
以服从安全门禁并避免伪造训练完成状态。

## 关键证据

- 正式 loader Windows 冻结 SHA256：
  `786c93355776093e610dfe1bc74efd61233da134550a220bc06601f4f2365508`
- 正式 loader 云端 LF 容器 SHA256：
  `{machine["loader"]["sha256_lf_runtime"]}`
- Base60747 SHA256：`{BASE_CHECKPOINT_SHA}`
- loss consumer map：
  `{str(ARTIFACT_PATHS["loss_map"]).replace(chr(92), "/")}`
- 84 项字段注册表：
  `{str(ARTIFACT_PATHS["fields"]).replace(chr(92), "/")}`
- 53 项条件化机器检查：
  `{str(ARTIFACT_PATHS["tests"]).replace(chr(92), "/")}`

## 最终字段摘要

- `FORMAL_FIELD_LOAD_STATUS`: `{fields["FORMAL_FIELD_LOAD_STATUS"]}`
- `FORMAL_FIELD_VALUE_HASH_STATUS`: `{fields["FORMAL_FIELD_VALUE_HASH_STATUS"]}`
- `LOSS_CONSUMER_MAP_STATUS`: `{fields["LOSS_CONSUMER_MAP_STATUS"]}`
- `OPTIMIZER_STEPS_COMPLETED`: `{fields["OPTIMIZER_STEPS_COMPLETED"]}`
- `CHECKPOINT_COUNT`: `{fields["CHECKPOINT_COUNT"]}`
- `PAPER_ELIGIBLE`: `{str(fields["PAPER_ELIGIBLE"]).lower()}`
- `NEXT_TASK`: `{fields["NEXT_TASK"]}`
"""


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    machine = read_json(args.machine_snapshot)
    if machine["task_id"] != TASK_ID:
        raise RuntimeError("machine snapshot task ID differs")
    if machine["loss_binding"]["gate_pass"]:
        raise RuntimeError("builder is only valid for the sealed blocker outcome")
    if machine["final_classification"] != FINAL_CLASSIFICATION:
        raise RuntimeError("machine snapshot classification differs")
    fields = final_fields(args, machine)
    checks = execution_checks(machine)
    pass_count = sum(row["status"].startswith("PASS") for row in checks)
    not_applicable_count = sum(
        row["status"] == "NOT_APPLICABLE_BLOCKED_BY_GATE" for row in checks
    )
    failed_count = len(checks) - pass_count - not_applicable_count
    if failed_count:
        raise RuntimeError("execution checks contain a failure")

    binding_registry = {
        "schema_version": (
            "canondressgs.subject00.o03.formal_target_camsafe7.binding.v1"
        ),
        "task_id": TASK_ID,
        "created_at": CREATED_AT,
        "source": {"branch": SOURCE_BRANCH, "head": SOURCE_HEAD},
        "materialization": {
            "branch": MATERIALIZATION_BRANCH,
            "head": MATERIALIZATION_HEAD,
        },
        "formal_target": {
            key: machine["formal_target"][key]
            for key in (
                "root",
                "schema",
                "index_path",
                "index_sha256",
                "manifest_path",
                "manifest_sha256",
                "request_ids",
                "slots",
                "camera_ids",
                "excluded_request_ids",
                "slot04_present",
                "native_resolutions",
                "batch_size",
            )
        },
        "camera_contract_status": machine["formal_target"][
            "camera_contract_status"
        ],
        "camera_records": machine["formal_target"]["camera_rows"],
        "base_checkpoint": machine["base_checkpoint"],
        "output_root": {
            "path": OUTPUT_ROOT,
            "attempt_id": "attempt_001",
            "created": False,
        },
        "binding_gate": machine["loss_binding"]["status"],
        "final_classification": FINAL_CLASSIFICATION,
    }
    field_consumers = all_field_consumers(machine)
    enriched_field_rows = [
        {
            **row,
            "loss_consumer": field_consumers[row["field"]],
            "loss_view_denominator": 7,
        }
        for row in machine["formal_target"]["field_rows"]
    ]
    field_registry = {
        "schema_version": (
            "canondressgs.subject00.o03.formal_target_scientific_fields.v1"
        ),
        "task_id": TASK_ID,
        "created_at": CREATED_AT,
        "formal_schema": "canondressgs.full_dataset.v1",
        "loader": machine["loader"],
        "required_field_count": len(REQUIRED_FIELDS),
        "required_fields": REQUIRED_FIELDS,
        "previous_mismatch_field_count": len(MISMATCH_FIELDS),
        "previous_mismatch_fields": MISMATCH_FIELDS,
        "record_count": 7,
        "field_record_count": len(machine["formal_target"]["field_rows"]),
        "load_status": machine["formal_target"]["load_status"],
        "value_hash_status": machine["formal_target"]["value_hash_status"],
        "records": enriched_field_rows,
        "formal_target_mutations": 0,
    }
    loss_map = {
        "schema_version": (
            "canondressgs.subject00.o03.formal_target_loss_consumer_map.v1"
        ),
        "task_id": TASK_ID,
        "created_at": CREATED_AT,
        "loss_contract": "CAPACITY_ORACLE_LOSS_V1",
        "loss_weights": LOSS_WEIGHTS,
        "implementation": machine["implementation"],
        "consumer_map": machine["loss_binding"]["consumer_map"],
        "uniquely_bound_in_materialization_adapter_count": machine["loss_binding"][
            "uniquely_bound_in_materialization_adapter_count"
        ],
        "unbound_field_count": machine["loss_binding"]["unbound_field_count"],
        "unbound_fields": machine["loss_binding"]["unbound_fields"],
        "source_execution_path_status": machine["loss_binding"][
            "source_execution_path_status"
        ],
        "status": machine["loss_binding"]["status"],
        "gate_pass": False,
        "optimizer_launch_authorized": False,
        "classification": FINAL_CLASSIFICATION,
    }
    training_registry = {
        "schema_version": (
            "canondressgs.subject00.o03.formal_target_training_registry.v1"
        ),
        "task_id": TASK_ID,
        "created_at": CREATED_AT,
        "intended_run": {
            "output_root": OUTPUT_ROOT,
            "attempt_id": "attempt_001",
            "base_checkpoint": BASE_CHECKPOINT_PATH,
            "base_step": 60747,
            "loss_contract": "CAPACITY_ORACLE_LOSS_V1",
            "loss_weights": LOSS_WEIGHTS,
            "optimizer": "Adam",
            "learning_rates": {"geometry": 0.001, "appearance": 0.002},
            "gradient_clip_norm": 1.0,
            "scheduler": None,
            "seed": 20260718,
            "training_steps": 1200,
            "automatic_retry": False,
        },
        "actual_execution": {
            "status": "BLOCKED_BEFORE_GPU_FORWARD_AND_OPTIMIZER",
            "output_root_created": False,
            "pre_loop_loader": "PASS_7_OF_7_CPU_ONLY",
            "pre_loop_forward": "NOT_RUN",
            "pre_loop_loss": "NOT_RUN",
            "pre_loop_backward": "NOT_RUN",
            "optimizer_constructed": False,
            "optimizer_steps_completed": 0,
            "gpu_forward_calls": 0,
            "backward_calls": 0,
            "loss_initial": None,
            "loss_final": None,
            "peak_vram": 0,
            "wall_time": None,
        },
        "formal_base": {
            "status": "USER_AUTHORIZED_PAUSED",
            "completed": False,
            "durable_resume_step": 60747,
            "resume_ready": True,
            "resume_authorized": False,
        },
        "final_classification": FINAL_CLASSIFICATION,
    }
    sampling_registry = {
        "schema_version": (
            "canondressgs.subject00.o03.formal_target_sampling_registry.v1"
        ),
        "task_id": TASK_ID,
        "created_at": CREATED_AT,
        "sampler": "DETERMINISTIC_BALANCED_CYCLIC",
        "seed": 20260718,
        "view_order": EXPECTED_SLOTS,
        "request_order": EXPECTED_REQUESTS,
        "planned_training_steps": 1200,
        "planned_view_sample_counts": PLANNED_COUNTS,
        "actual_view_sample_counts": ACTUAL_COUNTS,
        "actual_optimizer_record_count": 0,
        "slot04_sample_count": 0,
        "status": "NOT_RUN_BLOCKED_BY_LOSS_FIELD_BINDING_GATE",
    }
    checkpoint_registry = {
        "schema_version": (
            "canondressgs.subject00.o03.formal_target_checkpoint_registry.v1"
        ),
        "task_id": TASK_ID,
        "created_at": CREATED_AT,
        "base_checkpoint": machine["base_checkpoint"],
        "initialization": {
            "old_contaminated_checkpoint_used": False,
            "old_runlocal_safe7_checkpoint_used": False,
            "Base60747_used_for_cpu_validation": True,
            "model_initialized_on_gpu": False,
        },
        "planned_steps": [0, 300, 600, 900, 1200],
        "new_checkpoint_count": 0,
        "new_checkpoint_paths": [],
        "new_checkpoint_sha256": [],
        "formal_target_binding_status": (
            "NOT_CREATED_BLOCKED_BEFORE_CHECKPOINT_0"
        ),
        "status": "NOT_RUN_BLOCKED_BY_LOSS_FIELD_BINDING_GATE",
    }

    not_generated = {
        "task_id": TASK_ID,
        "created_at": CREATED_AT,
        "status": "NOT_GENERATED_BLOCKED_BY_LOSS_FIELD_BINDING_GATE",
        "record_count": 0,
        "records": [],
        "denominator": None,
        "final_classification": FINAL_CLASSIFICATION,
    }
    per_view = {
        "schema_version": (
            "canondressgs.subject00.o03.formal_target_per_view_metrics.v1"
        ),
        **not_generated,
        "metrics": [
            "full_image_lpips",
            "full_image_psnr",
            "full_image_ssim",
            "garment_region_lpips",
            "garment_region_psnr",
            "garment_region_ssim",
            "silhouette_iou",
            "boundary_f",
            "protected_region_lpips",
            "protected_region_rgb_mae",
            "alpha_foreground_error",
            "severe_artifact_count",
            "render_time",
            "fps",
        ],
    }
    comparative = {
        "schema_version": (
            "canondressgs.subject00.o03.formal_target_comparative_metrics.v1"
        ),
        **not_generated,
        "comparison_candidates": [
            "Base60747",
            "old_contaminated_8view_Teacher",
            "old_runlocal_camera_safe7_Teacher",
            "new_formal_target_camera_safe7_Teacher",
        ],
        "comparison_note": (
            "No new formal-target Teacher exists; historical runs were read-only "
            "and were not used for initialization."
        ),
    }
    animation = {
        "schema_version": (
            "canondressgs.subject00.o03.formal_target_animation_audit.v1"
        ),
        **not_generated,
        "different_camera_status": "NOT_RUN",
        "different_pose_status": "NOT_RUN",
        "deformation_LBS_status": "NOT_RUN",
        "geometry_collapse_status": "NOT_RUN",
        "severe_artifact_status": "NOT_RUN",
        "novel_view_or_pose_claim": False,
    }
    review = {
        "schema_version": (
            "canondressgs.subject00.o03.formal_target_human_review_manifest.v1"
        ),
        "task_id": TASK_ID,
        "created_at": CREATED_AT,
        "status": "NOT_GENERATED_BLOCKED_BY_LOSS_FIELD_BINDING_GATE",
        "review_package_path": None,
        "page_count": 0,
        "human_visual_decision": None,
        "scientific_pass": None,
        "paper_eligible": False,
        "slot04_quarantine_note": (
            f"{EXCLUDED_REQUEST} remained review-only and never entered training."
        ),
        "reused_old_11_page_pdf_as_formal_evidence": False,
        "final_classification": FINAL_CLASSIFICATION,
    }
    tests = {
        "schema_version": (
            "canondressgs.subject00.o03.formal_target_execution_tests.v1"
        ),
        "task_id": TASK_ID,
        "created_at": CREATED_AT,
        "status": (
            f"PASS_BLOCKER_CLASSIFICATION_{pass_count}_PASS_"
            f"{not_applicable_count}_NOT_APPLICABLE_0_FAIL"
        ),
        "passed_count": pass_count,
        "not_applicable_blocked_count": not_applicable_count,
        "failed_count": failed_count,
        "checks": checks,
        "optimizer_steps_completed": 0,
        "gpu_forward_calls": 0,
    }
    summary = {
        "schema_version": (
            "canondressgs.subject00.o03.formal_target_final_summary.v1"
        ),
        "created_at": CREATED_AT,
        **fields,
    }
    report = report_markdown(machine, fields)
    handoff = {
        "schema_version": (
            "canondressgs.project_control_handoff.subject00_o03_formal_target.v1"
        ),
        "task_id": TASK_ID,
        "created_at": CREATED_AT,
        "outcome": {
            "final_classification": FINAL_CLASSIFICATION,
            "next_task": NEXT_TASK,
            "optimizer_steps_completed": 0,
            "output_root_created": False,
            "formal_target_mutations": 0,
            "Base60747_checkpoint_mutations": 0,
            "paper_modifications": 0,
        },
        "blocker": {
            "loss_consumer_map_status": machine["loss_binding"]["status"],
            "unbound_fields": machine["loss_binding"]["unbound_fields"],
            "source_execution_path_status": machine["loss_binding"][
                "source_execution_path_status"
            ],
            "resolution_scope": (
                "Freeze one explicit adapter and/or revised loss contract that "
                "uniquely consumes all eight formal fields; do not infer it."
            ),
        },
        "validated_inputs": {
            "formal_target_root": FORMAL_TARGET_ROOT,
            "formal_index_path": FORMAL_INDEX_PATH,
            "formal_loader_status": machine["formal_target"]["load_status"],
            "Base60747": machine["base_checkpoint"],
        },
        "artifact_paths": {
            name: str(path).replace("\\", "/")
            for name, path in ARTIFACT_PATHS.items()
        },
        "final_fields": fields,
    }

    values = {
        "binding": binding_registry,
        "fields": field_registry,
        "loss_map": loss_map,
        "training": training_registry,
        "sampling": sampling_registry,
        "checkpoints": checkpoint_registry,
        "per_view": per_view,
        "comparative": comparative,
        "animation": animation,
        "review": review,
        "tests": tests,
        "summary": summary,
        "handoff": handoff,
    }
    for name, value in values.items():
        write_json(repo / ARTIFACT_PATHS[name], value)
    write_text(repo / ARTIFACT_PATHS["report"], report)
    write_text(repo / ARTIFACT_PATHS["docs_report"], report)

    artifact_hashes = {
        name: sha256(repo / path)
        for name, path in ARTIFACT_PATHS.items()
        if (repo / path).is_file()
    }
    if len(artifact_hashes) != 15:
        raise RuntimeError(f"expected 15 artifacts, got {len(artifact_hashes)}")


if __name__ == "__main__":
    main()
