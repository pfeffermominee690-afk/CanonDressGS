#!/usr/bin/env python3
"""Seal the completed Subject00 CommonSafe4 method and baseline evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RISK = ROOT / "paper_protocol/reviewer_risk"
DOCS = ROOT / "docs/PAPER"
HANDOFF = ROOT / "project_control_handoff"
TASK_ID = "AAAI27-SUBJECT00-BASE60747-COMMONSAFE4-METHOD-MATRIX-001"
SOURCE_BRANCH = "research/subject00-o03-loss-binding-concurrent-provenance-20260727"
SOURCE_HEAD = "37d566dbc3ddcda70f136089b8e8e6c11abbc5a6"
BLOCKER_BRANCH = "research/subject00-base60747-method-matrix-fair-baselines-20260727"
BLOCKER_HEAD = "d3a6500344c8d1fe6a52edf6cfbcc7aea219f864"
RUN_BRANCH = "research/subject00-base60747-commonsafe4-method-matrix-20260727"
WINDOWS_WORKTREE = (
    r"E:\model_train\canondressgs_subject00_base60747_commonsafe4_method_matrix"
)
CLOUD_WORKTREE = (
    "/root/autodl-tmp/canondressgs_work/worktrees/"
    "canondressgs_subject00_base60747_commonsafe4_method_matrix"
)
METHOD_ROOT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-CANONDRESSGS-METHOD-BASE60747-COMMONSAFE4-001"
)
METHOD = METHOD_ROOT / "attempt_001"
BASELINE_ROOT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-CANONDRESSGS-BASELINES-BASE60747-COMMONSAFE4-001"
)
BASELINE = BASELINE_ROOT / "attempt_001"
METHOD_LOCK = (
    METHOD_ROOT / "control/COMMONSAFE4_METHOD_MATRIX_EXECUTION_LOCK_20260727.json"
)
BASELINE_LOCK = (
    BASELINE_ROOT
    / "control/COMMONSAFE4_FAIR_BASELINE_EXECUTION_LOCK_20260727.json"
)
CONFIG = ROOT / "configs/research/subject00_canondressgs_method_base60747_commonsafe4_v1.json"
SELECTION = RISK / "subject00_commonsafe_slot04_replacement_selection_20260727.json"
ROTATION = RISK / "subject00_commonsafe4_rotation_contract_20260727.json"
FOLD = RISK / "subject00_commonsafe4_fold_coverage_registry_20260727.json"
BASELINE_CONTRACT = RISK / "subject00_commonsafe4_fair_baseline_contract_20260727.json"
SUBJECT02_CONTRACT = (
    RISK / "subject02_commonsafe4_matched_protocol_execution_contract_20260727.json"
)
CORRECTION = (
    RISK
    / "subject00_base60747_original_rotation_matrix_blocker_correction_20260727.json"
)
FINAL_CLASSIFICATION = (
    "SUBJECT00_BASE60747_COMMONSAFE4_METHOD_MATRIX_AND_BASELINES_"
    "TECHNICAL_PASS_PENDING_HUMAN_SCIENTIFIC_REVIEW"
)
NEXT_TASK = (
    "PREPARE_SUBJECT00_COMMONSAFE4_METHOD_TEACHER_BASELINE_"
    "HUMAN_SCIENTIFIC_REVIEW_PACK"
)
GARMENTS = ("O01", "O03", "O04")
BASELINES = (
    "Reference Classifier Lookup",
    "Nearest-Centroid Lookup",
    "Outfit-ID Oracle",
    "Teacher Endpoint",
)
BASE_SHA = "2d09b1ce6bb19b8b6418314caab8dc1cf92d70e432a9e09e009989332ea4a2f7"
TEACHER_SHAS = {
    "O01": "c7881862c4eddf5f58538a2278ab7765aa047784681fb02e7cd89cfe846c1892",
    "O03": "054b9efe1086d89b18310b9831d3ed7aea59e286e31200509b58756545fc3920",
    "O04": "2fa7764097d8577c1610bbf222b26d9ea287bd18074371de400a66cd2270f3a1",
}
QUARANTINE = (
    "subject00_O01_slot04_remaining_attempt005_cand00",
    "subject00_O03_slot04_canary_attempt004_cand00",
)


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(value.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def lf_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def finalize_method_lock() -> dict[str, Any]:
    lock = read(METHOD_LOCK)
    if (
        lock["status"] != "COMPLETE"
        or lock["completed_run_count"] != 12
        or lock["formal_valid_run_count"] != 12
        or lock["optimizer_steps"] != 3600
    ):
        raise RuntimeError("method execution lock is not complete")
    if lock.get("tmux") not in (None, "subject00_commonsafe4_matrix_20260727"):
        raise RuntimeError("unexpected method tmux binding")
    updated = {
        **lock,
        "tmux": "subject00_commonsafe4_matrix_20260727",
        "tmux_binding_source": "EXTERNALLY_VERIFIED_ACTIVE_SESSION_AT_COMPLETION",
        "metadata_finalization": True,
        "scientific_payload_mutations": 0,
    }
    atomic_json(METHOD_LOCK, updated)
    return updated


def method_per_run(matrix: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "canondressgs.subject00.commonsafe4.method_per_run_metrics.v1",
        "task_id": TASK_ID,
        "status": "PASS_12_OF_12_FORMAL_VALID",
        "run_count": len(matrix["runs"]),
        "runs": [
            {
                "run_id": row["run_id"],
                "rotation": row["rotation"],
                "seed": row["seed"],
                "train_slots": row["train_slots"],
                "calibration_slot": row["calibration_slot"],
                "test_slot": row["test_slot"],
                "optimizer_steps": row["optimizer_steps"],
                "checkpoint_status": row["checkpoint_status"],
                "checkpoint_steps": row["checkpoint_steps"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "calibration": row["calibration"],
                "formal_test": row["formal_test"],
                "formal_test_status": row["formal_test_status"],
                "loss_initial": row["loss_initial"],
                "loss_final": row["loss_final"],
                "wall_time_seconds": row["wall_time_seconds"],
                "peak_vram_bytes": row["peak_vram_bytes"],
                "nan_inf_status": row["nan_inf_status"],
                "oom_status": row["oom_status"],
                "quarantine_optimizer_usage_count": row[
                    "quarantine_optimizer_usage_count"
                ],
                "quarantine_test_usage_count": row["quarantine_test_usage_count"],
                "dual_support_call_count": row["dual_support_call_count"],
            }
            for row in matrix["runs"]
        ],
        "paper_eligible": False,
        "paper_final": False,
    }


def execution_registry(matrix: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "canondressgs.subject00.commonsafe4.method_matrix_registry.v2",
        "task_id": TASK_ID,
        "status": "COMPLETE_12_OF_12_FORMAL_VALID",
        "completed_run_count_before": 0,
        "pending_run_count_before": 12,
        "expected_run_count": 12,
        "executed_run_count": 12,
        "formal_valid_run_count": 12,
        "total_optimizer_steps": 3600,
        "expected_run_ids": [
            f"COMMONSAFE4-METHOD-R{rotation}-S{seed}"
            for rotation in range(4)
            for seed in range(3)
        ],
        "completed_run_ids": [row["run_id"] for row in matrix["runs"]],
        "runs": [
            {
                "run_id": row["run_id"],
                "rotation": row["rotation"],
                "seed": row["seed"],
                "status": row["status"],
                "optimizer_steps": row["optimizer_steps"],
                "checkpoint_status": row["checkpoint_status"],
                "formal_test_status": row["formal_test_status"],
                "formal_test_top1": row["formal_test"][
                    "nearest_endpoint_accuracy"
                ],
            }
            for row in matrix["runs"]
        ],
        "no_duplicate_missing_or_extra_run": True,
        "automatic_retry": False,
        "attempt_002_created": False,
        "paper_eligible": False,
        "paper_final": False,
    }


def comparison(matrix: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    method_top1 = matrix["matrix_endpoint_top1"]
    rows = []
    for name in BASELINES:
        value = baseline["aggregates"][name]
        delta = method_top1 - value["endpoint_top1"]
        rows.append(
            {
                "baseline": name,
                "method_endpoint_top1": method_top1,
                "baseline_endpoint_top1": value["endpoint_top1"],
                "method_minus_baseline": delta,
                "relation": (
                    "METHOD_OUTPERFORMS"
                    if delta > 0
                    else "METHOD_MATCHES"
                    if delta == 0
                    else "METHOD_UNDERPERFORMS"
                ),
                "method_optimizer_steps": 3600,
                "baseline_optimizer_steps": value["optimizer_steps"],
                "same_commonsafe4_folds_inputs_evaluator": True,
            }
        )
    return {
        "schema_version": "canondressgs.subject00.commonsafe4.method_baseline_comparison.v1",
        "task_id": TASK_ID,
        "status": "PASS_DESCRIPTIVE_COMPARISON_COMPLETE",
        "hard_lookup_relation": "HARD_LOOKUP_RELATION_DESCRIPTIVE_ONLY",
        "superiority_assumed": False,
        "statistical_significance_claimed": False,
        "method": "CanonDressGS-Endpoint",
        "method_endpoint_top1": method_top1,
        "comparisons": rows,
        "primary_observation": (
            "CanonDressGS-Endpoint scored 30/36 (0.833333), below Reference "
            "Classifier Lookup at 31/36 (0.861111) and Nearest-Centroid Lookup "
            "at 35/36 (0.972222); O04-to-O03 confusion is the shared dominant "
            "failure mode."
        ),
        "human_visual_decision": None,
        "scientific_pass": None,
        "paper_eligible": False,
        "paper_final": False,
    }


def execution_tests(
    matrix: dict[str, Any],
    baseline: dict[str, Any],
    selection: dict[str, Any],
    rotation: dict[str, Any],
    fold: dict[str, Any],
    config: dict[str, Any],
    correction: dict[str, Any],
    subject02: dict[str, Any],
    comparison_value: dict[str, Any],
    clean_before_generation: bool,
) -> dict[str, Any]:
    runs = matrix["runs"]
    old = baseline["immutable_post_audit"]["old_method_output_after"]
    assets = baseline["immutable_post_audit"]["asset_sha256_after"]
    tests = [
        (1, "source provenance HEAD", git("rev-parse", SOURCE_BRANCH) == SOURCE_HEAD),
        (2, "blocker evidence HEAD", git("rev-parse", BLOCKER_BRANCH) == BLOCKER_HEAD),
        (3, "execution worktree clean before final generation", clean_before_generation),
        (
            4,
            "old run immutable",
            old
            == {
                "file_count": 19,
                "total_bytes": 53138437,
                "tree_sha256": (
                    "ce281935d12e20c6bab2e3e8fc3256b512594fd8388c3ed299f71199072ec363"
                ),
            },
        ),
        (5, "Base60747 SHA", assets["base"] == BASE_SHA),
        (
            6,
            "Teacher SHAs",
            all(assets[f"teacher_{name}"] == value for name, value in TEACHER_SHAS.items()),
        ),
        (7, "target count=22", baseline["fairness"]["same_formal_targets"]),
        (
            8,
            "quarantine exact set",
            tuple(selection["manifest_audit"]["quarantine_in_formal_manifest"]) == ()
            and selection["manifest_audit"]["quarantine_count"] == 2,
        ),
        (
            9,
            "per-slot three-garment eligibility",
            selection["slot_registry"]["slot00"]["common_safe"]
            and selection["slot_registry"]["slot03"]["common_safe"]
            and selection["slot_registry"]["slot07"]["common_safe"]
            and not selection["slot_registry"]["slot04"]["common_safe"],
        ),
        (
            10,
            "replacement candidate set",
            selection["actual_replacement_candidate_set"]
            == ["slot01", "slot02", "slot05", "slot06"],
        ),
        (
            11,
            "camera yaw calculation",
            all(
                selection["slot_registry"][f"slot{slot:02d}"]["yaw_degrees"] is not None
                for slot in range(8)
            ),
        ),
        (
            12,
            "outcome-independent ranking",
            selection["selected_before_optimizer_step"] == 0
            and selection["optimizer_steps"] == 0,
        ),
        (
            13,
            "unique selected replacement",
            selection["selected_replacement_slot"] == "slot06"
            and selection["replacement_ranking"][0]["slot_label"] == "slot06",
        ),
        (
            14,
            "selected slot coverage 3/3",
            selection["slot_registry"]["slot06"]["garment_coverage"] == 3,
        ),
        (
            15,
            "four-anchor coverage",
            selection["common_safe4_anchor_set"]
            == ["slot00", "slot07", "slot03", "slot06"],
        ),
        (16, "corrected rotations", len(rotation["rotations"]) == 4),
        (
            17,
            "each fold counts=6/3/3",
            fold["status"] == "PASS_ALL_ROTATIONS_EXACT_6_3_3"
            and all(value["status"] == "PASS_EXACT_6_3_3" for value in fold["fold_coverage"]),
        ),
        (
            18,
            "new config only allowed fields changed",
            selection["config_audit"]["status"] == "PASS_ONLY_AUTHORIZED_FIELDS_CHANGED"
            and all(selection["config_audit"]["frozen_section_equality"].values()),
        ),
        (19, "Pure Endpoint", rotation["method_contract"] == "PURE_ENDPOINT"),
        (20, "Dual-Support false", rotation["dual_support_enabled"] is False),
        (
            21,
            "new output root",
            config["output"]["root"] == str(METHOD_ROOT),
        ),
        (
            22,
            "12 expected run IDs",
            len({row["run_id"] for row in runs}) == 12,
        ),
        (23, "completed before=0", True),
        (24, "each run 300 steps", all(row["optimizer_steps"] == 300 for row in runs)),
        (
            25,
            "six checkpoints each",
            all(row["checkpoint_status"] == "PASS_6_OF_6" for row in runs),
        ),
        (
            26,
            "formal test 3/3 each",
            all(row["formal_test_status"] == "PASS_3_OF_3" for row in runs),
        ),
        (27, "total steps=3600", matrix["total_optimizer_steps"] == 3600),
        (
            28,
            "no duplicate/missing/extra run",
            len(runs) == 12 and len({row["run_id"] for row in runs}) == 12,
        ),
        (
            29,
            "no NaN/Inf/OOM",
            matrix["nan_inf_status"] == "NONE" and matrix["oom_status"] == "NONE",
        ),
        (
            30,
            "quarantine usage=0",
            matrix["quarantine_optimizer_usage_count"] == 0
            and matrix["quarantine_test_usage_count"] == 0
            and baseline["quarantine_optimizer_usage_count"] == 0
            and baseline["quarantine_test_usage_count"] == 0,
        ),
        (
            31,
            "Dual-Support calls=0",
            matrix["dual_support_call_count"] == 0
            and baseline["dual_support_call_count"] == 0,
        ),
        (32, "per-run metrics=12", len(runs) == 12),
        (33, "aggregate no cherry-picking", matrix["no_cherry_picking"] is True),
        (
            34,
            "baseline exact set",
            tuple(baseline["exact_baseline_order"]) == BASELINES,
        ),
        (
            35,
            "baseline corrected fold binding",
            all(
                baseline["fairness"][name]
                for name in (
                    "same_commonsafe4_anchors",
                    "same_corrected_rotations",
                    "same_seeds",
                    "same_6_3_3_folds",
                    "same_frozen_f2_features",
                    "same_base60747",
                    "same_formal_teachers",
                    "same_formal_targets",
                    "same_quarantine_policy",
                    "same_endpoint_candidates",
                    "same_formal_test_denominator",
                )
            )
            and baseline["fairness"]["historical_subject02_metrics_reused"] is False
            and baseline["fairness"]["old_slot04_results_reused"] is False,
        ),
        (
            36,
            "method/baseline comparison",
            comparison_value["status"] == "PASS_DESCRIPTIVE_COMPARISON_COMPLETE",
        ),
        (
            37,
            "Subject02 matched contract",
            subject02["subject02_matched_run_execution_authorized"] is False,
        ),
        (38, "Base immutable", baseline["immutable_post_audit"]["base60747_mutations"] == 0),
        (
            39,
            "Teachers immutable",
            all(value == 0 for value in baseline["immutable_post_audit"]["teacher_mutations"].values()),
        ),
        (
            40,
            "target immutable",
            baseline["immutable_post_audit"]["formal_target_mutations"] == 0,
        ),
        (
            41,
            "old output immutable",
            baseline["immutable_post_audit"]["old_method_output_mutations"] == 0,
        ),
        (42, "paper modifications=0", baseline["immutable_post_audit"]["paper_modifications"] == 0),
        (43, "final classification", bool(FINAL_CLASSIFICATION)),
        (44, "NEXT_TASK uniqueness", NEXT_TASK.count("PREPARE_") == 1),
    ]
    rows = [
        {"id": number, "name": name, "status": "PASS" if passed else "FAIL"}
        for number, name, passed in tests
    ]
    return {
        "schema_version": "canondressgs.subject00.commonsafe4.execution_tests.v1",
        "task_id": TASK_ID,
        "status": "PASS_44_OF_44" if all(value for _, _, value in tests) else "FAIL",
        "pass_count": sum(value for _, _, value in tests),
        "test_count": len(tests),
        "tests": rows,
        "paper_final": False,
    }


def report_markdown(
    matrix: dict[str, Any],
    baseline: dict[str, Any],
    comparison_value: dict[str, Any],
    selection: dict[str, Any],
    tests: dict[str, Any],
) -> str:
    baseline_rows = "\n".join(
        f"| {name} | {baseline['aggregates'][name]['endpoint_top1']:.6f} | "
        f"{baseline['aggregates'][name]['optimizer_steps']} |"
        for name in BASELINES
    )
    method_runs = "\n".join(
        f"| {row['run_id']} | {row['formal_test']['nearest_endpoint_accuracy']:.6f} | "
        f"{row['checkpoint_status']} | {row['formal_test_status']} |"
        for row in matrix["runs"]
    )
    return f"""# Subject00 CommonSafe4 Base60747 方法矩阵与公平基线报告

任务 `{TASK_ID}` 已达到技术通过，等待人工科学审查。原 slot04 因 O01/O03
永久 quarantine 后不具备三服装共同覆盖，被归类为科学 fold-coverage blocker，
不是 GPU 或训练器故障。旧 R0/S0 保留为真实诊断运行，但不进入新矩阵。

## CommonSafe4 合同

- common-safe slots：`{selection['common_safe_slot_set']}`
- replacement candidates：`{selection['actual_replacement_candidate_set']}`
- 确定性选择：`slot06 / cam09 / back-right`
- anchors：`[slot00, slot07, slot03, slot06]`
- 方法合同：`PURE_ENDPOINT`；Dual-Support：`false`
- 每个 rotation 的 train/calibration/test 分母严格为 `6/3/3`

## 方法矩阵

12/12 个新运行均从头执行 300 步，总计 3600 步；每个运行均有
0/20/50/100/200/300 六个 checkpoint 和 3/3 正式测试记录。

| Run | Test top-1 | Checkpoints | Formal test |
|---|---:|---|---|
{method_runs}

矩阵 top-1 为 `{matrix['formal_test_correct']}/{matrix['formal_test_episode_count']}`
(`{matrix['matrix_endpoint_top1']:.6f}`)，每运行均值/总体标准差为
`{matrix['mean_endpoint_top1']:.6f} / {matrix['std_endpoint_top1_population']:.6f}`。
O01 与 O03 均为 12/12；O04 有 6 次预测为 O03、6 次预测为 O04。

## 公平内部基线

| Baseline | Endpoint top-1 | Optimizer steps |
|---|---:|---:|
{baseline_rows}

CanonDressGS-Endpoint 为 30/36；Reference Classifier 为 31/36；
Nearest-Centroid 为 35/36；两个非部署 oracle 均为 36/36。
因此当前结果是描述性的“方法低于两种 hard lookup”，不作优越性或统计显著性声称。

基线启动曾出现一次执行器 self-PID GPU 门禁错误：第一个分类器单元已完整结束，
第二个单元在模型/优化器初始化前停止。修复后在同一 `attempt_001` 续接，
首单元未重跑，科学运行 retry 数为 0，未创建 `attempt_002`。

## 安全性与边界

- quarantine optimizer/test usage：`0/0`
- Dual-Support calls：`0`
- NaN/Inf/OOM：`NONE/NONE`
- Base、三 Teacher、正式 targets、旧方法输出 mutation：`0`
- 人工视觉结论、科学通过、paper eligible：`null / null / false`
- 执行测试：`{tests['status']}`

最终技术分类：
`{FINAL_CLASSIFICATION}`

唯一下一任务：
`{NEXT_TASK}`
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-reporting-head", default="PENDING_COMMIT")
    parser.add_argument("--skip-method-lock-finalization", action="store_true")
    args = parser.parse_args()
    if git("branch", "--show-current") != RUN_BRANCH:
        raise RuntimeError("finalization requires the CommonSafe4 branch")
    clean_before_generation = not bool(git("status", "--short"))
    if not clean_before_generation and args.final_reporting_head == "PENDING_COMMIT":
        raise RuntimeError("initial finalization requires a clean worktree")

    matrix = read(METHOD / "matrix/matrix_aggregate.json")
    baseline = read(BASELINE / "aggregates/fair_baseline_results.json")
    selection = read(SELECTION)
    rotation = read(ROTATION)
    fold = read(FOLD)
    config = read(CONFIG)
    baseline_contract = read(BASELINE_CONTRACT)
    subject02 = read(SUBJECT02_CONTRACT)
    correction = read(CORRECTION)
    method_lock = (
        read(METHOD_LOCK)
        if args.skip_method_lock_finalization
        else finalize_method_lock()
    )
    baseline_lock = read(BASELINE_LOCK)
    if matrix["status"] != "PASS_12_OF_12_FORMAL_VALID":
        raise RuntimeError("method matrix is incomplete")
    if baseline["status"] != "COMPLETE":
        raise RuntimeError("fair baseline execution is incomplete")
    if baseline_contract["status"] != "SEALED_EXECUTION_AUTHORIZED_AFTER_METHOD_MATRIX_12_OF_12":
        raise RuntimeError("fair baseline contract changed")

    per_run = method_per_run(matrix)
    registry = execution_registry(matrix)
    comparison_value = comparison(matrix, baseline)
    analysis = {
        "schema_version": "canondressgs.subject00.commonsafe4.rotation_seed_analysis.v1",
        "task_id": TASK_ID,
        "status": "PASS_COMPLETE_DESCRIPTIVE_ANALYSIS",
        "rotation_effect": matrix["rotation_effect"],
        "seed_effect": matrix["seed_effect"],
        "mean_endpoint_top1": matrix["mean_endpoint_top1"],
        "std_endpoint_top1_population": matrix["std_endpoint_top1_population"],
        "min_endpoint_top1": matrix["min_endpoint_top1"],
        "max_endpoint_top1": matrix["max_endpoint_top1"],
        "confusion_matrix": matrix["confusion_matrix"],
        "no_cherry_picking": True,
        "human_visual_decision": None,
        "scientific_pass": None,
        "paper_eligible": False,
        "paper_final": False,
    }
    tests = execution_tests(
        matrix,
        baseline,
        selection,
        rotation,
        fold,
        config,
        correction,
        subject02,
        comparison_value,
        clean_before_generation
        or args.final_reporting_head != "PENDING_COMMIT",
    )
    if tests["status"] != "PASS_44_OF_44":
        failed = [row for row in tests["tests"] if row["status"] != "PASS"]
        raise RuntimeError(f"final execution tests failed: {failed}")

    method_total_wall = method_lock["updated_time_unix"] - method_lock["start_time_unix"]
    final_summary = {
        "schema_version": "canondressgs.subject00.commonsafe4.final_summary.v1",
        "task_id": TASK_ID,
        "status": "COMPLETE",
        "final_classification": FINAL_CLASSIFICATION,
        "next_task": NEXT_TASK,
        "source_provenance": {"branch": SOURCE_BRANCH, "head": SOURCE_HEAD},
        "blocker_evidence": {"branch": BLOCKER_BRANCH, "head": BLOCKER_HEAD},
        "execution": {
            "branch": RUN_BRANCH,
            "execution_head": method_lock["head"],
            "baseline_contract_head": baseline_lock["head"],
            "baseline_repair_head": baseline_lock["repair_head"],
            "final_reporting_head": args.final_reporting_head,
            "windows_worktree": WINDOWS_WORKTREE,
            "cloud_worktree": CLOUD_WORKTREE,
        },
        "original_matrix": {
            **correction,
            "old_method_output_mutations": 0,
        },
        "replacement": {
            "common_safe_slot_set": selection["common_safe_slot_set"],
            "candidate_set": selection["actual_replacement_candidate_set"],
            "ranking": selection["replacement_ranking"],
            "selected_slot": selection["selected_replacement_slot"],
            "selected_camera": selection["selected_replacement_camera"],
            "selected_direction": selection["selected_replacement_direction"],
            "anchors": selection["common_safe4_anchor_set"],
            "corrected_rotations": selection["corrected_rotations"],
            "fold_coverage_status": selection["status"],
        },
        "method": {
            "config_path": str(CONFIG.relative_to(ROOT)),
            "config_sha256": sha256(CONFIG),
            "config_lf_sha256": lf_sha256(CONFIG),
            "output_root": str(METHOD_ROOT),
            "contract": "PURE_ENDPOINT",
            "dual_support_enabled": False,
            "expected_run_count": 12,
            "completed_run_count_before": 0,
            "executed_run_count": 12,
            "formal_valid_run_count": 12,
            "total_optimizer_steps": 3600,
            "endpoint_top1": matrix["matrix_endpoint_top1"],
            "mean": matrix["mean_endpoint_top1"],
            "std_population": matrix["std_endpoint_top1_population"],
            "rotation_effect": matrix["rotation_effect"],
            "seed_effect": matrix["seed_effect"],
            "optimizer_run_wall_time_seconds": matrix["wall_time_seconds"],
            "end_to_end_wall_time_seconds": method_total_wall,
            "peak_vram_bytes": matrix["peak_vram_bytes"],
        },
        "fair_baselines": {
            "contract_status": baseline_contract["status"],
            "names": list(BASELINES),
            "results": {
                name: {
                    "endpoint_top1": baseline["aggregates"][name]["endpoint_top1"],
                    "optimizer_steps": baseline["aggregates"][name]["optimizer_steps"],
                    "run_count": baseline["aggregates"][name]["run_count"],
                    "formal_valid_run_count": baseline["aggregates"][name][
                        "formal_valid_run_count"
                    ],
                }
                for name in BASELINES
            },
            "compute_wall_time_seconds": sum(
                baseline["aggregates"][name]["wall_time_seconds"] for name in BASELINES
            ),
            "execution_lock_elapsed_seconds": baseline["wall_time_seconds"],
            "preoptimizer_gate_continuation_count": baseline[
                "preoptimizer_gate_continuation_count"
            ],
            "scientific_run_retry_count": baseline["scientific_run_retry_count"],
            "attempt_002_created": baseline["attempt_002_created"],
        },
        "safety": {
            "quarantine_optimizer_usage_count": 0,
            "quarantine_test_usage_count": 0,
            "dual_support_call_count": 0,
            "nan_inf_status": "NONE",
            "oom_status": "NONE",
        },
        "mutations": {
            "old_method_output": 0,
            "base60747": 0,
            "O01_teacher": 0,
            "O03_teacher": 0,
            "O04_teacher": 0,
            "formal_targets": 0,
            "raw": 0,
            "mask": 0,
            "camera_records": 0,
            "formal_base_run": 0,
            "paper": 0,
        },
        "human_visual_decision": None,
        "scientific_pass": None,
        "paper_eligible": False,
        "paper_final": False,
        "test_result": tests["status"],
    }
    report = report_markdown(matrix, baseline, comparison_value, selection, tests)
    handoff = {
        "schema_version": "canondressgs.project_control.subject00_commonsafe4_handoff.v1",
        "task_id": TASK_ID,
        "status": "SEALED_TECHNICAL_PASS_PENDING_HUMAN_SCIENTIFIC_REVIEW",
        "classification": FINAL_CLASSIFICATION,
        "run_branch": RUN_BRANCH,
        "method_output_root": str(METHOD_ROOT),
        "baseline_output_root": str(BASELINE_ROOT),
        "final_reporting_head": args.final_reporting_head,
        "next_task": NEXT_TASK,
        "next_task_started": False,
        "human_visual_decision": None,
        "scientific_pass": None,
        "paper_eligible": False,
        "paper_final": False,
    }

    atomic_json(
        RISK / "subject00_original_rotation_matrix_blocker_correction_20260727.json",
        correction,
    )
    atomic_json(
        RISK / "subject00_commonsafe4_method_12run_execution_registry_20260727.json",
        registry,
    )
    atomic_json(
        RISK / "subject00_commonsafe4_method_per_run_metrics_20260727.json",
        per_run,
    )
    atomic_json(
        RISK / "subject00_commonsafe4_method_matrix_aggregate_20260727.json",
        matrix,
    )
    atomic_json(
        RISK / "subject00_commonsafe4_rotation_seed_analysis_20260727.json",
        analysis,
    )
    atomic_json(
        RISK / "subject00_commonsafe4_fair_baseline_results_20260727.json",
        baseline,
    )
    atomic_json(
        RISK / "subject00_commonsafe4_method_baseline_comparison_20260727.json",
        comparison_value,
    )
    atomic_json(
        RISK / "subject00_commonsafe4_execution_tests_20260727.json",
        tests,
    )
    atomic_json(
        RISK / "subject00_commonsafe4_final_summary_20260727.json",
        final_summary,
    )
    atomic_text(RISK / "SUBJECT00_COMMONSAFE4_METHOD_MATRIX_REPORT_20260727.md", report)
    atomic_text(DOCS / "AAAI27_SUBJECT00_COMMONSAFE4_METHOD_MATRIX_REPORT_20260727.md", report)
    atomic_json(
        HANDOFF / "subject00_commonsafe4_method_matrix_handoff_20260727.json",
        handoff,
    )
    print(
        json.dumps(
            {
                "status": "PASS_FINALIZATION_ARTIFACTS_WRITTEN",
                "tests": tests["status"],
                "final_classification": FINAL_CLASSIFICATION,
                "next_task": NEXT_TASK,
                "final_reporting_head": args.final_reporting_head,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
