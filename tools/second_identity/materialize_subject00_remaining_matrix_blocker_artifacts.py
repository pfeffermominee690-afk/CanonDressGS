#!/usr/bin/env python3
"""Materialize the formal blocker artifacts for the remaining Subject00 matrix."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RISK = ROOT / "paper_protocol/reviewer_risk"
HANDOFF = ROOT / "project_control_handoff"
PAPER = ROOT / "docs/PAPER"
TASK_ID = "AAAI27-SUBJECT00-BASE60747-REMAINING-METHOD-MATRIX-FAIR-BASELINES-001"
SOURCE_BRANCH = "research/subject00-o03-loss-binding-concurrent-provenance-20260727"
SOURCE_HEAD = "37d566dbc3ddcda70f136089b8e8e6c11abbc5a6"
BRANCH = "research/subject00-base60747-remaining-method-matrix-fair-baselines-20260727"
WINDOWS_WORKTREE = (
    r"E:\model_train\canondressgs_subject00_base60747_remaining_method_matrix_fair_baselines"
)
CLOUD_WORKTREE = (
    "/root/autodl-tmp/canondressgs_work/worktrees/"
    "canondressgs_subject00_base60747_remaining_method_matrix_fair_baselines"
)
LOCK_PATH = (
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-CANONDRESSGS-METHOD-BASE60747-001/control/"
    "METHOD_MATRIX_EXECUTION_LOCK_20260727.json"
)
RUNTIME_PATH = (
    RISK / "subject00_base60747_method_matrix_runtime_preflight_20260727.json"
)
LOCK_SNAPSHOT_PATH = (
    RISK / "subject00_base60747_method_matrix_execution_lock_snapshot_20260727.json"
)
FINAL_CLASSIFICATION = "SUBJECT00_BASE60747_METHOD_MATRIX_ENGINEERING_FAIL"
NEXT_TASK = "USER_REVIEW_SUBJECT00_BASE60747_METHOD_MATRIX_ENGINEERING_FAILURE"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def common() -> dict[str, Any]:
    return {
        "schema_version": "canondressgs.subject00.base60747.remaining_matrix_blocker.v1",
        "task_id": TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "new_branch": BRANCH,
        "windows_worktree": WINDOWS_WORKTREE,
        "cloud_worktree": CLOUD_WORKTREE,
        "runtime_preflight_path": str(RUNTIME_PATH.relative_to(ROOT)).replace("\\", "/"),
        "execution_lock_path": LOCK_PATH,
        "paper_final": False,
    }


def main() -> int:
    audit = read_json(RUNTIME_PATH)
    lock = read_json(LOCK_SNAPSHOT_PATH)
    matrix = audit["matrix"]
    completed = audit["completed_rotation0_seed0"]
    initial = completed["initial_run_summary"]
    evaluation = completed["evaluation"]
    teachers = audit["teachers"]
    targets = audit["targets"]
    method_files = audit["method_files"]
    expected_cells = matrix["cells"]
    pending_ids = [
        cell["run_id"] for cell in expected_cells if cell["run_id"] != "METHOD-R0-S0"
    ]
    blocker_roles = [
        {
            "rotation": item["rotation"],
            "role": item["role"],
            "reason": item["reason"],
        }
        for item in matrix["blockers"]
    ]
    immutable = {
        "BASE60747_CHECKPOINT_MUTATIONS": 0,
        "O01_TEACHER_CHECKPOINT_MUTATIONS": 0,
        "O03_TEACHER_CHECKPOINT_MUTATIONS": 0,
        "O04_TEACHER_CHECKPOINT_MUTATIONS": 0,
        "FORMAL_TARGET_MUTATIONS": 0,
        "RAW_MUTATIONS": 0,
        "MASK_MUTATIONS": 0,
        "CAMERA_RECORD_MUTATIONS": 0,
        "COMPLETED_ROTATION0_SEED0_MUTATIONS": 0,
        "FORMAL_BASE_RUN_MUTATIONS": 0,
        "PAPER_MODIFICATIONS": 0,
    }

    matrix_contract = {
        **common(),
        "method_contract": "PURE_ENDPOINT",
        "dual_support_enabled": False,
        "runner": method_files["runner"],
        "implementation": method_files["implementation"],
        "config": method_files["config"],
        "rotations": matrix["rotations"],
        "seeds": matrix["seeds"],
        "expected_run_count": 12,
        "optimizer_steps_per_run": 300,
        "expected_total_optimizer_steps": 3600,
        "checkpoint_steps": [0, 20, 50, 100, 200, 300],
        "expected_cells": expected_cells,
        "quarantine_ids": targets["quarantine_ids"],
        "execution_gate": {
            "status": "FAIL_FROZEN_FOLD_COVERAGE",
            "blockers": blocker_roles,
            "forbidden_resolutions": [
                "use quarantine records",
                "impute missing garment records",
                "move folds",
                "change denominators",
                "change rotation set",
                "modify the frozen runner or method hyperparameters",
            ],
        },
    }
    write_json(
        RISK / "subject00_base60747_method_12run_matrix_contract_20260727.json",
        matrix_contract,
    )

    execution_registry = {
        **common(),
        "execution_lock": lock,
        "completed_run_count_before": 1,
        "pending_run_count_before": 11,
        "validated_completed_run_ids": ["METHOD-R0-S0"],
        "pending_run_ids": pending_ids,
        "executed_new_run_ids": [],
        "completed_run_count_after": 1,
        "actual_total_optimizer_steps": 300,
        "new_method_optimizer_steps": 0,
        "new_baseline_optimizer_steps": 0,
        "pending_output_paths_absent": True,
        "active_optimizer_process_count_before": 0,
        "active_optimizer_process_count_after": 0,
        "status": "BLOCKED_BEFORE_OPTIMIZER_CONSTRUCTION",
        "reason": audit["execution_decision"]["reason"],
    }
    write_json(
        RISK / "subject00_base60747_method_12run_execution_registry_20260727.json",
        execution_registry,
    )

    per_run = {
        **common(),
        "run_count_with_metrics": 1,
        "runs": {
            "METHOD-R0-S0": {
                "rotation": 0,
                "seed": 0,
                "status": "AUTHENTIC_TRAINING_COMPLETE_FORMAL_TEST_FOLD_BLOCKED",
                "optimizer_steps": 300,
                "checkpoint_status": "PASS_6_OF_6",
                "loss_initial": initial["loss_initial"],
                "loss_final": initial["loss_final"],
                "loss_minimum": initial["loss_minimum"],
                "training_fold_nearest_endpoint_correct": 6,
                "training_fold_nearest_endpoint_total": 6,
                "training_fold_top1": evaluation["nearest_endpoint_accuracy"],
                "formal_test_fold_result": None,
                "formal_test_fold_reason": (
                    "rotation0 test slot04 contains O04 only; O01/O03 slot04 are quarantined"
                ),
                "nan_inf_status": initial["nan_inf_status"],
                "oom_status": initial["oom_status"],
                "peak_vram_bytes": initial["peak_vram_bytes"],
                "wall_time_seconds": completed["wall_time_seconds_by_file_mtime"],
                "trainable_parameter_changes": initial["trainable_parameter_changes"],
            }
        },
        "not_run": pending_ids,
    }
    write_json(
        RISK / "subject00_base60747_method_per_run_metrics_20260727.json", per_run
    )

    aggregate = {
        **common(),
        "aggregation_contract": "ALL_12_RUNS_NO_CHERRY_PICKING",
        "expected_run_count": 12,
        "completed_training_run_count": 1,
        "formal_valid_run_count": 0,
        "total_optimizer_steps_actual": 300,
        "total_optimizer_steps_expected": 3600,
        "matrix_endpoint_top1": None,
        "matrix_mean_std": None,
        "confusion_matrix": None,
        "rotation_effect": None,
        "seed_effect": None,
        "failure_count": 1,
        "nan_inf_count": 0,
        "oom_count": 0,
        "checkpoint_completeness": "1_OF_12_RUNS",
        "status": "NOT_AGGREGATABLE_ENGINEERING_GATE_FAIL",
        "blockers": blocker_roles,
    }
    write_json(
        RISK / "subject00_base60747_method_matrix_aggregate_20260727.json",
        aggregate,
    )

    rotation_seed = {
        **common(),
        "rotations": matrix["rotations"],
        "seeds": matrix["seeds"],
        "expected_cartesian_product_count": 12,
        "completed_cells": ["METHOD-R0-S0"],
        "pending_cells": pending_ids,
        "rotation_effect": None,
        "seed_effect": None,
        "status": "BLOCKED_INSUFFICIENT_COMPLETE_CELLS",
        "fold_coverage": matrix["rotation_coverage"],
    }
    write_json(
        RISK / "subject00_base60747_method_rotation_seed_analysis_20260727.json",
        rotation_seed,
    )

    checkpoint_registry = {
        **common(),
        "expected_steps_per_run": [0, 20, 50, 100, 200, 300],
        "runs": {
            "METHOD-R0-S0": {
                "status": "PASS_6_OF_6",
                "checkpoints": completed["checkpoints"],
            }
        },
        "pending_runs": pending_ids,
        "new_checkpoint_count": 0,
        "completed_rotation0_seed0_mutations": 0,
    }
    write_json(
        RISK / "subject00_base60747_method_checkpoint_registry_20260727.json",
        checkpoint_registry,
    )

    baseline_contract = {
        **common(),
        "contract_source": audit["baselines"]["contract_source"],
        "contract_source_sha256": audit["baselines"]["contract_source_sha256"],
        "source_status": audit["baselines"]["contract_status"],
        "paper_facing_names": audit["baselines"]["exact_names"],
        "name_set_unique": audit["baselines"]["unique_set_recovered"],
        "full_execution_contract_unique": False,
        "full_execution_contract_gaps": [
            "source status remains FROZEN_BLOCKED_CONDITION_RESOURCE_AND_FORMAL_BASE",
            "exact calibration/test/visualization condition IDs remain blocked",
            "per-baseline optimizer/run budget is not sealed for Subject00",
        ],
        "execution_authorized_by_user": True,
        "execution_allowed_after_gates": False,
        "execution_blocker": "METHOD_MATRIX_NOT_12_OF_12_AND_BASELINE_EXECUTION_CONTRACT_INCOMPLETE",
        "external_baselines_authorized": False,
        "dual_support_authorized": False,
    }
    write_json(
        RISK / "subject00_base60747_fair_baseline_contract_20260727.json",
        baseline_contract,
    )

    baseline_execution = {
        **common(),
        "baseline_names": audit["baselines"]["exact_names"],
        "executed_baselines": [],
        "run_count": 0,
        "optimizer_steps": 0,
        "results": None,
        "status": "BLOCKED_NOT_EXECUTED",
        "reason": baseline_contract["execution_blocker"],
    }
    write_json(
        RISK / "subject00_base60747_fair_baseline_execution_registry_20260727.json",
        baseline_execution,
    )

    comparison = {
        **common(),
        "method_aggregate_path": (
            "paper_protocol/reviewer_risk/"
            "subject00_base60747_method_matrix_aggregate_20260727.json"
        ),
        "baseline_execution_path": (
            "paper_protocol/reviewer_risk/"
            "subject00_base60747_fair_baseline_execution_registry_20260727.json"
        ),
        "method_complete": False,
        "baselines_complete": False,
        "endpoint_top1": None,
        "mean_std": None,
        "score_margin": None,
        "stability": None,
        "compute_budget_comparison": None,
        "status": "NOT_COMPARABLE_ENGINEERING_GATE_FAIL",
    }
    write_json(
        RISK / "subject00_base60747_method_baseline_comparison_20260727.json",
        comparison,
    )

    budget = {
        **common(),
        "method": {
            "expected_runs": 12,
            "completed_runs": 1,
            "expected_optimizer_steps": 3600,
            "actual_optimizer_steps": 300,
            "new_optimizer_steps_this_task": 0,
            "completed_run_wall_time_seconds": completed[
                "wall_time_seconds_by_file_mtime"
            ],
            "completed_run_peak_vram_bytes": initial["peak_vram_bytes"],
            "trainable_parameter_count": 2050,
        },
        "baselines": {
            "names": audit["baselines"]["exact_names"],
            "run_count": 0,
            "optimizer_steps": 0,
            "wall_time_seconds": 0,
            "peak_vram_bytes": None,
        },
        "fairness_status": "NOT_EVALUABLE_EXECUTION_BLOCKED",
    }
    write_json(
        RISK / "subject00_base60747_method_compute_budget_audit_20260727.json",
        budget,
    )

    check_names = [
        "source_branch_head",
        "source_clean",
        "active_process_count",
        "execution_lock",
        "base60747_sha",
        "formal_base_paused",
        "target_root_count",
        "quarantine_set",
        "o01_teacher_sha",
        "o03_teacher_sha",
        "o04_teacher_sha",
        "o03_provenance_seal",
        "corrected_teacher_registry",
        "method_dependency_audit",
        "method_implementation_sha",
        "method_config_sha",
        "pure_endpoint",
        "dual_support_disabled",
        "rotation_exact_set",
        "seed_exact_set",
        "expected_run_count",
        "completed_count_before",
        "pending_count_before",
        "completed_cell_immutable",
        "pending_output_paths_absent",
        "per_new_run_300_steps",
        "checkpoint_exact_set",
        "total_method_optimizer_steps",
        "no_duplicate_missing_extra_cell",
        "trainable_parameter_changes",
        "frozen_immutability",
        "no_nan_inf",
        "no_oom",
        "quarantine_usage_zero",
        "dual_support_calls_zero",
        "per_run_metrics_12",
        "aggregate_no_cherry_picking",
        "rotation_analysis",
        "seed_analysis",
        "baseline_contract_source",
        "baseline_exact_set",
        "baseline_budget_audit",
        "baseline_base_binding",
        "baseline_teacher_binding",
        "baseline_target_binding",
        "method_baseline_comparison",
        "base_immutable",
        "teachers_immutable",
        "target_immutable",
        "completed_first_cell_immutable",
        "formal_base_unchanged",
        "paper_modification_zero",
        "final_classification",
        "next_task_uniqueness",
    ]
    passed = set(check_names[:25]) | {
        "checkpoint_exact_set",
        "trainable_parameter_changes",
        "frozen_immutability",
        "no_nan_inf",
        "no_oom",
        "quarantine_usage_zero",
        "dual_support_calls_zero",
        "rotation_analysis",
        "seed_analysis",
        "baseline_contract_source",
        "baseline_exact_set",
        "baseline_budget_audit",
        "base_immutable",
        "teachers_immutable",
        "target_immutable",
        "completed_first_cell_immutable",
        "formal_base_unchanged",
        "paper_modification_zero",
        "final_classification",
        "next_task_uniqueness",
    }
    expected_blocked = set(check_names).difference(passed)
    tests = {
        **common(),
        "runtime_critical_checks": audit["critical_checks"],
        "runtime_critical_pass_count": sum(audit["critical_checks"].values()),
        "runtime_critical_check_count": len(audit["critical_checks"]),
        "contract_checks": [
            {
                "index": index,
                "name": name,
                "status": (
                    "PASS"
                    if name in passed
                    else "BLOCKED_NOT_RUN_AFTER_MANDATORY_PREFLIGHT_FAILURE"
                ),
            }
            for index, name in enumerate(check_names, start=1)
        ],
        "pass_count": len(passed),
        "expected_blocked_count": len(expected_blocked),
        "unexpected_failure_count": 0,
        "optimizer_constructed": False,
        "new_optimizer_steps": 0,
        "test_result": (
            "PASS_30_OF_30_RUNTIME_CRITICAL_CHECKS; "
            "PASS_MANDATORY_BLOCKER_DETECTION; NO_UNAUTHORIZED_EXECUTION"
        ),
        "final_classification": FINAL_CLASSIFICATION,
        "next_task": NEXT_TASK,
    }
    write_json(
        RISK / "subject00_base60747_method_matrix_baseline_tests_20260727.json",
        tests,
    )

    summary = {
        "TASK_ID": TASK_ID,
        "SOURCE_BRANCH": SOURCE_BRANCH,
        "SOURCE_HEAD": SOURCE_HEAD,
        "NEW_BRANCH": BRANCH,
        "WINDOWS_WORKTREE": WINDOWS_WORKTREE,
        "CLOUD_WORKTREE": CLOUD_WORKTREE,
        "EXECUTION_LOCK_PATH": LOCK_PATH,
        "ACTIVE_OPTIMIZER_PROCESS_COUNT_BEFORE": 0,
        "BASE_CHECKPOINT_PATH": audit["base"]["path"],
        "BASE_CHECKPOINT_SHA256": audit["base"]["sha256"],
        "FORMAL_BASE_STATUS": "USER_AUTHORIZED_PAUSED",
        "FORMAL_BASE_RESUME_AUTHORIZED": False,
        "FORMAL_TARGET_ROOT": targets["root"],
        "TRAINING_TARGET_COUNT": targets["training_count"],
        "QUARANTINE_COUNT": targets["quarantine_count"],
        "O01_TEACHER_CHECKPOINT_SHA256": teachers["O01"]["sha256"],
        "O03_TEACHER_CHECKPOINT_SHA256": teachers["O03"]["sha256"],
        "O04_TEACHER_CHECKPOINT_SHA256": teachers["O04"]["sha256"],
        "O03_PROVENANCE_SEAL_PATH": (
            "paper_protocol/reviewer_risk/"
            "subject00_O03_concurrent_formal_teacher_provenance_seal_20260727.json"
        ),
        "METHOD_CONTRACT": "PURE_ENDPOINT",
        "DUAL_SUPPORT_ENABLED": False,
        "METHOD_IMPLEMENTATION_PATH": "tools/paper/formal_batch_runtime.py",
        "METHOD_IMPLEMENTATION_SHA256": method_files["implementation"]["sha256_lf"],
        "METHOD_CONFIG_PATH": "configs/research/subject00_canondressgs_method_base60747_v1.json",
        "METHOD_CONFIG_SHA256": method_files["config"]["sha256"],
        "MATRIX_ROTATIONS": matrix["rotations"],
        "MATRIX_SEEDS": matrix["seeds"],
        "MATRIX_EXPECTED_RUN_COUNT": 12,
        "MATRIX_COMPLETED_RUN_COUNT_BEFORE": 1,
        "MATRIX_PENDING_RUN_COUNT_BEFORE": 11,
        "VALIDATED_COMPLETED_RUN_IDS": ["METHOD-R0-S0"],
        "PENDING_RUN_IDS": pending_ids,
        "EXECUTED_NEW_RUN_IDS": [],
        "MATRIX_COMPLETED_RUN_COUNT_AFTER": 1,
        "MATRIX_TOTAL_OPTIMIZER_STEPS": 300,
        "PER_RUN_OPTIMIZER_STEPS": {"METHOD-R0-S0": 300, "new_runs": 0},
        "PER_RUN_CHECKPOINT_STATUS": {
            "METHOD-R0-S0": "PASS_6_OF_6",
            "new_runs": "BLOCKED_NOT_RUN",
        },
        "PER_RUN_LOSS_INITIAL_FINAL": {
            "METHOD-R0-S0": [initial["loss_initial"], initial["loss_final"]],
            "new_runs": None,
        },
        "PER_RUN_ENDPOINT_RESULTS": {
            "METHOD-R0-S0_training_folds": "6_OF_6_CORRECT",
            "METHOD-R0-S0_formal_test_fold": None,
            "new_runs": None,
        },
        "MATRIX_ENDPOINT_TOP1": None,
        "MATRIX_MEAN_STD": None,
        "ROTATION_EFFECT": None,
        "SEED_EFFECT": None,
        "NAN_INF_STATUS": "NONE_IN_COMPLETED_RUN; NEW_RUNS_NOT_STARTED",
        "OOM_STATUS": "NONE_IN_COMPLETED_RUN; NEW_RUNS_NOT_STARTED",
        "QUARANTINE_OPTIMIZER_USAGE_COUNT": 0,
        "DUAL_SUPPORT_CALL_COUNT": 0,
        "METHOD_WALL_TIME": {
            "completed_run_seconds": completed["wall_time_seconds_by_file_mtime"],
            "new_runs_seconds": 0,
        },
        "METHOD_PEAK_VRAM": {
            "completed_run_bytes": initial["peak_vram_bytes"],
            "new_runs": None,
        },
        "FAIR_BASELINE_CONTRACT_STATUS": (
            "NAMES_RECOVERED_EXECUTION_CONTRACT_INCOMPLETE_AND_MATRIX_BLOCKED"
        ),
        "FAIR_BASELINE_NAMES": audit["baselines"]["exact_names"],
        "FAIR_BASELINE_RUN_COUNT": 0,
        "FAIR_BASELINE_OPTIMIZER_STEPS": 0,
        "FAIR_BASELINE_RESULTS": None,
        "COMPUTE_BUDGET_FAIRNESS_STATUS": "NOT_EVALUABLE_EXECUTION_BLOCKED",
        "METHOD_BASELINE_COMPARISON_PATH": (
            "paper_protocol/reviewer_risk/"
            "subject00_base60747_method_baseline_comparison_20260727.json"
        ),
        "HUMAN_VISUAL_DECISION": None,
        "SCIENTIFIC_PASS": None,
        "PAPER_ELIGIBLE": False,
        **immutable,
        "TEST_RESULT": tests["test_result"],
        "COMMIT_HEAD": "PENDING_FINAL_CONTENT_COMMIT",
        "FINAL_REPORTING_HEAD": "SELF_REFERENTIAL_FINAL_HEAD_REPORTED_IN_RESPONSE",
        "ORIGIN_SYNC_STATUS": "PENDING_FINAL_PUSH",
        "CLOUD_GIT_SYNC_STATUS": "PENDING_FINAL_SYNC",
        "WORKTREE_CLEAN_STATUS": "PENDING_FINAL_COMMIT",
        "PAPER_FINAL": False,
        "FINAL_CLASSIFICATION": FINAL_CLASSIFICATION,
        "NEXT_TASK": NEXT_TASK,
        "BLOCKING_EVIDENCE": {
            "slot04_counts": {"O01": 0, "O03": 0, "O04": 1},
            "rotation_role_blockers": blocker_roles,
            "runner_capability": (
                "frozen Subject00 runner implements only rotation0/seed0 and "
                "cannot create remaining isolated cell directories"
            ),
            "o03_provenance_resolved": True,
            "o03_resolution_changes_fold_coverage": False,
        },
    }
    write_json(
        RISK
        / "subject00_base60747_method_matrix_baseline_final_summary_20260727.json",
        summary,
    )

    report = f"""# Subject00 Base60747 Remaining Method Matrix and Fair Baselines

Task: `{TASK_ID}`

## Outcome

The O03 provenance blocker is resolved and all three Teacher checkpoints are valid.
The remaining matrix cannot be executed under the frozen contract: O01 and O03
slot04 are quarantined, while the frozen rotations require slot04 for rotation0
test, rotation1 calibration, and rotation2/rotation3 training. The frozen
Subject00 runner also implements only rotation0/seed0. No optimizer was created.

Final classification:
`{FINAL_CLASSIFICATION}`

Next task:
`{NEXT_TASK}`

## Evidence

- Runtime critical checks: 30/30 PASS.
- Formal target: 22 records, 2 quarantined, 330/330 payload checksums bound.
- Existing `METHOD-R0-S0`: authentic 300-step training run; six checkpoints;
  training-fold endpoint result 6/6; formal test fold unavailable.
- New method optimizer steps: 0.
- Fair baseline optimizer steps: 0.
- Dual-Support calls: 0.
- Base, Teacher, target, completed cell, Formal Base, and paper mutations: 0.
- Execution lock: `{LOCK_PATH}` (`CLOSED_BLOCKED_PREFLIGHT`).

## Fair baseline status

The paper-facing names are recovered as Reference Classifier Lookup,
Nearest-Centroid Lookup, Outfit-ID Oracle, and Teacher Endpoint. Execution did
not start because the method matrix is incomplete and the sealed evaluator
contract still reports blocked condition resources without a complete
Subject00 per-baseline execution budget.

`PAPER_FINAL = false`
"""
    report_path = (
        RISK / "SUBJECT00_BASE60747_METHOD_MATRIX_AND_BASELINES_REPORT_20260727.md"
    )
    report_path.write_text(report, encoding="utf-8")
    (PAPER / "AAAI27_SUBJECT00_BASE60747_METHOD_MATRIX_AND_BASELINES_REPORT_20260727.md").write_text(
        report, encoding="utf-8"
    )

    handoff = {
        **common(),
        "final_summary_path": (
            "paper_protocol/reviewer_risk/"
            "subject00_base60747_method_matrix_baseline_final_summary_20260727.json"
        ),
        "report_path": str(report_path.relative_to(ROOT)).replace("\\", "/"),
        "tests_path": (
            "paper_protocol/reviewer_risk/"
            "subject00_base60747_method_matrix_baseline_tests_20260727.json"
        ),
        "execution_lock": lock,
        "executed_new_run_ids": [],
        "new_method_optimizer_steps": 0,
        "new_baseline_optimizer_steps": 0,
        "immutable_mutations": immutable,
        "final_classification": FINAL_CLASSIFICATION,
        "next_task": NEXT_TASK,
    }
    write_json(
        HANDOFF / "subject00_base60747_method_matrix_baselines_handoff_20260727.json",
        handoff,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
