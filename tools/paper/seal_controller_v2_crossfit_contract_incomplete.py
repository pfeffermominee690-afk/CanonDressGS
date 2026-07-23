"""Seal the pre-result Controller V2 micro-pilot contract failure.

This is intentionally a no-training program.  The design archive does not
uniquely freeze the training schedule required by Stage 1, so this tool records
the missing fields and emits explicit NOT_RUN artifacts.  It never imports
torch, creates an optimizer, writes a checkpoint, or invokes a renderer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


TASK_ID = "AAAI27-CONTROLLER-V2-CROSSFIT-MICRO-PILOT-001"
SOURCE_HEAD = "4f8c94405e68932e791f69c59ae129c863b1280d"
SOURCE_BRANCH = "research/compatibility-gated-controller-v2-design-20260723"
RUN_BRANCH = (
    "research/compatibility-gated-controller-v2-crossfit-micro-pilot-20260723"
)
CLASSIFICATION = "CONTROLLER_V2_MICRO_PILOT_CONTRACT_INCOMPLETE"
NEXT_TASK = "REPAIR_CONTROLLER_V2_MICRO_PILOT_TRAINING_CONTRACT"

DESIGN_HASHES = {
    "docs/PAPER/AAAI27_COMPATIBILITY_GATED_CONTROLLER_V2_DESIGN_20260723.md":
        "cad8ef69adeecc59cc91a3d23562ba97c298e7d071f73345ab34903ca6d38b3d",
    "docs/PAPER/AAAI27_TRI_MODE_GARMENT_COMPOSITION_20260723.md":
        "3edc8d643ac8ad7105e04f1b68a4cf5dcd7c81d4a0f7534b7a33a7d4caca1a11",
    "docs/PAPER/AAAI27_CONTROLLER_V2_CROSSFIT_PROTOCOL_20260723.md":
        "66eff019559111e079a3c1ab2acab0062f033176c006cccc8913c0325eb3ced0",
    "paper_protocol/reviewer_risk/controller_v2_design_protocol.yaml":
        "0f9c15f907c4ccb843ece8aeecf0cebe74652ca9a8811d496a6052ad3d5499d2",
    "paper_protocol/reviewer_risk/controller_v2_crossfit_splits.json":
        "8d201b5aff26426cb155acb3210e053ef033fdb7d66d2ad586f209144082a586",
    "paper_protocol/reviewer_risk/controller_v2_compatibility_manifests.json":
        "69f7c64b2a9ff962b4e2dc8a638b924149a6c3722153d735c80a3af7ecfeecd1",
    "paper_protocol/reviewer_risk/controller_v2_forward_boundary.json":
        "4e577f1a9afea658eaabd2adf02613f8462f934d951e708cac817ac051562d70",
    "paper_protocol/reviewer_risk/controller_v2_optimizer_training_plan.json":
        "7d822a5d7e243d6efb6d8100678bcd99bc435eea6051cbd1c03111b7692f41cc",
    "paper_protocol/reviewer_risk/controller_v2_evaluator_contract.json":
        "b166be276b6eb0145977d16f64040d6c86256d986cf63769ca5ab109a4f1bb43",
    "paper_protocol/reviewer_risk/controller_v2_dry_run_summary.json":
        "2b0d2029fe069c7bff90472684ea9b7ad25985f8afacfc8910ecdf261070edad",
}

ZERO_COUNTS = {
    "training_runs": 0,
    "v2_training_runs": 0,
    "matched_v1_training_runs": 0,
    "training_steps": 0,
    "training_forward_batches": 0,
    "backward_calls": 0,
    "optimizer_creations": 0,
    "optimizer_steps": 0,
    "scheduler_steps": 0,
    "checkpoint_loads": 0,
    "checkpoint_writes": 0,
    "inference_runs": 0,
    "calibration_runs": 0,
    "test_prediction_runs": 0,
    "renderer_calls": 0,
    "logical_renders": 0,
    "reused_renders": 0,
    "new_renders": 0,
    "metric_aggregations": 0,
    "visual_sheets_generated": 0,
    "visual_sheets_opened": 0,
    "paper_final": 0,
}

RESOURCE_GATE = {
    "observed_at": "2026-07-23T19:30:31+08:00",
    "host": "autodl-container-ef19489c10-464381bb",
    "gpu": {
        "index": 0,
        "name": "NVIDIA GeForce RTX 4090",
        "memory_total_mib": 24564,
        "memory_used_mib": 0,
        "memory_free_mib": 24081,
        "utilization_percent": 0,
        "compute_process_count": 0,
    },
    "storage": {
        "mount": "/root/autodl-tmp",
        "total_bytes": 268435456000,
        "used_bytes": 219930812416,
        "free_bytes": 48504643584,
        "minimum_required_bytes": 26843545600,
        "free_space_gate": "PASS",
        "total_inodes": 730799040,
        "used_inodes": 25883214,
        "free_inodes": 704915826,
    },
    "activity": {
        "avatarex_extraction_active": False,
        "avatarex_full_jpeg_audit_active": False,
        "subject00_runtime_active": False,
        "other_formal_render_active": False,
        "cpu_or_disk_heavy_task_active": False,
        "only_idle_services_observed": [
            "tensorboard",
            "jupyter-lab",
            "autopanel",
        ],
    },
    "status": "PASS",
}

CONTRACT_CHECKS = [
    {
        "id": 1,
        "field": "rotation_record_ids",
        "status": "MISSING",
        "evidence": "Cross-fit archive freezes fold indices but no exact record IDs.",
    },
    {
        "id": 2,
        "field": "train_calibration_test_counts",
        "status": "MISSING",
        "evidence": "No per-rotation record denominators are present.",
    },
    {
        "id": 3,
        "field": "fold_membership",
        "status": "PASS",
        "evidence": "controller_v2_crossfit_splits.json freezes four disjoint rotations.",
    },
    {
        "id": 4,
        "field": "duplicate_preservation",
        "status": "MISSING",
        "evidence": "No duplicate identity/count or preservation rule is materialized.",
    },
    {
        "id": 5,
        "field": "batch_size",
        "status": "MISSING",
        "evidence": "No batch size is frozen in the design artifacts.",
    },
    {
        "id": 6,
        "field": "per_step_record_exposure",
        "status": "MISSING",
        "evidence": "No records-per-step or exposure count is frozen.",
    },
    {
        "id": 7,
        "field": "data_order_algorithm",
        "status": "MISSING",
        "evidence": "No shuffle/permutation algorithm or schedule hash rule is frozen.",
    },
    {
        "id": 8,
        "field": "optimizer_type",
        "status": "MISSING",
        "evidence": "Optimizer plan states no optimizer was created but names no future type.",
    },
    {
        "id": 9,
        "field": "learning_rate",
        "status": "MISSING",
        "evidence": "No learning rate is present.",
    },
    {
        "id": 10,
        "field": "weight_decay",
        "status": "MISSING",
        "evidence": "No weight decay is present.",
    },
    {
        "id": 11,
        "field": "total_optimizer_steps",
        "status": "MISSING",
        "evidence": "No future optimizer-step budget is present.",
    },
    {
        "id": 12,
        "field": "checkpoint_cadence",
        "status": "MISSING",
        "evidence": "No step-indexed checkpoint cadence is present.",
    },
    {
        "id": 13,
        "field": "final_checkpoint_rule",
        "status": "MISSING",
        "evidence": "Best-selection is forbidden, but the exact final-step rule is absent.",
    },
    {
        "id": 14,
        "field": "lambda_mix",
        "status": "MISSING",
        "evidence": "The loss formula names lambda_mix without a numeric value.",
    },
    {
        "id": 15,
        "field": "lambda_weight",
        "status": "MISSING",
        "evidence": "The loss formula names lambda_weight without a numeric value.",
    },
    {
        "id": 16,
        "field": "lambda_cons",
        "status": "MISSING",
        "evidence": "The loss formula names lambda_cons without a numeric value.",
    },
    {
        "id": 17,
        "field": "nuisance_augmentation_types",
        "status": "PASS",
        "evidence": "Blur, mask erosion/dilation, and assignment permutation are frozen.",
    },
    {
        "id": 18,
        "field": "augmentation_severity",
        "status": "MISSING",
        "evidence": "Mild is named but no numerical severity is frozen.",
    },
    {
        "id": 19,
        "field": "augmentation_exposure_schedule",
        "status": "MISSING",
        "evidence": "No frequency, ordering, or step schedule is frozen.",
    },
    {
        "id": 20,
        "field": "information_ablation_training_contract",
        "status": "MISSING",
        "evidence": "Safe-fallback semantics exist but no training records/exposure are frozen.",
    },
    {
        "id": 21,
        "field": "mixedness_threshold_grid",
        "status": "PASS",
        "evidence": "0.10 through 0.90 in 0.10 increments is frozen.",
    },
    {
        "id": 22,
        "field": "pair_confidence_scalar_definition",
        "status": "PASS",
        "evidence": "TOP2_VS_TOP3_PROBABILITY_MARGIN is frozen.",
    },
    {
        "id": 23,
        "field": "pair_confidence_threshold_grid",
        "status": "PASS",
        "evidence": "The nine-value pair-confidence grid is frozen.",
    },
    {
        "id": 24,
        "field": "calibration_objective",
        "status": "PASS",
        "evidence": "The three-level lexicographic objective is frozen.",
    },
    {
        "id": 25,
        "field": "calibration_tie_break",
        "status": "MISSING",
        "evidence": "No deterministic threshold tie-break is specified.",
    },
    {
        "id": 26,
        "field": "seed_list",
        "status": "PASS",
        "evidence": "Seeds 0, 1, and 2 are frozen.",
    },
    {
        "id": 27,
        "field": "final_evaluator_denominators",
        "status": "MISSING",
        "evidence": "Metric families are listed but exact query denominators are absent.",
    },
    {
        "id": 28,
        "field": "perturbation_representative_ids",
        "status": "MISSING",
        "evidence": "Perturbation classes exist but representative record IDs are absent.",
    },
    {
        "id": 29,
        "field": "visual_grade_schema",
        "status": "MISSING",
        "evidence": "No complete category-by-grade rubric is frozen for this pilot.",
    },
    {
        "id": 30,
        "field": "success_gates",
        "status": "PASS",
        "evidence": "The design report freezes pair, routing, weight, visual, and ablation gates.",
    },
]


REPORT_PATHS = [
    "docs/PAPER/AAAI27_CONTROLLER_V2_CROSSFIT_MICRO_PILOT_20260723.md",
    "docs/PAPER/AAAI27_CONTROLLER_V2_MATCHED_V1_COMPARISON_20260723.md",
    "docs/PAPER/AAAI27_CONTROLLER_V2_ROUTING_AND_WEIGHT_ANALYSIS_20260723.md",
    "docs/PAPER/AAAI27_CONTROLLER_V2_VISUAL_FAILURE_ANALYSIS_20260723.md",
    "docs/PAPER/AAAI27_CONTROLLER_V2_EFFICIENCY_20260723.md",
]

MACHINE_PATHS = [
    "paper_protocol/reviewer_risk/controller_v2_micro_pilot_training_contract.json",
    "paper_protocol/reviewer_risk/controller_v2_micro_pilot_training_results.json",
    "paper_protocol/reviewer_risk/controller_v1_matched_crossfit_results.json",
    "paper_protocol/reviewer_risk/controller_v2_calibration_results.json",
    "paper_protocol/reviewer_risk/controller_v2_test_predictions.json",
    "paper_protocol/reviewer_risk/controller_v2_routing_results.json",
    "paper_protocol/reviewer_risk/controller_v2_visual_results.json",
    "paper_protocol/reviewer_risk/controller_v2_perturbation_results.json",
    "paper_protocol/reviewer_risk/controller_v2_efficiency_results.json",
    "paper_protocol/reviewer_risk/controller_v2_visual_review.json",
    "paper_protocol/reviewer_risk/controller_v2_micro_pilot_final_summary.json",
]

HANDOFF_PATH = (
    "project_control_handoff/controller_v2_crossfit_micro_pilot_handoff.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once_text(path: Path, text: str) -> None:
    if not text.endswith("\n"):
        text += "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"append-only collision: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_once_json(path: Path, value: Any) -> None:
    write_once_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def verify_source(repo_root: Path) -> None:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    if head != SOURCE_HEAD:
        ancestor_check = subprocess.run(
            ["git", "merge-base", "--is-ancestor", SOURCE_HEAD, head],
            cwd=repo_root,
            check=False,
        )
        if ancestor_check.returncode != 0:
            raise RuntimeError(f"source HEAD is not an ancestor: {head}")
    for relative, expected in DESIGN_HASHES.items():
        blob = subprocess.check_output(
            ["git", "show", f"{SOURCE_HEAD}:{relative}"], cwd=repo_root
        )
        actual = hashlib.sha256(blob).hexdigest()
        if actual != expected:
            raise RuntimeError(f"design hash mismatch: {relative}: {actual}")
        changed = subprocess.run(
            ["git", "diff", "--quiet", SOURCE_HEAD, "--", relative],
            cwd=repo_root,
            check=False,
        )
        if changed.returncode != 0:
            raise RuntimeError(f"frozen design artifact changed: {relative}")


def training_contract() -> dict[str, Any]:
    passed = sum(row["status"] == "PASS" for row in CONTRACT_CHECKS)
    missing = [row["field"] for row in CONTRACT_CHECKS if row["status"] == "MISSING"]
    rotations = [
        {
            "rotation": rotation,
            "train_record_count": None,
            "calibration_record_count": None,
            "test_record_count": None,
            "pure_count": None,
            "mixed_count": None,
            "pair_coverage": None,
            "aab_coverage": None,
            "abb_coverage": None,
            "duplicate_count": None,
            "exact_record_ids": None,
            "record_manifest_sha256": None,
            "status": "UNRESOLVED_CONTRACT_INCOMPLETE",
        }
        for rotation in range(4)
    ]
    return {
        "schema_version": (
            "canondressgs.research.controller_v2_micro_pilot_training_contract.v1"
        ),
        "task_id": TASK_ID,
        "status": "CONTRACT_INCOMPLETE",
        "classification": CLASSIFICATION,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "run_branch": RUN_BRANCH,
        "resource_io_gate": RESOURCE_GATE,
        "design_artifacts": [
            {"path": path, "sha256": digest}
            for path, digest in DESIGN_HASHES.items()
        ],
        "contract_check_count": len(CONTRACT_CHECKS),
        "contract_pass_count": passed,
        "contract_missing_count": len(missing),
        "contract_checks": CONTRACT_CHECKS,
        "missing_fields": missing,
        "rotation_contracts": rotations,
        "model_families": {
            "V2": {
                "planned_run_count": 12,
                "authorized_run_count": 0,
                "status": "NOT_AUTHORIZED_CONTRACT_INCOMPLETE",
            },
            "MATCHED_V1": {
                "planned_run_count": 12,
                "authorized_run_count": 0,
                "status": "NOT_ESTABLISHED_CONTRACT_INCOMPLETE",
            },
        },
        "baseline_comparability": {
            "status": "INCOMPLETE",
            "reason": (
                "The common record sequence, exposure budget, optimizer schedule, "
                "and exact final-checkpoint rule are not frozen."
            ),
            "historical_formal_v1_may_substitute": False,
        },
        "execution_counts": ZERO_COUNTS,
        "paper_final": 0,
        "training_authorized": False,
    }


def not_run_artifact(name: str) -> dict[str, Any]:
    return {
        "schema_version": f"canondressgs.research.{name}.v1",
        "task_id": TASK_ID,
        "status": "NOT_RUN_CONTRACT_INCOMPLETE",
        "classification": CLASSIFICATION,
        "reason": "Stage 1 training contract is not uniquely frozen.",
        "records": [],
        "denominator": 0,
        "execution_counts": ZERO_COUNTS,
        "paper_final": 0,
    }


def final_summary(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": (
            "canondressgs.research.controller_v2_micro_pilot_final_summary.v1"
        ),
        "task_id": TASK_ID,
        "status": "FINAL",
        "final_classification": CLASSIFICATION,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "run_branch": RUN_BRANCH,
        "resource_gate": "PASS",
        "contract_completeness": "FAIL",
        "contract_pass_count": contract["contract_pass_count"],
        "contract_missing_count": contract["contract_missing_count"],
        "missing_fields": contract["missing_fields"],
        "matched_v1_comparability": "INCOMPLETE_NOT_STARTED",
        "all_scientific_metrics": "NOT_RUN",
        "visual_review": {
            "expected_main_sheets_if_training_authorized": 240,
            "generated": 0,
            "opened": 0,
            "status": "NOT_RUN_CONTRACT_INCOMPLETE",
        },
        "information_boundary": {
            "gt_inference_use": 0,
            "target_forward_use": 0,
            "reason": "No training, calibration, inference, or rendering was executed.",
        },
        "frozen_upstream_mutation_count": 0,
        "execution_counts": ZERO_COUNTS,
        "paper_final": 0,
        "next_task_started": False,
        "next_task": NEXT_TASK,
    }


def main_report(contract: dict[str, Any]) -> str:
    missing = "\n".join(f"- `{field}`" for field in contract["missing_fields"])
    return f"""# Controller V2 Cross-Fit Micro-Pilot: Pre-Result Contract Gate

**RESEARCH MICRO-PILOT — NOT PAPER FINAL**

- Task: `{TASK_ID}`
- Source: `{SOURCE_BRANCH}@{SOURCE_HEAD}`
- Branch: `{RUN_BRANCH}`
- Classification: `{CLASSIFICATION}`

## Stage 0 resource and activity gate

The RTX 4090 was idle with 24,081 MiB free GPU memory and no compute process.
`/root/autodl-tmp` had 48,504,643,584 free bytes (about 45.2 GiB), above the
25 GiB gate, and 704,915,826 free inodes. No AvatarReX extraction/full JPEG
audit, subject00 runtime, formal renderer, or CPU/disk-heavy task was active.
Stage 0 therefore passed.

## Stage 1 contract result

Only {contract["contract_pass_count"]}/30 required fields were uniquely frozen;
{contract["contract_missing_count"]}/30 were missing. The absent fields are:

{missing}

The design correctly freezes cross-fit membership, nuisance types, both
threshold grids, the pair-confidence scalar, the calibration objective, seeds,
and success gates. It does not freeze the data records or executable training
schedule. Supplying those values from code defaults would violate the task.

## Matched V1 comparability

A strict matched V1 baseline cannot be instantiated without the same exact
record IDs, data order, exposures, optimizer-step budget, and final-checkpoint
rule. Historical Formal V1 is not substituted because its protocol exposed all
condition folds.

## Mandatory stop

Training runs, forward batches, backward calls, optimizer creations/steps,
scheduler steps, checkpoint loads/writes, calibration, inference, render calls,
metrics, and visual sheets all remain 0. No downstream scientific result or
success-gate judgment was manufactured. PAPER_FINAL remains 0.
"""


def blocked_report(title: str, subject: str) -> str:
    return f"""# {title}

**RESEARCH MICRO-PILOT — NOT PAPER FINAL**

Status: `NOT_RUN_CONTRACT_INCOMPLETE`.

{subject} was not executed because Stage 1 did not uniquely freeze the
record-level training and evaluation contract. All denominators and scientific
metrics remain unreported rather than being inferred or fabricated. Training,
optimizer, checkpoint, inference, renderer, metric, and visual-review counts
are 0.
"""


def seal_repository(repo_root: Path) -> None:
    contract = training_contract()
    summary = final_summary(contract)
    risk = repo_root / "paper_protocol/reviewer_risk"
    write_once_json(
        risk / "controller_v2_micro_pilot_training_contract.json", contract
    )
    stubs = {
        "controller_v2_micro_pilot_training_results.json":
            "controller_v2_micro_pilot_training_results",
        "controller_v1_matched_crossfit_results.json":
            "controller_v1_matched_crossfit_results",
        "controller_v2_calibration_results.json":
            "controller_v2_calibration_results",
        "controller_v2_test_predictions.json": "controller_v2_test_predictions",
        "controller_v2_routing_results.json": "controller_v2_routing_results",
        "controller_v2_visual_results.json": "controller_v2_visual_results",
        "controller_v2_perturbation_results.json":
            "controller_v2_perturbation_results",
        "controller_v2_efficiency_results.json":
            "controller_v2_efficiency_results",
        "controller_v2_visual_review.json": "controller_v2_visual_review",
    }
    for file_name, schema_name in stubs.items():
        value = not_run_artifact(schema_name)
        if file_name == "controller_v2_visual_review.json":
            value.update(
                {
                    "expected_main_sheet_count_if_authorized": 240,
                    "generated_main_sheet_count": 0,
                    "actual_opened_count": 0,
                    "diagnostic_sheet_count": 0,
                    "reviewer": None,
                    "review_time": None,
                    "original_detail": False,
                    "grades": [],
                }
            )
        write_once_json(risk / file_name, value)
    write_once_json(
        risk / "controller_v2_micro_pilot_final_summary.json", summary
    )

    reports = {
        REPORT_PATHS[0]: main_report(contract),
        REPORT_PATHS[1]: blocked_report(
            "Controller V2 Matched V1 Comparison",
            "Matched V1 training and comparison",
        ),
        REPORT_PATHS[2]: blocked_report(
            "Controller V2 Routing and Weight Analysis",
            "Calibration, held-out routing, and weight evaluation",
        ),
        REPORT_PATHS[3]: blocked_report(
            "Controller V2 Visual Failure Analysis",
            "Rendering and human visual review",
        ),
        REPORT_PATHS[4]: blocked_report(
            "Controller V2 Efficiency",
            "Controller and renderer efficiency measurement",
        ),
    }
    for relative, text in reports.items():
        write_once_text(repo_root / relative, text)

    handoff = {
        "schema_version": (
            "canondressgs.research.controller_v2_crossfit_micro_pilot_handoff.v1"
        ),
        "task_id": TASK_ID,
        "status": "PASS",
        "final_classification": CLASSIFICATION,
        "source_head": SOURCE_HEAD,
        "run_branch": RUN_BRANCH,
        "resource_gate": "PASS",
        "contract_gate": "FAIL",
        "training_authorized": False,
        "matched_v1_comparability": "INCOMPLETE",
        "required_report_paths": REPORT_PATHS,
        "required_machine_paths": MACHINE_PATHS,
        "execution_counts": ZERO_COUNTS,
        "frozen_upstream_mutation_count": 0,
        "paper_final": 0,
        "next_task_started": False,
        "next_task": NEXT_TASK,
    }
    write_once_json(repo_root / HANDOFF_PATH, handoff)


def seal_output(output_root: Path) -> None:
    attempt = output_root / "attempt_001"
    contract = training_contract()
    summary = final_summary(contract)
    write_once_json(
        attempt / "contract/controller_v2_micro_pilot_training_contract.json",
        contract,
    )
    write_once_json(attempt / "audits/resource_io_gate.json", RESOURCE_GATE)
    write_once_json(
        attempt / "audits/contract_completeness.json",
        {
            "status": "FAIL",
            "classification": CLASSIFICATION,
            "checks": CONTRACT_CHECKS,
            "execution_counts": ZERO_COUNTS,
        },
    )
    write_once_json(attempt / "audits/final_summary.json", summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--write-repository-artifacts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    verify_source(repo_root)
    if args.write_repository_artifacts:
        seal_repository(repo_root)
    if args.output_root is not None:
        seal_output(args.output_root.resolve())
    if not args.write_repository_artifacts and args.output_root is None:
        raise ValueError("no output target was requested")


if __name__ == "__main__":
    main()
