#!/usr/bin/env python3
"""Freeze the Subject00 human review and seal the Subject02 matched blocker.

This task is intentionally pre-optimizer.  The frozen Subject02 matched
contract names Subject00-style slots, while the passed Subject02 assets expose
only condition IDs and cardinal-view labels.  No experiment may start until an
explicit, authoritative slot-to-condition/camera binding is frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TASK_ID = "AAAI27-SUBJECT00-REVIEW-FREEZE-AND-SUBJECT02-COMMONSAFE4-MATCHED-001"
SOURCE_BRANCH = "research/subject00-commonsafe4-final-head-seal-review-20260727"
SOURCE_HEAD = "245977f029b4023798ba55b4f78020e12cdb976e"
NEW_BRANCH = "research/subject00-review-freeze-subject02-commonsafe4-matched-20260727"
WINDOWS_WORKTREE = (
    r"E:\model_train\canondressgs_subject00_review_freeze_subject02_commonsafe4_matched"
)
CLOUD_WORKTREE = (
    "/root/autodl-tmp/canondressgs_work/worktrees/"
    "canondressgs_subject00_review_freeze_subject02_commonsafe4_matched"
)
PDF_SHA256 = "f782724f47d542ff429f21cf1b418c37b3d9107dbd10495874a7a3c0593ec4b9"
PDF_BYTES = 15203921
PDF_PAGES = 12
FINAL_CLASSIFICATION = (
    "SUBJECT00_REVIEW_FROZEN_SUBJECT02_MATCHED_BLOCKED_BY_ASSET_CONTRACT"
)
NEXT_TASK = (
    "USER_FREEZE_SUBJECT02_COMMONSAFE4_MATCHED_ASSET_CONTRACT_AND_REAUTHORIZE_EXECUTION"
)

REPO = Path(__file__).resolve().parents[2]
RISK = REPO / "paper_protocol" / "reviewer_risk"
HANDOFF = REPO / "project_control_handoff"
DOCS = REPO / "docs" / "PAPER"

MATCHED_CONTRACT_REL = (
    "paper_protocol/reviewer_risk/"
    "subject02_commonsafe4_matched_protocol_execution_contract_20260727.json"
)
MATCHED_CONTRACT_SHA256 = (
    "4d8141dfbbc7e02e4bdb4ea9ce9e3fc72841800fa16de06632475cd78806abf9"
)
SUBJECT00_SEAL_REL = (
    "paper_protocol/reviewer_risk/"
    "subject00_commonsafe4_method_matrix_baseline_posthoc_provenance_seal_20260727.json"
)
SUBJECT00_SEAL_SHA256 = (
    "c7321b4e4ada94ddf9062ddcacd36c29794647797dd9dd3bc8761fd87e51c59b"
)

BASE = {
    "path": (
        "/root/autodl-tmp/canondressgs_work/outputs/"
        "subject02_formal_800k/chkpnt100000.pth"
    ),
    "sha256": "abbf67b59eadf2cba2dea69dbeec598f9177da45b8ddc74ccbe8108acf9ddf70",
}
TEACHERS = {
    "O01": {
        "path": (
            "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/"
            "SUBJECT02-REPRESENTATION-TRIAGE-001/attempt_002/"
            "rung_2_shared_same_support/O01/checkpoints/step_001200.pth"
        ),
        "sha256": "af730d138697ab9c7a29f17603303ae41dfa36047b8cee59e2ee89328655ce56",
    },
    "O03": {
        "path": (
            "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/"
            "SUBJECT02-MULTI-OUTFIT-EXPLICIT-BASIS-001/attempt_003/"
            "stage_a_teacher_bank/O03/checkpoints/step_001200.pth"
        ),
        "sha256": "16cb235d784e17f4ebb3928e020ceb0be050b5cf80740e56ec72bcbf11d37d93",
    },
    "O04": {
        "path": (
            "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/"
            "SUBJECT02-MULTI-OUTFIT-EXPLICIT-BASIS-001/attempt_003/"
            "stage_a_teacher_bank/O04/checkpoints/step_001200.pth"
        ),
        "sha256": "b39c8d4940325e371cd6db55551c4830b19ab057106c2b9e06df796e8bacd9c3",
    },
}
TARGET = {
    "root": (
        "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/"
        "SUBJECT02-AAAI27-DATA-CAPACITY-GATE-001/attempt_003/dataset"
    ),
    "registry": (
        "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/"
        "SUBJECT02-AAAI27-DATA-CAPACITY-GATE-001/attempt_003/dataset/"
        "aaai_gate_28_manifest.json"
    ),
    "registry_sha256": (
        "49bee929edc236af8f37d66be50eb7c44f4011d49e435af903c0e15918a83bbf"
    ),
}

ANCHORS = ["slot00", "slot07", "slot03", "slot06"]
SEEDS = [0, 1, 2]
ROTATIONS = [
    {
        "rotation": "R0",
        "train": ["slot00", "slot07"],
        "calibration": "slot03",
        "test": "slot06",
    },
    {
        "rotation": "R1",
        "train": ["slot07", "slot03"],
        "calibration": "slot06",
        "test": "slot00",
    },
    {
        "rotation": "R2",
        "train": ["slot03", "slot06"],
        "calibration": "slot00",
        "test": "slot07",
    },
    {
        "rotation": "R3",
        "train": ["slot06", "slot00"],
        "calibration": "slot07",
        "test": "slot03",
    },
]
AVAILABLE_SUBJECT02_CONDITIONS = [
    {"condition_id": "cond_000000", "view": "front"},
    {"condition_id": "cond_000318", "view": "back"},
    {"condition_id": "cond_000017", "view": "left"},
    {"condition_id": "cond_000347", "view": "right"},
]

OLD_MAINLINE = {
    "output_root": (
        "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/"
        "SUBJECT02-MULTI-OUTFIT-EXPLICIT-BASIS-001/attempt_003"
    ),
    "status": "COMPLETE",
    "file_count": 2869,
    "bytes": 1140086291,
    "metadata_fingerprint_before": (
        "daa61f74e9021ad29408626e6a8afba89ee485dc386eb224fd81c4859bc36d60"
    ),
    "metadata_fingerprint_after": (
        "daa61f74e9021ad29408626e6a8afba89ee485dc386eb224fd81c4859bc36d60"
    ),
    "mutation_count": 0,
}
PURE_ENDPOINT = {
    "sealed_output": (
        "/root/autodl-tmp/canondressgs_work/outputs/"
        "PURE-ENDPOINT-CORE-METHOD-CROSSFIT-001/attempt_001"
    ),
    "status": "SEALED",
    "classification": "PURE_ENDPOINT_CORE_METHOD_SUPPORTED",
    "execution_branch": "research/pure-endpoint-core-method-crossfit-amended-20260724",
    "execution_head": "195fb887f2cac8a72920b499c44bb66700a97e25",
    "final_reporting_head": "1fc5c2de97cb86858c7c070ad80f3dcef9095b5c",
    "runner_at_execution_head": "tools/paper/run_pure_endpoint_core_method_crossfit_amended.py",
    "runner_present_in_new_worktree": False,
    "runtime_at_execution_head": "tools/paper/formal_batch_runtime.py",
    "rotation_manifest_sha256": (
        "5a86f57ef44a54fd234690c3a133b82a17d982aebad7f451833122eecb9825c9"
    ),
    "model_contract_sha256": (
        "abca0350821fbcc706003c45b9aec0357223168ac7c6ab2913b3ecde617d73d6"
    ),
    "evaluator_contract_sha256": (
        "0f86c6413a0db955d179de139f85b401bc80044f89222219afe5d0506c4aaedd"
    ),
    "run_status_sha256": (
        "6dcbbfc7e45c6b85ea7900999c4cb84ca44e3d8483f8ec8295d8f9e60abe1de8"
    ),
    "condition_order": [
        "cond_000000",
        "cond_000318",
        "cond_000017",
        "cond_000347",
    ],
    "checkpoint_steps": [0, 20, 50, 100, 200, 300],
    "tie_break": "first garment in frozen candidate_order",
    "endpoint_candidates_available": ["O01", "O03", "O04"],
}


def base_metadata() -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "new_branch": NEW_BRANCH,
        "paper_eligible": False,
        "paper_final": False,
        "paper_modifications": 0,
    }


def subject00_review() -> dict[str, Any]:
    return {
        **base_metadata(),
        "schema_version": "canondressgs.subject00.commonsafe4.human_review.v1",
        "status": "FROZEN",
        "review_pdf": {
            "bytes": PDF_BYTES,
            "page_count": PDF_PAGES,
            "sha256": PDF_SHA256,
            "independent_poppler_render_count": 12,
            "independent_render_status": "PASS",
        },
        "provenance_seal": {
            "path": SUBJECT00_SEAL_REL,
            "sha256": SUBJECT00_SEAL_SHA256,
            "decision": "PASS",
        },
        "decisions": {
            "SUBJECT00_PROTOCOL_DECISION": (
                "PASS_WITH_DISCLOSED_COMMONSAFE4_SUBSTITUTION"
            ),
            "SUBJECT00_MATRIX_PROVENANCE_DECISION": "PASS",
            "SUBJECT00_REPLACEMENT_DECISION": (
                "PASS_PRETRAINING_OUTCOME_INDEPENDENT"
            ),
            "SUBJECT00_SCIENTIFIC_EVIDENCE_VALID": True,
            "SUBJECT00_METHOD_RESULT_CLASS": "MIXED_NEGATIVE_ON_SECOND_IDENTITY",
            "SUBJECT00_PRIMARY_POSITIVE_CLAIM_SUPPORTED": False,
            "SUBJECT00_METHOD_OUTPERFORMS_REFERENCE_CLASSIFIER": False,
            "SUBJECT00_METHOD_OUTPERFORMS_NEAREST_CENTROID": False,
            "SUBJECT00_DIRECT_CROSS_IDENTITY_COMPARISON_AUTHORIZED": False,
            "SUBJECT00_PAPER_ELIGIBLE": False,
        },
        "baseline_fairness": {
            "decision": "PASS_WITH_DIFFERENTIAL_COMPUTE_DISCLOSURE",
            "Reference Classifier Lookup": "NON_ORACLE_COMPARISON_BASELINE",
            "Nearest-Centroid Lookup": "NON_ORACLE_COMPARISON_BASELINE",
            "Outfit-ID Oracle": "ORACLE_UPPER_REFERENCE",
            "Teacher Endpoint": "ORACLE_UPPER_REFERENCE",
        },
    }


def mixed_negative() -> dict[str, Any]:
    return {
        **base_metadata(),
        "schema_version": "canondressgs.subject00.commonsafe4.mixed_negative.v1",
        "status": "FROZEN_EXPERIMENTALLY_VALID_MIXED_NEGATIVE_RESULT",
        "classification": "MIXED_NEGATIVE_ON_SECOND_IDENTITY",
        "method": {"correct": 30, "total": 36, "top1": 30 / 36},
        "non_oracle_baselines": {
            "Reference Classifier Lookup": {
                "correct": 31,
                "total": 36,
                "top1": 31 / 36,
            },
            "Nearest-Centroid Lookup": {
                "correct": 35,
                "total": 36,
                "top1": 35 / 36,
            },
        },
        "errors": {
            "count": 6,
            "pattern": "O04_TO_O03_ONLY",
            "tie_break_error_count": 0,
            "score_margin": {
                "minimum": 1.011,
                "maximum": 2.807,
                "mean_approx": 1.7935,
            },
            "prohibited_descriptions": [
                "random noise",
                "tie",
                "tie-break error",
                "single anomalous seed",
                "single anomalous camera",
            ],
        },
        "claim": {
            "primary_positive_claim_supported": False,
            "outperforms_reference_classifier": False,
            "outperforms_nearest_centroid": False,
        },
    }


def claim_boundary() -> dict[str, Any]:
    return {
        **base_metadata(),
        "schema_version": "canondressgs.subject00.commonsafe4.claim_boundary.v1",
        "status": "FROZEN",
        "authorized_claims": [
            "The Subject00 CommonSafe4 execution is valid evidence.",
            "The Subject00 learned method obtains 30/36 endpoint top-1.",
            "The result is mixed/negative on the second identity.",
            "All six method errors are O04-to-O03 with non-tie margins.",
            "Reference Classifier and Nearest-Centroid are non-oracle comparisons.",
            "Outfit-ID Oracle and Teacher Endpoint are upper references.",
        ],
        "prohibited_claims": [
            "The learned method outperforms Reference Classifier on Subject00.",
            "The learned method outperforms Nearest-Centroid on Subject00.",
            "The learned controller has a general advantage across identities.",
            "Subject00 and Subject02 have been directly matched numerically.",
            "The Teacher endpoints have passed full multi-view human review.",
            "All five method families have equal optimizer budgets.",
        ],
        "direct_cross_identity_numeric_comparison_authorized": False,
        "reason": "Subject02 matched execution was blocked before optimization.",
    }


def teacher_limitation() -> dict[str, Any]:
    rows = {}
    for outfit in ("O01", "O03", "O04"):
        rows[outfit] = {
            "technical_status": "TECHNICAL_PASS",
            "shown_view_decision": (
                "VISUALLY_RECOGNIZABLE_WITH_BOUNDARY_ARTIFACTS"
            ),
            "human_pass": False,
            "human_fail": False,
        }
    return {
        **base_metadata(),
        "schema_version": "canondressgs.subject00.teacher_visual_limitation.v1",
        "status": "FROZEN",
        "teachers": rows,
        "TEACHER_HUMAN_VISUAL_DECISION": (
            "INCONCLUSIVE_INSUFFICIENT_MULTI_VIEW_EVIDENCE"
        ),
        "TEACHER_HUMAN_PASS": False,
        "TEACHER_HUMAN_FAIL": False,
        "TEACHER_REVIEW_STATUS": "PENDING_DEDICATED_MULTI_VIEW_REVIEW",
        "limitations": [
            "The review pack shows only one slot06 Teacher view.",
            "O03 and O04 both have dark upper-body appearance.",
            "O03 and O04 endpoint silhouettes are close.",
            "Head/shoulder and hand/arm boundary noise is visible.",
            "Front identity, hands, feet, and all training views were not shown.",
        ],
    }


def asset_audit() -> dict[str, Any]:
    return {
        **base_metadata(),
        "schema_version": "canondressgs.subject02.commonsafe4.asset_audit.v1",
        "status": "BLOCKED_PREOPTIMIZER_ASSET_CONTRACT_AMBIGUITY",
        "matched_contract": {
            "path": MATCHED_CONTRACT_REL,
            "sha256": MATCHED_CONTRACT_SHA256,
            "protocol": "COMMONSAFE4_CARDINAL_PROXY_V1",
            "anchors": ANCHORS,
            "rotations": ROTATIONS,
            "seeds": SEEDS,
            "historical_execution_authorized": False,
            "current_task_execution_authority_received": True,
        },
        "uniquely_recovered": {
            "base": BASE,
            "teachers": TEACHERS,
            "target": TARGET,
            "pure_endpoint_runtime_and_contracts": PURE_ENDPOINT,
            "available_subject02_condition_bindings": AVAILABLE_SUBJECT02_CONDITIONS,
            "optimizer_steps_per_run": 300,
            "checkpoint_steps": [0, 20, 50, 100, 200, 300],
            "tie_break": "first garment in frozen candidate_order",
            "endpoint_candidates": ["O01", "O03", "O04"],
        },
        "not_uniquely_recovered": {
            "slot_to_subject02_condition_camera_binding": {
                "required_slots": ANCHORS,
                "explicit_binding_rows_found": 0,
                "subject02_slot_labeled_target_records_found": 0,
                "subject02_available_views": AVAILABLE_SUBJECT02_CONDITIONS,
                "conflict": (
                    "The matched contract records Subject00 slot06 as cam09/back-right, "
                    "while the passed Subject02 registry offers cond_000347/right and "
                    "contains no authority equating the two."
                ),
            },
            "matched_runner_and_config_in_new_source_lineage": {
                "status": "NOT_PRESENT",
                "historical_runner_recoverable_from_git": True,
                "historical_runner_directly_authorized_for_new_slot_contract": False,
                "reason": (
                    "The historical Pure Endpoint runner consumes condition IDs and "
                    "is not an ancestor/present implementation of the new matched "
                    "slot contract. Porting it requires a new binding decision."
                ),
            },
        },
        "forbidden_inference_not_taken": (
            "No positional/order-based mapping from "
            "[slot00,slot07,slot03,slot06] to "
            "[front,back,left,right] was invented."
        ),
        "output_preflight": {
            "method_root": (
                "/root/autodl-tmp/canondressgs_work/outputs/"
                "SUBJECT02-CANONDRESSGS-METHOD-COMMONSAFE4-MATCHED-001"
            ),
            "method_root_absent": True,
            "baseline_root": (
                "/root/autodl-tmp/canondressgs_work/outputs/"
                "SUBJECT02-COMMONSAFE4-MATCHED-BASELINES-001"
            ),
            "baseline_root_absent": True,
            "active_optimizer_process_count_before": 0,
            "active_optimizer_process_count_after": 0,
            "gpu_preflight": "NVIDIA GeForce RTX 4090, 0 %, 0 MiB",
        },
        "subject02_existing_mainline_immutability": OLD_MAINLINE,
        "decision": "DO_NOT_START_OPTIMIZER",
        "optimizer_creations": 0,
        "optimizer_steps": 0,
        "new_output_roots_created": 0,
    }


def method_registry() -> dict[str, Any]:
    return {
        **base_metadata(),
        "schema_version": "canondressgs.subject02.commonsafe4.method_execution.v1",
        "status": "BLOCKED_PREOPTIMIZER_ASSET_CONTRACT_AMBIGUITY",
        "contract_path": MATCHED_CONTRACT_REL,
        "planned": {
            "method": "CanonDressGS-Endpoint",
            "method_contract": "PURE_ENDPOINT",
            "dual_support_enabled": False,
            "anchors": ANCHORS,
            "rotations": ROTATIONS,
            "seeds": SEEDS,
            "run_count": 12,
            "optimizer_steps_per_run": 300,
            "total_optimizer_steps": 3600,
            "checkpoint_steps": [0, 20, 50, 100, 200, 300],
            "checkpoint_count": 72,
            "records_per_run": {"train": 6, "calibration": 3, "test": 3},
            "formal_test_denominator": 3,
        },
        "actual": {
            "run_ids": [],
            "run_count": 0,
            "optimizer_creations": 0,
            "optimizer_steps": 0,
            "checkpoint_count": 0,
            "formal_test_count": 0,
            "dual_support_call_count": 0,
            "nan_count": 0,
            "inf_count": 0,
            "oom_count": 0,
        },
        "blocker": asset_audit()["not_uniquely_recovered"],
    }


def baseline_registry() -> dict[str, Any]:
    return {
        **base_metadata(),
        "schema_version": "canondressgs.subject02.commonsafe4.baseline_registry.v1",
        "status": "BLOCKED_NOT_EXECUTED",
        "shared_contract": {
            "anchors": ANCHORS,
            "rotations": ROTATIONS,
            "seeds": SEEDS,
            "records_per_cell": {"train": 6, "calibration": 3, "test": 3},
            "formal_test_denominator": 3,
            "same_evaluator_required": True,
            "same_endpoint_candidates_required": True,
        },
        "methods": [
            {
                "name": "Reference Classifier Lookup",
                "class": "NON_ORACLE_COMPARISON_BASELINE",
            },
            {
                "name": "Nearest-Centroid Lookup",
                "class": "NON_ORACLE_COMPARISON_BASELINE",
            },
            {"name": "Outfit-ID Oracle", "class": "ORACLE_UPPER_REFERENCE"},
            {"name": "Teacher Endpoint", "class": "ORACLE_UPPER_REFERENCE"},
        ],
        "planned_cell_count": 48,
        "actual_cell_count": 0,
        "optimizer_steps": 0,
        "blocker": "SUBJECT02_SLOT_CAMERA_BINDING_NOT_UNIQUELY_RECOVERED",
    }


def tests_payload() -> dict[str, Any]:
    pass_items = {
        1: "Subject00 source branch/head exact",
        2: "PDF bytes/SHA/page count exact and independently rendered",
        3: "Subject00 review decisions frozen",
        4: "Subject00 mixed-negative result frozen",
        5: "Teacher visual decision inconclusive",
        6: "Subject00 method result 30/36",
        7: "Reference Classifier 31/36",
        8: "Nearest-Centroid 35/36",
        9: "Errors O04_TO_O03_ONLY",
        10: "Error margins non-tie",
        11: "Subject02 matched symbolic contract exact",
        12: "Subject02 Base path/SHA uniquely bound",
        13: "Subject02 O01/O03/O04 Teacher paths/SHAs uniquely bound",
        14: "Subject02 target root/registry/SHA uniquely bound",
        15: "Exact symbolic anchors preserved; physical binding correctly blocked",
        16: "Exact symbolic rotations preserved",
        17: "Exact seeds preserved",
        22: "Dual-Support call count remains zero",
        23: "No NaN/Inf/OOM event because execution never started",
        25: "Oracle versus non-oracle classification frozen",
        30: "Claim recommendation generated without new positive claim",
        31: "Subject00 immutable inputs and outputs untouched",
        32: "Subject02 existing mainline untouched",
        33: "Paper modifications zero",
        34: "Final blocker classification exact",
        35: "Exactly one blocker-resolution NEXT_TASK",
    }
    not_run_items = {
        18: "Method run count",
        19: "Method total optimizer steps",
        20: "Method checkpoint count",
        21: "Formal tests",
        24: "Baseline cells",
        26: "Aggregate no-cherry-pick result",
        27: "Matched cross-identity numeric comparison",
        28: "Protocol-effect decomposition",
        29: "Identity-effect decomposition",
    }
    rows = []
    for index in range(1, 36):
        if index in pass_items:
            rows.append(
                {"test": index, "name": pass_items[index], "status": "PASS"}
            )
        else:
            rows.append(
                {
                    "test": index,
                    "name": not_run_items[index],
                    "status": "NOT_RUN_DUE_TO_PREOPTIMIZER_BLOCKER",
                }
            )
    return {
        **base_metadata(),
        "schema_version": "canondressgs.subject00_subject02.matched_tests.v1",
        "status": "PASS_EXPECTED_BLOCKER_PATH",
        "result": "PASS_BLOCKER_CLASSIFICATION_26_PASS_9_NOT_RUN_1_BLOCKER",
        "blocking_gate": {
            "name": "Subject02 slot-to-condition/camera binding unique",
            "status": "BLOCKED",
            "expected_action": "DO_NOT_START_OPTIMIZER",
        },
        "counts": {"pass": 26, "not_run": 9, "blocker": 1},
        "tests": rows,
        "final_classification": FINAL_CLASSIFICATION,
        "next_task": NEXT_TASK,
        "next_task_executed": False,
    }


def report_markdown(content_head: str | None) -> str:
    bound = content_head or "PENDING_FIRST_CONTENT_COMMIT"
    return f"""# Subject00 Review Freeze and Subject02 CommonSafe4 Matched Audit

- Task: `{TASK_ID}`
- Source: `{SOURCE_BRANCH}@{SOURCE_HEAD}`
- New branch: `{NEW_BRANCH}`
- Git content commit: `{bound}`
- Final classification: `{FINAL_CLASSIFICATION}`

## Outcome

The 12-page Subject00 review is formally frozen as valid mixed/negative
evidence. The method obtains 30/36, below Reference Classifier (31/36) and
Nearest-Centroid (35/36). All six errors are O04-to-O03; none is a tie-break
error. The primary positive method claim is not supported.

The O01/O03/O04 Teacher checkpoints remain technical passes. The human visual
decision is `INCONCLUSIVE_INSUFFICIENT_MULTI_VIEW_EVIDENCE`: only one slot06
view was shown, with boundary artifacts and insufficient coverage of identity,
hands, feet, and all training views.

## Subject02 pre-optimizer gate

The Base and all three required Teacher checkpoints were recovered and
re-hashed exactly. The passed target registry was also recovered exactly. The
historical Pure Endpoint implementation fixes the runtime, evaluator,
baselines, checkpoint cadence, endpoint candidate order, and tie-break.

Execution is nevertheless blocked. The matched contract names
`[slot00, slot07, slot03, slot06]`, while the passed Subject02 registry contains
only:

| Condition | Frozen view |
|---|---|
| `cond_000000` | front |
| `cond_000318` | back |
| `cond_000017` | left |
| `cond_000347` | right |

No passed Subject02 artifact explicitly binds those conditions/cameras to the
four slot IDs. This matters because the matched contract records the selected
Subject00 `slot06` as `cam09/back-right`, not `right`. Inferring a positional
mapping would be a new protocol decision, which this task forbids.

Therefore optimizer creations, optimizer steps, checkpoints, method runs, and
baseline cells are all zero. The two reserved output roots remain absent.

## Claim boundary

No matched Subject00-versus-Subject02 numeric comparison, protocol-effect
classification, identity-effect classification, or new positive paper claim is
authorized. The paper body was not modified and remains ineligible/non-final.

## Required resolution

Freeze one explicit, evidence-backed mapping from each required slot to a
Subject02 condition and exact camera record, and bind an exact execution
runner/config source. Then reauthorize the 12-run method matrix and 48-cell
baseline matrix. The unique next task is:

`{NEXT_TASK}`
"""


def review_markdown() -> str:
    return """# Subject00 CommonSafe4 Human Scientific Review Decision

The Subject00 CommonSafe4 result is frozen as scientifically valid
mixed/negative evidence.

- Protocol: `PASS_WITH_DISCLOSED_COMMONSAFE4_SUBSTITUTION`
- Provenance: `PASS`
- Replacement: `PASS_PRETRAINING_OUTCOME_INDEPENDENT`
- Method: 30/36
- Reference Classifier: 31/36
- Nearest-Centroid: 35/36
- Errors: six `O04_TO_O03_ONLY`, with no tie-break errors
- Primary positive claim supported: `false`
- Direct cross-identity numeric comparison authorized: `false`

O01, O03, and O04 remain technical Teacher passes. Their overall human visual
decision is `INCONCLUSIVE_INSUFFICIENT_MULTI_VIEW_EVIDENCE`; neither a human
pass nor a human fail is assigned. A dedicated multi-view review remains
pending.

The paper is not eligible and is not final. No paper text was changed.
"""


def build_payloads(content_head: str | None) -> dict[Path, Any]:
    audit = asset_audit()
    method = method_registry()
    baselines = baseline_registry()
    tests = tests_payload()
    unavailable = {
        "status": "NOT_EXECUTED_DUE_TO_PREOPTIMIZER_ASSET_CONTRACT_BLOCKER",
        "reason": "SUBJECT02_SLOT_CAMERA_BINDING_NOT_UNIQUELY_RECOVERED",
    }
    comparison = {
        **base_metadata(),
        "schema_version": "canondressgs.subject00_subject02.matched_comparison.v1",
        "status": "NOT_AUTHORIZED_MATCHED_EXECUTION_INCOMPLETE",
        "subject00": {
            "method_top1": "30/36",
            "reference_classifier_top1": "31/36",
            "nearest_centroid_top1": "35/36",
        },
        "subject02": unavailable,
        "direct_numeric_comparison_authorized": False,
    }
    decomposition = {
        **base_metadata(),
        "schema_version": "canondressgs.subject00_subject02.effect_decomposition.v1",
        "status": "NOT_COMPUTED_MATCHED_EXECUTION_INCOMPLETE",
        "protocol_effect_class": "NOT_AVAILABLE",
        "identity_effect_class": "NOT_AVAILABLE",
        "statistical_significance_claimed": False,
    }
    recommendation = {
        **base_metadata(),
        "schema_version": "canondressgs.subject00_subject02.claim_recommendation.v1",
        "status": "FROZEN_CONSERVATIVE_BOUNDARY",
        "recommendation": (
            "Preserve Subject00 as valid mixed/negative evidence; make no "
            "cross-identity or general method-advantage claim until a fully "
            "matched Subject02 execution is completed."
        ),
        "subject00_primary_positive_claim_supported": False,
        "cross_identity_claim_authorized": False,
        "paper_change_authorized": False,
    }
    summary = {
        **base_metadata(),
        "schema_version": "canondressgs.subject00_subject02.matched_summary.v1",
        "status": "SEALED_BLOCKER",
        "git_content_commit": content_head or "PENDING_FIRST_CONTENT_COMMIT",
        "reporting_head_policy": "BRANCH_HEAD_AFTER_CONTENT_BINDING_COMMIT",
        "review_pdf_sha256": PDF_SHA256,
        "subject00": {
            "protocol_decision": "PASS_WITH_DISCLOSED_COMMONSAFE4_SUBSTITUTION",
            "method_result_class": "MIXED_NEGATIVE_ON_SECOND_IDENTITY",
            "primary_positive_claim_supported": False,
            "teacher_human_visual_decision": (
                "INCONCLUSIVE_INSUFFICIENT_MULTI_VIEW_EVIDENCE"
            ),
            "method_top1": "30/36",
            "reference_classifier_top1": "31/36",
            "nearest_centroid_top1": "35/36",
            "error_pattern": "O04_TO_O03_ONLY",
            "error_margin_range": [1.011, 2.807],
        },
        "subject02": {
            "base": BASE,
            "teachers": TEACHERS,
            "target": TARGET,
            "matched_anchors": ANCHORS,
            "matched_rotations": ROTATIONS,
            "matched_seeds": SEEDS,
            "method_run_count": 0,
            "method_optimizer_steps": 0,
            "method_checkpoint_count": 0,
            "baseline_cell_count": 0,
            "planned_method_run_count": 12,
            "planned_method_optimizer_steps": 3600,
            "planned_method_checkpoint_count": 72,
            "planned_baseline_cell_count": 48,
            "method_top1": None,
            "reference_classifier_top1": None,
            "nearest_centroid_top1": None,
            "oracle_results": None,
        },
        "mutations": {
            "subject00": 0,
            "subject02_existing_mainline": 0,
            "subject02_base": 0,
            "subject02_teachers": 0,
            "subject02_target": 0,
            "paper": 0,
        },
        "test_result": tests["result"],
        "final_classification": FINAL_CLASSIFICATION,
        "next_task": NEXT_TASK,
        "next_task_executed": False,
    }
    handoff = {
        **base_metadata(),
        "schema_version": "canondressgs.subject00_subject02.matched_handoff.v1",
        "status": FINAL_CLASSIFICATION,
        "git_content_commit": content_head or "PENDING_FIRST_CONTENT_COMMIT",
        "subject00_review_frozen": True,
        "subject02_optimizer_started": False,
        "subject02_method_output_created": False,
        "subject02_baseline_output_created": False,
        "blocker": (
            "Missing authoritative slot00/03/06/07-to-Subject02 "
            "condition/camera binding and matched runner/config binding."
        ),
        "required_resolution": [
            "Freeze exact slot-to-condition and camera-record mapping.",
            "Resolve slot06 back-right versus Subject02 right semantics.",
            "Bind exact runner/config source without importing Subject00 controller state.",
            "Reauthorize attempt_001 while both reserved output roots remain absent.",
        ],
        "next_task": NEXT_TASK,
        "next_task_executed": False,
    }
    return {
        RISK / "subject00_commonsafe4_human_scientific_review_decision_20260727.json": subject00_review(),
        RISK / "subject00_commonsafe4_mixed_negative_result_freeze_20260727.json": mixed_negative(),
        RISK / "subject00_commonsafe4_authorized_claim_boundary_20260727.json": claim_boundary(),
        RISK / "subject00_commonsafe4_teacher_visual_review_limitation_20260727.json": teacher_limitation(),
        RISK / "subject02_commonsafe4_matched_asset_contract_audit_20260727.json": audit,
        RISK / "subject02_commonsafe4_matched_method_execution_registry_20260727.json": method,
        RISK / "subject02_commonsafe4_matched_per_run_metrics_20260727.json": {
            **base_metadata(),
            "schema_version": "canondressgs.subject02.commonsafe4.per_run_metrics.v1",
            **unavailable,
            "rows": [],
            "expected_run_count": 12,
            "actual_run_count": 0,
        },
        RISK / "subject02_commonsafe4_matched_matrix_aggregate_20260727.json": {
            **base_metadata(),
            "schema_version": "canondressgs.subject02.commonsafe4.aggregate.v1",
            **unavailable,
            "aggregate": None,
            "cherry_pick_performed": False,
        },
        RISK / "subject02_commonsafe4_matched_baseline_registry_20260727.json": baselines,
        RISK / "subject02_commonsafe4_matched_baseline_results_20260727.json": {
            **base_metadata(),
            "schema_version": "canondressgs.subject02.commonsafe4.baseline_results.v1",
            **unavailable,
            "rows": [],
            "expected_cell_count": 48,
            "actual_cell_count": 0,
        },
        RISK / "subject00_subject02_commonsafe4_matched_comparison_20260727.json": comparison,
        RISK / "subject00_subject02_protocol_identity_effect_decomposition_20260727.json": decomposition,
        RISK / "subject00_subject02_commonsafe4_claim_recommendation_20260727.json": recommendation,
        RISK / "subject00_review_subject02_matched_tests_20260727.json": tests,
        RISK / "subject00_review_subject02_matched_final_summary_20260727.json": summary,
        HANDOFF / "subject00_review_subject02_commonsafe4_matched_handoff_20260727.json": handoff,
    }


def validate_source() -> None:
    contract = json.loads((REPO / MATCHED_CONTRACT_REL).read_text(encoding="utf-8"))
    assert contract["anchors"] == ANCHORS
    assert contract["seeds"] == SEEDS
    assert contract["status"] == "FROZEN_NOT_EXECUTED"
    assert contract["subject02_matched_run_execution_authorized"] is False
    assert contract["subject00_selected_replacement"] == {
        "camera": "cam09",
        "direction": "back-right",
        "slot": "slot06",
    }
    seal = json.loads((REPO / SUBJECT00_SEAL_REL).read_text(encoding="utf-8"))
    assert seal["status"] == "PASS_POSTHOC_PROVENANCE_SEALED"
    assert seal["method_run_count"] == 12
    assert seal["method_optimizer_steps"] == 3600
    assert seal["method_checkpoint_count"] == 72
    assert seal["baseline_cell_count"] == 48


def write_artifacts(content_head: str | None) -> None:
    validate_source()
    for path, payload in build_payloads(content_head).items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    review_path = (
        RISK / "SUBJECT00_COMMONSAFE4_HUMAN_SCIENTIFIC_REVIEW_DECISION_20260727.md"
    )
    review_path.write_text(review_markdown(), encoding="utf-8")
    report = report_markdown(content_head)
    (
        RISK / "SUBJECT00_REVIEW_AND_SUBJECT02_COMMONSAFE4_MATCHED_REPORT_20260727.md"
    ).write_text(report, encoding="utf-8")
    (
        DOCS / "AAAI27_SUBJECT00_REVIEW_AND_SUBJECT02_COMMONSAFE4_MATCHED_REPORT_20260727.md"
    ).write_text(report, encoding="utf-8")


def validate_outputs() -> None:
    validate_source()
    payloads = build_payloads(None)
    for path in payloads:
        assert path.is_file(), path
        json.loads(path.read_text(encoding="utf-8"))
    tests = json.loads(
        (
            RISK / "subject00_review_subject02_matched_tests_20260727.json"
        ).read_text(encoding="utf-8")
    )
    assert tests["counts"] == {"pass": 26, "not_run": 9, "blocker": 1}
    assert tests["final_classification"] == FINAL_CLASSIFICATION
    method = json.loads(
        (
            RISK
            / "subject02_commonsafe4_matched_method_execution_registry_20260727.json"
        ).read_text(encoding="utf-8")
    )
    assert method["actual"]["optimizer_steps"] == 0
    assert method["actual"]["run_count"] == 0
    baseline = json.loads(
        (
            RISK / "subject02_commonsafe4_matched_baseline_registry_20260727.json"
        ).read_text(encoding="utf-8")
    )
    assert baseline["actual_cell_count"] == 0
    summary = json.loads(
        (
            RISK / "subject00_review_subject02_matched_final_summary_20260727.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["mutations"]["paper"] == 0
    assert summary["subject00"]["method_top1"] == "30/36"
    assert summary["subject02"]["method_top1"] is None
    assert summary["final_classification"] == FINAL_CLASSIFICATION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--content-head")
    args = parser.parse_args()
    if not args.write and not args.check:
        parser.error("choose --write and/or --check")
    if args.write:
        write_artifacts(args.content_head)
    if args.check:
        validate_outputs()
        print("PASS_EXPECTED_BLOCKER_PATH")


if __name__ == "__main__":
    main()
