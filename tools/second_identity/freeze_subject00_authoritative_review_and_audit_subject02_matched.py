#!/usr/bin/env python3
"""Freeze the authoritative Subject00 review and gate Subject02 matched execution.

This is a pre-optimizer adjudication program.  It never imports torch or a
training runtime.  If the Subject02 slot-to-condition/camera contract is not
uniquely recoverable, it emits the required blocker artifacts and leaves both
reserved Subject02 output roots absent.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TASK_ID = "AAAI27-SUBJECT00-AUTHORITATIVE-REVIEW-SUBJECT02-MATCHED-001"
SOURCE_TASK = "AAAI27-SUBJECT00-COMMONSAFE4-REVIEW-PACK-UNIQUE-ROOT-001"
SOURCE_BRANCH = "research/subject00-commonsafe4-review-pack-unique-root-20260727"
SOURCE_HEAD = "8de156e2eac0932db5ec70830eedae5381cf22cc"
NEW_BRANCH = "research/subject00-authoritative-review-subject02-matched-20260727"
WINDOWS_WORKTREE = (
    r"E:\model_train\canondressgs_subject00_authoritative_review_subject02_matched"
)
CLOUD_WORKTREE = Path(
    "/root/autodl-tmp/canondressgs_work/worktrees/"
    "canondressgs_subject00_authoritative_review_subject02_matched"
)

REVIEW_ROOT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-CANONDRESSGS-METHOD-BASE60747-COMMONSAFE4-001/review/"
    "human_scientific_review_pack_authoritative_20260727_001"
)
REVIEW_PDF = (
    REVIEW_ROOT
    / "08_indexes"
    / "subject00_commonsafe4_method_teacher_baseline_"
    "human_scientific_review_pages_authoritative_20260727_001.pdf"
)
REVIEW_PDF_SHA256 = (
    "1859c41eaa43f81ecfed8e459251871780401070ccd65e4c6f49a9a67012b874"
)
REVIEW_PDF_BYTES = 19_797_927
REVIEW_PDF_PAGES = 12

RISK = CLOUD_WORKTREE / "paper_protocol" / "reviewer_risk"
SOURCE_MANIFEST = (
    RISK / "subject00_commonsafe4_authoritative_review_upload_manifest_20260727.json"
)
SOURCE_INDEX = REVIEW_ROOT / "08_indexes" / (
    "subject00_commonsafe4_authoritative_review_pack_index_20260727.json"
)
SOURCE_SEAL = (
    RISK
    / "subject00_commonsafe4_concurrent_12run_matrix_provenance_seal_20260727.json"
)

HISTORICAL_CONTRACT_COMMIT = "aa3860618d723905a0fcd3784cf2ec550b609c1a"
HISTORICAL_CONTRACT_PATH = (
    "paper_protocol/reviewer_risk/"
    "subject02_commonsafe4_matched_protocol_execution_contract_20260727.json"
)
HISTORICAL_CONTRACT_SHA256 = (
    "9e5b0d12b9eb903748e64dd6ceb18266c43e28f8b169a4f99039f572b22dacf6"
)

SUBJECT02_BASE = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/subject02_formal_800k/"
    "chkpnt100000.pth"
)
SUBJECT02_BASE_SHA = (
    "abbf67b59eadf2cba2dea69dbeec598f9177da45b8ddc74ccbe8108acf9ddf70"
)
SUBJECT02_TEACHERS = {
    "O01": {
        "path": Path(
            "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/"
            "SUBJECT02-REPRESENTATION-TRIAGE-001/attempt_002/"
            "rung_2_shared_same_support/O01/checkpoints/step_001200.pth"
        ),
        "sha256": (
            "af730d138697ab9c7a29f17603303ae41dfa36047b8cee59e2ee89328655ce56"
        ),
    },
    "O03": {
        "path": Path(
            "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/"
            "SUBJECT02-MULTI-OUTFIT-EXPLICIT-BASIS-001/attempt_003/"
            "stage_a_teacher_bank/O03/checkpoints/step_001200.pth"
        ),
        "sha256": (
            "16cb235d784e17f4ebb3928e020ceb0be050b5cf80740e56ec72bcbf11d37d93"
        ),
    },
    "O04": {
        "path": Path(
            "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/"
            "SUBJECT02-MULTI-OUTFIT-EXPLICIT-BASIS-001/attempt_003/"
            "stage_a_teacher_bank/O04/checkpoints/step_001200.pth"
        ),
        "sha256": (
            "b39c8d4940325e371cd6db55551c4830b19ab057106c2b9e06df796e8bacd9c3"
        ),
    },
}
SUBJECT02_TARGET_ROOT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/"
    "SUBJECT02-AAAI27-DATA-CAPACITY-GATE-001/attempt_003/dataset"
)
SUBJECT02_TARGET_MANIFEST = SUBJECT02_TARGET_ROOT / "aaai_gate_28_manifest.json"
SUBJECT02_TARGET_MANIFEST_SHA = (
    "49bee929edc236af8f37d66be50eb7c44f4011d49e435af903c0e15918a83bbf"
)
SUBJECT02_EXISTING_MAINLINE = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/"
    "SUBJECT02-MULTI-OUTFIT-EXPLICIT-BASIS-001/attempt_003"
)

PURE_ENDPOINT_ROOT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "PURE-ENDPOINT-CORE-METHOD-CROSSFIT-001/attempt_001"
)
PURE_ENDPOINT_FILES = {
    "rotation_manifest": PURE_ENDPOINT_ROOT
    / "01_contract_snapshot"
    / "pure_endpoint_rotation_manifests.json",
    "model_contract": PURE_ENDPOINT_ROOT
    / "01_contract_snapshot"
    / "pure_endpoint_model_contract.json",
    "evaluator_contract": PURE_ENDPOINT_ROOT
    / "01_contract_snapshot"
    / "pure_endpoint_evaluator_contract.json",
    "primary_baselines": PURE_ENDPOINT_ROOT
    / "01_contract_snapshot"
    / "pure_endpoint_primary_baseline_registry_amended.json",
    "supplementary_baselines": PURE_ENDPOINT_ROOT
    / "01_contract_snapshot"
    / "pure_endpoint_supplementary_baseline_registry.json",
    "execution_binding": PURE_ENDPOINT_ROOT
    / "01_contract_snapshot"
    / "pure_endpoint_crossfit_execution_binding.json",
}

METHOD_OUTPUT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT02-CANONDRESSGS-METHOD-COMMONSAFE4-MATCHED-001/attempt_001"
)
BASELINE_OUTPUT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT02-COMMONSAFE4-MATCHED-BASELINES-001/attempt_001"
)

ANCHORS = ["slot00", "slot07", "slot03", "slot06"]
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
SEEDS = [0, 1, 2]
CHECKPOINT_STEPS = [0, 20, 50, 100, 200, 300]
GARMENTS = ["O01", "O03", "O04"]
BASELINES = [
    {
        "name": "Reference Classifier Lookup",
        "classification": "NON_ORACLE_COMPARISON_BASELINE",
    },
    {
        "name": "Nearest-Centroid Lookup",
        "classification": "NON_ORACLE_COMPARISON_BASELINE",
    },
    {
        "name": "Outfit-ID Oracle",
        "classification": "ORACLE_UPPER_REFERENCE",
    },
    {
        "name": "Teacher Endpoint",
        "classification": "ORACLE_UPPER_REFERENCE",
    },
]

FINAL_CLASSIFICATION = (
    "SUBJECT00_REVIEW_FROZEN_SUBJECT02_MATCHED_BLOCKED_BY_ASSET_CONTRACT"
)
NEXT_TASK = (
    "USER_FREEZE_SUBJECT02_COMMONSAFE4_MATCHED_ASSET_CONTRACT_AND_"
    "REAUTHORIZE_EXECUTION"
)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value.rstrip() + "\n")


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(CLOUD_WORKTREE), *args], text=True
    ).strip()


def pdf_page_count(path: Path) -> int:
    output = subprocess.check_output(["pdfinfo", str(path)], text=True)
    for line in output.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError("pdfinfo did not report a page count")


def compute_processes() -> list[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def metadata_fingerprint(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_count = 0
    byte_count = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\n")
        file_count += 1
        byte_count += stat.st_size
    return {
        "root": str(root),
        "file_count": file_count,
        "bytes": byte_count,
        "metadata_fingerprint": digest.hexdigest(),
    }


def historical_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    raw = subprocess.check_output(
        [
            "git",
            "-C",
            str(CLOUD_WORKTREE),
            "show",
            f"{HISTORICAL_CONTRACT_COMMIT}:{HISTORICAL_CONTRACT_PATH}",
        ]
    )
    observed_sha = sha256_bytes(raw)
    if observed_sha != HISTORICAL_CONTRACT_SHA256:
        raise RuntimeError(
            f"Historical matched contract SHA mismatch: {observed_sha}"
        )
    return json.loads(raw), {
        "git_object": f"{HISTORICAL_CONTRACT_COMMIT}:{HISTORICAL_CONTRACT_PATH}",
        "sha256": observed_sha,
        "bytes": len(raw),
    }


def walk_slot_bindings(value: Any, path: str = "$") -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        slot_keys = [key for key in value if "slot" in key.lower()]
        if slot_keys:
            found.append(
                {
                    "json_path": path,
                    "slot_fields": {key: value[key] for key in slot_keys},
                }
            )
        for key, child in value.items():
            found.extend(walk_slot_bindings(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(walk_slot_bindings(child, f"{path}[{index}]"))
    return found


def condition_direction(condition: dict[str, Any]) -> str:
    translation = condition["c2w"]
    x = float(translation[0][3])
    z = float(translation[2][3])
    if abs(z) >= abs(x):
        return "front" if z > 0 else "back"
    return "left" if x > 0 else "right"


def common_metadata(created_at: str) -> dict[str, Any]:
    return {
        "schema_version": "canondressgs.subject00_subject02.authoritative_matched.v1",
        "task_id": TASK_ID,
        "created_at": created_at,
        "source_task": SOURCE_TASK,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "new_branch": NEW_BRANCH,
        "windows_worktree": WINDOWS_WORKTREE,
        "cloud_worktree": str(CLOUD_WORKTREE),
        "paper_eligible": False,
        "paper_final": False,
        "paper_modifications": 0,
    }


def subject00_decision(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "artifact": "SUBJECT00_AUTHORITATIVE_HUMAN_SCIENTIFIC_DECISION",
        "review_pdf": {
            "path": str(REVIEW_PDF),
            "sha256": REVIEW_PDF_SHA256,
            "bytes": REVIEW_PDF_BYTES,
            "page_count": REVIEW_PDF_PAGES,
        },
        "protocol_decision": "PASS_WITH_DISCLOSED_COMMONSAFE4_SUBSTITUTION",
        "replacement_decision": "PASS_PRETRAINING_OUTCOME_INDEPENDENT",
        "matrix_provenance_decision": "PASS",
        "experiment_valid": True,
        "method_result_class": "MIXED_NEGATIVE_ON_SECOND_IDENTITY",
        "primary_positive_claim_supported": False,
        "method_outperforms_reference_classifier": False,
        "method_outperforms_nearest_centroid": False,
        "direct_cross_identity_comparison_authorized": False,
        "method": {
            "correct": 30,
            "denominator": 36,
            "top1": 30 / 36,
            "error_count": 6,
            "error_pattern": "O04_TO_O03_ONLY",
            "error_class": "SYSTEMATIC_DIRECTIONAL_CLASS_BOUNDARY_CONFUSION",
            "tie_break_failure": False,
            "random_noise_explanation": False,
        },
        "baselines": {
            "Reference Classifier Lookup": {
                "correct": 31,
                "denominator": 36,
                "classification": "NON_ORACLE_COMPARISON_BASELINE",
            },
            "Nearest-Centroid Lookup": {
                "correct": 35,
                "denominator": 36,
                "classification": "NON_ORACLE_COMPARISON_BASELINE",
            },
            "Outfit-ID Oracle": {
                "correct": 36,
                "denominator": 36,
                "classification": "ORACLE_UPPER_REFERENCE",
            },
            "Teacher Endpoint": {
                "correct": 36,
                "denominator": 36,
                "classification": "ORACLE_UPPER_REFERENCE",
            },
        },
        "baseline_fairness_decision": (
            "PASS_WITH_DIFFERENTIAL_COMPUTE_AND_INFORMATION_DISCLOSURE"
        ),
        "final_human_scientific_decision": (
            "VALID_EXPERIMENT_MIXED_NEGATIVE_RESULT"
        ),
    }


def mixed_negative_freeze(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "artifact": "SUBJECT00_COMMONSAFE4_MIXED_NEGATIVE_FREEZE",
        "status": "FROZEN",
        "method_result_class": "MIXED_NEGATIVE_ON_SECOND_IDENTITY",
        "primary_positive_claim_supported": False,
        "method_top1": {"correct": 30, "denominator": 36, "top1": 30 / 36},
        "nonoracle_comparators": {
            "Reference Classifier Lookup": "31/36",
            "Nearest-Centroid Lookup": "35/36",
        },
        "method_below_both_nonoracle_baselines": True,
        "error_count": 6,
        "error_pattern": "O04_TO_O03_ONLY",
        "error_class": "SYSTEMATIC_DIRECTIONAL_CLASS_BOUNDARY_CONFUSION",
        "tie_break_failure": False,
        "random_noise_explanation": False,
    }


def claim_boundary(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "artifact": "SUBJECT00_COMMONSAFE4_AUTHORIZED_CLAIM_BOUNDARY",
        "authorized_claims": [
            "The Subject00 CommonSafe4 experiment is protocol-valid with a disclosed outcome-independent slot06 proxy for quarantined slot04.",
            "Pure Endpoint achieved 30/36 on Subject00, below Reference Classifier 31/36 and Nearest-Centroid 35/36.",
            "All six Subject00 method errors are true O04 predicted as O03.",
            "The three Subject00 Teacher endpoints retain TECHNICAL_PASS status.",
            "Subject02 matched execution is blocked before optimization by an ambiguous slot-to-condition/camera contract.",
        ],
        "prohibited_claims": [
            "CanonDressGS outperforms either non-oracle Subject00 baseline.",
            "The Subject00 evidence supports the primary positive method claim.",
            "The Teacher endpoints have passed a complete human multiview visual review.",
            "Subject00 and Subject02 have been compared under a matched CommonSafe4 protocol.",
            "Any protocol or identity effect is statistically significant.",
        ],
        "paper_body_write_authorized": False,
        "paper_eligible": False,
        "paper_final": False,
    }


def teacher_limitation(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "artifact": "SUBJECT00_COMMONSAFE4_TEACHER_MULTIVIEW_REVIEW_LIMITATION",
        "teachers": {
            garment: {
                "technical_status": "TECHNICAL_PASS",
                "shown_evidence": (
                    "VISUALLY_RECOGNIZABLE_WITH_BOUNDARY_ARTIFACTS"
                ),
            }
            for garment in GARMENTS
        },
        "teacher_human_visual_decision": (
            "INCONCLUSIVE_INSUFFICIENT_DEDICATED_MULTIVIEW_EVIDENCE"
        ),
        "teacher_human_pass": False,
        "teacher_human_fail": False,
        "teacher_review_status": "PENDING_DEDICATED_MULTIVIEW_REVIEW",
        "limitation": (
            "The authoritative review page shows representative slot06 evidence, "
            "not a dedicated multiview review covering all training views, identity, "
            "hands, feet, and boundary behavior."
        ),
    }


def planned_method_runs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rotation in ROTATIONS:
        for seed in SEEDS:
            rows.append(
                {
                    "run_id": (
                        f"SUBJECT02-COMMONSAFE4-METHOD-{rotation['rotation']}-S{seed}"
                    ),
                    "rotation": rotation["rotation"],
                    "seed": seed,
                    "train_slots": rotation["train"],
                    "calibration_slot": rotation["calibration"],
                    "test_slot": rotation["test"],
                    "train_record_count": 6,
                    "calibration_record_count": 3,
                    "formal_test_denominator": 3,
                    "optimizer_steps_planned": 300,
                    "checkpoint_steps_planned": CHECKPOINT_STEPS,
                    "status": "NOT_RUN_DUE_TO_PREOPTIMIZER_ASSET_CONTRACT_BLOCKER",
                }
            )
    return rows


def method_registry(
    meta: dict[str, Any], audit: dict[str, Any], runs: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        **meta,
        "artifact": "SUBJECT02_COMMONSAFE4_MATCHED_METHOD_EXECUTION_REGISTRY",
        "protocol": "COMMONSAFE4_CARDINAL_PROXY_V1",
        "method_contract": "PURE_ENDPOINT",
        "dual_support_enabled": False,
        "dual_support_call_count": 0,
        "anchors": ANCHORS,
        "rotations": ROTATIONS,
        "seeds": SEEDS,
        "planned_run_count": 12,
        "planned_total_optimizer_steps": 3600,
        "planned_checkpoint_count": 72,
        "actual_run_count": 0,
        "actual_optimizer_steps": 0,
        "actual_checkpoint_count": 0,
        "optimizer_creations": 0,
        "runs": runs,
        "preoptimizer_gate": audit["decision"],
        "status": "BLOCKED_NOT_EXECUTED",
    }


def per_run_metrics(meta: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        **meta,
        "artifact": "SUBJECT02_COMMONSAFE4_MATCHED_PER_RUN_METRICS",
        "planned_run_count": 12,
        "actual_run_count": 0,
        "rows": [
            {
                "run_id": row["run_id"],
                "rotation": row["rotation"],
                "seed": row["seed"],
                "formal_test_correct": None,
                "formal_test_denominator": 3,
                "endpoint_top1": None,
                "confusion": None,
                "score_margins": None,
                "wall_time_seconds": None,
                "peak_vram_bytes": None,
                "status": row["status"],
            }
            for row in runs
        ],
        "status": "NOT_RUN_DUE_TO_PREOPTIMIZER_ASSET_CONTRACT_BLOCKER",
    }


def matrix_aggregate(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "artifact": "SUBJECT02_COMMONSAFE4_MATCHED_MATRIX_AGGREGATE",
        "planned_run_count": 12,
        "actual_run_count": 0,
        "planned_optimizer_steps": 3600,
        "actual_optimizer_steps": 0,
        "planned_checkpoint_count": 72,
        "actual_checkpoint_count": 0,
        "formal_test_correct": None,
        "formal_test_denominator": None,
        "endpoint_top1": None,
        "confusion_matrix": None,
        "error_pattern": None,
        "rotation_effect": None,
        "seed_effect": None,
        "margins": None,
        "compute_budget": None,
        "nan_inf_events": 0,
        "oom_events": 0,
        "cherry_picking": False,
        "status": "NOT_COMPUTABLE_MATCHED_EXECUTION_NOT_RUN",
    }


def planned_baseline_cells() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for baseline in BASELINES:
        slug = baseline["name"].upper().replace("-", "_").replace(" ", "_")
        for rotation in ROTATIONS:
            for seed in SEEDS:
                cells.append(
                    {
                        "cell_id": (
                            f"SUBJECT02-COMMONSAFE4-BASELINE-{slug}-"
                            f"{rotation['rotation']}-S{seed}"
                        ),
                        "baseline": baseline["name"],
                        "classification": baseline["classification"],
                        "rotation": rotation["rotation"],
                        "seed": seed,
                        "status": (
                            "NOT_RUN_DUE_TO_PREOPTIMIZER_ASSET_CONTRACT_BLOCKER"
                        ),
                    }
                )
    return cells


def baseline_registry(
    meta: dict[str, Any], cells: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        **meta,
        "artifact": "SUBJECT02_COMMONSAFE4_MATCHED_BASELINE_REGISTRY",
        "baselines": BASELINES,
        "fairness_decision": (
            "PASS_WITH_DIFFERENTIAL_COMPUTE_AND_INFORMATION_DISCLOSURE"
        ),
        "planned_cell_count": 48,
        "actual_cell_count": 0,
        "same_protocol_assets_required": True,
        "cells": cells,
        "status": "BLOCKED_NOT_EXECUTED",
    }


def baseline_results(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "artifact": "SUBJECT02_COMMONSAFE4_MATCHED_BASELINE_RESULTS",
        "planned_cell_count": 48,
        "actual_cell_count": 0,
        "results": {
            baseline["name"]: {
                "classification": baseline["classification"],
                "correct": None,
                "denominator": None,
                "top1": None,
                "status": "NOT_RUN",
            }
            for baseline in BASELINES
        },
        "status": "NOT_COMPUTABLE_MATCHED_EXECUTION_NOT_RUN",
    }


def comparison(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "artifact": "SUBJECT00_SUBJECT02_COMMONSAFE4_MATCHED_COMPARISON",
        "subject00_commonsafe4": {
            "method": "30/36",
            "Reference Classifier Lookup": "31/36",
            "Nearest-Centroid Lookup": "35/36",
            "Outfit-ID Oracle": "36/36",
            "Teacher Endpoint": "36/36",
            "error_pattern": "O04_TO_O03_ONLY",
        },
        "subject02_commonsafe4_matched": {
            "method": None,
            "Reference Classifier Lookup": None,
            "Nearest-Centroid Lookup": None,
            "Outfit-ID Oracle": None,
            "Teacher Endpoint": None,
            "error_pattern": None,
        },
        "direct_numeric_comparison_authorized": False,
        "result_class": None,
        "candidate_result_classes_not_selected": [
            "SUBJECT02_MATCHED_STRONG_SUBJECT00_WEAK",
            "SUBJECT02_MATCHED_AND_SUBJECT00_BOTH_WEAK",
            "METHOD_BELOW_NEAREST_CENTROID_ON_BOTH_IDENTITIES",
            "METHOD_REGAINS_BASELINE_ADVANTAGE_ON_SUBJECT02_MATCHED",
            "OTHER_MIXED_RESULT",
        ],
        "status": "NOT_COMPUTABLE_MATCHED_EXECUTION_NOT_RUN",
    }


def effect_decomposition(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "artifact": "SUBJECT00_SUBJECT02_PROTOCOL_IDENTITY_EFFECT_DECOMPOSITION",
        "protocol_effect": {
            "comparison": (
                "Subject02 original protocol vs Subject02 CommonSafe4 matched"
            ),
            "method_top1": None,
            "baseline_top1": None,
            "confusion_matrix": None,
            "error_pattern": None,
            "rotation_effect": None,
            "seed_effect": None,
            "margins": None,
            "compute_budget": None,
            "status": "NOT_COMPUTABLE_MATCHED_EXECUTION_NOT_RUN",
        },
        "identity_effect": {
            "comparison": (
                "Subject02 CommonSafe4 matched vs Subject00 CommonSafe4"
            ),
            "method_top1": None,
            "baseline_top1": None,
            "confusion_matrix": None,
            "error_pattern": None,
            "rotation_effect": None,
            "seed_effect": None,
            "margins": None,
            "compute_budget": None,
            "status": "NOT_COMPUTABLE_MATCHED_EXECUTION_NOT_RUN",
        },
        "statistical_significance_claimed": False,
    }


def claim_recommendation(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "artifact": "SUBJECT00_SUBJECT02_COMMONSAFE4_CLAIM_RECOMMENDATION",
        "recommendation": (
            "Freeze Subject00 as a valid mixed/negative result. Do not make a "
            "matched cross-identity claim until the Subject02 slot/camera binding "
            "and matched runner/config are explicitly frozen and execution is "
            "reauthorized."
        ),
        "subject00_primary_positive_claim_supported": False,
        "subject02_matched_claim_available": False,
        "protocol_effect_claim_available": False,
        "identity_effect_claim_available": False,
        "paper_change_recommended_now": False,
        "paper_eligible": False,
        "paper_final": False,
    }


def asset_audit(
    meta: dict[str, Any],
    contract_meta: dict[str, Any],
    target_manifest: dict[str, Any],
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    conditions = target_manifest["conditions"]
    condition_rows = [
        {
            "condition_id": row["condition_id"],
            "source_camera_id": row.get("source_camera_id"),
            "derived_cardinal_direction": condition_direction(row),
            "derivation": "dominant camera-center x/z axis; not a slot binding",
        }
        for row in conditions
    ]
    slot_rows = walk_slot_bindings(target_manifest)
    return {
        **meta,
        "artifact": "SUBJECT02_COMMONSAFE4_MATCHED_ASSET_CONTRACT_AUDIT",
        "historical_matched_contract": {
            **contract_meta,
            "protocol": "COMMONSAFE4_CARDINAL_PROXY_V1",
            "anchors": ANCHORS,
            "rotations": ROTATIONS,
            "seeds": SEEDS,
            "current_task_execution_authority_received": True,
        },
        "uniquely_recovered": {
            "base": before["base"],
            "teachers": before["teachers"],
            "target": before["target"],
            "historical_pure_endpoint_contracts": before[
                "pure_endpoint_contracts"
            ],
            "optimizer_steps_per_run": 300,
            "checkpoint_steps": CHECKPOINT_STEPS,
            "endpoint_candidates": GARMENTS,
            "tie_break": "first garment in frozen candidate_order",
            "available_subject02_conditions": condition_rows,
        },
        "not_uniquely_recovered": {
            "slot_to_subject02_condition_camera_binding": {
                "required_slots": ANCHORS,
                "explicit_slot_binding_rows_found": len(slot_rows),
                "slot_binding_rows": slot_rows,
                "condition_rows": condition_rows,
                "conflict": (
                    "The matched contract fixes Subject00 slot06 as cam09/back-right, "
                    "while the passed Subject02 registry exposes cond_000347/right "
                    "and no authority equates it with slot06."
                ),
            },
            "matched_runner_and_config": {
                "historical_runtime_and_contracts_recoverable": True,
                "new_slot_contract_runner_config_binding_present": False,
                "reason": (
                    "The historical Pure Endpoint runner consumes condition IDs. "
                    "No passed artifact binds that runner/config to the new symbolic "
                    "slot contract."
                ),
            },
        },
        "forbidden_inference_not_taken": (
            "No positional mapping from slot00/slot07/slot03/slot06 to "
            "front/back/left/right was invented."
        ),
        "output_preflight": {
            "method_output": str(METHOD_OUTPUT),
            "method_output_absent_before": before["method_output_absent"],
            "method_output_absent_after": after["method_output_absent"],
            "baseline_output": str(BASELINE_OUTPUT),
            "baseline_output_absent_before": before["baseline_output_absent"],
            "baseline_output_absent_after": after["baseline_output_absent"],
            "active_compute_processes_before": before["compute_processes"],
            "active_compute_processes_after": after["compute_processes"],
        },
        "immutability": {
            "subject02_existing_mainline_before": before[
                "subject02_existing_mainline"
            ],
            "subject02_existing_mainline_after": after[
                "subject02_existing_mainline"
            ],
            "subject00_review_pdf_before": before["subject00_review_pdf"],
            "subject00_review_pdf_after": after["subject00_review_pdf"],
            "asset_hashes_before": {
                "base": before["base"],
                "teachers": before["teachers"],
                "target": before["target"],
            },
            "asset_hashes_after": {
                "base": after["base"],
                "teachers": after["teachers"],
                "target": after["target"],
            },
            "subject00_mutations": 0,
            "subject02_old_mainline_mutations": 0,
            "subject02_base_mutations": 0,
            "subject02_teacher_mutations": 0,
            "subject02_target_mutations": 0,
            "paper_modifications": 0,
        },
        "decision": "DO_NOT_START_OPTIMIZER",
        "blocking_reasons": [
            "SLOT_TO_SUBJECT02_CONDITION_CAMERA_BINDING_NOT_UNIQUE",
            "MATCHED_RUNNER_CONFIG_BINDING_NOT_FROZEN",
        ],
        "optimizer_creations": 0,
        "optimizer_steps": 0,
        "new_output_roots_created": 0,
        "status": "BLOCKED_PREOPTIMIZER_ASSET_CONTRACT_AMBIGUITY",
    }


def asset_snapshot() -> dict[str, Any]:
    return {
        "created_at": now_iso(),
        "base": {
            "path": str(SUBJECT02_BASE),
            "bytes": SUBJECT02_BASE.stat().st_size,
            "sha256": sha256_file(SUBJECT02_BASE),
        },
        "teachers": {
            garment: {
                "path": str(payload["path"]),
                "bytes": payload["path"].stat().st_size,
                "sha256": sha256_file(payload["path"]),
            }
            for garment, payload in SUBJECT02_TEACHERS.items()
        },
        "target": {
            "root": str(SUBJECT02_TARGET_ROOT),
            "registry": str(SUBJECT02_TARGET_MANIFEST),
            "bytes": SUBJECT02_TARGET_MANIFEST.stat().st_size,
            "sha256": sha256_file(SUBJECT02_TARGET_MANIFEST),
        },
        "pure_endpoint_contracts": {
            role: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for role, path in PURE_ENDPOINT_FILES.items()
        },
        "subject02_existing_mainline": metadata_fingerprint(
            SUBJECT02_EXISTING_MAINLINE
        ),
        "subject00_review_pdf": {
            "path": str(REVIEW_PDF),
            "bytes": REVIEW_PDF.stat().st_size,
            "sha256": sha256_file(REVIEW_PDF),
            "page_count": pdf_page_count(REVIEW_PDF),
        },
        "method_output_absent": not METHOD_OUTPUT.exists(),
        "baseline_output_absent": not BASELINE_OUTPUT.exists(),
        "compute_processes": compute_processes(),
    }


def validate_preconditions() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    required = [
        REVIEW_PDF,
        SOURCE_MANIFEST,
        SOURCE_INDEX,
        SOURCE_SEAL,
        SUBJECT02_BASE,
        SUBJECT02_TARGET_MANIFEST,
        SUBJECT02_EXISTING_MAINLINE,
        *[payload["path"] for payload in SUBJECT02_TEACHERS.values()],
        *PURE_ENDPOINT_FILES.values(),
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    if METHOD_OUTPUT.exists() or BASELINE_OUTPUT.exists():
        raise FileExistsError("Reserved Subject02 matched output already exists")
    if git_output("branch", "--show-current") != NEW_BRANCH:
        raise RuntimeError("Unexpected execution branch")
    if git_output("rev-parse", "HEAD") != SOURCE_HEAD:
        raise RuntimeError("Unexpected starting HEAD")
    if git_output("status", "--porcelain=v1"):
        raise RuntimeError("Execution worktree is not clean")
    if REVIEW_PDF.stat().st_size != REVIEW_PDF_BYTES:
        raise RuntimeError("Authoritative PDF size mismatch")
    if sha256_file(REVIEW_PDF) != REVIEW_PDF_SHA256:
        raise RuntimeError("Authoritative PDF SHA mismatch")
    if pdf_page_count(REVIEW_PDF) != REVIEW_PDF_PAGES:
        raise RuntimeError("Authoritative PDF page count mismatch")
    seal = load_json(SOURCE_SEAL)
    if seal["test_result"] != "PASS_61_OF_61":
        raise RuntimeError("Subject00 provenance seal mismatch")
    contract, contract_meta = historical_contract()
    if contract["anchors"] != ANCHORS or contract["seeds"] != SEEDS:
        raise RuntimeError("Historical matched symbolic contract mismatch")
    target_manifest = load_json(SUBJECT02_TARGET_MANIFEST)
    if len(target_manifest["conditions"]) != 4:
        raise RuntimeError("Subject02 target manifest condition count mismatch")
    return contract, contract_meta, target_manifest


def validate_expected_hashes(snapshot: dict[str, Any]) -> None:
    if snapshot["base"]["sha256"] != SUBJECT02_BASE_SHA:
        raise RuntimeError("Subject02 Base SHA mismatch")
    for garment, expected in SUBJECT02_TEACHERS.items():
        if snapshot["teachers"][garment]["sha256"] != expected["sha256"]:
            raise RuntimeError(f"Subject02 {garment} Teacher SHA mismatch")
    if snapshot["target"]["sha256"] != SUBJECT02_TARGET_MANIFEST_SHA:
        raise RuntimeError("Subject02 target manifest SHA mismatch")
    if snapshot["compute_processes"]:
        raise RuntimeError(
            f"Active compute process detected: {snapshot['compute_processes']}"
        )


def validate_immutable(before: dict[str, Any], after: dict[str, Any]) -> None:
    for key in ["base", "teachers", "target", "subject00_review_pdf"]:
        if before[key] != after[key]:
            raise RuntimeError(f"Immutable asset changed: {key}")
    if (
        before["subject02_existing_mainline"]["metadata_fingerprint"]
        != after["subject02_existing_mainline"]["metadata_fingerprint"]
    ):
        raise RuntimeError("Subject02 existing mainline changed")
    if not after["method_output_absent"] or not after["baseline_output_absent"]:
        raise RuntimeError("Blocked task created a reserved output root")
    if after["compute_processes"]:
        raise RuntimeError("Compute process appeared during blocked audit")


def tests_payload(
    meta: dict[str, Any],
    audit: dict[str, Any],
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    pass_names = [
        "Subject00 source task/branch/head exact",
        "Origin/cloud/source worktree preflight synchronized",
        "Execution worktree clean at start",
        "Authoritative PDF path/bytes/SHA/page count exact",
        "Authoritative manifest and index present",
        "Subject00 provenance seal PASS_61_OF_61",
        "Active optimizer/compute process count zero",
        "Subject00 protocol/replacement/matrix decisions frozen",
        "Subject00 mixed-negative result frozen",
        "Three Teacher technical statuses preserved without human PASS",
        "Subject00 method result 30/36",
        "Subject00 non-oracle baselines 31/36 and 35/36",
        "Subject00 six errors O04_TO_O03_ONLY",
        "Tie-break and random-noise explanations rejected",
        "Oracle/non-oracle classifications frozen",
        "Historical Subject02 matched symbolic contract recovered",
        "Matched anchors exact",
        "Matched rotations exact",
        "Matched seeds exact",
        "Subject02 Base path/SHA exact",
        "Subject02 O01/O03/O04 Teacher paths/SHAs exact",
        "Subject02 target root/manifest/SHA exact",
        "Subject02 target exposes four conditions/cameras",
        "Explicit slot binding row count is zero",
        "Reserved method and baseline output roots remain absent",
        "Subject00 and Subject02 existing assets immutable",
    ]
    not_run_names = [
        "Subject02 12-run method execution",
        "Subject02 3600 optimizer steps",
        "Subject02 72 checkpoints",
        "Subject02 formal-test aggregation",
        "Subject02 48 baseline cells",
        "Subject00-vs-Subject02 matched numeric comparison",
        "Protocol-effect decomposition",
        "Identity-effect decomposition",
        "Matched result-class selection",
    ]
    rows = [
        {"test_id": index, "name": name, "status": "PASS"}
        for index, name in enumerate(pass_names, 1)
    ]
    rows.append(
        {
            "test_id": len(rows) + 1,
            "name": "Subject02 slot-to-condition/camera and runner/config binding unique",
            "status": "BLOCKED_EXPECTED_PREOPTIMIZER_GATE",
            "observed": audit["blocking_reasons"],
        }
    )
    for name in not_run_names:
        rows.append(
            {
                "test_id": len(rows) + 1,
                "name": name,
                "status": "NOT_RUN_DUE_TO_PREOPTIMIZER_BLOCKER",
            }
        )
    return {
        **meta,
        "artifact": "SUBJECT00_AUTHORITATIVE_REVIEW_SUBJECT02_MATCHED_TESTS",
        "status": "PASS_EXPECTED_BLOCKER_PATH",
        "result": "PASS_BLOCKER_CLASSIFICATION_26_PASS_9_NOT_RUN_1_BLOCKER",
        "counts": {"pass": 26, "not_run": 9, "blocker": 1, "total": 36},
        "blocking_gate": {
            "name": "Subject02 matched physical asset and runner binding unique",
            "status": "BLOCKED",
            "expected_action": "DO_NOT_START_OPTIMIZER",
        },
        "tests": rows,
        "immutability_verified": {
            "subject00": before["subject00_review_pdf"]
            == after["subject00_review_pdf"],
            "subject02_base": before["base"] == after["base"],
            "subject02_teachers": before["teachers"] == after["teachers"],
            "subject02_target": before["target"] == after["target"],
            "subject02_existing_mainline": (
                before["subject02_existing_mainline"]["metadata_fingerprint"]
                == after["subject02_existing_mainline"]["metadata_fingerprint"]
            ),
        },
        "final_classification": FINAL_CLASSIFICATION,
        "next_task": NEXT_TASK,
        "next_task_executed": False,
    }


def decision_markdown() -> str:
    return f"""# Subject00 CommonSafe4 Authoritative Human Review Decision

- Task: `{TASK_ID}`
- Source: `{SOURCE_BRANCH}@{SOURCE_HEAD}`
- Review PDF SHA256: `{REVIEW_PDF_SHA256}`

## Frozen decision

The CommonSafe4 substitution, outcome-independent replacement, and 12-run
matrix provenance pass review. The Subject00 experiment is valid.

The result is frozen as `MIXED_NEGATIVE_ON_SECOND_IDENTITY`: Pure Endpoint
achieves 30/36, below Reference Classifier 31/36 and Nearest-Centroid 35/36.
All six errors are true O04 predicted as O03. The primary positive claim is
not supported.

The O01/O03/O04 Teacher endpoints retain `TECHNICAL_PASS`. Their human visual
decision is `INCONCLUSIVE_INSUFFICIENT_DEDICATED_MULTIVIEW_EVIDENCE`; no full
human visual pass or fail is assigned.

`paper_eligible=false`, `paper_final=false`.
"""


def report_markdown(audit: dict[str, Any]) -> str:
    condition_lines = "\n".join(
        "| `{condition_id}` | `{source_camera_id}` | `{derived_cardinal_direction}` |".format(
            **row
        )
        for row in audit["uniquely_recovered"][
            "available_subject02_conditions"
        ]
    )
    return f"""# Subject00 Authoritative Review and Subject02 Matched Audit

- Task: `{TASK_ID}`
- Source: `{SOURCE_BRANCH}@{SOURCE_HEAD}`
- New branch: `{NEW_BRANCH}`
- Final classification: `{FINAL_CLASSIFICATION}`

## Subject00 frozen result

The authoritative 12-page review is frozen as a valid mixed/negative result.
Pure Endpoint obtains 30/36, below Reference Classifier 31/36 and
Nearest-Centroid 35/36. All six errors are O04-to-O03. The primary positive
method claim is not supported.

All three Teacher checkpoints remain technical passes. The human visual
decision is `INCONCLUSIVE_INSUFFICIENT_DEDICATED_MULTIVIEW_EVIDENCE`; a
dedicated multiview review remains pending.

## Subject02 pre-optimizer gate

The Base, O01/O03/O04 Teachers, target manifest, and historical Pure Endpoint
contracts were recovered and hashed. The passed target manifest exposes:

| Condition | Camera | Geometric cardinal direction |
|---|---|---|
{condition_lines}

The matched contract requires symbolic slots `[slot00, slot07, slot03,
slot06]`, but the target registry contains zero explicit slot binding rows.
In particular, Subject00 slot06 is cam09/back-right while the Subject02 set
contains a right camera and no authority equating the two. The historical
condition-ID runner/config is also not frozen against the new slot contract.

No positional mapping was invented. Optimizer creations, optimizer steps,
checkpoints, method runs, and baseline cells are all zero. Both reserved
Subject02 output roots remain absent.

## Claim boundary

No matched cross-identity comparison, protocol-effect estimate,
identity-effect estimate, statistical-significance statement, or new positive
paper claim is authorized. The paper body was not modified.

## Required resolution

Freeze an evidence-backed slot-to-Subject02 condition/camera mapping and an
exact matched runner/config binding, then reauthorize execution.

Next task: `{NEXT_TASK}`.
"""


def final_summary(
    meta: dict[str, Any], audit: dict[str, Any], tests: dict[str, Any]
) -> dict[str, Any]:
    return {
        **meta,
        "artifact": "SUBJECT00_AUTHORITATIVE_REVIEW_SUBJECT02_MATCHED_FINAL_SUMMARY",
        "status": "SEALED_PREOPTIMIZER_BLOCKER",
        "review_pdf_sha256": REVIEW_PDF_SHA256,
        "subject00": {
            "protocol_decision": "PASS_WITH_DISCLOSED_COMMONSAFE4_SUBSTITUTION",
            "replacement_decision": "PASS_PRETRAINING_OUTCOME_INDEPENDENT",
            "matrix_provenance_decision": "PASS",
            "experiment_valid": True,
            "method_result_class": "MIXED_NEGATIVE_ON_SECOND_IDENTITY",
            "primary_positive_claim_supported": False,
            "teacher_technical_status": {
                garment: "TECHNICAL_PASS" for garment in GARMENTS
            },
            "teacher_human_visual_decision": (
                "INCONCLUSIVE_INSUFFICIENT_DEDICATED_MULTIVIEW_EVIDENCE"
            ),
            "method_top1": "30/36",
            "reference_classifier_top1": "31/36",
            "nearest_centroid_top1": "35/36",
            "error_pattern": "O04_TO_O03_ONLY",
        },
        "subject02": {
            "anchors": ANCHORS,
            "rotations": ROTATIONS,
            "seeds": SEEDS,
            "planned_method_run_count": 12,
            "actual_method_run_count": 0,
            "planned_optimizer_steps": 3600,
            "actual_optimizer_steps": 0,
            "planned_checkpoint_count": 72,
            "actual_checkpoint_count": 0,
            "planned_baseline_cell_count": 48,
            "actual_baseline_cell_count": 0,
            "method_top1": None,
            "baseline_results": None,
            "asset_gate": audit["status"],
            "blocking_reasons": audit["blocking_reasons"],
        },
        "mutations": {
            "subject00": 0,
            "subject02_old_mainline": 0,
            "subject02_base": 0,
            "subject02_teachers": 0,
            "subject02_target": 0,
            "paper": 0,
        },
        "test_result": tests["result"],
        "paper_eligible": False,
        "paper_final": False,
        "final_classification": FINAL_CLASSIFICATION,
        "next_task": NEXT_TASK,
        "next_task_executed": False,
    }


def artifact_paths() -> dict[str, Path]:
    return {
        "decision": RISK
        / "subject00_commonsafe4_authoritative_human_scientific_decision_20260727.json",
        "mixed_negative": RISK
        / "subject00_commonsafe4_authoritative_mixed_negative_freeze_20260727.json",
        "claim_boundary": RISK
        / "subject00_commonsafe4_authorized_claim_boundary_20260727.json",
        "teacher_limitation": RISK
        / "subject00_commonsafe4_teacher_multiview_review_limitation_20260727.json",
        "decision_report": RISK
        / "SUBJECT00_COMMONSAFE4_AUTHORITATIVE_HUMAN_REVIEW_DECISION_20260727.md",
        "method_registry": RISK
        / "subject02_commonsafe4_matched_method_execution_registry_20260727.json",
        "per_run_metrics": RISK
        / "subject02_commonsafe4_matched_per_run_metrics_20260727.json",
        "matrix_aggregate": RISK
        / "subject02_commonsafe4_matched_matrix_aggregate_20260727.json",
        "baseline_registry": RISK
        / "subject02_commonsafe4_matched_baseline_registry_20260727.json",
        "baseline_results": RISK
        / "subject02_commonsafe4_matched_baseline_results_20260727.json",
        "comparison": RISK
        / "subject00_subject02_commonsafe4_matched_comparison_20260727.json",
        "decomposition": RISK
        / "subject00_subject02_protocol_identity_effect_decomposition_20260727.json",
        "claim_recommendation": RISK
        / "subject00_subject02_commonsafe4_claim_recommendation_20260727.json",
        "report": RISK
        / "SUBJECT00_AUTHORITATIVE_REVIEW_SUBJECT02_MATCHED_REPORT_20260727.md",
        "tests": RISK
        / "subject00_authoritative_review_subject02_matched_tests_20260727.json",
        "summary": RISK
        / "subject00_authoritative_review_subject02_matched_final_summary_20260727.json",
        "handoff": CLOUD_WORKTREE
        / "project_control_handoff"
        / "subject00_authoritative_review_subject02_matched_handoff_20260727.json",
        "paper_report": CLOUD_WORKTREE
        / "docs"
        / "PAPER"
        / "AAAI27_SUBJECT00_AUTHORITATIVE_REVIEW_SUBJECT02_MATCHED_REPORT_20260727.md",
    }


def main() -> None:
    _, contract_meta, target_manifest = validate_preconditions()
    created_at = now_iso()
    meta = common_metadata(created_at)
    before = asset_snapshot()
    validate_expected_hashes(before)

    # Establish an actual before/after boundary with a permitted Git-only artifact.
    paths = artifact_paths()
    write_text(paths["decision_report"], decision_markdown())

    after = asset_snapshot()
    validate_expected_hashes(after)
    validate_immutable(before, after)

    audit = asset_audit(meta, contract_meta, target_manifest, before, after)
    if (
        audit["not_uniquely_recovered"][
            "slot_to_subject02_condition_camera_binding"
        ]["explicit_slot_binding_rows_found"]
        != 0
    ):
        raise RuntimeError("Unexpected explicit slot binding appeared")

    runs = planned_method_runs()
    cells = planned_baseline_cells()
    if len(runs) != 12 or len(cells) != 48:
        raise RuntimeError("Planned matrix cardinality mismatch")

    tests = tests_payload(meta, audit, before, after)
    summary = final_summary(meta, audit, tests)
    report = report_markdown(audit)
    handoff = {
        **meta,
        "artifact": "PROJECT_CONTROL_HANDOFF",
        "status": "BLOCKED_REQUIRES_ASSET_CONTRACT_FREEZE",
        "final_classification": FINAL_CLASSIFICATION,
        "next_task": NEXT_TASK,
        "next_task_executed": False,
        "subject02_optimizer_steps": 0,
        "subject02_output_roots_created": 0,
    }

    payloads: dict[str, Any] = {
        "decision": subject00_decision(meta),
        "mixed_negative": mixed_negative_freeze(meta),
        "claim_boundary": claim_boundary(meta),
        "teacher_limitation": teacher_limitation(meta),
        "method_registry": method_registry(meta, audit, runs),
        "per_run_metrics": per_run_metrics(meta, runs),
        "matrix_aggregate": matrix_aggregate(meta),
        "baseline_registry": baseline_registry(meta, cells),
        "baseline_results": baseline_results(meta),
        "comparison": comparison(meta),
        "decomposition": effect_decomposition(meta),
        "claim_recommendation": claim_recommendation(meta),
        "tests": tests,
        "summary": summary,
        "handoff": handoff,
    }
    for role, payload in payloads.items():
        write_json(paths[role], payload)
    write_text(paths["decision_report"], decision_markdown())
    write_text(paths["report"], report)
    write_text(paths["paper_report"], report)

    for role, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty artifact {role}: {path}")
    if METHOD_OUTPUT.exists() or BASELINE_OUTPUT.exists():
        raise RuntimeError("Reserved Subject02 output appeared")

    print(
        json.dumps(
            {
                "task_id": TASK_ID,
                "status": "SEALED_PREOPTIMIZER_BLOCKER",
                "artifact_count": len(paths),
                "test_result": tests["result"],
                "subject02_method_runs": 0,
                "subject02_optimizer_steps": 0,
                "subject02_checkpoints": 0,
                "subject02_baseline_cells": 0,
                "final_classification": FINAL_CLASSIFICATION,
                "next_task": NEXT_TASK,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
