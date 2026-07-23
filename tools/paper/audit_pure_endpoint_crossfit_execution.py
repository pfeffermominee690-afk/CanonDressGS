from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
RISK = ROOT / "paper_protocol/reviewer_risk"
DOCS = ROOT / "docs/PAPER"
HANDOFF = ROOT / "project_control_handoff"

TASK_ID = "AAAI27-PURE-ENDPOINT-CORE-METHOD-CROSSFIT-001"
SOURCE_BRANCH = "research/pure-endpoint-core-method-protocol-20260724"
SOURCE_HEAD = "9ca0f44bdd5480a429e8cd5705df1341746d4381"
RUN_BRANCH = "research/pure-endpoint-core-method-crossfit-20260724"
CLASSIFICATION = "PURE_ENDPOINT_PROTOCOL_HASH_MISMATCH"
SECONDARY_BLOCKER = "PURE_ENDPOINT_BASELINE_COMPARABILITY_INCOMPLETE"
NEXT_TASK = "NONE_FROZEN_FOR_PURE_ENDPOINT_PROTOCOL_HASH_MISMATCH"

OUTFITS = ("O01", "O02", "O03", "O04", "O08")
SEEDS = (0, 1, 2)
MILESTONES = (0, 20, 50, 100, 200, 300)
SNAPSHOT_TIME = "2026-07-24T04:57:02.531614642+08:00"

SOURCE_PROTOCOL_FILES = (
    "docs/PAPER/AAAI27_PURE_ENDPOINT_CORE_METHOD_PROTOCOL_20260724.md",
    "docs/PAPER/AAAI27_PURE_ENDPOINT_BASELINE_REGISTRY_20260724.md",
    "docs/PAPER/AAAI27_PURE_ENDPOINT_CLAIM_BOUNDARY_20260724.md",
    "paper_protocol/reviewer_risk/pure_endpoint_crossfit_protocol.yaml",
    "paper_protocol/reviewer_risk/pure_endpoint_rotation_manifests.json",
    "paper_protocol/reviewer_risk/pure_endpoint_model_contract.json",
    "paper_protocol/reviewer_risk/pure_endpoint_baseline_registry.json",
    "paper_protocol/reviewer_risk/pure_endpoint_evaluator_contract.json",
    "paper_protocol/reviewer_risk/pure_endpoint_success_gates.json",
    "paper_protocol/reviewer_risk/pure_endpoint_protocol_final_summary.json",
    "project_control_handoff/pure_endpoint_protocol_handoff.json",
)

SNAPSHOT_OUTPUTS = {
    "execution_contract": RISK / "pure_endpoint_execution_contract.json",
    "training_schedules": RISK / "pure_endpoint_training_schedules.json",
    "asset_snapshot": RISK / "pure_endpoint_asset_snapshot.json",
}

RESULT_OUTPUTS = {
    "training": RISK / "pure_endpoint_training_results.json",
    "baselines": RISK / "pure_endpoint_baseline_results.json",
    "predictions": RISK / "pure_endpoint_test_predictions.json",
    "coefficients": RISK / "pure_endpoint_coefficient_results.json",
    "renders": RISK / "pure_endpoint_render_results.json",
    "parity": RISK / "pure_endpoint_parity_results.json",
    "perturbations": RISK / "pure_endpoint_perturbation_results.json",
    "visual_review": RISK / "pure_endpoint_visual_review.json",
    "efficiency": RISK / "pure_endpoint_efficiency_results.json",
    "summary": RISK / "pure_endpoint_final_summary.json",
    "handoff": HANDOFF / "pure_endpoint_crossfit_handoff.json",
}

DOC_OUTPUTS = {
    "results": DOCS / "AAAI27_PURE_ENDPOINT_CROSSFOLD_RESULTS_20260724.md",
    "baselines": DOCS / "AAAI27_PURE_ENDPOINT_BASELINE_COMPARISON_20260724.md",
    "coefficient": DOCS / "AAAI27_PURE_ENDPOINT_COEFFICIENT_AND_PARITY_20260724.md",
    "perturbation": DOCS / "AAAI27_PURE_ENDPOINT_PERTURBATION_AND_SAFETY_20260724.md",
    "visual": DOCS / "AAAI27_PURE_ENDPOINT_VISUAL_REVIEW_20260724.md",
    "efficiency": DOCS / "AAAI27_PURE_ENDPOINT_EFFICIENCY_20260724.md",
}


def strict_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=strict_object
    )


def sha256(path: Path, *, normalize_lf: bool = False) -> str:
    data = path.read_bytes()
    if normalize_lf:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def canonical_sha(value: Any) -> str:
    data = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8", newline="\n")


def protocol_hash_audit() -> dict[str, Any]:
    summary = read_json(RISK / "pure_endpoint_protocol_final_summary.json")
    declared = summary.get("protocol_artifact_sha256_lf")
    records = []
    for relative in SOURCE_PROTOCOL_FILES:
        actual = sha256(ROOT / relative, normalize_lf=True)
        expected = declared.get(relative) if isinstance(declared, Mapping) else None
        records.append(
            {
                "path": relative,
                "actual_sha256_lf": actual,
                "declared_sha256_lf": expected if expected is not None else "MISSING",
                "pass": expected == actual,
            }
        )
    return {
        "status": CLASSIFICATION,
        "required_summary_field": "protocol_artifact_sha256_lf",
        "summary_field_present": isinstance(declared, Mapping),
        "checked_count": len(records),
        "passed_count": sum(row["pass"] for row in records),
        "missing_declaration_count": sum(
            row["declared_sha256_lf"] == "MISSING" for row in records
        ),
        "records": records,
    }


def baseline_audit() -> dict[str, Any]:
    registry = read_json(RISK / "pure_endpoint_baseline_registry.json")
    direct = next(
        row for row in registry["methods"]
        if row["paper_name"] == "Direct Residual Decoder"
    )
    required = (
        "architecture",
        "parameter_count",
        "trainable_modules",
        "frozen_modules",
        "training_steps",
        "seed",
        "initialization",
        "optimizer",
        "loss",
        "calibration_rule",
        "checkpoint_rule",
        "renderer",
        "test_denominator",
        "target_information_boundary",
    )
    missing = [name for name in required if name not in direct]
    config_path = ROOT / "configs/research/subject02_residual_decoder_capacity_v7.yaml"
    config_text = config_path.read_text(encoding="utf-8")
    evidence = {
        "historical_config": str(config_path.relative_to(ROOT)).replace("\\", "/"),
        "historical_config_sha256_lf": sha256(config_path, normalize_lf=True),
        "historical_source_sha256_lf": sha256(
            ROOT / direct["historical_source"], normalize_lf=True
        ),
        "registry_source_sha256_lf": direct["historical_source_sha256_lf"],
        "historical_outfits_are_only_O01_O08": "outfits: [O01, O08]" in config_text,
        "historical_stage_steps_are_1000": config_text.count("max_steps: 1000") == 2,
        "historical_milestones_are_not_crossfit_milestones": (
            config_text.count("milestones: [0, 100, 300, 600, 1000]") == 2
        ),
        "historical_stage_a_uses_diagnostic_latent": (
            "diagnostic_latent_only: true" in config_text
        ),
        "historical_stage_b_has_multiple_learning_rates": all(
            value in config_text
            for value in (
                "encoder_lr: 0.0001",
                "aggregator_lr: 0.0001",
                "completer_lr: 0.001",
                "residual_decoder_v7_lr: 0.001",
            )
        ),
        "historical_attempt_status": "V7-D",
        "historical_reference_conditioned_stage_b_executed": False,
    }
    return {
        "status": SECONDARY_BLOCKER,
        "method": "Direct Residual Decoder",
        "required_field_count": len(required),
        "missing_field_count": len(missing),
        "missing_fields": missing,
        "ambiguity": [
            "The registry does not choose V7 stage A versus stage B as the matched model.",
            "The historical contract covers two outfits and 1000 steps, while this task requires five outfits and 300 steps.",
            "The registry does not freeze whether V7 multi-rate optimization or the primary Adam lr=0.02 contract applies.",
            "The historical reference-conditioned stage B has no executed checkpoint or validated trajectory.",
        ],
        "evidence": evidence,
    }


def source_rows() -> dict[str, Mapping[str, Any]]:
    source = read_json(RISK / "dual_support_controller_training_manifest.json")
    return {row["record_id"]: row for row in source["query_sets"]}


def build_schedules() -> dict[str, Any]:
    payload = read_json(RISK / "pure_endpoint_rotation_manifests.json")
    by_id = source_rows()
    rotations = []
    for rotation in payload["rotations"]:
        train_ids = set(rotation["partitions"]["train"]["record_ids"])
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for record_id in sorted(train_ids):
            groups[by_id[record_id]["logical_input_sha256"]].append(by_id[record_id])
        unique_queries = []
        by_fold_garment: dict[tuple[str, str], dict[str, Any]] = {}
        for logical_id, rows in sorted(groups.items()):
            representative = min(rows, key=lambda row: row["record_id"])
            garment = representative["garment_labels"][0]
            fold = representative["target_view_fold"]
            query = {
                "logical_query_id": logical_id,
                "representative_record_id": representative["record_id"],
                "protocol_duplicate_record_ids": sorted(row["record_id"] for row in rows),
                "garment": garment,
                "fold": fold,
                "reference_paths": [
                    {
                        "image_path": row["image_path"],
                        "image_sha256": row["image_sha256"],
                        "clothing_mask_path": row["clothing_mask_path"],
                        "clothing_mask_sha256": row["clothing_mask_sha256"],
                    }
                    for row in representative["source_references"]
                ],
                "target_endpoint_id": garment,
                "training_weight": 1,
            }
            unique_queries.append(query)
            by_fold_garment[(fold, garment)] = query
        cycle = []
        for batch_index, fold in enumerate(rotation["train_folds"]):
            queries = [by_fold_garment[(fold, garment)] for garment in OUTFITS]
            cycle.append(
                {
                    "batch_index": batch_index,
                    "fold": fold,
                    "garment_order": list(OUTFITS),
                    "logical_query_ids": [row["logical_query_id"] for row in queries],
                    "representative_record_ids": [
                        row["representative_record_id"] for row in queries
                    ],
                }
            )
        steps = []
        for step in range(1, 301):
            batch = cycle[(step - 1) % 2]
            steps.append(
                {
                    "step": step,
                    "cycle_index": (step - 1) // 2,
                    "batch_index": batch["batch_index"],
                    "fold": batch["fold"],
                    "logical_query_ids": batch["logical_query_ids"],
                    "representative_record_ids": batch["representative_record_ids"],
                }
            )
        if len(unique_queries) != 10 or any(len(rows) != 4 for rows in groups.values()):
            raise ValueError("pure endpoint unique-query grouping mismatch")
        exposure = {row["logical_query_id"]: 0 for row in unique_queries}
        for step in steps:
            for logical_id in step["logical_query_ids"]:
                exposure[logical_id] += 1
        if set(exposure.values()) != {150}:
            raise ValueError("pure endpoint exposure distribution mismatch")
        rotations.append(
            {
                "rotation": rotation["rotation"],
                "train_folds": rotation["train_folds"],
                "calibration_fold": rotation["calibration_fold"],
                "test_fold": rotation["test_fold"],
                "protocol_counts": {"train": 40, "calibration": 20, "test": 20},
                "unique_counts": {"train": 10, "calibration": 5, "test": 5},
                "duplicate_excess": {"train": 30, "calibration": 15, "test": 15},
                "unique_train_queries": unique_queries,
                "cycle": cycle,
                "cycle_sha256": canonical_sha(cycle),
                "steps": steps,
                "schedule_sha256": canonical_sha(steps),
                "exposure_distribution": exposure,
                "seed_schedule_sha256": {
                    str(seed): canonical_sha(steps) for seed in SEEDS
                },
            }
        )
    return {
        "schema_version": "canondressgs.paper.pure_endpoint_training_schedules.v1",
        "task_id": TASK_ID,
        "status": "FROZEN_NOT_AUTHORIZED_BY_PROTOCOL_HASH_GATE",
        "source_head": SOURCE_HEAD,
        "run_branch": RUN_BRANCH,
        "seeds": list(SEEDS),
        "garment_order": list(OUTFITS),
        "batch_size": 5,
        "batches_per_cycle": 2,
        "cycles": 150,
        "training_steps": 300,
        "unique_query_exposure_per_run": 150,
        "duplicate_training_weight": 0,
        "rotations": rotations,
        "optimizer_creation_authorized": False,
        "blocker": CLASSIFICATION,
    }


def execution_counts() -> dict[str, int]:
    return {
        "training_runs": 0,
        "optimizer_creations": 0,
        "optimizer_steps": 0,
        "forward_training_batches": 0,
        "backward_calls": 0,
        "checkpoint_writes": 0,
        "feature_forwards": 0,
        "test_inferences": 0,
        "formal_pure_inferences": 0,
        "perturbation_inferences": 0,
        "logical_renders": 0,
        "unique_physical_renders": 0,
        "visual_sheets": 0,
    }


def expected_training_counts() -> dict[str, int]:
    return {
        "seed_count": 3,
        "rotation_count": 4,
        "main_method_runs": 12,
        "learned_baseline_runs": 24,
        "shared_linear_and_endpoint_training_runs": 12,
        "total_unique_training_runs": 36,
        "optimizer_creations": 36,
        "optimizer_steps": 10800,
        "forward_training_batches": 10800,
        "backward_calls": 10800,
        "checkpoint_writes": 216,
    }


def asset_snapshot(hash_audit: Mapping[str, Any]) -> dict[str, Any]:
    manifest = read_json(ROOT / "paper_protocol/frozen_asset_manifest.json")
    selected_actual = {
        row["asset_id"]: row["fingerprint"] for row in manifest["assets"]
        if row["asset_id"] in {
            "teacher_checkpoint_O01",
            "teacher_checkpoint_O02",
            "teacher_checkpoint_O03",
            "teacher_checkpoint_O04",
            "teacher_checkpoint_O08",
            "rank4_explicit_basis",
            "coefficient_normalization",
            "mmlphuman_checkpoint",
            "backbone_checkpoint",
            "backbone_state",
            "target_reference_manifest",
            "renderer_source_bundle",
        }
    }
    return {
        "schema_version": "canondressgs.paper.pure_endpoint_asset_snapshot.v1",
        "task_id": TASK_ID,
        "status": CLASSIFICATION,
        "source_head": SOURCE_HEAD,
        "protocol_hash_audit": hash_audit,
        "official_frozen_asset_verification": {
            "command": (
                "/root/miniconda3/bin/python tools/paper/"
                "verify_seen_outfit_paper_assets.py --output-root "
                "/root/autodl-tmp/canondressgs_work/outputs"
            ),
            "timestamp": "2026-07-24T05:08:00+08:00",
            "status": "PASS",
            "verified_external": True,
            "asset_count": 19,
            "failed_assets": [],
            "selected_actual_sha256_or_fingerprint": selected_actual,
        },
        "frozen_feature_cache": {
            "path": (
                "/root/autodl-tmp/canondressgs_work/outputs/AAAI27-SEEN-"
                "OUTFIT-PAPER/shared_preflight/frozen_reference_feature_rows_v1.pt"
            ),
            "sha256": "30cf19a99bbc620112928d678a0257bfabd3420be66fe1053ded3ce7e90875cb",
            "episode_count": 24,
            "normal_f2_shape_per_episode": [3, 256],
            "target_forward_leakage": False,
            "outfit_id_in_model": False,
        },
        "mutations": {
            "source_protocol": 0,
            "controller_historical_attempts": 0,
            "teacher_endpoints": 0,
            "rank4_basis": 0,
            "frozen_f2": 0,
            "mmlp_human": 0,
            "renderer": 0,
            "reference_assets": 0,
            "target_endpoint_assets": 0,
            "identity_masks": 0,
            "subject00": 0,
            "avatarrex": 0,
        },
    }


def execution_contract(
    hash_audit: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": "canondressgs.paper.pure_endpoint_execution_contract.v1",
        "task_id": TASK_ID,
        "status": "BLOCKED_BEFORE_OPTIMIZER",
        "classification": CLASSIFICATION,
        "secondary_blockers": [SECONDARY_BLOCKER],
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "run_branch": RUN_BRANCH,
        "paper_method": "CanonDressGS-Endpoint",
        "scientific_protocol": "PURE-ENDPOINT CONDITION-FOLD CROSS-FIT",
        "resource_gate": {
            "timestamp": SNAPSHOT_TIME,
            "status": "PASS",
            "host": "autodl-container-ef19489c10-464381bb",
            "gpu": "NVIDIA GeForce RTX 4090",
            "driver": "580.76.05",
            "gpu_memory_used_mib": 0,
            "gpu_memory_free_mib": 24081,
            "gpu_utilization_percent": 0,
            "gpu_compute_process_count": 0,
            "active_formal_gpu_task_count": 0,
            "active_large_io_task_count": 0,
            "free_bytes": 39750483968,
            "minimum_free_bytes": 25 * 1024**3,
            "free_inodes": 674977634,
        },
        "api_and_credentials": {
            "active_codex_session_continuity": True,
            "provider": "SJWen Proxy",
            "model_provider_id": "sjwen_proxy",
            "wire_api": "responses",
            "authentication_mode": "environment-key provider authentication",
            "requires_openai_auth": False,
            "credential_scan": "PASS_NO_REAL_SECRET_MATCH",
            "credential_values_persisted": False,
            "cli_status_note": (
                "codex login status was unavailable because local service_tier=default "
                "is rejected by the installed CLI; config was not modified"
            ),
        },
        "protocol_hash_verification": hash_audit,
        "baseline_comparability": baseline,
        "seed_contract": {
            "seeds": list(SEEDS),
            "fresh_process_required": True,
            "fresh_initialization_required": True,
            "independent_pid_required": True,
            "best_seed_selection": False,
        },
        "model_contract": {
            "architecture": "LayerNorm(512) -> Linear(512,4)",
            "trainable_parameters": 3076,
            "rank": 4,
            "basis_explained_variance": 0.9999999999978457,
            "coefficient_target": "standardized frozen teacher endpoint coefficient",
            "loss": "mean SmoothL1 beta=1 over four standardized coefficients",
            "optimizer": "Adam lr=0.02 weight_decay=0",
            "steps": 300,
            "milestones": list(MILESTONES),
        },
        "planned_training_counts_if_unblocked": expected_training_counts(),
        "actual_execution_counts": execution_counts(),
        "optimizer_creation_authorized": False,
        "formal_output_attempt_created": False,
        "paper_final": False,
        "paper_final_count": 0,
        "next_task": NEXT_TASK,
        "next_task_started": False,
    }


def blocked_payload(kind: str, **extra: Any) -> dict[str, Any]:
    payload = {
        "schema_version": f"canondressgs.paper.pure_endpoint_{kind}.v1",
        "task_id": TASK_ID,
        "status": "NOT_RUN_PROTOCOL_HASH_MISMATCH",
        "classification": CLASSIFICATION,
        "source_head": SOURCE_HEAD,
        "run_branch": RUN_BRANCH,
        "actual_execution_counts": execution_counts(),
        "paper_final": False,
        "paper_final_count": 0,
    }
    payload.update(extra)
    return payload


def seal_blocker(
    hash_audit: Mapping[str, Any], baseline: Mapping[str, Any]
) -> None:
    methods = [
        "Base Avatar",
        "Teacher Upper Bound",
        "Outfit-ID Oracle",
        "Reference Classifier Lookup",
        "Nearest-Centroid Lookup",
        "Direct Residual Decoder",
        "Linear Coefficient Predictor",
        "CanonDressGS-Endpoint",
    ]
    write_json(
        RESULT_OUTPUTS["training"],
        blocked_payload(
            "training_results",
            runs=[],
            planned_counts_if_unblocked=expected_training_counts(),
            numerical_checks_executed=0,
        ),
    )
    write_json(
        RESULT_OUTPUTS["baselines"],
        blocked_payload(
            "baseline_results",
            methods=[
                {
                    "paper_name": name,
                    "status": "NOT_RUN_PROTOCOL_HASH_MISMATCH",
                    "training_runs": 0,
                    "test_inferences": 0,
                    "formal_renders": 0,
                }
                for name in methods
            ],
            comparability_audit=baseline,
        ),
    )
    write_json(
        RESULT_OUTPUTS["predictions"],
        blocked_payload(
            "test_predictions",
            predictions=[],
            required_protocol_weighted_count=80,
            required_unique_query_count=20,
            evaluated_protocol_weighted_count=0,
            evaluated_unique_query_count=0,
        ),
    )
    write_json(
        RESULT_OUTPUTS["coefficients"],
        blocked_payload(
            "coefficient_results",
            coefficient_mae_status="NOT_EVALUATED",
            coefficient_rmse_status="NOT_EVALUATED",
            basis_reconstruction_status="NOT_EVALUATED",
            records=[],
        ),
    )
    write_json(
        RESULT_OUTPUTS["renders"],
        blocked_payload(
            "render_results",
            logical_render_count=0,
            unique_physical_render_count=0,
            reused_count=0,
            new_count=0,
            failed_count=0,
            records=[],
        ),
    )
    write_json(
        RESULT_OUTPUTS["parity"],
        blocked_payload(
            "parity_results",
            gate="NOT_EVALUATED",
            exact_parity_count=0,
            tolerance_parity_count=0,
            parity_failure_count=0,
            evaluated_count=0,
            records=[],
        ),
    )
    write_json(
        RESULT_OUTPUTS["perturbations"],
        blocked_payload(
            "perturbation_results",
            variants=[
                "mild_blur",
                "mask_erosion",
                "mask_dilation",
                "assignment_permutation",
                "single_reference",
                "complete_reference_dropout",
            ],
            evaluated_inferences=0,
            records=[],
        ),
    )
    write_json(
        RESULT_OUTPUTS["visual_review"],
        blocked_payload(
            "visual_review",
            required_main_sheet_count=60,
            opened_main_sheet_count=0,
            required_formal_pure_sheet_count_status="UNRESOLVED_EXECUTION_CONTRACT",
            opened_formal_pure_sheet_count=0,
            original_detail_opened_count=0,
            records=[],
        ),
    )
    write_json(
        RESULT_OUTPUTS["efficiency"],
        blocked_payload(
            "efficiency_results",
            canon_dress_gs_endpoint_trainable_parameters=3076,
            parameter_count_reverified_during_execution=False,
            training_wall_time_seconds=0.0,
            render_time_seconds=0.0,
            peak_vram_bytes=0,
            checkpoint_bytes=0,
            external_output_archive_bytes=0,
        ),
    )
    final_summary = blocked_payload(
        "final_summary",
        status="BLOCKED_BEFORE_OPTIMIZER",
        classification=CLASSIFICATION,
        classification_reason=(
            "The source final summary does not declare the required protocol artifact "
            "hashes, so exact pre-optimizer agreement cannot be established."
        ),
        secondary_blockers=[SECONDARY_BLOCKER],
        protocol_hash_audit=hash_audit,
        baseline_comparability=baseline,
        primary_gates={
            "macro_5way_top1": "NOT_EVALUATED",
            "every_rotation_top1": "NOT_EVALUATED",
            "every_garment_recall": "NOT_EVALUATED",
            "endpoint_parity": "NOT_EVALUATED",
            "identity_contamination": "NOT_EVALUATED",
            "severe_wrong_outfit_rate": "NOT_EVALUATED",
        },
        next_task=NEXT_TASK,
        next_task_started=False,
    )
    write_json(RESULT_OUTPUTS["summary"], final_summary)
    write_json(
        RESULT_OUTPUTS["handoff"],
        {
            "schema_version": "canondressgs.project_control.pure_endpoint_crossfit_handoff.v1",
            "task_id": TASK_ID,
            "status": "BLOCKED_BEFORE_OPTIMIZER",
            "classification": CLASSIFICATION,
            "secondary_blockers": [SECONDARY_BLOCKER],
            "source_head": SOURCE_HEAD,
            "run_branch": RUN_BRANCH,
            "execution_contract": str(
                SNAPSHOT_OUTPUTS["execution_contract"].relative_to(ROOT)
            ).replace("\\", "/"),
            "final_summary": str(RESULT_OUTPUTS["summary"].relative_to(ROOT)).replace("\\", "/"),
            "actual_execution_counts": execution_counts(),
            "frozen_mutation_count": 0,
            "paper_final": False,
            "paper_final_count": 0,
            "next_task": NEXT_TASK,
            "next_task_started": False,
            "stop_rule": "Repair and re-freeze protocol hashes and baseline execution contract before any optimizer is created.",
        },
    )

    common = f"""Task: {TASK_ID}

Status: BLOCKED_BEFORE_OPTIMIZER

Classification: {CLASSIFICATION}

The source protocol final summary does not declare the artifact hashes required
by the execution task. Exact agreement therefore cannot be established. The
official frozen-asset verifier passed all 19 external assets, so this is a
protocol sealing failure rather than external asset corruption.

Direct Residual Decoder is also not execution-complete: its registry omits 14
required fields, while historical V7 covers only O01/O08 at 1000 steps and has
no executed reference-conditioned Stage B trajectory. No baseline definition
was invented to bridge that ambiguity.

Training, optimizer creation, forward batches, backward calls, checkpoint
writes, inference, rendering, perturbation evaluation, and visual review are
all zero. PAPER_FINAL=false.
"""
    titles = {
        "results": "AAAI-27 Pure Endpoint Cross-Fit Results",
        "baselines": "AAAI-27 Pure Endpoint Baseline Comparison",
        "coefficient": "AAAI-27 Pure Endpoint Coefficient and Parity",
        "perturbation": "AAAI-27 Pure Endpoint Perturbation and Safety",
        "visual": "AAAI-27 Pure Endpoint Visual Review",
        "efficiency": "AAAI-27 Pure Endpoint Efficiency",
    }
    suffixes = {
        "results": "No scientific performance result was produced.",
        "baselines": "All eight baselines are NOT_RUN; Direct Residual Decoder comparability is incomplete.",
        "coefficient": "Coefficient, reconstruction, margin, tie, and parity metrics are NOT_EVALUATED.",
        "perturbation": "All nuisance, ablation, safety, and formal-pure evaluations are NOT_EVALUATED.",
        "visual": "Required sheets opened: 0. No visual grade was assigned.",
        "efficiency": "Trainable parameter execution recheck, timing, VRAM, checkpoint, and archive measurements were not run.",
    }
    for key, path in DOC_OUTPUTS.items():
        write_text(path, f"# {titles[key]}\n\n{common}\n{suffixes[key]}")


def generate(phase: str) -> dict[str, Any]:
    hash_audit = protocol_hash_audit()
    baseline = baseline_audit()
    schedules = build_schedules()
    write_json(SNAPSHOT_OUTPUTS["training_schedules"], schedules)
    write_json(SNAPSHOT_OUTPUTS["asset_snapshot"], asset_snapshot(hash_audit))
    write_json(
        SNAPSHOT_OUTPUTS["execution_contract"],
        execution_contract(hash_audit, baseline),
    )
    if phase == "seal-blocker":
        seal_blocker(hash_audit, baseline)
    return {
        "status": "PASS",
        "phase": phase,
        "classification": CLASSIFICATION,
        "snapshot_outputs": len(SNAPSHOT_OUTPUTS),
        "sealed_outputs": len(RESULT_OUTPUTS) + len(DOC_OUTPUTS)
        if phase == "seal-blocker" else 0,
        "optimizer_created": 0,
        "training_runs": 0,
        "formal_renderer_runs": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the pure endpoint cross-fit execution gate"
    )
    parser.add_argument(
        "--phase", choices=("snapshot", "seal-blocker"), required=True
    )
    args = parser.parse_args()
    print(json.dumps(generate(args.phase), sort_keys=True))


if __name__ == "__main__":
    main()
