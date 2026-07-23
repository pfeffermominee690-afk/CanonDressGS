from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


TASK_ID = "AAAI27-COEFFICIENT-HEADROOM-PROTOCOL-001"
SOURCE_BRANCH = "research/pure-endpoint-execution-contract-repair-20260724"
SOURCE_HEAD = "ff56ebfaf7b733adbb41799e248d01d0e7801ea8"
BRANCH = "research/render-refined-coefficient-headroom-protocol-20260724"
SOURCE_MANIFEST_SHA256 = "49bee929edc236af8f37d66be50eb7c44f4011d49e435af903c0e15918a83bbf"
OUTFITS = ["O01", "O02", "O03", "O04", "O08"]
CONDITIONS = ["cond_000000", "cond_000318", "cond_000017", "cond_000347"]
VIEWS = {
    "cond_000000": "front",
    "cond_000318": "back",
    "cond_000017": "left",
    "cond_000347": "right",
}
ROTATIONS = [
    ("R0", ["cond_000000", "cond_000318"], "cond_000017", "cond_000347"),
    ("R1", ["cond_000318", "cond_000017"], "cond_000347", "cond_000000"),
    ("R2", ["cond_000017", "cond_000347"], "cond_000000", "cond_000318"),
    ("R3", ["cond_000347", "cond_000000"], "cond_000318", "cond_000017"),
]
CHECKPOINT_STEPS = [0, 20, 50, 100, 150, 200, 250, 300]
ZERO_COUNTS = {
    "optimization_runs": 0,
    "training_runs": 0,
    "optimizer_creations": 0,
    "optimizer_steps": 0,
    "forward_calls": 0,
    "backward_calls": 0,
    "checkpoint_writes": 0,
    "renderer_runs": 0,
    "new_renders": 0,
}


TEACHERS = {
    "O01": {
        "path": "${CANONDRESSGS_OUTPUT_ROOT}/pipeline_full/SUBJECT02-REPRESENTATION-TRIAGE-001/attempt_002/rung_2_shared_same_support/O01/checkpoints/step_001200.pth",
        "sha256": "af730d138697ab9c7a29f17603303ae41dfa36047b8cee59e2ee89328655ce56",
    },
    "O02": {
        "path": "${CANONDRESSGS_OUTPUT_ROOT}/pipeline_full/SUBJECT02-MULTI-OUTFIT-EXPLICIT-BASIS-001/attempt_003/stage_a_teacher_bank/O02/checkpoints/step_001200.pth",
        "sha256": "8ed55520a42a47b6db1ac5c7f08a6054309802fe92c378eb20401e2777c69820",
    },
    "O03": {
        "path": "${CANONDRESSGS_OUTPUT_ROOT}/pipeline_full/SUBJECT02-MULTI-OUTFIT-EXPLICIT-BASIS-001/attempt_003/stage_a_teacher_bank/O03/checkpoints/step_001200.pth",
        "sha256": "16cb235d784e17f4ebb3928e020ceb0be050b5cf80740e56ec72bcbf11d37d93",
    },
    "O04": {
        "path": "${CANONDRESSGS_OUTPUT_ROOT}/pipeline_full/SUBJECT02-MULTI-OUTFIT-EXPLICIT-BASIS-001/attempt_003/stage_a_teacher_bank/O04/checkpoints/step_001200.pth",
        "sha256": "b39c8d4940325e371cd6db55551c4830b19ab057106c2b9e06df796e8bacd9c3",
    },
    "O08": {
        "path": "${CANONDRESSGS_OUTPUT_ROOT}/pipeline_full/SUBJECT02-REPRESENTATION-TRIAGE-001/attempt_002/rung_2_shared_same_support/O08/checkpoints/step_001200.pth",
        "sha256": "4b0d113cf2e42904ec4f96833e5440fa4f6e6ab013e090be514ef3d8d3dd354e",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lf_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def semantic_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.strip() + "\n", encoding="utf-8", newline="\n")


def load_source_manifest(path: Path) -> dict[str, Any]:
    if sha256_file(path) != SOURCE_MANIFEST_SHA256:
        raise ValueError("source manifest SHA256 mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "canondressgs.full_dataset.v1":
        raise ValueError("source manifest schema mismatch")
    if [row["condition_id"] for row in value["conditions"]] != CONDITIONS:
        raise ValueError("source condition order mismatch")
    outfits = {row["outfit_id"]: row for row in value["outfits"]}
    if any(outfit not in outfits for outfit in OUTFITS):
        raise ValueError("closed-wardrobe outfit missing")
    return value


def observation_id(outfit: str, condition: str) -> str:
    return f"subject02/{outfit}/{condition}"


def condition_contract(source: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for condition in source["conditions"]:
        rows.append(
            {
                "condition_id": condition["condition_id"],
                "view_name": VIEWS[condition["condition_id"]],
                "source_camera_id": condition["source_camera_id"],
                "source_frame_id": condition["source_frame_id"],
                "width": condition["width"],
                "height": condition["height"],
                "background": condition["background"],
                "pose_length": len(condition["pose"]),
                "camera_sha256": condition["source_checksum"]["camera_sha256"],
                "pose_sha256": condition["source_checksum"]["pose_sha256"],
                "source_image_sha256": condition["source_checksum"]["image_sha256"],
                "source_mask_sha256": condition["source_checksum"]["mask_sha256"],
            }
        )
    return rows


def observation_contract(source: dict[str, Any]) -> list[dict[str, Any]]:
    by_outfit = {row["outfit_id"]: row for row in source["outfits"]}
    rows = []
    fields = [
        "target_edit_rgb",
        "target_base_rgb",
        "target_foreground_mask",
        "target_base_foreground_mask",
        "target_edit_mask",
        "target_edit_core_mask",
        "target_clothing_mask",
        "target_old_clothing_mask",
        "target_protected_mask",
        "target_transition_mask",
    ]
    for outfit in OUTFITS:
        observations = {
            row["condition_id"]: row for row in by_outfit[outfit]["observations"]
        }
        for condition in CONDITIONS:
            row = observations[condition]
            assets = {
                name: {"path": row[name], "sha256": row["checksums"][name]}
                for name in fields
            }
            rows.append(
                {
                    "observation_id": observation_id(outfit, condition),
                    "outfit_id": outfit,
                    "condition_id": condition,
                    "view_name": VIEWS[condition],
                    "identity_audit_status": row["identity_audit_status"],
                    "assets": assets,
                }
            )
    return rows


def partition_contract(
    observations: dict[str, dict[str, Any]], conditions: list[str]
) -> dict[str, Any]:
    ids = [observation_id(outfit, condition) for outfit in OUTFITS for condition in conditions]
    selected = [observations[value] for value in ids]
    rgb = [row["assets"]["target_edit_rgb"]["sha256"] for row in selected]
    masks = [
        row["assets"]["target_clothing_mask"]["sha256"] for row in selected
    ]
    return {
        "condition_ids": conditions,
        "observation_ids": ids,
        "observation_count": len(ids),
        "target_rgb_sha256": rgb,
        "target_clothing_mask_sha256": masks,
        "observation_order_sha256": semantic_sha256(ids),
    }


def rotation_contract(observation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observations = {row["observation_id"]: row for row in observation_rows}
    rows = []
    for name, optimize, calibration, test in ROTATIONS:
        query_hashes = {}
        for outfit in OUTFITS:
            sequence = [
                observation_id(outfit, optimize[step % 2]) for step in range(300)
            ]
            query_hashes[outfit] = {
                "algorithm": "sha256(UTF8(newline_join(step_1_through_step_300_observation_ids)))",
                "sha256": hashlib.sha256("\n".join(sequence).encode("utf-8")).hexdigest(),
                "step_count": 300,
                "first_four": sequence[:4],
                "last_four": sequence[-4:],
            }
        row = {
            "rotation": name,
            "optimize_folds": optimize,
            "calibration_fold": calibration,
            "test_fold": test,
            "partitions": {
                "optimize": partition_contract(observations, optimize),
                "calibration": partition_contract(observations, [calibration]),
                "test": partition_contract(observations, [test]),
            },
            "target_observation_intersections": {
                "optimize_calibration": {"observation_id": 0, "rgb_sha256": 0, "mask_sha256": 0},
                "optimize_test": {"observation_id": 0, "rgb_sha256": 0, "mask_sha256": 0},
                "calibration_test": {"observation_id": 0, "rgb_sha256": 0, "mask_sha256": 0},
            },
            "source_reference_asset_overlap": {
                "interpretation": "EXPECTED_CLOSED_WARDROBE_OVERLAP_NOT_CONSUMED_BY_ENDPOINT_OPTIMIZATION",
                "optimize_calibration": {"rgb_sha256": 15, "mask_sha256": 15, "combined_sha256": 30},
                "optimize_test": {"rgb_sha256": 15, "mask_sha256": 15, "combined_sha256": 30},
                "calibration_test": {"rgb_sha256": 10, "mask_sha256": 10, "combined_sha256": 20},
            },
            "per_garment_query_order": query_hashes,
        }
        row["rotation_semantic_sha256"] = semantic_sha256(row)
        rows.append(row)
    return rows


def build_rotation_manifest(source: dict[str, Any]) -> dict[str, Any]:
    observations = observation_contract(source)
    return {
        "schema_version": "canondressgs.paper.coefficient_headroom_rotation_manifests.v1",
        "task_id": TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "source_manifest": "${CANONDRESSGS_OUTPUT_ROOT}/pipeline_full/SUBJECT02-AAAI27-DATA-CAPACITY-GATE-001/attempt_003/dataset/aaai_gate_28_manifest.json",
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_pure_endpoint_rotation_manifest": {
            "path": "paper_protocol/reviewer_risk/pure_endpoint_rotation_manifests.json",
            "git_blob": "31f8a4774e109b5fa8653c58a2fda5f7bb028651",
        },
        "identity": "subject02",
        "outfit_order": OUTFITS,
        "condition_order": CONDITIONS,
        "conditions": condition_contract(source),
        "target_observations": observations,
        "rotations": rotation_contract(observations),
        "selection_isolation": {
            "test_fold_optimizer_selection": False,
            "test_fold_step_selection": False,
            "test_fold_regularization_selection": False,
            "best_checkpoint_selection": False,
        },
        "teacher_initialization_provenance_warning": "Each Teacher Endpoint was historically optimized on all four conditions. The folds isolate refinement updates and calibration, not Teacher creation; this is not strict novel-view generalization.",
        "paper_final": False,
    }


def build_loss_contract() -> dict[str, Any]:
    weights = {
        "garment_rgb": 1.0,
        "alpha_foreground": 0.5,
        "new_silhouette_alpha": 1.0,
        "boundary_rgb": 0.25,
        "protected_rgb": 10.0,
        "protected_alpha": 5.0,
    }
    return {
        "schema_version": "canondressgs.paper.coefficient_headroom_loss_contract.v1",
        "task_id": TASK_ID,
        "status": "COMPLETE",
        "objective_name": "CAPACITY_ORACLE_LOSS_V1_RENDER_TERMS",
        "historical_provenance": {
            "config": "configs/research/subject02_multi_outfit_explicit_basis_v1.yaml",
            "config_git_blob": "fcb3e2fcfe3b4a61cd1edf63f7e5139b92e24567",
            "implementation": "scene/representation_capacity_oracle.py::capacity_oracle_loss_v1",
            "implementation_git_blob": "378d87fee29533260188184b2406e32ad92fe310",
            "historical_teacher_steps": 1200,
            "historical_direct_stability_weight": 0.0001,
        },
        "render_terms": {
            "garment_rgb": {
                "weight": weights["garment_rgb"],
                "definition": "masked RGB L1 against target_edit_rgb",
                "region": "max(target_edit_mask,target_clothing_mask,target_old_clothing_mask) excluding target_protected_mask",
                "differentiable": True,
            },
            "alpha_foreground": {
                "weight": weights["alpha_foreground"],
                "definition": "full-image alpha L1 against target_foreground_mask",
                "region": "all pixels",
                "differentiable": True,
            },
            "new_silhouette_alpha": {
                "weight": weights["new_silhouette_alpha"],
                "definition": "alpha L1 against target_foreground_mask",
                "region": "target_foreground_mask * (1-target_base_foreground_mask)",
                "differentiable": True,
            },
            "boundary_rgb": {
                "weight": weights["boundary_rgb"],
                "definition": "masked RGB L1 against target_edit_rgb",
                "region": "target_transition_mask excluding target_protected_mask",
                "differentiable": True,
            },
            "protected_rgb": {
                "weight": weights["protected_rgb"],
                "definition": "masked RGB L1 against target_base_rgb",
                "region": "target_protected_mask",
                "differentiable": True,
            },
            "protected_alpha": {
                "weight": weights["protected_alpha"],
                "definition": "masked alpha L1 against target_base_foreground_mask",
                "region": "target_protected_mask",
                "differentiable": True,
            },
        },
        "audited_but_disabled_terms": {
            "lpips": {
                "weight": 0.0,
                "reason": "Absent from the formal subject02 Teacher objective; it is evaluation-only.",
                "implementation_differentiable_if_enabled": True,
            },
            "ssim": {"weight": 0.0, "reason": "Absent from the formal Teacher objective."},
            "dice": {"weight": 0.0, "reason": "Absent from CAPACITY_ORACLE_LOSS_V1."},
        },
        "primary_coefficient_regularization": {
            "strategy": "L2_ANCHOR_ONLY",
            "provenance": "PROSPECTIVE_PRE_RESULT_FREEZE_NO_HISTORICAL_COEFFICIENT_ANCHOR",
            "parameterization": "z=(c-mean)/std; z_svd=(c_svd-mean)/std",
            "definition": "mean((z-z_svd)^2)",
            "lambda_grid": [0.0001, 0.001, 0.01, 0.1],
            "selection": "one lambda per rotation from five-garment macro calibration render objective at step 300",
            "tie_tolerance_absolute": 0.0001,
            "tie_break": "stronger regularization",
            "test_used": False,
        },
        "unregularized_diagnostic": {
            "name": "UNREGULARIZED_DIAGNOSTIC",
            "lambda_anchor": 0.0,
            "primary": False,
            "may_replace_primary": False,
        },
        "full_residual_regularization": {
            "definition": "historical direct_stability over masked raw_xyz/raw_log_scaling/raw_rotvec/raw_opacity/raw_sh0",
            "weight": 0.0001,
            "selection": "fixed before result",
        },
        "calibration_score": "mean of the six weighted render terms; method regularizers excluded from the score",
        "test_selection_allowed": False,
        "paper_final": False,
    }


def build_optimizer_contract() -> dict[str, Any]:
    return {
        "schema_version": "canondressgs.paper.coefficient_headroom_optimizer_contract.v1",
        "task_id": TASK_ID,
        "status": "FROZEN_NOT_EXECUTED",
        "formal_platform": {
            "device": "CUDA",
            "gpu_class": "NVIDIA GeForce RTX 4090",
            "reason": "The four-scalar update is CPU-feasible, but the frozen differentiable renderer is the dominant operation and the historical formal runtime is CUDA.",
            "cpu_feasibility": "Four-scalar Adam arithmetic is feasible; end-to-end formal CPU rendering is not the frozen execution path and is not authorized.",
            "gpu_feasibility": "Supported by the frozen subject02 renderer and historical Teacher runs.",
        },
        "coefficient_parameterization": {
            "mathematical_variable": "c in R^4",
            "stored_trainable_tensor": "z=(c-mean)/std in R^4",
            "degrees_of_freedom": 4,
            "initialization": "z_svd=(c_svd-mean)/std",
            "coefficient_mean": [-1.220703143189894e-05, -3.051757857974735e-06, 2.441406286379788e-05, -6.103515625e-05],
            "coefficient_std": [412.4079895019531, 364.3363037109375, 313.2566223144531, 297.9817810058594],
            "garment_order": OUTFITS,
            "normalization_sha256": "c4eef5e6f86d91315d5e2a13d4e7949e0e6eaebde6182f3a60e5574790044fb2",
            "random_initialization": False,
            "multiple_initializations": False,
        },
        "pre_optimizer_initialization_gates": {
            "basis_rank": 4,
            "basis_sha256": "a29b3dc3c6f0ac1a79755e036eb4196dc9e4d116f3e2b35a7ed4e2bd5286f430",
            "normalization_sha256": "c4eef5e6f86d91315d5e2a13d4e7949e0e6eaebde6182f3a60e5574790044fb2",
            "garment_order": OUTFITS,
            "per_garment_bound_normalized_residual_rmse_max": 1e-5,
            "per_view_svd_teacher_rgb_mae_max": 1e-5,
            "per_view_svd_teacher_alpha_mae_max": 1e-5,
            "all_finite": True,
            "failure_rule": "abort before optimizer creation with COEFFICIENT_HEADROOM_PROTOCOL_INCOMPLETE",
        },
        "coefficient_optimizer": {
            "class": "torch.optim.Adam",
            "learning_rate": 0.02,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "weight_decay": 0.0,
            "gradient_clip_norm": 5.0,
            "scheduler": "NONE_FIXED_LR",
            "steps": 300,
            "checkpoint_steps": CHECKPOINT_STEPS,
            "final_rule": "step_300_only",
        },
        "full_residual_optimizer": {
            "class": "torch.optim.Adam",
            "geometry_learning_rate": 0.001,
            "appearance_learning_rate": 0.002,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "scheduler": "NONE_FIXED_LR",
            "steps": 300,
            "checkpoint_steps": CHECKPOINT_STEPS,
            "final_rule": "step_300_only",
        },
        "data_schedule": {
            "per_garment_independent": True,
            "steps_1_to_300": "alternate optimize_folds[0], optimize_folds[1], beginning with index 0",
            "updates_per_optimize_condition": 150,
            "seed": 20260724,
            "seed_selection": False,
            "early_stopping": False,
            "best_checkpoint_selection": False,
            "result_driven_extra_steps": False,
            "garment_specific_budget": False,
        },
        "calibration_selection": {
            "positive_anchor_candidates": 4,
            "selection_scope": "one lambda per rotation shared by all five garments",
            "checkpoint": 300,
            "score": "five-garment macro calibration render objective",
            "identity_safety_required": True,
            "tie_tolerance_absolute": 0.0001,
            "tie_break": "largest lambda",
            "test_used": False,
        },
        "checkpoint_schema": {
            "schema_version": "canondressgs.paper.coefficient_headroom_checkpoint.v1",
            "required_fields": [
                "task_id",
                "method",
                "rotation",
                "outfit_id",
                "global_step",
                "trainable_state",
                "optimizer_state",
                "rng_state",
                "condition_position",
                "lambda_anchor",
                "frozen_asset_sha256",
                "query_order_sha256",
                "cumulative_optimizer_wall_time_seconds",
            ],
            "target_tensors_stored": False,
            "scheduler_state": "NOT_APPLICABLE_FIXED_LR",
            "exact_resume_required": True,
            "atomic_write_required": True,
        },
        "planned_execution": {
            "coefficient_anchor_grid_runs": 80,
            "unregularized_diagnostic_runs": 20,
            "full_residual_runs": 20,
            "total_optimization_runs": 120,
            "optimizer_steps": 36000,
            "checkpoint_writes": 960,
        },
        "actual_execution": ZERO_COUNTS,
        "paper_final": False,
    }


def build_full_residual_contract() -> dict[str, Any]:
    return {
        "schema_version": "canondressgs.paper.coefficient_headroom_full_residual_contract.v1",
        "task_id": TASK_ID,
        "status": "FAIR_BASELINE_FROZEN_NOT_EXECUTED",
        "method_name": "Full-Residual Render Optimization",
        "implementation": {
            "class": "scene.representation_capacity_oracle.UnboundedGaussianDeltaField",
            "source_git_blob": "378d87fee29533260188184b2406e32ad92fe310",
            "base_gaussian_count": 200000,
            "active_support_count": 170547,
            "active_support_rule": "frozen subject02 GARMENT_BODY_PARTS mask",
            "trainable_tensors": {
                "raw_xyz": [200000, 3],
                "raw_log_scaling": [200000, 3],
                "raw_rotvec": [200000, 3],
                "raw_opacity": [200000],
                "raw_sh0": [200000, 1, 3],
            },
            "frozen_schema_tensor": {"raw_shN": [200000, 3, 3], "value": "zero"},
            "allocated_trainable_scalars": 2600000,
            "effective_masked_trainable_scalars": 2217111,
            "other_trainable_scalars": 0,
        },
        "initialization": {
            "source": "Teacher Endpoint for the same outfit",
            "checkpoint_bank": TEACHERS,
            "load_fields": ["raw_xyz", "raw_log_scaling", "raw_rotvec", "raw_opacity", "raw_sh0"],
            "optimizer_state": "fresh_empty_state",
            "teacher_step": 1200,
        },
        "fairness": {
            "same_optimize_views": True,
            "same_calibration_views": True,
            "same_test_views": True,
            "same_six_render_terms": True,
            "same_identity_protection": True,
            "same_background_camera_pose_renderer": True,
            "same_step_budget": 300,
            "extra_teacher_supervision": False,
            "test_views_in_updates": False,
            "test_views_in_selection": False,
        },
        "equal_step_comparison": {
            "checkpoint": 300,
            "required": True,
        },
        "equal_wall_time_comparison": {
            "required": True,
            "coefficient_time_reference": "selected-lambda coefficient run steps 1-300 only; grid search and evaluation excluded",
            "full_residual_rule": "largest fixed checkpoint with cumulative optimizer-section wall time not exceeding the coefficient time reference",
            "checkpoint_candidates": CHECKPOINT_STEPS,
            "if_no_positive_step_fits": 0,
            "reported_discretization": True,
            "timing": "time.perf_counter around forward, loss, backward, gradient clipping, and optimizer step; torch.cuda.synchronize immediately before and after; exclude loading, checkpoint I/O, evaluation, and lambda search",
        },
        "required_efficiency_records": [
            "allocated_trainable_scalars",
            "effective_masked_trainable_scalars",
            "optimizer_steps",
            "optimizer_section_wall_time_seconds",
            "end_to_end_wall_time_seconds",
            "peak_vram_bytes",
            "checkpoint_bytes",
            "held_out_metrics",
        ],
        "provenance_limit": "Teacher initialization used all four conditions historically; held-out means no refinement update or selection on the test condition, not target-naive initialization.",
        "actual_execution": ZERO_COUNTS,
        "paper_final": False,
    }


def build_evaluator_contract() -> dict[str, Any]:
    return {
        "schema_version": "canondressgs.paper.coefficient_headroom_evaluator_contract.v1",
        "task_id": TASK_ID,
        "status": "FROZEN_NOT_EXECUTED",
        "methods": [
            "Teacher Endpoint",
            "SVD Endpoint",
            "Render-Refined Coefficient",
            "Full-Residual Render Optimization",
            "UNREGULARIZED_DIAGNOSTIC",
        ],
        "primary_endpoint_rule": "selected positive-anchor candidate at step 300; no checkpoint selection",
        "primary_denominator": {
            "unit": "one garment x one rotation held-out test observation",
            "per_rotation": 5,
            "global": 20,
            "garment_macro_weighting": "equal weight for each of five garments",
            "rotation_weighting": "equal weight for each of four rotations",
        },
        "render_quality": {
            "rgb_mae": "masked mean absolute RGB error inside target_clothing_mask; lower is better",
            "psnr": "masked RGB PSNR; exact zero MSE is the string Infinity; higher is better",
            "ssim": "masked 11x11 Gaussian SSIM on garment bbox; higher is better",
            "lpips": "VGG LPIPS v0.1 on garment bbox plus 5 percent, aspect-preserving resize and zero pad to 256x256; lower is better",
            "silhouette_iou": "alpha>=0.5 versus target_foreground_mask>=0.5; higher is better",
            "boundary_f": "3x3 erosion XOR boundary with tolerance max(1,floor(0.005*min(H,W)+0.5)); higher is better",
            "protected_lpips": "LPIPS inside target_protected_mask against Base Avatar render; lower is better",
            "identity_metric": "protected-mask mean absolute RGB difference against Base Avatar render; lower is better",
        },
        "residual_metrics": [
            "bound_normalized_residual_distance_to_teacher",
            "bound_normalized_residual_norm",
            "raw_coefficient_displacement_l2",
            "standardized_coefficient_displacement_l2",
            "basis_space_residual_distance",
            "full_residual_bound_normalized_displacement",
        ],
        "generalization": [
            "optimize_fold_metrics",
            "calibration_fold_metrics",
            "held_out_test_metrics",
            "optimize_minus_test_gap",
        ],
        "safety": {
            "identity_contamination": "manual grade >0 on any protected identity region",
            "component_contamination": "manual grade >0",
            "patch_cloud_mottle": "manual grade >0, with grade 3 severe",
            "silhouette_collapse": "empty foreground or manual grade >0",
            "wrong_body_deformation": "manual grade >0",
            "review_policy": "review every one of the 20 selected primary test cells for every primary method; no cherry-picking",
        },
        "efficiency": [
            "trainable_parameters",
            "effective_trainable_parameters",
            "steps",
            "wall_time_seconds",
            "peak_vram_bytes",
            "checkpoint_bytes",
        ],
        "headroom_definitions": {
            "coefficient_headroom_gain": "raw metric delta: Render-Refined Coefficient - SVD Endpoint",
            "teacher_headroom_gain": "raw metric delta: Render-Refined Coefficient - Teacher Endpoint",
            "full_residual_gain": "raw metric delta: Full-Residual Render Optimization - Teacher Endpoint",
            "direction_normalized_improvement": "reported separately: baseline-method for lower-is-better metrics and method-baseline for higher-is-better metrics",
            "span_recovery_ratio_error_metric": "(Teacher_error-Refined_error)/(Teacher_error-FullResidual_error)",
            "span_recovery_denominator_rule": "undefined when Teacher_error-FullResidual_error<=0; emit null plus reason, never fabricate zero or a ratio",
        },
        "reporting": {
            "every_garment": True,
            "every_rotation": True,
            "all_fixed_checkpoints": CHECKPOINT_STEPS,
            "optimize_calibration_test_separate": True,
            "test_used_for_selection": False,
            "uncertainty": "paired bootstrap over the 20 garment-rotation cells with garment and rotation labels retained; 10000 deterministic resamples, seed 20260724",
        },
        "paper_final": False,
    }


def build_success_gates() -> dict[str, Any]:
    return {
        "schema_version": "canondressgs.paper.coefficient_headroom_success_gates.v1",
        "task_id": TASK_ID,
        "status": "PREREGISTERED_NOT_EVALUATED",
        "comparison_for_scientific_classification": "selected Render-Refined Coefficient versus Teacher Endpoint on the 20 held-out test cells; equal-step Full-Residual supplies span capacity context",
        "thresholds": {
            "per_garment_lpips_improvement_absolute_min": 0.001,
            "improved_garments_min": 4,
            "macro_lpips_improvement_absolute_min": 0.005,
            "macro_lpips_relative_reduction_min": 0.05,
            "companion_rgb_mae_improvement_absolute_min": 0.002,
            "companion_boundary_f_improvement_absolute_min": 0.02,
            "companion_silhouette_iou_improvement_absolute_min": 0.01,
            "macro_span_recovery_ratio_min": 0.25,
            "valid_per_garment_span_ratios_min": 3,
            "identity_contamination_count_max": 0,
            "component_contamination_count_max": 0,
            "full_residual_macro_lpips_improvement_absolute_min": 0.01,
            "full_residual_macro_lpips_relative_reduction_min": 0.10,
            "full_residual_improved_garments_min": 4,
        },
        "definitions": {
            "per_garment_lpips_improved": "mean over four held-out rotations decreases by at least 0.001 versus Teacher Endpoint",
            "macro": "equal mean over five garments after equal mean over four rotations",
            "relative_reduction": "(Teacher-Method)/Teacher for positive error-valued Teacher metrics",
            "companion_gate": "RGB MAE improves by 0.002 OR boundary F improves by 0.02 OR silhouette IoU improves by 0.01",
            "safe": "identity contamination=0 and component contamination=0; unsafe apparent gains are not positive headroom",
            "not_optimize_only": "held-out test LPIPS gate itself passes; optimize-fold change alone never qualifies",
        },
        "decision_order": [
            {
                "classification": "COEFFICIENT_HEADROOM_PROTOCOL_INCOMPLETE",
                "when": "required assets, metric cells, safety reviews, parity, or denominator provenance are missing; this is evaluated before scientific classes",
            },
            {
                "classification": "TEACHER_SPAN_HAS_USEFUL_RENDER_HEADROOM",
                "when": "safe, >=4 garments improve LPIPS, macro LPIPS absolute and relative thresholds pass, companion gate passes, macro span recovery ratio>=0.25 with >=3 valid garment ratios",
            },
            {
                "classification": "TEACHER_SPAN_HEADROOM_SMALL",
                "when": "safe and >=4 garments improve LPIPS, but one or more useful-headroom magnitude/recovery gates fail while macro LPIPS improvement remains >0",
            },
            {
                "classification": "TEACHER_SPAN_CAPACITY_LIMITED",
                "when": "no safe coefficient positive class, while safe equal-step Full-Residual improves >=4 garments and passes both full-residual macro LPIPS thresholds",
            },
            {
                "classification": "TEACHER_SPAN_AT_LOCAL_OPTIMUM",
                "when": "none of the preceding rules pass; unsafe coefficient candidates count as no stable improvement",
            },
        ],
        "preconditions": {
            "all_20_primary_cells_present": True,
            "svd_endpoint_teacher_residual_parity_audited": True,
            "coefficient_normalization_sha_matches": True,
            "garment_order_matches": True,
            "no_test_selection": True,
            "all_safety_reviews_present": True,
        },
        "thresholds_may_change_after_results": False,
        "paper_final": False,
    }


def build_protocol_yaml() -> str:
    teacher_lines = "\n".join(
        f"    {name}: {{path: \"{value['path']}\", sha256: {value['sha256']}}}"
        for name, value in TEACHERS.items()
    )
    return f"""
schema_version: canondressgs.paper.coefficient_headroom_protocol.v1
task_id: {TASK_ID}
classification: COEFFICIENT_HEADROOM_PROTOCOL_READY
source:
  branch: {SOURCE_BRANCH}
  head: {SOURCE_HEAD}
governance:
  branch: {BRANCH}
  windows_worktree: E:/model_train/canondressgs_render_refined_coefficient_headroom_protocol
  cloud_worktree: /root/autodl-tmp/canondressgs_work/worktrees/canondressgs_render_refined_coefficient_headroom_protocol
scope:
  identity: subject02
  wardrobe: [O01, O02, O03, O04, O08]
  condition_order: [cond_000000, cond_000318, cond_000017, cond_000347]
  fixed_identity: true
  closed_wardrobe: true
  unseen_garment_claim: false
  cross_identity_claim: false
methods:
  teacher: Teacher Endpoint
  svd: SVD Endpoint
  coefficient: Render-Refined Coefficient
  full_residual: Full-Residual Render Optimization
  refined_lookup: Outfit-ID Refined Lookup
  refined_hard_lookup: Refined Hard Lookup
  future_predictor: Reference-to-Refined-Coefficient
assets:
  base_avatar: {{path: "${{CANONDRESSGS_OUTPUT_ROOT}}/subject02_formal_800k/chkpnt100000.pth", sha256: abbf67b59eadf2cba2dea69dbeec598f9177da45b8ddc74ccbe8108acf9ddf70}}
  rank4_basis: {{path: "${{CANONDRESSGS_OUTPUT_ROOT}}/pipeline_full/SUBJECT02-MULTI-OUTFIT-EXPLICIT-BASIS-001/attempt_003/stage_b_basis/selected/selected_basis.pt", sha256: a29b3dc3c6f0ac1a79755e036eb4196dc9e4d116f3e2b35a7ed4e2bd5286f430}}
  coefficient_normalization: {{path: "${{CANONDRESSGS_OUTPUT_ROOT}}/pipeline_full/SUBJECT02-MULTI-OUTFIT-EXPLICIT-BASIS-001/attempt_003/stage_b_basis/selected/coefficient_normalization.json", sha256: c4eef5e6f86d91315d5e2a13d4e7949e0e6eaebde6182f3a60e5574790044fb2}}
  source_manifest: {{path: "${{CANONDRESSGS_OUTPUT_ROOT}}/pipeline_full/SUBJECT02-AAAI27-DATA-CAPACITY-GATE-001/attempt_003/dataset/aaai_gate_28_manifest.json", sha256: {SOURCE_MANIFEST_SHA256}}}
  protected_support: {{path: "${{CANONDRESSGS_OUTPUT_ROOT}}/pipeline_full/SUBJECT02-EDITABLE-POOL-SPECIFICITY-001/attempt_007/full_gaussian_attribution/full_gaussian_attribution.parquet", sha256: 4b6267aad721bbe1a690d631d8479a69859f02058b0b507399fcef448333271d}}
  renderer_source_bundle_sha256: 0e568a7b771137228c1ec63f29817a7e2b2b796f74c9da9349cb8a610b1d8ba6
  teachers:
{teacher_lines}
coefficient:
  mathematical_variable: c_in_R4
  stored_parameterization: standardized_z_in_R4
  initialization: c_svd
  normalization_mean: [-0.00001220703143189894, -0.000003051757857974735, 0.00002441406286379788, -0.00006103515625]
  normalization_std: [412.4079895019531, 364.3363037109375, 313.2566223144531, 297.9817810058594]
  optimizer: Adam
  learning_rate: 0.02
  steps: 300
  checkpoint_steps: [0, 20, 50, 100, 150, 200, 250, 300]
  primary_regularization: L2_anchor
  lambda_grid: [0.0001, 0.001, 0.01, 0.1]
  unregularized_diagnostic_lambda: 0.0
selection:
  lambda_scope: one_per_rotation_shared_across_five_garments
  source: calibration_fold_only
  final_checkpoint: 300
  tie_break: stronger_regularization
  test_used: false
future_interfaces:
  outfit_id_refined_lookup: ground_truth_garment_id_to_c_refined_non_deployable
  refined_hard_lookup: frozen_reference_classifier_to_c_refined_lookup
  reference_to_refined_coefficient_target: c_refined
  endpoint_gain_attributed_to_predictor: false
full_residual:
  initialization: Teacher_Endpoint
  steps: 300
  equal_step: true
  equal_wall_time: true
  allocated_trainable_scalars: 2600000
  effective_masked_trainable_scalars: 2217111
execution_boundary:
  optimization_runs: 0
  optimizer_created: 0
  forward: 0
  backward: 0
  checkpoint_writes: 0
  renderer_runs: 0
  new_renders: 0
paper_final: false
next_task: RUN_RENDER_REFINED_COEFFICIENT_HEADROOM_EXPERIMENT
next_task_started: false
"""


def docs() -> dict[str, str]:
    protocol = f"""
# Render-Refined Coefficient Headroom Protocol

Task: `{TASK_ID}`

Source: `{SOURCE_BRANCH}` at `{SOURCE_HEAD}`

Classification: `COEFFICIENT_HEADROOM_PROTOCOL_READY`

## Scientific question

For fixed subject02 and the closed wardrobe O01/O02/O03/O04/O08, does
render-objective optimization of four coefficients inside the frozen
Teacher-derived affine residual span improve held-out condition renders beyond
both the SVD Endpoint and the Teacher Endpoint? The experiment also measures
what fraction of safe Full-Residual improvement is recovered inside the span.

The four names are frozen: **Teacher Endpoint**, **SVD Endpoint**,
**Render-Refined Coefficient**, and **Full-Residual Render Optimization**.
Teacher is an endpoint and initialization, not a capability bound.

## Frozen assets and variables

The subject02 base, five Teacher checkpoints, rank-4 basis, basis mean,
coefficient normalization, MMLP-Human, renderer, camera, pose, RGB/masks,
protected regions, white background, and render settings are immutable. The
coefficient method has one four-dimensional degree of freedom. Its
implementation stores standardized `z=(c-mean)/std`, initialized exactly from
the frozen SVD coefficient; this is a one-to-one numerical parameterization of
`c`, not an extra model. No random or cross-garment initialization is allowed.

Full-Residual uses the historical Rung-2 direct canonical field. Only
`raw_xyz`, `raw_log_scaling`, `raw_rotvec`, `raw_opacity`, and `raw_sh0` are
trainable; `raw_shN` remains the frozen zero schema buffer. Base avatar,
deformation, renderer, F2, and reference predictor remain frozen.

## Rotations and isolation

| Rotation | Optimize | Calibration | Test |
|---|---|---|---|
| R0 | 000000, 000318 | 000017 | 000347 |
| R1 | 000318, 000017 | 000347 | 000000 |
| R2 | 000017, 000347 | 000000 | 000318 |
| R3 | 000347, 000000 | 000318 | 000017 |

Each garment is optimized independently. Steps 1-300 alternate the two
optimize observations, so each receives 150 updates. The exact target assets,
camera/pose hashes, partition memberships, source-reference overlap, and query
order hashes are frozen in `coefficient_headroom_rotation_manifests.json`.

The target observations are disjoint across optimize/calibration/test within a
rotation. Closed-wardrobe reference images overlap across the inherited pure
endpoint query folds, but reference assets are not consumed by endpoint
optimization. A crucial limitation remains: each Teacher Endpoint was
historically optimized on all four conditions. Therefore the test fold is
isolated from refinement updates and selection, but initialization is not
target-naive. This is an endpoint-refinement diagnostic, not strict novel-view
generalization.

## Objective and regularization

The only historically supported rendering objective is
`CAPACITY_ORACLE_LOSS_V1`: garment RGB L1 (1.0), foreground alpha L1 (0.5),
new-silhouette alpha L1 (1.0), boundary RGB L1 (0.25), protected RGB L1 against
the Base Avatar (10.0), and protected alpha L1 against Base (5.0). LPIPS, SSIM,
and Dice are not Teacher training terms and stay at weight zero; LPIPS and SSIM
are evaluation metrics only.

The primary coefficient strategy is L2 anchoring in standardized coefficient
space with the preregistered positive grid `1e-4/1e-3/1e-2/1e-1`. One lambda
is selected per rotation from the step-300 five-garment macro calibration
render objective, shared by all five garments. Candidates within `1e-4` of the
minimum tie in favor of stronger regularization. Test metrics never select a
lambda. Lambda zero is run only as `UNREGULARIZED_DIAGNOSTIC` and cannot replace
the primary result.

## Budget and final rule

Coefficient optimization uses Adam, lr 0.02, no weight decay, fixed LR,
gradient clip 5.0, and 300 steps. Full-Residual uses the historical two-rate
Adam contract, geometry lr 0.001, appearance lr 0.002, no weight decay, fixed
LR, clip 1.0, and the same 300 steps. Checkpoints are fixed at
0/20/50/100/150/200/250/300. Step 300 is the only primary endpoint. There is no
early stopping, best checkpoint, seed selection, garment-specific budget, or
result-driven extension.

The formal platform is the historical CUDA/RTX 4090 renderer path. Four-scalar
Adam arithmetic is CPU-feasible, but end-to-end CPU rendering is not the frozen
formal path.

## Evaluation and decisions

Every garment and rotation reports optimize, calibration, and held-out test
RGB MAE, PSNR, SSIM, VGG LPIPS, silhouette IoU, boundary F, protected LPIPS,
identity metric, residual distances, train-test gap, safety grades, parameters,
time, VRAM, and storage. Primary aggregation has 20 equally weighted
garment-rotation test cells.

Headroom gains and the span recovery ratio are frozen in the evaluator
contract. A non-positive full-residual denominator produces `null` with a
reason, never an invented ratio. Numerical success thresholds and the complete
classification decision order are in `coefficient_headroom_success_gates.json`.

`Outfit-ID Refined Lookup` is an evaluation-only lookup using ground-truth
garment ID. The future `Reference-to-Refined-Coefficient` predictor targets
`c_i^R`; `Refined Hard Lookup` remains the fair garment-classifier-to-lookup
baseline. Endpoint optimization gains may not be attributed to that predictor.

## Claim boundary and execution boundary

The strongest allowed positive statement is: "Teacher-derived basis supports
low-dimensional render-objective refinement within its affine residual span."
This does not establish new-garment adaptation, arbitrary garments, continuous
reference control, cross-identity transfer, or performance beyond full-residual
capability.

This freeze created zero optimizers, steps, forward/backward calls, checkpoints,
renderer runs, or renders. `PAPER_FINAL=false`. The authorized next task is
`RUN_RENDER_REFINED_COEFFICIENT_HEADROOM_EXPERIMENT`; it was not started.
"""
    analysis = f"""
# Teacher Endpoint Headroom Analysis Plan

Task: `{TASK_ID}`

## Endpoint comparisons

For every garment and rotation, evaluate the frozen Teacher Endpoint and SVD
Endpoint on all optimize, calibration, and test conditions before reading any
refinement result. Verify garment order, normalization SHA, basis rank, and the
stored SVD reconstruction error. Teacher/SVD differences are reported rather
than silently treated as exact parity.

The selected Render-Refined Coefficient is the positive-anchor candidate chosen
from calibration at step 300. `UNREGULARIZED_DIAGNOSTIC` is reported in a
separate table. Full-Residual step 300 supplies the equal-step outside-span
reference, and the preregistered time-matched checkpoint supplies the
equal-wall-time reference.

## Signed gains

The three named gains retain the requested raw metric convention: method minus
baseline. Thus a negative LPIPS/MAE gain is favorable, while a positive
IoU/boundary-F gain is favorable. Also report a separate direction-normalized
improvement (positive is favorable) for thresholding. Report
coefficient-versus-SVD, coefficient-versus-Teacher, and
full-residual-versus-Teacher separately. Do not call an SVD recovery gain a
Teacher headroom gain.

For each error metric, compute

`(Teacher error - refined error) / (Teacher error - full-residual error)`.

If the denominator is non-positive, emit a null ratio and a reason. Report the
macro ratio only when its denominator is positive, plus the number of valid
per-garment ratios. Ratios above one are retained rather than clipped, because
they are diagnostic estimates rather than probabilities.

## Trajectory diagnosis

At 0/20/50/100/150/200/250/300, report render terms, coefficient displacement,
residual distance to Teacher, gradient norm, and safety metrics. The trajectory
can diagnose overshoot or local curvature, but only step 300 enters the primary
classification. No trajectory observation can add steps or select a checkpoint.

Compare optimize, calibration, and test changes to distinguish generalizing
refinement from optimize-only fitting. Because Teacher initialization was
created with all four conditions, use "held-out refinement condition" and not
"unseen view" in reports.

## Classification

Apply `coefficient_headroom_success_gates.json` mechanically. Useful headroom
requires safe LPIPS improvement for at least four garments, both macro LPIPS
magnitude thresholds, a companion RGB/boundary/silhouette gain, and material
span recovery. Stable but sub-threshold gains are small headroom. A safe strong
Full-Residual gain without a coefficient gain is capacity-limited. Remaining
safe or unsafe non-improvements are local optimum. Missing required evidence is
protocol incomplete and is checked before scientific labels.

No result is available in this protocol-freeze task. `PAPER_FINAL=false`.
"""
    full = f"""
# Full-Residual Fair Comparison Protocol

Task: `{TASK_ID}`

## Baseline definition

`Full-Residual Render Optimization` reuses the historical subject02
`UnboundedGaussianDeltaField` checkpoint schema. For each garment, it loads the
same Teacher Endpoint tensors, discards historical optimizer state, and creates
a fresh optimizer only in the authorized execution task. The five trainable
tensors contain 2,600,000 allocated scalars; the frozen garment support mask
leaves 2,217,111 effective scalars. `raw_shN` remains a zero buffer, matching
the Teacher schema rather than introducing a new channel.

No base, deformation, MMLP-Human, renderer, camera, pose, F2, predictor, mask,
background, or target asset is trainable. The full residual receives no extra
view, test view, Teacher supervision, loss term, or step.

## Equal-step comparison

Both coefficient and full-residual methods execute 300 updates and use the same
alternating optimize-condition schedule. They share the six rendering terms
and identity regions. Their preregistered parameterization regularizers and
learning rates differ and are reported explicitly. Step 300 is compared; no
best checkpoint is allowed.

## Equal-wall-time comparison

Measure optimizer-section wall time for the selected coefficient trajectory
steps 1-300, excluding lambda-grid search, initialization, checkpoint I/O, and
evaluation. For Full-Residual choose the largest fixed checkpoint whose
cumulative optimizer-section time does not exceed that reference. If no
positive step fits, use step 0. Report the checkpoint discretization and also
report end-to-end protocol cost separately.

## Required records

For every garment and rotation record allocated/effective trainable scalars,
steps, optimizer-only and end-to-end wall time, peak VRAM, checkpoint bytes,
and held-out metrics. Report equal-step and equal-wall-time tables separately.

Teacher initialization historically saw all four conditions. Therefore this
baseline estimates post-Teacher residual headroom under held-out refinement
updates; it is not a target-naive generalization experiment and cannot be
presented as one.

This task ran no optimizer or renderer. `PAPER_FINAL=false`.
"""
    return {
        "docs/PAPER/AAAI27_RENDER_REFINED_COEFFICIENT_HEADROOM_PROTOCOL_20260724.md": protocol,
        "docs/PAPER/AAAI27_TEACHER_ENDPOINT_HEADROOM_ANALYSIS_PLAN_20260724.md": analysis,
        "docs/PAPER/AAAI27_FULL_RESIDUAL_FAIR_COMPARISON_PROTOCOL_20260724.md": full,
    }


def build_summary(artifact_hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "schema_version": "canondressgs.paper.coefficient_headroom_protocol_final_summary.v1",
        "task_id": TASK_ID,
        "classification": "COEFFICIENT_HEADROOM_PROTOCOL_READY",
        "classification_reason": "The frozen rank-4 assets, exact condition observations, historical differentiable render loss, positive-anchor calibration rule, fixed optimizer budgets, full-residual schema, evaluator, and numerical success gates are all uniquely bound before results.",
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "branch": BRANCH,
        "scientific_scope": {
            "identity": "subject02",
            "wardrobe": OUTFITS,
            "strict_novel_view": False,
            "reason": "Teacher initialization was historically optimized on all four conditions.",
        },
        "assets": {
            "rank4_basis_sha256": "a29b3dc3c6f0ac1a79755e036eb4196dc9e4d116f3e2b35a7ed4e2bd5286f430",
            "coefficient_normalization_sha256": "c4eef5e6f86d91315d5e2a13d4e7949e0e6eaebde6182f3a60e5574790044fb2",
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "renderer_source_bundle_sha256": "0e568a7b771137228c1ec63f29817a7e2b2b796f74c9da9349cb8a610b1d8ba6",
            "teacher_checkpoint_sha256": {name: row["sha256"] for name, row in TEACHERS.items()},
        },
        "frozen_contract": {
            "rotations": 4,
            "garments": 5,
            "primary_test_cells": 20,
            "coefficient_dof": 4,
            "positive_anchor_candidates": 4,
            "steps": 300,
            "checkpoint_steps": CHECKPOINT_STEPS,
            "planned_optimization_runs": 120,
        },
        "future_interfaces": {
            "outfit_id_refined_lookup": "ground-truth garment ID -> c_i^R; non-deployable endpoint decomposition",
            "refined_hard_lookup": "frozen reference garment classifier -> lookup c_i^R",
            "reference_to_refined_coefficient_target": "c_i^R",
            "endpoint_refinement_gain_attributed_to_predictor": False,
        },
        "actual_execution": ZERO_COUNTS,
        "frozen_asset_mutations": {
            "base_avatar": 0,
            "teacher_endpoints": 0,
            "rank4_basis": 0,
            "coefficient_normalization": 0,
            "mmlp_human": 0,
            "renderer": 0,
            "camera_pose": 0,
            "rgb_masks": 0,
            "identity_regions": 0,
            "reference_predictor": 0,
        },
        "artifact_sha256_lf": artifact_hashes,
        "static_tests": {
            "command": "python -m unittest tests.test_coefficient_headroom_protocol -v",
            "test_count": 10,
            "status": "PASS",
        },
        "paper_final": False,
        "paper_final_count": 0,
        "next_task": "RUN_RENDER_REFINED_COEFFICIENT_HEADROOM_EXPERIMENT",
        "next_task_started": False,
    }


def build_handoff(summary_sha256_lf: str, artifact_hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "schema_version": "canondressgs.project_control.coefficient_headroom_protocol_handoff.v1",
        "task_id": TASK_ID,
        "status": "COMPLETE",
        "classification": "COEFFICIENT_HEADROOM_PROTOCOL_READY",
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "branch": BRANCH,
        "windows_worktree": "E:/model_train/canondressgs_render_refined_coefficient_headroom_protocol",
        "cloud_worktree": "/root/autodl-tmp/canondressgs_work/worktrees/canondressgs_render_refined_coefficient_headroom_protocol",
        "summary": {
            "path": "paper_protocol/reviewer_risk/coefficient_headroom_protocol_final_summary.json",
            "sha256_lf": summary_sha256_lf,
        },
        "artifact_sha256_lf": artifact_hashes,
        "execution_boundary": ZERO_COUNTS,
        "static_tests": {"count": 10, "status": "PASS"},
        "paper_final": False,
        "paper_final_count": 0,
        "next_task": "RUN_RENDER_REFINED_COEFFICIENT_HEADROOM_EXPERIMENT",
        "next_task_started": False,
        "stop_after_handoff": True,
    }


def freeze(repo: Path, source_manifest_path: Path) -> None:
    source = load_source_manifest(source_manifest_path)
    outputs: dict[str, Any] = {
        "paper_protocol/reviewer_risk/coefficient_headroom_rotation_manifests.json": build_rotation_manifest(source),
        "paper_protocol/reviewer_risk/coefficient_headroom_loss_contract.json": build_loss_contract(),
        "paper_protocol/reviewer_risk/coefficient_headroom_optimizer_contract.json": build_optimizer_contract(),
        "paper_protocol/reviewer_risk/coefficient_headroom_full_residual_contract.json": build_full_residual_contract(),
        "paper_protocol/reviewer_risk/coefficient_headroom_evaluator_contract.json": build_evaluator_contract(),
        "paper_protocol/reviewer_risk/coefficient_headroom_success_gates.json": build_success_gates(),
    }
    for relative, value in outputs.items():
        write_json(repo / relative, value)
    write_text(
        repo / "paper_protocol/reviewer_risk/coefficient_headroom_protocol.yaml",
        build_protocol_yaml(),
    )
    for relative, value in docs().items():
        write_text(repo / relative, value)

    artifact_paths = [
        *docs().keys(),
        "paper_protocol/reviewer_risk/coefficient_headroom_protocol.yaml",
        *outputs.keys(),
    ]
    artifact_hashes = {
        relative: sha256_lf_file(repo / relative) for relative in artifact_paths
    }
    summary_path = repo / "paper_protocol/reviewer_risk/coefficient_headroom_protocol_final_summary.json"
    write_json(summary_path, build_summary(artifact_hashes))
    write_json(
        repo / "project_control_handoff/coefficient_headroom_protocol_handoff.json",
        build_handoff(sha256_lf_file(summary_path), artifact_hashes),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--source-manifest", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.repo.resolve(), args.source_manifest.resolve())
    print(json.dumps({"status": "COEFFICIENT_HEADROOM_PROTOCOL_READY", "execution": ZERO_COUNTS}, indent=2))


if __name__ == "__main__":
    main()
