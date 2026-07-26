#!/usr/bin/env python3
"""Seal the blocked O03 equal-step Full Avatar micro-pilot preflight."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


TASK_ID = "AAAI27-FULL-AVATAR-FINETUNING-O03-EQUAL-STEP-MICRO-PILOT-001"
SOURCE_BRANCH = "research/external-baseline-feasibility-from-bundle-rerun-20260726"
SOURCE_HEAD = "572637f08aef21fde35dcd7f8879ff74549c2afc"
NEW_BRANCH = "research/full-avatar-finetuning-o03-equal-step-micropilot-20260726"
BUNDLE_SHA256 = "f12e5bb8ce2c36e12bdb911fdae41cd7e3e2ca0e9d2520dc2c57d2263ffd9e61"
FINAL_CLASSIFICATION = "FULL_AVATAR_O03_EQUALSTEP_MICROPILOT_PREFLIGHT_BLOCKED"
NEXT_TASK = "USER_RESOLVE_FULL_AVATAR_MICROPILOT_PREFLIGHT_BLOCKER"

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_protocol" / "external_baselines" / "full_avatar_o03_equal_step_micropilot"
BUNDLE = ROOT / "paper_protocol" / "external_baselines" / "source_bundle" / "canondressgs_external_baseline_multi_source_evidence_bundle.json"
FROZEN_ASSETS = ROOT / "paper_protocol" / "frozen_asset_manifest.json"
ROTATIONS = ROOT / "paper_protocol" / "reviewer_risk" / "pure_endpoint_rotation_manifests.json"
SUMMARY_PATH = OUT / "full_avatar_o03_equal_step_micropilot_final_summary.json"
PREFLIGHT_PATH = OUT / "full_avatar_o03_equal_step_micropilot_preflight.json"
REPORT_PATH = ROOT / "docs" / "PAPER" / "AAAI27_FULL_AVATAR_O03_EQUALSTEP_MICROPILOT_PREFLIGHT_20260726.md"
HANDOFF_PATH = ROOT / "project_control_handoff" / "full_avatar_o03_equal_step_micropilot_handoff.json"

BASE_PATH = "/root/autodl-tmp/canondressgs_work/outputs/subject02_formal_800k/chkpnt100000.pth"
BASE_SHA256 = "abbf67b59eadf2cba2dea69dbeec598f9177da45b8ddc74ccbe8108acf9ddf70"
BASE_CONFIG_PATH = "/root/autodl-tmp/canondressgs_work/outputs/subject02_formal_800k/config.yaml"
BASE_CONFIG_SHA256 = "538dc4feb5c66ad85274ac6b0cb11a2819d6ba67e06ee8bdccbd1c01c2c7aec6"
TEACHER_PATH = "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-MULTI-OUTFIT-EXPLICIT-BASIS-001/attempt_003/stage_a_teacher_bank/O03/checkpoints/step_001200.pth"
TEACHER_SHA256 = "16cb235d784e17f4ebb3928e020ceb0be050b5cf80740e56ec72bcbf11d37d93"
TARGET_ROOT = "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-AAAI27-DATA-CAPACITY-GATE-001/attempt_003/dataset"
TARGET_MANIFEST = f"{TARGET_ROOT}/aaai_gate_28_manifest.json"
TARGET_MANIFEST_SHA256 = "49bee929edc236af8f37d66be50eb7c44f4011d49e435af903c0e15918a83bbf"
OUTPUT_ROOT = "/root/autodl-tmp/canondressgs_work/outputs/FULL-AVATAR-FINETUNE-O03-EQUALSTEP-MICROPILOT-001"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def validate_local_sources() -> tuple[dict, dict, dict]:
    if sha256(BUNDLE) != BUNDLE_SHA256:
        raise RuntimeError("frozen bundle SHA256 mismatch")
    bundle = json.loads(BUNDLE.read_text(encoding="utf-8"))
    if not (bundle["all_classifications_match"] and bundle["all_numeric_checks_pass"] and bundle["all_claim_boundaries_pass"]):
        raise RuntimeError("frozen bundle gate failed")
    assets = json.loads(FROZEN_ASSETS.read_text(encoding="utf-8"))
    rotations = json.loads(ROTATIONS.read_text(encoding="utf-8"))
    return bundle, assets, rotations


def main() -> None:
    bundle, assets, rotations = validate_local_sources()
    selected_assets = {row["asset_id"]: row for row in assets["assets"]}
    base_asset = selected_assets["mmlphuman_checkpoint"]
    teacher_asset = selected_assets["teacher_checkpoint_O03"]
    target_asset = selected_assets["target_reference_manifest"]
    if base_asset["fingerprint"] != BASE_SHA256 or teacher_asset["fingerprint"] != TEACHER_SHA256:
        raise RuntimeError("base or O03 Teacher registry mismatch")
    if target_asset["fingerprint"] != TARGET_MANIFEST_SHA256:
        raise RuntimeError("target manifest registry mismatch")

    rotation_rows = [{
        "rotation": row["rotation"],
        "train": row["train_folds"],
        "calibration": row["calibration_fold"],
        "test": row["test_fold"],
        "logical_disjointness_pass": row["logical_disjointness_pass"],
        "manifest_sha256": row["rotation_manifest_sha256"],
    } for row in rotations["rotations"]]

    target_registry = {
        "condition_order": ["cond_000000", "cond_000318", "cond_000017", "cond_000347"],
        "rgb_sha256": {
            "cond_000000": "b11e0feb45b747cffe74613b4fb7d6cf9e290e08f2251317ffa49ee64c512117",
            "cond_000318": "2884331627d856c774de8368aad9af043dac5d4b0957accd85a8afab5f7b954a",
            "cond_000017": "32c70a455396ea33848d49b73520f7c7995dfa04dfde20b8158b2ff7b6d65bb9",
            "cond_000347": "0fa201528e834029cdcaa887d8ec76b855c8dc6680bd5ad79ba9228fa559b7d6",
        },
        "foreground_mask_sha256": {
            "cond_000000": "82f7dd71f6ba9fb37e07c02649f3900a3f6f6e08ee08d3d94925548de4458f10",
            "cond_000318": "8b21f3060feb2ff306b04bb048fed323b8c23fcf577976e7c3cef024b6065340",
            "cond_000017": "56b2f4ef9178d0ac1e9e9d65da862f44253a81cce924f7b764f2fc38d4fbe185",
            "cond_000347": "7957e74d8e494ac3819d7f35c07fb27dc85f6c05fbe89765f6a2e6edda561a6b",
        },
        "garment_mask_sha256": {
            "cond_000000": "a25b9f0e7586fbfe3ce5e6c6035cd5df2fb5cae540a0a46c27be4027d0a222c6",
            "cond_000318": "fccc2d7523792ad2adf66772e3068815026e9cb13f877e83a1a2a0235ef05897",
            "cond_000017": "bdfdc73a9f66657cabcd5fab0ddc88a82fddb703b64a6468184323cf185794b9",
            "cond_000347": "4123167a1bf3aff44d130e3d03a4d49cc606321f31891b4939e77e92be5189fd",
        },
        "camera_sha256": {
            "cond_000000": "21607b527da70cd1fe1ce5ff8448806188745c916f59d4ff9e5e0e74ff0e84c0",
            "cond_000318": "6a7af5a1502b7bcb34f52dd190cd42a21833bdc8274a84c8dbf8e84275616278",
            "cond_000017": "8f11a271d6bd72edd813579516d07b86c4f9d2e60e25502877e537c80e91bfed",
            "cond_000347": "f3cbe09fe8650081aa1e4c6ab2bb02d955dd5901457625cf0cc5958806705305",
        },
        "pose_sha256": {
            "cond_000000": "36489b2e1ff768d9c0f7ebab540e811c80f3b6b49f2c5e2a8404c0afdb8a4222",
            "cond_000318": "538812c7b333cc0134a81ba95abdd6f81eb234a4fcabc0489f831b188a063447",
            "cond_000017": "55566abdf8025c657e6d10ef338e81f809d892ba8fa963d5689525647e1eabd5",
            "cond_000347": "7c5d8844474c8902b7b87f55ea5125710f9b593dcc2d100d80db1bf46d4b054f",
        },
    }

    candidate_groups = {
        "dxyz": 30000, "scales": 600000, "quats": 800000, "opacities": 200000,
        "sh0": 600000, "shN": 1800000, "dxyz_bs": 450000,
        "dscales_bs": 9000000, "dquats_bs": 12000000,
        "dopacities_bs": 3000000, "dsh0_bs": 9000000,
        "dshN_bs": 27000000, "encoder_feat_params": 91017000,
        "xyz_offset": 600000,
    }
    blockers = [
        {
            "id": "SPLIT_ROTATION_NOT_SELECTED",
            "evidence": "Four frozen, logically disjoint rotations exist; neither the feasibility contract nor the authorized task selects one for this single run.",
            "required_resolution": "Freeze one rotation id, or explicitly authorize all-view view-transductive adaptation and revise the no-target-test-RGB clause.",
        },
        {
            "id": "LOSS_CONTRACT_NOT_SELECTED",
            "evidence": "Base provenance defines full-image L1/LPIPS/smoothness/scaling losses, while Teacher provenance defines CAPACITY_ORACLE_LOSS_V1 with region masks. The Full Avatar contract does not select either objective or define a mapping.",
            "required_resolution": "Freeze one exact loss expression, weights, mask roles, crop, and reduction denominator.",
        },
        {
            "id": "OPTIMIZER_INITIALIZATION_NOT_SELECTED",
            "evidence": "The Base checkpoint contains 14 optimizer states and 9 scheduler states, but the Full Avatar contract does not say whether to restore them or create fresh optimizers, nor how Teacher geometry/appearance rates map to backbone groups.",
            "required_resolution": "Freeze restore-versus-reset, optimizer classes, every group learning rate, scheduler state, and gradient clipping policy.",
        },
        {
            "id": "SEED_NOT_UNIQUE",
            "evidence": "Base config freezes seed 0 and Teacher construction freezes seed 20260718; the Full Avatar contract requests preregistered seeds but contains no Full Avatar seed value.",
            "required_resolution": "Freeze one exact seed for this one-run micro-pilot.",
        },
    ]

    checkpoint_file_bytes = 1_706_610_302
    milestone_bytes = checkpoint_file_bytes * 5
    auxiliary_estimate = 2 * 1024**3
    free_before = 46_450_372_608
    safety_minimum = 25 * 1024**3
    estimated_remaining = free_before - milestone_bytes - auxiliary_estimate

    preflight = {
        "schema_version": "canondressgs.full_avatar_o03_equal_step.preflight.v1",
        "task_id": TASK_ID,
        "source": {"branch": SOURCE_BRANCH, "head": SOURCE_HEAD, "bundle_sha256": BUNDLE_SHA256, "source_tests": "45/45 PASS"},
        "execution": {
            "new_branch": NEW_BRANCH,
            "windows_worktree": r"E:\model_train\canondressgs_full_avatar_finetuning_o03_equal_step_micropilot",
            "cloud_worktree": "/root/autodl-tmp/canondressgs_work/worktrees/canondressgs_full_avatar_finetuning_o03_equal_step_micropilot",
            "output_root": OUTPUT_ROOT,
            "output_root_created": False,
            "attempt_id": "attempt_001_PLANNED_NOT_CREATED",
        },
        "base_avatar": {
            "path": BASE_PATH, "sha256": BASE_SHA256, "step": 100000,
            "config_path": BASE_CONFIG_PATH, "config_sha256": BASE_CONFIG_SHA256,
            "subject": "THuman4.0 subject02", "canonical_gaussian_count": 200000,
            "control_anchor_count": 10000, "pose_feature_count": 300,
            "architecture": "MMLP-Human GaussianModel checkpoint v2 with SMPL-X/LBS deformation and gsplat renderer",
            "binding": "checkpoint embeds Gaussian state, LBS/deformation state, optimizer/scheduler state, and RNG state",
            "sealed_status": "PASS_FROZEN_ASSET_REGISTRY_AND_LIVE_SHA_MATCH",
            "visual_status": "REAL_VERIFIED_BY_FORMAL_BACKBONE_ACCEPTANCE",
        },
        "target_contract": {
            "garment": "O03", "root": TARGET_ROOT, "manifest": TARGET_MANIFEST,
            "manifest_sha256": TARGET_MANIFEST_SHA256, "rgb_count": 4,
            "primary_mask_count": 8, "all_unique_mask_file_count": 44,
            "camera_count": 4, "registry": target_registry,
            "identity_audit": {"cond_000000": "PASS", "cond_000318": "PASS", "cond_000017": "PASS", "cond_000347": "WARN"},
            "candidate_rotations": rotation_rows, "selected_rotation": None,
            "overlap_audit": "ALL_ROTATIONS_LOGICALLY_DISJOINT; EXPECTED_CLOSED_WARDROBE_REFERENCE_ASSET_OVERLAP",
            "target_leakage_status": "NOT_EVALUABLE_UNTIL_ONE_SPLIT_OR_VIEW_TRANSDUCTIVE_CONTRACT_IS_SELECTED",
            "teacher_used_conditions": ["cond_000000", "cond_000318", "cond_000017", "cond_000347"],
        },
        "equal_step": {
            "teacher_checkpoint": TEACHER_PATH, "teacher_checkpoint_sha256": TEACHER_SHA256,
            "n_teacher_per_garment": 1200, "full_avatar_finetune_steps": 1200,
            "status": "PASS_UNIQUE_BUDGET_FROM_SEALED_O03_TEACHER_METADATA",
        },
        "parameter_provenance": {
            "candidate_base_optimizer_groups": candidate_groups,
            "candidate_trainable_parameter_count": sum(candidate_groups.values()),
            "candidate_trainable_tensor_count": 23,
            "candidate_frozen_checkpoint_tensor_values": 14_166_176,
            "approved_trainable_groups": None,
            "approved_trainable_parameter_count": None,
            "approved_frozen_groups": None,
            "approved_frozen_parameter_count": None,
            "reason": "Optimizer initialization and loss contract are not frozen for Full Avatar fine-tuning.",
        },
        "checkpoint_provenance": {
            "base_checkpoint_file_bytes": checkpoint_file_bytes,
            "model_and_buffer_tensor_bytes": 686_237_756,
            "optimizer_state_tensor_bytes": 1_018_376_084,
            "serialization_and_non_tensor_bytes": 1_996_462,
            "required_milestones": [0, 300, 600, 900, 1200],
            "estimated_total_milestone_bytes": milestone_bytes,
        },
        "storage": {
            "free_bytes_before_output_creation": free_before,
            "minimum_remaining_bytes": safety_minimum,
            "estimated_auxiliary_bytes": auxiliary_estimate,
            "estimated_remaining_bytes": estimated_remaining,
            "status": "PASS",
        },
        "gpu": {
            "model": "NVIDIA GeForce RTX 4090", "count": 1,
            "free_vram_bytes": 24_837_292_032, "driver": "580.76.05",
            "pytorch": "2.4.1+cu121", "cuda_runtime": "12.1",
            "renderer": "gsplat 1.5.3+pt24cu121 IMPORT_PASS",
            "status": "PASS_SINGLE_GPU",
        },
        "blockers": blockers,
        "operations": {
            "optimizer_steps_completed": 0, "training_runs": 0, "backward_calls": 0,
            "checkpoint_count": 0, "render_inferences": 0, "generated_images": 0,
            "data_mutations": 0, "paper_modifications": 0,
        },
        "preflight_status": "BLOCKED_BEFORE_OUTPUT_CREATION_AND_OPTIMIZER_STEP",
        "paper_final": False,
        "final_classification": FINAL_CLASSIFICATION,
        "next_task": NEXT_TASK,
        "frozen_bundle_export_count": bundle["export_count"],
    }
    write_json(PREFLIGHT_PATH, preflight)

    unavailable = "NOT_RUN_PREFLIGHT_BLOCKED"
    summary = {
        "schema_version": "canondressgs.full_avatar_o03_equal_step.final_summary.v1",
        "task_id": TASK_ID, "source_branch": SOURCE_BRANCH, "source_head": SOURCE_HEAD,
        "new_branch": NEW_BRANCH, "windows_worktree": preflight["execution"]["windows_worktree"],
        "cloud_worktree": preflight["execution"]["cloud_worktree"], "output_root": OUTPUT_ROOT,
        "attempt_id": preflight["execution"]["attempt_id"], "subject": "THuman4.0 subject02", "garment": "O03",
        "base_checkpoint_path": BASE_PATH, "base_checkpoint_sha256": BASE_SHA256, "base_checkpoint_step": 100000,
        "target_data_root": TARGET_ROOT, "target_rgb_count": 4, "target_mask_count": 8, "camera_count": 4,
        "train_calibration_test_split": "UNRESOLVED_FOUR_FROZEN_ROTATIONS_NO_SELECTION",
        "target_leakage_status": preflight["target_contract"]["target_leakage_status"],
        "n_teacher_per_garment": 1200, "full_avatar_finetune_steps": 1200,
        "trainable_groups": "UNRESOLVED_NOT_AUTHORIZED", "trainable_parameter_count": None,
        "frozen_groups": "UNRESOLVED_NOT_AUTHORIZED", "frozen_parameter_count": None,
        "storage_preflight": "PASS", "cloud_free_bytes_before": free_before,
        "gpu": "NVIDIA GeForce RTX 4090 x1", "cuda": "12.1", "pytorch": "2.4.1+cu121",
        "optimizer": "UNRESOLVED_RESTORE_VS_RESET_AND_GROUP_MAPPING", "learning_rate": "UNRESOLVED",
        "scheduler": "UNRESOLVED_RESTORE_VS_RESET", "seed": "UNRESOLVED_0_VS_20260718",
        "optimizer_steps_completed": 0, "checkpoint_count": 0, "checkpoint_paths": [], "checkpoint_sha256": [],
        "resume_exactness_status": unavailable, "gradient_audit_status": unavailable,
        "frozen_mutation_status": "NO_MUTATION_NO_TRAINING", "loss_initial": None, "loss_final": None,
        "wall_time": unavailable, "peak_vram": unavailable,
        "full_image_lpips": None, "full_image_psnr": None, "full_image_ssim": None,
        "garment_region_lpips": None, "garment_region_psnr": None, "garment_region_ssim": None,
        "silhouette_iou": None, "boundary_f": None, "protected_region_lpips": None,
        "protected_region_rgb_mae": None, "identity_contamination": unavailable,
        "severe_artifact_count": unavailable, "animation_compatibility_status": unavailable,
        "different_pose_status": unavailable, "different_camera_status": unavailable,
        "final_checkpoint_bytes": 0, "optimizer_state_bytes": 0, "total_milestone_bytes": 0,
        "projected_five_garment_storage": unavailable, "projected_five_garment_wall_time": unavailable,
        "canondressgs_onboarding_cost": "FROZEN_EVIDENCE_RETAINED; NOT_RECOMPUTED_IN_BLOCKED_PREFLIGHT",
        "contact_sheet_path": unavailable, "metrics_path": unavailable,
        "paper_modifications": 0, "paper_final": False,
        "final_classification": FINAL_CLASSIFICATION, "next_task": NEXT_TASK,
        "preflight_path": str(PREFLIGHT_PATH.relative_to(ROOT)).replace("\\", "/"),
        "blocker_ids": [row["id"] for row in blockers],
    }
    write_json(SUMMARY_PATH, summary)

    report = f"""# Full Avatar O03 Equal-Step Micro-Pilot Preflight

Task: `{TASK_ID}`

## Decision

`{FINAL_CLASSIFICATION}`

The source branch, source HEAD, 45 feasibility tests, frozen bundle hash, formal Base Avatar, O03 Teacher checkpoint, O03 target manifest, cloud storage, single-GPU environment, and renderer import all passed. The equal-step budget is uniquely `1200` optimizer steps.

Training did not start because the Full Avatar execution contract is under-specified in four scientifically consequential places. Four frozen condition rotations exist and none is selected for this one-run pilot. The Base and Teacher provenance define different loss contracts, but no Full Avatar loss is selected. The Base checkpoint contains optimizer and scheduler state, but restore-versus-reset and group learning-rate mapping are not selected. Finally, seed `0` and seed `20260718` are both provenance-backed, but neither is designated as the Full Avatar seed.

Choosing any of these values automatically would violate the instruction not to guess the split, loss, learning rate, trainable contract, or deterministic seed. The cloud output root and `attempt_001` were therefore not created. Optimizer steps, backward calls, checkpoints, renders, generated images, data mutations, and paper modifications all remain zero.

## Bound facts

- Base checkpoint: `{BASE_PATH}` at step `100000`, SHA256 `{BASE_SHA256}`
- O03 Teacher: `{TEACHER_PATH}`, SHA256 `{TEACHER_SHA256}`
- Equal-step budget: `1200`
- O03 observations: `4` RGB, `4` foreground masks, `4` garment masks, `4` cameras
- Cloud free bytes before output creation: `{free_before}`
- Storage gate: `PASS`; estimated five full checkpoints plus 2 GiB auxiliary data leave `{estimated_remaining}` bytes
- GPU: one NVIDIA GeForce RTX 4090; PyTorch 2.4.1+cu121; CUDA 12.1; gsplat import passed

## Required resolution

Freeze exactly one split/evaluation contract, one loss contract, one optimizer initialization and learning-rate contract, and one seed. No other garment or baseline may start from this task.

Unique next task: `{NEXT_TASK}`

`PAPER_FINAL=false`
"""
    write_text(REPORT_PATH, report)

    handoff = {
        "schema_version": "canondressgs.project_control.full_avatar_o03_equal_step_micropilot.v1",
        "task_id": TASK_ID, "branch": NEW_BRANCH, "base_head": SOURCE_HEAD,
        "preflight": str(PREFLIGHT_PATH.relative_to(ROOT)).replace("\\", "/"),
        "summary": str(SUMMARY_PATH.relative_to(ROOT)).replace("\\", "/"),
        "report": str(REPORT_PATH.relative_to(ROOT)).replace("\\", "/"),
        "classification": FINAL_CLASSIFICATION, "next_task": NEXT_TASK,
        "next_task_started": False, "optimizer_steps": 0,
        "output_root_created": False, "paper_final": False,
    }
    write_json(HANDOFF_PATH, handoff)


if __name__ == "__main__":
    main()
