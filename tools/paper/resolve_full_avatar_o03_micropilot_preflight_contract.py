#!/usr/bin/env python3
"""Freeze the zero-step Full Avatar O03 execution contract and storage correction."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "AAAI27-FULL-AVATAR-O03-MICROPILOT-PREFLIGHT-RESOLUTION-001"
SOURCE_BRANCH = "research/full-avatar-finetuning-o03-equal-step-micropilot-20260726"
SOURCE_HEAD = "efe576f599b6bee45aad00bd10cb079a403dac05"
NEW_BRANCH = "research/full-avatar-o03-micropilot-preflight-resolution-20260726"
OLD_PREFLIGHT = ROOT / "paper_protocol/external_baselines/full_avatar_o03_equal_step_micropilot/full_avatar_o03_equal_step_micropilot_preflight.json"
OLD_SUMMARY = ROOT / "paper_protocol/external_baselines/full_avatar_o03_equal_step_micropilot/full_avatar_o03_equal_step_micropilot_final_summary.json"
OVERLAY = ROOT / "paper_protocol/external_baselines/full_avatar_o03_preflight_correction_overlay_20260726.json"
CONTRACT = ROOT / "paper_protocol/external_baselines/FULL_AVATAR_O03_EQUALSTEP_EXECUTION_CONTRACT_20260726.md"
MANIFEST = ROOT / "paper_protocol/external_baselines/full_avatar_o03_equalstep_execution_manifest_draft_20260726.json"
AUDIT = ROOT / "paper_protocol/external_baselines/full_avatar_o03_preflight_resolution_audit_20260726.json"
SUMMARY = ROOT / "paper_protocol/external_baselines/full_avatar_o03_preflight_resolution_final_summary_20260726.json"
REPORT = ROOT / "docs/PAPER/AAAI27_FULL_AVATAR_O03_PREFLIGHT_CORRECTION_20260726.md"
HANDOFF = ROOT / "project_control_handoff/full_avatar_o03_preflight_resolution_handoff.json"

OLD_FREE = 46_450_372_608
OLD_PROJECTED_FREE = 35_769_837_450
LIVE_FREE = 46_450_196_480
FORMAL_GATE = 40_802_189_312
OPERATIONAL_MARGIN = 2_147_483_648
RECOMMENDED_FINAL_FREE = FORMAL_GATE + OPERATIONAL_MARGIN
CHECKPOINT_BYTES = 1_706_610_302
CHECKPOINT_COUNT = 5
AUXILIARY_RESERVATION = {
    "logs": 268_435_456,
    "metrics": 134_217_728,
    "review_renders": 1_610_612_736,
    "runtime_temporary_non_checkpoint": 67_108_864,
    "git_and_report": 67_108_864,
}
AVATARREX_BYTES = 12_569_755_256
AVATARREX_SHA256 = "531bd1c71ad9b35f6ae0e2595ee531aa7ba1f83c242f18f2d0505b2dcd5fbcc1"

PARAMETERS = {
    "dxyz": 30_000,
    "scales": 600_000,
    "quats": 800_000,
    "opacities": 200_000,
    "sh0": 600_000,
    "shN": 1_800_000,
    "dxyz_bs": 450_000,
    "dscales_bs": 9_000_000,
    "dquats_bs": 12_000_000,
    "dopacities_bs": 3_000_000,
    "dsh0_bs": 9_000_000,
    "dshN_bs": 27_000_000,
    "encoder_feat_params": 91_017_000,
    "xyz_offset": 600_000,
}

LEARNING_RATES = {
    "dxyz": 0.00016487699756168338,
    "scales": 0.00037494710466721114,
    "quats": 0.00037494710466721114,
    "opacities": 0.00037494710466721114,
    "sh0": 0.00037494710466721114,
    "shN": 0.000018747355233360707,
    "dxyz_bs": 0.00002198670099195661,
    "dscales_bs": 0.0001,
    "dquats_bs": 0.0001,
    "dopacities_bs": 0.0001,
    "dsh0_bs": 0.0001,
    "dshN_bs": 0.0000025,
    "encoder_feat_params": 0.00037494710466721114,
    "xyz_offset": 0.0007498942093344223,
}

LOSS_WEIGHTS = {
    "garment_rgb": 1.0,
    "alpha_foreground": 0.5,
    "new_silhouette_alpha": 1.0,
    "boundary_rgb": 0.25,
    "protected_rgb": 10.0,
    "protected_alpha": 5.0,
    "stability": 0.0001,
}

TARGET_CHECKSUMS = {
    "cond_000000": {
        "target_base_foreground_mask": "09955b8ad995aaf824a478b62e6bc7e7d97e69b1689a9762ee799ea29274a5eb",
        "target_base_rgb": "f05effa5ee2e79eef1f68b5fb6bb6b58869a36929c46aaa2126ab5d756a91d2b",
        "target_clothing_mask": "a25b9f0e7586fbfe3ce5e6c6035cd5df2fb5cae540a0a46c27be4027d0a222c6",
        "target_clothing_mask_raw": "14f3996c05834388813f6656d0b32f1f7348f5c6b3aa050939b73f23959a2341",
        "target_edit_core_mask": "4bcfe33212e7a9ba763450f52e4ea3c9f337d527dad9a6ed7e84d16b5b31a664",
        "target_edit_mask": "a41ca7325085a28885df039b1431b8af9025aab2eb6103024978890f9b4b07f9",
        "target_edit_rgb": "b11e0feb45b747cffe74613b4fb7d6cf9e290e08f2251317ffa49ee64c512117",
        "target_foreground_mask": "82f7dd71f6ba9fb37e07c02649f3900a3f6f6e08ee08d3d94925548de4458f10",
        "target_old_clothing_mask": "54b60eda3ba9d3b66f8ec2a435ae464bae33a6257f4fd3fa21257befb2c11c0a",
        "target_preserve_mask": "23f7f2b46d95f1ec10145fb3f534854c67011a79e4a6421d51b045c9a7b732f0",
        "target_protected_mask": "c78c4e5cb8b16800ea6bdf339d34e03311f5f3422323c7c1ca2d71c42e910a17",
        "target_revealed_skin_mask": "e66e4b081edf6111beb8caf60edadf38dfa9a4aa49e6469dea0abdcb1a0956d7",
        "target_transition_mask": "0830ef251e895102e6348c5d15f254ed3e2bef725722bc5fbb9ac5ad609eae37",
    },
    "cond_000318": {
        "target_base_foreground_mask": "d3760119347e3cb7403c6d3a7b990b15dd602d0dde1d7b30f17bd55bead46649",
        "target_base_rgb": "99e220dbe08d6ee9cdbad5f9645c110dba499acf17b1c37c3ed806319eb7eda3",
        "target_clothing_mask": "fccc2d7523792ad2adf66772e3068815026e9cb13f877e83a1a2a0235ef05897",
        "target_clothing_mask_raw": "6b752cd3d26843aae28079094bd014401a8ff6df0ff730b9a74762c455577cef",
        "target_edit_core_mask": "d6f4a847f4b318c26834ef123fadc6ea4fcd64e6549233c3ffbaf7380fa4e803",
        "target_edit_mask": "6b1d2eb972e5cc70329fc13443095b95879d3e33cd45342d28eac7dd0ba5cf34",
        "target_edit_rgb": "2884331627d856c774de8368aad9af043dac5d4b0957accd85a8afab5f7b954a",
        "target_foreground_mask": "8b21f3060feb2ff306b04bb048fed323b8c23fcf577976e7c3cef024b6065340",
        "target_old_clothing_mask": "6c00d7377c9ae79ed204feff34f3cf1e4f1986896f98ec886ddb202183dea03a",
        "target_preserve_mask": "f5e4aca7c2f3d564811b9a17ce9979791a83c4d49dcae1a424b55808e4b1b895",
        "target_protected_mask": "df256a0d252cdb354491251fed6d0dcdc4faccdc0e27d71bb603f7108d4564be",
        "target_revealed_skin_mask": "fc3beef814664f2934174815a3f1540a7fa22645c9f6403cb97c4d30559743fe",
        "target_transition_mask": "88a188d337f6caa94f121b1d9c4f6a076e6c5a8c686cbd8c480a3599fe79bedc",
    },
    "cond_000017": {
        "target_base_foreground_mask": "0d69beb1d02ef653160613be8e8272575d535af82025db961a3f9805f62e1385",
        "target_base_rgb": "8c5bae54c1cb34e2db0434b5e2e85b2861a2bbf69f92fd00c36b2e6c8e2ff312",
        "target_clothing_mask": "bdfdc73a9f66657cabcd5fab0ddc88a82fddb703b64a6468184323cf185794b9",
        "target_clothing_mask_raw": "4c28bba223329ffd814a2aeca3559fe561e366e876232297d5b39046e80d0275",
        "target_edit_core_mask": "8976f9d4e4720bf336ee2d8cd20f5bc1c15200f6308ce2efe7b38e302534cc04",
        "target_edit_mask": "c426b1414e977b46d9ebe7fcec2147928ca6158b8746b9a3fb46826a51cc89df",
        "target_edit_rgb": "32c70a455396ea33848d49b73520f7c7995dfa04dfde20b8158b2ff7b6d65bb9",
        "target_foreground_mask": "56b2f4ef9178d0ac1e9e9d65da862f44253a81cce924f7b764f2fc38d4fbe185",
        "target_old_clothing_mask": "18f70e61996955080f62b7daf0adf3c8bc89b16ba4a2b00becfb7901e3167cd2",
        "target_preserve_mask": "c599d473ee6952b1c8438092e8fa0abe072387a0e8dac672483154301e00bd8a",
        "target_protected_mask": "b0adbf657e4260bd4403ae0ca0fd4f334f311b29a04b54d2a2ac52719158a874",
        "target_revealed_skin_mask": "fc3beef814664f2934174815a3f1540a7fa22645c9f6403cb97c4d30559743fe",
        "target_transition_mask": "c503e4c3c0941aa28d2462fee4d85c9b369d06911ce27aa5fe5fe9ed288729a0",
    },
    "cond_000347": {
        "target_base_foreground_mask": "b94a239c7a7cfa1d48cb8c72e0044b493f91d82d68a9264ddd0c51b1af69f888",
        "target_base_rgb": "43f28c9f03f19c4e93f3c489cf467bff8fea7d6f0c7cf57d2fb69d61e988a305",
        "target_clothing_mask": "4123167a1bf3aff44d130e3d03a4d49cc606321f31891b4939e77e92be5189fd",
        "target_clothing_mask_raw": "79c156c01554499eaee3c21e575e49750ac9917e1c90e63868161502f4ea207b",
        "target_edit_core_mask": "41cb52e9ae07dc57d9f0a3e21e6de989a06da4bc2fe6c5abc7d47cdda98f6cc2",
        "target_edit_mask": "3a65eb73065af04c878158427282e2b323ac57c54dfe0315b96a99b25c4dcdc7",
        "target_edit_rgb": "0fa201528e834029cdcaa887d8ec76b855c8dc6680bd5ad79ba9228fa559b7d6",
        "target_foreground_mask": "7957e74d8e494ac3819d7f35c07fb27dc85f6c05fbe89765f6a2e6edda561a6b",
        "target_old_clothing_mask": "c74a97ec0e0cfaca0d87b8f1e221d1de372e5035325f61f564b31c28cedd1e22",
        "target_preserve_mask": "12bd46db4c254a5cdea9fc70c7efad57a9cbee187815254267b1f8b671209b3d",
        "target_protected_mask": "5a6c2395ee490a1f0805bfc996622481e0a445c25f16a36009040a50090a87dc",
        "target_revealed_skin_mask": "fc3beef814664f2934174815a3f1540a7fa22645c9f6403cb97c4d30559743fe",
        "target_transition_mask": "74da18e4837267e2e67ba2e501cb54d357d4785b3891e18ca80c27ddca229106",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def data_paths(condition: str) -> dict[str, dict[str, str]]:
    dataset = "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-AAAI27-DATA-CAPACITY-GATE-001/attempt_003/dataset"
    result: dict[str, dict[str, str]] = {}
    for role, checksum in TARGET_CHECKSUMS[condition].items():
        if role == "target_base_rgb":
            path = f"/root/autodl-tmp/canondressgs_work/data/subject02_dual_target_v5_2/rgb/base/{condition}.png"
        elif role == "target_edit_rgb":
            path = f"{dataset}/rgb/edit/O03/{condition}.png"
        else:
            path = f"{dataset}/masks/{role}/O03/{condition}.png"
        result[role] = {"path": path, "sha256": checksum}
    return result


def main() -> None:
    old = json.loads(OLD_PREFLIGHT.read_text(encoding="utf-8"))
    old_summary = json.loads(OLD_SUMMARY.read_text(encoding="utf-8"))
    assert old_summary["optimizer_steps_completed"] == 0
    assert old_summary["checkpoint_count"] == 0
    assert old_summary["storage_preflight"] == "PASS"
    assert sha256(OLD_PREFLIGHT) == "4788e9503999e30faf9c4e325068d02d38001e8bcfefd0fa558c57df0bf1bdc4"
    assert sum(PARAMETERS.values()) == 156_097_000
    assert sum(AUXILIARY_RESERVATION.values()) == OPERATIONAL_MARGIN

    original_consumption = OLD_FREE - OLD_PROJECTED_FREE
    original_shortfall = FORMAL_GATE - OLD_PROJECTED_FREE
    steady = CHECKPOINT_BYTES * CHECKPOINT_COUNT + sum(AUXILIARY_RESERVATION.values())
    peak = steady + CHECKPOINT_BYTES
    projected_final = LIVE_FREE - steady
    projected_minimum = LIVE_FREE - peak
    safety_shortfall = FORMAL_GATE - projected_minimum
    release_final = LIVE_FREE + AVATARREX_BYTES - steady
    release_minimum = LIVE_FREE + AVATARREX_BYTES - peak

    overlay = {
        "schema_version": "canondressgs.full_avatar_o03.preflight_correction_overlay.v1",
        "task_id": TASK_ID,
        "original_preflight": str(OLD_PREFLIGHT.relative_to(ROOT)).replace("\\", "/"),
        "original_preflight_sha256": sha256(OLD_PREFLIGHT),
        "original_report_preserved": True,
        "original_storage_classification": "PASS",
        "corrected_storage_classification": "FAIL_BELOW_SAFETY_GATE",
        "cloud_free_bytes_before": OLD_FREE,
        "projected_consumption_bytes": original_consumption,
        "projected_final_free_bytes": OLD_PROJECTED_FREE,
        "formal_safety_gate_bytes": FORMAL_GATE,
        "safety_shortfall_bytes": original_shortfall,
        "correction_kind": "APPEND_ONLY_OVERLAY_NO_ORIGINAL_MUTATION",
    }
    write_json(OVERLAY, overlay)

    lr_gamma = {
        "dxyz": 0.01 ** (1.0 / 1200),
        "default_exponential": 0.1 ** (1.0 / 1200),
    }
    scheduled = ["dxyz", "scales", "quats", "opacities", "sh0", "shN", "dxyz_bs", "encoder_feat_params", "xyz_offset"]
    constant = ["dscales_bs", "dquats_bs", "dopacities_bs", "dsh0_bs", "dshN_bs"]
    scheduler_groups = {
        name: {
            "family": "torch.optim.lr_scheduler.ExponentialLR",
            "initial_lr": LEARNING_RATES[name],
            "gamma": lr_gamma["dxyz"] if name == "dxyz" else lr_gamma["default_exponential"],
            "local_steps": 1200,
            "final_lr_after_1200_steps": LEARNING_RATES[name] * (0.01 if name == "dxyz" else 0.1),
        }
        for name in scheduled
    }
    scheduler_groups.update({
        name: {
            "family": "constant_no_scheduler_as_in_formal_base_trainer",
            "initial_lr": LEARNING_RATES[name],
            "local_steps": 1200,
            "final_lr_after_1200_steps": LEARNING_RATES[name],
        }
        for name in constant
    })

    target_registry = old["target_contract"]["registry"]
    targets = {
        condition: {
            "assets": data_paths(condition),
            "camera": {
                "path": old["target_contract"]["manifest"] + f"#conditions[{condition}].camera",
                "sha256": target_registry["camera_sha256"][condition],
            },
            "pose": {
                "path": old["target_contract"]["manifest"] + f"#conditions[{condition}].pose",
                "sha256": target_registry["pose_sha256"][condition],
            },
        }
        for condition in target_registry["condition_order"]
    }

    manifest = {
        "schema_version": "canondressgs.full_avatar_o03_equalstep.execution_manifest_draft.v1",
        "task_id": TASK_ID,
        "status": "FROZEN_STORAGE_RESOLUTION_PENDING",
        "source": {"branch": SOURCE_BRANCH, "head": SOURCE_HEAD, "new_branch": NEW_BRANCH},
        "base": old["base_avatar"],
        "target": {
            "subject": "THuman4.0 subject02",
            "garment": "O03",
            "manifest_path": old["target_contract"]["manifest"],
            "manifest_sha256": old["target_contract"]["manifest_sha256"],
            "conditions": targets,
        },
        "view_protocol": {
            "name": "VIEW_TRANSDUCTIVE_MATCHED_ADAPTATION",
            "optimization_view_count": 4,
            "target_space_evaluation_view_count": 4,
            "optimization_evaluation_view_overlap": "4/4",
            "target_leakage_disclosure": "INTENTIONAL_PRIVILEGED_ADAPTATION_ON_REGISTERED_TARGET_VIEWS_DISCLOSED",
            "method_role": "PRIVILEGED_PER_GARMENT_ADAPTATION_CONTROL",
            "different_pose_camera_role": "ANIMATION_COMPATIBILITY_REVIEW_ONLY_NO_TARGET_SPACE_AVERAGE",
        },
        "loss": {
            "contract": "EXACT_O03_TEACHER_OBJECTIVE_MATCH",
            "name": "CAPACITY_ORACLE_LOSS_V1",
            "weights": LOSS_WEIGHTS,
            "ssim_component": "ABSENT_IN_FORMAL_O03_TEACHER_OBJECTIVE",
            "perceptual_component": "ABSENT_IN_FORMAL_O03_TEACHER_OBJECTIVE",
            "mask_semantics": {
                "garment": "max(edit_mask, clothing_mask, old_clothing_mask) * (1-protected_mask)",
                "foreground": "target_foreground",
                "new_silhouette": "target_foreground * (1-base_foreground)",
                "boundary": "transition_mask * (1-protected_mask)",
                "protected": "protected_mask",
            },
            "denominator": "sum(abs_error * expanded_mask) / max(sum(expanded_mask), 1); alpha_foreground uses an all-pixel mask",
            "stability": "direct_stability: sum of mean squared trainable-support values over raw_xyz, raw_log_scaling, raw_rotvec, raw_opacity, raw_sh0",
            "implementation": "scene/representation_capacity_oracle.py::capacity_oracle_loss_v1 and direct_stability; tools/run_multi_outfit_explicit_basis.py::run_teacher",
            "implementation_sha256": {
                "scene/representation_capacity_oracle.py": "1fad8f7185766fedf85c9bc2e60238ad69c6ef7ac96f28263a320733c14e187c",
                "tools/run_multi_outfit_explicit_basis.py": "aa495762810bb4095e12d508eb1a951bdec469e1ac8700f690bbc3651e7e70ce",
            },
            "resolved_teacher_config_path": "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-MULTI-OUTFIT-EXPLICIT-BASIS-001/attempt_003/contract/config_resolved.yaml",
            "resolved_teacher_config_sha256": "f0c4c8d3c63b935575c43a00c44ca9a1bd472fbed87329c019ff0d87a8d2c60f",
        },
        "parameters": {
            "policy": "ALL_FORMAL_BASE_OPTIMIZER_GROUPS",
            "trainable_groups": PARAMETERS,
            "trainable_tensor_count": 23,
            "trainable_parameter_count": sum(PARAMETERS.values()),
            "trainable_parameter_bytes_float32": sum(PARAMETERS.values()) * 4,
            "frozen_parameter_count_within_adaptation_model": 0,
            "frozen_checkpoint_state_value_count": 14_166_176,
            "frozen_checkpoint_state_note": "Immutable tensors/buffers are state, not nn.Parameter values, and are excluded from frozen_parameter_count.",
            "forbidden_initialization": ["CanonDressGS endpoint coefficients", "Teacher residuals", "garment residuals", "snapped endpoint", "O03 Teacher checkpoint"],
        },
        "optimizer": {
            "state_policy": "FRESH_RESET_NO_BASE_MOMENTUM_RESTORE",
            "groups": {
                name: {
                    "class": "torch.optim.AdamW" if name == "encoder_feat_params" else "torch.optim.Adam",
                    "lr": LEARNING_RATES[name],
                    "betas": [0.9, 0.999],
                    "eps": 1e-15,
                    "weight_decay": 0.001 if name == "encoder_feat_params" else 0.0,
                }
                for name in PARAMETERS
            },
            "gradient_clip": "NONE_AS_IN_FORMAL_BASE_TRAINER",
            "source_registry": "scene/gaussian_model.py::GaussianModel.training_setup",
            "source_sha256": "131eb92881a483b71d8e4f27efb585c0bd794615f4c7a0cf39af6bf6f09538b5",
        },
        "scheduler": {
            "policy": "FRESH_LOCAL_1200_STEP_SCHEDULE_ANCHORED_AT_BASE_STEP100000_EFFECTIVE_LR",
            "groups": scheduler_groups,
            "base_moments_restored": False,
            "base_scheduler_state_restored": False,
        },
        "determinism": {
            "seed": 0,
            "covered": ["Python", "NumPy", "PyTorch CPU", "PyTorch CUDA", "data-loader generator", "sampling order"],
            "sampling_order": ["cond_000000", "cond_000318", "cond_000017", "cond_000347"],
        },
        "execution": {
            "steps": 1200,
            "checkpoint_milestones": [0, 300, 600, 900, 1200],
            "gpu": "single NVIDIA GeForce RTX 4090",
            "output_root": "/root/autodl-tmp/canondressgs_work/outputs/FULL-AVATAR-FINETUNE-O03-EQUALSTEP-MICROPILOT-001",
            "attempt_id": "attempt_001",
            "resume_rule": "After any optimizer step, resume only from the latest complete atomic checkpoint; restart from zero is forbidden.",
            "generation_authorized": False,
            "training_authorized": False,
        },
        "storage": {
            "live_free_bytes": LIVE_FREE,
            "checkpoint_count": CHECKPOINT_COUNT,
            "per_checkpoint_reservation_bytes": CHECKPOINT_BYTES,
            "checkpoint_reservation_basis": "Exact formal Base checkpoint including mature optimizer state; conservative for fresh step-0.",
            "optimizer_state_tensor_bytes_per_mature_checkpoint": 1_018_376_084,
            "auxiliary_reservation_bytes": AUXILIARY_RESERVATION,
            "steady_state_storage_bytes": steady,
            "atomic_temporary_checkpoint_bytes": CHECKPOINT_BYTES,
            "peak_write_storage_bytes": peak,
            "projected_final_free_bytes": projected_final,
            "projected_minimum_free_bytes": projected_minimum,
            "formal_safety_gate_bytes": FORMAL_GATE,
            "recommended_operational_margin_bytes": OPERATIONAL_MARGIN,
            "recommended_final_free_bytes": RECOMMENDED_FINAL_FREE,
            "formal_safety_shortfall_bytes": safety_shortfall,
            "gate": "FAIL_BELOW_SAFETY_GATE",
            "resolution": "PLAN_A_PENDING_AVATARREX_PLAN_B_COMPLETION",
        },
        "authorization": {"training_authorized": False, "generation_authorized": False},
    }
    write_json(MANIFEST, manifest)

    audit = {
        "schema_version": "canondressgs.full_avatar_o03.preflight_resolution_audit.v1",
        "task_id": TASK_ID,
        "correction": overlay,
        "scientific_contract": {
            "view": manifest["view_protocol"],
            "loss": manifest["loss"],
            "parameters": manifest["parameters"],
            "optimizer": manifest["optimizer"],
            "scheduler": manifest["scheduler"],
            "seed": 0,
            "steps": 1200,
        },
        "storage": manifest["storage"],
        "avatarrex": {
            "windows_archive_path": "E:\\canondressgs_archive\\cloud_datasets\\AvatarReX\\payload\\root\\autodl-tmp\\datasets\\avatarrex_second_dataset_staging\\downloads\\avatarrex_lbn1.7z",
            "windows_archive_status": "PASS_BYTE_AND_SHA_MATCH",
            "windows_archive_bytes": AVATARREX_BYTES,
            "windows_archive_sha256": AVATARREX_SHA256,
            "windows_extracted_copy_status": "PASS_60834_FILES_FULL_SCAN_VERIFIED",
            "cloud_archive_path": "/root/autodl-tmp/avatarrex_lbn1.7z",
            "cloud_archive_status": "PASS_BYTE_AND_SHA_MATCH_NOT_DELETED",
            "cloud_archive_bytes": AVATARREX_BYTES,
            "cloud_archive_sha256": AVATARREX_SHA256,
            "cloud_is_unique_copy": False,
            "cloud_plan_b_staging_status": "INCOMPLETE_ONLY_SMPL_PARAMS_NPZ_PRESENT",
            "cloud_plan_b_staging_file_count": 1,
            "historical_complete_extraction_file_count": 60_834,
            "reupload_source_exists": True,
            "recommended_plan": "PLAN_A_PENDING_AVATARREX_PLAN_B_COMPLETION",
            "future_plan_after_plan_b": "USER_AUTHORIZED_DELETE_VERIFIED_CLOUD_DUPLICATE_AVATARREX_ARCHIVE",
            "expected_release_bytes": AVATARREX_BYTES,
            "projected_final_free_after_release_and_training": release_final,
            "projected_minimum_free_after_release_and_atomic_write": release_minimum,
            "deleted_by_this_task": False,
        },
        "operations": {
            "optimizer_steps": 0,
            "forward_backward_scientific_runs": 0,
            "renderer_inferences": 0,
            "checkpoint_writes": 0,
            "output_root_creations": 0,
            "avatarrex_deletions": 0,
            "data_modifications": 0,
            "paper_modifications": 0,
        },
        "training_authorized": False,
        "paper_final": False,
        "final_classification": "FULL_AVATAR_O03_EXECUTION_CONTRACT_FROZEN_STORAGE_RESOLUTION_PENDING",
        "next_task": "USER_AUTHORIZE_AVATARREX_CLOUD_DUPLICATE_DELETION_FOR_FULL_AVATAR_MICROPILOT",
    }
    write_json(AUDIT, audit)

    contract_text = f"""# Full Avatar O03 Equal-Step Execution Contract

Status: `FROZEN_STORAGE_RESOLUTION_PENDING`. Training and generation are not authorized.

## Identity, Views, and Data

- Base: `{old['base_avatar']['path']}`, SHA256 `{old['base_avatar']['sha256']}`, step `100000`.
- Target manifest: `{old['target_contract']['manifest']}`, SHA256 `{old['target_contract']['manifest_sha256']}`.
- Protocol: `VIEW_TRANSDUCTIVE_MATCHED_ADAPTATION`. All four registered O03 views are used for adaptation and matched target-space evaluation; overlap is `4/4`.
- Disclosure: `INTENTIONAL_PRIVILEGED_ADAPTATION_ON_REGISTERED_TARGET_VIEWS_DISCLOSED`. This is a privileged per-garment adaptation control, not reference-controlled deployment, novel-view, leave-one-view-out, or target-independent adaptation.
- Different-pose and different-camera renders are animation-compatibility review only and do not enter target-space means without real targets.

## Objective

Use `EXACT_O03_TEACHER_OBJECTIVE_MATCH`: `CAPACITY_ORACLE_LOSS_V1` with weights `{json.dumps(LOSS_WEIGHTS, sort_keys=True)}`. The formal Teacher objective has no SSIM or perceptual component; exact matching keeps both absent. Mask roles, active-pixel denominators, stability definition, implementation paths, file hashes, and resolved config SHA are frozen in the manifest.

## Parameters and Optimization

- Policy: `ALL_FORMAL_BASE_OPTIMIZER_GROUPS`; 14 groups, 23 tensors, `{sum(PARAMETERS.values()):,}` float32 parameters (`{sum(PARAMETERS.values()) * 4:,}` bytes).
- Fresh optimizer state: `FRESH_RESET_NO_BASE_MOMENTUM_RESTORE`. Base momentum, scheduler state, scaler, loader cursor, and RNG state are not restored.
- Adam is used for 13 groups; `encoder_feat_params` uses AdamW with weight decay `0.001`. Betas are `(0.9, 0.999)`, epsilon is `1e-15`, and Base trainer gradient clipping is absent.
- Initial per-group LRs are the exact effective values stored at Base step 100000. Nine original ExponentialLR groups restart locally with a 1200-step horizon and the original family terminal factors (`0.01` for `dxyz`, `0.1` for the other eight); five originally unscheduled groups stay constant.
- Seed is `0` for Python, NumPy, PyTorch CPU/CUDA, loader generator, and the fixed four-condition sampling order.

## Steps, Checkpoints, and Resume

Exactly `1200` optimizer steps are required. Atomic checkpoints are fixed at local steps `0, 300, 600, 900, 1200`, with no best-checkpoint selection. After any optimizer step, only exact resume from the latest complete checkpoint is allowed; restarting from zero is forbidden.

## Storage Gate

The append-only correction changes the old storage classification from `PASS` to `FAIL_BELOW_SAFETY_GATE`. The current byte-exact conservative reservation is `{steady:,}` bytes at steady state and `{peak:,}` bytes during an atomic checkpoint write. From the captured `{LIVE_FREE:,}` free bytes, minimum projected free space is `{projected_minimum:,}`, short of the `{FORMAL_GATE:,}` formal gate by `{safety_shortfall:,}` bytes.

The Windows AvatarReX 7z and cloud 7z both rehash to `{AVATARREX_SHA256}` with `{AVATARREX_BYTES:,}` bytes, and Windows also contains a fully verified 60,834-file extraction. Current cloud Plan B staging is incomplete, so the allowed recommendation is `PLAN_A_PENDING_AVATARREX_PLAN_B_COMPLETION`. No deletion occurred. If Plan B completes and the user separately authorizes deletion, projected minimum free during training becomes `{release_minimum:,}` bytes.

## Authorization

Output root: `/root/autodl-tmp/canondressgs_work/outputs/FULL-AVATAR-FINETUNE-O03-EQUALSTEP-MICROPILOT-001`; attempt: `attempt_001`. Both remain uncreated. `training_authorized=false`, `generation_authorized=false`, optimizer steps `0`, and `PAPER_FINAL=false`.
"""
    CONTRACT.write_text(contract_text, encoding="utf-8")

    report_text = f"""# Full Avatar O03 Preflight Correction

Task: `{TASK_ID}`

## Storage Correction

The original preflight is preserved at `{OVERLAY.parent.relative_to(ROOT) / 'full_avatar_o03_equal_step_micropilot/full_avatar_o03_equal_step_micropilot_preflight.json'}` with SHA256 `{sha256(OLD_PREFLIGHT)}`. Its `PASS` storage label was arithmetically wrong: `{OLD_FREE:,} - {OLD_PROJECTED_FREE:,} = {original_consumption:,}` bytes projected consumption, while `{FORMAL_GATE:,} - {OLD_PROJECTED_FREE:,} = {original_shortfall:,}` bytes remain below the formal safety gate. The append-only overlay therefore sets `FAIL_BELOW_SAFETY_GATE`.

## Frozen Contract

The four scientific blockers are resolved without training: four-view matched transductive adaptation with explicit disclosure; exact seven-term O03 Teacher capacity loss; all 14 formal Base optimizer groups with fresh state; Base-step-100000 effective LRs mapped to a local 1200-step scheduler; and seed 0. SSIM and perceptual terms are absent because the formal O03 Teacher objective does not contain them.

## Storage Resolution

Steady reservation is `{steady:,}` bytes and atomic-write peak is `{peak:,}` bytes. Captured free space is `{LIVE_FREE:,}`, giving `{projected_minimum:,}` minimum projected free bytes and a `{safety_shortfall:,}`-byte formal shortfall. Windows and cloud AvatarReX archives match `{AVATARREX_BYTES:,}` bytes and SHA256 `{AVATARREX_SHA256}`. Cloud Plan B staging is currently incomplete, so deletion remains prohibited and the resolution is `PLAN_A_PENDING_AVATARREX_PLAN_B_COMPLETION`.

No optimizer step, output directory, checkpoint, render, deletion, data mutation, or paper modification occurred. Final classification: `FULL_AVATAR_O03_EXECUTION_CONTRACT_FROZEN_STORAGE_RESOLUTION_PENDING`. `training_authorized=false`; `PAPER_FINAL=false`.
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report_text, encoding="utf-8")

    summary = {
        "schema_version": "canondressgs.full_avatar_o03.preflight_resolution_final_summary.v1",
        "task_id": TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "new_branch": NEW_BRANCH,
        "worktree": "E:\\model_train\\canondressgs_full_avatar_o03_micropilot_preflight_resolution",
        "original_preflight_report": str(OLD_PREFLIGHT.relative_to(ROOT)).replace("\\", "/"),
        "original_storage_classification": "PASS",
        "corrected_storage_classification": "FAIL_BELOW_SAFETY_GATE",
        "cloud_free_bytes": LIVE_FREE,
        "projected_consumption_bytes": original_consumption,
        "projected_final_free_bytes": projected_final,
        "formal_safety_gate_bytes": FORMAL_GATE,
        "safety_shortfall_bytes": safety_shortfall,
        "recommended_margin_bytes": OPERATIONAL_MARGIN,
        "view_protocol": "VIEW_TRANSDUCTIVE_MATCHED_ADAPTATION",
        "optimization_view_count": 4,
        "evaluation_view_count": 4,
        "view_overlap": "4/4",
        "target_leakage_disclosure": "INTENTIONAL_PRIVILEGED_ADAPTATION_ON_REGISTERED_TARGET_VIEWS_DISCLOSED",
        "loss_contract": "EXACT_O03_TEACHER_OBJECTIVE_MATCH",
        "loss_components": LOSS_WEIGHTS,
        "trainable_group_policy": "ALL_FORMAL_BASE_OPTIMIZER_GROUPS",
        "trainable_groups": list(PARAMETERS),
        "trainable_parameter_count": sum(PARAMETERS.values()),
        "frozen_groups": ["camera/calibration", "target RGB/masks/manifests", "SMPL-X observations", "evaluator/metric networks", "renderer/rasterizer", "view protocol", "non-parameter checkpoint state"],
        "frozen_parameter_count": 0,
        "optimizer": "Adam x13; AdamW x1",
        "optimizer_state_policy": "FRESH_RESET_NO_BASE_MOMENTUM_RESTORE",
        "learning_rates": LEARNING_RATES,
        "lr_policy": "FORMAL_BASE_EFFECTIVE_LR_AT_CHECKPOINT_STEP_100000",
        "scheduler": "ExponentialLR x9; constant x5",
        "scheduler_policy": "FRESH_LOCAL_1200_STEP_SCHEDULE_ANCHORED_AT_BASE_STEP100000_EFFECTIVE_LR",
        "seed": 0,
        "steps": 1200,
        "checkpoint_milestones": [0, 300, 600, 900, 1200],
        "steady_state_storage_bytes": steady,
        "peak_write_storage_bytes": peak,
        "projected_minimum_free_bytes": projected_minimum,
        "windows_avatarrex_archive_status": "PASS_BYTE_AND_SHA_MATCH",
        "cloud_avatarrex_archive_status": "PASS_BYTE_AND_SHA_MATCH_NOT_DELETED",
        "avatarrex_plan_b_status": "INCOMPLETE_ONLY_SMPL_PARAMS_NPZ_PRESENT",
        "recommended_storage_plan": "PLAN_A_PENDING_AVATARREX_PLAN_B_COMPLETION",
        "expected_release_bytes": AVATARREX_BYTES,
        "projected_free_after_release_and_training": release_final,
        "projected_minimum_free_after_release_and_atomic_write": release_minimum,
        "training_authorized": False,
        "execution_contract_path": str(CONTRACT.relative_to(ROOT)).replace("\\", "/"),
        "execution_manifest_path": str(MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "paper_modifications": 0,
        "optimizer_steps": 0,
        "paper_final": False,
        "final_classification": "FULL_AVATAR_O03_EXECUTION_CONTRACT_FROZEN_STORAGE_RESOLUTION_PENDING",
        "next_task": "USER_AUTHORIZE_AVATARREX_CLOUD_DUPLICATE_DELETION_FOR_FULL_AVATAR_MICROPILOT",
    }
    write_json(SUMMARY, summary)
    write_json(HANDOFF, {
        "task_id": TASK_ID,
        "branch": NEW_BRANCH,
        "source_head": SOURCE_HEAD,
        "classification": summary["final_classification"],
        "training_authorized": False,
        "optimizer_steps": 0,
        "paper_modifications": 0,
        "paper_final": False,
        "next_task": summary["next_task"],
        "contract": summary["execution_contract_path"],
        "manifest": summary["execution_manifest_path"],
    })


if __name__ == "__main__":
    main()
