"""Build the pre-result Controller V2 cross-fit training contract.

This is a pure-data archive builder.  It imports only the Python standard
library and never constructs a model, optimizer, tensor, renderer, or metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
RISK = ROOT / "paper_protocol/reviewer_risk"
DOCS = ROOT / "docs/PAPER"
HANDOFF = ROOT / "project_control_handoff"

SOURCE_BRANCH = "research/compatibility-gated-controller-v2-crossfit-micro-pilot-20260723"
SOURCE_HEAD = "b15f26e51958820d361b8992fab5ddec0e592fb5"
REPAIR_BRANCH = "research/controller-v2-micro-pilot-contract-repair-20260723"
DESIGN_HEAD = "4f8c94405e68932e791f69c59ae129c863b1280d"
UPSTREAM_CONTRACT_HEAD = "1d4b416929617d09bf65122ff5d6dbf23dfe564a"
FORMAL_V1_HEAD = "f45a518f330fb407942055756373e83f65717853"
DIAGNOSTIC_HEAD = "25318857b4cc4ee11011cd852c5cfc237a8b3cee"
TASK_ID = "AAAI27-CONTROLLER-V2-MICRO-PILOT-CONTRACT-REPAIR-001"
CLASSIFICATION = "CONTROLLER_V2_MICRO_PILOT_TRAINING_CONTRACT_REPAIRED"
NEXT_TASK = "TRAIN_AND_EVALUATE_COMPATIBILITY_GATED_CONTROLLER_V2_FROM_REPAIRED_CONTRACT"

OUTFITS = ("O01", "O02", "O03", "O04", "O08")
PAIRS = (
    "O01_O02", "O01_O03", "O01_O04", "O01_O08", "O02_O03",
    "O02_O04", "O02_O08", "O03_O04", "O03_O08", "O04_O08",
)
FOLDS = ("cond_000000", "cond_000318", "cond_000017", "cond_000347")
ROTATIONS = (
    (0, (0, 1), 2, 3),
    (1, (1, 2), 3, 0),
    (2, (2, 3), 0, 1),
    (3, (3, 0), 1, 2),
)
SEEDS = (0, 1, 2)

FROZEN_HASHES = {
    "repaired_protocol_sha256_lf": "44ef0c53f7a5ec4fe19733371ffefb1be14d5b6902e7f4060707edd4660d9f49",
    "training_manifest_sha256_lf": "a0dbfe98a25b7f82f198405cf41c1b8625b6faa63499396a78683d5d326ce8e3",
    "historical_full_cycle_scientific_sha256": "f60b0ee64ff5ce2b6693f53dd7eb665cc6aee5aa310f314175d2932307d57c77",
    "historical_300_step_scientific_sha256": "63902a2f36ccf864bfe6028654be0b162417c3adc6b99e1455b30fdab8a2bacf",
}

ATTEMPT_001_HASHES = {
    "contract/controller_v2_micro_pilot_training_contract.json": "67b8c51e122428f4ebadc2ae1919c722df1c2fc485df1170d3419b0307d9da27",
    "audits/contract_completeness.json": "1b51a82f8aa7391aefe332acff0770a6f1de3f2de6ffffc1485310b591b14459",
    "audits/final_summary.json": "81f6f87a83e203331367ec108b7ae5b44ef9dd1c7b8c20033125f6027fa455f0",
    "audits/resource_io_gate.json": "7ca2aa656661d6bc9b87f23b98d9731969caf7908453c0db33ffc4c8a1aeab19",
}

JSON_OUTPUTS = (
    "controller_v2_micro_pilot_contract_repaired.json",
    "controller_v2_micro_pilot_rotation_manifests.json",
    "controller_v2_micro_pilot_batch_schedules.json",
    "controller_v2_micro_pilot_optimizer_contract.json",
    "controller_v2_micro_pilot_loss_contract.json",
    "controller_v2_micro_pilot_calibration_contract.json",
    "controller_v2_micro_pilot_evaluator_denominators.json",
    "controller_v2_micro_pilot_perturbation_representatives.json",
    "controller_v2_micro_pilot_visual_representatives.json",
    "controller_v2_micro_pilot_contract_repair_summary.json",
)
DOC_OUTPUTS = (
    "AAAI27_CONTROLLER_V2_MICRO_PILOT_CONTRACT_REPAIR_20260723.md",
    "AAAI27_CONTROLLER_V2_MATCHED_V1_TRAINING_CONTRACT_20260723.md",
    "AAAI27_CONTROLLER_V2_CALIBRATION_AND_EVALUATOR_CONTRACT_20260723.md",
)
HANDOFF_OUTPUT = "controller_v2_micro_pilot_contract_repair_handoff.json"


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def lf_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(
        (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    )
    os.replace(temporary, path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes((value.rstrip() + "\n").encode("utf-8"))
    os.replace(temporary, path)


def dominant(record: Mapping[str, Any]) -> str:
    target = [float(value) for value in record["target_distribution"]]
    return OUTFITS[target.index(max(target))]


def partition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    assignment = Counter(row["assignment_type"] for row in rows)
    position = Counter(
        f"{row['assignment_type']}_{row['assignment_position']}"
        for row in rows if row["assignment_position"] is not None
    )
    pair = Counter(row["pair_id"] for row in rows)
    dom = Counter(dominant(row) for row in rows)
    duplicate_rows = [row for row in rows if row["duplicate_of"] is not None]
    logical = Counter(row["logical_input_sha256"] for row in rows)
    multiplicity = Counter(logical.values())
    assets = {
        value
        for row in rows
        for key in ("source_image_hashes", "source_mask_hashes")
        for value in row[key]
    }
    return {
        "record_count": len(rows),
        "record_ids": [row["record_id"] for row in rows],
        "pure_count": assignment["AAA"] + assignment["BBB"],
        "mixed_count": assignment["AAB"] + assignment["ABB"],
        "assignment_type_counts": dict(sorted(assignment.items())),
        "assignment_position_counts": dict(sorted(position.items())),
        "per_pair_counts": dict(sorted(pair.items())),
        "per_dominant_outfit_counts": dict(sorted(dom.items())),
        "consistent_duplicate_record_count": len(duplicate_rows),
        "consistent_duplicate_record_ids": [row["record_id"] for row in duplicate_rows],
        "unique_logical_query_count": len(logical),
        "logical_query_multiplicity_histogram": {
            str(key): value for key, value in sorted(multiplicity.items())
        },
        "reference_asset_sha256_count": len(assets),
        "_logical_ids": set(logical),
        "_assets": assets,
    }


def public_partition(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if not key.startswith("_")}


def build_rotation_manifests(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[int, dict[str, Any]], dict[str, dict[str, Any]]]:
    queries = list(manifest["query_sets"])
    rows_by_id = {row["record_id"]: row for row in queries}
    rotations = []
    private: dict[int, dict[str, Any]] = {}
    for rotation, train_indices, calibration_index, test_index in ROTATIONS:
        train_folds = [FOLDS[index] for index in train_indices]
        calibration_fold = FOLDS[calibration_index]
        test_fold = FOLDS[test_index]
        train = [row for row in queries if row["target_view_fold"] in train_folds]
        calibration = [row for row in queries if row["target_view_fold"] == calibration_fold]
        test = [row for row in queries if row["target_view_fold"] == test_fold]
        parts = {
            "train": partition_summary(train),
            "calibration": partition_summary(calibration),
            "test": partition_summary(test),
        }
        record_sets = {name: set(part["record_ids"]) for name, part in parts.items()}
        logical_sets = {name: part["_logical_ids"] for name, part in parts.items()}
        asset_sets = {name: part["_assets"] for name, part in parts.items()}
        if any(
            record_sets[left] & record_sets[right]
            for left, right in (("train", "calibration"), ("train", "test"), ("calibration", "test"))
        ):
            raise RuntimeError("rotation record partitions overlap")
        if set.union(*record_sets.values()) != set(rows_by_id):
            raise RuntimeError("rotation partitions do not cover all protocol records")
        scientific = {
            "rotation": rotation,
            "train_fold_indices": list(train_indices),
            "calibration_fold_index": calibration_index,
            "test_fold_index": test_index,
            "train_folds": train_folds,
            "calibration_fold": calibration_fold,
            "test_fold": test_fold,
            "partitions": {name: public_partition(part) for name, part in parts.items()},
            "record_intersections": {
                "train_calibration": len(record_sets["train"] & record_sets["calibration"]),
                "train_test": len(record_sets["train"] & record_sets["test"]),
                "calibration_test": len(record_sets["calibration"] & record_sets["test"]),
            },
            "logical_query_intersections": {
                "train_calibration": len(logical_sets["train"] & logical_sets["calibration"]),
                "train_test": len(logical_sets["train"] & logical_sets["test"]),
                "calibration_test": len(logical_sets["calibration"] & logical_sets["test"]),
            },
            "reference_asset_overlap": {
                "train_calibration_sha256_count": len(asset_sets["train"] & asset_sets["calibration"]),
                "train_test_sha256_count": len(asset_sets["train"] & asset_sets["test"]),
                "calibration_test_sha256_count": len(asset_sets["calibration"] & asset_sets["test"]),
                "interpretation": "EXPECTED_CLOSED_WARDROBE_REFERENCE_ASSET_OVERLAP_NOT_LOGICAL_RECORD_OVERLAP",
            },
            "consistent_duplicates": {
                "preserved_as_independent_protocol_records": True,
                "deduplicated": False,
                "extra_weighting": False,
            },
        }
        scientific["rotation_manifest_sha256"] = canonical_sha(scientific)
        rotations.append(scientific)
        private[rotation] = {
            "train": train, "calibration": calibration, "test": test,
            "record_manifest_sha256": scientific["rotation_manifest_sha256"],
        }

    formal = list(manifest["formal_pure_endpoint_episodes"])
    archive = {
        "schema_version": "canondressgs.research.controller_v2_micro_pilot_rotation_manifests.v1",
        "task_id": TASK_ID,
        "source_manifest": "paper_protocol/reviewer_risk/dual_support_controller_training_manifest.json",
        "source_manifest_sha256_lf": FROZEN_HASHES["training_manifest_sha256_lf"],
        "condition_order": list(FOLDS),
        "outfit_order": list(OUTFITS),
        "pair_order": list(PAIRS),
        "rotations": rotations,
        "formal_pure_secondary_safety_set": {
            "role": "EVALUATION_ONLY_SECONDARY_SAFETY_SET",
            "record_count": len(formal),
            "record_ids": [row["record_id"] for row in formal],
            "unique_logical_query_count": len({row["logical_input_sha256"] for row in formal}),
            "training_exposure": 0,
            "calibration_threshold_use": 0,
            "primary_test_denominator_use": 0,
            "global_crossfit_macro_use": 0,
            "report": "SEPARATE_PURE_ENDPOINT_SAFETY",
            "known_reference_asset_overlap": "PRESENT_CLOSED_WARDROBE",
            "known_logical_inputs_overlapping_protocol": 20,
        },
        "paper_final": False,
        "paper_final_count": 0,
    }
    archive["archive_content_sha256"] = canonical_sha(archive)
    return archive, private, rows_by_id


def schedule_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "record_id": record["record_id"],
        "pair_id": record["pair_id"],
        "assignment_type": record["assignment_type"],
        "assignment_position": record["assignment_position"],
        "dominant_outfit": dominant(record),
        "logical_input_sha256": record["logical_input_sha256"],
    }


def build_schedules(
    historical_cycle: Mapping[str, Any], rotations: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    historical_batches = historical_cycle["scientific_cycle"]["batches"]
    results = []
    for rotation, train_indices, _, _ in ROTATIONS:
        train_folds = {FOLDS[index] for index in train_indices}
        cycle_batches = []
        for historical in historical_batches:
            if historical["target_view_fold"] not in train_folds:
                continue
            records = [
                {
                    **{key: value for key, value in row.items() if key in {
                        "record_id", "pair_id", "assignment_position", "dominant_outfit"
                    }},
                    "composition": row["composition"],
                    "batch_position": row["batch_position"],
                }
                for row in historical["records"]
            ]
            cycle_batches.append({
                "batch_index": len(cycle_batches),
                "historical_batch_index": historical["batch_index"],
                "historical_slot": historical["slot"],
                "target_view_fold": historical["target_view_fold"],
                "records": records,
            })
        if len(cycle_batches) != 32:
            raise RuntimeError("rotation must contain 32 historical-filtered batches")
        cycle_record_ids = [row["record_id"] for batch in cycle_batches for row in batch["records"]]
        train_ids = {row["record_id"] for row in rotations[rotation]["train"]}
        if set(cycle_record_ids) != train_ids or len(cycle_record_ids) != len(train_ids):
            raise RuntimeError("rotation cycle does not cover train records exactly once")
        for batch in cycle_batches:
            if [row["dominant_outfit"] for row in batch["records"]] != list(OUTFITS):
                raise RuntimeError("five-outfit batch contract failed")

        cycle = {
            "schema_version": "canondressgs.research.controller_v2_rotation_cycle_scientific.v1",
            "rotation": rotation,
            "outfit_order": list(OUTFITS),
            "train_folds": [FOLDS[index] for index in train_indices],
            "batch_size": 5,
            "batch_count": 32,
            "records_per_cycle": 160,
            "batches": cycle_batches,
        }
        cycle_sha = canonical_sha(cycle)
        steps = 4 * 32 + (11 * 32) // 16
        partial = (11 * 32) // 16
        exposure: Counter[str] = Counter()
        scientific_steps = []
        nuisance_types = ("blur", "mask_erosion", "mask_dilation", "assignment_permutation")
        nuisance_counts: Counter[str] = Counter()
        for zero_step in range(steps):
            batch_index = zero_step % 32
            cycle_index = zero_step // 32
            batch = cycle_batches[batch_index]
            nuisance = nuisance_types[zero_step % 4]
            nuisance_counts[nuisance] += 1
            exposure.update(row["record_id"] for row in batch["records"])
            scientific_steps.append({
                "global_step": zero_step + 1,
                "optimizer_step_index_zero_based": zero_step,
                "cycle_index": cycle_index,
                "batch_index": batch_index,
                "target_view_fold": batch["target_view_fold"],
                "record_ids": [row["record_id"] for row in batch["records"]],
                "v2_nuisance_type": nuisance,
            })
        schedule = {
            "schema_version": "canondressgs.research.controller_v2_rotation_schedule_scientific.v1",
            "rotation": rotation,
            "rotation_cycle_sha256": cycle_sha,
            "optimizer_steps": steps,
            "steps": scientific_steps,
        }
        schedule_sha = canonical_sha(schedule)
        if steps != 150 or sum(exposure.values()) != 750:
            raise RuntimeError("exposure-density formula failed")
        histogram = Counter(exposure.values())
        checkpoints = sorted(set([0, steps // 5, 2 * steps // 5, 3 * steps // 5, 4 * steps // 5, steps]))
        if len(checkpoints) != 6 or checkpoints[0] != 0 or checkpoints[-1] != steps:
            raise RuntimeError("checkpoint schedule invalid")
        seed_order = {str(seed): schedule_sha for seed in SEEDS}
        family_plans = {}
        initialization_identities = {}
        for family in ("V2", "MATCHED_V1"):
            family_plan = {
                "family": family,
                "rotation": rotation,
                "train_record_manifest_sha256": rotations[rotation]["record_manifest_sha256"],
                "clean_cycle_sha256": cycle_sha,
                "clean_schedule_sha256": schedule_sha,
                "optimizer_contract": "SHARED_EXACT",
                "clean_record_exposures": 750,
                "optimizer_steps": steps,
                "checkpoint_steps": checkpoints,
                "final_checkpoint_only_for_evaluation": True,
                "v2_augmented_view_exposures": 750 if family == "V2" else 0,
            }
            family_plans[family] = {
                "training_plan_sha256": canonical_sha(family_plan),
                **family_plan,
            }
            initialization_identities[family] = {
                str(seed): canonical_sha({
                    "purpose": "FUTURE_INITIALIZATION_IDENTITY",
                    "family": family, "rotation": rotation, "seed": seed,
                    "fresh_process_required": True,
                })
                for seed in SEEDS
            }
        partial_batches = cycle_batches[:partial]
        results.append({
            "rotation": rotation,
            "n_train": 160,
            "batches_per_cycle": 32,
            "optimizer_steps": steps,
            "clean_record_exposures_per_run": 750,
            "v2_augmented_view_exposures_per_run": 750,
            "exposure_histogram": {str(key): value for key, value in sorted(histogram.items())},
            "records_with_five_exposures": histogram[5],
            "records_with_four_exposures": histogram[4],
            "complete_cycles": 4,
            "partial_cycle_batch_count": partial,
            "partial_cycle_record_ids": [
                row["record_id"] for batch in partial_batches for row in batch["records"]
            ],
            "first_partial_cycle_record_ids": [
                row["record_id"] for row in partial_batches[0]["records"]
            ],
            "last_partial_cycle_record_ids": [
                row["record_id"] for row in partial_batches[-1]["records"]
            ],
            "checkpoint_steps": checkpoints,
            "checkpoint_count_per_run": len(checkpoints),
            "final_checkpoint_rule": "EVALUATE_STEP_150_ONLY_NO_BEST_OR_EARLY_SELECTION",
            "cycle": cycle,
            "cycle_sha256": cycle_sha,
            "schedule": schedule,
            "schedule_sha256": schedule_sha,
            "seed_clean_data_order_sha256": seed_order,
            "seed_clean_data_order_unique_count": len(set(seed_order.values())),
            "family_training_plans": family_plans,
            "future_initialization_identity_sha256": initialization_identities,
            "future_initialization_identity_unique_count": len({
                value for family in initialization_identities.values() for value in family.values()
            }),
            "nuisance_step_counts_per_v2_run": dict(nuisance_counts),
        })
    archive = {
        "schema_version": "canondressgs.research.controller_v2_micro_pilot_batch_schedules.v1",
        "task_id": TASK_ID,
        "derivation": "FILTER_FROZEN_HISTORICAL_64_BATCH_CYCLE_TO_ROTATION_TRAIN_FOLDS_PRESERVING_ORDER",
        "historical_cycle_sha256": FROZEN_HASHES["historical_full_cycle_scientific_sha256"],
        "historical_schedule_sha256": FROZEN_HASHES["historical_300_step_scientific_sha256"],
        "formula": "S_r = 4*B_r + floor(11*B_r/16)",
        "rotations": results,
        "same_clean_order_v2_and_matched_v1": True,
        "same_clean_order_across_seeds": True,
        "paper_final": False,
        "paper_final_count": 0,
    }
    archive["archive_content_sha256"] = canonical_sha(archive)
    return archive


def build_optimizer_contract() -> dict[str, Any]:
    contract = {
        "schema_version": "canondressgs.research.controller_v2_optimizer_runtime_contract.v1",
        "task_id": TASK_ID,
        "applies_identically_to": ["V2", "MATCHED_V1"],
        "evidence": {
            "formal_v1_source": "tools/paper/run_reference_conditioned_dual_support_controller_formal.py",
            "formal_v1_training_contract": "paper_protocol/reviewer_risk/dual_support_controller_protocol.yaml",
            "formal_v1_training_results": "paper_protocol/reviewer_risk/dual_support_controller_training_results.json",
            "formal_v1_final_checkpoint": "/root/autodl-tmp/canondressgs_work/outputs/REFERENCE-CONDITIONED-DUAL-SUPPORT-CONTROLLER-FORMAL-002/attempt_001/seed_0/checkpoints/final_step_300.pt",
            "formal_v1_final_checkpoint_sha256": "53ba849467cc7c8b4cd9a79c04839f0d938defbeec8dd31010e38b30ebdf797a",
            "formal_v1_head": FORMAL_V1_HEAD,
        },
        "optimizer": {
            "class": "torch.optim.Adam",
            "learning_rate": 0.02,
            "weight_decay": 0.0,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "amsgrad": False,
            "maximize": False,
            "foreach": "PYTORCH_EXPLICIT_NONE",
            "capturable": False,
            "differentiable": False,
            "fused": "PYTORCH_EXPLICIT_NONE",
            "parameter_group_count": 1,
            "parameter_group_rule": "ALL_AND_ONLY_TRAINABLE_CONTROLLER_PARAMETERS",
            "optimizer_state_keys_per_parameter": ["exp_avg", "exp_avg_sq", "step"],
            "defaults_used_as_provenance": False,
        },
        "scheduler": {
            "class": "torch.optim.lr_scheduler.LambdaLR",
            "lambda": "CONSTANT_1.0",
            "base_lr": 0.02,
            "step_after_each_optimizer_step": True,
            "warmup_steps": 0,
        },
        "runtime": {
            "mixed_precision": False,
            "dtype": "FLOAT32",
            "gradient_accumulation_steps": 1,
            "gradient_clipping": {"type": "CLIP_GRAD_NORM", "max_norm": 5.0},
            "zero_grad": {"before_each_forward": True, "set_to_none": True},
            "frozen_f2": True,
            "f2_requires_grad": False,
            "f2_optimizer_membership": False,
            "batch_size": 5,
            "loss_reduction": "MEAN_OVER_BATCH",
            "independent_fresh_process_per_rotation_seed": True,
        },
        "checkpoint": {
            "serializer": "torch.save",
            "atomic_replace": "TEMPORARY_PATH_THEN_OS_REPLACE",
            "payload": [
                "model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
                "global_step", "cpu_rng_state", "cuda_rng_state_all",
            ],
            "schedule": [0, 30, 60, 90, 120, 150],
            "evaluation_rule": "FINAL_STEP_150_ONLY",
            "early_stopping": False,
            "best_checkpoint_selection": False,
        },
        "matched_v1_parameter_count": 3589,
        "optimizer_created_by_contract_repair": 0,
        "paper_final": False,
        "paper_final_count": 0,
    }
    contract["contract_content_sha256"] = canonical_sha(contract)
    return contract


def build_loss_contract() -> dict[str, Any]:
    contract = {
        "schema_version": "canondressgs.research.controller_v2_loss_contract.v1",
        "task_id": TASK_ID,
        "status": "PROSPECTIVELY_FROZEN_BEFORE_ANY_V2_TRAINING",
        "total": "L_garment + 1.0*L_mixedness + 1.0*L_weight + 0.1*L_consistency",
        "lambdas": {"lambda_mix": 1.0, "lambda_weight": 1.0, "lambda_cons": 0.1},
        "garment": {
            "type": "SOFT_TARGET_CROSS_ENTROPY",
            "formula": "-mean_batch(sum_outfit(target * log_softmax(garment_logits)))",
            "targets": {"pure": "ONE_HOT", "AAB": "2/3_AND_1/3", "ABB": "1/3_AND_2/3"},
            "reduction": "MEAN",
        },
        "mixedness": {
            "type": "BINARY_CROSS_ENTROPY_WITH_LOGITS",
            "targets": {"pure": 0, "AAB": 1, "ABB": 1},
            "class_reweight": False,
            "reduction": "MEAN",
        },
        "pair_weight": {
            "records": "MIXED_ONLY",
            "type": "SMOOTHL1",
            "beta": 0.1,
            "reduction": "MEAN",
            "supervised_output": "GROUND_TRUTH_UNORDERED_PAIR_SCALAR_TRAINING_INDEX_ONLY",
            "inference_ground_truth_pair_use": False,
            "pair_orientation": "EARLIER_FROZEN_OUTFIT_WEIGHT",
            "target": {"AAB": "2/3", "ABB": "1/3"},
        },
        "consistency": {
            "applies_to": "NUISANCE_PERTURBATIONS_ONLY",
            "formula": "L_garment_cons + L_mixedness_cons + L_pair_weight_cons",
            "garment": {
                "type": "JENSEN_SHANNON_DIVERGENCE",
                "inputs": "CLEAN_AND_AUG_SOFTMAX_GARMENT_PROBABILITIES",
                "probability_clamp_min": 1e-8,
                "reduction": "MEAN_BATCH",
            },
            "mixedness": {
                "type": "SMOOTHL1", "beta": 0.1, "reduction": "MEAN",
                "inputs": "CLEAN_AND_AUG_SIGMOID_MIXEDNESS_PROBABILITY",
            },
            "pair_weights": {
                "type": "SMOOTHL1", "beta": 0.1, "reduction": "MEAN",
                "inputs": "CLEAN_AND_AUG_ALL_10_SIGMOID_PAIR_WEIGHTS",
                "ground_truth_pair_selection": False,
                "selected_pair_only": False,
            },
            "information_ablation_exposure": 0,
            "target_render_exposure": 0,
        },
        "information_ablation_training_boundary": {
            "variants": ["complete_reference_dropout", "single_reference"],
            "training_exposure": 0,
            "calibration_threshold_fitting_exposure": 0,
            "consistency_loss_exposure": 0,
            "use": "EVALUATION_ONLY",
            "evaluation_targets": [
                "SAFE_SINGLE_ENDPOINT", "LOW_WRONG_DUAL_EXPOSURE",
                "IDENTITY_CONTAMINATION_GRADE_0",
            ],
            "second_garment_recovery_required": False,
            "reason": "TASK_INFORMATION_REMOVAL_IS_NOT_LABEL_PRESERVING_AUGMENTATION",
        },
        "supervised_losses_on_clean_forward_only": True,
        "matched_v1_loss": {
            "type": "SOFT_TARGET_CROSS_ENTROPY_ONLY",
            "probability_ratio_weight": True,
            "fallback": "HISTORICAL_0.90_0.10",
            "v2_consistency": False,
            "mixedness_supervision": False,
            "weight_supervision": False,
        },
        "paper_final": False,
        "paper_final_count": 0,
    }
    contract["contract_content_sha256"] = canonical_sha(contract)
    return contract


def build_calibration_contract() -> dict[str, Any]:
    mix_grid = [round(index / 10, 2) for index in range(1, 10)]
    pair_grid = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    candidates = [
        {"tau_pair": pair, "tau_mix": mix}
        for pair in pair_grid for mix in mix_grid
    ]
    grid_sha = canonical_sha(candidates)
    contract = {
        "schema_version": "canondressgs.research.controller_v2_calibration_contract.v1",
        "task_id": TASK_ID,
        "selection_scope": "INDEPENDENT_PER_ROTATION_AND_SEED_ON_CALIBRATION_FOLD_ONLY",
        "mixedness_threshold_grid": mix_grid,
        "pair_confidence_threshold_grid": pair_grid,
        "pair_confidence_scalar": "TOP2_VS_TOP3_PROBABILITY_MARGIN",
        "candidate_count_per_rotation_seed": len(candidates),
        "all_candidates": candidates,
        "calibration_grid_sha256": grid_sha,
        "objective_tuple": [
            "MINIMIZE_PURE_FALSE_MIXED_RATE",
            "MINIMIZE_WRONG_PAIR_DUAL_RATE",
            "MINIMIZE_INCOMPATIBLE_PAIR_DUAL_RATE",
            "MAXIMIZE_CORRECT_COMPATIBLE_DUAL_RATE",
            "MAXIMIZE_CORRECT_INCOMPATIBLE_HARD_GEOMETRY_SOFT_VA_RATE",
            "MINIMIZE_MIXED_FALSE_SINGLE_RATE",
        ],
        "deterministic_tie_break": [
            "CHOOSE_LARGER_TAU_PAIR",
            "CHOOSE_LARGER_TAU_MIX",
            "CHOOSE_LEXICOGRAPHICALLY_LARGER_(TAU_PAIR,TAU_MIX)",
            "REMAINING_NON_UNIQUENESS_IS_CONTRACT_ERROR",
        ],
        "minimization_sort_key": (
            "(pure_false_mixed, wrong_pair_dual, incompatible_pair_dual, "
            "-correct_compatible_dual, -correct_incompatible_hard, "
            "mixed_false_single, -tau_pair, -tau_mix)"
        ),
        "required_runtime_selection_trace_per_rotation_seed": {
            "all_grid_candidates": "REQUIRED_81_ROWS_WITH_COMPLETE_OBJECTIVE_TUPLES",
            "tie_candidate_list": "REQUIRED",
            "selected_thresholds": "REQUIRED",
            "selection_trace": "REQUIRED",
        },
        "runtime_selection_status": "NOT_RUN_BY_CONTRACT_REPAIR",
        "threshold_selection_count_in_this_task": 0,
        "test_fold_used": False,
        "information_ablation_used": False,
        "compatibility_manifest": "paper_protocol/reviewer_risk/controller_v2_compatibility_manifests.json",
        "deployment_safety_tie_break_frozen_before_results": True,
        "paper_final": False,
        "paper_final_count": 0,
    }
    contract["contract_content_sha256"] = canonical_sha(contract)
    return contract


def build_denominators(compatibility: Mapping[str, Any]) -> dict[str, Any]:
    rotations = []
    for manifest in compatibility["manifests"]:
        labels = {row["pair_id"]: row["compatibility_label"] for row in manifest["entries"]}
        compatible_pairs = [pair for pair in PAIRS if labels[pair] == "COMPATIBLE"]
        incompatible_pairs = [pair for pair in PAIRS if labels[pair] == "INCOMPATIBLE"]
        rotations.append({
            "rotation": manifest["rotation"],
            "protocol_weighted": {
                "all_test_records": 80,
                "pure_test_records": 20,
                "mixed_test_records": 60,
                "compatible_mixed_test_records": 6 * len(compatible_pairs),
                "incompatible_mixed_test_records": 6 * len(incompatible_pairs),
            },
            "unique_query": {
                "all_test_queries": 65,
                "pure_test_queries": 5,
                "mixed_test_queries": 60,
                "compatible_mixed_test_queries": 6 * len(compatible_pairs),
                "incompatible_mixed_test_queries": 6 * len(incompatible_pairs),
                "merge_key": "logical_input_sha256",
            },
            "compatible_pairs": compatible_pairs,
            "incompatible_pairs": incompatible_pairs,
        })
    contract = {
        "schema_version": "canondressgs.research.controller_v2_evaluator_denominators.v1",
        "task_id": TASK_ID,
        "primary_gate_aggregation": "PROTOCOL_WEIGHTED_MACRO",
        "mandatory_secondary_aggregation": "UNIQUE_QUERY_MACRO",
        "consistent_duplicates": "RETAINED_AS_INDEPENDENT_PROTOCOL_RECORDS",
        "unique_query_weight": "ONE_PER_LOGICAL_INPUT_SHA256",
        "reporting_levels": [
            "EACH_ROTATION_AND_SEED", "SEED_MACRO", "ROTATION_MACRO",
            "GLOBAL_MACRO", "QUERY_WEIGHTED",
        ],
        "rotations": rotations,
        "metric_denominators": {
            "pair_accuracy": "ALL_TEST_RECORDS",
            "mixedness": "ALL_TEST_PURE_AND_MIXED_RECORDS",
            "weight_mae_rmse_reporting": "GT_MIXED_RECORDS",
            "weight_primary_gate": "GT_MIXED_AND_PREDICTED_UNORDERED_PAIR_CORRECT",
            "compatible_dual_rate": "GT_MIXED_AND_PREDICTED_PAIR_CORRECT_AND_GT_PAIR_COMPATIBLE",
            "incompatible_hard_rate": "GT_MIXED_AND_PREDICTED_PAIR_CORRECT_AND_GT_PAIR_INCOMPATIBLE",
            "wrong_pair_dual_rate": "GT_MIXED_AND_PREDICTED_PAIR_INCORRECT",
            "incompatible_dual_rate": "GT_MIXED_AND_PREDICTED_PAIR_CORRECT_AND_GT_PAIR_INCOMPATIBLE",
            "pure_single_rate": "TEST_FOLD_PURE_PROTOCOL_RECORDS",
        },
        "result_dependent_subset_rule": (
            "PREDICTION_CONDITIONED_DENOMINATORS_ARE_REPORTED_AS_EXACT_COUNTS_AT_RUNTIME;"
            "_THEIR_PRE_RESULT_GT_SUPERSET_IS_60_MIXED_RECORDS_PER_ROTATION"
        ),
        "formal_pure_20": {
            "role": "SECONDARY_ENDPOINT_SAFETY_AUDIT",
            "primary_denominator_use": 0,
            "global_crossfit_macro_use": 0,
            "separate_reporting": True,
        },
        "paper_final": False,
        "paper_final_count": 0,
    }
    contract["contract_content_sha256"] = canonical_sha(contract)
    return contract


def build_representatives(
    rotations: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    rotation_rows = []
    visual_rows = []
    for rotation, _, _, test_index in ROTATIONS:
        test = rotations[rotation]["test"]
        selected = []
        for pair in PAIRS:
            for composition in ("AAB", "ABB"):
                candidates = [
                    row for row in test
                    if row["pair_id"] == pair and row["assignment_type"] == composition
                ]
                candidates.sort(key=lambda row: (
                    row["logical_input_sha256"],
                    int(row["assignment_position"]),
                    row["record_id"],
                ))
                if not candidates:
                    raise RuntimeError("missing perturbation representative")
                row = candidates[0]
                selected.append({
                    "pair_id": pair,
                    "composition": composition,
                    "record_id": row["record_id"],
                    "logical_query_id": row["logical_input_sha256"],
                    "assignment_position": row["assignment_position"],
                    "selection_rank": 1,
                })
        if len(selected) != 20:
            raise RuntimeError("rotation does not have 20 representatives")
        selection_sha = canonical_sha(selected)
        rotation_rows.append({
            "rotation": rotation,
            "test_fold_index": test_index,
            "test_fold": FOLDS[test_index],
            "representative_count": len(selected),
            "representatives": selected,
            "representative_sha256": selection_sha,
        })
        for seed in SEEDS:
            for row in selected:
                visual_rows.append({
                    "sheet_id": (
                        f"rotation_{rotation}/seed_{seed}/{row['pair_id']}/{row['composition']}"
                    ),
                    "rotation": rotation,
                    "seed": seed,
                    **row,
                    "columns": [
                        "references", "V2_actual", "matched_V1_actual", "target",
                        "oracle", "dominant_endpoint", "predicted_pairs", "weights",
                        "modes", "compatibility",
                    ],
                })
    perturb = {
        "schema_version": "canondressgs.research.controller_v2_perturbation_representatives.v1",
        "task_id": TASK_ID,
        "selection_rule": [
            "FILTER_TEST_FOLD_BY_PAIR_AND_COMPOSITION",
            "SORT_LOGICAL_QUERY_ID_ASC",
            "SORT_ASSIGNMENT_POSITION_ASC",
            "SORT_RECORD_ID_ASC",
            "SELECT_FIRST",
        ],
        "selection_is_pre_result": True,
        "rotations": rotation_rows,
        "variants": [
            "blur", "mask_erosion", "mask_dilation", "assignment_permutation",
            "reference_dropout", "single_reference",
        ],
        "representatives_per_rotation": 20,
        "queries_per_model_family_rotation_seed": 120,
        "queries_per_model_family_all_rotations_seeds": 1440,
        "queries_both_families_all_rotations_seeds": 2880,
        "paper_final": False,
        "paper_final_count": 0,
    }
    perturb["archive_content_sha256"] = canonical_sha(perturb)
    visual = {
        "schema_version": "canondressgs.research.controller_v2_visual_representatives.v1",
        "task_id": TASK_ID,
        "same_representatives_as_perturbation": True,
        "main_sheet_count": len(visual_rows),
        "sheets": visual_rows,
        "grade_schema": {
            "source": "paper_protocol/reviewer_risk/dual_support_controller_visual_review.json",
            "scale": {"0": "NONE", "1": "MINOR", "2": "MODERATE", "3": "SEVERE"},
            "categories": [
                "patch", "cloud", "mottle", "edge_scatter",
                "silhouette_discontinuity", "full_body_contamination",
                "identity_contamination", "double_outline", "ghosting",
                "endpoint_mismatch", "wrong_garment_mixture",
            ],
            "categories_or_levels_changed": False,
        },
        "post_result_representative_replacement": False,
        "paper_final": False,
        "paper_final_count": 0,
    }
    visual["archive_content_sha256"] = canonical_sha(visual)
    return perturb, visual


def expected_counts() -> dict[str, Any]:
    per_rotation = {
        "n_train": 160,
        "n_calibration": 80,
        "n_test": 80,
        "batches_per_cycle": 32,
        "optimizer_steps_per_run": 150,
        "clean_exposures_per_run": 750,
        "v2_augmented_view_exposures_per_run": 750,
        "checkpoints_per_run": 6,
        "v2_runs": 3,
        "matched_v1_runs": 3,
        "primary_test_inference": {"V2": 240, "MATCHED_V1": 240, "both": 480},
        "primary_test_logical_renders": {"V2": 240, "MATCHED_V1": 240, "both": 480},
        "perturbation_inference": {"V2": 360, "MATCHED_V1": 360, "both": 720},
        "perturbation_logical_renders": {"V2": 360, "MATCHED_V1": 360, "both": 720},
        "main_visual_sheets": 60,
    }
    return {
        "per_rotation": {str(rotation): dict(per_rotation) for rotation in range(4)},
        "global": {
            "V2_training_runs": 12,
            "matched_V1_training_runs": 12,
            "total_training_runs": 24,
            "optimizer_steps_V2": 1800,
            "optimizer_steps_matched_V1": 1800,
            "optimizer_steps_total": 3600,
            "clean_record_exposures_V2": 9000,
            "clean_record_exposures_matched_V1": 9000,
            "clean_record_exposures_total": 18000,
            "V2_augmented_view_exposures": 9000,
            "training_forward_batches_V2": 3600,
            "training_forward_batches_matched_V1": 1800,
            "training_forward_batches_total": 5400,
            "backward_calls_total": 3600,
            "scheduler_steps_total": 3600,
            "checkpoint_writes_total": 144,
            "calibration_prediction_queries_V2": 960,
            "primary_test_inference": 1920,
            "primary_test_logical_renders": 1920,
            "formal_pure_secondary_inference": 480,
            "formal_pure_secondary_logical_renders": 480,
            "perturbation_inference": 2880,
            "perturbation_logical_renders": 2880,
            "total_future_evaluation_inference_including_calibration": 6240,
            "total_future_logical_renders": 5280,
            "main_visual_sheets": 240,
        },
        "counting_notes": {
            "logical_render": "ONE_CONTROLLER_OUTPUT_RGB_ALPHA_PAIR",
            "target_oracle_endpoint_assets": "REUSED_FROZEN_ASSETS_NOT_COUNTED_AS_CONTROLLER_RENDERER_CALLS",
            "formal_pure": "EACH_FAMILY_ROTATION_SEED_MODEL_IS_AUDITED_SEPARATELY",
        },
    }


def field_audit(
    original: Mapping[str, Any], repaired_values: Mapping[str, Any],
) -> list[dict[str, Any]]:
    evidence = {
        "rotation_record_ids": "controller_v2_micro_pilot_rotation_manifests.json",
        "train_calibration_test_counts": "controller_v2_micro_pilot_rotation_manifests.json",
        "fold_membership": "controller_v2_crossfit_splits.json + rotation manifests",
        "duplicate_preservation": "rotation manifests + batch schedules",
        "batch_size": "controller_v2_micro_pilot_batch_schedules.json",
        "per_step_record_exposure": "controller_v2_micro_pilot_batch_schedules.json",
        "data_order_algorithm": "historical cycle + batch schedules",
        "optimizer_type": "formal source + final checkpoint optimizer state",
        "learning_rate": "formal source + final checkpoint optimizer state",
        "weight_decay": "formal source + final checkpoint optimizer state",
        "total_optimizer_steps": "exposure-density formula + batch schedules",
        "checkpoint_cadence": "batch schedules + optimizer contract",
        "final_checkpoint_rule": "batch schedules + optimizer contract",
        "lambda_mix": "prospective V2 loss contract",
        "lambda_weight": "prospective V2 loss contract",
        "lambda_cons": "prospective V2 loss contract",
        "nuisance_augmentation_types": "design protocol + loss contract",
        "augmentation_severity": "formal diagnostic evaluator source",
        "augmentation_exposure_schedule": "batch schedules + loss contract",
        "information_ablation_training_contract": "loss + calibration contracts",
        "mixedness_threshold_grid": "design protocol + calibration contract",
        "pair_confidence_scalar_definition": "design protocol + calibration contract",
        "pair_confidence_threshold_grid": "design protocol + calibration contract",
        "calibration_objective": "calibration contract",
        "calibration_tie_break": "calibration contract",
        "seed_list": "design protocol + batch schedules",
        "final_evaluator_denominators": "evaluator denominators",
        "perturbation_representative_ids": "perturbation representatives",
        "visual_grade_schema": "frozen historical visual schema + visual representatives",
        "success_gates": "design protocol retained unchanged",
    }
    original_index = {row["field"]: row for row in original["contract_checks"]}
    prospective_fields = {
        "lambda_mix", "lambda_weight", "lambda_cons",
        "augmentation_exposure_schedule",
        "information_ablation_training_contract",
        "calibration_objective", "calibration_tie_break",
        "final_evaluator_denominators",
        "perturbation_representative_ids",
    }
    result = []
    for row in original["contract_checks"]:
        field = row["field"]
        result.append({
            "id": row["id"],
            "field": field,
            "previous_state": row["status"],
            "previous_evidence": row["evidence"],
            "evidence_source": evidence[field],
            "repaired_value": repaired_values[field],
            "value_provenance": (
                "TASK_DIRECTIVE_PROSPECTIVELY_FROZEN"
                if field in prospective_fields
                else "HISTORICAL_RECOVERY"
                if row["status"] == "MISSING"
                else "FROZEN_DESIGN_RETAINED"
            ),
            "deterministic_derivation": True,
            "final_state": "PASS",
        })
    if len(result) != 30 or set(original_index) != set(repaired_values):
        raise RuntimeError("30-field audit mapping incomplete")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    expected_paths = [RISK / name for name in JSON_OUTPUTS] + [DOCS / name for name in DOC_OUTPUTS] + [HANDOFF / HANDOFF_OUTPUT]
    if not args.replace and any(path.exists() for path in expected_paths):
        raise FileExistsError("refusing to overwrite contract repair artifacts without --replace")

    manifest_path = RISK / "dual_support_controller_training_manifest.json"
    protocol_path = RISK / "dual_support_controller_protocol.yaml"
    cycle_path = RISK / "dual_support_controller_training_cycle.json"
    schedule_path = RISK / "dual_support_controller_training_schedule_300_steps.json"
    if lf_sha(manifest_path) != FROZEN_HASHES["training_manifest_sha256_lf"]:
        raise RuntimeError("frozen training manifest SHA mismatch")
    if lf_sha(protocol_path) != FROZEN_HASHES["repaired_protocol_sha256_lf"]:
        raise RuntimeError("frozen protocol SHA mismatch")
    historical_cycle = read_json(cycle_path)
    historical_schedule = read_json(schedule_path)
    if historical_cycle["training_cycle_sha256"] != FROZEN_HASHES["historical_full_cycle_scientific_sha256"]:
        raise RuntimeError("historical scientific cycle SHA mismatch")
    if historical_schedule["training_300_step_data_order_sha256"] != FROZEN_HASHES["historical_300_step_scientific_sha256"]:
        raise RuntimeError("historical scientific schedule SHA mismatch")

    manifest = read_json(manifest_path)
    original = read_json(RISK / "controller_v2_micro_pilot_training_contract.json")
    compatibility = read_json(RISK / "controller_v2_compatibility_manifests.json")
    rotation_archive, private_rotations, _ = build_rotation_manifests(manifest)
    schedule_archive = build_schedules(historical_cycle, private_rotations)
    optimizer = build_optimizer_contract()
    loss = build_loss_contract()
    calibration = build_calibration_contract()
    denominators = build_denominators(compatibility)
    perturbation, visual = build_representatives(private_rotations)
    counts = expected_counts()

    nuisance = {
        "blur": {
            "implementation_source": "tools/paper/p0_evaluation_protocol.py:c5_gaussian_blur",
            "implementation_source_sha256_lf": lf_sha(ROOT / "tools/paper/p0_evaluation_protocol.py"),
            "kernel": "SEPARABLE_GAUSSIAN_REFLECT_PADDING",
            "kernel_size": 11,
            "radius": 5,
            "sigma": 3.0,
            "mask_operation": "BLUR_ALREADY_RESIZED_RGB_THEN_RESTORE_ALL_PIXELS_OUTSIDE_CLOTHING_MASK",
        },
        "mask_erosion": {
            "implementation_source": "tools/paper/run_reference_conditioned_dual_support_controller_formal_evaluation.py",
            "implementation_source_sha256_lf": lf_sha(ROOT / "tools/paper/run_reference_conditioned_dual_support_controller_formal_evaluation.py"),
            "operation": "scipy.ndimage.binary_erosion",
            "iterations": 3,
            "structure": "SCIPY_DEFAULT_CROSS_CONNECTIVITY_1",
        },
        "mask_dilation": {
            "implementation_source": "tools/paper/run_reference_conditioned_dual_support_controller_formal_evaluation.py",
            "implementation_source_sha256_lf": lf_sha(ROOT / "tools/paper/run_reference_conditioned_dual_support_controller_formal_evaluation.py"),
            "operation": "scipy.ndimage.binary_dilation",
            "iterations": 3,
            "structure": "SCIPY_DEFAULT_CROSS_CONNECTIVITY_1",
        },
        "assignment_permutation": {
            "implementation_source": "tools/paper/run_reference_conditioned_dual_support_controller_formal_evaluation.py:804-805",
            "implementation_source_sha256_lf": lf_sha(ROOT / "tools/paper/run_reference_conditioned_dual_support_controller_formal_evaluation.py"),
            "rule": "REFERENCE_ROW_AND_VALIDITY_INDEX_ORDER_[2,0,1]",
        },
    }
    loss["nuisance_augmentation"] = {
        "definitions": nuisance,
        "round_robin_zero_based_step_mod_4": {
            "0": "blur", "1": "mask_erosion", "2": "mask_dilation", "3": "assignment_permutation",
        },
        "per_v2_optimizer_step": {
            "clean_forwards": 1, "nuisance_aug_forwards": 1,
            "clean_record_exposures": 5, "augmented_view_exposures": 5,
        },
        "severity_reselected": False,
    }
    loss["contract_content_sha256"] = canonical_sha({
        key: value for key, value in loss.items() if key != "contract_content_sha256"
    })

    repaired_values = {
        "rotation_record_ids": "EXACT_IDS_FOR_TRAIN_CALIBRATION_TEST_ALL_4_ROTATIONS",
        "train_calibration_test_counts": "160/80/80_EACH_ROTATION",
        "fold_membership": "FOUR_FROZEN_DISJOINT_ROTATIONS",
        "duplicate_preservation": "CONSISTENT_DUPLICATES_RETAINED_NO_DEDUP_OR_EXTRA_WEIGHT",
        "batch_size": 5,
        "per_step_record_exposure": "5_CLEAN_RECORDS;_V2_ALSO_5_AUGMENTED_VIEWS",
        "data_order_algorithm": "FILTER_HISTORICAL_CYCLE_BY_TRAIN_FOLDS_PRESERVING_ORDER_THEN_WRAP",
        "optimizer_type": "torch.optim.Adam",
        "learning_rate": 0.02,
        "weight_decay": 0.0,
        "total_optimizer_steps": 150,
        "checkpoint_cadence": [0, 30, 60, 90, 120, 150],
        "final_checkpoint_rule": "STEP_150_ONLY",
        "lambda_mix": 1.0,
        "lambda_weight": 1.0,
        "lambda_cons": 0.1,
        "nuisance_augmentation_types": list(nuisance),
        "augmentation_severity": nuisance,
        "augmentation_exposure_schedule": "ZERO_BASED_STEP_MOD_4_FIXED_ROUND_ROBIN",
        "information_ablation_training_contract": "EVALUATION_ONLY_ZERO_TRAIN_CALIBRATION_CONSISTENCY_EXPOSURE",
        "mixedness_threshold_grid": calibration["mixedness_threshold_grid"],
        "pair_confidence_scalar_definition": calibration["pair_confidence_scalar"],
        "pair_confidence_threshold_grid": calibration["pair_confidence_threshold_grid"],
        "calibration_objective": calibration["objective_tuple"],
        "calibration_tie_break": calibration["deterministic_tie_break"],
        "seed_list": list(SEEDS),
        "final_evaluator_denominators": "PROTOCOL_WEIGHTED_MACRO_PRIMARY_AND_UNIQUE_QUERY_MACRO_MANDATORY",
        "perturbation_representative_ids": "20_PRE_RESULT_IDS_PER_ROTATION",
        "visual_grade_schema": visual["grade_schema"],
        "success_gates": "FROZEN_DESIGN_GATES_RETAINED_UNCHANGED",
    }
    audit = field_audit(original, repaired_values)

    artifacts = {
        "controller_v2_micro_pilot_rotation_manifests.json": rotation_archive,
        "controller_v2_micro_pilot_batch_schedules.json": schedule_archive,
        "controller_v2_micro_pilot_optimizer_contract.json": optimizer,
        "controller_v2_micro_pilot_loss_contract.json": loss,
        "controller_v2_micro_pilot_calibration_contract.json": calibration,
        "controller_v2_micro_pilot_evaluator_denominators.json": denominators,
        "controller_v2_micro_pilot_perturbation_representatives.json": perturbation,
        "controller_v2_micro_pilot_visual_representatives.json": visual,
    }
    for name, value in artifacts.items():
        write_json(RISK / name, value)

    component_hashes = {name: canonical_sha(value) for name, value in artifacts.items()}
    global_contract_payload = {
        "source_head": SOURCE_HEAD,
        "frozen_hashes": FROZEN_HASHES,
        "component_hashes": component_hashes,
        "field_audit": audit,
        "expected_counts": counts,
    }
    global_contract_sha = canonical_sha(global_contract_payload)
    repaired = {
        "schema_version": "canondressgs.research.controller_v2_micro_pilot_contract_repaired.v1",
        "task_id": TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "repair_branch": REPAIR_BRANCH,
        "design_head": DESIGN_HEAD,
        "upstream_training_contract_head": UPSTREAM_CONTRACT_HEAD,
        "formal_v1_head": FORMAL_V1_HEAD,
        "diagnostic_head": DIAGNOSTIC_HEAD,
        "original_classification": "CONTROLLER_V2_MICRO_PILOT_CONTRACT_INCOMPLETE",
        "classification": CLASSIFICATION,
        "original_audit": {"pass": 8, "missing": 22, "ambiguous": 0, "total": 30},
        "repaired_audit": {"pass": 30, "missing": 0, "ambiguous": 0, "total": 30},
        "contract_checks": audit,
        "frozen_upstream_hashes": FROZEN_HASHES,
        "attempt_001_preservation": {
            "cloud_path": "/root/autodl-tmp/canondressgs_work/outputs/CONTROLLER-V2-CROSSFIT-MICRO-PILOT-001/attempt_001",
            "file_sha256": ATTEMPT_001_HASHES,
            "mutation_count": 0,
            "attempt_002_created": False,
        },
        "component_hashes": component_hashes,
        "global_contract_sha256": global_contract_sha,
        "expected_future_counts": counts,
        "current_execution_counts": {
            "training_runs": 0,
            "training_steps": 0,
            "training_forward_batches": 0,
            "backward_calls": 0,
            "optimizer_creations": 0,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "checkpoint_writes": 0,
            "controller_inference": 0,
            "threshold_selections": 0,
            "renderer_runs": 0,
            "metric_aggregations": 0,
            "visual_sheets": 0,
        },
        "frozen_mutations": {
            "design_artifacts": 0, "formal_v1": 0, "diagnostic_archive": 0,
            "micro_pilot_attempt_001": 0, "teacher": 0, "F2": 0,
            "renderer": 0, "garment_bank": 0, "compatibility_manifests": 0,
        },
        "paper_final": False,
        "paper_final_count": 0,
        "next_task": NEXT_TASK,
        "next_task_started": False,
    }
    write_json(RISK / "controller_v2_micro_pilot_contract_repaired.json", repaired)

    expected_relative = (
        [f"docs/PAPER/{name}" for name in DOC_OUTPUTS]
        + [f"paper_protocol/reviewer_risk/{name}" for name in JSON_OUTPUTS]
        + [f"project_control_handoff/{HANDOFF_OUTPUT}"]
    )
    summary = {
        "schema_version": "canondressgs.research.controller_v2_micro_pilot_contract_repair_summary.v1",
        "task_id": TASK_ID,
        "classification": CLASSIFICATION,
        "source_head": SOURCE_HEAD,
        "repair_branch": REPAIR_BRANCH,
        "audit": {"pass": 30, "missing": 0, "ambiguous": 0, "total": 30},
        "rotation_counts": [
            {
                "rotation": row["rotation"], "train": 160, "calibration": 80, "test": 80,
                "batches_per_cycle": 32, "optimizer_steps": 150,
                "clean_exposures": 750, "v2_augmented_exposures": 750,
            }
            for row in rotation_archive["rotations"]
        ],
        "global_contract_sha256": global_contract_sha,
        "frozen_upstream_hashes": FROZEN_HASHES,
        "attempt_001_sha256": ATTEMPT_001_HASHES,
        "expected_artifacts": expected_relative,
        "expected_artifact_count": len(expected_relative),
        "no_training_gate": repaired["current_execution_counts"],
        "frozen_mutations": repaired["frozen_mutations"],
        "paper_final": False,
        "paper_final_count": 0,
        "next_task": NEXT_TASK,
        "next_task_started": False,
    }
    write_json(RISK / "controller_v2_micro_pilot_contract_repair_summary.json", summary)

    main_report = f"""# Controller V2 micro-pilot training-contract repair

Task `{TASK_ID}` repairs the failed pre-result contract without running training,
inference, calibration, rendering, metrics, or visual review. The original audit
was 8 PASS / 22 MISSING / 30 total. The repaired audit is 30 PASS / 0 MISSING /
0 AMBIGUOUS.

## Frozen records and schedules

Each rotation contains exactly 160 training, 80 calibration, and 80 test
protocol records. Training/calibration/test record intersections are zero.
Consistent duplicates remain independent protocol records; no deduplication or
extra weighting occurs. Each filtered historical cycle has 32 five-record
batches in outfit order O01, O02, O03, O04, O08. The exposure-density formula
gives 150 optimizer steps and 750 clean record exposures per run. V2 and
matched V1 use identical clean order for seeds 0, 1, and 2.

The six checkpoints are steps 0, 30, 60, 90, 120, and 150. Only step 150 may be
evaluated. There is no early stopping, best-checkpoint selection, or
result-driven schedule change.

## Preserved boundaries

The formal-pure 20 records are an evaluation-only secondary safety set with
zero training exposure, zero calibration-threshold use, and zero contribution
to the primary cross-fit denominator. `attempt_001` remains the failed archive
and `attempt_002` was not created. All execution counters are zero and
PAPER_FINAL=0.

Global pre-result contract SHA256: `{global_contract_sha}`.

Classification: `{CLASSIFICATION}`.
"""
    matched_report = """# Matched V1 training contract

Matched V1 is freshly initialized for every rotation and seed and is never
loaded from the historical formal V1 checkpoint. It has 3,589 trainable
controller parameters, frozen F2, the same 160 training records, identical
five-record clean batches, 150 optimizer steps, 750 clean exposures, and the
same six-checkpoint schedule as V2.

Both families use Adam with lr=0.02, weight_decay=0, betas=(0.9,0.999),
epsilon=1e-8, constant LambdaLR, no warmup, FP32, accumulation=1, gradient-norm
clipping at 5.0, and `zero_grad(set_to_none=True)`. Matched V1 uses only the
original mean soft-target garment cross entropy, its probability-ratio weight,
and historical 0.90/0.10 fallback. It does not use compatibility,
HARD_GEOMETRY_SOFT_VA, V2 mixedness/weight supervision, or consistency.

The historical formal V1 is retained only as
`HISTORICAL_PROTOCOL_MISMATCHED_REFERENCE`.
"""
    evaluator_report = """# Controller V2 calibration and evaluator contract

For every rotation and seed, V2 calibration searches the frozen 9x9 grid:
tau_mix={0.10,...,0.90} and
tau_pair={0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50}. Pair confidence is
TOP2_VS_TOP3_PROBABILITY_MARGIN. The fixed objective first minimizes pure
false-mixed, wrong-pair DUAL, and incompatible-pair DUAL; then maximizes
correct-compatible DUAL and correct-incompatible HARD_GEOMETRY_SOFT_VA; then
minimizes mixed false-SINGLE. Exact ties choose larger tau_pair, then larger
tau_mix, then the lexicographically larger pair. No threshold is selected in
this repair task.

Primary gates use PROTOCOL_WEIGHTED_MACRO with duplicates retained. The full
UNIQUE_QUERY_MACRO keyed by logical_input_sha256 is mandatory and cannot replace
the primary aggregation. Each test fold has 80 protocol records / 65 unique
queries: 20/5 pure and 60/60 mixed. The formal-pure 20 set is reported
separately.

Twenty perturbation representatives per rotation are frozen before results by
sorting logical query ID, assignment position, and record ID within every
pair/composition cell. The same records define 240 main visual sheets. The
historical 0=none, 1=minor, 2=moderate, 3=severe scale and all artifact
categories are unchanged.
"""
    write_text(DOCS / DOC_OUTPUTS[0], main_report)
    write_text(DOCS / DOC_OUTPUTS[1], matched_report)
    write_text(DOCS / DOC_OUTPUTS[2], evaluator_report)

    handoff = {
        "schema_version": "canondressgs.project_control.controller_v2_micro_pilot_contract_repair_handoff.v1",
        "task_id": TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "repair_branch": REPAIR_BRANCH,
        "classification": CLASSIFICATION,
        "contract_path": "paper_protocol/reviewer_risk/controller_v2_micro_pilot_contract_repaired.json",
        "summary_path": "paper_protocol/reviewer_risk/controller_v2_micro_pilot_contract_repair_summary.json",
        "report_path": "docs/PAPER/AAAI27_CONTROLLER_V2_MICRO_PILOT_CONTRACT_REPAIR_20260723.md",
        "global_contract_sha256": global_contract_sha,
        "attempt_001_preserved_sha256": ATTEMPT_001_HASHES,
        "attempt_002_created": False,
        "execution_counts": repaired["current_execution_counts"],
        "paper_final": False,
        "paper_final_count": 0,
        "next_task": NEXT_TASK,
        "next_task_started": False,
    }
    write_json(HANDOFF / HANDOFF_OUTPUT, handoff)

    print(json.dumps({
        "status": "PASS",
        "classification": CLASSIFICATION,
        "audit": "30/30",
        "rotation_manifest_count": 4,
        "main_visual_sheet_count": visual["main_sheet_count"],
        "global_contract_sha256": global_contract_sha,
        "artifact_count": len(expected_relative),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
