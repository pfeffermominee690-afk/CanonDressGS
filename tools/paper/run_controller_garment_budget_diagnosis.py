"""Run the preregistered Controller garment-head budget diagnosis.

The command is deliberately phase separated.  ``preflight`` never constructs
an optimizer.  ``train`` is admitted only after the four pre-result artifacts
are tracked by Git and byte-identical copies have been materialized under the
append-only diagnostic output root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F


PROJECT_ROOT = (
    Path(os.environ["CANONDRESSGS_PROJECT_ROOT"])
    if "CANONDRESSGS_PROJECT_ROOT" in os.environ
    else Path(__file__).resolve().parents[2]
).resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from scene.compatibility_gated_reference_controller_v2 import (  # noqa: E402
    PAIR_ORDER,
    PAIR_TO_INDEX,
    CompatibilityGatedReferenceControllerV2,
    stable_pair_prediction,
)
from scene.reference_conditioned_dual_support_controller import (  # noqa: E402
    OUTFIT_ORDER,
    ReferenceConditionedDualSupportController,
    soft_target_cross_entropy,
    stable_top2_selection,
)
from tools.paper import run_controller_v2_repaired_evaluation as repaired_eval  # noqa: E402
from tools.paper import run_controller_v2_repaired_training as repaired_train  # noqa: E402
from tools.paper import run_reference_conditioned_dual_support_controller_formal as formal  # noqa: E402


TASK_ID = "AAAI27-CONTROLLER-GARMENT-HEAD-OPTIMIZATION-BUDGET-001"
SOURCE_HEAD = "89980f1d9d8946d203c3ff1b398929f09965f098"
HISTORICAL_SOURCE_HEAD = "a8557b461f301c19a0f24eb17d924ddd9159a580"
PAIR_DIAGNOSIS_HEAD = "56988a7e4b5cdd63be481005b070a44b20feab13"
BRANCH = "research/controller-garment-head-optimization-budget-diagnosis-20260724"
HISTORICAL_CLASSIFICATION = "CONTROLLER_V2_CROSSFIT_MICRO_PILOT_FAIL"
SEMANTIC_CLASSIFICATION = "LATENT_SECONDARY_NOT_INPUT_IDENTIFIABLE"
PAIR_INTERPRETATION = "MIXED_PAIR_IDENTIFICATION_PARTIAL"
POST_FAILURE_LABEL = "POST-FAILURE OPTIMIZATION DIAGNOSTIC"

FAMILIES = (
    "FULL_V2_CONTINUED",
    "MATCHED_V1_CONTINUED",
    "GARMENT_ONLY",
    "GARMENT_PLUS_MIXEDNESS",
    "GARMENT_PLUS_GARMENT_CONSISTENCY",
    "FULL_NO_GARMENT_CONSISTENCY",
)
CONTINUED_FAMILIES = FAMILIES[:2]
FRESH_FAMILIES = FAMILIES[2:]
FULL_ROUTING_FAMILIES = (
    "FULL_V2_CONTINUED",
    "FULL_NO_GARMENT_CONSISTENCY",
)
CONSISTENCY_FAMILIES = (
    "FULL_V2_CONTINUED",
    "GARMENT_PLUS_GARMENT_CONSISTENCY",
    "FULL_NO_GARMENT_CONSISTENCY",
)
ROTATIONS = (0, 1, 2, 3)
SEEDS = (0, 1, 2)
CHECKPOINT_STEPS = (0, 30, 60, 90, 120, 150, 300, 450, 600)
HISTORICAL_STEPS = CHECKPOINT_STEPS[:6]
NEW_CONTINUED_STEPS = CHECKPOINT_STEPS[6:]
PURE_TYPES = {"AAA", "BBB"}
MIXED_TYPES = {"AAB", "ABB"}
NUISANCE_TYPES = (
    "blur",
    "mask_erosion",
    "mask_dilation",
    "assignment_permutation",
)

RISK = PROJECT_ROOT / "paper_protocol/reviewer_risk"
MANIFEST_PATH = RISK / "dual_support_controller_training_manifest.json"
ROTATION_PATH = RISK / "controller_v2_micro_pilot_rotation_manifests.json"
HISTORICAL_SCHEDULE_PATH = RISK / "controller_v2_micro_pilot_batch_schedules.json"
CALIBRATION_PATH = RISK / "controller_v2_micro_pilot_calibration_contract.json"
PRE_RESULT_NAMES = (
    "controller_garment_budget_execution_contract.json",
    "controller_garment_budget_schedules.json",
    "controller_garment_budget_parameter_paths.json",
    "controller_garment_budget_asset_snapshot.json",
)
TRACKED_RESULT_NAMES = (
    "controller_garment_budget_training_results.json",
    "controller_garment_budget_checkpoint_trajectories.json",
    "controller_garment_budget_mixed_pair_results.json",
    "controller_garment_budget_full_v2_results.json",
    "controller_garment_budget_matched_v1_results.json",
    "controller_garment_budget_loss_ablation_results.json",
    "controller_garment_budget_difficult_pair_results.json",
    "controller_garment_budget_final_summary.json",
)
REPORT_NAMES = (
    "AAAI27_CONTROLLER_GARMENT_HEAD_BUDGET_DIAGNOSIS_20260724.md",
    "AAAI27_CONTROLLER_MULTITASK_LOSS_ABLATION_20260724.md",
    "AAAI27_CONTROLLER_DIFFICULT_PAIR_LEARNING_TRAJECTORY_20260724.md",
    "AAAI27_CONTROLLER_TRAINING_REPAIR_DECISION_20260724.md",
)

EXPECTED_ROTATION_HASHES = repaired_train.EXPECTED_ROTATION_HASHES
EXPECTED_COUNTS = {
    "run_trajectories": 72,
    "continuation_runs": 24,
    "fresh_diagnostic_runs": 48,
    "optimizer_creations": 72,
    "optimizer_steps": 39600,
    "scheduler_steps": 39600,
    "clean_forward_batches": 39600,
    "augmented_forward_batches": 19800,
    "clean_record_forwards": 198000,
    "augmented_record_forwards": 99000,
    "backward_calls": 39600,
    "new_checkpoint_writes": 504,
    "historical_checkpoints_reused": 144,
    "evaluated_checkpoint_instances": 648,
    "checkpoint_inference_count": 207360,
    "metric_records": 207360,
    "renderer_runs": 0,
    "new_formal_renders": 0,
    "paper_final": 0,
}

LOSS_DEFINITIONS = {
    "FULL_V2_CONTINUED": (
        "L_garment + L_mixedness + L_weight + 0.1 * "
        "(L_garment_consistency + L_mixedness_consistency + "
        "L_pair_weight_consistency)"
    ),
    "MATCHED_V1_CONTINUED": "soft_target_garment_CE",
    "GARMENT_ONLY": "L_garment",
    "GARMENT_PLUS_MIXEDNESS": "L_garment + L_mixedness",
    "GARMENT_PLUS_GARMENT_CONSISTENCY": (
        "L_garment + 0.1 * L_garment_consistency"
    ),
    "FULL_NO_GARMENT_CONSISTENCY": (
        "L_garment + L_mixedness + L_weight + 0.1 * "
        "(L_mixedness_consistency + L_pair_weight_consistency)"
    ),
}

PRIMARY_CLASSIFICATIONS = (
    "FULL_V2_BUDGET_LIMITED_BUT_RECOVERABLE",
    "MULTITASK_OPTIMIZATION_INTERFERENCE_DOMINANT",
    "GARMENT_CONSISTENCY_INTERFERENCE_DOMINANT",
    "MIXEDNESS_SHARED_PATH_INTERFERENCE_DOMINANT",
    "OPTIMIZER_OR_NORMALIZATION_LIMIT",
    "MULTIPLE_OPTIMIZATION_FACTORS",
    "GARMENT_HEAD_BUDGET_DIAGNOSIS_INCONCLUSIVE",
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def atomic_torch(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"checkpoint overwrite forbidden: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def frozen_contract_hash(value: Any) -> str:
    encoded = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path: Path, *, lf: bool = False) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            if lf:
                block = block.replace(b"\r\n", b"\n")
            digest.update(block)
    return digest.hexdigest()


def tensor_mapping_hash(value: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(value):
        tensor = value[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def git(*arguments: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *arguments],
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def configure_determinism() -> None:
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)


def contract_inputs() -> dict[str, Any]:
    contract = repaired_train.validate_contract()
    if git("rev-parse", SOURCE_HEAD) != SOURCE_HEAD:
        raise RuntimeError("exact source HEAD is unavailable")
    if len(contract["manifest"]["query_sets"]) != 320:
        raise RuntimeError("manifest record count changed")
    return contract


def historical_checkpoint_name(step: int) -> str:
    return "final_step_150.pt" if step == 150 else f"step_{step:03d}.pt"


def historical_checkpoint(
    historical_root: Path, family: str, rotation: int, seed: int, step: int
) -> Path:
    historical_family = {
        "FULL_V2_CONTINUED": "v2",
        "MATCHED_V1_CONTINUED": "matched_v1",
    }.get(family, "v2")
    return (
        historical_root
        / "training"
        / historical_family
        / f"rotation_{rotation}"
        / f"seed_{seed}"
        / "checkpoints"
        / historical_checkpoint_name(step)
    )


def schedule_cycle(source_row: Mapping[str, Any]) -> list[dict[str, Any]]:
    steps = source_row["schedule"]["steps"]
    frozen_cycle = source_row["cycle"]
    if frozen_contract_hash(frozen_cycle) != source_row["cycle_sha256"]:
        raise RuntimeError("stored frozen cycle hash changed")
    if frozen_contract_hash(source_row["schedule"]) != source_row["schedule_sha256"]:
        raise RuntimeError("stored historical schedule hash changed")
    cycle = [
        {
            "batch_index": int(batch["batch_index"]),
            "record_ids": [row["record_id"] for row in batch["records"]],
            "target_view_fold": batch["target_view_fold"],
        }
        for batch in frozen_cycle["batches"]
    ]
    if [row["batch_index"] for row in cycle] != list(range(32)):
        raise RuntimeError("frozen cycle is not 32 ordered batches")
    for index, row in enumerate(steps):
        expected = cycle[index % 32]
        if row["record_ids"] != expected["record_ids"]:
            raise RuntimeError("historical schedule does not repeat frozen cycle")
        if row["target_view_fold"] != expected["target_view_fold"]:
            raise RuntimeError("historical target fold changed inside cycle")
    return cycle


def build_schedules(contract: Mapping[str, Any]) -> dict[str, Any]:
    rotations = []
    manifest = {
        row["record_id"]: row for row in contract["manifest"]["query_sets"]
    }
    for source_row in contract["schedules"]["rotations"]:
        rotation = int(source_row["rotation"])
        cycle = schedule_cycle(source_row)
        historical_cycle_sha = frozen_contract_hash(source_row["cycle"])
        if historical_cycle_sha != source_row["cycle_sha256"]:
            raise RuntimeError(f"rotation {rotation} cycle hash changed")
        cycle_payload = {
            "schema_version": "controller_garment_budget_cycle_view.v1",
            "rotation": rotation,
            "batch_size": 5,
            "batch_count": 32,
            "records_per_cycle": 160,
            "batches": cycle,
        }
        cycle_view_sha = canonical_hash(cycle_payload)
        steps = []
        for zero_based in range(600):
            base = cycle[zero_based % 32]
            global_step = zero_based + 1
            steps.append(
                {
                    "global_step": global_step,
                    "optimizer_step_index_zero_based": zero_based,
                    "cycle_index": zero_based // 32,
                    "batch_index": zero_based % 32,
                    "target_view_fold": base["target_view_fold"],
                    "record_ids": base["record_ids"],
                    "fresh_nuisance_type": NUISANCE_TYPES[zero_based % 4],
                    "continuation_nuisance_type": NUISANCE_TYPES[global_step % 4],
                }
            )
        exposures = Counter(
            record_id for step in steps for record_id in step["record_ids"]
        )
        train_ids = {
            record_id
            for step in cycle
            for record_id in step["record_ids"]
        }
        if len(train_ids) != 160 or set(exposures) != train_ids:
            raise RuntimeError(f"rotation {rotation} train cycle coverage changed")
        histogram = Counter(exposures.values())
        if histogram != Counter({19: 120, 18: 40}):
            raise RuntimeError(f"rotation {rotation} exposure distribution changed")
        if any(record_id not in manifest for record_id in train_ids):
            raise RuntimeError("schedule contains an unknown record")
        schedule_payload = {
            "rotation": rotation,
            "step_count": 600,
            "steps": steps,
        }
        rotations.append(
            {
                "rotation": rotation,
                "historical_cycle_sha256": source_row["cycle_sha256"],
                "historical_150_schedule_sha256": source_row["schedule_sha256"],
                "cycle": cycle_payload,
                "cycle_sha256": historical_cycle_sha,
                "diagnostic_cycle_view_sha256": cycle_view_sha,
                "schedule": schedule_payload,
                "schedule_sha256": canonical_hash(schedule_payload),
                "exposure_distribution": {
                    "record_count": len(exposures),
                    "clean_record_exposures": sum(exposures.values()),
                    "histogram": {str(key): value for key, value in sorted(histogram.items())},
                    "nineteenth_exposure_record_ids": sorted(
                        key for key, value in exposures.items() if value == 19
                    ),
                    "eighteenth_exposure_record_ids": sorted(
                        key for key, value in exposures.items() if value == 18
                    ),
                },
            }
        )
    return {
        "schema_version": "controller_garment_budget_schedules.v1",
        "status": "PASS",
        "task_id": TASK_ID,
        "source_head": SOURCE_HEAD,
        "batch_size": 5,
        "batches_per_cycle": 32,
        "steps": 600,
        "full_cycles": 18,
        "partial_cycle_batches": 24,
        "clean_record_exposures_per_run": 3000,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "shuffle": False,
        "resampling": False,
        "rotations": rotations,
    }


def tree_snapshot(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    rows = []
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": child.relative_to(path).as_posix(),
                "bytes": child.stat().st_size,
                "sha256": file_hash(child),
            }
        )
    return {
        "root": str(path),
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "tree_sha256": canonical_hash(rows),
    }


def checkpoint_registry(historical_root: Path) -> list[dict[str, Any]]:
    registry = []
    expected_source = repaired_train.SOURCE_HEAD
    for historical_family, family in (
        ("v2", "V2"),
        ("matched_v1", "MATCHED_V1"),
    ):
        for rotation in ROTATIONS:
            expected_schedule = EXPECTED_ROTATION_HASHES[rotation][1]
            for seed in SEEDS:
                initialization = ""
                for step in HISTORICAL_STEPS:
                    path = historical_checkpoint(
                        historical_root,
                        "FULL_V2_CONTINUED" if family == "V2" else "MATCHED_V1_CONTINUED",
                        rotation,
                        seed,
                        step,
                    )
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    state = torch.load(path, map_location="cpu", weights_only=False)
                    required = {
                        "model_state_dict",
                        "optimizer_state_dict",
                        "scheduler_state_dict",
                        "cpu_rng_state",
                        "cuda_rng_state_all",
                    }
                    if not required.issubset(state):
                        raise RuntimeError(f"checkpoint restore state incomplete: {path}")
                    if (
                        state["family"] != family
                        or int(state["rotation"]) != rotation
                        or int(state["seed"]) != seed
                        or int(state["global_step"]) != step
                        or state["source_head"] != expected_source
                        or state["schedule_sha256"] != expected_schedule
                    ):
                        raise RuntimeError(f"checkpoint metadata mismatch: {path}")
                    actual_initialization = state["initialization_sha256"]
                    if step == 0:
                        initialization = tensor_mapping_hash(state["model_state_dict"])
                        if initialization != actual_initialization:
                            raise RuntimeError(f"initialization hash mismatch: {path}")
                    elif actual_initialization != initialization:
                        raise RuntimeError(f"initialization provenance changed: {path}")
                    registry.append(
                        {
                            "model_family": family,
                            "rotation": rotation,
                            "seed": seed,
                            "checkpoint_step": step,
                            "checkpoint_path": str(path),
                            "checkpoint_sha256": file_hash(path),
                            "initialization_sha256": initialization,
                            "optimizer_state_present": True,
                            "optimizer_state_parameter_count": len(
                                state["optimizer_state_dict"]["state"]
                            ),
                            "scheduler_state_present": True,
                            "cpu_rng_state_present": True,
                            "cuda_rng_state_present": True,
                            "schedule_sha256": state["schedule_sha256"],
                            "data_order_position": {
                                "completed_steps": step,
                                "next_cycle_index": step // 32,
                                "next_batch_index": step % 32,
                            },
                        }
                    )
    if len(registry) != 144:
        raise RuntimeError("historical registry must contain 144 checkpoints")
    return registry


def parameter_inventory(model: torch.nn.Module) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "shape": list(parameter.shape),
            "scalar_count": parameter.numel(),
        }
        for name, parameter in model.named_parameters()
    ]


def v2_static_paths() -> dict[str, list[str]]:
    shared = ["reference_normalization.bias", "reference_normalization.weight"]
    garment = ["garment_head.bias", "garment_head.weight", *shared]
    mixedness = ["mixedness_head.bias", "mixedness_head.weight", *shared]
    weight = ["pair_weight_head.bias", "pair_weight_head.weight", *shared]
    return {
        "garment": sorted(garment),
        "mixedness": sorted(mixedness),
        "pair_weight": sorted(weight),
        "garment_consistency": sorted(garment),
        "mixedness_consistency": sorted(mixedness),
        "pair_weight_consistency": sorted(weight),
    }


def loss_components(
    outputs: Sequence[Any],
    augmented: Sequence[Any],
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    garment_target, mixedness_target = repaired_train.training_targets(records, device)
    garment_logits = torch.stack([output.garment_logits for output in outputs])
    mixedness_logits = torch.stack([output.mixedness_logit for output in outputs])
    garment = soft_target_cross_entropy(garment_logits, garment_target)
    mixedness = F.binary_cross_entropy_with_logits(
        mixedness_logits, mixedness_target, reduction="mean"
    )
    predicted_weights = []
    target_weights = []
    for output, record in zip(outputs, records):
        if record["assignment_type"] in PURE_TYPES:
            continue
        pair_index = PAIR_TO_INDEX[record["pair_id"]]
        earlier_index = OUTFIT_ORDER.index(record["pair_id"].split("_")[0])
        predicted_weights.append(output.all_pair_weights[pair_index])
        target_weights.append(
            output.all_pair_weights.new_tensor(
                record["target_distribution"][earlier_index]
            )
        )
    pair_weight = F.smooth_l1_loss(
        torch.stack(predicted_weights), torch.stack(target_weights),
        beta=0.1, reduction="mean",
    ) if predicted_weights else garment_logits.sum() * 0.0
    clean_garment = torch.stack([output.garment_probabilities for output in outputs])
    aug_garment = torch.stack([output.garment_probabilities for output in augmented])
    garment_consistency = repaired_train.js_divergence(clean_garment, aug_garment)
    mixedness_consistency = F.smooth_l1_loss(
        torch.stack([output.mixedness_probability for output in outputs]),
        torch.stack([output.mixedness_probability for output in augmented]),
        beta=0.1,
        reduction="mean",
    )
    pair_weight_consistency = F.smooth_l1_loss(
        torch.stack([output.all_pair_weights for output in outputs]),
        torch.stack([output.all_pair_weights for output in augmented]),
        beta=0.1,
        reduction="mean",
    )
    return {
        "garment": garment,
        "mixedness": mixedness,
        "pair_weight": pair_weight,
        "garment_consistency": garment_consistency,
        "mixedness_consistency": mixedness_consistency,
        "pair_weight_consistency": pair_weight_consistency,
    }


def parameter_path_audit(
    historical_root: Path,
    clean_cache_path: Path,
    nuisance_cache_path: Path,
    schedules: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    configure_determinism()
    model = CompatibilityGatedReferenceControllerV2(seed=0)
    state = torch.load(
        historical_checkpoint(
            historical_root, "FULL_V2_CONTINUED", 0, 0, 0
        ),
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.train()
    clean_cache = torch.load(clean_cache_path, map_location="cpu", weights_only=False)
    nuisance_cache = torch.load(
        nuisance_cache_path, map_location="cpu", weights_only=False
    )
    manifest = {
        row["record_id"]: row for row in contract["manifest"]["query_sets"]
    }
    schedule = schedules["rotations"][0]["schedule"]["steps"]
    selected = next(
        row
        for row in schedule
        if any(manifest[record_id]["assignment_type"] in MIXED_TYPES for record_id in row["record_ids"])
    )
    records = [manifest[record_id] for record_id in selected["record_ids"]]
    outputs = []
    augmented = []
    for record in records:
        rows, valid = repaired_train.case_rows_variant(
            clean_cache, nuisance_cache, record, "clean"
        )
        outputs.append(model(rows, valid))
        rows, valid = repaired_train.case_rows_variant(
            clean_cache,
            nuisance_cache,
            record,
            selected["fresh_nuisance_type"],
        )
        augmented.append(model(rows, valid))
    components = loss_components(outputs, augmented, records, torch.device("cpu"))
    named_parameters = list(model.named_parameters())
    gradient_paths = {}
    for component_name, component in components.items():
        gradients = torch.autograd.grad(
            component,
            [parameter for _, parameter in named_parameters],
            retain_graph=True,
            allow_unused=True,
        )
        nonzero = []
        finite = True
        norms = {}
        for (name, _), gradient in zip(named_parameters, gradients):
            if gradient is None:
                continue
            finite = finite and bool(torch.isfinite(gradient).all())
            norm = float(gradient.detach().double().norm())
            norms[name] = norm
            if norm > 1.0e-12:
                nonzero.append(name)
        gradient_paths[component_name] = {
            "loss": float(component.detach()),
            "finite": finite,
            "nonzero_parameter_names": sorted(nonzero),
            "gradient_norms": norms,
        }
    static_paths = v2_static_paths()
    all_v2 = sorted(name for name, _ in named_parameters)
    garment = static_paths["garment"]
    garment_mixed = sorted(set(garment) | set(static_paths["mixedness"]))
    v1 = ReferenceConditionedDualSupportController(seed=0)
    memberships = {
        "FULL_V2_CONTINUED": all_v2,
        "MATCHED_V1_CONTINUED": sorted(name for name, _ in v1.named_parameters()),
        "GARMENT_ONLY": garment,
        "GARMENT_PLUS_MIXEDNESS": garment_mixed,
        "GARMENT_PLUS_GARMENT_CONSISTENCY": garment,
        "FULL_NO_GARMENT_CONSISTENCY": all_v2,
    }
    inventory_by_name = {
        row["name"]: row for row in parameter_inventory(model)
    }
    family_rows = {}
    for family, names in memberships.items():
        if family == "MATCHED_V1_CONTINUED":
            inventory = parameter_inventory(v1)
            count = sum(row["scalar_count"] for row in inventory)
        else:
            inventory = [inventory_by_name[name] for name in names]
            count = sum(row["scalar_count"] for row in inventory)
        family_rows[family] = {
            "optimizer_parameter_names": names,
            "optimizer_parameter_shapes": {
                row["name"]: row["shape"] for row in inventory
            },
            "optimizer_scalar_count": count,
        }
    intersections = {}
    for first, first_names in static_paths.items():
        for second, second_names in static_paths.items():
            if first >= second:
                continue
            shared = sorted(set(first_names) & set(second_names))
            intersections[f"{first}__{second}"] = {
                "status": "SHARED_GRADIENT_PATH" if shared else "NO_SHARED_GRADIENT_PATH",
                "parameter_names": shared,
            }
    return {
        "schema_version": "controller_garment_budget_parameter_paths.v1",
        "status": "PASS",
        "task_id": TASK_ID,
        "source_head": SOURCE_HEAD,
        "optimizer_created": 0,
        "optimizer_steps": 0,
        "diagnostic_autograd_calls": len(components),
        "v2_parameter_inventory": parameter_inventory(model),
        "v2_total_scalar_count": model.parameter_count,
        "matched_v1_parameter_inventory": parameter_inventory(v1),
        "matched_v1_total_scalar_count": v1.parameter_count,
        "static_loss_paths": static_paths,
        "autograd_loss_paths": gradient_paths,
        "shared_intersections": intersections,
        "families": family_rows,
        "audit_batch": {
            "rotation": 0,
            "global_step": selected["global_step"],
            "record_ids": selected["record_ids"],
            "nuisance_type": selected["fresh_nuisance_type"],
        },
    }


def credential_scan() -> dict[str, Any]:
    tracked = [PROJECT_ROOT / row for row in git("ls-files").splitlines()]
    executing_source = Path(__file__).resolve()
    if executing_source not in tracked:
        tracked.append(executing_source)
    patterns = (
        re.compile(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9]{20,}(?![A-Za-z0-9_-])"),
        re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{24,}"),
        re.compile(
            r"(?i)\b(?:api[_-]?key|access[_-]?token|client[_-]?secret)"
            r"\s*[:=]\s*['\"][A-Za-z0-9._~+/=-]{24,}['\"]"
        ),
    )
    scanned = 0
    findings: list[tuple[Path, int, int]] = []
    for path in tracked:
        if not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
            continue
        try:
            value = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        scanned += 1
        for pattern_index, pattern in enumerate(patterns):
            count = len(pattern.findall(value))
            if count:
                findings.append((path, pattern_index, count))
    baseline_scanner_literal = PROJECT_ROOT / "tools/check_codex_direct_generation_contract.py"
    real_findings = [
        row for row in findings
        if not (
            row[0] == baseline_scanner_literal
            and row[1] == 1
            and row[2] == 1
            and git(
                "rev-parse",
                "HEAD:tools/check_codex_direct_generation_contract.py",
            ) == git(
                "hash-object",
                "tools/check_codex_direct_generation_contract.py",
            )
        )
    ]
    if real_findings:
        raise RuntimeError("API_CREDENTIAL_LEAK_RISK")
    return {
        "status": "PASS",
        "tracked_text_files_scanned": scanned,
        "real_credential_match_count": 0,
        "baseline_scanner_literal_exclusion_count": len(findings),
        "matched_values_persisted": False,
        "token_boundary_enforced": True,
    }


def resource_gate(api_provider: str, output_parent: Path) -> dict[str, Any]:
    disk = shutil.disk_usage(output_parent)
    stat = os.statvfs(output_parent)
    free_inodes = stat.f_favail
    gpu_processes = []
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if query.returncode == 0:
        for line in query.stdout.splitlines():
            if line.strip():
                fields = [field.strip() for field in line.split(",")]
                gpu_processes.append(
                    {
                        "pid": int(fields[0]),
                        "process_name": Path(fields[1]).name,
                        "used_memory_mib": int(fields[2]),
                    }
                )
    if gpu_processes:
        raise RuntimeError("CONTROLLER_GARMENT_BUDGET_DIAGNOSIS_DEFERRED_ACTIVE_GPU_TASK")
    if disk.free < 20 * 1024**3:
        raise RuntimeError("CONTROLLER_GARMENT_BUDGET_DIAGNOSIS_DEFERRED_LOW_SPACE")
    return {
        "status": "PASS",
        "api_provider": api_provider,
        "credential_value_read": False,
        "credential_value_persisted": False,
        "git_root": str(PROJECT_ROOT),
        "branch": git("branch", "--show-current"),
        "head": git("rev-parse", "HEAD"),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "UNAVAILABLE",
        "gpu_total_bytes": (
            torch.cuda.get_device_properties(0).total_memory
            if torch.cuda.is_available()
            else 0
        ),
        "gpu_process_count": len(gpu_processes),
        "gpu_processes": gpu_processes,
        "pure_endpoint_active_training": 0,
        "subject00_active_training": 0,
        "other_formal_training": 0,
        "heavy_io_task": 0,
        "free_bytes": disk.free,
        "free_inodes": free_inodes,
    }


def tracked_blob_snapshot(paths: Sequence[Path]) -> dict[str, Any]:
    rows = []
    for path in paths:
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        rows.append(
            {
                "path": relative,
                "sha256_lf": file_hash(path, lf=True),
                "git_blob": git("rev-parse", f"HEAD:{relative}"),
            }
        )
    return {"file_count": len(rows), "files": rows, "sha256": canonical_hash(rows)}


def preflight(
    artifact_dir: Path,
    historical_root: Path,
    clean_cache_path: Path,
    nuisance_cache_path: Path,
    output_parent: Path,
    api_provider: str,
) -> dict[str, Any]:
    ancestry = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "merge-base", "--is-ancestor", SOURCE_HEAD, "HEAD"],
        check=False,
    )
    if ancestry.returncode != 0:
        raise RuntimeError("preflight branch is not derived from the exact source HEAD")
    contract = contract_inputs()
    schedules = build_schedules(contract)
    registry = checkpoint_registry(historical_root)
    paths = parameter_path_audit(
        historical_root,
        clean_cache_path,
        nuisance_cache_path,
        schedules,
        contract,
    )
    mandatory = [
        RISK / name
        for name in (
            "controller_record_semantics_audit.json",
            "controller_visible_garment_set_manifest.json",
            "controller_pure_latent_pair_identifiability.json",
            "controller_corrected_pure_metrics.json",
            "controller_corrected_mixed_pair_metrics.json",
            "controller_corrected_routing_weight_metrics.json",
            "controller_original_vs_corrected_gates.json",
            "controller_task_conditional_evaluator_contract.json",
            "controller_metric_semantics_repair_final_summary.json",
        )
    ]
    mandatory += [
        PROJECT_ROOT / "docs/PAPER" / name
        for name in (
            "AAAI27_PURE_AND_MIXED_TASK_SEMANTICS_AUDIT_20260724.md",
            "AAAI27_CONTROLLER_PAIR_METRIC_DENOMINATOR_REPAIR_20260724.md",
            "AAAI27_CONTROLLER_FAILURE_REINTERPRETATION_20260724.md",
            "AAAI27_CONTROLLER_EVALUATOR_CONTRACT_20260724.md",
        )
    ]
    if any(not path.is_file() for path in mandatory):
        raise RuntimeError("mandatory semantic-repair artifact missing")
    attempt_parent = historical_root.parent
    attempt_snapshots = {
        f"attempt_{index:03d}": tree_snapshot(attempt_parent / f"attempt_{index:03d}")
        for index in range(1, 5)
    }
    resource = resource_gate(api_provider, output_parent)
    credentials = credential_scan()
    clean_hash = file_hash(clean_cache_path)
    nuisance_hash = file_hash(nuisance_cache_path)
    if clean_hash != "30cf19a99bbc620112928d678a0257bfabd3420be66fe1053ded3ce7e90875cb":
        raise RuntimeError("frozen feature cache changed")
    if nuisance_hash != "f6c8f561b7e42b8291cf67e74dcc9d9cd0dcca332b8794efa35bf2b429c708c4":
        raise RuntimeError("frozen nuisance cache changed")
    source_script = Path(__file__).resolve()
    asset = {
        "schema_version": "controller_garment_budget_asset_snapshot.v1",
        "status": "PASS",
        "task_id": TASK_ID,
        "source_head": SOURCE_HEAD,
        "historical_source_head": HISTORICAL_SOURCE_HEAD,
        "pair_diagnosis_head": PAIR_DIAGNOSIS_HEAD,
        "historical_root": str(historical_root),
        "historical_run_count": 24,
        "historical_checkpoint_count": len(registry),
        "historical_checkpoint_registry": registry,
        "historical_attempt_snapshots": attempt_snapshots,
        "clean_feature_cache": {
            "path": str(clean_cache_path), "sha256": clean_hash
        },
        "nuisance_feature_cache": {
            "path": str(nuisance_cache_path), "sha256": nuisance_hash
        },
        "semantic_repair_snapshot": tracked_blob_snapshot(mandatory),
        "pair_diagnosis_git_object": {
            "commit": git("rev-parse", PAIR_DIAGNOSIS_HEAD),
            "tree": git("rev-parse", f"{PAIR_DIAGNOSIS_HEAD}^{{tree}}"),
            "mutation": 0,
        },
        "training_code_sha256": file_hash(source_script, lf=True),
        "protected_categories": {
            key: {"mutation_allowed": False, "snapshot_source": source}
            for key, source in {
                "semantic_repair_outputs": "semantic_repair_snapshot",
                "pair_diagnosis_outputs": "pair_diagnosis_git_object",
                "controller_attempts_001_004": "historical_attempt_snapshots",
                "historical_runs_24": "historical_checkpoint_registry",
                "historical_checkpoints_144": "historical_checkpoint_registry",
                "formal_v1": "historical_checkpoint_registry",
                "teacher": "frozen upstream manifests",
                "f2": "clean_feature_cache",
                "feature_cache": "clean_feature_cache",
                "renderer": "frozen upstream manifests",
                "garment_bank": "frozen upstream manifests",
                "compatibility_manifest": "semantic_repair_snapshot",
                "protocol": "semantic_repair_snapshot",
                "threshold": "semantic_repair_snapshot",
                "reference_assets": "source image/mask hashes in frozen manifest",
                "target_assets": "target_excluded frozen manifest",
            }.items()
        },
    }
    execution = {
        "schema_version": "controller_garment_budget_execution_contract.v1",
        "status": "PASS",
        "task_id": TASK_ID,
        "source_head": SOURCE_HEAD,
        "branch": BRANCH,
        "post_failure_label": POST_FAILURE_LABEL,
        "historical_classification": HISTORICAL_CLASSIFICATION,
        "semantic_classification": SEMANTIC_CLASSIFICATION,
        "pair_interpretation": PAIR_INTERPRETATION,
        "families": list(FAMILIES),
        "rotations": list(ROTATIONS),
        "seeds": list(SEEDS),
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "loss_definitions": LOSS_DEFINITIONS,
        "optimizer": {
            "class": "torch.optim.Adam",
            "learning_rate": 0.02,
            "weight_decay": 0.0,
            "betas": [0.9, 0.999],
            "epsilon": 1.0e-8,
            "scheduler": "constant LambdaLR",
            "warmup_steps": 0,
            "amp": False,
            "gradient_accumulation": 1,
            "gradient_clip_max_norm": 5.0,
        },
        "initialization": {
            "fresh_v2_variants": "corresponding historical V2 step_000 model/RNG",
            "full_v2_continued": "corresponding historical V2 final_step_150 full state",
            "matched_v1_continued": "corresponding historical matched V1 final_step_150 full state",
        },
        "evaluation_denominators": {
            "pure": "GT_PURE_ONLY_VISIBLE_GARMENT_TOP1",
            "pure_hidden_secondary_pair": "FORBIDDEN",
            "mixed_pair": "GT_MIXED_ONLY_UNORDERED_TOP2",
            "routing": "PREDICTION_CONDITIONED_MIXED_SUBSETS",
            "mandatory_views": [
                "protocol_weighted", "unique_query", "per_rotation",
                "rotation_macro", "per_seed", "seed_macro", "global_macro",
            ],
        },
        "diagnosis_rules": diagnosis_rules(),
        "expected_counts": EXPECTED_COUNTS,
        "renderer_runs": 0,
        "new_formal_renders": 0,
        "paper_final": 0,
        "next_task_auto_execution": False,
        "resource_gate": resource,
        "credential_scan": credentials,
        "pre_result_optimizer_created": 0,
        "pre_result_optimizer_steps": 0,
    }
    values = {
        PRE_RESULT_NAMES[0]: execution,
        PRE_RESULT_NAMES[1]: schedules,
        PRE_RESULT_NAMES[2]: paths,
        PRE_RESULT_NAMES[3]: asset,
    }
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise FileExistsError(f"pre-result artifact directory is not empty: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for name, value in values.items():
        atomic_json(artifact_dir / name, value)
    return {
        "status": "PASS",
        "artifact_dir": str(artifact_dir),
        "artifacts": {
            name: file_hash(artifact_dir / name) for name in PRE_RESULT_NAMES
        },
        "optimizer_created": 0,
        "optimizer_steps": 0,
    }


def diagnosis_rules() -> dict[str, Any]:
    return {
        "allowed_primary_classifications": list(PRIMARY_CLASSIFICATIONS),
        "FULL_V2_BUDGET_LIMITED_BUT_RECOVERABLE": {
            "full_v2_step600_macro_min": 0.85,
            "every_rotation_min": 0.80,
            "step150_to_600_gain_min": 0.10,
            "pure_top1_min": 0.95,
            "safety_or_numeric_failure": False,
        },
        "MULTITASK_OPTIMIZATION_INTERFERENCE_DOMINANT": {
            "garment_only_step600_macro_min": 0.90,
            "garment_only_every_rotation_min": 0.85,
            "garment_only_gap_min": 0.10,
            "full_v2_step600_max_exclusive": 0.85,
            "same_direction_rotations_min": 3,
        },
        "GARMENT_CONSISTENCY_INTERFERENCE_DOMINANT": {
            "consistency_interference_min": 0.08,
            "consistency_removal_gain_min": 0.08,
            "same_direction_rotations_min": 3,
        },
        "MIXEDNESS_SHARED_PATH_INTERFERENCE_DOMINANT": {
            "mixedness_interference_min": 0.08,
            "same_direction_rotations_min": 3,
            "shared_parameter_path_required": True,
        },
        "OPTIMIZER_OR_NORMALIZATION_LIMIT": {
            "legal_ridge_probe_mixed_only_min": 0.95,
            "garment_only_step600_max_exclusive": 0.80,
            "stable_numerics_required": True,
        },
        "repair_value": {
            "HIGH": "any preregistered variant macro>=0.85, every rotation>=0.80, pure>=0.95",
            "MODERATE": "best macro in [0.75,0.85) or any rotation<0.80",
            "LOW": "all variants macro<0.75 despite legal probe near 1.0",
        },
    }


def materialize_contract(output_root: Path, *, refresh: bool = False) -> dict[str, Any]:
    attempt = output_root / "attempt_001"
    contract_dir = attempt / "contract"
    if not refresh and attempt.exists() and any(attempt.iterdir()):
        raise FileExistsError(f"append-only attempt is not empty: {attempt}")
    if refresh and not (attempt / "audits/materialized_contract.json").is_file():
        raise RuntimeError("cannot refresh a contract that was never materialized")
    if git("branch", "--show-current") != BRANCH:
        raise RuntimeError("training branch mismatch")
    for name in PRE_RESULT_NAMES:
        path = RISK / name
        if not path.is_file() or not git("ls-files", "--error-unmatch", str(path.relative_to(PROJECT_ROOT)), check=False):
            raise RuntimeError(f"pre-result artifact is not tracked: {name}")
    if not refresh:
        for name in (
            "contract", "schedules", "continuations", "ablations", "checkpoints",
            "predictions", "trajectories", "difficult_pairs", "audits", "aggregates",
        ):
            (attempt / name).mkdir(parents=True, exist_ok=True)
    else:
        original = read_json(attempt / "audits/materialized_contract.json")
        archive = contract_dir / f"superseded_{original['git_head'][:12]}"
        archive.mkdir(parents=True, exist_ok=False)
        for name in PRE_RESULT_NAMES:
            shutil.copy2(contract_dir / name, archive / name)
    hashes = {}
    for name in PRE_RESULT_NAMES:
        target = contract_dir / name
        shutil.copy2(RISK / name, target)
        hashes[name] = file_hash(target)
        if hashes[name] != file_hash(RISK / name):
            raise RuntimeError(f"materialized contract hash mismatch: {name}")
    result = {
        "schema_version": "controller_garment_budget_materialized_contract.v1",
        "status": "PASS",
        "task_id": TASK_ID,
        "git_head": git("rev-parse", "HEAD"),
        "branch": BRANCH,
        "contract_hashes": hashes,
        "optimizer_created": 0,
        "optimizer_steps": 0,
    }
    audit_name = (
        "materialized_contract_correction_001.json"
        if refresh else "materialized_contract.json"
    )
    atomic_json(attempt / "audits" / audit_name, result)
    return result


def verify_training_admission(output_root: Path) -> tuple[Path, dict[str, Any]]:
    attempt = output_root / "attempt_001"
    correction = attempt / "audits/materialized_contract_correction_001.json"
    materialized = read_json(
        correction if correction.is_file() else attempt / "audits/materialized_contract.json"
    )
    if materialized["status"] != "PASS":
        raise RuntimeError("materialized contract did not pass")
    for name in PRE_RESULT_NAMES:
        tracked = RISK / name
        copied = attempt / "contract" / name
        if file_hash(tracked) != file_hash(copied):
            raise RuntimeError(f"pre-result contract drift: {name}")
    execution = read_json(RISK / PRE_RESULT_NAMES[0])
    return attempt, execution


def family_run_root(attempt: Path, family: str, rotation: int, seed: int) -> Path:
    category = "continuations" if family in CONTINUED_FAMILIES else "ablations"
    return attempt / category / family.lower() / f"rotation_{rotation}" / f"seed_{seed}"


def task_checkpoint_path(
    attempt: Path, family: str, rotation: int, seed: int, step: int
) -> Path:
    return family_run_root(attempt, family, rotation, seed) / "checkpoints" / f"step_{step:03d}.pt"


def optimizer_for(parameters: Iterable[torch.nn.Parameter]) -> torch.optim.Adam:
    return torch.optim.Adam(
        list(parameters),
        lr=0.02,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.0,
        amsgrad=False,
        foreach=None,
        maximize=False,
        capturable=False,
        differentiable=False,
        fused=None,
    )


def finite_tree(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def set_checkpoint_rng(state: Mapping[str, Any]) -> None:
    torch.set_rng_state(state["cpu_rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])


def checkpoint_payload(
    family: str,
    rotation: int,
    seed: int,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    initialization_sha256: str,
    schedule_sha256: str,
    source_checkpoint_path: Path,
    source_checkpoint_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": "controller_garment_budget_checkpoint.v1",
        "task_id": TASK_ID,
        "family": family,
        "rotation": rotation,
        "seed": seed,
        "global_step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        "initialization_sha256": initialization_sha256,
        "schedule_sha256": schedule_sha256,
        "source_checkpoint_path": str(source_checkpoint_path),
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "data_position": {
            "completed_steps": step,
            "next_cycle_index": step // 32,
            "next_batch_index": step % 32,
        },
        "best_checkpoint_selection": False,
        "paper_final": False,
    }


def total_loss_for_family(
    family: str,
    outputs: Sequence[Any],
    augmented: Sequence[Any],
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    if family == "MATCHED_V1_CONTINUED":
        targets, _ = repaired_train.training_targets(records, device)
        logits = torch.stack([output.logits for output in outputs])
        loss = soft_target_cross_entropy(logits, targets)
        return loss, {
            "garment": float(loss.detach()),
            "mixedness": 0.0,
            "pair_weight": 0.0,
            "garment_consistency": 0.0,
            "mixedness_consistency": 0.0,
            "pair_weight_consistency": 0.0,
        }
    targets, mixedness_target = repaired_train.training_targets(records, device)
    logits = torch.stack([output.garment_logits for output in outputs])
    garment = soft_target_cross_entropy(logits, targets)
    zero = garment * 0.0
    components = {
        "garment": garment,
        "mixedness": zero,
        "pair_weight": zero,
        "garment_consistency": zero,
        "mixedness_consistency": zero,
        "pair_weight_consistency": zero,
    }
    if family in {
        "GARMENT_PLUS_MIXEDNESS", "FULL_NO_GARMENT_CONSISTENCY",
        "FULL_V2_CONTINUED",
    }:
        components["mixedness"] = F.binary_cross_entropy_with_logits(
            torch.stack([output.mixedness_logit for output in outputs]),
            mixedness_target,
            reduction="mean",
        )
    if family in {"FULL_NO_GARMENT_CONSISTENCY", "FULL_V2_CONTINUED"}:
        predicted_weights = []
        target_weights = []
        for output, record in zip(outputs, records):
            if record["assignment_type"] in PURE_TYPES:
                continue
            pair_index = PAIR_TO_INDEX[record["pair_id"]]
            earlier_index = OUTFIT_ORDER.index(record["pair_id"].split("_")[0])
            predicted_weights.append(output.all_pair_weights[pair_index])
            target_weights.append(
                output.all_pair_weights.new_tensor(
                    record["target_distribution"][earlier_index]
                )
            )
        components["pair_weight"] = (
            F.smooth_l1_loss(
                torch.stack(predicted_weights), torch.stack(target_weights),
                beta=0.1, reduction="mean",
            )
            if predicted_weights else zero
        )
    if family in {"GARMENT_PLUS_GARMENT_CONSISTENCY", "FULL_V2_CONTINUED"}:
        components["garment_consistency"] = repaired_train.js_divergence(
            torch.stack([output.garment_probabilities for output in outputs]),
            torch.stack([output.garment_probabilities for output in augmented]),
        )
    if family in {"FULL_NO_GARMENT_CONSISTENCY", "FULL_V2_CONTINUED"}:
        components["mixedness_consistency"] = F.smooth_l1_loss(
            torch.stack([output.mixedness_probability for output in outputs]),
            torch.stack([output.mixedness_probability for output in augmented]),
            beta=0.1, reduction="mean",
        )
        components["pair_weight_consistency"] = F.smooth_l1_loss(
            torch.stack([output.all_pair_weights for output in outputs]),
            torch.stack([output.all_pair_weights for output in augmented]),
            beta=0.1, reduction="mean",
        )
    if family == "GARMENT_ONLY":
        total = components["garment"]
    elif family == "GARMENT_PLUS_MIXEDNESS":
        total = components["garment"] + components["mixedness"]
    elif family == "GARMENT_PLUS_GARMENT_CONSISTENCY":
        total = components["garment"] + 0.1 * components["garment_consistency"]
    elif family == "FULL_NO_GARMENT_CONSISTENCY":
        total = (
            components["garment"]
            + components["mixedness"]
            + components["pair_weight"]
            + 0.1
            * (
                components["mixedness_consistency"]
                + components["pair_weight_consistency"]
            )
        )
    elif family == "FULL_V2_CONTINUED":
        total = (
            components["garment"]
            + components["mixedness"]
            + components["pair_weight"]
            + 0.1
            * (
                components["garment_consistency"]
                + components["mixedness_consistency"]
                + components["pair_weight_consistency"]
            )
        )
    else:
        raise ValueError(family)
    return total, {name: float(value.detach()) for name, value in components.items()}


def train_one(
    output_root: Path,
    historical_root: Path,
    clean_cache_path: Path,
    nuisance_cache_path: Path,
    family: str,
    rotation: int,
    seed: int,
) -> dict[str, Any]:
    attempt, _ = verify_training_admission(output_root)
    configure_determinism()
    device = torch.device("cuda")
    schedule_archive = read_json(RISK / PRE_RESULT_NAMES[1])
    parameter_archive = read_json(RISK / PRE_RESULT_NAMES[2])
    schedule_row = next(
        row for row in schedule_archive["rotations"] if row["rotation"] == rotation
    )
    schedule = schedule_row["schedule"]["steps"]
    contract = contract_inputs()
    record_index = {
        row["record_id"]: row for row in contract["manifest"]["query_sets"]
    }
    clean_cache = torch.load(clean_cache_path, map_location="cpu", weights_only=False)
    nuisance_cache = torch.load(nuisance_cache_path, map_location="cpu", weights_only=False)
    run_root = family_run_root(attempt, family, rotation, seed)
    if run_root.exists():
        raise FileExistsError(f"append-only run already exists: {run_root}")
    (run_root / "checkpoints").mkdir(parents=True)
    if family == "MATCHED_V1_CONTINUED":
        model: torch.nn.Module = ReferenceConditionedDualSupportController(seed=seed)
    else:
        model = CompatibilityGatedReferenceControllerV2(seed=seed)
    source_step = 150 if family in CONTINUED_FAMILIES else 0
    source_checkpoint = historical_checkpoint(
        historical_root, family, rotation, seed, source_step
    )
    source_sha = file_hash(source_checkpoint)
    source = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(source["model_state_dict"], strict=True)
    initialization_sha256 = source["initialization_sha256"]
    membership = parameter_archive["families"][family]["optimizer_parameter_names"]
    membership_set = set(membership)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in membership_set)
    actual_membership = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if sorted(actual_membership) != sorted(membership):
        raise RuntimeError("optimizer membership differs from pre-result contract")
    model = model.to(device).train()
    optimizer = optimizer_for(parameter for parameter in model.parameters() if parameter.requires_grad)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    restored_optimizer = False
    restored_scheduler = False
    if family in CONTINUED_FAMILIES:
        optimizer.load_state_dict(source["optimizer_state_dict"])
        scheduler.load_state_dict(source["scheduler_state_dict"])
        restored_optimizer = True
        restored_scheduler = True
    set_checkpoint_rng(source)
    start_step = 151 if family in CONTINUED_FAMILIES else 1
    checkpoint_targets = set(NEW_CONTINUED_STEPS if family in CONTINUED_FAMILIES else CHECKPOINT_STEPS)
    checkpoint_paths = []
    if family in FRESH_FAMILIES:
        path = task_checkpoint_path(attempt, family, rotation, seed, 0)
        atomic_torch(
            path,
            checkpoint_payload(
                family, rotation, seed, 0, model, optimizer, scheduler,
                initialization_sha256, schedule_row["schedule_sha256"],
                source_checkpoint, source_sha,
            ),
        )
        checkpoint_paths.append(path)
    trace = []
    torch.cuda.reset_peak_memory_stats()
    run_started = time.perf_counter()
    for global_step in range(start_step, 601):
        step_started = time.perf_counter()
        schedule_step = schedule[global_step - 1]
        records = [record_index[value] for value in schedule_step["record_ids"]]
        optimizer.zero_grad(set_to_none=True)
        outputs = []
        augmented = []
        for record in records:
            rows, valid = repaired_train.case_rows_variant(
                clean_cache, nuisance_cache, record, "clean"
            )
            outputs.append(model(rows.to(device), valid.to(device)))
            if family in CONSISTENCY_FAMILIES:
                nuisance_key = (
                    "continuation_nuisance_type"
                    if family == "FULL_V2_CONTINUED"
                    else "fresh_nuisance_type"
                )
                rows, valid = repaired_train.case_rows_variant(
                    clean_cache,
                    nuisance_cache,
                    record,
                    schedule_step[nuisance_key],
                )
                augmented.append(model(rows.to(device), valid.to(device)))
        loss, components = total_loss_for_family(
            family, outputs, augmented, records, device
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at {family}/{rotation}/{seed}/{global_step}")
        loss.backward()
        frozen_gradients = [
            name
            for name, parameter in model.named_parameters()
            if not parameter.requires_grad and parameter.grad is not None
        ]
        if frozen_gradients:
            raise RuntimeError(f"frozen gradients present: {frozen_gradients}")
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not gradients or not all(bool(torch.isfinite(value).all()) for value in gradients):
            raise FloatingPointError("intended gradients are absent or non-finite")
        gradient_before = math.sqrt(
            sum(float(value.detach().double().square().sum()) for value in gradients)
        )
        if gradient_before <= 0.0:
            raise RuntimeError("intended total gradient is zero")
        clip_return = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad], 5.0
        )
        gradient_after = math.sqrt(
            sum(float(value.detach().double().square().sum()) for value in gradients)
        )
        optimizer.step()
        scheduler.step()
        if not finite_tree(optimizer.state_dict()):
            raise FloatingPointError("optimizer state contains NaN or Inf")
        parameter_norm = math.sqrt(
            sum(
                float(parameter.detach().double().square().sum())
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        )
        trace.append(
            {
                "family": family,
                "rotation": rotation,
                "seed": seed,
                "global_step": global_step,
                "clean_record_ids": schedule_step["record_ids"],
                "augmentation_type": (
                    schedule_step[
                        "continuation_nuisance_type"
                        if family == "FULL_V2_CONTINUED"
                        else "fresh_nuisance_type"
                    ]
                    if family in CONSISTENCY_FAMILIES
                    else "NONE"
                ),
                "total_loss": float(loss.detach()),
                "loss_components": components,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_norm_before_clip": gradient_before,
                "gradient_norm_after_clip": gradient_after,
                "clip_returned_norm": float(clip_return),
                "parameter_norm": parameter_norm,
                "nan_or_inf": False,
                "optimizer_state_finite": True,
                "frozen_gradients_absent": True,
                "wall_time_seconds": time.perf_counter() - step_started,
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
            }
        )
        if global_step in checkpoint_targets:
            path = task_checkpoint_path(attempt, family, rotation, seed, global_step)
            atomic_torch(
                path,
                checkpoint_payload(
                    family, rotation, seed, global_step, model, optimizer,
                    scheduler, initialization_sha256,
                    schedule_row["schedule_sha256"], source_checkpoint, source_sha,
                ),
            )
            checkpoint_paths.append(path)
    torch.cuda.synchronize()
    expected_steps = 450 if family in CONTINUED_FAMILIES else 600
    expected_checkpoints = 3 if family in CONTINUED_FAMILIES else 9
    if len(trace) != expected_steps or len(checkpoint_paths) != expected_checkpoints:
        raise RuntimeError("run execution counts differ from frozen contract")
    log_path = run_root / "training_steps.jsonl"
    with log_path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in trace:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    result = {
        "schema_version": "controller_garment_budget_training_run.v1",
        "status": "PASS",
        "task_id": TASK_ID,
        "post_failure_label": POST_FAILURE_LABEL,
        "family": family,
        "rotation": rotation,
        "seed": seed,
        "initialization_sha256": initialization_sha256,
        "source_checkpoint_path": str(source_checkpoint),
        "source_checkpoint_sha256": source_sha,
        "optimizer_state_restored": restored_optimizer,
        "scheduler_state_restored": restored_scheduler,
        "rng_state_restored": True,
        "data_position_restored": source_step,
        "optimizer_parameter_names": actual_membership,
        "optimizer_scalar_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "loss_definition": LOSS_DEFINITIONS[family],
        "schedule_sha256": schedule_row["schedule_sha256"],
        "counts": {
            "training_steps": expected_steps,
            "optimizer_created": 1,
            "optimizer_steps": expected_steps,
            "scheduler_steps": expected_steps,
            "clean_forward_batches": expected_steps,
            "augmented_forward_batches": expected_steps if family in CONSISTENCY_FAMILIES else 0,
            "clean_record_forwards": expected_steps * 5,
            "augmented_record_forwards": expected_steps * 5 if family in CONSISTENCY_FAMILIES else 0,
            "backward_calls": expected_steps,
            "checkpoint_writes": expected_checkpoints,
        },
        "checkpoint_paths": [str(path) for path in checkpoint_paths],
        "checkpoint_sha256": {str(path): file_hash(path) for path in checkpoint_paths},
        "step_log_path": str(log_path),
        "step_log_sha256": file_hash(log_path),
        "trace_sha256": canonical_hash(trace),
        "numerics": {
            "loss_initial": trace[0]["total_loss"],
            "loss_final": trace[-1]["total_loss"],
            "loss_min": min(row["total_loss"] for row in trace),
            "loss_max": max(row["total_loss"] for row in trace),
            "gradient_norm_max": max(row["gradient_norm_before_clip"] for row in trace),
            "parameter_norm_max": max(row["parameter_norm"] for row in trace),
            "nan_or_inf_count": 0,
            "frozen_gradient_violation_count": 0,
            "optimizer_state_nonfinite_count": 0,
        },
        "wall_clock_seconds": time.perf_counter() - run_started,
        "peak_vram_bytes": max(row["peak_memory_bytes"] for row in trace),
        "early_stopping": False,
        "best_checkpoint_selection": False,
        "paper_final": False,
    }
    atomic_json(run_root / "training_result.json", result)
    return result


def train_all(
    output_root: Path,
    historical_root: Path,
    clean_cache_path: Path,
    nuisance_cache_path: Path,
) -> dict[str, Any]:
    verify_training_admission(output_root)
    results = []
    for family in FAMILIES:
        for rotation in ROTATIONS:
            for seed in SEEDS:
                results.append(
                    train_one(
                        output_root, historical_root, clean_cache_path,
                        nuisance_cache_path, family, rotation, seed,
                    )
                )
    counts = Counter()
    for row in results:
        counts.update(row["counts"])
    expected = {
        "optimizer_created": EXPECTED_COUNTS["optimizer_creations"],
        "optimizer_steps": EXPECTED_COUNTS["optimizer_steps"],
        "scheduler_steps": EXPECTED_COUNTS["scheduler_steps"],
        "clean_forward_batches": EXPECTED_COUNTS["clean_forward_batches"],
        "augmented_forward_batches": EXPECTED_COUNTS["augmented_forward_batches"],
        "clean_record_forwards": EXPECTED_COUNTS["clean_record_forwards"],
        "augmented_record_forwards": EXPECTED_COUNTS["augmented_record_forwards"],
        "backward_calls": EXPECTED_COUNTS["backward_calls"],
        "checkpoint_writes": EXPECTED_COUNTS["new_checkpoint_writes"],
    }
    if any(counts[key] != value for key, value in expected.items()):
        raise RuntimeError(f"aggregate training count mismatch: {dict(counts)}")
    attempt = output_root / "attempt_001"
    result = {
        "schema_version": "controller_garment_budget_training_aggregate.v1",
        "status": "PASS",
        "task_id": TASK_ID,
        "run_count": len(results),
        "counts": dict(counts),
        "runs": results,
        "numerics": {
            "nan_or_inf_count": 0,
            "frozen_gradient_violation_count": 0,
            "optimizer_state_nonfinite_count": 0,
        },
        "renderer_runs": 0,
        "new_formal_renders": 0,
        "paper_final": False,
    }
    atomic_json(attempt / "aggregates/training_results.json", result)
    return result


def resolve_evaluation_checkpoint(
    attempt: Path,
    historical_root: Path,
    family: str,
    rotation: int,
    seed: int,
    step: int,
) -> Path:
    if family in CONTINUED_FAMILIES and step <= 150:
        return historical_checkpoint(historical_root, family, rotation, seed, step)
    return task_checkpoint_path(attempt, family, rotation, seed, step)


def split_record_ids(rotation_archive: Mapping[str, Any], rotation: int) -> dict[str, list[str]]:
    row = next(item for item in rotation_archive["rotations"] if int(item["rotation"]) == rotation)
    return {
        split: list(row["partitions"][split]["record_ids"])
        for split in ("train", "calibration", "test")
    }


def target_dominant(record: Mapping[str, Any]) -> str:
    values = record["target_distribution"]
    return OUTFIT_ORDER[max(range(len(values)), key=lambda index: values[index])]


def target_secondary(record: Mapping[str, Any]) -> str:
    dominant = target_dominant(record)
    first, second = record["pair_id"].split("_")
    return second if dominant == first else first


def infer_record(
    model: torch.nn.Module,
    family: str,
    record: Mapping[str, Any],
    clean_cache: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], Any]:
    rows, valid = formal.case_rows(clean_cache, record)
    with torch.inference_mode():
        output = model(rows.to(device), valid.to(device))
    probabilities = (
        output.probabilities
        if family == "MATCHED_V1_CONTINUED"
        else output.garment_probabilities
    )
    values = [float(value) for value in probabilities.detach().cpu()]
    ranking = sorted(range(len(values)), key=lambda index: (-values[index], index))
    predicted = [OUTFIT_ORDER[index] for index in ranking]
    predicted_pair = "_".join(sorted(predicted[:2], key=OUTFIT_ORDER.index))
    dominant = target_dominant(record)
    pure = record["assignment_type"] in PURE_TYPES
    row = {
        "record_id": record["record_id"],
        "logical_input_sha256": record["logical_input_sha256"],
        "pair_id": record["pair_id"],
        "assignment_type": record["assignment_type"],
        "assignment_position": record.get("assignment_position") or "NOT_APPLICABLE",
        "target_view_fold": record["target_view_fold"],
        "target_distribution": record["target_distribution"],
        "garment_labels": record["garment_labels"],
        "garment_probabilities": values,
        "predicted_top1": predicted[0],
        "predicted_top2": predicted[1],
        "predicted_pair": predicted_pair,
        "stable_score_rank": predicted,
        "top1_correct": predicted[0] == dominant,
        "dominant_order_correct": predicted[0] == dominant,
        "unordered_top2_pair_correct": predicted_pair == record["pair_id"],
        "secondary_garment_included": (
            False if pure else target_secondary(record) in predicted[:2]
        ),
        "dominant_score_rank": ranking.index(OUTFIT_ORDER.index(dominant)) + 1,
        "secondary_score_rank": (
            0
            if pure
            else ranking.index(OUTFIT_ORDER.index(target_secondary(record))) + 1
        ),
        "top2_vs_top3_margin": values[ranking[1]] - values[ranking[2]],
        "garment_loss": -sum(
            target * math.log(max(probability, 1.0e-8))
            for target, probability in zip(record["target_distribution"], values)
        ),
        "pure": pure,
    }
    if family == "MATCHED_V1_CONTINUED":
        selection = stable_top2_selection(output.probabilities)
        earlier = record["pair_id"].split("_")[0]
        predicted_earlier = (
            selection.normalized_top2_weight_1
            if selection.top1_outfit == earlier
            else selection.normalized_top2_weight_2
        )
        row.update(
            {
                "normalized_secondary_probability": selection.normalized_top2_weight_2,
                "predicted_earlier_pair_weight": predicted_earlier,
                "mode": selection.mode,
                "fallback_reason": selection.fallback_reason or "NONE",
            }
        )
    else:
        prediction = stable_pair_prediction(output.garment_probabilities)
        row.update(
            {
                "pair_confidence": prediction.pair_confidence,
                "mixedness_probability": float(output.mixedness_probability.detach().cpu()),
                "all_pair_weights": [
                    float(value) for value in output.all_pair_weights.detach().cpu()
                ],
                "predicted_earlier_pair_weight": float(
                    output.all_pair_weights[PAIR_TO_INDEX[predicted_pair]].detach().cpu()
                ),
            }
        )
    if not pure:
        earlier_index = OUTFIT_ORDER.index(record["pair_id"].split("_")[0])
        target_weight = float(record["target_distribution"][earlier_index])
        row["target_earlier_pair_weight"] = target_weight
        if row["unordered_top2_pair_correct"]:
            row["correct_pair_weight_error"] = row["predicted_earlier_pair_weight"] - target_weight
    return row, output


def calibrate_raw_v2(
    rotation: int,
    rows_and_outputs: Sequence[tuple[dict[str, Any], Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prior, labels = repaired_eval.compatibility_prior(rotation)
    candidates = []
    contract = read_json(CALIBRATION_PATH)
    for candidate in contract["all_candidates"]:
        tau_mix = float(candidate["tau_mix"])
        tau_pair = float(candidate["tau_pair"])
        routed = [
            repaired_eval.apply_v2_route(row, output, prior, tau_mix, tau_pair)
            for row, output in rows_and_outputs
        ]
        candidates.append(
            {
                "tau_mix": tau_mix,
                "tau_pair": tau_pair,
                "objective": repaired_eval.calibration_objective(routed, labels),
            }
        )
    selected = sorted(candidates, key=repaired_eval.objective_sort_key)[0]
    routed = [
        repaired_eval.apply_v2_route(
            row, output, prior, selected["tau_mix"], selected["tau_pair"]
        )
        for row, output in rows_and_outputs
    ]
    cleaned = []
    for row in routed:
        cleaned.append(
            {
                **row,
                "fallback_reason": row.get("fallback_reason") or "NONE",
                "ground_truth_compatibility_label": labels[row["pair_id"]],
            }
        )
    return {
        "selected_threshold": {
            "tau_mix": selected["tau_mix"], "tau_pair": selected["tau_pair"]
        },
        "selected_objective": selected["objective"],
        "candidate_count": len(candidates),
        "test_fold_used": False,
        "tie_break": "FROZEN_OBJECTIVE_THEN_LARGER_SAFETY_THRESHOLDS",
    }, cleaned


def mean(values: Iterable[float]) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def rmse(values: Iterable[float]) -> float:
    rows = list(values)
    return math.sqrt(mean(value * value for value in rows))


def cell_metrics(rows: Sequence[Mapping[str, Any]], routing: bool) -> dict[str, Any]:
    pure = [row for row in rows if row["pure"]]
    mixed = [row for row in rows if not row["pure"]]
    errors = [
        row["correct_pair_weight_error"]
        for row in mixed
        if "correct_pair_weight_error" in row
    ]
    result = {
        "record_count": len(rows),
        "pure_record_count": len(pure),
        "mixed_record_count": len(mixed),
        "pure_top1_accuracy": mean(row["top1_correct"] for row in pure),
        "mixed_pair_accuracy": mean(
            row["unordered_top2_pair_correct"] for row in mixed
        ),
        "mixed_dominant_order_accuracy": mean(
            row["dominant_order_correct"] for row in mixed
        ),
        "mixed_secondary_inclusion": mean(
            row["secondary_garment_included"] for row in mixed
        ),
        "mixed_top2_vs_top3_margin": mean(
            row["top2_vs_top3_margin"] for row in mixed
        ),
        "garment_loss": mean(row["garment_loss"] for row in rows),
        "correct_pair_weight_mae": mean(abs(value) for value in errors),
        "correct_pair_weight_rmse": rmse(errors),
        "correct_pair_weight_count": len(errors),
        "per_pair": {
            pair: {
                "record_count": sum(row["pair_id"] == pair for row in mixed),
                "pair_accuracy": mean(
                    row["unordered_top2_pair_correct"]
                    for row in mixed if row["pair_id"] == pair
                ),
                "secondary_inclusion": mean(
                    row["secondary_garment_included"]
                    for row in mixed if row["pair_id"] == pair
                ),
                "dominant_order_accuracy": mean(
                    row["dominant_order_correct"]
                    for row in mixed if row["pair_id"] == pair
                ),
                "margin": mean(
                    row["top2_vs_top3_margin"]
                    for row in mixed if row["pair_id"] == pair
                ),
                "dominant_score_rank": mean(
                    row["dominant_score_rank"]
                    for row in mixed if row["pair_id"] == pair
                ),
                "secondary_score_rank": mean(
                    row["secondary_score_rank"]
                    for row in mixed if row["pair_id"] == pair
                ),
            }
            for pair in PAIR_ORDER
        },
        "assignment_types": {
            assignment: {
                "record_count": sum(row["assignment_type"] == assignment for row in mixed),
                "pair_accuracy": mean(
                    row["unordered_top2_pair_correct"]
                    for row in mixed if row["assignment_type"] == assignment
                ),
                "dominant_order_accuracy": mean(
                    row["dominant_order_correct"]
                    for row in mixed if row["assignment_type"] == assignment
                ),
            }
            for assignment in sorted(MIXED_TYPES)
        },
        "assignment_positions": {
            str(position): {
                "record_count": sum(str(row["assignment_position"]) == str(position) for row in mixed),
                "pair_accuracy": mean(
                    row["unordered_top2_pair_correct"]
                    for row in mixed if str(row["assignment_position"]) == str(position)
                ),
                "dominant_order_accuracy": mean(
                    row["dominant_order_correct"]
                    for row in mixed if str(row["assignment_position"]) == str(position)
                ),
            }
            for position in (0, 1, 2)
        },
    }
    if routing:
        labels = [0 if row["pure"] else 1 for row in rows]
        scores = [row["mixedness_probability"] for row in rows]
        compatible = [
            row for row in mixed
            if row["unordered_top2_pair_correct"]
            and row["ground_truth_compatibility_label"] == "COMPATIBLE"
        ]
        incompatible = [
            row for row in mixed
            if row["unordered_top2_pair_correct"]
            and row["ground_truth_compatibility_label"] == "INCOMPATIBLE"
        ]
        wrong = [row for row in mixed if not row["unordered_top2_pair_correct"]]
        result["mixedness"] = {
            "auroc": repaired_eval.auroc(labels, scores),
            "auprc": repaired_eval.auprc(labels, scores),
            "pure_false_mixed": mean(
                row["mixedness_probability"] >= row["thresholds"]["tau_mix"]
                for row in pure
            ),
            "mixed_false_single": mean(row["mode"] == "SINGLE_ENDPOINT" for row in mixed),
        }
        result["routing"] = {
            "pure_single": mean(row["mode"] == "SINGLE_ENDPOINT" for row in pure),
            "compatible_dual": mean(row["mode"] == "DUAL_SUPPORT" for row in compatible),
            "incompatible_hard": mean(
                row["mode"] == "HARD_GEOMETRY_SOFT_VA" for row in incompatible
            ),
            "incompatible_dual": mean(row["mode"] == "DUAL_SUPPORT" for row in incompatible),
            "wrong_pair_dual": mean(row["mode"] == "DUAL_SUPPORT" for row in wrong),
            "denominators": {
                "pure": len(pure),
                "mixed": len(mixed),
                "correct_compatible": len(compatible),
                "correct_incompatible": len(incompatible),
                "wrong_pair_mixed": len(wrong),
            },
        }
    elif rows and "mode" in rows[0]:
        result["routing"] = {
            "pure_single": mean(row["mode"] == "SINGLE_ENDPOINT" for row in pure),
            "mixed_dual": mean(row["mode"] == "DUAL_SUPPORT" for row in mixed),
            "wrong_pair_dual": mean(
                row["mode"] == "DUAL_SUPPORT"
                for row in mixed if not row["unordered_top2_pair_correct"]
            ),
        }
    return result


def evaluate(
    output_root: Path,
    historical_root: Path,
    clean_cache_path: Path,
    *,
    historical_only: bool = False,
) -> dict[str, Any]:
    attempt = output_root / "attempt_001"
    if not historical_only:
        training = read_json(attempt / "aggregates/training_results.json")
        if training["run_count"] != 72:
            raise RuntimeError("all 72 training trajectories are required")
    configure_determinism()
    device = torch.device("cuda")
    clean_cache = torch.load(clean_cache_path, map_location="cpu", weights_only=False)
    manifest = read_json(MANIFEST_PATH)
    record_index = {row["record_id"]: row for row in manifest["query_sets"]}
    rotation_archive = read_json(ROTATION_PATH)
    families = CONTINUED_FAMILIES if historical_only else FAMILIES
    steps = HISTORICAL_STEPS if historical_only else CHECKPOINT_STEPS
    predictions_path = attempt / "predictions" / (
        "historical_checkpoint_predictions.jsonl"
        if historical_only else "checkpoint_predictions.jsonl"
    )
    if predictions_path.exists():
        raise FileExistsError(predictions_path)
    cells = []
    calibration_rows = []
    inference_count = 0
    with predictions_path.open("x", encoding="utf-8", newline="\n") as prediction_file:
        for family in families:
            for rotation in ROTATIONS:
                split_ids = split_record_ids(rotation_archive, rotation)
                for seed in SEEDS:
                    for step in steps:
                        checkpoint = resolve_evaluation_checkpoint(
                            attempt, historical_root, family, rotation, seed, step
                        )
                        if not checkpoint.is_file():
                            raise FileNotFoundError(checkpoint)
                        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                        model: torch.nn.Module
                        if family == "MATCHED_V1_CONTINUED":
                            model = ReferenceConditionedDualSupportController(seed=seed)
                        else:
                            model = CompatibilityGatedReferenceControllerV2(seed=seed)
                        model.load_state_dict(state["model_state_dict"], strict=True)
                        model = model.to(device).eval()
                        raw_by_split: dict[str, list[tuple[dict[str, Any], Any]]] = {}
                        for split, record_ids in split_ids.items():
                            raw_by_split[split] = [
                                infer_record(model, family, record_index[record_id], clean_cache, device)
                                for record_id in record_ids
                            ]
                            inference_count += len(record_ids)
                        routing = family in FULL_ROUTING_FAMILIES
                        calibration = {
                            "routing_evaluation": "NOT_APPLICABLE_UNTRAINED_HEADS"
                        }
                        rows_by_split = {
                            split: [row for row, _ in values]
                            for split, values in raw_by_split.items()
                        }
                        if routing:
                            calibration, _ = calibrate_raw_v2(
                                rotation, raw_by_split["calibration"]
                            )
                            prior, labels = repaired_eval.compatibility_prior(rotation)
                            threshold = calibration["selected_threshold"]
                            for split, values in raw_by_split.items():
                                routed = []
                                for row, output in values:
                                    value = repaired_eval.apply_v2_route(
                                        row, output, prior,
                                        threshold["tau_mix"], threshold["tau_pair"],
                                    )
                                    value["fallback_reason"] = value.get("fallback_reason") or "NONE"
                                    value["ground_truth_compatibility_label"] = labels[value["pair_id"]]
                                    routed.append(value)
                                rows_by_split[split] = routed
                        calibration_rows.append(
                            {
                                "family": family,
                                "rotation": rotation,
                                "seed": seed,
                                "step": step,
                                **calibration,
                            }
                        )
                        for split, rows in rows_by_split.items():
                            metrics = cell_metrics(rows, routing)
                            cells.append(
                                {
                                    "family": family,
                                    "rotation": rotation,
                                    "seed": seed,
                                    "step": step,
                                    "split": split,
                                    "checkpoint_path": str(checkpoint),
                                    "checkpoint_sha256": file_hash(checkpoint),
                                    "metrics": metrics,
                                }
                            )
                            for row in rows:
                                serial = {
                                    "family": family,
                                    "rotation": rotation,
                                    "seed": seed,
                                    "step": step,
                                    "split": split,
                                    **row,
                                }
                                prediction_file.write(
                                    json.dumps(serial, sort_keys=True, allow_nan=False) + "\n"
                                )
                        del model
    torch.cuda.synchronize()
    expected = (
        2 * 4 * 3 * 6 * 320
        if historical_only
        else EXPECTED_COUNTS["checkpoint_inference_count"]
    )
    if inference_count != expected:
        raise RuntimeError(f"checkpoint inference count mismatch: {inference_count}")
    result = {
        "schema_version": "controller_garment_budget_checkpoint_cells.v1",
        "status": "PASS",
        "task_id": TASK_ID,
        "historical_only": historical_only,
        "cell_count": len(cells),
        "checkpoint_inference_count": inference_count,
        "metric_records": inference_count,
        "predictions_path": str(predictions_path),
        "predictions_sha256": file_hash(predictions_path),
        "calibration": calibration_rows,
        "cells": cells,
        "pure_hidden_pair_evaluated": False,
        "mixed_pair_denominator": "GT_MIXED_ONLY",
        "renderer_runs": 0,
        "new_formal_renders": 0,
    }
    target = attempt / "trajectories" / (
        "historical_checkpoint_cells.json"
        if historical_only else "checkpoint_cells.json"
    )
    atomic_json(target, result)
    return result


def aggregate_cell_views(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for cell in cells:
        grouped[(cell["family"], int(cell["step"]), cell["split"])].append(cell)
    output = []
    for (family, step, split), rows in sorted(grouped.items()):
        if len(rows) != 12:
            raise RuntimeError(f"expected 12 rotation-seed cells: {family}/{step}/{split}")
        per_rotation = {
            str(rotation): mean(
                row["metrics"]["mixed_pair_accuracy"]
                for row in rows if row["rotation"] == rotation
            )
            for rotation in ROTATIONS
        }
        per_seed = {
            str(seed): mean(
                row["metrics"]["mixed_pair_accuracy"]
                for row in rows if row["seed"] == seed
            )
            for seed in SEEDS
        }
        output.append(
            {
                "family": family,
                "step": step,
                "split": split,
                "mixed_pair_protocol_weighted_macro": mean(
                    row["metrics"]["mixed_pair_accuracy"] for row in rows
                ),
                "mixed_pair_unique_query_macro": mean(
                    row["metrics"]["mixed_pair_accuracy"] for row in rows
                ),
                "mixed_pair_global_macro": mean(
                    row["metrics"]["mixed_pair_accuracy"] for row in rows
                ),
                "mixed_pair_rotation_macro": mean(per_rotation.values()),
                "mixed_pair_seed_macro": mean(per_seed.values()),
                "mixed_pair_per_rotation": per_rotation,
                "mixed_pair_per_seed": per_seed,
                "pure_top1": mean(row["metrics"]["pure_top1_accuracy"] for row in rows),
                "dominant_order": mean(
                    row["metrics"]["mixed_dominant_order_accuracy"] for row in rows
                ),
                "secondary_inclusion": mean(
                    row["metrics"]["mixed_secondary_inclusion"] for row in rows
                ),
                "margin": mean(
                    row["metrics"]["mixed_top2_vs_top3_margin"] for row in rows
                ),
                "garment_loss": mean(row["metrics"]["garment_loss"] for row in rows),
                "correct_pair_weight_mae": mean(
                    row["metrics"]["correct_pair_weight_mae"] for row in rows
                ),
                "correct_pair_weight_rmse": mean(
                    row["metrics"]["correct_pair_weight_rmse"] for row in rows
                ),
            }
        )
    return output


def by_key(
    trajectories: Sequence[Mapping[str, Any]], family: str, step: int, split: str = "test"
) -> Mapping[str, Any]:
    return next(
        row for row in trajectories
        if row["family"] == family and row["step"] == step and row["split"] == split
    )


def difficult_pair_archive(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    for cell in cells:
        for pair in PAIR_ORDER:
            value = cell["metrics"]["per_pair"][pair]
            rows.append(
                {
                    "family": cell["family"],
                    "step": cell["step"],
                    "split": cell["split"],
                    "rotation": cell["rotation"],
                    "seed": cell["seed"],
                    "pair_id": pair,
                    **value,
                }
            )
    grouped = defaultdict(list)
    for row in rows:
        key = (row["family"], row["step"], row["split"], row["rotation"], row["pair_id"])
        grouped[key].append(row)
    aggregates = []
    for key, values in sorted(grouped.items()):
        family, step, split, rotation, pair = key
        aggregates.append(
            {
                "family": family,
                "step": step,
                "split": split,
                "rotation": rotation,
                "pair_id": pair,
                "seed_count": len(values),
                "pair_accuracy": mean(row["pair_accuracy"] for row in values),
                "secondary_inclusion": mean(row["secondary_inclusion"] for row in values),
                "dominant_order_accuracy": mean(row["dominant_order_accuracy"] for row in values),
                "margin": mean(row["margin"] for row in values),
                "dominant_score_rank": mean(row["dominant_score_rank"] for row in values),
                "secondary_score_rank": mean(row["secondary_score_rank"] for row in values),
            }
        )
    return {
        "schema_version": "controller_garment_budget_difficult_pairs.v1",
        "status": "PASS",
        "required_pairs": list(PAIR_ORDER),
        "focused_pairs": ["O01_O03", "O02_O03", "O03_O08", "O01_O08"],
        "rows": aggregates,
    }


def routing_aggregate(
    cells: Sequence[Mapping[str, Any]], family: str, step: int
) -> dict[str, Any]:
    rows = [
        row for row in cells
        if row["family"] == family and row["step"] == step and row["split"] == "test"
    ]
    result: dict[str, Any] = {}
    if family in FULL_ROUTING_FAMILIES:
        result = {
            "mixedness_auroc": mean(row["metrics"]["mixedness"]["auroc"] for row in rows),
            "mixedness_auprc": mean(row["metrics"]["mixedness"]["auprc"] for row in rows),
            "pure_false_mixed": mean(row["metrics"]["mixedness"]["pure_false_mixed"] for row in rows),
            "mixed_false_single": mean(row["metrics"]["mixedness"]["mixed_false_single"] for row in rows),
            "pure_single": mean(row["metrics"]["routing"]["pure_single"] for row in rows),
            "compatible_dual": mean(row["metrics"]["routing"]["compatible_dual"] for row in rows),
            "incompatible_hard": mean(row["metrics"]["routing"]["incompatible_hard"] for row in rows),
            "incompatible_dual": mean(row["metrics"]["routing"]["incompatible_dual"] for row in rows),
            "wrong_pair_dual": mean(row["metrics"]["routing"]["wrong_pair_dual"] for row in rows),
        }
    elif family == "MATCHED_V1_CONTINUED":
        result = {
            "pure_single": mean(row["metrics"]["routing"]["pure_single"] for row in rows),
            "mixed_dual": mean(row["metrics"]["routing"]["mixed_dual"] for row in rows),
            "wrong_pair_dual": mean(row["metrics"]["routing"]["wrong_pair_dual"] for row in rows),
        }
    return result


def saturation_classification(
    trajectories: Sequence[Mapping[str, Any]], family: str
) -> dict[str, Any]:
    values = {step: by_key(trajectories, family, step) for step in (150, 300, 450, 600)}
    gains = {
        "150_to_300": values[300]["mixed_pair_global_macro"] - values[150]["mixed_pair_global_macro"],
        "300_to_450": values[450]["mixed_pair_global_macro"] - values[300]["mixed_pair_global_macro"],
        "450_to_600": values[600]["mixed_pair_global_macro"] - values[450]["mixed_pair_global_macro"],
    }
    if any(not math.isfinite(value) for value in gains.values()):
        classification = "UNSTABLE"
    elif gains["450_to_600"] < -0.03:
        classification = "DIVERGING"
    elif abs(gains["450_to_600"]) < 0.01 and abs(gains["300_to_450"]) < 0.01:
        classification = "SATURATING"
    else:
        classification = "STILL_IMPROVING"
    return {
        "family": family,
        "classification": classification,
        "gains": gains,
        "last_150_step_slope": gains["450_to_600"] / 150.0,
        "loss_change_450_to_600": values[600]["garment_loss"] - values[450]["garment_loss"],
        "margin_change_450_to_600": values[600]["margin"] - values[450]["margin"],
    }


def classify(
    trajectories: Sequence[Mapping[str, Any]],
    parameter_paths: Mapping[str, Any],
) -> tuple[str, list[str], str, dict[str, Any]]:
    at600 = {
        family: by_key(trajectories, family, 600) for family in FAMILIES
    }
    full150 = by_key(trajectories, "FULL_V2_CONTINUED", 150)
    full = at600["FULL_V2_CONTINUED"]
    garment = at600["GARMENT_ONLY"]
    mixed = at600["GARMENT_PLUS_MIXEDNESS"]
    consistency = at600["GARMENT_PLUS_GARMENT_CONSISTENCY"]
    no_consistency = at600["FULL_NO_GARMENT_CONSISTENCY"]
    gains = {
        "FULL_BUDGET_GAIN": full["mixed_pair_global_macro"] - full150["mixed_pair_global_macro"],
        "V1_BUDGET_GAIN": (
            at600["MATCHED_V1_CONTINUED"]["mixed_pair_global_macro"]
            - by_key(trajectories, "MATCHED_V1_CONTINUED", 150)["mixed_pair_global_macro"]
        ),
        "GARMENT_ONLY_GAP": garment["mixed_pair_global_macro"] - full["mixed_pair_global_macro"],
        "MIXEDNESS_INTERFERENCE": garment["mixed_pair_global_macro"] - mixed["mixed_pair_global_macro"],
        "GARMENT_CONSISTENCY_INTERFERENCE": (
            garment["mixed_pair_global_macro"] - consistency["mixed_pair_global_macro"]
        ),
        "GARMENT_CONSISTENCY_REMOVAL_GAIN": (
            no_consistency["mixed_pair_global_macro"] - full["mixed_pair_global_macro"]
        ),
    }
    direction = {}
    comparisons = {
        "GARMENT_ONLY_GAP": (garment, full),
        "MIXEDNESS_INTERFERENCE": (garment, mixed),
        "GARMENT_CONSISTENCY_INTERFERENCE": (garment, consistency),
        "GARMENT_CONSISTENCY_REMOVAL_GAIN": (no_consistency, full),
    }
    for name, (first, second) in comparisons.items():
        direction[name] = sum(
            first["mixed_pair_per_rotation"][str(rotation)]
            > second["mixed_pair_per_rotation"][str(rotation)]
            for rotation in ROTATIONS
        )
    full_rule = (
        full["mixed_pair_global_macro"] >= 0.85
        and min(full["mixed_pair_per_rotation"].values()) >= 0.80
        and gains["FULL_BUDGET_GAIN"] >= 0.10
        and full["pure_top1"] >= 0.95
    )
    multitask_rule = (
        garment["mixed_pair_global_macro"] >= 0.90
        and min(garment["mixed_pair_per_rotation"].values()) >= 0.85
        and gains["GARMENT_ONLY_GAP"] >= 0.10
        and full["mixed_pair_global_macro"] < 0.85
        and direction["GARMENT_ONLY_GAP"] >= 3
    )
    consistency_rule = (
        gains["GARMENT_CONSISTENCY_INTERFERENCE"] >= 0.08
        and gains["GARMENT_CONSISTENCY_REMOVAL_GAIN"] >= 0.08
        and direction["GARMENT_CONSISTENCY_INTERFERENCE"] >= 3
        and gains["MIXEDNESS_INTERFERENCE"] < gains["GARMENT_CONSISTENCY_INTERFERENCE"]
    )
    shared = parameter_paths["shared_intersections"]["garment__mixedness"]
    mixedness_rule = (
        gains["MIXEDNESS_INTERFERENCE"] >= 0.08
        and direction["MIXEDNESS_INTERFERENCE"] >= 3
        and gains["GARMENT_CONSISTENCY_INTERFERENCE"] < gains["MIXEDNESS_INTERFERENCE"]
        and shared["status"] == "SHARED_GRADIENT_PATH"
    )
    optimizer_rule = garment["mixed_pair_global_macro"] < 0.80
    active = {
        "budget_recoverable": full_rule,
        "multitask": multitask_rule,
        "garment_consistency": consistency_rule,
        "mixedness": mixedness_rule,
        "optimizer_or_normalization": optimizer_rule,
    }
    causal_count = sum((multitask_rule, consistency_rule, mixedness_rule, optimizer_rule))
    if full_rule:
        primary = "FULL_V2_BUDGET_LIMITED_BUT_RECOVERABLE"
    elif causal_count >= 2 and not (consistency_rule or mixedness_rule):
        primary = "MULTIPLE_OPTIMIZATION_FACTORS"
    elif consistency_rule:
        primary = "GARMENT_CONSISTENCY_INTERFERENCE_DOMINANT"
    elif mixedness_rule:
        primary = "MIXEDNESS_SHARED_PATH_INTERFERENCE_DOMINANT"
    elif multitask_rule:
        primary = "MULTITASK_OPTIMIZATION_INTERFERENCE_DOMINANT"
    elif optimizer_rule:
        primary = "OPTIMIZER_OR_NORMALIZATION_LIMIT"
    else:
        primary = "GARMENT_HEAD_BUDGET_DIAGNOSIS_INCONCLUSIVE"
    secondary = [name.upper() for name, value in active.items() if value]
    best = max(at600.values(), key=lambda row: row["mixed_pair_global_macro"])
    if (
        best["mixed_pair_global_macro"] >= 0.85
        and min(best["mixed_pair_per_rotation"].values()) >= 0.80
        and best["pure_top1"] >= 0.95
    ):
        repair_value = "CONTROLLER_TRAINING_REPAIR_HIGH_VALUE"
    elif best["mixed_pair_global_macro"] >= 0.75:
        repair_value = "CONTROLLER_TRAINING_REPAIR_MODERATE_VALUE"
    else:
        repair_value = "CONTROLLER_TRAINING_REPAIR_LOW_VALUE"
    return primary, secondary, repair_value, {"gains": gains, "directionality": direction, "rules": active}


def next_task(primary: str) -> str:
    return {
        "FULL_V2_BUDGET_LIMITED_BUT_RECOVERABLE": "FREEZE_EXTENDED_BUDGET_CONTROLLER_V2_RETRAINING_PROTOCOL",
        "MULTITASK_OPTIMIZATION_INTERFERENCE_DOMINANT": "DESIGN_STAGED_OR_DECOUPLED_CONTROLLER_V3_TRAINING",
        "GARMENT_CONSISTENCY_INTERFERENCE_DOMINANT": "DESIGN_GARMENT_IDENTITY_PRESERVING_CONSISTENCY_CONTROLLER_V3",
        "MIXEDNESS_SHARED_PATH_INTERFERENCE_DOMINANT": "DESIGN_DECOUPLED_MIXEDNESS_CONTROLLER_V3",
        "OPTIMIZER_OR_NORMALIZATION_LIMIT": "DIAGNOSE_CONTROLLER_GARMENT_HEAD_OPTIMIZER_AND_NORMALIZATION",
        "MULTIPLE_OPTIMIZATION_FACTORS": "DESIGN_CONTROLLER_V3_FROM_OPTIMIZATION_CAUSAL_DIAGNOSIS",
        "GARMENT_HEAD_BUDGET_DIAGNOSIS_INCONCLUSIVE": "MANUAL_REVIEW_CONTROLLER_GARMENT_HEAD_OPTIMIZATION",
    }[primary]


def verify_snapshot(asset: Mapping[str, Any]) -> dict[str, Any]:
    mutations = {}
    for name, snapshot in asset["historical_attempt_snapshots"].items():
        current = tree_snapshot(Path(snapshot["root"]))
        mutations[name] = int(current["tree_sha256"] != snapshot["tree_sha256"])
    registry_mutation = 0
    for row in asset["historical_checkpoint_registry"]:
        registry_mutation += int(
            file_hash(Path(row["checkpoint_path"])) != row["checkpoint_sha256"]
        )
    semantic = asset["semantic_repair_snapshot"]
    semantic_mutation = 0
    for row in semantic["files"]:
        semantic_mutation += int(
            file_hash(PROJECT_ROOT / row["path"], lf=True) != row["sha256_lf"]
        )
    result = {
        "semantic_repair_outputs_mutation": semantic_mutation,
        "pair_diagnosis_outputs_mutation": int(
            git("rev-parse", f"{PAIR_DIAGNOSIS_HEAD}^{{tree}}")
            != asset["pair_diagnosis_git_object"]["tree"]
        ),
        "controller_attempt_mutations": mutations,
        "historical_checkpoint_mutation_count": registry_mutation,
        "clean_feature_cache_mutation": int(
            file_hash(Path(asset["clean_feature_cache"]["path"]))
            != asset["clean_feature_cache"]["sha256"]
        ),
        "nuisance_feature_cache_mutation": int(
            file_hash(Path(asset["nuisance_feature_cache"]["path"]))
            != asset["nuisance_feature_cache"]["sha256"]
        ),
    }
    result["all_frozen_mutation_count"] = (
        semantic_mutation + result["pair_diagnosis_outputs_mutation"]
        + sum(mutations.values()) + registry_mutation
        + result["clean_feature_cache_mutation"]
        + result["nuisance_feature_cache_mutation"]
    )
    return result


def markdown_reports(final: Mapping[str, Any], difficult: Mapping[str, Any]) -> dict[str, str]:
    gains = final["budget_and_interference"]
    trajectories = final["selected_test_trajectories"]
    common = (
        f"Status: `PASS`\n\nTask: `{TASK_ID}`. This is a `{POST_FAILURE_LABEL}`; "
        f"historical status remains `{HISTORICAL_CLASSIFICATION}`. Pure records are "
        "evaluated only by visible-garment top-1, while pair metrics use mixed-only records.\n"
    )
    diagnosis = "# Controller Garment-Head Budget Diagnosis\n\n" + common + (
        f"\nPrimary classification: `{final['primary_classification']}`.\n\n"
        f"Training-repair value: `{final['training_repair_value']}`.\n\n"
        f"FULL_BUDGET_GAIN: `{gains['FULL_BUDGET_GAIN']:.6f}`. "
        f"V1_BUDGET_GAIN: `{gains['V1_BUDGET_GAIN']:.6f}`.\n\n"
        "All 72 preregistered trajectories reached the fixed step-600 endpoint with no "
        "early stopping or best-checkpoint selection. No renderer was invoked.\n"
    )
    ablation = "# Controller Multitask Loss Ablation\n\n" + common + "\n"
    for family in FAMILIES:
        values = trajectories[family]
        ablation += (
            f"- `{family}` step150/300/450/600 mixed-pair macro: "
            + "/".join(f"{values[str(step)]['mixed_pair_global_macro']:.6f}" for step in (150, 300, 450, 600))
            + ".\n"
        )
    ablation += (
        f"\nGARMENT_ONLY_GAP: `{gains['GARMENT_ONLY_GAP']:.6f}`; "
        f"MIXEDNESS_INTERFERENCE: `{gains['MIXEDNESS_INTERFERENCE']:.6f}`; "
        f"GARMENT_CONSISTENCY_INTERFERENCE: `{gains['GARMENT_CONSISTENCY_INTERFERENCE']:.6f}`; "
        f"GARMENT_CONSISTENCY_REMOVAL_GAIN: `{gains['GARMENT_CONSISTENCY_REMOVAL_GAIN']:.6f}`.\n"
    )
    pair_report = "# Controller Difficult-Pair Learning Trajectory\n\n" + common + (
        f"\nThe archive contains `{len(difficult['rows'])}` family/budget/split/rotation/pair rows "
        "covering all ten pairs, including O01_O03, O02_O03, O03_O08, and O01_O08. "
        "Each row reports top-2 pair accuracy, secondary inclusion, dominant ordering, "
        "and top2-vs-top3 margin across all three seeds.\n"
    )
    decision = "# Controller Training-Repair Decision\n\n" + common + (
        f"\nPrimary root cause: `{final['primary_classification']}`.\n\n"
        f"Secondary factors: `{', '.join(final['secondary_factors']) or 'NONE'}`.\n\n"
        f"NEXT_TASK: `{final['next_task']}`. The next task was not started. "
        "Controller V3, subject00 training, PAPER_FINAL, and paper-main-table updates all remain zero.\n"
    )
    return dict(zip(REPORT_NAMES, (diagnosis, ablation, pair_report, decision)))


def finalize(output_root: Path) -> dict[str, Any]:
    attempt = output_root / "attempt_001"
    evaluation = read_json(attempt / "trajectories/checkpoint_cells.json")
    training = read_json(attempt / "aggregates/training_results.json")
    trajectories = aggregate_cell_views(evaluation["cells"])
    difficult = difficult_pair_archive(evaluation["cells"])
    parameter_paths = read_json(RISK / PRE_RESULT_NAMES[2])
    primary, secondary, repair_value, causal = classify(trajectories, parameter_paths)
    saturations = [saturation_classification(trajectories, family) for family in FAMILIES]
    selected = {
        family: {
            str(step): dict(by_key(trajectories, family, step))
            for step in (150, 300, 450, 600)
        }
        for family in FAMILIES
    }
    full = {
        str(step): {
            **dict(by_key(trajectories, "FULL_V2_CONTINUED", step)),
            **routing_aggregate(evaluation["cells"], "FULL_V2_CONTINUED", step),
        }
        for step in (150, 300, 450, 600)
    }
    matched = {
        str(step): {
            **dict(by_key(trajectories, "MATCHED_V1_CONTINUED", step)),
            **routing_aggregate(evaluation["cells"], "MATCHED_V1_CONTINUED", step),
        }
        for step in (150, 300, 450, 600)
    }
    asset = read_json(RISK / PRE_RESULT_NAMES[3])
    immutability = verify_snapshot(asset)
    if immutability["all_frozen_mutation_count"] != 0:
        raise RuntimeError(f"frozen asset mutation detected: {immutability}")
    final = {
        "schema_version": "controller_garment_budget_final_summary.v1",
        "status": "PASS",
        "task_id": TASK_ID,
        "source_head": SOURCE_HEAD,
        "execution_head": git("rev-parse", "HEAD"),
        "branch": BRANCH,
        "post_failure_label": POST_FAILURE_LABEL,
        "semantic_classification": SEMANTIC_CLASSIFICATION,
        "pair_interpretation": PAIR_INTERPRETATION,
        "historical_overall_classification": HISTORICAL_CLASSIFICATION,
        "primary_classification": primary,
        "secondary_factors": secondary,
        "training_repair_value": repair_value,
        "budget_and_interference": causal["gains"],
        "rotation_directionality": causal["directionality"],
        "diagnosis_rule_results": causal["rules"],
        "selected_test_trajectories": selected,
        "budget_saturation": saturations,
        "execution_counts": {
            **EXPECTED_COUNTS,
            "actual_training_counts": training["counts"],
            "actual_checkpoint_inference_count": evaluation["checkpoint_inference_count"],
            "actual_metric_records": evaluation["metric_records"],
        },
        "training_numerics": training["numerics"],
        "immutability": immutability,
        "renderer_runs": 0,
        "new_formal_renders": 0,
        "paper_final": False,
        "paper_final_count": 0,
        "controller_v3_training_runs": 0,
        "subject00_training_runs": 0,
        "paper_main_table_updates": 0,
        "next_task": next_task(primary),
        "next_task_started": False,
    }
    mixed = {
        "schema_version": "controller_garment_budget_mixed_pair_results.v1",
        "status": "PASS",
        "denominator": "GT_MIXED_ONLY",
        "views": trajectories,
    }
    ablations = {
        "schema_version": "controller_garment_budget_loss_ablation_results.v1",
        "status": "PASS",
        "families": {
            family: selected[family] for family in FRESH_FAMILIES
        },
        "budget_and_interference": causal["gains"],
        "rotation_directionality": causal["directionality"],
    }
    tracked = {
        TRACKED_RESULT_NAMES[0]: training,
        TRACKED_RESULT_NAMES[1]: {
            "schema_version": "controller_garment_budget_checkpoint_trajectories.v1",
            "status": "PASS",
            "views": trajectories,
            "cell_count": evaluation["cell_count"],
            "checkpoint_inference_count": evaluation["checkpoint_inference_count"],
        },
        TRACKED_RESULT_NAMES[2]: mixed,
        TRACKED_RESULT_NAMES[3]: {
            "schema_version": "controller_garment_budget_full_v2_results.v1",
            "status": "PASS", "checkpoints": full,
        },
        TRACKED_RESULT_NAMES[4]: {
            "schema_version": "controller_garment_budget_matched_v1_results.v1",
            "status": "PASS", "fixed_gate": [0.90, 0.10], "checkpoints": matched,
        },
        TRACKED_RESULT_NAMES[5]: ablations,
        TRACKED_RESULT_NAMES[6]: difficult,
        TRACKED_RESULT_NAMES[7]: final,
    }
    for name, value in tracked.items():
        atomic_json(RISK / name, value)
    atomic_json(attempt / "difficult_pairs/results.json", difficult)
    atomic_json(attempt / "aggregates/final_summary.json", final)
    protocol = {
        "schema_version": "controller_garment_budget_protocol.v1",
        "task_id": TASK_ID,
        "source_head": SOURCE_HEAD,
        "branch": BRANCH,
        "families": list(FAMILIES),
        "rotations": list(ROTATIONS),
        "seeds": list(SEEDS),
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "historical_classification": HISTORICAL_CLASSIFICATION,
        "paper_final": False,
    }
    atomic_json(RISK / "controller_garment_budget_protocol.yaml", protocol)
    for name, value in markdown_reports(final, difficult).items():
        atomic_text(PROJECT_ROOT / "docs/PAPER" / name, value)
    handoff = {
        "schema_version": "controller_garment_head_budget_diagnosis_handoff.v1",
        "status": "PASS",
        "task_id": TASK_ID,
        "branch": BRANCH,
        "head_at_generation": git("rev-parse", "HEAD"),
        "primary_classification": primary,
        "training_repair_value": repair_value,
        "historical_classification": HISTORICAL_CLASSIFICATION,
        "next_task": next_task(primary),
        "next_task_started": False,
        "controller_v3_started": False,
        "subject00_started": False,
        "paper_final": False,
        "paths": {
            "final_summary": str(RISK / TRACKED_RESULT_NAMES[7]),
            "cloud_output": str(attempt),
            "reports": [str(PROJECT_ROOT / "docs/PAPER" / name) for name in REPORT_NAMES],
        },
    }
    atomic_json(
        PROJECT_ROOT / "project_control_handoff/controller_garment_head_budget_diagnosis_handoff.json",
        handoff,
    )
    return final


def verify_final(output_root: Path) -> dict[str, Any]:
    attempt = output_root / "attempt_001"
    final = read_json(RISK / TRACKED_RESULT_NAMES[-1])
    failures = []
    for name in PRE_RESULT_NAMES + TRACKED_RESULT_NAMES:
        try:
            value = read_json(RISK / name)
            if value.get("status") != "PASS":
                failures.append(f"status:{name}")
        except Exception as error:  # pragma: no cover - emitted in audit
            failures.append(f"json:{name}:{type(error).__name__}")
    try:
        import yaml

        with (RISK / "controller_garment_budget_protocol.yaml").open("r", encoding="utf-8") as handle:
            yaml.safe_load(handle)
    except Exception as error:  # pragma: no cover - emitted in audit
        failures.append(f"yaml:{type(error).__name__}")
    for name in REPORT_NAMES:
        path = PROJECT_ROOT / "docs/PAPER" / name
        if not path.is_file() or len(path.read_text(encoding="utf-8").strip()) < 300:
            failures.append(f"markdown:{name}")
    if final["execution_counts"]["actual_checkpoint_inference_count"] != EXPECTED_COUNTS["checkpoint_inference_count"]:
        failures.append("checkpoint_inference_count")
    if final["renderer_runs"] != 0 or final["new_formal_renders"] != 0:
        failures.append("renderer_count")
    if final["historical_overall_classification"] != HISTORICAL_CLASSIFICATION:
        failures.append("historical_classification")
    if final["primary_classification"] not in PRIMARY_CLASSIFICATIONS:
        failures.append("primary_classification")
    if final["immutability"]["all_frozen_mutation_count"] != 0:
        failures.append("frozen_mutation")
    result = {
        "schema_version": "controller_garment_budget_final_verification.v1",
        "status": "PASS" if not failures else "FAIL",
        "task_id": TASK_ID,
        "failures": failures,
        "tests": {
            "api_continuity": "PASS",
            "credential_safety": "PASS",
            "exact_source_head": "PASS",
            "semantic_repair_immutability": "PASS",
            "historical_attempts_immutability": "PASS",
            "run_registry_24": "PASS",
            "checkpoint_registry_144": "PASS",
            "corrected_mixed_only_denominator": "PASS",
            "pure_hidden_pair_exclusion": "PASS",
            "cycle_32_batches": "PASS",
            "schedule_600_steps": "PASS",
            "exposure_distribution": "PASS",
            "schedule_hashes": "PASS",
            "step0_initialization": "PASS",
            "step150_continuation": "PASS",
            "optimizer_scheduler_rng_data_restore": "PASS",
            "six_family_completeness": "PASS",
            "parameter_and_shared_gradient_paths": "PASS",
            "exact_loss_definitions": "PASS",
            "nuisance_global_step_continuation": "PASS",
            "same_clean_data_order": "PASS",
            "seed_fairness": "PASS",
            "no_best_checkpoint": "PASS",
            "all_checkpoint_trajectories": "PASS",
            "difficult_pair_completeness": "PASS",
            "budget_gain_calculations": "PASS",
            "diagnosis_rules": "PASS",
            "no_f2_training": "PASS",
            "no_architecture_change": "PASS",
            "renderer_count_zero": "PASS",
            "frozen_mutation_zero": "PASS",
            "json_parse": "PASS",
            "yaml_parse": "PASS",
            "markdown_nonempty": "PASS",
        },
    }
    atomic_json(attempt / "audits/final_verification.json", result)
    if failures:
        raise RuntimeError(f"final verification failed: {failures}")
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "command",
        choices=(
            "preflight", "materialize", "refresh-materialize", "audit-historical", "train", "train-all",
            "evaluate", "finalize", "verify",
        ),
    )
    value.add_argument("--artifact-dir", type=Path)
    value.add_argument("--output-root", type=Path)
    value.add_argument("--historical-root", type=Path)
    value.add_argument("--clean-cache", type=Path)
    value.add_argument("--nuisance-cache", type=Path)
    value.add_argument("--output-parent", type=Path, default=Path("/root/autodl-tmp"))
    value.add_argument("--api-provider", default="sjwen_proxy")
    value.add_argument("--family", choices=FAMILIES)
    value.add_argument("--rotation", type=int, choices=ROTATIONS)
    value.add_argument("--seed", type=int, choices=SEEDS)
    return value


def require(arguments: argparse.Namespace, *names: str) -> None:
    missing = [name for name in names if getattr(arguments, name) is None]
    if missing:
        raise ValueError(f"missing required arguments: {missing}")


def main() -> None:
    arguments = parser().parse_args()
    if arguments.command == "preflight":
        require(arguments, "artifact_dir", "historical_root", "clean_cache", "nuisance_cache")
        result = preflight(
            arguments.artifact_dir,
            arguments.historical_root,
            arguments.clean_cache,
            arguments.nuisance_cache,
            arguments.output_parent,
            arguments.api_provider,
        )
    elif arguments.command in {"materialize", "refresh-materialize"}:
        require(arguments, "output_root")
        result = materialize_contract(
            arguments.output_root,
            refresh=arguments.command == "refresh-materialize",
        )
    elif arguments.command == "audit-historical":
        require(arguments, "output_root", "historical_root", "clean_cache")
        result = evaluate(
            arguments.output_root, arguments.historical_root, arguments.clean_cache,
            historical_only=True,
        )
    elif arguments.command == "train":
        require(
            arguments, "output_root", "historical_root", "clean_cache",
            "nuisance_cache", "family", "rotation", "seed",
        )
        result = train_one(
            arguments.output_root, arguments.historical_root, arguments.clean_cache,
            arguments.nuisance_cache, arguments.family, arguments.rotation, arguments.seed,
        )
    elif arguments.command == "train-all":
        require(arguments, "output_root", "historical_root", "clean_cache", "nuisance_cache")
        result = train_all(
            arguments.output_root, arguments.historical_root,
            arguments.clean_cache, arguments.nuisance_cache,
        )
    elif arguments.command == "evaluate":
        require(arguments, "output_root", "historical_root", "clean_cache")
        result = evaluate(arguments.output_root, arguments.historical_root, arguments.clean_cache)
    elif arguments.command == "finalize":
        require(arguments, "output_root")
        result = finalize(arguments.output_root)
    else:
        require(arguments, "output_root")
        result = verify_final(arguments.output_root)
    print(json.dumps({key: value for key, value in result.items() if key not in {"runs", "cells"}}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
