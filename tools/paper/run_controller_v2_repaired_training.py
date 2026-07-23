"""Train the repaired Controller V2 cross-fit micro-pilot and matched V1.

The executable consumes only the prospectively repaired contract frozen at
1a2059301e3a0ec7fa0f73591b69b7b428e986c4.  Each ``train`` invocation is one
fresh family/rotation/seed process.  It never loads a historical controller
checkpoint and evaluates only the final step-150 checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scene.compatibility_gated_reference_controller_v2 import (  # noqa: E402
    PAIR_TO_INDEX,
    CompatibilityGatedReferenceControllerV2,
)
from scene.reference_conditioned_dual_support_controller import (  # noqa: E402
    OUTFIT_ORDER,
    ReferenceConditionedDualSupportController,
    soft_target_cross_entropy,
)
from tools.paper import run_p0_color_spatial_soft_control_evaluations as sealed  # noqa: E402
from tools.paper import run_reference_conditioned_dual_support_controller_formal as formal  # noqa: E402


TASK_ID = "AAAI27-CONTROLLER-V2-CROSSFIT-MICRO-PILOT-REPAIRED-001"
SOURCE_HEAD = "1a2059301e3a0ec7fa0f73591b69b7b428e986c4"
RUN_BRANCH = "research/controller-v2-crossfit-micro-pilot-from-repaired-contract-20260723"
OUTPUT_NAME = "CONTROLLER-V2-CROSSFIT-MICRO-PILOT-001"
ATTEMPT = "attempt_002"
SEEDS = (0, 1, 2)
FAMILIES = ("V2", "MATCHED_V1")
CHECKPOINT_STEPS = (0, 30, 60, 90, 120, 150)
FEATURE_CACHE_RELATIVE = (
    "AAAI27-SEEN-OUTFIT-PAPER/shared_preflight/"
    "frozen_reference_feature_rows_v1.pt"
)

RISK = PROJECT_ROOT / "paper_protocol/reviewer_risk"
MANIFEST = RISK / "dual_support_controller_training_manifest.json"
ROTATIONS = RISK / "controller_v2_micro_pilot_rotation_manifests.json"
SCHEDULES = RISK / "controller_v2_micro_pilot_batch_schedules.json"
REPAIRED = RISK / "controller_v2_micro_pilot_contract_repaired.json"
OPTIMIZER = RISK / "controller_v2_micro_pilot_optimizer_contract.json"
LOSS = RISK / "controller_v2_micro_pilot_loss_contract.json"

EXPECTED_HASHES = {
    "repaired_protocol_sha256_lf":
        "44ef0c53f7a5ec4fe19733371ffefb1be14d5b6902e7f4060707edd4660d9f49",
    "source_manifest_sha256_lf":
        "a0dbfe98a25b7f82f198405cf41c1b8625b6faa63499396a78683d5d326ce8e3",
    "historical_cycle_scientific_sha256":
        "f60b0ee64ff5ce2b6693f53dd7eb665cc6aee5aa310f314175d2932307d57c77",
    "historical_schedule_scientific_sha256":
        "63902a2f36ccf864bfe6028654be0b162417c3adc6b99e1455b30fdab8a2bacf",
}
EXPECTED_ROTATION_HASHES = {
    0: (
        "e32dc5c98494f51673ef0d42cc03e71ead26d993da93521291b315ebc33d41fd",
        "d6221272b531e01e848563b8c7bb5e924209e0b341f45cfc3a9e816c17f8fa36",
    ),
    1: (
        "93a8f5873a7ac3f8d01492c01cd4a2b39068db3c988fce6aca14d58bf4b094bb",
        "f2305c089e843a78045cd43200bbfd7c9ba7e967cecb89395422253f7776530a",
    ),
    2: (
        "9e03c9fffc1eefcd9abaeae6e2140fec7b8018a022fc6d606fd1ebbfa0cdd849",
        "e6c87095797384f46142cb93c1e5a407d808287fdc83d7b1df99aa3e9309dfa9",
    ),
    3: (
        "01fb6be48611fb5fb5e63b0205053337e8ab48d5c50325ba1c0ff44c2bb4ac2a",
        "0b45485d601ae49991195e66947457370f9d2b08474bd85e3d8bd5182b828ebd",
    ),
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path: Path, *, lf: bool = False) -> str:
    value = path.read_bytes()
    if lf:
        value = value.replace(b"\r\n", b"\n")
    return hashlib.sha256(value).hexdigest()


def tensor_mapping_hash(value: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(value):
        tensor = value[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            value, handle, indent=2, sort_keys=True, ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")
    os.replace(temporary, path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def configure_determinism() -> None:
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)


def validate_contract() -> dict[str, Any]:
    required = (MANIFEST, ROTATIONS, SCHEDULES, REPAIRED, OPTIMIZER, LOSS)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"REPAIRED-CONTRACT-MISSING: {missing}")
    repaired = read_json(REPAIRED)
    rotations = read_json(ROTATIONS)
    schedules = read_json(SCHEDULES)
    manifest = read_json(MANIFEST)
    if (
        repaired["repair_branch"]
        != "research/controller-v2-micro-pilot-contract-repair-20260723"
        or repaired["classification"]
        != "CONTROLLER_V2_MICRO_PILOT_TRAINING_CONTRACT_REPAIRED"
    ):
        raise RuntimeError("REPAIRED-CONTRACT-SOURCE-MISMATCH")
    actual = {
        "repaired_protocol_sha256_lf":
            repaired["frozen_upstream_hashes"]["repaired_protocol_sha256_lf"],
        "source_manifest_sha256_lf": rotations["source_manifest_sha256_lf"],
        "historical_cycle_scientific_sha256":
            schedules["historical_cycle_sha256"],
        "historical_schedule_scientific_sha256":
            schedules["historical_schedule_sha256"],
    }
    if actual != EXPECTED_HASHES:
        raise RuntimeError(f"REPAIRED-CONTRACT-HASH-MISMATCH: {actual}")
    if len(manifest["query_sets"]) != 320:
        raise RuntimeError("REPAIRED-CONTRACT-RECORD-COUNT-MISMATCH")
    for row in schedules["rotations"]:
        rotation = int(row["rotation"])
        scientific = row["schedule"]
        hashes = (
            row["cycle_sha256"],
            row["schedule_sha256"],
        )
        if hashes != EXPECTED_ROTATION_HASHES[rotation]:
            raise RuntimeError(
                f"REPAIRED-CONTRACT-ROTATION-HASH-MISMATCH: {rotation}"
            )
        steps = scientific["steps"]
        if len(steps) != 150 or any(len(step["record_ids"]) != 5 for step in steps):
            raise RuntimeError(
                f"REPAIRED-CONTRACT-SCHEDULE-SHAPE-MISMATCH: {rotation}"
            )
        if [step["global_step"] for step in steps] != list(range(1, 151)):
            raise RuntimeError(
                f"REPAIRED-CONTRACT-STEP-ORDER-MISMATCH: {rotation}"
            )
    return {
        "manifest": manifest,
        "rotations": rotations,
        "schedules": schedules,
        "repaired": repaired,
        "optimizer": read_json(OPTIMIZER),
        "loss": read_json(LOSS),
    }


def environment_record() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_bytes": (
            torch.cuda.get_device_properties(0).total_memory
            if torch.cuda.is_available() else None
        ),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def attempt_root(output_root: Path) -> Path:
    return output_root / ATTEMPT


def training_root(output_root: Path, family: str, rotation: int, seed: int) -> Path:
    return (
        attempt_root(output_root) / "training" / family.lower()
        / f"rotation_{rotation}" / f"seed_{seed}"
    )


def augmented_cache_path(output_root: Path) -> Path:
    return attempt_root(output_root) / "features/v2_nuisance_feature_rows.pt"


def build_augmented_cache(
    output_root: Path, asset_root: Path, clean_cache: Mapping[str, Any]
) -> dict[str, Any]:
    output = augmented_cache_path(output_root)
    if output.exists():
        raise FileExistsError(output)
    os.environ["CANONDRESSGS_ASSET_ROOT"] = str(asset_root)
    sealed.RUN_BRANCH = RUN_BRANCH
    sealed.SOURCE_HEAD = SOURCE_HEAD
    runtime = sealed.EvaluationRuntime(
        attempt_root(output_root) / "_no_render_output", asset_root, {}
    )
    variants: dict[str, dict[str, dict[str, torch.Tensor]]] = {
        name: {} for name in ("blur", "mask_erosion", "mask_dilation")
    }
    conditions = ("cond_000000", "cond_000318", "cond_000017", "cond_000347")
    with torch.inference_mode():
        for outfit in OUTFIT_ORDER:
            for target in conditions:
                legal = [condition for condition in conditions if condition != target]
                observations = [
                    runtime.observation(outfit, condition) for condition in legal
                ]
                base_images = [item[0] for item in observations]
                base_masks = [item[1] for item in observations]
                key = f"{outfit}/{target}"
                for variant in variants:
                    images = [image.copy() for image in base_images]
                    masks = [mask.copy() for mask in base_masks]
                    if variant == "blur":
                        images = [
                            sealed.frozen.c5_gaussian_blur(image, mask)
                            for image, mask in zip(images, masks)
                        ]
                    elif variant == "mask_erosion":
                        masks = [
                            sealed.ndimage.binary_erosion(
                                mask, iterations=3
                            ).astype(bool)
                            for mask in masks
                        ]
                    elif variant == "mask_dilation":
                        masks = [
                            sealed.ndimage.binary_dilation(
                                mask, iterations=3
                            ).astype(bool)
                            for mask in masks
                        ]
                    rows, valid = runtime.f2_rows(images, masks)
                    variants[variant][key] = {
                        "f2": rows.detach().cpu(),
                        "valid": valid.detach().cpu(),
                    }
    payload = {
        "schema_version":
            "canondressgs.research.controller_v2_nuisance_feature_rows.v1",
        "task_id": TASK_ID,
        "source_head": SOURCE_HEAD,
        "feature_cache_input_sha256": file_hash(
            asset_root / FEATURE_CACHE_RELATIVE
        ),
        "definitions": {
            "blur": {
                "kernel_size": 11, "radius": 5, "sigma": 3.0,
                "outside_mask_restored": True,
            },
            "mask_erosion": {"iterations": 3, "structure": "SCIPY_DEFAULT"},
            "mask_dilation": {"iterations": 3, "structure": "SCIPY_DEFAULT"},
            "assignment_permutation": {"indices": [2, 0, 1]},
        },
        "episodes_per_variant": 20,
        "variants": variants,
    }
    atomic_torch(output, payload)
    del runtime
    torch.cuda.empty_cache()
    return {
        "path": str(output),
        "sha256": file_hash(output),
        "variants": list(variants),
        "episodes_per_variant": 20,
    }


def prepare(
    output_root: Path, asset_root: Path, execution_snapshot_head: str
) -> dict[str, Any]:
    contract = validate_contract()
    if not torch.cuda.is_available():
        raise RuntimeError("RESOURCE-GATE-CUDA-UNAVAILABLE")
    if attempt_root(output_root).exists():
        raise FileExistsError(
            f"append-only repaired attempt already exists: {attempt_root(output_root)}"
        )
    clean_cache_path = asset_root / FEATURE_CACHE_RELATIVE
    clean_cache = torch.load(clean_cache_path, map_location="cpu")
    if clean_cache.get("schema_version") != (
        "canondressgs.paper_frozen_reference_rows.v1"
    ):
        raise RuntimeError("FROZEN-F2-CACHE-SCHEMA-MISMATCH")
    for family in FAMILIES:
        for rotation in range(4):
            for seed in SEEDS:
                root = training_root(output_root, family, rotation, seed)
                (root / "checkpoints").mkdir(parents=True, exist_ok=False)
                (root / "logs").mkdir(parents=True, exist_ok=False)
    for name in (
        "audits", "features", "calibration", "predictions", "routing",
        "perturbations", "renders", "visuals", "metrics", "aggregates",
        "reports", "fingerprints",
    ):
        (attempt_root(output_root) / name).mkdir(parents=True, exist_ok=True)
    nuisance = build_augmented_cache(output_root, asset_root, clean_cache)
    smoke_v2 = CompatibilityGatedReferenceControllerV2(seed=0)
    smoke_v1 = ReferenceConditionedDualSupportController(seed=0)
    if smoke_v2.parameter_count != 9232 or smoke_v1.parameter_count != 3589:
        raise RuntimeError("CONTROLLER-PARAMETER-COUNT-MISMATCH")
    result = {
        "schema_version":
            "canondressgs.research.controller_v2_repaired_preflight.v1",
        "status": "PASS",
        "task_id": TASK_ID,
        "source_head": SOURCE_HEAD,
        "execution_snapshot_head": execution_snapshot_head,
        "run_branch": RUN_BRANCH,
        "environment": environment_record(),
        "clean_feature_cache": {
            "path": str(clean_cache_path),
            "sha256": file_hash(clean_cache_path),
            "episode_count": len(clean_cache["episodes"]),
        },
        "nuisance_feature_cache": nuisance,
        "contract_global_sha256":
            contract["repaired"]["global_contract_sha256"],
        "expected_runs": 24,
        "expected_optimizer_steps": 3600,
        "expected_checkpoints": 144,
        "counts": {
            "training": 0, "backward": 0, "optimizer_created": 0,
            "optimizer_step": 0, "scheduler_step": 0,
            "checkpoint_write": 0, "paper_final": 0,
        },
    }
    atomic_json(attempt_root(output_root) / "audits/preflight.json", result)
    return result


def case_rows_variant(
    clean_cache: Mapping[str, Any],
    nuisance_cache: Mapping[str, Any],
    record: Mapping[str, Any],
    variant: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if variant == "clean":
        return formal.case_rows(clean_cache, record)
    rows, validity = [], []
    target = record["target_view_fold"]
    if variant == "assignment_permutation":
        clean_rows, clean_valid = formal.case_rows(clean_cache, record)
        indices = torch.tensor([2, 0, 1])
        return clean_rows[indices], clean_valid[indices]
    archive = nuisance_cache["variants"][variant]
    for position, outfit in enumerate(record["garment_labels"]):
        episode = archive[f"{outfit}/{target}"]
        rows.append(episode["f2"][position])
        validity.append(episode["valid"][position])
    return torch.stack(rows), torch.stack(validity)


def js_divergence(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first = first.clamp_min(1.0e-8)
    second = second.clamp_min(1.0e-8)
    middle = 0.5 * (first + second)
    return 0.5 * (
        (first * (first.log() - middle.log())).sum(dim=-1)
        + (second * (second.log() - middle.log())).sum(dim=-1)
    ).mean()


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
) -> dict[str, Any]:
    return {
        "schema_version":
            "canondressgs.research.controller_v2_repaired_checkpoint.v1",
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
        "source_head": SOURCE_HEAD,
        "loads_historical_controller_checkpoint": False,
        "best_checkpoint_selection": False,
        "paper_final": False,
    }


def optimizer_for(model: torch.nn.Module) -> torch.optim.Adam:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.Adam(
        parameters,
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


def write_checkpoint(
    root: Path,
    family: str,
    rotation: int,
    seed: int,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    initialization_sha256: str,
    schedule_sha256: str,
) -> Path:
    name = (
        "final_step_150.pt" if step == 150
        else f"step_{step:03d}.pt"
    )
    path = root / "checkpoints" / name
    atomic_torch(
        path,
        checkpoint_payload(
            family, rotation, seed, step, model, optimizer, scheduler,
            initialization_sha256, schedule_sha256,
        ),
    )
    return path


def training_targets(
    records: Sequence[Mapping[str, Any]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    garments = torch.tensor(
        [record["target_distribution"] for record in records],
        dtype=torch.float32, device=device,
    )
    mixedness = torch.tensor(
        [
            0.0 if record["assignment_type"] in {"AAA", "BBB"} else 1.0
            for record in records
        ],
        dtype=torch.float32, device=device,
    )
    return garments, mixedness


def v2_loss(
    outputs: Sequence[Any],
    augmented: Sequence[Any],
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    garment_target, mixedness_target = training_targets(records, device)
    garment_logits = torch.stack([output.garment_logits for output in outputs])
    mixedness_logits = torch.stack([output.mixedness_logit for output in outputs])
    garment = soft_target_cross_entropy(garment_logits, garment_target)
    mixedness = F.binary_cross_entropy_with_logits(
        mixedness_logits, mixedness_target, reduction="mean"
    )
    predicted_weights, target_weights = [], []
    for output, record in zip(outputs, records):
        if record["assignment_type"] in {"AAA", "BBB"}:
            continue
        pair_index = PAIR_TO_INDEX[record["pair_id"]]
        earlier = record["pair_id"].split("_")[0]
        earlier_index = OUTFIT_ORDER.index(earlier)
        predicted_weights.append(output.all_pair_weights[pair_index])
        target_weights.append(
            output.all_pair_weights.new_tensor(
                record["target_distribution"][earlier_index]
            )
        )
    weight = F.smooth_l1_loss(
        torch.stack(predicted_weights),
        torch.stack(target_weights),
        beta=0.1,
        reduction="mean",
    )
    clean_garment = torch.stack(
        [output.garment_probabilities for output in outputs]
    )
    aug_garment = torch.stack(
        [output.garment_probabilities for output in augmented]
    )
    garment_cons = js_divergence(clean_garment, aug_garment)
    mixed_cons = F.smooth_l1_loss(
        torch.stack([output.mixedness_probability for output in outputs]),
        torch.stack([output.mixedness_probability for output in augmented]),
        beta=0.1,
        reduction="mean",
    )
    weight_cons = F.smooth_l1_loss(
        torch.stack([output.all_pair_weights for output in outputs]),
        torch.stack([output.all_pair_weights for output in augmented]),
        beta=0.1,
        reduction="mean",
    )
    consistency = garment_cons + mixed_cons + weight_cons
    total = garment + mixedness + weight + 0.1 * consistency
    return total, {
        "total": float(total.detach()),
        "garment": float(garment.detach()),
        "mixedness": float(mixedness.detach()),
        "pair_weight": float(weight.detach()),
        "consistency": float(consistency.detach()),
        "garment_consistency": float(garment_cons.detach()),
        "mixedness_consistency": float(mixed_cons.detach()),
        "all_pair_weight_consistency": float(weight_cons.detach()),
    }


def train(
    output_root: Path,
    asset_root: Path,
    family: str,
    rotation: int,
    seed: int,
) -> dict[str, Any]:
    if family not in FAMILIES or rotation not in range(4) or seed not in SEEDS:
        raise ValueError("family/rotation/seed outside repaired contract")
    configure_determinism()
    contract = validate_contract()
    preflight = read_json(attempt_root(output_root) / "audits/preflight.json")
    if preflight.get("status") != "PASS":
        raise RuntimeError("REPAIRED-PREFLIGHT-NOT-PASSED")
    clean_cache = torch.load(
        asset_root / FEATURE_CACHE_RELATIVE, map_location="cpu"
    )
    nuisance_cache = torch.load(
        augmented_cache_path(output_root), map_location="cpu"
    )
    schedule_row = next(
        row for row in contract["schedules"]["rotations"]
        if int(row["rotation"]) == rotation
    )
    steps = schedule_row["schedule"]["steps"]
    record_index = {
        row["record_id"]: row for row in contract["manifest"]["query_sets"]
    }
    scheduled_records = [
        [record_index[record_id] for record_id in step["record_ids"]]
        for step in steps
    ]
    root = training_root(output_root, family, rotation, seed)
    result_path = root / "training_result.json"
    if result_path.exists() or any((root / "checkpoints").iterdir()):
        raise FileExistsError(
            f"fresh-process run output is not empty: {family}/{rotation}/{seed}"
        )
    device = torch.device("cuda")
    model: torch.nn.Module
    if family == "V2":
        model = CompatibilityGatedReferenceControllerV2(seed=seed)
        expected_parameters = 9232
    else:
        model = ReferenceConditionedDualSupportController(seed=seed)
        expected_parameters = 3589
    model = model.to(device).train()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != expected_parameters:
        raise RuntimeError("TRAINABLE-PARAMETER-COUNT-MISMATCH")
    initialization_sha256 = tensor_mapping_hash(
        {name: value.detach().cpu() for name, value in model.state_dict().items()}
    )
    optimizer = optimizer_for(model)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda _: 1.0
    )
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups for parameter in group["params"]
    }
    trainable_ids = {
        id(parameter) for parameter in model.parameters()
        if parameter.requires_grad
    }
    if optimizer_ids != trainable_ids:
        raise RuntimeError("OPTIMIZER-MEMBERSHIP-MISMATCH")
    checkpoint_paths = [
        write_checkpoint(
            root, family, rotation, seed, 0, model, optimizer, scheduler,
            initialization_sha256, schedule_row["schedule_sha256"],
        )
    ]
    trace: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for step_index, (step, records) in enumerate(
        zip(steps, scheduled_records), start=1
    ):
        optimizer.zero_grad(set_to_none=True)
        clean_outputs = []
        augmented_outputs = []
        for record in records:
            rows, valid = case_rows_variant(
                clean_cache, nuisance_cache, record, "clean"
            )
            clean_outputs.append(model(rows.to(device), valid.to(device)))
            if family == "V2":
                rows, valid = case_rows_variant(
                    clean_cache, nuisance_cache, record,
                    step["v2_nuisance_type"],
                )
                augmented_outputs.append(
                    model(rows.to(device), valid.to(device))
                )
        garment_target, _ = training_targets(records, device)
        if family == "V2":
            loss, components = v2_loss(
                clean_outputs, augmented_outputs, records, device
            )
        else:
            logits = torch.stack([output.logits for output in clean_outputs])
            loss = soft_target_cross_entropy(logits, garment_target)
            components = {
                "total": float(loss.detach()),
                "garment": float(loss.detach()),
            }
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite loss: {family}/{rotation}/{seed}/step{step_index}"
            )
        loss.backward()
        squared = sum(
            float(parameter.grad.detach().double().square().sum())
            for parameter in model.parameters() if parameter.grad is not None
        )
        gradient_norm = squared ** 0.5
        clipped = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        trace.append({
            "global_step": step_index,
            "cycle_index": step["cycle_index"],
            "batch_index": step["batch_index"],
            "target_view_fold": step["target_view_fold"],
            "record_ids": step["record_ids"],
            "nuisance_type": (
                step["v2_nuisance_type"] if family == "V2" else None
            ),
            "losses": components,
            "gradient_norm": gradient_norm,
            "clip_returned_norm": float(clipped),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "nan_or_inf": False,
        })
        if step_index in CHECKPOINT_STEPS[1:]:
            checkpoint_paths.append(
                write_checkpoint(
                    root, family, rotation, seed, step_index, model,
                    optimizer, scheduler, initialization_sha256,
                    schedule_row["schedule_sha256"],
                )
            )
    torch.cuda.synchronize()
    wall_clock = time.perf_counter() - started
    if len(checkpoint_paths) != 6:
        raise RuntimeError("CHECKPOINT-COUNT-MISMATCH")
    expected_clean = 150
    expected_aug = 150 if family == "V2" else 0
    result = {
        "schema_version":
            "canondressgs.research.controller_v2_repaired_training_run.v1",
        "status": "PASS",
        "task_id": TASK_ID,
        "family": family,
        "rotation": rotation,
        "seed": seed,
        "source_head": SOURCE_HEAD,
        "initialization_policy": "RANDOM_SEEDED_INITIALIZATION",
        "initialization_sha256": initialization_sha256,
        "fresh_process": True,
        "historical_controller_checkpoint_loads": 0,
        "parameter_count": parameter_count,
        "trainable_parameter_names": [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ],
        "frozen_f2_requires_grad": False,
        "frozen_f2_optimizer_membership": False,
        "schedule_sha256": schedule_row["schedule_sha256"],
        "cycle_sha256": schedule_row["cycle_sha256"],
        "counts": {
            "training_steps": 150,
            "clean_forward_batches": expected_clean,
            "augmented_forward_batches": expected_aug,
            "record_forwards_clean": 750,
            "record_forwards_augmented": 750 if family == "V2" else 0,
            "backward_calls": 150,
            "optimizer_created": 1,
            "optimizer_steps": 150,
            "scheduler_steps": 150,
            "checkpoint_writes": 6,
        },
        "optimizer": {
            "class": "torch.optim.Adam",
            "learning_rate": 0.02,
            "betas": [0.9, 0.999],
            "epsilon": 1.0e-8,
            "weight_decay": 0.0,
            "amsgrad": False,
            "foreach": None,
            "fused": None,
            "capturable": False,
            "differentiable": False,
            "maximize": False,
            "parameter_group_count": 1,
            "membership_exact": True,
        },
        "scheduler": {
            "class": "torch.optim.lr_scheduler.LambdaLR",
            "constant": 1.0,
            "warmup_steps": 0,
        },
        "gradient_clip_max_norm": 5.0,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "checkpoint_paths": [str(path) for path in checkpoint_paths],
        "final_checkpoint": str(checkpoint_paths[-1]),
        "final_checkpoint_sha256": file_hash(checkpoint_paths[-1]),
        "final_only_evaluated": True,
        "early_stopping": False,
        "best_checkpoint_selection": False,
        "extension_steps": 0,
        "trace": trace,
        "trace_sha256": canonical_hash(trace),
        "numerics": {
            "loss_initial": trace[0]["losses"],
            "loss_final": trace[-1]["losses"],
            "loss_min": min(row["losses"]["total"] for row in trace),
            "loss_max": max(row["losses"]["total"] for row in trace),
            "gradient_norm_max": max(row["gradient_norm"] for row in trace),
            "nan_or_inf_count": 0,
        },
        "wall_clock_seconds": wall_clock,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "paper_final": False,
        "paper_final_count": 0,
    }
    atomic_json(result_path, result)
    return result


def aggregate(output_root: Path) -> dict[str, Any]:
    runs = []
    for family in FAMILIES:
        for rotation in range(4):
            for seed in SEEDS:
                path = (
                    training_root(output_root, family, rotation, seed)
                    / "training_result.json"
                )
                if not path.is_file():
                    raise RuntimeError(f"TRAINING-RUN-MISSING: {path}")
                runs.append(read_json(path))
    counts = {
        key: sum(run["counts"][key] for run in runs)
        for key in (
            "training_steps", "clean_forward_batches",
            "augmented_forward_batches", "record_forwards_clean",
            "record_forwards_augmented", "backward_calls",
            "optimizer_created", "optimizer_steps", "scheduler_steps",
            "checkpoint_writes",
        )
    }
    expected = {
        "training_steps": 3600,
        "clean_forward_batches": 3600,
        "augmented_forward_batches": 1800,
        "record_forwards_clean": 18000,
        "record_forwards_augmented": 9000,
        "backward_calls": 3600,
        "optimizer_created": 24,
        "optimizer_steps": 3600,
        "scheduler_steps": 3600,
        "checkpoint_writes": 144,
    }
    if counts != expected:
        raise RuntimeError(f"AGGREGATE-TRAINING-COUNT-MISMATCH: {counts}")
    value = {
        "schema_version":
            "canondressgs.research.controller_v2_repaired_training_aggregate.v1",
        "status": "PASS",
        "task_id": TASK_ID,
        "source_head": SOURCE_HEAD,
        "run_count": len(runs),
        "v2_run_count": sum(run["family"] == "V2" for run in runs),
        "matched_v1_run_count":
            sum(run["family"] == "MATCHED_V1" for run in runs),
        "counts": counts,
        "expected_counts": expected,
        "initialization_unique_by_family_seed": {
            family: {
                str(seed): sorted({
                    run["initialization_sha256"] for run in runs
                    if run["family"] == family and run["seed"] == seed
                })
                for seed in SEEDS
            }
            for family in FAMILIES
        },
        "runs": runs,
        "paper_final": False,
        "paper_final_count": 0,
    }
    atomic_json(
        attempt_root(output_root) / "aggregates/training_results.json", value
    )
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("validate", "prepare", "train", "aggregate"),
        required=True,
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--execution-snapshot-head")
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--rotation", type=int, choices=range(4))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    arguments = parser.parse_args()
    if arguments.phase == "validate":
        result = {
            "status": "PASS",
            "global_contract_sha256":
                validate_contract()["repaired"]["global_contract_sha256"],
        }
    else:
        if arguments.output_root is None:
            parser.error(f"{arguments.phase} requires --output-root")
        output_root = arguments.output_root.resolve()
        if arguments.phase in {"prepare", "train"} and arguments.asset_root is None:
            parser.error(f"{arguments.phase} requires --asset-root")
        if arguments.phase == "prepare":
            if not arguments.execution_snapshot_head:
                parser.error("prepare requires --execution-snapshot-head")
            result = prepare(
                output_root, arguments.asset_root.resolve(),
                arguments.execution_snapshot_head,
            )
        elif arguments.phase == "train":
            if (
                arguments.family is None
                or arguments.rotation is None
                or arguments.seed is None
            ):
                parser.error("train requires --family, --rotation, and --seed")
            result = train(
                output_root, arguments.asset_root.resolve(),
                arguments.family, arguments.rotation, arguments.seed,
            )
        else:
            result = aggregate(output_root)
    print(json.dumps(result, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
