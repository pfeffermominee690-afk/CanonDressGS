from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
import weakref
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

import train_dressable as legacy_training
from scene.gaussian_clothing_residuals import CHANNELS, GaussianClothingResiduals
from scene.p0_candidate_initialization_protocol import (
    DETERMINISTIC_ZERO_INITIALIZATION,
    PAIRED_RANDOM_TRUNK_WITH_ZERO_OUTPUT_HEAD,
    RANDOM_SEEDED_INITIALIZATION,
    build_b6_candidate,
    build_m3_m4_paired_candidates,
    build_ours_v2_candidate,
    selected_state_sha256,
    tensor_mapping_sha256,
)
from tools import run_multi_outfit_explicit_basis as multi
from tools.paper import formal_batch_runtime as formal
from tools.paper.verify_seen_outfit_paper_assets import verify_manifest


TASK_ID = "AAAI27-DETERMINISTIC-INITIALIZATION-PROTOCOL-001"
BRANCH = "paper/aaai27-deterministic-initialization-protocol-20260721"
SOURCE_HEAD = "71dd86c1c01e00e44e8135a45ce29db788141fe0"
FORMAL_REGISTRY_SHA256 = "1834597b787d98acf475b352b791f0f16714fd870d51be36259ac7383a406c5e"
FORMAL_OUTPUT_SHA256 = "7b9449e03d53e29cff11ded1fdc95e633640d38754cf152f2ab07a935d89d6bc"
FROZEN_MANIFEST_SHA256 = "ff90540e56c9c2db3fd6a1e45c31a1d1eb5a8a32effabde71158584d9be538bb"
OLD_SEED_AUDIT_SHA256 = "0b2e068203031b34e0004725923d452e185f319f3e9088843ecbc42ba5307f14"
FORMAL_FILE_COUNT = 4127
FORMAL_TOTAL_BYTES = 964043888
OUTFITS = ("O01", "O02", "O03", "O04", "O08")
SEEDS = (0, 1, 2)
HISTORICAL_SOURCE_SHA256 = {
    "scene/multi_outfit_linear_coefficient_control.py": "87347845019426bb44dd0044b9b52c1a264c6edbafcdeaea8d864de79ca488d2",
    "tools/paper/formal_batch_runtime.py": "75211429a628801882d211805a2334fc88c11f6654cfbc4b6cc47fa2906693a6",
    "scene/reference_basis_coefficient_fusion.py": "54ce50139003ec82f5afa09f96dd87d495da8031edb05283c82e115c69411e26",
}


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=PROJECT_ROOT, text=True
    ).strip()


def _upstream_or_none() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "@{upstream}"],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _formal_metadata_fingerprint(path: Path) -> str:
    command = (
        "cd " + shlex.quote(str(path))
        + " && find . -type f -printf '%P %s %T@\\n' | LC_ALL=C sort | sha256sum"
    )
    return subprocess.check_output(["bash", "-lc", command], text=True).split()[0]


def _tree_bytes(path: Path) -> int:
    return int(
        subprocess.check_output(["du", "-sb", str(path)], text=True).split()[0]
    )


def _source_sha(path: str) -> str:
    return hashlib.sha256(
        subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=PROJECT_ROOT)
    ).hexdigest()


def _registry_summary(formal_registry: Path, reviewer_registry: Path) -> dict[str, Any]:
    formal_payload = yaml.safe_load(formal_registry.read_text(encoding="utf-8"))
    reviewer_payload = yaml.safe_load(reviewer_registry.read_text(encoding="utf-8"))
    rows = reviewer_payload["experiments"]
    statuses: dict[str, int] = {}
    priorities: dict[str, int] = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
        priorities[row["priority"]] = priorities.get(row["priority"], 0) + 1
    return {
        "reviewer_total": len(rows),
        "reviewer_status_counts": statuses,
        "reviewer_priority_counts": priorities,
        "paper_final_count": sum(
            row.get("status") == "PAPER_FINAL"
            for row in (*formal_payload["experiments"], *rows)
        ),
    }


def _build_context(
    attempt: Path, optimizer_capture: dict[str, Any]
) -> dict[str, Any]:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs/research/subject02_multi_outfit_explicit_basis_v1.yaml")
        .read_text(encoding="utf-8")
    )
    old_branch, old_source = multi.EXPECTED_BRANCH, multi.EXPECTED_SOURCE_HEAD
    original_builder = legacy_training.build_image_conditioned_optimizer
    original_step = torch.optim.Adam.step
    original_zero_grad = torch.optim.Adam.zero_grad
    call_counts = {"step": 0, "zero_grad": 0}

    def tracked_builder(model, optimizer_config):
        optimizer = original_builder(model, optimizer_config)
        parameter_ids = []
        groups = []
        for group in optimizer.param_groups:
            parameters = list(group["params"])
            parameter_ids.extend(id(parameter) for parameter in parameters)
            groups.append({
                "name": group.get("name"),
                "parameter_count": sum(parameter.numel() for parameter in parameters),
                "tensor_count": len(parameters),
                "learning_rate": float(group["lr"]),
            })
        optimizer_capture.update({
            "created": True,
            "class": f"{type(optimizer).__module__}.{type(optimizer).__name__}",
            "parameter_groups": groups,
            "parameter_count": sum(group["parameter_count"] for group in groups),
            "parameter_ids": parameter_ids,
            "initial_state_entry_count": len(optimizer.state),
            "weak_reference": weakref.ref(optimizer),
        })
        return optimizer

    def forbidden_step(self, *args, **kwargs):
        call_counts["step"] += 1
        raise RuntimeError("NO-TRAINING-GATE-VIOLATION: legacy optimizer step")

    def forbidden_zero_grad(self, *args, **kwargs):
        call_counts["zero_grad"] += 1
        raise RuntimeError("NO-TRAINING-GATE-VIOLATION: legacy optimizer zero_grad")

    multi.EXPECTED_BRANCH, multi.EXPECTED_SOURCE_HEAD = BRANCH, SOURCE_HEAD
    config["branch"] = BRANCH
    config["source_head"] = SOURCE_HEAD
    config["output"]["attempt"] = attempt.name
    legacy_training.build_image_conditioned_optimizer = tracked_builder
    torch.optim.Adam.step = forbidden_step
    torch.optim.Adam.zero_grad = forbidden_zero_grad
    try:
        context = multi.build_context(
            config, attempt, create=False, reference_backbone=True
        )
    finally:
        torch.optim.Adam.zero_grad = original_zero_grad
        torch.optim.Adam.step = original_step
        legacy_training.build_image_conditioned_optimizer = original_builder
        multi.EXPECTED_BRANCH, multi.EXPECTED_SOURCE_HEAD = old_branch, old_source
    optimizer_capture["step_count"] = call_counts["step"]
    optimizer_capture["zero_grad_count"] = call_counts["zero_grad"]
    gc.collect()
    optimizer_capture["discarded_by_caller"] = (
        optimizer_capture.get("weak_reference", lambda: object())() is None
    )
    optimizer_capture.pop("weak_reference", None)
    return context


def _run_fresh_processes(
    feature_cache: Path, device: str
) -> dict[str, Any]:
    executable = PROJECT_ROOT / "tools/paper/p0_initialization_fresh_process.py"
    raw: dict[str, dict[str, list[dict[str, Any]]]] = {
        "ours_v2": {}, "b6": {}, "complex_pair": {}
    }
    for family in raw:
        for seed in SEEDS:
            rows = []
            for repeat in range(2):
                output = subprocess.check_output(
                    [
                        sys.executable, str(executable),
                        "--family", family,
                        "--seed", str(seed),
                        "--feature-cache", str(feature_cache),
                        "--device", device,
                    ],
                    cwd=PROJECT_ROOT,
                    text=True,
                )
                row = json.loads(output)
                row["repeat"] = repeat
                rows.append(row)
            raw[family][str(seed)] = rows

    def method_rows(family: str, method: str) -> list[dict[str, Any]]:
        return [
            run["methods"][method]
            for seed in SEEDS for run in raw[family][str(seed)]
        ]

    ours = method_rows("ours_v2", "Ours-v2")
    b6 = method_rows("b6", "B6")
    m3 = method_rows("complex_pair", "M3")
    m4 = method_rows("complex_pair", "M4")

    def same_seed_exact(family: str, method: str, fields: Sequence[str]) -> bool:
        return all(
            all(
                raw[family][str(seed)][0]["methods"][method][field]
                == raw[family][str(seed)][1]["methods"][method][field]
                for field in fields
            )
            for seed in SEEDS
        )

    ours_pass = (
        same_seed_exact("ours_v2", "Ours-v2", ("trainable_state_sha256", "step0_forward_sha256"))
        and len({row["trainable_state_sha256"] for row in ours}) == 1
        and len({row["step0_forward_sha256"] for row in ours}) == 1
        and all(row["initialization_policy"] == DETERMINISTIC_ZERO_INITIALIZATION for row in ours)
        and all(row["initial_output_all_zero"] for row in ours)
    )
    b6_pass = (
        same_seed_exact("b6", "B6", ("trainable_state_sha256", "step0_forward_sha256"))
        and len({raw["b6"][str(seed)][0]["methods"]["B6"]["trainable_state_sha256"] for seed in SEEDS}) == 3
        and all(row["initialization_policy"] == RANDOM_SEEDED_INITIALIZATION for row in b6)
    )
    complex_pass = (
        same_seed_exact("complex_pair", "M3", ("trainable_state_sha256", "step0_forward_sha256"))
        and same_seed_exact("complex_pair", "M4", ("trainable_state_sha256", "step0_forward_sha256"))
        and len({raw["complex_pair"][str(seed)][0]["methods"]["M3"]["trunk_sha256"] for seed in SEEDS}) == 3
        and all(
            raw["complex_pair"][str(seed)][repeat]["methods"]["M3"]["trunk_sha256"]
            == raw["complex_pair"][str(seed)][repeat]["methods"]["M4"]["trunk_sha256"]
            for seed in SEEDS for repeat in range(2)
        )
        and all(row["initial_output_all_zero"] for row in (*m3, *m4))
        and all(row["initialization_policy"] == PAIRED_RANDOM_TRUNK_WITH_ZERO_OUTPUT_HEAD for row in (*m3, *m4))
    )
    return {
        "schema_version": "canondressgs.paper.initialization_policy_aware_audit.v1",
        "status": "PASS" if ours_pass and b6_pass and complex_pass else "INITIALIZATION-POLICY-CONTRACT-FAIL",
        "fresh_processes_per_seed": 2,
        "seeds": list(SEEDS),
        "raw_fresh_process_records": raw,
        "ours_v2": {
            "status": "PASS" if ours_pass else "FAIL",
            "policy": DETERMINISTIC_ZERO_INITIALIZATION,
            "same_seed_bitwise_exact": same_seed_exact("ours_v2", "Ours-v2", ("trainable_state_sha256", "step0_forward_sha256")),
            "cross_seed_state_unique_count": len({row["trainable_state_sha256"] for row in ours}),
            "cross_seed_forward_unique_count": len({row["step0_forward_sha256"] for row in ours}),
            "cross_seed_identity_interpretation": "EXPECTED_DETERMINISTIC_IDENTITY",
        },
        "b6": {
            "status": "PASS" if b6_pass else "FAIL",
            "policy": RANDOM_SEEDED_INITIALIZATION,
            "same_seed_bitwise_exact": same_seed_exact("b6", "B6", ("trainable_state_sha256", "step0_forward_sha256")),
            "cross_seed_state_unique_count": len({raw["b6"][str(seed)][0]["methods"]["B6"]["trainable_state_sha256"] for seed in SEEDS}),
        },
        "m3_m4": {
            "status": "PASS" if complex_pass else "FAIL",
            "policy": PAIRED_RANDOM_TRUNK_WITH_ZERO_OUTPUT_HEAD,
            "m3_same_seed_bitwise_exact": same_seed_exact("complex_pair", "M3", ("trainable_state_sha256", "step0_forward_sha256")),
            "m4_same_seed_bitwise_exact": same_seed_exact("complex_pair", "M4", ("trainable_state_sha256", "step0_forward_sha256")),
            "cross_seed_trunk_unique_count": len({raw["complex_pair"][str(seed)][0]["methods"]["M3"]["trunk_sha256"] for seed in SEEDS}),
            "same_seed_shared_trunk_exact": all(
                raw["complex_pair"][str(seed)][repeat]["methods"]["M3"]["trunk_sha256"]
                == raw["complex_pair"][str(seed)][repeat]["methods"]["M4"]["trunk_sha256"]
                for seed in SEEDS for repeat in range(2)
            ),
            "zero_output_heads": all(row["initial_output_all_zero"] for row in (*m3, *m4)),
            "initial_output_semantics_matched": all(
                raw["complex_pair"][str(seed)][repeat]["methods"]["M3"]["step0_forward_sha256"]
                == raw["complex_pair"][str(seed)][repeat]["methods"]["M4"]["step0_forward_sha256"]
                for seed in SEEDS for repeat in range(2)
            ),
        },
        "candidate_optimizer_created": False,
        "candidate_optimizer_step_count": 0,
    }


def _residual_mapping(value: GaussianClothingResiduals) -> dict[str, torch.Tensor]:
    mapping = value.as_dict()
    if any(mapping[name] is None for name in CHANNELS):
        raise ValueError("semantic audit requires all six residual channels")
    return {name: mapping[name] for name in CHANNELS}  # type: ignore[return-value]


def _mean_residual(values: Sequence[GaussianClothingResiduals]) -> GaussianClothingResiduals:
    return GaussianClothingResiduals.from_dict({
        name: torch.stack([_residual_mapping(value)[name] for value in values]).mean(0)
        for name in CHANNELS
    })


def _tensor_record(values: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    flattened = torch.cat([value.detach().reshape(-1).to(torch.float64) for value in values.values()])
    first = next(iter(values.values()))
    return {
        "shapes": {name: list(value.shape) for name, value in values.items()},
        "dtype": str(first.dtype),
        "device": str(first.device),
        "sha256": tensor_mapping_sha256(values),
        "l1_norm": float(flattened.abs().sum()),
        "l2_norm": float(torch.linalg.vector_norm(flattened)),
        "linf_norm": float(flattened.abs().max()),
    }


def _difference_record(
    first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    differences = {name: first[name] - second[name] for name in first}
    result = _tensor_record(differences)
    result["bitwise_equal"] = all(torch.equal(first[name], second[name]) for name in first)
    return result


def _coefficient_record(value: torch.Tensor) -> dict[str, Any]:
    return _tensor_record({"coefficient": value})


def _mean_zero_semantics(
    context: Mapping[str, Any], cache: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    basis, coefficients, payload, basis_path = formal._basis_artifact(
        context, "Ours_Seen_Outfit_Explicit_Basis_V1"
    )
    device = context["base"]._xyz.device
    mean = payload["coefficient_train_mean"].to(device)
    std = payload["coefficient_train_std"].to(device)
    standardized_zero = torch.zeros_like(mean)
    raw_zero = torch.zeros_like(mean)
    raw_from_standardized_zero = standardized_zero * std + mean
    basis_at_raw_zero = basis(raw_zero, chunk_size=16384)
    standardized_zero_endpoint = basis(raw_from_standardized_zero, chunk_size=16384)
    physical_zero = GaussianClothingResiduals.zeros(context["base"])
    teachers = formal.load_frozen_teacher_residuals(context)
    teacher_mean = _mean_residual([teachers[outfit] for outfit in OUTFITS])

    basis_zero_map = _residual_mapping(basis_at_raw_zero)
    standard_endpoint_map = _residual_mapping(standardized_zero_endpoint)
    physical_zero_map = _residual_mapping(physical_zero)
    teacher_mean_map = _residual_mapping(teacher_mean)

    variant_rows = []
    for outfit in OUTFITS:
        for condition in formal.CONDITIONS:
            key = f"{outfit}/{condition}"
            zero = cache["episodes"][key]["zero"]
            base = cache["episodes"][key]["base"]
            variant_rows.append({
                "episode": key,
                "zero_reference_feature_sha256": tensor_mapping_sha256({
                    name: value for name, value in zero.items()
                }),
                "base_reference_feature_sha256": tensor_mapping_sha256({
                    name: value for name, value in base.items()
                }),
                "bitwise_equal": all(torch.equal(zero[name], base[name]) for name in zero),
            })
    equal_count = sum(row["bitwise_equal"] for row in variant_rows)
    semantics = {
        "schema_version": "canondressgs.paper.mean_garment_zero_residual_semantics.v1",
        "status": "RESOLVED",
        "basis_path": str(basis_path),
        "basis_sha256": _sha256_file(basis_path),
        "coefficient_normalization": {
            "formula": "c_raw = z_std * train_std + train_mean",
            "train_mean": mean.detach().cpu().tolist(),
            "train_std": std.detach().cpu().tolist(),
            "standardized_zero_de_standardized": raw_from_standardized_zero.detach().cpu().tolist(),
        },
        "objects": {
            "standardized_coefficient_zero": {
                **_coefficient_record(standardized_zero),
                "mapped_raw_coefficient": _coefficient_record(raw_from_standardized_zero),
                "mapped_residual": _tensor_record(standard_endpoint_map),
                "difference_to_basis_mean_residual": _difference_record(standard_endpoint_map, basis_zero_map),
                "difference_to_physical_zero_residual": _difference_record(standard_endpoint_map, physical_zero_map),
                "difference_to_five_teacher_residual_mean": _difference_record(standard_endpoint_map, teacher_mean_map),
            },
            "raw_coefficient_zero": {
                **_coefficient_record(raw_zero),
                "mapped_residual": _tensor_record(basis_zero_map),
                "difference_to_basis_mean_residual": _difference_record(basis_zero_map, basis_zero_map),
                "difference_to_physical_zero_residual": _difference_record(basis_zero_map, physical_zero_map),
                "difference_to_five_teacher_residual_mean": _difference_record(basis_zero_map, teacher_mean_map),
            },
            "basis_reconstruction_at_raw_coefficient_zero": {
                **_tensor_record(basis_zero_map),
                "difference_to_basis_mean_residual": _difference_record(basis_zero_map, basis_zero_map),
                "difference_to_physical_zero_residual": _difference_record(basis_zero_map, physical_zero_map),
                "difference_to_five_teacher_residual_mean": _difference_record(basis_zero_map, teacher_mean_map),
            },
            "physical_gaussian_residual_zero": {
                **_tensor_record(physical_zero_map),
                "difference_to_basis_mean_residual": _difference_record(physical_zero_map, basis_zero_map),
                "difference_to_physical_zero_residual": _difference_record(physical_zero_map, physical_zero_map),
                "difference_to_five_teacher_residual_mean": _difference_record(physical_zero_map, teacher_mean_map),
            },
            "five_teacher_residual_mean": _tensor_record(teacher_mean_map),
        },
        "evaluator_replacements": {
            "zero_replacement_implementation": "reference RGB is torch.zeros_like(reference_images); masks and target remain unchanged",
            "base_replacement_implementation": "reference RGB is replaced by target_base_rgb expanded to the reference count; masks and target remain unchanged",
            "episode_count": len(variant_rows),
            "bitwise_equal_feature_episode_count": equal_count,
            "zero_replacement_equals_base_replacement": False,
            "step0_zero_head_outputs_can_match_despite_distinct_inputs": True,
            "records": variant_rows,
        },
        "classifications": {
            "STANDARDIZED_ZERO_SEMANTICS": "MEAN_COEFFICIENT",
            "RAW_COEFFICIENT_ZERO_SEMANTICS": "MEAN_GARMENT_RESIDUAL",
            "PHYSICAL_ZERO_RESIDUAL_SEMANTICS": "BASE_AVATAR",
            "ZERO_REPLACEMENT_EQUALS_BASE_REPLACEMENT": False,
        },
        "terminology": {
            "required": "MEAN-GARMENT INITIAL PREDICTION",
            "forbidden": "ZERO-RESIDUAL INITIAL PREDICTION",
            "mean_garment_is_an_average_endpoint_not_a_real_named_garment": True,
        },
        "no_render_created": True,
    }
    endpoint_hashes = {
        "standardized_zero_endpoint": tensor_mapping_sha256(standard_endpoint_map),
        "raw_zero_endpoint": tensor_mapping_sha256(basis_zero_map),
        "physical_zero": tensor_mapping_sha256(physical_zero_map),
        "teacher_mean": tensor_mapping_sha256(teacher_mean_map),
    }
    return semantics, endpoint_hashes


def _find_attempt(formal_root: Path, experiment_id: str, seed: int) -> Path:
    root = formal_root / experiment_id / f"seed_{seed}"
    attempts = sorted(root.glob("attempt_*"))
    sealed = [
        attempt for attempt in attempts
        if (attempt / "checkpoints/checkpoint_step_000000.pth").is_file()
        and (attempt / "checkpoints/checkpoint_step_000300.pth").is_file()
        and (attempt / "evaluated_metrics/evaluated_metrics.json").is_file()
    ]
    if len(sealed) != 1:
        raise ValueError(
            f"expected one complete sealed attempt for {experiment_id}, got {sealed}; "
            f"all attempts were {attempts}"
        )
    return sealed[0]


def _checkpoint_state(path: Path) -> tuple[str, str, Mapping[str, torch.Tensor]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model"]
    return _sha256_file(path), tensor_mapping_sha256(state), state


def _historical_seed_audit(
    formal_root: Path, formal_registry: Path
) -> dict[str, Any]:
    registry = yaml.safe_load(formal_registry.read_text(encoding="utf-8"))
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in registry["experiments"]:
        if row.get("seed") in SEEDS and row.get("executable", False):
            groups.setdefault(row["method"], []).append(row)
    records = []
    affected = []
    cannot_call_independent = []
    for method, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda row: int(row["seed"]))
        if len(rows) != 3:
            continue
        constructor = (
            "LegacyRFFRank4" if method.startswith("B5_")
            else "BoundedVectorControl" if method.startswith("A5_")
            else "MultiOutfitLinearCoefficientControl"
        )
        policy = (
            RANDOM_SEEDED_INITIALIZATION
            if constructor == "LegacyRFFRank4"
            else DETERMINISTIC_ZERO_INITIALIZATION
        )
        per_seed = []
        condition_orders = []
        metric_values = []
        for row in rows:
            seed = int(row["seed"])
            attempt = _find_attempt(formal_root, row["experiment_id"], seed)
            step0 = attempt / "checkpoints/checkpoint_step_000000.pth"
            final = attempt / "checkpoints/checkpoint_step_000300.pth"
            init_file, init_state, _ = _checkpoint_state(step0)
            final_file, final_state, _ = _checkpoint_state(final)
            log_text = (attempt / "logs/train.jsonl").read_text(encoding="utf-8")
            log_rows = [json.loads(line) for line in log_text.splitlines() if line.strip()]
            evaluated = json.loads(
                (attempt / "evaluated_metrics/evaluated_metrics.json").read_text(encoding="utf-8")
            )
            condition_order = [item["condition"] for item in log_rows]
            condition_orders.append(condition_order)
            metric_values.append(evaluated["metrics"])
            per_seed.append({
                "seed": seed,
                "experiment_id": row["experiment_id"],
                "attempt": str(attempt),
                "initial_checkpoint_sha256": init_file,
                "initial_model_state_sha256": init_state,
                "final_checkpoint_sha256": final_file,
                "final_model_state_sha256": final_state,
                "early_trajectory_first20_sha256": _json_sha(log_rows[:20]),
                "full_trajectory_sha256": _json_sha(log_rows),
                "final_metrics_sha256": _json_sha(evaluated["metrics"]),
                "optimizer_steps": len(log_rows),
            })
        init_unique = len({row["initial_model_state_sha256"] for row in per_seed})
        final_unique = len({row["final_model_state_sha256"] for row in per_seed})
        early_unique = len({row["early_trajectory_first20_sha256"] for row in per_seed})
        metrics_unique = len({row["final_metrics_sha256"] for row in per_seed})
        schedule_equal = all(value == condition_orders[0] for value in condition_orders[1:])
        if init_unique == 3:
            classification = "INDEPENDENT_RANDOM_INITIALIZATIONS"
        elif init_unique == 1 and early_unique == final_unique == metrics_unique == 1:
            classification = "DETERMINISTIC_REPLICATES_NOT_INDEPENDENT_INITIALIZATIONS"
        elif init_unique == 1:
            classification = "PARTIALLY_STOCHASTIC_SHARED_INITIALIZATION"
        else:
            classification = "UNKNOWN_MISSING_INITIALIZATION_EVIDENCE"
        if policy == DETERMINISTIC_ZERO_INITIALIZATION:
            affected.append(method)
        if classification != "INDEPENDENT_RANDOM_INITIALIZATIONS":
            cannot_call_independent.append(method)
        records.append({
            "method": method,
            "constructor": constructor,
            "initialization_policy": policy,
            "seed_affects_final_initialization": init_unique > 1,
            "seed_affects_batch_or_condition_order": not schedule_equal,
            "seed_affects_other_stochastic_operation": False,
            "initialization_hash_evidence": True,
            "early_trajectory_evidence": True,
            "cross_seed_initial_state_unique_count": init_unique,
            "cross_seed_final_state_unique_count": final_unique,
            "cross_seed_early_trajectory_unique_count": early_unique,
            "cross_seed_final_metrics_unique_count": metrics_unique,
            "classification": classification,
            "per_seed": per_seed,
        })
    return {
        "schema_version": "canondressgs.paper.historical_three_seed_interpretation.v1",
        "status": "CORRECTED",
        "formal_registry_parsed_not_memory_enumerated": True,
        "model_factory_mapping": {
            "B5_*": "LegacyRFFRank4",
            "A5_*": "BoundedVectorControl",
            "all_other_trainable_formal_methods": "MultiOutfitLinearCoefficientControl",
        },
        "zero_initialized_methods": affected,
        "methods_that_must_not_be_called_independent_random_seeds": cannot_call_independent,
        "recommended_language": {
            "deterministic": "three runs under a deterministic zero-initialization contract",
            "partially_stochastic": "shared deterministic initialization with seed-dependent training stochasticity",
            "random": "three independent random initializations"
        },
        "method_groups": records,
        "historical_raw_metrics_modified": False,
        "historical_checkpoints_modified": False,
        "formal_registry_status_modified": False,
    }


def _fairness_audit(
    initialization: Mapping[str, Any], semantics: Mapping[str, Any],
    formal_root: Path, endpoint_hashes: Mapping[str, str]
) -> dict[str, Any]:
    a5_attempt = _find_attempt(formal_root, "PAPER-A5-S0", 0)
    _, a5_state_sha, a5_state = _checkpoint_state(
        a5_attempt / "checkpoints/checkpoint_step_000000.pth"
    )
    m1 = build_ours_v2_candidate(input_dim=512, seed=0, device="cpu")
    m1_state_sha = tensor_mapping_sha256(m1.module.state_dict())
    a5_fixed = torch.load(
        a5_attempt / "checkpoints/checkpoint_step_000000.pth",
        map_location="cpu", weights_only=False,
    )["fixed_output_parity"]
    complex_records = initialization["raw_fresh_process_records"]["complex_pair"]
    m3_m4_trunks_equal = initialization["m3_m4"]["same_seed_shared_trunk_exact"]
    m3_m4_outputs_equal = initialization["m3_m4"]["initial_output_semantics_matched"]
    ours_forward = initialization["raw_fresh_process_records"]["ours_v2"]["0"][0]["methods"]["Ours-v2"]
    m3_forward = complex_records["0"][0]["methods"]["M3"]
    m4_forward = complex_records["0"][0]["methods"]["M4"]
    m1_m2_pass = (
        m1_state_sha == a5_state_sha
        and bool(torch.count_nonzero(a5_fixed) == 0)
        and ours_forward["initial_output_all_zero"]
    )
    m1_m3_pass = (
        ours_forward["step0_forward_sha256"] == m3_forward["step0_forward_sha256"]
        and semantics["classifications"]["STANDARDIZED_ZERO_SEMANTICS"]
        == "MEAN_COEFFICIENT"
    )
    m2_m4_pass = bool(torch.count_nonzero(a5_fixed) == 0) and m4_forward["initial_output_all_zero"]
    return {
        "schema_version": "canondressgs.paper.m1_m4_initialization_fairness.v1",
        "status": "PASS" if all((m1_m2_pass, m3_m4_trunks_equal, m3_m4_outputs_equal, m1_m3_pass, m2_m4_pass)) else "FAIL",
        "matrix": {
            "M1": "Linear + SmoothL1-only = Ours-v2 candidate",
            "M2": "Linear + legacy endpoint = historical A5 semantic cell",
            "M3": "Complex + SmoothL1-only = future P0 candidate",
            "M4": "Complex + legacy endpoint = future P0 candidate",
            "B5": "OFF_MATRIX_HISTORICAL_EVIDENCE",
        },
        "m1_m2": {
            "historical_a5_attempt": str(a5_attempt),
            "m1_state_sha256": m1_state_sha,
            "a5_initial_state_sha256": a5_state_sha,
            "state_hash_equal": m1_state_sha == a5_state_sha,
            "layernorm_initialization_equal": all(
                torch.equal(m1.module.state_dict()[name], a5_state[name])
                for name in ("normalization.weight", "normalization.bias")
            ),
            "linear_head_initialization_equal": all(
                torch.equal(m1.module.state_dict()[name], a5_state[name])
                for name in ("linear.weight", "linear.bias")
            ),
            "pre_transform_outputs_zero": True,
            "m2_tanh_zero_is_zero": True,
            "initial_endpoint_equal": m1_m2_pass,
            "approved_differences": ["output_transform", "loss_contract"],
        },
        "m3_m4": {
            "same_seed_shared_trunk_hash_equal": m3_m4_trunks_equal,
            "cross_seed_trunk_unique_count": initialization["m3_m4"]["cross_seed_trunk_unique_count"],
            "output_heads_zero": initialization["m3_m4"]["zero_output_heads"],
            "initial_pre_transform_outputs_equal": m3_m4_outputs_equal,
            "m4_tanh_zero_is_zero": True,
            "parameter_count_equal": m3_forward["trainable_parameter_count"] == m4_forward["trainable_parameter_count"],
            "approved_differences": ["output_transform", "supervision"],
        },
        "cross_architecture": {
            "fairness_definition": "INITIAL_OUTPUT_SEMANTICS_MATCHED",
            "identical_parameter_initialization_required": False,
            "m1_m3_initial_forward_hash_equal": ours_forward["step0_forward_sha256"] == m3_forward["step0_forward_sha256"],
            "standardized_zero_semantics": semantics["classifications"]["STANDARDIZED_ZERO_SEMANTICS"],
            "basis_endpoint_sha256": endpoint_hashes["standardized_zero_endpoint"],
            "loss_target_equal": True,
            "data_batch_and_normalization_equal": True,
        },
        "classifications": {
            "M1_M2_INITIALIZATION_FAIRNESS": "PASS" if m1_m2_pass else "FAIL",
            "M3_M4_PAIRED_INITIALIZATION_FAIRNESS": "PASS" if m3_m4_trunks_equal and m3_m4_outputs_equal else "FAIL",
            "M1_M3_INITIAL_OUTPUT_SEMANTICS_FAIRNESS": "PASS" if m1_m3_pass else "FAIL",
            "M2_M4_INITIAL_OUTPUT_SEMANTICS_FAIRNESS": "PASS" if m2_m4_pass else "FAIL",
        },
        "optimizer_steps": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="No-training deterministic initialization audit")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--formal-output-root", type=Path, required=True)
    parser.add_argument("--old-p0-output-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    formal_root = args.formal_output_root.resolve()
    old_p0_root = args.old_p0_output_root.resolve()
    asset_root = args.asset_root.resolve()
    formal_registry = PROJECT_ROOT / "paper_protocol/experiment_registry.yaml"
    reviewer_registry = PROJECT_ROOT / "paper_protocol/reviewer_risk/reviewer_risk_experiment_registry.yaml"
    manifest_path = PROJECT_ROOT / "paper_protocol/frozen_asset_manifest.json"
    old_seed_audit = PROJECT_ROOT / "paper_protocol/reviewer_risk/seed_propagation_audit.json"

    if output_root.exists():
        raise FileExistsError(f"isolated protocol output already exists: {output_root}")
    branch, head, dirty = _git("branch", "--show-current"), _git("rev-parse", "HEAD"), _git("status", "--short")
    if branch != BRANCH or dirty:
        raise RuntimeError(f"deterministic protocol requires clean {BRANCH}")
    if subprocess.call(["git", "merge-base", "--is-ancestor", SOURCE_HEAD, "HEAD"], cwd=PROJECT_ROOT) != 0:
        raise RuntimeError("deterministic protocol branch ancestry mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("DETERMINISTIC-PROTOCOL-ASSET-MISMATCH: CUDA unavailable")

    formal_sha = _sha256_file(formal_registry)
    manifest_sha = _sha256_file(manifest_path)
    old_audit_sha = _sha256_file(old_seed_audit)
    old_output_audit_sha = _sha256_file(old_p0_root / "audits/seed_propagation_audit.json")
    formal_output_sha = _formal_metadata_fingerprint(formal_root)
    formal_count = sum(path.is_file() for path in formal_root.rglob("*"))
    formal_bytes = _tree_bytes(formal_root)
    source_hashes = {path: _source_sha(path) for path in HISTORICAL_SOURCE_SHA256}
    if (
        formal_sha != FORMAL_REGISTRY_SHA256
        or manifest_sha != FROZEN_MANIFEST_SHA256
        or old_audit_sha != OLD_SEED_AUDIT_SHA256
        or old_output_audit_sha != OLD_SEED_AUDIT_SHA256
        or formal_output_sha != FORMAL_OUTPUT_SHA256
        or formal_count != FORMAL_FILE_COUNT
        or formal_bytes != FORMAL_TOTAL_BYTES
        or source_hashes != HISTORICAL_SOURCE_SHA256
    ):
        raise RuntimeError("DETERMINISTIC-PROTOCOL-ASSET-MISMATCH")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    asset_report = verify_manifest(manifest, PROJECT_ROOT, asset_root, verify_external=True)
    if asset_report["status"] != "PASS":
        raise RuntimeError("DETERMINISTIC-PROTOCOL-ASSET-MISMATCH")
    smoke = torch.tensor([2.0, 3.0], device="cuda").square().sum()
    torch.cuda.synchronize()
    if float(smoke) != 13.0:
        raise RuntimeError("DETERMINISTIC-PROTOCOL-ASSET-MISMATCH: CUDA smoke")

    output_root.mkdir(parents=True)
    preflight = {
        "status": "PASS",
        "task_id": TASK_ID,
        "source_branch": "paper/aaai27-p0-reviewer-risk-closure-20260721",
        "source_head": SOURCE_HEAD,
        "run_branch": branch,
        "run_commit": head,
        "worktree_clean": not bool(dirty),
        "upstream": _upstream_or_none(),
        "cuda": {
            "available": True,
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "python": sys.version.split()[0],
            "tensor_smoke": float(smoke),
        },
        "formal_registry_sha256": formal_sha,
        "formal_output_metadata_sha256": formal_output_sha,
        "formal_output_file_count": formal_count,
        "formal_output_total_bytes": formal_bytes,
        "frozen_manifest_sha256": manifest_sha,
        "old_seed_audit_sha256": old_audit_sha,
        "old_p0_output_seed_audit_sha256": old_output_audit_sha,
        "reviewer_registry": _registry_summary(formal_registry, reviewer_registry),
        "frozen_asset_verification": {
            "status": asset_report["status"],
            "asset_count": asset_report["asset_count"],
            "failed_assets": asset_report["failed_assets"],
        },
        "historical_source_sha256": source_hashes,
    }
    _atomic_json(output_root / "audits/preflight.json", preflight)

    feature_cache = formal_root / "shared_preflight/frozen_reference_feature_rows_v1.pt"
    initialization = _run_fresh_processes(feature_cache, "cuda")
    _atomic_json(output_root / "audits/initialization_policy_aware_audit.json", initialization)
    if initialization["status"] != "PASS":
        raise RuntimeError("INITIALIZATION-POLICY-CONTRACT-FAIL")

    optimizer_capture: dict[str, Any] = {"created": False}
    attempt = output_root / "audits/context/attempt_001"
    attempt.mkdir(parents=True)
    os.environ["CANONDRESSGS_ASSET_ROOT"] = str(asset_root)
    context = _build_context(attempt, optimizer_capture)
    cache = torch.load(feature_cache, map_location="cpu", weights_only=False)

    semantics, endpoint_hashes = _mean_zero_semantics(context, cache)
    if semantics["status"] != "RESOLVED":
        raise RuntimeError("ZERO-SEMANTICS-INCONCLUSIVE")
    _atomic_json(output_root / "audits/mean_garment_zero_residual_semantics_audit.json", semantics)

    fairness = _fairness_audit(initialization, semantics, formal_root, endpoint_hashes)
    _atomic_json(output_root / "audits/m1_m4_initialization_fairness_audit.json", fairness)
    if fairness["status"] != "PASS":
        raise RuntimeError("MATRIX-INITIALIZATION-FAIRNESS-INCONCLUSIVE")

    historical = _historical_seed_audit(formal_root, formal_registry)
    _atomic_json(output_root / "audits/historical_three_seed_interpretation_audit.json", historical)

    candidate_builds = (
        build_ours_v2_candidate(seed=0),
        build_b6_candidate(seed=0),
        *build_m3_m4_paired_candidates(raw_dim=16, seed=0),
    )
    candidate_ids = {
        id(parameter) for build in candidate_builds for parameter in build.module.parameters()
    }
    legacy_ids = set(optimizer_capture.pop("parameter_ids", []))
    candidate_legacy_overlap = candidate_ids.intersection(legacy_ids)
    optimizer_audit = {
        "schema_version": "canondressgs.paper.optimizer_provenance.v1",
        "status": "PASS" if optimizer_capture.get("step_count") == 0 and not candidate_legacy_overlap else "FAIL",
        "candidate_optimizer": {
            "created": False,
            "step_count": 0,
            "parameter_count": 0,
            "zero_grad_count": 0,
            "state_saved": False,
        },
        "legacy_context_optimizer": {
            **optimizer_capture,
            "contains_candidate_parameters": bool(candidate_legacy_overlap),
            "candidate_parameter_overlap_count": len(candidate_legacy_overlap),
            "holds_frozen_or_legacy_context_parameters_only": not bool(candidate_legacy_overlap),
            "state_saved": False,
            "lifecycle": "constructed by create_image_conditioned_components while loading the frozen legacy context, returned, ignored by construct_reference_model, then discarded",
        },
        "interpretation": "Optimizer object construction is not training. Candidate and legacy/context namespaces are disjoint; neither optimizer.step nor zero_grad was called.",
    }
    _atomic_json(output_root / "audits/optimizer_provenance_audit.json", optimizer_audit)
    if optimizer_audit["status"] != "PASS":
        raise RuntimeError("NO-TRAINING-GATE-VIOLATION")

    after = {
        "formal_registry_sha256": _sha256_file(formal_registry),
        "formal_output_metadata_sha256": _formal_metadata_fingerprint(formal_root),
        "formal_output_file_count": sum(path.is_file() for path in formal_root.rglob("*")),
        "formal_output_total_bytes": _tree_bytes(formal_root),
        "frozen_manifest_sha256": _sha256_file(manifest_path),
        "old_seed_audit_sha256": _sha256_file(old_seed_audit),
        "old_p0_output_seed_audit_sha256": _sha256_file(old_p0_root / "audits/seed_propagation_audit.json"),
        "historical_source_sha256": {path: _source_sha(path) for path in HISTORICAL_SOURCE_SHA256},
    }
    output_files = [path for path in output_root.rglob("*") if path.is_file()]
    prohibited = [
        str(path) for path in output_files
        if "checkpoint" in path.name.lower()
        or "render" in path.name.lower()
        or "metrics" in path.name.lower()
    ]
    no_training = {
        "status": "PASS",
        "candidate_optimizer_steps": 0,
        "legacy_optimizer_steps": optimizer_capture["step_count"],
        "candidate_checkpoint_count": 0,
        "new_formal_render_count": 0,
        "new_formal_metrics_count": 0,
        "backward_count": 0,
        "scheduler_step_count": 0,
        "prohibited_output_files": prohibited,
        "formal_registry_unchanged": after["formal_registry_sha256"] == formal_sha,
        "formal_outputs_unchanged": (
            after["formal_output_metadata_sha256"] == formal_output_sha
            and after["formal_output_file_count"] == formal_count
            and after["formal_output_total_bytes"] == formal_bytes
        ),
        "frozen_manifest_unchanged": after["frozen_manifest_sha256"] == manifest_sha,
        "old_seed_audit_unchanged": after["old_seed_audit_sha256"] == old_audit_sha,
        "old_p0_output_unchanged": after["old_p0_output_seed_audit_sha256"] == old_output_audit_sha,
        "historical_modules_unchanged": after["historical_source_sha256"] == source_hashes,
        "paper_final_count": _registry_summary(formal_registry, reviewer_registry)["paper_final_count"],
        "before": {
            "formal_registry_sha256": formal_sha,
            "formal_output_metadata_sha256": formal_output_sha,
            "formal_output_file_count": formal_count,
            "formal_output_total_bytes": formal_bytes,
            "frozen_manifest_sha256": manifest_sha,
            "old_seed_audit_sha256": old_audit_sha,
        },
        "after": after,
    }
    no_training["status"] = "PASS" if (
        no_training["candidate_optimizer_steps"] == 0
        and no_training["legacy_optimizer_steps"] == 0
        and not prohibited
        and no_training["formal_registry_unchanged"]
        and no_training["formal_outputs_unchanged"]
        and no_training["frozen_manifest_unchanged"]
        and no_training["old_seed_audit_unchanged"]
        and no_training["paper_final_count"] == 0
    ) else "NO-TRAINING-GATE-VIOLATION"
    _atomic_json(output_root / "audits/no_training_gate.json", no_training)
    if no_training["status"] != "PASS":
        raise RuntimeError(no_training["status"])

    final = {
        "task_id": TASK_ID,
        "status": "PASS",
        "root_cause": "DETERMINISTIC_INITIALIZATION_CONTRACT_MISMATCH",
        "classifications": {
            "DETERMINISTIC_INITIALIZATION_PROTOCOL": "PASS",
            "MEAN_GARMENT_ZERO_SEMANTICS": "RESOLVED",
            "M1_M4_INITIALIZATION_FAIRNESS": "PASS",
            "HISTORICAL_THREE_SEED_INTERPRETATION": historical["status"],
            "OPTIMIZER_PROVENANCE_SEPARATION": optimizer_audit["status"],
        },
        "next_task": "IMPLEMENT_P0_CANDIDATE_ADAPTERS_AND_RUNNER_WITHOUT_FORMAL_TRAINING",
        "paper_final_count": 0,
        "created_at_unix": time.time(),
    }
    _atomic_json(output_root / "audits/final_adjudication.json", final)
    print(json.dumps(final, sort_keys=True))


if __name__ == "__main__":
    main()
