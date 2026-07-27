#!/usr/bin/env python3
"""Launch the frozen Subject00 CanonDressGS-Endpoint contract from Base60747."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene.explicit_gaussian_residual_basis import build_svd_basis
from scene.frozen_f2_linear_coefficient_control import FrozenF2ReferenceFeatureExtractor
from scene.full_dressable_dataset import FullDressableTrainingDataset
from scene.gaussian_clothing_residuals import GaussianClothingResiduals
from scene.multi_outfit_linear_coefficient_control import MultiOutfitLinearCoefficientControl
from tools import run_multi_outfit_explicit_basis as multi


TASK_ID = "AAAI27-SUBJECT00-BASE60747-ACCELERATED-METHOD-LAUNCH-001"
SCHEMA = "canondressgs.subject00.canondressgs_endpoint_method_base60747.v1"
BRANCH = "research/subject00-base60747-accelerated-method-launch-20260727"
GARMENTS = ("O01", "O03", "O04")
TRAIN_SLOTS = (0, 7)
QUARANTINE = (
    "subject00_O01_slot04_remaining_attempt005_cand00",
    "subject00_O03_slot04_canary_attempt004_cand00",
)
CHECKPOINT_STEPS = (0, 20, 50, 100, 200, 300)
BASE_SHA = "2d09b1ce6bb19b8b6418314caab8dc1cf92d70e432a9e09e009989332ea4a2f7"
MANIFEST_SHA = "602820292fea47fe8116bf64824e6e2a8313a3ec31b8fc7cdd66a90c6bad33c1"
F2_SHA = "3e4b668fa9f9d5f79c5fd0f56941779c98e78fc3c4bcffa4b3f4268f53f16371"
F2_FINGERPRINT = "3ed5ec6d04d3b9444d234252b5bc5b718ff879f8ec6220ec9704499cce8e77d7"
CONFIG_PATH = REPO_ROOT / "configs/research/subject00_canondressgs_method_base60747_v1.json"
REGISTRY_PATH = (
    REPO_ROOT
    / "paper_protocol/reviewer_risk/subject00_base60747_three_garment_teacher_registry_20260727.json"
)
SUBJECT02_CONFIG = REPO_ROOT / "configs/research/subject02_multi_outfit_explicit_basis_v1.yaml"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write((json.dumps(value, indent=2, sort_keys=True, default=str) + "\n").encode())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def git_state() -> dict[str, Any]:
    branch = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "branch", "--show-current"], text=True
    ).strip()
    head = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True
    ).strip()
    if branch != BRANCH or dirty:
        raise RuntimeError(f"formal method requires clean {BRANCH}; dirty={bool(dirty)}")
    return {"branch": branch, "head": head, "clean": True}


def require_idle_gpu() -> dict[str, Any]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if query:
        raise RuntimeError(f"another GPU process is active: {query}")
    name = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).strip()
    return {"status": "PASS_IDLE", "process_count": 0, "gpu_name": name}


def validate_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = read_json(config_path)
    if config.get("schema_version") != SCHEMA or config.get("task_id") != TASK_ID:
        raise RuntimeError("method config identity changed")
    if config.get("execution_branch") != BRANCH:
        raise RuntimeError("method execution branch changed")
    method = config["method_contract"]
    if (
        method["status"] != "UNIQUE_RECOVERED_FROM_PASSED_SUBJECT02_MAINLINE"
        or method["precedent_head"] != "9ca0f44bdd5480a429e8cd5705df1341746d4381"
        or method["dual_support_policy"] != "DISABLED_EXCLUDED_FROM_PURE_ENDPOINT_PROTOCOL"
    ):
        raise RuntimeError("unique pure-endpoint method contract changed")
    base = config["base"]
    if (
        base["sha256"] != BASE_SHA
        or int(base["step"]) != 60747
        or base["formal_status"] != "USER_AUTHORIZED_PAUSED"
        or base["resume_authorized"] is not False
    ):
        raise RuntimeError("Base60747 binding changed")
    targets = config["targets"]
    if (
        targets["manifest_sha256"] != MANIFEST_SHA
        or int(targets["training_record_count"]) != 22
        or int(targets["quarantine_count"]) != 2
        or tuple(targets["quarantine_exclusions"]) != QUARANTINE
        or int(targets["normal_reference_count"]) != 3
    ):
        raise RuntimeError("formal target/quarantine contract changed")
    f2 = config["frozen_f2"]
    if (
        f2["checkpoint_sha256"] != F2_SHA
        or f2["backbone_fingerprint"] != F2_FINGERPRINT
        or int(f2["controller_input_dimension"]) != 512
        or f2["trainable"] is not False
    ):
        raise RuntimeError("frozen F2 contract changed")
    optimization = config["optimization"]
    if (
        optimization["optimizer"] != "torch.optim.Adam"
        or float(optimization["learning_rate"]) != 0.02
        or float(optimization["weight_decay"]) != 0.0
        or tuple(optimization["betas"]) != (0.9, 0.999)
        or float(optimization["epsilon"]) != 1e-8
        or int(optimization["steps_per_run"]) != 300
        or tuple(optimization["checkpoint_steps"]) != CHECKPOINT_STEPS
        or int(optimization["initial_stability_step"]) != 200
        or float(optimization["gradient_clip_norm"]) != 5.0
        or float(optimization["loss"]["coefficient_smooth_l1_beta"]) != 1.0
        or any(
            float(optimization["loss"][name]) != 0.0
            for name in ("classification_weight", "pairwise_geometry_weight", "render_weight")
        )
    ):
        raise RuntimeError("pure-endpoint optimization contract changed")
    initial = config["protocol"]["initial_execution"]
    if (
        int(initial["rotation"]) != 0
        or int(initial["seed"]) != 0
        or tuple(initial["train_slots"]) != TRAIN_SLOTS
    ):
        raise RuntimeError("initial rotation/seed contract changed")
    if config["paper_eligible"] is not False or config["paper_final"] is not False:
        raise RuntimeError("paper eligibility must remain false")
    return config


def validate_registry(config: Mapping[str, Any]) -> dict[str, Any]:
    registry = read_json(REGISTRY_PATH)
    if registry.get("teacher_garment_count") != 3:
        raise RuntimeError("Teacher registry garment count changed")
    if registry["formal_targets"]["manifest_sha256"] != MANIFEST_SHA:
        raise RuntimeError("Teacher registry target manifest changed")
    for garment in GARMENTS:
        record = registry["garments"][garment]
        binding = config["teachers"]["checkpoints"][garment]
        if (
            record["technical_status"] != "TECHNICAL_PASS"
            or record["paper_eligible"] is not False
            or record["teacher_checkpoint"] != binding["path"]
            or record["teacher_checkpoint_sha256"] != binding["sha256"]
        ):
            raise RuntimeError(f"{garment} Teacher registry binding changed")
    return registry


def input_gate(config: Mapping[str, Any], registry: Mapping[str, Any]) -> dict[str, Any]:
    base = Path(config["base"]["checkpoint"])
    target_root = Path(config["targets"]["root"])
    manifest = target_root / config["targets"]["manifest"]
    f2_checkpoint = Path(config["frozen_f2"]["checkpoint"])
    paths = {"base": base, "manifest": manifest, "f2_checkpoint": f2_checkpoint}
    expected = {"base": BASE_SHA, "manifest": MANIFEST_SHA, "f2_checkpoint": F2_SHA}
    actual = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual[name] = sha256_file(path)
        if actual[name] != expected[name]:
            raise RuntimeError(f"immutable {name} SHA changed")
    teacher_hashes = {}
    for garment in GARMENTS:
        binding = config["teachers"]["checkpoints"][garment]
        path = Path(binding["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        teacher_hashes[garment] = sha256_file(path)
        if teacher_hashes[garment] != binding["sha256"]:
            raise RuntimeError(f"{garment} Teacher checkpoint SHA changed")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            int(payload.get("global_step", -1)) != 1200
            or int(payload.get("optimizer_step", -1)) != 1200
            or payload["metadata"].get("base_checkpoint_sha256") != BASE_SHA
            or int(payload["metadata"].get("quarantine_count_in_optimizer", -1)) != 0
        ):
            raise RuntimeError(f"{garment} Teacher checkpoint metadata is not final/clean")
    manifest_payload = read_json(manifest)
    observations = [
        item
        for outfit in manifest_payload["outfits"]
        for item in outfit["observations"]
    ]
    request_ids = [item["condition_id"] for item in observations]
    if (
        len(request_ids) != 22
        or len(set(request_ids)) != 22
        or set(QUARANTINE).intersection(request_ids)
    ):
        raise RuntimeError("formal target denominator/quarantine gate failed")
    output_root = Path(config["output"]["root"])
    attempt = output_root / config["output"]["attempt"]
    if attempt.exists():
        raise FileExistsError(f"isolated output attempt already exists: {attempt}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output_root.parent)
    minimum = 30 * (1 << 30)
    if usage.free < minimum:
        raise RuntimeError(f"storage gate failed: free={usage.free}, required={minimum}")
    return {
        "paths": {name: str(path) for name, path in paths.items()},
        "sha256": actual,
        "teacher_sha256": teacher_hashes,
        "training_target_count": len(request_ids),
        "quarantine_count": 2,
        "quarantine_count_in_manifest": 0,
        "storage": {
            "status": "PASS",
            "free_bytes": usage.free,
            "minimum_free_bytes": minimum,
        },
        "registry_status": registry["status"],
    }


def teacher_residuals(
    config: Mapping[str, Any],
) -> tuple[dict[str, GaussianClothingResiduals], dict[str, Any]]:
    values: dict[str, GaussianClothingResiduals] = {}
    metadata: dict[str, Any] = {}
    mapping = {
        "raw_xyz": "delta_xyz",
        "raw_log_scaling": "delta_log_scaling",
        "raw_rotvec": "delta_rotvec",
        "raw_opacity": "delta_opacity_logit",
        "raw_sh0": "delta_sh0",
        "raw_shN": "delta_shN",
    }
    for garment in GARMENTS:
        binding = config["teachers"]["checkpoints"][garment]
        payload = torch.load(binding["path"], map_location="cpu", weights_only=False)
        model = payload["model"]
        missing = sorted(set(mapping).difference(model))
        if missing:
            raise RuntimeError(f"{garment} Teacher residual channels missing: {missing}")
        channels = {
            target: model[source].detach().float().contiguous()
            for source, target in mapping.items()
        }
        residual = GaussianClothingResiduals(**channels)
        tensors = tuple(channels.values())
        if any(not torch.isfinite(value).all() for value in tensors):
            raise RuntimeError(f"{garment} Teacher contains NaN/Inf")
        values[garment] = residual
        metadata[garment] = {
            "checkpoint": binding["path"],
            "sha256": binding["sha256"],
            "global_step": int(payload["global_step"]),
            "channel_shapes": {name: list(value.shape) for name, value in channels.items()},
        }
    return values, metadata


def save_basis(
    output_dir: Path, config: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    teachers, teacher_metadata = teacher_residuals(config)
    decomposition = build_svd_basis(
        teachers,
        config["basis"]["channel_bounds"],
        GARMENTS,
        rank=int(config["basis"]["rank"]),
    )
    mean, basis = decomposition.basis.normalized_fields()
    coefficient_matrix = torch.stack(
        [decomposition.teacher_coefficients[garment].double() for garment in GARMENTS]
    )
    coefficient_mean = coefficient_matrix.mean(0)
    coefficient_std = coefficient_matrix.std(0, unbiased=False)
    if torch.any(coefficient_std <= 0) or not torch.isfinite(coefficient_matrix).all():
        raise RuntimeError("rank2 endpoint coordinates are degenerate/nonfinite")
    standardized = {
        garment: (
            decomposition.teacher_coefficients[garment].double() - coefficient_mean
        ) / coefficient_std
        for garment in GARMENTS
    }
    artifact = {
        "schema_version": "canondressgs.subject00.base60747_rank2_endpoint_basis.v1",
        "task_id": TASK_ID,
        "candidate_order": list(GARMENTS),
        "rank": 2,
        "channel_bounds": config["basis"]["channel_bounds"],
        "mean_normalized": {name: value.detach().cpu() for name, value in mean.items()},
        "basis_normalized": {name: value.detach().cpu() for name, value in basis.items()},
        "teacher_coefficients": {
            garment: decomposition.teacher_coefficients[garment].detach().cpu()
            for garment in GARMENTS
        },
        "coefficient_mean": coefficient_mean.cpu(),
        "coefficient_std_population": coefficient_std.cpu(),
        "standardized_teacher_coefficients": {
            garment: value.cpu() for garment, value in standardized.items()
        },
        "metadata": decomposition.metadata,
        "teacher_bindings": teacher_metadata,
        "base_checkpoint_sha256": BASE_SHA,
        "target_manifest_sha256": MANIFEST_SHA,
        "frozen": True,
        "paper_eligible": False,
    }
    path = output_dir / "basis/subject00_base60747_rank2_endpoint_basis.pt"
    atomic_torch_save(path, artifact)
    audit = {
        "status": "PASS",
        "artifact": str(path),
        "artifact_sha256": sha256_file(path),
        "rank": 2,
        "basis_fingerprint": decomposition.metadata["basis_fingerprint"],
        "teacher_coefficients": {
            garment: decomposition.teacher_coefficients[garment].tolist()
            for garment in GARMENTS
        },
        "coefficient_mean": coefficient_mean.tolist(),
        "coefficient_std_population": coefficient_std.tolist(),
        "standardized_teacher_coefficients": {
            garment: value.tolist() for garment, value in standardized.items()
        },
        "teacher_bindings": teacher_metadata,
    }
    atomic_json(output_dir / "basis/basis_audit.json", audit)
    return {garment: value.float() for garment, value in standardized.items()}, audit


def build_f2_extractor(
    config: Mapping[str, Any], git: Mapping[str, Any]
) -> tuple[FrozenF2ReferenceFeatureExtractor, dict[str, Any], dict[str, Any]]:
    subject02 = yaml.safe_load(SUBJECT02_CONFIG.read_text(encoding="utf-8"))
    old_branch, old_head = multi.EXPECTED_BRANCH, multi.EXPECTED_SOURCE_HEAD
    multi.EXPECTED_BRANCH, multi.EXPECTED_SOURCE_HEAD = BRANCH, git["head"]
    try:
        context = multi._build_cs_context(subject02)
    finally:
        multi.EXPECTED_BRANCH, multi.EXPECTED_SOURCE_HEAD = old_branch, old_head
    if context["backbone_before"] != F2_FINGERPRINT:
        raise RuntimeError("frozen F2 backbone fingerprint mismatch")
    device = torch.device("cuda")
    extractor = FrozenF2ReferenceFeatureExtractor(
        context["legacy_model"].clothing_observation_encoder.backbone[:-1],
        feature_dim=int(config["frozen_f2"]["feature_channel_dimension"]),
    ).to(device).eval()
    for parameter in extractor.parameters():
        if parameter.requires_grad:
            raise RuntimeError("frozen F2 unexpectedly trainable")
    provenance = {
        "checkpoint": config["frozen_f2"]["checkpoint"],
        "checkpoint_sha256": F2_SHA,
        "backbone_fingerprint": context["backbone_before"],
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in extractor.parameters() if parameter.requires_grad
        ),
    }
    retained = {
        "base": context["base"],
        "legacy_model": context["legacy_model"],
        "backbone_before": context["backbone_before"],
    }
    return extractor, provenance, retained


def aggregate_native_references(
    extractor: FrozenF2ReferenceFeatureExtractor,
    references: list[dict[str, Any]],
) -> tuple[torch.Tensor, dict[str, Any]]:
    if len(references) != 3:
        raise RuntimeError("normal reference set must contain exactly K=3")
    rows, ids, native_shapes, denominators = [], [], [], []
    with torch.no_grad():
        for reference in references:
            image = reference["rgb"].unsqueeze(0).cuda(non_blocking=False)
            mask = reference["clothing_mask"].unsqueeze(0).cuda(non_blocking=False)
            output = extractor(
                image,
                mask,
                torch.ones(1, 1, device=image.device, dtype=image.dtype),
            )
            rows.append(output.per_reference_f2.reshape(-1))
            ids.append(reference["condition_id"])
            native_shapes.append(list(reference["rgb"].shape))
            denominators.append(float(output.pooling_denominator.min()))
            del image, mask, output
    per_reference = torch.stack(rows)
    feature = torch.cat((per_reference.mean(0), per_reference.max(0).values)).reshape(1, -1)
    if feature.shape != (1, 512) or not torch.isfinite(feature).all():
        raise RuntimeError("native K3 frozen F2 set feature is invalid")
    if min(denominators) <= 0:
        raise RuntimeError("native reference clothing mask pooled to zero")
    return feature.detach(), {
        "reference_condition_ids": ids,
        "target_excluded": True,
        "reference_count": 3,
        "native_shapes_chw": native_shapes,
        "pooling_denominators": denominators,
    }


def extract_features(
    output_dir: Path,
    config: Mapping[str, Any],
    git: Mapping[str, Any],
) -> tuple[dict[tuple[str, int], torch.Tensor], dict[str, Any]]:
    manifest = Path(config["targets"]["root"]) / config["targets"]["manifest"]
    dataset = FullDressableTrainingDataset(manifest, split="train", reference_count=3, seed=0)
    extractor, provenance, retained = build_f2_extractor(config, git)
    features: dict[tuple[str, int], torch.Tensor] = {}
    episodes: dict[str, Any] = {}
    for garment in GARMENTS:
        for slot in TRAIN_SLOTS:
            matches = [
                (index, outfit, observation)
                for index, (outfit, observation) in enumerate(dataset.samples)
                if outfit["outfit_id"] == garment
                and f"_slot{slot:02d}_" in observation["condition_id"]
            ]
            if len(matches) != 1:
                raise RuntimeError(f"{garment} slot{slot:02d} target is not unique/available")
            index, outfit, target = matches[0]
            selected = dataset._references(outfit, target["condition_id"], index)
            if target["condition_id"] in {value["condition_id"] for value in selected}:
                raise RuntimeError("target leaked into its reference set")
            if set(QUARANTINE).intersection(value["condition_id"] for value in selected):
                raise RuntimeError("quarantine record entered method reference sampler")
            references = [dataset._observation(outfit, value, True) for value in selected]
            feature, episode = aggregate_native_references(extractor, references)
            features[(garment, slot)] = feature.cpu()
            episodes[f"{garment}/slot{slot:02d}"] = {
                "target_condition_id": target["condition_id"],
                **episode,
            }
    backbone_after = multi.o01._state_fingerprint(
        retained["legacy_model"].clothing_observation_encoder.backbone.state_dict()
    )
    if backbone_after != retained["backbone_before"]:
        raise RuntimeError("frozen F2 backbone mutated during feature extraction")
    payload = {
        "schema_version": "canondressgs.subject00.native_k3_f2_features.v1",
        "features": {
            f"{garment}/slot{slot:02d}": value
            for (garment, slot), value in features.items()
        },
        "episodes": episodes,
        "provenance": provenance,
        "backbone_after": backbone_after,
        "quarantine_count": 0,
    }
    path = output_dir / "features/rotation00_train_native_k3_f2.pt"
    atomic_torch_save(path, payload)
    audit = {
        "status": "PASS",
        "artifact": str(path),
        "artifact_sha256": sha256_file(path),
        "episode_count": len(features),
        "episodes": episodes,
        "provenance": provenance,
        "backbone_bitwise_frozen": backbone_after == retained["backbone_before"],
        "quarantine_count": 0,
        "target_or_teacher_field_in_prediction_forward": False,
    }
    atomic_json(output_dir / "features/feature_audit.json", audit)
    del extractor, retained
    torch.cuda.empty_cache()
    return features, audit


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def save_checkpoint(
    path: Path,
    *,
    step: int,
    controller: MultiOutfitLinearCoefficientControl,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    history: list[dict[str, Any]],
    config: Mapping[str, Any],
    git: Mapping[str, Any],
    basis_audit: Mapping[str, Any],
    feature_audit: Mapping[str, Any],
) -> None:
    atomic_torch_save(
        path,
        {
            "schema_version": "canondressgs.subject00.canondressgs_endpoint_checkpoint.v1",
            "task_id": TASK_ID,
            "rotation": 0,
            "seed": 0,
            "global_step": step,
            "optimizer_step": step,
            "model": controller.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": rng_state(),
            "condition_position": step % 2,
            "history": history,
            "bindings": {
                "base_checkpoint": config["base"]["checkpoint"],
                "base_checkpoint_sha256": BASE_SHA,
                "target_manifest_sha256": MANIFEST_SHA,
                "teacher_checkpoint_sha256": {
                    garment: config["teachers"]["checkpoints"][garment]["sha256"]
                    for garment in GARMENTS
                },
                "basis_artifact_sha256": basis_audit["artifact_sha256"],
                "feature_artifact_sha256": feature_audit["artifact_sha256"],
                "f2_checkpoint_sha256": F2_SHA,
                "f2_backbone_fingerprint": F2_FINGERPRINT,
                "quarantine_count": 0,
                "git": dict(git),
            },
            "paper_eligible": False,
        },
    )


def nearest_endpoint(
    prediction: torch.Tensor, targets: Mapping[str, torch.Tensor]
) -> tuple[str, dict[str, float]]:
    distances = {
        garment: float((prediction - target).square().sum())
        for garment, target in targets.items()
    }
    selected = min(GARMENTS, key=lambda garment: (distances[garment], GARMENTS.index(garment)))
    return selected, distances


def evaluate(
    controller: MultiOutfitLinearCoefficientControl,
    features: Mapping[tuple[str, int], torch.Tensor],
    targets: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    controller.eval()
    rows, squared_errors, correct = [], [], 0
    with torch.no_grad():
        for slot in TRAIN_SLOTS:
            for garment in GARMENTS:
                prediction = controller(features[(garment, slot)].cuda()).standardized_coefficients
                target = targets[garment].to(prediction)
                selected, distances = nearest_endpoint(prediction, {
                    name: value.to(prediction) for name, value in targets.items()
                })
                error = (prediction - target).square()
                squared_errors.extend(error.tolist())
                correct += int(selected == garment)
                rows.append({
                    "garment": garment,
                    "slot": slot,
                    "prediction": prediction.tolist(),
                    "target": target.tolist(),
                    "selected_endpoint": selected,
                    "correct": selected == garment,
                    "squared_distances": distances,
                })
    count = len(rows)
    return {
        "status": "PASS_FINITE",
        "rotation": 0,
        "seed": 0,
        "checkpoint_step": 300,
        "episode_count": count,
        "finite_prediction_count": count,
        "standardized_coefficient_rmse": float(np.sqrt(np.mean(squared_errors))),
        "nearest_endpoint_correct": correct,
        "nearest_endpoint_total": count,
        "nearest_endpoint_accuracy": correct / count,
        "rows": rows,
        "continuous_prediction_rendered": False,
        "human_visual_status": None,
        "scientific_pass": None,
        "paper_eligible": False,
    }


def train_initial_run(
    output_dir: Path,
    config: Mapping[str, Any],
    git: Mapping[str, Any],
    features_cpu: Mapping[tuple[str, int], torch.Tensor],
    targets_cpu: Mapping[str, torch.Tensor],
    basis_audit: Mapping[str, Any],
    feature_audit: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    features = {key: value.to(device) for key, value in features_cpu.items()}
    targets = {key: value.to(device) for key, value in targets_cpu.items()}
    controller = MultiOutfitLinearCoefficientControl(512, 2).to(device)
    if sum(parameter.numel() for parameter in controller.parameters()) != 2050:
        raise RuntimeError("controller parameter count changed")
    optimizer = torch.optim.Adam(
        controller.parameters(),
        lr=0.02,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    checkpoint_root = output_dir / "checkpoints/rotation_00_seed_000"
    history: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats()
    save_checkpoint(
        checkpoint_root / "step_000000.pth",
        step=0,
        controller=controller,
        optimizer=optimizer,
        scheduler=scheduler,
        history=history,
        config=config,
        git=git,
        basis_audit=basis_audit,
        feature_audit=feature_audit,
    )
    initial_state = {
        name: value.detach().cpu().clone() for name, value in controller.state_dict().items()
    }
    all_gradients_finite = True
    initial_stability: dict[str, Any] | None = None
    for step in range(1, 301):
        slot = TRAIN_SLOTS[(step - 1) % len(TRAIN_SLOTS)]
        optimizer.zero_grad(set_to_none=True)
        predictions = torch.stack([
            controller(features[(garment, slot)]).standardized_coefficients
            for garment in GARMENTS
        ])
        target_batch = torch.stack([targets[garment] for garment in GARMENTS])
        loss = F.smooth_l1_loss(predictions, target_batch, beta=1.0, reduction="mean")
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at optimizer step {step}")
        loss.backward()
        gradients_finite = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in controller.parameters()
        )
        all_gradients_finite = all_gradients_finite and gradients_finite
        if not gradients_finite:
            raise FloatingPointError(f"nonfinite gradient at optimizer step {step}")
        gradient_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), 5.0)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"nonfinite gradient norm at optimizer step {step}")
        optimizer.step()
        scheduler.step()
        row = {
            "optimizer_step": step,
            "condition_slot": slot,
            "loss": float(loss.detach()),
            "gradient_norm_before_clip": float(gradient_norm),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        if step % 10 == 0 or step in CHECKPOINT_STEPS:
            atomic_json(output_dir / "training/latest.json", {
                "status": "RUNNING",
                "rotation": 0,
                "seed": 0,
                **row,
                "nan_inf_status": "NONE",
                "oom_status": "NONE",
                "quarantine_count": 0,
                "paper_eligible": False,
            })
        if step == 200:
            initial_stability = {
                "status": "PASS",
                "initial_stability_step": 200,
                "optimizer_step_monotonic": [item["optimizer_step"] for item in history]
                == list(range(1, 201)),
                "loss_finite": all(np.isfinite(item["loss"]) for item in history),
                "gradients_finite": all_gradients_finite,
                "nan_inf_status": "NONE",
                "oom_status": "NONE",
                "target_teacher_binding": "PASS",
                "quarantine_count": 0,
                "output_root_isolated": True,
                "base_checkpoint_mutations": 0,
            }
            if not all(
                initial_stability[name]
                for name in ("optimizer_step_monotonic", "loss_finite", "gradients_finite")
            ):
                raise RuntimeError("step200 initial stability gate failed")
            atomic_json(output_dir / "training/initial_stability_step200.json", initial_stability)
        if step in CHECKPOINT_STEPS:
            save_checkpoint(
                checkpoint_root / f"step_{step:06d}.pth",
                step=step,
                controller=controller,
                optimizer=optimizer,
                scheduler=scheduler,
                history=history,
                config=config,
                git=git,
                basis_audit=basis_audit,
                feature_audit=feature_audit,
            )
    if initial_stability is None:
        raise RuntimeError("step200 initial stability audit was not written")
    trainable_changed = {
        name: not torch.equal(value, controller.state_dict()[name].detach().cpu())
        for name, value in initial_state.items()
    }
    if not trainable_changed["linear.weight"]:
        raise RuntimeError("controller linear weight did not train")
    evaluation = evaluate(controller, features, targets)
    atomic_json(output_dir / "evaluation/rotation00_seed000_step300.json", evaluation)
    result = {
        "status": "COMPLETE_INITIAL_ROTATION_SEED",
        "method_training_authorized": True,
        "rotation": 0,
        "seed": 0,
        "optimizer_steps_completed": 300,
        "current_step": 300,
        "planned_optimizer_steps": 3600,
        "initial_stability_status": "PASS",
        "loss_initial": history[0]["loss"],
        "loss_final": history[-1]["loss"],
        "loss_minimum": min(item["loss"] for item in history),
        "optimizer_step_monotonic": [item["optimizer_step"] for item in history]
        == list(range(1, 301)),
        "loss_finite": all(np.isfinite(item["loss"]) for item in history),
        "gradients_finite": all_gradients_finite,
        "nan_inf_status": "NONE",
        "oom_status": "NONE",
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "trainable_parameter_changes": trainable_changed,
        "quarantine_count": 0,
        "base_checkpoint_mutations": 0,
        "target_mutations": 0,
        "mask_mutations": 0,
        "teacher_checkpoint_mutations": 0,
        "evaluation": {
            key: evaluation[key]
            for key in (
                "standardized_coefficient_rmse",
                "nearest_endpoint_correct",
                "nearest_endpoint_total",
                "nearest_endpoint_accuracy",
            )
        },
        "paper_eligible": False,
    }
    atomic_json(output_dir / "training/initial_run_summary.json", result)
    return result, evaluation


def immutable_post_audit(config: Mapping[str, Any], before: Mapping[str, Any]) -> dict[str, Any]:
    after = {
        "base": sha256_file(Path(config["base"]["checkpoint"])),
        "manifest": sha256_file(
            Path(config["targets"]["root"]) / config["targets"]["manifest"]
        ),
        "teachers": {
            garment: sha256_file(Path(config["teachers"]["checkpoints"][garment]["path"]))
            for garment in GARMENTS
        },
    }
    checks = {
        "base_unchanged": after["base"] == before["sha256"]["base"] == BASE_SHA,
        "manifest_unchanged": after["manifest"] == before["sha256"]["manifest"] == MANIFEST_SHA,
        "teachers_unchanged": after["teachers"] == before["teacher_sha256"],
        "base_checkpoint_mutations": 0,
        "target_mutations": 0,
        "mask_mutations": 0,
        "teacher_checkpoint_mutations": 0,
    }
    if not all(checks[name] for name in ("base_unchanged", "manifest_unchanged", "teachers_unchanged")):
        raise RuntimeError("immutable input mutation detected")
    return {"status": "PASS", "sha256_after": after, "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    config = validate_config(args.config.resolve())
    git = git_state()
    gpu = require_idle_gpu()
    registry = validate_registry(config)
    gate = input_gate(config, registry)
    output_dir = Path(config["output"]["root"]) / config["output"]["attempt"]
    for directory in ("contract", "input_audit", "basis", "features", "training", "evaluation"):
        (output_dir / directory).mkdir(parents=True, exist_ok=False)
    atomic_json(output_dir / "contract/config_resolved.json", config)
    atomic_json(output_dir / "input_audit/preflight.json", {
        "status": "PASS",
        "task_id": TASK_ID,
        "git": git,
        "gpu": gpu,
        **gate,
        "method_contract_status": config["method_contract"]["status"],
        "method_training_authorized": True,
        "paper_eligible": False,
    })
    atomic_json(output_dir / "RUN_STATUS.json", {
        "task_id": TASK_ID,
        "status": "RUNNING",
        "stage": "BASIS_CONSTRUCTION",
        "optimizer_steps_completed": 0,
        "current_step": 0,
        "initial_stability_status": "PENDING",
        "paper_eligible": False,
    })
    try:
        standardized_targets, basis_audit = save_basis(output_dir, config)
        atomic_json(output_dir / "RUN_STATUS.json", {
            "task_id": TASK_ID,
            "status": "RUNNING",
            "stage": "FROZEN_F2_FEATURE_EXTRACTION",
            "optimizer_steps_completed": 0,
            "current_step": 0,
            "initial_stability_status": "PENDING",
            "paper_eligible": False,
        })
        features, feature_audit = extract_features(output_dir, config, git)
        atomic_json(output_dir / "RUN_STATUS.json", {
            "task_id": TASK_ID,
            "status": "RUNNING",
            "stage": "CONTROLLER_TRAINING_ROTATION00_SEED000",
            "optimizer_steps_completed": 0,
            "current_step": 0,
            "initial_stability_status": "PENDING",
            "paper_eligible": False,
        })
        result, evaluation = train_initial_run(
            output_dir,
            config,
            git,
            features,
            standardized_targets,
            basis_audit,
            feature_audit,
        )
        immutable = immutable_post_audit(config, gate)
        atomic_json(output_dir / "input_audit/immutable_post_audit.json", immutable)
        final = {
            "task_id": TASK_ID,
            "status": "METHOD_TRAINING_RUNNING_INITIAL_ROTATION_COMPLETE",
            "classification": (
                "SUBJECT00_BASE60747_THREE_GARMENT_TEACHERS_PASS_"
                "METHOD_TRAINING_RUNNING_INITIAL_STABILITY_PASS"
            ),
            "method_contract_status": config["method_contract"]["status"],
            "method_training_authorized": True,
            "optimizer_steps_completed": result["optimizer_steps_completed"],
            "current_step": result["current_step"],
            "planned_optimizer_steps": result["planned_optimizer_steps"],
            "initial_stability_status": result["initial_stability_status"],
            "nan_inf_status": result["nan_inf_status"],
            "oom_status": result["oom_status"],
            "peak_vram_bytes": result["peak_vram_bytes"],
            "storage_gate_status": gate["storage"]["status"],
            "base_checkpoint_mutations": 0,
            "target_mutations": 0,
            "mask_mutations": 0,
            "paper_modifications": 0,
            "paper_eligible": False,
            "paper_final": False,
            "initial_run": result,
            "evaluation": {
                key: evaluation[key]
                for key in (
                    "standardized_coefficient_rmse",
                    "nearest_endpoint_correct",
                    "nearest_endpoint_total",
                    "nearest_endpoint_accuracy",
                )
            },
            "next_task": (
                "Execute remaining preregistered rotations/seeds and fair Base60747 baselines; "
                "do not alter the paper before human visual/scientific adjudication."
            ),
        }
        atomic_json(output_dir / "FINAL_REPORT.json", final)
        atomic_json(output_dir / "RUN_STATUS.json", {
            "task_id": TASK_ID,
            "status": "RUNNING",
            "stage": "INITIAL_ROTATION_SEED_COMPLETE_REMAINING_RUNS_PENDING",
            "optimizer_steps_completed": result["optimizer_steps_completed"],
            "current_step": result["current_step"],
            "initial_stability_status": result["initial_stability_status"],
            "nan_inf_status": "NONE",
            "oom_status": "NONE",
            "paper_eligible": False,
        })
        print(json.dumps(final, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        atomic_json(output_dir / "RUN_STATUS.json", {
            "task_id": TASK_ID,
            "status": "FAIL",
            "stage": "METHOD_LAUNCH_FAILED",
            "error_type": type(error).__name__,
            "error": str(error),
            "paper_eligible": False,
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
