from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_dressable as training  # noqa: E402
from scene.explicit_gaussian_residual_basis import (  # noqa: E402
    ExplicitGaussianResidualBasis,
    ReferenceBasisCoefficientPredictor,
)
from scene.gaussian_clothing_residuals import CHANNELS, GaussianClothingResiduals  # noqa: E402
from scene.reference_basis_coefficient_fusion import (  # noqa: E402
    MaskAwareReferenceTokenEncoderV1,
    ReferenceSetCoefficientFusionV1,
    ReferenceSetCoefficientOutput,
    ReferenceTokenOutput,
    build_reference_coefficient_fusion,
)
from tools import diagnose_image_conditioned_overfit_failure as diagnosis  # noqa: E402
from tools import run_explicit_gaussian_residual_basis as legacy  # noqa: E402
from tools import run_image_conditioned_overfit_o01 as o01  # noqa: E402
from tools import run_residual_decoder_capacity_v7 as v7  # noqa: E402
from tools import run_residual_field_parameterization as parameterization  # noqa: E402
from tools.run_module4b_canonical_oracle_micropilot import (  # noqa: E402
    _base_gradient_count,
    _base_named_tensors,
    _tensor_state_fingerprint,
)
from utils.mmlphuman_anchor_deformation import MMLPHumanAnchorDeformationAdapter  # noqa: E402


SCHEMA = "canondressgs.reference_basis_coefficient_fusion.v1"
TASK_ID = "SUBJECT02-REFERENCE-TO-BASIS-COEFFICIENT-FUSION-001"
EXPECTED_BRANCH = "research/reference-basis-coefficient-fusion-20260720"
EXPECTED_SOURCE_HEAD = "4ff14cd52b5b204908b47173915b3d66ff51f58b"
OUTFITS = ("O01", "O08")
CONDITIONS = o01.CONDITIONS
VIEWS = o01.VIEWS
TEACHER = {"O01": -1.0, "O08": 1.0}
FORBIDDEN_FORWARD_FIELDS = {
    "target_rgb", "target_edit_rgb", "target_base_rgb", "target_mask",
    "target_pose", "target_camera", "outfit_id", "cloth_id", "teacher",
    "teacher_coefficient", "diagnostic_latent",
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True, default=str) + "\n")
        handle.flush()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=PROJECT_ROOT, text=True).strip()


def object_fingerprint(value: Any) -> str:
    return o01.object_fingerprint(value)


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA or config.get("task_id") != TASK_ID:
        raise ValueError("reference-basis fusion schema/task mismatch")
    if config.get("branch") != EXPECTED_BRANCH or config.get("source_head") != EXPECTED_SOURCE_HEAD:
        raise ValueError("reference-basis fusion governance contract changed")
    if tuple(config.get("outfits", ())) != OUTFITS or tuple(config.get("conditions", ())) != CONDITIONS:
        raise ValueError("two-outfit/four-view protocol changed")
    if config["coefficient_fusion"]["type"] != "mask_aware_reference_set_v1":
        raise ValueError("formal run requires mask_aware_reference_set_v1")
    if int(config["training"]["max_steps"]) != 500:
        raise ValueError("formal micro-pilot must contain exactly 500 optimizer steps")
    if config["training"]["paired_outfits_per_step"] != ["O01", "O08"]:
        raise ValueError("paired-batch contract changed")
    frozen_acceptance = {
        "coefficient_mae_max": 0.10, "o01_mean_max": -0.80,
        "o08_mean_min": 0.80, "separation_min": 1.60,
        "correct_sign_count_min": 8, "swapped_coefficient_margin_min": 1.20,
        "correct_episode_wins_min": 8, "reference_replacement_change_min": 0.20,
        "permutation_max_abs_diff_max": 1.0e-5,
    }
    if config["acceptance"] != frozen_acceptance:
        raise ValueError("pre-registered acceptance thresholds changed")
    if any(bool(value) for value in config["permissions"].values()):
        raise ValueError("forbidden permission enabled")
    return config


def environment_snapshot() -> dict[str, Any]:
    result = o01.environment_snapshot()
    result["command"] = [sys.executable, *sys.argv]
    return result


def _legacy_config() -> dict[str, Any]:
    return yaml.safe_load(
        (PROJECT_ROOT / "configs/research/subject02_explicit_gaussian_residual_basis_v1.yaml").read_text(
            encoding="utf-8"
        )
    )


def _verify_governance() -> dict[str, Any]:
    branch = git_output("branch", "--show-current")
    head = git_output("rev-parse", "HEAD")
    dirty = git_output("status", "--short")
    if branch != EXPECTED_BRANCH or dirty:
        raise RuntimeError(
            f"formal run requires clean {EXPECTED_BRANCH}; branch={branch} dirty={bool(dirty)}"
        )
    if subprocess.call(
        ["git", "merge-base", "--is-ancestor", EXPECTED_SOURCE_HEAD, "HEAD"], cwd=PROJECT_ROOT
    ) != 0:
        raise RuntimeError("formal branch is not descended from the frozen explicit-basis HEAD")
    return {"branch": branch, "head": head, "clean": True, "source_head": EXPECTED_SOURCE_HEAD}


def _path_contract(config: Mapping[str, Any]) -> tuple[dict[str, Path], dict[str, str]]:
    paths = {
        "source_manifest": Path(config["inputs"]["source_manifest"]),
        "stable_protected_attribution": Path(config["inputs"]["stable_protected_attribution"]),
        "explicit_basis_artifact": Path(config["inputs"]["explicit_basis_artifact"]),
        "legacy_stage_c_checkpoint": Path(config["inputs"]["legacy_stage_c_checkpoint"]),
        "p1_predictions": Path(config["inputs"]["p1_predictions"]),
        "oracle_o01_checkpoint": Path(config["inputs"]["oracle_o01_checkpoint"]),
        "oracle_o08_checkpoint": Path(config["inputs"]["oracle_o08_checkpoint"]),
        "base_checkpoint": Path(config["base"]["model_dir"]) / config["base"]["checkpoint_path"],
        "lbs_grid": Path(config["base"]["lbs_grid_path"]),
    }
    expected = {
        "source_manifest": config["inputs"]["source_manifest_sha256"],
        "stable_protected_attribution": config["inputs"]["stable_protected_attribution_sha256"],
        "explicit_basis_artifact": config["inputs"]["explicit_basis_artifact_sha256"],
        "legacy_stage_c_checkpoint": config["inputs"]["legacy_stage_c_checkpoint_sha256"],
        "p1_predictions": config["inputs"]["p1_predictions_sha256"],
        "oracle_o01_checkpoint": config["inputs"]["oracle_o01_checkpoint_sha256"],
        "oracle_o08_checkpoint": config["inputs"]["oracle_o08_checkpoint_sha256"],
        "base_checkpoint": config["base"]["checkpoint_sha256"],
    }
    actual = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual[name] = sha256(path)
        if name in expected and actual[name] != expected[name]:
            raise ValueError(f"immutable input SHA mismatch: {name}")
    return paths, actual


def _load_basis(path: Path, device: torch.device) -> tuple[ExplicitGaussianResidualBasis, dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "canondressgs.explicit_gaussian_residual_basis_artifact.v1":
        raise ValueError("explicit basis artifact schema mismatch")
    basis = ExplicitGaussianResidualBasis(
        payload["mean_normalized"], payload["basis_normalized"], payload["channel_bounds"]
    ).to(device)
    if basis.fingerprint() != payload["metadata"]["basis_fingerprint"]:
        raise ValueError("explicit basis fingerprint mismatch")
    coefficients = {
        name: value.to(device=device, dtype=next(basis.buffers()).dtype)
        for name, value in payload["teacher_coefficients"].items()
    }
    for parameter in basis.parameters():
        parameter.requires_grad_(False)
    return basis, coefficients, payload


def build_context(config: dict[str, Any], output_dir: Path, *, create: bool) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("formal reference-basis fusion requires CUDA")
    device = torch.device("cuda")
    seed = int(config["training"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    governance = _verify_governance()
    paths, hashes = _path_contract(config)
    explicit_attempt = Path(config["inputs"]["explicit_basis_attempt"])
    explicit_tree_before = legacy.immutable_tree_metadata_fingerprint(explicit_attempt)
    if create:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"formal attempt already exists and is non-empty: {output_dir}")
        for directory in (
            "contract", "input_audit", "stage_0", "training/checkpoints",
            "training/milestones", "evaluation", "visual_acceptance", "final_adjudication",
        ):
            (output_dir / directory).mkdir(parents=True, exist_ok=True)
        atomic_text(output_dir / "contract/config_resolved.yaml", yaml.safe_dump(config, sort_keys=False))
        atomic_text(output_dir / "contract/command.txt", " ".join([sys.executable, *sys.argv]))
        atomic_json(output_dir / "input_audit/environment.json", environment_snapshot())
        atomic_json(output_dir / "RUN_STATUS.json", {
            "task_id": TASK_ID, "status": "RUNNING", "stage": "INPUT_AUDIT",
            "optimizer_steps": 0, "updated_at_unix": time.time(),
        })

    samples, episodes, protocol = diagnosis._generic_outfit_data(paths["source_manifest"])
    base = training.load_frozen_mmlphuman_base(
        config["base"]["model_dir"], paths["base_checkpoint"], device=device
    )
    if int(base._xyz.shape[0]) != int(config["base"]["gaussian_count"]):
        raise ValueError("base Gaussian count changed")
    protected_cpu, attribution = o01._load_stable_protected_mask(
        paths["stable_protected_attribution"], int(config["inputs"]["stable_protected_count"]),
        int(base._xyz.shape[0]),
    )
    protected_mask = protected_cpu.to(device)
    base_before = _tensor_state_fingerprint(_base_named_tensors(base))
    adapter = MMLPHumanAnchorDeformationAdapter.from_mmlphuman_base(
        base, lbs_grid_path=paths["lbs_grid"]
    )
    background = torch.tensor(config["render"]["background"], device=device, dtype=base._xyz.dtype)
    for key in list(samples):
        samples[key] = diagnosis._to_device_nested(samples[key], device)
        episodes[key] = diagnosis._to_device_nested(episodes[key], device)
    geometries = {}
    for condition in CONDITIONS:
        geometries[condition] = training.prepare_real_reference_geometry(
            base, adapter, episodes[f"O01/{condition}"], background
        )
        for outfit in OUTFITS:
            key = f"{outfit}/{condition}"
            support, _ = o01._base_only_protected_support(
                base, samples[key], protected_mask, background,
                float(config["render"]["protected_support_alpha_threshold"]),
            )
            samples[key]["target_protected_mask"] = torch.maximum(
                samples[key]["target_protected_mask"], support
            )

    # Recreate the frozen legacy backbone at the exact seed used by the old
    # Stage-C run; its frozen weights were intentionally not stored in that checkpoint.
    construction_rng = legacy.rng_state()
    old_config = _legacy_config()
    old_seed = int(old_config["stage_b"]["seed"])
    random.seed(old_seed); np.random.seed(old_seed); torch.manual_seed(old_seed); torch.cuda.manual_seed_all(old_seed)
    legacy_model, graph = legacy.construct_reference_model(base, old_config, device)
    legacy_predictor = ReferenceBasisCoefficientPredictor(
        int(old_config["model"]["embedding_dim"]), int(old_config["model"]["local_feature_dim"]),
        1, int(old_config["stage_c"]["predictor_hidden_dim"]),
    ).to(device)
    checkpoint = torch.load(paths["legacy_stage_c_checkpoint"], map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != "C" or int(checkpoint.get("global_step", -1)) != 496:
        raise ValueError("legacy Stage-C checkpoint contract changed")
    legacy._load_stage_c_state(legacy_model, legacy_predictor, checkpoint["model"])
    legacy.set_requires_grad(legacy_model, False); legacy.set_requires_grad(legacy_predictor, False)
    legacy_model.eval(); legacy_predictor.eval()
    legacy.restore_rng(construction_rng)
    backbone_before = o01._state_fingerprint(
        legacy_model.clothing_observation_encoder.backbone.state_dict()
    )
    basis, teacher_coefficients, basis_payload = _load_basis(paths["explicit_basis_artifact"], device)

    input_audit = audit_inputs(config, samples, episodes, protocol)
    if create:
        atomic_json(output_dir / "input_audit/input_manifest.json", {
            "task_id": TASK_ID, "git": governance,
            "paths": {name: str(path) for name, path in paths.items()}, "sha256": hashes,
            "explicit_basis_attempt": str(explicit_attempt),
            "explicit_basis_attempt_tree_metadata_fingerprint": explicit_tree_before,
            "basis_fingerprint": basis.fingerprint(), "legacy_graph": graph,
            "historical_checkpoint_warning": {
                "historical_sha256": config["base"]["historical_checkpoint_sha256_warning"],
                "current_sha256": config["base"]["checkpoint_sha256"], "same_file": False,
            },
            "prediction_forward_fields": [
                "reference RGB", "reference clothing mask", "reference foreground mask",
                "reference pose", "reference camera",
            ],
            "forbidden_prediction_fields": sorted(FORBIDDEN_FORWARD_FIELDS),
            "target_view_used_in_prediction_forward": False,
            "teacher_usage": "coefficient supervision and evaluation only",
            "attribution": attribution,
        })
        atomic_json(output_dir / "input_audit/reference_input_audit.json", input_audit)
    return {
        "config": config, "output_dir": output_dir, "paths": paths, "samples": samples,
        "episodes": episodes, "protocol": protocol, "base": base, "background": background,
        "protected_mask": protected_mask, "geometries": geometries,
        "legacy_model": legacy_model, "legacy_predictor": legacy_predictor,
        "basis": basis, "teacher_coefficients": teacher_coefficients,
        "basis_payload": basis_payload, "base_before": base_before,
        "backbone_before": backbone_before, "explicit_tree_before": explicit_tree_before,
        "explicit_attempt": explicit_attempt, "input_audit": input_audit,
    }


def audit_inputs(
    config: Mapping[str, Any], samples: Mapping[str, Any], episodes: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    rows, errors = {}, []
    rgb_hashes, cloth_hashes, foreground_hashes = {}, {}, {}
    for key in sorted(episodes):
        episode = episodes[key]
        images = episode["reference_images"].detach().float().cpu()
        clothing = episode["reference_cloth_masks"].detach().float().cpu()
        foreground = episode["reference_foreground_masks"].detach().float().cpu()
        pose = episode["reference_poses"].detach().float().cpu()
        w2c = episode["reference_w2c"].detach().float().cpu()
        condition_ids = list(episode["reference_condition_ids"])
        target = samples[key]["target_condition_id"]
        if target in condition_ids or len(condition_ids) != len(set(condition_ids)):
            errors.append(f"{key}: target overlap or duplicate reference")
        if not all(torch.isfinite(value).all() for value in (images, clothing, foreground, pose, w2c)):
            errors.append(f"{key}: nonfinite reference input")
        outside = float((clothing > foreground + 1e-4).float().mean())
        areas = clothing.mean(dim=(1, 2, 3)).tolist()
        if any(value <= 0 or value >= 1 for value in areas):
            errors.append(f"{key}: empty or full clothing mask")
        if outside > 0:
            errors.append(f"{key}: clothing mask outside foreground")
        source_records = []
        outfit = key.split("/")[0]
        for condition in condition_ids:
            record = samples[f"{outfit}/{condition}"]["source_record"]
            paths = {
                "rgb": Path(record["rgb"]),
                "clothing_mask": Path(record["clothing_mask"]),
                "foreground_mask": Path(record["foreground_mask"]),
            }
            hashes = {name: sha256(path) for name, path in paths.items()}
            rgb_hashes[f"{outfit}/{condition}"] = hashes["rgb"]
            cloth_hashes[f"{outfit}/{condition}"] = hashes["clothing_mask"]
            foreground_hashes[f"{outfit}/{condition}"] = hashes["foreground_mask"]
            source_records.append({
                "condition": condition, "paths": {name: str(path) for name, path in paths.items()},
                "sha256": hashes,
            })
        rows[key] = {
            "target": target, "references": condition_ids,
            "rgb_shape": list(images.shape), "pose_shape": list(pose.shape),
            "w2c_shape": list(w2c.shape), "image_range": [float(images.min()), float(images.max())],
            "clothing_mask_area": areas, "foreground_mask_area": foreground.mean(dim=(1, 2, 3)).tolist(),
            "clothing_outside_foreground_ratio": outside, "source_records": source_records,
            "per_reference_fingerprints": {
                "rgb": [object_fingerprint(images[index]) for index in range(images.shape[0])],
                "clothing_mask": [object_fingerprint(clothing[index]) for index in range(clothing.shape[0])],
                "foreground_mask": [object_fingerprint(foreground[index]) for index in range(foreground.shape[0])],
                "pose": [object_fingerprint(pose[index]) for index in range(pose.shape[0])],
                "w2c": [object_fingerprint(w2c[index]) for index in range(w2c.shape[0])],
            },
        }
    return {
        "status": "PASS" if not errors else "FAIL", "errors": errors, "episodes": rows,
        "distinct_rgb_files": len(set(rgb_hashes.values())),
        "distinct_clothing_masks": len(set(cloth_hashes.values())),
        "distinct_foreground_masks": len(set(foreground_hashes.values())),
        "different_outfit_rgb_hashes": all(
            rgb_hashes[f"O01/{condition}"] != rgb_hashes[f"O08/{condition}"] for condition in CONDITIONS
        ),
        "different_outfit_clothing_mask_hashes": all(
            cloth_hashes[f"O01/{condition}"] != cloth_hashes[f"O08/{condition}"] for condition in CONDITIONS
        ),
        "pose_camera_preprocessing_finite": not any("nonfinite" in value for value in errors),
        "each_episode_uses_distinct_reference_rgb": all(
            len(set(value["per_reference_fingerprints"]["rgb"])) == 3 for value in rows.values()
        ),
        "each_episode_uses_distinct_reference_pose_camera": all(
            len(set(zip(
                value["per_reference_fingerprints"]["pose"],
                value["per_reference_fingerprints"]["w2c"],
            ))) == 3 for value in rows.values()
        ),
        "permutation_set_integrity": all(
            len(set(value["references"])) == 3 and value["target"] not in value["references"]
            for value in rows.values()
        ),
        "protocol": protocol,
    }


def _subset_episode(episode: Mapping[str, Any], indices: Sequence[int]) -> dict[str, Any]:
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("reference subset must contain unique indices")
    count = int(episode["reference_images"].shape[0])
    if min(indices) < 0 or max(indices) >= count:
        raise IndexError("reference subset index is out of range")
    index = torch.tensor(indices, dtype=torch.long, device=episode["reference_images"].device)
    result = {}
    tensor_fields = (
        "reference_images", "reference_cloth_masks", "reference_foreground_masks",
        "reference_poses", "reference_R_global", "reference_Rh", "reference_Th",
        "reference_K", "reference_w2c", "reference_valid_mask",
    )
    for name in tensor_fields:
        if name in episode:
            result[name] = episode[name].index_select(0, index)
    result["reference_cameras"] = [episode["reference_cameras"][value] for value in indices]
    result["reference_condition_ids"] = [episode["reference_condition_ids"][value] for value in indices]
    return result


def _base_reference_images(context: Mapping[str, Any], outfit: str, episode: Mapping[str, Any]) -> torch.Tensor:
    return torch.stack([
        context["samples"][f"{outfit}/{condition}"]["target_base_rgb"]
        for condition in episode["reference_condition_ids"]
    ])


def reference_variants(context: Mapping[str, Any], outfit: str, condition: str) -> dict[str, dict[str, Any]]:
    key = f"{outfit}/{condition}"
    other = "O08" if outfit == "O01" else "O01"
    episode = context["episodes"][key]
    swapped = context["episodes"][f"{other}/{condition}"]
    zero = dict(episode); zero["reference_images"] = torch.zeros_like(episode["reference_images"])
    base = dict(episode); base["reference_images"] = _base_reference_images(context, outfit, episode)
    return {
        "correct": dict(episode),
        "swapped": dict(swapped),
        "permutation": _subset_episode(episode, [2, 0, 1]),
        "single_reference": _subset_episode(episode, [0]),
        "dropout_0": _subset_episode(episode, [1, 2]),
        "dropout_1": _subset_episode(episode, [0, 2]),
        "dropout_2": _subset_episode(episode, [0, 1]),
        "zero_rgb": zero,
        "base_rgb": base,
    }


def _variant_geometry(geometry: Mapping[str, Any], variant: str) -> dict[str, Any]:
    indices = {
        "permutation": [2, 0, 1],
        "single_reference": [0],
        "dropout_0": [1, 2],
        "dropout_1": [0, 2],
        "dropout_2": [0, 1],
    }.get(variant)
    if indices is None:
        return dict(geometry)
    result = dict(geometry)
    index = torch.tensor(indices, dtype=torch.long, device=geometry["surface_depth_maps"].device)
    for name in ("surface_depth_maps", "surface_alpha_maps"):
        result[name] = geometry[name].index_select(0, index)
    return result


def _new_forward_inputs(episode: Mapping[str, Any]) -> dict[str, Any]:
    values = {
        "reference_images": episode["reference_images"],
        "reference_clothing_masks": episode["reference_cloth_masks"],
        "reference_foreground_masks": episode["reference_foreground_masks"],
        "reference_poses": episode["reference_poses"],
        "reference_w2c": episode["reference_w2c"],
        "reference_valid_mask": episode["reference_valid_mask"],
    }
    leaked = FORBIDDEN_FORWARD_FIELDS.intersection(values)
    if leaked:
        raise RuntimeError(f"forbidden prediction input: {sorted(leaked)}")
    return values


def coefficient_forward(
    token_encoder: MaskAwareReferenceTokenEncoderV1,
    fusion: ReferenceSetCoefficientFusionV1,
    episode: Mapping[str, Any],
) -> tuple[torch.Tensor, ReferenceTokenOutput, ReferenceSetCoefficientOutput]:
    token_output = token_encoder(**_new_forward_inputs(episode))
    fusion_output = fusion(token_output.tokens, token_output.valid_mask)
    return fusion_output.coefficient, token_output, fusion_output


def _tensor_stats(value: torch.Tensor) -> dict[str, Any]:
    detached = value.detach().float()
    return {
        "shape": list(detached.shape), "finite": bool(torch.isfinite(detached).all()),
        "mean": float(detached.mean()), "std": float(detached.std(unbiased=False)),
        "l1_mean": float(detached.abs().mean()), "l2_norm": float(torch.linalg.vector_norm(detached)),
        "minimum": float(detached.min()), "maximum": float(detached.max()),
    }


def _distance(first: torch.Tensor, second: torch.Tensor) -> dict[str, float]:
    first = first.detach().float().reshape(-1)
    second = second.detach().float().reshape(-1)
    if first.numel() != second.numel():
        raise ValueError("feature-distance shapes differ")
    difference = first - second
    return {
        "l1": float(difference.abs().mean()),
        "l2": float(torch.linalg.vector_norm(difference) / math.sqrt(max(1, difference.numel()))),
        "cosine": float(F.cosine_similarity(first.reshape(1, -1), second.reshape(1, -1), eps=1e-12)),
    }


def _raw_probe_features(
    model: Any, episode: Mapping[str, Any], token_encoder: MaskAwareReferenceTokenEncoderV1,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        images = episode["reference_images"]
        feature_maps = model.clothing_observation_encoder.backbone[:-1](images)
        cloth = F.interpolate(episode["reference_cloth_masks"], feature_maps.shape[-2:], mode="area").clamp(0, 1)
        foreground = F.interpolate(episode["reference_foreground_masks"], feature_maps.shape[-2:], mode="area").clamp(0, 1)
        cloth_mean = token_encoder._weighted_mean(feature_maps, cloth)
        cloth_max = token_encoder._masked_max(feature_maps, cloth)
        foreground_mean = token_encoder._weighted_mean(feature_maps, foreground)
        f0 = feature_maps.mean(dim=(0, 2, 3))
        f1 = cloth_mean.mean(dim=0)
        f2 = torch.cat((cloth_mean.mean(dim=0), cloth_max.mean(dim=0)))
        f3 = (foreground_mean - cloth_mean).mean(dim=0)
    return {"F0": f0, "F1": f1, "F2": f2, "F3": f3}


def _legacy_features(context: Mapping[str, Any], key: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    outfit, condition = key.split("/")
    model, predictor = context["legacy_model"], context["legacy_predictor"]
    episode = context["episodes"][key]
    geometry = context["geometries"][condition]
    with torch.no_grad():
        coefficient, online, coefficient_output = legacy.stage_c_coefficients(
            model, predictor, episode, geometry
        )
    completion = online["completion"]
    confidence = completion.confidence
    pooled = (completion.completed_anchor_features * confidence).sum(dim=0) / confidence.sum().clamp_min(1e-8)
    features = {
        "F4": online["global_clothing_embedding"].reshape(-1),
        "F5": pooled.reshape(-1),
        "F6": coefficient_output.fused_feature.reshape(-1),
    }
    summary = {
        "coefficient": float(coefficient),
        "global_embedding": _tensor_stats(online["global_clothing_embedding"]),
        "completed_local": _tensor_stats(completion.completed_anchor_features),
        "fused_input": _tensor_stats(coefficient_output.fused_feature),
    }
    return features, summary


def _legacy_gradient_trace(context: Mapping[str, Any], key: str) -> dict[str, Any]:
    _, condition = key.split("/")
    episode = dict(context["episodes"][key])
    episode["reference_images"] = episode["reference_images"].detach().clone().requires_grad_(True)
    coefficient, online, coefficient_output = legacy.stage_c_coefficients(
        context["legacy_model"], context["legacy_predictor"], episode,
        context["geometries"][condition],
    )
    values = {
        "global_embedding": online["global_clothing_embedding"],
        "completed_local": online["completion"].completed_anchor_features,
        "fused_input": coefficient_output.fused_feature,
        "reference_rgb": episode["reference_images"],
    }
    gradients = torch.autograd.grad(
        coefficient.sum(), list(values.values()), allow_unused=True, retain_graph=False
    )
    result = {}
    for (name, value), gradient in zip(values.items(), gradients):
        result[name] = {
            "value": _tensor_stats(value),
            "gradient_present": gradient is not None,
            "gradient_finite": bool(gradient is not None and torch.isfinite(gradient).all()),
            "gradient_l2": 0.0 if gradient is None else float(torch.linalg.vector_norm(gradient)),
            "gradient_nonzero_count": 0 if gradient is None else int(torch.count_nonzero(gradient)),
        }
    return result


def _legacy_trace_values(
    context: Mapping[str, Any], episode: Mapping[str, Any], geometry: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    model = context["legacy_model"]
    inputs = o01._forward_inputs(episode, geometry)
    with torch.no_grad():
        encoded = model.clothing_observation_encoder(
            inputs["reference_images"], inputs["reference_cloth_masks"],
            inputs["reference_valid_mask"],
        )
        coefficient, online, coefficient_output = legacy.stage_c_coefficients(
            model, context["legacy_predictor"], episode, geometry
        )
    completion, projection, observed = online["completion"], online["projection"], online["observed"]
    cloth_feature_mask = encoded["feature_cloth_masks"]
    foreground_feature_mask = F.interpolate(
        episode["reference_foreground_masks"], encoded["feature_maps"].shape[-2:], mode="area"
    ).clamp(0, 1)
    cloth_pooled = MaskAwareReferenceTokenEncoderV1._weighted_mean(
        encoded["feature_maps"], cloth_feature_mask
    )
    foreground_pooled = MaskAwareReferenceTokenEncoderV1._weighted_mean(
        encoded["feature_maps"], foreground_feature_mask
    )
    return {
        "01_reference_rgb": episode["reference_images"],
        "02_foreground_and_clothing_masks": torch.cat(
            (episode["reference_foreground_masks"], episode["reference_cloth_masks"]), dim=1
        ),
        "03_frozen_backbone_spatial_feature": encoded["feature_maps"],
        "04_clothing_mask_pooled_feature": cloth_pooled,
        "05_foreground_mask_pooled_feature": foreground_pooled,
        "06_observation_adaptation_output": encoded["per_view_features"],
        "07_projected_sampled_feature": projection["sampled_features"],
        "08_aggregated_observed_feature": observed["observed_clothing_feature"],
        "09_completed_local_feature": completion.completed_anchor_features,
        "10_global_feature": online["global_clothing_embedding"],
        "11_fused_coefficient_input": coefficient_output.fused_feature,
        "12_scalar_coefficient": coefficient,
    }


def _legacy_twelve_layer_trace(context: Mapping[str, Any], key: str) -> dict[str, Any]:
    _, condition = key.split("/")
    layers = _legacy_trace_values(
        context, context["episodes"][key], context["geometries"][condition]
    )
    return {name: _tensor_stats(value) for name, value in layers.items()}


def _comparison_vector(value: torch.Tensor, maximum: int = 4096) -> torch.Tensor:
    flat = value.detach().float().reshape(-1)
    indices = torch.linspace(0, max(flat.numel() - 1, 0), maximum, device=flat.device).long()
    return flat.index_select(0, indices).cpu()


def _linear_probe(
    features: Mapping[str, torch.Tensor], labels: Mapping[str, float], ridge: float,
) -> dict[str, Any]:
    keys = [f"{outfit}/{condition}" for condition in CONDITIONS for outfit in OUTFITS]
    x = torch.stack([features[key].detach().double().cpu().reshape(-1) for key in keys])
    y = torch.tensor([labels[key] for key in keys], dtype=torch.float64)
    predictions = torch.empty_like(y)
    folds = {}
    for condition in CONDITIONS:
        test = torch.tensor([key.endswith(condition) for key in keys], dtype=torch.bool)
        train = ~test
        x_train, y_train = x[train], y[train]
        mean = x_train.mean(0, keepdim=True)
        scale = x_train.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
        normalized_train = (x_train - mean) / scale
        normalized_test = (x[test] - mean) / scale
        y_mean = y_train.mean()
        centered_y = y_train - y_mean
        kernel = normalized_train @ normalized_train.T
        alpha = torch.linalg.solve(
            kernel + ridge * torch.eye(kernel.shape[0], dtype=kernel.dtype), centered_y
        )
        fold_prediction = normalized_test @ normalized_train.T @ alpha + y_mean
        predictions[test] = fold_prediction
        folds[condition] = {
            "test_keys": [key for key, selected in zip(keys, test.tolist()) if selected],
            "prediction": fold_prediction.tolist(), "target": y[test].tolist(),
        }
    signs = torch.sign(predictions) == torch.sign(y)
    centroids = {
        outfit: x[torch.tensor([key.startswith(outfit) for key in keys])].mean(0)
        for outfit in OUTFITS
    }
    within = []
    for outfit in OUTFITS:
        subset = x[torch.tensor([key.startswith(outfit) for key in keys])]
        for first in range(subset.shape[0]):
            for second in range(first + 1, subset.shape[0]):
                within.append(float(torch.linalg.vector_norm(subset[first] - subset[second]) / math.sqrt(x.shape[1])))
    between = []
    o01_values = x[torch.tensor([key.startswith("O01") for key in keys])]
    o08_values = x[torch.tensor([key.startswith("O08") for key in keys])]
    for first in o01_values:
        for second in o08_values:
            between.append(float(torch.linalg.vector_norm(first - second) / math.sqrt(x.shape[1])))
    return {
        "leave_one_view_out_accuracy": float(signs.double().mean()),
        "sign_correct_count": int(signs.sum()),
        "coefficient_mae": float((predictions - y).abs().mean()),
        "predictions": {key: float(value) for key, value in zip(keys, predictions)},
        "folds": folds,
        "centroid_distance": float(torch.linalg.vector_norm(centroids["O08"] - centroids["O01"]) / math.sqrt(x.shape[1])),
        "within_outfit_distance_mean": float(np.mean(within)),
        "between_outfit_distance_mean": float(np.mean(between)),
        "between_within_ratio": float(np.mean(between) / max(np.mean(within), 1e-12)),
        "feature_dimension": int(x.shape[1]),
    }


def run_stage_0(context: Mapping[str, Any]) -> dict[str, Any]:
    output_dir, config = context["output_dir"], context["config"]
    started = time.time()
    spatial_backbone = context["legacy_model"].clothing_observation_encoder.backbone[:-1]
    token_encoder = MaskAwareReferenceTokenEncoderV1(
        spatial_backbone, int(config["legacy_reference_model"]["image_feature_dim"]),
        int(config["coefficient_fusion"]["token_dim"]),
        int(config["coefficient_fusion"]["token_hidden_dim"]),
    ).to(context["base"]._xyz)
    features_by_probe: dict[str, dict[str, torch.Tensor]] = {name: {} for name in config["stage_0"]["probes"]}
    legacy_rows, traces, gradients, layer_vectors = {}, {}, {}, {}
    labels = {}
    for condition in CONDITIONS:
        for outfit in OUTFITS:
            key = f"{outfit}/{condition}"; labels[key] = TEACHER[outfit]
            raw = _raw_probe_features(context["legacy_model"], context["episodes"][key], token_encoder)
            old, summary = _legacy_features(context, key)
            for name, value in {**raw, **old}.items():
                features_by_probe[name][key] = value
            legacy_rows[key] = summary
            layer_values = _legacy_trace_values(
                context, context["episodes"][key], context["geometries"][condition]
            )
            traces[key] = {name: _tensor_stats(value) for name, value in layer_values.items()}
            layer_vectors[key] = {
                name: _comparison_vector(value) for name, value in layer_values.items()
            }
            gradients[key] = _legacy_gradient_trace(context, key)

    probes = {
        name: _linear_probe(values, labels, float(config["stage_0"]["ridge_lambda"]))
        for name, values in features_by_probe.items()
    }
    same_target_distances = {}
    same_target_layer_distances = {}
    variant_rows = {}
    with torch.no_grad():
        for condition in CONDITIONS:
            key_a, key_b = f"O01/{condition}", f"O08/{condition}"
            same_target_distances[condition] = {
                name: _distance(features_by_probe[name][key_a], features_by_probe[name][key_b])
                for name in features_by_probe
            }
            same_target_layer_distances[condition] = {
                name: _distance(layer_vectors[key_a][name], layer_vectors[key_b][name])
                for name in layer_vectors[key_a]
            }
            for outfit in OUTFITS:
                key = f"{outfit}/{condition}"
                correct = legacy_rows[key]["coefficient"]
                row = {}
                for variant, episode in reference_variants(context, outfit, condition).items():
                    variant_geometry = _variant_geometry(context["geometries"][condition], variant)
                    variant_layers = _legacy_trace_values(context, episode, variant_geometry)
                    coefficient = variant_layers["12_scalar_coefficient"]
                    row[variant] = {
                        "coefficient": float(coefficient),
                        "change_from_correct": abs(float(coefficient) - correct),
                        "references": list(episode["reference_condition_ids"]),
                        "layer_change_from_correct": {
                            name: _distance(
                                layer_vectors[key][name], _comparison_vector(value)
                            )
                            for name, value in variant_layers.items()
                        },
                    }
                variant_rows[key] = row
    layer_within_between = {}
    keys_by_outfit = {
        outfit: [f"{outfit}/{condition}" for condition in CONDITIONS] for outfit in OUTFITS
    }
    for layer in next(iter(layer_vectors.values())):
        within, between = [], []
        for outfit in OUTFITS:
            keys = keys_by_outfit[outfit]
            for first in range(len(keys)):
                for second in range(first + 1, len(keys)):
                    within.append(_distance(layer_vectors[keys[first]][layer], layer_vectors[keys[second]][layer])["l2"])
        for first in keys_by_outfit["O01"]:
            for second in keys_by_outfit["O08"]:
                between.append(_distance(layer_vectors[first][layer], layer_vectors[second][layer])["l2"])
        layer_within_between[layer] = {
            "within_outfit_l2_mean": float(np.mean(within)),
            "between_outfit_l2_mean": float(np.mean(between)),
            "between_within_ratio": float(np.mean(between) / max(np.mean(within), 1e-12)),
        }
    input_failure = context["input_audit"]["status"] != "PASS"
    mask_aware_separable = max(
        probes["F1"]["leave_one_view_out_accuracy"], probes["F2"]["leave_one_view_out_accuracy"]
    ) == 1.0
    legacy_mid_separable = max(
        probes[name]["leave_one_view_out_accuracy"] for name in ("F4", "F5", "F6")
    ) == 1.0
    if input_failure or not mask_aware_separable:
        classification = "RF-INPUT"
    elif not legacy_mid_separable:
        classification = "RF-FUSION"
    else:
        classification = "RF-HEAD"
    report = {
        "status": "PASS" if not input_failure else "FAIL",
        "classification": classification,
        "input_or_mask_contract_failure": input_failure,
        "continue_to_direct_fusion": not input_failure,
        "legacy_checkpoint": str(context["paths"]["legacy_stage_c_checkpoint"]),
        "legacy_checkpoint_sha256": sha256(context["paths"]["legacy_stage_c_checkpoint"]),
        "legacy_rows": legacy_rows, "variant_rows": variant_rows,
        "same_target_o01_o08_distances": same_target_distances,
        "same_target_o01_o08_layer_distances": same_target_layer_distances,
        "layer_within_between_distances": layer_within_between,
        "gradient_to_coefficient": gradients, "twelve_layer_trace": traces,
        "linear_probes": probes, "runtime_seconds": time.time() - started,
    }
    atomic_json(output_dir / "stage_0/stage_0_diagnostics.json", report)
    _stage_0_visuals(output_dir, report)
    atomic_json(output_dir / "RUN_STATUS.json", {
        "task_id": TASK_ID, "status": "RUNNING" if report["continue_to_direct_fusion"] else "FAIL",
        "stage": "STAGE_0_COMPLETE", "optimizer_steps": 0,
        "classification": classification, "updated_at_unix": time.time(),
    })
    return report


def _stage_0_visuals(output_dir: Path, report: Mapping[str, Any]) -> None:
    import matplotlib.pyplot as plt

    names = list(report["linear_probes"])
    accuracy = [report["linear_probes"][name]["leave_one_view_out_accuracy"] for name in names]
    ratio = [report["linear_probes"][name]["between_within_ratio"] for name in names]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].bar(names, accuracy); axes[0].set_ylim(0, 1.05); axes[0].set_title("4-fold leave-one-view-out sign accuracy")
    axes[1].bar(names, ratio); axes[1].set_title("between / within outfit distance")
    figure.savefig(output_dir / "visual_acceptance/stage_0_probe_separability.png", dpi=150)
    plt.close(figure)


def _make_new_modules(context: Mapping[str, Any]) -> tuple[MaskAwareReferenceTokenEncoderV1, ReferenceSetCoefficientFusionV1]:
    built = build_reference_coefficient_fusion(
        context["config"],
        spatial_backbone=context["legacy_model"].clothing_observation_encoder.backbone[:-1],
        feature_dim=int(context["config"]["legacy_reference_model"]["image_feature_dim"]),
    )
    if built is None:
        raise RuntimeError("formal configuration unexpectedly selected legacy fusion")
    encoder, fusion = built
    return encoder.to(context["base"]._xyz), fusion.to(context["base"]._xyz)


def _trainable_groups(
    encoder: MaskAwareReferenceTokenEncoderV1, fusion: ReferenceSetCoefficientFusionV1,
) -> dict[str, list[torch.nn.Parameter]]:
    groups = {
        "token_adapter": [parameter for parameter in encoder.token_adapter.parameters() if parameter.requires_grad],
        "reference_shared_mlp": [parameter for parameter in fusion.reference_mlp.parameters() if parameter.requires_grad],
        "set_attention": [parameter for parameter in fusion.attention_score.parameters() if parameter.requires_grad],
        "coefficient_head": [parameter for parameter in fusion.coefficient_head.parameters() if parameter.requires_grad],
    }
    if any(not values for values in groups.values()):
        raise RuntimeError("a registered trainable group is empty")
    return groups


def _gradient_snapshot(groups: Mapping[str, Sequence[torch.nn.Parameter]]) -> dict[str, Any]:
    result = {}
    for name, parameters in groups.items():
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        result[name] = {
            "parameter_count": sum(parameter.numel() for parameter in parameters),
            "gradient_tensor_count": len(gradients),
            "gradient_finite": bool(gradients and all(torch.isfinite(value).all() for value in gradients)),
            "gradient_l2": 0.0 if not gradients else float(torch.sqrt(sum(value.detach().float().square().sum() for value in gradients))),
            "gradient_nonzero_count": sum(int(torch.count_nonzero(value)) for value in gradients),
        }
    return result


def _module_state(
    encoder: MaskAwareReferenceTokenEncoderV1, fusion: ReferenceSetCoefficientFusionV1,
) -> dict[str, Any]:
    return {
        "token_adapter": encoder.token_adapter.state_dict(),
        "reference_set_fusion": fusion.state_dict(),
    }


def _load_module_state(
    encoder: MaskAwareReferenceTokenEncoderV1, fusion: ReferenceSetCoefficientFusionV1,
    state: Mapping[str, Any],
) -> None:
    encoder.token_adapter.load_state_dict(state["token_adapter"], strict=True)
    fusion.load_state_dict(state["reference_set_fusion"], strict=True)


def _rng_state() -> dict[str, Any]:
    return legacy.rng_state()


def _save_checkpoint(
    path: Path, step: int, encoder: MaskAwareReferenceTokenEncoderV1,
    fusion: ReferenceSetCoefficientFusionV1, optimizer: torch.optim.Optimizer,
    *, condition_position: int, backbone_fingerprint: str, basis_sha256: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "schema_version": "canondressgs.reference_basis_fusion_checkpoint.v1",
        "global_step": int(step), "model": _module_state(encoder, fusion),
        "optimizer": optimizer.state_dict(), "rng": _rng_state(),
        "condition_position": int(condition_position),
        "backbone_fingerprint": backbone_fingerprint, "basis_sha256": basis_sha256,
        "target_view_used_in_prediction_forward": False,
    }, temporary)
    os.replace(temporary, path)


def _coefficient_metrics(
    context: Mapping[str, Any], encoder: MaskAwareReferenceTokenEncoderV1,
    fusion: ReferenceSetCoefficientFusionV1,
) -> dict[str, Any]:
    encoder.eval(); fusion.eval()
    rows = {}
    with torch.no_grad():
        for condition in CONDITIONS:
            for outfit in OUTFITS:
                key = f"{outfit}/{condition}"
                coefficient, tokens, fused = coefficient_forward(encoder, fusion, context["episodes"][key])
                rows[key] = {
                    "coefficient": float(coefficient), "target": TEACHER[outfit],
                    "absolute_error": abs(float(coefficient) - TEACHER[outfit]),
                    "correct_sign": bool(float(coefficient) * TEACHER[outfit] > 0),
                    "token_std": float(tokens.tokens.std(unbiased=False)),
                    "token_norm": float(torch.linalg.vector_norm(tokens.tokens)),
                    "set_feature_std": float(fused.set_feature.std(unbiased=False)),
                    "attention": fused.attention_weights.reshape(-1).tolist(),
                }
                variants = reference_variants(context, outfit, condition)
                variant_values = {}
                for name, episode in variants.items():
                    value, token_value, fusion_value = coefficient_forward(encoder, fusion, episode)
                    variant_values[name] = {
                        "coefficient": float(value),
                        "token_norm": float(torch.linalg.vector_norm(token_value.tokens)),
                        "set_feature_norm": float(torch.linalg.vector_norm(fusion_value.set_feature)),
                    }
                    if name == "swapped":
                        rows[key]["correct_swapped_token_distance"] = float(
                            torch.linalg.vector_norm(tokens.tokens - token_value.tokens)
                            / math.sqrt(tokens.tokens.numel())
                        )
                        rows[key]["correct_swapped_fusion_distance"] = float(
                            torch.linalg.vector_norm(fused.set_feature - fusion_value.set_feature)
                            / math.sqrt(fused.set_feature.numel())
                        )
                rows[key]["variants"] = variant_values
                rows[key]["zero_change"] = abs(
                    variant_values["zero_rgb"]["coefficient"] - float(coefficient)
                )
                rows[key]["base_change"] = abs(
                    variant_values["base_rgb"]["coefficient"] - float(coefficient)
                )
                rows[key]["permutation_difference"] = abs(
                    variant_values["permutation"]["coefficient"] - float(coefficient)
                )
                rows[key]["dropout_correct_sign_count"] = sum(
                    int(variant_values[name]["coefficient"] * TEACHER[outfit] > 0)
                    for name in ("single_reference", "dropout_0", "dropout_1", "dropout_2")
                )
    values = {outfit: [rows[f"{outfit}/{condition}"]["coefficient"] for condition in CONDITIONS] for outfit in OUTFITS}
    result = {
        "episodes": rows,
        "coefficient_mae": float(np.mean([value["absolute_error"] for value in rows.values()])),
        "correct_sign_count": sum(int(value["correct_sign"]) for value in rows.values()),
        "outfit_mean": {outfit: float(np.mean(items)) for outfit, items in values.items()},
        "outfit_std": {outfit: float(np.std(items)) for outfit, items in values.items()},
    }
    result["separation"] = result["outfit_mean"]["O08"] - result["outfit_mean"]["O01"]
    result.update({
        "sign_accuracy": result["correct_sign_count"] / 8.0,
        "paired_margin": result["separation"],
        "mean_correct_swapped_coefficient_change": float(np.mean([
            abs(value["variants"]["swapped"]["coefficient"] - value["coefficient"])
            for value in rows.values()
        ])),
        "mean_correct_swapped_token_distance": float(np.mean([
            value["correct_swapped_token_distance"] for value in rows.values()
        ])),
        "mean_correct_swapped_fusion_distance": float(np.mean([
            value["correct_swapped_fusion_distance"] for value in rows.values()
        ])),
        "mean_zero_change": float(np.mean([value["zero_change"] for value in rows.values()])),
        "mean_base_change": float(np.mean([value["base_change"] for value in rows.values()])),
        "permutation_max_abs_diff": float(max(value["permutation_difference"] for value in rows.values())),
        "single_dropout_sign_accuracy": float(np.mean([
            value["dropout_correct_sign_count"] / 4.0 for value in rows.values()
        ])),
    })
    encoder.train(); fusion.train()
    return result


def _loss_pair(
    c01: torch.Tensor, c08: torch.Tensor, config: Mapping[str, Any]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_config = config["training"]["loss"]
    targets = torch.tensor([-1.0, 1.0], device=c01.device, dtype=c01.dtype)
    values = torch.cat((c01.reshape(1), c08.reshape(1)))
    coefficient_loss = F.smooth_l1_loss(values, targets)
    sign_loss = F.relu(float(loss_config["sign_margin"]) - targets * values).mean()
    pair_loss = F.relu(float(loss_config["pair_margin"]) - torch.abs(c08 - c01)).mean()
    total = (
        float(loss_config["coefficient_weight"]) * coefficient_loss
        + float(loss_config["sign_weight"]) * sign_loss
        + float(loss_config["pair_weight"]) * pair_loss
    )
    return total, {
        "coefficient": coefficient_loss, "sign": sign_loss, "pair": pair_loss,
    }


def run_training(context: Mapping[str, Any]) -> dict[str, Any]:
    config, output_dir = context["config"], context["output_dir"]
    stage0_path = output_dir / "stage_0/stage_0_diagnostics.json"
    if not stage0_path.is_file():
        raise FileNotFoundError("Stage 0 diagnostics must exist before training")
    stage0 = json.loads(stage0_path.read_text(encoding="utf-8"))
    if not stage0["continue_to_direct_fusion"]:
        raise RuntimeError("real input/mask failure blocks direct-fusion training")
    seed = int(config["training"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    encoder, fusion = _make_new_modules(context)
    groups = _trainable_groups(encoder, fusion)
    optimizer = torch.optim.Adam(
        [{"name": name, "params": values} for name, values in groups.items()],
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    parameters = [parameter for values in groups.values() for parameter in values]
    initial_state = object_fingerprint(_module_state(encoder, fusion))
    milestones = {"0": _coefficient_metrics(context, encoder, fusion)}
    atomic_json(output_dir / "training/milestones/step_000000/metrics.json", milestones["0"])
    collapse_count = 0; collapse_warnings = []; final_gradients = {}; started = time.time()
    torch.cuda.reset_peak_memory_stats()
    final_checkpoint = None
    for step in range(1, int(config["training"]["max_steps"]) + 1):
        condition = CONDITIONS[(step - 1) % len(CONDITIONS)]
        optimizer.zero_grad(set_to_none=True)
        c01, _, _ = coefficient_forward(encoder, fusion, context["episodes"][f"O01/{condition}"])
        c08, _, _ = coefficient_forward(encoder, fusion, context["episodes"][f"O08/{condition}"])
        loss, parts = _loss_pair(c01, c08, config)
        if not torch.isfinite(loss):
            raise FloatingPointError("paired coefficient objective is NaN or Inf")
        loss.backward()
        final_gradients = _gradient_snapshot(groups)
        torch.nn.utils.clip_grad_norm_(
            parameters, float(config["training"]["gradient_clip_norm"]), error_if_nonfinite=True
        )
        optimizer.step()
        gap = abs(float(c08.detach() - c01.detach()))
        collapse_count = collapse_count + 1 if gap < float(config["training"]["collapse_warning_mean_gap"]) else 0
        if collapse_count and collapse_count % int(config["training"]["collapse_warning_steps"]) == 0:
            collapse_warnings.append({
                "warning": "MEAN_COEFFICIENT_COLLAPSE", "step": step,
                "consecutive_steps": collapse_count, "gap": gap,
            })
        append_jsonl(output_dir / "training/training.jsonl", {
            "step": step, "condition": condition, "c01": float(c01.detach()),
            "c08": float(c08.detach()), "gap": gap, "loss": float(loss.detach()),
            "parts": {name: float(value.detach()) for name, value in parts.items()},
        })
        if step in config["training"]["milestones"]:
            metrics = _coefficient_metrics(context, encoder, fusion)
            metrics["gradient_groups"] = final_gradients
            milestones[str(step)] = metrics
            milestone_dir = output_dir / f"training/milestones/step_{step:06d}"
            atomic_json(milestone_dir / "metrics.json", metrics)
            final_checkpoint = output_dir / f"training/checkpoints/checkpoint_step_{step:06d}.pth"
            _save_checkpoint(
                final_checkpoint, step, encoder, fusion, optimizer,
                condition_position=step % len(CONDITIONS),
                backbone_fingerprint=context["backbone_before"],
                basis_sha256=sha256(context["paths"]["explicit_basis_artifact"]),
            )
            atomic_json(output_dir / "RUN_STATUS.json", {
                "task_id": TASK_ID, "status": "RUNNING", "stage": "TRAINING",
                "optimizer_steps": step, "checkpoint": str(final_checkpoint),
                "checkpoint_sha256": sha256(final_checkpoint), "updated_at_unix": time.time(),
            })
    if final_checkpoint is None:
        raise RuntimeError("training did not emit its final checkpoint")
    final_metrics = milestones[str(config["training"]["max_steps"])]
    checkpoint_resume = _checkpoint_roundtrip(
        context, encoder, fusion, optimizer, final_checkpoint, final_metrics
    )
    result = {
        "status": "COMPLETE", "optimizer_steps": int(config["training"]["max_steps"]),
        "initial_state_fingerprint": initial_state,
        "final_state_fingerprint": object_fingerprint(_module_state(encoder, fusion)),
        "milestones": milestones, "final": final_metrics,
        "gradients": final_gradients, "collapse_warnings": collapse_warnings,
        "checkpoint": str(final_checkpoint), "checkpoint_sha256": sha256(final_checkpoint),
        "checkpoint_resume": checkpoint_resume,
        "runtime_seconds": time.time() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
        "paired_batch_every_step": True,
    }
    atomic_json(output_dir / "training/training_metrics.json", result)
    _training_visuals(output_dir, milestones)
    return result


def _checkpoint_roundtrip(
    context: Mapping[str, Any], encoder: MaskAwareReferenceTokenEncoderV1,
    fusion: ReferenceSetCoefficientFusionV1, optimizer: torch.optim.Optimizer,
    checkpoint_path: Path, final_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    restored_encoder, restored_fusion = _make_new_modules(context)
    restored_groups = _trainable_groups(restored_encoder, restored_fusion)
    restored_optimizer = torch.optim.Adam(
        [{"name": name, "params": values} for name, values in restored_groups.items()],
        lr=float(context["config"]["training"]["learning_rate"]),
        weight_decay=float(context["config"]["training"]["weight_decay"]),
    )
    _load_module_state(restored_encoder, restored_fusion, checkpoint["model"])
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    encoder.eval(); fusion.eval(); restored_encoder.eval(); restored_fusion.eval()
    fixed = context["episodes"]["O01/cond_000017"]
    with torch.no_grad():
        before = coefficient_forward(encoder, fusion, fixed)[0]
        after = coefficient_forward(restored_encoder, restored_fusion, fixed)[0]
    current_rng = _rng_state(); legacy.restore_rng(checkpoint["rng"])
    rng_exact = object_fingerprint(_rng_state()) == object_fingerprint(checkpoint["rng"])
    legacy.restore_rng(current_rng)
    result = {
        "global_step_exact": int(checkpoint["global_step"]) == int(context["config"]["training"]["max_steps"]),
        "model_state_exact": object_fingerprint(_module_state(encoder, fusion)) == object_fingerprint(_module_state(restored_encoder, restored_fusion)),
        "optimizer_state_exact": object_fingerprint(optimizer.state_dict()) == object_fingerprint(restored_optimizer.state_dict()),
        "rng_state_exact": rng_exact,
        "condition_position_exact": int(checkpoint["condition_position"]) == 0,
        "fixed_coefficient_bitwise_exact": torch.equal(before, after),
        "fixed_coefficient_max_abs_diff": float((before - after).abs().max()),
        "backbone_fingerprint_exact": checkpoint["backbone_fingerprint"] == context["backbone_before"],
        "basis_sha256_exact": checkpoint["basis_sha256"] == sha256(context["paths"]["explicit_basis_artifact"]),
        "final_metrics_finite": all(math.isfinite(float(value)) for value in (
            final_metrics["coefficient_mae"], final_metrics["separation"]
        )),
    }
    result["pass"] = all(value for name, value in result.items() if name != "fixed_coefficient_max_abs_diff")
    encoder.train(); fusion.train()
    return result


def _training_visuals(output_dir: Path, milestones: Mapping[str, Any]) -> None:
    import matplotlib.pyplot as plt

    steps = sorted(int(value) for value in milestones)
    o01_values = [milestones[str(step)]["outfit_mean"]["O01"] for step in steps]
    o08_values = [milestones[str(step)]["outfit_mean"]["O08"] for step in steps]
    mae = [milestones[str(step)]["coefficient_mae"] for step in steps]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].plot(steps, o01_values, "o-", label="O01"); axes[0].plot(steps, o08_values, "o-", label="O08")
    axes[0].axhline(-1, color="gray", linestyle="--"); axes[0].axhline(1, color="gray", linestyle="--")
    axes[0].set_title("Outfit coefficient means"); axes[0].legend()
    axes[1].plot(steps, mae, "o-"); axes[1].set_title("Coefficient MAE")
    figure.savefig(output_dir / "visual_acceptance/coefficient_learning_curve.png", dpi=150)
    plt.close(figure)


def _masked_mae(first: torch.Tensor, second: torch.Tensor, mask: torch.Tensor) -> float:
    return o01._masked_mae(first, second, mask)


def _residual_distribution(residual: GaussianClothingResiduals) -> dict[str, Any]:
    return {
        name: {
            "mean": float(value.detach().float().mean()),
            "std": float(value.detach().float().std(unbiased=False)),
            "mean_abs": float(value.detach().float().abs().mean()),
            "max_abs": float(value.detach().float().abs().max()),
            "nonzero_ratio": float((value.detach().abs() > 1e-9).float().mean()),
        }
        for name, value in residual.as_dict().items()
    }


def _variant_evaluation(
    context: Mapping[str, Any], encoder: MaskAwareReferenceTokenEncoderV1,
    fusion: ReferenceSetCoefficientFusionV1, outfit: str, condition: str,
    variant: str, episode: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    coefficient, tokens, fused = coefficient_forward(encoder, fusion, episode)
    target_coefficient = context["teacher_coefficients"][outfit]
    prediction = context["basis"](
        coefficient, chunk_size=int(context["config"]["basis"]["chunk_size"])
    )
    target = context["basis"](
        target_coefficient, chunk_size=int(context["config"]["basis"]["chunk_size"])
    )
    residual_metrics = parameterization.residual_metrics(
        prediction, target, context["basis_payload"]["channel_bounds"], active_epsilon=1e-8
    )
    sample = context["samples"][f"{outfit}/{condition}"]
    predicted_rgb, predicted_alpha = parameterization.render_prediction(
        context["base"], sample, prediction, context["background"]
    )
    oracle_rgb, oracle_alpha = parameterization.render_prediction(
        context["base"], sample, target, context["background"]
    )
    garment = diagnosis._garment_mask(sample)
    protected = sample["target_protected_mask"]
    background = 1.0 - sample["target_foreground_mask"]
    row = {
        "variant": variant, "coefficient": float(coefficient),
        "target_coefficient": float(target_coefficient),
        "coefficient_error": abs(float(coefficient) - float(target_coefficient)),
        "correct_sign": bool(float(coefficient) * TEACHER[outfit] > 0),
        "references": list(episode["reference_condition_ids"]),
        "reference_count": int(episode["reference_images"].shape[0]),
        "token_norm": float(torch.linalg.vector_norm(tokens.tokens)),
        "token_std": float(tokens.tokens.std(unbiased=False)),
        "set_feature_norm": float(torch.linalg.vector_norm(fused.set_feature)),
        "attention_weights": fused.attention_weights.reshape(-1).tolist(),
        "residual_normalized_rmse": residual_metrics["normalized_rmse"],
        "residual_direction_cosine": residual_metrics["direction_cosine"],
        "residual_top_10pct_overlap": residual_metrics["top_10pct_overlap"],
        "oracle_garment_rgb_mae": _masked_mae(predicted_rgb, oracle_rgb, garment),
        "oracle_alpha_mae": float((predicted_alpha - oracle_alpha).abs().mean()),
        "protected_rgb_mae_from_base": _masked_mae(predicted_rgb, sample["target_base_rgb"], protected),
        "background_rgb_mae_from_base": _masked_mae(predicted_rgb, sample["target_base_rgb"], background),
        "residual_distribution": _residual_distribution(prediction),
        "target_pose_camera_fixed": True,
        "target_view_used_in_prediction_forward": False,
    }
    cache = {
        "predicted_rgb": predicted_rgb.detach().cpu(), "predicted_alpha": predicted_alpha.detach().cpu(),
        "oracle_rgb": oracle_rgb.detach().cpu(), "oracle_alpha": oracle_alpha.detach().cpu(),
    }
    return row, cache


def run_evaluation(context: Mapping[str, Any]) -> dict[str, Any]:
    config, output_dir = context["config"], context["output_dir"]
    training_path = output_dir / "training/training_metrics.json"
    if not training_path.is_file():
        raise FileNotFoundError("training metrics are required before formal evaluation")
    training_result = json.loads(training_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(training_result["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder, fusion = _make_new_modules(context)
    _load_module_state(encoder, fusion, checkpoint["model"])
    encoder.eval(); fusion.eval()
    rows, visual_cache = {}, {}
    started = time.time(); torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for condition in CONDITIONS:
            for outfit in OUTFITS:
                key = f"{outfit}/{condition}"
                rows[key] = {}
                for variant, episode in reference_variants(context, outfit, condition).items():
                    row, cache = _variant_evaluation(
                        context, encoder, fusion, outfit, condition, variant, episode
                    )
                    rows[key][variant] = row
                    if variant in {"correct", "swapped"} or (
                        condition == "cond_000318" and variant in {
                            "single_reference", "dropout_0", "dropout_1", "dropout_2",
                            "zero_rgb", "base_rgb",
                        }
                    ):
                        visual_cache[f"{key}/{variant}"] = cache
    correct_rows = [rows[f"{outfit}/{condition}"]["correct"] for condition in CONDITIONS for outfit in OUTFITS]
    coefficient_mae = float(np.mean([value["coefficient_error"] for value in correct_rows]))
    outfit_mean = {
        outfit: float(np.mean([rows[f"{outfit}/{condition}"]["correct"]["coefficient"] for condition in CONDITIONS]))
        for outfit in OUTFITS
    }
    coefficient_margins, residual_margins, render_margins = [], [], []
    wins = 0; permutation_differences = []; replacement_changes = []
    single_dropout_signs = []
    for condition in CONDITIONS:
        for outfit in OUTFITS:
            values = rows[f"{outfit}/{condition}"]
            correct, swapped = values["correct"], values["swapped"]
            c_margin = swapped["coefficient_error"] - correct["coefficient_error"]
            r_margin = swapped["residual_normalized_rmse"] - correct["residual_normalized_rmse"]
            render_margin = swapped["oracle_garment_rgb_mae"] - correct["oracle_garment_rgb_mae"]
            coefficient_margins.append(c_margin); residual_margins.append(r_margin); render_margins.append(render_margin)
            if c_margin > 0 and r_margin > 0 and render_margin > 0:
                wins += 1
            permutation_differences.append(abs(values["permutation"]["coefficient"] - correct["coefficient"]))
            replacement_changes.append(max(
                abs(values["zero_rgb"]["coefficient"] - correct["coefficient"]),
                abs(values["base_rgb"]["coefficient"] - correct["coefficient"]),
            ))
            single_dropout_signs.extend(
                values[name]["correct_sign"]
                for name in ("single_reference", "dropout_0", "dropout_1", "dropout_2")
            )
    aggregate = {
        "variant_count": sum(len(value) for value in rows.values()),
        "coefficient_mae": coefficient_mae,
        "outfit_mean": outfit_mean,
        "outfit_std": {
            outfit: float(np.std([rows[f"{outfit}/{condition}"]["correct"]["coefficient"] for condition in CONDITIONS]))
            for outfit in OUTFITS
        },
        "separation": outfit_mean["O08"] - outfit_mean["O01"],
        "correct_sign_count": sum(int(value["correct_sign"]) for value in correct_rows),
        "mean_swapped_coefficient_margin": float(np.mean(coefficient_margins)),
        "minimum_swapped_coefficient_margin": float(np.min(coefficient_margins)),
        "mean_swapped_residual_margin": float(np.mean(residual_margins)),
        "minimum_swapped_residual_margin": float(np.min(residual_margins)),
        "mean_swapped_render_margin": float(np.mean(render_margins)),
        "minimum_swapped_render_margin": float(np.min(render_margins)),
        "correct_episode_wins": wins,
        "permutation_max_abs_diff": float(np.max(permutation_differences)),
        "reference_replacement_change_mean": float(np.mean(replacement_changes)),
        "reference_replacement_change_minimum": float(np.min(replacement_changes)),
        "single_dropout_correct_sign_count": sum(int(value) for value in single_dropout_signs),
        "single_dropout_case_count": len(single_dropout_signs),
        "mean_correct_garment_rgb_mae": float(np.mean([value["oracle_garment_rgb_mae"] for value in correct_rows])),
        "mean_correct_protected_rgb_mae": float(np.mean([value["protected_rgb_mae_from_base"] for value in correct_rows])),
        "mean_correct_background_rgb_mae": float(np.mean([value["background_rgb_mae_from_base"] for value in correct_rows])),
    }
    acceptance = config["acceptance"]
    checks = {
        "formal_72_variants": aggregate["variant_count"] == 72,
        "coefficient_mae": aggregate["coefficient_mae"] <= float(acceptance["coefficient_mae_max"]),
        "o01_mean": outfit_mean["O01"] <= float(acceptance["o01_mean_max"]),
        "o08_mean": outfit_mean["O08"] >= float(acceptance["o08_mean_min"]),
        "separation": aggregate["separation"] >= float(acceptance["separation_min"]),
        "correct_sign_8_of_8": aggregate["correct_sign_count"] >= int(acceptance["correct_sign_count_min"]),
        "swapped_coefficient_margin_each": aggregate["minimum_swapped_coefficient_margin"] >= float(acceptance["swapped_coefficient_margin_min"]),
        "swapped_residual_margin_each": aggregate["minimum_swapped_residual_margin"] > 0,
        "swapped_render_margin_each": aggregate["minimum_swapped_render_margin"] > 0,
        "correct_episode_wins": wins >= int(acceptance["correct_episode_wins_min"]),
        "zero_or_base_change_each": aggregate["reference_replacement_change_minimum"] >= float(acceptance["reference_replacement_change_min"]),
        "permutation_invariant": aggregate["permutation_max_abs_diff"] <= float(acceptance["permutation_max_abs_diff_max"]),
        "single_and_dropout_sign": aggregate["single_dropout_correct_sign_count"] == aggregate["single_dropout_case_count"],
        "target_forward_leakage_zero": all(
            not value[variant]["target_view_used_in_prediction_forward"]
            for value in rows.values() for variant in value
        ),
        "checkpoint_resume": bool(training_result["checkpoint_resume"]["pass"]),
        "all_trainable_groups_gradient_nonzero_finite": all(
            value["gradient_finite"] and value["gradient_l2"] > 0
            for value in training_result["gradients"].values()
        ),
        "base_bitwise_frozen": context["base_before"] == _tensor_state_fingerprint(_base_named_tensors(context["base"])),
        "base_gradient_zero": _base_gradient_count(context["base"]) == 0,
        "image_backbone_bitwise_frozen": context["backbone_before"] == o01._state_fingerprint(
            context["legacy_model"].clothing_observation_encoder.backbone.state_dict()
        ),
        "image_backbone_gradient_zero": sum(
            parameter.grad is not None for parameter in encoder.spatial_backbone.parameters()
        ) == 0,
        "legacy_mmlp_gradient_zero": sum(
            parameter.grad is not None
            for parameter in context["legacy_model"].dressable_model.anchor_clothing_mlp.parameters()
        ) == 0,
        "basis_frozen": sum(parameter.numel() for parameter in context["basis"].parameters()) == 0,
        "explicit_basis_attempt_immutable": legacy.immutable_tree_metadata_fingerprint(
            context["explicit_attempt"]
        ) == context["explicit_tree_before"],
    }
    numeric_pass = all(checks.values())
    result = {
        "status": "NUMERIC_PASS_VISUAL_PENDING" if numeric_pass else "FAIL",
        "rows": rows, "aggregate": aggregate, "checks": checks,
        "runtime_seconds": time.time() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
        "visual_status": "PENDING_ACTUAL_INSPECTION" if numeric_pass else "NOT_ELIGIBLE_NUMERIC_FAIL",
    }
    atomic_json(output_dir / "evaluation/formal_72_variant_metrics.json", result)
    _evaluation_visuals(context, result, visual_cache)
    atomic_json(output_dir / "RUN_STATUS.json", {
        "task_id": TASK_ID, "status": result["status"], "stage": "EVALUATION_COMPLETE",
        "optimizer_steps": int(config["training"]["max_steps"]),
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
        "updated_at_unix": time.time(),
    })
    return result


def _evaluation_visuals(
    context: Mapping[str, Any], report: Mapping[str, Any], visual_cache: Mapping[str, Any],
) -> None:
    import matplotlib.pyplot as plt

    output_dir = context["output_dir"]
    rows = report["rows"]
    # Coefficient distribution and correct/swapped exchange.
    labels = [f"{outfit}-{VIEWS[condition]}" for condition in CONDITIONS for outfit in OUTFITS]
    correct = [rows[f"{outfit}/{condition}"]["correct"]["coefficient"] for condition in CONDITIONS for outfit in OUTFITS]
    swapped = [rows[f"{outfit}/{condition}"]["swapped"]["coefficient"] for condition in CONDITIONS for outfit in OUTFITS]
    figure, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    x = np.arange(len(labels)); axes[0].bar(x, correct); axes[0].set_xticks(x, labels, rotation=30, ha="right")
    axes[0].axhline(-1, color="gray", linestyle="--"); axes[0].axhline(1, color="gray", linestyle="--")
    axes[0].set_title("Correct-reference coefficients")
    axes[1].plot(x, correct, "o-", label="correct"); axes[1].plot(x, swapped, "x--", label="swapped")
    axes[1].set_xticks(x, labels, rotation=30, ha="right"); axes[1].legend(); axes[1].set_title("Correct/swapped exchange")
    figure.savefig(output_dir / "visual_acceptance/coefficient_distribution_and_swap.png", dpi=150); plt.close(figure)

    # Correct/swapped rendered contact sheets.
    for outfit in OUTFITS:
        contact_rows = []
        for condition in CONDITIONS:
            key = f"{outfit}/{condition}"; sample = context["samples"][key]
            correct_cache = visual_cache[f"{key}/correct"]; swapped_cache = visual_cache[f"{key}/swapped"]
            contact_rows.append((f"{outfit}/{VIEWS[condition]}", [
                ("base", sample["target_base_rgb"], 3), ("target", sample["target_edit_rgb"], 3),
                ("Oracle basis", correct_cache["oracle_rgb"], 3),
                ("correct reference", correct_cache["predicted_rgb"], 3),
                ("swapped reference", swapped_cache["predicted_rgb"], 3),
                ("correct error", (correct_cache["predicted_rgb"] - correct_cache["oracle_rgb"]).abs(), 3),
            ]))
        o01._save_contact_sheet(
            output_dir / f"visual_acceptance/{outfit}_correct_swapped_render_contact_sheet.png",
            contact_rows,
        )

    # Single/dropout and zero/base counterfactuals on the fixed back target.
    for family, names in (
        ("single_dropout", ("correct", "single_reference", "dropout_0", "dropout_1", "dropout_2")),
        ("zero_base", ("correct", "zero_rgb", "base_rgb")),
    ):
        contact_rows = []
        for outfit in OUTFITS:
            key = f"{outfit}/cond_000318"
            contact_rows.append((f"{outfit}/back", [
                (name, visual_cache[f"{key}/{name}"]["predicted_rgb"], 3) for name in names
            ]))
        o01._save_contact_sheet(
            output_dir / f"visual_acceptance/{family}_render_contact_sheet.png", contact_rows
        )

    # Numeric coefficient robustness plots.
    back_rows = {outfit: rows[f"{outfit}/cond_000318"] for outfit in OUTFITS}
    names = ["correct", "single_reference", "dropout_0", "dropout_1", "dropout_2", "zero_rgb", "base_rgb"]
    figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for outfit in OUTFITS:
        axis.plot(names, [back_rows[outfit][name]["coefficient"] for name in names], "o-", label=outfit)
    axis.tick_params(axis="x", rotation=25); axis.legend(); axis.set_title("Back-view reference variants")
    figure.savefig(output_dir / "visual_acceptance/reference_variant_coefficients.png", dpi=150); plt.close(figure)

    # Basis field and predicted residual magnitude.
    xyz = context["base"]._xyz.detach().float().cpu()
    sampled = torch.arange(0, xyz.shape[0], max(1, xyz.shape[0] // 80000))
    mean, components = context["basis"].normalized_fields()
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    fields = {
        "basis mean": torch.stack([value.reshape(value.shape[0], -1).square().mean(1) for value in mean.values()]).mean(0).sqrt(),
        "difference basis": torch.stack([value[0].reshape(value.shape[1], -1).square().mean(1) for value in components.values()]).mean(0).sqrt(),
    }
    coefficient = report["aggregate"]["outfit_mean"]["O08"]
    residual = context["basis"](torch.tensor([coefficient], device=context["base"]._xyz.device))
    fields["O08 predicted residual"] = parameterization.row_magnitude(
        residual, context["basis_payload"]["channel_bounds"]
    ).detach().cpu()
    for axis, (name, magnitude) in zip(axes, fields.items()):
        plot = axis.scatter(xyz[sampled, 0], xyz[sampled, 2], c=magnitude[sampled], s=1, cmap="magma")
        axis.set_title(name); axis.set_aspect("equal"); figure.colorbar(plot, ax=axis, fraction=0.046)
    figure.savefig(output_dir / "visual_acceptance/basis_and_predicted_residual_magnitude.png", dpi=140); plt.close(figure)

    # Token/statistical overview from the formal rows.
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].bar(labels, [rows[f"{outfit}/{condition}"]["correct"]["token_norm"] for condition in CONDITIONS for outfit in OUTFITS])
    axes[0].tick_params(axis="x", rotation=30); axes[0].set_title("Reference token norm")
    axes[1].bar(labels, [rows[f"{outfit}/{condition}"]["correct"]["set_feature_norm"] for condition in CONDITIONS for outfit in OUTFITS])
    axes[1].tick_params(axis="x", rotation=30); axes[1].set_title("Fused set-feature norm")
    figure.savefig(output_dir / "visual_acceptance/reference_token_and_set_statistics.png", dpi=150); plt.close(figure)


def finalize(context: Mapping[str, Any], visual_path: Path) -> dict[str, Any]:
    output_dir = context["output_dir"]
    stage0 = json.loads((output_dir / "stage_0/stage_0_diagnostics.json").read_text(encoding="utf-8"))
    training_result = json.loads((output_dir / "training/training_metrics.json").read_text(encoding="utf-8"))
    evaluation = json.loads((output_dir / "evaluation/formal_72_variant_metrics.json").read_text(encoding="utf-8"))
    visual = json.loads(visual_path.read_text(encoding="utf-8"))
    required_visual_fields = {"images_actually_opened", "inspection_method", "observations", "visual_acceptance_status"}
    if not required_visual_fields.issubset(visual):
        raise ValueError("visual observations are incomplete")
    if not visual["images_actually_opened"]:
        raise ValueError("visual acceptance cannot be adjudicated without opening images")
    visual_pass = visual["visual_acceptance_status"] == "PASS"
    numeric_pass = evaluation["status"] == "NUMERIC_PASS_VISUAL_PENDING"
    status = "PASS" if numeric_pass and visual_pass else "FAIL"
    if stage0["classification"] == "RF-INPUT" and stage0["input_or_mask_contract_failure"]:
        case, next_task = "RF-I", "FIX_REFERENCE_DATA_AND_MASK_PREPROCESSING"
    elif status == "PASS":
        case, next_task = "RF-P", "EXPAND_EXPLICIT_RESIDUAL_BASIS_TO_MULTI_OUTFIT"
    else:
        case, next_task = "RF-F", "CALIBRATE_REFERENCE_FEATURE_BACKBONE_OR_SUPERVISION"
    freeze = {
        "base_bitwise_unchanged": context["base_before"] == _tensor_state_fingerprint(_base_named_tensors(context["base"])),
        "base_gradient_count": _base_gradient_count(context["base"]),
        "image_backbone_bitwise_unchanged": context["backbone_before"] == o01._state_fingerprint(
            context["legacy_model"].clothing_observation_encoder.backbone.state_dict()
        ),
        "image_backbone_gradient_count": sum(
            parameter.grad is not None
            for parameter in context["legacy_model"].clothing_observation_encoder.backbone.parameters()
        ),
        "legacy_mmlp_gradient_count": sum(
            parameter.grad is not None
            for parameter in context["legacy_model"].dressable_model.anchor_clothing_mlp.parameters()
        ),
        "explicit_basis_attempt_immutable": legacy.immutable_tree_metadata_fingerprint(
            context["explicit_attempt"]
        ) == context["explicit_tree_before"],
        "basis_sha256": sha256(context["paths"]["explicit_basis_artifact"]),
    }
    result = {
        "task_id": TASK_ID, "status": status, "case": case,
        "stage_0_classification": stage0["classification"], "next_task": next_task,
        "allow_multi_outfit_expansion": status == "PASS",
        "optimizer_steps": training_result["optimizer_steps"],
        "checkpoint": training_result["checkpoint"],
        "checkpoint_sha256": training_result["checkpoint_sha256"],
        "aggregate": evaluation["aggregate"], "numeric_checks": evaluation["checks"],
        "checkpoint_resume": training_result["checkpoint_resume"],
        "freeze_evidence": freeze, "visual_acceptance": visual,
        "target_view_used_in_prediction_forward": False,
        "legacy_explicit_basis_attempt_modified": False,
    }
    atomic_json(output_dir / "visual_acceptance/visual_acceptance.json", visual)
    atomic_json(output_dir / "final_adjudication/final_adjudication.json", result)
    lines = [
        f"# {TASK_ID} Final Adjudication", "", f"- Final status: **{status}**",
        f"- Decision case: **{case}**", f"- Stage-0 classification: **{stage0['classification']}**",
        f"- Optimizer steps: {training_result['optimizer_steps']}",
        f"- Coefficient MAE: {evaluation['aggregate']['coefficient_mae']:.8f}",
        f"- O01/O08 means: {evaluation['aggregate']['outfit_mean']['O01']:.8f} / {evaluation['aggregate']['outfit_mean']['O08']:.8f}",
        f"- Separation: {evaluation['aggregate']['separation']:.8f}",
        f"- Correct/swapped wins: {evaluation['aggregate']['correct_episode_wins']}/8",
        f"- Visual acceptance: {visual['visual_acceptance_status']}",
        f"- Checkpoint resume: {'PASS' if training_result['checkpoint_resume']['pass'] else 'FAIL'}",
        f"- Frozen base/backbone/basis: {'PASS' if all((freeze['base_bitwise_unchanged'], freeze['image_backbone_bitwise_unchanged'], freeze['explicit_basis_attempt_immutable'])) else 'FAIL'}",
        f"- Next task: `{next_task}`", "",
        "Teacher coefficients were used only as supervision/evaluation targets. The prediction forward did not receive target views, outfit IDs, teacher coefficients, or diagnostic latents.",
    ]
    atomic_text(output_dir / "GATE_ACCEPTANCE.md", "\n".join(lines))
    atomic_json(output_dir / "RUN_STATUS.json", {
        "task_id": TASK_ID, "status": status, "stage": "FINALIZED",
        "optimizer_steps": training_result["optimizer_steps"],
        "checkpoint": training_result["checkpoint"],
        "checkpoint_sha256": training_result["checkpoint_sha256"],
        "case": case, "next_task": next_task, "updated_at_unix": time.time(),
    })
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reference-to-basis coefficient fusion closure")
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs/research/subject02_reference_basis_coefficient_fusion_v1.yaml",
    )
    parser.add_argument("--phase", choices=("stage0", "train", "evaluate", "all", "finalize"), default="all")
    parser.add_argument("--visual-observations", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args(); config = load_config(args.config)
    output_dir = Path(config["output"]["task_root"]) / config["output"]["attempt"]
    create = args.phase in {"stage0", "all"}
    context = None
    try:
        context = build_context(config, output_dir, create=create)
        if args.phase in {"stage0", "all"}:
            stage0 = run_stage_0(context)
            if not stage0["continue_to_direct_fusion"]:
                raise RuntimeError("Stage 0 found a real reference input/mask contract failure")
        if args.phase in {"train", "all"}:
            run_training(context)
        if args.phase in {"evaluate", "all"}:
            run_evaluation(context)
        if args.phase == "finalize":
            if args.visual_observations is None:
                raise ValueError("--visual-observations is required for finalize")
            finalize(context, args.visual_observations)
    except Exception as error:
        if output_dir.exists():
            atomic_json(output_dir / "failure_diagnostics.json", {
                "task_id": TASK_ID, "phase": args.phase, "exception_type": type(error).__name__,
                "exception": str(error), "traceback": traceback.format_exc(),
                "updated_at_unix": time.time(),
            })
            current_steps = 0
            status_path = output_dir / "RUN_STATUS.json"
            if status_path.is_file():
                try:
                    current_steps = int(json.loads(status_path.read_text(encoding="utf-8")).get("optimizer_steps", 0))
                except Exception:
                    current_steps = 0
            atomic_json(status_path, {
                "task_id": TASK_ID, "status": "FAILED_ATTEMPT", "failure_phase": args.phase,
                "optimizer_steps": current_steps, "exception": str(error),
                "updated_at_unix": time.time(),
            })
        raise


if __name__ == "__main__":
    main()
