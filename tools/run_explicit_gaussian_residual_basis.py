from __future__ import annotations

import argparse
import hashlib
import json
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
    DiagnosticCoefficientPredictor,
    ExplicitGaussianResidualBasis,
    ReferenceBasisCoefficientPredictor,
    build_centered_difference_basis,
    normalized_residual_dict,
)
from scene.gaussian_clothing_residuals import (  # noqa: E402
    CHANNELS,
    GaussianClothingResiduals,
    apply_protected_full_residual_guard,
)
from scene.image_conditioned_failure_diagnostics import (  # noqa: E402
    assert_forward_boundary,
    reference_variant,
)
from tools import diagnose_image_conditioned_overfit_failure as diagnosis  # noqa: E402
from tools import run_image_conditioned_overfit_o01 as o01  # noqa: E402
from tools import run_residual_decoder_capacity_v7 as v7  # noqa: E402
from tools import run_residual_field_parameterization as parameterization  # noqa: E402
from tools.run_module4b_canonical_oracle_micropilot import (  # noqa: E402
    _base_gradient_count,
    _base_named_tensors,
    _tensor_state_fingerprint,
)
from utils.mmlphuman_anchor_deformation import MMLPHumanAnchorDeformationAdapter  # noqa: E402


SCHEMA = "canondressgs.explicit_gaussian_residual_basis.v1"
TASK_ID = "SUBJECT02-EXPLICIT-GAUSSIAN-RESIDUAL-BASIS-001"
EXPECTED_BRANCH = "research/explicit-gaussian-residual-basis-20260720"
EXPECTED_SOURCE_HEAD = "79c36ecba73ddf6525d38a0fb70d8d2816af1dd8"
OUTFITS = ("O01", "O08")
CONDITIONS = o01.CONDITIONS
VIEWS = o01.VIEWS


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def immutable_tree_metadata_fingerprint(path: Path) -> str:
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        stat = item.stat()
        digest.update(str(item.relative_to(path)).replace("\\", "/").encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def git_output(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=PROJECT_ROOT, text=True).strip()


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA or config.get("task_id") != TASK_ID:
        raise ValueError("explicit-basis schema/task mismatch")
    if config.get("branch") != EXPECTED_BRANCH or config.get("source_head") != EXPECTED_SOURCE_HEAD:
        raise ValueError("explicit-basis governance contract changed")
    if tuple(config.get("outfits", ())) != OUTFITS or tuple(config.get("conditions", ())) != CONDITIONS:
        raise ValueError("two-outfit/four-view protocol changed")
    if int(config["basis"]["rank"]) != 1 or config["basis"]["method"] != "ordered_two_outfit_centered_half_difference":
        raise ValueError("current controlled basis must be rank-1 centered difference")
    if int(config["stage_b"]["max_steps"]) > 200 or int(config["stage_c"]["max_steps"]) > 500:
        raise ValueError("explicit-basis step ceiling changed")
    if int(config["stage_c"]["max_steps"]) % 8:
        raise ValueError("Stage C must contain complete balanced eight-episode cycles")
    frozen = {
        "stage_a": {"mean_bound_normalized_rmse_max": 0.002, "mean_direction_cosine_min": 0.995,
                    "mean_top_10pct_overlap_min": 0.98, "per_outfit_render_garment_mae_max": 0.005},
        "stage_b": {"coefficient_mae_max": 0.01, "reconstructed_residual_rmse_max": 0.01,
                    "direction_cosine_min": 0.98},
        "stage_c": {"coefficient_mae_max": 0.10, "reconstructed_residual_rmse_max": 0.08,
                    "direction_cosine_min": 0.80, "correct_episode_wins_min": 7},
    }
    for stage, values in frozen.items():
        for name, expected in values.items():
            if float(config[stage]["acceptance"][name]) != expected:
                raise ValueError(f"{stage} acceptance threshold changed: {name}")
    if any(bool(value) for value in config["permissions"].values()):
        raise ValueError("explicit-basis contract enabled a forbidden permission")
    return config


def bounds(config: Mapping[str, Any]) -> dict[str, float]:
    return parameterization.bounds(config)


def model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    model = config["model"]
    return {
        "base": {"require_real_base": True},
        "model": {
            "offset_mode": "anchor_film", "hidden_dim": int(model["hidden_dim"]),
            "num_layers": int(model["num_layers"]), "hyper_hidden_dim": int(model["hyper_hidden_dim"]),
            "anchor_num_frequencies": int(model["anchor_num_frequencies"]),
            "anchor_graph_k": int(model["anchor_graph_k"]),
            "enable_delta_xyz": True, "enable_delta_scaling": True, "enable_delta_opacity": True,
            "decoder": {"type": "legacy"}, "dressable_channels": dict(model["dressable_channels"]),
        },
        "image_conditioning": {
            "embedding_dim": int(model["embedding_dim"]), "feature_dim": int(model["image_feature_dim"]),
            "local_feature_dim": int(model["local_feature_dim"]),
            "freeze_backbone": bool(model["freeze_image_backbone"]),
        },
        "optimizer": dict(config["stage_c"]["optimizer"]),
    }


def construct_reference_model(base: Any, config: Mapping[str, Any], device: torch.device):
    model, _, _, _ = training.create_image_conditioned_components(
        o01.DatasetModelContract(), model_config(config), base_model=base, device=device,
    )
    graph_config = config["model"]["online_completion"]
    graph_indices, graph_weights, graph_diagnostics = o01.build_surface_aware_anchor_graph(
        base.xyz_vt.to(device), int(graph_config["graph_k"]), base.nbr_vt.to(device),
    )
    model.initialize_online_completion(
        graph_indices, graph_weights, hidden_dim=int(graph_config["hidden_dim"]),
        num_blocks=int(graph_config["blocks"]),
    )
    if model.residual_decoder_type != "legacy" or model.support_conditioned_residual_decoder_v7 is not None:
        raise RuntimeError("basis mode must not instantiate the per-Gaussian V7 residual decoder")
    return model, graph_diagnostics


def set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: Mapping[str, Any]) -> None:
    """Restore the RNG schema emitted by :func:`rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: Path, *, stage: str, step: int, model: Mapping[str, Any], optimizer: Any,
    scheduler: Any | None, extra: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "stage": stage, "global_step": int(step), "model": dict(model),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng": rng_state(), "extra": dict(extra),
    }, temporary)
    temporary.replace(path)


def _basis_artifact_path(output_dir: Path) -> Path:
    return output_dir / "stage_a" / "basis" / "explicit_basis.pt"


def save_basis_artifact(
    output_dir: Path, decomposition: Any, channel_bounds: Mapping[str, float],
) -> dict[str, Any]:
    basis = decomposition.basis
    mean, components = basis.normalized_fields()
    payload = {
        "schema_version": "canondressgs.explicit_gaussian_residual_basis_artifact.v1",
        "channel_bounds": dict(channel_bounds), "rank": basis.rank,
        "mean_normalized": {name: value.detach().cpu() for name, value in mean.items()},
        "basis_normalized": {name: value.detach().cpu() for name, value in components.items()},
        "teacher_coefficients": {
            name: value.detach().cpu() for name, value in decomposition.teacher_coefficients.items()
        },
        "metadata": decomposition.metadata,
    }
    path = _basis_artifact_path(output_dir); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp"); torch.save(payload, temporary); temporary.replace(path)
    return {"path": str(path), "sha256": sha256(path), "metadata": decomposition.metadata}


def load_basis_artifact(output_dir: Path, device: torch.device) -> tuple[ExplicitGaussianResidualBasis, dict[str, torch.Tensor], dict[str, Any]]:
    path = _basis_artifact_path(output_dir)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "canondressgs.explicit_gaussian_residual_basis_artifact.v1":
        raise ValueError("basis artifact schema mismatch")
    basis = ExplicitGaussianResidualBasis(
        payload["mean_normalized"], payload["basis_normalized"], payload["channel_bounds"],
    ).to(device)
    if basis.fingerprint() != payload["metadata"]["basis_fingerprint"]:
        raise ValueError("basis artifact fingerprint mismatch")
    coefficients = {name: value.to(device=device, dtype=next(basis.buffers()).dtype) for name, value in payload["teacher_coefficients"].items()}
    return basis, coefficients, payload


def _load_p1_predictions(path: Path, device: torch.device) -> dict[str, GaussianClothingResiduals]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        outfit: GaussianClothingResiduals.from_dict({name: payload[outfit][name].to(device) for name in CHANNELS})
        for outfit in OUTFITS
    }


def _coefficient_error(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float((prediction - target).abs().mean())


def compose_predictions(
    basis: ExplicitGaussianResidualBasis,
    coefficients: Mapping[str, torch.Tensor],
    chunk_size: int,
) -> dict[str, GaussianClothingResiduals]:
    return {outfit: basis(coefficients[outfit], chunk_size=chunk_size) for outfit in OUTFITS}


def balanced_episode_schedule() -> list[tuple[str, str]]:
    return [(outfit, condition) for condition in CONDITIONS for outfit in OUTFITS]


def subset_reference_episode(
    episode: Mapping[str, Any], geometry: Mapping[str, Any], keep_indices: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not keep_indices or len(set(keep_indices)) != len(keep_indices):
        raise ValueError("reference subset must contain unique retained indices")
    count = int(episode["reference_images"].shape[0])
    if min(keep_indices) < 0 or max(keep_indices) >= count:
        raise IndexError("reference subset index is out of range")
    index = torch.tensor(keep_indices, dtype=torch.long, device=episode["reference_images"].device)
    result = {}
    for name in ("reference_images", "reference_cloth_masks", "reference_foreground_masks", "reference_poses", "reference_valid_mask"):
        result[name] = episode[name].index_select(0, index)
    result["reference_cameras"] = [episode["reference_cameras"][value] for value in keep_indices]
    result["reference_condition_ids"] = [episode["reference_condition_ids"][value] for value in keep_indices]
    geometry_result = dict(geometry)
    for name in ("surface_depth_maps", "surface_alpha_maps"):
        geometry_result[name] = geometry[name].index_select(0, index)
    return result, geometry_result


def _base_reference_images(context: Mapping[str, Any], outfit: str, episode: Mapping[str, Any]) -> torch.Tensor:
    return torch.stack([
        context["samples"][f"{outfit}/{condition}"]["target_base_rgb"]
        for condition in episode["reference_condition_ids"]
    ])


def _common_input_context(config: dict[str, Any], output_dir: Path, *, create: bool, load_reference_model: bool) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("formal explicit-basis closure requires CUDA")
    device = torch.device("cuda")
    seed = int(config["stage_b"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    branch, dirty = git_output("branch", "--show-current"), git_output("status", "--short")
    if branch != EXPECTED_BRANCH or dirty:
        raise RuntimeError(f"formal run requires clean {EXPECTED_BRANCH}; branch={branch} dirty={bool(dirty)}")
    if subprocess.call(["git", "merge-base", "--is-ancestor", EXPECTED_SOURCE_HEAD, "HEAD"], cwd=PROJECT_ROOT) != 0:
        raise RuntimeError("formal branch is not descended from the frozen PF-B HEAD")
    paths = {
        "source_manifest": Path(config["inputs"]["source_manifest"]),
        "stable_protected_attribution": Path(config["inputs"]["stable_protected_attribution"]),
        "parameterization_final_adjudication": Path(config["inputs"]["parameterization_attempt"]) / "final_adjudication/final_adjudication.json",
        "p1_predictions": Path(config["inputs"]["p1_predictions"]),
        "base_checkpoint": Path(config["base"]["model_dir"]) / config["base"]["checkpoint_path"],
        "lbs_grid": Path(config["base"]["lbs_grid_path"]),
    }
    expected = {
        "source_manifest": config["inputs"]["source_manifest_sha256"],
        "stable_protected_attribution": config["inputs"]["stable_protected_attribution_sha256"],
        "parameterization_final_adjudication": config["inputs"]["parameterization_final_adjudication_sha256"],
        "p1_predictions": config["inputs"]["p1_predictions_sha256"],
        "base_checkpoint": config["base"]["checkpoint_sha256"],
    }
    hashes = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[name] = sha256(path)
        if name in expected and hashes[name] != expected[name]:
            raise ValueError(f"immutable input SHA mismatch: {name}")
    parameterization_attempt = Path(config["inputs"]["parameterization_attempt"])
    parameterization_fingerprint = immutable_tree_metadata_fingerprint(parameterization_attempt)
    renderer_sources = {
        "gaussian_model": PROJECT_ROOT / "scene/gaussian_model.py",
        "production_render_adapter": PROJECT_ROOT / "tools/run_module4b_canonical_oracle_micropilot.py",
        "registered_sh1_wrapper": PROJECT_ROOT / "tools/run_image_conditioned_overfit_o01.py",
    }
    renderer_fingerprints = {name: sha256(path) for name, path in renderer_sources.items()}
    if create:
        for name in ("contract", "input_audit", "visual_acceptance", "final_adjudication"):
            (output_dir / name).mkdir(parents=True, exist_ok=True)
        atomic_text(output_dir / "contract/config_resolved.yaml", yaml.safe_dump(config, sort_keys=False))
        atomic_text(output_dir / "contract/command.txt", " ".join([sys.executable, *sys.argv]))
        atomic_json(output_dir / "input_audit/environment.json", o01.environment_snapshot())
        atomic_json(output_dir / "input_audit/input_manifest.json", {
            "task_id": TASK_ID,
            "git": {"branch": branch, "commit": git_output("rev-parse", "HEAD"), "clean": True, "source_head": EXPECTED_SOURCE_HEAD},
            "paths": {name: str(path) for name, path in paths.items()}, "sha256": hashes,
            "parameterization_attempt": str(parameterization_attempt),
            "parameterization_attempt_tree_metadata_fingerprint": parameterization_fingerprint,
            "renderer_source_paths": {name: str(path) for name, path in renderer_sources.items()},
            "renderer_source_sha256": renderer_fingerprints,
            "target_images_in_prediction_forward": False,
            "teacher_usage": "stage_a_offline_basis_construction_and_evaluation_only",
            "historical_checkpoint_warning": {
                "historical_sha256": config["base"]["historical_checkpoint_sha256_warning"],
                "current_sha256": config["base"]["checkpoint_sha256"], "same_file": False,
            },
        })
    samples, episodes, protocol = diagnosis._generic_outfit_data(paths["source_manifest"])
    base = training.load_frozen_mmlphuman_base(config["base"]["model_dir"], paths["base_checkpoint"], device=device)
    if int(base._xyz.shape[0]) != int(config["base"]["gaussian_count"]):
        raise ValueError("base Gaussian count differs from explicit-basis contract")
    protected_cpu, attribution = o01._load_stable_protected_mask(
        paths["stable_protected_attribution"], int(config["inputs"]["stable_protected_count"]), int(base._xyz.shape[0]),
    )
    protected_mask = protected_cpu.to(device)
    base_before = _tensor_state_fingerprint(_base_named_tensors(base))
    adapter = MMLPHumanAnchorDeformationAdapter.from_mmlphuman_base(base, lbs_grid_path=paths["lbs_grid"])
    background = torch.tensor(config["render"]["background"], device=device, dtype=base._xyz.dtype)
    for key in list(samples):
        samples[key] = diagnosis._to_device_nested(samples[key], device)
        episodes[key] = diagnosis._to_device_nested(episodes[key], device)
    geometries = {}
    for condition in CONDITIONS:
        geometries[condition] = training.prepare_real_reference_geometry(base, adapter, episodes[f"O01/{condition}"], background)
        for outfit in OUTFITS:
            key = f"{outfit}/{condition}"
            support, _ = o01._base_only_protected_support(
                base, samples[key], protected_mask, background, float(config["render"]["protected_support_alpha_threshold"]),
            )
            samples[key]["target_protected_mask"] = torch.maximum(samples[key]["target_protected_mask"], support)
    model = graph = backbone_before = None
    if load_reference_model:
        model, graph = construct_reference_model(base, config, device)
        backbone_before = o01._state_fingerprint(model.clothing_observation_encoder.backbone.state_dict())
    if create:
        atomic_json(output_dir / "input_audit/resolved_protocol.json", {
            "protocol": protocol, "attribution": attribution,
            "basis": {"rank": 1, "method": config["basis"]["method"], "space": config["basis"]["space"]},
            "reference_model_graph": graph,
            "prediction_forward_fields": ["reference RGB/masks", "reference pose/camera", "target pose/camera"],
            "forbidden_prediction_fields": ["target RGB/masks", "outfit_id", "diagnostic latent", "teacher residual", "teacher coefficient"],
        })
    return {
        "config": config, "output_dir": output_dir, "paths": paths, "base": base,
        "samples": samples, "episodes": episodes, "geometries": geometries,
        "protected_mask": protected_mask, "background": background, "model": model,
        "base_before": base_before, "backbone_before": backbone_before,
        "parameterization_fingerprint": parameterization_fingerprint,
        "renderer_sources": renderer_sources, "renderer_fingerprints": renderer_fingerprints,
    }


def _load_stage_a_teachers(context: Mapping[str, Any]) -> tuple[dict[str, GaussianClothingResiduals], dict[str, Any]]:
    config, device, base = context["config"], context["base"]._xyz.device, context["base"]
    targets, metadata = {}, {}
    for outfit in OUTFITS:
        path = Path(config["inputs"][f"oracle_{outfit.lower()}_checkpoint"])
        expected = config["inputs"][f"oracle_{outfit.lower()}_checkpoint_sha256"]
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"immutable Oracle input mismatch: {outfit}")
        oracle, metadata[outfit] = diagnosis._load_oracle(base, path, device)
        targets[outfit] = apply_protected_full_residual_guard(oracle.residuals(base), context["protected_mask"])
        if any(value.requires_grad or value.grad_fn is not None for value in targets[outfit].as_dict().values()):
            raise AssertionError("offline Oracle teacher retained an autograd graph")
    return targets, metadata


def _render_cache(
    context: Mapping[str, Any], predictions: Mapping[str, GaussianClothingResiduals],
) -> dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    cache = {outfit: {} for outfit in OUTFITS}
    with torch.no_grad():
        for outfit in OUTFITS:
            for condition in CONDITIONS:
                rgb, alpha = parameterization.render_prediction(
                    context["base"], context["samples"][f"{outfit}/{condition}"],
                    predictions[outfit], context["background"],
                )
                cache[outfit][condition] = (rgb.detach().cpu(), alpha.detach().cpu())
    return cache


def _basis_field_visuals(context: Mapping[str, Any], basis: ExplicitGaussianResidualBasis) -> None:
    import matplotlib.pyplot as plt

    visual = context["output_dir"] / "visual_acceptance"; visual.mkdir(parents=True, exist_ok=True)
    xyz = context["base"]._xyz.detach().float().cpu()
    sampled = torch.arange(0, xyz.shape[0], max(1, xyz.shape[0] // 80000))
    mean, components = basis.normalized_fields()
    figure, axes = plt.subplots(2, 6, figsize=(25, 8), constrained_layout=True)
    for row, (label, fields) in enumerate((("mean", mean), ("difference basis", components))):
        for column, name in enumerate(CHANNELS):
            value = fields[name][0] if row else fields[name]
            magnitude = value.detach().float().cpu().reshape(xyz.shape[0], -1).norm(dim=1)
            plot = axes[row, column].scatter(xyz[sampled, 0], xyz[sampled, 2], c=magnitude[sampled], s=1, cmap="magma")
            axes[row, column].set_title(f"{label} {name}"); axes[row, column].set_aspect("equal")
            figure.colorbar(plot, ax=axes[row, column], fraction=0.046)
    figure.savefig(visual / "basis_mean_and_difference_six_channel.png", dpi=110); plt.close(figure)
    for label, fields in (("mean_field", mean), ("difference_basis", components)):
        magnitudes = []
        for name in CHANNELS:
            value = fields[name][0] if label == "difference_basis" else fields[name]
            magnitudes.append(value.detach().float().cpu().reshape(xyz.shape[0], -1).square().mean(1))
        magnitude = torch.stack(magnitudes, 1).mean(1).sqrt()
        figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
        plot = axis.scatter(xyz[sampled, 0], xyz[sampled, 2], c=magnitude[sampled], s=1, cmap="magma")
        axis.set_title(label.replace("_", " ")); axis.set_aspect("equal"); figure.colorbar(plot, ax=axis)
        figure.savefig(visual / f"{label}.png", dpi=120); plt.close(figure)


def _stage_a_visuals(
    context: Mapping[str, Any], targets: Mapping[str, GaussianClothingResiduals],
    predictions: Mapping[str, GaussianClothingResiduals], cache: Mapping[str, Any],
) -> None:
    for outfit in OUTFITS:
        rows = []
        for condition in CONDITIONS:
            sample = context["samples"][f"{outfit}/{condition}"]
            oracle_rgb = cache[outfit][condition]["oracle"][0]
            predicted_rgb = cache[outfit][condition]["prediction"][0]
            garment = diagnosis._garment_mask(sample); protected = sample["target_protected_mask"]
            rows.append((f"{outfit}/{VIEWS[condition]}", [
                ("base", sample["target_base_rgb"], 3), ("target", sample["target_edit_rgb"], 3),
                ("Oracle", oracle_rgb, 3), ("basis teacher reconstruction", predicted_rgb, 3),
                ("Oracle-basis abs", (oracle_rgb - predicted_rgb).abs(), 3),
                ("torso/sleeve", v7.crop_tensor(predicted_rgb, garment, vertical="upper"), 3),
                ("trousers", v7.crop_tensor(predicted_rgb, garment, vertical="lower"), 3),
                ("shoes/protected", v7.crop_tensor(predicted_rgb, protected, vertical="lower"), 3),
            ]))
        o01._save_contact_sheet(context["output_dir"] / "visual_acceptance" / f"stage_a_{outfit}_contact_sheet.png", rows)


def run_stage_a(context: dict[str, Any]) -> dict[str, Any]:
    config, output_dir = context["config"], context["output_dir"]
    started = time.time(); torch.cuda.reset_peak_memory_stats()
    targets, oracle_meta = _load_stage_a_teachers(context)
    decomposition = build_centered_difference_basis(targets, bounds(config), OUTFITS)
    basis = decomposition.basis.to(context["base"]._xyz)
    artifact = save_basis_artifact(output_dir, decomposition, bounds(config))
    coefficients = {name: value.to(context["base"]._xyz) for name, value in decomposition.teacher_coefficients.items()}
    with torch.no_grad():
        predictions = compose_predictions(basis, coefficients, int(config["basis"]["chunk_size"]))
        report, cache = parameterization.evaluate_predictions(
            phase="stage_a", step=0, predictions=predictions, targets=targets,
            base=context["base"], samples=context["samples"], background=context["background"],
            config=config, output_dir=output_dir,
        )
        non_chunked = compose_predictions(basis, coefficients, basis.gaussian_count)
    parity = all(torch.equal(getattr(predictions[outfit], name), getattr(non_chunked[outfit], name)) for outfit in OUTFITS for name in CHANNELS)
    acceptance = config["stage_a"]["acceptance"]
    checks = {
        "mean_normalized_rmse": report["mean"]["normalized_rmse"] <= float(acceptance["mean_bound_normalized_rmse_max"]),
        "mean_direction_cosine": report["mean"]["direction_cosine"] >= float(acceptance["mean_direction_cosine_min"]),
        "mean_top_10pct_overlap": report["mean"]["top_10pct_overlap"] >= float(acceptance["mean_top_10pct_overlap_min"]),
        "o01_render_garment_mae": report["render_per_outfit_garment_mae"]["O01"] <= float(acceptance["per_outfit_render_garment_mae_max"]),
        "o08_render_garment_mae": report["render_per_outfit_garment_mae"]["O08"] <= float(acceptance["per_outfit_render_garment_mae_max"]),
        "chunked_non_chunked_bitwise_exact": parity,
        "optimizer_not_created": True,
        "basis_has_no_trainable_parameters": sum(parameter.numel() for parameter in basis.parameters()) == 0,
        "base_bitwise_frozen": context["base_before"] == _tensor_state_fingerprint(_base_named_tensors(context["base"])),
        "base_gradient_zero": _base_gradient_count(context["base"]) == 0,
    }
    status = "NUMERIC_PASS_VISUAL_PENDING" if all(checks.values()) else "FAIL"
    _stage_a_visuals(context, targets, predictions, cache); _basis_field_visuals(context, basis)
    result = {
        "status": status, "optimizer_steps": 0, "final": report, "checks": checks,
        "basis": {**artifact, "rank": basis.rank, "mean_scalar_count": basis.mean_scalar_count,
                  "basis_scalar_count": basis.basis_scalar_count, "total_explicit_scalar_count": basis.explicit_scalar_count,
                  "teacher_coefficients": {name: value.detach().cpu().tolist() for name, value in coefficients.items()}},
        "oracle_metadata": oracle_meta, "runtime_seconds": time.time() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "visual_status": "PENDING_ACTUAL_INSPECTION" if status.startswith("NUMERIC_PASS") else "NOT_ELIGIBLE_NUMERIC_FAIL",
    }
    atomic_json(output_dir / "stage_a/stage_a_metrics.json", result)
    atomic_json(output_dir / "stage_a/partial_status.json", {"status": status, "optimizer_steps": 0, "basis_sha256": artifact["sha256"]})
    return result


def _load_visual_observations(path: Path, required: Sequence[str]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("images_actually_opened") is not True or not payload.get("inspection_method"):
        raise ValueError("visual observations must record an actual image inspection")
    if not payload.get("images") or not payload.get("observations"):
        raise ValueError("visual observations require image paths and concrete observations")
    for name in required:
        if payload.get(name) not in {"PASS", "WARN", "FAIL"}:
            raise ValueError(f"visual observations lack {name}")
    return payload


def _require_stage(output_dir: Path, stage: str, visual: Mapping[str, Any]) -> dict[str, Any]:
    metrics = json.loads((output_dir / stage / f"{stage}_metrics.json").read_text(encoding="utf-8"))
    if metrics["status"] != "NUMERIC_PASS_VISUAL_PENDING" or visual.get(f"{stage}_status") != "PASS":
        raise RuntimeError(f"{stage} numeric and actual-visual PASS is required")
    return metrics


def _fixed_diagnostic_latents(config: Mapping[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    dimension = int(config["stage_b"]["diagnostic_latent_dim"])
    generator = torch.Generator(device="cpu").manual_seed(int(config["stage_b"]["seed"]))
    common = torch.randn(dimension, generator=generator, dtype=dtype) * 0.02
    first, second = common.clone(), common.clone(); first[0] = -1.0; second[0] = 1.0
    return {"O01": first.to(device), "O08": second.to(device)}


def _diagnostic_predictions(
    predictor: DiagnosticCoefficientPredictor, latents: Mapping[str, torch.Tensor],
    basis: ExplicitGaussianResidualBasis, chunk_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, GaussianClothingResiduals]]:
    coefficients = {outfit: predictor(latents[outfit]) for outfit in OUTFITS}
    return coefficients, compose_predictions(basis, coefficients, chunk_size)


def _stage_b_evaluate(
    context: Mapping[str, Any], predictor: DiagnosticCoefficientPredictor,
    latents: Mapping[str, torch.Tensor], basis: ExplicitGaussianResidualBasis,
    teacher_coefficients: Mapping[str, torch.Tensor], step: int, render: bool,
) -> dict[str, Any]:
    with torch.no_grad():
        coefficients, predictions = _diagnostic_predictions(
            predictor, latents, basis, int(context["config"]["basis"]["chunk_size"]),
        )
        targets = compose_predictions(basis, teacher_coefficients, int(context["config"]["basis"]["chunk_size"]))
        report, cache = parameterization.evaluate_predictions(
            phase="stage_b", step=step, predictions=predictions, targets=targets,
            base=context["base"], samples=context["samples"], background=context["background"],
            config=context["config"], output_dir=context["output_dir"],
        )
    report["coefficients"] = {name: value.detach().cpu().tolist() for name, value in coefficients.items()}
    report["teacher_coefficients"] = {name: value.detach().cpu().tolist() for name, value in teacher_coefficients.items()}
    report["coefficient_mae"] = float(np.mean([_coefficient_error(coefficients[name], teacher_coefficients[name]) for name in OUTFITS]))
    report["coefficient_separation_ratio"] = float(
        torch.linalg.vector_norm(coefficients["O08"] - coefficients["O01"]) /
        torch.linalg.vector_norm(teacher_coefficients["O08"] - teacher_coefficients["O01"]).clamp_min(1e-12)
    )
    if render:
        for outfit in OUTFITS:
            rows = []
            for condition in CONDITIONS:
                sample = context["samples"][f"{outfit}/{condition}"]
                oracle_rgb = cache[outfit][condition]["oracle"][0]; predicted_rgb = cache[outfit][condition]["prediction"][0]
                garment = diagnosis._garment_mask(sample); protected = sample["target_protected_mask"]
                rows.append((f"{outfit}/{VIEWS[condition]}", [
                    ("base", sample["target_base_rgb"], 3), ("target", sample["target_edit_rgb"], 3),
                    ("Oracle/basis teacher", oracle_rgb, 3), ("diagnostic coefficient", predicted_rgb, 3),
                    ("Oracle-diagnostic abs", (oracle_rgb - predicted_rgb).abs(), 3),
                    ("torso/sleeve", v7.crop_tensor(predicted_rgb, garment, vertical="upper"), 3),
                    ("trousers", v7.crop_tensor(predicted_rgb, garment, vertical="lower"), 3),
                    ("shoes/protected", v7.crop_tensor(predicted_rgb, protected, vertical="lower"), 3),
                ]))
            o01._save_contact_sheet(context["output_dir"] / "visual_acceptance" / f"stage_b_{outfit}_contact_sheet.png", rows)
    return report


def _restore_stage_b_coefficient_metrics(
    report: dict[str, Any], predictor: DiagnosticCoefficientPredictor,
    latents: Mapping[str, torch.Tensor], teacher_coefficients: Mapping[str, torch.Tensor],
) -> None:
    """Recreate coefficient-only fields that are appended after milestone persistence."""
    with torch.no_grad():
        coefficients = {outfit: predictor(latents[outfit]) for outfit in OUTFITS}
    report["coefficients"] = {
        name: value.detach().cpu().tolist() for name, value in coefficients.items()
    }
    report["teacher_coefficients"] = {
        name: value.detach().cpu().tolist() for name, value in teacher_coefficients.items()
    }
    report["coefficient_mae"] = float(np.mean([
        _coefficient_error(coefficients[name], teacher_coefficients[name]) for name in OUTFITS
    ]))
    report["coefficient_separation_ratio"] = float(
        torch.linalg.vector_norm(coefficients["O08"] - coefficients["O01"])
        / torch.linalg.vector_norm(
            teacher_coefficients["O08"] - teacher_coefficients["O01"]
        ).clamp_min(1e-12)
    )


def run_stage_b(
    context: dict[str, Any], visual: Mapping[str, Any], *, resume_acceptance_only: bool = False,
) -> dict[str, Any]:
    _require_stage(context["output_dir"], "stage_a", visual)
    config, output_dir, base = context["config"], context["output_dir"], context["base"]
    basis, teacher_coefficients, _ = load_basis_artifact(output_dir, base._xyz.device)
    set_requires_grad(basis, False)
    latents = _fixed_diagnostic_latents(config, base._xyz.device, base._xyz.dtype)
    predictor = DiagnosticCoefficientPredictor(
        int(config["stage_b"]["diagnostic_latent_dim"]), basis.rank,
        int(config["stage_b"]["predictor_hidden_dim"]),
    ).to(base._xyz)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=float(config["stage_b"]["learning_rate"]))
    started = time.time(); torch.cuda.reset_peak_memory_stats(); final_gradient = {}
    checkpoint_path = output_dir / "stage_b/checkpoints/checkpoint_step_000200.pth"
    if resume_acceptance_only:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Stage-B acceptance-only resume requires {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("stage") != "B" or int(checkpoint.get("global_step", -1)) != 200:
            raise RuntimeError("Stage-B checkpoint is not the completed 200-step checkpoint")
        predictor.load_state_dict(checkpoint["model"]["coefficient_predictor"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        milestones = {
            str(step): json.loads(
                (output_dir / f"stage_b/milestones/step_{step:06d}/metrics.json").read_text(encoding="utf-8")
            )
            for step in (0, 20, 100, 200)
        }
        optimizer.zero_grad(set_to_none=True)
        predicted = predictor(latents["O08"]); target = teacher_coefficients["O08"]
        loss = F.smooth_l1_loss(predicted, target)
        loss.backward()
        final_gradient = parameterization.trainable_snapshot({"coefficient_predictor": list(predictor.parameters())})
    else:
        milestones = {"0": _stage_b_evaluate(context, predictor, latents, basis, teacher_coefficients, 0, False)}
        for step in range(1, int(config["stage_b"]["max_steps"]) + 1):
            outfit = OUTFITS[(step - 1) % 2]
            optimizer.zero_grad(set_to_none=True)
            predicted = predictor(latents[outfit]); target = teacher_coefficients[outfit]
            loss = F.smooth_l1_loss(predicted, target)
            if not torch.isfinite(loss): raise FloatingPointError("Stage B coefficient loss is NaN or Inf")
            loss.backward(); final_gradient = parameterization.trainable_snapshot({"coefficient_predictor": list(predictor.parameters())})
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), float(config["stage_b"]["gradient_clip_norm"]), error_if_nonfinite=True)
            optimizer.step(); append_jsonl(output_dir / "stage_b/training.jsonl", {"step": step, "outfit": outfit, "coefficient_loss": float(loss.detach())})
            if step in config["stage_b"]["milestones"]:
                milestones[str(step)] = _stage_b_evaluate(
                    context, predictor, latents, basis, teacher_coefficients, step,
                    step == int(config["stage_b"]["max_steps"]),
                )
    final = milestones[str(config["stage_b"]["max_steps"])]
    if resume_acceptance_only:
        _restore_stage_b_coefficient_metrics(final, predictor, latents, teacher_coefficients)
    if not resume_acceptance_only:
        save_checkpoint(
            checkpoint_path, stage="B", step=200, model={"coefficient_predictor": predictor.state_dict()},
            optimizer=optimizer, scheduler=None, extra={"diagnostic_latents": {name: value.cpu() for name, value in latents.items()}},
        )
    restored = DiagnosticCoefficientPredictor(
        int(config["stage_b"]["diagnostic_latent_dim"]), basis.rank,
        int(config["stage_b"]["predictor_hidden_dim"]),
    ).to(base._xyz)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=float(config["stage_b"]["learning_rate"]))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    restored.load_state_dict(checkpoint["model"]["coefficient_predictor"], strict=True); restored_optimizer.load_state_dict(checkpoint["optimizer"])
    with torch.no_grad():
        output_exact = all(torch.equal(predictor(latents[name]), restored(latents[name])) for name in OUTFITS)
    current_rng = rng_state(); restore_rng(checkpoint["rng"])
    rng_exact = o01.object_fingerprint(rng_state()) == o01.object_fingerprint(checkpoint["rng"]); restore_rng(current_rng)
    resume = {
        "global_step_exact": int(checkpoint["global_step"]) == 200,
        "model_state_exact": _tensor_state_fingerprint(predictor.state_dict().items()) == _tensor_state_fingerprint(restored.state_dict().items()),
        "optimizer_state_exact": o01.object_fingerprint(optimizer.state_dict()) == o01.object_fingerprint(restored_optimizer.state_dict()),
        "rng_state_exact": rng_exact, "fixed_output_bitwise_exact": output_exact,
    }
    resume["pass"] = all(resume.values())
    acceptance = config["stage_b"]["acceptance"]
    checks = {
        "coefficient_mae": final["coefficient_mae"] <= float(acceptance["coefficient_mae_max"]),
        "reconstructed_residual_rmse": final["mean"]["normalized_rmse"] <= float(acceptance["reconstructed_residual_rmse_max"]),
        "direction_cosine": final["mean"]["direction_cosine"] >= float(acceptance["direction_cosine_min"]),
        "coefficient_separation": final["coefficient_separation_ratio"] > 0.9,
        "predictor_gradient_finite_nonzero": final_gradient["coefficient_predictor"]["gradient_finite"] and final_gradient["coefficient_predictor"]["gradient_l2"] > 0,
        "basis_frozen": sum(parameter.numel() for parameter in basis.parameters()) == 0,
        "checkpoint_resume": resume["pass"], "target_forward_leakage_zero": True,
        "base_bitwise_frozen": context["base_before"] == _tensor_state_fingerprint(_base_named_tensors(base)),
    }
    status = "NUMERIC_PASS_VISUAL_PENDING" if all(checks.values()) else "FAIL"
    result = {
        "status": status, "optimizer_steps": 200, "acceptance_only_resume": resume_acceptance_only,
        "milestones": milestones, "final": final,
        "checks": checks, "gradients": final_gradient, "checkpoint_resume": resume,
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
        "predictor_parameter_count": sum(parameter.numel() for parameter in predictor.parameters()),
        "runtime_seconds": time.time() - started, "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "visual_status": "PENDING_ACTUAL_INSPECTION" if status.startswith("NUMERIC_PASS") else "NOT_ELIGIBLE_NUMERIC_FAIL",
    }
    atomic_json(output_dir / "stage_b/stage_b_metrics.json", result)
    atomic_json(output_dir / "stage_b/partial_status.json", {"status": status, "completed_step": 200, "checkpoint": str(checkpoint_path), "checkpoint_sha256": result["checkpoint_sha256"]})
    return result


def stage_c_trainable_groups(model: Any, predictor: ReferenceBasisCoefficientPredictor) -> dict[str, list[torch.nn.Parameter]]:
    set_requires_grad(model, False); set_requires_grad(predictor, True)
    modules = {
        "encoder_projection": model.clothing_observation_encoder.projection_head,
        "encoder_embedding_norm": model.clothing_observation_encoder.embedding_norm,
        "aggregator": model.multiview_aggregator,
        "completion": model.canonical_clothing_completer,
    }
    groups = {}
    for name, module in modules.items():
        if module is None: raise RuntimeError(f"Stage C module is missing: {name}")
        set_requires_grad(module, True); groups[name] = list(module.parameters())
    groups["coefficient_predictor"] = list(predictor.parameters())
    if any(parameter.requires_grad for parameter in model.clothing_observation_encoder.backbone.parameters()):
        raise AssertionError("Stage C unfroze the image backbone")
    if model.support_conditioned_residual_decoder_v7 is not None:
        raise AssertionError("Stage C instantiated the forbidden per-Gaussian decoder")
    return groups


def stage_c_optimizer(model: Any, predictor: ReferenceBasisCoefficientPredictor, config: Mapping[str, Any]):
    groups = stage_c_trainable_groups(model, predictor); optimizer_config = config["stage_c"]["optimizer"]
    parameter_groups = [
        {"name": "encoder", "params": groups["encoder_projection"] + groups["encoder_embedding_norm"], "lr": float(optimizer_config["encoder_lr"])},
        {"name": "aggregator", "params": groups["aggregator"], "lr": float(optimizer_config["aggregator_lr"])},
        {"name": "completion", "params": groups["completion"], "lr": float(optimizer_config["completer_lr"])},
        {"name": "coefficient_predictor", "params": groups["coefficient_predictor"], "lr": float(optimizer_config["coefficient_predictor_lr"])},
    ]
    optimizer = torch.optim.Adam(parameter_groups, weight_decay=float(optimizer_config["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    return optimizer, scheduler, groups


def stage_c_coefficients(
    model: Any, predictor: ReferenceBasisCoefficientPredictor,
    episode: Mapping[str, Any], geometry: Mapping[str, Any],
) -> tuple[torch.Tensor, Mapping[str, Any], Any]:
    inputs = o01._forward_inputs(episode, geometry)
    assert_forward_boundary(inputs)
    online = model.encode_clothing_online(**inputs)
    completion = online["completion"]
    coefficient_output = predictor(
        online["global_clothing_embedding"], completion.completed_anchor_features, completion.confidence,
    )
    return coefficient_output.coefficients, online, coefficient_output


def stage_c_prediction(
    model: Any, predictor: ReferenceBasisCoefficientPredictor, basis: ExplicitGaussianResidualBasis,
    episode: Mapping[str, Any], geometry: Mapping[str, Any], *, chunk_size: int,
) -> tuple[torch.Tensor, GaussianClothingResiduals, Mapping[str, Any], Any]:
    coefficients, online, coefficient_output = stage_c_coefficients(model, predictor, episode, geometry)
    residuals = basis(coefficients, chunk_size=chunk_size)
    return coefficients, residuals, online, coefficient_output


def _episode_render_error(
    context: Mapping[str, Any], outfit: str, condition: str,
    prediction: GaussianClothingResiduals, target: GaussianClothingResiduals,
) -> tuple[dict[str, float], tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    sample = context["samples"][f"{outfit}/{condition}"]
    predicted = parameterization.render_prediction(context["base"], sample, prediction, context["background"])
    oracle = parameterization.render_prediction(context["base"], sample, target, context["background"])
    garment = diagnosis._garment_mask(sample)
    return {
        "garment_rgb_mae": o01._masked_mae(predicted[0], oracle[0], garment),
        "alpha_mae": float((predicted[1] - oracle[1]).abs().mean()),
    }, predicted, oracle


def _stage_c_variant(
    context: Mapping[str, Any], model: Any, predictor: ReferenceBasisCoefficientPredictor,
    basis: ExplicitGaussianResidualBasis, teacher_coefficients: Mapping[str, torch.Tensor],
    outfit: str, condition: str, episode: Mapping[str, Any], geometry: Mapping[str, Any],
) -> dict[str, Any]:
    coefficient, residual, online, coefficient_output = stage_c_prediction(
        model, predictor, basis, episode, geometry, chunk_size=int(context["config"]["basis"]["chunk_size"]),
    )
    target = basis(teacher_coefficients[outfit], chunk_size=int(context["config"]["basis"]["chunk_size"]))
    metrics = parameterization.residual_metrics(
        residual, target, bounds(context["config"]), active_epsilon=float(context["config"]["metrics"]["active_epsilon"]),
    )
    render, rendered, oracle = _episode_render_error(context, outfit, condition, residual, target)
    return {
        "coefficient": coefficient.detach(), "coefficient_error": _coefficient_error(coefficient, teacher_coefficients[outfit]),
        "residual": residual, "residual_metrics": metrics, "render_metrics": render,
        "rendered": (rendered[0].detach(), rendered[1].detach()), "oracle": (oracle[0].detach(), oracle[1].detach()),
        "global_feature": online["global_clothing_embedding"].detach(),
        "pooled_local_feature": coefficient_output.pooled_local_feature.detach(),
        "fused_feature": coefficient_output.fused_feature.detach(),
    }


def _serializable_variant(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "coefficient": value["coefficient"].cpu().tolist(), "coefficient_error": value["coefficient_error"],
        "residual_rmse": value["residual_metrics"]["normalized_rmse"],
        "residual_cosine": value["residual_metrics"]["direction_cosine"],
        "top_10pct_overlap": value["residual_metrics"]["top_10pct_overlap"],
        "render_garment_rgb_mae": value["render_metrics"]["garment_rgb_mae"],
        "render_alpha_mae": value["render_metrics"]["alpha_mae"],
    }


def evaluate_stage_c(
    context: Mapping[str, Any], model: Any, predictor: ReferenceBasisCoefficientPredictor,
    basis: ExplicitGaussianResidualBasis, teacher_coefficients: Mapping[str, torch.Tensor],
    *, step: int, full_variants: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model.eval(); predictor.eval(); rows, runtime_cache = {}, {}
    with torch.no_grad():
        for condition in CONDITIONS:
            for outfit in OUTFITS:
                key = f"{outfit}/{condition}"
                correct = _stage_c_variant(
                    context, model, predictor, basis, teacher_coefficients, outfit, condition,
                    context["episodes"][key], context["geometries"][condition],
                )
                runtime_cache[key] = {"correct": correct}; rows[key] = {"correct": _serializable_variant(correct)}
        mean = {
            "coefficient_mae": float(np.mean([rows[key]["correct"]["coefficient_error"] for key in rows])),
            "normalized_rmse": float(np.mean([rows[key]["correct"]["residual_rmse"] for key in rows])),
            "direction_cosine": float(np.mean([rows[key]["correct"]["residual_cosine"] for key in rows])),
            "top_10pct_overlap": float(np.mean([rows[key]["correct"]["top_10pct_overlap"] for key in rows])),
            "render_garment_rgb_mae": float(np.mean([rows[key]["correct"]["render_garment_rgb_mae"] for key in rows])),
            "render_alpha_mae": float(np.mean([rows[key]["correct"]["render_alpha_mae"] for key in rows])),
        }
        outfit_coefficients = {
            outfit: torch.stack([runtime_cache[f"{outfit}/{condition}"]["correct"]["coefficient"] for condition in CONDITIONS]).mean(0)
            for outfit in OUTFITS
        }
        teacher_distance = torch.linalg.vector_norm(teacher_coefficients["O08"] - teacher_coefficients["O01"]).clamp_min(1e-12)
        separation_ratio = float(torch.linalg.vector_norm(outfit_coefficients["O08"] - outfit_coefficients["O01"]) / teacher_distance)
        report: dict[str, Any] = {
            "step": step, "episodes": rows, "mean": mean,
            "outfit_mean_coefficients": {name: value.cpu().tolist() for name, value in outfit_coefficients.items()},
            "coefficient_separation_ratio": separation_ratio,
        }
        if full_variants:
            coefficient_margins, residual_margins, render_margins, wins = [], [], [], 0
            zero_changes, base_changes, feature_changes = [], [], []
            for condition in CONDITIONS:
                for outfit in OUTFITS:
                    key = f"{outfit}/{condition}"; other = "O08" if outfit == "O01" else "O01"
                    correct = runtime_cache[key]["correct"]
                    variants: dict[str, Any] = {}
                    variants["swapped"] = _stage_c_variant(
                        context, model, predictor, basis, teacher_coefficients, outfit, condition,
                        context["episodes"][f"{other}/{condition}"], context["geometries"][condition],
                    )
                    permuted_episode, permuted_geometry = reference_variant(
                        context["episodes"][key], context["geometries"][condition], "permuted",
                    )
                    variants["permuted"] = _stage_c_variant(context, model, predictor, basis, teacher_coefficients, outfit, condition, permuted_episode, permuted_geometry)
                    single_episode, single_geometry = reference_variant(
                        context["episodes"][key], context["geometries"][condition], "single_reference",
                    )
                    variants["single_reference"] = _stage_c_variant(context, model, predictor, basis, teacher_coefficients, outfit, condition, single_episode, single_geometry)
                    dropout_values = []
                    for dropped in range(3):
                        subset, subset_geometry = subset_reference_episode(
                            context["episodes"][key], context["geometries"][condition], [index for index in range(3) if index != dropped],
                        )
                        dropout_values.append(_stage_c_variant(context, model, predictor, basis, teacher_coefficients, outfit, condition, subset, subset_geometry))
                    zero_episode, zero_geometry = reference_variant(
                        context["episodes"][key], context["geometries"][condition], "zero_rgb_masks_kept",
                    )
                    variants["zero_rgb_masks_kept"] = _stage_c_variant(context, model, predictor, basis, teacher_coefficients, outfit, condition, zero_episode, zero_geometry)
                    base_episode, base_geometry = reference_variant(
                        context["episodes"][key], context["geometries"][condition], "base_rgb_substitution",
                        substitute_images=_base_reference_images(context, outfit, context["episodes"][key]),
                    )
                    variants["base_rgb_replacement"] = _stage_c_variant(context, model, predictor, basis, teacher_coefficients, outfit, condition, base_episode, base_geometry)
                    swapped = variants["swapped"]
                    coefficient_margin = swapped["coefficient_error"] - correct["coefficient_error"]
                    residual_margin = swapped["residual_metrics"]["normalized_rmse"] - correct["residual_metrics"]["normalized_rmse"]
                    render_margin = swapped["render_metrics"]["garment_rgb_mae"] - correct["render_metrics"]["garment_rgb_mae"]
                    coefficient_margins.append(coefficient_margin); residual_margins.append(residual_margin); render_margins.append(render_margin)
                    if coefficient_margin > 0 and residual_margin > 0 and render_margin > 0: wins += 1
                    zero_change = float((variants["zero_rgb_masks_kept"]["coefficient"] - correct["coefficient"]).abs().mean())
                    base_change = float((variants["base_rgb_replacement"]["coefficient"] - correct["coefficient"]).abs().mean())
                    zero_changes.append(zero_change); base_changes.append(base_change)
                    feature_changes.append({
                        "global_correct_vs_swapped_l2": float(torch.linalg.vector_norm(correct["global_feature"] - swapped["global_feature"])),
                        "local_correct_vs_swapped_l2": float(torch.linalg.vector_norm(correct["pooled_local_feature"] - swapped["pooled_local_feature"])),
                        "fused_correct_vs_zero_l2": float(torch.linalg.vector_norm(correct["fused_feature"] - variants["zero_rgb_masks_kept"]["fused_feature"])),
                        "fused_correct_vs_base_l2": float(torch.linalg.vector_norm(correct["fused_feature"] - variants["base_rgb_replacement"]["fused_feature"])),
                    })
                    rows[key]["variants"] = {name: _serializable_variant(value) for name, value in variants.items()}
                    rows[key]["dropout"] = [_serializable_variant(value) for value in dropout_values]
                    rows[key]["margins"] = {"coefficient": coefficient_margin, "residual": residual_margin, "render": render_margin}
                    rows[key]["reference_sensitivity"] = {"zero_coefficient_change": zero_change, "base_coefficient_change": base_change, **feature_changes[-1]}
                    rows[key]["target_pose_camera_fixed_for_swap"] = True
                    # Final contact sheets only consume the correct/swapped pair.  Keep
                    # counterfactual tensors out of the long-lived cache so the 72-way
                    # evaluation does not retain several GiB of residual/render tensors.
                    runtime_cache[key]["swapped"] = variants["swapped"]
            report.update({
                "mean_correct_vs_swapped_coefficient_margin": float(np.mean(coefficient_margins)),
                "mean_correct_vs_swapped_residual_margin": float(np.mean(residual_margins)),
                "mean_correct_vs_swapped_render_margin": float(np.mean(render_margins)),
                "correct_episode_wins": wins,
                "mean_zero_rgb_coefficient_change": float(np.mean(zero_changes)),
                "mean_base_rgb_replacement_coefficient_change": float(np.mean(base_changes)),
                "feature_change_mean": {name: float(np.mean([value[name] for value in feature_changes])) for name in feature_changes[0]},
                "target_forward_leakage": False,
                "mean_outfit_collapse": separation_ratio <= 0.5,
            })
    milestone = context["output_dir"] / "stage_c/milestones" / f"step_{step:06d}"
    atomic_json(milestone / "metrics.json", report)
    model.train(); predictor.train()
    return report, runtime_cache


def _load_diagnostic_reconstruction(
    context: Mapping[str, Any], basis: ExplicitGaussianResidualBasis,
) -> dict[str, GaussianClothingResiduals]:
    config, checkpoint_path = context["config"], context["output_dir"] / "stage_b/checkpoints/checkpoint_step_000200.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    predictor = DiagnosticCoefficientPredictor(
        int(config["stage_b"]["diagnostic_latent_dim"]), basis.rank,
        int(config["stage_b"]["predictor_hidden_dim"]),
    ).to(context["base"]._xyz)
    predictor.load_state_dict(checkpoint["model"]["coefficient_predictor"], strict=True); predictor.eval()
    latents = {name: value.to(context["base"]._xyz) for name, value in checkpoint["extra"]["diagnostic_latents"].items()}
    with torch.no_grad():
        _, predictions = _diagnostic_predictions(predictor, latents, basis, int(config["basis"]["chunk_size"]))
    return predictions


def _coefficient_visuals(
    context: Mapping[str, Any], teacher_coefficients: Mapping[str, torch.Tensor], final: Mapping[str, Any],
) -> None:
    import matplotlib.pyplot as plt

    visual = context["output_dir"] / "visual_acceptance"
    labels = list(OUTFITS); teacher = [float(teacher_coefficients[name][0]) for name in labels]
    predicted = [float(final["outfit_mean_coefficients"][name][0]) for name in labels]
    x = np.arange(len(labels)); figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.bar(x - 0.18, teacher, 0.36, label="teacher"); axis.bar(x + 0.18, predicted, 0.36, label="correct reference")
    axis.set_xticks(x, labels); axis.set_ylabel("rank-1 coefficient"); axis.legend(); axis.axhline(0, color="black", linewidth=0.8)
    figure.savefig(visual / "coefficient_distribution.png", dpi=140); plt.close(figure)
    rows = final["episodes"]; correct = [float(rows[f"{name}/{condition}"]["correct"]["coefficient"][0]) for name in labels for condition in CONDITIONS]
    swapped = [float(rows[f"{name}/{condition}"]["variants"]["swapped"]["coefficient"][0]) for name in labels for condition in CONDITIONS]
    labels_episode = [f"{name}-{VIEWS[condition]}" for name in labels for condition in CONDITIONS]
    x = np.arange(len(labels_episode)); figure, axis = plt.subplots(figsize=(14, 5), constrained_layout=True)
    axis.plot(x, correct, "o-", label="correct"); axis.plot(x, swapped, "x--", label="swapped")
    axis.set_xticks(x, labels_episode, rotation=30, ha="right"); axis.legend(); axis.axhline(0, color="black", linewidth=0.8)
    figure.savefig(visual / "correct_swapped_coefficient_comparison.png", dpi=140); plt.close(figure)


def _stage_c_visuals(
    context: Mapping[str, Any], basis: ExplicitGaussianResidualBasis,
    teacher_coefficients: Mapping[str, torch.Tensor], runtime_cache: Mapping[str, Any], final: Mapping[str, Any],
) -> None:
    import matplotlib.pyplot as plt

    p1 = _load_p1_predictions(context["paths"]["p1_predictions"], context["base"]._xyz.device)
    teacher = compose_predictions(basis, teacher_coefficients, int(context["config"]["basis"]["chunk_size"]))
    diagnostic = _load_diagnostic_reconstruction(context, basis)
    p1_cache, teacher_cache, diagnostic_cache = _render_cache(context, p1), _render_cache(context, teacher), _render_cache(context, diagnostic)
    for outfit in OUTFITS:
        rows = []
        for condition in CONDITIONS:
            key = f"{outfit}/{condition}"; sample = context["samples"][key]
            correct, swapped = runtime_cache[key]["correct"], runtime_cache[key]["swapped"]
            oracle_rgb = teacher_cache[outfit][condition][0]
            garment = diagnosis._garment_mask(sample); protected = sample["target_protected_mask"]
            rows.append((f"{outfit}/{VIEWS[condition]}", [
                ("base", sample["target_base_rgb"], 3), ("target", sample["target_edit_rgb"], 3),
                ("Oracle", oracle_rgb, 3), ("P1", p1_cache[outfit][condition][0], 3),
                ("explicit basis teacher", teacher_cache[outfit][condition][0], 3),
                ("diagnostic coefficient", diagnostic_cache[outfit][condition][0], 3),
                ("correct reference", correct["rendered"][0], 3), ("swapped reference", swapped["rendered"][0], 3),
                ("Oracle-reference abs", (oracle_rgb - correct["rendered"][0]).abs(), 3),
                ("torso/sleeve", v7.crop_tensor(correct["rendered"][0], garment, vertical="upper"), 3),
                ("trousers", v7.crop_tensor(correct["rendered"][0], garment, vertical="lower"), 3),
                ("shoes/protected", v7.crop_tensor(correct["rendered"][0], protected, vertical="lower"), 3),
            ]))
        o01._save_contact_sheet(context["output_dir"] / "visual_acceptance" / f"stage_c_{outfit}_contact_sheet.png", rows)
    xyz = context["base"]._xyz.detach().float().cpu(); sampled = torch.arange(0, xyz.shape[0], max(1, xyz.shape[0] // 80000))
    figure, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    for row, outfit in enumerate(OUTFITS):
        condition = "cond_000017"; key = f"{outfit}/{condition}"
        correct, swapped = runtime_cache[key]["correct"]["residual"], runtime_cache[key]["swapped"]["residual"]
        target = teacher[outfit]
        values = [
            ("Oracle magnitude", parameterization.row_magnitude(target, bounds(context["config"]))),
            ("correct magnitude", parameterization.row_magnitude(correct, bounds(context["config"]))),
            ("swapped magnitude", parameterization.row_magnitude(swapped, bounds(context["config"]))),
            ("correct error magnitude", parameterization.row_magnitude(GaussianClothingResiduals(**{
                name: getattr(correct, name) - getattr(target, name) for name in CHANNELS
            }), bounds(context["config"]))),
        ]
        for column, (title, magnitude) in enumerate(values):
            magnitude = magnitude.detach().cpu(); plot = axes[row, column].scatter(xyz[sampled, 0], xyz[sampled, 2], c=magnitude[sampled], s=1, cmap="magma")
            axes[row, column].set_title(f"{outfit} {title}"); axes[row, column].set_aspect("equal"); figure.colorbar(plot, ax=axes[row, column], fraction=0.046)
    figure.savefig(context["output_dir"] / "visual_acceptance/residual_magnitude_difference.png", dpi=120); plt.close(figure)
    _coefficient_visuals(context, teacher_coefficients, final)


def _stage_c_state(model: Any, predictor: ReferenceBasisCoefficientPredictor) -> dict[str, Any]:
    return {
        "encoder_projection": model.clothing_observation_encoder.projection_head.state_dict(),
        "encoder_embedding_norm": model.clothing_observation_encoder.embedding_norm.state_dict(),
        "aggregator": model.multiview_aggregator.state_dict(),
        "completion": model.canonical_clothing_completer.state_dict(),
        "coefficient_predictor": predictor.state_dict(),
    }


def _load_stage_c_state(model: Any, predictor: ReferenceBasisCoefficientPredictor, state: Mapping[str, Any]) -> None:
    model.clothing_observation_encoder.projection_head.load_state_dict(state["encoder_projection"], strict=True)
    model.clothing_observation_encoder.embedding_norm.load_state_dict(state["encoder_embedding_norm"], strict=True)
    model.multiview_aggregator.load_state_dict(state["aggregator"], strict=True)
    model.canonical_clothing_completer.load_state_dict(state["completion"], strict=True)
    predictor.load_state_dict(state["coefficient_predictor"], strict=True)


def run_stage_c(context: dict[str, Any], visual: Mapping[str, Any]) -> dict[str, Any]:
    _require_stage(context["output_dir"], "stage_a", visual); _require_stage(context["output_dir"], "stage_b", visual)
    config, output_dir, base, model = context["config"], context["output_dir"], context["base"], context["model"]
    if model is None: raise RuntimeError("Stage C requires the reference model")
    basis, teacher_coefficients, basis_payload = load_basis_artifact(output_dir, base._xyz.device); set_requires_grad(basis, False)
    predictor = ReferenceBasisCoefficientPredictor(
        int(config["model"]["embedding_dim"]), int(config["model"]["local_feature_dim"]),
        basis.rank, int(config["stage_c"]["predictor_hidden_dim"]),
    ).to(base._xyz)
    optimizer, scheduler, groups = stage_c_optimizer(model, predictor, config)
    milestones = {}; initial, _ = evaluate_stage_c(context, model, predictor, basis, teacher_coefficients, step=0, full_variants=False); milestones["0"] = initial
    schedule = balanced_episode_schedule(); counts = {f"{outfit}/{condition}": 0 for condition in CONDITIONS for outfit in OUTFITS}
    outfit_counts = {outfit: 0 for outfit in OUTFITS}; view_counts = {condition: 0 for condition in CONDITIONS}
    started = time.time(); torch.cuda.reset_peak_memory_stats(); final_gradients = {}
    for step in range(1, int(config["stage_c"]["max_steps"]) + 1):
        outfit, condition = schedule[(step - 1) % len(schedule)]; key = f"{outfit}/{condition}"
        optimizer.zero_grad(set_to_none=True)
        coefficients, _, _ = stage_c_coefficients(model, predictor, context["episodes"][key], context["geometries"][condition])
        loss = F.smooth_l1_loss(coefficients, teacher_coefficients[outfit])
        if not torch.isfinite(loss): raise FloatingPointError("Stage C coefficient loss is NaN or Inf")
        loss.backward(); final_gradients = parameterization.trainable_snapshot(groups)
        parameters = [parameter for values in groups.values() for parameter in values]
        torch.nn.utils.clip_grad_norm_(parameters, float(config["stage_c"]["gradient_clip_norm"]), error_if_nonfinite=True)
        optimizer.step(); scheduler.step(); counts[key] += 1; outfit_counts[outfit] += 1; view_counts[condition] += 1
        append_jsonl(output_dir / "stage_c/training.jsonl", {"step": step, "outfit": outfit, "condition": condition, "coefficient_loss": float(loss.detach())})
        if step in config["stage_c"]["milestones"]:
            report, cache = evaluate_stage_c(
                context, model, predictor, basis, teacher_coefficients, step=step,
                full_variants=step == int(config["stage_c"]["max_steps"]),
            ); milestones[str(step)] = report
            checkpoint_path = output_dir / "stage_c/checkpoints" / f"checkpoint_step_{step:06d}.pth"
            save_checkpoint(
                checkpoint_path, stage="C", step=step, model=_stage_c_state(model, predictor),
                optimizer=optimizer, scheduler=scheduler,
                extra={"schedule_position": step % len(schedule), "episode_counts": dict(counts),
                       "outfit_counts": dict(outfit_counts), "view_counts": dict(view_counts),
                       "basis_sha256": sha256(_basis_artifact_path(output_dir)), "target_view_used_in_forward": False},
            )
            atomic_json(output_dir / "stage_c/partial_status.json", {"status": "RUNNING", "completed_step": step, "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path)})
    final = milestones[str(config["stage_c"]["max_steps"])]
    _stage_c_visuals(context, basis, teacher_coefficients, cache, final)
    checkpoint_path = output_dir / "stage_c/checkpoints" / f"checkpoint_step_{int(config['stage_c']['max_steps']):06d}.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    restored_model, _ = construct_reference_model(base, config, base._xyz.device)
    restored_predictor = ReferenceBasisCoefficientPredictor(
        int(config["model"]["embedding_dim"]), int(config["model"]["local_feature_dim"]),
        basis.rank, int(config["stage_c"]["predictor_hidden_dim"]),
    ).to(base._xyz)
    restored_optimizer, restored_scheduler, _ = stage_c_optimizer(restored_model, restored_predictor, config)
    _load_stage_c_state(restored_model, restored_predictor, checkpoint["model"])
    restored_optimizer.load_state_dict(checkpoint["optimizer"]); restored_scheduler.load_state_dict(checkpoint["scheduler"])
    fixed_key = "O01/cond_000017"
    model.eval(); predictor.eval(); restored_model.eval(); restored_predictor.eval()
    with torch.no_grad():
        before, _, _ = stage_c_coefficients(model, predictor, context["episodes"][fixed_key], context["geometries"]["cond_000017"])
        after, _, _ = stage_c_coefficients(restored_model, restored_predictor, context["episodes"][fixed_key], context["geometries"]["cond_000017"])
    current_rng = rng_state(); restore_rng(checkpoint["rng"])
    rng_exact = o01.object_fingerprint(rng_state()) == o01.object_fingerprint(checkpoint["rng"]); restore_rng(current_rng)
    resume = {
        "global_step_exact": int(checkpoint["global_step"]) == int(config["stage_c"]["max_steps"]),
        "model_state_exact": o01.object_fingerprint(_stage_c_state(model, predictor)) == o01.object_fingerprint(_stage_c_state(restored_model, restored_predictor)),
        "optimizer_state_exact": o01.object_fingerprint(optimizer.state_dict()) == o01.object_fingerprint(restored_optimizer.state_dict()),
        "scheduler_state_exact": o01.object_fingerprint(scheduler.state_dict()) == o01.object_fingerprint(restored_scheduler.state_dict()),
        "rng_state_exact": rng_exact, "schedule_position_exact": int(checkpoint["extra"]["schedule_position"]) == 0,
        "fixed_coefficient_bitwise_exact": torch.equal(before, after),
    }
    resume["pass"] = all(resume.values())
    acceptance = config["stage_c"]["acceptance"]
    checks = {
        "coefficient_mae": final["mean"]["coefficient_mae"] <= float(acceptance["coefficient_mae_max"]),
        "reconstructed_residual_rmse": final["mean"]["normalized_rmse"] <= float(acceptance["reconstructed_residual_rmse_max"]),
        "direction_cosine": final["mean"]["direction_cosine"] >= float(acceptance["direction_cosine_min"]),
        "correct_vs_swapped_coefficient_margin": final["mean_correct_vs_swapped_coefficient_margin"] > float(acceptance["correct_vs_swapped_coefficient_margin_min"]),
        "correct_vs_swapped_residual_margin": final["mean_correct_vs_swapped_residual_margin"] > float(acceptance["correct_vs_swapped_residual_margin_min"]),
        "correct_vs_swapped_render_margin": final["mean_correct_vs_swapped_render_margin"] > float(acceptance["correct_vs_swapped_render_margin_min"]),
        "correct_episode_wins": int(final["correct_episode_wins"]) >= int(acceptance["correct_episode_wins_min"]),
        "zero_rgb_changes_coefficient": final["mean_zero_rgb_coefficient_change"] >= float(acceptance["reference_variant_coefficient_change_min"]),
        "base_rgb_changes_coefficient": final["mean_base_rgb_replacement_coefficient_change"] >= float(acceptance["reference_variant_coefficient_change_min"]),
        "no_mean_outfit_collapse": not bool(final["mean_outfit_collapse"]),
        "target_forward_leakage_zero": not bool(final["target_forward_leakage"]),
        "all_trainable_groups_gradient_finite_nonzero": all(value["gradient_finite"] and value["gradient_l2"] > 0 for value in final_gradients.values()),
        "episode_updates_balanced": len(set(counts.values())) == 1 and set(counts.values()) == {62},
        "basis_frozen": sum(parameter.numel() for parameter in basis.parameters()) == 0 and sha256(_basis_artifact_path(output_dir)) == checkpoint["extra"]["basis_sha256"],
        "base_bitwise_frozen": context["base_before"] == _tensor_state_fingerprint(_base_named_tensors(base)),
        "base_gradient_zero": _base_gradient_count(base) == 0,
        "image_backbone_bitwise_frozen": context["backbone_before"] == o01._state_fingerprint(model.clothing_observation_encoder.backbone.state_dict()),
        "image_backbone_gradient_zero": sum(parameter.grad is not None for parameter in model.clothing_observation_encoder.backbone.parameters()) == 0,
        "legacy_mmlp_gradient_zero": sum(parameter.grad is not None for parameter in model.dressable_model.anchor_clothing_mlp.parameters()) == 0,
        "per_gaussian_decoder_absent": model.support_conditioned_residual_decoder_v7 is None,
        "checkpoint_resume": resume["pass"],
    }
    status = "NUMERIC_PASS_VISUAL_PENDING" if all(checks.values()) else "FAIL"
    result = {
        "status": status, "optimizer_steps": int(config["stage_c"]["max_steps"]),
        "milestones": milestones, "final": final, "checks": checks, "gradients": final_gradients,
        "updates": {"episodes": counts, "outfits": outfit_counts, "views": view_counts},
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path), "checkpoint_resume": resume,
        "coefficient_predictor_parameter_count": sum(parameter.numel() for parameter in predictor.parameters()),
        "total_trainable_parameter_count": sum(parameter.numel() for values in groups.values() for parameter in values),
        "runtime_seconds": time.time() - started, "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "basis_artifact_sha256": sha256(_basis_artifact_path(output_dir)), "basis_metadata": basis_payload["metadata"],
        "visual_status": "PENDING_ACTUAL_INSPECTION" if status.startswith("NUMERIC_PASS") else "NOT_ELIGIBLE_NUMERIC_FAIL",
    }
    atomic_json(output_dir / "stage_c/stage_c_metrics.json", result)
    atomic_json(output_dir / "stage_c/partial_status.json", {"status": status, "completed_step": int(config["stage_c"]["max_steps"]), "checkpoint": str(checkpoint_path), "checkpoint_sha256": result["checkpoint_sha256"]})
    return result


def finalize(context: Mapping[str, Any], visual_path: Path) -> dict[str, Any]:
    output_dir = context["output_dir"]
    visual = _load_visual_observations(visual_path, ("stage_a_status", "stage_b_status", "stage_c_status"))
    stage_a = json.loads((output_dir / "stage_a/stage_a_metrics.json").read_text(encoding="utf-8"))
    stage_b_path, stage_c_path = output_dir / "stage_b/stage_b_metrics.json", output_dir / "stage_c/stage_c_metrics.json"
    stage_b = json.loads(stage_b_path.read_text(encoding="utf-8")) if stage_b_path.is_file() else {"status": "NOT_RUN", "optimizer_steps": 0}
    stage_c = json.loads(stage_c_path.read_text(encoding="utf-8")) if stage_c_path.is_file() else {"status": "NOT_RUN", "optimizer_steps": 0}
    a_pass = stage_a["status"] == "NUMERIC_PASS_VISUAL_PENDING" and visual["stage_a_status"] == "PASS"
    b_pass = stage_b["status"] == "NUMERIC_PASS_VISUAL_PENDING" and visual["stage_b_status"] == "PASS"
    c_pass = stage_c["status"] == "NUMERIC_PASS_VISUAL_PENDING" and visual["stage_c_status"] == "PASS"
    if not a_pass:
        case, next_task, allow = "LB-0", "FIX_EXPLICIT_BASIS_COMPOSITION_CONTRACT", False
    elif not b_pass:
        case, next_task, allow = "LB-H", "FIX_BASIS_COEFFICIENT_HEAD", False
    elif not c_pass:
        case, next_task, allow = "LB-R", "REDESIGN_REFERENCE_TO_BASIS_COEFFICIENT_FUSION", False
    else:
        case, next_task, allow = "LB-P", "EXPAND_RESIDUAL_BASIS_TO_MULTI_OUTFIT_AND_HELD_OUT_EVALUATION", True
    manifest = json.loads((output_dir / "input_audit/input_manifest.json").read_text(encoding="utf-8"))
    model = context["model"]
    freeze = {
        "base_before": context["base_before"], "base_after": _tensor_state_fingerprint(_base_named_tensors(context["base"])),
        "base_gradient_count": _base_gradient_count(context["base"]),
        "parameterization_outputs_immutable": immutable_tree_metadata_fingerprint(Path(manifest["parameterization_attempt"])) == manifest["parameterization_attempt_tree_metadata_fingerprint"],
        "renderer_source_before": context["renderer_fingerprints"],
        "renderer_source_after": {name: sha256(path) for name, path in context["renderer_sources"].items()},
    }
    freeze["base_bitwise_unchanged"] = freeze["base_before"] == freeze["base_after"]
    freeze["renderer_source_unchanged"] = freeze["renderer_source_before"] == freeze["renderer_source_after"]
    if model is not None:
        freeze.update({
            "image_backbone_before": context["backbone_before"],
            "image_backbone_after": o01._state_fingerprint(model.clothing_observation_encoder.backbone.state_dict()),
            "image_backbone_gradient_count": sum(parameter.grad is not None for parameter in model.clothing_observation_encoder.backbone.parameters()),
            "legacy_mmlp_gradient_count": sum(parameter.grad is not None for parameter in model.dressable_model.anchor_clothing_mlp.parameters()),
            "per_gaussian_decoder_absent": model.support_conditioned_residual_decoder_v7 is None,
        })
        freeze["image_backbone_bitwise_unchanged"] = freeze["image_backbone_before"] == freeze["image_backbone_after"]
    if not freeze["base_bitwise_unchanged"] or freeze["base_gradient_count"] != 0 or not freeze["parameterization_outputs_immutable"] or not freeze["renderer_source_unchanged"]:
        raise AssertionError("explicit-basis frozen asset contract failed")
    if model is not None and (not freeze["image_backbone_bitwise_unchanged"] or freeze["image_backbone_gradient_count"] != 0 or freeze["legacy_mmlp_gradient_count"] != 0 or not freeze["per_gaussian_decoder_absent"]):
        raise AssertionError("Stage C frozen model contract failed")
    result = {
        "task_id": TASK_ID, "run_commit": manifest["git"]["commit"], "finalization_commit": git_output("rev-parse", "HEAD"),
        "stage_a": stage_a, "stage_b": stage_b, "stage_c": stage_c,
        "visual_acceptance": visual, "freeze": freeze, "target_forward_leakage": False,
        "final_case": case, "status": "PASS" if case == "LB-P" else "FAIL",
        "multi_outfit_basis_expansion_allowed": allow, "next_task": next_task,
        "method_boundary": "controlled O01/O08 validation only; no unseen-outfit or arbitrary-garment claim",
    }
    atomic_json(output_dir / "visual_acceptance/visual_acceptance.json", visual)
    atomic_text(output_dir / "visual_acceptance/VISUAL_ACCEPTANCE.md", "\n".join([
        f"# {TASK_ID} visual acceptance", "", f"- Images actually opened: `{visual['images_actually_opened']}`",
        f"- Inspection method: {visual['inspection_method']}",
        f"- Stage A: **{visual['stage_a_status']}**", f"- Stage B: **{visual['stage_b_status']}**", f"- Stage C: **{visual['stage_c_status']}**",
        "", "## Images", "", *[f"- `{item}`" for item in visual["images"]],
        "", "## Observations", "", *[f"- {item}" for item in visual["observations"]],
    ]))
    atomic_json(output_dir / "final_adjudication/final_adjudication.json", result)
    atomic_text(output_dir / "final_adjudication/FINAL_ADJUDICATION.md", "\n".join([
        f"# {TASK_ID}", "", f"- Run commit: `{result['run_commit']}`", f"- Finalization commit: `{result['finalization_commit']}`",
        f"- Stage A: **{stage_a['status']} / visual {visual['stage_a_status']}**",
        f"- Stage B: **{stage_b['status']} / visual {visual['stage_b_status']}**",
        f"- Stage C: **{stage_c['status']} / visual {visual['stage_c_status']}**",
        f"- Final case: **{case}**", f"- Multi-outfit basis expansion allowed: `{allow}`", f"- Next unique task: `{next_task}`",
    ]))
    atomic_json(output_dir / "RUN_STATUS.json", {
        "task_id": TASK_ID, "status": "COMPLETE", "final_case": case,
        "optimizer_steps": int(stage_a["optimizer_steps"]) + int(stage_b["optimizer_steps"]) + int(stage_c["optimizer_steps"]),
        "completed_at": time.time(),
    })
    return result


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config.resolve())
    output_dir = Path(args.output_dir or Path(config["output"]["task_root"]) / config["output"]["attempt"])
    create = args.phase == "stage_a"
    if create:
        if output_dir.exists(): raise FileExistsError(f"append-only explicit-basis attempt already exists: {output_dir}")
        output_dir.mkdir(parents=True)
        atomic_json(output_dir / "RUN_STATUS.json", {"task_id": TASK_ID, "status": "RUNNING", "phase": "input_audit", "optimizer_steps": 0})
    elif not output_dir.is_dir():
        raise FileNotFoundError(output_dir)
    load_reference_model = args.phase in {"stage_c", "finalize"}
    context = _common_input_context(config, output_dir, create=create, load_reference_model=load_reference_model)
    try:
        if args.phase == "stage_a":
            atomic_json(output_dir / "RUN_STATUS.json", {"task_id": TASK_ID, "status": "RUNNING", "phase": "STAGE_A", "optimizer_steps": 0})
            result = run_stage_a(context)
            atomic_json(output_dir / "RUN_STATUS.json", {"task_id": TASK_ID, "status": "AWAITING_STAGE_A_VISUAL_INSPECTION" if result["status"].startswith("NUMERIC_PASS") else "AWAITING_FINAL_ADJUDICATION", "phase": "STAGE_A_COMPLETE", "optimizer_steps": 0})
        elif args.phase == "stage_b":
            if args.visual_observations is None: raise ValueError("Stage B requires actual Stage-A visual observations")
            visual = _load_visual_observations(args.visual_observations, ("stage_a_status",))
            atomic_json(output_dir / "visual_acceptance/stage_a_visual_acceptance.json", visual)
            atomic_json(output_dir / "RUN_STATUS.json", {"task_id": TASK_ID, "status": "RUNNING", "phase": "STAGE_B", "optimizer_steps": 0})
            result = run_stage_b(context, visual, resume_acceptance_only=args.resume_acceptance_only)
            atomic_json(output_dir / "RUN_STATUS.json", {"task_id": TASK_ID, "status": "AWAITING_STAGE_B_VISUAL_INSPECTION" if result["status"].startswith("NUMERIC_PASS") else "AWAITING_FINAL_ADJUDICATION", "phase": "STAGE_B_COMPLETE", "optimizer_steps": 200})
        elif args.phase == "stage_c":
            if args.visual_observations is None: raise ValueError("Stage C requires actual Stage-A/B visual observations")
            visual = _load_visual_observations(args.visual_observations, ("stage_a_status", "stage_b_status"))
            atomic_json(output_dir / "visual_acceptance/stage_b_visual_acceptance.json", visual)
            atomic_json(output_dir / "RUN_STATUS.json", {"task_id": TASK_ID, "status": "RUNNING", "phase": "STAGE_C", "optimizer_steps": 200})
            result = run_stage_c(context, visual)
            atomic_json(output_dir / "RUN_STATUS.json", {"task_id": TASK_ID, "status": "AWAITING_FINAL_VISUAL_INSPECTION" if result["status"].startswith("NUMERIC_PASS") else "AWAITING_FINAL_ADJUDICATION", "phase": "STAGE_C_COMPLETE", "optimizer_steps": 200 + int(config["stage_c"]["max_steps"])})
        elif args.phase == "finalize":
            if args.visual_observations is None: raise ValueError("finalize requires actual visual observations")
            finalize(context, args.visual_observations)
        else:
            raise ValueError(args.phase)
    except Exception as error:
        atomic_json(output_dir / "RUN_STATUS.json", {
            "task_id": TASK_ID, "status": "FAILED_TOOL_OR_RUNTIME", "phase": args.phase,
            "exception_type": type(error).__name__, "exception": str(error), "traceback": traceback.format_exc(),
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the preregistered explicit Gaussian residual-basis ladder")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("stage_a", "stage_b", "stage_c", "finalize"), required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--visual-observations", type=Path, default=None)
    parser.add_argument("--resume-acceptance-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
