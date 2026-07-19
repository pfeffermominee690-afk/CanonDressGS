from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_dressable as training  # noqa: E402
from scene.clothing_gate_bundle import ClothingGateBundle  # noqa: E402
from scene.full_dressable_dataset import DUAL_TARGET_FIELDS, FullDressableTrainingDataset  # noqa: E402
from scene.gaussian_clothing_residuals import (  # noqa: E402
    CHANNELS,
    GaussianClothingResiduals,
    apply_protected_full_residual_guard,
)
from scene.image_conditioned_failure_diagnostics import (  # noqa: E402
    COUNTERFACTUAL_CHANNELS,
    DiagnosticOutfitLatents,
    assert_forward_boundary,
    channel_comparison,
    decision_case,
    normalized_residual_regression_loss,
    reference_variant,
    residual_distance_scalar,
    select_residual_channels,
)
from scene.representation_capacity_oracle import UnboundedGaussianDeltaField  # noqa: E402
from tools import run_image_conditioned_overfit_o01 as o01  # noqa: E402
from tools.check_real_image_conditioned_one_batch import save_render_tensor  # noqa: E402
from tools.run_module4b_canonical_oracle_micropilot import (  # noqa: E402
    _base_gradient_count,
    _base_named_tensors,
    _tensor_state_fingerprint,
)
from utils.mmlphuman_anchor_deformation import MMLPHumanAnchorDeformationAdapter  # noqa: E402


SCHEMA = "canondressgs.image_conditioned_failure_diagnosis.v1"
TASK_ID = "SUBJECT02-IMAGE-CONDITIONED-FAILURE-DIAGNOSIS-001"
EXPECTED_SOURCE_HEAD = "2024a895d3d13e5004ebfeaf9615034fb3430b7c"
EXPECTED_BRANCH = "research/image-conditioned-failure-diagnosis-20260720"
OUTFITS = ("O01", "O08")
CONDITIONS = o01.CONDITIONS
VIEWS = o01.VIEWS
DIRECTORIES = (
    "contract", "input_audit", "current_checkpoint_static_audit", "reference_sensitivity",
    "oracle_residual_comparison", "head_counterfactuals", "probe_D_decoder_capacity",
    "probe_C_reference_conditioning", "probe_R_render_objective_drift",
    "visual_acceptance", "final_adjudication",
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True, default=str) + "\n")
        handle.flush()


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA or config.get("task_id") != TASK_ID:
        raise ValueError("not the preregistered O01 failure diagnosis contract")
    if config.get("branch") != EXPECTED_BRANCH or config.get("source_head") != EXPECTED_SOURCE_HEAD:
        raise ValueError("diagnosis branch/source contract changed")
    if tuple(config.get("outfits", ())) != OUTFITS or tuple(config.get("conditions", ())) != CONDITIONS:
        raise ValueError("two-outfit/four-view protocol changed")
    if any(bool(value) for value in config["permissions"].values()):
        raise ValueError("diagnosis contract enabled a forbidden mutation")
    if int(config["probe_d"]["max_steps"]) > 300 or int(config["probe_c"]["max_steps"]) > 500:
        raise ValueError("diagnostic probe step ceiling changed")
    if int(config["probe_r"]["max_steps"]) > 200:
        raise ValueError("render-objective drift probe exceeds 200 steps")
    return config


def _generic_outfit_data(manifest_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    dataset = FullDressableTrainingDataset(manifest_path, "train", reference_count=3, seed=0)
    if dataset.manifest.get("schema_version") != "canondressgs.full_dataset.v1":
        raise ValueError("source manifest is not full dataset v1")
    if dataset.supervision_mode != "dual_target_region_aware_v1":
        raise ValueError("diagnosis requires dual_target_region_aware_v1")
    outfits = {item["outfit_id"]: item for item in dataset.outfits if item["outfit_id"] in OUTFITS}
    if set(outfits) != set(OUTFITS):
        raise ValueError("source manifest lacks O01 or O08")
    samples: dict[str, dict[str, Any]] = {}
    episodes: dict[str, dict[str, Any]] = {}
    protocol: dict[str, Any] = {"outfits": list(OUTFITS), "conditions": list(CONDITIONS), "episodes": {}}
    for outfit_id in OUTFITS:
        outfit = outfits[outfit_id]
        observations = {item["condition_id"]: item for item in outfit["observations"]}
        if set(CONDITIONS).difference(observations):
            raise ValueError(f"{outfit_id} lacks a fixed four-view condition")
        for condition in CONDITIONS:
            observation = observations[condition]
            missing = DUAL_TARGET_FIELDS.difference(observation)
            if missing:
                raise ValueError(f"{outfit_id}/{condition} lacks dual-target fields: {sorted(missing)}")
            target = dataset._observation(outfit, observation, True)
            sample: dict[str, Any] = {
                "target_edit_rgb": dataset._image(observation["target_edit_rgb"], 3),
                "target_base_rgb": dataset._image(observation["target_base_rgb"], 3),
                "target_foreground_mask": target["foreground_mask"],
                "target_clothing_mask": target["clothing_mask"],
                "target_pose": target["pose"], "target_Rh": target["R_global"], "target_Th": target["Th"],
                "target_camera": {"K": target["K"], "w2c": target["w2c"], "width": target["width"], "height": target["height"]},
                "target_condition_id": condition, "outfit_id": outfit_id, "source_record": observation,
            }
            for name in sorted(DUAL_TARGET_FIELDS.difference({"target_edit_rgb", "target_base_rgb"})):
                sample[name] = dataset._image(observation[name], 1)
            reference_ids = tuple(value for value in CONDITIONS if value != condition)
            references = [dataset._observation(outfit, observations[value], True) for value in reference_ids]
            episode = dataset._stack_references(references)
            episode["reference_cloth_masks"] = episode.pop("reference_clothing_masks")
            if condition in episode["reference_condition_ids"] or len(set(episode["reference_condition_ids"])) != 3:
                raise AssertionError("reference-target overlap in diagnosis episode")
            key = f"{outfit_id}/{condition}"
            samples[key], episodes[key] = sample, episode
            protocol["episodes"][key] = {
                "outfit_id": outfit_id, "target": condition, "target_view": VIEWS[condition],
                "references": list(reference_ids), "target_in_references": False,
            }
    return samples, episodes, protocol


def _to_device_nested(value: Any, device: torch.device) -> Any:
    return o01._to_device(value, device)


def _load_oracle(base: Any, path: Path, device: torch.device) -> tuple[UnboundedGaussianDeltaField, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("global_step", -1)) != 1200 or checkpoint.get("target_tensors_stored") is not False:
        raise ValueError(f"unexpected Rung-2 checkpoint contract: {path}")
    oracle = UnboundedGaussianDeltaField(base).to(device)
    oracle.load_state_dict(checkpoint["model"], strict=True)
    oracle.eval()
    return oracle, {
        "path": str(path), "global_step": int(checkpoint["global_step"]),
        "target_tensors_stored": bool(checkpoint["target_tensors_stored"]),
        "metadata": checkpoint.get("metadata", {}),
    }


def _residual_cpu(residual: GaussianClothingResiduals) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in residual.as_dict().items()}


def _snapshot(output: Mapping[str, Any], rgb: torch.Tensor, alpha: torch.Tensor) -> dict[str, Any]:
    completion = output["completion"]
    return {
        "global_clothing_embedding": output["global_clothing_embedding"].detach().cpu(),
        "observed_surface_feature": output["observed"]["observed_surface_feature"].detach().cpu(),
        "observed_clothing_feature": output["observed"]["observed_clothing_feature"].detach().cpu(),
        "observed_probability": completion.observed_clothing_probability.detach().cpu(),
        "observation_coverage": completion.observation_coverage.detach().cpu(),
        "completed_anchor_features": completion.completed_anchor_features.detach().cpu(),
        "geometry_gate": completion.geometry_gate.detach().cpu(),
        "appearance_gate": completion.appearance_gate.detach().cpu(),
        "confidence": completion.confidence.detach().cpu(),
        "raw_anchor_residuals": {name: value.detach().cpu() for name, value in output["raw_anchor_residuals"].as_dict().items()},
        "bounded_anchor_residuals": {name: value.detach().cpu() for name, value in output["bounded_anchor_residuals"].as_dict().items()},
        "gated_anchor_residuals": {name: value.detach().cpu() for name, value in output["gated_anchor_residuals"].as_dict().items()},
        "gaussian_residuals": _residual_cpu(output["gaussian_residuals"]),
        "rendered_rgb": rgb.detach().cpu(), "rendered_alpha": alpha.detach().cpu(),
    }


def _tensor_distance(first: torch.Tensor, second: torch.Tensor) -> dict[str, float]:
    first, second = first.float(), second.float()
    difference = (first - second).abs()
    denominator = 0.5 * (first.square().mean().sqrt() + second.square().mean().sqrt())
    return {
        "mae": float(difference.mean()), "max_abs": float(difference.max()),
        "normalized_l2": float((first - second).square().mean().sqrt() / denominator.clamp_min(1e-12)),
        "cosine_similarity": float(torch.nn.functional.cosine_similarity(first.reshape(1, -1), second.reshape(1, -1), eps=1e-12)),
    }


def _bundle_from_cpu(values: Mapping[str, torch.Tensor], device: torch.device) -> GaussianClothingResiduals:
    return GaussianClothingResiduals.from_dict({name: value.to(device) for name, value in values.items()})


def _sensitivity_distance(first: Mapping[str, Any], second: Mapping[str, Any], bounds: Mapping[str, float]) -> dict[str, Any]:
    residual = channel_comparison(
        _bundle_from_cpu(first["gaussian_residuals"], torch.device("cpu")),
        _bundle_from_cpu(second["gaussian_residuals"], torch.device("cpu")), bounds,
    )
    return {
        "global_embedding": _tensor_distance(first["global_clothing_embedding"], second["global_clothing_embedding"]),
        "observed_feature": _tensor_distance(first["observed_clothing_feature"], second["observed_clothing_feature"]),
        "completed_feature": _tensor_distance(first["completed_anchor_features"], second["completed_anchor_features"]),
        "geometry_gate": _tensor_distance(first["geometry_gate"], second["geometry_gate"]),
        "appearance_gate": _tensor_distance(first["appearance_gate"], second["appearance_gate"]),
        "residual": residual,
        "residual_bound_normalized_rmse": residual_distance_scalar(residual),
        "rendered_rgb_mae": float((first["rendered_rgb"] - second["rendered_rgb"]).abs().mean()),
        "rendered_alpha_mae": float((first["rendered_alpha"] - second["rendered_alpha"]).abs().mean()),
    }


def _region_denominators(regions: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    values = {}
    for name, mask in regions.items():
        count = int((mask >= 0.5).sum())
        values[name] = {"mask_pixel_count": count, "rgb_element_denominator": count * 3, "alpha_element_denominator": count}
    return values


def _garment_mask(sample: Mapping[str, Any]) -> torch.Tensor:
    mask = sample["target_clothing_mask"]
    for name in ("target_edit_mask", "target_old_clothing_mask", "target_transition_mask"):
        mask = torch.maximum(mask, sample[name])
    return mask


def _masked_mae(first: torch.Tensor, second: torch.Tensor, mask: torch.Tensor) -> float:
    return o01._masked_mae(first, second, mask)


def _row_concentration(residual: GaussianClothingResiduals, mask: torch.Tensor) -> dict[str, float]:
    mask = mask.reshape(-1).bool()
    result = {}
    for name, value in residual.as_dict().items():
        absolute = value.detach().abs().reshape(value.shape[0], -1).sum(1)
        result[name] = float(absolute[mask].sum() / absolute.sum().clamp_min(1e-12))
    return result


def _gaussian_to_anchor_score(model: Any, residual: GaussianClothingResiduals) -> torch.Tensor:
    values = []
    for value in residual.as_dict().values():
        values.append(value.detach().float().reshape(value.shape[0], -1).norm(dim=1))
    gaussian_score = torch.stack(values).mean(0)
    indices = model.dressable_model.gaussian_anchor_indices
    weights = model.dressable_model.gaussian_anchor_weights
    anchor_count = int(model.dressable_model.anchor_features.shape[0])
    numerator = torch.zeros(anchor_count, device=gaussian_score.device)
    denominator = torch.zeros(anchor_count, device=gaussian_score.device)
    numerator.scatter_add_(0, indices.reshape(-1), (gaussian_score[:, None] * weights).reshape(-1))
    denominator.scatter_add_(0, indices.reshape(-1), weights.reshape(-1))
    return numerator / denominator.clamp_min(1e-12)


def _connected_components(score: torch.Tensor, graph: torch.Tensor) -> dict[str, Any]:
    score, graph = score.detach().cpu(), graph.detach().cpu()
    positive = score > 1e-8
    threshold = torch.quantile(score[positive], 0.50) if positive.any() else torch.tensor(float("inf"))
    active = score >= threshold
    seen = torch.zeros_like(active)
    sizes: list[int] = []
    for start in torch.nonzero(active, as_tuple=False).reshape(-1).tolist():
        if seen[start]:
            continue
        queue, size = deque([start]), 0
        seen[start] = True
        while queue:
            node = queue.popleft(); size += 1
            for neighbor in graph[node].tolist():
                if active[neighbor] and not seen[neighbor]:
                    seen[neighbor] = True; queue.append(int(neighbor))
        sizes.append(size)
    sizes.sort(reverse=True)
    return {
        "threshold": float(threshold) if torch.isfinite(threshold) else None,
        "active_anchor_count": int(active.sum()), "component_count": len(sizes),
        "largest_component": sizes[0] if sizes else 0,
        "largest_component_fraction": 0.0 if not sizes else sizes[0] / max(int(active.sum()), 1),
        "top_component_sizes": sizes[:20],
    }


def _set_requires_grad(module: torch.nn.Module, value: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(value)


def _trainable_group_snapshot(groups: Mapping[str, list[torch.nn.Parameter]]) -> dict[str, Any]:
    result = {}
    for name, parameters in groups.items():
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        result[name] = {
            "parameter_count": sum(parameter.numel() for parameter in parameters),
            "gradient_tensor_count": len(gradients),
            "gradient_finite": all(torch.isfinite(gradient).all().item() for gradient in gradients),
            "gradient_l2": float(torch.sqrt(sum((gradient.detach().float().square().sum() for gradient in gradients), torch.tensor(0.0, device=parameters[0].device)))) if gradients else 0.0,
        }
    return result


def _predict_from_latent(
    model: Any, latent: torch.Tensor, protected_mask: torch.Tensor,
) -> tuple[GaussianClothingResiduals, Any, Any]:
    count = int(model.dressable_model.anchor_features.shape[0])
    reference = model.dressable_model.anchor_features
    ones = torch.ones(count, 1, device=reference.device, dtype=reference.dtype)
    gate = ClothingGateBundle(ones, ones, ones, gate_source="diagnostic_fixed_one_decoder_isolation")
    local = torch.zeros(count, model.multiview_aggregator.output_dim, device=reference.device, dtype=reference.dtype)
    raw, bounded, gated = model.dressable_model.compute_film_anchor_residuals(
        gate, clothing_embedding=latent.to(reference), anchor_clothing_features=local,
    )
    gaussian = model.dressable_model.interpolate_anchor_clothing_residuals(gated)
    gaussian = apply_protected_full_residual_guard(gaussian, protected_mask)
    return gaussian, raw, bounded


def _mean_head_saturation(residuals: GaussianClothingResiduals, bounds: Mapping[str, float]) -> float:
    stats = o01._residual_statistics(residuals, bounds)
    return float(np.mean([stats[name]["bound_saturation_ratio"] for name in CHANNELS]))


def _oracle_regression_average(
    model: Any, episodes: Mapping[str, Any], geometries: Mapping[str, Any], protected_mask: torch.Tensor,
    targets: Mapping[str, GaussianClothingResiduals], bounds: Mapping[str, float],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    losses, rows = [], []
    for outfit_id in OUTFITS:
        for condition in CONDITIONS:
            key = f"{outfit_id}/{condition}"
            inputs = o01._forward_inputs(episodes[key], geometries[condition])
            assert_forward_boundary(inputs)
            output = model.compute_online_six_channel_residuals(protected_gaussian_mask=protected_mask, **inputs)
            loss, parts = normalized_residual_regression_loss(output["gaussian_residuals"], targets[outfit_id], bounds)
            losses.append(loss)
            rows.append({"key": key, "loss": float(loss.detach()), "parts": {name: float(value.detach()) for name, value in parts.items()}})
    return torch.stack(losses).mean(), rows


def _static_and_sensitivity(
    model: Any, base: Any, samples: Mapping[str, Any], episodes: Mapping[str, Any], geometries: Mapping[str, Any],
    protected_mask: torch.Tensor, background: torch.Tensor, config: Mapping[str, Any], output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, tuple[Any, Any, Any, Any, Any]]]:
    bounds = o01._bounds(config)
    static_dir = output_dir / "current_checkpoint_static_audit"
    static_rows, static_results, cached = [], {}, {}
    model.eval()
    with torch.no_grad():
        for condition in CONDITIONS:
            key = f"O01/{condition}"
            result = o01._loss_and_output(model, base, samples[key], episodes[key], geometries[condition], protected_mask, background, config)
            objective, output, rgb, alpha, extras = result
            metrics = o01._evaluation_metrics(samples[key], objective, output, rgb, alpha, bounds, protected_mask)
            static_results[condition] = {
                "view": VIEWS[condition], "loss": float(objective.total), "metrics": metrics,
                "loss_parts": {name: float(value) for name, value in objective.parts.items()},
                "metric_denominators": _region_denominators(extras["regions"]),
                "visibility": {
                    "per_view_visible_anchor_counts": [int(value) for value in (output["projection"]["per_view_visibility"] > 0).sum(dim=1).reshape(-1).cpu()],
                    "observed_anchor_count": int((output["completion"].observation_coverage > 0).sum()),
                    "observed_clothing_probability_mean": float(output["completion"].observed_clothing_probability.mean()),
                },
            }
            torch.save(_snapshot(output, rgb, alpha), static_dir / f"{condition}_snapshot.pt")
            save_render_tensor(static_dir / f"{condition}_rgb.png", rgb, 3)
            save_render_tensor(static_dir / f"{condition}_alpha.png", alpha, 1)
            static_rows.append((condition, [
                ("base", samples[key]["target_base_rgb"], 3), ("target", samples[key]["target_edit_rgb"], 3),
                ("prediction", rgb, 3), ("abs error", (rgb - samples[key]["target_edit_rgb"]).abs(), 3),
                ("alpha", alpha, 1),
            ]))
            cached[condition] = result
    o01._save_contact_sheet(static_dir / "static_four_view_contact_sheet.png", static_rows)
    left = static_results["cond_000017"]
    static_summary = {
        "checkpoint_step": 2000, "four_view": static_results,
        "left_view_failure_is_real": bool(left["metrics"]["edit_reduction"] < 0.10),
        "left_view_metric_denominator_is_explicit": True,
        "left_view_metric_denominators": left["metric_denominators"],
        "mean": {name: float(np.mean([item["metrics"][name] for item in static_results.values()])) for name in (
            "edit_reduction", "target_closer_fraction", "protected_rgb_mae", "background_rgb_mae",
        )},
    }
    _atomic_json(static_dir / "static_audit.json", static_summary)

    sensitivity_dir = output_dir / "reference_sensitivity"
    target_condition = config["diagnosis"]["fixed_sensitivity_target"]
    target_key = f"O01/{target_condition}"
    base_episode, base_geometry = episodes[target_key], geometries[target_condition]
    base_images = torch.stack([
        samples[f"O01/{condition}"]["target_base_rgb"] for condition in base_episode["reference_condition_ids"]
    ])
    cases: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    modes = {
        "R0_correct_O01": "correct", "R1_permuted": "permuted",
        "R2_same_reference_tripled": "same_reference_tripled", "R3_single_reference": "single_reference",
        "R4_zero_rgb_masks_kept": "zero_rgb_masks_kept", "R5_base_rgb_substitution": "base_rgb_substitution",
        "R7_color_perturbation": "color_perturbation",
    }
    for name, mode in modes.items():
        cases[name] = reference_variant(base_episode, base_geometry, mode, substitute_images=base_images if mode == "base_rgb_substitution" else None)
    cases["R6_O08_references"] = (episodes[f"O08/{target_condition}"], base_geometry)

    snapshots, rows = {}, []
    with torch.no_grad():
        for case_name, (episode, geometry) in cases.items():
            inputs = o01._forward_inputs(episode, geometry)
            assert_forward_boundary(inputs)
            output = model.compute_online_six_channel_residuals(protected_gaussian_mask=protected_mask, **inputs)
            residuals = output["gaussian_residuals"]
            overrides = model.dressable_model.compose_canonical_gaussian_overrides(residuals, CHANNELS)
            rgb, alpha = o01._render_sh1(base, samples[target_key], overrides, background)
            snap = _snapshot(output, rgb, alpha)
            snapshots[case_name] = snap
            torch.save(snap, sensitivity_dir / f"{case_name}.pt")
            save_render_tensor(sensitivity_dir / f"{case_name}_rgb.png", rgb, 3)
            rows.append((case_name, [("prediction", rgb, 3), ("target", samples[target_key]["target_edit_rgb"], 3), ("alpha", alpha, 1)]))
    o01._save_contact_sheet(sensitivity_dir / "reference_sensitivity_contact_sheet.png", rows)
    baseline = snapshots["R0_correct_O01"]
    comparisons = {name: _sensitivity_distance(baseline, value, bounds) for name, value in snapshots.items() if name != "R0_correct_O01"}
    threshold = config["diagnosis"]["sensitivity_reference_ignored_thresholds"]
    critical = [comparisons[name] for name in ("R4_zero_rgb_masks_kept", "R5_base_rgb_substitution", "R6_O08_references")]
    reference_ignored = all(
        item["residual_bound_normalized_rmse"] <= float(threshold["residual_bound_normalized_rmse_max"])
        and item["rendered_rgb_mae"] <= float(threshold["render_rgb_mae_max"])
        for item in critical
    )
    features_changed_but_residual_same = any(
        item["completed_feature"]["normalized_l2"] > 1e-4 and item["residual_bound_normalized_rmse"] <= float(threshold["residual_bound_normalized_rmse_max"])
        for item in critical
    )
    sensitivity = {
        "fixed_target_condition": target_condition, "fixed_target_view": VIEWS[target_condition],
        "target_pose_camera_changed": False, "target_images_entered_forward": False,
        "cases": {name: {"reference_ids": list(cases[name][0]["reference_condition_ids"])} for name in cases},
        "comparisons_to_R0": comparisons, "reference_ignored": reference_ignored,
        "conditioning_decoder_collapse": features_changed_but_residual_same,
    }
    _atomic_json(sensitivity_dir / "reference_sensitivity.json", sensitivity)
    return static_summary, sensitivity, cached


def _oracle_and_counterfactuals(
    model: Any, base: Any, samples: Mapping[str, Any], episodes: Mapping[str, Any], geometries: Mapping[str, Any],
    protected_mask: torch.Tensor, background: torch.Tensor, config: Mapping[str, Any], output_dir: Path,
    cached: Mapping[str, tuple[Any, Any, Any, Any, Any]], oracle_targets: Mapping[str, GaussianClothingResiduals],
    oracle_modules: Mapping[str, UnboundedGaussianDeltaField],
) -> tuple[dict[str, Any], dict[str, Any]]:
    bounds = o01._bounds(config)
    oracle_dir = output_dir / "oracle_residual_comparison"
    per_view, rows = {}, []
    support = oracle_modules["O01"].trainable_support[:, 0]
    for condition in CONDITIONS:
        objective, output, rgb, alpha, _ = cached[condition]
        network = output["gaussian_residuals"]
        oracle = oracle_targets["O01"]
        comparison = channel_comparison(network, oracle, bounds)
        oracle_overrides = model.dressable_model.compose_canonical_gaussian_overrides(oracle, CHANNELS)
        oracle_rgb, oracle_alpha = o01._render_sh1(base, samples[f"O01/{condition}"], oracle_overrides, background)
        per_view[condition] = {
            "view": VIEWS[condition], "channels": comparison,
            "mean_bound_normalized_rmse": residual_distance_scalar(comparison),
            "network_garment_support_concentration": _row_concentration(network, support),
            "oracle_garment_support_concentration": _row_concentration(oracle, support),
            "network_protected_concentration": _row_concentration(network, protected_mask),
            "oracle_protected_concentration": _row_concentration(oracle, protected_mask),
            "render_network_to_oracle": {
                "garment_rgb_mae": _masked_mae(rgb, oracle_rgb, _garment_mask(samples[f"O01/{condition}"])),
                "alpha_mae": float((alpha - oracle_alpha).abs().mean()),
            },
        }
        save_render_tensor(oracle_dir / f"{condition}_oracle_rgb.png", oracle_rgb, 3)
        rows.append((condition, [
            ("base", samples[f"O01/{condition}"]["target_base_rgb"], 3),
            ("target", samples[f"O01/{condition}"]["target_edit_rgb"], 3),
            ("network", rgb, 3), ("oracle", oracle_rgb, 3),
            ("network error", (rgb - samples[f"O01/{condition}"]["target_edit_rgb"]).abs(), 3),
            ("oracle error", (oracle_rgb - samples[f"O01/{condition}"]["target_edit_rgb"]).abs(), 3),
        ]))
    o01._save_contact_sheet(oracle_dir / "network_oracle_four_view_contact_sheet.png", rows)
    network_score = _gaussian_to_anchor_score(model, cached[CONDITIONS[0]][1]["gaussian_residuals"])
    oracle_score = _gaussian_to_anchor_score(model, oracle_targets["O01"])
    oracle_report = {
        "oracle_usage": "diagnostic_loss_and_offline_evaluation_only",
        "oracle_never_entered_formal_prediction_forward": True,
        "views": per_view,
        "mean_bound_normalized_rmse": float(np.mean([item["mean_bound_normalized_rmse"] for item in per_view.values()])),
        "network_connectedness": _connected_components(network_score, model.anchor_graph_indices),
        "oracle_connectedness": _connected_components(oracle_score, model.anchor_graph_indices),
        "interpretation": {
            "direction_similarity": float(np.mean([np.mean([channel["cosine_similarity"] for channel in item["channels"].values()]) for item in per_view.values()])),
            "amplitude_and_location_reported_per_channel": True,
            "scale_sh_opacity_reported_separately": True,
            "support_overlap_and_connectedness_reported": True,
        },
    }
    _atomic_json(oracle_dir / "oracle_residual_comparison.json", oracle_report)

    counter_dir = output_dir / "head_counterfactuals"
    counter_report, counter_rows = {}, []
    low, high = (float(value) for value in config["diagnosis"]["nonsaturated_gate_range"])
    for condition in CONDITIONS:
        objective, output, _, _, _ = cached[condition]
        sample = samples[f"O01/{condition}"]
        variants: dict[str, GaussianClothingResiduals] = {
            name: select_residual_channels(output["gaussian_residuals"], selected)
            for name, selected in COUNTERFACTUAL_CHANNELS.items()
        }
        completion = output["completion"]
        gate = ClothingGateBundle(
            completion.geometry_gate.clamp(low, high), completion.appearance_gate.clamp(low, high),
            completion.confidence, gate_source="diagnostic_clamped_nonsaturated_gate",
        )
        gated_anchor = gate.apply(output["bounded_anchor_residuals"])
        variants["H7_nonsaturated_gates"] = apply_protected_full_residual_guard(
            model.dressable_model.interpolate_anchor_clothing_residuals(gated_anchor), protected_mask,
        )
        metrics, images = {}, []
        for name in config["diagnosis"]["head_counterfactuals"]:
            residual = variants[name]
            overrides = model.dressable_model.compose_canonical_gaussian_overrides(residual, CHANNELS)
            rgb, alpha = o01._render_sh1(base, sample, overrides, background)
            metrics[name] = {
                "target_garment_rgb_mae": _masked_mae(rgb, sample["target_edit_rgb"], _garment_mask(sample)),
                "base_garment_rgb_mae": _masked_mae(rgb, sample["target_base_rgb"], _garment_mask(sample)),
                "protected_rgb_mae": _masked_mae(rgb, sample["target_base_rgb"], objective.regions["protected_identity"]),
                "background_rgb_mae": _masked_mae(rgb, sample["target_base_rgb"], objective.regions["background"]),
                "alpha_target_mae": float((alpha - sample["target_foreground_mask"]).abs().mean()),
                "residual_statistics": o01._residual_statistics(residual, bounds),
            }
            images.append((name, rgb, 3))
        counter_report[condition] = {"view": VIEWS[condition], "counterfactuals": metrics}
        counter_rows.append((condition, images))
    o01._save_contact_sheet(counter_dir / "head_counterfactual_contact_sheet.png", counter_rows)
    _atomic_json(counter_dir / "head_counterfactuals.json", counter_report)
    return oracle_report, counter_report


def _probe_d(
    model: Any, base: Any, samples: Mapping[str, Any], protected_mask: torch.Tensor, background: torch.Tensor,
    config: Mapping[str, Any], output_dir: Path, targets: Mapping[str, GaussianClothingResiduals],
) -> dict[str, Any]:
    directory = output_dir / "probe_D_decoder_capacity"
    contract = config["probe_d"]
    bounds = o01._bounds(config)
    _set_requires_grad(model, False)
    latent = DiagnosticOutfitLatents(OUTFITS, int(config["model"]["embedding_dim"]), int(config["diagnosis"]["seed"])).to(base._xyz)
    mlp = model.dressable_model.anchor_clothing_mlp
    trainable_modules = {
        "diagnostic_latents": latent,
        "film_hypernetwork": model.dressable_model.clothing_film_generator,
        "shared_decoder": mlp.hidden_layers,
        "head_xyz": mlp.output_layer, "head_scaling": mlp.scaling_head,
        "head_rotation": mlp.rotation_head, "head_opacity": mlp.opacity_head,
        "head_sh0": mlp.sh0_head, "head_shN": mlp.shN_head,
    }
    for module in trainable_modules.values():
        _set_requires_grad(module, True)
    latent_parameters = list(latent.parameters())
    decoder_parameters = [parameter for name, module in trainable_modules.items() if name != "diagnostic_latents" for parameter in module.parameters()]
    optimizer = torch.optim.Adam([
        {"name": "diagnostic_latents", "params": latent_parameters, "lr": float(contract["latent_learning_rate"])},
        {"name": "decoder", "params": decoder_parameters, "lr": float(contract["decoder_learning_rate"])},
    ])
    groups = {name: list(module.parameters()) for name, module in trainable_modules.items()}
    initial_predictions = {outfit: _predict_from_latent(model, latent(outfit), protected_mask)[0] for outfit in OUTFITS}
    initial_losses = {}
    for outfit in OUTFITS:
        loss, _ = normalized_residual_regression_loss(initial_predictions[outfit], targets[outfit], bounds)
        initial_losses[outfit] = float(loss.detach())
    history, final_gradients = [], {}
    for step in range(1, int(contract["max_steps"]) + 1):
        outfit = OUTFITS[(step - 1) % len(OUTFITS)]
        optimizer.zero_grad(set_to_none=True)
        prediction, _, _ = _predict_from_latent(model, latent(outfit), protected_mask)
        loss, parts = normalized_residual_regression_loss(prediction, targets[outfit], bounds)
        loss.backward()
        final_gradients = _trainable_group_snapshot(groups)
        if not all(value["gradient_finite"] for value in final_gradients.values()):
            raise FloatingPointError("Probe D produced non-finite gradients")
        torch.nn.utils.clip_grad_norm_(latent_parameters + decoder_parameters, float(contract["gradient_clip_norm"]), error_if_nonfinite=True)
        optimizer.step()
        row = {"step": step, "outfit": outfit, "loss": float(loss.detach()), **{f"loss_{name}": float(value.detach()) for name, value in parts.items()}}
        history.append(row); _append_jsonl(directory / "training.jsonl", row)
        if step == 1 or step % 25 == 0:
            _atomic_json(directory / "partial_status.json", {"status": "RUNNING", "completed_step": step, "last": row})
    predictions, final_losses = {}, {}
    with torch.no_grad():
        for outfit in OUTFITS:
            predictions[outfit] = _predict_from_latent(model, latent(outfit), protected_mask)[0]
            loss, _ = normalized_residual_regression_loss(predictions[outfit], targets[outfit], bounds)
            final_losses[outfit] = float(loss)
    comparisons = {outfit: channel_comparison(predictions[outfit], targets[outfit], bounds) for outfit in OUTFITS}
    predicted_separation = residual_distance_scalar(channel_comparison(predictions["O01"], predictions["O08"], bounds))
    oracle_separation = residual_distance_scalar(channel_comparison(targets["O01"], targets["O08"], bounds))
    separation_ratio = predicted_separation / max(oracle_separation, 1e-12)
    render_rows, render_metrics = [], []
    with torch.no_grad():
        for outfit in OUTFITS:
            prediction_overrides = model.dressable_model.compose_canonical_gaussian_overrides(predictions[outfit], CHANNELS)
            oracle_overrides = model.dressable_model.compose_canonical_gaussian_overrides(targets[outfit], CHANNELS)
            for condition in CONDITIONS:
                sample = samples[f"{outfit}/{condition}"]
                predicted_rgb, predicted_alpha = o01._render_sh1(base, sample, prediction_overrides, background)
                oracle_rgb, oracle_alpha = o01._render_sh1(base, sample, oracle_overrides, background)
                metric = {
                    "outfit": outfit, "condition": condition, "view": VIEWS[condition],
                    "garment_rgb_mae_to_oracle": _masked_mae(predicted_rgb, oracle_rgb, _garment_mask(sample)),
                    "alpha_mae_to_oracle": float((predicted_alpha - oracle_alpha).abs().mean()),
                }
                render_metrics.append(metric)
                render_rows.append((f"{outfit}/{condition}", [("target", sample["target_edit_rgb"], 3), ("Probe D", predicted_rgb, 3), ("Oracle", oracle_rgb, 3)]))
    o01._save_contact_sheet(directory / "probe_d_oracle_render_contact_sheet.png", render_rows)
    initial_mean, final_mean = float(np.mean(list(initial_losses.values()))), float(np.mean(list(final_losses.values())))
    loss_drop = (initial_mean - final_mean) / max(initial_mean, 1e-12)
    mean_rmse = float(np.mean([residual_distance_scalar(value) for value in comparisons.values()]))
    mean_render = float(np.mean([value["garment_rgb_mae_to_oracle"] for value in render_metrics]))
    mean_saturation = float(np.mean([_mean_head_saturation(predictions[outfit], bounds) for outfit in OUTFITS]))
    thresholds = contract["acceptance"]
    checks = {
        "regression_loss_drop": loss_drop >= float(thresholds["regression_loss_drop_fraction_min"]),
        "final_bound_normalized_rmse": mean_rmse <= float(thresholds["final_bound_normalized_rmse_max"]),
        "outfit_separation": separation_ratio >= float(thresholds["outfit_separation_ratio_min"]),
        "oracle_render_recovery": mean_render <= float(thresholds["mean_oracle_render_garment_rgb_mae_max"]),
        "no_excessive_head_saturation": mean_saturation <= float(thresholds["mean_head_bound_hit_ratio_max"]),
        "all_gradients_finite_nonzero": all(value["gradient_finite"] and value["gradient_l2"] > 0 for value in final_gradients.values()),
        "base_gradient_zero": _base_gradient_count(base) == 0,
        "target_images_entered_forward": False,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL", "steps": int(contract["max_steps"]),
        "probe_scope": "diagnostic independent outfit latent plus existing FiLM/shared decoder/residual heads",
        "formal_inference_candidate": False, "gate_mode": contract["gate_mode"],
        "initial_loss": initial_losses, "final_loss": final_losses, "loss_drop_fraction": loss_drop,
        "comparisons": comparisons, "mean_bound_normalized_rmse": mean_rmse,
        "predicted_outfit_separation": predicted_separation, "oracle_outfit_separation": oracle_separation,
        "outfit_separation_ratio": separation_ratio, "render_metrics": render_metrics,
        "mean_oracle_render_garment_rgb_mae": mean_render, "mean_head_bound_hit_ratio": mean_saturation,
        "gradients": final_gradients, "checks": checks,
    }
    torch.save({"model": model.state_dict(), "diagnostic_latents": latent.state_dict(), "optimizer": optimizer.state_dict(), "step": int(contract["max_steps"]), "formal_inference_candidate": False}, directory / "checkpoint_step_000300.pth")
    _atomic_json(directory / "probe_d_metrics.json", report)
    return report


def _probe_c(
    model: Any, optimizer: Any, base: Any, samples: Mapping[str, Any], episodes: Mapping[str, Any], geometries: Mapping[str, Any],
    protected_mask: torch.Tensor, background: torch.Tensor, config: Mapping[str, Any], output_dir: Path,
    targets: Mapping[str, GaussianClothingResiduals],
) -> dict[str, Any]:
    directory = output_dir / "probe_C_reference_conditioning"
    contract, bounds = config["probe_c"], o01._bounds(config)
    model.train()
    initial_loss, _ = _oracle_regression_average(model, episodes, geometries, protected_mask, targets, bounds)
    history, final_gradient = [], {}
    schedule = [(outfit, condition) for condition in CONDITIONS for outfit in OUTFITS]
    for step in range(1, int(contract["max_steps"]) + 1):
        outfit, condition = schedule[(step - 1) % len(schedule)]
        key = f"{outfit}/{condition}"
        optimizer.zero_grad(set_to_none=True)
        inputs = o01._forward_inputs(episodes[key], geometries[condition]); assert_forward_boundary(inputs)
        output = model.compute_online_six_channel_residuals(protected_gaussian_mask=protected_mask, **inputs)
        loss, parts = normalized_residual_regression_loss(output["gaussian_residuals"], targets[outfit], bounds)
        loss.backward()
        final_gradient = o01._gradient_snapshot(model, base)
        trainable = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        torch.nn.utils.clip_grad_norm_(trainable, float(contract["gradient_clip_norm"]), error_if_nonfinite=True)
        optimizer.step()
        row = {"step": step, "outfit": outfit, "condition": condition, "loss": float(loss.detach()), **{f"loss_{name}": float(value.detach()) for name, value in parts.items()}}
        history.append(row); _append_jsonl(directory / "training.jsonl", row)
        if step == 1 or step % 25 == 0:
            _atomic_json(directory / "partial_status.json", {"status": "RUNNING", "completed_step": step, "last": row})
    model.eval()
    final_loss, final_rows = _oracle_regression_average(model, episodes, geometries, protected_mask, targets, bounds)
    correct_advantages, correct_predictions, render_rows = [], {outfit: [] for outfit in OUTFITS}, []
    other = {"O01": "O08", "O08": "O01"}
    with torch.no_grad():
        for outfit, condition in schedule:
            correct_key, swapped_key = f"{outfit}/{condition}", f"{other[outfit]}/{condition}"
            correct_inputs = o01._forward_inputs(episodes[correct_key], geometries[condition]); assert_forward_boundary(correct_inputs)
            swapped_inputs = o01._forward_inputs(episodes[swapped_key], geometries[condition]); assert_forward_boundary(swapped_inputs)
            correct = model.compute_online_six_channel_residuals(protected_gaussian_mask=protected_mask, **correct_inputs)
            swapped = model.compute_online_six_channel_residuals(protected_gaussian_mask=protected_mask, **swapped_inputs)
            correct_loss, _ = normalized_residual_regression_loss(correct["gaussian_residuals"], targets[outfit], bounds)
            swapped_loss, _ = normalized_residual_regression_loss(swapped["gaussian_residuals"], targets[outfit], bounds)
            correct_advantages.append(float(swapped_loss - correct_loss))
            correct_predictions[outfit].append(correct["gaussian_residuals"])
            overrides = model.dressable_model.compose_canonical_gaussian_overrides(correct["gaussian_residuals"], CHANNELS)
            rgb, _ = o01._render_sh1(base, samples[correct_key], overrides, background)
            render_rows.append((correct_key, [("target", samples[correct_key]["target_edit_rgb"], 3), ("correct refs", rgb, 3)]))
    o01._save_contact_sheet(directory / "probe_c_correct_reference_contact_sheet.png", render_rows)
    mean_predictions = {}
    for outfit in OUTFITS:
        mean_predictions[outfit] = GaussianClothingResiduals.from_dict({
            name: torch.stack([getattr(value, name) for value in correct_predictions[outfit]]).mean(0)
            for name in CHANNELS
        })
    predicted_separation = residual_distance_scalar(channel_comparison(mean_predictions["O01"], mean_predictions["O08"], bounds))
    oracle_separation = residual_distance_scalar(channel_comparison(targets["O01"], targets["O08"], bounds))
    separation_ratio = predicted_separation / max(oracle_separation, 1e-12)
    loss_drop = float((initial_loss - final_loss) / initial_loss.clamp_min(1e-12))
    saturation = float(np.mean([_mean_head_saturation(mean_predictions[outfit], bounds) for outfit in OUTFITS]))
    thresholds = contract["acceptance"]
    checks = {
        "regression_converged": loss_drop >= float(thresholds["regression_loss_drop_fraction_min"]),
        "correct_reference_beats_swapped": float(np.mean(correct_advantages)) >= float(thresholds["correct_reference_oracle_advantage_min"]),
        "outfits_distinguishable": separation_ratio >= float(thresholds["predicted_outfit_separation_ratio_min"]),
        "no_excessive_head_saturation": saturation <= float(thresholds["mean_head_bound_hit_ratio_max"]),
        "balanced_updates": all(sum(row["outfit"] == outfit for row in history) == len(history) // 2 for outfit in OUTFITS),
        "target_images_entered_forward": False, "outfit_id_entered_forward": False,
        "base_gradient_zero": _base_gradient_count(base) == 0,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL", "steps": int(contract["max_steps"]),
        "initial_loss": float(initial_loss), "final_loss": float(final_loss), "loss_drop_fraction": loss_drop,
        "correct_reference_oracle_advantages": correct_advantages,
        "mean_correct_reference_oracle_advantage": float(np.mean(correct_advantages)),
        "predicted_outfit_separation": predicted_separation, "oracle_outfit_separation": oracle_separation,
        "outfit_separation_ratio": separation_ratio, "mean_head_bound_hit_ratio": saturation,
        "final_episode_regression": final_rows, "gradient_final": final_gradient, "checks": checks,
    }
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": int(contract["max_steps"]), "oracle_residual_in_formal_forward": False}, directory / "checkpoint_step_000500.pth")
    _atomic_json(directory / "probe_c_metrics.json", report)
    return report


def _evaluate_image_objective(
    model: Any, base: Any, samples: Mapping[str, Any], episodes: Mapping[str, Any], geometries: Mapping[str, Any],
    protected_mask: torch.Tensor, background: torch.Tensor, config: Mapping[str, Any],
) -> dict[str, Any]:
    rows, residual_saturations = [], []
    model.eval()
    with torch.no_grad():
        for outfit in OUTFITS:
            for condition in CONDITIONS:
                key = f"{outfit}/{condition}"
                objective, output, rgb, alpha, _ = o01._loss_and_output(model, base, samples[key], episodes[key], geometries[condition], protected_mask, background, config)
                metrics = o01._evaluation_metrics(samples[key], objective, output, rgb, alpha, o01._bounds(config), protected_mask)
                rows.append({"key": key, "loss": float(objective.total), **metrics})
                residual_saturations.append(metrics["residuals"]["abnormal_gaussian_fraction"])
    return {
        "rows": rows,
        "mean_edit_reduction": float(np.mean([row["edit_reduction"] for row in rows])),
        "mean_target_closer": float(np.mean([row["target_closer_fraction"] for row in rows])),
        "mean_protected_rgb_mae": float(np.mean([row["protected_rgb_mae"] for row in rows])),
        "mean_background_rgb_mae": float(np.mean([row["background_rgb_mae"] for row in rows])),
        "mean_abnormal_gaussian_fraction": float(np.mean(residual_saturations)),
    }


def _probe_r(
    model: Any, optimizer: Any, base: Any, samples: Mapping[str, Any], episodes: Mapping[str, Any], geometries: Mapping[str, Any],
    protected_mask: torch.Tensor, background: torch.Tensor, config: Mapping[str, Any], output_dir: Path,
    targets: Mapping[str, GaussianClothingResiduals],
) -> dict[str, Any]:
    directory, contract, bounds = output_dir / "probe_R_render_objective_drift", config["probe_r"], o01._bounds(config)
    pre_oracle, _ = _oracle_regression_average(model, episodes, geometries, protected_mask, targets, bounds)
    pre_image = _evaluate_image_objective(model, base, samples, episodes, geometries, protected_mask, background, config)
    history, schedule = [], [(outfit, condition) for condition in CONDITIONS for outfit in OUTFITS]
    model.train()
    for step in range(1, int(contract["max_steps"]) + 1):
        outfit, condition = schedule[(step - 1) % len(schedule)]; key = f"{outfit}/{condition}"
        optimizer.zero_grad(set_to_none=True)
        objective, _, _, _, _ = o01._loss_and_output(model, base, samples[key], episodes[key], geometries[condition], protected_mask, background, config)
        objective.total.backward()
        trainable = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True); optimizer.step()
        row = {"step": step, "outfit": outfit, "condition": condition, "image_objective": float(objective.total.detach())}
        history.append(row); _append_jsonl(directory / "training.jsonl", row)
        if step == 1 or step % 25 == 0:
            _atomic_json(directory / "partial_status.json", {"status": "RUNNING", "completed_step": step, "last": row})
    post_oracle, _ = _oracle_regression_average(model, episodes, geometries, protected_mask, targets, bounds)
    post_image = _evaluate_image_objective(model, base, samples, episodes, geometries, protected_mask, background, config)
    drift = contract["drift"]
    signals = {
        "oracle_regression_drift": float((post_oracle - pre_oracle) / pre_oracle.clamp_min(1e-12)) >= float(drift["oracle_regression_increase_fraction"]),
        "abnormal_gaussian_drift": post_image["mean_abnormal_gaussian_fraction"] - pre_image["mean_abnormal_gaussian_fraction"] >= float(drift["abnormal_gaussian_increase"]),
        "edit_reduction_drift": pre_image["mean_edit_reduction"] - post_image["mean_edit_reduction"] >= float(drift["mean_edit_reduction_drop"]),
        "protected_drift": post_image["mean_protected_rgb_mae"] - pre_image["mean_protected_rgb_mae"] >= float(drift["protected_rgb_mae_increase"]),
    }
    report = {
        "status": "DEGRADED" if any(signals.values()) else "STABLE", "steps": int(contract["max_steps"]),
        "teacher_oracle_loss_enabled": False, "oracle_residual_entered_formal_forward": False,
        "pre_oracle_regression": float(pre_oracle), "post_oracle_regression": float(post_oracle),
        "oracle_regression_change_fraction": float((post_oracle - pre_oracle) / pre_oracle.clamp_min(1e-12)),
        "pre_image_objective_evaluation": pre_image, "post_image_objective_evaluation": post_image,
        "drift_signals": signals,
    }
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": int(contract["max_steps"]), "teacher_oracle_loss_enabled": False}, directory / "checkpoint_step_000200.pth")
    _atomic_json(directory / "probe_r_metrics.json", report)
    return report


def run(args: argparse.Namespace) -> None:
    config_path = args.config.resolve(); config = _load_config(config_path)
    task_root = Path(config["output"]["task_root"])
    output_dir = Path(args.output_dir or task_root / config["output"]["attempt"])
    if output_dir.exists():
        raise FileExistsError(f"append-only diagnosis attempt already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    for name in DIRECTORIES:
        (output_dir / name).mkdir()
    started = time.time()
    status = {"task_id": TASK_ID, "status": "RUNNING", "failure_stage": "contract", "started_at": started, "optimizer_steps": 0}
    _atomic_json(output_dir / "RUN_STATUS.json", status)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("formal failure diagnosis requires CUDA")
        device = torch.device("cuda")
        seed = int(config["diagnosis"]["seed"])
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        commit, branch, dirty = o01.git_output("rev-parse", "HEAD"), o01.git_output("branch", "--show-current"), o01.git_output("status", "--short")
        if branch != EXPECTED_BRANCH or dirty:
            raise RuntimeError(f"formal diagnosis requires clean {EXPECTED_BRANCH}; branch={branch} dirty={bool(dirty)}")
        paths = {
            "source_manifest": Path(config["inputs"]["source_manifest"]),
            "stable_protected_attribution": Path(config["inputs"]["stable_protected_attribution"]),
            "o01_checkpoint": Path(config["inputs"]["o01_checkpoint"]),
            "oracle_o01_checkpoint": Path(config["inputs"]["oracle_o01_checkpoint"]),
            "oracle_o08_checkpoint": Path(config["inputs"]["oracle_o08_checkpoint"]),
            "base_checkpoint": Path(config["base"]["model_dir"]) / config["base"]["checkpoint_path"],
            "lbs_grid": Path(config["base"]["lbs_grid_path"]),
        }
        expected = {
            "source_manifest": config["inputs"]["source_manifest_sha256"],
            "stable_protected_attribution": config["inputs"]["stable_protected_attribution_sha256"],
            "o01_checkpoint": config["inputs"]["o01_checkpoint_sha256"],
            "oracle_o01_checkpoint": config["inputs"]["oracle_o01_checkpoint_sha256"],
            "oracle_o08_checkpoint": config["inputs"]["oracle_o08_checkpoint_sha256"],
            "base_checkpoint": config["base"]["checkpoint_sha256"],
        }
        hashes = {}
        for name, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            hashes[name] = o01.sha256_file(path)
            if name in expected and hashes[name] != expected[name]:
                raise ValueError(f"immutable input SHA mismatch: {name}")
        (output_dir / "contract" / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        _atomic_text(output_dir / "contract" / "command.txt", " ".join([sys.executable, *sys.argv]))
        _atomic_json(output_dir / "input_audit" / "environment.json", o01.environment_snapshot())
        _atomic_json(output_dir / "input_audit" / "input_manifest.json", {
            "task_id": TASK_ID, "git": {"branch": branch, "commit": commit, "clean": True, "source_head": EXPECTED_SOURCE_HEAD},
            "paths": {name: str(path) for name, path in paths.items()}, "sha256": hashes,
            "o01_attempt_immutable": True, "oracle_outputs_immutable": True, "two_outfit_prep_immutable": True,
            "historical_checkpoint_warning": {"historical_sha256": config["base"]["historical_checkpoint_sha256_warning"], "current_sha256": config["base"]["checkpoint_sha256"], "same_file": False},
        })

        status["failure_stage"] = "load_data_base_and_checkpoints"; _atomic_json(output_dir / "RUN_STATUS.json", status)
        samples, episodes, protocol = _generic_outfit_data(paths["source_manifest"])
        base = training.load_frozen_mmlphuman_base(config["base"]["model_dir"], paths["base_checkpoint"], device=device)
        if int(base._xyz.shape[0]) != int(config["base"]["gaussian_count"]):
            raise ValueError("base Gaussian count differs from contract")
        protected_cpu, attribution = o01._load_stable_protected_mask(paths["stable_protected_attribution"], int(config["inputs"]["stable_protected_count"]), int(base._xyz.shape[0]))
        protected_mask = protected_cpu.to(device)
        base_fingerprint_before = _tensor_state_fingerprint(_base_named_tensors(base))
        model, optimizer, _, _, graph = o01._construct_model(base, config, device)
        checkpoint = torch.load(paths["o01_checkpoint"], map_location="cpu", weights_only=False)
        if int(checkpoint["step"]) != 2000 or checkpoint["metadata"]["git_commit"] != EXPECTED_SOURCE_HEAD:
            raise ValueError("O01 checkpoint metadata differs from frozen run")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        backbone_before = o01._state_fingerprint(model.clothing_observation_encoder.backbone.state_dict())
        adapter = MMLPHumanAnchorDeformationAdapter.from_mmlphuman_base(base, lbs_grid_path=paths["lbs_grid"])
        background = torch.tensor(config["render"]["background"], device=device, dtype=base._xyz.dtype)
        for key in list(samples):
            samples[key] = _to_device_nested(samples[key], device); episodes[key] = _to_device_nested(episodes[key], device)
        geometries = {}
        for condition in CONDITIONS:
            geometries[condition] = training.prepare_real_reference_geometry(base, adapter, episodes[f"O01/{condition}"], background)
            for outfit in OUTFITS:
                key = f"{outfit}/{condition}"
                support, _ = o01._base_only_protected_support(base, samples[key], protected_mask, background, float(config["render"]["protected_support_alpha_threshold"]))
                samples[key]["target_protected_mask"] = torch.maximum(samples[key]["target_protected_mask"], support)
        oracle_modules, oracle_targets, oracle_meta = {}, {}, {}
        for outfit in OUTFITS:
            oracle_modules[outfit], oracle_meta[outfit] = _load_oracle(base, paths[f"oracle_{outfit.lower()}_checkpoint"], device)
            oracle_targets[outfit] = apply_protected_full_residual_guard(oracle_modules[outfit].residuals(base), protected_mask)
        _atomic_json(output_dir / "input_audit" / "resolved_protocol.json", {"protocol": protocol, "graph": graph, "oracles": oracle_meta, "target_images_in_prediction_forward": False})

        status["failure_stage"] = "S0_static_reference_oracle_head_audits"; _atomic_json(output_dir / "RUN_STATUS.json", status)
        static, sensitivity, cached = _static_and_sensitivity(model, base, samples, episodes, geometries, protected_mask, background, config, output_dir)
        oracle_report, counterfactuals = _oracle_and_counterfactuals(model, base, samples, episodes, geometries, protected_mask, background, config, output_dir, cached, oracle_targets, oracle_modules)

        status["failure_stage"] = "Probe_D_decoder_capacity"; _atomic_json(output_dir / "RUN_STATUS.json", status)
        probe_d = _probe_d(model, base, samples, protected_mask, background, config, output_dir, oracle_targets)
        status["optimizer_steps"] = int(probe_d["steps"]); _atomic_json(output_dir / "RUN_STATUS.json", status)
        probe_c: dict[str, Any] = {"status": "NOT_RUN_PROBE_D_FAILED", "steps": 0}
        probe_r: dict[str, Any] = {"status": "NOT_RUN_PROBE_C_NOT_PASSED", "steps": 0}
        if probe_d["status"] == "PASS":
            status["failure_stage"] = "Probe_C_reference_conditioning"; _atomic_json(output_dir / "RUN_STATUS.json", status)
            model, optimizer, _, _, _ = o01._construct_model(base, config, device)
            model.load_state_dict(checkpoint["model"], strict=True); optimizer.load_state_dict(checkpoint["optimizer"])
            probe_c = _probe_c(model, optimizer, base, samples, episodes, geometries, protected_mask, background, config, output_dir, oracle_targets)
            status["optimizer_steps"] += int(probe_c["steps"]); _atomic_json(output_dir / "RUN_STATUS.json", status)
            if probe_c["status"] == "PASS":
                status["failure_stage"] = "Probe_R_render_objective_drift"; _atomic_json(output_dir / "RUN_STATUS.json", status)
                probe_r = _probe_r(model, optimizer, base, samples, episodes, geometries, protected_mask, background, config, output_dir, oracle_targets)
                status["optimizer_steps"] += int(probe_r["steps"]); _atomic_json(output_dir / "RUN_STATUS.json", status)

        adjudication = decision_case(
            probe_d_pass=probe_d["status"] == "PASS", probe_c_ran=probe_c["steps"] > 0,
            probe_c_pass=probe_c["status"] == "PASS", probe_r_ran=probe_r["steps"] > 0,
            probe_r_degraded=probe_r["status"] == "DEGRADED",
        )
        base_fingerprint_after = _tensor_state_fingerprint(_base_named_tensors(base))
        backbone_after = o01._state_fingerprint(model.clothing_observation_encoder.backbone.state_dict())
        freeze = {
            "base_before": base_fingerprint_before, "base_after": base_fingerprint_after,
            "base_bitwise_unchanged": base_fingerprint_before == base_fingerprint_after,
            "base_gradient_count": _base_gradient_count(base),
            "frozen_image_backbone_before": backbone_before, "frozen_image_backbone_after": backbone_after,
            "frozen_image_backbone_bitwise_unchanged": backbone_before == backbone_after,
            "frozen_image_backbone_gradient_count": sum(parameter.grad is not None for parameter in model.clothing_observation_encoder.backbone.parameters()),
        }
        if not freeze["base_bitwise_unchanged"] or freeze["base_gradient_count"] != 0 or not freeze["frozen_image_backbone_bitwise_unchanged"]:
            raise AssertionError("frozen base/backbone contract failed")
        result = {
            "task_id": TASK_ID, "run_commit": commit, "source_head": EXPECTED_SOURCE_HEAD,
            "static_audit": static, "reference_sensitivity": sensitivity,
            "oracle_residual_comparison": oracle_report,
            "head_counterfactuals_path": str(output_dir / "head_counterfactuals" / "head_counterfactuals.json"),
            "probe_d": probe_d, "probe_c": probe_c, "probe_r": probe_r,
            "freeze": freeze, "optimizer_steps": status["optimizer_steps"],
            "final_case": adjudication, "formal_two_outfit_long_training_allowed": False,
            "visual_status": "PENDING_ACTUAL_INSPECTION",
        }
        _atomic_json(output_dir / "final_adjudication" / "diagnosis.json", result)
        _atomic_text(output_dir / "final_adjudication" / "FINAL_ADJUDICATION.md", "\n".join([
            f"# {TASK_ID}", "", f"- Run commit: `{commit}`", f"- Case: **{adjudication['case']}**",
            f"- Root-cause state: **{adjudication['state']}**", f"- Next task: `{adjudication['next_task']}`",
            f"- Probe D: **{probe_d['status']}** ({probe_d['steps']} steps)",
            f"- Probe C: **{probe_c['status']}** ({probe_c['steps']} steps)",
            f"- Probe R: **{probe_r['status']}** ({probe_r['steps']} steps)",
            "- Formal two-outfit long training: **NOT ALLOWED**", "- Visual status: **PENDING_ACTUAL_INSPECTION**",
            "", "Oracle residuals were used only as diagnostic regression targets/offline evaluation and never entered the formal image-conditioned forward.",
        ]))
        status.update({"status": "AWAITING_VISUAL_INSPECTION", "failure_stage": None, "finished_at": time.time(), "final_case": adjudication["case"]})
        _atomic_json(output_dir / "RUN_STATUS.json", status)
    except Exception as error:
        status.update({"status": "FAIL", "finished_at": time.time(), "exception_type": type(error).__name__, "exception": str(error), "traceback": traceback.format_exc()})
        _atomic_json(output_dir / "RUN_STATUS.json", status)
        raise


def finalize_visual(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    status_path = output_dir / "RUN_STATUS.json"; diagnosis_path = output_dir / "final_adjudication" / "diagnosis.json"
    status = json.loads(status_path.read_text(encoding="utf-8")); diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    if status["status"] != "AWAITING_VISUAL_INSPECTION":
        raise RuntimeError("diagnosis is not awaiting visual inspection")
    if args.visual_status not in {"PASS", "WARN", "FAIL"} or len(args.visual_note) < 4:
        raise ValueError("visual finalization requires PASS/WARN/FAIL and at least four concrete observations")
    opened = [
        output_dir / "current_checkpoint_static_audit" / "static_four_view_contact_sheet.png",
        output_dir / "reference_sensitivity" / "reference_sensitivity_contact_sheet.png",
        output_dir / "oracle_residual_comparison" / "network_oracle_four_view_contact_sheet.png",
        output_dir / "head_counterfactuals" / "head_counterfactual_contact_sheet.png",
        output_dir / "probe_D_decoder_capacity" / "probe_d_oracle_render_contact_sheet.png",
    ]
    if diagnosis["probe_c"]["steps"] > 0:
        opened.append(output_dir / "probe_C_reference_conditioning" / "probe_c_correct_reference_contact_sheet.png")
    if any(not path.is_file() for path in opened):
        raise FileNotFoundError("a required diagnostic visual is missing")
    visual = {
        "images_actually_opened": [str(path) for path in opened],
        "inspection_method": "Codex local image viewer at full contact-sheet resolution",
        "observations": list(args.visual_note), "status": args.visual_status,
    }
    _atomic_json(output_dir / "visual_acceptance" / "visual_acceptance.json", visual)
    _atomic_text(output_dir / "visual_acceptance" / "VISUAL_ACCEPTANCE.md", "\n".join([
        "# Failure-diagnosis visual acceptance", "", f"- Status: **{args.visual_status}**",
        "- Images actually opened:", *[f"  - `{path}`" for path in visual["images_actually_opened"]],
        "", "## Concrete observations", "", *[f"- {value}" for value in args.visual_note],
    ]))
    diagnosis["visual_status"] = args.visual_status; diagnosis["visual_acceptance"] = visual
    diagnosis["status"] = "PASS" if args.visual_status in {"PASS", "WARN"} else "FAIL"
    _atomic_json(diagnosis_path, diagnosis)
    status.update({"status": diagnosis["status"], "visual_status": args.visual_status, "finished_at": time.time()})
    _atomic_json(status_path, status)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preregistered O01 image-conditioned failure diagnosis")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/research/subject02_image_conditioned_failure_diagnosis_v1.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--finalize-visual", action="store_true")
    parser.add_argument("--visual-status", choices=("PASS", "WARN", "FAIL"))
    parser.add_argument("--visual-note", action="append", default=[])
    args = parser.parse_args()
    if args.finalize_visual:
        if args.output_dir is None:
            raise ValueError("--finalize-visual requires --output-dir")
        finalize_visual(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
