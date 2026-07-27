from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F


COLLAPSE_STATES = (
    "COLLAPSE_FEATURE",
    "COLLAPSE_RESIDUAL",
    "COLLAPSE_RENDER",
    "MEAN_OUTFIT_COLLAPSE",
    "REFERENCE_IGNORED",
)


def _float_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
        raise TypeError(f"{name} must be a floating-point tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return value.detach().float().cpu()


def cosine_distance(first: torch.Tensor, second: torch.Tensor) -> float:
    a = _float_tensor(first, "first").reshape(-1)
    b = _float_tensor(second, "second").reshape(-1)
    if a.shape != b.shape:
        raise ValueError("distance tensors must have identical shape")
    if a.norm() == 0 and b.norm() == 0:
        return 0.0
    if a.norm() == 0 or b.norm() == 0:
        return 1.0
    return float(1.0 - F.cosine_similarity(a[None], b[None]).item())


def normalized_l2(first: torch.Tensor, second: torch.Tensor, epsilon: float = 1e-12) -> float:
    a = _float_tensor(first, "first").reshape(-1)
    b = _float_tensor(second, "second").reshape(-1)
    if a.shape != b.shape:
        raise ValueError("distance tensors must have identical shape")
    scale = 0.5 * (a.square().mean().sqrt() + b.square().mean().sqrt())
    return float((a - b).square().mean().sqrt() / scale.clamp_min(epsilon))


def feature_distance(first: torch.Tensor, second: torch.Tensor) -> dict[str, float]:
    return {
        "cosine_distance": cosine_distance(first, second),
        "normalized_l2": normalized_l2(first, second),
    }


def _distribution(values: torch.Tensor) -> dict[str, float | int]:
    flat = _float_tensor(values, "distribution").reshape(-1)
    if flat.numel() == 0:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "count": flat.numel(),
        "mean": float(flat.mean()),
        "p50": float(torch.quantile(flat, 0.50)),
        "p95": float(torch.quantile(flat, 0.95)),
    }


def local_anchor_feature_distance(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    observed_mask: torch.Tensor | None = None,
    completed_mask: torch.Tensor | None = None,
    garment_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    a = _float_tensor(first, "first local features")
    b = _float_tensor(second, "second local features")
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("local features must have identical [anchors, features] shape")
    per_anchor = (a - b).square().mean(dim=1).sqrt()

    def selected(mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return per_anchor
        boolean = torch.as_tensor(mask).detach().cpu().reshape(-1) >= 0.5
        if boolean.shape[0] != per_anchor.shape[0]:
            raise ValueError("anchor mask length differs from local feature count")
        return per_anchor[boolean]

    return {
        "all": _distribution(per_anchor),
        "observed_anchors": _distribution(selected(observed_mask)),
        "completed_anchors": _distribution(selected(completed_mask)),
        "garment_related_anchors": _distribution(selected(garment_mask)),
        "global": feature_distance(a, b),
    }


def _correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    a, b = first.reshape(-1), second.reshape(-1)
    a = a - a.mean(); b = b - b.mean()
    denominator = a.norm() * b.norm()
    if denominator <= 1e-12:
        return 1.0 if torch.equal(first, second) else 0.0
    return float(torch.dot(a, b) / denominator)


def gate_distance(first: torch.Tensor, second: torch.Tensor, threshold: float = 0.5) -> dict[str, Any]:
    a = _float_tensor(first, "first gate")
    b = _float_tensor(second, "second gate")
    if a.shape != b.shape:
        raise ValueError("gate tensors must have identical shape")
    active_a, active_b = a >= threshold, b >= threshold
    union = torch.logical_or(active_a, active_b).sum()
    intersection = torch.logical_and(active_a, active_b).sum()
    return {
        **feature_distance(a, b),
        "mean_absolute_distance": float((a - b).abs().mean()),
        "active_fraction_first": float(active_a.float().mean()),
        "active_fraction_second": float(active_b.float().mean()),
        "active_iou": 1.0 if union == 0 else float(intersection.float() / union.float()),
        "correlation": _correlation(a, b),
        "threshold": float(threshold),
    }


def residual_channel_distance(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    bound: float,
    active_epsilon: float = 1e-8,
) -> dict[str, Any]:
    if not isinstance(bound, (int, float)) or float(bound) <= 0:
        raise ValueError("residual bound must be positive")
    a = _float_tensor(first, "first residual")
    b = _float_tensor(second, "second residual")
    if a.shape != b.shape or a.ndim < 1:
        raise ValueError("residual tensors must have identical non-scalar shape")
    delta = (a - b) / float(bound)
    a_rows, b_rows = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
    active_a = a_rows.norm(dim=1) > active_epsilon
    active_b = b_rows.norm(dim=1) > active_epsilon
    union = torch.logical_or(active_a, active_b).sum()
    intersection = torch.logical_and(active_a, active_b).sum()
    return {
        "bound": float(bound),
        "bound_normalized_l1": float(delta.abs().mean()),
        "bound_normalized_l2": float(delta.square().mean().sqrt()),
        "cosine_distance": cosine_distance(a, b),
        "active_gaussian_overlap_iou": (
            1.0 if union == 0 else float(intersection.float() / union.float())
        ),
        "active_fraction_first": float(active_a.float().mean()),
        "active_fraction_second": float(active_b.float().mean()),
    }


def residual_distance(
    first: Mapping[str, torch.Tensor],
    second: Mapping[str, torch.Tensor],
    bounds: Mapping[str, float],
) -> dict[str, Any]:
    if set(first) != set(second):
        raise ValueError("residual channel sets differ")
    missing_bounds = sorted(set(first).difference(bounds))
    if missing_bounds:
        raise ValueError(f"residual bounds are missing channels: {missing_bounds}")
    return {
        name: residual_channel_distance(first[name], second[name], bound=float(bounds[name]))
        for name in sorted(first)
    }


def _as_chw(value: torch.Tensor, name: str) -> torch.Tensor:
    tensor = _float_tensor(value, name)
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"{name} must be HW, CHW, or HWC")
    if tensor.shape[0] in {1, 3}:
        return tensor
    if tensor.shape[-1] in {1, 3}:
        return tensor.permute(2, 0, 1)
    raise ValueError(f"cannot infer channel dimension for {name}")


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    active = _as_chw(mask, "mask")[:1] >= 0.5
    if value.shape[-2:] != active.shape[-2:]:
        raise ValueError("mask resolution differs from render")
    expanded = active.expand(value.shape[0], -1, -1)
    if not expanded.any():
        return 0.0
    return float(value[expanded].mean())


def render_distance(
    first_rgb: torch.Tensor,
    first_alpha: torch.Tensor,
    second_rgb: torch.Tensor,
    second_alpha: torch.Tensor,
    *,
    garment_mask: torch.Tensor,
    lpips_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, Any]:
    rgb_a, rgb_b = _as_chw(first_rgb, "first_rgb"), _as_chw(second_rgb, "second_rgb")
    alpha_a, alpha_b = _as_chw(first_alpha, "first_alpha")[:1], _as_chw(second_alpha, "second_alpha")[:1]
    if rgb_a.shape != rgb_b.shape or alpha_a.shape != alpha_b.shape:
        raise ValueError("render pairs have mismatched shape")
    silhouette_a, silhouette_b = alpha_a >= 0.5, alpha_b >= 0.5
    result: dict[str, Any] = {
        "rgb_garment_region_distance": _masked_mean((rgb_a - rgb_b).abs(), garment_mask),
        "alpha_distance": float((alpha_a - alpha_b).abs().mean()),
        "silhouette_disagreement": float(torch.logical_xor(silhouette_a, silhouette_b).float().mean()),
        "lpips": {"available": lpips_fn is not None, "value": None},
    }
    if lpips_fn is not None:
        result["lpips"]["value"] = float(lpips_fn(rgb_a.unsqueeze(0), rgb_b.unsqueeze(0)))
    return result


def mean_outfit_collapse_metrics(
    first_prediction_rgb: torch.Tensor,
    second_prediction_rgb: torch.Tensor,
    first_target_rgb: torch.Tensor,
    second_target_rgb: torch.Tensor,
    *,
    garment_mask: torch.Tensor,
) -> dict[str, Any]:
    """Compare each prediction with its own target and the two-target pixel mean."""

    prediction_a = _as_chw(first_prediction_rgb, "first_prediction_rgb")
    prediction_b = _as_chw(second_prediction_rgb, "second_prediction_rgb")
    target_a = _as_chw(first_target_rgb, "first_target_rgb")
    target_b = _as_chw(second_target_rgb, "second_target_rgb")
    if not (prediction_a.shape == prediction_b.shape == target_a.shape == target_b.shape):
        raise ValueError("mean-outfit comparison tensors must share shape")
    mean_target = 0.5 * (target_a + target_b)
    own_a = _masked_mean((prediction_a - target_a).abs(), garment_mask)
    own_b = _masked_mean((prediction_b - target_b).abs(), garment_mask)
    mean_a = _masked_mean((prediction_a - mean_target).abs(), garment_mask)
    mean_b = _masked_mean((prediction_b - mean_target).abs(), garment_mask)
    advantages = [mean_a - own_a, mean_b - own_b]
    return {
        "first": {"own_target_error": own_a, "mean_target_error": mean_a, "own_target_advantage": advantages[0]},
        "second": {"own_target_error": own_b, "mean_target_error": mean_b, "own_target_advantage": advantages[1]},
        "mean_outfit_margin": sum(advantages) / 2.0,
        "both_predictions_closer_to_mean_than_own_target": all(value < 0 for value in advantages),
    }


def classify_collapse_risk(metrics: Mapping[str, Any], thresholds: Mapping[str, float] | None = None) -> dict[str, Any]:
    if thresholds is None:
        return {
            "status": "THRESHOLDS_UNCALIBRATED_REPORT_ONLY",
            "active_states": [],
            "candidate_states": list(COLLAPSE_STATES),
            "thresholds_frozen": False,
        }
    required = {
        "feature_distance_min",
        "residual_distance_min",
        "render_distance_min",
        "reference_sensitivity_min",
        "mean_outfit_margin_min",
    }
    missing = sorted(required.difference(thresholds))
    if missing:
        raise ValueError(f"collapse thresholds are incomplete: {missing}")
    scalar = metrics["classification_scalars"]
    active: list[str] = []
    feature_small = scalar["feature_distance"] < thresholds["feature_distance_min"]
    residual_small = scalar["residual_distance"] < thresholds["residual_distance_min"]
    render_small = scalar["render_distance"] < thresholds["render_distance_min"]
    if feature_small:
        active.append("COLLAPSE_FEATURE")
    if not feature_small and residual_small:
        active.append("COLLAPSE_RESIDUAL")
    if not residual_small and render_small:
        active.append("COLLAPSE_RENDER")
    if scalar["reference_sensitivity"] < thresholds["reference_sensitivity_min"]:
        active.append("REFERENCE_IGNORED")
    if scalar["mean_outfit_margin"] < thresholds["mean_outfit_margin_min"]:
        active.append("MEAN_OUTFIT_COLLAPSE")
    return {
        "status": "NO_COLLAPSE_SIGNAL" if not active else active[0],
        "active_states": active,
        "candidate_states": list(COLLAPSE_STATES),
        "thresholds_frozen": True,
    }


def measure_conditioning_collapse(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    residual_bounds: Mapping[str, float],
    garment_anchor_mask: torch.Tensor | None = None,
    garment_render_mask: torch.Tensor | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    observed_probability = 0.5 * (
        first["observed_probability"].detach().float().cpu()
        + second["observed_probability"].detach().float().cpu()
    )
    completed_probability = 0.5 * (
        first["completed_probability"].detach().float().cpu()
        + second["completed_probability"].detach().float().cpu()
    )
    observed_mask = observed_probability >= 0.5
    completed_mask = torch.logical_not(observed_mask) & (completed_probability >= 0.5)
    local = local_anchor_feature_distance(
        first["local_anchor_features"],
        second["local_anchor_features"],
        observed_mask=observed_mask,
        completed_mask=completed_mask,
        garment_mask=garment_anchor_mask,
    )
    geometry_gate = gate_distance(first["geometry_gate"], second["geometry_gate"])
    appearance_gate = gate_distance(first["appearance_gate"], second["appearance_gate"])
    residuals = residual_distance(first["gaussian_residual"], second["gaussian_residual"], residual_bounds)
    render = None
    if garment_render_mask is not None:
        render = render_distance(
            first["final_rgb"], first["final_alpha"], second["final_rgb"], second["final_alpha"],
            garment_mask=garment_render_mask,
        )
    mean_outfit = None
    if garment_render_mask is not None and "target_rgb" in first and "target_rgb" in second:
        mean_outfit = mean_outfit_collapse_metrics(
            first["final_rgb"],
            second["final_rgb"],
            first["target_rgb"],
            second["target_rgb"],
            garment_mask=garment_render_mask,
        )
    residual_scalar = sum(item["bound_normalized_l2"] for item in residuals.values()) / max(len(residuals), 1)
    render_scalar = 0.0 if render is None else (
        render["rgb_garment_region_distance"] + render["alpha_distance"] + render["silhouette_disagreement"]
    ) / 3.0
    report: dict[str, Any] = {
        "global_feature_distance": feature_distance(
            first["global_clothing_embedding"], second["global_clothing_embedding"]
        ),
        "local_anchor_feature_distance": local,
        "gate_distance": {"geometry_gate": geometry_gate, "appearance_gate": appearance_gate},
        "residual_distance": residuals,
        "render_distance": render,
        "mean_outfit_collapse": mean_outfit,
        "input_sensitivity": {
            "feature_changed": feature_distance(first["global_clothing_embedding"], second["global_clothing_embedding"]),
            "residual_changed": residual_scalar,
            "render_changed": render_scalar,
        },
        "classification_scalars": {
            "feature_distance": local["global"]["normalized_l2"],
            "residual_distance": residual_scalar,
            "render_distance": render_scalar,
            "reference_sensitivity": render_scalar,
            "mean_outfit_margin": (
                float(mean_outfit["mean_outfit_margin"])
                if mean_outfit is not None
                else float(first.get("mean_outfit_margin", 0.0) + second.get("mean_outfit_margin", 0.0)) / 2.0
            ),
        },
    }
    report["collapse_risk"] = classify_collapse_risk(report, thresholds)
    return report


def _load_tensor_bundle(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise TypeError("tensor bundle must contain a dictionary")
    local = value.get("local_anchor_features.completed", value.get("local_anchor_features"))
    return {
        "global_clothing_embedding": value["global_clothing_embedding"],
        "local_anchor_features": local,
        "observed_probability": value["observed_probability"],
        "completed_probability": value["completed_probability"],
        "geometry_gate": value["geometry_gate"],
        "appearance_gate": value["appearance_gate"],
        "gaussian_residual": {
            key.removeprefix("gaussian_residual."): tensor
            for key, tensor in value.items() if key.startswith("gaussian_residual.")
        },
        "final_rgb": value.get("final_rgb"),
        "final_alpha": value.get("final_alpha"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure multi-outfit reference-conditioning separation.")
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--bounds-json", type=Path, required=True)
    parser.add_argument("--garment-render-mask", type=Path)
    parser.add_argument("--thresholds-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bounds = json.loads(args.bounds_json.read_text(encoding="utf-8"))
    thresholds = None if args.thresholds_json is None else json.loads(args.thresholds_json.read_text(encoding="utf-8"))
    mask = None if args.garment_render_mask is None else torch.load(args.garment_render_mask, map_location="cpu", weights_only=False)
    report = measure_conditioning_collapse(
        _load_tensor_bundle(args.first),
        _load_tensor_bundle(args.second),
        residual_bounds=bounds,
        garment_render_mask=mask,
        thresholds=thresholds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "collapse_risk": report["collapse_risk"]}, indent=2))


if __name__ == "__main__":
    main()
