from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F


OBJECTIVE_NAME = "MULTI_VIEW_SCREEN_SPACE_SUPPORT_PLACEMENT_OBJECTIVE_V1"
SUPPORT_RENDER_NAME = "FIXED_ATTRIBUTE_SUPPORT_RENDER_V1"
FORWARD_ALLOWED_FIELDS = frozenset({"pose", "Rh", "Th", "camera"})
FORWARD_FORBIDDEN_TOKENS = (
    "target_rgb", "target_mask", "target_clothing", "target_silhouette",
    "outfit", "teacher", "p1", "reference_image",
)


@dataclass(frozen=True)
class EditableGaussianPool:
    indices: torch.Tensor
    seed_mask: torch.Tensor
    graph_added_mask: torch.Tensor
    protected_excluded_mask: torch.Tensor
    expanded_anchor_mask: torch.Tensor


def _require_vector(value: torch.Tensor, count: int, name: str) -> torch.Tensor:
    if value.ndim != 1 or value.shape[0] != count:
        raise ValueError(f"{name} must have shape [{count}], got {tuple(value.shape)}")
    if not torch.isfinite(value.float()).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return value


def validate_pool_source_names(names: Sequence[str]) -> None:
    allowed = {
        "base_gaussian", "base_old_clothing", "base_foreground",
        "base_protected", "base_visibility", "anchor_graph", "body_part",
    }
    unknown = set(names).difference(allowed)
    if unknown:
        raise ValueError(f"target-dependent or unknown editable-pool source: {sorted(unknown)}")


def build_editable_gaussian_pool(
    *,
    old_clothing_hits: torch.Tensor,
    garment_envelope_hits: torch.Tensor,
    protected_hits: torch.Tensor,
    visible_hits: torch.Tensor,
    dominant_anchor: torch.Tensor,
    anchor_neighbors: torch.Tensor,
    protected_min_views: int = 2,
) -> EditableGaussianPool:
    """Build the shared pool using base-derived evidence only.

    Hit vectors are counts over the four registered base views. The caller must
    obtain them before consulting any outfit target. A stable protected Gaussian
    is excluded before and after one-hop anchor expansion.
    """

    count = int(dominant_anchor.shape[0])
    old = _require_vector(old_clothing_hits, count, "old_clothing_hits")
    envelope = _require_vector(garment_envelope_hits, count, "garment_envelope_hits")
    protected = _require_vector(protected_hits, count, "protected_hits")
    visible = _require_vector(visible_hits, count, "visible_hits")
    dominant = _require_vector(dominant_anchor, count, "dominant_anchor").long()
    if anchor_neighbors.ndim != 2 or anchor_neighbors.shape[0] <= 0:
        raise ValueError("anchor_neighbors must have shape [A,K]")
    if dominant.min() < 0 or dominant.max() >= anchor_neighbors.shape[0]:
        raise IndexError("dominant anchor is out of range")
    if protected_min_views <= 0:
        raise ValueError("protected_min_views must be positive")

    stable_protected = (
        (protected >= protected_min_views)
        & (protected >= envelope)
        & (old == 0)
    )
    seed = ((old > 0) | (envelope > 0)) & (visible > 0) & ~stable_protected
    seed_anchors = torch.zeros(
        anchor_neighbors.shape[0], dtype=torch.bool, device=dominant.device,
    )
    seed_anchors[dominant[seed]] = True
    expanded = seed_anchors.clone()
    if seed_anchors.any():
        expanded[anchor_neighbors[seed_anchors].reshape(-1).long()] = True
    candidate = expanded[dominant] & (visible > 0)
    pool_mask = candidate & ~stable_protected
    graph_added = pool_mask & ~seed
    return EditableGaussianPool(
        indices=torch.where(pool_mask)[0],
        seed_mask=seed,
        graph_added_mask=graph_added,
        protected_excluded_mask=candidate & stable_protected,
        expanded_anchor_mask=expanded,
    )


def fixed_support_attributes(
    posed_xyz: torch.Tensor,
    posed_covariance: torch.Tensor,
    editable_indices: torch.Tensor,
    *,
    opacity: float = 0.05,
) -> dict[str, torch.Tensor]:
    if posed_xyz.ndim != 2 or posed_xyz.shape[1] != 3:
        raise ValueError("posed_xyz must have shape [N,3]")
    if posed_covariance.shape != (posed_xyz.shape[0], 3, 3):
        raise ValueError("posed_covariance must have shape [N,3,3]")
    if editable_indices.ndim != 1 or editable_indices.dtype != torch.long:
        raise TypeError("editable_indices must be a 1D int64 tensor")
    if not 0 < opacity < 1:
        raise ValueError("fixed support opacity must be in (0,1)")
    if editable_indices.numel() and (
        editable_indices.min() < 0 or editable_indices.max() >= posed_xyz.shape[0]
    ):
        raise IndexError("editable index is out of range")
    means = posed_xyz[editable_indices]
    return {
        "means": means,
        "covars": posed_covariance[editable_indices].detach(),
        "opacities": torch.full(
            (editable_indices.numel(),), opacity, device=means.device, dtype=means.dtype,
        ),
        "colors": torch.ones(
            (editable_indices.numel(), 3), device=means.device, dtype=means.dtype,
        ),
    }


def fixed_attribute_support_render(
    posed_xyz: torch.Tensor,
    posed_covariance: torch.Tensor,
    editable_indices: torch.Tensor,
    camera: Mapping[str, Any],
    *,
    opacity: float = 0.05,
    rasterizer: Callable[..., tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]]] | None = None,
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    if rasterizer is None:
        from gsplat import rasterization as rasterizer
    attributes = fixed_support_attributes(
        posed_xyz, posed_covariance, editable_indices, opacity=opacity,
    )
    image, alpha, info = rasterizer(
        means=attributes["means"], quats=None, scales=None,
        opacities=attributes["opacities"], colors=attributes["colors"],
        viewmats=camera["w2c"][None], Ks=camera["K"][None],
        width=int(camera["width"]), height=int(camera["height"]),
        packed=False, near_plane=0.1,
        backgrounds=torch.zeros((1, 3), device=posed_xyz.device, dtype=posed_xyz.dtype),
        covars=attributes["covars"],
    )
    del image
    if alpha.ndim != 4 or alpha.shape[0] != 1 or alpha.shape[-1] != 1:
        raise ValueError(f"support alpha must be [1,H,W,1], got {tuple(alpha.shape)}")
    support = alpha[0].permute(2, 0, 1).contiguous()
    if not torch.isfinite(support).all():
        raise FloatingPointError("support render contains NaN or Inf")
    return support, info


def build_placement_masks(
    sample: Mapping[str, torch.Tensor],
    regions: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    required_sample = {"target_clothing_mask"}
    required_regions = {
        "trusted_expansion", "trusted_removal", "protected_identity",
        "silhouette_uncertain", "background",
    }
    if missing := required_sample.difference(sample):
        raise KeyError(f"missing placement sample fields: {sorted(missing)}")
    if missing := required_regions.difference(regions):
        raise KeyError(f"missing placement region fields: {sorted(missing)}")
    safe = sample["target_clothing_mask"].to(regions["trusted_expansion"])
    target = torch.maximum(safe, regions["trusted_expansion"])
    target = target * (1 - regions["protected_identity"]) * (1 - regions["silhouette_uncertain"])
    forbidden = torch.maximum(regions["trusted_removal"], regions["background"])
    return (target >= 0.5).to(target), (forbidden >= 0.5).to(forbidden)


def _active_normalized(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    active = (mask >= 0.5).to(values)
    if active.shape != values.shape:
        active = active.expand_as(values)
    return (values * active).sum() / active.sum().clamp_min(1)


def support_coverage_loss(
    support: torch.Tensor, target_mask: torch.Tensor, *, threshold: float = 0.65,
) -> torch.Tensor:
    underfill = F.relu(float(threshold) - support)
    values = F.smooth_l1_loss(underfill, torch.zeros_like(underfill), reduction="none")
    return _active_normalized(values, target_mask)


def support_spill_loss(
    support: torch.Tensor, forbidden_mask: torch.Tensor, *, threshold: float = 0.10,
) -> torch.Tensor:
    spill = F.relu(support - float(threshold))
    values = F.smooth_l1_loss(spill, torch.zeros_like(spill), reduction="none")
    return _active_normalized(values, forbidden_mask)


def graph_flow_smoothness(
    delta_xyz: torch.Tensor,
    graph_edges: torch.Tensor,
    *,
    xyz_bound: float,
) -> torch.Tensor:
    if delta_xyz.ndim != 2 or delta_xyz.shape[1] != 3:
        raise ValueError("delta_xyz must have shape [N,3]")
    if graph_edges.ndim != 2 or graph_edges.shape[1] != 2:
        raise ValueError("graph_edges must have shape [E,2]")
    if xyz_bound <= 0:
        raise ValueError("xyz_bound must be positive")
    if graph_edges.numel() == 0:
        return delta_xyz.sum() * 0
    if graph_edges.min() < 0 or graph_edges.max() >= delta_xyz.shape[0]:
        raise IndexError("flow graph edge is out of range")
    difference = (delta_xyz[graph_edges[:, 0]] - delta_xyz[graph_edges[:, 1]]) / float(xyz_bound)
    return F.smooth_l1_loss(difference, torch.zeros_like(difference), reduction="mean")


def equal_weight_view_loss(values: Sequence[torch.Tensor]) -> torch.Tensor:
    if not values:
        raise ValueError("at least one view loss is required")
    if any(value.numel() != 1 or not torch.isfinite(value) for value in values):
        raise FloatingPointError("view losses must be finite scalars")
    return torch.stack(tuple(values)).mean()


def placement_loss(
    supports: Sequence[torch.Tensor],
    target_masks: Sequence[torch.Tensor],
    forbidden_masks: Sequence[torch.Tensor],
    delta_xyz: torch.Tensor,
    graph_edges: torch.Tensor,
    *,
    xyz_bound: float,
    weights: Mapping[str, float],
) -> dict[str, torch.Tensor]:
    if not (len(supports) == len(target_masks) == len(forbidden_masks)):
        raise ValueError("support/mask view counts differ")
    coverage = equal_weight_view_loss([
        support_coverage_loss(value, mask) for value, mask in zip(supports, target_masks)
    ])
    spill = equal_weight_view_loss([
        support_spill_loss(value, mask) for value, mask in zip(supports, forbidden_masks)
    ])
    flow = graph_flow_smoothness(delta_xyz, graph_edges, xyz_bound=xyz_bound)
    total = (
        float(weights["coverage"]) * coverage
        + float(weights["spill"]) * spill
        + float(weights["flow"]) * flow
    )
    return {"total": total, "coverage": coverage, "spill": spill, "flow": flow}


def _average_ranks(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.tensor(tuple(values), dtype=torch.float64)
    order = torch.argsort(tensor)
    ranks = torch.empty_like(tensor)
    start = 0
    while start < tensor.numel():
        end = start + 1
        while end < tensor.numel() and tensor[order[end]] == tensor[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman inputs must have equal length >=2")
    x, y = _average_ranks(left), _average_ranks(right)
    x, y = x - x.mean(), y - y.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    return float((x * y).sum() / denominator) if denominator > 0 else 0.0


def qualify_support_proxy(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_p1_margin: float = 0.15,
    minimum_spearman: float = 0.80,
    underfill_threshold: float = 0.65,
) -> dict[str, Any]:
    lookup = {(row["state"], row["view"]): row for row in rows}
    required = {(state, view) for state in ("P1", "P2", "P3") for view in ("back", "right")}
    if missing := required.difference(lookup):
        raise KeyError(f"missing support-proxy rows: {sorted(missing)}")
    margins = {
        view: float(lookup[("P1", view)]["mean_trusted_support"])
        - float(lookup[("P3", view)]["mean_trusted_support"])
        for view in ("back", "right")
    }
    ordered = [lookup[key] for key in sorted(required)]
    correlation = spearman_correlation(
        [float(row["mean_trusted_support"]) for row in ordered],
        [float(row["instrumented_n_pre"]) for row in ordered],
    )
    underfill = {
        f"{state}_{view}": (
            float(lookup[(state, view)]["mean_trusted_support"]) < underfill_threshold
            and float(lookup[(state, view)]["low_support_hole_ratio"]) > 0
        )
        for state in ("P2", "P3") for view in ("back", "right")
    }
    spill_detected = {
        f"{state}_{view}": float(lookup[(state, view)]["spill_active_fraction"]) > 0
        for state in ("P2", "P3") for view in ("back", "right")
    }
    checks = {
        "p1_margin_back": margins["back"] >= minimum_p1_margin,
        "p1_margin_right": margins["right"] >= minimum_p1_margin,
        "spearman": correlation >= minimum_spearman,
        "p2_p3_underfill": all(underfill.values()),
        "trailing_spill_detected": all(spill_detected.values()),
    }
    passed = all(checks.values())
    return {
        "status": "PASS" if passed else "SUPPORT_PROXY_INVALID",
        "checks": checks,
        "p1_minus_p3_margin": margins,
        "spearman": correlation,
        "underfill": underfill,
        "spill_detected": spill_detected,
        "optimizer_allowed": passed,
    }


def gradient_direction_statistics(
    negative_screen_gradient: torch.Tensor,
    desired_direction: torch.Tensor,
) -> dict[str, float]:
    if negative_screen_gradient.shape != desired_direction.shape or negative_screen_gradient.shape[-1] != 2:
        raise ValueError("screen-gradient tensors must have matching [N,2] shape")
    cosine = F.cosine_similarity(negative_screen_gradient, desired_direction, dim=-1, eps=1e-12)
    return {
        "positive_fraction": float((cosine > 0).float().mean()) if cosine.numel() else 0.0,
        "median_cosine": float(cosine.median()) if cosine.numel() else 0.0,
        "finite_fraction": float(torch.isfinite(cosine).float().mean()) if cosine.numel() else 1.0,
    }


def frozen_calibration_scale(
    placement_gradient: float,
    edit_gradient: float,
    *,
    target_ratio: float = 0.75,
) -> float:
    if placement_gradient <= 0 or edit_gradient <= 0:
        raise ValueError("calibration gradients must be positive")
    if not 0.5 <= target_ratio <= 1.0:
        raise ValueError("placement target ratio must be in [0.5,1.0]")
    return target_ratio * edit_gradient / placement_gradient


def assert_target_free_forward_fields(fields: Sequence[str]) -> None:
    if set(fields).difference(FORWARD_ALLOWED_FIELDS):
        raise ValueError("placement forward contains a non-registered field")
    lowered = tuple(field.lower() for field in fields)
    if any(token in field for field in lowered for token in FORWARD_FORBIDDEN_TOKENS):
        raise ValueError("target/teacher/outfit field entered placement forward")


def g1_contract(base_contract: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base_contract)
    result["additional_objective"] = OBJECTIVE_NAME
    result["initialization"] = "zero_residual"
    result["other_variables_changed"] = False
    return result


def g2_contract() -> dict[str, Any]:
    return {
        "eligible_only_after_g1": True,
        "initialization": "zero_residual",
        "stage_P": {"steps": [0, 159], "trainable": ["delta_xyz"]},
        "stage_J": {
            "steps": [160, 999],
            "trainable": ["xyz", "scaling", "rotation", "opacity", "sh0"],
        },
        "outfit_specific_tuning": False,
    }


def o01_regression_pass(
    rows: Sequence[Mapping[str, float]],
    baseline: Mapping[str, Mapping[str, float]],
) -> bool:
    for row in rows:
        view = str(row["view"])
        if float(row["raw_silhouette_iou"]) < 0.90:
            return False
        if float(row["trusted_expansion_recall"]) < float(baseline[view]["trusted_expansion_recall"]) - 0.02:
            return False
        if float(row["trusted_removal_recall"]) < float(baseline[view]["trusted_removal_recall"]) - 0.03:
            return False
        if float(row["protected_mae"]) > 0.01 or float(row["background_leakage"]) > 0.03:
            return False
    return sum(float(row["target_closer_fraction"]) for row in rows) / max(len(rows), 1) >= 0.88
