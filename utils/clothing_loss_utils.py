from __future__ import annotations

import torch
import torch.nn.functional as F


_OFFSET_FIELDS = ("delta_xyz", "delta_scaling", "delta_opacity")


def masked_smooth_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    beta: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return an elementwise weighted Smooth L1 mean without detaching prediction."""

    if not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("prediction and target must be torch.Tensor instances")
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes must match, got {tuple(prediction.shape)} "
            f"and {tuple(target.shape)}"
        )
    if not torch.is_floating_point(prediction) or not torch.is_floating_point(target):
        raise TypeError("prediction and target must have floating-point dtypes")
    if beta < 0:
        raise ValueError(f"beta must be non-negative, got {beta}")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    element_loss = F.smooth_l1_loss(prediction, target, reduction="none", beta=beta)
    if valid_mask is None and weight is None:
        return element_loss.mean()

    combined_weight = torch.ones_like(prediction)
    if valid_mask is not None:
        mask = _broadcast_to_prediction(valid_mask, prediction, "valid_mask")
        combined_weight = combined_weight * mask
    if weight is not None:
        broadcast_weight = _broadcast_to_prediction(weight, prediction, "weight")
        if torch.any(broadcast_weight < 0):
            raise ValueError("weight must be non-negative")
        combined_weight = combined_weight * broadcast_weight
    if not torch.isfinite(combined_weight).all():
        raise ValueError("mask and weight must contain only finite values")

    numerator = (element_loss * combined_weight).sum()
    denominator = combined_weight.sum()
    return numerator / (denominator + prediction.new_tensor(eps))


def anchor_offset_supervision_loss(
    predicted_offsets: dict[str, torch.Tensor],
    target_offsets: dict[str, torch.Tensor],
    valid_mask: torch.Tensor | None = None,
    cloth_region_weight: torch.Tensor | None = None,
    xyz_weight: float = 1.0,
    scaling_weight: float = 1.0,
    opacity_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Supervise any available subset of xyz, scaling, and opacity offsets."""

    reference = _get_reference_tensor(predicted_offsets)
    zero = reference.new_zeros(())
    losses: dict[str, torch.Tensor] = {}
    output_names = {
        "delta_xyz": "xyz",
        "delta_scaling": "scaling",
        "delta_opacity": "opacity",
    }
    for field, output_name in output_names.items():
        if field not in target_offsets:
            losses[output_name] = zero
            continue
        if field not in predicted_offsets:
            raise KeyError(f"predicted_offsets is missing target field: {field}")
        losses[output_name] = masked_smooth_l1_loss(
            predicted_offsets[field],
            target_offsets[field],
            valid_mask=valid_mask,
            weight=cloth_region_weight,
        )

    losses["total"] = (
        xyz_weight * losses["xyz"]
        + scaling_weight * losses["scaling"]
        + opacity_weight * losses["opacity"]
    )
    return losses


def anchor_graph_smoothness_loss(
    anchor_offsets: torch.Tensor,
    anchor_edges: torch.Tensor,
    edge_weights: torch.Tensor | None = None,
    norm: str = "l2",
) -> torch.Tensor:
    """Penalize offset discontinuities over an anchor graph."""

    if not isinstance(anchor_offsets, torch.Tensor) or anchor_offsets.ndim != 2:
        raise ValueError("anchor_offsets must be a Tensor[A, C]")
    if not isinstance(anchor_edges, torch.Tensor):
        raise TypeError("anchor_edges must be a torch.Tensor")
    if anchor_edges.ndim != 2 or anchor_edges.shape[1] != 2:
        raise ValueError(f"anchor_edges must have shape [E, 2], got {tuple(anchor_edges.shape)}")
    if anchor_edges.dtype != torch.long:
        raise TypeError("anchor_edges must have dtype torch.int64")
    if norm not in {"l1", "l2"}:
        raise ValueError("norm must be 'l1' or 'l2'")
    if anchor_edges.shape[0] == 0:
        return anchor_offsets.sum() * 0

    edges = anchor_edges.to(device=anchor_offsets.device)
    num_anchors = anchor_offsets.shape[0]
    if torch.any(edges < 0) or torch.any(edges >= num_anchors):
        raise IndexError(f"anchor edge indices must be in [0, {num_anchors})")
    differences = anchor_offsets[edges[:, 0]] - anchor_offsets[edges[:, 1]]
    order = 1 if norm == "l1" else 2
    edge_losses = torch.linalg.vector_norm(differences, ord=order, dim=-1)
    if edge_weights is None:
        return edge_losses.mean()

    if not isinstance(edge_weights, torch.Tensor):
        raise TypeError("edge_weights must be a torch.Tensor")
    if edge_weights.shape == (anchor_edges.shape[0], 1):
        edge_weights = edge_weights.squeeze(-1)
    if edge_weights.shape != (anchor_edges.shape[0],):
        raise ValueError(
            f"edge_weights must have shape [E] or [E, 1], got {tuple(edge_weights.shape)}"
        )
    weights = edge_weights.to(device=anchor_offsets.device, dtype=anchor_offsets.dtype)
    if not torch.isfinite(weights).all() or torch.any(weights < 0):
        raise ValueError("edge_weights must be finite and non-negative")
    return (edge_losses * weights).sum() / (weights.sum() + anchor_offsets.new_tensor(1e-8))


def non_clothing_region_loss(
    offsets: dict[str, torch.Tensor],
    cloth_region_weight: torch.Tensor,
    xyz_weight: float = 1.0,
    scaling_weight: float = 1.0,
    opacity_weight: float = 1.0,
    outside_threshold: float | None = None,
) -> dict[str, torch.Tensor]:
    """Penalize offsets outside the provided soft clothing region."""

    reference = _get_reference_tensor(offsets)
    if not isinstance(cloth_region_weight, torch.Tensor):
        raise TypeError("cloth_region_weight must be a torch.Tensor")
    region = cloth_region_weight.to(device=reference.device, dtype=reference.dtype)
    if not torch.isfinite(region).all() or torch.any(region < 0) or torch.any(region > 1):
        raise ValueError("cloth_region_weight values must be finite and in [0, 1]")
    if outside_threshold is None:
        # Original soft complementary weighting.
        noncloth_weight = 1 - region
    else:
        threshold = float(outside_threshold)
        if threshold < 0 or threshold > 1:
            raise ValueError(
                "outside_threshold must be within [0, 1], "
                f"got {threshold}"
            )

        # Penalize only anchors that are genuinely outside the
        # propagated clothing region. This avoids shrinking valid
        # clothing offsets in soft transition regions.
        noncloth_weight = (region <= threshold).to(
            dtype=reference.dtype
        )

    losses = _zero_offset_loss_dict(reference)
    output_names = {
        "delta_xyz": "xyz",
        "delta_scaling": "scaling",
        "delta_opacity": "opacity",
    }
    for field, output_name in output_names.items():
        if field in offsets:
            losses[output_name] = masked_smooth_l1_loss(
                offsets[field],
                torch.zeros_like(offsets[field]),
                weight=noncloth_weight,
            )
    losses["total"] = (
        xyz_weight * losses["xyz"]
        + scaling_weight * losses["scaling"]
        + opacity_weight * losses["opacity"]
    )
    return losses


def offset_magnitude_regularization(
    offsets: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return per-field mean-square offset magnitudes and their sum."""

    reference = _get_reference_tensor(offsets)
    losses = _zero_offset_loss_dict(reference)
    output_names = {
        "delta_xyz": "xyz",
        "delta_scaling": "scaling",
        "delta_opacity": "opacity",
    }
    for field, output_name in output_names.items():
        if field in offsets:
            losses[output_name] = offsets[field].square().mean()
    losses["total"] = losses["xyz"] + losses["scaling"] + losses["opacity"]
    return losses


def clothing_embedding_consistency_loss(
    embedding_a: torch.Tensor,
    embedding_b: torch.Tensor,
) -> torch.Tensor:
    """Return MSE between two embeddings of the same clothing observation set."""

    if not isinstance(embedding_a, torch.Tensor) or not isinstance(
        embedding_b, torch.Tensor
    ):
        raise TypeError("embedding_a and embedding_b must be torch.Tensor instances")
    if embedding_a.shape != embedding_b.shape:
        raise ValueError(
            "embedding shapes must match, got "
            f"{tuple(embedding_a.shape)} and {tuple(embedding_b.shape)}"
        )
    if embedding_a.numel() == 0:
        raise ValueError("clothing embeddings cannot be empty")
    if not torch.is_floating_point(embedding_a) or not torch.is_floating_point(
        embedding_b
    ):
        raise TypeError("clothing embeddings must have floating-point dtypes")
    if embedding_a.device != embedding_b.device or embedding_a.dtype != embedding_b.dtype:
        raise ValueError("clothing embeddings must have matching device and dtype")
    if not torch.isfinite(embedding_a).all() or not torch.isfinite(embedding_b).all():
        raise ValueError("clothing embeddings contain NaN or Inf")
    return F.mse_loss(embedding_a, embedding_b)


def clothing_offset_consistency_loss(
    offsets_a: dict[str, torch.Tensor],
    offsets_b: dict[str, torch.Tensor],
    valid_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compare two same-clothing anchor predictions with masked Smooth L1."""

    if not isinstance(offsets_a, dict) or not isinstance(offsets_b, dict):
        raise TypeError("offsets_a and offsets_b must be dictionaries")
    missing_a = [field for field in _OFFSET_FIELDS if field not in offsets_a]
    missing_b = [field for field in _OFFSET_FIELDS if field not in offsets_b]
    if missing_a or missing_b:
        raise KeyError(
            f"offset consistency requires {_OFFSET_FIELDS}; "
            f"missing_a={missing_a}, missing_b={missing_b}"
        )
    losses = {
        "xyz": masked_smooth_l1_loss(
            offsets_a["delta_xyz"],
            offsets_b["delta_xyz"],
            valid_mask=valid_mask,
        ),
        "scaling": masked_smooth_l1_loss(
            offsets_a["delta_scaling"],
            offsets_b["delta_scaling"],
            valid_mask=valid_mask,
        ),
        "opacity": masked_smooth_l1_loss(
            offsets_a["delta_opacity"],
            offsets_b["delta_opacity"],
            valid_mask=valid_mask,
        ),
    }
    losses["total"] = losses["xyz"] + losses["scaling"] + losses["opacity"]
    return losses


def anchor_feature_consistency_loss(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    visibility_a: torch.Tensor | None = None,
    visibility_b: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return joint-visibility weighted MSE for two per-anchor feature sets."""

    if not isinstance(features_a, torch.Tensor) or not isinstance(
        features_b, torch.Tensor
    ):
        raise TypeError("features_a and features_b must be torch.Tensor instances")
    if features_a.ndim != 2 or features_a.shape != features_b.shape:
        raise ValueError("features_a and features_b must have matching shape [A,D]")
    if min(features_a.shape) <= 0:
        raise ValueError("anchor feature tensors must be non-empty")
    if not torch.is_floating_point(features_a) or not torch.is_floating_point(features_b):
        raise TypeError("anchor feature tensors must have floating-point dtypes")
    if features_a.device != features_b.device or features_a.dtype != features_b.dtype:
        raise ValueError("anchor feature tensors must have matching device and dtype")
    if not torch.isfinite(features_a).all() or not torch.isfinite(features_b).all():
        raise ValueError("anchor feature tensors contain NaN or Inf")
    if eps <= 0:
        raise ValueError("eps must be positive")

    num_anchors = features_a.shape[0]
    weight = features_a.new_ones(num_anchors, 1)
    for name, visibility in (
        ("visibility_a", visibility_a),
        ("visibility_b", visibility_b),
    ):
        if visibility is None:
            continue
        if not isinstance(visibility, torch.Tensor) or visibility.shape != (
            num_anchors,
            1,
        ):
            raise ValueError(f"{name} must have shape [A,1]")
        value = visibility.to(device=features_a.device, dtype=features_a.dtype)
        if not torch.isfinite(value).all() or torch.any(value < 0) or torch.any(value > 1):
            raise ValueError(f"{name} values must be finite and in [0,1]")
        weight = weight * value
    squared_error = (features_a - features_b).square().mean(dim=-1, keepdim=True)
    return (squared_error * weight).sum() / (weight.sum() + features_a.new_tensor(eps))


def _broadcast_to_prediction(
    value: torch.Tensor,
    prediction: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    value = value.to(device=prediction.device, dtype=prediction.dtype)
    if value.shape == prediction.shape:
        return value
    if prediction.ndim > 0 and value.shape == prediction.shape[:-1]:
        return value.unsqueeze(-1).expand_as(prediction)
    if prediction.ndim > 0 and value.shape == (*prediction.shape[:-1], 1):
        return value.expand_as(prediction)
    raise ValueError(
        f"{name} shape {tuple(value.shape)} cannot broadcast to prediction "
        f"shape {tuple(prediction.shape)}"
    )


def _get_reference_tensor(offsets: dict[str, torch.Tensor]) -> torch.Tensor:
    if not isinstance(offsets, dict):
        raise TypeError("offsets must be a dict of tensors")
    for field in _OFFSET_FIELDS:
        if field in offsets:
            value = offsets[field]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"offsets[{field!r}] must be a torch.Tensor")
            return value
    raise ValueError(f"offsets must contain at least one of {_OFFSET_FIELDS}")


def _zero_offset_loss_dict(reference: torch.Tensor) -> dict[str, torch.Tensor]:
    zero = reference.new_zeros(())
    return {"xyz": zero, "scaling": zero, "opacity": zero, "total": zero}
