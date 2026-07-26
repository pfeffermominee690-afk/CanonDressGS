"""Target-independent Gaussian-pool attribution and dual-pool contracts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch


EXCLUSION_PRIORITY = (
    "NONFINITE_OR_INVALID",
    "INVALID_OR_UNASSIGNED_ANCHOR",
    "STABLE_PROTECTED",
    "NO_BASE_VISIBILITY",
    "NO_OLD_CLOTHING_EVIDENCE",
    "OUTSIDE_BASE_GARMENT_ENVELOPE",
    "OTHER_EXPLICIT_REASON",
)
SOURCE_CATEGORIES = (
    "EXCLUDED_PROTECTED",
    "BASE_INVISIBLE",
    "NON_GARMENT_BASE_REGION",
    "OUTSIDE_ENVELOPE",
    "ANCHOR_GROUP_MISMATCH",
    "FORMALLY_NONTRAINABLE",
    "OTHER_EXPLICIT",
)


@dataclass(frozen=True)
class CandidatePools:
    coverage: torch.Tensor
    spill_p0: torch.Tensor
    spill_p1: torch.Tensor
    spill_p2: torch.Tensor
    stable_protected: torch.Tensor


def _mask_from_indices(count: int, indices: torch.Tensor) -> torch.Tensor:
    if indices.ndim != 1 or indices.dtype != torch.long:
        raise TypeError("indices must be a 1D int64 tensor")
    if indices.numel() and (indices.min() < 0 or indices.max() >= count):
        raise IndexError("pool index out of range")
    result = torch.zeros(count, dtype=torch.bool, device=indices.device)
    result[indices] = True
    return result


def stable_protected_mask(
    protected_hits: torch.Tensor,
    garment_envelope_hits: torch.Tensor,
    old_clothing_hits: torch.Tensor,
    *,
    minimum_views: int = 2,
) -> torch.Tensor:
    for name, value in (
        ("protected_hits", protected_hits),
        ("garment_envelope_hits", garment_envelope_hits),
        ("old_clothing_hits", old_clothing_hits),
    ):
        if value.ndim != 1 or not torch.isfinite(value.float()).all():
            raise ValueError(f"{name} must be a finite vector")
    if not (protected_hits.shape == garment_envelope_hits.shape == old_clothing_hits.shape):
        raise ValueError("hit vector shapes differ")
    if minimum_views <= 0:
        raise ValueError("minimum_views must be positive")
    return (
        (protected_hits >= minimum_views)
        & (protected_hits >= garment_envelope_hits)
        & (old_clothing_hits == 0)
    )


def anchor_components(anchor_neighbors: torch.Tensor) -> torch.Tensor:
    """Deterministic undirected connected components for the fixed anchor graph."""

    if anchor_neighbors.ndim != 2 or anchor_neighbors.shape[0] <= 0:
        raise ValueError("anchor_neighbors must have shape [A,K]")
    graph = anchor_neighbors.detach().cpu().long()
    count = graph.shape[0]
    if graph.numel() and (graph.min() < 0 or graph.max() >= count):
        raise IndexError("anchor graph index out of range")
    parent = list(range(count))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            if left > right:
                left, right = right, left
            parent[right] = left

    for source, row in enumerate(graph.tolist()):
        for target in row:
            union(source, int(target))
    roots = [find(index) for index in range(count)]
    canonical = {root: index for index, root in enumerate(sorted(set(roots)))}
    return torch.tensor([canonical[root] for root in roots], dtype=torch.long, device=anchor_neighbors.device)


def build_candidate_pools(
    *,
    current_editable: torch.Tensor,
    formal_trainable: torch.Tensor,
    stable_protected: torch.Tensor,
    dominant_anchor: torch.Tensor,
    anchor_neighbors: torch.Tensor,
) -> CandidatePools:
    count = formal_trainable.numel()
    if formal_trainable.shape != (count,) or stable_protected.shape != (count,) or dominant_anchor.shape != (count,):
        raise ValueError("pool evidence must contain one value per Gaussian")
    current_mask = _mask_from_indices(count, current_editable)
    formal = formal_trainable.detach().bool()
    protected = stable_protected.detach().bool()
    if torch.any(current_mask & protected):
        raise ValueError("frozen coverage pool overlaps stable protected")
    p1_mask = formal & ~protected
    anchor_count = anchor_neighbors.shape[0]
    dominant = dominant_anchor.detach().long()
    if dominant.numel() and (dominant.min() < 0 or dominant.max() >= anchor_count):
        raise IndexError("dominant anchor out of range")
    selected_anchors = torch.zeros(anchor_count, dtype=torch.bool, device=dominant.device)
    selected_anchors[dominant[current_mask]] = True
    expanded = selected_anchors.clone()
    if selected_anchors.any():
        expanded[anchor_neighbors[selected_anchors].reshape(-1).long()] = True
    p2_mask = formal & ~protected & (current_mask | expanded[dominant])
    return CandidatePools(
        coverage=current_editable.detach().clone(),
        spill_p0=current_editable.detach().clone(),
        spill_p1=torch.where(p1_mask)[0],
        spill_p2=torch.where(p2_mask)[0],
        stable_protected=torch.where(protected)[0],
    )


def explicit_membership_reasons(
    *,
    current_editable: torch.Tensor,
    finite_valid: torch.Tensor,
    visible_hits: torch.Tensor,
    old_clothing_hits: torch.Tensor,
    garment_envelope_hits: torch.Tensor,
    stable_protected: torch.Tensor,
    dominant_anchor: torch.Tensor,
    expanded_anchor_mask: torch.Tensor,
) -> tuple[list[str], list[str]]:
    count = visible_hits.numel()
    vectors = (
        finite_valid, old_clothing_hits, garment_envelope_hits,
        stable_protected, dominant_anchor,
    )
    if any(value.shape != (count,) for value in vectors):
        raise ValueError("membership evidence shape mismatch")
    membership = _mask_from_indices(count, current_editable)
    inclusion: list[str] = []
    exclusion: list[str] = []
    for index in range(count):
        if membership[index]:
            if old_clothing_hits[index] > 0:
                include = "BASE_OLD_CLOTHING_SEED"
            elif garment_envelope_hits[index] > 0:
                include = "BASE_GARMENT_ENVELOPE_SEED"
            else:
                include = "ONE_HOP_ANCHOR_GRAPH_EXPANSION"
            inclusion.append(include); exclusion.append("")
            continue
        inclusion.append("")
        anchor = int(dominant_anchor[index])
        if not bool(finite_valid[index]):
            reason = "NONFINITE_OR_INVALID"
        elif anchor < 0 or anchor >= expanded_anchor_mask.numel():
            reason = "INVALID_OR_UNASSIGNED_ANCHOR"
        elif bool(stable_protected[index]):
            reason = "STABLE_PROTECTED"
        elif visible_hits[index] <= 0:
            reason = "NO_BASE_VISIBILITY"
        elif old_clothing_hits[index] <= 0 and garment_envelope_hits[index] <= 0:
            reason = "NO_OLD_CLOTHING_EVIDENCE"
        elif garment_envelope_hits[index] <= 0:
            reason = "OUTSIDE_BASE_GARMENT_ENVELOPE"
        elif not bool(expanded_anchor_mask[anchor]):
            reason = "OTHER_EXPLICIT_REASON:ANCHOR_NOT_REACHED_BY_FROZEN_ONE_HOP_GRAPH"
        else:
            reason = "OTHER_EXPLICIT_REASON:VISIBLE_NONPROTECTED_FAILED_FROZEN_POOL_PREDICATE"
        exclusion.append(reason)
    if any((not inc and not exc) or (inc and exc) for inc, exc in zip(inclusion, exclusion)):
        raise AssertionError("each Gaussian must have exactly one membership reason")
    return inclusion, exclusion


def weighted_pool_recall(
    contributor_indices: Sequence[int] | np.ndarray,
    active_indices: Sequence[int] | np.ndarray,
    active_alpha_mass: Sequence[float] | np.ndarray,
    pool_indices: torch.Tensor,
) -> dict[str, float | int]:
    contributors = np.asarray(contributor_indices, dtype=np.int64)
    active = np.asarray(active_indices, dtype=np.int64)
    mass = np.asarray(active_alpha_mass, dtype=np.float64)
    if active.shape != mass.shape:
        raise ValueError("active indices and alpha mass shapes differ")
    pool = set(int(value) for value in pool_indices.detach().cpu().tolist())
    unique = set(int(value) for value in contributors.tolist())
    covered_unique = unique.intersection(pool)
    npre_hit = sum(int(value) in pool for value in contributors.tolist())
    active_hit = sum(int(value) in pool for value in active.tolist())
    alpha_hit = sum(float(weight) for value, weight in zip(active.tolist(), mass.tolist()) if int(value) in pool)
    return {
        "unique_contributor_count": len(unique),
        "index_recall": len(covered_unique) / max(len(unique), 1),
        "npre_occurrences": int(contributors.size),
        "npre_weighted_recall": npre_hit / max(int(contributors.size), 1),
        "active_occurrences": int(active.size),
        "active_contribution_recall": active_hit / max(int(active.size), 1),
        "alpha_mass": float(mass.sum()),
        "alpha_mass_recall": alpha_hit / max(float(mass.sum()), 1e-12),
    }


def cloud_source_category(
    *,
    stable_protected: bool,
    base_visible: bool,
    garment_body_part: bool,
    envelope_evidence: bool,
    anchor_group_member: bool,
    formal_trainable: bool,
) -> str:
    if stable_protected:
        return "EXCLUDED_PROTECTED"
    if not base_visible:
        return "BASE_INVISIBLE"
    if not garment_body_part:
        return "NON_GARMENT_BASE_REGION"
    if not envelope_evidence:
        return "OUTSIDE_ENVELOPE"
    if anchor_group_member:
        return "ANCHOR_GROUP_MISMATCH"
    if not formal_trainable:
        return "FORMALLY_NONTRAINABLE"
    return "OTHER_EXPLICIT"


def pool_sha256(indices: torch.Tensor) -> str:
    array = indices.detach().cpu().numpy().astype(np.int64, copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def qualify_dual_pool(
    *,
    coverage_qualification: Mapping[str, object],
    cloud_recall: Mapping[str, float],
    cloud_proxy_spearman: float,
    cloud_to_background_ratio: float,
    stable_protected_overlap: int,
    protected_loss_pixels: int,
    o01_background_risk_fraction: float,
) -> dict[str, object]:
    gates = {
        "coverage": coverage_qualification.get("status") == "PASS",
        "cloud_npre_recall": float(cloud_recall["npre_weighted_recall"]) >= 0.90,
        "cloud_active_recall": float(cloud_recall["active_contribution_recall"]) >= 0.90,
        "cloud_alpha_recall": float(cloud_recall["alpha_mass_recall"]) >= 0.90,
        "cloud_proxy_positive_correlation": cloud_proxy_spearman > 0,
        "cloud_above_background": cloud_to_background_ratio > 1.0,
        "stable_protected_overlap": stable_protected_overlap == 0,
        "protected_loss_empty": protected_loss_pixels == 0,
        "o01_background_safe": o01_background_risk_fraction <= 0.05,
    }
    return {"status": "PASS" if all(gates.values()) else "FAIL", "gates": gates}
