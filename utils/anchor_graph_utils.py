from __future__ import annotations

import hashlib
from collections import deque
from typing import Any

import torch

GRAPH_VERSION = 1


def build_surface_aware_anchor_graph(
    anchors: torch.Tensor,
    graph_k: int = 8,
    topology_neighbors: torch.Tensor | None = None,
    chunk_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Build a deterministic fixed-width graph, preferring supplied topology."""

    if anchors.ndim != 2 or anchors.shape[1] != 3 or not torch.isfinite(anchors).all():
        raise ValueError("anchors must be finite [A,3]")
    if not isinstance(graph_k, int) or graph_k < 1 or graph_k >= anchors.shape[0]:
        raise ValueError("graph_k must satisfy 1 <= K < A")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if topology_neighbors is None:
        indices = _chunked_xyz_knn(anchors, graph_k, chunk_size)
        source = "canonical_xyz_knn_fallback"
    else:
        indices = _expand_topology(anchors, topology_neighbors, graph_k, chunk_size)
        source = "base_anchor_topology_nbr_vt_plus_two_hop"
    neighbor_xyz = anchors[indices]
    distances = torch.linalg.vector_norm(neighbor_xyz - anchors[:, None], dim=-1)
    sigma = distances.median(dim=1, keepdim=True).values.clamp_min(1e-8)
    weights = torch.exp(-distances.square() / (2 * sigma.square()))
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    validate_anchor_graph(indices, weights, anchors.shape[0], graph_k)
    fingerprint = hashlib.sha256(
        indices.detach().cpu().contiguous().numpy().tobytes()
        + weights.detach().cpu().float().contiguous().numpy().tobytes()
    ).hexdigest()
    diagnostics = {
        "graph_version": GRAPH_VERSION, "graph_source": source,
        "anchor_count": anchors.shape[0], "graph_k": graph_k,
        "fingerprint": fingerprint,
        "distance": {
            "min": float(distances.min()), "median": float(distances.median()),
            "mean": float(distances.mean()), "max": float(distances.max()),
        },
        "metadata_paths": [], "normal_filter": "unavailable",
        "body_part_filter": "unavailable", "deterministic": True,
    }
    return indices, weights, diagnostics


def validate_anchor_graph(indices: torch.Tensor, weights: torch.Tensor, anchor_count: int, graph_k: int) -> None:
    if indices.dtype != torch.long or tuple(indices.shape) != (anchor_count, graph_k):
        raise ValueError("graph indices must be long [A,K]")
    if tuple(weights.shape) != (anchor_count, graph_k) or not torch.is_floating_point(weights):
        raise ValueError("graph weights must be floating [A,K]")
    if indices.device != weights.device or not torch.isfinite(weights).all():
        raise ValueError("graph tensors must share a device and weights must be finite")
    if indices.min() < 0 or indices.max() >= anchor_count:
        raise IndexError("graph index out of range")
    if (indices == torch.arange(anchor_count, device=indices.device)[:, None]).any():
        raise ValueError("graph cannot contain self edges")
    if any(len(set(row)) != graph_k for row in indices.detach().cpu().tolist()):
        raise ValueError("graph rows cannot contain duplicate neighbors")
    if not torch.allclose(weights.sum(1), torch.ones(anchor_count, device=weights.device, dtype=weights.dtype), atol=1e-5, rtol=0):
        raise ValueError("graph weight rows must sum to one")


def _expand_topology(anchors, topology, graph_k, chunk_size):
    topology = torch.as_tensor(topology, device=anchors.device, dtype=torch.long)
    if topology.ndim != 2 or topology.shape[0] != anchors.shape[0]:
        raise ValueError("topology_neighbors must have shape [A,K0]")
    if topology.min() < 0 or topology.max() >= anchors.shape[0]:
        raise IndexError("topology neighbor out of range")
    cpu = topology.detach().cpu().tolist(); result=[]
    for root,row in enumerate(cpu):
        selected=[]; seen={root}; queue=deque(sorted(set(row)))
        while queue and len(selected)<graph_k:
            node=queue.popleft()
            if node in seen: continue
            seen.add(node); selected.append(node)
            for nxt in sorted(set(cpu[node])):
                if nxt not in seen: queue.append(nxt)
        if len(selected)<graph_k:
            candidates=_chunked_xyz_knn(anchors[root:root+1], graph_k+1, chunk_size, database=anchors)[0].tolist()
            selected.extend(x for x in candidates if x not in seen and x != root) 
        chosen=torch.tensor(selected[:graph_k],device=anchors.device)
        order=torch.argsort(torch.linalg.vector_norm(anchors[chosen]-anchors[root],dim=-1),stable=True)
        result.append(chosen[order])
    return torch.stack(result).long()


def _chunked_xyz_knn(queries, k, chunk_size, database=None):
    database = queries if database is None else database
    rows=[]
    for start in range(0,queries.shape[0],chunk_size):
        q=queries[start:start+chunk_size]
        distances=torch.cdist(q,database)
        if queries.data_ptr()==database.data_ptr():
            idx=torch.arange(start,start+q.shape[0],device=q.device); distances[torch.arange(q.shape[0],device=q.device),idx]=torch.inf
        rows.append(torch.topk(distances,k,largest=False,sorted=True).indices)
    return torch.cat(rows)
