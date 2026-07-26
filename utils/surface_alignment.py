from __future__ import annotations

from collections import deque
from itertools import combinations

import torch
import torch.nn.functional as F


def project_points_to_mesh_surface(
    points: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    max_distance: float | None = None,
    chunk_size: int = 4096,
) -> dict[str, torch.Tensor]:
    """Project points to their closest mesh triangles using bounded chunks."""

    _validate_mesh_inputs(points, vertices, faces)
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if max_distance is not None and max_distance < 0:
        raise ValueError("max_distance must be non-negative")

    mesh_vertices = vertices.to(device=points.device, dtype=points.dtype)
    mesh_faces = faces.to(device=points.device)
    triangles = mesh_vertices[mesh_faces]
    num_faces = triangles.shape[0]
    face_chunk_size = min(num_faces, max(1, 250_000 // min(chunk_size, points.shape[0])))

    all_faces = []
    all_barycentric = []
    all_surface_points = []
    all_distances = []
    for point_start in range(0, points.shape[0], chunk_size):
        point_chunk = points[point_start : point_start + chunk_size]
        chunk_count = point_chunk.shape[0]
        best_squared = torch.full(
            (chunk_count,),
            float("inf"),
            device=points.device,
            dtype=points.dtype,
        )
        best_faces = torch.full((chunk_count,), -1, device=points.device, dtype=torch.long)
        best_barycentric = torch.zeros(chunk_count, 3, device=points.device, dtype=points.dtype)
        best_surface = torch.zeros(chunk_count, 3, device=points.device, dtype=points.dtype)

        for face_start in range(0, num_faces, face_chunk_size):
            triangle_chunk = triangles[face_start : face_start + face_chunk_size]
            surface, barycentric = _closest_points_on_triangles(point_chunk, triangle_chunk)
            squared = (point_chunk[:, None, :] - surface).square().sum(dim=-1)
            local_squared, local_indices = squared.min(dim=1)
            rows = torch.arange(chunk_count, device=points.device)
            update = local_squared < best_squared
            best_squared = torch.where(update, local_squared, best_squared)
            best_faces = torch.where(update, local_indices + face_start, best_faces)
            best_barycentric = torch.where(
                update[:, None],
                barycentric[rows, local_indices],
                best_barycentric,
            )
            best_surface = torch.where(
                update[:, None],
                surface[rows, local_indices],
                best_surface,
            )

        all_faces.append(best_faces)
        all_barycentric.append(best_barycentric)
        all_surface_points.append(best_surface)
        all_distances.append(torch.sqrt(best_squared.clamp_min(0))[:, None])

    face_indices = torch.cat(all_faces)
    barycentric = torch.cat(all_barycentric)
    surface_points = torch.cat(all_surface_points)
    distances = torch.cat(all_distances)
    valid_mask = torch.ones_like(distances)
    if max_distance is not None:
        valid_mask = (distances <= max_distance).to(dtype=points.dtype)
    output = {
        "face_indices": face_indices,
        "barycentric": barycentric,
        "surface_points": surface_points,
        "distances": distances,
        "valid_mask": valid_mask,
    }
    for name, value in output.items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f"mesh projection produced non-finite {name}")
    return output


def build_face_adjacency(faces: torch.Tensor) -> list[list[int]]:
    """Build sparse edge-sharing face adjacency without a dense face matrix."""

    if not isinstance(faces, torch.Tensor) or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape [F, 3]")
    if faces.dtype != torch.long:
        raise TypeError("faces must have dtype torch.int64")
    face_list = faces.detach().cpu().tolist()
    adjacency = [set() for _ in face_list]
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for face_id, face in enumerate(face_list):
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = (min(first, second), max(first, second))
            edge_to_faces.setdefault(edge, []).append(face_id)
    for shared_faces in edge_to_faces.values():
        for first, second in combinations(shared_faces, 2):
            adjacency[first].add(second)
            adjacency[second].add(first)
    return [sorted(neighbors) for neighbors in adjacency]


def compute_face_hop_neighborhood(
    face_adjacency: list[list[int]],
    source_face: int,
    max_hops: int,
) -> torch.Tensor:
    """Return all faces reachable within the requested edge-sharing hops."""

    distances = _face_hop_distances(face_adjacency, source_face, max_hops)
    return torch.tensor(sorted(distances), dtype=torch.long)


def aggregate_gaussians_to_surface_anchors(
    gaussians: dict[str, torch.Tensor],
    gaussian_projection: dict[str, torch.Tensor],
    anchor_projection: dict[str, torch.Tensor],
    faces: torch.Tensor,
    face_adjacency: list[list[int]],
    max_face_hops: int = 1,
    surface_temperature: float = 0.05,
    normal_distance_temperature: float = 0.03,
    max_surface_distance: float | None = None,
    k_fallback: int = 8,
) -> dict[str, torch.Tensor]:
    """Aggregate Gaussian properties using mesh topology before spatial fallback."""

    xyz = _require_matrix(gaussians, "xyz", 3)
    scale = _require_matrix(gaussians, "scale", 3, xyz.shape[0])
    opacity = _require_matrix(gaussians, "opacity", 1, xyz.shape[0])
    gaussian_projection = _validate_projection(gaussian_projection, xyz.shape[0], xyz)
    anchor_surface = _require_matrix(anchor_projection, "surface_points", 3)
    anchor_projection = _validate_projection(anchor_projection, anchor_surface.shape[0], xyz)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype != torch.long:
        raise ValueError("faces must be int64 Tensor[F, 3]")
    if len(face_adjacency) != faces.shape[0]:
        raise ValueError("face_adjacency length must match face count")
    if max_face_hops < 0:
        raise ValueError("max_face_hops must be non-negative")
    if surface_temperature <= 0 or normal_distance_temperature <= 0:
        raise ValueError("surface and normal distance temperatures must be positive")
    if max_surface_distance is not None and max_surface_distance < 0:
        raise ValueError("max_surface_distance must be non-negative")
    if not isinstance(k_fallback, int) or k_fallback <= 0:
        raise ValueError("k_fallback must be a positive integer")

    gaussian_surface = gaussian_projection["surface_points"]
    gaussian_faces = gaussian_projection["face_indices"]
    gaussian_normal_distance = gaussian_projection["distances"].squeeze(-1)
    gaussian_valid = gaussian_projection["valid_mask"].squeeze(-1) > 0
    anchor_faces = anchor_projection["face_indices"]
    anchor_valid = anchor_projection["valid_mask"].squeeze(-1) > 0
    gaussian_residual = xyz - gaussian_surface

    result_lists: dict[str, list[torch.Tensor]] = {
        "aggregated_xyz": [],
        "aggregated_scale": [],
        "aggregated_opacity": [],
        "aggregated_surface_residual": [],
        "nearest_surface_distance": [],
        "candidate_count": [],
        "valid_mask": [],
        "fallback_mask": [],
    }
    for anchor_id in range(anchor_surface.shape[0]):
        face_id = int(anchor_faces[anchor_id].item())
        hop_distances = _face_hop_distances(face_adjacency, face_id, max_face_hops)
        same_face_mask = gaussian_valid & (gaussian_faces == face_id)
        if torch.any(same_face_mask):
            candidate_mask = same_face_mask
        else:
            neighborhood = torch.tensor(
                list(hop_distances),
                device=xyz.device,
                dtype=torch.long,
            )
            candidate_mask = gaussian_valid & torch.isin(gaussian_faces, neighborhood)
        candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)
        if candidate_indices.numel() > 0:
            surface_distances = torch.linalg.vector_norm(
                gaussian_surface[candidate_indices] - anchor_surface[anchor_id],
                dim=-1,
            )
            if max_surface_distance is not None:
                keep = surface_distances <= max_surface_distance
                candidate_indices = candidate_indices[keep]
                surface_distances = surface_distances[keep]

        used_fallback = candidate_indices.numel() == 0
        candidate_count = candidate_indices.numel()
        if used_fallback:
            fallback_count = min(k_fallback, xyz.shape[0])
            euclidean_distances = torch.linalg.vector_norm(
                xyz - anchor_surface[anchor_id],
                dim=-1,
            )
            nearest_distances, candidate_indices = torch.topk(
                euclidean_distances,
                k=fallback_count,
                largest=False,
                sorted=True,
            )
            weights = torch.softmax(-nearest_distances / surface_temperature, dim=0)
            nearest_distance = nearest_distances[0]
        else:
            selected_faces = gaussian_faces[candidate_indices].detach().cpu().tolist()
            hop_values = torch.tensor(
                [hop_distances[int(selected)] for selected in selected_faces],
                device=xyz.device,
                dtype=xyz.dtype,
            )
            normal_distances = gaussian_normal_distance[candidate_indices]
            scores = (
                -surface_distances / surface_temperature
                - normal_distances / normal_distance_temperature
                - hop_values
            )
            weights = torch.softmax(scores, dim=0)
            nearest_distance = surface_distances.min()

        def weighted(values: torch.Tensor) -> torch.Tensor:
            return (values[candidate_indices] * weights[:, None]).sum(dim=0)

        is_valid = anchor_valid[anchor_id]
        if max_surface_distance is not None:
            is_valid = is_valid & (nearest_distance <= max_surface_distance)
        result_lists["aggregated_xyz"].append(weighted(xyz))
        result_lists["aggregated_scale"].append(weighted(scale))
        result_lists["aggregated_opacity"].append(weighted(opacity))
        result_lists["aggregated_surface_residual"].append(weighted(gaussian_residual))
        result_lists["nearest_surface_distance"].append(nearest_distance.reshape(1))
        result_lists["candidate_count"].append(xyz.new_tensor([candidate_count]))
        result_lists["valid_mask"].append(is_valid.to(dtype=xyz.dtype).reshape(1))
        result_lists["fallback_mask"].append(xyz.new_tensor([float(used_fallback)]))

    output = {
        "anchor_xyz": anchor_surface,
        **{key: torch.stack(values) for key, values in result_lists.items()},
        "anchor_face_indices": anchor_faces,
    }
    for name, value in output.items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f"surface aggregation produced non-finite {name}")
    return output


def build_surface_aware_anchor_offset_target(
    base_anchor_params: dict[str, torch.Tensor],
    dressed_aggregation: dict[str, torch.Tensor],
    anchor_projection: dict[str, torch.Tensor],
    gaussian_projection: dict[str, torch.Tensor],
    cloth_region_weight: torch.Tensor | None = None,
    offset_mode: str = "surface_residual",
) -> dict[str, torch.Tensor]:
    """Build canonical clothing residuals relative to the body mesh surface.

    In ``surface_residual`` mode, ``delta_xyz`` is the canonical-space garment
    residual measured from the shared body surface, rather than an absolute
    dressed-point minus anchor-point displacement.
    """

    del gaussian_projection  # Aggregated residuals already encode this projection.
    anchor_xyz = _require_matrix(base_anchor_params, "anchor_xyz", 3)
    base_scale = _require_matrix(base_anchor_params, "scale", 3, anchor_xyz.shape[0])
    base_opacity = _require_matrix(base_anchor_params, "opacity", 1, anchor_xyz.shape[0])
    aggregated_xyz = _require_matrix(
        dressed_aggregation, "aggregated_xyz", 3, anchor_xyz.shape[0]
    )
    aggregated_scale = _require_matrix(
        dressed_aggregation, "aggregated_scale", 3, anchor_xyz.shape[0]
    )
    aggregated_opacity = _require_matrix(
        dressed_aggregation, "aggregated_opacity", 1, anchor_xyz.shape[0]
    )
    aggregated_residual = _require_matrix(
        dressed_aggregation,
        "aggregated_surface_residual",
        3,
        anchor_xyz.shape[0],
    )
    anchor_surface = _require_matrix(
        anchor_projection, "surface_points", 3, anchor_xyz.shape[0]
    )
    valid_mask = _require_matrix(
        dressed_aggregation, "valid_mask", 1, anchor_xyz.shape[0]
    )
    fallback_mask = _require_matrix(
        dressed_aggregation, "fallback_mask", 1, anchor_xyz.shape[0]
    )
    face_indices = dressed_aggregation["anchor_face_indices"]
    if face_indices.shape != (anchor_xyz.shape[0],) or face_indices.dtype != torch.long:
        raise ValueError("anchor_face_indices must be int64 Tensor[A]")
    if offset_mode == "absolute_difference":
        delta_xyz = aggregated_xyz - anchor_xyz
    elif offset_mode == "surface_residual":
        base_surface_residual = anchor_xyz - anchor_surface
        delta_xyz = aggregated_residual - base_surface_residual
    else:
        raise ValueError("offset_mode must be 'absolute_difference' or 'surface_residual'")
    delta_scaling = aggregated_scale - base_scale
    delta_opacity = aggregated_opacity - base_opacity

    if cloth_region_weight is None:
        change = torch.linalg.vector_norm(delta_xyz, dim=-1, keepdim=True)
        change = change + delta_scaling.abs().mean(dim=-1, keepdim=True)
        excess = torch.relu(change - 0.005)
        region = excess / (excess + 0.005)
    else:
        region = cloth_region_weight
        if region.shape == (anchor_xyz.shape[0],):
            region = region[:, None]
        if region.shape != (anchor_xyz.shape[0], 1):
            raise ValueError("cloth_region_weight must have shape [A] or [A, 1]")
        region = region.to(device=anchor_xyz.device, dtype=anchor_xyz.dtype)
        if not torch.isfinite(region).all() or torch.any(region < 0) or torch.any(region > 1):
            raise ValueError("cloth_region_weight must be finite and in [0, 1]")
    region = region * valid_mask

    target = {
        "anchor_xyz": anchor_xyz.clone(),
        "delta_xyz": delta_xyz,
        "delta_scaling": delta_scaling,
        "delta_opacity": delta_opacity,
        "valid_mask": valid_mask,
        "cloth_region_weight": region.clamp(0, 1),
        "fallback_mask": fallback_mask,
        "surface_face_indices": face_indices,
    }
    if "surface_normals" in anchor_projection:
        normals = _require_matrix(
            anchor_projection, "surface_normals", 3, anchor_xyz.shape[0]
        )
        normals = F.normalize(normals, dim=-1, eps=1e-12)
        normal_offset = (delta_xyz * normals).sum(dim=-1, keepdim=True)
        target["normal_offset"] = normal_offset
        target["tangential_offset"] = delta_xyz - normal_offset * normals
    for name, value in target.items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f"surface target produced non-finite {name}")
    return target


def compute_vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Compute winding-dependent unit vertex normals with safe degeneracy handling."""

    if vertices.ndim != 2 or vertices.shape[1] != 3 or not torch.isfinite(vertices).all():
        raise ValueError("vertices must be finite Tensor[V, 3]")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype != torch.long:
        raise ValueError("faces must be int64 Tensor[F, 3]")
    mesh_faces = faces.to(device=vertices.device)
    if torch.any(mesh_faces < 0) or torch.any(mesh_faces >= vertices.shape[0]):
        raise IndexError("faces contain out-of-range vertex indices")
    triangles = vertices[mesh_faces]
    face_normals = torch.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
        dim=-1,
    )
    normals = torch.zeros_like(vertices)
    for corner in range(3):
        normals.index_add_(0, mesh_faces[:, corner], face_normals)
    return F.normalize(normals, dim=-1, eps=1e-12)


def interpolate_face_normals(
    vertex_normals: torch.Tensor,
    faces: torch.Tensor,
    face_indices: torch.Tensor,
    barycentric: torch.Tensor,
) -> torch.Tensor:
    """Barycentrically interpolate and normalize mesh normals at projected points."""

    if vertex_normals.ndim != 2 or vertex_normals.shape[1] != 3:
        raise ValueError("vertex_normals must have shape [V, 3]")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype != torch.long:
        raise ValueError("faces must be int64 Tensor[F, 3]")
    if face_indices.ndim != 1 or face_indices.dtype != torch.long:
        raise ValueError("face_indices must be int64 Tensor[P]")
    if barycentric.shape != (face_indices.shape[0], 3):
        raise ValueError("barycentric must have shape [P, 3]")
    mesh_faces = faces.to(device=vertex_normals.device)
    indices = face_indices.to(device=vertex_normals.device)
    if torch.any(indices < 0) or torch.any(indices >= mesh_faces.shape[0]):
        raise IndexError("face_indices are out of range")
    barycentric = barycentric.to(device=vertex_normals.device, dtype=vertex_normals.dtype)
    corner_normals = vertex_normals[mesh_faces[indices]]
    interpolated = (corner_normals * barycentric[..., None]).sum(dim=1)
    return F.normalize(interpolated, dim=-1, eps=1e-12)


def _closest_points_on_triangles(
    points: torch.Tensor,
    triangles: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    p = points[:, None, :]
    a = triangles[None, :, 0, :]
    b = triangles[None, :, 1, :]
    c = triangles[None, :, 2, :]
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = (ab * ap).sum(dim=-1)
    d2 = (ac * ap).sum(dim=-1)
    bp = p - b
    d3 = (ab * bp).sum(dim=-1)
    d4 = (ac * bp).sum(dim=-1)
    cp = p - c
    d5 = (ab * cp).sum(dim=-1)
    d6 = (ac * cp).sum(dim=-1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    eps = torch.finfo(points.dtype).eps

    denominator = (va + vb + vc).clamp_min(eps)
    face_v = vb / denominator
    face_w = vc / denominator
    barycentric = torch.stack((1 - face_v - face_w, face_v, face_w), dim=-1)

    def replace(mask: torch.Tensor, values: torch.Tensor) -> None:
        nonlocal barycentric
        barycentric = torch.where(mask[..., None], values, barycentric)

    zeros = torch.zeros_like(d1)
    ones = torch.ones_like(d1)
    replace((d1 <= 0) & (d2 <= 0), torch.stack((ones, zeros, zeros), dim=-1))
    replace((d3 >= 0) & (d4 <= d3), torch.stack((zeros, ones, zeros), dim=-1))
    replace((d6 >= 0) & (d5 <= d6), torch.stack((zeros, zeros, ones), dim=-1))

    edge_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    v_ab = d1 / (d1 - d3).clamp_min(eps)
    replace(edge_ab, torch.stack((1 - v_ab, v_ab, zeros), dim=-1))
    edge_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    w_ac = d2 / (d2 - d6).clamp_min(eps)
    replace(edge_ac, torch.stack((1 - w_ac, zeros, w_ac), dim=-1))
    edge_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    w_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6)).clamp_min(eps)
    replace(edge_bc, torch.stack((zeros, 1 - w_bc, w_bc), dim=-1))

    degenerate = torch.linalg.vector_norm(torch.cross(ab, ac, dim=-1), dim=-1) <= eps
    if torch.any(degenerate):
        degenerate_barycentric = _closest_degenerate_barycentric(p, a, b, c, eps)
        replace(degenerate, degenerate_barycentric)
    barycentric = barycentric.clamp_min(0)
    barycentric = barycentric / barycentric.sum(dim=-1, keepdim=True).clamp_min(eps)
    surface = barycentric[..., 0:1] * a + barycentric[..., 1:2] * b + barycentric[..., 2:3] * c
    return surface, barycentric


def _closest_degenerate_barycentric(
    p: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    def segment(first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        direction = second - first
        t = ((p - first) * direction).sum(dim=-1) / direction.square().sum(dim=-1).clamp_min(eps)
        t = t.clamp(0, 1)
        return first + t[..., None] * direction, t

    point_ab, t_ab = segment(a, b)
    point_ac, t_ac = segment(a, c)
    point_bc, t_bc = segment(b, c)
    points = torch.stack((point_ab, point_ac, point_bc), dim=-2)
    squared = (p.unsqueeze(-2) - points).square().sum(dim=-1)
    choice = squared.argmin(dim=-1)
    candidates = torch.stack(
        (
            torch.stack((1 - t_ab, t_ab, torch.zeros_like(t_ab)), dim=-1),
            torch.stack((1 - t_ac, torch.zeros_like(t_ac), t_ac), dim=-1),
            torch.stack((torch.zeros_like(t_bc), 1 - t_bc, t_bc), dim=-1),
        ),
        dim=-2,
    )
    return torch.gather(
        candidates,
        dim=-2,
        index=choice[..., None, None].expand(*choice.shape, 1, 3),
    ).squeeze(-2)


def _face_hop_distances(
    adjacency: list[list[int]],
    source_face: int,
    max_hops: int,
) -> dict[int, int]:
    if not isinstance(max_hops, int) or max_hops < 0:
        raise ValueError("max_hops must be a non-negative integer")
    if not 0 <= source_face < len(adjacency):
        raise IndexError("source_face is out of range")
    distances = {source_face: 0}
    queue = deque([source_face])
    while queue:
        current = queue.popleft()
        if distances[current] == max_hops:
            continue
        for neighbor in adjacency[current]:
            if not 0 <= neighbor < len(adjacency):
                raise IndexError("face_adjacency contains an out-of-range face")
            if neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    return distances


def _validate_mesh_inputs(
    points: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
) -> None:
    for name, value in (("points", points), ("vertices", vertices)):
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != 3:
            raise ValueError(f"{name} must have shape [N, 3]")
        if value.shape[0] == 0 or not torch.is_floating_point(value):
            raise ValueError(f"{name} must be non-empty and floating point")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or Inf")
    if not isinstance(faces, torch.Tensor) or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape [F, 3]")
    if faces.shape[0] == 0 or faces.dtype != torch.long:
        raise ValueError("faces must be non-empty with dtype torch.int64")
    if torch.any(faces < 0) or torch.any(faces >= vertices.shape[0]):
        raise IndexError("faces contain out-of-range vertex indices")


def _validate_projection(
    projection: dict[str, torch.Tensor],
    rows: int,
    reference: torch.Tensor,
) -> dict[str, torch.Tensor]:
    expected = {
        "face_indices": (rows,),
        "barycentric": (rows, 3),
        "surface_points": (rows, 3),
        "distances": (rows, 1),
        "valid_mask": (rows, 1),
    }
    output = dict(projection)
    for key, shape in expected.items():
        if key not in output or not isinstance(output[key], torch.Tensor):
            raise KeyError(f"projection is missing tensor field {key!r}")
        if tuple(output[key].shape) != shape:
            raise ValueError(f"projection {key} has shape {tuple(output[key].shape)}, expected {shape}")
        if key == "face_indices":
            if output[key].dtype != torch.long:
                raise TypeError("projection face_indices must be int64")
            output[key] = output[key].to(device=reference.device)
        else:
            output[key] = output[key].to(device=reference.device, dtype=reference.dtype)
            if not torch.isfinite(output[key]).all():
                raise ValueError(f"projection {key} contains NaN or Inf")
    return output


def _require_matrix(
    values: dict[str, torch.Tensor],
    key: str,
    columns: int,
    rows: int | None = None,
) -> torch.Tensor:
    if key not in values or not isinstance(values[key], torch.Tensor):
        raise KeyError(f"missing tensor field {key!r}")
    value = values[key]
    if value.ndim != 2 or value.shape[1] != columns:
        raise ValueError(f"{key} must have shape [N, {columns}]")
    if rows is not None and value.shape[0] != rows:
        raise ValueError(f"{key} has {value.shape[0]} rows, expected {rows}")
    if value.shape[0] == 0 or not torch.isfinite(value).all():
        raise ValueError(f"{key} must be non-empty and finite")
    return value
