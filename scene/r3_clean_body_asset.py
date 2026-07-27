from __future__ import annotations

from dataclasses import dataclass, replace
import heapq
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


PROVENANCE = {"observed": 0, "mirrored": 1, "geodesic": 2, "regional_prior": 3}
IDENTITY_PARTS = (7, 8, 10, 11, 15, *range(20, 55))
SUPPORT_PARTS = (0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 16, 17, 18, 19)
EXCLUDED_SUPPORT_PARTS = tuple(sorted(set(range(55)).difference(SUPPORT_PARTS)))
LEFT_RIGHT_PARTS = {
    1: 2, 2: 1, 4: 5, 5: 4, 7: 8, 8: 7, 10: 11, 11: 10,
    13: 14, 14: 13, 16: 17, 17: 16, 18: 19, 19: 18, 20: 21, 21: 20,
}


def _finite(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be a finite tensor")
    return value


def density_matched_opacity(reference_opacity: float, reference_count: int, count: int) -> float:
    if not (0.0 < reference_opacity < 1.0) or reference_count <= 0 or count <= 0:
        raise ValueError("opacity and counts must be valid")
    return float(1.0 - (1.0 - reference_opacity) ** (reference_count / count))


def classify_base_shell(
    dominant_part: torch.Tensor,
    surface_distance: torch.Tensor,
    old_garment: torch.Tensor,
    skin_distance: torch.Tensor,
    *,
    body_near_distance: float = 0.02,
    skin_distance_max: float = 0.15,
) -> dict[str, torch.Tensor]:
    part = torch.as_tensor(dominant_part, dtype=torch.long)
    distance = _finite(torch.as_tensor(surface_distance, dtype=torch.float32), "surface_distance")
    old = torch.as_tensor(old_garment, dtype=torch.bool)
    skin = _finite(torch.as_tensor(skin_distance, dtype=torch.float32), "skin_distance")
    if not (part.shape == distance.shape == old.shape == skin.shape) or part.ndim != 1:
        raise ValueError("shell inputs must share shape [N]")
    identity = torch.isin(part, torch.tensor(IDENTITY_PARTS)) & ~old
    old_shell = old & ~identity
    ambiguous = ~(identity | old_shell)
    observed_skin = (distance <= body_near_distance) & (skin <= skin_distance_max) & ~old_shell
    s1_keep = ~old_shell
    s2_keep = identity | observed_skin
    if torch.any(identity & old_shell) or torch.any(identity & ambiguous) or torch.any(old_shell & ambiguous):
        raise AssertionError("shell classes are not mutually exclusive")
    if not torch.all(identity | old_shell | ambiguous):
        raise AssertionError("shell classes do not cover the base")
    return {
        "static_identity": identity,
        "old_garment_shell": old_shell,
        "ambiguous": ambiguous,
        "confirmed_skin": observed_skin,
        "s1_keep": s1_keep,
        "s2_keep": s2_keep,
    }


def _vertex_parts(faces: torch.Tensor, vertex_count: int, vertex_lbs: torch.Tensor) -> torch.Tensor:
    if vertex_lbs.shape[0] != vertex_count or vertex_lbs.shape[1] < 55:
        raise ValueError("formal vertex LBS must have shape [V,>=55]")
    return vertex_lbs[:, :55].argmax(1)


def _adjacency(vertices: np.ndarray, faces: np.ndarray) -> list[list[tuple[int, float]]]:
    neighbors: list[dict[int, float]] = [dict() for _ in range(len(vertices))]
    for tri in faces:
        for left, right in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            a, b = int(left), int(right)
            distance = float(np.linalg.norm(vertices[a] - vertices[b]))
            previous = neighbors[a].get(b)
            if previous is None or distance < previous:
                neighbors[a][b] = distance
                neighbors[b][a] = distance
    return [list(row.items()) for row in neighbors]


def _multi_source_geodesic(
    adjacency: list[list[tuple[int, float]]],
    allowed: np.ndarray,
    sources: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    distance = np.full(len(adjacency), np.inf, dtype=np.float64)
    owner = np.full(len(adjacency), -1, dtype=np.int64)
    queue: list[tuple[float, int]] = []
    for source in sources.tolist():
        if allowed[source]:
            distance[source] = 0.0
            owner[source] = source
            heapq.heappush(queue, (0.0, source))
    while queue:
        current_distance, vertex = heapq.heappop(queue)
        if current_distance != distance[vertex]:
            continue
        for neighbor, weight in adjacency[vertex]:
            if not allowed[neighbor]:
                continue
            candidate = current_distance + weight
            if candidate < distance[neighbor]:
                distance[neighbor] = candidate
                owner[neighbor] = owner[vertex]
                heapq.heappush(queue, (candidate, neighbor))
    return distance, owner


@dataclass(frozen=True)
class SkinAppearanceField:
    rgb: torch.Tensor
    confidence: torch.Tensor
    provenance: torch.Tensor
    nearest_observed_distance: torch.Tensor
    body_part: torch.Tensor
    source_vertex: torch.Tensor

    def validate(self) -> "SkinAppearanceField":
        rgb = _finite(self.rgb, "rgb")
        count = rgb.shape[0]
        if rgb.shape != (count, 3) or torch.any((rgb < 0) | (rgb > 1)):
            raise ValueError("skin RGB must be [V,3] in [0,1]")
        for name in ("confidence", "provenance", "nearest_observed_distance", "body_part", "source_vertex"):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor) or value.shape != (count,):
                raise ValueError(f"{name} must have shape [V]")
        if not set(self.provenance.tolist()).issubset(set(PROVENANCE.values())):
            raise ValueError("unknown skin provenance code")
        if torch.any((self.confidence < 0) | (self.confidence > 1)):
            raise ValueError("skin confidence must be in [0,1]")
        return self


def build_skin_appearance_field(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    vertex_lbs: torch.Tensor,
    observation_vertex: torch.Tensor,
    observation_rgb: torch.Tensor,
    observation_confidence: torch.Tensor,
    *,
    maximum_geodesic_distance: float = 0.35,
) -> SkinAppearanceField:
    vertices = _finite(vertices.detach().float().cpu(), "vertices")
    faces = torch.as_tensor(faces, dtype=torch.long).cpu()
    lbs = _finite(vertex_lbs.detach().float().cpu(), "vertex_lbs")
    observed_vertex = torch.as_tensor(observation_vertex, dtype=torch.long).cpu()
    observed_rgb = _finite(observation_rgb.detach().float().cpu(), "observation_rgb")
    observed_weight = _finite(observation_confidence.detach().float().cpu(), "observation_confidence")
    if observed_rgb.shape != (observed_vertex.numel(), 3) or observed_weight.shape != observed_vertex.shape:
        raise ValueError("observation arrays have incompatible shapes")
    if observed_vertex.numel() == 0 or torch.any((observed_vertex < 0) | (observed_vertex >= len(vertices))):
        raise ValueError("observations must reference valid subject02 surface vertices")
    if torch.any((observed_rgb < 0) | (observed_rgb > 1)) or torch.any(observed_weight <= 0):
        raise ValueError("invalid observed subject02 sample")
    count = len(vertices)
    part = _vertex_parts(faces, count, lbs)
    weighted_rgb = torch.zeros(count, 3)
    total_weight = torch.zeros(count)
    weighted_rgb.index_add_(0, observed_vertex, observed_rgb * observed_weight[:, None])
    total_weight.index_add_(0, observed_vertex, observed_weight)
    has_observation = total_weight > 0
    rgb = torch.zeros(count, 3)
    rgb[has_observation] = weighted_rgb[has_observation] / total_weight[has_observation, None]
    confidence = torch.zeros(count)
    confidence[has_observation] = (1.0 - torch.exp(-total_weight[has_observation] / 2.0)).clamp(0.80, 1.0)
    provenance = torch.full((count,), PROVENANCE["regional_prior"], dtype=torch.long)
    provenance[has_observation] = PROVENANCE["observed"]
    source_vertex = torch.full((count,), -1, dtype=torch.long)
    source_vertex[has_observation] = torch.where(has_observation)[0]
    nearest_distance = torch.full((count,), float("inf"))
    nearest_distance[has_observation] = 0.0

    # Mirror only between paired anatomical regions; no cross-identity or cross-part fill.
    xyz = vertices.numpy()
    for target_part, source_part in LEFT_RIGHT_PARTS.items():
        target = torch.where((part == target_part) & ~has_observation)[0]
        source = torch.where((part == source_part) & has_observation)[0]
        if target.numel() == 0 or source.numel() == 0:
            continue
        reflected = vertices[target].clone(); reflected[:, 0] *= -1
        source_xyz = vertices[source]
        distance_chunks: list[torch.Tensor] = []
        local_chunks: list[torch.Tensor] = []
        for start in range(0, len(reflected), 1024):
            distances = torch.cdist(reflected[start:start + 1024], source_xyz)
            value, index = distances.min(1)
            distance_chunks.append(value); local_chunks.append(index)
        distance = torch.cat(distance_chunks)
        local = torch.cat(local_chunks)
        accepted = distance <= 0.08
        if not accepted.any():
            continue
        destination = target[accepted]
        origin = source[local[accepted]]
        rgb[destination] = rgb[origin]
        confidence[destination] = (confidence[origin] * 0.75).clamp_max(0.79)
        provenance[destination] = PROVENANCE["mirrored"]
        source_vertex[destination] = origin
        nearest_distance[destination] = distance[accepted]

    adjacency = _adjacency(xyz, faces.numpy())
    seeded = provenance != PROVENANCE["regional_prior"]
    for anatomical_part in torch.unique(part).tolist():
        allowed = (part.numpy() == anatomical_part)
        sources = torch.where((part == anatomical_part) & seeded)[0].numpy()
        if len(sources) == 0:
            continue
        distances, owners = _multi_source_geodesic(adjacency, allowed, sources)
        target_np = allowed & (provenance.numpy() == PROVENANCE["regional_prior"]) & np.isfinite(distances)
        target_np &= distances <= maximum_geodesic_distance
        destination = torch.from_numpy(np.where(target_np)[0])
        if destination.numel() == 0:
            continue
        origin = torch.from_numpy(owners[target_np])
        rgb[destination] = rgb[origin]
        decay = torch.exp(-torch.from_numpy(distances[target_np]).float() / max(maximum_geodesic_distance, 1e-6))
        confidence[destination] = (confidence[origin] * 0.55 * decay).clamp(0.25, 0.59)
        provenance[destination] = PROVENANCE["geodesic"]
        source_vertex[destination] = origin
        nearest_distance[destination] = torch.from_numpy(distances[target_np]).float()

    # Remaining regions receive only a bounded low-frequency subject02 regional prior.
    observed_colors = observed_rgb[observed_weight >= torch.quantile(observed_weight, 0.25)]
    global_median = observed_colors.median(0).values
    remaining = provenance == PROVENANCE["regional_prior"]
    y = vertices[:, 1]
    y_normalized = (y - y.min()) / (y.max() - y.min()).clamp_min(1e-6)
    luminance_modulation = (0.94 + 0.10 * y_normalized).unsqueeze(1)
    rgb[remaining] = (global_median * luminance_modulation[remaining]).clamp(0, 1)
    confidence[remaining] = 0.15
    source_vertex[remaining] = -1
    nearest_distance[remaining] = float("inf")
    return SkinAppearanceField(rgb, confidence, provenance, nearest_distance, part, source_vertex).validate()


def _matrix_to_quaternion_wxyz(matrix: torch.Tensor) -> torch.Tensor:
    value = matrix.detach().float().cpu()
    if value.ndim != 3 or value.shape[1:] != (3, 3):
        raise ValueError("rotation matrix must have shape [N,3,3]")
    m00, m01, m02 = value[:, 0, 0], value[:, 0, 1], value[:, 0, 2]
    m10, m11, m12 = value[:, 1, 0], value[:, 1, 1], value[:, 1, 2]
    m20, m21, m22 = value[:, 2, 0], value[:, 2, 1], value[:, 2, 2]
    candidates = torch.stack((
        torch.stack((1 + m00 + m11 + m22, m21 - m12, m02 - m20, m10 - m01), 1),
        torch.stack((m21 - m12, 1 + m00 - m11 - m22, m10 + m01, m02 + m20), 1),
        torch.stack((m02 - m20, m10 + m01, 1 - m00 + m11 - m22, m12 + m21), 1),
        torch.stack((m10 - m01, m02 + m20, m12 + m21, 1 - m00 - m11 + m22), 1),
    ), 1)
    magnitude = torch.sqrt(torch.clamp(torch.stack((
        1 + m00 + m11 + m22,
        1 + m00 - m11 - m22,
        1 - m00 + m11 - m22,
        1 - m00 - m11 + m22,
    ), 1), min=0.0))
    candidates = candidates / (2.0 * magnitude.clamp_min(1e-8)[:, :, None])
    row = torch.arange(len(value))
    result = candidates[row, magnitude.argmax(1)]
    result = F.normalize(result, dim=1, eps=1e-12)
    result[result[:, 0] < 0] *= -1
    return result


@dataclass(frozen=True)
class CleanBodySupport:
    canonical_xyz: torch.Tensor
    canonical_normals: torch.Tensor
    log_scaling: torch.Tensor
    quaternion_wxyz: torch.Tensor
    opacity_logit: torch.Tensor
    sh0: torch.Tensor
    shN: torch.Tensor
    body_part: torch.Tensor
    face_indices: torch.Tensor
    barycentric: torch.Tensor
    lbs_weights: torch.Tensor
    skin_confidence: torch.Tensor
    skin_provenance: torch.Tensor
    metadata: Mapping[str, Any]

    @property
    def count(self) -> int:
        return int(self.canonical_xyz.shape[0])

    def validate(self) -> "CleanBodySupport":
        count = self.count
        expected = {
            "canonical_xyz": (count, 3), "canonical_normals": (count, 3),
            "log_scaling": (count, 3), "quaternion_wxyz": (count, 4),
            "opacity_logit": (count,), "sh0": (count, 1, 3), "shN": (count, 0, 3),
            "body_part": (count,), "face_indices": (count,), "barycentric": (count, 3),
            "lbs_weights": (count, 55), "skin_confidence": (count,), "skin_provenance": (count,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"{name} contains NaN/Inf")
            if value.requires_grad or isinstance(value, torch.nn.Parameter):
                raise ValueError("clean body support must remain frozen")
        if not set(self.body_part.tolist()).issubset(set(SUPPORT_PARTS)):
            raise ValueError("support includes face, hands, or feet")
        if not torch.allclose(self.barycentric.sum(1), torch.ones(count), atol=1e-5, rtol=0):
            raise ValueError("invalid barycentric coordinates")
        if not torch.allclose(self.lbs_weights.sum(1), torch.ones(count), atol=1e-5, rtol=0):
            raise ValueError("invalid formal LBS weights")
        if not torch.allclose(torch.linalg.vector_norm(self.quaternion_wxyz, dim=1), torch.ones(count), atol=1e-5, rtol=0):
            raise ValueError("rotation quaternion must be normalized wxyz")
        if self.shN.numel() != 0:
            raise ValueError("SHN must be disabled")
        return self

    def to(self, device: torch.device | str, dtype: torch.dtype = torch.float32) -> "CleanBodySupport":
        values: dict[str, Any] = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, torch.Tensor):
                target_dtype = dtype if value.is_floating_point() else value.dtype
                values[name] = value.to(device=device, dtype=target_dtype).detach()
            else:
                values[name] = value
        return replace(self, **values).validate()

    def tensor_dict(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__ if isinstance(getattr(self, name), torch.Tensor)}


def sample_clean_body_support(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    vertex_lbs: torch.Tensor,
    skin_field: SkinAppearanceField,
    *,
    count: int,
    seed: int,
    inward_offset: float,
    opacity: float,
    tangent_scale_multiplier: float = 1.12,
    normal_scale: float = 0.0015,
) -> CleanBodySupport:
    vertices = _finite(vertices.detach().float().cpu(), "vertices")
    faces = torch.as_tensor(faces, dtype=torch.long).cpu()
    weights = _finite(vertex_lbs.detach().float().cpu()[:, :55], "vertex_lbs")
    field = skin_field.validate()
    if field.rgb.shape[0] != vertices.shape[0] or not 0 < opacity < 1 or count <= 0:
        raise ValueError("invalid support inputs")
    tri = vertices[faces]
    cross = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
    twice_area = torch.linalg.vector_norm(cross, dim=1)
    normals = F.normalize(cross, dim=1, eps=1e-12)
    face_weights = weights[faces].mean(1)
    dominant = face_weights.argmax(1)
    allowed = torch.isin(dominant, torch.tensor(SUPPORT_PARTS)) & (twice_area > 1e-12)
    candidates = torch.where(allowed)[0]
    if candidates.numel() == 0:
        raise RuntimeError("no eligible clean-body surface faces")
    # Area weighting plus a bounded curvature proxy based on face-normal disagreement.
    vertex_normal_sum = torch.zeros_like(vertices)
    for slot in range(3):
        vertex_normal_sum.index_add_(0, faces[:, slot], normals * twice_area[:, None])
    vertex_normals = F.normalize(vertex_normal_sum, dim=1, eps=1e-12)
    curvature = 1.0 - (vertex_normals[faces] * normals[:, None]).sum(2).mean(1).clamp(-1, 1)
    probability = twice_area[candidates] * (1.0 + curvature[candidates].clamp(0, 1))
    probability = probability / probability.sum()
    generator = torch.Generator(device="cpu"); generator.manual_seed(seed)
    chosen = candidates[torch.multinomial(probability, count, replacement=True, generator=generator)]
    u = torch.rand(count, generator=generator); v = torch.rand(count, generator=generator)
    root = torch.sqrt(u)
    bary = torch.stack((1 - root, root * (1 - v), root * v), dim=1)
    chosen_tri = tri[chosen]
    chosen_normals = normals[chosen]
    surface_xyz = (chosen_tri * bary[:, :, None]).sum(1)
    xyz = surface_xyz - float(inward_offset) * chosen_normals
    point_weights = (weights[faces[chosen]] * bary[:, :, None]).sum(1).clamp_min(0)
    point_weights /= point_weights.sum(1, keepdim=True).clamp_min(1e-12)
    tangent = F.normalize(chosen_tri[:, 1] - chosen_tri[:, 0], dim=1, eps=1e-12)
    bitangent = F.normalize(torch.cross(chosen_normals, tangent, dim=1), dim=1, eps=1e-12)
    tangent = F.normalize(torch.cross(bitangent, chosen_normals, dim=1), dim=1, eps=1e-12)
    frame = torch.stack((tangent, bitangent, chosen_normals), dim=2)
    quaternion = _matrix_to_quaternion_wxyz(frame)
    total_area = 0.5 * twice_area[candidates].sum()
    spacing = torch.sqrt(total_area / count).clamp_min(1e-5)
    scaling = torch.tensor([
        float(spacing * tangent_scale_multiplier), float(spacing * tangent_scale_multiplier), float(normal_scale)
    ]).expand(count, -1)
    vertex_color = field.rgb[faces[chosen]]
    rgb = (vertex_color * bary[:, :, None]).sum(1).clamp(0, 1)
    confidence = (field.confidence[faces[chosen]] * bary).sum(1).clamp(0, 1)
    provenance_candidates = field.provenance[faces[chosen]]
    provenance = provenance_candidates.gather(1, bary.argmax(1, keepdim=True)).squeeze(1)
    c0 = 0.28209479177387814
    support = CleanBodySupport(
        canonical_xyz=xyz.detach(), canonical_normals=chosen_normals.detach(),
        log_scaling=scaling.log().detach(), quaternion_wxyz=quaternion.detach(),
        opacity_logit=torch.full((count,), float(torch.logit(torch.tensor(opacity)))),
        sh0=((rgb - 0.5) / c0).reshape(count, 1, 3).detach(),
        shN=torch.empty(count, 0, 3), body_part=dominant[chosen].detach(),
        face_indices=chosen.detach(), barycentric=bary.detach(), lbs_weights=point_weights.detach(),
        skin_confidence=confidence.detach(), skin_provenance=provenance.detach(),
        metadata={
            "count": count, "seed": seed, "sampling": "surface-area and bounded curvature weighted",
            "inward_offset_m": inward_offset, "opacity": opacity,
            "tangent_spacing_m": float(spacing), "normal_scale_m": normal_scale,
            "rotation_convention": "quaternion wxyz, local tangent/bitangent/normal frame",
            "appearance": "subject02-only provenance-aware degree-0 field", "sh_degree": 0,
            "trainable": False,
        },
    )
    return support.validate()


def deform_clean_body_support(
    support: CleanBodySupport,
    rigid_transforms: torch.Tensor,
    Rh: torch.Tensor,
    Th: torch.Tensor,
) -> dict[str, torch.Tensor]:
    support.validate()
    device, dtype = support.canonical_xyz.device, support.canonical_xyz.dtype
    rigid = _finite(torch.as_tensor(rigid_transforms, device=device, dtype=dtype), "rigid_transforms")
    rh = _finite(torch.as_tensor(Rh, device=device, dtype=dtype), "Rh")
    th = _finite(torch.as_tensor(Th, device=device, dtype=dtype), "Th")
    if rigid.shape != (55, 4, 4) or rh.shape != (3, 3) or th.shape != (3,):
        raise ValueError("formal deformation shapes are invalid")
    blended = torch.einsum("nj,jab->nab", support.lbs_weights, rigid)
    homogeneous = F.pad(support.canonical_xyz, (0, 1), value=1)
    body_xyz = torch.einsum("nij,nj->ni", blended, homogeneous)[:, :3]
    xyz = torch.einsum("ij,nj->ni", rh, body_xyz) + th
    linear = torch.einsum("ij,njk->nik", rh, blended[:, :3, :3])
    local_rotation = torch.empty(support.count, 3, 3, device=device, dtype=dtype)
    # Reconstruct covariance directly from formal wxyz frame through scipy-independent tensor math.
    w, x, y, z = support.quaternion_wxyz.unbind(1)
    local_rotation[:, 0, 0] = 1 - 2 * (y*y + z*z); local_rotation[:, 0, 1] = 2 * (x*y - z*w); local_rotation[:, 0, 2] = 2 * (x*z + y*w)
    local_rotation[:, 1, 0] = 2 * (x*y + z*w); local_rotation[:, 1, 1] = 1 - 2 * (x*x + z*z); local_rotation[:, 1, 2] = 2 * (y*z - x*w)
    local_rotation[:, 2, 0] = 2 * (x*z - y*w); local_rotation[:, 2, 1] = 2 * (y*z + x*w); local_rotation[:, 2, 2] = 1 - 2 * (x*x + y*y)
    scaling = support.log_scaling.exp()
    covariance = local_rotation @ torch.diag_embed(scaling.square()) @ local_rotation.transpose(1, 2)
    covariance = linear @ covariance @ linear.transpose(1, 2)
    inverse_transpose = torch.linalg.pinv(linear).transpose(1, 2)
    normals = F.normalize(torch.einsum("nij,nj->ni", inverse_transpose, support.canonical_normals), dim=1, eps=1e-12)
    if not all(torch.isfinite(value).all() for value in (xyz, covariance, normals)):
        raise FloatingPointError("posed clean-body support contains NaN/Inf")
    return {"xyz": xyz, "covariance": covariance, "normals": normals}


def adjudicate_clean_body_pilot(metrics: Mapping[str, Any], visual_status: str) -> dict[str, Any]:
    geometry = bool(metrics.get("geometry_alignment_pass"))
    pose = bool(metrics.get("pose_validation_pass"))
    covered = float(metrics.get("covered_support_visible_fraction_max", float("inf"))) <= 0.01
    old_s1 = float(metrics.get("s1_old_garment_residual_max", float("inf"))) <= 0.01
    old_s2 = float(metrics.get("s2_old_garment_residual_max", float("inf"))) <= 0.01
    leakage_s1 = float(metrics.get("s1_background_leakage_max", float("inf"))) <= 0.01
    leakage_s2 = float(metrics.get("s2_background_leakage_max", float("inf"))) <= 0.01
    identity = bool(metrics.get("identity_preserved"))
    base_exact = bool(metrics.get("base_bitwise_exact"))
    abnormal = float(metrics.get("abnormal_gaussian_fraction_max", float("inf"))) == 0.0
    provenance = bool(metrics.get("provenance_complete"))
    strategy = "S1" if old_s1 and leakage_s1 else "S2" if old_s2 and leakage_s2 else "NONE"
    hard = geometry and pose and covered and identity and base_exact and abnormal and provenance and strategy != "NONE"
    if hard and visual_status == "PASS":
        status = "CLEAN_BODY_ASSET_PILOT_PASS"
    elif not geometry or not pose or not base_exact or visual_status == "FAIL":
        status = "CLEAN_BODY_ASSET_PILOT_FAIL"
    else:
        status = "CLEAN_BODY_ASSET_PILOT_PARTIAL"
    return {
        "status": status,
        "recommended_strategy": strategy,
        "formal_clean_body_base_pilot_allowed": status == "CLEAN_BODY_ASSET_PILOT_PASS",
        "minimal_o00_oracle_allowed": False,
        "module4b_rerun_allowed": False,
        "formal_image_conditioned_training_allowed": False,
    }
