from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import io
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F


LEFT_ARM_PARTS = (16, 18)
RIGHT_ARM_PARTS = (17, 19)
ARM_PARTS = LEFT_ARM_PARTS + RIGHT_ARM_PARTS
HAND_AND_FINGER_PARTS = tuple(range(20, 55))


def _require_tensor(value: Any, name: str, ndim: int, last: int | None = None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != ndim:
        raise ValueError(f"{name} must be a rank-{ndim} tensor")
    if last is not None and value.shape[-1] != last:
        raise ValueError(f"{name} must end in {last}, got {tuple(value.shape)}")
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")
    return value


@dataclass(frozen=True)
class ArmSupportProbe:
    """Frozen, output-only anatomical support used by the R3 diagnostic.

    This is intentionally a plain dataclass rather than an ``nn.Module``.  Its
    tensors are detached and never registered as parameters, so constructing a
    probe cannot create an optimizer-visible training surface.
    """

    canonical_xyz: torch.Tensor
    canonical_normals: torch.Tensor
    canonical_covariance: torch.Tensor
    lbs_weights: torch.Tensor
    face_indices: torch.Tensor
    barycentric: torch.Tensor
    body_part: torch.Tensor
    side: torch.Tensor
    opacity: torch.Tensor
    rgb: torch.Tensor
    metadata: Mapping[str, Any]

    def validate(self, maximum_count: int = 20_000) -> "ArmSupportProbe":
        xyz = _require_tensor(self.canonical_xyz, "canonical_xyz", 2, 3)
        count = xyz.shape[0]
        if count <= 0 or count > maximum_count:
            raise ValueError(f"support count must be in [1,{maximum_count}], got {count}")
        for name, value, ndim, last in (
            ("canonical_normals", self.canonical_normals, 2, 3),
            ("canonical_covariance", self.canonical_covariance, 3, None),
            ("lbs_weights", self.lbs_weights, 2, None),
            ("barycentric", self.barycentric, 2, 3),
            ("opacity", self.opacity, 1, None),
            ("rgb", self.rgb, 2, 3),
        ):
            tensor = _require_tensor(value, name, ndim, last)
            if tensor.shape[0] != count:
                raise ValueError(f"{name} count differs from canonical_xyz")
        if self.canonical_covariance.shape[1:] != (3, 3):
            raise ValueError("canonical_covariance must have shape [N,3,3]")
        for name, value in (("face_indices", self.face_indices), ("body_part", self.body_part), ("side", self.side)):
            if not isinstance(value, torch.Tensor) or value.shape != (count,):
                raise ValueError(f"{name} must have shape [N]")
        if torch.any(self.face_indices < 0):
            raise ValueError("face_indices must be non-negative")
        if not torch.allclose(self.barycentric.sum(1), torch.ones(count, device=xyz.device, dtype=xyz.dtype), atol=1e-5, rtol=0):
            raise ValueError("barycentric rows must sum to one")
        if torch.any(self.barycentric < -1e-6):
            raise ValueError("barycentric coordinates must be non-negative")
        if torch.any(self.lbs_weights < -1e-6):
            raise ValueError("lbs weights must be non-negative")
        if not torch.allclose(self.lbs_weights.sum(1), torch.ones(count, device=xyz.device, dtype=xyz.dtype), atol=1e-5, rtol=0):
            raise ValueError("lbs weight rows must sum to one")
        if not set(self.body_part.detach().cpu().tolist()).issubset(set(ARM_PARTS)):
            raise ValueError("support includes a non-arm body part")
        if torch.any((self.opacity <= 0) | (self.opacity >= 1)):
            raise ValueError("diagnostic opacity must be strictly between zero and one")
        if torch.any((self.rgb < 0) | (self.rgb > 1)):
            raise ValueError("RGB must be in [0,1]")
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, torch.Tensor) and (value.requires_grad or isinstance(value, torch.nn.Parameter)):
                raise ValueError(f"{field} must be frozen and detached")
        return self

    @property
    def count(self) -> int:
        return int(self.canonical_xyz.shape[0])

    def to(self, device: torch.device | str, dtype: torch.dtype = torch.float32) -> "ArmSupportProbe":
        values: dict[str, Any] = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, torch.Tensor):
                values[name] = value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype).detach()
            else:
                values[name] = value
        return replace(self, **values).validate()

    def state_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _face_geometry(vertices: torch.Tensor, faces: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tri = vertices[faces.long()]
    cross = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
    twice_area = torch.linalg.vector_norm(cross, dim=1)
    normals = F.normalize(cross, dim=1, eps=1e-12)
    return tri, 0.5 * twice_area, normals


def _balanced_area_sample(
    areas: torch.Tensor,
    part: torch.Tensor,
    count: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    choices: list[torch.Tensor] = []
    groups = list(ARM_PARTS)
    base = count // len(groups)
    remainder = count - base * len(groups)
    for position, group in enumerate(groups):
        candidates = torch.where(part.cpu() == group)[0]
        if candidates.numel() == 0:
            raise RuntimeError(f"SMPL-X arm surface has no candidate faces for part {group}")
        probability = areas.detach().cpu()[candidates].clamp_min(1e-12)
        probability = probability / probability.sum()
        requested = base + (1 if position < remainder else 0)
        selected = torch.multinomial(probability, requested, replacement=True, generator=generator)
        choices.append(candidates[selected])
    return torch.cat(choices)


def sample_arm_support_surface(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    vertex_lbs_weights: torch.Tensor,
    *,
    count: int,
    seed: int,
    skin_rgb: torch.Tensor,
    inward_offset: float = 0.003,
    diagnostic_opacity: float = 0.65,
    tangent_scale_multiplier: float = 1.15,
    normal_scale: float = 0.0025,
    maximum_count: int = 20_000,
) -> ArmSupportProbe:
    """Area-sample only SMPL-X upper-arm/forearm faces.

    Face labels are derived from the official SMPL-X LBS table.  Parts 16--19
    are the left/right upper arms and forearms; wrist, hand and finger parts
    20--54 are excluded before sampling.
    """

    vertices = _require_tensor(vertices, "vertices", 2, 3).detach().float().cpu()
    faces = _require_tensor(faces, "faces", 2, 3).detach().long().cpu()
    weights = _require_tensor(vertex_lbs_weights, "vertex_lbs_weights", 2).detach().float().cpu()
    if weights.shape[0] != vertices.shape[0] or weights.shape[1] < 55:
        raise ValueError("vertex_lbs_weights must match vertices and contain the formal 55 joints")
    if count <= 0 or count > maximum_count:
        raise ValueError(f"count must be in [1,{maximum_count}]")
    if not 0 < diagnostic_opacity < 1:
        raise ValueError("diagnostic_opacity must be in (0,1)")
    skin = torch.as_tensor(skin_rgb, dtype=torch.float32).reshape(-1)
    if skin.shape != (3,) or not torch.isfinite(skin).all() or torch.any((skin < 0) | (skin > 1)):
        raise ValueError("skin_rgb must be finite RGB in [0,1]")

    tri, face_area, face_normals = _face_geometry(vertices, faces)
    face_weights = weights[faces].mean(dim=1)
    dominant = face_weights.argmax(dim=1)
    hand_influence = face_weights[:, list(HAND_AND_FINGER_PARTS)].sum(dim=1)
    arm_influence = face_weights[:, list(ARM_PARTS)].sum(dim=1)
    candidate = (
        torch.isin(dominant, torch.tensor(ARM_PARTS))
        & (arm_influence >= 0.50)
        & (hand_influence < 0.50)
        & (face_area > 1e-12)
    )
    candidate_faces = torch.where(candidate)[0]
    if candidate_faces.numel() < len(ARM_PARTS):
        raise RuntimeError("insufficient formal SMPL-X arm faces after excluding hands")
    chosen_local = _balanced_area_sample(
        face_area[candidate_faces], dominant[candidate_faces], count, seed,
    )
    chosen_faces = candidate_faces[chosen_local]

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 17)
    u = torch.rand(count, generator=generator)
    v = torch.rand(count, generator=generator)
    root = torch.sqrt(u)
    bary = torch.stack((1 - root, root * (1 - v), root * v), dim=1)
    selected_tri = tri[chosen_faces]
    normals = face_normals[chosen_faces]
    xyz_surface = (selected_tri * bary.unsqueeze(-1)).sum(dim=1)
    xyz = xyz_surface - float(inward_offset) * normals

    selected_vertex_weights = weights[faces[chosen_faces]]
    point_weights = (selected_vertex_weights * bary.unsqueeze(-1)).sum(dim=1).clamp_min(0)
    point_weights = point_weights / point_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)

    tangent = F.normalize(selected_tri[:, 1] - selected_tri[:, 0], dim=1, eps=1e-12)
    bitangent = F.normalize(torch.cross(normals, tangent, dim=1), dim=1, eps=1e-12)
    tangent = F.normalize(torch.cross(bitangent, normals, dim=1), dim=1, eps=1e-12)
    frame = torch.stack((tangent, bitangent, normals), dim=2)
    total_area = face_area[candidate_faces].sum()
    spacing = torch.sqrt(total_area / float(count)).clamp_min(1e-4)
    scales = torch.tensor(
        [float(spacing * tangent_scale_multiplier), float(spacing * tangent_scale_multiplier), float(normal_scale)],
        dtype=torch.float32,
    ).expand(count, -1)
    covariance = frame @ torch.diag_embed(scales.square()) @ frame.transpose(1, 2)
    selected_part = dominant[chosen_faces]
    side = torch.where(torch.isin(selected_part, torch.tensor(LEFT_ARM_PARTS)), -1, 1)
    probe = ArmSupportProbe(
        canonical_xyz=xyz.detach(),
        canonical_normals=normals.detach(),
        canonical_covariance=covariance.detach(),
        lbs_weights=point_weights.detach(),
        face_indices=chosen_faces.detach(),
        barycentric=bary.detach(),
        body_part=selected_part.detach(),
        side=side.detach(),
        opacity=torch.full((count,), float(diagnostic_opacity)),
        rgb=skin.reshape(1, 3).expand(count, -1).clone().detach(),
        metadata={
            "seed": int(seed),
            "sampling": "balanced per anatomical part, area-weighted within part",
            "arm_part_indices": list(ARM_PARTS),
            "excluded_hand_and_finger_indices": list(HAND_AND_FINGER_PARTS),
            "candidate_face_count": int(candidate_faces.numel()),
            "candidate_surface_area": float(total_area),
            "inward_offset_m": float(inward_offset),
            "diagnostic_opacity": float(diagnostic_opacity),
            "tangent_spacing_m": float(spacing),
            "normal_scale_m": float(normal_scale),
            "appearance_source": "subject02-only skin statistic supplied by R3 runner",
            "trainable": False,
        },
    )
    return probe.validate(maximum_count)


def deform_arm_support(
    probe: ArmSupportProbe,
    rigid_transforms: torch.Tensor,
    Rh: torch.Tensor,
    Th: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Apply the exact formal 55-joint transforms used by MMLP-Human."""

    probe.validate()
    reference = probe.canonical_xyz
    rigid = torch.as_tensor(rigid_transforms, device=reference.device, dtype=reference.dtype)
    rh = torch.as_tensor(Rh, device=reference.device, dtype=reference.dtype)
    th = torch.as_tensor(Th, device=reference.device, dtype=reference.dtype)
    if rigid.shape != (probe.lbs_weights.shape[1], 4, 4):
        raise ValueError("rigid_transforms must match probe LBS joint count")
    if rh.shape != (3, 3) or th.shape != (3,):
        raise ValueError("Rh/Th shapes must be [3,3]/[3]")
    for name, value in (("rigid_transforms", rigid), ("Rh", rh), ("Th", th)):
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"{name} contains NaN or Inf")
    blended = torch.einsum("nj,jab->nab", probe.lbs_weights, rigid)
    homogeneous = F.pad(reference, (0, 1), value=1)
    body_xyz = torch.einsum("nij,nj->ni", blended, homogeneous)[:, :3]
    posed_xyz = torch.einsum("ij,nj->ni", rh, body_xyz) + th
    linear = torch.einsum("ij,njk->nik", rh, blended[:, :3, :3])
    posed_covariance = linear @ probe.canonical_covariance @ linear.transpose(1, 2)
    inverse_transpose = torch.linalg.pinv(linear).transpose(1, 2)
    posed_normals = F.normalize(
        torch.einsum("nij,nj->ni", inverse_transpose, probe.canonical_normals),
        dim=1,
        eps=1e-12,
    )
    for name, value in (("xyz", posed_xyz), ("covariance", posed_covariance), ("normals", posed_normals)):
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"posed support {name} contains NaN or Inf")
    return {"xyz": posed_xyz, "covariance": posed_covariance, "normals": posed_normals}


def support_state_fingerprint(probe: ArmSupportProbe) -> str:
    buffer = io.BytesIO()
    torch.save({key: value for key, value in probe.state_dict().items() if isinstance(value, torch.Tensor)}, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def covered_support_visibility_fraction(
    base_rgb: torch.Tensor,
    base_alpha: torch.Tensor,
    combined_rgb: torch.Tensor,
    combined_alpha: torch.Tensor,
    *,
    difference_threshold: float = 0.01,
) -> torch.Tensor:
    """Fraction of base-foreground pixels changed by covered support."""

    for name, value in (
        ("base_rgb", base_rgb), ("base_alpha", base_alpha),
        ("combined_rgb", combined_rgb), ("combined_alpha", combined_alpha),
    ):
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError(f"{name} must be a finite tensor")
    if base_rgb.shape != combined_rgb.shape or base_alpha.shape != combined_alpha.shape:
        raise ValueError("base and combined render shapes must match")
    if base_rgb.ndim != 3 or base_rgb.shape[0] != 3 or base_alpha.shape != (1, *base_rgb.shape[1:]):
        raise ValueError("renders must be RGB/alpha CHW")
    change = torch.maximum(
        (combined_rgb - base_rgb).abs().amax(0, keepdim=True),
        (combined_alpha - base_alpha).abs(),
    )
    foreground = base_alpha >= 0.5
    return ((change > difference_threshold) & foreground).sum() / foreground.sum().clamp_min(1)


def adjudicate_r3_candidate(
    asset: Mapping[str, Any],
    probe: Mapping[str, Any],
) -> dict[str, str]:
    """Apply the frozen R3 decision matrix without tuning acceptance limits."""

    probe_status = str(probe.get("status"))
    geometry = str(asset.get("clean_body_geometry_available"))
    texture = str(asset.get("clean_body_texture_available"))
    binding = bool(asset.get("formal_lbs_binding_available"))
    if geometry == "true" and texture == "true" and binding:
        return {
            "recommended_candidate": "CANDIDATE_A",
            "recommended_shell_handling": "B3",
            "recommended_next_stage": "BUILD_CLEAN_BODY_BASE_PILOT",
        }
    if probe_status == "ARM_SUPPORT_PROBE_PASS" and binding and texture in {"true", "partial"}:
        return {
            "recommended_candidate": "CANDIDATE_B",
            "recommended_shell_handling": "B1",
            "recommended_next_stage": "BUILD_FROZEN_BODY_SUPPORT_PILOT",
        }
    if probe_status == "ARM_SUPPORT_PROBE_PASS_LOCAL_ONLY" and binding:
        return {
            "recommended_candidate": "CANDIDATE_C",
            "recommended_shell_handling": "B1",
            "recommended_next_stage": "BUILD_LOCAL_SUPPORT_PATCH_PILOT",
        }
    return {
        "recommended_candidate": "R3_UNRESOLVED",
        "recommended_shell_handling": "not_applicable",
        "recommended_next_stage": "ACQUIRE_OR_RECONSTRUCT_CLEAN_BODY_ASSET",
    }
