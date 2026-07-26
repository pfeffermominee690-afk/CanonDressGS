from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import io
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from scene.r3_body_support_probe import (
    ARM_PARTS,
    HAND_AND_FINGER_PARTS,
    ArmSupportProbe,
    sample_arm_support_surface,
)
from scene.r3_clean_body_asset import PROVENANCE, SkinAppearanceField, build_skin_appearance_field


ARM_SKIN_REGION_NAMES = (
    "left_upper_arm",
    "left_lower_arm",
    "right_upper_arm",
    "right_lower_arm",
    "shoulder_transition",
    "wrist_transition",
)
ARM_SKIN_REGION_INDEX = {name: index for index, name in enumerate(ARM_SKIN_REGION_NAMES)}
_C0 = 0.28209479177387814


def _finite(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be a finite tensor")
    return value


def _matrix_to_quaternion_wxyz(matrix: torch.Tensor) -> torch.Tensor:
    value = _finite(matrix.detach().float().cpu(), "rotation matrix")
    if value.ndim != 3 or value.shape[1:] != (3, 3):
        raise ValueError("rotation matrix must have shape [N,3,3]")
    m00, m01, m02 = value[:, 0, 0], value[:, 0, 1], value[:, 0, 2]
    m10, m11, m12 = value[:, 1, 0], value[:, 1, 1], value[:, 1, 2]
    m20, m21, m22 = value[:, 2, 0], value[:, 2, 1], value[:, 2, 2]
    magnitudes = torch.sqrt(torch.clamp(torch.stack((
        1 + m00 + m11 + m22,
        1 + m00 - m11 - m22,
        1 - m00 + m11 - m22,
        1 - m00 - m11 + m22,
    ), 1), min=0.0))
    candidates = torch.stack((
        torch.stack((1 + m00 + m11 + m22, m21 - m12, m02 - m20, m10 - m01), 1),
        torch.stack((m21 - m12, 1 + m00 - m11 - m22, m10 + m01, m02 + m20), 1),
        torch.stack((m02 - m20, m10 + m01, 1 - m00 + m11 - m22, m12 + m21), 1),
        torch.stack((m10 - m01, m02 + m20, m12 + m21, 1 - m00 - m11 + m22), 1),
    ), 1)
    candidates = candidates / (2.0 * magnitudes.clamp_min(1e-8)[:, :, None])
    row = torch.arange(len(value))
    result = F.normalize(candidates[row, magnitudes.argmax(1)], dim=1, eps=1e-12)
    result[result[:, 0] < 0] *= -1
    return result


def classify_arm_skin_regions(vertices: torch.Tensor, vertex_lbs: torch.Tensor) -> torch.Tensor:
    """Assign the six frozen O00 arm-skin regions on the subject02 surface."""

    xyz = _finite(vertices.detach().float().cpu(), "vertices")
    weights = _finite(vertex_lbs.detach().float().cpu(), "vertex_lbs")
    if xyz.ndim != 2 or xyz.shape[1] != 3 or weights.shape[0] != xyz.shape[0] or weights.shape[1] < 55:
        raise ValueError("vertices/LBS must have shapes [V,3] and [V,>=55]")
    part = weights[:, :55].argmax(1)
    region = torch.full((len(xyz),), -1, dtype=torch.long)
    region[part == 16] = ARM_SKIN_REGION_INDEX["left_upper_arm"]
    region[part == 18] = ARM_SKIN_REGION_INDEX["left_lower_arm"]
    region[part == 17] = ARM_SKIN_REGION_INDEX["right_upper_arm"]
    region[part == 19] = ARM_SKIN_REGION_INDEX["right_lower_arm"]

    radial = xyz[:, 0].abs()
    for anatomical_part in (16, 17):
        selected = part == anatomical_part
        if selected.any():
            cutoff = torch.quantile(radial[selected], 0.18)
            region[selected & (radial <= cutoff)] = ARM_SKIN_REGION_INDEX["shoulder_transition"]
    for anatomical_part in (18, 19):
        selected = part == anatomical_part
        if selected.any():
            cutoff = torch.quantile(radial[selected], 0.82)
            region[selected & (radial >= cutoff)] = ARM_SKIN_REGION_INDEX["wrist_transition"]
    return region


def classify_sampled_arm_skin_regions(canonical_xyz: torch.Tensor, body_part: torch.Tensor) -> torch.Tensor:
    """Transfer the same six-region contract directly to sampled arm points.

    Labeling the transition by a single barycentric owner can omit a narrow
    transition when no sampled point chooses that owner.  Applying the frozen
    proximal/distal 18% rule within each already-sampled formal arm part keeps
    the geometry and sampling unchanged while guaranteeing explicit shoulder
    and wrist transition populations.
    """

    xyz = _finite(canonical_xyz.detach().float().cpu(), "canonical_xyz")
    part = torch.as_tensor(body_part, dtype=torch.long).detach().cpu()
    if xyz.ndim != 2 or xyz.shape[1] != 3 or part.shape != (len(xyz),):
        raise ValueError("sampled arm region inputs must have shapes [N,3]/[N]")
    if not set(part.tolist()).issubset(set(ARM_PARTS)):
        raise ValueError("sampled arm regions include a non-arm part")
    region = torch.empty_like(part)
    region[part == 16] = ARM_SKIN_REGION_INDEX["left_upper_arm"]
    region[part == 18] = ARM_SKIN_REGION_INDEX["left_lower_arm"]
    region[part == 17] = ARM_SKIN_REGION_INDEX["right_upper_arm"]
    region[part == 19] = ARM_SKIN_REGION_INDEX["right_lower_arm"]
    radial = xyz[:, 0].abs()
    for anatomical_part in (16, 17):
        selected = part == anatomical_part
        cutoff = torch.quantile(radial[selected], 0.18)
        region[selected & (radial <= cutoff)] = ARM_SKIN_REGION_INDEX["shoulder_transition"]
    for anatomical_part in (18, 19):
        selected = part == anatomical_part
        cutoff = torch.quantile(radial[selected], 0.82)
        region[selected & (radial >= cutoff)] = ARM_SKIN_REGION_INDEX["wrist_transition"]
    if set(region.tolist()) != set(range(len(ARM_SKIN_REGION_NAMES))):
        raise RuntimeError("sampled support does not realize all six frozen skin regions")
    return region


@dataclass(frozen=True)
class Subject02ArmSkinField:
    rgb: torch.Tensor
    confidence: torch.Tensor
    provenance: torch.Tensor
    region: torch.Tensor
    body_part: torch.Tensor
    source_vertex: torch.Tensor
    metadata: Mapping[str, Any]

    def validate(self) -> "Subject02ArmSkinField":
        count = int(self.rgb.shape[0])
        expected = {
            "rgb": (count, 3), "confidence": (count,), "provenance": (count,),
            "region": (count,), "body_part": (count,), "source_vertex": (count,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"{name} contains NaN/Inf")
            if value.requires_grad or isinstance(value, torch.nn.Parameter):
                raise ValueError("subject02 arm skin field must remain frozen")
        if torch.any((self.rgb < 0) | (self.rgb > 1)) or torch.any((self.confidence < 0) | (self.confidence > 1)):
            raise ValueError("skin RGB/confidence is out of range")
        arm = torch.isin(self.body_part, torch.tensor(ARM_PARTS))
        if set(self.region[arm].tolist()) != set(range(len(ARM_SKIN_REGION_NAMES))):
            raise ValueError("arm skin field must distinguish all six frozen regions")
        if self.metadata.get("identity") != "subject02" or self.metadata.get("external_identity_used") is not False:
            raise ValueError("arm skin field provenance is not subject02-only")
        return self


def build_subject02_arm_skin_field(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    vertex_lbs: torch.Tensor,
    observation_vertex: torch.Tensor,
    observation_rgb: torch.Tensor,
    observation_confidence: torch.Tensor,
    *,
    expected_observed_pixel_count: int = 63_445,
    maximum_geodesic_distance: float = 0.35,
) -> Subject02ArmSkinField:
    """Build a six-region, low-frequency arm field from subject02 observations only."""

    observed_vertex = torch.as_tensor(observation_vertex, dtype=torch.long).cpu()
    observed_rgb = _finite(torch.as_tensor(observation_rgb).detach().float().cpu(), "observation_rgb")
    observed_confidence = _finite(
        torch.as_tensor(observation_confidence).detach().float().cpu(), "observation_confidence",
    )
    if observed_vertex.numel() != expected_observed_pixel_count:
        raise ValueError(
            f"subject02 observation count changed: {observed_vertex.numel()} != {expected_observed_pixel_count}"
        )
    base: SkinAppearanceField = build_skin_appearance_field(
        vertices, faces, vertex_lbs, observed_vertex, observed_rgb, observed_confidence,
        maximum_geodesic_distance=maximum_geodesic_distance,
    )
    region = classify_arm_skin_regions(vertices, vertex_lbs)
    body_part = torch.as_tensor(vertex_lbs).detach().float().cpu()[:, :55].argmax(1)
    rgb = base.rgb.clone()
    arm = region >= 0
    observed_regions = region[observed_vertex]
    region_statistics: dict[str, Any] = {}
    for index, name in enumerate(ARM_SKIN_REGION_NAMES):
        vertex_selection = region == index
        direct = observed_regions == index
        if direct.any():
            weights = observed_confidence[direct].clamp_min(1e-6)
            color = (observed_rgb[direct] * weights[:, None]).sum(0) / weights.sum()
            direct_count = int(direct.sum())
            source = "observed_subject02_pixels"
        else:
            color = base.rgb[vertex_selection].median(0).values
            direct_count = 0
            source = "subject02_same_region_low_frequency_fallback"
        if vertex_selection.any():
            coordinate = torch.as_tensor(vertices).detach().float().cpu()[vertex_selection, 1]
            normalized = (coordinate - coordinate.min()) / (coordinate.max() - coordinate.min()).clamp_min(1e-6)
            modulation = (0.97 + 0.06 * normalized)[:, None]
            rgb[vertex_selection] = (
                0.55 * base.rgb[vertex_selection] + 0.45 * color[None] * modulation
            ).clamp(0, 1)
        region_statistics[name] = {
            "vertex_count": int(vertex_selection.sum()), "direct_observation_count": direct_count,
            "source": source, "rgb_mean": rgb[vertex_selection].mean(0).tolist(),
        }
    if not arm.any():
        raise RuntimeError("subject02 surface contains no formal arm vertices")
    field = Subject02ArmSkinField(
        rgb=rgb.detach(), confidence=base.confidence.detach(), provenance=base.provenance.detach(),
        region=region.detach(), body_part=body_part.detach(), source_vertex=base.source_vertex.detach(),
        metadata={
            "identity": "subject02",
            "observed_pixel_count": int(observed_vertex.numel()),
            "expected_observed_pixel_count": int(expected_observed_pixel_count),
            "external_identity_used": False,
            "jay_rose_generic_clay_or_generated_texture_used": False,
            "allowed_propagation": ["left-right symmetry", "same-region geodesic", "within-region low-frequency"],
            "regions": region_statistics,
            "provenance_codes": dict(PROVENANCE),
        },
    )
    return field.validate()


@dataclass(frozen=True)
class O00ArmSupport:
    canonical_xyz: torch.Tensor
    canonical_normals: torch.Tensor
    covariance: torch.Tensor
    log_scaling: torch.Tensor
    quaternion_wxyz: torch.Tensor
    opacity: torch.Tensor
    sh0: torch.Tensor
    shN: torch.Tensor
    rgb: torch.Tensor
    lbs_weights: torch.Tensor
    face_indices: torch.Tensor
    barycentric: torch.Tensor
    body_part: torch.Tensor
    side: torch.Tensor
    skin_region: torch.Tensor
    skin_confidence: torch.Tensor
    skin_provenance: torch.Tensor
    metadata: Mapping[str, Any]

    @property
    def count(self) -> int:
        return int(self.canonical_xyz.shape[0])

    def validate(self, expected_count: int = 12_000) -> "O00ArmSupport":
        count = self.count
        if count != expected_count:
            raise ValueError(f"O00 arm support must contain exactly {expected_count} Gaussians")
        expected = {
            "canonical_xyz": (count, 3), "canonical_normals": (count, 3),
            "covariance": (count, 3, 3), "log_scaling": (count, 3),
            "quaternion_wxyz": (count, 4), "opacity": (count,),
            "sh0": (count, 1, 3), "shN": (count, 0, 3), "rgb": (count, 3),
            "lbs_weights": (count, 55), "face_indices": (count,), "barycentric": (count, 3),
            "body_part": (count,), "side": (count,), "skin_region": (count,),
            "skin_confidence": (count,), "skin_provenance": (count,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"{name} contains NaN/Inf")
            if value.requires_grad or isinstance(value, torch.nn.Parameter):
                raise ValueError("O00 arm support must be frozen")
        if not set(self.body_part.tolist()).issubset(set(ARM_PARTS)):
            raise ValueError("O00 support includes a non-arm body part")
        if set(self.skin_region.tolist()) != set(range(len(ARM_SKIN_REGION_NAMES))):
            raise ValueError("O00 support does not contain all six skin regions")
        if torch.any((self.opacity <= 0) | (self.opacity >= 1)):
            raise ValueError("support opacity must be in (0,1)")
        if torch.any((self.rgb < 0) | (self.rgb > 1)):
            raise ValueError("support RGB must be in [0,1]")
        if not torch.allclose(
            self.lbs_weights.sum(1),
            torch.ones(count, device=self.lbs_weights.device, dtype=self.lbs_weights.dtype),
            atol=1e-5,
            rtol=0,
        ):
            raise ValueError("support formal LBS rows must sum to one")
        if not torch.allclose(
            torch.linalg.vector_norm(self.quaternion_wxyz, dim=1),
            torch.ones(count, device=self.quaternion_wxyz.device, dtype=self.quaternion_wxyz.dtype),
            atol=1e-5,
            rtol=0,
        ):
            raise ValueError("support wxyz rotations must be normalized")
        if self.shN.numel() != 0:
            raise ValueError("O00 arm support SHN must be disabled")
        if self.metadata.get("appearance_identity") != "subject02" or self.metadata.get("trainable") is not False:
            raise ValueError("support provenance/frozen metadata is invalid")
        return self

    def to(self, device: torch.device | str, dtype: torch.dtype = torch.float32) -> "O00ArmSupport":
        values: dict[str, Any] = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, torch.Tensor):
                values[name] = value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype).detach()
            else:
                values[name] = value
        return replace(self, **values).validate(self.count)

    def tensor_dict(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__ if isinstance(getattr(self, name), torch.Tensor)}


def build_o00_arm_support(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    vertex_lbs: torch.Tensor,
    skin_field: Subject02ArmSkinField,
    *,
    count: int,
    seed: int,
    inward_offset: float,
    opacity_scale: float,
    density_reference_opacity: float = 0.65,
    density_reference_count: int = 12_000,
    tangent_scale_multiplier: float = 1.15,
    normal_scale: float = 0.0025,
) -> O00ArmSupport:
    if count != 12_000:
        raise ValueError("the frozen O00 closure uses exactly one 12k Medium support")
    if not 0 < opacity_scale <= 1:
        raise ValueError("global opacity_scale must be in (0,1]")
    field = skin_field.validate()
    density_opacity = 1.0 - (1.0 - float(density_reference_opacity)) ** (float(density_reference_count) / count)
    opacity = min(0.999, density_opacity * float(opacity_scale))
    median = field.rgb[torch.isin(field.body_part, torch.tensor(ARM_PARTS))].median(0).values
    probe: ArmSupportProbe = sample_arm_support_surface(
        vertices, faces, vertex_lbs, count=count, seed=seed, skin_rgb=median,
        inward_offset=inward_offset, diagnostic_opacity=opacity,
        tangent_scale_multiplier=tangent_scale_multiplier, normal_scale=normal_scale,
        maximum_count=12_000,
    )
    selected_vertices = torch.as_tensor(faces, dtype=torch.long).cpu()[probe.face_indices]
    vertex_rgb = field.rgb[selected_vertices]
    rgb = (vertex_rgb * probe.barycentric[:, :, None]).sum(1).clamp(0, 1)
    confidence = (field.confidence[selected_vertices] * probe.barycentric).sum(1).clamp(0, 1)
    provenance_candidates = field.provenance[selected_vertices]
    provenance = provenance_candidates.gather(1, probe.barycentric.argmax(1, keepdim=True)).squeeze(1)
    region = classify_sampled_arm_skin_regions(probe.canonical_xyz, probe.body_part)

    normal = probe.canonical_normals
    reference_axis = torch.zeros_like(normal); reference_axis[:, 2] = 1
    parallel = normal[:, 2].abs() > 0.90
    reference_axis[parallel] = torch.tensor([0.0, 1.0, 0.0])
    tangent = F.normalize(torch.cross(reference_axis, normal, dim=1), dim=1, eps=1e-12)
    bitangent = F.normalize(torch.cross(normal, tangent, dim=1), dim=1, eps=1e-12)
    frame = torch.stack((tangent, bitangent, normal), dim=2)
    quaternion = _matrix_to_quaternion_wxyz(frame)
    tangent_scale = float(probe.metadata["tangent_spacing_m"]) * float(tangent_scale_multiplier)
    scaling = torch.tensor([tangent_scale, tangent_scale, float(normal_scale)]).expand(count, -1)
    covariance = frame @ torch.diag_embed(scaling.square()) @ frame.transpose(1, 2)
    selected_surface = (
        torch.as_tensor(vertices).detach().float().cpu()[selected_vertices] * probe.barycentric[:, :, None]
    ).sum(1)
    signed_surface_offset = ((probe.canonical_xyz - selected_surface) * normal).sum(1)
    outside_fraction = float((signed_surface_offset > 1e-6).float().mean())
    support = O00ArmSupport(
        canonical_xyz=probe.canonical_xyz.detach(), canonical_normals=normal.detach(),
        covariance=covariance.detach(), log_scaling=scaling.log().detach(),
        quaternion_wxyz=quaternion.detach(), opacity=torch.full((count,), opacity),
        sh0=((rgb - 0.5) / _C0).reshape(count, 1, 3).detach(), shN=torch.empty(count, 0, 3),
        rgb=rgb.detach(), lbs_weights=probe.lbs_weights[:, :55].detach(),
        face_indices=probe.face_indices.detach(), barycentric=probe.barycentric.detach(),
        body_part=probe.body_part.detach(), side=probe.side.detach(), skin_region=region.detach(),
        skin_confidence=confidence.detach(), skin_provenance=provenance.detach(),
        metadata={
            **dict(probe.metadata), "count": count, "appearance_identity": "subject02",
            "skin_observed_pixel_count": int(field.metadata["observed_pixel_count"]),
            "skin_region_names": list(ARM_SKIN_REGION_NAMES), "density_reference_opacity": density_reference_opacity,
            "density_reference_count": density_reference_count, "density_aware_opacity": density_opacity,
            "global_opacity_scale": float(opacity_scale), "final_opacity": opacity,
            "rotation_convention": "local tangent/bitangent/normal frame, quaternion wxyz",
            "SH_degree": 0, "SHN_enabled": False, "trainable": False,
            "external_identity_or_generated_texture_used": False,
            "anatomical_envelope_outside_fraction": outside_fraction,
            "signed_surface_offset_min_m": float(signed_surface_offset.min()),
            "signed_surface_offset_max_m": float(signed_surface_offset.max()),
        },
    )
    return support.validate()


def deform_o00_arm_support(
    support: O00ArmSupport,
    rigid_transforms: torch.Tensor,
    Rh: torch.Tensor,
    Th: torch.Tensor,
) -> dict[str, torch.Tensor]:
    support.validate(support.count)
    device, dtype = support.canonical_xyz.device, support.canonical_xyz.dtype
    rigid = _finite(torch.as_tensor(rigid_transforms, device=device, dtype=dtype), "rigid_transforms")
    rh = _finite(torch.as_tensor(Rh, device=device, dtype=dtype), "Rh")
    th = _finite(torch.as_tensor(Th, device=device, dtype=dtype), "Th")
    if rigid.shape != (55, 4, 4) or rh.shape != (3, 3) or th.shape != (3,):
        raise ValueError("formal O00 deformation shapes are invalid")
    blended = torch.einsum("nj,jab->nab", support.lbs_weights, rigid)
    homogeneous = F.pad(support.canonical_xyz, (0, 1), value=1)
    body_xyz = torch.einsum("nij,nj->ni", blended, homogeneous)[:, :3]
    xyz = torch.einsum("ij,nj->ni", rh, body_xyz) + th
    linear = torch.einsum("ij,njk->nik", rh, blended[:, :3, :3])
    covariance = linear @ support.covariance @ linear.transpose(1, 2)
    inverse_transpose = torch.linalg.pinv(linear).transpose(1, 2)
    normals = F.normalize(torch.einsum("nij,nj->ni", inverse_transpose, support.canonical_normals), dim=1, eps=1e-12)
    if not all(torch.isfinite(value).all() for value in (xyz, covariance, normals)):
        raise FloatingPointError("posed O00 support contains NaN/Inf")
    return {"xyz": xyz, "covariance": covariance, "normals": normals}


def support_tensor_fingerprint(support: O00ArmSupport) -> str:
    buffer = io.BytesIO()
    torch.save(support.tensor_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def adjudicate_o00_support(
    rows: list[Mapping[str, Any]],
    *,
    outside_fraction: float,
    base_bitwise_exact: bool,
    shoulder_visual: str,
    wrist_visual: str,
) -> dict[str, Any]:
    if len(rows) != 4:
        raise ValueError("O00 support adjudication requires exactly four views")
    recall = [float(row["hole_repair_recall"]) for row in rows]
    leakage = [float(row["background_leakage"]) for row in rows]
    covered = [float(row["covered_support_visibility"]) for row in rows]
    finite = all(bool(row["finite"]) and not bool(row.get("pose_explosion", False)) for row in rows)
    visual_ok = shoulder_visual in {"PASS", "WARN"} and wrist_visual in {"PASS", "WARN"}
    quantitative = (
        min(recall) >= 0.90 and sum(recall) / 4 >= 0.95
        and max(leakage) <= 0.05 and sum(leakage) / 4 <= 0.03
        and max(covered) <= 0.01 and float(outside_fraction) <= 0.02
        and finite and bool(base_bitwise_exact)
    )
    status = "ARM_SUPPORT_PASS" if quantitative and visual_ok else "ARM_SUPPORT_FAIL"
    return {
        "status": status, "quantitative_pass": quantitative, "visual_pass_or_warn": visual_ok,
        "hole_repair_recall_min": min(recall), "hole_repair_recall_mean": sum(recall) / 4,
        "background_leakage_max": max(leakage), "background_leakage_mean": sum(leakage) / 4,
        "covered_support_visibility_max": max(covered), "outside_fraction": float(outside_fraction),
        "four_views_finite": finite, "base_bitwise_exact": bool(base_bitwise_exact),
        "oracle_allowed": status == "ARM_SUPPORT_PASS",
    }
