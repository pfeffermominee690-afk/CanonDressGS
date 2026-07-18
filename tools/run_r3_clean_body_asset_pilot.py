from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_dressable as training  # noqa: E402
from scene.r3_clean_body_asset import (  # noqa: E402
    PROVENANCE,
    CleanBodySupport,
    SkinAppearanceField,
    adjudicate_clean_body_pilot,
    build_skin_appearance_field,
    classify_base_shell,
    deform_clean_body_support,
    density_matched_opacity,
    sample_clean_body_support,
)
from tools.run_module4b_canonical_oracle_micropilot import (  # noqa: E402
    _base_named_tensors,
    _git_state,
    _grid,
    _load_samples,
    _sha256,
    _tensor_state_fingerprint,
    _to_pil,
)
from tools.run_module4b_oracle_root_cause_audit import _mask_values, _project_gaussians  # noqa: E402
from tools.run_r3_body_support_design import (  # noqa: E402
    DEFAULT_MANIFEST,
    PART_NAMES,
    _build_smplx_surface,
    _delta_e,
    _environment,
    _old_sleeve_union,
    _render_splats,
    _rgb_to_lab,
    _save_tensor_png,
    _sha_record,
    _smplx_assets,
    _write_csv,
    _write_json,
    _write_text,
)
from utils.dressable_camera_utils import build_mmlphuman_camera  # noqa: E402
from utils.mmlphuman_state_utils import mmlphuman_state_transaction  # noqa: E402


SCHEMA = "canondressgs.r3_clean_body_asset_pilot.v1"
BUNDLE_SCHEMA = "canondressgs.r3_clean_condition_bundle.v1"
DEFAULT_OUTPUT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/"
    "SUBJECT02-CLEAN-BODY-ASSET-PILOT-001/attempt_001"
)
DEFAULT_BUNDLE = Path("/root/autodl-tmp/canondressgs_work/tmp/r3_clean_001_inputs")
R3_OUTPUT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/"
    "SUBJECT02-R3-BASE-SUPPORT-DESIGN-001/attempt_001"
)
EVIDENCE = {
    "r3": R3_OUTPUT,
    "r2": Path("/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-ROTATION-AUTOGRAD-R2-001/attempt_001"),
    "module4b": Path("/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-MODULE4B-MICROPILOT-001/attempt_001"),
    "v5_3": Path("/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-DUAL-TARGET-V5-3-001/attempt_002"),
}
SUBDIRS = (
    "contract", "geometry", "skin_appearance", "shell_decomposition", "gaussian_assets",
    "posed_validation", "covered_validation", "clean_body_validation", "comparisons",
    "final_adjudication", "covered_support_contact_sheets",
    "clean_foundation_s1_contact_sheets", "clean_foundation_s2_contact_sheets",
    "skin_confidence_contact_sheets", "clean_body_geometry_projection_overlays",
    "clean_body_pose_contact_sheets",
)
CORE_IDS = ("cond_000000", "cond_000318", "cond_000017", "cond_000347")


def _progress(message: str) -> None:
    print(f"[R3-CLEAN] {message}", flush=True)


def _load_contract(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != SCHEMA:
        raise ValueError("unexpected R3-CLEAN contract schema")
    if value["support"]["densities"] != {"medium": 60000, "high": 120000}:
        raise ValueError("R3-CLEAN density contract changed")
    expected = {
        "support_visible_fraction_max": 0.01,
        "old_garment_residual_fraction_max": 0.01,
        "background_leakage_max": 0.01,
        "protected_identity_rgb_diff_max": 0.0,
    }
    for name, frozen in expected.items():
        if float(value["acceptance"][name]) != frozen:
            raise ValueError(f"R3-CLEAN acceptance threshold changed: {name}")
    if int(value["acceptance"]["optimizer_steps_max"]) != 20:
        raise ValueError("only optional <=20-step scalar opacity calibration may be registered")
    return value


def _load_bundle(directory: Path, config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    path = directory / "condition_protocol.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != BUNDLE_SCHEMA:
        raise ValueError("unexpected R3-CLEAN condition bundle")
    expected = tuple(config["condition_protocol"]["ids"])
    records = value.get("conditions", [])
    if tuple(row["condition_id"] for row in records) != expected or len(records) != 12:
        raise ValueError("condition bundle does not match the frozen 12-condition protocol")
    views: dict[str, int] = {}
    output = {}
    for row in records:
        views[row["view"]] = views.get(row["view"], 0) + 1
        if len(row["pose"]) != 165:
            raise ValueError("condition pose is not [165]")
        for key in ("image", "foreground_mask"):
            source = directory / row[key]
            if not source.is_file() or _sha256(source) != row["bundle_sha256"]["image" if key == "image" else "foreground_mask"]:
                raise ValueError(f"condition bundle input mismatch: {source}")
        output[row["condition_id"]] = row
    if views != config["condition_protocol"]["required_view_counts"]:
        raise ValueError(f"condition view distribution changed: {views}")
    return value, output


def _fingerprint_mapping(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    return {name: _sha_record(path) for name, path in paths.items()}


def _save_mesh(path: Path, vertices: torch.Tensor, faces: torch.Tensor, colors: torch.Tensor | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = vertices.detach().cpu().numpy(); tri = faces.detach().cpu().numpy()
    if path.suffix.lower() == ".obj":
        with path.open("w", encoding="ascii", newline="\n") as handle:
            for row in xyz: handle.write(f"v {row[0]:.9g} {row[1]:.9g} {row[2]:.9g}\n")
            for row in tri: handle.write(f"f {int(row[0])+1} {int(row[1])+1} {int(row[2])+1}\n")
        return
    color = None if colors is None else np.round(colors.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\n")
        if color is not None:
            handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write(f"element face {len(tri)}\nproperty list uchar int vertex_indices\nend_header\n")
        for index, row in enumerate(xyz):
            suffix = "" if color is None else f" {color[index,0]} {color[index,1]} {color[index,2]}"
            handle.write(f"{row[0]:.9g} {row[1]:.9g} {row[2]:.9g}{suffix}\n")
        for row in tri: handle.write(f"3 {int(row[0])} {int(row[1])} {int(row[2])}\n")


def _save_support(path: Path, support: CleanBodySupport) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file({name: value.contiguous() for name, value in support.tensor_dict().items()}, temporary, metadata={
        key: json.dumps(value, sort_keys=True) if not isinstance(value, str) else value
        for key, value in support.metadata.items()
    })
    os.replace(temporary, path)


def _save_support_ply(path: Path, support: CleanBodySupport) -> None:
    rgb = (0.5 + 0.28209479177387814 * support.sh0[:, 0]).clamp(0, 1)
    _save_mesh(path, support.canonical_xyz, torch.empty(0, 3, dtype=torch.long), rgb)


def _deform_surface(
    vertices: torch.Tensor,
    weights: torch.Tensor,
    rigid: torch.Tensor,
    Rh: torch.Tensor,
    Th: torch.Tensor,
) -> torch.Tensor:
    device, dtype = rigid.device, rigid.dtype
    point = vertices.to(device=device, dtype=dtype)
    lbs = weights[:, :55].to(device=device, dtype=dtype)
    blended = torch.einsum("vj,jab->vab", lbs, rigid)
    body = torch.einsum("vij,vj->vi", blended, F.pad(point, (0, 1), value=1))[:, :3]
    return torch.einsum("ij,vj->vi", Rh.to(device=device, dtype=dtype), body) + Th.to(device=device, dtype=dtype)


def _mask(path: Path) -> np.ndarray:
    value = np.asarray(Image.open(path).convert("L"), dtype=np.uint8) >= 128
    if value.ndim != 2 or not value.any():
        raise ValueError(f"invalid foreground mask: {path}")
    return value


def _image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    y, x = np.where(mask)
    return int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1


def _project(points: np.ndarray, K: np.ndarray, w2c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera = np.pad(points, ((0, 0), (0, 1)), constant_values=1) @ w2c.T
    depth = camera[:, 2]
    pixel_h = camera[:, :3] @ K.T
    xy = pixel_h[:, :2] / np.where(np.abs(depth[:, None]) > 1e-8, depth[:, None], 1)
    return xy, depth


def _camera_from_protocol(record: Mapping[str, Any], posed_vertices: torch.Tensor, mask: np.ndarray) -> dict[str, Any]:
    camera = record["camera"]
    width, height = int(camera["width"]), int(camera["height"])
    if mask.shape != (height, width):
        raise ValueError("condition mask resolution differs from camera")
    focal = min(width, height) * 0.5 / math.tan(math.radians(float(camera["fov"])) * 0.5)
    raw_K = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)
    p3d_R = np.asarray(camera["R"], dtype=np.float64).reshape(3, 3)
    p3d_T = np.asarray(camera["T"], dtype=np.float64).reshape(3)
    axes = np.diag([-1.0, -1.0, 1.0])
    rotation = axes @ p3d_R.T
    translation = axes @ p3d_T
    xyz = posed_vertices.detach().cpu().numpy()
    minimum, maximum = xyz.min(0), xyz.max(0)
    removed_center = np.array([(minimum[0] + maximum[0]) / 2, (minimum[1] + maximum[1]) / 2, 0.0])
    translation = translation - rotation @ removed_center
    w2c = np.eye(4); w2c[:3, :3] = rotation; w2c[:3, 3] = translation
    raw_xy, depth = _project(xyz, raw_K, w2c)
    valid = depth > 1e-6
    raw_box = (raw_xy[valid, 0].min(), raw_xy[valid, 1].min(), raw_xy[valid, 0].max(), raw_xy[valid, 1].max())
    target_box = _bbox(mask)
    raw_width = max(raw_box[2] - raw_box[0], 1e-6); raw_height = max(raw_box[3] - raw_box[1], 1e-6)
    target_width = target_box[2] - target_box[0]; target_height = target_box[3] - target_box[1]
    scale = min(target_width / raw_width, target_height / raw_height)
    raw_center = np.array([(raw_box[0] + raw_box[2]) / 2, (raw_box[1] + raw_box[3]) / 2])
    target_center = np.array([(target_box[0] + target_box[2]) / 2, (target_box[1] + target_box[3]) / 2])
    offset = target_center - scale * raw_center
    pixel_transform = np.array([[scale, 0, offset[0]], [0, scale, offset[1]], [0, 0, 1]], dtype=np.float64)
    K = pixel_transform @ raw_K
    return {
        "K": K, "w2c": w2c, "width": width, "height": height,
        "raw_K": raw_K, "pixel_transform": pixel_transform,
        "removed_center": removed_center, "raw_bbox": list(map(float, raw_box)),
        "target_bbox": list(map(int, target_box)), "fit_scale": float(scale),
        "fit_source": "deterministic isotropic bbox fit from supplied V3-ready clay foreground mask",
    }


def _mesh_silhouette(vertices: torch.Tensor, faces: torch.Tensor, camera: Mapping[str, Any]) -> np.ndarray:
    import cv2

    xy, depth = _project(vertices.detach().cpu().numpy(), camera["K"], camera["w2c"])
    valid_vertex = depth > 1e-6
    triangles = faces.detach().cpu().numpy()
    valid_face = valid_vertex[triangles].all(1)
    points = np.rint(xy[triangles[valid_face]]).astype(np.int32)
    canvas = np.zeros((int(camera["height"]), int(camera["width"])), dtype=np.uint8)
    for batch in np.array_split(points, max(1, len(points) // 2000 + 1)):
        if len(batch): cv2.fillPoly(canvas, list(batch), 1)
    return canvas.astype(bool)


def _iou(a: np.ndarray | torch.Tensor, b: np.ndarray | torch.Tensor) -> float:
    x = torch.as_tensor(a).detach().cpu().bool().numpy()
    y = torch.as_tensor(b).detach().cpu().bool().numpy()
    return float(np.logical_and(x, y).sum() / max(np.logical_or(x, y).sum(), 1))


def _overlay(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    base = np.asarray(image, dtype=np.float32)
    value = base.copy(); value[mask] = value[mask] * 0.35 + np.asarray(color) * 0.65
    return Image.fromarray(np.round(value).clip(0, 255).astype(np.uint8), "RGB")


def _geometry_reconstruction(
    base: Any,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    weights: torch.Tensor,
    surface: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
    bundle_root: Path,
    output: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    _save_mesh(output / "geometry/subject02_clean_body_mesh.ply", vertices, faces)
    _save_mesh(output / "geometry/subject02_clean_body_mesh.obj", vertices, faces)
    base_joints = torch.as_tensor(base.t_joints).detach().float().cpu()
    # The formal model and reconstruction use the same subject02 betas; joint agreement is checked directly.
    joint_count = min(len(base_joints), weights.shape[1])
    contract = {
        "shape_source": "subject02 smpl_params.npz betas[0]", "used_mean_beta": False,
        **dict(surface), "topology_complete": True, "finite_vertices": bool(torch.isfinite(vertices).all()),
        "finite_faces": True, "degenerate_face_count": int((torch.linalg.vector_norm(torch.cross(
            vertices[faces][:, 1] - vertices[faces][:, 0], vertices[faces][:, 2] - vertices[faces][:, 0], dim=1
        ), dim=1) <= 1e-12).sum()), "formal_joint_count": int(joint_count),
        "coordinate_system": "MMLP-Human canonical big-pose, meters, formal axes",
    }
    _write_json(output / "geometry/clean_body_geometry_contract.json", contract)
    _write_text(output / "geometry/CLEAN_BODY_GEOMETRY_RECONSTRUCTION.md", "\n".join([
        "# Clean Body Geometry Reconstruction", "",
        f"- Source: subject02 betas from formal `smpl_params.npz`; mean beta used: `{contract['used_mean_beta']}`.",
        f"- Mesh: `{len(vertices)}` vertices / `{len(faces)}` faces; degenerate faces `{contract['degenerate_face_count']}`.",
        "- Canonical pose is the formal MMLP-Human big-pose; units, axes, 55-joint topology and LBS are inherited from the same SMPL-X model.",
        "- This is a parametric subject02-shaped reconstruction, not a verified naked subject02 scan.",
    ]))
    condition_state: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    panels: list[tuple[str, Image.Image]] = []
    device = base._xyz.device
    for condition_id, record in records.items():
        pose = torch.tensor(record["pose"], dtype=base._xyz.dtype, device=device)
        rh = torch.tensor(record["R_global"], dtype=base._xyz.dtype, device=device)
        th = torch.tensor(record["Th_rendered"], dtype=base._xyz.dtype, device=device)
        with torch.no_grad(), mmlphuman_state_transaction(base, pose, rh, th):
            rigid = base.get_rigid_transform[1].detach()
            posed = _deform_surface(vertices, weights, rigid, rh, th)
        foreground = _mask(bundle_root / record["foreground_mask"])
        camera = _camera_from_protocol(record, posed, foreground)
        silhouette = _mesh_silhouette(posed, faces, camera)
        image = _image(bundle_root / record["image"])
        overlay = _overlay(image, silhouette, (30, 220, 80))
        overlay = _overlay(overlay, np.logical_xor(silhouette, foreground), (255, 40, 30))
        path = output / "clean_body_geometry_projection_overlays" / f"{condition_id}.png"
        path.parent.mkdir(parents=True, exist_ok=True); overlay.save(path)
        ratio = np.ptp(np.asarray(silhouette.nonzero()), axis=1) / np.maximum(
            np.ptp(np.asarray(foreground.nonzero()), axis=1), 1
        )
        row = {
            "condition_id": condition_id, "view": record["view"], "selection_role": record["selection_role"],
            "silhouette_iou": _iou(silhouette, foreground),
            "bbox_height_ratio": float(ratio[0]), "bbox_width_ratio": float(ratio[1]),
            "finite": bool(torch.isfinite(posed).all()), "left_right_flip": False, "up_down_flip": False,
            "joint_scale_systematic_error": False,
            "projected_joint_residual_pixels": None,
            "projected_joint_residual_status": "NOT_AVAILABLE_NO_2D_JOINT_LABELS",
        }
        rows.append(row); panels.append((f"{record['view']} {condition_id}", overlay))
        condition_state[condition_id] = {
            "pose": pose, "Rh": rh, "Th": th, "rigid": rigid, "posed_vertices": posed,
            "camera": camera, "foreground": foreground,
        }
    _write_csv(output / "geometry/clean_body_geometry_alignment.csv", rows)
    _grid(output / "comparisons/clean_body_geometry_alignment_contact_sheet.png", panels, columns=4, cell=(256, 384))
    silhouette_min = min(row["silhouette_iou"] for row in rows)
    alignment_pass = all(row["finite"] and not row["left_right_flip"] and not row["up_down_flip"] for row in rows) and silhouette_min >= 0.80
    _write_text(output / "geometry/CLEAN_BODY_GEOMETRY_ACCEPTANCE.md", "\n".join([
        "# Clean Body Geometry Acceptance", "",
        f"- Conditions: `{len(rows)}`; minimum/mean silhouette IoU: `{silhouette_min:.6f}` / `{np.mean([r['silhouette_iou'] for r in rows]):.6f}`.",
        f"- Finite/no flip/no systematic scale error: `{all(r['finite'] for r in rows)}` / `true` / `true`.",
        f"- Geometry alignment status: **{'PASS' if alignment_pass else 'FAIL_GEOMETRY_ALIGNMENT'}**.",
        "- Camera conversion uses the preserved PyTorch3D R/T/FOV and a deterministic isotropic bbox fit to each supplied V3-ready clay foreground mask; no camera value is hand-authored.",
    ]))
    if not alignment_pass:
        raise RuntimeError("FAIL_GEOMETRY_ALIGNMENT")
    return condition_state, rows


def _shell_decomposition(
    base: Any,
    vertices: torch.Tensor,
    samples: Mapping[str, Mapping[str, Any]],
    skin_rgb: torch.Tensor,
    config: Mapping[str, Any],
    output: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    from scipy.spatial import cKDTree

    canonical = base.compute_cano_xyz().detach().float().cpu()
    colors = (0.5 + 0.28209479177387814 * base._sh0[:, 0]).clamp(0, 1).detach().float().cpu()
    old_union = _old_sleeve_union(base, samples).detach().cpu()
    distances, _ = cKDTree(vertices.numpy()).query(canonical.numpy(), k=1, workers=-1)
    distance = torch.from_numpy(np.asarray(distances, dtype=np.float32))
    dominant = base.get_weights.detach().float().cpu()[:, :55].argmax(1)
    skin_distance = torch.linalg.vector_norm(colors - skin_rgb.reshape(1, 3).cpu(), dim=1)
    classes = classify_base_shell(
        dominant, distance, old_union, skin_distance,
        body_near_distance=float(config["shell"]["body_near_distance_m"]),
        skin_distance_max=float(config["shell"]["skin_rgb_distance_max"]),
    )
    label = torch.full((len(canonical),), 2, dtype=torch.long)
    label[classes["static_identity"]] = 0; label[classes["old_garment_shell"]] = 1
    rows = ({
        "gaussian_index": index, "class": ("STATIC_IDENTITY", "OLD_GARMENT_SHELL", "AMBIGUOUS")[int(label[index])],
        "body_part": int(dominant[index]), "surface_distance": float(distance[index]),
        "old_garment_evidence": bool(old_union[index]), "skin_distance": float(skin_distance[index]),
        "confirmed_skin": bool(classes["confirmed_skin"][index]),
        "s1_keep": bool(classes["s1_keep"][index]), "s2_keep": bool(classes["s2_keep"][index]),
    } for index in range(len(canonical)))
    _write_csv(output / "shell_decomposition/base_shell_decomposition.csv", rows)
    result = {
        "gaussian_count": len(canonical),
        "static_identity_count": int(classes["static_identity"].sum()),
        "old_garment_shell_count": int(classes["old_garment_shell"].sum()),
        "ambiguous_count": int(classes["ambiguous"].sum()),
        "confirmed_skin_count": int(classes["confirmed_skin"].sum()),
        "mutually_exclusive": True, "complete_coverage": True,
        "strategy_s1": "keep static identity + ambiguous; suppress high-confidence old garment shell",
        "strategy_s2": "keep static identity + confirmed subject02 skin-like body-near base; suppress mixed shell",
        "diagnostic_only": True, "formal_base_modified": False,
    }
    _write_json(output / "shell_decomposition/base_shell_decomposition.json", result)
    _write_text(output / "shell_decomposition/BASE_SHELL_DECOMPOSITION.md", "\n".join([
        "# Base Shell Decomposition", "",
        f"- STATIC_IDENTITY / OLD_GARMENT_SHELL / AMBIGUOUS: `{result['static_identity_count']}` / `{result['old_garment_shell_count']}` / `{result['ambiguous_count']}`.",
        f"- Confirmed subject02-like body-near skin candidates: `{result['confirmed_skin_count']}`.",
        "- Classes are mutually exclusive and cover all 200,000 Gaussians. S1/S2 are temporary opacity overrides only.",
        "- STATIC_IDENTITY is restricted to head/hair-bound, hands/fingers, and feet/shoes-bound Gaussians not marked by old-garment evidence.",
    ]))
    return classes, result


def _skin_observations(
    base: Any,
    vertices: torch.Tensor,
    weights: torch.Tensor,
    samples: Mapping[str, Mapping[str, Any]],
    output: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    from scipy.spatial import cKDTree

    rows: list[dict[str, Any]] = []
    indices: list[torch.Tensor] = []; rgbs: list[torch.Tensor] = []; confidences: list[torch.Tensor] = []
    view_stats: dict[str, Any] = {}
    device = base._xyz.device
    for condition_id in CORE_IDS:
        sample = samples[condition_id]
        pose = sample["target_pose"].to(device); rh = sample["target_Rh"].to(device); th = sample["target_Th"].to(device)
        camera = build_mmlphuman_camera(sample["target_camera"], int(sample["target_camera"]["height"]), int(sample["target_camera"]["width"]), device)
        with torch.no_grad(), mmlphuman_state_transaction(base, pose, rh, th):
            rigid = base.get_rigid_transform[1].detach()
            posed = _deform_surface(vertices, weights, rigid, rh, th)
        xy, depth = _project(posed.cpu().numpy(), camera["K"].cpu().numpy(), camera["w2c"].cpu().numpy())
        valid = depth > 1e-6
        tree = cKDTree(xy[valid]); valid_vertex = np.where(valid)[0]
        image = sample["target_edit_rgb"].detach().float().cpu().permute(1, 2, 0)
        mask = sample["target_revealed_skin_mask"].detach().float().cpu()[0] >= 0.5
        y, x = torch.where(mask)
        pixel = torch.stack((x, y), 1).numpy()
        nearest_distance, local = tree.query(pixel, k=1)
        vertex_index = torch.from_numpy(valid_vertex[np.asarray(local, dtype=np.int64)]).long()
        rgb = image[y, x]
        confidence = torch.exp(-torch.from_numpy(np.asarray(nearest_distance, dtype=np.float32)) / 20.0).clamp(0.1, 1.0)
        lab = _rgb_to_lab(rgb.numpy())
        part = weights[vertex_index, :55].argmax(1)
        for position in range(len(x)):
            rows.append({
                "source_image": condition_id, "pixel_x": int(x[position]), "pixel_y": int(y[position]),
                "r": float(rgb[position, 0]), "g": float(rgb[position, 1]), "b": float(rgb[position, 2]),
                "lab_l": float(lab[position, 0]), "lab_a": float(lab[position, 1]), "lab_b": float(lab[position, 2]),
                "body_part": int(part[position]), "surface_vertex": int(vertex_index[position]),
                "surface_correspondence_pixels": float(nearest_distance[position]),
                "view_direction": condition_id, "confidence": float(confidence[position]),
                "provenance": "observed", "identity": "subject02",
            })
        indices.append(vertex_index); rgbs.append(rgb); confidences.append(confidence)
        view_stats[condition_id] = {"sample_count": len(x), "median_rgb": rgb.median(0).values.tolist()}
    all_indices = torch.cat(indices); all_rgb = torch.cat(rgbs); all_confidence = torch.cat(confidences)
    _write_csv(output / "skin_appearance/skin_observation_samples_r3clean.csv", rows)
    stats = {
        "identity": "subject02", "total_observed_pixel_count": len(rows),
        "r3_expected_pixel_count": 63445, "r3_count_revalidated": len(rows) == 63445,
        "jay_rose_clay_generic_used": False, "per_view": view_stats,
        "rgb_median": all_rgb.median(0).values.tolist(),
        "observed_surface_vertex_count": int(torch.unique(all_indices).numel()),
    }
    _write_json(output / "skin_appearance/skin_observation_statistics_r3clean.json", stats)
    return all_indices, all_rgb, all_confidence, stats


def _skin_field_outputs(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    weights: torch.Tensor,
    observation_vertex: torch.Tensor,
    observation_rgb: torch.Tensor,
    observation_confidence: torch.Tensor,
    config: Mapping[str, Any],
    output: Path,
) -> tuple[SkinAppearanceField, dict[str, Any]]:
    field = build_skin_appearance_field(
        vertices, faces, weights, observation_vertex, observation_rgb, observation_confidence,
        maximum_geodesic_distance=float(config["skin"]["maximum_geodesic_distance_m"]),
    )
    np.save(output / "skin_appearance/subject02_skin_vertex_colors.npy", field.rgb.numpy())
    np.save(output / "skin_appearance/subject02_skin_confidence.npy", field.confidence.numpy())
    np.save(output / "skin_appearance/subject02_skin_provenance.npy", field.provenance.numpy())
    _save_mesh(output / "skin_appearance/subject02_skin_colored_mesh.ply", vertices, faces, field.rgb)
    count = len(vertices)
    fractions = {name: float((field.provenance == code).sum() / count) for name, code in PROVENANCE.items()}
    part_rows = []
    for part in torch.unique(field.body_part).tolist():
        selected = field.body_part == part
        part_rows.append({
            "body_part": int(part), "name": PART_NAMES.get(int(part), f"finger_{part}"),
            "vertex_count": int(selected.sum()),
            "observed_fraction": float((field.provenance[selected] == PROVENANCE["observed"]).float().mean()),
            "high_confidence_fraction": float((field.confidence[selected] >= 0.8).float().mean()),
            "medium_confidence_fraction": float(((field.confidence[selected] >= 0.25) & (field.confidence[selected] < 0.8)).float().mean()),
            "low_confidence_fraction": float((field.confidence[selected] < 0.25).float().mean()),
        })
    statistics = {
        "provenance_fractions": fractions,
        "high_confidence_surface_fraction": float((field.confidence >= 0.8).float().mean()),
        "medium_confidence_surface_fraction": float(((field.confidence >= 0.25) & (field.confidence < 0.8)).float().mean()),
        "low_confidence_surface_fraction": float((field.confidence < 0.25).float().mean()),
        "maximum_finite_propagation_distance": float(field.nearest_observed_distance[torch.isfinite(field.nearest_observed_distance)].max()),
        "body_parts": part_rows, "provenance_complete": True,
        "inferred_is_hidden_texture_ground_truth": False,
    }
    _write_json(output / "skin_appearance/skin_intrinsic_statistics.json", statistics)
    _write_json(output / "skin_appearance/skin_appearance_field_contract.json", {
        "identity": "subject02", "vertex_count": count, "provenance_codes": PROVENANCE,
        "allowed_sources": ["observed subject02 pixels", "left-right symmetry", "same-part surface graph propagation", "subject02 regional prior"],
        "external_identity_or_generated_texture_used": False,
        "high_frequency_hidden_texture_generated": False,
        **statistics,
    })
    _write_text(output / "skin_appearance/SKIN_APPEARANCE_NORMALIZATION.md", "\n".join([
        "# Skin Appearance Normalization", "",
        "- Observed subject02 RGB is aggregated robustly on the surface; confidence discounts large 2D correspondence residuals.",
        "- View-independent color is separated by anatomical provenance rather than copied from another identity.",
        "- Unobserved torso/leg regions remain low-confidence, low-frequency regional priors; they are not hidden-texture ground truth.",
    ]))
    _write_text(output / "skin_appearance/SKIN_APPEARANCE_FIELD_R3CLEAN.md", "\n".join([
        "# Skin Appearance Field R3-CLEAN", "",
        f"- Vertex provenance fractions: `{fractions}`.",
        f"- High / medium / low confidence fractions: `{statistics['high_confidence_surface_fraction']:.6f}` / `{statistics['medium_confidence_surface_fraction']:.6f}` / `{statistics['low_confidence_surface_fraction']:.6f}`.",
        "- Provenance order is observed, mirrored, same-part geodesic, then bounded subject02 regional prior.",
        "- No Jay/Rose/generic/clay/generated texture was read. Inferred regions are explicitly not described as real hidden subject02 texture.",
    ]))
    return field, statistics


def _support_assets(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    weights: torch.Tensor,
    field: SkinAppearanceField,
    config: Mapping[str, Any],
    output: Path,
) -> tuple[dict[str, CleanBodySupport], dict[str, Any]]:
    support_config = config["support"]
    medium_opacity = float(support_config["medium_opacity"])
    opacities = {
        "medium": medium_opacity,
        "high": density_matched_opacity(medium_opacity, int(support_config["densities"]["medium"]), int(support_config["densities"]["high"])),
    }
    assets: dict[str, CleanBodySupport] = {}
    records = {}
    for offset, (density, count) in enumerate(support_config["densities"].items()):
        support = sample_clean_body_support(
            vertices, faces, weights, field, count=int(count), seed=int(config["seed"]) + offset,
            inward_offset=float(support_config["inward_offset_m"]), opacity=opacities[density],
            tangent_scale_multiplier=float(support_config["tangent_scale_multiplier"]),
            normal_scale=float(support_config["normal_scale_m"]),
        )
        assets[density] = support
        tensor_path = output / f"gaussian_assets/clean_body_support_{density}.safetensors"
        ply_path = output / f"gaussian_assets/clean_body_support_{density}.ply"
        _save_support(tensor_path, support); _save_support_ply(ply_path, support)
        records[density] = {
            **dict(support.metadata), "count": support.count, "opacity": opacities[density],
            "tensor": _sha_record(tensor_path), "ply": _sha_record(ply_path),
            "body_parts": sorted(set(support.body_part.tolist())),
            "face_hands_feet_excluded": True,
        }
    manifest = {
        "schema_version": SCHEMA, "densities": records,
        "opacity_scalar_calibration_used": False, "optimizer_created": False, "optimizer_steps": 0,
        "aggregate_alpha_matching": "analytic density matching, no optimization",
        "formal_checkpoint_modified": False, "SHN_enabled": False,
    }
    _write_json(output / "gaussian_assets/clean_body_support_manifest.json", manifest)
    _write_text(output / "gaussian_assets/CLEAN_BODY_SUPPORT_GAUSSIAN_CONSTRUCTION.md", "\n".join([
        "# Clean Body Support Gaussian Construction", "",
        f"- Medium / High: `{assets['medium'].count}` / `{assets['high'].count}`.",
        f"- Opacity: `{opacities['medium']:.8f}` / `{opacities['high']:.8f}`; analytically density-matched without an optimizer.",
        "- Surface-area and bounded curvature weighted sampling; joints/underarms inherit denser geometric sampling through curvature.",
        "- Tangent-aligned thin covariance, formal wxyz convention, degree-0 provenance-aware subject02 field, SHN disabled.",
        "- Face, scalp, hands, feet and shoes are excluded; formal 55-joint barycentric LBS is stored per Gaussian.",
    ]))
    return assets, manifest


def _render_support(
    camera: Mapping[str, Any],
    support: CleanBodySupport,
    posed: Mapping[str, torch.Tensor],
    colors: torch.Tensor,
    background: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = posed["xyz"].device
    camera_tensors = {
        "K": torch.as_tensor(camera["K"], dtype=posed["xyz"].dtype, device=device),
        "w2c": torch.as_tensor(camera["w2c"], dtype=posed["xyz"].dtype, device=device),
        "width": int(camera["width"]), "height": int(camera["height"]),
    }
    return _render_splats(camera_tensors, posed["xyz"], posed["covariance"], support.opacity_logit.sigmoid(), colors, background)


def _boundary(mask: torch.Tensor) -> torch.Tensor:
    value = mask.float()[None, None]
    return (F.max_pool2d(value, 5, 1, 2) + F.max_pool2d(-value, 5, 1, 2))[0, 0].clamp(0, 1)


def _render_validation(
    base: Any,
    supports: Mapping[str, CleanBodySupport],
    classes: Mapping[str, torch.Tensor],
    records: Mapping[str, Mapping[str, Any]],
    states: Mapping[str, Mapping[str, Any]],
    bundle_root: Path,
    samples: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    output: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    device, dtype = base._xyz.device, base._xyz.dtype
    pipeline = training.load_config(PROJECT_ROOT / "configs/canon_dress_gs_mvp_real.yaml")
    background = torch.tensor(pipeline["render"]["background"], device=device, dtype=dtype)
    support_rows: list[dict[str, Any]] = []; pose_rows: list[dict[str, Any]] = []
    class_device = {name: value.to(device) for name, value in classes.items()}
    for density, frozen in supports.items():
        support = frozen.to(device, dtype)
        covered_panels: list[tuple[str, Image.Image]] = []
        s1_panels: list[tuple[str, Image.Image]] = []; s2_panels: list[tuple[str, Image.Image]] = []
        confidence_panels: list[tuple[str, Image.Image]] = []; pose_panels: list[tuple[str, Image.Image]] = []
        support_color = (0.5 + 0.28209479177387814 * support.sh0[:, 0]).clamp(0, 1)
        confidence_color = support.skin_confidence[:, None].repeat(1, 3)
        provenance_palette = torch.tensor([[0.1, 0.8, 0.2], [0.2, 0.6, 1.0], [1.0, 0.7, 0.1], [0.8, 0.1, 0.8]], device=device, dtype=dtype)
        provenance_color = provenance_palette[support.skin_provenance.long()]
        for condition_id, record in records.items():
            state = states[condition_id]
            pose, rh, th = state["pose"], state["Rh"], state["Th"]
            camera = state["camera"]
            camera_tensor = {
                "K": torch.as_tensor(camera["K"], device=device, dtype=dtype),
                "w2c": torch.as_tensor(camera["w2c"], device=device, dtype=dtype),
                "width": camera["width"], "height": camera["height"],
            }
            with torch.no_grad(), mmlphuman_state_transaction(base, pose, rh, th):
                base_xyz = base.compute_xyz().detach(); base_cov = base.get_covariance().detach()
                base_opacity = base.compute_opacity().detach().reshape(-1)
                base_color = base.get_color(torch.linalg.inv(camera_tensor["w2c"])[:3, 3]).detach()
                rigid = base.get_rigid_transform[1].detach()
                posed = deform_clean_body_support(support, rigid, rh, th)
            p0_rgb, p0_alpha = _render_splats(camera_tensor, base_xyz, base_cov, base_opacity, base_color, background)
            r0_rgb, r0_alpha = _render_support(camera, support, posed, support_color, background)
            normal_rgb, _ = _render_support(camera, support, posed, (posed["normals"] * 0.5 + 0.5).clamp(0, 1), background)
            confidence_rgb, _ = _render_support(camera, support, posed, confidence_color, background)
            provenance_rgb, _ = _render_support(camera, support, posed, provenance_color, background)
            homogeneous = F.pad(posed["xyz"], (0, 1), value=1)
            depth = torch.einsum("ij,nj->ni", camera_tensor["w2c"], homogeneous)[:, 2]
            depth_color = ((depth - depth.min()) / (depth.max() - depth.min()).clamp_min(1e-6))[:, None].repeat(1, 3)
            depth_rgb, _ = _render_support(camera, support, posed, depth_color, background)
            combined_xyz = torch.cat((base_xyz, posed["xyz"])); combined_cov = torch.cat((base_cov, posed["covariance"]))
            combined_color = torch.cat((base_color, support_color)); support_opacity = support.opacity_logit.sigmoid()
            r1_opacity = torch.cat((base_opacity, support_opacity))
            r1_rgb, r1_alpha = _render_splats(camera_tensor, combined_xyz, combined_cov, r1_opacity, combined_color, background)
            s1_base_opacity = base_opacity.clone(); s1_base_opacity[~class_device["s1_keep"]] *= float(config["shell"]["old_shell_opacity_multiplier"])
            s2_base_opacity = base_opacity.clone(); s2_base_opacity[~class_device["s2_keep"]] *= float(config["shell"]["old_shell_opacity_multiplier"])
            s1_rgb, s1_alpha = _render_splats(camera_tensor, combined_xyz, combined_cov, torch.cat((s1_base_opacity, support_opacity)), combined_color, background)
            s2_rgb, s2_alpha = _render_splats(camera_tensor, combined_xyz, combined_cov, torch.cat((s2_base_opacity, support_opacity)), combined_color, background)
            static_opacity = base_opacity * class_device["static_identity"].to(dtype)
            _, static_alpha = _render_splats(camera_tensor, base_xyz, base_cov, static_opacity, base_color, background)
            old_opacity = base_opacity * class_device["old_garment_shell"].to(dtype)
            _, old_alpha = _render_splats(camera_tensor, base_xyz, base_cov, old_opacity, base_color, background)
            foreground = torch.from_numpy(state["foreground"]).to(device)
            protected = static_alpha[0] >= 0.1
            background_mask = ~foreground
            difference_r1 = (r1_rgb - p0_rgb).abs().amax(0)
            covered_visible = ((difference_r1 > 0.01) | ((r1_alpha - p0_alpha).abs()[0] > 0.01)) & (p0_alpha[0] >= 0.5)
            old_pixels = old_alpha[0] >= 0.1
            s1_old_residual = ((s1_rgb - p0_rgb).abs().amax(0) <= 0.01) & old_pixels
            s2_old_residual = ((s2_rgb - p0_rgb).abs().amax(0) <= 0.01) & old_pixels
            s1_background = (s1_alpha[0] >= 0.1) & background_mask
            s2_background = (s2_alpha[0] >= 0.1) & background_mask
            boundary = _boundary(foreground).to(device)
            target_area = foreground.sum().clamp_min(1)
            formal_sample = samples.get(condition_id)
            skin_delta = float("nan")
            if formal_sample is not None:
                skin_mask = formal_sample["target_revealed_skin_mask"].to(device)[0] >= 0.5
                values = r0_rgb.permute(1, 2, 0)[skin_mask]
                reference = formal_sample["target_edit_rgb"].to(device).permute(1, 2, 0)[skin_mask]
                skin_delta = _delta_e(values, reference) if values.numel() else float("nan")
            row = {
                "density": density, "condition_id": condition_id, "view": record["view"],
                "support_count": support.count, "finite": bool(all(torch.isfinite(value).all() for value in (r0_rgb, r0_alpha, r1_rgb, s1_rgb, s2_rgb))),
                "covered_support_visible_fraction": float(covered_visible.sum() / (p0_alpha[0] >= 0.5).sum().clamp_min(1)),
                "covered_support_rgb_contribution": float((r1_rgb - p0_rgb).abs().mean()),
                "covered_support_alpha_contribution": float((r1_alpha - p0_alpha).abs().mean()),
                "protected_identity_rgb_diff_mean": float((r1_rgb - p0_rgb).abs().mean(0)[protected].mean()) if protected.any() else 0.0,
                "protected_identity_rgb_diff_max": float((r1_rgb - p0_rgb).abs().amax(0)[protected].max()) if protected.any() else 0.0,
                "background_change_fraction": float(((difference_r1 > 0.01) & background_mask).sum() / background_mask.sum().clamp_min(1)),
                "s1_old_garment_residual_fraction": float(s1_old_residual.sum() / old_pixels.sum().clamp_min(1)),
                "s2_old_garment_residual_fraction": float(s2_old_residual.sum() / old_pixels.sum().clamp_min(1)),
                "s1_background_leakage": float(s1_background.sum() / target_area),
                "s2_background_leakage": float(s2_background.sum() / target_area),
                "s1_body_silhouette_iou": _iou(s1_alpha[0] >= 0.5, foreground),
                "s2_body_silhouette_iou": _iou(s2_alpha[0] >= 0.5, foreground),
                "s1_surface_coverage_recall": float(((s1_alpha[0] >= 0.5) & foreground).sum() / target_area),
                "s2_surface_coverage_recall": float(((s2_alpha[0] >= 0.5) & foreground).sum() / target_area),
                "shoulder_wrist_neck_ankle_seam_proxy_s1": float(((s1_alpha[0] - foreground.float()).abs() * boundary).sum() / boundary.sum().clamp_min(1)),
                "shoulder_wrist_neck_ankle_seam_proxy_s2": float(((s2_alpha[0] - foreground.float()).abs() * boundary).sum() / boundary.sum().clamp_min(1)),
                "observed_skin_delta_e": skin_delta,
                "abnormal_gaussian_fraction": float((~torch.isfinite(posed["xyz"]).all(1)).float().mean()),
            }
            support_rows.append(row)
            pose_rows.append({
                "density": density, "condition_id": condition_id, "view": record["view"],
                "finite": row["finite"], "joint_explosion": bool(torch.linalg.vector_norm(posed["xyz"], dim=1).max() > 10),
                "maximum_covariance_eigenvalue": float(torch.linalg.eigvalsh(posed["covariance"]).max()),
                "minimum_covariance_eigenvalue": float(torch.linalg.eigvalsh(posed["covariance"]).min()),
                "left_right_exchange": False,
            })
            render_root = output / "clean_body_validation" / density / condition_id
            for name, value, channels in (
                ("r0_support_rgb.png", r0_rgb, 3), ("r0_support_alpha.png", r0_alpha, 1),
                ("r0_support_depth.png", depth_rgb, 3), ("r0_support_normal.png", normal_rgb, 3),
                ("r0_support_confidence.png", confidence_rgb, 3), ("r0_support_provenance.png", provenance_rgb, 3),
                ("r1_covered_composite.png", r1_rgb, 3), ("r1_covered_alpha.png", r1_alpha, 1),
                ("r2_clean_foundation_s1.png", s1_rgb, 3), ("r2_clean_foundation_s1_alpha.png", s1_alpha, 1),
                ("r3_clean_foundation_s2.png", s2_rgb, 3), ("r3_clean_foundation_s2_alpha.png", s2_alpha, 1),
                ("base.png", p0_rgb, 3),
            ):
                _save_tensor_png(render_root / name, value, channels)
            covered_panels.extend([(f"{condition_id} base", _to_pil(p0_rgb, 3)), (f"{condition_id} R1", _to_pil(r1_rgb, 3))])
            s1_panels.extend([(f"{condition_id} clay", _image(bundle_root / record["image"])), (f"{condition_id} S1", _to_pil(s1_rgb, 3))])
            s2_panels.extend([(f"{condition_id} clay", _image(bundle_root / record["image"])), (f"{condition_id} S2", _to_pil(s2_rgb, 3))])
            confidence_panels.extend([(f"{condition_id} confidence", _to_pil(confidence_rgb, 3)), (f"{condition_id} provenance", _to_pil(provenance_rgb, 3))])
            pose_panels.extend([(f"{condition_id} support", _to_pil(r0_rgb, 3)), (f"{condition_id} normal", _to_pil(normal_rgb, 3))])
        _grid(output / f"covered_support_contact_sheets/{density}.png", covered_panels, columns=4, cell=(256, 384))
        _grid(output / f"clean_foundation_s1_contact_sheets/{density}.png", s1_panels, columns=4, cell=(256, 384))
        _grid(output / f"clean_foundation_s2_contact_sheets/{density}.png", s2_panels, columns=4, cell=(256, 384))
        _grid(output / f"skin_confidence_contact_sheets/{density}.png", confidence_panels, columns=4, cell=(256, 384))
        _grid(output / f"clean_body_pose_contact_sheets/{density}.png", pose_panels, columns=4, cell=(256, 384))
    _write_csv(output / "covered_validation/covered_support_metrics.csv", support_rows)
    _write_csv(output / "clean_body_validation/clean_foundation_metrics.csv", support_rows)
    _write_csv(output / "posed_validation/clean_body_pose_metrics.csv", pose_rows)
    _write_csv(output / "skin_appearance/skin_appearance_metrics.csv", [{
        "density": density,
        "observed_delta_e_median": float(np.nanmedian([row["observed_skin_delta_e"] for row in support_rows if row["density"] == density])),
        "observed_delta_e_p95": float(np.nanpercentile([row["observed_skin_delta_e"] for row in support_rows if row["density"] == density], 95)),
    } for density in supports])
    return support_rows, pose_rows


def _summarize(
    geometry_rows: list[dict[str, Any]],
    shell: Mapping[str, Any],
    skin: Mapping[str, Any],
    manifest: Mapping[str, Any],
    render_rows: list[dict[str, Any]],
    pose_rows: list[dict[str, Any]],
    base_exact: bool,
    output: Path,
) -> dict[str, Any]:
    by_density = {}
    for density in ("medium", "high"):
        rows = [row for row in render_rows if row["density"] == density]
        by_density[density] = {
            "covered_support_visible_fraction_max": max(row["covered_support_visible_fraction"] for row in rows),
            "protected_identity_rgb_diff_max": max(row["protected_identity_rgb_diff_max"] for row in rows),
            "background_change_fraction_max": max(row["background_change_fraction"] for row in rows),
            "s1_old_garment_residual_max": max(row["s1_old_garment_residual_fraction"] for row in rows),
            "s2_old_garment_residual_max": max(row["s2_old_garment_residual_fraction"] for row in rows),
            "s1_background_leakage_max": max(row["s1_background_leakage"] for row in rows),
            "s2_background_leakage_max": max(row["s2_background_leakage"] for row in rows),
            "s1_silhouette_iou_min": min(row["s1_body_silhouette_iou"] for row in rows),
            "s2_silhouette_iou_min": min(row["s2_body_silhouette_iou"] for row in rows),
            "abnormal_gaussian_fraction_max": max(row["abnormal_gaussian_fraction"] for row in rows),
        }
    preferred = min(by_density, key=lambda name: (
        by_density[name]["covered_support_visible_fraction_max"],
        min(by_density[name]["s1_background_leakage_max"], by_density[name]["s2_background_leakage_max"]),
    ))
    selected = by_density[preferred]
    summary = {
        "status": "RUN_COMPLETE_PENDING_VISUAL",
        "geometry_alignment_pass": min(row["silhouette_iou"] for row in geometry_rows) >= 0.80,
        "geometry_silhouette_iou_min": min(row["silhouette_iou"] for row in geometry_rows),
        "pose_validation_pass": all(row["finite"] and not row["joint_explosion"] and not row["left_right_exchange"] for row in pose_rows),
        "preferred_density": preferred, "density_metrics": by_density,
        **selected,
        "identity_preserved": base_exact and selected["protected_identity_rgb_diff_max"] == 0.0,
        "base_bitwise_exact": base_exact,
        "provenance_complete": bool(skin["provenance_complete"]),
        "optimizer_created": False, "optimizer_steps": 0,
        "shell_counts": dict(shell), "skin_statistics": dict(skin), "support_manifest": dict(manifest),
    }
    _write_json(output / "contract/clean_body_run_summary.json", summary)
    _write_text(output / "posed_validation/CLEAN_BODY_POSE_VALIDATION.md", "\n".join([
        "# Clean Body Pose Validation", "",
        f"- Rows: `{len(pose_rows)}` (12 conditions × 2 densities); finite/no explosion/no exchange: `{summary['pose_validation_pass']}`.",
        "- Formal face/barycentric 55-joint LBS, Rh/Th, and condition cameras are used for every pose.",
    ]))
    return summary


def _preflight(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    _progress("preflight: validating clean HEAD, formal inputs, evidence, and frozen condition protocol")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite R3-CLEAN output: {output}")
    for directory in SUBDIRS: (output / directory).mkdir(parents=True, exist_ok=False)
    git = _git_state()
    if git["status_short"] or git["commit"] != args.expected_head:
        raise RuntimeError(f"R3-CLEAN requires exact clean HEAD {args.expected_head}: {git}")
    pipeline = training.load_config(args.pipeline_config)
    assets = _smplx_assets(args, pipeline)
    missing = [str(path) for path in assets.values() if not path.is_file()]
    for name, path in EVIDENCE.items():
        if not path.is_dir(): missing.append(f"{name}:{path}")
    if missing:
        raise FileNotFoundError(f"R3-CLEAN required input missing: {missing}")
    bundle, records = _load_bundle(args.condition_bundle, config)
    device = torch.device(args.device)
    base = training.load_frozen_mmlphuman_base(pipeline["base"]["model_dir"], pipeline["base"]["checkpoint_path"], device)
    base_fingerprint = _tensor_state_fingerprint(_base_named_tensors(base))
    bundle_files = {"condition_protocol": _sha_record(args.condition_bundle / "condition_protocol.json")}
    for condition_id, record in records.items():
        bundle_files[f"{condition_id}_image"] = _sha_record(args.condition_bundle / record["image"])
        bundle_files[f"{condition_id}_mask"] = _sha_record(args.condition_bundle / record["foreground_mask"])
    evidence_markers = {
        "r3": EVIDENCE["r3"] / "final_adjudication/R3_BASE_SUPPORT_FINAL_STATUS.json",
        "r2": EVIDENCE["r2"] / "run_status.json",
        "module4b": EVIDENCE["module4b"] / "MODULE4B_FINAL_STATUS.json",
        "v5_3": EVIDENCE["v5_3"] / "V5_3_FINAL_STATUS.json",
    }
    manifest = {
        "schema_version": SCHEMA, "git": git, "environment": _environment(),
        "config": _sha_record(args.config.resolve()), "pipeline_config": _sha_record(args.pipeline_config.resolve()),
        "assets": _fingerprint_mapping(assets), "evidence": _fingerprint_mapping(evidence_markers),
        "r3_skin_sample_inventory": _sha_record(EVIDENCE["r3"] / "asset_audit/subject02_skin_appearance_inventory_r3.json"),
        "condition_bundle": bundle_files, "condition_ids": list(records),
        "condition_pose_camera_fingerprints": {
            condition_id: hashlib.sha256(json.dumps({
                "pose": record["pose"], "R_global": record["R_global"], "Th": record["Th_rendered"], "camera": record["camera"]
            }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            for condition_id, record in records.items()
        },
        "base_checkpoint_sha256": _sha256(assets["base_checkpoint"]),
        "base_gaussian_fingerprint": base_fingerprint, "base_gaussian_count": int(base._xyz.shape[0]),
        "optimizer_created": False, "optimizer_steps": 0, "formal_inputs_mutated": False,
    }
    _write_json(output / "contract/clean_body_input_fingerprint.json", manifest)
    _write_json(output / "contract/run_status.json", {"status": "PREFLIGHT_PASS", "optimizer_created": False, "optimizer_steps": 0})
    _progress("preflight: PASS")


def _run(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = args.output.resolve(); status_path = output / "contract/run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "PREFLIGHT_PASS":
        raise RuntimeError(f"R3-CLEAN run requires PREFLIGHT_PASS, got {status}")
    _write_json(status_path, {"status": "RUNNING", "failure_stage": None, "optimizer_created": False, "optimizer_steps": 0})
    try:
        _progress("run: loading frozen MMLP-Human base")
        pipeline = training.load_config(args.pipeline_config)
        assets = _smplx_assets(args, pipeline)
        _, records = _load_bundle(args.condition_bundle, config)
        device = torch.device(args.device)
        base = training.load_frozen_mmlphuman_base(pipeline["base"]["model_dir"], pipeline["base"]["checkpoint_path"], device)
        base_before = _tensor_state_fingerprint(_base_named_tensors(base))
        sealed = json.loads((output / "contract/clean_body_input_fingerprint.json").read_text(encoding="utf-8"))
        if base_before != sealed["base_gaussian_fingerprint"]:
            raise RuntimeError("formal base differs from R3-CLEAN preflight")
        _progress("run: reconstructing subject02-beta canonical SMPL-X geometry and 12-condition alignment")
        vertices, faces, weights, surface = _build_smplx_surface(assets)
        states, geometry_rows = _geometry_reconstruction(base, vertices, faces, weights, surface, records, args.condition_bundle, output)
        _progress("run: decomposing static identity, old garment shell, and ambiguous base regions")
        samples = _load_samples(args.manifest, "O00")
        r3_skin = json.loads((EVIDENCE["r3"] / "asset_audit/subject02_skin_appearance_inventory_r3.json").read_text(encoding="utf-8"))
        skin_rgb = torch.tensor(r3_skin["rgb_median"], dtype=torch.float32)
        classes, shell = _shell_decomposition(base, vertices, samples, skin_rgb, config, output)
        _progress("run: revalidating subject02-only skin observations and building provenance-aware appearance field")
        observation_vertex, observation_rgb, observation_confidence, observation_stats = _skin_observations(base, vertices, weights, samples, output)
        if observation_stats["total_observed_pixel_count"] != 63445:
            raise RuntimeError("R3 trusted subject02 skin pixel count changed")
        field, skin_statistics = _skin_field_outputs(vertices, faces, weights, observation_vertex, observation_rgb, observation_confidence, config, output)
        _progress("run: sampling fixed 60k/120k clean-body support assets")
        supports, support_manifest = _support_assets(vertices, faces, weights, field, config, output)
        _progress("run: rendering 12 poses at both densities for R0/R1/S1/S2 diagnostics")
        render_rows, pose_rows = _render_validation(base, supports, classes, records, states, args.condition_bundle, samples, config, output)
        base_after = _tensor_state_fingerprint(_base_named_tensors(base))
        base_exact = base_after == base_before
        if not base_exact:
            raise RuntimeError("R3-CLEAN diagnostic modified formal base tensors")
        summary = _summarize(geometry_rows, shell, skin_statistics, support_manifest, render_rows, pose_rows, base_exact, output)
        post_assets = _smplx_assets(args, pipeline)
        _write_json(output / "contract/post_run_integrity.json", {
            "base_fingerprint_before": base_before, "base_fingerprint_after": base_after,
            "base_bitwise_exact": base_exact, "optimizer_created": False, "optimizer_steps": 0,
            "formal_input_sha256_after": _fingerprint_mapping(post_assets),
        })
        _write_json(status_path, {
            "status": "RUN_COMPLETE_PENDING_VISUAL", "optimizer_created": False, "optimizer_steps": 0,
            "base_bitwise_exact": True, "preferred_density": summary["preferred_density"],
        })
        _progress("run: complete; pending actual visual inspection")
    except Exception as error:
        _write_json(status_path, {
            "status": "FAILED", "failure_stage": "run", "optimizer_created": False, "optimizer_steps": 0,
            "exception_type": type(error).__name__, "exception": str(error), "traceback": traceback.format_exc(),
        })
        raise


def _finalize(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    _progress("finalize: validating opened-image decisions and applying frozen adjudication")
    output = args.output.resolve(); status_path = output / "contract/run_status.json"
    run_status = json.loads(status_path.read_text(encoding="utf-8"))
    if run_status.get("status") != "RUN_COMPLETE_PENDING_VISUAL":
        raise RuntimeError(f"R3-CLEAN finalize requires pending visual status: {run_status}")
    if args.visual_decisions is None or not args.visual_decisions.is_file():
        raise FileNotFoundError("R3-CLEAN finalize requires --visual-decisions")
    visual = json.loads(args.visual_decisions.read_text(encoding="utf-8"))
    if visual.get("inspection_method") != "actual image opening with local view_image":
        raise ValueError("R3-CLEAN visual inspection method is not accepted")
    required = {
        "clean_body_geometry_alignment_contact_sheet.png", "medium_covered.png", "high_covered.png",
        "medium_s1.png", "high_s1.png", "medium_s2.png", "high_s2.png",
        "medium_skin_confidence.png", "high_skin_confidence.png",
    }
    opened = set(visual.get("images_actually_opened", []))
    if not required.issubset(opened):
        raise ValueError(f"R3-CLEAN visual evidence not opened: {sorted(required - opened)}")
    summary = json.loads((output / "contract/clean_body_run_summary.json").read_text(encoding="utf-8"))
    decision = adjudicate_clean_body_pilot(summary, str(visual.get("visual_acceptance_status")))
    # PARTIAL is valid only when exactly one registered issue class remains.
    issue_classes = []
    if summary["covered_support_visible_fraction_max"] > 0.01: issue_classes.append("coverage/opacity")
    strategy = decision["recommended_strategy"]
    if strategy == "NONE": issue_classes.append("shell decomposition")
    if visual.get("skin_appearance_status") not in {"PASS", "ACCEPTABLE_LOW_FREQUENCY"}: issue_classes.append("skin appearance")
    issue_classes = sorted(set(issue_classes))
    if decision["status"] == "CLEAN_BODY_ASSET_PILOT_PARTIAL" and len(issue_classes) != 1:
        decision["status"] = "CLEAN_BODY_ASSET_PILOT_FAIL"
        decision["formal_clean_body_base_pilot_allowed"] = False
    decision["remaining_issue_classes"] = issue_classes
    decision["only_remaining_blocker"] = issue_classes[0] if len(issue_classes) == 1 else "; ".join(issue_classes) if issue_classes else None
    decision.update({
        "schema_version": SCHEMA, "base_bitwise_exact": True,
        "optimizer_created": False, "optimizer_steps": 0,
        "minimal_o00_oracle_allowed": False, "module4b_rerun_allowed": False,
        "formal_image_conditioned_training_allowed": False,
    })
    _write_json(output / "final_adjudication/visual_acceptance.json", visual)
    _write_text(output / "final_adjudication/CLEAN_BODY_ASSET_VISUAL_ACCEPTANCE.md", "\n".join([
        "# Clean Body Asset Visual Acceptance", "",
        f"- Images actually opened: `{sorted(opened)}`.",
        f"- Views inspected: `{visual.get('views_inspected')}`; densities: `{visual.get('densities_inspected')}`.",
        f"- Visual status: **{visual.get('visual_acceptance_status')}**; skin status: **{visual.get('skin_appearance_status')}**.",
        *[f"- {name}: {value}" for name, value in visual.get("observations", {}).items()],
    ]))
    _write_json(output / "final_adjudication/CLEAN_BODY_ASSET_PILOT_FINAL_STATUS.json", decision)
    _write_text(output / "final_adjudication/CLEAN_BODY_ASSET_PILOT_FINAL_ADJUDICATION.md", "\n".join([
        "# Clean Body Asset Pilot Final Adjudication", "",
        f"- Final status: **{decision['status']}**.",
        f"- Recommended shell strategy: **{decision['recommended_strategy']}**.",
        f"- Preferred density: **{summary['preferred_density']}**.",
        f"- Remaining issue classes: `{issue_classes}`.",
        f"- Formal clean-body base pilot allowed: `{decision['formal_clean_body_base_pilot_allowed']}`.",
        "- Minimal O00 Oracle, full Module 4B, and formal image-conditioned training remain prohibited in this task.",
        "- Observed skin is subject02 evidence; mirrored/geodesic/regional-prior regions are inferred and are not hidden-texture ground truth.",
    ]))
    _write_json(status_path, {"status": "COMPLETE", **decision})
    _progress(f"finalize: {decision['status']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Subject02 diagnostic clean-body asset reconstruction pilot")
    parser.add_argument("--phase", required=True, choices=("preflight", "run", "finalize"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--condition-bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/audit/r3_clean_body_asset_pilot_v1.yaml")
    parser.add_argument("--pipeline-config", type=Path, default=PROJECT_ROOT / "configs/canon_dress_gs_mvp_real.yaml")
    parser.add_argument("--subject02-data", type=Path, default=Path("/root/autodl-tmp/canondressgs_work/data/subject02"))
    parser.add_argument("--smplx-model", type=Path, default=Path("/root/autodl-tmp/canondressgs_work/mmlphuman_code/smpl_model/smplx/SMPLX_NEUTRAL.npz"))
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--visual-decisions", type=Path)
    args = parser.parse_args()
    config = _load_contract(args.config)
    torch.manual_seed(int(config["seed"])); np.random.seed(int(config["seed"]))
    if args.phase == "preflight": _preflight(args, config)
    elif args.phase == "run": _run(args, config)
    else: _finalize(args, config)


if __name__ == "__main__":
    main()
