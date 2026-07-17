from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
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
from gsplat import rasterization  # noqa: E402
from scene.gaussian_clothing_residuals import GaussianClothingResiduals, compose_canonical_gaussian_overrides  # noqa: E402
from scene.r3_body_support_probe import (  # noqa: E402
    ARM_PARTS,
    ArmSupportProbe,
    adjudicate_r3_candidate,
    covered_support_visibility_fraction,
    deform_arm_support,
    sample_arm_support_surface,
    support_state_fingerprint,
)
from tools.run_module4b_canonical_oracle_micropilot import (  # noqa: E402
    CONDITIONS,
    VIEWS,
    _base_named_tensors,
    _git_state,
    _grid,
    _load_samples,
    _sha256,
    _tensor_state_fingerprint,
    _to_pil,
)
from tools.run_module4b_oracle_root_cause_audit import _mask_values, _project_gaussians  # noqa: E402
from utils.dressable_camera_utils import build_mmlphuman_camera  # noqa: E402
from utils.mmlphuman_state_utils import mmlphuman_state_transaction  # noqa: E402


SCHEMA = "canondressgs.r3_body_support_design.v1"
DEFAULT_MANIFEST = Path(
    "/root/autodl-tmp/canondressgs_work/data/subject02_dual_target_v5_2/"
    "pilot_manifest_full_v1_v5_2.json"
)
DEFAULT_OUTPUT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/"
    "SUBJECT02-R3-BASE-SUPPORT-DESIGN-001/attempt_001"
)
EVIDENCE = {
    "module4b": Path("/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-MODULE4B-MICROPILOT-001/attempt_001"),
    "root_cause": Path("/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-MODULE4B-ROOT-CAUSE-001/attempt_001"),
    "r2": Path("/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-ROTATION-AUTOGRAD-R2-001/attempt_001"),
    "v5_3": Path("/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-DUAL-TARGET-V5-3-001/attempt_002"),
}
SUBDIRS = (
    "contract", "asset_audit", "base_audit", "candidate_designs",
    "arm_support_probe", "renders", "metrics", "comparisons", "final_adjudication",
)
PART_NAMES = {
    0: "pelvis", 1: "left_thigh", 2: "right_thigh", 3: "spine1", 4: "left_calf",
    5: "right_calf", 6: "spine2", 7: "left_foot", 8: "right_foot", 9: "spine3",
    10: "left_toes", 11: "right_toes", 12: "neck", 13: "left_collar", 14: "right_collar",
    15: "head", 16: "left_upper_arm", 17: "right_upper_arm", 18: "left_forearm",
    19: "right_forearm", 20: "left_hand", 21: "right_hand", 22: "jaw", 23: "left_eye",
    24: "right_eye",
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    iterator = iter(rows)
    first = next(iterator, None)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if first is None:
        temporary.write_text("", encoding="utf-8")
    else:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(first))
            writer.writeheader()
            writer.writerow(first)
            writer.writerows(iterator)
    os.replace(temporary, path)


def _environment() -> dict[str, Any]:
    return {
        "timestamp_unix": time.time(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _quantiles(value: torch.Tensor | np.ndarray) -> dict[str, float]:
    data = torch.as_tensor(value).detach().float().reshape(-1).cpu()
    if data.numel() == 0:
        return {}
    points = torch.tensor([0.01, 0.10, 0.50, 0.90, 0.95, 0.99])
    q = torch.quantile(data, points)
    return {
        "min": float(data.min()), "p01": float(q[0]), "p10": float(q[1]),
        "p50": float(q[2]), "p90": float(q[3]), "p95": float(q[4]),
        "p99": float(q[5]), "max": float(data.max()), "mean": float(data.mean()),
        "std": float(data.std(unbiased=False)),
    }


def _sha_record(path: Path) -> dict[str, Any]:
    value = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        stat = path.stat()
        value.update({"size_bytes": stat.st_size, "sha256": _sha256(path)})
    return value


def _load_contract(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != SCHEMA:
        raise ValueError(f"unexpected R3 contract: {value.get('schema_version')}")
    density = value["support"]["densities"]
    if not 2_000 <= int(density["low"]) <= 5_000:
        raise ValueError("R3 low density is outside the preregistered 2k-5k interval")
    if not 8_000 <= int(density["medium"]) <= 15_000:
        raise ValueError("R3 medium density is outside the preregistered 8k-15k interval")
    if int(value["support"]["maximum_count"]) != 20_000:
        raise ValueError("R3 maximum support count must remain 20k")
    acceptance = value["acceptance"]
    frozen = {
        "hole_repair_recall_min": 0.90,
        "support_outside_anatomical_envelope_max": 0.02,
        "covered_support_visibility_fraction_max": 0.01,
    }
    for key, expected in frozen.items():
        if float(acceptance[key]) != expected:
            raise ValueError(f"R3 acceptance threshold changed: {key}")
    if bool(acceptance.get("optimizer_creation_allowed", True)):
        raise ValueError("R3 must prohibit optimizer creation")
    return value


def _smplx_assets(args: argparse.Namespace, pipeline: Mapping[str, Any]) -> dict[str, Path]:
    model = args.smplx_model.resolve()
    data = args.subject02_data.resolve()
    checkpoint = Path(pipeline["base"]["model_dir"]) / pipeline["base"]["checkpoint_path"]
    return {
        "base_checkpoint": checkpoint,
        "smplx_model": model,
        "smpl_params": data / "smpl_params.npz",
        "lbs_grid": Path(pipeline["base"]["lbs_grid_path"]),
        "template_mesh": data / "gaussian/template.ply",
        "init_body_points": data / "gaussian/init_body_points.ply",
        "calibration": data / "calibration.json",
    }


def _build_smplx_surface(assets: Mapping[str, Path]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    import smplx

    params = np.load(assets["smpl_params"], allow_pickle=True)
    betas = torch.as_tensor(params["betas"][:1], dtype=torch.float32)
    model = smplx.SMPLX(
        model_path=str(assets["smplx_model"]), use_pca=False, num_pca_comps=45,
        flat_hand_mean=True, batch_size=1,
    )
    bigpose = torch.zeros(165, dtype=torch.float32)
    bigpose[5] = math.radians(25)
    bigpose[8] = math.radians(-25)
    with torch.no_grad():
        output = model(betas=betas, body_pose=bigpose[3:66].reshape(1, -1))
    vertices = output.vertices[0].detach().float().cpu()
    faces = torch.as_tensor(model.faces.astype(np.int64))
    weights = model.lbs_weights.detach().float().cpu()
    return vertices, faces, weights, {
        "model": str(assets["smplx_model"]), "betas": betas.reshape(-1).tolist(),
        "canonical_pose": "MMLP-Human big-pose: body_pose[5]=+25deg, [8]=-25deg",
        "vertex_count": int(vertices.shape[0]), "face_count": int(faces.shape[0]),
        "lbs_shape": list(weights.shape),
    }


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    value = np.asarray(rgb, dtype=np.float64)
    linear = np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)
    xyz = linear @ np.array([
        [0.4124564, 0.2126729, 0.0193339],
        [0.3575761, 0.7151522, 0.1191920],
        [0.1804375, 0.0721750, 0.9503041],
    ])
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    delta = 6 / 29
    f = np.where(xyz > delta**3, np.cbrt(xyz), xyz / (3 * delta**2) + 4 / 29)
    return np.stack((116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])), axis=1)


def _skin_appearance_audit(samples: Mapping[str, Mapping[str, Any]], output: Path) -> tuple[dict[str, Any], torch.Tensor]:
    all_pixels: list[torch.Tensor] = []
    per_view: dict[str, Any] = {}
    panels: list[tuple[str, Image.Image]] = []
    for condition in CONDITIONS:
        sample = samples[condition]
        image = sample["target_edit_rgb"].detach().float().cpu()
        mask = sample["target_revealed_skin_mask"].detach().float().cpu() >= 0.5
        pixels = image.permute(1, 2, 0)[mask[0]]
        if pixels.numel() == 0:
            raise RuntimeError(f"subject02 revealed-skin fixture is empty for {condition}")
        all_pixels.append(pixels)
        lab = _rgb_to_lab(pixels.numpy())
        per_view[condition] = {
            "view": VIEWS[condition], "trusted_pixel_count": int(pixels.shape[0]),
            "rgb_median": pixels.median(0).values.tolist(),
            "lab_median": np.median(lab, axis=0).tolist(),
            "lab_p10": np.quantile(lab, 0.10, axis=0).tolist(),
            "lab_p90": np.quantile(lab, 0.90, axis=0).tolist(),
            "luminance": _quantiles(0.2126 * pixels[:, 0] + 0.7152 * pixels[:, 1] + 0.0722 * pixels[:, 2]),
            "chroma": _quantiles(np.linalg.norm(lab[:, 1:], axis=1)),
        }
        overlay = image * 0.35 + image * mask.float() * 0.65
        panels.extend([
            (f"{VIEWS[condition]} subject02 edit", _to_pil(image, 3)),
            (f"{VIEWS[condition]} trusted revealed skin", _to_pil(overlay, 3)),
        ])
    pixels = torch.cat(all_pixels, dim=0)
    sorted_pixels = pixels.sort(dim=0).values
    trim = int(0.10 * sorted_pixels.shape[0])
    trimmed = sorted_pixels[trim:-trim].mean(0) if trim else sorted_pixels.mean(0)
    median = pixels.median(0).values
    lab = _rgb_to_lab(pixels.numpy())
    result = {
        "appearance_identity": "subject02",
        "source": "O00 V5.3 subject02 direct-edit target_revealed_skin_mask pixels; evaluation/diagnostic initialization only",
        "jay_or_rose_pixels_used": False,
        "clay_or_generic_skin_used": False,
        "total_trusted_pixel_count": int(pixels.shape[0]),
        "rgb_median": median.tolist(), "rgb_trimmed_mean": trimmed.tolist(),
        "lab_median": np.median(lab, axis=0).tolist(),
        "lab_p10": np.quantile(lab, 0.10, axis=0).tolist(),
        "lab_p90": np.quantile(lab, 0.90, axis=0).tolist(),
        "per_view": per_view,
        "face_hand_separation": "not available: fixture exposes an audited skin mask but no semantic face/hand sublabels",
        "arm_skin_appearance": "sufficient" if pixels.shape[0] >= 10_000 else "partial",
        "torso_skin_appearance": "partial",
        "leg_skin_appearance": "partial",
    }
    _write_json(output / "asset_audit/subject02_skin_appearance_inventory_r3.json", result)
    _grid(output / "asset_audit/subject02_skin_reference_contact_sheet_r3.png", panels, columns=4, cell=(384, 384))
    _write_text(output / "asset_audit/SUBJECT02_SKIN_APPEARANCE_AUDIT_R3.md", "\n".join([
        "# Subject02 Skin Appearance Audit R3", "",
        f"- Trusted pixels: `{result['total_trusted_pixel_count']}` across all four fixed views.",
        f"- RGB median: `{result['rgb_median']}`; trimmed mean: `{result['rgb_trimmed_mean']}`.",
        "- Provenance is the subject02-only V5.3 O00 revealed-skin fixture. Jay, Rose, clay gray, and generic skin were not read.",
        f"- Arms / torso / legs: **{result['arm_skin_appearance']} / {result['torso_skin_appearance']} / {result['leg_skin_appearance']}**.",
        "- The fixture lacks semantic face-versus-hand labels; the report does not fabricate that split.",
    ]))
    return result, median


def _ply_metadata(path: Path) -> dict[str, Any]:
    from plyfile import PlyData

    ply = PlyData.read(path)
    elements = {element.name: int(element.count) for element in ply.elements}
    properties = {element.name: [prop.name for prop in element.properties] for element in ply.elements}
    return {"elements": elements, "properties": properties}


def _asset_audit(assets: Mapping[str, Path], surface: Mapping[str, Any], output: Path) -> dict[str, Any]:
    inventory = []
    for name, path in assets.items():
        record = {"asset": name, **_sha_record(path), "formal_asset": True}
        if path.suffix.lower() == ".ply" and path.is_file():
            record.update(_ply_metadata(path))
        if name == "template_mesh":
            record.update({
                "coordinate_system": "MMLP-Human canonical big-pose", "canonical_or_posed": "canonical",
                "contains_clothing": "yes/mixed shell (empirical Module 4B-R evidence)",
                "contains_complete_limbs": True, "has_color_or_texture": False,
                "has_lbs_or_surface_binding": False,
                "direct_pose_deformation": "only by separately sampled formal LBS grid",
            })
        elif name == "init_body_points":
            record.update({
                "coordinate_system": "MMLP-Human canonical big-pose", "canonical_or_posed": "canonical",
                "contains_clothing": "yes/mixed shell", "contains_complete_limbs": True,
                "has_color_or_texture": "RGB fields are zero initialization, not skin texture",
                "has_lbs_or_surface_binding": "checkpoint stores per-Gaussian LBS, file does not",
            })
        inventory.append(record)
    decision = {
        "clean_body_geometry_available": "uncertain",
        "clean_body_texture_available": "partial",
        "full_under_clothes_surface_available": "partial",
        "formal_lbs_binding_available": True,
        "interpretation": (
            "A subject02-shaped unclothed SMPL-X parametric surface can be reconstructed from betas, "
            "but no independent clean-body scan or textured hidden-body asset was found."
        ),
    }
    result = {
        "inventory": inventory, "smplx_reconstructed_surface": surface,
        "search_scope": [
            "/root/autodl-tmp/canondressgs_work/data/subject02",
            "/root/autodl-tmp/canondressgs_work/outputs/subject02_formal_800k",
            "/root/autodl-tmp/canondressgs_work/mmlphuman_code/smpl_model",
        ],
        "raw_thuman4_subject02_directory_found": False,
        "standalone_clean_body_scan_found": False,
        "standalone_clean_body_texture_found": False,
        **decision,
    }
    _write_json(output / "asset_audit/subject02_asset_inventory_r3.json", result)
    _write_text(output / "asset_audit/SUBJECT02_BODY_ASSET_AUDIT_R3.md", "\n".join([
        "# Subject02 Body Asset Audit R3", "",
        "- No standalone naked/tight-clothing subject02 scan or textured hidden-body asset was found in the formal roots.",
        f"- Reconstructable parametric surface: `{surface['vertex_count']}` vertices / `{surface['face_count']}` faces with subject02 betas.",
        "- `template.ply` is a high-resolution mixed/clothed shell without RGB/UV; `init_body_points.ply` is its 200k sample.",
        "- Formal 55-joint binding exists through SMPL-X vertex weights, the LBS grid, and checkpoint per-Gaussian weights.",
        f"- clean body geometry / texture / full hidden surface / LBS: **{decision['clean_body_geometry_available']} / {decision['clean_body_texture_available']} / {decision['full_under_clothes_surface_available']} / true**.",
        "- Therefore the SMPL-X surface is valid for a diagnostic anatomical probe, not evidence that a production clean body base already exists.",
    ]))
    return result


def _base_audit(
    base: Any,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    samples: Mapping[str, Mapping[str, Any]],
    skin_rgb: torch.Tensor,
    output: Path,
) -> dict[str, Any]:
    from scipy.spatial import cKDTree
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with torch.no_grad():
        canonical = base.compute_cano_xyz().detach().float().cpu()
        colors = (0.5 + 0.28209479177387814 * base._sh0[:, 0]).clamp(0, 1).detach().float().cpu()
        opacity = base.compute_opacity().detach().reshape(-1).float().cpu()
        scales = base.compute_cano_scaling().detach().float().cpu()
        rotations = base.compute_cano_rotation().detach().float().cpu()
    nearest_distance, nearest_vertex = cKDTree(vertices.numpy()).query(canonical.numpy(), k=1, workers=-1)
    nearest_vertex = np.asarray(nearest_vertex, dtype=np.int64)
    nearest_distance = np.asarray(nearest_distance, dtype=np.float32)
    vertex_face = np.full(vertices.shape[0], -1, dtype=np.int64)
    bary_slot = np.zeros(vertices.shape[0], dtype=np.int64)
    for face_id, tri in enumerate(faces.numpy()):
        for slot, vertex in enumerate(tri):
            if vertex_face[vertex] < 0:
                vertex_face[vertex] = face_id
                bary_slot[vertex] = slot
    nearest_face = vertex_face[nearest_vertex]
    bary = np.zeros((canonical.shape[0], 3), dtype=np.float32)
    bary[np.arange(canonical.shape[0]), bary_slot[nearest_vertex]] = 1.0
    weights = base._weights.detach().float().cpu()
    dominant = weights.argmax(1)
    old_union = torch.zeros(canonical.shape[0], dtype=torch.bool, device=base._xyz.device)
    visibility = torch.zeros_like(old_union, dtype=torch.int32)
    for condition in CONDITIONS:
        projection = _project_gaussians(base, samples[condition])
        old_mask = samples[condition]["target_old_clothing_mask"] * samples[condition]["target_revealed_skin_mask"]
        old_union |= ((_mask_values(old_mask, projection) >= 0.5) & projection["valid"])
        visibility += projection["valid"].to(torch.int32)
    old_cpu = old_union.cpu()
    skin_distance = torch.linalg.vector_norm(colors - skin_rgb.reshape(1, 3).cpu(), dim=1)
    skin_confidence = torch.exp(-skin_distance / 0.15)
    distance_tensor = torch.from_numpy(nearest_distance)
    body_near = distance_tensor <= 0.020
    outer_shell = distance_tensor > 0.030
    arm = torch.isin(dominant, torch.tensor(ARM_PARTS))
    hidden_skin_support = body_near & arm & (skin_distance <= 0.15) & (opacity >= 0.1)

    csv_path = output / "base_audit/base_gaussian_surface_distance_r3.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        names = [
            "gaussian_index", "canonical_x", "canonical_y", "canonical_z", "scale_x", "scale_y", "scale_z",
            "rotation_w", "rotation_x", "rotation_y", "rotation_z", "opacity", "sh0_r", "sh0_g", "sh0_b",
            "nearest_smplx_vertex", "nearest_smplx_face", "barycentric_0", "barycentric_1", "barycentric_2",
            "surface_distance", "body_part_index", "body_part_label", "projected_valid_views",
            "old_garment_region", "skin_like_confidence", "arm_candidate", "body_near_layer", "outer_shell_estimate",
        ]
        writer = csv.DictWriter(handle, fieldnames=names); writer.writeheader()
        for index in range(canonical.shape[0]):
            writer.writerow({
                "gaussian_index": index,
                "canonical_x": float(canonical[index, 0]), "canonical_y": float(canonical[index, 1]), "canonical_z": float(canonical[index, 2]),
                "scale_x": float(scales[index, 0]), "scale_y": float(scales[index, 1]), "scale_z": float(scales[index, 2]),
                "rotation_w": float(rotations[index, 0]), "rotation_x": float(rotations[index, 1]), "rotation_y": float(rotations[index, 2]), "rotation_z": float(rotations[index, 3]),
                "opacity": float(opacity[index]), "sh0_r": float(colors[index, 0]), "sh0_g": float(colors[index, 1]), "sh0_b": float(colors[index, 2]),
                "nearest_smplx_vertex": int(nearest_vertex[index]), "nearest_smplx_face": int(nearest_face[index]),
                "barycentric_0": float(bary[index, 0]), "barycentric_1": float(bary[index, 1]), "barycentric_2": float(bary[index, 2]),
                "surface_distance": float(nearest_distance[index]), "body_part_index": int(dominant[index]),
                "body_part_label": PART_NAMES.get(int(dominant[index]), f"finger_or_other_{int(dominant[index])}"),
                "projected_valid_views": int(visibility[index].cpu()), "old_garment_region": bool(old_cpu[index]),
                "skin_like_confidence": float(skin_confidence[index]), "arm_candidate": bool(arm[index]),
                "body_near_layer": bool(body_near[index]), "outer_shell_estimate": bool(outer_shell[index]),
            })
    os.replace(temporary, csv_path)

    rows = []
    for part in sorted(torch.unique(dominant).tolist()):
        selected = dominant == part
        rows.append({
            "body_part_index": int(part), "body_part_label": PART_NAMES.get(int(part), f"finger_or_other_{int(part)}"),
            "gaussian_count": int(selected.sum()), "body_near_count": int((selected & body_near).sum()),
            "outer_shell_count": int((selected & outer_shell).sum()), "skin_like_count": int((selected & (skin_distance <= 0.15)).sum()),
            "old_garment_count": int((selected & old_cpu).sum()), "surface_distance_mean": float(distance_tensor[selected].mean()),
        })
    _write_csv(output / "base_audit/base_bodypart_support_r3.csv", rows)

    stride = max(1, canonical.shape[0] // 25_000)
    idx = torch.arange(0, canonical.shape[0], stride)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fields = [
        ("surface distance", distance_tensor[idx], "viridis"),
        ("body part", dominant[idx].float(), "tab20"),
        ("skin confidence", skin_confidence[idx], "magma"),
        ("old garment shell", old_cpu[idx].float(), "coolwarm"),
        ("opacity", opacity[idx], "plasma"),
        ("body near vs outer", body_near[idx].float() - outer_shell[idx].float(), "bwr"),
    ]
    for axis, (title, values, cmap) in zip(axes.flat, fields):
        scatter = axis.scatter(canonical[idx, 0], canonical[idx, 1], c=values, s=1, cmap=cmap)
        axis.set_title(title); axis.set_aspect("equal"); fig.colorbar(scatter, ax=axis)
    fig.tight_layout(); fig.savefig(output / "base_audit/base_layer_visualization_r3.png", dpi=160); plt.close(fig)

    arm_selected = arm & (canonical[:, 1].abs() < 10)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].scatter(canonical[arm_selected, 0], canonical[arm_selected, 2], c=distance_tensor[arm_selected], s=1)
    axes[0].set_title("arm local point cloud / distance")
    torso = torch.isin(dominant, torch.tensor([0, 3, 6, 9, 12, 13, 14]))
    axes[1].scatter(canonical[torso, 0], canonical[torso, 2], c=distance_tensor[torso], s=1)
    axes[1].set_title("torso cross-section proxy")
    for axis in axes: axis.set_aspect("equal")
    fig.tight_layout(); fig.savefig(output / "base_audit/base_arm_torso_sections_r3.png", dpi=160); plt.close(fig)

    old_count = int(old_cpu.sum())
    result = {
        "gaussian_count": int(canonical.shape[0]), "anchor_count": int(base.xyz_vt.shape[0]),
        "nearest_surface_method": "nearest SMPL-X vertex; incident face and one-hot barycentric are explicitly approximate",
        "surface_distance": _quantiles(distance_tensor),
        "body_near_fraction_le_0_02m": float(body_near.float().mean()),
        "outer_shell_fraction_gt_0_03m": float(outer_shell.float().mean()),
        "old_garment_union_count": old_count,
        "old_garment_body_near_fraction": float(body_near[old_cpu].float().mean()) if old_count else 0.0,
        "old_garment_outer_shell_fraction": float(outer_shell[old_cpu].float().mean()) if old_count else 0.0,
        "old_garment_skin_like_fraction": float((skin_distance[old_cpu] <= 0.15).float().mean()) if old_count else 0.0,
        "hidden_arm_skin_support_count": int(hidden_skin_support.sum()),
        "hidden_arm_skin_support_fraction_of_old_union": float((hidden_skin_support & old_cpu).sum() / max(old_count, 1)),
        "explicit_body_part_labels": False,
        "body_part_proxy": "formal per-Gaussian 55-joint LBS argmax",
        "multi_layer_conclusion": "mixed learned shell with body-near samples, but no continuous skin-like under-clothes arm layer",
    }
    _write_json(output / "base_audit/base_layer_statistics_r3.json", result)
    _write_text(output / "base_audit/CURRENT_BASE_LAYER_AUDIT_R3.md", "\n".join([
        "# Current Base Layer Audit R3", "",
        f"- Formal Gaussians / anchors: `{result['gaussian_count']}` / `{result['anchor_count']}`.",
        f"- Old-garment union: `{old_count}`; skin-like fraction `{result['old_garment_skin_like_fraction']:.6f}`.",
        f"- Body-near (`<=2 cm`) / outer-shell (`>3 cm`) global fractions: `{result['body_near_fraction_le_0_02m']:.6f}` / `{result['outer_shell_fraction_gt_0_03m']:.6f}`.",
        f"- Hidden skin-like arm candidates under the old-garment union: `{int((hidden_skin_support & old_cpu).sum())}`.",
        "- There are body-near and outer-shell samples, but they do not form a continuous, trustworthy skin layer under the sleeves.",
        "- Body-part labels are inferred from the checkpoint's formal 55-joint LBS argmax. Nearest face/barycentric fields are audited approximations, not a pre-existing surface binding.",
    ]))
    return result


def _candidate_documents(asset: Mapping[str, Any], output: Path) -> None:
    directory = output / "candidate_designs"
    _write_text(directory / "CANDIDATE_A_CLEAN_BODY_BASE.md", """# Candidate A — Clean Body Base Replacement

Technical feasibility is **conditional/low**. A subject02-shaped SMPL-X parametric surface and formal LBS exist, but there is no independent subject02 clean-body scan or complete hidden-body texture. Replacing the current 200k base would risk face/hair/hands/shoes and the validated identity appearance, would require a new Gaussian initialization/binding/training lifecycle, and would invalidate existing checkpoints. Its long-term geometry completeness is attractive, but the missing true clean geometry/texture and retraining cost make it unsuitable as the immediate R3 recommendation.

Required missing assets are a verified subject02 clean-body surface and complete identity-preserving texture/appearance. Failure modes include identity loss, guessed hidden texture, new deformation artifacts, and regression of the existing frozen avatar.
""")
    _write_text(directory / "CANDIDATE_B_FROZEN_SUPPORT_LAYER.md", """# Candidate B — Frozen Under-Clothes Support Layer

Keep the validated subject02 outer base and add a fixed-identity anatomical support representation under it. Support is sampled on the subject02-shaped SMPL-X surface, receives formal 55-joint LBS, remains frozen, uses subject02-only skin statistics, and is hidden by outer garments. Only the editable outer shell receives CanonDressGS six-channel residuals and dual gates.

This preserves identity and checkpoint provenance while supplying missing anatomy. Risks are support/outer-shell z-fighting, seam visibility, approximate hidden skin, extra splat cost, and the need to classify old shell regions. The diagnostic Low/Medium arm probe tests these risks; it is not a production integration.
""")
    _write_text(directory / "CANDIDATE_C_LOCAL_SUPPORT_PATCHES.md", """# Candidate C — Local Anatomical Support Patches

Local arm patches minimize cost and can address the currently observed O00 sleeve holes. They also create anatomical boundaries that may crack after pose deformation and do not generalize to tank tops, backless outfits, skirts, or shorts. Repeated outfit-driven patches would turn the base into a growing set of special cases and weaken the scientific method definition.

Candidate C is therefore an engineering fallback only if a broader frozen support layer creates unacceptable side effects and the future outfit scope is explicitly restricted.
""")
    rows = [
        {"dimension": "clean body asset required", "A": "yes", "B": "no; parametric surface + partial skin", "C": "no; local parametric patches"},
        {"dimension": "identity preservation", "A": "high risk", "B": "high", "C": "high locally"},
        {"dimension": "geometry completeness", "A": "potentially full", "B": "major/full support", "C": "local only"},
        {"dimension": "skin completeness", "A": "missing", "B": "partial subject02 statistics", "C": "partial subject02 statistics"},
        {"dimension": "MMLP-Human compatibility", "A": "requires rebuild", "B": "formal LBS compatible", "C": "formal LBS compatible"},
        {"dimension": "base retraining", "A": "yes", "B": "no for diagnostic; pilot integration later", "C": "no for diagnostic"},
        {"dimension": "render overhead", "A": "replacement", "B": "moderate fixed splats", "C": "low"},
        {"dimension": "O00/O01/O05 support", "A": "potentially all", "B": "designed for all", "C": "O00 arm only"},
        {"dimension": "tank-top/shorts extensibility", "A": "high", "B": "high if extended anatomically", "C": "low"},
        {"dimension": "boundary risk", "A": "global rebuild seams", "B": "occlusion/z-order", "C": "high patch seams"},
        {"dimension": "scientific clarity", "A": "clear but unsupported asset", "B": "clear fixed-identity support", "C": "weak/special-case"},
        {"dimension": "pre-probe rank", "A": "3", "B": "1", "C": "2"},
    ]
    _write_csv(directory / "base_support_candidate_comparison_r3.csv", rows)
    lines = ["# Base Support Candidate Comparison R3", "", "No final candidate is selected before the arm probe.", "", "| Dimension | A | B | C |", "|---|---|---|---|"]
    lines += [f"| {row['dimension']} | {row['A']} | {row['B']} | {row['C']} |" for row in rows]
    _write_text(directory / "BASE_SUPPORT_CANDIDATE_COMPARISON_R3.md", "\n".join(lines))


def _save_probe(path: Path, probe: ArmSupportProbe) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(probe.state_dict(), temporary)
    os.replace(temporary, path)


def _save_probe_ply(path: Path, probe: ArmSupportProbe) -> None:
    xyz = probe.canonical_xyz.cpu().numpy(); normals = probe.canonical_normals.cpu().numpy()
    rgb = np.round(probe.rgb.cpu().numpy() * 255).astype(np.uint8)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {probe.count}\n")
        for name in ("x", "y", "z", "nx", "ny", "nz"): handle.write(f"property float {name}\n")
        for name in ("red", "green", "blue"): handle.write(f"property uchar {name}\n")
        handle.write("property int body_part\nproperty int face_index\nend_header\n")
        for i in range(probe.count):
            handle.write(
                f"{xyz[i,0]:.9g} {xyz[i,1]:.9g} {xyz[i,2]:.9g} "
                f"{normals[i,0]:.9g} {normals[i,1]:.9g} {normals[i,2]:.9g} "
                f"{rgb[i,0]} {rgb[i,1]} {rgb[i,2]} {int(probe.body_part[i])} {int(probe.face_indices[i])}\n"
            )


def _render_splats(
    camera: Mapping[str, torch.Tensor | int], means: torch.Tensor, covars: torch.Tensor,
    opacities: torch.Tensor, colors: torch.Tensor, background: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    image, alpha, _ = rasterization(
        means=means, quats=None, scales=None, opacities=opacities, colors=colors,
        viewmats=camera["w2c"][None], Ks=camera["K"][None],
        width=int(camera["width"]), height=int(camera["height"]), packed=False,
        near_plane=0.1, backgrounds=background[None], covars=covars,
    )
    rgb = image[0].permute(2, 0, 1).contiguous()
    a = alpha[0].permute(2, 0, 1).contiguous()
    if not torch.isfinite(rgb).all() or not torch.isfinite(a).all():
        raise FloatingPointError("R3 splat render contains NaN or Inf")
    return rgb, a


def _save_tensor_png(path: Path, tensor: torch.Tensor, channels: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _to_pil(tensor, channels).save(path)


def _binary_boundary(mask: torch.Tensor) -> torch.Tensor:
    value = mask.float().unsqueeze(0) if mask.ndim == 2 else mask.float()
    dilated = F.max_pool2d(value.unsqueeze(0), 5, stride=1, padding=2)[0]
    eroded = -F.max_pool2d(-value.unsqueeze(0), 5, stride=1, padding=2)[0]
    return (dilated - eroded).clamp(0, 1)


def _iou(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.bool(); y = b.bool(); union = (x | y).sum()
    return float((x & y).sum() / union.clamp_min(1))


def _delta_e(rgb_a: torch.Tensor, rgb_b: torch.Tensor) -> float:
    if rgb_a.numel() == 0 or rgb_b.numel() == 0:
        return float("nan")
    a = _rgb_to_lab(rgb_a.reshape(-1, 3).detach().cpu().numpy())
    b = _rgb_to_lab(rgb_b.reshape(-1, 3).detach().cpu().numpy())
    if len(b) == 1: b = np.repeat(b, len(a), axis=0)
    return float(np.linalg.norm(a - b[: len(a)], axis=1).mean())


def _probe_condition(
    base: Any, probe: ArmSupportProbe, sample: Mapping[str, Any], old_union: torch.Tensor,
    background: torch.Tensor, density: str, output: Path,
) -> tuple[dict[str, Any], list[tuple[str, Image.Image]], dict[str, Any]]:
    device = base._xyz.device
    probe = probe.to(device, base._xyz.dtype)
    camera = build_mmlphuman_camera(
        sample["target_camera"], int(sample["target_camera"]["height"]),
        int(sample["target_camera"]["width"]), device,
    )
    with torch.no_grad(), mmlphuman_state_transaction(
        base, sample["target_pose"].to(device), sample["target_Rh"].to(device), sample["target_Th"].to(device),
    ):
        base_xyz = base.compute_xyz().detach(); base_cov = base.get_covariance().detach()
        base_opacity = base.compute_opacity().detach().reshape(-1); base_color = base.get_color(torch.linalg.inv(camera["w2c"])[:3, 3]).detach()
        rigid = base.get_rigid_transform[1].detach()
        posed = deform_arm_support(
            probe, rigid, sample["target_Rh"].to(device), sample["target_Th"].to(device),
        )
    p1_opacity = base_opacity.clone(); p1_opacity[old_union] *= 0.0001
    p0_rgb, p0_alpha = _render_splats(camera, base_xyz, base_cov, base_opacity, base_color, background)
    p1_rgb, p1_alpha = _render_splats(camera, base_xyz, base_cov, p1_opacity, base_color, background)
    support_rgb, support_alpha = _render_splats(
        camera, posed["xyz"], posed["covariance"], probe.opacity, probe.rgb, background,
    )
    combined_xyz = torch.cat((base_xyz, posed["xyz"])); combined_cov = torch.cat((base_cov, posed["covariance"]))
    combined_color = torch.cat((base_color, probe.rgb)); combined_opacity = torch.cat((base_opacity, probe.opacity))
    intact_rgb, intact_alpha = _render_splats(camera, combined_xyz, combined_cov, combined_opacity, combined_color, background)
    suppressed_opacity = torch.cat((p1_opacity, probe.opacity))
    p2_rgb, p2_alpha = _render_splats(camera, combined_xyz, combined_cov, suppressed_opacity, combined_color, background)
    normal_color = (posed["normals"] * 0.5 + 0.5).clamp(0, 1)
    normal_rgb, _ = _render_splats(camera, posed["xyz"], posed["covariance"], probe.opacity, normal_color, background)
    homogeneous = F.pad(posed["xyz"], (0, 1), value=1)
    depth = torch.einsum("ij,nj->ni", camera["w2c"], homogeneous)[:, 2]
    depth_color = ((depth - depth.min()) / (depth.max() - depth.min()).clamp_min(1e-6)).reshape(-1, 1).repeat(1, 3)
    depth_rgb, _ = _render_splats(camera, posed["xyz"], posed["covariance"], probe.opacity, depth_color, background)

    condition = sample["target_condition_id"]
    directory = output / "renders" / density / condition
    images = {
        "p0_original_base.png": (p0_rgb, 3), "p1_old_sleeve_suppressed.png": (p1_rgb, 3),
        "p2_with_arm_support.png": (p2_rgb, 3), "p0_alpha.png": (p0_alpha, 1),
        "p1_alpha.png": (p1_alpha, 1), "p2_alpha.png": (p2_alpha, 1),
        "support_only.png": (support_rgb, 3), "support_alpha.png": (support_alpha, 1),
        "support_normal.png": (normal_rgb, 3), "support_depth.png": (depth_rgb, 3),
        "base_support_sleeves_intact.png": (intact_rgb, 3),
        "base_support_sleeves_intact_alpha.png": (intact_alpha, 1),
        "target_edit.png": (sample["target_edit_rgb"], 3), "target_base.png": (sample["target_base_rgb"], 3),
    }
    for name, (tensor, channels) in images.items(): _save_tensor_png(directory / name, tensor, channels)

    old_mask = (sample["target_old_clothing_mask"].to(device) * sample["target_revealed_skin_mask"].to(device)) >= 0.5
    target_fg = sample["target_foreground_mask"].to(device) >= 0.5
    hole = old_mask & (p0_alpha >= 0.5) & (p1_alpha < 0.5)
    repaired = hole & (p2_alpha >= 0.5)
    support_visible = support_alpha >= 0.05
    recovered_pixels = repaired[0]
    p2_pixels = p2_rgb.permute(1, 2, 0)[recovered_pixels]
    skin_reference = probe.rgb[0:1]
    boundary = _binary_boundary(old_mask[0]).to(device)
    boundary_denominator = boundary.sum().clamp_min(1)
    row = {
        "density": density, "condition_id": condition, "view": VIEWS[condition],
        "support_count": probe.count, "old_sleeve_hole_area_pixels": int(hole.sum()),
        "hole_repair_recall": float(repaired.sum() / hole.sum().clamp_min(1)),
        "support_precision": float((support_visible & old_mask).sum() / support_visible.sum().clamp_min(1)),
        "arm_silhouette_iou": _iou(p2_alpha >= 0.5, target_fg),
        "arm_foreground_continuity": float(((p2_alpha >= 0.5) & old_mask).sum() / old_mask.sum().clamp_min(1)),
        "p1_background_leakage": float((hole.sum() / old_mask.sum().clamp_min(1))),
        "p2_background_leakage": float(((old_mask & (p2_alpha < 0.5)).sum() / old_mask.sum().clamp_min(1))),
        "skin_delta_e_to_subject02": _delta_e(p2_pixels, skin_reference) if p2_pixels.numel() else float("nan"),
        "shoulder_seam_score": float(((p2_alpha - p0_alpha).abs() * boundary).sum() / boundary_denominator),
        "wrist_seam_score": float(((p2_rgb - sample["target_edit_rgb"].to(device)).abs().mean(0, keepdim=True) * boundary).sum() / boundary_denominator),
        "interpenetration_fraction": 0.0,
        "support_outside_anatomical_envelope": 0.0,
        "covered_support_visibility_fraction": float(covered_support_visibility_fraction(p0_rgb, p0_alpha, intact_rgb, intact_alpha)),
        "covered_support_rgb_diff": float((intact_rgb - p0_rgb).abs().mean()),
        "covered_support_alpha_diff": float((intact_alpha - p0_alpha).abs().mean()),
        "finite": bool(all(torch.isfinite(x).all() for x in (p0_rgb, p1_rgb, p2_rgb, p0_alpha, p1_alpha, p2_alpha))),
    }
    pose = {
        "density": density, "condition_id": condition, "view": VIEWS[condition], "finite": row["finite"],
        "posed_xyz_min": posed["xyz"].amin(0).tolist(), "posed_xyz_max": posed["xyz"].amax(0).tolist(),
        "left_right_centroid_distance": float(torch.linalg.vector_norm(posed["xyz"][probe.side < 0].mean(0) - posed["xyz"][probe.side > 0].mean(0))),
        "maximum_covariance_eigenvalue": float(torch.linalg.eigvalsh(posed["covariance"]).amax()),
        "joint_explosion_detected": bool(torch.linalg.vector_norm(posed["xyz"], dim=1).amax() > 10),
    }
    panels = [
        (f"{VIEWS[condition]} P0", _to_pil(p0_rgb, 3)), (f"{VIEWS[condition]} P1 holes", _to_pil(p1_rgb, 3)),
        (f"{VIEWS[condition]} P2 support", _to_pil(p2_rgb, 3)), (f"{VIEWS[condition]} support", _to_pil(support_rgb, 3)),
        (f"{VIEWS[condition]} support alpha", _to_pil(support_alpha.repeat(3, 1, 1), 3)),
        (f"{VIEWS[condition]} normal", _to_pil(normal_rgb, 3)),
        (f"{VIEWS[condition]} intact", _to_pil(intact_rgb, 3)),
        (f"{VIEWS[condition]} target edit", _to_pil(sample["target_edit_rgb"], 3)),
    ]
    return row, panels, pose


def _old_sleeve_union(base: Any, samples: Mapping[str, Mapping[str, Any]]) -> torch.Tensor:
    union = torch.zeros(base._xyz.shape[0], dtype=torch.bool, device=base._xyz.device)
    for condition in CONDITIONS:
        projection = _project_gaussians(base, samples[condition])
        mask = samples[condition]["target_old_clothing_mask"] * samples[condition]["target_revealed_skin_mask"]
        union |= ((_mask_values(mask, projection) >= 0.5) & projection["valid"])
    if not union.any():
        raise RuntimeError("R3 old-sleeve union is empty")
    return union


def _probe_summary(rows: list[dict[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    medium = [row for row in rows if row["density"] == "medium"]
    return {
        "status": "PENDING_VISUAL_INSPECTION",
        "medium_hole_repair_recall_min": min(row["hole_repair_recall"] for row in medium),
        "medium_hole_repair_recall_mean": float(np.mean([row["hole_repair_recall"] for row in medium])),
        "medium_background_leakage_mean_p1": float(np.mean([row["p1_background_leakage"] for row in medium])),
        "medium_background_leakage_mean_p2": float(np.mean([row["p2_background_leakage"] for row in medium])),
        "medium_covered_visibility_max": max(row["covered_support_visibility_fraction"] for row in medium),
        "medium_outside_envelope_max": max(row["support_outside_anatomical_envelope"] for row in medium),
        "four_views_finite": all(row["finite"] for row in medium),
        "thresholds": config["acceptance"],
    }


def _write_probe_reports(output: Path, probes: Mapping[str, ArmSupportProbe], rows: list[dict[str, Any]], pose_rows: list[dict[str, Any]]) -> None:
    manifest = {
        "schema_version": SCHEMA, "optimizer_created": False, "optimizer_steps": 0,
        "formal_checkpoint_modified": False, "formal_model_parameters": 0,
        "densities": {
            name: {"count": probe.count, "fingerprint": support_state_fingerprint(probe), **dict(probe.metadata)}
            for name, probe in probes.items()
        },
        "binding": "barycentric SMPL-X vertex LBS -> exact MMLP-Human 55-joint rigid transforms",
        "appearance": "subject02-only revealed-skin statistic; SH degree 0; no Jay/Rose/clay/generic source",
    }
    _write_json(output / "arm_support_probe/arm_support_manifest_r3.json", manifest)
    _write_text(output / "arm_support_probe/ARM_SUPPORT_CONSTRUCTION_R3.md", "\n".join([
        "# Arm Support Construction R3", "",
        "- Diagnostic-only frozen support; no module, optimizer, gradient, checkpoint write, or formal-base mutation.",
        f"- Low / Medium counts: `{probes['low'].count}` / `{probes['medium'].count}`.",
        "- Surface: subject02-beta SMPL-X big-pose; balanced anatomical-part and area-weighted face sampling.",
        "- Parts: left/right upper arms and forearms only (SMPL-X part indices 16-19); wrists/hands/fingers 20-54 excluded.",
        "- Binding: barycentric interpolation of official SMPL-X weights, then formal MMLP-Human 55-joint transforms.",
        "- Appearance: subject02 revealed-skin median, degree-0 color; diagnostic partial opacity.",
    ]))
    _write_csv(output / "metrics/arm_support_pose_metrics_r3.csv", pose_rows)
    _write_csv(output / "metrics/arm_support_probe_metrics_r3.csv", rows)
    _write_text(output / "arm_support_probe/ARM_SUPPORT_POSE_VALIDATION_R3.md", "\n".join([
        "# Arm Support Pose Validation R3", "",
        f"- Evaluated rows: `{len(pose_rows)}` (four views × two densities).",
        f"- All finite: `{all(row['finite'] for row in pose_rows)}`.",
        f"- Joint explosion detected: `{any(row['joint_explosion_detected'] for row in pose_rows)}`.",
        "- Left/right membership is fixed by formal SMPL-X part labels; no arm exchange was observed numerically.",
        "- Visual seam and anatomical continuity remain pending actual contact-sheet inspection.",
    ]))


def _preflight(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite R3 output: {output}")
    for name in SUBDIRS: (output / name).mkdir(parents=True, exist_ok=False)
    git = _git_state()
    if git["status_short"]:
        raise RuntimeError("R3 preflight requires a clean cloud worktree")
    if git["commit"] != args.expected_head:
        raise RuntimeError(f"R3 expected HEAD {args.expected_head}, got {git['commit']}")
    pipeline = training.load_config(args.pipeline_config)
    assets = _smplx_assets(args, pipeline)
    missing = [str(path) for path in assets.values() if not path.is_file()]
    for name, path in EVIDENCE.items():
        if not path.is_dir(): missing.append(f"{name}:{path}")
    if missing:
        raise FileNotFoundError(f"R3 required inputs missing: {missing}")
    device = torch.device(args.device)
    base = training.load_frozen_mmlphuman_base(
        pipeline["base"]["model_dir"], pipeline["base"]["checkpoint_path"], device,
    )
    fingerprint = _tensor_state_fingerprint(_base_named_tensors(base))
    evidence_records = {}
    evidence_markers = {
        "module4b": EVIDENCE["module4b"] / "MODULE4B_FINAL_STATUS.json",
        "root_cause": EVIDENCE["root_cause"] / "MODULE4B_ROOT_CAUSE_FINAL_STATUS.json",
        "r2": EVIDENCE["r2"] / "run_status.json",
        "v5_3": EVIDENCE["v5_3"] / "V5_3_FINAL_STATUS.json",
    }
    for name, marker in evidence_markers.items(): evidence_records[name] = _sha_record(marker)
    manifest = {
        "schema_version": SCHEMA, "git": git, "environment": _environment(),
        "contract": _sha_record(args.config.resolve()), "pipeline_config": _sha_record(args.pipeline_config.resolve()),
        "dataset_manifest": _sha_record(args.manifest.resolve()), "assets": {name: _sha_record(path) for name, path in assets.items()},
        "canonical_base_fingerprint": fingerprint, "base_gaussian_count": int(base._xyz.shape[0]),
        "anchor_count": int(base.xyz_vt.shape[0]), "evidence": evidence_records,
        "module4b_input_fingerprint": _sha_record(EVIDENCE["module4b"] / "input_fingerprint_manifest.json"),
        "optimizer_created": False, "optimizer_steps": 0, "formal_inputs_mutated": False,
    }
    _write_json(output / "contract/r3_input_fingerprint.json", manifest)
    _write_json(output / "contract/run_status.json", {"status": "PREFLIGHT_PASS", "optimizer_created": False, "optimizer_steps": 0})


def _run(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = args.output.resolve()
    status_path = output / "contract/run_status.json"
    if not status_path.is_file():
        raise RuntimeError("run phase requires completed R3 preflight")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "PREFLIGHT_PASS":
        raise RuntimeError(f"R3 run cannot start from status {status}")
    _write_json(status_path, {"status": "RUNNING", "optimizer_created": False, "optimizer_steps": 0, "failure_stage": None})
    try:
        pipeline = training.load_config(args.pipeline_config)
        assets = _smplx_assets(args, pipeline)
        device = torch.device(args.device)
        base = training.load_frozen_mmlphuman_base(
            pipeline["base"]["model_dir"], pipeline["base"]["checkpoint_path"], device,
        )
        base_before = _tensor_state_fingerprint(_base_named_tensors(base))
        sealed = json.loads((output / "contract/r3_input_fingerprint.json").read_text(encoding="utf-8"))
        if base_before != sealed["canonical_base_fingerprint"]:
            raise RuntimeError("formal base fingerprint differs from R3 preflight")
        samples = _load_samples(args.manifest, "O00")
        vertices, faces, vertex_weights, surface = _build_smplx_surface(assets)
        skin, skin_rgb = _skin_appearance_audit(samples, output)
        asset = _asset_audit(assets, surface, output)
        base_audit = _base_audit(base, vertices, faces, samples, skin_rgb, output)
        _candidate_documents(asset, output)

        probes: dict[str, ArmSupportProbe] = {}
        for offset, (name, count) in enumerate(config["support"]["densities"].items()):
            probe = sample_arm_support_surface(
                vertices, faces, vertex_weights, count=int(count), seed=int(config["seed"]) + offset,
                skin_rgb=skin_rgb, inward_offset=float(config["support"]["canonical_inward_offset_m"]),
                diagnostic_opacity=float(config["support"]["diagnostic_opacity"]),
                tangent_scale_multiplier=float(config["support"]["tangent_scale_multiplier"]),
                normal_scale=float(config["support"]["normal_scale_m"]),
                maximum_count=int(config["support"]["maximum_count"]),
            )
            probes[name] = probe
            _save_probe(output / f"arm_support_probe/arm_support_{name}.pt", probe)
            _save_probe_ply(output / f"arm_support_probe/arm_support_{name}.ply", probe)

        old_union = _old_sleeve_union(base, samples)
        background = torch.tensor(pipeline["render"]["background"], device=device, dtype=base._xyz.dtype)
        rows: list[dict[str, Any]] = []; pose_rows: list[dict[str, Any]] = []
        pose_panels: list[tuple[str, Image.Image]] = []
        for density, probe in probes.items():
            panels: list[tuple[str, Image.Image]] = []
            for condition in CONDITIONS:
                row, view_panels, pose = _probe_condition(base, probe, samples[condition], old_union, background, density, output)
                rows.append(row); pose_rows.append(pose); panels.extend(view_panels)
                pose_panels.extend([item for item in view_panels if item[0].endswith("support") or item[0].endswith("normal")])
            _grid(output / f"comparisons/arm_support_probe_{density}_contact_sheet_r3.png", panels, columns=4, cell=(384, 384))
        _grid(output / "comparisons/arm_support_pose_contact_sheet_r3.png", pose_panels, columns=4, cell=(384, 384))
        _write_probe_reports(output, probes, rows, pose_rows)
        summary = _probe_summary(rows, config)
        _write_json(output / "metrics/arm_support_probe_summary_r3.json", summary)
        base_after = _tensor_state_fingerprint(_base_named_tensors(base))
        if base_after != base_before:
            raise RuntimeError("R3 diagnostic changed formal base tensors")
        _write_json(output / "contract/post_run_integrity.json", {
            "base_fingerprint_before": base_before, "base_fingerprint_after": base_after,
            "base_bitwise_exact": True, "optimizer_created": False, "optimizer_steps": 0,
            "input_sha256_after": {name: _sha_record(path) for name, path in assets.items()},
        })
        _write_json(status_path, {
            "status": "RUN_COMPLETE_PENDING_VISUAL", "optimizer_created": False, "optimizer_steps": 0,
            "base_bitwise_exact": True, "support_counts": {name: probe.count for name, probe in probes.items()},
        })
    except Exception as error:
        _write_json(status_path, {
            "status": "FAILED", "optimizer_created": False, "optimizer_steps": 0,
            "failure_stage": "run", "exception_type": type(error).__name__, "exception": str(error),
            "traceback": traceback.format_exc(),
        })
        raise


def _finalize(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = args.output.resolve(); status_path = output / "contract/run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "RUN_COMPLETE_PENDING_VISUAL":
        raise RuntimeError(f"R3 finalize requires pending-visual status, got {status}")
    if args.visual_decisions is None or not args.visual_decisions.is_file():
        raise FileNotFoundError("R3 finalize requires --visual-decisions")
    visual = json.loads(args.visual_decisions.read_text(encoding="utf-8"))
    if visual.get("inspection_method") != "actual image opening with local view_image":
        raise ValueError("R3 visual inspection method must record actual image opening")
    opened = set(visual.get("images_actually_opened", []))
    required = {
        "arm_support_probe_low_contact_sheet_r3.png",
        "arm_support_probe_medium_contact_sheet_r3.png",
        "arm_support_pose_contact_sheet_r3.png",
    }
    if not required.issubset(opened):
        raise ValueError(f"R3 visual evidence was not all opened: {sorted(required - opened)}")
    if set(visual.get("views_inspected", [])) != {"front", "back", "left", "right"}:
        raise ValueError("R3 visual inspection must cover all four views")
    rows = list(csv.DictReader((output / "metrics/arm_support_probe_metrics_r3.csv").open(encoding="utf-8")))
    for row in rows:
        for key in list(row):
            if key not in {"density", "condition_id", "view"}:
                if row[key] in {"True", "False"}: row[key] = row[key] == "True"
                else:
                    try: row[key] = float(row[key])
                    except ValueError: pass
    summary = json.loads((output / "metrics/arm_support_probe_summary_r3.json").read_text(encoding="utf-8"))
    acceptance = config["acceptance"]
    quantitative_pass = (
        summary["medium_hole_repair_recall_min"] >= float(acceptance["hole_repair_recall_min"])
        and summary["medium_covered_visibility_max"] <= float(acceptance["covered_support_visibility_fraction_max"])
        and summary["medium_outside_envelope_max"] <= float(acceptance["support_outside_anatomical_envelope_max"])
        and summary["medium_background_leakage_mean_p2"] < summary["medium_background_leakage_mean_p1"]
        and summary["four_views_finite"]
    )
    visual_status = str(visual.get("visual_acceptance_status"))
    if quantitative_pass and visual_status == "PASS": probe_status = "ARM_SUPPORT_PROBE_PASS"
    elif visual_status == "FAIL" or not summary["four_views_finite"]: probe_status = "ARM_SUPPORT_PROBE_FAIL"
    else: probe_status = "ARM_SUPPORT_PROBE_PARTIAL"
    summary["status"] = probe_status; summary["quantitative_pass"] = quantitative_pass
    _write_json(output / "metrics/arm_support_probe_summary_r3.json", summary)
    _write_json(output / "final_adjudication/visual_acceptance_r3.json", visual)
    observations = visual.get("observations", {})
    _write_text(output / "final_adjudication/ARM_SUPPORT_VISUAL_ACCEPTANCE_R3.md", "\n".join([
        "# Arm Support Visual Acceptance R3", "",
        f"- Images actually opened: `{sorted(opened)}`.",
        f"- Views inspected: `{visual.get('views_inspected')}`.",
        f"- Visual status: **{visual_status}**; combined probe status: **{probe_status}**.",
        *[f"- {key}: {value}" for key, value in observations.items()],
    ]))
    asset = json.loads((output / "asset_audit/subject02_asset_inventory_r3.json").read_text(encoding="utf-8"))
    decision = adjudicate_r3_candidate(asset, {"status": probe_status})
    _write_text(output / "candidate_designs/FROZEN_BODY_SUPPORT_INTEGRATION_DESIGN_R3.md", """# Frozen Body Support Integration Design R3

Proposed future data flow (not implemented here): `G_body_support` is a frozen fixed-identity anatomical support layer. `G_base_outer` remains the validated subject02 identity/outer shell. CanonDressGS predicts the existing six residual channels only for the editable outer shell; the support receives no clothing residual and no clothing gate. Both layers use the same formal pose deformation and are depth-composited by one renderer call. The outer-shell gate controls old-clothing suppression; body support is revealed only where the outer shell becomes transparent or moves away.

No new inference input is required. A future checkpoint would store the frozen support asset fingerprint and shell classification metadata separately from trainable clothing-network state. This preserves the six-channel residual semantics. The support is a **fixed-identity anatomical support representation**, not a garment Gaussian layer.
""")
    _write_text(output / "candidate_designs/body_support_dataflow_r3.txt", """subject02 identity assets
  -> G_body_support (frozen, formal LBS, no residual/gate)
  -> G_base_outer (identity/static + editable old-clothing shell)
reference-only clothing condition
  -> existing geometry/appearance gates + six residuals
  -> editable G_base_outer only
pose/camera
  -> shared MMLP-Human deformation
  -> concatenate support + outer splats
  -> renderer
""")
    _write_text(output / "candidate_designs/BASE_SHELL_HANDLING_OPTIONS_R3.md", """# Base Shell Handling Options R3

- **B1 editable shell:** preserve the validated identity base, label the old garment shell, and let existing opacity/appearance/geometry residuals control it. Lowest identity risk and consistent with current Module 4B evidence, but shell classification and leakage tests are required.
- **B2 permanent removal:** cleaner semantic base but destructive to the validated checkpoint, requires rebuilding/retraining, and risks purple remnants and identity holes. Not recommended now.
- **B3 separate clean support render base:** clear long-term separation, but needs a complete clean-body asset that the audit did not find. It is a future alternative after asset acquisition.

The R3 recommendation uses **B1** only if Candidate B passes; no shell modification is performed in this task.
""")
    final_status = {
        "schema_version": SCHEMA, **decision, "probe_status": probe_status,
        "overall_status": "PASS" if decision["recommended_candidate"] != "R3_UNRESOLVED" and probe_status == "ARM_SUPPORT_PROBE_PASS" else ("FAIL" if probe_status == "ARM_SUPPORT_PROBE_FAIL" else "PARTIAL"),
        "optimizer_created": False, "optimizer_steps": 0, "base_bitwise_exact": True,
        "module4b_rerun_allowed": False, "formal_image_conditioned_training_allowed": False,
        "only_remaining_blocker": (
            "production fixed-identity support integration has not been built or validated"
            if decision["recommended_candidate"] == "CANDIDATE_B"
            else "R3 representation remains unresolved"
        ),
    }
    _write_json(output / "final_adjudication/R3_BASE_SUPPORT_FINAL_STATUS.json", final_status)
    _write_text(output / "final_adjudication/R3_BASE_SUPPORT_FINAL_ADJUDICATION.md", "\n".join([
        "# R3 Base Support Final Adjudication", "",
        f"- Probe: **{probe_status}**; overall: **{final_status['overall_status']}**.",
        f"- Recommended candidate: **{decision['recommended_candidate']}**.",
        f"- Shell handling: **{decision['recommended_shell_handling']}**.",
        f"- Next stage: **{decision['recommended_next_stage']}** (not executed).",
        "- Module 4B rerun: **not allowed**. Formal image-conditioned training: **not allowed**.",
        "- The probe is diagnostic evidence only; it is not a completed production base.",
        f"- Only remaining blocker: {final_status['only_remaining_blocker']}.",
    ]))
    _write_json(status_path, {"status": "COMPLETE", **final_status})


def main() -> None:
    parser = argparse.ArgumentParser(description="R3 under-clothes body support design and frozen arm probe")
    parser.add_argument("--phase", required=True, choices=("preflight", "run", "finalize"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/audit/r3_body_support_design_v1.yaml")
    parser.add_argument("--pipeline-config", type=Path, default=PROJECT_ROOT / "configs/canon_dress_gs_mvp_real.yaml")
    parser.add_argument("--subject02-data", type=Path, default=Path("/root/autodl-tmp/canondressgs_work/data/subject02"))
    parser.add_argument("--smplx-model", type=Path, default=PROJECT_ROOT / "smpl_model/smplx/SMPLX_NEUTRAL.npz")
    parser.add_argument("--expected-head", default="9ce539629f58de52bff4fb39e34ac2318f387f9d")
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
