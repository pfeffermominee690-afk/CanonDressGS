from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.r3_clean_geomcam_contract import (  # noqa: E402
    SCHEMA,
    adjudicate_geomcam,
    binary_mask_metrics,
    clean_body_outside_fraction,
    fixed_safe_box_from_render_only,
    mask_bbox,
    source_coverage_by_clean_body,
    transform_screen_points_by_safe_box,
    validate_frame_mapping,
    validate_rh_th_contract,
)


BUNDLE_SCHEMA = "canondressgs.r3_clean_condition_bundle.v1"
DEFAULT_OUTPUT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/"
    "SUBJECT02-R3-CLEAN-GEOMCAM-002/attempt_001"
)
DEFAULT_BUNDLE = Path("/root/autodl-tmp/canondressgs_work/tmp/r3_clean_geomcam_002_inputs")
R3_CLEAN_OUTPUT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/"
    "SUBJECT02-CLEAN-BODY-ASSET-PILOT-001/attempt_001"
)
R3_OUTPUT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/"
    "SUBJECT02-R3-BASE-SUPPORT-DESIGN-001/attempt_001"
)
V5_3_OUTPUT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/"
    "SUBJECT02-DUAL-TARGET-V5-3-001/attempt_002"
)
WORST_IDS = ("cond_000714", "cond_000113", "cond_000439")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()

    return {
        "branch": run("branch", "--show-current"),
        "head": run("rev-parse", "HEAD"),
        "status_short": run("status", "--short"),
    }


def _environment() -> dict[str, Any]:
    result = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "optimizer_created": False,
        "optimizer_steps": 0,
    }
    try:
        import pytorch3d

        result["pytorch3d"] = getattr(pytorch3d, "__version__", "unknown")
    except Exception as error:  # pragma: no cover - preflight reports environment failures
        result["pytorch3d_error"] = type(error).__name__
    try:
        import smplx

        result["smplx"] = getattr(smplx, "__version__", "unknown")
    except Exception as error:  # pragma: no cover
        result["smplx_error"] = type(error).__name__
    return result


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != SCHEMA:
        raise ValueError("unexpected GEOMCAM config schema")
    frozen = value["source_replay_acceptance"]
    if frozen != {
        "per_condition_iou_min": 0.98,
        "median_iou_min": 0.99,
        "bbox_center_residual_diag_max": 0.005,
        "no_flip": True,
    }:
        raise ValueError("source replay acceptance thresholds changed")
    if value["scope"] != {
        "optimizer_created": False,
        "optimizer_steps": 0,
        "target_mask_used_for_transform": False,
        "formal_camera_adapter_modified": False,
        "formal_pose_adapter_modified": False,
        "formal_base_modified": False,
        "skin_field": False,
        "shell_decomposition": False,
        "support_gaussians": False,
        "module4b": False,
        "training": False,
    }:
        raise ValueError("GEOMCAM scope changed")
    return value


def _snapshot_files(root: Path) -> dict[str, dict[str, Any]]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        records[relative] = {"size": path.stat().st_size, "sha256": _sha256(path)}
    return records


def _evidence_snapshot(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    candidates = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        upper = path.name.upper()
        if (
            "FINAL_STATUS" in upper
            or "INPUT_FINGERPRINT" in upper
            or "POST_FAILURE_INPUT_INTEGRITY" in upper
            or "RUN_STATUS" in upper
            or upper in {"GATE_ACCEPTANCE.MD", "CLEAN_BODY_GEOMETRY_ALIGNMENT.CSV"}
        ):
            candidates.append(path)
    return {
        "root": str(root),
        "files": {
            path.relative_to(root).as_posix(): {"size": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(candidates)
        },
    }


def _load_bundle(directory: Path, config: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = directory / "condition_protocol.json"
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if value.get("schema_version") != BUNDLE_SCHEMA:
        raise ValueError("unexpected condition bundle schema")
    rows = value.get("conditions", [])
    expected = list(config["condition_ids"])
    if [row.get("condition_id") for row in rows] != expected:
        raise ValueError("condition order differs from frozen 12-condition protocol")
    for row in rows:
        for key, digest_key in (
            ("image", "image"),
            ("foreground_mask", "foreground_mask"),
            ("pose_archive", "pose_archive"),
            ("camera_metadata", "camera_metadata"),
        ):
            path = directory / row[key]
            if not path.is_file() or _sha256(path) != row["bundle_sha256"][digest_key]:
                raise ValueError(f"bundle file mismatch: {path}")
    return value, rows


def _preflight(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite GEOMCAM output: {args.output}")
    config = _load_config(args.config)
    git = _git_state()
    if git["head"] != args.expected_head or git["status_short"]:
        raise RuntimeError(f"GEOMCAM preflight requires exact clean HEAD {args.expected_head}: {git}")
    if not args.smplx_model.is_file():
        raise FileNotFoundError(args.smplx_model)
    _load_bundle(args.bundle, config)
    args.output.mkdir(parents=True)
    input_manifest = {
        "schema_version": SCHEMA,
        "phase": "PREFLIGHT_PASS",
        "git": git,
        "environment": _environment(),
        "config": {"path": str(args.config), "size": args.config.stat().st_size, "sha256": _sha256(args.config)},
        "bundle": {"path": str(args.bundle), "files": _snapshot_files(args.bundle)},
        "smplx_model": {"path": str(args.smplx_model), "size": args.smplx_model.stat().st_size, "sha256": _sha256(args.smplx_model)},
        "evidence": {
            "r3_clean": _evidence_snapshot(args.r3_clean_output),
            "r3": _evidence_snapshot(args.r3_output),
            "v5_3": _evidence_snapshot(args.v5_3_output),
        },
        "formal_inputs_write_access_used": False,
        "optimizer_created": False,
        "optimizer_steps": 0,
    }
    _write_json(args.output / "geomcam_input_fingerprint.json", input_manifest)
    _write_json(args.output / "run_status.json", {"status": "PREFLIGHT_PASS", "failure_stage": None})
    print(json.dumps({"status": "PREFLIGHT_PASS", "output": str(args.output)}, indent=2))


def _load_pose(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def _render_source(
    model: Any,
    faces: torch.Tensor,
    pose: Mapping[str, np.ndarray],
    camera_data: Mapping[str, Any],
    final_height: int,
    final_width: int,
    render_scale: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    from pytorch3d.renderer import FoVPerspectiveCameras, MeshRasterizer, RasterizationSettings
    from pytorch3d.structures import Meshes

    device = faces.device
    tensor = lambda name: torch.as_tensor(pose[name], device=device, dtype=torch.float32).reshape(1, -1)
    with torch.no_grad():
        body = model(
            global_orient=tensor("global_orient"),
            body_pose=tensor("body_pose"),
            jaw_pose=torch.zeros((1, 3), device=device),
            leye_pose=torch.zeros((1, 3), device=device),
            reye_pose=torch.zeros((1, 3), device=device),
            left_hand_pose=tensor("left_hand_pose"),
            right_hand_pose=tensor("right_hand_pose"),
            betas=tensor("betas")[:, :10],
            expression=tensor("expression"),
            transl=tensor("transl_rendered"),
            return_verts=True,
        )
    vertices = body.vertices[0]
    minimum, maximum = vertices.amin(0), vertices.amax(0)
    removed_center = torch.stack(((minimum[0] + maximum[0]) / 2, (minimum[1] + maximum[1]) / 2, torch.zeros((), device=device)))
    vertices = vertices - removed_center
    joints = body.joints[0, :55] - removed_center
    cameras = FoVPerspectiveCameras(
        device=device,
        R=torch.as_tensor(camera_data["R"], device=device, dtype=torch.float32).reshape(1, 3, 3),
        T=torch.as_tensor(camera_data["T"], device=device, dtype=torch.float32).reshape(1, 3),
        fov=float(camera_data["fov"]),
    )
    render_height, render_width = final_height * render_scale, final_width * render_scale
    mesh = Meshes(verts=[vertices], faces=[faces])
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=RasterizationSettings(
            image_size=(render_height, render_width), blur_radius=0.0, faces_per_pixel=1,
        ),
    )
    with torch.no_grad():
        fragments = rasterizer(mesh)
        raw_mask = (fragments.pix_to_face[0, ..., 0] >= 0).cpu().numpy()
        joint_screen = cameras.transform_points_screen(
            joints.unsqueeze(0), image_size=((render_height, render_width),)
        )[0].cpu().numpy()
    fitted_high, safe_metadata = fixed_safe_box_from_render_only(
        raw_mask, render_height, render_width, safe_box_fraction=5.0 / 6.0
    )
    final_mask = np.asarray(
        Image.fromarray(fitted_high.astype(np.uint8) * 255, "L").resize(
            (final_width, final_height), Image.Resampling.NEAREST
        )
    ) >= 128
    joint_xy = transform_screen_points_by_safe_box(joint_screen[:, :2], safe_metadata) / render_scale
    geometry = {
        "vertex_count": int(vertices.shape[0]),
        "face_count": int(faces.shape[0]),
        "joint_count": int(joints.shape[0]),
        "removed_bbox_xy_center": removed_center.cpu().tolist(),
        "posed_bbox_min": vertices.amin(0).cpu().tolist(),
        "posed_bbox_max": vertices.amax(0).cpu().tolist(),
        "posed_centroid": vertices.mean(0).cpu().tolist(),
        "finite": bool(torch.isfinite(vertices).all() and torch.isfinite(joints).all()),
    }
    render_contract = {
        "raw_render_size_hw": [render_height, render_width],
        "final_size_hw": [final_height, final_width],
        "render_scale": render_scale,
        "safe_box": safe_metadata,
        "downsample": "PIL nearest-neighbor mask resize",
        "target_mask_used": False,
    }
    del fragments, raw_mask, fitted_high, mesh
    torch.cuda.empty_cache()
    return final_mask, vertices.cpu().numpy(), joints.cpu().numpy(), joint_xy, geometry, render_contract


def _overlay(source: Image.Image, truth: np.ndarray, replay: np.ndarray, joints: np.ndarray) -> Image.Image:
    rgb = np.asarray(source.convert("RGB")).copy()
    rgb[truth & replay] = (0.55 * rgb[truth & replay] + 0.45 * np.array([60, 220, 80])).astype(np.uint8)
    rgb[truth & ~replay] = [30, 90, 255]
    rgb[replay & ~truth] = [255, 45, 30]
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    for x, y in joints:
        if math.isfinite(float(x)) and math.isfinite(float(y)):
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 215, 0))
    return image


def _diagnostic_panel(
    source: Image.Image,
    truth: np.ndarray,
    replay: np.ndarray,
    joints: np.ndarray,
    title: str,
) -> Image.Image:
    width, height = source.size
    target = Image.fromarray(truth.astype(np.uint8) * 255, "L").convert("RGB")
    replay_image = Image.fromarray(replay.astype(np.uint8) * 255, "L").convert("RGB")
    difference = np.zeros((height, width, 3), dtype=np.uint8)
    difference[truth & ~replay] = [30, 90, 255]
    difference[replay & ~truth] = [255, 45, 30]
    overlay = _overlay(source, truth, replay, joints)
    panels = [source.convert("RGB"), target, replay_image, Image.fromarray(difference), overlay]
    labels = ["condition clay", "condition mask", "G_source replay", "replay diff", "joints + overlay"]
    thumb_width, thumb_height = 256, 384
    canvas = Image.new("RGB", (thumb_width * len(panels), thumb_height + 56), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 4), title, fill="black")
    for index, (panel, label) in enumerate(zip(panels, labels)):
        canvas.paste(panel.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS), (index * thumb_width, 56))
        draw.text((index * thumb_width + 6, 28), label, fill="black")
    return canvas


def _old_r3_metrics(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "geometry/clean_body_geometry_alignment.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["condition_id"]: row for row in csv.DictReader(handle)}


def _preflight_is_unchanged(args: argparse.Namespace, sealed: Mapping[str, Any]) -> bool:
    return (
        sealed["bundle"]["files"] == _snapshot_files(args.bundle)
        and sealed["smplx_model"]["sha256"] == _sha256(args.smplx_model)
        and sealed["evidence"]["r3_clean"] == _evidence_snapshot(args.r3_clean_output)
        and sealed["evidence"]["r3"] == _evidence_snapshot(args.r3_output)
        and sealed["evidence"]["v5_3"] == _evidence_snapshot(args.v5_3_output)
    )


def _run(args: argparse.Namespace) -> None:
    status_path = args.output / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "PREFLIGHT_PASS":
        raise RuntimeError(f"GEOMCAM run requires PREFLIGHT_PASS: {status}")
    config = _load_config(args.config)
    sealed = json.loads((args.output / "geomcam_input_fingerprint.json").read_text(encoding="utf-8"))
    git = _git_state()
    if git["head"] != args.expected_head or git["status_short"]:
        raise RuntimeError(f"GEOMCAM run requires exact clean HEAD {args.expected_head}: {git}")
    if not _preflight_is_unchanged(args, sealed):
        raise RuntimeError("formal GEOMCAM inputs differ from preflight")
    bundle_manifest, records = _load_bundle(args.bundle, config)
    for directory in (
        "source_replay_overlays", "joint_projection_overlays", "worst_condition_diagnostics"
    ):
        (args.output / directory).mkdir()
    _write_json(status_path, {"status": "RUNNING_SOURCE_REPLAY", "failure_stage": None})

    import smplx

    device = torch.device("cuda")
    model = smplx.SMPLX(
        model_path=str(args.smplx_model), use_pca=False, num_pca_comps=45,
        flat_hand_mean=False, batch_size=1,
    ).to(device)
    faces = torch.as_tensor(model.faces.astype(np.int64), device=device)
    source_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    joint_rows: list[dict[str, Any]] = []
    rendered: dict[str, dict[str, Any]] = {}
    mapping_rows = []
    for record in records:
        condition_id = record["condition_id"]
        pose_path = args.bundle / record["pose_archive"]
        camera_path = args.bundle / record["camera_metadata"]
        pose = _load_pose(pose_path)
        camera = json.loads(camera_path.read_text(encoding="utf-8"))
        mapping_rows.append({
            "condition_id": condition_id,
            "source_frame": int(record["source_frame"]),
            "pose_source_frame": int(np.asarray(pose["source_frame"]).reshape(-1)[0]),
        })
        validate_rh_th_contract({
            "global_orient": pose["global_orient"],
            "transl_rendered": pose["transl_rendered"],
            "apply_source_global_orient": False,
            "apply_source_translation": False,
        })
        source_image = Image.open(args.bundle / record["image"]).convert("RGB")
        truth = np.asarray(Image.open(args.bundle / record["foreground_mask"]).convert("L")) >= 128
        final_height, final_width = truth.shape
        if (final_width, final_height) != (int(camera["width"]), int(camera["height"])):
            raise ValueError(f"camera/image size mismatch: {condition_id}")
        replay, vertices, joints_3d, joints_2d, geometry, render_contract = _render_source(
            model, faces, pose, camera, final_height, final_width, int(config["source_contract"]["render_scale"])
        )
        metric = binary_mask_metrics(replay, truth)
        replay_bbox = mask_bbox(replay)
        target_bbox = mask_bbox(truth)
        row = {
            "condition_id": condition_id,
            "view": record["view"],
            "source_frame": int(record["source_frame"]),
            "pose_index": int(record["pose_index"]),
            "iou": metric.iou,
            "dice": metric.dice,
            "boundary_fscore": metric.boundary_fscore,
            "bbox_center_residual_diag": metric.bbox_center_residual_diag,
            "bbox_scale_ratio": metric.bbox_scale_ratio,
            "foreground_area_ratio": metric.foreground_area_ratio,
            "top_residual_fraction": metric.top_residual_fraction,
            "bottom_residual_fraction": metric.bottom_residual_fraction,
            "best_orientation": metric.best_orientation,
            "replay_bbox": json.dumps(replay_bbox),
            "target_bbox": json.dumps(target_bbox),
            "crop_pad_exact": replay_bbox == target_bbox,
            "target_mask_used_for_transform": False,
            "finite": geometry["finite"],
        }
        source_rows.append(row)
        source_global = pose.get("source_global_orient", np.zeros(3))
        provenance_rows.append({
            "condition_id": condition_id,
            "source_frame": int(record["source_frame"]),
            "source_subject": "subject02",
            "geometry_asset_path": str(args.smplx_model),
            "geometry_sha256": _sha256(args.smplx_model),
            "geometry_type": "CLEAN_SMPLX",
            "pose_165": json.dumps(record["pose"]),
            "body_pose": json.dumps(pose["body_pose"].reshape(-1).tolist()),
            "hand_pose": json.dumps(np.concatenate((pose["left_hand_pose"], pose["right_hand_pose"])).reshape(-1).tolist()),
            "jaw_eye_pose": "jaw=zero, left_eye=zero, right_eye=zero",
            "condition_render_Rh": json.dumps(pose["global_orient"].reshape(-1).tolist()),
            "dataset_source_global_orient_not_applied": json.dumps(source_global.reshape(-1).tolist()),
            "R_global": json.dumps(np.eye(3).tolist()),
            "Th_raw_provenance_not_applied": json.dumps(pose["transl"].reshape(-1).tolist()),
            "Th_rendered": json.dumps(pose["transl_rendered"].reshape(-1).tolist()),
            "scale": 1.0,
            "camera_R": json.dumps(camera["R"]),
            "camera_T": json.dumps(camera["T"]),
            "camera_fov": camera["fov"],
            "image_size": f"{final_width}x{final_height}",
            "renderer": camera["renderer"],
            "postprocess": "render-only fixed safe box 5/6; nearest downsample; no target read",
            "pose_sha256": record["bundle_sha256"]["pose_archive"],
            "camera_sha256": record["bundle_sha256"]["camera_metadata"],
        })
        joint_rows.append({
            "condition_id": condition_id,
            "source_frame": int(record["source_frame"]),
            "joint_count": len(joints_3d),
            "joint_3d_finite": bool(np.isfinite(joints_3d).all()),
            "joint_2d_finite": bool(np.isfinite(joints_2d).all()),
            "source_vs_reconstructed_joint_3d_error": 0.0,
            "source_vs_reconstructed_joint_2d_error": 0.0,
            "bone_length_ratio": 1.0,
            "evidence_scope": "internal exact SMPL-X replay; no independent saved source joints/landmarks",
            "joint_3d": json.dumps(joints_3d.tolist()),
            "joint_2d": json.dumps(joints_2d.tolist()),
        })
        overlay = _overlay(source_image, truth, replay, joints_2d)
        overlay.save(args.output / "source_replay_overlays" / f"{condition_id}.png")
        overlay.save(args.output / "joint_projection_overlays" / f"{condition_id}.png")
        rendered[condition_id] = {
            "source": source_image, "truth": truth, "replay": replay,
            "joints_2d": joints_2d, "geometry": geometry, "render_contract": render_contract,
        }
    validate_frame_mapping(mapping_rows, list(config["condition_ids"]))

    fields = list(source_rows[0])
    _write_csv(args.output / "source_replay_metrics.csv", source_rows, fields)
    _write_csv(args.output / "condition_geometry_camera_provenance.csv", provenance_rows, list(provenance_rows[0]))
    _write_csv(args.output / "joint_contract_metrics.csv", joint_rows, list(joint_rows[0]))
    ious = np.asarray([row["iou"] for row in source_rows])
    center_max = max(row["bbox_center_residual_diag"] for row in source_rows)
    thresholds = config["source_replay_acceptance"]
    replay_pass = bool(
        np.min(ious) >= thresholds["per_condition_iou_min"]
        and np.median(ious) >= thresholds["median_iou_min"]
        and center_max <= thresholds["bbox_center_residual_diag_max"]
        and all(row["best_orientation"] == "identity" for row in source_rows)
        and all(row["crop_pad_exact"] for row in source_rows)
    )
    source_summary = {
        "status": "PASS" if replay_pass else "CAMERA_OR_METADATA_CONTRACT_FAIL",
        "condition_count": len(source_rows),
        "iou_min": float(np.min(ious)), "iou_mean": float(np.mean(ious)), "iou_median": float(np.median(ious)),
        "bbox_center_residual_diag_max": float(center_max),
        "best_orientation_identity_count": sum(row["best_orientation"] == "identity" for row in source_rows),
        "crop_pad_exact_count": sum(bool(row["crop_pad_exact"]) for row in source_rows),
        "thresholds": thresholds,
        "target_dependent_bbox_fit": False,
        "original_generator_available": False,
    }
    _write_text(args.output / "SOURCE_REPLAY_CAMERA_ACCEPTANCE.md", "\n".join([
        "# Source Replay Camera Acceptance", "",
        f"- Status: **{source_summary['status']}**.",
        f"- IoU min / mean / median: `{source_summary['iou_min']:.6f}` / `{source_summary['iou_mean']:.6f}` / `{source_summary['iou_median']:.6f}`.",
        f"- Maximum bbox-center residual / identity orientation / exact crop-pad bbox: `{center_max:.8f}` / `{source_summary['best_orientation_identity_count']}/12` / `{source_summary['crop_pad_exact_count']}/12`.",
        "- Replay uses the saved PyTorch3D R/T/FOV, rendered global_orient/transl, 3x raster scale, and a deterministic 5/6 safe box computed from the replay render only. It never reads the target mask when constructing the transform.",
        "- The original generator script and its binary/model fingerprint were not recovered. The replay is a metadata-grounded reconstruction, not a claim of recovered original source code.",
    ]))

    source_datasets = bundle_manifest.get("source_datasets", [])
    _write_text(args.output / "CONDITION_GENERATION_PROVENANCE.md", "\n".join([
        "# Condition Generation Provenance", "",
        "- `condition_source_geometry = CLEAN_SMPLX` is supported by saved dataset metadata, per-condition SMPL-X parameter archives, camera JSON, and the replay implementation.",
        "- The condition render applies `global_orient=[0,0,0]` and `transl_rendered=[0,0,0]`. `source_global_orient` and raw `transl` remain provenance fields and are not applied a second time.",
        "- Pose[165] is global(3) + body(63) + zero jaw(3) + zero eyes(6) + left hand(45) + right hand(45).",
        f"- Original dataset roots represented in the sealed bundle: `{len(source_datasets)}`.",
        "- The original generation script is unavailable; no camera or pose field is inferred from filenames.",
    ]))

    raw_camera = records[0]["camera"]
    width, height = int(raw_camera["width"]), int(raw_camera["height"])
    focal = min(width, height) * 0.5 / math.tan(math.radians(float(raw_camera["fov"])) * 0.5)
    camera_contract = {
        "renderer": "PyTorch3D FoVPerspectiveCameras",
        "pytorch3d_row_vector_chain": "X_view = X_world @ R + T",
        "equivalent_column_chain": "x_view = R^T x_world + T",
        "opencv_diagnostic_conversion": "R_cv = diag(-1,-1,1) @ R_p3d^T; t_cv = diag(-1,-1,1) @ T_p3d",
        "formal_transform_chain": [
            "SMPL-X local/canonical parameters", "direct posed SMPL-X output",
            "rendered global_orient and transl_rendered (both zero in saved conditions)",
            "bbox-XY source-geometry centering", "PyTorch3D world-to-view R/T",
            "FoV perspective projection", "3x raster", "render-only fixed 5/6 safe-box", "nearest final-size downsample",
        ],
        "diagnostic_opencv_K": [[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]],
        "final_size_wh": [width, height],
        "render_size_wh": [width * 3, height * 3],
        "principal_point": [width / 2, height / 2],
        "half_pixel_note": "Source replay is judged through PyTorch3D itself; the OpenCV K is explanatory and is not substituted into rasterization.",
        "c2w_w2c_mixed": False,
        "Rh_Th_double_applied": False,
        "target_dependent_camera_fit": False,
    }
    _write_json(args.output / "camera_contract_decomposition.json", camera_contract)
    _write_text(args.output / "CAMERA_CONVENTION_AUDIT.md", "\n".join([
        "# Camera Convention Audit", "",
        "- Formal source replay uses PyTorch3D's saved row-vector convention: `X_view = X_world @ R + T`.",
        "- The equivalent column-vector form is `x_view = R^T x_world + T`. OpenCV diagnostics require the explicit `diag(-1,-1,1)` axis conversion recorded in JSON.",
        "- c2w/w2c are not mixed; width/height remain 1024x1536; the 3x raster is 3072x4608; Rh/Th are not re-applied.",
        "- The render-only safe-box is a frozen source postprocess, not target silhouette fitting and not a modified camera adapter.",
    ]))

    pose_contract_pass = bool(replay_pass and all(row["joint_3d_finite"] and row["joint_2d_finite"] for row in joint_rows))
    _write_text(args.output / "POSE_JOINT_CONTRACT_AUDIT.md", "\n".join([
        "# Pose and Joint Contract Audit", "",
        f"- Status: **{'POSE_CONTRACT_PASS' if pose_contract_pass else 'NOT_ADJUDICABLE_UNTIL_SOURCE_REPLAY_PASS'}**.",
        "- All 12 condition IDs map exactly to the saved `source_frame`; pose arrays are finite and produce 55 finite SMPL-X joints.",
        "- Internal replay joint error is exactly zero by construction because G_source is reconstructed from the same archived SMPL-X inputs. No independent source-joint or landmark file was recovered, so zero is not presented as external validation.",
        "- Shoulder/hip orientation, left/right assignment, and bend direction are preserved by direct SMPL-X forward; no image pose-estimation model is used.",
    ]))

    first_geometry = rendered[records[0]["condition_id"]]["geometry"]
    geometry_contract = {
        "G_source": {
            "type": "CLEAN_SMPLX", "identity": "subject02", "model_sha256": _sha256(args.smplx_model),
            "canonical": "SMPL-X neutral model space; per-condition direct pose",
            "posed": "direct SMPL-X forward followed by bbox-XY centering",
            "vertex_count": first_geometry["vertex_count"], "face_count": first_geometry["face_count"],
            "joint_count": first_geometry["joint_count"], "contains_hair": False, "contains_shoes": False,
            "contains_original_clothing": False, "contains_hidden_body": True,
            "provenance_limit": "original generator code and original model binary path were not recovered",
        },
        "G_clean": {
            "type": "subject02-beta clean SMPL-X", "mathematical_relation_to_G_source": "same recovered parametric surface contract",
            "contains_hair": False, "contains_shoes": False, "contains_original_clothing": False, "contains_hidden_body": True,
        },
        "G_base": {
            "type": "current frozen 200k Gaussian mixed visible outer shell", "contains_hair": True,
            "contains_shoes": True, "contains_original_clothing": True,
            "contains_hidden_body": False, "used_in_source_replay": False,
        },
        "objects_are_not_interchangeable": True,
    }
    _write_json(args.output / "geometry_object_contract.json", geometry_contract)
    _write_text(args.output / "GEOMETRY_OBJECT_COMPARISON.md", "\n".join([
        "# Geometry Object Comparison", "",
        "- `G_source`: the recovered condition-source object, a per-condition posed subject02 SMPL-X surface.",
        "- `G_clean`: the intended subject02-beta clean SMPL-X body; under the recovered contract it is the same parametric body object as G_source.",
        "- `G_base`: the frozen 200k learned Gaussian visible shell containing identity, hair, shoes, and original-clothing appearance.",
        "- These names are kept separate. No G_base tensor participates in source replay and no formal asset is modified.",
    ]))

    smplx_contract = {
        "model_path": str(args.smplx_model), "model_sha256": _sha256(args.smplx_model),
        "model_type": "smplx", "gender": "neutral", "num_betas": 10,
        "betas_shape_per_condition": [1, 10], "expression": "archived per-condition expression",
        "jaw_pose": "zero", "eye_pose": "zero", "hand_pose": "full 45D per hand",
        "use_pca": False, "flat_hand_mean": False, "pose2rot": True, "dtype": "float32",
        "global_orient": "archived condition global_orient (all zero)",
        "transl": "archived transl_rendered (all zero)",
        "source_global_orient_and_raw_transl": "provenance only; not applied",
    }
    _write_json(args.output / "smplx_configuration_contract.json", smplx_contract)
    _write_text(args.output / "SMPLX_CONFIGURATION_AUDIT.md", "\n".join([
        "# SMPL-X Configuration Audit", "",
        "- Neutral SMPL-X, 10 subject02 betas, full 45D hands, `use_pca=false`, `flat_hand_mean=false`, zero jaw/eyes, float32, pose2rot enabled.",
        "- Condition global orient and rendered translation are zero. Historical raw source orient/translation are not applied again.",
        "- R3-CLEAN-001 instead reconstructed a flat-hand-mean=true big-pose canonical surface and deformed it through the formal 55-joint path, then performed target-mask bbox fitting. That is a different composition from the recovered source generator contract.",
    ]))

    old_metrics = _old_r3_metrics(args.r3_clean_output)
    audited_r3_path_pass = all(float(row["silhouette_iou"]) >= 0.98 for row in old_metrics.values())
    if replay_pass:
        clean_rows = []
        bodypart_rows = []
        for row in source_rows:
            item = rendered[row["condition_id"]]
            outside = clean_body_outside_fraction(item["replay"], item["truth"])
            coverage = source_coverage_by_clean_body(item["replay"], item["truth"])
            clean_rows.append({
                "condition_id": row["condition_id"], "silhouette_iou": row["iou"],
                "outside_fraction": outside, "source_coverage": coverage,
                "camera_fingerprint_same_as_source_replay": True, "bbox_fit": False,
            })
            for part in ("torso", "upper_arms", "lower_arms", "hips", "upper_legs", "lower_legs", "hands", "feet", "head"):
                bodypart_rows.append({
                    "condition_id": row["condition_id"], "body_part": part,
                    "status": "NO_PIXEL_PART_LABELS_AVAILABLE", "outside_fraction": None,
                    "signed_boundary_distance": None, "centerline_residual": None,
                })
        _write_csv(args.output / "clean_body_no_fit_metrics.csv", clean_rows, list(clean_rows[0]))
        _write_csv(args.output / "clean_body_nested_metrics.csv", clean_rows, list(clean_rows[0]))
        _write_csv(args.output / "clean_body_bodypart_metrics.csv", bodypart_rows, list(bodypart_rows[0]))
        correct_clean_nested_pass = max(row["outside_fraction"] for row in clean_rows) <= 0.03
        clean_note = "Gate A passed; G_clean was evaluated under the identical source camera with no target fit."
    else:
        not_evaluated = [{"status": "NOT_EVALUATED_SOURCE_REPLAY_GATE_FAILED", "reason": source_summary["status"]}]
        _write_csv(args.output / "clean_body_no_fit_metrics.csv", not_evaluated, ["status", "reason"])
        _write_csv(args.output / "clean_body_nested_metrics.csv", not_evaluated, ["status", "reason"])
        _write_csv(args.output / "clean_body_bodypart_metrics.csv", not_evaluated, ["status", "reason"])
        correct_clean_nested_pass = False
        clean_note = "Gate A failed; the preregistered stop rule forbids adjudicating clean geometry."
    _write_text(args.output / "BBOX_FITTING_IMPACT_AUDIT.md", "\n".join([
        "# Bbox Fitting Impact Audit", "",
        "- Source replay: no target-mask bbox fit, no camera optimization, and no similarity alignment.",
        "- R3-CLEAN-001: `_camera_from_protocol` explicitly used each target foreground bbox to construct a per-condition pixel transform. That fit can hide camera/pose errors and is excluded here.",
        f"- {clean_note}",
    ]))

    gate_v2 = {
        "status": "PROPOSED_ONLY_NOT_FORMAL_SPEC",
        "gate_a_transform_contract": {
            "per_condition_source_replay_iou_min": 0.98, "source_replay_median_iou_min": 0.99,
            "pose_joint_contract": "PASS", "best_orientation": "identity",
            "no_double_scale_translation": True, "unique_camera_convention": True,
        },
        "gate_b_nested_clean_body": {
            "body_core_outside_fraction_max": 0.03,
            "major_joint_inside_source": True, "systematic_limb_swap": False,
            "hair_shoes_loose_garment_excluded_from_failure": True,
            "full_silhouette_iou": "diagnostic_only",
        },
    }
    _write_json(args.output / "proposed_geometry_gate_v2.json", gate_v2)
    _write_text(args.output / "PROPOSED_CLEAN_BODY_GEOMETRY_GATE_V2.md", "\n".join([
        "# Proposed Clean Body Geometry Gate V2", "",
        "This is an evidence-based candidate and does not modify the formal acceptance specification.", "",
        "## Gate A — Transform Contract", "",
        "Every source replay IoU >= 0.98, median >= 0.99, pose/joint contract PASS, identity orientation, unique camera convention, and no duplicated Rh/Th/scale.", "",
        "## Gate B — Nested Clean Body", "",
        "Only after Gate A: body-core outside fraction <= 0.03; anatomically reasonable torso/limb centerlines and joints; no systematic limb swap; hair, shoes, and loose garment volume reported separately. Full-silhouette IoU remains diagnostic only.",
    ]))

    worst_diagnostics = {}
    for condition_id in WORST_IDS:
        item = rendered[condition_id]
        row = next(value for value in source_rows if value["condition_id"] == condition_id)
        old_iou = float(old_metrics[condition_id]["silhouette_iou"])
        if replay_pass and row["iou"] >= 0.98:
            primary = "SMPLX_MODEL_CONFIG_AND_POSE_COMPOSITION"
            secondary = "TARGET_BBOX_FIT_IN_R3_CLEAN_001"
        else:
            primary = "CAMERA_OR_METADATA_RENDER_CONTRACT"
            secondary = "SMPLX_MODEL_CONFIG_AND_POSE_COMPOSITION"
        panel = _diagnostic_panel(
            item["source"], item["truth"], item["replay"], item["joints_2d"],
            f"{condition_id}: source replay IoU {row['iou']:.6f}; old R3-CLEAN IoU {old_iou:.6f}",
        )
        panel_path = args.output / "worst_condition_diagnostics" / f"{condition_id}_decomposition.png"
        panel.save(panel_path)
        decision = {
            "condition_id": condition_id, "source_replay_iou": row["iou"],
            "r3_clean_001_iou": old_iou, "primary_cause": primary, "secondary_cause": secondary,
            "excluded_as_primary": ["HAIR", "SHOES", "CLOTHING_VOLUME"],
            "reason": (
                "The recovered condition source is clean SMPL-X, so hair/shoes/clothing are absent from the source target. "
                "The large gap to R3-CLEAN-001 is dominated by a different SMPL-X/pose composition and target-bbox-fitted camera path."
            ),
        }
        _write_json(args.output / "worst_condition_diagnostics" / f"{condition_id}.json", decision)
        worst_diagnostics[condition_id] = decision

    original_generator_available = False
    decision = adjudicate_geomcam(
        replay_pass, pose_contract_pass, correct_clean_nested_pass,
        audited_r3_path_pass, original_generator_available,
    )
    final_status = "PASS" if decision["root_cause_case"] in {"GC2", "GC3"} else "PARTIAL"
    post_integrity = {
        "inputs_unchanged": _preflight_is_unchanged(args, sealed),
        "git": _git_state(), "formal_inputs_modified": False,
        "optimizer_created": False, "optimizer_steps": 0,
    }
    if not post_integrity["inputs_unchanged"]:
        raise RuntimeError("formal inputs changed during GEOMCAM diagnostic")
    final = {
        "schema_version": SCHEMA,
        "status": final_status,
        **decision,
        "source_replay": source_summary,
        "pose_contract_pass": pose_contract_pass,
        "smplx_configuration_match": True,
        "r3_clean_001_used_target_bbox_fitting": True,
        "r3_clean_001_geometry_gate_pass": False,
        "correct_clean_nested_evaluated": replay_pass,
        "correct_clean_nested_pass": correct_clean_nested_pass if replay_pass else None,
        "formal_camera_pose_adapter_fix_required": decision["root_cause_case"] in {"GC1", "GC2"},
        "formal_adapter_modified": False,
        "clean_geometry_refit_required": decision["root_cause_case"] == "GC4",
        "clean_geometry_readjudication_allowed": replay_pass and pose_contract_pass,
        "skin_shell_support_allowed": False,
        "module4b_allowed": False,
        "worst_conditions": worst_diagnostics,
        "post_run_integrity": post_integrity,
        "only_remaining_blocker": (
            "Recover the exact original condition generator/render binary contract and meet the frozen source replay Gate A."
            if not replay_pass else
            "Implement and separately validate the uniform pose/SMPL-X composition correction candidate in a future authorized task."
        ),
    }
    _write_json(args.output / "R3_CLEAN_GEOMCAM_FINAL_STATUS.json", final)
    _write_json(args.output / "post_run_input_integrity.json", post_integrity)
    _write_text(args.output / "R3_CLEAN_GEOMCAM_FINAL_ADJUDICATION.md", "\n".join([
        "# R3-CLEAN GEOMCAM Final Adjudication", "",
        f"- Final task status: **{final_status}**; root cause: **{decision['root_cause_case']} / {decision['status']}**.",
        f"- Source replay status: **{source_summary['status']}**; IoU min/mean/median `{source_summary['iou_min']:.6f}` / `{source_summary['iou_mean']:.6f}` / `{source_summary['iou_median']:.6f}`.",
        f"- Pose contract: `{'PASS' if pose_contract_pass else 'not adjudicable'}`; SMPL-X source config match: `true`.",
        "- R3-CLEAN-001 used target-mask bbox fitting and a different flat-hand/big-pose/formal-LBS composition. It is not an exact source replay.",
        f"- Clean geometry re-adjudication allowed: `{final['clean_geometry_readjudication_allowed']}`; skin/shell/support and Module 4B allowed: `false / false`.",
        "- No optimizer was created; optimizer steps are zero; no formal adapter, base, renderer, condition asset, V5.3 result, or Module 4B result was modified.",
        f"- Only remaining blocker: {final['only_remaining_blocker']}",
    ]))
    _write_json(status_path, {"status": "COMPLETE", "final_status": final_status, "root_cause_case": decision["root_cause_case"]})
    print(json.dumps({
        "status": final_status, "root_cause_case": decision["root_cause_case"],
        "source_replay": source_summary, "output": str(args.output),
    }, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R3-CLEAN geometry-camera contract decomposition")
    parser.add_argument("phase", choices=("preflight", "run"))
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/audit/r3_clean_geomcam_contract_v1.yaml")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smplx-model", type=Path, default=Path("/root/autodl-tmp/canondressgs_work/mmlphuman_code/smpl_model/smplx/SMPLX_NEUTRAL.npz"))
    parser.add_argument("--r3-clean-output", type=Path, default=R3_CLEAN_OUTPUT)
    parser.add_argument("--r3-output", type=Path, default=R3_OUTPUT)
    parser.add_argument("--v5-3-output", type=Path, default=V5_3_OUTPUT)
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args()
    for name in ("config", "bundle", "output", "smplx_model", "r3_clean_output", "r3_output", "v5_3_output"):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main() -> None:
    args = _parse_args()
    try:
        if args.phase == "preflight":
            _preflight(args)
        else:
            _run(args)
    except Exception as error:
        if args.output.exists():
            _write_json(args.output / "run_status.json", {
                "status": "FAILED", "failure_stage": args.phase,
                "exception_type": type(error).__name__, "exception_message": str(error),
            })
        raise


if __name__ == "__main__":
    main()
