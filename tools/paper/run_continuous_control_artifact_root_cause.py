"""Inference-only attribution of continuous-control artifacts.

This executor is append-only beneath a dedicated research-diagnostic output
root.  It consumes the sealed P0 archive and frozen paper assets, never trains,
never writes a checkpoint, and never regenerates the existing FULL grid.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage, stats


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene.gaussian_clothing_residuals import CHANNELS, GaussianClothingResiduals
from tools import diagnose_image_conditioned_overfit_failure as diagnosis
from tools.paper import run_p0_color_spatial_soft_control_evaluations as sealed
from tools.paper.verify_seen_outfit_paper_assets import verify_manifest


TASK_ID = "AAAI27-CONTINUOUS-CONTROL-ROOT-CAUSE-001"
SOURCE_HEAD = "371d812864614cd561e33edfe3f6c043b38415d2"
RUN_BRANCH = "research/continuous-control-artifact-root-cause-20260722"
PROTOCOL_PATH = PROJECT_ROOT / "paper_protocol/reviewer_risk/continuous_control_root_cause_protocol.yaml"
PROTOCOL_SHA256 = "4d5abd9c368b40e9e622573b43ceb9ba15cf2774a0907be5d798be2fc41db98a"
SEALED_SUMMARY = PROJECT_ROOT / "paper_protocol/reviewer_risk/p0_color_spatial_soft_control_final_summary.json"
SEALED_REVIEW = PROJECT_ROOT / "paper_protocol/reviewer_risk/p0_color_spatial_soft_control_visual_review.json"
SEALED_INTERPOLATION = PROJECT_ROOT / "paper_protocol/reviewer_risk/p0_basis_interpolation_results.json"
SEALED_MIXED = PROJECT_ROOT / "paper_protocol/reviewer_risk/p0_mixed_reference_results.json"
FROZEN_MANIFEST = PROJECT_ROOT / "paper_protocol/frozen_asset_manifest.json"
OUTFITS = ("O01", "O02", "O03", "O04", "O08")
PAIRS = tuple(itertools.combinations(OUTFITS, 2))
CONDITIONS = ("cond_000000", "cond_000318", "cond_000017", "cond_000347")
ALPHAS = (0.25, 0.50, 0.75)
VARIANTS = {
    "XYZ_ONLY": ("delta_xyz",),
    "SCALE_ROT_ONLY": ("delta_log_scaling", "delta_rotvec"),
    "OPACITY_ONLY": ("delta_opacity_logit",),
    "SH_ONLY": ("delta_sh0", "delta_shN"),
    "GEOMETRY_ALL": ("delta_xyz", "delta_log_scaling", "delta_rotvec"),
    "APPEARANCE_ALL": ("delta_opacity_logit", "delta_sh0", "delta_shN"),
}
STABLE = {"O01_O02", "O01_O04", "O03_O04"}
BOUNDS = {
    "delta_xyz": 0.05,
    "delta_log_scaling": 0.35,
    "delta_rotvec": 0.2617993878,
    "delta_opacity_logit": 2.0,
    "delta_sh0": 0.25,
    "delta_shN": 0.10,
}
CORE_GRADES = ("cloud", "mottle", "edge_scatter", "silhouette_discontinuity")
ALL_GRADES = CORE_GRADES + ("full_body_contamination", "identity_contamination")


def sha256(path: Path, *, lf: bool = False) -> str:
    data = path.read_bytes()
    if lf:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any, *, replace: bool = False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(f"append-only artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=PROJECT_ROOT, text=True).strip()


def protocol() -> dict[str, Any]:
    if sha256(PROTOCOL_PATH, lf=True) != PROTOCOL_SHA256:
        raise RuntimeError("ROOT-CAUSE-ASSET-MISMATCH: protocol fingerprint")
    value = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    design = value["design"]
    pairs = tuple(tuple(item) for item in design["unordered_pair_order"])
    labels = design["stable_pair_labels"]
    if pairs != PAIRS or tuple(design["target_view_order"]) != CONDITIONS:
        raise RuntimeError("ROOT-CAUSE-ASSET-MISMATCH: frozen pair/view order")
    if tuple(float(item) for item in design["alpha"]) != ALPHAS:
        raise RuntimeError("ROOT-CAUSE-ASSET-MISMATCH: alpha grid")
    if {key for key, label in labels.items() if label == "stable"} != STABLE:
        raise RuntimeError("ROOT-CAUSE-ASSET-MISMATCH: stable labels")
    if Counter(labels.values()) != Counter({"stable": 3, "unstable": 7}):
        raise RuntimeError("ROOT-CAUSE-ASSET-MISMATCH: 3/7 labels")
    if value["residual_contract"]["variants"] != {key: list(channels) for key, channels in VARIANTS.items()}:
        raise RuntimeError("ROOT-CAUSE-ASSET-MISMATCH: variants")
    return value


def tree_manifest(path: Path) -> dict[str, Any]:
    return sealed.tree_manifest(path)


def attempt_path(output_root: Path, name: str) -> Path:
    if not name.startswith("attempt_") or not name[8:].isdigit():
        raise ValueError("attempt must use attempt_NNN")
    return output_root / name


def full_index() -> tuple[dict[tuple[str, str, float], dict[str, Any]], dict[str, Any]]:
    result = read_json(SEALED_INTERPOLATION)
    if result["render_count"] != 440 or result["expected_render_count"] != 440:
        raise RuntimeError("ROOT-CAUSE-ASSET-MISMATCH: FULL count")
    if not result["endpoint_parity_pass"] or len(result["records"]) != 440:
        raise RuntimeError("ROOT-CAUSE-ASSET-MISMATCH: FULL endpoint parity")
    index = {(row["pair_id"], row["view_id"], float(row["alpha"])): row for row in result["records"]}
    if len(index) != 440:
        raise RuntimeError("ROOT-CAUSE-ASSET-MISMATCH: duplicate FULL records")
    missing = [
        value
        for row in result["records"]
        for value in (Path(row["rgb_path"]), Path(row["alpha_path"]))
        if not value.is_file()
    ]
    if missing:
        raise RuntimeError(f"ROOT-CAUSE-ASSET-MISMATCH: missing FULL files {missing[:3]}")
    return index, result


def validate_source_and_archive() -> dict[str, Any]:
    if git("branch", "--show-current") != RUN_BRANCH:
        raise RuntimeError("ROOT-CAUSE-ASSET-MISMATCH: branch")
    if git("status", "--short"):
        raise RuntimeError("root-cause executor requires a clean worktree")
    if subprocess.call(["git", "merge-base", "--is-ancestor", SOURCE_HEAD, "HEAD"], cwd=PROJECT_ROOT):
        raise RuntimeError("ROOT-CAUSE-ASSET-MISMATCH: source ancestry")
    expected = protocol()["archive_gate"]
    for key, path in (
        ("final_summary", SEALED_SUMMARY),
        ("visual_review", SEALED_REVIEW),
        ("interpolation_results", SEALED_INTERPOLATION),
    ):
        if sha256(path) != expected[key]["sha256"]:
            raise RuntimeError(f"ROOT-CAUSE-ASSET-MISMATCH: {key} fingerprint")
    summary = read_json(SEALED_SUMMARY)
    review = read_json(SEALED_REVIEW)
    mixed = read_json(SEALED_MIXED)
    if summary["paper_final_count"] != 0 or summary["counts"]["interpolation_renders"] != 440:
        raise RuntimeError("ROOT-CAUSE-ASSET-MISMATCH: sealed summary")
    if sum(bool(item["actual_opened"]) for item in review["items"]) != 57:
        raise RuntimeError("ROOT-CAUSE-ASSET-MISMATCH: sealed visual review")
    stable = {key for key, value in mixed["stable_pairs"].items() if value["stable_non_endpoint_pair"]}
    if mixed["stable_non_endpoint_pair_count"] != 3 or stable != STABLE:
        raise RuntimeError("ROOT-CAUSE-ASSET-MISMATCH: mixed-reference labels")
    full, interpolation = full_index()
    return {
        "summary": summary,
        "review": review,
        "mixed": mixed,
        "full": full,
        "interpolation": interpolation,
    }


def run_preflight(attempt: Path, asset_root: Path) -> dict[str, Any]:
    result_path = attempt / "audits/preflight.json"
    if result_path.is_file():
        return read_json(result_path)
    sealed_data = validate_source_and_archive()
    manifest = read_json(FROZEN_MANIFEST)
    asset_verification = verify_manifest(manifest, PROJECT_ROOT, asset_root, verify_external=True)
    if asset_verification["status"] != "PASS" or asset_verification["asset_count"] != 19:
        raise RuntimeError("ROOT-CAUSE-ASSET-MISMATCH: frozen manifest")
    smoke = torch.tensor([6.0, 7.0], device="cuda").sum()
    torch.cuda.synchronize()
    if float(smoke) != 13.0:
        raise RuntimeError("ROOT-CAUSE-ASSET-MISMATCH: CUDA smoke")
    attempt.mkdir(parents=True, exist_ok=False)
    for name in ("channel_interpolation", "support_conflict", "role_conflict", "correlations", "visuals", "audits", "aggregates"):
        (attempt / name).mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": "canondressgs.research.continuous_control_root_cause_preflight.v1",
        "status": "PASS",
        "task_id": TASK_ID,
        "label": "RESEARCH DIAGNOSTIC — NOT PAPER FINAL",
        "run_branch": RUN_BRANCH,
        "execution_head": git("rev-parse", "HEAD"),
        "source_head": SOURCE_HEAD,
        "protocol_sha256_lf": PROTOCOL_SHA256,
        "archive": {
            "sealed": True,
            "final_summary_sha256": sha256(SEALED_SUMMARY),
            "visual_review_sha256": sha256(SEALED_REVIEW),
            "interpolation_sha256": sha256(SEALED_INTERPOLATION),
            "visual_review_count": 57,
            "full_render_count": len(sealed_data["full"]),
            "pair_count": len(PAIRS),
            "view_count": len(CONDITIONS),
            "stable_count": 3,
            "unstable_count": 7,
        },
        "frozen_asset_verification": asset_verification,
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "cuda_tensor_smoke": float(smoke),
            "cuda": torch.version.cuda,
            "pytorch": torch.__version__,
            "python": platform.python_version(),
        },
        "frozen_trees_before": {
            "formal": tree_manifest(asset_root / sealed.FORMAL_NAME),
            "p0": tree_manifest(asset_root / sealed.P0_NAME),
            "sealed_evaluation": tree_manifest(asset_root / sealed.OUTPUT_NAME),
        },
        "counts": {
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_created": 0,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "checkpoint_writes": 0,
            "paper_final": 0,
        },
        "paper_final": False,
    }
    atomic_json(result_path, result)
    return result


def runtime(attempt: Path, asset_root: Path, frozen_protocol: Mapping[str, Any]) -> sealed.EvaluationRuntime:
    os.environ["CANONDRESSGS_ASSET_ROOT"] = str(asset_root)
    sealed.RUN_BRANCH = RUN_BRANCH
    sealed.SOURCE_HEAD = SOURCE_HEAD
    torch.set_grad_enabled(False)
    value = sealed.EvaluationRuntime(attempt, asset_root, frozen_protocol)
    if torch.is_grad_enabled():
        raise RuntimeError("ROOT-CAUSE-NO-TRAINING-GATE-VIOLATION: grad enabled")
    return value


def combine_residuals(left: Any, right: Any, selected: Sequence[str], alpha: float) -> GaussianClothingResiduals:
    selected_set = set(selected)
    values: dict[str, torch.Tensor] = {}
    for name in CHANNELS:
        a = getattr(left, name)
        b = getattr(right, name)
        weight = alpha if name in selected_set else 0.5
        values[name] = (1.0 - weight) * a + weight * b
    return GaussianClothingResiduals.from_dict(values)


def alpha_id(alpha: float) -> str:
    return f"a{int(round(alpha * 1000)):03d}"


def channel_paths(attempt: Path, variant: str, pair_id: str, condition: str, alpha: float) -> tuple[Path, Path]:
    root = attempt / "channel_interpolation" / variant / pair_id / condition
    return root / f"{alpha_id(alpha)}_rgb.png", root / f"{alpha_id(alpha)}_alpha.png"


def masked_mae(first: torch.Tensor, second: torch.Tensor, mask: torch.Tensor) -> float:
    denominator = mask.sum().clamp_min(1.0) * first.shape[0]
    return float(((first - second).abs() * mask).sum() / denominator)


def channel_expected(attempt: Path) -> list[dict[str, Any]]:
    return [
        {
            "variant": variant,
            "pair": [left, right],
            "pair_id": f"{left}_{right}",
            "view_id": condition,
            "alpha": alpha,
            "rgb_path": str(channel_paths(attempt, variant, f"{left}_{right}", condition, alpha)[0]),
            "alpha_path": str(channel_paths(attempt, variant, f"{left}_{right}", condition, alpha)[1]),
        }
        for variant in VARIANTS
        for left, right in PAIRS
        for condition in CONDITIONS
        for alpha in ALPHAS
    ]


def run_channel_render(runtime_value: sealed.EvaluationRuntime, full: Mapping[tuple[str, str, float], Mapping[str, Any]]) -> dict[str, Any]:
    attempt = runtime_value.attempt
    result_path = attempt / "aggregates/channel_interpolation_results.json"
    if result_path.is_file():
        return read_json(result_path)
    expected = channel_expected(attempt)
    existing_at_start = 0
    for row in expected:
        rgb_exists = Path(row["rgb_path"]).is_file()
        alpha_exists = Path(row["alpha_path"]).is_file()
        if rgb_exists != alpha_exists:
            raise RuntimeError(f"ROOT-CAUSE-ASSET-MISMATCH: partial channel item {row}")
        existing_at_start += int(rgb_exists and alpha_exists)
    atomic_json(
        attempt / "audits/channel_resume_manifest.json",
        {
            "status": "RUNNING",
            "expected": 720,
            "completed_before_resume": existing_at_start,
            "missing_before_resume": 720 - existing_at_start,
            "completed_items_repeated": 0,
        },
        replace=True,
    )
    endpoint_residuals = {
        outfit: runtime_value.basis(runtime_value.coefficients[outfit], chunk_size=16384)
        for outfit in OUTFITS
    }
    records: list[dict[str, Any]] = []
    created = 0
    sheet_paths: list[str] = []
    for variant, channels in VARIANTS.items():
        for left, right in PAIRS:
            pair_id = f"{left}_{right}"
            sheet_rows: list[tuple[str, list[tuple[str, Path]]]] = []
            for condition in CONDITIONS:
                garment = diagnosis._garment_mask(runtime_value.context["samples"][f"{left}/{condition}"])
                protected = runtime_value.context["samples"][f"{left}/{condition}"]["target_protected_mask"]
                endpoint_a = sealed.image_tensor(Path(full[(pair_id, condition, 0.0)]["rgb_path"]), 3).to(runtime_value.device)
                endpoint_b = sealed.image_tensor(Path(full[(pair_id, condition, 1.0)]["rgb_path"]), 3).to(runtime_value.device)
                panels: list[tuple[str, Path]] = []
                for alpha in ALPHAS:
                    rgb_path, rendered_alpha_path = channel_paths(attempt, variant, pair_id, condition, alpha)
                    existed = rgb_path.is_file() and rendered_alpha_path.is_file()
                    residual = combine_residuals(endpoint_residuals[left], endpoint_residuals[right], channels, alpha)
                    rgb, rendered_alpha = sealed.render_or_load(
                        runtime_value, left, condition, residual, rgb_path, rendered_alpha_path
                    )
                    created += int(not existed)
                    reference = (1.0 - alpha) * endpoint_a + alpha * endpoint_b
                    silhouette_iou, boundary_fscore, tolerance = sealed.silhouette_metrics(rendered_alpha, garment)
                    records.append({
                        "variant": variant,
                        "selected_channels": list(channels),
                        "pair": [left, right],
                        "pair_id": pair_id,
                        "view_id": condition,
                        "alpha": alpha,
                        "rgb_path": str(rgb_path),
                        "alpha_path": str(rendered_alpha_path),
                        "garment_rgb_mae": masked_mae(rgb, reference, garment),
                        "lpips": sealed.lpips_distance(runtime_value, rgb, reference, garment),
                        "silhouette_iou": silhouette_iou,
                        "boundary_fscore": boundary_fscore,
                        "boundary_tolerance": tolerance,
                        "protected_lpips": sealed.lpips_distance(runtime_value, rgb, endpoint_a, protected),
                        "reference_contract": "LINEAR_RGB_BLEND_OF_EXISTING_FULL_ENDPOINTS",
                    })
                    panels.append((f"a={alpha:.2f}", rgb_path))
                sheet_rows.append((condition, panels))
            sheet = attempt / "visuals/channel_interpolation" / variant / f"{pair_id}.png"
            sealed.contact_sheet(sheet, sheet_rows)
            sheet_paths.append(str(sheet))
    if len(records) != 720 or created != 720 - existing_at_start:
        raise RuntimeError("channel render accounting mismatch")
    aggregate: dict[str, Any] = {}
    for variant in VARIANTS:
        selected = [row for row in records if row["variant"] == variant]
        aggregate[variant] = {
            metric: float(statistics.median(float(row[metric]) for row in selected))
            for metric in ("garment_rgb_mae", "lpips", "silhouette_iou", "boundary_fscore", "protected_lpips")
        }
    result = {
        "schema_version": "canondressgs.research.channel_interpolation_results.v1",
        "status": "AUTOMATIC_METRICS_COMPLETE_MANUAL_REVIEW_REQUIRED",
        "label": "RESEARCH DIAGNOSTIC — NOT PAPER FINAL",
        "protocol_sha256_lf": PROTOCOL_SHA256,
        "render_count": len(records),
        "expected_render_count": 720,
        "new_render_count": created,
        "reused_completed_render_count": existing_at_start,
        "repeated_completed_render_count": 0,
        "reused_full_render_count": 440,
        "full_render_regeneration_count": 0,
        "records": records,
        "automatic_aggregates": aggregate,
        "contact_sheets": sheet_paths,
        "manual_attribution_pending": True,
        "counts": {
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_created": 0,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "checkpoint_writes": 0,
        },
        "paper_final": False,
    }
    atomic_json(result_path, result)
    atomic_json(
        attempt / "audits/channel_resume_manifest.json",
        {
            "status": "COMPLETE",
            "expected": 720,
            "completed_before_resume": existing_at_start,
            "newly_completed": created,
            "completed_after": 720,
            "completed_items_repeated": 0,
        },
        replace=True,
    )
    return result


def normalized_channels(residual: Any) -> dict[str, torch.Tensor]:
    return {
        name: getattr(residual, name).detach().double().reshape(getattr(residual, name).shape[0], -1) / BOUNDS[name]
        for name in CHANNELS
    }


def residual_matrix(residual: Any) -> torch.Tensor:
    values = normalized_channels(residual)
    return torch.cat([values[name] for name in CHANNELS], dim=1)


def residual_energy(residual: Any) -> torch.Tensor:
    return residual_matrix(residual).square().mean(dim=1).sqrt()


def top_indices(energy: torch.Tensor, fraction: float) -> np.ndarray:
    values = energy.detach().cpu().numpy()
    count = int(math.ceil(values.shape[0] * fraction))
    indices = np.arange(values.shape[0])
    return np.lexsort((indices, -values))[:count]


def cosine_concatenated(first: torch.Tensor, second: torch.Tensor, mask: torch.Tensor) -> float:
    a = first[mask].reshape(-1)
    b = second[mask].reshape(-1)
    denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    return float((a @ b) / denominator) if float(denominator) > 0 else 1.0 if torch.equal(a, b) else 0.0


def scalar_sign_conflict(first: torch.Tensor, second: torch.Tensor, epsilon: float = 1e-8) -> float:
    active = (first.abs() > epsilon) & (second.abs() > epsilon)
    denominator = int(active.sum())
    return float(((first * second < 0) & active).sum() / denominator) if denominator else 0.0


def median_direction_cosine(first: torch.Tensor, second: torch.Tensor, epsilon: float = 1e-8) -> tuple[float, float]:
    first_norm = torch.linalg.vector_norm(first, dim=1)
    second_norm = torch.linalg.vector_norm(second, dim=1)
    active = (first_norm > epsilon) & (second_norm > epsilon)
    if not bool(active.any()):
        return 1.0, 0.0
    values = torch.nn.functional.cosine_similarity(first[active], second[active], dim=1)
    return float(values.median()), float((values < 0).double().mean())


def symmetric_boundary_distance(first: torch.Tensor, second: torch.Tensor) -> float:
    a = first.detach().cpu().numpy().astype(bool)
    b = second.detach().cpu().numpy().astype(bool)
    a_boundary = a ^ ndimage.binary_erosion(a)
    b_boundary = b ^ ndimage.binary_erosion(b)
    if not a_boundary.any() or not b_boundary.any():
        return float(max(a.shape))
    distance_b = ndimage.distance_transform_edt(~b_boundary)
    distance_a = ndimage.distance_transform_edt(~a_boundary)
    return float(0.5 * (distance_b[a_boundary].mean() + distance_a[b_boundary].mean()))


def component_count(points: np.ndarray, bbox_min: np.ndarray, voxel_size: float) -> int:
    coordinates = np.floor((points - bbox_min) / voxel_size).astype(np.int64)
    coordinates = np.clip(coordinates, 0, 127)
    grid = np.zeros((128, 128, 128), dtype=np.bool_)
    grid[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] = True
    _, count = ndimage.label(grid, structure=np.ones((3, 3, 3), dtype=np.uint8))
    return int(count)


def transition_state(geometry: torch.Tensor, appearance: torch.Tensor) -> torch.Tensor:
    return geometry.to(torch.int64) + 2 * appearance.to(torch.int64)


def run_support_and_role(runtime_value: sealed.EvaluationRuntime, full: Mapping[tuple[str, str, float], Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    support_path = runtime_value.attempt / "aggregates/support_conflict_results.json"
    role_path = runtime_value.attempt / "aggregates/per_gaussian_role_conflict_results.json"
    if support_path.is_file() and role_path.is_file():
        return read_json(support_path), read_json(role_path)
    teachers = runtime_value.teachers
    base_xyz = runtime_value.context["base"]._xyz.detach().double()
    base_opacity = runtime_value.context["base"]._opacity.detach().double().reshape(-1)
    protected = runtime_value.context["protected_mask"].detach().bool().reshape(-1)
    bbox_min = base_xyz.min(0).values
    bbox_max = base_xyz.max(0).values
    bbox_diagonal = float(torch.linalg.vector_norm(bbox_max - bbox_min))
    voxel_size = bbox_diagonal / 128.0
    matrices = {name: residual_matrix(value) for name, value in teachers.items()}
    energies = {name: value.square().mean(dim=1).sqrt() for name, value in matrices.items()}
    channels = {name: normalized_channels(value) for name, value in teachers.items()}
    support_rows: list[dict[str, Any]] = []
    role_rows: list[dict[str, Any]] = []
    for left, right in PAIRS:
        pair_id = f"{left}_{right}"
        energy_a, energy_b = energies[left], energies[right]
        active_a, active_b = energy_a > 1e-8, energy_b > 1e-8
        union = active_a | active_b
        overlap = active_a & active_b
        a_only = active_a & ~active_b
        b_only = active_b & ~active_a
        union_count = int(union.sum())
        top10_a, top10_b = set(top_indices(energy_a, 0.10).tolist()), set(top_indices(energy_b, 0.10).tolist())
        top20_a_np, top20_b_np = top_indices(energy_a, 0.20), top_indices(energy_b, 0.20)
        top20_a, top20_b = set(top20_a_np.tolist()), set(top20_b_np.tolist())
        xyz_a = channels[left]["delta_xyz"]
        xyz_b = channels[right]["delta_xyz"]
        xyz_cosine, xyz_conflict = median_direction_cosine(xyz_a, xyz_b)
        opacity_a = channels[left]["delta_opacity_logit"].reshape(-1)
        opacity_b = channels[right]["delta_opacity_logit"].reshape(-1)
        activation_a = torch.sigmoid(base_opacity + getattr(teachers[left], "delta_opacity_logit").detach().double().reshape(-1)) - torch.sigmoid(base_opacity)
        activation_b = torch.sigmoid(base_opacity + getattr(teachers[right], "delta_opacity_logit").detach().double().reshape(-1)) - torch.sigmoid(base_opacity)
        centroid_a = (base_xyz * energy_a[:, None]).sum(0) / energy_a.sum().clamp_min(1e-12)
        centroid_b = (base_xyz * energy_b[:, None]).sum(0) / energy_b.sum().clamp_min(1e-12)
        silhouette_ious: list[float] = []
        boundary_distances: list[float] = []
        for condition in CONDITIONS:
            alpha_a = sealed.image_tensor(Path(full[(pair_id, condition, 0.0)]["alpha_path"]), 1)[0] >= 0.5
            alpha_b = sealed.image_tensor(Path(full[(pair_id, condition, 1.0)]["alpha_path"]), 1)[0] >= 0.5
            intersection = int((alpha_a & alpha_b).sum())
            union_pixels = int((alpha_a | alpha_b).sum())
            silhouette_ious.append(intersection / union_pixels if union_pixels else 1.0)
            boundary_distances.append(symmetric_boundary_distance(alpha_a, alpha_b))
        component_a = component_count(base_xyz[top20_a_np].cpu().numpy(), bbox_min.cpu().numpy(), voxel_size)
        component_b = component_count(base_xyz[top20_b_np].cpu().numpy(), bbox_min.cpu().numpy(), voxel_size)
        sh0_diff = float((channels[left]["delta_sh0"][union] - channels[right]["delta_sh0"][union]).square().mean().sqrt()) if bool(union.any()) else 0.0
        shn_diff = float((channels[left]["delta_shN"][union] - channels[right]["delta_shN"][union]).square().mean().sqrt()) if bool(union.any()) else 0.0
        support_rows.append({
            "pair": [left, right],
            "pair_id": pair_id,
            "stable_label": pair_id in STABLE,
            "top_10_support_overlap": len(top10_a & top10_b) / len(top10_a),
            "top_20_support_overlap": len(top20_a & top20_b) / len(top20_a),
            "weighted_jaccard": float(torch.minimum(energy_a, energy_b).sum() / torch.maximum(energy_a, energy_b).sum().clamp_min(1e-12)),
            "support_union_count": union_count,
            "support_union_fraction": union_count / energy_a.numel(),
            "A_only_support_fraction": int(a_only.sum()) / union_count if union_count else 0.0,
            "B_only_support_fraction": int(b_only.sum()) / union_count if union_count else 0.0,
            "exclusive_fraction_sum": int((a_only | b_only).sum()) / union_count if union_count else 0.0,
            "overlap_residual_cosine": cosine_concatenated(matrices[left], matrices[right], overlap),
            "overlap_xyz_direction_cosine": xyz_cosine,
            "xyz_direction_conflict": xyz_conflict,
            "xyz_sign_conflict_fraction": scalar_sign_conflict(xyz_a, xyz_b),
            "scale_sign_conflict_fraction": scalar_sign_conflict(channels[left]["delta_log_scaling"], channels[right]["delta_log_scaling"]),
            "opacity_activation_conflict_fraction": scalar_sign_conflict(activation_a, activation_b),
            "opacity_sign_conflict_fraction": scalar_sign_conflict(opacity_a, opacity_b),
            "SH0_disagreement": sh0_diff,
            "SHN_disagreement": shn_diff,
            "sh_disagreement": 0.5 * (sh0_diff + shn_diff),
            "teacher_silhouette_iou": float(statistics.median(silhouette_ious)),
            "boundary_distance": float(statistics.median(boundary_distances)),
            "protected_support_conflict": int((union & protected).sum()) / union_count if union_count else 0.0,
            "support_centroid_distance": float(torch.linalg.vector_norm(centroid_a - centroid_b)) / bbox_diagonal,
            "connected_components_A_top20": component_a,
            "connected_components_B_top20": component_b,
            "connected_component_difference": abs(component_a - component_b),
            "artifact_contributors": {
                "overlap_support_count": int(overlap.sum()),
                "A_only_support_count": int(a_only.sum()),
                "B_only_support_count": int(b_only.sum()),
                "protected_support_count": int((union & protected).sum()),
            },
            "support_compatibility_profile": "PENDING_STABLE_UNSTABLE_AGGREGATION",
        })
        geometry_a = torch.cat([channels[left][name] for name in ("delta_xyz", "delta_log_scaling", "delta_rotvec")], 1).square().mean(1).sqrt() > 1e-8
        geometry_b = torch.cat([channels[right][name] for name in ("delta_xyz", "delta_log_scaling", "delta_rotvec")], 1).square().mean(1).sqrt() > 1e-8
        appearance_a = torch.cat([channels[left][name] for name in ("delta_opacity_logit", "delta_sh0", "delta_shN")], 1).square().mean(1).sqrt() > 1e-8
        appearance_b = torch.cat([channels[right][name] for name in ("delta_opacity_logit", "delta_sh0", "delta_shN")], 1).square().mean(1).sqrt() > 1e-8
        state_a, state_b = transition_state(geometry_a, appearance_a), transition_state(geometry_b, appearance_b)
        transition = torch.zeros((4, 4), dtype=torch.int64, device=state_a.device)
        for source in range(4):
            for target in range(4):
                transition[source, target] = ((state_a == source) & (state_b == target)).sum()
        sh0_a = channels[left]["delta_sh0"]
        sh0_b = channels[right]["delta_sh0"]
        _, sh_conflict = median_direction_cosine(sh0_a, sh0_b)
        ratios = torch.maximum(energy_a, energy_b) / torch.minimum(energy_a, energy_b).clamp_min(1e-8)
        role_rows.append({
            "pair": [left, right],
            "pair_id": pair_id,
            "stable_label": pair_id in STABLE,
            "residual_magnitude_ratio": {
                "median": float(ratios.median()),
                "p95": float(torch.quantile(ratios, 0.95)),
                "max": float(ratios.max()),
            },
            "displacement_direction_conflict_fraction": xyz_conflict,
            "opacity_activation_deactivation_conflict_fraction": scalar_sign_conflict(activation_a, activation_b),
            "SH_color_direction_conflict_fraction": sh_conflict,
            "geometry_active_appearance_inactive_A_fraction": float((geometry_a & ~appearance_a).double().mean()),
            "geometry_active_appearance_inactive_B_fraction": float((geometry_b & ~appearance_b).double().mean()),
            "appearance_active_geometry_inactive_A_fraction": float((appearance_a & ~geometry_a).double().mean()),
            "appearance_active_geometry_inactive_B_fraction": float((appearance_b & ~geometry_b).double().mean()),
            "channel_role_transition_matrix": transition.cpu().tolist(),
            "transition_state_order": ["BOTH_INACTIVE", "GEOMETRY_ONLY", "APPEARANCE_ONLY", "BOTH_ACTIVE"],
            "regional_mapping": {
                "status": "REGION_MAPPING_UNAVAILABLE",
                "reason": "No frozen sleeves/trousers Gaussian mapping exists; no posthoc manual partition was created.",
                "available_frozen_partition": "protected versus non-protected membership only",
            },
            "per_gaussian_role_conflict_profile": "QUANTIFIED_WITHOUT_POSTHOC_REGION_LABELS",
        })
    support_result = {
        "schema_version": "canondressgs.research.support_conflict_results.v1",
        "status": "COMPLETE",
        "label": "RESEARCH DIAGNOSTIC — NOT PAPER FINAL",
        "protocol_sha256_lf": PROTOCOL_SHA256,
        "active_threshold": 1e-8,
        "pair_count": len(support_rows),
        "records": support_rows,
        "counts": {"training_steps": 0, "backward_calls": 0, "optimizer_created": 0, "checkpoint_writes": 0},
        "paper_final": False,
    }
    role_result = {
        "schema_version": "canondressgs.research.per_gaussian_role_conflict_results.v1",
        "status": "COMPLETE",
        "label": "RESEARCH DIAGNOSTIC — NOT PAPER FINAL",
        "protocol_sha256_lf": PROTOCOL_SHA256,
        "pair_count": len(role_rows),
        "records": role_rows,
        "counts": {"training_steps": 0, "backward_calls": 0, "optimizer_created": 0, "checkpoint_writes": 0},
        "paper_final": False,
    }
    atomic_json(support_path, support_result)
    atomic_json(role_path, role_result)
    return support_result, role_result


def draw_support_projection(
    image: Image.Image,
    box: tuple[int, int, int, int],
    xyz: np.ndarray,
    classes: np.ndarray,
    axes: tuple[int, int],
    title: str,
) -> None:
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = box
    draw.rectangle(box, outline="white", width=1)
    draw.text((left + 4, top + 4), title, fill="white")
    selected = classes > 0
    points = xyz[selected]
    labels = classes[selected]
    if not len(points):
        return
    minimum = xyz[:, axes].min(0)
    maximum = xyz[:, axes].max(0)
    span = np.maximum(maximum - minimum, 1e-12)
    normalized = (points[:, axes] - minimum) / span
    width, height = right - left - 8, bottom - top - 24
    colors = {1: (180, 70, 220), 2: (255, 80, 60), 3: (70, 150, 255), 4: (255, 220, 60)}
    for point, label in zip(normalized, labels):
        x = left + 4 + int(point[0] * width)
        y = bottom - 4 - int(point[1] * height)
        image.putpixel((max(left + 1, min(right - 1, x)), max(top + 1, min(bottom - 1, y))), colors[int(label)])


def support_classes(left_energy: torch.Tensor, right_energy: torch.Tensor, protected: torch.Tensor) -> np.ndarray:
    active_left = left_energy > 1e-8
    active_right = right_energy > 1e-8
    values = torch.zeros_like(active_left, dtype=torch.uint8)
    values[active_left & active_right] = 1
    values[active_left & ~active_right] = 2
    values[~active_left & active_right] = 3
    values[(active_left | active_right) & protected] = 4
    return values.cpu().numpy()


def make_support_overlay(
    path: Path,
    pair_id: str,
    full: Mapping[tuple[str, str, float], Mapping[str, Any]],
    xyz: np.ndarray,
    classes: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (1600, 900), "black")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), f"{pair_id} support conflict | purple overlap | red A-only | blue B-only | yellow protected", fill="white")
    draw_support_projection(canvas, (10, 35, 520, 545), xyz, classes, (0, 1), "canonical XY")
    draw_support_projection(canvas, (535, 35, 1045, 545), xyz, classes, (0, 2), "canonical XZ")
    draw_support_projection(canvas, (1060, 35, 1570, 545), xyz, classes, (1, 2), "canonical YZ")
    thumb_w, thumb_h = 380, 285
    for index, condition in enumerate(CONDITIONS):
        source = Image.open(full[(pair_id, condition, 0.5)]["rgb_path"]).convert("RGB")
        source.thumbnail((thumb_w, thumb_h))
        x = 10 + index * 397
        y = 585
        canvas.paste(source, (x, y))
        draw.text((x, 560), f"FULL a=0.5 {condition}", fill="white")
    canvas.save(path)


def run_visuals(runtime_value: sealed.EvaluationRuntime, full: Mapping[tuple[str, str, float], Mapping[str, Any]]) -> dict[str, Any]:
    manifest_path = runtime_value.attempt / "audits/visual_manifest.json"
    if manifest_path.is_file():
        return read_json(manifest_path)
    teachers = runtime_value.teachers
    energies = {name: residual_energy(value) for name, value in teachers.items()}
    protected = runtime_value.context["protected_mask"].detach().bool().reshape(-1)
    xyz = runtime_value.context["base"]._xyz.detach().cpu().numpy()
    support_sheets: list[str] = []
    for left, right in PAIRS:
        pair_id = f"{left}_{right}"
        path = runtime_value.attempt / "visuals/support_conflict" / f"{pair_id}.png"
        make_support_overlay(path, pair_id, full, xyz, support_classes(energies[left], energies[right], protected))
        support_sheets.append(str(path))
    comparison_sheets: list[str] = []
    for label, selected_pairs in (
        ("stable", [pair for pair in PAIRS if f"{pair[0]}_{pair[1]}" in STABLE]),
        ("unstable", [pair for pair in PAIRS if f"{pair[0]}_{pair[1]}" not in STABLE]),
    ):
        rows = []
        for left, right in selected_pairs:
            pair_id = f"{left}_{right}"
            rows.append((pair_id, [(condition, Path(full[(pair_id, condition, 0.5)]["rgb_path"])) for condition in CONDITIONS]))
        path = runtime_value.attempt / "visuals/stable_unstable" / f"{label}.png"
        sealed.contact_sheet(path, rows)
        comparison_sheets.append(str(path))
    channel_sheets = [
        str(runtime_value.attempt / "visuals/channel_interpolation" / variant / f"{left}_{right}.png")
        for variant in VARIANTS
        for left, right in PAIRS
    ]
    full_sheets = read_json(SEALED_INTERPOLATION)["contact_sheets"]
    all_paths = full_sheets + channel_sheets + support_sheets + comparison_sheets
    missing = [path for path in all_paths if not Path(path).is_file()]
    if len(full_sheets) != 10 or len(channel_sheets) != 60 or len(support_sheets) != 10 or len(comparison_sheets) != 2 or missing:
        raise RuntimeError(f"visual manifest mismatch: {missing[:3]}")
    result = {
        "schema_version": "canondressgs.research.continuous_control_root_cause_visual_manifest.v1",
        "status": "MANUAL_REVIEW_REQUIRED",
        "label": "RESEARCH DIAGNOSTIC — NOT PAPER FINAL",
        "counts": {"FULL": 10, "CHANNEL": 60, "SUPPORT_OVERLAY": 10, "STABLE_UNSTABLE_COMPARISON": 2, "total": 82},
        "full_sheets": full_sheets,
        "channel_sheets": channel_sheets,
        "support_conflict_overlays": support_sheets,
        "stable_unstable_comparison_sheets": comparison_sheets,
        "all_paths_exist": True,
        "paper_final": False,
    }
    atomic_json(manifest_path, result)
    return result


def median(values: Iterable[float]) -> float:
    return float(statistics.median(float(value) for value in values))


def exact_permutation_p(values: Sequence[float], stable_indices: set[int]) -> float:
    observed_stable = [value for index, value in enumerate(values) if index in stable_indices]
    observed_unstable = [value for index, value in enumerate(values) if index not in stable_indices]
    observed = abs(median(observed_stable) - median(observed_unstable))
    count = 0
    total = 0
    for assignment in itertools.combinations(range(len(values)), 3):
        assignment_set = set(assignment)
        statistic = abs(
            median(value for index, value in enumerate(values) if index in assignment_set)
            - median(value for index, value in enumerate(values) if index not in assignment_set)
        )
        count += int(statistic >= observed - 1e-15)
        total += 1
    if total != 120:
        raise RuntimeError("exact permutation enumeration mismatch")
    return count / total


def correlation_results(support: Mapping[str, Any], full_scores: Mapping[str, int], alignment_count: int) -> dict[str, Any]:
    rows = list(support["records"])
    features = [
        ("top_10_support_overlap", "higher"),
        ("top_20_support_overlap", "higher"),
        ("weighted_jaccard", "higher"),
        ("exclusive_fraction_sum", "lower"),
        ("overlap_residual_cosine", "higher"),
        ("xyz_direction_conflict", "lower"),
        ("opacity_activation_conflict_fraction", "lower"),
        ("sh_disagreement", "lower"),
        ("teacher_silhouette_iou", "higher"),
        ("boundary_distance", "lower"),
    ]
    labels = np.asarray([1.0 if row["stable_label"] else 0.0 for row in rows])
    stable_indices = {index for index, row in enumerate(rows) if row["stable_label"]}
    records = []
    for name, direction in features:
        values = np.asarray([float(row[name]) for row in rows])
        rho = float(stats.spearmanr(values, labels).statistic)
        if not math.isfinite(rho):
            rho = 0.0
        stable_median = median(values[index] for index in stable_indices)
        unstable_median = median(values[index] for index in range(len(values)) if index not in stable_indices)
        gap = stable_median - unstable_median
        direction_pass = (rho >= 0.0 and gap >= 0.0) if direction == "higher" else (rho <= 0.0 and gap <= 0.0)
        records.append({
            "feature": name,
            "hypothesized_stable_direction": direction,
            "stable_median": stable_median,
            "unstable_median": unstable_median,
            "median_gap_stable_minus_unstable": gap,
            "spearman_rho_with_stable_label": rho,
            "exact_permutation_p_value": exact_permutation_p(values.tolist(), stable_indices),
            "rank_order": [rows[index]["pair_id"] for index in np.argsort(values).tolist()],
            "strong": abs(rho) >= 0.60 and direction_pass,
            "direction_pass": direction_pass,
        })
    strong_count = sum(bool(row["strong"]) for row in records)
    artifact_scores = [int(full_scores[row["pair_id"]]) for row in rows]
    artifact_rho = float(stats.spearmanr(artifact_scores, labels).statistic)
    if not math.isfinite(artifact_rho):
        artifact_rho = 0.0
    return {
        "schema_version": "canondressgs.research.stable_pair_correlation_results.v1",
        "status": "COMPLETE",
        "label": "RESEARCH DIAGNOSTIC — NOT PAPER FINAL",
        "stable_pairs": sorted(STABLE),
        "unstable_pairs": [row["pair_id"] for row in rows if not row["stable_label"]],
        "records": records,
        "strong_metric_count": strong_count,
        "exclusive_support_alignment_count": alignment_count,
        "strong_support_conflict_evidence": strong_count >= 3 and alignment_count >= 6,
        "full_artifact_score_spearman_with_stable_label": artifact_rho,
        "classifier_trained": False,
        "posthoc_features_added": False,
        "paper_final": False,
    }


def review_index(review: Mapping[str, Any], kind: str) -> list[Mapping[str, Any]]:
    return [item for item in review["items"] if item["kind"] == kind]


def grade_max(item: Mapping[str, Any], *, midpoint: bool = False) -> int:
    grades = item["alpha_0_50_grades"] if midpoint and "alpha_0_50_grades" in item else item["grades"]
    return max(int(grades[name]) for name in CORE_GRADES)


def channel_attribution(channel: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, Any]:
    full_items = {item["pair_id"]: item for item in review_index(review, "FULL")}
    channel_items = {(item["variant"], item["pair_id"]): item for item in review_index(review, "CHANNEL")}
    reproduction: dict[str, Any] = {}
    for variant in VARIANTS:
        pair_rows = []
        for left, right in PAIRS:
            pair_id = f"{left}_{right}"
            full_grade = grade_max(full_items[pair_id], midpoint=True)
            variant_grade = grade_max(channel_items[(variant, pair_id)], midpoint=True)
            reproduced = variant_grade >= 2 and abs(variant_grade - full_grade) <= 1
            pair_rows.append({
                "pair_id": pair_id,
                "full_midpoint_max_core_grade": full_grade,
                "variant_midpoint_max_core_grade": variant_grade,
                "reproduces_full_artifact": reproduced,
            })
        reproduction[variant] = {
            "reproduction_count": sum(bool(row["reproduces_full_artifact"]) for row in pair_rows),
            "none_minor_count": sum(int(row["variant_midpoint_max_core_grade"]) <= 1 for row in pair_rows),
            "pairs": pair_rows,
        }
    geometry = reproduction["GEOMETRY_ALL"]
    appearance = reproduction["APPEARANCE_ALL"]
    opacity = reproduction["OPACITY_ONLY"]
    sh_only = reproduction["SH_ONLY"]
    full_moderate = sum(grade_max(item, midpoint=True) >= 2 for item in full_items.values())
    if geometry["reproduction_count"] >= 6 and appearance["none_minor_count"] >= 6:
        classification = "GEOMETRY_INTERPOLATION_DOMINANT"
    elif appearance["reproduction_count"] >= 6 and geometry["none_minor_count"] >= 6:
        classification = "APPEARANCE_INTERPOLATION_DOMINANT"
    elif opacity["reproduction_count"] >= 6 and max(
        reproduction[name]["reproduction_count"] for name in ("XYZ_ONLY", "SCALE_ROT_ONLY", "SH_ONLY")
    ) < opacity["reproduction_count"]:
        classification = "OPACITY_INTERPOLATION_DOMINANT"
    elif sh_only["reproduction_count"] >= 6 and max(
        reproduction[name]["reproduction_count"] for name in ("XYZ_ONLY", "SCALE_ROT_ONLY", "OPACITY_ONLY")
    ) < sh_only["reproduction_count"]:
        classification = "SH_INTERPOLATION_DOMINANT"
    elif geometry["none_minor_count"] >= 6 and appearance["none_minor_count"] >= 6 and full_moderate >= 6:
        classification = "CROSS_CHANNEL_COUPLING_DOMINANT"
    else:
        classification = "NO_SINGLE_CHANNEL_DOMINANT"
    result = dict(channel)
    result.update({
        "status": "COMPLETE",
        "manual_attribution_pending": False,
        "visual_reproduction": reproduction,
        "full_moderate_severe_pair_count": full_moderate,
        "channel_attribution": classification,
        "identifiability_note": (
            "At alpha=0.5 every preregistered variant equals the same six-channel midpoint because "
            "all selected and non-selected channels use weight 0.5. Midpoint reproduction counts are "
            "therefore descriptive but cannot identify a unique channel cause."
        ),
    })
    return result


def choose_root_cause(attribution: Mapping[str, Any], correlations: Mapping[str, Any]) -> tuple[str, list[str], str]:
    classification = attribution["channel_attribution"]
    support_strong = bool(correlations["strong_support_conflict_evidence"])
    secondary: list[str] = []
    if support_strong and classification == "NO_SINGLE_CHANNEL_DOMINANT":
        primary = "SUPPORT_MISMATCH"
        secondary = ["CROSS_CHANNEL_COUPLING"]
    elif classification == "GEOMETRY_INTERPOLATION_DOMINANT":
        primary = "GEOMETRY_INTERPOLATION"
    elif classification == "APPEARANCE_INTERPOLATION_DOMINANT":
        primary = "APPEARANCE_INTERPOLATION"
    elif classification == "OPACITY_INTERPOLATION_DOMINANT":
        primary = "OPACITY_INTERPOLATION"
    elif classification == "CROSS_CHANNEL_COUPLING_DOMINANT":
        primary = "CROSS_CHANNEL_COUPLING"
        if support_strong:
            secondary = ["SUPPORT_MISMATCH"]
    else:
        primary = "MULTIPLE_FACTORS"
        secondary = ["SUPPORT_MISMATCH"] if support_strong else ["CROSS_CHANNEL_COUPLING"]
    next_task = (
        "RUN_SUPPORT_AWARE_HYBRID_CONTROL_MICRO_PILOT"
        if primary == "SUPPORT_MISMATCH" or (primary == "MULTIPLE_FACTORS" and support_strong)
        else "RUN_GEOMETRY_APPEARANCE_DISENTANGLED_BASIS_MICRO_PILOT"
    )
    return primary, secondary, next_task


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def write_reports(
    attribution: Mapping[str, Any],
    support: Mapping[str, Any],
    role: Mapping[str, Any],
    correlations: Mapping[str, Any],
    review: Mapping[str, Any],
    primary: str,
    secondary: Sequence[str],
    next_task: str,
) -> dict[str, str]:
    attribution_rows = [
        [name, value["reproduction_count"], value["none_minor_count"]]
        for name, value in attribution["visual_reproduction"].items()
    ]
    support_rows = [
        [
            row["pair_id"],
            "stable" if row["stable_label"] else "unstable",
            f"{row['top_10_support_overlap']:.4f}",
            f"{row['weighted_jaccard']:.4f}",
            f"{row['exclusive_fraction_sum']:.4f}",
            f"{row['xyz_direction_conflict']:.4f}",
            f"{row['opacity_activation_conflict_fraction']:.4f}",
        ]
        for row in support["records"]
    ]
    correlation_rows = [
        [
            row["feature"],
            f"{row['stable_median']:.5g}",
            f"{row['unstable_median']:.5g}",
            f"{row['spearman_rho_with_stable_label']:.4f}",
            f"{row['exact_permutation_p_value']:.4f}",
            row["strong"],
        ]
        for row in correlations["records"]
    ]
    root_report = (
        "# AAAI27 Continuous-Control Artifact Root Cause\n\n"
        "**RESEARCH DIAGNOSTIC — NOT PAPER FINAL**\n\n"
        f"- Source: `{SOURCE_HEAD}` on the sealed P0 evaluation branch.\n"
        f"- Protocol SHA-256: `{PROTOCOL_SHA256}`.\n"
        "- FULL interpolation was reused read-only: 440/440; no FULL render was regenerated.\n"
        "- New channel-group renders: 720/720.\n"
        f"- Manual visual review: {review['actual_opened_count']}/{review['expected_count']} sheets opened.\n"
        f"- PRIMARY_FAILURE_SOURCE: **{primary}**.\n"
        f"- SECONDARY_FAILURE_SOURCES: `{json.dumps(list(secondary))}`.\n"
        f"- NEXT_TASK: **{next_task}** (not started).\n\n"
        "## Channel attribution\n\n"
        + markdown_table(["variant", "FULL reproduction pairs", "NONE/MINOR pairs"], attribution_rows)
        + "\n\n"
        + attribution["identifiability_note"]
        + "\n\n## Stable versus unstable preregistered correlations\n\n"
        + markdown_table(["feature", "stable median", "unstable median", "rho", "exact p", "strong"], correlation_rows)
        + "\n\n"
        f"Strong preregistered metrics: {correlations['strong_metric_count']}/10; "
        f"exclusive-support visual alignments: {correlations['exclusive_support_alignment_count']}/10.\n\n"
        "## Scientific boundary\n\n"
        "All observed patch, mottle, cloud, edge-scatter, full-body contamination, identity-contamination, "
        "and silhouette-discontinuity grades are retained item by item. No pair, channel, or failure was "
        "removed; no parameter, threshold, stable label, teacher, basis, or protocol was changed.\n\n"
        "Training steps, backward calls, optimizer creations/steps, scheduler steps, and checkpoint writes were all zero. PAPER_FINAL=0.\n"
    )
    channel_report = (
        "# AAAI27 Channel Interpolation Attribution\n\n"
        "**RESEARCH DIAGNOSTIC — NOT PAPER FINAL**\n\n"
        + markdown_table(["variant", "FULL reproduction pairs", "NONE/MINOR pairs"], attribution_rows)
        + "\n\n"
        f"Classification: **{attribution['channel_attribution']}**.\n\n"
        + attribution["identifiability_note"]
        + "\n\nEvery variant obeyed the frozen rule: selected channels used alpha, while all other channels used the fixed 0.5 teacher midpoint.\n"
    )
    support_report = (
        "# AAAI27 Support Compatibility Analysis\n\n"
        "**RESEARCH DIAGNOSTIC — NOT PAPER FINAL**\n\n"
        + markdown_table(
            ["pair", "label", "top10", "weighted Jaccard", "exclusive", "xyz conflict", "opacity conflict"],
            support_rows,
        )
        + "\n\n## Preregistered stable/unstable analysis\n\n"
        + markdown_table(["feature", "stable median", "unstable median", "rho", "exact p", "strong"], correlation_rows)
        + "\n\nNo sleeves/trousers Gaussian mapping was invented. The per-Gaussian report uses only the frozen index order and protected membership.\n"
    )
    outputs = {
        "docs/PAPER/AAAI27_CONTINUOUS_CONTROL_ARTIFACT_ROOT_CAUSE_20260722.md": root_report,
        "docs/PAPER/AAAI27_CHANNEL_INTERPOLATION_ATTRIBUTION_20260722.md": channel_report,
        "docs/PAPER/AAAI27_SUPPORT_COMPATIBILITY_ANALYSIS_20260722.md": support_report,
    }
    for relative, content in outputs.items():
        path = PROJECT_ROOT / relative
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
    return outputs


def copy_json_to_repo(relative: str, value: Mapping[str, Any]) -> None:
    atomic_json(PROJECT_ROOT / relative, value)


def run_finalize(attempt: Path, asset_root: Path, review_path: Path) -> dict[str, Any]:
    summary_path = attempt / "aggregates/continuous_control_root_cause_final_summary.json"
    if summary_path.is_file():
        return read_json(summary_path)
    channel = read_json(attempt / "aggregates/channel_interpolation_results.json")
    support = read_json(attempt / "aggregates/support_conflict_results.json")
    role = read_json(attempt / "aggregates/per_gaussian_role_conflict_results.json")
    manifest = read_json(attempt / "audits/visual_manifest.json")
    review = read_json(review_path)
    if review["expected_count"] != 82 or review["actual_opened_count"] != 82:
        raise RuntimeError("manual review must persist 82/82 actual opens")
    if len(review["items"]) != 82 or any(not item["actual_opened"] for item in review["items"]):
        raise RuntimeError("manual review item contract")
    if {item["source_path"] for item in review["items"]} != set(manifest["full_sheets"] + manifest["channel_sheets"] + manifest["support_conflict_overlays"] + manifest["stable_unstable_comparison_sheets"]):
        raise RuntimeError("manual review paths do not match visual manifest")
    full_scores = {item["pair_id"]: grade_max(item, midpoint=True) for item in review_index(review, "FULL")}
    alignment_count = sum(bool(item["support_exclusive_alignment"]) for item in review_index(review, "SUPPORT_OVERLAY"))
    attribution = channel_attribution(channel, review)
    correlations = correlation_results(support, full_scores, alignment_count)
    primary, secondary, next_task = choose_root_cause(attribution, correlations)
    preflight = read_json(attempt / "audits/preflight.json")
    frozen_after = {
        "formal": tree_manifest(asset_root / sealed.FORMAL_NAME),
        "p0": tree_manifest(asset_root / sealed.P0_NAME),
        "sealed_evaluation": tree_manifest(asset_root / sealed.OUTPUT_NAME),
    }
    frozen_unchanged = frozen_after == preflight["frozen_trees_before"]
    if not frozen_unchanged:
        raise RuntimeError("ROOT-CAUSE-NO-TRAINING-GATE-VIOLATION: frozen output mutation")
    asset_verification_after = verify_manifest(read_json(FROZEN_MANIFEST), PROJECT_ROOT, asset_root, verify_external=True)
    if asset_verification_after["status"] != "PASS":
        raise RuntimeError("ROOT-CAUSE-NO-TRAINING-GATE-VIOLATION: frozen asset mutation")
    correlations["paper_final"] = False
    summary = {
        "schema_version": "canondressgs.research.continuous_control_root_cause_final_summary.v1",
        "status": "COMPLETE",
        "label": "RESEARCH DIAGNOSTIC — NOT PAPER FINAL",
        "task_id": TASK_ID,
        "source_head": SOURCE_HEAD,
        "run_branch": RUN_BRANCH,
        "execution_head": git("rev-parse", "HEAD"),
        "protocol_sha256_lf": PROTOCOL_SHA256,
        "archive_seal_gate": "PASS",
        "counts": {
            "channel_renders": 720,
            "reused_full_renders": 440,
            "full_rerenders": 0,
            "visual_review": 82,
            "stable_pairs": 3,
            "unstable_pairs": 7,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_created": 0,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "checkpoint_writes": 0,
            "teacher_mutation": 0,
            "basis_mutation": 0,
            "formal_output_mutation": 0,
            "paper_final": 0,
        },
        "channel_attribution": attribution["channel_attribution"],
        "primary_failure_source": primary,
        "secondary_failure_sources": secondary,
        "evidence_summary": {
            "strong_preregistered_metric_count": correlations["strong_metric_count"],
            "exclusive_support_alignment_count": alignment_count,
            "strong_support_conflict_evidence": correlations["strong_support_conflict_evidence"],
            "channel_midpoint_identifiable": False,
            "channel_midpoint_reason": attribution["identifiability_note"],
        },
        "frozen_assets_before": preflight["frozen_asset_verification"],
        "frozen_assets_after": asset_verification_after,
        "frozen_trees_before": preflight["frozen_trees_before"],
        "frozen_trees_after": frozen_after,
        "frozen_assets_unchanged": frozen_unchanged,
        "visual_review_path": str(review_path),
        "next_task": next_task,
        "next_task_started": False,
        "paper_final": False,
        "paper_final_count": 0,
    }
    atomic_json(attempt / "aggregates/channel_interpolation_results_final.json", attribution)
    atomic_json(attempt / "aggregates/stable_pair_correlation_results.json", correlations)
    atomic_json(attempt / "aggregates/continuous_control_root_cause_visual_review.json", review)
    atomic_json(summary_path, summary)
    copy_json_to_repo("paper_protocol/reviewer_risk/channel_interpolation_results.json", attribution)
    copy_json_to_repo("paper_protocol/reviewer_risk/support_conflict_results.json", support)
    copy_json_to_repo("paper_protocol/reviewer_risk/per_gaussian_role_conflict_results.json", role)
    copy_json_to_repo("paper_protocol/reviewer_risk/stable_pair_correlation_results.json", correlations)
    copy_json_to_repo("paper_protocol/reviewer_risk/continuous_control_root_cause_visual_review.json", review)
    copy_json_to_repo("paper_protocol/reviewer_risk/continuous_control_root_cause_final_summary.json", summary)
    write_reports(attribution, support, role, correlations, review, primary, secondary, next_task)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--attempt", default="attempt_001")
    parser.add_argument("--phase", choices=("preflight", "render", "analyze", "visuals", "finalize", "all"), default="all")
    parser.add_argument("--visual-review", type=Path)
    args = parser.parse_args()
    frozen_protocol = protocol()
    attempt = attempt_path(args.output_root.resolve(), args.attempt)
    if args.phase in {"preflight", "all"}:
        result = run_preflight(attempt, args.asset_root.resolve())
        print(json.dumps({"phase": "preflight", "status": result["status"]}, sort_keys=True))
        if args.phase == "preflight":
            return
    if not (attempt / "audits/preflight.json").is_file():
        raise RuntimeError("preflight must complete before diagnostic phases")
    validate_source_and_archive()
    full, _ = full_index()
    if args.phase in {"render", "analyze", "visuals", "all"}:
        runtime_value = runtime(attempt, args.asset_root.resolve(), frozen_protocol)
    if args.phase in {"render", "all"}:
        result = run_channel_render(runtime_value, full)
        print(json.dumps({"phase": "render", "render_count": result["render_count"]}, sort_keys=True))
        if args.phase == "render":
            return
    if args.phase in {"analyze", "all"}:
        support, role = run_support_and_role(runtime_value, full)
        print(json.dumps({"phase": "analyze", "support_pairs": support["pair_count"], "role_pairs": role["pair_count"]}, sort_keys=True))
        if args.phase == "analyze":
            return
    if args.phase in {"visuals", "all"}:
        result = run_visuals(runtime_value, full)
        print(json.dumps({"phase": "visuals", "visual_count": result["counts"]["total"]}, sort_keys=True))
        if args.phase == "visuals":
            return
    if args.phase in {"finalize", "all"}:
        if args.visual_review is None:
            if args.phase == "all":
                print(json.dumps({"phase": "finalize", "status": "MANUAL_REVIEW_REQUIRED"}, sort_keys=True))
                return
            raise ValueError("--visual-review is required for finalize")
        result = run_finalize(attempt, args.asset_root.resolve(), args.visual_review.resolve())
        print(json.dumps({
            "phase": "finalize",
            "status": result["status"],
            "primary": result["primary_failure_source"],
            "next_task": result["next_task"],
        }, sort_keys=True))


if __name__ == "__main__":
    main()
