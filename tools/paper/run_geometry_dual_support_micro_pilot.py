"""Inference-only geometry-disentangled dual-support micro-pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml
from gsplat import rasterization
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene.gaussian_clothing_residuals import (  # noqa: E402
    CHANNELS,
    GaussianClothingResiduals,
    compose_canonical_gaussian_overrides,
)
from tools import diagnose_image_conditioned_overfit_failure as diagnosis  # noqa: E402
from tools.check_real_image_conditioned_one_batch import save_render_tensor  # noqa: E402
from tools.paper import run_continuous_control_artifact_root_cause as previous  # noqa: E402
from tools.paper import run_continuous_control_causal_attribution as causal  # noqa: E402
from tools.paper import run_p0_color_spatial_soft_control_evaluations as sealed  # noqa: E402
from tools.paper.verify_seen_outfit_paper_assets import verify_manifest  # noqa: E402
from utils.dressable_camera_utils import build_mmlphuman_camera  # noqa: E402
from utils.mmlphuman_state_utils import mmlphuman_state_transaction  # noqa: E402


TASK_ID = "AAAI27-GEOMETRY-DUAL-SUPPORT-MICRO-PILOT-001"
SOURCE_HEAD = "8c43524b7ca0ee8b3c795dbe349466f30b29354f"
RUN_BRANCH = "research/geometry-dual-support-micro-pilot-20260722"
PROTOCOL_PATH = PROJECT_ROOT / "paper_protocol/reviewer_risk/geometry_dual_support_micro_pilot_protocol.yaml"
PROTOCOL_SHA256 = "559a367b69be867dab0075b480f44384a78aae7309b3374672a949773f08b552"
FROZEN_MANIFEST = PROJECT_ROOT / "paper_protocol/frozen_asset_manifest.json"
CAUSAL_REPORT = PROJECT_ROOT / "docs/PAPER/AAAI27_CONTINUOUS_CONTROL_CAUSAL_ATTRIBUTION_20260722.md"
CAUSAL_SUMMARY = PROJECT_ROOT / "paper_protocol/reviewer_risk/continuous_control_causal_attribution_final_summary.json"
CAUSAL_REVIEW = PROJECT_ROOT / "paper_protocol/reviewer_risk/continuous_control_causal_attribution_visual_review.json"
CAUSAL_PROTOCOL = PROJECT_ROOT / "paper_protocol/reviewer_risk/continuous_control_causal_attribution_protocol.yaml"
CAUSAL_OUTPUT = Path("/root/autodl-tmp/canondressgs_work/outputs/CONTINUOUS-CONTROL-CAUSAL-ATTRIBUTION-001")

OUTFITS = ("O01", "O02", "O03", "O04", "O08")
CONDITIONS = ("cond_000000", "cond_000318", "cond_000017", "cond_000347")
DIRECTIONS = ("A_TO_B", "B_TO_A")
ALPHAS = (0.20, 0.50, 0.80)
SELECTED_PAIRS = (("O01", "O02"), ("O01", "O03"), ("O01", "O08"))
STABLE_PAIRS = {"O01_O02"}
VARIANTS = ("FULL_LINEAR_BASELINE", "HARD_GEOMETRY_SOFT_VA", "DUAL_SUPPORT_GEOMETRY_BLEND")
NEW_VARIANTS = VARIANTS[1:]
GEOMETRY_CHANNELS = ("delta_xyz", "delta_log_scaling", "delta_rotvec")
VA_CHANNELS = ("delta_opacity_logit", "delta_sh0", "delta_shN")
VISUAL_CATEGORIES = (
    "patch",
    "cloud",
    "mottle",
    "edge_scatter",
    "silhouette_discontinuity",
    "full_body_contamination",
    "identity_contamination",
    "double_outline_ghosting",
)
CORE_CATEGORIES = VISUAL_CATEGORIES[:6]
ACTIVE_OPACITY_THRESHOLD = 1.0 / 255.0
LOGIT_EPSILON = 1.0e-6
PARITY_TOLERANCE = 1.0e-6


previous.TASK_ID = TASK_ID
previous.RUN_BRANCH = RUN_BRANCH
previous.SOURCE_HEAD = SOURCE_HEAD


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()


def sha256(path: Path, *, lf: bool = False) -> str:
    data = path.read_bytes()
    if lf:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any, *, replace: bool = False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(f"append-only JSON already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def write_new_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"append-only text already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8", newline="\n")


def alpha_id(alpha: float) -> str:
    return f"a{int(round(alpha * 1000)):03d}"


def direction_outfits(left: str, right: str, direction: str) -> tuple[str, str]:
    if direction == "A_TO_B":
        return left, right
    if direction == "B_TO_A":
        return right, left
    raise KeyError(direction)


def protocol() -> dict[str, Any]:
    if sha256(PROTOCOL_PATH, lf=True) != PROTOCOL_SHA256:
        raise RuntimeError("MICRO-PILOT-PROTOCOL-MISMATCH: fingerprint")
    value = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    selection = value["selection"]
    expected = [f"{left}_{right}" for left, right in SELECTED_PAIRS]
    if selection["selected_pairs"] != expected:
        raise RuntimeError("MICRO-PILOT-PROTOCOL-MISMATCH: pair selection")
    if tuple(value["design"]["directions"]) != DIRECTIONS:
        raise RuntimeError("MICRO-PILOT-PROTOCOL-MISMATCH: directions")
    if tuple(float(item) for item in value["design"]["alpha_values"]) != ALPHAS:
        raise RuntimeError("MICRO-PILOT-PROTOCOL-MISMATCH: alpha grid")
    if tuple(value["design"]["target_conditions"]) != CONDITIONS:
        raise RuntimeError("MICRO-PILOT-PROTOCOL-MISMATCH: conditions")
    thresholds = value["success_thresholds"]
    if thresholds["minimum_core_grade_drop"] != 1:
        raise RuntimeError("MICRO-PILOT-PROTOCOL-MISMATCH: grade threshold")
    if float(thresholds["minimum_severe_count_reduction_fraction"]) != 0.5:
        raise RuntimeError("MICRO-PILOT-PROTOCOL-MISMATCH: severe threshold")
    return value


def archive_fingerprints() -> dict[str, Any]:
    return {
        "causal_report_sha256": sha256(CAUSAL_REPORT),
        "causal_summary_sha256": sha256(CAUSAL_SUMMARY),
        "causal_review_sha256": sha256(CAUSAL_REVIEW),
        "causal_protocol_sha256": sha256(CAUSAL_PROTOCOL),
        "sealed_full_manifest_sha256": sha256(causal.SEALED_INTERPOLATION),
        "causal_output_tree": previous.tree_manifest(CAUSAL_OUTPUT),
    }


def frozen_trees(asset_root: Path) -> dict[str, Any]:
    return {
        "formal": previous.tree_manifest(asset_root / sealed.FORMAL_NAME),
        "p0": previous.tree_manifest(asset_root / sealed.P0_NAME),
        "sealed_evaluation": previous.tree_manifest(asset_root / sealed.OUTPUT_NAME),
        "causal_attribution": previous.tree_manifest(CAUSAL_OUTPUT),
    }


def validate_source(*, require_clean: bool = True) -> dict[str, Any]:
    if git("branch", "--show-current") != RUN_BRANCH:
        raise RuntimeError("MICRO-PILOT-GOVERNANCE-MISMATCH: branch")
    if require_clean and git("status", "--short"):
        raise RuntimeError("micro-pilot executor requires a clean worktree")
    if subprocess.call(["git", "merge-base", "--is-ancestor", SOURCE_HEAD, "HEAD"], cwd=PROJECT_ROOT):
        raise RuntimeError("MICRO-PILOT-GOVERNANCE-MISMATCH: source ancestry")
    frozen = protocol()
    causal_summary = read_json(CAUSAL_SUMMARY)
    if causal_summary["primary_failure_source"] != "GEOMETRY_MAIN_EFFECT":
        raise RuntimeError("MICRO-PILOT-GOVERNANCE-MISMATCH: causal primary")
    if causal_summary["secondary_failure_sources"] != [] or causal_summary["paper_final_count"] != 0:
        raise RuntimeError("MICRO-PILOT-GOVERNANCE-MISMATCH: causal archive")
    full, interpolation = causal.full_index()
    return {
        "protocol": frozen,
        "causal_summary": causal_summary,
        "full": full,
        "interpolation": interpolation,
        "archive_fingerprints": archive_fingerprints(),
    }


def run_preflight(output_root: Path, asset_root: Path) -> dict[str, Any]:
    attempt = output_root / "attempt_001"
    result_path = attempt / "audits/preflight.json"
    if result_path.is_file():
        return read_json(result_path)
    if output_root.exists():
        raise RuntimeError("append-only output root exists without a completed preflight")
    governance = validate_source()
    frozen_assets = verify_manifest(read_json(FROZEN_MANIFEST), PROJECT_ROOT, asset_root, verify_external=True)
    if frozen_assets["status"] != "PASS":
        raise RuntimeError("MICRO-PILOT-ASSET-MISMATCH: frozen assets")
    selected_full = {
        (f"{left}_{right}", condition, alpha)
        for left, right in SELECTED_PAIRS
        for condition in CONDITIONS
        for alpha in ALPHAS
    }
    if len(selected_full) != 36 or any(key not in governance["full"] for key in selected_full):
        raise RuntimeError("MICRO-PILOT-ASSET-MISMATCH: sealed FULL selection")
    smoke = torch.tensor([11.0, 13.0], device="cuda").sum()
    torch.cuda.synchronize()
    if float(smoke) != 24.0:
        raise RuntimeError("MICRO-PILOT-ASSET-MISMATCH: CUDA smoke")
    for name in ("renders", "metrics", "visuals", "audits", "aggregates"):
        (attempt / name).mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": "canondressgs.research.geometry_dual_support_preflight.v1",
        "status": "PASS",
        "label": "RESEARCH MICRO-PILOT — NOT PAPER FINAL",
        "task_id": TASK_ID,
        "source_head": SOURCE_HEAD,
        "execution_head": git("rev-parse", "HEAD"),
        "run_branch": RUN_BRANCH,
        "protocol_sha256_lf": PROTOCOL_SHA256,
        "selected_pairs": [f"{left}_{right}" for left, right in SELECTED_PAIRS],
        "selected_stable_pairs": sorted(STABLE_PAIRS),
        "selected_unstable_pairs": ["O01_O03", "O01_O08"],
        "full_reuse_logical_entries": 72,
        "full_reuse_unique_files": 36,
        "full_regenerated_files": 0,
        "expected_hard_renders": 72,
        "expected_dual_renders": 72,
        "archive_fingerprints_before": governance["archive_fingerprints"],
        "frozen_assets_before": frozen_assets,
        "frozen_trees_before": frozen_trees(asset_root),
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "paper_final": False,
    }
    atomic_json(result_path, result)
    return result


def endpoint_residuals(runtime_value: sealed.EvaluationRuntime) -> dict[str, GaussianClothingResiduals]:
    return {
        outfit: runtime_value.basis(runtime_value.coefficients[outfit], chunk_size=16384)
        for outfit in OUTFITS
    }


def hard_geometry_residual(
    source: GaussianClothingResiduals,
    target: GaussianClothingResiduals,
    earlier: GaussianClothingResiduals,
    alpha: float,
) -> GaussianClothingResiduals:
    if alpha < 0.5:
        geometry = source
    elif alpha > 0.5:
        geometry = target
    else:
        geometry = earlier
    values: dict[str, torch.Tensor] = {}
    for name in CHANNELS:
        if name in GEOMETRY_CHANNELS:
            values[name] = getattr(geometry, name)
        else:
            values[name] = (1.0 - alpha) * getattr(source, name) + alpha * getattr(target, name)
    return GaussianClothingResiduals.from_dict(values)


def hard_geometry_choice(
    source_outfit: str, target_outfit: str, earlier_outfit: str, alpha: float,
) -> str:
    if alpha < 0.5:
        return source_outfit
    if alpha > 0.5:
        return target_outfit
    return earlier_outfit


def stable_logit(probability: torch.Tensor) -> torch.Tensor:
    return torch.logit(probability.clamp(LOGIT_EPSILON, 1.0 - LOGIT_EPSILON))


def chw(value: torch.Tensor, channels: int) -> torch.Tensor:
    if value.ndim == 3 and value.shape[0] == channels:
        result = value
    elif value.ndim == 3 and value.shape[-1] == channels:
        result = value.permute(2, 0, 1)
    else:
        raise ValueError(f"unexpected render shape {tuple(value.shape)}")
    if not torch.isfinite(result).all():
        raise FloatingPointError("render contains NaN or Inf")
    return result.contiguous()


def branch_primitives(
    base: Any,
    camera: Mapping[str, Any],
    residual: GaussianClothingResiduals,
    weight: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    overrides = compose_canonical_gaussian_overrides(base, residual, CHANNELS).as_dict()
    means = base.compute_xyz(overrides)
    covariances = base.get_covariance(canonical_overrides=overrides)
    camera_position = torch.linalg.inv_ex(camera["w2c"])[0][:3, 3]
    colors = base.get_color(camera_position, overrides)
    endpoint_opacity = base.compute_opacity(overrides).reshape(-1)
    effective_opacity = endpoint_opacity * float(weight)
    legal_logit = stable_logit(effective_opacity)
    if not all(torch.isfinite(value).all() for value in (means, covariances, colors, endpoint_opacity, legal_logit)):
        raise FloatingPointError("dual-support branch contains NaN or Inf")
    tensors = {
        "means": means,
        "covariances": covariances,
        "colors": colors,
        "opacities": effective_opacity,
    }
    tensor_bytes = sum(value.numel() * value.element_size() for value in tensors.values())
    diagnostics = {
        "weight": float(weight),
        "gaussian_count": int(means.shape[0]),
        "active_gaussian_count": int((effective_opacity >= ACTIVE_OPACITY_THRESHOLD).sum()),
        "endpoint_opacity_min": float(endpoint_opacity.min()),
        "endpoint_opacity_max": float(endpoint_opacity.max()),
        "effective_opacity_min": float(effective_opacity.min()),
        "effective_opacity_max": float(effective_opacity.max()),
        "effective_opacity_mass": float(effective_opacity.double().sum()),
        "stable_inverse_sigmoid_min": float(legal_logit.min()),
        "stable_inverse_sigmoid_max": float(legal_logit.max()),
        "stable_inverse_sigmoid_finite": True,
        "primitive_tensor_bytes": int(tensor_bytes),
    }
    return tensors, diagnostics


def render_branches(
    runtime_value: sealed.EvaluationRuntime,
    left: str,
    condition: str,
    branches: Sequence[tuple[str, GaussianClothingResiduals, float]],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    base = runtime_value.context["base"]
    sample = runtime_value.context["samples"][f"{left}/{condition}"]
    background = runtime_value.context["background"]
    height = int(sample["target_camera"]["height"])
    width = int(sample["target_camera"]["width"])
    camera = build_mmlphuman_camera(sample["target_camera"], height, width, base._xyz.device)
    original_degree = int(base.sh_degree)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    branch_rows: list[dict[str, Any]] = []
    primitives: list[dict[str, torch.Tensor]] = []
    try:
        base.sh_degree = 1
        with mmlphuman_state_transaction(
            base,
            sample["target_pose"].to(base._xyz.device),
            sample["target_Rh"].to(base._xyz.device),
            sample["target_Th"].to(base._xyz.device),
        ):
            for outfit, residual, weight in branches:
                if weight == 0.0:
                    branch_rows.append({
                        "outfit": outfit,
                        "weight": 0.0,
                        "gaussian_count": int(base._xyz.shape[0]),
                        "active_gaussian_count": 0,
                        "effective_opacity_mass": 0.0,
                        "zero_weight_branch_pruned": True,
                        "stable_inverse_sigmoid_finite": True,
                    })
                    continue
                primitive, row = branch_primitives(base, camera, residual, weight)
                row.update({"outfit": outfit, "zero_weight_branch_pruned": False})
                primitives.append(primitive)
                branch_rows.append(row)
            if not primitives:
                raise RuntimeError("all dual-support branches have zero opacity weight")
            means = torch.cat([item["means"] for item in primitives], dim=0)
            covariances = torch.cat([item["covariances"] for item in primitives], dim=0)
            colors = torch.cat([item["colors"] for item in primitives], dim=0)
            opacities = torch.cat([item["opacities"] for item in primitives], dim=0)
            image, alpha, _ = rasterization(
                means=means,
                quats=None,
                scales=None,
                opacities=opacities,
                colors=colors,
                viewmats=camera["w2c"][None],
                Ks=camera["K"][None],
                width=width,
                height=height,
                packed=False,
                near_plane=0.1,
                backgrounds=background[None],
                covars=covariances,
            )
    finally:
        base.sh_degree = original_degree
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    rgb, rendered_alpha = chw(image[0], 3), chw(alpha[0], 1)
    diagnostics = {
        "render_time_seconds": float(elapsed),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "base_gaussian_count": int(base._xyz.shape[0]),
        "total_gaussian_count": int(sum(row["gaussian_count"] for row in branch_rows)),
        "rasterized_gaussian_count": int(sum(item["means"].shape[0] for item in primitives)),
        "active_gaussian_count": int(sum(row["active_gaussian_count"] for row in branch_rows)),
        "opacity_mass": float(sum(row["effective_opacity_mass"] for row in branch_rows)),
        "branch_primitive_tensor_bytes": int(sum(row.get("primitive_tensor_bytes", 0) for row in branch_rows)),
        "branches": branch_rows,
        "numerical_stability": "PASS",
    }
    return rgb, rendered_alpha, diagnostics


def render_paths(
    attempt: Path, variant: str, pair_id: str, direction: str, condition: str, alpha: float,
) -> tuple[Path, Path, Path]:
    root = attempt / "renders" / variant / pair_id / direction / condition
    stem = alpha_id(alpha)
    return root / f"{stem}_rgb.png", root / f"{stem}_alpha.png", root / f"{stem}_diagnostics.json"


def save_render(path: Path, value: torch.Tensor, channels: int) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_render_tensor(path, value.detach().cpu(), channels)


def render_or_load(
    paths: tuple[Path, Path, Path],
    render: Any,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], bool]:
    rgb_path, alpha_path, diagnostics_path = paths
    existing = [rgb_path.is_file(), alpha_path.is_file(), diagnostics_path.is_file()]
    if any(existing) and not all(existing):
        raise RuntimeError(f"partial render record: {paths}")
    if all(existing):
        return (
            sealed.image_tensor(rgb_path, 3).cuda(),
            sealed.image_tensor(alpha_path, 1).cuda(),
            read_json(diagnostics_path),
            False,
        )
    with torch.inference_mode():
        rgb, alpha, diagnostics = render()
    save_render(rgb_path, rgb, 3)
    save_render(alpha_path, alpha, 1)
    atomic_json(diagnostics_path, diagnostics)
    return rgb.detach(), alpha.detach(), diagnostics, True


def parity_paths(
    attempt: Path, variant: str, key: str, condition: str, endpoint_alpha: float,
) -> tuple[Path, Path, Path]:
    root = attempt / "renders/endpoint_parity" / variant / key / condition
    stem = alpha_id(endpoint_alpha)
    return root / f"{stem}_rgb.png", root / f"{stem}_alpha.png", root / f"{stem}_diagnostics.json"


def tensor_max_abs(first: torch.Tensor, second: torch.Tensor) -> float:
    return float((first - second).abs().max())


def run_endpoint_parity(
    runtime_value: sealed.EvaluationRuntime,
    attempt: Path,
    endpoints: Mapping[str, GaussianClothingResiduals],
) -> dict[str, Any]:
    result_path = attempt / "audits/endpoint_parity.json"
    if result_path.is_file():
        value = read_json(result_path)
        if value["status"] != "PASS":
            raise RuntimeError("MICRO-PILOT-ENDPOINT-PARITY-FAIL")
        return value
    records = []
    created = defaultdict(int)
    teacher_cache: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}
    maximum = 0.0
    for left, right in SELECTED_PAIRS:
        pair_id = f"{left}_{right}"
        for direction in DIRECTIONS:
            source_outfit, target_outfit = direction_outfits(left, right, direction)
            for endpoint_alpha, endpoint_outfit in ((0.0, source_outfit), (1.0, target_outfit)):
                for condition in CONDITIONS:
                    teacher_key = (endpoint_outfit, condition)
                    if teacher_key not in teacher_cache:
                        paths = parity_paths(attempt, "TEACHER", endpoint_outfit, condition, 0.0)
                        teacher_rgb, teacher_alpha, _, was_created = render_or_load(
                            paths,
                            lambda left=left, condition=condition, endpoint_outfit=endpoint_outfit: (
                                *sealed.p0._render(runtime_value.context, left, condition, endpoints[endpoint_outfit]),
                                {
                                    "render_time_seconds": 0.0,
                                    "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
                                    "endpoint_outfit": endpoint_outfit,
                                    "teacher_reference": True,
                                },
                            ),
                        )
                        created["TEACHER"] += int(was_created)
                        teacher_cache[teacher_key] = (teacher_rgb, teacher_alpha)
                    teacher_rgb, teacher_alpha = teacher_cache[teacher_key]
                    earlier = endpoints[left]
                    hard = hard_geometry_residual(
                        endpoints[source_outfit], endpoints[target_outfit], earlier, endpoint_alpha
                    )
                    hard_rgb, hard_alpha, hard_diag, hard_created = render_or_load(
                        parity_paths(attempt, "HARD_GEOMETRY_SOFT_VA", f"{pair_id}/{direction}", condition, endpoint_alpha),
                        lambda hard=hard, endpoint_outfit=endpoint_outfit: render_branches(
                            runtime_value, left, condition, ((endpoint_outfit, hard, 1.0),)
                        ),
                    )
                    dual_rgb, dual_alpha, dual_diag, dual_created = render_or_load(
                        parity_paths(attempt, "DUAL_SUPPORT_GEOMETRY_BLEND", f"{pair_id}/{direction}", condition, endpoint_alpha),
                        lambda endpoint_alpha=endpoint_alpha: render_branches(
                            runtime_value,
                            left,
                            condition,
                            (
                                (source_outfit, endpoints[source_outfit], 1.0 - endpoint_alpha),
                                (target_outfit, endpoints[target_outfit], endpoint_alpha),
                            ),
                        ),
                    )
                    created["HARD_GEOMETRY_SOFT_VA"] += int(hard_created)
                    created["DUAL_SUPPORT_GEOMETRY_BLEND"] += int(dual_created)
                    for variant, rgb, alpha, diagnostics in (
                        ("HARD_GEOMETRY_SOFT_VA", hard_rgb, hard_alpha, hard_diag),
                        ("DUAL_SUPPORT_GEOMETRY_BLEND", dual_rgb, dual_alpha, dual_diag),
                    ):
                        rgb_max = tensor_max_abs(rgb, teacher_rgb)
                        alpha_max = tensor_max_abs(alpha, teacher_alpha)
                        maximum = max(maximum, rgb_max, alpha_max)
                        row = {
                            "pair_id": pair_id,
                            "direction": direction,
                            "condition": condition,
                            "endpoint_alpha": endpoint_alpha,
                            "endpoint_outfit": endpoint_outfit,
                            "variant": variant,
                            "rgb_max_abs": rgb_max,
                            "alpha_max_abs": alpha_max,
                            "tolerance": PARITY_TOLERANCE,
                            "pass": max(rgb_max, alpha_max) <= PARITY_TOLERANCE,
                            "diagnostics": diagnostics,
                        }
                        records.append(row)
                        if not row["pass"]:
                            failure = {
                                "status": "MICRO-PILOT-ENDPOINT-PARITY-FAIL",
                                "failed_record": row,
                                "records_completed": len(records),
                                "paper_final": False,
                            }
                            atomic_json(attempt / "audits/MICRO-PILOT-ENDPOINT-PARITY-FAIL.json", failure)
                            raise RuntimeError("MICRO-PILOT-ENDPOINT-PARITY-FAIL")
    if len(records) != 96:
        raise RuntimeError("endpoint parity accounting mismatch")
    result = {
        "schema_version": "canondressgs.research.geometry_dual_support_endpoint_parity.v1",
        "status": "PASS",
        "logical_check_count": len(records),
        "check_count_by_variant": {variant: sum(row["variant"] == variant for row in records) for variant in NEW_VARIANTS},
        "unique_teacher_render_count": len(teacher_cache),
        "newly_created_render_count": dict(created),
        "maximum_render_max_abs": maximum,
        "tolerance": PARITY_TOLERANCE,
        "records": records,
        "paper_final": False,
    }
    atomic_json(result_path, result)
    return result


def residual_displacement(
    source: GaussianClothingResiduals,
    residuals: Sequence[GaussianClothingResiduals],
) -> dict[str, float]:
    values = torch.cat([
        (residual.delta_xyz - source.delta_xyz).detach().double().norm(dim=-1)
        for residual in residuals
    ])
    return {
        "displacement_rms": float(values.square().mean().sqrt()),
        "displacement_p95": float(torch.quantile(values, 0.95)),
        "displacement_max": float(values.max()),
    }


def duplicate_support_statistics(
    source: GaussianClothingResiduals, target: GaussianClothingResiduals,
) -> dict[str, float | int]:
    distance = (source.delta_xyz - target.delta_xyz).detach().double().norm(dim=-1)
    return {
        "corresponding_gaussian_count": int(distance.numel()),
        "canonical_geometry_distance_median": float(distance.median()),
        "canonical_geometry_distance_p95": float(torch.quantile(distance, 0.95)),
        "canonical_geometry_distance_max": float(distance.max()),
        "near_duplicate_fraction_at_1e_4": float((distance <= 1.0e-4).double().mean()),
    }


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    return float((value * mask).sum() / mask.sum().clamp_min(1.0))


def image_metrics(
    runtime_value: sealed.EvaluationRuntime,
    left: str,
    condition: str,
    rgb: torch.Tensor,
    alpha: torch.Tensor,
    reference: torch.Tensor,
    source_rgb: torch.Tensor,
) -> dict[str, float]:
    sample = runtime_value.context["samples"][f"{left}/{condition}"]
    garment = diagnosis._garment_mask(sample)
    protected = sample["target_protected_mask"]
    silhouette_iou, boundary_fscore, tolerance = sealed.silhouette_metrics(alpha, garment)
    outside = 1.0 - garment
    return {
        "garment_rgb_mae": causal.masked_mae(rgb, reference, garment),
        "garment_lpips": sealed.lpips_distance(runtime_value, rgb, reference, garment),
        "silhouette_iou": silhouette_iou,
        "boundary_fscore": boundary_fscore,
        "boundary_tolerance": tolerance,
        "protected_lpips": sealed.lpips_distance(runtime_value, rgb, source_rgb, protected),
        "identity_metric": causal.masked_mae(rgb, source_rgb, protected),
        "outside_garment_opacity": masked_mean(alpha, outside),
    }


def opacity_statistics(
    runtime_value: sealed.EvaluationRuntime,
    left: str,
    condition: str,
    residual: GaussianClothingResiduals,
) -> dict[str, Any]:
    base = runtime_value.context["base"]
    sample = runtime_value.context["samples"][f"{left}/{condition}"]
    overrides = compose_canonical_gaussian_overrides(base, residual, CHANNELS).as_dict()
    with mmlphuman_state_transaction(
        base,
        sample["target_pose"].to(base._xyz.device),
        sample["target_Rh"].to(base._xyz.device),
        sample["target_Th"].to(base._xyz.device),
    ):
        opacity = base.compute_opacity(overrides).reshape(-1)
    return {
        "base_gaussian_count": int(opacity.numel()),
        "total_gaussian_count": int(opacity.numel()),
        "rasterized_gaussian_count": int(opacity.numel()),
        "active_gaussian_count": int((opacity >= ACTIVE_OPACITY_THRESHOLD).sum()),
        "opacity_mass": float(opacity.double().sum()),
        "render_time_seconds": 0.0,
        "peak_vram_bytes": 0,
        "numerical_stability": "PASS",
    }


def save_mask(path: Path, mask: torch.Tensor) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    save_render_tensor(path, mask.detach().cpu(), 1)


def geometry_support_overlay(
    attempt: Path,
    runtime_value: sealed.EvaluationRuntime,
    endpoints: Mapping[str, GaussianClothingResiduals],
    left: str,
    right: str,
    direction: str,
) -> str:
    path = attempt / "visuals/geometry_support" / f"{left}_{right}" / f"{direction}.png"
    if path.is_file():
        return str(path)
    import matplotlib.pyplot as plt

    source_outfit, target_outfit = direction_outfits(left, right, direction)
    source, target, earlier = endpoints[source_outfit], endpoints[target_outfit], endpoints[left]
    base_xyz = runtime_value.context["base"]._xyz.detach()
    source_xyz = (base_xyz + source.delta_xyz).float().cpu()
    target_xyz = (base_xyz + target.delta_xyz).float().cpu()
    stride = max(1, source_xyz.shape[0] // 25000)
    sampled = torch.arange(0, source_xyz.shape[0], stride)
    figure, axes = plt.subplots(1, 6, figsize=(24, 4), constrained_layout=True)
    axes[0].scatter(source_xyz[sampled, 0], source_xyz[sampled, 2], s=0.3, c="#1f77b4")
    axes[0].set_title(f"source {source_outfit}")
    axes[1].scatter(target_xyz[sampled, 0], target_xyz[sampled, 2], s=0.3, c="#ff7f0e")
    axes[1].set_title(f"target {target_outfit}")
    for axis, alpha in zip(axes[2:5], ALPHAS):
        hard = hard_geometry_residual(source, target, earlier, alpha)
        xyz = (base_xyz + hard.delta_xyz).float().cpu()
        axis.scatter(xyz[sampled, 0], xyz[sampled, 2], s=0.3, c="#2ca02c")
        axis.set_title(f"hard {alpha:.2f}")
    axes[5].scatter(source_xyz[sampled, 0], source_xyz[sampled, 2], s=0.25, c="#1f77b4", alpha=0.55)
    axes[5].scatter(target_xyz[sampled, 0], target_xyz[sampled, 2], s=0.25, c="#ff7f0e", alpha=0.55)
    axes[5].set_title("dual immutable supports")
    for axis in axes:
        axis.set_aspect("equal")
        axis.set_xlabel("canonical x")
        axis.set_ylabel("canonical z")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return str(path)


def run_formal_renders(
    runtime_value: sealed.EvaluationRuntime,
    attempt: Path,
    full: Mapping[tuple[str, str, float], Mapping[str, Any]],
    endpoints: Mapping[str, GaussianClothingResiduals],
) -> dict[str, Any]:
    result_path = attempt / "aggregates/geometry_dual_support_execution.json"
    if result_path.is_file():
        return read_json(result_path)
    existing_before = {variant: 0 for variant in NEW_VARIANTS}
    for left, right in SELECTED_PAIRS:
        pair_id = f"{left}_{right}"
        for direction in DIRECTIONS:
            for condition in CONDITIONS:
                for alpha in ALPHAS:
                    for variant in NEW_VARIANTS:
                        presence = [path.is_file() for path in render_paths(attempt, variant, pair_id, direction, condition, alpha)]
                        if any(presence) and not all(presence):
                            raise RuntimeError("partial formal render")
                        existing_before[variant] += int(all(presence))
    records = []
    full_hashes: set[tuple[str, str]] = set()
    created = defaultdict(int)
    support_paths = []
    for left, right in SELECTED_PAIRS:
        pair_id = f"{left}_{right}"
        for direction in DIRECTIONS:
            source_outfit, target_outfit = direction_outfits(left, right, direction)
            source, target, earlier = endpoints[source_outfit], endpoints[target_outfit], endpoints[left]
            duplicate = duplicate_support_statistics(source, target)
            support_paths.append(geometry_support_overlay(attempt, runtime_value, endpoints, left, right, direction))
            for condition in CONDITIONS:
                garment = diagnosis._garment_mask(runtime_value.context["samples"][f"{left}/{condition}"])
                save_mask(attempt / "metrics/frozen_garment_masks" / left / f"{condition}.png", garment)
                source_row = full[(pair_id, condition, causal.endpoint_stored_alpha(direction))]
                target_row = full[(pair_id, condition, 1.0 - causal.endpoint_stored_alpha(direction))]
                source_rgb = sealed.image_tensor(Path(source_row["rgb_path"]), 3).to(runtime_value.device)
                target_rgb = sealed.image_tensor(Path(target_row["rgb_path"]), 3).to(runtime_value.device)
                for alpha in ALPHAS:
                    reference = (1.0 - alpha) * source_rgb + alpha * target_rgb
                    full_row = full[(pair_id, condition, causal.stored_alpha(direction, alpha))]
                    full_rgb_path, full_alpha_path = Path(full_row["rgb_path"]), Path(full_row["alpha_path"])
                    full_rgb = sealed.image_tensor(full_rgb_path, 3).to(runtime_value.device)
                    full_alpha = sealed.image_tensor(full_alpha_path, 1).to(runtime_value.device)
                    full_residual = causal.full_residual(runtime_value, left, right, direction, alpha)
                    full_diag = opacity_statistics(runtime_value, left, condition, full_residual)
                    full_record = {
                        "pair_id": pair_id,
                        "pair": [left, right],
                        "stable_label": pair_id in STABLE_PAIRS,
                        "direction": direction,
                        "source_outfit": source_outfit,
                        "target_outfit": target_outfit,
                        "render_target_outfit": left,
                        "condition": condition,
                        "alpha": alpha,
                        "variant": "FULL_LINEAR_BASELINE",
                        "rgb_path": str(full_rgb_path),
                        "alpha_path": str(full_alpha_path),
                        "source_rgb_path": source_row["rgb_path"],
                        "source_alpha_path": source_row["alpha_path"],
                        "target_rgb_path": target_row["rgb_path"],
                        "target_alpha_path": target_row["alpha_path"],
                        "rgb_sha256": sha256(full_rgb_path),
                        "alpha_sha256": sha256(full_alpha_path),
                        "reused": True,
                        **image_metrics(runtime_value, left, condition, full_rgb, full_alpha, reference, source_rgb),
                        **full_diag,
                        **residual_displacement(source, (full_residual,)),
                    }
                    records.append(full_record)
                    full_hashes.add((full_record["rgb_sha256"], full_record["alpha_sha256"]))
                    hard = hard_geometry_residual(source, target, earlier, alpha)
                    hard_rgb, hard_alpha, hard_diag, hard_created = render_or_load(
                        render_paths(attempt, "HARD_GEOMETRY_SOFT_VA", pair_id, direction, condition, alpha),
                        lambda hard=hard: render_branches(
                            runtime_value,
                            left,
                            condition,
                            ((hard_geometry_choice(source_outfit, target_outfit, left, alpha), hard, 1.0),),
                        ),
                    )
                    created["HARD_GEOMETRY_SOFT_VA"] += int(hard_created)
                    records.append({
                        "pair_id": pair_id,
                        "pair": [left, right],
                        "stable_label": pair_id in STABLE_PAIRS,
                        "direction": direction,
                        "source_outfit": source_outfit,
                        "target_outfit": target_outfit,
                        "render_target_outfit": left,
                        "condition": condition,
                        "alpha": alpha,
                        "variant": "HARD_GEOMETRY_SOFT_VA",
                        "geometry_endpoint": hard_geometry_choice(source_outfit, target_outfit, left, alpha),
                        "rgb_path": str(render_paths(attempt, "HARD_GEOMETRY_SOFT_VA", pair_id, direction, condition, alpha)[0]),
                        "alpha_path": str(render_paths(attempt, "HARD_GEOMETRY_SOFT_VA", pair_id, direction, condition, alpha)[1]),
                        "reused": not hard_created,
                        **image_metrics(runtime_value, left, condition, hard_rgb, hard_alpha, reference, source_rgb),
                        **hard_diag,
                        **residual_displacement(source, (hard,)),
                    })
                    dual_rgb, dual_alpha, dual_diag, dual_created = render_or_load(
                        render_paths(attempt, "DUAL_SUPPORT_GEOMETRY_BLEND", pair_id, direction, condition, alpha),
                        lambda: render_branches(
                            runtime_value,
                            left,
                            condition,
                            (
                                (source_outfit, source, 1.0 - alpha),
                                (target_outfit, target, alpha),
                            ),
                        ),
                    )
                    created["DUAL_SUPPORT_GEOMETRY_BLEND"] += int(dual_created)
                    records.append({
                        "pair_id": pair_id,
                        "pair": [left, right],
                        "stable_label": pair_id in STABLE_PAIRS,
                        "direction": direction,
                        "source_outfit": source_outfit,
                        "target_outfit": target_outfit,
                        "render_target_outfit": left,
                        "condition": condition,
                        "alpha": alpha,
                        "variant": "DUAL_SUPPORT_GEOMETRY_BLEND",
                        "branch_weights": {source_outfit: 1.0 - alpha, target_outfit: alpha},
                        "rgb_path": str(render_paths(attempt, "DUAL_SUPPORT_GEOMETRY_BLEND", pair_id, direction, condition, alpha)[0]),
                        "alpha_path": str(render_paths(attempt, "DUAL_SUPPORT_GEOMETRY_BLEND", pair_id, direction, condition, alpha)[1]),
                        "reused": not dual_created,
                        "duplicate_support": duplicate,
                        **image_metrics(runtime_value, left, condition, dual_rgb, dual_alpha, reference, source_rgb),
                        **dual_diag,
                        **residual_displacement(source, (source, target)),
                    })
    expected = len(SELECTED_PAIRS) * len(DIRECTIONS) * len(CONDITIONS) * len(ALPHAS)
    if len(records) != expected * len(VARIANTS):
        raise RuntimeError("formal metric matrix is incomplete")
    if len(full_hashes) != 36:
        raise RuntimeError("FULL unique reuse accounting mismatch")
    result = {
        "schema_version": "canondressgs.research.geometry_dual_support_execution.v1",
        "status": "AUTOMATIC_METRICS_COMPLETE_MANUAL_REVIEW_REQUIRED",
        "label": "RESEARCH MICRO-PILOT — NOT PAPER FINAL",
        "task_id": TASK_ID,
        "protocol_sha256_lf": PROTOCOL_SHA256,
        "selected_pairs": [f"{left}_{right}" for left, right in SELECTED_PAIRS],
        "alpha_grid": list(ALPHAS),
        "full_reuse_logical_entries": expected,
        "full_reuse_unique_files": len(full_hashes),
        "full_regenerated_files": 0,
        "hard_render_count": expected,
        "dual_render_count": expected,
        "existing_before": existing_before,
        "newly_created": dict(created),
        "records": records,
        "geometry_support_overlays": support_paths,
        "counts": {
            "training_steps": 0,
            "backward_calls": 0,
            "diagnostic_optimizer_created": 0,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "checkpoint_writes": 0,
        },
        "paper_final": False,
    }
    atomic_json(result_path, result)
    return result


def run_execute(output_root: Path, asset_root: Path) -> dict[str, Any]:
    attempt = output_root / "attempt_001"
    if not (attempt / "audits/preflight.json").is_file():
        raise RuntimeError("preflight is required before execute")
    governance = validate_source()
    with previous.NoTrainingProvenance(attempt, "execute") as provenance:
        runtime_value = previous.runtime(attempt, asset_root, protocol())
        provenance.capture_frozen_runtime_baseline()
        endpoints = endpoint_residuals(runtime_value)
        parity = run_endpoint_parity(runtime_value, attempt, endpoints)
        execution = run_formal_renders(runtime_value, attempt, governance["full"], endpoints)
    return {"parity": parity, "execution": execution}


def contact_sheet(path: Path, rows: Sequence[tuple[str, Sequence[tuple[str, Path]]]]) -> None:
    sealed.contact_sheet(path, rows)


def silhouette_overlay(rgb_path: Path, alpha_path: Path, mask_path: Path, output: Path) -> None:
    if output.is_file():
        return
    with Image.open(rgb_path) as opened:
        rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8).copy()
    with Image.open(alpha_path) as opened:
        alpha = np.asarray(opened.convert("L"), dtype=np.uint8) >= 128
    with Image.open(mask_path) as opened:
        target = np.asarray(opened.convert("L"), dtype=np.uint8) >= 128

    def boundary(mask: np.ndarray) -> np.ndarray:
        eroded = mask.copy()
        eroded[1:] &= mask[:-1]
        eroded[:-1] &= mask[1:]
        eroded[:, 1:] &= mask[:, :-1]
        eroded[:, :-1] &= mask[:, 1:]
        return mask & ~eroded

    rgb[boundary(alpha)] = np.array([255, 0, 0], dtype=np.uint8)
    rgb[boundary(target)] = np.array([0, 255, 0], dtype=np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".png.tmp")
    Image.fromarray(rgb).save(temporary, format="PNG")
    os.replace(temporary, output)


def aggregate_metrics(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "garment_rgb_mae", "garment_lpips", "silhouette_iou", "boundary_fscore",
        "protected_lpips", "identity_metric", "outside_garment_opacity",
        "render_time_seconds", "peak_vram_bytes", "active_gaussian_count",
        "opacity_mass", "displacement_rms", "displacement_p95",
    )
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(row["pair_id"], row["variant"])].append(row)
    result = []
    for (pair_id, variant), rows in sorted(grouped.items()):
        result.append({
            "pair_id": pair_id,
            "stable_label": pair_id in STABLE_PAIRS,
            "variant": variant,
            "record_count": len(rows),
            "means": {field: float(statistics.fmean(float(row[field]) for row in rows)) for field in fields},
            "medians": {field: float(statistics.median(float(row[field]) for row in rows)) for field in fields},
            "maxima": {field: max(float(row[field]) for row in rows) for field in fields},
        })
    return result


def run_analyze(output_root: Path) -> dict[str, Any]:
    attempt = output_root / "attempt_001"
    result_path = attempt / "aggregates/geometry_dual_support_analysis.json"
    if result_path.is_file():
        return read_json(result_path)
    validate_source()
    execution = read_json(attempt / "aggregates/geometry_dual_support_execution.json")
    records = execution["records"]
    index = {
        (row["pair_id"], row["direction"], float(row["alpha"]), row["condition"], row["variant"]): row
        for row in records
    }
    if len(index) != 216:
        raise RuntimeError("analysis record index mismatch")
    main_sheets, opacity_sheets, silhouette_sheets = [], [], []
    for left, right in SELECTED_PAIRS:
        pair_id = f"{left}_{right}"
        for direction in DIRECTIONS:
            main_rows, opacity_rows, silhouette_rows = [], [], []
            for condition in CONDITIONS:
                first = index[(pair_id, direction, ALPHAS[0], condition, "FULL_LINEAR_BASELINE")]
                main_panels: list[tuple[str, Path]] = [("Teacher source", Path(first["source_rgb_path"]))]
                opacity_panels: list[tuple[str, Path]] = [("Teacher source", Path(first["source_alpha_path"]))]
                silhouette_panels: list[tuple[str, Path]] = []
                mask_path = attempt / "metrics/frozen_garment_masks" / left / f"{condition}.png"
                for alpha in ALPHAS:
                    for variant in VARIANTS:
                        row = index[(pair_id, direction, alpha, condition, variant)]
                        label = f"{alpha:.2f} {variant.replace('_GEOMETRY_', '_G_').replace('_BASELINE', '')}"
                        main_panels.append((label, Path(row["rgb_path"])))
                        opacity_panels.append((label, Path(row["alpha_path"])))
                        component = (
                            attempt / "visuals/silhouette_components" / pair_id / direction / condition
                            / f"{alpha_id(alpha)}_{variant}.png"
                        )
                        silhouette_overlay(Path(row["rgb_path"]), Path(row["alpha_path"]), mask_path, component)
                        silhouette_panels.append((label, component))
                main_panels.append(("Teacher target", Path(first["target_rgb_path"])))
                opacity_panels.append(("Teacher target", Path(first["target_alpha_path"])))
                main_rows.append((condition, main_panels))
                opacity_rows.append((condition, opacity_panels))
                silhouette_rows.append((condition, silhouette_panels))
            main_path = attempt / "visuals/main" / pair_id / f"{direction}.png"
            opacity_path = attempt / "visuals/opacity" / pair_id / f"{direction}.png"
            silhouette_path = attempt / "visuals/silhouette" / pair_id / f"{direction}.png"
            contact_sheet(main_path, main_rows)
            contact_sheet(opacity_path, opacity_rows)
            contact_sheet(silhouette_path, silhouette_rows)
            main_sheets.append(str(main_path))
            opacity_sheets.append(str(opacity_path))
            silhouette_sheets.append(str(silhouette_path))
    support_sheets = execution["geometry_support_overlays"]
    manifest = {
        "schema_version": "canondressgs.research.geometry_dual_support_visual_manifest.v1",
        "status": "READY_FOR_MANUAL_REVIEW",
        "main_sheets": main_sheets,
        "opacity_diagnostic_sheets": opacity_sheets,
        "silhouette_overlay_sheets": silhouette_sheets,
        "geometry_support_overlay_sheets": support_sheets,
        "expected_actual_open_count": 24,
        "paper_final": False,
    }
    paths = main_sheets + opacity_sheets + silhouette_sheets + support_sheets
    if len(paths) != 24 or len(set(paths)) != 24 or any(not Path(path).is_file() for path in paths):
        raise RuntimeError("visual manifest accounting mismatch")
    atomic_json(attempt / "audits/visual_manifest.json", manifest)
    aggregates = aggregate_metrics(records)
    aggregate_index = {(row["pair_id"], row["variant"]): row for row in aggregates}
    comparisons = []
    for left, right in SELECTED_PAIRS:
        pair_id = f"{left}_{right}"
        baseline = aggregate_index[(pair_id, "FULL_LINEAR_BASELINE")]["means"]
        for variant in NEW_VARIANTS:
            candidate = aggregate_index[(pair_id, variant)]["means"]
            comparisons.append({
                "pair_id": pair_id,
                "variant": variant,
                "silhouette_iou_relative_degradation": max(
                    0.0, (baseline["silhouette_iou"] - candidate["silhouette_iou"]) / max(abs(baseline["silhouette_iou"]), 1.0e-12)
                ),
                "garment_lpips_relative_degradation": max(
                    0.0, (candidate["garment_lpips"] - baseline["garment_lpips"]) / max(abs(baseline["garment_lpips"]), 1.0e-12)
                ),
                "baseline": baseline,
                "candidate": candidate,
            })
    result = {
        "schema_version": "canondressgs.research.geometry_dual_support_analysis.v1",
        "status": "AUTOMATIC_ANALYSIS_COMPLETE_MANUAL_REVIEW_REQUIRED",
        "label": "RESEARCH MICRO-PILOT — NOT PAPER FINAL",
        "task_id": TASK_ID,
        "record_count": len(records),
        "aggregates": aggregates,
        "comparisons": comparisons,
        "visual_manifest": manifest,
        "paper_final": False,
    }
    atomic_json(result_path, result)
    return result


def validate_manual_review(review: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    expected = (
        manifest["main_sheets"]
        + manifest["opacity_diagnostic_sheets"]
        + manifest["silhouette_overlay_sheets"]
        + manifest["geometry_support_overlay_sheets"]
    )
    if review["status"] != "COMPLETE" or review["expected_count"] != 24:
        raise RuntimeError("manual review status mismatch")
    if review["actual_opened_count"] != 24 or len(review["items"]) != 24:
        raise RuntimeError("manual review count mismatch")
    if [item["source_path"] for item in review["items"]] != expected:
        raise RuntimeError("manual review path order mismatch")
    if not all(item["actual_opened"] for item in review["items"]):
        raise RuntimeError("manual review contains unopened paths")
    main_items = [item for item in review["items"] if item["kind"] == "MAIN_SHEET"]
    if len(main_items) != 6:
        raise RuntimeError("manual main-sheet count mismatch")
    expected_coordinates = {
        (variant, alpha, condition)
        for variant in VARIANTS
        for alpha in ALPHAS
        for condition in CONDITIONS
    }
    for item in main_items:
        grades = item["grades_by_variant_alpha_view"]
        if len(grades) != 36:
            raise RuntimeError("manual grade matrix is incomplete")
        if {(row["variant"], float(row["alpha"]), row["condition"]) for row in grades} != expected_coordinates:
            raise RuntimeError("manual grade coordinates mismatch")
        for row in grades:
            if set(row["grades"]) != set(VISUAL_CATEGORIES):
                raise RuntimeError("manual grade categories mismatch")
            if any(not isinstance(value, int) or not 0 <= value <= 3 for value in row["grades"].values()):
                raise RuntimeError("manual grade outside frozen scale")


def visual_grade_index(review: Mapping[str, Any]) -> dict[tuple[str, str, str, float, str], dict[str, int]]:
    result = {}
    for item in review["items"]:
        if item["kind"] != "MAIN_SHEET":
            continue
        for row in item["grades_by_variant_alpha_view"]:
            key = (item["pair_id"], item["direction"], row["variant"], float(row["alpha"]), row["condition"])
            result[key] = {name: int(value) for name, value in row["grades"].items()}
    if len(result) != 216:
        raise RuntimeError("visual grade index mismatch")
    return result


def classify(
    review: Mapping[str, Any], analysis: Mapping[str, Any], parity: Mapping[str, Any], frozen_unchanged: bool,
) -> dict[str, Any]:
    grades = visual_grade_index(review)
    pair_rows = []
    grade_pass_count = 0
    severe_pass_count = 0
    for left, right in SELECTED_PAIRS:
        pair_id = f"{left}_{right}"
        maximums = {}
        severe = {}
        for variant in ("FULL_LINEAR_BASELINE", "DUAL_SUPPORT_GEOMETRY_BLEND"):
            maximums[variant] = max(
                grades[(pair_id, direction, variant, 0.5, condition)][category]
                for direction in DIRECTIONS for condition in CONDITIONS for category in CORE_CATEGORIES
            )
            severe[variant] = sum(
                grades[(pair_id, direction, variant, alpha, condition)][category] == 3
                for direction in DIRECTIONS for alpha in ALPHAS for condition in CONDITIONS for category in CORE_CATEGORIES
            )
        drop = maximums["FULL_LINEAR_BASELINE"] - maximums["DUAL_SUPPORT_GEOMETRY_BLEND"]
        reduction = (
            (severe["FULL_LINEAR_BASELINE"] - severe["DUAL_SUPPORT_GEOMETRY_BLEND"])
            / max(severe["FULL_LINEAR_BASELINE"], 1)
        )
        grade_pass = drop >= 1
        severe_pass = reduction >= 0.5
        grade_pass_count += int(grade_pass)
        severe_pass_count += int(severe_pass)
        pair_rows.append({
            "pair_id": pair_id,
            "full_alpha_0_5_max_core_grade": maximums["FULL_LINEAR_BASELINE"],
            "dual_alpha_0_5_max_core_grade": maximums["DUAL_SUPPORT_GEOMETRY_BLEND"],
            "core_grade_drop": drop,
            "core_grade_drop_pass": grade_pass,
            "full_severe_core_count": severe["FULL_LINEAR_BASELINE"],
            "dual_severe_core_count": severe["DUAL_SUPPORT_GEOMETRY_BLEND"],
            "severe_reduction_fraction": reduction,
            "severe_reduction_pass": severe_pass,
        })
    identity_max = max(
        grades[(f"{left}_{right}", direction, "DUAL_SUPPORT_GEOMETRY_BLEND", alpha, condition)]["identity_contamination"]
        for left, right in SELECTED_PAIRS for direction in DIRECTIONS for alpha in ALPHAS for condition in CONDITIONS
    )
    ghosting_max = max(
        grades[(f"{left}_{right}", direction, "DUAL_SUPPORT_GEOMETRY_BLEND", alpha, condition)]["double_outline_ghosting"]
        for left, right in SELECTED_PAIRS for direction in DIRECTIONS for alpha in ALPHAS for condition in CONDITIONS
    )
    dual_comparisons = [row for row in analysis["comparisons"] if row["variant"] == "DUAL_SUPPORT_GEOMETRY_BLEND"]
    silhouette_pass = all(row["silhouette_iou_relative_degradation"] <= 0.05 for row in dual_comparisons)
    lpips_pass = all(row["garment_lpips_relative_degradation"] <= 0.10 for row in dual_comparisons)
    gates = {
        "endpoint_parity": parity["status"] == "PASS",
        "alpha_0_5_core_grade_reduction": grade_pass_count >= 2,
        "severe_count_reduction": severe_pass_count >= 2,
        "identity_contamination_zero": identity_max == 0,
        "silhouette_iou_bound": silhouette_pass,
        "garment_lpips_bound": lpips_pass,
        "no_severe_double_outline_ghosting": ghosting_max < 3,
        "frozen_unchanged": frozen_unchanged,
        "no_best_direction_or_view_selection": True,
    }
    safety = all(gates[key] for key in (
        "endpoint_parity", "identity_contamination_zero", "no_severe_double_outline_ghosting", "frozen_unchanged"
    ))
    efficacy = (gates["alpha_0_5_core_grade_reduction"], gates["severe_count_reduction"])
    if all(gates.values()):
        classification = "DUAL_SUPPORT_MICRO_PILOT_PASS"
    elif safety and any(efficacy):
        classification = "DUAL_SUPPORT_MICRO_PILOT_PARTIAL"
    else:
        classification = "DUAL_SUPPORT_MICRO_PILOT_FAIL"
    next_task = (
        "RUN_ALL_10_PAIR_DUAL_SUPPORT_EVALUATION"
        if classification != "DUAL_SUPPORT_MICRO_PILOT_FAIL"
        else "DESIGN_GEOMETRY_CORRESPONDENCE_WARP_DIAGNOSTIC"
    )
    return {
        "classification": classification,
        "gates": gates,
        "pair_artifact_comparisons": pair_rows,
        "grade_reduction_pair_count": grade_pass_count,
        "severe_reduction_pair_count": severe_pass_count,
        "identity_contamination_maximum_grade": identity_max,
        "double_outline_ghosting_maximum_grade": ghosting_max,
        "quantitative_comparisons": dual_comparisons,
        "next_task": next_task,
        "next_task_started": False,
    }


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def run_finalize(output_root: Path, asset_root: Path, review_path: Path) -> dict[str, Any]:
    attempt = output_root / "attempt_001"
    validate_source()
    preflight = read_json(attempt / "audits/preflight.json")
    execution = read_json(attempt / "aggregates/geometry_dual_support_execution.json")
    analysis = read_json(attempt / "aggregates/geometry_dual_support_analysis.json")
    parity = read_json(attempt / "audits/endpoint_parity.json")
    manifest = read_json(attempt / "audits/visual_manifest.json")
    review = read_json(review_path)
    validate_manual_review(review, manifest)
    assets_after = verify_manifest(read_json(FROZEN_MANIFEST), PROJECT_ROOT, asset_root, verify_external=True)
    trees_after = frozen_trees(asset_root)
    archives_after = archive_fingerprints()
    frozen_unchanged = (
        assets_after == preflight["frozen_assets_before"]
        and trees_after == preflight["frozen_trees_before"]
        and archives_after == preflight["archive_fingerprints_before"]
    )
    if not frozen_unchanged:
        raise RuntimeError("MICRO-PILOT-FROZEN-MUTATION")
    provenance = previous.aggregate_optimizer_provenance(attempt)
    decision = classify(review, analysis, parity, frozen_unchanged)
    counts = {
        "full_reuse_logical_entries": execution["full_reuse_logical_entries"],
        "full_reuse_unique_files": execution["full_reuse_unique_files"],
        "full_regenerated_files": execution["full_regenerated_files"],
        "hard_renders": execution["hard_render_count"],
        "dual_renders": execution["dual_render_count"],
        "endpoint_parity_checks": parity["logical_check_count"],
        "visual_review": review["actual_opened_count"],
        "training_steps": 0,
        "backward_calls": provenance["backward_count"],
        "diagnostic_optimizer_created": int(provenance["diagnostic_optimizer"]["created"]),
        "diagnostic_optimizer_steps": provenance["diagnostic_optimizer"]["step_count"],
        "legacy_optimizer_creation_count": provenance["legacy_context_optimizer"]["creation_count"],
        "legacy_optimizer_zero_grad": provenance["legacy_context_optimizer"]["zero_grad_count"],
        "legacy_optimizer_steps": provenance["legacy_context_optimizer"]["step_count"],
        "scheduler_steps": provenance["legacy_context_optimizer"]["scheduler_step_count"],
        "checkpoint_writes": provenance["checkpoint_write_count"],
        "teacher_mutation": 0,
        "basis_mutation": 0,
        "formal_output_mutation": 0,
        "paper_final": 0,
    }
    results = {
        "schema_version": "canondressgs.research.geometry_dual_support_micro_pilot_results.v1",
        "status": "COMPLETE",
        "label": "RESEARCH MICRO-PILOT — NOT PAPER FINAL",
        "task_id": TASK_ID,
        "protocol_sha256_lf": PROTOCOL_SHA256,
        "execution": execution,
        "analysis": analysis,
        "endpoint_parity": parity,
        "decision": decision,
        "optimizer_provenance": provenance,
        "paper_final": False,
    }
    summary = {
        "schema_version": "canondressgs.research.geometry_dual_support_micro_pilot_final_summary.v1",
        "status": "COMPLETE",
        "label": "RESEARCH MICRO-PILOT — NOT PAPER FINAL",
        "task_id": TASK_ID,
        "source_head": SOURCE_HEAD,
        "execution_head": git("rev-parse", "HEAD"),
        "run_branch": RUN_BRANCH,
        "protocol_sha256_lf": PROTOCOL_SHA256,
        "selected_stable_pairs": ["O01_O02"],
        "selected_unstable_pairs": ["O01_O03", "O01_O08"],
        "counts": counts,
        "endpoint_parity": {
            "status": parity["status"],
            "maximum_render_max_abs": parity["maximum_render_max_abs"],
            "tolerance": parity["tolerance"],
        },
        "decision": decision,
        "archive_fingerprints_before": preflight["archive_fingerprints_before"],
        "archive_fingerprints_after": archives_after,
        "frozen_assets_before": preflight["frozen_assets_before"],
        "frozen_assets_after": assets_after,
        "frozen_trees_before": preflight["frozen_trees_before"],
        "frozen_trees_after": trees_after,
        "frozen_unchanged": frozen_unchanged,
        "visual_review_path": "paper_protocol/reviewer_risk/geometry_dual_support_micro_pilot_visual_review.json",
        "next_task": decision["next_task"],
        "next_task_started": False,
        "paper_final": False,
        "paper_final_count": 0,
    }
    atomic_json(attempt / "aggregates/geometry_dual_support_micro_pilot_results.json", results)
    atomic_json(attempt / "aggregates/geometry_dual_support_micro_pilot_visual_review.json", review)
    atomic_json(attempt / "aggregates/geometry_dual_support_micro_pilot_final_summary.json", summary)
    atomic_json(PROJECT_ROOT / "paper_protocol/reviewer_risk/geometry_dual_support_micro_pilot_results.json", results)
    atomic_json(PROJECT_ROOT / "paper_protocol/reviewer_risk/geometry_dual_support_micro_pilot_visual_review.json", review)
    atomic_json(PROJECT_ROOT / "paper_protocol/reviewer_risk/geometry_dual_support_micro_pilot_final_summary.json", summary)
    pair_rows = decision["pair_artifact_comparisons"]
    quantitative = decision["quantitative_comparisons"]
    report = [
        "# AAAI27 Geometry-Disentangled Dual-Support Micro-Pilot",
        "",
        "**RESEARCH MICRO-PILOT — NOT PAPER FINAL**",
        "",
        f"- Source HEAD: `{SOURCE_HEAD}`.",
        f"- Protocol SHA-256: `{PROTOCOL_SHA256}`.",
        "- Selected stable pair: `O01_O02`; selected unstable pairs: `O01_O03`, `O01_O08`.",
        f"- FULL reused: {counts['full_reuse_logical_entries']} logical / {counts['full_reuse_unique_files']} unique / 0 regenerated.",
        f"- HARD renders: {counts['hard_renders']}; DUAL renders: {counts['dual_renders']}.",
        f"- Endpoint parity: `{parity['status']}`; maximum absolute difference `{parity['maximum_render_max_abs']}`.",
        f"- Manual visual review: {review['actual_opened_count']}/24.",
        f"- Final classification: **{decision['classification']}**.",
        f"- NEXT_TASK: **{decision['next_task']}** (not started).",
        "",
        "## Artifact-grade comparison",
        "",
        markdown_table(
            ["pair", "FULL max", "DUAL max", "drop", "FULL severe", "DUAL severe", "severe reduction"],
            [[row["pair_id"], row["full_alpha_0_5_max_core_grade"], row["dual_alpha_0_5_max_core_grade"], row["core_grade_drop"], row["full_severe_core_count"], row["dual_severe_core_count"], f"{row['severe_reduction_fraction']:.3f}"] for row in pair_rows],
        ),
        "",
        "## Quantitative comparison",
        "",
        markdown_table(
            ["pair", "silhouette degradation", "garment LPIPS degradation"],
            [[row["pair_id"], f"{row['silhouette_iou_relative_degradation']:.5f}", f"{row['garment_lpips_relative_degradation']:.5f}"] for row in quantitative],
        ),
        "",
        "## Governance",
        "",
        "No training, backward, diagnostic optimizer, optimizer step, scheduler step, checkpoint write, teacher mutation, basis mutation, formal-output mutation, pair selection, direction selection, view selection, or threshold adjustment occurred. The legacy context optimizer was construction-only, unused, unsaved, and discarded. PAPER_FINAL=0.",
        "",
        "The result is limited to the three preregistered seen garment pairs, two directions, three alpha values, and four frozen target views. It does not establish arbitrary garment interpolation, unseen garment generation, novel-pose behavior, novel-view behavior, or cross-identity generalization.",
    ]
    write_new_text(PROJECT_ROOT / "docs/PAPER/AAAI27_GEOMETRY_DUAL_SUPPORT_MICRO_PILOT_20260722.md", "\n".join(report))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("preflight", "execute", "analyze", "finalize", "all"), default="all")
    parser.add_argument("--visual-review", type=Path)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    asset_root = args.asset_root.resolve()
    if args.phase in {"preflight", "all"}:
        value = run_preflight(output_root, asset_root)
        print(json.dumps({"phase": "preflight", "status": value["status"]}, sort_keys=True))
        if args.phase == "preflight":
            return
    if args.phase in {"execute", "all"}:
        value = run_execute(output_root, asset_root)
        print(json.dumps({
            "phase": "execute",
            "endpoint_parity": value["parity"]["status"],
            "hard_renders": value["execution"]["hard_render_count"],
            "dual_renders": value["execution"]["dual_render_count"],
        }, sort_keys=True))
        if args.phase == "execute":
            return
    if args.phase in {"analyze", "all"}:
        value = run_analyze(output_root)
        print(json.dumps({"phase": "analyze", "status": value["status"], "visual_count": 24}, sort_keys=True))
        if args.phase == "analyze":
            return
    if args.phase in {"finalize", "all"}:
        if args.visual_review is None:
            if args.phase == "all":
                print(json.dumps({"phase": "finalize", "status": "MANUAL_REVIEW_REQUIRED"}, sort_keys=True))
                return
            raise ValueError("--visual-review is required for finalize")
        value = run_finalize(output_root, asset_root, args.visual_review.resolve())
        print(json.dumps({
            "phase": "finalize",
            "status": value["status"],
            "classification": value["decision"]["classification"],
            "next_task": value["next_task"],
        }, sort_keys=True))


if __name__ == "__main__":
    main()
