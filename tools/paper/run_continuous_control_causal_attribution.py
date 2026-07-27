"""Endpoint-anchored, inference-only causal attribution for continuous control.

The executor is append-only below its dedicated output root.  It reuses the
sealed endpoint and FULL interpolation files, renders only the six proper
non-empty factor subsets, records the repaired optimizer provenance, and never
trains or writes a checkpoint.
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml
from PIL import Image, ImageChops, ImageEnhance, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene.gaussian_clothing_residuals import CHANNELS, GaussianClothingResiduals
from tools import diagnose_image_conditioned_overfit_failure as diagnosis
from tools.paper import run_continuous_control_artifact_root_cause as previous
from tools.paper import run_p0_color_spatial_soft_control_evaluations as sealed
from tools.paper.verify_seen_outfit_paper_assets import verify_manifest


TASK_ID = "AAAI27-CONTINUOUS-CONTROL-CAUSAL-ATTRIBUTION-001"
REPAIR_TASK_ID = "AAAI27-CONTINUOUS-CONTROL-CAUSAL-ATTRIBUTION-ALPHA-REPAIR-001"
SOURCE_HEAD = "68623c36eee70c0b41aefb480f8f85e668eae201"
FAILED_HEAD = "0ebaa9503b4b9bd8b9e88e8c98ca2709584cae27"
RUN_BRANCH = "research/continuous-control-causal-attribution-20260722"
PROTOCOL_PATH = PROJECT_ROOT / "paper_protocol/reviewer_risk/continuous_control_causal_attribution_protocol.yaml"
PROTOCOL_SHA256 = "aa5cc2bc0f5deb1f9a4dacde26178fb1c322ed2fd314dfca308b00f6b98d21c4"
CORRECTION_PATH = PROJECT_ROOT / "paper_protocol/reviewer_risk/continuous_control_causal_attribution_protocol_correction.json"
FROZEN_MANIFEST = PROJECT_ROOT / "paper_protocol/frozen_asset_manifest.json"
SEALED_INTERPOLATION = PROJECT_ROOT / "paper_protocol/reviewer_risk/p0_basis_interpolation_results.json"
SEALED_MIXED = PROJECT_ROOT / "paper_protocol/reviewer_risk/p0_mixed_reference_results.json"
PREVIOUS_SUMMARY = PROJECT_ROOT / "paper_protocol/reviewer_risk/continuous_control_root_cause_final_summary.json"
PREVIOUS_REVIEW = PROJECT_ROOT / "paper_protocol/reviewer_risk/continuous_control_root_cause_visual_review.json"
PREVIOUS_PROTOCOL = PROJECT_ROOT / "paper_protocol/reviewer_risk/continuous_control_root_cause_protocol.yaml"
PREVIOUS_OUTPUT = Path("/root/autodl-tmp/canondressgs_work/outputs/CONTINUOUS-CONTROL-ROOT-CAUSE-001")

OUTFITS = ("O01", "O02", "O03", "O04", "O08")
PAIRS = tuple(itertools.combinations(OUTFITS, 2))
DIRECTIONS = ("A_TO_B", "B_TO_A")
CONDITIONS = ("cond_000000", "cond_000318", "cond_000017", "cond_000347")
ALPHAS = (0.20, 0.50, 0.80)
STABLE = {"O01_O02", "O01_O04", "O03_O04"}
FACTORS = {
    "G": ("delta_xyz", "delta_log_scaling", "delta_rotvec"),
    "V": ("delta_opacity_logit",),
    "A": ("delta_sh0", "delta_shN"),
}
SUBSETS = {
    "000": (),
    "100": ("G",),
    "010": ("V",),
    "001": ("A",),
    "110": ("G", "V"),
    "101": ("G", "A"),
    "011": ("V", "A"),
    "111": ("G", "V", "A"),
}
SUBSET_ORDER = tuple(SUBSETS)
NEW_SUBSETS = ("100", "010", "001", "110", "101", "011")
EFFECTS = ("G", "V", "A", "GxV", "GxA", "VxA", "GxVxA")
METRICS = (
    "garment_rgb_mae",
    "lpips",
    "silhouette_iou",
    "boundary_fscore",
    "protected_lpips",
    "displacement_rms",
    "displacement_p95",
    "opacity_activation_rms",
    "opacity_activation_p95",
    "sh0_difference_rms",
    "shN_difference_rms",
)
VISUAL_CATEGORIES = (
    "cloud",
    "mottle",
    "edge_scatter",
    "full_body_contamination",
    "identity_contamination",
    "silhouette_discontinuity",
    "patch_artifact",
)


# Reuse the already audited gate implementation under this task namespace.
previous.TASK_ID = TASK_ID
previous.RUN_BRANCH = RUN_BRANCH
previous.SOURCE_HEAD = SOURCE_HEAD


def sha256(path: Path, *, lf: bool = False) -> str:
    data = path.read_bytes()
    if lf:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any, *, replace: bool = False) -> None:
    previous.atomic_json(path, value, replace=replace)


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=PROJECT_ROOT, text=True).strip()


def protocol() -> dict[str, Any]:
    if sha256(PROTOCOL_PATH, lf=True) != PROTOCOL_SHA256:
        raise RuntimeError("CAUSAL-ATTRIBUTION-ALPHA-REPAIR-MISMATCH: protocol fingerprint")
    value = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    design = value["design"]
    if tuple(tuple(row) for row in design["unordered_pair_order"]) != PAIRS:
        raise RuntimeError("CAUSAL-ATTRIBUTION-ALPHA-REPAIR-MISMATCH: pair order")
    if tuple(design["direction_order"]) != DIRECTIONS:
        raise RuntimeError("CAUSAL-ATTRIBUTION-ALPHA-REPAIR-MISMATCH: direction order")
    if tuple(design["target_view_order"]) != CONDITIONS:
        raise RuntimeError("CAUSAL-ATTRIBUTION-ALPHA-REPAIR-MISMATCH: view order")
    if tuple(float(value) for value in design["alpha"]) != ALPHAS:
        raise RuntimeError("CAUSAL-ATTRIBUTION-ALPHA-REPAIR-MISMATCH: alpha grid")
    labels = design["stable_pair_labels"]
    if Counter(labels.values()) != Counter({"stable": 3, "unstable": 7}):
        raise RuntimeError("CAUSAL-ATTRIBUTION-ALPHA-REPAIR-MISMATCH: 3/7 labels")
    if {key for key, label in labels.items() if label == "stable"} != STABLE:
        raise RuntimeError("CAUSAL-ATTRIBUTION-ALPHA-REPAIR-MISMATCH: stable identities")
    factors = value["factor_contract"]["factors"]
    if factors != {key: list(channels) for key, channels in FACTORS.items()}:
        raise RuntimeError("CAUSAL-ATTRIBUTION-ALPHA-REPAIR-MISMATCH: factor groups")
    if value["factor_contract"]["subsets"] != {key: list(items) for key, items in SUBSETS.items()}:
        raise RuntimeError("CAUSAL-ATTRIBUTION-ALPHA-REPAIR-MISMATCH: subsets")
    return value


def full_index() -> tuple[dict[tuple[str, str, float], dict[str, Any]], dict[str, Any]]:
    result = read_json(SEALED_INTERPOLATION)
    if result["render_count"] != 440 or result["expected_render_count"] != 440:
        raise RuntimeError("CAUSAL-ATTRIBUTION-ALPHA-REPAIR-MISMATCH: sealed FULL count")
    if not result["endpoint_parity_pass"] or len(result["records"]) != 440:
        raise RuntimeError("CAUSAL-ATTRIBUTION-ALPHA-REPAIR-MISMATCH: endpoint parity")
    index = {
        (row["pair_id"], row["view_id"], round(float(row["alpha"]), 2)): row
        for row in result["records"]
    }
    if len(index) != 440:
        raise RuntimeError("CAUSAL-ATTRIBUTION-ALPHA-REPAIR-MISMATCH: duplicate FULL row")
    selected = [row for row in result["records"] if round(float(row["alpha"]), 2) in ALPHAS]
    if len(selected) != 120:
        raise RuntimeError("CAUSAL-ATTRIBUTION-ALPHA-REPAIR-MISMATCH: 120 unique FULL entries")
    missing = [
        path
        for row in selected
        for path in (Path(row["rgb_path"]), Path(row["alpha_path"]))
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(f"CAUSAL-ATTRIBUTION-ALPHA-REPAIR-MISMATCH: missing FULL files {missing[:3]}")
    return index, result


def previous_archive_fingerprints() -> dict[str, Any]:
    return {
        "root_cause_summary_sha256": sha256(PREVIOUS_SUMMARY),
        "root_cause_visual_review_sha256": sha256(PREVIOUS_REVIEW),
        "root_cause_protocol_sha256": sha256(PREVIOUS_PROTOCOL),
        "sealed_full_manifest_sha256": sha256(SEALED_INTERPOLATION),
        "previous_root_cause_output_tree": previous.tree_manifest(PREVIOUS_OUTPUT),
    }


def validate_governance(output_root: Path, *, require_clean: bool = True) -> dict[str, Any]:
    if git("branch", "--show-current") != RUN_BRANCH:
        raise RuntimeError("CAUSAL-ATTRIBUTION-ALPHA-REPAIR-MISMATCH: branch")
    if require_clean and git("status", "--short"):
        raise RuntimeError("causal-attribution executor requires a clean worktree")
    if subprocess.call(["git", "merge-base", "--is-ancestor", SOURCE_HEAD, "HEAD"], cwd=PROJECT_ROOT):
        raise RuntimeError("CAUSAL-ATTRIBUTION-ALPHA-REPAIR-MISMATCH: source ancestry")
    frozen = protocol()
    correction = read_json(CORRECTION_PATH)
    if correction["original_alpha_values"] != [0.25, 0.5, 0.75]:
        raise RuntimeError("CAUSAL-ATTRIBUTION-ALPHA-REPAIR-MISMATCH: original alpha audit")
    if correction["corrected_alpha_values"] != [0.2, 0.5, 0.8]:
        raise RuntimeError("CAUSAL-ATTRIBUTION-ALPHA-REPAIR-MISMATCH: corrected alpha audit")
    if correction["new_protocol_sha256_lf"] != PROTOCOL_SHA256:
        raise RuntimeError("CAUSAL-ATTRIBUTION-ALPHA-REPAIR-MISMATCH: correction SHA")
    gates = frozen["previous_archive_gate"]
    for path, expected in (
        (PREVIOUS_SUMMARY, gates["final_summary"]["sha256_lf"]),
        (PREVIOUS_REVIEW, gates["visual_review"]["sha256_lf"]),
        (PREVIOUS_PROTOCOL, gates["previous_protocol"]["sha256_lf"]),
    ):
        if sha256(path, lf=True) != expected:
            raise RuntimeError(f"CAUSAL-ATTRIBUTION-ALPHA-REPAIR-MISMATCH: previous archive {path.name}")
    summary = read_json(PREVIOUS_SUMMARY)
    review = read_json(PREVIOUS_REVIEW)
    if summary["paper_final_count"] != 0 or summary["counts"]["channel_renders"] != 720:
        raise RuntimeError("CAUSAL-ATTRIBUTION-ALPHA-REPAIR-MISMATCH: previous summary")
    if review["actual_opened_count"] != 82:
        raise RuntimeError("CAUSAL-ATTRIBUTION-ALPHA-REPAIR-MISMATCH: previous review")
    failure_path = output_root / "attempt_001/audits/FAILED_PRE_RESULT_ALPHA_GRID_ASSET_MISMATCH.json"
    if not failure_path.is_file():
        raise RuntimeError("CAUSAL-ATTRIBUTION-ALPHA-REPAIR-MISMATCH: failed attempt missing")
    failure = read_json(failure_path)
    if failure["status"] != "FAILED_PRE_RESULT_ALPHA_GRID_ASSET_MISMATCH":
        raise RuntimeError("CAUSAL-ATTRIBUTION-ALPHA-REPAIR-MISMATCH: failed attempt status")
    if any(int(value) for value in failure["counts"].values()):
        raise RuntimeError("CAUSAL-ATTRIBUTION-ALPHA-REPAIR-MISMATCH: failed attempt nonzero count")
    current_fingerprints = previous_archive_fingerprints()
    if current_fingerprints != failure["previous_archive_fingerprints"]:
        raise RuntimeError("CAUSAL-ATTRIBUTION-ALPHA-REPAIR-MISMATCH: previous archive mutation")
    full, interpolation = full_index()
    return {
        "protocol": frozen,
        "correction": correction,
        "failure": failure,
        "previous_summary": summary,
        "previous_review": review,
        "previous_archive_fingerprints": current_fingerprints,
        "full": full,
        "interpolation": interpolation,
    }


def run_preflight(attempt: Path, output_root: Path, asset_root: Path) -> dict[str, Any]:
    result_path = attempt / "audits/preflight.json"
    if result_path.is_file():
        return read_json(result_path)
    if attempt.exists():
        raise RuntimeError("append-only attempt_002 already exists without preflight")
    governance = validate_governance(output_root)
    manifest = read_json(FROZEN_MANIFEST)
    assets = verify_manifest(manifest, PROJECT_ROOT, asset_root, verify_external=True)
    if assets["status"] != "PASS" or assets["asset_count"] != 19:
        raise RuntimeError("CAUSAL-ATTRIBUTION-ASSET-MISMATCH: frozen assets")
    smoke = torch.tensor([8.0, 9.0], device="cuda").sum()
    torch.cuda.synchronize()
    if float(smoke) != 17.0:
        raise RuntimeError("CAUSAL-ATTRIBUTION-ASSET-MISMATCH: CUDA tensor smoke")
    attempt.mkdir(parents=True)
    for name in ("factor_injection", "factorial_metrics", "visuals", "audits", "aggregates"):
        (attempt / name).mkdir(parents=True, exist_ok=True)
    full_unique = {
        (row["pair_id"], row["view_id"], round(float(row["alpha"]), 2))
        for row in governance["interpolation"]["records"]
        if round(float(row["alpha"]), 2) in ALPHAS
    }
    result = {
        "schema_version": "canondressgs.research.continuous_control_causal_attribution_preflight.v1",
        "status": "PASS",
        "label": "RESEARCH DIAGNOSTIC — NOT PAPER FINAL",
        "task_id": TASK_ID,
        "repair_task_id": REPAIR_TASK_ID,
        "run_branch": RUN_BRANCH,
        "execution_head": git("rev-parse", "HEAD"),
        "source_head": SOURCE_HEAD,
        "protocol_sha256_lf": PROTOCOL_SHA256,
        "previous_failed_attempt": governance["failure"],
        "previous_archive_fingerprints": governance["previous_archive_fingerprints"],
        "alpha_grid": list(ALPHAS),
        "full_reuse_logical_entries": 240,
        "full_reuse_unique_files": len(full_unique),
        "full_regenerated_files": 0,
        "endpoint_reuse_logical_entries": 80,
        "new_factor_render_count": 1440,
        "pair_count": 10,
        "direction_count": 2,
        "view_count": 4,
        "stable_count": 3,
        "unstable_count": 7,
        "frozen_asset_verification": assets,
        "frozen_trees_before": {
            "formal": previous.tree_manifest(asset_root / sealed.FORMAL_NAME),
            "p0": previous.tree_manifest(asset_root / sealed.P0_NAME),
            "sealed_evaluation": previous.tree_manifest(asset_root / sealed.OUTPUT_NAME),
            "previous_root_cause": previous.tree_manifest(PREVIOUS_OUTPUT),
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "cuda_tensor_smoke": float(smoke),
            "cuda": torch.version.cuda,
            "pytorch": torch.__version__,
            "python": platform.python_version(),
        },
        "counts": {
            "training_steps": 0,
            "backward_calls": 0,
            "diagnostic_optimizer_created": 0,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "checkpoint_writes": 0,
            "paper_final": 0,
        },
        "paper_final": False,
    }
    atomic_json(result_path, result)
    return result


def alpha_id(alpha: float) -> str:
    return f"a{int(round(alpha * 1000)):03d}"


def stored_alpha(direction: str, logical_alpha: float) -> float:
    if direction == "A_TO_B":
        return round(logical_alpha, 2)
    if direction == "B_TO_A":
        return round(1.0 - logical_alpha, 2)
    raise ValueError(direction)


def endpoint_stored_alpha(direction: str) -> float:
    return 0.0 if direction == "A_TO_B" else 1.0


def direction_outfits(left: str, right: str, direction: str) -> tuple[str, str]:
    if direction == "A_TO_B":
        return left, right
    if direction == "B_TO_A":
        return right, left
    raise ValueError(direction)


def subset_channels(subset: str) -> tuple[str, ...]:
    return tuple(channel for factor in SUBSETS[subset] for channel in FACTORS[factor])


def inject_residual(
    source: GaussianClothingResiduals,
    target: GaussianClothingResiduals,
    subset: str,
    alpha: float,
) -> GaussianClothingResiduals:
    selected = set(subset_channels(subset))
    values: dict[str, torch.Tensor] = {}
    for name in CHANNELS:
        source_value = getattr(source, name)
        if name in selected:
            target_value = getattr(target, name)
            values[name] = source_value + alpha * (target_value - source_value)
        else:
            values[name] = source_value
    return GaussianClothingResiduals.from_dict(values)


def residual_bitwise_equal(first: Any, second: Any) -> bool:
    return all(torch.equal(getattr(first, name), getattr(second, name)) for name in CHANNELS)


def full_residual(
    runtime_value: sealed.EvaluationRuntime,
    left: str,
    right: str,
    direction: str,
    logical_alpha: float,
) -> GaussianClothingResiduals:
    mapped = stored_alpha(direction, logical_alpha)
    coefficient = (1.0 - mapped) * runtime_value.coefficients[left] + mapped * runtime_value.coefficients[right]
    return runtime_value.basis(coefficient, chunk_size=16384)


def render_paths(
    attempt: Path,
    subset: str,
    pair_id: str,
    direction: str,
    condition: str,
    alpha: float,
) -> tuple[Path, Path]:
    root = attempt / "factor_injection" / subset / pair_id / direction / condition
    stem = alpha_id(alpha)
    return root / f"{stem}_rgb.png", root / f"{stem}_alpha.png"


def expected_new_items(attempt: Path) -> list[dict[str, Any]]:
    return [
        {
            "subset": subset,
            "pair": [left, right],
            "pair_id": f"{left}_{right}",
            "direction": direction,
            "view_id": condition,
            "alpha": alpha,
            "rgb_path": str(render_paths(attempt, subset, f"{left}_{right}", direction, condition, alpha)[0]),
            "alpha_path": str(render_paths(attempt, subset, f"{left}_{right}", direction, condition, alpha)[1]),
        }
        for subset in NEW_SUBSETS
        for left, right in PAIRS
        for direction in DIRECTIONS
        for alpha in ALPHAS
        for condition in CONDITIONS
    ]


def masked_mae(first: torch.Tensor, second: torch.Tensor, mask: torch.Tensor) -> float:
    denominator = mask.sum().clamp_min(1.0) * first.shape[0]
    return float(((first - second).abs() * mask).sum() / denominator)


def residual_statistics(
    runtime_value: sealed.EvaluationRuntime,
    source: GaussianClothingResiduals,
    residual: GaussianClothingResiduals,
) -> dict[str, float]:
    xyz = (residual.delta_xyz - source.delta_xyz).detach().double().norm(dim=-1)
    base_opacity = runtime_value.context["base"]._opacity.detach().double()
    source_opacity = torch.sigmoid(base_opacity + source.delta_opacity_logit.detach().double())
    result_opacity = torch.sigmoid(base_opacity + residual.delta_opacity_logit.detach().double())
    opacity = (result_opacity - source_opacity).abs().reshape(-1)
    sh0 = (residual.delta_sh0 - source.delta_sh0).detach().double().reshape(-1)
    shn = (residual.delta_shN - source.delta_shN).detach().double().reshape(-1)
    return {
        "displacement_rms": float(xyz.square().mean().sqrt()),
        "displacement_p95": float(torch.quantile(xyz, 0.95)),
        "opacity_activation_rms": float(opacity.square().mean().sqrt()),
        "opacity_activation_p95": float(torch.quantile(opacity, 0.95)),
        "sh0_difference_rms": float(sh0.square().mean().sqrt()),
        "shN_difference_rms": float(shn.square().mean().sqrt()),
    }


def image_metrics(
    runtime_value: sealed.EvaluationRuntime,
    left: str,
    condition: str,
    rgb: torch.Tensor,
    rendered_alpha: torch.Tensor,
    reference: torch.Tensor,
    source_rgb: torch.Tensor,
) -> dict[str, float]:
    sample = runtime_value.context["samples"][f"{left}/{condition}"]
    garment = diagnosis._garment_mask(sample)
    protected = sample["target_protected_mask"]
    silhouette_iou, boundary_fscore, tolerance = sealed.silhouette_metrics(rendered_alpha, garment)
    return {
        "garment_rgb_mae": masked_mae(rgb, reference, garment),
        "lpips": sealed.lpips_distance(runtime_value, rgb, reference, garment),
        "silhouette_iou": silhouette_iou,
        "boundary_fscore": boundary_fscore,
        "boundary_tolerance": tolerance,
        "protected_lpips": sealed.lpips_distance(runtime_value, rgb, source_rgb, protected),
    }


def implementation_parity(
    runtime_value: sealed.EvaluationRuntime,
    full: Mapping[tuple[str, str, float], Mapping[str, Any]],
) -> dict[str, Any]:
    selected_checks = 0
    unselected_checks = 0
    empty_checks = 0
    full_checks = 0
    reverse_checks = 0
    maximum_full_formula_error = 0.0
    endpoints = {
        outfit: runtime_value.basis(runtime_value.coefficients[outfit], chunk_size=16384)
        for outfit in OUTFITS
    }
    for left, right in PAIRS:
        pair_id = f"{left}_{right}"
        for direction in DIRECTIONS:
            source_outfit, target_outfit = direction_outfits(left, right, direction)
            source, target = endpoints[source_outfit], endpoints[target_outfit]
            empty = inject_residual(source, target, "000", ALPHAS[0])
            if not residual_bitwise_equal(empty, source):
                raise RuntimeError("FACTOR-INJECTION-IMPLEMENTATION-MISMATCH: empty subset")
            empty_checks += 1
            for alpha in ALPHAS:
                for subset in NEW_SUBSETS:
                    residual = inject_residual(source, target, subset, alpha)
                    selected = set(subset_channels(subset))
                    for name in CHANNELS:
                        actual = getattr(residual, name)
                        if name in selected:
                            expected = getattr(source, name) + alpha * (getattr(target, name) - getattr(source, name))
                            if not torch.equal(actual, expected):
                                raise RuntimeError("FACTOR-INJECTION-IMPLEMENTATION-MISMATCH: selected channel")
                            selected_checks += 1
                        else:
                            if not torch.equal(actual, getattr(source, name)):
                                raise RuntimeError("FACTOR-INJECTION-IMPLEMENTATION-MISMATCH: unselected channel")
                            unselected_checks += 1
                sealed_full = full_residual(runtime_value, left, right, direction, alpha)
                mapped = stored_alpha(direction, alpha)
                direct_full = inject_residual(source, target, "111", alpha)
                maximum_full_formula_error = max(
                    maximum_full_formula_error,
                    max(float((getattr(direct_full, name) - getattr(sealed_full, name)).abs().max()) for name in CHANNELS),
                )
                repeated = full_residual(runtime_value, left, right, direction, alpha)
                if not residual_bitwise_equal(sealed_full, repeated):
                    raise RuntimeError("FACTOR-INJECTION-IMPLEMENTATION-MISMATCH: FULL residual reconstruction")
                for condition in CONDITIONS:
                    if (pair_id, condition, mapped) not in full:
                        raise RuntimeError("CAUSAL-ATTRIBUTION-ALPHA-REPAIR-MISMATCH: FULL mapping")
                full_checks += 1
                reverse = full_residual(runtime_value, left, right, "B_TO_A", alpha)
                forward = full_residual(runtime_value, left, right, "A_TO_B", round(1.0 - alpha, 2))
                if not residual_bitwise_equal(reverse, forward):
                    raise RuntimeError("FACTOR-INJECTION-IMPLEMENTATION-MISMATCH: reverse FULL residual")
                reverse_checks += 1
    return {
        "schema_version": "canondressgs.research.factor_injection_implementation_parity.v1",
        "status": "PASS",
        "alpha_grid": list(ALPHAS),
        "selected_channel_bitwise_checks": selected_checks,
        "unselected_channel_bitwise_checks": unselected_checks,
        "empty_subset_bitwise_checks": empty_checks,
        "full_basis_reconstruction_bitwise_checks": full_checks,
        "reverse_direction_bitwise_checks": reverse_checks,
        "maximum_endpoint_formula_vs_sealed_basis_absolute_error": maximum_full_formula_error,
        "full_manifest_mapping_checks": 10 * 2 * 3 * 4,
        "target_pose_camera_rule": "FROZEN_LEFT_PAIR_TARGET_FOR_BOTH_RESIDUAL_DIRECTIONS",
        "paper_final": False,
    }


def logical_image_path(
    attempt: Path,
    full: Mapping[tuple[str, str, float], Mapping[str, Any]],
    pair_id: str,
    direction: str,
    condition: str,
    subset: str,
    alpha: float,
) -> tuple[Path, Path]:
    if subset == "000":
        row = full[(pair_id, condition, endpoint_stored_alpha(direction))]
        return Path(row["rgb_path"]), Path(row["alpha_path"])
    if subset == "111":
        row = full[(pair_id, condition, stored_alpha(direction, alpha))]
        return Path(row["rgb_path"]), Path(row["alpha_path"])
    return render_paths(attempt, subset, pair_id, direction, condition, alpha)


def make_contact_sheets(
    attempt: Path,
    full: Mapping[tuple[str, str, float], Mapping[str, Any]],
) -> dict[str, list[str]]:
    pair_direction: list[str] = []
    pair_summary: list[str] = []
    for left, right in PAIRS:
        pair_id = f"{left}_{right}"
        for direction in DIRECTIONS:
            rows = []
            for condition in CONDITIONS:
                panels = []
                for subset in SUBSET_ORDER:
                    for alpha in ALPHAS:
                        rgb_path, _ = logical_image_path(attempt, full, pair_id, direction, condition, subset, alpha)
                        panels.append((f"{subset}@{alpha:.2f}", rgb_path))
                rows.append((condition, panels))
            sheet = attempt / "visuals/pair_direction" / pair_id / f"{direction}.png"
            sealed.contact_sheet(sheet, rows)
            pair_direction.append(str(sheet))
        summary_rows = []
        for direction in DIRECTIONS:
            for condition in CONDITIONS:
                panels = []
                for subset in SUBSET_ORDER:
                    rgb_path, _ = logical_image_path(attempt, full, pair_id, direction, condition, subset, 0.5)
                    panels.append((subset, rgb_path))
                summary_rows.append((f"{direction}/{condition}", panels))
        sheet = attempt / "visuals/pair_summary" / f"{pair_id}.png"
        sealed.contact_sheet(sheet, summary_rows)
        pair_summary.append(str(sheet))
    stable_unstable: list[str] = []
    for label, pairs in (
        ("stable", [pair for pair in PAIRS if f"{pair[0]}_{pair[1]}" in STABLE]),
        ("unstable", [pair for pair in PAIRS if f"{pair[0]}_{pair[1]}" not in STABLE]),
    ):
        rows = []
        for left, right in pairs:
            pair_id = f"{left}_{right}"
            panels = []
            for direction in DIRECTIONS:
                for condition in CONDITIONS:
                    rgb_path, _ = logical_image_path(attempt, full, pair_id, direction, condition, "111", 0.5)
                    panels.append((f"{direction}/{condition}", rgb_path))
            rows.append((pair_id, panels))
        sheet = attempt / "visuals/stable_unstable" / f"{label}.png"
        sealed.contact_sheet(sheet, rows)
        stable_unstable.append(str(sheet))
    return {
        "pair_direction_sheets": pair_direction,
        "pair_summary_sheets": pair_summary,
        "stable_unstable_comparison_sheets": stable_unstable,
    }


def run_render(
    runtime_value: sealed.EvaluationRuntime,
    output_root: Path,
    full: Mapping[tuple[str, str, float], Mapping[str, Any]],
) -> dict[str, Any]:
    attempt = runtime_value.attempt
    result_path = attempt / "aggregates/factor_injection_results.json"
    if result_path.is_file():
        return read_json(result_path)
    expected = expected_new_items(attempt)
    existing_at_start = 0
    for row in expected:
        rgb_exists = Path(row["rgb_path"]).is_file()
        alpha_exists = Path(row["alpha_path"]).is_file()
        if rgb_exists != alpha_exists:
            raise RuntimeError(f"CAUSAL-ATTRIBUTION-ASSET-MISMATCH: partial render {row}")
        existing_at_start += int(rgb_exists and alpha_exists)
    atomic_json(
        attempt / "audits/factor_render_resume_manifest.json",
        {
            "status": "RUNNING",
            "expected": 1440,
            "completed_before_resume": existing_at_start,
            "missing_before_resume": 1440 - existing_at_start,
            "completed_items_repeated": 0,
        },
        replace=True,
    )
    parity = implementation_parity(runtime_value, full)
    atomic_json(attempt / "audits/factor_implementation_parity.json", parity)
    endpoints = {
        outfit: runtime_value.basis(runtime_value.coefficients[outfit], chunk_size=16384)
        for outfit in OUTFITS
    }
    records: list[dict[str, Any]] = []
    full_records: list[dict[str, Any]] = []
    endpoint_records: list[dict[str, Any]] = []
    created = 0
    residual_cache: dict[tuple[str, str, str, float], tuple[GaussianClothingResiduals, dict[str, float]]] = {}
    for left, right in PAIRS:
        pair_id = f"{left}_{right}"
        for direction in DIRECTIONS:
            source_outfit, target_outfit = direction_outfits(left, right, direction)
            source = endpoints[source_outfit]
            target = endpoints[target_outfit]
            for subset in NEW_SUBSETS:
                for alpha in ALPHAS:
                    residual = inject_residual(source, target, subset, alpha)
                    residual_cache[(pair_id, direction, subset, alpha)] = (
                        residual,
                        residual_statistics(runtime_value, source, residual),
                    )
            for alpha in ALPHAS:
                residual = full_residual(runtime_value, left, right, direction, alpha)
                residual_cache[(pair_id, direction, "111", alpha)] = (
                    residual,
                    residual_statistics(runtime_value, source, residual),
                )
            residual_cache[(pair_id, direction, "000", 0.0)] = (
                source,
                residual_statistics(runtime_value, source, source),
            )
            for condition in CONDITIONS:
                source_row = full[(pair_id, condition, endpoint_stored_alpha(direction))]
                target_row = full[(pair_id, condition, 1.0 - endpoint_stored_alpha(direction))]
                source_rgb = sealed.image_tensor(Path(source_row["rgb_path"]), 3).to(runtime_value.device)
                source_alpha = sealed.image_tensor(Path(source_row["alpha_path"]), 1).to(runtime_value.device)
                target_rgb = sealed.image_tensor(Path(target_row["rgb_path"]), 3).to(runtime_value.device)
                endpoint_metrics_by_alpha = []
                for alpha in ALPHAS:
                    reference = (1.0 - alpha) * source_rgb + alpha * target_rgb
                    endpoint_metrics_by_alpha.append({
                        "alpha": alpha,
                        **image_metrics(runtime_value, left, condition, source_rgb, source_alpha, reference, source_rgb),
                        **residual_cache[(pair_id, direction, "000", 0.0)][1],
                    })
                endpoint_records.append({
                    "subset": "000",
                    "pair": [left, right],
                    "pair_id": pair_id,
                    "direction": direction,
                    "source_outfit": source_outfit,
                    "target_outfit": target_outfit,
                    "render_target_outfit": left,
                    "view_id": condition,
                    "rgb_path": source_row["rgb_path"],
                    "alpha_path": source_row["alpha_path"],
                    "sealed_alpha": endpoint_stored_alpha(direction),
                    "metrics_by_logical_alpha": endpoint_metrics_by_alpha,
                    "reused": True,
                })
                for alpha in ALPHAS:
                    reference = (1.0 - alpha) * source_rgb + alpha * target_rgb
                    for subset in NEW_SUBSETS:
                        residual, residual_metrics = residual_cache[(pair_id, direction, subset, alpha)]
                        rgb_path, alpha_path = render_paths(attempt, subset, pair_id, direction, condition, alpha)
                        existed = rgb_path.is_file() and alpha_path.is_file()
                        rgb, rendered_alpha = sealed.render_or_load(
                            runtime_value, left, condition, residual, rgb_path, alpha_path
                        )
                        created += int(not existed)
                        records.append({
                            "subset": subset,
                            "factors": list(SUBSETS[subset]),
                            "selected_channels": list(subset_channels(subset)),
                            "pair": [left, right],
                            "pair_id": pair_id,
                            "stable_label": pair_id in STABLE,
                            "direction": direction,
                            "source_outfit": source_outfit,
                            "target_outfit": target_outfit,
                            "render_target_outfit": left,
                            "view_id": condition,
                            "alpha": alpha,
                            "rgb_path": str(rgb_path),
                            "alpha_path": str(alpha_path),
                            **image_metrics(runtime_value, left, condition, rgb, rendered_alpha, reference, source_rgb),
                            **residual_metrics,
                        })
                    full_row = full[(pair_id, condition, stored_alpha(direction, alpha))]
                    full_rgb = sealed.image_tensor(Path(full_row["rgb_path"]), 3).to(runtime_value.device)
                    full_alpha = sealed.image_tensor(Path(full_row["alpha_path"]), 1).to(runtime_value.device)
                    full_records.append({
                        "subset": "111",
                        "factors": list(SUBSETS["111"]),
                        "pair": [left, right],
                        "pair_id": pair_id,
                        "stable_label": pair_id in STABLE,
                        "direction": direction,
                        "source_outfit": source_outfit,
                        "target_outfit": target_outfit,
                        "render_target_outfit": left,
                        "view_id": condition,
                        "alpha": alpha,
                        "sealed_alpha": stored_alpha(direction, alpha),
                        "rgb_path": full_row["rgb_path"],
                        "alpha_path": full_row["alpha_path"],
                        **image_metrics(runtime_value, left, condition, full_rgb, full_alpha, reference, source_rgb),
                        **residual_cache[(pair_id, direction, "111", alpha)][1],
                        "reused": True,
                    })
    if len(records) != 1440 or created != 1440 - existing_at_start:
        raise RuntimeError("factor render accounting mismatch")
    if len(full_records) != 240 or len(endpoint_records) != 80:
        raise RuntimeError("endpoint/FULL logical reuse accounting mismatch")
    unique_full = {(row["rgb_path"], row["alpha_path"]) for row in full_records}
    if len(unique_full) != 120:
        raise RuntimeError("FULL unique reuse accounting mismatch")
    sheets = make_contact_sheets(attempt, full)
    result = {
        "schema_version": "canondressgs.research.factor_injection_results.v1",
        "status": "AUTOMATIC_METRICS_COMPLETE_MANUAL_REVIEW_REQUIRED",
        "label": "RESEARCH DIAGNOSTIC — NOT PAPER FINAL",
        "task_id": TASK_ID,
        "protocol_sha256_lf": PROTOCOL_SHA256,
        "alpha_grid": list(ALPHAS),
        "new_render_count": len(records),
        "expected_new_render_count": 1440,
        "newly_rendered_count": created,
        "reused_completed_render_count": existing_at_start,
        "repeated_completed_render_count": 0,
        "endpoint_reuse_logical_entries": len(endpoint_records),
        "full_reuse_logical_entries": len(full_records),
        "full_reuse_unique_files": len(unique_full),
        "full_regenerated_files": 0,
        "records": records,
        "endpoint_reuse_records": endpoint_records,
        "full_reuse_records": full_records,
        "factor_implementation_parity": parity,
        "contact_sheets": sheets,
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
    atomic_json(
        attempt / "audits/factor_render_resume_manifest.json",
        {
            "status": "COMPLETE",
            "expected": 1440,
            "completed_before_resume": existing_at_start,
            "newly_completed": created,
            "completed_after": 1440,
            "completed_items_repeated": 0,
            "full_regenerated_files": 0,
        },
        replace=True,
    )
    return result


def response_index(injection: Mapping[str, Any]) -> dict[tuple[str, str, float, str, str], dict[str, Any]]:
    index: dict[tuple[str, str, float, str, str], dict[str, Any]] = {}
    for row in injection["records"] + injection["full_reuse_records"]:
        key = (row["pair_id"], row["direction"], float(row["alpha"]), row["view_id"], row["subset"])
        if key in index:
            raise RuntimeError(f"duplicate factorial response {key}")
        index[key] = row
    for row in injection["endpoint_reuse_records"]:
        for metrics in row["metrics_by_logical_alpha"]:
            expanded = {
                key: value
                for key, value in row.items()
                if key != "metrics_by_logical_alpha"
            }
            expanded.update(metrics)
            key = (row["pair_id"], row["direction"], float(metrics["alpha"]), row["view_id"], "000")
            if key in index:
                raise RuntimeError(f"duplicate endpoint response {key}")
            index[key] = expanded
    if len(index) != 10 * 2 * 3 * 4 * 8:
        raise RuntimeError("factorial response matrix is incomplete")
    return index


def effect_sign(effect: str, subset: str) -> int:
    levels = {
        "G": 1 if subset[0] == "1" else -1,
        "V": 1 if subset[1] == "1" else -1,
        "A": 1 if subset[2] == "1" else -1,
    }
    if effect == "G":
        return levels["G"]
    if effect == "V":
        return levels["V"]
    if effect == "A":
        return levels["A"]
    if effect == "GxV":
        return levels["G"] * levels["V"]
    if effect == "GxA":
        return levels["G"] * levels["A"]
    if effect == "VxA":
        return levels["V"] * levels["A"]
    if effect == "GxVxA":
        return levels["G"] * levels["V"] * levels["A"]
    raise ValueError(effect)


def factorial_contrasts(values: Mapping[str, float]) -> dict[str, dict[str, float]]:
    if set(values) != set(SUBSET_ORDER):
        raise ValueError("factorial contrast requires all eight subsets")
    ordered = np.asarray([float(values[subset]) for subset in SUBSET_ORDER], dtype=np.float64)
    deviation = float(np.std(ordered, ddof=0))
    result: dict[str, dict[str, float]] = {}
    for effect in EFFECTS:
        signs = np.asarray([effect_sign(effect, subset) for subset in SUBSET_ORDER], dtype=np.float64)
        positive = float(ordered[signs > 0].mean())
        negative = float(ordered[signs < 0].mean())
        raw = 0.5 * (positive - negative)
        result[effect] = {
            "positive_mean": positive,
            "negative_mean": negative,
            "effect": raw,
            "standard_deviation": deviation,
            "standardized_effect": raw / max(deviation, 1.0e-8),
        }
    return result


def artifact_oriented(metric: str, value: float) -> float:
    return -value if metric in {"silhouette_iou", "boundary_fscore"} else value


def median(values: Iterable[float]) -> float:
    selected = [float(value) for value in values]
    return float(statistics.median(selected)) if selected else 0.0


def automatic_effect_summaries(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in records:
        groups[(row["pair_id"], row["direction"], row["effect"], row["metric"])].append(
            abs(float(row["oriented_standardized_effect"]))
        )
    return [
        {
            "pair_id": key[0],
            "stable_label": key[0] in STABLE,
            "direction": key[1],
            "effect": key[2],
            "metric": key[3],
            "median_absolute_standardized_effect": median(values),
        }
        for key, values in sorted(groups.items())
    ]


def strongest_effects(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    artifact_metrics = {"garment_rgb_mae", "lpips", "silhouette_iou", "boundary_fscore", "protected_lpips"}
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in summaries:
        if row["metric"] in artifact_metrics:
            groups[(row["pair_id"], row["direction"], row["effect"])].append(
                float(row["median_absolute_standardized_effect"])
            )
    scores = {
        key: median(values)
        for key, values in groups.items()
    }
    result = []
    for left, right in PAIRS:
        pair_id = f"{left}_{right}"
        for direction in DIRECTIONS:
            candidates = [(effect, scores[(pair_id, direction, effect)]) for effect in EFFECTS]
            effect, score = max(candidates, key=lambda item: (item[1], -EFFECTS.index(item[0])))
            result.append({
                "pair_id": pair_id,
                "stable_label": pair_id in STABLE,
                "direction": direction,
                "effect": effect,
                "score": score,
                "all_effect_scores": {name: scores[(pair_id, direction, name)] for name in EFFECTS},
            })
    return result


EFFECT_SUBSET = {
    "G": "100",
    "V": "010",
    "A": "001",
    "GxV": "110",
    "GxA": "101",
    "VxA": "011",
    "GxVxA": "111",
}


def save_difference_overlay(source_path: Path, candidate_path: Path, output: Path) -> None:
    if output.exists():
        return
    with Image.open(source_path) as opened_source, Image.open(candidate_path) as opened_candidate:
        source = opened_source.convert("RGB")
        candidate = opened_candidate.convert("RGB")
        difference = ImageChops.difference(source, candidate).convert("L")
        difference = ImageEnhance.Contrast(difference).enhance(4.0)
        difference = ImageOps.colorize(difference, black=(0, 0, 0), white=(255, 40, 0))
        overlay = Image.blend(source, difference, 0.55)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".png.tmp")
    overlay.save(temporary, format="PNG")
    os.replace(temporary, output)


def make_strongest_overlays(
    attempt: Path,
    full: Mapping[tuple[str, str, float], Mapping[str, Any]],
    strongest: Sequence[Mapping[str, Any]],
) -> list[str]:
    index = {(row["pair_id"], row["direction"]): row for row in strongest}
    sheets: list[str] = []
    for left, right in PAIRS:
        pair_id = f"{left}_{right}"
        for direction in DIRECTIONS:
            selected = index[(pair_id, direction)]
            subset = EFFECT_SUBSET[selected["effect"]]
            rows = []
            for condition in CONDITIONS:
                source_path, _ = logical_image_path(attempt, full, pair_id, direction, condition, "000", 0.5)
                panels: list[tuple[str, Path]] = [("SOURCE", source_path)]
                for alpha in ALPHAS:
                    candidate_path, _ = logical_image_path(attempt, full, pair_id, direction, condition, subset, alpha)
                    full_path, _ = logical_image_path(attempt, full, pair_id, direction, condition, "111", alpha)
                    overlay = (
                        attempt / "visuals/strongest_factor/components" / pair_id / direction / condition
                        / f"{selected['effect']}_{alpha_id(alpha)}.png"
                    )
                    save_difference_overlay(source_path, candidate_path, overlay)
                    panels.extend([
                        (f"{selected['effect']}@{alpha:.2f}", candidate_path),
                        (f"DIFF@{alpha:.2f}", overlay),
                        (f"FULL@{alpha:.2f}", full_path),
                    ])
                rows.append((condition, panels))
            sheet = attempt / "visuals/strongest_factor" / pair_id / f"{direction}.png"
            sealed.contact_sheet(sheet, rows)
            sheets.append(str(sheet))
    return sheets


def run_analyze(
    attempt: Path,
    full: Mapping[tuple[str, str, float], Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    effects_path = attempt / "aggregates/factorial_effect_results.json"
    profiles_path = attempt / "aggregates/factor_causal_profiles.json"
    stable_path = attempt / "aggregates/stable_pair_factor_analysis.json"
    if effects_path.is_file() and profiles_path.is_file() and stable_path.is_file():
        return read_json(effects_path), read_json(profiles_path), read_json(stable_path)
    injection = read_json(attempt / "aggregates/factor_injection_results.json")
    index = response_index(injection)
    records: list[dict[str, Any]] = []
    for left, right in PAIRS:
        pair_id = f"{left}_{right}"
        for direction in DIRECTIONS:
            for alpha in ALPHAS:
                for condition in CONDITIONS:
                    for metric in METRICS:
                        values = {
                            subset: artifact_oriented(metric, float(index[(pair_id, direction, alpha, condition, subset)][metric]))
                            for subset in SUBSET_ORDER
                        }
                        contrasts = factorial_contrasts(values)
                        for effect, values_for_effect in contrasts.items():
                            records.append({
                                "pair_id": pair_id,
                                "stable_label": pair_id in STABLE,
                                "direction": direction,
                                "alpha": alpha,
                                "view_id": condition,
                                "metric": metric,
                                "metric_orientation": "HIGHER_IS_MORE_ARTIFACT" if metric not in {"silhouette_iou", "boundary_fscore"} else "LOWER_IS_MORE_ARTIFACT",
                                "effect": effect,
                                "oriented_effect": values_for_effect["effect"],
                                "oriented_standardized_effect": values_for_effect["standardized_effect"],
                                "positive_mean": values_for_effect["positive_mean"],
                                "negative_mean": values_for_effect["negative_mean"],
                                "response_standard_deviation": values_for_effect["standard_deviation"],
                            })
    summaries = automatic_effect_summaries(records)
    strongest = strongest_effects(summaries)
    strong_rows = []
    for effect in EFFECTS:
        for metric in METRICS:
            rows = [row for row in summaries if row["effect"] == effect and row["metric"] == metric]
            by_pair: dict[str, dict[str, float]] = defaultdict(dict)
            for row in rows:
                by_pair[row["pair_id"]][row["direction"]] = float(row["median_absolute_standardized_effect"])
            consistent_pairs = sum(
                values.get("A_TO_B", 0.0) >= 0.8 and values.get("B_TO_A", 0.0) >= 0.8
                for values in by_pair.values()
            )
            strong_rows.append({
                "effect": effect,
                "metric": metric,
                "direction_consistent_pair_count": consistent_pairs,
                "strong_evidence": consistent_pairs >= 6,
            })
    effect_result = {
        "schema_version": "canondressgs.research.factorial_effect_results.v1",
        "status": "AUTOMATIC_COMPLETE_MANUAL_GRADES_REQUIRED",
        "label": "RESEARCH DIAGNOSTIC — NOT PAPER FINAL",
        "task_id": TASK_ID,
        "protocol_sha256_lf": PROTOCOL_SHA256,
        "factorial_cell_count": 10 * 2 * 3 * 4,
        "effect_metric_record_count": len(records),
        "records": records,
        "pair_direction_summaries": summaries,
        "strong_continuous_evidence": strong_rows,
        "strongest_effect_by_pair_direction": strongest,
        "factorial_contrasts_deterministic": True,
        "paper_final": False,
    }
    automatic_profiles = {
        "schema_version": "canondressgs.research.factor_causal_profiles.v1",
        "status": "AUTOMATIC_COMPLETE_MANUAL_NECESSITY_SUFFICIENCY_REQUIRED",
        "label": "RESEARCH DIAGNOSTIC — NOT PAPER FINAL",
        "task_id": TASK_ID,
        "records": strongest,
        "manual_profile_pending": True,
        "paper_final": False,
    }
    effect_pair_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in strongest:
        for effect, score in row["all_effect_scores"].items():
            effect_pair_values[(row["pair_id"], effect)].append(float(score))
    stable_rows = []
    for effect in EFFECTS:
        stable_values = [median(values) for (pair_id, name), values in effect_pair_values.items() if name == effect and pair_id in STABLE]
        unstable_values = [median(values) for (pair_id, name), values in effect_pair_values.items() if name == effect and pair_id not in STABLE]
        stable_rows.append({
            "feature": effect,
            "stable_median": median(stable_values),
            "unstable_median": median(unstable_values),
            "median_gap": median(stable_values) - median(unstable_values),
            "stable_count": len(stable_values),
            "unstable_count": len(unstable_values),
        })
    stable_result = {
        "schema_version": "canondressgs.research.stable_pair_factor_analysis.v1",
        "status": "AUTOMATIC_COMPLETE_MANUAL_FULL_SEVERITY_REQUIRED",
        "label": "RESEARCH DIAGNOSTIC — NOT PAPER FINAL",
        "task_id": TASK_ID,
        "stable_pairs": sorted(STABLE),
        "unstable_pairs": sorted({f"{left}_{right}" for left, right in PAIRS} - STABLE),
        "records": stable_rows,
        "stable_labels_changed": False,
        "paper_final": False,
    }
    atomic_json(effects_path, effect_result)
    atomic_json(profiles_path, automatic_profiles)
    atomic_json(stable_path, stable_result)
    overlays = make_strongest_overlays(attempt, full, strongest)
    sheets = injection["contact_sheets"]
    manifest = {
        "schema_version": "canondressgs.research.continuous_control_causal_attribution_visual_manifest.v1",
        "status": "READY_FOR_MANUAL_REVIEW",
        "pair_direction_sheets": sheets["pair_direction_sheets"],
        "pair_summary_sheets": sheets["pair_summary_sheets"],
        "stable_unstable_comparison_sheets": sheets["stable_unstable_comparison_sheets"],
        "strongest_factor_attribution_overlays": overlays,
        "expected_actual_open_count": 52,
        "paper_final": False,
    }
    paths = (
        manifest["pair_direction_sheets"]
        + manifest["pair_summary_sheets"]
        + manifest["stable_unstable_comparison_sheets"]
        + manifest["strongest_factor_attribution_overlays"]
    )
    if len(paths) != 52 or len(set(paths)) != 52 or any(not Path(path).is_file() for path in paths):
        raise RuntimeError("visual manifest accounting mismatch")
    atomic_json(attempt / "audits/visual_manifest.json", manifest)
    return effect_result, automatic_profiles, stable_result


def validate_manual_review(review: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    expected_paths = (
        manifest["pair_direction_sheets"]
        + manifest["pair_summary_sheets"]
        + manifest["stable_unstable_comparison_sheets"]
        + manifest["strongest_factor_attribution_overlays"]
    )
    if review["status"] != "COMPLETE" or review["expected_count"] != 52:
        raise RuntimeError("manual visual review status/count mismatch")
    if review["actual_opened_count"] != 52 or len(review["items"]) != 52:
        raise RuntimeError("manual visual review is incomplete")
    if [item["source_path"] for item in review["items"]] != expected_paths:
        raise RuntimeError("manual visual review path order mismatch")
    if not all(item["actual_opened"] for item in review["items"]):
        raise RuntimeError("manual visual review contains unopened paths")
    pair_items = [item for item in review["items"] if item["kind"] == "PAIR_DIRECTION"]
    if len(pair_items) != 20:
        raise RuntimeError("manual pair-direction review count mismatch")
    for item in pair_items:
        grades = item["grades_by_subset_alpha_view"]
        if len(grades) != 8 * 3 * 4:
            raise RuntimeError("pair-direction grade matrix is incomplete")
        keys = {(row["subset"], float(row["alpha"]), row["view_id"]) for row in grades}
        expected = {(subset, alpha, view) for subset in SUBSET_ORDER for alpha in ALPHAS for view in CONDITIONS}
        if keys != expected:
            raise RuntimeError("pair-direction grade coordinates mismatch")
        for row in grades:
            if set(row["grades"]) != set(VISUAL_CATEGORIES):
                raise RuntimeError("visual grade categories mismatch")
            if any(not isinstance(value, int) or value < 0 or value > 3 for value in row["grades"].values()):
                raise RuntimeError("visual grade outside frozen scale")


def visual_grade_index(review: Mapping[str, Any]) -> dict[tuple[str, str, str, float, str], int]:
    result = {}
    for item in review["items"]:
        if item["kind"] != "PAIR_DIRECTION":
            continue
        for row in item["grades_by_subset_alpha_view"]:
            key = (item["pair_id"], item["direction"], row["subset"], float(row["alpha"]), row["view_id"])
            result[key] = max(int(value) for value in row["grades"].values())
    if len(result) != 20 * 8 * 3 * 4:
        raise RuntimeError("visual grade index is incomplete")
    return result


def direction_evidence(
    grades: Mapping[tuple[str, str, str, float, str], int],
    pair_id: str,
    direction: str,
) -> dict[str, Any]:
    sufficiency_subset = {"G": "100", "V": "010", "A": "001"}
    without_subset = {"G": "011", "V": "101", "A": "110"}
    interaction_subsets = {
        "GxV": ("110", "100", "010"),
        "GxA": ("101", "100", "001"),
        "VxA": ("011", "010", "001"),
    }
    result: dict[str, Any] = {"sufficiency": {}, "necessity": {}, "interactions": {}}
    for factor in ("G", "V", "A"):
        suff_alpha = {}
        nec_alpha = {}
        for alpha in ALPHAS:
            suff_views = 0
            nec_views = 0
            for view in CONDITIONS:
                full = grades[(pair_id, direction, "111", alpha, view)]
                single = grades[(pair_id, direction, sufficiency_subset[factor], alpha, view)]
                without = grades[(pair_id, direction, without_subset[factor], alpha, view)]
                suff_views += int(single >= 2 and abs(single - full) <= 1)
                nec_views += int(full >= 2 and full - without >= 1)
            suff_alpha[f"{alpha:.2f}"] = suff_views
            nec_alpha[f"{alpha:.2f}"] = nec_views
        result["sufficiency"][factor] = {
            "consistent_views_by_alpha": suff_alpha,
            "pass": sum(value >= 3 for value in suff_alpha.values()) >= 2,
        }
        result["necessity"][factor] = {
            "consistent_views_by_alpha": nec_alpha,
            "pass": sum(value >= 3 for value in nec_alpha.values()) >= 2,
        }
    for effect, (combo, first, second) in interaction_subsets.items():
        alpha_counts = {}
        for alpha in ALPHAS:
            count = 0
            for view in CONDITIONS:
                combination = grades[(pair_id, direction, combo, alpha, view)]
                independent = max(
                    grades[(pair_id, direction, first, alpha, view)],
                    grades[(pair_id, direction, second, alpha, view)],
                )
                count += int(combination >= 2 and combination - independent >= 1)
            alpha_counts[f"{alpha:.2f}"] = count
        result["interactions"][effect] = {
            "consistent_views_by_alpha": alpha_counts,
            "pass": sum(value >= 3 for value in alpha_counts.values()) >= 2,
        }
    three_counts = {}
    for alpha in ALPHAS:
        count = 0
        for view in CONDITIONS:
            full = grades[(pair_id, direction, "111", alpha, view)]
            two_factor = max(grades[(pair_id, direction, subset, alpha, view)] for subset in ("110", "101", "011"))
            count += int(full >= 2 and full - two_factor >= 1)
        three_counts[f"{alpha:.2f}"] = count
    result["interactions"]["GxVxA"] = {
        "consistent_views_by_alpha": three_counts,
        "pass": sum(value >= 3 for value in three_counts.values()) >= 2,
    }
    return result


PRIMARY_NAMES = {
    "G": "GEOMETRY_MAIN_EFFECT",
    "V": "VISIBILITY_MAIN_EFFECT",
    "A": "APPEARANCE_MAIN_EFFECT",
    "GxV": "GEOMETRY_VISIBILITY_INTERACTION",
    "GxA": "GEOMETRY_APPEARANCE_INTERACTION",
    "VxA": "VISIBILITY_APPEARANCE_INTERACTION",
    "GxVxA": "THREE_WAY_INTERACTION",
}


def causal_profiles(
    review: Mapping[str, Any],
    effects: Mapping[str, Any],
) -> tuple[dict[str, Any], str, list[str], str]:
    grades = visual_grade_index(review)
    records = []
    pair_pass_counts = {
        "sufficiency": Counter(),
        "necessity": Counter(),
        "interactions": Counter(),
    }
    for left, right in PAIRS:
        pair_id = f"{left}_{right}"
        directions = {direction: direction_evidence(grades, pair_id, direction) for direction in DIRECTIONS}
        pair_summary = {"sufficiency": {}, "necessity": {}, "interactions": {}}
        for category, names in (
            ("sufficiency", ("G", "V", "A")),
            ("necessity", ("G", "V", "A")),
            ("interactions", ("GxV", "GxA", "VxA", "GxVxA")),
        ):
            for name in names:
                passed = all(directions[direction][category][name]["pass"] for direction in DIRECTIONS)
                pair_summary[category][name] = passed
                pair_pass_counts[category][name] += int(passed)
        records.append({
            "pair_id": pair_id,
            "stable_label": pair_id in STABLE,
            "directions": directions,
            "direction_consistent_pair_evidence": pair_summary,
            "factor_causal_profile": "PREREGISTERED_VISUAL_NECESSITY_SUFFICIENCY_COMPLETE",
        })
    sufficient = [factor for factor in ("G", "V", "A") if pair_pass_counts["sufficiency"][factor] >= 6]
    main = [
        factor
        for factor in sufficient
        if pair_pass_counts["necessity"][factor] >= 6
    ]
    interaction = [
        effect
        for effect in ("GxV", "GxA", "VxA")
        if pair_pass_counts["interactions"][effect] >= 6
    ]
    three_way = pair_pass_counts["interactions"]["GxVxA"] >= 6
    if len(sufficient) >= 2:
        primary = "MULTIPLE_INDEPENDENT_FACTORS"
    elif len(main) == 1 and all(pair_pass_counts["sufficiency"][other] < 6 for other in ("G", "V", "A") if other != main[0]):
        primary = PRIMARY_NAMES[main[0]]
    elif len(interaction) == 1 and not sufficient:
        primary = PRIMARY_NAMES[interaction[0]]
    elif three_way and not sufficient and not interaction:
        primary = "THREE_WAY_INTERACTION"
    else:
        primary = "CAUSAL_ATTRIBUTION_INCONCLUSIVE"
    secondary: list[str] = []
    for factor in main:
        name = PRIMARY_NAMES[factor]
        if name != primary:
            secondary.append(name)
    for effect in interaction:
        name = PRIMARY_NAMES[effect]
        if name != primary:
            secondary.append(name)
    if three_way and primary != "THREE_WAY_INTERACTION":
        secondary.append("THREE_WAY_INTERACTION")
    summaries = effects["strongest_effect_by_pair_direction"]
    global_main = {
        factor: median(row["all_effect_scores"][factor] for row in summaries)
        for factor in ("G", "V", "A")
    }
    strongest_main = max(global_main, key=lambda key: (global_main[key], -("G", "V", "A").index(key)))
    if primary in {"GEOMETRY_MAIN_EFFECT", "GEOMETRY_APPEARANCE_INTERACTION"}:
        next_task = "RUN_GEOMETRY_APPEARANCE_DISENTANGLED_BASIS_MICRO_PILOT"
    elif primary == "THREE_WAY_INTERACTION" and pair_pass_counts["necessity"]["G"] >= 6:
        next_task = "RUN_GEOMETRY_APPEARANCE_DISENTANGLED_BASIS_MICRO_PILOT"
    elif primary in {"VISIBILITY_MAIN_EFFECT", "GEOMETRY_VISIBILITY_INTERACTION"}:
        next_task = "RUN_VISIBILITY_GATED_GARMENT_SPACE_MICRO_PILOT"
    elif primary == "VISIBILITY_APPEARANCE_INTERACTION" and pair_pass_counts["necessity"]["V"] >= 6:
        next_task = "RUN_VISIBILITY_GATED_GARMENT_SPACE_MICRO_PILOT"
    elif primary == "APPEARANCE_MAIN_EFFECT":
        next_task = "RUN_APPEARANCE_DISENTANGLED_CONTROL_MICRO_PILOT"
    elif primary == "MULTIPLE_INDEPENDENT_FACTORS":
        next_task = {
            "G": "RUN_GEOMETRY_APPEARANCE_DISENTANGLED_BASIS_MICRO_PILOT",
            "V": "RUN_VISIBILITY_GATED_GARMENT_SPACE_MICRO_PILOT",
            "A": "RUN_APPEARANCE_DISENTANGLED_CONTROL_MICRO_PILOT",
        }[strongest_main]
    else:
        next_task = "DESIGN_TEACHER_ALIGNMENT_AND_SEMANTIC_CORRESPONDENCE_DIAGNOSTIC"
    result = {
        "schema_version": "canondressgs.research.factor_causal_profiles.v1",
        "status": "COMPLETE",
        "label": "RESEARCH DIAGNOSTIC — NOT PAPER FINAL",
        "task_id": TASK_ID,
        "records": records,
        "pair_pass_counts": {
            category: dict(counts)
            for category, counts in pair_pass_counts.items()
        },
        "direction_consistency_required": True,
        "alpha_pass_rule": "at least 2 of 3 alpha levels, each with at least 3 of 4 views",
        "global_main_effect_magnitudes": global_main,
        "largest_main_effect": strongest_main,
        "primary_failure_source": primary,
        "secondary_failure_sources": secondary,
        "next_task": next_task,
        "next_task_started": False,
        "paper_final": False,
    }
    return result, primary, secondary, next_task


def stable_analysis_final(
    automatic: Mapping[str, Any],
    review: Mapping[str, Any],
) -> dict[str, Any]:
    grades = visual_grade_index(review)
    full_pair_scores = {}
    for left, right in PAIRS:
        pair_id = f"{left}_{right}"
        full_pair_scores[pair_id] = median(
            grades[(pair_id, direction, "111", alpha, view)]
            for direction in DIRECTIONS
            for alpha in ALPHAS
            for view in CONDITIONS
        )
    result = dict(automatic)
    result.update({
        "status": "COMPLETE",
        "full_visual_severity_by_pair": full_pair_scores,
        "full_visual_severity_stable_median": median(value for pair, value in full_pair_scores.items() if pair in STABLE),
        "full_visual_severity_unstable_median": median(value for pair, value in full_pair_scores.items() if pair not in STABLE),
        "stable_labels_changed": False,
        "primary_decision_from_stable_label": False,
        "paper_final": False,
    })
    return result


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    values = ["| " + " | ".join(map(str, headers)) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    values.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(values)


def write_new_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"append-only report already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8", newline="\n")


def write_reports(
    injection: Mapping[str, Any],
    effects: Mapping[str, Any],
    profiles: Mapping[str, Any],
    stable: Mapping[str, Any],
    review: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> None:
    pass_counts = profiles["pair_pass_counts"]
    root = [
        "# AAAI27 Continuous-Control Causal Attribution",
        "",
        "**RESEARCH DIAGNOSTIC — NOT PAPER FINAL**",
        "",
        f"- Source HEAD: `{SOURCE_HEAD}`.",
        f"- Corrected protocol SHA-256: `{PROTOCOL_SHA256}`.",
        "- Previous channel attribution: `SCIENTIFICALLY_INCONCLUSIVE_DUE_MIDPOINT_DEGENERACY`.",
        f"- New factor renders: {injection['new_render_count']}/1440.",
        f"- Endpoint logical reuse: {injection['endpoint_reuse_logical_entries']}/80.",
        f"- FULL logical reuse: {injection['full_reuse_logical_entries']}/240; unique sealed entries: {injection['full_reuse_unique_files']}/120; regenerated: 0.",
        f"- Manual visual review: {review['actual_opened_count']}/52.",
        f"- PRIMARY_FAILURE_SOURCE: **{summary['primary_failure_source']}**.",
        f"- SECONDARY_FAILURE_SOURCES: `{json.dumps(summary['secondary_failure_sources'])}`.",
        f"- NEXT_TASK: **{summary['next_task']}** (not started).",
        "",
        "## Necessity and sufficiency",
        "",
        markdown_table(
            ["factor", "sufficient pairs", "necessary pairs"],
            [[factor, pass_counts["sufficiency"].get(factor, 0), pass_counts["necessity"].get(factor, 0)] for factor in ("G", "V", "A")],
        ),
        "",
        "## Interactions",
        "",
        markdown_table(
            ["interaction", "direction-consistent pairs"],
            [[effect, pass_counts["interactions"].get(effect, 0)] for effect in ("GxV", "GxA", "VxA", "GxVxA")],
        ),
        "",
        "## Governance",
        "",
        "The failed pre-result alpha-grid attempt is preserved with zero renders and metrics. The sealed evaluation and previous root-cause archives are unchanged. No training, backward, diagnostic optimizer, optimizer step, scheduler step, or checkpoint write occurred. PAPER_FINAL=0.",
    ]
    factorial = [
        "# AAAI27 Factorial Channel Analysis",
        "",
        "**RESEARCH DIAGNOSTIC — NOT PAPER FINAL**",
        "",
        f"Factorial cells: {effects['factorial_cell_count']}; effect/metric records: {effects['effect_metric_record_count']}.",
        "",
        "The 2^3 decomposition uses G=geometry, V=visibility, and A=appearance. Non-selected channels remain bitwise equal to the source endpoint. The 000 endpoint and 111 FULL images are reused from the sealed interpolation archive.",
        "",
        "## Stable versus unstable",
        "",
        markdown_table(
            ["feature", "stable median", "unstable median", "gap"],
            [[row["feature"], f"{row['stable_median']:.5f}", f"{row['unstable_median']:.5f}", f"{row['median_gap']:.5f}"] for row in stable["records"]],
        ),
        "",
        "Stable labels remain the frozen 3/10 set and do not determine the primary classification.",
    ]
    write_new_text(PROJECT_ROOT / "docs/PAPER/AAAI27_CONTINUOUS_CONTROL_CAUSAL_ATTRIBUTION_20260722.md", "\n".join(root))
    write_new_text(PROJECT_ROOT / "docs/PAPER/AAAI27_FACTORIAL_CHANNEL_ANALYSIS_20260722.md", "\n".join(factorial))


def copy_json_to_repo(relative: str, value: Mapping[str, Any]) -> None:
    atomic_json(PROJECT_ROOT / relative, value)


def run_finalize(attempt: Path, output_root: Path, asset_root: Path, review_path: Path) -> dict[str, Any]:
    validate_governance(output_root)
    preflight = read_json(attempt / "audits/preflight.json")
    injection = read_json(attempt / "aggregates/factor_injection_results.json")
    effects = read_json(attempt / "aggregates/factorial_effect_results.json")
    stable_automatic = read_json(attempt / "aggregates/stable_pair_factor_analysis.json")
    manifest = read_json(attempt / "audits/visual_manifest.json")
    review = read_json(review_path)
    validate_manual_review(review, manifest)
    profiles, primary, secondary, next_task = causal_profiles(review, effects)
    stable = stable_analysis_final(stable_automatic, review)
    optimizer = previous.aggregate_optimizer_provenance(attempt)
    assets_after = verify_manifest(read_json(FROZEN_MANIFEST), PROJECT_ROOT, asset_root, verify_external=True)
    trees_after = {
        "formal": previous.tree_manifest(asset_root / sealed.FORMAL_NAME),
        "p0": previous.tree_manifest(asset_root / sealed.P0_NAME),
        "sealed_evaluation": previous.tree_manifest(asset_root / sealed.OUTPUT_NAME),
        "previous_root_cause": previous.tree_manifest(PREVIOUS_OUTPUT),
    }
    archive_after = previous_archive_fingerprints()
    if assets_after != preflight["frozen_asset_verification"]:
        raise RuntimeError("CAUSAL-ATTRIBUTION-ASSET-MISMATCH: frozen assets changed")
    if trees_after != preflight["frozen_trees_before"]:
        raise RuntimeError("CAUSAL-ATTRIBUTION-ASSET-MISMATCH: frozen output tree changed")
    if archive_after != preflight["previous_archive_fingerprints"]:
        raise RuntimeError("CAUSAL-ATTRIBUTION-ASSET-MISMATCH: previous archive changed")
    summary = {
        "schema_version": "canondressgs.research.continuous_control_causal_attribution_final_summary.v1",
        "status": "COMPLETE",
        "label": "RESEARCH DIAGNOSTIC — NOT PAPER FINAL",
        "task_id": TASK_ID,
        "repair_task_id": REPAIR_TASK_ID,
        "source_head": SOURCE_HEAD,
        "run_branch": RUN_BRANCH,
        "execution_head": git("rev-parse", "HEAD"),
        "protocol_sha256_lf": PROTOCOL_SHA256,
        "protocol_correction_path": "paper_protocol/reviewer_risk/continuous_control_causal_attribution_protocol_correction.json",
        "old_alpha_grid": [0.25, 0.5, 0.75],
        "alpha_grid": list(ALPHAS),
        "previous_failed_attempt": preflight["previous_failed_attempt"],
        "previous_archive_unchanged": True,
        "counts": {
            "new_factor_renders": injection["new_render_count"],
            "endpoint_reuse_logical_entries": injection["endpoint_reuse_logical_entries"],
            "full_reuse_logical_entries": injection["full_reuse_logical_entries"],
            "full_reuse_unique_files": injection["full_reuse_unique_files"],
            "full_regenerated_files": injection["full_regenerated_files"],
            "visual_review": review["actual_opened_count"],
            "stable_pairs": 3,
            "unstable_pairs": 7,
            "training_steps": 0,
            "backward_calls": optimizer["backward_count"],
            "diagnostic_optimizer_created": int(optimizer["diagnostic_optimizer"]["created"]),
            "diagnostic_optimizer_steps": optimizer["diagnostic_optimizer"]["step_count"],
            "legacy_context_optimizer_creation_count": optimizer["legacy_context_optimizer"]["creation_count"],
            "legacy_context_optimizer_zero_grad": optimizer["legacy_context_optimizer"]["zero_grad_count"],
            "legacy_context_optimizer_steps": optimizer["legacy_context_optimizer"]["step_count"],
            "scheduler_steps": optimizer["legacy_context_optimizer"]["scheduler_step_count"],
            "checkpoint_writes": optimizer["checkpoint_write_count"],
            "teacher_mutation": 0,
            "basis_mutation": 0,
            "formal_output_mutation": 0,
            "paper_final": 0,
        },
        "factor_implementation_parity": injection["factor_implementation_parity"],
        "primary_failure_source": primary,
        "secondary_failure_sources": secondary,
        "factor_evidence": profiles["pair_pass_counts"],
        "stable_pair_factor_analysis": stable,
        "visual_review_path": "paper_protocol/reviewer_risk/continuous_control_causal_attribution_visual_review.json",
        "optimizer_provenance": optimizer,
        "optimizer_provenance_path": "paper_protocol/reviewer_risk/continuous_control_causal_attribution_optimizer_provenance.json",
        "root_cause_no_training_gate": "PASS",
        "frozen_assets_before": preflight["frozen_asset_verification"],
        "frozen_assets_after": assets_after,
        "frozen_trees_before": preflight["frozen_trees_before"],
        "frozen_trees_after": trees_after,
        "previous_archive_fingerprints_before": preflight["previous_archive_fingerprints"],
        "previous_archive_fingerprints_after": archive_after,
        "frozen_assets_unchanged": True,
        "next_task": next_task,
        "next_task_started": False,
        "paper_final": False,
        "paper_final_count": 0,
    }
    atomic_json(attempt / "aggregates/factor_causal_profiles_final.json", profiles)
    atomic_json(attempt / "aggregates/stable_pair_factor_analysis_final.json", stable)
    atomic_json(attempt / "aggregates/continuous_control_causal_attribution_visual_review.json", review)
    atomic_json(attempt / "aggregates/continuous_control_causal_attribution_final_summary.json", summary)
    copy_json_to_repo("paper_protocol/reviewer_risk/factor_injection_results.json", injection)
    copy_json_to_repo("paper_protocol/reviewer_risk/factorial_effect_results.json", effects)
    copy_json_to_repo("paper_protocol/reviewer_risk/factor_causal_profiles.json", profiles)
    copy_json_to_repo("paper_protocol/reviewer_risk/stable_pair_factor_analysis.json", stable)
    copy_json_to_repo("paper_protocol/reviewer_risk/continuous_control_causal_attribution_visual_review.json", review)
    copy_json_to_repo("paper_protocol/reviewer_risk/continuous_control_causal_attribution_optimizer_provenance.json", optimizer)
    copy_json_to_repo("paper_protocol/reviewer_risk/continuous_control_causal_attribution_final_summary.json", summary)
    write_reports(injection, effects, profiles, stable, review, summary)
    return summary


def runtime(attempt: Path, asset_root: Path, frozen_protocol: Mapping[str, Any]) -> sealed.EvaluationRuntime:
    return previous.runtime(attempt, asset_root, frozen_protocol)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--attempt", default="attempt_002")
    parser.add_argument("--phase", choices=("preflight", "render", "analyze", "finalize", "all"), default="all")
    parser.add_argument("--visual-review", type=Path)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if args.attempt != "attempt_002":
        raise ValueError("the first formal causal run must be append-only attempt_002")
    attempt = output_root / args.attempt
    frozen_protocol = protocol()
    if args.phase in {"preflight", "all"}:
        result = run_preflight(attempt, output_root, args.asset_root.resolve())
        print(json.dumps({"phase": "preflight", "status": result["status"]}, sort_keys=True))
        if args.phase == "preflight":
            return
    if not (attempt / "audits/preflight.json").is_file():
        raise RuntimeError("preflight must complete before causal phases")
    governance = validate_governance(output_root)
    full = governance["full"]
    if args.phase in {"render", "all"}:
        with previous.NoTrainingProvenance(attempt, "render") as provenance:
            runtime_value = runtime(attempt, args.asset_root.resolve(), frozen_protocol)
            provenance.capture_frozen_runtime_baseline()
            result = run_render(runtime_value, output_root, full)
            print(json.dumps({
                "phase": "render",
                "new_render_count": result["new_render_count"],
                "full_reuse_logical_entries": result["full_reuse_logical_entries"],
                "full_reuse_unique_files": result["full_reuse_unique_files"],
            }, sort_keys=True))
        if args.phase == "render":
            return
    if args.phase in {"analyze", "all"}:
        effects, _, _ = run_analyze(attempt, full)
        print(json.dumps({"phase": "analyze", "effect_metric_records": effects["effect_metric_record_count"]}, sort_keys=True))
        if args.phase == "analyze":
            return
    if args.phase in {"finalize", "all"}:
        if args.visual_review is None:
            if args.phase == "all":
                print(json.dumps({"phase": "finalize", "status": "MANUAL_REVIEW_REQUIRED"}, sort_keys=True))
                return
            raise ValueError("--visual-review is required for finalize")
        result = run_finalize(attempt, output_root, args.asset_root.resolve(), args.visual_review.resolve())
        print(json.dumps({
            "phase": "finalize",
            "status": result["status"],
            "primary": result["primary_failure_source"],
            "next_task": result["next_task"],
        }, sort_keys=True))


if __name__ == "__main__":
    main()
