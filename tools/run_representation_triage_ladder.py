from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_dressable as training  # noqa: E402
from scene.representation_capacity_oracle import (  # noqa: E402
    CAPACITY_LOSS_NAME,
    GARMENT_INITIALIZATION,
    GARMENT_LAYER_POINT_COUNT,
    AugmentedGarmentCapacityOracle,
    UnboundedGaussianDeltaField,
    capacity_oracle_loss_v1,
    decide_representation_case,
    direct_stability,
    garment_stability,
)
from tools.run_module4b_canonical_oracle_micropilot import (  # noqa: E402
    _base_gradient_count,
    _base_named_tensors,
    _environment,
    _git_state,
    _grid,
    _load_samples,
    _save_render_set,
    _tensor_state_fingerprint,
)
from tools.run_r3_body_support_design import _render_splats  # noqa: E402
from utils.dressable_camera_utils import build_mmlphuman_camera  # noqa: E402
from utils.mmlphuman_state_utils import mmlphuman_state_transaction  # noqa: E402


SCHEMA = "canondressgs.representation_triage.v1"
OUTFITS = ("O01", "O08")
CONDITIONS = ("cond_000000", "cond_000318", "cond_000017", "cond_000347")
VIEWS = dict(zip(CONDITIONS, ("front", "back", "left", "right")))
RUNG0_DEFINITION = "BOUNDED_RESIDUAL+V5.3_OBJECTIVE+SHARED_CANONICAL+ORIGINAL_200K_SUPPORT"
DEFAULT_OUTPUT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/"
    "SUBJECT02-REPRESENTATION-TRIAGE-001/attempt_001"
)
DEFAULT_SOURCE = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/"
    "SUBJECT02-AAAI27-DATA-CAPACITY-GATE-001/attempt_003"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True, default=str) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != SCHEMA:
        raise ValueError("unexpected representation triage schema")
    if tuple(value["outfits"]) != OUTFITS or tuple(value["conditions"]) != CONDITIONS:
        raise ValueError("representation triage outfit/condition contract changed")
    if int(value["base_gaussian_count"]) != 200_000 or bool(value["shn_enabled"]):
        raise ValueError("representation triage requires original 200k base and SHN disabled")
    if value["capacity_loss"]["name"] != CAPACITY_LOSS_NAME or value["capacity_loss"]["preserve_old_garment"]:
        raise ValueError("capacity loss contract changed")
    if int(value["rung_1"]["steps"]) != 600 or int(value["rung_2"]["steps"]) != 1200 or int(value["rung_3"]["steps"]) != 1200:
        raise ValueError("triage step schedule changed")
    if int(value["rung_3"]["garment_gaussian_count"]) != GARMENT_LAYER_POINT_COUNT:
        raise ValueError("garment layer must contain exactly 30,000 points")
    if value["rung_3"]["initialization"] != GARMENT_INITIALIZATION:
        raise ValueError("garment initialization contract changed")
    if value["decision"]["primary_outfit"] != "O08":
        raise ValueError("O08 must remain the primary low-support diagnostic")
    return value


def select_manifest_records(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten the frozen full-dataset-v1 outfit/observation contract."""

    outfits = manifest.get("outfits")
    if not isinstance(outfits, list):
        raise ValueError("triage manifest must contain the full-dataset-v1 outfits list")
    selected: list[dict[str, Any]] = []
    for outfit in outfits:
        if not isinstance(outfit, Mapping) or outfit.get("outfit_id") not in OUTFITS:
            continue
        observations = outfit.get("observations")
        if not isinstance(observations, list):
            raise ValueError(f"{outfit.get('outfit_id')} observations must be a list")
        for observation in observations:
            if not isinstance(observation, Mapping) or observation.get("condition_id") not in CONDITIONS:
                continue
            selected.append({"outfit_id": outfit["outfit_id"], **dict(observation)})
    if len(selected) != 8 or len({(row["outfit_id"], row["condition_id"]) for row in selected}) != 8:
        raise ValueError("triage manifest does not contain exactly eight unique samples")
    return selected


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()


def target_free_state(sample: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """The only values admitted to the renderer/forward path."""

    camera = sample["target_camera"]
    return {
        "pose": sample["target_pose"].to(device),
        "Rh": sample["target_Rh"].to(device),
        "Th": sample["target_Th"].to(device),
        "camera": {
            "K": camera["K"].to(device), "w2c": camera["w2c"].to(device),
            "width": int(camera["width"]), "height": int(camera["height"]),
        },
    }


def chw(value: torch.Tensor, channels: int) -> torch.Tensor:
    if value.ndim == 3 and value.shape[0] == channels:
        result = value
    elif value.ndim == 3 and value.shape[-1] == channels:
        result = value.permute(2, 0, 1)
    else:
        raise ValueError(f"render tensor shape is not CHW/HWC: {tuple(value.shape)}")
    if not torch.isfinite(result).all():
        raise FloatingPointError("render contains NaN/Inf")
    return result.contiguous()


def render_direct(
    base: Any,
    state: Mapping[str, Any],
    overrides: Any,
    background: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    camera = build_mmlphuman_camera(state["camera"], state["camera"]["height"], state["camera"]["width"], base._xyz.device)
    with mmlphuman_state_transaction(base, state["pose"], state["Rh"], state["Th"]):
        rendered = base.render(camera, background=background, canonical_overrides=overrides.as_dict())
    return chw(rendered[0], 3), chw(rendered[1], 1)


def render_augmented(
    base: Any,
    state: Mapping[str, Any],
    oracle: AugmentedGarmentCapacityOracle,
    background: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = base._xyz.device
    camera = build_mmlphuman_camera(state["camera"], state["camera"]["height"], state["camera"]["width"], device)
    overrides = oracle.base_overrides(base)
    with mmlphuman_state_transaction(base, state["pose"], state["Rh"], state["Th"]):
        base_xyz = base.compute_xyz(overrides.as_dict())
        base_covariance = base.get_covariance(canonical_overrides=overrides.as_dict())
        base_opacity = base.compute_opacity(overrides.as_dict()).reshape(-1)
        camera_position = torch.linalg.inv(camera["w2c"])[:3, 3]
        base_color = base.get_color(camera_position, overrides.as_dict())
        posed = oracle.garment.deform(base.get_rigid_transform[1], state["Rh"], state["Th"])
    return _render_splats(
        camera,
        torch.cat((base_xyz, posed.xyz)),
        torch.cat((base_covariance, posed.covariance)),
        torch.cat((base_opacity, posed.opacity)),
        torch.cat((base_color, posed.color)),
        background,
    )


def loss_for_sample(
    rgb: torch.Tensor,
    alpha: torch.Tensor,
    sample: Mapping[str, Any],
    weights: Mapping[str, float],
    stability: torch.Tensor,
) -> dict[str, torch.Tensor]:
    device = rgb.device
    return capacity_oracle_loss_v1(
        rgb, alpha,
        target_rgb=sample["target_edit_rgb"].to(device),
        base_rgb=sample["target_base_rgb"].to(device),
        target_foreground=sample["target_foreground_mask"].to(device),
        base_foreground=sample["target_base_foreground_mask"].to(device),
        edit_mask=sample["target_edit_mask"].to(device),
        clothing_mask=sample["target_clothing_mask"].to(device),
        old_clothing_mask=sample["target_old_clothing_mask"].to(device),
        protected_mask=sample["target_protected_mask"].to(device),
        transition_mask=sample["target_transition_mask"].to(device),
        weights=weights,
        stability=stability,
    )


def masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    selected = mask.to(prediction)
    if selected.ndim == 2:
        selected = selected[None]
    selected = selected.expand_as(prediction)
    return float(((prediction - target.to(prediction)).abs() * selected).sum() / selected.sum().clamp_min(1))


def capacity_metrics(rgb: torch.Tensor, alpha: torch.Tensor, sample: Mapping[str, Any]) -> dict[str, float]:
    device = rgb.device
    target = sample["target_edit_rgb"].to(device)
    base = sample["target_base_rgb"].to(device)
    protected = sample["target_protected_mask"].to(device)
    garment = torch.maximum(
        torch.maximum(sample["target_edit_mask"].to(device), sample["target_clothing_mask"].to(device)),
        sample["target_old_clothing_mask"].to(device),
    ) * (1 - protected)
    pixel_mask = garment[0] >= 0.5
    target_distance = (rgb - target).abs().mean(0)
    base_distance = (rgb - base).abs().mean(0)
    selected = pixel_mask.sum().clamp_min(1)
    target_closer = float(((target_distance < base_distance) & pixel_mask).sum() / selected)
    base_closer = float(((base_distance < target_distance) & pixel_mask).sum() / selected)
    tie = max(0.0, 1.0 - target_closer - base_closer)
    target_fg = sample["target_foreground_mask"].to(device) >= 0.5
    pred_fg = alpha >= 0.5
    intersection = int((target_fg & pred_fg).sum())
    union = int((target_fg | pred_fg).sum())
    outside = ~target_fg
    return {
        "garment_target_mae": masked_l1(rgb, target, garment),
        "garment_base_mae": masked_l1(rgb, base, garment),
        "target_closer_fraction": target_closer,
        "base_closer_fraction": base_closer,
        "purple_base_retention_score": base_closer,
        "tie_fraction": tie,
        "silhouette_iou": intersection / max(union, 1),
        "protected_mae": masked_l1(rgb, base, protected),
        "background_leakage": float(alpha[outside].mean()) if outside.any() else 0.0,
    }


def abnormal_direct(field: UnboundedGaussianDeltaField, base: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    thresholds = config["abnormal_gaussian"]
    residuals = field.residuals(base)
    overrides = field(base)
    displacement = torch.linalg.vector_norm(residuals.delta_xyz.detach(), dim=1)
    scale_ratio = torch.exp(residuals.delta_log_scaling.detach().abs()).amax(1)
    opacity = torch.sigmoid(overrides.opacity.detach()).reshape(-1)
    center = base._xyz.detach().mean(0)
    radius = torch.linalg.vector_norm(base._xyz.detach() - center, dim=1).max() * float(thresholds["outside_base_radius_multiplier"])
    outside = torch.linalg.vector_norm(overrides.xyz.detach() - center, dim=1) > radius
    abnormal = (
        (displacement > float(thresholds["xyz_displacement_m"]))
        | (scale_ratio > float(thresholds["scale_ratio_max"]))
        | (opacity < float(thresholds["opacity_saturation_low"]))
        | (opacity > float(thresholds["opacity_saturation_high"]))
        | outside
    )
    return _distribution_payload(displacement, scale_ratio, opacity, outside, abnormal)


def abnormal_garment(oracle: AugmentedGarmentCapacityOracle, base: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    thresholds = config["abnormal_gaussian"]
    garment = oracle.garment
    displacement = torch.linalg.vector_norm((garment.xyz - garment.initial_xyz).detach(), dim=1)
    scale_ratio = torch.exp((garment.log_scaling - garment.initial_log_scaling).detach().abs()).amax(1)
    opacity = torch.sigmoid(garment.opacity_logit.detach()).reshape(-1)
    center = base._xyz.detach().mean(0)
    radius = torch.linalg.vector_norm(base._xyz.detach() - center, dim=1).max() * float(thresholds["outside_base_radius_multiplier"])
    outside = torch.linalg.vector_norm(garment.xyz.detach() - center, dim=1) > radius
    abnormal = (
        (displacement > float(thresholds["xyz_displacement_m"]))
        | (scale_ratio > float(thresholds["scale_ratio_max"]))
        | (opacity < float(thresholds["opacity_saturation_low"]))
        | (opacity > float(thresholds["opacity_saturation_high"]))
        | outside
    )
    payload = _distribution_payload(displacement, scale_ratio, opacity, outside, abnormal)
    payload["point_count"] = garment.point_count
    payload["formal_lbs_rows_sum_max_error"] = float((garment.formal_lbs_weights.sum(1) - 1).abs().max())
    payload["initialization"] = GARMENT_INITIALIZATION
    return payload


def _quantiles(value: torch.Tensor) -> dict[str, float]:
    flat = value.detach().float().reshape(-1)
    return {name: float(torch.quantile(flat, q)) for name, q in (("p50", .5), ("p95", .95), ("p99", .99), ("max", 1.0))}


def _distribution_payload(
    displacement: torch.Tensor,
    scale_ratio: torch.Tensor,
    opacity: torch.Tensor,
    outside: torch.Tensor,
    abnormal: torch.Tensor,
) -> dict[str, Any]:
    return {
        "xyz_displacement": _quantiles(displacement),
        "scale_ratio": _quantiles(scale_ratio),
        "opacity": {**_quantiles(opacity), "min": float(opacity.min())},
        "outside_radius_fraction": float(outside.float().mean()),
        "opacity_saturation_fraction": float(((opacity < .005) | (opacity > .995)).float().mean()),
        "scale_abnormal_fraction": float((scale_ratio > 8).float().mean()),
        "abnormal_gaussian_fraction": float(abnormal.float().mean()),
    }


def save_histograms(path: Path, distribution: Mapping[str, Any], model: Any, kind: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if kind == "direct":
        displacement = torch.linalg.vector_norm(model.raw_xyz.detach(), dim=1).cpu().numpy()
        ratio = torch.exp(model.raw_log_scaling.detach().abs()).amax(1).cpu().numpy()
        opacity = torch.sigmoid(model.raw_opacity.detach()).reshape(-1).cpu().numpy()
    else:
        garment = model.garment
        displacement = torch.linalg.vector_norm((garment.xyz - garment.initial_xyz).detach(), dim=1).cpu().numpy()
        ratio = torch.exp((garment.log_scaling - garment.initial_log_scaling).detach().abs()).amax(1).cpu().numpy()
        opacity = torch.sigmoid(garment.opacity_logit.detach()).reshape(-1).cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].hist(displacement, bins=80); axes[0].set_title("xyz displacement")
    axes[1].hist(np.clip(ratio, 0, 10), bins=80); axes[1].set_title("scale ratio (clipped 10)")
    axes[2].hist(opacity, bins=80); axes[2].set_title("opacity")
    fig.suptitle(f"abnormal fraction={distribution['abnormal_gaussian_fraction']:.6f}")
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=150); plt.close(fig)


def save_loss_curve(path: Path, history: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(9, 5))
    for name in ("total", "garment_rgb", "alpha_foreground", "new_silhouette_alpha", "protected_rgb"):
        axis.plot([row["step"] for row in history], [row[name] for row in history], label=name, linewidth=1)
    axis.set_xlabel("optimizer step"); axis.set_ylabel("loss"); axis.legend(); axis.grid(alpha=.2)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=150); plt.close(fig)


def optimizer_for(model: Any, config: Mapping[str, Any]) -> tuple[torch.optim.Optimizer, list[str]]:
    root = config["optimizer"]
    groups, names = model.parameter_groups(root["geometry_lr"], root["appearance_lr"])
    return torch.optim.Adam(groups), names


def save_checkpoint(path: Path, model: Any, optimizer: torch.optim.Optimizer, step: int, metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "global_step": int(step),
        "metadata": dict(metadata), "target_tensors_stored": False,
    }, temporary)
    os.replace(temporary, path)


def run_audit(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    root = args.output.resolve()
    if root.exists():
        raise FileExistsError(root)
    current = git_output("rev-parse", "HEAD")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", config["source_no_go_commit"], current],
        cwd=PROJECT_ROOT,
    ).returncode == 0
    tagged = git_output("rev-list", "-n", "1", config["sealed_tag"])
    if not ancestor or tagged != config["source_no_go_commit"]:
        raise RuntimeError("research branch is not descended from the exact sealed AAAI NO_GO tag")
    if not args.source.is_dir() or not args.manifest.is_file():
        raise FileNotFoundError("source attempt or fixture manifest is missing")
    source_status = json.loads((args.source / "final_adjudication/AAAI27_28_IMAGE_GATE_FINAL_STATUS.json").read_text(encoding="utf-8"))
    if source_status.get("go_no_go") != "NO_GO":
        raise RuntimeError("source AAAI gate is not sealed NO_GO")
    rung0: dict[str, Any] = {"definition": RUNG0_DEFINITION, "status": "VISUAL_FAIL", "outfits": {}}
    for outfit in OUTFITS:
        run = args.source / "oracle" / outfit
        files = {
            "config": run / "config.json", "metrics": run / "metrics.json",
            "final_status": run / "FINAL_STATUS.json",
            "training_contact_sheet": run / "diagnostic_panels/oracle_training_contact_sheet.png",
            "final_four_view": run / "diagnostic_panels/step_000480_four_view.png",
        }
        if not all(path.is_file() for path in files.values()):
            raise FileNotFoundError(f"Rung 0 evidence incomplete for {outfit}")
        metrics = json.loads(files["metrics"].read_text(encoding="utf-8"))
        rung0["outfits"][outfit] = {
            "numeric_status": metrics["numeric"]["status"], "visual_status": metrics["visual_status"],
            "final_status": json.loads(files["final_status"].read_text(encoding="utf-8"))["final_status"],
            "visual_failure": metrics["visual_acceptance"]["observations"],
            "files": {name: {"path": str(path), "sha256": sha256(path)} for name, path in files.items()},
        }
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = select_manifest_records(manifest)
    for name in (
        "contract", "input_audit", "existing_evidence", "rung_1_single_view",
        "rung_2_shared_same_support", "rung_3_augmented_garment_layer",
        "comparisons", "visual_acceptance", "final_adjudication",
    ):
        (root / name).mkdir(parents=True, exist_ok=False)
    atomic_json(root / "contract/config_resolved.json", config)
    atomic_json(root / "contract/run_manifest.json", {
        "schema_version": SCHEMA, "task_id": config["task_id"], "git": _git_state(), "environment": _environment(),
        "source_attempt": str(args.source), "source_attempt_status_sha256": sha256(args.source / "final_adjudication/AAAI27_28_IMAGE_GATE_FINAL_STATUS.json"),
        "dataset_manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest)},
        "pipeline_config": {"path": str(args.pipeline_config), "sha256": sha256(args.pipeline_config)},
        "target_fields_loss_only": True, "image_conditioning_used": False, "new_target_generation": False,
    })
    atomic_json(root / "input_audit/eight_sample_input_audit.json", {
        "status": "PASS", "sample_count": 8,
        "samples": [{"outfit_id": row["outfit_id"], "condition_id": row["condition_id"], "observation": row} for row in selected],
    })
    atomic_json(root / "existing_evidence/RUNG_0_EVIDENCE.json", rung0)
    atomic_text(root / "existing_evidence/RUNG_0_EVIDENCE.md", "\n".join([
        "# Rung 0 — sealed existing failure", "", f"- definition: `{RUNG0_DEFINITION}`", "- status: **VISUAL_FAIL**",
        "- O01 and O08 are referenced by path and SHA256; no optimizer was rerun.",
    ]))
    atomic_json(root / "final_adjudication/RUN_STATUS.json", {"status": "AUDIT_COMPLETE", "optimizer_steps": 0})
    print(json.dumps({"status": "AUDIT_COMPLETE", "output": str(root)}, indent=2))


def _load_runtime(args: argparse.Namespace, outfit: str) -> tuple[Any, dict[str, dict[str, Any]], torch.Tensor, torch.device]:
    device = torch.device(args.device)
    pipeline = training.load_config(args.pipeline_config)
    base = training.load_frozen_mmlphuman_base(pipeline["base"]["model_dir"], pipeline["base"]["checkpoint_path"], device)
    if int(base._xyz.shape[0]) != 200_000:
        raise ValueError("triage requires the original 200k base")
    samples = _load_samples(args.manifest, outfit)
    background = torch.tensor(pipeline["render"]["background"], device=device, dtype=base._xyz.dtype)
    return base, samples, background, device


def _evaluate_one(
    base: Any,
    sample: Mapping[str, Any],
    background: torch.Tensor,
    render: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    output_dir: Path | None = None,
) -> tuple[dict[str, float], list[tuple[str, Image.Image]]]:
    with torch.no_grad():
        rgb, alpha = render()
    metrics = capacity_metrics(rgb, alpha, sample)
    panels: list[tuple[str, Image.Image]] = []
    if output_dir is not None:
        panels = _save_render_set(output_dir, sample["target_condition_id"], rgb, alpha, sample)
    return metrics, panels


def _adjudicate_rung1(metrics: Mapping[str, float], abnormal: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    passed = config["rung_1"]["pass"]
    warned = config["rung_1"]["warn"]
    if (
        metrics["garment_error_reduction"] >= passed["garment_error_reduction_min"]
        and metrics["target_closer_fraction"] >= passed["target_closer_fraction_min"]
        and metrics["silhouette_iou"] >= passed["silhouette_iou_min"]
        and metrics["protected_mae"] <= passed["protected_mae_max"]
        and abnormal["abnormal_gaussian_fraction"] <= passed["abnormal_gaussian_fraction_max"]
        and metrics["background_leakage"] <= passed["background_leakage_max"]
    ):
        return "RUNG1_NUMERIC_PASS"
    if (
        metrics["garment_error_reduction"] >= warned["garment_error_reduction_min"]
        and metrics["target_closer_fraction"] >= warned["target_closer_fraction_min"]
        and metrics["protected_mae"] <= warned["protected_mae_max"]
        and abnormal["abnormal_gaussian_fraction"] <= warned["abnormal_gaussian_fraction_max"]
    ):
        return "RUNG1_NUMERIC_WARN"
    return "RUNG1_NUMERIC_FAIL"


def run_rung1(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    if args.outfit not in OUTFITS or args.condition not in CONDITIONS:
        raise ValueError("Rung 1 outfit/condition is outside the frozen protocol")
    run_dir = args.output / "rung_1_single_view" / args.outfit / args.condition
    if run_dir.exists():
        raise FileExistsError(run_dir)
    for name in ("checkpoints", "renders", "diagnostics"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "RUN_STATUS.json"; atomic_json(status_path, {"status": "RUNNING", "optimizer_steps": 0})
    started = time.perf_counter()
    try:
        torch.manual_seed(int(config["seed"])); np.random.seed(int(config["seed"]))
        base, samples, background, device = _load_runtime(args, args.outfit)
        sample = samples[args.condition]
        state = target_free_state(sample, device)
        field = UnboundedGaussianDeltaField(base).to(device)
        optimizer, group_names = optimizer_for(field, config)
        base_before = _tensor_state_fingerprint(_base_named_tensors(base))
        initial_metrics, panels = _evaluate_one(
            base, sample, background, lambda: render_direct(base, state, field(base), background), run_dir / "renders/step_000000",
        )
        _grid(run_dir / "diagnostics/step_000000_contact_sheet.png", panels, columns=5, cell=(256, 384))
        history: list[dict[str, Any]] = []
        gradient_seen = {name: 0 for name, parameter in field.named_parameters() if parameter.requires_grad}
        render_steps = set(config["rung_1"]["render_steps"])
        record_steps = set(config["rung_1"]["record_steps"])
        for step in range(1, int(config["rung_1"]["steps"]) + 1):
            field.train(); optimizer.zero_grad(set_to_none=True)
            rgb, alpha = render_direct(base, state, field(base), background)
            parts = loss_for_sample(rgb, alpha, sample, config["capacity_loss"], direct_stability(field))
            if not torch.isfinite(parts["total"]):
                raise FloatingPointError(f"non-finite Rung 1 objective at step {step}")
            parts["total"].backward()
            for name, parameter in field.named_parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(f"non-finite gradient in {name}")
                    if torch.count_nonzero(parameter.grad): gradient_seen[name] += 1
            norm = torch.nn.utils.clip_grad_norm_(field.parameters(), float(config["optimizer"]["gradient_clip_norm"]))
            if not torch.isfinite(norm): raise FloatingPointError("non-finite clipped gradient")
            optimizer.step()
            row = {"step": step, **{name: float(value.detach()) for name, value in parts.items()}, "gradient_norm": float(norm)}
            history.append(row); append_jsonl(run_dir / "state_records.jsonl", row)
            if step in record_steps:
                atomic_json(run_dir / f"diagnostics/step_{step:06d}_loss.json", row)
            if step in render_steps:
                metrics, milestone_panels = _evaluate_one(
                    base, sample, background, lambda: render_direct(base, state, field(base), background), run_dir / f"renders/step_{step:06d}",
                )
                atomic_json(run_dir / f"diagnostics/step_{step:06d}_metrics.json", metrics)
                _grid(run_dir / f"diagnostics/step_{step:06d}_contact_sheet.png", milestone_panels, columns=5, cell=(256, 384))
            atomic_json(status_path, {"status": "RUNNING", "optimizer_steps": step})
        final_metrics, final_panels = _evaluate_one(
            base, sample, background, lambda: render_direct(base, state, field(base), background), run_dir / "renders/final",
        )
        final_metrics["garment_error_reduction"] = 1 - final_metrics["garment_target_mae"] / max(initial_metrics["garment_target_mae"], 1e-12)
        abnormal = abnormal_direct(field, base, config)
        numeric = _adjudicate_rung1(final_metrics, abnormal, config)
        base_after = _tensor_state_fingerprint(_base_named_tensors(base))
        save_checkpoint(run_dir / "checkpoints/step_000600.pth", field, optimizer, 600, {
            "rung": 1, "outfit": args.outfit, "condition": args.condition, "optimizer_groups": group_names,
            "target_used_only_for_loss": True,
        })
        write_csv(run_dir / "loss_curve.csv", history); save_loss_curve(run_dir / "diagnostics/loss_curve.png", history)
        save_histograms(run_dir / "diagnostics/parameter_histograms.png", abnormal, field, "direct")
        _grid(run_dir / "diagnostics/final_contact_sheet.png", final_panels, columns=5, cell=(256, 384))
        result = {
            "status": "RUNG1_COMPLETE_PENDING_VISUAL", "numeric_status": numeric, "visual_status": "PENDING_ACTUAL_IMAGE_INSPECTION",
            "outfit_id": args.outfit, "condition_id": args.condition, "view": VIEWS[args.condition],
            "optimizer_steps": 600, "initial_metrics": initial_metrics, "final_metrics": final_metrics,
            "parameter_distributions": abnormal, "gradient_steps_nonzero": gradient_seen,
            "base_fingerprint_before": base_before, "base_fingerprint_after": base_after,
            "base_bitwise_exact": base_before == base_after, "base_gradient_count": _base_gradient_count(base),
            "target_fields_used_only_after_forward": True, "forward_state_fields": sorted(state),
            "shared_across_views": False, "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        }
        atomic_json(run_dir / "metrics.json", result); atomic_json(status_path, result); print(json.dumps(result, indent=2))
    except Exception as error:
        atomic_json(status_path, {"status": "FAILED", "exception_type": type(error).__name__, "exception": str(error), "traceback": traceback.format_exc()})
        raise


def adjudicate_rung1(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    decisions = json.loads(args.visual_decisions.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []; eligibility: dict[str, Any] = {}
    for outfit in OUTFITS:
        accepted = 0; failed = 0
        for condition in CONDITIONS:
            key = f"{outfit}/{condition}"; decision = decisions.get(key)
            if not decision or decision.get("status") not in {"PASS", "WARN", "FAIL"}:
                raise ValueError(f"missing Rung 1 visual decision: {key}")
            if not decision.get("images_actually_opened") or not decision.get("observations"):
                raise ValueError(f"Rung 1 visual evidence missing: {key}")
            path = args.output / "rung_1_single_view" / outfit / condition / "metrics.json"
            metrics = json.loads(path.read_text(encoding="utf-8")); numeric = metrics["numeric_status"]
            if numeric == "RUNG1_NUMERIC_PASS" and decision["status"] in {"PASS", "WARN"}: final = "RUNG1_CAPACITY_PASS"
            elif numeric in {"RUNG1_NUMERIC_PASS", "RUNG1_NUMERIC_WARN"} and decision["status"] in {"PASS", "WARN"}: final = "RUNG1_CAPACITY_WARN"
            else: final = "RUNG1_CAPACITY_FAIL"
            accepted += final in {"RUNG1_CAPACITY_PASS", "RUNG1_CAPACITY_WARN"}; failed += final == "RUNG1_CAPACITY_FAIL"
            metrics.update({"visual_acceptance": decision, "visual_status": decision["status"], "final_status": final})
            atomic_json(path, metrics); atomic_json(path.parent / "FINAL_STATUS.json", {"final_status": final, "numeric": numeric, "visual": decision["status"]})
            rows.append({"outfit_id": outfit, "condition_id": condition, "view": VIEWS[condition], "numeric": numeric, "visual": decision["status"], "final": final, **metrics["final_metrics"]})
        eligibility[outfit] = {
            "pass_or_warn_views": accepted, "failed_views": failed,
            "rung_2_eligible": accepted >= int(config["rung_1"]["rung_2_min_pass_or_warn_views"]),
            "rung_3_eligible_from_rung1": failed >= 3,
        }
    summary = {"status": "COMPLETE", "rows": rows, "eligibility": eligibility}
    atomic_json(args.output / "rung_1_single_view/RUNG1_SUMMARY.json", summary)
    write_csv(args.output / "rung_1_single_view/RUNG1_SUMMARY.csv", rows)
    print(json.dumps(summary, indent=2))


def _shared_numeric(
    rung: str,
    per_view: list[dict[str, Any]],
    abnormal: Mapping[str, Any],
    config: Mapping[str, Any],
) -> str:
    root = config[rung]; reductions = [row["garment_error_reduction"] for row in per_view]
    target_closer = [row["target_closer_fraction"] for row in per_view]
    protected = [row["protected_mae"] for row in per_view]
    mean = float(np.mean(reductions))
    if rung == "rung_2":
        passed = root["pass"]
        if min(reductions) >= passed["per_view_reduction_min"] and mean >= passed["mean_reduction_min"] and min(target_closer) >= passed["per_view_target_closer_min"] and max(protected) <= passed["protected_mae_max"] and abnormal["abnormal_gaussian_fraction"] <= passed["abnormal_gaussian_fraction_max"]:
            return "RUNG2_NUMERIC_PASS"
        warned = root["warn"]
        correct = sum(value >= warned["direction_reduction_min"] for value in reductions)
        if mean >= warned["mean_reduction_min"] and correct >= warned["direction_correct_min_views"] and max(protected) <= warned["protected_mae_max"]:
            return "RUNG2_NUMERIC_WARN"
        return "RUNG2_NUMERIC_FAIL"
    passed = root["pass"]
    if min(reductions) >= passed["per_view_reduction_min"] and mean >= passed["mean_reduction_min"] and min(row["silhouette_iou"] for row in per_view) >= passed["per_view_silhouette_iou_min"] and min(target_closer) >= passed["per_view_target_closer_min"] and max(protected) <= passed["protected_mae_max"] and abnormal["abnormal_gaussian_fraction"] <= passed["abnormal_gaussian_fraction_max"]:
        return "RUNG3_NUMERIC_PASS"
    warned = root["warn"]
    if mean >= warned["mean_reduction_min"] and max(protected) <= warned["protected_mae_max"]:
        return "RUNG3_NUMERIC_WARN"
    return "RUNG3_NUMERIC_FAIL"


def gradient_cosines(
    model: Any,
    samples: Mapping[str, Mapping[str, Any]],
    render_for: Callable[[str], tuple[torch.Tensor, torch.Tensor]],
    stability_for: Callable[[], torch.Tensor],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    parameters = [value for value in model.parameters() if value.requires_grad]
    gradients: dict[str, list[torch.Tensor | None]] = {}
    for condition in CONDITIONS:
        rgb, alpha = render_for(condition)
        loss = loss_for_sample(rgb, alpha, samples[condition], config["capacity_loss"], stability_for())["total"]
        values = torch.autograd.grad(loss, parameters, retain_graph=False, allow_unused=True)
        gradients[condition] = [None if value is None else value.detach() for value in values]
    pairs = []
    for left_index, left in enumerate(CONDITIONS):
        for right in CONDITIONS[left_index + 1:]:
            dot = torch.zeros((), device=parameters[0].device, dtype=torch.float64)
            left_norm = torch.zeros_like(dot); right_norm = torch.zeros_like(dot)
            for a, b in zip(gradients[left], gradients[right]):
                if a is None or b is None: continue
                dot += (a.double() * b.double()).sum(); left_norm += a.double().square().sum(); right_norm += b.double().square().sum()
            cosine = float(dot / (torch.sqrt(left_norm * right_norm).clamp_min(1e-30)))
            pairs.append({"left": left, "right": right, "cosine": cosine})
    return {
        "pairs": pairs, "mean_cosine": float(np.mean([row["cosine"] for row in pairs])),
        "min_cosine": min(row["cosine"] for row in pairs),
        "negative_pair_fraction": float(np.mean([row["cosine"] < 0 for row in pairs])),
    }


def _run_shared(args: argparse.Namespace, config: Mapping[str, Any], rung: str) -> None:
    if args.outfit not in OUTFITS:
        raise ValueError("shared rung outfit is outside the frozen protocol")
    rung1 = json.loads((args.output / "rung_1_single_view/RUNG1_SUMMARY.json").read_text(encoding="utf-8"))
    if rung == "rung_2":
        eligible = rung1["eligibility"][args.outfit]["rung_2_eligible"]
        parent = "rung_2_shared_same_support"; kind = "direct"
    else:
        r2_status = args.output / "rung_2_shared_same_support" / args.outfit / "FINAL_STATUS.json"
        r2_failed = r2_status.is_file() and json.loads(r2_status.read_text(encoding="utf-8")).get("final_status") == "RUNG2_SHARED_SUPPORT_FAIL"
        eligible = rung1["eligibility"][args.outfit]["rung_3_eligible_from_rung1"] or r2_failed
        parent = "rung_3_augmented_garment_layer"; kind = "garment"
    run_dir = args.output / parent / args.outfit
    if not eligible:
        atomic_json(args.output / parent / f"{args.outfit}_SKIPPED.json", {"status": "SKIPPED_NOT_ELIGIBLE", "rung": rung})
        print(json.dumps({"status": "SKIPPED_NOT_ELIGIBLE", "outfit": args.outfit, "rung": rung}, indent=2)); return
    if run_dir.exists(): raise FileExistsError(run_dir)
    for name in ("checkpoints", "renders", "diagnostics"): (run_dir / name).mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "RUN_STATUS.json"; atomic_json(status_path, {"status": "RUNNING", "optimizer_steps": 0})
    started = time.perf_counter()
    try:
        torch.manual_seed(int(config["seed"])); np.random.seed(int(config["seed"]))
        base, samples, background, device = _load_runtime(args, args.outfit)
        states = {condition: target_free_state(sample, device) for condition, sample in samples.items()}
        if rung == "rung_2":
            model: Any = UnboundedGaussianDeltaField(base).to(device)
            render_for = lambda condition: render_direct(base, states[condition], model(base), background)
            stability_for = lambda: direct_stability(model)
            abnormal_for = lambda: abnormal_direct(model, base, config)
        else:
            model = AugmentedGarmentCapacityOracle(base, seed=int(config["seed"]), point_count=GARMENT_LAYER_POINT_COUNT).to(device)
            render_for = lambda condition: render_augmented(base, states[condition], model, background)
            stability_for = lambda: garment_stability(model)
            abnormal_for = lambda: abnormal_garment(model, base, config)
        optimizer, group_names = optimizer_for(model, config)
        base_before = _tensor_state_fingerprint(_base_named_tensors(base))
        initial_per_view: dict[str, dict[str, float]] = {}
        initial_panels: list[tuple[str, Image.Image]] = []
        for condition in CONDITIONS:
            metrics, panels = _evaluate_one(base, samples[condition], background, lambda c=condition: render_for(c), run_dir / "renders/step_000000")
            initial_per_view[condition] = metrics; initial_panels.extend(panels)
        _grid(run_dir / "diagnostics/step_000000_four_view.png", initial_panels, columns=4)
        cosine_initial = gradient_cosines(model, samples, render_for, stability_for, config)
        history: list[dict[str, Any]] = []; gradient_seen = {name: 0 for name, value in model.named_parameters() if value.requires_grad}
        record_steps = set(config[rung]["record_steps"]); render_steps = set(config[rung]["render_steps"])
        steps = int(config[rung]["steps"])
        for step in range(1, steps + 1):
            condition = CONDITIONS[(step - 1) % 4]
            model.train(); optimizer.zero_grad(set_to_none=True)
            rgb, alpha = render_for(condition)
            parts = loss_for_sample(rgb, alpha, samples[condition], config["capacity_loss"], stability_for())
            if not torch.isfinite(parts["total"]): raise FloatingPointError(f"non-finite {rung} objective at {step}")
            parts["total"].backward()
            for name, parameter in model.named_parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all(): raise FloatingPointError(f"non-finite gradient in {name}")
                    if torch.count_nonzero(parameter.grad): gradient_seen[name] += 1
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["optimizer"]["gradient_clip_norm"]))
            if not torch.isfinite(norm): raise FloatingPointError("non-finite clipped gradient")
            optimizer.step()
            row = {"step": step, "condition_id": condition, "view": VIEWS[condition], **{name: float(value.detach()) for name, value in parts.items()}, "gradient_norm": float(norm)}
            history.append(row); append_jsonl(run_dir / "state_records.jsonl", row)
            if step in record_steps: atomic_json(run_dir / f"diagnostics/step_{step:06d}_loss.json", row)
            if step in render_steps:
                milestone_panels: list[tuple[str, Image.Image]] = []; milestone_rows = []
                for current in CONDITIONS:
                    metrics, panels = _evaluate_one(base, samples[current], background, lambda c=current: render_for(c), run_dir / f"renders/step_{step:06d}")
                    milestone_rows.append({"condition_id": current, "view": VIEWS[current], **metrics}); milestone_panels.extend(panels)
                atomic_json(run_dir / f"diagnostics/step_{step:06d}_metrics.json", milestone_rows)
                _grid(run_dir / f"diagnostics/step_{step:06d}_four_view.png", milestone_panels, columns=4)
            atomic_json(status_path, {"status": "RUNNING", "optimizer_steps": step, "last_condition": condition})
        final_per_view = []; final_panels: list[tuple[str, Image.Image]] = []
        for condition in CONDITIONS:
            metrics, panels = _evaluate_one(base, samples[condition], background, lambda c=condition: render_for(c), run_dir / "renders/final")
            metrics["garment_error_reduction"] = 1 - metrics["garment_target_mae"] / max(initial_per_view[condition]["garment_target_mae"], 1e-12)
            final_per_view.append({"condition_id": condition, "view": VIEWS[condition], **metrics}); final_panels.extend(panels)
        abnormal = abnormal_for(); numeric = _shared_numeric(rung, final_per_view, abnormal, config)
        cosine_final = gradient_cosines(model, samples, render_for, stability_for, config)
        base_after = _tensor_state_fingerprint(_base_named_tensors(base))
        save_checkpoint(run_dir / f"checkpoints/step_{steps:06d}.pth", model, optimizer, steps, {
            "rung": int(rung[-1]), "outfit": args.outfit, "optimizer_groups": group_names,
            "target_used_only_for_loss": True, "shared_canonical_field": True,
            "garment_point_count": GARMENT_LAYER_POINT_COUNT if rung == "rung_3" else 0,
        })
        write_csv(run_dir / "loss_curve.csv", history); save_loss_curve(run_dir / "diagnostics/loss_curve.png", history)
        save_histograms(run_dir / "diagnostics/parameter_histograms.png", abnormal, model, kind)
        _grid(run_dir / "diagnostics/final_four_view_contact_sheet.png", final_panels, columns=4)
        result = {
            "status": f"{rung.upper()}_COMPLETE_PENDING_VISUAL", "numeric_status": numeric, "visual_status": "PENDING_ACTUAL_IMAGE_INSPECTION",
            "outfit_id": args.outfit, "optimizer_steps": steps, "per_condition_update_count": {condition: steps // 4 for condition in CONDITIONS},
            "initial_per_view": initial_per_view, "final_per_view": final_per_view,
            "mean_garment_error_reduction": float(np.mean([row["garment_error_reduction"] for row in final_per_view])),
            "parameter_distributions": abnormal, "gradient_steps_nonzero": gradient_seen,
            "view_gradient_cosine": {"initial": cosine_initial, "final": cosine_final},
            "canonical_to_posed_consistency": {"shared_canonical_field": True, "formal_lbs": True, "four_views_finite": True},
            "base_fingerprint_before": base_before, "base_fingerprint_after": base_after, "base_bitwise_exact": base_before == base_after,
            "base_gradient_count": _base_gradient_count(base), "target_fields_used_only_after_forward": True,
            "garment_gaussian_count": GARMENT_LAYER_POINT_COUNT if rung == "rung_3" else 0,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        }
        atomic_json(run_dir / "metrics.json", result); atomic_json(status_path, result); print(json.dumps(result, indent=2))
    except Exception as error:
        atomic_json(status_path, {"status": "FAILED", "exception_type": type(error).__name__, "exception": str(error), "traceback": traceback.format_exc()})
        raise


def adjudicate_shared(args: argparse.Namespace, rung: str) -> None:
    decisions = json.loads(args.visual_decisions.read_text(encoding="utf-8"))
    parent = "rung_2_shared_same_support" if rung == "rung_2" else "rung_3_augmented_garment_layer"
    summary: dict[str, Any] = {"status": "COMPLETE", "rung": rung, "outfits": {}}
    for outfit in OUTFITS:
        path = args.output / parent / outfit / "metrics.json"
        if not path.is_file():
            summary["outfits"][outfit] = {"final_status": "SKIPPED_NOT_RUN"}; continue
        decision = decisions.get(outfit)
        if not decision or decision.get("status") not in {"PASS", "WARN", "FAIL"} or not decision.get("images_actually_opened") or not decision.get("observations"):
            raise ValueError(f"visual decision missing for {rung} {outfit}")
        metrics = json.loads(path.read_text(encoding="utf-8")); numeric = metrics["numeric_status"]
        prefix = "RUNG2_SHARED_SUPPORT" if rung == "rung_2" else "RUNG3_GARMENT_LAYER"
        if numeric.endswith("PASS") and decision["status"] in {"PASS", "WARN"}: final = prefix + "_PASS"
        elif numeric.endswith(("PASS", "WARN")) and decision["status"] in {"PASS", "WARN"}: final = prefix + "_WARN"
        else: final = prefix + "_FAIL"
        metrics.update({"visual_acceptance": decision, "visual_status": decision["status"], "final_status": final})
        atomic_json(path, metrics); atomic_json(path.parent / "FINAL_STATUS.json", {"final_status": final, "numeric": numeric, "visual": decision["status"]})
        summary["outfits"][outfit] = {"final_status": final, "numeric": numeric, "visual": decision["status"], "observations": decision["observations"]}
    atomic_json(args.output / parent / ("RUNG2_SUMMARY.json" if rung == "rung_2" else "RUNG3_SUMMARY.json"), summary)
    print(json.dumps(summary, indent=2))


def finalize(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    rung1 = json.loads((args.output / "rung_1_single_view/RUNG1_SUMMARY.json").read_text(encoding="utf-8"))
    rung2_path = args.output / "rung_2_shared_same_support/RUNG2_SUMMARY.json"
    rung3_path = args.output / "rung_3_augmented_garment_layer/RUNG3_SUMMARY.json"
    rung2 = json.loads(rung2_path.read_text(encoding="utf-8")) if rung2_path.is_file() else {"outfits": {}}
    rung3 = json.loads(rung3_path.read_text(encoding="utf-8")) if rung3_path.is_file() else {"outfits": {}}
    cases = {}
    for outfit in OUTFITS:
        r2 = rung2["outfits"].get(outfit, {}).get("final_status")
        r3 = rung3["outfits"].get(outfit, {}).get("final_status")
        cases[outfit] = decide_representation_case(rung1["eligibility"][outfit]["pass_or_warn_views"], r2, r3)
    overall = cases[config["decision"]["primary_outfit"]]
    conclusions = {
        "A": ("objective_or_residual_parameterization", "REDESIGN_OBJECTIVE_AND_RESIDUAL_PARAMETERIZATION", False, False),
        "B": ("shared_canonical_original_support", "BUILD_LAYERED_CANONICAL_GARMENT_REPRESENTATION", True, False),
        "C": ("original_support_topology", "CLEAN_BODY_AND_INDEPENDENT_GARMENT_LAYER", True, True),
        "D": ("canonical_pose_camera_or_lbs_consistency", "AUDIT_CANONICAL_POSE_CAMERA_ALIGNMENT", False, False),
        "E": ("target_geometry_and_render_contract_unresolved", "AUDIT_TARGET_GEOMETRY_AND_RENDER_CONTRACT", False, False),
    }
    failure, next_task, garment_recommended, clean_body_recommended = conclusions[overall]
    final = {
        "schema_version": SCHEMA, "status": "COMPLETE", "overall_case": overall, "per_outfit_case": cases,
        "primary_outfit": config["decision"]["primary_outfit"], "failure_classification": failure,
        "independent_garment_gaussian_layer_recommended": garment_recommended,
        "clean_body_or_identity_base_recommended": clean_body_recommended,
        "image_conditioned_training_allowed": False, "further_target_generation_allowed": False,
        "formal_garment_layer_implemented": False, "next_unique_task": next_task,
    }
    atomic_json(args.output / "final_adjudication/REPRESENTATION_TRIAGE_FINAL_STATUS.json", final)
    atomic_text(args.output / "final_adjudication/REPRESENTATION_TRIAGE_FINAL_ADJUDICATION.md", "\n".join([
        "# CanonDressGS Representation Triage", "", f"- final case: **Case {overall}**",
        f"- O01 / O08: `{cases['O01']}` / `{cases['O08']}`", f"- primary failure: `{failure}`",
        f"- recommend independent garment layer: `{garment_recommended}`", f"- recommend clean body/identity base: `{clean_body_recommended}`",
        "- image-conditioned training: `not allowed`", "- further target generation: `not allowed`",
        f"- next unique task: `{next_task}`",
    ]))
    atomic_json(args.output / "final_adjudication/RUN_STATUS.json", final)
    print(json.dumps(final, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed-open Gaussian representation triage ladder")
    parser.add_argument("--phase", required=True, choices=(
        "audit", "rung1", "adjudicate-rung1", "rung2", "adjudicate-rung2",
        "rung3", "adjudicate-rung3", "finalize",
    ))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/research/subject02_representation_triage_v1.yaml")
    parser.add_argument("--pipeline-config", type=Path, default=PROJECT_ROOT / "configs/canon_dress_gs_mvp_real.yaml")
    parser.add_argument("--outfit")
    parser.add_argument("--condition")
    parser.add_argument("--visual-decisions", type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args(); config = load_config(args.config)
    if args.phase == "audit": run_audit(args, config)
    elif args.phase == "rung1": run_rung1(args, config)
    elif args.phase == "adjudicate-rung1": adjudicate_rung1(args, config)
    elif args.phase == "rung2": _run_shared(args, config, "rung_2")
    elif args.phase == "adjudicate-rung2": adjudicate_shared(args, "rung_2")
    elif args.phase == "rung3": _run_shared(args, config, "rung_3")
    elif args.phase == "adjudicate-rung3": adjudicate_shared(args, "rung_3")
    else: finalize(args, config)


if __name__ == "__main__":
    main()
