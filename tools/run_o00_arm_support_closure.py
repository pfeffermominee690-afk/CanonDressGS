from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import sys
import traceback
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_dressable as training  # noqa: E402
from scene.o00_arm_support import (  # noqa: E402
    ARM_SKIN_REGION_NAMES,
    O00ArmSupport,
    adjudicate_o00_support,
    build_o00_arm_support,
    build_subject02_arm_skin_field,
    deform_o00_arm_support,
    support_tensor_fingerprint,
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
from tools.run_r3_body_support_design import (  # noqa: E402
    DEFAULT_MANIFEST,
    _build_smplx_surface,
    _environment,
    _old_sleeve_union,
    _render_splats,
    _save_tensor_png,
    _sha_record,
    _smplx_assets,
    _write_csv,
    _write_json,
    _write_text,
    covered_support_visibility_fraction,
)
from tools.run_r3_clean_body_asset_pilot import _skin_observations  # noqa: E402
from utils.dressable_camera_utils import build_mmlphuman_camera  # noqa: E402
from utils.mmlphuman_state_utils import mmlphuman_state_transaction  # noqa: E402


SCHEMA = "canondressgs.o00_arm_support_closure.v1"
DEFAULT_OUTPUT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/"
    "SUBJECT02-O00-ARM-SUPPORT-CLOSURE-001/attempt_001"
)
SUBDIRS = ("contract", "skin_field", "support", "calibration", "renders", "visuals", "final")


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != SCHEMA:
        raise ValueError("unexpected O00 support closure schema")
    if int(value["support"]["count"]) != 12_000 or not bool(value["constraints"]["only_medium_density"]):
        raise ValueError("O00 support must remain the single 12k Medium asset")
    offsets = tuple(float(item) for item in value["calibration"]["inward_offsets_m"])
    scales = tuple(float(item) for item in value["calibration"]["opacity_scales"])
    if len(offsets) * len(scales) > 9 or len(offsets) * len(scales) != int(value["calibration"]["maximum_combinations"]):
        raise ValueError("O00 support calibration must contain exactly the preregistered <=9 combinations")
    expected = {
        "hole_repair_recall_per_view_min": 0.90,
        "hole_repair_recall_mean_min": 0.95,
        "background_leakage_per_view_max": 0.05,
        "background_leakage_mean_max": 0.03,
        "covered_support_visibility_per_view_max": 0.01,
        "anatomical_envelope_outside_max": 0.02,
    }
    for name, frozen in expected.items():
        if float(value["acceptance"][name]) != frozen:
            raise ValueError(f"O00 support threshold changed: {name}")
    if bool(value["calibration"].get("target_rgb_used_for_calibration", True)):
        raise ValueError("target RGB is forbidden for O00 support calibration")
    return value


def _save_support(path: Path, support: O00ArmSupport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"tensors": support.tensor_dict(), "metadata": dict(support.metadata)}, temporary)
    os.replace(temporary, path)


def load_o00_arm_support(path: Path, expected_count: int = 12_000) -> O00ArmSupport:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    support = O00ArmSupport(metadata=payload["metadata"], **payload["tensors"])
    return support.validate(expected_count)


def _save_support_ply(path: Path, support: O00ArmSupport) -> None:
    xyz = support.canonical_xyz.cpu().numpy()
    rgb = np.round(support.rgb.cpu().numpy() * 255).astype(np.uint8)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {support.count}\n")
        for name in ("x", "y", "z"): handle.write(f"property float {name}\n")
        for name in ("red", "green", "blue"): handle.write(f"property uchar {name}\n")
        handle.write("property int body_part\nproperty int skin_region\nend_header\n")
        for index in range(support.count):
            handle.write(
                f"{xyz[index,0]:.9g} {xyz[index,1]:.9g} {xyz[index,2]:.9g} "
                f"{rgb[index,0]} {rgb[index,1]} {rgb[index,2]} "
                f"{int(support.body_part[index])} {int(support.skin_region[index])}\n"
            )


def _view_state(base: Any, sample: Mapping[str, Any], background: torch.Tensor) -> dict[str, Any]:
    device = base._xyz.device
    camera = build_mmlphuman_camera(
        sample["target_camera"], int(sample["target_camera"]["height"]),
        int(sample["target_camera"]["width"]), device,
    )
    with torch.no_grad(), mmlphuman_state_transaction(
        base, sample["target_pose"].to(device), sample["target_Rh"].to(device), sample["target_Th"].to(device),
    ):
        xyz = base.compute_xyz().detach()
        covariance = base.get_covariance().detach()
        opacity = base.compute_opacity().detach().reshape(-1)
        color = base.get_color(torch.linalg.inv(camera["w2c"])[:3, 3]).detach()
        rigid = base.get_rigid_transform[1].detach()
    rgb, alpha = _render_splats(camera, xyz, covariance, opacity, color, background)
    return {
        "camera": camera, "xyz": xyz, "covariance": covariance, "opacity": opacity,
        "color": color, "rigid": rigid, "p0_rgb": rgb, "p0_alpha": alpha,
    }


def _calibration_row(
    state: Mapping[str, Any],
    support: O00ArmSupport,
    sample: Mapping[str, Any],
    old_union: torch.Tensor,
    background: torch.Tensor,
    old_sleeve_multiplier: float,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    device = state["xyz"].device
    candidate = support.to(device, state["xyz"].dtype)
    posed = deform_o00_arm_support(
        candidate, state["rigid"], sample["target_Rh"].to(device), sample["target_Th"].to(device),
    )
    p1_opacity = state["opacity"].clone(); p1_opacity[old_union] *= old_sleeve_multiplier
    p1_rgb, p1_alpha = _render_splats(
        state["camera"], state["xyz"], state["covariance"], p1_opacity, state["color"], background,
    )
    combined_xyz = torch.cat((state["xyz"], posed["xyz"]))
    combined_covariance = torch.cat((state["covariance"], posed["covariance"]))
    combined_color = torch.cat((state["color"], candidate.rgb))
    p2_rgb, p2_alpha = _render_splats(
        state["camera"], combined_xyz, combined_covariance,
        torch.cat((p1_opacity, candidate.opacity)), combined_color, background,
    )
    p3_rgb, p3_alpha = _render_splats(
        state["camera"], combined_xyz, combined_covariance,
        torch.cat((state["opacity"], candidate.opacity)), combined_color, background,
    )
    support_rgb, support_alpha = _render_splats(
        state["camera"], posed["xyz"], posed["covariance"], candidate.opacity, candidate.rgb, background,
    )
    old_mask = (
        sample["target_old_clothing_mask"].to(device) * sample["target_revealed_skin_mask"].to(device)
    ) >= 0.5
    hole = old_mask & (state["p0_alpha"] >= 0.5) & (p1_alpha < 0.5)
    repaired = hole & (p2_alpha >= 0.5)
    target_foreground = sample["target_foreground_mask"].to(device) >= 0.5
    support_visible = support_alpha >= 0.05
    recall = float(repaired.sum() / hole.sum().clamp_min(1))
    direct_background = float((support_visible & ~target_foreground).sum() / support_visible.sum().clamp_min(1))
    row = {
        "condition_id": sample["target_condition_id"], "view": VIEWS[sample["target_condition_id"]],
        "hole_pixels": int(hole.sum()), "hole_repair_recall": recall,
        "background_leakage": 1.0 - recall,
        "direct_support_background_fraction": direct_background,
        "covered_support_visibility": float(covered_support_visibility_fraction(
            state["p0_rgb"], state["p0_alpha"], p3_rgb, p3_alpha,
        )),
        "finite": bool(all(torch.isfinite(value).all() for value in (
            state["p0_rgb"], state["p0_alpha"], p1_rgb, p1_alpha, p2_rgb, p2_alpha, p3_rgb, p3_alpha,
        ))),
        "pose_explosion": bool(torch.linalg.vector_norm(posed["xyz"], dim=1).max() > 10),
        "posed_xyz_norm_max": float(torch.linalg.vector_norm(posed["xyz"], dim=1).max()),
    }
    images = {
        "p0_rgb": state["p0_rgb"], "p0_alpha": state["p0_alpha"],
        "p1_rgb": p1_rgb, "p1_alpha": p1_alpha, "p2_rgb": p2_rgb, "p2_alpha": p2_alpha,
        "p3_rgb": p3_rgb, "p3_alpha": p3_alpha,
        "support_rgb": support_rgb, "support_alpha": support_alpha,
    }
    return row, images


def _combination_summary(
    rows: list[dict[str, Any]],
    acceptance: Mapping[str, Any],
    outside_fraction: float,
) -> dict[str, Any]:
    recall = [row["hole_repair_recall"] for row in rows]
    leakage = [row["background_leakage"] for row in rows]
    covered = [row["covered_support_visibility"] for row in rows]
    summary = {
        "hole_repair_recall_min": min(recall), "hole_repair_recall_mean": float(np.mean(recall)),
        "background_leakage_max": max(leakage), "background_leakage_mean": float(np.mean(leakage)),
        "covered_support_visibility_max": max(covered), "covered_support_visibility_mean": float(np.mean(covered)),
        "direct_support_background_fraction_max": max(row["direct_support_background_fraction"] for row in rows),
        "anatomical_envelope_outside_fraction": float(outside_fraction),
        "finite": all(row["finite"] and not row["pose_explosion"] for row in rows),
    }
    summary["covered_pass"] = summary["covered_support_visibility_max"] <= float(acceptance["covered_support_visibility_per_view_max"])
    summary["repair_pass"] = (
        summary["hole_repair_recall_min"] >= float(acceptance["hole_repair_recall_per_view_min"])
        and summary["hole_repair_recall_mean"] >= float(acceptance["hole_repair_recall_mean_min"])
    )
    summary["background_pass"] = (
        summary["background_leakage_max"] <= float(acceptance["background_leakage_per_view_max"])
        and summary["background_leakage_mean"] <= float(acceptance["background_leakage_mean_max"])
    )
    summary["outside_pass"] = summary["anatomical_envelope_outside_fraction"] <= float(acceptance["anatomical_envelope_outside_max"])
    summary["hard_pass"] = all(summary[name] for name in ("covered_pass", "repair_pass", "background_pass", "outside_pass", "finite"))
    return summary


def _selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        not bool(row["covered_pass"]), not bool(row["repair_pass"]),
        not bool(row["background_pass"]), not bool(row["outside_pass"]),
        float(row["covered_support_visibility_max"]), -float(row["hole_repair_recall_min"]),
        float(row["background_leakage_mean"]), float(row["anatomical_envelope_outside_fraction"]),
        float(row["inward_offset_m"]), -float(row["opacity_scale"]),
    )


def _preflight(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite O00 arm-support output: {output}")
    for name in SUBDIRS:
        (output / name).mkdir(parents=True, exist_ok=False)
    git = _git_state()
    if git["status_short"] or git["commit"] != args.expected_head:
        raise RuntimeError(f"O00 support preflight requires clean expected HEAD {args.expected_head}: {git}")
    pipeline = training.load_config(args.pipeline_config)
    assets = _smplx_assets(args, pipeline)
    missing = [str(path) for path in assets.values() if not path.is_file()]
    if not args.manifest.is_file(): missing.append(str(args.manifest))
    if missing:
        raise FileNotFoundError(f"O00 support inputs missing: {missing}")
    device = torch.device(args.device)
    base = training.load_frozen_mmlphuman_base(
        pipeline["base"]["model_dir"], pipeline["base"]["checkpoint_path"], device,
    )
    if int(base._xyz.shape[0]) != 200_000:
        raise ValueError("O00 closure requires the formal 200k subject02 base")
    manifest = {
        "schema_version": SCHEMA, "git": git, "environment": _environment(),
        "config": _sha_record(args.config.resolve()), "pipeline_config": _sha_record(args.pipeline_config.resolve()),
        "dataset_manifest": _sha_record(args.manifest.resolve()),
        "assets": {name: _sha_record(path) for name, path in assets.items()},
        "base_gaussian_count": int(base._xyz.shape[0]),
        "base_fingerprint": _tensor_state_fingerprint(_base_named_tensors(base)),
        "conditions": [{"condition_id": condition, "view": VIEWS[condition]} for condition in CONDITIONS],
        "target_rgb_used_for_calibration": False,
        "optimizer_created": False, "optimizer_steps": 0,
    }
    _write_json(output / "contract/input_manifest.json", manifest)
    _write_json(output / "contract/run_status.json", {"status": "PREFLIGHT_PASS", "optimizer_steps": 0})


def _calibrate(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = args.output.resolve()
    status_path = output / "contract/run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "PREFLIGHT_PASS":
        raise RuntimeError(f"O00 support calibration cannot start from {status}")
    _write_json(status_path, {"status": "CALIBRATING", "optimizer_steps": 0, "failure_stage": None})
    try:
        pipeline = training.load_config(args.pipeline_config)
        assets = _smplx_assets(args, pipeline)
        device = torch.device(args.device)
        base = training.load_frozen_mmlphuman_base(
            pipeline["base"]["model_dir"], pipeline["base"]["checkpoint_path"], device,
        )
        base_before = _tensor_state_fingerprint(_base_named_tensors(base))
        sealed = json.loads((output / "contract/input_manifest.json").read_text(encoding="utf-8"))
        if base_before != sealed["base_fingerprint"]:
            raise RuntimeError("base fingerprint differs from O00 support preflight")
        samples = _load_samples(args.manifest, "O00")
        vertices, faces, weights, surface = _build_smplx_surface(assets)
        observation_vertex, observation_rgb, observation_confidence, observation_stats = _skin_observations(
            base, vertices, weights, samples, output,
        )
        field = build_subject02_arm_skin_field(
            vertices, faces, weights, observation_vertex, observation_rgb, observation_confidence,
            expected_observed_pixel_count=int(config["support"]["subject02_observed_pixel_count"]),
        )
        _write_json(output / "skin_field/subject02_arm_skin_field.json", {
            **dict(field.metadata), "surface": surface,
            "region_names": list(ARM_SKIN_REGION_NAMES),
            "observation_statistics": observation_stats,
            "arm_vertex_count": int((field.region >= 0).sum()),
        })
        torch.save({
            "rgb": field.rgb, "confidence": field.confidence, "provenance": field.provenance,
            "region": field.region, "body_part": field.body_part, "source_vertex": field.source_vertex,
        }, output / "skin_field/subject02_arm_skin_field.pt")

        old_union = _old_sleeve_union(base, samples)
        background = torch.tensor(pipeline["render"]["background"], device=device, dtype=base._xyz.dtype)
        states = {condition: _view_state(base, samples[condition], background) for condition in CONDITIONS}
        combinations: list[dict[str, Any]] = []
        all_rows: dict[str, list[dict[str, Any]]] = {}
        supports: dict[str, O00ArmSupport] = {}
        support_config = config["support"]
        for inward in config["calibration"]["inward_offsets_m"]:
            for opacity_scale in config["calibration"]["opacity_scales"]:
                identifier = f"offset_{float(inward):.4f}_opacity_{float(opacity_scale):.2f}".replace(".", "p")
                support = build_o00_arm_support(
                    vertices, faces, weights, field, count=int(support_config["count"]), seed=int(config["seed"]),
                    inward_offset=float(inward), opacity_scale=float(opacity_scale),
                    density_reference_opacity=float(support_config["density_reference_opacity"]),
                    density_reference_count=int(support_config["density_reference_count"]),
                    tangent_scale_multiplier=float(support_config["tangent_scale_multiplier"]),
                    normal_scale=float(support_config["normal_scale_m"]),
                )
                rows = []
                for condition in CONDITIONS:
                    row, _ = _calibration_row(
                        states[condition], support, samples[condition], old_union, background,
                        float(support_config["old_sleeve_opacity_multiplier"]),
                    )
                    rows.append(row)
                summary = _combination_summary(
                    rows, config["acceptance"],
                    float(support.metadata["anatomical_envelope_outside_fraction"]),
                )
                record = {
                    "candidate_id": identifier, "inward_offset_m": float(inward),
                    "opacity_scale": float(opacity_scale), "final_opacity": float(support.opacity[0]), **summary,
                }
                combinations.append(record); all_rows[identifier] = rows; supports[identifier] = support
                _write_json(output / f"calibration/{identifier}_per_view.json", {"candidate": record, "views": rows})
        selected = min(combinations, key=_selection_key)
        selected_id = selected["candidate_id"]
        support = supports[selected_id]
        support_path = output / "support/o00_arm_support_12000.pt"
        _save_support(support_path, support)
        _save_support_ply(output / "support/o00_arm_support_12000.ply", support)
        _write_csv(output / "calibration/arm_support_calibration.csv", combinations)
        _write_json(output / "calibration/arm_support_frozen_config.json", {
            "schema_version": SCHEMA, "selection_rule": list(config["calibration"]["selection_priority"]),
            "selected": selected, "support_count": support.count,
            "support_tensor_fingerprint": support_tensor_fingerprint(support),
            "support_file": _sha_record(support_path), "target_rgb_used_for_calibration": False,
            "calibration_combination_count": len(combinations), "pre_registered_grid": {
                "inward_offsets_m": config["calibration"]["inward_offsets_m"],
                "opacity_scales": config["calibration"]["opacity_scales"],
            },
        })

        visual_panels: list[tuple[str, Image.Image]] = []
        selected_rows: list[dict[str, Any]] = []
        for condition in CONDITIONS:
            row, images = _calibration_row(
                states[condition], support, samples[condition], old_union, background,
                float(support_config["old_sleeve_opacity_multiplier"]),
            )
            selected_rows.append(row)
            directory = output / "renders" / condition
            for name, value in images.items():
                _save_tensor_png(directory / f"{name}.png", value, 1 if "alpha" in name else 3)
            _save_tensor_png(directory / "target_edit_evaluation_only.png", samples[condition]["target_edit_rgb"], 3)
            visual_panels.extend([
                (f"{VIEWS[condition]} P0 base", _to_pil(images["p0_rgb"], 3)),
                (f"{VIEWS[condition]} P1 sleeve suppressed", _to_pil(images["p1_rgb"], 3)),
                (f"{VIEWS[condition]} P2 + support", _to_pil(images["p2_rgb"], 3)),
                (f"{VIEWS[condition]} P3 covered", _to_pil(images["p3_rgb"], 3)),
                (f"{VIEWS[condition]} support", _to_pil(images["support_rgb"], 3)),
                (f"{VIEWS[condition]} support alpha", _to_pil(images["support_alpha"], 1)),
                (f"{VIEWS[condition]} target eval only", _to_pil(samples[condition]["target_edit_rgb"], 3)),
            ])
        _grid(output / "visuals/o00_arm_support_four_view_contact_sheet.png", visual_panels, columns=7, cell=(320, 320))
        _write_json(output / "calibration/selected_support_metrics.json", {
            "selected": selected, "views": selected_rows,
            "target_rgb_used_for_calibration": False,
            "target_rgb_shown_only_after_freeze_for_visual_evaluation": True,
        })
        base_after = _tensor_state_fingerprint(_base_named_tensors(base))
        if base_before != base_after:
            raise RuntimeError("O00 support calibration modified formal base tensors")
        _write_json(output / "contract/post_support_integrity.json", {
            "base_fingerprint_before": base_before, "base_fingerprint_after": base_after,
            "base_bitwise_exact": True, "support_fingerprint": support_tensor_fingerprint(support),
            "support_trainable_parameter_count": 0, "optimizer_created": False, "optimizer_steps": 0,
        })
        _write_json(status_path, {
            "status": "SUPPORT_COMPLETE_PENDING_VISUAL", "optimizer_steps": 0,
            "selected_candidate": selected, "support_count": support.count, "base_bitwise_exact": True,
        })
    except Exception as error:
        _write_json(status_path, {
            "status": "FAILED", "optimizer_steps": 0, "failure_stage": "support_calibration",
            "exception_type": type(error).__name__, "exception": str(error), "traceback": traceback.format_exc(),
        })
        raise


def _finalize(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = args.output.resolve(); status_path = output / "contract/run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "SUPPORT_COMPLETE_PENDING_VISUAL":
        raise RuntimeError(f"support finalize requires pending visual state, got {status}")
    opened = [item.strip() for item in args.images_opened.split(",") if item.strip()]
    required = "o00_arm_support_four_view_contact_sheet.png"
    if args.inspection_method != "actual image opening with local view_image" or required not in opened:
        raise ValueError("support finalize requires actual opening of the four-view contact sheet")
    metrics = json.loads((output / "calibration/selected_support_metrics.json").read_text(encoding="utf-8"))
    integrity = json.loads((output / "contract/post_support_integrity.json").read_text(encoding="utf-8"))
    visual = {
        "inspection_method": args.inspection_method, "images_actually_opened": opened,
        "views_inspected": list(VIEWS.values()), "visual_status": args.visual_status,
        "shoulder_visual": args.shoulder_visual, "wrist_visual": args.wrist_visual,
        "brown_tube_detected": bool(args.brown_tube_detected),
        "observations": args.visual_observation,
    }
    decision = adjudicate_o00_support(
        metrics["views"], outside_fraction=float(metrics["selected"]["anatomical_envelope_outside_fraction"]),
        base_bitwise_exact=bool(integrity["base_bitwise_exact"]),
        shoulder_visual=args.shoulder_visual, wrist_visual=args.wrist_visual,
    )
    if args.visual_status == "FAIL" or bool(args.brown_tube_detected):
        decision["status"] = "ARM_SUPPORT_FAIL"; decision["oracle_allowed"] = False
        decision["visual_pass_or_warn"] = False
    _write_json(output / "final/arm_support_visual_acceptance.json", visual)
    _write_json(output / "final/ARM_SUPPORT_FINAL_STATUS.json", {**decision, "selected": metrics["selected"]})
    _write_text(output / "final/ARM_SUPPORT_ACCEPTANCE.md", "\n".join([
        "# O00 Frozen Arm Support Acceptance", "",
        f"- Status: **{decision['status']}**.",
        f"- Selected inward offset / opacity scale: `{metrics['selected']['inward_offset_m']}` / `{metrics['selected']['opacity_scale']}`.",
        f"- Repair min / mean: `{decision['hole_repair_recall_min']:.6f}` / `{decision['hole_repair_recall_mean']:.6f}`.",
        f"- Background leakage max / mean: `{decision['background_leakage_max']:.6f}` / `{decision['background_leakage_mean']:.6f}`.",
        f"- Covered visibility max: `{decision['covered_support_visibility_max']:.6f}`.",
        f"- Shoulder / wrist visual: `{args.shoulder_visual}` / `{args.wrist_visual}`.",
        f"- Brown tube detected: `{bool(args.brown_tube_detected)}`.",
        f"- Base bitwise exact: `{decision['base_bitwise_exact']}`; support trainable parameters: `0`.",
        f"- Oracle allowed: `{decision['oracle_allowed']}`.",
        f"- Inspection: `{args.inspection_method}`; opened `{opened}`.",
        f"- Observation: {args.visual_observation}",
    ]))
    _write_json(status_path, {"status": "SUPPORT_FINALIZED", **decision})


def main() -> None:
    parser = argparse.ArgumentParser(description="O00 under-clothes arm support closure")
    parser.add_argument("--phase", required=True, choices=("preflight", "calibrate", "finalize"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/audit/o00_arm_support_closure_v1.yaml")
    parser.add_argument("--pipeline-config", type=Path, default=PROJECT_ROOT / "configs/canon_dress_gs_mvp_real.yaml")
    parser.add_argument("--subject02-data", type=Path, default=Path("/root/autodl-tmp/canondressgs_work/data/subject02"))
    parser.add_argument(
        "--smplx-model", type=Path,
        default=Path("/root/autodl-tmp/canondressgs_work/mmlphuman_code/smpl_model/smplx/SMPLX_NEUTRAL.npz"),
    )
    parser.add_argument("--expected-head", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inspection-method", default="")
    parser.add_argument("--images-opened", default="")
    parser.add_argument("--visual-status", choices=("PASS", "WARN", "FAIL"), default="FAIL")
    parser.add_argument("--shoulder-visual", choices=("PASS", "WARN", "FAIL"), default="FAIL")
    parser.add_argument("--wrist-visual", choices=("PASS", "WARN", "FAIL"), default="FAIL")
    parser.add_argument("--brown-tube-detected", action="store_true")
    parser.add_argument("--visual-observation", default="")
    args = parser.parse_args()
    config = _load_config(args.config)
    torch.manual_seed(int(config["seed"])); np.random.seed(int(config["seed"]))
    if args.phase == "preflight":
        if not args.expected_head: raise ValueError("preflight requires --expected-head")
        _preflight(args, config)
    elif args.phase == "calibrate": _calibrate(args, config)
    else: _finalize(args, config)


if __name__ == "__main__":
    main()
