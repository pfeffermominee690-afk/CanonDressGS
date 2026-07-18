from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
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
from scene.gaussian_clothing_residuals import CHANNELS  # noqa: E402
from scene.o00_arm_support import O00ArmSupport, deform_o00_arm_support, support_tensor_fingerprint  # noqa: E402
from scene.o00_fixed_open_oracle import (  # noqa: E402
    FixedOpenGaussianOracle,
    fixed_open_oracle_contract,
    fixed_open_stage_for_step,
)
from tools.run_module4b_canonical_oracle_micropilot import (  # noqa: E402
    CONDITIONS,
    VIEWS,
    _aggregate_views,
    _append_jsonl,
    _base_gradient_count,
    _base_named_tensors,
    _custom_metrics,
    _environment,
    _git_state,
    _grid,
    _load_samples,
    _loss,
    _regularization,
    _residual_statistics,
    _save_render_set,
    _sha256,
    _specialized_metrics,
    _tensor_state_fingerprint,
    _transition_targets,
    _write_csv,
    _write_json,
    _write_text,
    evaluate_fit_history,
)
from tools.run_o00_arm_support_closure import load_o00_arm_support  # noqa: E402
from tools.run_r3_body_support_design import _render_splats, _sha_record  # noqa: E402
from utils.dressable_camera_utils import build_mmlphuman_camera  # noqa: E402
from utils.full_training_checkpoint_utils import save_full_training_checkpoint  # noqa: E402
from utils.mmlphuman_state_utils import mmlphuman_state_transaction  # noqa: E402
from utils.rendering_loss_utils import compute_static_transition_gradient_cap  # noqa: E402


SCHEMA = "canondressgs.o00_fixed_open_gaussian.v1"
DEFAULT_OUTPUT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/"
    "SUBJECT02-O00-ARM-SUPPORT-CLOSURE-001/attempt_001"
)
DEFAULT_MANIFEST = Path(
    "/root/autodl-tmp/canondressgs_work/data/subject02_dual_target_v5_2/"
    "pilot_manifest_full_v1_v5_2.json"
)


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != SCHEMA:
        raise ValueError("unexpected O00 fixed-open Oracle schema")
    if value.get("outfit") != "O00" or tuple(value["round_robin"]) != CONDITIONS or int(value["steps"]) != 480:
        raise ValueError("O00 fixed-open protocol changed")
    oracle = value["oracle"]
    if any(float(oracle[name]) != 1.0 for name in ("geometry_gate", "appearance_gate", "opacity_gate")):
        raise ValueError("O00 Oracle gates must remain fixed open")
    if bool(oracle["train_gate"]) or bool(oracle["enable_shn"]):
        raise ValueError("O00 Oracle cannot train gates or SHN")
    expected = {
        "edit_reduction_min": 0.10, "clothing_reduction_min": 0.10,
        "protected_final_max": 0.005, "protected_and_preserve_relative_max": 1.10,
    }
    for name, frozen in expected.items():
        if float(value["acceptance"][name]) != frozen:
            raise ValueError(f"O00 Oracle threshold changed: {name}")
    return value


def _optimizer(oracle: FixedOpenGaussianOracle, config: Mapping[str, Any]):
    root = config["oracle"]["optimizer"]
    groups, names = oracle.parameter_groups({
        "geometry_residuals": float(root["geometry_residuals_lr"]),
        "appearance_residuals": float(root["appearance_residuals_lr"]),
    })
    return torch.optim.Adam(groups), names


def _render(
    base: Any,
    support: O00ArmSupport,
    sample: Mapping[str, Any],
    overrides: Any,
    background: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = base._xyz.device
    candidate = support.to(device, base._xyz.dtype)
    camera = build_mmlphuman_camera(
        sample["target_camera"], int(sample["target_camera"]["height"]),
        int(sample["target_camera"]["width"]), device,
    )
    with mmlphuman_state_transaction(
        base, sample["target_pose"].to(device), sample["target_Rh"].to(device), sample["target_Th"].to(device),
    ):
        base_xyz = base.compute_xyz(overrides.as_dict())
        base_covariance = base.get_covariance(canonical_overrides=overrides.as_dict())
        base_opacity = base.compute_opacity(overrides.as_dict()).reshape(-1)
        base_color = base.get_color(torch.linalg.inv(camera["w2c"])[:3, 3], overrides.as_dict())
        rigid = base.get_rigid_transform[1]
        posed = deform_o00_arm_support(
            candidate, rigid, sample["target_Rh"].to(device), sample["target_Th"].to(device),
        )
    return _render_splats(
        camera,
        torch.cat((base_xyz, posed["xyz"])),
        torch.cat((base_covariance, posed["covariance"])),
        torch.cat((base_opacity, candidate.opacity)),
        torch.cat((base_color, candidate.rgb)),
        background,
    )


def _gradient_norm(loss: torch.Tensor, parameters: list[torch.Tensor]) -> float:
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    total = sum(
        (gradient.double().square().sum() for gradient in gradients if gradient is not None),
        torch.zeros((), dtype=torch.float64, device=loss.device),
    )
    return float(torch.sqrt(total).detach())


def _transition_cap(
    base: Any,
    support: O00ArmSupport,
    oracle: FixedOpenGaussianOracle,
    sample: Mapping[str, Any],
    transition_target: torch.Tensor,
    background: torch.Tensor,
    device: torch.device,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    oracle.configure_stage(1)
    output = oracle(base)
    rgb, alpha = _render(base, support, sample, output.canonical_overrides, background)
    original = float(config["loss"]["alpha_global"]) * float(config["loss"]["alpha_transition_original"])
    parts = _loss(rgb, alpha, sample, transition_target, device, config, original)
    parameters = [value for value in oracle.parameters() if value.requires_grad]
    rgb_norm = _gradient_norm(parts["edit"] + parts["clothing"], parameters)
    transition_norm = _gradient_norm(parts["alpha_transition"], parameters)
    result = compute_static_transition_gradient_cap(
        rgb_norm, transition_norm, original,
        cap_fraction=float(config["loss"]["transition_gradient_cap_fraction"]),
        epsilon=float(config["loss"]["transition_gradient_cap_epsilon"]),
    )
    result.update({
        "computed_once_at_step": 0, "frozen_for_steps": [0, 480],
        "reference_parameter_set": "O00 fixed-open stage-1 geometry/scale/rotation/opacity parameters",
        "support_included_in_renderer": True,
    })
    oracle.zero_grad(set_to_none=True)
    return result


def _evaluate_views(
    base: Any,
    support: O00ArmSupport,
    oracle: FixedOpenGaussianOracle,
    samples: Mapping[str, Mapping[str, Any]],
    transitions: Mapping[str, torch.Tensor],
    background: torch.Tensor,
    device: torch.device,
    config: Mapping[str, Any],
    coefficient: float,
    output_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], Any, list[tuple[str, Image.Image]]]:
    rows: list[dict[str, Any]] = []; panels: list[tuple[str, Image.Image]] = []
    oracle.eval()
    with torch.no_grad():
        output = oracle(base)
        for condition in CONDITIONS:
            sample = samples[condition]
            rgb, alpha = _render(base, support, sample, output.canonical_overrides, background)
            parts = _loss(rgb, alpha, sample, transitions[condition], device, config, coefficient)
            row = {
                "condition_id": condition, "view": VIEWS[condition],
                **{name: float(value.detach()) for name, value in parts.items()},
                **_custom_metrics(rgb, alpha, sample), **_specialized_metrics("O00", rgb, alpha, sample),
            }
            rows.append(row)
            if output_dir is not None:
                panels.extend(_save_render_set(output_dir, condition, rgb, alpha, sample))
    return rows, output, panels


def _save_checkpoint(
    path: Path,
    oracle: FixedOpenGaussianOracle,
    optimizer: torch.optim.Optimizer,
    group_names: list[str],
    step: int,
    config: Mapping[str, Any],
    manifest: Path,
    support_path: Path,
    coefficient: float,
) -> None:
    save_full_training_checkpoint(
        path, model=oracle, optimizer=optimizer, optimizer_group_names=group_names,
        scheduler=None, scaler=None,
        training_state={
            "global_step": step, "optimizer_step": step, "epoch": 0,
            "batch_index": step % 4, "gradient_accumulation_position": 0,
            "best_metric": None,
            "sampler_state": {"round_robin": list(CONDITIONS), "next_index": step % 4},
            "stage": oracle.stage,
        },
        data_state={
            "manifest": str(manifest.resolve()), "manifest_sha256": _sha256(manifest),
            "outfit_id": "O00", "conditions": list(CONDITIONS),
            "support_path": str(support_path.resolve()), "support_sha256": _sha256(support_path),
        },
        method_state={
            "schema_version": SCHEMA, "oracle_kind": "fixed_open_gaussian",
            "shared_canonical_field": True, "geometry_gate": 1.0,
            "appearance_gate": 1.0, "opacity_gate": 1.0,
            "transition_coefficient": coefficient,
            "config_sha256": _sha256(Path(config["_source_path"])),
        },
    )


def _parameter_diagnostics(
    oracle: FixedOpenGaussianOracle,
    initial: Mapping[str, torch.Tensor],
    gradient_seen: Mapping[str, int],
) -> dict[str, Any]:
    result = {}
    for name, value in oracle.named_parameters():
        delta = (value.detach().cpu() - initial[name]).abs()
        result[name] = {
            "gradient_steps_nonzero": int(gradient_seen.get(name, 0)),
            "max_abs_update": float(delta.max()), "mean_abs_update": float(delta.mean()),
            "requires_grad_final": bool(value.requires_grad),
        }
    return result


def _run(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    root = args.output.resolve()
    support_status_path = root / "final/ARM_SUPPORT_FINAL_STATUS.json"
    if not support_status_path.is_file():
        raise FileNotFoundError("O00 fixed-open Oracle requires finalized arm-support evidence")
    support_status = json.loads(support_status_path.read_text(encoding="utf-8"))
    if support_status.get("status") != "ARM_SUPPORT_PASS" or not support_status.get("oracle_allowed"):
        raise RuntimeError("ARM_SUPPORT_PASS is mandatory before the O00 Oracle")
    run_dir = root / "oracle"
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite O00 Oracle output: {run_dir}")
    for name in ("checkpoints", "renders", "residuals", "visuals", "contract", "final"):
        (run_dir / name).mkdir(parents=True, exist_ok=False)
    status_path = run_dir / "run_status.json"
    _write_json(status_path, {"status": "RUNNING", "optimizer_steps": 0, "failure_stage": None})
    started = time.perf_counter()
    try:
        git = _git_state()
        if git["status_short"] or git["commit"] != args.expected_head:
            raise RuntimeError(f"O00 Oracle requires clean expected HEAD {args.expected_head}: {git}")
        device = torch.device(args.device)
        pipeline = training.load_config(args.pipeline_config)
        base = training.load_frozen_mmlphuman_base(
            pipeline["base"]["model_dir"], pipeline["base"]["checkpoint_path"], device,
        )
        if int(base._xyz.shape[0]) != int(config["constraints"]["base_gaussian_count"]):
            raise ValueError("O00 Oracle base Gaussian count changed")
        support_path = root / "support/o00_arm_support_12000.pt"
        support = load_o00_arm_support(support_path, int(config["constraints"]["arm_support_count"]))
        support_before = support_tensor_fingerprint(support)
        base_before = _tensor_state_fingerprint(_base_named_tensors(base))
        samples = _load_samples(args.manifest, "O00")
        transitions = _transition_targets(samples, device)
        oracle = FixedOpenGaussianOracle(base, bounds=config["oracle"]["bounds"]).to(device)
        contract = fixed_open_oracle_contract(oracle)
        if not contract["pass"]:
            raise RuntimeError(f"fixed-open Oracle contract failed: {contract}")
        optimizer, group_names = _optimizer(oracle, config)
        initial = {name: value.detach().cpu().clone() for name, value in oracle.named_parameters()}
        gradient_seen = {name: 0 for name, _ in oracle.named_parameters()}
        background = torch.tensor(pipeline["render"]["background"], device=device, dtype=base._xyz.dtype)
        cap = _transition_cap(
            base, support, oracle, samples[CONDITIONS[0]], transitions[CONDITIONS[0]],
            background, device, config,
        )
        coefficient = float(cap["final_frozen_coefficient"])
        _write_json(run_dir / "contract/transition_gradient_cap.json", cap)
        resolved = {name: value for name, value in config.items() if name != "_source_path"}
        resolved["resolved_run"] = {
            "output": str(run_dir), "manifest": str(args.manifest.resolve()),
            "pipeline_config": str(args.pipeline_config.resolve()), "support": str(support_path),
            "transition_coefficient": coefficient,
        }
        (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
        _write_json(run_dir / "contract/input_manifest.json", {
            "schema_version": SCHEMA, "git": git, "environment": _environment(),
            "config": _sha_record(Path(config["_source_path"])),
            "pipeline_config": _sha_record(args.pipeline_config.resolve()),
            "dataset_manifest": _sha_record(args.manifest.resolve()), "support": _sha_record(support_path),
            "support_fingerprint": support_before, "support_count": support.count,
            "base_count": int(base._xyz.shape[0]), "base_fingerprint": base_before,
            "oracle_contract": contract, "V5_3_loss": True, "R2_rotation_path": True,
            "target_fields_used_only_after_prediction_for_loss_and_evaluation": True,
            "image_conditioning_used": False, "teacher_used": False,
        })
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        history: list[dict[str, Any]] = []
        per_view: dict[str, Any] = {}
        milestone_steps = set(int(value) for value in config["render_steps"])
        log_steps = set(int(value) for value in config["log_steps"])

        def record_state(step: int, condition: str) -> dict[str, Any]:
            oracle.eval()
            output = oracle(base)
            rgb, alpha = _render(base, support, samples[condition], output.canonical_overrides, background)
            parts = _loss(rgb, alpha, samples[condition], transitions[condition], device, config, coefficient)
            regularizer, regularizer_parts = _regularization(output, config)
            objective = parts["total"] + regularizer
            row = {
                "step": step, "stage": oracle.stage, "condition_id": condition, "view": VIEWS[condition],
                **{name: float(value.detach()) for name, value in parts.items()},
                "regularization": float(regularizer.detach()), "objective": float(objective.detach()),
                **_custom_metrics(rgb, alpha, samples[condition]),
                **_specialized_metrics("O00", rgb, alpha, samples[condition]),
            }
            if not all(math.isfinite(float(value)) for value in row.values() if isinstance(value, (int, float))):
                raise FloatingPointError(f"non-finite O00 state record at step {step}")
            history.append(row); _append_jsonl(run_dir / "state_records.jsonl", row)
            return row

        oracle.configure_stage(0)
        record_state(0, CONDITIONS[0])
        rows, milestone_output, panels = _evaluate_views(
            base, support, oracle, samples, transitions, background, device, config, coefficient,
            run_dir / "renders/step_000000",
        )
        per_view["0"] = {"conditions": rows, "aggregate": _aggregate_views(rows)}
        _grid(run_dir / "visuals/step_000000_four_view_panel.png", panels, columns=4)
        _write_json(run_dir / "residuals/step_000000.json", _residual_statistics(milestone_output, oracle, config, base))
        _save_checkpoint(
            run_dir / "checkpoints/step_000000.pth", oracle, optimizer, group_names, 0,
            config, args.manifest, support_path, coefficient,
        )

        for step in range(1, 481):
            stage = fixed_open_stage_for_step(step)
            oracle.configure_stage(stage); oracle.train()
            condition = CONDITIONS[(step - 1) % 4]
            optimizer.zero_grad(set_to_none=True)
            output = oracle(base)
            rgb, alpha = _render(base, support, samples[condition], output.canonical_overrides, background)
            parts = _loss(rgb, alpha, samples[condition], transitions[condition], device, config, coefficient)
            regularizer, _ = _regularization(output, config)
            objective = parts["total"] + regularizer
            if not torch.isfinite(objective):
                raise FloatingPointError(f"non-finite O00 objective at optimizer step {step}")
            objective.backward()
            for name, value in oracle.named_parameters():
                if value.grad is not None:
                    if not torch.isfinite(value.grad).all():
                        raise FloatingPointError(f"non-finite O00 gradient in {name} at step {step}")
                    if torch.count_nonzero(value.grad).item() > 0: gradient_seen[name] += 1
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [value for value in oracle.parameters() if value.requires_grad],
                float(config["oracle"]["optimizer"]["gradient_clip_norm"]),
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"non-finite clipped O00 gradient at step {step}")
            optimizer.step()
            record_state(step, condition)
            if step in log_steps:
                _save_checkpoint(
                    run_dir / f"checkpoints/step_{step:06d}.pth", oracle, optimizer, group_names, step,
                    config, args.manifest, support_path, coefficient,
                )
            if step in milestone_steps:
                rows, milestone_output, panels = _evaluate_views(
                    base, support, oracle, samples, transitions, background, device, config, coefficient,
                    run_dir / f"renders/step_{step:06d}",
                )
                per_view[str(step)] = {"conditions": rows, "aggregate": _aggregate_views(rows)}
                _grid(run_dir / f"visuals/step_{step:06d}_four_view_panel.png", panels, columns=4)
                _write_json(
                    run_dir / f"residuals/step_{step:06d}.json",
                    _residual_statistics(milestone_output, oracle, config, base),
                )
            _write_json(status_path, {
                "status": "RUNNING", "optimizer_steps": step, "last_condition": condition,
                "stage": stage, "failure_stage": None,
            })

        fit = evaluate_fit_history(history)
        final_output = oracle(base)
        residual_stats = _residual_statistics(final_output, oracle, config, base)
        parameter_diagnostics = _parameter_diagnostics(oracle, initial, gradient_seen)
        base_after = _tensor_state_fingerprint(_base_named_tensors(base))
        support_after = support_tensor_fingerprint(support)
        expected_nonzero = ("raw_xyz", "raw_log_scaling", "raw_rotvec", "raw_opacity", "raw_sh0")
        gradient_contract = {
            "expected_nonzero_parameters": list(expected_nonzero),
            "all_expected_received_nonzero_gradient": all(gradient_seen[name] > 0 for name in expected_nonzero),
            "raw_shN_gradient_steps": gradient_seen["raw_shN"],
            "raw_shN_output_strictly_zero": bool(torch.count_nonzero(final_output.raw_residuals.delta_shN).item() == 0),
            "gate_parameter_count": len(fixed_open_oracle_contract(oracle)["gate_parameter_names"]),
        }
        gradient_contract["pass"] = (
            gradient_contract["all_expected_received_nonzero_gradient"]
            and gradient_contract["raw_shN_gradient_steps"] == 0
            and gradient_contract["raw_shN_output_strictly_zero"]
            and gradient_contract["gate_parameter_count"] == 0
        )
        frozen = {
            "base_fingerprint_before": base_before, "base_fingerprint_after": base_after,
            "base_bitwise_exact": base_before == base_after,
            "base_parameter_with_gradient_count": _base_gradient_count(base),
            "support_fingerprint_before": support_before, "support_fingerprint_after": support_after,
            "support_bitwise_exact": support_before == support_after,
            "support_trainable_parameter_count": 0,
        }
        metrics = {
            "fit": fit, "initial_four_view": per_view["0"], "final_four_view": per_view["480"],
            "per_view_metrics": per_view, "residual_statistics": residual_stats,
            "parameter_diagnostics": parameter_diagnostics, "gradient_contract": gradient_contract,
            "frozen": frozen, "optimizer_steps": 480,
            "per_condition_update_count": {condition: 120 for condition in CONDITIONS},
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            "visual_acceptance_status": "PENDING_ACTUAL_IMAGE_INSPECTION",
        }
        _write_json(run_dir / "metrics.json", metrics)
        _write_json(run_dir / "per_view_metrics.json", per_view)
        _write_json(run_dir / "residual_statistics.json", residual_stats)
        _write_csv(run_dir / "metrics_history.csv", history)
        summary_panels = []
        for step in config["render_steps"]:
            summary_panels.append((
                f"step {step}", Image.open(run_dir / f"visuals/step_{int(step):06d}_four_view_panel.png").convert("RGB"),
            ))
        _grid(run_dir / "visuals/o00_oracle_training_contact_sheet.png", summary_panels, columns=2, cell=(1024, 720))
        numerical_pass = (
            fit["classification"] == "STRONG_FIT" and fit["last80_slopes"]["edit"] < 0
            and fit["last80_slopes"]["clothing"] < 0
            and frozen["base_bitwise_exact"] and frozen["support_bitwise_exact"]
            and frozen["base_parameter_with_gradient_count"] == 0 and gradient_contract["pass"]
            and max(value["bound_hit_fraction"] for value in residual_stats["channels"].values()) < 0.99
        )
        _write_json(status_path, {
            "status": "ORACLE_COMPLETE_PENDING_VISUAL", "optimizer_steps": 480,
            "fit_classification": fit["classification"], "numerical_pass": numerical_pass,
            "visual_acceptance_status": "PENDING_ACTUAL_IMAGE_INSPECTION",
        })
        _write_text(run_dir / "RUN_ACCEPTANCE.md", "\n".join([
            "# O00 Fixed-Open Gaussian Oracle", "",
            "- Optimizer steps: `480`; shared canonical residual: `true`; gate parameters: `0`.",
            f"- Fit classification: `{fit['classification']}`; numerical pass: `{numerical_pass}`.",
            f"- Base/support bitwise exact: `{frozen['base_bitwise_exact']}` / `{frozen['support_bitwise_exact']}`.",
            "- Visual acceptance remains pending actual opening of the milestone contact sheet.",
        ]))
    except Exception as error:
        _write_json(status_path, {
            "status": "FAILED", "failure_stage": "fixed_open_oracle",
            "exception_type": type(error).__name__, "exception": str(error), "traceback": traceback.format_exc(),
        })
        raise


def _finalize(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    root = args.output.resolve(); run_dir = root / "oracle"; status_path = run_dir / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "ORACLE_COMPLETE_PENDING_VISUAL":
        raise RuntimeError(f"O00 Oracle finalize requires pending visual state, got {status}")
    opened = [item.strip() for item in args.images_opened.split(",") if item.strip()]
    required = "o00_oracle_training_contact_sheet.png"
    if args.inspection_method != "actual image opening with local view_image" or required not in opened:
        raise ValueError("O00 Oracle finalize requires actual opening of its training contact sheet")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    fit = metrics["fit"]
    numerical_pass = bool(status["numerical_pass"])
    if numerical_pass and args.visual_status in {"PASS", "WARN"}:
        oracle_status = "O00_ORACLE_PASS"
    elif fit["classification"] in {"STRONG_FIT", "PARTIAL_FIT"} and args.visual_status != "FAIL":
        oracle_status = "O00_ORACLE_PARTIAL"
    else:
        oracle_status = "O00_ORACLE_FAIL"
    visual = {
        "inspection_method": args.inspection_method, "images_actually_opened": opened,
        "views_inspected": list(VIEWS.values()), "visual_status": args.visual_status,
        "long_sleeve_removed": args.long_sleeve_removed,
        "short_sleeve_formed": args.short_sleeve_formed,
        "continuous_upper_arm": args.continuous_upper_arm,
        "tshirt_and_jeans_direction": args.tshirt_and_jeans_direction,
        "four_view_consistency": args.four_view_consistency,
        "protected_regions_preserved": args.protected_regions_preserved,
        "floaters_or_transparency_cloud": args.floaters_or_transparency_cloud,
        "observations": args.visual_observation,
    }
    final = {
        "status": oracle_status, "numerical_pass": numerical_pass,
        "visual_acceptance": visual, "fit": fit,
        "base_and_support_frozen": bool(metrics["frozen"]["base_bitwise_exact"] and metrics["frozen"]["support_bitwise_exact"]),
    }
    _write_json(run_dir / "final/O00_ORACLE_VISUAL_ACCEPTANCE.json", visual)
    _write_json(run_dir / "final/O00_ORACLE_FINAL_STATUS.json", final)
    _write_text(run_dir / "final/O00_ORACLE_ACCEPTANCE.md", "\n".join([
        "# O00 Fixed-Open Gaussian Oracle Acceptance", "",
        f"- Status: **{oracle_status}**; numerical fit: `{fit['classification']}`.",
        f"- Edit / clothing reduction: `{fit['means']['edit_reduction']:.6f}` / `{fit['means']['clothing_reduction']:.6f}`.",
        f"- Last-80 edit / clothing slopes: `{fit['last80_slopes']['edit']:.9g}` / `{fit['last80_slopes']['clothing']:.9g}`.",
        f"- Long sleeve removed / short sleeve formed / arm continuous: `{args.long_sleeve_removed}` / `{args.short_sleeve_formed}` / `{args.continuous_upper_arm}`.",
        f"- T-shirt + jeans direction / four-view consistency: `{args.tshirt_and_jeans_direction}` / `{args.four_view_consistency}`.",
        f"- Protected regions preserved: `{args.protected_regions_preserved}`; floaters/cloud: `{args.floaters_or_transparency_cloud}`.",
        f"- Base/support frozen: `{final['base_and_support_frozen']}`.",
        f"- Observation: {args.visual_observation}",
    ]))
    _write_json(status_path, {"status": "COMPLETE", "optimizer_steps": 480, **final})


def main() -> None:
    parser = argparse.ArgumentParser(description="O00 fixed-open Gaussian Oracle with frozen arm support")
    parser.add_argument("--phase", required=True, choices=("run", "finalize"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/oracle/o00_fixed_open_gaussian_v1.yaml")
    parser.add_argument("--pipeline-config", type=Path, default=PROJECT_ROOT / "configs/canon_dress_gs_mvp_real.yaml")
    parser.add_argument("--expected-head", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inspection-method", default="")
    parser.add_argument("--images-opened", default="")
    parser.add_argument("--visual-status", choices=("PASS", "WARN", "FAIL"), default="FAIL")
    for name in (
        "long_sleeve_removed", "short_sleeve_formed", "continuous_upper_arm",
        "tshirt_and_jeans_direction", "four_view_consistency", "protected_regions_preserved",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", choices=("PASS", "WARN", "FAIL"), default="FAIL")
    parser.add_argument("--floaters-or-transparency-cloud", choices=("YES", "NO"), default="YES")
    parser.add_argument("--visual-observation", default="")
    args = parser.parse_args()
    config = _load_config(args.config); config["_source_path"] = str(args.config.resolve())
    torch.manual_seed(int(config["seed"])); np.random.seed(int(config["seed"]))
    if args.phase == "run":
        if not args.expected_head: raise ValueError("run requires --expected-head")
        _run(args, config)
    else:
        _finalize(args, config)


if __name__ == "__main__":
    main()
