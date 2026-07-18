from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_dressable as training  # noqa: E402
from scene.o00_fixed_open_oracle import (  # noqa: E402
    FixedOpenGaussianOracle,
    fixed_open_oracle_contract,
    fixed_open_stage_for_step,
)
from tools.run_aaai27_data_capacity_preflight import select_benchmark  # noqa: E402
from tools.run_module4b_canonical_oracle_micropilot import (  # noqa: E402
    CONDITIONS,
    VIEWS,
    _aggregate_views,
    _base_gradient_count,
    _base_named_tensors,
    _compute_transition_cap,
    _environment,
    _evaluate_views,
    _git_state,
    _grid,
    _load_samples,
    _loss,
    _regularization,
    _residual_statistics,
    _save_field_panel,
    _tensor_state_fingerprint,
    _transition_targets,
)
from utils.full_training_checkpoint_utils import (  # noqa: E402
    load_full_training_checkpoint,
    save_full_training_checkpoint,
)


OUTFITS = ("O01", "O02", "O03", "O04", "O06", "O07", "O08")
SCHEMA = "canondressgs.aaai27.fixed_open_capacity_oracle.v1"
LOG_STEPS = (0, 1, 10, 20, 40, 80, 120, 160, 240, 320, 400, 480)
RENDER_STEPS = (0, 40, 80, 160, 240, 320, 400, 480)


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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def json_fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def load_config(gate_path: Path, oracle_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    gate = yaml.safe_load(gate_path.read_text(encoding="utf-8"))
    template = yaml.safe_load(oracle_path.read_text(encoding="utf-8"))
    if tuple(gate["candidate_outfits"]) != OUTFITS:
        raise ValueError("candidate outfit contract changed")
    if tuple(gate["oracle"]["round_robin"]) != tuple(CONDITIONS) or int(gate["oracle"]["steps"]) != 480:
        raise ValueError("Oracle condition order or steps changed")
    if tuple(gate["oracle"]["log_steps"]) != LOG_STEPS or tuple(gate["oracle"]["render_steps"]) != RENDER_STEPS:
        raise ValueError("Oracle evidence schedule changed")
    if gate["oracle"]["fixed_gates"] != {"geometry": 1.0, "appearance": 1.0, "opacity": 1.0}:
        raise ValueError("fixed-open gates changed")
    if gate["oracle"]["train_gate"] or gate["oracle"]["enable_shn"]:
        raise ValueError("gate or SHN training is forbidden")
    if gate["oracle"]["bounds"] != template["oracle"]["bounds"]:
        raise ValueError("residual bounds differ from the frozen Oracle template")
    expected_rates = gate["oracle"]["optimizer"]
    actual_rates = template["oracle"]["optimizer"]
    for key in ("class", "geometry_residuals_lr", "appearance_residuals_lr", "gradient_clip_norm"):
        if expected_rates[key] != actual_rates[key]:
            raise ValueError(f"optimizer setting changed: {key}")
    if template["loss"]["contract"] != "v5_3_region_aware_dual_target":
        raise ValueError("Oracle template is not V5.3")
    config = {
        "oracle": {
            "bounds": copy.deepcopy(gate["oracle"]["bounds"]),
            "optimizer": copy.deepcopy(gate["oracle"]["optimizer"]),
        },
        "loss": copy.deepcopy(template["loss"]),
        "log_steps": list(LOG_STEPS),
        "render_steps": list(RENDER_STEPS),
    }
    return gate, config


def optimizer(oracle: FixedOpenGaussianOracle, config: Mapping[str, Any]):
    rates = config["oracle"]["optimizer"]
    groups, names = oracle.parameter_groups({
        "geometry_residuals": float(rates["geometry_residuals_lr"]),
        "appearance_residuals": float(rates["appearance_residuals_lr"]),
    })
    return torch.optim.Adam(groups), names


def save_checkpoint(
    path: Path,
    oracle: FixedOpenGaussianOracle,
    optim: torch.optim.Optimizer,
    group_names: list[str],
    step: int,
    manifest: Path,
    outfit: str,
    config: Mapping[str, Any],
    coefficient: float,
) -> None:
    save_full_training_checkpoint(
        path,
        model=oracle,
        optimizer=optim,
        optimizer_group_names=group_names,
        scheduler=None,
        scaler=None,
        training_state={
            "global_step": step,
            "optimizer_step": step,
            "epoch": 0,
            "batch_index": step % 4,
            "gradient_accumulation_position": 0,
            "best_metric": None,
            "sampler_state": {"round_robin": list(CONDITIONS), "next_index": step % 4},
            "stage": oracle.stage,
        },
        data_state={
            "manifest": str(manifest.resolve()),
            "manifest_sha256": sha256(manifest),
            "outfit_id": outfit,
            "conditions": list(CONDITIONS),
        },
        method_state={
            "schema_version": SCHEMA,
            "oracle_kind": "gaussian_level_fixed_open_canonical",
            "config_sha256": json_fingerprint(config),
            "transition_coefficient": coefficient,
            "shared_canonical_field": True,
        },
    )


def load_checkpoint_exact(
    path: Path,
    base: Any,
    config: Mapping[str, Any],
    manifest: Path,
    outfit: str,
    step: int,
    coefficient: float,
    device: torch.device,
) -> tuple[FixedOpenGaussianOracle, torch.optim.Optimizer, list[str], dict[str, Any]]:
    replacement = FixedOpenGaussianOracle(base, bounds=config["oracle"]["bounds"]).to(device)
    replacement.configure_stage(fixed_open_stage_for_step(step))
    replacement_optimizer, names = optimizer(replacement, config)
    payload = load_full_training_checkpoint(
        path,
        model=replacement,
        optimizer=replacement_optimizer,
        optimizer_group_names=names,
        scheduler=None,
        scaler=None,
        expected_method_state={
            "schema_version": SCHEMA,
            "oracle_kind": "gaussian_level_fixed_open_canonical",
            "config_sha256": json_fingerprint(config),
            "transition_coefficient": coefficient,
            "shared_canonical_field": True,
        },
        expected_data_state={
            "manifest": str(manifest.resolve()),
            "manifest_sha256": sha256(manifest),
            "outfit_id": outfit,
            "conditions": list(CONDITIONS),
        },
    )
    return replacement, replacement_optimizer, names, payload


def linear_slope(rows: list[dict[str, Any]], name: str) -> float:
    return float(np.polyfit(
        np.asarray([row["step"] for row in rows], dtype=np.float64),
        np.asarray([row[name] for row in rows], dtype=np.float64),
        1,
    )[0])


def numeric_adjudication(
    history: list[dict[str, Any]], residual: Mapping[str, Any], acceptance: Mapping[str, Any]
) -> dict[str, Any]:
    if [row["step"] for row in history] != list(range(481)):
        raise ValueError("history must contain steps 0 through 480 exactly once")
    first = history[0:20]
    last = history[441:481]
    slope_rows = history[401:481]
    metrics = {}
    for name in ("edit", "clothing", "protected", "preserve", "total", "objective"):
        metrics[f"{name}_first20"] = float(np.mean([row[name] for row in first]))
        metrics[f"{name}_last40"] = float(np.mean([row[name] for row in last]))
    for name in ("edit", "clothing"):
        metrics[f"{name}_reduction"] = 1 - metrics[f"{name}_last40"] / max(metrics[f"{name}_first20"], 1e-12)
        metrics[f"{name}_last80_slope"] = linear_slope(slope_rows, name)
    finite = all(
        math.isfinite(float(value))
        for row in history for value in row.values() if isinstance(value, (int, float))
    )
    protected_final = float(history[-1]["protected"])
    preserve_final = float(history[-1]["preserve"])
    protected_ok = protected_final < float(acceptance["protected_final_max"]) and protected_final <= float(history[0]["protected"]) * float(acceptance["protected_and_preserve_relative_max"]) + 1e-12
    preserve_ok = preserve_final <= float(history[0]["preserve"]) * float(acceptance["protected_and_preserve_relative_max"]) + 1e-12
    slopes_ok = metrics["edit_last80_slope"] <= 0 and metrics["clothing_last80_slope"] <= 0
    total_down = metrics["total_last40"] < metrics["total_first20"]
    maximum_bound_hit = max(float(item["bound_hit_fraction"]) for item in residual["channels"].values())
    anomaly_free = maximum_bound_hit < 0.99
    common = finite and protected_ok and preserve_ok and slopes_ok and total_down and anomaly_free
    if common and metrics["edit_reduction"] >= float(acceptance["pass_reduction_min"]) and metrics["clothing_reduction"] >= float(acceptance["pass_reduction_min"]):
        status = "CAPACITY_NUMERIC_PASS"
    elif common and metrics["edit_reduction"] >= float(acceptance["warn_reduction_min"]) and metrics["clothing_reduction"] >= float(acceptance["warn_reduction_min"]):
        status = "CAPACITY_NUMERIC_WARN"
    else:
        status = "CAPACITY_NUMERIC_FAIL"
    return {
        "status": status,
        "metrics": metrics,
        "protected_final": protected_final,
        "preserve_final": preserve_final,
        "finite": finite,
        "protected_ok": protected_ok,
        "preserve_ok": preserve_ok,
        "slopes_nonpositive": slopes_ok,
        "total_decreased": total_down,
        "maximum_bound_hit_fraction": maximum_bound_hit,
        "numerical_anomaly_free": anomaly_free,
    }


def run_smoke(args: argparse.Namespace, gate: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    destination = args.output.resolve() / "oracle" / "technical_smoke"
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    atomic_json(destination / "status.json", {"status": "RUNNING", "optimizer_steps": 0})
    device = torch.device(args.device)
    pipeline = training.load_config(args.pipeline_config)
    base = training.load_frozen_mmlphuman_base(pipeline["base"]["model_dir"], pipeline["base"]["checkpoint_path"], device)
    if int(base._xyz.shape[0]) != 200000:
        raise ValueError("technical smoke requires the frozen 200k base")
    samples = _load_samples(args.manifest, "O01")
    transitions = _transition_targets(samples, device)
    oracle = FixedOpenGaussianOracle(base, bounds=config["oracle"]["bounds"]).to(device)
    oracle.configure_stage(3)
    contract = fixed_open_oracle_contract(oracle)
    if not contract["pass"]:
        raise RuntimeError("fixed-open contract failed")
    optim, names = optimizer(oracle, config)
    background = torch.tensor(pipeline["render"]["background"], device=device, dtype=base._xyz.dtype)
    base_before = _tensor_state_fingerprint(_base_named_tensors(base))
    cap = _compute_transition_cap(base, oracle, samples[CONDITIONS[0]], transitions[CONDITIONS[0]], background, device, config)
    coefficient = float(cap["final_frozen_coefficient"])
    oracle.configure_stage(3)
    optim.zero_grad(set_to_none=True)
    output = oracle(base)
    from tools.run_module4b_canonical_oracle_micropilot import _render  # noqa: E402
    rendered_rgb, rendered_alpha = _render(base, samples[CONDITIONS[0]], output.canonical_overrides, background)
    parts = _loss(rendered_rgb, rendered_alpha, samples[CONDITIONS[0]], transitions[CONDITIONS[0]], device, config, coefficient)
    regularizer, _ = _regularization(output, config)
    objective = parts["total"] + regularizer
    objective.backward()
    gradients = {
        name: {
            "present": value.grad is not None,
            "finite": bool(value.grad is not None and torch.isfinite(value.grad).all()),
            "nonzero": bool(value.grad is not None and torch.count_nonzero(value.grad).item() > 0),
        }
        for name, value in oracle.named_parameters()
    }
    torch.nn.utils.clip_grad_norm_([value for value in oracle.parameters() if value.requires_grad], float(config["oracle"]["optimizer"]["gradient_clip_norm"]))
    optim.step()
    # Persist the checkpoint in the formal step-1 schedule state. The smoke
    # used stage 3 only to exercise all five enabled residual channels once.
    oracle.configure_stage(1)
    checkpoint = destination / "discarded_smoke_checkpoint.pth"
    save_checkpoint(checkpoint, oracle, optim, names, 1, args.manifest, "O01", config, coefficient)
    state_before = _tensor_state_fingerprint(oracle.state_dict().items())
    replacement, _, _, payload = load_checkpoint_exact(checkpoint, base, config, args.manifest, "O01", 1, coefficient, device)
    state_after = _tensor_state_fingerprint(replacement.state_dict().items())
    base_after = _tensor_state_fingerprint(_base_named_tensors(base))
    required = ("raw_xyz", "raw_log_scaling", "raw_rotvec", "raw_opacity", "raw_sh0")
    passed = (
        torch.isfinite(objective)
        and all(gradients[name]["finite"] and gradients[name]["nonzero"] for name in required)
        and gradients["raw_shN"]["present"] is False
        and base_before == base_after
        and _base_gradient_count(base) == 0
        and len(contract["gate_parameter_names"]) == 0
        and state_before == state_after
        and int(payload["training_state"]["global_step"]) == 1
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "optimizer_steps": 1,
        "discarded_update": True,
        "rgb_shape": list(rendered_rgb.shape),
        "alpha_shape": list(rendered_alpha.shape),
        "objective": float(objective.detach()),
        "gradients": gradients,
        "base_bitwise_exact": base_before == base_after,
        "base_gradient_count": _base_gradient_count(base),
        "mmlphuman_gradient_count": _base_gradient_count(base),
        "gate_parameter_count": len(contract["gate_parameter_names"]),
        "SHN_enabled": False,
        "checkpoint_state_exact": state_before == state_after,
        "checkpoint_global_step_exact": int(payload["training_state"]["global_step"]) == 1,
        "V5_3_loss": True,
        "R2_rotation_path": contract["rotation_path"],
    }
    atomic_json(destination / "smoke_result.json", result)
    atomic_json(destination / "status.json", result)
    if not passed:
        raise RuntimeError(f"Oracle technical smoke failed: {result}")
    print(json.dumps(result, indent=2))


def run_outfit(args: argparse.Namespace, gate: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    if args.outfit not in OUTFITS:
        raise ValueError("outfit is outside the preregistered candidate set")
    root = args.output.resolve()
    smoke = json.loads((root / "oracle/technical_smoke/status.json").read_text(encoding="utf-8"))
    if smoke.get("status") != "PASS":
        raise RuntimeError("technical smoke PASS is mandatory")
    run_dir = root / "oracle" / args.outfit
    if run_dir.exists():
        raise FileExistsError(run_dir)
    for name in ("checkpoints", "renders", "diagnostic_panels", "final"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "run_status.json"
    atomic_json(status_path, {"status": "RUNNING", "optimizer_steps": 0})
    started = time.perf_counter()
    try:
        device = torch.device(args.device)
        pipeline = training.load_config(args.pipeline_config)
        base = training.load_frozen_mmlphuman_base(pipeline["base"]["model_dir"], pipeline["base"]["checkpoint_path"], device)
        if int(base._xyz.shape[0]) != 200000:
            raise ValueError("formal Oracle requires the original 200k base only")
        samples = _load_samples(args.manifest, args.outfit)
        transitions = _transition_targets(samples, device)
        oracle = FixedOpenGaussianOracle(base, bounds=config["oracle"]["bounds"]).to(device)
        contract = fixed_open_oracle_contract(oracle)
        if not contract["pass"] or oracle.element_count != 200000:
            raise RuntimeError("formal fixed-open Oracle contract failed")
        optim, group_names = optimizer(oracle, config)
        background = torch.tensor(pipeline["render"]["background"], device=device, dtype=base._xyz.dtype)
        base_before = _tensor_state_fingerprint(_base_named_tensors(base))
        initial = {name: value.detach().cpu().clone() for name, value in oracle.named_parameters()}
        gradient_seen = {name: 0 for name, _ in oracle.named_parameters()}
        cap = _compute_transition_cap(base, oracle, samples[CONDITIONS[0]], transitions[CONDITIONS[0]], background, device, config)
        coefficient = float(cap["final_frozen_coefficient"])
        atomic_json(run_dir / "config.json", {"gate": gate, "runtime": config, "transition_gradient_cap": cap})
        atomic_json(run_dir / "manifest.json", {
            "schema_version": SCHEMA,
            "git": _git_state(),
            "environment": _environment(),
            "outfit_id": args.outfit,
            "conditions": [{"condition_id": item, "view": VIEWS[item]} for item in CONDITIONS],
            "dataset_manifest": {"path": str(args.manifest.resolve()), "sha256": sha256(args.manifest)},
            "pipeline_config": {"path": str(args.pipeline_config.resolve()), "sha256": sha256(args.pipeline_config)},
            "base_gaussian_count": 200000,
            "shared_canonical_field": True,
            "condition_specific_residuals": False,
            "target_fields_loss_only": True,
            "image_conditioning_used": False,
            "teacher_used": False,
            "oracle_contract": contract,
        })
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        history: list[dict[str, Any]] = []
        per_view: dict[str, Any] = {}

        def record(step: int, condition: str) -> None:
            oracle.eval()
            output = oracle(base)
            from tools.run_module4b_canonical_oracle_micropilot import _render, _custom_metrics  # noqa: E402
            rendered_rgb, rendered_alpha = _render(base, samples[condition], output.canonical_overrides, background)
            parts = _loss(rendered_rgb, rendered_alpha, samples[condition], transitions[condition], device, config, coefficient)
            regularizer, regularizer_parts = _regularization(output, config)
            row = {
                "step": step,
                "stage": oracle.stage,
                "condition_id": condition,
                "view": VIEWS[condition],
                **{name: float(value.detach()) for name, value in parts.items()},
                "regularization": float(regularizer.detach()),
                "objective": float((parts["total"] + regularizer).detach()),
                **{f"regularizer_{name}": value for name, value in regularizer_parts.items()},
                **_custom_metrics(rendered_rgb, rendered_alpha, samples[condition]),
            }
            if not all(math.isfinite(float(value)) for value in row.values() if isinstance(value, (int, float))):
                raise FloatingPointError(f"non-finite metric at step {step}")
            history.append(row)
            append_jsonl(run_dir / "state_records.jsonl", row)

        def milestone(step: int) -> None:
            rows, output, panels = _evaluate_views(
                base, oracle, samples, transitions, background, device, config, coefficient,
                args.outfit, run_dir / "renders" / f"step_{step:06d}",
            )
            per_view[str(step)] = {"conditions": rows, "aggregate": _aggregate_views(rows)}
            _grid(run_dir / "diagnostic_panels" / f"step_{step:06d}_four_view.png", panels, columns=4)
            _save_field_panel(run_dir / "diagnostic_panels" / f"step_{step:06d}_residual_fields.png", base, output, "gaussian")
            atomic_json(run_dir / "diagnostic_panels" / f"step_{step:06d}_residuals.json", _residual_statistics(output, oracle, config, base))

        oracle.configure_stage(0)
        record(0, CONDITIONS[0])
        milestone(0)
        save_checkpoint(run_dir / "checkpoints/step_000000.pth", oracle, optim, group_names, 0, args.manifest, args.outfit, config, coefficient)
        for step in range(1, 481):
            oracle.configure_stage(fixed_open_stage_for_step(step))
            oracle.train()
            condition = CONDITIONS[(step - 1) % 4]
            optim.zero_grad(set_to_none=True)
            output = oracle(base)
            from tools.run_module4b_canonical_oracle_micropilot import _render  # noqa: E402
            rendered_rgb, rendered_alpha = _render(base, samples[condition], output.canonical_overrides, background)
            parts = _loss(rendered_rgb, rendered_alpha, samples[condition], transitions[condition], device, config, coefficient)
            regularizer, _ = _regularization(output, config)
            objective = parts["total"] + regularizer
            if not torch.isfinite(objective):
                raise FloatingPointError(f"non-finite objective at step {step}")
            objective.backward()
            for name, value in oracle.named_parameters():
                if value.grad is not None:
                    if not torch.isfinite(value.grad).all():
                        raise FloatingPointError(f"non-finite gradient in {name} at step {step}")
                    if torch.count_nonzero(value.grad).item() > 0:
                        gradient_seen[name] += 1
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [value for value in oracle.parameters() if value.requires_grad],
                float(config["oracle"]["optimizer"]["gradient_clip_norm"]),
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"non-finite clipped gradient at step {step}")
            optim.step()
            record(step, condition)
            if step in LOG_STEPS:
                save_checkpoint(run_dir / f"checkpoints/step_{step:06d}.pth", oracle, optim, group_names, step, args.manifest, args.outfit, config, coefficient)
            if step in RENDER_STEPS:
                milestone(step)
            atomic_json(status_path, {"status": "RUNNING", "optimizer_steps": step, "last_condition": condition})

        final_output = oracle(base)
        residual = _residual_statistics(final_output, oracle, config, base)
        numeric = numeric_adjudication(history, residual, gate["numeric_acceptance"])
        base_after = _tensor_state_fingerprint(_base_named_tensors(base))
        parameter_diagnostics = {}
        for name, value in oracle.named_parameters():
            delta = (value.detach().cpu() - initial[name]).abs()
            parameter_diagnostics[name] = {
                "gradient_steps_nonzero": gradient_seen[name],
                "max_abs_update": float(delta.max()),
                "mean_abs_update": float(delta.mean()),
            }
        required = ("raw_xyz", "raw_log_scaling", "raw_rotvec", "raw_opacity", "raw_sh0")
        gradient_pass = all(gradient_seen[name] > 0 for name in required) and gradient_seen["raw_shN"] == 0
        frozen = {
            "base_fingerprint_before": base_before,
            "base_fingerprint_after": base_after,
            "base_bitwise_exact": base_before == base_after,
            "base_gradient_count": _base_gradient_count(base),
            "mmlphuman_gradient_count": _base_gradient_count(base),
            "image_backbone_instantiated": False,
        }
        metrics = {
            "numeric": numeric,
            "per_view_metrics": per_view,
            "residual_statistics": residual,
            "parameter_diagnostics": parameter_diagnostics,
            "gradient_contract_pass": gradient_pass,
            "frozen": frozen,
            "optimizer_steps": 480,
            "per_condition_update_count": {condition: 120 for condition in CONDITIONS},
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            "visual_status": "PENDING_ACTUAL_IMAGE_INSPECTION",
        }
        write_csv(run_dir / "metrics.csv", history)
        per_view_rows = []
        for step, item in per_view.items():
            for row in item["conditions"]:
                per_view_rows.append({"step": int(step), **row})
        write_csv(run_dir / "per_view_metrics.csv", per_view_rows)
        atomic_json(run_dir / "metrics.json", metrics)
        panels = [
            (f"step {step}", Image.open(run_dir / "diagnostic_panels" / f"step_{step:06d}_four_view.png").convert("RGB"))
            for step in RENDER_STEPS
        ]
        _grid(run_dir / "diagnostic_panels" / "oracle_training_contact_sheet.png", panels, columns=2, cell=(1024, 720))
        infrastructure_pass = frozen["base_bitwise_exact"] and frozen["base_gradient_count"] == 0 and gradient_pass
        preliminary = {
            "status": "ORACLE_COMPLETE_PENDING_VISUAL",
            "optimizer_steps": 480,
            "numeric_status": numeric["status"],
            "infrastructure_pass": infrastructure_pass,
            "visual_status": "PENDING_ACTUAL_IMAGE_INSPECTION",
        }
        atomic_json(run_dir / "FINAL_STATUS.json", preliminary)
        atomic_text(run_dir / "FINAL_STATUS.md", "\n".join([
            f"# {args.outfit} Fixed-Open Gaussian Oracle", "",
            f"- numeric status: **{numeric['status']}**",
            f"- edit reduction: `{numeric['metrics']['edit_reduction']:.6f}`",
            f"- clothing reduction: `{numeric['metrics']['clothing_reduction']:.6f}`",
            f"- base bitwise exact: `{frozen['base_bitwise_exact']}`",
            "- visual status: `PENDING_ACTUAL_IMAGE_INSPECTION`",
        ]))
        atomic_json(status_path, preliminary)
        print(json.dumps(preliminary, indent=2))
    except Exception as error:
        atomic_json(status_path, {
            "status": "FAILED",
            "optimizer_steps": json.loads(status_path.read_text(encoding="utf-8")).get("optimizer_steps", 0),
            "exception_type": type(error).__name__,
            "exception": str(error),
            "traceback": traceback.format_exc(),
        })
        raise


def finalize(args: argparse.Namespace, gate: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    root = args.output.resolve()
    decisions = json.loads(args.visual_decisions.read_text(encoding="utf-8"))
    if set(decisions) != set(OUTFITS):
        raise ValueError("visual decisions must cover all seven outfits")
    support_summary = json.loads((root / "support_compatibility/support_compatibility_summary.json").read_text(encoding="utf-8"))
    if support_summary.get("status") != "COMPLETE":
        raise RuntimeError("support audit is not finalized")
    support = {item["outfit_id"]: item for item in support_summary["outfits"]}
    checker = json.loads((root / "checker/aaai_gate_28_checker.json").read_text(encoding="utf-8"))
    if checker.get("status") != "PASS":
        raise RuntimeError("AAAI_GATE_28 checker is not PASS")
    generation = json.loads(args.generation_status.read_text(encoding="utf-8"))
    if generation.get("status") != "PASS" or int(generation.get("record_count", 0)) != 28:
        raise RuntimeError("generation gate is not PASS 28/28")
    rows, outfit_status = [], {}
    for outfit in OUTFITS:
        decision = decisions[outfit]
        visual = decision.get("status")
        if visual not in {"CAPACITY_VISUAL_PASS", "CAPACITY_VISUAL_WARN", "CAPACITY_VISUAL_FAIL"}:
            raise ValueError(f"invalid visual decision: {outfit}")
        if not decision.get("images_actually_opened") or not decision.get("observations"):
            raise ValueError(f"visual evidence missing: {outfit}")
        run_dir = root / "oracle" / outfit
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        numeric = metrics["numeric"]["status"]
        risk = support[outfit]["risk"]
        structural_failure = bool(decision.get("missing_body_support") or decision.get("identity_contamination"))
        infrastructure_pass = metrics["frozen"]["base_bitwise_exact"] and metrics["frozen"]["base_gradient_count"] == 0 and metrics["gradient_contract_pass"]
        if numeric == "CAPACITY_NUMERIC_PASS" and visual in {"CAPACITY_VISUAL_PASS", "CAPACITY_VISUAL_WARN"} and risk in {"LOW_SUPPORT_RISK", "MEDIUM_SUPPORT_RISK"} and not structural_failure and infrastructure_pass:
            final = "OUTFIT_GATE_PASS"
        elif numeric in {"CAPACITY_NUMERIC_PASS", "CAPACITY_NUMERIC_WARN"} and visual in {"CAPACITY_VISUAL_PASS", "CAPACITY_VISUAL_WARN"} and risk in {"LOW_SUPPORT_RISK", "MEDIUM_SUPPORT_RISK"} and not structural_failure and infrastructure_pass:
            final = "OUTFIT_GATE_RESERVE"
        else:
            final = "OUTFIT_GATE_FAIL"
        outfit_status[outfit] = final
        item = {
            "outfit_id": outfit,
            "support_risk": risk,
            "numeric_status": numeric,
            "visual_status": visual,
            "final_status": final,
            "edit_reduction": metrics["numeric"]["metrics"]["edit_reduction"],
            "clothing_reduction": metrics["numeric"]["metrics"]["clothing_reduction"],
            "edit_last80_slope": metrics["numeric"]["metrics"]["edit_last80_slope"],
            "clothing_last80_slope": metrics["numeric"]["metrics"]["clothing_last80_slope"],
            "protected_final": metrics["numeric"]["protected_final"],
            "missing_body_support": bool(decision.get("missing_body_support")),
            "identity_contamination": bool(decision.get("identity_contamination")),
            "observations": decision["observations"],
        }
        rows.append(item)
        metrics["visual_acceptance"] = decision
        metrics["visual_status"] = visual
        atomic_json(run_dir / "metrics.json", metrics)
        atomic_json(run_dir / "FINAL_STATUS.json", item)
        atomic_text(run_dir / "FINAL_STATUS.md", "\n".join([
            f"# {outfit} Fixed-Open Gaussian Oracle", "",
            f"- final outfit gate: **{final}**",
            f"- numeric / visual / support: `{numeric}` / `{visual}` / `{risk}`",
            f"- edit / clothing reduction: `{item['edit_reduction']:.6f}` / `{item['clothing_reduction']:.6f}`",
            f"- observation: {item['observations']}",
        ]))
    selection = select_benchmark(outfit_status)
    selection.update({
        "schema_version": "canondressgs.aaai27.outfit_selection.v1",
        "outfit_status": outfit_status,
        "generation_provider": "CODEX_IMAGE_GENERATION_SKILL",
        "formal_training_allowed": False,
        "full_benchmark_generation_allowed": selection["status"] == "GO",
    })
    selection_dir = root / "selection"
    write_csv(root / "aaai_outfit_capacity_summary.csv", rows)
    write_csv(root / "aaai_outfit_support_risk_summary.csv", [
        {"outfit_id": outfit, **{key: value for key, value in support[outfit].items() if key != "outfit_id"}}
        for outfit in OUTFITS
    ])
    write_csv(root / "aaai_outfit_visual_acceptance.csv", [
        {"outfit_id": outfit, **decisions[outfit]} for outfit in OUTFITS
    ])
    atomic_json(selection_dir / "AAAI27_OUTFIT_SELECTION_FINAL.json", selection)
    atomic_text(selection_dir / "AAAI27_OUTFIT_SELECTION_ADJUDICATION.md", "\n".join([
        "# AAAI-27 Outfit Selection Adjudication", "",
        f"- decision: **{selection['status']}**",
        f"- train outfits: `{selection['train_outfits']}`",
        f"- unseen outfits: `{selection['unseen_outfits']}`",
        f"- target count: `{selection['target_count']}`",
        "",
        *[f"- {row['outfit_id']}: {row['final_status']} ({row['numeric_status']}, {row['visual_status']}, {row['support_risk']})" for row in rows],
    ]))
    if selection["status"] == "GO":
        full = json.loads(args.full_target_manifest.read_text(encoding="utf-8"))
        selected = set(selection["selected_outfits"])
        full["provisional_outfits"] = selection["selected_outfits"]
        full["train_outfits"] = selection["train_outfits"]
        full["unseen_test_outfits"] = selection["unseen_outfits"]
        full["record_count"] = selection["target_count"]
        full["records"] = [record for record in full["records"] if record["outfit_id"] in selected]
        for record in full["records"]:
            record["status"] = "PLANNED_NOT_GENERATED"
            record["outfit_split"] = "train" if record["outfit_id"] in selection["train_outfits"] else "unseen_test"
            record["planned_mask_contract"] = "v5_3_region_aware_dual_target"
            record["no_target_forward_condition"] = True
        if len(full["records"]) != selection["target_count"]:
            raise AssertionError("updated full manifest record count mismatch")
        atomic_json(selection_dir / "updated_full_target_manifest.json", full)
    final = {
        "schema_version": "canondressgs.aaai27.data_capacity_gate.final.v1",
        "status": "PASS" if selection["status"] == "GO" else "FAIL",
        "go_no_go": selection["status"],
        "outfits": rows,
        "selection": selection,
        "formal_training_started": False,
        "remaining_targets_generated": 0,
    }
    atomic_json(root / "final_adjudication/AAAI27_28_IMAGE_GATE_FINAL_STATUS.json", final)
    atomic_text(root / "final_adjudication/AAAI27_28_IMAGE_GATE_FINAL_ADJUDICATION.md", "\n".join([
        "# AAAI-27 28-Image Data and Capacity Gate", "",
        f"- final decision: **{selection['status']}**",
        f"- benchmark size: `{selection['target_count']}` planned targets",
        "- remaining target generation: `not executed`",
        "- formal image-conditioned training: `not started`",
    ]))
    print(json.dumps(final, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AAAI fixed-open Gaussian representation-capacity gate")
    parser.add_argument("--phase", required=True, choices=("smoke", "run", "finalize"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--oracle-template", type=Path, required=True)
    parser.add_argument("--pipeline-config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--outfit")
    parser.add_argument("--visual-decisions", type=Path)
    parser.add_argument("--generation-status", type=Path)
    parser.add_argument("--full-target-manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gate, config = load_config(args.config.resolve(), args.oracle_template.resolve())
    seed = int(gate["oracle"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if args.phase == "smoke":
        run_smoke(args, gate, config)
    elif args.phase == "run":
        run_outfit(args, gate, config)
    else:
        finalize(args, gate, config)


if __name__ == "__main__":
    main()
