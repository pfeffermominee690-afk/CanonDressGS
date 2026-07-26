from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import traceback
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_dressable as training  # noqa: E402
from scene.gaussian_clothing_residuals import CanonicalGaussianOverrides  # noqa: E402
from tools.run_module4b_canonical_oracle_micropilot import (  # noqa: E402
    _base_named_tensors,
    _git_state,
    _load_samples,
    _loss,
    _regularization,
    _render,
    _sha256,
    _tensor_state_fingerprint,
    _transition_targets,
    build_oracle,
    load_contract,
)


SCHEMA = "canondressgs.rotation_autograd_closure_r2.v1"


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def _gradient_record(parameter: torch.Tensor) -> dict[str, Any]:
    gradient = parameter.grad
    return {
        "requires_grad": bool(parameter.requires_grad),
        "gradient_present": gradient is not None,
        "gradient_finite": bool(gradient is not None and torch.isfinite(gradient).all()),
        "gradient_norm": None if gradient is None else float(torch.linalg.vector_norm(gradient.detach().float())),
        "gradient_nonzero_count": 0 if gradient is None else int(torch.count_nonzero(gradient)),
    }


def test_zero_initialized_rotvec_receives_renderer_gradient(gradient: torch.Tensor) -> None:
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def _run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"R2 smoke output already exists: {output}")
    output.mkdir(parents=True)
    _write_json(output / "run_status.json", {"status": "RUNNING", "optimizer_steps": 0})

    config = load_contract(args.config)
    device = torch.device(args.device)
    pipeline = training.load_config(args.pipeline_config)
    base = training.load_frozen_mmlphuman_base(
        pipeline["base"]["model_dir"], pipeline["base"]["checkpoint_path"], device,
    )
    samples = _load_samples(args.manifest, "O00")
    if args.condition not in samples:
        raise KeyError(f"condition is not in the fixed O00 protocol: {args.condition}")
    sample = samples[args.condition]
    transitions = _transition_targets(samples, device)
    background = torch.tensor(pipeline["render"]["background"], device=device, dtype=base._xyz.dtype)
    cap_path = args.module4b_root.resolve() / "O00/gaussian_oracle/transition_gradient_cap.json"
    cap = json.loads(cap_path.read_text(encoding="utf-8"))
    coefficient = float(cap["final_frozen_coefficient"])

    before = _tensor_state_fingerprint(_base_named_tensors(base))
    oracle = build_oracle("gaussian", base, config, device).to(device)
    oracle.configure_stage(2)
    oracle.zero_grad(set_to_none=True)
    output_state = oracle(base)
    if torch.count_nonzero(output_state.raw_residuals.delta_rotvec) != 0:
        raise RuntimeError("R2 smoke requires exact zero raw rotation initialization")
    if output_state.canonical_overrides.rotation.grad_fn is None:
        raise RuntimeError("composed rotation has no grad_fn at zero initialization")

    direct_base = CanonicalGaussianOverrides(
        xyz=base._xyz,
        scaling=base._scaling,
        rotation=base._rotation,
        opacity=base._opacity,
        sh0=base._sh0,
        shN=base._shN,
    ).validate(base)
    base_rgb, base_alpha = _render(base, sample, direct_base, background)
    rgb, alpha = _render(base, sample, output_state.canonical_overrides, background)
    normalized_base_rotation = torch.nn.functional.normalize(base._rotation, dim=-1)
    quaternion_max_abs = float((output_state.canonical_overrides.rotation - normalized_base_rotation).abs().max())
    rgb_difference = (rgb - base_rgb).abs()
    alpha_difference = (alpha - base_alpha).abs()

    height, width = rgb.shape[-2:]
    x_weight = torch.linspace(-1.0, 1.0, width, device=device, dtype=rgb.dtype).reshape(1, 1, width)
    y_weight = torch.linspace(-0.5, 1.5, height, device=device, dtype=rgb.dtype).reshape(1, height, 1)
    renderer_probe = (rgb * x_weight).mean() + (alpha * y_weight).mean()
    renderer_rotation_gradient = torch.autograd.grad(
        renderer_probe, oracle.raw_rotvec, retain_graph=True, allow_unused=False,
    )[0]
    test_zero_initialized_rotvec_receives_renderer_gradient(renderer_rotation_gradient)

    parts = _loss(
        rgb, alpha, sample, transitions[args.condition], device, config, coefficient,
    )
    regularizer, regularizer_parts = _regularization(output_state, config)
    objective = parts["total"] + regularizer
    if not torch.isfinite(objective):
        raise FloatingPointError("formal V5.3 objective is non-finite")
    objective.backward()

    gradients = {
        "rotation": _gradient_record(oracle.raw_rotvec),
        "xyz": _gradient_record(oracle.raw_xyz),
        "scaling": _gradient_record(oracle.raw_log_scaling),
        "opacity": _gradient_record(oracle.raw_opacity),
        "sh0": _gradient_record(oracle.raw_sh0),
        "shN": _gradient_record(oracle.raw_shN),
        "geometry_gate": _gradient_record(oracle.geometry_gate_logits),
        "appearance_gate": _gradient_record(oracle.appearance_gate_logits),
    }
    base_gradients = [
        name for name, value in _base_named_tensors(base)
        if value.is_leaf and value.grad is not None and torch.count_nonzero(value.grad) > 0
    ]
    after = _tensor_state_fingerprint(_base_named_tensors(base))
    renderer_gradient_norm = float(torch.linalg.vector_norm(renderer_rotation_gradient.detach().float()))
    forward_regression = {
        "quaternion_vs_normalized_base_max_abs": quaternion_max_abs,
        "rgb_mean_abs": float(rgb_difference.mean()),
        "rgb_max_abs": float(rgb_difference.max()),
        "alpha_mean_abs": float(alpha_difference.mean()),
        "alpha_max_abs": float(alpha_difference.max()),
        "renderer_mean_tolerance": 1e-3,
        "pass": bool(
            quaternion_max_abs <= 1e-6
            and float(rgb_difference.mean()) <= 1e-3
            and float(alpha_difference.mean()) <= 1e-3
        ),
    }
    passed = bool(
        rgb.grad_fn is not None
        and alpha.grad_fn is not None
        and renderer_gradient_norm > 0
        and gradients["rotation"]["gradient_present"]
        and gradients["rotation"]["gradient_finite"]
        and gradients["rotation"]["gradient_nonzero_count"] > 0
        and not base_gradients
        and before == after
        and forward_regression["pass"]
    )
    result = {
        "schema_version": SCHEMA,
        "status": "PASS" if passed else "FAIL",
        "git": _git_state(),
        "condition": {"outfit_id": "O00", "condition_id": args.condition},
        "inputs": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": _sha256(args.manifest),
            "pipeline_config": str(args.pipeline_config.resolve()),
            "pipeline_config_sha256": _sha256(args.pipeline_config),
            "oracle_config": str(args.config.resolve()),
            "oracle_config_sha256": _sha256(args.config),
            "transition_cap": str(cap_path),
            "transition_cap_sha256": _sha256(cap_path),
        },
        "optimizer_created": False,
        "optimizer_steps": 0,
        "raw_rotvec_exact_zero": True,
        "rotation_parameter_requires_grad": bool(oracle.raw_rotvec.requires_grad),
        "composed_rotation_grad_fn": type(output_state.canonical_overrides.rotation.grad_fn).__name__,
        "raw_render_rgb_grad_fn": type(rgb.grad_fn).__name__,
        "raw_render_alpha_grad_fn": type(alpha.grad_fn).__name__,
        "renderer_probe_rotation_gradient_norm": renderer_gradient_norm,
        "renderer_probe_rotation_gradient_nonzero_count": int(torch.count_nonzero(renderer_rotation_gradient)),
        "formal_v5_3_loss": {name: float(value.detach()) for name, value in parts.items()},
        "regularization": float(regularizer.detach()),
        "regularization_parts": regularizer_parts,
        "objective": float(objective.detach()),
        "gradients": gradients,
        "base_nonzero_gradient_names": base_gradients,
        "base_fingerprint_before": before,
        "base_fingerprint_after": after,
        "base_bitwise_exact": before == after,
        "forward_regression": forward_regression,
        "classification": "PATH_FIXED_REAL_BATCH_GRADIENT_NONZERO" if gradients["rotation"]["gradient_nonzero_count"] > 0 else "PATH_FIXED_REAL_BATCH_LOW_SENSITIVITY",
    }
    _write_json(output / "rotation_real_smoke_r2.json", result)
    _write_text(output / "rotation_real_smoke_r2.md", "\n".join([
        "# Rotation Real Smoke R2", "",
        f"- Status: **{result['status']}**.",
        f"- Condition: `O00/{args.condition}`.",
        "- Formal V5.3 loss and renderer were used; no optimizer was created and optimizer steps are zero.",
        f"- Composed rotation grad_fn: `{result['composed_rotation_grad_fn']}`.",
        f"- Renderer probe rotation gradient norm: `{renderer_gradient_norm:.12g}`.",
        f"- Formal V5.3 rotation gradient norm: `{gradients['rotation']['gradient_norm']}`.",
        f"- Base bitwise exact / nonzero gradients: `{before == after}` / `{base_gradients}`.",
        f"- Zero-forward regression: `{json.dumps(forward_regression, sort_keys=True)}`.",
        f"- Classification: `{result['classification']}`.",
    ]))
    _write_json(output / "run_status.json", {
        "status": "COMPLETE", "final_status": result["status"], "optimizer_steps": 0,
    })
    if not passed:
        raise RuntimeError(f"R2 real smoke failed: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Real O00 zero-rotation autograd smoke for R2 closure")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--module4b-root", type=Path, required=True)
    parser.add_argument("--condition", default="cond_000000")
    parser.add_argument("--manifest", type=Path, default=Path("/root/autodl-tmp/canondressgs_work/data/subject02_dual_target_v5_2/pilot_manifest_full_v1_v5_2.json"))
    parser.add_argument("--pipeline-config", type=Path, default=PROJECT_ROOT / "configs/canon_dress_gs_mvp_real.yaml")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/oracle/module4b_canonical_capacity_v1.yaml")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    try:
        _run(args)
    except Exception as error:
        if args.output.exists():
            _write_json(args.output / "run_status.json", {
                "status": "FAILED", "final_status": "FAIL", "optimizer_steps": 0,
                "exception_type": type(error).__name__, "exception": str(error),
                "traceback": traceback.format_exc(),
            })
        raise


if __name__ == "__main__":
    main()
