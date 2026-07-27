from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.check_oracle_interrupted_resume import Fixture
from utils.full_training_checkpoint_utils import load_full_training_checkpoint


PARAMETER_CHANNELS = {
    "raw_xyz": "xyz", "raw_log_scaling": "scaling", "raw_rotvec": "rotation",
    "raw_opacity": "opacity", "raw_sh0": "sh0", "raw_shN": "shN",
    "geometry_gate_logits": "geometry_gate", "appearance_gate_logits": "appearance_gate",
}


def numeric_stats(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    a, b = left.detach().float().cpu(), right.detach().float().cpu()
    difference = (a - b).abs().reshape(-1)
    threshold = 0.0
    l2 = torch.linalg.vector_norm(difference)
    reference_l2 = torch.linalg.vector_norm(a.reshape(-1)).clamp_min(1e-30)
    quantiles = torch.quantile(difference, torch.tensor([.5, .95, .99])) if difference.numel() else torch.zeros(3)
    differing = difference > threshold
    return {
        "max_abs_diff": float(difference.max()) if difference.numel() else 0.0,
        "mean_abs_diff": float(difference.mean()) if difference.numel() else 0.0,
        "median_abs_diff": float(quantiles[0]), "p95_abs_diff": float(quantiles[1]),
        "p99_abs_diff": float(quantiles[2]), "l2_diff": float(l2),
        "relative_l2_diff": float(l2 / reference_l2),
        "differing_element_count": int(differing.sum()), "total_element_count": difference.numel(),
        "differing_fraction": float(differing.float().mean()) if difference.numel() else 0.0,
        "bitwise_equal": bool(torch.equal(a, b)),
    }


def flatten_tensors(value: Any, path: str = "", result: dict[str, torch.Tensor] | None = None):
    result = {} if result is None else result
    if isinstance(value, torch.Tensor): result[path] = value
    elif isinstance(value, dict):
        for key, item in value.items(): flatten_tensors(item, f"{path}.{key}".strip("."), result)
    elif isinstance(value, list):
        for index, item in enumerate(value): flatten_tensors(item, f"{path}[{index}]", result)
    return result


def compare_results(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    a, b = flatten_tensors(left), flatten_tensors(right)
    common = sorted(set(a).intersection(b))
    return {name: numeric_stats(a[name], b[name]) for name in common}


def summarize_parameter_difference(left, right):
    rows = {}
    for name in PARAMETER_CHANNELS:
        rows[PARAMETER_CHANNELS[name]] = numeric_stats(left["parameters"][name], right["parameters"][name])
    all_left = torch.cat([left["parameters"][name].reshape(-1).float().cpu() for name in PARAMETER_CHANNELS])
    all_right = torch.cat([right["parameters"][name].reshape(-1).float().cpu() for name in PARAMETER_CHANNELS])
    rows["all_parameters"] = numeric_stats(all_left, all_right)
    return rows


def load_fixture(args, kind):
    fixture = Fixture(args, kind)
    load_full_training_checkpoint(
        args.resume_source, model=fixture.model, optimizer=fixture.optimizer,
        optimizer_group_names=fixture.group_names, scheduler=None, scaler=None,
        expected_method_state=fixture.method_state, expected_data_state=fixture.data_state,
    )
    return fixture


def non_raster_step(fixture: Fixture):
    fixture.optimizer.zero_grad(set_to_none=True)
    output = fixture.model(fixture.base)
    target = fixture.teacher_gaussian if fixture.kind == "gaussian" else fixture.teacher_anchor
    supervision = sum(F.smooth_l1_loss(getattr(output.raw_residuals, name), getattr(target, name)) for name in (
        "delta_xyz", "delta_log_scaling", "delta_rotvec", "delta_opacity_logit", "delta_sh0", "delta_shN",
    ))
    active = ((target.delta_xyz.reshape(target.delta_xyz.shape[0], -1).abs().sum(1) + target.delta_sh0.reshape(target.delta_sh0.shape[0], -1).abs().sum(1)) > 1e-8).float().reshape(-1, 1)
    gate = F.binary_cross_entropy(output.geometry_gate.clamp(1e-6, 1-1e-6), active) + F.binary_cross_entropy(output.appearance_gate.clamp(1e-6, 1-1e-6), active)
    regularization = (
        sum(output.regularization.residual_magnitude.values())
        + output.regularization.geometry_gate_sparsity + output.regularization.appearance_gate_sparsity
        + output.regularization.gate_binary + output.regularization.graph_gate_smoothness
        + output.regularization.graph_residual_smoothness
    )
    total = 5 * supervision + .1 * gate + 1e-4 * regularization
    pre = {
        "raw": {name: getattr(output.raw_residuals, name).detach().cpu() for name in (
            "delta_xyz", "delta_log_scaling", "delta_rotvec", "delta_opacity_logit", "delta_sh0", "delta_shN",
        )},
        "geometry_gate": output.geometry_gate.detach().cpu(), "appearance_gate": output.appearance_gate.detach().cpu(),
        "loss": total.detach().cpu(),
    }
    total.backward()
    gradients = {name: None if parameter.grad is None else parameter.grad.detach().cpu() for name, parameter in fixture.model.named_parameters()}
    fixture.optimizer.step()
    post = {name: parameter.detach().cpu() for name, parameter in fixture.model.named_parameters()}
    return {"pre": pre, "gradients": gradients, "parameters": post, "optimizer": fixture.optimizer.state_dict()}


def traced_full_step(fixture: Fixture):
    fixture.optimizer.zero_grad(set_to_none=True)
    output, rendered, parts = fixture.loss()
    rendered[0].retain_grad(); rendered[1].retain_grad()
    overrides = output.canonical_overrides.as_dict()
    for value in overrides.values(): value.retain_grad()
    pre = fixture.snapshot(output, rendered, parts)
    parts["closure_total"].backward()
    return {
        "pre": pre,
        "image_loss_gradients": {
            "dLoss_dRGB": rendered[0].grad.detach().cpu(),
            "dLoss_dAlpha": rendered[1].grad.detach().cpu(),
        },
        "raster_backward_canonical_attribute_gradients": {
            name: None if value.grad is None else value.grad.detach().cpu()
            for name, value in overrides.items()
        },
        "raw_parameter_gradients": {
            name: None if parameter.grad is None else parameter.grad.detach().cpu()
            for name, parameter in fixture.model.named_parameters()
        },
    }


def worker(args):
    fixture = load_fixture(args, args.kind)
    if args.mode == "no_raster":
        result = non_raster_step(fixture)
    elif args.mode == "trace":
        result = traced_full_step(fixture)
    elif args.mode == "three_step":
        result = {"steps": []}
        for step in range(3):
            result["steps"].append(fixture.one_step())
    else:
        result = fixture.one_step()
    torch.save(result, args.worker_output)


def worker_command(args, kind, mode, output):
    forwarded = []
    for name in (
        "pipeline-config", "oracle-config", "request", "teacher-anchor-residuals",
        "fixture-manifest", "split", "gate7-root", "output-dir",
    ):
        forwarded.extend((f"--{name}", str(getattr(args, name.replace("-", "_")))))
    return [
        sys.executable, str(Path(__file__).resolve()), *forwarded, "--device", args.device,
        "--worker", "--kind", kind, "--mode", mode, "--resume-source", str(args.resume_source),
        "--worker-output", str(output),
    ]


def run_workers(args, kind, mode, count, directory):
    directory.mkdir(parents=True, exist_ok=True)
    processes, paths = [], []
    for index in range(count):
        path = directory / f"{mode}_{index + 1}.pt"; paths.append(path)
        processes.append(subprocess.Popen(worker_command(args, kind, mode, path)))
    for process in processes:
        if process.wait() != 0: raise RuntimeError(f"{mode} worker failed")
    return [torch.load(path, map_location="cpu", weights_only=False) for path in paths]


def pairwise(results):
    rows = []
    for left_index, right_index in itertools.combinations(range(len(results)), 2):
        comparison = compare_results(results[left_index], results[right_index])
        parameters = summarize_parameter_difference(results[left_index], results[right_index])
        rows.append({
            "pair": f"C{left_index + 1}-C{right_index + 1}", "tensor_differences": comparison,
            "channelwise_parameter_differences": parameters,
            "post_parameter_max": parameters["all_parameters"]["max_abs_diff"],
            "post_parameter_p99": parameters["all_parameters"]["p99_abs_diff"],
            "post_parameter_relative_l2": parameters["all_parameters"]["relative_l2_diff"],
        })
    return rows


def first_difference(pair):
    ordered = [
        ("raw_oracle_parameters", "pre.residuals"),
        ("bounded_residuals", "pre.residuals"),
        ("gates", "pre.geometry_gate"),
        ("composed_canonical_attributes", "pre.overrides"),
        ("raster_forward_rgb_alpha", "pre.rgb"),
        ("scalar_loss", "pre.losses"),
        ("raw_parameter_gradients", "gradients"),
        ("adam_moments", "optimizer.state"),
        ("updated_parameters", "parameters"),
        ("post_step_residuals", "post.residuals"),
        ("post_step_rgb_alpha", "post.rgb"),
    ]
    differences = pair["tensor_differences"]
    for stage, prefix in ordered:
        matching = [value for name, value in differences.items() if name.startswith(prefix)]
        if any(value["max_abs_diff"] > 0 for value in matching):
            return {"first_nonzero_difference_stage": stage, "supporting_paths": [name for name, value in differences.items() if name.startswith(prefix) and value["max_abs_diff"] > 0][:20]}
    return {"first_nonzero_difference_stage": "none", "supporting_paths": []}


def environment():
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    try:
        smi = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"], text=True).strip()
    except Exception as error: smi = f"unavailable: {error}"
    try:
        import gsplat
        gsplat_version = getattr(gsplat, "__version__", "unknown")
    except Exception as error: gsplat_version = f"unavailable: {error}"
    strict_error = None
    try:
        torch.use_deterministic_algorithms(True)
        sample = torch.ones(1, 1, 4, 4, device="cuda", requires_grad=True)
        F.pad(sample, (1, 1, 1, 1), mode="reflect").sum().backward()
    except Exception as error:
        strict_error = repr(error)
    finally:
        torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "gpu_model": gpu_name, "gpu_uuid_driver": smi, "cuda_runtime": torch.version.cuda,
        "pytorch": torch.__version__, "gsplat": gsplat_version, "python": platform.python_version(),
        "cudnn": torch.backends.cudnn.version(), "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "rasterizer_backend": "MMLP-Human gsplat/custom CUDA rasterizer",
        "atomic_accumulation": "inferred likely in CUDA raster backward; backend does not expose a deterministic guarantee",
        "strict_deterministic_error": strict_error,
        "pytorch_deterministic_warning_present": strict_error is not None,
    }


def main():
    p = argparse.ArgumentParser()
    for name in (
        "pipeline-config", "oracle-config", "request", "teacher-anchor-residuals",
        "fixture-manifest", "split", "gate7-root", "output-dir",
    ): p.add_argument(f"--{name}", required=True, type=Path)
    p.add_argument("--resume-source", type=Path); p.add_argument("--device", default="cuda")
    p.add_argument("--worker", action="store_true"); p.add_argument("--kind", choices=("gaussian", "anchor"))
    p.add_argument("--mode", choices=("full", "no_raster", "three_step", "trace"), default="full"); p.add_argument("--worker-output", type=Path)
    p.add_argument("--trace-only", action="store_true")
    args = p.parse_args()
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    if args.worker: worker(args); return
    root = args.output_dir.resolve(); root.mkdir(parents=True, exist_ok=True)
    if args.trace_only:
        for kind in ("gaussian", "anchor"):
            args.resume_source = args.output_dir.parent / "GATE7-ORACLE-CLOSURE-001" / f"{kind}_resume" / "resume_source.pth"
            directory = root / kind; directory.mkdir(exist_ok=True)
            traces = run_workers(args, kind, "trace", 2, directory / "workers_trace")
            comparison = compare_results(traces[0], traces[1])
            stages = (
                ("pre_raster_and_forward", ("pre.",)),
                ("loss_gradient_wrt_rgb_alpha", ("image_loss_gradients.",)),
                ("cuda_raster_backward_attribute_gradients", ("raster_backward_canonical_attribute_gradients.",)),
                ("raw_oracle_parameter_gradients", ("raw_parameter_gradients.",)),
            )
            first = "none"; paths = []
            for stage, prefixes in stages:
                paths = [name for name, value in comparison.items() if name.startswith(prefixes) and value["max_abs_diff"] > 0]
                if paths: first = stage; break
            (directory / "first_difference_trace.json").write_text(json.dumps({
                "first_nonzero_difference_stage": first, "supporting_paths": paths,
                "tensor_differences": comparison,
            }, indent=2), encoding="utf-8")
        return
    env = environment(); (root / "environment_determinism.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    protocol = {
        "control_control_processes": 5, "resume_control_repeats": 3, "no_raster_processes": 2,
        "three_step_processes": 2, "same_resume_source": True,
        "acceptance": {"strict_pre_raster_atol": 1e-7, "full_cuda_absolute_cap": 1e-6, "relative_l2_cap": 1e-6, "resume_vs_control_multiplier": 1.25},
    }
    (root / "experiment_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    summary = {}
    for kind in ("gaussian", "anchor"):
        directory = root / kind; directory.mkdir(exist_ok=True)
        args.resume_source = args.output_dir.parent / "GATE7-ORACLE-CLOSURE-001" / f"{kind}_resume" / "resume_source.pth"
        controls = run_workers(args, kind, "full", 5, directory / "workers_control")
        control_pairs = pairwise(controls)
        (directory / "control_control_pairwise.json").write_text(json.dumps(control_pairs, indent=2), encoding="utf-8")
        resumes = run_workers(args, kind, "full", 6, directory / "workers_resume")
        resume_rows = []
        for repeat in range(3):
            comparison = compare_results(resumes[2 * repeat], resumes[2 * repeat + 1])
            params = summarize_parameter_difference(resumes[2 * repeat], resumes[2 * repeat + 1])
            resume_rows.append({"repeat": repeat + 1, "tensor_differences": comparison, "channelwise_parameter_differences": params, "post_parameter_max": params["all_parameters"]["max_abs_diff"], "post_parameter_relative_l2": params["all_parameters"]["relative_l2_diff"]})
        (directory / "resume_control_repeats.json").write_text(json.dumps(resume_rows, indent=2), encoding="utf-8")
        no_raster = run_workers(args, kind, "no_raster", 2, directory / "workers_no_raster")
        no_raster_comparison = compare_results(no_raster[0], no_raster[1])
        no_raster_max = max((value["max_abs_diff"] for value in no_raster_comparison.values()), default=0.0)
        no_raster_result = {"max_abs_diff": no_raster_max, "bitwise_equal": all(value["bitwise_equal"] for value in no_raster_comparison.values()), "differences": no_raster_comparison, "status": "PASS" if no_raster_max <= 1e-7 else "FAIL"}
        (directory / "no_raster_exact_parity.json").write_text(json.dumps(no_raster_result, indent=2), encoding="utf-8")
        trace = first_difference(control_pairs[0]); (directory / "first_difference_trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
        three = run_workers(args, kind, "three_step", 2, directory / "workers_three_step")
        growth = []
        for step in range(3):
            comparison = compare_results(three[0]["steps"][step], three[1]["steps"][step])
            params = summarize_parameter_difference(three[0]["steps"][step], three[1]["steps"][step])
            growth.append({
                "step": step + 1, "parameter_max_diff": params["all_parameters"]["max_abs_diff"],
                "parameter_relative_l2": params["all_parameters"]["relative_l2_diff"],
                "loss_max_diff": max((value["max_abs_diff"] for name, value in comparison.items() if "loss" in name), default=0.0),
                "rgb_alpha_max_diff": max((value["max_abs_diff"] for name, value in comparison.items() if name.endswith(("rgb", "alpha"))), default=0.0),
                "gate_max_diff": max((value["max_abs_diff"] for name, value in comparison.items() if "gate" in name), default=0.0),
                "residual_max_diff": max((value["max_abs_diff"] for name, value in comparison.items() if "residual" in name), default=0.0),
            })
        (directory / "three_step_divergence.json").write_text(json.dumps(growth, indent=2), encoding="utf-8")
        channelwise = {
            "control_control": [row["channelwise_parameter_differences"] for row in control_pairs],
            "resume_control": [row["channelwise_parameter_differences"] for row in resume_rows],
        }
        (directory / "channelwise_differences.json").write_text(json.dumps(channelwise, indent=2), encoding="utf-8")
        d_control_max = max(row["post_parameter_max"] for row in control_pairs)
        d_control_p99 = max(row["post_parameter_p99"] for row in control_pairs)
        d_resume_max = max(row["post_parameter_max"] for row in resume_rows)
        resume_relative = max(row["post_parameter_relative_l2"] for row in resume_rows)
        bound = max(1.25 * d_control_max, d_control_max + 1e-7)
        growth_max = max(row["parameter_max_diff"] for row in growth)
        passed = (
            no_raster_result["status"] == "PASS" and d_control_max > 0
            and trace["first_nonzero_difference_stage"] in {"raw_parameter_gradients", "adam_moments", "updated_parameters", "post_step_residuals", "post_step_rgb_alpha"}
            and d_resume_max <= bound and d_resume_max <= 1e-6 and resume_relative <= 1e-6
            and growth_max <= 1e-6
        )
        summary[kind] = {
            "D_control_max": d_control_max, "D_control_p99": d_control_p99,
            "D_resume_max": d_resume_max, "resume_bound": bound,
            "resume_relative_l2_max": resume_relative, "no_raster": no_raster_result["status"],
            "first_difference": trace["first_nonzero_difference_stage"],
            "three_step_max": growth_max, "status": "PASS" if passed else "FAIL",
        }
    overall = all(value["status"] == "PASS" for value in summary.values())
    document = {"methods": summary, "status": "PASS" if overall else "FAIL"}
    (root / "noise_floor_summary.json").write_text(json.dumps(document, indent=2), encoding="utf-8")
    (root / "forward.log").write_text(json.dumps({"environment": env, "summary": document}, indent=2), encoding="utf-8")
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
