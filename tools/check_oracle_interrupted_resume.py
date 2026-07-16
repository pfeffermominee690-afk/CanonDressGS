from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_dressable as training
from scene.full_attribute_oracle import AnchorResidualOracle, GaussianResidualOracle
from scene.gaussian_clothing_residuals import (
    AnchorClothingResiduals, RESIDUAL_CONTRACT_VERSION,
    compose_canonical_gaussian_overrides, interpolate_anchor_clothing_residuals,
)
from utils.dressable_camera_utils import build_mmlphuman_camera
from utils.full_training_checkpoint_utils import (
    load_full_training_checkpoint, random_state_fingerprint,
    save_full_training_checkpoint, validate_full_training_checkpoint,
)
from utils.mmlphuman_state_utils import mmlphuman_state_transaction
from utils.oracle_loss_utils import OracleLossWeights, oracle_rendering_loss


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def chw(value: torch.Tensor, channels: int) -> torch.Tensor:
    if value.ndim == 3 and value.shape[0] == channels: return value
    if value.ndim == 3 and value.shape[-1] == channels: return value.permute(2, 0, 1).contiguous()
    raise ValueError("ambiguous renderer output")


class Fixture:
    def __init__(self, args: argparse.Namespace, kind: str):
        self.args, self.kind = args, kind
        self.device = torch.device(args.device)
        self.pipeline = training.load_config(args.pipeline_config)
        self.oracle_config = yaml.safe_load(args.oracle_config.read_text(encoding="utf-8"))
        self.base = training.load_frozen_mmlphuman_base(
            self.pipeline["base"]["model_dir"], self.pipeline["base"]["checkpoint_path"], self.device,
        )
        self.request = torch.load(args.request, map_location="cpu", weights_only=False)
        teacher = AnchorClothingResiduals(**torch.load(args.teacher_anchor_residuals, map_location=self.device, weights_only=True))
        self.indices = self.base.nbr_gs.to(self.device, dtype=torch.long)
        self.weights = self.base.nbr_gs_invdist.to(self.device, dtype=self.base._xyz.dtype)
        self.weights = self.weights / self.weights.sum(1, keepdim=True)
        self.teacher_anchor = teacher
        self.teacher_gaussian = interpolate_anchor_clothing_residuals(teacher, self.indices, self.weights, self.base)
        common = dict(bounds=self.oracle_config["bounds"], initial_gate_probability=self.oracle_config["initial_gate_probability"], enable_shn=False)
        if kind == "gaussian":
            self.model = GaussianResidualOracle(self.base, **common).to(self.device)
        else:
            self.anchor_edges = training.build_anchor_knn_edges(self.base.xyz_vt.to(self.device), 4)
            self.model = AnchorResidualOracle(
                self.base, self.base.xyz_vt.shape[0], self.indices, self.weights,
                graph_edges=self.anchor_edges, **common,
            ).to(self.device)
        self.model.configure_stage(3)
        lrs = {"geometry_residuals": .03, "appearance_residuals": .05, "geometry_gate": .03, "appearance_gate": .03}
        groups, self.group_names = self.model.parameter_groups(lrs)
        self.optimizer = torch.optim.Adam(groups)
        self.background = torch.tensor(self.pipeline["render"]["background"], device=self.device, dtype=self.base._xyz.dtype)
        self.sample = self._target()
        self.method_state = {
            "oracle_type": self.model.oracle_type, "kind": kind,
            "base_checkpoint_sha256": sha(Path(self.pipeline["base"]["model_dir"]) / self.pipeline["base"]["checkpoint_path"]),
            "fixture_fingerprint": sha(args.fixture_manifest), "split_fingerprint": sha(args.split),
            "config_fingerprint": sha(args.oracle_config),
            "residual_contract_version": RESIDUAL_CONTRACT_VERSION,
            "residual_bounds": self.oracle_config["bounds"], "effective_sh_degree": 0,
            "anchor_graph_fingerprint": None if kind == "gaussian" else tensor_sha(self.anchor_edges),
            "interpolation_fingerprint": None if kind == "gaussian" else hashlib.sha256(
                (tensor_sha(self.indices) + tensor_sha(self.weights)).encode()
            ).hexdigest(),
        }
        self.data_state = {
            "fixture_fingerprint": sha(args.fixture_manifest), "split_fingerprint": sha(args.split),
            "sampler_state": {"epoch": 16, "position": 2, "next_condition_id": "target"},
            "batch_fingerprint": self.batch_fingerprint(),
        }

    def _target(self):
        r = self.request
        condition = {"pose": r["target_pose"], "Rh": r["target_Rh"], "Th": r["target_Th"], "camera": r["target_camera"]}
        teacher_overrides = compose_canonical_gaussian_overrides(self.base, self.teacher_gaussian)
        with torch.no_grad():
            base = self.render(condition, None); target = self.render(condition, teacher_overrides)
        br, ba, tr, ta = chw(base[0], 3), chw(base[1], 1), chw(target[0], 3), chw(target[1], 1)
        foreground = (ta > .01).float()
        clothing = ((((tr - br).abs().mean(0, keepdim=True) > .005) | ((ta - ba).abs() > .005)).float() * foreground)
        return {**condition, "base_rgb": br, "base_alpha": ba, "target_rgb": tr, "foreground": foreground, "clothing": clothing}

    def render(self, sample, overrides):
        height, width = int(self.request["target_height"]), int(self.request["target_width"])
        camera = build_mmlphuman_camera(sample["camera"], height, width, self.device)
        with mmlphuman_state_transaction(self.base, sample["pose"].to(self.device), sample["Rh"].to(self.device), sample["Th"].to(self.device)):
            return self.base.render(camera, background=self.background, canonical_overrides=None if overrides is None else overrides.as_dict())

    def batch_fingerprint(self):
        payload = b"target"
        for name in ("target_pose", "target_Rh", "target_Th"):
            payload += self.request[name].contiguous().numpy().tobytes()
        return hashlib.sha256(payload).hexdigest()

    def loss(self):
        output = self.model(self.base)
        rendered = self.render(self.sample, output.canonical_overrides)
        weights = OracleLossWeights.from_mapping({**self.oracle_config["loss"], "lpips": 0.0})
        parts = oracle_rendering_loss(
            prediction_rgb=rendered[0], prediction_alpha=rendered[1],
            target_rgb=self.sample["target_rgb"], target_foreground_mask=self.sample["foreground"],
            target_clothing_mask=self.sample["clothing"], base_rgb=self.sample["base_rgb"],
            base_alpha=self.sample["base_alpha"], oracle_output=output, weights=weights,
        )
        target = self.teacher_gaussian if self.kind == "gaussian" else self.teacher_anchor
        supervision = sum(F.smooth_l1_loss(getattr(output.raw_residuals, name), getattr(target, name)) for name in (
            "delta_xyz", "delta_log_scaling", "delta_rotvec", "delta_opacity_logit", "delta_sh0", "delta_shN",
        ))
        active = ((target.delta_xyz.reshape(target.delta_xyz.shape[0], -1).abs().sum(1) + target.delta_sh0.reshape(target.delta_sh0.shape[0], -1).abs().sum(1)) > 1e-8).float().reshape(-1, 1)
        gate = F.binary_cross_entropy(output.geometry_gate.clamp(1e-6, 1-1e-6), active) + F.binary_cross_entropy(output.appearance_gate.clamp(1e-6, 1-1e-6), active)
        parts = {**parts, "supervision": supervision, "gate_supervision": gate}
        parts["closure_total"] = parts["total"] + 5 * supervision + .1 * gate
        return output, rendered, parts

    def snapshot(self, output, rendered, parts):
        result = {
            "batch_fingerprint": self.batch_fingerprint(),
            "residuals": {name: getattr(output.raw_residuals, name).detach().cpu() for name in (
                "delta_xyz", "delta_log_scaling", "delta_rotvec", "delta_opacity_logit", "delta_sh0", "delta_shN",
            )},
            "gaussian_residuals": {name: getattr(output.gaussian_residuals, name).detach().cpu() for name in (
                "delta_xyz", "delta_log_scaling", "delta_rotvec", "delta_opacity_logit", "delta_sh0", "delta_shN",
            )},
            "geometry_gate": output.geometry_gate.detach().cpu(), "appearance_gate": output.appearance_gate.detach().cpu(),
            "overrides": {name: value.detach().cpu() for name, value in output.canonical_overrides.as_dict().items()},
            "rgb": rendered[0].detach().cpu(), "alpha": rendered[1].detach().cpu(),
            "losses": {name: value.detach().cpu() for name, value in parts.items()},
        }
        return result

    def one_step(self):
        base_before = {name: getattr(self.base, name).detach().cpu().clone() for name in ("_xyz", "_scaling", "_rotation", "_opacity", "_sh0", "_shN")}
        rng_before = self.rng()
        self.optimizer.zero_grad(set_to_none=True)
        output, rendered, parts = self.loss()
        pre = self.snapshot(output, rendered, parts)
        parts["closure_total"].backward()
        gradients = {name: None if parameter.grad is None else parameter.grad.detach().cpu().clone() for name, parameter in self.model.named_parameters()}
        group_norms = {}
        for name, group in zip(self.group_names, self.optimizer.param_groups):
            values = [p.grad.detach().float().reshape(-1) for p in group["params"] if p.grad is not None]
            group_norms[name] = float(torch.linalg.vector_norm(torch.cat(values))) if values else 0.0
        frozen_grad_count = sum(getattr(self.base, name).grad is not None for name in base_before)
        self.optimizer.step()
        with torch.no_grad():
            post_output, post_render, post_parts = self.loss()
        post = self.snapshot(post_output, post_render, post_parts)
        base_max = max(float((getattr(self.base, name).detach().cpu() - value).abs().max()) for name, value in base_before.items())
        return {
            "pre": pre, "post": post, "gradients": gradients, "parameter_group_gradient_norms": group_norms,
            "parameters": {name: value.detach().cpu().clone() for name, value in self.model.named_parameters()},
            "optimizer": copy.deepcopy(self.optimizer.state_dict()), "rng_before": rng_before, "rng_after": self.rng(),
            "frozen_base_gradient_count": frozen_grad_count, "frozen_base_parameter_max_diff": base_max,
        }

    def rng(self):
        from utils.full_training_checkpoint_utils import capture_random_state
        return random_state_fingerprint(capture_random_state())


def validate_closure_checkpoint(checkpoint, fixture: Fixture):
    validate_full_training_checkpoint(
        checkpoint, model=fixture.model, optimizer=fixture.optimizer,
        optimizer_group_names=fixture.group_names, expected_method_state=fixture.method_state,
        expected_data_state=fixture.data_state,
    )
    current = fixture.model.state_dict()
    if set(checkpoint["model_state"]) != set(current): raise ValueError("trainable parameter set mismatch")
    for name, value in current.items():
        if checkpoint["model_state"][name].shape != value.shape: raise ValueError(f"trainable parameter shape mismatch: {name}")


def compare(left: Any, right: Any, path="", output=None):
    output = {} if output is None else output
    if isinstance(left, torch.Tensor):
        diff = (left.float() - right.float()).abs()
        atol, rtol = (1e-6, 1e-5) if path.endswith(("rgb", "alpha")) else (1e-7, 1e-6)
        output[path] = {"max_abs": float(diff.max()) if diff.numel() else 0.0, "mean_abs": float(diff.mean()) if diff.numel() else 0.0, "allclose": bool(torch.allclose(left, right, atol=atol, rtol=rtol)), "bitwise_equal": bool(torch.equal(left, right))}
    elif isinstance(left, dict):
        if set(left) != set(right): output[path] = {"allclose": False, "key_mismatch": True}
        else:
            for key in left: compare(left[key], right[key], f"{path}.{key}".strip("."), output)
    elif isinstance(left, list):
        for index, value in enumerate(left): compare(value, right[index], f"{path}[{index}]", output)
    elif left != right:
        output[path] = {"allclose": False, "left": str(left), "right": str(right)}
    return output


def worker(args):
    fixture = Fixture(args, args.kind)
    checkpoint = load_full_training_checkpoint(
        args.resume_source, model=fixture.model, optimizer=fixture.optimizer,
        optimizer_group_names=fixture.group_names, scheduler=None, scaler=None,
        expected_method_state=fixture.method_state, expected_data_state=fixture.data_state,
    )
    result = fixture.one_step()
    torch.save(result, args.worker_output)
    save_full_training_checkpoint(
        args.after_checkpoint, model=fixture.model, optimizer=fixture.optimizer,
        optimizer_group_names=fixture.group_names, scheduler=None, scaler=None,
        training_state={**checkpoint["training_state"], "global_step": int(checkpoint["training_state"]["global_step"]) + 1},
        data_state=fixture.data_state, method_state=fixture.method_state,
    )


def main():
    parser = argparse.ArgumentParser()
    for name in ("pipeline-config", "oracle-config", "request", "teacher-anchor-residuals", "fixture-manifest", "split", "gate7-root", "output-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--worker", action="store_true"); parser.add_argument("--kind", choices=("gaussian", "anchor"))
    parser.add_argument("--resume-source", type=Path); parser.add_argument("--worker-output", type=Path); parser.add_argument("--after-checkpoint", type=Path)
    args = parser.parse_args()
    if args.worker: worker(args); return
    root = args.output_dir.resolve(); root.mkdir(parents=True, exist_ok=True)
    all_metrics, mismatch = {}, {}
    for kind in ("gaussian", "anchor"):
        target = root / f"{kind}_resume"; target.mkdir(exist_ok=True)
        fixture = Fixture(args, kind)
        original_path = args.gate7_root / f"{kind}_oracle" / "last_checkpoint.pth"
        original = load_full_training_checkpoint(
            original_path, model=fixture.model, optimizer=fixture.optimizer,
            optimizer_group_names=fixture.group_names, scheduler=None, scaler=None,
            expected_method_state={"oracle_type": fixture.model.oracle_type, "kind": kind},
            expected_data_state={"fixture_only": True, "conditions": ["reference_0", "reference_1", "target"]},
        )
        source = target / "resume_source.pth"
        save_full_training_checkpoint(
            source, model=fixture.model, optimizer=fixture.optimizer, optimizer_group_names=fixture.group_names,
            scheduler=None, scaler=None,
            training_state={**original["training_state"], "sampler_state": fixture.data_state["sampler_state"]},
            data_state=fixture.data_state, method_state=fixture.method_state,
        )
        control = fixture.one_step(); torch.save(control, target / "control_result.pt")
        save_full_training_checkpoint(
            target / "control_after_next_step.pth", model=fixture.model, optimizer=fixture.optimizer,
            optimizer_group_names=fixture.group_names, scheduler=None, scaler=None,
            training_state={"global_step": 51, "stage": 3, "sampler_state": fixture.data_state["sampler_state"]},
            data_state=fixture.data_state, method_state=fixture.method_state,
        )
        command = [sys.executable, str(Path(__file__).resolve()), *sum(([f"--{name}", str(getattr(args, name.replace('-', '_')))] for name in (
            "pipeline-config", "oracle-config", "request", "teacher-anchor-residuals", "fixture-manifest", "split", "gate7-root", "output-dir",
        )), []), "--device", args.device, "--worker", "--kind", kind, "--resume-source", str(source), "--worker-output", str(target / "resumed_result.pt"), "--after-checkpoint", str(target / "resumed_after_next_step.pth")]
        subprocess.run(command, check=True)
        resumed = torch.load(target / "resumed_result.pt", map_location="cpu", weights_only=False)
        comparisons = compare(control, resumed)
        passed = all(value.get("allclose", True) for value in comparisons.values())
        metrics = {
            "status": "PASS" if passed else "FAIL", "independent_process_command": command,
            "comparisons": comparisons, "batch_fingerprint_equal": control["pre"]["batch_fingerprint"] == resumed["pre"]["batch_fingerprint"],
            "rng_pre_equal": control["rng_before"] == resumed["rng_before"], "rng_post_equal": control["rng_after"] == resumed["rng_after"],
            "sampler_state": fixture.data_state["sampler_state"],
            "frozen_base_gradient_count": control["frozen_base_gradient_count"],
            "frozen_base_parameter_max_diff": control["frozen_base_parameter_max_diff"],
        }
        (target / "parity_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        optimizer_rows = {key: value for key, value in comparisons.items() if key.startswith("optimizer")}
        (target / "optimizer_state_comparison.json").write_text(json.dumps(optimizer_rows, indent=2), encoding="utf-8")
        (target / "rng_comparison.json").write_text(json.dumps({key: metrics[key] for key in ("rng_pre_equal", "rng_post_equal", "sampler_state", "batch_fingerprint_equal")}, indent=2), encoding="utf-8")
        all_metrics[kind] = metrics
        source_payload = torch.load(source, map_location="cpu", weights_only=False)
        mutations = {
            "oracle_type": ("method_state", "oracle_type", "wrong"),
            "base_checkpoint_sha256": ("method_state", "base_checkpoint_sha256", "0" * 64),
            "fixture_fingerprint": ("method_state", "fixture_fingerprint", "0" * 64),
            "split_fingerprint": ("method_state", "split_fingerprint", "0" * 64),
            "residual_contract_version": ("method_state", "residual_contract_version", -1),
            "residual_bounds": ("method_state", "residual_bounds", {"xyz": 99}),
            "effective_sh_degree": ("method_state", "effective_sh_degree", 1),
            "anchor_graph_fingerprint": ("method_state", "anchor_graph_fingerprint", "bad"),
            "interpolation_fingerprint": ("method_state", "interpolation_fingerprint", "bad"),
            "unsupported_checkpoint_version": (None, "training_checkpoint_version", 999),
        }
        for name, (section, key, value) in mutations.items():
            changed = copy.deepcopy(source_payload)
            if section is None: changed[key] = value
            else: changed[section][key] = value
            try: validate_closure_checkpoint(changed, Fixture(args, kind)); mismatch[f"{kind}.{name}"] = "NOT_REJECTED"
            except Exception: mismatch[f"{kind}.{name}"] = "REJECTED"
        changed = copy.deepcopy(source_payload); changed["optimizer_state"]["parameter_groups"][0]["fingerprint"] = "bad"
        try: validate_closure_checkpoint(changed, Fixture(args, kind)); mismatch[f"{kind}.optimizer_parameter_group_fingerprint"] = "NOT_REJECTED"
        except Exception: mismatch[f"{kind}.optimizer_parameter_group_fingerprint"] = "REJECTED"
        changed = copy.deepcopy(source_payload); first = next(iter(changed["model_state"])); changed["model_state"][first] = changed["model_state"][first].reshape(-1)[:1]
        try: validate_closure_checkpoint(changed, Fixture(args, kind)); mismatch[f"{kind}.trainable_parameter_shape"] = "NOT_REJECTED"
        except Exception: mismatch[f"{kind}.trainable_parameter_shape"] = "REJECTED"
    (root / "mismatch_rejection_tests.json").write_text(json.dumps(mismatch, indent=2), encoding="utf-8")
    print(json.dumps({"parity": {k: v["status"] for k, v in all_metrics.items()}, "mismatch": mismatch}, indent=2))


if __name__ == "__main__":
    main()
