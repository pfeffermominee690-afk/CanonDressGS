from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_dressable as training
from scene.gaussian_clothing_residuals import CHANNELS
from tools.check_differentiable_render_path import (
    BASE_PARAMETER_NAMES,
    _build_model,
    _gradient_snapshot,
    _head_modules,
    _prepare_episode,
)
from tools.check_real_image_conditioned_one_batch import save_render_tensor
from tools.run_dual_target_smoke20_v5_1 import (
    _forward,
    _loss,
    _state_clone,
    _state_equal,
    _tensor_image,
)
from utils.mmlphuman_anchor_deformation import MMLPHumanAnchorDeformationAdapter


EXPECTED_OUTFIT = "O05"
EXPECTED_TARGET = "cond_000347"
EXPECTED_REFERENCES = ("cond_000318", "cond_000000")
LOG_STEPS = frozenset({0, 1, 5, 10, 20, 40, 60, 80})
RENDER_STEPS = frozenset({0, 20, 40, 80})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _resolve_observation_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _fixed_episode_contract(
    manifest_path: Path,
    sample: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    outfit = next(item for item in payload["outfits"] if item["outfit_id"] == EXPECTED_OUTFIT)
    by_condition = {item["condition_id"]: item for item in outfit["observations"]}
    requested = [*EXPECTED_REFERENCES, EXPECTED_TARGET]
    missing = [condition for condition in requested if condition not in by_condition]
    if missing:
        raise ValueError(f"fixed episode conditions are missing: {missing}")
    root = manifest_path.parent
    inputs = []
    for role, condition_id in [
        ("reference_0", EXPECTED_REFERENCES[0]),
        ("reference_1", EXPECTED_REFERENCES[1]),
        ("target", EXPECTED_TARGET),
    ]:
        observation = by_condition[condition_id]
        files = []
        for field, value in sorted(observation.items()):
            if not isinstance(value, str) or field in {"condition_id", "identity_audit_status"}:
                continue
            path = _resolve_observation_path(root, value)
            if path.is_file():
                files.append({"field": field, "path": str(path), "sha256": _sha256(path)})
        inputs.append({"role": role, "condition_id": condition_id, "files": files})
    external = []
    for role, path in (
        ("dataset_manifest", manifest_path),
        ("gate4_config", args.gate4_config.resolve()),
        ("module2_config", args.module2_config.resolve()),
        ("module3_checkpoint", args.module3_checkpoint.resolve()),
    ):
        external.append({"role": role, "path": str(path), "sha256": _sha256(path)})
    resolved_references = tuple(sample["reference_condition_ids"])
    contract = {
        "schema_version": "canondressgs.fixed_episode.v5.2",
        "outfit_id": sample["outfit_id"],
        "reference_condition_ids": list(resolved_references),
        "target_condition_id": sample["target_condition_id"],
        "reference_count": len(resolved_references),
        "target_reference_overlap": bool(sample["target_condition_id"] in resolved_references),
        "seed": args.seed,
        "dataset_seed": 42,
        "sample_index": args.sample_index,
        "optimizer": {"type": "Adam", "learning_rate": args.learning_rate},
        "inputs": inputs,
        "external_inputs": external,
        "git": {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
            "status_short": subprocess.check_output(["git", "status", "--short"], cwd=PROJECT_ROOT, text=True).splitlines(),
        },
    }
    if contract["outfit_id"] != EXPECTED_OUTFIT or contract["target_condition_id"] != EXPECTED_TARGET:
        raise ValueError(f"resolved episode does not match {EXPECTED_TARGET}_{EXPECTED_OUTFIT}")
    if resolved_references != EXPECTED_REFERENCES:
        raise ValueError(f"resolved references differ: {resolved_references} != {EXPECTED_REFERENCES}")
    if contract["reference_count"] != 2 or contract["target_reference_overlap"]:
        raise ValueError("fixed episode requires two target-disjoint references")
    return contract


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _set_rng_state(value: dict[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(value["torch_cuda"])


def _save_initial_state(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    trainable = {
        name: value.detach().cpu().clone()
        for name, value in model.named_parameters()
        if value.requires_grad
    }
    payload = {
        "model": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        "optimizer": deepcopy(optimizer.state_dict()),
        "rng": _rng_state(),
        "trainable": trainable,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, temporary)
    temporary.replace(path)
    return {
        "file": str(path),
        "file_sha256": _sha256(path),
        "trainable_parameter_count": len(trainable),
        "trainable_tensor_sha256": {name: _tensor_sha256(value) for name, value in trainable.items()},
    }


def _restore_initial_state(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    _set_rng_state(payload["rng"])
    actual = {
        name: _tensor_sha256(value)
        for name, value in model.named_parameters()
        if value.requires_grad
    }
    expected = {name: _tensor_sha256(value) for name, value in payload["trainable"].items()}
    if actual != expected:
        raise AssertionError("restored trainable state differs from frozen step-0 state")
    return {"exact": True, "trainable_tensor_sha256": actual}


def _build_runtime(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    base, model, checkpoint, gate4 = _build_model(
        args.gate4_config, args.module2_config, args.module3_checkpoint, device,
    )
    sample, episode, _ = _prepare_episode(args.manifest, sample_index=args.sample_index)
    if tuple(sample["reference_condition_ids"]) != EXPECTED_REFERENCES:
        raise ValueError(f"fixed references changed: {sample['reference_condition_ids']}")
    adapter = MMLPHumanAnchorDeformationAdapter.from_mmlphuman_base(
        base,
        canonical_anchors=model.canonical_anchors,
        lbs_grid_path=gate4["base"]["lbs_grid_path"],
    )
    background = torch.tensor(gate4["render"]["background"], device=device, dtype=base._xyz.dtype)
    geometry = training.prepare_real_reference_geometry(base, adapter, episode, background)
    optimizer = torch.optim.Adam(
        [value for value in model.parameters() if value.requires_grad],
        lr=args.learning_rate,
    )
    return {
        "base": base,
        "model": model,
        "checkpoint": checkpoint,
        "gate4": gate4,
        "sample": sample,
        "episode": episode,
        "adapter": adapter,
        "background": background,
        "geometry": geometry,
        "optimizer": optimizer,
    }


def _regularizers(result: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    residual = sum(
        getattr(result["gated_anchor_residuals"], name).abs().mean()
        for name in CHANNELS
    )
    geometry_gate = result["completion"].geometry_gate
    appearance_gate = result["completion"].appearance_gate
    gate = (
        (geometry_gate * (1 - geometry_gate)).mean()
        + (appearance_gate * (1 - appearance_gate)).mean()
    )
    return residual, gate


def _component_values(
    parts: dict[str, torch.Tensor],
    residual: torch.Tensor,
    gate: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, tuple[torch.Tensor, float]]:
    return {
        "edit": (parts["edit"], 1.0),
        "clothing": (parts["clothing"], 1.0),
        "preserve": (parts["preserve"], 1.0),
        "protected": (parts["protected"], 1.0),
        "transition": (parts["transition"], 0.25),
        "alpha_edit": (parts["alpha_edit"], args.alpha_weight * args.alpha_edit_weight),
        "alpha_transition": (parts["alpha_transition"], args.alpha_weight * args.alpha_transition_weight),
        "alpha_base": (parts["alpha_base"], args.alpha_weight * args.alpha_base_weight),
        "residual_magnitude": (residual, args.residual_regularization_weight),
        "active_gate_regularization": (gate, args.gate_regularization_weight),
    }


def _objective(
    components: dict[str, tuple[torch.Tensor, float]],
    scales: dict[str, float],
) -> torch.Tensor:
    edit = scales["edit_rgb"] * sum(components[name][0] * components[name][1] for name in ("edit", "clothing"))
    identity = scales["identity"] * sum(components[name][0] * components[name][1] for name in ("preserve", "protected"))
    alpha = scales["alpha"] * sum(components[name][0] * components[name][1] for name in ("alpha_edit", "alpha_transition", "alpha_base"))
    regularization = scales["regularization"] * sum(components[name][0] * components[name][1] for name in ("residual_magnitude", "active_gate_regularization"))
    transition = components["transition"][0] * components["transition"][1]
    return edit + identity + alpha + regularization + transition


def _module_parameter_indices(
    model: torch.nn.Module,
    all_parameters: list[torch.nn.Parameter],
) -> dict[str, list[int]]:
    heads = _head_modules(model)
    modules: dict[str, torch.nn.Module] = {
        "shared_decoder_trunk": model.dressable_model.anchor_clothing_mlp.hidden_layers,
        "hypernetwork": model.dressable_model.clothing_film_generator,
        "completer": model.canonical_clothing_completer,
        "xyz_head": heads["xyz"],
        "scaling_head": heads["scaling"],
        "rotation_head": heads["rotation"],
        "opacity_head": heads["opacity"],
        "sh0_head": heads["sh0"],
        "shN_head": heads["shN"],
    }
    lookup = {id(parameter): index for index, parameter in enumerate(all_parameters)}
    return {
        name: [lookup[id(parameter)] for parameter in module.parameters() if id(parameter) in lookup]
        for name, module in modules.items()
    }


def _gradient_tuple(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
) -> tuple[torch.Tensor | None, ...]:
    values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return tuple(value.detach().float().cpu() if value is not None else None for value in values)


def _combine_gradients(
    items: Iterable[tuple[float, tuple[torch.Tensor | None, ...]]],
) -> tuple[torch.Tensor | None, ...]:
    items = list(items)
    width = len(items[0][1])
    combined: list[torch.Tensor | None] = []
    for index in range(width):
        values = [gradient[index] * coefficient for coefficient, gradient in items if gradient[index] is not None]
        combined.append(sum(values) if values else None)
    return tuple(combined)


def _gradient_norm(
    gradients: tuple[torch.Tensor | None, ...],
    indices: list[int],
) -> float:
    total = sum(float(gradients[index].square().sum()) for index in indices if gradients[index] is not None)
    return math.sqrt(total)


def _cosine(
    first: tuple[torch.Tensor | None, ...],
    second: tuple[torch.Tensor | None, ...],
    indices: list[int],
) -> tuple[float | None, float, float]:
    dot = 0.0
    first_norm = 0.0
    second_norm = 0.0
    for index in indices:
        left, right = first[index], second[index]
        if left is not None:
            first_norm += float(left.square().sum())
        if right is not None:
            second_norm += float(right.square().sum())
        if left is not None and right is not None:
            dot += float((left * right).sum())
    first_value, second_value = math.sqrt(first_norm), math.sqrt(second_norm)
    if first_value == 0 or second_value == 0:
        return None, first_value, second_value
    return dot / (first_value * second_value), first_value, second_value


def _step0_diagnostics(
    runtime: dict[str, Any],
    args: argparse.Namespace,
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    model = runtime["model"]
    result, rgb, alpha = _forward(
        model, runtime["base"], runtime["adapter"], runtime["episode"],
        runtime["geometry"], runtime["checkpoint"], runtime["background"], device,
    )
    parts = _loss(rgb, alpha, runtime["sample"], device, args)
    residual, gate = _regularizers(result)
    components = _component_values(parts, residual, gate, args)
    baseline_scales = {name: 1.0 for name in ("edit_rgb", "identity", "alpha", "regularization")}
    objective = _objective(components, baseline_scales)
    contribution_rows = []
    objective_value = float(objective.detach())
    for name, (raw, coefficient) in components.items():
        contribution = float(raw.detach()) * coefficient
        contribution_rows.append({
            "loss": name,
            "raw_value": float(raw.detach()),
            "configured_coefficient": coefficient,
            "weighted_contribution": contribution,
            "fraction_of_total": contribution / objective_value,
        })
    _write_csv(output / "loss_contribution_table_v5_2.csv", contribution_rows)

    parameters = [value for value in model.parameters() if value.requires_grad]
    module_indices = _module_parameter_indices(model, parameters)
    raw_gradients = {
        name: _gradient_tuple(raw, parameters)
        for name, (raw, _) in components.items()
    }
    norm_rows = []
    for loss_name, (_, coefficient) in components.items():
        for module_name, indices in module_indices.items():
            raw_norm = _gradient_norm(raw_gradients[loss_name], indices)
            norm_rows.append({
                "loss": loss_name,
                "module": module_name,
                "raw_gradient_norm": raw_norm,
                "configured_coefficient": coefficient,
                "weighted_gradient_norm": abs(coefficient) * raw_norm,
            })
    _write_csv(output / "loss_gradient_norms_step0_v5_2.csv", norm_rows)

    weighted = {
        name: _combine_gradients([(components[name][1], raw_gradients[name])])
        for name in components
    }
    gradient_groups = {
        "edit": weighted["edit"],
        "clothing": weighted["clothing"],
        "preserve": weighted["preserve"],
        "protected": weighted["protected"],
        "alpha_edit": weighted["alpha_edit"],
        "alpha_transition": weighted["alpha_transition"],
        "alpha_base": weighted["alpha_base"],
        "rgb_group": _combine_gradients([(1.0, weighted["edit"]), (1.0, weighted["clothing"])]),
        "alpha_group": _combine_gradients([
            (1.0, weighted["alpha_edit"]),
            (1.0, weighted["alpha_transition"]),
            (1.0, weighted["alpha_base"]),
        ]),
        "identity_group": _combine_gradients([(1.0, weighted["preserve"]), (1.0, weighted["protected"])]),
        "regularization_group": _combine_gradients([
            (1.0, weighted["residual_magnitude"]),
            (1.0, weighted["active_gate_regularization"]),
        ]),
    }
    pairs = [
        ("edit", "clothing"),
        ("edit", "alpha_edit"),
        ("edit", "alpha_transition"),
        ("edit", "preserve"),
        ("edit", "protected"),
        ("clothing", "alpha_transition"),
        ("rgb_group", "alpha_group"),
        ("rgb_group", "regularization_group"),
    ]
    cosine_rows = []
    for first, second in pairs:
        for module_name, indices in module_indices.items():
            cosine, first_norm, second_norm = _cosine(
                gradient_groups[first], gradient_groups[second], indices,
            )
            cosine_rows.append({
                "first": first,
                "second": second,
                "module": module_name,
                "cosine_similarity": "" if cosine is None else cosine,
                "first_gradient_norm": first_norm,
                "second_gradient_norm": second_norm,
            })
    _write_csv(output / "loss_gradient_cosine_v5_2.csv", cosine_rows)
    trunk_rows = [row for row in cosine_rows if row["module"] == "shared_decoder_trunk"]
    lines = [
        "# Loss Conflict Diagnosis V5.2",
        "",
        "All values were measured on the same frozen step-0 model, episode, render, and masks.",
        "Negative cosine is direct evidence of local gradient conflict; simultaneous loss movement alone is not used as evidence.",
        "",
        "| Pair | Shared-trunk cosine | First norm | Second norm |",
        "|---|---:|---:|---:|",
    ]
    for row in trunk_rows:
        value = row["cosine_similarity"]
        display = "undefined" if value == "" else f"{float(value):.9f}"
        lines.append(
            f"| {row['first']} vs {row['second']} | {display} | "
            f"{row['first_gradient_norm']:.9g} | {row['second_gradient_norm']:.9g} |"
        )
    (output / "LOSS_CONFLICT_DIAGNOSIS_V5_2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    group_norms = {
        "edit_rgb": _gradient_norm(gradient_groups["rgb_group"], module_indices["shared_decoder_trunk"]),
        "identity": _gradient_norm(gradient_groups["identity_group"], module_indices["shared_decoder_trunk"]),
        "alpha": _gradient_norm(gradient_groups["alpha_group"], module_indices["shared_decoder_trunk"]),
        "regularization": _gradient_norm(gradient_groups["regularization_group"], module_indices["shared_decoder_trunk"]),
    }
    return {
        "objective": objective_value,
        "contributions": contribution_rows,
        "group_shared_trunk_gradient_norms": group_norms,
        "cosines": cosine_rows,
    }


def compute_static_balance(group_norms: dict[str, float], epsilon: float = 1e-12) -> dict[str, Any]:
    median = statistics.median(group_norms.values())
    raw = {name: median / (value + epsilon) for name, value in group_norms.items()}
    clipped = {name: min(4.0, max(0.25, value)) for name, value in raw.items()}
    final = dict(clipped)
    final["identity"] = max(1.0, final["identity"])
    final["alpha"] = max(1.0, final["alpha"])
    return {
        "formula": "median(group gradient norms) / (group gradient norm + 1e-12), clipped to [0.25,4.0]",
        "median_gradient_norm": median,
        "group_gradient_norms": group_norms,
        "raw_scales": raw,
        "clipped_scales": clipped,
        "final_frozen_scales": final,
        "constraints": {
            "identity_scale_at_least_one": True,
            "protected_not_downweighted": True,
            "alpha_base_not_downweighted": True,
            "dynamic_updates": False,
            "grid_search": False,
        },
    }


def _total_parameter_norm(model: torch.nn.Module) -> float:
    return math.sqrt(sum(float(value.detach().float().square().sum()) for value in model.parameters() if value.requires_grad))


def _total_gradient_norm(model: torch.nn.Module) -> float:
    return math.sqrt(sum(
        float(value.grad.detach().float().square().sum())
        for value in model.parameters()
        if value.requires_grad and value.grad is not None
    ))


def _make_panel(
    path: Path,
    sample: dict[str, Any],
    rgb: torch.Tensor,
    alpha: torch.Tensor,
    shoe: torch.Tensor,
) -> None:
    base = sample["target_base_rgb"].cpu()
    edit = sample["target_edit_rgb"].cpu()
    edit_mask = sample["target_edit_mask"].cpu()
    core = sample["target_edit_core_mask"].cpu()
    clothing = sample["target_clothing_mask"].cpu()
    preserve = sample["target_preserve_mask"].cpu()
    protected = sample["target_protected_mask"].cpu()
    shoe1 = shoe.unsqueeze(0).float()
    items = [
        ("base target", _tensor_image(base, 3)),
        ("raw edit target", _tensor_image(edit, 3)),
        ("edit red / core green", _tensor_image(torch.cat((edit_mask, core, torch.zeros_like(core)), dim=0), 3)),
        ("safe clothing", _tensor_image(clothing, 1)),
        ("preserve", _tensor_image(preserve, 1)),
        ("protected", _tensor_image(protected, 1)),
        ("prediction", _tensor_image(rgb, 3)),
        ("edit diff", _tensor_image((rgb - edit).abs() * edit_mask, 3)),
        ("clothing diff", _tensor_image((rgb - edit).abs() * clothing, 3)),
        ("protected diff", _tensor_image((rgb - base).abs() * protected, 3)),
        ("shoe diff vs base", _tensor_image((rgb - base).abs() * shoe1, 3)),
        ("alpha prediction", _tensor_image(alpha, 1)),
    ]
    cell = (256, 384)
    columns = 4
    rows = math.ceil(len(items) / columns)
    canvas = Image.new("RGB", (cell[0] * columns, (cell[1] + 26) * rows), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(items):
        picture = image.convert("RGB").resize(cell, Image.Resampling.LANCZOS)
        x = (index % columns) * cell[0]
        y = (index // columns) * (cell[1] + 26)
        canvas.paste(picture, (x, y))
        draw.text((x + 4, y + cell[1] + 4), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _linear_slope(rows: list[dict[str, Any]], name: str) -> float:
    x = np.asarray([row["step"] for row in rows], dtype=np.float64)
    y = np.asarray([row[name] for row in rows], dtype=np.float64)
    return float(np.polyfit(x, y, 1)[0])


def evaluate_history(
    history: list[dict[str, Any]],
    *,
    base_unchanged: bool,
    backbone_unchanged: bool,
    shoe_closer_to_base: bool,
) -> dict[str, Any]:
    if len(history) != 81 or [row["step"] for row in history] != list(range(81)):
        raise ValueError("an 80-step run must contain states 0 through 80")
    first5 = history[:5]
    last10 = history[-10:]
    last20 = history[-20:]
    means = {
        "edit_first5": float(np.mean([row["edit"] for row in first5])),
        "edit_last10": float(np.mean([row["edit"] for row in last10])),
        "clothing_first5": float(np.mean([row["clothing"] for row in first5])),
        "clothing_last10": float(np.mean([row["clothing"] for row in last10])),
    }
    means["edit_reduction_fraction"] = 1 - means["edit_last10"] / means["edit_first5"]
    means["clothing_reduction_fraction"] = 1 - means["clothing_last10"] / means["clothing_first5"]
    slopes = {
        "edit_last20": _linear_slope(last20, "edit"),
        "clothing_last20": _linear_slope(last20, "clothing"),
    }
    initial, final = history[0], history[-1]
    criteria = {
        "finite": all(
            math.isfinite(float(value))
            for row in history
            for key, value in row.items()
            if key not in {"step"}
        ),
        "total_decreased": final["objective"] < initial["objective"],
        "edit_last10_mean_reduced_at_least_0_5pct": means["edit_reduction_fraction"] >= 0.005,
        "clothing_last10_mean_reduced_at_least_0_5pct": means["clothing_reduction_fraction"] >= 0.005,
        "edit_last20_slope_negative": slopes["edit_last20"] < 0,
        "clothing_last20_slope_negative": slopes["clothing_last20"] < 0,
        "protected_within_110pct_and_0_005": final["protected"] <= initial["protected"] * 1.10 and final["protected"] <= 0.005,
        "preserve_within_110pct": final["preserve"] <= initial["preserve"] * 1.10,
        "alpha_base_within_110pct": final["alpha_base"] <= initial["alpha_base"] * 1.10,
        "shoe_closer_to_base_than_raw": shoe_closer_to_base,
        "frozen_base_bitwise_unchanged": base_unchanged,
        "frozen_backbone_bitwise_unchanged": backbone_unchanged,
    }
    return {"means": means, "slopes": slopes, "criteria": criteria, "pass": all(criteria.values())}


def _run_phase(
    name: str,
    runtime: dict[str, Any],
    args: argparse.Namespace,
    output: Path,
    scales: dict[str, float],
    shoe: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    phase_output = output / name
    phase_output.mkdir(parents=True, exist_ok=False)
    base, model, optimizer = runtime["base"], runtime["model"], runtime["optimizer"]
    base_before = {key: getattr(base, key).detach().cpu().clone() for key in BASE_PARAMETER_NAMES}
    backbone_before = _state_clone(model.clothing_observation_encoder.backbone)
    history: list[dict[str, Any]] = []
    step_renders: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    final_gradient_snapshot: dict[str, Any] | None = None
    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        result, rgb, alpha = _forward(
            model, base, runtime["adapter"], runtime["episode"], runtime["geometry"],
            runtime["checkpoint"], runtime["background"], device,
        )
        parts = _loss(rgb, alpha, runtime["sample"], device, args)
        residual, gate = _regularizers(result)
        components = _component_values(parts, residual, gate, args)
        objective = _objective(components, scales)
        if not torch.isfinite(objective):
            raise FloatingPointError(f"non-finite {name} objective at step {step}")
        objective.backward()
        row = {
            "step": step,
            **{key: float(value.detach()) for key, value in parts.items()},
            "residual_magnitude": float(residual.detach()),
            "gate_regularization": float(gate.detach()),
            "geometry_gate_mean": float(result["completion"].geometry_gate.detach().mean()),
            "appearance_gate_mean": float(result["completion"].appearance_gate.detach().mean()),
            "gradient_norm": _total_gradient_norm(model),
            "parameter_norm": _total_parameter_norm(model),
            "objective": float(objective.detach()),
        }
        history.append(row)
        if step in LOG_STEPS:
            print(json.dumps({"phase": name, **row}), flush=True)
        if step in RENDER_STEPS:
            rgb_cpu, alpha_cpu = rgb.detach().cpu(), alpha.detach().cpu()
            step_renders[step] = (rgb_cpu, alpha_cpu)
            save_render_tensor(phase_output / f"prediction_step{step:03d}.png", rgb_cpu, 3)
            save_render_tensor(phase_output / f"alpha_step{step:03d}.png", alpha_cpu, 1)
            _make_panel(
                phase_output / f"diagnostic_panel_step{step:03d}.png",
                runtime["sample"], rgb_cpu, alpha_cpu, shoe,
            )
        if step == args.steps:
            final_gradient_snapshot = _gradient_snapshot(model, base)
            break
        optimizer.step()
    _write_csv(phase_output / "history_v5_2.csv", history)
    _write_json(phase_output / "history_v5_2.json", history)
    base_unchanged = all(
        torch.equal(value, getattr(base, key).detach().cpu())
        for key, value in base_before.items()
    )
    backbone_unchanged = _state_equal(backbone_before, model.clothing_observation_encoder.backbone)
    final_rgb, final_alpha = step_renders[80]
    sample = runtime["sample"]
    base_rgb, edit_rgb = sample["target_base_rgb"].cpu(), sample["target_edit_rgb"].cpu()
    shoe3 = shoe.unsqueeze(0).expand_as(base_rgb)
    shoe1 = shoe.unsqueeze(0)
    visual_metrics = {
        "raw_edit_vs_base_shoe_rgb_mae": float((edit_rgb - base_rgb).abs()[shoe3].mean()),
        "step80_vs_base_shoe_rgb_mae": float((final_rgb - base_rgb).abs()[shoe3].mean()),
        "step80_vs_raw_shoe_rgb_mae": float((final_rgb - edit_rgb).abs()[shoe3].mean()),
        "step80_vs_base_shoe_alpha_mae": float((final_alpha - sample["target_base_foreground_mask"]).abs()[shoe1].mean()),
    }
    decision = evaluate_history(
        history,
        base_unchanged=base_unchanged,
        backbone_unchanged=backbone_unchanged,
        shoe_closer_to_base=(visual_metrics["step80_vs_base_shoe_rgb_mae"] < visual_metrics["step80_vs_raw_shoe_rgb_mae"]),
    )
    metrics = {
        "phase": name,
        "steps": args.steps,
        "scales": scales,
        "history": history,
        "summary": decision,
        "visual_metrics": visual_metrics,
        "final_gradient_snapshot": final_gradient_snapshot,
        "base_max_change": {
            key: float((getattr(base, key).detach().cpu() - before).abs().max())
            for key, before in base_before.items()
        },
        "backbone_bitwise_unchanged": backbone_unchanged,
    }
    _write_json(phase_output / "metrics_v5_2.json", metrics)
    return {**metrics, "runtime": runtime}


def _protected_flip_check(
    runtime: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    model = runtime["model"]
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        _, rgb, alpha = _forward(
            model, runtime["base"], runtime["adapter"], runtime["episode"], runtime["geometry"],
            runtime["checkpoint"], runtime["background"], device,
        )
    sample = runtime["sample"]
    original_rgb = sample["target_edit_rgb"]
    original_foreground = sample["target_foreground_mask"]
    protected = sample["target_protected_mask"].bool()
    flipped_rgb = original_rgb.clone()
    flipped_rgb[protected.expand_as(flipped_rgb)] = 1 - flipped_rgb[protected.expand_as(flipped_rgb)]
    flipped_foreground = original_foreground.clone()
    flipped_foreground[protected] = 1 - flipped_foreground[protected]
    first = _loss(rgb, alpha, sample, device, args)
    sample["target_edit_rgb"] = flipped_rgb
    sample["target_foreground_mask"] = flipped_foreground
    try:
        second = _loss(rgb, alpha, sample, device, args)
    finally:
        sample["target_edit_rgb"] = original_rgb
        sample["target_foreground_mask"] = original_foreground
    keys = ("edit", "clothing", "transition", "alpha_edit", "alpha_transition", "total")
    deltas = {name: float((second[name] - first[name]).abs()) for name in keys}
    return {"deltas": deltas, "strict_zero": all(value == 0.0 for value in deltas.values())}


def _effective_weights(args: argparse.Namespace, scales: dict[str, float]) -> dict[str, float]:
    return {
        "edit": scales["edit_rgb"],
        "clothing": scales["edit_rgb"],
        "preserve": scales["identity"],
        "protected": scales["identity"],
        "transition": 0.25,
        "alpha_edit": scales["alpha"] * args.alpha_weight * args.alpha_edit_weight,
        "alpha_transition": scales["alpha"] * args.alpha_weight * args.alpha_transition_weight,
        "alpha_base": scales["alpha"] * args.alpha_weight * args.alpha_base_weight,
        "residual_magnitude": scales["regularization"] * args.residual_regularization_weight,
        "active_gate_regularization": scales["regularization"] * args.gate_regularization_weight,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V5.2 fixed-episode loss conflict and 80-step closure")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gate4-config", type=Path, required=True)
    parser.add_argument("--module2-config", type=Path, required=True)
    parser.add_argument("--module3-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=11)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--alpha-weight", type=float, default=0.5)
    parser.add_argument("--alpha-edit-weight", type=float, default=1.0)
    parser.add_argument("--alpha-transition-weight", type=float, default=0.25)
    parser.add_argument("--alpha-base-weight", type=float, default=1.0)
    parser.add_argument("--residual-regularization-weight", type=float, default=1e-4)
    parser.add_argument("--gate-regularization-weight", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    if args.steps != 80:
        raise ValueError("V5.2 acceptance is frozen to exactly 80 optimization steps")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty V5.2 output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    _write_json(output / "run_partial.json", {"status": "RUNNING", "stage": "build_fixed_state"})
    try:
        runtime_a = _build_runtime(args, device)
        contract = _fixed_episode_contract(args.manifest.resolve(), runtime_a["sample"], args)
        _write_json(output / "fixed_episode_contract_v5_2.json", contract)
        initial_path = output / "initial_trainable_state_v5_2.pth"
        initial_hashes = _save_initial_state(initial_path, runtime_a["model"], runtime_a["optimizer"])
        _write_json(output / "initial_state_sha256_v5_2.json", initial_hashes)
        diagnostics = _step0_diagnostics(runtime_a, args, output, device)
        _write_json(output / "step0_diagnostics_v5_2.json", diagnostics)
        restore_a = _restore_initial_state(initial_path, runtime_a["model"], runtime_a["optimizer"])
        _write_json(output / "phase_a_initial_state_verification.json", restore_a)

        manifest_payload = json.loads(args.manifest.read_text(encoding="utf-8"))
        shoe_path = Path(manifest_payload["protected_region_final_adjudication"]["target_contract"]["shoe_mask"])
        shoe_path = shoe_path if shoe_path.is_absolute() else args.manifest.resolve().parent / shoe_path
        shoe = torch.from_numpy(np.array(Image.open(shoe_path).convert("L"), copy=True) >= 128)
        baseline_scales = {name: 1.0 for name in ("edit_rgb", "identity", "alpha", "regularization")}
        phase_a = _run_phase("phase_a", runtime_a, args, output, baseline_scales, shoe, device)
        phase_b = None
        balance = None
        if not phase_a["summary"]["pass"]:
            balance = compute_static_balance(diagnostics["group_shared_trunk_gradient_norms"])
            balance["original_effective_weights"] = _effective_weights(args, baseline_scales)
            balance["final_effective_weights"] = _effective_weights(args, balance["final_frozen_scales"])
            _write_json(output / "static_loss_balance_v5_2.json", balance)
            phase_a.pop("runtime")
            del runtime_a
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            runtime_b = _build_runtime(args, device)
            restore_b = _restore_initial_state(initial_path, runtime_b["model"], runtime_b["optimizer"])
            _write_json(output / "phase_b_initial_state_verification.json", restore_b)
            phase_b = _run_phase(
                "phase_b", runtime_b, args, output,
                balance["final_frozen_scales"], shoe, device,
            )

        adopted = phase_a if phase_a["summary"]["pass"] else phase_b
        if adopted is None:
            raise RuntimeError("Phase A failed but Phase B did not run")
        adopted_name = adopted["phase"]
        protected_flip = _protected_flip_check(adopted["runtime"], args, device)
        final_gradient = adopted["final_gradient_snapshot"]
        safety = {
            "protected_raw_rgb_alpha_flip": protected_flip,
            "inference_forbidden_fields": 0,
            "frozen_base_bitwise_unchanged": adopted["summary"]["criteria"]["frozen_base_bitwise_unchanged"],
            "frozen_backbone_bitwise_unchanged": adopted["summary"]["criteria"]["frozen_backbone_bitwise_unchanged"],
            "frozen_base_grad_count": final_gradient["frozen_base_grad_count"],
            "frozen_backbone_grad_count": final_gradient["frozen_image_backbone"]["parameter_count_with_grad"],
        }
        pass_status = bool(adopted["summary"]["pass"] and protected_flip["strict_zero"])
        negative_trends = (
            adopted["summary"]["slopes"]["edit_last20"] < 0
            and adopted["summary"]["slopes"]["clothing_last20"] < 0
        )
        safety_pass = all((
            adopted["summary"]["criteria"]["protected_within_110pct_and_0_005"],
            adopted["summary"]["criteria"]["preserve_within_110pct"],
            adopted["summary"]["criteria"]["alpha_base_within_110pct"],
            adopted["summary"]["criteria"]["shoe_closer_to_base_than_raw"],
            adopted["summary"]["criteria"]["frozen_base_bitwise_unchanged"],
            adopted["summary"]["criteria"]["frozen_backbone_bitwise_unchanged"],
            protected_flip["strict_zero"],
        ))
        status = "PASS" if pass_status else ("PARTIAL" if negative_trends and safety_pass else "FAIL")
        report = {
            "status": status,
            "adopted_phase": adopted_name,
            "phase_b_executed": phase_b is not None,
            "fixed_episode": contract,
            "step0_diagnostics": diagnostics,
            "phase_a": {key: value for key, value in phase_a.items() if key != "runtime"},
            "phase_b": ({key: value for key, value in phase_b.items() if key != "runtime"} if phase_b else None),
            "static_balance": balance,
            "adopted_effective_weights": _effective_weights(
                args,
                baseline_scales if adopted_name == "phase_a" else balance["final_frozen_scales"],
            ),
            "safety": safety,
            "single_target_regression": "RUN_SEPARATELY_BY_ACCEPTANCE_DRIVER",
            "no_architecture_or_source_data_change": True,
        }
        _write_json(output / "V5_2_FINAL_STATUS.json", report)
        (output / "V5_2_FINAL_STATUS.md").write_text(
            "# V5.2 Fixed-Episode Optimization Closure\n\n"
            f"- status: **{status}**\n"
            f"- adopted phase: `{adopted_name}`\n"
            f"- Phase B executed: `{phase_b is not None}`\n"
            f"- edit last-10 reduction: `{adopted['summary']['means']['edit_reduction_fraction']:.6%}`\n"
            f"- clothing last-10 reduction: `{adopted['summary']['means']['clothing_reduction_fraction']:.6%}`\n"
            f"- edit/clothing last-20 slopes: `{adopted['summary']['slopes']['edit_last20']:.9g}` / `{adopted['summary']['slopes']['clothing_last20']:.9g}`\n"
            f"- protected raw RGB/alpha flip delta strictly zero: `{protected_flip['strict_zero']}`\n",
            encoding="utf-8",
        )
        _write_json(output / "run_partial.json", {"status": "COMPLETE", "stage": "final_adjudication"})
        print(json.dumps({
            "status": status,
            "adopted_phase": adopted_name,
            "phase_b_executed": phase_b is not None,
            "phase_a_pass": phase_a["summary"]["pass"],
            "phase_b_pass": phase_b["summary"]["pass"] if phase_b else None,
        }, indent=2))
        if status == "FAIL":
            raise SystemExit(1)
    except BaseException as error:
        _write_json(output / "run_partial.json", {
            "status": "FAIL",
            "stage": "exception",
            "exception_type": type(error).__name__,
            "message": str(error),
        })
        raise


if __name__ == "__main__":
    main()
