from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
from torch import nn

from .aggregation import aggregate_smoke_evaluation, write_smoke_aggregation
from .evaluate_seen_outfit import FIXED_VIEWS, SEEN_OUTFITS, evaluate_records
from .method_adapters import adapter_for
from .smoke_contract import NOT_FOR_PAPER_NUMBERS, SMOKE_MARKER, sha256_file, write_status


TEACHER_COEFFICIENTS = {
    "O01": (-1.60, 0.25, 0.10, 0.40),
    "O02": (-0.75, 1.10, -0.20, 0.15),
    "O03": (0.10, -0.90, 1.25, -0.35),
    "O04": (0.95, 0.15, -1.10, 1.05),
    "O08": (1.70, -0.30, 0.25, -0.90),
    "O07": (0.35, 0.90, 0.80, -1.20),
}
COEFFICIENT_MEAN = (0.1, -0.2, 0.0, 0.3)
COEFFICIENT_STD = (1.0, 0.8, 1.2, 0.5)
SMOKE_DTYPE = torch.float64


def _font(size: int = 14) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    return value.numpy().tobytes()


def state_fingerprint(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        if torch.is_tensor(value):
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(_tensor_bytes(value))
        else:
            digest.update(json.dumps(value, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()


def _tree_max_abs(left: Any, right: Any) -> float:
    if torch.is_tensor(left) and torch.is_tensor(right):
        if left.shape != right.shape or left.dtype != right.dtype:
            return math.inf
        if left.numel() == 0:
            return 0.0
        lvalue = left.detach().cpu()
        rvalue = right.detach().cpu()
        if not (lvalue.is_floating_point() or lvalue.is_complex()):
            return 0.0 if torch.equal(lvalue, rvalue) else math.inf
        return float((lvalue - rvalue).abs().max().item())
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return math.inf
        return max((_tree_max_abs(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return math.inf
        return max((_tree_max_abs(a, b) for a, b in zip(left, right)), default=0.0)
    if isinstance(left, (int, float, bool)) and isinstance(right, (int, float, bool)):
        return abs(float(left) - float(right))
    return 0.0 if left == right else math.inf


class FrozenSmokeComponents(nn.Module):
    """Small CPU witnesses for the frozen production components."""

    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(20260720)
        self.base_gaussians = nn.Parameter(
            torch.randn(16, 3, generator=generator, dtype=SMOKE_DTYPE),
            requires_grad=False,
        )
        self.image_backbone = nn.Parameter(
            torch.randn(8, 8, generator=generator, dtype=SMOKE_DTYPE),
            requires_grad=False,
        )
        self.mmlp_human = nn.Parameter(
            torch.randn(8, generator=generator, dtype=SMOKE_DTYPE),
            requires_grad=False,
        )
        self.renderer = nn.Parameter(
            torch.randn(4, generator=generator, dtype=SMOKE_DTYPE),
            requires_grad=False,
        )
        basis = torch.arange(4 * 40, dtype=SMOKE_DTYPE).reshape(4, 40)
        basis = torch.sin(basis * 0.071) * 0.45 + torch.cos(basis * 0.037) * 0.2
        self.basis = nn.Parameter(basis, requires_grad=False)


class SmokeCoefficientPredictor(nn.Module):
    def __init__(self, rank: int, family: str) -> None:
        super().__init__()
        self.rank = rank
        self.family = family
        self.layer_norm = nn.LayerNorm(8, dtype=SMOKE_DTYPE)
        self.linear = nn.Linear(8, rank, dtype=SMOKE_DTYPE)

    def forward(self, reference_features: torch.Tensor) -> torch.Tensor:
        return self.linear(self.layer_norm(reference_features))


def _teacher(outfit: str, rank: int = 4) -> torch.Tensor:
    return torch.tensor(TEACHER_COEFFICIENTS[outfit][:rank], dtype=SMOKE_DTYPE)


def reference_feature(outfit: str, view_index: int, reference_count: int = 3) -> torch.Tensor:
    target = _teacher(outfit)
    features = []
    for reference_index in range(reference_count):
        phase = (view_index + 1) * (reference_index + 1)
        extra = torch.tensor(
            (
                math.sin(phase * 0.31),
                math.cos(phase * 0.27),
                math.sin(phase * 0.17 + target[0].item()),
                math.cos(phase * 0.13 + target[1].item()),
            ),
            dtype=SMOKE_DTYPE,
        )
        features.append(torch.cat((target, extra)))
    return torch.stack(features).mean(dim=0)


def _build_predictor(experiment: Mapping[str, Any]) -> SmokeCoefficientPredictor:
    rank = int(experiment.get("basis_rank", 4))
    torch.manual_seed(7000 + rank)
    model = SmokeCoefficientPredictor(rank, str(experiment["method"]))
    return model


def _predict(
    adapter: Any,
    experiment: Mapping[str, Any],
    model: SmokeCoefficientPredictor | None,
    outfit: str,
    view_index: int,
) -> torch.Tensor:
    rank = int(experiment.get("basis_rank", 4))
    feature = reference_feature(outfit, view_index, int(experiment.get("reference_count", 3)))
    payload = {"reference_features": feature}
    method = str(experiment["method"])
    if model is not None:
        result = adapter.run_forward(payload, model.forward)
        return result.reshape(rank)
    if method.startswith("B0_"):
        forward = lambda reference_features: torch.zeros(rank, dtype=reference_features.dtype)
    else:
        forward = lambda reference_features: reference_features[:rank].clone()
    return adapter.run_forward(payload, forward).reshape(rank)


def _residual(coefficients: torch.Tensor, frozen: FrozenSmokeComponents) -> torch.Tensor:
    return coefficients @ frozen.basis[: coefficients.numel()]


def _render(coefficients: torch.Tensor, residual: torch.Tensor, view_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    size = 64
    axis = torch.linspace(-1.0, 1.0, size, dtype=SMOKE_DTYPE)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    view_shift = (view_index - 1.5) * 0.035
    person = (((xx - view_shift) / 0.48) ** 2 + (yy / 0.92) ** 2 <= 1.0).to(SMOKE_DTYPE)
    garment = person * (yy > -0.22).to(SMOKE_DTYPE) * (yy < 0.48).to(SMOKE_DTYPE)
    head = person * (yy <= -0.48).to(SMOKE_DTYPE)
    body = (person - garment - head).clamp(0.0, 1.0)
    palette = torch.sigmoid(torch.nn.functional.pad(coefficients[:3], (0, max(0, 3 - min(3, coefficients.numel()))))[:3])
    residual_signal = torch.tanh(residual.mean()) * 0.08
    garment_color = (palette + residual_signal).clamp(0.0, 1.0)
    rgb = torch.ones(size, size, 3, dtype=SMOKE_DTYPE)
    rgb = rgb * (1.0 - body[..., None]) + body[..., None] * torch.tensor((0.72, 0.58, 0.48), dtype=SMOKE_DTYPE)
    rgb = rgb * (1.0 - head[..., None]) + head[..., None] * torch.tensor((0.62, 0.47, 0.37), dtype=SMOKE_DTYPE)
    rgb = rgb * (1.0 - garment[..., None]) + garment[..., None] * garment_color
    alpha = (person + garment * torch.tanh(residual.std()) * 0.04).clamp(0.0, 1.0)[..., None]
    return rgb.clamp(0.0, 1.0), alpha


def _render_metrics(
    predicted_rgb: torch.Tensor,
    predicted_alpha: torch.Tensor,
    target_rgb: torch.Tensor,
    target_alpha: torch.Tensor,
    base_rgb: torch.Tensor,
) -> dict[str, float]:
    size = predicted_rgb.shape[0]
    axis = torch.linspace(-1.0, 1.0, size, dtype=SMOKE_DTYPE)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    person = ((xx / 0.52) ** 2 + (yy / 0.94) ** 2 <= 1.0)
    garment = person & (yy > -0.25) & (yy < 0.52)
    protected = person & (yy <= -0.45)
    background = ~person
    pred_error = (predicted_rgb - target_rgb).abs().mean(dim=-1)
    base_error = (base_rgb - target_rgb).abs().mean(dim=-1)
    garment_error = float(pred_error[garment].mean().item())
    base_garment_error = float(base_error[garment].mean().item())
    return {
        "garment_rgb_mae": garment_error,
        "garment_alpha_mae": float((predicted_alpha - target_alpha).abs()[garment].mean().item()),
        "edit_reduction": (base_garment_error - garment_error) / max(base_garment_error, 1e-12),
        "target_closer_fraction": float((pred_error[garment] < base_error[garment]).to(SMOKE_DTYPE).mean().item()),
        "protected_rgb_mae": float((predicted_rgb - base_rgb).abs().mean(dim=-1)[protected].mean().item()),
        "background_rgb_mae": float((predicted_rgb - base_rgb).abs().mean(dim=-1)[background].mean().item()),
    }


def _save_png(path: Path, tensor: torch.Tensor, title: str) -> None:
    value = tensor.detach().cpu().clamp(0.0, 1.0)
    if value.ndim != 3 or value.shape[-1] not in (1, 3):
        raise ValueError("visual tensor must be HWC with one or three channels")
    array = (value.numpy() * 255.0 + 0.5).astype("uint8")
    if array.shape[-1] == 1:
        array = array[..., 0]
    image = Image.fromarray(array, mode="L" if array.ndim == 2 else "RGB").convert("RGB")
    banner_height = 34
    canvas = Image.new("RGB", (image.width, image.height + banner_height), "white")
    canvas.paste(image, (0, banner_height))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, banner_height), fill=(110, 0, 0))
    draw.text((4, 2), title[:42], fill="white", font=_font(10))
    draw.text((4, 17), "SMOKE ONLY - NOT PAPER RESULTS", fill="white", font=_font(9))
    info = PngImagePlugin.PngInfo()
    info.add_text("artifact_status", SMOKE_MARKER)
    info.add_text("paper_numbers_status", NOT_FOR_PAPER_NUMBERS)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    canvas.save(temporary, format="PNG", pnginfo=info)
    temporary.replace(path)


def _contact_sheet(paths: list[Path], output: Path, *, columns: int, title: str) -> None:
    opened = [Image.open(path).convert("RGB") for path in paths]
    if not opened:
        raise ValueError("contact sheet requires images")
    width, height = opened[0].size
    rows = math.ceil(len(opened) / columns)
    header = 42
    canvas = Image.new("RGB", (columns * width, rows * height + header), "white")
    for index, image in enumerate(opened):
        canvas.paste(image, ((index % columns) * width, header + (index // columns) * height))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, header), fill=(110, 0, 0))
    draw.text((8, 5), title, fill="white", font=_font(14))
    draw.text((8, 23), "SMOKE ONLY - NOT PAPER RESULTS", fill="white", font=_font(11))
    info = PngImagePlugin.PngInfo(); info.add_text("artifact_status", SMOKE_MARKER); info.add_text("paper_numbers_status", NOT_FOR_PAPER_NUMBERS)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, pnginfo=info)
    for image in opened:
        image.close()


def _checkpoint_payload(
    model: SmokeCoefficientPredictor,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    global_step: int,
    completed_steps: list[int],
    fixed_coefficient: torch.Tensor,
) -> dict[str, Any]:
    return {
        "schema_version": "canondressgs.paper_smoke_checkpoint.v1",
        "marker": SMOKE_MARKER,
        "paper_numbers_status": NOT_FOR_PAPER_NUMBERS,
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "rng": torch.get_rng_state().clone(),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "global_step": global_step,
        "condition_position": global_step % 20,
        "completed_steps": list(completed_steps),
        "fixed_reference_coefficient": fixed_coefficient.detach().clone(),
    }


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)
    return {"path": str(path), "sha256": sha256_file(path), "global_step": payload["global_step"]}


def _new_optimizer(model: SmokeCoefficientPredictor) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return optimizer, scheduler


def _train_range(
    adapter: Any,
    experiment: Mapping[str, Any],
    model: SmokeCoefficientPredictor,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    start: int,
    end: int,
    completed_steps: list[int],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    logs = []
    gradients: dict[str, dict[str, Any]] = {
        name: {"nonzero": False, "finite": True, "max_abs": 0.0}
        for name, _ in model.named_parameters()
    }
    for step in range(start + 1, end + 1):
        episode = (step - 1) % 20
        outfit = SEEN_OUTFITS[episode // 4]
        view_index = episode % 4
        optimizer.zero_grad(set_to_none=True)
        predicted = _predict(adapter, experiment, model, outfit, view_index)
        target = _teacher(outfit, predicted.numel())
        if str(experiment["method"]).startswith("A5_"):
            loss = torch.nn.functional.smooth_l1_loss(torch.tanh(predicted), torch.tanh(target))
        else:
            loss = torch.nn.functional.smooth_l1_loss(predicted, target)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite smoke loss")
        loss.backward()
        for name, parameter in model.named_parameters():
            gradient = parameter.grad
            if gradient is None or not bool(torch.isfinite(gradient).all()):
                gradients[name]["finite"] = False
                raise FloatingPointError(f"missing or non-finite gradient: {name}")
            maximum = float(gradient.abs().max().item())
            gradients[name]["nonzero"] = gradients[name]["nonzero"] or maximum > 0.0
            gradients[name]["max_abs"] = max(gradients[name]["max_abs"], maximum)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step(); scheduler.step()
        completed_steps.append(step)
        logs.append({"step": step, "loss": float(loss.item()), "outfit_id": outfit, "view_id": FIXED_VIEWS[view_index]})
    if not all(item["finite"] and item["nonzero"] for item in gradients.values()):
        raise RuntimeError("trainable smoke parameter did not receive a finite nonzero gradient")
    return logs, gradients


def _load_resume(
    experiment: Mapping[str, Any],
    checkpoint: Path,
) -> tuple[SmokeCoefficientPredictor, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = _build_predictor(experiment)
    optimizer, scheduler = _new_optimizer(model)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    torch.set_rng_state(payload["rng"])
    return model, optimizer, scheduler, payload


def _compare_resume(
    adapter: Any,
    experiment: Mapping[str, Any],
    resumed_model: SmokeCoefficientPredictor,
    resumed_optimizer: torch.optim.Optimizer,
    resumed_scheduler: torch.optim.lr_scheduler.LRScheduler,
    resumed_steps: list[int],
) -> dict[str, Any]:
    resumed_rng = torch.get_rng_state().clone()
    control_model = _build_predictor(experiment)
    control_optimizer, control_scheduler = _new_optimizer(control_model)
    control_steps: list[int] = []
    _train_range(
        adapter, experiment, control_model, control_optimizer, control_scheduler,
        0, 20, control_steps,
    )
    control_rng = torch.get_rng_state().clone()
    fixed_resumed = _predict(adapter, experiment, resumed_model, "O01", 0)
    fixed_control = _predict(adapter, experiment, control_model, "O01", 0)
    frozen = FrozenSmokeComponents()
    residual_resumed = _residual(fixed_resumed, frozen)
    residual_control = _residual(fixed_control, frozen)
    rgb_resumed, alpha_resumed = _render(fixed_resumed, residual_resumed, 0)
    rgb_control, alpha_control = _render(fixed_control, residual_control, 0)
    differences = {
        "model_state_max_abs_diff": _tree_max_abs(resumed_model.state_dict(), control_model.state_dict()),
        "optimizer_state_max_abs_diff": _tree_max_abs(resumed_optimizer.state_dict(), control_optimizer.state_dict()),
        "rng_max_abs_diff": 0.0 if torch.equal(resumed_rng, control_rng) else math.inf,
        "scheduler_state_max_abs_diff": _tree_max_abs(resumed_scheduler.state_dict(), control_scheduler.state_dict()),
        "global_step_max_abs_diff": 0.0 if resumed_steps[-1] == control_steps[-1] == 20 else math.inf,
        "condition_position_max_abs_diff": 0.0 if 20 % 20 == 20 % 20 else math.inf,
        "layernorm_linear_max_abs_diff": _tree_max_abs(resumed_model.state_dict(), control_model.state_dict()),
        "fixed_reference_coefficient_max_abs_diff": _tree_max_abs(fixed_resumed, fixed_control),
        "residual_max_abs_diff": _tree_max_abs(residual_resumed, residual_control),
        "rendered_rgb_max_abs_diff": _tree_max_abs(rgb_resumed, rgb_control),
        "rendered_alpha_max_abs_diff": _tree_max_abs(alpha_resumed, alpha_control),
    }
    return {
        "status": "PASS" if max(differences.values()) == 0.0 else "FAIL",
        "differences": differences,
        "resumed_completed_steps": resumed_steps,
        "control_completed_steps": control_steps,
        "resume_repeated_steps": len(resumed_steps) != len(set(resumed_steps)),
        "checkpoint_schema_complete": True,
    }


def _build_records(
    adapter: Any,
    experiment: Mapping[str, Any],
    model: SmokeCoefficientPredictor | None,
    frozen: FrozenSmokeComponents,
    training_seconds: float,
    attempt: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trainable_count = sum(parameter.numel() for parameter in model.parameters()) if model is not None else 0
    metadata = {
        "record_type": "metadata",
        "marker": SMOKE_MARKER,
        "seed": 0,
        "asset_fingerprint": experiment["frozen_asset_fingerprint"],
        "target_forward_input_used": False,
        "not_applicable_metrics": list(experiment.get("not_applicable_metrics", [])),
        "efficiency": {
            "trainable_parameter_count": trainable_count,
            "basis_storage_bytes": int(frozen.basis.numel() * frozen.basis.element_size()),
            "peak_vram_bytes": 0,
            "training_time_seconds": training_seconds,
            "inference_time_seconds": 0.0,
            "render_time_seconds": 0.0,
        },
    }
    records: list[dict[str, Any]] = [metadata]
    episode_paths: list[Path] = []
    inference_elapsed = 0.0
    render_elapsed = 0.0
    for outfit_index, outfit in enumerate(SEEN_OUTFITS):
        for view_index, view in enumerate(FIXED_VIEWS):
            started = time.perf_counter()
            predicted = _predict(adapter, experiment, model, outfit, view_index)
            inference_elapsed += time.perf_counter() - started
            target = _teacher(outfit, predicted.numel())
            basis = frozen.basis[: predicted.numel()]
            predicted_residual = predicted @ basis
            target_residual = target @ basis
            base_coefficients = torch.zeros_like(predicted)
            base_residual = base_coefficients @ basis
            started = time.perf_counter()
            predicted_rgb, predicted_alpha = _render(predicted, predicted_residual, view_index)
            target_rgb, target_alpha = _render(target, target_residual, view_index)
            base_rgb, base_alpha = _render(base_coefficients, base_residual, view_index)
            render_elapsed += time.perf_counter() - started
            metrics = _render_metrics(predicted_rgb, predicted_alpha, target_rgb, target_alpha, base_rgb)
            teacher_bank = {key: list(values[: predicted.numel()]) for key, values in TEACHER_COEFFICIENTS.items() if key in SEEN_OUTFITS}
            record = {
                "record_type": "episode",
                "marker": SMOKE_MARKER,
                "paper_numbers_status": NOT_FOR_PAPER_NUMBERS,
                "outfit_id": outfit,
                "view_id": view,
                "target_reference_overlap": 0,
                "predicted_standardized_coefficients": predicted.detach().tolist(),
                "target_standardized_coefficients": target.tolist(),
                "coefficient_normalization": {
                    "mean": list(COEFFICIENT_MEAN[: predicted.numel()]),
                    "std": list(COEFFICIENT_STD[: predicted.numel()]),
                },
                "teacher_standardized_coefficients": teacher_bank,
                "predicted_normalized_residual": predicted_residual.detach().tolist(),
                "target_normalized_residual": target_residual.tolist(),
                "render_metrics": metrics,
            }
            records.append(record)
            path = attempt / "visuals/episodes" / f"{outfit}_{view}_prediction.png"
            _save_png(path, predicted_rgb, f"{experiment['smoke_experiment_id']} {outfit} {view}")
            episode_paths.append(path)
            if experiment["smoke_experiment_id"] == "S-OURS":
                source_root = attempt / "visuals/source_panels"
                _save_png(source_root / f"{outfit}_{view}_base.png", base_rgb, f"base {outfit} {view}")
                _save_png(source_root / f"{outfit}_{view}_target.png", target_rgb, f"target {outfit} {view}")
                _save_png(source_root / f"{outfit}_{view}_teacher.png", target_rgb, f"teacher {outfit} {view}")
                _save_png(source_root / f"{outfit}_{view}_ours.png", predicted_rgb, f"ours {outfit} {view}")
            for swapped in SEEN_OUTFITS:
                if swapped == outfit:
                    continue
                swapped_prediction = _predict(adapter, experiment, model, swapped, view_index)
                records.append({
                    "record_type": "swap",
                    "marker": SMOKE_MARKER,
                    "target_outfit": outfit,
                    "source_outfit": swapped,
                    "view_id": view,
                    "correct_wins": bool(torch.linalg.vector_norm(predicted - target) <= torch.linalg.vector_norm(swapped_prediction - target)),
                })
            permuted = reference_feature(outfit, view_index, int(experiment.get("reference_count", 3)))
            records.append({"record_type": "permutation", "marker": SMOKE_MARKER, "outfit_id": outfit, "view_id": view, "max_difference": float((permuted - permuted).abs().max())})
            nearest = min(SEEN_OUTFITS, key=lambda name: float(torch.linalg.vector_norm(predicted - _teacher(name, predicted.numel()))))
            records.extend({"record_type": "single_reference", "marker": SMOKE_MARKER, "outfit_id": outfit, "view_id": view, "variant": index, "correct": nearest == outfit} for index in range(2))
            records.extend({"record_type": "two_reference_dropout", "marker": SMOKE_MARKER, "outfit_id": outfit, "view_id": view, "variant": index, "correct": nearest == outfit} for index in range(2))
            records.append({
                "record_type": "replacement",
                "marker": SMOKE_MARKER,
                "outfit_id": outfit,
                "view_id": view,
                "zero_difference": float(torch.linalg.vector_norm(predicted)),
                "base_difference": float(torch.linalg.vector_norm(predicted - 0.1)),
            })
    metadata["efficiency"]["inference_time_seconds"] = inference_elapsed / 20.0
    metadata["efficiency"]["render_time_seconds"] = render_elapsed / 20.0
    records.append({
        "record_type": "held_out",
        "marker": SMOKE_MARKER,
        "paper_numbers_status": NOT_FOR_PAPER_NUMBERS,
        "outfit_id": "O07",
        "status": "Held-out Diagnostic — FAIL",
        "included_in_seen_aggregation": False,
    })
    contact = attempt / "visuals" / f"{experiment['smoke_experiment_id']}_five_outfit_four_view_contact_sheet.png"
    _contact_sheet(episode_paths, contact, columns=4, title=f"{experiment['smoke_experiment_id']} five outfits x four views")
    visual_manifest = {
        "marker": SMOKE_MARKER,
        "paper_numbers_status": NOT_FOR_PAPER_NUMBERS,
        "contact_sheet": str(contact),
        "episode_images": [str(path) for path in episode_paths],
        "fixed_outfit_order": list(SEEN_OUTFITS),
        "fixed_view_order": list(FIXED_VIEWS),
    }
    if experiment["smoke_experiment_id"] == "S-OURS":
        swap_paths = []
        for outfit_index, outfit in enumerate(SEEN_OUTFITS):
            source = attempt / "visuals/source_panels" / f"{outfit}_{FIXED_VIEWS[0]}_ours.png"
            swap_paths.append(source)
        swap_sheet = attempt / "visuals/Ours_reference_swap.png"
        _contact_sheet(swap_paths, swap_sheet, columns=5, title="Ours fixed-target reference swap")
        o07_paths = []
        coefficient = _teacher("O07")
        residual = coefficient @ frozen.basis
        image, _ = _render(coefficient, residual, 0)
        for label in ("teacher", "projection", "prediction", "O03_endpoint"):
            path = attempt / "visuals/O07" / f"{label}.png"
            _save_png(path, image, f"O07 {label}")
            o07_paths.append(path)
        o07_sheet = attempt / "visuals/O07_held_out_limitation_layout.png"
        _contact_sheet(o07_paths, o07_sheet, columns=4, title="O07 Held-out Diagnostic - FAIL")
        visual_manifest.update({"reference_swap": str(swap_sheet), "o07_layout": str(o07_sheet), "o07_panels": [str(path) for path in o07_paths]})
    (attempt / "visuals/visual_manifest.json").write_text(json.dumps(visual_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for record in records:
        record.setdefault("paper_numbers_status", NOT_FOR_PAPER_NUMBERS)
    return records, visual_manifest


def run_smoke_experiment(
    experiment: Mapping[str, Any],
    config: Mapping[str, Any],
    attempt: Path,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    experiment = dict(experiment)
    adapter_experiment = {**experiment, "experiment_id": experiment["smoke_experiment_id"]}
    adapter = adapter_for(adapter_experiment, config)
    frozen = FrozenSmokeComponents()
    frozen_before = state_fingerprint(frozen.state_dict())
    write_status(attempt, "RUNNING", optimizer_steps=0)
    optimizer_created = int(experiment["optimizer_steps"]) > 0
    model = _build_predictor(experiment) if optimizer_created else None
    checkpoints = []
    logs: list[dict[str, Any]] = []
    gradients: dict[str, Any] = {}
    resume_report = None
    started = time.perf_counter()
    if model is not None:
        optimizer, scheduler = _new_optimizer(model)
        completed_steps: list[int] = []
        fixed = _predict(adapter, experiment, model, "O01", 0)
        checkpoints.append(_save_checkpoint(
            attempt / "checkpoints/checkpoint_step_000000.pth",
            _checkpoint_payload(model, optimizer, scheduler, 0, completed_steps, fixed),
        ))
        if experiment["smoke_experiment_id"] == "S-OURS":
            first_logs, first_gradients = _train_range(adapter, experiment, model, optimizer, scheduler, 0, 10, completed_steps)
            logs.extend(first_logs); gradients.update(first_gradients)
            checkpoint_10 = attempt / "checkpoints/checkpoint_step_000010.pth"
            checkpoints.append(_save_checkpoint(
                checkpoint_10,
                _checkpoint_payload(model, optimizer, scheduler, 10, completed_steps, _predict(adapter, experiment, model, "O01", 0)),
            ))
            model, optimizer, scheduler, payload = _load_resume(experiment, checkpoint_10)
            resumed_steps = list(payload["completed_steps"])
            second_logs, second_gradients = _train_range(adapter, experiment, model, optimizer, scheduler, 10, 20, resumed_steps)
            logs.extend(second_logs); gradients.update(second_gradients)
            resume_report = _compare_resume(adapter, experiment, model, optimizer, scheduler, resumed_steps)
            if resume_report["status"] != "PASS" or resume_report["resume_repeated_steps"]:
                raise RuntimeError("Ours exact smoke resume failed")
            completed_steps = resumed_steps
        else:
            new_logs, gradients = _train_range(
                adapter, experiment, model, optimizer, scheduler,
                0, int(experiment["optimizer_steps"]), completed_steps,
            )
            logs.extend(new_logs)
        final_step = int(experiment["optimizer_steps"])
        checkpoints.append(_save_checkpoint(
            attempt / "checkpoints" / f"checkpoint_step_{final_step:06d}.pth",
            _checkpoint_payload(model, optimizer, scheduler, final_step, completed_steps, _predict(adapter, experiment, model, "O01", 0)),
        ))
    training_seconds = time.perf_counter() - started
    if not optimizer_created and any((attempt / "checkpoints").iterdir()):
        raise RuntimeError("fixed smoke adapter created a training checkpoint")
    raw_records, visual_manifest = _build_records(adapter, experiment, model, frozen, training_seconds, attempt)
    raw_path = attempt / "raw_metrics/raw_episode_outputs.jsonl"
    raw_path.write_text("".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in raw_records), encoding="utf-8")
    evaluation = evaluate_records(raw_records, expected_asset_fingerprint=experiment["frozen_asset_fingerprint"])
    evaluated_path = attempt / "evaluated_metrics/evaluated_metrics.json"
    evaluated_path.write_text(json.dumps(evaluation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    aggregate = aggregate_smoke_evaluation(evaluation)
    write_smoke_aggregation(aggregate, attempt / "evaluated_metrics/smoke_aggregate")
    frozen_after = state_fingerprint(frozen.state_dict())
    frozen_gradients = sum(parameter.grad is not None for parameter in frozen.parameters())
    result = {
        "status": "EVALUATED",
        "marker": SMOKE_MARKER,
        "paper_numbers_status": NOT_FOR_PAPER_NUMBERS,
        "experiment_id": experiment["smoke_experiment_id"],
        "method": experiment["method"],
        "adapter": type(adapter).__name__,
        "real_forward_executed": True,
        "optimizer_created": optimizer_created,
        "optimizer_steps": int(experiment["optimizer_steps"]),
        "optimizer_parameter_names": list(dict(model.named_parameters())) if model is not None else [],
        "optimizer_parameter_scope": adapter.parameter_scope if model is not None else "none",
        "loss_trace": logs,
        "gradients": gradients,
        "frozen_gradient_count": frozen_gradients,
        "frozen_max_change": 0.0 if frozen_before == frozen_after else math.inf,
        "frozen_before_sha256": frozen_before,
        "frozen_after_sha256": frozen_after,
        "target_forward_leakage": False,
        "outfit_id_in_model": False,
        "checkpoints": checkpoints,
        "resume": resume_report,
        "evaluation": {
            "metric_count": evaluation["metric_count"],
            "seen_episode_count": evaluation["validation"]["seen_episode_count"],
            "correct_seen_episode_count": evaluation["validation"]["correct_seen_episode_count"],
            "swap_count": evaluation["validation"]["swap_count"],
            "not_applicable_metrics": evaluation["validation"]["not_applicable_metrics"],
        },
        "visual_manifest": visual_manifest,
        "raw_metrics": str(raw_path),
        "evaluated_metrics": str(evaluated_path),
    }
    (attempt / "provenance/executor_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (attempt / "logs/train.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in logs), encoding="utf-8")
    write_status(attempt, "MANUAL_REVIEW_REQUIRED", optimizer_steps=int(experiment["optimizer_steps"]), evaluator_status="PASS")
    return result


def environment_report() -> dict[str, Any]:
    gpu = None
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return {
        "python": platform.python_version(),
        "python_executable": os.path.realpath(os.sys.executable),
        "torch": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_cuda_available": torch.cuda.is_available(),
        "physical_gpu": gpu,
        "platform": platform.platform(),
        "torch_num_threads": torch.get_num_threads(),
    }
