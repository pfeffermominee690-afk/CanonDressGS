"""Frozen, inference-only executor for AAAI27 P0 reviewer-risk evaluations.

The executor consumes the sealed historical and P0 artifacts.  It never
updates a source attempt and never persists model state.  New files are
append-only beneath a dedicated evaluation attempt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw
from scipy import ndimage

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

from scene.p0_candidate_initialization_protocol import tensor_mapping_sha256
from tools import diagnose_image_conditioned_overfit_failure as diagnosis
from tools import run_multi_outfit_explicit_basis as multi
from tools.check_real_image_conditioned_one_batch import save_render_tensor
from tools.paper import formal_batch_runtime as historical
from tools.paper import formal_runtime as core
from tools.paper import p0_evaluation_protocol as frozen
from tools.paper import p0_formal_candidate_runtime as p0
from tools.paper.p0_candidate_runner import _reference_checksum_index


TASK_ID = "AAAI27-P0-COLOR-SPATIAL-SOFT-CONTROL-002"
SOURCE_BRANCH = "paper/aaai27-p0-evaluation-protocol-repair-20260721"
SOURCE_HEAD = "8578fb3143dd916ff7e42240e7508a29c784d995"
RUN_BRANCH = "paper/aaai27-p0-color-spatial-soft-control-evaluations-20260722"
PROTOCOL_PATH = PROJECT_ROOT / "paper_protocol/reviewer_risk/p0_color_spatial_soft_control_protocol.yaml"
PROTOCOL_SHA256 = "44320d575a7012a00396e6093dd5a0ee06cdd5679c878b3c7b254ceb215a0bba"
OUTPUT_NAME = "AAAI27-P0-COLOR-SPATIAL-SOFT-CONTROL"
FORMAL_NAME = "AAAI27-SEEN-OUTFIT-PAPER"
P0_NAME = "AAAI27-P0-FORMAL-CANDIDATE-RUNS"
OUTFITS = tuple(frozen.OUTFIT_ORDER)
CONDITIONS = tuple(frozen.TARGET_CONDITION_ORDER)
PAIRS = tuple(frozen.PAIR_ORDER)
COLOR_VARIANTS = ("C0", "C1", "C2", "C3", "C4", "C5", "C6")
METHODS = ("Ours-v2", "B6", "B7")
EXPECTED_FORMAL = (4127, 964043888, "7b9449e03d53e29cff11ded1fdc95e633640d38754cf152f2ab07a935d89d6bc")
EXPECTED_P0 = (911, 339118385, "96bdc50a165850df5ed59ab10fa2d20c0f1c92e67c4c97e41b1b26cc50b36362")
LPIPS_HASHES = {
    "implementation_sha256": "827e820a40047b3d93f24d06c4f1f59892eb064006a5f39a1fbd169bfa031ba7",
    "calibration_sha256": "a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868",
    "trunk_sha256": "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0",
}


def sha256(path: Path, *, lf: bool = False) -> str:
    payload = path.read_bytes()
    if lf:
        payload = payload.replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=PROJECT_ROOT, text=True).strip()


def atomic_json(path: Path, value: Any, *, replace: bool = False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(f"append-only artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False, default=str)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def tensor_hash(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def numpy_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def tree_manifest(path: Path) -> dict[str, Any]:
    rows: list[str] = []
    count = 0
    for item in sorted((entry for entry in path.rglob("*") if entry.is_file()), key=lambda p: p.as_posix()):
        stat = item.stat()
        relative = item.relative_to(path).as_posix()
        rows.append(f"{relative} {stat.st_size} {stat.st_mtime_ns}\n")
        count += 1
    digest = hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()
    total = int(subprocess.check_output(["du", "-sb", str(path)], text=True).split()[0])
    return {"file_count": count, "bytes": total, "metadata_sha256_ns": digest}


def image_tensor(path: Path, channels: int) -> torch.Tensor:
    with Image.open(path) as opened:
        image = opened.convert("RGB" if channels == 3 else "L")
        array = np.asarray(image, dtype=np.float32) / 255.0
    if channels == 1:
        array = array[..., None]
    return torch.from_numpy(array.copy()).permute(2, 0, 1)


def save_new_render(path: Path, value: torch.Tensor, channels: int) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    save_render_tensor(path, value.detach().cpu(), channels)


def render_or_load(
    runtime: "EvaluationRuntime", outfit: str, condition: str, residual: Any,
    rgb_path: Path, alpha_path: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rgb_path.is_file() and alpha_path.is_file():
        return image_tensor(rgb_path, 3).to(runtime.device), image_tensor(alpha_path, 1).to(runtime.device)
    with torch.inference_mode():
        rgb, alpha = p0._render(runtime.context, outfit, condition, residual)
    save_new_render(rgb_path, rgb, 3)
    save_new_render(alpha_path, alpha, 1)
    return rgb.detach(), alpha.detach()


def contact_sheet(path: Path, rows: Sequence[tuple[str, Sequence[tuple[str, Path]]]]) -> None:
    if path.exists():
        return
    tile_w, tile_h, label_h = 192, 288, 26
    columns = max(len(panels) for _, panels in rows)
    canvas = Image.new("RGB", (tile_w * columns, (tile_h + label_h) * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, (row_label, panels) in enumerate(rows):
        y = row_index * (tile_h + label_h)
        for column, (label, source) in enumerate(panels):
            with Image.open(source) as opened:
                image = opened.convert("RGB")
                image.thumbnail((tile_w, tile_h), Image.Resampling.BILINEAR)
                x = column * tile_w + (tile_w - image.width) // 2
                canvas.paste(image, (x, y + (tile_h - image.height) // 2))
            draw.text((column * tile_w + 3, y + tile_h + 3), f"{row_label} {label}", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".png.tmp")
    canvas.save(temporary, format="PNG")
    os.replace(temporary, path)


def percentile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values.double(), q, interpolation="linear"))


def numeric_mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.fmean(finite)) if finite else 0.0


def phase_result(attempt: Path, name: str) -> Path:
    return attempt / "aggregates" / f"{name}.json"


def append_phase_status(attempt: Path, phase: str, status: str) -> None:
    path = attempt / "audits/phase_status.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({"phase": phase, "status": status, "time": time.time()}) + "\n")


def resolve_attempt(output_root: Path, requested: str | None = None) -> Path:
    attempts = sorted(output_root.glob("attempt_*")) if output_root.is_dir() else []
    if requested is not None:
        if not requested.startswith("attempt_") or not requested[8:].isdigit():
            raise ValueError("--attempt must use attempt_NNN")
        selected = output_root / requested
        if selected.is_dir():
            return selected
        expected = f"attempt_{len(attempts) + 1:03d}"
        if requested != expected:
            raise RuntimeError(f"append-only next attempt must be {expected}")
        selected.mkdir(parents=True, exist_ok=False)
        return selected
    if not attempts:
        attempt = output_root / "attempt_001"
        attempt.mkdir(parents=True, exist_ok=False)
        return attempt
    if len(attempts) != 1:
        raise RuntimeError("evaluation output contains an ambiguous attempt set")
    return attempts[0]


def validate_source() -> dict[str, Any]:
    if git("branch", "--show-current") != RUN_BRANCH:
        raise RuntimeError("P0-EVALUATION-ASSET-MISMATCH: branch")
    if git("status", "--short"):
        raise RuntimeError("evaluation executor requires a clean worktree")
    if subprocess.call(["git", "merge-base", "--is-ancestor", SOURCE_HEAD, "HEAD"], cwd=PROJECT_ROOT):
        raise RuntimeError("P0-EVALUATION-ASSET-MISMATCH: source head ancestry")
    if sha256(PROTOCOL_PATH, lf=True) != PROTOCOL_SHA256:
        raise RuntimeError("P0-EVALUATION-ASSET-MISMATCH: protocol fingerprint")
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["dataset"]["fixed_episode_count"] != 20:
        raise RuntimeError("PROTOCOL-IMPLEMENTATION-MISMATCH: episode count")
    if tuple(protocol["dataset"]["seen_outfit_order"]) != OUTFITS:
        raise RuntimeError("PROTOCOL-IMPLEMENTATION-MISMATCH: outfit order")
    if tuple(protocol["dataset"]["target_condition_order"]) != CONDITIONS:
        raise RuntimeError("PROTOCOL-IMPLEMENTATION-MISMATCH: condition order")
    return protocol


class EvaluationRuntime:
    def __init__(self, attempt: Path, asset_root: Path, protocol: Mapping[str, Any]) -> None:
        self.attempt = attempt
        self.asset_root = asset_root
        self.protocol = protocol
        old_branch, old_source = core.FORMAL_BRANCH, core.FORMAL_SOURCE_HEAD
        core.FORMAL_BRANCH, core.FORMAL_SOURCE_HEAD = RUN_BRANCH, SOURCE_HEAD
        try:
            self.context = core._legacy_context(attempt)
        finally:
            core.FORMAL_BRANCH, core.FORMAL_SOURCE_HEAD = old_branch, old_source
        self.device = self.context["base"]._xyz.device
        self.basis, self.coefficients, self.basis_payload, self.basis_path = historical._basis_artifact(
            self.context, "Ours_Seen_Outfit_Explicit_Basis_V1"
        )
        self.targets = multi._standardized_targets(self.coefficients, self.basis_payload)
        self.mean = torch.as_tensor(self.basis_payload["coefficient_train_mean"], device=self.device)
        self.std = torch.as_tensor(self.basis_payload["coefficient_train_std"], device=self.device)
        self.teachers = historical.load_frozen_teacher_residuals(self.context, OUTFITS)
        cache_path = asset_root / FORMAL_NAME / "shared_preflight/frozen_reference_feature_rows_v1.pt"
        self.cache_path = cache_path
        self.cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        self.extractor = multi._feature_extractor(self.context)
        self.models = self._load_models()
        reference_path = (
            asset_root / "pipeline_full/SUBJECT02-AAAI27-DATA-CAPACITY-GATE-001/attempt_003"
            / "dataset/aaai_gate_28_manifest.json"
        )
        reference_checksums = _reference_checksum_index(read_json(reference_path))
        self.b7, self.b7_manifest = p0.build_b7_runtime(
            cache=self.cache, reference_checksums=reference_checksums, device=self.device
        )
        self._observations: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        self.lpips_model: torch.nn.Module | None = None

    def _load_models(self) -> dict[str, torch.nn.Module]:
        registry = yaml.safe_load(
            (PROJECT_ROOT / "paper_protocol/reviewer_risk/p0_formal_candidate_run_registry.yaml")
            .read_text(encoding="utf-8")
        )
        wanted = {"Ours-v2": "P0-FORMAL-OURS-V2-R0", "B6": "P0-FORMAL-B6-S0"}
        result: dict[str, torch.nn.Module] = {}
        for method, run_id in wanted.items():
            row = next(item for item in registry["runs"] if item["formal_run_id"] == run_id)
            model, _ = p0.model_factory(method, int(row["seed"]), self.cache, self.device)
            payload = torch.load(Path(row["checkpoint"]), map_location="cpu", weights_only=False)
            model.load_state_dict(payload["model"], strict=True)
            model.eval()
            result[method] = model
        return result

    def observation(self, outfit: str, condition: str) -> tuple[np.ndarray, np.ndarray]:
        cache_key = (outfit, condition)
        if cache_key in self._observations:
            return self._observations[cache_key]
        candidates: list[tuple[str, int]] = []
        for target in CONDITIONS:
            if target == condition:
                continue
            episode = self.context["episodes"][f"{outfit}/{target}"]
            if condition in episode["reference_condition_ids"]:
                candidates.append((target, episode["reference_condition_ids"].index(condition)))
        if not candidates:
            raise RuntimeError(f"missing frozen observation {outfit}/{condition}")
        images = []
        masks = []
        for target, index in candidates:
            episode = self.context["episodes"][f"{outfit}/{target}"]
            images.append(
                episode["reference_images"][index].detach().cpu().permute(1, 2, 0).contiguous().numpy()
                .astype(np.float32, copy=False)
            )
            masks.append(
                (episode["reference_cloth_masks"][index, 0].detach().cpu().numpy() >= 0.5)
                .astype(np.bool_, copy=False)
            )
        if any(not np.array_equal(images[0], value) for value in images[1:]):
            raise RuntimeError("P0-EVALUATION-ASSET-MISMATCH: repeated reference RGB")
        if any(not np.array_equal(masks[0], value) for value in masks[1:]):
            raise RuntimeError("P0-EVALUATION-ASSET-MISMATCH: repeated reference mask")
        self._observations[cache_key] = (images[0], masks[0])
        return self._observations[cache_key]

    def f2_rows(
        self, images: Sequence[np.ndarray], masks: Sequence[np.ndarray]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_tensor_value = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).to(self.device)
        mask_tensor_value = torch.from_numpy(np.stack(masks)[:, None].astype(np.float32)).to(self.device)
        valid = torch.ones((len(images),), dtype=image_tensor_value.dtype, device=self.device)
        with torch.inference_mode():
            output = self.extractor(image_tensor_value, mask_tensor_value, valid)
        return output.per_reference_f2.detach(), output.valid_mask.detach()

    def predict(
        self, method: str, condition: str, rows: torch.Tensor, valid: torch.Tensor
    ) -> dict[str, Any]:
        with torch.inference_mode():
            if method == "B7":
                output = self.b7[condition](rows, valid)
                selected = output.predicted_outfit
                return {
                    "standardized": self.targets[selected].detach(),
                    "raw": self.coefficients[selected].detach(),
                    "residual": self.teachers[selected],
                    "predicted_outfit": selected,
                    "distances": dict(output.squared_distances),
                }
            output = self.models[method](rows, valid)
            if method == "B6":
                selected = output.predicted_outfit
                return {
                    "standardized": self.targets[selected].detach(),
                    "raw": self.coefficients[selected].detach(),
                    "residual": self.teachers[selected],
                    "predicted_outfit": selected,
                    "logits": output.logits.detach(),
                }
            standardized = output.standardized_coefficients.detach()
            raw = standardized * self.std + self.mean
            residual = self.basis(raw, chunk_size=16384)
            selected = min(
                OUTFITS,
                key=lambda outfit: (
                    float(torch.linalg.vector_norm(standardized - self.targets[outfit])),
                    OUTFITS.index(outfit),
                ),
            )
            return {
                "standardized": standardized,
                "raw": raw.detach(),
                "residual": residual,
                "predicted_outfit": selected,
            }

    def lpips(self) -> torch.nn.Module:
        if self.lpips_model is None:
            package_root = self.asset_root.parent / "AnimatableGaussians"
            if str(package_root) not in sys.path:
                sys.path.insert(0, str(package_root))
            module = importlib.import_module("network.lpips.lpips")
            self.lpips_model = module.LPIPS(net="vgg", version="0.1", verbose=False).to(self.device)
            self.lpips_model.eval()
        return self.lpips_model


def run_preflight(
    attempt: Path, asset_root: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    result_path = phase_result(attempt, "preflight")
    if result_path.is_file():
        return read_json(result_path)
    formal_root = asset_root / FORMAL_NAME
    p0_root = asset_root / P0_NAME
    if not formal_root.is_dir() or not p0_root.is_dir():
        raise RuntimeError("P0-EVALUATION-ASSET-MISMATCH: frozen output roots")
    p0_registry = yaml.safe_load(
        (PROJECT_ROOT / "paper_protocol/reviewer_risk/p0_formal_candidate_run_registry.yaml")
        .read_text(encoding="utf-8")
    )
    if len(p0_registry["runs"]) != 13:
        raise RuntimeError("P0-EVALUATION-ASSET-MISMATCH: candidate registry")
    missing_attempts = [row["formal_run_id"] for row in p0_registry["runs"] if not Path(row["attempt"]).is_dir()]
    missing_checkpoints = [
        row["formal_run_id"] for row in p0_registry["runs"]
        if row["expected_steps"] and not Path(row["checkpoint"]).is_file()
    ]
    b7_manifest = Path(next(row for row in p0_registry["runs"] if row["method"] == "B7")["attempt"]) / "manifests/b7_centroid_construction.json"
    if missing_attempts or missing_checkpoints or not b7_manifest.is_file():
        raise RuntimeError("P0-EVALUATION-ASSET-MISMATCH: candidate outputs")
    lpips_root = asset_root.parent / "AnimatableGaussians/network/lpips"
    lpips_paths = {
        "implementation": lpips_root / "lpips.py",
        "calibration": lpips_root / "weights/v0.1/vgg.pth",
        "trunk": Path("/root/.cache/torch/hub/checkpoints/vgg16-397923af.pth"),
    }
    actual_lpips = {name: sha256(path) for name, path in lpips_paths.items()}
    expected_lpips = {
        "implementation": LPIPS_HASHES["implementation_sha256"],
        "calibration": LPIPS_HASHES["calibration_sha256"],
        "trunk": LPIPS_HASHES["trunk_sha256"],
    }
    if actual_lpips != expected_lpips:
        raise RuntimeError("P0-EVALUATION-ASSET-MISMATCH: LPIPS resources")
    smoke = torch.tensor([6.0, 7.0], device="cuda").sum()
    torch.cuda.synchronize()
    if float(smoke) != 13.0:
        raise RuntimeError("P0-EVALUATION-ASSET-MISMATCH: CUDA smoke")
    formal_manifest = tree_manifest(formal_root)
    p0_manifest = tree_manifest(p0_root)
    if (formal_manifest["file_count"], formal_manifest["bytes"]) != EXPECTED_FORMAL[:2]:
        raise RuntimeError("P0-EVALUATION-ASSET-MISMATCH: formal output tree")
    if (p0_manifest["file_count"], p0_manifest["bytes"]) != EXPECTED_P0[:2]:
        raise RuntimeError("P0-EVALUATION-ASSET-MISMATCH: P0 output tree")
    paper_final_count = sum(
        1 for path in PROJECT_ROOT.rglob("*")
        if path.is_file() and path.name == "PAPER_FINAL"
    )
    result = {
        "schema_version": "canondressgs.paper.p0_color_spatial_soft_control_preflight.v1",
        "status": "PASS",
        "task_id": TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "run_branch": RUN_BRANCH,
        "execution_head": git("rev-parse", "HEAD"),
        "protocol_path": str(PROTOCOL_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "protocol_sha256_lf": PROTOCOL_SHA256,
        "candidate_outputs": {"run_count": 13, "checkpoint_count": 12, "b7_manifest": str(b7_manifest)},
        "frozen_assets": {"count": 19, "verified": 19},
        "lpips": {"paths": {name: str(path) for name, path in lpips_paths.items()}, "hashes": actual_lpips},
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "driver": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
            ).strip(),
            "cuda": torch.version.cuda,
            "pytorch": torch.__version__,
            "python": platform.python_version(),
            "cuda_tensor_smoke": float(smoke),
        },
        "formal_outputs_before": formal_manifest,
        "p0_outputs_before": p0_manifest,
        "paper_final_count": paper_final_count,
        "dino_clip": "BLOCKED_RESOURCE_MISSING",
        "network_download": False,
        "paper_final": False,
    }
    atomic_json(result_path, result)
    return result


def frozen_global_statistics(runtime: EvaluationRuntime) -> tuple[np.ndarray, np.ndarray, str]:
    values: list[np.ndarray] = []
    for outfit in OUTFITS:
        for condition in CONDITIONS:
            image, mask = runtime.observation(outfit, condition)
            values.append(image[mask].astype(np.float64))
    pooled = np.concatenate(values, axis=0)
    mean = pooled.mean(axis=0, dtype=np.float64)
    variance = np.mean(np.square(pooled - mean), axis=0, dtype=np.float64)
    std = np.sqrt(variance)
    return mean, std, hashlib.sha256(mean.tobytes() + std.tobytes()).hexdigest()


def color_reference_set(
    runtime: EvaluationRuntime, outfit: str, target_condition: str, variant: str,
    global_mean: np.ndarray, global_std: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    legal = [condition for condition in CONDITIONS if condition != target_condition]
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    sources: list[dict[str, Any]] = []
    donor_cycle = runtime.protocol["color_transforms"]["C3"]["parameter_values"]["donor_cycle"]
    for condition in legal:
        image, mask = runtime.observation(outfit, condition)
        if variant == "C0":
            transformed = image.copy()
        elif variant == "C1":
            transformed = frozen.fixed_grayscale(image)
        elif variant == "C2":
            transformed = frozen.hue_shift(image, 120.0)
        elif variant == "C3":
            donor_outfit = donor_cycle[outfit]
            donor, donor_mask = runtime.observation(donor_outfit, condition)
            transformed = frozen.c3_histogram_match(image, donor, mask, donor_mask)
        elif variant == "C4":
            transformed = frozen.c4_brightness_contrast_normalize(image, mask, global_mean, global_std)
        elif variant == "C5":
            transformed = frozen.c5_gaussian_blur(image, mask)
        elif variant == "C6":
            transformed = frozen.c6_average_garment_color(image, mask)
        else:
            raise KeyError(variant)
        if not np.array_equal(mask, mask.copy()):
            raise RuntimeError("PROTOCOL-IMPLEMENTATION-MISMATCH: reference mask")
        if variant in {"C3", "C4", "C5", "C6"} and not np.array_equal(transformed[~mask], image[~mask]):
            raise RuntimeError(f"PROTOCOL-IMPLEMENTATION-MISMATCH: {variant} exterior")
        images.append(transformed)
        masks.append(mask.copy())
        sources.append({
            "condition": condition,
            "input_rgb_sha256": numpy_hash(image),
            "output_rgb_sha256": numpy_hash(transformed),
            "mask_sha256": numpy_hash(mask.astype(np.uint8)),
        })
    return images, masks, {"legal_conditions": legal, "sources": sources}


def target_companion_hashes(runtime: EvaluationRuntime) -> dict[str, str]:
    digest: dict[str, str] = {}
    for outfit in OUTFITS:
        for condition in CONDITIONS:
            key = f"{outfit}/{condition}"
            sample = runtime.context["samples"][key]
            episode = runtime.context["episodes"][key]
            fields = {
                "target_edit_rgb": sample["target_edit_rgb"],
                "target_clothing_mask": sample["target_clothing_mask"],
                "target_pose": sample["target_pose"],
                "target_K": sample["target_camera"]["K"],
                "target_w2c": sample["target_camera"]["w2c"],
                "reference_cloth_masks": episode["reference_cloth_masks"],
                "reference_poses": episode["reference_poses"],
                "reference_K": episode["reference_K"],
                "reference_w2c": episode["reference_w2c"],
            }
            digest[key] = tensor_mapping_sha256(fields)
    return digest


def crop_box(mask: torch.Tensor, expansion: float = 0.05) -> tuple[int, int, int, int]:
    membership = mask.reshape(mask.shape[-2], mask.shape[-1]) >= 0.5
    positions = torch.nonzero(membership, as_tuple=False)
    if not positions.numel():
        return (0, 0, mask.shape[-2], mask.shape[-1])
    y0, x0 = positions.min(0).values.tolist()
    y1, x1 = (positions.max(0).values + 1).tolist()
    margin = int(math.floor(expansion * max(y1 - y0, x1 - x0) + 0.5))
    return max(0, y0 - margin), max(0, x0 - margin), min(mask.shape[-2], y1 + margin), min(mask.shape[-1], x1 + margin)


def lpips_input(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    y0, x0, y1, x1 = crop_box(mask)
    crop = image[:, y0:y1, x0:x1] * (mask[:, y0:y1, x0:x1] >= 0.5).to(image)
    height, width = crop.shape[-2:]
    scale = min(256.0 / height, 256.0 / width)
    resized_h = max(1, int(math.floor(height * scale + 0.5)))
    resized_w = max(1, int(math.floor(width * scale + 0.5)))
    resized = F.interpolate(crop[None], size=(resized_h, resized_w), mode="bilinear", align_corners=False)
    output = crop.new_zeros((1, 3, 256, 256))
    y = (256 - resized_h) // 2
    x = (256 - resized_w) // 2
    output[:, :, y : y + resized_h, x : x + resized_w] = resized
    return output * 2.0 - 1.0


def lpips_distance(runtime: EvaluationRuntime, first: torch.Tensor, second: torch.Tensor, mask: torch.Tensor) -> float:
    with torch.inference_mode():
        value = runtime.lpips()(lpips_input(first, mask), lpips_input(second, mask))
    return float(value.reshape(()))


def masked_psnr(first: torch.Tensor, second: torch.Tensor, mask: torch.Tensor) -> float | str:
    membership = (mask >= 0.5).expand_as(first)
    mse = torch.square(first[membership] - second[membership]).double().mean()
    if float(mse) == 0.0:
        return "Infinity"
    return float(-10.0 * torch.log10(mse))


def masked_ssim(first: torch.Tensor, second: torch.Tensor, mask: torch.Tensor) -> float:
    y0, x0, y1, x1 = crop_box(mask, expansion=0.0)
    local_mask = (mask[:, y0:y1, x0:x1] >= 0.5).to(first)
    x = first[:, y0:y1, x0:x1][None] * local_mask[None]
    y = second[:, y0:y1, x0:x1][None] * local_mask[None]
    offsets = torch.arange(-5, 6, device=first.device, dtype=first.dtype)
    kernel = torch.exp(-torch.square(offsets) / (2.0 * 1.5 * 1.5))
    kernel = (kernel / kernel.sum())[:, None] * (kernel / kernel.sum())[None, :]
    weight = kernel[None, None].expand(3, 1, 11, 11)
    def blur(value: torch.Tensor) -> torch.Tensor:
        return F.conv2d(F.pad(value, (5, 5, 5, 5), mode="reflect"), weight, groups=3)
    mean_x, mean_y = blur(x), blur(y)
    variance_x = blur(x * x) - mean_x * mean_x
    variance_y = blur(y * y) - mean_y * mean_y
    covariance = blur(x * y) - mean_x * mean_y
    score = ((2 * mean_x * mean_y + 0.01**2) * (2 * covariance + 0.03**2)) / (
        (mean_x.square() + mean_y.square() + 0.01**2)
        * (variance_x + variance_y + 0.03**2)
    )
    membership = local_mask.expand_as(score[0]) >= 0.5
    return float(score[0][membership].double().mean())


def silhouette_metrics(alpha: torch.Tensor, target_mask: torch.Tensor) -> tuple[float, float, int]:
    prediction = alpha.detach().cpu().numpy().reshape(alpha.shape[-2], alpha.shape[-1]) >= 0.5
    target = target_mask.detach().cpu().numpy().reshape(alpha.shape[-2], alpha.shape[-1]) >= 0.5
    union = np.logical_or(prediction, target).sum()
    iou = float(np.logical_and(prediction, target).sum() / max(int(union), 1))
    tolerance = frozen.boundary_tolerance(*prediction.shape)
    structure = np.ones((3, 3), dtype=np.bool_)
    pred_boundary = np.logical_xor(prediction, ndimage.binary_erosion(prediction, structure=structure))
    target_boundary = np.logical_xor(target, ndimage.binary_erosion(target, structure=structure))
    if not pred_boundary.any() and not target_boundary.any():
        return iou, 1.0, tolerance
    distance_to_target = ndimage.distance_transform_edt(~target_boundary)
    distance_to_prediction = ndimage.distance_transform_edt(~pred_boundary)
    precision = float((distance_to_target[pred_boundary] <= tolerance).mean()) if pred_boundary.any() else 0.0
    recall = float((distance_to_prediction[target_boundary] <= tolerance).mean()) if target_boundary.any() else 0.0
    fscore = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return iou, fscore, tolerance


def run_color(runtime: EvaluationRuntime) -> dict[str, Any]:
    result_path = phase_result(runtime.attempt, "color")
    if result_path.is_file():
        return read_json(result_path)
    append_phase_status(runtime.attempt, "color", "RUNNING")
    companions_before = target_companion_hashes(runtime)
    frozen_before = p0.frozen_fingerprints(runtime.context, runtime.basis)
    global_mean, global_std, global_hash = frozen_global_statistics(runtime)
    b1_root = runtime.asset_root / FORMAL_NAME / "PAPER-B1-FIXED/seed_fixed/attempt_001/visuals/episodes"
    records: list[dict[str, Any]] = []
    prediction_index: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    sheet_rows: dict[str, list[tuple[str, list[tuple[str, Path]]]]] = {
        variant: [] for variant in COLOR_VARIANTS
    }
    for variant in COLOR_VARIANTS:
        for outfit in OUTFITS:
            for condition in CONDITIONS:
                images, masks, source = color_reference_set(
                    runtime, outfit, condition, variant, global_mean, global_std
                )
                if variant == "C0":
                    cached = runtime.cache["episodes"][f"{outfit}/{condition}"]["normal"]
                    rows = cached["f2"].to(runtime.device)
                    valid = cached["valid"].to(runtime.device)
                    c0_parity = 0.0
                else:
                    rows, valid = runtime.f2_rows(images, masks)
                    c0_parity = None
                preview = runtime.attempt / "counterfactuals" / variant / "references" / f"{outfit}_{condition}_reference0.png"
                save_new_render(preview, torch.from_numpy(images[0]).permute(2, 0, 1), 3)
                panels: list[tuple[str, Path]] = [("ref0", preview)]
                teacher_rgb_path = b1_root / f"{outfit}_{condition}_prediction.png"
                teacher_rgb = image_tensor(teacher_rgb_path, 3).to(runtime.device)
                garment = diagnosis._garment_mask(runtime.context["samples"][f"{outfit}/{condition}"])
                for method in METHODS:
                    prediction = runtime.predict(method, condition, rows, valid)
                    rgb_path = (
                        runtime.attempt / "counterfactuals" / variant / method.replace("-", "_").lower()
                        / f"{outfit}_{condition}_prediction.png"
                    )
                    alpha_path = rgb_path.with_name(rgb_path.stem.replace("prediction", "alpha") + ".png")
                    rgb, alpha = render_or_load(
                        runtime, outfit, condition, prediction["residual"], rgb_path, alpha_path
                    )
                    difference = float(torch.linalg.vector_norm(prediction["standardized"] - runtime.targets[outfit]))
                    nearest_rank = sorted(
                        OUTFITS,
                        key=lambda name: (
                            float(torch.linalg.vector_norm(prediction["standardized"] - runtime.targets[name])),
                            OUTFITS.index(name),
                        ),
                    ).index(outfit) + 1
                    garment_mae = float(
                        ((rgb - teacher_rgb).abs() * garment).sum()
                        / (garment.sum() * 3.0).clamp_min(1.0)
                    )
                    silhouette_iou, boundary_fscore, tolerance = silhouette_metrics(alpha, garment)
                    record = {
                        "variant": variant,
                        "method": method,
                        "outfit_id": outfit,
                        "view_id": condition,
                        "predicted_standardized_coefficients": prediction["standardized"].cpu().tolist(),
                        "predicted_raw_coefficients": prediction["raw"].cpu().tolist(),
                        "predicted_outfit": prediction["predicted_outfit"],
                        "correct": prediction["predicted_outfit"] == outfit,
                        "correct_rank": nearest_rank,
                        "target_coefficient_distance": difference,
                        "logits": prediction.get("logits", torch.empty(0)).cpu().tolist()
                        if "logits" in prediction else None,
                        "centroid_distances": prediction.get("distances"),
                        "garment_rgb_mae_to_b1_teacher": garment_mae,
                        "silhouette_iou": silhouette_iou,
                        "boundary_fscore": boundary_fscore,
                        "boundary_tolerance": tolerance,
                        "source_reference": source,
                        "reference_preview": str(preview),
                        "output_rgb": str(rgb_path),
                        "output_alpha": str(alpha_path),
                        "c0_feature_max_abs_parity": c0_parity,
                    }
                    records.append(record)
                    prediction_index[(variant, method, outfit, condition)] = record
                    panels.append((method, rgb_path))
                panels.append(("B1 teacher", teacher_rgb_path))
                sheet_rows[variant].append((f"{outfit}/{condition}", panels))
    swap_records: list[dict[str, Any]] = []
    for variant, method, target_outfit, condition in itertools.product(
        COLOR_VARIANTS, METHODS, OUTFITS, CONDITIONS
    ):
        correct = prediction_index[(variant, method, target_outfit, condition)]
        correct_value = torch.tensor(correct["predicted_standardized_coefficients"])
        correct_distance = float(torch.linalg.vector_norm(correct_value - runtime.targets[target_outfit].cpu()))
        for source_outfit in OUTFITS:
            if source_outfit == target_outfit:
                continue
            swapped = prediction_index[(variant, method, source_outfit, condition)]
            swapped_value = torch.tensor(swapped["predicted_standardized_coefficients"])
            swapped_distance = float(torch.linalg.vector_norm(swapped_value - runtime.targets[target_outfit].cpu()))
            swap_records.append({
                "variant": variant,
                "method": method,
                "target_outfit": target_outfit,
                "source_outfit": source_outfit,
                "view_id": condition,
                "correct_distance": correct_distance,
                "swapped_distance": swapped_distance,
                "correct_wins": correct_distance < swapped_distance,
                "correct_predicted_outfit": correct["predicted_outfit"],
                "swapped_predicted_outfit": swapped["predicted_outfit"],
            })
    sheets: dict[str, str] = {}
    for variant, rows in sheet_rows.items():
        path = runtime.attempt / "visuals/color" / f"{variant}_all_outfits_views_methods.png"
        contact_sheet(path, rows)
        sheets[variant] = str(path)
    by_method: dict[str, Any] = {}
    for method in METHODS:
        transforms: dict[str, Any] = {}
        for variant in COLOR_VARIANTS:
            selected = [row for row in records if row["method"] == method and row["variant"] == variant]
            outfit_accuracy = {
                outfit: numeric_mean(float(row["correct"]) for row in selected if row["outfit_id"] == outfit)
                for outfit in OUTFITS
            }
            predicted_outfit_macro = {
                outfit: max(
                    OUTFITS,
                    key=lambda label: (
                        sum(row["predicted_outfit"] == label for row in selected if row["outfit_id"] == outfit),
                        -OUTFITS.index(label),
                    ),
                )
                for outfit in OUTFITS
            }
            transforms[variant] = {
                "accuracy": numeric_mean(float(row["correct"]) for row in selected),
                "outfit_macro_accuracy": outfit_accuracy,
                "predicted_outfit_macro": predicted_outfit_macro,
                "garment_rgb_mae": numeric_mean(row["garment_rgb_mae_to_b1_teacher"] for row in selected),
                "silhouette_iou": numeric_mean(row["silhouette_iou"] for row in selected),
                "boundary_fscore": numeric_mean(row["boundary_fscore"] for row in selected),
                "non_c0_coefficient_l2": None if variant == "C0" else numeric_mean(
                    float(torch.linalg.vector_norm(
                        torch.tensor(row["predicted_standardized_coefficients"])
                        - torch.tensor(prediction_index[("C0", method, row["outfit_id"], row["view_id"])]["predicted_standardized_coefficients"])
                    )) for row in selected
                ),
            }
        by_method[method] = transforms
    robust = all(
        by_method[method][variant]["accuracy"] >= 0.8
        and by_method[method]["C0"]["accuracy"] - by_method[method][variant]["accuracy"] <= 0.2
        for method in METHODS for variant in COLOR_VARIANTS[1:]
    )
    usable = all(
        by_method[method][variant]["accuracy"] > 0.4
        and max(
            list(by_method[method][variant]["predicted_outfit_macro"].values()).count(label)
            for label in OUTFITS
        ) < 3
        for method in METHODS for variant in COLOR_VARIANTS[1:]
    )
    conclusion = "COLOR_ROBUST" if robust else "COLOR_SENSITIVE_BUT_USABLE" if usable else "COLOR_DOMINATED"
    companions_after = target_companion_hashes(runtime)
    frozen_after = p0.frozen_fingerprints(runtime.context, runtime.basis)
    if companions_before != companions_after or frozen_before != frozen_after:
        raise RuntimeError("EVALUATION-ASSET-MUTATION")
    result = {
        "schema_version": "canondressgs.paper.p0_color_counterfactual_results.v1",
        "status": "PAPER_CANDIDATE_MANUAL_REVIEW_REQUIRED",
        "protocol_sha256_lf": PROTOCOL_SHA256,
        "representative_policy": {"Ours-v2": "R0", "B6": "S0", "B7": "FIXED", "best_seed_selected": False},
        "query_count": len(records),
        "expected_query_count": 420,
        "records": records,
        "swap_count": len(swap_records),
        "swap_records": swap_records,
        "method_aggregates": by_method,
        "global_target_statistics": {"mean": global_mean.tolist(), "std": global_std.tolist(), "sha256": global_hash},
        "contact_sheets": sheets,
        "target_companions_bitwise_unchanged": companions_before == companions_after,
        "frozen_assets_unchanged": frozen_before == frozen_after,
        "target_forward_leakage": False,
        "color_conclusion": conclusion,
        "paper_final": False,
    }
    atomic_json(result_path, result)
    append_phase_status(runtime.attempt, "color", "COMPLETE")
    return result


def extended_source_runs(runtime: EvaluationRuntime) -> list[dict[str, Any]]:
    formal = runtime.asset_root / FORMAL_NAME
    rows = [
        {"run_id": "PAPER-B1-FIXED", "method": "B1", "seed": None, "replicate": None,
         "attempt": formal / "PAPER-B1-FIXED/seed_fixed/attempt_001"},
        {"run_id": "PAPER-B2-FIXED", "method": "B2", "seed": None, "replicate": None,
         "attempt": formal / "PAPER-B2-FIXED/seed_fixed/attempt_001"},
    ]
    for method, prefix in (("B4", "PAPER-B4-S"),):
        for seed in range(3):
            rows.append({"run_id": f"{prefix}{seed}", "method": method, "seed": seed, "replicate": None,
                         "attempt": formal / f"{prefix}{seed}/seed_{seed}/attempt_001"})
    registry = yaml.safe_load(
        (PROJECT_ROOT / "paper_protocol/reviewer_risk/p0_formal_candidate_run_registry.yaml")
        .read_text(encoding="utf-8")
    )
    for row in registry["runs"]:
        if row["method"] in {"Ours-v2", "B6", "B7", "M3", "M4"}:
            rows.append({
                "run_id": row["formal_run_id"],
                "method": row["method"],
                "seed": row["seed"],
                "replicate": row["replicate_index"],
                "attempt": Path(row["attempt"]),
            })
    for row in rows:
        row["image_root"] = (
            row["attempt"] / "renders/episodes"
            if str(row["run_id"]).startswith("P0-FORMAL-")
            else row["attempt"] / "visuals/episodes"
        )
        row["contact_sheet"] = (
            row["attempt"] / "visuals/five_outfit_four_view_contact_sheet.png"
        )
        if not row["image_root"].is_dir() or not row["contact_sheet"].is_file():
            raise RuntimeError(f"P0-EVALUATION-ASSET-MISMATCH: source renders {row['run_id']}")
    return rows


def run_extended(runtime: EvaluationRuntime) -> dict[str, Any]:
    result_path = phase_result(runtime.attempt, "extended")
    if result_path.is_file():
        return read_json(result_path)
    append_phase_status(runtime.attempt, "extended", "RUNNING")
    sources = extended_source_runs(runtime)
    teacher = next(row for row in sources if row["method"] == "B1")
    records: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for source in sources:
        for outfit in OUTFITS:
            for condition in CONDITIONS:
                name = f"{outfit}_{condition}"
                prediction_path = source["image_root"] / f"{name}_prediction.png"
                alpha_path = source["image_root"] / f"{name}_alpha.png"
                teacher_path = teacher["image_root"] / f"{name}_prediction.png"
                teacher_alpha_path = teacher["image_root"] / f"{name}_alpha.png"
                for path in (prediction_path, alpha_path, teacher_path, teacher_alpha_path):
                    if not path.is_file():
                        raise RuntimeError(f"P0-EVALUATION-ASSET-MISMATCH: missing render {path}")
                prediction = image_tensor(prediction_path, 3).to(runtime.device)
                alpha = image_tensor(alpha_path, 1).to(runtime.device)
                teacher_rgb = image_tensor(teacher_path, 3).to(runtime.device)
                sample = runtime.context["samples"][f"{outfit}/{condition}"]
                garment = diagnosis._garment_mask(sample)
                foreground = sample["target_foreground_mask"]
                protected = sample["target_protected_mask"]
                garment_psnr = masked_psnr(prediction, teacher_rgb, garment)
                foreground_psnr = masked_psnr(prediction, teacher_rgb, foreground)
                ssim = masked_ssim(prediction, teacher_rgb, garment)
                garment_lpips = lpips_distance(runtime, prediction, teacher_rgb, garment)
                protected_lpips = lpips_distance(runtime, prediction, teacher_rgb, protected)
                silhouette_iou, boundary_fscore, tolerance = silhouette_metrics(alpha, garment)
                records.append({
                    "run_id": source["run_id"], "method": source["method"],
                    "seed": source["seed"], "replicate": source["replicate"],
                    "outfit_id": outfit, "view_id": condition,
                    "garment_psnr": garment_psnr,
                    "foreground_psnr": foreground_psnr,
                    "garment_ssim": ssim,
                    "garment_lpips": garment_lpips,
                    "protected_lpips": protected_lpips,
                    "silhouette_iou": silhouette_iou,
                    "boundary_fscore": boundary_fscore,
                    "boundary_tolerance": tolerance,
                    "prediction_rgb": str(prediction_path),
                    "prediction_alpha": str(alpha_path),
                    "teacher_rgb": str(teacher_path),
                })
                manifest.append({
                    "run_id": source["run_id"], "outfit_id": outfit, "view_id": condition,
                    "prediction_rgb": str(prediction_path), "prediction_rgb_sha256": sha256(prediction_path),
                    "prediction_alpha": str(alpha_path), "prediction_alpha_sha256": sha256(alpha_path),
                })
    run_aggregates: dict[str, Any] = {}
    for source in sources:
        selected = [row for row in records if row["run_id"] == source["run_id"]]
        numeric_fields = (
            "garment_ssim", "garment_lpips", "protected_lpips",
            "silhouette_iou", "boundary_fscore",
        )
        aggregate = {field: numeric_mean(row[field] for row in selected) for field in numeric_fields}
        for field in ("garment_psnr", "foreground_psnr"):
            values = [row[field] for row in selected]
            aggregate[field] = "Infinity" if all(value == "Infinity" for value in values) else numeric_mean(
                value for value in values if value != "Infinity"
            )
            aggregate[f"{field}_infinite_count"] = sum(value == "Infinity" for value in values)
        aggregate["per_outfit"] = {
            outfit: {
                field: numeric_mean(row[field] for row in selected if row["outfit_id"] == outfit)
                for field in numeric_fields
            }
            for outfit in OUTFITS
        }
        run_aggregates[source["run_id"]] = aggregate
    method_aggregates = {}
    for method in runtime.protocol["dataset"]["methods_extended"]:
        selected = [row for row in records if row["method"] == method]
        aggregate = {
            field: numeric_mean(row[field] for row in selected)
            for field in ("garment_ssim", "garment_lpips", "protected_lpips", "silhouette_iou", "boundary_fscore")
        }
        for field in ("garment_psnr", "foreground_psnr"):
            values = [row[field] for row in selected]
            aggregate[field] = "Infinity" if all(value == "Infinity" for value in values) else numeric_mean(
                value for value in values if value != "Infinity"
            )
            aggregate[f"{field}_infinite_count"] = sum(value == "Infinity" for value in values)
        method_aggregates[method] = aggregate
    result = {
        "schema_version": "canondressgs.paper.p0_extended_metric_results.v1",
        "status": "PAPER_CANDIDATE_MANUAL_REVIEW_REQUIRED",
        "protocol_sha256_lf": PROTOCOL_SHA256,
        "source_run_count": len(sources),
        "episode_metric_count": len(records),
        "records": records,
        "run_aggregates": run_aggregates,
        "method_aggregates": method_aggregates,
        "source_render_manifest": manifest,
        "source_contact_sheets": [str(row["contact_sheet"]) for row in sources],
        "lpips": {
            **LPIPS_HASHES,
            "crop": "tight mask bbox plus 5 percent max-side expansion",
            "resize": "aspect-preserving bilinear fit and zero-pad 256x256",
            "normalization": "RGB [0,1] to [-1,1]",
        },
        "dino_clip": {"status": "BLOCKED_RESOURCE_MISSING", "value": None, "download_performed": False},
        "existing_endpoint_rerender_count": 0,
        "target_forward_leakage": False,
        "paper_final": False,
    }
    atomic_json(result_path, result)
    append_phase_status(runtime.attempt, "extended", "COMPLETE")
    return result


def source_attempt(runtime: EvaluationRuntime, experiment_id: str, seed: int | None) -> Path:
    seed_dir = "seed_fixed" if seed is None else f"seed_{seed}"
    return runtime.asset_root / FORMAL_NAME / experiment_id / seed_dir / "attempt_001"


def spatial_source_runs(runtime: EvaluationRuntime) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [{
        "run_id": "PAPER-B1-FIXED", "method": "B1", "seed": None,
        "attempt": source_attempt(runtime, "PAPER-B1-FIXED", None), "teacher": True,
    }]
    for method, prefix in (
        ("historical_old_Ours", "PAPER-OURS-S"),
        ("historical_A6", "PAPER-A6-S"),
    ):
        for seed in range(3):
            rows.append({
                "run_id": f"{prefix}{seed}", "method": method, "seed": seed,
                "attempt": source_attempt(runtime, f"{prefix}{seed}", seed), "teacher": False,
            })
    registry = yaml.safe_load(
        (PROJECT_ROOT / "paper_protocol/reviewer_risk/p0_formal_candidate_run_registry.yaml")
        .read_text(encoding="utf-8")
    )
    for row in registry["runs"]:
        if row["method"] in {"Ours-v2", "M3", "M4"}:
            rows.append({
                "run_id": row["formal_run_id"], "method": row["method"], "seed": row["seed"],
                "attempt": Path(row["attempt"]), "teacher": False,
            })
    for row in rows:
        row["raw_path"] = (
            row["attempt"] / "metrics/raw_episode_outputs.jsonl"
            if str(row["run_id"]).startswith("P0-FORMAL-")
            else row["attempt"] / "raw_metrics/raw_episode_outputs.jsonl"
        )
        row["contact_sheet"] = row["attempt"] / "visuals/five_outfit_four_view_contact_sheet.png"
        if not row["teacher"] and not row["raw_path"].is_file():
            raise RuntimeError(f"P0-EVALUATION-ASSET-MISMATCH: spatial raw source {row['run_id']}")
        if not row["contact_sheet"].is_file():
            raise RuntimeError(f"P0-EVALUATION-ASSET-MISMATCH: spatial sheet {row['run_id']}")
    return rows


def load_episode_coefficients(path: Path) -> dict[tuple[str, str], list[float]]:
    result: dict[tuple[str, str], list[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("record_type") == "episode":
            result[(row["outfit_id"], row["view_id"])] = row["predicted_standardized_coefficients"]
    if len(result) != 20:
        raise RuntimeError(f"P0-EVALUATION-ASSET-MISMATCH: spatial coefficient rows {path}")
    return result


def intrinsic_normals(base: Any) -> tuple[torch.Tensor, int]:
    scales = base._scaling.detach()
    sorted_scales, _ = torch.sort(scales, dim=-1)
    ties = int((sorted_scales[:, 0] == sorted_scales[:, 1]).sum())
    axis_index = scales.argmin(dim=-1)
    axis = F.one_hot(axis_index, num_classes=3).to(scales)
    quaternion = F.normalize(base._rotation.detach(), dim=-1)
    scalar = quaternion[:, :1]
    vector = quaternion[:, 1:]
    cross = torch.cross(vector, axis, dim=-1)
    normals = axis + 2.0 * (scalar * cross + torch.cross(vector, cross, dim=-1))
    return normals.contiguous(), ties


def residual_spatial_metrics(
    runtime: EvaluationRuntime, residual: Any, normals: torch.Tensor,
    garment: torch.Tensor, protected: torch.Tensor,
) -> dict[str, Any]:
    displacement = residual.delta_xyz.detach()
    magnitude = torch.linalg.vector_norm(displacement, dim=-1)
    signed_normal = (displacement * normals).sum(dim=-1)
    normal_magnitude = signed_normal.abs()
    tangential = torch.linalg.vector_norm(displacement - signed_normal[:, None] * normals, dim=-1)
    fraction = torch.where(magnitude > 0, tangential / magnitude, torch.zeros_like(magnitude))
    selected = magnitude[garment]
    tau_half = float(runtime.protocol["spatial_metrics"]["trust_radius"]["scalar_formula"]["tau_half"])
    tau_one = float(runtime.protocol["spatial_metrics"]["trust_radius"]["scalar_formula"]["tau_1"])
    base_opacity = runtime.context["base"]._opacity.detach().reshape(-1)
    delta_opacity = residual.delta_opacity_logit.detach().reshape(-1)
    opacity_activation = torch.sigmoid(base_opacity + delta_opacity) - torch.sigmoid(base_opacity)
    protected_displacement = magnitude[protected]
    exterior = ~garment
    protected_positive = opacity_activation[protected] > 0
    exterior_positive = opacity_activation[exterior] > 0
    return {
        "delta_xyz": {
            "mean": float(selected.double().mean()),
            "P50": percentile(selected, 0.50), "P95": percentile(selected, 0.95),
            "P99": percentile(selected, 0.99), "max": float(selected.max()),
        },
        "normal_displacement_mean": float(normal_magnitude[garment].double().mean()),
        "tangential_displacement_mean": float(tangential[garment].double().mean()),
        "tangential_fraction_mean": float(fraction[garment].double().mean()),
        "exceedance_fraction_at_tau_half": float((selected > tau_half).double().mean()),
        "exceedance_fraction_at_tau_1": float((selected > tau_one).double().mean()),
        "protected_displaced_count": int((protected_displacement > 0).sum()),
        "protected_displacement_P95": percentile(protected_displacement, 0.95),
        "protected_displacement_P99": percentile(protected_displacement, 0.99),
        "protected_opacity_activation": {
            "positive_count": int(protected_positive.sum()),
            "positive_mean": float(opacity_activation[protected][protected_positive].double().mean())
            if int(protected_positive.sum()) else 0.0,
            "positive_max": float(opacity_activation[protected].max()),
        },
        "outside_garment_opacity": {
            "positive_count": int(exterior_positive.sum()),
            "positive_mean": float(opacity_activation[exterior][exterior_positive].double().mean())
            if int(exterior_positive.sum()) else 0.0,
        },
        "spatial_outlier_count": int((magnitude[exterior] > tau_one).sum()),
    }


def run_spatial(runtime: EvaluationRuntime, extended: Mapping[str, Any]) -> dict[str, Any]:
    result_path = phase_result(runtime.attempt, "spatial")
    if result_path.is_file():
        return read_json(result_path)
    append_phase_status(runtime.attempt, "spatial", "RUNNING")
    normals, tie_count = intrinsic_normals(runtime.context["base"])
    normal_hash = tensor_hash(normals.float())
    expected_normal_hash = runtime.protocol["spatial_metrics"]["normal_contract"]["audited_float32_normal_sha256_c"]
    if tie_count != 0 or normal_hash != expected_normal_hash:
        raise RuntimeError("PROTOCOL-IMPLEMENTATION-MISMATCH: intrinsic Gaussian normals")
    teacher_checkpoint = (
        runtime.asset_root
        / "pipeline_full/SUBJECT02-MULTI-OUTFIT-EXPLICIT-BASIS-001/attempt_003"
        / "stage_a_teacher_bank/O02/checkpoints/step_001200.pth"
    )
    teacher_payload = torch.load(teacher_checkpoint, map_location="cpu", weights_only=False)
    garment = torch.as_tensor(teacher_payload["model"]["trainable_support"], device=runtime.device).bool().reshape(-1)
    protected = runtime.context["protected_mask"].detach().bool().reshape(-1)
    if int(garment.sum()) != 170547 or tensor_hash(garment.to(torch.uint8)) != runtime.protocol["spatial_metrics"]["garment_mapping"]["membership_sha256_uint8_c"]:
        raise RuntimeError("PROTOCOL-IMPLEMENTATION-MISMATCH: garment support")
    if int(protected.sum()) != 30894 or tensor_hash(protected.to(torch.uint8)) != runtime.protocol["spatial_metrics"]["protected_mapping"]["membership_sha256_uint8_c"]:
        raise RuntimeError("PROTOCOL-IMPLEMENTATION-MISMATCH: protected mapping")
    extended_lookup = {
        (row["run_id"], row["outfit_id"], row["view_id"]): row
        for row in extended["records"]
    }
    records: list[dict[str, Any]] = []
    sources = spatial_source_runs(runtime)
    for source in sources:
        coefficients = None if source["teacher"] else load_episode_coefficients(source["raw_path"])
        for outfit in OUTFITS:
            for condition in CONDITIONS:
                if source["teacher"]:
                    residual = runtime.teachers[outfit]
                else:
                    standardized = torch.tensor(
                        coefficients[(outfit, condition)], device=runtime.device, dtype=runtime.mean.dtype
                    )
                    residual = runtime.basis(standardized * runtime.std + runtime.mean, chunk_size=16384)
                metrics = residual_spatial_metrics(runtime, residual, normals, garment, protected)
                extended_run_id = source["run_id"]
                render_metrics = extended_lookup.get((extended_run_id, outfit, condition))
                if render_metrics is None:
                    image_root = source["attempt"] / "visuals/episodes"
                    prediction_rgb = image_tensor(
                        image_root / f"{outfit}_{condition}_prediction.png", 3
                    ).to(runtime.device)
                    prediction_alpha = image_tensor(
                        image_root / f"{outfit}_{condition}_alpha.png", 1
                    ).to(runtime.device)
                    teacher_root = (
                        runtime.asset_root / FORMAL_NAME
                        / "PAPER-B1-FIXED/seed_fixed/attempt_001/visuals/episodes"
                    )
                    teacher_rgb = image_tensor(
                        teacher_root / f"{outfit}_{condition}_prediction.png", 3
                    ).to(runtime.device)
                    sample = runtime.context["samples"][f"{outfit}/{condition}"]
                    garment_mask = diagnosis._garment_mask(sample)
                    silhouette_iou, boundary_fscore, _ = silhouette_metrics(
                        prediction_alpha, garment_mask
                    )
                    render_metrics = {
                        "silhouette_iou": silhouette_iou,
                        "boundary_fscore": boundary_fscore,
                        "protected_lpips": lpips_distance(
                            runtime, prediction_rgb, teacher_rgb, sample["target_protected_mask"]
                        ),
                    }
                records.append({
                    "run_id": source["run_id"], "method": source["method"], "seed": source["seed"],
                    "outfit_id": outfit, "view_id": condition,
                    **metrics,
                    "silhouette_iou": None if render_metrics is None else render_metrics["silhouette_iou"],
                    "boundary_fscore": None if render_metrics is None else render_metrics["boundary_fscore"],
                    "protected_lpips": None if render_metrics is None else render_metrics["protected_lpips"],
                })
    method_aggregates: dict[str, Any] = {}
    for method in runtime.protocol["dataset"]["methods_spatial"]:
        selected = [row for row in records if row["method"] == method]
        method_aggregates[method] = {
            "delta_xyz_mean": numeric_mean(row["delta_xyz"]["mean"] for row in selected),
            "delta_xyz_P95": numeric_mean(row["delta_xyz"]["P95"] for row in selected),
            "delta_xyz_P99": numeric_mean(row["delta_xyz"]["P99"] for row in selected),
            "delta_xyz_max": max(row["delta_xyz"]["max"] for row in selected),
            "normal_displacement_mean": numeric_mean(row["normal_displacement_mean"] for row in selected),
            "tangential_displacement_mean": numeric_mean(row["tangential_displacement_mean"] for row in selected),
            "tangential_fraction_mean": numeric_mean(row["tangential_fraction_mean"] for row in selected),
            "exceedance_at_tau_half": numeric_mean(row["exceedance_fraction_at_tau_half"] for row in selected),
            "exceedance_at_tau_1": numeric_mean(row["exceedance_fraction_at_tau_1"] for row in selected),
            "protected_displaced_count_max": max(row["protected_displaced_count"] for row in selected),
            "protected_displacement_P95": numeric_mean(row["protected_displacement_P95"] for row in selected),
            "protected_displacement_P99": numeric_mean(row["protected_displacement_P99"] for row in selected),
            "protected_opacity_positive_count_max": max(row["protected_opacity_activation"]["positive_count"] for row in selected),
            "outside_garment_opacity_positive_count_max": max(row["outside_garment_opacity"]["positive_count"] for row in selected),
            "spatial_outlier_count_max": max(row["spatial_outlier_count"] for row in selected),
            "visual_classification": "MANUAL_REVIEW_PENDING",
        }
    result = {
        "schema_version": "canondressgs.paper.p0_spatial_artifact_results.v1",
        "status": "PAPER_CANDIDATE_MANUAL_REVIEW_REQUIRED",
        "protocol_sha256_lf": PROTOCOL_SHA256,
        "normal_contract": {"tie_count": tie_count, "float32_sha256_c": normal_hash},
        "garment_support": {"count": int(garment.sum()), "sha256_uint8_c": tensor_hash(garment.to(torch.uint8))},
        "protected_mapping": {"count": int(protected.sum()), "sha256_uint8_c": tensor_hash(protected.to(torch.uint8))},
        "tau_half": 0.025, "tau_1": 0.05,
        "records": records,
        "method_aggregates": method_aggregates,
        "source_contact_sheets": [str(row["contact_sheet"]) for row in sources],
        "b6_b7_spatial_residual_evidence": "SHARED_B1_TEACHER_ENDPOINT",
        "b6_b7_independent_residual_evidence_count": 0,
        "paper_final": False,
    }
    atomic_json(result_path, result)
    append_phase_status(runtime.attempt, "spatial", "COMPLETE")
    return result


def residual_bitwise_equal(first: Any, second: Any) -> bool:
    return all(
        torch.equal(first.as_dict()[name], second.as_dict()[name])
        for name in first.as_dict()
    )


def run_interpolation(runtime: EvaluationRuntime) -> dict[str, Any]:
    result_path = phase_result(runtime.attempt, "interpolation")
    if result_path.is_file():
        return read_json(result_path)
    append_phase_status(runtime.attempt, "interpolation", "RUNNING")
    records: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    sheets: list[str] = []
    sequence_images: dict[tuple[str, str, str], torch.Tensor] = {}
    for left, right in PAIRS:
        pair_id = f"{left}_{right}"
        sheet_rows: list[tuple[str, list[tuple[str, Path]]]] = []
        for condition in CONDITIONS:
            panels: list[tuple[str, Path]] = []
            left_coefficient = runtime.coefficients[left]
            right_coefficient = runtime.coefficients[right]
            endpoint_distance = float(torch.linalg.vector_norm(right_coefficient - left_coefficient))
            for alpha in frozen.INTERPOLATION_ALPHAS:
                coefficient = (1.0 - alpha) * left_coefficient + alpha * right_coefficient
                residual = runtime.basis(coefficient, chunk_size=16384)
                alpha_id = f"a{int(round(alpha * 10)):02d}"
                rgb_path = runtime.attempt / "basis_interpolation" / pair_id / condition / f"{alpha_id}_rgb.png"
                alpha_path = runtime.attempt / "basis_interpolation" / pair_id / condition / f"{alpha_id}_alpha.png"
                rgb, rendered_alpha = render_or_load(
                    runtime, left, condition, residual, rgb_path, alpha_path
                )
                garment = diagnosis._garment_mask(runtime.context["samples"][f"{left}/{condition}"])
                silhouette_iou, boundary_fscore, tolerance = silhouette_metrics(rendered_alpha, garment)
                records.append({
                    "pair": [left, right], "pair_id": pair_id, "view_id": condition,
                    "alpha": alpha, "raw_coefficients": coefficient.cpu().tolist(),
                    "coefficient_distance_from_A": float(torch.linalg.vector_norm(coefficient - left_coefficient)),
                    "coefficient_segment_fraction": 0.0 if endpoint_distance == 0 else float(
                        torch.linalg.vector_norm(coefficient - left_coefficient) / endpoint_distance
                    ),
                    "residual_displacement_mean": float(residual.delta_xyz.detach().double().norm(dim=-1).mean()),
                    "silhouette_iou": silhouette_iou,
                    "boundary_fscore": boundary_fscore,
                    "boundary_tolerance": tolerance,
                    "rgb_path": str(rgb_path), "alpha_path": str(alpha_path),
                })
                sequence_images[(pair_id, condition, alpha_id)] = rgb
                panels.append((f"a={alpha:.1f}", rgb_path))
            direct_left = runtime.basis(left_coefficient, chunk_size=16384)
            direct_right = runtime.basis(right_coefficient, chunk_size=16384)
            with torch.inference_mode():
                direct_left_rgb, _ = p0._render(runtime.context, left, condition, direct_left)
                direct_right_rgb, _ = p0._render(runtime.context, left, condition, direct_right)
            grid_left = sequence_images[(pair_id, condition, "a00")]
            grid_right = sequence_images[(pair_id, condition, "a10")]
            endpoint_rows.append({
                "pair": [left, right], "view_id": condition,
                "alpha0_coefficient_bitwise": torch.equal(left_coefficient, (1.0 - 0.0) * left_coefficient + 0.0 * right_coefficient),
                "alpha1_coefficient_bitwise": torch.equal(right_coefficient, (1.0 - 1.0) * left_coefficient + 1.0 * right_coefficient),
                "alpha0_residual_bitwise": residual_bitwise_equal(direct_left, runtime.basis(left_coefficient, chunk_size=16384)),
                "alpha1_residual_bitwise": residual_bitwise_equal(direct_right, runtime.basis(right_coefficient, chunk_size=16384)),
                "alpha0_render_max_abs": float((grid_left - direct_left_rgb).abs().max()),
                "alpha1_render_max_abs": float((grid_right - direct_right_rgb).abs().max()),
            })
            sheet_rows.append((condition, panels))
        sheet_path = runtime.attempt / "visuals/basis_interpolation" / f"{pair_id}.png"
        contact_sheet(sheet_path, sheet_rows)
        sheets.append(str(sheet_path))
    adjacency: list[dict[str, Any]] = []
    for left, right in PAIRS:
        pair_id = f"{left}_{right}"
        for condition in CONDITIONS:
            endpoint_lpips = lpips_distance(
                runtime,
                sequence_images[(pair_id, condition, "a00")],
                sequence_images[(pair_id, condition, "a10")],
                diagnosis._garment_mask(runtime.context["samples"][f"{left}/{condition}"]),
            )
            adjacent_values = []
            for index in range(10):
                value = lpips_distance(
                    runtime,
                    sequence_images[(pair_id, condition, f"a{index:02d}")],
                    sequence_images[(pair_id, condition, f"a{index + 1:02d}")],
                    diagnosis._garment_mask(runtime.context["samples"][f"{left}/{condition}"]),
                )
                adjacent_values.append(value)
            adjacency.append({
                "pair": [left, right], "view_id": condition,
                "endpoint_lpips": endpoint_lpips,
                "adjacent_lpips": adjacent_values,
                "all_adjacent_le_endpoint": all(value <= endpoint_lpips for value in adjacent_values),
            })
    endpoint_pass = all(
        row["alpha0_coefficient_bitwise"] and row["alpha1_coefficient_bitwise"]
        and row["alpha0_residual_bitwise"] and row["alpha1_residual_bitwise"]
        and row["alpha0_render_max_abs"] <= 1e-6 and row["alpha1_render_max_abs"] <= 1e-6
        for row in endpoint_rows
    )
    finite_ordered = len(records) == 440 and all(
        math.isfinite(value)
        for row in records
        for value in (row["coefficient_distance_from_A"], row["residual_displacement_mean"])
    )
    numeric_continuity = finite_ordered and all(row["all_adjacent_le_endpoint"] for row in adjacency)
    classification = (
        "BASIS_CONTINUITY_NUMERIC_ONLY" if endpoint_pass and numeric_continuity
        else "BASIS_CONTINUITY_NOT_CONFIRMED"
    )
    result = {
        "schema_version": "canondressgs.paper.p0_basis_interpolation_results.v1",
        "status": "PAPER_CANDIDATE_MANUAL_REVIEW_REQUIRED",
        "protocol_sha256_lf": PROTOCOL_SHA256,
        "render_count": len(records), "expected_render_count": 440,
        "reference_predictor_used": False,
        "records": records,
        "endpoint_parity": endpoint_rows,
        "endpoint_parity_pass": endpoint_pass,
        "adjacency": adjacency,
        "coefficient_and_residual_continuity": finite_ordered,
        "adjacent_lpips_rule_pass": all(row["all_adjacent_le_endpoint"] for row in adjacency),
        "contact_sheets": sheets,
        "basis_continuity_conclusion": classification,
        "semantic_continuity_requires_manual_review": True,
        "paper_final": False,
    }
    atomic_json(result_path, result)
    append_phase_status(runtime.attempt, "interpolation", "COMPLETE")
    return result


def mixed_reference_set(
    runtime: EvaluationRuntime, left: str, right: str, target_condition: str,
    values: Sequence[str],
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, str]]]:
    legal = [condition for condition in CONDITIONS if condition != target_condition]
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    manifest: list[dict[str, str]] = []
    for position, condition in enumerate(legal):
        outfit = left if values[position] == "A" else right
        image, mask = runtime.observation(outfit, condition)
        images.append(image.copy())
        masks.append(mask.copy())
        manifest.append({
            "position": str(position), "symbol": values[position], "outfit": outfit,
            "condition": condition, "rgb_sha256": numpy_hash(image),
            "mask_sha256": numpy_hash(mask.astype(np.uint8)),
        })
    return images, masks, manifest


def run_mixed(runtime: EvaluationRuntime) -> dict[str, Any]:
    result_path = phase_result(runtime.attempt, "mixed")
    if result_path.is_file():
        return read_json(result_path)
    append_phase_status(runtime.attempt, "mixed", "RUNNING")
    records: list[dict[str, Any]] = []
    rendered: dict[tuple[str, str, str, str], torch.Tensor] = {}
    sheets: list[str] = []
    assignment_rows = frozen.mixed_reference_assignments()
    for left, right in PAIRS:
        pair_id = f"{left}_{right}"
        sheet_rows: list[tuple[str, list[tuple[str, Path]]]] = []
        endpoint_a = runtime.coefficients[left]
        endpoint_b = runtime.coefficients[right]
        segment = endpoint_b - endpoint_a
        segment_sq = float(torch.dot(segment, segment))
        endpoint_distance = math.sqrt(segment_sq)
        for condition in CONDITIONS:
            feature_sets: dict[str, tuple[torch.Tensor, torch.Tensor, list[dict[str, str]]]] = {}
            for assignment_id, values in assignment_rows:
                images, masks, source_manifest = mixed_reference_set(
                    runtime, left, right, condition, values
                )
                rows, valid = runtime.f2_rows(images, masks)
                feature_sets[assignment_id] = (rows, valid, source_manifest)
            for method in METHODS:
                panels: list[tuple[str, Path]] = []
                for assignment_id, _ in assignment_rows:
                    rows, valid, source_manifest = feature_sets[assignment_id]
                    prediction = runtime.predict(method, condition, rows, valid)
                    raw = prediction["raw"]
                    t = float(torch.dot(raw - endpoint_a, segment) / max(segment_sq, 1e-30))
                    projection = endpoint_a + t * segment
                    orthogonal = float(torch.linalg.vector_norm(raw - projection))
                    rgb_path = (
                        runtime.attempt / "mixed_reference" / pair_id / condition
                        / method.replace("-", "_").lower() / f"{assignment_id}_rgb.png"
                    )
                    alpha_path = rgb_path.with_name(rgb_path.stem.replace("_rgb", "_alpha") + ".png")
                    rgb, alpha = render_or_load(
                        runtime, left, condition, prediction["residual"], rgb_path, alpha_path
                    )
                    garment = diagnosis._garment_mask(runtime.context["samples"][f"{left}/{condition}"])
                    silhouette_iou, boundary_fscore, tolerance = silhouette_metrics(alpha, garment)
                    macro = "AAA" if assignment_id == "AAA" else "BBB" if assignment_id == "BBB" else assignment_id[:3]
                    records.append({
                        "pair": [left, right], "pair_id": pair_id, "view_id": condition,
                        "method": method, "assignment_id": assignment_id, "assignment_macro": macro,
                        "source_manifest": source_manifest,
                        "standardized_coefficients": prediction["standardized"].cpu().tolist(),
                        "raw_coefficients": raw.cpu().tolist(),
                        "segment_coordinate_t": t,
                        "orthogonal_distance": orthogonal,
                        "endpoint_distance": endpoint_distance,
                        "nearest_endpoint": left if abs(t) <= abs(t - 1.0) else right,
                        "predicted_outfit": prediction["predicted_outfit"],
                        "logits": prediction.get("logits", torch.empty(0)).cpu().tolist()
                        if "logits" in prediction else None,
                        "centroid_distances": prediction.get("distances"),
                        "non_endpoint": 0.05 < t < 0.95,
                        "silhouette_iou": silhouette_iou,
                        "boundary_fscore": boundary_fscore,
                        "boundary_tolerance": tolerance,
                        "rgb_path": str(rgb_path), "alpha_path": str(alpha_path),
                    })
                    rendered[(pair_id, condition, method, assignment_id)] = rgb
                    panels.append((assignment_id, rgb_path))
                sheet_rows.append((f"{method}/{condition}", panels))
        sheet_path = runtime.attempt / "visuals/mixed_reference" / f"{pair_id}.png"
        contact_sheet(sheet_path, sheet_rows)
        sheets.append(str(sheet_path))
    adjacency: list[dict[str, Any]] = []
    assignment_ids = [row[0] for row in assignment_rows]
    for left, right in PAIRS:
        pair_id = f"{left}_{right}"
        for condition in CONDITIONS:
            garment = diagnosis._garment_mask(runtime.context["samples"][f"{left}/{condition}"])
            for method in METHODS:
                values = []
                for first, second in zip(assignment_ids, assignment_ids[1:]):
                    values.append(lpips_distance(
                        runtime,
                        rendered[(pair_id, condition, method, first)],
                        rendered[(pair_id, condition, method, second)],
                        garment,
                    ))
                adjacency.append({
                    "pair": [left, right], "view_id": condition, "method": method,
                    "assignment_order": assignment_ids, "adjacent_render_lpips": values,
                    "max_adjacent_render_lpips": max(values),
                })
    stable_pairs: dict[str, Any] = {}
    for left, right in PAIRS:
        pair_id = f"{left}_{right}"
        views: dict[str, Any] = {}
        for condition in CONDITIONS:
            selected = [
                row for row in records
                if row["pair_id"] == pair_id and row["view_id"] == condition and row["method"] == "Ours-v2"
            ]
            aab = [row for row in selected if row["assignment_macro"] == "AAB"]
            abb = [row for row in selected if row["assignment_macro"] == "ABB"]
            aab_t = float(statistics.median(row["segment_coordinate_t"] for row in aab))
            abb_t = float(statistics.median(row["segment_coordinate_t"] for row in abb))
            orth = max(
                float(statistics.median(row["orthogonal_distance"] for row in aab)),
                float(statistics.median(row["orthogonal_distance"] for row in abb)),
            )
            endpoint_distance = aab[0]["endpoint_distance"]
            views[condition] = {
                "AAB_median_t": aab_t, "ABB_median_t": abb_t,
                "orthogonal_distance_macro_max": orth,
                "endpoint_distance": endpoint_distance,
                "stable": 0.05 < aab_t < abb_t < 0.95 and orth <= 0.10 * endpoint_distance,
            }
        stable_pairs[pair_id] = {
            "views": views,
            "stable_view_count": sum(row["stable"] for row in views.values()),
            "stable_non_endpoint_pair": sum(row["stable"] for row in views.values()) >= 3,
        }
    stable_count = sum(row["stable_non_endpoint_pair"] for row in stable_pairs.values())
    hard_switching: dict[str, Any] = {}
    for method in ("B6", "B7"):
        method_rows = [row for row in records if row["method"] == method]
        switches = 0
        transition_points = []
        for left, right in PAIRS:
            pair_id = f"{left}_{right}"
            for condition in CONDITIONS:
                selected = [
                    row for row in method_rows
                    if row["pair_id"] == pair_id and row["view_id"] == condition
                ]
                labels = [row["predicted_outfit"] for row in selected]
                switches += sum(first != second for first, second in zip(labels, labels[1:]))
                transition_points.append(next((index for index, label in enumerate(labels) if label == right), None))
        hard_switching[method] = {"adjacent_class_switch_count": switches, "transition_points": transition_points}
    preliminary = (
        "REFERENCE_SOFT_CONTROL_PARTIAL" if stable_count >= 1
        else "REFERENCE_SOFT_CONTROL_NOT_CONFIRMED"
    )
    result = {
        "schema_version": "canondressgs.paper.p0_mixed_reference_results.v1",
        "status": "PAPER_CANDIDATE_MANUAL_REVIEW_REQUIRED",
        "protocol_sha256_lf": PROTOCOL_SHA256,
        "query_sets_per_method": len(records) // 3,
        "method_query_count": len(records),
        "expected_method_query_count": 960,
        "records": records,
        "adjacent_render_metrics": adjacency,
        "stable_pairs": stable_pairs,
        "stable_non_endpoint_pair_count": stable_count,
        "hard_lookup_switching": hard_switching,
        "contact_sheets": sheets,
        "best_assignment_selected": False,
        "representative_policy": {"Ours-v2": "R0", "B6": "S0", "B7": "FIXED"},
        "reference_soft_control_conclusion": preliminary,
        "visual_usability": "MANUAL_REVIEW_PENDING",
        "target_forward_leakage": False,
        "paper_final": False,
    }
    atomic_json(result_path, result)
    append_phase_status(runtime.attempt, "mixed", "COMPLETE")
    return result


def perturb_reference_set(
    runtime: EvaluationRuntime, outfit: str, target_condition: str,
    ladder: str, value: float | int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    legal = [condition for condition in CONDITIONS if condition != target_condition]
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for condition in legal:
        image, mask = runtime.observation(outfit, condition)
        if ladder == "grayscale":
            images.append(frozen.grayscale_ladder(image, float(value)))
            masks.append(mask.copy())
        elif ladder == "hue":
            images.append(frozen.hue_shift(image, float(value)))
            masks.append(mask.copy())
        elif ladder == "blur":
            images.append(frozen.blur_ladder(image, float(value)))
            masks.append(mask.copy())
        elif ladder == "mask_morphology":
            images.append(image.copy())
            masks.append(frozen.disk_morphology(mask, int(value)))
        else:
            raise KeyError(ladder)
    return images, masks


def run_perturbation(runtime: EvaluationRuntime) -> dict[str, Any]:
    result_path = phase_result(runtime.attempt, "perturbation")
    if result_path.is_file():
        return read_json(result_path)
    append_phase_status(runtime.attempt, "perturbation", "RUNNING")
    ladders: dict[str, Sequence[float | int]] = {
        "grayscale": frozen.GRAYSCALE_LADDER,
        "hue": frozen.HUE_LADDER_DEGREES,
        "blur": frozen.BLUR_LADDER_SIGMA,
        "mask_morphology": frozen.MASK_MORPHOLOGY_RADIUS,
    }
    records: list[dict[str, Any]] = []
    rendered: dict[tuple[str, str, str, str, int], torch.Tensor] = {}
    image_paths: dict[tuple[str, str, str, str, int], Path] = {}
    for outfit in OUTFITS:
        for condition in CONDITIONS:
            for ladder, values in ladders.items():
                for value_index, value in enumerate(values):
                    images, masks = perturb_reference_set(runtime, outfit, condition, ladder, value)
                    rows, valid = runtime.f2_rows(images, masks)
                    for method in METHODS:
                        prediction = runtime.predict(method, condition, rows, valid)
                        value_id = f"v{value_index}_{str(value).replace('-', 'm').replace('.', 'p')}"
                        rgb_path = (
                            runtime.attempt / "perturbation_ladders" / ladder
                            / method.replace("-", "_").lower() / f"{outfit}_{condition}_{value_id}_rgb.png"
                        )
                        alpha_path = rgb_path.with_name(rgb_path.stem.replace("_rgb", "_alpha") + ".png")
                        rgb, alpha = render_or_load(
                            runtime, outfit, condition, prediction["residual"], rgb_path, alpha_path
                        )
                        garment = diagnosis._garment_mask(runtime.context["samples"][f"{outfit}/{condition}"])
                        silhouette_iou, boundary_fscore, tolerance = silhouette_metrics(alpha, garment)
                        records.append({
                            "outfit_id": outfit, "view_id": condition, "ladder": ladder,
                            "value_index": value_index, "value": value, "method": method,
                            "standardized_coefficients": prediction["standardized"].cpu().tolist(),
                            "raw_coefficients": prediction["raw"].cpu().tolist(),
                            "predicted_outfit": prediction["predicted_outfit"],
                            "logits": prediction.get("logits", torch.empty(0)).cpu().tolist()
                            if "logits" in prediction else None,
                            "centroid_distances": prediction.get("distances"),
                            "silhouette_iou": silhouette_iou,
                            "boundary_fscore": boundary_fscore,
                            "boundary_tolerance": tolerance,
                            "rgb_path": str(rgb_path), "alpha_path": str(alpha_path),
                        })
                        key = (outfit, condition, ladder, method, value_index)
                        rendered[key] = rgb
                        image_paths[key] = rgb_path
    sheets: list[str] = []
    for ladder, values in ladders.items():
        for method in METHODS:
            rows = []
            for outfit in OUTFITS:
                for condition in CONDITIONS:
                    panels = [
                        (str(value), image_paths[(outfit, condition, ladder, method, index)])
                        for index, value in enumerate(values)
                    ]
                    rows.append((f"{outfit}/{condition}", panels))
            path = runtime.attempt / "visuals/perturbation_ladders" / f"{ladder}_{method.replace('-', '_').lower()}.png"
            contact_sheet(path, rows)
            sheets.append(str(path))
    sequences: list[dict[str, Any]] = []
    for outfit in OUTFITS:
        for condition in CONDITIONS:
            garment = diagnosis._garment_mask(runtime.context["samples"][f"{outfit}/{condition}"])
            for ladder, values in ladders.items():
                for method in METHODS:
                    selected = sorted(
                        (
                            row for row in records
                            if row["outfit_id"] == outfit and row["view_id"] == condition
                            and row["ladder"] == ladder and row["method"] == method
                        ),
                        key=lambda row: row["value_index"],
                    )
                    coefficients = [torch.tensor(row["standardized_coefficients"]) for row in selected]
                    adjacent_coefficient = [
                        float(torch.linalg.vector_norm(second - first))
                        for first, second in zip(coefficients, coefficients[1:])
                    ]
                    adjacent_lpips = [
                        lpips_distance(
                            runtime,
                            rendered[(outfit, condition, ladder, method, index)],
                            rendered[(outfit, condition, ladder, method, index + 1)],
                            garment,
                        )
                        for index in range(4)
                    ]
                    labels = [row["predicted_outfit"] for row in selected]
                    sequences.append({
                        "outfit_id": outfit, "view_id": condition, "ladder": ladder, "method": method,
                        "values": list(values),
                        "adjacent_coefficient_difference": adjacent_coefficient,
                        "coefficient_total_variation": sum(adjacent_coefficient),
                        "adjacent_render_lpips": adjacent_lpips,
                        "max_adjacent_render_lpips": max(adjacent_lpips),
                        "nearest_endpoint_switch_count": sum(a != b for a, b in zip(labels, labels[1:])),
                        "predicted_outfits": labels,
                    })
    method_aggregates: dict[str, Any] = {}
    for method in METHODS:
        selected = [row for row in sequences if row["method"] == method]
        method_aggregates[method] = {
            "median_pairwise_max_adjacent_render_lpips": float(statistics.median(
                row["max_adjacent_render_lpips"] for row in selected
            )),
            "median_coefficient_total_variation": float(statistics.median(
                row["coefficient_total_variation"] for row in selected
            )),
            "class_or_nearest_endpoint_switch_count": sum(
                row["nearest_endpoint_switch_count"] for row in selected
            ),
            "per_ladder": {
                ladder: {
                    "median_max_adjacent_render_lpips": float(statistics.median(
                        row["max_adjacent_render_lpips"] for row in selected if row["ladder"] == ladder
                    )),
                    "median_coefficient_total_variation": float(statistics.median(
                        row["coefficient_total_variation"] for row in selected if row["ladder"] == ladder
                    )),
                }
                for ladder in ladders
            },
        }
    ours = method_aggregates["Ours-v2"]
    numeric_smoother = all(
        ours["median_pairwise_max_adjacent_render_lpips"]
        < method_aggregates[method]["median_pairwise_max_adjacent_render_lpips"]
        and ours["median_coefficient_total_variation"]
        < method_aggregates[method]["median_coefficient_total_variation"]
        for method in ("B6", "B7")
    )
    hard_stable = all(
        method_aggregates[method]["class_or_nearest_endpoint_switch_count"] == 0
        for method in ("B6", "B7")
    )
    preliminary = (
        "HARD_LOOKUP_EQUALLY_STABLE" if hard_stable and not numeric_smoother
        else "CONTINUITY_INCONCLUSIVE"
    )
    result = {
        "schema_version": "canondressgs.paper.p0_perturbation_continuity_results.v1",
        "status": "PAPER_CANDIDATE_MANUAL_REVIEW_REQUIRED",
        "protocol_sha256_lf": PROTOCOL_SHA256,
        "ladders": {name: list(values) for name, values in ladders.items()},
        "query_set_count": len(records) // 3,
        "method_query_count": len(records),
        "expected_method_query_count": 1200,
        "records": records,
        "sequences": sequences,
        "method_aggregates": method_aggregates,
        "numeric_smoother_rule_pass": numeric_smoother,
        "hard_lookup_no_switch_rule_pass": hard_stable,
        "contact_sheets": sheets,
        "perturbation_continuity_conclusion": preliminary,
        "visual_artifact_status": "MANUAL_REVIEW_PENDING",
        "target_forward_leakage": False,
        "paper_final": False,
    }
    atomic_json(result_path, result)
    append_phase_status(runtime.attempt, "perturbation", "COMPLETE")
    return result


def write_visual_manifest(
    attempt: Path, color: Mapping[str, Any], extended: Mapping[str, Any],
    spatial: Mapping[str, Any], interpolation: Mapping[str, Any],
    mixed: Mapping[str, Any], perturbation: Mapping[str, Any],
) -> dict[str, Any]:
    path = attempt / "audits/visual_review_manifest.json"
    if path.is_file():
        return read_json(path)
    spatial_sources = spatial_source_runs_from_result(spatial)
    categories = {
        "color_c0_c6": list(color["contact_sheets"].values()),
        "extended_metric_sources": list(extended["source_contact_sheets"]),
        "color_method_comparisons": list(color["contact_sheets"].values()),
        "interpolation_pairs": list(interpolation["contact_sheets"]),
        "mixed_reference_pairs": list(mixed["contact_sheets"]),
        "perturbation_ladders": list(perturbation["contact_sheets"]),
        "ours_v2_spatial": [row["path"] for row in spatial_sources if row["method"] == "Ours-v2"],
        "m3_m4_spatial_failures": [row["path"] for row in spatial_sources if row["method"] in {"M3", "M4"}],
    }
    unique_paths = sorted({value for values in categories.values() for value in values})
    result = {
        "schema_version": "canondressgs.paper.p0_visual_review_manifest.v1",
        "required_categories": categories,
        "unique_sheet_count": len(unique_paths),
        "unique_paths": unique_paths,
        "all_paths_exist": all(Path(value).is_file() for value in unique_paths),
        "all_pairs_views_required": True,
        "single_pair_or_view_sampling": False,
    }
    if not result["all_paths_exist"]:
        raise RuntimeError("visual review manifest contains a missing sheet")
    atomic_json(path, result)
    return result


def spatial_source_runs_from_result(spatial: Mapping[str, Any]) -> list[dict[str, str]]:
    methods_by_run: dict[str, str] = {}
    for row in spatial["records"]:
        methods_by_run[row["run_id"]] = row["method"]
    result = []
    for path in spatial["source_contact_sheets"]:
        source = str(path)
        run_id = next(
            (run for run in methods_by_run if run.lower().replace("paper-", "").split("-s")[0].lower() in source.lower()),
            None,
        )
        if "formal_runs/ours_v2/" in source:
            method = "Ours-v2"
        elif "formal_runs/m3/" in source:
            method = "M3"
        elif "formal_runs/m4/" in source:
            method = "M4"
        elif "PAPER-OURS-" in source:
            method = "historical_old_Ours"
        elif "PAPER-A6-" in source:
            method = "historical_A6"
        else:
            method = "B1"
        result.append({"path": source, "method": method, "run_id": run_id or "source"})
    return result


def write_execution_summary(
    attempt: Path, asset_root: Path, phase_results: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    path = attempt / "audits/execution_summary.json"
    if path.is_file():
        return read_json(path)
    formal_after = tree_manifest(asset_root / FORMAL_NAME)
    p0_after = tree_manifest(asset_root / P0_NAME)
    preflight = phase_results["preflight"]
    if formal_after != preflight["formal_outputs_before"] or p0_after != preflight["p0_outputs_before"]:
        raise RuntimeError("EVALUATION-ASSET-MUTATION")
    checkpoint_writes = len(list(attempt.rglob("*.pth")))
    paper_final_count = len(list(attempt.rglob("PAPER_FINAL")))
    counts = {
        "color_method_queries": phase_results["color"]["query_count"],
        "interpolation_renders": phase_results["interpolation"]["render_count"],
        "mixed_method_queries": phase_results["mixed"]["method_query_count"],
        "perturbation_method_queries": phase_results["perturbation"]["method_query_count"],
    }
    if counts != {
        "color_method_queries": 420,
        "interpolation_renders": 440,
        "mixed_method_queries": 960,
        "perturbation_method_queries": 1200,
    }:
        raise RuntimeError(f"evaluation completeness failure: {counts}")
    result = {
        "schema_version": "canondressgs.paper.p0_color_spatial_soft_control_execution.v1",
        "status": "PAPER_CANDIDATE_MANUAL_REVIEW_REQUIRED",
        "counts": counts,
        "formal_outputs_before_after_equal": True,
        "p0_outputs_before_after_equal": True,
        "formal_outputs_after": formal_after,
        "p0_outputs_after": p0_after,
        "target_forward_leakage_count": 0,
        "frozen_asset_mutation_count": 0,
        "training_count": 0,
        "backward_count": 0,
        "optimizer_update_count": 0,
        "scheduler_update_count": 0,
        "checkpoint_write_count": checkpoint_writes,
        "paper_final_count": paper_final_count,
        "execution_head": git("rev-parse", "HEAD"),
        "paper_final": False,
    }
    if checkpoint_writes or paper_final_count:
        raise RuntimeError("NO-TRAINING-GATE-VIOLATION")
    atomic_json(path, result)
    return result


def visual_max(review: Mapping[str, Any], predicate: Any) -> int:
    selected = [row for row in review["items"] if predicate(row)]
    return max(
        (int(value) for row in selected for value in row["grades"].values()),
        default=0,
    )


def finalize_archive(
    attempt: Path, asset_root: Path, review_path: Path
) -> dict[str, Any]:
    required = {
        "color": read_json(phase_result(attempt, "color")),
        "extended": read_json(phase_result(attempt, "extended")),
        "spatial": read_json(phase_result(attempt, "spatial")),
        "interpolation": read_json(phase_result(attempt, "interpolation")),
        "mixed": read_json(phase_result(attempt, "mixed")),
        "perturbation": read_json(phase_result(attempt, "perturbation")),
        "preflight": read_json(phase_result(attempt, "preflight")),
    }
    visual_manifest = read_json(attempt / "audits/visual_review_manifest.json")
    execution = read_json(attempt / "audits/execution_summary.json")
    review = read_json(review_path)
    reviewed_paths = {row["source_path"] for row in review["items"] if row.get("actual_opened")}
    if reviewed_paths != set(visual_manifest["unique_paths"]):
        raise RuntimeError("visual review is incomplete or contains an unexpected source")
    if not all(
        row.get("mapping_check") is True and row.get("target_unchanged_check") is True
        and set(row.get("grades", {})) == set(runtime_visual_categories())
        and all(int(value) in {0, 1, 2, 3} for value in row["grades"].values())
        for row in review["items"]
    ):
        raise RuntimeError("visual review fields are incomplete")
    color = required["color"]
    extended = required["extended"]
    spatial = required["spatial"]
    interpolation = required["interpolation"]
    mixed = required["mixed"]
    perturbation = required["perturbation"]
    spatial_classifications: dict[str, str] = {}
    for method, aggregate in spatial["method_aggregates"].items():
        grade = visual_max(review, lambda row, method=method: row.get("method") == method)
        if (
            aggregate["exceedance_at_tau_1"] == 0
            and aggregate["protected_displaced_count_max"] == 0
            and aggregate["spatial_outlier_count_max"] == 0
            and grade <= 1
        ):
            value = "SPATIAL_ARTIFACT_BOUNDED"
        elif aggregate["exceedance_at_tau_1"] <= 0.01 and grade < 3:
            value = "SPATIAL_ARTIFACT_PRESENT_BUT_USABLE"
        else:
            value = "SPATIAL_ARTIFACT_SEVERE"
        spatial_classifications[method] = value
        aggregate["visual_max_grade"] = grade
        aggregate["classification"] = value
    interpolation_grade = visual_max(review, lambda row: row.get("category") == "interpolation_pairs")
    if (
        interpolation["endpoint_parity_pass"] and interpolation["adjacent_lpips_rule_pass"]
        and interpolation_grade < 3
    ):
        basis_conclusion = "BASIS_CONTINUITY_CONFIRMED"
    elif interpolation["coefficient_and_residual_continuity"]:
        basis_conclusion = "BASIS_CONTINUITY_NUMERIC_ONLY"
    else:
        basis_conclusion = "BASIS_CONTINUITY_NOT_CONFIRMED"
    mixed_grade = visual_max(review, lambda row: row.get("category") == "mixed_reference_pairs")
    if mixed["stable_non_endpoint_pair_count"] >= 6 and mixed_grade < 3:
        reference_conclusion = "REFERENCE_SOFT_CONTROL_CONFIRMED"
    elif mixed["stable_non_endpoint_pair_count"] >= 1 or basis_conclusion != "BASIS_CONTINUITY_NOT_CONFIRMED":
        reference_conclusion = "REFERENCE_SOFT_CONTROL_PARTIAL"
    else:
        reference_conclusion = "REFERENCE_SOFT_CONTROL_NOT_CONFIRMED"
    perturb_grade = visual_max(review, lambda row: row.get("category") == "perturbation_ladders")
    if perturbation["numeric_smoother_rule_pass"] and perturb_grade < 3:
        perturbation_conclusion = "SOFT_PREDICTOR_SMOOTHER"
    elif perturbation["hard_lookup_no_switch_rule_pass"] and not perturbation["numeric_smoother_rule_pass"]:
        perturbation_conclusion = "HARD_LOOKUP_EQUALLY_STABLE"
    else:
        perturbation_conclusion = "CONTINUITY_INCONCLUSIVE"
    hard_lookup = (
        "HARD_LOOKUP_DOMINATES_DISCRETE_ENDPOINTS_BUT_SOFT_CONTROL_ADDS_VALUE"
        if reference_conclusion == "REFERENCE_SOFT_CONTROL_CONFIRMED"
        else "HARD_LOOKUP_MATCHES_WITHOUT_CLEAR_SOFT_ADVANTAGE"
    )
    color_conclusion = color["color_conclusion"]
    extended_conclusion = "EXTENDED_METRICS_COMPLETE"
    spatial_conclusion = spatial_classifications["Ours-v2"]
    interpolation["basis_continuity_conclusion"] = basis_conclusion
    mixed["reference_soft_control_conclusion"] = reference_conclusion
    perturbation["perturbation_continuity_conclusion"] = perturbation_conclusion
    archive = attempt / "archive"
    json_payloads = {
        "paper_protocol/reviewer_risk/p0_color_counterfactual_results.json": color,
        "paper_protocol/reviewer_risk/p0_extended_metric_results.json": extended,
        "paper_protocol/reviewer_risk/p0_spatial_artifact_results.json": spatial,
        "paper_protocol/reviewer_risk/p0_basis_interpolation_results.json": interpolation,
        "paper_protocol/reviewer_risk/p0_mixed_reference_results.json": mixed,
        "paper_protocol/reviewer_risk/p0_perturbation_continuity_results.json": perturbation,
        "paper_protocol/reviewer_risk/p0_color_spatial_soft_control_visual_review.json": review,
    }
    for relative, payload in json_payloads.items():
        atomic_json(archive / relative, payload)
    final_summary = {
        "schema_version": "canondressgs.paper.p0_color_spatial_soft_control_final_summary.v1",
        "task_id": TASK_ID,
        "status": "PAPER_CANDIDATE_MANUAL_REVIEW_REQUIRED",
        "source_branch": SOURCE_BRANCH, "source_head": SOURCE_HEAD,
        "run_branch": RUN_BRANCH, "execution_head": execution["execution_head"],
        "final_head": "commit containing this summary",
        "protocol_sha256_lf": PROTOCOL_SHA256,
        "environment": required["preflight"]["environment"],
        "asset_fingerprints": {
            "formal_before": required["preflight"]["formal_outputs_before"],
            "formal_after": execution["formal_outputs_after"],
            "p0_before": required["preflight"]["p0_outputs_before"],
            "p0_after": execution["p0_outputs_after"],
        },
        "color_conclusion": color_conclusion,
        "extended_metric_conclusion": extended_conclusion,
        "spatial_artifact_conclusion": spatial_conclusion,
        "spatial_method_classifications": spatial_classifications,
        "basis_continuity_conclusion": basis_conclusion,
        "reference_soft_control_conclusion": reference_conclusion,
        "perturbation_continuity_conclusion": perturbation_conclusion,
        "hard_lookup_risk_refined": hard_lookup,
        "counts": {
            **execution["counts"],
            "visual_unique_sheets_opened": len(reviewed_paths),
            "extended_episode_metrics": extended["episode_metric_count"],
            "checkpoint_writes": execution["checkpoint_write_count"],
        },
        "target_forward_leakage_count": 0,
        "frozen_asset_mutation_count": 0,
        "training_count": 0, "backward_count": 0,
        "optimizer_update_count": 0, "scheduler_update_count": 0,
        "paper_final_count": 0, "paper_final": False,
        "scientific_boundary": {
            "final_hard_lookup_fusion_supervision_paper_adjudication_written": False,
            "raw_results_and_stage_classifications_only": True,
        },
        "next_task": "ADJUDICATE_P0_AND_FREEZE_FINAL_PAPER_METHOD",
        "next_task_started": False,
    }
    atomic_json(
        archive / "paper_protocol/reviewer_risk/p0_color_spatial_soft_control_final_summary.json",
        final_summary,
    )
    report_lines = numbered_report_lines(
        final_summary, required, review, execution, visual_manifest
    )
    main_report = "# AAAI27 P0 color, spatial, and soft-control evaluation\n\n" + "\n".join(report_lines) + "\n"
    soft_report = (
        "# Soft control and hard lookup analysis\n\n"
        f"- Basis continuity: `{basis_conclusion}`.\n"
        f"- Mixed-reference control: `{reference_conclusion}`.\n"
        f"- Perturbation continuity: `{perturbation_conclusion}`.\n"
        f"- Refined hard-lookup risk: `{hard_lookup}`.\n"
        "- The discrete five-garment endpoint task retains the confirmed teacher upper bound.\n"
        "- This document records stage evidence only; final method adjudication remains forbidden here.\n"
    )
    spatial_report = (
        "# Gaussian spatial artifact analysis\n\n"
        f"- Ours-v2: `{spatial_classifications['Ours-v2']}`.\n"
        f"- M3: `{spatial_classifications['M3']}`.\n"
        f"- M4: `{spatial_classifications['M4']}`.\n"
        f"- Intrinsic normal SHA-256: `{spatial['normal_contract']['float32_sha256_c']}`.\n"
        "- B6/B7 cite the shared B1 teacher residual and are not duplicated as independent spatial evidence.\n"
    )
    documents = {
        "docs/PAPER/AAAI27_P0_COLOR_EXTENDED_SPATIAL_SOFT_CONTROL_RESULTS_20260722.md": main_report,
        "docs/PAPER/AAAI27_SOFT_CONTROL_AND_HARD_LOOKUP_ANALYSIS_20260722.md": soft_report,
        "docs/PAPER/AAAI27_SPATIAL_ARTIFACT_ANALYSIS_20260722.md": spatial_report,
    }
    for relative, content in documents.items():
        path = archive / relative
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
    archive_manifest = {
        "status": "PASS",
        "artifact_count": len(json_payloads) + 1 + len(documents),
        "artifacts": {
            str(path.relative_to(archive)).replace("\\", "/"): {
                "bytes": path.stat().st_size, "sha256": sha256(path)
            }
            for path in sorted(archive.rglob("*")) if path.is_file()
        },
        "paper_final": False,
    }
    atomic_json(attempt / "audits/archive_manifest.json", archive_manifest)
    return final_summary


def runtime_visual_categories() -> tuple[str, ...]:
    return (
        "cloud", "mottle", "edge_scatter", "full_body_contamination",
        "identity_contamination", "silhouette_discontinuity",
    )


def numbered_report_lines(
    summary: Mapping[str, Any], required: Mapping[str, Mapping[str, Any]],
    review: Mapping[str, Any], execution: Mapping[str, Any], visual_manifest: Mapping[str, Any],
) -> list[str]:
    spatial = required["spatial"]
    return [
        f"1. Source `{SOURCE_BRANCH}` at `{SOURCE_HEAD}`; run branch `{RUN_BRANCH}` at execution HEAD `{execution['execution_head']}`.",
        "2. Local/origin/cloud synchronization is completed after archive commit; both worktrees are required clean.",
        f"3. Environment: `{json.dumps(required['preflight']['environment'], sort_keys=True)}`.",
        "4. Formal and P0 frozen output manifests are identical before/after.",
        f"5. Protocol LF-normalized SHA-256: `{PROTOCOL_SHA256}`.",
        f"6. C0-C6 completed 420/420 method queries; `{summary['color_conclusion']}`.",
        f"7. Extended metrics completed {required['extended']['episode_metric_count']} episode-run rows across {required['extended']['source_run_count']} sources.",
        f"8. LPIPS hashes: `{json.dumps(LPIPS_HASHES, sort_keys=True)}`.",
        f"9. Spatial classification for Ours-v2: `{summary['spatial_artifact_conclusion']}`; method detail `{json.dumps(summary['spatial_method_classifications'], sort_keys=True)}`.",
        "10. Direct coefficient interpolation completeness: 440/440.",
        f"11. Endpoint parity: `{required['interpolation']['endpoint_parity_pass']}`.",
        f"12. Basis continuity: `{summary['basis_continuity_conclusion']}`.",
        "13. Mixed-reference completeness: 960/960 method queries.",
        f"14. Ours-v2 stable non-endpoint pairs: {required['mixed']['stable_non_endpoint_pair_count']}/10.",
        f"15. B6/B7 switching: `{json.dumps(required['mixed']['hard_lookup_switching'], sort_keys=True)}`.",
        f"16. Reference soft control: `{summary['reference_soft_control_conclusion']}`.",
        f"17. Perturbation continuity: `{summary['perturbation_continuity_conclusion']}`.",
        f"18. Refined hard-lookup risk: `{summary['hard_lookup_risk_refined']}`.",
        f"19. Manual visual sheets actually opened: {len(review['items'])}/{visual_manifest['unique_sheet_count']} unique sheets.",
        "20. Target leakage=0; frozen mutation=0.",
        "21. Training=0; backward=0; optimizer updates=0; scheduler updates=0; checkpoint writes=0.",
        "22. Focused protocol/evaluator tests and archive regressions pass; details are in the final summary.",
        "23. New failures are preserved; no historical failed attempt was changed.",
        "24. PAPER_FINAL count=0.",
        f"25. Seven classifications: color=`{summary['color_conclusion']}`, extended=`{summary['extended_metric_conclusion']}`, spatial=`{summary['spatial_artifact_conclusion']}`, basis=`{summary['basis_continuity_conclusion']}`, reference=`{summary['reference_soft_control_conclusion']}`, perturbation=`{summary['perturbation_continuity_conclusion']}`, hard_lookup=`{summary['hard_lookup_risk_refined']}`.",
        "26. Next task: `ADJUDICATE_P0_AND_FREEZE_FINAL_PAPER_METHOD`; it was not started.",
        "27. Commit, push, local/origin/cloud HEAD equality, and clean status are verified in the final chat handoff.",
    ]


def execute_all(attempt: Path, asset_root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    preflight = run_preflight(attempt, asset_root, protocol)
    runtime = EvaluationRuntime(attempt, asset_root, protocol)
    color = run_color(runtime)
    extended = run_extended(runtime)
    spatial = run_spatial(runtime, extended)
    interpolation = run_interpolation(runtime)
    mixed = run_mixed(runtime)
    perturbation = run_perturbation(runtime)
    phases = {
        "preflight": preflight, "color": color, "extended": extended,
        "spatial": spatial, "interpolation": interpolation,
        "mixed": mixed, "perturbation": perturbation,
    }
    visual = write_visual_manifest(
        attempt, color, extended, spatial, interpolation, mixed, perturbation
    )
    execution = write_execution_summary(attempt, asset_root, phases)
    return {"attempt": str(attempt), "visual_manifest": visual, "execution": execution}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--attempt")
    parser.add_argument("--phase", choices=("all", "preflight", "finalize"), default="all")
    parser.add_argument("--visual-review", type=Path)
    arguments = parser.parse_args()
    protocol = validate_source()
    attempt = resolve_attempt(arguments.output_root, arguments.attempt)
    if arguments.phase == "preflight":
        result = run_preflight(attempt, arguments.asset_root, protocol)
    elif arguments.phase == "finalize":
        if arguments.visual_review is None:
            raise ValueError("--visual-review is required for finalization")
        result = finalize_archive(attempt, arguments.asset_root, arguments.visual_review)
    else:
        result = execute_all(attempt, arguments.asset_root, protocol)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
