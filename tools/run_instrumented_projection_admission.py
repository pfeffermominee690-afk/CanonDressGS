from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene.instrumented_projection_admission import (  # noqa: E402
    ALPHA_CUTOFF,
    InstrumentationOptions,
    REJECTION_REASONS,
    backend_first_rejections,
    backend_pixel_pipeline,
    counterfactual_alpha,
    independent_gaussian_projection_oracle_v1,
    independent_pixel_support,
    projection_agreement,
    run_backend_projection_debug,
    same_index_displacement,
    support_funnel_counts,
)
from tools.run_alpha_raster_audit import (  # noqa: E402
    _checkpoint_path,
    _environment,
    _git_state,
    _largest_component,
    _mask_numpy,
    _tensor_state_fingerprint,
    atomic_json,
    atomic_text,
    deterministic_mask_indices,
    load_runtime,
    load_state_model,
    render_explicit,
    sha256,
    target_free_state,
    trusted_regions,
)
from utils.dressable_camera_utils import build_mmlphuman_camera  # noqa: E402


SCHEMA = "canondressgs.instrumented_projection_admission.v1"
STATES = ("P1", "P2", "P3")
VIEWS = {"cond_000000": "front", "cond_000318": "back", "cond_000347": "right"}
SAMPLES = (
    ("O08", "cond_000318"),
    ("O08", "cond_000347"),
    ("O08", "cond_000000"),
    ("O01", "cond_000318"),
    ("O01", "cond_000347"),
)
OUTPUT_DIRS = (
    "contract",
    "input_audit",
    "independent_projection",
    "backend_projection",
    "pre_tile_admission",
    "tile_intersection",
    "contributor_pipeline",
    "stage_comparison",
    "synthetic_unit_scenes",
    "visualizations",
    "final_adjudication",
)
DEFAULT_OUTPUT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/"
    "SUBJECT02-INSTRUMENTED-PROJECTION-ADMISSION-001/attempt_001"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git(*arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def resolve_ref(branch: str) -> str:
    for candidate in (branch, f"origin/{branch}", f"cloud/{branch}"):
        process = subprocess.run(["git", "rev-parse", candidate], cwd=PROJECT_ROOT, capture_output=True, text=True)
        if process.returncode == 0:
            return process.stdout.strip()
    raise RuntimeError(f"cannot resolve frozen ref: {branch}")


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != SCHEMA or value.get("task_id") != "SUBJECT02-INSTRUMENTED-PROJECTION-ADMISSION-001":
        raise ValueError("wrong projection/admission audit config")
    if int(value["permissions"]["optimizer_steps"]) != 0 or value["permissions"]["training"]:
        raise ValueError("diagnostic contract forbids optimization")
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _source_evidence_hashes(config: Mapping[str, Any]) -> dict[str, str]:
    root = Path(config["source_alpha_audit_output"])
    relative = (
        "RUN_STATUS.json",
        "contract/run_manifest.json",
        "final_adjudication/ALPHA_RASTER_AUDIT_FINAL_STATUS.json",
        "final_adjudication/GATE_ACCEPTANCE.md",
    )
    result = {}
    for item in relative:
        path = root / item
        if not path.is_file():
            raise FileNotFoundError(path)
        result[str(path)] = sha256(path)
    return result


def _options(config: Mapping[str, Any], *, enabled: bool) -> InstrumentationOptions:
    renderer = config["renderer_contract"]
    return InstrumentationOptions(
        enabled=enabled,
        eps2d=float(renderer["eps2d"]),
        near_plane=float(renderer["near_plane"]),
        far_plane=float(renderer["far_plane"]),
        radius_clip=float(renderer["radius_clip"]),
        tile_size=int(renderer["tile_size"]),
        alpha_cutoff=float(renderer["alpha_cutoff"]),
        alpha_clamp_max=float(renderer["alpha_clamp_max"]),
    )


def _projection_inputs(base: Any, overrides: Any, tensors: Mapping[str, torch.Tensor], camera: Mapping[str, Any]) -> dict[str, torch.Tensor | int]:
    return {
        "canonical_xyz": overrides.xyz,
        "canonical_scaling": base.compute_cano_scaling(overrides.as_dict()),
        "canonical_quaternion": base.compute_cano_rotation(overrides.as_dict()),
        "canonical_opacity": overrides.opacity,
        "posed_xyz": tensors["posed_xyz"],
        "posed_covariance": tensors["posed_covariance"],
        "opacity": tensors["effective_opacity"],
        "w2c": camera["w2c"],
        "K": camera["K"],
        "width": int(camera["width"]),
        "height": int(camera["height"]),
    }


class ParquetSink:
    def __init__(self, path: Path):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError("formal audit requires pyarrow for genuine Parquet output") from error
        self.pa, self.pq, self.path, self.writer = pa, pq, path, None

    def append(self, columns: Mapping[str, Any]) -> None:
        table = self.pa.table(columns)
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = self.pq.ParquetWriter(self.path, table.schema, compression="zstd", use_dictionary=True)
        self.writer.write_table(table)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def _numpy(value: torch.Tensor, dtype: Any | None = None) -> np.ndarray:
    result = value.detach().cpu().numpy()
    return result.astype(dtype, copy=False) if dtype is not None else result


def append_projection_parquet(
    sink: ParquetSink,
    state: str,
    outfit: str,
    condition: str,
    projection: Mapping[str, Any],
    backend: Mapping[str, Any],
    stages: torch.Tensor,
    codes: torch.Tensor,
    emitted: torch.Tensor,
) -> None:
    n = int(projection["gaussian_index"].numel())
    cov = projection["covariance2d"]
    columns = {
        "state": np.repeat(state, n),
        "outfit": np.repeat(outfit, n),
        "condition": np.repeat(condition, n),
        "view": np.repeat(VIEWS[condition], n),
        "gaussian_index": _numpy(projection["gaussian_index"], np.int32),
        "camera_x": _numpy(projection["camera_xyz"][:, 0], np.float32),
        "camera_y": _numpy(projection["camera_xyz"][:, 1], np.float32),
        "camera_z": _numpy(projection["camera_xyz"][:, 2], np.float32),
        "depth": _numpy(projection["depth"], np.float32),
        "projected_x": _numpy(projection["projected_mean"][:, 0], np.float32),
        "projected_y": _numpy(projection["projected_mean"][:, 1], np.float32),
        "cov_xx": _numpy(cov[:, 0, 0], np.float32),
        "cov_xy": _numpy(cov[:, 0, 1], np.float32),
        "cov_yy": _numpy(cov[:, 1, 1], np.float32),
        "conic_a": _numpy(projection["conic"][:, 0], np.float32),
        "conic_b": _numpy(projection["conic"][:, 1], np.float32),
        "conic_c": _numpy(projection["conic"][:, 2], np.float32),
        "determinant": _numpy(projection["determinant"], np.float32),
        "radius_x": _numpy(projection["radii_xy"][:, 0], np.int32),
        "radius_y": _numpy(projection["radii_xy"][:, 1], np.int32),
        "tile_min_x": _numpy(projection["tile_bbox_min"][:, 0], np.int16),
        "tile_min_y": _numpy(projection["tile_bbox_min"][:, 1], np.int16),
        "tile_max_x": _numpy(projection["tile_bbox_max"][:, 0], np.int16),
        "tile_max_y": _numpy(projection["tile_bbox_max"][:, 1], np.int16),
        "independent_tile_count": _numpy(projection["tile_count"], np.int32),
        "backend_radius_x": _numpy(backend["radii"][0, :, 0], np.int32),
        "backend_radius_y": _numpy(backend["radii"][0, :, 1], np.int32),
        "backend_tile_count": _numpy(emitted, np.int32),
        "first_rejection_stage": _numpy(stages, np.int8),
        "first_rejection_code": _numpy(codes, np.int8),
    }
    for index in range(6):
        columns[f"s{index}_pass"] = _numpy(projection["stage_masks_s0_s5"][:, index], np.bool_)
    columns["s6_pass"] = _numpy(emitted > 0, np.bool_)
    sink.append(columns)


def _projection_center_mask(projection: Mapping[str, Any], image_mask: np.ndarray) -> torch.Tensor:
    means = projection["projected_mean"].detach().cpu()
    x, y = torch.floor(means[:, 0]).long(), torch.floor(means[:, 1]).long()
    valid = (x >= 0) & (x < image_mask.shape[1]) & (y >= 0) & (y < image_mask.shape[0])
    result = torch.zeros(means.shape[0], dtype=torch.bool)
    if valid.any():
        source = torch.from_numpy(image_mask.astype(np.bool_))
        result[valid] = source[y[valid], x[valid]]
    return result.to(projection["projected_mean"].device)


def _region_masks(outfit: str, sample: Mapping[str, Any], regions: Mapping[str, torch.Tensor], alpha: np.ndarray) -> dict[str, np.ndarray]:
    target = _mask_numpy(sample["target_foreground_mask"])
    expansion = _mask_numpy(regions["trusted_expansion"])
    boundary = cv2.dilate(target.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool) ^ cv2.erode(target.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    masks = {
        "trusted_expansion": expansion,
        "correctly_covered_expansion": expansion & (alpha >= .5),
        "trusted_removal": _mask_numpy(regions["trusted_removal"]),
        "trailing_cloud": _largest_component((alpha >= .05) & (~target)),
        "normal_background": _mask_numpy(regions["background"]) & (alpha < .01),
    }
    if outfit == "O01":
        masks["O01_normal_boundary"] = boundary
    return masks


def _trace_region(
    projection: Mapping[str, Any],
    backend: Mapping[str, Any],
    opacity: torch.Tensor,
    mask: np.ndarray,
    options: InstrumentationOptions,
    maximum_pixels: int = 128,
) -> tuple[set[int], set[int], set[int]]:
    tiles: set[int] = set()
    eligible: set[int] = set()
    contributed: set[int] = set()
    for y, x in deterministic_mask_indices(mask, maximum_pixels):
        trace = backend_pixel_pipeline(backend, opacity, y, x, tile_size=options.tile_size)
        tiles.update(trace["tile_candidates"])
        eligible.update(trace["alpha_eligible"])
        contributed.update(trace["contributed"])
    return tiles, eligible, contributed


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {name: 0.0 for name in ("mean", "median", "p10", "p90")}
    return {"mean": float(array.mean()), "median": float(np.median(array)), "p10": float(np.quantile(array, .1)), "p90": float(np.quantile(array, .9))}


def _save_density(path: Path, projection: Mapping[str, Any], width: int, height: int) -> None:
    means = projection["projected_mean"].detach().cpu().numpy()
    valid = projection["stage_masks_s0_s5"][:, 4].detach().cpu().numpy()
    histogram, _, _ = np.histogram2d(means[valid, 1], means[valid, 0], bins=(height, width), range=((0, height), (0, width)))
    image = np.log1p(histogram)
    image = (255 * image / max(float(image.max()), 1e-12)).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, "L").save(path)


def _make_contact_sheet(path: Path, panels: Sequence[tuple[str, Path]], columns: int = 3) -> None:
    loaded = [(label, Image.open(image_path).convert("RGB")) for label, image_path in panels if image_path.is_file()]
    if not loaded:
        return
    panel_width = 420
    thumbs = []
    for label, image in loaded:
        scale = panel_width / image.width
        thumb = image.resize((panel_width, max(1, int(image.height * scale))), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (panel_width, thumb.height + 36), "white")
        canvas.paste(thumb, (0, 36)); ImageDraw.Draw(canvas).text((8, 10), label, fill="black")
        thumbs.append(canvas)
    rows = math.ceil(len(thumbs) / columns); cell_height = max(item.height for item in thumbs)
    sheet = Image.new("RGB", (columns * panel_width, rows * cell_height), "white")
    for index, item in enumerate(thumbs):
        sheet.paste(item, ((index % columns) * panel_width, (index // columns) * cell_height))
    path.parent.mkdir(parents=True, exist_ok=True); sheet.save(path)


def _synthetic_scenes(root: Path, device: torch.device, options: InstrumentationOptions, real_context: Mapping[str, Any]) -> list[dict[str, Any]]:
    from gsplat import rasterization

    K = torch.tensor([[100., 0., 32.], [0., 100., 32.], [0., 0., 1.]], device=device)
    w2c = torch.eye(4, device=device)
    specifications = [
        ("single_center", torch.tensor([[0., 0., 2.]], device=device), torch.diag(torch.tensor([.0025, .0025, .001], device=device))[None], torch.tensor([.8], device=device), (32, 32)),
        ("single_image_boundary", torch.tensor([[-.63, 0., 2.]], device=device), torch.diag(torch.tensor([.0025, .0025, .001], device=device))[None], torch.tensor([.8], device=device), (32, 0)),
        ("single_tile_boundary", torch.tensor([[-.32, -.32, 2.]], device=device), torch.diag(torch.tensor([.0025, .0025, .001], device=device))[None], torch.tensor([.8], device=device), (16, 16)),
        ("thin_minor_radius", torch.tensor([[0., 0., 2.]], device=device), torch.diag(torch.tensor([.01, 1e-6, .001], device=device))[None], torch.tensor([.8], device=device), (32, 32)),
        ("anisotropic_major_minor", torch.tensor([[0., 0., 2.]], device=device), torch.diag(torch.tensor([.02, .0001, .001], device=device))[None], torch.tensor([.8], device=device), (32, 32)),
        ("near_plane", torch.tensor([[0., 0., .1001]], device=device), torch.diag(torch.tensor([1e-5, 1e-5, 1e-5], device=device))[None], torch.tensor([.8], device=device), (32, 32)),
        ("multi_alpha_accumulation", torch.tensor([[0., 0., 2.], [0., 0., 2.1], [0., 0., 2.2]], device=device), torch.eye(3, device=device)[None].repeat(3, 1, 1) * .001, torch.tensor([.2, .2, .2], device=device), (32, 32)),
        ("equal_depth_sort", torch.tensor([[-.01, 0., 2.], [.01, 0., 2.]], device=device), torch.eye(3, device=device)[None].repeat(2, 1, 1) * .001, torch.tensor([.4, .4], device=device), (32, 32)),
        ("large_tile_bbox", torch.tensor([[0., 0., 2.]], device=device), torch.diag(torch.tensor([.2, .2, .001], device=device))[None], torch.tensor([.8], device=device), (32, 32)),
    ]
    rows = []
    for name, means, covars, opacity, pixel_yx in specifications:
        n = means.shape[0]
        projection = independent_gaussian_projection_oracle_v1(
            canonical_xyz=means, canonical_scaling=torch.zeros((n, 3), device=device),
            canonical_quaternion=torch.tensor([[1., 0., 0., 0.]], device=device).repeat(n, 1),
            canonical_opacity=opacity, posed_xyz=means, posed_covariance=covars, opacity=opacity,
            w2c=w2c, K=K, width=64, height=64, options=InstrumentationOptions(**{**options.__dict__, "enabled": False}),
        )
        backend = run_backend_projection_debug(means=means, covars=covars, opacities=opacity, w2c=w2c, K=K, width=64, height=64, options=options)
        colors = torch.ones((n, 3), device=device)
        _, alpha, _ = rasterization(means=means, quats=None, scales=None, opacities=opacity, colors=colors, viewmats=w2c[None], Ks=K[None], width=64, height=64, packed=False, near_plane=options.near_plane, covars=covars)
        y, x = pixel_yx
        trace = backend_pixel_pipeline(backend, opacity, y, x, tile_size=options.tile_size)
        reference = counterfactual_alpha(projection, torch.arange(n, device=device), [(y, x)])
        rows.append({"scene": name, "projection_pass": bool(projection_agreement(projection, backend)["pass"]), "production_alpha": float(alpha[0, y, x, 0]), "backend_reconstructed_alpha": trace["alpha"], "float64_no_cutoff_alpha": float(reference[0]), "tile_candidates": len(trace["tile_candidates"])})

    ids = torch.linspace(0, real_context["means"].shape[0] - 1, steps=256, device=device).long()
    means, covars, opacity = real_context["means"][ids], real_context["covars"][ids], real_context["opacity"][ids]
    camera = real_context["camera"]
    projection = independent_gaussian_projection_oracle_v1(
        canonical_xyz=means, canonical_scaling=torch.zeros_like(means), canonical_quaternion=torch.tensor([[1., 0., 0., 0.]], device=device).repeat(ids.numel(), 1), canonical_opacity=opacity,
        posed_xyz=means, posed_covariance=covars, opacity=opacity, w2c=camera["w2c"], K=camera["K"], width=int(camera["width"]), height=int(camera["height"]), options=InstrumentationOptions(**{**options.__dict__, "enabled": False}),
    )
    backend = run_backend_projection_debug(means=means, covars=covars, opacities=opacity, w2c=camera["w2c"], K=camera["K"], width=int(camera["width"]), height=int(camera["height"]), options=options)
    rows.append({"scene": "real_O08_reduced_256", **projection_agreement(projection, backend)})
    write_csv(root / "synthetic_unit_scenes/synthetic_scene_results.csv", rows)
    atomic_json(root / "synthetic_unit_scenes/synthetic_scene_results.json", rows)
    return rows


def _case_adjudication(config: Mapping[str, Any], projection_rows: Sequence[Mapping[str, Any]], aggregate_rows: Sequence[Mapping[str, Any]], rejection_counts: Counter[str]) -> dict[str, Any]:
    projection_pass = all(bool(row["pass"]) for row in projection_rows)
    by_state = {state: [row for row in aggregate_rows if row["state"] == state] for state in STATES}
    mean_pre = {state: float(np.mean([row["n_pre_mean"] for row in rows])) for state, rows in by_state.items()}
    placement = {state: mean_pre[state] / max(mean_pre["P1"], 1e-12) for state in ("P2", "P3")}
    admission = {state: float(np.mean([row["admission_retention_mean"] for row in rows])) for state, rows in by_state.items()}
    pixel = {state: float(np.mean([row["pixel_retention_mean"] for row in rows])) for state, rows in by_state.items()}
    total_rejected = sum(rejection_counts.values())
    dominant_reason, dominant_count = rejection_counts.most_common(1)[0] if rejection_counts else ("NONE", 0)
    dominant_fraction = dominant_count / max(total_rejected, 1)
    threshold = config["case_thresholds"]
    if not projection_pass:
        case = "IC"
    elif max(placement.values()) <= float(threshold["placement_n_pre_ratio_max"]) and min(admission[state] for state in ("P2", "P3")) >= float(threshold["placement_admission_retention_min"]) and dominant_fraction < float(threshold["admission_named_rejection_fraction_min"]):
        case = "IP"
    elif min(placement.values()) >= float(threshold["admission_n_pre_ratio_min"]) and max(admission[state] for state in ("P2", "P3")) < float(threshold["admission_retention_max"]) and dominant_fraction >= float(threshold["admission_named_rejection_fraction_min"]):
        case = "IA"
    elif min(placement.values()) < float(threshold["mixed_placement_ratio_max"]) and min(admission[state] for state in ("P2", "P3")) < float(threshold["mixed_admission_retention_max"]):
        case = "IM"
    else:
        case = "IU"
    causes = {"IP": "learned/projected support placement failure", "IA": "backend pre-tile admission/culling failure", "IC": "projection/covariance contract disagreement", "IM": "placement and admission both material", "IU": "unresolved by preregistered thresholds"}
    tasks = {"IP": "REDESIGN_SCREEN_SPACE_COVERAGE_PLACEMENT_OBJECTIVE", "IA": "FIX_GSPLAT_PRE_TILE_ADMISSION_CONTRACT", "IC": "FIX_GAUSSIAN_PROJECTION_COVARIANCE_CONTRACT", "IM": "SEPARATE_PLACEMENT_AND_ADMISSION_FAILURES", "IU": "VENDOR_GSPLAT_KERNEL_LEVEL_TRACE"}
    return {"preliminary_case": case, "root_cause": causes[case], "next_unique_task": tasks[case], "projection_contract_pass": projection_pass, "mean_n_pre": mean_pre, "placement_ratio": placement, "admission_retention": admission, "pixel_retention": pixel, "dominant_rejection_reason": dominant_reason, "dominant_rejection_fraction": dominant_fraction}


def _artifact_manifest(root: Path, *, phase: str) -> None:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "artifact_manifest.json"):
        rows.append({"relative_path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    atomic_json(root / "contract/artifact_manifest.json", {"phase": phase, "generated_at": now(), "git_head": git("rev-parse", "HEAD"), "file_count": len(rows), "files": rows})


def postprocess_existing(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    """Derive compact summaries from the immutable per-Gaussian Parquet output."""

    import pyarrow.parquet as pq

    root = args.output.resolve()
    status = json.loads((root / "RUN_STATUS.json").read_text(encoding="utf-8"))
    if status.get("status") != "STATIC_AUDIT_COMPLETE_AWAITING_VISUAL":
        raise RuntimeError("postprocess requires a completed static audit awaiting visual inspection")
    table = pq.read_table(
        root / "independent_projection/independent_projection_gaussians.parquet",
        columns=("state", "outfit", "condition", "first_rejection_stage", "first_rejection_code"),
    ).to_pandas()
    rows: list[dict[str, Any]] = []
    for (state, outfit, condition), group in table.groupby(["state", "outfit", "condition"], sort=True):
        counts: Counter[tuple[int, str]] = Counter()
        for stage_value, code_value in zip(group["first_rejection_stage"], group["first_rejection_code"]):
            code = int(code_value); stage = int(stage_value)
            reason = "EMITTED_NORMALLY" if code < 0 else REJECTION_REASONS[code]
            counts[(stage, reason)] += 1
        total = len(group)
        for (first_stage, reason), count in sorted(counts.items()):
            rows.append({"state": state, "outfit": outfit, "condition": condition, "view": VIEWS[condition], "first_rejection_stage": f"S{first_stage}", "reason": reason, "count": count, "fraction": count / max(total, 1), "total_gaussians": total})
    write_csv(root / "pre_tile_admission/stage_rejection_summary.csv", rows)
    atomic_json(root / "pre_tile_admission/stage_rejection_summary.json", rows)
    unexposed = sum(row["count"] for row in rows if row["reason"] == "BACKEND_UNEXPOSED_REJECTION")
    reason_totals = Counter()
    for row in rows:
        reason_totals[row["reason"]] += row["count"]
    atomic_text(root / "pre_tile_admission/REJECTION_REASON_REPORT.md", "\n".join([
        "# Projection / admission rejection reasons", "",
        f"- backend unexposed rejection count: `{unexposed}`",
        f"- backend unexposed fraction over registered Gaussian-state-view rows: `{unexposed / max(int(table.shape[0]), 1):.12f}`",
        "- `EMITTED_NORMALLY` means the Gaussian reached at least one tile intersection.",
        "- Target masks are not used in these per-Gaussian stage decisions.", "",
        "## Totals", "", *[f"- {reason}: {count}" for reason, count in reason_totals.most_common()],
    ]))
    projection = json.loads((root / "backend_projection/projection_agreement.json").read_text(encoding="utf-8"))
    aggregates = list(csv.DictReader((root / "stage_comparison/pixel_support_aggregates.csv").open(encoding="utf-8")))
    displacement = json.loads((root / "stage_comparison/same_index_displacement_summary.json").read_text(encoding="utf-8"))
    counterfactual = json.loads((root / "pre_tile_admission/pre_tile_rejection_counterfactual.json").read_text(encoding="utf-8"))
    preliminary = json.loads((root / "final_adjudication/PRELIMINARY_STATUS.json").read_text(encoding="utf-8"))
    numeric_keys = ("projected_mean_abs_median", "projected_mean_abs_p99", "depth_relative_p99", "radius_abs_p99", "radius_relative_p99", "covariance_abs_p99", "conic_abs_p99")
    atomic_json(root / "final_adjudication/evidence_summary.json", {"postprocess_commit": git("rev-parse", "HEAD"), "preliminary": preliminary, "projection_maxima": {key: max(float(row[key]) for row in projection) for key in numeric_keys}, "pixel_support_aggregates": aggregates, "same_index_displacement": displacement, "counterfactual": counterfactual, "global_rejection_totals": dict(reason_totals), "backend_unexposed_count": unexposed, "backend_unexposed_fraction": unexposed / max(int(table.shape[0]), 1), "optimizer_steps": 0})
    _artifact_manifest(root, phase="POSTPROCESSED_AWAITING_VISUAL")
    print(json.dumps({"status": "POSTPROCESS_PASS", "rows": int(table.shape[0]), "backend_unexposed_count": unexposed, "reason_totals": dict(reason_totals)}, indent=2))


def run_audit(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    root = args.output.resolve()
    if root.exists():
        raise FileExistsError(f"append-only output already exists: {root}")
    for name in OUTPUT_DIRS:
        (root / name).mkdir(parents=True, exist_ok=False)
    status_path = root / "RUN_STATUS.json"
    stage = "contract"
    atomic_json(status_path, {"status": "RUNNING", "stage": stage, "optimizer_steps": 0, "started_at": now()})
    parquet: ParquetSink | None = None
    started = time.perf_counter()
    try:
        head = git("rev-parse", "HEAD")
        if git("branch", "--show-current") != config["research_branch"] or git("status", "--short"):
            raise RuntimeError("formal audit requires the clean registered research branch")
        if subprocess.run(["git", "merge-base", "--is-ancestor", config["source_head"], head], cwd=PROJECT_ROOT).returncode:
            raise RuntimeError("run commit is not descended from frozen RF adjudication")
        if git("rev-list", "-n", "1", config["source_tag"]) != config["source_head"]:
            raise RuntimeError("source annotated tag drift")
        frozen_before = {branch: resolve_ref(branch) for branch in config["frozen_branches"]}
        if frozen_before != config["frozen_branches"]:
            raise RuntimeError("frozen branch drift")
        source_hashes_before = _source_evidence_hashes(config)
        checkpoint_manifest = []
        for state in STATES:
            for outfit in config["outfits"]:
                path = _checkpoint_path(config, state, outfit)
                actual = sha256(path)
                if actual != config["states"][state]["sha256"][outfit]:
                    raise RuntimeError(f"checkpoint drift: {state}/{outfit}")
                checkpoint_manifest.append({"state": state, "outfit": outfit, "path": str(path), "sha256": actual})
        atomic_json(root / "contract/config_resolved.json", config)
        atomic_json(root / "contract/run_manifest.json", {"schema_version": SCHEMA, "task_id": config["task_id"], "run_commit": head, "git": _git_state(), "environment": _environment(), "optimizer_steps": 0, "instrumentation_default_enabled": False, "installed_gsplat_modified": False, "source_alpha_audit_hashes": source_hashes_before, "checkpoints": checkpoint_manifest, "permissions": config["permissions"]})
        atomic_json(root / "input_audit/fixed_input_contract.json", {"status": "PASS", "samples": SAMPLES, "states": STATES, "source_alpha_audit_output": config["source_alpha_audit_output"], "target_fields_in_projection_forward": False, "optimizer_steps": 0})

        stage = "projection_and_admission"
        atomic_json(status_path, {"status": "RUNNING", "stage": stage, "optimizer_steps": 0, "updated_at": now()})
        parquet = ParquetSink(root / "independent_projection/independent_projection_gaussians.parquet")
        options_on, options_off = _options(config, enabled=True), _options(config, enabled=False)
        projection_rows: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []
        funnel_rows: list[dict[str, Any]] = []
        funnel_region_rows: list[dict[str, Any]] = []
        pixel_rows: list[dict[str, Any]] = []
        aggregate_rows: list[dict[str, Any]] = []
        debug_regression_rows: list[dict[str, Any]] = []
        rejection_counts: Counter[str] = Counter()
        base_fingerprints_before: dict[str, str] = {}
        base_fingerprints_after: dict[str, str] = {}
        contexts: dict[tuple[str, str], dict[str, Any]] = {}
        fixed_fn_masks: dict[str, np.ndarray] = {}
        density_panels: list[tuple[str, Path]] = []
        real_synthetic_context: dict[str, Any] | None = None

        for outfit in config["outfits"]:
            base, samples, background, device = load_runtime(config, outfit, args.device)
            base_fingerprints_before[outfit] = _tensor_state_fingerprint({"xyz": base._xyz, "scaling": base._scaling, "rotation": base._rotation, "opacity": base._opacity, "sh0": base._sh0, "shN": base._shN}.items())
            required_conditions = [condition for item_outfit, condition in SAMPLES if item_outfit == outfit]
            states = {condition: target_free_state(samples[condition], device) for condition in required_conditions}
            regions_by_condition = {condition: trusted_regions(samples[condition], config) for condition in required_conditions}

            if outfit == "O08":
                _, p3_overrides = load_state_model(config, "P3", outfit, base)
                for condition in ("cond_000318", "cond_000347"):
                    with torch.no_grad():
                        _, p3_alpha, _, _ = render_explicit(base, states[condition], p3_overrides, background)
                    fixed_fn_masks[condition] = _mask_numpy(regions_by_condition[condition]["trusted_expansion"]) & (_mask_numpy(p3_alpha) < .5)

            for state_name in STATES:
                _, overrides = load_state_model(config, state_name, outfit, base)
                for condition in required_conditions:
                    with torch.no_grad():
                        rgb0, alpha0, info0, tensors = render_explicit(base, states[condition], overrides, background)
                    camera = build_mmlphuman_camera(states[condition]["camera"], states[condition]["camera"]["height"], states[condition]["camera"]["width"], device)
                    inputs = _projection_inputs(base, overrides, tensors, camera)
                    disabled = run_backend_projection_debug(means=inputs["posed_xyz"], covars=inputs["posed_covariance"], opacities=inputs["opacity"], w2c=inputs["w2c"], K=inputs["K"], width=inputs["width"], height=inputs["height"], options=options_off)
                    with torch.no_grad():
                        rgb_off, alpha_off, _, _ = render_explicit(base, states[condition], overrides, background)
                    backend = run_backend_projection_debug(means=inputs["posed_xyz"], covars=inputs["posed_covariance"], opacities=inputs["opacity"], w2c=inputs["w2c"], K=inputs["K"], width=inputs["width"], height=inputs["height"], options=options_on)
                    with torch.no_grad():
                        rgb_on, alpha_on, _, _ = render_explicit(base, states[condition], overrides, background)
                    off_rgb = float((rgb0 - rgb_off).abs().max()); off_alpha = float((alpha0 - alpha_off).abs().max())
                    on_rgb = float((rgb0 - rgb_on).abs().max()); on_alpha = float((alpha0 - alpha_on).abs().max())
                    binary_equal = bool(torch.equal(alpha0 >= .5, alpha_on >= .5))
                    debug_regression_rows.append({"state": state_name, "outfit": outfit, "condition": condition, "debug_disabled_returned_none": disabled is None, "off_rgb_max_abs": off_rgb, "off_alpha_max_abs": off_alpha, "off_rgb_bitwise": bool(torch.equal(rgb0, rgb_off)), "off_alpha_bitwise": bool(torch.equal(alpha0, alpha_off)), "on_rgb_max_abs": on_rgb, "on_alpha_max_abs": on_alpha, "on_rgb_bitwise": bool(torch.equal(rgb0, rgb_on)), "on_alpha_bitwise": bool(torch.equal(alpha0, alpha_on)), "on_binary_alpha_equal": binary_equal})
                    if off_rgb != 0 or off_alpha != 0 or on_rgb > 1e-7 or on_alpha > 1e-7 or not binary_equal:
                        raise RuntimeError(f"instrumentation changed render output: {state_name}/{outfit}/{condition}")

                    projection = independent_gaussian_projection_oracle_v1(**inputs, options=options_off)
                    agreement = projection_agreement(projection, backend)
                    projection_rows.append({"state": state_name, "outfit": outfit, "condition": condition, "view": VIEWS[condition], **agreement})
                    stages, codes, emitted = backend_first_rejections(projection, backend)
                    append_projection_parquet(parquet, state_name, outfit, condition, projection, backend, stages, codes, emitted)
                    counts = support_funnel_counts(projection["stage_masks_s0_s5"], emitted)
                    summary_rows.append({"state": state_name, "outfit": outfit, "condition": condition, "view": VIEWS[condition], **{f"N{index}": value for index, value in enumerate(counts)}, "backend_unexposed_count": int((codes == REJECTION_REASONS.index("BACKEND_UNEXPOSED_REJECTION")).sum())})
                    funnel_rows.append({"state": state_name, "outfit": outfit, "condition": condition, "view": VIEWS[condition], **{f"N{index}": value for index, value in enumerate(counts)}})
                    alpha_np = _mask_numpy(alpha0) if alpha0.dtype == torch.bool else alpha0[0].detach().cpu().numpy()
                    region_masks = _region_masks(outfit, samples[condition], regions_by_condition[condition], alpha_np)
                    global_stage_counts = support_funnel_counts(projection["stage_masks_s0_s5"], emitted)
                    for region_name, region_mask in region_masks.items():
                        tile_ids, eligible_ids, contributed_ids = _trace_region(projection, backend, inputs["opacity"], region_mask, options_on)
                        funnel_region_rows.append({"state": state_name, "outfit": outfit, "condition": condition, "view": VIEWS[condition], "region": region_name, "n0_n6_scope": "global_gaussians_before_region_pixel_query", **{f"N{index}": value for index, value in enumerate(global_stage_counts)}, "N7": len(tile_ids), "N8": len(eligible_ids), "N9": len(contributed_ids), "sampled_pixels": min(int(region_mask.sum()), 128)})

                    density_path = root / f"visualizations/density_{state_name}_{outfit}_{VIEWS[condition]}.png"
                    _save_density(density_path, projection, inputs["width"], inputs["height"])
                    density_panels.append((f"{state_name} {outfit} {VIEWS[condition]}", density_path))
                    if outfit == "O08" and condition in ("cond_000318", "cond_000347"):
                        fn_pixels = [(int(y), int(x)) for y, x in np.argwhere(fixed_fn_masks[condition])]
                        support = independent_pixel_support(projection, fn_pixels)
                        per_state_rows = []
                        pre_ids_union: set[int] = set()
                        for row in support:
                            trace = backend_pixel_pipeline(backend, inputs["opacity"], row["y"], row["x"], tile_size=options_on.tile_size)
                            pre = set(row.pop("pre_tile_contributor_indices")); tile = set(trace["tile_candidates"])
                            alpha_eligible = set(trace["alpha_eligible"]); contributed = set(trace["contributed"])
                            s7 = pre & tile; s8 = pre & alpha_eligible; s9 = pre & contributed
                            pre_ids_union.update(pre)
                            n_pre, n_emitted, n_active = len(pre), len(s7), len(s9)
                            missing = pre - tile
                            for index in missing:
                                code = int(codes[index])
                                reason = REJECTION_REASONS[code] if code >= 0 else "NOT_IN_TARGET_PIXEL_TILE"
                                rejection_counts[reason] += 1
                            record = {"state": state_name, "outfit": outfit, "condition": condition, "view": VIEWS[condition], **row, "N_pre": n_pre, "N_emitted": n_emitted, "N_active": n_active, "placement_ratio_vs_p1": None, "admission_retention": n_emitted / max(n_pre, 1), "pixel_retention": n_active / max(n_emitted, 1), "stage_status_counts": {"S7_DEPTH_SORTED": len(s7), "S8_PIXEL_ALPHA_ELIGIBLE": len(s8), "S9_PIXEL_CONTRIBUTED": len(s9)}, "pre_tile_contributor_indices": sorted(pre), "s7_depth_sorted_indices": sorted(s7), "s8_pixel_alpha_eligible_indices": sorted(s8), "s9_pixel_contributed_indices": sorted(s9), "missing_pre_tile_indices": sorted(missing), "backend_alpha": trace["alpha"]}
                            append_jsonl(root / "independent_projection/independent_pixel_support.jsonl", record)
                            append_jsonl(root / "contributor_pipeline/pixel_stage_pipeline.jsonl", record)
                            per_state_rows.append(record); pixel_rows.append(record)
                        aggregate_rows.append({"state": state_name, "condition": condition, "view": VIEWS[condition], "pixel_count": len(per_state_rows), "n_pre_mean": _summary([row["N_pre"] for row in per_state_rows])["mean"], "n_emitted_mean": _summary([row["N_emitted"] for row in per_state_rows])["mean"], "n_active_mean": _summary([row["N_active"] for row in per_state_rows])["mean"], "admission_retention_mean": _summary([row["admission_retention"] for row in per_state_rows])["mean"], "pixel_retention_mean": _summary([row["pixel_retention"] for row in per_state_rows])["mean"], **{f"n_pre_{key}": value for key, value in _summary([row["N_pre"] for row in per_state_rows]).items()}, **{f"n_emitted_{key}": value for key, value in _summary([row["N_emitted"] for row in per_state_rows]).items()}, **{f"n_active_{key}": value for key, value in _summary([row["N_active"] for row in per_state_rows]).items()}})
                        contexts[(state_name, condition)] = {"canonical": overrides.xyz.detach().cpu(), "posed": inputs["posed_xyz"].detach().cpu(), "screen": projection["projected_mean"].detach().cpu(), "opacity": inputs["opacity"].detach().cpu(), "projection": {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value for key, value in projection.items()}, "backend_codes": codes.detach().cpu(), "fn_pixels": fn_pixels, "pixel_rows": per_state_rows, "pre_ids_union": sorted(pre_ids_union), "regions": {key: value.copy() for key, value in region_masks.items()}}
                        if state_name == "P3" and condition == "cond_000318":
                            real_synthetic_context = {"means": inputs["posed_xyz"], "covars": inputs["posed_covariance"], "opacity": inputs["opacity"], "camera": camera}

            base_fingerprints_after[outfit] = _tensor_state_fingerprint({"xyz": base._xyz, "scaling": base._scaling, "rotation": base._rotation, "opacity": base._opacity, "sh0": base._sh0, "shN": base._shN}.items())
            if base_fingerprints_after[outfit] != base_fingerprints_before[outfit]:
                raise RuntimeError(f"frozen base changed: {outfit}")

        parquet.close(); parquet = None
        write_csv(root / "backend_projection/projection_agreement.csv", projection_rows)
        atomic_json(root / "backend_projection/projection_agreement.json", projection_rows)
        write_csv(root / "independent_projection/independent_projection_summary.csv", summary_rows)
        write_csv(root / "stage_comparison/support_funnel.csv", funnel_rows)
        write_csv(root / "stage_comparison/support_funnel_by_region.csv", funnel_region_rows)
        write_csv(root / "stage_comparison/pixel_support_aggregates.csv", aggregate_rows)
        write_csv(root / "contract/debug_numerical_regression.csv", debug_regression_rows)
        atomic_json(root / "backend_projection/rejection_reason_counts.json", dict(rejection_counts))

        # Fill exact placement ratios against P1 at the same fixed pixels.
        p1_lookup = {(row["condition"], row["y"], row["x"]): row["N_pre"] for row in pixel_rows if row["state"] == "P1"}
        ratio_rows = []
        for row in pixel_rows:
            ratio_rows.append({"state": row["state"], "condition": row["condition"], "view": row["view"], "y": row["y"], "x": row["x"], "N_pre": row["N_pre"], "N_emitted": row["N_emitted"], "N_active": row["N_active"], "placement_ratio": row["N_pre"] / max(p1_lookup[(row["condition"], row["y"], row["x"])], 1), "admission_retention": row["admission_retention"], "pixel_retention": row["pixel_retention"]})
        write_csv(root / "stage_comparison/placement_admission_pixel_metrics.csv", ratio_rows)

        # Deterministic same-base-index analysis of P1 supporters.
        displacement_rows: list[dict[str, Any]] = []
        displacement_summary: list[dict[str, Any]] = []
        for condition in ("cond_000318", "cond_000347"):
            source = contexts[("P1", condition)]
            support_ids = torch.tensor(source["pre_ids_union"], dtype=torch.long)
            for state in ("P2", "P3"):
                target = contexts[(state, condition)]
                matched = same_index_displacement(support_ids, source["posed"], target["posed"], source["screen"], target["screen"])
                xyz_delta = target["canonical"][support_ids] - source["canonical"][support_ids]
                posed_delta = matched["canonical_or_posed_displacement"]
                screen_delta = matched["screen_displacement"]
                expansion = target["regions"]["trusted_expansion"]
                cloud = target["regions"]["trailing_cloud"]
                source_in = _projection_center_mask(source["projection"], expansion).cpu()[support_ids]
                target_in = _projection_center_mask(target["projection"], expansion).cpu()[support_ids]
                target_cloud = _projection_center_mask(target["projection"], cloud).cpu()[support_ids]
                for local, gid in enumerate(support_ids.tolist()):
                    displacement_rows.append({"condition": condition, "view": VIEWS[condition], "target_state": state, "gaussian_index": gid, "canonical_dx": float(xyz_delta[local, 0]), "canonical_dy": float(xyz_delta[local, 1]), "canonical_dz": float(xyz_delta[local, 2]), "canonical_displacement": float(torch.linalg.vector_norm(xyz_delta[local])), "posed_displacement": float(torch.linalg.vector_norm(posed_delta[local])), "screen_dx": float(screen_delta[local, 0]), "screen_dy": float(screen_delta[local, 1]), "screen_displacement": float(torch.linalg.vector_norm(screen_delta[local])), "source_in_target_expansion": bool(source_in[local]), "target_in_target_expansion": bool(target_in[local]), "moved_out_of_target_expansion": bool(source_in[local] and not target_in[local]), "moved_to_trailing_cloud": bool(target_cloud[local]), "opacity_change": float(target["opacity"][gid] - source["opacity"][gid])})
                selected = [row for row in displacement_rows if row["condition"] == condition and row["target_state"] == state]
                displacement_summary.append({"condition": condition, "view": VIEWS[condition], "target_state": state, "matched_count": len(selected), "screen_displacement_mean": float(np.mean([row["screen_displacement"] for row in selected])) if selected else 0, "moved_out_fraction": float(np.mean([row["moved_out_of_target_expansion"] for row in selected])) if selected else 0, "trailing_cloud_fraction": float(np.mean([row["moved_to_trailing_cloud"] for row in selected])) if selected else 0, "opacity_change_mean": float(np.mean([row["opacity_change"] for row in selected])) if selected else 0})
        write_csv(root / "stage_comparison/same_index_displacement.csv", displacement_rows)
        atomic_json(root / "stage_comparison/same_index_displacement_summary.json", displacement_summary)
        figure, axes = plt.subplots(2, 2, figsize=(12, 10))
        for axis, (condition, state) in zip(axes.ravel(), (("cond_000318", "P2"), ("cond_000318", "P3"), ("cond_000347", "P2"), ("cond_000347", "P3"))):
            rows = [row for row in displacement_rows if row["condition"] == condition and row["target_state"] == state]
            sampled = rows[::max(1, len(rows) // 2000)]
            axis.quiver([contexts[("P1", condition)]["screen"][row["gaussian_index"], 0] for row in sampled], [contexts[("P1", condition)]["screen"][row["gaussian_index"], 1] for row in sampled], [row["screen_dx"] for row in sampled], [row["screen_dy"] for row in sampled], angles="xy", scale_units="xy", scale=1, width=.002)
            axis.set_title(f"{state} O08 {VIEWS[condition]} same-index flow"); axis.invert_yaxis(); axis.set_aspect("equal")
        figure.tight_layout(); figure.savefig(root / "visualizations/placement_flow_maps.png", dpi=160); plt.close(figure)

        # Static rejected-Gaussian counterfactual on P3 fixed FN and normal background pixels.
        counterfactual_rows = []
        for condition in ("cond_000318", "cond_000347"):
            context = contexts[("P3", condition)]
            rejected_ids = sorted({index for row in context["pixel_rows"] for index in row["missing_pre_tile_indices"]})
            rejected = torch.tensor(rejected_ids, dtype=torch.long)
            fn_pixels = context["fn_pixels"]
            background_pixels = deterministic_mask_indices(context["regions"]["normal_background"], int(config["diagnostic_sampling"]["normal_background_pixels_per_view"]))
            cf_fn = counterfactual_alpha(context["projection"], rejected, fn_pixels).cpu().numpy() if rejected.numel() else np.zeros(len(fn_pixels))
            cf_bg = counterfactual_alpha(context["projection"], rejected, background_pixels).cpu().numpy() if rejected.numel() else np.zeros(len(background_pixels))
            baseline_fn = np.asarray([row["backend_alpha"] for row in context["pixel_rows"]])
            combined_fn = 1 - (1 - baseline_fn) * (1 - cf_fn)
            recovered = int((combined_fn >= .5).sum())
            introduced_fp = int((cf_bg >= .5).sum())
            counterfactual_rows.append({"condition": condition, "view": VIEWS[condition], "rejected_gaussian_count": len(rejected_ids), "trusted_fn_count": len(fn_pixels), "trusted_fn_recovered": recovered, "trusted_fn_recovery_fraction": recovered / max(len(fn_pixels), 1), "normal_background_sample_count": len(background_pixels), "new_background_fp": introduced_fp, "cloud_created": introduced_fp > 0})
        atomic_json(root / "pre_tile_admission/pre_tile_rejection_counterfactual.json", counterfactual_rows)
        write_csv(root / "pre_tile_admission/pre_tile_rejection_counterfactual.csv", counterfactual_rows)

        synthetic_rows = _synthetic_scenes(root, device, options_on, real_synthetic_context)
        synthetic_pass = all(bool(row.get("projection_pass", True)) for row in synthetic_rows)
        adjudication = _case_adjudication(config, projection_rows, aggregate_rows, rejection_counts)
        adjudication.update({"status": "AWAITING_VISUAL_INSPECTION", "optimizer_steps": 0, "base_fingerprints_before": base_fingerprints_before, "base_fingerprints_after": base_fingerprints_after, "debug_off_bitwise": all(row["off_rgb_bitwise"] and row["off_alpha_bitwise"] for row in debug_regression_rows), "debug_on_max_rgb": max(row["on_rgb_max_abs"] for row in debug_regression_rows), "debug_on_max_alpha": max(row["on_alpha_max_abs"] for row in debug_regression_rows), "synthetic_scenes_pass": synthetic_pass, "source_evidence_hashes_unchanged": _source_evidence_hashes(config) == source_hashes_before, "frozen_refs_unchanged": {branch: resolve_ref(branch) for branch in config["frozen_branches"]} == frozen_before, "elapsed_seconds": time.perf_counter() - started})
        atomic_json(root / "final_adjudication/PRELIMINARY_STATUS.json", adjudication)
        atomic_text(root / "stage_comparison/SUPPORT_FUNNEL_REPORT.md", "\n".join(["# P1/P2/P3 support funnel", "", "The CSV files preserve the preregistered N0-N9 definitions. N7-N9 are unions over deterministic pixels in each named evaluation region; target masks are applied only after projection.", "", *[f"- {row['state']} {row['view']}: N_pre={row['n_pre_mean']:.3f}, N_emitted={row['n_emitted_mean']:.3f}, N_active={row['n_active_mean']:.3f}, admission={row['admission_retention_mean']:.6f}, pixel={row['pixel_retention_mean']:.6f}" for row in aggregate_rows]]))
        atomic_text(root / "stage_comparison/PLACEMENT_DIAGNOSIS.md", "\n".join(["# Same-index placement diagnosis", "", "Matching is exactly by frozen base Gaussian index; no nearest-neighbour remapping is used.", "", *[f"- {row['target_state']} {row['view']}: mean screen displacement={row['screen_displacement_mean']:.4f}px, moved-out fraction={row['moved_out_fraction']:.6f}, trailing-cloud fraction={row['trailing_cloud_fraction']:.6f}, mean opacity change={row['opacity_change_mean']:.6f}" for row in displacement_summary]]))
        _make_contact_sheet(root / "visualizations/projection_density_contact_sheet.png", density_panels)
        _make_contact_sheet(root / "visualizations/final_visual_contact_sheet.png", [("projection density", root / "visualizations/projection_density_contact_sheet.png"), ("same-index flow", root / "visualizations/placement_flow_maps.png")], columns=2)
        atomic_json(root / "visualizations/VISUAL_INSPECTION_REQUIRED.json", {"status": "AWAITING_ACTUAL_IMAGE_OPEN", "required": ["projection_density_contact_sheet.png", "placement_flow_maps.png", "final_visual_contact_sheet.png"], "preliminary_case": adjudication["preliminary_case"]})
        atomic_json(status_path, {"status": "STATIC_AUDIT_COMPLETE_AWAITING_VISUAL", "stage": "visual_acceptance", "optimizer_steps": 0, "preliminary_case": adjudication["preliminary_case"], "completed_at": now()})
        print(json.dumps({"status": "STATIC_AUDIT_COMPLETE_AWAITING_VISUAL", "output": str(root), **adjudication}, indent=2))
    except Exception as error:
        if parquet is not None:
            parquet.close()
        atomic_json(status_path, {"status": "TOOL_FAILURE", "failure_stage": stage, "optimizer_steps": 0, "exception_type": type(error).__name__, "exception": str(error), "traceback": traceback.format_exc(), "failed_at": now()})
        raise


def finalize_visual(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    root = args.output.resolve()
    status = json.loads((root / "RUN_STATUS.json").read_text(encoding="utf-8"))
    preliminary = json.loads((root / "final_adjudication/PRELIMINARY_STATUS.json").read_text(encoding="utf-8"))
    if status.get("status") != "STATIC_AUDIT_COMPLETE_AWAITING_VISUAL":
        raise RuntimeError("audit is not awaiting visual inspection")
    if args.final_case != preliminary["preliminary_case"]:
        raise ValueError("visual finalization cannot override the preregistered numerical case")
    observations = json.loads(args.visual_observations)
    if not observations.get("images_actually_opened") or observations.get("inspection_method") != "view_image_original_resolution_and_contact_sheets":
        raise ValueError("actual image-open evidence is required")
    final = {**preliminary, "status": "PASS", "audit_status": "COMPLETE", "final_case": args.final_case, "visual_acceptance": observations, "renderer_change_authorized": args.final_case == "IA", "coverage_placement_objective_change_authorized": args.final_case == "IP", "seven_outfit_rerun_authorized": False, "target_generation_authorized": False, "formal_training_authorized": False, "optimizer_steps": 0, "completed_at": now()}
    atomic_json(root / "visualizations/visual_acceptance.json", observations)
    atomic_text(root / "visualizations/VISUAL_ACCEPTANCE.md", observations["markdown"])
    atomic_json(root / "final_adjudication/INSTRUMENTED_PROJECTION_ADMISSION_FINAL_STATUS.json", final)
    atomic_text(root / "final_adjudication/GATE_ACCEPTANCE.md", "\n".join(["# Instrumented projection/admission final adjudication", "", "- status: **PASS (diagnostic audit complete)**", f"- final case: **{args.final_case}**", f"- root cause: `{final['root_cause']}`", f"- next unique task: `{final['next_unique_task']}`", "- optimizer steps: `0`", "- installed gsplat, production renderer, losses, bounds, checkpoints and historical outputs were not modified."]))
    atomic_json(root / "RUN_STATUS.json", {"status": "PASS", "final_case": args.final_case, "optimizer_steps": 0, "completed_at": now()})
    _artifact_manifest(root, phase="FINAL")
    print(json.dumps(final, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-step staged Gaussian projection/admission audit")
    parser.add_argument("command", choices=("audit", "postprocess", "finalize-visual"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--final-case", choices=("IP", "IA", "IC", "IM", "IU"))
    parser.add_argument("--visual-observations")
    args = parser.parse_args()
    if args.command == "finalize-visual" and (args.final_case is None or args.visual_observations is None):
        parser.error("finalize-visual requires --final-case and --visual-observations")
    return args


def main() -> None:
    args = parse_args(); config = load_config(args.config)
    if args.command == "audit":
        run_audit(args, config)
    elif args.command == "postprocess":
        postprocess_existing(args, config)
    else:
        finalize_visual(args, config)


if __name__ == "__main__":
    main()
