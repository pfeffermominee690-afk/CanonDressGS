from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import torch
except ModuleNotFoundError:  # Static governance tests also run on CPU-only hosts.
    torch = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


TASK_ID = "AAAI27-LOO-BASIS-RENDERER-PARITY-REPAIR-001"
SOURCE_BRANCH = "research/leave-one-garment-out-basis-adaptation-cache-repaired-20260724"
SOURCE_HEAD = "6ddd3a637b322a48e965bff8bb11a16213264c37"
EXECUTION_HEAD = "a868df3c3811483c5dd07456d8cabfdc3577d5f4"
RESULT_HEAD = "8436a6187da02ca3c21ef546f0940208a02fdac3"
REPAIR_BRANCH = "research/loo-basis-renderer-parity-closure-repair-20260724"
OUTPUT_NAME = "LOO-BASIS-RENDERER-PARITY-REPAIR-001"
ORIGINAL_OUTPUT_NAME = "LOO-BASIS-ADAPTATION-001"
ATTEMPT_NAME = "attempt_001"
DIAGNOSTIC_LABEL = "DIAGNOSTIC_ONLY_NOT_SCIENTIFIC_EVALUATION"
OUTFITS = ("O01", "O02", "O03", "O04", "O08")
CONDITION = "cond_000000"
RENDER_GATE = 1e-5
EXPECTED_ORIGINAL_BASIS_FINGERPRINT = (
    "a7a16acf8c0920c5d4bba80c76be87f343e708cba7a8836d4bba3fee176410b9"
)
EXPECTED_ORIGINAL_BASIS_SHA256 = (
    "71144cafcdfd070942dc0a808b65216539d0320c3a801a1cde7fb0b75ecd40c6"
)
EXPECTED_ORIGINAL_NORMALIZATION_SHA256 = (
    "91a4121a58fcf8b839e60284290022b9b2963e4b8a26c339f0dd5e7dab7f1d19"
)
EXPECTED_ORIGINAL_MAX_ALPHA = 0.003845691680908203
EXPECTED_ORIGINAL_SINGULAR_VALUES = (
    879.3351213828762,
    806.7408088938806,
    697.1994598902255,
    5.195989704080849e-05,
)
CHANNEL_VARIANTS = {
    "xyz_only": ("delta_xyz",),
    "rotation_only": ("delta_rotvec",),
    "scale_only": ("delta_log_scaling",),
    "opacity_only": ("delta_opacity_logit",),
    "sh0_only": ("delta_sh0",),
    "shN_only": ("delta_shN",),
    "geometry_combined": ("delta_xyz", "delta_log_scaling", "delta_rotvec"),
    "appearance_combined": ("delta_opacity_logit", "delta_sh0", "delta_shN"),
    "all_channels": (
        "delta_xyz",
        "delta_log_scaling",
        "delta_rotvec",
        "delta_opacity_logit",
        "delta_sh0",
        "delta_shN",
    ),
}


def git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def write_json(path: Path, value: Any, *, replace: bool = False) -> None:
    payload = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"refusing to replace non-identical artifact: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def output_paths(asset_root: Path) -> tuple[Path, Path]:
    original = asset_root / ORIGINAL_OUTPUT_NAME / ATTEMPT_NAME
    diagnostic = asset_root / OUTPUT_NAME / ATTEMPT_NAME
    return original, diagnostic


def tree_manifest(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    rows = []
    for path in sorted(
        (item for item in root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        rows.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    aggregate_rows = [
        {"path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"]}
        for row in rows
    ]
    file_content_bytes = sum(row["bytes"] for row in rows)
    du = subprocess.run(
        ["du", "-sb", str(root)], capture_output=True, text=True,
    )
    tree_apparent_bytes = (
        int(du.stdout.split()[0]) if du.returncode == 0 else file_content_bytes
    )
    return {
        "schema_version": "canondressgs.paper.immutable_tree_manifest.v1",
        "root": str(root),
        "file_count": len(rows),
        "total_bytes": tree_apparent_bytes,
        "file_content_bytes": file_content_bytes,
        "byte_count_definition": (
            "total_bytes is GNU du -sb tree apparent bytes; file_content_bytes is the "
            "sum of the ordered regular-file sizes"
        ),
        "aggregate_sha256": canonical_sha(aggregate_rows),
        "aggregate_definition": "sha256(canonical JSON of ordered path/bytes/sha256 rows)",
        "files": rows,
    }


def original_immutability(asset_root: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    original, _ = output_paths(asset_root)
    actual = tree_manifest(original)
    stable = all(
        actual[name] == expected[name]
        for name in (
            "file_count", "total_bytes", "file_content_bytes", "aggregate_sha256", "files",
        )
    )
    return {
        "status": "PASS" if stable else "FAIL",
        "mutation_count": 0 if stable else 1,
        "expected": {
            name: expected[name]
            for name in (
                "file_count", "total_bytes", "file_content_bytes", "aggregate_sha256",
            )
        },
        "actual": {
            name: actual[name]
            for name in (
                "file_count", "total_bytes", "file_content_bytes", "aggregate_sha256",
            )
        },
    }


def preflight(asset_root: Path) -> dict[str, Any]:
    original, diagnostic = output_paths(asset_root)
    if (asset_root / ORIGINAL_OUTPUT_NAME / "attempt_002").exists():
        raise RuntimeError("forbidden LOO attempt_002 exists")
    if not original.is_dir():
        raise FileNotFoundError(original)
    if git("branch", "--show-current") != REPAIR_BRANCH:
        raise RuntimeError("preflight requires the repair branch")
    if subprocess.call(
        ["git", "merge-base", "--is-ancestor", SOURCE_HEAD, "HEAD"], cwd=ROOT,
    ) != 0 or git("status", "--short"):
        raise RuntimeError("preflight requires a clean descendant of the source reporting HEAD")
    diagnostic.mkdir(parents=True, exist_ok=True)
    manifest = tree_manifest(original)
    if manifest["file_count"] != 66 or manifest["total_bytes"] != 71_076_354:
        raise RuntimeError("original attempt count/bytes do not match the frozen contract")
    write_json(diagnostic / "00_preflight/original_attempt_manifest.json", manifest)
    basis = original / "02_basis_construction/O01/loo_basis.pt"
    normalization = original / "02_basis_construction/O01/coefficient_normalization.json"
    if sha256(basis) != EXPECTED_ORIGINAL_BASIS_SHA256:
        raise RuntimeError("original basis artifact SHA mismatch")
    if sha256(normalization) != EXPECTED_ORIGINAL_NORMALIZATION_SHA256:
        raise RuntimeError("original normalization SHA mismatch")
    result = {
        "schema_version": "canondressgs.paper.loo_basis_renderer_parity_preflight.v1",
        "task_id": TASK_ID,
        "status": "PASS",
        "diagnostic_label": DIAGNOSTIC_LABEL,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "execution_head": EXECUTION_HEAD,
        "result_head": RESULT_HEAD,
        "repair_branch": REPAIR_BRANCH,
        "head": git("rev-parse", "HEAD"),
        "original_attempt": str(original),
        "diagnostic_attempt": str(diagnostic),
        "original_attempt_manifest_sha256": sha256(
            diagnostic / "00_preflight/original_attempt_manifest.json"
        ),
        "original_file_count": manifest["file_count"],
        "original_total_bytes": manifest["total_bytes"],
        "original_file_content_bytes": manifest["file_content_bytes"],
        "original_aggregate_sha256": manifest["aggregate_sha256"],
        "original_basis_sha256": EXPECTED_ORIGINAL_BASIS_SHA256,
        "original_normalization_sha256": EXPECTED_ORIGINAL_NORMALIZATION_SHA256,
        "attempt_002_absent": True,
        "optimizer_creations": 0,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "formal_metrics": 0,
        "PAPER_FINAL": False,
    }
    write_json(diagnostic / "00_preflight/preflight.json", result)
    return result


def load_manifest(asset_root: Path) -> dict[str, Any]:
    _, diagnostic = output_paths(asset_root)
    path = diagnostic / "00_preflight/original_attempt_manifest.json"
    if not path.is_file():
        raise RuntimeError("preflight manifest must exist before any diagnostic render")
    return json.loads(path.read_text(encoding="utf-8"))


def require_diagnostic_runtime(asset_root: Path) -> tuple[Path, dict[str, Any]]:
    _, diagnostic = output_paths(asset_root)
    manifest = load_manifest(asset_root)
    immutable = original_immutability(asset_root, manifest)
    if immutable["status"] != "PASS":
        raise RuntimeError("original attempt mutated after preflight")
    if (asset_root / ORIGINAL_OUTPUT_NAME / "attempt_002").exists():
        raise RuntimeError("forbidden LOO attempt_002 exists")
    if git("branch", "--show-current") != REPAIR_BRANCH:
        raise RuntimeError("diagnosis requires the repair branch")
    if subprocess.call(
        ["git", "merge-base", "--is-ancestor", SOURCE_HEAD, "HEAD"], cwd=ROOT,
    ) != 0:
        raise RuntimeError("repair HEAD is not descended from the frozen source HEAD")
    if git("status", "--short"):
        raise RuntimeError("diagnosis requires a clean worktree")
    return diagnostic, immutable


def record_known_interruption(asset_root: Path) -> dict[str, Any]:
    diagnostic, immutable = require_diagnostic_runtime(asset_root)
    result = {
        "schema_version": "canondressgs.paper.loo_diagnostic_interruption_ledger.v1",
        "task_id": TASK_ID,
        "diagnostic_label": DIAGNOSTIC_LABEL,
        "rows": [
            {
                "run_id": "diagnose_run_001",
                "status": "INTERRUPTED_IMPLEMENTATION_ERROR",
                "last_completed_stage": "residual_channel_parity",
                "error": "incorrect CHANNELS import in numerical unflatten helper",
                "diagnostic_render_calls": 68,
                "optimizer_creations": 0,
                "optimizer_steps": 0,
                "checkpoint_writes": 0,
                "formal_metrics": 0,
            },
            {
                "run_id": "validate_repair_run_001",
                "status": "FAILED_DIAGNOSTIC_GATE_IMPLEMENTATION",
                "last_completed_stage": "all_split_repaired_validation",
                "error": "post-projection means2d info was incorrectly included in the renderer-input gate",
                "diagnostic_render_calls": 80,
                "optimizer_creations": 0,
                "optimizer_steps": 0,
                "checkpoint_writes": 0,
                "formal_metrics": 0,
            },
        ],
        "diagnostic_render_calls": 148,
        "original_attempt_immutability": immutable,
        "attempt_002_absent": not (
            asset_root / ORIGINAL_OUTPUT_NAME / "attempt_002"
        ).exists(),
        "PAPER_FINAL": False,
    }
    write_json(
        diagnostic / "00_preflight/diagnostic_interruption_ledger.json",
        result,
        replace=True,
    )
    return result


def prior_interrupted_render_calls(diagnostic: Path) -> int:
    path = diagnostic / "00_preflight/diagnostic_interruption_ledger.json"
    if not path.is_file():
        return 0
    return int(json.loads(path.read_text(encoding="utf-8"))["diagnostic_render_calls"])


def runtime_context(diagnostic: Path) -> dict[str, Any]:
    from tools.paper import formal_runtime

    old_branch = formal_runtime.FORMAL_BRANCH
    old_head = formal_runtime.FORMAL_SOURCE_HEAD
    formal_runtime.FORMAL_BRANCH = REPAIR_BRANCH
    formal_runtime.FORMAL_SOURCE_HEAD = SOURCE_HEAD
    try:
        return formal_runtime._legacy_context(diagnostic)
    finally:
        formal_runtime.FORMAL_BRANCH = old_branch
        formal_runtime.FORMAL_SOURCE_HEAD = old_head


def load_teachers(context: Mapping[str, Any]) -> dict[str, Any]:
    from tools.paper.formal_batch_runtime import load_frozen_teacher_residuals

    return load_frozen_teacher_residuals(context, OUTFITS)


def split_rows() -> list[dict[str, Any]]:
    path = ROOT / "paper_protocol/reviewer_risk/loo_basis_manifests.json"
    return json.loads(path.read_text(encoding="utf-8"))["splits"]


def flatten_normalized(
    residual: Any, bounds: Mapping[str, float], *, dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, dict[str, tuple[int, ...]]]:
    from scene.explicit_gaussian_residual_basis import CHANNEL_TO_BOUND
    from scene.gaussian_clothing_residuals import CHANNELS

    values = {}
    shapes = {}
    for name in CHANNELS:
        value = getattr(residual, name)
        if dtype is not None:
            value = value.to(dtype=dtype)
        value = value / float(bounds[CHANNEL_TO_BOUND[name]])
        values[name] = value
        shapes[name] = tuple(value.shape)
    return torch.cat([values[name].reshape(-1) for name in CHANNELS]), shapes


def unflatten(flat: torch.Tensor, shapes: Mapping[str, tuple[int, ...]]) -> dict[str, torch.Tensor]:
    from scene.gaussian_clothing_residuals import CHANNELS

    result = {}
    start = 0
    for name in CHANNELS:
        count = math.prod(shapes[name])
        result[name] = flat[start:start + count].reshape(shapes[name])
        start += count
    if start != flat.numel():
        raise ValueError("flat field length mismatch")
    return result


def fix_component_signs(basis: torch.Tensor) -> torch.Tensor:
    result = basis.clone()
    for index in range(result.shape[0]):
        pivot = int(result[index].abs().argmax())
        if result[index, pivot] < 0:
            result[index].neg_()
    return result


def coefficient_solve(
    centered: torch.Tensor, basis: torch.Tensor, solver: str,
) -> torch.Tensor:
    if solver == "orthonormal_projection":
        return centered @ basis.t()
    if solver == "lstsq":
        return torch.linalg.lstsq(basis.t(), centered.t()).solution.t()
    if solver == "pinv":
        return centered @ torch.linalg.pinv(basis)
    if solver == "svd_coordinates":
        # The signed V rows are orthonormal, so SVD coordinates equal B^T projection.
        return centered @ basis.t()
    raise ValueError(f"unknown coefficient solver: {solver}")


def make_decomposition(
    teachers: Mapping[str, Any],
    bounds: Mapping[str, float],
    order: Sequence[str],
    *,
    dtype: torch.dtype,
    zero_sum: bool,
    solver: str = "orthonormal_projection",
) -> tuple[Any, dict[str, Any]]:
    from scene.explicit_gaussian_residual_basis import ExplicitGaussianResidualBasis

    rows = []
    shapes = None
    for name in order:
        row, current_shapes = flatten_normalized(teachers[name], bounds, dtype=dtype)
        rows.append(row)
        if shapes is None:
            shapes = current_shapes
        elif shapes != current_shapes:
            raise ValueError("teacher shapes differ")
    matrix = torch.stack(rows)
    mean = matrix.mean(0)
    centered = matrix - mean
    if zero_sum:
        centered = centered.clone()
        centered[-1] = -centered[:-1].sum(0)
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    basis_flat = fix_component_signs(vh[:3])
    coefficients = coefficient_solve(matrix - mean, basis_flat, solver)
    assert shapes is not None
    mean_fields = unflatten(mean, shapes)
    basis_fields = {name: [] for name in mean_fields}
    for component in basis_flat:
        fields = unflatten(component, shapes)
        for name, value in fields.items():
            basis_fields[name].append(value)
    basis = ExplicitGaussianResidualBasis(
        mean_fields,
        {name: torch.stack(values) for name, values in basis_fields.items()},
        bounds,
    )
    decomposition = {
        "basis": basis,
        "teacher_coefficients": {
            name: coefficients[index] for index, name in enumerate(order)
        },
    }
    centered_sum = centered.sum(0)
    metadata = {
        "dtype": str(dtype),
        "zero_sum": zero_sum,
        "solver": solver,
        "singular_values": [float(value) for value in singular_values.detach().cpu()],
        "fourth_singular_value": float(singular_values[3]),
        "relative_fourth_singular_value": float(singular_values[3] / singular_values[0]),
        "centered_sum_l2": float(torch.linalg.vector_norm(centered_sum.double())),
        "centered_sum_max_abs": float(centered_sum.abs().max()),
        "basis_fingerprint": basis.fingerprint(),
    }
    return decomposition, metadata


def reconstruct(decomposition: Mapping[str, Any], name: str) -> Any:
    return decomposition["basis"](
        decomposition["teacher_coefficients"][name], chunk_size=16384,
    )


def residual_error(first: Any, second: Any) -> dict[str, Any]:
    from scene.gaussian_clothing_residuals import CHANNELS

    rows = {}
    squared = 0.0
    count = 0
    for name in CHANNELS:
        a = getattr(first, name).double()
        b = getattr(second, name).double()
        difference = a - b
        squared += float(difference.square().sum())
        count += difference.numel()
        rows[name] = {
            "shape": list(a.shape),
            "teacher_dtype": str(getattr(first, name).dtype),
            "reconstructed_dtype": str(getattr(second, name).dtype),
            "max_abs": float(difference.abs().max()),
            "mae": float(difference.abs().mean()),
            "rmse": float(torch.sqrt(difference.square().mean())),
            "nonzero_count": int(torch.count_nonzero(difference)),
            "affected_gaussian_count": int(
                torch.count_nonzero(difference.reshape(difference.shape[0], -1).abs().amax(1))
            ),
            "teacher_sha256": tensor_sha(getattr(first, name)),
            "reconstructed_sha256": tensor_sha(getattr(second, name)),
        }
    return {
        "channels": rows,
        "aggregate_rmse": math.sqrt(squared / max(count, 1)),
        "max_abs": max(row["max_abs"] for row in rows.values()),
        "bitwise_equal_after_teacher_dtype_cast": all(
            torch.equal(
                getattr(first, name),
                getattr(second, name).to(getattr(first, name)),
            )
            for name in CHANNELS
        ),
    }


def image_error(first: torch.Tensor, second: torch.Tensor) -> dict[str, Any]:
    difference = first.double() - second.double()
    return {
        "max_abs": float(difference.abs().max()),
        "mae": float(difference.abs().mean()),
        "rmse": float(torch.sqrt(difference.square().mean())),
        "nonzero_count": int(torch.count_nonzero(difference)),
        "first_sha256": tensor_sha(first),
        "second_sha256": tensor_sha(second),
        "bitwise_equal": torch.equal(first, second),
    }


def tensor_error(first: torch.Tensor, second: torch.Tensor) -> dict[str, Any]:
    difference = first.detach().double() - second.detach().double()
    if difference.numel() == 0:
        maximum = mae = rmse = 0.0
    else:
        maximum = float(difference.abs().max())
        mae = float(difference.abs().mean())
        rmse = float(torch.sqrt(difference.square().mean()))
    return {
        "shape": list(first.shape),
        "first_dtype": str(first.dtype),
        "second_dtype": str(second.dtype),
        "device": str(first.device),
        "first_min": float(first.detach().min()) if first.numel() else None,
        "first_max": float(first.detach().max()) if first.numel() else None,
        "first_mean": float(first.detach().double().mean()) if first.numel() else None,
        "first_std": float(first.detach().double().std(unbiased=False)) if first.numel() else None,
        "max_abs": maximum,
        "mae": mae,
        "rmse": rmse,
        "nonzero_count": int(torch.count_nonzero(difference)),
        "affected_gaussian_count": int(
            torch.count_nonzero(difference.reshape(difference.shape[0], -1).abs().amax(1))
        ) if difference.ndim >= 1 and difference.shape[0] else 0,
        "first_sha256": tensor_sha(first),
        "second_sha256": tensor_sha(second),
    }


def render_formal(
    context: Mapping[str, Any], sample: Mapping[str, Any], residual: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    from tools.run_residual_field_parameterization import render_prediction

    base = context["base"]
    cast = residual.to(device=base._xyz.device, dtype=base._xyz.dtype)
    return render_prediction(base, sample, cast, context["background"])


def render_formal_repaired(
    context: Mapping[str, Any], sample: Mapping[str, Any], residual: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    from tools.run_residual_field_parameterization import render_prediction

    return render_prediction(
        context["base"], sample, residual, context["background"],
    )


def capture_render(
    context: Mapping[str, Any], sample: Mapping[str, Any], residual: Any,
    *,
    explicit_entry_cast: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
    from gsplat import rasterization
    from scene.gaussian_clothing_residuals import compose_canonical_gaussian_overrides
    from tools import run_alpha_raster_audit as audit

    base = context["base"]
    renderer_residual = (
        residual.to(device=base._xyz.device, dtype=base._xyz.dtype)
        if explicit_entry_cast else residual
    )
    overrides = compose_canonical_gaussian_overrides(base, renderer_residual)
    state = audit.target_free_state(sample, base._xyz.device)
    camera = audit.build_mmlphuman_camera(
        state["camera"], state["camera"]["height"], state["camera"]["width"],
        base._xyz.device,
    )
    with audit.mmlphuman_state_transaction(base, state["pose"], state["Rh"], state["Th"]):
        means = base.compute_xyz(overrides.as_dict())
        covariances = base.get_covariance(canonical_overrides=overrides.as_dict())
        opacities = base.compute_opacity(overrides.as_dict()).reshape(-1)
        camera_position = torch.linalg.inv(camera["w2c"])[:3, 3]
        colors = base.get_color(camera_position, overrides.as_dict())
    image, alpha, info = rasterization(
        means=means,
        quats=None,
        scales=None,
        opacities=opacities,
        colors=colors,
        viewmats=camera["w2c"][None],
        Ks=camera["K"][None],
        width=int(camera["width"]),
        height=int(camera["height"]),
        packed=False,
        near_plane=0.1,
        backgrounds=context["background"][None],
        covars=covariances,
        rasterize_mode="classic",
    )
    rgb = audit.chw(image[0], 3)
    opacity = audit.chw(alpha[0], 1)
    tensors = {
        "L3/canonical_xyz": overrides.xyz,
        "L3/canonical_log_scaling": overrides.scaling,
        "L3/canonical_rotation": overrides.rotation,
        "L3/canonical_opacity_logit": overrides.opacity,
        "L3/canonical_sh0": overrides.sh0,
        "L3/canonical_shN": overrides.shN,
        "L4/activated_scale": torch.exp(overrides.scaling),
        "L4/normalized_rotation": overrides.rotation,
        "L4/effective_opacity": torch.sigmoid(overrides.opacity),
        "L5/canonical_xyz": overrides.xyz,
        "L5/canonical_scale": torch.exp(overrides.scaling),
        "L5/canonical_rotation": overrides.rotation,
        "L5/canonical_opacity": torch.sigmoid(overrides.opacity),
        "L5/canonical_sh0": overrides.sh0,
        "L5/canonical_shN": overrides.shN,
        "L6/posed_xyz": means,
        "L6/posed_covariance": covariances,
        "L6/effective_opacity": opacities,
        "L6/posed_color": colors,
        "L7/raster_means": means,
        "L7/raster_covariance": covariances,
        "L7/raster_opacity": opacities,
        "L7/raster_color": colors,
    }
    summary = {
        "active_gaussian_count": int(torch.count_nonzero(opacities > 0)),
        "visible_gaussian_count": int(torch.count_nonzero(info["radii"][0].amin(-1) > 0)),
        "tiles_per_gaussian_sum": int(info["tiles_per_gauss"][0].sum()),
        "tiles_per_gaussian_max": int(info["tiles_per_gauss"][0].max()),
        "camera_sha256": canonical_sha({
            "w2c": tensor_sha(camera["w2c"]),
            "K": tensor_sha(camera["K"]),
            "width": int(camera["width"]),
            "height": int(camera["height"]),
        }),
        "pose_sha256": canonical_sha({
            name: tensor_sha(state[name]) for name in ("pose", "Rh", "Th")
        }),
        "background_sha256": tensor_sha(context["background"]),
    }
    return rgb, opacity, tensors, summary


def hybrid_residual(teacher: Any, rebuilt: Any, selected: Iterable[str]) -> Any:
    from scene.gaussian_clothing_residuals import CHANNELS, GaussianClothingResiduals

    chosen = set(selected)
    return GaussianClothingResiduals(**{
        name: getattr(rebuilt if name in chosen else teacher, name)
        for name in CHANNELS
    })


def anchor_diagnosis(
    teachers: Mapping[str, Any], bounds: Mapping[str, float], order: Sequence[str],
    svd_basis: torch.Tensor,
) -> dict[str, Any]:
    rows = [flatten_normalized(teachers[name], bounds, dtype=torch.float64)[0] for name in order]
    matrix = torch.stack(rows)
    anchor = matrix[0]
    differences = matrix[1:] - anchor
    _, singular_values, vh = torch.linalg.svd(differences, full_matrices=False)
    anchor_basis = vh[:3]
    coefficients = torch.linalg.lstsq(anchor_basis.t(), (matrix - anchor).t()).solution.t()
    reconstructed = anchor + coefficients @ anchor_basis
    cross = fix_component_signs(svd_basis) @ fix_component_signs(anchor_basis).t()
    cosines = torch.linalg.svdvals(cross).clamp(-1, 1)
    angles = torch.rad2deg(torch.acos(cosines))
    difference = reconstructed - matrix
    return {
        "dtype": "torch.float64",
        "anchor_garment": order[0],
        "difference_singular_values": [float(value) for value in singular_values],
        "rank": int(torch.linalg.matrix_rank(differences)),
        "principal_angles_degrees": [float(value) for value in angles],
        "principal_angle_max_degrees": float(angles.max()),
        "reconstruction_max_abs": float(difference.abs().max()),
        "reconstruction_rmse": float(torch.sqrt(difference.square().mean())),
    }


def diagnose(asset_root: Path) -> dict[str, Any]:
    from scene.explicit_gaussian_residual_basis import build_svd_basis
    from scene.gaussian_clothing_residuals import CHANNELS, apply_protected_full_residual_guard

    diagnostic, immutable = require_diagnostic_runtime(asset_root)
    context = runtime_context(diagnostic)
    teachers = load_teachers(context)
    split = split_rows()[0]
    order = list(split["basis_garments"])
    bounds = split["basis_contract"]["channel_bounds"]
    sample_by_name = {name: context["samples"][f"{name}/{CONDITION}"] for name in order}

    original = build_svd_basis(
        {name: teachers[name] for name in order}, bounds, order, 3,
    )
    original_repeat = build_svd_basis(
        {name: teachers[name] for name in order}, bounds, order, 3,
    )
    matrix32 = torch.stack([
        flatten_normalized(teachers[name], bounds, dtype=None)[0] for name in order
    ])
    centered32 = matrix32 - matrix32.mean(0)
    reported_singular = torch.linalg.svdvals(centered32.double())
    reproduction_rows = {}
    rebuilt_original = {}
    original_max_alpha = 0.0
    original_render_count = 0
    for name in order:
        rebuilt = original.basis(original.teacher_coefficients[name], chunk_size=16384)
        rebuilt_original[name] = rebuilt
        with torch.inference_mode():
            teacher_rgb, teacher_alpha = render_formal(context, sample_by_name[name], teachers[name])
            rebuilt_rgb, rebuilt_alpha = render_formal(context, sample_by_name[name], rebuilt)
        original_render_count += 2
        rgb_error = image_error(teacher_rgb, rebuilt_rgb)
        alpha_error = image_error(teacher_alpha, rebuilt_alpha)
        original_max_alpha = max(original_max_alpha, alpha_error["max_abs"])
        reproduction_rows[name] = {
            "residual": residual_error(teachers[name], rebuilt),
            "rgb": rgb_error,
            "alpha": alpha_error,
            "status": "PASS" if max(rgb_error["max_abs"], alpha_error["max_abs"]) <= RENDER_GATE else "FAIL",
        }
    reproduction = {
        "schema_version": "canondressgs.paper.loo_original_failure_reproduction.v1",
        "task_id": TASK_ID,
        "diagnostic_label": DIAGNOSTIC_LABEL,
        "split_id": "LOO-O01",
        "basis_garments": order,
        "source_teacher_fingerprints": original.metadata["teacher_fingerprints"],
        "singular_values": [float(value) for value in reported_singular],
        "expected_singular_values": list(EXPECTED_ORIGINAL_SINGULAR_VALUES),
        "singular_values_max_abs_error": max(
            abs(float(actual) - expected)
            for actual, expected in zip(reported_singular, EXPECTED_ORIGINAL_SINGULAR_VALUES)
        ),
        "basis_fingerprint": original.basis.fingerprint(),
        "expected_basis_fingerprint": EXPECTED_ORIGINAL_BASIS_FINGERPRINT,
        "repeat_fingerprint_match": original.basis.fingerprint() == original_repeat.basis.fingerprint(),
        "residual_parity": "PASS" if max(
            row["residual"]["aggregate_rmse"] for row in reproduction_rows.values()
        ) <= RENDER_GATE else "FAIL",
        "renderer_parity": "PASS" if all(
            row["status"] == "PASS" for row in reproduction_rows.values()
        ) else "FAIL",
        "renderer_gate": RENDER_GATE,
        "max_alpha_error": original_max_alpha,
        "expected_max_alpha_error": EXPECTED_ORIGINAL_MAX_ALPHA,
        "rows": reproduction_rows,
        "diagnostic_render_calls": original_render_count,
    }
    reproduction["exact_failure_reproduced"] = (
        reproduction["basis_fingerprint"] == EXPECTED_ORIGINAL_BASIS_FINGERPRINT
        and reproduction["repeat_fingerprint_match"]
        and reproduction["residual_parity"] == "PASS"
        and reproduction["renderer_parity"] == "FAIL"
        and reproduction["singular_values_max_abs_error"] <= 1e-9
        and abs(original_max_alpha - EXPECTED_ORIGINAL_MAX_ALPHA) <= 1e-9
    )
    if not reproduction["exact_failure_reproduced"]:
        write_json(diagnostic / "01_diagnostics/original_reproduction.json", reproduction)
        raise RuntimeError("LOO_BASIS_PARITY_FAILURE_NOT_REPRODUCIBLE")

    determinism_rows = {}
    determinism_render_count = 0
    for name in order:
        captures = []
        with torch.inference_mode():
            for _ in range(3):
                captures.append(capture_render(context, sample_by_name[name], teachers[name]))
                determinism_render_count += 1
        first = captures[0]
        determinism_rows[name] = {
            "runs": [
                {
                    "rgb_sha256": tensor_sha(item[0]),
                    "alpha_sha256": tensor_sha(item[1]),
                    **item[3],
                }
                for item in captures
            ],
            "comparisons_to_first": [
                {"rgb": image_error(first[0], item[0]), "alpha": image_error(first[1], item[1])}
                for item in captures[1:]
            ],
        }
    determinism = {
        "schema_version": "canondressgs.paper.loo_renderer_determinism_floor.v1",
        "task_id": TASK_ID,
        "diagnostic_label": DIAGNOSTIC_LABEL,
        "repeat_count": 3,
        "rows": determinism_rows,
        "max_same_state_rgb_error": max(
            comparison["rgb"]["max_abs"]
            for row in determinism_rows.values()
            for comparison in row["comparisons_to_first"]
        ),
        "max_same_state_alpha_error": max(
            comparison["alpha"]["max_abs"]
            for row in determinism_rows.values()
            for comparison in row["comparisons_to_first"]
        ),
        "diagnostic_render_calls": determinism_render_count,
    }

    layer_rows = {}
    layer_render_count = 0
    for name in order:
        teacher = teachers[name]
        rebuilt = rebuilt_original[name]
        teacher_guarded = apply_protected_full_residual_guard(teacher, context["protected_mask"])
        rebuilt_guarded = apply_protected_full_residual_guard(rebuilt, context["protected_mask"])
        with torch.inference_mode():
            t_rgb, t_alpha, t_tensors, t_summary = capture_render(
                context, sample_by_name[name], teacher_guarded,
            )
            r_rgb, r_alpha, r_tensors, r_summary = capture_render(
                context, sample_by_name[name], rebuilt_guarded,
            )
        layer_render_count += 2
        layers = {
            "L0/raw_residual/" + channel: tensor_error(
                getattr(teacher, channel), getattr(rebuilt, channel)
            )
            for channel in CHANNELS
        }
        layers.update({
            "L1/rank3_reconstructed/" + channel: tensor_error(
                getattr(teacher, channel), getattr(rebuilt, channel)
            )
            for channel in CHANNELS
        })
        layers.update({
            "L2/protected_residual/" + channel: tensor_error(
                getattr(teacher_guarded, channel), getattr(rebuilt_guarded, channel)
            )
            for channel in CHANNELS
        })
        layers.update({key: tensor_error(t_tensors[key], r_tensors[key]) for key in t_tensors})
        first_above_gate = next(
            (key for key, value in layers.items() if value["max_abs"] > RENDER_GATE),
            None,
        )
        layer_rows[name] = {
            "layers": layers,
            "first_layer_above_renderer_gate": first_above_gate,
            "teacher_summary": t_summary,
            "reconstructed_summary": r_summary,
            "rgb": image_error(t_rgb, r_rgb),
            "alpha": image_error(t_alpha, r_alpha),
        }
    layer_audit = {
        "schema_version": "canondressgs.paper.loo_render_state_layer_parity.v1",
        "task_id": TASK_ID,
        "diagnostic_label": DIAGNOSTIC_LABEL,
        "rows": layer_rows,
        "diagnostic_render_calls": layer_render_count,
    }

    channel_rows = {}
    channel_render_count = 0
    for name in order:
        with torch.inference_mode():
            teacher_rgb, teacher_alpha, _, teacher_summary = capture_render(
                context, sample_by_name[name], teachers[name],
            )
        channel_render_count += 1
        variants = {}
        for variant, selected in CHANNEL_VARIANTS.items():
            hybrid = hybrid_residual(teachers[name], rebuilt_original[name], selected)
            with torch.inference_mode():
                rgb, alpha, _, summary = capture_render(context, sample_by_name[name], hybrid)
            channel_render_count += 1
            rgb_difference = image_error(teacher_rgb, rgb)
            alpha_difference = image_error(teacher_alpha, alpha)
            variants[variant] = {
                "selected_channels": list(selected),
                "rgb": rgb_difference,
                "alpha": alpha_difference,
                "active_gaussian_change": summary["active_gaussian_count"] - teacher_summary["active_gaussian_count"],
                "visibility_change": summary["visible_gaussian_count"] - teacher_summary["visible_gaussian_count"],
                "parity_gate": RENDER_GATE,
                "status": "PASS" if max(
                    rgb_difference["max_abs"], alpha_difference["max_abs"]
                ) <= RENDER_GATE else "FAIL",
            }
        channel_rows[name] = variants
    channel_audit = {
        "schema_version": "canondressgs.paper.loo_residual_channel_parity.v1",
        "task_id": TASK_ID,
        "diagnostic_label": DIAGNOSTIC_LABEL,
        "rows": channel_rows,
        "diagnostic_render_calls": channel_render_count,
    }

    numerical_paths = {}
    numerical_decompositions = {}
    numerical_render_count = 0
    for label, dtype, zero_sum in (
        ("N0_float32", torch.float32, False),
        ("N1_float64", torch.float64, False),
        ("N2_float64_zero_sum", torch.float64, True),
    ):
        decomposition, metadata = make_decomposition(
            teachers, bounds, order, dtype=dtype, zero_sum=zero_sum,
        )
        numerical_decompositions[label] = decomposition
        rows = {}
        for name in order:
            rebuilt = reconstruct(decomposition, name)
            with torch.inference_mode():
                teacher_rgb, teacher_alpha = render_formal(context, sample_by_name[name], teachers[name])
                rebuilt_rgb, rebuilt_alpha = render_formal(context, sample_by_name[name], rebuilt)
            numerical_render_count += 2
            rows[name] = {
                "residual": residual_error(teachers[name], rebuilt),
                "rgb": image_error(teacher_rgb, rebuilt_rgb),
                "alpha": image_error(teacher_alpha, rebuilt_alpha),
                "status": "PASS" if max(
                    float((teacher_rgb - rebuilt_rgb).abs().max()),
                    float((teacher_alpha - rebuilt_alpha).abs().max()),
                ) <= RENDER_GATE else "FAIL",
            }
        numerical_paths[label] = {**metadata, "rows": rows}

    n1_basis = torch.cat([
        field.reshape(3, -1)
        for field in numerical_decompositions["N1_float64"]["basis"].normalized_fields()[1].values()
    ], dim=1)
    affine = anchor_diagnosis(teachers, bounds, order, n1_basis)

    solver_rows = {}
    for dtype in (torch.float32, torch.float64):
        dtype_rows = {}
        for solver in ("orthonormal_projection", "lstsq", "pinv", "svd_coordinates"):
            decomposition, metadata = make_decomposition(
                teachers, bounds, order, dtype=dtype, zero_sum=dtype == torch.float64,
                solver=solver,
            )
            dtype_rows[solver] = {
                **metadata,
                "reconstruction": {
                    name: residual_error(teachers[name], reconstruct(decomposition, name))
                    for name in order
                },
                "coefficient_sha256": canonical_sha({
                    name: tensor_sha(value)
                    for name, value in decomposition["teacher_coefficients"].items()
                }),
            }
        solver_rows[str(dtype)] = dtype_rows
    solver_audit = {
        "schema_version": "canondressgs.paper.loo_coefficient_solver_audit.v1",
        "task_id": TASK_ID,
        "diagnostic_label": DIAGNOSTIC_LABEL,
        "selection_rule": "mathematical correctness, closure, stability, determinism; never render outcome",
        "rows": solver_rows,
    }

    candidate_split_rows = []
    candidate_render_count = 0
    for candidate_split in split_rows():
        held_out = candidate_split["held_out_garment"]
        candidate_order = list(candidate_split["basis_garments"])
        candidate_bounds = candidate_split["basis_contract"]["channel_bounds"]
        decomposition, metadata = make_decomposition(
            teachers, candidate_bounds, candidate_order,
            dtype=torch.float64, zero_sum=True,
        )
        endpoints = {}
        for name in candidate_order:
            rebuilt = reconstruct(decomposition, name)
            sample = context["samples"][f"{name}/{CONDITION}"]
            with torch.inference_mode():
                teacher_rgb, teacher_alpha = render_formal(context, sample, teachers[name])
                rebuilt_rgb, rebuilt_alpha = render_formal(context, sample, rebuilt)
            candidate_render_count += 2
            rgb = image_error(teacher_rgb, rebuilt_rgb)
            alpha = image_error(teacher_alpha, rebuilt_alpha)
            endpoints[name] = {
                "residual": residual_error(teachers[name], rebuilt),
                "rgb": rgb,
                "alpha": alpha,
                "status": "PASS" if max(rgb["max_abs"], alpha["max_abs"]) <= RENDER_GATE else "FAIL",
            }
        candidate_split_rows.append({
            "split_id": candidate_split["split_id"],
            "held_out_garment": held_out,
            "basis_garments": candidate_order,
            **metadata,
            "rank": 3,
            "held_out_teacher_use_count": 0,
            "endpoints": endpoints,
            "status": "PASS" if all(row["status"] == "PASS" for row in endpoints.values()) else "FAIL",
        })
    candidate_validation = {
        "schema_version": "canondressgs.paper.loo_candidate_all_split_parity.v1",
        "task_id": TASK_ID,
        "diagnostic_label": DIAGNOSTIC_LABEL,
        "algorithm": "float64_mean_center_svd_rank3_zero_sum_then_single_renderer_entry_cast",
        "renderer_gate": RENDER_GATE,
        "splits": candidate_split_rows,
        "split_pass_count": sum(row["status"] == "PASS" for row in candidate_split_rows),
        "endpoint_pass_count": sum(
            endpoint["status"] == "PASS"
            for row in candidate_split_rows
            for endpoint in row["endpoints"].values()
        ),
        "diagnostic_render_calls": candidate_render_count,
    }

    metric_composition = {
        "schema_version": "canondressgs.paper.loo_composition_metric_audit.v1",
        "task_id": TASK_ID,
        "diagnostic_label": DIAGNOSTIC_LABEL,
        "teacher_composition_function": "scene.gaussian_clothing_residuals.compose_canonical_gaussian_overrides",
        "basis_composition_function": "scene.gaussian_clothing_residuals.compose_canonical_gaussian_overrides",
        "composition_path_equivalent": True,
        "channel_order": list(CHANNELS),
        "protected_guard_input_applied_to_both": True,
        "scale_parameterization": "additive raw log scaling then exp in renderer",
        "opacity_parameterization": "additive raw opacity logit then sigmoid in renderer",
        "rotation_parameterization": "normalize(q_base * axis_angle(delta_rotvec)), wxyz",
        "sh_layout_shared": True,
        "raw_shN_freeze_semantics_shared": True,
        "metric_tensor_domain": "unquantized float renderer outputs",
        "png_used_for_metric": False,
        "alpha_definition_shared": True,
        "camera_pose_background_equivalent": all(
            row["teacher_summary"]["camera_sha256"] == row["reconstructed_summary"]["camera_sha256"]
            and row["teacher_summary"]["pose_sha256"] == row["reconstructed_summary"]["pose_sha256"]
            and row["teacher_summary"]["background_sha256"] == row["reconstructed_summary"]["background_sha256"]
            for row in layer_rows.values()
        ),
        "renderer_gate": RENDER_GATE,
    }

    numerical = {
        "schema_version": "canondressgs.paper.loo_basis_numerical_closure.v1",
        "task_id": TASK_ID,
        "diagnostic_label": DIAGNOSTIC_LABEL,
        "original_dtype_pipeline": {
            "teacher_load": "torch.float32",
            "normalization": "torch.float32",
            "stack": "torch.float32",
            "mean": "torch.float32",
            "centering": "torch.float32",
            "svd": "torch.float32",
            "rank_statistics_cast": "torch.float64 after float32 centering",
            "basis_storage": "torch.float32",
            "coefficient_projection": "torch.float32",
            "reconstruction": "torch.float32",
            "renderer_cast": "none because residual already matches base dtype",
        },
        "paths": numerical_paths,
        "diagnostic_render_calls": numerical_render_count,
    }
    affine_payload = {
        "schema_version": "canondressgs.paper.loo_affine_rank3_closure.v1",
        "task_id": TASK_ID,
        "diagnostic_label": DIAGNOSTIC_LABEL,
        **affine,
    }

    successful_run_render_calls = sum((
        original_render_count,
        determinism_render_count,
        layer_render_count,
        channel_render_count,
        numerical_render_count,
        candidate_render_count,
    ))
    interrupted_render_calls = prior_interrupted_render_calls(diagnostic)
    total_render_calls = successful_run_render_calls + interrupted_render_calls
    final_immutable = original_immutability(asset_root, load_manifest(asset_root))
    summary = {
        "schema_version": "canondressgs.paper.loo_basis_renderer_parity_diagnosis_summary.v1",
        "task_id": TASK_ID,
        "diagnostic_label": DIAGNOSTIC_LABEL,
        "status": "PASS",
        "original_failure_reproduced": reproduction["exact_failure_reproduced"],
        "same_state_max_rgb": determinism["max_same_state_rgb_error"],
        "same_state_max_alpha": determinism["max_same_state_alpha_error"],
        "candidate_split_pass_count": candidate_validation["split_pass_count"],
        "candidate_endpoint_pass_count": candidate_validation["endpoint_pass_count"],
        "diagnostic_render_calls": total_render_calls,
        "successful_run_diagnostic_render_calls": successful_run_render_calls,
        "interrupted_run_diagnostic_render_calls": interrupted_render_calls,
        "optimizer_creations": 0,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "formal_metrics": 0,
        "scientific_visual_sheets": 0,
        "attempt_002_created": False,
        "original_attempt_immutability": final_immutable,
        "PAPER_FINAL": False,
    }
    if final_immutable["status"] != "PASS":
        raise RuntimeError("original attempt mutated during diagnosis")
    outputs = {
        "original_reproduction.json": reproduction,
        "renderer_determinism_floor.json": determinism,
        "render_state_layer_parity.json": layer_audit,
        "residual_channel_parity.json": channel_audit,
        "basis_numerical_closure.json": numerical,
        "affine_rank3_closure.json": affine_payload,
        "coefficient_solver_audit.json": solver_audit,
        "composition_metric_audit.json": metric_composition,
    }
    for name, payload in outputs.items():
        write_json(diagnostic / "01_diagnostics" / name, payload)
    write_json(diagnostic / "02_candidate_validation/all_split_parity.json", candidate_validation)
    write_json(diagnostic / "diagnosis_summary.json", summary)
    return summary


def verify_preflight(asset_root: Path) -> dict[str, Any]:
    _, diagnostic = output_paths(asset_root)
    manifest = load_manifest(asset_root)
    immutable = original_immutability(asset_root, manifest)
    errors = []
    if immutable["status"] != "PASS":
        errors.append("original_attempt_mutated")
    if (asset_root / ORIGINAL_OUTPUT_NAME / "attempt_002").exists():
        errors.append("attempt_002_exists")
    for path in diagnostic.rglob("*.json"):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            errors.append(f"json:{path}:{type(error).__name__}")
    temporary = [str(path) for path in diagnostic.rglob("*.tmp")]
    errors.extend(f"temporary:{path}" for path in temporary)
    return {
        "schema_version": "canondressgs.paper.loo_basis_renderer_parity_preflight_verification.v1",
        "task_id": TASK_ID,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "original_attempt_immutability": immutable,
        "attempt_002_absent": not (asset_root / ORIGINAL_OUTPUT_NAME / "attempt_002").exists(),
        "PAPER_FINAL": False,
    }


def repaired_basis_semantic_sha(decomposition: Any) -> str:
    mean, components = decomposition.basis.normalized_fields()
    return canonical_sha({
        "basis_fingerprint": decomposition.basis.fingerprint(),
        "mean": {name: tensor_sha(value) for name, value in mean.items()},
        "components": {name: tensor_sha(value) for name, value in components.items()},
        "coefficients": {
            name: tensor_sha(value)
            for name, value in decomposition.teacher_coefficients.items()
        },
    })


def validate_repair(asset_root: Path) -> dict[str, Any]:
    from scene.explicit_gaussian_residual_basis import (
        build_svd_basis,
        normalized_residual_dict,
    )
    from scene.gaussian_clothing_residuals import CHANNELS

    diagnostic, immutable = require_diagnostic_runtime(asset_root)
    diagnosis_path = diagnostic / "diagnosis_summary.json"
    if not diagnosis_path.is_file():
        raise RuntimeError("diagnosis must pass before repaired validation")
    diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    if diagnosis["status"] != "PASS" or not diagnosis["original_failure_reproduced"]:
        raise RuntimeError("diagnosis did not reproduce the frozen failure")
    context = runtime_context(diagnostic)
    teachers = load_teachers(context)
    split_results = []
    validation_render_calls = 0
    for split in split_rows():
        held_out = split["held_out_garment"]
        order = list(split["basis_garments"])
        bounds = split["basis_contract"]["channel_bounds"]
        normalized = {
            name: normalized_residual_dict(
                teachers[name], bounds, dtype=torch.float64,
            )
            for name in order
        }
        matrix = torch.stack([
            torch.cat([normalized[name][field].reshape(-1) for field in CHANNELS])
            for name in order
        ])
        centered = matrix - matrix.mean(0)
        centered = centered.clone()
        centered[-1] = -centered[:-1].sum(0)
        singular_values = torch.linalg.svdvals(centered)
        tolerance = max(4, int(centered.shape[1])) * torch.finfo(
            centered.dtype
        ).eps * float(singular_values[0])
        numerical_rank = int((singular_values > tolerance).sum())
        selected_rank = min(numerical_rank, 3)
        decomposition = build_svd_basis(
            {name: teachers[name] for name in order}, bounds, order, selected_rank,
        )
        repeated = build_svd_basis(
            {name: teachers[name] for name in order}, bounds, order, selected_rank,
        )
        coefficient_matrix = torch.stack([
            decomposition.teacher_coefficients[name] for name in order
        ])
        normalization = {
            "held_out_garment": held_out,
            "basis_garments": order,
            "mean": coefficient_matrix.mean(0).detach().cpu().tolist(),
            "std": coefficient_matrix.std(0, unbiased=False).clamp_min(1e-8).detach().cpu().tolist(),
            "held_out_in_normalization": False,
            "dtype": "torch.float64",
        }
        endpoint_results = {}
        for name in order:
            sample = context["samples"][f"{name}/{CONDITION}"]
            rebuilt = decomposition.basis(
                decomposition.teacher_coefficients[name], chunk_size=16384,
            )
            with torch.inference_mode():
                teacher_rgb, teacher_alpha = render_formal_repaired(
                    context, sample, teachers[name],
                )
                rebuilt_rgb, rebuilt_alpha = render_formal_repaired(
                    context, sample, rebuilt,
                )
                _, _, teacher_inputs, teacher_summary = capture_render(
                    context, sample, teachers[name], explicit_entry_cast=False,
                )
                _, _, rebuilt_inputs, rebuilt_summary = capture_render(
                    context, sample, rebuilt, explicit_entry_cast=False,
                )
            validation_render_calls += 4
            rgb = image_error(teacher_rgb, rebuilt_rgb)
            alpha = image_error(teacher_alpha, rebuilt_alpha)
            input_rows = {
                key: tensor_error(teacher_inputs[key], rebuilt_inputs[key])
                for key in teacher_inputs
                if key.startswith("L7/raster_")
            }
            input_max = max(row["max_abs"] for row in input_rows.values())
            residual = residual_error(teachers[name], rebuilt)
            endpoint_results[name] = {
                "residual": residual,
                "rgb": rgb,
                "alpha": alpha,
                "renderer_input": input_rows,
                "renderer_input_max_abs": input_max,
                "camera_pose_background_equal": (
                    teacher_summary["camera_sha256"] == rebuilt_summary["camera_sha256"]
                    and teacher_summary["pose_sha256"] == rebuilt_summary["pose_sha256"]
                    and teacher_summary["background_sha256"] == rebuilt_summary["background_sha256"]
                ),
                "status": "PASS" if (
                    max(rgb["max_abs"], alpha["max_abs"]) <= RENDER_GATE
                    and input_max <= RENDER_GATE
                    and residual["max_abs"] <= 1e-12
                ) else "FAIL",
            }
        basis_sha = repaired_basis_semantic_sha(decomposition)
        split_result = {
            "split_id": split["split_id"],
            "held_out_garment": held_out,
            "basis_garments": order,
            "singular_values": [float(value) for value in singular_values],
            "numerical_rank_tolerance": tolerance,
            "numerical_rank": numerical_rank,
            "selected_rank": selected_rank,
            "centered_sum_l2": float(torch.linalg.vector_norm(centered.sum(0))),
            "centered_sum_max_abs": float(centered.sum(0).abs().max()),
            "basis_fingerprint": decomposition.basis.fingerprint(),
            "basis_semantic_sha256": basis_sha,
            "normalization_sha256": canonical_sha(normalization),
            "normalization": normalization,
            "repeat_fingerprint_match": (
                decomposition.basis.fingerprint() == repeated.basis.fingerprint()
                and basis_sha == repaired_basis_semantic_sha(repeated)
            ),
            "held_out_teacher_in_basis_count": 0,
            "held_out_teacher_in_normalization_count": 0,
            "full_five_garment_basis_reused": False,
            "endpoints": endpoint_results,
            "status": "PASS" if all(
                row["status"] == "PASS" for row in endpoint_results.values()
            ) else "FAIL",
        }
        split_results.append(split_result)
        del matrix, centered, decomposition, repeated
        torch.cuda.empty_cache()
    split_pass_count = sum(row["status"] == "PASS" for row in split_results)
    endpoint_pass_count = sum(
        endpoint["status"] == "PASS"
        for split in split_results
        for endpoint in split["endpoints"].values()
    )
    renderer_input_pass_count = sum(
        endpoint["renderer_input_max_abs"] <= RENDER_GATE
        for split in split_results
        for endpoint in split["endpoints"].values()
    )
    final_immutable = original_immutability(asset_root, load_manifest(asset_root))
    result = {
        "schema_version": "canondressgs.paper.loo_all_split_basis_parity_repaired.v1",
        "task_id": TASK_ID,
        "diagnostic_label": DIAGNOSTIC_LABEL,
        "status": "PASS" if (
            split_pass_count == 5
            and endpoint_pass_count == 20
            and renderer_input_pass_count == 20
            and final_immutable["status"] == "PASS"
        ) else "FAIL",
        "algorithm": "float64_mean_center_rank3_svd_strict_zero_sum",
        "coefficient_solver": "orthonormal_basis_transpose_projection",
        "renderer_entry_cast": "shared canonical composition once to base dtype",
        "renderer_gate": RENDER_GATE,
        "split_pass_count": split_pass_count,
        "endpoint_pass_count": endpoint_pass_count,
        "renderer_input_pass_count": renderer_input_pass_count,
        "splits": split_results,
        "validation_diagnostic_render_calls": validation_render_calls,
        "prior_diagnostic_render_calls": (
            int(diagnosis["successful_run_diagnostic_render_calls"])
            + prior_interrupted_render_calls(diagnostic)
        ),
        "actual_diagnostic_render_calls": (
            int(diagnosis["successful_run_diagnostic_render_calls"])
            + prior_interrupted_render_calls(diagnostic)
            + validation_render_calls
        ),
        "optimizer_creations": 0,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "formal_metrics": 0,
        "scientific_visual_sheets": 0,
        "attempt_002_created": False,
        "held_out_teacher_use_count": 0,
        "original_attempt_immutability": final_immutable,
        "PAPER_FINAL": False,
    }
    write_json(
        diagnostic / "03_repaired_validation/all_split_parity.json",
        result,
        replace=True,
    )
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Diagnose and repair LOO basis renderer parity")
    result.add_argument(
        "command",
        choices=(
            "preflight", "record-known-interruption", "diagnose", "validate-repair",
            "verify-preflight",
        ),
    )
    result.add_argument("--asset-root", type=Path)
    return result


def main() -> None:
    arguments = parser().parse_args()
    asset_root = arguments.asset_root
    if asset_root is None:
        value = os.environ.get("CANONDRESSGS_ASSET_ROOT")
        if not value:
            raise RuntimeError("--asset-root or CANONDRESSGS_ASSET_ROOT is required")
        asset_root = Path(value)
    commands = {
        "preflight": preflight,
        "record-known-interruption": record_known_interruption,
        "diagnose": diagnose,
        "validate-repair": validate_repair,
        "verify-preflight": verify_preflight,
    }
    output = commands[arguments.command](asset_root)
    print(json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
