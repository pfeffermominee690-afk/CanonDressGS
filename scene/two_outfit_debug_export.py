from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

import torch


EXPORT_SCHEMA_VERSION = "canondressgs.reference_conditioning_debug.v1"
FORBIDDEN_EXPORT_NAMES = frozenset({
    "target_feature",
    "target_features",
    "target_image_feature",
    "target_image_features",
    "target_embedding",
})


def tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    detached = value.detach()
    finite = torch.isfinite(detached) if torch.is_floating_point(detached) else torch.ones_like(detached, dtype=torch.bool)
    flattened = detached.float().reshape(-1)
    result: dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": detached.numel(),
        "requires_grad": bool(value.requires_grad),
        "finite": bool(finite.all()),
    }
    if flattened.numel():
        result.update({
            "min": float(flattened.min()),
            "max": float(flattened.max()),
            "mean": float(flattened.mean()),
            "std": float(flattened.std(unbiased=False)),
            "l1_mean": float(flattened.abs().mean()),
            "l2_rms": float(flattened.square().mean().sqrt()),
        })
    return result


def _residual_mapping(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "as_dict"):
        return value.as_dict()
    raise TypeError("residual export expects a mapping or residual dataclass")


def collect_reference_conditioning_debug(
    forward_output: Mapping[str, Any],
    *,
    outfit_id: str,
    episode_id: str,
    step: int,
    commit: str,
    config_hash: str,
    include_full_tensors: bool = False,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Read debug values from an already-computed formal forward output.

    Tensors are detached only for the export copy.  The supplied output is never
    edited, reordered, or recomputed.
    """

    completion = forward_output.get("completion")
    observed = forward_output.get("observed", {})
    render = forward_output.get("render")
    rgb = forward_output.get("final_rgb")
    alpha = forward_output.get("final_alpha")
    if isinstance(render, (tuple, list)):
        if rgb is None and len(render) > 0 and isinstance(render[0], torch.Tensor):
            rgb = render[0]
        if alpha is None and len(render) > 1 and isinstance(render[1], torch.Tensor):
            alpha = render[1]

    values: dict[str, Any] = {
        "global_clothing_embedding": forward_output.get("global_clothing_embedding"),
        "local_anchor_features": {
            "observed": (
                getattr(completion, "observed_anchor_features", None)
                if completion is not None
                else observed.get("observed_clothing_feature")
            ),
            "completed": getattr(completion, "completed_anchor_features", None),
        },
        "observed_probability": (
            getattr(completion, "observed_clothing_probability", None)
            if completion is not None
            else observed.get("observed_clothing_probability")
        ),
        "completed_probability": getattr(completion, "confidence", None),
        "geometry_gate": getattr(completion, "geometry_gate", forward_output.get("geometry_gate")),
        "appearance_gate": getattr(completion, "appearance_gate", forward_output.get("appearance_gate")),
        "anchor_residual": _residual_mapping(
            forward_output.get("gated_anchor_residuals", forward_output.get("anchor_residuals"))
        ),
        "gaussian_residual": _residual_mapping(
            forward_output.get("gaussian_residuals", forward_output.get("gaussian_offsets"))
        ),
        "final_rgb": rgb,
        "final_alpha": alpha,
    }

    summaries: dict[str, Any] = {}
    full: dict[str, torch.Tensor] = {}

    def visit(prefix: str, value: Any) -> None:
        if value is None:
            summaries[prefix] = None
        elif isinstance(value, torch.Tensor):
            summaries[prefix] = tensor_summary(value)
            if include_full_tensors:
                full[prefix] = value.detach().cpu().clone()
        elif isinstance(value, Mapping):
            summaries[prefix] = {}
            for key in sorted(value):
                child_name = f"{prefix}.{key}"
                visit(child_name, value[key])
                summaries[prefix][key] = summaries.pop(child_name)
        else:
            raise TypeError(f"unsupported debug value at {prefix}: {type(value).__name__}")

    for name, value in values.items():
        visit(name, value)
    lowered = {name.lower() for name in summaries}
    if FORBIDDEN_EXPORT_NAMES.intersection(lowered):
        raise RuntimeError("target image features entered the debug export")
    record = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "metadata": {
            "outfit_id": str(outfit_id),
            "episode_id": str(episode_id),
            "step": int(step),
            "commit": str(commit),
            "config_hash": str(config_hash),
            "full_tensors_included": bool(include_full_tensors),
        },
        "summaries": summaries,
        "target_image_features_exported": False,
    }
    return record, full


def run_with_optional_debug(
    forward_fn: Callable[..., Mapping[str, Any]],
    *args: Any,
    debug_enabled: bool = False,
    debug_metadata: Mapping[str, Any] | None = None,
    include_full_tensors: bool = False,
    **kwargs: Any,
) -> tuple[Mapping[str, Any], dict[str, Any] | None, dict[str, torch.Tensor]]:
    output = forward_fn(*args, **kwargs)
    if not debug_enabled:
        return output, None, {}
    if debug_metadata is None:
        raise ValueError("debug_metadata is required when debug export is enabled")
    record, tensors = collect_reference_conditioning_debug(
        output,
        outfit_id=str(debug_metadata["outfit_id"]),
        episode_id=str(debug_metadata["episode_id"]),
        step=int(debug_metadata["step"]),
        commit=str(debug_metadata["commit"]),
        config_hash=str(debug_metadata["config_hash"]),
        include_full_tensors=include_full_tensors,
    )
    return output, record, tensors


def write_reference_conditioning_debug(
    output_dir: str | Path,
    record: Mapping[str, Any],
    full_tensors: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, str]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = record["metadata"]
    tag = "__".join(
        str(metadata[name]).replace("/", "_").replace("\\", "_")
        for name in ("outfit_id", "episode_id", "step", "commit", "config_hash")
    )
    summary_path = directory / f"conditioning_debug__{tag}.json"
    summary_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths = {"summary": str(summary_path)}
    if full_tensors:
        tensor_path = directory / f"conditioning_debug__{tag}.pt"
        torch.save(dict(full_tensors), tensor_path)
        paths["full_tensors"] = str(tensor_path)
    return paths
