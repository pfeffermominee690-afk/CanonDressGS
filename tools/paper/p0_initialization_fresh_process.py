from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from scene.p0_candidate_initialization_protocol import (
    CandidateBuild,
    build_b6_candidate,
    build_m3_m4_paired_candidates,
    build_ours_v2_candidate,
    tensor_mapping_sha256,
)


OUTFITS = ("O01", "O02", "O03", "O04", "O08")
CONDITION = "cond_000000"


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()


def _rng_fingerprints() -> dict[str, Any]:
    return {
        "python": _object_sha256(random.getstate()),
        "numpy": _object_sha256(np.random.get_state()),
        "torch_cpu": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        "torch_cuda": [
            hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest()
            for value in torch.cuda.get_rng_state_all()
        ] if torch.cuda.is_available() else [],
    }


def _feature_value(
    cache: Mapping[str, Any], outfit: str, kind: str, device: torch.device
) -> torch.Tensor:
    row = cache["episodes"][f"{outfit}/{CONDITION}"]["normal"]
    rows = row[kind].to(device)
    valid = row["valid"].to(device)
    if kind == "f2":
        denominator = valid.sum().clamp_min(1e-8)
        mean = (rows * valid).sum(0, keepdim=True) / denominator
        maximum = rows.masked_fill(valid <= 0, torch.finfo(rows.dtype).min).amax(
            0, keepdim=True
        )
        return torch.cat((mean, maximum), dim=-1)
    if kind == "rff":
        if rows.shape[0] != 3 or valid.shape != (3, 1):
            raise ValueError("protocol complex audit requires three references")
        return torch.cat((rows.reshape(-1), valid.reshape(-1)))
    raise KeyError(kind)


def _selected_sha(module: torch.nn.Module, predicate) -> str:
    values = {
        name: value for name, value in module.state_dict().items() if predicate(name)
    }
    return "NOT_PRESENT" if not values else tensor_mapping_sha256(values)


def _snapshot(build: CandidateBuild, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    module = build.module
    outputs = []
    pre_transform = []
    with torch.no_grad():
        for outfit in OUTFITS:
            result = module(batch[outfit])
            if hasattr(result, "logits"):
                value = result.logits
                pre = value
            else:
                value = result.standardized_coefficients
                pre = result.pre_transform_output
            outputs.append(value.detach().cpu())
            pre_transform.append(pre.detach().cpu())
    output_matrix = torch.stack(outputs)
    pre_matrix = torch.stack(pre_transform)
    state = module.state_dict()
    manifest = build.manifest()
    return {
        **manifest,
        "trainable_state_sha256": tensor_mapping_sha256(state),
        "layernorm_sha256": _selected_sha(
            module, lambda name: "norm" in name.lower()
        ),
        "trunk_sha256": _selected_sha(
            module, lambda name: name.startswith("trunk.")
        ),
        "head_sha256": _selected_sha(
            module,
            lambda name: name.startswith("linear.") or name.startswith("output_head."),
        ),
        "step0_forward": output_matrix.tolist(),
        "step0_forward_sha256": tensor_mapping_sha256({"output": output_matrix}),
        "pre_transform_sha256": tensor_mapping_sha256({"pre_transform": pre_matrix}),
        "initial_output_all_zero": bool(torch.count_nonzero(output_matrix) == 0),
        "candidate_optimizer_parameter_count": 0,
        "legacy_optimizer_parameter_count": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fresh-process initialization snapshot")
    parser.add_argument("--family", choices=("ours_v2", "b6", "complex_pair"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested fresh process")
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng_before = _rng_fingerprints()
    cache = torch.load(args.feature_cache, map_location="cpu", weights_only=False)

    if args.family in {"ours_v2", "b6"}:
        batch = {
            outfit: _feature_value(cache, outfit, "f2", device) for outfit in OUTFITS
        }
        input_dim = int(next(iter(batch.values())).numel())
        build = (
            build_ours_v2_candidate(input_dim=input_dim, seed=args.seed, device=device)
            if args.family == "ours_v2"
            else build_b6_candidate(input_dim=input_dim, seed=args.seed, device=device)
        )
        methods = {build.identity: _snapshot(build, batch)}
    else:
        batch = {
            outfit: _feature_value(cache, outfit, "rff", device) for outfit in OUTFITS
        }
        raw_dim = (int(next(iter(batch.values())).numel()) - 3) // 3
        m3, m4 = build_m3_m4_paired_candidates(
            raw_dim=raw_dim, seed=args.seed, device=device
        )
        methods = {"M3": _snapshot(m3, batch), "M4": _snapshot(m4, batch)}

    batch_cpu = {name: value.detach().cpu() for name, value in batch.items()}
    result = {
        "schema_version": "canondressgs.paper.initialization_fresh_process.v1",
        "family": args.family,
        "seed": args.seed,
        "pid": __import__("os").getpid(),
        "python_rng_before_model": rng_before["python"],
        "numpy_rng_before_model": rng_before["numpy"],
        "torch_cpu_rng_before_model": rng_before["torch_cpu"],
        "torch_cuda_rng_before_model": rng_before["torch_cuda"],
        "rng_after_model": _rng_fingerprints(),
        "fixed_first_batch_sha256": tensor_mapping_sha256(batch_cpu),
        "device": str(device),
        "methods": methods,
        "candidate_optimizer_created": False,
        "candidate_optimizer_step_count": 0,
        "backward_called": False,
        "checkpoint_created": False,
        "render_created": False,
        "formal_metrics_created": False,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
