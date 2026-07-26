from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
import yaml
from torch import nn

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

from scene.multi_outfit_linear_coefficient_control import MultiOutfitLinearCoefficientControl
from tools import run_multi_outfit_explicit_basis as multi
from tools.paper import formal_batch_runtime as formal
from tools.paper.verify_seen_outfit_paper_assets import verify_manifest


TASK_ID = "AAAI27-P0-REVIEWER-RISK-CLOSURE-001"
BRANCH = "paper/aaai27-p0-reviewer-risk-closure-20260721"
SOURCE_HEAD = "4bd47b7b667d1ade230b6b0ae3d611f550548c84"
# The formal executor is Linux.  These are the LF checkout byte fingerprints;
# the corresponding Windows CRLF checkout fingerprints are recorded separately
# by the adjudication task and are not used as Linux execution gates.
FORMAL_REGISTRY_SHA256 = "1834597b787d98acf475b352b791f0f16714fd870d51be36259ac7383a406c5e"
FROZEN_MANIFEST_SHA256 = "ff90540e56c9c2db3fd6a1e45c31a1d1eb5a8a32effabde71158584d9be538bb"
REVIEWER_REGISTRY_BEFORE_SHA256 = "139e763a04cbbdf626af194c92c388b7505a6ce1f9454d49afca743e09a0990c"
FORMAL_OUTPUT_BEFORE_SHA256 = "7b9449e03d53e29cff11ded1fdc95e633640d38754cf152f2ab07a935d89d6bc"
TRAIN_OUTFITS = ("O01", "O02", "O03", "O04", "O08")
SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class ClassifierOutput:
    logits: torch.Tensor


class ReferenceClassifier(nn.Module):
    """Reference-only B6 step-0 head; labels are not accepted by forward."""

    def __init__(self, input_dim: int, class_count: int = 5) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, class_count)

    def forward(self, frozen_reference_feature: torch.Tensor) -> ClassifierOutput:
        if frozen_reference_feature.ndim == 1:
            frozen_reference_feature = frozen_reference_feature.unsqueeze(0)
        if frozen_reference_feature.ndim != 2 or frozen_reference_feature.shape[0] != 1:
            raise ValueError("B6 expects one aggregated reference-only feature")
        logits = self.linear(self.normalization(frozen_reference_feature)).reshape(-1)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("B6 logits contain NaN or Inf")
        return ClassifierOutput(logits=logits)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=PROJECT_ROOT, text=True
    ).strip()


def _object_sha(value: Any) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()


def _tensor_mapping_sha(values: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


def _selected_state_sha(model: nn.Module, predicate: Callable[[str], bool]) -> str:
    selected = {
        name: value for name, value in model.state_dict().items() if predicate(name)
    }
    if not selected:
        return "NOT_PRESENT"
    return _tensor_mapping_sha(selected)


def _rng_fingerprints() -> dict[str, Any]:
    cuda_states = torch.cuda.get_rng_state_all()
    return {
        "python": _object_sha(random.getstate()),
        "numpy": _object_sha(np.random.get_state()),
        "torch_cpu": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        "torch_cuda": [
            hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest()
            for value in cuda_states
        ],
    }


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _formal_metadata_fingerprint(path: Path) -> str:
    command = (
        "cd " + shlex.quote(str(path))
        + " && find . -type f -printf '%P %s %T@\\n'"
        + " | LC_ALL=C sort | sha256sum"
    )
    return subprocess.check_output(["bash", "-lc", command], text=True).split()[0]


def _build_context(attempt: Path) -> dict[str, Any]:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs/research/subject02_multi_outfit_explicit_basis_v1.yaml")
        .read_text(encoding="utf-8")
    )
    old_branch, old_source = multi.EXPECTED_BRANCH, multi.EXPECTED_SOURCE_HEAD
    multi.EXPECTED_BRANCH, multi.EXPECTED_SOURCE_HEAD = BRANCH, SOURCE_HEAD
    config["branch"] = BRANCH
    config["source_head"] = SOURCE_HEAD
    config["output"]["attempt"] = attempt.name
    try:
        return multi.build_context(
            config, attempt, create=False, reference_backbone=True
        )
    finally:
        multi.EXPECTED_BRANCH, multi.EXPECTED_SOURCE_HEAD = old_branch, old_source


def _feature_batches(
    cache: Mapping[str, Any], device: torch.device
) -> dict[str, dict[str, torch.Tensor]]:
    condition = formal.CONDITIONS[0]
    batches: dict[str, dict[str, torch.Tensor]] = {}
    for family, kind in (
        ("ours_v2", "f2"),
        ("b6_reference_classifier", "f2"),
        ("complex_fusion_corrected", "rff"),
    ):
        batches[family] = {
            outfit: formal.feature_value(
                cache, f"{outfit}/{condition}", kind, device,
                indices=(0, 1, 2),
            )
            for outfit in TRAIN_OUTFITS
        }
    return batches


def _batch_sha(batch: Mapping[str, torch.Tensor]) -> str:
    return _tensor_mapping_sha(batch)


def _model_builder(
    family: str, sample: torch.Tensor, device: torch.device
) -> nn.Module:
    if family == "ours_v2":
        model: nn.Module = MultiOutfitLinearCoefficientControl(sample.numel(), 4)
    elif family == "b6_reference_classifier":
        model = ReferenceClassifier(sample.numel(), len(TRAIN_OUTFITS))
    elif family == "complex_fusion_corrected":
        raw_dim = (sample.numel() - 3) // 3
        model = formal.LegacyRFFRank4(raw_dim=raw_dim, rank=4)
    else:
        raise KeyError(family)
    return model.to(device)


def _forward_values(
    family: str, model: nn.Module, batch: Mapping[str, torch.Tensor]
) -> tuple[list[list[float]], str]:
    outputs = []
    with torch.no_grad():
        for outfit in TRAIN_OUTFITS:
            result = model(batch[outfit])
            value = (
                result.logits
                if family == "b6_reference_classifier"
                else result.standardized_coefficients
            )
            outputs.append(value.detach().cpu())
    matrix = torch.stack(outputs)
    return matrix.tolist(), _tensor_mapping_sha({"first_balanced_forward": matrix})


def _snapshot(
    family: str, seed: int, batch: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, Any]:
    _seed(seed)
    rng_before = _rng_fingerprints()
    model = _model_builder(family, next(iter(batch.values())), device)
    rng_after = _rng_fingerprints()
    forward, forward_sha = _forward_values(family, model, batch)
    state = model.state_dict()
    return {
        "seed": seed,
        "rng_before_model_initialization": rng_before,
        "rng_after_model_initialization": rng_after,
        "trainable_parameter_initialization_sha256": _tensor_mapping_sha(state),
        "layernorm_parameter_sha256": _selected_state_sha(
            model, lambda name: "norm" in name.lower()
        ),
        "linear_or_head_parameter_sha256": _selected_state_sha(
            model,
            lambda name: "linear" in name.lower() or "fusion" in name.lower(),
        ),
        "first_balanced_forward": forward,
        "first_balanced_forward_sha256": forward_sha,
        "trainable_parameter_count": sum(value.numel() for value in model.parameters()),
        "optimizer_created": False,
    }


def _audit_family(
    family: str, batch: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, Any]:
    rows = []
    repeat_rows = []
    for seed in SEEDS:
        first = _snapshot(family, seed, batch, device)
        repeat = _snapshot(family, seed, batch, device)
        repeat_exact = all(
            first[name] == repeat[name]
            for name in (
                "trainable_parameter_initialization_sha256",
                "layernorm_parameter_sha256",
                "linear_or_head_parameter_sha256",
                "first_balanced_forward_sha256",
            )
        )
        rows.append(first)
        repeat_rows.append({
            "seed": seed,
            "bitwise_exact": repeat_exact,
            "repeat_initialization_sha256": repeat[
                "trainable_parameter_initialization_sha256"
            ],
            "repeat_forward_sha256": repeat["first_balanced_forward_sha256"],
        })
    initialization_shas = {
        row["trainable_parameter_initialization_sha256"] for row in rows
    }
    batch_sha = _batch_sha(batch)
    schedule = [
        {
            "step": step,
            "condition": formal.CONDITIONS[(step - 1) % len(formal.CONDITIONS)],
            "outfit_order": list(TRAIN_OUTFITS),
            "scheduler_multiplier": 1.0,
        }
        for step in range(1, 301)
    ]
    passed = len(initialization_shas) == len(SEEDS) and all(
        row["bitwise_exact"] for row in repeat_rows
    )
    return {
        "status": "SEED-PROPAGATION-PASS" if passed else "SEED-PROPAGATION-FAIL",
        "seed_values": list(SEEDS),
        "seed_is_passed_to_model_construction": True,
        "initialization_varies_by_seed": len(initialization_shas) == len(SEEDS),
        "initialization_unique_count": len(initialization_shas),
        "same_seed_repeat_bitwise_exact": all(
            row["bitwise_exact"] for row in repeat_rows
        ),
        "first_balanced_batch_sha256": batch_sha,
        "first_balanced_batch_fixed_across_seeds": True,
        "episode_and_asset_set_fixed_across_seeds": True,
        "dataloader_scheduler_order_sha256": hashlib.sha256(
            json.dumps(schedule, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "dataloader_scheduler_order": {
            "outfit_order": list(TRAIN_OUTFITS),
            "condition_round_robin": list(formal.CONDITIONS),
            "optimizer_steps": 300,
            "scheduler": "constant_lambda_1.0",
        },
        "snapshots": rows,
        "same_seed_repeats": repeat_rows,
    }


def _registry_summary(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    rows = payload["experiments"]
    statuses: dict[str, int] = {}
    priorities: dict[str, int] = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
        priorities[row["priority"]] = priorities.get(row["priority"], 0) + 1
    strict = [
        row for row in rows
        if row["experiment_id"] == "RR-STRICT-VIEW-ONE-FOLD-CANARY"
    ]
    if len(strict) != 1 or strict[0]["status"] != "BLOCKED_PENDING_AUTHORIZATION":
        raise RuntimeError("strict-view canary authorization state changed")
    return {
        "total_count": len(rows),
        "status_counts": statuses,
        "priority_counts": priorities,
        "experiment_ids": [row["experiment_id"] for row in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="P0 seed-propagation hard gate")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--formal-output-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    formal_output_root = args.formal_output_root.resolve()
    asset_root = args.asset_root.resolve()
    registry_path = PROJECT_ROOT / "paper_protocol/reviewer_risk/reviewer_risk_experiment_registry.yaml"
    formal_registry_path = PROJECT_ROOT / "paper_protocol/experiment_registry.yaml"
    manifest_path = PROJECT_ROOT / "paper_protocol/frozen_asset_manifest.json"

    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    dirty = _git("status", "--short")
    if branch != BRANCH or dirty:
        raise RuntimeError(f"P0 preflight requires clean {BRANCH}")
    if subprocess.call(
        ["git", "merge-base", "--is-ancestor", SOURCE_HEAD, "HEAD"], cwd=PROJECT_ROOT
    ) != 0:
        raise RuntimeError("P0 branch is not descended from the adjudication HEAD")
    if not torch.cuda.is_available():
        raise RuntimeError("P0_GPU_PREFLIGHT_FAIL")
    if tuple(formal.OUTFITS) != TRAIN_OUTFITS or "O06" in formal.OUTFITS or "O07" in formal.OUTFITS:
        raise RuntimeError("P0 train-outfit isolation changed")

    formal_registry_sha = _sha256_file(formal_registry_path)
    manifest_sha = _sha256_file(manifest_path)
    reviewer_registry_sha = _sha256_file(registry_path)
    formal_output_sha = _formal_metadata_fingerprint(formal_output_root)
    if formal_registry_sha != FORMAL_REGISTRY_SHA256:
        raise RuntimeError("P0_ASSET_MISMATCH: formal registry")
    if manifest_sha != FROZEN_MANIFEST_SHA256:
        raise RuntimeError("P0_ASSET_MISMATCH: frozen manifest")
    if reviewer_registry_sha != REVIEWER_REGISTRY_BEFORE_SHA256:
        raise RuntimeError("P0_ASSET_MISMATCH: reviewer-risk registry before")
    if formal_output_sha != FORMAL_OUTPUT_BEFORE_SHA256:
        raise RuntimeError("P0_ASSET_MISMATCH: formal output tree")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    asset_report = verify_manifest(
        manifest, PROJECT_ROOT, asset_root, verify_external=True
    )
    if asset_report["status"] != "PASS":
        raise RuntimeError("P0_ASSET_MISMATCH: " + ",".join(asset_report["failed_assets"]))

    smoke = (torch.tensor([2.0, 3.0], device="cuda").square()).sum()
    torch.cuda.synchronize()
    if float(smoke) != 13.0:
        raise RuntimeError("P0_GPU_PREFLIGHT_FAIL: CUDA tensor smoke")

    attempt = output_root / "audits/seed_preflight/attempt_001"
    attempt.mkdir(parents=True, exist_ok=True)
    os.environ["CANONDRESSGS_ASSET_ROOT"] = str(asset_root)
    context = _build_context(attempt)
    cache_path = formal_output_root / "shared_preflight/frozen_reference_feature_rows_v1.pt"
    if not cache_path.is_file():
        raise RuntimeError("P0_ASSET_MISMATCH: frozen reference feature cache missing")
    cache, actual_cache_path, cache_sha, cache_seconds = formal.build_shared_feature_cache(
        context, formal_output_root
    )
    if actual_cache_path != cache_path:
        raise RuntimeError("P0_ASSET_MISMATCH: feature cache path")

    device = context["base"]._xyz.device
    batches = _feature_batches(cache, device)
    family_reports = {
        family: _audit_family(family, batches[family], device)
        for family in (
            "ours_v2", "b6_reference_classifier", "complex_fusion_corrected"
        )
    }
    failed_families = [
        name for name, report in family_reports.items()
        if report["status"] != "SEED-PROPAGATION-PASS"
    ]
    status = "SEED-PROPAGATION-PASS" if not failed_families else "SEED-PROPAGATION-FAIL"
    report = {
        "schema_version": "canondressgs.paper.p0_seed_propagation_audit.v1",
        "task_id": TASK_ID,
        "status": status,
        "hard_gate": True,
        "failure_action": (
            None if not failed_families else {
                "stop_all_p0_training": True,
                "p0_trainable_adapter_optimizer_created": False,
                "next_task": "REPAIR_PAPER_SEED_PROTOCOL_AND_RERUN_TRAINABLE_GROUPS",
                "silent_repair_forbidden": True,
            }
        ),
        "failed_families": failed_families,
        "run_commit": head,
        "branch": branch,
        "worktree_clean_before_audit": not bool(dirty),
        "preflight": {
            "gpu_cuda": {
                "cuda_available": True,
                "device_name": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_tensor_smoke": float(smoke),
            },
            "registry": _registry_summary(registry_path),
            "formal_registry_sha256": formal_registry_sha,
            "formal_output_metadata_sha256": formal_output_sha,
            "reviewer_risk_registry_before_sha256": reviewer_registry_sha,
            "frozen_asset_manifest_sha256": manifest_sha,
            "frozen_asset_verification": {
                "status": asset_report["status"],
                "asset_count": asset_report["asset_count"],
                "failed_assets": asset_report["failed_assets"],
            },
            "train_outfits": list(TRAIN_OUTFITS),
            "held_out_O07_used_for_training": False,
            "reserve_O06_used": False,
            "strict_view_status": "BLOCKED_PENDING_AUTHORIZATION",
            "view_classification": "VIEW-TRANSDUCTIVE",
            "feature_cache_path": str(cache_path),
            "feature_cache_sha256": cache_sha,
            "feature_cache_load_seconds": cache_seconds,
        },
        "p0_trainable_adapter_optimizer_created_before_gate": False,
        "families": family_reports,
        "conclusion": (
            "All audited trainable adapters vary initialization by seed."
            if not failed_families
            else "At least one audited trainable adapter has identical initialization for seeds 0/1/2; the P0 hard gate failed before any P0 adapter optimizer was created."
        ),
        "created_at_unix": time.time(),
    }
    audit_path = output_root / "audits/seed_propagation_audit.json"
    _atomic_json(audit_path, report)
    print(json.dumps({
        "status": status,
        "failed_families": failed_families,
        "audit": str(audit_path),
        "p0_trainable_adapter_optimizer_created": False,
    }, sort_keys=True))
    if status != "SEED-PROPAGATION-PASS":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
