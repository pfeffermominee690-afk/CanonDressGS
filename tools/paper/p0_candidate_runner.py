from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

from scene.gaussian_clothing_residuals import CHANNELS
from scene.p0_candidate_adapters import (
    FIXED_VIEWS,
    PREDICTION_FORBIDDEN_INPUTS,
    SEEN_OUTFITS,
    B6ReferenceClassifierHardLookupAdapter,
    OursV2CandidateAdapter,
    build_b7_fold_adapter,
    build_candidate_optimizer,
    build_m3_m4_candidate_adapters,
    candidate_parameter_manifest,
)
from scene.p0_candidate_initialization_protocol import (
    selected_state_sha256,
    tensor_mapping_sha256,
)
from tools import run_multi_outfit_explicit_basis as multi
from tools.paper.verify_seen_outfit_paper_assets import verify_manifest


TASK_ID = "AAAI27-P0-CANDIDATE-ADAPTERS-RUNNER-001"
SOURCE_HEAD = "42a28b386f6f32e23e5e16680408820efd8a65c7"
BRANCH = "paper/aaai27-p0-candidate-adapters-runner-20260721"
FORMAL_REGISTRY_SHA256 = "1834597b787d98acf475b352b791f0f16714fd870d51be36259ac7383a406c5e"
REVIEWER_REGISTRY_SHA256 = "138b4e1fb1a275944117d8f5e170349e12b8b6d442aebaa05b2b8ab149a1b63d"
FROZEN_MANIFEST_SHA256 = "ff90540e56c9c2db3fd6a1e45c31a1d1eb5a8a32effabde71158584d9be538bb"
OLD_SEED_AUDIT_SHA256 = "0b2e068203031b34e0004725923d452e185f319f3e9088843ecbc42ba5307f14"
FORMAL_OUTPUT_SHA256 = "7b9449e03d53e29cff11ded1fdc95e633640d38754cf152f2ab07a935d89d6bc"
FORMAL_FILE_COUNT = 4127
FORMAL_TOTAL_BYTES = 964043888
PREVIOUS_PROTOCOL_OUTPUT_SHA256 = "642cd8fa42b5879ae6a212941272ea24aec8ed6eb7e80ba5d8d2f74d4cf10960"
PREVIOUS_PROTOCOL_OUTPUT_FILE_COUNT = 12
PREVIOUS_PROTOCOL_OUTPUT_TOTAL_BYTES = 306862
DETERMINISTIC_PROTOCOL_ARTIFACT_SHA256 = "2769f8fb6261b36566bf5ca09905d28062082adec329f9dc5b39768f7a138757"
DETERMINISTIC_PROTOCOL_ARTIFACTS = (
    "docs/PAPER/AAAI27_DETERMINISTIC_INITIALIZATION_PROTOCOL_ADJUDICATION_20260721.md",
    "docs/PAPER/AAAI27_HISTORICAL_THREE_SEED_INTERPRETATION_ADDENDUM_20260721.md",
    "docs/PAPER/AAAI27_MEAN_GARMENT_ZERO_RESIDUAL_SEMANTICS_20260721.md",
    "paper_protocol/reviewer_risk/deterministic_initialization_contract.json",
    "paper_protocol/reviewer_risk/final_adjudication.json",
    "paper_protocol/reviewer_risk/historical_three_seed_interpretation_audit.json",
    "paper_protocol/reviewer_risk/initialization_policy_aware_audit.json",
    "paper_protocol/reviewer_risk/m1_m4_initialization_fairness_audit.json",
    "paper_protocol/reviewer_risk/mean_garment_zero_residual_semantics_audit.json",
    "paper_protocol/reviewer_risk/no_training_gate.json",
    "paper_protocol/reviewer_risk/optimizer_provenance_audit.json",
)
MANIFEST_PATH = PROJECT_ROOT / "paper_protocol/reviewer_risk/p0_candidate_runtime_manifest.yaml"


def _sha256(path: Path, *, normalize_lf: bool = False) -> str:
    data = path.read_bytes()
    if normalize_lf:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=PROJECT_ROOT, text=True
    ).strip()


def _tree_metadata_sha256(path: Path) -> str:
    command = (
        "cd " + shlex.quote(str(path))
        + " && find . -type f -printf '%P %s %T@\\n' | LC_ALL=C sort | sha256sum"
    )
    return subprocess.check_output(["bash", "-lc", command], text=True).split()[0]


def _tree_total_bytes(path: Path) -> int:
    return int(subprocess.check_output(["du", "-sb", str(path)], text=True).split()[0])


def deterministic_protocol_artifact_sha256() -> str:
    digest = hashlib.sha256()
    for relative in DETERMINISTIC_PROTOCOL_ARTIFACTS:
        path = PROJECT_ROOT / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path, normalize_lf=True)))
    return digest.hexdigest()


def load_runtime_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_runtime_manifest(payload)
    return payload


def validate_runtime_manifest(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != "canondressgs.paper.p0_candidate_runtime_manifest.v1":
        raise ValueError("P0 candidate runtime manifest schema mismatch")
    rows = list(payload.get("candidates", []))
    if len(rows) != 13 or payload.get("candidate_count") != 13:
        raise ValueError("P0 runtime manifest must contain exactly 13 candidates")
    if sum(bool(row["trainable"]) for row in rows) != 12:
        raise ValueError("P0 runtime manifest must contain 12 trainable candidates")
    if sum(not bool(row["trainable"]) for row in rows) != 1:
        raise ValueError("P0 runtime manifest must contain one non-trainable candidate")
    if len({row["runtime_id"] for row in rows}) != 13:
        raise ValueError("P0 runtime IDs must be unique")
    if len({row["reviewer_risk_experiment_id"] for row in rows}) != 13:
        raise ValueError("reviewer-risk experiment references must be unique")
    expected_methods = {"Ours-v2": 3, "B6": 3, "B7": 1, "M3": 3, "M4": 3}
    actual_methods = {
        method: sum(row["method"] == method for row in rows)
        for method in expected_methods
    }
    if actual_methods != expected_methods:
        raise ValueError(f"P0 runtime method counts mismatch: {actual_methods}")
    required = {
        "runtime_id", "reviewer_risk_experiment_id", "method", "replicate_index",
        "seed", "initialization_policy", "trainable", "optimizer_policy",
        "loss_contract", "reference_input_contract", "forbidden_inputs",
        "output_root", "output_subdir", "status", "source_head",
        "frozen_asset_manifest", "evaluator_contract", "formal_run_authorized",
    }
    allowed = set(payload["allowed_statuses"])
    forbidden = set(payload["forbidden_statuses"])
    for row in rows:
        if not required.issubset(row):
            raise ValueError(f"runtime entry missing fields: {row.get('runtime_id')}")
        if row["status"] not in allowed or row["status"] in forbidden:
            raise ValueError(f"runtime entry uses a forbidden status: {row['runtime_id']}")
        if row["source_head"] != SOURCE_HEAD or row["formal_run_authorized"] is not False:
            raise ValueError(f"runtime source/authorization mismatch: {row['runtime_id']}")
        if set(row["forbidden_inputs"]) != {
            "target_rgb", "target_mask", "target_pose", "target_camera",
            "ground_truth_coefficient", "teacher_residual", "outfit_id",
        }:
            raise ValueError(f"runtime forward boundary mismatch: {row['runtime_id']}")
    ours = [row for row in rows if row["method"] == "Ours-v2"]
    if [row["replicate_index"] for row in ours] != [0, 1, 2]:
        raise ValueError("Ours-v2 deterministic replicate indices must be 0/1/2")
    b7 = next(row for row in rows if row["method"] == "B7")
    if b7["trainable"] or b7["optimizer_policy"] != "NONE" or b7["seed"] is not None:
        raise ValueError("B7 must be fixed and non-trainable")


def _preflight(
    *, formal_root: Path, asset_root: Path, previous_protocol_root: Path
) -> dict[str, Any]:
    if _git("branch", "--show-current") != BRANCH:
        raise RuntimeError("P0 candidate runner requires its isolated branch")
    if _git("status", "--short"):
        raise RuntimeError("P0 candidate runner requires a clean worktree")
    if subprocess.call(
        ["git", "merge-base", "--is-ancestor", SOURCE_HEAD, "HEAD"], cwd=PROJECT_ROOT
    ) != 0:
        raise RuntimeError("P0 candidate runner source ancestry mismatch")
    formal_registry = PROJECT_ROOT / "paper_protocol/experiment_registry.yaml"
    reviewer_registry = PROJECT_ROOT / "paper_protocol/reviewer_risk/reviewer_risk_experiment_registry.yaml"
    frozen_manifest = PROJECT_ROOT / "paper_protocol/frozen_asset_manifest.json"
    old_seed = PROJECT_ROOT / "paper_protocol/reviewer_risk/seed_propagation_audit.json"
    hashes = {
        "formal_registry_sha256": _sha256(formal_registry),
        "reviewer_registry_sha256": _sha256(reviewer_registry),
        "frozen_manifest_sha256": _sha256(frozen_manifest),
        "old_seed_audit_sha256": _sha256(old_seed),
        "deterministic_protocol_artifacts_sha256": deterministic_protocol_artifact_sha256(),
        "formal_output_metadata_sha256": _tree_metadata_sha256(formal_root),
        "previous_protocol_output_metadata_sha256": _tree_metadata_sha256(previous_protocol_root),
    }
    expected = {
        "formal_registry_sha256": FORMAL_REGISTRY_SHA256,
        "reviewer_registry_sha256": REVIEWER_REGISTRY_SHA256,
        "frozen_manifest_sha256": FROZEN_MANIFEST_SHA256,
        "old_seed_audit_sha256": OLD_SEED_AUDIT_SHA256,
        "deterministic_protocol_artifacts_sha256": DETERMINISTIC_PROTOCOL_ARTIFACT_SHA256,
        "formal_output_metadata_sha256": FORMAL_OUTPUT_SHA256,
        "previous_protocol_output_metadata_sha256": PREVIOUS_PROTOCOL_OUTPUT_SHA256,
    }
    formal_count = sum(path.is_file() for path in formal_root.rglob("*"))
    formal_bytes = _tree_total_bytes(formal_root)
    previous_count = sum(path.is_file() for path in previous_protocol_root.rglob("*"))
    previous_bytes = _tree_total_bytes(previous_protocol_root)
    if (
        hashes != expected
        or formal_count != FORMAL_FILE_COUNT
        or formal_bytes != FORMAL_TOTAL_BYTES
        or previous_count != PREVIOUS_PROTOCOL_OUTPUT_FILE_COUNT
        or previous_bytes != PREVIOUS_PROTOCOL_OUTPUT_TOTAL_BYTES
    ):
        raise RuntimeError("P0-CANDIDATE-ASSET-MISMATCH")
    manifest_payload = json.loads(frozen_manifest.read_text(encoding="utf-8"))
    verification = verify_manifest(
        manifest_payload, PROJECT_ROOT, asset_root, verify_external=True
    )
    if verification["status"] != "PASS":
        raise RuntimeError("P0-CANDIDATE-ASSET-MISMATCH")
    formal_payload = yaml.safe_load(formal_registry.read_text(encoding="utf-8"))
    reviewer_payload = yaml.safe_load(reviewer_registry.read_text(encoding="utf-8"))
    paper_final = sum(
        row.get("status") == "PAPER_FINAL"
        for row in (*formal_payload["experiments"], *reviewer_payload["experiments"])
    )
    if paper_final != 0:
        raise RuntimeError("P0-CANDIDATE-ASSET-MISMATCH: PAPER_FINAL")
    smoke = torch.tensor([2.0, 3.0], device="cuda").square().sum()
    torch.cuda.synchronize()
    if float(smoke) != 13.0:
        raise RuntimeError("P0-CANDIDATE-ASSET-MISMATCH: CUDA smoke")
    return {
        "status": "PASS",
        "task_id": TASK_ID,
        "source_head": SOURCE_HEAD,
        "run_commit": _git("rev-parse", "HEAD"),
        "branch": BRANCH,
        "upstream": _upstream_or_none(),
        "cuda": {
            "available": True,
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "python": sys.version.split()[0],
            "tensor_smoke": float(smoke),
        },
        **hashes,
        "formal_output_file_count": formal_count,
        "formal_output_total_bytes": formal_bytes,
        "previous_protocol_output_file_count": previous_count,
        "previous_protocol_output_total_bytes": previous_bytes,
        "frozen_asset_verification": {
            "status": verification["status"],
            "asset_count": verification["asset_count"],
            "failed_assets": verification["failed_assets"],
        },
        "paper_final_count": paper_final,
    }


def _upstream_or_none() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "@{upstream}"], cwd=PROJECT_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _next_attempt(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    indices = [
        int(path.name.split("_", 1)[1])
        for path in root.glob("attempt_*")
        if path.name.split("_", 1)[1].isdigit()
    ]
    attempt = root / f"attempt_{max(indices, default=0) + 1:03d}"
    attempt.mkdir(parents=False, exist_ok=False)
    return attempt


def _float_losses(values: Mapping[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(value.detach()) for name, value in values.items()}


def _basis_assets(asset_root: Path, device: torch.device) -> tuple[Any, dict[str, torch.Tensor], dict[str, Any], Path]:
    path = (
        asset_root
        / "pipeline_full/SUBJECT02-MULTI-OUTFIT-EXPLICIT-BASIS-001/attempt_003"
        / "stage_b_basis/selected/selected_basis.pt"
    )
    if _sha256(path) != "a29b3dc3c6f0ac1a79755e036eb4196dc9e4d116f3e2b35a7ed4e2bd5286f430":
        raise RuntimeError("P0-CANDIDATE-ASSET-MISMATCH: selected basis")
    basis, coefficients, payload = multi.load_basis_artifact(path, device)
    matrix = torch.stack([coefficients[outfit] for outfit in SEEN_OUTFITS])
    payload = dict(payload)
    payload["coefficient_train_mean"] = matrix.mean(0)
    payload["coefficient_train_std"] = matrix.std(0, unbiased=False)
    return basis, coefficients, payload, path


def _feature_rows(
    cache: Mapping[str, Any], outfit: str, condition: str, kind: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    row = cache["episodes"][f"{outfit}/{condition}"]["normal"]
    return row[kind].to(device), row["valid"].to(device)


def _teacher_asset_bank(frozen_manifest: Mapping[str, Any]) -> dict[str, Any]:
    assets = {row["asset_id"]: row for row in frozen_manifest["assets"]}
    return {
        outfit: {
            "asset_id": f"teacher_checkpoint_{outfit}",
            "sha256": assets[f"teacher_checkpoint_{outfit}"]["fingerprint"],
            "role": "FROZEN_TEACHER_RESIDUAL_LOOKUP_ONLY",
        }
        for outfit in SEEN_OUTFITS
    }


def _reference_checksum_index(reference_manifest: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for outfit in reference_manifest["outfits"]:
        name = outfit["outfit_id"]
        if name not in SEEN_OUTFITS:
            continue
        index[name] = {
            observation["condition_id"]: observation["checksums"]["target_edit_rgb"]
            for observation in outfit["observations"]
        }
    if set(index) != set(SEEN_OUTFITS) or any(set(row) != set(FIXED_VIEWS) for row in index.values()):
        raise RuntimeError("P0-CANDIDATE-ASSET-MISMATCH: reference checksum index")
    return index


def _residual_record(residual: Any) -> dict[str, Any]:
    mapping = residual.as_dict()
    tensors = {name: mapping[name] for name in CHANNELS}
    if any(value is None for value in tensors.values()):
        raise ValueError("basis reconstruction omitted a residual channel")
    exact = {name: value for name, value in tensors.items() if value is not None}
    return {
        "sha256": tensor_mapping_sha256(exact),
        "shapes": {name: list(value.shape) for name, value in exact.items()},
        "dtype": str(next(iter(exact.values())).dtype),
        "device": str(next(iter(exact.values())).device),
    }


def _optimizer_report(
    module: torch.nn.Module, basis: torch.nn.Module
) -> tuple[torch.optim.Optimizer | None, dict[str, Any]]:
    optimizer, report = build_candidate_optimizer(module)
    candidate_ids = {id(value) for value in module.parameters() if value.requires_grad}
    frozen_ids = {id(value) for value in basis.parameters()} | {id(value) for value in basis.buffers()}
    optimizer_ids = set()
    if optimizer is not None:
        optimizer_ids = {
            id(value) for group in optimizer.param_groups for value in group["params"]
        }
    report.update({
        "candidate_parameter_membership_complete": optimizer_ids == candidate_ids,
        "candidate_frozen_overlap_count": len(candidate_ids & frozen_ids),
        "optimizer_frozen_overlap_count": len(optimizer_ids & frozen_ids),
        "candidate_legacy_overlap_count": 0,
        "initial_state_entry_count": 0 if optimizer is None else len(optimizer.state),
    })
    return optimizer, report


def _coefficient_step0(
    *,
    entry: Mapping[str, Any],
    adapter: torch.nn.Module,
    feature_kind: str,
    cache: Mapping[str, Any],
    condition: str,
    targets: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    basis: torch.nn.Module,
    basis_path: Path,
    counterpart_trunk_sha256: str | None = None,
) -> dict[str, Any]:
    outputs = []
    inputs: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for outfit in SEEN_OUTFITS:
            rows, valid = _feature_rows(cache, outfit, condition, feature_kind, mean.device)
            result = adapter(rows, valid)
            outputs.append(result.standardized_coefficients)
            inputs[f"{outfit}/features"] = rows
            inputs[f"{outfit}/valid"] = valid
        predicted = torch.stack(outputs)
        losses = adapter.training_loss(predicted, targets)
        raw = predicted * std + mean
        representative_residual = basis(raw[0], chunk_size=16384)
    optimizer, optimizer_report = _optimizer_report(adapter, basis)
    parameter = candidate_parameter_manifest(adapter)
    result = {
        "output_kind": "STANDARDIZED_COEFFICIENTS",
        "fixed_condition": condition,
        "fixed_first_batch_sha256": tensor_mapping_sha256(inputs),
        "predicted_standardized_coefficients": predicted.detach().cpu().tolist(),
        "predicted_standardized_sha256": tensor_mapping_sha256({"prediction": predicted}),
        "raw_coefficients": raw.detach().cpu().tolist(),
        "raw_coefficient_sha256": tensor_mapping_sha256({"raw": raw}),
        "basis_reconstruction": _residual_record(representative_residual),
        "basis_sha256": _sha256(basis_path),
        "semantic_classification": "MEAN-GARMENT INITIAL PREDICTION",
        "initial_output_all_zero": bool(torch.count_nonzero(predicted) == 0),
        "step0_loss_no_backward": _float_losses(losses),
        "candidate": parameter,
        "candidate_optimizer": optimizer_report,
        "legacy_context_optimizer": {
            "created": False, "class": None, "groups": [], "parameter_count": 0,
            "parameter_names": [], "candidate_overlap_count": 0,
            "frozen_overlap_count": 0, "zero_grad_count": 0, "step_count": 0,
            "lifecycle": "not constructed by the feature-cache/basis-only candidate dry-run",
            "discarded": False, "state_saved": False,
        },
    }
    if entry["method"] in {"M3", "M4"}:
        result["trunk_sha256"] = selected_state_sha256(adapter.candidate, "trunk")
        result["output_head_sha256"] = selected_state_sha256(adapter.candidate, "output_head")
        result["same_seed_counterpart_trunk_sha256"] = counterpart_trunk_sha256
        result["same_seed_trunk_parity"] = result["trunk_sha256"] == counterpart_trunk_sha256
    del optimizer
    return result


def _b6_step0(
    *, entry: Mapping[str, Any], adapter: B6ReferenceClassifierHardLookupAdapter,
    cache: Mapping[str, Any], teacher_bank: Mapping[str, Any], basis: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    outputs = []
    inputs: dict[str, torch.Tensor] = {}
    selections = []
    with torch.no_grad():
        for outfit in SEEN_OUTFITS:
            rows, valid = _feature_rows(cache, outfit, FIXED_VIEWS[0], "f2", device)
            prediction = adapter(rows, valid)
            outputs.append(prediction.logits)
            selected = adapter.lookup_predicted_teacher(prediction, teacher_bank)
            selections.append({
                "query_outfit_used_only_for_batch_index": outfit,
                "predicted_class": prediction.predicted_class,
                "predicted_outfit": prediction.predicted_outfit,
                "selected_teacher_residual_asset": selected,
            })
            inputs[f"{outfit}/features"] = rows
            inputs[f"{outfit}/valid"] = valid
        logits = torch.stack(outputs)
        labels = torch.arange(len(SEEN_OUTFITS), device=device)
        losses = adapter.training_loss(logits, labels)
    optimizer, optimizer_report = _optimizer_report(adapter, basis)
    result = {
        "output_kind": "FIVE_CLASS_LOGITS_AND_PREDICTED_CLASS_HARD_LOOKUP",
        "fixed_condition": FIXED_VIEWS[0],
        "fixed_first_batch_sha256": tensor_mapping_sha256(inputs),
        "logits": logits.detach().cpu().tolist(),
        "logits_sha256": tensor_mapping_sha256({"logits": logits}),
        "selections": selections,
        "tie_handling": "torch.argmax_first_index_in_frozen_outfit_order",
        "class_order": list(SEEN_OUTFITS),
        "outfit_label_role": "CROSS_ENTROPY_LOSS_TARGET_ONLY",
        "hard_lookup_key": "PREDICTED_CLASS_ONLY",
        "confusion_matrix_schema": adapter.confusion_matrix_schema(),
        "step0_loss_no_backward": _float_losses(losses),
        "candidate": candidate_parameter_manifest(adapter),
        "candidate_optimizer": optimizer_report,
        "legacy_context_optimizer": {
            "created": False, "class": None, "groups": [], "parameter_count": 0,
            "parameter_names": [], "candidate_overlap_count": 0,
            "frozen_overlap_count": 0, "zero_grad_count": 0, "step_count": 0,
            "lifecycle": "not constructed by the feature-cache/basis-only candidate dry-run",
            "discarded": False, "state_saved": False,
        },
    }
    del optimizer
    return result


def _b7_step0(
    *, cache: Mapping[str, Any], reference_checksums: Mapping[str, Mapping[str, str]],
    teacher_bank: Mapping[str, Any], basis: torch.nn.Module, device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    folds = {}
    queries = []
    input_tensors: dict[str, torch.Tensor] = {}
    for condition in FIXED_VIEWS:
        rows_by_outfit, valid_by_outfit, hashes = {}, {}, {}
        legal = [view for view in FIXED_VIEWS if view != condition]
        for outfit in SEEN_OUTFITS:
            rows, valid = _feature_rows(cache, outfit, condition, "f2", device)
            rows_by_outfit[outfit], valid_by_outfit[outfit] = rows, valid
            hashes[outfit] = [reference_checksums[outfit][view] for view in legal]
            input_tensors[f"{outfit}/{condition}/features"] = rows
            input_tensors[f"{outfit}/{condition}/valid"] = valid
        adapter, manifest = build_b7_fold_adapter(
            target_condition=condition,
            fold_rows=rows_by_outfit,
            fold_validity=valid_by_outfit,
            reference_file_hashes=hashes,
        )
        folds[condition] = manifest
        with torch.no_grad():
            for outfit in SEEN_OUTFITS:
                prediction = adapter(rows_by_outfit[outfit], valid_by_outfit[outfit])
                selected = adapter.lookup_nearest_teacher(prediction, teacher_bank)
                queries.append({
                    "fold": condition,
                    "query_outfit_used_only_for_episode_index": outfit,
                    "query_sha256": tensor_mapping_sha256({"query": prediction.query_feature}),
                    "per_class_squared_distances": dict(prediction.squared_distances),
                    "nearest_outfit": prediction.predicted_outfit,
                    "selected_teacher_residual_asset": selected,
                    "target_exclusion_proof": manifest["target_condition_excluded"],
                })
    adapter_for_optimizer, _ = build_b7_fold_adapter(
        target_condition=FIXED_VIEWS[0],
        fold_rows={outfit: _feature_rows(cache, outfit, FIXED_VIEWS[0], "f2", device)[0] for outfit in SEEN_OUTFITS},
        fold_validity={outfit: _feature_rows(cache, outfit, FIXED_VIEWS[0], "f2", device)[1] for outfit in SEEN_OUTFITS},
        reference_file_hashes={
            outfit: [reference_checksums[outfit][view] for view in FIXED_VIEWS[1:]]
            for outfit in SEEN_OUTFITS
        },
    )
    optimizer, optimizer_report = _optimizer_report(adapter_for_optimizer, basis)
    if optimizer is not None:
        raise RuntimeError("B7 unexpectedly created a candidate optimizer")
    return ({
        "output_kind": "NEAREST_CENTROID_PREDICTED_OUTFIT_HARD_LOOKUP",
        "fixed_first_batch_sha256": tensor_mapping_sha256(input_tensors),
        "fold_count": len(folds),
        "query_count": len(queries),
        "queries": queries,
        "distance_metric": "FEATURE_STANDARDIZATION_THEN_SQUARED_L2",
        "tie_handling": "REGISTERED_SEEN_OUTFIT_ORDER",
        "outfit_label_role": "OFFLINE_CENTROID_GROUPING_ONLY",
        "candidate": candidate_parameter_manifest(adapter_for_optimizer),
        "candidate_optimizer": optimizer_report,
        "legacy_context_optimizer": {
            "created": False, "class": None, "groups": [], "parameter_count": 0,
            "parameter_names": [], "candidate_overlap_count": 0,
            "frozen_overlap_count": 0, "zero_grad_count": 0, "step_count": 0,
            "lifecycle": "not constructed by the feature-cache-only B7 dry-run",
            "discarded": False, "state_saved": False,
        },
    }, {
        "schema_version": "canondressgs.paper.p0_b7_all_fold_centroids.v1",
        "status": "PASS",
        "folds": folds,
        "query_count": len(queries),
        "target_rgb_used": False,
        "target_mask_used": False,
        "all_target_conditions_excluded": all(
            row["target_condition_excluded"] for row in folds.values()
        ),
    })


def _dry_run_entry(
    *, entry: Mapping[str, Any], adapter: torch.nn.Module | None,
    paired: Mapping[int, tuple[torch.nn.Module, torch.nn.Module]],
    cache: Mapping[str, Any], targets: torch.Tensor, mean: torch.Tensor,
    std: torch.Tensor, basis: torch.nn.Module, basis_path: Path,
    teacher_bank: Mapping[str, Any], reference_checksums: Mapping[str, Mapping[str, str]],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    method = entry["method"]
    centroid_manifest = None
    if method == "Ours-v2":
        step0 = _coefficient_step0(
            entry=entry, adapter=adapter, feature_kind="f2", cache=cache,
            condition=FIXED_VIEWS[0], targets=targets, mean=mean, std=std,
            basis=basis, basis_path=basis_path,
        )
    elif method == "B6":
        step0 = _b6_step0(
            entry=entry, adapter=adapter, cache=cache, teacher_bank=teacher_bank,
            basis=basis, device=device,
        )
    elif method == "B7":
        step0, centroid_manifest = _b7_step0(
            cache=cache, reference_checksums=reference_checksums,
            teacher_bank=teacher_bank, basis=basis, device=device,
        )
    elif method in {"M3", "M4"}:
        seed = int(entry["seed"])
        m3, m4 = paired[seed]
        current = m3 if method == "M3" else m4
        counterpart = m4 if method == "M3" else m3
        step0 = _coefficient_step0(
            entry=entry, adapter=current, feature_kind="rff", cache=cache,
            condition=FIXED_VIEWS[0], targets=targets, mean=mean, std=std,
            basis=basis, basis_path=basis_path,
            counterpart_trunk_sha256=selected_state_sha256(counterpart.candidate, "trunk"),
        )
    else:
        raise ValueError(f"unsupported P0 method: {method}")
    report = {
        "schema_version": "canondressgs.paper.p0_candidate_step0_dry_run.v1",
        "status": "DRY_RUN_PASS",
        "task_id": TASK_ID,
        "runtime_id": entry["runtime_id"],
        "reviewer_risk_experiment_id": entry["reviewer_risk_experiment_id"],
        "method": method,
        "replicate_index": entry["replicate_index"],
        "seed": entry["seed"],
        "initialization_policy": entry["initialization_policy"],
        "trainable": entry["trainable"],
        "optimizer_policy": entry["optimizer_policy"],
        "loss_contract": entry["loss_contract"],
        "reference_input_contract": entry["reference_input_contract"],
        "forward_boundary": {
            "allowed_inputs": [
                "reference_rgb", "reference_clothing_mask", "reference_foreground_mask",
                "frozen_reference_features", "reference_validity", "fixed_normalization",
            ],
            "forbidden_inputs": list(PREDICTION_FORBIDDEN_INPUTS),
            "target_forward_input_used": False,
            "outfit_id_in_prediction_forward": False,
            "target_pose_camera_may_enter_only_downstream_frozen_mmlp_renderer": True,
        },
        "step0": step0,
        "backward_count": 0,
        "candidate_optimizer_step_count": 0,
        "legacy_optimizer_step_count": 0,
        "scheduler_step_count": 0,
        "checkpoint_created": False,
        "render_created": False,
        "formal_metrics_created": False,
        "formal_run_authorized": False,
        "performance_conclusion_allowed": False,
    }
    return report, centroid_manifest


def run_dry_run(
    *, manifest: Mapping[str, Any], output_root: Path, formal_root: Path,
    asset_root: Path, previous_protocol_root: Path, device_name: str,
) -> dict[str, Any]:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("P0-CANDIDATE-ASSET-MISMATCH: CUDA unavailable")
    device = torch.device(device_name)
    preflight = _preflight(
        formal_root=formal_root, asset_root=asset_root,
        previous_protocol_root=previous_protocol_root,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    preflight_path = output_root / "preflight.json"
    if preflight_path.exists():
        preflight_path = output_root / f"preflight_{int(time.time())}.json"
    _atomic_json(preflight_path, preflight)
    feature_cache_path = formal_root / "shared_preflight/frozen_reference_feature_rows_v1.pt"
    cache = torch.load(feature_cache_path, map_location="cpu", weights_only=False)
    if cache.get("schema_version") != "canondressgs.paper_frozen_reference_rows.v1":
        raise RuntimeError("P0-CANDIDATE-ASSET-MISMATCH: feature cache")
    basis, coefficients, payload, basis_path = _basis_assets(asset_root, device)
    mean = payload["coefficient_train_mean"].to(device)
    std = payload["coefficient_train_std"].to(device)
    targets = torch.stack([(coefficients[outfit] - mean) / std for outfit in SEEN_OUTFITS])
    frozen_manifest = json.loads(
        (PROJECT_ROOT / "paper_protocol/frozen_asset_manifest.json").read_text(encoding="utf-8")
    )
    teacher_bank = _teacher_asset_bank(frozen_manifest)
    reference_manifest_path = (
        asset_root / "pipeline_full/SUBJECT02-AAAI27-DATA-CAPACITY-GATE-001/attempt_003"
        / "dataset/aaai_gate_28_manifest.json"
    )
    reference_manifest = json.loads(reference_manifest_path.read_text(encoding="utf-8"))
    reference_checksums = _reference_checksum_index(reference_manifest)
    raw_dim = int(cache["episodes"][f"{SEEN_OUTFITS[0]}/{FIXED_VIEWS[0]}"]["normal"]["rff"].shape[1])
    paired = {
        seed: build_m3_m4_candidate_adapters(raw_dim=raw_dim, seed=seed, device=device)
        for seed in (0, 1, 2)
    }
    adapters: dict[str, torch.nn.Module | None] = {}
    for entry in manifest["candidates"]:
        if entry["method"] == "Ours-v2":
            torch.manual_seed(int(entry["seed"]))
            adapters[entry["runtime_id"]] = OursV2CandidateAdapter().to(device)
        elif entry["method"] == "B6":
            adapters[entry["runtime_id"]] = B6ReferenceClassifierHardLookupAdapter(
                seed=int(entry["seed"])
            ).to(device)
        elif entry["method"] == "B7":
            adapters[entry["runtime_id"]] = None
        else:
            adapters[entry["runtime_id"]] = paired[int(entry["seed"])][
                0 if entry["method"] == "M3" else 1
            ]
    records = []
    for entry in manifest["candidates"]:
        attempt = _next_attempt(output_root / entry["output_subdir"])
        try:
            report, centroid = _dry_run_entry(
                entry=entry, adapter=adapters[entry["runtime_id"]], paired=paired,
                cache=cache, targets=targets, mean=mean, std=std, basis=basis,
                basis_path=basis_path, teacher_bank=teacher_bank,
                reference_checksums=reference_checksums, device=device,
            )
            report["attempt"] = attempt.name
            report["assets"] = {
                "feature_cache_sha256": _sha256(feature_cache_path),
                "basis_sha256": _sha256(basis_path),
                "reference_manifest_sha256": _sha256(reference_manifest_path),
                "frozen_backbone_sha256": cache["backbone_fingerprint"],
                "teacher_residual_assets": teacher_bank,
                "base_mmlp_renderer_frozen": True,
            }
            _atomic_json(attempt / "dry_run.json", report)
            if centroid is not None:
                _atomic_json(attempt / "centroid_construction_manifest.json", centroid)
            records.append({
                "runtime_id": entry["runtime_id"], "status": report["status"],
                "attempt": str(attempt), "method": entry["method"],
                "candidate_optimizer_created": report["step0"]["candidate_optimizer"]["created"],
                "candidate_parameter_count": report["step0"]["candidate"]["parameter_count"],
            })
        except Exception as error:
            _atomic_json(attempt / "failure.json", {
                "status": "FAILED", "runtime_id": entry["runtime_id"],
                "error_type": type(error).__name__, "error": str(error),
                "attempt_preserved": True,
            })
            raise
    summary = {
        "schema_version": "canondressgs.paper.p0_candidate_dry_run_summary.v1",
        "status": "PASS" if len(records) == 13 and all(row["status"] == "DRY_RUN_PASS" for row in records) else "FAIL",
        "task_id": TASK_ID,
        "run_commit": _git("rev-parse", "HEAD"),
        "candidate_count": len(records),
        "trainable_candidate_count": sum(row["candidate_optimizer_created"] for row in records),
        "non_trainable_candidate_count": sum(not row["candidate_optimizer_created"] for row in records),
        "records": records,
        "backward_count": 0,
        "candidate_optimizer_step_count": 0,
        "legacy_optimizer_step_count": 0,
        "scheduler_step_count": 0,
        "checkpoint_count": 0,
        "render_count": 0,
        "formal_metrics_count": 0,
        "formal_run_authorized": False,
        "paper_final_count": 0,
    }
    index = len(list(output_root.glob("dry_run_summary_*.json"))) + 1
    _atomic_json(output_root / f"dry_run_summary_{index:03d}.json", summary)
    if summary["status"] != "PASS":
        raise RuntimeError("P0-CANDIDATE-IMPLEMENTATION-TEST-FAIL")
    return summary


def runtime_plan(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "canondressgs.paper.p0_candidate_run_plan.v1",
        "status": "PREFLIGHT_READY",
        "candidate_count": 13,
        "trainable_count": 12,
        "non_trainable_count": 1,
        "planned_optimizer_steps_per_trainable_candidate": 300,
        "planned_total_optimizer_steps": 3600,
        "executed_optimizer_steps": 0,
        "formal_run_authorized": False,
        "candidates": [
            {
                key: row[key]
                for key in (
                    "runtime_id", "method", "replicate_index", "seed", "trainable",
                    "initialization_policy", "output_root", "status", "formal_run_authorized",
                )
            }
            for row in manifest["candidates"]
        ],
    }


def runtime_status(manifest: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    records = []
    for entry in manifest["candidates"]:
        attempts = sorted((output_root / entry["output_subdir"]).glob("attempt_*"))
        latest = attempts[-1] if attempts else None
        dry_run = latest / "dry_run.json" if latest is not None else None
        failure = latest / "failure.json" if latest is not None else None
        if dry_run is not None and dry_run.is_file():
            status, reason = json.loads(dry_run.read_text(encoding="utf-8"))["status"], None
        elif failure is not None and failure.is_file():
            payload = json.loads(failure.read_text(encoding="utf-8"))
            status, reason = "FAILED", payload["error"]
        else:
            status, reason = entry["status"], None
        records.append({
            "runtime_id": entry["runtime_id"], "implementation_status": "IMPLEMENTED",
            "dry_run_status": status, "formal_run_authorized": False,
            "latest_attempt": None if latest is None else latest.name,
            "failure_reason": reason,
        })
    return {
        "schema_version": "canondressgs.paper.p0_candidate_status.v1",
        "candidate_count": len(records),
        "records": records,
        "dry_run_pass_count": sum(row["dry_run_status"] == "DRY_RUN_PASS" for row in records),
        "formal_run_authorized": False,
    }


def _common_arguments(parser: argparse.ArgumentParser, *, output: bool) -> None:
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--formal-output-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--previous-protocol-root", type=Path, required=True)
    if output:
        parser.add_argument("--output-root", type=Path, required=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified no-training P0 candidate runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    validate_parser = subparsers.add_parser("validate")
    _common_arguments(validate_parser, output=False)
    dry_parser = subparsers.add_parser("dry-run")
    _common_arguments(dry_parser, output=True)
    dry_parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    status_parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = load_runtime_manifest(args.manifest)
    if args.command == "plan":
        result = runtime_plan(manifest)
    elif args.command == "validate":
        result = _preflight(
            formal_root=args.formal_output_root.resolve(),
            asset_root=args.asset_root.resolve(),
            previous_protocol_root=args.previous_protocol_root.resolve(),
        )
    elif args.command == "dry-run":
        result = run_dry_run(
            manifest=manifest, output_root=args.output_root.resolve(),
            formal_root=args.formal_output_root.resolve(),
            asset_root=args.asset_root.resolve(),
            previous_protocol_root=args.previous_protocol_root.resolve(),
            device_name=args.device,
        )
    elif args.command == "status":
        result = runtime_status(manifest, args.output_root.resolve())
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
