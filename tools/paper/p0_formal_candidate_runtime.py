from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import yaml

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

from scene.p0_candidate_adapters import (
    FIXED_VIEWS,
    SEEN_OUTFITS,
    B6ReferenceClassifierHardLookupAdapter,
    OursV2CandidateAdapter,
    P0ComplexCandidateAdapter,
    build_b7_fold_adapter,
    build_m3_m4_candidate_adapters,
    candidate_parameter_manifest,
)
from scene.p0_candidate_initialization_protocol import (
    seed_all,
    selected_state_sha256,
    tensor_mapping_sha256,
)
from tools import diagnose_image_conditioned_overfit_failure as diagnosis
from tools import run_image_conditioned_overfit_o01 as o01
from tools import run_multi_outfit_explicit_basis as multi
from tools import run_residual_field_parameterization as parameterization
from tools.check_real_image_conditioned_one_batch import save_render_tensor
from tools.paper import formal_batch_runtime as historical
from tools.paper import formal_runtime as core
from tools.paper.aggregation import aggregate_evaluations
from tools.paper.evaluate_seen_outfit import ALL_METRICS, evaluate_records, load_raw_records
from tools.paper.p0_candidate_runner import (
    _reference_checksum_index,
    _sha256,
    _tree_metadata_sha256,
    _tree_total_bytes,
    deterministic_protocol_artifact_sha256,
)
from tools.paper.verify_seen_outfit_paper_assets import verify_manifest


TASK_ID = "AAAI27-P0-FORMAL-CANDIDATE-RUNS-001"
SOURCE_HEAD = "d75c4fa5a426fe90314eba08e94509940ecafb23"
RUN_BRANCH = "paper/aaai27-p0-formal-candidate-runs-20260721"
REGISTRY_PATH = PROJECT_ROOT / "paper_protocol/reviewer_risk/p0_formal_candidate_run_registry.yaml"
CANDIDATE_MANIFEST_PATH = PROJECT_ROOT / "paper_protocol/reviewer_risk/p0_candidate_runtime_manifest.yaml"
TRAINING_AUDIT_PATH = PROJECT_ROOT / "paper_protocol/reviewer_risk/p0_formal_training_contract_audit.json"
TRUNK_AUDIT_PATH = PROJECT_ROOT / "paper_protocol/reviewer_risk/m3_m4_complex_trunk_provenance.json"
M4_PARITY_PATH = PROJECT_ROOT / "paper_protocol/reviewer_risk/m4_a5_supervision_contract_parity.json"
FORMAL_REGISTRY_SHA256 = "1834597b787d98acf475b352b791f0f16714fd870d51be36259ac7383a406c5e"
REVIEWER_REGISTRY_SHA256 = "138b4e1fb1a275944117d8f5e170349e12b8b6d442aebaa05b2b8ab149a1b63d"
FROZEN_MANIFEST_SHA256 = "ff90540e56c9c2db3fd6a1e45c31a1d1eb5a8a32effabde71158584d9be538bb"
CANDIDATE_MANIFEST_SHA256 = "0501168192c9707fa0ea4f49d3872b979b5106b183f54e4542daa1c6341eee82"
CANDIDATE_CODE_AGGREGATE_SHA256 = "92eaf05af2f8661ab29354d671b63f467c0be111825a324422a0e81952c57da8"
M4_PARITY_SHA256 = "fee0fbda7e67d3a41d1442304e661d7853241ced1226489ed4a47940a276363c"
HISTORICAL_FORMAL_RUNTIME_SHA256 = "75211429a628801882d211805a2334fc88c11f6654cfbc4b6cc47fa2906693a6"
FORMAL_TREE = (4127, 964043888, "7b9449e03d53e29cff11ded1fdc95e633640d38754cf152f2ab07a935d89d6bc")
DETERMINISTIC_TREE = (12, 306862, "642cd8fa42b5879ae6a212941272ea24aec8ed6eb7e80ba5d8d2f74d4cf10960")
SMOKE_TREE = (16, 151001, "79f2cb3c8a3b7c8a3869c179a402404798069b53181f33a1b7fb438c611f7117")
REVIEWER_RISK_TREE = (1, 31115, "fd27cb70fc1d3168a74f888e37379e8ec66cadf46c99f76ecdacb77a1b1e3a3c")
DETERMINISTIC_ARTIFACT_SHA256 = "2769f8fb6261b36566bf5ca09905d28062082adec329f9dc5b39768f7a138757"
MILESTONES = (0, 20, 50, 100, 200, 300)
OPTIMIZER_STEPS = 300
CONDITIONS = tuple(FIXED_VIEWS)
OUTFITS = tuple(SEEN_OUTFITS)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=PROJECT_ROOT, text=True
    ).strip()


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def candidate_code_aggregate_sha256() -> str:
    digest = hashlib.sha256()
    for relative in (
        "scene/p0_candidate_initialization_protocol.py",
        "scene/p0_candidate_adapters.py",
        "tools/paper/p0_candidate_runner.py",
    ):
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(_sha256(PROJECT_ROOT / relative).encode("ascii") + b"\n")
    return digest.hexdigest()


def tree_record(path: Path) -> dict[str, Any]:
    return {
        "file_count": sum(1 for item in path.rglob("*") if item.is_file()),
        "total_bytes": _tree_total_bytes(path),
        "metadata_sha256": _tree_metadata_sha256(path),
    }


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_registry(payload)
    return payload


def validate_registry(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != "canondressgs.paper.p0_formal_candidate_run_registry.v1":
        raise ValueError("formal candidate registry schema mismatch")
    rows = list(payload.get("runs", []))
    if len(rows) != 13 or len({row["formal_run_id"] for row in rows}) != 13:
        raise ValueError("formal candidate registry requires 13 unique runs")
    expected = {"Ours-v2": 3, "B6": 3, "B7": 1, "M3": 3, "M4": 3}
    actual = {name: sum(row["method"] == name for row in rows) for name in expected}
    if actual != expected:
        raise ValueError(f"formal candidate method counts mismatch: {actual}")
    allowed = set(payload["allowed_statuses"])
    for row in rows:
        if row["status"] not in allowed or row["status"] == "PAPER_FINAL":
            raise ValueError("formal registry contains a forbidden status")
        if not row["formal_run_authorized"]:
            raise ValueError("formal registry contains an unauthorized run")
        if row["expected_steps"] != (0 if row["method"] == "B7" else 300):
            raise ValueError("formal registry step budget mismatch")
    if payload.get("paper_final") is not False:
        raise ValueError("formal candidate registry cannot be PAPER_FINAL")


def _assert_tree(name: str, path: Path, expected: tuple[int, int, str]) -> dict[str, Any]:
    actual = tree_record(path)
    expected_record = {
        "file_count": expected[0], "total_bytes": expected[1],
        "metadata_sha256": expected[2],
    }
    if actual != expected_record:
        raise RuntimeError(f"P0-FORMAL-ASSET-MISMATCH: {name}")
    return actual


def preflight(*, asset_root: Path, formal_root: Path) -> dict[str, Any]:
    if git("branch", "--show-current") != RUN_BRANCH:
        raise RuntimeError("formal P0 executor requires its isolated branch")
    dirty = [line for line in git("status", "--porcelain").splitlines() if line]
    if dirty:
        raise RuntimeError("formal P0 executor requires a clean worktree")
    if subprocess.call(
        ["git", "merge-base", "--is-ancestor", SOURCE_HEAD, "HEAD"], cwd=PROJECT_ROOT
    ) != 0:
        raise RuntimeError("formal P0 source ancestry mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("P0-FORMAL-ASSET-MISMATCH: CUDA unavailable")
    if sha256_lf(PROJECT_ROOT / "paper_protocol/experiment_registry.yaml") != FORMAL_REGISTRY_SHA256:
        raise RuntimeError("P0-FORMAL-ASSET-MISMATCH: formal registry")
    if sha256_lf(PROJECT_ROOT / "paper_protocol/reviewer_risk/reviewer_risk_experiment_registry.yaml") != REVIEWER_REGISTRY_SHA256:
        raise RuntimeError("P0-FORMAL-ASSET-MISMATCH: reviewer registry")
    frozen_manifest_path = PROJECT_ROOT / "paper_protocol/frozen_asset_manifest.json"
    if sha256_lf(frozen_manifest_path) != FROZEN_MANIFEST_SHA256:
        raise RuntimeError("P0-FORMAL-ASSET-MISMATCH: frozen manifest")
    if sha256_lf(CANDIDATE_MANIFEST_PATH) != CANDIDATE_MANIFEST_SHA256:
        raise RuntimeError("P0-FORMAL-ASSET-MISMATCH: candidate runtime manifest")
    if candidate_code_aggregate_sha256() != CANDIDATE_CODE_AGGREGATE_SHA256:
        raise RuntimeError("P0-FORMAL-ASSET-MISMATCH: candidate code")
    if _sha256(M4_PARITY_PATH) != M4_PARITY_SHA256:
        raise RuntimeError("M4-LEGACY-CONTRACT-MISMATCH: parity artifact")
    if _sha256(PROJECT_ROOT / "tools/paper/formal_batch_runtime.py") != HISTORICAL_FORMAL_RUNTIME_SHA256:
        raise RuntimeError("P0-FORMAL-ASSET-MISMATCH: historical formal runtime")
    if deterministic_protocol_artifact_sha256() != DETERMINISTIC_ARTIFACT_SHA256:
        raise RuntimeError("P0-FORMAL-ASSET-MISMATCH: deterministic protocol")
    training_audit = json.loads(TRAINING_AUDIT_PATH.read_text(encoding="utf-8"))
    trunk_audit = json.loads(TRUNK_AUDIT_PATH.read_text(encoding="utf-8"))
    m4_audit = json.loads(M4_PARITY_PATH.read_text(encoding="utf-8"))
    if training_audit.get("status") != "PASS":
        raise RuntimeError("OPTIMIZER-CONTRACT-AMBIGUOUS")
    if trunk_audit.get("status") != "PASS":
        raise RuntimeError("COMPLEX-TRUNK-CONTRACT-MISMATCH")
    if m4_audit.get("status") != "PASS":
        raise RuntimeError("M4-LEGACY-CONTRACT-MISMATCH")
    registry = load_registry()
    manifest = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
    verification = verify_manifest(manifest, PROJECT_ROOT, asset_root, verify_external=True)
    if verification["status"] != "PASS":
        raise RuntimeError("P0-FORMAL-ASSET-MISMATCH: frozen assets")
    trees = {
        "formal": _assert_tree("formal output", formal_root, FORMAL_TREE),
        "reviewer_risk": _assert_tree(
            "reviewer-risk closure", asset_root / "AAAI27-REVIEWER-RISK-CLOSURE", REVIEWER_RISK_TREE
        ),
        "deterministic_protocol": _assert_tree(
            "deterministic protocol", asset_root / "AAAI27-P0-DETERMINISTIC-INITIALIZATION-PROTOCOL", DETERMINISTIC_TREE
        ),
        "candidate_smoke": _assert_tree(
            "candidate smoke", asset_root / "AAAI27-P0-CANDIDATE-ADAPTERS-RUNNER-SMOKE", SMOKE_TREE
        ),
    }
    formal_registry = yaml.safe_load(
        (PROJECT_ROOT / "paper_protocol/experiment_registry.yaml").read_text(encoding="utf-8")
    )
    reviewer_registry = yaml.safe_load(
        (PROJECT_ROOT / "paper_protocol/reviewer_risk/reviewer_risk_experiment_registry.yaml").read_text(encoding="utf-8")
    )
    paper_final = sum(
        row.get("status") == "PAPER_FINAL"
        for row in (*formal_registry["experiments"], *reviewer_registry["experiments"])
    )
    if paper_final or registry.get("paper_final"):
        raise RuntimeError("PAPER-FINAL-VIOLATION")
    smoke = torch.tensor([6.0, 7.0], device="cuda").sum()
    torch.cuda.synchronize()
    if float(smoke) != 13.0:
        raise RuntimeError("P0-FORMAL-ASSET-MISMATCH: CUDA tensor smoke")
    return {
        "schema_version": "canondressgs.paper.p0_formal_preflight.v1",
        "status": "PASS", "task_id": TASK_ID,
        "source_head": SOURCE_HEAD, "run_commit": git("rev-parse", "HEAD"),
        "branch": RUN_BRANCH,
        "candidate_code_aggregate_sha256": CANDIDATE_CODE_AGGREGATE_SHA256,
        "candidate_runtime_manifest_sha256": CANDIDATE_MANIFEST_SHA256,
        "formal_registry_sha256": FORMAL_REGISTRY_SHA256,
        "reviewer_registry_sha256": REVIEWER_REGISTRY_SHA256,
        "frozen_manifest_sha256": FROZEN_MANIFEST_SHA256,
        "deterministic_protocol_artifacts_sha256": DETERMINISTIC_ARTIFACT_SHA256,
        "frozen_asset_verification": verification,
        "trees": trees, "paper_final_count": paper_final,
        "environment": {
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "python": platform.python_version(),
            "cuda_tensor_smoke": float(smoke),
        },
    }


def prepare_context(canary_attempt: Path) -> dict[str, Any]:
    old_branch, old_source = core.FORMAL_BRANCH, core.FORMAL_SOURCE_HEAD
    core.FORMAL_BRANCH, core.FORMAL_SOURCE_HEAD = RUN_BRANCH, SOURCE_HEAD
    try:
        return core._legacy_context(canary_attempt)
    finally:
        core.FORMAL_BRANCH, core.FORMAL_SOURCE_HEAD = old_branch, old_source


def feature_rows(
    cache: Mapping[str, Any], key: str, kind: str, device: torch.device,
    *, variant: str = "normal", indices: Sequence[int] = (0, 1, 2),
) -> tuple[torch.Tensor, torch.Tensor]:
    row = cache["episodes"][key][variant]
    index = torch.tensor(tuple(indices), dtype=torch.long)
    values = row[kind].index_select(0, index).to(device)
    valid = row["valid"].index_select(0, index).to(device)
    if kind == "rff" and values.shape[0] < 3:
        padded = values.new_zeros((3, values.shape[1]))
        padded_valid = valid.new_zeros((3, 1))
        padded[: values.shape[0]] = values
        padded_valid[: valid.shape[0]] = valid
        values, valid = padded, padded_valid
    return values, valid


def model_factory(
    method: str, seed: int, cache: Mapping[str, Any], device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    seed_all(seed)
    if method == "Ours-v2":
        row = cache["episodes"][f"O01/{CONDITIONS[0]}"]["normal"]["f2"]
        model: torch.nn.Module = OursV2CandidateAdapter(input_dim=2 * int(row.shape[1]))
        metadata = {"counterpart_trunk_sha256": None}
    elif method == "B6":
        row = cache["episodes"][f"O01/{CONDITIONS[0]}"]["normal"]["f2"]
        model = B6ReferenceClassifierHardLookupAdapter(
            seed=seed, input_dim=2 * int(row.shape[1])
        )
        metadata = {"counterpart_trunk_sha256": None}
    elif method in {"M3", "M4"}:
        raw_dim = int(
            cache["episodes"][f"O01/{CONDITIONS[0]}"]["normal"]["rff"].shape[1]
        )
        m3, m4 = build_m3_m4_candidate_adapters(raw_dim=raw_dim, seed=seed)
        model = m3 if method == "M3" else m4
        counterpart = m4 if method == "M3" else m3
        own = selected_state_sha256(model.candidate, "trunk")
        other = selected_state_sha256(counterpart.candidate, "trunk")
        if own != other:
            raise RuntimeError("COMPLEX-TRUNK-CONTRACT-MISMATCH")
        metadata = {
            "raw_dim": raw_dim, "trunk_sha256": own,
            "counterpart_trunk_sha256": other, "same_seed_trunk_bitwise_equal": True,
        }
    else:
        raise ValueError(f"unsupported trainable method: {method}")
    return model.to(device), metadata


def optimizer_factory(model: torch.nn.Module) -> tuple[torch.optim.Optimizer, Any]:
    parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(
        parameters, lr=0.02, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0.0, amsgrad=False,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return optimizer, scheduler


def nested_sha256(value: Any) -> str:
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0" + str(tensor.dtype).encode() + b"\0")
            digest.update(json.dumps(list(tensor.shape)).encode() + b"\0")
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=lambda key: str(key)):
                update(str(key)); update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"sequence\0")
            for element in item: update(element)
        else:
            digest.update(type(item).__name__.encode() + b"\0")
            digest.update(repr(item).encode("utf-8") + b"\0")

    update(value)
    return digest.hexdigest()


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(dict(value), sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def forward_training_batch(
    model: torch.nn.Module, method: str, cache: Mapping[str, Any],
    condition: str, targets: Mapping[str, torch.Tensor], device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
    outputs = []
    selections = []
    kind = "rff" if method in {"M3", "M4"} else "f2"
    for outfit in OUTFITS:
        rows, valid = feature_rows(cache, f"{outfit}/{condition}", kind, device)
        prediction = model(rows, valid)
        if method == "B6":
            outputs.append(prediction.logits)
            selections.append(prediction.predicted_outfit)
        else:
            outputs.append(prediction.standardized_coefficients)
    matrix = torch.stack(outputs)
    if method == "B6":
        labels = torch.arange(len(OUTFITS), device=device)
        losses = model.training_loss(matrix, labels)
        accuracy = float((matrix.argmax(1) == labels).float().mean().detach())
        metrics = {
            "classification_accuracy": accuracy,
            "predicted_outfits": selections,
            "logits_sha256": tensor_mapping_sha256({"logits": matrix}),
        }
    else:
        target = torch.stack([targets[outfit] for outfit in OUTFITS])
        losses = model.training_loss(matrix, target)
        nearest = [
            min(OUTFITS, key=lambda name: float(torch.linalg.vector_norm(row - targets[name])))
            for row in matrix.detach()
        ]
        metrics = {
            "standardized_coefficient_rmse": float(torch.sqrt(torch.mean((matrix - target) ** 2)).detach()),
            "nearest_teacher_accuracy": sum(a == b for a, b in zip(nearest, OUTFITS)) / len(OUTFITS),
            "predicted_standardized_sha256": tensor_mapping_sha256({"prediction": matrix}),
        }
    return matrix, losses, metrics


def parameter_norm(model: torch.nn.Module) -> float:
    values = [value.detach().reshape(-1) for value in model.parameters() if value.requires_grad]
    return float(torch.linalg.vector_norm(torch.cat(values)))


def checkpoint_payload(
    *, run: Mapping[str, Any], model: torch.nn.Module,
    optimizer: torch.optim.Optimizer, scheduler: Any, step: int,
    fixed_output: torch.Tensor, last_train_row: Mapping[str, Any] | None,
    build_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "canondressgs.paper.p0_formal_candidate_checkpoint.v1",
        "task_id": TASK_ID, "formal_run_id": run["formal_run_id"],
        "method": run["method"], "seed": run["seed"],
        "replicate_index": run["replicate_index"],
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "rng": core._rng_state(), "global_step": int(step),
        "condition_position": int(step % len(CONDITIONS)),
        "fixed_output": fixed_output.detach().cpu().clone(),
        "last_train_row": None if last_train_row is None else dict(last_train_row),
        "build_metadata": dict(build_metadata),
        "frozen_asset_fingerprint": FROZEN_MANIFEST_SHA256,
        "source_head": SOURCE_HEAD, "run_commit": git("rev-parse", "HEAD"),
    }


def load_training_state(
    *, path: Path, run: Mapping[str, Any], model: torch.nn.Module,
    optimizer: torch.optim.Optimizer, scheduler: Any,
) -> tuple[int, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("formal_run_id") != run["formal_run_id"] or payload.get("method") != run["method"]:
        raise RuntimeError("RESUME-CONTRACT-VIOLATION: checkpoint identity")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    core._restore_rng(payload["rng"])
    return int(payload["global_step"]), payload


def _milestone_record(
    *, step: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    fixed_output: torch.Tensor, metrics: Mapping[str, Any], row: Mapping[str, Any] | None,
    checkpoint_path: Path,
) -> dict[str, Any]:
    return {
        "step": step,
        "candidate_state_sha256": tensor_mapping_sha256(model.state_dict()),
        "optimizer_state_sha256": nested_sha256(optimizer.state_dict()),
        "fixed_batch_output_sha256": tensor_mapping_sha256({"output": fixed_output}),
        "coefficient_or_classification_metrics": dict(metrics),
        "train_loss": None if row is None else row["loss"],
        "gradient_norm_before_clip": None if row is None else row["gradient_norm_before_clip"],
        "gradient_norm_after_clip_bound": 5.0,
        "parameter_norm": parameter_norm(model),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
    }


def train_candidate(
    *, attempt: Path, run: Mapping[str, Any], cache: Mapping[str, Any],
    targets: Mapping[str, torch.Tensor], device: torch.device,
    stop_after_step: int = OPTIMIZER_STEPS,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    method, seed = str(run["method"]), int(run["seed"])
    model, build_metadata = model_factory(method, seed, cache, device)
    optimizer, scheduler = optimizer_factory(model)
    expected_ids = {id(value) for value in model.parameters() if value.requires_grad}
    optimizer_ids = {id(value) for group in optimizer.param_groups for value in group["params"]}
    if expected_ids != optimizer_ids:
        raise RuntimeError("optimizer candidate parameter membership mismatch")
    latest = attempt / "checkpoints/checkpoint_latest.pth"
    log_path = attempt / "logs/train.jsonl"
    intent = attempt / "provenance/STEP_IN_PROGRESS.json"
    start_step = 0
    resume_payload: dict[str, Any] | None = None
    if latest.is_file():
        start_step, resume_payload = load_training_state(
            path=latest, run=run, model=model, optimizer=optimizer, scheduler=scheduler
        )
    rows = read_jsonl(log_path)
    if resume_payload is not None and len(rows) == start_step - 1:
        if resume_payload.get("last_train_row") is None:
            raise RuntimeError("RESUME-CONTRACT-VIOLATION: missing durable train row")
        append_jsonl(log_path, resume_payload["last_train_row"])
        rows.append(dict(resume_payload["last_train_row"]))
    if len(rows) != start_step or [row["step"] for row in rows] != list(range(1, start_step + 1)):
        raise RuntimeError("RESUME-CONTRACT-VIOLATION: trajectory log")
    if intent.is_file():
        pending = json.loads(intent.read_text(encoding="utf-8"))
        if int(pending["step"]) > start_step:
            raise RuntimeError("RESUME-CONTRACT-VIOLATION: optimizer step may be uncheckpointed")
        intent.unlink()
    summary_path = attempt / "metrics/training_summary.json"
    completed_summary: dict[str, Any] | None = None
    if start_step == OPTIMIZER_STEPS and summary_path.is_file():
        previous_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous_summary.get("status") == "TRAINED":
            completed_summary = previous_summary
    fixed_condition = CONDITIONS[0]
    with torch.no_grad():
        fixed_output, fixed_losses, fixed_metrics = forward_training_batch(
            model, method, cache, fixed_condition, targets, device
        )
    milestone_paths = [
        str(path) for path in sorted((attempt / "checkpoints").glob("checkpoint_step_*.pth"))
    ]
    if start_step == 0:
        initial_path = attempt / "checkpoints/checkpoint_step_000000.pth"
        initial_payload = checkpoint_payload(
            run=run, model=model, optimizer=optimizer, scheduler=scheduler, step=0,
            fixed_output=fixed_output, last_train_row=None, build_metadata=build_metadata,
        )
        atomic_torch(initial_path, initial_payload)
        atomic_torch(latest, initial_payload)
        milestone = _milestone_record(
            step=0, model=model, optimizer=optimizer, fixed_output=fixed_output,
            metrics=fixed_metrics, row=None, checkpoint_path=initial_path,
        )
        atomic_json(attempt / "metrics/milestones/step_000000.json", milestone)
        if str(initial_path) not in milestone_paths:
            milestone_paths.append(str(initial_path))
    gradient_seen: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"finite": True, "nonzero": False, "max_l2": 0.0}
    )
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(start_step + 1, min(stop_after_step, OPTIMIZER_STEPS) + 1):
        atomic_json(intent, {
            "formal_run_id": run["formal_run_id"], "step": step,
            "checkpointed_global_step_before": step - 1,
        })
        condition = CONDITIONS[(step - 1) % len(CONDITIONS)]
        output, losses, training_metrics = forward_training_batch(
            model, method, cache, condition, targets, device
        )
        total = losses["total"]
        if not torch.isfinite(total):
            raise FloatingPointError("formal candidate loss is NaN or Inf")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        squared_norm = 0.0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            gradient = parameter.grad
            finite = gradient is not None and bool(torch.isfinite(gradient).all())
            if not finite:
                raise FloatingPointError(f"candidate gradient invalid: {name}")
            norm = float(torch.linalg.vector_norm(gradient))
            squared_norm += norm * norm
            group = name.split(".", 2)[0]
            gradient_seen[group]["finite"] &= finite
            gradient_seen[group]["nonzero"] |= bool(torch.count_nonzero(gradient))
            gradient_seen[group]["max_l2"] = max(gradient_seen[group]["max_l2"], norm)
        gradient_norm = math.sqrt(squared_norm)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            fixed_output, _, fixed_metrics = forward_training_batch(
                model, method, cache, fixed_condition, targets, device
            )
        row = {
            "step": step, "condition": condition, "loss": float(total.detach()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "gradient_norm_before_clip": gradient_norm,
            "parameter_norm": parameter_norm(model),
            **{name: float(value.detach()) for name, value in losses.items() if name != "total"},
        }
        payload = checkpoint_payload(
            run=run, model=model, optimizer=optimizer, scheduler=scheduler, step=step,
            fixed_output=fixed_output, last_train_row=row, build_metadata=build_metadata,
        )
        atomic_torch(latest, payload)
        append_jsonl(log_path, row)
        if step in MILESTONES:
            milestone_path = attempt / "checkpoints" / f"checkpoint_step_{step:06d}.pth"
            atomic_torch(milestone_path, payload)
            milestone = _milestone_record(
                step=step, model=model, optimizer=optimizer,
                fixed_output=fixed_output, metrics=fixed_metrics,
                row=row, checkpoint_path=milestone_path,
            )
            atomic_json(attempt / "metrics/milestones" / f"step_{step:06d}.json", milestone)
            if str(milestone_path) not in milestone_paths:
                milestone_paths.append(str(milestone_path))
        intent.unlink()
    global_step = min(stop_after_step, OPTIMIZER_STEPS)
    if global_step < OPTIMIZER_STEPS:
        report = {
            "status": "INTERRUPTED_RESUMABLE", "optimizer_steps": global_step,
            "resume_checkpoint": str(latest), "optimizer_steps_repeated": 0,
        }
        atomic_json(attempt / "metrics/training_summary.json", report)
        return model, report
    reloaded, reloaded_meta = model_factory(method, seed, cache, device)
    reload_optimizer, reload_scheduler = optimizer_factory(reloaded)
    reload_step, reload_payload = load_training_state(
        path=latest, run=run, model=reloaded,
        optimizer=reload_optimizer, scheduler=reload_scheduler,
    )
    with torch.no_grad():
        reload_output, _, _ = forward_training_batch(
            reloaded, method, cache, fixed_condition, targets, device
        )
    resume_acceptance = {
        "status": "PASS",
        "global_step_exact": reload_step == OPTIMIZER_STEPS,
        "model_state_max_abs_diff": core._tree_max_abs(model.state_dict(), reloaded.state_dict()),
        "optimizer_state_max_abs_diff": core._tree_max_abs(optimizer.state_dict(), reload_optimizer.state_dict()),
        "scheduler_state_max_abs_diff": core._tree_max_abs(scheduler.state_dict(), reload_scheduler.state_dict()),
        "fixed_output_max_abs_diff": float((fixed_output - reload_output).abs().max()),
        "fixed_output_bitwise_exact": torch.equal(fixed_output, reload_output),
        "optimizer_steps_repeated": 0,
    }
    if not resume_acceptance["global_step_exact"] or any(
        float(value) != 0.0 for key, value in resume_acceptance.items() if key.endswith("diff")
    ) or not resume_acceptance["fixed_output_bitwise_exact"]:
        raise RuntimeError("RESUME-CONTRACT-VIOLATION: final reload")
    rows = read_jsonl(log_path)
    if len(rows) != OPTIMIZER_STEPS or [row["step"] for row in rows] != list(range(1, 301)):
        raise RuntimeError("STEP-COUNT-MISMATCH")
    if completed_summary is not None:
        report = dict(completed_summary)
        report["optimizer_steps_repeated"] = 0
        report["post_training_resume_count"] = int(
            completed_summary.get("post_training_resume_count", 0)
        ) + 1
        report["checkpoint_resume"] = {
            **resume_acceptance,
            "resumed_after_completed_training": True,
            "resume_start_step": OPTIMIZER_STEPS,
        }
        atomic_json(summary_path, report)
        return model, report
    report = {
        "schema_version": "canondressgs.paper.p0_formal_training_summary.v1",
        "status": "TRAINED", "formal_run_id": run["formal_run_id"],
        "method": method, "optimizer_steps": OPTIMIZER_STEPS,
        "optimizer_steps_repeated": 0, "loss_first": rows[0]["loss"],
        "loss_last": rows[-1]["loss"], "trajectory_sha256": _sha256(log_path),
        "candidate": candidate_parameter_manifest(model),
        "optimizer": {
            "class": "torch.optim.Adam", "learning_rate": 0.02,
            "weight_decay": 0.0, "betas": [0.9, 0.999], "epsilon": 1e-8,
            "scheduler": "constant_lambda_1.0", "parameter_membership_complete": True,
        },
        "build_metadata": build_metadata, "gradient_groups": dict(gradient_seen),
        "milestone_checkpoints": milestone_paths,
        "final_checkpoint": str(attempt / "checkpoints/checkpoint_step_000300.pth"),
        "final_checkpoint_sha256": _sha256(attempt / "checkpoints/checkpoint_step_000300.pth"),
        "checkpoint_resume": resume_acceptance,
        "training_time_seconds": time.perf_counter() - started,
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    }
    atomic_json(summary_path, report)
    return model, report


def build_b7_runtime(
    *, cache: Mapping[str, Any], reference_checksums: Mapping[str, Mapping[str, str]],
    device: torch.device,
) -> tuple[dict[str, torch.nn.Module], dict[str, Any]]:
    adapters: dict[str, torch.nn.Module] = {}
    folds: dict[str, Any] = {}
    for condition in CONDITIONS:
        rows_by_outfit: dict[str, torch.Tensor] = {}
        valid_by_outfit: dict[str, torch.Tensor] = {}
        legal = [view for view in CONDITIONS if view != condition]
        hashes = {}
        for outfit in OUTFITS:
            rows, valid = feature_rows(
                cache, f"{outfit}/{condition}", "f2", device
            )
            rows_by_outfit[outfit], valid_by_outfit[outfit] = rows, valid
            hashes[outfit] = [reference_checksums[outfit][view] for view in legal]
        adapter, manifest = build_b7_fold_adapter(
            target_condition=condition, fold_rows=rows_by_outfit,
            fold_validity=valid_by_outfit, reference_file_hashes=hashes,
        )
        adapters[condition] = adapter.to(device)
        folds[condition] = manifest
    manifest = {
        "schema_version": "canondressgs.paper.p0_formal_b7_centroids.v1",
        "status": "PASS", "fold_count": len(folds), "folds": folds,
        "target_rgb_used": False, "target_mask_used": False,
        "all_target_conditions_excluded": all(
            row["target_condition_excluded"] for row in folds.values()
        ),
        "distance_metric": "FEATURE_STANDARDIZATION_THEN_SQUARED_L2",
        "tie_handling": "REGISTERED_SEEN_OUTFIT_ORDER",
    }
    return adapters, manifest


def predict(
    *, method: str, model: torch.nn.Module | None,
    b7_adapters: Mapping[str, torch.nn.Module] | None,
    cache: Mapping[str, Any], key: str, condition: str,
    targets: Mapping[str, torch.Tensor], coefficients: Mapping[str, torch.Tensor],
    coefficient_mean: torch.Tensor, coefficient_std: torch.Tensor,
    basis: torch.nn.Module, teacher_residuals: Mapping[str, Any],
    device: torch.device, variant: str = "normal",
    indices: Sequence[int] = (0, 1, 2),
) -> dict[str, Any]:
    if method == "B7":
        if b7_adapters is None:
            raise ValueError("B7 adapters are required")
        rows, valid = feature_rows(
            cache, key, "f2", device, variant=variant, indices=indices
        )
        with torch.no_grad():
            result = b7_adapters[condition](rows, valid)
        selected = result.predicted_outfit
        standardized = targets[selected].detach()
        residual = teacher_residuals[selected]
        return {
            "standardized": standardized, "residual": residual,
            "predicted_outfit": selected,
            "squared_distances": dict(result.squared_distances),
            "query_sha256": tensor_mapping_sha256({"query": result.query_feature}),
        }
    if model is None:
        raise ValueError("trainable method requires a model")
    kind = "rff" if method in {"M3", "M4"} else "f2"
    rows, valid = feature_rows(
        cache, key, kind, device, variant=variant, indices=indices
    )
    with torch.no_grad():
        result = model(rows, valid)
    if method == "B6":
        selected = result.predicted_outfit
        return {
            "standardized": targets[selected].detach(),
            "residual": teacher_residuals[selected],
            "predicted_outfit": selected, "predicted_class": result.predicted_class,
            "logits": result.logits.detach(),
        }
    standardized = result.standardized_coefficients.detach()
    raw = standardized * coefficient_std + coefficient_mean
    residual = basis(raw, chunk_size=16384)
    selected = min(
        OUTFITS,
        key=lambda outfit: float(torch.linalg.vector_norm(standardized - targets[outfit])),
    )
    return {
        "standardized": standardized, "raw": raw.detach(),
        "residual": residual, "predicted_outfit": selected,
    }


def _render(
    context: Mapping[str, Any], outfit: str, condition: str, residual: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    return multi._render_residual(context, outfit, condition, residual)


def _target_tensor_sha256(value: torch.Tensor) -> str:
    return tensor_mapping_sha256({"target": value.detach()})


def evaluate_candidate(
    *, context: Mapping[str, Any], attempt: Path, run: Mapping[str, Any],
    model: torch.nn.Module | None, b7_adapters: Mapping[str, torch.nn.Module] | None,
    b7_manifest: Mapping[str, Any] | None, cache: Mapping[str, Any], basis: Any,
    coefficients: Mapping[str, torch.Tensor], basis_payload: Mapping[str, Any],
    teacher_residuals: Mapping[str, Any], training: Mapping[str, Any],
    basis_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    episode_root = attempt / "renders/episodes"
    episode_root.mkdir(parents=True, exist_ok=True)
    visuals = attempt / "visuals"
    visuals.mkdir(parents=True, exist_ok=True)
    method = str(run["method"])
    device = context["base"]._xyz.device
    mean = torch.as_tensor(basis_payload["coefficient_train_mean"], device=device)
    std = torch.as_tensor(basis_payload["coefficient_train_std"], device=device)
    targets = multi._standardized_targets(coefficients, basis_payload)
    if model is not None:
        model.eval()
    metadata_seed = 0 if run.get("seed") is None else int(run["seed"])
    records: list[dict[str, Any]] = [{
        "record_type": "metadata", "seed": metadata_seed,
        "asset_fingerprint": FROZEN_MANIFEST_SHA256,
        "target_forward_input_used": False, "outfit_id_in_model": False,
        "not_applicable_metrics": [],
        "efficiency": {
            "trainable_parameter_count": float(training.get("candidate", {}).get("parameter_count", 0)),
            "basis_storage_bytes": float(basis_path.stat().st_size),
            "peak_vram_bytes": float(training.get("peak_vram_bytes", 0)),
            "training_time_seconds": float(training.get("training_time_seconds", 0.0)),
            "inference_time_seconds": 0.0, "render_time_seconds": 0.0,
        },
    }]
    correct_cache: dict[str, dict[str, Any]] = {}
    render_seconds = 0.0
    inference_seconds = 0.0
    visual_rows = []
    manual_review_rows = []
    confusion = [[0 for _ in OUTFITS] for _ in OUTFITS]
    hard_lookup_correct = {outfit: 0 for outfit in OUTFITS}
    query_rows = []
    for outfit_index, outfit in enumerate(OUTFITS):
        panels = []
        for condition in CONDITIONS:
            key = f"{outfit}/{condition}"
            infer_started = time.perf_counter()
            prediction = predict(
                method=method, model=model, b7_adapters=b7_adapters,
                cache=cache, key=key, condition=condition, targets=targets,
                coefficients=coefficients, coefficient_mean=mean, coefficient_std=std,
                basis=basis, teacher_residuals=teacher_residuals, device=device,
            )
            inference_seconds += time.perf_counter() - infer_started
            target_teacher = (
                teacher_residuals[outfit]
                if method in {"B6", "B7"}
                else basis(coefficients[outfit], chunk_size=16384)
            )
            render_started = time.perf_counter()
            predicted_rgb, predicted_alpha = _render(
                context, outfit, condition, prediction["residual"]
            )
            teacher_rgb, teacher_alpha = _render(
                context, outfit, condition, target_teacher
            )
            render_seconds += time.perf_counter() - render_started
            sample = context["samples"][key]
            residual_metrics = core._residual_metric_payload(
                prediction["residual"], target_teacher,
                context["config"]["basis"]["channel_bounds"],
            )
            render_metrics = core._render_metrics(
                sample, predicted_rgb, predicted_alpha, teacher_rgb, teacher_alpha
            )
            episode = {
                "record_type": "episode", "outfit_id": outfit, "view_id": condition,
                "target_reference_overlap": 0,
                "predicted_standardized_coefficients": prediction["standardized"].cpu().tolist(),
                "target_standardized_coefficients": targets[outfit].cpu().tolist(),
                "coefficient_normalization": {"mean": mean.cpu().tolist(), "std": std.cpu().tolist()},
                "teacher_standardized_coefficients": {
                    name: value.cpu().tolist() for name, value in targets.items()
                },
                "precomputed_residual_metrics": residual_metrics,
                "residual_metric_source": "production_full_gaussian_field_v1",
                "render_metrics": render_metrics,
                "predicted_outfit": prediction["predicted_outfit"],
                "target_rgb_sha256": _target_tensor_sha256(sample["target_edit_rgb"]),
            }
            if "logits" in prediction:
                episode["predicted_class"] = prediction["predicted_class"]
                episode["logits"] = prediction["logits"].cpu().tolist()
            if "squared_distances" in prediction:
                episode["per_class_squared_distances"] = prediction["squared_distances"]
                episode["query_sha256"] = prediction["query_sha256"]
                query_rows.append({
                    "outfit_id": outfit, "view_id": condition,
                    "predicted_outfit": prediction["predicted_outfit"],
                    "per_class_squared_distances": prediction["squared_distances"],
                    "query_sha256": prediction["query_sha256"],
                })
            records.append(episode)
            correct_cache[key] = {
                **prediction, "rgb": predicted_rgb.detach(),
                "alpha": predicted_alpha.detach(), "teacher_rgb": teacher_rgb.detach(),
                "teacher_residual": target_teacher,
            }
            if method in {"B6", "B7"}:
                predicted_index = OUTFITS.index(prediction["predicted_outfit"])
                confusion[outfit_index][predicted_index] += 1
                hard_lookup_correct[outfit] += int(prediction["predicted_outfit"] == outfit)
            save_render_tensor(
                episode_root / f"{outfit}_{condition}_prediction.png", predicted_rgb, 3
            )
            save_render_tensor(
                episode_root / f"{outfit}_{condition}_alpha.png", predicted_alpha, 1
            )
            panels.extend([
                (f"{condition} target", sample["target_edit_rgb"].detach().cpu(), 3),
                (f"{condition} base", sample["target_base_rgb"].detach().cpu(), 3),
                (f"{condition} teacher", teacher_rgb.detach().cpu(), 3),
                (f"{condition} prediction", predicted_rgb.detach().cpu(), 3),
            ])
            if condition == "cond_000318":
                source_outfit = OUTFITS[(outfit_index + 1) % len(OUTFITS)]
                correct_reference = context["episodes"][key]["reference_images"][0].detach().cpu()
                swapped_reference = context["episodes"][f"{source_outfit}/{condition}"]["reference_images"][0].detach().cpu()
                selected_label = prediction["predicted_outfit"]
                manual_review_rows.append((outfit, [
                    ("base", sample["target_base_rgb"].detach().cpu(), 3),
                    ("target", sample["target_edit_rgb"].detach().cpu(), 3),
                    ("teacher", teacher_rgb.detach().cpu(), 3),
                    (f"prediction selected={selected_label}", predicted_rgb.detach().cpu(), 3),
                    (f"correct reference {outfit}", correct_reference, 3),
                    (f"swapped reference {source_outfit}", swapped_reference, 3),
                    (f"selected endpoint {selected_label}", predicted_rgb.detach().cpu(), 3),
                ]))
        visual_rows.append((outfit, panels))
    o01._save_contact_sheet(visuals / "five_outfit_four_view_contact_sheet.png", visual_rows)
    o01._save_contact_sheet(visuals / "manual_review_contact_sheet.png", manual_review_rows)

    swap_rows = []
    bounds = context["config"]["basis"]["channel_bounds"]
    for target_outfit in OUTFITS:
        for condition in CONDITIONS:
            target_key = f"{target_outfit}/{condition}"
            correct = correct_cache[target_key]
            correct_distance = float(torch.linalg.vector_norm(correct["standardized"] - targets[target_outfit]))
            correct_render = next(
                row["render_metrics"]["garment_rgb_mae"] for row in records
                if row["record_type"] == "episode" and row["outfit_id"] == target_outfit and row["view_id"] == condition
            )
            for source_outfit in OUTFITS:
                if source_outfit == target_outfit:
                    continue
                source_key = f"{source_outfit}/{condition}"
                swapped = predict(
                    method=method, model=model, b7_adapters=b7_adapters,
                    cache=cache, key=source_key, condition=condition, targets=targets,
                    coefficients=coefficients, coefficient_mean=mean, coefficient_std=std,
                    basis=basis, teacher_residuals=teacher_residuals, device=device,
                )
                swapped_rgb, _ = _render(
                    context, target_outfit, condition, swapped["residual"]
                )
                garment = diagnosis._garment_mask(context["samples"][target_key])
                swapped_render = o01._masked_mae(
                    swapped_rgb, correct["teacher_rgb"], garment
                )
                correct_residual_error = parameterization.residual_metrics(
                    correct["residual"], correct["teacher_residual"], bounds,
                    active_epsilon=1e-8,
                )["normalized_rmse"]
                swapped_residual_error = parameterization.residual_metrics(
                    swapped["residual"], correct["teacher_residual"], bounds,
                    active_epsilon=1e-8,
                )["normalized_rmse"]
                swapped_distance = float(
                    torch.linalg.vector_norm(swapped["standardized"] - targets[target_outfit])
                )
                records.append({
                    "record_type": "swap", "source_outfit": source_outfit,
                    "target_outfit": target_outfit, "view_id": condition,
                    "correct_wins": bool(
                        correct_distance < swapped_distance
                        and float(correct_residual_error) < float(swapped_residual_error)
                        and correct_render < swapped_render
                    ),
                    "correct_predicted_outfit": correct["predicted_outfit"],
                    "swapped_predicted_outfit": swapped["predicted_outfit"],
                })
                if condition == "cond_000318":
                    swap_rows.append((f"{target_outfit}<-{source_outfit}", [
                        ("teacher", correct["teacher_rgb"].cpu(), 3),
                        (f"correct {correct['predicted_outfit']}", correct["rgb"].cpu(), 3),
                        (f"swapped {swapped['predicted_outfit']}", swapped_rgb.detach().cpu(), 3),
                    ]))
    o01._save_contact_sheet(visuals / "reference_swap_contact_sheet.png", swap_rows)

    for outfit in OUTFITS:
        for condition in CONDITIONS:
            key = f"{outfit}/{condition}"
            correct = correct_cache[key]
            variants = {
                "permutation": ("normal", (2, 0, 1)),
                "single": ("normal", (0,)),
                "dropout": ("normal", (0, 1)),
                "zero": ("zero", (0, 1, 2)),
                "base": ("base", (0, 1, 2)),
            }
            predicted = {
                name: predict(
                    method=method, model=model, b7_adapters=b7_adapters,
                    cache=cache, key=key, condition=condition, targets=targets,
                    coefficients=coefficients, coefficient_mean=mean, coefficient_std=std,
                    basis=basis, teacher_residuals=teacher_residuals, device=device,
                    variant=variant, indices=indices,
                )
                for name, (variant, indices) in variants.items()
            }
            records.extend([
                {
                    "record_type": "permutation", "outfit_id": outfit,
                    "view_id": condition,
                    "max_difference": float((correct["standardized"] - predicted["permutation"]["standardized"]).abs().max()),
                },
                {
                    "record_type": "single_reference", "outfit_id": outfit,
                    "view_id": condition,
                    "correct": predicted["single"]["predicted_outfit"] == outfit,
                },
                {
                    "record_type": "two_reference_dropout", "outfit_id": outfit,
                    "view_id": condition,
                    "correct": predicted["dropout"]["predicted_outfit"] == outfit,
                },
                {
                    "record_type": "replacement", "outfit_id": outfit,
                    "view_id": condition,
                    "zero_difference": float(torch.linalg.vector_norm(correct["standardized"] - predicted["zero"]["standardized"])),
                    "base_difference": float(torch.linalg.vector_norm(correct["standardized"] - predicted["base"]["standardized"])),
                },
            ])
    records[0]["efficiency"]["inference_time_seconds"] = inference_seconds
    records[0]["efficiency"]["render_time_seconds"] = render_seconds
    counts = {
        kind: sum(row["record_type"] == kind for row in records)
        for kind in sorted({row["record_type"] for row in records})
    }
    required_counts = {
        "metadata": 1, "episode": 20, "swap": 80, "permutation": 20,
        "single_reference": 20, "two_reference_dropout": 20, "replacement": 20,
    }
    if counts != required_counts:
        raise RuntimeError(f"EVALUATOR-CONTRACT-FAIL: {counts}")
    raw_path = attempt / "metrics/raw_episode_outputs.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8", newline="\n",
    )
    evaluated = evaluate_records(
        load_raw_records(raw_path), expected_asset_fingerprint=FROZEN_MANIFEST_SHA256
    )
    if evaluated.get("status") != "PASS" or evaluated.get("metric_count") != 27:
        raise RuntimeError("EVALUATOR-CONTRACT-FAIL: 27 metrics")
    atomic_json(attempt / "metrics/evaluated_metrics.json", evaluated)
    if method in {"B6", "B7"}:
        hard_lookup = {
            "schema_version": "canondressgs.paper.p0_hard_lookup_classification.v1",
            "method": method, "class_order": list(OUTFITS),
            "accuracy": sum(sum(row) for row in confusion if row) and sum(
                confusion[index][index] for index in range(len(OUTFITS))
            ) / 20.0,
            "per_outfit_accuracy": {
                outfit: hard_lookup_correct[outfit] / 4.0 for outfit in OUTFITS
            },
            "confusion_matrix": confusion,
            "row_role": "ground_truth_loss_or_offline_group_label",
            "column_role": "predicted_class_or_centroid",
            "ground_truth_used_for_lookup": False,
            "manual_class_correction": False,
        }
        atomic_json(attempt / "metrics/hard_lookup_classification.json", hard_lookup)
    if b7_manifest is not None:
        b7_payload = dict(b7_manifest)
        b7_payload["queries"] = query_rows
        b7_payload["query_count"] = len(query_rows)
        atomic_json(attempt / "manifests/b7_centroid_construction.json", b7_payload)
    summary = {
        "schema_version": "canondressgs.paper.p0_formal_evaluation_summary.v1",
        "status": "EVALUATED", "formal_run_id": run["formal_run_id"],
        "method": method, "record_counts": counts, "metric_count": 27,
        "metrics": evaluated["metrics"], "target_forward_leakage": False,
        "outfit_id_in_prediction_forward": False,
        "manual_review_required": True,
        "raw_metrics": str(raw_path),
        "evaluated_metrics": str(attempt / "metrics/evaluated_metrics.json"),
        "visuals": [
            str(visuals / "five_outfit_four_view_contact_sheet.png"),
            str(visuals / "reference_swap_contact_sheet.png"),
            str(visuals / "manual_review_contact_sheet.png"),
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(attempt / "metrics/evaluation_summary.json", summary)
    return evaluated, summary


def frozen_fingerprints(context: Mapping[str, Any], basis: Any) -> dict[str, Any]:
    return {
        "base": multi._tensor_state_fingerprint(multi._base_named_tensors(context["base"])),
        "backbone": o01._state_fingerprint(
            context["legacy_model"].clothing_observation_encoder.backbone.state_dict()
        ),
        "basis": basis.fingerprint(),
    }


def execute_run(
    *, context: Mapping[str, Any], cache: Mapping[str, Any],
    cache_path: Path, run: Mapping[str, Any], attempt: Path,
    asset_root: Path,
) -> dict[str, Any]:
    device = context["base"]._xyz.device
    basis, coefficients, basis_payload, basis_path = historical._basis_artifact(
        context, "Ours_Seen_Outfit_Explicit_Basis_V1"
    )
    targets = multi._standardized_targets(coefficients, basis_payload)
    before = frozen_fingerprints(context, basis)
    teacher_residuals = historical.load_frozen_teacher_residuals(context, OUTFITS)
    b7_adapters = None
    b7_manifest = None
    method = str(run["method"])
    if method == "B7":
        reference_path = (
            asset_root / "pipeline_full/SUBJECT02-AAAI27-DATA-CAPACITY-GATE-001/attempt_003"
            / "dataset/aaai_gate_28_manifest.json"
        )
        reference_checksums = _reference_checksum_index(
            json.loads(reference_path.read_text(encoding="utf-8"))
        )
        b7_adapters, b7_manifest = build_b7_runtime(
            cache=cache, reference_checksums=reference_checksums, device=device
        )
        model = None
        training = {
            "schema_version": "canondressgs.paper.p0_formal_training_summary.v1",
            "status": "TRAINED", "formal_run_id": run["formal_run_id"],
            "method": "B7", "optimizer_created": False, "optimizer_steps": 0,
            "optimizer_steps_repeated": 0,
            "candidate": {"parameter_count": 0, "parameter_names": []},
            "training_time_seconds": 0.0,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
            "checkpoint_resume": {"status": "NOT_APPLICABLE"},
        }
        atomic_json(attempt / "metrics/training_summary.json", training)
    else:
        model, training = train_candidate(
            attempt=attempt, run=run, cache=cache, targets=targets, device=device
        )
        if training["status"] != "TRAINED":
            raise RuntimeError("formal run stopped before step 300")
    evaluated, evaluation = evaluate_candidate(
        context=context, attempt=attempt, run=run, model=model,
        b7_adapters=b7_adapters, b7_manifest=b7_manifest, cache=cache,
        basis=basis, coefficients=coefficients, basis_payload=basis_payload,
        teacher_residuals=teacher_residuals, training=training,
        basis_path=basis_path,
    )
    after = frozen_fingerprints(context, basis)
    frozen_gradients = (
        multi._base_gradient_count(context["base"])
        + sum(
            parameter.grad is not None
            for parameter in context["legacy_model"].clothing_observation_encoder.backbone.parameters()
        )
        + sum(parameter.grad is not None for parameter in basis.parameters())
    )
    if before != after or frozen_gradients:
        raise RuntimeError("FROZEN-ASSET-MUTATION")
    result = {
        "schema_version": "canondressgs.paper.p0_formal_executor_result.v1",
        "status": "MANUAL_REVIEW_REQUIRED", "task_id": TASK_ID,
        "formal_run_id": run["formal_run_id"], "method": method,
        "attempt": str(attempt), "optimizer_steps": training["optimizer_steps"],
        "optimizer_steps_repeated": training.get("optimizer_steps_repeated", 0),
        "final_checkpoint": training.get("final_checkpoint"),
        "evaluation": evaluation, "metrics": evaluated["metrics"],
        "metric_count": evaluated["metric_count"],
        "frozen_before": before, "frozen_after": after,
        "frozen_parameter_max_change": 0.0,
        "frozen_gradient_count": int(frozen_gradients),
        "target_forward_leakage": False,
        "outfit_id_in_prediction_forward": False,
        "paper_final": False,
        "source_head": SOURCE_HEAD, "run_commit": git("rev-parse", "HEAD"),
        "candidate_code_aggregate_sha256": CANDIDATE_CODE_AGGREGATE_SHA256,
        "feature_cache": str(cache_path), "feature_cache_sha256": _sha256(cache_path),
        "basis_path": str(basis_path), "basis_sha256": _sha256(basis_path),
    }
    atomic_json(attempt / "provenance/executor_result.json", result)
    atomic_json(attempt / "RUN_STATUS.json", {
        "status": "MANUAL_REVIEW_REQUIRED", "formal_run_id": run["formal_run_id"],
        "optimizer_steps": training["optimizer_steps"], "paper_final": False,
        "updated_at_unix": time.time(),
    })
    del model, basis, teacher_residuals
    torch.cuda.empty_cache()
    return result


def _numeric_max_difference(left: Any, right: Any) -> float:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return math.inf
        return max((_numeric_max_difference(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return math.inf
        return max((_numeric_max_difference(a, b) for a, b in zip(left, right)), default=0.0)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    return 0.0 if left == right else math.inf


def _latest_attempt(output_root: Path, run: Mapping[str, Any]) -> Path:
    root = output_root / "formal_runs" / run["output_subdir"]
    attempts = sorted(root.glob("attempt_*"))
    if not attempts:
        raise RuntimeError(f"missing formal attempt: {run['formal_run_id']}")
    return attempts[-1]


def _evaluated(attempt: Path) -> dict[str, Any]:
    return json.loads((attempt / "metrics/evaluated_metrics.json").read_text(encoding="utf-8"))


def deterministic_reproducibility(
    output_root: Path, runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    attempts = [_latest_attempt(output_root, run) for run in runs]
    evaluations = [_evaluated(attempt) for attempt in attempts]
    checkpoints = [
        torch.load(
            attempt / "checkpoints/checkpoint_step_000300.pth",
            map_location="cpu", weights_only=False,
        )
        for attempt in attempts
    ]
    checkpoint_max = max(
        core._tree_max_abs(checkpoints[0]["model"], item["model"])
        for item in checkpoints[1:]
    )
    trajectories = [read_jsonl(attempt / "logs/train.jsonl") for attempt in attempts]
    trajectory_max = max(
        _numeric_max_difference(trajectories[0], item) for item in trajectories[1:]
    )
    metric_spreads = {
        metric: max(float(item["metrics"][metric]) for item in evaluations)
        - min(float(item["metrics"][metric]) for item in evaluations)
        for metric in ALL_METRICS
        if all(item["metrics"].get(metric) is not None for item in evaluations)
    }
    milestone_state_hashes = {
        str(step): [
            json.loads(
                (attempt / "metrics/milestones" / f"step_{step:06d}.json").read_text(encoding="utf-8")
            )["candidate_state_sha256"]
            for attempt in attempts
        ]
        for step in MILESTONES
    }
    state_hashes_exact = all(len(set(values)) == 1 for values in milestone_state_hashes.values())
    classification = (
        "EXACT_DETERMINISTIC_REPRODUCTION"
        if checkpoint_max == 0.0 and trajectory_max == 0.0
        and max(metric_spreads.values(), default=0.0) == 0.0 and state_hashes_exact
        else "SHARED_INIT_WITH_RUNTIME_DIVERGENCE"
    )
    return {
        "schema_version": "canondressgs.paper.ours_v2_deterministic_reproducibility.v1",
        "status": "PASS", "classification": classification,
        "replicate_count": 3, "replicates_are_random_samples": False,
        "sample_standard_deviation_reported": False,
        "checkpoint_model_max_abs_difference": checkpoint_max,
        "trajectory_max_abs_difference": trajectory_max,
        "metric_max_spread": max(metric_spreads.values(), default=0.0),
        "per_metric_spread": metric_spreads,
        "milestone_candidate_state_hashes": milestone_state_hashes,
        "milestone_states_bitwise_exact": state_hashes_exact,
        "trajectory_sha256": [_sha256(attempt / "logs/train.jsonl") for attempt in attempts],
        "final_checkpoint_model_state_sha256": [
            tensor_mapping_sha256(item["model"]) for item in checkpoints
        ],
        "runtime_nondeterminism_evidence": classification != "EXACT_DETERMINISTIC_REPRODUCTION",
        "replicate_metrics": [item["metrics"] for item in evaluations],
    }


def fixed_aggregation(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "PASS", "aggregation_order": ["episode", "outfit_macro", "fixed_result"],
        "fixed_non_trainable": True, "best_seed_selected": False,
        "pixel_weighted_micro_average": False, "held_out_included": False,
        "metrics": dict(evaluation["metrics"]),
        "per_episode_metrics": list(evaluation["per_episode_metrics"]),
    }


def ours_aggregation(
    evaluations: Sequence[Mapping[str, Any]], reproducibility: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = {}
    for metric in ALL_METRICS:
        values = [item["metrics"].get(metric) for item in evaluations]
        applicable = [float(value) for value in values if value is not None]
        metrics[metric] = {
            "replicate_values": values,
            "descriptive_mean": fmean(applicable) if applicable else None,
            "max_spread": max(applicable) - min(applicable) if applicable else None,
            "sample_std": None,
        }
    return {
        "status": "PASS",
        "aggregation_order": ["episode", "outfit_macro", "replicate", "deterministic_reproducibility"],
        "replicates_are_random_samples": False,
        "sample_standard_deviation_reported": False,
        "best_replicate_selected": False,
        "pixel_weighted_micro_average": False, "held_out_included": False,
        "reproducibility_classification": reproducibility["classification"],
        "metrics": metrics,
        "per_replicate_metrics": [item["metrics"] for item in evaluations],
        "per_episode_metrics": [
            row for item in evaluations for row in item["per_episode_metrics"]
        ],
    }


def _historical_a5(formal_root: Path) -> list[dict[str, Any]]:
    paths = sorted(formal_root.glob(
        "PAPER-A5-S*/seed_*/attempt_*/evaluated_metrics/evaluated_metrics.json"
    ))
    if len(paths) != 3:
        raise RuntimeError("M4-LEGACY-CONTRACT-MISMATCH: historical A5 results")
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def aggregate_all(
    *, output_root: Path, formal_root: Path, registry: Mapping[str, Any],
) -> dict[str, Any]:
    by_method: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    attempts: dict[str, Path] = {}
    evaluations: dict[str, dict[str, Any]] = {}
    results: dict[str, dict[str, Any]] = {}
    for run in registry["runs"]:
        attempt = _latest_attempt(output_root, run)
        status = json.loads((attempt / "RUN_STATUS.json").read_text(encoding="utf-8"))
        if status.get("status") != "MANUAL_REVIEW_REQUIRED":
            raise RuntimeError(f"incomplete formal run: {run['formal_run_id']}")
        evaluation = _evaluated(attempt)
        result = json.loads((attempt / "provenance/executor_result.json").read_text(encoding="utf-8"))
        attempts[run["formal_run_id"]] = attempt
        evaluations[run["formal_run_id"]] = evaluation
        results[run["formal_run_id"]] = result
        by_method[run["method"]].append(run)
    aggregates = output_root / "aggregates"
    reproduction = deterministic_reproducibility(output_root, by_method["Ours-v2"])
    atomic_json(aggregates / "ours_v2_deterministic_reproducibility.json", reproduction)
    ours_evaluations = [evaluations[run["formal_run_id"]] for run in by_method["Ours-v2"]]
    ours_result = ours_aggregation(ours_evaluations, reproduction)
    atomic_json(aggregates / "ours_v2_replicates.json", ours_result)
    method_aggregates: dict[str, Any] = {"Ours-v2": ours_result}
    for method in ("B6", "M3", "M4"):
        items = [evaluations[run["formal_run_id"]] for run in by_method[method]]
        aggregate = aggregate_evaluations(items)
        atomic_json(aggregates / f"{method.lower()}_seed_mean_std.json", aggregate)
        method_aggregates[method] = aggregate
    b7_evaluation = evaluations[by_method["B7"][0]["formal_run_id"]]
    b7_result = fixed_aggregation(b7_evaluation)
    atomic_json(aggregates / "b7_fixed.json", b7_result)
    method_aggregates["B7"] = b7_result
    a5 = _historical_a5(formal_root)
    matrix = {
        "schema_version": "canondressgs.paper.p0_m1_m4_raw_matrix.v1",
        "status": "RAW_RESULTS_MANUAL_REVIEW_REQUIRED",
        "scientific_freeze": False,
        "cells": {
            "M1_linear_corrected_ours_v2": [item["metrics"] for item in ours_evaluations],
            "M2_linear_legacy_a5_historical": [item["metrics"] for item in a5],
            "M3_complex_corrected": [
                evaluations[run["formal_run_id"]]["metrics"] for run in by_method["M3"]
            ],
            "M4_complex_legacy": [
                evaluations[run["formal_run_id"]]["metrics"] for run in by_method["M4"]
            ],
        },
        "B5": "OFF_MATRIX_HISTORICAL_EVIDENCE",
        "best_seed_selected": False,
    }
    atomic_json(aggregates / "m1_m4_raw_matrix.json", matrix)
    raw_index = {
        "schema_version": "canondressgs.paper.p0_formal_raw_result_index.v1",
        "task_id": TASK_ID, "status": "MANUAL_REVIEW_REQUIRED",
        "paper_final": False, "run_count": 13,
        "runs": [
            {
                "formal_run_id": run["formal_run_id"], "method": run["method"],
                "seed": run["seed"], "replicate_index": run["replicate_index"],
                "attempt": str(attempts[run["formal_run_id"]]),
                "optimizer_steps": results[run["formal_run_id"]]["optimizer_steps"],
                "checkpoint": results[run["formal_run_id"]]["final_checkpoint"],
                "metrics": results[run["formal_run_id"]]["evaluation"]["evaluated_metrics"],
                "provenance": str(attempts[run["formal_run_id"]] / "provenance/executor_result.json"),
                "visuals": results[run["formal_run_id"]]["evaluation"]["visuals"],
                "status": "MANUAL_REVIEW_REQUIRED",
            }
            for run in registry["runs"]
        ],
        "aggregates": {
            "ours_v2": str(aggregates / "ours_v2_replicates.json"),
            "b6": str(aggregates / "b6_seed_mean_std.json"),
            "b7": str(aggregates / "b7_fixed.json"),
            "m3": str(aggregates / "m3_seed_mean_std.json"),
            "m4": str(aggregates / "m4_seed_mean_std.json"),
            "m1_m4": str(aggregates / "m1_m4_raw_matrix.json"),
        },
    }
    atomic_json(output_root / "manifests/p0_formal_raw_result_index.json", raw_index)
    summary = {
        "schema_version": "canondressgs.paper.p0_formal_aggregate_summary.v1",
        "status": "MANUAL_REVIEW_REQUIRED", "run_count": 13,
        "trainable_run_count": 12, "non_trainable_run_count": 1,
        "total_optimizer_steps": sum(
            results[run["formal_run_id"]]["optimizer_steps"] for run in registry["runs"]
        ),
        "metric_count_per_run": 27, "correct_episode_count_per_run": 20,
        "swap_tuple_count_per_run": 80, "best_seed_selected": False,
        "paper_final": False,
        "ours_v2_reproducibility": reproduction["classification"],
        "raw_result_index": str(output_root / "manifests/p0_formal_raw_result_index.json"),
    }
    atomic_json(aggregates / "summary.json", summary)
    return summary
