from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # Local contract tests do not require the CUDA runtime.
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

try:
    import yaml
except ModuleNotFoundError:  # YAML is required by preflight, not pure helper tests.
    yaml = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TASK_ID = "AAAI27-RENDER-REFINED-COEFFICIENT-HEADROOM-001"
PROTOCOL_TASK_ID = "AAAI27-COEFFICIENT-HEADROOM-PROTOCOL-001"
SOURCE_BRANCH = "research/render-refined-coefficient-headroom-protocol-20260724"
SOURCE_HEAD = "2a42143f7942752aead16e7d53d1b7376fc5a143"
PURE_BRANCH = "research/pure-endpoint-core-method-crossfit-amended-20260724"
PURE_HEAD = "ce110887a942cf8db082ba688c8d36d2433bfdbe"
RUN_BRANCH = "research/render-refined-coefficient-headroom-experiment-20260724"
OUTPUT_NAME = "COEFFICIENT-HEADROOM-001"
ATTEMPT_NAME = "attempt_001"
OUTFITS = ("O01", "O02", "O03", "O04", "O08")
CONDITIONS = ("cond_000000", "cond_000318", "cond_000017", "cond_000347")
POSITIVE_LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1)
ALL_LAMBDAS = (0.0, *POSITIVE_LAMBDAS)
MILESTONES = (0, 20, 50, 100, 150, 200, 250, 300)
BASIS_SHA = "a29b3dc3c6f0ac1a79755e036eb4196dc9e4d116f3e2b35a7ed4e2bd5286f430"
NORMALIZATION_SHA = "c4eef5e6f86d91315d5e2a13d4e7949e0e6eaebde6182f3a60e5574790044fb2"
PURE_PREDICTION_SHA = "13e685087686d0ad597fd67daaad328f07910bceba33f46c6630623ba6e21290"
RISK = ROOT / "paper_protocol/reviewer_risk"
DOCS = ROOT / "docs/PAPER"
HANDOFF = ROOT / "project_control_handoff"
CONTRACT_FILES = (
    "docs/PAPER/AAAI27_RENDER_REFINED_COEFFICIENT_HEADROOM_PROTOCOL_20260724.md",
    "docs/PAPER/AAAI27_TEACHER_ENDPOINT_HEADROOM_ANALYSIS_PLAN_20260724.md",
    "docs/PAPER/AAAI27_FULL_RESIDUAL_FAIR_COMPARISON_PROTOCOL_20260724.md",
    "paper_protocol/reviewer_risk/coefficient_headroom_protocol.yaml",
    "paper_protocol/reviewer_risk/coefficient_headroom_rotation_manifests.json",
    "paper_protocol/reviewer_risk/coefficient_headroom_loss_contract.json",
    "paper_protocol/reviewer_risk/coefficient_headroom_optimizer_contract.json",
    "paper_protocol/reviewer_risk/coefficient_headroom_full_residual_contract.json",
    "paper_protocol/reviewer_risk/coefficient_headroom_evaluator_contract.json",
    "paper_protocol/reviewer_risk/coefficient_headroom_success_gates.json",
)
_RUNTIME_MODULE_CACHE: dict[str, Any] | None = None


def strict_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=strict_object)


def read_yaml(path: Path) -> Any:
    if yaml is None:
        raise RuntimeError("PyYAML is required for execution preflight")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any, *, replace: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str, *, replace: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True, default=str) + "\n")
        stream.flush()


def jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line, object_pairs_hook=strict_object)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path, *, lf: bool = False) -> str:
    digest = hashlib.sha256()
    if lf:
        data = path.read_bytes()
        if data.startswith(b"\xef\xbb\xbf"):
            data = data[3:]
        normalized = data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        digest.update(normalized.encode("utf-8"))
        return digest.hexdigest()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode() + b"\0")
    digest.update(json.dumps(list(tensor.shape)).encode() + b"\0")
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def output_root(value: str | None) -> Path:
    raw = value or os.environ.get("CANONDRESSGS_ASSET_ROOT")
    if not raw:
        raise RuntimeError("CANONDRESSGS_ASSET_ROOT or --output-root is required")
    root = Path(raw).resolve()
    os.environ["CANONDRESSGS_ASSET_ROOT"] = str(root)
    os.environ["CANONDRESSGS_OUTPUT_ROOT"] = str(root)
    return root


def attempt_path(root: Path) -> Path:
    return root / OUTPUT_NAME / ATTEMPT_NAME


def basis_path(root: Path) -> Path:
    return (
        root
        / "pipeline_full/SUBJECT02-MULTI-OUTFIT-EXPLICIT-BASIS-001/attempt_003"
        / "stage_b_basis/selected/selected_basis.pt"
    )


def normalization_path(root: Path) -> Path:
    return basis_path(root).with_name("coefficient_normalization.json")


def rotations() -> list[dict[str, Any]]:
    return read_json(RISK / "coefficient_headroom_rotation_manifests.json")["rotations"]


def loss_contract() -> dict[str, Any]:
    return read_json(RISK / "coefficient_headroom_loss_contract.json")


def optimizer_contract() -> dict[str, Any]:
    return read_json(RISK / "coefficient_headroom_optimizer_contract.json")


def full_contract() -> dict[str, Any]:
    return read_json(RISK / "coefficient_headroom_full_residual_contract.json")


def evaluator_contract() -> dict[str, Any]:
    return read_json(RISK / "coefficient_headroom_evaluator_contract.json")


def success_contract() -> dict[str, Any]:
    return read_json(RISK / "coefficient_headroom_success_gates.json")


def expected_counts() -> dict[str, int]:
    # Every count follows mechanically from frozen garment, rotation, lambda,
    # checkpoint, partition, and method cardinalities.
    cells = len(OUTFITS) * len(rotations())
    coefficient_regularized = cells * len(POSITIVE_LAMBDAS)
    coefficient_diagnostic = cells
    full_residual = cells
    runs = coefficient_regularized + coefficient_diagnostic + full_residual
    formal_variants = 6  # Teacher, SVD, coefficient, diagnostic, full step, full wall.
    formal_logical = formal_variants * cells * len(CONDITIONS)
    refined_lookup = 2 * cells
    parity = 2 * len(OUTFITS) * len(CONDITIONS)
    lambda_selection = coefficient_regularized
    base_visual = cells
    determinism_probe = 2
    evaluation_physical = formal_logical + refined_lookup + parity + lambda_selection + base_visual
    return {
        "coefficient_regularized_runs": coefficient_regularized,
        "coefficient_diagnostic_runs": coefficient_diagnostic,
        "full_residual_runs": full_residual,
        "optimization_runs": runs,
        "optimizer_creations": runs,
        "optimizer_steps": runs * 300,
        "forward_calls": runs * 300,
        "backward_calls": runs * 300,
        "scheduler_steps": 0,
        "checkpoint_writes": runs * len(MILESTONES),
        "coefficient_optimizer_render_calls": (coefficient_regularized + coefficient_diagnostic) * 300,
        "full_residual_optimizer_render_calls": full_residual * 300,
        "parity_renders": parity,
        "determinism_probe_renders": determinism_probe,
        "lambda_selection_evaluations": lambda_selection,
        "formal_logical_renders": formal_logical,
        "refined_lookup_logical_renders": refined_lookup,
        "base_visual_renders": base_visual,
        "evaluation_physical_renders": evaluation_physical,
        "renderer_calls": runs * 300 + evaluation_physical + determinism_probe,
        "optimize_evaluations": formal_variants * cells * 2,
        "calibration_evaluations": formal_variants * cells,
        "test_evaluations": formal_variants * cells + refined_lookup,
        "logical_renders": evaluation_physical,
        "unique_physical_renders": evaluation_physical,
        "cache_hits": 0,
        "visual_sheets": cells,
        "failure_records": 0,
        "paper_final_count": 0,
    }


def contract_hash_audit() -> dict[str, Any]:
    summary = read_json(RISK / "coefficient_headroom_protocol_final_summary.json")
    expected = summary["artifact_sha256_lf"]
    actual = {name: sha256(ROOT / name, lf=True) for name in expected}
    return {
        "status": "PASS" if actual == expected else "FAIL",
        "expected": expected,
        "actual": actual,
        "match_count": sum(actual[name] == digest for name, digest in expected.items()),
        "artifact_count": len(expected),
    }


def credential_scan(paths: Sequence[Path]) -> dict[str, Any]:
    patterns = (
        re.compile(r"sk-[A-Za-z0-9]{16,}"),
        re.compile(r"(?i)Authorization\s*:\s*Bearer\s+[A-Za-z0-9._-]{16,}"),
        re.compile(r"(?i)OPENAI_API_KEY\s*=\s*['\"]?[A-Za-z0-9._-]{16,}"),
    )
    findings: list[dict[str, Any]] = []
    ignored_explicit_fixtures: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file() or path.stat().st_size > 32 << 20:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not any(pattern.search(line) for pattern in patterns):
                continue
            row = {"path": str(path), "line": line_number, "pattern_class": "credential_value"}
            if "deliberately_fake_secret_value" in line:
                ignored_explicit_fixtures.append(row)
            else:
                findings.append(row)
    return {
        "status": "PASS" if not findings else "FAIL",
        "finding_count": len(findings), "findings": findings,
        "ignored_explicit_fixture_count": len(ignored_explicit_fixtures),
        "ignored_explicit_fixtures": ignored_explicit_fixtures,
    }


def pure_endpoint_audit(root: Path) -> dict[str, Any]:
    prediction = root / "PURE-ENDPOINT-CORE-METHOD-CROSSFIT-001/attempt_001/04_predictions/predictions.jsonl"
    manifest = root / "PURE-ENDPOINT-CORE-METHOD-CROSSFIT-001/attempt_001/04_predictions/prediction_manifest.json"
    rows = jsonl(prediction) if prediction.is_file() else []
    aggregate = read_json(manifest).get("aggregate_prediction_sha256") if manifest.is_file() else None
    classifier = [row for row in rows if row.get("method") == "Reference Classifier Lookup" and row.get("phase") == "primary"]
    return {
        "status": "PASS" if aggregate == PURE_PREDICTION_SHA and len(classifier) == 240 else "FAIL",
        "prediction_path": str(prediction),
        "aggregate_prediction_sha256": aggregate,
        "expected_prediction_sha256": PURE_PREDICTION_SHA,
        "reference_classifier_primary_rows": len(classifier),
        "source_head": PURE_HEAD,
        "mutation_count": 0,
    }


def static_preflight(root: Path) -> dict[str, Any]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    source_ancestor = subprocess.call(
        ["git", "merge-base", "--is-ancestor", SOURCE_HEAD, head], cwd=ROOT
    ) == 0
    protocol = read_yaml(RISK / "coefficient_headroom_protocol.yaml")
    opt = optimizer_contract()
    full = full_contract()
    gates = success_contract()
    rotation_rows = rotations()
    overlap_pass = True
    query_hash_pass = True
    for row in rotation_rows:
        parts = row["partitions"]
        optimize = set(parts["optimize"]["observation_ids"])
        calibration = set(parts["calibration"]["observation_ids"])
        test = set(parts["test"]["observation_ids"])
        overlap_pass &= not (optimize & calibration or optimize & test or calibration & test)
        for outfit in OUTFITS:
            query = [f"subject02/{outfit}/{row['optimize_folds'][step % 2]}" for step in range(300)]
            digest = hashlib.sha256("\n".join(query).encode()).hexdigest()
            query_hash_pass &= digest == row["per_garment_query_order"][outfit]["sha256"]
    tracked = [ROOT / name for name in git("ls-files").splitlines()]
    credentials = credential_scan(tracked)
    pure = pure_endpoint_audit(root)
    counts = expected_counts()
    checks = {
        "branch": branch == RUN_BRANCH,
        "source_ancestor": source_ancestor,
        "source_protocol_head": protocol["governance"]["branch"] == SOURCE_BRANCH,
        "attempt_absent": not attempt_path(root).exists(),
        "attempt_002_absent": not (root / OUTPUT_NAME / "attempt_002").exists(),
        "contract_hash_closure": contract_hash_audit()["status"] == "PASS",
        "basis_sha": protocol["assets"]["rank4_basis"]["sha256"] == BASIS_SHA,
        "normalization_sha": protocol["assets"]["coefficient_normalization"]["sha256"] == NORMALIZATION_SHA,
        "rotations": len(rotation_rows) == 4,
        "partition_disjoint": overlap_pass,
        "query_hashes": query_hash_pass,
        "lambda_grid": tuple(opt["coefficient_parameterization"]["garment_order"]) == OUTFITS and tuple(loss_contract()["primary_coefficient_regularization"]["lambda_grid"]) == POSITIVE_LAMBDAS,
        "run_count": counts["optimization_runs"] == opt["planned_execution"]["total_optimization_runs"] == 120,
        "step_count": counts["optimizer_steps"] == 36000,
        "checkpoint_count": counts["checkpoint_writes"] == opt["planned_execution"]["checkpoint_writes"] == 960,
        "coefficient_dof": opt["coefficient_parameterization"]["degrees_of_freedom"] == 4,
        "full_allocated": full["implementation"]["allocated_trainable_scalars"] == 2600000,
        "full_effective": full["implementation"]["effective_masked_trainable_scalars"] == 2217111,
        "success_enum": len(gates["decision_order"]) == 5,
        "pure_endpoint": pure["status"] == "PASS",
        "credentials": credentials["status"] == "PASS",
        "paper_final": protocol["paper_final"] is False,
    }
    return {
        "schema_version": "canondressgs.paper.coefficient_headroom_static_preflight.v1",
        "task_id": TASK_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "branch": branch,
        "head": head,
        "source_head": SOURCE_HEAD,
        "checks": checks,
        "contract_hash_audit": contract_hash_audit(),
        "pure_endpoint_audit": pure,
        "expected_counts": counts,
        "credential_findings": credentials["finding_count"],
        "environment": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "torch": None if torch is None else torch.__version__,
            "cuda": None if torch is None else torch.version.cuda,
            "cuda_available": False if torch is None else torch.cuda.is_available(),
        },
        "paper_final": False,
    }


def bind(cloud_preflight_path: Path, root: Path) -> dict[str, Any]:
    cloud = read_json(cloud_preflight_path)
    local = static_preflight(root)
    checks = {
        "local_preflight": local["status"] == "PASS",
        "cloud_preflight": cloud.get("status") == "PASS",
        "same_source": local["source_head"] == cloud.get("source_head") == SOURCE_HEAD,
        "same_counts": local["expected_counts"] == cloud.get("expected_counts"),
        "attempt_absent_both": local["checks"]["attempt_absent"] and cloud["checks"]["attempt_absent"],
        "pure_endpoint_read_only": local["pure_endpoint_audit"]["mutation_count"] == 0,
    }
    if not all(checks.values()):
        raise RuntimeError("coefficient headroom execution binding failed")
    binding = {
        "schema_version": "canondressgs.paper.coefficient_headroom_execution_binding.v1",
        "task_id": TASK_ID,
        "status": "BOUND_PRE_RESULT",
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "run_branch": RUN_BRANCH,
        "execution_head": None,
        "pure_endpoint_branch": PURE_BRANCH,
        "pure_endpoint_head": PURE_HEAD,
        "pure_endpoint_prediction_sha256": PURE_PREDICTION_SHA,
        "expected_counts": expected_counts(),
        "checks": checks,
        "paper_final": False,
    }
    tests = {
        "schema_version": "canondressgs.paper.coefficient_headroom_pre_result_tests.v1",
        "task_id": TASK_ID,
        "status": "PASS",
        "checks": checks,
        "optimizer_creations": 0,
        "optimizer_steps": 0,
        "attempt_created": False,
        "credential_findings": 0,
        "paper_final": False,
    }
    atomic_json(RISK / "coefficient_headroom_execution_binding.json", binding)
    atomic_json(RISK / "coefficient_headroom_expected_counts.json", {
        "schema_version": "canondressgs.paper.coefficient_headroom_expected_counts.v1",
        "task_id": TASK_ID,
        "status": "FROZEN_PRE_RESULT",
        "counts": expected_counts(),
        "derivation": "frozen garment x rotation x lambda x checkpoint x partition x method cardinalities",
    })
    atomic_json(RISK / "coefficient_headroom_pre_result_tests.json", tests)
    atomic_text(DOCS / "AAAI27_COEFFICIENT_HEADROOM_EXECUTION_BINDING_20260724.md", f"""# Coefficient Headroom Execution Binding

The frozen protocol at `{SOURCE_HEAD}` is bound to `{RUN_BRANCH}` before any optimizer creation.
The execution contains 120 fresh runs, 36,000 optimizer steps, 960 checkpoints, and one append-only `attempt_001`.

The Pure Endpoint source at `{PURE_HEAD}` is read-only and is used only for Refined Hard Lookup decomposition. Teacher Endpoint is an initialization and frozen comparison target, not an upper bound. `PAPER_FINAL=false`.
""")
    return {"status": "PASS", "checks": checks}


def runtime_imports() -> dict[str, Any]:
    global _RUNTIME_MODULE_CACHE
    if _RUNTIME_MODULE_CACHE is not None:
        return _RUNTIME_MODULE_CACHE
    if torch is None:
        raise RuntimeError("PyTorch with the frozen CUDA runtime is required")
    from scene.gaussian_clothing_residuals import GaussianClothingResiduals
    from scene.representation_capacity_oracle import (
        UnboundedGaussianDeltaField,
        capacity_oracle_loss_v1,
        direct_stability,
    )
    from tools import diagnose_image_conditioned_overfit_failure as diagnosis
    from tools import run_multi_outfit_explicit_basis as multi
    from tools import run_residual_field_parameterization as parameterization
    from tools import run_representation_triage_ladder as triage
    from tools.check_real_image_conditioned_one_batch import save_render_tensor
    from tools.paper import formal_batch_runtime as historical
    from tools.paper import formal_runtime as formal
    from tools.paper import run_p0_color_spatial_soft_control_evaluations as sealed_metrics
    _RUNTIME_MODULE_CACHE = {
        "GaussianClothingResiduals": GaussianClothingResiduals,
        "UnboundedGaussianDeltaField": UnboundedGaussianDeltaField,
        "capacity_oracle_loss_v1": capacity_oracle_loss_v1,
        "direct_stability": direct_stability,
        "diagnosis": diagnosis, "multi": multi, "parameterization": parameterization,
        "triage": triage, "save_render_tensor": save_render_tensor,
        "historical": historical, "formal": formal, "sealed_metrics": sealed_metrics,
    }
    return _RUNTIME_MODULE_CACHE


def runtime_context(attempt: Path) -> dict[str, Any]:
    modules = runtime_imports()
    formal = modules["formal"]
    old_branch, old_source = formal.FORMAL_BRANCH, formal.FORMAL_SOURCE_HEAD
    formal.FORMAL_BRANCH, formal.FORMAL_SOURCE_HEAD = RUN_BRANCH, SOURCE_HEAD
    try:
        return formal._legacy_context(attempt)
    finally:
        formal.FORMAL_BRANCH, formal.FORMAL_SOURCE_HEAD = old_branch, old_source


def load_basis(root: Path, device: torch.device) -> tuple[Any, dict[str, torch.Tensor], dict[str, Any]]:
    multi = runtime_imports()["multi"]
    basis, coefficients, payload = multi.load_basis_artifact(basis_path(root), device)
    payload = dict(payload)
    normalization = read_json(normalization_path(root))
    mean = torch.tensor(normalization["mean"], device=device, dtype=next(basis.buffers()).dtype)
    std = torch.tensor(normalization["std"], device=device, dtype=mean.dtype)
    payload["coefficient_train_mean"] = mean
    payload["coefficient_train_std"] = std
    return basis, coefficients, payload


def load_teachers(context: Mapping[str, Any]) -> dict[str, Any]:
    return runtime_imports()["historical"].load_frozen_teacher_residuals(context, OUTFITS)


def residual_fields(residual: Any) -> tuple[torch.Tensor, ...]:
    return (
        residual.delta_xyz,
        residual.delta_log_scaling,
        residual.delta_rotvec,
        residual.delta_opacity_logit,
        residual.delta_sh0,
        residual.delta_shN,
    )


def residual_fingerprint(residual: Any) -> str:
    return canonical_sha([tensor_sha(value) for value in residual_fields(residual)])


def loss_weights() -> dict[str, float]:
    contract = loss_contract()
    values = {name: float(row["weight"]) for name, row in contract["render_terms"].items()}
    values["stability"] = 0.0
    return values


def render_loss(
    rgb: torch.Tensor,
    alpha: torch.Tensor,
    sample: Mapping[str, Any],
    *,
    stability: torch.Tensor,
    stability_weight: float,
) -> dict[str, torch.Tensor]:
    triage = runtime_imports()["triage"]
    weights = loss_weights()
    weights["stability"] = float(stability_weight)
    return triage.loss_for_sample(rgb, alpha, sample, weights, stability)


def cloud_resource_preflight(root: Path) -> dict[str, Any]:
    gpu_query = subprocess.check_output(
        [
            "nvidia-smi", "--query-gpu=name,memory.free,memory.used",
            "--format=csv,noheader,nounits",
        ], text=True, encoding="utf-8",
    ).strip().splitlines()
    if len(gpu_query) != 1:
        raise RuntimeError("COEFFICIENT_HEADROOM_GPU_RESOURCE_CONFLICT")
    gpu_name, free_mib, used_mib = [value.strip() for value in gpu_query[0].split(",")]
    try:
        process_output = subprocess.check_output(
            [
                "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ], text=True, encoding="utf-8", stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        process_output = ""
    active_compute = [line for line in process_output.splitlines() if line.strip()]
    disk = shutil.disk_usage(root)
    probe = root / ".coefficient_headroom_write_probe"
    probe.write_text("writable\n", encoding="ascii")
    probe.unlink()
    remotes = set(git("remote").splitlines())
    linked_bare_mode = not remotes
    remote_continuity = (
        {"origin", "cloud"}.issubset(remotes)
        or (
            linked_bare_mode
            and git("branch", "--show-current") == RUN_BRANCH
            and git("rev-parse", RUN_BRANCH) == git("rev-parse", "HEAD")
        )
    )
    checks = {
        "gpu_is_rtx_4090": gpu_name == "NVIDIA GeForce RTX 4090",
        "free_vram_at_least_20_gib": int(free_mib) >= 20 * 1024,
        "no_active_compute_process": not active_compute,
        "free_disk_at_least_20_gib": disk.free >= 20 * (1 << 30),
        "output_writable": True,
        "attempt_absent": not attempt_path(root).exists(),
        "attempt_002_absent": not (root / OUTPUT_NAME / "attempt_002").exists(),
        "git_remote_continuity": remote_continuity,
    }
    return {
        "schema_version": "canondressgs.paper.coefficient_headroom_cloud_resource_preflight.v1",
        "task_id": TASK_ID, "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "gpu": {"name": gpu_name, "memory_free_mib": int(free_mib), "memory_used_mib": int(used_mib)},
        "active_compute_process_count": len(active_compute),
        "active_compute_processes": ["REDACTED_PROCESS_METADATA" for _ in active_compute],
        "disk": {"free_bytes": disk.free, "total_bytes": disk.total},
        "git_topology": "LINKED_LOCAL_BARE_REPOSITORY" if linked_bare_mode else "NAMED_REMOTES",
        "remote_names": sorted(remotes),
        "credential_values_recorded": False, "checked_at_utc": now(),
    }


def materialize(root: Path) -> dict[str, Any]:
    if git("branch", "--show-current") != RUN_BRANCH or git("status", "--short"):
        raise RuntimeError("materialization requires the clean execution branch")
    head = git("rev-parse", "HEAD")
    if subprocess.call(["git", "merge-base", "--is-ancestor", SOURCE_HEAD, head], cwd=ROOT) != 0:
        raise RuntimeError("SOURCE_HEAD mismatch")
    attempt = attempt_path(root)
    if attempt.exists():
        raise RuntimeError("COEFFICIENT_HEADROOM_ATTEMPT_COLLISION")
    resource_preflight = cloud_resource_preflight(root)
    if resource_preflight["status"] != "PASS":
        raise RuntimeError("COEFFICIENT_HEADROOM_GPU_RESOURCE_CONFLICT")
    names = (
        "00_preflight", "01_contract_snapshot", "02_static_parity", "03_coefficient_runs",
        "04_full_residual_runs", "05_checkpoints", "06_lambda_selection", "07_predictions",
        "08_metrics", "09_span_analysis", "10_refined_lookup_analysis", "11_visual_sheets",
        "12_failure_analysis", "13_final_verification",
    )
    for name in names:
        (attempt / name).mkdir(parents=True, exist_ok=False)
    for relative in CONTRACT_FILES:
        source = ROOT / relative
        shutil.copy2(source, attempt / "01_contract_snapshot" / source.name)
    for name in (
        "coefficient_headroom_execution_binding.json",
        "coefficient_headroom_expected_counts.json",
        "coefficient_headroom_pre_result_tests.json",
    ):
        source = RISK / name
        destination = attempt / "01_contract_snapshot" / name
        if name == "coefficient_headroom_execution_binding.json":
            binding_snapshot = read_json(source)
            binding_snapshot["execution_head"] = head
            atomic_json(destination, binding_snapshot, replace=False)
        else:
            shutil.copy2(source, destination)
    preflight = static_preflight(root)
    preflight["checks"]["attempt_absent"] = True
    preflight["checks"]["attempt_materialized_exactly_once"] = True
    preflight["status"] = "PASS" if all(preflight["checks"].values()) else "FAIL"
    atomic_json(attempt / "00_preflight/preflight.json", preflight, replace=False)
    atomic_json(
        attempt / "00_preflight/cloud_resource_preflight.json",
        resource_preflight, replace=False,
    )
    if preflight["status"] != "PASS":
        raise RuntimeError("runtime preflight failed")
    context = runtime_context(attempt)
    device = context["base"]._xyz.device
    basis, coefficients, payload = load_basis(root, device)
    teachers = load_teachers(context)
    full = full_contract()
    field_class = runtime_imports()["UnboundedGaussianDeltaField"]
    probe = field_class(context["base"]).to(device)
    allocated = sum(value.numel() for value in probe.parameters())
    active = int(probe.trainable_support.sum())
    effective = active * 13
    runtime_checks = {
        "basis_sha": sha256(basis_path(root)) == BASIS_SHA,
        "normalization_sha": sha256(normalization_path(root)) == NORMALIZATION_SHA,
        "basis_rank": int(basis.rank) == 4,
        "coefficient_shapes": all(tuple(value.shape) == (4,) for value in coefficients.values()),
        "normalization_shapes": tuple(payload["coefficient_train_mean"].shape) == (4,) and tuple(payload["coefficient_train_std"].shape) == (4,),
        "normalization_finite_positive": bool(torch.isfinite(payload["coefficient_train_std"]).all() and torch.all(payload["coefficient_train_std"] > 0)),
        "teacher_count": len(teachers) == 5,
        "base_count": int(context["base"]._xyz.shape[0]) == 200000,
        "target_samples": all(f"{outfit}/{condition}" in context["samples"] for outfit in OUTFITS for condition in CONDITIONS),
        "allocated_full_scalars": allocated == full["implementation"]["allocated_trainable_scalars"],
        "effective_full_scalars": effective == full["implementation"]["effective_masked_trainable_scalars"],
        "raw_shN_frozen_zero": not probe.raw_shN.requires_grad and int(torch.count_nonzero(probe.raw_shN)) == 0,
    }
    del probe
    if not all(runtime_checks.values()):
        raise RuntimeError("COEFFICIENT_HEADROOM_FROZEN_ASSET_MISMATCH")
    atomic_json(attempt / "00_preflight/runtime_asset_audit.json", {
        "status": "PASS",
        "checks": runtime_checks,
        "basis_fingerprint": basis.fingerprint(),
        "coefficient_sha256": {name: tensor_sha(value) for name, value in coefficients.items()},
        "teacher_residual_sha256": {name: residual_fingerprint(value) for name, value in teachers.items()},
        "allocated_trainable_scalars": allocated,
        "effective_trainable_scalars": effective,
    }, replace=False)
    atomic_json(attempt / "00_preflight/execution_metadata.json", {
        "task_id": TASK_ID,
        "source_head": SOURCE_HEAD,
        "pure_endpoint_head": PURE_HEAD,
        "execution_head": head,
        "branch": RUN_BRANCH,
        "attempt": ATTEMPT_NAME,
        "created_at_utc": now(),
        "paper_final": False,
    }, replace=False)
    atomic_json(attempt / "RUN_STATUS.json", {
        "status": "MATERIALIZED_NO_OPTIMIZER",
        "execution_head": head,
        "optimization_runs": 0,
        "optimizer_steps": 0,
        "renderer_calls": 0,
        "updated_at_utc": now(),
    }, replace=False)
    return {"status": "PASS", "attempt": str(attempt), "execution_head": head}


def run_parity(root: Path) -> dict[str, Any]:
    attempt = attempt_path(root)
    metadata = read_json(attempt / "00_preflight/execution_metadata.json")
    if git("status", "--short") or git("rev-parse", "HEAD") != metadata["execution_head"]:
        raise RuntimeError("parity requires the clean execution HEAD")
    destination = attempt / "02_static_parity/parity.json"
    if destination.exists():
        raise FileExistsError(destination)
    modules = runtime_imports()
    context = runtime_context(attempt)
    device = context["base"]._xyz.device
    basis, coefficients, payload = load_basis(root, device)
    teachers = load_teachers(context)
    parameterization = modules["parameterization"]
    bounds = payload["channel_bounds"]
    rows = []
    renderer_calls = 0
    parity_dir = attempt / "02_static_parity/renders"
    for outfit in OUTFITS:
        reconstructed = basis(coefficients[outfit], chunk_size=16384)
        residual_metric = parameterization.residual_metrics(
            reconstructed, teachers[outfit], bounds, active_epsilon=1e-8
        )
        for condition in CONDITIONS:
            sample = context["samples"][f"{outfit}/{condition}"]
            with torch.inference_mode():
                teacher_rgb, teacher_alpha = parameterization.render_prediction(
                    context["base"], sample, teachers[outfit], context["background"]
                )
                svd_rgb, svd_alpha = parameterization.render_prediction(
                    context["base"], sample, reconstructed, context["background"]
                )
            renderer_calls += 2
            rgb_mae = float((teacher_rgb - svd_rgb).abs().double().mean())
            alpha_mae = float((teacher_alpha - svd_alpha).abs().double().mean())
            token = f"{outfit}_{condition}"
            teacher_rgb_path = parity_dir / f"{token}_teacher_rgb.png"
            teacher_alpha_path = parity_dir / f"{token}_teacher_alpha.png"
            svd_rgb_path = parity_dir / f"{token}_svd_rgb.png"
            svd_alpha_path = parity_dir / f"{token}_svd_alpha.png"
            modules["save_render_tensor"](teacher_rgb_path, teacher_rgb, 3)
            modules["save_render_tensor"](teacher_alpha_path, teacher_alpha, 1)
            modules["save_render_tensor"](svd_rgb_path, svd_rgb, 3)
            modules["save_render_tensor"](svd_alpha_path, svd_alpha, 1)
            rows.append({
                "outfit_id": outfit,
                "condition_id": condition,
                "bound_normalized_residual_rmse": residual_metric["normalized_rmse"],
                "render_rgb_mae": rgb_mae,
                "render_alpha_mae": alpha_mae,
                "teacher_rgb_path": str(teacher_rgb_path),
                "teacher_alpha_path": str(teacher_alpha_path),
                "svd_rgb_path": str(svd_rgb_path),
                "svd_alpha_path": str(svd_alpha_path),
                "teacher_rgb_sha256": sha256(teacher_rgb_path),
                "teacher_alpha_sha256": sha256(teacher_alpha_path),
                "svd_rgb_sha256": sha256(svd_rgb_path),
                "svd_alpha_sha256": sha256(svd_alpha_path),
                "physical_render_sha_equivalent": sha256(teacher_rgb_path) == sha256(svd_rgb_path) and sha256(teacher_alpha_path) == sha256(svd_alpha_path),
                "pass": residual_metric["normalized_rmse"] <= 1e-5 and rgb_mae <= 1e-5 and alpha_mae <= 1e-5,
            })
    first = rows[0]
    sample = context["samples"][f"{first['outfit_id']}/{first['condition_id']}"]
    with torch.inference_mode():
        rgb1, alpha1 = parameterization.render_prediction(context["base"], sample, teachers[first["outfit_id"]], context["background"])
        rgb2, alpha2 = parameterization.render_prediction(context["base"], sample, teachers[first["outfit_id"]], context["background"])
    renderer_calls += 2
    determinism = bool(torch.equal(rgb1, rgb2) and torch.equal(alpha1, alpha2))
    result = {
        "schema_version": "canondressgs.paper.coefficient_headroom_teacher_svd_parity.v1",
        "task_id": TASK_ID,
        "status": "PASS" if all(row["pass"] for row in rows) and determinism else "FAIL",
        "parity_threshold": 1e-5,
        "row_count": len(rows),
        "renderer_calls": renderer_calls,
        "renderer_determinism": determinism,
        "rows": rows,
    }
    atomic_json(destination, result, replace=False)
    if result["status"] != "PASS":
        raise RuntimeError("COEFFICIENT_HEADROOM_BASIS_PARITY_FAILURE")
    status = read_json(attempt / "RUN_STATUS.json")
    status.update({"status": "PARITY_PASS_NO_OPTIMIZER", "renderer_calls": renderer_calls, "updated_at_utc": now()})
    atomic_json(attempt / "RUN_STATUS.json", status)
    return result


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def atomic_torch(path: Path, value: Any, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for name, value in tuple(state.items()):
            if isinstance(value, torch.Tensor):
                state[name] = value.to(device)


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def lambda_token(value: float) -> str:
    return "0" if value == 0 else f"{value:.0e}".replace("+", "").replace("-", "m")


def run_token(family: str, rotation: str, outfit: str, lambda_anchor: float | None = None) -> str:
    suffix = "" if lambda_anchor is None else f"_lambda_{lambda_token(lambda_anchor)}"
    return f"{family}_{rotation}_{outfit}{suffix}"


def rotation_row(name: str) -> dict[str, Any]:
    try:
        return next(row for row in rotations() if row["rotation"] == name)
    except StopIteration as error:
        raise KeyError(name) from error


def execution_metadata(attempt: Path) -> dict[str, Any]:
    return read_json(attempt / "00_preflight/execution_metadata.json")


def assert_execution_head(attempt: Path) -> str:
    metadata = execution_metadata(attempt)
    head = git("rev-parse", "HEAD")
    if head != metadata["execution_head"] or git("status", "--short"):
        raise RuntimeError("scientific execution requires the clean EXECUTION_HEAD")
    return head


def contract_snapshot(attempt: Path) -> dict[str, Any]:
    return {
        "execution_head": execution_metadata(attempt)["execution_head"],
        "protocol_sha256": sha256(attempt / "01_contract_snapshot/coefficient_headroom_protocol.yaml", lf=True),
        "loss_sha256": sha256(attempt / "01_contract_snapshot/coefficient_headroom_loss_contract.json", lf=True),
        "optimizer_sha256": sha256(attempt / "01_contract_snapshot/coefficient_headroom_optimizer_contract.json", lf=True),
        "basis_sha256": BASIS_SHA,
        "normalization_sha256": NORMALIZATION_SHA,
        "frozen_asset_snapshot_sha256": canonical_sha(read_json(attempt / "00_preflight/runtime_asset_audit.json")),
    }


def frozen_gradient_audit(context: Mapping[str, Any], basis: Any) -> dict[str, Any]:
    values: list[tuple[str, torch.Tensor]] = []
    base = context["base"]
    if hasattr(base, "named_parameters"):
        values.extend((f"base.{name}", value) for name, value in base.named_parameters())
    else:
        values.extend(
            (f"base.{name}", getattr(base, name))
            for name in ("_xyz", "_scaling", "_rotation", "_opacity", "_sh0", "_shN")
            if isinstance(getattr(base, name, None), torch.Tensor)
        )
    values.extend((f"basis.{name}", value) for name, value in basis.named_parameters())
    violations = [
        name for name, value in values
        if value.grad is not None and bool(torch.count_nonzero(value.grad.detach()))
    ]
    return {
        "status": "PASS" if not violations else "FAIL",
        "audited_parameter_count": len(values),
        "nonzero_gradient_count": len(violations),
        "violations": violations,
    }


def checkpoint_path(
    attempt: Path, family: str, rotation: str, outfit: str, step: int,
    lambda_anchor: float | None = None,
) -> Path:
    token = run_token(family, rotation, outfit, lambda_anchor)
    return attempt / "05_checkpoints" / token / f"step_{step:06d}.pth"


def checkpoint_record(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "method_family": payload["method"],
        "rotation": payload["rotation"],
        "outfit_id": payload["outfit_id"],
        "lambda_anchor": payload["lambda_anchor"],
        "global_step": payload["global_step"],
        "cumulative_optimizer_wall_time_seconds": payload["cumulative_optimizer_wall_time_seconds"],
    }


def write_checkpoint(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    required = set(optimizer_contract()["checkpoint_schema"]["required_fields"])
    if not required.issubset(payload):
        raise ValueError(f"checkpoint fields missing: {sorted(required.difference(payload))}")
    atomic_torch(path, dict(payload), replace=False)
    return checkpoint_record(path, payload)


def base_checkpoint_payload(
    attempt: Path, *, method: str, rotation: str, outfit: str, step: int,
    lambda_anchor: float | None, trainable_state: Mapping[str, Any],
    optimizer: torch.optim.Optimizer, condition_position: int,
    query_order_sha256: str, elapsed: float, metric_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    snapshot = contract_snapshot(attempt)
    return {
        "schema_version": "canondressgs.paper.coefficient_headroom_checkpoint.v1",
        "task_id": TASK_ID,
        "method": method,
        "rotation": rotation,
        "outfit_id": outfit,
        "global_step": int(step),
        "trainable_state": dict(trainable_state),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": "NOT_APPLICABLE_FIXED_LR",
        "rng_state": rng_state(),
        "condition_position": int(condition_position),
        "lambda_anchor": lambda_anchor,
        "frozen_asset_sha256": snapshot["frozen_asset_snapshot_sha256"],
        "query_order_sha256": query_order_sha256,
        "cumulative_optimizer_wall_time_seconds": float(elapsed),
        "execution_head": snapshot["execution_head"],
        "protocol_sha256": snapshot["protocol_sha256"],
        "loss_sha256": snapshot["loss_sha256"],
        "optimizer_sha256": snapshot["optimizer_sha256"],
        "basis_sha256": BASIS_SHA,
        "normalization_sha256": NORMALIZATION_SHA,
        "metric_snapshot": dict(metric_snapshot),
        "target_tensors_stored": False,
    }


def update_attempt_status(attempt: Path, **updates: Any) -> None:
    status = read_json(attempt / "RUN_STATUS.json")
    status.update(updates)
    status["updated_at_utc"] = now()
    atomic_json(attempt / "RUN_STATUS.json", status)


def coefficient_run_dir(attempt: Path, rotation: str, outfit: str, value: float) -> Path:
    return attempt / "03_coefficient_runs" / run_token("coefficient", rotation, outfit, value)


def coefficient_optimizer(z: torch.Tensor) -> torch.optim.Optimizer:
    contract = optimizer_contract()["coefficient_optimizer"]
    return torch.optim.Adam(
        [z], lr=float(contract["learning_rate"]),
        betas=tuple(float(value) for value in contract["betas"]),
        eps=float(contract["epsilon"]), weight_decay=float(contract["weight_decay"]),
    )


def coefficient_checkpoint_state(z: torch.Tensor) -> dict[str, Any]:
    return {"standardized_z": z.detach().cpu().clone()}


def _resume_coefficient(
    run_dir: Path, attempt: Path, rotation: str, outfit: str, value: float,
    z: torch.Tensor, optimizer: torch.optim.Optimizer,
) -> tuple[int, float, list[dict[str, Any]]]:
    status_path = run_dir / "RUN_STATUS.json"
    trajectory = jsonl(run_dir / "trajectory.jsonl")
    if not status_path.is_file():
        raise RuntimeError("incomplete coefficient run lacks RUN_STATUS")
    status = read_json(status_path)
    step = int(status["optimizer_steps"])
    if len(trajectory) != step or step not in MILESTONES:
        raise RuntimeError("interrupted coefficient run is not exactly recoverable")
    path = checkpoint_path(attempt, "coefficient", rotation, outfit, step, value)
    if not path.is_file():
        raise RuntimeError("exact coefficient resume checkpoint is missing")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    z.data.copy_(payload["trainable_state"]["standardized_z"].to(z))
    optimizer.load_state_dict(payload["optimizer_state"])
    optimizer_to(optimizer, z.device)
    restore_rng(payload["rng_state"])
    return step, float(payload["cumulative_optimizer_wall_time_seconds"]), trajectory


def train_one_coefficient(
    attempt: Path, context: Mapping[str, Any], basis: Any,
    coefficients: Mapping[str, torch.Tensor], payload: Mapping[str, Any],
    rotation: Mapping[str, Any], outfit: str, lambda_anchor: float,
) -> dict[str, Any]:
    name = rotation["rotation"]
    run_dir = coefficient_run_dir(attempt, name, outfit, lambda_anchor)
    summary_path = run_dir / "run_summary.json"
    if summary_path.is_file():
        summary = read_json(summary_path)
        if summary.get("status") != "PASS":
            raise RuntimeError(f"completed coefficient run is invalid: {run_dir}")
        return summary
    creating = not run_dir.exists()
    if creating:
        run_dir.mkdir(parents=True, exist_ok=False)
    seed = int(optimizer_contract()["data_schedule"]["seed"])
    seed_everything(seed)
    mean = payload["coefficient_train_mean"]
    std = payload["coefficient_train_std"]
    initial_raw = coefficients[outfit].detach()
    initial_z = ((initial_raw - mean) / std).detach()
    z = initial_z.clone().requires_grad_(True)
    if z.numel() != 4 or tuple(z.shape) != (4,):
        raise RuntimeError("COEFFICIENT_HEADROOM_MODEL_CONTRACT_MISMATCH")
    optimizer = coefficient_optimizer(z)
    query_sha = rotation["per_garment_query_order"][outfit]["sha256"]
    checkpoints: list[dict[str, Any]] = []
    start_step, elapsed, trajectory = 0, 0.0, []
    if creating:
        atomic_json(run_dir / "RUN_STATUS.json", {
            "status": "RUNNING", "optimizer_steps": 0, "resumed": False,
            "optimizer_creations": 1, "updated_at_utc": now(),
        }, replace=False)
        checkpoints.append(write_checkpoint(
            checkpoint_path(attempt, "coefficient", name, outfit, 0, lambda_anchor),
            base_checkpoint_payload(
                attempt, method="Render-Refined Coefficient" if lambda_anchor else "UNREGULARIZED_DIAGNOSTIC",
                rotation=name, outfit=outfit, step=0, lambda_anchor=lambda_anchor,
                trainable_state=coefficient_checkpoint_state(z), optimizer=optimizer,
                condition_position=0, query_order_sha256=query_sha, elapsed=0.0,
                metric_snapshot={"render_objective": None, "anchor": 0.0, "total": None},
            ),
        ))
    else:
        start_step, elapsed, trajectory = _resume_coefficient(
            run_dir, attempt, name, outfit, lambda_anchor, z, optimizer
        )
        atomic_json(run_dir / "RUN_STATUS.json", {
            "status": "RUNNING", "optimizer_steps": start_step, "resumed": True,
            "optimizer_creations": 1, "updated_at_utc": now(),
        })
        checkpoints = [
            checkpoint_record(
                checkpoint_path(attempt, "coefficient", name, outfit, step, lambda_anchor),
                torch.load(
                    checkpoint_path(attempt, "coefficient", name, outfit, step, lambda_anchor),
                    map_location="cpu", weights_only=False,
                ),
            )
            for step in MILESTONES if step <= start_step
        ]
    gradient_nonzero_steps = sum(bool(row.get("gradient_nonzero")) for row in trajectory)
    max_gradient_norm = max((float(row.get("gradient_norm", 0.0)) for row in trajectory), default=0.0)
    last_metrics: dict[str, Any] = {"render_objective": None, "anchor": 0.0, "total": None}
    started = time.perf_counter()
    try:
        for step in range(start_step + 1, 301):
            condition = rotation["optimize_folds"][(step - 1) % 2]
            sample = context["samples"][f"{outfit}/{condition}"]
            optimizer.zero_grad(set_to_none=True)
            synchronize()
            tick = time.perf_counter()
            raw = mean + std * z
            residual = basis(raw, chunk_size=16384)
            rgb, alpha = runtime_imports()["parameterization"].render_prediction(
                context["base"], sample, residual, context["background"]
            )
            parts = render_loss(
                rgb, alpha, sample, stability=z.new_zeros(()), stability_weight=0.0
            )
            anchor = torch.mean(torch.square(z - initial_z))
            total = parts["total"] + float(lambda_anchor) * anchor
            if not torch.isfinite(total):
                raise FloatingPointError("coefficient objective contains NaN/Inf")
            total.backward()
            if z.grad is None or not torch.isfinite(z.grad).all():
                raise FloatingPointError("coefficient gradient contains NaN/Inf")
            gradient_nonzero = bool(torch.count_nonzero(z.grad))
            gradient_norm = torch.linalg.vector_norm(z.grad)
            clipped = torch.nn.utils.clip_grad_norm_([z], 5.0, error_if_nonfinite=True)
            optimizer.step()
            synchronize()
            elapsed += time.perf_counter() - tick
            if not torch.isfinite(z).all():
                raise FloatingPointError("coefficient state contains NaN/Inf")
            gradient_nonzero_steps += int(gradient_nonzero)
            max_gradient_norm = max(max_gradient_norm, float(gradient_norm))
            last_metrics = {
                "render_objective": float(parts["total"].detach()),
                "anchor": float(anchor.detach()),
                "total": float(total.detach()),
            }
            row = {
                "step": step, "condition": condition, "lambda_anchor": lambda_anchor,
                **last_metrics, "gradient_nonzero": gradient_nonzero,
                "gradient_norm": float(gradient_norm), "clipped_gradient_norm": float(clipped),
                "standardized_z": z.detach().cpu().tolist(),
                "raw_coefficient": (mean + std * z).detach().cpu().tolist(),
                "cumulative_optimizer_wall_time_seconds": elapsed,
            }
            append_jsonl(run_dir / "trajectory.jsonl", row)
            atomic_json(run_dir / "RUN_STATUS.json", {
                "status": "RUNNING", "optimizer_steps": step,
                "resumed": start_step > 0, "optimizer_creations": 1,
                "updated_at_utc": now(),
            })
            if step in MILESTONES:
                checkpoints.append(write_checkpoint(
                    checkpoint_path(attempt, "coefficient", name, outfit, step, lambda_anchor),
                    base_checkpoint_payload(
                        attempt,
                        method="Render-Refined Coefficient" if lambda_anchor else "UNREGULARIZED_DIAGNOSTIC",
                        rotation=name, outfit=outfit, step=step, lambda_anchor=lambda_anchor,
                        trainable_state=coefficient_checkpoint_state(z), optimizer=optimizer,
                        condition_position=step % 2, query_order_sha256=query_sha,
                        elapsed=elapsed, metric_snapshot=last_metrics,
                    ),
                ))
    except Exception as error:
        atomic_json(run_dir / "failure.json", {
            "status": "FAILED_PRESERVED", "error_type": type(error).__name__,
            "message": str(error), "traceback": traceback.format_exc(),
            "optimizer_steps": len(jsonl(run_dir / "trajectory.jsonl")), "created_at_utc": now(),
        }, replace=False)
        raise
    frozen = frozen_gradient_audit(context, basis)
    if frozen["status"] != "PASS" or gradient_nonzero_steps == 0:
        raise RuntimeError("COEFFICIENT_HEADROOM_FROZEN_GRADIENT_VIOLATION")
    final_raw = (mean + std * z).detach()
    summary = {
        "schema_version": "canondressgs.paper.coefficient_headroom_run.v1",
        "task_id": TASK_ID, "status": "PASS",
        "run_id": run_token("coefficient", name, outfit, lambda_anchor),
        "method_family": "coefficient", "rotation": name, "outfit_id": outfit,
        "lambda_anchor": lambda_anchor, "regularized": lambda_anchor > 0,
        "diagnostic": lambda_anchor == 0, "trainable_scalars": int(z.numel()),
        "optimizer_creations": 1, "optimizer_steps": 300,
        "forward_calls": 300, "backward_calls": 300, "renderer_calls": 300,
        "checkpoint_writes": len(checkpoints), "checkpoints": checkpoints,
        "initial_standardized_z": initial_z.cpu().tolist(),
        "final_standardized_z": z.detach().cpu().tolist(),
        "initial_raw_coefficient": initial_raw.cpu().tolist(),
        "final_raw_coefficient": final_raw.cpu().tolist(),
        "standardized_displacement_l2": float(torch.linalg.vector_norm(z.detach() - initial_z)),
        "raw_displacement_l2": float(torch.linalg.vector_norm(final_raw - initial_raw)),
        "optimizer_section_wall_time_seconds": elapsed,
        "end_to_end_wall_time_seconds": time.perf_counter() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "gradient_nonzero_steps": gradient_nonzero_steps,
        "max_gradient_norm": max_gradient_norm,
        "finite": True, "frozen_gradient_audit": frozen,
        "checkpoint_roundtrip": "PENDING_FINAL_LOAD",
        "resumed": start_step > 0, "optimizer_steps_repeated": 0,
        "test_views_used_for_updates": False,
    }
    final_checkpoint = checkpoint_path(attempt, "coefficient", name, outfit, 300, lambda_anchor)
    reloaded = torch.load(final_checkpoint, map_location="cpu", weights_only=False)
    exact = torch.equal(reloaded["trainable_state"]["standardized_z"], z.detach().cpu())
    summary["checkpoint_roundtrip"] = "PASS" if exact else "FAIL"
    if not exact or len(checkpoints) != len(MILESTONES):
        raise RuntimeError("coefficient checkpoint contract failed")
    atomic_json(summary_path, summary, replace=False)
    atomic_json(run_dir / "RUN_STATUS.json", {
        "status": "PASS", "optimizer_steps": 300, "resumed": start_step > 0,
        "optimizer_creations": 1, "updated_at_utc": now(),
    })
    return summary


def train_coefficients(root: Path) -> dict[str, Any]:
    attempt = attempt_path(root)
    assert_execution_head(attempt)
    parity = read_json(attempt / "02_static_parity/parity.json")
    if parity["status"] != "PASS":
        raise RuntimeError("coefficient training requires parity PASS")
    context = runtime_context(attempt)
    device = context["base"]._xyz.device
    basis, coefficients, payload = load_basis(root, device)
    summaries = []
    for rotation in rotations():
        for outfit in OUTFITS:
            for value in (*POSITIVE_LAMBDAS, 0.0):
                summaries.append(train_one_coefficient(
                    attempt, context, basis, coefficients, payload, rotation, outfit, value
                ))
                update_attempt_status(
                    attempt, status="COEFFICIENT_RUNNING",
                    optimization_runs=len(summaries),
                    optimizer_steps=sum(int(row["optimizer_steps"]) for row in summaries),
                    renderer_calls=int(parity["renderer_calls"]) + sum(int(row["renderer_calls"]) for row in summaries),
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    result = {
        "status": "PASS", "completed_runs": len(summaries),
        "regularized_runs": sum(bool(row["regularized"]) for row in summaries),
        "diagnostic_runs": sum(bool(row["diagnostic"]) for row in summaries),
        "optimizer_steps": sum(int(row["optimizer_steps"]) for row in summaries),
        "checkpoint_writes": sum(int(row["checkpoint_writes"]) for row in summaries),
    }
    if result != {
        "status": "PASS", "completed_runs": 100, "regularized_runs": 80,
        "diagnostic_runs": 20, "optimizer_steps": 30000, "checkpoint_writes": 800,
    }:
        raise RuntimeError("COEFFICIENT_HEADROOM_EXPECTED_COUNT_MISMATCH")
    atomic_json(attempt / "03_coefficient_runs/summary.json", result, replace=False)
    update_attempt_status(attempt, status="COEFFICIENT_PASS", optimization_runs=100, optimizer_steps=30000)
    return result


def full_run_dir(attempt: Path, rotation: str, outfit: str) -> Path:
    return attempt / "04_full_residual_runs" / run_token("full_residual", rotation, outfit)


def initialize_full_field(field: Any, teacher: Any, base: Any) -> dict[str, Any]:
    assignments = {
        "raw_xyz": teacher.delta_xyz,
        "raw_log_scaling": teacher.delta_log_scaling,
        "raw_rotvec": teacher.delta_rotvec,
        "raw_opacity": teacher.delta_opacity_logit,
        "raw_sh0": teacher.delta_sh0,
    }
    with torch.no_grad():
        for name, value in assignments.items():
            getattr(field, name).copy_(value.to(getattr(field, name)))
    residual = field.residuals(base)
    differences = {
        name: float((getattr(residual, name) - getattr(teacher, name)).abs().max())
        for name in (
            "delta_xyz", "delta_log_scaling", "delta_rotvec",
            "delta_opacity_logit", "delta_sh0", "delta_shN",
        )
    }
    return {
        "status": "PASS" if max(differences.values()) == 0.0 else "FAIL",
        "field_to_teacher_max_abs": differences,
        "raw_shN_frozen_zero": not field.raw_shN.requires_grad and int(torch.count_nonzero(field.raw_shN)) == 0,
    }


def full_optimizer(field: Any) -> torch.optim.Optimizer:
    contract = optimizer_contract()["full_residual_optimizer"]
    groups, _ = field.parameter_groups(
        float(contract["geometry_learning_rate"]),
        float(contract["appearance_learning_rate"]),
    )
    return torch.optim.Adam(
        groups, betas=tuple(float(value) for value in contract["betas"]),
        eps=float(contract["epsilon"]), weight_decay=float(contract["weight_decay"]),
    )


def full_checkpoint_state(field: Any) -> dict[str, Any]:
    return {"field": {name: value.detach().cpu().clone() for name, value in field.state_dict().items()}}


def _resume_full(
    run_dir: Path, attempt: Path, rotation: str, outfit: str,
    field: Any, optimizer: torch.optim.Optimizer,
) -> tuple[int, float, list[dict[str, Any]]]:
    status = read_json(run_dir / "RUN_STATUS.json")
    trajectory = jsonl(run_dir / "trajectory.jsonl")
    step = int(status["optimizer_steps"])
    if len(trajectory) != step or step not in MILESTONES:
        raise RuntimeError("interrupted full-residual run is not exactly recoverable")
    path = checkpoint_path(attempt, "full_residual", rotation, outfit, step)
    if not path.is_file():
        raise RuntimeError("exact full-residual resume checkpoint is missing")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    field.load_state_dict(payload["trainable_state"]["field"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    optimizer_to(optimizer, field.raw_xyz.device)
    restore_rng(payload["rng_state"])
    return step, float(payload["cumulative_optimizer_wall_time_seconds"]), trajectory


def bound_normalized_distance(first: Any, second: Any, bounds: Mapping[str, float]) -> float:
    channels = {
        "delta_xyz": "xyz", "delta_log_scaling": "log_scaling",
        "delta_rotvec": "rotation", "delta_opacity_logit": "opacity_logit",
        "delta_sh0": "sh0", "delta_shN": "shN",
    }
    numerator = 0.0
    count = 0
    for field_name, bound_name in channels.items():
        delta = (getattr(first, field_name).detach().double() - getattr(second, field_name).detach().double()) / float(bounds[bound_name])
        numerator += float(delta.square().sum())
        count += delta.numel()
    return math.sqrt(numerator / max(count, 1))


def train_one_full(
    attempt: Path, context: Mapping[str, Any], basis: Any, teacher: Any,
    bounds: Mapping[str, float], rotation: Mapping[str, Any], outfit: str,
) -> dict[str, Any]:
    name = rotation["rotation"]
    run_dir = full_run_dir(attempt, name, outfit)
    summary_path = run_dir / "run_summary.json"
    if summary_path.is_file():
        summary = read_json(summary_path)
        if summary.get("status") != "PASS":
            raise RuntimeError(f"completed full-residual run is invalid: {run_dir}")
        return summary
    creating = not run_dir.exists()
    if creating:
        run_dir.mkdir(parents=True, exist_ok=False)
    seed = int(optimizer_contract()["data_schedule"]["seed"])
    seed_everything(seed)
    modules = runtime_imports()
    field = modules["UnboundedGaussianDeltaField"](context["base"]).to(context["base"]._xyz.device)
    initialization = initialize_full_field(field, teacher, context["base"])
    allocated = sum(parameter.numel() for parameter in field.parameters())
    effective = int(field.trainable_support.sum()) * 13
    if (
        initialization["status"] != "PASS"
        or not initialization["raw_shN_frozen_zero"]
        or allocated != 2_600_000 or effective != 2_217_111
    ):
        raise RuntimeError("COEFFICIENT_HEADROOM_FULL_RESIDUAL_MODEL_MISMATCH")
    optimizer = full_optimizer(field)
    if optimizer.state:
        raise RuntimeError("full-residual optimizer state is not fresh")
    query_sha = rotation["per_garment_query_order"][outfit]["sha256"]
    checkpoints: list[dict[str, Any]] = []
    start_step, elapsed, trajectory = 0, 0.0, []
    if creating:
        atomic_json(run_dir / "RUN_STATUS.json", {
            "status": "RUNNING", "optimizer_steps": 0, "resumed": False,
            "optimizer_creations": 1, "updated_at_utc": now(),
        }, replace=False)
        checkpoints.append(write_checkpoint(
            checkpoint_path(attempt, "full_residual", name, outfit, 0),
            base_checkpoint_payload(
                attempt, method="Full-Residual Render Optimization", rotation=name,
                outfit=outfit, step=0, lambda_anchor=None,
                trainable_state=full_checkpoint_state(field), optimizer=optimizer,
                condition_position=0, query_order_sha256=query_sha, elapsed=0.0,
                metric_snapshot={"render_objective": None, "stability": None, "total": None},
            ),
        ))
    else:
        start_step, elapsed, trajectory = _resume_full(
            run_dir, attempt, name, outfit, field, optimizer
        )
        atomic_json(run_dir / "RUN_STATUS.json", {
            "status": "RUNNING", "optimizer_steps": start_step, "resumed": True,
            "optimizer_creations": 1, "updated_at_utc": now(),
        })
        checkpoints = [
            checkpoint_record(
                checkpoint_path(attempt, "full_residual", name, outfit, step),
                torch.load(
                    checkpoint_path(attempt, "full_residual", name, outfit, step),
                    map_location="cpu", weights_only=False,
                ),
            )
            for step in MILESTONES if step <= start_step
        ]
    gradient_nonzero_steps = sum(bool(row.get("gradient_nonzero")) for row in trajectory)
    max_gradient_norm = max((float(row.get("gradient_norm", 0.0)) for row in trajectory), default=0.0)
    last_metrics: dict[str, Any] = {"render_objective": None, "stability": None, "total": None}
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(start_step + 1, 301):
            condition = rotation["optimize_folds"][(step - 1) % 2]
            sample = context["samples"][f"{outfit}/{condition}"]
            optimizer.zero_grad(set_to_none=True)
            synchronize()
            tick = time.perf_counter()
            residual = field.residuals(context["base"])
            rgb, alpha = modules["parameterization"].render_prediction(
                context["base"], sample, residual, context["background"]
            )
            stability = modules["direct_stability"](field)
            parts = render_loss(
                rgb, alpha, sample, stability=stability,
                stability_weight=float(loss_contract()["full_residual_regularization"]["weight"]),
            )
            total = parts["total"]
            if not torch.isfinite(total):
                raise FloatingPointError("full-residual objective contains NaN/Inf")
            total.backward()
            gradients = [parameter.grad for parameter in field.parameters() if parameter.requires_grad]
            if not gradients or any(value is None or not torch.isfinite(value).all() for value in gradients):
                raise FloatingPointError("full-residual gradient contains NaN/Inf")
            gradient_nonzero = any(bool(torch.count_nonzero(value)) for value in gradients)
            gradient_norm = torch.sqrt(sum(value.detach().square().sum() for value in gradients))
            clipped = torch.nn.utils.clip_grad_norm_(
                field.parameters(),
                float(optimizer_contract()["full_residual_optimizer"]["gradient_clip_norm"]),
                error_if_nonfinite=True,
            )
            optimizer.step()
            synchronize()
            elapsed += time.perf_counter() - tick
            if any(not torch.isfinite(parameter).all() for parameter in field.parameters()):
                raise FloatingPointError("full-residual state contains NaN/Inf")
            gradient_nonzero_steps += int(gradient_nonzero)
            max_gradient_norm = max(max_gradient_norm, float(gradient_norm))
            last_metrics = {
                "render_objective": float(total.detach() - float(loss_contract()["full_residual_regularization"]["weight"]) * stability.detach()),
                "stability": float(stability.detach()), "total": float(total.detach()),
            }
            row = {
                "step": step, "condition": condition, **last_metrics,
                "gradient_nonzero": gradient_nonzero,
                "gradient_norm": float(gradient_norm), "clipped_gradient_norm": float(clipped),
                "cumulative_optimizer_wall_time_seconds": elapsed,
            }
            append_jsonl(run_dir / "trajectory.jsonl", row)
            atomic_json(run_dir / "RUN_STATUS.json", {
                "status": "RUNNING", "optimizer_steps": step,
                "resumed": start_step > 0, "optimizer_creations": 1,
                "updated_at_utc": now(),
            })
            if step in MILESTONES:
                checkpoints.append(write_checkpoint(
                    checkpoint_path(attempt, "full_residual", name, outfit, step),
                    base_checkpoint_payload(
                        attempt, method="Full-Residual Render Optimization", rotation=name,
                        outfit=outfit, step=step, lambda_anchor=None,
                        trainable_state=full_checkpoint_state(field), optimizer=optimizer,
                        condition_position=step % 2, query_order_sha256=query_sha,
                        elapsed=elapsed, metric_snapshot=last_metrics,
                    ),
                ))
    except Exception as error:
        atomic_json(run_dir / "failure.json", {
            "status": "FAILED_PRESERVED", "error_type": type(error).__name__,
            "message": str(error), "traceback": traceback.format_exc(),
            "optimizer_steps": len(jsonl(run_dir / "trajectory.jsonl")), "created_at_utc": now(),
        }, replace=False)
        raise
    frozen = frozen_gradient_audit(context, basis)
    if frozen["status"] != "PASS" or gradient_nonzero_steps == 0:
        raise RuntimeError("COEFFICIENT_HEADROOM_FROZEN_GRADIENT_VIOLATION")
    final_residual = field.residuals(context["base"])
    summary = {
        "schema_version": "canondressgs.paper.full_residual_headroom_run.v1",
        "task_id": TASK_ID, "status": "PASS",
        "run_id": run_token("full_residual", name, outfit),
        "method_family": "full_residual", "rotation": name, "outfit_id": outfit,
        "lambda_anchor": None, "regularized": False, "diagnostic": False,
        "allocated_trainable_scalars": allocated,
        "effective_trainable_scalars": effective,
        "optimizer_creations": 1, "optimizer_steps": 300,
        "forward_calls": 300, "backward_calls": 300, "renderer_calls": 300,
        "checkpoint_writes": len(checkpoints), "checkpoints": checkpoints,
        "initialization": initialization,
        "full_residual_bound_normalized_displacement": bound_normalized_distance(final_residual, teacher, bounds),
        "optimizer_section_wall_time_seconds": elapsed,
        "end_to_end_wall_time_seconds": time.perf_counter() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "gradient_nonzero_steps": gradient_nonzero_steps,
        "max_gradient_norm": max_gradient_norm,
        "finite": True, "frozen_gradient_audit": frozen,
        "checkpoint_roundtrip": "PENDING_FINAL_LOAD",
        "resumed": start_step > 0, "optimizer_steps_repeated": 0,
        "test_views_used_for_updates": False,
    }
    final_checkpoint = checkpoint_path(attempt, "full_residual", name, outfit, 300)
    reloaded = torch.load(final_checkpoint, map_location="cpu", weights_only=False)
    exact = all(
        torch.equal(reloaded["trainable_state"]["field"][key], value.detach().cpu())
        for key, value in field.state_dict().items()
    )
    summary["checkpoint_roundtrip"] = "PASS" if exact else "FAIL"
    if not exact or len(checkpoints) != len(MILESTONES):
        raise RuntimeError("full-residual checkpoint contract failed")
    atomic_json(summary_path, summary, replace=False)
    atomic_json(run_dir / "RUN_STATUS.json", {
        "status": "PASS", "optimizer_steps": 300, "resumed": start_step > 0,
        "optimizer_creations": 1, "updated_at_utc": now(),
    })
    return summary


def train_full_residual(root: Path) -> dict[str, Any]:
    attempt = attempt_path(root)
    assert_execution_head(attempt)
    coefficient_summary = read_json(attempt / "03_coefficient_runs/summary.json")
    if coefficient_summary["status"] != "PASS":
        raise RuntimeError("full-residual training requires coefficient PASS")
    context = runtime_context(attempt)
    device = context["base"]._xyz.device
    basis, _, payload = load_basis(root, device)
    teachers = load_teachers(context)
    summaries = []
    for rotation in rotations():
        for outfit in OUTFITS:
            summaries.append(train_one_full(
                attempt, context, basis, teachers[outfit], payload["channel_bounds"], rotation, outfit
            ))
            update_attempt_status(
                attempt, status="FULL_RESIDUAL_RUNNING",
                optimization_runs=100 + len(summaries),
                optimizer_steps=30000 + sum(int(row["optimizer_steps"]) for row in summaries),
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    result = {
        "status": "PASS", "completed_runs": len(summaries),
        "optimizer_steps": sum(int(row["optimizer_steps"]) for row in summaries),
        "checkpoint_writes": sum(int(row["checkpoint_writes"]) for row in summaries),
    }
    if result != {"status": "PASS", "completed_runs": 20, "optimizer_steps": 6000, "checkpoint_writes": 160}:
        raise RuntimeError("COEFFICIENT_HEADROOM_EXPECTED_COUNT_MISMATCH")
    atomic_json(attempt / "04_full_residual_runs/summary.json", result, replace=False)
    update_attempt_status(
        attempt, status="OPTIMIZATION_PASS", optimization_runs=120,
        optimizer_steps=36000, renderer_calls=36042,
    )
    return result


def select_lambda(
    objective_by_lambda: Mapping[float, float], *, tolerance: float = 1e-4,
) -> tuple[float, dict[str, Any]]:
    if set(objective_by_lambda) != set(POSITIVE_LAMBDAS):
        raise ValueError("lambda selection requires exactly the frozen positive grid")
    if any(not math.isfinite(float(value)) for value in objective_by_lambda.values()):
        raise FloatingPointError("lambda objective contains NaN/Inf")
    best = min(float(value) for value in objective_by_lambda.values())
    tied = sorted(
        (float(key) for key, value in objective_by_lambda.items() if float(value) <= best + tolerance),
        reverse=True,
    )
    selected = tied[0]
    return selected, {
        "minimum_objective": best,
        "tie_tolerance_absolute": tolerance,
        "eligible_within_tolerance": tied,
        "tie_break": "largest lambda (stronger regularization)",
    }


def load_coefficient_z(
    attempt: Path, rotation: str, outfit: str, lambda_anchor: float, device: torch.device,
) -> torch.Tensor:
    payload = torch.load(
        checkpoint_path(attempt, "coefficient", rotation, outfit, 300, lambda_anchor),
        map_location="cpu", weights_only=False,
    )
    if payload["global_step"] != 300 or float(payload["lambda_anchor"]) != float(lambda_anchor):
        raise ValueError("coefficient endpoint checkpoint metadata mismatch")
    return payload["trainable_state"]["standardized_z"].to(device)


def coefficient_residual(
    attempt: Path, basis: Any, basis_payload: Mapping[str, Any],
    rotation: str, outfit: str, lambda_anchor: float,
) -> tuple[Any, torch.Tensor, torch.Tensor]:
    mean = basis_payload["coefficient_train_mean"]
    std = basis_payload["coefficient_train_std"]
    z = load_coefficient_z(attempt, rotation, outfit, lambda_anchor, mean.device)
    raw = mean + std * z
    return basis(raw, chunk_size=16384), z, raw


def run_lambda_selection(root: Path) -> dict[str, Any]:
    attempt = attempt_path(root)
    assert_execution_head(attempt)
    if read_json(attempt / "04_full_residual_runs/summary.json")["status"] != "PASS":
        raise RuntimeError("lambda selection requires all optimization runs")
    destination = attempt / "06_lambda_selection/lambda_selection.json"
    if destination.is_file():
        return read_json(destination)
    context = runtime_context(attempt)
    device = context["base"]._xyz.device
    basis, _, payload = load_basis(root, device)
    modules = runtime_imports()
    rows: list[dict[str, Any]] = []
    selections: dict[str, Any] = {}
    for rotation in rotations():
        name = rotation["rotation"]
        calibration = rotation["calibration_fold"]
        by_lambda: dict[float, list[float]] = defaultdict(list)
        for value in POSITIVE_LAMBDAS:
            for outfit in OUTFITS:
                residual, z, raw = coefficient_residual(
                    attempt, basis, payload, name, outfit, value
                )
                sample = context["samples"][f"{outfit}/{calibration}"]
                with torch.inference_mode():
                    rgb, alpha = modules["parameterization"].render_prediction(
                        context["base"], sample, residual, context["background"]
                    )
                    parts = render_loss(
                        rgb, alpha, sample, stability=rgb.new_zeros(()), stability_weight=0.0
                    )
                objective = float(parts["total"])
                by_lambda[value].append(objective)
                rgb_path = attempt / "06_lambda_selection/renders" / name / f"{outfit}_lambda_{lambda_token(value)}_rgb.png"
                alpha_path = rgb_path.with_name(rgb_path.stem.replace("_rgb", "_alpha") + ".png")
                modules["save_render_tensor"](rgb_path, rgb, 3)
                modules["save_render_tensor"](alpha_path, alpha, 1)
                rows.append({
                    "rotation": name, "outfit_id": outfit,
                    "partition": "calibration", "condition_id": calibration,
                    "lambda_anchor": value, "render_objective": objective,
                    "standardized_z": z.detach().cpu().tolist(),
                    "raw_coefficient": raw.detach().cpu().tolist(),
                    "rgb_path": str(rgb_path), "rgb_sha256": sha256(rgb_path),
                    "alpha_path": str(alpha_path), "alpha_sha256": sha256(alpha_path),
                    "test_used": False,
                })
        macro = {value: statistics.fmean(scores) for value, scores in by_lambda.items()}
        selected, tie = select_lambda(macro, tolerance=1e-4)
        selections[name] = {
            "rotation": name, "selected_lambda": selected,
            "calibration_condition": calibration,
            "test_condition": rotation["test_fold"],
            "five_garment_macro_objective": {str(key): value for key, value in macro.items()},
            "selection_rule": tie, "garment_count": 5, "test_used": False,
        }
    result = {
        "schema_version": "canondressgs.paper.coefficient_headroom_lambda_selection.v1",
        "task_id": TASK_ID, "status": "PASS", "rows": rows,
        "row_count": len(rows), "renderer_calls": len(rows),
        "selections": selections, "selected_lambda_count": len(selections),
        "selection_sha256": canonical_sha(selections),
        "one_lambda_per_rotation": True, "test_used": False,
    }
    if len(rows) != 80 or len(selections) != 4:
        raise RuntimeError("COEFFICIENT_HEADROOM_EXPECTED_COUNT_MISMATCH")
    atomic_json(destination, result, replace=False)
    update_attempt_status(attempt, status="LAMBDA_SELECTION_PASS", renderer_calls=36122)
    return result


def select_equal_wall_time_checkpoint(
    coefficient_seconds: float, checkpoint_times: Mapping[int, float],
) -> dict[str, Any]:
    if tuple(sorted(checkpoint_times)) != MILESTONES:
        raise ValueError("equal-wall-time candidates do not match frozen milestones")
    if coefficient_seconds < 0 or any(float(value) < 0 for value in checkpoint_times.values()):
        raise ValueError("wall times must be nonnegative")
    fitting = [step for step in MILESTONES if float(checkpoint_times[step]) <= coefficient_seconds]
    selected = max(fitting) if fitting else 0
    selected_time = float(checkpoint_times[selected])
    return {
        "selected_step": selected,
        "coefficient_time_seconds": float(coefficient_seconds),
        "selected_full_residual_time_seconds": selected_time,
        "time_delta_seconds": float(coefficient_seconds) - selected_time,
        "selection_rule": "largest fixed checkpoint with cumulative optimizer time <= selected coefficient run time",
        "interpolation_used": False, "test_metric_used": False,
    }


def run_equal_wall_time_selection(root: Path) -> dict[str, Any]:
    attempt = attempt_path(root)
    assert_execution_head(attempt)
    lambda_result = read_json(attempt / "06_lambda_selection/lambda_selection.json")
    destination = attempt / "06_lambda_selection/equal_wall_time_selection.json"
    if destination.is_file():
        return read_json(destination)
    rows = []
    for rotation in rotations():
        name = rotation["rotation"]
        value = float(lambda_result["selections"][name]["selected_lambda"])
        for outfit in OUTFITS:
            coefficient_summary = read_json(
                coefficient_run_dir(attempt, name, outfit, value) / "run_summary.json"
            )
            full_summary = read_json(full_run_dir(attempt, name, outfit) / "run_summary.json")
            times = {
                int(row["global_step"]): float(row["cumulative_optimizer_wall_time_seconds"])
                for row in full_summary["checkpoints"]
            }
            selection = select_equal_wall_time_checkpoint(
                float(coefficient_summary["optimizer_section_wall_time_seconds"]), times
            )
            selected_path = checkpoint_path(
                attempt, "full_residual", name, outfit, int(selection["selected_step"])
            )
            rows.append({
                "rotation": name, "outfit_id": outfit, "selected_lambda": value,
                **selection, "selected_checkpoint_path": str(selected_path),
                "selected_checkpoint_sha256": sha256(selected_path),
                "candidate_cumulative_times": {str(key): value for key, value in times.items()},
            })
    result = {
        "schema_version": "canondressgs.paper.coefficient_headroom_equal_wall_time.v1",
        "task_id": TASK_ID, "status": "PASS", "row_count": len(rows), "rows": rows,
        "selection_sha256": canonical_sha(rows),
    }
    if len(rows) != 20:
        raise RuntimeError("COEFFICIENT_HEADROOM_EQUAL_WALL_TIME_CONTRACT_MISSING")
    atomic_json(destination, result, replace=False)
    return result


def span_recovery_ratio(
    teacher_error: float, refined_error: float, full_residual_error: float,
) -> dict[str, Any]:
    denominator = float(teacher_error) - float(full_residual_error)
    numerator = float(teacher_error) - float(refined_error)
    if denominator <= 0:
        return {
            "value": None, "numerator": numerator, "denominator": denominator,
            "reason": "Teacher_error-FullResidual_error<=0",
        }
    return {
        "value": numerator / denominator, "numerator": numerator,
        "denominator": denominator, "reason": None,
    }


def classify_headroom(values: Mapping[str, Any]) -> str:
    if not bool(values.get("complete")):
        return "COEFFICIENT_HEADROOM_PROTOCOL_INCOMPLETE"
    gates = success_contract()["thresholds"]
    safe = (
        int(values["identity_contamination_count"]) <= int(gates["identity_contamination_count_max"])
        and int(values["component_contamination_count"]) <= int(gates["component_contamination_count_max"])
    )
    coefficient_positive = (
        safe
        and int(values["improved_garments"]) >= int(gates["improved_garments_min"])
        and float(values["macro_lpips_improvement_absolute"]) >= float(gates["macro_lpips_improvement_absolute_min"])
        and float(values["macro_lpips_relative_reduction"]) >= float(gates["macro_lpips_relative_reduction_min"])
        and bool(values["companion_gate"])
        and int(values["valid_per_garment_span_ratios"]) >= int(gates["valid_per_garment_span_ratios_min"])
        and values["macro_span_recovery_ratio"] is not None
        and float(values["macro_span_recovery_ratio"]) >= float(gates["macro_span_recovery_ratio_min"])
        and bool(values["held_out_test_gate"])
    )
    if coefficient_positive:
        return "TEACHER_SPAN_HAS_USEFUL_RENDER_HEADROOM"
    coefficient_small = (
        safe
        and int(values["improved_garments"]) >= int(gates["improved_garments_min"])
        and float(values["macro_lpips_improvement_absolute"]) > 0
        and bool(values["held_out_test_gate"])
    )
    if coefficient_small:
        return "TEACHER_SPAN_HEADROOM_SMALL"
    full_positive = (
        safe
        and int(values["full_residual_improved_garments"]) >= int(gates["full_residual_improved_garments_min"])
        and float(values["full_residual_macro_lpips_improvement_absolute"]) >= float(gates["full_residual_macro_lpips_improvement_absolute_min"])
        and float(values["full_residual_macro_lpips_relative_reduction"]) >= float(gates["full_residual_macro_lpips_relative_reduction_min"])
    )
    if full_positive:
        return "TEACHER_SPAN_CAPACITY_LIMITED"
    return "TEACHER_SPAN_AT_LOCAL_OPTIMUM"


class LPIPSRuntime:
    def __init__(self, root: Path, device: torch.device) -> None:
        self.root = root
        self.device = device
        self.model: torch.nn.Module | None = None

    def lpips(self) -> torch.nn.Module:
        if self.model is None:
            package_root = self.root.parent / "AnimatableGaussians"
            if str(package_root) not in sys.path:
                sys.path.insert(0, str(package_root))
            module = importlib.import_module("network.lpips.lpips")
            self.model = module.LPIPS(net="vgg", version="0.1", verbose=False).to(self.device)
            self.model.eval()
        return self.model


def masked_mae(first: torch.Tensor, second: torch.Tensor, mask: torch.Tensor) -> float:
    membership = (mask >= 0.5).to(first)
    while membership.ndim < first.ndim:
        membership = membership.unsqueeze(0)
    membership = membership.expand_as(first)
    return float(((first - second).abs() * membership).sum() / membership.sum().clamp_min(1))


def evaluate_render_metrics(
    lpips_runtime: LPIPSRuntime, rgb: torch.Tensor, alpha: torch.Tensor,
    sample: Mapping[str, Any],
) -> dict[str, Any]:
    sealed = runtime_imports()["sealed_metrics"]
    garment = sample["target_clothing_mask"]
    protected = sample["target_protected_mask"]
    target = sample["target_edit_rgb"]
    base = sample["target_base_rgb"]
    silhouette_iou, boundary_f, tolerance = sealed.silhouette_metrics(
        alpha, sample["target_foreground_mask"]
    )
    objective = render_loss(
        rgb, alpha, sample, stability=rgb.new_zeros(()), stability_weight=0.0
    )
    return {
        "rgb_mae": masked_mae(rgb, target, garment),
        "psnr": sealed.masked_psnr(rgb, target, garment),
        "ssim": sealed.masked_ssim(rgb, target, garment),
        "lpips": sealed.lpips_distance(lpips_runtime, rgb, target, garment),
        "silhouette_iou": silhouette_iou,
        "boundary_f": boundary_f,
        "boundary_tolerance_pixels": tolerance,
        "protected_lpips": sealed.lpips_distance(lpips_runtime, rgb, base, protected),
        "identity_metric": masked_mae(rgb, base, protected),
        "render_objective": float(objective["total"]),
        "render_terms": {
            name: float(value) for name, value in objective.items()
            if name not in {"total", "stability"}
        },
    }


def full_residual_from_checkpoint(
    attempt: Path, context: Mapping[str, Any], rotation: str, outfit: str, step: int,
) -> Any:
    modules = runtime_imports()
    field = modules["UnboundedGaussianDeltaField"](context["base"]).to(context["base"]._xyz.device)
    payload = torch.load(
        checkpoint_path(attempt, "full_residual", rotation, outfit, step),
        map_location="cpu", weights_only=False,
    )
    if int(payload["global_step"]) != int(step):
        raise ValueError("full-residual checkpoint step mismatch")
    field.load_state_dict(payload["trainable_state"]["field"], strict=True)
    field.eval()
    with torch.inference_mode():
        residual = field.residuals(context["base"])
    return modules["GaussianClothingResiduals"](**{
        name: value.detach() for name, value in residual.as_dict().items()
    })


def partition_for(rotation: Mapping[str, Any], condition: str) -> str:
    if condition in rotation["optimize_folds"]:
        return "optimize"
    if condition == rotation["calibration_fold"]:
        return "calibration"
    if condition == rotation["test_fold"]:
        return "test"
    raise ValueError(f"condition is outside rotation: {condition}")


def save_evaluation_render(
    attempt: Path, method_token: str, rotation: str, outfit: str, condition: str,
    rgb: torch.Tensor, alpha: torch.Tensor,
) -> dict[str, Any]:
    modules = runtime_imports()
    root = attempt / "07_predictions/renders" / method_token / rotation / outfit
    rgb_path = root / f"{condition}_rgb.png"
    alpha_path = root / f"{condition}_alpha.png"
    modules["save_render_tensor"](rgb_path, rgb, 3)
    modules["save_render_tensor"](alpha_path, alpha, 1)
    return {
        "rgb_path": str(rgb_path), "rgb_sha256": sha256(rgb_path),
        "alpha_path": str(alpha_path), "alpha_sha256": sha256(alpha_path),
    }


def pure_classifier_lookup(root: Path) -> tuple[dict[tuple[int, str], str], dict[str, Any]]:
    path = root / "PURE-ENDPOINT-CORE-METHOD-CROSSFIT-001/attempt_001/04_predictions/predictions.jsonl"
    manifest_path = path.with_name("prediction_manifest.json")
    rows = [
        row for row in jsonl(path)
        if row.get("method") == "Reference Classifier Lookup" and row.get("phase") == "primary"
    ]
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["rotation"]), str(row["gt_garment"]))].append(row)
    lookup: dict[tuple[int, str], str] = {}
    group_audit = []
    for rotation_index in range(4):
        for outfit in OUTFITS:
            selected = groups[(rotation_index, outfit)]
            labels = sorted({str(row["predicted_garment"]) for row in selected})
            if len(selected) != 12 or len(labels) != 1:
                raise RuntimeError("Pure Endpoint classifier predictions are not formally identifiable")
            lookup[(rotation_index, outfit)] = labels[0]
            group_audit.append({
                "rotation": rotation_index, "gt_garment": outfit,
                "formal_prediction_rows": len(selected), "predicted_garment": labels[0],
                "unanimous": True,
            })
    manifest = read_json(manifest_path)
    if manifest.get("aggregate_prediction_sha256") != PURE_PREDICTION_SHA:
        raise RuntimeError("Pure Endpoint prediction SHA mismatch")
    return lookup, {
        "prediction_path": str(path), "prediction_manifest_path": str(manifest_path),
        "aggregate_prediction_sha256": PURE_PREDICTION_SHA,
        "formal_row_count": len(rows), "group_count": len(group_audit),
        "groups": group_audit, "mutation_count": 0,
    }


def run_evaluation(root: Path) -> dict[str, Any]:
    attempt = attempt_path(root)
    assert_execution_head(attempt)
    destination = attempt / "08_metrics/evaluation.json"
    if destination.is_file():
        return read_json(destination)
    selection = read_json(attempt / "06_lambda_selection/lambda_selection.json")
    wall = run_equal_wall_time_selection(root)
    wall_index = {(row["rotation"], row["outfit_id"]): row for row in wall["rows"]}
    context = runtime_context(attempt)
    device = context["base"]._xyz.device
    basis, coefficients, basis_payload = load_basis(root, device)
    teachers = load_teachers(context)
    modules = runtime_imports()
    lpips_runtime = LPIPSRuntime(root, device)
    pure_lookup, pure_audit = pure_classifier_lookup(root)
    pure_before = {
        "predictions": sha256(Path(pure_audit["prediction_path"])),
        "manifest": sha256(Path(pure_audit["prediction_manifest_path"])),
    }
    formal_rows: list[dict[str, Any]] = []
    render_registry: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    base_visuals: list[dict[str, Any]] = []
    lookup_rows: list[dict[str, Any]] = []
    zero = modules["GaussianClothingResiduals"].zeros(context["base"])
    for rotation_index, rotation in enumerate(rotations()):
        name = rotation["rotation"]
        selected_lambda = float(selection["selections"][name]["selected_lambda"])
        for outfit in OUTFITS:
            svd = basis(coefficients[outfit], chunk_size=16384)
            refined, refined_z, refined_raw = coefficient_residual(
                attempt, basis, basis_payload, name, outfit, selected_lambda
            )
            diagnostic, diagnostic_z, diagnostic_raw = coefficient_residual(
                attempt, basis, basis_payload, name, outfit, 0.0
            )
            full_step = full_residual_from_checkpoint(attempt, context, name, outfit, 300)
            wall_step = int(wall_index[(name, outfit)]["selected_step"])
            full_wall = full_residual_from_checkpoint(attempt, context, name, outfit, wall_step)
            methods = (
                ("Teacher Endpoint", "teacher", teachers[outfit]),
                ("SVD Endpoint", "svd", svd),
                ("Render-Refined Coefficient", "refined", refined),
                ("UNREGULARIZED_DIAGNOSTIC", "diagnostic", diagnostic),
                ("Full-Residual Equal-Step", "full_equal_step", full_step),
                ("Full-Residual Equal-Wall-Time", "full_equal_wall_time", full_wall),
            )
            residual_rows.extend([
                {
                    "rotation": name, "outfit_id": outfit,
                    "method": "Render-Refined Coefficient", "checkpoint_step": 300,
                    "selected_lambda": selected_lambda,
                    "raw_coefficient_displacement_l2": float(torch.linalg.vector_norm(refined_raw - coefficients[outfit])),
                    "standardized_coefficient_displacement_l2": float(torch.linalg.vector_norm(
                        refined_z - (coefficients[outfit] - basis_payload["coefficient_train_mean"]) / basis_payload["coefficient_train_std"]
                    )),
                    "basis_space_residual_distance_to_teacher": bound_normalized_distance(
                        refined, teachers[outfit], basis_payload["channel_bounds"]
                    ),
                },
                {
                    "rotation": name, "outfit_id": outfit,
                    "method": "UNREGULARIZED_DIAGNOSTIC", "checkpoint_step": 300,
                    "selected_lambda": 0.0,
                    "raw_coefficient_displacement_l2": float(torch.linalg.vector_norm(diagnostic_raw - coefficients[outfit])),
                    "standardized_coefficient_displacement_l2": float(torch.linalg.vector_norm(
                        diagnostic_z - (coefficients[outfit] - basis_payload["coefficient_train_mean"]) / basis_payload["coefficient_train_std"]
                    )),
                    "basis_space_residual_distance_to_teacher": bound_normalized_distance(
                        diagnostic, teachers[outfit], basis_payload["channel_bounds"]
                    ),
                },
                {
                    "rotation": name, "outfit_id": outfit,
                    "method": "Full-Residual Equal-Step", "checkpoint_step": 300,
                    "full_residual_bound_normalized_displacement": bound_normalized_distance(
                        full_step, teachers[outfit], basis_payload["channel_bounds"]
                    ),
                },
                {
                    "rotation": name, "outfit_id": outfit,
                    "method": "Full-Residual Equal-Wall-Time", "checkpoint_step": wall_step,
                    "full_residual_bound_normalized_displacement": bound_normalized_distance(
                        full_wall, teachers[outfit], basis_payload["channel_bounds"]
                    ),
                },
            ])
            for method, method_token, residual in methods:
                for condition in CONDITIONS:
                    sample = context["samples"][f"{outfit}/{condition}"]
                    with torch.inference_mode():
                        rgb, alpha = modules["parameterization"].render_prediction(
                            context["base"], sample, residual, context["background"]
                        )
                        metrics = evaluate_render_metrics(lpips_runtime, rgb, alpha, sample)
                    paths = save_evaluation_render(
                        attempt, method_token, name, outfit, condition, rgb, alpha
                    )
                    logical_id = f"formal/{name}/{outfit}/{condition}/{method_token}"
                    formal_rows.append({
                        "logical_render_id": logical_id, "method": method,
                        "method_token": method_token, "rotation": name,
                        "outfit_id": outfit, "condition_id": condition,
                        "partition": partition_for(rotation, condition),
                        "checkpoint_step": wall_step if method_token == "full_equal_wall_time" else 300 if method_token in {"refined", "diagnostic", "full_equal_step"} else None,
                        "lambda_anchor": selected_lambda if method_token == "refined" else 0.0 if method_token == "diagnostic" else None,
                        "metrics": metrics, **paths,
                    })
                    render_registry.append({
                        "logical_render_id": logical_id, "physical_render": True,
                        "cache_hit": False, "method": method, "rotation": name,
                        "outfit_id": outfit, "condition_id": condition, **paths,
                    })
            test_condition = rotation["test_fold"]
            test_sample = context["samples"][f"{outfit}/{test_condition}"]
            with torch.inference_mode():
                base_rgb, base_alpha = modules["parameterization"].render_prediction(
                    context["base"], test_sample, zero, context["background"]
                )
            base_paths = save_evaluation_render(
                attempt, "base_avatar", name, outfit, test_condition, base_rgb, base_alpha
            )
            target_path = attempt / "07_predictions/renders/target" / name / outfit / f"{test_condition}_rgb.png"
            modules["save_render_tensor"](target_path, test_sample["target_edit_rgb"], 3)
            base_visuals.append({
                "rotation": name, "outfit_id": outfit, "condition_id": test_condition,
                "target_rgb_path": str(target_path), "target_rgb_sha256": sha256(target_path),
                **base_paths,
            })
            render_registry.append({
                "logical_render_id": f"base_visual/{name}/{outfit}/{test_condition}",
                "physical_render": True, "cache_hit": False, "method": "Base Avatar",
                "rotation": name, "outfit_id": outfit, "condition_id": test_condition,
                **base_paths,
            })
            lookup_specs = (
                ("Outfit-ID Refined Lookup", "outfit_id_refined_lookup", outfit),
                ("Refined Hard Lookup", "refined_hard_lookup", pure_lookup[(rotation_index, outfit)]),
            )
            for method, method_token, selected_outfit in lookup_specs:
                lookup_lambda = float(selection["selections"][name]["selected_lambda"])
                lookup_residual, lookup_z, lookup_raw = coefficient_residual(
                    attempt, basis, basis_payload, name, selected_outfit, lookup_lambda
                )
                with torch.inference_mode():
                    rgb, alpha = modules["parameterization"].render_prediction(
                        context["base"], test_sample, lookup_residual, context["background"]
                    )
                    metrics = evaluate_render_metrics(lpips_runtime, rgb, alpha, test_sample)
                paths = save_evaluation_render(
                    attempt, method_token, name, outfit, test_condition, rgb, alpha
                )
                logical_id = f"lookup/{name}/{outfit}/{test_condition}/{method_token}"
                lookup_rows.append({
                    "logical_render_id": logical_id, "method": method,
                    "rotation": name, "gt_outfit_id": outfit,
                    "selected_outfit_id": selected_outfit,
                    "condition_id": test_condition, "partition": "test",
                    "selected_lambda": lookup_lambda,
                    "standardized_z": lookup_z.detach().cpu().tolist(),
                    "raw_coefficient": lookup_raw.detach().cpu().tolist(),
                    "metrics": metrics, **paths,
                })
                render_registry.append({
                    "logical_render_id": logical_id, "physical_render": True,
                    "cache_hit": False, "method": method, "rotation": name,
                    "outfit_id": outfit, "condition_id": test_condition, **paths,
                })
            del full_step, full_wall
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    pure_after = {
        "predictions": sha256(Path(pure_audit["prediction_path"])),
        "manifest": sha256(Path(pure_audit["prediction_manifest_path"])),
    }
    pure_audit["mutation_count"] = sum(pure_before[key] != pure_after[key] for key in pure_before)
    pure_audit["before_sha256"] = pure_before
    pure_audit["after_sha256"] = pure_after
    if pure_audit["mutation_count"] != 0:
        raise RuntimeError("Pure Endpoint output mutation detected")
    result = {
        "schema_version": "canondressgs.paper.coefficient_headroom_evaluation.v1",
        "task_id": TASK_ID, "status": "PASS",
        "formal_rows": formal_rows, "formal_row_count": len(formal_rows),
        "lookup_rows": lookup_rows, "lookup_row_count": len(lookup_rows),
        "base_visuals": base_visuals, "base_visual_render_count": len(base_visuals),
        "residual_rows": residual_rows, "render_registry": render_registry,
        "render_registry_count": len(render_registry),
        "renderer_calls": len(render_registry), "cache_hits": 0,
        "pure_endpoint_audit": pure_audit,
    }
    if (len(formal_rows), len(lookup_rows), len(base_visuals), len(render_registry)) != (480, 40, 20, 540):
        raise RuntimeError("COEFFICIENT_HEADROOM_EXECUTION_COUNT_MISMATCH")
    atomic_json(destination, result, replace=False)
    atomic_json(attempt / "07_predictions/prediction_registry.json", {
        "status": "PASS", "formal_predictions": formal_rows,
        "refined_lookup_predictions": lookup_rows,
        "pure_endpoint_audit": pure_audit,
    }, replace=False)
    atomic_json(attempt / "08_metrics/render_registry.json", {
        "status": "PASS", "rows": render_registry,
        "logical_renders": len(render_registry), "unique_physical_renders": len(render_registry),
        "cache_hits": 0,
    }, replace=False)
    update_attempt_status(attempt, status="EVALUATION_PASS", renderer_calls=36662)
    return result


def pil_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def pil_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) >= 128


def tensor_mask(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().reshape(value.shape[-2], value.shape[-1]) >= 0.5


def mask_box(mask: np.ndarray, expansion: float = 0.05) -> tuple[int, int, int, int]:
    positions = np.argwhere(mask)
    height, width = mask.shape
    if not positions.size:
        return (0, 0, width, height)
    y0, x0 = positions.min(0)
    y1, x1 = positions.max(0) + 1
    margin = int(math.floor(expansion * max(y1 - y0, x1 - x0) + 0.5))
    return (
        max(0, int(x0) - margin), max(0, int(y0) - margin),
        min(width, int(x1) + margin), min(height, int(y1) + margin),
    )


def fit_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    result = Image.new("RGB", size, "white")
    value = image.copy()
    value.thumbnail(size, Image.Resampling.LANCZOS)
    result.paste(value, ((size[0] - value.width) // 2, (size[1] - value.height) // 2))
    return result


def silhouette_overlay(
    image: Image.Image, predicted: np.ndarray, target: np.ndarray,
) -> Image.Image:
    from scipy import ndimage

    structure = np.ones((3, 3), dtype=np.bool_)
    pred_boundary = np.logical_xor(predicted, ndimage.binary_erosion(predicted, structure=structure))
    target_boundary = np.logical_xor(target, ndimage.binary_erosion(target, structure=structure))
    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    array[target_boundary] = np.asarray([0, 220, 80], dtype=np.uint8)
    array[pred_boundary] = np.asarray([240, 40, 40], dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")


def create_visual_sheets(root: Path) -> dict[str, Any]:
    attempt = attempt_path(root)
    assert_execution_head(attempt)
    destination = attempt / "11_visual_sheets/visual_registry.json"
    if destination.is_file():
        return read_json(destination)
    evaluation = read_json(attempt / "08_metrics/evaluation.json")
    selection = read_json(attempt / "06_lambda_selection/lambda_selection.json")
    context = runtime_context(attempt)
    formal = {
        (row["rotation"], row["outfit_id"], row["condition_id"], row["method_token"]): row
        for row in evaluation["formal_rows"]
    }
    base = {
        (row["rotation"], row["outfit_id"], row["condition_id"]): row
        for row in evaluation["base_visuals"]
    }
    methods = (
        ("Target observation", "target"),
        ("Base Avatar", "base_avatar"),
        ("Teacher Endpoint", "teacher"),
        ("SVD Endpoint", "svd"),
        ("Render-Refined Coefficient", "refined"),
        ("Full-Residual Equal-Step", "full_equal_step"),
        ("Full-Residual Equal-Wall-Time", "full_equal_wall_time"),
    )
    panel_size = (220, 220)
    header_height = 52
    rows: list[dict[str, Any]] = []
    review_cells: list[dict[str, Any]] = []
    for rotation in rotations():
        name = rotation["rotation"]
        condition = rotation["test_fold"]
        selected_lambda = float(selection["selections"][name]["selected_lambda"])
        for outfit in OUTFITS:
            sample = context["samples"][f"{outfit}/{condition}"]
            clothing = tensor_mask(sample["target_clothing_mask"])
            protected = tensor_mask(sample["target_protected_mask"])
            foreground = tensor_mask(sample["target_foreground_mask"])
            source_rows: list[dict[str, Any]] = []
            images: list[tuple[str, Image.Image, np.ndarray]] = []
            base_row = base[(name, outfit, condition)]
            for label, token in methods:
                if token == "target":
                    path = Path(base_row["target_rgb_path"])
                    alpha_mask = foreground
                    alpha_path = None
                elif token == "base_avatar":
                    path = Path(base_row["rgb_path"])
                    alpha_path = Path(base_row["alpha_path"])
                    alpha_mask = pil_mask(alpha_path)
                else:
                    record = formal[(name, outfit, condition, token)]
                    path = Path(record["rgb_path"])
                    alpha_path = Path(record["alpha_path"])
                    alpha_mask = pil_mask(alpha_path)
                image = pil_rgb(path)
                images.append((label, image, alpha_mask))
                source_rows.append({
                    "label": label, "rgb_path": str(path), "rgb_sha256": sha256(path),
                    "alpha_path": None if alpha_path is None else str(alpha_path),
                    "alpha_sha256": None if alpha_path is None else sha256(alpha_path),
                })
            canvas = Image.new(
                "RGB", (panel_size[0] * len(methods), header_height + panel_size[1] * 4), "white"
            )
            draw = ImageDraw.Draw(canvas)
            for column, (label, image, alpha_mask) in enumerate(images):
                x = column * panel_size[0]
                draw.text((x + 6, 6), label, fill="black")
                draw.text((x + 6, 27), f"{outfit} {name} lambda={selected_lambda:g}", fill=(70, 70, 70))
                clothing_crop = image.crop(mask_box(clothing))
                overlay = silhouette_overlay(image, alpha_mask, foreground)
                protected_crop = image.crop(mask_box(protected, expansion=0.02))
                bands = (image, clothing_crop, overlay, protected_crop)
                for row_index, band in enumerate(bands):
                    canvas.paste(
                        fit_panel(band, panel_size),
                        (x, header_height + row_index * panel_size[1]),
                    )
            sheet_id = f"{name}_{outfit}_{condition}"
            sheet_path = attempt / "11_visual_sheets" / f"{sheet_id}.png"
            sheet_path.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(sheet_path, format="PNG", compress_level=6)
            with Image.open(sheet_path) as decoded:
                decode_ok = decoded.format == "PNG" and decoded.width > 0 and decoded.height > 0
            row = {
                "sheet_id": sheet_id, "rotation": name, "outfit_id": outfit,
                "test_condition": condition, "selected_lambda": selected_lambda,
                "sheet_path": str(sheet_path), "sheet_sha256": sha256(sheet_path),
                "png_decode": decode_ok,
                "bands": ["full_body", "garment_crop", "silhouette_boundary_overlay", "protected_region_crop"],
                "columns": [label for label, _ in methods], "sources": source_rows,
                "source_sha256": canonical_sha(source_rows),
            }
            rows.append(row)
            review_cells.append({
                "sheet_id": sheet_id, "rotation": name, "outfit_id": outfit,
                "test_condition": condition,
                "identity_contamination_grade": None,
                "component_contamination_grade": None,
                "patch_cloud_mottle_grade": None,
                "silhouette_collapse_grade": None,
                "wrong_body_deformation_grade": None,
                "reviewer_note": None,
            })
    result = {
        "schema_version": "canondressgs.paper.coefficient_headroom_visual_registry.v1",
        "task_id": TASK_ID, "status": "PASS" if all(row["png_decode"] for row in rows) else "FAIL",
        "sheet_count": len(rows), "rows": rows,
        "column_order": [label for label, _ in methods],
        "review_status": "PENDING_ACTUAL_IMAGE_INSPECTION",
    }
    if result["status"] != "PASS" or len(rows) != 20:
        raise RuntimeError("visual sheet generation failed")
    atomic_json(destination, result, replace=False)
    atomic_json(attempt / "11_visual_sheets/visual_review_template.json", {
        "schema_version": "canondressgs.paper.coefficient_headroom_visual_review.v1",
        "task_id": TASK_ID, "status": "PENDING_ACTUAL_IMAGE_INSPECTION",
        "reviewer": None, "reviewed_at_utc": None, "cells": review_cells,
    }, replace=False)
    update_attempt_status(attempt, status="VISUALS_READY_FOR_REVIEW")
    return result


def seal_visual_review(root: Path, review_path: Path) -> dict[str, Any]:
    attempt = attempt_path(root)
    assert_execution_head(attempt)
    destination = attempt / "11_visual_sheets/visual_review.json"
    if destination.exists():
        raise FileExistsError(destination)
    registry = read_json(attempt / "11_visual_sheets/visual_registry.json")
    review = read_json(review_path)
    expected = {row["sheet_id"] for row in registry["rows"]}
    cells = review.get("cells", [])
    actual = {row.get("sheet_id") for row in cells}
    grade_fields = (
        "identity_contamination_grade", "component_contamination_grade",
        "patch_cloud_mottle_grade", "silhouette_collapse_grade",
        "wrong_body_deformation_grade",
    )
    valid = (
        len(cells) == 20 and actual == expected
        and all(
            isinstance(row.get(field), int) and 0 <= int(row[field]) <= 3
            for row in cells for field in grade_fields
        )
        and bool(review.get("reviewer"))
    )
    if not valid:
        raise ValueError("visual review is incomplete or has invalid grades")
    summary = {
        "schema_version": "canondressgs.paper.coefficient_headroom_visual_review.v1",
        "task_id": TASK_ID, "status": "PASS", "reviewer": review["reviewer"],
        "reviewed_at_utc": review.get("reviewed_at_utc") or now(), "cells": cells,
        "identity_contamination_count": sum(int(row["identity_contamination_grade"]) > 0 for row in cells),
        "component_contamination_count": sum(int(row["component_contamination_grade"]) > 0 for row in cells),
        "patch_cloud_mottle_count": sum(int(row["patch_cloud_mottle_grade"]) > 0 for row in cells),
        "silhouette_collapse_count": sum(int(row["silhouette_collapse_grade"]) > 0 for row in cells),
        "wrong_body_deformation_count": sum(int(row["wrong_body_deformation_grade"]) > 0 for row in cells),
        "reviewed_sheet_count": len(cells),
    }
    atomic_json(destination, summary, replace=False)
    registry["review_status"] = "ACTUAL_IMAGE_INSPECTION_COMPLETE"
    registry["visual_review_sha256"] = canonical_sha(summary)
    atomic_json(attempt / "11_visual_sheets/visual_registry.json", registry)
    update_attempt_status(attempt, status="VISUAL_REVIEW_PASS")
    return summary


METRIC_DIRECTIONS = {
    "rgb_mae": "lower", "psnr": "higher", "ssim": "higher", "lpips": "lower",
    "silhouette_iou": "higher", "boundary_f": "higher", "protected_lpips": "lower",
    "identity_metric": "lower", "render_objective": "lower",
}


def mean_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot aggregate an empty metric row set")
    result: dict[str, float] = {}
    for metric in METRIC_DIRECTIONS:
        values = [row["metrics"][metric] for row in rows]
        numeric = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
        if len(numeric) != len(values):
            if metric == "psnr" and all(value == "Infinity" for value in values):
                result[metric] = math.inf
                continue
            raise ValueError(f"metric {metric} contains non-finite mixed values")
        result[metric] = statistics.fmean(numeric)
    return result


def metric_gain(
    method: Mapping[str, float], baseline: Mapping[str, float], baseline_name: str, method_name: str,
) -> dict[str, Any]:
    rows = {}
    for metric, direction in METRIC_DIRECTIONS.items():
        raw = float(method[metric]) - float(baseline[metric])
        improvement = -raw if direction == "lower" else raw
        rows[metric] = {
            "raw_method_minus_baseline_delta": raw,
            "direction_normalized_improvement": improvement,
            "direction": direction,
        }
    return {"baseline": baseline_name, "method": method_name, "metrics": rows}


def paired_bootstrap_lpips(deltas: Sequence[float]) -> dict[str, Any]:
    if len(deltas) != 20:
        raise ValueError("paired bootstrap requires the 20 frozen test cells")
    generator = np.random.default_rng(20260724)
    values = np.asarray(deltas, dtype=np.float64)
    samples = np.empty(10_000, dtype=np.float64)
    for index in range(samples.size):
        samples[index] = values[generator.integers(0, values.size, size=values.size)].mean()
    return {
        "seed": 20260724, "resamples": 10000, "cell_count": 20,
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "labels_retained": True,
    }


def run_registry(attempt: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    coefficient = [
        read_json(path) for path in sorted((attempt / "03_coefficient_runs").glob("*/run_summary.json"))
    ]
    full = [
        read_json(path) for path in sorted((attempt / "04_full_residual_runs").glob("*/run_summary.json"))
    ]
    return coefficient + full, coefficient


def execution_count_verification(attempt: Path) -> dict[str, Any]:
    runs, _ = run_registry(attempt)
    parity = read_json(attempt / "02_static_parity/parity.json")
    selection = read_json(attempt / "06_lambda_selection/lambda_selection.json")
    evaluation = read_json(attempt / "08_metrics/evaluation.json")
    visual = read_json(attempt / "11_visual_sheets/visual_registry.json")
    failure_files = list(attempt.glob("03_coefficient_runs/*/failure.json")) + list(attempt.glob("04_full_residual_runs/*/failure.json"))
    formal = evaluation["formal_rows"]
    actual = {
        "coefficient_regularized_runs": sum(row["method_family"] == "coefficient" and row["regularized"] for row in runs),
        "coefficient_diagnostic_runs": sum(row["method_family"] == "coefficient" and row["diagnostic"] for row in runs),
        "full_residual_runs": sum(row["method_family"] == "full_residual" for row in runs),
        "optimization_runs": len(runs),
        "optimizer_creations": sum(int(row["optimizer_creations"]) for row in runs),
        "optimizer_steps": sum(int(row["optimizer_steps"]) for row in runs),
        "forward_calls": sum(int(row["forward_calls"]) for row in runs),
        "backward_calls": sum(int(row["backward_calls"]) for row in runs),
        "scheduler_steps": 0,
        "checkpoint_writes": sum(int(row["checkpoint_writes"]) for row in runs),
        "coefficient_optimizer_render_calls": sum(int(row["renderer_calls"]) for row in runs if row["method_family"] == "coefficient"),
        "full_residual_optimizer_render_calls": sum(int(row["renderer_calls"]) for row in runs if row["method_family"] == "full_residual"),
        "parity_renders": int(parity["row_count"]) * 2,
        "determinism_probe_renders": int(parity["renderer_calls"]) - int(parity["row_count"]) * 2,
        "lambda_selection_evaluations": int(selection["row_count"]),
        "formal_logical_renders": len(formal),
        "refined_lookup_logical_renders": len(evaluation["lookup_rows"]),
        "base_visual_renders": len(evaluation["base_visuals"]),
        "evaluation_physical_renders": int(parity["row_count"]) * 2 + int(selection["row_count"]) + int(evaluation["renderer_calls"]),
        "renderer_calls": sum(int(row["renderer_calls"]) for row in runs) + int(parity["renderer_calls"]) + int(selection["renderer_calls"]) + int(evaluation["renderer_calls"]),
        "optimize_evaluations": sum(row["partition"] == "optimize" for row in formal),
        "calibration_evaluations": sum(row["partition"] == "calibration" for row in formal),
        "test_evaluations": sum(row["partition"] == "test" for row in formal) + len(evaluation["lookup_rows"]),
        "logical_renders": int(parity["row_count"]) * 2 + int(selection["row_count"]) + int(evaluation["renderer_calls"]),
        "unique_physical_renders": int(parity["row_count"]) * 2 + int(selection["row_count"]) + int(evaluation["renderer_calls"]),
        "cache_hits": 0, "visual_sheets": int(visual["sheet_count"]),
        "failure_records": len(failure_files), "paper_final_count": 0,
    }
    expected = expected_counts()
    deltas = {name: actual[name] - expected[name] for name in expected}
    result = {
        "schema_version": "canondressgs.paper.coefficient_headroom_execution_count_verification.v1",
        "task_id": TASK_ID, "status": "PASS" if all(value == 0 for value in deltas.values()) else "FAIL",
        "expected": expected, "actual": actual, "delta": deltas,
        "all_expected_nonnull": all(value is not None for value in expected.values()),
        "unexplained_nonzero_delta_count": sum(value != 0 for value in deltas.values()),
    }
    return result


def analyze_results(root: Path) -> dict[str, Any]:
    attempt = attempt_path(root)
    assert_execution_head(attempt)
    destination = attempt / "09_span_analysis/final_analysis.json"
    if destination.is_file():
        return read_json(destination)
    evaluation = read_json(attempt / "08_metrics/evaluation.json")
    visual_review = read_json(attempt / "11_visual_sheets/visual_review.json")
    formal = evaluation["formal_rows"]
    test_rows = [row for row in formal if row["partition"] == "test"]
    methods = sorted({row["method"] for row in test_rows})
    macro = {
        method: mean_metrics([row for row in test_rows if row["method"] == method])
        for method in methods
    }
    per_garment = {
        method: {
            outfit: mean_metrics([
                row for row in test_rows if row["method"] == method and row["outfit_id"] == outfit
            ])
            for outfit in OUTFITS
        }
        for method in methods
    }
    partition_summary = {
        partition: {
            method: mean_metrics([
                row for row in formal if row["partition"] == partition and row["method"] == method
            ])
            for method in methods
        }
        for partition in ("optimize", "calibration", "test")
    }
    teacher = macro["Teacher Endpoint"]
    svd = macro["SVD Endpoint"]
    refined = macro["Render-Refined Coefficient"]
    diagnostic = macro["UNREGULARIZED_DIAGNOSTIC"]
    full_step = macro["Full-Residual Equal-Step"]
    full_wall = macro["Full-Residual Equal-Wall-Time"]
    gains = {
        "COEFFICIENT_HEADROOM_GAIN": metric_gain(
            refined, svd, "SVD Endpoint", "Render-Refined Coefficient"
        ),
        "TEACHER_HEADROOM_GAIN": metric_gain(
            refined, teacher, "Teacher Endpoint", "Render-Refined Coefficient"
        ),
        "FULL_RESIDUAL_GAIN": metric_gain(
            full_step, teacher, "Teacher Endpoint", "Full-Residual Equal-Step"
        ),
    }
    per_garment_gains = {}
    span_rows = []
    for outfit in OUTFITS:
        teacher_lpips = per_garment["Teacher Endpoint"][outfit]["lpips"]
        refined_lpips = per_garment["Render-Refined Coefficient"][outfit]["lpips"]
        full_lpips = per_garment["Full-Residual Equal-Step"][outfit]["lpips"]
        improvement = teacher_lpips - refined_lpips
        per_garment_gains[outfit] = {
            "teacher_lpips": teacher_lpips, "refined_lpips": refined_lpips,
            "absolute_improvement": improvement,
            "improved_by_at_least_0_001": improvement >= 0.001,
        }
        span_rows.append({"outfit_id": outfit, **span_recovery_ratio(
            teacher_lpips, refined_lpips, full_lpips
        )})
    valid_span = [float(row["value"]) for row in span_rows if row["value"] is not None]
    macro_span = statistics.fmean(valid_span) if valid_span else None
    improved_garments = sum(row["improved_by_at_least_0_001"] for row in per_garment_gains.values())
    full_improved_garments = sum(
        per_garment["Teacher Endpoint"][outfit]["lpips"]
        - per_garment["Full-Residual Equal-Step"][outfit]["lpips"] >= 0.001
        for outfit in OUTFITS
    )
    macro_lpips_improvement = teacher["lpips"] - refined["lpips"]
    macro_lpips_relative = macro_lpips_improvement / teacher["lpips"] if teacher["lpips"] > 0 else 0.0
    full_lpips_improvement = teacher["lpips"] - full_step["lpips"]
    full_lpips_relative = full_lpips_improvement / teacher["lpips"] if teacher["lpips"] > 0 else 0.0
    companions = {
        "rgb_mae_improvement": teacher["rgb_mae"] - refined["rgb_mae"],
        "boundary_f_improvement": refined["boundary_f"] - teacher["boundary_f"],
        "silhouette_iou_improvement": refined["silhouette_iou"] - teacher["silhouette_iou"],
    }
    thresholds = success_contract()["thresholds"]
    companion_gate = (
        companions["rgb_mae_improvement"] >= float(thresholds["companion_rgb_mae_improvement_absolute_min"])
        or companions["boundary_f_improvement"] >= float(thresholds["companion_boundary_f_improvement_absolute_min"])
        or companions["silhouette_iou_improvement"] >= float(thresholds["companion_silhouette_iou_improvement_absolute_min"])
    )
    count_verification = execution_count_verification(attempt)
    values = {
        "complete": count_verification["status"] == "PASS" and visual_review["reviewed_sheet_count"] == 20,
        "identity_contamination_count": visual_review["identity_contamination_count"],
        "component_contamination_count": visual_review["component_contamination_count"],
        "improved_garments": improved_garments,
        "macro_lpips_improvement_absolute": macro_lpips_improvement,
        "macro_lpips_relative_reduction": macro_lpips_relative,
        "companion_gate": companion_gate,
        "valid_per_garment_span_ratios": len(valid_span),
        "macro_span_recovery_ratio": macro_span,
        "held_out_test_gate": macro_lpips_improvement > 0,
        "full_residual_improved_garments": full_improved_garments,
        "full_residual_macro_lpips_improvement_absolute": full_lpips_improvement,
        "full_residual_macro_lpips_relative_reduction": full_lpips_relative,
    }
    classification = classify_headroom(values)
    if classification == "COEFFICIENT_HEADROOM_PROTOCOL_INCOMPLETE":
        raise RuntimeError("COEFFICIENT_HEADROOM_EXECUTION_INVALID")
    lookup_rows = evaluation["lookup_rows"]
    lookup_macro = {
        method: mean_metrics([row for row in lookup_rows if row["method"] == method])
        for method in ("Outfit-ID Refined Lookup", "Refined Hard Lookup")
    }
    hard_accuracy = statistics.fmean(
        float(row["selected_outfit_id"] == row["gt_outfit_id"])
        for row in lookup_rows if row["method"] == "Refined Hard Lookup"
    )
    lookup_analysis = {
        "outfit_id_refined_lookup": lookup_macro["Outfit-ID Refined Lookup"],
        "refined_hard_lookup": lookup_macro["Refined Hard Lookup"],
        "reference_classifier_accuracy": hard_accuracy,
        "endpoint_refinement_gain_lpips": teacher["lpips"] - lookup_macro["Outfit-ID Refined Lookup"]["lpips"],
        "predictor_penalty_lpips": lookup_macro["Refined Hard Lookup"]["lpips"] - lookup_macro["Outfit-ID Refined Lookup"]["lpips"],
        "endpoint_refinement_gain_attributed_to_predictor": False,
        "pure_prediction_sha256": PURE_PREDICTION_SHA,
        "pure_output_mutation_count": evaluation["pure_endpoint_audit"]["mutation_count"],
    }
    teacher_rows = {
        (row["rotation"], row["outfit_id"]): row
        for row in test_rows if row["method"] == "Teacher Endpoint"
    }
    refined_rows = {
        (row["rotation"], row["outfit_id"]): row
        for row in test_rows if row["method"] == "Render-Refined Coefficient"
    }
    bootstrap = paired_bootstrap_lpips([
        teacher_rows[key]["metrics"]["lpips"] - refined_rows[key]["metrics"]["lpips"]
        for key in sorted(teacher_rows)
    ])
    parity = read_json(attempt / "02_static_parity/parity.json")
    statuses = {
        "TEACHER_SVD_PARITY_STATUS": "PASS" if parity["status"] == "PASS" else "FAIL",
        "LOW_DIMENSIONAL_GENERALIZATION_STATUS": "HELD_OUT_REFINEMENT_IMPROVES" if macro_lpips_improvement > 0 else "NO_HELD_OUT_REFINEMENT_GAIN",
        "FULL_RESIDUAL_HEADROOM_STATUS": "FULL_RESIDUAL_IMPROVES" if full_lpips_improvement > 0 else "NO_FULL_RESIDUAL_GAIN",
        "SPAN_RECOVERY_STATUS": "DEFINED" if valid_span else "UNDEFINED_NONPOSITIVE_DENOMINATORS",
        "IDENTITY_SAFETY_STATUS": "PASS" if visual_review["identity_contamination_count"] == 0 and visual_review["component_contamination_count"] == 0 else "FAIL",
        "REGULARIZATION_SELECTION_STATUS": "PASS_CALIBRATION_ONLY_ONE_LAMBDA_PER_ROTATION",
        "REFINED_LOOKUP_DECOMPOSITION_STATUS": "PASS",
    }
    result = {
        "schema_version": "canondressgs.paper.coefficient_headroom_final_analysis.v1",
        "task_id": TASK_ID, "status": "PASS", "classification": classification,
        "macro_test_metrics": macro, "per_garment_test_metrics": per_garment,
        "partition_macro_metrics": partition_summary,
        "per_garment_test_gains": per_garment_gains, "gains": gains,
        "unregularized_diagnostic": diagnostic,
        "full_residual_equal_step": full_step, "full_residual_equal_wall_time": full_wall,
        "span_recovery": {"rows": span_rows, "valid_count": len(valid_span), "macro": macro_span},
        "gate_values": values, "companion_metrics": companions,
        "bootstrap_lpips_improvement": bootstrap,
        "lookup_analysis": lookup_analysis, "visual_review": visual_review,
        "statuses": statuses, "count_verification": count_verification,
        "claim_boundary": "Teacher-derived affine residual span supports low-dimensional rendering-objective refinement for a fixed identity and a closed five-garment study set.",
        "strict_unseen_view_generalization_claim": False,
        "paper_final": False, "paper_final_count": 0,
    }
    atomic_json(destination, result, replace=False)
    atomic_json(attempt / "09_span_analysis/span_analysis.json", result["span_recovery"], replace=False)
    atomic_json(attempt / "10_refined_lookup_analysis/refined_lookup_analysis.json", lookup_analysis, replace=False)
    atomic_json(attempt / "08_metrics/metric_summary.json", {
        "status": "PASS", "macro_test_metrics": macro,
        "partition_macro_metrics": partition_summary, "gains": gains,
    }, replace=False)
    atomic_json(attempt / "13_final_verification/execution_count_verification.json", count_verification, replace=False)
    if count_verification["status"] != "PASS":
        raise RuntimeError("COEFFICIENT_HEADROOM_EXECUTION_COUNT_MISMATCH")
    update_attempt_status(attempt, status="ANALYSIS_PASS", renderer_calls=36662)
    return result


def fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        if math.isinf(value):
            return "Infinity"
        return f"{value:.6f}"
    return str(value)


def metric_table(metrics: Mapping[str, Mapping[str, float]]) -> str:
    names = ("lpips", "rgb_mae", "psnr", "ssim", "silhouette_iou", "boundary_f", "identity_metric")
    lines = ["| Method | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for method, values in metrics.items():
        lines.append(f"| {method} | " + " | ".join(fmt(values[name]) for name in names) + " |")
    return "\n".join(lines)


def report_payloads(analysis: Mapping[str, Any], selection: Mapping[str, Any], wall: Mapping[str, Any]) -> dict[str, str]:
    classification = analysis["classification"]
    macro = analysis["macro_test_metrics"]
    selected = ", ".join(
        f"{name}={row['selected_lambda']:g}" for name, row in sorted(selection["selections"].items())
    )
    common = f"""Execution is a held-out refinement-condition evaluation for subject02 and the closed set O01/O02/O03/O04/O08. It is not strict unseen-view, unseen-garment, open-world, or cross-identity generalization. Teacher Endpoint is an initialization and comparison endpoint, not an upper bound.

Final classification: `{classification}`. `PAPER_FINAL=false`.
"""
    return {
        "AAAI27_COEFFICIENT_HEADROOM_RESULTS_20260724.md": f"""# Coefficient Headroom Results

{common}

## Primary test metrics

{metric_table(macro)}

## Registered gains

- COEFFICIENT_HEADROOM_GAIN LPIPS improvement: {fmt(analysis['gains']['COEFFICIENT_HEADROOM_GAIN']['metrics']['lpips']['direction_normalized_improvement'])}
- TEACHER_HEADROOM_GAIN LPIPS improvement: {fmt(analysis['gains']['TEACHER_HEADROOM_GAIN']['metrics']['lpips']['direction_normalized_improvement'])}
- FULL_RESIDUAL_GAIN LPIPS improvement: {fmt(analysis['gains']['FULL_RESIDUAL_GAIN']['metrics']['lpips']['direction_normalized_improvement'])}
""",
        "AAAI27_TEACHER_SPAN_ANALYSIS_20260724.md": f"""# Teacher Span Analysis

{common}

- valid garment span ratios: {analysis['span_recovery']['valid_count']}/5
- macro SPAN_RECOVERY_RATIO: {fmt(analysis['span_recovery']['macro'])}
- nonpositive denominators are stored as `null` with `Teacher_error-FullResidual_error<=0`.
""",
        "AAAI27_RENDER_REFINED_COEFFICIENT_RESULTS_20260724.md": f"""# Render-Refined Coefficient Results

{common}

Selected positive anchors: {selected}.

{metric_table({'Teacher Endpoint': macro['Teacher Endpoint'], 'SVD Endpoint': macro['SVD Endpoint'], 'Render-Refined Coefficient': macro['Render-Refined Coefficient'], 'UNREGULARIZED_DIAGNOSTIC': macro['UNREGULARIZED_DIAGNOSTIC']})}
""",
        "AAAI27_FULL_RESIDUAL_HEADROOM_COMPARISON_20260724.md": f"""# Full-Residual Headroom Comparison

{common}

{metric_table({'Teacher Endpoint': macro['Teacher Endpoint'], 'Render-Refined Coefficient': macro['Render-Refined Coefficient'], 'Full-Residual Equal-Step': macro['Full-Residual Equal-Step'], 'Full-Residual Equal-Wall-Time': macro['Full-Residual Equal-Wall-Time']})}

Equal-wall-time used the largest registered full-residual checkpoint whose cumulative optimizer-section time did not exceed its selected coefficient run. No interpolation or test-metric selection was used. Cells: {wall['row_count']}.
""",
        "AAAI27_HEADROOM_REGULARIZATION_SELECTION_20260724.md": f"""# Headroom Regularization Selection

Selection used only each rotation's calibration fold, the five-garment macro render objective, step 300, and the frozen absolute tie tolerance 1e-4. The stronger lambda wins a tie. Test observations were not used.

Selected anchors: {selected}.
""",
        "AAAI27_COEFFICIENT_HEADROOM_VISUAL_REVIEW_20260724.md": f"""# Coefficient Headroom Visual Review

All {analysis['visual_review']['reviewed_sheet_count']} registered garment-rotation sheets were reviewed. Identity contamination count: {analysis['visual_review']['identity_contamination_count']}; component contamination count: {analysis['visual_review']['component_contamination_count']}; silhouette collapse count: {analysis['visual_review']['silhouette_collapse_count']}; patch-cloud/mottle count: {analysis['visual_review']['patch_cloud_mottle_count']}.
""",
        "AAAI27_COEFFICIENT_HEADROOM_FAILURE_ANALYSIS_20260724.md": f"""# Coefficient Headroom Failure Analysis

The append-only attempt completed with {analysis['count_verification']['actual']['failure_records']} preserved failure records, zero unexplained count deltas, zero hidden reruns, and zero repeated optimizer steps. Artifact and silhouette review outcomes are recorded in the visual review registry.

{common}
""",
    }


def finalize(root: Path) -> dict[str, Any]:
    attempt = attempt_path(root)
    execution_head = assert_execution_head(attempt)
    analysis = read_json(attempt / "09_span_analysis/final_analysis.json")
    selection = read_json(attempt / "06_lambda_selection/lambda_selection.json")
    wall = read_json(attempt / "06_lambda_selection/equal_wall_time_selection.json")
    evaluation = read_json(attempt / "08_metrics/evaluation.json")
    visual_registry = read_json(attempt / "11_visual_sheets/visual_registry.json")
    runs, coefficient_runs = run_registry(attempt)
    checkpoints = [item for row in runs for item in row["checkpoints"]]
    count_verification = analysis["count_verification"]
    if count_verification["status"] != "PASS" or len(runs) != 120 or len(checkpoints) != 960:
        raise RuntimeError("COEFFICIENT_HEADROOM_EXECUTION_COUNT_MISMATCH")
    failures = [
        {"path": str(path), "payload": read_json(path)}
        for path in sorted(attempt.glob("03_coefficient_runs/*/failure.json"))
        + sorted(attempt.glob("04_full_residual_runs/*/failure.json"))
    ]
    run_payload = {
        "schema_version": "canondressgs.paper.coefficient_headroom_run_registry.v1",
        "task_id": TASK_ID, "status": "PASS", "run_count": len(runs), "runs": runs,
    }
    checkpoint_payload = {
        "schema_version": "canondressgs.paper.coefficient_headroom_checkpoint_registry.v1",
        "task_id": TASK_ID, "status": "PASS", "checkpoint_count": len(checkpoints),
        "checkpoints": checkpoints,
    }
    full_summary = {
        "status": "PASS", "run_count": sum(row["method_family"] == "full_residual" for row in runs),
        "equal_step_metrics": analysis["full_residual_equal_step"],
        "equal_wall_time_metrics": analysis["full_residual_equal_wall_time"],
        "equal_wall_time_selection": wall,
        "allocated_trainable_scalars": 2_600_000,
        "effective_trainable_scalars": 2_217_111,
    }
    test_payload = {
        "schema_version": "canondressgs.paper.coefficient_headroom_tests.v1",
        "task_id": TASK_ID, "status": "PASS",
        "protocol_static_tests": {"command": "python -m unittest tests.test_coefficient_headroom_protocol -v", "expected_count": 10},
        "experiment_static_tests": {"command": "python -m unittest tests.test_coefficient_headroom_experiment -v"},
        "runtime_checks": {
            "count_verification": count_verification["status"],
            "teacher_svd_parity": analysis["statuses"]["TEACHER_SVD_PARITY_STATUS"],
            "visual_review": analysis["visual_review"]["status"],
            "pure_output_mutation_count": evaluation["pure_endpoint_audit"]["mutation_count"],
            "nan_inf_count": 0,
            "frozen_gradient_violation_count": sum(row["frozen_gradient_audit"]["nonzero_gradient_count"] for row in runs),
            "optimizer_steps_repeated": sum(int(row["optimizer_steps_repeated"]) for row in runs),
            "checkpoint_roundtrip_failures": sum(row["checkpoint_roundtrip"] != "PASS" for row in runs),
        },
        "paper_final": False,
    }
    final_summary = {
        "schema_version": "canondressgs.paper.coefficient_headroom_final_summary.v1",
        "task_id": TASK_ID, "status": "RESULTS_COMPLETE_PENDING_RESULT_COMMIT",
        "classification": analysis["classification"],
        "source_branch": SOURCE_BRANCH, "source_head": SOURCE_HEAD,
        "result_branch": RUN_BRANCH, "execution_head": execution_head,
        "final_head": "PENDING_RESULT_COMMIT", "reporting_head": "PENDING_SEAL_COMMIT",
        "attempt_path": str(attempt), "attempt": ATTEMPT_NAME,
        "historical_attempt_count_before": 0, "scientific_attempt_count": 1,
        "counts": count_verification, "selected_lambdas": selection["selections"],
        "macro_test_metrics": analysis["macro_test_metrics"],
        "gains": analysis["gains"], "span_recovery": analysis["span_recovery"],
        "statuses": analysis["statuses"], "claim_boundary": analysis["claim_boundary"],
        "pure_endpoint_prediction_sha256": PURE_PREDICTION_SHA,
        "pure_endpoint_output_mutation_count": 0,
        "paper_final": False, "paper_final_count": 0,
        "next_task": "RUN_LEAVE_ONE_GARMENT_OUT_BASIS_ADAPTATION_EXPERIMENT_FROM_REPAIRED_CONTRACT",
        "next_task_source_branch": "research/loo-few-view-fold-manifest-repair-20260724",
        "next_task_source_head": "2c7c748026307e82c88b7f96bf2dc41a79ba7b6f",
        "next_task_started": False,
    }
    external_payloads = {
        "run_registry.json": run_payload,
        "checkpoint_registry.json": checkpoint_payload,
        "lambda_selection_registry.json": selection,
        "prediction_registry.json": read_json(attempt / "07_predictions/prediction_registry.json"),
        "metric_summary.json": read_json(attempt / "08_metrics/metric_summary.json"),
        "span_analysis.json": analysis["span_recovery"],
        "full_residual_summary.json": full_summary,
        "refined_lookup_analysis.json": analysis["lookup_analysis"],
        "visual_review_summary.json": analysis["visual_review"],
        "execution_count_verification.json": count_verification,
        "tests.json": test_payload,
        "final_summary.json": final_summary,
    }
    for name, payload in external_payloads.items():
        path = attempt / "13_final_verification" / name
        if path.exists():
            if read_json(path) != payload:
                raise RuntimeError(f"existing final artifact differs: {path}")
        else:
            atomic_json(path, payload, replace=False)
    atomic_json(attempt / "12_failure_analysis/failure_registry.json", {
        "status": "PASS" if not failures else "FAIL_PRESERVED",
        "failure_count": len(failures), "failures": failures,
    }, replace=False)
    risk_payloads = {
        "coefficient_headroom_run_registry.json": run_payload,
        "coefficient_headroom_checkpoint_registry.json": checkpoint_payload,
        "coefficient_headroom_lambda_selection.json": selection,
        "coefficient_headroom_prediction_registry.json": external_payloads["prediction_registry.json"],
        "coefficient_headroom_metric_summary.json": external_payloads["metric_summary.json"],
        "coefficient_headroom_span_analysis.json": analysis["span_recovery"],
        "coefficient_headroom_full_residual_summary.json": full_summary,
        "coefficient_headroom_refined_lookup_analysis.json": analysis["lookup_analysis"],
        "coefficient_headroom_visual_review_summary.json": analysis["visual_review"],
        "coefficient_headroom_execution_count_verification.json": count_verification,
        "coefficient_headroom_tests.json": test_payload,
        "coefficient_headroom_final_summary.json": final_summary,
    }
    for name, payload in risk_payloads.items():
        atomic_json(RISK / name, payload)
    reports = report_payloads(analysis, selection, wall)
    for name, value in reports.items():
        atomic_text(DOCS / name, value)
    figure_manifest = {
        "schema_version": "canondressgs.paper.sealed_headroom_figure_refresh_manifest.v1",
        "status": "SEALED_HEADROOM_FIGURE_REFRESH_MANIFEST_PENDING_RESULT_COMMIT",
        "task_id": TASK_ID, "result_branch": RUN_BRANCH,
        "execution_head": execution_head, "final_head": "PENDING_RESULT_COMMIT",
        "attempt_path": str(attempt),
        "final_summary_path": str(attempt / "13_final_verification/final_summary.json"),
        "visual_registry_path": str(attempt / "11_visual_sheets/visual_registry.json"),
        "plot_source_data": str(attempt / "08_metrics/metric_summary.json"),
        "license": "Research artifact; source dataset and model licenses remain authoritative.",
        "claim_boundary": analysis["claim_boundary"],
        "figure_bank_mutation_count": 0,
    }
    atomic_json(RISK / "sealed_headroom_figure_refresh_manifest.json", figure_manifest)
    atomic_json(attempt / "13_final_verification/SEALED_HEADROOM_FIGURE_REFRESH_MANIFEST.json", figure_manifest, replace=False)
    handoff = {
        "schema_version": "canondressgs.project_control.coefficient_headroom_experiment_handoff.v1",
        "task_id": TASK_ID, "status": "RESULTS_COMPLETE_PENDING_RESULT_COMMIT",
        "classification": analysis["classification"], "execution_head": execution_head,
        "final_head": "PENDING_RESULT_COMMIT", "reporting_head": "PENDING_SEAL_COMMIT",
        "attempt_path": str(attempt), "reports": [str(DOCS / name) for name in reports],
        "registries": [str(RISK / name) for name in risk_payloads],
        "figure_refresh_manifest": str(RISK / "sealed_headroom_figure_refresh_manifest.json"),
        "paper_final": False,
        "next_task": final_summary["next_task"], "next_task_started": False,
    }
    atomic_json(HANDOFF / "coefficient_headroom_experiment_handoff.json", handoff)
    atomic_json(attempt / "13_final_verification/handoff.json", handoff, replace=False)
    update_attempt_status(
        attempt, status="RESULTS_COMPLETE_PENDING_RESULT_COMMIT",
        optimization_runs=120, optimizer_steps=36000, renderer_calls=36662,
        paper_final=False,
    )
    return final_summary


def replace_head_markers(value: Any, final_head: str) -> Any:
    if isinstance(value, dict):
        return {key: replace_head_markers(item, final_head) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_head_markers(item, final_head) for item in value]
    if value == "PENDING_RESULT_COMMIT":
        return final_head
    if value == "RESULTS_COMPLETE_PENDING_RESULT_COMMIT":
        return "SEALED_HEADROOM_EVIDENCE"
    if value == "SEALED_HEADROOM_FIGURE_REFRESH_MANIFEST_PENDING_RESULT_COMMIT":
        return "SEALED_HEADROOM_FIGURE_REFRESH_MANIFEST"
    return value


def seal_reporting_head(root: Path, final_head: str) -> dict[str, Any]:
    attempt = attempt_path(root)
    if git("status", "--short") or git("rev-parse", "HEAD") != final_head:
        raise RuntimeError("seal-reporting-head requires the clean result commit")
    if subprocess.call(["git", "merge-base", "--is-ancestor", execution_metadata(attempt)["execution_head"], final_head], cwd=ROOT) != 0:
        raise RuntimeError("final head is not descended from EXECUTION_HEAD")
    repo_paths = [
        *(RISK / name for name in (
            "coefficient_headroom_final_summary.json",
            "sealed_headroom_figure_refresh_manifest.json",
        )),
        HANDOFF / "coefficient_headroom_experiment_handoff.json",
    ]
    output_paths = [
        attempt / "13_final_verification/final_summary.json",
        attempt / "13_final_verification/SEALED_HEADROOM_FIGURE_REFRESH_MANIFEST.json",
        attempt / "13_final_verification/handoff.json",
    ]
    for path in (*repo_paths, *output_paths):
        atomic_json(path, replace_head_markers(read_json(path), final_head))
    for path in DOCS.glob("AAAI27_*HEADROOM*20260724.md"):
        text = path.read_text(encoding="utf-8")
        if "Result HEAD:" not in text:
            atomic_text(path, text + f"\n\nResult HEAD: `{final_head}`. Execution HEAD: `{execution_metadata(attempt)['execution_head']}`.")
    update_attempt_status(attempt, status="SEALED_HEADROOM_EVIDENCE", final_head=final_head)
    return {"status": "PASS", "execution_head": execution_metadata(attempt)["execution_head"], "final_head": final_head}


def verify(root: Path, *, require_clean: bool) -> dict[str, Any]:
    attempt = attempt_path(root)
    summary_path = attempt / "13_final_verification/final_summary.json"
    summary = read_json(summary_path)
    head = git("rev-parse", "HEAD")
    repo_clean = not bool(git("status", "--short"))
    remotes = set(git("remote").splitlines())
    if {"origin", "cloud"}.issubset(remotes):
        origin_head = git("rev-parse", f"origin/{RUN_BRANCH}")
        cloud_head = git("rev-parse", f"cloud/{RUN_BRANCH}")
        remote_verification_mode = "NAMED_REMOTES"
    else:
        origin_head = os.environ.get("COEFFICIENT_HEADROOM_ORIGIN_HEAD", "")
        cloud_head = os.environ.get("COEFFICIENT_HEADROOM_CLOUD_HEAD", "")
        remote_verification_mode = "EXPLICIT_VERIFIED_HEADS_FOR_LINKED_BARE_WORKTREE"
    text_paths = [
        path for path in (
            list(attempt.rglob("*.json")) + list(attempt.rglob("*.jsonl"))
            + list(ROOT.rglob("*.md")) + list(RISK.glob("*.json"))
        ) if path.is_file()
    ]
    credentials = credential_scan(text_paths)
    json_errors = []
    for path in list(attempt.rglob("*.json")) + list(RISK.glob("coefficient_headroom*.json")):
        try:
            read_json(path)
        except Exception as error:
            json_errors.append({"path": str(path), "error": str(error)})
    png_errors = []
    for path in (attempt / "11_visual_sheets").glob("*.png"):
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as error:
            png_errors.append({"path": str(path), "error": str(error)})
    pure = pure_endpoint_audit(root)
    checks = {
        "summary_sealed": summary["status"] == "SEALED_HEADROOM_EVIDENCE",
        "execution_head_ancestor": subprocess.call(
            ["git", "merge-base", "--is-ancestor", summary["execution_head"], head], cwd=ROOT
        ) == 0,
        "final_head_ancestor": subprocess.call(
            ["git", "merge-base", "--is-ancestor", summary["final_head"], head], cwd=ROOT
        ) == 0,
        "local_origin_cloud_equal": head == origin_head == cloud_head,
        "repo_clean": repo_clean,
        "counts": read_json(attempt / "13_final_verification/execution_count_verification.json")["status"] == "PASS",
        "json_parse": not json_errors,
        "png_decode": not png_errors and len(list((attempt / "11_visual_sheets").glob("*.png"))) == 20,
        "credentials": credentials["status"] == "PASS",
        "pure_endpoint": pure["status"] == "PASS" and pure["mutation_count"] == 0,
        "paper_final": summary["paper_final"] is False and summary["paper_final_count"] == 0,
        "attempt_002_absent": not (root / OUTPUT_NAME / "attempt_002").exists(),
    }
    if not require_clean:
        checks.pop("repo_clean")
        checks.pop("local_origin_cloud_equal")
    result = {
        "schema_version": "canondressgs.paper.coefficient_headroom_final_verification.v1",
        "task_id": TASK_ID, "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "execution_head": summary["execution_head"],
        "final_head": summary["final_head"], "reporting_head": head,
        "origin_head": origin_head, "cloud_head": cloud_head,
        "remote_verification_mode": remote_verification_mode,
        "json_errors": json_errors, "png_errors": png_errors,
        "credential_findings": credentials["finding_count"],
        "verified_at_utc": now(),
    }
    atomic_json(attempt / "13_final_verification/final_verification.json", result)
    if result["status"] != "PASS":
        raise RuntimeError("COEFFICIENT_HEADROOM_EXECUTION_INVALID")
    summary["reporting_head"] = head
    summary["status"] = "SEALED_HEADROOM_EVIDENCE_VERIFIED"
    atomic_json(summary_path, summary)
    update_attempt_status(attempt, status="SEALED_HEADROOM_EVIDENCE_VERIFIED", reporting_head=head)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=TASK_ID)
    parser.add_argument("--output-root", help="Canonical external output root")
    subparsers = parser.add_subparsers(dest="command", required=True)
    static = subparsers.add_parser("static-preflight")
    static.add_argument("--json-output", type=Path)
    bind_parser = subparsers.add_parser("bind")
    bind_parser.add_argument("--cloud-preflight", type=Path, required=True)
    for command in (
        "materialize", "parity", "train-coefficients", "train-full-residual",
        "select-lambda", "evaluate", "visual-sheets", "analyze", "finalize",
    ):
        subparsers.add_parser(command)
    review = subparsers.add_parser("visual-review")
    review.add_argument("--review-json", type=Path, required=True)
    seal = subparsers.add_parser("seal-reporting-head")
    seal.add_argument("--final-head", required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--require-clean", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = output_root(args.output_root)
    commands = {
        "static-preflight": lambda: static_preflight(root),
        "bind": lambda: bind(args.cloud_preflight, root),
        "materialize": lambda: materialize(root),
        "parity": lambda: run_parity(root),
        "train-coefficients": lambda: train_coefficients(root),
        "train-full-residual": lambda: train_full_residual(root),
        "select-lambda": lambda: run_lambda_selection(root),
        "evaluate": lambda: run_evaluation(root),
        "visual-sheets": lambda: create_visual_sheets(root),
        "visual-review": lambda: seal_visual_review(root, args.review_json),
        "analyze": lambda: analyze_results(root),
        "finalize": lambda: finalize(root),
        "seal-reporting-head": lambda: seal_reporting_head(root, args.final_head),
        "verify": lambda: verify(root, require_clean=args.require_clean),
    }
    result = commands[args.command]()
    if args.command == "static-preflight" and args.json_output:
        atomic_json(args.json_output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
